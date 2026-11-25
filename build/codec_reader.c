/*
 * codec_reader.c -- H.264 per-macroblock signal reader for CodecOPSD codec pruning.
 *
 * Port of the reader that ships prebuilt (Linux only) in CodecOPSD's release wheel.
 * The algorithm is from:
 *   "On-Policy Codec Distillation for Efficient All-in-One Video Generation" (CodecOPSD),
 *   Wang, Blackburn-Matzen, Saha, Gadelha, Cheng. arXiv:2611.31184
 * and the FFmpeg patch it needs is theirs (LGPL 2.1+), redistributed with that wheel.
 *
 * Requires an FFmpeg 5.1 built with that patch and with --disable-asm
 * (the patch only changes the C IDCT; with asm enabled SIMD versions are used instead).
 *
 * Contract (mirrors codecopsd.codec_gazer._signals):
 *   out.bin = motion[t][mbh][mbw] int32  | (mx^2 + my^2) / motion_scale^2, rounded -- i.e. squared
 *            res[t][mbh][mbw]    int32  | sum of |luma - 128| over the 16x16 macroblock
 *            skip[t][mbh][mbw]   uint8  | 1 if the macroblock was coded as skipped
 *   The patch makes the decoder write 128 + residual, so residual = luma - 128.
 *   A pixel value of 128 therefore means "no residual", which is why |luma - 128| is the residual.
 *
 * usage: codec_reader <in.mp4> <frames> <width> <height> <threads> <out.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include <libavutil/opt.h>

#define MB 16

static void die(const char *msg, int err)
{
    char buf[256];
    if (err)
        av_strerror(err, buf, sizeof(buf));
    fprintf(stderr, "codec_reader: %s%s%s\n", msg, err ? ": " : "", err ? buf : "");
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "usage: %s <in.mp4> <frames> <width> <height> <threads> <out.bin>\n", argv[0]);
        return 2;
    }
    const char *in_path = argv[1];
    const int want_frames = atoi(argv[2]);
    const int width  = atoi(argv[3]);
    const int height = atoi(argv[4]);
    const int threads = atoi(argv[5]);
    const char *out_path = argv[6];

    if (want_frames <= 0 || width <= 0 || height <= 0)
        die("bad frame/size arguments", 0);

    const int mbw = (width  + MB - 1) / MB;
    const int mbh = (height + MB - 1) / MB;
    const size_t n_mb = (size_t)mbw * (size_t)mbh;

    int32_t *motion = calloc(n_mb * (size_t)want_frames, sizeof(int32_t));
    int32_t *res    = calloc(n_mb * (size_t)want_frames, sizeof(int32_t));
    uint8_t *skip   = calloc(n_mb * (size_t)want_frames, sizeof(uint8_t));
    if (!motion || !res || !skip)
        die("out of memory", 0);

    AVFormatContext *fmt = NULL;
    int err = avformat_open_input(&fmt, in_path, NULL, NULL);
    if (err < 0) die("cannot open input", err);
    err = avformat_find_stream_info(fmt, NULL);
    if (err < 0) die("cannot read stream info", err);

    const AVCodec *dec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!dec) die("no h264 decoder in this build", 0);

    int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vstream < 0) die("no video stream", vstream);

    AVCodecContext *cctx = avcodec_alloc_context3(dec);
    if (!cctx) die("cannot allocate codec context", 0);
    err = avcodec_parameters_to_context(cctx, fmt->streams[vstream]->codecpar);
    if (err < 0) die("cannot copy codec parameters", err);
    cctx->thread_count = threads > 0 ? threads : 1;
    cctx->thread_type  = FF_THREAD_FRAME | FF_THREAD_SLICE;
    /* the motion vectors only reach the caller if the decoder is asked for them */
    cctx->export_side_data |= AV_CODEC_EXPORT_DATA_MVS;
    err = avcodec_open2(cctx, dec, NULL);
    if (err < 0) die("cannot open decoder", err);

    if (cctx->width != width || cctx->height != height)
        fprintf(stderr, "codec_reader: note: stream is %dx%d, expected %dx%d\n",
                cctx->width, cctx->height, width, height);

    AVPacket *pkt = av_packet_alloc();
    AVFrame  *frm = av_frame_alloc();
    if (!pkt || !frm) die("cannot allocate packet/frame", 0);

    int fi = 0;
    while (fi < want_frames) {
        err = av_read_frame(fmt, pkt);
        if (err < 0) break;                        /* EOF or error: stop */
        if (pkt->stream_index != vstream) { av_packet_unref(pkt); continue; }

        err = avcodec_send_packet(cctx, pkt);
        av_packet_unref(pkt);
        if (err < 0) continue;

        while (fi < want_frames) {
            err = avcodec_receive_frame(cctx, frm);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) break;
            if (err < 0) die("decode error", err);

            if (frm->format != AV_PIX_FMT_YUV420P && frm->format != AV_PIX_FMT_YUV444P &&
                frm->format != AV_PIX_FMT_YUV422P && frm->format != AV_PIX_FMT_GRAY8)
                die("unsupported pixel format (expected 8-bit planar luma)", 0);

            /* --- residual: |luma - 128| summed per 16x16 macroblock --- */
            {
                const uint8_t *Y = frm->data[0];
                const int ls = frm->linesize[0];
                int32_t *dst = res + (size_t)fi * n_mb;
                for (int my = 0; my < mbh; my++) {
                    for (int mx = 0; mx < mbw; mx++) {
                        const int y0 = my * MB, x0 = mx * MB;
                        const int y1 = y0 + MB < frm->height ? y0 + MB : frm->height;
                        const int x1 = x0 + MB < frm->width  ? x0 + MB : frm->width;
                        int32_t sum = 0;
                        for (int y = y0; y < y1; y++) {
                            const uint8_t *row = Y + (size_t)y * ls;
                            for (int x = x0; x < x1; x++) {
                                int d = (int)row[x] - 128;
                                sum += d < 0 ? -d : d;
                            }
                        }
                        dst[(size_t)my * mbw + mx] = sum;
                    }
                }
            }

            /* --- motion + skip from the exported side data --- */
            {
                AVFrameSideData *sd = av_frame_get_side_data(frm, AV_FRAME_DATA_MOTION_VECTORS);
                if (getenv("CODEC_READER_DEBUG"))
                    fprintf(stderr, "debug: frame %d side_data_count=%d mvs=%s export=%d\n",
                            fi, frm->nb_side_data, sd ? "yes" : "no", (int)cctx->export_side_data);
                if (sd) {
                    const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
                    const int nb = (int)(sd->size / sizeof(AVMotionVector));
                    int32_t *mo = motion + (size_t)fi * n_mb;
                    if (getenv("CODEC_READER_DEBUG")) {
                        int nz = 0;
                        for (int i = 0; i < nb; i++)
                            if (mvs[i].motion_x || mvs[i].motion_y) nz++;
                        fprintf(stderr, "debug: frame %d nb=%d nonzero=%d scale0=%d mv0=(%d,%d) flags0=%u\n",
                                fi, nb, nz, nb ? mvs[0].motion_scale : -1,
                                nb ? (int)mvs[0].motion_x : 0, nb ? (int)mvs[0].motion_y : 0,
                                nb ? (unsigned)mvs[0].flags : 0);
                    }
                    uint8_t *sk = skip   + (size_t)fi * n_mb;
                    for (int i = 0; i < nb; i++) {
                        const AVMotionVector *v = &mvs[i];
                        int mx = v->dst_x / MB, my = v->dst_y / MB;
                        if (mx < 0 || mx >= mbw || my < 0 || my >= mbh) continue;
                        const int64_t scale = v->motion_scale ? (int64_t)v->motion_scale : 1;
                        /* squared vector length in pixels, rounded */
                        const int64_t px2 = ((int64_t)v->motion_x * v->motion_x +
                                             (int64_t)v->motion_y * v->motion_y) / (scale * scale);
                        const size_t k = (size_t)my * mbw + mx;
                        if (px2 > mo[k]) mo[k] = (int32_t)px2;
                        if (v->flags & 1) sk[k] = 1;
                    }
                }
            }

            fi++;
            av_frame_unref(frm);
        }
    }

    /* flush: the tail frames may still sit in the decoder */
    if (fi < want_frames) {
        avcodec_send_packet(cctx, NULL);
        while (fi < want_frames) {
            err = avcodec_receive_frame(cctx, frm);
            if (err < 0) break;
            /* the residual/motion extraction is duplicated from above on purpose:
               keeping the flush path identical avoids an off-by-one in the tail */
            {
                const uint8_t *Y = frm->data[0];
                const int ls = frm->linesize[0];
                int32_t *dst = res + (size_t)fi * n_mb;
                for (int my = 0; my < mbh; my++)
                    for (int mx = 0; mx < mbw; mx++) {
                        const int y0 = my * MB, x0 = mx * MB;
                        const int y1 = y0 + MB < frm->height ? y0 + MB : frm->height;
                        const int x1 = x0 + MB < frm->width  ? x0 + MB : frm->width;
                        int32_t sum = 0;
                        for (int y = y0; y < y1; y++) {
                            const uint8_t *row = Y + (size_t)y * ls;
                            for (int x = x0; x < x1; x++) {
                                int d = (int)row[x] - 128;
                                sum += d < 0 ? -d : d;
                            }
                        }
                        dst[(size_t)my * mbw + mx] = sum;
                    }
            }
            {
                AVFrameSideData *sd = av_frame_get_side_data(frm, AV_FRAME_DATA_MOTION_VECTORS);
                if (sd) {
                    const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
                    const int nb = (int)(sd->size / sizeof(AVMotionVector));
                    int32_t *mo = motion + (size_t)fi * n_mb;
                    uint8_t *sk = skip   + (size_t)fi * n_mb;
                    for (int i = 0; i < nb; i++) {
                        const AVMotionVector *v = &mvs[i];
                        int mx = v->dst_x / MB, my = v->dst_y / MB;
                        if (mx < 0 || mx >= mbw || my < 0 || my >= mbh) continue;
                        const int64_t scale = v->motion_scale ? (int64_t)v->motion_scale : 1;
                        const int64_t px2 = ((int64_t)v->motion_x * v->motion_x +
                                             (int64_t)v->motion_y * v->motion_y) / (scale * scale);
                        const size_t k = (size_t)my * mbw + mx;
                        if (px2 > mo[k]) mo[k] = (int32_t)px2;
                        if (v->flags & 1) sk[k] = 1;
                    }
                }
            }
            fi++;
            av_frame_unref(frm);
        }
    }

    if (fi < want_frames)
        fprintf(stderr, "codec_reader: warning: decoded %d of %d frames\n", fi, want_frames);

    FILE *fp = fopen(out_path, "wb");
    if (!fp) die("cannot open output", 0);
    const size_t total = n_mb * (size_t)want_frames;
    fwrite(motion, sizeof(int32_t), total, fp);
    fwrite(res,    sizeof(int32_t), total, fp);
    fwrite(skip,   sizeof(uint8_t), total, fp);
    fclose(fp);

    free(motion); free(res); free(skip);
    av_frame_free(&frm); av_packet_free(&pkt);
    avcodec_free_context(&cctx);
    avformat_close_input(&fmt);
    return 0;
}
