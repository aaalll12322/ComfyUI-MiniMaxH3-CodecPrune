"""MiniMax-H3 reference-to-video conditioning with codec pruning.

Same inputs as ComfyUI's MiniMaxH3ReferenceToVideo, plus `prune` / `drop`. The only
difference in behaviour: each reference block carries a `"keep"` mask, and the runtime
patch (patch.py) leaves the dropped reference token rows out of the packed sequence while
keeping every kept token at its original rotary position.
"""
import logging

import node_helpers
from comfy_api.latest import io

from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    _empty_av_latent,
    _encode_ref_audio,
    _resize,
    adapt_canvas,
)

log = logging.getLogger("codecprune")


def _keep_picture(img, prune):
    """comfy IMAGE [1,H,W,3] in 0..1 -> bool [1,h/32,w/32] or None."""
    if not prune:
        return None
    from codecopsd_vendor.codec_gazer import keep_mask
    x = img[:1].permute(0, 3, 1, 2).contiguous().float() * 2.0 - 1.0   # -> [1,3,H,W] in [-1,1]
    return keep_mask(x[0], kind="image").cpu()


def _keep_video(frames, prune, drop):
    """comfy IMAGE [T,H,W,3] in 0..1 -> bool [T',h/32,w/32] or None."""
    if not prune:
        return None
    from codecopsd_vendor.codec_gazer import MINIMAX_H3, keep_mask
    x = frames.permute(0, 3, 1, 2).contiguous().float() * 2.0 - 1.0   # -> [T,3,H,W] in [-1,1]
    return keep_mask(x, kind="video", grid=MINIMAX_H3, drop=drop).cpu()


class MiniMaxH3RefToVideoCodecPrune(io.ComfyNode):
    """ref2va conditioning whose reference tokens are pruned by an H.264 codec."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RefToVideoCodecPrune",
            display_name="MiniMax H3 Reference to Video (Codec Prune)",
            category="model/conditioning/minimax",
            description="Reference conditioning for MiniMax H3 that drops the reference tokens "
                        "an H.264 codec can predict for free (static / redundant regions), so the "
                        "condition sequence is shorter. Keep tags as <Picture i> / <Video k> / <Audio j>.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE. Without it reference images/videos only condition the text encoder."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Audio VAE. Without it reference audio only conditions the text encoder."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="Reference image sizing, as in the stock node."),
                io.Boolean.Input("prune", default=True,
                                 tooltip="Drop reference tokens the codec can predict for free. Off = exact stock behaviour."),
                io.Float.Input("drop", default=0.70, min=0.0, max=0.95, step=0.05,
                               tooltip="Share of a reference video's tokens to drop (0 = pure threshold decision). "
                                       "Pictures always use the threshold decision."),
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Audio.Input("ref_video_audio"), prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, prompt, width, height, length, ref_image_size="match", prune=True, drop=0.70,
                vae=None, audio_vae=None, ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []
        ref_blocks = []
        stats = []

        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, ((width * height) / (w * h)) ** 0.5)
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            if vae is not None:
                z = vae.encode(resized)
                blk = {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z}
                keep = _keep_picture(resized, prune)
                if keep is not None:
                    # a picture has exactly one token frame
                    keep = _crop_to_grid(keep, 1, blk["latent_h"] // 2, blk["latent_w"] // 2)
                    blk["keep"] = keep
                    stats.append(("picture %dx%d" % (tw, th), float(keep.float().mean())))
                ref_blocks.append(blk)

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            if soundtrack is not None:
                ref_items.append({"type": "audio"})
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            ref_items.append({"type": "video", "data": frames[sample_idx],
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            if vae is None:
                continue
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None and audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            blk = {"kind": "video_audio" if ref_audio_t else "video",
                   "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                   "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent}
            keep = _keep_video(frames, prune, drop)
            if keep is not None:
                keep = _crop_to_grid(keep, blk["latent_t"], blk["latent_h"] // 2, blk["latent_w"] // 2)
                blk["keep"] = keep
                stats.append(("video %dx%d/%df" % (cw, ch, n), float(keep.float().mean())))
            ref_blocks.append(blk)

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            ref_items.append({"type": "audio"})
            if audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
                ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        if stats:
            for label, kept in stats:
                log.info("[codecprune] %s: kept %.1f%% of reference tokens", label, 100 * kept)

        return io.NodeOutput(cond, latent)


def _crop_to_grid(keep, t, h, w):
    """Codec masks are sized to ceil(px/32); the packed layout uses floor(latent/2)."""
    if keep.shape[0] < t or keep.shape[1] < h or keep.shape[2] < w:
        raise ValueError("codec mask %s is smaller than the layout grid (%d,%d,%d); the "
                         "reference size is not 32-aligned" % (tuple(keep.shape), t, h, w))
    return keep[:t, :h, :w].contiguous()
