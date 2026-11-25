"""Runtime patching for codec pruning of MiniMax-H3 reference tokens.

ComfyUI's source files are never modified. `comfy.model_base` builds its packed layout
through the module attribute `comfy.ldm.minimax.model.PackedLayout`, so replacing that
attribute on the module is enough for both call sites (extra_conds and the model's own
_forward) to pick up the pruned class.

A reference block carries an optional `"keep"` tensor (bool, one entry per dense reference
token row). `extra_conds` copies `refs` into the payload verbatim and only reads `latent` /
`audio_latent` from it, so the extra key survives all the way to the layout.

Row order (established by the comfy layout code, verified against it):
  picture : (row, col) over the 2x2-patch grid of one latent frame
  video   : (token frame, row, col)  -- the same order `_video_grid` reshapes into
which is exactly torch.flatten order of the (token frame, h/32, w/32) keep mask.
"""
import logging

import torch

log = logging.getLogger(__name__)

_PATCHED = False


def _mod():
    import comfy.ldm.minimax.model as mm
    return mm


def _require(mm, *names):
    missing = [n for n in names if not hasattr(mm, n)]
    if missing:
        raise RuntimeError(
            "ComfyUI's comfy/ldm/minimax/model.py does not expose %s -- this ComfyUI build "
            "is not compatible with the codec-prune patch (expected the H3 packed-layout "
            "implementation)." % ", ".join(missing))
    return mm


def keep_flat(keep, n_full, label="reference"):
    """Validate a keep mask against the dense row count; None means "keep everything"."""
    if keep is None:
        return None
    if torch.is_tensor(keep):
        k = keep.reshape(-1).to(torch.bool).cpu()
    else:
        k = torch.as_tensor(keep, dtype=torch.bool).reshape(-1).cpu()
    if k.numel() != n_full:
        raise ValueError("%s: keep mask has %d entries, the dense layout has %d rows"
                         % (label, k.numel(), n_full))
    if bool(k.all()):
        return None
    if not bool(k.any()):
        k = k.clone()
        k[0] = True          # never drop every token of a reference
    return k


def build_pruned_layout_class(mm):
    _require(mm, "_frame_grid", "_video_grid", "_ref_t_span", "_video_t_spans", "_audio_grid")
    _frame_grid, _video_grid = mm._frame_grid, mm._video_grid
    _ref_t_span, _video_t_spans, _audio_grid = mm._ref_t_span, mm._video_t_spans, mm._audio_grid

    class PrunedPackedLayout(mm.PackedLayout):
        """`PackedLayout` that honours `blk["keep"]` on reference blocks."""

        def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
            frame, w_grid = _frame_grid(latent_h, latent_w)
            frame_rows = frame.shape[0]

            segments = [("text", text_len)]
            g = torch.zeros(text_len, 3, dtype=torch.float64)
            g[:, 0] = torch.arange(text_len, dtype=torch.float64)
            pos = [g]

            img_pos, img_update = [], []
            audio_pos = [torch.zeros(0, dtype=torch.long)]
            audio_update = [torch.zeros(0, dtype=torch.bool)]
            row = text_len

            target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
            cursor = float(text_len)
            for blk in refs or ():
                cursor += _ref_t_span(blk)

            if keyframes:
                for kf in keyframes:
                    cond_t = cursor + mm.FRAME_RESCALE * kf["resolved_frame_index"]
                    video_latent = kf.get("latent")
                    if video_latent is not None:
                        vt = video_latent.shape[2]
                        n = vt * frame_rows
                        segments.append(("cond", n))
                        pos.append(mm._video_grid(vt, frame, cond_t))
                        img_pos.append(torch.arange(row, row + n))
                        img_update.append(torch.zeros(n, dtype=torch.bool))
                        row += n
                    audio_latent = kf.get("audio_latent")
                    if audio_latent is not None:
                        rt = audio_latent.shape[-1]
                        segments.append(("cond_audio", rt * 2))
                        pos.append(_audio_grid(cond_t, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2

            if refs:
                cursor = float(text_len)
                for blk in refs:
                    kind = blk["kind"]
                    if kind == "image":
                        r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                        n_full = r_frame.shape[0]
                        gr = torch.empty(n_full, 3, dtype=torch.float64)
                        gr[:, 0] = cursor
                        gr[:, 1:] = r_frame
                        k = keep_flat(blk.get("keep"), n_full, "reference picture")
                        if k is not None:
                            gr = gr[k]
                            n = int(k.sum())
                        else:
                            n = n_full
                        segments.append(("ref_img", n))
                        pos.append(gr)
                        img_pos.append(torch.arange(row, row + n))
                        img_update.append(torch.zeros(n, dtype=torch.bool))
                        row += n
                        cursor += 1.0
                    elif kind == "audio":
                        rt = blk["ref_audio_t"]
                        if rt > 0:
                            segments.append(("ref_audio", rt * 2))
                            pos.append(_audio_grid(cursor, rt, *target_audio_w))
                            audio_pos.append(torch.arange(row, row + rt * 2))
                            audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                            row += rt * 2
                        cursor += float(rt)
                    elif kind in ("video", "video_audio"):
                        rt = blk["ref_audio_t"]
                        vt = blk["latent_t"]
                        r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                        if rt > 0:
                            segments.append(("ref_audio", rt * 2))
                            pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                            audio_pos.append(torch.arange(row, row + rt * 2))
                            audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                            row += rt * 2
                        n_full = vt * r_frame.shape[0]
                        pg = _video_grid(vt, r_frame, cursor)
                        k = keep_flat(blk.get("keep"), n_full, "reference video")
                        if k is not None:
                            pg = pg[k]
                            n = int(k.sum())
                        else:
                            n = n_full
                        segments.append(("ref_img", n))
                        pos.append(pg)
                        img_pos.append(torch.arange(row, row + n))
                        img_update.append(torch.zeros(n, dtype=torch.bool))
                        row += n
                        cursor += max(float(rt), sum(_video_t_spans(vt)))

            if audio_t:
                segments.append(("audio", audio_t * 2))
                pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
                audio_pos.append(torch.arange(row, row + audio_t * 2))
                audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
                row += audio_t * 2

            n_video = latent_t * frame_rows
            segments.append(("video", n_video))
            pos.append(mm._video_grid(latent_t, frame, cursor))
            img_pos.append(torch.arange(row, row + n_video))
            img_update.append(torch.ones(n_video, dtype=torch.bool))
            row += n_video

            self.seq_len = row
            self.position_ids = torch.cat(pos)
            self.img_pos = torch.cat(img_pos)
            self.img_update = torch.cat(img_update)
            self.audio_pos = torch.cat(audio_pos)
            self.audio_update = torch.cat(audio_update)
            # deliberately the same 5-tuple as upstream: extra_conds stores a fresh layout in
            # the payload for every run, and _forward only compares this to decide reuse.
            self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
            seg_abs, off = [], 0
            for kind, n in segments:
                seg_abs.append((off, off + n, kind))
                off += n
            self.segments = seg_abs

    return PrunedPackedLayout


def build_pruned_cond_rows(mm):
    _require(mm, "patchify_video")

    def _cond_video_rows(self, payload, device):
        """Same as upstream, but visual condition rows follow `blk["keep"]`."""
        rows = []
        aug = payload.get("visual_cond_noise_aug", mm.VISUAL_COND_TIMESTEP)
        seed = int(payload.get("seed", 0))
        items = [(kf["latent"], None) for kf in (payload.get("keyframes") or ()) if kf.get("latent") is not None]
        items += [(r["latent"], r.get("keep")) for r in (payload.get("refs") or ()) if "latent" in r]
        for z, keep in items:
            r = mm.patchify_video(z.to(torch.float32), self.patch_size)
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            k = keep_flat(keep, r.shape[0], "condition rows")
            if k is not None:
                r = r[k.to(r.device)]
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    return _cond_video_rows


def install():
    """Replace the layout class and the condition-row builder. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True
    mm = _mod()
    try:
        cls = build_pruned_layout_class(mm)
        fn = build_pruned_cond_rows(mm)
    except RuntimeError as e:
        log.warning("[codecprune] patch skipped: %s", e)
        return False

    original_rows = mm.MiniMaxH3Model._cond_video_rows
    mm.PackedLayout = cls
    mm.MiniMaxH3Model._cond_video_rows = fn
    _PATCHED = True
    log.info("[codecprune] patched: PackedLayout=%s, _cond_video_rows=%s",
             cls.__name__, getattr(fn, "__name__", "wrapper"))
    log.info("[codecprune] original _cond_video_rows kept at %r for rollback", original_rows)
    return True
