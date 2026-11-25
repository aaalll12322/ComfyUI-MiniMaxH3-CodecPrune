"""ComfyUI-MiniMaxH3-CodecPrune

Codec pruning for MiniMax-H3 reference conditioning (ref2va / image references).

Drops the reference tokens an H.264 codec can predict for free -- static backgrounds,
flat regions, content repeated from earlier frames -- and keeps every kept token at its
original rotary position, so the packed condition sequence gets shorter without moving
anything else. Method and decision logic: CodecOPSD (arXiv:2611.31184).

ComfyUI's own files are not modified: the packed-layout class and the condition-row
builder are replaced on their module object at import time (see codecprune_patch.py).
Set CODEC_PRUNE=0 to skip the patch entirely.
"""
import logging
import os
import sys

log = logging.getLogger("codecprune")

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

# The upstream decision code imports its prebuilt reader as `codecopsd_codec`. That package
# ships wheels for Linux only, so the lookup order is:
#   - an installed codecopsd-codec (Linux) is used as-is; it is the reference reader;
#   - otherwise the bundled codec_reader.exe stands in, with the same contract.
try:
    import codecopsd_codec as _reader                   # upstream wheel present
    log.info("[codecprune] reader: upstream codecopsd_codec")
except ImportError:
    try:
        import codecprune_reader as _reader
        sys.modules.setdefault("codecopsd_codec", _reader)
        log.info("[codecprune] reader: bundled codec_reader.exe")
    except Exception as e:                               # noqa: BLE001 - never break ComfyUI startup
        log.warning("[codecprune] reader unavailable: %s", e)
        _reader = None

# Pin the encoder so the masks do not depend on the user's imageio-ffmpeg version:
# use bin/ffmpeg.exe when this install ships one, else fall back to imageio's binary.
try:
    from codecopsd_vendor.codec_gazer import codec as _codec
    _ff = os.path.join(_PLUGIN_DIR, "bin", "ffmpeg.exe")
    if os.path.isfile(_ff):
        _codec.FFMPEG_EXE = _ff
        log.info("[codecprune] encoder: %s", _ff)
    else:
        log.info("[codecprune] encoder: imageio-ffmpeg (no bin/ffmpeg.exe shipped)")
except Exception as e:                                   # noqa: BLE001
    log.warning("[codecprune] could not bind the encoder: %s", e)

_PATCHED = False
if os.environ.get("CODEC_PRUNE", "1") not in ("0", "false", "False", ""):
    try:
        import codecprune_patch as _patch
        _PATCHED = _patch.install()
    except Exception as e:                               # noqa: BLE001
        log.warning("[codecprune] patch failed, falling back to stock behaviour: %s", e)

from comfy_api.latest import ComfyExtension, io  # noqa: E402
from codecprune_nodes import MiniMaxH3RefToVideoCodecPrune  # noqa: E402


class CodecPruneExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MiniMaxH3RefToVideoCodecPrune]


async def comfy_entrypoint() -> CodecPruneExtension:
    return CodecPruneExtension()


__all__ = ["comfy_entrypoint", "MiniMaxH3RefToVideoCodecPrune"]
