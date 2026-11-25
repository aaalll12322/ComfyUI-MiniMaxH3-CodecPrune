"""Windows-native stand-in for the prebuilt `codecopsd_codec` extension.

The upstream package only ships Linux wheels, so the reader is compiled here as a
standalone exe (`bin/codec_reader.exe`) that links a patched FFmpeg 5.1 statically.
This module exposes the same single function the extension does, so upstream's
`codec_gazer.codec` can stay untouched:

    read(path, t, h, w, threads) -> (motion_bytes, residual_bytes, skip_bytes)

Contract (see `build/codec_reader.c`):
    motion  int32 [t][h/16][w/16]  squared motion-vector length in pixels
    residual int32 [t][h/16][w/16] sum of |luma - 128| over the 16x16 macroblock
    skip    uint8 [t][h/16][w/16]  1 if the macroblock was skipped
"""
import os
import subprocess
import tempfile

_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "codec_reader.exe")


def read(path, t, h, w, threads):
    if not os.path.isfile(_EXE):
        raise RuntimeError(
            "codec_reader.exe not found at %s -- the patched FFmpeg reader was not built "
            "for this install (see README: Building the reader)" % _EXE)

    mbw = (w + 15) // 16
    mbh = (h + 15) // 16
    n = mbw * mbh * t

    fd, out = tempfile.mkstemp(suffix=".signals.bin")
    os.close(fd)
    try:
        p = subprocess.run(
            [_EXE, str(path), str(t), str(w), str(h), str(max(1, int(threads))), out],
            capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("codec_reader failed (%d): %s" % (p.returncode, p.stderr.strip()[:500]))
        with open(out, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(out)
        except OSError:
            pass

    if len(data) != n * 9:
        raise RuntimeError("codec_reader produced %d bytes, expected %d" % (len(data), n * 9))
    return data[:4 * n], data[4 * n:8 * n], data[8 * n:9 * n]


def install():
    """Make `import codecopsd_codec` resolve to this module."""
    import sys
    sys.modules.setdefault("codecopsd_codec", sys.modules[__name__])
