#!/bin/sh
# Build bin/codec_reader.exe: the H.264 per-macroblock reader of codec pruning.
#
# Needs MSYS2 (mingw-w64) on Windows, or any gcc toolchain on Linux/macOS.
#   pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-make make patch diffutils
#
# The FFmpeg below is patched: its IDCT writes 128 + residual instead of
# prediction + residual, so "decoded luma - 128" IS the residual. The patch comes from
# the CodecOPSD release wheel (LGPL 2.1+ requires it to be shipped there); see
# FFMPEG-NOTICE.txt for the exact source and configure line.
set -e

FFMPEG_VER=5.1
FFMPEG_SHA256=32b56fb01ce90d452958ae25e91c9564abf49ed5453c127bec23c63e530aa8fa
FFMPEG_URL=https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.bz2

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
SRC="$HERE/ffmpeg-$FFMPEG_VER"
OUT="$ROOT/bin"

if [ ! -d "$SRC" ]; then
    echo "== fetching FFmpeg $FFMPEG_VER"
    curl -L -o "$HERE/ffmpeg-$FFMPEG_VER.tar.bz2" "$FFMPEG_URL"
    echo "$FFMPEG_SHA256  $HERE/ffmpeg-$FFMPEG_VER.tar.bz2" | sha256sum -c -
    tar -xf "$HERE/ffmpeg-$FFMPEG_VER.tar.bz2" -C "$HERE"
fi

echo "== applying the residual patch"
cd "$SRC"
patch -p1 < "$HERE/ffmpeg-5.1.patch"

echo "== configure (decode only, no asm on purpose)"
# --disable-asm is REQUIRED: the patch only changes the C IDCT, and with asm enabled the
# SIMD versions are used instead, which would silently produce real pixels, not residuals.
# h263 is not used by the reader, but enabling it is what pulls in mpegvideodec, and the
# motion-vector export path lives behind CONFIG_MPEGVIDEODEC. Dropping it loses all MVs.
./configure \
    --enable-pic --disable-asm --disable-autodetect --disable-everything \
    --disable-programs --disable-doc --disable-avdevice --disable-avfilter \
    --disable-swscale --disable-swresample --disable-network \
    --enable-decoder=h264 --enable-decoder=h263 \
    --enable-demuxer=mov --enable-parser=h264 --enable-protocol=file \
    --prefix="$HERE/prefix"

make -j"$(nproc)"
make install

echo "== building the reader"
cd "$ROOT"
gcc -O2 -o "$OUT/codec_reader.exe" "$HERE/codec_reader.c" \
    -I"$HERE/prefix/include" -L"$HERE/prefix/lib" \
    -lavformat -lavcodec -lavutil -lm -lbcrypt -static -Wl,--gc-sections

echo "== done: $OUT/codec_reader.exe"
"$OUT/codec_reader.exe" || true
