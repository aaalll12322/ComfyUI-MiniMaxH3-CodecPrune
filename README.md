# ComfyUI-MiniMaxH3-CodecPrune

Reference-token pruning for MiniMax-H3 conditioning in ComfyUI, driven by an H.264 codec.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[中文说明](README.zh-CN.md)

## Links

- Paper: [arXiv:2611.31184](https://arxiv.org/abs/2611.31184)
- Original project: [Kiteretsu77/CodecOPSD](https://github.com/Kiteretsu77/CodecOPSD)
- Project page: [kiteretsu77.github.io/CodecOPSD_website](https://kiteretsu77.github.io/CodecOPSD_website/)
- Official LoRA (Ref2VA): [HikariDawn/CodecOPSD-H3-Ref2VA](https://huggingface.co/HikariDawn/CodecOPSD-H3-Ref2VA)

## Overview

MiniMax-H3 appends every reference picture and video to its input token sequence, so the
conditioning cost grows with each reference. A large share of those tokens repeats
information that earlier frames already carry.

The node re-encodes each reference with H.264 and drops the tokens the encoder predicts for
free: static backgrounds, flat regions, content unchanged since an earlier frame. Kept
tokens stay at their original rotary positions, so the packed sequence gets shorter while
everything else stays put.

This is an integration, not a reimplementation. The method, the thresholds and the
decision code belong to [CodecOPSD](https://github.com/Kiteretsu77/CodecOPSD)
(arXiv:2611.31184) and are vendored unchanged. What this project adds is the ComfyUI side:

- a Windows H.264 reader, since the upstream one ships prebuilt for Linux only;
- a conditioning node and a runtime patch that shorten the packed sequence without
  touching ComfyUI's sources;
- a converter for the official LoRA, whose diffusers naming does not match ComfyUI's fused
  attention.

ComfyUI's own source files are not modified.

## Performance

The gain equals the share of the sequence occupied by reference tokens, multiplied by the
pruning rate. With few reference tokens the effect approaches zero.

| Reference | Sequence | Wall clock |
|---|---|---|
| One picture (1,032 → 407 tokens) | 19,902 → 19,277 (−3.1%) | 30 → 30 s/it, below resolution |
| One 5 s video (37,296 → 11,189 tokens) | 56,166 → 30,059 (−46.5%) | 85 → 47 s/it (1.81×) |

Both rows use a 0.4–0.5 MP, 124-frame output. The second row was measured with SolAttn
sparse attention enabled in both runs. The first row's 625-token reduction is real, but too
small to register above run-to-run variation, which is why it is listed as no measurable
change rather than as a gain.

Reference videos dominate the conditioning sequence; a single picture does not. The
extension targets multi-reference and video-reference workflows.

Output cannot be compared pixel by pixel against an unpruned run: a different sequence
length changes the sampling trajectory, so the same seed produces a different video
(measured PSNR between the two versions: 17.5 dB). Visual inspection is the relevant
check.

## Installation

**Platform.** The reader comes from upstream, which publishes Linux wheels (x86_64 and
aarch64) only. On Linux, install that wheel and it is used in place of the bundled binary:

```bash
pip install https://github.com/Kiteretsu77/CodecOPSD/releases/download/v0.1.0/codecopsd_codec-0.1.0-cp310-abi3-linux_x86_64.whl
```

On Windows the bundled `bin/codec_reader.exe` is used instead. Other platforms build it
from `build/`; that path is untested and the script needs its Windows-only `-lbcrypt` link
flag removed.

Masks also depend on the encoder build: on Windows the imageio-ffmpeg binary and the one
validated here are the same file, on Linux and macOS they are a different FFmpeg build, so
pruning decisions may differ slightly.

1. Place this directory in `ComfyUI/custom_nodes/`.
2. Provide the reader: the upstream wheel on Linux, the bundled
   `bin/codec_reader.exe` on Windows (see [Building the reader](#building-the-reader)
   otherwise).
3. Restart ComfyUI.

`bin/ffmpeg.exe` is **optional**: it pins the encoder that produces the masks. Without it,
the imageio-ffmpeg binary already present in the ComfyUI environment is used. To pin it on
Windows, extract `binaries/ffmpeg-win-x86_64-v7.1.exe` from imageio-ffmpeg's `win_amd64`
wheel (`pip download imageio-ffmpeg==0.7.0 --platform win_amd64 --only-binary=:all:`, then
unzip) and save it as `bin/ffmpeg.exe`.

The node is registered as **MiniMax H3 Reference to Video (Codec Prune)** under
`model/conditioning/minimax`. A quick check after installing: run the same seed and inputs
with `prune` on and off; the console prints a `[codecprune] ... kept N%` line in the first
case only, and with `prune` off the node behaves like the stock one.

`CODEC_PRUNE=0` disables the runtime patch entirely and restores stock behaviour. On
Windows run `set CODEC_PRUNE=0` before launching ComfyUI, or set it as a system environment
variable; on Linux and macOS prefix the launch command.

## Usage

The node accepts the same inputs as the stock `MiniMax H3 Reference to Video` node, plus
two settings:

| Setting | Default | Description |
|---|---|---|
| `prune` | `true` | Disabled reproduces the stock node exactly, tensor for tensor |
| `drop` | `0.70` | Share of a reference **video**'s tokens to drop; `0` uses the threshold decision alone. Pictures always use the threshold decision |

Prompt tags remain `<Picture i>`, `<Video k>` and `<Audio j>`.

## Companion LoRA

The official CodecOPSD LoRA
([HikariDawn/CodecOPSD-H3-Ref2VA](https://huggingface.co/HikariDawn/CodecOPSD-H3-Ref2VA))
uses diffusers naming and cannot be loaded as-is: H3 attention in ComfyUI fuses
`to_q`/`to_k`/`to_v` into a single `qkv_proj`, while the three projections carry
independent LoRA updates, each with its own A matrix. Concatenating only the B matrices
would pair each of them with the wrong A, so the merge has to be block-diagonal.

Convert it once:

```bash
python tools/convert_codecopsd_lora.py codecopsd_minimax_h3_ref2va.safetensors \
       codecopsd_h3_ref2va_comfy.safetensors --fp16
```

The converter performs three steps:

- renames `to_out.0` → `out_proj`, `ff.net.0.proj` → `mlp.fc1`, `ff.net.2` → `mlp.fc2`;
- merges `to_q`/`to_k`/`to_v` into one `qkv_proj` adapter in exact block-diagonal form,
  `delta = blockdiag(B_q,B_k,B_v) @ cat(A_q,A_k,A_v)`, taking rank 128 to 384;
- adds the per-module `alpha` ComfyUI expects. The applied scale is `alpha/rank`, so alpha
  follows the rank from 128 to 384 and the trained scale of 1.0 is preserved.

Load the converted file with the standard **Load LoRA** node at strength 1.0. The LoRA was
distilled on pruned sequences and is intended for use with `prune` enabled.

The converter needs `torch` and `safetensors`, both already present in a ComfyUI
environment.

## How it works

`comfy/ldm/minimax/model.py` is left untouched. At import time the extension replaces two
objects on their module:

- `comfy.ldm.minimax.model.PackedLayout` — a subclass that honours an optional `keep`
  tensor on each reference block;
- `MiniMaxH3Model._cond_video_rows` — a wrapper that filters the condition rows the same
  way.

`comfy.model_base` resolves `PackedLayout` through the module attribute, so replacing the
class covers both call sites. The `keep` mask travels inside each `minimax_refs` block;
`extra_conds` reads only `latent` and `audio_latent` from that block and passes the
remaining fields through unchanged.

If a future ComfyUI release changes these internals, the patch reports the mismatch, logs
a warning, and falls back to stock behaviour.

## Limitations

- The reference **picture** mask drops 3–6% more tokens than the reference implementation
  (877 vs 902, and 3,807 vs 4,051, on the upstream example). Nine interpolation and
  quantisation combinations were tested without closing the gap, so the difference
  originates in a resize detail of the reference pipeline rather than in the algorithm.
  Reference **video** masks match exactly (3,688 and 11,189 of 37,296 on the same example).
- The runtime patch depends on ComfyUI internals and is tied to the current H3
  implementation.
- Pruning adds one H.264 encode and decode per reference on the CPU, roughly one second
  for a 2048-short-edge picture.

## Building the reader

`bin/codec_reader.exe` is the only compiled component. It decodes the H.264 stream and
returns per-macroblock motion, residual and skip signals, linking a patched FFmpeg 5.1
statically.

```bash
# MSYS2: pacman -S --needed mingw-w64-x86_64-gcc make patch diffutils
sh build/build_reader.sh
```

Two requirements are not obvious and are documented in the script:

- `--disable-asm` is mandatory. The patch modifies the C IDCT only; with assembly enabled
  the SIMD paths are used and the patch has no effect.
- `--enable-decoder=h263` appears unrelated but is required. Motion-vector export is
  guarded by `CONFIG_MPEGVIDEODEC`, which the h263 decoder selects.

The script targets MSYS2 on Windows. On Linux and macOS remove `-lbcrypt` from the final
link line; it is a Windows-only library and the script has no branch for other platforms.

## Test setup

Single Tesla V100-SXM2-16GB (sm_70), ComfyUI 0.46.0 on Windows. Base model
`minimax_h3_hybrid_fl2va_ref2va_b25-49-int8`, accelerated with
`dmad_minimax_h3_4step_full_critic_comfyui` at 4 steps, euler + beta.

The scope is narrow: two workflows (one picture reference, one 5 s video reference), one
GPU, one model configuration. Multi-reference combinations, other resolutions, other
hardware and long-run stability were not measured, and the absolute timings do not
transfer to other cards. The sequence-length ratios should hold, since they depend on the
token counts rather than on the hardware.

## License

Project code is MIT (see [LICENSE](LICENSE)). Bundled components carry their own licenses,
listed below; the MIT terms do not apply to them:

- `codecopsd_vendor/` is the pruning decision code from CodecOPSD, © 2026 The CodecOPSD
  Authors, free for research use; commercial use requires permission from its authors.
- `bin/codec_reader.exe` and `build/ffmpeg-5.1.patch` are FFmpeg 5.1 under the GNU LGPL
  2.1 or later, with the patch redistributed as that license requires. The patch, the
  license text, the source notice and a build script are included in `build/`.
- `bin/ffmpeg.exe` is optional and not committed (87 MB). When present it pins the encoder
  to the build the masks were validated against; otherwise the imageio-ffmpeg binary from
  the ComfyUI environment is used.

## Citation

```bibtex
@article{wang2026codecopsd,
  title   = {On-Policy Codec Distillation for Efficient All-in-One Video Generation},
  author  = {Wang, Boyang and Blackburn-Matzen, Kevin and Saha, Oindrila and Gadelha, Matheus and Cheng, Zezhou},
  journal = {arXiv preprint arXiv:2611.31184},
  year    = {2026}
}
```
