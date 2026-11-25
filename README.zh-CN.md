# ComfyUI-MiniMaxH3-CodecPrune

在 ComfyUI 中为 MiniMax-H3 的参考条件做基于 H.264 编解码器的 token 剪枝。

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.md)

## 相关链接

- 论文：[arXiv:2611.31184](https://arxiv.org/abs/2611.31184)
- 原项目：[Kiteretsu77/CodecOPSD](https://github.com/Kiteretsu77/CodecOPSD)
- 项目主页：[kiteretsu77.github.io/CodecOPSD_website](https://kiteretsu77.github.io/CodecOPSD_website/)
- 官方 LoRA（Ref2VA）：[HikariDawn/CodecOPSD-H3-Ref2VA](https://huggingface.co/HikariDawn/CodecOPSD-H3-Ref2VA)

## 概述

MiniMax-H3 把每一张参考图、每一段参考视频都追加到输入 token 序列，参考越多，条件部分的计算开销越大。而这些 token 中有相当一部分重复了前序帧已经携带的信息。

节点用 H.264 重新编码每个参考，丢弃编码器可以免费预测出的 token：静止背景、平坦区域，以及相对前面帧没有变化的内容。被保留的 token 停留在其原始旋转位置上，因此打包序列变短，其余部分不受影响。

本项目是适配层，不是重新实现。方法、阈值与判定代码均属于 [CodecOPSD](https://github.com/Kiteretsu77/CodecOPSD)（arXiv:2611.31184），以原样引入。本项目补充的是 ComfyUI 一侧：

- Windows 下的 H.264 reader——上游仅提供 Linux 预编译包；
- 一个条件节点，以及在源码之外缩短打包序列的运行时补丁；
- 官方 LoRA 的格式转换——其 diffusers 命名与 ComfyUI 融合后的注意力结构不匹配。

ComfyUI 自身源码不作修改。

## 性能

收益等于参考 token 在序列中的占比乘以剪枝率。参考 token 数量少时，收益趋近于零。

| 参考条件 | 序列长度 | 单步耗时 |
|---|---|---|
| 一张图（1,032 → 407 token） | 19,902 → 19,277（−3.1%） | 30 → 30 s/it，低于测量分辨率 |
| 一段 5 秒视频（37,296 → 11,189 token） | 56,166 → 30,059（−46.5%） | 85 → 47 s/it（1.81×） |

两行数据均基于 0.4–0.5 MP、124 帧的输出。第二行的两次运行都启用了 SolAttn 稀疏注意力。第一行减少的 625 个 token 是真实的，但幅度低于运行间的波动，因此记为“无可测变化”而非收益。

参考视频在条件序列中占主导，单张参考图则不是。本扩展面向多参考与视频参考的工作流。

输出无法与未剪枝结果逐像素比较：序列长度不同会改变采样轨迹，相同种子也会生成不同的视频（两版实测 PSNR 为 17.5 dB）。有效的检查方式是目视对比。

## 安装

**平台支持。** reader 来自上游，而上游仅发布 Linux wheel（x86_64 与 aarch64）。Linux 上安装该 wheel 即可，它会替代随附的二进制：

```bash
pip install https://github.com/Kiteretsu77/CodecOPSD/releases/download/v0.1.0/codecopsd_codec-0.1.0-cp310-abi3-linux_x86_64.whl
```

Windows 上使用随附的 `bin/codec_reader.exe`。其他平台需从 `build/` 构建，该路径未经验证，且脚本需去掉 Windows 专用的 `-lbcrypt` 链接参数。

掩码还取决于编码器构建：Windows 上 imageio-ffmpeg 的二进制与本仓库验证所用的是同一个文件，Linux 与 macOS 上则是另一套 FFmpeg 构建，剪枝判定可能略有差异。

1. 将本目录放入 `ComfyUI/custom_nodes/`。
2. 准备 reader：Linux 用上游 wheel，Windows 用随附的 `bin/codec_reader.exe`（否则见[构建 reader](#构建-reader)）。
3. 重启 ComfyUI。

`bin/ffmpeg.exe` 为**可选项**：用于固定生成掩码的编码器。缺失时改用 ComfyUI 环境中已有的 imageio-ffmpeg 可执行文件。如需固定 Windows 版本，可从 imageio-ffmpeg 的 `win_amd64` wheel 中取出 `binaries/ffmpeg-win-x86_64-v7.1.exe`（`pip download imageio-ffmpeg==0.7.0 --platform win_amd64 --only-binary=:all:`，解压后取得），另存为 `bin/ffmpeg.exe`。

节点注册为 **MiniMax H3 Reference to Video (Codec Prune)**，位于 `model/conditioning/minimax`。安装后的快速检查：用相同种子与输入，分别以 `prune` 开启和关闭各运行一次；控制台仅在开启时输出 `[codecprune] ... kept N%`，关闭时节点行为与原生节点一致。

`CODEC_PRUNE=0` 可完全关闭运行时补丁并恢复原生行为。Windows 下在启动 ComfyUI 前执行 `set CODEC_PRUNE=0`，或将其设为系统环境变量；Linux 与 macOS 下在启动命令前加前缀。

## 使用

节点的输入与原版 `MiniMax H3 Reference to Video` 相同，另有两个设置项：

| 设置 | 默认值 | 说明 |
|---|---|---|
| `prune` | `true` | 关闭时与原版节点逐张量一致 |
| `drop` | `0.70` | 参考**视频**要丢弃的 token 比例；`0` 表示仅使用阈值判定。图片始终使用阈值判定 |

提示词中的标记仍为 `<Picture i>`、`<Video k>`、`<Audio j>`。

## 配套 LoRA

官方 CodecOPSD LoRA（[HikariDawn/CodecOPSD-H3-Ref2VA](https://huggingface.co/HikariDawn/CodecOPSD-H3-Ref2VA)）采用 diffusers 命名，无法直接加载：ComfyUI 中 H3 的注意力将 `to_q`/`to_k`/`to_v` 融合为单个 `qkv_proj`，而三个投影各自携带独立的 LoRA 更新、各有自己的 A 矩阵。只拼接 B 矩阵会让它们与错误的 A 配对，因此必须按块对角形式合并。

执行一次转换：

```bash
python tools/convert_codecopsd_lora.py codecopsd_minimax_h3_ref2va.safetensors \
       codecopsd_h3_ref2va_comfy.safetensors --fp16
```

转换包含三步：

- 将 `to_out.0` 重命名为 `out_proj`，`ff.net.0.proj` 重命名为 `mlp.fc1`，`ff.net.2` 重命名为 `mlp.fc2`；
- 将 `to_q`/`to_k`/`to_v` 以精确的块对角形式合并为单个 `qkv_proj` 适配器，`delta = blockdiag(B_q,B_k,B_v) @ cat(A_q,A_k,A_v)`，秩由 128 变为 384；
- 补齐 ComfyUI 所需的每模块 `alpha`。实际缩放为 `alpha/rank`，因此 alpha 随秩从 128 调整为 384，训练时的 1.0 缩放得以保持。

使用标准 **Load LoRA** 节点加载转换后的文件，强度 1.0。该 LoRA 是在剪枝后的序列上蒸馏得到，需在 `prune` 开启时使用。

转换脚本依赖 `torch` 与 `safetensors`，ComfyUI 环境均已包含。

## 实现方式

`comfy/ldm/minimax/model.py` 保持不变。扩展在导入时替换模块上的两个对象：

- `comfy.ldm.minimax.model.PackedLayout` —— 一个支持参考块可选 `keep` 张量的子类；
- `MiniMaxH3Model._cond_video_rows` —— 以相同规则过滤条件行的包装函数。

`comfy.model_base` 通过模块属性引用 `PackedLayout`，因此替换类对象即可覆盖两处调用点。`keep` 掩码随每个 `minimax_refs` 块传递；`extra_conds` 只从该块读取 `latent` 与 `audio_latent`，其余字段原样透传。

若后续 ComfyUI 版本改动这些内部结构，补丁会检测到不匹配、记录警告，并退回到原生行为。

## 限制

- 参考**图片**掩码比参考实现多丢弃 3–6%（在上游示例中为 877 对 902、3,807 对 4,051）。已测试九种插值与量化组合，差异未能消除，说明来源是参考流程中的某个缩放细节，而非算法本身。参考**视频**掩码完全一致（同一示例中为 3,688 与 11,189，总数 37,296）。
- 运行时补丁依赖 ComfyUI 内部结构，与当前 H3 实现绑定。
- 剪枝会为每个参考增加一次 CPU 上的 H.264 编码与解码，2048 短边的图片约需一秒。

## 构建 reader

`bin/codec_reader.exe` 是唯一的编译组件。它解码 H.264 码流并返回每个宏块的运动、残差与 skip 信号，静态链接打过补丁的 FFmpeg 5.1。

```bash
# MSYS2: pacman -S --needed mingw-w64-x86_64-gcc make patch diffutils
sh build/build_reader.sh
```

脚本中记录了两个不直观的要求：

- `--disable-asm` 是必需的。补丁只修改 C 版 IDCT；启用汇编时会走 SIMD 路径，补丁不生效。
- `--enable-decoder=h263` 看似无关，但同样必需。运动矢量导出受 `CONFIG_MPEGVIDEODEC` 控制，而该项由 h263 解码器引入。

脚本面向 Windows 的 MSYS2。在 Linux 与 macOS 上需从最后的链接命令中去掉 `-lbcrypt`；它是 Windows 专用库，脚本未针对其他平台分支。

## 测试环境

单卡 Tesla V100-SXM2-16GB（sm_70），Windows，ComfyUI 0.46.0。基座模型 `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8`，配合 `dmad_minimax_h3_4step_full_critic_comfyui` 加速 LoRA，4 步，euler + beta。

测试范围有限：仅覆盖两种工作流（一张图参考、一段 5 秒视频参考）、单卡、单一模型配置。多参考组合、其他分辨率、其他硬件与长时间稳定性均未测量，绝对耗时不能直接套用到其他显卡。序列长度的比例应当成立，因为它取决于 token 数量而非硬件。

## 许可

本项目代码为 MIT（见 [LICENSE](LICENSE)）。随附组件各有独立许可，列于下方；MIT 条款不适用于它们：

- `codecopsd_vendor/` 为 CodecOPSD 的剪枝判定代码，© 2026 The CodecOPSD Authors，研究用途免费；商用需获得其作者授权。
- `bin/codec_reader.exe` 与 `build/ffmpeg-5.1.patch` 为 FFmpeg 5.1，遵循 GNU LGPL 2.1 及以后版本；补丁按该许可要求随附分发。补丁、许可全文、源码说明与构建脚本均位于 `build/`。
- `bin/ffmpeg.exe` 为可选项，未纳入版本库（87 MB）。存在时可将编码器固定为掩码验证时所用的构建；缺失时使用 ComfyUI 环境中的 imageio-ffmpeg 可执行文件。

## 引用

```bibtex
@article{wang2026codecopsd,
  title   = {On-Policy Codec Distillation for Efficient All-in-One Video Generation},
  author  = {Wang, Boyang and Blackburn-Matzen, Kevin and Saha, Oindrila and Gadelha, Matheus and Cheng, Zezhou},
  journal = {arXiv preprint arXiv:2611.31184},
  year    = {2026}
}
```
