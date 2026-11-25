"""Convert the CodecOPSD H3 Ref2VA LoRA (diffusers naming) into ComfyUI's H3 naming.

ComfyUI's H3 attention uses one fused `qkv_proj` Linear where diffusers has separate
`to_q` / `to_k` / `to_v`. The three LoRAs have different A matrices, so B cannot simply be
concatenated: the exact merged adapter is block-diagonal.

    delta_qkv = [B_q@A_q ; B_k@A_k ; B_v@A_v]
              = blockdiag(B_q, B_k, B_v) @ cat(A_q, A_k, A_v)

so up = blockdiag(...) [3*inner, 3*rank] and down = cat(...) [3*rank, hidden]. Scale is
alpha/rank in ComfyUI, so alpha must become 3*rank to keep the trained scale of 1.0.

    python convert_codecopsd_lora.py <in.safetensors> <out.safetensors> [--fp16]
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# diffusers module name (after transformer_blocks.N) -> ComfyUI module name
DIRECT = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
}
QKV = ["attn.to_q", "attn.to_k", "attn.to_v"]      # order matters: q, k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--fp16", action="store_true", help="store fp16 (half the size)")
    ap.add_argument("--dtype", default=None, help="explicit dtype, e.g. bf16")
    args = ap.parse_args()

    dt = torch.float16 if args.fp16 else torch.float32
    if args.dtype:
        dt = getattr(torch, args.dtype)

    with safe_open(args.src, "pt") as f:
        keys = list(f.keys())
        meta = f.metadata() or {}
        rank = f.get_slice("transformer_blocks.0.attn.to_q.lora_A.weight").get_shape()[0]
        alpha0 = float(meta.get("lora_alpha", rank))
        layers = sorted({int(k.split(".")[1]) for k in keys})
        print("source: %d keys, %d layers, rank %d, alpha %g" % (len(keys), len(layers), rank, alpha0))

        out = {}
        n_qkv = n_direct = 0

        for i in layers:
            # ---- fused qkv: block-diagonal ----
            parts_A, parts_B, present = [], [], []
            for m in QKV:
                a = "transformer_blocks.%d.%s.lora_A.weight" % (i, m)
                b = "transformer_blocks.%d.%s.lora_B.weight" % (i, m)
                if a in keys and b in keys:
                    parts_A.append(f.get_tensor(a).to(dt))
                    parts_B.append(f.get_tensor(b).to(dt))
                    present.append(m)
                else:
                    parts_A.append(None)          # filled with zeros below
                    parts_B.append(None)
            if any(p is not None for p in parts_A):
                in_dim = next(p for p in parts_A if p is not None).shape[1]
                out_dim = next(p for p in parts_B if p is not None).shape[0]
                dense_A = torch.zeros(len(QKV) * rank, in_dim, dtype=dt)
                dense_B = torch.zeros(len(QKV) * out_dim, len(QKV) * rank, dtype=dt)
                for j, (a, b) in enumerate(zip(parts_A, parts_B)):
                    if a is None:
                        continue
                    dense_A[j * rank:(j + 1) * rank] = a
                    dense_B[j * out_dim:(j + 1) * out_dim, j * rank:(j + 1) * rank] = b
                base = "diffusion_model.blocks.%d.attn.qkv_proj" % i
                out[base + ".lora_A.weight"] = dense_A
                out[base + ".lora_B.weight"] = dense_B
                # keep the trained scale: alpha/rank must stay alpha0/rank
                out[base + ".alpha"] = torch.tensor(alpha0 * len(QKV), dtype=torch.float32)
                n_qkv += 1

            # ---- 1:1 renames ----
            for src_name, dst_name in DIRECT.items():
                a = "transformer_blocks.%d.%s.lora_A.weight" % (i, src_name)
                b = "transformer_blocks.%d.%s.lora_B.weight" % (i, src_name)
                if a not in keys or b not in keys:
                    continue
                base = "diffusion_model.blocks.%d.%s" % (i, dst_name)
                out[base + ".lora_A.weight"] = f.get_tensor(a).to(dt)
                out[base + ".lora_B.weight"] = f.get_tensor(b).to(dt)
                out[base + ".alpha"] = torch.tensor(alpha0, dtype=torch.float32)
                n_direct += 1

    meta_out = {
        "format": "comfyui-lora",
        "converted_from": os.path.basename(args.src),
        "source_metadata": json.dumps(meta),
        "note": "qkv fused block-diagonal, alpha scaled to keep alpha/rank constant",
    }
    save_file(out, args.dst, metadata=meta_out)
    size = os.path.getsize(args.dst) / (1 << 30)
    print("fused qkv modules: %d | renamed modules: %d | tensors: %d | %.2f GiB"
          % (n_qkv, n_direct, len(out), size))
    print("wrote", args.dst)


if __name__ == "__main__":
    sys.exit(main())
