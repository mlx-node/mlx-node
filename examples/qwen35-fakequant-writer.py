#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx", "safetensors", "numpy"]
# ///
"""
Qwen3.5 FAKE-QUANT writer: produce bf16 model copies whose quantizable weights
have been ROUND-TRIPPED through a weight quantizer and back to bf16. Normal bf16
inference over such a copy then SIMULATES that quant's weight quality — no new
kernels needed.

Two schemes (8-bit), matching examples/qwen35-int8-quant-accuracy.ts / the
weight-fidelity probe (examples/qwen35-quant-weight-fidelity.py):

  sym8 (Option B): per-OUTPUT-CHANNEL symmetric int8.
      W[N,K]; s[n] = max_k |W[n,k]| / 127
      Wq = round(W / s[:,None]).clip(-127, 127)
      W_recon = Wq * s[:,None]    (ONE scale/row, no zero-point, no groups)

  aff8 (current Q8_K_XL): MLX affine per-GROUP int8.
      mx.quantize(W, group_size=64, bits=8, mode="affine")
      mx.dequantize(...)         (per-group scale + bias along K)

QUANTIZED set (exactly the production convert path): attn q/k/v/o,
mlp gate/up/down, GDN linear_attn.in_proj_qkv, linear_attn.in_proj_z,
linear_attn.out_proj. EXCLUDED (copied unchanged): embed_tokens (tied lm_head),
ALL *norm*, conv1d, A_log, dt_bias, in_proj_a, in_proj_b, vision_tower.

  Note: production fuses in_proj_qkv+in_proj_z into in_proj_qkvz (concat on the
  OUTPUT axis) before quantizing. Both schemes are invariant to output-axis
  concat (sym8 scale is per output row; aff8 groups run along K), so quantizing
  the split tensors here is numerically identical to quantizing the fused one.

Output dirs are valid bf16 model dirs: config.json / tokenizer / etc. copied,
model.safetensors overwritten with round-tripped weights (all tensors bf16).

Usage:
  uv run examples/qwen35-fakequant-writer.py \
    --src /Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16 \
    --out-sym8 /tmp/qwen35-0.8b-sym8 \
    --out-aff8 /tmp/qwen35-0.8b-aff8
"""

import argparse
import glob
import json
import os
import shutil
import struct
import sys

import numpy as np
import mlx.core as mx

GROUP_SIZE = 64
BITS = 8

# Quantizable suffixes -> type bucket (the EXACT production set).
TYPE_OF_SUFFIX = {
    "self_attn.q_proj.weight": "q_proj",
    "self_attn.k_proj.weight": "k_proj",
    "self_attn.v_proj.weight": "v_proj",
    "self_attn.o_proj.weight": "o_proj",
    "mlp.gate_proj.weight": "gate_proj",
    "mlp.up_proj.weight": "up_proj",
    "mlp.down_proj.weight": "down_proj",
    "linear_attn.in_proj_qkv.weight": "qkvz",
    "linear_attn.in_proj_z.weight": "qkvz",
    "linear_attn.out_proj.weight": "out_proj",
}

EXCLUDE_SUBSTR = [
    "vision_tower", "visual.", "lm_head", "embed_tokens", "embedding.",
    "layernorm", "rms_norm", "_norm.", ".norm.", "conv1d", "A_log", "dt_bias",
    "in_proj_a.", "in_proj_b.", "in_proj_ba.", "mtp.", "mtp_",
]


def is_quantizable(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    for bad in EXCLUDE_SUBSTR:
        if bad in key:
            return False
    for suf in TYPE_OF_SUFFIX:
        if key.endswith(suf):
            return True
    return False


# ---------------------------------------------------------------------------
# safetensors raw IO (numpy). We keep everything as numpy; bf16 handled as u16.
# ---------------------------------------------------------------------------
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def read_raw(path, base, info):
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        return f.read(e - s)


def bf16_bytes_to_f32(buf, shape):
    u16 = np.frombuffer(buf, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape).astype(np.float32)


def f32_to_bf16_round(arr_f32: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16, returned as a numpy bf16-view via mlx
    (mlx provides a real bfloat16 dtype, so safetensors saves a valid BF16)."""
    a = mx.array(arr_f32.astype(np.float32)).astype(mx.bfloat16)
    mx.eval(a)
    return a


# ---------------------------------------------------------------------------
# Quant schemes (operate in f32, return f32 reconstruction).
# ---------------------------------------------------------------------------
def sym8_recon(W: np.ndarray) -> np.ndarray:
    absmax = np.max(np.abs(W), axis=1, keepdims=True)
    s = absmax / 127.0
    s = np.where(s == 0.0, 1.0, s)
    Wq = np.clip(np.round(W / s), -127.0, 127.0)
    return (Wq * s).astype(np.float32)


def aff8_recon(W: np.ndarray) -> np.ndarray:
    w = mx.array(W.astype(np.float32))
    wq, sc, bi = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    wr = mx.dequantize(wq, sc, bi, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    mx.eval(wr)
    return np.array(wr, dtype=np.float32)


def model_files(model_dir):
    idx = glob.glob(f"{model_dir}/model.safetensors.index.json")
    if idx:
        with open(idx[0]) as f:
            wmap = json.load(f)["weight_map"]
        files = sorted(set(wmap.values()))
        return [os.path.join(model_dir, fn) for fn in files]
    return sorted(glob.glob(f"{model_dir}/*.safetensors"))


def copy_aux(src, dst):
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn.endswith(".safetensors.index.json"):
            continue
        sp = os.path.join(src, fn)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(dst, fn))


def build(src, out_sym8, out_aff8):
    files = model_files(src)
    if not files:
        sys.exit(f"no safetensors under {src}")
    if len(files) != 1:
        # 0.8B is single-file; keep this writer simple & loud if not.
        print(f"WARNING: {len(files)} shards; writing a single merged model.safetensors each")

    copy_aux(src, out_sym8)
    copy_aux(src, out_aff8)

    sym_tensors = {}
    aff_tensors = {}
    n_quant = 0
    n_copy = 0
    # diagnostic accumulators (param-weighted rel-RMSE vs bf16 source recon)
    sse_sym = den_sym = 0.0
    sse_aff = den_aff = 0.0

    for path in files:
        hdr, base = read_header(path)
        for key, info in hdr.items():
            if key == "__metadata__":
                continue
            dtype = info["dtype"]
            shape = info["shape"]
            raw = read_raw(path, base, info)

            if is_quantizable(key) and dtype == "BF16" and len(shape) == 2:
                W = bf16_bytes_to_f32(raw, shape)
                Rs = sym8_recon(W)
                Ra = aff8_recon(W)
                sym_tensors[key] = f32_to_bf16_round(Rs)
                aff_tensors[key] = f32_to_bf16_round(Ra)
                n_quant += 1
                sse_sym += float(np.sum((Rs - W) ** 2)); den_sym += float(np.sum(W ** 2))
                sse_aff += float(np.sum((Ra - W) ** 2)); den_aff += float(np.sum(W ** 2))
            else:
                # copy byte-identical (preserve original dtype/bytes exactly)
                if dtype == "BF16":
                    t = mx.array(np.frombuffer(raw, dtype=np.uint16).copy()).view(mx.bfloat16).reshape(shape)
                elif dtype == "F32":
                    t = mx.array(np.frombuffer(raw, dtype=np.float32).copy().reshape(shape))
                elif dtype == "F16":
                    t = mx.array(np.frombuffer(raw, dtype=np.float16).copy().reshape(shape))
                elif dtype in ("I32",):
                    t = mx.array(np.frombuffer(raw, dtype=np.int32).copy().reshape(shape))
                elif dtype in ("U32",):
                    t = mx.array(np.frombuffer(raw, dtype=np.uint32).copy().reshape(shape))
                else:
                    raise ValueError(f"unhandled copy dtype {dtype} for {key}")
                mx.eval(t)
                sym_tensors[key] = t
                aff_tensors[key] = t
                n_copy += 1

    # mlx writes BF16 natively; pass the mlx-array dict straight to save_safetensors.
    meta = {"format": "pt", "fakequant": "qwen35-fakequant-writer"}
    for d in (sym_tensors, aff_tensors):
        mx.eval(list(d.values()))
    mx.save_safetensors(os.path.join(out_sym8, "model.safetensors"), sym_tensors,
                        metadata={**meta, "scheme": "sym8"})
    mx.save_safetensors(os.path.join(out_aff8, "model.safetensors"), aff_tensors,
                        metadata={**meta, "scheme": "aff8"})

    rmse_sym = (sse_sym / den_sym) ** 0.5 if den_sym else 0.0
    rmse_aff = (sse_aff / den_aff) ** 0.5 if den_aff else 0.0
    print(json.dumps({
        "src": src,
        "out_sym8": out_sym8,
        "out_aff8": out_aff8,
        "n_quantized": n_quant,
        "n_copied": n_copy,
        "pw_rel_rmse_sym8_vs_bf16": rmse_sym,
        "pw_rel_rmse_aff8_vs_bf16": rmse_aff,
        "note": "round-trip recon measured in f32 BEFORE the final bf16 store",
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16")
    ap.add_argument("--out-sym8", default="/tmp/qwen35-0.8b-sym8")
    ap.add_argument("--out-aff8", default="/tmp/qwen35-0.8b-aff8")
    a = ap.parse_args()
    build(a.src, a.out_sym8, a.out_aff8)


if __name__ == "__main__":
    main()
