#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx", "safetensors", "numpy"]
# ///
"""
Qwen3.5 weight-quant fidelity probe: Option B (per-channel symmetric int8)
vs current production Q8 (per-group affine int8).

For every QUANTIZABLE linear weight (the exact set the production convert path
quantizes), reconstruct the weight two ways from the bf16 ORIGINAL and measure
how close each reconstruction is to the bf16 source:

  sym8 (Option B): per-OUTPUT-CHANNEL symmetric int8.
      W[N,K]; s[n] = max_k |W[n,k]| / 127
      Wq = round(W / s[:,None]).clip(-127, 127)
      W_recon = Wq * s[:,None]
      ONE scale per row, no zero-point, no groups.

  aff8 (current Q8_K_XL): MLX affine per-GROUP int8.
      mx.quantize(W, group_size=64, bits=8, mode="affine")
      mx.dequantize(...)
      Per-group (64 along K) scale + bias (zero-point).

Fidelity vs bf16 original W (cast to f32 for the math, reported in f32):
  - per-output-ROW cosine similarity
  - per-layer relative RMSE = ||W_recon - W|| / ||W||   (Frobenius)

Aggregation:
  - PARAM-WEIGHTED overall (each layer weighted by its element count)
  - per-layer-TYPE (q/k/v/o, gate/up/down, qkvz, out_proj)
  - WORST layer (lowest cosine) per quant scheme

The production convert path (`should_quantize`, crates/.../convert.rs) quantizes:
  attn q/k/v/o, mlp gate/up/down, GDN in_proj_qkv+in_proj_z (fused->in_proj_qkvz),
  GDN out_proj.
Excluded (and excluded here): embed_tokens, lm_head, ALL *norm* weights,
  conv1d, A_log, dt_bias, vision_tower, and in_proj_a/in_proj_b (these are
  merged into in_proj_ba which production EXCLUDES from quant).

Note on split vs fused: production fuses in_proj_qkv+in_proj_z into in_proj_qkvz
(concat along the OUTPUT axis) before quantizing. Both schemes are invariant to
output-axis concatenation:
  - sym8 scale is per output row -> identical whether split or fused.
  - aff8 groups run along the INPUT (K) axis -> identical whether split or fused.
So we measure the two source tensors directly and bucket them under type "qkvz".
"""

import sys
import glob
import json
import struct
import math
from collections import defaultdict

import numpy as np
import mlx.core as mx

GROUP_SIZE = 64
BITS = 8

# ----------------------------------------------------------------------------
# Which source tensors are quantized in production, and their type bucket.
# Keyed by the dotted suffix that appears in the safetensors key.
# ----------------------------------------------------------------------------
TYPE_OF_SUFFIX = {
    "self_attn.q_proj.weight": "q_proj",
    "self_attn.k_proj.weight": "k_proj",
    "self_attn.v_proj.weight": "v_proj",
    "self_attn.o_proj.weight": "o_proj",
    "mlp.gate_proj.weight": "gate_proj",
    "mlp.up_proj.weight": "up_proj",
    "mlp.down_proj.weight": "down_proj",
    "linear_attn.in_proj_qkv.weight": "qkvz",   # fused into in_proj_qkvz in prod
    "linear_attn.in_proj_z.weight": "qkvz",     # fused into in_proj_qkvz in prod
    "linear_attn.out_proj.weight": "out_proj",
}

# Hard excludes (defensive; also covered by the suffix allow-list above).
EXCLUDE_SUBSTR = [
    "vision_tower", "visual.", "lm_head", "embed_tokens", "embedding.",
    "layernorm", "rms_norm", "_norm.", "conv1d", "A_log", "dt_bias",
    "in_proj_a.", "in_proj_b.", "in_proj_ba.", "mtp.", "mtp_",
]


def classify(key: str):
    if not key.endswith(".weight"):
        return None
    for bad in EXCLUDE_SUBSTR:
        if bad in key:
            return None
    for suf, typ in TYPE_OF_SUFFIX.items():
        if key.endswith(suf):
            return typ
    return None


# ----------------------------------------------------------------------------
# safetensors mmap loader (returns a bf16 tensor as f32 numpy)
# ----------------------------------------------------------------------------
_DT = {
    "F32": np.float32, "F16": np.float16, "BF16": "bf16",
    "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def load_tensor_f32(path, base_off, info):
    dtype = info["dtype"]
    shape = info["shape"]
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base_off + s)
        buf = f.read(e - s)
    if dtype == "BF16":
        u16 = np.frombuffer(buf, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(buf, dtype=np.float32)
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    return arr.reshape(shape).astype(np.float32)


# ----------------------------------------------------------------------------
# Quant schemes
# ----------------------------------------------------------------------------
def sym8_recon(W_f32: np.ndarray) -> np.ndarray:
    """Per-output-channel symmetric int8."""
    absmax = np.max(np.abs(W_f32), axis=1, keepdims=True)  # [N,1]
    s = absmax / 127.0
    s = np.where(s == 0.0, 1.0, s)  # avoid div0 on all-zero rows
    Wq = np.round(W_f32 / s)
    Wq = np.clip(Wq, -127.0, 127.0)
    return Wq * s


def aff8_recon_mx(W_f32: np.ndarray) -> np.ndarray:
    """MLX affine per-group int8 -> dequantized, returned as f32 numpy."""
    w = mx.array(W_f32)
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    wr = mx.dequantize(wq, scales, biases, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    mx.eval(wr)
    return np.array(wr, dtype=np.float32)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def row_cosine(W: np.ndarray, R: np.ndarray) -> np.ndarray:
    num = np.sum(W * R, axis=1)
    dW = np.linalg.norm(W, axis=1)
    dR = np.linalg.norm(R, axis=1)
    denom = dW * dR
    out = np.ones_like(num)
    nz = denom > 0
    out[nz] = num[nz] / denom[nz]
    return out  # [N]


def rel_rmse(W: np.ndarray, R: np.ndarray) -> float:
    num = np.linalg.norm(R - W)
    den = np.linalg.norm(W)
    return float(num / den) if den > 0 else 0.0


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def model_files(model_dir):
    idx = glob.glob(f"{model_dir}/model.safetensors.index.json")
    if idx:
        with open(idx[0]) as f:
            wmap = json.load(f)["weight_map"]
        files = sorted(set(wmap.values()))
        return [f"{model_dir}/{fn}" for fn in files]
    return sorted(glob.glob(f"{model_dir}/*.safetensors"))


def analyze(model_dir):
    files = model_files(model_dir)
    # per-layer records: (key, type, n_elem, cos_sym, rmse_sym, cos_aff, rmse_aff)
    records = []
    for path in files:
        hdr, base = read_header(path)
        for key, info in hdr.items():
            if key == "__metadata__":
                continue
            typ = classify(key)
            if typ is None:
                continue
            W = load_tensor_f32(path, base, info)
            if W.ndim != 2:
                continue
            n_elem = W.size

            Rs = sym8_recon(W)
            Ra = aff8_recon_mx(W)

            cos_s = row_cosine(W, Rs)
            cos_a = row_cosine(W, Ra)
            rec = {
                "key": key,
                "type": typ,
                "n": n_elem,
                "cos_sym_mean": float(np.mean(cos_s)),
                "cos_sym_min": float(np.min(cos_s)),
                "rmse_sym": rel_rmse(W, Rs),
                "cos_aff_mean": float(np.mean(cos_a)),
                "cos_aff_min": float(np.min(cos_a)),
                "rmse_aff": rel_rmse(W, Ra),
            }
            records.append(rec)
            del W, Rs, Ra
    return records


def pw(records, field):
    tot = sum(r["n"] for r in records)
    if tot == 0:
        return 0.0
    return sum(r[field] * r["n"] for r in records) / tot


def fmt_cos(x):
    return f"{x:.6f}"


def fmt_rmse(x):
    return f"{x:.5f}"


def report(model_dir, records):
    print("=" * 92)
    print(f"MODEL: {model_dir}")
    n_layers = len(records)
    n_params = sum(r["n"] for r in records)
    print(f"  quantizable 2D linears: {n_layers}   total params: {n_params/1e6:.1f}M")
    print("-" * 92)

    # Param-weighted overall
    s_cos = pw(records, "cos_sym_mean")
    a_cos = pw(records, "cos_aff_mean")
    s_rmse = pw(records, "rmse_sym")
    a_rmse = pw(records, "rmse_aff")
    print("PARAM-WEIGHTED OVERALL")
    print(f"  {'scheme':6s} {'pw mean cos':>14s} {'pw rel-RMSE':>14s}")
    print(f"  {'sym8':6s} {fmt_cos(s_cos):>14s} {fmt_rmse(s_rmse):>14s}")
    print(f"  {'aff8':6s} {fmt_cos(a_cos):>14s} {fmt_rmse(a_rmse):>14s}")
    cos_gap = a_cos - s_cos
    rmse_ratio = (s_rmse / a_rmse) if a_rmse > 0 else float("inf")
    print(f"  GAP    cos(aff-sym)={cos_gap:+.6f}   rmse sym/aff={rmse_ratio:.3f}x "
          f"(>1 => sym8 worse)")
    print("-" * 92)

    # Per layer type
    print("PER-LAYER-TYPE  (param-weighted within type)")
    print(f"  {'type':10s} {'#':>3s} {'sym cos':>10s} {'aff cos':>10s} {'cosΔ':>9s} "
          f"{'sym rmse':>9s} {'aff rmse':>9s} {'rmse x':>7s}")
    by_type = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    order = ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj", "qkvz", "out_proj"]
    type_rows = {}
    for t in order:
        rs = by_type.get(t)
        if not rs:
            continue
        sc = pw(rs, "cos_sym_mean")
        ac = pw(rs, "cos_aff_mean")
        sr = pw(rs, "rmse_sym")
        ar = pw(rs, "rmse_aff")
        rx = (sr / ar) if ar > 0 else float("inf")
        type_rows[t] = (len(rs), sc, ac, ac - sc, sr, ar, rx)
        print(f"  {t:10s} {len(rs):>3d} {sc:>10.6f} {ac:>10.6f} {ac-sc:>+9.6f} "
              f"{sr:>9.5f} {ar:>9.5f} {rx:>7.2f}")
    print("-" * 92)

    # Worst layers
    worst_sym = min(records, key=lambda r: r["cos_sym_mean"])
    worst_aff = min(records, key=lambda r: r["cos_aff_mean"])
    # worst by rmse too
    worst_sym_rmse = max(records, key=lambda r: r["rmse_sym"])
    print("WORST LAYERS")
    print(f"  sym8 worst mean-cos: {worst_sym['cos_sym_mean']:.6f} "
          f"(min-row {worst_sym['cos_sym_min']:.6f}, rmse {worst_sym['rmse_sym']:.5f}) "
          f"[{worst_sym['type']}]")
    print(f"      same layer aff8 mean-cos: {worst_sym['cos_aff_mean']:.6f} "
          f"(rmse {worst_sym['rmse_aff']:.5f})  GAP cos {worst_sym['cos_aff_mean']-worst_sym['cos_sym_mean']:+.6f}")
    print(f"      key: {worst_sym['key']}")
    print(f"  sym8 worst rel-RMSE: {worst_sym_rmse['rmse_sym']:.5f} "
          f"(cos {worst_sym_rmse['cos_sym_mean']:.6f}) [{worst_sym_rmse['type']}]")
    print(f"      same layer aff8 rmse: {worst_sym_rmse['rmse_aff']:.5f}  "
          f"ratio {worst_sym_rmse['rmse_sym']/worst_sym_rmse['rmse_aff']:.2f}x")
    print(f"      key: {worst_sym_rmse['key']}")
    print(f"  aff8 worst mean-cos: {worst_aff['cos_aff_mean']:.6f} "
          f"(rmse {worst_aff['rmse_aff']:.5f}) [{worst_aff['type']}]")
    print("=" * 92)
    print()
    return {
        "model": model_dir,
        "n_layers": n_layers,
        "n_params": n_params,
        "pw_cos_sym": s_cos, "pw_cos_aff": a_cos,
        "pw_rmse_sym": s_rmse, "pw_rmse_aff": a_rmse,
        "cos_gap": cos_gap, "rmse_ratio": rmse_ratio,
        "type_rows": type_rows,
        "worst_sym": worst_sym, "worst_aff": worst_aff,
        "worst_sym_rmse": worst_sym_rmse,
    }


def main():
    dirs = sys.argv[1:] or [
        "/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16",
        "/Volumes/P4510/models/Qwen3.5-4B-mlx",
    ]
    summaries = []
    for d in dirs:
        recs = analyze(d)
        summaries.append(report(d, recs))

    # final cross-model verdict line (machine-friendly)
    print("#" * 92)
    print("VERDICT SUMMARY (param-weighted)")
    for s in summaries:
        name = s["model"].rstrip("/").split("/")[-1]
        print(f"  {name:36s} sym8 cos={s['pw_cos_sym']:.6f} rmse={s['pw_rmse_sym']:.5f} | "
              f"aff8 cos={s['pw_cos_aff']:.6f} rmse={s['pw_rmse_aff']:.5f} | "
              f"cosΔ={s['cos_gap']:+.6f} rmse x{s['rmse_ratio']:.2f}")
    print("#" * 92)


if __name__ == "__main__":
    main()
