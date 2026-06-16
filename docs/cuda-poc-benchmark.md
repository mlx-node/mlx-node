# mlx-node CUDA PoC — GB10 (DGX Spark) vs Apple M3 Max

PoC scope: device-agnostic eager fallbacks, **no custom CUDA kernels**. The
question this answers is "does mlx-node run on NVIDIA, and how fast vs Apple",
not "is the CUDA path tuned".

- **GB10**: NVIDIA GB10 (Grace-Blackwell, sm_121), 128 GB LPDDR5X (~273 GB/s),
  forced eager + flat KV (`MLX_QWEN35_FORCE_EAGER=1 MLX_QWEN35_PAGED_OVERRIDE=0`)
  because paged attention needs Metal kernels.
- **M3 Max**: Apple M3 Max 128 GB (~400 GB/s), numbers from the HF model cards
  (same `examples/lm.ts` harness → apples-to-apples for decode).
- **Harness** (`examples/lm.ts`): 4-turn capitals chat, `temp 0.6`,
  `reasoningEffort 'low'`. Decode = best tok/s over turns 2–4, median of 3 runs.

## Decode — apples-to-apples (same harness both sides)

```
 model                    GB10 dec   M3 Max dec   GB10/M3    NVFP4 vs Q4
 ───────────────────────  ────────   ──────────   ───────   ─────────────
 27B dense   Q4-affine      9.0        15.3        0.59×
 27B dense   NVFP4          9.1        14.9        0.61×     +1%  (tie)  on GB10
 35B-A3B MoE Q4-affine     42.1        55.1        0.76×
 35B-A3B MoE NVFP4         39.8        59.1        0.67×     −5% (slower) on GB10
```

Decode is memory-bandwidth-bound. GB10/M3 ≈ 0.59–0.76× tracks the bandwidth
ratio (273/400 = **0.68×**): dense sits just under it (eager per-op overhead +
GDN recurrence add non-bandwidth cost), MoE-Q4 just over it (more compute
headroom, GB10's compute helps). Cross-check: the separate `cuda-bench` 27B-Q4
decode = 8.87 tok/s, matching lm.ts 9.0.

**The DGX Spark decodes slower than a 2-year-old M3 Max laptop.** Expected —
it has less memory bandwidth and the PoC runs unfused fallbacks.

## Prefill — GB10 standalone (cuda-bench sweep, 128→8192 prompt)

Cards have no prefill numbers, so this is GB10-only.

```
 prompt tok   27B Q4     27B NVFP4
 ──────────   ───────    ─────────
   212        152        82
   692        169        90
  1333        165        91
  2613        163        91          (flat across a 48× length range)
  5173        155        —
 10293        156        —
```

Two findings:
1. **Prefill is flat vs length** → bottleneck is the GDN (gated-delta linear
   attention) recurrence, run **one timestep at a time** in the device-agnostic
   fallback (no CUDA GDN kernel). The quantized matmuls are the minority term.
2. **NVFP4 prefill is ~1.8× SLOWER than Q4-affine** (≈0.55× throughput,
   uniform across all lengths). No native Blackwell FP4 GEMM is exercised →
   MLX-CUDA dequantizes NVFP4, and its 2-level scaling (FP8 E4M3 block scale,
   group=16) is heavier than affine's 1-level scale+bias (group=64).

## Does NVFP4 help on the DGX? — No (this PoC)

```
              prefill          decode
 GB10  NVFP4  1.8× SLOWER      tie (dense) / 5% slower (MoE)
 M3Max NVFP4  (n/a on card)    3% slower (dense) / 7% FASTER (MoE)
```

NVFP4 only wins when native FP4 tensor cores drive the GEMM. The device-agnostic
PoC falls back to dequant, so on GB10 NVFP4 is pure overhead. (Note: on Apple,
MoE-NVFP4 *does* edge out Q4 — different code path.)

## Next perf levers (in impact order)

1. **CUDA GDN kernel** — kills the flat-prefill ceiling; biggest dense win.
2. **Native FP4 GEMM (qmm/qmv)** — turns NVFP4 from a penalty into a win.
3. **Paged attention on CUDA** — currently forced off (Metal-only kernels).
4. **Fused decode kernels** — close the sub-bandwidth-ratio gap on dense decode.

All four are real kernel-port milestones, out of scope for this PoC.
