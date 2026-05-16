# Autoresearch worklog — Qwen3.5-4B prefill, FlashQLA direction (2026-05-16)

End-to-end record of a single autoresearch loop targeting Qwen3.5-4B prefill
latency on Apple Silicon M3 Max. Bench: median time-to-first-delta of
`session.sendStream(prompt, …)` against `.cache/models/Qwen3.5-4B-mlx`,
correctness-gated by an 8-token greedy fingerprint at temperature=0.

Final shippable state: **-13.7% same-binary A/B at 1024-token prompt**
(median 725 ms vs 819 ms legacy), **-4.2% at 4096-token multi-chunk**
(2593 ms vs 2707 ms), max_abs_diff=0 throughout. Diff: 8 files modified,
~262 net LOC in `crates/mlx-core` and `crates/mlx-sys`. See
`autoresearch.jsonl` for the run-by-run primary data.

## TL;DR — what shipped

| Tag | Where | Toggle | Mechanism |
|---|---|---|---|
| **E37** | `qwen3_5/model.rs::forward_inner` | `MLX_DISABLE_E37_LAST_TOKEN_SLICE=1` | Slice `h` to last token before `final_norm + lm_head`. Skips ~T-1 rows of a `[B,T,2560] @ [2560,248320]` matmul. |
| **E38** | `qwen3_5/decoder_layer.rs::forward_with_optional_last_slice` | (same as E37) | Last layer slices `h` to last token AFTER attention residual, BEFORE post-attention norm + MLP. KV writes preserved on full T. |
| **E28+E31** | `qwen3_5/model.rs::chunked_prefill` | `MLX_PREFILL_SYNC_BETWEEN_CHUNKS=1` | `PREFILL_STEP_SIZE` 2048→1024 + `async_eval_layer_caches` (vs sync `eval`) between chunks. |
| **E39** | `transformer/mlp.rs::finalize_gate_up` + `mlx_fused_ops.cpp::mlx_swiglu_mlp_forward_stacked` | `MLX_DISABLE_E39_STACKED_MLP=1` | Pre-stack `[w_gate; w_up]^T` at model load. One matmul instead of two; per-call transposes baked in. |
| **E40** | `mlx_fused_ops.cpp` | (same as E39) | In the E39 stacked path, use `qwen35_common::swiglu()` (mlx::core::compile-cached fused `sigmoid·gate·up`) instead of inline ops. |
| **E5+E36** | `crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc` | (no toggle — kernel-internal) | Threadgroup `decay_mat[BT*BT]` and `decay_self[BT]` precompute. M5+ only (`CHUNK_MIN_GPU_GEN=17`); correctness verified on M3 by temporarily lowering the gate. |

The wins compose: each toggle reverts its piece independently, validated
by same-binary A/B (see methodology below).

## Scaling

| Prompt tokens | Chunks | best | legacy | Δ |
|---:|---:|---:|---:|---:|
| 1024 (single chunk) | 1 | 725 ms | 819 ms | **-13.7%** |
| 2048 | 2 | 1410 ms | 1537 ms | -8.3% |
| 4096 | 3-4 | 2593 ms | 2707 ms | -4.2% |
| 8192 | 7 | 5032 ms | 5040 ms | -0.15% |

Reason for the falloff at long prompts: E37+E38 save fixed work (one
lm_head + one last-layer MLP slice per chunk). E39+E40 save kernel-launch
overhead on the 32-per-chunk MLPs. None of these scale with full-attn
SDPA's O(T·T_ctx) cost — at 8192 that's where most of the time goes, and
the wins disappear in the noise.

## What didn't ship (and why)

Recorded so the next attempt doesn't re-discover the same dead ends.

### E41 — load-time Q+K+V projection stack (REJECTED, ~2% regression at 1024)

Mirror of E39's MLP gate+up stack but applied to attention's Q/K/V
projections. At model load, concat `[w_q; w_k; w_v]` along axis 0, transpose,
store. Forward issues one matmul → `[B,T,Q+K+V]`, then `slice_axis(2, …)`
into three views.

**Why it failed:** MLX silently materialized a contiguous copy when the
sliced view got reshaped to per-head `[B,T,H,D]`. The slice gave strides
that didn't match the reshape's target contiguous strides, so MLX inserted
~13.7 MB of memory traffic per slice × 3 per attn layer × 8 full-attn
layers = ~330 MB of hidden copying per chunk. That dwarfed the savings
from 16 fewer matmul kernel launches.

**Contrast with E39:** the MLP stack works because the downstream op is
elementwise (sigmoid·gate·up), then a matmul. Both handle strided arrays
without copy. Attention's per-head reshape is the killer.

**Fix path (not pursued):** a C++ helper that materializes the three
outputs as contiguous tensors via a single fused slice+copy. Doubles the
implementation effort; deferred.

### E42 — last-layer attention internal slice (REJECTED, mixed)

Push the E38 last-token slice INSIDE attention: after RoPE + cache write,
slice queries and gate to last token; SDPA becomes `[B,H,1,D] × [B,H,T_ctx,D]`,
o_proj becomes `[B,1,hidden]`. KV cache writes preserved on full T.

**Why it failed:** results inconsistent across prompt sizes.

| Prompt | Direction | Δ (median across 3-7 pairs) |
|---|---|---:|
| 1024 | slower | +2.2% |
| 2048 | slower | +2.8% |
| 4096 | faster | -1.6% |

Hypothesis: MLX's `mlx_fast_scaled_dot_product_attention` has at least two
dispatch regimes — a decode-tuned path (T_q=1, small T_ctx) and a
prefill-tuned path (large T_q). A "1 query attending to large T_ctx" at
non-decode time falls between them and is suboptimally kernel-tuned.
Wins only emerge at chunk count ≥3 when cumulative T_ctx savings dominate
per-chunk kernel-switch overhead. For the bench's headline 1024 case
that's a regression.

### E43 — skip per-chunk `clear_cache()` (REJECTED, within noise)

Toggle to skip the `crate::array::clear_cache()` call between prefill
chunks. Hypothesis: clearing every chunk forces MLX's allocator to redo
work that immediately gets re-allocated.

**Why it failed:** 4096 showed +1.2% slower (3 pairs, noisy). 8192 showed
+0.8% slower. Peak memory rose 0.07-0.2 GB (small relative to ~7 GB total).
The clear cost is small relative to chunk compute; skipping it changes
nothing measurable. Reverted.

## Methodology lessons

1. **Cross-session bench variance is ~10%** on this M3 Max. The early
   experiments (E13–E22) chased phantom wins that were just thermal/state
   drift. Only same-binary back-to-back A/B with env-var toggles is
   reliable — within-session variance drops to ~1-2%.

2. **Every experiment must have a `MLX_DISABLE_E<N>_<NAME>` toggle.** No
   exception. Without it you can't isolate which optimization is doing
   the work after stacking; with it you can A/B cheaply forever.

3. **Slice-then-reshape across a strided axis forces a copy in MLX.** If
   you're considering a load-time weight-stack optimization that ends in a
   reshape-into-per-head shape, expect a regression. Only purely elementwise
   downstream ops absorb strided views cheaply.

4. **Multi-chunk amplifies what's per-chunk.** E37 amplified from 1 chunk
   to N chunks. E42's regression also amplified — kernel-switch overhead
   per chunk added up. Test multi-chunk before declaring a single-chunk
   win shippable.

5. **MLX's lazy graph does NOT propagate downstream slicing back through
   matmul.** That's why E37 works at all — even though the logits are
   sliced to last token immediately after `lm_head`, the matmul still
   computes the full [B,T,vocab] without an explicit `slice` before it.

## Surface area (files touched)

```
crates/mlx-core/src/models/qwen3_5/decoder_layer.rs    (+51)  E38
crates/mlx-core/src/models/qwen3_5/model.rs            (+65)  E28/E31/E37
crates/mlx-core/src/models/qwen3_5/persistence.rs       (+4)  E39 (calls finalize_gate_up)
crates/mlx-core/src/models/qwen3_5/quantized_linear.rs  (+8)  E39 (MLPVariant wrapper)
crates/mlx-core/src/transformer/mlp.rs                 (+57)  E39 (finalize_gate_up + forward dispatch)
crates/mlx-sys/src/lib.rs                               (+9)  E39 (FFI binding)
crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc (+44)  E5+E36 (decay caches)
crates/mlx-sys/src/mlx_fused_ops.cpp                   (+37)  E39+E40 (stacked SwiGLU)
```

Plus harness, not under `crates/`:

```
autoresearch.md         — mission spec
autoresearch.ideas.md   — FlashQLA→Metal idea catalog
autoresearch.config.json — bench knobs
autoresearch.jsonl       — append-only experiment log (primary data)
autoresearch.sh          — driver
scripts/bench-gdn-prefill.ts — bench harness
```

## What's still on the table

In approximate order of (effort, expected gain):

- **Per-step GDN kernel D1**: process 2 v-columns per simdgroup in
  `gated_delta_step.metal.inc`. ~50 LOC, register-pressure risk, needs
  careful Metal correctness validation. M3 prefill GDN runs through this
  kernel today (see `gated_delta_kernel`, not `gated_delta_chunked`), so
  this is the highest-leverage M3 win available.
- **C++ text-prefill port** mirroring `mlx_qwen35_vlm_prefill`. Multi-hour
  scope; would compile-cache the per-chunk forward graph end-to-end.
  Expected 3-5% from FFI/graph-build savings.
- **FlashQLA algebraic ports to `gated_delta_chunked.metal.inc`**: A1, A2,
  A4 from `autoresearch.ideas.md`. All M5+ only because the chunked kernel
  is gated to gen ≥ 17. Won't register on this M3 bench but lay groundwork
  for the M5 cut.
- **Drop the M5 gate on chunked kernel** after A1-A4 land. If chunked beats
  per-step on M3 with the FlashQLA optimizations, `CHUNK_MIN_GPU_GEN` can
  come down. Highest theoretical M3 win, but blocked on the kernel work
  above.

## Methodology one-liner for the next attempt

> Build native. Bench `bash autoresearch.sh` for ON. Bench
> `MLX_DISABLE_E<N>_<NAME>=1 bash autoresearch.sh` for OFF, immediately
> after, in the same shell. Repeat 3 pairs. If pairs disagree in direction,
> the signal is below noise — reject or redesign. Never compare across
> sessions.
