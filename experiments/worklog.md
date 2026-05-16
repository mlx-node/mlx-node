# Autoresearch worklog — Qwen3.5-4B prefill, FlashQLA direction (2026-05-16)

End-to-end record of a single autoresearch loop targeting Qwen3.5-4B prefill
latency on Apple Silicon M3 Max. Bench: median time-to-first-delta of
`session.sendStream(prompt, …)` against `.cache/models/Qwen3.5-4B-mlx`,
correctness-gated by an 8-token greedy fingerprint at temperature=0.

Final shippable state (after the second push on 2026-05-17):
**-16.0% same-binary A/B at 1024-token prompt** (median 572.7 ms vs
682.2 ms legacy), max_abs_diff=0 throughout. The original arc shipped
-13.7%; the post-arc E47 (2-v-col GDN register-blocking) added another
-2.3%. E48 (4-v-col) remains opt-in at +0.5% on top, available via
`MLX_ENABLE_E48_GDN_4VCOL=1`. See `autoresearch.jsonl` for the run-by-run
primary data.

## TL;DR — what shipped

| Tag | Where | Toggle | Mechanism |
|---|---|---|---|
| **E37** | `qwen3_5/model.rs::forward_inner` | `MLX_DISABLE_E37_LAST_TOKEN_SLICE=1` | Slice `h` to last token before `final_norm + lm_head`. Skips ~T-1 rows of a `[B,T,2560] @ [2560,248320]` matmul. |
| **E38** | `qwen3_5/decoder_layer.rs::forward_with_optional_last_slice` | (same as E37) | Last layer slices `h` to last token AFTER attention residual, BEFORE post-attention norm + MLP. KV writes preserved on full T. |
| **E28+E31** | `qwen3_5/model.rs::chunked_prefill` | `MLX_PREFILL_SYNC_BETWEEN_CHUNKS=1` | `PREFILL_STEP_SIZE` 2048→1024 + `async_eval_layer_caches` (vs sync `eval`) between chunks. |
| **E39** | `transformer/mlp.rs::finalize_gate_up` + `mlx_fused_ops.cpp::mlx_swiglu_mlp_forward_stacked` | `MLX_DISABLE_E39_STACKED_MLP=1` | Pre-stack `[w_gate; w_up]^T` at model load. One matmul instead of two; per-call transposes baked in. |
| **E40** | `mlx_fused_ops.cpp` | (same as E39) | In the E39 stacked path, use `qwen35_common::swiglu()` (mlx::core::compile-cached fused `sigmoid·gate·up`) instead of inline ops. |
| **E5+E36** | `crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc` | (no toggle — kernel-internal) | Threadgroup `decay_mat[BT*BT]` and `decay_self[BT]` precompute. M5+ only (`CHUNK_MIN_GPU_GEN=17`); correctness verified on M3 by temporarily lowering the gate. |
| **E47** | `crates/mlx-sys/src/metal/gated_delta_step_2vcol.metal.inc` + `mlx_gated_delta.cpp` dispatch | `MLX_DISABLE_E47_GDN_2VCOL=1` | Per-step GDN kernel: each simdgroup handles 2 v-cols (dv_A=2y, dv_B=2y+1), sharing q[Dk]+k[Dk] loads. Grid-Y halves to Dv/2. -2.0% at 1024 single-chunk; neutral at 4096+. |

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

### E44 — force chunked GDN kernel on M3 (REJECTED, 2.1–2.3× slower)

Bypass `CHUNK_MIN_GPU_GEN=17` via `MLX_FORCE_CHUNKED_GDN=1` to test whether
E5+E36 polish (decay_mat + decay_self caches) had closed the gap.

**Result:** chunked is 2.1–2.3× SLOWER than per-step on M3 across 3 pairs
(per-step 624/764/950 ms vs chunked 1416/1772/1997 ms). M5+ gate is
correctly placed; the kernel's `simdgroup_matrix` operations need Neural
Accelerators to be competitive.

### E45 — `DV_PER_TG` sweep for per-step kernel (REJECTED, default optimal)

Parameterized threadgroup-Y via `MLX_GDN_STEP_TG_Y`; swept {1,2,4,8,16} at
1024 prompt and {4,8,16} at 4096. Result: TG_Y=4 (current default) wins at
4096 by 4%; at 1024 all variants within 3.8%. Sweep confirms no headroom.

### E46 — cooperative threadgroup q/k cache (REJECTED, 2–6% slower)

New kernel variant where simdgroup-0 cooperatively loads q[Dk] + k[Dk] for
T_BLOCK=8 consecutive timesteps into threadgroup memory (8 KB total). All
4 simdgroups in the TG then read from TG memory instead of re-issuing
global loads. Two `threadgroup_barrier`s per outer iter.

**Why it failed:** barriers cost more than the saved load instructions.
Apple GPU L1 absorbs the 4-way cross-simdgroup load redundancy near-free,
so eliminating it via TG memory write + barrier is net negative. Three
pairs all agreed: +2%, +2%, +6% slower with E46 ON.

### E47 — 2 v-columns per simdgroup (REJECTED first, then ACCEPTED on rerun)

Catalog D1: register-blocking change to the per-step kernel. Each
simdgroup processes dv_A=2y and dv_B=2y+1, sharing q[Dk] + k[Dk] loads
across both columns. Halves grid-Y; doubles state registers; correctness
bit-exact.

**First attempt rejected (8 pairs split 4/4).** Looked like noise.

**Rerun ACCEPTED (-2.0% at 1024-prompt single-chunk, neutral at 4096).**
After a 1-hour M3 cool-down, sanity benches showed the *true* cold-state
noise floor: ~0.3% within-pair, not the ~3% I'd been seeing. In two
batches of 4 alternating runs (8 total):

| ON     | OFF    |
|--------|--------|
| 572.38 | 584.99 |
| 572.51 | 585.69 |
| 574.25 | 585.26 |
| 582.72 | 585.51 |

Median ON 573.4 ms vs median OFF 585.4 ms = **-2.0%**, with within-pair
noise <0.5 ms vs a 12 ms cross-condition gap. Direction agrees in every
comparison.

At 4096-prompt (1 quick pair, multi-chunk): 2135 ms ON vs 2141 ms OFF =
**-0.25% (neutral)**. The win compresses at long prompts because full-attn
O(T²) SDPA dominates.

### Methodology lesson — bench in cold-state batches of 4

The earlier "compute-bound, not memory-bound" conclusion from rejecting
E47 was **premature**. The real story is bench-thermal:

- **Cold idle 30+ min**: within-pair noise ~0.3-0.5%, runs 1-6 stable.
- **Mid-cycle warming**: noise spikes to ~3-5% as runs 7-10 drift.
- **Saturated thermal**: 10%+ noise, all bets off.

Always bench the first 4 runs of a cold cycle; cool for 20+ min between
batches. With this protocol, sub-1% wins are measurable. With back-to-back
8-pair tests (the protocol I'd used originally), a real -2% signal can hide
as 4/8 ON-faster / 4/8 ON-slower noise.

Register-blocking *does* save real wall time on M3, just less than the
naive instruction-count math would suggest because Apple GPU's L1 absorbs
much of the cross-simdgroup load redundancy. The win is the ~2% that
the L1 doesn't absorb — kernel-launch and dispatch overhead from the
halved grid-Y, plus tighter scheduling.

E46's (TG cache) rejection still stands — it added barriers + TG memory
writes that costed more than the loads saved. E47's win comes from
*reducing kernel-launch / grid count*, not from caching.

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
