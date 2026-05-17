# Qwen3.5-4B prefill arc — handoff doc

Single-file handoff for whoever picks up Qwen3.5 prefill perf next.
Replaces the ephemeral autoresearch loop driver (`autoresearch.{md,sh,jsonl,config.json,ideas.md}`,
`scripts/bench-gdn-prefill.ts`, `experiments/worklog.md`) that ran the
arc. The companion `experiments/E53-text-prefill-compile-scope.md`
covers the one piece of unfinished work (uncalled C++ scaffolding at
commit `a5a2e7c`).

## TL;DR

Composed shipping state on this branch: **-16.4% same-binary A/B at
1024-prompt single-chunk prefill** on `Qwen3.5-4B-mlx` (Apple Silicon
M3 Max), correctness-gated by an 8-token greedy fingerprint at
temperature=0 (max_abs_diff=0 throughout).

Two pushes are bundled:

- **Original arc — -13.7%**: E37+E38 (last-token slicing) + E28+E31
  (chunk=1024 + async eval) + E39+E40 (MLP gate+up stack + compiled
  SwiGLU) + E5+E36 (chunked-kernel polish, M5+ gated).
- **Post-arc — additional -2.7%**: E47 (2-vcol GDN per-step
  register-blocking) + E51 (GDN in_proj stack). E48 (4-vcol) is
  available opt-in at +0.5% on top.

## Hardware context

- **Host**: Apple M3 Max (40 GPU cores, gen 15, Metal 4)
- **Important**: the chunked GDN kernel (`gated_delta_chunked`) is
  gated to `CHUNK_MIN_GPU_GEN >= 17` (M5+). On M1–M4 the per-step
  kernel beats it (verified empirically — see E44 rejection). So as
  shipped, M3 prefill runs through `gated_delta_kernel` (the per-step
  kernel — same code path as decode). Most M3-deployable optimizations
  target that kernel.
- Single GPU, runs are serial. The MLX submodule at
  `crates/mlx-sys/mlx` is pinned; don't bump for perf work.

## Shipped wins (composable, all independently revertable)

| Tag | Where | Toggle | Mechanism |
|---|---|---|---|
| **E37** | `qwen3_5/model.rs::forward_inner` | `MLX_DISABLE_E37_LAST_TOKEN_SLICE=1` | Slice `h` to last token before `final_norm + lm_head`. Skips ~T-1 rows of a `[B,T,2560] @ [2560,vocab]` matmul. |
| **E38** | `qwen3_5/decoder_layer.rs::forward_with_optional_last_slice` | (same as E37) | Last layer slices `h` to last token AFTER attention residual, BEFORE post-attention norm + MLP. KV writes preserved on full T. |
| **E28+E31** | `qwen3_5/model.rs::chunked_prefill` | `MLX_PREFILL_SYNC_BETWEEN_CHUNKS=1` | `PREFILL_STEP_SIZE` 2048→1024 + `async_eval_layer_caches` between chunks (vs the prior sync `eval`). |
| **E39** | `transformer/mlp.rs::finalize_gate_up` + `mlx_fused_ops.cpp::mlx_swiglu_mlp_forward_stacked` | `MLX_DISABLE_E39_STACKED_MLP=1` | Pre-stack `[w_gate; w_up]^T` at model load. One matmul instead of two; per-call transposes baked in. |
| **E40** | `mlx_fused_ops.cpp` | (same as E39) | In the E39 stacked path, use `qwen35_common::swiglu()` (mlx::core::compile-cached fused sigmoid·gate·up) instead of inline ops. |
| **E5+E36** | `crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc` | (no toggle — kernel-internal) | Threadgroup `decay_mat[BT*BT]` and `decay_self[BT]` precompute. **M5+ only** (`CHUNK_MIN_GPU_GEN=17`); correctness was verified on M3 by temporarily lowering the gate. |
| **E47** | `crates/mlx-sys/src/metal/gated_delta_step_2vcol.metal.inc` + `mlx_gated_delta.cpp` dispatch | `MLX_DISABLE_E47_GDN_2VCOL=1` | Per-step GDN kernel: each simdgroup handles 2 v-cols (dv_A=2y, dv_B=2y+1), sharing q[Dk]+k[Dk] loads. Grid-Y halves to Dv/2. -2.0% at 1024 single-chunk; neutral at 4096+. |
| **E48** *(opt-in)* | `gated_delta_step_4vcol.metal.inc` + dispatch | `MLX_ENABLE_E48_GDN_4VCOL=1` | 4-vcol variant of E47. +0.5% on top of E47 for large-Dv models. Default-off because the grid-Y reduction could regress on smaller-Dv shapes. |
| **E51** | `gated_delta_net.rs::finalize_in_proj` + `persistence.rs` | `MLX_DISABLE_E51_STACKED_GDN_IN_PROJ=1` | Load-time stack of `in_proj_qkvz` + `in_proj_ba` into `[hidden, qkvz_dim+ba_dim]^T`. Forward does one matmul + two axis-2 slices instead of two matmuls. -0.44% across 24 GDN layers. No-op for quantized variants. |
| **E55** | `qwen3_5/model.rs::PREFILL_STEP_SIZE` | (no toggle — const) | `PREFILL_STEP_SIZE` 1024 → 2048 to match mlx-lm's default and close the long-context gap. Halves chunk count at multi-chunk prompts. At T ≤ 2048 the const is irrelevant (single `remaining` branch). −5-7% at 20k. |

To verify the composed delta, run two same-binary passes with all
toggles flipped to legacy vs all on. The exact env-var combo for
"legacy" (re-creates the pre-arc baseline):

```bash
MLX_DISABLE_E37_LAST_TOKEN_SLICE=1 \
MLX_DISABLE_E39_STACKED_MLP=1 \
MLX_PREFILL_SYNC_BETWEEN_CHUNKS=1 \
MLX_DISABLE_E47_GDN_2VCOL=1 \
MLX_DISABLE_E51_STACKED_GDN_IN_PROJ=1 \
<bench command>
```

E55 (chunk=2048) is not env-toggle revertable since it's a const; flip
to 1024 in `model.rs` for an A/B if needed.

`E5+E36` chunked-kernel polish is M5+ gated so it doesn't affect the
M3 bench (no toggle).

## Scaling

Multi-chunk perf was measured during the original arc; not re-measured
after E47+E51.

| Prompt tokens | Chunks | best | legacy | Δ |
|---:|---:|---:|---:|---:|
| 1024 (single chunk) | 1 | 572 ms | 682 ms | **-16.4%** (with E47+E51) |
| 2048 | 2 | 1410 ms | 1537 ms | -8.3% |
| 4096 | 3-4 | 2593 ms | 2707 ms | -4.2% |
| 8192 | 7 | 5032 ms | 5040 ms | -0.15% |

Why the falloff at long prompts: E37+E38 save fixed work (one lm_head
+ one last-layer MLP slice per chunk). E39+E40 save kernel-launch
overhead on the 32-per-chunk MLPs. E47 saves per-timestep work in the
GDN kernel — that DOES scale with T, but the full-attn SDPA's
`O(T · T_ctx)` cost takes over at 8192+. None of the wins target SDPA.

## What didn't ship (and why)

Recorded so the next iteration doesn't re-discover the same dead ends.

| Tag | Tried | Verdict | Why |
|---|---|---|---|
| **E41** | Load-time Q+K+V projection stack (MLP-style for attention) | REJECT, +2% regression at 1024 | MLX inserted hidden contiguous-materialization copies (~330 MB/chunk) when the strided slice got reshaped to per-head. The fix (C++ fused slice+copy helper) was deferred — not worth doubling implementation effort for ~0.3% theoretical gain. |
| **E42** | Last-layer attention internal slice (queries→1 before SDPA) | REJECT, mixed | SDPA dispatch regimes differ between prefill (large T_q) and decode (T_q=1, small T_ctx). A "1 query, large T_ctx" call falls between — slower at 1024–2048, marginally faster at 4096+. Wrong tradeoff for the 1024 headline. |
| **E43** | Skip per-chunk `clear_cache()` | REJECT, within noise | The clear cost is small vs chunk compute; skipping just raises peak memory 0.07–0.2 GB without latency gain. |
| **E44** | Force chunked GDN kernel on M3 (bypass M5+ gate) | REJECT, 2.1–2.3× slower | The chunked kernel's `simdgroup_matrix` needs Neural Accelerators (M5+). On M3 it's bandwidth-bound and loses to per-step badly. M5+ gate stays. |
| **E45** | `DV_PER_TG` / `TG_Y` sweep for per-step kernel | REJECT, default optimal | TG_Y=4 wins at 4096 by 4%; at 1024 all variants within 3.8% (noise). Threadgroup geometry is in its sweet spot. |
| **E46** | Cooperative TG q/k cache across T_BLOCK=8 timesteps | REJECT, 2–6% slower | Threadgroup memory traffic + barrier cost outweighs L1 reuse. Apple GPU's L1 was absorbing the redundant loads near-free anyway. |
| **E49** | `TG_Y` env-var sweep on E47/E48 variants | REJECT, all within noise | TG_Y=4 stays optimal for both 2-vcol and 4-vcol kernels. Reverted the env-var addition to keep dispatch minimal. |
| **E50** | 8 v-cols per simdgroup (extend register-blocking past E48) | REJECT, ties E48 within 0.07% | Diminishing returns past 4 v-cols on M3: grid Y drops to Dv/8=16 → only 16 total TGs at B=1,Hv=4, occupancy-bound. |
| **E52** *(investigated, not implemented)* | Q+K+V stack revisited with reshape-then-slice | REJECT | Theoretical max gain ~0.3% (8 attn layers vs E51's 24 GDN); E41 downside risk (~2%) is precedented. Asymmetric risk profile not worth the implementation cost. |
| **E54** *(methodology)* | Re-validate E48 via cross-process bench | INCONCLUSIVE | Cross-process variance (~15.8% between two E47-default runs) dwarfs sub-1% kernel deltas. The earlier 0.5% E48 in-process measurement remains the best estimate. |

## Methodology

### Bench protocol

The bench measured **time-to-first-delta** of `session.sendStream(prompt, { config: { maxNewTokens: 1, temperature: 0 } })` on `Qwen3.5-4B-mlx`. Prefill dominates (decode of 1 token is ~10 ms on top of ~570 ms prefill). Correctness was gated by capturing an 8-token greedy fingerprint at temperature=0 in a separate run and comparing to a baseline.

The actual script lived at `scripts/bench-gdn-prefill.ts` (now deleted; see PR history if needed). The driver was `autoresearch.sh` reading `autoresearch.config.json` (also deleted). For a future researcher, the bench is a ~200-line TS file calling `loadModel` + a timed loop around `sendStream` — easy to recreate.

### Cold-state vs cross-process noise

This is the **single most important methodology lesson** from the arc:

- **In-process cold-state pairs** (model loaded once, env vars toggled between timed passes within ONE node process, ~4–6 measurements before thermal saturation): ~**0.3% noise floor**. This is what every shipped sub-1% win (E47, E48, E51) was measured against.
- **Cross-process A/B** (two separate `./autoresearch.sh` invocations): ~**15% noise floor**. The first cold launch is slowest; subsequent runs benefit from OS file cache + GPU thermal/cache state. Same E47-default config measured 694.77 ms in one process and 599.93 ms in another — same code.

The autoresearch.md formal "Adopt at ≥3%" threshold is calibrated for cross-process measurement variance. In-process cold-state lets you detect smaller wins reliably, but requires building the A/B harness yourself (the bench script that ran the arc was single-config; cold-state pairs were collected via a separate small harness invoked manually in the same process).

For ANY future sub-3% experiment, the in-process pattern is mandatory. The pattern looks like:

```ts
const model = await loadModel(modelPath);
async function timeOne(): Promise<number> {
  const session = new ChatSession(model, { system: '...' });
  const t0 = process.hrtime.bigint();
  const stream = session.sendStream(prompt, { config: { maxNewTokens: 1, temperature: 0 } });
  let firstDelta: bigint | null = null;
  for await (const ev of stream) {
    if (firstDelta === null) firstDelta = process.hrtime.bigint();
    if (ev.done) break;
  }
  return Number(firstDelta! - t0) / 1_000_000;
}
// 2 warmup runs (any config) — ignore.
// Then ~4 alternating cold-state pairs: A/B/A/B/A/B/A/B
// Within-pair delta is the signal; across pairs is the noise estimate.
// Once medians are ~5%+ above the minimum, thermal saturation has set in
// and later runs are warm-state — discard them.
```

### Decision rules (formal)

Inherited from `autoresearch.md`:

- **Adopt** when `prefill_ms_median` improves by ≥3% over current best AND `max_abs_diff == 0`. Lock change, update baseline.
- **Discard** otherwise: revert, log the hypothesis, move on.
- **Investigate** when correctness fails or the measurement looks suspicious — capture extra signals (per-run latencies, Metal counters via `MTL_CAPTURE_ENABLED=1`) before declaring win/loss.

E47 (-2.0%) and E51 (-0.44%) shipped as small but consistent wins, formally below the ≥3% threshold but validated rigorously via in-process cold-state pairs. The Adopt rule is a guardrail against cross-process noise; in-process measurement with replicated direction across ≥3 pairs counts as "adopt-quality."

## Ideas backlog (FlashQLA-direction, untried items only)

From the original FlashQLA idea catalogue, with M3-feasibility annotations:

| Idea | Source | M3-feasible? | Notes |
|---|---|---|---|
| **A1** Cache `exp(gcum[i])` in TG memory (chunked kernel) | FlashQLA `fused_fwd.py` | M5+ only | The chunked kernel is gated. Worth doing if/when chunked beats per-step on M3, or directly on M5+ silicon. |
| **A2** Fold decay into k_chunk at load time | FlashQLA `fused_fwd.py` | M5+ only | Depends on A1; numerical stability concern (sqrt of small numbers). |
| **A3** Reduce Phase-4 barriers | (audit our chunked kernel) | M5+ only | Apple GPU barriers are expensive; might save µs. |
| **A4** `1/decay` once per chunk for state update | derived from A1 | M5+ only | Extends A1. |
| **B1** WY-transform precomputation (kkt_solve analog) | FlashQLA `kkt_solve.py` | M5+ only, high payoff/risk | Splits the chunked kernel; eliminates O(BT) sequential dep. Parity test critical. |
| **B2** Separate `prepare_h` kernel | FlashQLA `prepare_h.py` | M5+ only | More global memory traffic vs more parallelism; probably bandwidth-bound loss on Apple, but try last. |
| **B3** Gate-driven early termination | FlashQLA CP module | M3-feasible, long-context wins | Skip state-update terms when `gcum[i] − gcum[end] < −10` (contribution below bf16 epsilon). Most impact at long-context prefill, less at 1024-prompt. |
| **C1** `BT = 64` | FlashQLA chunk size | M5+ only | Doubles arithmetic intensity; needs 32 KB threadgroup memory (tight on M5+). |
| **C2** `DV_PER_TG` sweep (chunked kernel) | tile tuning | M5+ only | 1 LOC + bench. |
| **C3** Grid-layout sweep along S dim | tile tuning | M5+ only | Same insight as B3 expressed as launch parallelism. |
| **C4** Drop M5+ gate after A/B/C wins | gate revisit | depends | Only after chunked beats per-step on M3. |
| **D2** Pre-reduce mask for prefill (masked per-step variants) | classic | M3-feasible | Replace per-timestep mask load with single `mask_len` int. Only useful for masked variants; the 1024-prompt bench uses unmasked. |

## The one unfinished piece — E53

**Scope**: C++ text-prefill compile-cache port. Expected 1–3% at 1024-prompt single-chunk, multi-hour scope. See `experiments/E53-text-prefill-compile-scope.md` for the full plan.

**Status on this branch**:

- Scaffolding landed at commit `a5a2e7c`: `mlx_qwen35_text_prefill.cpp` (FFI entry point) + `attn_prefill_fn_scalar` helper in `mlx_qwen35_common.h` + FFI declarations in `mlx-sys/src/lib.rs`. Builds clean, clippy clean, fmt clean.
- **No caller wired in yet.** The Rust dispatch (in `forward_inner` or one of the `chunked_prefill` call sites) is the next step. Cache transfer back to Rust `Qwen3_5LayerCache` slots (or directly to the compiled-decode globals via `mlx_qwen35_compiled_init_from_prefill`) is the hairy part.
- The VLM path (`mlx_qwen35_vlm_prefill` in `mlx_qwen35_vlm.cpp`) is the working template — text prefill is the same minus M-RoPE / mrope_section / rope_deltas / inputs_embeds-merging.

**Don't start E53 in a tail-end loop iteration.** Estimate is ~4 h with care; the cache lifecycle is where things break if rushed. The bench's correctness gate (max_abs_diff=0 on 8-token fingerprint) is non-negotiable, so decode must work correctly after the C++ prefill — meaning cache transfer cannot be skipped.

## File-level pointers

Where each shipped piece lives in the tree (post-cleanup):

- Per-step GDN kernels: `crates/mlx-sys/src/metal/gated_delta_step{,_2vcol,_4vcol,_vec,_mask,_vec_mask}.metal.inc`
- Chunked GDN kernel: `crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc`
- GDN kernel dispatch (variant selection + grid/threadgroup config): `crates/mlx-sys/src/mlx_gated_delta.cpp` lines ~155–180
- GDN forward (Rust): `crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs`, including `finalize_in_proj()` for E51
- MLP gate+up stack (Rust): `crates/mlx-core/src/transformer/mlp.rs::finalize_gate_up`
- MLP stacked forward (C++): `crates/mlx-sys/src/mlx_fused_ops.cpp::mlx_swiglu_mlp_forward_stacked`
- Layer forward + last-token slice: `crates/mlx-core/src/models/qwen3_5/decoder_layer.rs::forward_with_optional_last_slice`
- Top-level forward + chunked prefill driver: `crates/mlx-core/src/models/qwen3_5/model.rs::{forward_inner, chunked_prefill}`
- E53 scaffolding: `crates/mlx-sys/src/mlx_qwen35_text_prefill.cpp`, `mlx_qwen35_common.h::attn_prefill_fn_scalar`

## Out-of-scope for any continuation

- Modifying the MLX submodule at `crates/mlx-sys/mlx` — pinned.
- Touching other models, training, server, tokenizer code — one model, one feature.
- Read-only checkpoint: `.cache/models/Qwen3.5-4B-mlx`.
- M3-side optimizations that depend on Neural Accelerators (`simdgroup_matrix`) — those go in the chunked kernel, M5+ gated.
