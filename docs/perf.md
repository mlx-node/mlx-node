# Performance & profiling

## Profiling

Per-generation profiling is exposed from `@mlx-node/lm`:

```typescript
import { enableProfiling, disableProfiling } from '@mlx-node/lm';

enableProfiling();
// run generations...
disableProfiling();
```

The store lives in `crates/mlx-core/src/profiling.rs` (global `PROFILING_STORE: Mutex<Vec<GenerationProfile>>`, gate via `PROFILING_ENABLED: AtomicBool`). NAPI exports: `setProfilingEnabled`, `isProfilingEnabled`, `getProfilingData`, `resetProfilingData`.

The per-generation profiler (`crates/mlx-core/src/decode_profiler.rs`) records:

- TTFT (`time_to_first_token_ms`)
- Phase breakdown: `forward`, `sample`, `eval_token`, `extract`, `async_eval`
- Memory snapshots before / after each generation

> Note: MLX lazy evaluation means `prefillMs` measures only graph construction (~1 ms). Use `timeToFirstTokenMs` as the real prefill latency indicator.

## Environment variables

### Profiling and tracing

| Var                        | Effect                            |
| -------------------------- | --------------------------------- |
| `MLX_PROFILE_DECODE=1`     | Auto-enables profiling at startup |
| `MLX_NODE_LOG`             | Tracing-level filter              |
| `MLX_INFERENCE_TRACE_FILE` | Path for inference trace dump     |
| `MLX_DEBUG_GEMMA4_DUMP`    | Diagnostic dumps for Gemma4       |

### Compile / decode control

| Var                                                  | Effect                                                   |
| ---------------------------------------------------- | -------------------------------------------------------- |
| `MLX_NO_COMPILE=1`                                   | Disable compiled C++ forward path (Qwen3.5)              |
| `MLX_EVAL_ALL_CACHES=1`                              | Revert to eval-all-caches (default is token-only)        |
| `MLX_QWEN35_NATIVE_KV_WRITE` / `MLX_NATIVE_KV_WRITE` | Toggle native KV-write optimization on Qwen3.5 attention |
| `MLX_WEIGHT_MATERIALIZE_CHUNK_MB`                    | Weight-loading chunk size                                |

### Paged-attention

| Var                                     | Effect                                     |
| --------------------------------------- | ------------------------------------------ |
| `MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL` | Override decode-time `clear_cache` cadence |
| `MLX_PAGED_PREFILL_EVAL_INTERVAL`       | Override prefill `eval` cadence            |
| `MLX_PAGED_PREFILL_CHUNK_SIZE`          | Prefill chunk size                         |
| `MLX_TEST_PAGED`                        | Test-only paged-path toggle                |

### Memory pool

| Var                   | Effect                                   |
| --------------------- | ---------------------------------------- |
| `MLX_CACHE_LIMIT_GB`  | Hard Metal pool ceiling                  |
| `MLX_GPU_HEADROOM_GB` | Headroom term in the auto-sizing formula |

## MTP speculative decoding

Qwen3.5 / Qwen3.6 MTP (Multi-Token Prediction) speculative decoding adds eight
runtime knobs gating individual optimizations across the W6.5–Phase C perf
chain (plus one unconditional warmup hook for verify prewarm). All seven
env vars are read at most once per process and cached; the truthy/falsy
vocabulary is uniform (`1` / `true` / `on` and `0` / `false` / `off`,
case-insensitive, with `trim()`). The adaptive-depth knob is a
TypeScript `ChatConfig` field (not an env var) because it interacts with
the user-set `mtpDepth` and needs per-session resolution.

| Knob                          | Default | Workstream | Direction     | Notes                                                                                                                                                                                      |
| ----------------------------- | ------- | ---------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MLX_MTP_USE_TAPE_REPLAY`     | ON      | W6.6       | opt-OUT       | Set to `0` / `false` / `off` to fall back to the W6 Bug #4 K+1 main-model replay path. Dense only — MoE always uses K+1.                                                                   |
| (eager verify prewarm)        | always  | W6.7       | unconditional | No env var. Once-per-process `atomic<bool>` CAS at model load runs 10 dummy shapes (5 depths × 2 tape variants) to warm caches.                                                            |
| `mtpAdaptiveDepth` (TS field) | ON\*    | W6.8       | per-session   | TS `ChatConfig` field. \* defaults ON when `enableMtp=true` and `mtpDepth` is unset; defaults OFF (pinned) when `mtpDepth` is set explicitly.                                              |
| `MLX_MTP_CHAINED_CYCLES`      | OFF     | W6.5       | opt-IN        | Slower than the default Step-A path at depth ≥ 2 even after the W6.5-resume fix batched the `verify_hidden[K]` slice into the next-cycle `async_eval`. The residual ~18% gap on bf16/M3 Max traces to cross-cycle CPU bookkeeping, not the slice DMA. |
| `MLX_MTP_VERIFY_ASYNC_EVAL`   | OFF     | W6.9       | opt-IN        | Overlaps verify dispatch with the accept loop's CPU-side graph construction. Composes cleanly with all other flags.                                                                        |
| `MLX_MTP_FUSED_DRAFT`         | OFF     | W6.18      | opt-IN        | Fuses D draft steps into one compile()d graph. Currently no measured perf win on qwen3.6-27b-nvfp4-mtp / depth=3 / M3 Max; kept opt-in pending Step-A bypass follow-up where the infrastructure will pay off. Dense only — MoE always uses the per-step draft loop. |
| `MLX_MTP_SPARSE_ACCEPT`       | OFF     | W6.19      | opt-IN        | Batched argmax over D+1 verify positions at T=0 with no penalties; collapses D × full-vocab softmax materializations into one .eval(). Falls back to legacy per-position path at T>0 or when sampling penalties are active. Currently no measured perf win on qwen3.6-27b-nvfp4-mtp / depth=3 / M3 Max; kept opt-in pending hardware/model targets where MLX scheduler exposes the sync cost. |
| `MLX_MTP_BUCKETED_VERIFY`     | ON      | W6.29      | opt-OUT       | Per-bucket compiled verify graphs (`max_kv_len ∈ {256, 512, 1024, 2048, 4096, 8192}` + LEGACY fallback) so SDPA reads a static `[B, Hkv, bucket_kv_len, head_dim]` slice of the writeback cache. Eager prewarm at the prefill-offset bucket; lazy-trace others (~0.5 s per bucket-transition step). Measured at long decode (max_tokens=32768) on qwen3.5-4b / M3 Max: AR +12.0%, MTP +26.1%. No-op at default short prompts where the first bucket already covers the full cache. Set to `0` / `false` / `off` to force the legacy single-trace path. |
| `MLX_MTP_NO_PROMPT_PREFILL`   | OFF     | Phase C    | opt-OUT       | When unset (default), a fresh prefill captures per-prompt-token hiddens and commits the prompt prefix into the persistent MTP committed-history cache so the heads attend it from cycle 1. Set to `1` / `true` / `on` to keep the prefill logits-only — the MTP heads then build history only from decode-produced tokens. Dense only. Skipped automatically on cache-reuse / VLM / delta turns regardless of this knob (the prefill only sees the uncached suffix). |

## Committed-history MTP cache (Phase C)

Phase C replaced the per-cycle MTP draft cache (zeroed every cycle —
heads saw only the in-cycle draft chain) with a **persistent
committed-history cache**: each cycle commits the full `K+2` sequence
`[last_committed, d_0..d_{K-1}, boundary]` into a separate MTP K/V cache
so subsequent cycles' drafts attend the whole committed prefix.
Prompt-prefill seeds that cache from the prompt's hiddens before decode.

The committed-history cache is its **own coordinate space**, decoupled
from the main KV cache (the C++ `begin_cycle` anchors the draft RoPE
offset to `g_mtp_committed_len`, not the main offset). On turns that
skip prompt-prefill (cache-reuse / VLM / delta / `MLX_MTP_NO_PROMPT_PREFILL`
/ prompt < 2 tokens) the cache simply starts empty and fills
contiguously from decode tokens — internally consistent, and
speculative decoding stays verify-correct regardless of draft quality.

Measured on `qwen3.6-27b-nvfp4-mtp` / M3 Max / depth=3 / T=0:

| Path                                  | mean accepted/cycle | per-position acceptance   | MTP/AR decode |
| ------------------------------------- | ------------------- | ------------------------- | ------------- |
| committed-history + prompt-prefill    | 2.15                | `[0.854, 0.715, 0.585]`   | ~1.31×        |
| committed-history, no prompt-prefill  | 1.75                | `[0.812, 0.565, 0.381]`   | ~1.13×        |
| (pre-Phase-C per-cycle cache)         | 1.56                | —                         | < 1×          |

T=0 parity holds **in distribution**: every MTP-emitted token equals
`argmax(verify_logits)`. MTP and AR outputs agree on a contiguous prefix
and then diverge at an isolated argmax near-tie — and that flip can land
*early*. The recurring offset-16 "Autumn is often regarded / described"
flip was diagnosed: at that token AR and the batched verify rank the
**same** top-2 tokens within one bf16 ulp (AR logits 21.500 / 21.375;
verify 21.375 / 21.375), so the verify forward merely tie-breaks to the
other token. One flip then decorrelates all downstream text. This is
benign lossless speculative decoding (vLLM / MTPLX / dflash-mlx all
document it) — **not** a verify-path bug. Because a near-tie can flip at
any offset, `examples/qwen35-mtp-smoke.ts` treats text divergence as
informational only; the blocking correctness gate is acceptance health.

### Draft depth on M3 Max and M5 Max

The verify forward is one full 27B forward over `T = depth+1` tokens and
is ~58–62 % of the MTP cycle; its cost grows with depth while later draft
slots accept progressively less. Measured on
`qwen3.6-27b-nvfp4-mtp-oproj8` / T=0 / 256 tokens (depths 1–3 a
same-session sequential A/B; absolute ratios are thermal-sensitive
~±10 %, so the cross-depth ordering is the signal):

| Depth    | M3 Max ratio | M5 Max ratio | M5 K̄ | M5 per-position acceptance |
| -------- | ------------ | ------------ | ----- | --------------------------- |
| 1        | **1.14×**    | **1.15×**    | 0.87  | `[0.865]`                   |
| 2        | 1.12×        | 1.15×        | 1.42  | `[0.811, 0.608]`            |
| 3        | 0.93×        | 1.04×        | 1.98  | `[0.828, 0.656, 0.508]`     |
| adaptive | 1.07×        | 1.12–1.13×   | 1.00  | `[0.86, 0.54, 0.40, …]`     |

**Depth 1 is optimal on both M3 Max and M5 Max** — the 3rd draft slot's
~50 % acceptance still does not pay for the wider, slower verify forward.
On M5 Max depth 3 climbs from a net regression (0.93×) to marginally
positive (1.04×) — the Neural Accelerator does shave a little off the
wider verify — but it still loses to depth 1. The W6.8 adaptive policy
underperforms the depth-1 pin on both hosts. For this hardware/model
class, pin `mtpDepth: 1`.

**M5 Neural Accelerator does NOT widen the MTP/AR gap.** The headline
MTP/AR ratio at the optimal depth is essentially hardware-invariant: 1.14×
on M3 Max, 1.15× on M5 Max. The reason is symmetry — stock MLX `qmv` and
the AR forward path both benefit equally from the NA, so the MTP cycle's
verify forward gets faster in the same ratio as the AR baseline it is
measured against. See "Verify-kernel investigation" below for the M5 Max
microbench result that pins this down.

### Verify-kernel investigation — negative results

The verify forward is the bottleneck, and four attempts to make its
small-M (`M = depth+1`) quantized matmuls cheaper all **failed**:

| Attempt                                       | Result on M3 Max                                                                                              | Result on M5 Max                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Batched GEMM (`qmm` / `qmm_t_splitk`) at M=4  | ~38 % slower than `qmv`; split-K low-precision accumulation also degrades acceptance → ratio collapses to ~0.75× | (not re-tested — split-K accuracy regression is hardware-independent) |
| `multi3` multi-row `qmv` (W6.30 port)         | 0.94× geomean vs stock `qmv` — slower even in its native affine-4-bit M=3 envelope                              | **0.94× geomean** — identical to M3; the M5 Neural Accelerator helps stock `qmv` symmetrically (`oxnode examples/qmv-multi3-microbench.ts`, 2026-05-26) |
| `FUSED_DRAFT` / `SPARSE_ACCEPT` / `ASYNC_EVAL` | zero measured win (W6.18 / W6.19)                                                                              | (not re-tested — sync-collapse knobs are CPU-side; hardware-invariant) |
| End-to-end MTP/AR ratio at optimal depth      | 1.14× (depth 1)                                                                                               | **1.15× (depth 1)** — Neural Accelerator does not widen the gap     |

Stock MLX's small-M `qmv` is already near-optimal on both M3 and M5 Max:
at these shapes the verify is not dominated by per-row weight re-reads,
so reading weights once (the `qmm` / `multi3` approach) buys nothing and
the extra machinery costs more. Crucially, the M5 Neural Accelerator
benefits stock `qmv` and `multi3` in the same proportion, so the
microbench ratio is invariant — and the end-to-end MTP/AR ratio is
invariant too, because the AR baseline gets the same NA uplift as the
MTP verify forward.

**Why MTPLX reaches ~2.2× and this does not — it is NOT the hardware.**
The earlier write-up here hypothesised the M5 Neural Accelerator as the
sole differentiator. Direct measurement on M5 Max (2026-05-26) refutes
that: stock MLX on M5 Max gives the same 1.15× ceiling as M3 Max, and
`multi3` still loses 0.94× microbench-wise. The remaining candidates for
MTPLX's reported speedup are (a) their **private MLX fork** with retuned
small-M `qmv` kernels stock MLX does not have, (b) a different
quantization recipe or verify path, or (c) a different model size. None
of these is the Neural Accelerator alone. (dflash-mlx's 2.95–4.4× is a
*different* architecture — a separately-trained block-diffusion drafter —
and is out of scope for native MTP heads here.)

**Native-heads MTP/AR ceiling ≈ 1.1–1.15× on both M3 Max and M5 Max**
(depth 1). Reaching the 1.6–2.2× plan target on native heads is not
unlocked by upgrading to M5-class hardware; it would require either
custom verify-path kernels (MTPLX-style private fork) or the dflash
separately-trained-drafter architecture.

**Long-context behaviour.** The verify forward's attention cost scales
with context length, eroding the speculative advantage on long prompts:
on a ~1k-token prompt the MTP/AR ratio drops toward parity even though
per-cycle acceptance stays healthy. A future adaptive context-length
guard could fall back to plain AR decode once the prompt length crosses
the break-even point.

Interactions:

- `MLX_MTP_USE_TAPE_REPLAY=0` is safe to combine with all other flags.
- `MLX_MTP_VERIFY_ASYNC_EVAL=1` composes cleanly with every other knob;
  parity holds byte-exact at `T=0` across all combinations on qwen3.5-4b.
- Setting `mtpDepth` explicitly disables adaptive depth by default;
  pass `mtpAdaptiveDepth: true` alongside to keep adaptation enabled with
  `mtpDepth` as the initial seed.

Naming notes:

- The W6.9 flag was briefly drafted as `MLX_MTP_PREFETCH`. The current
  name reflects the actual mechanism (intra-cycle overlap with CPU-side
  graph construction, not cross-cycle draft staging). The literal
  "stash next-cycle draft handle, drain at cycle start" prefetch lives
  in a follow-up scoped to `MLX_MTP_CHAINED_CYCLES=1`.
- `MLX_MTP_CHAINED_CYCLES` (W6.5) and `MLX_MTP_FUSED_DRAFT` (W6.18)
  refer to independent mechanisms despite both touching the draft path:
  the former exports `verify_hidden[K]` CROSS-CYCLE; the latter fuses
  D draft steps WITHIN a cycle.

Cross-references:

- TS field JSDoc: `enableMtp` / `mtpDepth` / `mtpAdaptiveDepth` on
  `ChatSession.send` in `packages/lm/src/chat-session.ts`.
- Source of truth (env-var readers + inventory table):
  `crates/mlx-core/src/models/qwen3_5/chat_common.rs` (`mtp_use_tape_replay`,
  `mtp_chained_cycles_enabled`, `mtp_verify_async_eval`,
  `mtp_fused_draft_enabled`, `mtp_sparse_accept_enabled`,
  `mtp_no_prompt_prefill`). The W6.29
  bucket dispatcher opt-out lives in C++ (`bucketed_verify_disabled` in
  `crates/mlx-sys/src/mlx_qwen35.cpp`) because the bucket table and
  compile cache are C++-side state.
- Phase C committed-history MTP cache: C++ policy in
  `crates/mlx-sys/src/mlx_qwen35_mtp_compiled.cpp`
  (`mlx_qwen35_mtp_compiled_begin_cycle` / `_commit`, `g_mtp_committed_len`);
  prompt-prefill seed in `crates/mlx-core/src/models/qwen3_5/model.rs`
  (`chunked_prefill_with_hidden`, `prefill_mtp_commit`).
- W6.8 adaptive-depth policy:
  `crates/mlx-core/src/models/qwen3_5/adaptive_depth.rs`.
- Parity gate harness: `examples/qwen35-mtp-smoke.ts`.

## Key performance patterns

- `token.eval()` immediately after sampling — without it MLX builds an unbounded lazy graph.
- `synchronize_and_clear_cache()` every 256 steps — prevents memory accumulation during long generations.
- Dtype-aware scalar ops — any `f32` scalar in a binary op with bf16 promotes the **entire** result to f32.
- Token-only eval — caches materialize through the dependency graph; no need to eval every cache tensor explicitly.
- For bf16 / f16 data extraction: use `to_uint16_native()` instead of round-tripping through f32.

## GPU architecture detection

`mlx_gpu_architecture_gen()` (FFI in `crates/mlx-sys/src/lib.rs`) returns a generation number:

| Chip | Gen |
| ---- | --- |
| M1   | 13  |
| M2   | 14  |
| M3   | 15  |
| M4   | 16  |
| M5   | 17  |

The Qwen3.5 chunked GDN prefill kernel is gated on M5+ (`CHUNK_MIN_GPU_GEN = 17`) with a 64-token minimum sequence length — on M5, Neural Accelerators make `simdgroup_matrix` ops roughly 4× faster than the per-step kernel; on M1–M4 the per-step kernel wins.

## Quantization

| Scheme       | How it's invoked                                                                                                                                         |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4-bit affine | `mlx_quantized_matmul` (mode `affine`, configurable group size and bits)                                                                                 |
| MXFP8        | `mlx_gather_qmm` with `mode="mxfp8"` (used for MoE expert routing); returns `[quantized, scales]`                                                        |
| FP8 E4M3     | `mlx_dequantize` — dequant **before** expert stacking; no re-quantization after stacking                                                                 |
| FP8 KV cache | Paged-adapter only — `KVCacheDType::Fp8` with per-layer scale management via `KvScaleManager`. FP8 KV is intentionally rejected by the flat-path attach. |

### Recipes

`crates/mlx-core/src/convert.rs` supports:

- mlx-lm-style mixed-bit: `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`
- `qwen3_5` — Qwen3.5-tuned recipe
- `unsloth` — requires imatrix calibration

AWQ-style imatrix pre-scaling is supported for improved low-bit quality.

`quant_predicate` defaults: router gates → 8-bit; everything else → 4-bit.
