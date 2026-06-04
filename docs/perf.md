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
| `MLX_GDN_KERNEL=perstep\|chunked`                    | Force GDN recurrence kernel (default per-step on all archs; `chunked` is A/B-only and changes generated tokens by 1–2 bf16 ULP → different greedy continuation on some long prompts) |

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

| Knob                          | Default | Workstream | Direction     | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------- | ------- | ---------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MLX_MTP_USE_TAPE_REPLAY`     | ON      | W6.6       | opt-OUT       | Set to `0` / `false` / `off` to fall back to the W6 Bug #4 K+1 main-model replay path. Dense only — MoE always uses K+1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| (eager verify prewarm)        | always  | W6.7       | unconditional | No env var. Once-per-process `atomic<bool>` CAS at model load runs 10 dummy shapes (5 depths × 2 tape variants) to warm caches.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `mtpAdaptiveDepth` (TS field) | ON\*    | W6.8       | per-session   | TS `ChatConfig` field. \* defaults ON when `enableMtp=true` and `mtpDepth` is unset; defaults OFF (pinned) when `mtpDepth` is set explicitly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `MLX_MTP_CHAINED_CYCLES`      | M5+: ON / M1–M4: OFF | W6.5       | config-gated  | Each cycle's `verify_hidden[K]` slice seeds the next cycle's first MTP draft, skipping the Step-A target forward on full-accept cycles (slice batched into the next-cycle `async_eval`) — exactly MTPLX's 1-forward-per-cycle design. **Default ON on M5+ (GPU arch gen ≥ 17), default OFF on M1–M4 (gen 13–16)**; `MLX_MTP_CHAINED_CYCLES=0`/`false`/`off` forces OFF even on M5+, `=1`/`true`/`on` forces ON even on M1–M4. **CONFIG-DEPENDENT** (see "Same-checkpoint engine head-to-head"): on M3 Max / bf16 / nvfp4 / hard prompt a controlled A/B found it neutral@depth1 / a regression@depth3 (Step-A only ~14% of cycle there + an unresolved lazy-slice acceptance regression); but on **M5 Max / affine-int4** Step-A is ~37% of the cycle and removing it wins on BOTH a naive cold A/B (**≈1.17×, 45.0→52.8 tok/s**) and the controlled harness (**+23.7pp ratio, 1.116 vs 0.879, beyond noise; acceptance also up 1.30 vs 0.91**) — the biggest structural lever toward MTPLX parity. M5+ default ON is measured net-positive (affine +16% / nvfp4 byte-identical to AR); M1–M4 stays default OFF pending the lazy-slice eval-scheduling fix. |
| `MLX_MTP_VERIFY_ASYNC_EVAL`   | ON      | W6.9       | opt-OUT       | Overlaps verify dispatch with the accept loop's CPU-side graph construction via `async_eval((verify_logits, verify_hiddens))`; the first downstream `.eval()` syncs (semantic analog of MTPLX's `LAZY_VERIFY_LOGITS`). Default ON since Phase 3 (2026-05-26): M5 Max measured +5–6% ratio uplift at depth 3 on `qwen3.6-27b-nvfp4-mtp-oproj8` with byte-identical acceptance. Set to `0` / `false` / `off` to revert to the synchronous barrier. Composes cleanly with all other flags.                                                                                                                                                                                                                   |
| `MLX_MTP_SPARSE_ACCEPT`       | OFF     | W6.19      | opt-IN        | Batched argmax over D+1 verify positions at T=0 with no penalties; collapses D × full-vocab softmax materializations into one .eval(). Falls back to legacy per-position path at T>0 or when sampling penalties are active. Currently no measured perf win on qwen3.6-27b-nvfp4-mtp / depth=3 / M3 Max; kept opt-in pending hardware/model targets where MLX scheduler exposes the sync cost.                                                                                                                                                                                                                                                                                                             |
| `MLX_MTP_BUCKETED_VERIFY`     | ON      | W6.29      | opt-OUT       | Per-bucket compiled verify graphs (`max_kv_len ∈ {256, 512, 1024, 2048, 4096, 8192}` + LEGACY fallback) so SDPA reads a static `[B, Hkv, bucket_kv_len, head_dim]` slice of the writeback cache. Eager prewarm at the prefill-offset bucket; lazy-trace others (~0.5 s per bucket-transition step). Measured at long decode (max_tokens=32768) on qwen3.5-4b / M3 Max: AR +12.0%, MTP +26.1%. No-op at default short prompts where the first bucket already covers the full cache. Set to `0` / `false` / `off` to force the legacy single-trace path.                                                                                                                                                    |
| `MLX_MTP_NO_PROMPT_PREFILL`   | OFF     | Phase C    | opt-OUT       | When unset (default), a fresh prefill captures per-prompt-token hiddens and commits the prompt prefix into the persistent MTP committed-history cache so the heads attend it from cycle 1. Set to `1` / `true` / `on` to keep the prefill logits-only — the MTP heads then build history only from decode-produced tokens. Dense only. Skipped automatically on cache-reuse / VLM / delta turns regardless of this knob (the prefill only sees the uncached suffix).                                                                                                                                                                                                                                      |

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

Measured on `qwen3.6-27b-nvfp4-mtp` / M3 Max / depth=3 / T=0, on a
**moderately-predictable mixed prompt** (see the prompt-dependence note
below — these absolute numbers are meaningless without the prompt):

| Path                                 | mean accepted/cycle | per-position acceptance | MTP/AR decode |
| ------------------------------------ | ------------------- | ----------------------- | ------------- |
| committed-history + prompt-prefill   | 2.15                | `[0.854, 0.715, 0.585]` | ~1.31×        |
| committed-history, no prompt-prefill | 1.75                | `[0.812, 0.565, 0.381]` | ~1.13×        |
| (pre-Phase-C per-cycle cache)        | 1.56                | —                       | < 1×          |

> **Acceptance is dominated by prompt predictability, not by the
> checkpoint.** The numbers above are NOT a fixed property of the
> checkpoint — on the SAME `qwen3.6-27b-nvfp4-mtp` / depth=3 / T=0,
> committed-history + prompt-prefill acceptance ranges from **1.44/cycle**
> `[0.735, 0.471, 0.235]` on a novel three-paragraph prose essay (the
> `examples/qwen35-mtp-smoke.ts` default prompt — a deliberately HARD
> case) up to **2.76–3.00/cycle** (`[0.96–1.0, 0.92–1.0, 0.92–1.0]`, i.e.
> the depth-3 ceiling) on predictable text (counting, lists, recitation,
> repetition). MTP/AR decode speedup tracks acceptance directly: ~1.06×
> on the prose essay, **1.46×** (AR 20.8 → MTP 30.3 tok/s) on a counting
> prompt. So when comparing against another engine's headline number
> (e.g. MTPLX's ~2.24×, measured on its own unpublished "recorded" prompt
> at T=0.6), the prompt and sampler dominate — a single acceptance figure
> is only meaningful alongside the exact prompt that produced it.
> Generation length does NOT move acceptance (essay is flat ~1.44→1.30
> from 120→400 tokens). Verified 2026-05-30.

T=0 parity holds **in distribution**: every MTP-emitted token equals
`argmax(verify_logits)`. MTP and AR outputs agree on a contiguous prefix
and then diverge at an isolated argmax near-tie — and that flip can land
_early_. The recurring offset-16 "Autumn is often regarded / described"
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
| -------- | ------------ | ------------ | ---- | -------------------------- |
| 1        | **1.14×**    | **1.15×**    | 0.87 | `[0.865]`                  |
| 2        | 1.12×        | 1.15×        | 1.42 | `[0.811, 0.608]`           |
| 3        | 0.93×        | 1.04×        | 1.98 | `[0.828, 0.656, 0.508]`    |
| adaptive | 1.07×        | 1.12–1.13×   | 1.00 | `[0.86, 0.54, 0.40, …]`    |

> **Note (chained cycles is CONFIG-DEPENDENT — default ON on M5+ (gen ≥ 17),
> default OFF on M1–M4; see "Same-checkpoint engine head-to-head" below).
> `MLX_MTP_CHAINED_CYCLES=0/1` overrides either direction.**
> This table was measured **with** the Step-A target forward present every
> cycle (`MLX_MTP_CHAINED_CYCLES=OFF`). Chained cycles skips that forward on
> full-accept cycles, seeding the next draft from `verify_hidden[K]` — exactly
> MTPLX's design (the engine that the same-checkpoint head-to-head below shows
> is 1.40× faster precisely because it never runs Step-A).
>
> The win/loss depends on how large Step-A's share of the cycle is and whether
> the chained-hidden's lazy slice stalls the pipeline:
>
> - **M3 Max / bf16 / nvfp4 / hard prose prompt (the original verdict):** a
>   **controlled** A/B (`examples/qwen35-mtp-controlled-verdict.ts`: warmup +
>   cooldown + alternating-order interleaved pairs, 6 repeats/cell, T=0;
>   cross-config AR drift **1.0%** ⇒ fair, self-normalized) found it **not a
>   win** — depth 1 **1.11× ON vs 1.08× OFF** (inconclusive, within ±11pp
>   noise); depth 3 **0.90× ON vs 0.96× OFF** (a regression), driven by an
>   acceptance drop (meanAccepted 1.28 vs 1.43) plus a lazy-eval scheduling
>   stall on the chained-hidden slice. Here Step-A is a smaller fraction of
>   the cycle, so removing it doesn't pay.
> - **M5 Max / affine-int4 (2026-05-30):** Step-A is ~37 % of the cycle.
>   Both a naive cold count A/B (**45.0 → 52.8 tok/s, ≈1.17×**) AND the
>   **controlled** harness on the affine checkpoint (depth 3, self-normalized
>   ratio) agree it WINS: **chained-ON ratio 1.116 vs OFF 0.879, +23.7pp beyond
>   the ±19.8pp noise band**, with acceptance *higher* under ON (1.30 vs 0.91)
>   — no acceptance penalty here, the opposite of the M3/nvfp4 case.
>
> So it is now **config-gated by GPU arch gen**: default ON on M5+ (gen ≥ 17),
> where it is measured net-positive (affine +16 % / nvfp4 byte-identical to AR);
> default OFF on M1–M4 (gen 13–16) **until** the chained-hidden lazy-slice
> eval-scheduling stall (the depth-3 acceptance regression) is fixed so it also
> wins on M3/bf16/nvfp4. `MLX_MTP_CHAINED_CYCLES=0/1` overrides either direction.
> On M5-class + affine-quant it is the single biggest **structural** lever
> toward MTPLX parity. The earlier "Step-A is only ~14 % of cycle wall" claim is
> config-specific (M3/bf16); on M5/affine it is ~37 %. (A still-earlier naive
> same-binary A/B reported a spurious depth-1 "win" from a thermally-confounded
> run — the flag-invariant AR baseline had swung ~2×; the controlled harness
> exists to defeat exactly that confound.)

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

| Attempt                                      | Result on M3 Max                                                                                                 | Result on M5 Max                                                                                                                                              |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Batched GEMM (`qmm` / `qmm_t_splitk`) at M=4 | ~38 % slower than `qmv`; split-K low-precision accumulation also degrades acceptance → ratio collapses to ~0.75× | (not re-tested — split-K accuracy regression is hardware-independent)                                                                                         |
| `multi3` multi-row `qmv` (W6.30 port)        | 0.94× geomean vs stock `qmv` — slower even in its native affine-4-bit M=3 envelope                               | **0.94× geomean** — identical to M3; the M5 Neural Accelerator helps stock `qmv` symmetrically (kernel + microbench removed as a proven dead-end, 2026-06-01) |
| `SPARSE_ACCEPT`              | zero measured win (W6.19)                                                                                        | (not re-tested — sync-collapse knobs are CPU-side; hardware-invariant)                                                                                        |
| `VERIFY_ASYNC_EVAL` (W6.9)                   | zero measured win on M3 Max (overlap budget too small)                                                           | **+5–6% ratio** at depth 3 on `qwen3.6-27b-nvfp4-mtp-oproj8` (~1.07× → ~1.13×) with byte-identical acceptance. Flipped to default ON in Phase 3 (2026-05-26). |
| End-to-end MTP/AR ratio at optimal depth     | 1.14× (depth 1)                                                                                                  | **1.15× (depth 1)** — Neural Accelerator does not widen the gap                                                                                               |

Stock MLX's small-M `qmv` is already near-optimal on both M3 and M5 Max:
at these shapes the verify is not dominated by per-row weight re-reads,
so reading weights once (the `qmm` / `multi3` approach) buys nothing and
the extra machinery costs more. Crucially, the M5 Neural Accelerator
benefits stock `qmv` and `multi3` in the same proportion, so the
microbench ratio is invariant — and the end-to-end MTP/AR ratio is
invariant too, because the AR baseline gets the same NA uplift as the
MTP verify forward.

**Why MTPLX reaches ~2.2× and this does not — it is NOT the hardware, and NOT a fork.**
A direct audit of the MTPLX source (2026-05-30) settles this. MTPLX ships
**stock PyPI mlx 0.31.2** — there is no required private fork. The
`mlx-mtplx-…-qmm` fork is optional (surfaced only as
`optional_fast_mlx_fork_active`), and its only content is the same small-M
`qmv` retune we already benchmarked at **0.94×**. Its custom verify kernels
(multi-row `qmv`, fused-MLP, top-k lm*head) are **default-OFF probes whose
own docstrings call them "slower than stock."** MTPLX's headline 2.24× is an
**acceptance multiplier, not a kernel one**: depth-3 + T=0.6 + fan-pinned
M5 Max on a checkpoint hand-tuned for ~95% acceptance
(`Qwen3.6-27B-MTPLX-Optimized-Speed`, per-position `[1.00, 0.98, 0.94]`).
The monotonic depth ladder (AR 24.6 → D1 41.6 → D2 48.6 → D3 54.5 tok/s) is
driven by acceptance, not faster GEMMs. The one genuinely valuable custom
kernel they have — a fused GatedDeltaNet recurrence-from-conv-tape — is a
verify-dispatch micro-opt for recurrent layers, not a multiplier.
(dflash-mlx's 2.95–4.4× is a \_different* architecture — a separately-trained
block-diffusion drafter — out of scope for native MTP heads here.)

### Same-checkpoint engine head-to-head — the Step-A forward (2026-05-30)

The "acceptance multiplier" story above is **only half** the MTPLX gap. To
isolate the engine from the checkpoint and the prompt, we ran **MTPLX's own
engine on its own checkpoint** (`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`,
affine-int4) against **our engine on the identical checkpoint** — same M5 Max,
same count prompt, T=0 greedy, depth 3, **no fan-pinning on either side**
(MTPLX `thermal: not configured`). Our loader reads their checkpoint fully:
the trunk via the standard `quantization` key, and the `mtp.safetensors`
sidecar (merged in as a `model-*-of-*.safetensors` shard) with the affine
group_size inferred per-tensor from the scale-column shape — so the
non-standard `mtplx_mtp_quantization` key being unread does **not** mis-dequant
the heads (verified: they hit the perfect `[1,1,1]` depth-3 ceiling on
counting).

| Metric (count prompt, T=0, depth 3, no fan-pin) | Ours | MTPLX | Winner |
| ----------------------------------------------- | ---- | ----- | ------ |
| AR decode                                       | 25.82 tok/s | 22.74 tok/s | **us, 1.14×** |
| MTP acceptance / cycle                          | 2.94 `[1,1,1]` | 3.0 `[20/20,20/20,20/20]` | tie (both perfect) |
| **MTP D3 decode**                               | 40.18 tok/s | **56.43 tok/s** | **MTPLX, 1.40×** |

So at **matched (perfect) acceptance** and with our **faster** AR forward,
MTPLX's MTP cycle is still **1.40× faster**. That residual is NOT acceptance,
NOT quant format, NOT the verify kernel, NOT hardware — it is a **structural
difference in trunk forwards per cycle**:

- **MTPLX runs 1 full-trunk forward per full-accept cycle** — the `D+1`-wide
  verify (`generation.py:4737`). The next cycle's draft seed is the *same*
  forward's last-position hidden (`hidden = verify_hidden[:, -1:, :]`,
  `generation.py:5037`) and the bonus token is sampled from its logits
  (`pending_primary`, `generation.py:5051-5068`). No separate target forward.
- **We run 2** — a single-token **Step-A** main-model forward, gated by
  `do_step_a` (`chat_common.rs:3398`) and dispatched at `chat_common.rs:3408`
  via the `forward_with_hidden` closure (bound to `forward_compiled_with_hidden`,
  `model.rs:2560`) — **plus** the `D+1` verify. The Step-A forward exists only
  to produce the next draft seed, which the verify forward already computes and
  we discard on the default path.

Timing arithmetic matches the measurement exactly: our single-token trunk
forward ≈ 38.7 ms (25.82 tok/s), 4-wide verify ≈ 58 ms, drafts ≈ 9 ms.
Ours `= 38.7 + 9 + 58 ≈ 106 ms / 4 tok ≈ 40 tok/s` (measured 40.18); MTPLX
`= 9 + 58 ≈ 67 ms / 4 tok ≈ 56 tok/s` (measured 56.43, `verify_ms_per_call=57.97`,
`draft_time_s/20 ≈ 9.2 ms`). The ~38.7 ms Step-A forward — **~37 % of our
cycle on this M5/affine config** — is the entire gap.

**The lever exists and is wired, now default ON on M5+ (gen ≥ 17):
`MLX_MTP_CHAINED_CYCLES`** skips Step-A and seeds the next draft from
`verify_hidden[K]` — exactly MTPLX's trick, parity-exact at T=0. (Set `=0` to
force it OFF on M5 for a bisect.) Two measurements on this checkpoint agree it
wins:

> - **Naive cold count A/B:** **45.0 → 52.8 tok/s (≈1.17×)**, closing most of
>   the gap to MTPLX's 56.4 (remaining ~7 % is second-order).
> - **Controlled harness** (`examples/qwen35-mtp-controlled-verdict.ts`,
>   self-normalized MTP/AR ratio, interleaved, 4 repeats, depth 3): **chained-ON
>   ratio 1.116 vs OFF 0.879 — chained-ON WINS by +23.7pp, beyond the ±19.8pp
>   noise band.** Acceptance was *higher* with ON (meanAccepted 1.30 vs 0.91,
>   per-position `[0.633, 0.433, 0.237]` vs `[0.638, 0.213, 0.064]`), so there
>   is **no acceptance penalty here** — unlike the M3/nvfp4 case. (The
>   AR-stability gate flagged thermal instability, CV 15.2%, so absolute tok/s
>   are discounted; the self-normalized ratio and deterministic acceptance are
>   what the harness is designed to trust through exactly that.)

This **overturns the earlier "not a win" verdict for this config** (see the
chained-cycles note above): that verdict was measured on **M3 Max / bf16 /
nvfp4 / a hard prose prompt**, where Step-A is a smaller fraction of the cycle,
the chained-hidden's lazy-eval slice introduced a scheduling stall, AND chaining
*lowered* acceptance. On **M5 Max / affine-int4**, Step-A is ~37 % of the cycle,
removing it clearly wins, and acceptance does not regress. The flag is
config-dependent, not globally good-or-bad — so it is **config-gated by GPU arch
gen: default ON on M5+ (gen ≥ 17), default OFF on M1–M4 (gen 13–16)**, with
`MLX_MTP_CHAINED_CYCLES=0/1` overriding either way. **Remaining before flipping
the default ON on M1–M4 too:** (1) resolve the chained-hidden lazy-slice
eval-scheduling stall so it also wins on M3/bf16/nvfp4; (2) confirm the depth-3
acceptance asymmetry (`[…,0.213,0.064]` OFF vs `[…,0.433,0.237]` ON — the deeper
draft slots accept far better under ON on the warm essay) is a genuine effect,
not a warm-cache Step-A-path artifact worth fixing on the OFF path too.

**Bottom line:** the full MTPLX advantage decomposes into (1) an **acceptance**
component (the ~2.24× headline = T=0.6 + a prompt tuned for ~95 % acceptance)
and (2) a **structural per-cycle-forward** component (~1.40× at matched
acceptance, = the Step-A forward we run and they don't). (1) is a
prompt/sampler artifact; (2) is a real, addressable engine win via Step-A
elimination.

**Native-heads MTP/AR ceiling ≈ 1.1–1.15× on both M3 Max and M5 Max**
(depth 1, **with Step-A present**). The 1.40× same-checkpoint engine gap
above shows the ceiling is NOT intrinsic to native heads — eliminating the
redundant Step-A forward (already implemented behind
`MLX_MTP_CHAINED_CYCLES`) recovers most of it **without** any custom
verify-path kernel, private fork, or the dflash separately-trained-drafter
architecture. M5-class hardware does not unlock it; cycle structure does.

**Long-context behaviour.** The verify forward's attention cost scales
with context length, eroding the speculative advantage on long prompts:
on a ~1k-token prompt the MTP/AR ratio drops toward parity even though
per-cycle acceptance stays healthy. A future adaptive context-length
guard could fall back to plain AR decode once the prompt length crosses
the break-even point.

Interactions:

- `MLX_MTP_USE_TAPE_REPLAY=0` is safe to combine with all other flags.
- `MLX_MTP_VERIFY_ASYNC_EVAL` (default ON) composes cleanly with every
  other knob; parity holds byte-exact at `T=0` across all combinations
  on qwen3.5-4b.
- Setting `mtpDepth` explicitly disables adaptive depth by default;
  pass `mtpAdaptiveDepth: true` alongside to keep adaptation enabled with
  `mtpDepth` as the initial seed.

Naming notes:

- The W6.9 flag was briefly drafted as `MLX_MTP_PREFETCH`. The current
  name reflects the actual mechanism (intra-cycle overlap with CPU-side
  graph construction, not cross-cycle draft staging). The literal
  "stash next-cycle draft handle, drain at cycle start" prefetch lives
  in a follow-up scoped to `MLX_MTP_CHAINED_CYCLES=1`.

Cross-references:

- TS field JSDoc: `enableMtp` / `mtpDepth` / `mtpAdaptiveDepth` on
  `ChatSession.send` in `packages/lm/src/chat-session.ts`.
- Source of truth (env-var readers + inventory table):
  `crates/mlx-core/src/models/qwen3_5/chat_common.rs` (`mtp_use_tape_replay`,
  `mtp_chained_cycles_enabled`, `mtp_verify_async_eval`,
  `mtp_sparse_accept_enabled`,
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

The Qwen3.5 GDN recurrence uses the **per-step** kernel by default on **every** GPU generation. The alternative chunked prefill kernel (`crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc`) is pure scalar-FMA + `simd_sum` reductions — it contains **zero** `simdgroup_matrix` / NAX matmul instructions, so it never had a tensor-core advantage. It was once gated ON for M5+ (`gen >= 17`) on the theory that M5's memory bandwidth made its `O(BT²)` tiling a net win, but that was never A/B'd on M5. Measured 2026-06-04 on an M5 Max (gen 17, isolated worktree): the chunked kernel is **2.8–3.5× slower** end-to-end prefill TTFT than per-step (24–31× slower per isolated GDN call) across 580–5384 prompt tokens — and ~2× slower on M3. The gen gate was a stale inversion of an old M3 result and has been removed. Chunked is retained behind `MLX_GDN_KERNEL=chunked` for A/B only. **The two kernels are NOT token-identical**: they differ by 1–2 bf16 ULP (two valid reduction orderings), which can flip a greedy argmax and change the continuation on some long prompts. (Future, unclaimed: porting the chunked kernel's Phase-2/Phase-4 GEMMs to `simdgroup_matrix`/NAX `matmul2d` is a prefill-only opportunity that has **not** been done — and is moot while per-step is the hot path.)

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
