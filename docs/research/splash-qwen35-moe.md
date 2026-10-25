# Ornith 35B MoE comparison with Splash

## Scope and provenance

- Both engines use `Ornith-1.5-35B-Q4_K_M.gguf`: 40 layers, hidden width 2048,
  256 experts with top-8 routing, expert width 512, 16 attention heads, and two
  KV heads of width 256. Splash calls this architecture `Qwen3.6-35B-A3B`.
- The official companion is `incoai/Qwen3.6-35B-A3B-DFlash2`, revision
  `51ef7b6923ad6c14cb1bb41c37a9041446496ab6`: six layers, eight target taps
  `[1,6,11,16,22,27,32,37]`, and seven proposals per block. Fine-tunes need
  their own acceptance measurement.
- The source port follows Splash `1482f6bbc2abf12375332725302019594150b6cb`;
  its Apache-2.0 attribution remains in `THIRD_PARTY_NOTICES`.
- The active MLX pin has 10 direct commits from `mlx-node/mlx` branch `fn-d1`, pinned
  through `7aa7d680b`. CMake, bridge headers, Metal kernels, and preambles use
  that checkout directly: there is no parent-repo build-time patch stack or
  generated patched source tree. Prebuilts must match the clean commit and
  build key. See the [MLX fork guide](../mlx-fork.md).

## Architecture and invariants

### Grouped GGUF MoE

- The [grouped primitive](../../crates/mlx-sys/src/mlx_splash_moe.cpp) performs
  routing, expert grouping, activation gather, fused gate/up, down projection,
  and weighted combine without CPU routing readback.
- Q4_K/Q5_K/Q6_K three-plane and Tiled64 weights remain quantized; each
  projection may use a different format. Eligible routed planes move into
  tiled storage without retaining a second resident copy.
- The fast path requires Metal tensor ops, BF16 activations, 1–2048 physical
  rows, at most 256 experts/top-8, and dimensions divisible by 256. CPU,
  autograd, unsupported geometry/quantization, and `MLX_SPLASH_MOE=0` use the
  generic path. Replay validates geometry and retains encoder resources.
- Routing preserves BF16 softmax/top-k rounding, stable tie order, and BF16
  rounding after every denominator and expert-sum addition. Projection
  precision follows the existing native dispatch tiers. Rare one/two-row
  experts use the column path only where native uses vector precision; using
  it in grouped-GEMM tiers changes accumulation order.

### DFlash2 execution and cache ownership

- MoE DFlash2 uses the whole-turn exclusive lane and flat target caches. Its
  companion does not advertise native MTP weights or enter continuous
  batching. Disabling speculation returns to the existing paged/flat lifecycle.
- Draft context is reusable only when owner, conversation root, token
  provenance, and target frontier agree. Owner changes invalidate it.
- Compiled verifier keys include model identity, physical width, KV format,
  destination mode, and budget mode. Unsupported geometry uses eager
  verification, but a compiled failure after stores are constructed resets the
  turn; eager retry on possibly mutated caches is prohibited.
- All physical rows are written before acceptance. Stop, cancellation, and
  repetition limits clamp retention before one commit. Partial acceptance
  rewinds attention offsets, replays only retained GDN state, and appends only
  retained hidden taps. Draft work is submitted before target verification;
  committed state/context stores are submitted at the cycle boundary.
- [Mixed projection](../../crates/mlx-sys/src/mlx_mixed_projection.cpp) fuses
  eligible heterogeneous M8 projections while preserving each segment's
  split-K count and output allocation. The bounded GDN b/a fusion is limited
  to its measured 32+32 by 2048 contract; other widths stay independent.
- The 2048-slot long-context window exposes at most 2047 old rows and all eight
  current proposal rows, using absolute positions and RoPE base 10,000,000.

### Adaptive verification

- `mtpAdaptiveDepth: true` affects only unpenalized greedy full seven-proposal
  blocks. Sampled, penalized, teacher-forced, and tail blocks retain complete
  verification; `false` selects fixed verification. Fixed `all` still drafts.
- Physical width remains eight. The GPU live-row budget is a lazy graph input,
  not host truncation/readback. The selector preserves predecessor-path IDs,
  maximizes expected retained tokens per predicted cycle cost, and permits
  routed expert work to be skipped only when every masked output is finite zero.
- For R live rows, accepted drafts are capped at R-1 and the capped target
  argmax supplies the boundary; rollback still accounts for all eight writes.
  Policy training uses raw matches before stop clamping or invalid suffix rows.
- Adaptive policy state is per loaded model. Adaptive cycles settle committed
  GPU work before measuring cost, including rollback and draft-context work but
  excluding host-only emission. Fixed cycles keep asynchronous submission.

## Selected configuration

- Supported MoE geometry defaults to compiled draft, draft tensor attention,
  indirect expert dispatch, mixed projection, GDN blobs, and the measured
  affine/output register-kernel shapes. Explicit `0` overrides retain fallback
  paths. Concurrent encoding and the original command-buffer limits remain.
- The [context policy](../../crates/mlx-core/src/models/qwen3_5_moe/dflash2_decode.rs)
  uses retained-height FC and K/V projection for keep 1–6 and full eight-row
  projection for keep 7–8. This V28 crossover is now the default; its temporary
  context override was removed.
- Source-style expert/residual/routing/mixer/QK/head arithmetic and deferred
  GDN remain opt-in diagnostics, not selected defaults.
- Source convolution was OFF in V28 and is not a default. Later product and
  teacher-forced capture quartets were consistent but did not reproduce the
  older anomaly or find a defensible production fix. It remains excluded from
  parity claims pending fresh evidence.

## Correctness and current validation

- After the rare-expert accumulation correction, grouped output and route
  IDs/coefficients were bitwise equal across all 40 captured real layers.
  The 2047-position native candidate/baseline teacher check then reported
  top-1 agreement 1, top-256-plus-tail KL 0, and identical mean NLL
  `1.4252860731302848`. This covers those measured quantities and geometry,
  not every logit, schedule, or general model accuracy.
- The adaptive Rust policy matched the Splash reference's coefficients,
  cost-table float bits, and ordered labels at nine checkpoints through 3000
  observations, including ring wrap. Within-condition continuation traces are
  repeatable; arithmetic changes may legitimately change acceptance paths.
- Review regressions cover tiled resident weights, replay shapes and K/V
  aliases, cache clearing, companion geometry, adaptive cost settlement, and
  standalone JIT preambles.

Recorded validation on the rebased direct-fork source:

- Full Rust workspace tests and Clippy passed.
- TypeScript passed with BGE isolated and TF32 off; the identical BGE failure
  exists on clean `main`.
- Direct-source and prebuilt native builds passed.
- Real-checkpoint lifecycle checks passed 9/9 with BF16 KV and 9/9 with INT8 KV.
- Direct-fork build-script checks passed 4/4; serial-encoder OFF/ON checks passed.

## Performance evidence

Protocol: Apple M5 Max; the same Ornith Q4_K_M and official DFlash2; greedy;
BF16 KV and FP32 recurrent state; 1024 generated tokens plus 128 warmup tokens.
Decode uses 1023 tokens after the first token; engines are fresh and prefixes cold.

Pre-work target-only AR baseline `bfc6c7369`, median decode tok/s over three
quiet samples:

| Prompt tokens | Native AR |
| ------------: | --------: |
|            38 |    102.66 |
|          2048 |    101.06 |
|         32768 |     74.18 |

V28 used the experimental combined source-arithmetic/mixer profile and source
norms. Its median decode tok/s uses three accepted samples per engine/condition:

| Verification  | Prompt tokens | Native | Splash | Native vs Splash |
| ------------- | ------------: | -----: | -----: | ---------------: |
| Fixed `all`   |            38 | 284.50 | 285.57 |           -0.37% |
| Fixed `all`   |          2048 | 271.72 | 262.73 |           +3.42% |
| Fixed `all`   |         32768 | 214.31 | 220.60 |           -2.85% |
| Adaptive auto |            38 | 259.30 | 280.18 |           -7.45% |
| Adaptive auto |          2048 | 238.78 | 269.21 |          -11.31% |
| Adaptive auto |         32768 | 207.35 | 235.00 |          -11.76% |

Positive delta means native is faster. Later pairs experienced thermal
throttling, and adaptive-short results showed fresh-process variance; retain
clock telemetry and do not over-read small differences.

The AR baseline and optimized DFlash2 are different modes, so these tables do
**not** provide a mode-matched before/after ratio. V28 also predates the final
review fixes, rebases, and direct-fork migration. The final-source quiet-window
fixed/adaptive and AR matrix remains pending; no new timing runs during gaming.

## Reproduction and retained artifacts

1. Freeze source revision, addon and both Metal libraries with hashes, model
   hash, companion revision, and device. Keep each engine's binary/libraries
   together and record every rejected or skipped attempt.
2. Freeze identical token IDs for 38-, 2048-, and 32768-token prompts. Confirm
   the native chat template reproduces them; give Splash the frozen IDs through
   its completions API. Use one model process at a time and cold salted prefixes.
3. Warm the selected prompt separately, then alternate/reverse fresh-engine
   order (A/B/B/A). Require at least three accepted samples per condition and
   fresh adaptive models so learned budgets cannot cross conditions. Early
   stops are failures. Require a quiet machine before loading; retain and retry
   CPU-contaminated timed requests in a fresh engine, at most twice.
4. Record prefill/TTFT and decode separately, with cycles, output hashes, GPU
   clocks, CPU, RSS/allocator peaks, and swap. Native and Splash timing counters
   have different host-gap/first-emission accounting; do not equate them.
5. Use teacher-forced distributions on identical IDs for cross-engine numerics.
   Text hashes detect same-engine drift but cannot establish cross-engine parity.

Local evidence is retained rather than duplicated here. Packaged builds and
focused checks are under `.cache/benchmarks/splash-qwen35/`, notably
`candidate-v28-keep7/` and `candidate-v6-prefill-order/`. Raw paired runs are
under `$HOME/.cache/mlx-node-research/splash-qwen35-moe-20261022/v28-keep7/`:
`direct-short-2k-all/`, `direct-32k-all/`, `direct-all-auto/`, and
`direct-2k-splash-retry1/`. These caches are local artifacts, not tracked files.
