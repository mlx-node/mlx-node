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

## Int8 K/V default

The flat full-attention K/V of `qwen3_5_moe` now takes the same geometry
default as dense `qwen3_5`: int8 on Metal at head 256 when the GQA factor is at
most 8, BF16 elsewhere. The old MoE-only BF16 override in
`Qwen3_5MoeConfig::to_dense_config` is gone. `kvFormat: 'bf16'` (load option,
`--kv-format bf16`, or a top-level `kv_format` key in `config.json`) opts out.
MTP and DFlash2 draft K/V stay BF16, and the block-paged pool stays BF16.

Splash reference (`~/workspace/github/splash` at `031a25a6`):

| Cite                                                                   | Fact                                                                                               |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `runtime/engine/RuntimeResources.hpp:103`                              | `kv::Format kvFormat = kv::Format::Int8;` for every model                                          |
| `runtime/engine/NativeArguments.hpp:46`, `NativeArguments.cpp:126-129` | CLI default int8; `--kv-format bf16` is the only opt-out                                           |
| `runtime/engine/RuntimeResources.mm:727-728`                           | `kvLayout.format = config.kvFormat;` no per-family override                                        |
| `runtime/ops/PagedAttention.cpp:59-65`                                 | one kernel per geometry: 24/4/256, 16/2/256 (`_kv2_g8`), 32/2/256 (`_kv2_g16`), 24/2/256, 32/4/128 |
| `runtime/ops/PagedAttention.cpp:80`                                    | an unknown layout throws (no silent fallback)                                                      |

Which models change:

| Model                 | q / kv heads, head | gqa | Default             |
| --------------------- | ------------------ | --- | ------------------- |
| Qwen3.8-27B (dense)   | 24 / 4, 256        | 6   | int8 (unchanged)    |
| Qwen3.6-35B-A3B (MoE) | 16 / 2, 256        | 8   | **int8 (was BF16)** |
| Qwen3.6-122B (MoE)    | 32 / 2, 256        | 16  | BF16 (gap below)    |

### Gap: no gqa-16 int8 verify (122B stays BF16)

Splash serves 32/2/256 with its own `_kv2_g16` kernel. mlx-node's int8
segmented SDPA has none that serves a wide verify block at gqa 16:

```
two-pass stage 1 = 32 x gqa x rows threads <= 1024
gqa 8  -> widest chunk 4 rows -> 8-row verify = 4 + 4   ok
gqa 16 -> widest chunk 2 rows -> 3-4 rows split ok,
                                 5-8 rows: segmented_verify_head_rows = -1
          bf16 verify: falls back (segmented_verify_head_len is None)
          int8 verify: calls segmented_sdpa_int8 with no fallback
                       -> "unsupported segmented SDPA verify block"
```

So at gqa 16 every flat int8 verify block of 5 to 8 rows (MTP depth >= 4,
DFlash2's 8 rows) would be an error. `KvFormat::default_for_geometry` adds the
GQA bound (1..=8), and 122B stays BF16 by default. Lift: a gqa-16 int8 verify
kernel (Splash `_kv2_g16`). The bound is the two-pass pipeline's
`max_threads_per_threadgroup`; the unit test
`int8_verify_chunks_bound_the_default_gqa` pins it on the test device only.

### DFlash2 window (35B-A3B, M3 Pro inputs)

`dflash2_window_terms`, the S1 27B trigger inputs (working set 27 GiB, MLX
active 17,357 MiB, pool budget 5,051 MiB = 16,163 BF16 blocks of 327,680 B,
325-block runway, the 6 x 8 x 128 `incoai/Qwen3.6-35B-A3B-DFlash2` draft ring
56,795,136 B, 7 scratch rows), only the geometry swapped:

| Case          | Row bytes | Pool term | Reserve term | Window                    |
| ------------- | --------- | --------- | ------------ | ------------------------- |
| 27B int8 (S1) | 33,280    | 139,257   | 166,714      | 139,257                   |
| 35B BF16      | 20,480    | 244,217   | 275,721      | 244,217                   |
| 35B int8      | 10,400    | 475,129   | 542,966      | **262,144** (trained cap) |

Row bytes = 10 full-attention layers x 2 (K, V) x 2 KV heads x (512 B BF16 or
256 + 4 B int8). The int8 window hits the trained 262,144, so the gain shows as
the cap, not 2x. Pinned by `moe_dflash2_window_prices_the_default_format`.
Estimate only (not pinned): with 35B Q4_K_M weights on the 36 GB M3 Pro the
window would move 122,489 -> 237,561; its draft-size premise is not checked.

### Quality gate

Teacher: the same checkpoint with BF16 K/V (a view directory whose
`config.json` sets `"kv_format": "bf16"`). Data: 8 rows of 24 joined CNN
articles, `--seq 8192`, `--top-k 1024`. The first 2,048-token prefill chunk has
an empty prefix and never reads int8, so the tail numbers count only the
positions past it: `kl_tail = meanKlTopk x P / P_t`,
`top1_tail = 1 - (1 - top1Agreement) x P / P_t`, `P_t = P - rows x 2,048`.
Stop band: tail top-1 < 0.97 or tail KL > 0.05.

| Model                  | Run          | kvFormat | rows / positions | meanKlTopk | top1Agreement | tail KL | tail top-1 | teacherMeanNll | teacherTailMass |
| ---------------------- | ------------ | -------- | ---------------- | ---------- | ------------- | ------- | ---------- | -------------- | --------------- |
| 35B-A3B (bf16 weights) | bf16 control | `<..>`   | `<..>`           | `<..>`     | `<..>`        | -       | -          | `<..>`         | `<..>`          |
| 35B-A3B (bf16 weights) | default      | `<..>`   | `<..>`           | `<..>`     | `<..>`        | `<..>`  | `<..>`     | `<..>`         | `<..>`          |
| 27B (bf16 weights)     | bf16 control | `<..>`   | `<..>`           | `<..>`     | `<..>`        | -       | -          | `<..>`         | `<..>`          |
| 27B (bf16 weights)     | default      | `<..>`   | `<..>`           | `<..>`     | `<..>`        | `<..>`  | `<..>`     | `<..>`         | `<..>`          |

Commit `<commit>`, device `<device>`.

Gaps:

- DFlash2 acceptance on 35B, int8 vs BF16, teacher-forced: not run (no local
  35B DFlash2 draft). Lift: download `incoai/Qwen3.6-35B-A3B-DFlash2` and run
  one fixed transcript teacher-forced.
- Eval prefill reads the int8 prefix through dequantize + fused SDPA; decode
  and verify read it with the int8 kernels. Both read the same int8 values;
  the kernel error vs the dequantized reference is pinned by `kv_int8_tests`.
- `mlx eval cache` has no format flag and uses the load default, so on
  head-256 Qwen a teacher past 2,048 tokens is int8 K/V unless the checkpoint
  config says `bf16`.
- 122B stays BF16 (above).

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
