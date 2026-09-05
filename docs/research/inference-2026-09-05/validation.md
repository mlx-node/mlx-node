# Implementation validation

Measured 2026-09-05 on an Apple M5 Max, 128 GiB unified memory, macOS 26.6.2.
Baseline: `100a03ad10a14a7eab7d816773c072e39dc10206`. Changed build: the
accompanying worktree diff. Both native addons were built with the repository's
`vp run build:native` wrapper, the same Cargo.lock and vendored MLX revision.
No checkpoint conversion, quantization or new dependency was introduced.

## Full-model measurements

Qwen3-0.6B BF16, 128 output tokens per request, greedy sampling with alternating
row penalties. Each process warms each concurrency level once, then measures
three waves. Processes ran in baseline/changed/changed/baseline order, giving
six measured waves per build and concurrency level. No other test or benchmark
used the GPU during these runs. Model load is excluded; prompt prefill is included.
The scheduler reached the requested occupancy in every measured wave.

| Concurrent requests | Baseline median ms | Changed median ms | Changed aggregate tokens/s | Throughput change |
| ------------------- | ------------------ | ----------------- | -------------------------- | ----------------- |
| 1                   | 538.864            | 552.274           | 231.77                     | -2.43%            |
| 2                   | 1129.094           | 1099.628          | 232.81                     | +2.68%            |
| 4                   | 1219.492           | 1152.199          | 444.37                     | +5.84%            |
| 8                   | 1418.700           | 1251.960          | 817.92                     | +13.32%           |

These are one checkpoint and workload, with six samples rather than a statistical
performance guarantee. The single-request medians regress by 2.43%; their sample
ranges overlap. Do not infer a universal latency improvement. All waves generated
the same token counts and finish reasons. 83 of 90 paired long outputs were byte
identical. Baseline repeats also varied; every changed per-row output occurred
in the corresponding baseline row's output set. Variable arrival/batch shapes
and BF16 near-ties are a plausible explanation, consistent with the existing
bounded concurrency gate's documented limitation. This is not an exact long-output
parity result.

The workload is in [qwen3.ts](benchmarks/qwen3.ts). Default native reasoning
behavior was identical in both builds. The original temporary script also passed
`enableThinking: false`, which native config ignores; the typed reproduction
omits that ineffective field. Reproduce against separately built bindings:

```sh
MLX_AGENT_METRICS=0 MLX_PERSIST_PAGED_CACHE=0 MLX_SCHED_MAX_NUM_SEQS=8 \
  oxnode docs/research/inference-2026-09-05/benchmarks/qwen3.ts \
  /absolute/build/packages/core/index.cjs /absolute/qwen3-0.6b-mlx-bf16 \
  /tmp/qwen3-measurement.json revision-label
```

Gemma4-12B with the DSpark block-7 companion was measured with penalties,
200 output tokens and fixed maximum draft depth 7. One warmup and three measured
turns per build, baseline then changed, yielded medians of 3401.740 ms and
3406.059 ms (-0.13% throughput). There is no resolved end-to-end improvement in
this pair. All three paired outputs, finish reasons and acceptance counters were
identical: 42 speculative cycles and 3.7381 mean accepted draft tokens per cycle.
The isolated acceptance result below must not be promoted into a Gemma throughput
claim. Run [dspark.ts](benchmarks/dspark.ts) with arguments `bindingPath`,
`targetModelPath`, `draftModelPath`, `outputJson`, `revisionLabel`, and the same
`MLX_AGENT_METRICS=0 MLX_PERSIST_PAGED_CACHE=0` environment.

[Per-run measurements](measurements.json) retain timings, performance counters
and output SHA-256 hashes for both workloads, in execution order. Hashes make the
parity comparison auditable without retaining generated prose in the repository.

## Isolated completion costs

Release tests run both the former serial algorithm and the grouped algorithm
inside the same binary. They alternate order across seven measured rounds,
100 operations per round, after warmup; entries are medians. The sampling case
uses a 32,000-token vocabulary, mixed greedy/stochastic rows, penalties and one
forced row when present. These timings measure control flow, not full inference.

| Rows | Serial sampling µs | Grouped sampling µs | Speedup |
| ---- | ------------------ | ------------------- | ------- |
| 1    | 235.61             | 229.89              | 1.025×  |
| 2    | 752.62             | 477.15              | 1.577×  |
| 4    | 844.69             | 443.63              | 1.904×  |
| 8    | 1920.35            | 602.70              | 3.186×  |

| Draft tokens | Acceptance   | Serial µs | Grouped µs | Speedup |
| ------------ | ------------ | --------- | ---------- | ------- |
| 1            | All          | 389.18    | 212.49     | 1.831×  |
| 1            | Reject first | 186.72    | 201.89     | 0.925×  |
| 3            | All          | 744.16    | 239.50     | 3.107×  |
| 3            | Reject first | 183.76    | 241.21     | 0.762×  |
| 7            | All          | 1386.29   | 267.59     | 5.181×  |
| 7            | Reject first | 174.00    | 263.20     | 0.661×  |

The deterministic speculative path does extra hypothetical work on early
rejection: 89.20 µs more in the seven-draft worst case measured here, versus
1118.70 µs saved on full acceptance. Stochastic residual acceptance is unchanged.

```sh
cargo test -p mlx-core --release --lib benchmark_mixed_sampling_completion -- --ignored --nocapture --test-threads=1
cargo test -p mlx-core --release --lib benchmark_penalized_accept_completion -- --ignored --nocapture --test-threads=1
```

## Correctness and build checks

The new tests compare grouped sampling with the former serial path over 16 RNG
seeds, mixed penalties and forced-token rows. Preparation and host-read failures
remain local to their row. Deterministic speculative acceptance is compared with
a sequential oracle at every rejection frontier for draft lengths 0–5 and four
penalty-context lengths. Existing cache transaction and scheduler tests remain
part of the broader suite.

Completed local checks:

- Core debug unit suite, excluding the expensive int8 GEMM module: 3382 passed,
  103 ignored. Int8 GEMM module in release: 8 passed, 10 ignored.
- Core integration targets: 109 passed, 132 checkpoint/manual tests ignored.
- Paged-attention crate: 335 passed, 16 ignored, including SSD/cache and Metal
  graph coverage.
- The checkpoint-backed `concurrent_batched_parity` gate passed with the real
  Qwen3 BF16 model: serial/uniform/ragged and interleaved streaming output parity,
  N=8 occupancy, row-specific penalties, healthy-peer completion after failure,
  and execution of the unchanged fused greedy epilogue. Its two-token outputs
  deliberately bound BF16 near-tie amplification; the longer benchmark above
  records that limitation separately.
- `cargo clippy -p mlx-core --lib --tests -- -D warnings` passed.
- TypeScript broad run: 3118 passed initially. All 13 failed files were missing
  existing tokenizer/template or GSM8K fixtures in the new worktree. After copying
  those fixtures into ignored directories, all 13 files passed (178 tests). Only
  failed files were rerun; no failure remains from the original run. Other
  checkpoint-gated suites remain skipped.
- Canonical native build and TypeScript typecheck passed. Regenerated native
  declarations are synchronized; public signatures are unchanged.

The profile split is intentional: int8 GEMM contains a very large scalar CPU
oracle that is impractical in debug, while three existing frontier tests require
debug assertions. An initial all-release run reported those three failures;
they pass in the debug suite above. This is not a claim that the unmodified
all-release test command passes.

The core profile split and paged suite can be reproduced with:

```sh
cargo test -p mlx-core --lib -- --skip models::qwen3_5::int8_gemm::tests --test-threads=1
cargo test -p mlx-core --release --lib models::qwen3_5::int8_gemm::tests -- --test-threads=1
cargo test -p mlx-core --test '*'
cargo test -p mlx-paged-attn
```

Repository-wide `vp check` already failed on 33 formatting files at baseline.
Formatting the touched documentation reduced this to 29 pre-existing files;
unrelated files were left alone. Changed Rust and benchmark scripts pass formatting
and lint checks. Markdown links and structure were checked; no rendered-page
visual review was performed. No remote CI or cross-runtime benchmark was run.

## Scope of the result

The implementation removes duplicate command adapters and scheduler telemetry
forwarding across seven model families. It reduces completion boundaries in
mixed-row sampling and penalized greedy speculation, preserving row construction
order, per-owner state and the existing SSD lifecycle. It does not enable
scheduled speculative concurrency or migrate the Metal backend. Their concrete
prerequisites are in the [research report](report.md#next-stage-plan-and-gates).
