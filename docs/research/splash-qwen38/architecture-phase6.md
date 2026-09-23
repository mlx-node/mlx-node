# Architecture experiments: verifier outputs, settlement and packed loads

**Historical experiment report.** The five paths and their environment switches
were subsequently removed; see [the cleanup decision](cleanup.md). The results
below describe frozen phase-6 binaries, not the current runtime.

September 22, 2026. This implements the bounded experiments from
[the architectural follow-up](architecture-next.md) against the same GGUF
Qwen3.8-27B target, BF16 DFlash2 companion and pinned Splash source. Raw evidence
is retained under `.cache/benchmarks/splash-qwen38-phase6/`.

## Changes and decision gates at experiment close

| Experiment                              | Implementation                                                                                                                                                                 | Decision                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Discarded verifier state                | Dedicated GDN recurrence and preparation kernels omit terminal recurrent/history outputs only in detached verification. Arithmetic and replay tape remain unchanged.           | Exact operator/model checks pass; combined benefit is workload-dependent.                            |
| Prepared state settlement               | A per-turn compiled graph consumes immutable snapshots/tape and returns fresh convolution/recurrent state after the host stop clamp. Every retained count has a separate plan. | Every-prefix and continuation checks pass; remains opt-in.                                           |
| Shorter query retention                 | After acceptance/provenance validation, state-only replay aliases the compatible K handle in place of Q. Sequential fallback state is independent of Q.                        | Up to 1.5 MiB logical retention avoided at width eight; actual reclamation may occur later.          |
| Indexed graph replay                    | Cache immutable dependency indices and create fresh array descriptors per invocation. Captured arrays that detach invalidate the plan.                                         | Synthetic CPU benefit; combined model results do not justify a global default.                       |
| Aligned Q5/Q6 extraction                | Load existing packed bytes as aligned words, preserving decoded codes and floating-point operations.                                                                           | Exact production-kernel/model checks pass; mixed gains, remains opt-in.                              |
| SIMD activation sharing                 | Two owner-load/shuffle implementations, with unchanged reduction geometry and arithmetic.                                                                                      | Rejected: exact but substantially slower. No production dispatch.                                    |
| Byte layout / look-ahead                | Assessed after load experiments, as required by the original gate.                                                                                                             | Deferred: no measured transaction/stall evidence supports extra layout, residency or register costs. |
| Mutable state banks / native Metal plan | Reviewed ownership, host clamp, scheduling and compiled-call costs. Functional compiled settlement is the bounded first step.                                                  | Deferred: evidence does not justify mutable banks, permanent arenas or recorded-address replay.      |

The archived experiments used five controls, all off by default: `MLX_DFLASH2_VERIFY_OUTPUTS_ONLY`,
`MLX_DFLASH2_COMPILED_COMMIT`, `MLX_DFLASH2_RELEASE_REPLAY_Q`,
`MLX_COMPILE_INDEXED_REPLAY`, and `MLX_KQUANT_QMV_WORD_LOADS`. Value `1` enables
an experiment. Benchmark and continuation runners record these controls without
recording the complete process environment.

## Device adaptation

No new device-name, GPU-core-count or M5 performance threshold is used. Packed
loads query the actual pipeline's execution width, maximum threads and static
threadgroup memory, plus the device's maximum threadgroup-memory size. Missing or incompatible
pipelines retain the established implementation. The 32-lane reduction and
two-group mapping are inherited algorithm requirements; this experiment does
not change launch geometry.

These checks establish legality, not the fastest configuration. No automatic
best-value tuning is claimed. The existing segmented-attention planner computes
legal query widths from actual pipeline limits; the verifier repair below uses
those detected widths conservatively. A measured selector still needs a stable
model-level winner and cache invalidation by device, shader/driver identity and
workload shape. These results do not justify thresholds fitted to this machine.

## Correctness issues discovered and repaired

Cross-review found that captured arrays can lose their primitive after external
evaluation. The first indexed-replay draft cached whether records were constant
and could access a detached primitive. Final code detects the transition,
discards the index plan and uses original replacement. Shape-aware/shapeless
regressions cover fresh inputs, siblings, captured arrays, nested graphs and
external evaluation. A separate pre-existing fused-compile captured-expression
crash reproduces without indexed replay; the detachment regression isolates
replacement with fusion disabled.

The native suite exposed unsafe shapeless verification for unsupported attention
geometry. The tiny head-32 fixture enters unfused causal attention, whose mask
uses host-side prefix lengths. Growing-prefix invocations cannot reuse those
constants. Failure appeared after the first full-attention layer; earlier GDN
state and first attention K/V stayed exact. Disabling compiled verification
passed 15 repetitions. Source also permits stale-mask indexing through fused
elementwise shape propagation; the actual failing fused node was not captured.

The repair checks device, head dimensions, query width, GQA and both split lengths
before tracing. Unsupported shapes use eager verification, avoiding an unbounded
graph cache specialized at every prefix length. The requested model retains its
supported vector/segmented compiled path. Regression fixtures separately cover
head-32 fallback and head-64 compiled execution, keep counts 1–8, subsequent
cycles, draft-window wrap and reopening retained sessions. Mismatch diagnostics
report the first differing field/element instead of entire tensors.

## Operator evidence

Verifier-only recurrence outputs match exactly for widths 1–8, zero/nonzero state
and the existing one-, two- and four-column variants. Preparation compares all
five retained outputs across 32 cases. Omitting outputs removes 74.8125 MiB of
logical terminal stores and 96 allocator requests per width-eight verification
across 48 GDN layers. This is not measured DRAM traffic, 96 new Metal allocations
or a throughput forecast.

Final indexed replay benchmark: 12 alternating fresh processes, 2,000 calls each.
Original median 188.921 microseconds (186.821–215.850), indexed 150.750
(148.673–156.610): 20.2% less synthetic host graph construction time. Exact
evaluated outputs match. Background Node and intermittent external Rust
compilation overlapped; this is not idle-machine or model-throughput evidence.
See `indexed-replay/final-host/summary.json`.

Aligned extraction passed exhaustive code/position checks plus 200,000 mixed
groups under CPU sanitizers. The small GPU suite covers 147 format/shape/width
cases and 231 candidate configurations, including poisoned outputs and tails.
Production-library comparisons passed 64 Q5/Q6 shape cases with 960 alternating
timing pairs across eight fresh processes. Fixtures were capped at 256 MiB;
vocabulary-head shapes were explicitly skipped. Four-buffer requests sometimes
admitted only one to three buffers. Exact counts and per-shape results are in
[the QMV report](qmv-experiments.md).

Both owner-load implementations were dramatically slower despite exact results.
All measured variants reported the same execution width, maximum threads and
static scratch allocation; those properties cannot establish register pressure,
spills, cache traffic or occupancy. Build-overlap diagnostics remain archived
and excluded from promotion evidence.

## Final validation and throughput

The repaired release suite passed 3,830 tests, with 129 ignored. Three unchanged
tests intentionally require debug assertions and passed separately in the
existing debug executable. Fifteen fresh-process repetitions of the five
compiled-commit tests also passed. Current-source Clippy (library and tests,
warnings denied), formatting and whitespace checks pass. The unit-test binary
matches candidate-12; the subsequent source change replaces two fixed-size
`chunks_exact` loops with equivalent `as_chunks` loops for Clippy. Packaged
candidate-13 and its model checks cover that final source.

Candidate-12 with all five controls enabled exactly matches the retained baseline
in six deterministic lifecycle scenarios at 321 tokens, including streaming,
cached/cold continuation and long-context follow-up. Tokens, hashes, frontiers
and acceptance statistics match; peak monitored RSS was 21.09 GB. Sampled output
is a smoke check, not an equality claim. With fused preparation disabled, a
separate matched candidate-9/candidate-12 pair also agrees on every deterministic
scenario and acceptance statistic. Fused versus unfused preparation is not the
same comparison: text agrees here, but acceptance positions can differ.

Candidate-13's final packaged artifact also passes the 321-token lifecycle
comparison with all five controls enabled: six deterministic scenarios, matching
acceptance statistics, plus sampled smoke coverage. Peak monitored RSS was
21.03 GB. Its addon SHA-256 is
`cbee9e9e42b8feaf4b060a573952180dc1de75906b79ebd965009b79a8ddb892`;
the companion metallibs and source snapshots are identified in its manifest.
Final runtime source differs from that snapshot only in a two-line comment
clarifying that aligned-word loads have an opt-in production route while owner
loads remain research-only. An independent final source/evidence review found
no actionable runtime correctness issue; its device-limit wording correction
is incorporated here.

### Individual controls: diagnostic screen

Each fresh process used short/6K uncached prompts, 512 output tokens and depth
seven. All 16 cases match output hashes, input/cache metadata and acceptance
statistics exactly. The rates below are **not promotion evidence**: unrelated
builds overlapped seven profiles, and the two controls drift substantially.
No task-owned build/test overlapped the screen. See `screen13-activity.json`.

| Control               | Short tokens/s | 6K tokens/s |
| --------------------- | -------------: | ----------: |
| All off, first        |         21.855 |      17.769 |
| All five on           |         22.489 |      23.905 |
| Aligned words only    |         22.092 |      22.268 |
| Verifier outputs only |         22.095 |      21.682 |
| Compiled commit only  |         21.237 |      22.204 |
| Indexed replay only   |         21.356 |      22.492 |
| Query release only    |         20.860 |      26.606 |
| All off, last         |         24.012 |      23.350 |

The control changed by 9.9%/31.4%, larger than plausible attribution from this
single-sample matrix. A subsequent candidate-9 comparison overlapped another
task's GPU model benchmark; this task's process group was stopped, its timing
excluded, and the other task left untouched. The resource guard now detects
external benchmark scripts even when their CPU usage is low and stops the
owned job on a new overlap. Raw failed/interrupted records are preserved.

### Final combined controls: reversed pairs

Four fresh candidate-13 processes ran off/on/on/off. Each generated 1,024 tokens
at depth seven for the same short, 6K and 32K prompts, with no prompt-cache hits.
All 12 cases match exactly in generated text/hash, input hash, token counts,
finish reason and every acceptance statistic. See `pairs13r-analysis.json`.

| Prompt tokens | All off tokens/s, two runs | All on tokens/s, two runs | Median change | Paired changes |
| ------------- | -------------------------: | ------------------------: | ------------: | -------------: |
| 87            |            37.648 / 34.396 |           37.187 / 35.557 |         +1.0% |  −1.2% / +3.4% |
| 6,219         |            38.984 / 39.893 |           42.845 / 41.666 |         +7.1% |  +9.9% / +4.4% |
| 32,488        |            22.445 / 20.495 |           22.005 / 21.555 |         +1.4% |  −2.0% / +5.2% |

The paired changes compare the first on/off pair and then the reversed on/off
pair. Six-K improves in both; short and 32K have mixed signs. These are observed
results, not an established portable gain: off controls drift by −8.6%, +2.3%
and −8.7%, respectively. The guard detected no overlapping benchmark script,
but background Node/desktop activity remained. External build processes appear
in 1/33 samples of on-a and 20/35 samples of off-b (mostly waiting Cargo, with
Rust compilation in the final sample). GPU clocks and throttling were not
measured. This cohort is therefore not described as an idle-machine benchmark.
Peak monitored process RSS was 21.00–21.09 GB, with at least 56% available memory.

At the end of this experiment, all five controls stayed off by default. There is no context-length threshold or
device-specific selector inferred from these two samples. The correctness repair
for unsupported shapeless attention is active independently of those switches.
SIMD owner-load/shuffle remains rejected. Larger layout/arena/recorded-execution
changes remain gated on evidence described above. Splash parity is not achieved
or established by this phase.

### Earlier-artifact default-path check

A fresh candidate-9 then candidate-13 comparison, with all five controls off,
retained exact text, token counts and acceptance statistics for all three
1,024-token prompts. Initial decode rates were 38.880 → 31.331 (short),
41.395 → 37.415 (6K) and 18.779 → 19.796 (32K) tokens/s. No other model benchmark
or build appeared in either resource log, but desktop/Node activity remained.
This ordering raises a default-path regression concern for short/6K; it is not
dismissed based on source inspection. The follow-up production-kernel comparison found no broad slowdown in the
sampled default M8 Q5/Q6 kernels: median per-shape C9/C13 timing ratios were
1.0003 and 0.9980. All 64 cases/960 pairs were exact. Four of eight cohorts
included external CPU compilation, and vocabulary-head shapes were omitted;
this diagnostic does not exclude a dispatch, other-kernel or full-model
regression. See the [QMV report](qmv-experiments.md). The reversed short/6K model check is included in
`default-path-comparison.json` alongside the initial results.

The reversed pair produced 38.254 versus 34.245 tokens/s on short and 42.228
versus 36.869 on 6K (candidate-13 versus candidate-9): +11.7%/+14.5%, reversing
the initial −19.4%/−9.6% signs. All ten old/new cases retain exact text, input,
cache, finish and acceptance fields. Neither direction establishes a stable
performance difference: external builds appeared in 6/12 samples of the new
reverse run and 14/14 samples of the old reverse run. There was no observed
competing benchmark script. The initial large slowdown did not repeat, but
default-path performance equivalence remains unproven under this variability.
The conclusion is to retain the established default selections, keep the new
experiments opt-in, and make no overall speedup or Splash-parity claim.

## Archived experiment reproduction

This command applies only to the frozen candidate-13 addon and its matching
metallibs, whose source and binary hashes are in the manifest. The current build
has no such controls or experimental paths. The complete pre-removal sources,
including the research harnesses, are archived under
`.cache/benchmarks/splash-qwen38-phase7/before-{root,mlx}-source.tar.gz`.

```sh
MLX_DFLASH2_VERIFY_OUTPUTS_ONLY=1 \
MLX_DFLASH2_COMPILED_COMMIT=1 \
MLX_DFLASH2_RELEASE_REPLAY_Q=1 \
MLX_COMPILE_INDEXED_REPLAY=1 \
MLX_KQUANT_QMV_WORD_LOADS=1 \
oxnode docs/research/splash-qwen38/benchmark.ts \
  .cache/benchmarks/splash-qwen38-phase6/candidate-13/mlx-core.darwin-arm64.node \
  .cache/benchmarks/splash-qwen38-phase6/reproduction.json \
  dflash short,6k,32k 1 1024
```

The frozen manifest identifies the addon, both metallibs and exact source
snapshots. `validation-summary.json`, `lifecycle-candidate13-parity.json`,
`pairs13r-analysis.json` and the resource logs distinguish correctness results,
measured rates and interference. The first long-pair attempt was excluded after
the process monitor exceeded its output buffer; the monitor was repaired and
the complete cohort rerun under the `pairs13r-` prefix.
