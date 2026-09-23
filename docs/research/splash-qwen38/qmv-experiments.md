# Native QMV load experiments

**Historical experiment report.** Aligned-word/owner-load implementations and
their research harnesses were removed after they did not establish a repeatable
model-level win. Commands and source paths below describe the archived sources
in `.cache/benchmarks/splash-qwen38-phase7/before-{root,mlx}-source.tar.gz`; they
do not apply to the cleaned checkout. See [the cleanup decision](cleanup.md).

This phase-6 prototype isolates two controls in the existing K/IQ QMV body:

- `words`: Q5_K/Q6_K group-aligned uint32 loads and exact integer extraction.
- `owner`: one activation load per K lane, broadcast to the other output-row
  lanes in the same SIMD group.
- `both`: the combination, measured separately from either individual change.

In the archived build, both template controls defaulted to false. Its production
dispatcher exposed only the words control through `MLX_KQUANT_QMV_WORD_LOADS=1`, set before the
first quantized projection. It is restricted to BF16, non-batched Q5/Q6 with
M=2..8 and whole GGUF superblocks. It reuses the existing QMV row selection;
M=1, batched operations, other formats/dtypes and matrix routes retain their
baseline. Missing or unsupported experimental pipelines safely fall back to
the already-loaded baseline. Owner loads have no production route.
There is no new device-name, core-count or speed threshold.
The fixed 32-lane/two-group mapping is the existing algorithm's reduction
contract; the harness verifies it against the actual compiled pipeline.

The FP32 weight-decode expression, eight-value subchunk accumulation, group
order, shuffle reduction and BF16 output cast remain in their original order.
The extraction helper has CPU ASan/UBSan coverage: every possible Q5/Q6 code at
every group position with zero and maximum neighbors, plus 200,000 mixed groups.
Those tests and the Objective-C harness's warnings-as-errors syntax check pass.
Standalone Metal and Objective-C builds also pass with warnings as errors.

The coordinated small GPU suite passed 147 format/shape/M cases (all seven
formats, three shapes and M=2..8), covering 231 candidate configurations with
two independent fixtures each. Every default-wrapper output matched frozen
candidate-9; every candidate matched it exactly, including repeated poisoned
outputs, finiteness and tail guards. Evidence is
`.cache/benchmarks/splash-qwen38-phase6/qmv/small-parity-build-overlap.jsonl`.
Its 693 emitted timing pairs overlapped the root native build and are excluded
from every performance claim. A second owner-load implementation moved the
divergent branch outside the element/shuffle loop and passed the same complete
small suite (`small-parity-owner-chunk-build-overlap.jsonl`). The first library,
executable, source snapshots and AIR dump remain in `owner-element-prototype/`.

Both versions also passed exact qualification on eight production projection
shapes for Q5/Q6, with one to four fixtures under the 256 MiB cap.
Both head cases skipped explicitly. These screens overlapped Rust compilation
and are diagnostic, not promotion evidence. The first owner implementation's
per-shape median ratios were 0.038–0.148x for Q5 and 0.017–0.077x for Q6;
moving its branch raised them to 0.061–0.182x and 0.021–0.107x respectively.
Combining owner loads with word extraction was also consistently much slower.
Neither owner implementation should be selected in production.

Words-only Q5 had ratios 0.897–1.154x in the first screen and 1.010–1.162x in
the second. Q6 had 0.843–1.100x and 0.986–1.110x. These mixed/noisy records
justify quiet words-only repeats before any production decision. They do not
justify a layout sidecar or software-pipelining expansion. The raw files are
`q{5,6}-shapes-build-overlap.jsonl` and
`q{5,6}-shapes-owner-chunk-build-overlap.jsonl` under the phase-6 `qmv/` directory.
All their timings are excluded from accepted performance comparisons.

The initial AIR dump shows a conditional activation load inside the inner
element loop before the shuffle. Moving that branch improved its observed
diagnostic runtime, but did not make it competitive. AIR is intermediate code,
not final GPU instruction or spill/occupancy evidence. Every measured pipeline
reported execution width 32, maximum threads 1024 and zero static threadgroup
memory, despite these huge timing differences. Those properties cannot explain
the regression or predict fastest execution by themselves.

## Quiet production-kernel screening

After the task's builds, model jobs and other agent's host benchmark stopped,
the harness directly called candidate-11's production word-load symbols and
compared them with frozen candidate-9's baseline symbols. It also required the
standalone default wrapper to match candidate-9 exactly. Eight fresh processes
covered both formats and requested rings one/four, then repeated in reverse
process order; each shape had four warmups and 15 alternating timing pairs.
All 64 production candidate cases and 960 timing pairs preserved exact outputs,
finiteness and tail guards. There was no task build/model overlap. Desktop
activity and the pre-existing one-core Node worker remained uncontrolled.

Ratios below are paired median baseline/candidate GPU command times; above one
is faster. Two values denote the first and reversed process. They are operator
ratios, not whole-model throughput or measured bandwidth. Rotation means the
actual admitted number of independent matrices, not an asserted cold cache.

| Format / shape | Reused matrix | Bounded rotation |       Actual rotated count |
| -------------- | ------------: | ---------------: | -------------------------: |
| Q5 gdn_qkv     | 1.051 / 1.068 |    1.074 / 1.059 |                          4 |
| Q5 gdn_z       | 0.999 / 0.980 |    1.059 / 1.009 |                          4 |
| Q5 gdn_out     | 1.020 / 1.008 |    1.024 / 1.018 |                          4 |
| Q5 attn_q      | 1.066 / 1.057 |    0.992 / 0.992 |                          4 |
| Q5 down        | 1.022 / 1.019 |    1.082 / 1.047 |                          4 |
| Q5 up          | 1.047 / 1.073 |    1.060 / 1.050 |                          4 |
| Q5 attn_k      | 1.098 / 1.090 |    1.122 / 1.172 |                          4 |
| Q5 gate_up     | 1.039 / 0.936 |    1.095 / 1.050 |                          2 |
| Q6 gdn_qkv     | 1.044 / 1.042 |    1.055 / 1.066 |                          4 |
| Q6 gdn_z       | 1.056 / 1.048 |    1.025 / 1.034 |                          4 |
| Q6 gdn_out     | 1.040 / 1.050 |    1.045 / 1.037 |                          4 |
| Q6 attn_q      | 1.029 / 1.043 |    1.030 / 0.991 |                          4 |
| Q6 down        | 1.039 / 1.034 |    1.045 / 1.024 |                          3 |
| Q6 up          | 1.057 / 1.063 |    1.072 / 0.991 |                          3 |
| Q6 gate_up     | 1.031 / 1.132 |    1.038 / 1.026 | 1 (budget-limited; reused) |
| Q6 attn_k      | 1.043 / 1.052 |    1.062 / 1.056 |                          4 |

Both head shapes skipped under the fixture budget in every process. Q6's
gate/up did not rotate because only one matrix fit; that column must not be
read as evidence of a cache-independent gain. The 256 MiB cap was respected.

These results justify a real-model off/on experiment, not default promotion
or a table of current-device shape thresholds. The runtime adds **no automatic
best-value tuning**: it reuses the existing launch geometry, exposes an explicit
off-by-default candidate and checks actual pipeline capabilities only for
legality. A supported pipeline is not presumed faster. The source changes do
not add an M5 name, core-count rule or device-speed constant.

Evidence: `quiet-summary.json`, `quiet-c11-*-ring*.jsonl`,
`quiet-repeat-c11-*-ring*.jsonl`, matching job records/stderr and the recorded
baseline/production/harness SHA-256 values in the phase-6 `qmv/` directory.
The executable used here is `qmv-experiments-production`; its additional
`--production-lib .cache/benchmarks/splash-qwen38-phase6/candidate-11/mlx.metallib`
argument selects actual production word kernels and times them against the
frozen baseline. It does not silently substitute the standalone word prototype.

The quiet screen above did not measure C11's default production symbols.
For a subsequent default-path regression check, the harness also accepts
`--variant default --production-lib PATH`. This loads the unchanged production
symbol name from both the frozen baseline and the selected production library,
then applies the same exact qualification and alternating timing. It requires
an explicit production library and emits `production_default_kernel: true`.
The standalone default wrapper remains an untimed exactness check. This mode
needs only a host-harness rebuild; retain the frozen shader libraries. Neither
an output match nor a disabled `if constexpr` branch proves equal compiled
register allocation or default-path performance. The mode's availability is
not evidence of a measured result.

### C13 default production regression check

The subsequent targeted check directly measured C13's default production
symbols against frozen C9's default symbols. It used the same eight M=8
projection cases, 15 alternating pairs, four warmups and requested rings one
and four, followed by reversed process order: eight fresh-process cohorts.
All 64 candidate cases and 960
pairs were exactly equal with finite outputs and intact tail guards. Head
shapes still skipped; actual bounded ring counts remain those in the table
above. The standalone wrapper was only an untimed qualification step.

These are diagnostic measurements: no competing GPU workload was observed,
but four of eight cohorts sampled unrelated Cargo/compiler processes. Every
cohort retained uncontrolled desktop activity and the existing Node worker.
The initial preflight waited through an unrelated CPU build before GPU work.
Process snapshots are retained with each cohort; absence from the sampled
process list is not proof of absence between samples.

| Format / requested ring | Median of eight per-shape paired ratios, first / repeat | Per-shape range, first / repeat | CPU build observed, first / repeat |
| ----------------------- | ------------------------------------------------------: | ------------------------------: | ---------------------------------- |
| Q5 / 1                  |                                         1.0039 / 0.9991 |   0.9167–1.0085 / 0.9957–1.0062 | no / yes                           |
| Q5 / 4                  |                                         1.0032 / 0.9989 |   0.8995–1.0707 / 0.9817–1.0670 | no / yes                           |
| Q6 / 1                  |                                         1.0031 / 1.0063 |   0.9884–1.0769 / 0.9864–1.1062 | no / no                            |
| Q6 / 4                  |                                         0.9960 / 0.9952 |   0.9842–1.0132 / 0.9708–1.0013 | yes / yes                          |

Ratios are C9/C13 GPU command times, so below one means C13 is slower.
There is no broad sampled M=8 Q5/Q6 default-kernel regression of the size
suggested by the separate model-level investigation. Q5's initially slow
`gdn_qkv` and `up` cases did not repeat. Q6 `attn_k` was consistently slightly
slower (0.9864–0.9960 across four cohorts); Q6 `up` with three admitted rotating
matrices was 0.9870/0.9708, while reused `up` was 1.0109/1.0757. These patterns
do not establish a general default penalty or justify tuning thresholds.
The result does not qualify CPU dispatch overhead, M=1, other formats,
matrix kernels, graph scheduling or dependent whole-model execution.

Evidence is `default-c13-summary.json`, `default-c13-sha256.json`,
`default-c13-*.jsonl`, `default-repeat-c13-*.jsonl` and corresponding job records
under the phase-6 `qmv/` directory. Only the Objective-C harness was rebuilt;
no runtime or shader source was changed for this regression check.

```sh
xcrun clang++ -std=c++20 -O3 -Wall -Wextra -Werror -fobjc-arc \
  -mmacosx-version-min=26.0 docs/research/splash-qwen38/qmv-experiments.mm \
  -framework Metal -framework Foundation \
  -o .cache/benchmarks/splash-qwen38-phase6/qmv/qmv-experiments-default
.cache/benchmarks/splash-qwen38-phase6/qmv/qmv-experiments-default \
  .cache/benchmarks/splash-qwen38-phase4/candidate-9/mlx.metallib \
  .cache/benchmarks/splash-qwen38-phase6/qmv/experiments.metallib \
  --production-lib .cache/benchmarks/splash-qwen38-phase6/candidate-13/mlx.metallib \
  --suite shapes --mode q5k --variant default --ring 1 --max-mib 256 \
  --pairs 15 --warmups 4
```

Repeat with Q6 and requested ring four, then reverse process order. Coordinate
all executions with the task GPU slot; record CPU contention separately.

Byte-preserving layout and look-ahead remain deferred. The modest word-load
gain isolates code extraction/load scheduling at identical stored bytes; it
does not establish inefficient physical transactions or a long dependency
stall that either more invasive approach would remedy. Require a whole-model
gain first, then actual compiler/profiler evidence or a controlled reused versus
rotating comparison that identifies the next bottleneck. At that point, test
only one bounded single-matrix layout or one-chunk look-ahead at a time, keeping
GGUF parameters and reduction order unchanged. No model-wide sidecar is justified.

Run from the repository root:

```sh
sh docs/research/splash-qwen38/build-qmv-experiments.sh cpu
sh docs/research/splash-qwen38/build-qmv-experiments.sh build
.cache/benchmarks/splash-qwen38-phase6/qmv/qmv-experiments \
  .cache/benchmarks/splash-qwen38-phase4/candidate-9/mlx.metallib \
  .cache/benchmarks/splash-qwen38-phase6/qmv/experiments.metallib \
  --suite small --mode all --variant all --ring 2 --pairs 3 --warmups 1
```

Coordinate the `build` command and all GPU runs with the single task benchmark
slot. The harness builds without linking another MLX runtime and invokes both
the frozen production specialization and experimental kernels through the same
Metal queue. The default wrapper must match the frozen library exactly before
any candidate is measured. All outputs must be finite, fully written and
bit-identical; a guard beyond the final output catches tail stores. Each timed
arm is poisoned and validated outside its measurement boundary.

The small suite covers M=2..8, N tails, multiple superblocks and the existing
input-row tile-cap boundary for all seven native K/IQ modes. `words`/`both`
exist only for Q5/Q6; `owner` covers all seven. M=1 remains outside this QMV-wide
experiment because production uses a different arithmetic path. These tests
exercise contiguous, non-batched matrices; batched/strided integration is not
claimed or enabled.

After parity passes, run alternating fresh processes with ring one and a
bounded rotation, for example:

```sh
.cache/benchmarks/splash-qwen38-phase6/qmv/qmv-experiments \
  .cache/benchmarks/splash-qwen38-phase4/candidate-9/mlx.metallib \
  .cache/benchmarks/splash-qwen38-phase6/qmv/experiments.metallib \
  --suite shapes --mode q5k --variant all --ring 4 --max-mib 256 --pairs 15 --warmups 4
```

Repeat Q6 and any relevant owner-only modes. The shape suite includes the
recorded projection dimensions, a merged gate/up shape and the vocabulary
shape. The latter explicitly skips under the default 256 MiB fixture cap;
do not interpret a skip as validation. Ring size is reduced to fit the budget,
and the actual admitted count/bytes are printed. A second guard reserves 20%
of the running device's recommended allocation budget; this is not an OS-wide
memory-pressure measurement and does not replace the outer task memory guard.
The maximum allowed fixture budget is 2 GiB, never a model-wide sidecar.

JSON lines record pipeline capabilities, exact parity, every alternating timing
pair, GPU command time and CPU wall time, median ratio, fixture admission and
skips. `sha256.txt` captures the executable, candidate library and kernel/harness
sources at build time. Record the frozen baseline hash and surrounding resource
logs alongside each accepted run.

Fixtures are synthetic codes/metadata with real projection dimensions. A ring
contains independent matrices and outputs, not a dependent transformer chain.
Reused/rotating comparisons are screening evidence, not measured DRAM traffic,
cache capacity, instruction issue rate or occupancy. Maximum legal threads is
not a measured register count. Owner-load shuffle can lose to existing
hardware broadcast/cache behavior; aligned-word loads can lose to increased
live registers or duplicate compiler optimizations.

The explicit opt-in is for real-model experiments, not a default speed choice.
No automatic measured selector, default promotion or model-wide repack is
warranted unless isolated gains survive rotating weights, dependent chains,
exact real-model output/acceptance and continuation tests. Byte-preserving layout
and look-ahead remain conditional on evidence from these smaller experiments.
