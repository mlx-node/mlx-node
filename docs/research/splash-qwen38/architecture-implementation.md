# Architectural optimization experiments

This is the implementation record for the September 22 follow-up to
[architecture.md](architecture.md). The requested mixed-GGUF target and BF16
DFlash2 draft remain the comparison workload. An operator improvement does not
establish an equivalent whole-model gain or parity with Splash.

## Device selection contract

New launch decisions must inspect the running device and compiled pipeline:
SIMD execution width, maximum threads per group, and available/static
threadgroup memory. Workload dimensions constrain eligible variants. The
current machine's model name or locally fastest fixed group count must not
become a universal routing rule.

A kernel's required SIMD width and fixed reduction topology are algorithm
constraints, not evidence that its geometry is optimal on every device.
Reject unsupported configurations and use the established implementation.
Where several numerically valid geometries remain, analytical limits establish
legality; a bounded measurement is needed to establish which is fastest.
Synthetic capability profiles exercise fallback without requiring another Mac.

## Baseline and artifacts

The frozen starting artifact is
`.cache/benchmarks/splash-qwen38-phase2/final-validated/`.
Current experimental logs, resource guards, source snapshots and results live
under `.cache/benchmarks/splash-qwen38-phase4/`.

A fresh 1,024-token baseline completed with exact prior output hashes:

| Prompt                 | Decode tokens/s |   TTFT ms | Cycles |
| ---------------------- | --------------: | --------: | -----: |
| Short, 87 input tokens |         41.7474 |   173.408 |    274 |
| 6K, 6,219 input tokens |         41.0782 | 9,265.275 |    216 |

These single samples establish a fresh reference, not a new performance gain.
Use alternating processes and identical output/acceptance when comparing
candidates. Model load, diagnostic traces, and concurrent builds are excluded
from accepted throughput comparisons.

`compare-runs.mjs` validates checkpoint/configuration, prompt/output hashes,
token counts, cache state and acceptance before reporting timing differences.

## Command diagnostics

`MLX_METAL_COMMAND_TRACE=1` records CPU evaluation/compiled-graph spans,
scheduler backpressure, and completed command-buffer GPU intervals. It adds
logging and callbacks but does not introduce evaluation or timestamp barriers.
Run `analyze-command-trace.mjs TRACE_LOG OUTPUT_JSON` to merge overlapping GPU
intervals rather than double-counting them.

- CPU and GPU absolute timestamps are separate domains and are never subtracted.
- GPU command envelopes include internal waits; their union is not shader busy time.
- CPU evaluation spans include encoding and backpressure, and exclude the final
  synchronous wait in the outer evaluation API.
- `resourceBytes` counts unique bound buffer capacities per command, not memory traffic.
- `sizeUnits` records MLX's existing element-based submission heuristic, not bytes.
- Workload windows use callback arrival order; delayed completions may cross markers.
- Diagnostic throughput must not be mixed with the uninstrumented cohort.

## Candidate work

1. Segmented verifier attention: read prefix and new K/V separately while
   retaining the current causal/reduction contract.
2. Native quantized projection epilogues: preserve BF16 projection rounding
   before residual addition or SwiGLU.
3. Earlier state settlement: submit target and draft state after the host stop
   clamp and before token delivery, with a final durability boundary.
4. Convolution-state replay: omit history concatenation when the accepted
   prefix fully replaces the retained convolution window.
5. Skip genuinely empty Metal finalizations while preserving all event,
   retention, and explicit synchronization work.
6. At the existing 256-token checkpoint, retain allocator cache when runtime
   allocation budgets have sufficient headroom.
7. Use the command trace to assess prepared device commit, reusable execution
   plans, and allocator/submission policy with measured evidence.

The first combined native build passed on September 22. Its immutable artifact
and complete changed-source archives are in `phase4/candidate-4/`. With every
new experiment disabled, the 1,024-token short and 6K outputs and acceptance
counts matched the starting artifact exactly. Rates were 35.79 and 39.74
tokens/s; these samples were separated in time from the starting run and do
not establish a regression or gain under the observed desktop load.

The quantized epilogue experiment intentionally falls back for one input row.
Stock single-row QMV factors its scale/bias arithmetic differently from the
multi-row QMV kernel; algebraic equivalence is insufficient for exact BF16
output. Fusion currently preserves the existing multi-row reduction and
intermediate BF16 casts for rows two through eight.

`MLX_DSPARK_PRESSURE_CACHE_TRIM=1` is a separate, initially disabled experiment.
It checks fallible allocator active/cache/limit counters and the running Metal
device's current allocations and recommended working set. It retains one eighth
of headroom against both the allocator and device budgets, includes external
Metal allocations, and keeps the old trim when a probe is unavailable. Terminal
cleanup is unchanged. This is an allocation-budget heuristic, not an OS-wide
memory-pressure measurement. It was added after `candidate-4`; subsequent
builds and the validation below include it.

## First scheduling comparison

Three fresh processes per arm, ordered off/on/on/off/off/on, generated 1,024
tokens per prompt. The earlier-settlement candidate preserved every output
hash and acceptance count. Six deterministic lifecycle scenarios also matched
the frozen baseline, including streaming and cached continuation beyond the
draft window; sampled generation completed successfully.

| Prompt | Off median [range] tokens/s | On median [range] tokens/s |
| ------ | --------------------------: | -------------------------: |
| Short  |         33.91 [33.62–40.72] |        35.44 [33.47–36.23] |
| 6K     |         39.03 [38.94–44.34] |        40.79 [38.98–42.70] |

This does not establish a robust gain: the first pair favored off, the reversed
pair favored on, and the final pair was effectively tied. The candidate stays
off by default while the architectural experiments continue. Desktop activity
and the pre-existing CPU worker were present throughout; no task-owned build
or second model run overlapped these samples. `pmset` reported no recorded
thermal/performance warning, which is not a measurement of GPU clock stability.

The initial command trace had 19,975 submissions across load, warmup and both
128-token requests. Of 9,023 zero-dispatch submissions, 9,011 also had no encoder,
event or resource work. An opt-in `MLX_METAL_SKIP_EMPTY_FINALIZE=1` experiment
skips only these empty finalizations. It preserves event/fence/retention work
and explicit stream synchronization. The numerical, stream-lifetime and
whole-model checks below passed, but throughput did not establish a benefit.

The first traced requests had GPU command-envelope unions of 2,679.65 ms over
2,830.96 ms for short and 11,630.06 ms over 11,930.31 ms for 6K (including prefill).
These are not request-wall utilization percentages or isolated shader durations.
Measured evaluation spans totaled 13,969.51 ms, including 12,750.39 ms of internal
backpressure. Final synchronous waits are outside those spans. The first trace
logged before scheduler completion notification and can inflate backpressure;
subsequent instrumentation reports after notification/error propagation.

## Candidate correctness and review

The second frozen artifact, `phase4/candidate-5/`, includes allocation-budget
trimming. Its manifest records the addon and both Metal-library hashes, and
its source archives include changed tracked and untracked files. Enabling
segmented attention, both quantized epilogues, direct convolution replay,
empty-finalization skipping and allocation-budget trimming together preserved
the exact 1,024-token short/6K outputs and all acceptance counts. This run
overlapped compilation and is correctness evidence only.

The stream-lifetime stress test passed with empty-finalization skipping enabled
and the command-operation limit forced to one. It exercises cross-stream
dependencies, asynchronous views, dropped owners and allocator clearing. A
paired diagnostic trace reduced command buffers from 20,084 to 10,971;
zero-dispatch buffers fell from 9,128 to 13. Every remaining zero-dispatch
buffer carried completion signals. Dispatch counts were 149,293 and 149,292,
and both traces recorded 85,238 barriers. These include load and warmup and
must not be interpreted as a measured throughput improvement.

Review found two incorrect test assumptions, both corrected: a prefill seed
token is delivered before the first verifier commit, and lazy transpose
metadata need not expose its eventual strides before evaluation. The tests
now verify commit-before-callback ordering after the prefill seed, exact
evaluation of pending views, and rejection of evaluated unsupported layouts.
Hardware-dependent kernel tests explicitly distinguish an unsupported
pipeline fallback from an exercised fused kernel.

Final review restored the original residual-add operand order on both the
ordinary graph and fused path. It also made fusion reuse the existing QMV
selection policy, including `MLX_QMM_SPLITK_MIN_M`, so a requested matrix
reduction cannot silently be replaced with a different vector reduction.
`phase4/candidate-7/` freezes that reviewed implementation. There is no new
device-generation threshold in the epilogue route; it shares the reference
route and additionally validates the actual compiled pipeline capabilities.

Validation:

- 3,823 Rust unit tests passed with debug assertions; 129 manual tests were
  ignored. The two large integer-GEMM CPU reference tests were run and passed
  in release mode instead of repeating their billions of scalar operations
  in an unoptimized build.
- All five quantized epilogue integration tests passed, including 49/49
  residual cases, 49/49 SwiGLU cases, all seven one-row fallbacks, the exhaustive
  BF16 gate lookup, pending/evaluated views, malformed shapes, CPU fallback,
  synthetic device profiles and the explicit matrix-route override.
- All 24 attention tests passed (two manual benchmarks ignored). The revised
  segmented shader matched the existing BF16 result exactly across query
  widths 1–8, 19 prefix lengths including reduction boundaries through 32,769,
  batches, GQA heads and strided production layouts. A required-kernel
  diagnostic prevents silent fallback from passing this GPU test.
- Default and combined candidate options matched six deterministic real-model
  lifecycle scenarios at 32 output tokens, including streaming, cached/cold
  continuation and continuation of the 6K prompt. Sampled generation is a smoke
  test, not a cross-process transcript comparison.
- Type checking, linting, Rust formatting and both repository diff checks passed.
- Strict all-target Rust Clippy passed after replacing two checked `unwrap`
  calls in the optional epilogue wrappers with explicit missing-sidecar
  fallback and removing an identity map in the exhaustive test. These are
  defensive Rust changes; the GPU shaders and default arithmetic are unchanged.
- The workspace suite passed 3,928 tests and skipped 40; one existing dashboard
  packaging test failed because npm 12's workspace JSON is keyed by package
  name while the test assumes an array. The failure reproduced in isolation;
  inspecting the returned file list confirmed the required SPA and assets.
- `vp check` still reports formatting in 26 unrelated, unchanged files.

## Segmented attention measurements

Three fresh processes per arm, ordered on/off/on/off/on/off, gave these
uninstrumented 1,024-token results with exact output and acceptance parity:

| Prompt | Concat median [range] tokens/s | Segmented median [range] tokens/s |
| ------ | -----------------------------: | --------------------------------: |
| Short  |            35.44 [32.22–37.06] |               38.01 [36.01–44.36] |
| 6K     |            39.71 [38.70–44.11] |               42.44 [39.66–44.95] |

The medians improve by about 7%, but ordering and desktop-load variance prevent
a strong short-context speedup claim. At 6K, end-of-request allocator cache
dropped from approximately 3.04 GB to 0.737 GB with unchanged active memory.

The first clean 32K pair improved decode from 11.91 to 21.51 tokens/s and full
request time from 140.55 to 95.12 seconds. Allocator cache dropped from 18.43 GB
to 2.56 GB. Both paths emitted the same 1,024 tokens in 322 verifier cycles.
This is one pair, pending a clean reverse-order repeat. The attempted second
pair overlapped an unrelated native build and is explicitly excluded from
performance claims in `phase4/performance-exclusions.json`.

**The initial 80.6% decode gain did not reproduce in a later quiet comparison.**
Using `candidate-7` for both arms, concatenation delivered 21.73 tokens/s and
the segmented prototype 19.56 tokens/s, with exact output/acceptance parity.
A reverse-order repeat also favored concatenation, 23.14 versus 21.37 tokens/s.
The memory reduction reproduced (18.42 GB versus 2.56 GB allocator cache),
but the original throughput difference is not an established optimization.
The previous paragraph is retained as an observed sample, not an accepted gain.

A rotating-prefix operator test also found the first segmented kernel slower
at 6K and 32K. Inspection found per-key segment selection and dynamic stride
indexing in the hot loop. A follow-up splits the prefix and suffix traversal,
hoists their base addresses, and preserves the exact score order. The revised
shader is frozen in `phase4/candidate-8/`.

### Revised shader and build verification

The revised shader passed the expanded exact-result suite and six deterministic
lifecycle comparisons against the earlier frozen baseline (96-token short
scenarios and 32-token long-context scenarios),
including streaming and cached continuation (125 and 6,251 cached tokens).
The CMake kernel rule now lists `sdpa_segmented.h` as a dependency. Without
that entry, a header-only edit could leave an old Metal library packaged after
an otherwise successful native build. The unchanged hash exposed that issue;
the corrected release Metal library was rebuilt and copied into both native
package directories before tests and measurements were repeated.

`candidate-8/manifest.json` records the addon and both Metal-library hashes;
its source archives include tracked and untracked changes in both repositories.
At measurement time, `candidate8-source-review.json` verified that the packaged
files matched the frozen artifact and the runtime sources matched its source
archives. The final packaged build is `candidate-9`, which adds only the
explicit missing-sidecar fallback and test cleanup from strict Clippy review.
Both Metal-library hashes are identical to `candidate-8`; the performance
measurements below retain their original artifact identity.
The rebuilt `candidate-9` also passed all six lifecycle comparisons with every
new experimental option enabled, preserving output hashes, acceptance and
cached-token counts. Its sampled-generation smoke check passed. See
`candidate9-lifecycle-parity.json` and `candidate9-source-review.json`.

An alternating operator comparison used four independent prefix allocations,
24 timed pairs and exact-result checks before timing:

| Prefix tokens | Concat median ms | Segmented median ms | Time reduction |
| ------------- | ---------------: | ------------------: | -------------: |
| 87            |           0.2473 |              0.2518 |          -1.8% |
| 6,219         |           0.7434 |              0.5676 |          23.6% |
| 32,768        |           2.8894 |              1.9592 |          32.2% |

These measure the operation's host-plus-GPU completion cost, not GPU shader
time or a whole-model speedup. The prefix and suffix loops retain the baseline
online-softmax traversal and intermediate BF16 rounding. Pipeline capabilities
still determine the legal query split; this revision adds no device-name rule
or locally tuned launch geometry.

### Final whole-model comparison and default decision

Two fresh-process pairs used the same `candidate-8` addon and Metal libraries,
with only `MLX_DISABLE_SEGMENTED_VERIFY_SDPA=1` distinguishing the concat arm.
The order was off/on, then on/off. Each process generated 1,024 tokens for
short, 6K and 32K prompts. Both comparisons passed strict prompt, output,
configuration, cache-state and acceptance equality checks.

| Prompt | Pair 1 concat → segmented tokens/s | Pair 1 change | Pair 2 concat → segmented tokens/s | Pair 2 change |
| ------ | ---------------------------------: | ------------: | ---------------------------------: | ------------: |
| Short  |                      23.60 → 23.42 |         -0.8% |                      27.63 → 33.04 |        +19.6% |
| 6K     |                      27.05 → 28.09 |         +3.8% |                      32.37 → 33.37 |         +3.1% |
| 32K    |                      13.72 → 15.20 |        +10.8% |                      16.49 → 17.48 |         +6.0% |

The long-context gain survives the reversed order; the short results are too
variable to claim a gain. Absolute rates and unchanged prefill timings also
vary substantially, so these are observed paired improvements, not a stable
device-speed guarantee. There was no sampled native-build overlap or second
task-owned model job; the desktop and an unrelated one-core CPU worker remained
active. The two final pairs use the revised shader and must not be mixed with
the earlier prototype's negative comparisons or the withdrawn 80.6% claim.

At 6K, end-of-request allocator cache falls from 3.04 GB to 0.737 GB; at 32K,
from 18.43 GB to 2.56 GB (about 86% lower). Active model/cache memory is unchanged.
These are MLX allocator-cache counters, not a corresponding reduction in model
weight size or process RSS. Full-request time improves by 2.6%/6.8% at 6K and
7.3%/1.6% at 32K across the two pairs; prefill variability limits interpretation.

**Keep revised segmented verifier attention enabled by default.** Its exact
results, lower allocator retention, faster isolated operation and repeatable
long-context decode improvement support this decision. The disable flag retains
the old path, and unsupported pipelines retain their existing fallback.
Keep the other experiments below off by default because their performance
gates did not establish a robust benefit.

Evidence: `hoisted-pair-{1,2}-comparison.json`, the matching `.json`, `.log`,
`.job.json` and `.resources.jsonl` files in `phase4/`, and the immutable
`candidate-8/` artifact. All four jobs completed within the memory guard;
the sampled free-memory percentage stayed at or above 63%.

### Net change and remaining Splash gap

A final fresh-process check compared the frozen starting addon directly with
`candidate-8`, both using their default settings and 1,024 output tokens.
Short decode was 34.02 → 34.45 tokens/s; 6K was 39.70 → 40.19 tokens/s.
Every output and acceptance record matched. These roughly 1.2% differences are
within the observed variation; they do not establish a robust aggregate gain
for short/6K against the starting binary. The controlled same-binary attention
comparisons above isolate that kernel change more narrowly.

Splash was rerun from its unchanged local checkout after the attention pairs,
with one fresh server per prompt, a 40 GiB runtime budget and the same 1,024-token
limit. Every prompt's token IDs matched mlx-node exactly; all requests were
uncached and all three reported healthy Metal execution.

| Prompt | Latest mlx-node sample tokens/s | Fresh Splash sample tokens/s | mlx-node / Splash full request seconds |
| ------ | ------------------------------: | ---------------------------: | -------------------------------------: |
| Short  |                           34.45 |                        72.01 |                          29.86 / 14.36 |
| 6K     |                           40.19 |                        88.20 |                          36.01 / 18.70 |
| 32K    |                           17.48 |                        50.27 |                         130.91 / 78.62 |

The short/6K mlx-node samples are the final starting-binary comparison; 32K
is the latest segmented arm of the reversed attention pair. Splash's three
samples finished at 08:46–08:48 UTC; the final native short/6K process finished
at 08:50 UTC on September 22. These are single observations under a variable
desktop load, not a matched-precision engine speed ratio. Splash uses its
custom Q4 target/draft and Q8 KV; mlx-node retains the requested mixed GGUF,
BF16 draft and BF16 KV. Decode timing boundaries differ as documented in
[local-device.md](local-device.md). Splash used 274/206/303 cycles, versus
mlx-node's 274/216/322. Quantization, acceptance and execution all contribute
to the comparison; this experiment does not isolate their separate costs.

Splash parity has **not** been achieved. Raw evidence is retained as
`net-final-comparison.json`, `splash-candidate8-summary.json`, and their source
artifacts and prompt audits. The previous same-device three-repeat medians
remain in `local-device.md`; they are a separate cohort and are not blended
into these results.

## Other candidate screening

Quiet fresh-process 1,024-token runs of `candidate-7` gave these rates. Baseline
runs bracket the candidates; these are screening samples, not universal device
performance claims. Every output and acceptance record matched exactly.

| Candidate                        | Short tokens/s | 6K tokens/s |
| -------------------------------- | -------------: | ----------: |
| Baseline before                  |          33.21 |       37.66 |
| Residual fusion, first run       |          34.61 |       38.97 |
| SwiGLU fusion                    |          31.80 |       34.75 |
| Direct convolution replay        |          33.84 |       37.71 |
| Skip empty finalizations         |          33.51 |       37.62 |
| Allocation-budget cache trimming |          33.22 |       37.54 |
| Baseline after                   |          34.03 |       39.07 |
| Residual fusion, reversed repeat |          34.59 |       39.49 |

Residual fusion's reversed gain shrank to 1.65%/1.07%, and its isolated median
was effectively tied (0.402 versus 0.398 ms). SwiGLU fusion was slower both
in the full model and the rotating-weight dependent MLP chain (0.810 versus
1.068 ms median). The other candidates did not establish a throughput gain.
Reduced trimming retained about 7.70 GB of allocator cache after the short
request versus 1.23 GB normally. These experiments remain off by default.

The combined candidates also passed six deterministic lifecycle scenarios with
321-token short cases and 32-token long-context cases, crossing the periodic
trim boundary and preserving all 350 cached tokens on short continuation.
Default `candidate-7` matched the frozen earlier baseline's lifecycle records
with 96-token short cases. The 320-token fixture was
rejected because trailing whitespace is trimmed by the chat template; it is
not a runtime correctness failure.

## Remaining architectural limits

The new path removes verifier-only prefix concatenation. It does not eliminate
all cache updates, transient allocations, CPU submission or token-ID readback.
Acceptance already transfers only a small ID vector after lazy proposal and
verification; on this unified-memory machine, the large avoidable movement
identified here was GPU-side K/V copying. The commit-provenance change also
reuses the engine's already-read IDs instead of copying the verify IDs again.

The retained GDN replay is required: block verification carries FP32 recurrent
state, while a serial-equivalent committed state rounds per token. Removing
that replay or changing reduction topology can change later tokens even when
the immediate verification result appears correct.

Splash-style persistent execution plans and a device-count state commit remain
larger architectural work. They require explicit buffer ownership, bounded
scratch lifetimes, stop-clamp and cancellation handling, and continuation
validation. These experiments do not implement that redesign or DFlash
continuous batching. A monolithic arena alone would not reproduce Splash's
behavior because MLX currently tracks hazards at whole-buffer granularity.

Runtime capabilities establish legal launch choices, not a mathematically
guaranteed fastest configuration. The new planner computes the supported query
width from execution width, pipeline thread limits and threadgroup-memory
limits, with synthetic-profile fallback tests. The existing SDPA reduction
policy is shared rather than independently retuned, preserving its numerical
contract. No new M5-specific performance table is introduced.
