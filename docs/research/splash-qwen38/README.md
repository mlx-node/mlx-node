# Qwen3.8-27B / Splash performance investigation

Investigation date: 2026-09-21. The requested workload is the local
`Qwen3.8-27B-UD-Q4_K_XL.gguf` target with `qwen3.8-27b-dflash2`.

Status, September 23, 2026: cleanup has removed the imported Splash packed-Q4
loader, the DFlash depth override, and manual rollback switches for the retained
eligible optimizations, including two-column GDN and D256 full prefill attention.
Their device/shape guards and FP32 D256 TF32 requirement remain. The experimental
K/IQ residual/SwiGLU epilogues, empty-finalize shortcut and native Metal chunked
GDN experiment are deleted. Eager-commit submission, direct-convolution replay,
conditional pressure-cache trimming and the E48 four-column Qwen override are
also removed; historical measurements remain linked below. Qwen4's separate
four-column route is retained. CUDA/non-Metal chunked ops and their controls
and log-space decay math remain. BF16 remains the default; selecting dense Q8
now always reuses the target head. Full end-to-end screening and final rebuilt
runtime validation are complete. Dense load-time Q4, its private proposal-head
clone and Qwen DFlash adaptive fallback are removed. The [final cleanup report](cleanup-final.md)
records all decisions, tests and the 12-request reversed comparison: median
decode is 38.52 / 43.96 / 21.72 tokens/s on short / 6K / 32K, with exact output
and acceptance parity. Performance is essentially unchanged versus candidate 16;
no new speedup or Splash parity is claimed.

Superseded on October 2, 2026: the draft precision statements above. The draft
now always loads as affine Q4/group64 and reuses the target head. BF16 is no
longer the default, and Q8 is no longer an option. See
[Draft precision](#draft-precision-current).

The [exact Splash Q4 draft follow-up](exact-draft.md) is historical evidence from
candidate 16. Its lossless imported draft improved native 32K decode and reduced
memory, but slowed shorter fixtures; the loader and importer scripts have since
been removed. Frozen artifacts and measurements remain in the local cache.

The [remaining-control audit](flag-audit.md) preserves the pre-cleanup findings
and identifies the completed changes. The earlier [cleanup](cleanup.md)
covered only phase 6, not every previous experiment.

The [final same-device comparison](final-performance.md) measures the cleaned
candidate-14 runtime against local Splash with identical prompt token IDs and
1,024 generated tokens. It supersedes the earlier reference-only comparisons
for the BF16 baseline, while retaining the model-package and timing caveats.

## Draft precision (current)

October 2, 2026. The checkpoint ships BF16. The loader quantizes every dense
draft projection (`fc`, attention, MLP and both convolution kernel projections)
to affine Q4 with group size 64. The selector projection keeps checkpoint
precision. The draft reuses the target output head. There is no precision
switch; `MLX_DFLASH2_DRAFT_QUANT` and the BF16 and Q8 load paths are removed.

Q4/group64 was chosen over BF16 and Q8/group64 on this M5 Max: short / 6K / 32K
fixtures, 1,024 greedy tokens, five fresh processes per arm and fixture in
alternating order. A different draft precision produces a different transcript,
so raw tokens/s compares different texts. The decision therefore uses
teacher-forced (TF) acceptance: every arm verifies the same fixed BF16-draft
transcript through the production verifier. Time per committed token is the
clock-matched E2E time per cycle divided by the TF committed tokens per cycle.

| Fixture | TF acceptance, Q4 / BF16 [95% CI] | ms per committed token, BF16 / Q8 / Q4 | Raw E2E median tok/s, BF16 / Q8 / Q4 | Cycles, BF16 / Q8 / Q4 |
| ------- | --------------------------------: | -------------------------------------: | -----------------------------------: | ---------------------: |
| Short   |              0.996 [0.969, 1.025] |                  23.46 / 23.05 / 22.54 |                42.49 / 43.88 / 52.23 |        274 / 274 / 234 |
| 6K      |              0.995 [0.964, 1.029] |                  20.61 / 19.99 / 19.95 |                47.78 / 30.18 / 36.33 |        216 / 319 / 288 |
| 32K     |              1.009 [0.990, 1.030] |                  36.90 / 36.65 / 35.74 |                24.69 / 28.18 / 26.99 |        322 / 287 / 293 |

- TF acceptance of Q4 and Q8 is within 1% of BF16 on every fixture, and every
  95% CI includes 1.0.
- Q4 has the lowest time per committed token on every fixture. Summed over the
  three fixtures it is 78.23 ms, versus 80.97 ms for BF16 (-3.4%) and 79.69 ms
  for Q8/group64 (-1.8%). Q4 speedup over BF16 [approximate 95% CI]: short
  1.041 [1.009, 1.075], 6K 1.034 [0.982, 1.090], 32K 1.033 [1.000, 1.067].
- Raw E2E tokens/s changes sign by fixture: Q4 versus BF16 is +22.9% / -24.0% /
  +9.3%. Each arm decodes its own deterministic transcript with a different
  cycle count, so the raw numbers mostly measure transcript drift. They are
  reported, not used to decide.
- Clock-matched E2E time per cycle falls by 2.5-4.2 ms with Q4.
- Resident draft memory falls from 3.58 GiB to 1.18 GiB (-2.40 GiB). Load peak is
  unchanged.

Earlier pages reported a 6K drop for Q4 and Q8, for example Q8 6K committed
tokens per cycle 4.736 to 3.081. Teacher forcing shows that drop was transcript
drift: on the same BF16 text, Q8/group32 keeps 4.714 of 4.736 at 6K. The earlier
dense Q4 mode also quantized a private copy of the target output head; the
current draft reuses the target head. Evidence is under
`~/.cache/mlx-node-research/qwen38-arch-2026-09-23/impl/runs/{DRAFT,I8}/`
(`decision.json`, `out/fin-e2e-ab.txt`).

## Comparison boundaries

- Splash source: `7e3c67e8e3a9e9912ff6e02521457017cc4c65d0` in
  `/Users/brooklyn/workspace/github/splash`.
- Original mlx-node main: `784ed808`. Existing optimization branch:
  `377e78f30616b5b78b5fca2dc9f7ba18c1782327` (`perf/qwen35-dispatch-cuts`).
  This investigation builds on that existing work on
  `codex/splash-qwen38-performance`; the incremental change is the diff from
  `377e78f3`, not all of the inherited commits.
- Local machine: Apple M5 Max, 40 GPU cores, 128 GiB, macOS 27.0.
- Target: 17,559,178,144 bytes, SHA-256
  `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`.
- Draft: 3,848,817,896 bytes, 81 BF16 tensors, SHA-256
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.

Splash's pinned README reports 74 tokens/s short-prompt decode, 363 tokens/s
32K prefill, and 282 ms cached 32K first-token latency. Those measurements use
an M5 Pro with 16 GPU cores and 48 GB, Splash's custom Q4 target/draft package,
Q8 paged KV, selected SPEED-Bench HTTP requests, a 1,024-token output limit,
and medium reasoning. Our mixed GGUF target and supplied BF16 draft are not
that package. The results below are local mlx-node comparisons; they do not
establish parity with Splash.

The [follow-up experiments](follow-up.md) cover additional matrix, attention,
recurrent-kernel, draft-precision, and depth investigations. They retain historical Q8 head-sharing and depth measurements, a correction
to permanent AR fallback, and compile-time guards against unsupported NAX tiles. No measured general
throughput improvement establishes Splash parity.

The [same-device matrix investigation](matrix-path.md) implements and tests
eight additional Q5_K/Q4_K Metal matrix variants. All fail the measured
performance gate; the validated runtime is preserved.

The [architectural comparison](architecture.md) examines cycle scheduling,
CPU/GPU synchronization, persistent workspace, SIMD fusion, and repeated
verifier KV-prefix copies. It includes a prioritized implementation plan and
reproducible metadata/trace accounting, without claiming a new runtime gain.

The [architectural implementation results](architecture-implementation.md)
record the subsequent segmented-attention, quantized-epilogue, state-scheduling,
command-submission and allocation-policy experiments, including exact-result
tests, rejected performance candidates and the corrected long-context results.

The [architectural follow-up](architecture-next.md) adds three GPT-6 Astra source
audits and fresh candidate-9 diagnostics. It identifies discarded verifier state
outputs, exact packed-load and SIMD operand-sharing experiments, and the ownership
requirements for a larger native execution plan. New opportunities are separated
from previously implemented or rejected changes; no new speedup is claimed.

The [phase-6 experiment report](architecture-phase6.md) records five optional
paths and their measurements. None established an isolated, repeatable model-level
win. The [cleanup decision](cleanup.md) removes all five paths and switches while
retaining the earlier defaults, runtime capability checks and growing-prefix
attention correctness fix. Phase-6 rates describe archived binaries only.

Initial outcome: implemented and reviewed one fused greedy-selector kernel. Whole-model
median differences were only 0.5–1.3%, within observed run variation. Rejected
KV reservation and Q4 draft experiments did not improve the measured workload.
The Q4 draft result was later reversed; see [Draft precision](#draft-precision-current).

## Source findings and execution plan

| Area                    | Splash source at the pinned revision                                                                                     | mlx-node assessment                                                                                                                                                                                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Proposal selection      | `runtime/metal/kernels/decode/sampling.metal`, `draft_select_top16_sharded`, `draft_select_edges`, `draft_select_dflash` | The existing branch already has fused top-16 selection. Its dependent greedy predecessor walk still constructs repeated gather/reverse/argmax operations. Replace that walk with one dispatch, preserving mlx-node tie semantics.                                                                                                 |
| Verify and commit       | `runtime/model/Runtime.mm:2210-2298`, `encodeBatchAcceptance`, `encodeBatchGdnCommit`, `encodeDraftStateCommitBatch`     | Splash submits draft, verify, acceptance, and state commit in a common command graph. mlx-node still resolves acceptance on the CPU before state commit. This is a larger remaining architectural opportunity.                                                                                                                    |
| Small matrix operations | `runtime/ops/Linear.cpp`, `runtime/metal/kernels/decode/linear_q4.metal`                                                 | Splash specializes packed Q4 kernels for fixed verify and lane widths. The existing branch already adds wide K-quant matrix-vector tiles, merged projections, compiled verify, fused GDN preparation, and residual RMSNorm. Do not reimplement those changes.                                                                     |
| Context and memory      | `runtime/ops/PagedKv.hpp`, `runtime/ops/DraftAttention.cpp`, `runtime/engine/MemoryPlan.cpp`                             | Splash plans paged target KV and persistent draft rings. mlx-node DFlash uses the flat exclusive lane; its existing branch already skips draft-context work outside the retained sliding window. Test reserving only the known target prompt frontier, because MLX buffer donation determines whether reservation actually helps. |
| Recurrent correctness   | `crates/mlx-core/src/models/qwen3_5/dflash2_decode.rs`, `commit`                                                         | Preserve GDN replay even when all proposals are accepted: windowed verification retains FP32 state across the block, whereas serial-equivalent commit must round state per token. Skipping replay changes subsequent outputs.                                                                                                     |

The implemented selector accepts `[1,L,K]` int32 candidates and `[L,K,K]`
float32 scores, runs one dependent walk on the GPU, and returns the selected
path directly to verification. A reverse strict comparison preserves the
existing last-maximum rule, including signed zero and non-finite inputs.
Noncontiguous inputs are handled by MLX's custom-kernel wrapper. Unsupported
contracts and non-Metal execution retain the original lazy path. The manual
reference-path switch has been removed.

## Initial measurement protocol (historical binaries)

Frozen native addons and both companion metallibs are retained under
`.cache/benchmarks/splash-qwen38-20260921/{main,baseline,final}`. Each build
uses `vp run build:native`, not a standalone Cargo build. The artifact directory
also retains the rejected combined experiment under `candidate/` and contains model/environment identities, raw responses, output hashes,
per-sample timing and acceptance counts, build/check logs, process RSS and
system-memory observations.

`benchmark.ts` uses a short coding request (87 input tokens) and retained public
review fixtures (6,219 / 18,754 / 32,488 input tokens). Fixture file:
`.cache/benchmarks/fixtures-public-2026-09-11/qwen38-inputs.json`, SHA-256
`c6c1373328614919617deca668fe67301fe0df853f7780143d9690464148bf5b`.
The fixture is a local prerequisite, not downloaded by the runner.

Each fresh process loads the same target/draft, warms up for 16 tokens, resets
the session cache before each sample, and generates exactly 128 tokens with
greedy sampling, high reasoning, fixed proposal depth seven, no repeat
penalties, and performance reporting. Model load is excluded from timing.
Output equality and acceptance counts accompany performance comparisons.
Timed, uninstrumented model benchmarks run sequentially, without concurrent
builds or test suites. Correctness checks and operation-count traces may
overlap CPU compilation; their timings are excluded from the performance
comparison.

Example from the repository root:

```sh
oxnode docs/research/splash-qwen38/benchmark.ts \
  .cache/benchmarks/splash-qwen38-20260921/final/mlx-core.darwin-arm64.node \
  .cache/benchmarks/splash-qwen38-20260921/example.json \
  dflash short,6k,32k 1 128
oxnode docs/research/splash-qwen38/validate.ts \
  .cache/benchmarks/splash-qwen38-20260921/final/mlx-core.darwin-arm64.node \
  .cache/benchmarks/splash-qwen38-20260921/example-lifecycle.json
```

The session runner covers cold generation, cached continuation, cold replay of
the same conversation, streaming, sampled decoding, reset, and a follow-up
beyond the draft's 2,048-token window. Compare greedy scenarios between builds;
the sampled smoke test has no cross-run equality requirement.

## Initial results (historical binaries)

Three fresh processes per build, ordered baseline/final/final/baseline/baseline/final.
Each row generated 128 tokens. Decode rates below are median [minimum–maximum]
tokens/s; TTFT is median milliseconds. All 18 transcripts and their proposal
acceptance counts match the optimized baseline exactly.

| Input tokens |     Baseline decode |        Final decode | Median change | Baseline / final TTFT |
| -----------: | ------------------: | ------------------: | ------------: | --------------------: |
|           87 | 39.63 [39.34–42.31] | 39.86 [39.82–45.29] |        +0.58% |             182 / 178 |
|        6,219 | 26.34 [26.32–32.36] | 26.51 [26.10–26.86] |        +0.64% |       11,166 / 11,055 |
|       32,488 | 21.51 [21.30–22.65] | 21.80 [21.22–21.94] |        +1.33% |       62,931 / 62,571 |

These small median differences are within the observed variation. They do not
establish a material end-to-end speedup. In particular, unchanged prefill code
slowed during the cohort, so the faster early pilots cannot be attributed to
the patch. A persistent background Node process consumed about one CPU core;
WindowServer and occasional system services were also active. The resource
logs retain that activity. No thermal warning was reported before the cohort,
but GPU clock/temperature telemetry was not captured and the machine was not
otherwise isolated. No model, build, or test job from this task overlapped
these samples.

The final change is therefore a verified reduction in selector overhead, not
evidence that mlx-node has matched Splash. The selector-only and same-process
on/off measurements below distinguish that reduction from whole-model timing.

The historical selector ablation loaded one final-build model, warmed both
selector modes, then alternated fallback/fused, fused/fallback, fallback/fused.
Its runner is archived at
`.cache/benchmarks/splash-qwen38-phase8/selector-ablation-before.ts`; it requires
the earlier binary that still implements the removed reference-path switch. All six requests
produce exactly 1,024 tokens with the same transcript hash
`8593ecd2cdf11d5cff8255bf2cda5b0d81e0534f0f510d8e6c22c7266b478bb8`,
274 cycles, and 3.734 emitted tokens per cycle. Fallback median decode was
30.423 tokens/s [30.134–31.595]; fused was 30.585 [30.551–31.361], a +0.53%
median difference. Individual paired changes were -0.74%, +0.53%, and +1.38%.
This supports the same limited conclusion as the frozen-binary cohort.

The ignored `benchmark_fused_greedy_path_vs_lazy_graph` diagnostic measures
L=7/K=16 graph construction, evaluation, and terminal read, with shared
materialized inputs and alternating 50-iteration batches. In the debug Rust
test harness, median time fell from 899 to 238 microseconds (12 samples).
These are diagnostic host-plus-GPU timings, not production kernel-only latency
or an end-to-end multiplier. Run with `cargo test -p mlx-core --lib
benchmark_fused_greedy_path_vs_lazy_graph -- --ignored --nocapture`; the
repository's Cargo environment serializes Metal tests. Direct test-binary
invocations must explicitly pass `--test-threads=1`.

The final release addon also reduced the instrumented load/warmup/64-token
request from 35,554 to 34,186 evaluated MLX operations with the same output
hash and nine measured decode cycles. Enabling the fallback on the same addon
restored 35,554 operations. The profiler's `accept` time includes deferred GPU
execution of proposal and verification work; it is not a measurement of CPU
acceptance alone. Instrumented rates are excluded from the performance table.

The capacity-reservation experiment was removed from the final change. Its
8,192-row, four-chunk KV operator diagnostic measured unreserved samples of
2.984 / 2.236 / 1.762 / 4.019 ms versus reserved samples of
2.721 / 4.280 / 5.496 / 2.464 ms. It offered no reliable improvement, and the
combined experiment's 32K end-to-end pilot also showed no benefit. MLX
`SliceUpdate` can copy the backing buffer when donation is unavailable; a
larger reserved buffer therefore need not reduce work. The rejected patch and
raw measurements remain in the artifact directory for follow-up.

The since-removed dense Q4 draft mode failed its initial speed test: it saved
about 2.4 GiB of residency, but short/6K decode fell from approximately
44.81 / 33.34 to 31.19 / 22.45 tokens/s. These are single pilots, not final
paired estimates. The short transcript changed. The final change keeps the
supplied BF16 draft and does not alter quantization defaults.

Superseded on October 2, 2026. Those pilots compared different transcripts, and
that mode also quantized a private copy of the output head. A later
teacher-forced study chose affine Q4/group64 with the shared target head as the
only draft precision; see [Draft precision](#draft-precision-current).

## Validation and review

Qwen DFlash2 adaptive fallback has since been removed; both shared depth knobs
are ignored for this fixed-width companion. Native MTP and other draft families
retain their adaptive policies. Earlier adaptive measurements are historical;
see [cleanup-final.md](cleanup-final.md) for decision evidence and validation status.

- Canonical native build passed; final binaries and both metallibs are frozen
  in `final/` in the artifact directory.
- Rust debug suite: 3,806 passed, 116 ignored, 18 filtered. The filtered
  `int8_gemm` module's eight non-ignored tests passed in the release run; its
  ten manual benchmarks were already ignored. Its debug scalar-reference
  matrix loops are prohibitively slow, so that duplicate run was stopped.
- The initial release suite reported 3,815 passed, three failed, 127 ignored
  while the KV experiment was still present. All three failures expected
  debug assertions, which are disabled in release; all three subsequently
  passed in the debug run. No assertion or test was weakened.
- All four new kernel regressions passed in release and debug. The final
  serial diagnostic run passed those four plus the ignored microbenchmark.
  Coverage includes varying lengths, predecessor-dependent routes, ties,
  infinities, signed zero, NaNs, strided inputs, and malformed FFI contracts.
- Final real-model validation preserves all six deterministic baseline
  transcripts and cached-token counts, including streamed output, a cached
  125-token follow-up, and a cached 6,251-token long follow-up. Sampled
  decoding also completed through DFlash verification.
- Independent GPT 5.6 Sol review found no blocking production issue and added
  infinity and adversarial signed-zero regression cases. Its review covered
  the incremental selector change, not the entire inherited optimization
  branch.

- TypeScript session, history, Qwen configuration/generation, streaming, and
  MTP-default contracts: six files, 184 tests passed. This was a focused
  TypeScript run, not the whole monorepo suite.
- Strict all-target Rust Clippy, Rust formatting, TypeScript typecheck, and
  lint/format checks for the new research scripts passed.
- `vp check` remains blocked by the same 26 existing formatting failures
  recorded before edits, including the pre-existing table formatting in
  `docs/perf.md`. Unrelated files were not reformatted.

## Remaining work toward Splash

1. Reduce the cost of target verification for the exact mixed-GGUF matrix
   shapes. Follow-up profiling attributes about 85% of draft-plus-verification
   time to verification; the tested alternative tiles and K-lane widths did
   not improve it. Splash's packed-Q4 timings cannot predict mixed K-quant
   gains.
2. Keep acceptance counts and state-commit control on the GPU. Host acceptance
   arithmetic itself is under 0.3 ms per cycle in the follow-up profile, so the
   benefit would need to come from improved scheduling and fewer fences. This needs a
   device-count replay/commit contract, terminal-token handling, cancellation,
   partial-accept rollback, and continuation parity before it is safe to enable.
3. Add DFlash ownership to paged KV and continuous batching. Test mixed prefill
   and decode, prefix sharing, per-request draft rings, cancellation, and state
   isolation before measuring concurrent throughput.
4. Use matched hardware, model packages, prompts, reasoning settings, output
   limits, cache state, and HTTP concurrency for an actual Splash comparison.
   A faster isolated request does not establish cached-TTFT or server parity.
