# Default-path cleanup and full-request validation

Completed September 23, 2026 (Singapore time). Requested defaults and removals
are applied and validated. The final matched comparison preserves output and
acceptance exactly; performance is essentially unchanged. No new speedup or
Splash parity claim is made by this cleanup.

## Requested changes

- Eligible segmented attention, compiled verification, projection merges,
  fused GDN preparation/convolution and draft selectors no longer have manual
  rollback switches. Actual device, dtype, shape, quantization and growing-prefix
  correctness fallbacks remain.
- Q8 draft mode always shares the target head. BF16 remains the normal draft
  precision; selecting Q8 no longer creates a larger Q8 head clone. (Superseded
  later: the draft now always loads as affine Q4/group64 with no precision
  switch; see [flag-audit.md](flag-audit.md).)
- Qwen DFlash2 always takes its full proposal width from the loaded checkpoint.
  The terminal token budget can shorten a block. The shared `mtpDepth` API
  remains meaningful for native MTP and other external-draft families.
- Imported Splash packed-Q4 loader/importer support and experiment-only tests
  are removed. Historical checkpoints, frozen binaries and benchmark results
  remain evidence, not active supported paths.
- GDN snapshots retain immutable state aliases; the removed defensive-copy
  option is no longer described as protection against future in-place mutation.
  Such a change would require an actual independent copy evaluated first.

## Measurement protocol

Cohort A uses frozen candidate-16 and matching Metal libraries, the native
GGUF target and BF16 draft. Each process warms up with 16 tokens, then runs
cache-reset short, 6K and 32K requests with 1,024 generated tokens. All three
requests share the loaded model. Full request time, first-token latency,
decode throughput, output hashes, acceptance and memory are retained.

Competing model and compilation jobs repeatedly interrupted entire arms.
Cohort B therefore uses a separate fresh process for each fixture, with the
same warm-up, reset and generation length, and its own bracketing defaults.
Compare candidates within their cohort; the two protocols are not interchangeable.
The remaining cohort covers dense Q4/Q8, adaptive fallback and shared matrix/
submission overrides. Completed cohort-A candidates without an established
robust gain were rejected rather than promoted; they are not repeated in B.
No measured rate was used to exclude a sample.

The resource guard excludes recognized model/compiler overlap, enforces a
memory bound and retains job/resource records. Recognition is heuristic: a
large Node process is not proof of GPU activity. Some earlier formatter overlap
may have come from this task. Cohort B additionally excludes active TS compilers
and whole Cargo build/test sessions. Normal desktop activity remains a limitation.

Following the explicit GPU-availability instruction, the two completed early
cohort-B defaults are retained under `contended-cohort-b/` but excluded from
performance decisions. The user then specified the device's scheduling criterion:
no external Node process consuming 20 GB or more. Every resumed request checks
that criterion and recognized benchmark/native-test/compiler overlap before
launch; overlapping work during a request invalidates that attempt. Device GPU
utilization is recorded as telemetry, not an additional gate, because the user
clarified that the process criterion is sufficient on this machine. Normal
compositor activity can still vary. Our GPU jobs are serialized, and subagents
are restricted to source work during measurement.

Final rebuilt-default validation used reversed before/after pairs under the
same per-fixture protocol. Exact outputs and acceptance are required for changes
that preserve arithmetic; precision/reduction experiments are labeled separately.
No device-specific override will become a constant because it wins on one device.

Evidence is under `.cache/benchmarks/splash-qwen38-phase8/`: frozen harnesses,
`screen-plan.json`, `remaining-cases-plan.json`, raw responses, job/resource logs,
`screen-summary.json`, `remaining-cases-summary.json`, `screen-decisions.json`
and the protocol-transition records. Original interrupted attempt 1's log and
resources were accidentally overwritten while changing schedulers; its full job
status remains in the archived plan. No accepted measurement was overwritten.

## Decisions completed during screening

Both native K/IQ epilogue experiments were removed with their Rust dispatch,
FFI bridge, C++ implementation, Metal specializations and dedicated experiment
tests. The full-request screen preserved output/acceptance but residual fusion
changed request speed by -9.7% / +1.3% / +2.9% (short / 6K / 32K), and SwiGLU
by -16.6% / -11.2% / -9.9%, relative to bracketing defaults. Residual's mixed
result and earlier reversed comparisons did not justify retaining it. These
screen measurements include baseline drift; they are not isolated causal
estimates. Raw files and deleted source are archived under the phase-8 cache.

Eligible two-column GDN recurrence and NAX D256 attention retain their previous
default behavior without manual rollback switches. Shape/device/dtype/TF32
constraints remain. Qwen4's separate four-column/vector paths remain supported.

Empty-buffer finalization skipping was also removed: full-request speed changed
by -12.3% / -15.4% / -13.4% relative to bracketing defaults, with exact parity.

Early commit submission, direct convolution replay and allocation-budget cache
trim suppression completed all three fixtures with exact parity. Their bracketed
request-speed changes were +0.4/+4.7/+4.8%, +1.7/+4.7/+4.5% and
+3.2/+10.9/+9.4%, respectively. Those observations are not robust causal gains:
the surrounding defaults drifted, earlier comparisons did not establish a
repeatable gain, and trim suppression versus the following default was
-2.2/+3.6/+1.1% while keeping substantially more allocator cache. These uncertain
experiments were removed, preserving the previous default behavior.

Four-column Qwen GDN had mixed full-request results (+11.6/-6.0/-3.2% versus
the preceding default only), so its manual override was removed. Shared
Qwen4 capability-based four-column/vector choices remain. Native Metal chunked
GDN changed output/acceptance and had +3.5/-37.4/-33.8% full-request speed
versus the preceding default only; its implementation and selector were removed.
The separate CUDA/non-Metal chunked-ops path and log-space decay math remain.

Qwen DFlash adaptive fallback was removed after completing the full fixture set.
It produced 36.83 / 18.44 / 25.80 tokens/s, with changed output/acceptance.
The 6K case switched to AR after two speculative cycles and regressed 46.4%
in decode versus the preceding default (38.4% in full-request speed), well
beyond the measured baseline drift. Mixed gains on the other fixtures do not
justify making that behavior the only default. DFlash now always uses the
checkpoint proposal width, subject to the remaining token budget. Shared
native-MTP/other-family adaptive policies are unchanged. Removed source and
decision evidence are archived under `rejected-adaptive/`.

## Completed precision and backend decisions

Cohort B completed all 30 requests under the clarified GPU process gate. Baseline
rates varied materially, so bracketing observations are not isolated causal gains.
Dense Q4's mixed first screen was followed by a complete reverse-order confirmation
(Q4 then BF16 for each fixture):

| Fixture | Q4 decode tok/s | BF16 decode tok/s | Q4 full-request speed change |
| ------- | --------------: | ----------------: | ---------------------------: |
| Short   |           35.65 |             30.85 |                       +15.5% |
| 6K      |           34.05 |             36.16 |                        -4.4% |
| 32K     |           22.66 |             19.10 |                        +8.2% |

The repeated 6K regression prevents sole-default promotion. Dense Q4, its precision
aliases and private draft-head reconstruction/clone plumbing were removed. Output
and acceptance change with draft precision; these are E2E observations, not
arithmetic-parity comparisons. Evidence is in `q4-reverse-summary.json` and
`rejected-dense-q4/`.

Q8 with shared target head completed at 35.33 / 24.22 / 20.40 decode tokens/s.
Its bracketed full-request changes were +2.9% / -24.5% / -2.9%. On 6K, mean
accepted tokens per cycle dropped from BF16's 4.736 to 3.081, and cycles rose
from 216 to 332. BF16 remains the default; Q8 remains an explicit precision choice
with the requested unconditional target-head sharing.

Forced split-K at eight rows changed bracketed full-request speed by
-15.9% / -37.9% / -6.0% and changed output/acceptance. Command-buffer presets
256 and 1024 preserved exact outputs/acceptance but changed full-request speed
by -2.1% / -1.1% / -3.1% and -3.7% / -5.9% / -2.6%, respectively. The 1024
preset also reached 55.4 GB of peak MLX allocator usage on 6K versus about
27.1 GB for the default (including load/warm-up). No Qwen preset was promoted.
The upstream backend's shared controls and automatic heuristics remain available
to other models and diagnostics; they are not dead experiment implementations.

The guard initially waited an extra 60 seconds after recognized interference.
That extra delay was removed before the reverse confirmation and final paired
cohort: it now launches as soon as the process criterion and overlap checks pass.
This changes scheduling, not the measured workload. Desktop/thermal drift remains
a limitation, and only within-cohort comparisons are used for decisions.

## Final validation

The canonical final addon build passed, including declaration consistency and
addon/Metal-library smoke checks. Candidate-17 archives the addon, both Metal
libraries, source snapshot and verified target/draft checksums. A later comparison
found zero changes across its 49 native source entries.

The fresh release unit suite passed 3,819 tests, with 129 normally ignored tests
and four filtered tests. The four filtered cases were executed separately: the
strict fused selector passed in release mode, and three debug-assertion tests
passed in a fresh debug build. Additional GPU checks passed: strict BF16 segmented
attention, two D256 attention operator cases, and the real target/draft cache
transition smoke test. Compiled and eager verifier state/continuation tests passed
for every accepted prefix. The compiled test asserts actual compiled invocation
and builder counts, rather than silently accepting an eager fallback.

Strict selector output confirms native scan and merge execution for F32 at 8,192
vocabulary entries and BF16/F16 at the checkpoint's 248,320 entries. The strict
segmented test passed with the device-probed maximum query length of five.
Unsupported-device fallbacks remain, but this single-device validation does not
prove their performance or execution on another device.

Rust Clippy passed with warnings denied. Rust formatting, scoped formatting and
root/vendor diff checks passed. The type-aware TypeScript lint/type check passed
with 11 existing warnings outside this cleanup. The preflight full TypeScript suite reported
217 files / 3,944 tests passed, one file / one test failed, and 16 files / 40 tests
skipped. The failure is an unrelated dashboard package test: the installed `npm pack --json` returns an object keyed by package name, while
that test assumes an array. This was reproduced independently. The final native suite and real-model checks
cover the subsequently rebuilt runtime; no TypeScript runtime logic changed in
the remaining cleanup. Whole-repository formatting checks found 25 issues
outside this cleanup; those unrelated files were not rewritten.

All final lifecycle checks passed. BF16 matches the retained candidate-16 reference
and Q8 matches a fresh candidate-16 Q8/head-sharing run: six deterministic scenarios
each have exact transcript, token, cache and acceptance parity. BF16 reuses 350
short-prefix tokens and 6,251 long-prefix tokens; Q8 reuses 126 and 6,251. All three
sampled-generation smoke runs pass. These were correctness runs while CPU-only
compilation could continue; no timing claim uses those measurements. Evidence:
`lifecycle-final-summary.json`, `native-tests.json`, `debug-tests-final.json` and
`final-validation.json`, with raw/job/resource records.

Independent source review found and fixed an ignored segmented-attention test
that assumed every device supports an eight-row request split into two chunks.
Both parity and benchmark fixtures now respect the actual pipeline query limit.
The final audit found no remaining task-specific rollback switch or dead helper.
Shared backend controls, diagnostics and other-model paths remain intentionally.

## Final full-request comparison

All 12 requests completed, with no excluded/interrupted attempts. Each fixture
ran `before-a, after-a, after-b, before-b`, using frozen candidate-16 and final
candidate-17, the same live harness, checkpoint identities and prompt inputs.
Each fresh process warms up for 16 tokens, resets its cache and generates 1,024
checked tokens. GPU availability follows the user-defined external Node >=20 GB
criterion, with additional recognized benchmark/compiler overlap checks.

| Fixture (prompt tokens) | Before median decode tok/s | After median decode tok/s | After decode range | Full-request speed change |
| ----------------------- | -------------------------: | ------------------------: | -----------------: | ------------------------: |
| Short (87)              |                      39.00 |                     38.52 |        38.37–38.68 |                    -1.21% |
| 6K (6,219)              |                      44.03 |                     43.96 |        43.66–44.27 |                    -0.32% |
| 32K (32,488)            |                      21.73 |                     21.72 |        21.56–21.88 |                    +0.01% |

Every request preserves exact output hash, token counts, finish reason, cycles,
mean depth, mean accepted tokens and acceptance by position. Cycles remain
274 / 216 / 322 and mean accepted tokens remain 3.7336 / 4.7361 / 3.1770.
The maximum sampled process-tree RSS across these requests was 21.19 GB.

There are only two observations per binary per fixture on one device. Adjacent
full-request comparisons change sign when run in reverse order: short
-2.49% / +0.09%, 6K -1.38% / +0.73%, 32K -1.89% / +1.87%. Together with
baseline drift, this supports preserving performance rather than claiming a
new gain. Ordinary desktop/CPU activity and thermal drift remain limitations.
This final cohort compares cleanup against the existing optimized runtime;
no fresh Splash run was performed or mixed into these results.

Evidence: `final-cases-plan.json`, `final-cases-summary.json`,
`final-paired-summary.json`, `candidate-17/manifest.json` and per-request raw,
job, GPU-preflight, thermal and resource records in the phase-8 cache.

Independent final review recomputed all 12 raw-response hashes, acceptance fields,
medians, ranges and paired percentages. All 334 resource samples and 12 preflight
records showed no detected qualifying GPU/compiler overlap; the largest sampling
gap was 2.321 seconds. Ordinary Node and desktop activity remained. Source and
artifact integrity checks matched both frozen binaries, their Metal libraries,
the installed final addon copies and all 49 recorded native source entries.
