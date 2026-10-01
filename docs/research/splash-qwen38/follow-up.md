# Follow-up experiments

Historical measurements: the Q8 head-clone/reference control and DFlash depth
override described below have since been removed. Dense Q8 now unconditionally
reuses the target head; BF16 remains the default. The later K/IQ residual/SwiGLU
epilogue and empty-finalize experiments are also deleted; their historical
measurements remain in the [control audit](flag-audit.md#removed-epilogue-and-submission-experiments).
Two-column GDN and D256 full prefill attention keep their existing eligible
defaults without rollback switches; device/shape and FP32 D256 TF32 guards remain.
Eager-commit submission, direct-convolution replay, conditional pressure-cache
trimming and the E48 four-column Qwen override are also removed; the separate
Qwen4 four-column route remains. Their historical results are retained in the
[control audit](flag-audit.md#removed-scheduling-replay-and-column-experiments).
Final end-to-end screening keeps BF16 as the default and Q8 as an optional
precision with mandatory head sharing. Dense Q4 and adaptive fallback are removed.
The [final cleanup report](cleanup-final.md) records completed native/lifecycle
validation and the reversed before/after performance comparison.

Superseded on October 2, 2026: every statement on this page that BF16 is the
default, or that Q8 is an option. The draft now always loads as affine
Q4/group64 and reuses the target head; see [Draft precision](README.md#draft-precision-current).

These measurements continue the [initial investigation](README.md) using the
same target GGUF, supplied DFlash2 checkpoint, and M5 Max. Artifacts live under
`.cache/benchmarks/splash-qwen38-phase2`. They do not establish parity with
Splash's different packed-Q4 model package or published workload.

The later [same-device Splash benchmark](local-device.md) measures the actual
local checkout with matched prompt tokens: Splash median decode was 64.9/78.9
tokens/s for short/6K, versus mlx-node's 35.6/41.4 in that repeated cohort.
The report includes full request timing and the model/metric differences.

## Attribution

An instrumented 128-token short request attributed 2,053.6 ms to verification,
354.8 ms to draft construction/evaluation/selection, 8.7 ms to host acceptance,
and 12.2 ms to constructing state commit, over 31 cycles. Verification is about
85% of the measured draft-plus-verify time. Commit GPU work can execute at a
later evaluation boundary, and tracing changes scheduling; these are diagnostic
figures, not an uninstrumented throughput comparison. Moving acceptance alone
to the GPU cannot explain or close the remaining gap.

The 128-token profile emitted 4.10 tokens per cycle; the 1,024-token short
workload later emitted 3.73 per cycle. Output length and acceptance therefore
need to be retained alongside throughput.

The actual eight-row verifier uses the per-step GDN recurrence. The historical
BT32 Metal chunked-prefill experiment required at least 64 rows and was never
the verifier route. That Metal implementation has since been removed; see the
[cleanup report](cleanup-final.md) for the source-frozen cohort-A screening
result and its limits. The separate CUDA/non-Metal `chunked_ops` path, its
`perstep`/`chunked_ops` controls and direct log-space decay math remain.

## Rejected kernel and scheduling changes

| Experiment                            | Evidence                                                                                                                                                                                                                                                                               | Decision                                                                                                                                                                          |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Larger command buffers                | Default, 256 operations / 512 MiB, and 1,024 operations / 4,096 MiB did not show a gain; later arms were slower.                                                                                                                                                                       | Keep defaults. Order/system drift prevents attributing all of the regression to the setting.                                                                                      |
| Force small matrix split-K at M=8     | Short decode fell to about 34.2 tokens/s in the pilot and the transcript changed.                                                                                                                                                                                                      | Keep the existing routing.                                                                                                                                                        |
| BM16 NAX for native K/IQ verification | The initial 16x64 tile selected an unsupported 1x1 fragment shape and silently produced zero accumulators. Its apparent throughput improvement was invalid. Corrected 16x128 tiles passed numerical tests across seven quantization modes but were mostly slower than the vector path. | Remove the experimental routing and retain compile-time guards against unsupported tiles.                                                                                         |
| Wider K lanes in the vector kernel    | Paired C++ tests covered M=7, four formats, individual/merged FFN projections, down projection, and the Q6 vocabulary head. K-lanes 16 lost on 10/13 cases; its three gains were only 0.7–1.5%. K-lanes 32 lost everywhere.                                                            | Keep K-lanes 8. Archive the complete experiment in `rejected-wide-kl/`.                                                                                                           |
| Persistent draft attention ring       | Exact-source Python kernel tests covered sparse, nearly full, and wrapped rings, with adversarial oldest-row masking. Full-window append-plus-attention was 0.4329 ms versus 0.3917 ms for the existing path, about 10.5% slower.                                                      | Remove ring production code and archive its patch, kernels, tests, and harness in `rejected-ring/`.                                                                               |
| Four-column GDN recurrence            | Sixteen BF16/F32 fixtures had exact output/state parity. A 48-kernel dependent chain improved from 27.832 to 26.905 microseconds per kernel, while synchronized calls regressed from 229.544 to 238.320 microseconds.                                                                  | The E48 Qwen override has since been removed; the separate Qwen4 route remains. The chain saving is under 0.05 ms across 48 layers and did not justify a whole-model speed claim. |

A final SIMDgroup experiment kept K-lanes eight and changed only threadgroup
grouping from two SIMDgroups to four or eight. Both complete sweeps passed
132 bit-exact BF16 comparisons, covering all seven native K/IQ formats,
M=6/7/8, individual/down/merged projections, and the Q6 vocabulary head.
For the actual M=8 verifier, SG4's median ratios were only 1.008 and 1.010;
SG8's were 0.995 and 0.989. The M=8 Q6 head regressed with both settings in
both sweeps. The experiment was removed; its apply-checked patch, sources,
bundled-backend harness, logs, and summary are in `rejected-wide-simdgroups/`.

The Python operator harnesses use installed MLX
`0.31.1.dev20260308+1bac4583`, not the bundled backend revision. They screen
the exact shader bodies; they are not substitutes for native-addon validation.
The corrected NAX parity test used the bundled release backend. The wide-lane
matrix harness linked the bundled static library and explicitly selected its
matching metallib.

## Draft precision and depth

The dense Q4 option and its private proposal-head clone have since been removed.
BF16 remains the default; optional Q8 always shares the target head. Historical
precision measurements below are not results from the cleanup binary.

Superseded on October 2, 2026. The comparisons below use E2E runs, and each
precision decodes a different transcript. A later teacher-forced study found Q4
and Q8 acceptance within 1% of BF16 and chose affine Q4/group64 with the shared
target head as the only draft precision. See [Draft precision](README.md#draft-precision-current).

Q8 draft projections plus the existing Q8 head clone were measured in four
fresh processes ordered BF16/Q8/Q8/BF16. For 256 output tokens, median short
decode was 29.409 versus 31.095 tokens/s (+5.7%); 6,219-token context was
27.136 versus 27.500 (+1.3%). Short transcripts matched, while the longer
context transcript changed. This is not enough evidence to replace the
supplied BF16 draft by default.

The historical Q8 path also requantized the target's Q6 output head into a larger affine
Q8 clone: approximately 1.043 GB becomes 1.430 GB. This adds resident storage
and weight traffic despite the reduction in draft projection size. The
historical experiment compared that clone against reuse of the immutable target
head. Current Q8 always reuses the target head; the extra head-reuse switch and
Q8 clone branch have been removed. BF16 remains the default, and dense draft
precision must still be selected before loading the model.

Four fresh 256-token runs used reuse/clone/clone/reuse order. Reuse improved
short decode by 8.25% and 10.15% in the two pairs, and 6K-context decode by
7.62% and 12.25%. It saved 1.332 GiB of active MLX memory and 2.100 GiB of
peak MLX memory relative to the Q8 clone. Process RSS did not clearly
separate the arms, so these are allocator figures rather than RSS claims.

Much of the throughput improvement came from better proposals: short cycles
fell from 77 to 72 and 6K cycles from 76 to 73. The short transcript changed;
the 6K transcript was identical. Estimated whole-cycle time improved by only
1.2–7.3%, with system variation still present. This is a configuration-level
result, not a claim that the head kernel itself is 8–12% faster.

The more representative 1,024-token comparison against the default BF16
companion reverses the apparent benefit. Four fresh processes ran
BF16/reuse/reuse/BF16 with the same fixed depth seven:

| Input        | BF16 median [range], tokens/s | Q8 with reused head median [range], tokens/s | Change |
| ------------ | ----------------------------- | -------------------------------------------- | ------ |
| 87 tokens    | 39.460 [36.753–42.167]        | 34.626 [33.716–35.536]                       | −12.3% |
| 6,219 tokens | 43.326 [41.405–45.247]        | 28.386 [27.929–28.843]                       | −34.5% |

Transcripts differ across precision modes and repeat exactly within each arm.
Short cycles increase from 274 to 309; 6K cycles increase from 216 to 332.
Head reuse improved the historical Q8 configuration in the short pilot, but
this **does not make Q8 faster than BF16 for these long responses**. Keep BF16
as the recommendation for sustained decode. Current Q8 always avoids the larger head clone. This changes head ownership
within Q8, not the default draft precision or the general throughput conclusion.

The later 1,024-token Q8-clone check repeated each prompt twice and measured
36.798–37.042 tokens/s for the short prompt (272 cycles) and 27.710–28.134
for 6K (326 cycles). Each prompt repeated its transcript exactly. This was a
separately ordered check, not an alternating reuse/clone comparison, so it
does not support a paired speed estimate. It also shows that the reuse mode's
better acceptance in the 256-token pilot does not persist on longer outputs:
reuse needed 309 and 332 cycles, respectively. The memory saving is the
reliable benefit retained by unconditional head reuse within Q8.

The completed depth comparison generated 1,024 tokens per request, with three
alternating-order pairs in one process (`depth-long.json`). Each arm had a
stable transcript and cycle count across its three runs, but the arms produced
different transcripts.

| Input        | Depth seven median [range], tokens/s | Depth three median [range], tokens/s | Change |
| ------------ | ------------------------------------ | ------------------------------------ | ------ |
| 87 tokens    | 29.203 [29.098–29.362]               | 33.524 [33.303–33.532]               | +14.8% |
| 6,219 tokens | 34.004 [33.859–34.031]               | 26.552 [26.354–26.687]               | −21.9% |

Short-prompt cycles increased from 274 to 328 at depth three, but each cycle
was cheaper. The longer task increased from 216 to 391 cycles, overwhelming
that saving. The depth-three override has since been removed: DFlash now uses
its checkpoint proposal width, seven here. The shared `mtpDepth` option remains
available for native MTP, not DFlash. These historical depth changes were
scheduling experiments rather than byte-equivalent kernel substitutions.
Per-arm hashes, configurations and acceptance counts remain in
`.cache/benchmarks/splash-qwen38-phase2/depth-long.json`; the removed tuning
runner is archived at
`.cache/benchmarks/splash-qwen38-phase8/tuning-ablation-before.ts` and requires
the matching historical binary.

The since-removed DFlash2 adaptive option did not search these depths. It routed through
the DSpark loop and compares full-depth speculation with target-only AR,
using one AR probe and two speculative probes. The separate native-MTP
policy's five-token maximum does not apply. The early adaptive pilots kept
full speculation (mean depth 6.905), but needed 84 cycles versus 77 for fixed
depth seven and were slower. Their transcripts differed after the AR probe.
That historical adaptive mode could not discover the depth-three result;
there is no evidence for a generic depth-limit change or automatic prompt-
length rule.

## Permanent AR fallback (historical)

Qwen DFlash2 adaptive fallback and its private AR helpers have since been
removed. The implementation, tests, and results below describe historical
binaries, not current source or runnable validation. Both shared depth knobs
are ignored for this companion; other draft families and native MTP are
unchanged. See [cleanup-final.md](cleanup-final.md) for the decision evidence.

The adaptive-path audit found an actual Qwen DFlash2 mismatch. Calibration
used the lightweight one-anchor target forward, but after choosing permanent
AR fallback the stepper returned to its ordinary speculative verification
path: target-cache snapshot, GDN tape recording, and replay. The new turn-local
fallback latch routes one-anchor verification and commit through the existing
lightweight helpers instead. Draft proposals, device verification, multi-row
verification, and non-one-row commits are rejected after the transition.
The next turn constructs a fresh stepper with fallback disabled.

Draft-context fusion and append remain intact for continuation. Calibration's
AR timer still ends before that work, and a fallback response can leave some
of it lazy until the next proposal. This patch does not make the probe and
steady-state costs identical. `fallback-ablation.ts` therefore records both
the first response and the following cached turn's wall time, as well as
first-token latency, streamed output, and cold replay of that conversation.

The deterministic tiny-model regression compares three fallback cycles with
the previous single-row snapshot/tape/replay path. All logits, target-cache
arrays, draft-cache arrays, frontiers, token histories, and the next ordinary
draft proposal match exactly. It also checks clean transition and invalid-call
contracts, and asserts that fallback constructs neither snapshot nor tape.
The focused DFlash2 suite passes 17 tests with two manual tests ignored.

The real-model fixture asks for remainders modulo 37 of 96 deterministic
integers, generates 128 tokens with reasoning disabled, and enables the
adaptive option. On this M5 Max, it makes two speculative attempts before
permanently falling back. Its following 64-token turn reuses 751 cached tokens
and resumes normal drafting. The runner asserts that fallback actually occurs;
hardware-dependent calibration may make that fixture inapplicable elsewhere.
This diagnostic is separate from the fixed-depth coding throughput cohort.

Four fresh processes ran before/after/after/before with a 32-token warmup.
All four scenarios repeated exactly across both binaries: first response,
streaming, cached continuation, and cold replay. Both builds retained the same
two speculative calibration cycles and the same 751-token cached prefix.

| Metric                                        | Before median [range]     | After median [range]      |
| --------------------------------------------- | ------------------------- | ------------------------- |
| Fallback decode, tokens/s                     | 23.448 [23.226–23.670]    | 23.642 [23.603–23.681]    |
| First response + cached continuation, seconds | 7.934 [7.912–7.955]       | 7.832 [7.822–7.842]       |
| Cached continuation first-token latency, ms   | 120.774 [118.603–122.946] | 119.803 [119.056–120.551] |

The decode median rose 0.83%; combined two-turn wall time fell 1.28%, with
paired reductions of 0.89% and 1.68%. These are small changes within observed
system variation, not evidence of a material throughput improvement. The
retained change removes unnecessary rollback work and closes the mismatch
between the probe's target path and permanent fallback. It does not establish
Splash parity. Raw results, hashes, and wall-time calculations are retained in
`ar-fallback-pair-*.json` and `ar-fallback-summary.json`. The final addon and
matching metallibs are frozen in `final-validated/`.

## Measurement limits

Only one model benchmark or GPU test from this task runs at a time; timed
model runs exclude full native builds. A small compile-only NAX assertion
probe overlapped the exploratory depth cohort; its production-kernel compile
succeeded and its unsupported tile failed at the intended assertion. Background
applications remained active.
An unrelated Node process used about one CPU core; WindowServer, indexing,
and other system activity varied. Even unchanged baseline short decode moved
from about 39 to 24 tokens/s during this continuation. No thermal warning was
reported, but GPU clock and temperature telemetry were not captured.

Each guarded job retains its exact command, raw output, process RSS, available
memory, and background CPU observations. New research runners record an
allowlist of relevant performance environment variables. Do not compare
isolated early and late pilots as if they were a controlled speedup.

## Correctness and review of head reuse

The default configuration matches the previous frozen build in all six
deterministic lifecycle scenarios and cached-token counts. The Q8 clone also
passes the original 96-token fixture. Head reuse passed cold generation,
streaming equality, sampled acceptance, continuation with 126 cached tokens,
and a long follow-up with 6,251 cached tokens, using the 97-token fixture.

The original 96-token reuse run exposed a fixture boundary limitation. Its
response ends in `this.items.push(item);\n `, and Qwen's chat template trims
that whitespace when rendering the next turn. The completed history no
longer contains the exact cached text prefix, so the session deliberately
falls back to cold prefill. The failed run and its intermediate responses
are retained, not reported as a passing cache-hit test. A separate fresh
cold generation reproduced the fallback transcript exactly. The 97-token
response ends in `}`, satisfies the validator's explicit non-whitespace
boundary check, and exercises successful reuse. No production cache matching
was weakened to accept a changed prefix.

The canonical native build passed, as did 13 focused Rust tests (two manual
tests ignored), strict all-target Rust Clippy, 184 session/configuration/history/streaming tests, and the
positive/negative compile checks for both NAX helper copies. The default and
Q8 correctness runs overlapped test compilation; their timing is excluded.
Independent code review found no blocker in target-head selection,
sampled proposal probabilities, immutable sharing, or residency accounting.

The final fallback change additionally passed the 17-test DFlash2 suite and
30 speculative-engine tests, including stop and token-budget boundaries.
Strict all-target Clippy passed after replacing two panic-based error paths
with fallible returns. Independent review found no fallback correctness
blocker and identified the deferred-context timing caveat addressed by the
two-turn benchmark above.
