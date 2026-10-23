# Ornith 35B MoE comparison with Splash

Use the same `Ornith-1.5-35B-Q4_K_M.gguf` in both engines. Its Qwen3.5 MoE
configuration has 40 layers, hidden width 2048, 256 experts, top-8 routing,
512-wide experts, 16 attention heads and two KV heads of width 256. Splash
calls this architecture `Qwen3.6-35B-A3B`.

The matching official companion is `incoai/Qwen3.6-35B-A3B-DFlash2` at revision
`51ef7b6923ad6c14cb1bb41c37a9041446496ab6`. It has six layers, width 2048,
eight target taps `[1,6,11,16,22,27,32,37]`, and seven proposals per block.
A fine-tune's acceptance rate must be measured on that fine-tune.

## Protocol

- Freeze each native addon together with its adjacent metallibs. Record source
  revision, binary hashes, checkpoint hash, companion revision and device.
- Compare temperature zero, BF16 KV, FP32 recurrent state, 1024 generated tokens
  for the final matrix (128 for bounded diagnostics),
  the same prompt token IDs and no penalties. Record early stops as failures.
  Splash's default INT8 KV is a separate comparison.
- Separate first-request pipeline compilation, warm kernels with a cold prefix,
  and a replay with a retained prefix. The harness records warmup separately and
  salts each cold sample so another request cannot supply its prefix.
- Run one model process at a time. Alternate A/B/B/A order and take at least
  three valid samples per condition. Record global swap before/after, process
  peak RSS and allocator peak separately; reject runs with competing GPU work.
- First compare mlx-node AR before/after. Then compare mlx-node DFlash2 with
  Splash DFlash2. Splash always drafts; its `--verify-rows all` fixes verification
  width and does not disable the draft. Adaptive `--verify-rows auto` is a further
  product-performance comparison after fixed-width correctness is established.
- Report prefill/TTFT and decode separately. Splash `timings.predicted_per_second`
  excludes its entire first emission, potentially several tokens; native
  `metrics.decode_cycle_ms / decode_output_tokens` also captures the host gaps
  between cycles. Retain both metrics rather than treating them as identical.
- Output text hashes detect same-engine drift. Splash's completions strip special
  tokens, whereas native `rawText` retains its own rendering. Cross-engine
  correctness uses teacher-forced distributions on identical token IDs.

## Harness

`benchmark.ts --help` lists the fixture and engine runner commands. `fixture`
loads only a tokenizer and exports short, 2048-token and 32768-token prompts,
their token-ID hashes and little-endian `.u32` files. A native run verifies the
loaded model's template still produces those exact IDs. A Splash run sends the
frozen IDs to `/v1/completions`, so tokenizer/template differences cannot change
the workload. Each completed sample is saved before validation failures return.

The Splash server must be started on a local assembly made by its public Python
`install.local.prepare_local_model(target, draft=..., language_only=True)` API.
When using a separate build directory, set `install.paths.BINARY` to that build's
`splash` binary before preparation. The assembly records source-file SHA-256s and
links the original weights. Start the server with the same checkout's Python:

```sh
PYTHONPATH="$SPLASH" "$SPLASH/.venv/bin/python" -m server.server "$ASSEMBLY" \
  --model local/ornith-1.5-35b-a3b --binary "$BUILD/splash" --port 18282 \
  --max-context 40000 --kv-format bf16 --state-format fp32 \
  --verify-rows all --disable-ane --idle-release off --max-cache-disk 0
```

`--warmup 0` measures the first request after model load. The default warmup is
16 output tokens. `--replay` adds a second request without resetting its prefix.
Use `--warmup-case selected --warmup 128` to warm the selected prompt length
and a full output before timed repetitions. Record this separately from the
first-request result: compilation can otherwise dominate a new prompt length.
For the final paired matrix, the artifact driver `run-quiet-paired.py` applies
a declared load gate: three intervals below 25% system CPU and no observed
heavy native test/compiler process before loading an engine. A timed request
that exceeds 50% system CPU is retained and retried in a fresh engine, at most
twice. Skipped preflight attempts and rejected samples remain in its manifest.
GPU frequency is recorded rather than used as an automatic rejection rule,
because the algorithm itself can change frequency selection.
Use `--mode ar`, `--mode mtp`, or `--mode dflash --draft ...` for native runs.
Pass `--verify-rows all|auto` to both harness runners; native `auto` enables
`mtpAdaptiveDepth`, and the Splash runner checks the server's reported policy.
Splash's adaptive cost and acceptance model persists across requests. Start
each comparison condition in a fresh engine and record its training warmup;
otherwise earlier conditions can change later budgets.
Native `mtpMeanDepth` counts physically proposed draft tokens. Adaptive masking
keeps eight physical rows, so this metric does not reveal the selected prefix
width. Compare adaptive cycle counts with Splash's retained speculation
counters separately. Profiling toggles used by earlier experiments were
retired after validation; their historical logs remain in the artifact record.

## Teacher-forced numerics and profiling

`capture-teacher.ts ADDON TARGET TEXT CACHE [TOKENS=2048] [TOPK=256]` calls the
existing native quality API, with fresh caches and a 16-position head chunk.
It is bounded to 4096 tokens. The cache stores exact token IDs, full-vocabulary
argmax and target log-probability, plus top-K logits and the full normalizer.
This API takes a native model directory. For the GGUF, use `modelAssetsPath`
recorded after its exact-file benchmark load: that is its native prepared cache.
Keep the original GGUF hash as provenance. Packed Q4/Q6 data is preserved, but
the importer may cast original F32 norms and router weights to BF16; native and
Splash precision must therefore be audited before attributing cross-engine drift.
The head chunk does not control layer-stack geometry: this model prefills in
2048-token chunks, while Splash's teacher tool uses eight-row verification.

```sh
"$SPLASH/.venv/bin/python" teacher-parity.py "$CACHE/0000000000.safetensors" \
  --tokens "$CACHE/tokens.u32"
"$BUILD/engine-tests/teacher-logits" "$BUILD/splash.metallib" "$ASSEMBLY" \
  "$CACHE/tokens.u32" "$CACHE/splash" --window 256:64 \
  --kv-format bf16 --state-format fp32
"$SPLASH/.venv/bin/python" teacher-parity.py "$CACHE/0000000000.safetensors" \
  --splash-logits "$CACHE/splash-256.f32" --window 256:64 --output "$CACHE/parity.json"
```

Splash computes the final prefill position and 64 teacher-forced verify rows.
The comparison reports exact top-1 agreement and next-token NLL, and explicitly
labels top-K-plus-tail KL as a lower bound. Compare native baseline/candidate
with `scoreAgainstTeacher` using the same cache. A matching greedy transcript
alone does not verify close distributions; a differing transcript alone does
not prove a numerical defect.
For all 2047 prediction positions of a 2048-token fixture, collect windows
`2041:2040` and `2047:8`, then keep positions 0–2040 from the first and 2041–2046
from the second. The raw full-vocabulary dumps occupy about 2 GB. Compare
baseline and candidate against exactly the same selected Splash rows.

The frozen `candidate-v6-prefill-order` iteration resolves the earlier native
prefill discrepancy on this fixture: all 2047 prediction positions agree on
argmax, the top-256-plus-tail KL is zero, and candidate and baseline mean NLL
are both `1.4252860731302848`. The relevant Metal correction preserves the
native prefill tile accumulation order. The score establishes equivalence for
these measured quantities on this fixture; it does not establish bitwise
equality for every full-vocabulary logit or other sequence geometries. In
particular, eight-row speculative verification needs its own continuation
checks. The result is saved as
`candidate-v6-prefill-order-teacher-score.json` in the benchmark artifact root.

`acceptance.ts ADDON NATIVE_MODEL DRAFT RESULT_JSON [ADAPTIVE=false] [KV_FORMAT=bf16]` runs
bounded live checks separately from timing: cold synchronous/streaming parity,
warm continuation with prefix reuse, streaming and queued cancellation,
recovery under a new cache owner, sampled decoding with a repetition penalty,
and switching from autoregressive back to DFlash2 decoding. It uses 16–32 output
tokens per completed request and a 180-second process watchdog. Pass `true` as
the fifth argument to exercise adaptive verification. Use `int8` as the sixth
argument to check quantized KV caches. Run it with `oxnode` and one model process
at a time.

The v6 masked adaptive runtime passed all nine recorded live checks with BF16
KV and again with INT8 KV. These acceptance runs used `candidate-v6-testbuild`
before the separate Metal prefill-order correction; their manifests identify
the exact addon and settings. A 256-token adaptive diagnostic exercised row
budgets from one through eight. Neither correctness result is a performance
parity claim: final comparisons still require fresh-process AR before/after
and native-versus-Splash pairs at short, 2048-token and 32768-token prompts,
with both fixed and adaptive verification and matched telemetry.

A v7 fixed-width measurement on the M5 Max used the 2048-token prompt,
1024 output tokens, 128 warmup tokens, BF16 KV and alternating fresh processes.
Two native samples were 226.03 and 223.93 decode tokens/s; two Splash samples
were 271.54 and 270.20. All fully contained telemetry intervals reported
1620 MHz GPU frequency. Each engine reproduced its own output hash and cycle
count: 222 native cycles versus 233 Splash cycles. Native acceptance was higher
on this workload, while its decode time per cycle remained about 20.48 ms
versus 16.21 ms. This is a measured remaining performance gap, not parity or a
claim covering other prompt lengths. The raw runs and interval telemetry are
in `candidate-v7-all-1024-paired-2k` under the benchmark artifact root.

The v8 opt-in compiled draft passed independent-cache replay and dynamic
position, window, mask, and invalidation checks. Its live BF16 and INT8 runs
each passed all nine acceptance cases. On an exact 1024-token continuation,
four alternating eager/compiled runs produced identical recorded cycle
positions, proposal lengths, reference/live accepted counts, and target-flip
flags across all 235 cycles. This checks the recorded continuation behavior;
it does not compare every unrecorded logit. Teacher-forced wall time was about
5.6% lower with compilation, but this synchronized diagnostic ran during a CPU
build and is not a product throughput result. The frozen references, output
hashes, and arrays are in `candidate-v8-compiled-draft-tf-abba`.

The canonical v11 package was built against MLX source with the tracked patches,
and its packaged addon and both metallibs were frozen together. Its full teacher
check reproduces the baseline on all 2047 measured positions: argmax agreement
1, top-256-plus-tail KL 0, and mean NLL `1.4252860731302848`.
Serial-encoder OFF/ON runs also have identical recorded continuation paths
across 234 teacher-forced cycles, and both settings passed all nine live
acceptance checks with BF16 KV and again with INT8 KV. The frozen canonical
bundle excludes the older executables linked against prebuilt MLX.

The v11 serial-encoder product experiment found no gain at matched clocks.
With the same three opt-ins (compiled draft, selected affine register kernels,
and indirect expert dispatch), serial ON gave 219.46 and 219.37 decode tokens/s;
the final OFF sample gave 219.36. These samples reported 1619.75–1620 MHz mean
GPU frequency and 17–18% CPU activity over fully contained intervals. Prefill
was 5966/6048 tokens/s with serial ON and 6034 with it OFF. All generated 1024
tokens in 235 cycles with the same output hash. The first OFF sample was slower
at a lower mean GPU clock, and two earlier attempts were rejected by the
declared CPU-load gate; all attempts remain preserved in
`candidate-v11-serial-quiet-abba`. Concurrent encoding remains the default.

The v11 pointer-tagged command trace joins every primitive label to its
completed command without a count mismatch. Target-layer command buckets each
contain routing, expert, projection, and attention or GDN work, so their GPU
intervals cannot be assigned to MoE kernels alone. The observed sums were
11.203 ms per cycle for 30 GDN-layer commands and 3.720 ms for 10 attention-layer
commands; mixed commands containing generic draft attention totaled 2.958 ms.
These are diagnostic command intervals, not individually timed kernels or
additive hardware-utilization figures. The raw trace and join audit are in
`candidate-v11-command-profile`.

The v12 draft tensor-attention experiment uses the canonical v12 addon with an
explicitly frozen, ABI-matched corrected Metal library. The initial v12 Metal
library had a partial-context bug and is not eligible for this opt-in. The
corrected kernel passed 16 independent F64-oracle cases, seven single-trace
replay cases, and fallback checks. With attention enabled, BF16 and INT8 KV
each passed all nine live lifecycle cases. The route trace confirms six
`DraftAttention` primitives per full-width proposal and generic attention for
short tails. Its exact-ID continuation changes acceptance, so these results
do not claim bitwise equivalence to generic attention.

Three quiet, fresh-process ABBA experiments on the 2048-token fixture selected
the next experimental profile. Each used 1024 output tokens, 128 warmup tokens,
fixed eight-row verification, and BF16 KV. GPU means over fully contained
intervals were 1619.5–1620 MHz and every run passed the declared CPU gate.
The median decode rates were:

| Change                                                  |    OFF |     ON | Relative change |
| ------------------------------------------------------- | -----: | -----: | --------------: |
| Corrected draft tensor attention                        | 219.26 | 224.66 |          +2.46% |
| Selected affine register kernels, with tensor attention | 220.23 | 225.84 |          +2.55% |
| Target register output projections, with both above     | 223.30 | 233.35 |          +4.50% |

Rates are decode tokens/s. Every condition reproduced its own output hash;
the changes also alter acceptance and cycle counts. These separate ablations
are fixture-specific and their percentages must not be added together. They
do not establish parity with Splash or cover the remaining prompt lengths.
Artifacts are `candidate-v12-attention-quiet-abba`,
`candidate-v12-attention-af4-quiet-abba`, and
`candidate-v12-attention-register-output-quiet-abba`.

A guarded scheduling ABBA on the selected profile rejected larger command
buffers. Jointly increasing `MLX_MAX_MB_PER_BUFFER` from the device default 50
to 512 and `MLX_MAX_OPS_PER_BUFFER` from 50 to 1000 reduced median decode
throughput from 231.31 to 215.72 tokens/s (6.74% slower). All four processes
produced the same 1024-token hash and 228 speculative cycles, with GPU means
1619.25–1619.75 MHz. Peak allocator usage rose from 22.342 to 29.540 GiB;
system swap remained unchanged. A separate trace confirmed about 86.6 versus
14.1 commands per profiled iteration, so fewer commands alone did not improve
this workload. Default buffer limits remain selected. All quiet attempts,
50-GiB allocator and 1-GiB swap-growth guard observations are retained in
`candidate-v12-buffer512-ops1000-quiet-abba`.

Target-only AR regression checks compare the frozen native baseline against
the native candidate. Splash's inspected serving API exposes only `auto` and
`all` verification policies (`server/serve_options.py:701` and
`runtime/model/ModelFactory.hpp:102`); both retain DFlash. Its runtime encodes
the draft before target verification (`runtime/model/Runtime.cpp:2993`), and
its target verification/head API uses fixed eight-row blocks
(`runtime/model/QwenTarget.hpp:456`). The backend benchmark also explicitly
selects `VerifyRows::All` (`dev/benchmarks/backend_benchmark.mm:956`). Therefore
`--verify-rows all` must not be reported as source AR throughput. The production
native-versus-Splash performance comparison remains DFlash with matching
fixed or adaptive policies; no custom source AR extension is part of this work.

The v14 canonical artifact fixes an AIR cache-provenance bug found during the
v13 target-attention experiments: saved dependencies from another checkout
could validate an old shader without checking the current source identity.
All 16 v14 AIR inputs were audited against the current source tree, and raw
library-name checks found exactly the expected 73 segmented-attention and
2917 K-quant kernels. The target-attention numerical suites passed. V13
head/layout evidence remains labeled as-built; no invalid v13 grid, split, or
reducer timing is used. Four alternating v13/v14 default-profile continuation
runs reproduced all 236 recorded cycle tuples exactly.

A quiet v14 head-FP32 ABBA retained the selected five-flag profile and varied
only `MLX_DFLASH2_HEAD_FP32`. OFF produced 230.67 and 232.68 decode tokens/s
(median 231.67); ON produced 191.33 and 191.42 (median 191.38). GPU means were
1619.5–1619.75 MHz. Each condition reproduced its own output hash, but ON
required 275 speculative cycles versus 228 OFF, while decode wall per cycle
remained about 19.4 ms. This fixture rejects enabling the option despite its
earlier teacher-forced acceptance improvement. Failed CPU-load attempts remain
preserved separately in `candidate-v14-head-fp32-quiet-abba`.

An early target-only AR pair on the 2048-token fixture, with 256 output and
128 warmup tokens, measured 108.15 decode tokens/s for the native baseline and
106.64 for v14. Prefill was 5011 and 5175 tokens/s respectively. Both passed
the quiet gate, but each has only two fully contained request telemetry
intervals and one decode-only interval at 1620 MHz. The output hashes differ,
so this is a bounded directional check, not the final three-length regression
matrix. Raw data is in `candidate-v14-ar-2k-direction`.

`$BUILD/engine-tests/decode-profile $BUILD/splash.metallib $ASSEMBLY
--prompt-tokens 2048 --cycles 4 --kv-format bf16 --state-format fp32
--dispatch-list FILE` reports fused GPU/wall times and per-kernel dispatch
attribution for prefill and B1–B4 DFlash cycles. Its synthetic prompt differs
from the comparison fixture. The individually replayed dispatch sum includes
extra scheduling overhead and must not be presented as end-to-end throughput.

The v15 canonical package and all 12 source-linked test executables are frozen
with matching libraries and source snapshots. All 16 AIR source identities and
dependency content hashes were checked. Its full 2047-position teacher check
again matches the native baseline exactly in the measured argmax, top-256-plus-
tail KL, and NLL quantities. V14/V15 baseline continuation paths and both
prepared-input ablations reproduce all 236 recorded cycle tuples exactly.
The combined prepared-input and register-split path passed all nine live
lifecycle checks independently with BF16 and INT8 KV. A runtime trace confirms
the prepared consumers and selected split kernels execute in the full model.

Register split, full-context projection, and long-context exact partitioning
change recorded acceptance paths; each condition repeats exactly on its fixed
reference. These checks do not establish product throughput or general accuracy.
All three attempted v15 quiet preflights failed before inference because CPU
activity stayed above the declared threshold. The skipped attempts are retained
in `candidate-v15-gdn-prepared-quiet-abba`; no v15 product timing was collected.
The final matched native/Splash three-length fixed/adaptive matrix and native
AR baseline/candidate regression remain required.

Final source comparisons require at least three valid samples for each engine,
verification policy, and prompt length. Run adjacent engine pairs in original,
reversed, then original order; rejected attempts do not count toward this floor.
A single directional pair may guide optimization, but cannot replace this final
coverage. Native AR baseline/candidate checks likewise require three valid
samples per prompt length.

The v16 canonical artifact preserves the full teacher result and the v15
recorded baseline path. T8 storage also preserves every recorded cycle tuple.
The grouped register option changes the fixed-reference path from 236 to 238
cycles; each condition repeats exactly, but no product gain is established.
The original-arithmetic T64 M1 option restores the preserved baseline raw text
and hash for the 2048-prompt, 256-output AR fixture. All three new options
together passed nine adaptive lifecycle cases with BF16 KV and nine with INT8.
These checks cover numerical and runtime behavior, not final performance.

A bounded v16 32K trace proves that exact partitioning changes the actual
target attention launch from 96 to 65 partitions with 512-key planning and
the 256-thread reducer enabled. The generated outputs differ, consistent with
the changed reduction order. The attempted latest native/Splash performance
pair again failed all three quiet preflights before inference; skipped source
and adaptive runs are explicit. A later readiness signal allowed one native
fixed-policy sample: 232.06 decode tokens/s, 228 cycles, with four fully contained
telemetry intervals at 1619–1620 MHz and CPU mean 16.76%. The adjacent source
request was rejected after CPU activity returned to 100%, and both retries
failed preflight. This leaves an unpaired native sample, not a current speed
ratio or final performance result. All attempts are retained in
`candidate-v16-quiet-direct-retry1/all`.

A subsequent v16 directional pair passed the unchanged quiet gate for both
engines and policies, with matched decode GPU clocks of 1619.75–1620 MHz.
The 2048-token prompt, 1024-token output and 128-token selected-prompt warmup
gave the following results in fresh processes:

| Policy | Native decode tokens/s | Splash decode tokens/s | Native cycles | Splash cycles |
| ------ | ---------------------: | ---------------------: | ------------: | ------------: |
| all    |                 231.77 |                 271.31 |           228 |           233 |
| auto   |                 230.37 |                 291.67 |           252 |           229 |

Fixed-policy prefill was 6009.53 versus 6563.93 tokens/s. Adaptive cycle wall
time was 17.62 versus 15.32 ms. CPU means were 10.98–14.49%, and swap did not
change. This is one valid sample per engine/policy, not the final three-sample
matrix. It establishes a current remaining gap on this fixture: about 14.6%
for fixed decode and 21.0% for adaptive decode. All raw runs, exact environment,
engine-specific output hashes, and telemetry are in
`candidate-v16-quiet-direct-retry3`.

Three separate quiet ABBA screenings on the same fixed-policy base retained
the remaining options as opt-ins. T8 storage gave median OFF/ON rates of
230.52/232.29 tokens/s, but the final OFF sample exceeded both ON samples, so
the small median difference is inconclusive. SCAN8 gave 232.02/226.57 tokens/s
with the same output hash and 228 cycles. Register split gave 232.24/195.57
tokens/s and changed its repeated trajectory from 228 to 275 cycles; this
flag affects both target Q4 and draft affine projections. Every run passed
the CPU gate, decode clocks were 1619.25–1620 MHz, and swap was unchanged.
These screenings have two samples per condition and are preserved in
`candidate-v16-quiet-followup-ablations`. A following M1 AR quartet was skipped
before its first model load because quiet readiness closed; it supplies no
performance evidence, and M1 remains opt-in pending a quiet comparison.

The v17 canonical bundle freezes 16 source-linked test executables, its addon,
and both libraries. A verified source-test/AIR handshake precedes packaging,
and all 18 test/library hashes match afterward. Its promoted defaults, with
the five optimization overrides absent, reproduce all 236 recorded v16
continuation tuples. The full 2047-position teacher result remains exact, and
default adaptive BF16/INT8 each pass nine live lifecycle cases. A grouped
register test separately found a misaligned scalar-view edge; the correction
was applied after this immutable bundle and requires the next build's gate.
The standard aligned checkpoint experiments do not claim to cover that edge.

A dedicated no-warmup diagnostic captures actual compiled verification inputs
at layers 0, 5, 10 and 39 for four physical eight-row cycles. Explicit compiled
outputs supply replay arrays; captures are saved after successful acceptance
and logical commit construction. Fixed-policy capture OFF/ON produces identical
128-token raw text and 34 cycles. A separate adaptive capture exercises live
budgets 6, 3, 2 and 1. Each policy saves 16 finite, shape-checked tensor files
with token/frontier metadata and exact artifact provenance. Capture I/O changes
the adaptive cost observations, so these runs are never timing evidence.
Artifacts are in `candidate-v17-real-verify-capture`.

Quiet v17 ABBA screenings found no clear improvement from logical expert-buffer
accounting (229.96 OFF versus 230.28 ON median decode tokens/s) or paired expert
stages (233.23 versus 233.91). Both preserve every recorded continuation tuple
and all 228-cycle product hashes; pairwise speed changes are inconsistent.
The five promoted defaults remain unset during these comparisons. Raw attempts,
memory guards and telemetry are retained in `candidate-v17-logical-budget-quiet-abba`
and `candidate-v17-paired-stage-quiet-abba`; neither option is promoted by this
two-sample-per-condition screening.

Further v17 quiet ABBA screens kept the five promoted defaults unset and toggled
one option at a time on the 2048-prompt, 1024-output fixed-policy workload.
The hybrid condition holds its Q4-down filter fixed while toggling register
execution. Each condition has two valid samples; these are selection screens,
not the final three-sample matrix.

| Option                         | OFF median tokens/s | ON median tokens/s |  Change | OFF / ON cycles |
| ------------------------------ | ------------------: | -----------------: | ------: | --------------: |
| Hybrid register, Q4 down only  |              230.94 |             210.04 |  -9.05% |       228 / 243 |
| Selector affine Q4             |              231.84 |             184.31 | -20.50% |       228 / 286 |
| Full-width context projection  |              232.31 |             235.33 |  +1.30% |       228 / 221 |
| Target flat grid               |              231.50 |             231.16 |  -0.15% |       228 / 228 |
| Target 256-thread reducer      |              231.24 |             195.68 | -15.38% |       228 / 269 |
| Target page-aligned partitions |              229.89 |             229.92 |  +0.01% |       228 / 227 |
| Target FP32 reduction          |              231.42 |             169.65 | -26.69% |       228 / 289 |

All within-condition output hashes repeat. Flat-grid also preserves the
cross-condition hash; the other options change the generated trajectory.
GPU interval means were 1618–1620 MHz, and every run passed the unchanged
CPU/memory guards. FP32-reduction ON rates varied from 161.09 to 178.22 tokens/s
with CPU means 12.8% and 30.0%; both were slower than either OFF sample, but the
median is not a stable estimate of the exact regression. No option is promoted
from these screens. Full-context is a modest candidate for another repeated
comparison; flat-grid and page alignment show no clear gain, while the other
listed options regress on this fixture. Long-context partition choices remain
unmeasured in this quiet window. Raw artifacts are named
`candidate-v17-<option>-quiet-abba` using `hybrid-q4`, `selector-af4`,
`full-context`, `flat-grid`, `reducer256`, `page-splits`, and `fp32-reduction`.

V18 canonical defaults preserve all 236 V17 recorded continuation tuples and
the full 2047-position baseline teacher result (top-1 agreement 1, KL 0,
NLL 1.4252860731302848). Default adaptive BF16 and INT8 each pass nine lifecycle
cases. The earlier scalar-view alignment edge now passes its focused checks.
Source-style expert arithmetic remains diagnostic and is eligible only for the
supported physical M8 shape; it does not run on the wide teacher forward.

The source precision 2×2 uses the same fixed continuation in fresh processes,
with each condition repeated twice. These are acceptance diagnostics, not
own-trajectory performance or general accuracy results:

| Norm weights                | Source M8 math | Cycles | Reference accepted | Live accepted | Cycles with target/reference flip |
| --------------------------- | -------------- | -----: | -----------------: | ------------: | --------------------------------: |
| Imported BF16               | OFF            |    236 |                804 |           787 |                                21 |
| Imported BF16               | ON             |    234 |                813 |           789 |                                14 |
| Original F32, reference RMS | OFF            |    234 |                812 |           789 |                                20 |
| Original F32, reference RMS | ON             |    237 |                816 |           786 |                                22 |

Every condition repeats its recorded tuples exactly. The effects interact;
combining the options does not improve this reference-path acceptance.
The F32-norm loader initially rejected the GGUF epsilon's promoted-f32 JSON
value because its guard compared exact f64 bits. The failed load is preserved.
An artifact-only config view writes 1e-6 with exactly the same f32 bits
(`bd378635` little endian), keeps canonical weight and `.complete` identities,
and reproduces both source-OFF and source-ON control tuples exactly before
running the norm arm. The native cache is untouched; the loader guard fix is
for the next candidate.

Wide-forward scoring with the 81 original F32 residual/final norms yields
NLL 1.4731557018650876, baseline top-1 agreement 0.93258427 and bucketed KL
0.092338383. Source-M8 OFF/ON reports are exactly equal on this wide shape,
as expected. This is material drift on one fixture and supplies no M8-math
quality gate. Combined source math and F32-reference norms nevertheless pass
all nine adaptive lifecycle cases in each KV format. A separate eight-row
teacher diagnostic is staged off-tree: it will capture a new matched-schedule
baseline before source-math promotion. Its eager chunked score does not replace
compiled DFlash recovery and acceptance checks.

Evidence is retained in `candidate-v18-default-gates`,
`candidate-v18-source-precision-tf-matrix` (including the original refusal),
`candidate-v18-source-precision-epsilon-view-tf-matrix`,
`v18-epsilon-config-view-manifest.json`, `candidate-v18-wide-precision`,
`candidate-v18-source-combined-live`, and `teacher-m8-prototype`.

The subsequent V18 source-math quiet screen stopped at its first ON preflight:
only one OFF sample ran (189.50 tokens/s, 228 cycles, mean GPU 1265.25 MHz,
unchanged swap). No ON model process was launched, so there is no paired speed
evidence and no option is promoted. The unpaired result and rejected preflight
are preserved in `candidate-v18-source-math-quiet-abba`. V19's prepared quality
queue adds a fresh matched eight-row teacher baseline, original-cache epsilon
acceptance, separate body/residual/exact-norm conditions, and lifecycle checks.
Its next eligible quiet screens prioritize source-only arithmetic and native
M1 AR before marginal tensor schedules; final three-sample coverage remains
outstanding.

V19's focused exact-norm, residual, source-budget and teacher-schedule gates
pass. Its default V18/V19 continuation tuples and wide teacher score remain
exact. A fresh BF16 teacher baseline using 255 eight-row chunks plus a
seven-row tail self-scores exactly at NLL 1.4646127115649366. This differs from
the wide-forward baseline because the forward schedule changes arithmetic.
Against that same eight-row baseline, source body alone yields NLL
1.471082163253061 (+0.442%), top-1 agreement 0.94137763 and bucketed KL
0.08262988. These are one-fixture diagnostics, not a general accuracy verdict.
The original source operator oracle remains exact at the tested M8 layers.

The first V19 own-trajectory screen rejected its initial OFF preflight: all
12 CPU samples showed full utilization, and no model timing process launched.
Source-body and M1 AR timing remain queued. Artifacts are
`candidate-v19-default-quality`, `candidate-v19-m8-baseline`,
`candidate-v19-m8-body-score`, and `candidate-v19-source-body-quiet-abba`.

The completed V19 eight-row teacher comparison isolates the source expert
body, residual combine and exact source RMS. Each score uses the same 2047
positions and matched native schedule; source routing is not enabled here.

| Condition                           | Mean NLL | Change from native | Native top-1 agreement |
| ----------------------------------- | -------: | -----------------: | ---------------------: |
| Native                              | 1.464613 |                 0% |                1.00000 |
| Source body                         | 1.471082 |            +0.442% |                0.94138 |
| Source body and residual            | 1.472981 |            +0.571% |                0.92916 |
| Exact source RMS                    | 1.466517 |            +0.130% |                0.93845 |
| Source body and exact RMS           | 1.470576 |            +0.407% |                0.93454 |
| Source body, residual and exact RMS | 1.496323 |            +2.165% |                0.93258 |

These one-fixture results show interacting arithmetic changes, not a general
accuracy verdict. Original-cache F32 norms now load without the epsilon view;
the reference-RMS wide score exactly reproduces the earlier control. Default
and combined-source adaptive BF16/INT8 each pass all nine lifecycle cases.
Repeated source-body continuation tuples also exactly reproduce V18. Evidence
is in `candidate-v19-m8-summary.json`, `candidate-v19-wide-norms`,
`candidate-v19-default-live`, `candidate-v19-combined-source-live`, and
`candidate-v19-body-tf`.

A later source-body OFF/ON/ON/OFF product screen completed under the unchanged
CPU gate: 217.70/204.89/199.55/202.55 decode tokens/s. Each condition repeats its
own output, but source-body execution increases the cycle count from 228 to 278. Decode GPU means vary from 1406.5 to 1574.5 MHz, so the median difference
is not an isolated speed estimate. The option remains off. Earlier rejected
preflights and in-run failures are retained alongside the completed screen in
`candidate-v19-source-body-quiet-abba-retry3` and its preceding attempts.

The following M1 AR screen accepted its first OFF sample, then rejected ON
because CPU utilization reached 100%; no valid speed comparison exists.
Nevertheless, the ON raw text and hash exactly match the original baseline
256-token AR fixture. `candidate-v19-m1-ar-quiet-abba` keeps both the rejection
and this separate output check. Finally, the isolated compact source-column
library preserves all four canonical source-body continuation paths exactly
(234 cycles, 813 reference and 789 live accepted tokens) in
`candidate-v19-compact-source-tf`. This is numerical evidence only; its product
performance has not been measured. Final performance parity and the complete
three-sample matrix remain outstanding.

V20 canonical defaults preserve all recorded V19 continuation tuples and the
wide 2047-position teacher score exactly. Deferred GDN OFF/ON/ON/OFF also
preserves all 236 recorded continuation tuples. Default and deferred adaptive
execution each pass nine lifecycle cases in both BF16 and INT8 KV formats.
These results are in `candidate-v20-default-quality`, `candidate-v20-deferred-tf`
and `candidate-v20-default-deferred-live`.

A separate 128-token fixed-policy trace preserves raw output and all 34 cycles
while reducing explicit `GdnCommit` primitives from 32 to one. Both modes still
emit 1080 `GdnFusedComplete` primitives, whose shared name does not expose the
deferred library variant. This is indirect evidence of the changed settlement
schedule, supported by the focused state and recovery tests; it is not a
deferred-specific pipeline count or a throughput result. The remaining commit
occurs near the end before the final 60 complete-GDN calls. Raw logs and line
locations are retained in `candidate-v20-deferred-trace`.

V20 source routing has independent original-source operator gates and passes
all nine adaptive lifecycle cases in both KV formats, including its combined
residual/exact-RMS and deferred variants. A dedicated control/capture pair
preserves output and all 37 cycles. Four real M8 snapshots at layers 0/5/10/39
contain changing replay inputs and immutable router/shared-gate weights with
validated shapes and payload sizes (`candidate-v20-source-routing-capture`).
The fresh native teacher8 baseline again self-scores exactly. Source body
reproduces its previous score; adding source routing changes matched-schedule
NLL to 1.494320, versus native 1.464613. That native-reference drift is not by
itself a source-model accuracy verdict.

The aligned original Splash teacher uses an eight-token prefill followed by
254 eight-row verifies. Comparison covers the same positions 7..2039 for all
conditions, excluding the native first seven positions and final seven-row
tail. Source prefill/verify implementations still differ from the native eager
teacher path; compiled DFlash correctness is checked separately.

| Condition                                    | Mean NLL | True next-token top-1 | Argmax agreement with Splash | Bucketed KL to Splash, lower bound |
| -------------------------------------------- | -------: | --------------------: | ---------------------------: | ---------------------------------: |
| Original Splash                              | 1.456001 |               0.72356 |                            1 |                                  0 |
| Native                                       | 1.452847 |               0.72651 |                      0.92720 |                            0.10295 |
| Source body                                  | 1.459286 |               0.72159 |                      0.93212 |                            0.10731 |
| Source routing and body                      | 1.482431 |               0.71766 |                      0.93114 |                            0.09057 |
| Source routing, body, residual and exact RMS | 1.457457 |               0.72504 |                      0.93655 |                            0.08827 |

The combined source condition recovers most of the routing-only NLL change and
is close to Splash on this fixture, without exact logit or general accuracy
parity. A single hash-verified source logit file supports all comparisons in
`candidate-v20-aligned-source-teacher`; it will be reused by later diagnostics.
Source mixer output projections and Q/K norm/RoPE remain separate arithmetic
work at this point.

Two valid exploratory quiet quartets then used fresh processes, 2048 prompt
IDs, 1024 generated tokens, 128 selected warmup tokens, fixed all-row policy,
BF16 KV and FP32 recurrent state. Promoted defaults remained unset. Each arm
has two samples, so these are selection screens rather than the final matrix.

| Screen                     | OFF rates, tokens/s | ON rates, tokens/s | OFF / ON median | OFF / ON cycles |
| -------------------------- | ------------------- | ------------------ | --------------- | --------------- |
| Combined source arithmetic | 232.04, 229.26      | 231.91, 232.30     | 230.65 / 232.10 | 228 / 253       |
| Native deferred GDN        | 227.92, 231.81      | 227.28, 231.28     | 229.87 / 229.28 | 228 / 228       |

Every within-condition output hash repeats; deferred also preserves the same
output across conditions. Combined-source faster cycles are offset by its
larger cycle count. Neither screen establishes a useful throughput gain and
neither option is promoted. GPU means were 1619.25–1619.75 MHz and CPU means
16.5–18.9%; swap did not increase within either quartet. Deferred allocator
peak was 22.342 GiB. Earlier rejected preflights remain preserved. Raw product
evidence is `candidate-v20-combined-source-quiet-abba` and
`candidate-v20-deferred-quiet-abba-retry2`.

V21 adds opt-in source mixer projections and a direct deferred trace label.
Its staged queue checks default equivalence, mixer-only and combined-source
M8 teacher/continuation/lifecycle behavior, and real production activation
before another own-trajectory screen. Current native/Splash final all/auto
comparisons and native AR baseline/final coverage still require at least three
valid samples for every selected condition at all three prompt lengths.
Performance parity is not yet achieved.

## V21 source mixer interaction

Canonical V21 retains exact native defaults: the wide 2047-position teacher
and repeated fixed-reference paths match V20. All 54 adaptive BF16/INT8 live
scenarios pass across native, mixer-only and source-combined+mixer conditions.
The source mixer is independently opt-in; primitive traces prove its actual
selection. Repeated fixed-reference paths are deterministic within each
condition, but change across arithmetic conditions.

On the same 2033 source-aligned positions, source-combined+mixer has NLL
1.450820, true next-token top-1 0.72651, Splash argmax agreement 0.94048 and
bucketed KL lower bound 0.07243. The previous source-combined control has
NLL1.457457, agreement0.93655 and KL0.08827; original Splash NLL is1.456001.
This supports improved fidelity on this fixture, without exact logit or
general accuracy parity. Original source logits were reused only after hash
and schedule validation. `candidate-v21-quality-summary.json` binds all eight
quality phases and the canonical addon/Metal hashes.

Quiet fixed-all 2K/1024-output/128-warmup ABBA screens show a strong interaction:

| Profile and change         | OFF rates, tokens/s | ON rates, tokens/s | OFF / ON median | OFF / ON cycles |
| -------------------------- | ------------------- | ------------------ | --------------- | --------------- |
| Native plus mixer          | 233.81,232.48       | 210.19,210.28      | 233.14 /210.23  | 228 /248        |
| Source combined plus mixer | 231.15,232.81       | 259.63,258.66      | 231.98 /259.15  | 253 /226        |

Both use BF16 KV/FP32 state and leave the five promoted defaults unset. Each
condition repeats exact raw output; every request emits1024tokens and stops
for length. Decode clocks are1619.25–1620MHz, CPUmeans15.2–17.3%, allocator
peak22.342GiB, and swap is unchanged. Mixer-only regresses9.83%; combined
source+mixer improves11.71% versus its source control. These are two-sample
exploratory screens, not default promotion or the final matrix.

A fresh adjacent fixed-all directional pair then measures combined+mixer
258.843tokens/s versus original Splash271.424tokens/s, with226versus233cycles
and matched1619.33/1620MHz clocks. Native remains4.64% behind in decode. Native
prefill is6076tokens/s, versus Splash6450from HTTP TTFT or6534from its engine
prefill wall counter; these accounting definitions are retained separately.
This is one valid paired sample. Auto, short/32K, at least three valid samples
per final condition, and baseline/final native AR coverage remain outstanding.

The ABI-matched V21 compact-source Metal experiment passes ten linked
interaction gates and exact full-model combined+mixer fixed-reference ABBA:
all four217-cycle tuple arrays have identical hashes. This run records a
fresh combined+mixer reference, so its cycle count is not comparable to the
earlier shared native-reference fixtures. Its timing during V22 compilation
is ineligible; own-trajectory quiet performance remains pending.

Existing V21 pointer traces have no missing or mismatched command labels. An
explicit first-DraftAttention to last-GroupedMoe diagnostic window ranks mixed
target-layer commands as the largest remaining cost. Whole-command interval
unions overlap across families and cannot establish individual kernel times
or product throughput. A warm, explicitly phase-marked profile is queued
after V22 default correctness gates.

## V22 workspace and compact-helper screening

V22 canonical default wide teacher and fixed-reference paths remain exact.
Combined-source+mixer paths match V21, and replacing only the ABI-matched
compact helper library preserves those paths. Workspace OFF/ON is exact in
both native and combined policies. All 90 BF16/INT8 lifecycle cases pass across
canonical native/combined/workspace conditions and the compact combined
bundle. Actual workspace traces show 1360 acquisitions/1359 reuses for native
and 1320/1319 for combined, with no acquisition under OFF.

Independent quiet 2K / 1024-output / 128-warmup ABBA screens retain the canonical
helper and workspace OFF. Compact has median 256.51 versus 259.44 tokens/s
(−1.13%). Workspace has 258.96 versus 257.13 tokens/s (+0.71%), with its two ON
samples 257.29 and 260.64; this small effect is insufficient for promotion.
Every request produces the same 1024-token raw output and 226 cycles. Decode
clocks are 1619–1620 MHz, swap is unchanged, and workspace adds about 4.5 MiB to
allocator peak. These are exploratory two-sample conditions, not final proof.

The warm phase diagnostic uses explicit last-turn spans after 128 warmup
tokens. Combined has 17.65 ms wall/cycle, 16.72 ms command-interval union and 0.93 ms
without GPU overlap, with 85.3 commands/cycle. Commands containing GroupedMoe
account for 13.02 ms/cycle and draft-attention commands 2.36 ms/cycle. These are
whole-command intervals containing mixed work; families overlap and must not
be summed or reported as individual kernel times. The fixture differs from
Splash's synthetic decode profiler, so no per-stage source ratio is inferred.
`candidate-v22-evidence-summary.json` binds the quality, activation and screen
reports. Source QK/conv full-model checks and the final three-sample all/auto
and AR matrices remain pending.

An early 32K auto directional pair exposes a larger gap: native 172.389 versus
Splash 195.656 tokens/s (11.89% behind), with 305 versus 301 cycles. Decode
clocks are 1619.4/1619.75 MHz, CPU means 10.3/13.1%, no swap growth, and native
allocator peak 22.949 GiB. Both emit 1024 tokens from the identical 32768-token
fixture and stop for length, with distinct own-trajectory outputs. Native
prefill is 4207.67 tokens/s, versus Splash HTTP 4362.59 (engine wall counter
reported separately). A fixed-all native run measures 216.72 tokens/s and
227 cycles, but its Splash preflight failed; that result remains unpaired.
These single directional attempts do not replace the final repeated matrix.

## User stop and final V23 evidence

Work stopped at the user's request after the active V23 combined-head
fixed-reference quartet completed. V23 defaults retain exact wide teacher
and V22 continuation paths. A fresh forward8/logit8 native teacher baseline
has NLL1.46488534; faithful head-only has1.46456989, KL6.38e-5 and99.5603%
argmax agreement, with255 actual eligible SourceHeadQ6 projections traced.
Repeated native-head paths are deterministic (236 control versus229 head
cycles); combined-head paths are also deterministic (235 cycles in each arm,
but different tuples). The combined control's recorded tuples exactly match
V22. These are numerical results, not product throughput.

No V23 head quiet screen, head lifecycle matrix, final repeated comparison or
selected-policy AR matrix was started after the stop. All queued model work
is cancelled. No owned model, server, telemetry or benchmark conductor remains.
Frozen artifacts and evidence are preserved in
`benchmark-user-stop-handoff.json`; performance parity is not established.

## Resumed V23 validation and draft convolution

The resumed worktree and Splash checkout matched the frozen source identities.
All 72 native/head/combined/combined-head adaptive lifecycle cases passed in
both KV formats, and production traces confirmed the expected SourceHeadQ6
routes. The current-build QK and draft-convolution continuation suites were
deterministic within each condition and passed another 90 BF16/INT8 lifecycle
cases plus production activation checks.

SourceHeadQ6 did not improve the selected combined profile: its quiet 2K
fixed-all medians were 251.77 OFF versus 244.94 ON tokens/s. Full-context
projection also regressed the convolution profile. QK had an incomplete quiet
screen with a substantially worse accepted trajectory and is not promoted.

Draft convolution produced the first new candidate product win in this
resume. Its combined fixed-reference path uses 228 cycles versus 235 for the
combined control. A quiet fixed-all 2K ABBA at matched 1619–1620 MHz decode clocks gave
254.36 OFF versus 266.39 ON tokens/s (+4.73%), with 226 versus 216 product
cycles and repeatable output within each arm. These are two-sample exploratory
screens, not final coverage. Artifacts are
`candidate-v23-combined-conv-persistent-abba-resume` and the preceding
`candidate-v23-{native,combined}-conv-{tf,live}-resume` directories.

Fresh adjacent native/Splash directions using the convolution profile gave:

| Case and policy | Native decode tokens/s | Splash decode tokens/s |   Gap | Native / Splash cycles |
| --------------- | ---------------------: | ---------------------: | ----: | ---------------------: |
| 2K fixed-all    |                 267.51 |                 268.60 | 0.41% |              216 / 233 |
| 2K adaptive     |                 260.23 |                 263.79 | 1.35% |              237 / 259 |
| 32K adaptive    |                 215.86 |                 224.84 | 3.99% |              238 / 241 |

The 2K pairs had matched 1619–1620 MHz decode clocks. The 32K adaptive pair is
directional: after the first token, native averaged 1619.5 MHz and Splash
1508.5 MHz amid high temperatures. It is not a matched-clock parity claim.
Raw evidence is in `candidate-v23-combined-conv-direct-{2k-all,2k-auto,32k-auto}-resume`.
These performance results are provisional: a later short fixed-reference
quartet exposed nondeterminism specific to source convolution. Native and
convolution-disabled controls repeat exactly, so the path is on hold pending a
root-cause fix and fresh validation.

The full source-like 32K attention schedule (512-key planning, 256-thread
reduction, exact and page-aligned splits) is deterministic on the current
convolution profile and passed 18 BF16/INT8 lifecycle cases. Its fixed-all
ABBA medians were 170.96 OFF versus 164.05 ON tokens/s, so it is not promoted.
Workspace reuse was neutral at matched clocks, and global split made the
fixed-reference trajectory worse; neither is selected.

An eligible fixed-all 32K pair then measured native 175.86 versus Splash
195.30 tokens/s (9.96% behind), with 241 cycles each and after-first-token
clock means 1312.8/1298.25 MHz. This is one directional pair at reduced clocks,
not the final matrix. Exact-prompt profiling measured 24.01 ms native wall
and 22.69 ms command-GPU union per fixed cycle, versus 20.31 ms wall and
19.54 ms fused GPU for Splash. Host-only overhead does not explain the gap;
same-input MoE replay is the next attribution step. Evidence is in
`candidate-v23-combined-conv-direct-32k-all-resume-retry1`,
`candidate-v23-combined-conv-32k-all-phase-resume`, and
`candidate-v23-exact-source-32k-profile-resume`.

The final repeated short/2K/32K fixed/adaptive matrix and selected-policy AR
regression also remain outstanding. The cache-owned source-profile loader is
still separate; it is a migration aid and supplies no kernel speedup.

## V25 scheduling and V26/V27 determinism diagnostics

The early-boundary experiment scheduled committed state before detokenization
and emission on continuing DSpark cycles. It preserved adaptive-stop boundary
submission and passed 105 DSpark tests with four ignored. Two quiet, fixed-all
2K screens produced identical output hashes and 226 cycles in every run, but
the candidate was 0.40% slower in A/B/B/A order and 0.37% slower in B/A/A/B
order. The runtime change was reverted. The clean addon and both comparisons
are preserved in `candidate-v25-early-boundary` and
`v25-early-boundary/{2k-fixed-abba,2k-fixed-baab}`.

The source-convolution investigation then separated product and teacher-forced
acceptance capture. Four fresh product runs logged 207 cycles each, and four
fresh teacher-forced runs logged 208 cycles each. Every captured prefix,
verify ID, draft ID, target argmax, reference ID, acceptance count and boundary
matched within each quartet. The older anomaly did not reproduce, and the code
audit found no defensible production fix. Source convolution remains disabled
for parity claims. The diagnostic addons are frozen separately as
`candidate-v26-accept-trace` and `candidate-v27-tf-accept-trace`; their working
source instrumentation was removed after freezing. Comparisons are in
`v26-accept-trace/product-quartet/accept-trace-comparison.json` and
`v27-tf-accept-trace/tf-quartet/accept-trace-comparison.json`.

## Same-input target attention

The off-tree B1/Hq16/Hkv2/R8/D256/BF16 driver passed both Splash natural and
exp2 F64 oracles at 2048 and 32768 history tokens. Its 32K native-BF16 versus
Splash-natural pair measured 306.440 versus 270.418 microseconds per root, an
11.76% Splash reduction with 3.30% noise and complete clock samples. The 2K
comparisons were inconclusive, and the production, partial-reduction and
closest-arm comparisons lacked sufficient clock stability for promotion.
Artifacts are under `target-attention-same-input-v1/results`.

V24 exact-split F64/replay validation also passed. The required BF16 32K
page-off case reported 0.609451 maximum and 0.124595 mean ULP. Page-off reduced
32K time by 7.39% against native default in one repeat, while Splash was 15.52%
faster than page-off in another, but their mean-GPU-clock spreads were 8.57%
and 63.49%. These timings are directional only and do not justify a product
page-off change. See `20261023-v24-exact-split-f64-r2`,
`20261023-v24-pageoff-r2`, and `20261023-v24-pageoff-source` under the same
artifact root.

## Four-layer MoE stage attribution

The V23 native operator and pinned real M8 captures completed all 41 route,
F64, output-bit, occupancy, timing and down-profile checks for layers 0, 5, 10
and 39. The final run preserves native/source/source/native order, 16 warmups,
31 samples and independent down diagnostics in
`v23-q6-m8-four-cell/runs/timing-20261023-v24-down-r3`. Earlier attempts remain
preserved with their fragmented-log and profile-count failures.

The native log assembler now reconstructs human records across embedded Metal
command JSON without dropping or duplicating text. The profile validator also
accounts for the six scalar checksum roots: stages 1–6 each have one checksum
plus 47 timed roots, and stage 7 has 47 timed roots, for 335 roots in total.
All 17 saved native logs passed the repaired parser, and ten earlier profiles
reproduced their saved metrics exactly.

The completed run's GPU frequency ranged from 450 to 1564 MHz. Native and
source timings vary substantially between repeated arms, so they do not isolate
a repeatable Q4 or Q6 format deficit. No weight-plane repack or full expert-stack
conversion is justified by this evidence. The stage and down medians remain in
the run summary for a later quiet-window comparison.

## Keep-seven context crossover

The default context policy selects full eight-row projection only for keep 7
and 8; keep 1 through 6 retains sliced-height FC and K/V projection. Its former
`MLX_DFLASH2_CONTEXT_FULL_WIDTH` override was removed after validation. The V24
microbenchmark measured 114.94 microseconds for
retained-height keep 7 versus 102.16 at full M8, while keep 6 cost 101.81.
The implementation is in `qwen3_5_moe/dflash2_decode.rs` and the frozen
`candidate-v28-keep7` bundle contains source and validation provenance.

Focused checks passed six full-width context tests and two deferred-GDN tests.
The full DSpark suite passed 105 tests with four ignored. BF16 and INT8 each
passed all nine real-checkpoint lifecycle cases. A 1024-token activation trace
confirmed keep 6 projected six rows in 20 commits, keep 7 projected eight rows
in 19 commits, and keep 8 projected eight rows in 69 commits.

The first quiet 2K fixed-all screen used 226 stable cycles versus 211 hybrid
cycles, with median decode rates 242.55 versus 258.66 tokens/s (+6.64%). The
corresponding cycle costs were about 18.68 and 18.76 ms, so the measured gain
came from acceptance rather than per-cycle speed. A reverse-order retry kept
the same hashes and cycles but had a 15% clock ramp and is not a throughput
result. Teacher forcing repeated 235 stable and 231 hybrid cycles within each
arm. Its first difference was a keep-seven draft acceptance change with the
same target flip. These are deterministic, trajectory-inclusive results.

The three-round 32K fixed-all comparison measured median decode rates of
214.31 native versus 220.60 Splash tokens/s, a 2.85% deficit. Native repeated
223 cycles and Splash repeated 241, so the remaining gap is per-cycle cost.
The first two pairs had 1619–1620 MHz decode clocks in both engines; the third
was thermally throttled but preserved the same direction. Raw evidence is in
`v28-keep7/direct-32k-all`.

The three-round short fixed-all comparison measured 284.50 native versus
285.57 Splash tokens/s, a 0.37% deficit, with 203 native and 217 Splash cycles.
Decode clocks were 1619–1620 MHz. Raw evidence is in
`v28-keep7/direct-short-2k-all`.

The completed 2K fixed-all comparison measured 271.72 native versus 262.73
Splash tokens/s, a 3.42% native advantage, with 211 native and 233 Splash
cycles. The first two pairs were clock-matched; both third samples were
thermally throttled. The third Splash sample is preserved separately in
`v28-keep7/direct-2k-splash-retry1` after quiet-gate retries.

The three-round adaptive comparison showed larger deficits: 7.45% at short,
11.31% at 2K, and 11.76% at 32K. Median native rates were 259.30, 238.78 and
207.35 tokens/s versus Splash 280.18, 269.21 and 235.00. Adaptive trajectories
varied across fresh processes, as expected; output repetition was not required.
The first two pairs generally had matched clocks and the third was throttled.
Native fixed verification was faster than native adaptive at all three lengths.
Raw evidence is in `v28-keep7/direct-all-auto`.

A V29 experiment moved only the adaptive cost timestamp and committed-state
submission ahead of host-only emission, leaving the fixed schedule unchanged.
It passed 105 DSpark tests and both nine-case lifecycle matrices. All five
quiet-preflight attempts were skipped, so it has no valid performance result.
The source change was reverted and V28 remains selected. The diagnostic
candidate and rejected attempts are preserved in `candidate-v29-adaptive-cost`
and `v29-adaptive-cost/`.

The selected V28 AR regression could not be measured in this resume: all three
short preflight attempts were skipped because no quiet window was available.
Earlier V24 directional results remain historical rather than a V28 regression
matrix. The V29 working package was then restored to the frozen V28 addon. The
subsequent V30 cleanup and final review fixes are described below. Final fixed
and adaptive native/Splash evidence is complete for that earlier checkpoint; a
quiet-window AR check and final-source performance validation remain open.

## Cleanup and measurement status

The keep-7/8 context crossover is now the default, with keep 1 through 6 on
retained-height projection. The former context override was removed. MoE-only
context/budget tracing, phase and verify-layer timers, SDPA plan logging,
standalone verifier capture, and the teacher-forward schedule diagnostic were
removed with their module-local diagnostic tests. Dense diagnostics that
predate this work remain unchanged. Two dead acceptance helpers, two temporary
trace profiles and two Python cache directories were removed off-tree; benchmark
evidence and reusable drivers remain preserved.

The user subsequently noted active gaming on the machine. Recent timing
results are therefore treated as contaminated rather than final performance
claims. No further performance runs are started during this period. The cleaned
and rebased source completed its compile and correctness validation before PR
creation. A fresh quiet-window fixed/adaptive and AR matrix remains required
after gaming ends.

## Final review hardening

Focused review regressions now cover resident-weight accounting after grouped
expert tiling, mixed-projection replay operand shapes, cross-aliased K/V stores,
and cache clearing after `reuse_cache=false` turns. Embedded MoE draft discovery
checks companion geometry before selecting a draft beside a standalone GGUF.
Adaptive cycles settle committed GPU work before recording cycle cost; fixed
cycles retain their earlier schedule. Generated K-quant JIT preambles inline
their split and SG8 helpers so Qwen4 custom kernels compile from the same
sources as the prebuilt library. These are correctness fixes with focused
regression coverage. No performance benchmark was run after these fixes, so the
V28 measurements above remain historical evidence for that checkpoint.
