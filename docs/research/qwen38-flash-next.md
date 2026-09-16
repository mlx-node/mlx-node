# Qwen3.8-Flash-Next: implementation and mlx.fast research

Research: 15–17 September 2026. Starting revision:
`51bb61991479639c5237161652cc22223f3d60ee`. This report consolidates the
initial performance record, mlx.fast investigation, prefill port log, and
macOS 27 recheck. The [runtime guide](../qwen38-flash-next.md) covers usage.

## Result and limits

**The requested 90% throughput target is not established on the current host.**
Historical macOS 26.6.2 runs repeatedly exceeded the prefill threshold with
experimental BF16 operands and wide PLE projections enabled. Decode remained
below its threshold. Recent macOS 27 comparisons do not reproduce that prefill
result reliably, including when the unchanged control is repeated.

| Measurement                                | Prefill tokens/s | Decode tokens/s |
| ------------------------------------------ | ---------------: | --------------: |
| Published reference                        |          2512.69 |           63.26 |
| 90% threshold                              |         2261.421 |          56.934 |
| Historical v14 candidate, six samples      |         2317.060 |          52.934 |
| Historical v16 candidate, three samples    |         2302.827 |          53.989 |
| macOS 27 v24 control, pooled two processes |         2092.288 |          49.875 |
| 17 September three-port exact candidate    |         1962.355 |          52.313 |
| 17 September candidate with BF16/fusions   |         2250.736 |          53.764 |

These are measurements of particular builds and configurations, not a matched
engine comparison or a promise for the cleaned PR. The latest BF16/fusion candidate is
about 89.6%/85.0% of the published rates, still below both thresholds. A subsequent diagnostic returned to
about 2117 prefill tokens/s, but diagnostic timings are excluded from ordinary
throughput claims. No experimental arithmetic or later optional decode fusion
has been promoted on the basis of these drifting measurements.

## Reference, checkpoint, and method

Primary reference: [mlx.fast source at 8981cef](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine/tree/8981cef5a0a0c5b327f72cf040aa566fc61ff723),
[winning PR #580](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine/pull/580),
and the [published leaderboard](https://www.yukon.org/mlxfast), as inspected
during this research. The local clone is pinned to
`8981cef5a0a0c5b327f72cf040aa566fc61ff723`.

The reference uses affine-4/group-32 weights, a private 1024-token `botany`
tape, and serial autoregressive generation with depth zero. MTP does not
explain its score. Its winning artifact reports 64 generated steps while the
contract describes 128 checked steps; that discrepancy remains unresolved.
Our supplied checkpoint is `Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL`, with mixed
Q4_K/Q5_K and affine5/8 runtime banks. Both reported machines have a 40-core
M5 Max GPU and 128 GiB RAM. Prompt, quantization, thermals, and desktop load
are not matched.

Ordinary local comparisons use:

- 1024 input tokens and greedy 128-token output, no MTP or prefix reuse, unless
  explicitly labelled prefill-only (one output token).
- A fixed 63 GiB weight budget (67,645,734,912 bytes), 393 partial expert slots
  per layer, and no fully resident expert layers. Cache contents persist
  between legs. Earlier automatic-budget runs are identified separately.
- Two warmups per variant, generally three measured samples per variant, with
  interleaved ABBAAB order. Four-process ABBA rechecks use two warmups and three
  measurements in each process. Counts that differ are stated below.
- Identical input IDs, generated counts and output hashes within each exact
  comparison. Feature environments and immutable addon hashes are recorded.
- A process-group memory/pressure/timeout guard, normally an 88 GiB process
  ceiling and an 8 GiB competing-model threshold. Failed and interrupted runs
  are retained. No compiler or GPU probe overlaps ordinary performance runs.

The guard is not complete desktop isolation. OS file cache and thermal history
remain uncontrolled. A local 60 C request gate is distinct from the reference's
40 C per-phase gate; neither its results nor diagnostic samples are pooled with
ordinary warm runs. The 40 C attempt timed out without producing a sample.

The standard prompt consists of repetitive notes followed by an integer-list
request. A separate synthetic Rust cache-review prompt has 1023 input tokens;
it is not a captured multi-turn coding-agent workload.

| Identity                            | SHA-256                                                            |
| ----------------------------------- | ------------------------------------------------------------------ |
| Standard input IDs                  | `fb82a894b07a6c8c60cba0106e0831b98ed6f581cd525b0eff69df03ec264167` |
| Standard 128-token output           | `e9987578c1b7724ed4a9cf49f355c09600e9e4c04719154da85dd82d54f088a6` |
| Coding output at 63 GiB / 393 slots | `fa17b454f16e746f555cc09a7b4aa93313993ed2e2ce94f8bae7f91cebe45a96` |
| Prefill-only output                 | `6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b` |

## Source findings and shipped implementation

The reference was inspected locally before porting. MIT attribution remains in
[`MLXFAST-LICENSE.txt`](../../crates/mlx-sys/src/metal/MLXFAST-LICENSE.txt).
The ports adapt its execution to the existing GGUF arithmetic and bounded
partial residency; they do not replace the checkpoint or relax its admission.

| Reference source under `Runner/FastModel/`            | Local implementation and adaptation                                                                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `TrackPrefillSort.swift`                              | Ballot counts, stable ordering and inverse maps; partial-bank and tail handling retained.                                                       |
| `TrackFastMoE` and indirect dispatch                  | Device route maps and route tapes, validated before committing an output; missing slots restore and replay the original frontier.               |
| `TrackPrefillMixerAct.swift`                          | 32-row mixer activation and hyper-connection fusions with local BF16 product boundaries.                                                        |
| `TrackFastKernels2.swift`                             | GDN preparation/norm, wide normalization, injection and optional attention normalization/rotary; local reduction and rounding order retained.   |
| `TrackPrefillIndirectMetal.swift`                     | Packed group prefetch and vector stores for Q4_K/Q5_K gate/up and affine5/8 down; default F32/TF32 operands retained where required.            |
| `TrackFastModel.swift` / `TrackP12Prefill.swift`      | Shared Q8 projection pairing, split projections, bounded asynchronous submissions, and final head completion inside the route commit.           |
| `TrackFastPLE.swift`                                  | Direct four-tap convolution; wide PLE GEMM stays opt-in because its arithmetic can change continuation.                                         |
| `TrackBF16Functions.swift`                            | BF16 sigmoid table generated with this checkout's compiled sigmoid-multiply, rather than a different exponential approximation.                 |
| `TrackFastMixer.header1` / `TrackFastMoE.helpersCore` | Smaller custom-kernel helper headers, preserving register/dequantization code. Dense decode source shrinks from about 147 KB to 23,417 bytes.   |
| Reference static feature settings                     | A shared Rust/C++ thread-local settings cache scoped to one forward, refreshed on the next forward.                                             |
| `TrackFastGDNDecode` / `TrackMultiProj`               | Optional direct BF16 gate inputs and singleton pairing of the two dense 48-by-2560 gate projections. Wider windows retain separate projections. |

### Routing, lifetime, and completion

Warm partial banks previously required copying ten expert IDs to the CPU at
every layer. Tentative device routing now records the actual selections across
a token/window, retains every bank reader and owned recurrent/PLE state, and
joins them with the final output. It validates all original IDs against the
unchanged host mappings before publication. A missing slot uses an initialized,
bounds-checked placeholder, then discards that output, restores the recurrent
and paged frontier, and replays through the ordinary loader. A 16-token decode
cooldown bounds repeated misses.

Wide routing uses previous observed experts only as an eligibility hint. Fresh
routes still require commit validation. Slot replacement waits for prior
readers; only one deferred wide expert reduction can survive across a store.
MTP verification retains its separate completion/commit boundaries.

The reference's first-two/then-three-layer asynchronous schedule is bounded
inside these transactions. Final logits, route tapes, PLE/GDN state and page
writes complete together before validation. Optional deferred singleton PLE
was repaired in v25: the one-token branch now forwards the completion flag
while retaining an independent history copy and identical arithmetic.

Compact Q8 singleton projections read BF16 inputs and FP16 companions directly,
preserving lane ownership, K traversal, accumulation and output rounding.
Shared-expert fusions preserve the individual routed/shared products and their
final addition. Compiled graph wrappers receive every changing array explicitly.

### Defaults and optional controls

Eligible exact ports through reference-v9 are enabled, together with smaller
headers, `MLX_QWEN4_CACHE_FLAGS`, `MLX_QWEN4_DEFER_FINAL_COMPLETION`, and split
Metal residency sets. Runtime switches accept `0` for their control path.
Shape, dtype, hardware and transaction eligibility still select fallbacks.

Default prefill controls include `MLX_QWEN4_PREFILL_MIXER_BM32`,
`MLX_QWEN4_PREFILL_HC_MIX`, `MLX_QWEN4_PREFILL_GDN_NORM`,
`MLX_QWEN4_PREFILL_Q8_PREFETCH`, `MLX_QWEN4_PREFILL_SHARED_PAIR`,
`MLX_QWEN4_PREFILL_ROUTER`, `MLX_QWEN4_PREFILL_PLE_CONV`,
`MLX_QWEN4_PREFILL_PACKED_EXPERTS`, `MLX_QWEN4_PREFILL_REFERENCE_INDIRECT`,
`MLX_QWEN4_PREFILL_DEVICE_ROUTES`, `MLX_QWEN4_PREFILL_NORM`,
`MLX_QWEN4_PREFILL_ASYNC_WINDOW`, and `MLX_QWEN4_PREFILL_SHARED_Q8`.

`MLX_QWEN4_PREFILL_REFERENCE_BF16=1` and `MLX_QWEN4_PREFILL_PLE_GEMM=1`
are **arithmetic experiments, off by default**. Historical results exceeding
90% prefill used both. Rotary reuse/fusion, next-layer injection/norm, deferred
PLE (`MLX_QWEN4_DECODE_ASYNC_PLE`), mixer injection, shared combine, reference
down scheduling, attention normalization/rotary and direct/paired GDN gates
remain optional. Their existence is not evidence of a repeatable speedup.

### Bounded residency on macOS 27

The unchanged v19 addon repeatedly hit GPU timeouts on macOS 27. The reference
splits wired allocations across residency sets capped at 5% of the device's
recommended working set, with a 64 MiB floor and at most 32 sets. Porting this
behavior fixed the standard workload; selecting one set on the same new addon
still timed out. `MLX_RESIDENCY_SET_MAX_PCT=0` is the one-set rollback.

The overlay in `crates/mlx-sys/metal-residency` preserves allocation accounting,
locking, oversized-allocation handling, and attachment of new sets to every
queue before command-buffer commit. `build.rs` creates it under `OUT_DIR`, and
CMake plus the bridge consume identical headers. The MLX submodule is unchanged
by this optimization update. CPU/CUDA builds do not use the overlay. The build
also links the compiler runtime needed for the platform availability predicate.

Eleven standalone lifecycle/queue checks passed with at most 256 MiB of test
allocations. That research harness is archived outside the PR. The port does
not raise total wired bytes, weight capacity, expert slots, or system reserves.

Full hot weights occupy 83,783,503,360 bytes; fixed dense banks occupy
5,245,160,960 bytes. Full hot residency needs roughly 100 GiB of the planner's
live available estimate, including reserves/staging/workspace. Recorded
estimates did not support it, and reserves were not lowered. Routed storage is
only about 4% larger than equivalent reference affine4 banks; fixed Q8 storage
is about 1.8 times affine4/group32. Size alone does not explain the prefill gap.

## Measurements

Values below are medians in tokens/s. Comparisons are valid within their stated
run; cross-run differences are not attributed to code. All successful exact
legs preserve the hashes for their configuration.

### Initial support and decode pass

The initial support record measured 1320.2 prefill / 21.39 decode with automatic
72 GiB admission and 454 slots, versus 1292.8 prefill for its local control.
Peak guarded footprint was 73.44 GiB. This predates the reference ports.

| Same-run comparison                              | Control prefill/decode | Candidate prefill/decode |
| ------------------------------------------------ | ---------------------: | -----------------------: |
| Compact Q8, ballot counts, device routes         |        1359.28 / 21.66 |          1409.35 / 31.02 |
| Add GDN preparation and mixer fusion             |        1028.27 / 20.58 |          1156.92 / 31.51 |
| Singleton routing and reader completion          |        1394.19 / 22.16 |          1502.03 / 40.96 |
| Cached graphs and shared-expert fusion           |        1375.76 / 22.14 |          1473.02 / 44.87 |
| Corrected initial defaults, fixed 63 GiB         |        1369.31 / 21.98 |          1476.00 / 44.81 |
| Corrected defaults, automatic 71 GiB / 448 slots |        1351.77 / 21.46 |          1477.57 / 44.86 |
| Coding prompt, automatic 72 GiB / 454 slots      |         541.27 / 17.89 |           587.51 / 31.35 |

The corrected fixed-budget candidate's three prefill samples were
1481.70/1476.00/1467.50; decode 44.90/44.58/44.81. Peak footprint was 64.26 GiB.
The automatic-budget coding run ranged from 549.77–667.72 prefill and
30.68–31.99 decode, with a 73.06 GiB peak. Its output hash was
`241721e02a414d72215aafc838611b604466f4ccf74ccfbef6d2eada081d7417`.
It establishes parity within that capacity, not across cache capacities.

Earlier mixer-fusion results used an exponential approximation later found to
change coding output. A twelve-variant diagnostic isolated the issue. At a
sigmoid input of -6.84375, a BF16 boundary produced -0.032958984375 instead of
-0.03271484375. Generating the 128 KiB lookup table with the native compiled
sigmoid-multiply and a dynamic multiplier of one fixes that regression. The
four-row native regression and full coding output then match. Earlier timings
remain historical experiments, not final validated-default results.

### Prefill ports on macOS 26.6.2 (25G83)

| Run                                     |  Control | Candidate | Scope                                        |
| --------------------------------------- | -------: | --------: | -------------------------------------------- |
| v2 mixer/GDN                            | 1398.748 |  1392.847 | Prefill-only, six samples each; inconclusive |
| v3 Q8/pairing combined                  | 1321.657 |  1367.894 | Prefill-only, three samples each             |
| v4 routing/convolution                  | 1111.079 |  1157.181 | Prefill-only; control had drifted            |
| v5 packed always-indirect               | 1351.973 |  1512.304 | Prefill-only, four samples each              |
| v6 add device routing                   | 1414.352 |  1617.545 | Prefill-only, four samples each              |
| v7 exact ports plus asynchronous window | 1534.903 |  1972.589 | Prefill-only, three samples each             |
| v12 exact versus experimental BF16/PLE  | 2033.948 |  2325.742 | Prefill-only                                 |

V7's candidate samples were 2030.362/1972.589/1854.302; TTFT median fell from
667.143 to 519.115 ms. Isolated packed-kernel loop gains are not substituted
for these end-to-end measurements. Runs that changed arithmetic in one loaded
process sometimes nearly doubled latency; their mixed medians do not isolate
BF16 performance, so later checks used fixed process configurations.

V14's four-process ABBA comparison produced six candidate prefill samples:
2318.837, 2312.574, 2326.658, 2320.915, 2315.283, 2297.203. All exceed the
2261.421 threshold; median 2317.060 is 92.2% of the published reference.
Decode median 52.934 is below its target. V16 measured 2299.093/52.220 with
final completion off and 2302.827/53.989 with it on. Its candidate prefill
samples were 2302.827/2305.139/2292.461, all above threshold.

Later optional-port comparisons retain their slower results:

| Run/configuration                  |  Prefill | Decode |
| ---------------------------------- | -------: | -----: |
| v16 BF16 defaults                  | 2272.352 | 54.179 |
| v16 plus four reuse ports          | 2225.289 | 54.782 |
| v16 plus all earlier exact fusions | 2254.462 | 54.821 |
| v17 control                        | 1987.136 | 53.953 |
| v17 reference down schedule        | 2026.381 | 53.468 |
| v17 down plus decode ports         | 2064.572 | 55.338 |
| v18 attention plus decode ports    | 2089.648 | 55.824 |
| v19 attention plus decode ports    | 1688.352 | 54.002 |

The best listed decode median, 55.824, is about 88.2% of the reference. Late
samples slowed across configurations. No later option was promoted from these
results. The v19 clock decline is observed, but its cause is unproven.

### macOS 27.0 (26A428), High Power recheck

System Settings visibly confirms High Power on the adapter; `pmset` reports
AC `powermode=2`. Simultaneous `system_profiler` labels report Low Power Yes /
High Power No, so those labels cannot establish the selected mode. The earlier
Low Power explanation is withdrawn. No power setting was changed by the agent.

The 20-leg v23 comparison produced:

| Configuration              |  Prefill | Decode |
| -------------------------- | -------: | -----: |
| Attention/decode control   | 1838.928 | 46.137 |
| Direct GDN gate inputs     | 1795.497 | 46.924 |
| Add singleton gate pairing | 1794.451 | 46.417 |
| Disable attention fusion   | 1782.628 | 48.249 |

GPU temperature was 87–94 C and clocks varied. All outputs and memory guards
passed. This does not establish a gain from the optional GDN changes.

A separate residency 5%/2%/2%/5% ABBA comparison used a local 60 C per-request
gate. Process medians were 914.373/49.371, 815.719/48.730, 811.285/49.084 and
761.662/49.184. These are not pooled with warm comparisons. The 5% default stays.

V24 tested reference Metal 4.1 compatibility in four fresh processes, off/on/on/off:
2088.782/49.684, 2087.869/49.637, 2094.635/49.959 and 2104.472/50.116.
Pooled medians were 2092.288/49.875 off versus 2093.466/49.676 on. Sixteen
isolated cross-buffer cases matched, but no model gain was established.
**The compatibility experiment was removed during PR cleanup.**

V25 repairs deferred singleton PLE completion. Its 15-leg comparison measured
1020.780/45.011 with the flag off, 1028.449/45.066 on, and 974.036/45.030 for
a minimal BF16 configuration plus deferred PLE. A fresh fixed-configuration
v24/v25/v25/v24 comparison produced:

| Process |  Prefill | Decode |
| ------- | -------: | -----: |
| v24 A   | 2136.996 | 51.462 |
| v25 B   | 2024.711 | 49.856 |
| v25 C   | 1159.293 | 48.528 |
| v24 D   | 1309.421 | 48.248 |

The unchanged control slows too. Whole-request GPU samples ranged from
1606–1619 MHz in A to 1008–1447 MHz in D, with nearly continuous activity.
One-second samples do not isolate prefill or establish why clocks changed.
All hashes/guards passed; deferred PLE remains off by default.

The subsequent ordinary prefill-only samples were 1343.232/1502.858/1576.088.
A later route-trace diagnostic measured 2117.048/2117.173/2121.934 with all
four warm prefills committing resident hits and no replay. It rules out replay
in that diagnostic, not in every earlier slow run.

## Correctness, diagnostics, and rejected experiments

Default ports retain local arithmetic. Regressions cover mixed quantization,
partial tiles, offset views, changing graph inputs, router ties/NaNs/infinities,
reader lifetimes, paged/recurrent rollback, retained prefixes, MTP frontiers,
multiple owners, cancellation and media positions.

The experimental BF16/PLE candidate is not bit-identical. An eight-case,
128-step teacher-forced replay has 13/1024 strict argmax differences, at most
4/128 for one prompt, with finite logits. It passes the local 10% per-stream
rule, which is neither a general quality evaluation nor the private organizer's
gate. V25's eight fresh candidate journals equal v24's; its numerical result
reuses v24's fresh baseline replay on those identical histories, not a newly
run v25 baseline replay.

Two first-token distribution comparisons (248,320 logits) additionally found:
standard maximum error 1.06543, RMSE 0.18630, cosine 0.99525 and KL 0.00007653;
coding maximum error 0.90625, RMSE 0.16218, cosine 0.99624 and KL 0.00131370.
Both argmax IDs match, but coding continuation changes. These are limited
numerical evidence and do not justify default promotion.

Before cleanup, v25 passed 94 targeted native tests (one ignored real-checkpoint
harness), the complete 11-request checkpoint smoke with all 1365 blocks released,
254 forced decode replays (128 all-hit and 126 real misses), one forced wide
prefill replay, and all four exact coding comparison legs. Smoke covers AR,
streaming continuation, native/adaptive MTP, owner changes, cancellation recovery,
image continuation and replacement. Its smaller admitted plan had 366 slots.

The singleton PLE regression checks actual readiness before evaluating either
output, then exact F32/BF16 output and retained history across mixed singleton
and longer windows. A fresh CPU sample shows zero PLE event-wait samples versus
125/2282 model-thread samples before the fix (2557 samples afterward). This
confirms removal of the observed wait, not an end-to-end throughput gain.

Serialized profiles force stage completion and change scheduling. The initial
profile took 809.143 ms; ordinary execution of that build measured 701.416 ms.
A recent serialized profile took 618.668 ms, including experts 204.095 ms, GDN
projections 72.943 ms, attention branch 47.617 ms and recurrence 43.750 ms.
Warm windows had no expert misses, uploads, source reads or packed reads. These
intervals are neither ordinary throughput nor pure GPU durations.

Rejected or unpromoted research includes:

- FP16/TF32 operand substitutions that changed tens of thousands to over a
  million BF16 values across representative shapes; not equivalent storage ports.
- Wider dense NAX tiles and smaller singleton row groups without consistent gains.
- A wide router that matched IDs but changed normalized scores; only validated
  probability-first arithmetic is retained.
- GDN pairing across multi-token windows, which changed rounding; pairing is now
  restricted to singleton projections, with wider cached-pair fallback tested.
- Attention normalization using a different reduction tree; the retained optional
  port follows this checkout's reduction order and handles actual strided Q views.
- Metal 4.1 compatibility, removed after the four-process no-gain result.
- Vectorized fragment access: all 18 complete-output cases matched, including
  inactive experts and 1/15/16/17/31/32/33/65-row tails, but isolated changes ranged
  from -4.21% to +2.52%. The prototype was not integrated.

Full BF16 checkpoint generation, real-model contexts beyond 2048 tokens, audio,
video and image-bearing MTP remain unvalidated or unsupported as stated in the
runtime guide. Sparse thresholds are covered by small fixtures.

## Follow-up source audit, 17 September

Starting from the cleaned `f76ee4be` build, the renewed audit follows the
winning PR's source and mechanism list directly. Its four-row GDN schedule,
indirect expert tile table, packed NAX prefetch, split wide projection inputs,
route sorting, and first-two/then-three-layer submission schedule already
exist locally. The remaining exact candidates identified in this pass are:

| Reference mechanism                     | Local gap and port                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TrackFastKernels.leanRowsSource`       | The local four-row recurrence still used scalar loads, six pointer increments, and a scalar-array sequence length. `MLX_QWEN4_GDN_VECTOR_ROWS=1` uses vector loads, three shared offsets, and a template sequence length. It retains modulo head mapping, ordinary FP32 reductions, FP32 output, and FP32 state. The reference's Kahan arithmetic and consecutive head mapping are deliberately not copied into this GGUF path. |
| `TrackFastMoE` gate/up and down staging | Routed and shared gate/up activation writes were still serialized on lane zero. `MLX_QWEN4_EXPERT_LANE_STAGING=1` distributes complete reduced rows across lanes and independently enables distributed down-result staging at the existing four-row geometry. The older two-row schedule remains a separate choice.                                                                                                             |
| `TrackFastMixer.upMixSource`            | `MLX_QWEN4_MIXER_LANE_PRODUCTS=1` computes the four independent stream products in four lanes, then retains this checkout's ascending BF16 sum and division by four. It keeps the current weight layout and one-column SIMD ownership.                                                                                                                                                                                          |
| `TrackFastMixer.downInjectSource`       | `MLX_QWEN4_MIXER_DOWN_INJECT=1` fuses singleton mixer down/SiLU and injection projections into one dispatch, keeping the local affine8 lane walks and BF16 boundaries. It reads the existing two banks and adds no resident weights.                                                                                                                                                                                            |

These switches remain off pending repeatable full-model comparisons. The
recurrence regression checks every output and state bit for F32/BF16 inputs,
contiguous/strided/offset storage, modulo-tiled key heads, 7/9/1024-token windows and
one-token continuation. It passed on this host. Expert and mixer regressions
reuse ordinary projections and the native sigmoid halfway cases as independent
references. All 98 Qwen4 native tests passed on the complete three-port candidate (39.35 seconds).
The later down/injection fusion passed its independent projection regression,
including changed weights, offset views and unsupported-input fallback; the
expanded 99-test Qwen4 suite passed in 57.40 seconds.

A four-process, uncontended control/vector/vector/control check passed all
hashes and guards at the required 63 GiB/393-slot plan. Process medians were
1959.888/52.156, 1974.653/52.093, 1955.831/52.080, and 1932.010/52.212
(prefill/decode tokens/s). The small prefill change is insufficient to approach
90%; decode is unchanged. An isolated 1024-token recurrence probe matched
all outputs and final states exactly for both input dtypes. After its first
control block, FP32 control blocks took 1.463/1.444 ms versus candidate
1.382/1.394/1.404 ms; BF16 block times overlap. These kernel timings do not
replace the complete-request comparison.

The first same-binary combined check measured 1951.665/51.256 for its control
and 1962.355/52.313 for the three-port candidate (three measured requests per
process; 63 GiB/393 slots; hashes and guards passed). The control's last
prefill sample dropped to 1665.860, so this pair alone is insufficient for
promotion. Two subsequent optional-fusion runs stopped before loading because
another worktree was running Vitest. Those runs have no throughput result.
The later down/injection fusion is not included in this initial comparison.

With all three ports, the earlier optional fusions, BF16 prefill operands and
wide PLE enabled, an ordinary guarded run measured
2247.148/2250.736/2260.135 prefill and 53.851/53.764/53.403 decode tokens/s.
Its medians are 89.6%/85.0% of the published rates, still below both thresholds.
The standard output hash matched on every request; the prior BF16 numerical
limits still apply. This run also predates the down/injection fusion.

The following minimal-fusion run overlapped a `k2-horizon` Rust build and is
excluded from optimization comparisons. One-second telemetry reaches full
CPU utilization during that run. The guard now also rejects competing
mlx-node compilers; it had previously checked only model footprints and MLX
tests. This is evidence of contention in that particular run, not a measured
regression from disabling fusions.

A separate singleton GDN prototype ports the reference's first-state-row
prefetch and register-cached q/k values. It matched all three outputs for two
input phases. Warm control blocks took 153.474/153.187/153.607 microseconds;
candidate blocks took 155.159/150.237/150.881 microseconds. This small,
overlapping result does not justify another production variant. The prototype
remains in the external evidence archive.

A separate CPU-side audit found repeated deep clones of the immutable model
configuration in forward helpers. The decoder now shares that configuration
through `Arc`, matching the reference's retained configuration without changing
any tensor or arithmetic. Its measurement is separate from the kernel ports.

The shared-configuration candidate passed all 99 Qwen4 tests and the broader
core release suite (3561 passed, 122 ignored, three existing debug-only
assertions excluded). Its full-checkpoint smoke passed all 11 requests and
released all 1365 pages and state reservations. Smoke used its actual
63 GiB/366-slot auxiliary-model plan and a 65.21 GB peak physical footprint.
This was correctness-only validation under the memory/pressure guard; other
small tests/builds were allowed, so its timings are not throughput evidence.
The canonical addon build and strict all-target Clippy also passed.

PR #154's previous-head CI exposed a compact-gate rounding difference in the
older-GPU compiled GDN fallback. Forcing that fallback on this M5 reproduced
it: the compact recurrent state differed in 203922 elements by at most
1.8626451e-9 at the first step. The fallback now promotes gate inputs to F32
before entering the compiled graph, restoring the original input signature;
the fused Metal path remains unchanged. A child-process regression explicitly
forces this fallback on every Metal host, with the existing exact output,
state and history assertions intact. Both the original replay test and the new forced-fallback regression passed
with zero differences in outputs, states and retained histories.

A later four-process candidate/control/control/candidate check retained the
same 63 GiB/393-slot plan, guards and standard hash. The two three-port control
processes measured medians of 2244.979/53.939 and 2238.188/53.818 tokens/s.
The down/injection plus shared-config candidate measured 1939.126/52.888 and
2207.549/53.514. The first candidate's prefill slowdown did not repeat at that
magnitude, but this comparison does not demonstrate a gain from the new fusion.
A preceding 20-leg same-process feature screen also passed all hashes and the
guard, but had large within-variant timing swings and cannot establish a winner.
Those samples are retained rather than selected for a favorable result.

A separate packed-mixer probe ports the reference's two-column threadgroup,
eight-product staging and reordered rows, with a control retaining the local
layout. All outputs matched bit-for-bit for two changing input/weight sets.
Repeated warm blocks overlapped: control 141.5–151.2 microseconds, unpacked
reference geometry 143.3–150.3, packed geometry 140.9–167.2. This includes
per-dispatch evaluation overhead; it is not a full-model gain. No duplicate
mixer banks or associated residency change are added based on this result.

The reference also uses source/compile-option hashes as Metal library keys.
Our backend compares sources under the kernel name. A literal hash-key port
still scans the source on every evaluation; source inspection alone does not
establish it as a warm-dispatch speedup. It has not been promoted. The
reference's packed mixer-row layout additionally retains reordered weights;
that is a separate residency/accounting change from the parallel-product port.

The first fresh default run was admitted at 386 expert slots rather than 393,
so it is excluded from fixed-budget comparisons (prefill
1951.314/1893.519/1666.196; decode 52.043/52.018/51.285 tokens/s). A subsequent
BF16 run overlapped another worktree's native MLX GPU suite and is also excluded
(prefill 905.610/859.015/874.646; decode 41.203/41.414/40.927). Its low rates are
not evidence of a BF16 regression. The external harness now requires exactly
63 GiB/393 slots/1024-token windows and stops on competing native MLX tests,
including tests below the existing 8 GiB memory threshold. It stops only its
own child group. This observed interference applies to these runs; it does
not retroactively establish the cause of earlier timing drift.

New local evidence lives in
`~/Library/Caches/mlx-node/qwen4-reference-gap-20260917/`, including immutable
control/candidate builds, invocation records, memory guards, telemetry, and
external probes. No harness or temporary file is added to the repository.

## PR cleanup and validation

Standalone benchmark, smoke, fixture-generator, memory-guard, profiling and
prototype sources, raw measurements and temporary binaries are archived outside
the repository. This is the single Qwen3.8 research report in the PR. Native
regression tests and their deterministic synthetic fixture data remain in-tree.
No new performance measurement is inferred from cleanup.

Two review findings are corrected: GGUF extension matching is case-insensitive,
and the advertised scheduler capacity returns one under `MLX_SERVE_FORCE_SERIAL`.
Their regressions exercise renamed GGUF fixtures through logits and isolated
processes reading the real scheduler policy. The TypeScript chat-family assertion
now includes `qwen4_exp`.

Tiny unit fixtures use an explicit bounded fixture store and injected bank-load
admission/refresh callbacks. They no longer request the production multi-GiB
working reservation from a busy CI runner. Actual payload reads still use live
checks; production plan/reserve behavior is unchanged, and planner snapshot tests
retain its insufficient-headroom coverage.

Post-cleanup local validation (17 September):

- Canonical native addon build and packaged metallib smoke checks passed.
- Qwen native suite: 96 passed, zero ignored. Broader release core suite: 3558
  passed, 122 ignored; three existing debug-assertion tripwires were excluded
  because release builds disable their assertions.
- Full TypeScript run: 3750 passed, 40 skipped. Its six failures were all missing
  local GSM8K data; all eight dataset tests passed when retried with the existing
  dataset. The temporary dataset link was removed afterward.
- After rebuilding the addon, all 79 loader/registry/paged-policy/stream/agent
  tests across the five affected suites passed.
- TypeScript build/type checking, type-aware lint, strict all-target Rust Clippy,
  Rust formatting, scoped formatting/lint checks, and staged whitespace checks
  passed. Repository-wide `vp check` still encounters pre-existing formatting
  issues outside the PR; those unrelated files were not reformatted.
- Full-checkpoint smoke, numerical fidelity and performance were not rerun for
  this cleanup. Their pre-cleanup v25 evidence and limits are stated above.

Cleaned addon SHA-256:
`e5d6f3fb2cc2df130926abeac7ca7b9a5401728db66ccd690dc2931e149457be`.
The two packaged metallib hashes are unchanged from the v25 values below.
Remote CI is separate from these local checks.

## Evidence archive

Local evidence root:
`~/Library/Caches/mlx-node/qwen4-prefill-pass15-20260915/`.
The preceding audit remains at
`~/Library/Caches/mlx-node/qwen4-mlxfast-recheck-20260915/findings.md`.
These local artifacts are not committed and are not portable repository links.

The archive preserves exact plans, prompt IDs, token journals, per-leg samples,
output hashes, telemetry, guard records, failed runs, compiler invocations,
source snapshots and immutable native builds. Useful entry points include:

- `paired-default-final*`, `paired-coding-final*`, and the initial performance
  record under `cleanup-pr154-20260917/docs/research/`.
- `reference-v5-prefill-paired*`, `reference-v7-prefill-paired*`,
  `reference-v14*`, `reference-v16-final-paired*`, and `reference-v16-reuse-paired*`.
- `reference-v19-measurement.json`, `reference-v23-validation.json`,
  `reference-v24*`, `reference-v25-validation.json`,
  `reference-v25-ple-paired-summary.json`, and `reference-v25-fixed-recheck-summary.json`.
- `nax-vector-probe*`, `review-20260917-v25-final/`, its source patch/hash manifest,
  and `cleanup-pr154-20260917/removed.json` for the pre-cleanup archive inventory.

The pre-cleanup v25 addon SHA-256 is
`a5f47a8c0463d0bfd3ee6ede82993b85bb9267b1ba076aa23e0badb157112055`;
MLX metallib `7af3f85873830aecc1c516d2e7f7ed246d8e2aaacce95ee9ade765d2faf7a97c`;
paged metallib `bcc26de5b3cd65061a17a43fdd3ebe13e348d1e27caa3a5213f9d5c798df31d1`.
These identify the earlier correctness/performance evidence, not the subsequent
cleaned build. Ordinary timings must always be associated with their recorded
build, settings, workload and memory plan.
