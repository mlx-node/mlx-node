# Pure MLX TTS memory investigation

This investigation stays on `codex/tts-streaming`, with the production native
addon rebuilt from `3e8c9d8`. ANE work is preserved on `feat/tts-ane` and is not part
of these measurements. No model precision, sampling, conditioning, text
segmentation or tempo algorithm changes are involved.

The later `TTS_MLX_CACHE_LIMIT` implementation (`13a34b7`) and its new release
build were used for the follow-up at the end of this report. **Both 1 GiB and
0.5 GiB passed that repeat long-playback test.** Earlier failed runs remain below;
they establish failures under their recorded conditions, not that those cache
budgets are inherently unsuitable.

For the subsequent native allocation changes, smaller-cache sweeps and 1.7B
measurements, see [the cache-floor follow-up](tts-cache-floor.md).

> **RTF context (2026-10-10):** the 0.5–0.89 native RTF figures below were
> recorded on the earlier, slower decode path and the older converted 8-bit
> checkpoint; they are historical comparisons of allocator-cache policy, not
> current speed. On the current head with dense BF16 official checkpoints, the
> same CLI measures medians of 3 runs at **RTF 0.24–0.33** (0.6B ≈ 0.24–0.26,
> 1.7B ≈ 0.29–0.33; ~20 ms/frame vs ~26 ms/frame per 12 Hz frame). The
> allocator-policy conclusions and memory methodology below remain valid; only
> the absolute RTF level has moved.

## What the counters mean

- `physicalFootprintBytes`: macOS's process physical footprint from
  `proc_pid_rusage(RUSAGE_INFO_V4)`. This is the relevant process-level accounting
  for the unified-memory question; it differs from RSS.
- `peakPhysicalFootprintBytes`: the kernel's cumulative high-water mark **as of
  the latest successful probe**. External sampling can miss a final increase
  immediately before process exit.
- `rss` / `residentBytes`: resident memory; do not add it to physical footprint.
- MLX `activeBytes`: buffers owned by live arrays. `cacheBytes`: released buffers
  retained for reuse. `peakBytes` is a peak of active MLX memory, not peak process
  footprint. These overlap process accounting and must not be added to it.
- Node `heapUsed`, `external`, and `arrayBuffers` help distinguish JS allocations
  from native/Metal allocations; they are not independent additive categories.

The unchanged model has about 1.92 GiB of active MLX buffers near segment starts.
The default long run retained about 2.97 GiB in the allocator cache. Its JS heap
was only about 7.7 MiB. A historical approximately 2 GiB physical-footprint
measurement was taken **after load**, before voice preparation and generation;
comparing that with a steady-state long run confuses lifecycle phases.

## Protocol

Apple M4 / 32 GB, macOS 27.0.1, Node 24.21.0; production Node → Rust → MLX.
Official 0.6B Base converted with the existing 8-bit recipe, source revision
`5d83992436eae1d760afd27aff78a71d676296fc`. Fairy Chinese reference, seed 534,
unchanged sampling, 160 ms native chunks and 1.15 audio speed.

The addon SHA-256 used for the initial measurements below is
`482fdf61eb320a6eae4122efc34d9d6e6ddf2f70113be7b08b279f093d1fc931`.
The benchmark records the actual loaded addon and SDK paths, hashes, `execArgv`
and `NODE_OPTIONS`. Compile TypeScript first; ordinary Node resolves the built
SDK, without a source export condition. Xcode, other model runs, builds and
tracing were not run alongside the timed experiments. This is a live desktop,
not an isolated laboratory; background work and thermal/frequency drift remain
possible.

Long playback uses the original story twice (4,486 graphemes, 200 segments),
simulated delivery at 20 graphemes/s in two-grapheme chunks, a 3-second playback
ring and a 1-second **startup** prebuffer. Runtime ring capacity and startup
latency are separate controls. This restores the earlier smooth MLX playback
policy. The interrupted 1-second-ring / 30-graphemes/s trials are not acceptance
runs. The user reported the restored default run sounded substantially smoother.

## Long playback: keep the failure visible

| Measurement                                                      | Default allocator policy | 1 GiB allocator cache |
| ---------------------------------------------------------------- | -----------------------: | --------------------: |
| Raw generated audio                                              |                 848.32 s |              848.32 s |
| Actual played audio                                              |               737.6694 s |            737.6694 s |
| Segments / completion                                            |                200 / EOS |             200 / EOS |
| Native RTF                                                       |                 0.631612 |              0.808346 |
| Native first PCM                                                 |                214.52 ms |             252.19 ms |
| Processed first PCM from submission (includes text accumulation) |              1,217.03 ms |           1,256.00 ms |
| PCM delivery interval p95                                        |                153.99 ms |             202.22 ms |
| Player underrun callbacks                                        |                        0 |                 4,720 |
| Observed cumulative peak physical footprint                      |                5.096 GiB |             3.365 GiB |
| Median sampled physical footprint after 120 s                    |                5.093 GiB |             3.123 GiB |
| Peak RSS                                                         |                2.140 GiB |             2.101 GiB |
| Ten-minute playback gate                                         |                   passed |            **failed** |

Both WAV files have SHA-256
`f3f4ebf7ec634982f191f2fc0869f70f50f6e8034380f87c531dacba1bd0ad3c`.
This establishes identical saved PCM16 for this fixture, not identical delivery
timing. The original default long run predates Float32 hashing, so full-long
Float32 equality is not claimed. The callback count is not a count of distinct
audible interruptions or a measurement of missing duration.

These long runs were sequential, not randomized: their difference alone does
not establish that cache reduction caused the slowdown. Nevertheless the 1 GiB
run failed playback acceptance and cannot be promoted on the strength of memory
savings or faster-than-playback **average** RTF. Delivery variance matters.

The follow-up 0.5 GiB long run, started after 180 seconds without inference
(nominal thermal state throughout that pause), also failed: native RTF 0.796133,
541 underrun callbacks, peak footprint 3.274 GiB and median steady footprint
2.611 GiB. It completed all 200 segments / 737.6694 played seconds. Native first
PCM was 208.91 ms and delivered PCM interval p95 was 176.33 ms. WAV SHA-256 still
matched the two earlier long runs, and both Float32 hashes matched the 1 GiB
long run. Neither smaller budget passed this initial full playback gate. A repeated
default-cache control used the same updated harness, thermal sampling and
180-second inference pause to help distinguish cache effects from changes in
the machine's operating state.

That repeated default control also fell behind. During it, a point CPU sample
observed two Chrome renderer processes at 131.0% and 35.2%, plus other desktop
work. A later saved snapshot no longer showed those processes among the top
CPU users: this is evidence of changing load, not a sustained trace or proof
that Chrome caused every gap. The run was deliberately interrupted with SIGINT
when the user offered a quiet test window. The runner recorded exit 1 from the
benchmark's AbortError; no final benchmark report was produced. Partial logs,
memory observations and interruption provenance are retained separately, and
the run is **not** counted as a completed control or acceptance result.

After the user offered a quiet window, a more conservative **2 GiB** allocator
cache completed the full playback gate. The preceding 180-second pause contains
no inference. The model, input, seed, precision, playback policy and tempo were
unchanged:

| Measurement                          | Earlier default | 2 GiB cache, quiet-window retest |
| ------------------------------------ | --------------: | -------------------------------: |
| Actual played audio                  |      737.6694 s |                       737.6694 s |
| Segments / completion                |       200 / EOS |                        200 / EOS |
| Native RTF                           |        0.631612 |                         0.621080 |
| Native first PCM                     |       214.52 ms |                        196.47 ms |
| Player underrun callbacks            |               0 |                                0 |
| Observed peak physical footprint     |       5.096 GiB |                        4.232 GiB |
| Median sampled footprint after 120 s |       5.093 GiB |                        4.009 GiB |
| Peak RSS                             |       2.140 GiB |                        2.104 GiB |
| Ten-minute playback gate             |          passed |                           passed |

Steady footprint decreased by **1.085 GiB (21.3%)**, and observed peak by
**0.864 GiB (17.0%)**. WAV SHA-256 matches the original default; raw and processed
Float32 hashes match both smaller-cache long runs. Thus the memory setting did
not change generated audio in these fixtures. The earlier default and quiet
retest are not a paired performance experiment; the 1.7% lower RTF is an
observation, not a demonstrated speedup. Short alternating pairs below further
check performance under closer operating conditions.

This provides a measured, explicit deployment option for this dedicated TTS
process. It does not establish a universally optimal cache size or justify
changing the shared runtime default for other models. Smaller budgets retain
their initial failed acceptance results, alongside the successful repeats below.
No model-specific allocation rule was added.

The original `audioProcessing.realTimeFactor` summed native service time and DSP
service time. These intervals may overlap. The benchmark now calls this
`timing.measuredServiceTimeRatio`, separately records observed wall time, and
checks cancellation and actual drained playback duration. Native RTF remains
`synthesis.synthesisMs / (1000 * synthesis.audioSeconds)`. Input wait and consumer
wait are recorded separately and are not summed or subtracted as if disjoint.

## Correcting the early 0.5 GiB interpretation

Five alternating pairs originally produced group medians 0.545111 and 0.587790.
Calling that a demonstrated 7.8% regression was premature:

| Pair | Order             | Default RTF | 0.5 GiB RTF | Paired change |
| ---- | ----------------- | ----------: | ----------: | ------------: |
| 1    | default → limited |    0.519350 |    0.519757 |        +0.08% |
| 2    | limited → default |    0.518545 |    0.516383 |        −0.42% |
| 3    | default → limited |    0.545111 |    0.587790 |        +7.83% |
| 4    | limited → default |    0.674610 |    0.632901 |        −6.18% |
| 5    | default → limited |    0.746752 |    0.820384 |        +9.86% |

Default itself drifted by 43.8% from first to last pair. Median paired change was
+0.08%; geometric mean paired change was +2.07%. An exploratory 95% t interval
on the five log ratios spans −5.76% to +10.55%. Its assumptions are weak with
this small, time-correlated sample. Neither a regression nor equivalence is
established. All raw and processed Float32 hashes matched.

The separate five 1 GiB pairs also drifted. Their group-median ratio was +2.39%,
but median paired change was −1.12%, with individual changes −3.48% to +2.39%.
This is a further reason not to use a ratio of unpaired group medians as causal
performance evidence.

The follow-up protocol fixes six permutations of default / 0.5 / 1 GiB using
seed 534, then mirrors each permutation into a six-run block. Every condition
therefore has the same mean ordinal position within each block. All 36 runs use
fresh processes; 20 seconds separate blocks. Results compare each candidate's
mean log RTF with default **inside each block**, then summarize the six block
contrasts. This reduces linear order drift; it cannot eliminate nonlinear
thermal changes, carry-over or background load. Thermal state and process CPU
snapshots are recorded before/after every run. A nominal thermal-state enum is
not a GPU frequency measurement.

All 36 runs completed with EOS, the same addon/config hashes, the same input and
output sample counts, and identical raw/processed Float32 hashes. The system's
thermal state changed from nominal to fair during block 1 and remained fair for
the later observations. RTF across all conditions ranged from 0.517 to 0.890.

| Block                                      | 0.5 GiB change versus within-block default | 1 GiB change versus within-block default |
| ------------------------------------------ | -----------------------------------------: | ---------------------------------------: |
| 1                                          |                                     +1.03% |                                   +0.64% |
| 2                                          |                                     −4.51% |                                  −11.03% |
| 3                                          |                                    +11.05% |                                   +6.07% |
| 4                                          |                                     −0.89% |                                   −0.82% |
| 5                                          |                                     −0.39% |                                   +2.00% |
| 6                                          |                                     +0.66% |                                   −1.18% |
| Geometric mean                             |                                     +1.05% |                                   −0.86% |
| Exploratory 95% log-t interval, six blocks |                           −4.16% to +6.54% |                         −6.80% to +5.46% |

Thus a stable 7.8% penalty at 0.5 GiB is not demonstrated, and neither setting
has established performance equivalence. Memory savings are much less variable:
median observed peak footprint was 4.278 GiB (default), 3.269 GiB (0.5 GiB), and
3.324 GiB (1 GiB). These short runs cannot replace long playback acceptance.

The balanced runs switched addon provenance hashing to a bounded read stream.
Earlier runs read the entire addon into a temporary Node buffer. This change is
identical for every condition within each series; do not attribute the small
cross-series footprint reduction from that instrumentation fix to the allocator.
All run-level numbers and the predeclared order are preserved in
[the evidence JSON](research/tts-mlx-memory.json).

## Quiet-window 2 GiB follow-up

Immediately after the successful 2 GiB long playback, six more fresh-process
pairs alternated default → limited and limited → default. Both conditions use
the same final benchmark, built SDK, native addon, reference, seed, text, tempo
and one-second external memory probes. These are short synthesis runs without
playback, not additional ten-minute acceptance runs.

| Pair | Order             | Default RTF | 2 GiB RTF | Paired change |
| ---- | ----------------- | ----------: | --------: | ------------: |
| 1    | default → limited |    0.676279 |  0.638734 |        −5.55% |
| 2    | limited → default |    0.707475 |  0.655322 |        −7.37% |
| 3    | default → limited |    0.691349 |  0.714695 |        +3.38% |
| 4    | limited → default |    0.732642 |  0.717577 |        −2.06% |
| 5    | default → limited |    0.742933 |  0.696696 |        −6.22% |
| 6    | limited → default |    0.740147 |  0.763702 |        +3.18% |

Geometric mean paired RTF change was **−2.54%**; the exploratory 95% log-t
interval was **−7.38% to +2.56%**. Median paired first-PCM change was −45.01 ms.
All 12 processes completed EOS with identical raw/processed Float32 hashes and
sample counts. There was no consistent measured performance penalty at 2 GiB.
This small serial sample does not prove exact performance equivalence or a
speedup. ANECompilerService and XProtect CPU activity appeared in the saved
before/after snapshots despite the quiet user window; those records remain in
the evidence, rather than being filtered out as inconvenient observations.

The short-run median peak footprint was 4.278 GiB (default) versus 4.184 GiB
(2 GiB). Most of the saving appears in sustained synthesis as the default free
pool retains more buffers; load-time or short-run measurements alone miss it.
At that stage, **2 GiB was an explicit measured option**, retaining the three-second playback
ring and one-second startup prebuffer. The runtime default remains unchanged.

## Retest with the TTS-specific cache setting

At the user's request, the runtime now reads `TTS_MLX_CACHE_LIMIT` at TTS load
and registers its byte ceiling with the shared coordinator. The rebuilt release
addon has SHA-256
`52ff2a5bebcf43f5369a0a33bf771b3807ebdba8760a97c6dd60deb0a78569ec`.
The old global variable was explicitly unset for these runs. Source, binary,
input, environment, policy logs and measurements are retained in
[the follow-up evidence](research/tts-mlx-cache-env-retest.json); the earlier
evidence file is unchanged.

Each run started after 180 seconds without inference, used the same 200-segment
paced text and drained the entire playback. No model, build or device-profiling workload
ran alongside them; lightweight memory/thermal probes ran every five seconds
and desktop CPU snapshots every thirty seconds.

| Measurement                          | 1 GiB retest | 0.5 GiB retest |
| ------------------------------------ | -----------: | -------------: |
| Actual played audio                  |   737.6694 s |     737.6694 s |
| Segments / completion                |    200 / EOS |      200 / EOS |
| Player underrun callbacks            |        **0** |          **0** |
| Native RTF                           |     0.518218 |       0.598887 |
| Native first PCM                     |    206.90 ms |      186.07 ms |
| PCM delivery interval p95            |    151.52 ms |      152.25 ms |
| Observed peak physical footprint     |    3.328 GiB |      3.274 GiB |
| Median sampled footprint after 120 s |    3.091 GiB |      2.590 GiB |
| Peak RSS                             |    2.064 GiB |      2.064 GiB |
| Ten-minute playback gate             |   **passed** |     **passed** |

Both WAV hashes match the original default and every completed long run. Raw
and processed Float32 hashes also match the earlier hashed long runs. These
settings did not alter generated samples for this fixture. The 0.5 GiB setting
saved a further **0.501 GiB** of steady footprint versus 1 GiB.

The second run's native RTF was 15.6% higher, but this is not an isolated cache
cost estimate. The 1 GiB run had 76 nominal / 72 fair thermal observations;
the 0.5 GiB run had 28 nominal / 120 fair. A 180-second pause did not equalize
their full thermal histories, and the thermal enum does not measure frequency.
The adjacent short pairs below help check performance with less temporal
separation. Neither the earlier failures nor these successful repeats establish
a universal cache threshold. The playback verdict here uses actual drained
duration and underrun counters, without requiring a subjective listening reply.

Six short pairs then alternated 1 GiB → 0.5 GiB and 0.5 GiB → 1 GiB. All 12
fresh production processes completed EOS with identical raw/processed Float32
hashes and frame counts, the same model path, config/reference hashes and the
same new native addon. Checkpoint weight-file hashes were not recorded here.

| Pair | Order   | 1 GiB RTF | 0.5 GiB RTF | Paired change |
| ---- | ------- | --------: | ----------: | ------------: |
| 1    | 1 → 0.5 |  0.536257 |    0.526792 |        −1.77% |
| 2    | 0.5 → 1 |  0.541637 |    0.531565 |        −1.86% |
| 3    | 1 → 0.5 |  0.579211 |    0.608949 |        +5.13% |
| 4    | 0.5 → 1 |  0.743008 |    0.677532 |        −8.81% |
| 5    | 1 → 0.5 |  0.757934 |    0.795331 |        +4.93% |
| 6    | 0.5 → 1 |  0.870341 |    0.900442 |        +3.46% |

The geometric mean paired RTF change was **+0.057%** (0.5 GiB / 1 GiB);
the exploratory 95% log-t interval was **−5.56% to +6.01%**. Median paired
first-PCM change was −6.30 ms. Both conditions slowed substantially across this
serial series. Alternating adjacent runs reduces simple order bias but cannot
remove nonlinear thermal/background drift; neither exact equivalence nor a
stable cache-induced penalty is established.

Thus **0.5 GiB is also a measured usable option for this standalone workload**,
with 1 GiB available as another explicit setting. This updates the earlier
acceptance result without attributing all old failures to a single cause.
Retain the 3-second playback ring, 1-second startup prebuffer and 1.15 tempo;
do not silently change shared defaults or claim these results cover other
models, concurrent workloads or smaller playback buffers.

## Allocation and reclamation strategy

The source audit and community research support separating three lifetimes:

1. **Live state**: weights, prepared voices, writable KV and codec state. Preserve
   semantics and bound ownership; allocator trimming cannot free these.
2. **Reusable free buffers during work**: use a process-wide budget. MLX already
   matches nearby buffer sizes and evicts older free buffers when trimming.
   Avoid discarding the whole pool on every audio chunk or text segment.
3. **Truly idle process**: an explicit idle/pressure policy may reclaim free
   buffers, provided it coordinates all model loads and active streams and
   accounts for rewarming latency. A paused TTS input is still an active session.

[MLX's cache-limit API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)
limits free-buffer retention and trims on subsequent allocations; it is not a
process-memory ceiling. Our pinned Metal allocator uses this mechanism.
Use `TTS_MLX_CACHE_LIMIT` for TTS, in fractional GiB. It is read and validated
before model loading; the ceiling is registered after weights load and removed
with the model. Unset or `0` adds no TTS constraint and uses the coordinator's
normal policy. Invalid, negative, non-finite or out-of-range values fail loading.
This limits free-buffer retention during inference, not loading peaks or total
process memory. No machine/model-name rule is involved.

MLX has one process-wide free pool. The smallest live model ceiling constrains
the shared automatic/global policy, so other MLX models in the same process also
observe the cap. Removing a ceiling recomputes the surviving policy, or restores
the prior cap if no independent policy remains. The generic coordinator accepts
byte ceilings; only the TTS loader knows the TTS environment variable.

The earlier experiments used the existing process-wide `MLX_CACHE_LIMIT_GB`;
their recorded environment and hashes are preserved. That generic override
remains available to other modules. If both variables specify positive limits,
the smaller one wins. Its historical `0` means skip automatic policy, whereas
`TTS_MLX_CACHE_LIMIT=0` adds no TTS constraint; neither is MLX's API call
`set_cache_limit(0)`, which disables caching.

[MLX's KV-cache guide](https://ml-explore.github.io/mlx/build/html/usage/kv_cache.html)
explains that continuously growing concatenations both copy data and strand
old buffer sizes. Chunked capacity plus in-place updates avoids this. Our shared
`KVCache` already does this, and Code Predictor already resets logical length
while retaining capacity. Reserving every possible future token would trade
away memory unnecessarily; capacity changes need workload evidence.

[Community issue #3350](https://github.com/ml-explore/mlx/issues/3350) reports
retention from increasing allocation sizes. It motivates inspecting allocation
lifetimes, but its description is not a diagnosis of this TTS implementation.
[An idle-reclamation proposal](https://github.com/ml-explore/mlx-lm/pull/1546)
separates allocator buffers from semantic prompt caches and reports a next-call
rewarm cost. It is a closed proposal, not an assumed upstream feature. This
repository already has server-side idle coordination; the standalone TTS SDK
must not start an independent timer that ignores other native work.

[Community pressure handling](https://github.com/ARahim3/mlx-dspark/blob/main/CHANGELOG.md)
illustrates generation-thread reclamation at safe boundaries with hysteresis.
It is a future option for shared infrastructure, not a reason to poll global
memory and clear caches inside the audio callback. Such a policy needs pressure
injection and post-reclamation latency tests before adoption here.

Do not transplant speculative allocator fixes from issue titles. For example,
[MLX PR #3688](https://github.com/ml-explore/mlx/pull/3688) was rejected and the
reporting project later attributed its fault to aliased application KV state,
removing its retained-command-buffer workaround. No allocator fork, global
periodic clearing, quantization change or hidden cache default was introduced
by this investigation.

## Reproduction

```sh
vp run build:native
vp exec tsc -b packages/tts --force
vp exec tsc -p scripts/tts/tsconfig.benchmark.json
xcrun clang -O2 scripts/tts/process-memory.c -o /tmp/tts-process-memory

# Explicit candidate; omit TTS_MLX_CACHE_LIMIT for the default control.
# Keep NODE_OPTIONS empty so source conditions/TS loaders do not enter the run.
env -u NODE_OPTIONS -u MLX_CACHE_LIMIT_GB TTS_MLX_CACHE_LIMIT=2 \
node .cache/tts/benchmark/benchmark.js \
  --model .cache/tts/models/Base-8bit \
  --reference .cache/tts/fairy/reference.wav \
  --transcript-file .cache/tts/fairy/reference.txt \
  --text-file .cache/tts/ane/paced-story-twice.txt \
  --once --graphemes-per-second 20 --chunk-graphemes 2 \
  --play --speed 1.15 --prebuffer-seconds 1 --playback-buffer-seconds 3 \
  --output /tmp/tts-memory.wav --report /tmp/tts-memory.json

# From another terminal while Node is running; sample every five seconds.
/tmp/tts-process-memory <node-pid>
```

The PCM queue, playback ring and allocator cache are separate controls. Reducing
three seconds of mono Float32 playback capacity saves only hundreds of KiB and
is not an appropriate way to recover GiB of Metal memory. Preserve the playback
policy while investigating allocator budgets.
