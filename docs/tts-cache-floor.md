# Smaller TTS allocator caches and model memory

This follow-up tests the production MLX backend on the same Apple M4 / 32 GB
machine. It extends [the memory investigation](tts-memory.md). ANE and concurrent
LLM workloads are outside this experiment.

The optimized 0.6B Base passed full-story playback at **0.01 GiB**, the lowest
positive budget tested on this build. Both 1.7B models passed at **0.5 GiB**.
These are measured passes under the protocol below, not universal cache floors.
Run-level evidence is preserved in [the measurement data](research/tts-cache-floor.json).

`TTS_MLX_CACHE_LIMIT` is a free-buffer pool budget in **GiB**, not a limit on
process RAM or model weights. Zero retains the normal policy; it does not mean a
zero-byte cache. MLX changes the budget immediately but reclaims an existing
excess on subsequent allocation. Recycling one buffer can temporarily overshoot
the budget. Loading, live arrays, allocator reuse, RSS and process physical
footprint must therefore be reported separately.

## Protocol

- Ordinary Node 24.21.0 and a release/LTO native addon, with built SDK imports.
- Existing 8-bit, group-64 affine checkpoints; source codec precision unchanged.
  Base uses the Fairy reference at speed 1.15. CustomVoice uses `vivian`, and
  VoiceDesign uses `成年女性，音色清晰、沉稳，普通话发音自然。`, both at speed 1.
  Their expression instruction is `用平静沉稳的语气说。`.
- Seed 534, unchanged sampling, 160 ms PCM chunks, instruction KV cache off.
- Long playback: the same 4,486-grapheme story, available on a 20-graphemes/s virtual clock in
  two-grapheme pieces, with actual delivery subject to backpressure; 3-second playback capacity and 1-second startup prebuffer.
  Each long run follows a 180-second no-inference pause. Models, builds and
  tracing do not run concurrently with measurements.
- Physical footprint is sampled externally every 5 seconds during long runs;
  thermal state accompanies samples and desktop CPU snapshots occur every
  30 seconds. This is a live desktop, not an isolated performance lab.
- An underrun count measures callback invocations that lack samples, not the
  number of distinct audible breaks. Acceptance requires completed, drained
  playback of at least 600 seconds with zero underruns.

The original addon is
`52ff2a5bebcf43f5369a0a33bf771b3807ebdba8760a97c6dd60deb0a78569ec`.
The optimized addon is
`b83f62a863b1b7ffeab2c08c3e2f592393a131e35a882e236ed7543ec730cbb8`,
from commits `1672ae8` and `e238d1c`. The benchmark records the loaded file's hash,
model configuration, conditioning and execution options. Local model paths and
source revision metadata do not substitute for hashes of every weight file.

## Initial cache screening

Four fresh-process ABBA blocks compare a 0.5 GiB control against each candidate.
Each table cell shows the two observed RTF values, not a confidence interval.
These short tests have no playback and cannot establish an underrun-free floor.

| Candidate GiB | Candidate native RTF | Surrounding 0.5 GiB control RTF |
| ------------: | -------------------: | ------------------------------: |
|           0.3 |      0.5351 / 0.5161 |                 0.5082 / 0.5141 |
|           0.1 |      0.5272 / 0.5290 |                 0.5123 / 0.5156 |
|          0.03 |      0.5832 / 0.6474 |                 0.5145 / 0.6790 |
|          0.01 |      0.9216 / 0.9586 |                 0.7369 / 0.8291 |

All 16 runs completed with EOS and identical raw/processed Float32 hashes. Thermal
state changed from nominal to fair during block 2. The mirrored order reduces
linear drift, but cannot remove nonlinear thermal or background effects. At
1.15 speed, native RTF alone must be below approximately 0.870 even before other
service overhead; the 0.01 GiB samples already exceeded that budget. This is a
screening result, not a proof that a particular cache setting always fails.

## Native allocation changes

`KVCache::with_growth_step()` makes allocation granularity explicit while
preserving the default 256-row policy. Qwen's Code Predictor derives its step
from `num_code_groups`: two initial rows plus the remaining residual steps use
exactly G rows per frame. Reset retains capacity and hides old rows; writable
prefix forks still deep-copy their data. Context visibility, positions, sampling
and precision are unchanged. The tested checkpoints reserve 0.3125 MiB rather
than 5 MiB for these KV buffers. This is a buffer-capacity reduction, not a
promise of the same reduction in total process footprint.

The Qwen loader also selects only decoder weights when the validated voice mode
does not need reference audio. Base still loads both encoders. CustomVoice and
VoiceDesign discard the unused codec encoder's lazy arrays before evaluation and
weight accounting. Component selection remains in the Qwen adapter; the shared
allocator contains no TTS model-name policy.

The 22 KV tests pass, including nondefault growth, reservation, reset, isolated
forks and arithmetic rejection. With each of the three actual checkpoints,
teacher-forced logits match bit for bit for every residual code across three resets.
Fixed-seed complete short synthesis also has identical Float32 PCM before and
after the changes for all three models. An independent source review and its
confirmation pass found no outstanding correctness issues.

## Paired optimization results

Four fresh-process ABBA blocks compare old/new Base builds at the same 0.1 GiB
budget. Each contrast averages log RTF within its block.

|                          Block | Optimized RTF change |
| -----------------------------: | -------------------: |
|                              1 |               −5.65% |
|                              2 |               −8.42% |
|                              3 |               −2.35% |
|                              4 |               −4.41% |
|                 Geometric mean |               −5.23% |
| Exploratory 95% log-t interval |     −9.20% to −1.10% |

The interval uses four block contrasts (three degrees of freedom), not 16
independent runs. This is a small, time-correlated sample; it assumes more independence
than a single fanless desktop can guarantee. It supports retaining the change,
not a universal 5.23% speedup. All 16 Base outputs match the original Float32
hashes. The 1.7B before/after short runs establish output parity, not paired
throughput improvement.

Separate load-only probes at a fixed 0.1 GiB budget measured the following
process high-water marks before synthesis. Filesystem caches were already warm;
these are fresh-process loads, not cold-boot disk measurements.

| Model            | Original peak footprint | Optimized peak footprint |
| ---------------- | ----------------------: | -----------------------: |
| 1.7B CustomVoice |              2.9824 GiB |               2.7725 GiB |
| 1.7B VoiceDesign |              2.9824 GiB |               2.7725 GiB |

The MLX active-memory peak decreased by 224,945,408 bytes (214.52 MiB) in both
probes. Persistent active weight memory after load is unchanged: the removed
weights were already unused and would otherwise become recyclable buffers.
These measurements must not be described as a 214 MiB steady-state weight saving.

## Long playback results

All runs below completed the full story. The failed original-build run is
retained alongside the optimized-build results.

| Model/build                 |    Cache |   Played | Native RTF | Underrun callbacks | Steady physical median after 120 s | Result |
| --------------------------- | -------: | -------: | ---------: | -----------------: | ---------------------------------: | ------ |
| Base, original              |  0.1 GiB | 737.67 s |     0.6512 |                168 |                         2.2226 GiB | Failed |
| Base, optimized             |  0.1 GiB | 737.67 s |     0.5072 |                  0 |                         2.2069 GiB | Passed |
| Base, optimized             | 0.03 GiB | 737.67 s |     0.5041 |                  0 |                         2.1336 GiB | Passed |
| Base, optimized             | 0.01 GiB | 737.67 s |     0.5053 |                  0 |                         2.1215 GiB | Passed |
| 1.7B CustomVoice, optimized |  0.5 GiB | 982.24 s |     0.6289 |                  0 |                         3.3717 GiB | Passed |
| 1.7B VoiceDesign, optimized |  0.5 GiB | 836.32 s |     0.6365 |                  0 |                         3.3755 GiB | Passed |

The original 0.1 GiB run completed all 200 segments with EOS, matching the earlier
full-long raw and processed Float32 hashes exactly. Observed peak footprint was
3.1183 GiB; PCM delivery p95 was 152.95 ms and the maximum interval was 1,136.43 ms.
Its mean synthesis speed exceeded playback speed, yet output supply was not
uniform enough to pass the zero-underrun condition. Neither this failure nor a
later success by itself isolates allocator effects from the operating conditions.

The optimized 0.1 GiB run also completed all 200 segments with identical raw and
processed Float32 PCM, and passed the drained ten-minute playback gate. Native
first PCM was 174.81 ms, and delivery p95 was 151.29 ms. The approximately 22%
lower native RTF in this sequential long-run comparison is **not** a paired
causal estimate; the four-block comparison above is the relevant short-run
evidence for the allocation change. A single passing run does not establish a
universal safe cache floor.

The optimized 0.03 GiB run passed the same full-story gate with the same sample
counts and Float32 hashes. Observed peak footprint was 2.9744 GiB, native first
PCM 201.96 ms and delivery p95 151.40 ms.

The optimized 0.01 GiB run also completed all 200 segments with EOS, drained the
same 737.67 seconds, and passed with zero underruns. Its raw and processed
Float32 hashes and sample counts match the other Base long runs exactly.
Observed peak footprint was 2.9245 GiB, native first PCM 207.03 ms and delivery
p95 151.56 ms. This is the **lowest completed passing long-playback budget in
this sweep**, not an absolute minimum or a universal guarantee. The earlier
0.01 GiB screening used the original build and cannot be substituted for this
optimized-build measurement.

CustomVoice completed all 200 segments with EOS and drained 982.24 seconds of
audio with zero underrun callbacks. This establishes one full-story pass at
0.5 GiB with the stated 8-bit checkpoint and normal tempo; it does not transfer
the Base model’s lower tested cache floor to the 1.7B model.

VoiceDesign also completed all 200 segments with EOS and drained 836.32 seconds
of audio with zero underrun callbacks at 0.5 GiB. These two 1.7B passes use
normal tempo and the recorded 8-bit checkpoints; they do not establish results
for other precision recipes or tempo settings.

## Applying the result

`TTS_MLX_CACHE_LIMIT=0.01` is a tested option for this 0.6B Base workload. The
0.03 and 0.1 GiB settings also passed. In these sequential long runs, lowering
0.03 to 0.01 GiB saved only about 12.4 MiB in median physical footprint, with
similar observed native RTF. This is not a paired demonstration of equivalent
performance under every workload. Model weights and live computation dominate
the remaining roughly 2.12 GiB footprint; an allocator budget is not a process
memory cap.

For the two 1.7B models, the completed long-playback evidence applies to
`TTS_MLX_CACHE_LIMIT=0.5`, not to the Base model's lower settings. The default
allocator policy remains unchanged. Keep the recorded playback capacity,
prebuffer and tempo when reproducing these results, and inspect underrun logs
after changing the workload or competing desktop activity.

The measurement data contains per-run build identities, source revision
metadata, parameters, timing, PCM hashes and memory summaries. Long-run report
and memory-log hashes identify the retained local source files; individual
input-delivery records are omitted from the aggregate, with their count and
delivery completion timing preserved. Independent review checked the source
changes and the reported data, followed by confirmation after the final run.
