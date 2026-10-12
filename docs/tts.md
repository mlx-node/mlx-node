# Native streaming text to speech

`@mlx-node/tts` runs Qwen3-TTS in Node → Rust → MLX. It supports CustomVoice
presets, Base reference cloning, and 1.7B VoiceDesign descriptions. Instructions
are supported for the released 1.7B CustomVoice and VoiceDesign profiles.
HTTP serving and batching are outside this version.

```ts
import { loadTtsModel, createAudioPlayer, WavWriter } from '@mlx-node/tts';

const model = await loadTtsModel('./Qwen3-TTS-12Hz-0.6B-CustomVoice');
const player = await createAudioPlayer(model.capabilities.sampleRate);
const wav = await WavWriter.open('speech.wav', model.capabilities.sampleRate);
try {
  const stream = model.synthesizeStream('你好，欢迎使用语音合成。', {
    voice: 'vivian',
    language: 'chinese',
    seed: 42,
  });
  for await (const chunk of stream) {
    await wav.write(chunk);
    await player.write(chunk.samples);
  }
  console.log(await stream.completed, await player.finish());
} finally {
  player.cancel();
  await wav.close();
  await model.dispose();
}
```

For Base, call `prepareVoice({ audio: await readWav('reference.wav'), transcript })`
and pass the returned handle as `voice`. Preparation downmixes, resamples, extracts
the speaker embedding, encodes the reference and prepares its codec decoder prefix.
Preparation advances one codec frame at a time and materializes retained state,
bounding decoder temporary memory independently of reference duration. Each segment
forks the prepared prefix with isolated writable attention caches and immutable
convolution histories. Handles belong to one model and may be reused across
requests. Call `await voice.dispose()` to release a
reference that is no longer needed; repeated disposal is safe. Releases stay
valid while the model is busy and run after the active synthesis or
preparation finishes. A released handle
cannot be used for synthesis. A model retains at most eight prepared voices;
preparing beyond that evicts the oldest, and an evicted or released handle
fails on its next use. Model disposal releases all remaining references.
Reference length is validated against the encoder's configured context capacity
before resampling or inference; it is not silently truncated.

`synthesize` collects the same stream into one `Float32Array`. Use streaming for
long output. PCM is interleaved, `startSample` counts audio frames across all text
segments, and `segmentIndex` identifies the committed text segment.

## Incremental text and lifecycle

Pass an `AsyncIterable<TtsInputEvent>` as input. Strings accumulate
until a complete sentence or a configured phrase boundary is available. `flush`
commits the residual text; end of input flushes automatically. The default maximum
is 256 Unicode graphemes. Input chunks do not have to align to words or tokens.
A trailing period is held until disambiguated or flushed; Chinese terminal
punctuation can commit immediately. Each segment has fresh generator state and its
own codec state (initialized from the prepared reference prefix for Base), and
reuses the voice condition. Natural pauses between segments are expected.

The default output target is 160 ms, rounded up to whole codec frames. Native
PCM buffering is bounded to about one second and text buffering to two segments.
`chunkDurationMs`, `audioBufferSeconds`, `queuedSegments` and
`maxSegmentGraphemes` configure these policies. Slow consumers apply backpressure.

One model accepts one active synthesis or voice preparation. A concurrent request
fails with a busy error; a concurrent voice release is queued instead and
applies after the active operation finishes. `cancel()`, `AbortSignal`, early iterator return and
`dispose()` wake blocked endpoints and release generation state. Cancellation
rejects `completed` with `AbortError`. A stream has one consumer; the consumer must
iterate or cancel it. If every stream reference is dropped first, garbage
collection frees the model slot even while a caller `AbortSignal` stays alive.
Input errors reject both the iterator and `completed`.
An upstream iterator blocked on external I/O is not awaited during cancellation.
Model disposal also cancels voice preparation at computation-stage boundaries;
an already running MLX evaluation completes before its resources are released.

`maxDurationSeconds` is a per-segment limit; reaching it reports `finishReason:
'length'`. The SDK and CLI default it to 120 seconds (1,500 codec frames); that
ceiling also bounds the talker KV reservation to ~160 MiB per segment instead of
the checkpoint generation limit (~0.9 GiB at 8,192 frames). Direct native callers
that omit `max_frames` still use the checkpoint generation configuration.
Sampling options preserve the checkpoint defaults unless explicitly overridden.

## CLI and resources

```sh
mlx download model -m Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
mlx tts -m ./model --voice vivian --text '你好。' --play -o speech.wav
cat text.txt | mlx tts -m ./model --voice ryan --language english --play
mlx tts -m ./base --reference-audio reference.wav \
  --reference-text 'Exact words in the reference.' --text 'New words.' -o clone.wav
```

`--file` and piped stdin are decoded incrementally as UTF-8. WAV
input supports integer PCM and IEEE float. WAV output is streamed PCM16 and has
the standard RIFF 4 GiB size limit. Playback uses CoreAudio on macOS; its callback
only reads a preallocated bounded ring. CoreAudio performs device-rate conversion.
Playback statistics count render callback starvation. `firstPlaybackMs` uses the
output unit's presentation host timestamp when available (otherwise callback
delivery time); this is a device estimate, not an acoustic microphone measurement.
Normal completion waits for the final block's scheduled presentation to finish.

The downloader selects required child checkpoints from native-free family metadata
and pins the root and `speech_tokenizer` to one immutable revision. Conversion
keeps the codec at source precision and copies its separate config and weights.
The root's converted convolution layout is marked `mlx_node_tts_format: 1`;
unmarked official checkpoints use Hugging Face layout. Uniform supported MLX
quantization applies to eligible Talker and predictor linears; speaker weights and
embeddings remain dense. Quantized quality and speed require separate measurement.

## Validation and benchmark

See [the pure MLX memory investigation](tts-memory.md) for production benchmark
commands, physical footprint versus RSS, cache-budget experiments and retained
playback failures.

`TTS_MLX_CACHE_LIMIT=2` sets an optional MLX free-buffer cache ceiling, in GiB,
for the loaded TTS model's lifetime. Fractional values are supported; unset or
`0` uses the normal cache policy. This is a shared process pool, not a TTS-private
memory quota or a total RAM limit. Multiple live ceilings and a positive global
`MLX_CACHE_LIMIT_GB` compose by minimum. Invalid values fail before model loading.

During each segment's decode loop the model additionally tightens the pool to
the decode-time ceiling (`max(2 × step transient, 128 MiB)`), lifted when the
segment ends. That ceiling is process-wide like any other, so it composes by
minimum against concurrent models' ceilings while a TTS turn is in flight.

Native decoding advances explicit causal convolution, transpose-convolution
overlap and transformer cache state. Token ID zero is valid; output length is
based on actual frame count.

Run SDK/lifecycle tests with `vp test packages/tts/__test__`. Rust numerical
parity tests are opt-in and require separately supplied checkpoint weights and
development fixtures. Set `TTS_TEST_MODEL` to the checkpoint directory and
`TTS_TEST_GOLDENS` to the fixture directory, then run
`TTS_TEST_MODEL=<dir> TTS_TEST_GOLDENS=<dir> cargo test -p mlx-core --lib -- --ignored qwen3_tts`.
The ignored tests are `qwen3_tts_reference_parity`, `qwen3_tts_encoder_reference_parity`,
`qwen3_tts_icl_prompt_parity`, `qwen3_tts_teacher_parity`,
`qwen3_tts_predictor_cache_capacity_parity` and the instruction parity tests in
`model.rs`. Fixture generators and generated
numerical fixtures are not included in this repository.
The speaker frontend uses the periodic Hann window defined by
the [official speaker frontend](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py).
The pure-sinusoid Mel fixture permits small float32 FFT roundoff near the log floor;
speaker embedding and teacher logits have separate, tighter downstream checks.
The offline reference encoder uses full causal attention, matching
[Transformers 4.57.3 Mimi SDPA/eager](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/mimi/modeling_mimi.py).
Its explicit mask differs from the windowed FlashAttention path; a long-reference
token-ID fixture guards this distinction. The streaming codec decoder uses its
configured attention window. The encoder's `gelu` activation uses the exact erf
definition. `qwen3_tts_encoder_reference_parity` accepts standalone CPU encoder
fixtures (`wave`, `encoded`, `long_wave`, `long_encoded`); expected IDs have shape
`[1, frames, codebooks]`. This check is independent of decoder and tokenizer fixtures.

`scripts/tts/lifecycle.ts --model <custom-voice>` tests a full native queue,
cancellation, reuse, disposal during paused input and seeded waveform invariance
under different PCM chunk sizes. For Base, pass `--reference <wav>` and
`--transcript-file <txt>` together; this also checks voice release during an
active generation, idempotent and use-after-release rejection, replacement
preparation and cancellation during preparation.
`scripts/tts/quality.ts --custom <path> --base
<path> --output <directory>` writes multilingual preset and cloned samples.
ASR content agreement does not establish timbre similarity or naturalness;
listen to the WAVs. Speaker embedding similarity is a diagnostic, not a
universal quality threshold.

`scripts/tts/benchmark.ts` records load and preparation times, first materialized
PCM, block intervals, RTF, resident-memory samples and optional playback underruns.
`firstPlaybackFromSubmissionMs` brackets the native player clock's origin within
the JS open call, giving lower/upper bounds from text submission. `peakRssBytes`
uses the process-wide OS high-water mark through Node's `resourceUsage().maxRSS`;
`memory` contains steady-state samples and MLX active/peak/allocator-cache bytes.
Use `--seconds 600 --play` for the ten-minute test. The benchmark's
`--prebuffer-seconds` default is one second; the SDK and CLI playback default is
0.32 seconds. RTF is synthesis time / audio duration; native queue backpressure is excluded
from synthesis time. Report cold and warm runs separately. A passing short smoke
test does not establish the ten-minute realtime or perceptual-quality acceptance.
`passedTenMinutePlayback` covers the continuous playback subtest, including EOS;
review the memory trajectory and quality checks separately for full acceptance.
Use `--note` to record competing workloads or other measurement conditions.
`passedAudioContent` requires finite, non-silent processed PCM (RMS above
`1e-4`, roughly -80 dBFS — a sanity floor far below measured quiet speech, not
a speech-level threshold); it rejects a silent or corrupt stream but does not
verify speech content, which needs the external ASR oracle. `passedPcmHash`
enforces any digests supplied as `--expect-raw-pcm-sha256` or
`--expect-processed-pcm-sha256` and is `null` when none are given. Only
`passedAudioContent` is folded into the paced and ten-minute playback verdicts;
`passedPcmHash` is a standalone report field, so a hash mismatch does not fail
those verdicts.

For a finite, paced text response, add `--once --text-file story.txt
--graphemes-per-second 20 --chunk-graphemes 2 --first-chunk-ms 500`. The clock
simulates text availability independently of consumer pulls. When TTS backpressure
delays a pull, already available text can be delivered immediately. This is a
buffered LLM-response simulation; it does not run an LLM or measure tokenizer
tokens per second. Graphemes include punctuation and whitespace.

The report separates theoretical source completion, actual last text delivery,
and the subsequent EOF pull. Delivery timestamps are relative to the first input
pull; `clockOffsetMs` aligns them with text submission. PCM segment offsets allow
sample-exact extraction for content checks. `passedPacedPlayback` requires EOS,
complete input consumption, RTF below one, zero playback underruns, and the upper
bound of first playback to precede simulated source completion. Merely keeping an
input iterator open through backpressure does not establish that condition.

`examples/tts/new-eridu-night.txt` is an original, unofficial story in the world
of Zenless Zone Zero, written for the Chinese Fairy cloning experiment. It has
2,000 Han characters (2,205 non-whitespace characters including punctuation).
The benchmark contains no character-specific voice or pacing logic. Reference
audio stays in the ignored local cache; reports record its SHA-256 and transcript.

## Pitch-preserving speech speed

`changeAudioSpeed(source, speed)` is an independent PCM transform. CLI `--speed`
uses the same path for playback and WAV output. Speed defaults to `1` (exact PCM
passthrough); a factor such as `1.15` shortens duration while retaining pitch.
It does not change the voice condition, language, sampling rate or model tokens.

```ts
import { changeAudioSpeed } from '@mlx-node/tts';

const source = model.synthesizeStream(input, { voice, language: 'chinese' });
const output = changeAudioSpeed(source, 1.15);
for await (const chunk of output) {
  await writer.write(chunk);
  await player.write(chunk.samples);
}
const inference = await source.completed;
console.log(output.stats); // DSP call time, input frames and output frames
```

The independent implementation lives in the MIT-licensed `mlx-tts` Rust crate.
It has no MLX, N-API, device or model dependencies and forbids unsafe code. A thin
`PcmTempo` binding connects it to the SDK. It uses waveform-similarity overlap/add
(WSOLA), with a global time map, multichannel energy-aware normalized correlation,
coarse block-average search and full-resolution refinement. Complementary raised-
cosine windows join aligned grains. No third-party tempo implementation is linked
or vendored; the previous PICOLA experiment is only a local comparison baseline.

Window and search durations are configurable Rust policies, independent of
checkpoints. Channels contribute separately to correlation, preventing opposite-
phase stereo cancellation, and share edit positions. Source history and pending
output stay bounded. Final length is exactly `round(inputFrames / speed)` per
segment, independent of chunking. The default 40 ms grain and 8 ms search radius
trade lookahead for waveform continuity. The finite endpoint first finds an
aligned source window, overlaps at a fixed offset, then smoothly removes that
offset after the overlap. Its retained length is derived from search radius and
the configurable maximum local rate deviation (default 0.25); this avoids an
arbitrary-phase tail crossfade. Short clips reduce the search range to satisfy
the same rate bound. Linear interpolation is confined to that endpoint correction.
At 24 kHz and 1.15 speed the default first-output lookahead is 117 ms of input;
`lookahead_frames()` reports the actual threshold. This is an additional buffering
requirement, not a measured first-PCM latency. Quality and CPU cost must be measured
rather than assumed superior to other algorithms.

The adapter closes upstream on errors and early iterator exit, including before
the first pull. It flushes when it sees the next segment index or input EOF; it
does not guess boundaries using timers. Consequently a final lookahead tail can
wait for the next segment's first PCM when incremental text pauses. Speech-speed
processing is intended for speech, not a general music time-stretching engine.

Benchmark `timing.measuredServiceTimeRatio` divides native synthesis time plus
measured DSP call time by the transformed audio duration. Those work intervals
can overlap and omit untimed overhead, so this is a service-time ratio, not an
end-to-end wall-time ratio. `timing.observedStreamWallMs` records the stream's
actual host-clock interval, including sinks, input waits and backpressure. Raw
model statistics remain under `synthesis`. Playback underruns, actual first
presentation and input availability are measured separately; playback gates
require successful input completion, no cancellation and actual drained playback
matching the generated duration.
`rawSegmentBoundaryMs` measures source PCM delivery intervals before the tempo
transform, so flushing a previous segment's tail cannot disguise a delivery gap.
These wall intervals still include consumer pull delays, DSP and sink backpressure;
they do not isolate native synthesis time. Playback capacity is
configurable (`--playback-buffer-seconds` in the benchmark, `--buffer-seconds` in
the CLI) independently of startup prebuffer. The benchmark records both; its
playback acceptance gates require startup prebuffer of at most one second.

The tempo crate is independent code under this repository's MIT license. It does
not use Sonic sources, its AMDF/PICOLA algorithm, a C vendor copy, or an Apache
notice from that experiment. WSOLA is an established algorithm family; independent
implementation does not mean a newly invented algorithm or universal superiority.
The final independent review closed EOF padding, lost endpoint transient,
phase-cancellation, lookahead and narrow-search history findings. Twelve Rust tests
and 219 additional randomized, extreme and policy combinations checked exact
duration, chunk invariance, multichannel alignment and bounded state. The SDK
also checks early cancellation, segment flushes and upstream resource release.

## Measured results (2026-10-08)

Apple M4, 32 GiB unified memory, macOS kernel 27.0.0, Node 24.21.0.
Both runs used affine 8-bit/group-64 eligible linears, BF16 dense root weights,
source FP32 codec weights, seed 534, 160 ms target chunks and one second of
playback prebuffer. This precision is explicitly selected during conversion;
loading an official checkpoint preserves its original precision.

| Model                           | Played audio |   RTF | First CPU PCM | Underruns | EOS segments |
| ------------------------------- | -----------: | ----: | ------------: | --------: | -----------: |
| CustomVoice, Vivian             |     623.04 s | 0.537 |        113 ms |         0 |          132 |
| Base, prepared Vivian reference |     619.36 s | 0.590 |        553 ms |         0 |          147 |

Both started producing PCM while the incremental input was still open. Fresh-process model
loads took 692/773 ms (CustomVoice/Base). Base prepared its 6.40-second reference
once in 1803 ms. Estimated first presentation, measured from text submission, was
565–640 ms / 1329–1482 ms; the intervals bracket the native player clock's origin,
not a confidence interval or an acoustic measurement. The one-second prebuffer is
audio duration, not a promise of one-second startup wall time.

| Metric                                          |        CustomVoice |               Base |
| ----------------------------------------------- | -----------------: | -----------------: |
| PCM arrival interval p50 / p95 / p99            | 160 / 164 / 287 ms | 160 / 166 / 338 ms |
| Segment-boundary gap p50 / p95                  |       281 / 300 ms |       328 / 357 ms |
| OS peak RSS (decimal GB)                        |              2.204 |              2.228 |
| MLX peak allocated memory (decimal GB)          |              2.202 |              2.667 |
| Final MLX active / allocator cache (decimal GB) |      1.839 / 0.686 |      2.057 / 1.579 |

Over the second half of each run, active/cache memory returned to the same level
(differences below 32 bytes); sampled RSS grew by only 0.54/0.77 MB. Diagnostic
interval arrays are collected by the benchmark, outside the bounded TTS runtime.
The Base run overlapped CPU release-test compilation during its first part; neither
run had a competing GPU workload. Compilation had finished before CustomVoice.
These are single sustained runs, not repeated-run latency confidence intervals.

The failed Base baseline is retained: 619.36 seconds, RTF 1.223, 13,660 starving
callbacks, 5.17 GB MLX peak, and a 1606 ms median segment gap. It decoded the whole
reference again for every segment. The final implementation prepares that state
once and forks isolated caches. Both runs had background CPU compilation, with
different overlap durations, so the before/after numbers are not an isolated
microbenchmark of the optimization. Initial BF16 fresh-process smoke runs exceeded RTF 1;
the ten-minute realtime result above applies to the stated 8-bit configuration.

Uncommitted local reports (under the gitignored `.cache/`):
`.cache/tts/results/base-8bit-10min-final.json` (failed baseline),
`base-prefix-10min.json` and `custom-final-10min.json`. Model revisions:

- CustomVoice: `85e237c12c027371202489a0ec509ded67b5e4b5`.
- Base: `5d83992436eae1d760afd27aff78a71d676296fc`.
- Base reference: `quality-8bit/zh-vivian.wav`, generated by `quality.ts`, transcript
  `你好，欢迎使用语音合成。我们正在验证实时播放和声音质量。`.

To reproduce the selected precision and sustained test:

```sh
mlx convert -i ./Base -o ./Base-8bit -m qwen3_tts -q \
  --q-bits 8 --q-group-size 64 --dtype bfloat16
vp exec oxnode scripts/tts/benchmark.ts --model ./Base-8bit \
  --reference reference.wav --transcript 'Exact reference words.' \
  --seconds 600 --play --report base-report.json
```

Verification on the final native build:

- 78 TTS-scoped TS tests — `packages/tts/__test__` (51 across stream, audio,
  speed and qwen3 adapter suites), `__test__/tts` (14), `download-tts` (9) and
  `tts-input` (4); 31 selected Rust tests including the three real checkpoint
  oracle tests run above (fixture-gated `#[ignore]` tests — seven in total
  covering codec/encoder/ICL/teacher/predictor parity — ran with locally
  supplied `TTS_TEST_MODEL`/`TTS_TEST_GOLDENS` fixtures and are not runnable
  from this checkout); native lifecycle checks; native build; focused type
  checks and Clippy passed. Full workspace `build:ts` remains blocked by existing
  `packages/agent` pi-ai dependency/type incompatibilities in unchanged files.
  A real CLI smoke test with split UTF-8 byte sequences and delayed stdin input
  produced a valid 24 kHz WAV: 6.40 seconds, three EOS segments, RTF 0.487.
- Short and 22-second reference encodings match the oracle token IDs exactly.
  The long test exposed and now guards the encoder full-causal distinction above.
  Codec chunking across its attention window has maximum waveform error
  `2.30e-5`; repeated prepared-prefix forks have maximum error `2.17e-5` against
  whole-reference decoding, including an intervening unrelated continuation.
- Speaker embedding max error is `2.60e-5`, teacher-forced Talker logits `2.12e-4`,
  predictor logits at most `4.39e-5` in the float32 oracle checks. Real Base ICL
  prompt differences fit BF16 rounding (`0.015625` maximum).
- All 12 initial BF16/8-bit multilingual preset/clone samples and all six final
  8-bit samples pass ASR content review, allowing punctuation and number formatting.
  The final four preset PCM16 files are unchanged; both clone files differ by at
  most one PCM16 unit after prefix reuse. Clones rank their own reference highest
  among four presets in the ECAPA diagnostic (cosines 0.9906 / 0.9946).
- Independent review findings were fixed and a confirm review found no remaining
  static blockers. Naturalness, timbre and audible joins still require human
  listening; no perceptual-quality pass is claimed from ASR or embedding scores.

> PCM/hash staleness: the PCM16 comparisons above and the
> `rawPcmSha256`/`processedPcmSha256` digests recorded in the `docs/research`
> benchmark reports predate the sampler parity fixes (64-token repetition
> window, ICL 1.5 penalty floor, temperature-first filter ordering, EOS
> filtering, shared predictor sampling). The 64-token window changes generated
> PCM beyond 64 frames and the ICL floor changes every clone run, so those
> digests are historical records, not baselines for the current build.
> Regenerate them with `--expect-raw-pcm-sha256` /
> `--expect-processed-pcm-sha256` before reusing them as gates.

### Chinese Fairy voice and a paced 2000-character response

The original, unofficial narrative is `examples/tts/new-eridu-night.txt`: exactly
2000 Han characters, 2242 Unicode graphemes including punctuation/whitespace,
19 paragraphs and 100 committed text segments. This is a simulated LLM output
clock at 20 graphemes/second, two graphemes per delivery and a 500 ms first-chunk
delay. It is not a tokenizer-token throughput measurement.

Reference: community dataset
[`MigoXV/qwen3-tts-0.6b-voice-zenless-100-2026-07-23`](https://huggingface.co/datasets/MigoXV/qwen3-tts-0.6b-voice-zenless-100-2026-07-23),
revision `0fe4b9ec178eb13e7a075da3806b41aff2ae3301`, row 10, ID
`4912e6e8-7f93-4655-82cf-c3e1f7dff595`, labeled Chinese Fairy. The 12.707-second
24 kHz mono reference has SHA-256
`44e49825bd9dfb07987cd24a054ef65650562aa7a1d3ef8e5a6b1f8c60f99c83`.
The label is dataset provenance, not independent authentication of the game asset.
Native `prepareVoice` computes its own speaker embedding and reference codec state.

| Run                             |    Output | Effective RTF | Starving callbacks | Playback capacity / startup prebuffer |
| ------------------------------- | --------: | ------------: | -----------------: | ------------------------------------: |
| Original speed                  | 424.160 s |         0.579 |                  0 |                             1 s / 1 s |
| Retired PICOLA experiment, 1.15 | 368.835 s |         0.856 |                328 |                             1 s / 1 s |
| Independent Rust WSOLA, 1.15    | 368.835 s |         0.726 |                  0 |                             3 s / 1 s |

These are separate sustained runs, not an isolated algorithm A/B comparison:
playback capacity changed and inference time varied. The failed experiment is
retained. New DSP work totaled 2.417 s, against 265.413 s native synthesis time;
the transformed output has exactly 8,852,033 frames and all 100 segments reached
EOS. Fresh-process load took 732 ms, reference preparation 2996 ms. First CPU PCM arrived
1200 ms after submission, or 199 ms after the first sentence became available.
The estimated first presentation was 1749–1827 ms. The virtual text producer
finished at 112.5 s, while backpressure deferred the actual last delivery to
350.85 s. Playback therefore started while new text was still being produced.

OS peak RSS was 2.256 GB. Sampled RSS grew 1.15 MB over the second half; last sampled
MLX active memory changed by -176 bytes and allocator cache by +14.56 MB over
that interval. This single six-minute run demonstrates the finite paced-playback
case, not a fresh ten-minute acceptance or proof of indefinite memory stability.

The native speed adapter matches the pure Rust preview output sample-for-sample.
For that 13.76-second preview, processing took 25 ms and yielded 11.965 seconds.
ASR exactly matched the original text; pYIN voiced-F0 medians were 198 Hz before
and 200 Hz after processing. These diagnostics do not establish perceptual
superiority. Uncommitted local artifacts include `long-stream-wsola-1.15.{wav,json}`,
`wsola-implementation.json` (source hashes and policies), and the retained
`long-stream-picola-1.15.{wav,json}`, under `.cache/tts/fairy/`.

All 100 transformed segments were also transcribed with the pinned ASR oracle.
After removing punctuation/whitespace, 80 match the text exactly, versus 78 before
tempo processing. Six transcripts changed; differences mainly concern homophones
and proper-name spellings. Remaining ambiguities include `伊埃斯` transcribed as
`EIS` / `e i四`, and `以骸` as `野孩` in one segment. These require listening and
are not automatically classified as synthesis errors or corrected with a
character-specific pronunciation rule. No universal content/quality pass is inferred
from the ASR count. Detailed comparisons are in
`.cache/tts/fairy/wsola-1.15-evaluation/content-comparison.json`.

```sh
vp exec oxnode scripts/tts/benchmark.ts --model .cache/tts/models/Base-8bit \
  --reference .cache/tts/fairy/reference.wav \
  --transcript-file .cache/tts/fairy/reference.txt \
  --text-file examples/tts/new-eridu-night.txt --once \
  --graphemes-per-second 20 --chunk-graphemes 2 --first-chunk-ms 500 \
  --speed 1.15 --playback-buffer-seconds 3 --prebuffer-seconds 1 --play \
  --output fairy.wav --report fairy.json
```

## Instruction control and VoiceDesign

The native adapter supports instructions for the released **1.7B CustomVoice**
model and descriptions for **1.7B VoiceDesign**. Capability recognition uses the
checkpoint's `tts_model_type` and `tts_model_size`, not its directory name.
0.6B CustomVoice and Base retain their existing synthesis behavior and reject
nonempty instructions. Unknown VoiceDesign profiles fail at load time.

```ts
const model = await loadTtsModel(customVoicePath);
const stream = model.synthesizeStream(text, {
  voice: 'vivian',
  language: 'chinese',
  instruct: '用平静、清晰的语气说。',
});
for await (const chunk of stream) await sink.write(chunk);
```

For VoiceDesign, supply an explicit voice description:

```ts
const model = await loadTtsModel(voiceDesignPath);
const stream = model.synthesizeStream(text, {
  voice: { type: 'description', description: '成年女性，清晰沉稳的中低音，普通话。' },
  instruct: '用轻松、温和的语气说。',
});
```

`model.capabilities.conditioning` lists voice modes (`preset`, `reference`,
`description`) and instruction status (`supported`, `experimental`, `unsupported`).
The current public adapter does not expose experimental reference instructions.
SDK validation and native validation both enforce the capability contract.

Incremental input accepts strings and structured `text`, `flush`, and `instruct`
events. An instruction event first commits pending text under the previous
instruction, then applies the new instruction to subsequent text. Queued segments
retain their own immutable instruction snapshots. An identical instruction is a
no-op; `null` or whitespace clears delivery instructions without clearing a
VoiceDesign description. Control-only events do not produce empty audio segments.
See `examples/tts/instruct.ts` for a complete playback example.

VoiceDesign has one learned instruction channel. The adapter combines the voice
description and delivery instruction, separated by a newline, in one user
message. These application concepts do not guarantee independent acoustic
control: a delivery change can also change perceived voice identity. Descriptions
and instructions are never passed to the speech text segmenter. DSP speed control
remains an independent PCM transform; instructions are not parsed into speed
parameters.

### CLI input

```sh
mlx tts -m ./CustomVoice-1.7B --voice vivian --instruct '用平静的语气说。' \
  --text '现在是下午三点。' -o speech.wav
mlx tts -m ./VoiceDesign-1.7B --voice-description '成年男性，温暖清晰的中低音。' \
  --instruct-file delivery.txt --file speech.txt --play
mlx tts -m ./CustomVoice-1.7B --voice vivian --input-format jsonl --file events.jsonl --play
```

```jsonl
{"type":"text","text":"现在开始播报。"}
{"type":"instruct","value":"用兴奋的语气说。"}
{"type":"text","text":"我们终于可以出发了！"}
{"type":"instruct","value":null}
{"type":"flush"}
```

Text remains the default input format. JSONL is incremental UTF-8, allows a final
record without a newline, and has a 64 KiB record limit. In `jsonl` mode `--text`
is parsed as a single JSONL record, so use `{"type":"text","text":"…"}` rather than
plain text. Invalid records terminate the stream with a line-numbered error. File
input is opened lazily and destroyed on iterator completion. `--instruct` and
`--instruct-file` are mutually exclusive;
preset, reference audio, and voice description are mutually exclusive voice modes.

### Prefix caching

`loadTtsModel(path, { instructionCache: { enabled, maxBytes, maxEntries } })`
controls the Qwen adapter's model-local LRU. The budget defaults to 64 MiB and eight
entries; either limit set to zero disables it. Caching currently defaults **off**.
Numerical parity and lower repeated-instruction latency have been measured, but
a fixed-seed content A/B produced additional ASR mismatches when enabled. This is
an experimental optimization: split BF16 execution changes rounding and can
affect speech content. Evaluate the intended prompts before opting in. With caching enabled, only the
instruction prefix is prefilled and reused. Each continuation deep-copies its KV
buffers and absolute positions. Cache entries are materialized and charged for
actual allocated KV capacity, projected embeddings, and token storage; oversized
entries are used without retention. No generated speech or per-segment codec state
is stored in this cache. Total context checks include the prefix and generation
budget on cache hits as well as misses.

### Development validation

`instruct-quality.ts` produces native Chinese/English listening samples;
`--cache` compares fully consumed PCM cold/hot prefix timings. The benchmark
accepts `--instruct`, `--voice-description`, `--instruction-cache`, and
`--instructions-file` (a JSON array of strings/null). In paced mode it changes
instructions at paragraph boundaries without changing text or its virtual LLM
clock; repeated-input mode changes them between repetitions. Existing timing,
backpressure, playback and memory measurements apply unchanged.

Reference-conditioned instruction experiments are not a supported feature merely
because a prefix can be assembled. Content, reference identity, delivery control,
and segment seams require separate evaluation, including listening. Unsuccessful
or unconfirmed routes remain outside the public API.
