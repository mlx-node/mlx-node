# @mlx-node/asr

Local Qwen3-ASR transcription and realtime meeting capture on Apple Silicon.

## Convert the Hugging Face checkpoint

For the fastest tested decoder path, pack the Qwen text model as MXFP4. The
audio encoder and multimodal projector deliberately remain BF16 so speech
features are not quantized:

```bash
yarn mlx convert \
  -i .cache/models/qwen3-asr-1.7b-hf \
  -o .cache/models/qwen3-asr-1.7b-mlx-mxfp4 \
  -d bfloat16 \
  -q --q-mode mxfp4
```

Use a dense conversion when weight fidelity or quantization comparisons matter
more than decode throughput:

```bash
yarn mlx convert \
  -i .cache/models/qwen3-asr-1.7b-hf \
  -o .cache/models/qwen3-asr-1.7b-mlx \
  -d bfloat16
```

The converter detects `model_type: "qwen3_asr"`, canonicalizes the checkpoint
keys, and converts the three audio convolutions to MLX layout. Packed
conversions support uniform affine, MXFP4, and MXFP8 text weights; recipe-based
or per-layer quantization is rejected.

## Offline transcription

```typescript
import { Qwen3AsrModel } from '@mlx-node/asr';

const model = await Qwen3AsrModel.load('.cache/models/qwen3-asr-1.7b-mlx-mxfp4');
const pcm = new Float32Array(/* mono PCM samples */);
const result = await model.transcribe(pcm, {
  sampleRate: 16_000,
  language: 'en', // omit for language detection
});

console.log(result.text, result.realTimeFactor);
```

`transcribe()` accepts mono floating-point PCM at any positive sample rate and
resamples it to the model's native 16 kHz input.

## Streaming manually supplied audio

```typescript
const stream = await model.createStream({
  sampleRate: 48_000,
  chunkSeconds: 2,
  provisionalTokens: 5,
  unfixedChunks: 2,
  maxTokens: 32,
});

for await (const pcmChunk of yourAudioSource) {
  const revision = await stream.feed(pcmChunk);
  if (revision) {
    process.stdout.write(`\r${revision.stableText}\x1b[2m${revision.provisionalText}\x1b[0m`);
  }
}

const final = await stream.finish();
console.log(`\n${final.text}`);
```

Streaming follows Qwen's official rolling policy: every 2 seconds it feeds the
previous raw transcript minus the last 5 tokens back to the model. The first 2
chunks are decoded without transcript conditioning. `stableText` is the
current fixed frontier; render each result as a complete revision because a
later hypothesis can still move that frontier. The next revision may replace
`provisionalText`.

The audio tower itself is not causal. To keep long meetings realtime, completed
8-second local-attention windows are encoded once and cached. The stream keeps
the latest four completed windows plus the current partial window (less than 40
seconds total), bounds the decoder's transcript prefix to 150 tokens, and
reuses KV state through the longest unchanged audio prefix. These bounds keep
memory and per-revision work approximately constant. If `reachedMaxTokens` is
true, the continuation exhausted its 32-token budget and the current
provisional suffix is worth flagging for review. Repeated-token and stalled
decode guards automatically discard a degenerate provisional tail and
re-anchor the next chunk with fresh bounded audio context.

## Realtime meeting capture

`startMeetingTranscription()` captures the local microphone and the Mac's
system/output audio by default. Revisions are kept on separate source-tagged
tracks because the two devices have independent clocks and represent different
speakers.

```typescript
import { Qwen3AsrModel, qwen3AsrAudioDevices, startMeetingTranscription } from '@mlx-node/asr';

console.table(qwen3AsrAudioDevices());

const model = await Qwen3AsrModel.load('.cache/models/qwen3-asr-1.7b-mlx-mxfp4');
const meeting = await startMeetingTranscription(model, {
  stream: { chunkSeconds: 2, provisionalTokens: 5, unfixedChunks: 2, maxTokens: 32 },
  microphone: { feedMilliseconds: 100, ringSeconds: 10 },
  systemAudio: {
    feedMilliseconds: 100,
    ringSeconds: 10,
    // Optional: capture selected apps instead of all system output.
    // applicationBundleIds: ['us.zoom.xos', 'com.microsoft.teams2'],
  },
  onResult({ source, result }) {
    console.log(`[${source}] ${result.stableText}${result.provisionalText}`);
  },
  onError({ source, error }) {
    console.error(`[${source}]`, error);
  },
});

process.once('SIGINT', async () => {
  const final = await meeting.stop();
  console.log('local:', final.microphone?.result.text);
  console.log('remote:', final.systemAudio?.result.text);
});
```

Set either `microphone: false` or `systemAudio: false` for a single-track
session. The lower-level `startRealtimeTranscription()` API remains available
when you want to own exactly one source; its `capture.source` defaults to
`Qwen3AsrCaptureSource.Microphone`.

The native Core Audio callbacks only write packed mono float samples into
bounded single-producer/single-consumer rings. Resampling and MLX inference run
outside the realtime callbacks. Each capture automatically binds its ASR stream
to the selected device's actual sample rate. `feedMilliseconds` controls how
often capture drains into the model buffer; `chunkSeconds` controls the
transcription update cadence.

Packaged macOS hosts must include both `NSMicrophoneUsageDescription` and
`NSAudioCaptureUsageDescription` in `Info.plist`. macOS prompts separately for
microphone and system-audio permission. System capture uses a private Core
Audio tap and does not mute normal speaker playback.
