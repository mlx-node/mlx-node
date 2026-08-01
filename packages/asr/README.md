# @mlx-node/asr

Local Qwen3-ASR transcription and realtime microphone capture on Apple Silicon.

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
});

for await (const pcmChunk of yourAudioSource) {
  const revision = await stream.feed(pcmChunk);
  if (revision) {
    process.stdout.write(`\r${revision.stableText}${revision.provisionalText}`);
  }
}

const final = await stream.finish();
console.log(`\n${final.text}`);
```

Streaming uses rolling accumulated context. `stableText` excludes the trailing
provisional token window; the next revision may replace `provisionalText`.

## Realtime RustAudio/CPAL capture

```typescript
import { Qwen3AsrModel, qwen3AsrInputDevices, startRealtimeTranscription } from '@mlx-node/asr';

console.table(qwen3AsrInputDevices());

const model = await Qwen3AsrModel.load('.cache/models/qwen3-asr-1.7b-mlx-mxfp4');
const session = await startRealtimeTranscription(model, {
  stream: { chunkSeconds: 2 },
  capture: { feedMilliseconds: 100, ringSeconds: 10 },
  onResult(result) {
    process.stdout.write(`\r${result.stableText}${result.provisionalText}`);
  },
  onError(error) {
    console.error(error);
  },
});

process.once('SIGINT', async () => {
  const { result, capture } = await session.stop();
  console.log(`\n${result.text}`);
  console.log(`dropped microphone frames: ${capture.droppedFrames}`);
});
```

The Core Audio callback only converts and downmixes samples into a bounded
single-producer/single-consumer ring. Resampling and MLX inference run outside
the realtime callback. Starting capture automatically binds the ASR stream to
the device's actual sample rate. macOS may prompt the host process for
microphone permission the first time capture starts.
