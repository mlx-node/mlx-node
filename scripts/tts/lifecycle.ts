/** Opt-in native lifecycle and chunk-invariance checks. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { setTimeout as sleep } from 'node:timers/promises';
import { parseArgs } from 'node:util';

import { getMemorySnapshot, TtsNativeModel } from '@mlx-node/core';
import { loadTtsModel, readWav } from '@mlx-node/tts';
import type { TtsOptions } from '@mlx-node/tts';

const { values } = parseArgs({
  options: {
    model: { type: 'string' },
    voice: { type: 'string', default: 'vivian' },
    reference: { type: 'string' },
    'transcript-file': { type: 'string' },
    'voice-description': { type: 'string' },
    instruct: { type: 'string' },
    'instruction-cache': { type: 'boolean', default: false },
  },
});
if (!values.model) throw new Error('--model is required');
if ((values.reference === undefined) !== (values['transcript-file'] === undefined))
  throw new Error('--reference and --transcript-file must be supplied together');
if (values.reference && values['voice-description']) throw new Error('Choose reference audio or a voice description');
const reference = values.reference
  ? { audio: await readWav(values.reference), transcript: await readFile(values['transcript-file']!, 'utf8') }
  : undefined;
let referenceMono: Float32Array | undefined;
if (reference) {
  const channels = reference.audio.channels ?? 1;
  referenceMono = new Float32Array(reference.audio.samples.length / channels);
  for (let i = 0; i < reference.audio.samples.length; i++)
    referenceMono[Math.floor(i / channels)] += reference.audio.samples[i] / channels;
}
let nativeReferenceRelease: { activeBytesBefore: number; activeBytesAfter: number } | undefined;
let sdkReferenceRelease: { activeBytesBefore: number; activeBytesAfter: number } | undefined;
let releasedDuringGeneration = false;
let disposedPreparationMs: number | undefined;
const native = await TtsNativeModel.load(
  values.model,
  JSON.stringify({ instruction_cache: { enabled: values['instruction-cache'] } }),
);
try {
  const nativeVoiceId = reference
    ? await native.prepareVoice(referenceMono!, reference.audio.sampleRate, reference.transcript)
    : undefined;
  const condition = {
    voice: reference || values['voice-description'] ? undefined : values.voice,
    voice_description: values['voice-description'],
    prepared_voice_id: nativeVoiceId,
    instruct: values.instruct,
  };
  const stream = native.start(
    '请测试一个缓慢消费的语音流，并在缓冲区填满之后取消。',
    JSON.stringify({ ...condition, seed: 534, buffer_chunks: 1, chunk_frames: 1 }),
  );
  assert.ok((await stream.next())?.samples.length);
  await sleep(400);
  // Voice release no longer requires an idle model; it serializes behind the
  // running generation, so cancel before awaiting to keep this bounded.
  const releaseWhileBusy = nativeVoiceId ? native.releaseVoice(nativeVoiceId) : undefined;
  releasedDuringGeneration = nativeVoiceId !== undefined;
  void releaseWhileBusy?.catch(() => {}); // Keep the later await the single failure point.
  stream.cancel();
  await stream.waitFinished();
  await releaseWhileBusy;
  assert.equal(await stream.next(), null);
  const next = native.start('可以再次使用。', JSON.stringify({ ...condition, seed: 534, max_frames: 1 }));
  while (await next.next()) {
    /* Drain including the terminal marker. */
  }
  await next.waitFinished();
  if (reference && nativeVoiceId) {
    const activeBytesBefore = getMemorySnapshot().activeBytes;
    // The voice was already released during the generation above.
    await native.releaseVoice(nativeVoiceId);
    nativeReferenceRelease = { activeBytesBefore, activeBytesAfter: getMemorySnapshot().activeBytes };
    const released = native.start('已经释放的声音。', JSON.stringify({ ...condition, max_frames: 1 }));
    try {
      await assert.rejects(released.next(), /Prepared voice does not belong/);
    } finally {
      released.cancel();
      await released.waitFinished();
    }
    const replacementId = await native.prepareVoice(referenceMono!, reference.audio.sampleRate, reference.transcript);
    const replacement = native.start(
      '新准备的声音可以使用。',
      JSON.stringify({ ...condition, prepared_voice_id: replacementId, seed: 534, max_frames: 1 }),
    );
    let replacementSamples = 0;
    for (let chunk = await replacement.next(); chunk; chunk = await replacement.next())
      replacementSamples += chunk.samples.length;
    await replacement.waitFinished();
    assert.ok(replacementSamples > 0);
    await native.releaseVoice(replacementId);
  }
} finally {
  await native.dispose();
}
const model = await loadTtsModel(values.model, { instructionCache: { enabled: values['instruction-cache'] } });
try {
  let prepared = reference ? await model.prepareVoice(reference) : undefined;
  const options: TtsOptions = {
    voice:
      prepared ??
      (values['voice-description'] ? { type: 'description', description: values['voice-description'] } : values.voice),
    instruct: values.instruct,
  };
  const a = await model.synthesize('你好，欢迎使用实时语音合成。', {
    ...options,
    seed: 534,
    chunkDurationMs: 80,
  });
  const b = await model.synthesize('你好，欢迎使用实时语音合成。', {
    ...options,
    seed: 534,
    chunkDurationMs: 240,
  });
  assert.equal(a.samples.length, b.samples.length);
  let max = 0;
  for (let i = 0; i < a.samples.length; i++) max = Math.max(max, Math.abs(a.samples[i] - b.samples[i]));
  assert.ok(max < 0.005, `chunking waveform max error ${max}`);
  if (reference && prepared) {
    const live = model.synthesizeStream('请保持声音直到取消。', { ...options, maxDurationSeconds: 0.16 });
    assert.ok((await live[Symbol.asyncIterator]().next()).value?.samples.length);
    const activeBytesBefore = getMemorySnapshot().activeBytes;
    // Release serializes behind the running generation; cancel before awaiting.
    const released = prepared.dispose();
    void released.catch(() => {}); // Keep the later await the single failure point.
    live.cancel();
    await assert.rejects(live.completed, { name: 'AbortError' });
    await released;
    await prepared.dispose();
    sdkReferenceRelease = { activeBytesBefore, activeBytesAfter: getMemorySnapshot().activeBytes };
    assert.throws(() => model.synthesizeStream('已经释放的声音。', options), /Prepared voice does not belong/);
    prepared = await model.prepareVoice(reference);
    options.voice = prepared;
    assert.ok(
      (await model.synthesize('新准备的声音可以使用。', { ...options, seed: 534, maxDurationSeconds: 0.08 })).samples
        .length > 0,
    );
  }
  const input: AsyncIterable<string> = {
    [Symbol.asyncIterator]() {
      let first = true;
      return {
        next() {
          if (first) {
            first = false;
            return Promise.resolve({ done: false as const, value: '你好。' });
          }
          return new Promise<IteratorResult<string>>(() => {});
        },
      };
    },
  };
  const stream = model.synthesizeStream(input, options);
  await stream[Symbol.asyncIterator]().next();
  await model.dispose();
  await assert.rejects(stream.completed, { name: 'AbortError' });
  await prepared?.dispose();
  if (reference) {
    // The paused-input check above disposes its model, so use a fresh owner for
    // cancellation while reference preparation is still pending.
    const preparingModel = await loadTtsModel(values.model, {
      instructionCache: { enabled: values['instruction-cache'] },
    });
    const started = performance.now();
    const preparing = preparingModel.prepareVoice(reference);
    const cancelled = assert.rejects(preparing, /cancelled/);
    const disposed = preparingModel.dispose();
    let timeout!: ReturnType<typeof setTimeout>;
    try {
      await Promise.race([
        Promise.all([cancelled, disposed]),
        new Promise<never>((_, reject) => {
          timeout = setTimeout(
            () => reject(new Error('Disposal during voice preparation did not finish in 60s')),
            60_000,
          );
        }),
      ]);
    } finally {
      clearTimeout(timeout);
    }
    disposedPreparationMs = performance.now() - started;
  }
  console.log(
    JSON.stringify({
      cancelledFullNativeQueue: true,
      reusedModel: true,
      chunkingMaxError: max,
      disposedPausedInput: true,
      referenceLifecycle: reference
        ? {
            native: nativeReferenceRelease,
            sdk: sdkReferenceRelease,
            releasedDuringGeneration,
            rejectedReleasedVoices: true,
            preparedReplacementVoices: true,
            disposedPreparationMs,
          }
        : undefined,
    }),
  );
} finally {
  await model.dispose();
}
