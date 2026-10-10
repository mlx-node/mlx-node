import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { writeFile, readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { cpus, totalmem, platform, arch, release } from 'node:os';
import { join } from 'node:path';
/** Compile with tsconfig.benchmark.json, then run the emitted JS with ordinary
 * Node to benchmark the published SDK entry point and release native addon. */
import { parseArgs } from 'node:util';

import { getMemorySnapshot } from '@mlx-node/core';
import { loadTtsModel, readWav, WavWriter, createAudioPlayer, changeAudioSpeed } from '@mlx-node/tts';
import type { TtsOptions, AudioPlayer, AudioChunk, TtsInputEvent } from '@mlx-node/tts';

import { paragraphInstructions } from './instruction-input.js';
import { readModelProvenance } from './model-provenance.js';
import { pacedInput } from './paced-input.js';

async function fileSha256(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const bytes of createReadStream(path)) hash.update(bytes);
  return hash.digest('hex');
}

const { values } = parseArgs({
  options: {
    model: { type: 'string' },
    voice: { type: 'string', default: 'vivian' },
    reference: { type: 'string' },
    instruct: { type: 'string' },
    'voice-description': { type: 'string' },
    'instructions-file': { type: 'string' },
    'instruction-cache': { type: 'boolean' },
    transcript: { type: 'string' },
    seconds: { type: 'string', default: '30' },
    play: { type: 'boolean' },
    output: { type: 'string' },
    report: { type: 'string' },
    text: { type: 'string' },
    'text-file': { type: 'string' },
    'transcript-file': { type: 'string' },
    once: { type: 'boolean' },
    'graphemes-per-second': { type: 'string' },
    'chunk-graphemes': { type: 'string', default: '2' },
    'first-chunk-ms': { type: 'string', default: '500' },
    speed: { type: 'string', default: '1' },
    'playback-buffer-seconds': { type: 'string', default: '1' },
    'prebuffer-seconds': { type: 'string', default: '1' },
    'expect-raw-pcm-sha256': { type: 'string' },
    'expect-processed-pcm-sha256': { type: 'string' },
    note: { type: 'string' },
  },
});
if (!values.model) throw new Error('--model is required');
if (values.text && values['text-file']) throw new Error('Use --text or --text-file, not both');
if (values.transcript && values['transcript-file']) throw new Error('Use --transcript or --transcript-file, not both');
if (values['graphemes-per-second'] && !values.once) throw new Error('Paced input requires --once');
const target = Number(values.seconds);
if (!Number.isFinite(target) || target <= 0) throw new Error('--seconds must be positive');
const playbackPolicy = {
  bufferSeconds: Number(values['playback-buffer-seconds']),
  prebufferSeconds: Number(values['prebuffer-seconds']),
};
if (
  !Number.isFinite(playbackPolicy.bufferSeconds) ||
  playbackPolicy.bufferSeconds <= 0 ||
  playbackPolicy.bufferSeconds > 60 ||
  !Number.isFinite(playbackPolicy.prebufferSeconds) ||
  playbackPolicy.prebufferSeconds < 0 ||
  playbackPolicy.prebufferSeconds > playbackPolicy.bufferSeconds
) {
  throw new Error('Playback buffer must be within (0, 60] seconds and prebuffer within [0, buffer]');
}
for (const [flag, expected] of [
  ['--expect-raw-pcm-sha256', values['expect-raw-pcm-sha256']],
  ['--expect-processed-pcm-sha256', values['expect-processed-pcm-sha256']],
] as const)
  if (expected !== undefined && !/^[0-9a-f]{64}$/i.test(expected))
    throw new Error(`${flag} must be a SHA-256 hex digest`);
const text = values['text-file']
  ? await readFile(values['text-file'], 'utf8')
  : (values.text ??
    '我们正在测试连续的语音合成。声音应该清晰自然，句子之间保持合适的停顿。今天的温度是二十五度，下午三点十五分开始下一项测试。');
if (!text.trim()) throw new Error('Input text must be nonempty');
const controller = new AbortController();
const paced = values['graphemes-per-second']
  ? pacedInput(
      text,
      {
        graphemesPerSecond: Number(values['graphemes-per-second']),
        chunkGraphemes: Number(values['chunk-graphemes']),
        firstChunkMs: Number(values['first-chunk-ms']),
      },
      controller.signal,
    )
  : undefined;
const instructions: (string | null)[] = values['instructions-file']
  ? JSON.parse(await readFile(values['instructions-file'], 'utf8'))
  : [];
if (!Array.isArray(instructions) || instructions.some((v) => v !== null && typeof v !== 'string'))
  throw new Error('instructions-file must contain a JSON array of strings/null');
const start = performance.now();
const model = await loadTtsModel(values.model, { instructionCache: { enabled: values['instruction-cache'] } });
const loadMs = performance.now() - start;
try {
  const execution = {
    execArgv: process.execArgv,
    nodeOptions: process.env.NODE_OPTIONS ?? null,
    sdkEntry: import.meta.resolve('@mlx-node/tts'),
    coreEntry: import.meta.resolve('@mlx-node/core'),
    allocatorCacheLimitGiB: process.env.MLX_CACHE_LIMIT_GB ?? null,
    ttsAllocatorCacheLimitGiB: process.env.TTS_MLX_CACHE_LIMIT ?? null,
    nativeAddons: await Promise.all(
      Object.keys(createRequire(import.meta.url).cache)
        .filter((path) => path.endsWith('.node'))
        .sort()
        .map(async (path) => ({
          path,
          sha256: await fileSha256(path),
        })),
    ),
  };
  const prepareStart = performance.now();
  const transcript = values['transcript-file']
    ? await readFile(values['transcript-file'], 'utf8')
    : (values.transcript ?? '');
  const referenceSha256 = values.reference
    ? createHash('sha256')
        .update(await readFile(values.reference))
        .digest('hex')
    : null;
  const voice = values.reference
    ? await model.prepareVoice({
        audio: await readWav(values.reference),
        transcript,
      })
    : values['voice-description'] !== undefined
      ? { type: 'description' as const, description: values['voice-description'] }
      : values.voice;
  const voicePreparationMs = performance.now() - prepareStart;
  const configText = await readFile(join(values.model, 'config.json'), 'utf8');
  const checkpointConfig = JSON.parse(configText);
  async function optionalJson(name: string): Promise<unknown> {
    try {
      return JSON.parse(await readFile(join(values.model!, name), 'utf8'));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null;
      throw error;
    }
  }
  const checkpoint = {
    configSha256: createHash('sha256').update(configText).digest('hex'),
    revision: await readModelProvenance(values.model),
    generationDefaults: await optionalJson('generation_config.json'),
  };
  let generatedSeconds = 0;
  let progressAt = 0;
  let inputEnded = false;
  async function* input(): AsyncGenerator<TtsInputEvent> {
    if (paced) {
      yield* paragraphInstructions(paced, text, instructions);
    } else {
      let iteration = 0;
      do {
        if (instructions.length) yield { type: 'instruct', value: instructions[iteration++ % instructions.length] };
        yield text;
        yield { type: 'flush' as const };
      } while (!values.once && generatedSeconds < target);
    }
    inputEnded = true;
  }
  let player: AudioPlayer | undefined;
  let playerOpen: { start: number; end: number } | undefined;
  let writer: WavWriter | undefined;
  const options: TtsOptions = {
    voice,
    instruct: values.instruct,
    language: 'chinese',
    seed: 534,
    chunkDurationMs: 160,
    signal: controller.signal,
  };
  const memory: (NodeJS.MemoryUsage & {
    audioSeconds: number;
    activeBytes: number;
    cacheBytes: number;
    peakBytes: number;
  })[] = [];
  const intervals: number[] = [];
  const segmentFirstPcm: number[] = [];
  const rawPcmHash = createHash('sha256');
  const processedPcmHash = createHash('sha256');
  let processedSamples = 0;
  let nonFiniteSamples = 0;
  let processedPcmSquares = 0;
  const segments: {
    segmentIndex: number;
    startSample: number;
    endSample: number;
    firstPcmMs: number;
    lastPcmMs: number;
  }[] = [];
  let lastAt: number | undefined;
  let lastSegment = -1;
  let firstBeforeInputEnded = false;
  const cancel = () => {
    controller.abort();
    player?.cancel();
  };
  process.once('SIGINT', cancel);
  try {
    if (values.output) writer = await WavWriter.open(values.output, model.capabilities.sampleRate, 1);
    if (values.play) {
      const opened = performance.now();
      player = await createAudioPlayer(model.capabilities.sampleRate, 1, playbackPolicy);
      playerOpen = { start: opened, end: performance.now() };
    }
    const began = performance.now();
    const stream = model.synthesizeStream(input(), options);
    // Measure before tempo processing: a flushed previous-segment tail can
    // otherwise hide the source gap immediately before the next segment.
    let rawPrevious: { at: number; segmentIndex: number } | undefined;
    const measured: AsyncIterable<AudioChunk> = {
      [Symbol.asyncIterator]() {
        const iterator = stream[Symbol.asyncIterator]();
        return {
          async next() {
            const result = await iterator.next();
            if (!result.done) {
              const samples = result.value.samples;
              rawPcmHash.update(Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength));
              const at = performance.now();
              if (rawPrevious && rawPrevious.segmentIndex !== result.value.segmentIndex) {
                segmentFirstPcm.push(at - rawPrevious.at);
              }
              rawPrevious = { at, segmentIndex: result.value.segmentIndex };
            }
            return result;
          },
          return: (value) => (iterator.return ? iterator.return(value) : Promise.resolve({ done: true, value })),
          throw: (error) => (iterator.throw ? iterator.throw(error) : Promise.reject(error)),
        };
      },
    };
    const processed = changeAudioSpeed(measured, Number(values.speed));
    let firstPcmMs: number | undefined;
    for await (const chunk of processed) {
      const at = performance.now();
      processedPcmHash.update(Buffer.from(chunk.samples.buffer, chunk.samples.byteOffset, chunk.samples.byteLength));
      processedSamples += chunk.samples.length;
      for (const sample of chunk.samples) {
        if (Number.isFinite(sample)) processedPcmSquares += sample * sample;
        else nonFiniteSamples++;
      }
      if (firstPcmMs === undefined) {
        firstPcmMs = at - began;
        firstBeforeInputEnded = !inputEnded;
      }
      if (lastAt !== undefined) intervals.push(at - lastAt);
      if (chunk.segmentIndex !== lastSegment) {
        memory.push({ audioSeconds: generatedSeconds, ...process.memoryUsage(), ...getMemorySnapshot() });
        if (generatedSeconds >= progressAt) {
          console.error(
            `Generated ${generatedSeconds.toFixed(1)} seconds; RSS ${(process.memoryUsage().rss / 2 ** 20).toFixed(0)} MiB`,
          );
          progressAt = generatedSeconds + 30;
        }
        lastSegment = chunk.segmentIndex;
        segments.push({
          segmentIndex: chunk.segmentIndex,
          startSample: chunk.startSample,
          endSample: chunk.startSample,
          firstPcmMs: at - began,
          lastPcmMs: at - began,
        });
      }
      const segment = segments[segments.length - 1];
      segment.endSample = chunk.startSample + chunk.samples.length / chunk.channels;
      segment.lastPcmMs = at - began;
      lastAt = at;
      generatedSeconds += chunk.samples.length / chunk.channels / chunk.sampleRate;
      await writer?.write(chunk);
      await player?.write(chunk.samples);
    }
    const observedStreamWallMs = performance.now() - began;
    const synthesis = await stream.completed;
    // Native generation and JS DSP can overlap; this is a sum of service
    // intervals, not end-to-end elapsed time or total device compute time.
    const measuredServiceTimeMs = synthesis.synthesisMs + processed.stats.processingMs;
    const measuredServiceTimeRatio = generatedSeconds ? measuredServiceTimeMs / 1000 / generatedSeconds : null;
    const playback = await player?.finish();
    const playbackFinishWallMs = playback ? performance.now() - began : null;
    const playbackComplete =
      !!playback &&
      !controller.signal.aborted &&
      Math.abs(playback.playedSeconds - generatedSeconds) <= 2 / model.capabilities.sampleRate;
    const firstPlaybackFromSubmissionMs =
      playerOpen && playback?.firstPlaybackMs != null
        ? {
            lowerBound: playerOpen.start + playback.firstPlaybackMs - began,
            upperBound: playerOpen.end + playback.firstPlaybackMs - began,
          }
        : null;
    const simulatedCompletionFromSubmissionMs =
      paced?.startedAt === undefined ? null : paced.startedAt - began + paced.simulatedCompletionMs;
    const playbackBeforeSimulatedCompletion =
      firstPlaybackFromSubmissionMs && simulatedCompletionFromSubmissionMs !== null
        ? firstPlaybackFromSubmissionMs.upperBound < simulatedCompletionFromSubmissionMs
          ? true
          : firstPlaybackFromSubmissionMs.lowerBound >= simulatedCompletionFromSubmissionMs
            ? false
            : null
        : null;
    const percentile = (data: number[], p: number) =>
      [...data].sort((a, b) => a - b)[Math.min(data.length - 1, Math.floor(data.length * p))] ?? null;
    const rawPcmSha256 = rawPcmHash.digest('hex');
    const processedPcmSha256 = processedPcmHash.digest('hex');
    const expectedPcmSha256 = [
      ['raw', rawPcmSha256, values['expect-raw-pcm-sha256']],
      ['processed', processedPcmSha256, values['expect-processed-pcm-sha256']],
    ] as const;
    const pcmHashMismatches = expectedPcmSha256.filter(
      ([, actual, expected]) => expected !== undefined && actual !== expected.toLowerCase(),
    );
    for (const [name, actual, expected] of pcmHashMismatches)
      console.error(`PCM SHA-256 mismatch (${name}): expected ${expected}, got ${actual}`);
    // Finite, non-silent output only; detecting wrong-but-plausible speech
    // requires fixed-seed digests or the external ASR oracle. The 1e-4 RMS
    // floor (~-80 dBFS) is an order of magnitude below the quietest measured
    // speech stream, so it rejects silence and corrupt output without
    // penalizing legitimately quiet speech.
    const processedPcmRms = processedSamples ? Math.sqrt(processedPcmSquares / processedSamples) : 0;
    const passedAudioContent = generatedSeconds > 0 && nonFiniteSamples === 0 && processedPcmRms > 1e-4;
    const passedPcmHash = expectedPcmSha256.some(([, , expected]) => expected !== undefined)
      ? pcmHashMismatches.length === 0
      : null;
    const report = {
      date: new Date().toISOString(),
      model: values.model,
      checkpoint,
      capabilities: model.capabilities,
      reference: values.reference ? { path: values.reference, sha256: referenceSha256, transcript } : null,
      environment: {
        cpu: cpus()[0]?.model,
        memoryBytes: totalmem(),
        platform: platform(),
        arch: arch(),
        osRelease: release(),
        node: process.version,
        note: values.note ?? null,
      },
      execution,
      precision: {
        dtype: checkpointConfig.dtype ?? checkpointConfig.torch_dtype ?? 'checkpoint',
        quantization: checkpointConfig.quantization ?? null,
        codec: 'source precision',
      },
      parameters: {
        ...options,
        signal: undefined,
        instructions,
        instructionCache: values['instruction-cache'] ?? false,
      },
      input: {
        mode: paced ? 'simulated-llm-clock' : values.once ? 'complete-text-once' : 'repeated-text',
        textFile: values['text-file'] ?? null,
        text,
        graphemesPerSecond: paced ? Number(values['graphemes-per-second']) : null,
        chunkGraphemes: paced ? Number(values['chunk-graphemes']) : null,
        firstChunkMs: paced ? Number(values['first-chunk-ms']) : null,
        graphemes: paced?.graphemes,
        // Input clock starts at the first iterator pull; align it to submission.
        clockOffsetMs: paced?.startedAt === undefined ? null : paced.startedAt - began,
        simulatedCompletionMs: paced?.simulatedCompletionMs,
        playbackBeforeSimulatedCompletion,
        deliveryCompletionMs: paced?.deliveryCompletionMs,
        eofMs: paced?.eofMs,
        firstPcmBeforeSimulatedCompletion:
          paced?.startedAt !== undefined && firstPcmMs !== undefined
            ? firstPcmMs < paced.startedAt - began + paced.simulatedCompletionMs
            : null,
        deliveries: paced?.deliveries,
      },
      loadMs,
      voicePreparationMs,
      firstPcmMs,
      firstBeforeInputEnded,
      synthesis,
      audioProcessing: {
        ...processed.stats,
        rawPcmSha256,
        processedPcmSha256,
        processedPcmRms,
        nonFiniteSamples,
        speed: Number(values.speed),
        outputSeconds: generatedSeconds,
      },
      timing: {
        observedStreamWallMs,
        observedStreamWallRatio: generatedSeconds ? observedStreamWallMs / 1000 / generatedSeconds : null,
        playbackFinishWallMs,
        measuredServiceTimeMs,
        measuredServiceTimeRatio,
        note: 'Service intervals may overlap and omit untimed overhead. Wall time includes input, sinks and backpressure.',
      },
      segments,
      playbackPolicy,
      playback,
      playbackComplete,
      // Native timestamps start inside open(). Bracket that instant on the JS
      // monotonic clock rather than silently treating player-open as submission.
      firstPlaybackFromSubmissionMs,
      blockIntervalMs: {
        p50: percentile(intervals, 0.5),
        p95: percentile(intervals, 0.95),
        p99: percentile(intervals, 0.99),
        max: Math.max(...intervals),
      },
      rawSegmentBoundaryMs: {
        p50: percentile(segmentFirstPcm, 0.5),
        p95: percentile(segmentFirstPcm, 0.95),
        max: Math.max(...segmentFirstPcm),
      },
      // Node normalizes uv_getrusage().maxRSS to KiB on every supported platform.
      peakRssBytes: process.resourceUsage().maxRSS * 1024,
      sampledPeakRssBytes: Math.max(...memory.map((x) => x.rss)),
      memory,
      passedAudioContent,
      passedPcmHash,
      passedMeasuredServiceBudget:
        !controller.signal.aborted &&
        inputEnded &&
        synthesis.finishReason === 'eos' &&
        measuredServiceTimeRatio !== null &&
        measuredServiceTimeRatio < 1,
      passedPacedPlayback: paced
        ? passedAudioContent &&
          playbackComplete &&
          playbackPolicy.prebufferSeconds <= 1 &&
          playback.underruns === 0 &&
          playbackBeforeSimulatedCompletion === true &&
          inputEnded &&
          synthesis.finishReason === 'eos' &&
          measuredServiceTimeRatio !== null &&
          measuredServiceTimeRatio < 1
        : null,
      passedTenMinutePlayback:
        passedAudioContent &&
        playbackComplete &&
        playbackPolicy.prebufferSeconds <= 1 &&
        playback.playedSeconds >= 600 &&
        playback.underruns === 0 &&
        firstBeforeInputEnded &&
        (!paced || playbackBeforeSimulatedCompletion === true) &&
        synthesis.finishReason === 'eos' &&
        inputEnded &&
        measuredServiceTimeRatio !== null &&
        measuredServiceTimeRatio < 1,
    };
    console.log(JSON.stringify(report, null, 2));
    if (values.report) await writeFile(values.report, JSON.stringify(report, null, 2) + '\n');
  } finally {
    process.removeListener('SIGINT', cancel);
    controller.abort();
    player?.cancel();
    await writer?.close();
  }
} finally {
  await model.dispose();
}
