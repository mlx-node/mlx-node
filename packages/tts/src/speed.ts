import type { AudioChunk } from './types.js';

export interface AudioProcessingStats {
  processingMs: number;
  inputFrames: number;
  outputFrames: number;
}
export interface AudioSpeedStream extends AsyncIterable<AudioChunk> {
  /** Updated as the output is consumed. Upstream inference and sink waits are excluded. */
  readonly stats: Readonly<AudioProcessingStats>;
}

/** Pitch-preserving PCM transform. Each segment flushes its own short history;
 * frame offsets describe the transformed output. Early return closes upstream.
 * Speed 1 is an exact passthrough. Inference statistics remain on the source. */
export function changeAudioSpeed(source: AsyncIterable<AudioChunk>, speed: number): AudioSpeedStream {
  if (!Number.isFinite(speed) || speed < 0.25 || speed > 4)
    throw new RangeError('Speech speed must be finite and within 0.25..4');
  const stats: AudioProcessingStats = { processingMs: 0, inputFrames: 0, outputFrames: 0 };
  let consumed = false;
  let upstream: AsyncIterator<AudioChunk>;
  let closing: Promise<unknown> | undefined;
  let ended = false;
  function closeSource(): Promise<unknown> {
    return (closing ??= ended ? Promise.resolve() : Promise.resolve().then(() => upstream.return?.()));
  }
  async function* transform(): AsyncGenerator<AudioChunk> {
    let processor: import('@mlx-node/core').PcmTempo | undefined;
    let previous: AudioChunk | undefined;
    let expectedInput = 0;
    const timed = (operation: () => Float32Array): Float32Array => {
      const start = performance.now();
      try {
        return operation();
      } finally {
        stats.processingMs += performance.now() - start;
      }
    };
    const chunk = (samples: Float32Array, metadata: AudioChunk): AudioChunk => {
      const startSample = stats.outputFrames;
      stats.outputFrames += samples.length / metadata.channels;
      return { ...metadata, samples, startSample };
    };
    try {
      const Native = speed === 1 ? undefined : (await import('@mlx-node/core')).PcmTempo;
      while (true) {
        const item = await upstream.next();
        if (item.done) {
          ended = true;
          break;
        }
        const input = item.value;
        if (
          !Number.isSafeInteger(input.sampleRate) ||
          input.sampleRate <= 0 ||
          !Number.isSafeInteger(input.channels) ||
          input.channels <= 0 ||
          input.samples.length % input.channels ||
          input.startSample !== expectedInput ||
          !Number.isSafeInteger(input.segmentIndex) ||
          input.segmentIndex < 0
        )
          throw new Error('Invalid or discontinuous PCM stream');
        if (
          previous &&
          (input.sampleRate !== previous.sampleRate ||
            input.channels !== previous.channels ||
            input.segmentIndex < previous.segmentIndex)
        )
          throw new Error('PCM format or segment order changed');
        if (input.samples.some((value) => !Number.isFinite(value))) throw new Error('Non-finite PCM sample');
        // Fail before constructing the native processor: SpeechTempo accepts
        // 1000..=500_000 Hz and 1..=32 channels and would throw a generic error.
        if (Native && (input.sampleRate < 1000 || input.sampleRate > 500_000 || input.channels > 32))
          throw new RangeError('Invalid PCM stream');
        if (previous && previous.segmentIndex !== input.segmentIndex && processor) {
          const tail = timed(() => processor!.finish());
          processor.close();
          processor = undefined;
          if (tail.length) yield chunk(tail, previous);
        }
        if (Native && !processor) {
          const start = performance.now();
          processor = new Native(input.sampleRate, input.channels, speed);
          stats.processingMs += performance.now() - start;
        }
        stats.inputFrames += input.samples.length / input.channels;
        expectedInput += input.samples.length / input.channels;
        previous = input;
        const samples = processor ? timed(() => processor!.write(input.samples)) : input.samples;
        if (samples.length) yield chunk(samples, input);
      }
      if (processor && previous) {
        const tail = timed(() => processor!.finish());
        if (tail.length) yield chunk(tail, previous);
      }
    } finally {
      processor?.close();
      await closeSource();
    }
  }
  return {
    get stats() {
      return { ...stats };
    },
    [Symbol.asyncIterator]() {
      if (consumed) throw new Error('Audio speed stream can only be consumed once');
      consumed = true;
      upstream = source[Symbol.asyncIterator]();
      const iterator = transform();
      return {
        next: () => iterator.next(),
        async return() {
          // Begin source cancellation before awaiting a pending transform pull.
          // An unstarted generator does not execute its body or finally block.
          const closing = closeSource();
          try {
            return await iterator.return(undefined);
          } finally {
            await closing;
          }
        },
        async throw(error: unknown) {
          const closing = closeSource();
          try {
            return await iterator.throw(error);
          } finally {
            await closing;
          }
        },
      };
    },
  };
}
