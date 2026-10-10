import { setTimeout as delay } from 'node:timers/promises';

export interface InputDelivery {
  offset: number;
  graphemes: number;
  availableMs: number;
  deliveredMs: number;
}

/** A virtual LLM clock, independent of downstream pulls. Text that becomes
 * available during backpressure can be delivered immediately on the next pull.
 * This simulates a buffered text response, not an actual model/token benchmark. */
export function pacedInput(
  text: string,
  options: { graphemesPerSecond: number; chunkGraphemes: number; firstChunkMs: number },
  signal: AbortSignal,
) {
  const { graphemesPerSecond, chunkGraphemes, firstChunkMs } = options;
  if (!Number.isFinite(graphemesPerSecond) || graphemesPerSecond <= 0)
    throw new RangeError('graphemesPerSecond must be positive and finite');
  if (!Number.isSafeInteger(chunkGraphemes) || chunkGraphemes < 1)
    throw new RangeError('chunkGraphemes must be a positive integer');
  if (!Number.isFinite(firstChunkMs) || firstChunkMs < 0)
    throw new RangeError('firstChunkMs must be nonnegative and finite');
  const units = [...new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(text)].map((x) => x.segment);
  const deliveries: InputDelivery[] = [];
  let started: number | undefined;
  let eofMs: number | undefined;
  // The first chunk is ready after firstChunkMs; subsequent chunks become ready
  // when all their graphemes have been produced at the configured rate.
  const available = (end: number) =>
    firstChunkMs + ((end - Math.min(chunkGraphemes, units.length)) * 1000) / graphemesPerSecond;
  return {
    deliveries,
    graphemes: units.length,
    simulatedCompletionMs: units.length ? available(units.length) : 0,
    get deliveryCompletionMs() {
      return deliveries.at(-1)?.deliveredMs;
    },
    get eofMs() {
      return eofMs;
    },
    get startedAt() {
      return started;
    },
    async *[Symbol.asyncIterator]() {
      if (started !== undefined) throw new Error('Paced input can only be consumed once');
      started = performance.now();
      for (let offset = 0; offset < units.length; offset += chunkGraphemes) {
        const part = units.slice(offset, offset + chunkGraphemes);
        const availableMs = available(offset + part.length);
        const wait = availableMs - (performance.now() - started);
        signal.throwIfAborted();
        if (wait > 0) await delay(wait, undefined, { signal });
        signal.throwIfAborted();
        deliveries.push({ offset, graphemes: part.length, availableMs, deliveredMs: performance.now() - started });
        yield part.join('');
      }
      eofMs = performance.now() - started;
    },
  };
}
