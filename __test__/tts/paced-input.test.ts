import { setTimeout as delay } from 'node:timers/promises';

import { describe, expect, it } from 'vite-plus/test';

import { pacedInput } from '../../scripts/tts/paced-input.js';

describe('simulated LLM text clock', () => {
  it('preserves graphemes and keeps production independent of slow pulls', async () => {
    const text = '甲👨‍👩‍👦e\u0301乙丙';
    const source = pacedInput(
      text,
      { graphemesPerSecond: 1000, chunkGraphemes: 2, firstChunkMs: 0 },
      new AbortController().signal,
    );
    const iterator = source[Symbol.asyncIterator]();
    const parts = [(await iterator.next()).value];
    await delay(20);
    for (let part = await iterator.next(); !part.done; part = await iterator.next()) parts.push(part.value);
    expect(parts).toEqual(['甲👨‍👩‍👦', 'e\u0301乙', '丙']);
    expect(parts.join('')).toBe(text);
    expect(source.graphemes).toBe(5);
    expect(source.deliveries.map((x) => x.availableMs)).toEqual([0, 2, 3]);
    expect(source.simulatedCompletionMs).toBe(3);
    expect(source.deliveries[1].deliveredMs).toBeGreaterThan(10);
    expect(source.deliveryCompletionMs).toBeGreaterThan(source.simulatedCompletionMs);
    expect(source.eofMs).toBeGreaterThanOrEqual(source.deliveryCompletionMs!);
    await expect(source[Symbol.asyncIterator]().next()).rejects.toThrow('only be consumed once');
  });

  it('waits until text is available and aborts a pending first chunk', async () => {
    const source = pacedInput(
      '你好',
      { graphemesPerSecond: 20, chunkGraphemes: 2, firstChunkMs: 30 },
      new AbortController().signal,
    );
    for await (const _ of source) {
      /* drain */
    }
    expect(source.deliveries[0].deliveredMs).toBeGreaterThanOrEqual(25);
    const controller = new AbortController();
    const cancelled = pacedInput(
      '你好',
      { graphemesPerSecond: 20, chunkGraphemes: 2, firstChunkMs: 60_000 },
      controller.signal,
    );
    const pending = cancelled[Symbol.asyncIterator]().next();
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancelled.deliveries).toEqual([]);
    expect(cancelled.eofMs).toBeUndefined();
  });

  it('separates final text delivery from a delayed EOF pull', async () => {
    const source = pacedInput(
      '你好',
      { graphemesPerSecond: 20, chunkGraphemes: 2, firstChunkMs: 0 },
      new AbortController().signal,
    );
    const iterator = source[Symbol.asyncIterator]();
    expect((await iterator.next()).value).toBe('你好');
    const delivered = source.deliveryCompletionMs;
    expect(source.eofMs).toBeUndefined();
    await delay(20);
    expect((await iterator.next()).done).toBe(true);
    expect(source.deliveryCompletionMs).toBe(delivered);
    expect(source.eofMs! - delivered!).toBeGreaterThan(10);
  });

  it('rejects invalid clock settings before iteration', () => {
    const options = { graphemesPerSecond: 20, chunkGraphemes: 2, firstChunkMs: 500 };
    for (const value of [0, -1, NaN, Infinity])
      expect(() => pacedInput('a', { ...options, graphemesPerSecond: value }, new AbortController().signal)).toThrow();
    for (const value of [0, 1.5, Infinity])
      expect(() => pacedInput('a', { ...options, chunkGraphemes: value }, new AbortController().signal)).toThrow();
    for (const value of [-1, NaN, Infinity])
      expect(() => pacedInput('a', { ...options, firstChunkMs: value }, new AbortController().signal)).toThrow();
  });
});
