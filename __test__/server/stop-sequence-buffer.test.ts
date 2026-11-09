import { describe, expect, it } from 'vite-plus/test';

import { StopSequenceBuffer } from '../../packages/server/src/stop-sequence-buffer.js';

describe('StopSequenceBuffer', () => {
  it('passes text through transparently when no stop sequences are configured', () => {
    const buffer = new StopSequenceBuffer([]);

    expect(buffer.push('abc')).toEqual({ safeText: 'abc', matched: null });
    expect(buffer.push('def')).toEqual({ safeText: 'def', matched: null });
    expect(buffer.flush()).toEqual({ safeText: '', matched: null });
    expect(buffer.matched).toBeNull();
  });

  it('refuses an empty stop sequence, which the request validation rejects first', () => {
    expect(() => new StopSequenceBuffer(['', 'x'])).toThrow('must not be empty');
  });

  it('detects a whole stop sequence within a single push and suppresses the rest', () => {
    const buffer = new StopSequenceBuffer(['HALT']);

    expect(buffer.push('abcHALTxyz')).toEqual({ safeText: 'abc', matched: 'HALT' });
    expect(buffer.push('more')).toEqual({ safeText: '', matched: 'HALT' });
    expect(buffer.matched).toBe('HALT');
  });

  it('detects a stop sequence split across two pushes', () => {
    const buffer = new StopSequenceBuffer(['HALT']);

    expect(buffer.push('abcHAL')).toEqual({ safeText: 'abc', matched: null });
    expect(buffer.push('Tyz')).toEqual({ safeText: '', matched: 'HALT' });
    expect(buffer.matched).toBe('HALT');
  });

  it('releases a false partial that never completes a stop sequence', () => {
    const buffer = new StopSequenceBuffer(['HALT']);

    expect(buffer.push('abHAL')).toEqual({ safeText: 'ab', matched: null });
    expect(buffer.push('xz')).toEqual({ safeText: 'HALxz', matched: null });
    expect(buffer.flush()).toEqual({ safeText: '', matched: null });
    expect(buffer.matched).toBeNull();
  });

  it('detects a stop sequence at the very start of the text', () => {
    const buffer = new StopSequenceBuffer(['STOP']);

    expect(buffer.push('STOPnow')).toEqual({ safeText: '', matched: 'STOP' });
  });

  it('matches the earliest stop sequence when multiple are present', () => {
    const buffer = new StopSequenceBuffer(['END', 'HALT']);

    expect(buffer.push('aHALTbENDc')).toEqual({ safeText: 'a', matched: 'HALT' });
  });

  it('takes the first listed stop when two begin at the same index (Splash StopSearch)', () => {
    const buffer = new StopSequenceBuffer(['ab', 'abc']);

    expect(buffer.push('xabc')).toEqual({ safeText: 'x', matched: 'ab' });
  });

  it('never emits a held-back partial suffix prematurely', () => {
    const buffer = new StopSequenceBuffer(['STOP']);

    // "ST" could begin "STOP", so it must be withheld, not emitted.
    const first = buffer.push('abcST');
    expect(first.safeText).toBe('abc');
    expect(first.safeText).not.toContain('ST');
    expect(first.matched).toBeNull();

    // The withheld suffix is only released by flush when it cannot complete.
    expect(buffer.flush()).toEqual({ safeText: 'ST', matched: null });
  });

  it('honours a whitespace stop; Messages validation is what rejects one', () => {
    const buffer = new StopSequenceBuffer(['\n']);

    expect(buffer.push('one two\nthree')).toEqual({ safeText: 'one two', matched: '\n' });
  });

  it('commits the first stop to complete, without waiting for a longer one', () => {
    const buffer = new StopSequenceBuffer(['ab', 'abc']);

    expect(buffer.push('xab')).toEqual({ safeText: 'x', matched: 'ab' });
    expect(buffer.push('c tail')).toEqual({ safeText: '', matched: 'ab' });
  });

  it('commits a stop that completes while an earlier-starting stop is still partial', () => {
    const buffer = new StopSequenceBuffer(['xabc', 'ab']);

    expect(buffer.push('xab')).toEqual({ safeText: 'x', matched: 'ab' });
  });

  // Splash test_native_backend: tokens "a","a","ab" with stops ("ab","aab").
  it('ends where the earliest-starting stop begins when two complete in one chunk', () => {
    const buffer = new StopSequenceBuffer(['ab', 'aab']);

    expect(buffer.push('a')).toEqual({ safeText: '', matched: null });
    expect(buffer.push('a')).toEqual({ safeText: '', matched: null });
    expect(buffer.push('ab')).toEqual({ safeText: 'a', matched: 'aab' });
  });

  it('matches the earlier longer stop in a single chunk', () => {
    const buffer = new StopSequenceBuffer(['xabc', 'ab']);

    expect(buffer.push('xabc')).toEqual({ safeText: '', matched: 'xabc' });
  });

  it('keeps a self-overlapping partial across pushes (KMP fallback)', () => {
    const buffer = new StopSequenceBuffer(['aab']);

    expect(buffer.push('aaa')).toEqual({ safeText: 'a', matched: null });
    expect(buffer.pending).toBe('aa');
    expect(buffer.push('b!')).toEqual({ safeText: '', matched: 'aab' });
  });

  // Splash test_callback_streamer_holds_two_long_stops_in_linear_time: the
  // output runs 10,000 characters into one stop while the other is checked
  // again on every token. Steps count the characters each string primitive
  // touches. Mutation: the old per-push indexOf + every-suffix startsWith scan
  // → red (about 10^8 steps).
  it('holds two 10,000-character stops over 10,000 tokens in linear steps', () => {
    const length = 10_000;
    const buffer = new StopSequenceBuffer(['a'.repeat(length) + 'X', 'b'.repeat(length) + 'Y']);
    let released = '';
    const steps = countStringSteps(() => {
      for (let token = 0; token < length; token++) released += buffer.push('a').safeText;
      expect(released).toBe('');
      released += buffer.push('Z').safeText + buffer.flush().safeText;
    });
    expect(released).toBe('a'.repeat(length) + 'Z');
    expect(buffer.matched).toBeNull();
    expect(steps).toBeLessThan(40 * (length + length));
  });
});

/** Characters touched by string primitives while `run` executes. */
function countStringSteps(run: () => void): number {
  const cost: Record<string, (self: string, args: unknown[], result: unknown) => number> = {
    charCodeAt: () => 1,
    indexOf: (self) => self.length,
    includes: (self) => self.length,
    startsWith: (_self, args) => String(args[0]).length,
    endsWith: (_self, args) => String(args[0]).length,
    slice: (_self, _args, result) => (result as string).length,
  };
  const saved = Object.getOwnPropertyDescriptors(String.prototype);
  let steps = 0;
  for (const [name, measure] of Object.entries(cost)) {
    const original = saved[name]!.value as (...args: unknown[]) => unknown;
    Object.defineProperty(String.prototype, name, {
      ...saved[name],
      value(this: string, ...args: unknown[]) {
        const result = Reflect.apply(original, this, args);
        steps += measure(this, args, result);
        return result;
      },
    });
  }
  try {
    run();
  } finally {
    for (const name of Object.keys(cost)) Object.defineProperty(String.prototype, name, saved[name]!);
  }
  return steps;
}
