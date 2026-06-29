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

  it('treats only-empty stop sequences as no configuration', () => {
    const buffer = new StopSequenceBuffer(['', '']);

    expect(buffer.push('anything')).toEqual({ safeText: 'anything', matched: null });
    expect(buffer.flush()).toEqual({ safeText: '', matched: null });
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

  it('prefers the longest stop sequence on a tie at the same index', () => {
    const buffer = new StopSequenceBuffer(['ab', 'abc']);

    expect(buffer.push('xabc')).toEqual({ safeText: 'x', matched: 'abc' });
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
});
