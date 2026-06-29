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

  it('treats whitespace-only stop sequences as no configuration', () => {
    const buffer = new StopSequenceBuffer([' ', '\n', '\t']);

    expect(buffer.push('one two\nthree')).toEqual({ safeText: 'one two\nthree', matched: null });
    expect(buffer.flush()).toEqual({ safeText: '', matched: null });
    expect(buffer.matched).toBeNull();
  });

  it('holds a tail match when a longer same-index stop could still complete on a later push', () => {
    const buffer = new StopSequenceBuffer(['ab', 'abc']);

    // "ab" is complete at the tail, but "abc" (same start index) is still
    // viable, so the match must be held rather than committed early.
    expect(buffer.push('xab')).toEqual({ safeText: 'x', matched: null });
    // The next push completes the longer stop, which wins on the tie.
    expect(buffer.push('c tail')).toEqual({ safeText: '', matched: 'abc' });
    expect(buffer.matched).toBe('abc');
  });

  it('commits the short stop when the longer same-index stop is broken on a later push', () => {
    const buffer = new StopSequenceBuffer(['ab', 'abc']);

    expect(buffer.push('xab')).toEqual({ safeText: 'x', matched: null });
    // No "c" arrives, so the longer "abc" can never complete and "ab" wins.
    expect(buffer.push(' x')).toEqual({ safeText: '', matched: 'ab' });
    expect(buffer.matched).toBe('ab');
  });

  it('resolves a held tail match at flush when no longer stop completes', () => {
    const buffer = new StopSequenceBuffer(['ab', 'abc']);

    expect(buffer.push('xab')).toEqual({ safeText: 'x', matched: null });
    // Stream ends with only "ab" present; the held match resolves to "ab".
    expect(buffer.flush()).toEqual({ safeText: '', matched: 'ab' });
    expect(buffer.matched).toBe('ab');
  });

  it('holds a short tail match while an EARLIER-index longer stop could still complete', () => {
    // Finding A: "xab" contains the short stop "ab" at index 1, but it is also
    // a viable prefix of the longer stop "xabc" beginning at the EARLIER index
    // 0. Earliest-match-wins means the short "ab" must be HELD (not committed)
    // until the earlier "xabc" either completes or is broken.
    const buffer = new StopSequenceBuffer(['xabc', 'ab']);

    expect(buffer.push('xab')).toEqual({ safeText: '', matched: null });
    // The next char completes the earlier/longer "xabc", which wins.
    expect(buffer.push('c')).toEqual({ safeText: '', matched: 'xabc' });
    expect(buffer.matched).toBe('xabc');
  });

  it('commits the short stop when the earlier longer stop can no longer complete', () => {
    const buffer = new StopSequenceBuffer(['xabc', 'ab']);

    expect(buffer.push('xab')).toEqual({ safeText: '', matched: null });
    // "z" breaks "xabc" (it needed "c"); the short "ab" at index 1 commits
    // promptly, emitting the safe "x" before it — no hang, no dropped text.
    expect(buffer.push('z')).toEqual({ safeText: 'x', matched: 'ab' });
    expect(buffer.matched).toBe('ab');
  });

  it('matches the earlier longer stop in a single chunk', () => {
    const buffer = new StopSequenceBuffer(['xabc', 'ab']);

    expect(buffer.push('xabc')).toEqual({ safeText: '', matched: 'xabc' });
    expect(buffer.matched).toBe('xabc');
  });
});
