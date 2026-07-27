/**
 * F6 — percentages must never round into a claim the underlying counts
 * contradict.
 *
 * The shipped formatter was a bare `Math.round(fraction * 100)`, so a real run
 * with 16189 hits and 49 misses (99.6982 %) rendered as a flat "100%" directly
 * above the sub-line "16.2K hits · 49 misses". A cache page that says 100 %
 * while misses are visible in the same tile is lying.
 *
 * These live in `packages/dashboard/__test__` (node environment) and import the
 * SPA module directly, the way `shell.test.ts` and `cache.test.ts` already
 * reach into `ui/src/lib`. The rules themselves belong in `format.ts` rather
 * than inline in `cache.tsx` so they have one definition and one test; the
 * PAGES that call them are guarded separately by `pages.test.ts`, which mounts
 * the real components under a `happy-dom` environment scoped to that file.
 */
import { describe, expect, it } from 'vite-plus/test';

import { formatPercent, formatRate, percentInt } from '../ui/src/lib/format.js';

describe('formatPercent', () => {
  it('never rounds a non-full fraction up to 100%', () => {
    // The exact reported value: 16189 / (16189 + 49).
    expect(formatPercent(16189 / 16238)).toBe('>99%');
    expect(formatPercent(0.999)).toBe('>99%');
    expect(formatPercent(0.995)).toBe('>99%');
  });

  it('never rounds a non-zero fraction down to 0%', () => {
    expect(formatPercent(0.0004)).toBe('<1%');
    expect(formatPercent(0.004999)).toBe('<1%');
  });

  it('still reports genuine endpoints exactly', () => {
    // The quota meter produces an exact 1 via Math.min(1, used / quota), and an
    // exact 0 when nothing is stored. Those must keep reading 100% / 0%.
    expect(formatPercent(1)).toBe('100%');
    expect(formatPercent(0)).toBe('0%');
  });

  it('rounds ordinary values as before', () => {
    expect(formatPercent(0.73)).toBe('73%');
    expect(formatPercent(0.5)).toBe('50%');
    expect(formatPercent(0.126)).toBe('13%');
  });

  it('treats a non-finite fraction as zero (pre-existing guard, unchanged)', () => {
    expect(formatPercent(Number.NaN)).toBe('0%');
    expect(formatPercent(Number.POSITIVE_INFINITY)).toBe('0%');
  });
});

describe('percentInt', () => {
  // The usage meter drives aria-valuenow and the CSS width from its own
  // rounding, so fixing only the string would leave the announced value at 100
  // while the visible label reads >99%.
  it('saturates the same way the visible label does', () => {
    expect(percentInt(16189 / 16238)).toBe(99);
    expect(percentInt(0.9999)).toBe(99);
    expect(percentInt(0.0001)).toBe(1);
    expect(percentInt(1)).toBe(100);
    expect(percentInt(0)).toBe(0);
    expect(percentInt(0.42)).toBe(42);
  });
});

describe('formatRate', () => {
  // A fraction cannot know a miss ever happened; counts can. This is the
  // formatter the two hit-rate tiles use, and its contract is stated on the
  // integers, not on the rounding.
  it('cannot print 100% while a miss exists', () => {
    expect(formatRate(16189, 16238)).toBe('>99%');
    expect(formatRate(9999, 10000)).toBe('>99%');
    expect(formatRate(1_000_000, 1_000_001)).toBe('>99%');
  });

  it('cannot print 0% while a hit exists', () => {
    expect(formatRate(4, 10000)).toBe('<1%');
    expect(formatRate(1, 1_000_000)).toBe('<1%');
  });

  it('prints an exact 100% only when misses are truly zero', () => {
    expect(formatRate(5, 5)).toBe('100%');
    expect(formatRate(1, 1)).toBe('100%');
  });

  it('prints an exact 0% only when hits are truly zero', () => {
    expect(formatRate(0, 49)).toBe('0%');
  });

  it('reports no lookups as an em dash rather than 0%', () => {
    expect(formatRate(0, 0)).toBe('—');
    expect(formatRate(3, -1)).toBe('—');
    expect(formatRate(Number.NaN, 10)).toBe('—');
  });

  it('rounds ordinary rates', () => {
    expect(formatRate(73, 100)).toBe('73%');
    expect(formatRate(1, 3)).toBe('33%');
  });
});
