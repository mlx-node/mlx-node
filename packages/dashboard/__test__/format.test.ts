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

import {
  formatBucketLabel,
  formatBytes,
  formatBytesParts,
  formatClock,
  formatCount,
  formatDay,
  formatDayLong,
  formatDelta,
  formatDuration,
  formatDurationParts,
  formatNumber,
  formatPercent,
  formatRate,
  formatRelativeTime,
  formatTps,
  percentInt,
} from '../ui/src/lib/format.js';

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

  it('prints an unmeasured fraction as an em dash, never 0%', () => {
    expect(formatPercent(Number.NaN)).toBe('—');
    expect(formatPercent(Number.POSITIVE_INFINITY)).toBe('—');
    expect(formatPercent(null)).toBe('—');
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

describe('unmeasured values', () => {
  it('print an em dash in every formatter, never 0', () => {
    for (const value of [null, undefined, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      expect(formatBytes(value)).toBe('—');
      expect(formatCount(value)).toBe('—');
      expect(formatNumber(value)).toBe('—');
      expect(formatPercent(value)).toBe('—');
      expect(formatRate(value, 10)).toBe('—');
      expect(formatRelativeTime(value)).toBe('—');
      expect(formatDuration(value)).toBe('—');
      expect(formatTps(value)).toBe('—');
      expect(formatDay(value)).toBe('—');
      expect(formatDayLong(value)).toBe('—');
      expect(formatClock(value)).toBe('—');
      expect(formatDurationParts(value)).toEqual({ value: '—', unit: '' });
      expect(formatBytesParts(value)).toEqual({ value: '—', unit: '' });
    }
  });

  it('print a byte count under one in bytes', () => {
    expect(formatBytes(0.5)).toBe('1 B');
    expect(formatBytes(0.2)).toBe('0 B');
    expect(formatBytesParts(0.5)).toEqual({ value: '1', unit: 'B' });
  });

  it('keep a measured zero', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatCount(0)).toBe('0');
    expect(formatNumber(0)).toBe('0');
  });
});

describe('formatDuration', () => {
  it('follows the ms / s / m s / h m ladder', () => {
    expect(formatDuration(0.4)).toBe('<1 ms');
    expect(formatDuration(412)).toBe('412 ms');
    expect(formatDuration(999.4)).toBe('999 ms');
    expect(formatDuration(999.6)).toBe('1.00 s');
    expect(formatDuration(1240)).toBe('1.24 s');
    expect(formatDuration(12_400)).toBe('12.4 s');
    expect(formatDuration(59_960)).toBe('1m 00s');
    expect(formatDuration(125_000)).toBe('2m 05s');
    expect(formatDuration(4_320_000)).toBe('1h 12m');
  });

  it('splits value and unit for tiles', () => {
    expect(formatDurationParts(412)).toEqual({ value: '412', unit: 'ms' });
    expect(formatDurationParts(1240)).toEqual({ value: '1.24', unit: 's' });
    expect(formatDurationParts(125_000)).toEqual({ value: '2m 05s', unit: '' });
    expect(formatBytesParts(8.6 * 1024 ** 3)).toEqual({ value: '8.6', unit: 'GB' });
  });
});

describe('formatTps', () => {
  it('keeps one decimal below 100 and none from 100', () => {
    expect(formatTps(8.44)).toBe('8.4');
    expect(formatTps(42.66)).toBe('42.7');
    expect(formatTps(99.96)).toBe('100');
    expect(formatTps(128.4)).toBe('128');
    expect(formatTps(3976.2)).toBe('3,976');
  });
});

describe('formatDelta', () => {
  it('shows the sign, uses a real minus, and calls a small change flat', () => {
    expect(formatDelta(112, 100)).toEqual({ text: '+12%', direction: 'up' });
    expect(formatDelta(97, 100)).toEqual({ text: '\u22123%', direction: 'down' });
    expect(formatDelta(100.4, 100)).toEqual({ text: '0%', direction: 'flat' });
  });

  it('has no delta against a missing or zero previous value', () => {
    expect(formatDelta(5, 0)).toBeNull();
    expect(formatDelta(5, null)).toBeNull();
    expect(formatDelta(null, 5)).toBeNull();
  });
});

describe('dates and clocks', () => {
  const at = new Date(2026, 10, 3, 14, 5, 12).getTime();

  it('formats in local time with the en-US locale', () => {
    expect(formatDay(at)).toBe('Nov 3');
    expect(formatDayLong(at)).toBe('Tue, Nov 3');
    expect(formatClock(at)).toBe('14:05');
    expect(formatClock(at, true)).toBe('14:05:12');
  });

  it('labels a bucket by its grain', () => {
    expect(formatBucketLabel(at, '5m')).toBe('14:05');
    expect(formatBucketLabel(at, 'hour')).toBe('14:05');
    expect(formatBucketLabel(at, 'day')).toBe('Nov 3');
  });
});
