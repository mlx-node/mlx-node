/**
 * @vitest-environment happy-dom
 */

import { describe, expect, it } from 'vite-plus/test';

import { workSpans } from '../scripts/dev/gallery/data-sections.js';

describe('gallery agent timeline data', () => {
  it('never makes a span that ends before it starts, and keeps each lane in order', () => {
    const cases: Array<[number, number, number, number]> = [
      [5_700, 15_000, 2_400, 1],
      [5_750, 17_000, 2_800, 2],
      [5_850, 19_000, 3_100, 3],
      [0, 200_000, 24_000, 1],
      [201_000, 420_000, 31_000, 2],
      [202_000, 636_000, 52_000, 3],
      [5_300, 14_000, 2_900, 2],
    ];
    for (let seed = 0; seed < 20; seed += 1) cases.push([0, 400_000, 24_000, seed]);
    for (const [from, to, step, seed] of cases) {
      const spans = workSpans(from, to, step, seed);
      let previousEnd = from;
      for (const span of spans) {
        const where = `${String(from)}..${String(to)} step ${String(step)} seed ${String(seed)}: ${span.label} at ${String(span.startMs)}`;
        expect(span.endMs, where).not.toBeNull();
        expect(span.endMs! > span.startMs, where).toBe(true);
        expect(span.startMs >= previousEnd, where).toBe(true);
        previousEnd = span.endMs!;
      }
      expect(previousEnd).toBe(to);
    }
  });
});
