import { describe, expect, it } from 'vite-plus/test';

import { ALPHA_MAX, ALPHA_MIN, tintAlphaForRank } from '../../demo/jspace/argmax-tint';

describe('tintAlphaForRank', () => {
  it('argmax (rank 0) is byte-identical to the legacy 0.18 tint', () => {
    expect(tintAlphaForRank(0, 10)).toBeCloseTo(ALPHA_MAX, 10);
    expect(ALPHA_MAX).toBe(0.18);
  });
  it('top-K tail reaches ALPHA_MIN', () => {
    expect(tintAlphaForRank(9, 10)).toBeCloseTo(ALPHA_MIN, 10);
  });
  it('K=1 does not divide by zero and returns ALPHA_MAX', () => {
    expect(tintAlphaForRank(0, 1)).toBe(ALPHA_MAX);
  });
  it('is strictly decreasing and linear in rank', () => {
    const a = Array.from({ length: 10 }, (_, k) => tintAlphaForRank(k, 10));
    for (let k = 1; k < a.length; k++) expect(a[k]!).toBeLessThan(a[k - 1]!);
    for (let k = 0; k < a.length; k++) expect(a[k]!).toBeCloseTo(0.18 - 0.015 * k, 10);
  });
});
