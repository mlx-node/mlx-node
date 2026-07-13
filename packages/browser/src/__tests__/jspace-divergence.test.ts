import { describe, expect, it } from 'vitest';
import { argmaxDisagree, jaccardTopK } from '@/jlens-core/divergence';
import { divergingRamp } from '@/jlens-core/colors';
import type { LensCell } from '@/../src/inspector-types';

function cell(argmaxId: number, topKIds: number[]): LensCell {
  return {
    layer: 0, position: 0, argmaxId, topKIds,
    topKLogits: new Float32Array(topKIds.length),
    topKProbs: new Float32Array(topKIds.length),
    topKTexts: topKIds.map(String),
  };
}

describe('divergence metrics', () => {
  it('argmaxDisagree: 0 when top-1 matches, 1 when it differs', () => {
    expect(argmaxDisagree(cell(7, [7, 1]), cell(7, [7, 2]))).toBe(0);
    expect(argmaxDisagree(cell(7, [7]), cell(8, [8]))).toBe(1);
  });
  it('jaccardTopK: 0 identical, 1 disjoint, 0.5 half-overlap, 0 both-empty', () => {
    expect(jaccardTopK(cell(1, [1, 2, 3]), cell(1, [1, 2, 3]))).toBe(0);
    expect(jaccardTopK(cell(1, [1, 2]), cell(3, [3, 4]))).toBe(1);
    // A={1,2,3}, B={1,2,3,4,5,6} → inter=3 union=6 → similarity .5 → distance .5
    expect(jaccardTopK(cell(1, [1, 2, 3]), cell(1, [1, 2, 3, 4, 5, 6]))).toBeCloseTo(0.5, 6);
    expect(jaccardTopK(cell(0, []), cell(0, []))).toBe(0);
  });
});

describe('divergingRamp', () => {
  it('is blue at 0, near-white at 0.5, red at 1, and clamps', () => {
    expect(divergingRamp(0)).toBe('rgb(37, 99, 235)');
    expect(divergingRamp(1)).toBe('rgb(220, 38, 38)');
    expect(divergingRamp(0.5)).toBe('rgb(241, 245, 249)');
    expect(divergingRamp(-1)).toBe(divergingRamp(0));
    expect(divergingRamp(2)).toBe(divergingRamp(1));
  });
});
