import { describe, expect, it } from 'vitest';
import * as metrics from '@/jlens-core/workspace-metrics';
import { buildLensSlice } from '@/jlens-core/types';
import type { LensCell, LensReadoutRun } from '@/../src/inspector-types';

function cell(argmaxId: number, ids: number[], probs: number[]): LensCell {
  return {
    layer: 0, position: 0, argmaxId, topKIds: ids,
    topKLogits: new Float32Array(ids.length),
    topKProbs: Float32Array.from(probs),
    topKTexts: ids.map(String),
  };
}
// 3 layers × 1 position. argmax sequence: L0=5, L1=9, L2=9 (top). One pin (id 9).
function run(): LensReadoutRun {
  return {
    promptLen: 1, topK: 3, useJacobian: false, jacobianApplied: false, layers: [12, 18, 24],
    tokens: [{ text: 'x', id: 1 } as any],
    cells: [
      cell(5, [5, 9, 1], [0.9, 0.05, 0.05]),   // L0 peaked
      cell(9, [9, 5, 2], [0.34, 0.33, 0.33]),  // L1 flat-ish
      cell(9, [9, 5, 2], [0.34, 0.33, 0.33]),  // L2 (top)
    ],
    pinned: [{ tokenId: 9, tokenText: '9', ranks: Int32Array.from([2, 1, 1]) }],
  };
}

describe('workspace metrics', () => {
  const slice = buildLensSlice(run());
  it('conceptRankTrajectory: exact per-layer ranks off the pinned track', () => {
    expect(metrics.conceptRankTrajectory(slice, 0)).toEqual([2, 1, 1]);
  });
  it('conceptTopKAccuracy: 1 iff rank ≤ k', () => {
    expect(metrics.conceptTopKAccuracy(slice, 0, 1)).toEqual([0, 1, 1]);
    expect(metrics.conceptTopKAccuracy(slice, 0, 2)).toEqual([1, 1, 1]);
  });
  it('readoutEntropy: lower for peaked, higher for flat', () => {
    const peaked = metrics.readoutEntropy(slice, 0, 0);
    const flat = metrics.readoutEntropy(slice, 1, 0);
    expect(flat).toBeGreaterThan(peaked);
    expect(peaked).toBeGreaterThan(0);
  });
  it('readoutEffectiveDim: ~1 for a delta, near topK for uniform', () => {
    expect(metrics.readoutEffectiveDim(slice, 0, 0)).toBeGreaterThan(1);
    expect(metrics.readoutEffectiveDim(slice, 1, 0)).toBeGreaterThan(2.9); // ~3
  });
  it('topKSetStability: adjacent-layer Jaccard similarity, length layers−1', () => {
    // L0 ids {5,9,1} vs L1 {9,5,2}: inter {5,9}=2 union 4 → .5 ; L1 vs L2 identical → 1
    expect(metrics.topKSetStability(slice, 0)).toEqual([0.5, 1]);
  });
  it('motorFlipLayer: lowest top-anchored stable-argmax layer', () => {
    expect(metrics.motorFlipLayer(slice, 0)).toBe(1); // argmax 9 holds from L1 up
  });
  it('motorFlipLayer: commits only at the top when the run never stabilizes early', () => {
    const late = buildLensSlice({ ...run(), cells: [cell(5, [5], [1]), cell(6, [6], [1]), cell(9, [9], [1])] });
    expect(metrics.motorFlipLayer(late, 0)).toBe(2); // only the top row equals the top argmax
  });
  it('does NOT export the impossible full-distribution metrics', () => {
    expect((metrics as Record<string, unknown>).excessKurtosis).toBeUndefined();
    expect((metrics as Record<string, unknown>).residualAutocorrelation).toBeUndefined();
    expect((metrics as Record<string, unknown>).participationRatio).toBeUndefined();
  });
});
