import { describe, expect, it } from 'vitest';
import * as metrics from '@/jlens-core/workspace-metrics';
import { buildLensSlice } from '@/jlens-core/types';
import { reviveRun } from '@/jlens-core/revive';
import { STARTERS } from '@/jspace/starters';
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
  it('conceptRankTrajectory: per-layer ranks off the pinned track (exact below the cap)', () => {
    expect(metrics.conceptRankTrajectory(slice, 0)).toEqual([2, 1, 1]);
  });
  it('conceptRankTrajectory: RANK_CAP (999) is a censored floor, flagged by isCensoredRank', () => {
    // A concept that never surfaces early: the producer stores min(rank, 999), so a
    // 999 means "at/beyond cap — off-scale", NOT an exact rank. The trajectory
    // returns it verbatim; isCensoredRank marks it so the UI renders it off-scale.
    const censored = buildLensSlice({
      ...run(),
      // Cells consistent with the ranks: id 9 is absent from top-10 at L0 (rank 999)
      // and L1 (rank 40), present at L2 (rank 1) — so membership matches [0,0,1].
      cells: [
        cell(5, [5, 1, 2], [0.9, 0.05, 0.05]),
        cell(5, [5, 2, 3], [0.34, 0.33, 0.33]),
        cell(9, [9, 5, 2], [0.34, 0.33, 0.33]),
      ],
      pinned: [{ tokenId: 9, tokenText: '9', ranks: Int32Array.from([999, 40, 1]) }],
    });
    expect(metrics.conceptRankTrajectory(censored, 0)).toEqual([999, 40, 1]);
    expect(metrics.isCensoredRank(999)).toBe(true);
    expect(metrics.isCensoredRank(40)).toBe(false);
    // top-k accuracy reads set membership: id 9 surfaces only at L2.
    expect(metrics.conceptTopKAccuracy(censored, 0, 10)).toEqual([0, 0, 1]);
  });
  it('conceptTopKAccuracy: committed grammar-error ℓ22 is NOT a false positive (codex T4)', () => {
    // Shipped data: at grammar-error.jacobian ℓ22 / final position, pinned
    // ' Incorrect' (79034) has stored rank 10 but is ABSENT from the ten topKIds
    // (a cutoff tie). A rank<=10 impl WOULD false-positive here; set membership
    // must read 0. Guards the exact regression codex T4 found in committed data.
    const g = buildLensSlice(reviveRun(STARTERS['grammar-error']!.jacobian));
    expect(g.layers).toContain(22);
    const li = g.layers.indexOf(22);
    const pos = g.promptLen - 1;
    expect(g.rankAt(0, li, pos)).toBe(10); // the stored rank a rank<=10 test would trust
    expect(metrics.conceptTopKAccuracy(g, 0, 10)[li]).toBe(0); // membership: honest 0
  });
  it('conceptTopKAccuracy: 1 iff the concept is in the shipped top-k SET', () => {
    // id 9 sits at index 1 in each cell's topKIds ([5,9,1] / [9,5,2] / [9,5,2]).
    expect(metrics.conceptTopKAccuracy(slice, 0, 1)).toEqual([0, 1, 1]); // top-1: only L1/L2
    expect(metrics.conceptTopKAccuracy(slice, 0, 2)).toEqual([1, 1, 1]); // top-2: all
  });
  it('conceptTopKAccuracy: tie-proof — a rank<=k token absent from topKIds reads 0 (codex T4)', () => {
    // Backend rank ties at the cutoff, so a token can have rank<=k yet be excluded
    // from the selected top-k. Membership over topKIds is the honest exact signal.
    const tie = buildLensSlice({
      ...run(),
      // id 9 is ABSENT from L0's topKIds [5,1,2] but its stored rank is 5 (a cutoff tie).
      cells: [
        cell(5, [5, 1, 2], [0.5, 0.3, 0.2]),
        cell(9, [9, 5, 2], [0.5, 0.3, 0.2]),
        cell(9, [9, 5, 2], [0.5, 0.3, 0.2]),
      ],
      pinned: [{ tokenId: 9, tokenText: '9', ranks: Int32Array.from([5, 1, 1]) }],
    });
    // A rank<=10 impl would say [1,1,1]; set membership correctly says [0,1,1].
    expect(metrics.conceptTopKAccuracy(tie, 0, 10)).toEqual([0, 1, 1]);
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
  it('motorFlipLayer: null when the run is ONLY the top layer (no observed lock, codex whole-branch)', () => {
    // Output argmax 9 appears at no displayed layer below the top → degenerate.
    // Must be null (rendered "no lock"), NOT the top index (a false late commit).
    const late = buildLensSlice({ ...run(), cells: [cell(5, [5], [1]), cell(6, [6], [1]), cell(9, [9], [1])] });
    expect(metrics.motorFlipLayer(late, 0)).toBeNull();
    // A 1-layer slice also has no below-top evidence → null.
    const one = buildLensSlice({
      ...run(),
      layers: [24],
      cells: [cell(9, [9], [1])],
      pinned: [{ tokenId: 9, tokenText: '9', ranks: Int32Array.from([1]) }],
    });
    expect(metrics.motorFlipLayer(one, 0)).toBeNull();
  });
  it('motorFlipLayer: committed french-season is all no-lock on sparse baked layers (codex whole-branch)', () => {
    // Shipped baked frames sample only [6,8,10,12,14,16,17,18,20,22,24]; the output
    // argmax usually first wins in the ℓ22→ℓ24 gap, so EVERY french-season position
    // is a no-lock (null). This is exactly why the UI shows motor-flip on LIVE
    // (contiguous 1..24) runs only, never on sparse baked starters.
    const fs = buildLensSlice(reviveRun(STARTERS['french-season']!.jacobian));
    for (let p = 0; p < fs.promptLen; p++) expect(metrics.motorFlipLayer(fs, p)).toBeNull();
  });
  it('does NOT export the impossible full-distribution metrics', () => {
    expect((metrics as Record<string, unknown>).excessKurtosis).toBeUndefined();
    expect((metrics as Record<string, unknown>).residualAutocorrelation).toBeUndefined();
    expect((metrics as Record<string, unknown>).participationRatio).toBeUndefined();
  });
});
