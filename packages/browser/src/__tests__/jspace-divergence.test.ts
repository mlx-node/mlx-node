import { describe, expect, it } from 'vitest';
import { argmaxDisagree, jaccardTopK } from '@/jlens-core/divergence';
import { divergingRamp } from '@/jlens-core/colors';
import { reviveRun } from '@/jlens-core/revive';
import { STARTERS, STARTER_SLUGS } from '@/jspace/starters';
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

// The DivergenceGridCanvas paints jaccardTopK(logitSlice.cellAt(l,p),
// jacSlice.cellAt(l,p)) over ONE grid, reading BOTH runs of a baked frame by the
// SAME (layerIdx, pos). That indexing is only sound if, for every gallery frame,
// the two revived runs share their shape. Guard it here so a bad re-bake (drifted
// layers / topK / cell count between the logit and jacobian runs) fails loudly
// instead of silently mis-pairing cells in the divergence view.
describe('baked divergence pair alignment (every gallery frame)', () => {
  it('logit and jacobian runs revive to the same layers, topK===10, and cell count', () => {
    expect(STARTER_SLUGS.length).toBeGreaterThan(0);
    for (const slug of STARTER_SLUGS) {
      const frame = STARTERS[slug]!;
      const logit = reviveRun(frame.logit);
      const jac = reviveRun(frame.jacobian);
      // Same layer axis (identical numbers AND order) — the row order + gutter labels.
      expect(jac.layers).toEqual(logit.layers);
      // Same position axis, so cellAt(l,p) addresses the SAME cell in both runs.
      expect(jac.promptLen).toBe(logit.promptLen);
      // Full shipped top-10 on both sides (jaccardTopK's honest depth bound).
      expect(logit.topK).toBe(10);
      expect(jac.topK).toBe(10);
      // cells is layer-major/position-minor: length === layers × promptLen, matched.
      expect(logit.cells.length).toBe(logit.layers.length * logit.promptLen);
      expect(jac.cells.length).toBe(logit.cells.length);
      // COORDINATE IDENTITY (not just count): every flat cell index i must address
      // the SAME (layer, position) in both runs — the exact indexing the canvas
      // relies on (logit.cellAt(l,p) paired with jac.cellAt(l,p)). Equal counts
      // alone would still pass if the two runs' cells were ordered differently, so
      // assert per-cell layer/position equality across the whole flat grid.
      for (let i = 0; i < logit.cells.length; i++) {
        expect(jac.cells[i]!.layer).toBe(logit.cells[i]!.layer);
        expect(jac.cells[i]!.position).toBe(logit.cells[i]!.position);
      }
    }
  });
});
