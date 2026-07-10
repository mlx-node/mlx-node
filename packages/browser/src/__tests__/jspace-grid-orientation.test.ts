import { describe, expect, it } from 'vite-plus/test';

import type { LensReadoutRun } from '../inspector-types';
import { RANK_CAP } from '../../demo/jlens-core/colors';
import { buildLensSlice } from '../../demo/jlens-core/types';
import { displayRowOrder, normalizeSelected, offScaleLabel } from '../../demo/jspace/ArgmaxGridCanvas';
import { rankToY } from '../../demo/jspace/RankChart';

/** layers ASCENDING; cells layer-major: cells[layerIdx * promptLen + pos]. */
function fixture() {
  const layers = [6, 12, 24];
  const promptLen = 2;
  const cells = layers.flatMap((layer, li) =>
    Array.from({ length: promptLen }, (_, pos) => ({
      layer,
      position: pos,
      argmaxId: li * 10 + pos,
      topKIds: [li * 10 + pos],
      topKLogits: Float32Array.from([1]),
      topKProbs: Float32Array.from([1]),
      topKTexts: [`L${layer}P${pos}`],
    })),
  );
  const run: LensReadoutRun = {
    promptLen, topK: 1, useJacobian: false, jacobianApplied: false, layers,
    tokens: [{ id: 1, text: 'a' }, { id: 2, text: 'b' }],
    cells,
    pinned: [{ tokenId: 7, tokenText: 'x', ranks: Int32Array.from([1, 5, 50, 500, RANK_CAP, RANK_CAP]) }],
  };
  return buildLensSlice(run);
}

describe('grid orientation', () => {
  it('renders the DEEPEST layer in the TOP row', () => {
    const slice = fixture();
    const rows = displayRowOrder(slice);           // indices into slice.layers
    expect(slice.layers[rows[0]!]).toBe(24);        // deepest first
    expect(slice.layers[rows[rows.length - 1]!]).toBe(6);
  });

  it('reads cells through cellAt, layer-major', () => {
    const slice = fixture();
    expect(slice.cellAt(2, 1).topKTexts[0]).toBe('L24P1');
    expect(slice.cellAt(0, 0).topKTexts[0]).toBe('L6P0');
  });

  it('marks rank >= RANK_CAP as off-scale, never as a number', () => {
    expect(offScaleLabel(RANK_CAP)).toBe('≥999');
    expect(offScaleLabel(RANK_CAP + 1)).toBe('≥999');
    expect(offScaleLabel(998)).toBeNull();
  });
});

describe('normalizeSelected', () => {
  // fixture(): layers [6,12,24] (layerIdx 0..2), promptLen 2 (pos 0..1).
  it('passes an in-bounds coord through unchanged', () => {
    const slice = fixture();
    expect(normalizeSelected({ layerIdx: 1, pos: 0 }, slice)).toEqual({ layerIdx: 1, pos: 0 });
    expect(normalizeSelected({ layerIdx: 2, pos: 1 }, slice)).toEqual({ layerIdx: 2, pos: 1 });
  });

  it('rejects an out-of-range layerIdx (stale selection / unclamped permalink)', () => {
    const slice = fixture();
    expect(normalizeSelected({ layerIdx: 7, pos: 0 }, slice)).toBeNull();
    expect(normalizeSelected({ layerIdx: -1, pos: 0 }, slice)).toBeNull();
  });

  it('rejects an out-of-range pos', () => {
    const slice = fixture();
    expect(normalizeSelected({ layerIdx: 0, pos: 9 }, slice)).toBeNull();
    expect(normalizeSelected({ layerIdx: 0, pos: -1 }, slice)).toBeNull();
  });

  it('rejects a non-integer coord', () => {
    const slice = fixture();
    expect(normalizeSelected({ layerIdx: 1.5, pos: 0 }, slice)).toBeNull();
    expect(normalizeSelected({ layerIdx: 0, pos: Number.NaN }, slice)).toBeNull();
  });

  it('returns null for a null selection', () => {
    const slice = fixture();
    expect(normalizeSelected(null, slice)).toBeNull();
  });
});

describe('rank axis', () => {
  it('puts rank 1 at the TOP (bump-chart convention)', () => {
    expect(rankToY(1, 100)).toBeLessThan(rankToY(10, 100));
    expect(rankToY(10, 100)).toBeLessThan(rankToY(999, 100));
  });

  it('is logarithmic, not linear', () => {
    const a = rankToY(1, 100) - rankToY(10, 100);
    const b = rankToY(10, 100) - rankToY(100, 100);
    expect(Math.abs(a - b)).toBeLessThan(1); // equal decades ⇒ equal pixels
  });
});
