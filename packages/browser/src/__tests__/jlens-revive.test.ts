// Pure unit tests for the baked J-lens revive helper. These exercise the
// React-free `reviveRun` in demo/jlens-core/revive.ts and its handoff
// to `buildLensSlice` — no worker, no DOM, no React, no baked JSON on disk.
//
// The invariant under test (design decision D11): a serialized (baked) run
// round-trips back into a real `LensReadoutRun` with the flattened `number[]`
// typed arrays restored (`Float32Array` / `Int32Array`), and the SAME
// `buildLensSlice` view-model the LIVE worker path uses then indexes it
// correctly — so the jlens children render a baked frame unchanged.

import { describe, expect, test } from 'vitest';

import { reviveRun, type SerializedRun } from '../../demo/jlens-core/revive';
import { buildLensSlice } from '../../demo/jlens-core/types';

// A tiny 2-layer × 2-position run with one pinned token. Cells are stored
// layer-major then position-minor: flat(layerIdx, pos) = layerIdx * promptLen + pos.
// All float values are exactly representable in f32 so equality holds after the
// Float32Array round-trip.
function makeSerialized(): SerializedRun {
  return {
    promptLen: 2,
    topK: 3,
    useJacobian: true,
    jacobianApplied: true,
    layers: [6, 12],
    tokens: [
      { id: 100, text: 'La' },
      { id: 200, text: ' saison' },
    ],
    cells: [
      // flat 0 → (layer 6, pos 0)
      { layer: 6, position: 0, argmaxId: 11, topKIds: [11, 12, 13], topKLogits: [2.5, 1.25, 0.5], topKProbs: [0.5, 0.25, 0.125], topKTexts: ['a', 'b', 'c'] },
      // flat 1 → (layer 6, pos 1)
      { layer: 6, position: 1, argmaxId: 21, topKIds: [21, 22, 23], topKLogits: [3.5, 2.25, 1.5], topKProbs: [0.6, 0.2, 0.1], topKTexts: ['d', 'e', 'f'] },
      // flat 2 → (layer 12, pos 0)
      { layer: 12, position: 0, argmaxId: 31, topKIds: [31, 32, 33], topKLogits: [4.5, 3.25, 2.5], topKProbs: [0.7, 0.15, 0.05], topKTexts: ['g', 'h', 'i'] },
      // flat 3 → (layer 12, pos 1)
      { layer: 12, position: 1, argmaxId: 41, topKIds: [41, 42, 43], topKLogits: [5.5, 4.25, 3.5], topKProbs: [0.8, 0.1, 0.025], topKTexts: ['j', 'k', 'l'] },
    ],
    pinned: [{ tokenId: 200, tokenText: ' saison', ranks: [10, 20, 3, 1] }],
  };
}

describe('reviveRun', () => {
  test('restores typed arrays from number[] with matching length + values', () => {
    const run = reviveRun(makeSerialized());

    // cells[].topKLogits / topKProbs → Float32Array
    const c0 = run.cells[0]!;
    expect(c0.topKLogits).toBeInstanceOf(Float32Array);
    expect(c0.topKProbs).toBeInstanceOf(Float32Array);
    expect(c0.topKLogits.length).toBe(3);
    expect(c0.topKProbs.length).toBe(3);
    expect(Array.from(c0.topKLogits)).toEqual([2.5, 1.25, 0.5]);
    expect(Array.from(c0.topKProbs)).toEqual([0.5, 0.25, 0.125]);

    // pinned[].ranks → Int32Array
    const p0 = run.pinned[0]!;
    expect(p0.ranks).toBeInstanceOf(Int32Array);
    expect(p0.ranks.length).toBe(4);
    expect(Array.from(p0.ranks)).toEqual([10, 20, 3, 1]);
  });

  test('passes non-typed-array fields through unchanged', () => {
    const run = reviveRun(makeSerialized());
    expect(run.promptLen).toBe(2);
    expect(run.topK).toBe(3);
    expect(run.useJacobian).toBe(true);
    expect(run.jacobianApplied).toBe(true);
    expect(run.layers).toEqual([6, 12]);
    expect(run.tokens).toEqual([
      { id: 100, text: 'La' },
      { id: 200, text: ' saison' },
    ]);
    // topKIds / topKTexts stay plain arrays.
    expect(run.cells[2]!.topKIds).toEqual([31, 32, 33]);
    expect(run.cells[2]!.topKTexts).toEqual(['g', 'h', 'i']);
    expect(run.cells[2]!.argmaxId).toBe(31);
    expect(run.pinned[0]!.tokenId).toBe(200);
    expect(run.pinned[0]!.tokenText).toBe(' saison');
  });

  test('buildLensSlice(revived) indexes cells + ranks correctly', () => {
    const slice = buildLensSlice(reviveRun(makeSerialized()));

    expect(slice.layers).toEqual([6, 12]);
    expect(slice.promptLen).toBe(2);
    expect(slice.jacobianApplied).toBe(true);

    // cellAt(layerIdx, pos) reads cells[layerIdx * promptLen + pos].
    expect(slice.cellAt(0, 0).argmaxId).toBe(11); // flat 0
    expect(slice.cellAt(0, 1).argmaxId).toBe(21); // flat 1
    expect(slice.cellAt(1, 0).argmaxId).toBe(31); // flat 2
    expect(slice.cellAt(1, 1).argmaxId).toBe(41); // flat 3
    expect(slice.cellAt(1, 1).topKTexts[0]).toBe('j');

    // rankAt(pinnedIdx, layerIdx, pos) reads ranks[layerIdx * promptLen + pos].
    expect(slice.rankAt(0, 0, 0)).toBe(10); // flat 0
    expect(slice.rankAt(0, 0, 1)).toBe(20); // flat 1
    expect(slice.rankAt(0, 1, 0)).toBe(3); // flat 2
    expect(slice.rankAt(0, 1, 1)).toBe(1); // flat 3 — best rank at the deepest read
  });

  test('a logit (useJacobian:false) run round-trips with jacobianApplied false', () => {
    const s = makeSerialized();
    s.useJacobian = false;
    s.jacobianApplied = false;
    const slice = buildLensSlice(reviveRun(s));
    expect(slice.jacobianApplied).toBe(false);
  });
});
