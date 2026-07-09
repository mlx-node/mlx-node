// types.ts — the single normalized view-model the jlens children consume.
//
// `LensSliceData` is a thin, index-friendly wrapper over one raw
// `LensReadoutRun` from the worker (`lensReadout` in demo/lib/lens-client.ts).
// The raw run stores its `cells` layer-major / position-minor
// (`cells[layerIdx * promptLen + pos]`) and each pinned token's `ranks` in the
// SAME flat (layer, position) order. Rather than open-code that arithmetic in
// every child, we build the slice once and hand out `cellAt` / `rankAt`
// accessors. `layerIdx` throughout is an index into `layers` (ascending, so 0
// is the shallowest boundary); the grid/heatmap render the rows reversed
// themselves.

import type { LensCell, LensPinned, LensReadoutRun, TokenInfo } from '../../../../src/inspector-types';
import { RANK_CAP } from './colors';

export type LensSliceData = {
  /** Residual boundaries actually read, ascending (e.g. [2,4,…,24]). */
  layers: number[];
  /** Decoded prompt tokens — the position axis. */
  tokens: TokenInfo[];
  /** Prompt length P (columns). */
  promptLen: number;
  /** Effective top-K per cell after backend clamping. */
  topK: number;
  /** Honest flag: `false` for the plain logit lens (always, in the browser). */
  jacobianApplied: boolean;
  /** One rank track per pinned token, index-aligned with the request's pins. */
  pinned: LensPinned[];
  /** The cell at (layers[layerIdx], position pos). */
  cellAt(layerIdx: number, pos: number): LensCell;
  /** 1-based full-vocab rank (capped at {@link RANK_CAP}) of pinned[pinnedIdx]
   *  at that cell; {@link RANK_CAP} for any out-of-range lookup. */
  rankAt(pinnedIdx: number, layerIdx: number, pos: number): number;
};

/** Build the {@link LensSliceData} view-model from a raw worker run. */
export function buildLensSlice(run: LensReadoutRun): LensSliceData {
  const { layers, promptLen, tokens, topK, jacobianApplied, cells, pinned } = run;
  const flat = (layerIdx: number, pos: number) => layerIdx * promptLen + pos;

  return {
    layers,
    tokens,
    promptLen,
    topK,
    jacobianApplied,
    pinned,
    cellAt(layerIdx, pos) {
      return cells[flat(layerIdx, pos)]!;
    },
    rankAt(pinnedIdx, layerIdx, pos) {
      const track = pinned[pinnedIdx];
      if (!track) return RANK_CAP;
      const i = flat(layerIdx, pos);
      if (i < 0 || i >= track.ranks.length) return RANK_CAP;
      return track.ranks[i]!;
    },
  };
}
