import * as React from 'react';

import type { LensSliceData } from '../jlens-core/types';
import { type CellRef, SR_ONLY } from './grid-a11y-shared';

/**
 * useCanvasGridA11y — the SHARED `grid → row → gridcell` accessible model for the
 * canvas widgets whose full contents are painted on a `<canvas>` (which cannot
 * own DOM children). It emits ONE off-screen proxy row/cell in the DOM and the
 * matching aria attributes for the canvas, so assistive tech sees a coherent grid
 * even though only a single cell is materialised.
 *
 * Two canvas widgets need this identical model — ArgmaxGridCanvas (token text)
 * and RankHeatmapCanvas (rank field) — and the model took several review rounds
 * to get right, so it lives here ONCE and both consume it. Only the per-cell
 * DESCRIPTION differs between them, so the caller passes `describeCell`.
 *
 * Invariants (do not regress — they are what makes AT fire a gridcell-focus
 * event and never dangle a reference):
 *   - `hasCells = slice.layers.length > 0 && slice.promptLen > 0`.
 *   - `proxy = sel ?? (hasCells ? { layerIdx: layers.length - 1, pos: 0 } : null)`
 *     — the live selection, else the grid-entry anchor (deepest layer = display
 *     row 1, position 0 = column 1), else null for an empty slice.
 *   - `aria-owns` is UNCONDITIONAL whenever a proxy exists (the grid always owns
 *     >=1 row); only `aria-activedescendant` gates on a real `sel`.
 *   - the active-descendant IDREF is DERIVED from the cell coordinates and the
 *     gridcell node is `key`ed by them, so navigating A→B swaps the IDREF value
 *     AND recreates the node — WAI-ARIA ties the focus event to the IDREF value
 *     changing, not to the referenced node's text mutating.
 */
export function useCanvasGridA11y({
  slice,
  sel,
  ariaLabel,
  describeCell,
}: {
  slice: LensSliceData;
  /** The ALREADY-NORMALIZED (clamped/in-range) selection, or null. */
  sel: CellRef | null;
  ariaLabel: string;
  /** Per-widget description of the represented cell (token text vs rank, …). */
  describeCell: (ref: CellRef) => string;
}): {
  /** Spread onto the `<canvas>` (role/label/counts/owns/activedescendant). */
  gridProps: {
    role: 'grid' | undefined;
    'aria-label': string;
    'aria-rowcount': number | undefined;
    'aria-colcount': number | undefined;
    'aria-owns': string | undefined;
    'aria-activedescendant': string | undefined;
  };
  /** The off-screen `role="row"` → `role="gridcell"` proxy node (or null). */
  proxyNode: React.ReactNode;
} {
  const activeId = React.useId();
  const rowId = React.useId();

  const promptLen = slice.promptLen;
  // A zero-axis (empty) slice is not a grid: no role/counts/owned proxy at all,
  // rather than an empty grid with an unindexed row.
  const hasCells = slice.layers.length > 0 && promptLen > 0;
  // The ONE grid cell represented in the DOM (the canvas paints the rest). Since
  // the grid unconditionally owns this proxy row, it must ALWAYS be a real,
  // indexed cell — with no selection we anchor it to the grid-entry cell (deepest
  // layer = display row 1, position 0 = column 1, the SAME default onKeyDown
  // seeds from). `proxy` is null ONLY for an empty slice.
  const proxy: CellRef | null = sel ?? (hasCells ? { layerIdx: slice.layers.length - 1, pos: 0 } : null);
  const proxyDesc = proxy ? describeCell(proxy) : '';
  // Derive the proxy cell id from its coordinates and `key` the node by the same
  // coordinates so navigating A→B swaps the IDREF and recreates the cell.
  const cellKey = proxy ? `${proxy.layerIdx}:${proxy.pos}` : '';
  const cellId = proxy ? `${activeId}-${cellKey}` : undefined;

  const gridProps = {
    // Zero-axis slice: drop role/counts/owned proxy entirely.
    role: hasCells ? ('grid' as const) : undefined,
    'aria-label': ariaLabel,
    // Publish the REAL dimensions so AT does not infer a 1×1 grid from the single
    // materialised proxy row: rows = residual boundaries, cols = prompt positions.
    'aria-rowcount': hasCells ? slice.layers.length : undefined,
    'aria-colcount': hasCells ? promptLen : undefined,
    // A <canvas> cannot own DOM children, so the off-screen row is a SIBLING that
    // aria-owns re-parents. Unconditional (whenever a proxy exists) so the grid
    // always owns >=1 row and the row is never orphaned. Only activedescendant is
    // gated on `sel` — with no selection there is simply no active cell.
    'aria-owns': proxy ? rowId : undefined,
    'aria-activedescendant': sel ? cellId : undefined,
  };

  // Off-screen row wrapping the gridcell that aria-activedescendant targets —
  // preserves the grid → row → gridcell hierarchy a bare canvas-owned gridcell
  // would violate. Rows render deepest-at-top (displayRowOrder is reversed), so
  // the display row index is `layers.length - layerIdx`; the column index is
  // `pos + 1`. Rendered only when a real cell exists.
  const proxyNode: React.ReactNode = proxy ? (
    <div id={rowId} role="row" aria-rowindex={slice.layers.length - proxy.layerIdx} style={SR_ONLY}>
      <div key={cellKey} id={cellId} role="gridcell" aria-colindex={proxy.pos + 1}>
        {proxyDesc}
      </div>
    </div>
  ) : null;

  return { gridProps, proxyNode };
}
