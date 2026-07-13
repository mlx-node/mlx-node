import * as React from 'react';

import { divergingRamp } from '../jlens-core/colors';
import { jaccardTopK } from '../jlens-core/divergence';
import type { LensSliceData } from '../jlens-core/types';
import { CANVAS, CELL_H, CELL_W, GUTTER_W } from './canvas-theme';
import { displayRowOrder, normalizeSelected } from './ArgmaxGridCanvas';
import { SR_ONLY, type CellRef } from './grid-a11y-shared';
import { useCanvasGridA11y } from './useCanvasGridA11y';

/**
 * DivergenceGridCanvas — the position × layer field of DISAGREEMENT between the
 * two lenses, baked-only.
 *
 * A structural clone of RankHeatmapCanvas (same scroll / DPR resampling / column
 * virtualization / shared grid-a11y / `displayRowOrder` scaffolding), differing
 * ONLY in the per-cell scalar and its colour: instead of one pin's rank painted
 * `viridis(rankToScore(rank))`, each cell paints the Jaccard DISTANCE between the
 * logit lens's and the fitted Jacobian lens's top-10 id sets at that (layer,
 * position), through `divergingRamp` (blue = the lenses agree, red = they
 * disagree). Both slices are the SAME baked frame's two runs, so they share
 * `layers` / `promptLen` exactly (enforced by the gallery-pair alignment test);
 * dims + row order therefore come from `logitSlice` and every cell reads BOTH.
 *
 * Divergence is a FIELD, not a per-cell readout, so no token text is drawn on the
 * fills (mirrors RankHeatmapCanvas, which also labels only the ℓ gutter). This is
 * a starter-only surface — a live/custom run ships only one lens, so JSpaceApp
 * mounts this solely for a cold baked frame that carries both `.logit`+`.jacobian`.
 *
 * Canvas, not DOM, for the same reason as the sibling grids: at 24 layers × 128
 * positions a per-cell DOM would attach thousands of nodes/handlers. One canvas,
 * column virtualization, and a single hit-tested pointer/keyboard listener.
 */

const FONT = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
const CELL_PAD = 6;

/** The per-cell divergence scalar in [0,1]: top-10 Jaccard distance of the two lenses. */
function divergenceAt(logitSlice: LensSliceData, jacSlice: LensSliceData, layerIdx: number, pos: number): number {
  return jaccardTopK(logitSlice.cellAt(layerIdx, pos), jacSlice.cellAt(layerIdx, pos));
}

export function DivergenceGridCanvas({
  logitSlice,
  jacSlice,
  selected,
  onSelect,
  copy,
}: {
  /** The baked frame's plain logit-lens run. Provides the grid dims + row order. */
  logitSlice: LensSliceData;
  /** The SAME baked frame's fitted-Jacobian run — identical shape to logitSlice. */
  jacSlice: LensSliceData;
  selected: CellRef | null;
  onSelect: (ref: CellRef) => void;
  /** Localized aria strings (grid label + per-cell description template). The
   *  VISIBLE copy is already localized in JSpaceApp, so the aria-label / per-cell
   *  SR description / live announce must be too — they are built from these props,
   *  never string literals. `cellDesc` interpolates (ℓlayer, 1-based position,
   *  disagreement %). Glossary terms (Jacobian, logit lens, ℓ) stay English in zh. */
  copy: {
    ariaGrid: string;
    cellDesc: (layer: number, position: number, pct: number) => string;
  };
}) {
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  const [width, setWidth] = React.useState(0);
  const [scrollLeft, setScrollLeft] = React.useState(0);
  const [announce, setAnnounce] = React.useState('');

  const rowOrder = React.useMemo(() => displayRowOrder(logitSlice), [logitSlice]);
  const promptLen = logitSlice.promptLen;
  const contentWidth = GUTTER_W + promptLen * CELL_W;
  const contentHeight = rowOrder.length * CELL_H;

  // Clamp `selected` ONCE against THIS slice (it may be stale / out of range) and
  // route every dereference through `sel`, exactly as the sibling canvases do.
  const sel = React.useMemo(() => normalizeSelected(selected, logitSlice), [selected, logitSlice]);

  // SHARED grid → row → gridcell accessible model (identical to the sibling
  // canvases): a focusable, arrow-navigable control is a real `grid`, NOT
  // `role="img"`. Only the per-cell description differs — here it is the top-10
  // disagreement percentage between the two lenses at that (layer, position).
  const { gridProps, proxyNode } = useCanvasGridA11y({
    slice: logitSlice,
    sel,
    ariaLabel: copy.ariaGrid,
    describeCell: (ref) => {
      const d = divergenceAt(logitSlice, jacSlice, ref.layerIdx, ref.pos);
      return copy.cellDesc(logitSlice.layers[ref.layerIdx]!, ref.pos + 1, Math.round(d * 100));
    },
  });

  // Monotonic tick bumped on devicePixelRatio change so the draw effect resamples
  // the backing store even when the CSS width is unchanged (mirrors the siblings).
  const [resizeTick, setResizeTick] = React.useState(0);
  const lastDprRef = React.useRef(0);

  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => {
      setWidth(el.clientWidth);
      const dpr = window.devicePixelRatio || 1;
      if (dpr !== lastDprRef.current) {
        lastDprRef.current = dpr;
        setResizeTick((n) => n + 1);
      }
    };
    measure();
    let ro: ResizeObserver | undefined;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(measure);
      ro.observe(el);
    }
    window.addEventListener('resize', measure);
    return () => {
      ro?.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, []);

  // Announce the keyboard-selected cell and its divergence to assistive tech.
  React.useEffect(() => {
    if (sel) {
      const d = divergenceAt(logitSlice, jacSlice, sel.layerIdx, sel.pos);
      setAnnounce(copy.cellDesc(logitSlice.layers[sel.layerIdx]!, sel.pos + 1, Math.round(d * 100)));
    }
  }, [sel, logitSlice, jacSlice, copy]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssH = contentHeight;
    const cssW = width > 0 ? Math.min(width, contentWidth) : 0;
    canvas.width = Math.max(1, Math.round(cssW * dpr));
    canvas.height = Math.max(1, Math.round(cssH * dpr));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = CANVAS.bg;
    ctx.fillRect(0, 0, cssW, cssH);
    if (cssW <= 0) return;

    // Column virtualization: paint only the visible slice (+1 buffer each side).
    const firstCol = Math.max(0, Math.floor(scrollLeft / CELL_W) - 1);
    const lastCol = Math.min(promptLen - 1, Math.ceil((scrollLeft + (cssW - GUTTER_W)) / CELL_W) + 1);

    for (let pos = firstCol; pos <= lastCol; pos++) {
      const x = GUTTER_W + pos * CELL_W - scrollLeft;
      for (let r = 0; r < rowOrder.length; r++) {
        const layerIdx = rowOrder[r]!;
        const y = r * CELL_H;
        // Per-cell fill = the diverging ramp of the two lenses' top-10 Jaccard
        // distance. blue = agree, near-white = half-overlap, red = disjoint. No
        // token text: divergence is a field, not a per-cell readout.
        const d = divergenceAt(logitSlice, jacSlice, layerIdx, pos);
        ctx.fillStyle = divergingRamp(d);
        ctx.fillRect(x, y, CELL_W, CELL_H);
      }
    }

    // Grid lines (subtle).
    ctx.strokeStyle = CANVAS.grid;
    ctx.lineWidth = 1;
    for (let r = 0; r <= rowOrder.length; r++) {
      const y = Math.round(r * CELL_H) + 0.5;
      ctx.beginPath();
      ctx.moveTo(GUTTER_W, y);
      ctx.lineTo(cssW, y);
      ctx.stroke();
    }
    for (let pos = firstCol; pos <= lastCol + 1; pos++) {
      const vx = Math.round(GUTTER_W + pos * CELL_W - scrollLeft) + 0.5;
      if (vx < GUTTER_W) continue;
      ctx.beginPath();
      ctx.moveTo(vx, 0);
      ctx.lineTo(vx, cssH);
      ctx.stroke();
    }

    // Selection ring (before the gutter, so a scrolled-under cell is occluded).
    if (sel) {
      const r = rowOrder.indexOf(sel.layerIdx);
      if (r >= 0) {
        const sx = GUTTER_W + sel.pos * CELL_W - scrollLeft;
        const sy = r * CELL_H;
        ctx.strokeStyle = CANVAS.selectionRing;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx + 1, sy + 1, CELL_W - 2, CELL_H - 2);
      }
    }

    // Sticky layer-label gutter — painted last, on top of scrolled columns.
    ctx.fillStyle = CANVAS.bg;
    ctx.fillRect(0, 0, GUTTER_W, cssH);
    ctx.strokeStyle = CANVAS.grid;
    ctx.beginPath();
    ctx.moveTo(GUTTER_W + 0.5, 0);
    ctx.lineTo(GUTTER_W + 0.5, cssH);
    ctx.stroke();
    ctx.font = FONT;
    ctx.fillStyle = CANVAS.inkMuted;
    ctx.textBaseline = 'middle';
    for (let r = 0; r < rowOrder.length; r++) {
      const layerIdx = rowOrder[r]!;
      ctx.fillText(`ℓ${logitSlice.layers[layerIdx]}`, CELL_PAD, r * CELL_H + CELL_H / 2);
    }
  }, [logitSlice, jacSlice, sel, scrollLeft, width, resizeTick, contentHeight, contentWidth, promptLen, rowOrder]);

  function onScroll() {
    const el = scrollRef.current;
    if (el) setScrollLeft(el.scrollLeft);
  }

  /** Map a pointer event to the cell under it, or null (gutter / out of range). */
  function locate(e: React.MouseEvent<HTMLCanvasElement>): CellRef | null {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const offsetY = e.clientY - rect.top;
    if (offsetX < GUTTER_W) return null; // over the sticky label column
    const sl = scrollRef.current ? scrollRef.current.scrollLeft : scrollLeft;
    const pos = Math.floor((offsetX + sl - GUTTER_W) / CELL_W);
    const row = Math.floor(offsetY / CELL_H);
    if (pos < 0 || pos >= promptLen) return null;
    if (row < 0 || row >= rowOrder.length) return null;
    const layerIdx = rowOrder[row];
    if (layerIdx === undefined) return null;
    return { layerIdx, pos };
  }

  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    // Shift-scrub selects as the pointer moves (there is no hover callback here);
    // a plain move is a no-op. Mirrors the sibling canvases' Shift behaviour.
    if (!e.shiftKey) return;
    const ref = locate(e);
    if (ref) onSelect(ref);
  }

  function onClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const ref = locate(e);
    if (ref) onSelect(ref);
  }

  function ensureColumnVisible(pos: number) {
    const el = scrollRef.current;
    if (!el) return;
    const cellLeft = GUTTER_W + pos * CELL_W;
    const cellRight = cellLeft + CELL_W;
    const viewLeft = el.scrollLeft + GUTTER_W;
    const viewRight = el.scrollLeft + el.clientWidth;
    if (cellLeft < viewLeft) el.scrollLeft = cellLeft - GUTTER_W;
    else if (cellRight > viewRight) el.scrollLeft = cellRight - el.clientWidth;
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLCanvasElement>) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight' && e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    if (rowOrder.length === 0 || promptLen === 0) return;
    const cur = sel ?? { layerIdx: logitSlice.layers.length - 1, pos: 0 };
    let layerIdx = cur.layerIdx;
    let pos = cur.pos;
    if (e.key === 'ArrowLeft') pos = Math.max(0, pos - 1);
    else if (e.key === 'ArrowRight') pos = Math.min(promptLen - 1, pos + 1);
    // Rows are reversed (deepest at top), so ↑ moves DEEPER = layerIdx + 1.
    else if (e.key === 'ArrowUp') layerIdx = Math.min(logitSlice.layers.length - 1, layerIdx + 1);
    else if (e.key === 'ArrowDown') layerIdx = Math.max(0, layerIdx - 1);
    onSelect({ layerIdx, pos });
    ensureColumnVisible(pos);
  }

  const hasCells = rowOrder.length > 0 && promptLen > 0;

  return (
    <div ref={scrollRef} onScroll={onScroll} className="jspace-grid-scroll" style={{ overflowX: 'auto' }}>
      <div style={{ position: 'relative', width: contentWidth, height: contentHeight }}>
        <canvas
          ref={canvasRef}
          // A focusable, arrow-navigable control, so it is a real `grid`
          // (role/label/counts/owns/activedescendant from the SHARED hook), NOT
          // `role="img"`. A zero-axis slice yields no grid role/owns.
          {...gridProps}
          tabIndex={hasCells ? 0 : undefined}
          onMouseMove={onMouseMove}
          onClick={onClick}
          onKeyDown={onKeyDown}
          style={{ position: 'sticky', left: 0, top: 0, display: 'block' }}
        />
        {/* Off-screen grid → row → gridcell proxy owned by the canvas via aria-owns. */}
        {proxyNode}
        <div aria-live="polite" aria-atomic="true" style={SR_ONLY}>
          {announce}
        </div>
      </div>
    </div>
  );
}
