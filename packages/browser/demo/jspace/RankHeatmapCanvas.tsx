import * as React from 'react';

import { RANK_CAP, rankToScore, viridis } from '../jlens-core/colors';
import type { LensSliceData } from '../jlens-core/types';
import { CANVAS, CELL_H, CELL_W, GUTTER_W } from './canvas-theme';
import { type CellRef, displayRowOrder, normalizeSelected } from './ArgmaxGridCanvas';

/**
 * RankHeatmapCanvas — the position × layer rank field for ONE pinned token.
 *
 * This is NOT learn/widgets/jlens/RankHeatmap.tsx: that one collapses position
 * to the final token and shows every pin as a column. This one keeps the full
 * position axis for a SINGLE pin (`pinnedIdx`), so you can watch WHERE and at
 * WHICH depth the answer becomes the top guess. Both ship; they answer
 * different questions.
 *
 * Orientation contract (shared with ArgmaxGridCanvas): `x = position`,
 * `y = layer` with the DEEPEST layer in the TOP row (`displayRowOrder`). Every
 * rank is read ONLY through `slice.rankAt(pinnedIdx, layerIdx, pos)` (which
 * returns RANK_CAP for any out-of-range lookup). Fill is `viridis(t)` where the
 * rank→score mapping is `rankToScore` (rank 1 → hot/yellow, cap → cold/purple);
 * a rank at/above the cap is greyed out with `CANVAS.unranked`, matching
 * Anthropic's grey-out for unranked cells.
 *
 * Canvas, not DOM, for the same reason as ArgmaxGridCanvas: at 24 layers × 128
 * positions a per-cell DOM would attach thousands of nodes/handlers. One canvas,
 * column virtualization, and a single hit-tested pointer/keyboard listener.
 */

const FONT = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
const CELL_PAD = 6;

const SR_ONLY: React.CSSProperties = {
  position: 'absolute',
  width: 1,
  height: 1,
  overflow: 'hidden',
  clip: 'rect(0,0,0,0)',
  whiteSpace: 'nowrap',
  left: 0,
  top: 0,
};

export function RankHeatmapCanvas({
  slice,
  pinnedIdx,
  selected,
  onSelect,
}: {
  slice: LensSliceData;
  /** Which pinned token's rank field to render (index into slice.pinned). */
  pinnedIdx: number;
  selected: CellRef | null;
  onSelect: (ref: CellRef) => void;
}) {
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  const [width, setWidth] = React.useState(0);
  const [scrollLeft, setScrollLeft] = React.useState(0);
  const [announce, setAnnounce] = React.useState('');

  const rowOrder = React.useMemo(() => displayRowOrder(slice), [slice]);
  const promptLen = slice.promptLen;
  const contentWidth = GUTTER_W + promptLen * CELL_W;
  const contentHeight = rowOrder.length * CELL_H;

  // Clamp `selected` ONCE (it may be stale / out of range for THIS slice) and
  // route every dereference through `sel`, exactly as the canvas grid does.
  const sel = React.useMemo(() => normalizeSelected(selected, slice), [selected, slice]);

  // Monotonic tick bumped on devicePixelRatio change so the draw effect resamples
  // the backing store even when the CSS width is unchanged (mirrors ArgmaxGridCanvas).
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

  // Announce the keyboard-selected cell and its rank to assistive tech.
  React.useEffect(() => {
    if (sel) {
      const rank = slice.rankAt(pinnedIdx, sel.layerIdx, sel.pos);
      const rankText = rank >= RANK_CAP ? `≥${RANK_CAP}` : String(rank);
      setAnnounce(
        `Layer ${slice.layers[sel.layerIdx]}, position ${sel.pos + 1} of ${promptLen}: rank ${rankText}`,
      );
    }
  }, [sel, slice, pinnedIdx, promptLen]);

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
        const rank = slice.rankAt(pinnedIdx, layerIdx, pos);
        // rank ≥ cap (or out of range → rankAt returns RANK_CAP) ⇒ grey-out.
        // Otherwise viridis, going through rankToScore (the log mapping lives
        // there once — the heatmap `t` and colors.ts stay a single source).
        ctx.fillStyle = rank >= RANK_CAP ? CANVAS.unranked : viridis(rankToScore(rank));
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
      ctx.fillText(`ℓ${slice.layers[layerIdx]}`, CELL_PAD, r * CELL_H + CELL_H / 2);
    }
  }, [slice, sel, pinnedIdx, scrollLeft, width, resizeTick, contentHeight, contentWidth, promptLen, rowOrder]);

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
    // a plain move is a no-op. Mirrors the canvas grid's Shift behaviour.
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
    const cur = sel ?? { layerIdx: slice.layers.length - 1, pos: 0 };
    let layerIdx = cur.layerIdx;
    let pos = cur.pos;
    if (e.key === 'ArrowLeft') pos = Math.max(0, pos - 1);
    else if (e.key === 'ArrowRight') pos = Math.min(promptLen - 1, pos + 1);
    // Rows are reversed (deepest at top), so ↑ moves DEEPER = layerIdx + 1.
    else if (e.key === 'ArrowUp') layerIdx = Math.min(slice.layers.length - 1, layerIdx + 1);
    else if (e.key === 'ArrowDown') layerIdx = Math.max(0, layerIdx - 1);
    onSelect({ layerIdx, pos });
    ensureColumnVisible(pos);
  }

  const hasCells = rowOrder.length > 0 && promptLen > 0;
  const pinText = slice.pinned[pinnedIdx]?.tokenText ?? '';

  return (
    <div ref={scrollRef} onScroll={onScroll} className="jspace-grid-scroll" style={{ overflowX: 'auto' }}>
      <div style={{ position: 'relative', width: contentWidth, height: contentHeight }}>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`Rank of ${pinText || 'the pinned token'} by position and layer (deepest layer on top; brighter = higher rank)`}
          tabIndex={hasCells ? 0 : undefined}
          onMouseMove={onMouseMove}
          onClick={onClick}
          onKeyDown={onKeyDown}
          style={{ position: 'sticky', left: 0, top: 0, display: 'block' }}
        />
        <div aria-live="polite" aria-atomic="true" style={SR_ONLY}>
          {announce}
        </div>
      </div>
    </div>
  );
}
