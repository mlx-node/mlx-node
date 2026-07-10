import * as React from 'react';

import { cleanupTokenText } from '../learn/inspector/TopKBars';
import { RANK_CAP } from '../jlens-core/colors';
import type { LensSliceData } from '../jlens-core/types';
import { CANVAS, CELL_H, CELL_W, GUTTER_W } from './canvas-theme';

/**
 * ArgmaxGridCanvas — the canvas twin of the lesson's DOM `ArgmaxGrid`.
 *
 * Why canvas, not DOM: the DOM grid attaches three inline listeners per cell.
 * At 24 layers × 128 positions that is 9216 handlers and 3072 nodes — enough to
 * jank scrolling. This version draws every cell onto ONE `<canvas>`, virtualizes
 * the columns (only the visible slice is painted), and hit-tests pointer/keyboard
 * events against a single element. Anthropic's own slice viewer does the same.
 *
 * Orientation contract (silently invertible — pinned by the golden test):
 *   `slice.layers` is ASCENDING, but the DEEPEST layer renders in the TOP row.
 *   `displayRowOrder` reverses the layer indices; arrow `↑` therefore means
 *   DEEPER (`layerIdx + 1`). Cells are always read through `slice.cellAt`.
 */

export type CellRef = { layerIdx: number; pos: number };

/** Deepest layer at the top. `slice.layers` is ASCENDING; display reverses it. */
export function displayRowOrder(slice: LensSliceData): number[] {
  return slice.layers.map((_, i) => i).reverse();
}

/** RANK_CAP is overloaded: native ceiling, out-of-range sentinel, AND a real 999. */
export function offScaleLabel(rank: number): string | null {
  return rank >= RANK_CAP ? `≥${RANK_CAP}` : null;
}

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

/** Longest prefix of `text` (plus an ellipsis) that fits within `maxWidth`. */
function fitText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (text === '' || ctx.measureText(text).width <= maxWidth) return text;
  const ell = '…';
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (ctx.measureText(text.slice(0, mid) + ell).width <= maxWidth) lo = mid;
    else hi = mid - 1;
  }
  return lo <= 0 ? ell : text.slice(0, lo) + ell;
}

export function ArgmaxGridCanvas({
  slice,
  colorByPinnedId,
  selected,
  onHover,
  onSelect,
  showWhitespace,
  ariaLabel,
}: {
  slice: LensSliceData;
  /** argmax id → pin accent, for the "answer surfaces" tint. */
  colorByPinnedId: Map<number, string>;
  selected: CellRef | null;
  onHover: (ref: CellRef | null) => void;
  onSelect: (ref: CellRef) => void;
  showWhitespace: boolean;
  ariaLabel: string;
}) {
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const activeId = React.useId();

  // Viewport width (for column virtualization) and horizontal scroll offset.
  // Initialized to 0 so nothing touches `window` at module scope or in a
  // useState initializer — measured in effects below (client-only discipline).
  const [width, setWidth] = React.useState(0);
  const [scrollLeft, setScrollLeft] = React.useState(0);
  const [announce, setAnnounce] = React.useState('');

  const rowOrder = React.useMemo(() => displayRowOrder(slice), [slice]);
  const promptLen = slice.promptLen;
  const contentWidth = GUTTER_W + promptLen * CELL_W;
  const contentHeight = rowOrder.length * CELL_H;

  // Measure the scroll container's visible width, and keep it in sync with
  // resizes / DPR changes. ResizeObserver preferred; window resize as fallback.
  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => setWidth(el.clientWidth);
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

  // Announce the keyboard-selected cell to assistive tech.
  React.useEffect(() => {
    if (selected) {
      const c = slice.cellAt(selected.layerIdx, selected.pos);
      setAnnounce(
        `Layer ${slice.layers[selected.layerIdx]}, position ${selected.pos + 1} of ${promptLen}: ${c.topKTexts[0] ?? '∅'}`,
      );
    }
  }, [selected, slice, promptLen]);

  const selectedDesc = selected
    ? `Layer ${slice.layers[selected.layerIdx]}, position ${selected.pos + 1} of ${promptLen}: ${
        slice.cellAt(selected.layerIdx, selected.pos).topKTexts[0] ?? '∅'
      }`
    : '';

  // Single redraw pass. Keyed exactly on the brief's dependency list — no rAF
  // loop, this is not an animation.
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

    ctx.font = FONT;
    ctx.textBaseline = 'middle';
    const maxTextW = CELL_W - CELL_PAD * 2;

    // Column virtualization: paint only the visible slice (+1 buffer each side).
    const firstCol = Math.max(0, Math.floor(scrollLeft / CELL_W) - 1);
    const lastCol = Math.min(promptLen - 1, Math.ceil((scrollLeft + (cssW - GUTTER_W)) / CELL_W) + 1);

    for (let pos = firstCol; pos <= lastCol; pos++) {
      const x = GUTTER_W + pos * CELL_W - scrollLeft;
      for (let r = 0; r < rowOrder.length; r++) {
        const layerIdx = rowOrder[r]!;
        const y = r * CELL_H;
        const cell = slice.cellAt(layerIdx, pos);
        const accent = colorByPinnedId.get(cell.argmaxId);
        if (accent) {
          ctx.globalAlpha = 0.18; // pin accent, ~18% alpha
          ctx.fillStyle = accent;
          ctx.fillRect(x, y, CELL_W, CELL_H);
          ctx.globalAlpha = 1;
        }
        const raw = cell.topKTexts[0] ?? '';
        const label = showWhitespace ? cleanupTokenText(raw) : raw.startsWith(' ') ? raw.slice(1) : raw;
        ctx.fillStyle = CANVAS.ink;
        ctx.fillText(fitText(ctx, label, maxTextW), x + CELL_PAD, y + CELL_H / 2);
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

    // Selection ring (drawn before the gutter so a scrolled-under cell is
    // occluded by the sticky label column, matching the DOM twin).
    if (selected) {
      const r = rowOrder.indexOf(selected.layerIdx);
      if (r >= 0) {
        const sx = GUTTER_W + selected.pos * CELL_W - scrollLeft;
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
    ctx.fillStyle = CANVAS.inkMuted;
    ctx.textBaseline = 'middle';
    for (let r = 0; r < rowOrder.length; r++) {
      const layerIdx = rowOrder[r]!;
      ctx.fillText(`ℓ${slice.layers[layerIdx]}`, CELL_PAD, r * CELL_H + CELL_H / 2);
    }
  }, [slice, selected, colorByPinnedId, showWhitespace, scrollLeft, width]);

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

  const lastHoverKey = React.useRef<string | null>(null);

  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const ref = locate(e);
    if (!ref) {
      if (lastHoverKey.current !== null) {
        lastHoverKey.current = null;
        onHover(null);
      }
      return;
    }
    const key = `${ref.layerIdx}:${ref.pos}`;
    if (key === lastHoverKey.current && !e.shiftKey) return;
    lastHoverKey.current = key;
    // Shift held scrubs the selection instead of only hovering.
    if (e.shiftKey) onSelect(ref);
    else onHover(ref);
    const c = slice.cellAt(ref.layerIdx, ref.pos);
    setAnnounce(
      `Layer ${slice.layers[ref.layerIdx]}, position ${ref.pos + 1} of ${promptLen}: ${c.topKTexts[0] ?? '∅'}`,
    );
  }

  function onMouseLeave() {
    lastHoverKey.current = null;
    onHover(null);
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
    const cur = selected ?? { layerIdx: slice.layers.length - 1, pos: 0 };
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

  return (
    <div ref={scrollRef} onScroll={onScroll} className="jspace-grid-scroll" style={{ overflowX: 'auto' }}>
      <div style={{ position: 'relative', width: contentWidth, height: contentHeight }}>
        <canvas
          ref={canvasRef}
          role="grid"
          aria-label={ariaLabel}
          tabIndex={0}
          aria-activedescendant={selected ? activeId : undefined}
          onMouseMove={onMouseMove}
          onMouseLeave={onMouseLeave}
          onClick={onClick}
          onKeyDown={onKeyDown}
          style={{ position: 'sticky', left: 0, top: 0, display: 'block', outline: 'none' }}
        />
        {/* Off-screen gridcell that `aria-activedescendant` targets. */}
        <div id={activeId} role="gridcell" style={SR_ONLY}>
          {selectedDesc}
        </div>
        {/* Screen-reader live region — mirrors AttentionHeatmap's announce region. */}
        <div aria-live="polite" aria-atomic="true" style={SR_ONLY}>
          {announce}
        </div>
      </div>
    </div>
  );
}
