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

/** A `selected` from stale state or an unclamped permalink may be out of range
 *  for THIS slice. cellAt is `cells[layerIdx*promptLen+pos]`, so an unchecked
 *  coord aliases a different cell or throws. Return null for anything not an
 *  in-bounds integer cell. */
export function normalizeSelected(
  selected: CellRef | null,
  slice: LensSliceData,
): CellRef | null {
  if (!selected) return null;
  const { layerIdx, pos } = selected;
  if (!Number.isInteger(layerIdx) || !Number.isInteger(pos)) return null;
  if (layerIdx < 0 || layerIdx >= slice.layers.length) return null;
  if (pos < 0 || pos >= slice.promptLen) return null;
  return { layerIdx, pos };
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
  const rowId = React.useId();

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

  // A `selected` from stale parent state (slice shrank under it) or an unclamped
  // permalink may be out of range for THIS slice. Clamp ONCE and route every
  // dereference through `sel` so nothing indexes cellAt with a bad coord.
  // Memoized so it stays referentially stable for the draw effect's deps.
  const sel = React.useMemo(() => normalizeSelected(selected, slice), [selected, slice]);

  // Monotonic tick bumped whenever devicePixelRatio changes (browser zoom, or
  // dragging the window to a different-DPR monitor) so the draw effect resamples
  // the backing store even when the CSS width is unchanged. A same-value
  // `setWidth` is discarded by React, so `width` alone cannot trigger a redraw
  // on a pure DPR change — mirrors AttentionHeatmap.tsx's `resizeTick`.
  const [resizeTick, setResizeTick] = React.useState(0);
  const lastDprRef = React.useRef(0);

  // Measure the scroll container's visible width, and keep it in sync with
  // resizes / DPR changes. ResizeObserver preferred; window resize as fallback.
  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => {
      setWidth(el.clientWidth);
      // Zoom / monitor-move fires resize (and ResizeObserver) but may leave the
      // CSS width identical; force a redraw by bumping the tick on DPR change.
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

  // Announce the keyboard-selected cell to assistive tech.
  React.useEffect(() => {
    if (sel) {
      const c = slice.cellAt(sel.layerIdx, sel.pos);
      setAnnounce(
        `Layer ${slice.layers[sel.layerIdx]}, position ${sel.pos + 1} of ${promptLen}: ${c.topKTexts[0] ?? '∅'}`,
      );
    }
  }, [sel, slice, promptLen]);

  // The ONE grid cell represented in the DOM (the canvas paints the rest). Since
  // the grid unconditionally owns this proxy row, it must ALWAYS be a real,
  // indexed cell — never a blank unindexed placeholder. With no selection we
  // anchor it to the grid-entry cell (deepest layer = display row 1, position 0 =
  // column 1 — the SAME default onKeyDown seeds from), so AT reads a coherent
  // row 1 / column 1 rather than inferring a blank one. `proxy` is null ONLY for
  // a zero-axis (empty) slice, where the grid apparatus is dropped entirely.
  const hasCells = slice.layers.length > 0 && promptLen > 0;
  const proxy: CellRef | null = sel ?? (hasCells ? { layerIdx: slice.layers.length - 1, pos: 0 } : null);
  const proxyDesc = proxy
    ? `Layer ${slice.layers[proxy.layerIdx]}, position ${proxy.pos + 1} of ${promptLen}: ${
        slice.cellAt(proxy.layerIdx, proxy.pos).topKTexts[0] ?? '∅'
      }`
    : '';
  // The active-descendant IDREF must CHANGE VALUE on every cell move, or AT fires
  // no gridcell-focus event (WAI-ARIA ties the focus event to an
  // aria-activedescendant value change, not to mutating the referenced node's
  // text). Derive the proxy cell id from its coordinates and `key` the node by the
  // same coordinates so navigating A→B swaps the IDREF and recreates the cell.
  const cellKey = proxy ? `${proxy.layerIdx}:${proxy.pos}` : '';
  const cellId = proxy ? `${activeId}-${cellKey}` : undefined;

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
    ctx.fillStyle = CANVAS.inkMuted;
    ctx.textBaseline = 'middle';
    for (let r = 0; r < rowOrder.length; r++) {
      const layerIdx = rowOrder[r]!;
      ctx.fillText(`ℓ${slice.layers[layerIdx]}`, CELL_PAD, r * CELL_H + CELL_H / 2);
    }
  }, [slice, sel, colorByPinnedId, showWhitespace, scrollLeft, width, resizeTick]);

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
  // Track whether the PREVIOUS move was a Shift-scrub. A scrub clears the hover,
  // so the first non-Shift move afterwards must re-emit the hover even when it
  // lands on the same cell — the plain lastHoverKey short-circuit would strand
  // the tooltip on null (or a stale cell). Tracked separately from lastHoverKey.
  const wasScrubbing = React.useRef(false);

  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const ref = locate(e);
    if (!ref) {
      if (lastHoverKey.current !== null) {
        lastHoverKey.current = null;
        onHover(null);
      }
      wasScrubbing.current = e.shiftKey;
      return;
    }
    const key = `${ref.layerIdx}:${ref.pos}`;
    // Re-emit on the first non-Shift move after a scrub even on the same cell.
    const scrubJustEnded = wasScrubbing.current && !e.shiftKey;
    if (key === lastHoverKey.current && !e.shiftKey && !scrubJustEnded) {
      wasScrubbing.current = false;
      return;
    }
    lastHoverKey.current = key;
    if (e.shiftKey) {
      // Shift held scrubs the selection instead of only hovering. Clear the
      // hover so a consumer rendering `hovered ?? selected` follows the scrubbed
      // cell rather than showing the last hovered cell's tooltip.
      onHover(null);
      onSelect(ref);
    } else {
      onHover(ref);
    }
    wasScrubbing.current = e.shiftKey;
    const c = slice.cellAt(ref.layerIdx, ref.pos);
    setAnnounce(
      `Layer ${slice.layers[ref.layerIdx]}, position ${ref.pos + 1} of ${promptLen}: ${c.topKTexts[0] ?? '∅'}`,
    );
  }

  function onMouseLeave() {
    lastHoverKey.current = null;
    wasScrubbing.current = false;
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
    // Empty axes: nothing to navigate, and `slice.layers.length - 1` would seed
    // an out-of-range layerIdx of -1.
    if (rowOrder.length === 0 || promptLen === 0) return;
    // Seed from the clamped `sel`, never the raw (possibly stale) `selected`.
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

  return (
    <div ref={scrollRef} onScroll={onScroll} className="jspace-grid-scroll" style={{ overflowX: 'auto' }}>
      <div style={{ position: 'relative', width: contentWidth, height: contentHeight }}>
        <canvas
          ref={canvasRef}
          // A zero-axis (empty) slice is not a grid: drop role/counts/owned proxy
          // entirely rather than expose an empty grid with an unindexed row.
          role={hasCells ? 'grid' : undefined}
          aria-label={ariaLabel}
          tabIndex={0}
          // A <canvas> cannot own DOM children, so the off-screen active cell is a
          // SIBLING. A `grid` must own `row`s, and a `gridcell` requires a `row`
          // parent, so aria-owns UNCONDITIONALLY re-parents the off-screen ROW
          // (which wraps the gridcell): the grid always owns >=1 row and the row is
          // never orphaned, even with no selection. Only `aria-activedescendant` is
          // gated on `sel` — with no selection there is simply no active cell (the
          // proxy gridcell renders empty), which is absent, not a dangling ref.
          aria-owns={proxy ? rowId : undefined}
          aria-activedescendant={sel ? cellId : undefined}
          // The full grid is painted on the canvas; only ONE proxy row/cell lives
          // in the DOM. Publish the real dimensions so AT does not infer a 1×1
          // grid: rows = residual boundaries, cols = prompt positions. The proxy
          // row/cell carry their 1-based aria-rowindex / aria-colindex (below) so
          // the represented cell's true coordinates are exposed despite the virtualization.
          aria-rowcount={hasCells ? slice.layers.length : undefined}
          aria-colcount={hasCells ? promptLen : undefined}
          onMouseMove={onMouseMove}
          onMouseLeave={onMouseLeave}
          onClick={onClick}
          onKeyDown={onKeyDown}
          // Keyboard focus stays visible via `.jspace-grid-scroll canvas:focus-visible`
          // in styles.css — no bare `outline: none` here (it would win over that rule).
          style={{ position: 'sticky', left: 0, top: 0, display: 'block' }}
        />
        {/* Off-screen row wrapping the gridcell that `aria-activedescendant`
            targets — preserves the grid → row → gridcell ARIA hierarchy that a
            bare canvas-owned gridcell would violate. The proxy ALWAYS carries its
            true 1-based coordinates: rows render deepest-at-top (displayRowOrder is
            reversed), so the display row index is `layers.length - layerIdx`; the
            column index is `pos + 1`. Rendered only when a real cell exists (`proxy`
            is null for an empty slice); `aria-activedescendant` above still gates on
            an actual selection so no cell is announced "current" until one is chosen. */}
        {proxy && (
          <div id={rowId} role="row" aria-rowindex={slice.layers.length - proxy.layerIdx} style={SR_ONLY}>
            <div key={cellKey} id={cellId} role="gridcell" aria-colindex={proxy.pos + 1}>
              {proxyDesc}
            </div>
          </div>
        )}
        {/* Screen-reader live region — mirrors AttentionHeatmap's announce region. */}
        <div aria-live="polite" aria-atomic="true" style={SR_ONLY}>
          {announce}
        </div>
      </div>
    </div>
  );
}
