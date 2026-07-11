// DOM-level ARIA contract for the canvas argmax grid. The pure-function goldens
// live in jspace-grid-orientation.test.ts; this file mounts the real component
// in the browser test env (playwright chromium) and asserts the rendered
// accessibility structure a screen reader actually sees.
//
// The contract under test (a `role="grid"` must own a `role="row"` that owns the
// `role="gridcell"`):
//   - aria-owns is UNCONDITIONAL — the grid always owns its off-screen proxy row,
//     so the row is never orphaned and the grid always owns >=1 row, even with no
//     selection.
//   - aria-activedescendant is gated on a live, in-range selection — absent (not
//     dangling) when there is none, and pointing at the owned gridcell when there is.
import * as React from 'react';
import { flushSync } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { ArgmaxGridCanvas, type CellRef } from '../../demo/jspace/ArgmaxGridCanvas';
import { CELL_H, CELL_W, GUTTER_W } from '../../demo/jspace/canvas-theme';
import { buildLensSlice } from '../../demo/jlens-core/types';
import type { LensCell, LensReadoutRun } from '../inspector-types';

function makeRun(): LensReadoutRun {
  const layers = [2, 4];
  const promptLen = 2;
  const cells: LensCell[] = [];
  // layer-major, position-minor — matches LensReadoutRun.cells order.
  for (let li = 0; li < layers.length; li++) {
    for (let p = 0; p < promptLen; p++) {
      const id = 100 + li * 2 + p;
      cells.push({
        layer: layers[li]!,
        position: p,
        argmaxId: id,
        topKIds: [id],
        topKLogits: new Float32Array([1]),
        topKProbs: new Float32Array([0.5]),
        topKTexts: [`tok-${li}-${p}`],
      });
    }
  }
  return {
    promptLen,
    topK: 1,
    useJacobian: false,
    jacobianApplied: false,
    layers,
    tokens: [
      { id: 10, text: 'La' },
      { id: 11, text: ' saison' },
    ],
    cells,
    pinned: [],
  };
}

const slice = buildLensSlice(makeRun());

// A zero-axis slice: no layers, no positions. The grid apparatus must be dropped.
const emptySlice = buildLensSlice({
  promptLen: 0,
  topK: 1,
  useJacobian: false,
  jacobianApplied: false,
  layers: [],
  tokens: [],
  cells: [],
  pinned: [],
});

let root: Root | null = null;
let container: HTMLDivElement | null = null;

function renderSel(r: Root, selected: CellRef | null, sliceArg: typeof slice): void {
  flushSync(() => {
    r.render(
      React.createElement(ArgmaxGridCanvas, {
        slice: sliceArg,
        colorByPinnedId: new Map<number, string>(),
        selected,
        onHover: () => {},
        onSelect: () => {},
        showWhitespace: false,
        ariaLabel: 'test grid',
      }),
    );
  });
}

function mount(selected: CellRef | null, sliceArg: typeof slice = slice): HTMLCanvasElement {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  renderSel(root, selected, sliceArg);
  return container.querySelector('canvas')!;
}

// Re-render the SAME root (no remount) — simulates in-place keyboard navigation.
function rerender(selected: CellRef | null): HTMLCanvasElement {
  renderSel(root!, selected, slice);
  return container!.querySelector('canvas')!;
}

afterEach(() => {
  if (root) flushSync(() => root!.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('ArgmaxGridCanvas ARIA row ownership', () => {
  it('anchors the owned proxy to a real, indexed grid-entry cell with NO selection', () => {
    const canvas = mount(null);
    expect(canvas.getAttribute('role')).toBe('grid');
    // The full grid's dimensions are published even though only one proxy row
    // lives in the DOM — AT must not infer a 1×1 grid. Fixture is 2 layers × 2 pos.
    expect(canvas.getAttribute('aria-rowcount')).toBe('2');
    expect(canvas.getAttribute('aria-colcount')).toBe('2');

    const rowId = canvas.getAttribute('aria-owns');
    expect(rowId).toBeTruthy();
    const row = document.getElementById(rowId!);
    expect(row?.getAttribute('role')).toBe('row');
    const cell = row!.querySelector('[role="gridcell"]');
    expect(cell).not.toBeNull();

    // No selection ⇒ no ACTIVE cell (activedescendant absent), but the owned proxy
    // is still a real, indexed cell anchored to the grid-entry cell: deepest layer
    // (display row 1) / position 0 (column 1), with non-empty content.
    expect(canvas.getAttribute('aria-activedescendant')).toBeNull();
    expect(row!.getAttribute('aria-rowindex')).toBe('1');
    expect(cell!.getAttribute('aria-colindex')).toBe('1');
    expect(cell!.textContent).not.toBe('');
  });

  it('exposes the reversed 1-based row index and column index of the selected cell', () => {
    // Rows render deepest-at-top: layers=[2,4], so layerIdx 1 (layer 4, deepest)
    // is display row 1, and pos 1 is column 2 (1-based).
    const canvas = mount({ layerIdx: 1, pos: 1 });

    const row = document.getElementById(canvas.getAttribute('aria-owns')!)!;
    const activeId = canvas.getAttribute('aria-activedescendant');
    expect(activeId).toBeTruthy();

    const cell = document.getElementById(activeId!);
    expect(cell?.getAttribute('role')).toBe('gridcell');
    // The active gridcell is a genuine descendant of the owned row — the
    // grid→row→gridcell chain is intact, so the active-descendant ref is valid.
    expect(row.contains(cell)).toBe(true);
    expect(cell!.textContent).not.toBe('');
    expect(row.getAttribute('aria-rowindex')).toBe('1'); // layers.length - layerIdx = 2 - 1
    expect(cell!.getAttribute('aria-colindex')).toBe('2'); // pos + 1
  });

  it('maps the shallowest layer to the BOTTOM row index (reversed display order)', () => {
    // layerIdx 0 (layer 2, shallowest) is the bottom display row -> rowindex 2.
    const canvas = mount({ layerIdx: 0, pos: 0 });
    const row = document.getElementById(canvas.getAttribute('aria-owns')!)!;
    const cell = document.getElementById(canvas.getAttribute('aria-activedescendant')!)!;
    expect(row.getAttribute('aria-rowindex')).toBe('2'); // layers.length - 0
    expect(cell.getAttribute('aria-colindex')).toBe('1'); // pos 0 + 1
  });

  it('treats an out-of-range selection (normalized to null) like no selection', () => {
    const canvas = mount({ layerIdx: 99, pos: 99 });
    // Row ownership and the grid dimensions are unconditional...
    expect(canvas.getAttribute('aria-owns')).toBeTruthy();
    expect(canvas.getAttribute('aria-rowcount')).toBe('2');
    // ...the stale/out-of-range selection clamps to null ⇒ no active descendant...
    expect(canvas.getAttribute('aria-activedescendant')).toBeNull();
    // ...but the owned proxy still falls back to the indexed grid-entry cell.
    const row = document.getElementById(canvas.getAttribute('aria-owns')!)!;
    expect(row.getAttribute('aria-rowindex')).toBe('1');
    expect(row.querySelector('[role="gridcell"]')!.getAttribute('aria-colindex')).toBe('1');
  });

  it('swaps the active-descendant IDREF when navigating cell A → cell B on the SAME root', () => {
    // The active-descendant focus event fires on an IDREF VALUE change, not on the
    // referenced node's text mutating. Navigate without remounting and assert the
    // id actually changes and the old node is gone (recreated, not mutated).
    const canvas = mount({ layerIdx: 1, pos: 0 }); // cell A
    const idA = canvas.getAttribute('aria-activedescendant');
    expect(idA).toBeTruthy();
    expect(document.getElementById(idA!)?.getAttribute('role')).toBe('gridcell');

    rerender({ layerIdx: 0, pos: 1 }); // navigate to cell B, same root
    const idB = canvas.getAttribute('aria-activedescendant');
    expect(idB).toBeTruthy();
    expect(idB).not.toBe(idA); // IDREF changed ⇒ AT receives a gridcell-focus event
    expect(document.getElementById(idB!)?.getAttribute('role')).toBe('gridcell');
    // Old cell id is gone (keyed node recreated) — the reference is never stale.
    expect(document.getElementById(idA!)).toBeNull();
  });

  it('drops grid semantics entirely for a zero-axis (empty) slice', () => {
    const canvas = mount(null, emptySlice);
    // No grid, no counts, no owned proxy row — a blank canvas is not a 0×0 grid.
    expect(canvas.getAttribute('role')).toBeNull();
    expect(canvas.getAttribute('aria-owns')).toBeNull();
    expect(canvas.getAttribute('aria-rowcount')).toBeNull();
    expect(canvas.getAttribute('aria-colcount')).toBeNull();
    expect(canvas.getAttribute('aria-activedescendant')).toBeNull();
    expect(container!.querySelector('[role="row"]')).toBeNull();
  });
});

// F2: a keyboard/programmatic selection to a DIFFERENT cell than the one under
// the pointer strands the same-cell hover cache — the next intra-cell pointer
// move is short-circuited and the tooltip/charts keep showing the selected cell.
// The fix invalidates lastHoverKey on any `selected` change so hover re-emits.
describe('ArgmaxGridCanvas hover cache invalidation on selection (F2)', () => {
  it('re-emits onHover on the same cell after a keyboard/programmatic select', async () => {
    const onHover = vi.fn();
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    const el = (selected: CellRef | null) =>
      React.createElement(ArgmaxGridCanvas, {
        slice,
        colorByPinnedId: new Map<number, string>(),
        selected,
        onHover,
        onSelect: () => {},
        showWhitespace: false,
        ariaLabel: 'test grid',
      });

    flushSync(() => root!.render(el(null)));
    await new Promise((r) => setTimeout(r, 0)); // let measure/draw effects settle
    const canvas = container.querySelector('canvas')!;
    // Cell A = deepest display row (layerIdx 1 for layers [2,4]), position 0.
    const rect = canvas.getBoundingClientRect();
    const ax = rect.left + GUTTER_W + CELL_W / 2;
    const ay = rect.top + CELL_H / 2;
    const moveOverA = () =>
      canvas.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: ax, clientY: ay }));

    // (1) first move over A emits onHover(A).
    moveOverA();
    expect(onHover).toHaveBeenCalledWith({ layerIdx: 1, pos: 0 });

    // (2) keyboard nav selects a DIFFERENT cell B on the SAME root (pointer never
    // moves). The parent's activeCellRef is now B while the cursor is still over A.
    flushSync(() => root!.render(el({ layerIdx: 0, pos: 1 })));
    await new Promise((r) => setTimeout(r, 0)); // let the [selected] effect reset lastHoverKey

    // (3) a second intra-A move must RE-EMIT onHover(A) — pre-fix it was suppressed
    // by the stale same-cell cache; post-fix the [selected] effect cleared it.
    onHover.mockClear();
    moveOverA();
    expect(onHover).toHaveBeenCalledWith({ layerIdx: 1, pos: 0 });
  });

  it('re-emits onHover after a slice change even when selection is unchanged (mode toggle)', async () => {
    const onHover = vi.fn();
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    const el = (sliceArg: typeof slice, selected: CellRef | null) =>
      React.createElement(ArgmaxGridCanvas, {
        slice: sliceArg,
        colorByPinnedId: new Map<number, string>(),
        selected,
        onHover,
        onSelect: () => {},
        showWhitespace: false,
        ariaLabel: 'test grid',
      });

    // Two DISTINCT slice references, both with a cell at A (layerIdx 1, pos 0) so
    // `locate` returns the same key across the swap. S2 mirrors a Logit/Jacobian
    // mode toggle: `view.slice` swaps identity while `selected` stays null.
    const s1 = slice;
    const s2 = buildLensSlice(makeRun());
    expect(s2).not.toBe(s1); // distinct reference — the trigger under test

    flushSync(() => root!.render(el(s1, null)));
    await new Promise((r) => setTimeout(r, 0)); // let measure/draw effects settle
    const canvas = container.querySelector('canvas')!;
    // Cell A = deepest display row (layerIdx 1 for layers [2,4]), position 0.
    const rect = canvas.getBoundingClientRect();
    const ax = rect.left + GUTTER_W + CELL_W / 2;
    const ay = rect.top + CELL_H / 2;
    const moveOverA = () =>
      canvas.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: ax, clientY: ay }));

    // (1) first move over A emits onHover(A).
    moveOverA();
    expect(onHover).toHaveBeenCalledWith({ layerIdx: 1, pos: 0 });

    // (2) the slice swaps to S2 (mode toggle) on the SAME root; `selected` stays
    // null, so the [selected]-only effect would NOT reset the cache. The canvas
    // has no `key`, so it does not remount and lastHoverKey persists on A's key.
    onHover.mockClear();
    flushSync(() => root!.render(el(s2, null)));
    await new Promise((r) => setTimeout(r, 0)); // let the [selected, slice] effect reset lastHoverKey

    // (3) a second intra-A move must RE-EMIT onHover(A) — pre-fix (deps [selected])
    // the stale same-cell cache strands it; post-fix (deps [selected, slice]) the
    // slice change cleared the cache so hover re-fires.
    moveOverA();
    expect(onHover).toHaveBeenCalledWith({ layerIdx: 1, pos: 0 });
  });
});
