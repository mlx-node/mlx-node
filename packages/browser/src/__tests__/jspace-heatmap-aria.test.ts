// DOM-level ARIA contract for the canvas RANK HEATMAP. Mirrors
// jspace-grid-aria.test.ts: the heatmap is a focusable, arrow-navigable control,
// so it must expose the SAME `role="grid"` → `role="row"` → `role="gridcell"`
// model as ArgmaxGridCanvas — NOT `role="img"` (the adversarial finding). Both
// widgets share `useCanvasGridA11y`, so this file guards that the heatmap wired
// it up (real ranks in the proxy description, owned proxy row, activedescendant
// gated on a live selection).
import * as React from 'react';
import { flushSync } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { RankHeatmapCanvas } from '../../demo/jspace/RankHeatmapCanvas';
import type { CellRef } from '../../demo/jspace/ArgmaxGridCanvas';
import { RANK_CAP } from '../../demo/jlens-core/colors';
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
    // ONE pinned track so `slice.rankAt(0, …)` returns real ranks (flat
    // layer-major order over 2 layers × 2 positions). Rank 1 = best.
    pinned: [{ tokenId: 42, tokenText: ' Paris', ranks: Int32Array.from([1, 3, 7, 20]) }],
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
      React.createElement(RankHeatmapCanvas, {
        slice: sliceArg,
        pinnedIdx: 0,
        selected,
        onSelect: () => {},
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

describe('RankHeatmapCanvas ARIA row ownership', () => {
  it('is a role="grid" (not role="img") and publishes the real dimensions', () => {
    const canvas = mount(null);
    // The finding: a focusable, arrow-navigable canvas must NOT be role="img".
    expect(canvas.getAttribute('role')).toBe('grid');
    expect(canvas.getAttribute('role')).not.toBe('img');
    // Fixture is 2 layers × 2 positions.
    expect(canvas.getAttribute('aria-rowcount')).toBe('2');
    expect(canvas.getAttribute('aria-colcount')).toBe('2');
  });

  it('anchors the owned proxy to the indexed grid-entry cell with NO selection', () => {
    const canvas = mount(null);
    const rowId = canvas.getAttribute('aria-owns');
    expect(rowId).toBeTruthy();
    const row = document.getElementById(rowId!);
    expect(row?.getAttribute('role')).toBe('row');
    const cell = row!.querySelector('[role="gridcell"]');
    expect(cell).not.toBeNull();

    // No selection ⇒ no ACTIVE cell (activedescendant absent), but the proxy is
    // still a real, indexed cell: deepest layer (display row 1) / position 0
    // (column 1).
    expect(canvas.getAttribute('aria-activedescendant')).toBeNull();
    expect(row!.getAttribute('aria-rowindex')).toBe('1');
    expect(cell!.getAttribute('aria-colindex')).toBe('1');
    expect(cell!.textContent).not.toBe('');
  });

  it('points aria-activedescendant at an owned gridcell with real rank text', () => {
    // layers=[2,4]; layerIdx 1 (layer 4, deepest) = display row 1; pos 1 = col 2.
    // Flat rank index = layerIdx*promptLen + pos = 1*2 + 1 = 3 ⇒ rank 20.
    const canvas = mount({ layerIdx: 1, pos: 1 });
    const row = document.getElementById(canvas.getAttribute('aria-owns')!)!;
    const activeId = canvas.getAttribute('aria-activedescendant');
    expect(activeId).toBeTruthy();

    const cell = document.getElementById(activeId!);
    expect(cell?.getAttribute('role')).toBe('gridcell');
    // The active gridcell is a genuine descendant of the owned row.
    expect(row.contains(cell)).toBe(true);
    expect(row.getAttribute('aria-rowindex')).toBe('1'); // layers.length - layerIdx = 2 - 1
    expect(cell!.getAttribute('aria-colindex')).toBe('2'); // pos + 1
    // Non-empty, and a REAL rank (not a token label) — proves rankAt is wired.
    expect(cell!.textContent).not.toBe('');
    expect(cell!.textContent).toContain('rank 20');
  });

  it('swaps the active-descendant IDREF when navigating cell A → cell B on the SAME root', () => {
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

  it('renders off-scale ranks as "off scale", never as a bogus number', () => {
    // Rebuild with a capped rank at the selected cell (layerIdx 0, pos 0 ⇒ flat 0).
    const cappedRun = makeRun();
    cappedRun.pinned = [{ tokenId: 42, tokenText: ' Paris', ranks: Int32Array.from([RANK_CAP, 3, 7, 20]) }];
    const cappedSlice = buildLensSlice(cappedRun);
    const canvas = mount({ layerIdx: 0, pos: 0 }, cappedSlice);
    const cell = document.getElementById(canvas.getAttribute('aria-activedescendant')!)!;
    expect(cell.textContent).toContain('off scale');
    expect(cell.textContent).not.toContain('rank 999');
  });

  it('drops grid semantics entirely for a zero-axis (empty) slice', () => {
    const canvas = mount(null, emptySlice);
    expect(canvas.getAttribute('role')).toBeNull();
    expect(canvas.getAttribute('aria-owns')).toBeNull();
    expect(canvas.getAttribute('aria-rowcount')).toBeNull();
    expect(canvas.getAttribute('aria-colcount')).toBeNull();
    expect(canvas.getAttribute('aria-activedescendant')).toBeNull();
    expect(container!.querySelector('[role="row"]')).toBeNull();
  });
});
