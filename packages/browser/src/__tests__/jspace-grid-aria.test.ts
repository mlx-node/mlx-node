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
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { ArgmaxGridCanvas, type CellRef } from '../../demo/jspace/ArgmaxGridCanvas';
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

let root: Root | null = null;
let container: HTMLDivElement | null = null;

function mount(selected: CellRef | null): HTMLCanvasElement {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  flushSync(() => {
    root!.render(
      React.createElement(ArgmaxGridCanvas, {
        slice,
        colorByPinnedId: new Map<number, string>(),
        selected,
        onHover: () => {},
        onSelect: () => {},
        showWhitespace: false,
        ariaLabel: 'test grid',
      }),
    );
  });
  return container.querySelector('canvas')!;
}

afterEach(() => {
  if (root) flushSync(() => root!.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('ArgmaxGridCanvas ARIA row ownership', () => {
  it('owns a role=row containing a role=gridcell even with NO selection', () => {
    const canvas = mount(null);
    expect(canvas.getAttribute('role')).toBe('grid');

    const rowId = canvas.getAttribute('aria-owns');
    expect(rowId).toBeTruthy();
    const row = document.getElementById(rowId!);
    expect(row?.getAttribute('role')).toBe('row');
    expect(row!.querySelector('[role="gridcell"]')).not.toBeNull();

    // No selection ⇒ no active cell: the attribute is ABSENT, not a dangling ref.
    expect(canvas.getAttribute('aria-activedescendant')).toBeNull();
  });

  it('points aria-activedescendant at the owned gridcell when a cell is selected', () => {
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
  });

  it('treats an out-of-range selection (normalized to null) like no selection', () => {
    const canvas = mount({ layerIdx: 99, pos: 99 });
    // Row ownership is unconditional...
    expect(canvas.getAttribute('aria-owns')).toBeTruthy();
    // ...but the stale/out-of-range selection clamps to null ⇒ no active descendant.
    expect(canvas.getAttribute('aria-activedescendant')).toBeNull();
  });
});
