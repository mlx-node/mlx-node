// DOM-level contract for MotorFlipStrip (the LIVE-ONLY commit-depth view, so it
// is never reachable from the cold starter gallery — this render test is how its
// output is actually SEEN without a 1.6 GB model run). Guards the codex-final
// motor-flip fixes on a synthetic CONTIGUOUS 24-layer live-shaped slice:
//   - a real lock renders role="img" with the localized `position N settles at ℓL`
//   - a no-lock (top guess never holds continuously to the final layer) renders
//     role="img" with the localized `noLock(position)` naming its 1-based position
//   - the zero-layer / one-layer guard returns null (no <cellAt(top=-1,…)> crash)
import * as React from 'react';
import { flushSync } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { MotorFlipStrip } from '../../demo/jspace/MotorFlipStrip';
import { JSPACE_COPY } from '../../demo/jspace/JSpaceApp';
import { buildLensSlice } from '../../demo/jlens-core/types';
import type { LensCell, LensReadoutRun } from '../inspector-types';

function cell(argmaxId: number): LensCell {
  return {
    layer: 0,
    position: 0,
    argmaxId,
    topKIds: [argmaxId],
    topKLogits: new Float32Array([1]),
    topKProbs: new Float32Array([1]),
    topKTexts: [`t${argmaxId}`],
  };
}

// Contiguous 1..24 (live shape). `argmaxAt(layerIdx, pos)` decides each cell so we
// can script the three commit-depth outcomes precisely.
function makeLiveRun(argmaxAt: (layerIdx: number, pos: number) => number, promptLen: number): LensReadoutRun {
  const layers = Array.from({ length: 24 }, (_, i) => i + 1);
  const cells: LensCell[] = [];
  for (let li = 0; li < layers.length; li++) {
    for (let p = 0; p < promptLen; p++) cells.push(cell(argmaxAt(li, p)));
  }
  return {
    promptLen,
    topK: 1,
    useJacobian: false,
    jacobianApplied: false,
    layers,
    tokens: Array.from({ length: promptLen }, (_, i) => ({ id: i, text: `w${i}` })),
    cells,
    pinned: [],
  };
}

let root: Root | null = null;
let container: HTMLDivElement | null = null;

function mount(run: LensReadoutRun): HTMLElement {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  flushSync(() => {
    root!.render(
      React.createElement(MotorFlipStrip, { slice: buildLensSlice(run), copy: JSPACE_COPY.en.motorFlip }),
    );
  });
  return container;
}

afterEach(() => {
  if (root) flushSync(() => root!.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('MotorFlipStrip — live-only commit-depth render', () => {
  it('renders one role="img" cell per position with the right lock / no-lock name', () => {
    // pos 0: argmax 100 at EVERY layer → locks at the shallowest, ℓ1.
    // pos 1: argmax 200 ONLY at the final layer (201 below) → no stable lock.
    // pos 2: argmax 300 from ℓ13 up (301 below) → locks at ℓ13.
    const run = makeLiveRun((li, p) => {
      const phys = li + 1; // physical layer number
      if (p === 0) return 100;
      if (p === 1) return phys === 24 ? 200 : 201;
      return phys >= 13 ? 300 : 301;
    }, 3);
    const el = mount(run);
    const cells = [...el.querySelectorAll('[role="img"]')];
    expect(cells.length).toBe(3); // exactly one cell per prompt position
    const labels = cells.map((c) => c.getAttribute('aria-label') ?? '');
    const copy = JSPACE_COPY.en.motorFlip;
    expect(labels).toContain(copy.cellLabel(1, 1)); // pos 1 → ℓ1 (earliest)
    expect(labels).toContain(copy.cellLabel(3, 13)); // pos 3 → ℓ13 (mid)
    expect(labels).toContain(copy.noLock(2)); // pos 2 → no stable lock, names #2
    // no-lock copy must NOT claim "only matches at the final layer" (relapse-safe).
    expect(copy.noLock(2)).not.toMatch(/only/i);
  });

  it('renders nothing for a <2-layer slice (zero/one-layer guard, no cellAt(-1) crash)', () => {
    const zero = mount({
      promptLen: 2,
      topK: 1,
      useJacobian: false,
      jacobianApplied: false,
      layers: [],
      tokens: [{ id: 0, text: 'a' }, { id: 1, text: 'b' }],
      cells: [],
      pinned: [],
    });
    expect(zero.querySelector('section')).toBeNull();
    if (root) flushSync(() => root!.unmount());
    container?.remove();
    root = null;
    container = null;

    const one = mount({
      promptLen: 1,
      topK: 1,
      useJacobian: false,
      jacobianApplied: false,
      layers: [24],
      tokens: [{ id: 0, text: 'a' }],
      cells: [cell(9)],
      pinned: [],
    });
    expect(one.querySelector('section')).toBeNull();
  });
});
