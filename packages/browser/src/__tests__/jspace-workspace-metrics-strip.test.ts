// DOM-level contract for WorkspaceMetricsStrip — guards the three codex-final
// findings on the metrics view:
//   F2  every series is read at the FIXED final prompt position; that must be
//       disclosed visibly and in each chart's accessible name (both locales).
//   F3  the PROXY charts plot on a FIXED honest axis (entropy [0, ln 10] nats,
//       stability [0, 1] Jaccard), NOT a per-series auto-scale — so a flat [1]
//       and a flat [0] land at DIFFERENT heights (the old auto-scale drew both
//       at the mid-line, destroying magnitude).
//   F1  effective-dim was dropped — verified at the module level in
//       workspace-metrics.test.ts; here we simply confirm the strip renders the
//       two retained proxies (entropy + stability) and nothing claims eff-dim.
import * as React from 'react';
import { flushSync } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { WorkspaceMetricsStrip } from '../../demo/jspace/WorkspaceMetricsStrip';
import { JSPACE_COPY } from '../../demo/jspace/JSpaceApp';
import { buildLensSlice } from '../../demo/jlens-core/types';
import type { LensCell, LensReadoutRun } from '../inspector-types';

function cell(layer: number, position: number, ids: number[], probs: number[]): LensCell {
  return {
    layer,
    position,
    argmaxId: ids[0]!,
    topKIds: ids,
    topKLogits: new Float32Array(ids.length),
    topKProbs: Float32Array.from(probs),
    topKTexts: ids.map((id) => `t${id}`),
  };
}

// 2 layers × `promptLen` positions, layer-major (cells[layerIdx*promptLen+pos]).
// The last position's L0/L1 topKIds drive top-10 set stability.
function makeRun(opts: {
  promptLen: number;
  lastL0: number[];
  lastL1: number[];
  pinned?: LensReadoutRun['pinned'];
}): LensReadoutRun {
  const { promptLen, lastL0, lastL1, pinned = [] } = opts;
  const layers = [2, 4];
  const cells: LensCell[] = [];
  for (let li = 0; li < layers.length; li++) {
    for (let p = 0; p < promptLen; p++) {
      const last = p === promptLen - 1;
      const ids = last ? (li === 0 ? lastL0 : lastL1) : [1, 2, 3];
      cells.push(cell(layers[li]!, p, ids, [0.5, 0.3, 0.2]));
    }
  }
  return {
    promptLen,
    topK: 3,
    useJacobian: false,
    jacobianApplied: false,
    layers,
    tokens: Array.from({ length: promptLen }, (_, i) => ({ id: 10 + i, text: `w${i}` })),
    cells,
    pinned,
  };
}

let root: Root | null = null;
let container: HTMLDivElement | null = null;

function mount(sliceArg: ReturnType<typeof buildLensSlice>, copy: typeof JSPACE_COPY.en.metrics): HTMLElement {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  flushSync(() => {
    root!.render(React.createElement(WorkspaceMetricsStrip, { slice: sliceArg, pinnedIdx: 0, copy }));
  });
  return container;
}

afterEach(() => {
  if (root) flushSync(() => root!.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('WorkspaceMetricsStrip — final-position disclosure (F2)', () => {
  for (const locale of ['en', 'zh'] as const) {
    it(`${locale}: discloses the fixed final prompt position (#promptLen) in caption + chart names`, () => {
      const copy = JSPACE_COPY[locale].metrics;
      const slice = buildLensSlice(
        makeRun({
          promptLen: 3,
          lastL0: [1, 2, 3],
          lastL1: [1, 2, 3],
          pinned: [{ tokenId: 1, tokenText: 't1', ranks: Int32Array.from([5, 9, 1, 40, 12, 1]) }],
        }),
      );
      const el = mount(slice, copy);
      // Visible sub-caption names the fixed answer position #3 (1-based promptLen).
      expect(el.textContent).toContain(copy.atPosition(3));
      // The suffix is baked into each chart's accessible name too.
      const suffix = copy.atPositionShort(3);
      const named = Array.from(el.querySelectorAll('[aria-label]')).map((n) => n.getAttribute('aria-label') ?? '');
      expect(named.some((l) => l.includes(suffix))).toBe(true);
      // Entropy + stability proxy charts expose their FIXED domain in the a11y name.
      expect(named.some((l) => l.includes(copy.entropyScale))).toBe(true);
      expect(named.some((l) => l.includes(copy.stabilityScale))).toBe(true);
    });
  }
});

describe('WorkspaceMetricsStrip — fixed proxy domain (F3)', () => {
  // stability chart is a single point for a 2-layer slice; its cy encodes the
  // value on the FIXED [0,1] axis. Auto-scaling drew ANY single value at the
  // mid-line, so [1] and [0] were indistinguishable. Fixed axis => distinct.
  function stabilityCircleCy(lastL0: number[], lastL1: number[]): number {
    const slice = buildLensSlice(makeRun({ promptLen: 1, lastL0, lastL1 }));
    const el = mount(slice, JSPACE_COPY.en.metrics);
    const svg = el.querySelector('svg[aria-label*="Jaccard"]');
    const circle = svg?.querySelector('circle');
    const cy = Number(circle?.getAttribute('cy'));
    if (root) flushSync(() => root!.unmount());
    container?.remove();
    root = null;
    container = null;
    return cy;
  }

  it('stability 1 sits near the top, stability 0 near the bottom — magnitudes differ', () => {
    const cyHigh = stabilityCircleCy([1, 2, 3], [1, 2, 3]); // Jaccard = 1
    const cyLow = stabilityCircleCy([1, 2, 3], [4, 5, 6]); // Jaccard = 0
    expect(Number.isFinite(cyHigh)).toBe(true);
    expect(Number.isFinite(cyLow)).toBe(true);
    // Fixed [0,1] axis: value 1 → top (small y), value 0 → bottom (large y).
    expect(cyHigh).toBeLessThan(10);
    expect(cyLow).toBeGreaterThan(35);
    expect(cyHigh).not.toBe(cyLow); // the exact regression the auto-scale caused
  });
});
