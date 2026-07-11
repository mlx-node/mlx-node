// Full-app render tests for JSpaceApp (mounts the real component with the two
// context providers, and the worker-backed client modules mocked so no MLX
// worker is needed). Covers three re-gate fixes:
//   - F2 (jacobian-layers): layersFor is 1..24 for BOTH modes.
//   - F3 (stale permalink): a permalink-restored selection is NOT resurrected
//     after the user diverges (edits the prompt) and then runs; the intended
//     custom-permalink→Run restore still works.
//   - F5 (rank-vs-position chart): BOTH rank charts (by-depth AND by-position)
//     render alongside a pinned token.

import * as React from 'react';
import { flushSync } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import JSpaceApp, { layersFor } from '../../demo/jspace/JSpaceApp';
import { ModelLoaderProvider, type ModelLoaderContextValue } from '../../demo/providers/model-loader';
import { FreeChatProvider, type FreeChatContextValue } from '../../demo/providers/free-chat';
import type { LensCell, LensReadoutRun } from '../inspector-types';

// Worker-backed clients — replaced so the run pipeline is fully synchronous and
// deterministic without an MLX worker. `ctl.run` is the LensReadoutRun the mocked
// lensReadout resolves with; each test sets it before mounting.
const ctl = vi.hoisted(() => ({ run: null as unknown as LensReadoutRun }));

vi.mock('../../demo/lib/lens-client', () => ({
  lensReadout: async () => ctl.run,
  loadLensPack: async () => ({ loaded: 23, alreadyLoaded: true }),
}));

vi.mock('../../demo/lib/tokenizer-client', () => ({
  tokenize: async (_worker: unknown, text: string) =>
    Array.from({ length: Math.max(1, ctl.run?.promptLen ?? String(text).length) }, (_, i) => ({
      id: i + 1,
      text: String(text)[i] ?? 'x',
    })),
}));

// -- fixtures ---------------------------------------------------------------
function makeRun(nLayers: number, promptLen: number): LensReadoutRun {
  const layers = Array.from({ length: nLayers }, (_, i) => i + 1);
  const cells: LensCell[] = [];
  for (let li = 0; li < nLayers; li++) {
    for (let p = 0; p < promptLen; p++) {
      const id = 100 + li * promptLen + p;
      cells.push({
        layer: layers[li]!,
        position: p,
        argmaxId: id,
        topKIds: [id],
        topKLogits: new Float32Array([1]),
        topKProbs: new Float32Array([0.5]),
        topKTexts: [`t${li}-${p}`],
      });
    }
  }
  return {
    promptLen,
    topK: 1,
    useJacobian: false,
    jacobianApplied: false,
    layers,
    tokens: Array.from({ length: promptLen }, (_, p) => ({ id: p + 1, text: `w${p}` })),
    cells,
    pinned: [],
  };
}

function makeModelCtx(): ModelLoaderContextValue {
  return {
    status: 'ready',
    loadingText: '',
    loadingProgress: null,
    modelLine: null,
    errorBanner: null,
    hostedModelAvailable: true,
    deviceReady: true,
    loadKickoff: 0,
    kickoffLoad: () => {},
    kickoffDeviceOnly: () => {},
    clearLoadError: () => {},
    resetForModelLoad: () => {},
  };
}

function makeChatCtx(): FreeChatContextValue {
  return {
    temperature: 0,
    maxTokens: 0,
    toolsEnabled: false,
    reasoningEffort: 'off',
    generating: false,
    sendDisabled: false,
    setTemperature: () => {},
    setMaxTokens: () => {},
    setToolsEnabled: () => {},
    cycleReasoning: () => {},
    sendMessage: () => {},
    resetChat: () => {},
    promptRef: { current: null },
    chatRef: { current: null },
    sendRef: { current: null },
    imageButtonRef: { current: null },
    imageInputRef: { current: null },
    inspectedPrompt: null,
    setInspectedPrompt: () => {},
    mlxWorkerRef: { current: {} as unknown as Worker },
    inspectorAbortRef: { current: new AbortController() },
  };
}

// -- harness ----------------------------------------------------------------
let root: Root | null = null;
let container: HTMLDivElement | null = null;

function mountApp(hash: string): HTMLDivElement {
  window.location.hash = hash;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  flushSync(() => {
    root!.render(
      React.createElement(ModelLoaderProvider, {
        value: makeModelCtx(),
        children: React.createElement(FreeChatProvider, {
          value: makeChatCtx(),
          children: React.createElement(JSpaceApp),
        }),
      }),
    );
  });
  return container;
}

const tick = () => new Promise((r) => setTimeout(r, 0));

async function waitFor(pred: () => boolean, ms = 4000): Promise<void> {
  const start = Date.now();
  while (!pred()) {
    if (Date.now() - start > ms) throw new Error('waitFor timeout');
    await new Promise((r) => setTimeout(r, 10));
  }
}

/** The argmax grid canvas (its aria-label is unique among the app's canvases). */
const gridCanvas = () =>
  container!.querySelector('canvas[aria-label^="Argmax grid"]') as HTMLCanvasElement | null;

const activeDescendant = () => gridCanvas()?.getAttribute('aria-activedescendant') ?? null;

/** The LIVE grid specifically — the fake run is 4 positions wide, the starter is
 *  9, so the aria-label disambiguates the two without depending on cell content
 *  (grid cells are painted on the canvas, never in the DOM text). */
const liveGridCanvas = () => {
  const c = gridCanvas();
  return c && (c.getAttribute('aria-label') ?? '').includes('by 4 prompt positions') ? c : null;
};

function setPromptText(value: string): void {
  const ta = container!.querySelector('#jspace-prompt') as HTMLTextAreaElement;
  // Bypass React's value tracker (the prototype setter) so the input event is
  // seen as a real change on the controlled textarea.
  // oxlint-disable-next-line typescript/unbound-method
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!;
  setter.call(ta, value);
  ta.dispatchEvent(new Event('input', { bubbles: true }));
}

function clickRun(): void {
  const btn = Array.from(container!.querySelectorAll('button')).find(
    (b) => (b.textContent ?? '').trim() === 'Run',
  ) as HTMLButtonElement;
  if (!btn) throw new Error('Run button not found');
  btn.click();
}

afterEach(() => {
  if (root) flushSync(() => root!.unmount());
  container?.remove();
  root = null;
  container = null;
  ctl.run = null as unknown as LensReadoutRun;
  window.location.hash = '';
});

// -- F2: layersFor is 1..24 for both modes ----------------------------------
describe('layersFor — both lenses read the full 1..24 boundary stack (F2)', () => {
  it('returns identical 1..24 lists for jacobian and logit', () => {
    const expected = Array.from({ length: 24 }, (_, i) => i + 1);
    expect(layersFor('jacobian')).toEqual(layersFor('logit'));
    expect(layersFor('logit')).toEqual(expected);
    expect(layersFor('jacobian')).toHaveLength(24);
  });
});

// -- F3: a stale permalink selection must not resurrect after divergence -----
describe('permalink selection restore (F3)', () => {
  it('does NOT resurrect a starter selection after the prompt is edited then run', async () => {
    ctl.run = makeRun(4, 4); // 4 layers × 4 positions → (3,2) is in-bounds
    mountApp('#p=&mode=l&sel=3,2'); // empty prompt → starter grid, selection (3,2)
    await tick();

    // The starter grid renders the restored selection immediately.
    await waitFor(() => gridCanvas() !== null);
    expect(activeDescendant()).toBeTruthy();

    // User diverges: edit the prompt to a custom (non-cold) value → skeleton.
    setPromptText('hello world foo');
    await tick();

    // Run the custom prompt; the first live 'done' would, pre-fix, re-apply the
    // stale (3,2). Post-fix the textarea-edit cleared the pending restore.
    clickRun();
    await waitFor(() => liveGridCanvas() !== null);

    // No stale selection ring: selected is null on the fresh live slice.
    expect(liveGridCanvas()!.getAttribute('aria-activedescendant')).toBeNull();
  });

  it('DOES restore a custom-permalink selection on the intended Run-without-edit path', async () => {
    ctl.run = makeRun(4, 4);
    mountApp('#p=Hello&sel=1,0'); // custom prompt → skeleton until Run
    await tick();
    expect(gridCanvas()).toBeNull(); // skeleton has no grid yet

    clickRun(); // no edits → the restore is intended and must survive
    await waitFor(() => liveGridCanvas() !== null);

    // The restored selection (1,0) is re-applied once on the first live slice.
    expect(liveGridCanvas()!.getAttribute('aria-activedescendant')).toBeTruthy();
  });
});

// -- F5: both rank charts render alongside a pinned token --------------------
describe('rank charts — depth AND position axes both present (F5)', () => {
  it('renders a rank-vs-depth chart and a rank-vs-position chart on the starter', async () => {
    mountApp(''); // cold → model-free french-season starter (carries 4 concept pins)
    await tick();
    await waitFor(() => container!.querySelectorAll('svg[role="img"]').length >= 2);

    const labels = Array.from(container!.querySelectorAll('svg[role="img"]')).map((s) =>
      s.getAttribute('aria-label'),
    );
    // One chart is keyed on residual depth (ℓ), the other on prompt position.
    expect(labels).toContain('residual boundary ℓ');
    expect(labels).toContain('prompt position');
  });
});
