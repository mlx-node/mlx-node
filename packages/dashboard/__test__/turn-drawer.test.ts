/**
 * @vitest-environment happy-dom
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { useTurnDrawer } from '../ui/src/components/turn-drawer.js';
import type { TurnRow } from '../ui/src/lib/types.js';
import { turnRow, turnsPage } from './helpers/observability-fixtures.js';
import { deferred, renderPage, stubApi, waitFor, type RenderedPage } from './render.js';

let mounted: RenderedPage | undefined;
let restoreApi: (() => void) | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
  document.body.innerHTML = '';
});

function Harness({ known }: { known: readonly TurnRow[] }): ReactElement {
  const drawer = useTurnDrawer(known);
  const location = useLocation();
  return createElement('div', { 'data-search': location.search }, drawer.element);
}

let setKnown: ((rows: readonly TurnRow[]) => void) | undefined;

/** A table whose first row links to `t1`, plus the drawer; `setKnown` swaps the rows the page holds. */
function TableHarness({ initial }: { initial: readonly TurnRow[] }): ReactElement {
  const [known, set] = useState(initial);
  setKnown = set;
  const drawer = useTurnDrawer(known);
  const location = useLocation();
  const row = createElement(
    'tr',
    { tabIndex: -1, 'data-opener': '' },
    createElement(
      'td',
      null,
      createElement('a', { href: '/metrics?tab=turns&turn=t1', 'data-row-link': '', tabIndex: -1 }, 'row'),
    ),
  );
  return createElement(
    'div',
    { 'data-search': location.search },
    createElement('table', null, createElement('tbody', null, row)),
    drawer.element,
  );
}

async function mountDrawer(
  url: string,
  known: readonly TurnRow[],
  routes: Record<string, unknown> = {},
): Promise<RenderedPage> {
  restoreApi = stubApi(routes);
  mounted = await renderPage(
    createElement(MemoryRouter, { initialEntries: [url] }, createElement(Harness, { known })),
    () => document.querySelector('[role="dialog"]') !== null,
  );
  return mounted;
}

function drawerText(): string {
  return (document.querySelector('[role="dialog"]')?.textContent ?? '').split('\n').join(' ');
}

describe('TurnDrawer', () => {
  it('shows the timeline, tokens, speed, context and the record of a known turn', async () => {
    await mountDrawer('/metrics?turn=t1', [turnRow('t1', { mtpCycles: 12, mtpMeanAccepted: 2.31 })]);
    const text = drawerText();
    for (const heading of ['Timeline', 'Tokens', 'Speed', 'Context']) expect(text).toContain(heading);
    expect(document.querySelector('[role="dialog"] [aria-label="Turn details"]')).not.toBeNull();
    expect(text).toContain('12 cycles · 2.31 tokens per cycle');
    expect(text).toContain('10% of 131,072 tokens');
    expect(text).toContain('pid 812');
    expect(text).not.toContain('Stopped');
  });

  it('shows the three native waits and the GPU kernels a turn created', async () => {
    await mountDrawer('/?turn=t3', [
      turnRow('t3', {
        waitMs: 75_070,
        slotMs: 300,
        prepareMs: 1_200,
        ttftMs: 2_600,
        durationMs: 79_700,
        pipelinesCreated: 212,
        pipelineCreateMs: 74_900,
      }),
    ]);
    const text = drawerText();
    expect(text).toContain('Model busy');
    expect(text).toContain('Waiting for a slot');
    expect(text).toContain('Preparing');
    expect(text).toContain('GPU kernels created');
    expect(text).toContain('212 (');
  });

  it('a record without the waits or a kernel count shows neither', async () => {
    await mountDrawer('/?turn=t4', [turnRow('t4', { pipelinesCreated: 0, pipelineCreateMs: 0 })]);
    const text = drawerText();
    expect(text).not.toContain('Model busy');
    expect(text).not.toContain('Waiting for a slot');
    expect(text).not.toContain('GPU kernels created');
  });

  it('a turn that spent no time waiting shows no wait rows, and its setup holds only the wait behind others', async () => {
    await mountDrawer('/?turn=t6', [
      turnRow('t6', {
        waitMs: 0,
        slotMs: 0,
        prepareMs: 1_200,
        ttftMs: 2_300,
        setupMs: 600,
        prefillMs: 500,
        prefillTokens: 1_000,
      }),
    ]);
    const text = drawerText();
    expect(text).not.toContain('Model busy');
    expect(text).not.toContain('Waiting for a slot');
    expect(text).toContain('Preparing');
    expect(text).toContain('other requests on this model, first decode step');
    expect(text).not.toContain('prefix lookup');
  });

  it('a turn cancelled while preparing says so', async () => {
    await mountDrawer('/?turn=t5', [
      turnRow('t5', { outcome: 'cancelled', failPhase: 'prepare', prepareMs: 800, ttftMs: null, prefillMs: null }),
    ]);
    expect(drawerText()).toContain('Stopped while preparing.');
  });

  it('says where a failed turn stopped and that its error text is not kept', async () => {
    await mountDrawer('/?turn=t2', [turnRow('t2', { outcome: 'error', errorKind: 'exception', failPhase: 'prefill' })]);
    const text = drawerText();
    expect(text).toContain('Failed');
    expect(text).toContain('Stopped during prefill.');
    expect(text).toContain('Error text is not recorded. See the server log around this time.');
  });

  it('looks up a turn that is not on screen and reports a missing one', async () => {
    const calls: string[] = [];
    restoreApi = stubApi({ '/metrics/turns': turnsPage([]) }, { onCall: (call) => calls.push(call.path) });
    mounted = await renderPage(
      createElement(MemoryRouter, { initialEntries: ['/?turn=gone'] }, createElement(Harness, { known: [] })),
      () => drawerText().includes('Turn not found'),
    );
    expect(drawerText()).toContain('It may be older than 30 days.');
    expect(calls.some((path) => path.includes('traceId=gone'))).toBe(true);
  });

  it('keeps the shown turn when it drops out of the rows the page holds', async () => {
    const calls: string[] = [];
    const never = deferred(turnsPage([]));
    restoreApi = stubApi({ '/metrics/turns': never.body }, { onCall: (call) => calls.push(call.path) });
    mounted = await renderPage(
      createElement(
        MemoryRouter,
        { initialEntries: ['/?turn=t1'] },
        createElement(TableHarness, { initial: [turnRow('t1', { mtpCycles: 12, mtpMeanAccepted: 2.31 })] }),
      ),
      () => drawerText().includes('12 cycles'),
    );
    act(() => setKnown?.([turnRow('t9')]));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(drawerText()).toContain('12 cycles · 2.31 tokens per cycle');
    expect(document.querySelector('[role="dialog"] [data-placeholder]')).toBeNull();
    expect(calls.some((path) => path.includes('traceId=t1'))).toBe(false);
  });

  it('gives the focus back to the row that opened it', async () => {
    restoreApi = stubApi({});
    mounted = await renderPage(
      createElement(
        MemoryRouter,
        { initialEntries: ['/metrics?tab=turns&turn=t1'] },
        createElement(TableHarness, { initial: [turnRow('t1')] }),
      ),
      () => document.querySelector('[role="dialog"]') !== null,
    );
    const opener = document.querySelector('[data-opener]');
    act(() => {
      (document.activeElement ?? document.body).dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }),
      );
    });
    await waitFor(() => document.querySelector('[role="dialog"]') === null, 2000);
    expect(document.activeElement).toBe(opener);
  });

  it('counts one overlapping turn in the singular', async () => {
    await mountDrawer('/?turn=t3', [turnRow('t3', { concurrency: 1, concurrencyPeak: 2 })]);
    expect(drawerText()).toContain('1 turn at start · peak 2');
  });

  it('splits the first token into setup and a rated prefill', async () => {
    await mountDrawer('/?turn=t4', [
      turnRow('t4', { ttftMs: 1200, setupMs: 200, prefillMs: 1000, prefillTokens: 2000, prefillTps: 2000 }),
    ]);
    const text = drawerText();
    expect(text).toContain('First token');
    expect(text).toContain('1.20 s');
    expect(text).toContain('Setup');
    expect(text).toContain('prefix lookup, cache restore, allocation, first decode step');
    expect(text).toContain('2,000 tok/s');
    expect(text).toContain('2,000 tokens in 1.00 s');
  });

  it('gives a small prefill its size but no rate', async () => {
    await mountDrawer('/?turn=t5', [
      turnRow('t5', { ttftMs: 400, setupMs: 350, prefillMs: 50, prefillTokens: 100, prefillTps: null }),
    ]);
    expect(drawerText()).toContain('100 tokens in 50 ms · under 256 tokens');
  });

  it('lets the setup and prefill hints wrap in the narrow drawer instead of cutting them', async () => {
    await mountDrawer('/?turn=t5', [
      turnRow('t5', { ttftMs: 400, setupMs: 350, prefillMs: 50, prefillTokens: 100, prefillTps: null }),
    ]);
    const speed = document.querySelector('dl[aria-label="Speed"]');
    const hints = [...(speed?.querySelectorAll('dd span[data-wrap]') ?? [])].map((hint) => hint.textContent);
    expect(hints).toEqual([
      'prefix lookup, cache restore, allocation, first decode step',
      '100 tokens in 50 ms · under 256 tokens',
    ]);
  });

  it('gives a turn without a prefill window one To first token span and no prefill', async () => {
    await mountDrawer('/?turn=t6', [
      turnRow('t6', { ttftMs: 400, setupMs: null, prefillMs: null, prefillTokens: null, prefillTps: null }),
    ]);
    const text = drawerText();
    expect(text).toContain('not measured for this turn');
    expect(text).toContain('To first token');
  });

  it('counts one tool call, image, cycle, hit and miss in the singular', async () => {
    await mountDrawer('/?turn=t4', [
      turnRow('t4', {
        toolCount: 1,
        toolCallCount: 1,
        imageCount: 1,
        mtpCycles: 1,
        mtpMeanAccepted: 2,
        coldHits: 1,
        coldMisses: 1,
      }),
    ]);
    const text = drawerText();
    expect(text).toContain('1 tool offered · 1 tool call · 1 image');
    expect(text).toContain('1 cycle · 2.00 tokens per cycle');
    expect(text).toContain('1 hit · 1 miss');
    for (const plural of ['1 tool calls', '1 images', '1 cycles', '1 hits', '1 misses']) {
      expect(text).not.toContain(plural);
    }
  });

  it('removes ?turn= when it closes', async () => {
    const page = await mountDrawer('/metrics?tab=turns&turn=t1', [turnRow('t1')]);
    const search = (): string => page.container.querySelector('[data-search]')?.getAttribute('data-search') ?? '';
    expect(search()).toContain('turn=t1');
    const target = document.activeElement ?? document.body;
    act(() => {
      target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    });
    await waitFor(() => !search().includes('turn='));
    expect(search()).toBe('?tab=turns');
  });
});
