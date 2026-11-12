/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { alignedRange } from '../ui/src/lib/range.js';
import type { MetricsSpeculative, ModelsResponse } from '../ui/src/lib/types.js';
import Metrics from '../ui/src/pages/metrics.js';
import {
  busyTotals,
  byModelRow,
  dayStart,
  distributions,
  latency,
  quantiles,
  speculative,
  summary,
  timeseries,
  tsPoint,
  turnRow,
  turnsPage,
} from './helpers/observability-fixtures.js';
import {
  byRole,
  chartTable,
  deferred,
  statValue,
  observeWidth,
  renderPage,
  STUB_FAILURE,
  sequence,
  stubApi,
  waitFor,
  type RenderedPage,
  type StubbedApi,
} from './render.js';

let mounted: RenderedPage | undefined;
let api: StubbedApi | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  api?.();
  api = undefined;
  document.body.innerHTML = '';
});

const GGUF: ModelsResponse = {
  models: [
    {
      name: 'Qwen3.6-27B-GGUF',
      path: '/models/Qwen3.6-27B-GGUF',
      modelType: 'qwen3_5',
      quant: 'Q4_K_M',
      contextWindow: 131_072,
      sizeBytes: 1,
      fileCount: 1,
      servedIds: ['qwen3-8b'],
    },
  ],
  warnings: [],
  dir: '/models',
  host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
};

function routes(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    '/metrics/models': { models: [] },
    '/metrics/summary': summary(),
    '/metrics/timeseries': timeseries([
      tsPoint(dayStart(3), 'qwen3-8b'),
      tsPoint(dayStart(1), 'qwen3-8b', { decodeTps: quantiles(30, 3) }),
    ]),
    '/metrics/distributions': distributions(),
    '/metrics/speculative': speculative(),
    '/metrics/turns': turnsPage([turnRow('t1'), turnRow('t2')]),
    '/models': GGUF,
    ...overrides,
  };
}

function Location(): ReactElement {
  const location = useLocation();
  return createElement('output', { 'data-search': location.search });
}

async function mountMetrics(
  url: string,
  until: (text: string) => boolean,
  extra: Record<string, unknown> = {},
  onCall?: (call: ApiCall) => void,
): Promise<RenderedPage> {
  api = stubApi(routes(extra), { onCall });
  mounted = await renderPage(
    createElement(MemoryRouter, { initialEntries: [url] }, createElement(Metrics), createElement(Location)),
    until,
  );
  return mounted;
}

function search(page: RenderedPage): string {
  return page.container.querySelector('[data-search]')?.getAttribute('data-search') ?? '';
}

function click(element: Element | null | undefined): void {
  if (element === null || element === undefined) throw new Error('click: no element');
  act(() => {
    (element as HTMLElement).click();
  });
}

async function figure(page: RenderedPage, name: string): Promise<void> {
  await waitFor(() => byRole(page.container, 'figure', name) !== null, 2000);
}

function buttonNamed(name: string): HTMLButtonElement | undefined {
  return [...document.querySelectorAll('button')].find((button) => (button.textContent ?? '').trim() === name);
}

describe('Metrics page', () => {
  it('starts its page column with the sticky header, so the title sits where every other page has it', async () => {
    await mountMetrics('/metrics?tab=speculation', (text) => text.includes('No speculative decoding recorded'));
    const column = mounted!.container.querySelector('[data-page]');
    const [first, second] = [...(column?.children ?? [])];
    expect(first?.getAttribute('aria-hidden')).toBe('true');
    expect(first?.nextElementSibling).toBe(second);
    expect(second?.hasAttribute('data-sticky')).toBe(true);
  });

  it('asks only for the routes of the open tab', async () => {
    const paths: string[] = [];
    await mountMetrics(
      '/metrics?tab=speculation',
      (text) => text.includes('No speculative decoding recorded'),
      {},
      (call) => paths.push(call.path.split('?')[0]),
    );
    const asked = new Set(paths);
    expect(asked.has('/api/metrics/speculative') || asked.has('/metrics/speculative')).toBe(true);
    for (const other of ['summary', 'timeseries', 'distributions', 'turns']) {
      expect([...asked].some((path) => path.endsWith(`/metrics/${other}`))).toBe(false);
    }
    expect(mounted?.text()).toContain('Turn on MTP (enableMtp) for a model that supports it to record acceptance.');
  });

  it('selects the ttft tile from ?kpi= and names the chart after it', async () => {
    const page = await mountMetrics('/metrics?kpi=ttft', (text) => text.includes('Time to first token'));
    const tile = byRole(page.container, 'group', 'First token');
    expect(tile?.querySelector('[aria-pressed="true"]')).not.toBeNull();
    expect(byRole(page.container, 'group', 'Decode speed')?.querySelector('[aria-pressed="true"]')).toBeNull();
    await figure(page, 'Time to first token');
  });

  it('adds the model to every request and draws one series', async () => {
    const paths: string[] = [];
    const page = await mountMetrics(
      '/metrics?model=qwen3-8b',
      (text) => text.includes('Median with the p05–p95 range'),
      {},
      (call) => paths.push(call.path),
    );
    const metricPaths = paths.filter((path) => path.includes('/metrics/') && !path.includes('/metrics/models'));
    expect(metricPaths.length).toBeGreaterThan(0);
    for (const path of metricPaths) expect(path).toContain('model=qwen3-8b');
    await figure(page, 'Decode speed');
    const rows = chartTable(page.container, 'Decode speed');
    expect(rows[0]).toHaveLength(2);
  });

  it('keeps a bucket with no turn as no data and marks a low sample', async () => {
    const page = await mountMetrics('/metrics', (text) => text.includes('Median per day, per model'));
    await figure(page, 'Decode speed');
    const rows = chartTable(page.container, 'Decode speed');
    expect(rows).toHaveLength(7);
    const column = rows.map((row) => row[1]);
    expect(column[0]).toBe('no data');
    expect(column).toContain('42.0');
    expect(column).toContain('30.0 (n=3)');
    expect(column).not.toContain('0.0');
  });

  it('shows the quant of a GGUF checkpoint whose served id differs from its folder, under the name', async () => {
    const page = await mountMetrics('/metrics', (text) => text.includes('Q4_K_M'));
    const row = page.container.querySelector('tr[data-model="qwen3-8b"]');
    expect(row?.textContent).toContain('Q4_K_M');
    expect(row?.querySelector('span[data-model]')?.textContent).toBe('qwen3-8b');
    expect(page.container.querySelector('tr[data-model="gemma-4-e4b"]')?.textContent).not.toContain('Q4_K_M');
  });

  it('gives a model load its own segment of the wall time', async () => {
    const page = await mountMetrics('/metrics', (text) => text.includes('Where the time goes'));
    await waitFor(() => page.text().includes('Loading the model'));
    const bar = page.container.querySelector('[aria-label="Wall time by phase"]')?.closest('section') ?? page.container;
    expect(bar.textContent).toContain('Loading the model');
    expect(bar.textContent).toContain('8.00 s');
  });

  it('switches the failures list between failed and cancelled turns', async () => {
    const paths: string[] = [];
    const page = await mountMetrics(
      '/metrics?tab=failures',
      (text) => text.includes('How turns end'),
      {
        '/metrics/turns': turnsPage([
          turnRow('f1', { outcome: 'error', errorKind: 'exception', failPhase: 'prefill' }),
        ]),
      },
      (call) => paths.push(call.path),
    );
    await waitFor(() => paths.some((path) => path.includes('outcome=error')));
    const failedRow = (): Element | undefined =>
      [...page.container.querySelectorAll('table')]
        .find((table) => table.querySelector('caption')?.textContent === 'Failed turns')
        ?.querySelector('tbody tr td[data-truncate]')
        ?.closest('tr') ?? undefined;
    await waitFor(() => failedRow() !== undefined);
    const row = failedRow();
    const when = row?.querySelector('td [title]');
    expect(when?.className).toContain('whitespace-nowrap');
    const oneLine = [...(row?.querySelectorAll('td[data-truncate]') ?? [])].map(
      (cell) => cell.querySelector('[title]')?.getAttribute('title') ?? '',
    );
    expect(oneLine).toEqual(expect.arrayContaining(['exception', 'during prefill']));
    const outcome = byRole(page.container, 'group', 'Outcome');
    click(outcome === null ? null : byRole(outcome, 'button', 'Cancelled'));
    await waitFor(() => paths.some((path) => path.includes('outcome=cancelled')));
    expect(page.text()).toContain('Cancelled · during decode');
  });

  it('loads 200 more with the cursor and keeps every shown row over a live refetch', async () => {
    const paths: string[] = [];
    const first = turnsPage([turnRow('a'), turnRow('b')], 4, { beforeTs: 1000, beforeTraceId: 'b' });
    const older = turnsPage([turnRow('c'), turnRow('d')], 4, null);
    const fresh = turnsPage([turnRow('new'), turnRow('a')], 5, { beforeTs: 1000, beforeTraceId: 'a' });
    const page = await mountMetrics(
      '/metrics?tab=turns',
      (text) => text.includes('Load 200 more'),
      { '/metrics/turns': sequence(first, older, fresh) },
      (call) => paths.push(call.path),
    );
    click(buttonNamed('Load 200 more'));
    await waitFor(() => page.text().includes('Showing 4 of 4 turns'));
    expect(paths.some((path) => path.includes('beforeTs=1000') && path.includes('beforeTraceId=b'))).toBe(true);
    const keys = () => [...page.container.querySelectorAll('tr[data-model]')].length;
    expect(keys()).toBe(4);
    act(() => api?.emitLive({ type: 'data', version: 1, traces: true, runtime: false }));
    await waitFor(() => page.text().includes('Showing 5 of 5 turns'), 3000);
    expect(keys()).toBe(5);
    const ids = [...page.container.querySelectorAll('tr[data-model] a')].map((link) => link.getAttribute('href') ?? '');
    expect(ids.filter((href) => href.includes('turn=b'))).toHaveLength(1);
  });

  it('keeps a new turn after it falls off the refetched first page', async () => {
    const page = await mountMetrics('/metrics?tab=turns', (text) => text.includes('Load 200 more'), {
      '/metrics/turns': sequence(
        turnsPage([turnRow('a'), turnRow('b')], 4, { beforeTs: 1000, beforeTraceId: 'b' }),
        turnsPage([turnRow('c'), turnRow('d')], 4, null),
        turnsPage([turnRow('n1'), turnRow('a')], 5, { beforeTs: 1000, beforeTraceId: 'a' }),
        turnsPage([turnRow('n2'), turnRow('n1')], 6, { beforeTs: 1000, beforeTraceId: 'n1' }),
        turnsPage([turnRow('n3'), turnRow('n2')], 7, { beforeTs: 1000, beforeTraceId: 'n2' }),
      ),
    });
    click(buttonNamed('Load 200 more'));
    await waitFor(() => page.text().includes('Showing 4 of 4 turns'));
    for (const shown of ['Showing 5 of 5 turns', 'Showing 6 of 6 turns', 'Showing 7 of 7 turns']) {
      act(() => api?.emitLive({ type: 'data', version: 1, traces: true, runtime: false }));
      await waitFor(() => page.text().includes(shown), 3000);
    }
    const hrefs = [...page.container.querySelectorAll('tr[data-model] a')].map(
      (link) => link.getAttribute('href') ?? '',
    );
    for (const id of ['n3', 'n2', 'n1', 'a', 'b', 'c', 'd']) {
      expect(hrefs.filter((href) => href.includes(`turn=${id}`))).toHaveLength(1);
    }
  });

  it('drops the held pages when a full refetched first page shares no row with them', async () => {
    const paths: string[] = [];
    const page = await mountMetrics(
      '/metrics?tab=turns',
      (text) => text.includes('Load 200 more'),
      {
        '/metrics/turns': sequence(
          turnsPage([turnRow('a'), turnRow('b')], 4, { beforeTs: 1000, beforeTraceId: 'b' }),
          turnsPage([turnRow('c'), turnRow('d')], 4, null),
          turnsPage([turnRow('x2'), turnRow('x1')], 7, { beforeTs: 2000, beforeTraceId: 'x1' }),
          turnsPage([turnRow('x0'), turnRow('a')], 7, { beforeTs: 1000, beforeTraceId: 'a' }),
        ),
      },
      (call) => paths.push(call.path),
    );
    click(buttonNamed('Load 200 more'));
    await waitFor(() => page.text().includes('Showing 4 of 4 turns'));
    act(() => api?.emitLive({ type: 'data', version: 1, traces: true, runtime: false }));
    await waitFor(() => page.text().includes('Showing 2 of 7 turns'), 3000);
    click(buttonNamed('Load 200 more'));
    await waitFor(() => paths.some((path) => path.includes('beforeTraceId=x1')));
    await waitFor(() => page.text().includes('Showing 4 of 7 turns'));
    expect(page.text()).toContain('Load 200 more');
  });

  it('keeps the loaded older pages when a data event moves the aligned window to the next bucket', async () => {
    const paths: string[] = [];
    const first = turnsPage([turnRow('a'), turnRow('b')], 4, { beforeTs: 1000, beforeTraceId: 'b' });
    const older = turnsPage([turnRow('c'), turnRow('d')], 4, null);
    const fresh = turnsPage([turnRow('new'), turnRow('a')], 5, { beforeTs: 1000, beforeTraceId: 'a' });
    const page = await mountMetrics(
      '/metrics?tab=turns&range=1h',
      (text) => text.includes('Load 200 more'),
      { '/metrics/turns': sequence(first, older, fresh) },
      (call) => paths.push(call.path),
    );
    click(buttonNamed('Load 200 more'));
    await waitFor(() => page.text().includes('Showing 4 of 4 turns'));
    const asked = paths.length;
    const realNow = Date.now.bind(Date);
    const clock = vi.spyOn(Date, 'now').mockImplementation(() => realNow() + 6 * 60_000);
    try {
      act(() => api?.emitLive({ type: 'data', version: 1, traces: true, runtime: false }));
      await waitFor(() => paths.slice(asked).some((path) => !path.includes('beforeTs')), 3000);
      await waitFor(() => page.text().includes('Showing 5 of 5 turns'), 3000);
    } finally {
      clock.mockRestore();
    }
    const moved = paths.slice(asked).find((path) => !path.includes('beforeTs')) ?? '';
    expect(moved).not.toBe(paths[0]);
    expect([...page.container.querySelectorAll('tr[data-model]')]).toHaveLength(5);
  });

  it('opens the decode panel from ?detail= and asks for the previous period once', async () => {
    const paths: string[] = [];
    const page = await mountMetrics(
      '/metrics?detail=decode',
      () => byRole(document.body, 'region', 'Decode speed') !== null,
      {},
      (call) => paths.push(call.path),
    );
    const summaryPath = paths.find((path) => path.includes('/metrics/summary')) ?? '';
    const from = new URLSearchParams(summaryPath.split('?')[1] ?? '').get('from');
    await waitFor(() => paths.some((path) => path.includes(`to=${from ?? ''}`)));
    expect(paths.filter((path) => path.includes(`to=${from ?? ''}`) && path.includes('groupBy=none'))).toHaveLength(1);
    click(byRole(document.body, 'button', '24 hours'));
    await waitFor(() => search(page).includes('range=24h'));
    expect(search(page)).toContain('detail=decode');
    expect(byRole(document.body, 'region', 'Decode speed')).not.toBeNull();
    click(byRole(document.body, 'button', 'Close details'));
    await waitFor(() => !search(page).includes('detail='));
  });

  it('holds the KPI panel trend in a placeholder while the series loads, and says when it failed', async () => {
    const held = deferred(timeseries([]));
    await mountMetrics('/metrics?detail=decode', () => byRole(document.body, 'region', 'Trend') !== null, {
      '/metrics/timeseries': held.body,
    });
    const trend = (): Element | null => byRole(document.body, 'region', 'Trend');
    expect(trend()?.querySelector('figure')).toBeNull();
    expect(trend()?.querySelector('[role="alert"]')).toBeNull();
    act(() => held.release());
    await waitFor(() => trend()?.querySelector('figure') != null);
    mounted?.unmount();
    api?.();
    await mountMetrics('/metrics?detail=decode', () => byRole(document.body, 'region', 'Trend') !== null, {
      '/metrics/timeseries': STUB_FAILURE,
    });
    await waitFor(() => (trend()?.querySelector('[role="alert"]')?.textContent ?? '').includes('Try again'));
    expect(trend()?.querySelector('figure')).toBeNull();
  });

  it('prints the context fill percent once per model', async () => {
    await mountMetrics('/metrics?tab=usage', (text) => text.includes('Context fill by model'), {
      '/metrics/summary': summary({
        byModel: [
          byModelRow('qwen3-8b', 40),
          byModelRow('gemma-4', 7, { latency: { ...latency(7), ctxFillPct: quantiles(18, 7) } }),
        ],
      }),
    });
    const list = (): Element | null => document.querySelector('[aria-label="Context fill by model"]');
    await waitFor(() => (list()?.textContent ?? '').includes('60%'));
    const rows = [...(list()?.querySelectorAll('li') ?? [])].map((li) => li.textContent ?? '');
    expect(rows).toHaveLength(2);
    expect(rows.some((text) => text.endsWith('60%'))).toBe(true);
    expect(rows.some((text) => text.endsWith('18%median, n=7'))).toBe(true);
    for (const text of rows) expect(text).not.toContain('%%');
  });

  it('hides Last used in the models table at the 720 window (about 610 px) and gives the context meter the cell width', async () => {
    const restore = observeWidth(610);
    try {
      const page = await mountMetrics('/metrics', (text) => text.includes('Decode tok/s'), {
        '/metrics/summary': summary({ byModel: [byModelRow('qwen3-8b', 40)] }),
      });
      const head = (): string[] =>
        [...(page.container.querySelectorAll('thead th') ?? [])].map((cell) => (cell.textContent ?? '').trim());
      await waitFor(() => !head().includes('Context'));
      expect(head()).not.toContain('Last used');
    } finally {
      restore();
    }
    mounted?.unmount();
    api?.();
    const wide = observeWidth(1100);
    try {
      const page = await mountMetrics('/metrics', (text) => text.includes('Decode tok/s'), {
        '/metrics/summary': summary({ byModel: [byModelRow('qwen3-8b', 40)] }),
      });
      await waitFor(() => page.container.querySelector('[role="meter"][aria-label^="Context window in use"]') !== null);
      const meter = page.container.querySelector('[role="meter"][aria-label^="Context window in use"]');
      expect(meter?.closest('[data-entrance]')?.className).toContain('w-full');
      const median = [...page.container.querySelectorAll('tr[data-model] .inline-flex > [title^="n="]')];
      expect(median.length).toBeGreaterThan(0);
      for (const cell of median) expect(cell.className).toContain('whitespace-nowrap');
    } finally {
      wide();
    }
  });

  it('stacks reasoning as a part of the output, so a column adds up to the tokens the turns used', async () => {
    const page = await mountMetrics('/metrics?tab=usage', (text) => text.includes('Prompt reuse'));
    await figure(page, 'Tokens');
    const filled = chartTable(page.container, 'Tokens').find((row) => row[1] !== 'no data' && row[1] !== '0');
    expect(filled?.slice(1)).toEqual(['160,000', '40,000', '20,000', '5,000', '225,000']);
    const head = byRole(page.container, 'figure', 'Tokens')?.querySelector('thead')?.textContent ?? '';
    expect(head).toContain('Output (answer) (tokens)');
  });

  it('marks the page busy while a query of the open tab loads', async () => {
    const held = deferred(turnsPage([turnRow('t1')]));
    const page = await mountMetrics('/metrics?tab=turns', (text) => text.includes('Turns'), {
      '/metrics/turns': held.body,
    });
    const busy = (): string | null => page.container.querySelector('[data-page]')?.getAttribute('aria-busy') ?? null;
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(busy()).toBe('true');
    act(() => held.release());
    await waitFor(() => page.text().includes('Showing 1 of 1 turn'));
    await waitFor(() => busy() === null);
  });

  it('prints the first-token unit once in the by-model list', async () => {
    await mountMetrics(
      '/metrics?detail=ttft',
      () => document.querySelector('[aria-label="First token by model"]') !== null,
      { '/metrics/summary': summary({ byModel: [byModelRow('qwen3-8b', 40)] }) },
    );
    const list = document.querySelector('[aria-label="First token by model"]');
    await waitFor(() => (list?.textContent ?? '').includes('400 ms'));
    expect(list?.textContent).not.toContain('ms ms');
  });

  it('labels duration axes once: no unit on the ticks, 0 as 0, the duration histograms in their own units', async () => {
    const page = await mountMetrics('/metrics?tab=speed', (text) => text.includes('Average turn, by phase'));
    await figure(page, 'Time to first token');
    const ttft = chartTable(page.container, 'Time to first token');
    expect(ttft[0]?.[0]).toBe('0–100 ms');
    const head = byRole(page.container, 'figure', 'Time to first token')?.querySelector('thead')?.textContent ?? '';
    expect(head).toContain('Range');
    expect(head).not.toContain('(ms)');
    await figure(page, 'Average turn, by phase');
    const phases = byRole(page.container, 'figure', 'Average turn, by phase');
    const ticks = [...(phases?.querySelectorAll('[class*="gutter"] > span') ?? [])].map(
      (node) => node.textContent ?? '',
    );
    expect(ticks.length).toBeGreaterThan(2);
    expect(ticks[0]).toBe('0');
    for (const tick of ticks) expect(tick).not.toContain('ms');
    const cells = chartTable(page.container, 'Average turn, by phase').flat();
    expect(cells.some((cell) => cell.includes('ms') || cell.endsWith(' s'))).toBe(false);
  });

  it('prints every tick of a tok/s axis in one format', async () => {
    const ticksOf = (page: RenderedPage, name: string): string[] =>
      [...(byRole(page.container, 'figure', name)?.querySelectorAll('[class*="gutter"] > span') ?? [])].map(
        (node) => node.textContent ?? '',
      );
    const summaryPage = await mountMetrics('/metrics', (text) => text.includes('Median per day, per model'));
    await figure(summaryPage, 'Decode speed');
    const decode = ticksOf(summaryPage, 'Decode speed');
    expect(decode.length).toBeGreaterThan(2);
    expect(decode[0]).toBe('0');
    for (const tick of decode) expect(tick.endsWith('.0')).toBe(false);
    mounted?.unmount();
    api?.();
    const speedPage = await mountMetrics('/metrics?tab=speed', (text) => text.includes('Average turn, by phase'));
    await figure(speedPage, 'Prefill speed');
    const prefill = ticksOf(speedPage, 'Prefill speed');
    expect(prefill.length).toBeGreaterThan(2);
    for (const tick of prefill) expect(tick.includes('K')).toBe(false);
  });

  it('drops the open panel when the tab changes', async () => {
    const page = await mountMetrics('/metrics?detail=turns', () => byRole(document.body, 'region', 'Turns') !== null);
    const tab = byRole(page.container, 'tab', 'Usage');
    act(() => {
      tab?.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, button: 0 }));
    });
    await waitFor(() => search(page).includes('tab=usage'));
    expect(search(page)).not.toContain('detail=');
  });

  it('draws a held hourly answer on its own hours while the daily answer loads', async () => {
    const hours = alignedRange('24h').starts;
    const hourly = timeseries(
      hours.map((t) => tsPoint(t, 'qwen3-8b', { totals: busyTotals({ turns: 1, cachedTokens: 7 }) })),
      'none',
    );
    const daily = deferred(timeseries([tsPoint(dayStart(0), 'qwen3-8b')], 'none'));
    const paths: string[] = [];
    const page = await mountMetrics(
      '/metrics?tab=usage&range=24h',
      (text) => text.includes('Prompt reuse'),
      { '/metrics/timeseries': sequence(hourly, daily.body) },
      (call) => paths.push(call.path),
    );
    await figure(page, 'Tokens');
    await waitFor(() => chartTable(page.container, 'Tokens').length === 24);
    click(byRole(document.body, 'button', '7 days'));
    await waitFor(() => paths.some((path) => path.includes('/metrics/timeseries') && path.includes('bucket=day')));
    await waitFor(() => search(page).includes('range=7d'));
    const held = chartTable(page.container, 'Tokens');
    expect(held).toHaveLength(24);
    expect(held.every((row) => row[1] === '7')).toBe(true);
    act(() => daily.release());
    await waitFor(() => chartTable(page.container, 'Tokens').length === 7);
  });

  it('names a first-token quantile bar of a second or more in its own unit, once', async () => {
    const page = await mountMetrics('/metrics', (text) => text.includes('Decode tok/s'), {
      '/metrics/summary': summary({
        byModel: [byModelRow('qwen3-8b', 40, { latency: { ...latency(40), ttftMs: quantiles(1240, 40) } })],
      }),
    });
    const bar = (): Element | null =>
      page.container.querySelector('[role="img"][aria-label^="Time to first token of qwen3-8b"]');
    await waitFor(() => bar() !== null);
    expect(bar()?.getAttribute('aria-label')).toBe(
      'Time to first token of qwen3-8b: median 1.24 s, p05 620 ms, p95 1.86 s, 40 turns',
    );
  });

  it('reads the API 0-based draft positions as Position 1…n, prints % once, and draws the top 8 models over time', async () => {
    const model = (index: number): MetricsSpeculative['models'][number] => ({
      model: `mtp-${String(index)}`,
      turns: 20,
      mtpTurns: 20,
      cycles: 1000 - index,
      meanAccepted: 2.5,
      meanDrafts: 1.5,
      meanDepth: 3,
      acceptByPos: [
        { pos: 0, rate: 0.81, cycles: 1000 - index },
        { pos: 1, rate: 0.48, cycles: 1000 - index },
      ],
      decodeTpsWithMtp: quantiles(60, 20),
      decodeTpsWithoutMtp: quantiles(40, 20),
    });
    const page = await mountMetrics(
      '/metrics?tab=speculation',
      (text) => text.includes('Acceptance by draft position'),
      {
        '/metrics/speculative': speculative(Array.from({ length: 9 }, (_, index) => model(index))),
      },
    );
    await figure(page, 'Acceptance by draft position');
    const rows = chartTable(page.container, 'Acceptance by draft position');
    expect(rows.map((row) => row[0])).toEqual(['Position 1', 'Position 2']);
    expect(rows[0][1]).toBe('81');
    const sweep = page.container.querySelector('[role="img"][aria-label^="Share of drafts kept"]');
    expect(sweep?.getAttribute('aria-label')).toContain('81%');
    const slider = byRole(page.container, 'figure', 'Acceptance by draft position')?.querySelector('[role="slider"]');
    expect(slider?.getAttribute('aria-valuetext')).toContain('48%');
    expect(slider?.getAttribute('aria-valuetext')).not.toContain('% %');
    expect(slider?.getAttribute('aria-valuetext')).not.toContain('%%');
    await figure(page, 'Tokens per cycle over time');
    const head = byRole(page.container, 'figure', 'Tokens per cycle over time')?.querySelectorAll('thead th') ?? [];
    expect(head).toHaveLength(9);
    expect(page.text()).toContain('1.0 = no draft kept · +1 model in the table');
  });

  it('gives the model name the free width of the Models, Turns and Speculation tables, with the whole name as its title', async () => {
    const LONG = 'qwen3.6-35b-a3b-agentworld-instruct-ud-q4-k-xl-mlx-dwq-8bits';
    const check = (page: RenderedPage, caption: string): void => {
      const table = [...page.container.querySelectorAll('table')].find(
        (node) => node.querySelector('caption')?.textContent === caption,
      )!;
      const heads = [...table.querySelectorAll('thead th')].map((th) => (th.textContent ?? '').trim());
      const col = table.querySelectorAll('col')[heads.indexOf('Model')]!;
      expect(col.getAttribute('style') ?? '').not.toContain('width');
      const name = table.querySelector(`tbody [title="${LONG}"]`)!;
      expect([...name.querySelectorAll('[aria-hidden="true"]')].map((n) => n.textContent).join('')).toBe(LONG);
    };
    const next = (): void => {
      mounted?.unmount();
      mounted = undefined;
      api?.();
      api = undefined;
    };
    const models = await mountMetrics('/metrics', (text) => text.includes('Decode tok/s'), {
      '/metrics/summary': summary({ byModel: [byModelRow(LONG, 40)] }),
    });
    await waitFor(() => models.container.querySelector('tbody tr[data-model]') !== null);
    check(models, 'Models');
    next();
    const turns = await mountMetrics('/metrics?tab=turns', (text) => text.includes('Showing 1 of 1 turn'), {
      '/metrics/turns': turnsPage([turnRow('a', { model: LONG })], 1, null),
    });
    check(turns, 'Turns');
    next();
    const restore = observeWidth(909);
    try {
      const narrow = await mountMetrics('/metrics?tab=turns', (text) => text.includes('Showing 1 of 1 turn'), {
        '/metrics/turns': turnsPage([turnRow('a', { model: LONG })], 1, null),
      });
      const heads = (): string[] =>
        [...narrow.container.querySelectorAll('table thead th')].map((th) => (th.textContent ?? '').trim());
      await waitFor(() => !heads().includes('Context'));
      expect(heads()).toEqual(expect.arrayContaining(['When', 'Model', 'Decode tok/s']));
      expect(heads()).not.toContain('Prefill tok/s');
    } finally {
      restore();
    }
    next();
    const mtp: MetricsSpeculative['models'][number] = {
      model: LONG,
      turns: 20,
      mtpTurns: 20,
      cycles: 1000,
      meanAccepted: 2.5,
      meanDrafts: 1.5,
      meanDepth: 3,
      acceptByPos: [{ pos: 0, rate: 0.8, cycles: 1000 }],
      decodeTpsWithMtp: quantiles(60, 20),
      decodeTpsWithoutMtp: quantiles(40, 20),
    };
    const spec = await mountMetrics(
      '/metrics?tab=speculation',
      (text) => text.includes('Acceptance by draft position'),
      {
        '/metrics/speculative': speculative([mtp]),
      },
    );
    await waitFor(() => spec.container.querySelector('tbody tr[data-model]') !== null);
    check(spec, 'Speculation by model');
    const specTable = [...spec.container.querySelectorAll('table')].find(
      (node) => node.querySelector('caption')?.textContent === 'Speculation by model',
    )!;
    const specHeads = [...specTable.querySelectorAll('thead th')].map((th) => (th.textContent ?? '').trim());
    const widthOf = (head: string): string =>
      (specTable.querySelectorAll('col')[specHeads.indexOf(head)] as HTMLElement).style.width;
    expect([widthOf('Cycles'), widthOf('With MTP tok/s'), widthOf('Without tok/s')]).toEqual([
      '84px',
      '148px',
      '148px',
    ]);
  });

  it('states decode speed in tok/s once, in the column head, on the Turns and Speculation tables', async () => {
    const heads = (page: RenderedPage): string[] =>
      [...page.container.querySelectorAll('table thead th')].map((th) => (th.textContent ?? '').trim());
    const turns = await mountMetrics('/metrics?tab=turns', (text) => text.includes('Showing 1 of 1 turn'), {
      '/metrics/turns': turnsPage([turnRow('a', { decodeTps: 65.1 })], 1, null),
    });
    expect(heads(turns)).toContain('Decode tok/s');
    const row = turns.container.querySelector('tr[data-model]');
    expect(row?.textContent).toContain('65.1');
    expect(row?.textContent).not.toContain('tok/s');
    turns.unmount();
    mounted = undefined;
    api?.();
    api = undefined;

    const mtp: MetricsSpeculative['models'][number] = {
      model: 'mtp-0',
      turns: 20,
      mtpTurns: 20,
      cycles: 1000,
      meanAccepted: 2.5,
      meanDrafts: 1.5,
      meanDepth: 3,
      acceptByPos: [{ pos: 0, rate: 0.8, cycles: 1000 }],
      decodeTpsWithMtp: quantiles(60, 20),
      decodeTpsWithoutMtp: quantiles(40, 20),
    };
    const spec = await mountMetrics(
      '/metrics?tab=speculation',
      (text) => text.includes('Acceptance by draft position'),
      {
        '/metrics/speculative': speculative([mtp]),
      },
    );
    await waitFor(() => spec.container.querySelector('tr[data-model="mtp-0"]') !== null);
    expect(heads(spec)).toEqual(expect.arrayContaining(['With MTP tok/s', 'Without tok/s']));
    expect(spec.container.querySelector('tr[data-model="mtp-0"]')?.textContent).not.toContain('tok/s');
  });

  it('shows six speed tiles with Setup next to First token and a measured prefill speed', async () => {
    const page = await mountMetrics('/metrics?tab=speed', (text) => text.includes('Average turn, by phase'));
    await waitFor(() => page.container.querySelector('[aria-label="Setup"] [data-stat-value]') !== null);
    const group = byRole(page.container, 'group', 'Percentiles');
    const labels = [...(group?.querySelectorAll('[role="group"]') ?? [])].map((tile) =>
      tile.getAttribute('aria-label'),
    );
    expect(labels).toEqual(['First token', 'Setup', 'Prefill speed', 'Decode speed', 'Whole turn', 'Queue wait']);
    expect(statValue(page.container, 'Setup')).toBe('80 ms');
    expect(statValue(page.container, 'Prefill speed')).toBe('1,800 tok/s');
    expect(page.text()).toContain('Setup: before prefill, prefix lookup, cache restore and allocation');
    expect(page.text()).toContain('prefills of 256+ tokens');
  });

  it('says which turns the prefill speed covers on a mixed range', async () => {
    const mixed = summary({
      totals: busyTotals({ prefillMeasured: 60, firstTokenWallMs: 3_000 }),
      latency: { ...latency(), prefillTps: quantiles(1800, 40) },
    });
    const page = await mountMetrics('/metrics?tab=speed', (text) => text.includes('Rated 40 of'), {
      '/metrics/summary': mixed,
    });
    expect(page.text()).toContain('Rated 40 of 95 completed turns · 35 not measured · 20 under 256 tokens');
    expect(page.text()).toContain(
      'To first token: 35 turns without a prefill measurement, setup and prefill together.',
    );
  });

  it('names why a range has no prefill speed', async () => {
    const none = summary({
      totals: busyTotals({ prefillMeasured: 0, setupWallMs: 0, prefillWallMs: 0, firstTokenWallMs: 20_000 }),
      latency: { ...latency(), prefillTps: quantiles(null, 0), setupMs: quantiles(null, 0) },
    });
    const old = await mountMetrics('/metrics?tab=speed', (text) => text.includes('No measured prefill in this range'), {
      '/metrics/summary': none,
      '/metrics/turns': turnsPage([
        turnRow('o1', { prefillMs: null, prefillTokens: null, setupMs: null, prefillTps: null }),
      ]),
    });
    expect(byRole(old.container, 'group', 'Prefill speed')?.textContent).toContain('no measured prefill');
    expect(byRole(old.container, 'group', 'Setup')?.textContent).toContain('no measured prefill');
    expect(old.text()).toContain('Turns recorded before this version, and non-streamed turns, carry no prefill time.');
    old.unmount();
    mounted = undefined;
    api?.();
    api = undefined;

    const small = summary({ latency: { ...latency(), prefillTps: quantiles(null, 0) } });
    const page = await mountMetrics(
      '/metrics?tab=speed',
      (text) => text.includes('No prefill of 256+ tokens in this range'),
      { '/metrics/summary': small },
    );
    expect(byRole(page.container, 'group', 'Prefill speed')?.textContent).toContain('no prefill of 256+ tokens');
  });

  it('plots prefill time against prefill tokens, and only measured turns', async () => {
    const page = await mountMetrics(
      '/metrics?tab=speed',
      (text) => text.includes('Prefill time against prefill tokens'),
      {
        '/metrics/turns': turnsPage([
          turnRow('m1'),
          turnRow('o1', { prefillMs: null, prefillTokens: null, setupMs: null, prefillTps: null }),
        ]),
      },
    );
    await figure(page, 'Prefill time against prefill tokens');
    expect(page.text()).toContain('Prefill tokens');
    expect(page.text()).toContain('1 without a prefill measurement · hollow = ran beside other turns');
    expect(page.text()).not.toContain('First token against new prompt tokens');
  });

  it('names prefill tokens and prefill time once each in a scatter point tooltip', async () => {
    const page = await mountMetrics(
      '/metrics?tab=speed',
      (text) => text.includes('Prefill time against prefill tokens'),
      { '/metrics/turns': turnsPage([turnRow('m1')]) },
    );
    await figure(page, 'Prefill time against prefill tokens');
    const chart = byRole(page.container, 'figure', 'Prefill time against prefill tokens') as HTMLElement;
    const slider = byRole(chart, 'slider') as HTMLElement;
    act(() => {
      slider.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true }));
    });
    const before = chart.textContent ?? '';
    const count = (text: string, part: string): number => text.split(part).length - 1;
    const head = [...chart.querySelectorAll('table thead th')].map((th) => (th.textContent ?? '').trim());
    expect(head).toContain('Prefill time (ms)');
    expect(before).toContain('Setup');
    expect(before).not.toContain('Value');
    // Axis title and table head carry the tokens; chart label and table head carry the time; the tooltip adds one each.
    expect(count(before, 'Prefill tokens')).toBe(3);
    expect(count(before, 'Prefill time')).toBe(3);
  });

  it('shows — for a model without a rated prefill and a To first token segment for an old range', async () => {
    const old = summary({
      totals: busyTotals({ prefillMeasured: 0, setupWallMs: 0, prefillWallMs: 0, firstTokenWallMs: 20_000 }),
      byModel: [
        byModelRow('qwen3-8b', 60),
        byModelRow('gemma-4-e4b', 40, { latency: { ...latency(40), prefillTps: quantiles(null, 0) } }),
      ],
    });
    const page = await mountMetrics('/metrics', (text) => text.includes('Where the time goes'), {
      '/metrics/summary': old,
    });
    await waitFor(() => page.container.querySelector('tr[data-model="gemma-4-e4b"]') !== null);
    const row = page.container.querySelector('tr[data-model="gemma-4-e4b"]');
    const heads = [...(row?.closest('table')?.querySelectorAll('thead th') ?? [])].map((th) =>
      (th.textContent ?? '').trim(),
    );
    const column = heads.indexOf('Prefill tok/s');
    expect(column).toBeGreaterThan(-1);
    const cell = row?.querySelectorAll('td, th')[column];
    expect((cell?.textContent ?? '').trim()).toBe('—');
    const bar = page.container.querySelector('[aria-label="Wall time by phase"]')?.closest('section') ?? page.container;
    expect(bar.textContent).toContain('To first token');
    expect(bar.textContent).not.toContain('Setup');
    expect(page.text()).toContain('To first token: turns without a prefill measurement, setup and prefill together.');
  });

  it('gives the Turns table a rated prefill column', async () => {
    const page = await mountMetrics('/metrics?tab=turns', (text) => text.includes('Showing 2 of 2 turns'), {
      '/metrics/turns': turnsPage(
        [turnRow('a', { prefillTps: 2_400 }), turnRow('b', { prefillTps: null, prefillMs: 40, prefillTokens: 90 })],
        2,
        null,
      ),
    });
    const heads = [...page.container.querySelectorAll('table thead th')].map((th) => (th.textContent ?? '').trim());
    const column = heads.indexOf('Prefill tok/s');
    expect(column).toBeGreaterThan(heads.indexOf('First token'));
    const cells = [...page.container.querySelectorAll('tr[data-model]')].map((row) =>
      (row.querySelectorAll('td, th')[column]?.textContent ?? '').trim(),
    );
    expect(cells).toEqual(['2,400', '—']);
  });

  it('reads byModel quantiles for the low-sample chip in the table', async () => {
    const page = await mountMetrics('/metrics', (text) => text.includes('n=3'), {
      '/metrics/summary': summary({ byModel: [byModelRow('qwen3-8b', 3, { totals: busyTotals({ turns: 3 }) })] }),
    });
    expect(page.container.querySelector('tr[data-model="qwen3-8b"]')?.textContent).toContain('n=3');
  });
});
