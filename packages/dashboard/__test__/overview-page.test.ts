/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import type { ModelsResponse, RuntimeLive } from '../ui/src/lib/types.js';
import Overview from '../ui/src/pages/overview.js';
import {
  busyTotals,
  cacheResponse,
  dayStart,
  emptySummary,
  live,
  liveModel,
  liveTurn,
  process,
  summary,
  timeseries,
  turnRow,
  turnsPage,
} from './helpers/observability-fixtures.js';
import {
  allByRole,
  byRole,
  chartTable,
  renderPage,
  statValue,
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

const GGUF_MODELS: ModelsResponse = {
  models: [
    {
      name: 'Qwen3.6-27B-GGUF',
      path: '/models/Qwen3.6-27B-GGUF',
      modelType: 'qwen3_5',
      quant: 'Q4_K_M',
      contextWindow: 131_072,
      sizeBytes: 16 * 1024 ** 3,
      fileCount: 3,
      servedIds: ['Qwen3.6-27B-Q4_K_M'],
    },
  ],
  warnings: [],
  dir: '/models',
  host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
};

function routes(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    '/metrics/summary': summary(),
    '/metrics/timeseries': timeseries([]),
    '/metrics/turns': turnsPage([turnRow('t1'), turnRow('t2', { outcome: 'error', failPhase: 'decode' })]),
    '/models': GGUF_MODELS,
    '/cache': cacheResponse(),
    '/downloads': { jobs: [] },
    ...overrides,
  };
}

async function mountOverview(
  extra: Record<string, unknown> = {},
  url = '/',
  onCall?: (call: ApiCall) => void,
): Promise<RenderedPage> {
  api = stubApi(routes(extra), { onCall });
  mounted = await renderPage(
    createElement(MemoryRouter, { initialEntries: [url] }, createElement(Overview)),
    (text) => text.includes('Latest turns') && text.includes('Healthy'),
  );
  return mounted;
}

async function push(snapshot: RuntimeLive, until: (text: string) => boolean): Promise<void> {
  const page = mounted;
  if (page === undefined || api === undefined) throw new Error('push: nothing mounted');
  await waitFor(() => {
    act(() => api?.emitLive({ type: 'runtime', snapshot }));
    return until(page.text());
  }, 2000);
}

function engineText(page: RenderedPage): string {
  return byRole(page.container, 'region', 'Engine')?.textContent ?? page.text();
}

describe('Overview engine panel', () => {
  it('shows the rate and the turns in flight while running', async () => {
    const page = await mountOverview();
    await push(
      live({
        processes: [
          process({
            turns: [
              liveTurn('a', { model: 'claude-haiku-x', modelId: 'qwen3-8b', ctxWindow: 40_000, promptTokens: 10_000 }),
              liveTurn('b', { phase: 'prefill', prefillDone: 200, prefillTotal: 800, generatedTokens: null }),
            ],
          }),
        ],
      }),
      (text) => text.includes('in flight ·'),
    );
    const text = engineText(page);
    expect(text).toContain('desktop · :52114 · up');
    expect(text).toContain('42.7');
    const rate = byRole(page.container, 'region', 'Engine')?.querySelector('[data-stat-value]');
    const reserved = [...(rate?.querySelectorAll('[style]') ?? [])].filter((node) =>
      (node.getAttribute('style') ?? '').includes('min-width'),
    );
    expect(reserved).toHaveLength(0);
    expect(rate?.parentElement?.textContent).toContain('tok/s');
    expect(rate?.parentElement?.className.split(' ')).toContain('min-w-[7ch]');
    expect(text).toContain('qwen3-8b');
    const turns = byRole(page.container, 'list', 'Turns in flight');
    expect(turns?.querySelectorAll('li')).toHaveLength(2);
    const aliased = [...(turns?.querySelectorAll('li') ?? [])].find((li) =>
      li.querySelector('[title="claude-haiku-x"]'),
    );
    expect(aliased?.getAttribute('aria-label')?.startsWith('qwen3-8b,')).toBe(true);
    const badge = aliased?.querySelector('[data-model="qwen3-8b"]');
    expect(badge?.className).toContain('modelName');
    expect(badge?.nextElementSibling?.getAttribute('title')).toBe('claude-haiku-x');
    expect(turns?.querySelectorAll('[title="qwen3-8b"]')).toHaveLength(0);
  });

  it('says idle and when the last turn ended', async () => {
    const page = await mountOverview();
    await push(live({ processes: [process({ turns: [], decodeTps: 0 })] }), (text) => text.includes('Idle.'));
    expect(page.text()).toContain('Idle. Last turn finished 1m ago.');
  });

  it('says how to start a stopped engine, and the crash it stopped on', async () => {
    const page = await mountOverview();
    await push(
      live({
        status: 'stopped',
        processes: [],
        supervisor: {
          state: 'stopped',
          pid: null,
          generation: 1,
          consecutiveCrashes: 1,
          lastExit: { verdict: 'crash', code: 1, signal: null, atMs: Date.now() - 60_000 },
          healthStatus: null,
          nativeErrorCount: 0,
          nativeErrorsCapped: false,
          lastNativeErrorContext: null,
          lastNativeErrorAt: null,
          writtenAt: Date.now(),
        },
      }),
      (text) => text.includes('Inference is not running.'),
    );
    expect(page.text()).toContain('Start it from the menu bar, or run mlx serve in a terminal.');
    expect(page.text()).toContain('Last exit: crash at');
  });

  it('raises the failed alert with a way to the Engine page', async () => {
    const page = await mountOverview();
    await push(live({ status: 'failed', processes: [] }), (text) => text.includes('Inference failed'));
    const alert = byRole(page.container, 'alert');
    expect(alert?.textContent).toContain('Open Engine');
  });

  it('links a loaded GGUF checkpoint to its folder through its served id', async () => {
    const page = await mountOverview();
    await push(live({ processes: [process({ models: [liveModel('Qwen3.6-27B-Q4_K_M')] })] }), (text) =>
      text.includes('Loaded'),
    );
    const links = allByRole(page.container, 'link').map((link) => link.getAttribute('href'));
    expect(links).toContain('/models?focus=Qwen3.6-27B-GGUF');
  });
});

describe('Overview health and numbers', () => {
  it('links each health row to the page that explains it', async () => {
    const page = await mountOverview();
    const health = byRole(page.container, 'region', 'Health') ?? page.container;
    const hrefs = [...health.querySelectorAll('a')].map((link) => link.getAttribute('href'));
    expect(hrefs).toEqual(['/engine', '/engine', '/metrics?tab=failures&range=24h', '/cache', '/models']);
    expect(health.textContent).toContain('3 of 100 failed');
    expect(health.textContent).toContain('1 installed');
  });

  it('prints — for the medians and 0 for the counts with no turns', async () => {
    const page = await mountOverview({ '/metrics/summary': emptySummary() });
    await waitFor(() => statValue(page.container, 'Turns') === '0');
    expect(statValue(page.container, 'Decode speed')).toBe('—');
    expect(statValue(page.container, 'First token')).toBe('—');
    expect(page.text()).toContain('No turns in the last 24 hours');
  });

  it('draws 24 hourly columns', async () => {
    const page = await mountOverview();
    expect(chartTable(page.container, 'Turns per hour')).toHaveLength(24);
  });

  it('opens a latest turn in the drawer', async () => {
    const page = await mountOverview();
    const row = page.container.querySelector('a[href*="turn=t2"]');
    expect(row).not.toBeNull();
    await mountOverviewAt('/?turn=t2');
    await waitFor(() =>
      (document.querySelector('[role="dialog"]')?.textContent ?? '').includes('Stopped during decode.'),
    );
  });

  it('names Setup and Prefill in the phase bar of a measured turn', async () => {
    const page = await mountOverview();
    await waitFor(() => page.container.querySelector('[role="img"][aria-label^="Phases of the turn"]') !== null);
    const bar = page.container.querySelector('[role="img"][aria-label^="Phases of the turn"]');
    const label = bar?.getAttribute('aria-label') ?? '';
    expect(label).toContain('Setup');
    expect(label).toContain('Prefill');
    expect(label).not.toContain('To first token');
  });

  it('opens a KPI panel in place, with its footer link and its two extra requests only while open', async () => {
    const calls: string[] = [];
    const page = await mountOverview({}, '/', (call) => calls.push(call.path));
    const tile = byRole(page.container, 'group', 'Turns');
    expect(tile?.querySelectorAll('a')).toHaveLength(0);
    expect(tile?.querySelectorAll('button[aria-expanded="false"]')).toHaveLength(1);
    const before = calls.filter((path) => path.includes('groupBy=none')).length;
    expect(before).toBe(0);
    page.unmount();
    api?.();
    calls.length = 0;
    api = stubApi(routes(), { onCall: (call) => calls.push(call.path) });
    mounted = await renderPage(
      createElement(MemoryRouter, { initialEntries: ['/?detail=turns'] }, createElement(Overview)),
      () => byRole(document.body, 'region', 'Turns') !== null,
    );
    await waitFor(() => calls.filter((path) => path.includes('groupBy=none')).length === 2);
    const region = byRole(document.body, 'region', 'Turns');
    expect([...(region?.querySelectorAll('a') ?? [])].map((link) => link.getAttribute('href'))).toContain(
      '/metrics?range=24h&kpi=turns',
    );
  });
});

describe('Overview storage', () => {
  it('counts the cold-cache hit rate over the same 30 days as the Cache page', async () => {
    const page = await mountOverview({
      '/cache': cacheResponse({
        trend: [
          { t: dayStart(45), hits: 0, misses: 900, bytesWritten: 0, bytesRestored: 0 },
          { t: dayStart(0), hits: 90, misses: 10, bytesWritten: 0, bytesRestored: 0 },
        ],
      }),
    });
    const tile = (): string => byRole(page.container, 'group', 'Cold cache')?.textContent ?? '';
    await waitFor(() => tile().includes('hit rate'));
    expect(tile()).toContain('hit rate 90%');
  });

  it('prints the cold cache size with a small unit like every other tile', async () => {
    const page = await mountOverview();
    await waitFor(() => statValue(page.container, 'Cold cache') === '300 MB');
    const reading = byRole(page.container, 'group', 'Cold cache')?.querySelector('[data-stat-value]');
    expect(reading?.lastElementChild?.textContent).toBe('MB');
  });
});

describe('Overview last 24 hours', () => {
  it('marks a low sample only on the two speed tiles, not on Failed turns', async () => {
    const page = await mountOverview({
      '/metrics/summary': summary({ totals: busyTotals({ turns: 3, ok: 2, error: 1, cancelled: 0 }) }),
    });
    await waitFor(() => statValue(page.container, 'Failed turns') === '1');
    expect(byRole(page.container, 'group', 'Failed turns')?.textContent).not.toContain('n=3');
  });
});

describe('Overview latest turns', () => {
  /** Every observed box reports `width`: the table at the 720 window is about 610 px wide. */
  function observeAs(width: number): () => void {
    const original = globalThis.ResizeObserver;
    class Fixed {
      constructor(private readonly callback: ResizeObserverCallback) {}
      observe(target: Element): void {
        const entry = { target, contentRect: { width, height: 400 } } as unknown as ResizeObserverEntry;
        queueMicrotask(() => this.callback([entry], this as unknown as ResizeObserver));
      }
      unobserve(): void {}
      disconnect(): void {}
    }
    globalThis.ResizeObserver = Fixed as unknown as typeof ResizeObserver;
    return () => {
      globalThis.ResizeObserver = original;
    };
  }

  function headers(page: RenderedPage): string[] {
    const table = [...page.container.querySelectorAll('table')].find((node) =>
      (node.querySelector('caption')?.textContent ?? '').includes('Latest turns'),
    );
    return [...(table?.querySelectorAll('thead th') ?? [])].map((cell) => (cell.textContent ?? '').trim());
  }

  it('hides First token and Decode at the 720 window and shows them at 1180', async () => {
    const restore = observeAs(610);
    try {
      const page = await mountOverview();
      await waitFor(() => headers(page).includes('Tokens in → out') && !headers(page).includes('Decode'));
      expect(headers(page)).not.toContain('First token');
      expect(headers(page)).toContain('Phases');
    } finally {
      restore();
    }
    mounted?.unmount();
    api?.();
    const wide = observeAs(1100);
    try {
      const page = await mountOverview();
      await waitFor(() => headers(page).includes('Decode'));
      expect(headers(page)).toContain('First token');
      const table = [...page.container.querySelectorAll('table')].find((node) =>
        (node.querySelector('caption')?.textContent ?? '').includes('Latest turns'),
      )!;
      const cols = [...table.querySelectorAll('col')].map((col) => (col as HTMLElement).style.width);
      expect(cols.slice(0, 3)).toEqual(['116px', '', '96px']);
      const model = table.querySelector('tbody [data-model]')!.getAttribute('data-model')!;
      expect(table.querySelector(`tbody [title="${model}"]`)).not.toBeNull();
    } finally {
      wide();
    }
  });

  it('lists 2 turns in flight below the md page width and 3 above, and counts the rest', async () => {
    const four = live({
      processes: [
        process({ turns: ['a', 'b', 'c', 'd'].map((id, index) => liveTurn(id, { startedAt: 1_000 + index })) }),
      ],
    });
    const rows = (page: RenderedPage): number =>
      byRole(page.container, 'list', 'Turns in flight')?.querySelectorAll('li').length ?? 0;
    const restore = observeAs(700);
    try {
      const page = await mountOverview();
      await push(four, (text) => text.includes('+2 more'));
      expect(rows(page)).toBe(2);
    } finally {
      restore();
    }
    mounted?.unmount();
    api?.();
    const wide = observeAs(1100);
    try {
      const page = await mountOverview();
      await push(four, (text) => text.includes('+1 more'));
      expect(rows(page)).toBe(3);
    } finally {
      wide();
    }
  });
});

async function mountOverviewAt(url: string): Promise<void> {
  mounted?.unmount();
  api?.();
  api = stubApi(routes());
  mounted = await renderPage(createElement(MemoryRouter, { initialEntries: [url] }, createElement(Overview)), (text) =>
    text.includes('Latest turns'),
  );
}
