/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { WHOLE_PROMPT_CAPTURE_BLOCKS as SERVER_WHOLE_PROMPT_BLOCKS } from '../../server/src/host/persist-cache.js';
import type { ApiCall } from '../src/runtime.js';
import { toast } from '../ui/src/lib/toast.js';
import type { RuntimeLive } from '../ui/src/lib/types.js';
import Cache from '../ui/src/pages/cache.js';
import { WHOLE_PROMPT_CAPTURE_BLOCKS } from '../ui/src/pages/cache/disk-cache-line.js';
import { cacheResponse, dayStart, live, MIB, process } from './helpers/observability-fixtures.js';
import {
  byRole,
  chartTable,
  renderPage,
  sequence,
  statValue,
  STUB_FAILURE,
  stubApi,
  waitFor,
  type RenderedPage,
  type StubbedApi,
} from './render.js';

let mounted: RenderedPage | undefined;
let restoreApi: StubbedApi | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

async function mountCache(routes: Record<string, unknown>, onCall?: (call: ApiCall) => void): Promise<RenderedPage> {
  restoreApi = stubApi(routes, { onCall });
  mounted = await renderPage(createElement(MemoryRouter, null, createElement(Cache)), (text) =>
    text.includes('mlx-paged-v1'),
  );
  return mounted;
}

function click(element: Element | null): void {
  if (element === null) throw new Error('click: no element');
  act(() => {
    (element as HTMLElement).click();
  });
}

function buttonNamed(scope: ParentNode, name: string): HTMLButtonElement | null {
  return [...scope.querySelectorAll('button')].find((button) => (button.textContent ?? '').trim() === name) ?? null;
}

function dialog(): HTMLElement | null {
  const all = document.querySelectorAll<HTMLElement>('[role="dialog"]');
  return all.length === 0 ? null : all[all.length - 1];
}

/** The `[kept, would be evicted]` cells per age bucket. */
function ageRows(page: RenderedPage): string[][] {
  return chartTable(page.container, 'Objects by age').map((row) => row.slice(1, 3));
}

/** Push a snapshot until the disk cache line shows `state` (the watch opens a few turns after mount). */
async function pushLive(page: RenderedPage, snapshot: RuntimeLive, state: string): Promise<HTMLElement | null> {
  await waitFor(() => {
    act(() => restoreApi?.emitLive({ type: 'runtime', snapshot }));
    return page.container.querySelector(`[data-disk-cache="${state}"]`) !== null;
  }, 2000);
  return page.container.querySelector<HTMLElement>(`[data-disk-cache="${state}"]`);
}

describe('Cache page disk cache line', () => {
  const turns = 'Last 30 days, all cache folders: 412 turns with it on, 3 off';

  it('says the switch of the running engine, the most one turn may capture, and the turns that ran with the tier on and off', async () => {
    const page = await mountCache({
      '/cache': cacheResponse({ scope: { ...cacheResponse().scope, enabledTurns: 412, disabledTurns: 3 } }),
    });
    const on = await pushLive(page, live(), 'on');
    expect(on?.textContent).toBe(`Disk cache:On·Per turn: up to the whole prompt, 10.0 s cap·${turns}`);
    const native = await pushLive(
      page,
      live({ processes: [process({ pid: 900, coldCapture: { blocks: 128, ms: 250 } })] }),
      'on',
    );
    await waitFor(() => (native?.textContent ?? '').includes('Per turn: up to 128 prefix blocks, 250 ms cap'));
    const off = await pushLive(
      page,
      live({ processes: [process({ persistPagedCache: false, coldCapture: null })] }),
      'off',
    );
    expect(off?.textContent).toBe(`Disk cache:Off·${turns}`);
    const older = await pushLive(
      page,
      live({ processes: [process({ persistPagedCache: null, coldCapture: null })] }),
      'unknown',
    );
    expect(older?.textContent).toBe(`Disk cache:Unknown(engine started before this field)·${turns}`);
    const none = await pushLive(page, live({ status: 'stopped', processes: [] }), 'no-engine');
    expect(none?.textContent).toContain('No engine(inference is not running)');
  });

  it('names the engine it reads when more than one runs', async () => {
    const page = await mountCache({
      '/cache': cacheResponse({ scope: { ...cacheResponse().scope, enabledTurns: 412, disabledTurns: 3 } }),
    });
    const two = live({
      processes: [
        process({ pid: 4242 }),
        process({ pid: 77, host: 'serve', persistPagedCache: false, coldCapture: null }),
      ],
    });
    const line = await pushLive(page, two, 'on');
    expect(line?.textContent).toBe(
      `Disk cache:On(desktop engine, pid 4242)·Per turn: up to the whole prompt, 10.0 s cap·${turns}`,
    );
  });

  it('puts each dot at the end of the item before it, so a wrapped line never starts with one', async () => {
    const page = await mountCache({
      '/cache': cacheResponse({ scope: { ...cacheResponse().scope, enabledTurns: 412, disabledTurns: 3 } }),
    });
    const items = (line: HTMLElement | null): string[] => [...(line?.children ?? [])].map((i) => i.textContent ?? '');
    const on = await pushLive(page, live(), 'on');
    expect(items(on)).toEqual(['Disk cache:', 'On·', 'Per turn: up to the whole prompt, 10.0 s cap·', turns]);
    const older = await pushLive(
      page,
      live({ processes: [process({ persistPagedCache: null, coldCapture: null })] }),
      'unknown',
    );
    expect(items(older)).toEqual(['Disk cache:', 'Unknown(engine started before this field)·', turns]);
  });

  it('reads the whole-prompt cap the server writes', () => {
    expect(WHOLE_PROMPT_CAPTURE_BLOCKS).toBe(SERVER_WHOLE_PROMPT_BLOCKS);
  });
});

describe('Cache page', () => {
  it('announces the eviction preview after a threshold change, not after a refetch', async () => {
    const grown = cacheResponse();
    grown.disk.ageHistogram = grown.disk.ageHistogram.map((bucket) => ({ ...bucket, count: bucket.count + 1 }));
    const page = await mountCache({ '/cache': sequence(cacheResponse(), grown) });
    const live = (): Element[] => [...page.container.querySelectorAll('[aria-live="polite"]')];
    expect(live().some((node) => (node.textContent ?? '').includes('would be removed'))).toBe(false);
    act(() => restoreApi?.emitLive({ type: 'data', version: 1, traces: true, runtime: false }));
    await waitFor(() => page.text().includes('10 objects · 50.0 MB would be removed'), 3000);
    expect(live().some((node) => (node.textContent ?? '').includes('would be removed'))).toBe(false);
    click(byRole(page.container, 'button', '1 day'));
    await waitFor(() =>
      live().some((node) => (node.textContent ?? '').includes('41 objects · 200 MB would be removed')),
    );
  });

  it('moves buckets between Kept and Would be evicted when the threshold changes', async () => {
    const page = await mountCache({ '/cache': cacheResponse() });
    expect(ageRows(page)).toEqual([
      ['100', '0'],
      ['30', '0'],
      ['0', '6'],
      ['0', '2'],
    ]);
    expect(page.text()).toContain('8 objects · 50.0 MB would be removed');
    click(byRole(page.container, 'button', '1 day'));
    await waitFor(() => page.text().includes('38 objects · 200 MB would be removed'));
    expect(ageRows(page)).toEqual([
      ['100', '0'],
      ['0', '30'],
      ['0', '6'],
      ['0', '2'],
    ]);
  });

  it('keeps an alarmed health hint on one cut line with its full text in the title', async () => {
    const base = cacheResponse();
    const page = await mountCache({
      '/cache': cacheResponse({ health: { ...base.health, queueDrops: 3, queueDropsTotal: 114 } }),
    });
    const tile = byRole(page.container, 'group', 'Queue drops');
    const hint = [...(tile?.querySelectorAll('[title]') ?? [])].find((node) =>
      (node.getAttribute('title') ?? '').startsWith('cumulative max 114'),
    );
    expect(hint?.textContent).toContain('Check');
    expect(hint?.className.split(' ')).not.toContain('flex-wrap');
    const text = hint?.querySelector('.truncate');
    expect(text?.textContent).toBe(hint?.getAttribute('title'));
  });

  it('switches the age chart to bytes', async () => {
    const page = await mountCache({ '/cache': cacheResponse() });
    click(byRole(page.container, 'button', 'Bytes'));
    await waitFor(() => page.text().includes('Prefix blocks and state sidecars · MB'));
    expect(chartTable(page.container, 'Objects by age')[1].slice(1, 3)).toEqual(['150', '0']);
  });

  it('prints no data, not 0 %, for a day without lookups in the hit-rate view', async () => {
    const page = await mountCache({ '/cache': cacheResponse() });
    click(byRole(page.container, 'button', 'Hit rate'));
    await waitFor(() => page.text().includes('Hits ÷ lookups per day'));
    const rows = chartTable(page.container, 'Cache hit rate per day');
    expect(rows).toHaveLength(30);
    expect(rows[0][1]).toBe('no data');
    expect(rows[27][1]).toBe('>99');
  });

  it('reads a hit rate with its percent sign once', async () => {
    const page = await mountCache({
      '/cache': cacheResponse({
        trend: [{ t: dayStart(0), hits: 91, misses: 9, bytesWritten: MIB, bytesRestored: MIB }],
      }),
    });
    click(byRole(page.container, 'button', 'Hit rate'));
    await waitFor(() => page.text().includes('Hits ÷ lookups per day'));
    const slider = byRole(page.container, 'figure', 'Cache hit rate per day')?.querySelector('[role="slider"]');
    const reading = slider?.getAttribute('aria-valuetext') ?? '';
    expect(reading).toContain(': 91%');
    expect(reading).not.toContain('%%');
    expect(reading).not.toContain('% %');
  });

  it('points an empty trend at the turns that ran with the tier off', async () => {
    const base = cacheResponse();
    const page = await mountCache({
      '/cache': { ...base, trend: [], scope: { ...base.scope, disabledTurns: 4 } },
    });
    expect(page.text()).toContain('No hits or misses recorded for this cache.');
    expect(page.text()).toContain('4 turns ran with the cold tier off, so nothing was looked up.');
  });

  it('keeps the confirm open with the error when the eviction fails', async () => {
    const page = await mountCache({ '/cache': sequence(cacheResponse(), STUB_FAILURE, cacheResponse()) });
    click(buttonNamed(page.container, 'Evict older'));
    await waitFor(() => dialog() !== null);
    expect(dialog()?.textContent).toContain('Evict old cache objects?');
    click(buttonNamed(dialog() as HTMLElement, 'Evict'));
    await waitFor(() =>
      (dialog()?.querySelector('[role="alert"]')?.textContent ?? '').includes('Failed to update cache'),
    );
    expect(dialog()).not.toBeNull();
  });

  it('toasts the result and reloads after a clear', async () => {
    const success = vi.spyOn(toast, 'success');
    const calls: string[] = [];
    const emptied = cacheResponse();
    emptied.disk = {
      ...emptied.disk,
      entryCount: 0,
      sidecarCount: 0,
      totalBytes: 0,
      sidecarBytes: 0,
      ageHistogram: emptied.disk.ageHistogram.map((bucket) => ({ ...bucket, count: 0, bytes: 0 })),
    };
    const page = await mountCache(
      { '/cache': sequence(cacheResponse(), { removed: 138, freedBytes: 300 * MIB }, emptied) },
      (call) => calls.push(`${call.method} ${call.path}`),
    );
    click(buttonNamed(page.container, 'Clear all'));
    await waitFor(() => dialog() !== null);
    expect(dialog()?.textContent).toContain('Removes all 138 objects');
    click(buttonNamed(dialog() as HTMLElement, 'Clear all'));
    await waitFor(() => success.mock.calls.length > 0, 2000);
    expect(success.mock.calls[0][0]).toBe('Cleared cold cache');
    expect(success.mock.calls[0][1]).toEqual({ description: '138 objects removed · 300 MB freed' });
    await waitFor(() => page.text().includes('The tier is empty.'));
    expect(calls.filter((call) => call.startsWith('GET')).length).toBeGreaterThanOrEqual(2);
  });

  it('counts the tiles over the same 30 days as the chart footnotes', async () => {
    const page = await mountCache({
      '/cache': cacheResponse({
        trend: [
          { t: dayStart(45), hits: 50, misses: 50, bytesWritten: 100 * MIB, bytesRestored: 100 * MIB },
          { t: dayStart(0), hits: 3, misses: 1, bytesWritten: MIB, bytesRestored: 2 * MIB },
        ],
      }),
    });
    await waitFor(() => statValue(page.container, 'Bytes restored') !== '');
    expect(statValue(page.container, 'Bytes restored')).toBe('2.0 MB');
    expect(page.text()).toContain('1.0 MB written · 2.0 MB restored');
    expect(statValue(page.container, 'Hit rate')).toBe('75%');
    expect(page.text()).toContain('3 hits · 1 miss · this cache only');
  });

  it('joins trend days onto the last 30 local days', async () => {
    const page = await mountCache({
      '/cache': cacheResponse({
        trend: [{ t: dayStart(0), hits: 3, misses: 1, bytesWritten: MIB, bytesRestored: 2 * MIB }],
      }),
    });
    const rows = chartTable(page.container, 'Cache hit / miss trend');
    expect(rows).toHaveLength(30);
    expect(rows[29].slice(1, 3)).toEqual(['3', '1']);
    expect(rows[0].slice(1, 3)).toEqual(['0', '0']);
  });
});
