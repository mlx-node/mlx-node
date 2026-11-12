/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { MemoryRouter, useNavigate, type NavigateFunction } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { visibleColumns } from '../ui/src/components/arc-x/data-table/data-table.js';
import { ENGINE } from '../ui/src/lib/engine-strings.js';
import type { RuntimeLive, RuntimeSeries, SupervisorLive } from '../ui/src/lib/types.js';
import Engine from '../ui/src/pages/engine.js';
import { loadColumnsOf, procColumnsOf } from '../ui/src/pages/engine/history-section.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { liveModel, liveTurn, live, process, runtimeSeries, sched } from './helpers/observability-fixtures.js';
import {
  byLabel,
  byRole,
  chartTable,
  deferred,
  renderPage,
  STUB_FAILURE,
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

const SUPERVISOR: SupervisorLive = {
  state: 'lying',
  pid: 812,
  generation: 3,
  consecutiveCrashes: 2,
  lastExit: { verdict: 'crash', code: 1, signal: null, atMs: Date.now() - 60_000 },
  healthStatus: 'error',
  nativeErrorCount: 20,
  nativeErrorsCapped: true,
  lastNativeErrorContext: 'metal_kernel',
  lastNativeErrorAt: Date.now() - 30_000,
  writtenAt: Date.now(),
};

async function mountEngine(url = '/engine', routes: Record<string, unknown> = {}): Promise<RenderedPage> {
  api = stubApi({
    '/runtime/series': runtimeSeries(),
    '/models': { models: [], warnings: [], dir: '/models' },
    ...routes,
  });
  mounted = await renderPage(createElement(MemoryRouter, { initialEntries: [url] }, createElement(Engine)), (text) =>
    text.includes('History'),
  );
  return mounted;
}

/** Push a snapshot until the page shows `until` (the watch opens over the port a few turns after mount). */
async function push(snapshot: RuntimeLive, until: (text: string) => boolean): Promise<void> {
  const page = mounted;
  if (page === undefined || api === undefined) throw new Error('push: nothing mounted');
  await waitFor(() => {
    act(() => api?.emitLive({ type: 'runtime', snapshot }));
    return until(page.text());
  }, 2000);
}

const OWNER = 'qwen3.8-27b';

/** The owner's model: paged, limit 1, its turns never went through the scheduler. */
function ownerSeries(
  over: { allocatedBlocks?: number; turns?: number | null; rangeTurns?: number; unread?: boolean } = {},
): RuntimeSeries {
  const base = runtimeSeries();
  const entry = base.sched[0];
  const latest = sched({
    blockCapacity: 16_384,
    allocatedBlocks: over.allocatedBlocks ?? 0,
    freeBlocks: 16_384 - (over.allocatedBlocks ?? 0),
    reclaimableBlocks: 0,
    watermarkBlocks: 0,
    reservedBlocks: 0,
    memoryCapacityBytes: 16 * 1024 ** 3,
    maxBatchOccupancy: 0,
    turns: over.turns === undefined ? 9 : over.turns,
  });
  const zero = Object.fromEntries(Object.keys(entry.delta).map((key) => [key, 0])) as unknown as typeof entry.delta;
  return {
    ...base,
    sched: [
      {
        procKey: entry.procKey,
        model: OWNER,
        latest,
        snapshots: over.unread === true ? [] : [latest],
        delta: { ...zero, occupancy: [], turns: over.rangeTurns ?? 9 },
      },
    ],
  };
}

function ownerLive(model: Parameters<typeof liveModel>[1] = {}): RuntimeLive {
  return live({
    processes: [process({ models: [liveModel(OWNER, { maxConcurrent: 1, pool: null, ...model })], turns: [] })],
  });
}

function poolRegion(page: RenderedPage): HTMLElement | null {
  return byRole(page.container, 'region', 'Block pool and batching');
}

/** label -> value + hint text of the KeyValue list named `name`. */
function keyValueRows(scope: HTMLElement, name: string): Map<string, string> {
  const out = new Map<string, string>();
  const list = scope.querySelector(`dl[aria-label="${name}"]`);
  for (const dt of list?.querySelectorAll('dt') ?? []) {
    out.set(dt.textContent ?? '', dt.nextElementSibling?.textContent ?? '');
  }
  return out;
}

describe('Engine page', () => {
  it('shows one empty state when no process runs', async () => {
    const page = await mountEngine();
    await push(live({ status: 'stopped', processes: [] }), (text) => text.includes('Inference is not running'));
    expect(page.text()).toContain('Start it from the menu bar, or run mlx serve in a terminal.');
    expect(page.text()).not.toContain('Turns running in this process.');
  });

  it('prints the alert for each failing status', async () => {
    await mountEngine();
    await push(live({ status: 'failed', supervisor: { ...SUPERVISOR, state: 'failed' }, processes: [] }), (text) =>
      text.includes('Inference failed'),
    );
    expect(byRole(document.body, 'alert')?.textContent).toContain(
      'It stopped and was not restarted. Crashes in a row: 2.',
    );
    await push(
      live({ status: 'unresponsive', processes: [process({ status: 'unresponsive', ageMs: 42_000 })] }),
      (text) => text.includes('Inference is not responding'),
    );
    expect(mounted?.text()).toContain('No heartbeat for 42.0 s.');
    await push(live({ status: 'untrusted', supervisor: SUPERVISOR }), (text) =>
      text.includes('Output not trustworthy'),
    );
    expect(mounted?.text()).toContain('The process reported 20+ native errors, last in metal_kernel');
    // The Supervisor panel prints the same capped count.
    const panel = byRole(mounted!.container, 'region', 'Supervisor') ?? mounted!.container;
    expect(panel.textContent).toContain('20+');
    expect(mounted?.text()).toContain('not trustworthy');
  });

  it('follows the live snapshot in the Now items', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('Output rate42.7'));
    await push(live({ processes: [process({ decodeTps: 12.5 })] }), () =>
      statValue(page.container, 'Output rate').includes('12.5'),
    );
    expect(statValue(page.container, 'In flight')).toBe('1');
  });

  it('offers a process picker only with two live processes', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('Output rate'));
    expect(page.container.textContent).not.toContain('pid 812');
    await push(live({ processes: [process(), process({ pid: 913, port: 52115 })] }), (text) =>
      text.includes('desktop · :52114 · pid 812'),
    );
    expect(byLabel(page.container, 'Process')).not.toBeNull();
  });

  it('leaves a paused Output rate chart when another process is picked', async () => {
    let navigate: NavigateFunction | undefined;
    const Grab = (): null => {
      navigate = useNavigate();
      return null;
    };
    api = stubApi({ '/runtime/series': runtimeSeries(), '/models': { models: [], warnings: [], dir: '/models' } });
    mounted = await renderPage(
      createElement(MemoryRouter, { initialEntries: ['/engine?pid=812'] }, createElement(Grab), createElement(Engine)),
      (text) => text.includes('History'),
    );
    const page = mounted;
    const pair = (now: number) =>
      live({
        now,
        processes: [process({ pid: 812, decodeTps: 10 }), process({ pid: 913, port: 52115, decodeTps: 20 })],
      });
    const chart = (): Element | null => page.container.querySelector('figure[aria-label="Output rate"]');
    const value = (): string => chart()?.querySelector('[data-stat-value]')?.textContent ?? '';
    const pill = (): HTMLButtonElement | null => chart()?.querySelector('button[aria-pressed]') ?? null;
    const start = Date.now();
    await push(pair(start), () => value().includes('10.0'));
    await push(pair(start + 1500), () => value().includes('10.0'));
    act(() => pill()?.click());
    expect(chart()?.getAttribute('data-state')).toBe('paused');
    act(() => void navigate?.('/engine?pid=913'));
    await push(pair(start + 3000), (text) => text.includes('pid 913'));
    await push(pair(start + 4500), () => value().includes('20.0'));
    expect(pill()?.getAttribute('aria-pressed')).toBe('false');
    expect(pill()?.textContent).toBe('Live');
    expect(chart()?.getAttribute('data-state')).toBe('live');
  });

  it('explains a flat model and an unread pool', async () => {
    const empty = runtimeSeries({ sched: [] });
    const page = await mountEngine('/engine', { '/runtime/series': empty });
    await push(
      live({ processes: [process({ models: [liveModel('flat-model', { paged: false, pool: null })] })] }),
      (text) => text.includes('flat-model keeps no block pool'),
    );
    expect(poolRegion(page)?.textContent).toContain(
      'flat-model keeps no block pool. It runs one turn at a time. See each turn in Metrics.',
    );
    await push(live({ processes: [process({ models: [liveModel('paged-model', { pool: null })] })] }), (text) =>
      text.includes('There is no reading yet.'),
    );
    expect(page.text()).toContain('The pool is read when the model is idle. There is no reading yet.');
  });

  it('a model whose turns ran alone shows the pool as rows and one Batching note, not empty cards', async () => {
    const page = await mountEngine('/engine', { '/runtime/series': ownerSeries() });
    await push(ownerLive(), (text) => text.includes('turns ran in this range'));
    const region = poolRegion(page)!;
    const text = region.textContent ?? '';
    expect(region.querySelector('[role="img"]')).toBeNull();
    expect(text).not.toContain('Decode batch size');
    expect(text).not.toContain('Scheduler counters');
    const rows = keyValueRows(region, 'KV block pool');
    expect(rows.get('Capacity')).toBe('16,384 blocks16.0 GB');
    expect(rows.get('Held at rest')?.startsWith('0 blocks')).toBe(true);
    expect(rows.get('Held at rest')?.endsWith(' · after 9 turns')).toBe(true);
    expect(text).toContain(ENGINE.pool.noneHeld);
    expect(text).toContain('This model runs one turn at a time.');
    expect(text).toContain('9 turns ran in this range.');
    expect(text).toContain(ENGINE.pool.noScheduler);
    expect(region.querySelector('a[href="/metrics"]')?.textContent).toBe('Metrics');
  });

  it('a model with room for more turns whose turns ran alone does not say it runs one at a time', async () => {
    const page = await mountEngine('/engine', { '/runtime/series': ownerSeries() });
    await push(ownerLive({ maxConcurrent: 4 }), (text) => text.includes('turns ran in this range'));
    const text = poolRegion(page)?.textContent ?? '';
    expect(text).toContain(ENGINE.pool.eachAlone);
    expect(text).not.toContain(ENGINE.pool.oneAtATime);
  });

  it('a model that ran alone with blocks held draws the grid and the turn count, without the note', async () => {
    const page = await mountEngine('/engine', { '/runtime/series': ownerSeries({ allocatedBlocks: 1200 }) });
    await push(ownerLive(), (text) => text.includes('turns ran in this range'));
    const region = poolRegion(page)!;
    expect(region.querySelector('[role="img"]')).not.toBeNull();
    expect(region.textContent).toContain('1,200 of 16,384 blocks held · at rest · ');
    expect(region.textContent).toContain(' · after 9 turns');
    expect(region.textContent).not.toContain(ENGINE.pool.noneHeld);
  });

  it('a model with no turn in the range says why, in four cases', async () => {
    const cases: Array<[Parameters<typeof ownerSeries>[0], Parameters<typeof liveModel>[1], string]> = [
      [{ rangeTurns: 0 }, { inFlight: 1 }, ENGINE.pool.quietBusy],
      [{ rangeTurns: 0, turns: 0 }, {}, ENGINE.pool.quietNever],
      [{ rangeTurns: 0 }, {}, ENGINE.pool.quietRange],
      [{ rangeTurns: 0, unread: true }, {}, ENGINE.pool.quietUnread],
      [{ rangeTurns: 0, turns: 0, unread: true }, {}, ENGINE.pool.quietUnread],
    ];
    for (const [series, model, words] of cases) {
      const page = await mountEngine('/engine', { '/runtime/series': ownerSeries(series) });
      await push(ownerLive(model), (text) => text.includes(words));
      const text = poolRegion(page)?.textContent ?? '';
      expect(text).not.toContain('turns ran in this range');
      expect(text).not.toContain(ENGINE.pool.noneHeld);
      expect(text).not.toContain('Scheduler counters');
      for (const other of [
        ENGINE.pool.quietBusy,
        ENGINE.pool.quietNever,
        ENGINE.pool.quietRange,
        ENGINE.pool.quietUnread,
      ]) {
        if (other !== words) expect(text).not.toContain(other);
      }
      mounted?.unmount();
      mounted = undefined;
      api?.();
      api = undefined;
    }
  });

  it('a batched model keeps the three cards and counts its turns', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('largest since load 4'));
    const region = poolRegion(page)!;
    expect(region.textContent).toContain('Scheduler counters');
    expect(region.textContent).toContain(ENGINE.pool.countersDescription);
    const counters = keyValueRows(region, 'Scheduler counters');
    expect([...counters.keys()][0]).toBe('Turns');
    expect(counters.get('Turns')).toBe('415all lanes');
    expect(region.textContent).toContain(' · after 430 turns');
    expect(region.textContent).not.toContain(ENGINE.pool.batchingTitle);
  });

  it('lists every loaded model in the switch, so a flat model can be chosen while another has records', async () => {
    const page = await mountEngine('/engine?model=flat-model');
    await push(
      live({
        processes: [
          process({ models: [liveModel('qwen3-8b'), liveModel('flat-model', { paged: false, pool: null })] }),
        ],
      }),
      (text) => text.includes('flat-model keeps no block pool'),
    );
    const region = poolRegion(page)!;
    const options = [...(byRole(region, 'group', 'Model')?.querySelectorAll('button') ?? [])].map(
      (button) => button.textContent,
    );
    expect(options).toEqual(['qwen3-8b', 'flat-model']);
  });

  it('names the memory row by the pool limit and keeps no promise of a later reading', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('Block pool capacity'));
    expect(page.text()).toContain('limit of 1 paged model');
    expect(page.text()).not.toContain('Block pools at rest');
    for (const words of Object.values(ENGINE.pool)) {
      expect(words.includes('next idle moment')).toBe(false);
      expect(words.includes('busy period')).toBe(false);
    }
  });

  it('holds the pool region in its loading state, then its error state, never the empty one', async () => {
    const held = deferred(runtimeSeries());
    const page = await mountEngine('/engine', { '/runtime/series': held.body });
    await push(live(), (text) => text.includes('Output rate'));
    const pool = (): HTMLElement | null => byRole(page.container, 'region', 'Block pool and batching');
    expect(pool()?.querySelector('[aria-busy="true"]')).not.toBeNull();
    expect(pool()?.textContent).not.toContain(ENGINE.pool.unread);
    act(() => held.release());
    await waitFor(() => (pool()?.textContent ?? '').includes('1,716 of 4,096 blocks held'));
    mounted?.unmount();
    mounted = undefined;
    api?.();
    api = undefined;
    const failed = await mountEngine('/engine', { '/runtime/series': STUB_FAILURE });
    await push(live(), (text) => text.includes('Output rate'));
    const region = (): HTMLElement | null => byRole(failed.container, 'region', 'Block pool and batching');
    await waitFor(() => (region()?.textContent ?? '').includes('Try again'));
    expect(region()?.textContent).not.toContain(ENGINE.pool.unread);
  });

  it('keeps the pool pressure badge and its count on one line', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('Pool pressure'));
    const counters = byRole(page.container, 'region', 'Block pool and batching');
    const badge = [...(counters?.querySelectorAll('dd') ?? [])].find((dd) =>
      (dd.textContent ?? '').includes('Pool pressure'),
    );
    const line = badge?.querySelector('span.inline-flex');
    expect(line?.className.split(' ')).toContain('whitespace-nowrap');
    expect(line?.className.split(' ')).not.toContain('flex-wrap');
    expect(line?.textContent).toBe('Pool pressure3');
  });

  it('charts decode steps per batch size and reads the high-water mark as since load', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('largest since load 4'));
    expect(chartTable(page.container, 'Decode batch size').map((row) => row.slice(0, 2))).toEqual([
      ['Batch of 1', '100'],
      ['Batch of 2', '300'],
      ['Batch of 4', '100'],
    ]);
    expect(page.text()).toContain('mean batch 2.2 · largest since load 4 · 500 steps');
    expect(page.text()).toContain('1,716 of 4,096 blocks held · at rest');
  });

  it('lets a long hint shrink before a numeric value in the Memory rows (happy-dom has no layout: the rules are read)', () => {
    const rules = cssRules('components/arc-x/key-value/key-value.module.css');
    expect(cssValue(rules, ".root[data-layout='rows'] .value[data-numeric] .text", 'flex')).toBe('none');
    expect(cssValue(rules, ".root[data-layout='rows'] .hint", 'flex')).toBe('0 1 auto');
    expect(cssValue(rules, ".root[data-layout='rows'] .hint", 'text-overflow')).toBe('ellipsis');
  });

  it('gives the rows of a list one shared label column, so the values start at the same x (the rules are read)', () => {
    const rules = cssRules('components/arc-x/key-value/key-value.module.css');
    expect(cssValue(rules, ".root[data-layout='rows']", 'grid-template-columns')).toBe(
      'fit-content(50%) minmax(0, 1fr)',
    );
    expect(cssValue(rules, ".root[data-layout='rows'] > .row", 'grid-template-columns')).toBe('subgrid');
    expect(cssValue(rules, ".root[data-layout='rows'] > .row", 'grid-column')).toBe('1 / -1');
  });

  it('prints the memory unit once: plain axis numbers under GB, no low-sample rings on gauges', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('Memory over time'));
    const figure = (): Element | null => byRole(page.container, 'figure', 'Memory over time');
    await waitFor(() => (figure()?.querySelectorAll('[class*="gutter"] > span').length ?? 0) > 2);
    const ticks = [...(figure()?.querySelectorAll('[class*="gutter"] > span') ?? [])].map(
      (node) => node.textContent ?? '',
    );
    for (const tick of ticks) {
      expect(tick).not.toContain('B');
    }
    const rows = chartTable(page.container, 'Memory over time');
    const filled = rows.find((row) => row[1] !== 'no data');
    expect(filled?.slice(1)).toEqual(['18', '2', '30']);
    expect(rows.flat().some((cell) => cell.includes('n='))).toBe(false);
  });

  it('sums the history tiles', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('observed'));
    expect(statValue(page.container, 'Rejected requests')).toBe('2');
    expect(statValue(page.container, 'Busy time')).toBe('50%');
    expect(page.text()).toContain('30.0 s of 1m 00s observed');
    expect(statValue(page.container, 'Peak in flight')).toBe('2');
    expect(statValue(page.container, 'Model loads')).toBe('1');
    expect(page.text()).toContain('Queue full');
    const tile = byRole(page.container, 'group', 'Rejected requests');
    const hint = [...(tile?.querySelectorAll('[title]') ?? [])].find((node) =>
      (node.getAttribute('title') ?? '').startsWith('HTTP 429'),
    );
    expect(hint?.textContent).toContain('Queue full');
    expect(hint?.className.split(' ')).not.toContain('flex-wrap');
    expect(hint?.querySelector('.truncate')?.textContent).toBe(hint?.getAttribute('title'));
  });

  it('prints the alias the client used after the checkpoint badge of an in-flight turn', async () => {
    const page = await mountEngine();
    await push(
      live({
        processes: [
          process({
            turns: [
              liveTurn('a', { model: 'claude-haiku-x', modelId: 'qwen3-8b' }),
              liveTurn('b', { model: 'qwen3-8b', modelId: 'qwen3-8b' }),
            ],
          }),
        ],
      }),
      (text) => text.includes('claude-haiku-x'),
    );
    const rows = [...(byRole(page.container, 'list', 'Turns in flight')?.querySelectorAll('li') ?? [])];
    expect(rows.map((li) => li.querySelector('[title="claude-haiku-x"]') !== null)).toEqual([true, false]);
    expect(rows[0].getAttribute('aria-label')?.startsWith('qwen3-8b,')).toBe(true);
  });

  it('keeps the Loaded cell of the models table on one line in a column wide enough for it, the full text in its title', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('1h ago'));
    const table = [...page.container.querySelectorAll('table')].find((node) =>
      [...node.querySelectorAll('thead th')].some((th) => (th.textContent ?? '').trim() === 'Loaded'),
    );
    const heads = [...(table?.querySelectorAll('thead th') ?? [])];
    const index = heads.findIndex((th) => (th.textContent ?? '').trim() === 'Loaded');
    expect(table?.querySelectorAll('colgroup col')[index]?.getAttribute('style')).toContain('width: 116px');
    const cell = table?.querySelector('tbody tr')?.children[index];
    const line = cell?.querySelector('span');
    expect(line?.className.split(' ')).toContain('whitespace-nowrap');
    expect(line?.getAttribute('title')).toBe('1h ago · 8.20 s');
  });

  it('prints the quant on the caption line under the model name, not beside it', async () => {
    const local = {
      name: 'Qwen3-8B-4bit',
      path: '/models/Qwen3-8B-4bit',
      modelType: 'qwen3',
      quant: 'affine-4bit',
      contextWindow: 40_960,
      sizeBytes: 15 * 1024 ** 3,
      fileCount: 1,
      servedIds: ['qwen3-8b'],
    };
    const page = await mountEngine('/engine', { '/models': { models: [local], warnings: [], dir: '/models' } });
    await push(live(), (text) => text.includes('affine-4bit · 15.0 GB on disk'));
    const caption = [...page.container.querySelectorAll('span')].find(
      (node) => node.textContent === 'affine-4bit · 15.0 GB on disk',
    );
    expect(caption?.previousElementSibling?.getAttribute('data-model')).toBe('qwen3-8b');
    expect(caption?.previousElementSibling?.textContent).toBe('qwen3-8b');
  });

  it('the history tables drop columns by priority as the card narrows, and every shown column fits', () => {
    const ids = (columns: Parameters<typeof visibleColumns>[0], width: number): string[] =>
      visibleColumns(columns, width).map((column) => (column as { id: string }).id);
    const loads = loadColumnsOf();
    const procs = procColumnsOf(new Map());
    expect([360, 440, 560, 720].map((width) => ids(loads, width))).toEqual([
      ['when', 'model', 'result'],
      ['when', 'model', 'result', 'took'],
      ['when', 'model', 'result', 'took', 'kind'],
      ['when', 'model', 'result', 'took', 'kind'],
    ]);
    expect([360, 440, 560, 720].map((width) => ids(procs, width))).toEqual([
      ['started', 'ran', 'state'],
      ['started', 'ran', 'state', 'host'],
      ['started', 'ran', 'state', 'host', 'version'],
      ['started', 'ran', 'state', 'host', 'version', 'chip'],
    ]);
    const need = (columns: ReadonlyArray<{ flex?: true; minWidth?: number; width?: number | string }>): number =>
      columns.reduce(
        (sum, column) =>
          sum + (column.flex === true ? (column.minWidth ?? 120) : typeof column.width === 'number' ? column.width : 0),
        0,
      );
    for (const width of [360, 440, 560, 720]) {
      expect(need(visibleColumns(loads, width))).toBeLessThanOrEqual(width);
      expect(need(visibleColumns(procs, width))).toBeLessThanOrEqual(width);
    }
    expect(loads.filter((column) => column.flex === true).map((column) => column.id)).toEqual(['model']);
  });

  it('prints a long model name whole in its load row with the name as its title; a process with no exit says so', async () => {
    const long = 'qwen3.6-35b-a3b-agentworld-instruct-ud-q4-k-xl-mlx-dwq-8bits';
    const base = runtimeSeries();
    const series = { ...base, loads: base.loads.map((load) => ({ ...load, model: long })) };
    const page = await mountEngine('/engine', { '/runtime/series': series });
    await waitFor(() => page.text().includes('No exit record'));
    expect(page.text()).not.toContain('Ended without exit');
    const cell = page.container.querySelector(`table [title="${long}"]`);
    expect([...cell!.querySelectorAll('[aria-hidden="true"]')].map((n) => n.textContent).join('')).toBe(long);
  });

  it('has exactly one polite live region', async () => {
    const page = await mountEngine();
    await push(live(), (text) => text.includes('Output rate'));
    await waitFor(() =>
      (page.container.querySelector('[aria-live]')?.textContent ?? '').includes('Inference running.'),
    );
    const regions = [...page.container.querySelectorAll('[aria-live]')].filter(
      (node) => node.getAttribute('aria-live') !== 'off',
    );
    expect(regions).toHaveLength(1);
    expect(regions[0].getAttribute('aria-live')).toBe('polite');
    expect(regions[0].textContent).toContain('1 turn in flight. 42.7 tokens per second. Memory 60% of the limit.');
  });
});
