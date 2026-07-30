/**
 * @vitest-environment happy-dom
 */

/**
 * Structural guards for the loading → loaded swap on the Cache, Session detail
 * and Metrics pages.
 *
 * A skeleton is only doing its job if it occupies the SAME box as the content it
 * stands in for; otherwise every request that lands nudges the page and the whole
 * dashboard reads as jittery. The mismatch is invisible to a text assertion —
 * both states render, both say the right words — and it is equally invisible to a
 * pixel assertion here, because happy-dom performs no layout and answers 0 for
 * every `getBoundingClientRect`.
 *
 * So these tests assert the STRUCTURE that makes the two boxes equal: the bar
 * lives inside the very element that carries the font, sized `h-[1lh]` (one line
 * box of that element's own line-height), and the loading tree wraps it in the
 * same markup the loaded tree will. Class-for-class equality between the two
 * renders is the check; a bar that escapes its typographic wrapper, or one given
 * a hand-picked pixel height again, fails it.
 */

import { MessageChannel } from 'node:worker_threads';

import { categoryChartHeight } from '@/lib/chart';
import type {
  CacheResponse,
  MetricsOverviewResponse,
  SessionDetailResponse,
  SessionMetricsResponse,
} from '@/lib/types';
import Cache from '@/pages/cache';
import Metrics from '@/pages/metrics';
import SessionDetail from '@/pages/session-detail';
import { createElement, type ReactElement } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it } from 'vite-plus/test';

import { bindEventTargetPort } from '../src/rpc/port.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import { renderPage, stubApi, stubChartMetrics, type RenderedPage } from './render.js';

const MIB = 1024 * 1024;

/**
 * Connect the SPA to a port whose peer never answers.
 *
 * This is the only honest way to observe a loading tree. `renderPage` flushes
 * until its condition holds, and a stubbed route answers on the very next
 * macrotask, so anything that raced the reply would be asserting on whichever
 * state happened to win — green on a fast machine, red on a loaded one. With
 * nobody serving the far end every `useJson` stays `loading: true` for as long as
 * the test looks at it (the client's own 30 s deadline is the only thing that
 * would ever settle it, and the disposer clears that timer).
 *
 * `connectDashboardApi` also drops `json-cache`, which is what keeps a loading
 * state reachable after an earlier test in this file has already loaded the same
 * path: `useJson` seeds itself from that cache and would otherwise render the
 * previous body instead of a skeleton.
 */
function stubUnansweredApi(): () => void {
  const { port1, port2 } = new MessageChannel();
  port1.unref();
  port2.unref();
  connectDashboardApi(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
  return () => {
    disconnectDashboardApi();
    port1.close();
    port2.close();
  };
}

/** Mount `element` with nothing answering, read the loading tree, tear it down. */
async function whileLoading<T>(
  element: ReactElement,
  renderedWhen: string,
  read: (page: RenderedPage) => T,
): Promise<T> {
  const restore = stubUnansweredApi();
  try {
    const page = await renderPage(element, (text) => text.includes(renderedWhen));
    try {
      return read(page);
    } finally {
      page.unmount();
    }
  } finally {
    restore();
  }
}

/** Mount `element` against `routes`, read the settled tree, tear it down. */
async function whenLoaded<T>(
  element: ReactElement,
  routes: Record<string, unknown>,
  settledWhen: string,
  read: (page: RenderedPage) => T,
  options: { charts?: boolean } = {},
): Promise<T> {
  const restoreCharts = options.charts === true ? stubChartMetrics() : undefined;
  const restore = stubApi(routes);
  try {
    const page = await renderPage(element, (text) => text.includes(settledWhen));
    try {
      return read(page);
    } finally {
      page.unmount();
    }
  } finally {
    restore();
    restoreCharts?.();
  }
}

/** `class` of every match, in document order. */
function classesOf(page: RenderedPage, selector: string): string[] {
  return [...page.container.querySelectorAll(selector)].map((node) => node.getAttribute('class') ?? '');
}

/** Whether every listed element carries `utility` as a whole class. */
function allCarry(classes: string[], utility: string): boolean {
  return classes.length > 0 && classes.every((value) => value.split(/\s+/).includes(utility));
}

function cacheFixture(): CacheResponse {
  return {
    disk: {
      root: '/tmp/cold/mlx-paged-v1',
      exists: true,
      entryCount: 136,
      sidecarCount: 2,
      totalBytes: 300 * MIB,
      sidecarBytes: 100 * MIB,
      quotaBytes: 1024 * MIB,
      oldestMtime: Date.now() - 3 * 86_400_000,
      newestMtime: Date.now() - 3_600_000,
      ageHistogram: [{ label: '<1d', count: 138, bytes: 300 * MIB }],
    },
    trend: [{ day: '2026-07-20', hits: 16_000, misses: 40, bytesWritten: 8 * MIB, bytesRestored: 64 * MIB }],
    scope: {
      root: '/private/tmp/cold/mlx-paged-v1',
      trendWindowDays: 30,
      legacy: { turns: 0, hits: 0, misses: 0 },
      otherRoots: { turns: 0, hits: 0, misses: 0 },
      unattributed: { turns: 0, hits: 0, misses: 0 },
      disabledTurns: 0,
      unrootedSidecarCaptures: 0,
    },
    health: {
      // Every counter healthy on purpose. `alarm` tints a value `text-destructive`,
      // and that class is the one legitimate difference between the two renders —
      // it changes the ink, never the box — so a fixture that raises an alarm
      // would make a class-for-class comparison assert about colour instead of
      // about geometry.
      enqueued: 500,
      queueDrops: 0,
      evictions: 4,
      corruptions: 0,
      corruptionsTotal: 0,
      queueDropsTotal: 0,
      writeErrors: 0,
      writeErrorsTotal: 0,
      restoreDeclines: 0,
      restoreSuppressed: 0,
      sidecarCaptureReached: 20,
      sidecarChainEmpty: 1,
      sidecarBoundarySkips: 2,
      sidecarAlreadyPersisted: 16,
      sidecarEnqueued: 1,
      sidecarQueueDrops: 0,
      sidecarInstalled: 15,
    },
    restoreFamilies: ['gemma4', 'qwen3_5'],
  };
}

/** Health counters rendered by the Cold-tier health card, one value element each. */
const HEALTH_STATS = 8;

/** KPI tiles above the charts, each with a headline value and a sub-line. */
const CACHE_TILES = 4;

describe('Cache page — a health counter loads into the box it will occupy', () => {
  it('keeps the value in the same typographic element in both states', async () => {
    const loading = await whileLoading(createElement(Cache), 'Corruptions', (page) => classesOf(page, '.text-xl'));
    const loaded = await whenLoaded(createElement(Cache), { '/cache': cacheFixture() }, 'mlx-paged-v1', (page) =>
      classesOf(page, '.text-xl'),
    );

    // Same count, same classes, in the same order: the loading card renders every
    // counter it will render loaded, and each value sits in an element whose
    // typography — hence whose line box — is identical. Dropping the wrapper for
    // a bare bar leaves this list empty while loading, which is the shape that
    // made the card 4px per row shorter than the one that replaced it.
    expect(loading).toHaveLength(HEALTH_STATS);
    expect(loading).toEqual(loaded);
  });

  it('sizes each health bar from the line box of its own wrapper', async () => {
    const bars = await whileLoading(createElement(Cache), 'Corruptions', (page) =>
      classesOf(page, '.text-xl > [data-slot="skeleton"]'),
    );
    expect(bars).toHaveLength(HEALTH_STATS);
    // Not `h-6`, and not `h-5` either: a literal height is a second copy of the
    // type scale that nothing keeps in step with `text-xl leading-none`.
    expect(allCarry(bars, 'h-[1lh]')).toBe(true);
  });

  it('sizes the KPI tile bars from the tile wrappers that carry the fonts', async () => {
    const { values, subs } = await whileLoading(createElement(Cache), 'Corruptions', (page) => ({
      // StatTile renders `value` inside `text-3xl leading-none` and `sub` inside
      // the `text-sm` div right after it, so both bars measure themselves against
      // the type the tile already declares.
      values: classesOf(page, 'div.text-3xl > [data-slot="skeleton"]'),
      subs: classesOf(page, 'div.text-3xl + div > [data-slot="skeleton"]'),
    }));
    expect(values).toHaveLength(CACHE_TILES);
    expect(subs).toHaveLength(CACHE_TILES);
    expect(allCarry(values, 'h-[1lh]')).toBe(true);
    expect(allCarry(subs, 'h-[1lh]')).toBe(true);
  });
});

function sessionRoutes(): Record<string, unknown> {
  const detail: SessionDetailResponse = {
    session: {
      id: 'abc',
      path: '/sessions/abc.jsonl',
      cwd: '/Users/dev/workspace/mlx-node',
      name: 'Refactor the paged cache',
      created: Date.UTC(2026, 6, 20, 9, 30),
      modified: Date.UTC(2026, 6, 20, 11, 0),
      messageCount: 2,
      firstMessage: 'hello',
    },
    transcript: [],
  };
  // No turns and no traces: `models` stays empty, so the loaded header is the
  // badge-less variant the loading header is shaped to match.
  const metrics: SessionMetricsResponse = { sessionId: 'abc', turns: [], traces: [] };
  return { '/sessions/abc': detail, '/sessions/abc/metrics': metrics };
}

function sessionPage(): ReactElement {
  return createElement(
    MemoryRouter,
    { initialEntries: ['/sessions/abc'] },
    createElement(Routes, null, createElement(Route, { path: '/sessions/:id', element: createElement(SessionDetail) })),
  );
}

describe('Session detail — the loading header is the loaded header', () => {
  it('gives the title bar the title element’s own classes', async () => {
    // By class, not by tag: the loaded title is an `h1` and the loading one a
    // `div` (an empty heading is a heading a screen reader still announces), and
    // it is the classes that decide the box either way.
    const title = '.text-2xl.tracking-tight';
    const loading = await whileLoading(sessionPage(), 'Sessions', (page) => classesOf(page, title));
    const loaded = await whenLoaded(sessionPage(), sessionRoutes(), 'Refactor the paged cache', (page) =>
      classesOf(page, title),
    );
    expect(loading).toEqual(loaded);
    expect(loading).toHaveLength(1);
  });

  it('stands in for the title, the meta line and the button — and for nothing else', async () => {
    const { title, all } = await whileLoading(sessionPage(), 'Sessions', (page) => ({
      title: classesOf(page, '.text-2xl > [data-slot="skeleton"]'),
      all: classesOf(page, '[data-slot="skeleton"]'),
    }));
    expect(allCarry(title, 'h-[1lh]')).toBe(true);
    // Title, meta line, copy button. NOT the model badges: that row is gated on
    // data from the metrics request, a different fetch, so reserving it here
    // could not spare the page that row's reflow and would cost a collapse on
    // every session that names no model.
    expect(all).toHaveLength(3);
  });
});

function metricsFixture(models: string[]): MetricsOverviewResponse {
  return {
    range: { from: null, to: null },
    tokensByDay: [],
    throughputByModel: models.map((model, i) => ({
      model,
      avgDecodeTps: 120 - i,
      avgPrefillTps: 900,
      avgTtftMs: 200 + i,
      samples: 10,
    })),
    throughputTrend: [],
    mtpByModel: models.map((model, i) => ({ model, meanAccepted: 1.9 - i * 0.05, avgCycles: 4, samples: 5 })),
    modelShare: models.map((model) => ({ model, turns: 10, outputTokens: 100 })),
    totals: { turns: 30, traces: 30, inputTokens: 0, outputTokens: 0, cachedTokens: 0, reasoningTokens: 0 },
  };
}

/** Plot area of every chart card: the one child of each `CardContent`. */
function plotBoxes(page: RenderedPage): string[] {
  return [...page.container.querySelectorAll('[data-slot="card-content"] > div')].map(
    (node) => `${node.getAttribute('class') ?? ''} @ ${(node as HTMLElement).style.height || 'auto'}`,
  );
}

describe('Metrics page — the plot area is the same box either way', () => {
  it('hands the skeleton and the chart the same plot box', async () => {
    const models = ['qwen3-8b', 'gemma-4-e2b-qat', 'Lfm2.5-8B-A1B-mlx-q4'];
    const loading = await whileLoading(createElement(Metrics), 'Tokens per day', plotBoxes);
    const loaded = await whenLoaded(
      createElement(Metrics),
      { '/metrics/overview': metricsFixture(models) },
      'qwen3-8b',
      plotBoxes,
      { charts: true },
    );
    // `ChartCard` owns the height and `ChartBody` only swaps what fills it, so
    // every card — the `heightClass` ones and the three that size themselves from
    // their row count — presents the identical box before and after the request.
    expect(loading).toEqual(loaded);
    expect(loading.length).toBeGreaterThan(0);
  });

  it('gives a category chart a floor tall enough to cover the row counts it loads into', () => {
    // The three per-model charts take their height from data they do not have
    // while loading, so `rowCount = 0` is the skeleton's height. The floor is what
    // makes that harmless: any model set up to six rows resolves to the same box.
    expect(categoryChartHeight(0)).toBe(224);
    expect(categoryChartHeight(6)).toBe(224);
    // Past the floor the card must grow with its data — nothing knows the row
    // count before the rows arrive, so this growth is the chart's, not a
    // mis-sized placeholder's.
    expect(categoryChartHeight(8)).toBe(284);
  });
});
