/**
 * @vitest-environment happy-dom
 */

/**
 * Structural guards for the loading -> loaded swap on the Cache, Session detail and Metrics pages.
 *
 * A placeholder only does its job if it takes the SAME box as the content it stands in for; otherwise every answer
 * that lands nudges the page. happy-dom performs no layout, so the boxes cannot be measured here. What these tests
 * assert is the structure that makes them equal: a stat tile holds the same slots loading and loaded, and a chart
 * card's body keeps the same height in every state.
 */

import { MessageChannel } from 'node:worker_threads';

import type { SessionDetailResponse, SessionTimelineResponse, TimelineLane, TranscriptEntry } from '@/lib/types';
import Cache from '@/pages/cache';
import Metrics from '@/pages/metrics';
import SessionDetail from '@/pages/session-detail';
import { createElement, type ReactElement } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it } from 'vite-plus/test';

import { bindEventTargetPort } from '../src/rpc/port.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import {
  cacheResponse,
  dayStart,
  distributions,
  speculative,
  summary,
  timeseries,
  tsPoint,
  turnRow,
  turnsPage,
} from './helpers/observability-fixtures.js';
import { byRole, deferred, renderPage, stubApi, waitFor, type RenderedPage } from './render.js';

/**
 * Connect the SPA to a port whose peer never answers, so every request stays loading for as long as the test looks.
 * `connectDashboardApi` also drops the json cache, so an earlier loaded test cannot seed this one.
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

/** Mount `element` against `routes`, wait for `settled`, read the tree, tear it down. */
async function whenLoaded<T>(
  element: ReactElement,
  routes: Record<string, unknown>,
  settled: string | ((page: RenderedPage) => boolean),
  read: (page: RenderedPage) => T,
): Promise<T> {
  const restore = stubApi(routes);
  try {
    const page = await renderPage(element, () => true);
    try {
      await waitFor(() => (typeof settled === 'string' ? page.text().includes(settled) : settled(page)), 3000);
      return read(page);
    } finally {
      page.unmount();
    }
  } finally {
    restore();
  }
}

/** The content box of a stat tile. */
function tileContent(page: RenderedPage, label: string): Element {
  const tile = byRole(page.container, 'group', label);
  const content = tile === null ? undefined : [...tile.children].find((child) => child.tagName === 'DIV');
  if (content === undefined) throw new Error(`no tile ${label}`);
  return content;
}

/** Tag and class of each slot of each named tile, in order. */
function tileSlots(page: RenderedPage, labels: readonly string[]): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const label of labels) {
    out[label] = [...tileContent(page, label).children].map((child) => `${child.tagName}.${child.className}`);
  }
  return out;
}

/** Every chart card on the page: its title and the height its body holds. */
function chartBoxes(page: RenderedPage): string[] {
  return [...page.container.querySelectorAll('section[data-state]')].map((section) => {
    const title = section.querySelector('h3')?.textContent ?? '';
    const body = section.querySelector(':scope > div');
    return `${title} @ ${(body as HTMLElement | null)?.style.minHeight ?? 'none'}`;
  });
}

function noChartLoading(page: RenderedPage): boolean {
  return page.container.querySelector('section[data-state="loading"]') === null;
}

function cachePage(): ReactElement {
  return createElement(MemoryRouter, null, createElement(Cache));
}

const CACHE_TILES = ['Usage', 'Objects', 'Hit rate', 'Bytes restored', 'Oldest object'];

const HEALTH_TILES = [
  'Corruptions',
  'Queue drops',
  'Write errors',
  'Restore declines',
  'Sidecar captures',
  'Sidecar installs',
  'Evictions',
  'Turns with the tier off',
];

function cacheLoaded(page: RenderedPage): boolean {
  return page.text().includes('mlx-paged-v1') && noChartLoading(page);
}

describe('Cache page — the placeholders take the boxes they will hold', () => {
  it('gives every tile the same slots loading and loaded', async () => {
    const labels = [...CACHE_TILES, ...HEALTH_TILES];
    const loading = await whileLoading(cachePage(), 'Corruptions', (page) => tileSlots(page, labels));
    const loaded = await whenLoaded(cachePage(), { '/cache': cacheResponse() }, cacheLoaded, (page) =>
      tileSlots(page, labels),
    );
    for (const label of labels) expect(loading[label], label).toEqual(loaded[label]);
  });

  it('puts one placeholder in the value row of each health tile', async () => {
    const counts = await whileLoading(cachePage(), 'Corruptions', (page) =>
      HEALTH_TILES.map((label) => tileContent(page, label).children[1]?.querySelectorAll('[data-placeholder]').length),
    );
    expect(counts).toEqual(HEALTH_TILES.map(() => 1));
  });

  it('keeps each chart body the same height loading and loaded', async () => {
    const loading = await whileLoading(cachePage(), 'Corruptions', chartBoxes);
    const loaded = await whenLoaded(cachePage(), { '/cache': cacheResponse() }, cacheLoaded, chartBoxes);
    expect(loading.length).toBeGreaterThan(0);
    expect(loading).toEqual(loaded);
  });
});

const SESSION_ID = '019a0000-0000-7000-8000-000000000001';

function sessionRoutes(): Record<string, unknown> {
  // No model, no kind, not live, no turns: the badge row stays hidden, which is
  // the variant the loading header is shaped to match.
  const detail: SessionDetailResponse = {
    session: {
      id: SESSION_ID,
      origin: 'agent',
      title: 'Refactor the paged cache',
      cwd: '/Users/dev/workspace/mlx-node',
      model: null,
      branch: 'main',
      createdAt: '2026-07-20T09:30:00.000Z',
      lastActiveAt: '2026-07-20T11:00:00.000Z',
      messageCount: 2,
      live: false,
      kind: null,
      parentId: null,
      subagentCount: 0,
      tokens: null,
      gitRoot: null,
      headCommit: null,
      remotes: [],
      agentName: null,
      reasoningEffort: null,
      lastTurnSummary: null,
      parent: null,
      subagents: [],
    },
    transcript: [],
    turns: [],
    resumeCommand: `cd '/Users/dev/workspace/mlx-node' && mlx agent --resume ${SESSION_ID}`,
  };
  return { [`/sessions/${SESSION_ID}`]: detail };
}

function sessionPage(): ReactElement {
  return createElement(
    MemoryRouter,
    { initialEntries: [`/sessions/${SESSION_ID}`] },
    createElement(Routes, null, createElement(Route, { path: '/sessions/:id', element: createElement(SessionDetail) })),
  );
}

describe('Session detail — the loading header is the loaded header', () => {
  it('keeps the h1 in both states, holding a placeholder while loading', async () => {
    const loading = await whileLoading(sessionPage(), 'Directory', (page) => {
      const h1 = page.container.querySelector('h1');
      return {
        count: page.container.querySelectorAll('h1').length,
        bones: h1?.querySelectorAll('[data-placeholder]').length,
      };
    });
    const loaded = await whenLoaded(sessionPage(), sessionRoutes(), 'Refactor the paged cache', (page) => {
      const h1 = page.container.querySelector('h1');
      return {
        count: page.container.querySelectorAll('h1').length,
        text: h1?.textContent,
        bones: h1?.querySelectorAll('[data-placeholder]').length,
      };
    });
    expect(loading).toEqual({ count: 1, bones: 1 });
    expect(loaded).toEqual({ count: 1, text: 'Refactor the paged cache', bones: 0 });
  });

  it('stands in for the facts with the same panel, and shows no tabs until it knows the session', async () => {
    const loading = await whileLoading(sessionPage(), 'Directory', (page) => ({
      facts: byRole(page.container, 'region', 'Session details') !== null,
      tabs: byRole(page.container, 'tablist') !== null,
    }));
    const loaded = await whenLoaded(sessionPage(), sessionRoutes(), 'Refactor the paged cache', (page) => ({
      facts: byRole(page.container, 'region', 'Session details') !== null,
      tabs: byRole(page.container, 'tablist') !== null,
    }));
    expect(loading).toEqual({ facts: true, tabs: false });
    expect(loaded).toEqual({ facts: true, tabs: true });
  });
});

/** One finished turn that spawned `children` subagents (one listed twice), with its timeline route held or answered. */
function timelineRoutes(children: number, timeline: unknown): Record<string, unknown> {
  const base = sessionRoutes()[`/sessions/${SESSION_ID}`] as SessionDetailResponse;
  const ts = Date.UTC(2026, 6, 20, 10);
  const spawned: TranscriptEntry[] = Array.from({ length: children }, (_, index) => ({
    kind: 'subagent',
    ts: ts + 10,
    childId: `child-${String(index)}`,
    type: 'explore',
    description: `Child ${String(index)}`,
  }));
  const transcript: TranscriptEntry[] = [
    { kind: 'user', ts, text: 'Map the planner', promptIndex: 0 },
    ...spawned,
    ...spawned.slice(0, 1),
    { kind: 'turn-end', ts: ts + 4_000, stopReason: 'end_turn', elapsedMs: 4_000 },
  ];
  const turn = {
    promptId: 'p1',
    ts,
    inputTokens: 10,
    outputTokens: 5,
    cachedTokens: 0,
    reasoningTokens: 0,
    elapsedMs: 4_000,
    ttftMs: 100,
    prefillMs: null,
    prefillTokens: null,
    prefillCalls: null,
    modelCalls: null,
    decodeTps: 50,
    model: null,
  };
  return {
    [`/sessions/${SESSION_ID}`]: { ...base, transcript, turns: [turn] },
    [`/sessions/${SESSION_ID}/timeline`]: timeline,
  };
}

function timelineLane(id: string, index: number): TimelineLane {
  return {
    id,
    role: index === 0 ? 'main' : 'subagent',
    name: id,
    type: null,
    model: null,
    slot: index === 0 ? null : index - 1,
    sessionId: null,
    status: 'completed',
    carriedIn: false,
    carriedOut: false,
    noSteps: null,
    fileBytes: null,
    clamped: false,
    subagents: 0,
    tokenReports: [],
  };
}

function answeredTimeline(lanes: number): SessionTimelineResponse {
  const start = Date.UTC(2026, 6, 20, 10);
  return {
    turn: 1,
    turns: 1,
    open: false,
    live: false,
    stopReason: 'end_turn',
    asOfMs: start + 10_000,
    range: {
      startMs: start,
      turnStartMs: start,
      endMs: start + 4_000,
      lastEventMs: start + 4_000,
      startExact: true,
      leadMs: 0,
    },
    lanes: Array.from({ length: lanes }, (_, index) =>
      timelineLane(index === 0 ? 'main' : `child-${String(index - 1)}`, index),
    ),
    spans: [],
    handoffs: [],
    notes: { untimedEvents: 0, foldedSubagents: 0, droppedSpans: 0, unplacedRetries: 0 },
  };
}

/** The `--lanes` height hook of the turn's timeline frame and how many lanes it draws. */
function timelineShape(page: RenderedPage): { lanes: string; rows: number } {
  const section = byRole(page.container, 'region', 'Turn 1 timeline');
  const frame = section?.firstElementChild as HTMLElement | null;
  const rows = section?.querySelectorAll('[aria-label^="Steps"] > [role="group"], [data-placeholder-lane]').length ?? 0;
  return { lanes: frame?.style.getPropertyValue('--lanes') ?? '', rows };
}

describe('Session detail — the loading timeline holds the lanes it expects', () => {
  it('draws one placeholder lane per expected lane, with the height hook of the loaded timeline', async () => {
    const held = deferred(answeredTimeline(3));
    const loading = await whenLoaded(sessionPage(), timelineRoutes(2, held.body), 'Map the planner', (page) => ({
      busy: byRole(page.container, 'region', 'Turn 1 timeline')?.getAttribute('aria-busy'),
      ...timelineShape(page),
    }));
    held.release();
    const loaded = await whenLoaded(
      sessionPage(),
      timelineRoutes(2, answeredTimeline(3)),
      'Positions come from recorded event times',
      timelineShape,
    );
    expect(loading).toEqual({ busy: 'true', lanes: '3', rows: 3 });
    expect(loaded).toEqual({ lanes: '3', rows: 3 });
  });

  it('expects at most 10 lanes: the main agent, 8 subagents and the folded lane', async () => {
    const held = deferred(answeredTimeline(10));
    const loading = await whenLoaded(sessionPage(), timelineRoutes(12, held.body), 'Map the planner', timelineShape);
    held.release();
    const eight = deferred(answeredTimeline(9));
    const full = await whenLoaded(sessionPage(), timelineRoutes(8, eight.body), 'Map the planner', timelineShape);
    eight.release();
    expect(loading).toEqual({ lanes: '10', rows: 10 });
    expect(full).toEqual({ lanes: '9', rows: 9 });
  });
});

function metricsRoutes(): Record<string, unknown> {
  return {
    '/metrics/models': { models: [] },
    '/metrics/summary': summary(),
    '/metrics/timeseries': timeseries([tsPoint(dayStart(3), 'qwen3-8b'), tsPoint(dayStart(1), 'qwen3-8b')]),
    '/metrics/distributions': distributions(),
    '/metrics/speculative': speculative(),
    '/metrics/turns': turnsPage([turnRow('t1'), turnRow('t2')]),
    '/models': { models: [], warnings: [], dir: '/models' },
  };
}

function metricsAt(tab: string): ReactElement {
  return createElement(MemoryRouter, { initialEntries: [`/metrics?tab=${tab}`] }, createElement(Metrics));
}

/** Loading-state bone height of each chart card, read where the body says it should be. */
function boneHeights(page: RenderedPage): string[] {
  return [...page.container.querySelectorAll('section[data-state="loading"]')].map((section) => {
    const body = section.querySelector(':scope > div') as HTMLElement | null;
    const bone = body?.querySelector('[data-placeholder]') as HTMLElement | null;
    return `${body?.style.minHeight ?? ''} = ${bone?.style.height ?? ''}`;
  });
}

describe('Metrics page — the plot box is the same either way', () => {
  for (const tab of ['summary', 'speed', 'usage']) {
    it(`keeps every chart body of the ${tab} tab the same height loading and ready`, async () => {
      const loading = await whileLoading(metricsAt(tab), 'Metrics', chartBoxes);
      const loaded = await whenLoaded(
        metricsAt(tab),
        metricsRoutes(),
        (page) => page.container.querySelector('section[data-state]') !== null && noChartLoading(page),
        chartBoxes,
      );
      expect(loading.length).toBeGreaterThan(0);
      expect(loading).toEqual(loaded);
    });
  }

  it('fills the loading body with a bone of the body height', async () => {
    const heights = await whileLoading(metricsAt('summary'), 'Metrics', boneHeights);
    expect(heights.length).toBeGreaterThan(0);
    for (const pair of heights) {
      const [body, bone] = pair.split(' = ');
      expect(bone).toBe(body);
    }
  });

  it('gives the summary tiles the same slots loading and loaded', async () => {
    const labels = ['Turns', 'Output tokens', 'Decode speed', 'First token', 'Failed turns'];
    const loading = await whileLoading(metricsAt('summary'), 'Metrics', (page) => tileSlots(page, labels));
    const loaded = await whenLoaded(
      metricsAt('summary'),
      metricsRoutes(),
      (page) => page.container.querySelector('[data-stat-value]') !== null && noChartLoading(page),
      (page) => tileSlots(page, labels),
    );
    for (const label of labels) expect(loading[label], label).toEqual(loaded[label]);
  });
});
