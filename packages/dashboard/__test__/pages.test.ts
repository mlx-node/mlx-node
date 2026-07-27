/**
 * @vitest-environment happy-dom
 */

/**
 * Render guards for the three SPA pages that carry cold-tier logic in JSX.
 *
 * Before this file the suite imported only `ui/src/lib/{api,cold-tier,format,shell}`,
 * so `cache.tsx`, `overview.tsx` and `session-detail.tsx` could each be reverted
 * to their pre-fix content and the whole suite stayed green: the block-vs-object
 * counts, the cold-tier health tiles, the saturating hit rate and the per-session
 * cold reuse chips all live at the JSX call site, not in a helper. These tests
 * mount the real components against a stubbed `fetch` and assert on rendered
 * text, so a page reverting is a red test.
 *
 * `happy-dom` is scoped to this file by the docblock above — the repo default
 * environment stays `node` for every other suite.
 */

import type {
  CacheResponse,
  DownloadsResponse,
  MetricsOverviewResponse,
  ModelsResponse,
  SessionDetailResponse,
  SessionMetricsResponse,
  SessionsResponse,
  SessionTraceMetric,
  SessionTurnMetric,
} from '@/lib/types';
import Cache from '@/pages/cache';
import Overview from '@/pages/overview';
import SessionDetail from '@/pages/session-detail';
import { createElement } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { renderPage, stubFetch, type RenderedPage } from './render.js';

const MIB = 1024 * 1024;

/**
 * The reported shape that produced the original bug report: 136 KV blocks plus
 * 2 state sidecars, a histogram summing to 138, and 16189 hits against 49
 * misses (99.6982 %).
 */
function cacheFixture(overrides: Partial<CacheResponse> = {}): CacheResponse {
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
      newestMtime: Date.now() - 3600_000,
      ageHistogram: [
        { label: '<1d', count: 100, bytes: 100 * MIB },
        { label: '1-7d', count: 30, bytes: 150 * MIB },
        { label: '7-30d', count: 6, bytes: 40 * MIB },
        { label: '>30d', count: 2, bytes: 10 * MIB },
      ],
      ...overrides.disk,
    },
    trend: overrides.trend ?? [
      { day: '2026-07-20', hits: 16_000, misses: 40, bytesWritten: 8 * MIB, bytesRestored: 64 * MIB },
      { day: '2026-07-21', hits: 189, misses: 9, bytesWritten: 2 * MIB, bytesRestored: 16 * MIB },
    ],
    scope: {
      root: '/private/tmp/cold/mlx-paged-v1',
      trendWindowDays: 30,
      legacy: { turns: 7, hits: 11, misses: 13 },
      otherRoots: { turns: 3, hits: 17, misses: 19 },
      unattributed: { turns: 0, hits: 0, misses: 0 },
      disabledTurns: 2,
      ...overrides.scope,
    },
    health: {
      enqueued: 500,
      queueDrops: 1,
      evictions: 4,
      corruptions: 0,
      corruptionsTotal: 0,
      queueDropsTotal: 6,
      ...overrides.health,
    },
    restoreFamilies: overrides.restoreFamilies ?? ['gemma4', 'qwen3', 'qwen3_5', 'qwen3_5_moe'],
  };
}

/** An empty tier: nothing on disk, nothing recorded, so both empty states show. */
function emptyCacheFixture(): CacheResponse {
  const base = cacheFixture();
  return {
    ...base,
    disk: {
      ...base.disk,
      exists: false,
      entryCount: 0,
      sidecarCount: 0,
      totalBytes: 0,
      sidecarBytes: 0,
      oldestMtime: null,
      newestMtime: null,
      ageHistogram: base.disk.ageHistogram.map((b) => ({ ...b, count: 0, bytes: 0 })),
    },
    trend: [],
  };
}

let mounted: RenderedPage | undefined;
let restoreFetch: (() => void) | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreFetch?.();
  restoreFetch = undefined;
});

/**
 * Mount a page against `routes` and return its settled text.
 *
 * `settledWhen` is a POSITIVE substring that only the loaded page renders. It is
 * required so that no assertion can run against a loading skeleton — which
 * matters most for the tests whose assertions are all negative, since those pass
 * just as happily when nothing rendered at all.
 */
async function mount(
  element: Parameters<typeof renderPage>[0],
  routes: Record<string, unknown>,
  settledWhen: string,
): Promise<string> {
  restoreFetch = stubFetch(routes);
  mounted = await renderPage(element, (text) => text.includes(settledWhen));
  return mounted.text();
}

describe('Cache page', () => {
  it('reports blocks and sidecars as separate units and totals them as objects', async () => {
    const text = await mount(createElement(Cache), { '/cache': cacheFixture() }, 'mlx-paged-v1');
    // F2: the tile total must be blocks + sidecars (138), the same number the
    // age histogram charts, and the sub-line must name both units. Reverting
    // cache.tsx puts back a "Blocks 136" tile beside a 138-object histogram.
    expect(text).toContain('Objects');
    expect(text).toContain('138');
    expect(text).toContain('136 prefix blocks · 2 state sidecars');
    expect(text).toContain('Objects by age');
    expect(text).not.toContain('persisted prefix blocks');
  });

  it('splits the usage tile into block bytes and sidecar bytes', async () => {
    const text = await mount(createElement(Cache), { '/cache': cacheFixture() }, 'mlx-paged-v1');
    expect(text).toContain('200 MB blocks + 100 MB sidecars');
  });

  it('renders the cold-tier health counters the API now serves', async () => {
    // F3: `corruptionsTotal` / `queueDropsTotal` / `enqueued` / `evictions` are
    // plumbed agent → traces → API for exactly one reason: the stated
    // acceptance bar is "corruptions must be 0" and nothing shipped could show
    // it. Reverting cache.tsx deletes this whole card.
    const text = await mount(createElement(Cache), { '/cache': cacheFixture() }, 'mlx-paged-v1');
    expect(text).toContain('Cold-tier health');
    expect(text).toContain('Corruptions');
    expect(text).toContain('none seen (acceptance bar: 0)');
    expect(text).toContain('Queue drops');
    expect(text).toContain('cumulative max 6 · 500 enqueued');
    expect(text).toContain('Evictions');
    expect(text).toContain('Bytes restored');
  });

  it('flags a non-zero cumulative corruption total even when the window delta is 0', async () => {
    const text = await mount(
      createElement(Cache),
      {
        '/cache': cacheFixture({
          health: { enqueued: 9, queueDrops: 0, evictions: 0, corruptions: 0, corruptionsTotal: 3, queueDropsTotal: 0 },
        }),
      },
      'mlx-paged-v1',
    );
    expect(text).toContain('cumulative max 3 — investigate');
    expect(text).not.toContain('none seen (acceptance bar: 0)');
  });

  it('never rounds a hit rate with real misses up to 100%', async () => {
    // 16189 hits / 49 misses is 99.6982 %. `Math.round` renders that as "100%"
    // directly above the line that prints the 49 misses.
    const text = await mount(createElement(Cache), { '/cache': cacheFixture() }, 'mlx-paged-v1');
    expect(text).toContain('>99%');
    expect(text).not.toContain('100%');
    expect(text).toContain('16.2K hits · 49 misses · this cache only');
  });

  it('names the rows the root scope excluded instead of folding or dropping them', async () => {
    // F1: the trend is filtered to one cache root. Rows outside it are reported.
    const text = await mount(createElement(Cache), { '/cache': cacheFixture() }, 'mlx-paged-v1');
    expect(text).toContain('7 turns recorded before cache attribution existed (11 hits · 13 misses)');
    expect(text).toContain('they age out within 30 days');
    expect(text).toContain('3 turns ran against a different cache directory (17 hits · 19 misses)');
    expect(text).toContain('Lookups against THIS cache root, by day · last 30 days');
  });

  it('names the tier-on-but-unattributed bucket instead of letting its lookups vanish', async () => {
    // MINOR 7: `cold_enabled = 1` with a NULL root used to match no bucket at
    // all, so its hits appeared nowhere — not in the trend, not in any excluded
    // line. The page must say it out loud.
    const text = await mount(
      createElement(Cache),
      {
        '/cache': cacheFixture({
          scope: {
            root: '/private/tmp/cold/mlx-paged-v1',
            trendWindowDays: 30,
            legacy: { turns: 0, hits: 0, misses: 0 },
            otherRoots: { turns: 0, hits: 0, misses: 0 },
            unattributed: { turns: 4, hits: 777, misses: 5 },
            disabledTurns: 0,
          },
        }),
      },
      'mlx-paged-v1',
    );
    expect(text).toContain('4 turns ran with the cold tier on but recorded no cache directory (777 hits · 5 misses)');
  });

  it('lists the served restore families in the empty state rather than "(qwen3 dense)"', async () => {
    // F5: four families are allowlisted; the hint used to name one, and the
    // list is served over the wire so it cannot drift from the native gate.
    const text = await mount(createElement(Cache), { '/cache': emptyCacheFixture() }, 'mlx-paged-v1');
    expect(text).toContain('gemma4, qwen3, qwen3_5, qwen3_5_moe');
    expect(text).not.toContain('(qwen3 dense)');
    expect(text).toContain('No cold-tier objects persisted yet.');
    expect(text).toContain('Objects appear when a persistent paged cache is used');
  });

  it('points an empty trend at the excluded lookups instead of a dead end', async () => {
    const text = await mount(createElement(Cache), { '/cache': emptyCacheFixture() }, 'mlx-paged-v1');
    expect(text).toContain('No hits or misses recorded for this cache.');
    expect(text).toContain('60 lookups were recorded against a different (or unattributed) cache root');
  });

  it('reports an unmeasured hit rate as unknown, not as a confident 0%', async () => {
    // The ONLY input at which formatRate and formatPercent diverge, and so the
    // only assertion on this page that can distinguish them: at 0 lookups
    // formatRate returns '—' while formatPercent(0/0 = NaN) returns '0%'.
    // The other F6 case (16189/16238) cannot discriminate, because formatPercent
    // was given the same >=99.5 saturation in this change and both return '>99%'.
    // '0%' here would be its own small lie: nothing was ever looked up.
    const text = await mount(createElement(Cache), { '/cache': emptyCacheFixture() }, 'mlx-paged-v1');
    // Scoped to the Hit rate tile: '0%' legitimately appears elsewhere on an
    // empty page (quota usage really is 0% of the quota).
    expect(text).toContain('Hit rate—');
    expect(text).toContain('no lookups recorded for this cache');
    expect(text).not.toContain('Hit rate0%');
  });

  it('labels the evict confirmation in objects, matching the count it shows', async () => {
    // 9a: the dialog counted OBJECTS while its title said "blocks". Radix
    // portals the content to document.body, so read from there.
    restoreFetch = stubFetch({ '/cache': cacheFixture() });
    mounted = await renderPage(createElement(Cache), (text) => text.includes('mlx-paged-v1'));
    const evict = [...mounted.container.querySelectorAll('button')].find((b) => b.textContent?.includes('Evict older'));
    expect(evict).toBeDefined();
    expect(evict?.disabled).toBe(false);
    const { act } = await import('react');
    await act(async () => {
      evict?.click();
    });
    const dialog = (document.body.textContent ?? '').replace(/\s+/g, ' ');
    expect(dialog).toContain('Evict old cache objects?');
    expect(dialog).not.toContain('Evict old cache blocks?');
    // 7-30d (6) + >30d (2) = 8 objects, 50 MB.
    expect(dialog).toContain('8 objects · 50.0 MB');
  });
});

describe('Overview page', () => {
  function overviewRoutes(cache: CacheResponse): Record<string, unknown> {
    const models: ModelsResponse = { models: [], warnings: [], dir: '/models' };
    const sessions: SessionsResponse = { sessions: [], total: 0 };
    const downloads: DownloadsResponse = { jobs: [] };
    const metrics: MetricsOverviewResponse = {
      range: { from: null, to: null },
      tokensByDay: [],
      throughputByModel: [],
      throughputTrend: [],
      mtpByModel: [],
      modelShare: [],
      totals: { turns: 0, traces: 0, inputTokens: 0, outputTokens: 0, cachedTokens: 0, reasoningTokens: 0 },
    };
    return {
      '/models': models,
      '/sessions': sessions,
      '/downloads': downloads,
      '/metrics/overview': metrics,
      '/cache': cache,
    };
  }

  it('never rounds the cold-cache hit rate up to 100% on the landing page', async () => {
    // F6 at its second call site. Reverting overview.tsx restores
    // `formatPercent(hits / lookups)`, which is 100% for 16189/16238.
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Overview)),
      overviewRoutes(cacheFixture()),
      'hit rate',
    );
    expect(text).toContain('hit rate >99%');
    expect(text).not.toContain('hit rate 100%');
    // HONEST SCOPE: this asserts the rendered outcome, but it cannot fail if
    // overview.tsx swaps formatRate back to formatPercent. overview.tsx:119
    // guards the label on `cacheLookups > 0`, and above zero the two functions
    // agree on every input — so the rounding rule itself is pinned by
    // format.test.ts, and the page-level guard that DOES fail on revert is the
    // quota-meter test below (percentInt). Left in place because it still
    // catches the label disappearing or reverting to a raw Math.round.
  });

  it('saturates the quota meter width the same way the label is saturated', async () => {
    // `Math.round(quotaFraction * 100)` renders a 99.9%-full quota as a 100%
    // bar. `percentInt` caps it at 99 so the bar and the label agree.
    const nearlyFull = cacheFixture();
    nearlyFull.disk = { ...nearlyFull.disk, totalBytes: 9_999_999, quotaBytes: 10_000_000 };
    restoreFetch = stubFetch(overviewRoutes(nearlyFull));
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Overview)), (text) =>
      text.includes('of 9.5 MB'),
    );
    const widths = [...mounted.container.querySelectorAll<HTMLElement>('div[style*="width"]')].map(
      (el) => el.style.width,
    );
    expect(widths).toContain('99%');
    expect(widths).not.toContain('100%');
    expect(mounted.text()).toContain('>99% of 9.5 MB');
  });
});

describe('Metrics page — model usage share', () => {
  it('never labels a slice 100% while another model still holds turns', async () => {
    // 9b: the last surviving copy of the F6 rounding lie. 9999 of 10000 turns
    // rounds to "100%" beside a two-model legend.
    const { modelShareLabel } = await import('@/pages/metrics');
    expect(modelShareLabel(9999, 10_000)).toBe('9,999 turns · >99%');
    expect(modelShareLabel(1, 10_000)).toBe('1 turns · <1%');
    // The endpoints stay reachable when they are actually true.
    expect(modelShareLabel(10_000, 10_000)).toBe('10,000 turns · 100%');
    expect(modelShareLabel(0, 10_000)).toBe('0 turns · 0%');
  });
});

describe('Session detail page', () => {
  function trace(overrides: Partial<SessionTraceMetric>): SessionTraceMetric {
    return {
      traceId: 't1',
      ts: 1_700_000_000_000,
      model: 'qwen3',
      ttftMs: 120,
      prefillTps: 900,
      decodeTps: 60,
      mtpCycles: null,
      mtpMeanAccepted: null,
      durationMs: 1000,
      finishReason: 'stop',
      coldHits: 0,
      coldMisses: 0,
      coldBytesWritten: 0,
      coldBytesRestored: 0,
      ...overrides,
    };
  }

  function sessionRoutes(traces: SessionTraceMetric[]): Record<string, unknown> {
    const detail: SessionDetailResponse = {
      session: {
        id: 'abc',
        path: '/sessions/abc.jsonl',
        cwd: '/work',
        name: 'demo',
        created: 1_700_000_000_000,
        modified: 1_700_000_100_000,
        messageCount: 2,
        firstMessage: 'hello',
      },
      transcript: [],
    };
    // The chip row is gated on `turns.length > 0`, so a turn per trace.
    const turns: SessionTurnMetric[] = traces.map((t) => ({
      entryId: t.traceId,
      traceId: t.traceId,
      ts: t.ts,
      model: t.model,
      inputTokens: 10,
      outputTokens: 20,
      cachedTokens: 0,
      reasoningTokens: 0,
      ttftMs: t.ttftMs,
      prefillTps: t.prefillTps,
      decodeTps: t.decodeTps,
      mtpCycles: null,
      mtpMeanAccepted: null,
      durationMs: t.durationMs,
      finishReason: t.finishReason,
      coldHits: t.coldHits,
      coldMisses: t.coldMisses,
      coldBytesWritten: t.coldBytesWritten,
      coldBytesRestored: t.coldBytesRestored,
    }));
    const metrics: SessionMetricsResponse = { sessionId: 'abc', turns, traces };
    return { '/sessions/abc': detail, '/sessions/abc/metrics': metrics };
  }

  function page() {
    return createElement(
      MemoryRouter,
      { initialEntries: ['/sessions/abc'] },
      createElement(
        Routes,
        null,
        createElement(Route, { path: '/sessions/:id', element: createElement(SessionDetail) }),
      ),
    );
  }

  it('surfaces per-session cold reuse from the trace fields the API always returned', async () => {
    const text = await mount(
      page(),
      sessionRoutes([
        trace({ traceId: 't1', coldHits: 8000, coldMisses: 20, coldBytesRestored: 2 * MIB, coldBytesWritten: MIB }),
        trace({ traceId: 't2', coldHits: 8189, coldMisses: 29, coldBytesRestored: 2 * MIB, coldBytesWritten: MIB }),
      ]),
      'Cold hit rate',
    );
    expect(text).toContain('Cold hit rate');
    // 16189 / 16238 — the same non-rounding rule as the Cache page.
    expect(text).toContain('>99%');
    expect(text).toContain('Cold reuse');
    expect(text).toContain('4.0 MB');
    expect(text).toContain('2.0 MB written');
  });

  it('hides the cold chips entirely when the session recorded no cold lookups', async () => {
    // Anchored on a SIBLING stat in the same row the cold chips would occupy.
    // Without it these two negative assertions pass just as happily when the
    // page never rendered at all — which is how this test read before, and is
    // the exact defect class this file exists to catch.
    const text = await mount(page(), sessionRoutes([trace({})]), 'TTFT');
    expect(text).toContain('Turns');
    expect(text).not.toContain('Cold hit rate');
    expect(text).not.toContain('Cold reuse');
  });
});
