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

import { MessageChannel } from 'node:worker_threads';

import { categoryLabels, truncateMiddle } from '@/lib/chart';
import type {
  CacheResponse,
  CacheScope,
  CatalogItem,
  CatalogResponse,
  ColdTierHealth,
  DownloadJob,
  DownloadsResponse,
  MetricsOverviewResponse,
  ModelsResponse,
  SessionDetailResponse,
  SessionMetricsResponse,
  SessionRow,
  SessionsResponse,
  SessionTraceMetric,
  SessionTurnMetric,
} from '@/lib/types';
import Cache from '@/pages/cache';
import Metrics from '@/pages/metrics';
import Models from '@/pages/models';
import Overview from '@/pages/overview';
import SessionDetail from '@/pages/session-detail';
import Sessions from '@/pages/sessions';
import { act, createElement } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { toast } from 'sonner';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { DownloadEvent } from '../src/download.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import type { ApiCall } from '../src/runtime.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import {
  renderPage,
  sequence,
  stubApi,
  stubChartMetrics,
  TICK_CHAR_PX,
  TICK_LINE_PX,
  type RenderedPage,
} from './render.js';

const MIB = 1024 * 1024;

/**
 * `health` and `scope` are overridable field-by-field: between them they carry
 * two dozen counters, and a test that must restate all of them to change one
 * buries its own intent — and quietly acquires a second meaning every time a
 * counter is added.
 */
type CacheFixtureOverrides = Omit<Partial<CacheResponse>, 'health' | 'scope'> & {
  health?: Partial<ColdTierHealth>;
  scope?: Partial<CacheScope>;
};

/**
 * The reported shape that produced the original bug report: 136 KV blocks plus
 * 2 state sidecars, a histogram summing to 138, and 16189 hits against 49
 * misses (99.6982 %).
 */
function cacheFixture(overrides: CacheFixtureOverrides = {}): CacheResponse {
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
      unrootedSidecarCaptures: 0,
      ...overrides.scope,
    },
    health: {
      enqueued: 500,
      queueDrops: 1,
      evictions: 4,
      corruptions: 0,
      corruptionsTotal: 0,
      queueDropsTotal: 6,
      writeErrors: 0,
      writeErrorsTotal: 0,
      restoreDeclines: 0,
      restoreSuppressed: 0,
      // A healthy hybrid run: the capture ran on every turn, wrote its sidecar
      // once and re-found it thereafter, and the restore side installed what it
      // was handed. `sidecarEnqueued` / `sidecarQueueDrops` sit UNDER the
      // object-scoped `enqueued` / `queueDrops` above — they are the sidecar
      // share of the same queue, never a second population.
      sidecarCaptureReached: 20,
      sidecarChainEmpty: 1,
      sidecarBoundarySkips: 2,
      sidecarAlreadyPersisted: 16,
      sidecarEnqueued: 1,
      sidecarQueueDrops: 0,
      sidecarInstalled: 15,
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
let restoreApi: (() => void) | undefined;
let restoreMetrics: (() => void) | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
  restoreMetrics?.();
  restoreMetrics = undefined;
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
  restoreApi = stubApi(routes);
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
    expect(text).toContain('cumulative max 6 · 500 objects enqueued');
    expect(text).toContain('Evictions');
    expect(text).toContain('Bytes restored');
    expect(text).toContain('Write errors');
    expect(text).toContain('none seen (writes are landing)');
    expect(text).toContain('Restore declines');
  });

  /**
   * The reported operator scenario, at the pixel it is read from: a cache root
   * that accepts no writes.
   *
   * Every counter that existed before stays at zero — the queue was never
   * full, nothing was ever read back so nothing could be corrupt — which is
   * precisely why the row was unreadable. The card must say the root is
   * refusing writes, not stay silent because the four old counters are clean.
   */
  it('raises an alarm when the cache root is refusing writes', async () => {
    const text = await mount(
      createElement(Cache),
      {
        '/cache': cacheFixture({
          health: {
            enqueued: 42,
            queueDrops: 0,
            evictions: 0,
            corruptions: 0,
            corruptionsTotal: 0,
            queueDropsTotal: 0,
            writeErrors: 42,
            writeErrorsTotal: 42,
            restoreDeclines: 0,
            restoreSuppressed: 0,
          },
        }),
      },
      'mlx-paged-v1',
    );
    expect(text).toContain('cumulative max 42 — the cache root is refusing writes');
    expect(text).not.toContain('none seen (writes are landing)');
    // The pre-existing counters really are all clean, which is the point.
    expect(text).toContain('none seen (acceptance bar: 0)');
  });

  /**
   * The restore-side silent zero, at the same pixel.
   *
   * A refused restore performs no lookup, so `trend` is empty and the hit-rate
   * tile used to print "no lookups recorded for this cache" — which reads as
   * "nothing ran" when in fact reuse was offered and refused. Opposite
   * diagnoses, same rendering.
   */
  it('says reuse was refused rather than "no lookups" when restores were declined', async () => {
    const text = await mount(
      createElement(Cache),
      {
        '/cache': cacheFixture({
          trend: [],
          health: {
            enqueued: 12,
            queueDrops: 0,
            evictions: 0,
            corruptions: 0,
            corruptionsTotal: 0,
            queueDropsTotal: 0,
            writeErrors: 0,
            writeErrorsTotal: 0,
            restoreDeclines: 3,
            restoreSuppressed: 1,
          },
        }),
      },
      'mlx-paged-v1',
    );
    expect(text).toContain('4 restores refused · no lookup was performed');
    expect(text).not.toContain('no lookups recorded for this cache');
    // And the card keeps the two causes apart, since the fixes differ.
    expect(text).toContain('1 suppressed after restore · neither counts as a lookup');
  });

  it('flags a non-zero cumulative corruption total even when the window delta is 0', async () => {
    const text = await mount(
      createElement(Cache),
      {
        '/cache': cacheFixture({
          health: {
            enqueued: 9,
            queueDrops: 0,
            evictions: 0,
            corruptions: 0,
            corruptionsTotal: 3,
            queueDropsTotal: 0,
            writeErrors: 0,
            writeErrorsTotal: 0,
            restoreDeclines: 0,
            restoreSuppressed: 0,
          },
        }),
      },
      'mlx-paged-v1',
    );
    expect(text).toContain('cumulative max 3 — investigate');
    expect(text).not.toContain('none seen (acceptance bar: 0)');
  });

  /**
   * `installed` is the one cold-tier counter that cannot be inferred from any
   * other number on this page. Every `install_*_cold_sidecar` early-return
   * falls through to a full O(prefix) replay that produces CORRECT state, so a
   * regression from "restored and INSTALLED" to "restored and silently
   * re-derived" leaves the hit rate, the trend, the bytes and the corruption
   * count all exactly as they were. A counter that reaches the API and is
   * never rendered is the same defect one layer up.
   */
  it('renders the sidecar install and capture counters', async () => {
    const text = await mount(createElement(Cache), { '/cache': cacheFixture() }, 'mlx-paged-v1');
    expect(text).toContain('Sidecar installs');
    expect(text).toContain('restored prefixes that reused family state instead of replaying it');
    expect(text).toContain('Sidecar captures');
    // The write-side outcomes of those 20 captures: 1 wrote, 16 re-found the
    // same chain on disk, 3 (1 chain-empty + 2 boundary skips) found no anchor.
    expect(text).toContain('1 written · 16 already on disk · 3 found no anchor');
  });

  /**
   * The regression the counter exists for, at the pixel it is read from.
   *
   * The prefix WAS restored (real hits, real bytes) and the family's state
   * sidecar WAS on disk — the capture re-found it — yet nothing was installed,
   * so every restored turn silently replayed the whole prefix. Every other
   * number on the page is identical to a healthy run, which is why the tile
   * has to say it.
   */
  it('flags restored prefixes whose state was re-derived instead of installed', async () => {
    const text = await mount(
      createElement(Cache),
      { '/cache': cacheFixture({ health: { sidecarInstalled: 0, sidecarAlreadyPersisted: 12 } }) },
      'mlx-paged-v1',
    );
    expect(text).toContain('state was on disk and prefixes restored — every one re-derived by a full replay');
    expect(text).not.toContain('restored prefixes that reused family state instead of replaying it');
    // The counters that CAN be read off the rest of the page are untouched:
    // this is a clean cache by every pre-existing measure.
    expect(text).toContain('none seen (acceptance bar: 0)');
    expect(text).toContain('none seen (writes are landing)');
  });

  /**
   * The write-side dead end: the capture ran on every turn and never once
   * wrote a sidecar or found one already there, so nothing hybrid will ever
   * restore its state. `alreadyPersisted` is what separates this from the
   * healthy steady state, where `enqueued` is legitimately 0 because the first
   * turn already wrote the chain every later turn re-selects.
   */
  it('flags a capture that never lands or finds a sidecar', async () => {
    const text = await mount(
      createElement(Cache),
      {
        '/cache': cacheFixture({
          health: {
            sidecarCaptureReached: 30,
            sidecarChainEmpty: 12,
            sidecarBoundarySkips: 18,
            sidecarAlreadyPersisted: 0,
            sidecarEnqueued: 0,
          },
        }),
      },
      'mlx-paged-v1',
    );
    expect(text).toContain('nothing written and nothing on disk — no prefix can restore state');
    expect(text).not.toContain('already on disk');
  });

  // Dense families never enter a sidecar capture at all (`record_capture_reached`
  // is called only by gemma4 / qwen3_5 / qwen3_5_moe), so a zero here is the
  // NORMAL qwen3 reading and must not wear an alarm.
  it('reads a zero capture count as a dense family, not a fault', async () => {
    const text = await mount(
      createElement(Cache),
      {
        '/cache': cacheFixture({
          health: {
            sidecarCaptureReached: 0,
            sidecarChainEmpty: 0,
            sidecarBoundarySkips: 0,
            sidecarAlreadyPersisted: 0,
            sidecarEnqueued: 0,
            sidecarInstalled: 0,
          },
        }),
      },
      'mlx-paged-v1',
    );
    expect(text).toContain('not reached — dense families keep no state outside the pool');
    expect(text).not.toContain('nothing written and nothing on disk');
    // No install, but nothing restored state either — not the replay alarm.
    expect(text).not.toContain('every one re-derived by a full replay');
  });

  /**
   * The same zero, a different cause, and the one the dense-family reading
   * hides. `record_capture_reached()` runs above the `adapter.cold_tier()`
   * guard in every family's capture, so a hybrid turn under
   * `--no-persist-cache` — or one whose tier failed to open — counts the
   * capture and then carries no root for the scope to match. Those captures
   * arrive as `scope.unrootedSidecarCaptures`, and reading "dense families keep
   * no state outside the pool" off them names the wrong cause for the exact
   * failure these counters were added to record.
   */
  it('does not read a hybrid capture that never got a tier as a dense family', async () => {
    const text = await mount(
      createElement(Cache),
      {
        '/cache': cacheFixture({
          scope: { unrootedSidecarCaptures: 34 },
          health: {
            sidecarCaptureReached: 0,
            sidecarChainEmpty: 0,
            sidecarBoundarySkips: 0,
            sidecarAlreadyPersisted: 0,
            sidecarEnqueued: 0,
            sidecarInstalled: 0,
          },
        }),
      },
      'mlx-paged-v1',
    );
    expect(text).toContain('34 ran with no cache attached');
    expect(text).not.toContain('not reached — dense families keep no state outside the pool');
  });

  // The sidecar queue counters are the sidecar SHARE of the object-scoped
  // numbers beside them (one admission bumps both), so the card must frame them
  // as a share and never as a second population a reader could add.
  it('labels the sidecar queue counters as a share of the object counters', async () => {
    const text = await mount(
      createElement(Cache),
      { '/cache': cacheFixture({ health: { sidecarEnqueued: 12, sidecarQueueDrops: 3 } }) },
      'mlx-paged-v1',
    );
    expect(text).toContain('500 objects enqueued · 12 sidecars among them, 3 of the drops');
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
    restoreApi = stubApi({ '/cache': cacheFixture() });
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
  function overviewRoutes(cache: CacheResponse, sessions?: SessionsResponse): Record<string, unknown> {
    const models: ModelsResponse = { models: [], warnings: [], dir: '/models' };
    const downloads: DownloadsResponse = { jobs: [] };
    // Deliberately enormous, and deliberately UNRELATED to the session list.
    // This route is the machine-wide one; nothing on the Sessions tile may read
    // from it, so these numbers must not appear.
    const metrics: MetricsOverviewResponse = {
      range: { from: null, to: null },
      tokensByDay: [],
      throughputByModel: [],
      throughputTrend: [],
      mtpByModel: [],
      modelShare: [],
      totals: {
        turns: 9_000,
        traces: 9_000,
        inputTokens: 1_500_000,
        outputTokens: 500_000,
        cachedTokens: 0,
        reasoningTokens: 0,
      },
    };
    return {
      '/models': models,
      '/sessions': sessions ?? { sessions: [], total: 0, tokens: 0, cwds: [] },
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

  it('takes both halves of the Sessions tile from the session list, not the machine-wide metrics', async () => {
    // The reported shape: three sessions holding 1,672 tokens between them, in a
    // dashboard whose machine has 2,000,000 tokens of history across every
    // session directory it has ever indexed. The tile counted the three and then
    // captioned them with the two million, and the caption did not move when the
    // dashboard was pointed at a different --session-dir.
    const sessions: SessionsResponse = { sessions: [], total: 3, tokens: 1_672, cwds: [] };
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Overview)),
      overviewRoutes(cacheFixture(), sessions),
      // Anchored on a SIBLING tile, deliberately: settling on this tile's own
      // wording would turn a regression into a 2s timeout instead of an
      // assertion that names the number it found.
      'hit rate',
    );
    expect(text).toContain('1,672 tokens · these sessions only');
    // 1_500_000 + 500_000 formatted by formatCount. Its presence anywhere on the
    // tile means the machine-wide route is still being read.
    expect(text).not.toContain('2M');
    expect(text).not.toContain('last 7 days');
  });

  it('saturates the quota meter width the same way the label is saturated', async () => {
    // `Math.round(quotaFraction * 100)` renders a 99.9%-full quota as a 100%
    // bar. `percentInt` caps it at 99 so the bar and the label agree.
    const nearlyFull = cacheFixture();
    nearlyFull.disk = { ...nearlyFull.disk, totalBytes: 9_999_999, quotaBytes: 10_000_000 };
    restoreApi = stubApi(overviewRoutes(nearlyFull));
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

describe('Sessions page — the directory filter', () => {
  const OLD_DIR = '/tmp/only-old';
  const RECENT_DIR = '/tmp/recent';

  /**
   * Exactly what the server hands the page once the index outgrows the 500-row
   * cap: a full page of rows from ONE directory, a `total` that says 501 matched,
   * and a directory list naming the one whose sessions all fell off the page.
   * A shorter fixture would not reach the footnote, which is the moment the page
   * tells the user to reach older sessions with this very filter.
   */
  function truncatedList(): SessionsResponse {
    const sessions: SessionRow[] = Array.from({ length: 500 }, (_, i) => ({
      id: `new-${i}`,
      path: `${RECENT_DIR}/new-${i}.jsonl`,
      cwd: RECENT_DIR,
      name: `Session ${i}`,
      created: 10_000 + i,
      modified: 10_000 + i,
      messageCount: 1,
      firstMessage: null,
      models: [],
      inputTokens: 0,
      outputTokens: 0,
    }));
    return { sessions, total: 501, tokens: 0, cwds: [OLD_DIR, RECENT_DIR] };
  }

  /**
   * Open a Radix select by its trigger's aria-label and read the option labels.
   * The listbox is portalled to `document.body`, and the trigger opens on a
   * primary MOUSE pointerdown — nothing else in Radix's handler qualifies.
   */
  async function openOptions(label: string): Promise<string[]> {
    const trigger = [...mounted!.container.querySelectorAll('button')].find(
      (b) => b.getAttribute('aria-label') === label,
    );
    expect(trigger).toBeDefined();
    const { act } = await import('react');
    await act(async () => {
      trigger?.dispatchEvent(
        new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerId: 1, pointerType: 'mouse' }),
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(trigger?.getAttribute('aria-expanded')).toBe('true');
    return [...document.body.querySelectorAll('[role="option"]')].map((o) => (o.textContent ?? '').trim());
  }

  it('offers every indexed directory, not only those on the page of rows it was served', async () => {
    // The page's own footnote sends the user to this dropdown to reach older
    // sessions. Building its options from the served rows makes that advice
    // circular: the directories worth choosing are exactly the ones missing.
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Sessions)),
      { '/sessions': truncatedList() },
      'Session 0',
    );
    expect(text).toContain('Showing the newest 500 of 501 matching sessions');
    expect(text).not.toContain(OLD_DIR);

    const options = await openOptions('Filter by directory');
    expect(options).toContain(OLD_DIR);
    expect(options).toContain(RECENT_DIR);
  });
});

describe('Metrics page — per-model bar charts', () => {
  /**
   * Eleven models, because eleven is what the bug report had and because the
   * defect is a function of how many rows share a fixed-height plot. Two of them
   * agree for their first 21 characters on purpose.
   */
  const MODELS = [
    'Gemma-4-31B-IT-UD-Q4_K_XL-mlx',
    'Gemma-4-31B-IT-UD-Q4_K_M-mlx',
    'gemma-4-26b-a4b-nvfp4-mlx',
    'Qwen3.5-30B-A3B-Instruct-mlx',
    'Qwen3.5-30B-A3B-Thinking-mlx',
    'Lfm2.5-8B-A1B-mlx-q4',
    'AgentWorld-35B-UD-Q4_K_XL',
    'Ornith-1.0-35B-UD-Q5_K_M',
    'qwen3-8b',
    'gemma-4-e2b-qat',
    'Harrier-embed-1B',
  ];

  /**
   * Only the three per-model bar charts carry data. The tokens/day, both trend
   * charts and the usage-share pie stay empty on purpose, so every rendered
   * chart in the page is one of the three under test and the assertions cannot
   * accidentally read a numeric y-axis.
   */
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
      modelShare: [],
      totals: { turns: 0, traces: 0, inputTokens: 0, outputTokens: 0, cachedTokens: 0, reasoningTokens: 0 },
    };
  }

  /**
   * Mount the Metrics page with chart measurement stubbed in. The settle anchor
   * is a model id short enough to survive shortening verbatim, so it can only
   * appear once the charts have actually drawn their category axis.
   */
  async function mountMetrics(models: string[], options: { tickLinePx?: number } = {}): Promise<RenderedPage> {
    restoreMetrics = stubChartMetrics(options);
    restoreApi = stubApi({ '/metrics/overview': metricsFixture(models) });
    mounted = await renderPage(createElement(Metrics), (text) => text.includes('qwen3-8b'));
    return mounted;
  }

  /** The three rendered plots, in page order: decode, TTFT, MTP. */
  function plots(page: RenderedPage): Element[] {
    return [...page.container.querySelectorAll('.recharts-responsive-container')];
  }

  /**
   * Category tick labels of one plot with the geometry that decides whether they
   * are readable: `left` is where the glyphs start, and a left-of-zero label is
   * outside the SVG and therefore cut off mid-name.
   */
  function categoryTicks(plot: Element): Array<{ label: string; left: number; y: number }> {
    return [...plot.querySelectorAll('[class*="yAxis-tick-labels"] text')].map((node) => {
      const label = node.textContent ?? '';
      const x = Number(node.getAttribute('x'));
      const width = label.length * TICK_CHAR_PX;
      return {
        label,
        left: node.getAttribute('text-anchor') === 'end' ? x - width : x,
        y: Number(node.getAttribute('y')),
      };
    });
  }

  it('draws a category label for every bar, with room for it', async () => {
    const page = await mountMetrics(MODELS);
    const charts = plots(page);
    expect(charts).toHaveLength(3);
    for (const chart of charts) {
      expect(chart.querySelectorAll('.recharts-bar-rectangle')).toHaveLength(MODELS.length);
      const ticks = categoryTicks(chart);
      // A skipped tick on a numeric axis is inferable; a skipped model name is
      // not. recharts hides overlapping ticks unless `interval={0}` says not to.
      expect(ticks).toHaveLength(MODELS.length);
      // …and `interval={0}` alone would only stack them on top of each other, so
      // the rows must also carry more than a line of text apiece. recharts' own
      // collision rule is the tick height plus its 5px `minTickGap`.
      const gaps = ticks.slice(1).map((tick, i) => tick.y - ticks[i].y);
      for (const gap of gaps) expect(gap).toBeGreaterThan(TICK_LINE_PX + 5);
    }
  });

  it('still names every model when the tick text is taller than the row it sits in', async () => {
    // Sizing the rows generously is not the same as guaranteeing a label. Once
    // the measured text outgrows its row, recharts is back to hiding whichever
    // ticks it thinks collide — and it hides model NAMES, silently. Only
    // `interval={0}` says "draw them all anyway"; without it this render loses
    // five of the eleven.
    //
    // The ids are suffixed rather than reused: recharts memoises every text
    // measurement in a module-level LRU keyed by the string, so a name already
    // measured by another test in this file would keep that test's height and
    // this one would silently assert nothing.
    const page = await mountMetrics(
      MODELS.map((model) => `${model}-tall`),
      { tickLinePx: 34 },
    );
    for (const chart of plots(page)) {
      expect(chart.querySelectorAll('.recharts-bar-rectangle')).toHaveLength(MODELS.length);
      expect(categoryTicks(chart)).toHaveLength(MODELS.length);
    }
  });

  it('keeps every category label inside the plot instead of running off its left edge', async () => {
    const page = await mountMetrics(MODELS);
    for (const chart of plots(page)) {
      const ticks = categoryTicks(chart);
      expect(ticks.length).toBeGreaterThan(0);
      for (const tick of ticks) {
        expect({ label: tick.label, left: tick.left }).toMatchObject({ left: expect.any(Number) });
        expect(tick.left).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('never renders two different models under the same name', async () => {
    // `Gemma-4-31B-IT-UD-Q4_K_XL-mlx` and `Gemma-4-31B-IT-UD-Q4_K_M-mlx` agree
    // for 21 characters, so any head-keeping shortener collapses them onto one
    // string — two rows of a comparison chart claiming to be the same model.
    const page = await mountMetrics(MODELS);
    for (const chart of plots(page)) {
      const labels = categoryTicks(chart).map((tick) => tick.label);
      expect(new Set(labels).size).toBe(labels.length);
    }
    const rendered = categoryTicks(plots(page)[0]).map((tick) => tick.label);
    expect(rendered).toContain('Gemma-4-31B-I…-Q4_K_XL-mlx');
    expect(rendered).toContain('Gemma-4-31B-I…D-Q4_K_M-mlx');
  });

  it('numbers the survivors when shortening cannot separate two ids at all', async () => {
    // The residual case the render tests cannot reach: two ids that differ only
    // past the budget on BOTH ends. A label may be ugly; it may not be a lie.
    const head = 'a'.repeat(20);
    const tail = 'b'.repeat(20);
    const labels = categoryLabels([`${head}X${tail}`, `${head}Y${tail}`]);
    expect(new Set(labels.values()).size).toBe(2);
    expect([...labels.values()][1]).toMatch(/ \(2\)$/);
  });

  it('shortens from the middle so both ends of an id survive', () => {
    expect(truncateMiddle('Gemma-4-31B-IT-UD-Q4_K_XL-mlx', 26)).toBe('Gemma-4-31B-I…-Q4_K_XL-mlx');
    expect(truncateMiddle('Gemma-4-31B-IT-UD-Q4_K_XL-mlx', 26)).toHaveLength(26);
    // Short enough already: untouched, no ellipsis.
    expect(truncateMiddle('qwen3-8b', 26)).toBe('qwen3-8b');
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

describe('Models page — the Install affordance', () => {
  const REPO = 'Brooooooklyn/Qwen3.6-27B-NVFP4-mlx';

  function catalogRoutes(item: Partial<CatalogItem>, jobs: DownloadJob[] = []): Record<string, unknown> {
    const models: ModelsResponse = { models: [], warnings: [], dir: '/models' };
    const catalog: CatalogResponse = {
      items: [
        {
          label: 'Qwen3.6-27B',
          hfRepo: REPO,
          sizeGb: 22.2,
          description: 'Best tool use',
          slug: 'qwen3.6-27b-nvfp4-mlx',
          installed: false,
          present: false,
          blockedByForeignDir: false,
          ...item,
        },
      ],
    };
    const downloads: DownloadsResponse = { jobs };
    return { '/models': models, '/catalog': catalog, '/downloads': downloads };
  }

  /** Trimmed label of every rendered button. */
  function buttonLabels(): string[] {
    return Array.from(mounted!.container.querySelectorAll('button')).map((b) => (b.textContent ?? '').trim());
  }

  function downloadJob(overrides: Partial<DownloadJob>): DownloadJob {
    return { id: 'job-1', repo: REPO, state: 'running', receivedBytes: 0, totalBytes: 1024, ...overrides };
  }

  /** Job ids every mounted card subscribed to, in order. */
  let openedStreams: string[] = [];

  /** Deliver one download event to every open subscription, as the runtime would. */
  let emitDownload: (type: string, payload: Record<string, unknown>) => void = () => {};

  /**
   * Stand in for the runtime's download subscribe. Hydrating a running job mounts
   * `DownloadProgress`, which subscribes; this records the job id it named and,
   * unlike a pure no-op, can DELIVER events — a terminal event is the only way to
   * reach what the page does when a job settles from the subscription rather than
   * from a request it made itself, which no call-level assertion can observe.
   *
   * The events travel the real port, so a test that emits one has to let the hop
   * land ({@link settle}) before asserting.
   */
  function downloadSubscribeStub(): (jobId: string, listener: (event: DownloadEvent) => void) => () => void {
    const listeners = new Set<(event: DownloadEvent) => void>();
    emitDownload = (type, payload) => {
      // Safe to iterate live: an unsubscribe can only follow the port hop this
      // emit starts, never interleave with it.
      for (const listener of listeners) listener({ type, ...payload } as unknown as DownloadEvent);
    };
    return (jobId, listener) => {
      openedStreams.push(jobId);
      listeners.add(listener);
      return () => listeners.delete(listener);
    };
  }

  /**
   * Install the route stub behind a recorder. A dismiss has no rendered
   * consequence on this page — it is a `DELETE` and nothing else — so these
   * tests have to read the calls, not only the DOM.
   */
  function recordRequests(routes: Record<string, unknown>): Array<{ method: string; url: string }> {
    const calls: Array<{ method: string; url: string }> = [];
    openedStreams = [];
    restoreApi = stubApi(routes, {
      onCall: (call) => calls.push({ method: call.method, url: call.path }),
      subscribe: downloadSubscribeStub(),
    });
    return calls;
  }

  /** Let a port hop (and the render it causes) land. */
  async function settle(): Promise<void> {
    const { act } = await import('react');
    for (let i = 0; i < 10; i++) {
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
    }
  }

  /** How many times the page fetched `path` — a refetch is otherwise invisible. */
  function gets(calls: Array<{ method: string; url: string }>, path: string): number {
    return calls.filter((call) => call.method === 'GET' && call.url.split('?')[0] === path).length;
  }

  /** Job ids the page asked the server to dismiss, in order. */
  function dismissed(calls: Array<{ method: string; url: string }>): string[] {
    return calls.filter((call) => call.method === 'DELETE').map((call) => call.url.replace('/api/downloads/', ''));
  }

  async function mountModels(): Promise<void> {
    mounted = await renderPage(createElement(Models), (text) => text.includes('Qwen3.6-27B'));
    // The hydration effect fires once `/downloads` lands, which is after the
    // catalog text the settle condition waits on; give it and its calls room.
    await settle();
  }

  afterEach(() => {
    emitDownload = () => {};
    openedStreams = [];
  });

  it('offers Install for a recommended model that is simply absent', async () => {
    // The over-correction guard: the card must keep its button when the slug is
    // free, or the blocked branch below could be "fixed" by never offering Install.
    await mount(createElement(Models), catalogRoutes({}), 'Qwen3.6-27B');
    expect(buttonLabels()).toContain('Install');
  });

  it('states the blockage instead of an Install that the runner always refuses', async () => {
    // `<slug>` is occupied by an unowned directory (an interrupted `mlx download`).
    // The download's ownership preflight refuses it every time, so the button would
    // do nothing but raise a red toast.
    const text = await mount(createElement(Models), catalogRoutes({ blockedByForeignDir: true }), 'Qwen3.6-27B');
    expect(buttonLabels()).not.toContain('Install');
    // And it says WHICH directory is in the way and what to do about it.
    expect(text).toContain('qwen3.6-27b-nvfp4-mlx');
    expect(text).toMatch(/remove/i);
  });

  it('still shows an installed model as installed', async () => {
    await mount(createElement(Models), catalogRoutes({ present: true, installed: true }), 'Qwen3.6-27B');
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Install');
  });

  it('dismisses a job that settled while the page was unmounted, without a word', async () => {
    // `DownloadProgress` sends the terminal DELETE, and it was not mounted when
    // these jobs settled — so nothing dismissed them and the server registry
    // keeps the rows for the life of the process, with the Overview tile
    // reporting "1 completed" beside a model that is simply installed.
    const failed = vi.spyOn(toast, 'error');
    try {
      const calls = recordRequests({
        ...catalogRoutes({ present: true, installed: true }, [
          downloadJob({ id: 'job-done', state: 'done' }),
          downloadJob({ id: 'job-gone', state: 'cancelled' }),
        ]),
        '/downloads/job-done': { cancelled: true, id: 'job-done' },
        '/downloads/job-gone': { cancelled: true, id: 'job-gone' },
      });
      await mountModels();
      expect(dismissed(calls)).toEqual(['job-done', 'job-gone']);
      // Both outcomes are already known to the user — the model is installed, the
      // cancel was their own — so neither may raise a failure notice.
      expect(failed.mock.calls).toEqual([]);
      expect(buttonLabels()).toContain('Installed');
    } finally {
      failed.mockRestore();
    }
  });

  it('reverts to Install when a job is cancelled from somewhere else', async () => {
    // The card never clicked Cancel. The server answers a cancel while the job is
    // still `running` — it lets `processJob` emit the terminal frame as it unwinds
    // — so a cancel from a second tab, or from `shutdown()` aborting the in-flight
    // job, arrives only as a stream frame. `cancelled` used to render "Cancelled"
    // and call nothing back, unlike `done`/`error`, so the card sat on a stopped
    // job beside a live Cancel button with nothing to correct it: no query polls
    // and `downloads` is never reloaded.
    const failed = vi.spyOn(toast, 'error');
    try {
      recordRequests({
        ...catalogRoutes({}, [downloadJob({ id: 'job-run', state: 'running' })]),
        '/downloads/job-run': { cancelled: true, id: 'job-run' },
      });
      await mountModels();
      expect(buttonLabels()).toContain('Cancel');

      emitDownload('cancelled', { id: 'job-run' });
      await settle();
      expect(buttonLabels()).toContain('Install');
      expect(buttonLabels()).not.toContain('Cancel');
      // A cancel is not a failure, and whoever issued it already saw their own
      // confirmation — so this must stay silent, unlike the error path.
      expect(failed.mock.calls).toEqual([]);
    } finally {
      failed.mockRestore();
    }
  });

  it('reloads a catalog it read from BEFORE the download published', async () => {
    // Mounting ACROSS a publish reads the two snapshots either side of it. The
    // runner sets `job.state = 'done'` strictly AFTER `publish()` renames the
    // staging dir into place, so `/catalog` answering first (`present: false`)
    // and `/downloads` answering after (`done`) is an ordinary interleaving.
    //
    // Nothing repaired it: `useJson` has no polling and no focus revalidate,
    // `downloads.reload()` has no call site, and the terminal job is deliberately
    // kept out of `active` — so no `DownloadProgress` mounts and the reload in
    // `onDownloadDone` never runs. The card offered Install for a model that was
    // already installed, and clicking it cost a full round trip to reach `done`.
    const absent = catalogRoutes({}, [downloadJob({ id: 'job-done', state: 'done' })]);
    const published = catalogRoutes({ present: true, installed: true }, [
      downloadJob({ id: 'job-done', state: 'done' }),
    ]);
    const calls = recordRequests({
      ...absent,
      '/catalog': sequence(absent['/catalog'], published['/catalog']),
      '/models': sequence(absent['/models'], published['/models']),
      '/downloads/job-done': { cancelled: true, id: 'job-done' },
    });
    await mountModels();
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Install');
    // Exactly one repair, and exactly one dismiss: `reload` bumps only its own
    // hook's nonce and nothing reloads `/downloads`, so `downloads.data` keeps one
    // identity per mount and the effect body runs once. These counts catch a
    // BOUNDED extra refetch. They do not catch an unbounded one — naming the
    // unstable `models`/`catalog` objects in the effect's deps instead of their
    // stable `reload`s spins forever and wedges the run (SIGKILL) rather than
    // failing here, the same way the FIFO cases in models.test.ts do.
    expect(gets(calls, '/api/catalog')).toBe(2);
    expect(gets(calls, '/api/models')).toBe(2);
    expect(dismissed(calls)).toEqual(['job-done']);
  });

  it('reloads after a FAILED job too — a failure can still leave the model installed', async () => {
    // Why the reload is not gated on `done`. `publish()` renames staging into the
    // final dir and only THEN removes the backup, and that `rm` runs inside the
    // job's try: an EPERM there reports `error` for a model that is on disk.
    // (`recoverBackup` restoring a crashed publish before a cancel does the same.)
    // Gating on `done` would leave those cases showing Install for an installed
    // model — the very bug this repairs, in the state that reports a failure.
    const failed = vi.spyOn(toast, 'error');
    try {
      const absent = catalogRoutes({}, [downloadJob({ id: 'job-bad', state: 'error' })]);
      const published = catalogRoutes({ present: true, installed: true }, [
        downloadJob({ id: 'job-bad', state: 'error' }),
      ]);
      const calls = recordRequests({
        ...absent,
        '/catalog': sequence(absent['/catalog'], published['/catalog']),
        '/models': sequence(absent['/models'], published['/models']),
        '/downloads/job-bad': { cancelled: true, id: 'job-bad' },
      });
      await mountModels();
      expect(buttonLabels()).toContain('Installed');
      expect(gets(calls, '/api/catalog')).toBe(2);
      // The failure is still announced — repairing the view must not swallow it.
      expect(failed.mock.calls.length).toBe(1);
    } finally {
      failed.mockRestore();
    }
  });

  it('says a download failed while the page was unmounted rather than dismissing it in silence', async () => {
    // The failed job's only trace: the card reverts to a plain Install, exactly
    // as if the download had never been started. Dismissing it without a word
    // is the one terminal state that loses information the user cannot recover
    // from anywhere else on the page.
    const failed = vi.spyOn(toast, 'error');
    try {
      const calls = recordRequests({
        ...catalogRoutes({}, [downloadJob({ id: 'job-bad', state: 'error' })]),
        '/downloads/job-bad': { cancelled: true, id: 'job-bad' },
      });
      await mountModels();
      expect(
        failed.mock.calls.map((call) => [call[0], (call[1] as { description?: string } | undefined)?.description]),
      ).toEqual([['Download failed', REPO]]);
      expect(dismissed(calls)).toEqual(['job-bad']);
      // …and the card is back at Install, ready for the retry.
      expect(buttonLabels()).toContain('Install');
    } finally {
      failed.mockRestore();
    }
  });

  it('still hydrates a running job, and never dismisses it', async () => {
    // The over-correction guard. A dismiss loop that reaches a live job would
    // cancel a multi-GB download that is still running.
    const calls = recordRequests(catalogRoutes({}, [downloadJob({ id: 'job-run', state: 'running' })]));
    await mountModels();
    expect(openedStreams).toEqual(['job-run']);
    expect(buttonLabels()).toContain('Cancel');
    expect(buttonLabels()).not.toContain('Install');
    expect(dismissed(calls)).toEqual([]);
  });

  it('drops a stale active card when a replacement runtime omits the job', async () => {
    recordRequests(catalogRoutes({}, [downloadJob({ id: 'job-old', state: 'running' })]));
    await mountModels();
    expect(buttonLabels()).toContain('Cancel');

    // CONTROL PANEL restarted and handed the same mounted tree a new port. The
    // replacement runtime has no download registry entry for the old process's
    // job, so its GET is authoritative absence.
    const oldRuntime = restoreApi!;
    let newRuntime: (() => void) | undefined;
    try {
      newRuntime = stubApi(catalogRoutes({}, []), { subscribe: downloadSubscribeStub() });
      restoreApi = newRuntime;
      await settle();

      expect(buttonLabels()).toContain('Install');
      expect(buttonLabels()).not.toContain('Cancel');
    } finally {
      newRuntime?.();
      restoreApi = undefined;
      oldRuntime();
    }
  });

  it('replaces a stale job id when the replacement runtime has a new job for the same repo', async () => {
    recordRequests(catalogRoutes({}, [downloadJob({ id: 'job-old', state: 'running' })]));
    await mountModels();
    expect(openedStreams).toContain('job-old');

    const oldRuntime = restoreApi!;
    let newRuntime: (() => void) | undefined;
    try {
      openedStreams = [];
      newRuntime = stubApi(catalogRoutes({}, [downloadJob({ id: 'job-new', state: 'running' })]), {
        subscribe: downloadSubscribeStub(),
      });
      restoreApi = newRuntime;
      await settle();

      // DownloadProgress may briefly rebind the old id on the connection bump,
      // but the authoritative snapshot must replace it with the new runtime's id.
      expect(openedStreams.at(-1)).toBe('job-new');
      expect(buttonLabels()).toContain('Cancel');
    } finally {
      newRuntime?.();
      restoreApi = undefined;
      oldRuntime();
    }
  });

  it('keeps a current-generation POST result when an older concurrent downloads GET lands later', async () => {
    let resolveDownloads!: (body: DownloadsResponse) => void;
    const olderDownloads = new Promise<DownloadsResponse>((resolve) => {
      resolveDownloads = resolve;
    });
    const subscriptions: string[] = [];
    const routes = catalogRoutes({});
    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    const dispose = serveRuntimeOverPort(
      {
        call: async (call: ApiCall) => {
          const path = call.path.replace(/^\/api/u, '');
          if (call.method === 'GET' && path === '/downloads') {
            return { ok: true as const, status: 200, body: await olderDownloads };
          }
          if (call.method === 'POST' && path === '/downloads') {
            return { ok: true as const, status: 202, body: { id: 'job-local', repo: REPO } };
          }
          if (call.method === 'GET' && Object.hasOwn(routes, path)) {
            return { ok: true as const, status: 200, body: routes[path] };
          }
          return { ok: false as const, status: 404, code: 'E_NOT_FOUND', message: `no stub for ${path}` };
        },
        subscribe: (jobId) => {
          subscriptions.push(jobId);
          return () => {};
        },
      },
      bindEventTargetPort(port2),
    );
    connectDashboardApi(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
    restoreApi = () => {
      disconnectDashboardApi();
      dispose();
      port1.close();
      port2.close();
    };

    mounted = await renderPage(createElement(Models), (text) => text.includes('Qwen3.6-27B'));
    const install = [...mounted.container.querySelectorAll('button')].find((button) =>
      button.textContent?.includes('Install'),
    );
    expect(install).toBeDefined();
    await act(async () => {
      install!.click();
    });
    await settle();
    expect(subscriptions).toContain('job-local');
    expect(buttonLabels()).toContain('Cancel');

    // This GET began before the POST and therefore cannot disprove the job the
    // POST just returned, even though its response happens to arrive afterward.
    resolveDownloads({ jobs: [] });
    await settle();
    expect(buttonLabels()).toContain('Cancel');
    expect(buttonLabels()).not.toContain('Install');
  });

  it('hydrates a job that is mid-publish, and never dismisses it', async () => {
    // `committing` is the brief non-cancellable publish window. It is not
    // `running`, so a dismiss gated on `state !== "running"` fires a DELETE at a
    // job that is still installing a model — which is why the gate is the
    // terminal set instead.
    //
    // It is equally not absent: the catalog snapshot is read before the publish
    // renames the model into place, so an unhydrated card offers Install for a
    // model that is installing, subscribes to nothing, and — since neither query
    // polls — never learns the job finished.
    const calls = recordRequests(catalogRoutes({}, [downloadJob({ id: 'job-commit', state: 'committing' })]));
    await mountModels();
    expect(openedStreams).toEqual(['job-commit']);
    expect(buttonLabels()).not.toContain('Install');
    expect(dismissed(calls)).toEqual([]);
  });

  it('offers no Cancel for a publish the server refuses to cancel', async () => {
    // `cancel()` rejects a `committing` job outright, so the route 404s and the
    // button can only ever raise "Failed to cancel download".
    recordRequests(catalogRoutes({}, [downloadJob({ id: 'job-commit', state: 'committing' })]));
    await mountModels();
    expect(buttonLabels()).not.toContain('Cancel');
  });
});
