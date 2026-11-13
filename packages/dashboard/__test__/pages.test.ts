/**
 * @vitest-environment happy-dom
 */

/**
 * Render guards for the SPA pages that carry cold-tier logic in JSX.
 *
 * Before this file the suite imported only `ui/src/lib/{api,cold-tier,format,shell}`,
 * so `cache.tsx` and `overview.tsx` could each be reverted to their pre-fix
 * content and the whole suite stayed green: the block-vs-object counts, the
 * cold-tier health tiles and the saturating hit rate all live at the JSX call
 * site, not in a helper. These tests mount the real components against a
 * stubbed `fetch` and assert on rendered text, so a page reverting is a red
 * test.
 *
 * `happy-dom` is scoped to this file by the docblock above — the repo default
 * environment stays `node` for every other suite.
 */

import { MessageChannel } from 'node:worker_threads';

import { categoryLabels, truncateMiddle } from '@/lib/labels';
import { modelShareLabel } from '@/lib/metrics-view';
import { toast } from '@/lib/toast';
import type {
  CacheResponse,
  CacheScope,
  CatalogBlocker,
  CatalogItem,
  CatalogResponse,
  CatalogUpdatesResponse,
  ColdTierHealth,
  DownloadJob,
  DownloadsResponse,
  ModelsResponse,
} from '@/lib/types';
import Cache from '@/pages/cache';
import Metrics from '@/pages/metrics';
import Models from '@/pages/models';
import { blockerPhrase, CatalogDownloadAction, reinstallContent } from '@/pages/models/catalog';
import Overview from '@/pages/overview';
import { type AppCommand, usePageCommands } from '@/shell/commands';
import { act, createElement, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, onTestFinished, vi } from 'vite-plus/test';

import type { ApiResponse } from '../src/api/errors.js';
import type { DownloadEvent } from '../src/download.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import type { ApiCall } from '../src/runtime.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import { splitWhitespace } from '../ui/src/lib/text-scan.js';
import { reloadMountedQueries } from '../ui/src/lib/use-api.js';
import {
  byModelRow,
  dayStart,
  distributions,
  emptySummary,
  live,
  liveModel,
  process,
  summary,
  timeseries,
  turnsPage,
} from './helpers/observability-fixtures.js';
import {
  STUB_FAILURE,
  byRole,
  deferred,
  observeWidth,
  renderPage,
  sequence,
  statValue,
  stubApi,
  waitFor,
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
      { t: dayStart(2), hits: 16_000, misses: 40, bytesWritten: 8 * MIB, bytesRestored: 64 * MIB },
      { t: dayStart(1), hits: 189, misses: 9, bytesWritten: 2 * MIB, bytesRestored: 16 * MIB },
    ],
    scope: {
      root: '/private/tmp/cold/mlx-paged-v1',
      trendWindowDays: 30,
      legacy: { turns: 7, hits: 11, misses: 13 },
      otherRoots: { turns: 3, hits: 17, misses: 19 },
      unattributed: { turns: 0, hits: 0, misses: 0 },
      disabledTurns: 2,
      enabledTurns: 30,
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

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
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
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': cacheFixture() },
      'mlx-paged-v1',
    );
    // F2: the tile total must be blocks + sidecars (138), the same number the
    // age histogram charts, and the sub-line must name both units. Reverting
    // cache.tsx puts back a "Blocks 136" tile beside a 138-object histogram.
    expect(text).toContain('Objects');
    expect(text).toContain('138');
    expect(text).toContain('136 prefix blocks · 2 state sidecars');
    expect(text).toContain('Objects by age');
    expect(text).not.toContain('persisted prefix blocks');
  });

  it('shows the cache root under the home folder as ~/…, the canonical root on hover', async () => {
    await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      {
        '/cache': cacheFixture(),
        '/health': { status: 'ok', modelsDir: '', tracesDir: '', runtimeDir: '', homeDir: '/tmp' },
      },
      'mlx-paged-v1',
    );
    const root = (): string =>
      mounted!.container.querySelector('[title="/private/tmp/cold/mlx-paged-v1"]')?.textContent ?? '';
    await waitFor(() => root() === '~/cold/mlx-paged-v1');
    expect(mounted!.text()).not.toContain('/tmp/cold');
  });

  it('splits the usage tile into block bytes and sidecar bytes', async () => {
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': cacheFixture() },
      'mlx-paged-v1',
    );
    expect(text).toContain('200 MB blocks + 100 MB sidecars');
  });

  it('renders the cold-tier health counters the API now serves', async () => {
    // F3: `corruptionsTotal` / `queueDropsTotal` / `enqueued` / `evictions` are
    // plumbed agent → traces → API for exactly one reason: the stated
    // acceptance bar is "corruptions must be 0" and nothing shipped could show
    // it. Reverting cache.tsx deletes this whole card.
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': cacheFixture() },
      'mlx-paged-v1',
    );
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
      createElement(MemoryRouter, null, createElement(Cache)),
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
      createElement(MemoryRouter, null, createElement(Cache)),
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
          },
        }),
      },
      'mlx-paged-v1',
    );
    expect(text).toContain('3 restores refused · no lookup was performed');
    expect(text).not.toContain('no lookups recorded for this cache');
    expect(text).toContain('refused before lookup · not counted as a lookup');
  });

  it('flags a non-zero cumulative corruption total even when the window delta is 0', async () => {
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
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
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': cacheFixture() },
      'mlx-paged-v1',
    );
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
      createElement(MemoryRouter, null, createElement(Cache)),
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
      createElement(MemoryRouter, null, createElement(Cache)),
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
      createElement(MemoryRouter, null, createElement(Cache)),
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
      createElement(MemoryRouter, null, createElement(Cache)),
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
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': cacheFixture({ health: { sidecarEnqueued: 12, sidecarQueueDrops: 3 } }) },
      'mlx-paged-v1',
    );
    expect(text).toContain('500 objects enqueued · 12 sidecars among them, 3 of the drops');
  });

  it('never rounds a hit rate with real misses up to 100%', async () => {
    // 16189 hits / 49 misses is 99.6982 %. `Math.round` renders that as "100%"
    // directly above the line that prints the 49 misses.
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': cacheFixture() },
      'mlx-paged-v1',
    );
    expect(text).toContain('>99%');
    expect(text).not.toContain('100%');
    expect(text).toContain('16.2K hits · 49 misses · this cache only');
  });

  it('names the rows the root scope excluded instead of folding or dropping them', async () => {
    // F1: the trend is filtered to one cache root. Rows outside it are reported.
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': cacheFixture() },
      'mlx-paged-v1',
    );
    expect(text).toContain('7 turns recorded before cache attribution existed (11 hits · 13 misses)');
    expect(text).toContain('they age out within 30 days');
    expect(text).toContain('3 turns ran against a different cache directory (17 hits · 19 misses)');
    expect(text).toContain('Lookups against this cache root, per day · last 30 days');
  });

  it('names the tier-on-but-unattributed bucket instead of letting its lookups vanish', async () => {
    // MINOR 7: `cold_enabled = 1` with a NULL root used to match no bucket at
    // all, so its hits appeared nowhere — not in the trend, not in any excluded
    // line. The page must say it out loud.
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
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
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': emptyCacheFixture() },
      'mlx-paged-v1',
    );
    expect(text).toContain('gemma4, qwen3, qwen3_5, qwen3_5_moe');
    expect(text).not.toContain('(qwen3 dense)');
    expect(text).toContain('No cold-tier objects persisted yet.');
    expect(text).toContain('Objects appear when a persistent paged cache is used');
  });

  it('points an empty trend at the excluded lookups instead of a dead end', async () => {
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': emptyCacheFixture() },
      'mlx-paged-v1',
    );
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
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Cache)),
      { '/cache': emptyCacheFixture() },
      'mlx-paged-v1',
    );
    // Scoped to the Hit rate tile: '0%' legitimately appears elsewhere on an
    // empty page (quota usage really is 0% of the quota).
    expect(statValue(mounted!.container, 'Hit rate')).toBe('—');
    expect(text).toContain('no lookups recorded for this cache');
  });

  it('labels the evict confirmation in objects, matching the count it shows', async () => {
    // 9a: the dialog counted OBJECTS while its title said "blocks". The
    // confirmation is a portalled dialog, so read it from the document.
    restoreApi = stubApi({ '/cache': cacheFixture() });
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Cache)), (text) =>
      text.includes('mlx-paged-v1'),
    );
    const evict = [...mounted.container.querySelectorAll('button')].find((b) => b.textContent?.includes('Evict older'));
    expect(evict).toBeDefined();
    expect(evict?.disabled).toBe(false);
    await act(async () => {
      evict?.click();
    });
    await waitFor(() => document.querySelector('[role="dialog"]') !== null);
    const dialog = splitWhitespace(document.querySelector('[role="dialog"]')?.textContent ?? '').join(' ');
    expect(dialog).toContain('Evict old cache objects?');
    expect(dialog).not.toContain('Evict old cache blocks?');
    // 7-30d (6) + >30d (2) = 8 objects, 50 MB.
    expect(dialog).toContain('8 objects · 50.0 MB');
  });
});

describe('Overview page', () => {
  function overviewRoutes(cache: CacheResponse): Record<string, unknown> {
    const models: ModelsResponse = {
      models: [],
      warnings: [],
      dir: '/models',
      host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
    };
    const downloads: DownloadsResponse = { jobs: [] };
    return {
      '/models': models,
      '/downloads': downloads,
      '/cache': cache,
      '/metrics/summary': emptySummary(),
      '/metrics/timeseries': timeseries([]),
      '/metrics/turns': turnsPage([]),
    };
  }

  it('counts zero models when only companion weights are installed while retaining their disk usage', async () => {
    const routes = overviewRoutes(cacheFixture());
    routes['/models'] = {
      models: [],
      warnings: [],
      dir: '/models',
      companions: [
        { name: 'qwen3.8-27b-dflash2', path: '/models/qwen3.8-27b-dflash2', sizeBytes: 2 * MIB, fileCount: 2 },
      ],
    };
    await mount(createElement(MemoryRouter, null, createElement(Overview)), routes, 'including companions');
    const tile = byRole(mounted!.container, 'group', 'Local models');
    expect(statValue(mounted!.container, 'Local models')).toBe('0');
    expect(tile?.textContent).toContain('2.0 MB on disk, including companions');
  });

  it('never rounds the cold-cache hit rate up to 100% on the landing page', async () => {
    // The landing page's hit rate saturates like the Cache page's: 16189/16238 is not 100%.
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Overview)),
      overviewRoutes(cacheFixture()),
      'hit rate',
    );
    expect(text).toContain('hit rate >99%');
    expect(text).not.toContain('hit rate 100%');
  });

  it('saturates the quota meter the same way the label is saturated', async () => {
    // `Math.round(quotaFraction * 100)` renders a 99.9%-full quota as a 100%
    // bar. `percentInt` caps it at 99 so the bar and the label agree.
    const nearlyFull = cacheFixture();
    nearlyFull.disk = { ...nearlyFull.disk, totalBytes: 9_999_999, quotaBytes: 10_000_000 };
    restoreApi = stubApi(overviewRoutes(nearlyFull));
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Overview)), (text) =>
      text.includes('of 9.5 MB'),
    );
    const meter = byRole(mounted.container, 'meter', 'Cold cache quota');
    expect(meter?.getAttribute('aria-valuenow')).toBe('99');
    expect(meter?.getAttribute('aria-valuetext')).toContain('>99%');
    expect(mounted.text()).toContain('>99% of 9.5 MB');
  });
});

describe('Metrics page — the Models table', () => {
  /**
   * Eleven models, because eleven is what the bug report had. Two of them agree
   * for their first 21 characters on purpose.
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

  function metricsRoutes(models: string[]): Record<string, unknown> {
    return {
      '/metrics/models': {
        models: models.map((model) => ({ model, modelType: null, firstTs: 0, lastTs: 0, turns: 10 })),
      },
      '/metrics/summary': summary({ byModel: models.map((model, i) => byModelRow(model, 100 - i)) }),
      '/metrics/timeseries': timeseries([]),
      '/models': {
        models: [],
        warnings: [],
        dir: '/models',
        host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
      },
    };
  }

  it('lists every model as its own row', async () => {
    await mount(createElement(MemoryRouter, null, createElement(Metrics)), metricsRoutes(MODELS), 'qwen3-8b');
    const rows = [...mounted!.container.querySelectorAll('tr[data-model]')].map((row) =>
      row.getAttribute('data-model'),
    );
    expect(rows).toHaveLength(MODELS.length);
    expect(new Set(rows)).toEqual(new Set(MODELS));
  });

  it('never renders two different models under the same visible name', async () => {
    // `Gemma-4-31B-IT-UD-Q4_K_XL-mlx` and `Gemma-4-31B-IT-UD-Q4_K_M-mlx` agree
    // for 21 characters, so any head-keeping shortener collapses them onto one
    // string — two rows claiming to be the same model.
    await mount(createElement(MemoryRouter, null, createElement(Metrics)), metricsRoutes(MODELS), 'qwen3-8b');
    const names = [...mounted!.container.querySelectorAll('tr[data-model]')].map((row) =>
      splitWhitespace(row.querySelector('td')?.textContent ?? '').join(' '),
    );
    expect(new Set(names).size).toBe(MODELS.length);
  });

  it('numbers the survivors when shortening cannot separate two ids at all', () => {
    // Two ids that differ only past the budget on BOTH ends. A label may be
    // ugly; it may not be a lie.
    const head = 'a'.repeat(20);
    const tail = 'b'.repeat(20);
    const labels = categoryLabels([`${head}X${tail}`, `${head}Y${tail}`]);
    expect(new Set(labels.values()).size).toBe(2);
    expect([...labels.values()][1].endsWith(' (2)')).toBe(true);
  });

  it('shortens from the middle so both ends of an id survive', () => {
    expect(truncateMiddle('Gemma-4-31B-IT-UD-Q4_K_XL-mlx', 26)).toBe('Gemma-4-31B-I…-Q4_K_XL-mlx');
    expect(truncateMiddle('Gemma-4-31B-IT-UD-Q4_K_XL-mlx', 26)).toHaveLength(26);
    expect(truncateMiddle('qwen3-8b', 26)).toBe('qwen3-8b');
  });
});

describe('Metrics page — model usage share', () => {
  /** `qwen3-8b` holds the most turns AND sorts last alphabetically, which gives the order something to fail on. */
  const SHARE_MODELS = ['qwen3-8b', ...Array.from({ length: 24 }, (_, i) => `Gemma-4-31B-IT-UD-Q4_K_XL-mlx-${i}`)];

  it('lists all 25 models after "Show all 25", in usage rank', async () => {
    restoreApi = stubApi({
      '/metrics/models': { models: [] },
      '/metrics/summary': summary({ byModel: SHARE_MODELS.map((model, i) => byModelRow(model, 100 - i)) }),
      '/metrics/timeseries': timeseries([]),
      '/metrics/distributions': distributions(),
    });
    mounted = await renderPage(
      createElement(MemoryRouter, { initialEntries: ['/metrics?tab=usage'] }, createElement(Metrics)),
      (text) => text.includes('Show all 25'),
    );
    const more = [...mounted.container.querySelectorAll('button')].find((b) => b.textContent === 'Show all 25');
    await act(async () => {
      more?.click();
    });
    const list = (): HTMLElement | null => byRole(mounted!.container, 'list', 'Turns by model (turns)');
    await waitFor(() => (list()?.querySelectorAll('li').length ?? 0) === SHARE_MODELS.length);
    const first = list()?.querySelector('li [data-model]')?.getAttribute('data-model');
    expect(first).toBe('qwen3-8b');
  });

  it('never labels a share 100% while another model still holds turns', () => {
    expect(modelShareLabel(9999, 10_000)).toBe('9,999 turns · >99%');
    expect(modelShareLabel(1, 10_000)).toBe('1 turn · <1%');
    expect(modelShareLabel(10_000, 10_000)).toBe('10,000 turns · 100%');
    expect(modelShareLabel(0, 10_000)).toBe('0 turns · 0%');
  });
});

describe('Models page — the Install affordance', () => {
  const REPO = 'Brooooooklyn/Qwen3.8-27B-MXFP4-mlx';
  /** Card heading the settle predicates wait on, and the slug it installs to. */
  const LABEL = 'Qwen3.8-27B';
  const SLUG = 'qwen3.8-27b-mxfp4-mlx';

  function catalogRoutes(
    item: Partial<CatalogItem>,
    jobs: DownloadJob[] = [],
    updates: CatalogUpdatesResponse = { items: [] },
  ): Record<string, unknown> {
    const models: ModelsResponse = {
      models: [],
      warnings: [],
      dir: '/models',
      host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
    };
    const catalog: CatalogResponse = {
      items: [
        {
          label: LABEL,
          hfRepo: REPO,
          sizeGb: 23.3,
          description: 'Best tool use',
          slug: SLUG,
          installed: false,
          present: false,
          blockedByForeignDir: false,
          blocker: null,
          localRevision: null,
          localAssetsRepo: null,
          localAssetsRevision: null,
          ...item,
        },
      ],
    };
    const downloads: DownloadsResponse = { jobs };
    return { '/models': models, '/catalog': catalog, '/catalog/updates': updates, '/downloads': downloads };
  }

  /** Trimmed label of every rendered button, plus each card's "Installed" status (text beside a check, not a button). */
  function buttonLabels(): string[] {
    const buttons = Array.from(mounted!.container.querySelectorAll('button')).map((b) => (b.textContent ?? '').trim());
    const installed = Array.from(mounted!.container.querySelectorAll('[data-status="installed"]')).map(
      () => 'Installed',
    );
    return [...buttons, ...installed];
  }

  /** Labels of the buttons a user can actually press — a disabled one cannot act. */
  function liveButtonLabels(): string[] {
    return Array.from(mounted!.container.querySelectorAll('button'))
      .filter((button) => !button.disabled)
      .map((button) => (button.textContent ?? '').trim());
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

  /** Labels of the palette commands the mounted page registers, as of the last render. */
  let paletteLabels: string[] = [];
  function PaletteProbe(): null {
    paletteLabels = usePageCommands().map((command) => String(command.label));
    return null;
  }

  async function mountModels(): Promise<void> {
    mounted = await renderPage(
      createElement(MemoryRouter, null, createElement(Models), createElement(PaletteProbe)),
      (text) => text.includes(LABEL),
    );
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
    await mount(createElement(MemoryRouter, null, createElement(Models)), catalogRoutes({}), LABEL);
    expect(buttonLabels()).toContain('Install');
  });

  function draftItem(present = false): NonNullable<CatalogItem['draft']> {
    return {
      label: 'DFlash2',
      hfRepo: 'z-lab/Qwen3.8-27B-DFlash2',
      sizeGb: 3.85,
      slug: 'qwen3.8-27b-dflash2',
      present,
      installed: present,
      blockedByForeignDir: false,
      blocker: null,
      localRevision: null,
    };
  }

  it('keeps companion storage out of the model table and count, with a separate confirmed delete action', async () => {
    const draft = draftItem(true);
    const companion = { name: draft.slug, path: `/models/${draft.slug}`, sizeBytes: 2 * MIB, fileCount: 2 };
    const routes = catalogRoutes({ draft });
    const catalog = routes['/catalog'] as CatalogResponse;
    routes['/catalog'] = sequence(catalog, { items: [{ ...catalog.items[0], draft: draftItem(false) }] });
    const initial: ModelsResponse = {
      models: [],
      companions: [companion],
      warnings: [],
      dir: '/models',
      host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
    };
    routes['/models'] = sequence(initial, { ...initial, companions: [] });
    routes[`/models/${draft.slug}`] = { deleted: true, name: draft.slug };
    const calls = recordRequests(routes);
    await mountModels();
    expect(statValue(mounted!.container, 'Installed')).toBe('0');
    expect(statValue(mounted!.container, 'On disk')).toBe('2.0 MB');
    expect(byRole(mounted!.container, 'group', 'On disk')?.textContent).toContain('including companions');
    expect(mounted!.container.querySelector('tbody')?.textContent ?? '').not.toContain(draft.slug);
    const toggle = [...mounted!.container.querySelectorAll('button')].find((button) =>
      button.textContent?.includes('Companion weights (1)'),
    )!;
    await act(async () => toggle.click());
    const remove = mounted!.container.querySelector<HTMLButtonElement>(
      `button[aria-label="Delete companion ${draft.slug}"]`,
    )!;
    await act(async () => remove.click());
    expect(calls.filter((call) => call.method === 'DELETE')).toEqual([]);
    const dialog = document.querySelector('[role="dialog"]')!;
    expect(dialog.textContent).toContain('Delete companion weights?');
    expect(dialog.textContent).toContain(draft.slug);
    const confirm = [...dialog.querySelectorAll('button')].find((button) => button.textContent?.trim() === 'Delete')!;
    await act(async () => confirm.click());
    await settle();
    expect(calls.filter((call) => call.method === 'DELETE')).toEqual([
      { method: 'DELETE', url: `/api/models/${draft.slug}` },
    ]);
    await waitFor(() => !(mounted!.container.textContent ?? '').includes('Companion weights (1)'));
    expect(statValue(mounted!.container, 'On disk')).not.toContain('2.0');
    expect(buttonLabels()).toContain('Install DFlash2');
  });

  it.each([false, true])(
    'groups the companion download inside its installed target card (draft present: %s)',
    async (present) => {
      const draft = draftItem(present);
      await mount(
        createElement(MemoryRouter, null, createElement(Models)),
        catalogRoutes({ present: true, installed: true, draft }),
        LABEL,
      );
      const link = mounted!.container.querySelector(`a[href="https://huggingface.co/${draft.hfRepo}"]`)!;
      const card = link.closest('[data-model]')!;
      const target = mounted!.container.querySelector(`a[href="https://huggingface.co/${REPO}"]`)!;
      expect(card).toBe(target.closest('[data-model]'));
      expect(link.textContent).toBe('DFlash2');
      const companion = link.closest('[role="group"]')!;
      expect(companion.textContent).toContain('Used automatically');
      expect(companion.textContent).toContain('3.85 GB');
      const button = companion.querySelector('button');
      if (present) {
        expect(companion.querySelector('[data-status="installed"]')?.textContent).toContain('Installed');
        expect(button).toBeNull();
      } else {
        expect(button?.textContent?.trim()).toBe('Install DFlash2');
        expect(button?.disabled).toBe(false);
      }
    },
  );

  it('resumes a companion download and subscribes to its own progress', async () => {
    const draft = draftItem();
    recordRequests(
      catalogRoutes({ present: true, installed: true, draft }, [downloadJob({ id: 'draft-job', repo: draft.hfRepo })]),
    );
    await mountModels();
    expect(openedStreams).toContain('draft-job');
    const card = mounted!.container
      .querySelector(`a[href="https://huggingface.co/${draft.hfRepo}"]`)!
      .closest('[data-model]')!;
    const companion = card.querySelector('[role="group"]')!;
    expect(companion.textContent).toContain('Cancel');
    expect(companion.textContent).not.toContain('Installed');
    expect(card.textContent).toContain('Installed');
  });

  const LEFTOVER = 'imatrix_unsloth.gguf_file';
  const BLOCKER: CatalogBlocker = {
    kind: 'directory',
    fileCount: 1,
    sizeBytes: 192_223_904,
    truncated: false,
    names: [LEFTOVER],
    listed: false,
    modelLike: false,
    holdsModel: false,
    linkTarget: null,
    dangling: false,
  };

  it('names what is in the way and offers Reinstall instead of an Install the runner refuses', async () => {
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ blockedByForeignDir: true, blocker: BLOCKER }),
      LABEL,
    );
    expect(buttonLabels()).toContain('Reinstall');
    expect(buttonLabels()).not.toContain('Install');
    expect(text).toContain('Needs cleanup');
    expect(text).toContain(`${SLUG} is in the way: a directory with 1 file (183 MB) that the app did not create.`);
    expect(text).toContain('Reinstall downloads the model, then removes this directory.');
    expect(text).not.toContain('already exists and was not created by the app');
    expect(text.toLowerCase()).toContain('remove');
  });

  it.each<[string, boolean, string]>([
    ['not listed', false, 'Move or delete it in Finder to install this one.'],
    ['listed in Local models', true, 'Delete it in Local models to install this one.'],
  ])('offers no Reinstall over a model mlx serve lists (%s)', async (_name, listed, hint) => {
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ blockedByForeignDir: true, blocker: { ...BLOCKER, holdsModel: true, listed } }),
      LABEL,
    );
    expect(buttonLabels()).not.toContain('Reinstall');
    expect(buttonLabels()).not.toContain('Install');
    expect(text).toContain('Needs cleanup');
    expect(text).toContain(
      `${SLUG} holds a model that the app did not create, so Reinstall cannot replace it. ${hint}`,
    );
  });

  /** The visible text of a confirmation description. */
  function textOf(node: ReactNode): string {
    const holder = document.createElement('div');
    holder.innerHTML = renderToStaticMarkup(createElement('div', null, node));
    return splitWhitespace(holder.textContent ?? '').join(' ');
  }

  const blocker = (overrides: Partial<CatalogBlocker>): CatalogBlocker => ({ ...BLOCKER, names: [], ...overrides });

  it.each<[string, Partial<CatalogBlocker>, string, string]>([
    ['empty', { fileCount: 0, sizeBytes: 0 }, 'an empty directory', 'directory'],
    ['one file', { fileCount: 1, sizeBytes: 2048 }, 'a directory with 1 file (2.0\u00a0KB)', 'directory'],
    ['many files', { fileCount: 1234, sizeBytes: 2048 }, 'a directory with 1,234 files (2.0\u00a0KB)', 'directory'],
    ['truncated', { fileCount: 2000, truncated: true }, 'a directory with more than 2,000 files', 'directory'],
    [
      'truncated before any file was counted',
      { fileCount: 0, sizeBytes: 0, truncated: true },
      'a directory the app could not fully read',
      'directory',
    ],
    ['a file', { kind: 'file', fileCount: 1, sizeBytes: 2048 }, 'a file (2.0\u00a0KB)', 'file'],
    ['a link', { kind: 'symlink', fileCount: 0, sizeBytes: 0 }, 'a link', 'link'],
  ])('phrases the blocker on the card: %s', (_name, overrides, phrase, noun) => {
    expect(blockerPhrase(blocker(overrides))).toEqual({ phrase, noun });
  });

  const handlers = { onReinstall: () => Promise.resolve(), onReinstallStarted: () => {} };

  it.each<[string, Partial<CatalogBlocker>, string | null]>([
    ['empty', { fileCount: 0, sizeBytes: 0 }, 'The directory is empty.'],
    [
      'named files',
      { fileCount: 2, sizeBytes: 2048, names: ['a.bin', 'sub/b.bin'] },
      'It holds 2 files (2.0 KB): a.bin, sub/b.bin.',
    ],
    ['one named file', { fileCount: 1, sizeBytes: 2048, names: ['a.bin'] }, 'It holds 1 file (2.0 KB): a.bin.'],
    ['four files', { fileCount: 4, sizeBytes: 2048 }, 'It holds 4 files (2.0 KB).'],
    [
      'truncated',
      { fileCount: 2000, sizeBytes: 2048, truncated: true },
      'It holds more than 2,000 files (more than 2.0 KB).',
    ],
    [
      'truncated before any file was counted',
      { fileCount: 0, sizeBytes: 0, truncated: true },
      'The app could not read all of it, so its size is not known.',
    ],
    ['a file', { kind: 'file', fileCount: 1, sizeBytes: 2048 }, 'The file is 2.0 KB.'],
    ['a live link', { kind: 'symlink', fileCount: 0, linkTarget: '/Volumes/x/m' }, 'It points to /Volumes/x/m.'],
    [
      'a dangling link',
      { kind: 'symlink', fileCount: 0, linkTarget: '/Volumes/x/m', dangling: true },
      'It points to /Volumes/x/m, which does not exist.',
    ],
    ['a link with no text', { kind: 'symlink', fileCount: 0, linkTarget: null }, null],
  ])('says what Reinstall deletes: %s', (_name, overrides, contents) => {
    const item = blocker(overrides);
    const noun = blockerPhrase(item).noun;
    const text = textOf(reinstallContent({ slug: SLUG, blocker: item }, handlers).description);
    const first = `This downloads the model, then permanently deletes the ${noun} ${SLUG} and installs the model in its place.`;
    const last = 'Nothing is deleted if the download fails or you cancel it.';
    expect(text).toBe(contents === null ? `${first} ${last}` : `${first} ${contents} ${last}`);
  });

  it('truncates a long file name in the middle and keeps the full name on hover', () => {
    const long = `${'x'.repeat(30)}-middle-${'y'.repeat(30)}.gguf`;
    const html = renderToStaticMarkup(
      createElement(
        'div',
        null,
        reinstallContent({ slug: SLUG, blocker: blocker({ fileCount: 1, names: [long] }) }, handlers).description,
      ),
    );
    expect(html).toContain(`title="${long}"`);
    expect(html).toContain(truncateMiddle(long, 40));
  });

  it('never lets a size wrap between its number and its unit', () => {
    const html = renderToStaticMarkup(
      createElement(
        'div',
        null,
        reinstallContent({ slug: SLUG, blocker: blocker({ fileCount: 4, sizeBytes: 2048 }) }, handlers).description,
      ),
    );
    expect(html).toContain('It holds 4 files (2.0 KB).');
  });

  it.each<[string, Partial<CatalogBlocker>, string | undefined]>([
    [
      'a link, even one that lists',
      { kind: 'symlink', listed: true, modelLike: true },
      'This is a link. Only the link is removed. The target is not changed.',
    ],
    [
      'a listed directory',
      { listed: true, modelLike: true },
      'Local models lists this directory. Make sure that you do not need it.',
    ],
    ['a config.json', { modelLike: true }, 'This directory has a config.json. Make sure that you do not need it.'],
    ['nothing to warn about', {}, undefined],
  ])('warns first about %s', (_name, overrides, notice) => {
    const content = reinstallContent({ slug: SLUG, blocker: blocker(overrides) }, handlers);
    expect(content.notice).toBe(notice);
    expect(content).toMatchObject({
      title: 'Reinstall model?',
      confirmLabel: 'Reinstall',
      workingLabel: 'Starting…',
      successLabel: 'Started',
      tone: 'danger',
      errorTitle: 'Could not start the download',
    });
  });

  /**
   * Serve the catalog routes plus a POST /downloads answered by `post`. The route stub matches by path alone, and a
   * Reinstall needs the GET and the POST of `/downloads` to answer differently.
   */
  function serveReinstall(
    routes: Record<string, unknown>,
    post: (body: unknown) => ApiResponse,
  ): Array<{ method: string; path: string; body: unknown }> {
    const calls: Array<{ method: string; path: string; body: unknown }> = [];
    openedStreams = [];
    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    const dispose = serveRuntimeOverPort(
      {
        call: async (call: ApiCall) => {
          const path = (call.path.startsWith('/api') ? call.path.slice('/api'.length) : call.path).split('?')[0]!;
          calls.push({ method: call.method, path, body: call.body });
          if (call.method === 'POST' && path === '/downloads') return post(call.body);
          if (call.method === 'GET' && Object.hasOwn(routes, path)) {
            return { ok: true as const, status: 200, body: routes[path] };
          }
          return { ok: false as const, status: 404, code: 'E_NOT_FOUND', message: `no stub for ${path}` };
        },
        subscribe: downloadSubscribeStub(),
        watchLive: () => () => {},
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
    return calls;
  }

  function reinstallDialog(title = 'Reinstall model?'): HTMLElement | undefined {
    return [...document.querySelectorAll<HTMLElement>('[role="dialog"]')].find((el) => el.textContent?.includes(title));
  }

  function buttonNamed(scope: ParentNode, label: string): HTMLButtonElement {
    const button = [...scope.querySelectorAll<HTMLButtonElement>('button')].find(
      (b) => (b.textContent ?? '').trim() === label,
    );
    expect(button, label).toBeTruthy();
    return button!;
  }

  it('confirms, posts the numbers the user saw and nothing else, then follows the job', async () => {
    const calls = serveReinstall(catalogRoutes({ blockedByForeignDir: true, blocker: BLOCKER }), () => ({
      ok: true,
      status: 202,
      body: { id: 'job-reinstall', repo: REPO },
    }));
    await mountModels();
    await act(async () => buttonNamed(mounted!.container, 'Reinstall').click());
    const dialog = reinstallDialog()!;
    expect(dialog).toBeTruthy();
    const text = splitWhitespace(dialog.textContent ?? '').join(' ');
    expect(text).toContain(`permanently deletes the directory ${SLUG}`);
    expect(text).toContain(`It holds 1 file (183 MB): ${LEFTOVER}.`);
    expect(text).toContain('Nothing is deleted if the download fails or you cancel it.');
    expect(calls.filter((call) => call.method === 'POST')).toEqual([]);

    await act(async () => buttonNamed(dialog, 'Reinstall').click());
    await waitFor(() => openedStreams.includes('job-reinstall'));
    const posts = calls.filter((call) => call.method === 'POST');
    expect(posts.map((call) => call.body)).toEqual([
      { repo: REPO, replaceBlocker: { kind: 'directory', fileCount: 1, sizeBytes: 192_223_904, truncated: false } },
    ]);
    const card = mounted!.container.querySelector(`[data-model="${SLUG}"]`)!;
    await waitFor(() => (card.textContent ?? '').includes('Replaces the old directory when the download is complete.'));
    expect(buttonNamed(card, 'Cancel')).toBeTruthy();
    expect(card.textContent).not.toContain('Needs cleanup');
  });

  it('keeps a refused Reinstall in the confirmation, reloads the catalog, and raises no toast', async () => {
    const error = vi.spyOn(toast, 'error');
    const success = vi.spyOn(toast, 'success');
    onTestFinished(() => {
      error.mockRestore();
      success.mockRestore();
    });
    const calls = serveReinstall(catalogRoutes({ blockedByForeignDir: true, blocker: BLOCKER }), () => ({
      ok: false,
      status: 409,
      code: 'E_CONFLICT',
      message: `"${SLUG}" changed on disk. Check it, then try again.`,
    }));
    await mountModels();
    const catalogGets = calls.filter((call) => call.method === 'GET' && call.path === '/catalog').length;
    await act(async () => buttonNamed(mounted!.container, 'Reinstall').click());
    await act(async () => buttonNamed(reinstallDialog()!, 'Reinstall').click());
    await waitFor(() => reinstallDialog()?.querySelector('[role="alert"]') !== null);
    const alert = reinstallDialog()!.querySelector('[role="alert"]')!;
    expect(alert.textContent).toContain('Could not start the download');
    expect(alert.textContent).toContain('changed on disk. Check it, then try again.');
    await waitFor(() => calls.filter((call) => call.method === 'GET' && call.path === '/catalog').length > catalogGets);
    expect(error).not.toHaveBeenCalled();
    expect(success).not.toHaveBeenCalled();
    expect(openedStreams).toEqual([]);
  });

  it('turns Reinstall off while the last download is being checked', async () => {
    const props = {
      item: {
        hfRepo: REPO,
        slug: SLUG,
        installed: false,
        present: false,
        blockedByForeignDir: true,
        blocker: BLOCKER,
        localRevision: null,
        localAssetsRepo: null,
        localAssetsRevision: null,
      },
      updateAvailable: false,
      settling: true,
      updateSettling: false,
      job: undefined,
      onInstall: () => Promise.resolve(),
      onReinstall: () => Promise.resolve(),
      onReinstallStarted: () => {},
      onDone: () => {},
      onError: () => {},
      onCancelled: () => {},
      onCancel: () => {},
    };
    mounted = await renderPage(createElement(CatalogDownloadAction, props), (text) => text.includes('Reinstall'));
    const trigger = buttonNamed(mounted.container, 'Reinstall');
    expect(trigger.getAttribute('aria-disabled')).toBe('true');
    await act(async () => trigger.click());
    expect(reinstallDialog()).toBeUndefined();
  });

  it('offers the same Reinstall for a blocked companion, with its own title', async () => {
    const draft = {
      ...draftItem(false),
      blockedByForeignDir: true,
      blocker: { ...BLOCKER, fileCount: 0, sizeBytes: 0, names: [] },
    };
    serveReinstall(catalogRoutes({ present: true, installed: true, draft }), () => ({
      ok: true,
      status: 202,
      body: { id: 'job-draft', repo: draft.hfRepo },
    }));
    await mountModels();
    const companion = mounted!.container
      .querySelector(`a[href="https://huggingface.co/${draft.hfRepo}"]`)!
      .closest('[role="group"]')!;
    expect(companion.textContent).toContain('Needs cleanup');
    expect(companion.textContent).toContain(`${draft.slug} is in the way: an empty directory`);
    await act(async () => buttonNamed(companion, 'Reinstall').click());
    const dialog = reinstallDialog('Reinstall companion weights?')!;
    expect(dialog).toBeTruthy();
    expect(dialog.textContent).toContain('The directory is empty.');
  });

  it('still shows an installed model as installed', async () => {
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ present: true, installed: true }),
      LABEL,
    );
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Install');
  });

  it('offers Update available when upstream holds a different revision', async () => {
    // The whole point of the check: a re-upload lands new bytes at the SAME repo
    // id, so an installed model must still be offered the newer weights.
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40) }, [], {
        items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }],
      }),
      LABEL,
    );
    expect(buttonLabels()).toContain('Update available');
    expect(buttonLabels()).not.toContain('Installed');
  });

  it('stays Installed when the revisions match', async () => {
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40) }, [], {
        items: [{ hfRepo: REPO, remoteRevision: 'a'.repeat(40) }],
      }),
      LABEL,
    );
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Update available');
  });

  it('offers Update when only the tokenizer sidecars advanced', async () => {
    // A base-model tokenizer/template fix moves nothing in the primary repo, and
    // the Installed button is otherwise disabled — so without comparing the
    // sidecar source too, the repair path is unreachable by design.
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes(
        {
          present: true,
          installed: true,
          localRevision: 'a'.repeat(40),
          localAssetsRepo: 'base/model',
          localAssetsRevision: 'c'.repeat(40),
        },
        [],
        {
          items: [
            { hfRepo: REPO, remoteRevision: 'a'.repeat(40) },
            { hfRepo: 'base/model', remoteRevision: 'd'.repeat(40) },
          ],
        },
      ),
      LABEL,
    );
    expect(buttonLabels()).toContain('Update available');
  });

  it('stays Installed when the sidecar revision matches', async () => {
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes(
        {
          present: true,
          installed: true,
          localRevision: 'a'.repeat(40),
          localAssetsRepo: 'base/model',
          localAssetsRevision: 'c'.repeat(40),
        },
        [],
        {
          items: [
            { hfRepo: REPO, remoteRevision: 'a'.repeat(40) },
            { hfRepo: 'base/model', remoteRevision: 'c'.repeat(40) },
          ],
        },
      ),
      LABEL,
    );
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Update available');
  });

  it('stays Installed when the sidecar repo did not answer', async () => {
    // An unprobed sidecar repo (`null`) is "unknown", never "up to date" — and
    // it must not invent a badge either: no comparison, no update.
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes(
        {
          present: true,
          installed: true,
          localRevision: 'a'.repeat(40),
          localAssetsRepo: 'base/model',
          localAssetsRevision: 'c'.repeat(40),
        },
        [],
        {
          items: [
            { hfRepo: REPO, remoteRevision: 'a'.repeat(40) },
            { hfRepo: 'base/model', remoteRevision: null },
          ],
        },
      ),
      LABEL,
    );
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Update available');
  });

  it('offers Update when the catalog switched the sidecar source repo', async () => {
    // The marker names the OLD assets repo, which the sweep never probes —
    // it lists only the catalog's current one — so comparing the marker's
    // repo alone can never see the change and the migration job never runs.
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes(
        {
          present: true,
          installed: true,
          localRevision: 'a'.repeat(40),
          assetsRepo: 'base/new-model', // the entry's CURRENT source
          localAssetsRepo: 'base/model', // the marker's recorded one
          localAssetsRevision: 'c'.repeat(40),
        },
        [],
        {
          items: [
            { hfRepo: REPO, remoteRevision: 'a'.repeat(40) },
            { hfRepo: 'base/new-model', remoteRevision: 'e'.repeat(40) },
          ],
        },
      ),
      LABEL,
    );
    expect(buttonLabels()).toContain('Update available');
  });

  it('offers Update when the install predates sidecar provenance the entry prescribes', async () => {
    // localAssetsRepo null + a prescribed assetsRepo means the install lacks
    // the tokenizer sidecars entirely — a repairable gap, not "unknown".
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes(
        {
          present: true,
          installed: true,
          localRevision: 'a'.repeat(40),
          assetsRepo: 'base/model',
          localAssetsRepo: null,
          localAssetsRevision: null,
        },
        [],
        {
          items: [
            { hfRepo: REPO, remoteRevision: 'a'.repeat(40) },
            { hfRepo: 'base/model', remoteRevision: 'e'.repeat(40) },
          ],
        },
      ),
      LABEL,
    );
    expect(buttonLabels()).toContain('Update available');
  });

  it('stays Installed when the catalog dropped its sidecar source', async () => {
    // The reverse direction is deliberately quiet: the marker records an
    // assetsRepo the entry no longer prescribes, no job could clear that
    // badge, and the old repo is unprobed — Installed, no update.
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes(
        {
          present: true,
          installed: true,
          localRevision: 'a'.repeat(40),
          localAssetsRepo: 'base/model',
          localAssetsRevision: 'c'.repeat(40),
        },
        [],
        { items: [{ hfRepo: REPO, remoteRevision: 'a'.repeat(40) }] },
      ),
      LABEL,
    );
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Update available');
  });

  it('never offers Update for a checkpoint the dashboard does not own', async () => {
    // The join is the page's now, so the ownership gate is too: an Update is a
    // re-install, and the runner's preflight refuses every final dir it does not
    // own. `present` without `installed` is exactly that dir (an `mlx download`
    // install, or a dashboard one the user renamed), so the button could only
    // ever raise a red toast — however stale the bytes are.
    //
    // `/catalog` pairs a revision with ownership today, so this pins the gate on
    // the WIRE shape rather than on that invariant: the two halves of the
    // comparison arrive from two different requests, and only `installed` says
    // the stale one can be replaced.
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ present: true, installed: false, localRevision: 'a'.repeat(40) }, [], {
        items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }],
      }),
      LABEL,
    );
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Update available');
  });

  it('stays Installed when the update check could not reach Hugging Face', async () => {
    // Offline must degrade to exactly the old behaviour — never a badge the user
    // cannot act on, and never a blocked page. `/catalog` never touched the
    // network, so install state is unaffected either way.
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40) }, [], {
        items: [{ hfRepo: REPO, remoteRevision: null }],
      }),
      LABEL,
    );
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Update available');
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

  it('shows a live update job even when the update probe has not resolved', async () => {
    // Regression: `downloading` was qualified by catalog state, so a remount
    // mid-update — with `/catalog/updates` still loading or unable to resolve
    // this repo — rendered "Installed" over a live multi-gigabyte transfer,
    // with no progress and no Cancel. A live job must speak for itself.
    await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40) }, [downloadJob({})], {
        items: [],
      }),
      LABEL,
    );
    expect(buttonLabels()).toContain('Cancel');
    expect(buttonLabels()).not.toContain('Installed');
  });

  it('drops a stale update verdict when the post-download refresh FAILS', async () => {
    // `useJson` keeps its previous `data` when a reload fails and only sets
    // `error`. Consuming that blindly re-offers "Update available" for the
    // revision just installed, and admits repeated no-op jobs, until some later
    // probe happens to succeed.
    const before = catalogRoutes(
      { present: true, installed: true, localRevision: 'a'.repeat(40) },
      [downloadJob({ id: 'job-upd', state: 'done' })],
      { items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }] },
    );
    recordRequests({
      ...before,
      // First read succeeds; the post-download reload fails.
      '/catalog/updates': sequence(before['/catalog/updates'], STUB_FAILURE),
      '/downloads/job-upd': { cancelled: true, id: 'job-upd' },
    });
    await mountModels();
    expect(buttonLabels()).not.toContain('Update available');
    expect(buttonLabels()).toContain('Installed');
  });

  it('reloads the update check when a job settles as ERROR too', async () => {
    // `error` does not mean nothing was installed: `publish()` renames staging
    // into place and only THEN fsyncs and removes the backup, inside the job's
    // try. A failure there reports `error` for a model already on disk, so the
    // previous verdict must not survive it. BOTH halves of that verdict have to
    // move: the freshly installed `localRevision` lives on `/catalog`, and the
    // sha it is compared against on `/catalog/updates`.
    const failed = vi.spyOn(toast, 'error');
    try {
      const stale = catalogRoutes(
        { present: true, installed: true, localRevision: 'a'.repeat(40) },
        [downloadJob({ id: 'job-run', state: 'running' })],
        { items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }] },
      );
      const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'b'.repeat(40) }, [], {
        items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }],
      });
      const calls = recordRequests({
        ...stale,
        '/catalog': sequence(stale['/catalog'], fresh['/catalog']),
        '/catalog/updates': sequence(stale['/catalog/updates'], fresh['/catalog/updates']),
        '/downloads/job-run': { cancelled: true, id: 'job-run' },
      });
      await mountModels();
      expect(buttonLabels()).toContain('Cancel');

      emitDownload('error', { id: 'job-run', message: 'fsync failed after publish' });
      await settle();
      expect(gets(calls, '/api/catalog/updates')).toBe(2);
      expect(gets(calls, '/api/catalog')).toBe(2);
      expect(buttonLabels()).not.toContain('Update available');
      expect(buttonLabels()).toContain('Installed');
    } finally {
      failed.mockRestore();
    }
  });

  it('shows a failed catalog reload after a settle with Try again, which reopens Install', async () => {
    const failed = vi.spyOn(toast, 'error');
    try {
      const routes = catalogRoutes({}, [downloadJob({ id: 'job-run' })]);
      const catalog = routes['/catalog'];
      const calls = recordRequests({ ...routes, '/catalog': sequence(catalog, STUB_FAILURE, catalog) });
      await mountModels();

      emitDownload('error', { id: 'job-run', message: 'disk full' });
      await settle();
      expect(gets(calls, '/api/catalog')).toBe(2);
      expect(mounted!.text()).toContain('stubbed failure for /catalog');
      expect(mounted!.text()).toContain(LABEL);
      expect(liveButtonLabels()).not.toContain('Install');
      const retry = [...mounted!.container.querySelectorAll('button')].find(
        (button) => button.textContent?.trim() === 'Try again',
      );
      expect(retry).toBeDefined();

      await act(async () => retry!.click());
      await settle();
      expect(gets(calls, '/api/catalog')).toBe(3);
      expect(mounted!.text()).not.toContain('stubbed failure for /catalog');
      expect(liveButtonLabels()).toContain('Install');
    } finally {
      failed.mockRestore();
    }
  });

  it('cannot start a second job in the render between a settle and the refreshed catalog', async () => {
    // `setActive` clears synchronously while `reload()` only marks the previous
    // body refreshing, so one committed render still carries the pre-download
    // state: no job, the old `localRevision`, and the old remote sha. Left live,
    // that button allocates a job the server can only short-circuit to `done` —
    // two Hugging Face round trips, a spurious card and a spurious toast.
    const stale = catalogRoutes(
      { present: true, installed: true, localRevision: 'a'.repeat(40) },
      [downloadJob({ id: 'job-upd', state: 'running' })],
      { items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }] },
    );
    // Upstream moved again while that job ran, so once the refreshed bodies land
    // there is a GENUINE update to offer. That is what proves the window closes:
    // a suppression that never lifts would leave this button dead forever.
    const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'b'.repeat(40) }, [], {
      items: [{ hfRepo: REPO, remoteRevision: 'c'.repeat(40) }],
    });
    // Held, so the assertions below land INSIDE the window rather than after it.
    const held = deferred(fresh['/catalog']);
    recordRequests({
      ...stale,
      '/catalog': sequence(stale['/catalog'], held.body),
      '/catalog/updates': sequence(stale['/catalog/updates'], fresh['/catalog/updates']),
      '/downloads/job-upd': { cancelled: true, id: 'job-upd' },
    });
    await mountModels();

    emitDownload('done', { id: 'job-upd', outputDir: '/models/x' });
    await settle();
    // The refreshed catalog has not landed, so the card still reads the stale
    // comparison — but it must not be pressable while it does.
    expect(buttonLabels()).toContain('Update available');
    expect(liveButtonLabels()).not.toContain('Update available');
    expect(paletteLabels).not.toContain(`Update ${LABEL}`);

    held.release();
    await settle();
    expect(liveButtonLabels()).toContain('Update available');
    expect(paletteLabels).toContain(`Update ${LABEL}`);
  });

  it('closes that window on the updates half too, when the catalog lands first', async () => {
    // The verdict is a join of two independent responses, and the guard has to
    // hold until BOTH have moved. `/catalog` is a worker route reading only the
    // filesystem, while `/catalog/updates` may dial Hugging Face on a cold sha
    // cache — so the catalog routinely wins, and on its own it lifts the guard
    // over a remote sha resolved BEFORE the job ran. The card then offers
    // "Update available" for the revision the job just installed, which is the
    // pre-download verdict this mechanism exists to withhold.
    const stale = catalogRoutes(
      { present: false, installed: false, localRevision: null },
      [downloadJob({ id: 'job-join', state: 'running' })],
      { items: [{ hfRepo: REPO, remoteRevision: 'a'.repeat(40) }] },
    );
    // Upstream moved past what the job installed, so once BOTH bodies land there
    // is a genuine update to offer — a suppression that never lifts fails here.
    const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'b'.repeat(40) }, [], {
      items: [{ hfRepo: REPO, remoteRevision: 'c'.repeat(40) }],
    });
    // Only the remote shas are held: the catalog arrives first, which is the
    // whole point of the ordering under test.
    const held = deferred(fresh['/catalog/updates']);
    recordRequests({
      ...stale,
      '/models': sequence(stale['/models'], fresh['/models']),
      '/catalog': sequence(stale['/catalog'], fresh['/catalog']),
      '/catalog/updates': sequence(stale['/catalog/updates'], held.body),
      '/downloads/job-join': { cancelled: true, id: 'job-join' },
    });
    await mountModels();

    emitDownload('done', { id: 'job-join', outputDir: '/models/x' });
    await settle();
    // The fresh catalog HAS landed — `localRevision` is the installed one — but
    // it is being compared against the sha from before the job.
    expect(buttonLabels()).toContain('Update available');
    expect(liveButtonLabels()).not.toContain('Update available');

    held.release();
    await settle();
    expect(liveButtonLabels()).toContain('Update available');
  });

  it('closes that window on a FIRST install too, where the stale body still says absent', async () => {
    // The same one render, reached from the other side: nothing was installed
    // before, so the stale body says `present: false` and the branch falls
    // through to a live Install for the model that just finished installing.
    const stale = catalogRoutes({ present: false, installed: false, localRevision: null }, [
      downloadJob({ id: 'job-new', state: 'running' }),
    ]);
    const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'b'.repeat(40) });
    const held = deferred(fresh['/catalog']);
    recordRequests({
      ...stale,
      '/catalog': sequence(stale['/catalog'], held.body),
      '/downloads/job-new': { cancelled: true, id: 'job-new' },
    });
    await mountModels();

    emitDownload('done', { id: 'job-new', outputDir: '/models/x' });
    await settle();
    expect(buttonLabels()).toContain('Install');
    expect(liveButtonLabels()).not.toContain('Install');
    expect(paletteLabels).not.toContain(`Install ${LABEL}`);

    held.release();
    await settle();
    expect(buttonLabels()).toContain('Installed');
  });

  it('closes that window for a job that settled while the page was away', async () => {
    // Reached without any live event. A job that went terminal while this page
    // was unmounted (or across a reconnect) is skipped by the hydration above,
    // so `active` never holds it and no `DownloadProgress` ever fires — the
    // reconcile effect is the only thing that notices, and it reloads the same
    // snapshots. Its window is identical, and the guard has to be set there too.
    const stale = catalogRoutes({ present: false, installed: false, localRevision: null }, [
      downloadJob({ id: 'job-away', state: 'done' }),
    ]);
    // Upstream is ahead of what that job installed, so once the body lands there
    // is a real update to offer — a suppression that never lifts kills it.
    const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'b'.repeat(40) }, [], {
      items: [{ hfRepo: REPO, remoteRevision: 'c'.repeat(40) }],
    });
    const held = deferred(fresh['/catalog']);
    recordRequests({
      ...stale,
      '/catalog': sequence(stale['/catalog'], held.body),
      '/catalog/updates': sequence(stale['/catalog/updates'], fresh['/catalog/updates']),
      '/downloads/job-away': { cancelled: true, id: 'job-away' },
    });
    await mountModels();

    // The mount itself spans the publish: `/downloads` already says `done` while
    // `/catalog` still says absent.
    expect(buttonLabels()).toContain('Install');
    expect(liveButtonLabels()).not.toContain('Install');

    held.release();
    await settle();
    expect(liveButtonLabels()).toContain('Update available');
  });

  it('closes the updates half for a job that settled while the page was away too', async () => {
    // The reconcile effect records the pair from refs, not from the render, so
    // its updates half needs its own gate. Same ordering as the live case: the
    // filesystem catalog lands while the remote shas are still in flight.
    const stale = catalogRoutes(
      { present: false, installed: false, localRevision: null },
      [downloadJob({ id: 'job-away2', state: 'done' })],
      { items: [{ hfRepo: REPO, remoteRevision: 'a'.repeat(40) }] },
    );
    const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'b'.repeat(40) }, [], {
      items: [{ hfRepo: REPO, remoteRevision: 'c'.repeat(40) }],
    });
    const held = deferred(fresh['/catalog/updates']);
    recordRequests({
      ...stale,
      '/models': sequence(stale['/models'], fresh['/models']),
      '/catalog': sequence(stale['/catalog'], fresh['/catalog']),
      '/catalog/updates': sequence(stale['/catalog/updates'], held.body),
      '/downloads/job-away2': { cancelled: true, id: 'job-away2' },
    });
    await mountModels();

    // The installed revision is on screen, compared against the sha from before
    // the job. That verdict must not be actionable.
    expect(buttonLabels()).toContain('Update available');
    expect(liveButtonLabels()).not.toContain('Update available');

    held.release();
    await settle();
    expect(liveButtonLabels()).toContain('Update available');
  });

  it('closes that window when the job settles as ERROR, which can still have installed', async () => {
    // `publish()` renames staging into place and only THEN removes the backup,
    // inside the job's try, so an `error` can name a model already on disk. That
    // path clears `active` synchronously too, so it opens the same window.
    const failed = vi.spyOn(toast, 'error');
    try {
      const stale = catalogRoutes(
        { present: true, installed: true, localRevision: 'a'.repeat(40) },
        [downloadJob({ id: 'job-err', state: 'running' })],
        { items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }] },
      );
      const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'b'.repeat(40) }, [], {
        items: [{ hfRepo: REPO, remoteRevision: 'c'.repeat(40) }],
      });
      const held = deferred(fresh['/catalog']);
      recordRequests({
        ...stale,
        '/catalog': sequence(stale['/catalog'], held.body),
        '/catalog/updates': sequence(stale['/catalog/updates'], fresh['/catalog/updates']),
        '/downloads/job-err': { cancelled: true, id: 'job-err' },
      });
      await mountModels();

      emitDownload('error', { id: 'job-err', message: 'fsync failed after publish' });
      await settle();
      expect(buttonLabels()).toContain('Update available');
      expect(liveButtonLabels()).not.toContain('Update available');

      held.release();
      await settle();
      expect(liveButtonLabels()).toContain('Update available');
    } finally {
      failed.mockRestore();
    }
  });

  it('keeps following a job this page cancelled, so its terminal frame still lands', async () => {
    // The server accepts a cancel while the in-flight job is still `running` and
    // lets `processJob` emit the terminal frame as it unwinds, so the DELETE is
    // acknowledged FIRST. Clearing `active` there unmounted `DownloadProgress`
    // and closed its subscription before that frame arrived, so
    // `onDownloadCancelled` never ran and nothing refreshed the snapshots — the
    // card offered a live Install for whatever `recoverBackup` had restored,
    // until the page was remounted.
    //
    // Refreshing at the acknowledgement instead does not fix it: recovery renames
    // the crashed publish onto the final dir at `download.ts:868` and every
    // cancel boundary comes after it, so a reload issued there can read the disk
    // before the rename. Only the terminal frame proves the job is done with it.
    const stale = catalogRoutes({ present: false, installed: false, localRevision: null }, [
      downloadJob({ id: 'job-run', state: 'running' }),
    ]);
    const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40) }, [], {
      items: [{ hfRepo: REPO, remoteRevision: 'a'.repeat(40) }],
    });
    const calls = recordRequests({
      ...stale,
      '/models': sequence(stale['/models'], fresh['/models']),
      '/catalog': sequence(stale['/catalog'], fresh['/catalog']),
      '/catalog/updates': sequence(stale['/catalog/updates'], fresh['/catalog/updates']),
      '/downloads/job-run': { cancelled: true, id: 'job-run' },
    });
    await mountModels();
    const cancelButton = [...mounted!.container.querySelectorAll('button')].find((button) =>
      button.textContent?.includes('Cancel'),
    );
    expect(cancelButton).toBeDefined();
    await act(async () => {
      cancelButton!.click();
    });
    await settle();
    // Acknowledged: the Cancel is gone, because a second one answers 404 — but
    // the card is still following the job, so nothing offers Install yet.
    expect(dismissed(calls)).toEqual(['job-run']);
    expect(buttonLabels()).not.toContain('Cancel');
    expect(buttonLabels()).not.toContain('Install');
    expect(gets(calls, '/api/catalog')).toBe(1);

    // Only now does the unwinding job settle.
    emitDownload('cancelled', { id: 'job-run' });
    await settle();
    expect(gets(calls, '/api/catalog')).toBe(2);
    expect(gets(calls, '/api/models')).toBe(2);
    expect(gets(calls, '/api/catalog/updates')).toBe(2);
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Install');
  });

  it('does not resurrect a cancelled job whose terminal frame beat the acknowledgement', async () => {
    // The other order, and the server produces it: `cancel()` settles a QUEUED
    // job inside itself, emitting the terminal frame BEFORE the route replies.
    // `onDownloadCancelled` therefore clears the entry first, and the mark that
    // follows must not put a half-built job back on the card.
    const stale = catalogRoutes({ present: false, installed: false, localRevision: null }, [
      downloadJob({ id: 'job-queued', state: 'running' }),
    ]);
    const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40) }, [], {
      items: [{ hfRepo: REPO, remoteRevision: 'a'.repeat(40) }],
    });
    const heldDelete = deferred({ cancelled: true, id: 'job-queued' });
    recordRequests({
      ...stale,
      '/models': sequence(stale['/models'], fresh['/models']),
      '/catalog': sequence(stale['/catalog'], fresh['/catalog']),
      '/catalog/updates': sequence(stale['/catalog/updates'], fresh['/catalog/updates']),
      '/downloads/job-queued': heldDelete.body,
    });
    await mountModels();
    const cancelButton = [...mounted!.container.querySelectorAll('button')].find((button) =>
      button.textContent?.includes('Cancel'),
    );
    await act(async () => {
      cancelButton!.click();
    });
    // The frame lands while the DELETE is still in flight.
    emitDownload('cancelled', { id: 'job-queued' });
    await settle();
    expect(buttonLabels()).toContain('Installed');

    heldDelete.release();
    await settle();
    // Still settled: no progress card, no Cancel, no orphaned job.
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Cancel');
  });

  it('never marks a job the cancel did not name, when a new install beat the acknowledgement', async () => {
    // The third order this page can produce, and the one the id guard is for.
    // The terminal frame clears the entry, the user installs again, and only THEN
    // does the DELETE for the FIRST job resolve. Marking whatever sits under the
    // repo would put `cancelling` on the SECOND job — dropping the Cancel button
    // for a job nobody cancelled, with no terminal frame coming to undo it.
    //
    // Hand-rolled rather than `recordRequests`, because the stub resolves a route
    // by path alone: the POST that mints the second id and the GET that lists
    // jobs share the `/downloads` key.
    let releaseDelete!: () => void;
    const heldDelete = new Promise<void>((resolve) => {
      releaseDelete = resolve;
    });
    const routes = catalogRoutes({}, [downloadJob({ id: 'job-a', state: 'running' })]);
    const subscribe = downloadSubscribeStub();
    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    const dispose = serveRuntimeOverPort(
      {
        call: async (call: ApiCall) => {
          const path = call.path.startsWith('/api') ? call.path.slice('/api'.length) : call.path;
          if (call.method === 'POST' && path === '/downloads') {
            return { ok: true as const, status: 202, body: { id: 'job-b', repo: REPO } };
          }
          if (call.method === 'DELETE' && path === '/downloads/job-a') {
            await heldDelete;
            return { ok: true as const, status: 200, body: { cancelled: true, id: 'job-a' } };
          }
          if (call.method === 'GET' && Object.hasOwn(routes, path)) {
            return { ok: true as const, status: 200, body: routes[path] };
          }
          return { ok: false as const, status: 404, code: 'E_NOT_FOUND', message: `no stub for ${path}` };
        },
        subscribe,
        watchLive: () => () => {},
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

    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Models)), (text) =>
      text.includes(LABEL),
    );
    await settle();
    const cancelButton = [...mounted.container.querySelectorAll('button')].find((button) =>
      button.textContent?.includes('Cancel'),
    );
    expect(cancelButton).toBeDefined();
    await act(async () => {
      cancelButton!.click();
    });
    await settle();

    // The first job settles while its own DELETE is still held.
    emitDownload('cancelled', { id: 'job-a' });
    await settle();
    const install = [...mounted.container.querySelectorAll('button')].find((button) =>
      button.textContent?.includes('Install'),
    );
    expect(install).toBeDefined();
    // The refreshed bodies have landed, so the settle guard has already lifted.
    expect(install!.disabled).toBe(false);
    await act(async () => {
      install!.click();
    });
    await settle();
    expect(openedStreams.at(-1)).toBe('job-b');
    expect(buttonLabels()).toContain('Cancel');

    releaseDelete();
    await settle();
    // `job-b` was never cancelled, so it keeps its Cancel button.
    expect(buttonLabels()).toContain('Cancel');
  });

  it('reloads the snapshots when a job settles as CANCELLED, which can still have installed', async () => {
    // `cancelled` does not mean nothing reached disk. `processJob` runs
    // `recoverBackup` BEFORE its first cancel boundary: a crashed publish left an
    // owned backup under `.staging/<slug>.backup-<dead pid>-…`, and recovery
    // renames it onto the final dir. Every cancel check comes after that, so a
    // job cancelled anywhere in the fetch loop settles `cancelled` with a model
    // installed the page's catalog body has never seen.
    //
    // Nothing else repairs it. No query polls, `downloads.reload` has no call
    // site, and the reconcile effect runs only on mount and reconnect — so
    // without a reload here the card offers Install for an installed model until
    // the page is remounted.
    const failed = vi.spyOn(toast, 'error');
    try {
      const stale = catalogRoutes({ present: false, installed: false, localRevision: null }, [
        downloadJob({ id: 'job-run', state: 'running' }),
      ]);
      const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40) }, [], {
        items: [{ hfRepo: REPO, remoteRevision: 'a'.repeat(40) }],
      });
      const calls = recordRequests({
        ...stale,
        '/models': sequence(stale['/models'], fresh['/models']),
        '/catalog': sequence(stale['/catalog'], fresh['/catalog']),
        '/catalog/updates': sequence(stale['/catalog/updates'], fresh['/catalog/updates']),
        '/downloads/job-run': { cancelled: true, id: 'job-run' },
      });
      await mountModels();
      expect(buttonLabels()).toContain('Cancel');

      emitDownload('cancelled', { id: 'job-run' });
      await settle();
      expect(gets(calls, '/api/catalog')).toBe(2);
      expect(gets(calls, '/api/models')).toBe(2);
      // The job resolved a fresh upstream sha at `processJob:877` and folded it
      // into the server's cached map before it was cancelled, so this half is
      // stale too — and it costs no new network call to move it.
      expect(gets(calls, '/api/catalog/updates')).toBe(2);
      expect(buttonLabels()).toContain('Installed');
      expect(buttonLabels()).not.toContain('Install');
      // Still silent: a cancel is not a failure, and whoever issued it already
      // saw their own confirmation.
      expect(failed.mock.calls).toEqual([]);
    } finally {
      failed.mockRestore();
    }
  });

  it('closes that window when the job settles as CANCELLED too', async () => {
    // The reload above opens the very window this suite exists for: `setActive`
    // clears synchronously while the refreshed catalog is still in flight, so
    // one committed render carries a live Install for the model `recoverBackup`
    // just restored.
    const failed = vi.spyOn(toast, 'error');
    try {
      const stale = catalogRoutes({ present: false, installed: false, localRevision: null }, [
        downloadJob({ id: 'job-run', state: 'running' }),
      ]);
      // Upstream is ahead of the recovered bytes, so once the body lands there is
      // a genuine update to offer — a suppression that never lifts kills it.
      const fresh = catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40) }, [], {
        items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }],
      });
      const held = deferred(fresh['/catalog']);
      recordRequests({
        ...stale,
        '/catalog': sequence(stale['/catalog'], held.body),
        '/catalog/updates': sequence(stale['/catalog/updates'], fresh['/catalog/updates']),
        '/downloads/job-run': { cancelled: true, id: 'job-run' },
      });
      await mountModels();

      emitDownload('cancelled', { id: 'job-run' });
      await settle();
      expect(buttonLabels()).toContain('Install');
      expect(liveButtonLabels()).not.toContain('Install');

      held.release();
      await settle();
      expect(liveButtonLabels()).toContain('Update available');
    } finally {
      failed.mockRestore();
    }
  });

  it('reloads the update check when a job settles, so it stops re-offering Update', async () => {
    // Regression: `/catalog/updates` is a SEPARATE request, so it held the
    // pre-download verdict. Once the job cleared, the card re-offered "Update
    // available" for the revision just installed, and every click started
    // another job. The reconcile effect must refresh it alongside the catalog.
    const stale = catalogRoutes(
      { present: true, installed: true, localRevision: 'a'.repeat(40) },
      [downloadJob({ id: 'job-upd', state: 'done' })],
      { items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }] },
    );
    const fresh = catalogRoutes(
      { present: true, installed: true, localRevision: 'b'.repeat(40) },
      [downloadJob({ id: 'job-upd', state: 'done' })],
      { items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }] },
    );
    const calls = recordRequests({
      ...stale,
      '/catalog': sequence(stale['/catalog'], fresh['/catalog']),
      '/catalog/updates': sequence(stale['/catalog/updates'], fresh['/catalog/updates']),
      '/models': sequence(stale['/models'], fresh['/models']),
      '/downloads/job-upd': { cancelled: true, id: 'job-upd' },
    });
    await mountModels();
    // Settled on the newly installed revision: Installed, not a fresh Update.
    expect(buttonLabels()).toContain('Installed');
    expect(buttonLabels()).not.toContain('Update available');
    // Refetched exactly once, in step with the catalog it is compared against.
    expect(gets(calls, '/api/catalog/updates')).toBe(2);
    expect(gets(calls, '/api/catalog')).toBe(2);
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
          const path = call.path.startsWith('/api') ? call.path.slice('/api'.length) : call.path;
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
        watchLive: () => () => {},
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

    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Models)), (text) =>
      text.includes(LABEL),
    );
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

describe('Models page — local models and the drawer', () => {
  const NAME = 'qwen3.8-27b-gguf';
  const SERVED = 'Qwen3.8-27B-UD-Q4_K_XL';

  function modelsRoutes(extra: Record<string, unknown> = {}): Record<string, unknown> {
    const models: ModelsResponse = {
      models: [
        {
          name: NAME,
          path: `/models/${NAME}`,
          modelType: 'qwen3_5',
          quant: 'Q4_K_XL',
          contextWindow: 262_144,
          sizeBytes: 17 * 1024 ** 3,
          fileCount: 2,
          servedIds: [SERVED],
        },
      ],
      warnings: [],
      dir: '/models',
      host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
    };
    return {
      '/models': models,
      '/catalog': { items: [] },
      '/catalog/updates': { items: [] },
      '/downloads': { jobs: [] },
      '/metrics/summary': summary({ byModel: [byModelRow(SERVED, 7)] }),
      '/metrics/timeseries': timeseries([]),
      '/metrics/turns': turnsPage([]),
      ...extra,
    };
  }

  let restoreWidth: (() => void) | undefined;
  let api: ReturnType<typeof stubApi> | undefined;
  let calls: ApiCall[] = [];
  afterEach(() => {
    restoreWidth?.();
    restoreWidth = undefined;
    api = undefined;
    vi.restoreAllMocks();
  });

  async function mountModels(routes: Record<string, unknown>, url = '/models'): Promise<void> {
    restoreWidth = observeWidth(1180);
    calls = [];
    api = stubApi(routes, { onCall: (call) => calls.push(call) });
    restoreApi = api;
    mounted = await renderPage(
      createElement(MemoryRouter, { initialEntries: [url] }, createElement(Models), createElement(CommandsProbe)),
      (text) => text.includes(NAME) && text.includes('Recommended models'),
    );
  }

  let pageCommands: readonly AppCommand[] = [];
  function CommandsProbe(): null {
    pageCommands = usePageCommands();
    return null;
  }

  function modelRow(): HTMLElement {
    const tr = mounted!.container.querySelector<HTMLElement>(`tr[data-model="${NAME}"]`);
    expect(tr).toBeTruthy();
    return tr!;
  }

  function popover(): HTMLElement {
    const found = [...document.querySelectorAll<HTMLElement>('[role="dialog"]')].find((el) =>
      el.textContent?.includes('Delete model?'),
    );
    expect(found).toBeTruthy();
    return found!;
  }

  function buttonIn(scope: ParentNode, text: string): HTMLButtonElement {
    const button = [...scope.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === text);
    expect(button).toBeTruthy();
    return button!;
  }

  it('joins a GGUF checkpoint to its usage, live state and drawer through its served id', async () => {
    await mountModels(modelsRoutes());
    await waitFor(() => modelRow().textContent?.includes('7') === true);
    await waitFor(() => {
      act(() =>
        api?.emitLive({ type: 'runtime', snapshot: live({ processes: [process({ models: [liveModel(SERVED)] })] }) }),
      );
      return modelRow().textContent?.includes('Loaded') === true;
    }, 2000);
    await act(async () => modelRow().querySelector<HTMLAnchorElement>(`a[href="/models?focus=${NAME}"]`)!.click());
    await waitFor(() => calls.some((call) => call.path.includes(`model=${SERVED}`)));
    expect(
      calls.some((call) => call.path.startsWith('/api/metrics/summary') && call.path.includes(`model=${SERVED}`)),
    ).toBe(true);
  });

  it('a failed daily series keeps the turn count and says why the spark is missing', async () => {
    await mountModels(modelsRoutes({ '/metrics/timeseries': STUB_FAILURE }));
    await waitFor(() => modelRow().querySelector('[title="Daily turns could not be loaded"]') !== null);
    expect(modelRow().querySelector('[title="Daily turns could not be loaded"]')?.textContent).toBe('7');
    expect(modelRow().querySelector('[title="No recorded turns for this model in the last 30 days"]')).toBeNull();
  });

  it('the row delete opens a confirm at the button and keeps a failure inside it', async () => {
    await mountModels(modelsRoutes({ [`/models/${NAME}`]: STUB_FAILURE }));
    await act(async () => modelRow().querySelector<HTMLButtonElement>(`button[aria-label="Delete ${NAME}"]`)!.click());
    expect(popover().textContent).toContain(NAME);
    expect(calls.filter((call) => call.method === 'DELETE')).toEqual([]);
    await act(async () => buttonIn(popover(), 'Delete').click());
    await waitFor(() => popover().querySelector('[role="alert"]') !== null);
    expect(popover().querySelector('[role="alert"]')?.textContent).toContain('Failed to delete model');
  });

  it('a row delete that succeeds calls the toast and reloads the lists', async () => {
    const success = vi.spyOn(toast, 'success');
    const routes = modelsRoutes({ [`/models/${NAME}`]: { deleted: true, name: NAME } });
    routes['/models'] = sequence(routes['/models'], { ...(routes['/models'] as ModelsResponse), models: [] });
    await mountModels(routes);
    await act(async () => modelRow().querySelector<HTMLButtonElement>(`button[aria-label="Delete ${NAME}"]`)!.click());
    await act(async () => buttonIn(popover(), 'Delete').click());
    await waitFor(() => success.mock.calls.length > 0);
    expect(success).toHaveBeenCalledWith('Model deleted', { description: NAME });
    await waitFor(() => mounted!.container.querySelector(`tr[data-model="${NAME}"]`) === null);
    expect(calls.filter((call) => call.method === 'GET' && call.path.split('?')[0] === '/api/catalog').length).toBe(2);
  });

  /** `/models` before and after NAME was deleted. */
  function listings(): { before: ModelsResponse; after: ModelsResponse } {
    const before = modelsRoutes()['/models'] as ModelsResponse;
    return { before, after: { ...before, models: [] } };
  }

  async function deleteFromRow(): Promise<void> {
    await act(async () => modelRow().querySelector<HTMLButtonElement>(`button[aria-label="Delete ${NAME}"]`)!.click());
    await act(async () => buttonIn(popover(), 'Delete').click());
  }

  function rowGone(): boolean {
    return mounted!.container.querySelector(`tr[data-model="${NAME}"]`) === null;
  }

  it('a row delete takes the row, its tile count and its bytes off at once, while the listing is still loading', async () => {
    const success = vi.spyOn(toast, 'success');
    const { before, after } = listings();
    const held = deferred(after);
    await mountModels(
      modelsRoutes({ '/models': sequence(before, held.body), [`/models/${NAME}`]: { deleted: true, name: NAME } }),
    );
    expect(statValue(mounted!.container, 'Installed')).toBe('1');
    expect(mounted!.container.querySelector('[aria-label^="Disk use by model:"]')).not.toBeNull();
    await deleteFromRow();
    await waitFor(() => success.mock.calls.length > 0);
    await waitFor(rowGone);
    expect(calls.filter((call) => call.method === 'GET' && call.path === '/api/models').length).toBe(2);
    expect(statValue(mounted!.container, 'Installed')).toBe('0');
    expect(statValue(mounted!.container, 'On disk')).not.toContain('17');
    expect(document.querySelector(`button[aria-label="Delete ${NAME}"]`)).toBeNull();
    expect(mounted!.container.querySelector('[aria-label^="Disk use by model:"]')).toBeNull();
    held.release();
    for (let i = 0; i < 10; i++) {
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
    }
    expect(rowGone()).toBe(true);
  });

  it('a drawer delete takes the row off while the listing is still loading', async () => {
    const { before, after } = listings();
    const held = deferred(after);
    await mountModels(
      modelsRoutes({ '/models': sequence(before, held.body), [`/models/${NAME}`]: { deleted: true, name: NAME } }),
      `/models?focus=${NAME}`,
    );
    await waitFor(() => byRole(document, 'dialog', NAME) !== null);
    await act(async () => buttonIn(byRole(document, 'dialog', NAME)!, 'Delete model').click());
    await act(async () => buttonIn(popover(), 'Delete').click());
    await waitFor(() => byRole(document, 'dialog', NAME) === null, 2000);
    await waitFor(rowGone);
    expect(statValue(mounted!.container, 'Installed')).toBe('0');
    held.release();
  });

  it('a delete whose reload fails keeps the row off and says the list could not be refreshed', async () => {
    const { before, after } = listings();
    await mountModels(
      modelsRoutes({
        '/models': sequence(before, STUB_FAILURE, after),
        [`/models/${NAME}`]: { deleted: true, name: NAME },
      }),
    );
    await deleteFromRow();
    await waitFor(() => (mounted!.container.textContent ?? '').includes('Could not refresh the model list'));
    expect(rowGone()).toBe(true);
    await act(async () => buttonIn(mounted!.container, 'Try again').click());
    await waitFor(() => !(mounted!.container.textContent ?? '').includes('Could not refresh the model list'));
    expect(rowGone()).toBe(true);
  });

  it('a delete that answers not found takes the row off and fetches the list again', async () => {
    const success = vi.spyOn(toast, 'success');
    const { before, after } = listings();
    await mountModels(modelsRoutes({ '/models': sequence(before, after) }));
    await deleteFromRow();
    await waitFor(() => success.mock.calls.length > 0);
    expect(success).toHaveBeenCalledWith('Model was already gone', { description: NAME });
    await waitFor(rowGone);
    expect(calls.filter((call) => call.method === 'DELETE').map((call) => call.path)).toEqual([`/api/models/${NAME}`]);
    await waitFor(() => calls.filter((call) => call.method === 'GET' && call.path === '/api/models').length === 2);
  });

  it('shows the row again when the runtime still lists it after the delete', async () => {
    const { before } = listings();
    await mountModels(
      modelsRoutes({ '/models': sequence(before, before), [`/models/${NAME}`]: { deleted: true, name: NAME } }),
    );
    await deleteFromRow();
    await waitFor(() => calls.filter((call) => call.method === 'GET' && call.path === '/api/models').length === 2);
    await waitFor(() => !rowGone());
  });

  it('a listing read before the delete never seeds the next visit', async () => {
    const { before, after } = listings();
    const stale = deferred(before);
    const neverAnswers = deferred(after);
    const routes = modelsRoutes({
      '/models': sequence(before, stale.body, after, neverAnswers.body),
      [`/models/${NAME}`]: { deleted: true, name: NAME },
    });
    await mountModels(routes);
    act(() => reloadMountedQueries());
    await waitFor(() => calls.filter((call) => call.method === 'GET' && call.path === '/api/models').length === 2);
    await deleteFromRow();
    await waitFor(() => calls.filter((call) => call.method === 'GET' && call.path === '/api/models').length === 3);
    await waitFor(rowGone);
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
    }
    stale.release();
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
    }
    mounted!.unmount();
    mounted = await renderPage(
      createElement(MemoryRouter, { initialEntries: ['/models'] }, createElement(Models)),
      (text) => text.includes('Recommended models'),
    );
    expect(mounted.container.querySelector(`tr[data-model="${NAME}"]`)).toBeNull();
    neverAnswers.release();
  });

  it('shows the models folder and the drawer Path under the home folder as ~/…, copying the full path', async () => {
    const original = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    onTestFinished(() => {
      if (original === undefined) Reflect.deleteProperty(navigator, 'clipboard');
      else Object.defineProperty(navigator, 'clipboard', original);
    });
    const routes = modelsRoutes({
      '/health': { status: 'ok', modelsDir: '/models', tracesDir: '', runtimeDir: '', homeDir: '/models' },
    });
    await mountModels(routes, `/models?focus=${NAME}`);
    await waitFor(() => byRole(document, 'dialog', NAME) !== null);
    const drawer = byRole(document, 'dialog', NAME)!;
    await waitFor(() => drawer.querySelector(`[title="/models/${NAME}"]`)?.textContent === `~/${NAME}`);
    expect(mounted!.container.querySelector('[title="/models"]')?.textContent).toBe('~');
    await act(async () => byRole(drawer, 'button', 'Copy Path')?.click());
    expect(writeText).toHaveBeenCalledWith(`/models/${NAME}`);
  });

  it('the palette copies the full models folder and its toast shows it under ~', async () => {
    const original = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    onTestFinished(() => {
      if (original === undefined) Reflect.deleteProperty(navigator, 'clipboard');
      else Object.defineProperty(navigator, 'clipboard', original);
    });
    const success = vi.spyOn(toast, 'success');
    const routes = modelsRoutes({
      '/health': { status: 'ok', modelsDir: '/home/ann/models', tracesDir: '', runtimeDir: '', homeDir: '/home/ann' },
    });
    routes['/models'] = { ...(routes['/models'] as ModelsResponse), dir: '/home/ann/models' };
    await mountModels(routes);
    await waitFor(() => mounted!.container.querySelector('[title="/home/ann/models"]')?.textContent === '~/models');
    const copy = pageCommands.find((command) => command.id === 'models:copy-dir');
    expect(copy).toBeTruthy();
    await act(async () => {
      await copy!.run();
    });
    await waitFor(() => success.mock.calls.length > 0);
    expect(writeText).toHaveBeenCalledWith('/home/ann/models');
    expect(success).toHaveBeenCalledWith('Path copied', { description: '~/models' });
  });

  it('a pointer-down on the drawer closes the delete confirm and keeps the drawer', async () => {
    await mountModels(modelsRoutes(), `/models?focus=${NAME}`);
    const drawer = await (async () => {
      await waitFor(() => byRole(document, 'dialog', NAME) !== null);
      return byRole(document, 'dialog', NAME)!;
    })();
    await act(async () => buttonIn(drawer, 'Delete model').click());
    expect(popover()).toBeTruthy();
    const heading = drawer.querySelector('h2') ?? drawer;
    await waitFor(() => {
      // Radix defers a primary-button outside press to its click.
      heading.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
      heading.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      return ![...document.querySelectorAll('[role="dialog"]')].some((el) => el.textContent?.includes('Delete model?'));
    });
    expect(byRole(document, 'dialog', NAME)).toBeTruthy();
  });

  it('Escape closes the drawer and puts focus back on the row that opened it', async () => {
    await mountModels(modelsRoutes());
    const row = modelRow();
    await act(async () => {
      row.focus();
      row.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    await waitFor(() => byRole(document, 'dialog', NAME) !== null);
    await act(async () => {
      document.activeElement?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    await waitFor(() => byRole(document, 'dialog', NAME) === null, 2000);
    await waitFor(() => document.activeElement === modelRow(), 2000);
  });

  it('after a delete from the drawer the drawer is gone and focus is on the scroll root', async () => {
    const routes = modelsRoutes({ [`/models/${NAME}`]: { deleted: true, name: NAME } });
    await mountModels(routes);
    const link = modelRow().querySelector<HTMLAnchorElement>(`a[href="/models?focus=${NAME}"]`)!;
    await act(async () => {
      link.focus();
      link.click();
    });
    await waitFor(() => byRole(document, 'dialog', NAME) !== null);
    const drawer = byRole(document, 'dialog', NAME)!;
    await act(async () => buttonIn(drawer, 'Delete model').click());
    await act(async () => buttonIn(popover(), 'Delete').click());
    await waitFor(() => byRole(document, 'dialog', NAME) === null, 2000);
    for (let i = 0; i < 10; i++) {
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
    }
    await waitFor(() => document.activeElement?.hasAttribute('data-scroll-root') === true);
  });
});
