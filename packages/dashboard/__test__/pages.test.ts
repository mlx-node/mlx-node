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

import { INITIAL_PROGRESS } from '@/components/download-progress';
import { useHost } from '@/lib/host';
import { categoryLabels, truncateMiddle } from '@/lib/labels';
import { modelShareLabel } from '@/lib/metrics-view';
import { FOLDER_PICKED_MESSAGE, PICK_FOLDER_MESSAGE } from '@/lib/pick-folder';
import { toast } from '@/lib/toast';
import type {
  CacheResponse,
  CacheScope,
  AdoptPlanResponse,
  CatalogBlocker,
  CatalogBuildItem,
  CatalogItem,
  CatalogResponse,
  CatalogUpdatesResponse,
  ColdTierHealth,
  DownloadJob,
  DownloadsResponse,
  HostResponse,
  LocatePlanResponse,
  ModelsResponse,
} from '@/lib/types';
import Cache from '@/pages/cache';
import Metrics from '@/pages/metrics';
import Models from '@/pages/models';
import {
  catalogActionState,
  CatalogCardView,
  type CatalogCardViewProps,
  CatalogDownloadAction,
  cleanupContent,
  hostLine,
  hostNote,
  type LocateControl,
  locateContent,
  updateContent,
} from '@/pages/models/catalog';
import { type CatalogDownloadProps, useModelDownloads } from '@/pages/models/use-model-downloads';
import Overview from '@/pages/overview';
import { SystemOneModelsDrawer } from '@/pages/system-one/models-drawer';
import { type AppCommand, usePageCommands } from '@/shell/commands';
import { act, createElement, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, onTestFinished, vi } from 'vite-plus/test';

import { MODEL_CATALOG } from '../../agent/src/catalog.js';
import type { ApiResponse } from '../src/api/errors.js';
import type { DownloadEvent } from '../src/download.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import type { ApiCall } from '../src/runtime.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import { splitWhitespace } from '../ui/src/lib/text-scan.js';
import { reloadMountedQueries, useJson } from '../ui/src/lib/use-api.js';
import {
  byModelRow,
  dayStart,
  distributions,
  emptySummary,
  live,
  liveModel,
  prefixCacheResponse,
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
  const prefix = '/cache' in routes && !('/cache/prefix' in routes) ? { '/cache/prefix': prefixCacheResponse() } : {};
  restoreApi = stubApi({ ...prefix, ...routes });
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

  /** The state of the card's one build, plus the entry fields a test sets. */
  type CardOverrides = Partial<CatalogBuildItem> & Pick<CatalogItem, 'category' | 'draft'>;

  const HOST_64: HostResponse = { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null };

  function catalogRoutes(
    overrides: CardOverrides,
    jobs: DownloadJob[] = [],
    updates: CatalogUpdatesResponse = { items: [] },
  ): Record<string, unknown> {
    const { category, draft, ...build } = overrides;
    const models: ModelsResponse = {
      models: [],
      warnings: [],
      dir: '/models',
    };
    const catalog: CatalogResponse = {
      items: [
        {
          label: LABEL,
          description: 'Best tool use',
          ...(category === undefined ? {} : { category }),
          ...(draft === undefined ? {} : { draft }),
          builds: [
            {
              id: 'MXFP4',
              hfRepo: REPO,
              sizeGb: 23.3,
              source: 'MLX',
              slug: SLUG,
              installed: false,
              present: false,
              heldBy: null,
              blockedByForeignDir: false,
              blocker: null,
              localRevision: null,
              localAssetsRepo: null,
              localAssetsRevision: null,
              located: null,
              listed: false,
              ...build,
            },
          ],
        },
      ],
    };
    const downloads: DownloadsResponse = { jobs };
    return {
      '/models': models,
      '/host': HOST_64,
      '/catalog': catalog,
      '/catalog/updates': updates,
      '/downloads': downloads,
    };
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
    return {
      id: 'job-1',
      slug: SLUG,
      repo: REPO,
      kind: 'install',
      state: 'running',
      receivedBytes: 0,
      totalBytes: 1024,
      ...overrides,
    };
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
  /** Ids of the same commands. */
  let paletteIds: string[] = [];
  /** The same commands. */
  let paletteCommands: readonly AppCommand[] = [];
  function PaletteProbe(): null {
    const commands = usePageCommands();
    paletteCommands = commands;
    paletteLabels = commands.map((command) => String(command.label));
    paletteIds = commands.map((command) => command.id);
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

  it('keys the palette Install command by the build folder', async () => {
    recordRequests(catalogRoutes({}));
    await mountModels();
    // Two builds of one repository share `hfRepo`; only the folder tells their commands apart.
    expect(paletteIds).toContain(`models:install:${SLUG}`);
  });

  it('names the other build a folder holds and offers no Install into it', async () => {
    recordRequests(catalogRoutes({ heldBy: 'UD-Q4_K_M' }));
    await mountModels();
    const card = mounted!.container.querySelector(`[data-model="${SLUG}"]`)!;
    // Mutation: drop the held branch of catalogActionState → the card offers Install → red.
    expect(card.querySelector('[data-row="action"]')?.textContent).toContain('Holds UD-Q4_K_M');
    expect(buttonLabels()).not.toContain('Install');
    await act(async () => buttonNamed(card, 'Show why').click());
    expect(document.body.textContent).toContain(
      `${SLUG} holds UD-Q4_K_M. It is a working model, so this card does not remove it. Delete it in Local models to install MXFP4 here.`,
    );
    expect(paletteLabels).not.toContain(`Install ${LABEL}`);
  });

  it('follows a job by its folder, not by its repository', async () => {
    // A job of another build of the same repository installs to another folder.
    recordRequests(catalogRoutes({}, [downloadJob({}), downloadJob({ id: 'job-sibling', slug: `${SLUG}-ud-q4_k_m` })]));
    await mountModels();
    // Mutation: key `active` by job.repo → the card follows no job, or the sibling's → red.
    expect(openedStreams).toEqual(['job-1']);
  });

  /** {@link catalogRoutes} plus a streamed entry with a memory floor, on a Mac with `memGib` of RAM. */
  function floorRoutes(memGib: number): Record<string, unknown> {
    const routes = catalogRoutes({});
    const catalog = routes['/catalog'] as CatalogResponse;
    const first = catalog.items[0]!;
    const streamed: CatalogItem = {
      ...first,
      label: 'Hy4-flash',
      description: '512-expert MoE streamed from SSD',
      minMemoryGb: 96,
      builds: [
        {
          ...first.builds[0],
          id: 'Q4_K_M',
          hfRepo: 'AngelSlim/Hy4-flash-GGUF',
          slug: 'hy4-flash-gguf',
          sizeGb: 168.3,
        },
      ],
    };
    routes['/catalog'] = { items: [...catalog.items, streamed] };
    routes['/host'] = { totalMemBytes: memGib * 1024 ** 3, diskFreeBytes: null };
    return routes;
  }

  it('leaves out a recommended model whose memory floor this Mac misses', async () => {
    await mount(createElement(MemoryRouter, null, createElement(Models)), floorRoutes(64), LABEL);
    await settle();
    expect(mounted!.container.querySelector(`[data-model="${SLUG}"]`)).not.toBeNull();
    expect(mounted!.container.querySelector('[data-model="hy4-flash-gguf"]')).toBeNull();
  });

  it('offers a streamed model at its memory floor, with no weights-larger-than-memory note', async () => {
    const text = await mount(createElement(MemoryRouter, null, createElement(Models)), floorRoutes(96), 'Hy4-flash');
    expect(mounted!.container.querySelector('[data-model="hy4-flash-gguf"]')).not.toBeNull();
    expect(text).toContain(LABEL);
    expect(text).not.toContain("larger than this Mac's memory");
    const installs = Array.from(mounted!.container.querySelectorAll('button')).filter(
      (button) => (button.textContent ?? '').trim() === 'Install',
    );
    expect(installs).toHaveLength(2);
  });

  /** `/models` never answers: the walk of every model folder must not hold the cards back. */
  const modelsNever = (path: string): Promise<ApiResponse> | undefined =>
    path === '/models' ? new Promise<ApiResponse>(() => {}) : undefined;

  it.each<[number, boolean]>([
    [128, true],
    [64, false],
  ])('filters the cards by the /host memory while /models never answers (%i GiB)', async (gib, floored) => {
    serveJobs(
      floorRoutes(gib),
      () => ({ ok: false, status: 404, code: 'E_NOT_FOUND', message: 'no' }),
      undefined,
      modelsNever,
    );
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Models)), (text) =>
      text.includes(' installed'),
    );
    expect(mounted.container.querySelector('[data-model="hy4-flash-gguf"]') !== null).toBe(floored);
    expect(mounted.text()).toContain(floored ? '0 of 2 installed' : '0 of 1 installed');
  });

  it('keeps the loading shape and no count until /host answers, and asks for /catalog before /models', async () => {
    let answerHost!: () => void;
    const host = new Promise<ApiResponse>((resolve) => {
      answerHost = () =>
        resolve({ ok: true, status: 200, body: { totalMemBytes: 128 * 1024 ** 3, diskFreeBytes: null } });
    });
    const calls = serveJobs(
      floorRoutes(128),
      () => ({ ok: false, status: 404, code: 'E_NOT_FOUND', message: 'no' }),
      undefined,
      (path) => (path === '/host' ? host : modelsNever(path)),
    );
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Models)), (text) =>
      text.includes('Recommended models'),
    );
    await waitFor(() => calls.some((call) => call.path === '/catalog'));
    await settle();
    // Mutation: filter by an unknown host → "0 of 1 installed" for a list that is still filtered → red.
    expect(mounted.text()).not.toContain('installed');
    expect(mounted.container.querySelector('[data-model]')).toBeNull();
    expect(mounted.container.querySelector('[data-catalog-card]')).not.toBeNull();
    answerHost();
    await waitFor(() => (mounted!.text() ?? '').includes('0 of 2 installed'));
    const order = calls.filter((call) => call.method === 'GET').map((call) => call.path);
    expect(order.indexOf('/catalog')).toBeLessThan(order.indexOf('/models'));
  });

  it('links an installed card to Details and a held one to Local models while /models never answers', async () => {
    serveJobs(
      catalogRoutes({ present: true, installed: true, listed: true }),
      () => ({ ok: false, status: 404, code: 'E_NOT_FOUND', message: 'no' }),
      undefined,
      modelsNever,
    );
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Models)), (text) =>
      text.includes('Details'),
    );
    const card = mounted.container.querySelector(`[data-model="${SLUG}"]`)!;
    expect(card.querySelector('a[href^="/models"]')?.getAttribute('href')).toBe(`/models?focus=${SLUG}`);
    mounted.unmount();
    restoreApi?.();
    serveJobs(
      catalogRoutes({ heldBy: 'UD-Q4_K_M', listed: true }),
      () => ({ ok: false, status: 404, code: 'E_NOT_FOUND', message: 'no' }),
      undefined,
      modelsNever,
    );
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Models)), (text) =>
      text.includes('Show in Local models'),
    );
    expect(mounted.container.querySelector(`[data-model="${SLUG}"] a[href="/models?focus=${SLUG}"]`)).not.toBeNull();
  });

  it('hostNote keeps the weights note for entries without a memory floor only', () => {
    const host = { totalMemBytes: 128 * 1024 ** 3, diskFreeBytes: null };
    expect(hostNote({ sizeGb: 168.3, present: false }, undefined, host)).toContain("larger than this Mac's memory");
    expect(hostNote({ sizeGb: 168.3, present: false }, 96, host)).toBeNull();
  });

  it.each([96, 128])('hostNote gives Qwen3.8-Flash-Next and Hy4-flash no note on a %i GiB Mac', (gib) => {
    const host = { totalMemBytes: gib * 1024 ** 3, diskFreeBytes: null };
    for (const label of ['Qwen3.8-Flash-Next', 'Hy4-flash']) {
      const entry = MODEL_CATALOG.find((item) => item.label === label)!;
      expect(hostNote({ sizeGb: entry.builds[0].sizeGb, present: false }, entry.minMemoryGb, host), label).toBeNull();
    }
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
    expect(buttonLabels()).toContain('Add');
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
      expect(companion.getAttribute('aria-label')).toBe('DFlash2 add-on');
      expect(companion.closest('[data-row]')?.getAttribute('data-row')).toBe('extras');
      const button = companion.querySelector('button');
      if (present) {
        expect(companion.textContent).toContain('added');
        expect(companion.textContent).not.toContain('waits for the model');
        expect(button).toBeNull();
      } else {
        expect(companion.textContent).toContain('Optional');
        expect(companion.textContent).toContain('3.85 GB');
        expect(button?.textContent?.trim()).toBe('Add');
        expect(button?.getAttribute('aria-label')).toBe('Add DFlash2, 3.85 GB');
        expect(button?.disabled).toBe(false);
      }
    },
  );

  it('resumes a companion download and subscribes to its own progress', async () => {
    const draft = draftItem();
    recordRequests(
      catalogRoutes({ present: true, installed: true, draft }, [
        downloadJob({ id: 'draft-job', slug: draft.slug, repo: draft.hfRepo }),
      ]),
    );
    await mountModels();
    expect(openedStreams).toContain('draft-job');
    const card = mounted!.container
      .querySelector(`a[href="https://huggingface.co/${draft.hfRepo}"]`)!
      .closest('[data-model]')!;
    const companion = card.querySelector('[role="group"][aria-label="DFlash2 add-on"]')!;
    expect(companion.textContent).toContain('Cancel');
    expect(companion.textContent).not.toContain('added');
    expect(card.querySelector('[data-status="installed"]')?.textContent).toContain('Installed');
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
    linkToFile: false,
    locked: false,
  };
  const CLEANUP = 'Clean up & update…';

  /** A confirmation's text with its white space folded. */
  function dialogText(title = 'Clean up and update?'): string {
    return splitWhitespace(confirmDialog(title)?.textContent ?? '').join(' ');
  }

  it('names what is in the way and offers Clean up & update instead of an Install the runner refuses', async () => {
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ blockedByForeignDir: true, blocker: BLOCKER }),
      LABEL,
    );
    expect(buttonLabels()).toContain(CLEANUP);
    expect(buttonLabels()).not.toContain('Install');
    expect(text).toContain('Needs cleanup');
    // The long text is in the confirmation, not on the card.
    expect(text).not.toContain('that the app did not create.');
    await act(async () => buttonNamed(mounted!.container, CLEANUP).click());
    const dialog = dialogText();
    expect(dialog).toContain(`${SLUG} is a folder that the app did not create.`);
    expect(dialog).toContain(`To the Trash The folder ${SLUG}: 1 file (183 MB): ${LEFTOVER}.`);
    expect(dialog).toContain('Downloaded Qwen3.8-27B MXFP4, 23.3 GB, into a new folder.');
  });

  /** A real folder with a model the app did not create: the GGUF, the two head files and a README. */
  const HELD: CatalogBlocker = {
    ...BLOCKER,
    fileCount: 4,
    sizeBytes: 7_612_909_326,
    names: [],
    listed: true,
    holdsModel: true,
  };
  const NOTE = `${SLUG} holds a model that the app did not create. Update checks its files against the catalog and downloads the files that are missing. It deletes nothing.`;

  it.each<[string, boolean]>([
    ['not listed', false],
    ['listed in Local models', true],
  ])('offers Update over a real folder that holds a model (%s)', async (_name, listed) => {
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ blockedByForeignDir: true, blocker: { ...HELD, listed } }),
      LABEL,
    );
    expect(liveButtonLabels()).toContain('Update…');
    expect(buttonLabels()).not.toContain(CLEANUP);
    expect(buttonLabels()).not.toContain('Install');
    expect(text).toContain('Needs update');
    expect(text).not.toContain(NOTE);
    expect(text).not.toContain('Needs cleanup');
  });

  it.each<[string, Partial<CatalogBlocker>, string[]]>([
    [
      'a link to a folder of model files',
      { kind: 'symlink', fileCount: 0, sizeBytes: 0, linkTarget: '/Volumes/x/m' },
      [
        `${SLUG} is a link that the app did not create.`,
        'Removed Only the link. It points to /Volumes/x/m.',
        'Stays That folder and every file in it. The app does not change it and no longer uses it.',
        'The link is removed only when the download is complete and checked. If you cancel or the download fails, nothing changes. The old files keep their disk space.',
      ],
    ],
    [
      'a folder too large to list',
      { fileCount: 2000, truncated: true },
      [
        `${SLUG} is a folder that the app did not create.`,
        `To the Trash The whole folder ${SLUG}: more than 2,000 files (more than 7.1 GB). The app could not list all of it.`,
        'It moves to the Trash only when the download is complete and checked.',
      ],
    ],
    [
      'a listed folder too large to list',
      { fileCount: 2000, truncated: true, listed: true },
      ['Local models lists this folder as a model. Make sure that you do not need it.'],
    ],
  ])('offers Clean up & update over %s that holds a model, never Show why', async (_name, overrides, lines) => {
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ blockedByForeignDir: true, blocker: { ...HELD, listed: false, ...overrides } }),
      LABEL,
    );
    expect(buttonLabels()).not.toContain('Update…');
    expect(buttonLabels()).not.toContain('Show why');
    expect(buttonLabels()).not.toContain('Install');
    expect(text).toContain('Needs cleanup');
    await act(async () => buttonNamed(mounted!.container, CLEANUP).click());
    const dialog = dialogText();
    for (const line of lines) expect(dialog).toContain(line);
  });

  it('says No access with the reason for a folder the app cannot write, and offers nothing to start', async () => {
    const text = await mount(
      createElement(MemoryRouter, null, createElement(Models)),
      catalogRoutes({ blockedByForeignDir: true, blocker: { ...HELD, locked: true } }),
      LABEL,
    );
    expect(text).toContain('No access');
    expect(text).not.toContain('Needs cleanup');
    expect(buttonLabels()).not.toContain(CLEANUP);
    expect(buttonLabels()).not.toContain('Update…');
    await act(async () => buttonNamed(mounted!.container, 'Show why').click());
    expect(splitWhitespace(document.body.textContent ?? '').join(' ')).toContain(
      `The app has no write permission on ${SLUG}, so it cannot move it or add files to it. Change its permissions (Finder, Get Info), then open this page again.`,
    );
  });

  /** The visible text of a confirmation description. */
  function textOf(node: ReactNode): string {
    const holder = document.createElement('div');
    holder.innerHTML = renderToStaticMarkup(createElement('div', null, node));
    return splitWhitespace(holder.textContent ?? '').join(' ');
  }

  const blocker = (overrides: Partial<CatalogBlocker>): CatalogBlocker => ({ ...BLOCKER, names: [], ...overrides });
  const handlers = { onCleanup: () => Promise.resolve(), onConfirmStarted: () => {} };
  const WHAT = { name: 'Qwen3.8-27B MXFP4', sizeGb: 18.7, spaceNote: null };
  const LINK_NOTICE =
    'The link is removed only when the download is complete and checked. If you cancel or the download fails, nothing changes.';
  const TRASH_NOTICE =
    'It moves to the Trash only when the download is complete and checked. If you cancel or the download fails, nothing changes. You can take it back from the Trash. Its disk space is free only after you empty the Trash.';
  const DOWNLOADED = 'Downloaded Qwen3.8-27B MXFP4, 18.7 GB, into a new folder.';

  it.each<[string, Partial<CatalogBlocker>, string]>([
    [
      'a live link',
      { kind: 'symlink', fileCount: 0, linkTarget: '/Volumes/x/m' },
      `${SLUG} is a link that the app did not create. Removed Only the link. It points to /Volumes/x/m. Stays That folder and every file in it. The app does not change it and no longer uses it. ${DOWNLOADED}`,
    ],
    [
      'a dead link',
      { kind: 'symlink', fileCount: 0, linkTarget: '/Volumes/x/m', dangling: true },
      `${SLUG} is a link that the app did not create. Removed Only the link. It points to /Volumes/x/m, which is not there. ${DOWNLOADED}`,
    ],
    [
      'a link to a file',
      { kind: 'symlink', fileCount: 0, linkTarget: '/m/model.gguf', linkToFile: true },
      `${SLUG} is a link that the app did not create. Removed Only the link. It points to /m/model.gguf. ${DOWNLOADED}`,
    ],
    [
      'a link whose text is unknown',
      { kind: 'symlink', fileCount: 0, linkTarget: null },
      `${SLUG} is a link that the app did not create. Removed Only the link. Stays That folder and every file in it. The app does not change it and no longer uses it. ${DOWNLOADED}`,
    ],
    [
      'a folder with named files',
      { fileCount: 2, sizeBytes: 2048, names: ['a.bin', 'sub/b.bin'] },
      `${SLUG} is a folder that the app did not create. To the Trash The folder ${SLUG}: 2 files (2.0 KB): a.bin, sub/b.bin. ${DOWNLOADED}`,
    ],
    [
      'a folder of four files',
      { fileCount: 4, sizeBytes: 2048 },
      `${SLUG} is a folder that the app did not create. To the Trash The folder ${SLUG}: 4 files (2.0 KB). ${DOWNLOADED}`,
    ],
    [
      'an empty folder',
      { fileCount: 0, sizeBytes: 0 },
      `${SLUG} is a folder that the app did not create. To the Trash The empty folder ${SLUG}. ${DOWNLOADED}`,
    ],
    [
      'a folder too large to list',
      { fileCount: 2000, sizeBytes: 2048, truncated: true },
      `${SLUG} is a folder that the app did not create. To the Trash The whole folder ${SLUG}: more than 2,000 files (more than 2.0 KB). The app could not list all of it. ${DOWNLOADED}`,
    ],
    [
      'a folder the app could not read',
      { fileCount: 0, sizeBytes: 0, truncated: true },
      `${SLUG} is a folder that the app did not create. To the Trash The whole folder ${SLUG}. The app could not read it, so its size is not known. ${DOWNLOADED}`,
    ],
    [
      'a file',
      { kind: 'file', fileCount: 1, sizeBytes: 4096 },
      `${SLUG} is a file that the app did not create. To the Trash The file ${SLUG} (4.0 KB). ${DOWNLOADED}`,
    ],
  ])('says what Clean up & update removes and downloads: %s', (_name, overrides, expected) => {
    const content = cleanupContent({ slug: SLUG, blocker: blocker(overrides) }, WHAT, handlers);
    expect(textOf(content.description)).toBe(expected);
    expect(content).toMatchObject({
      title: 'Clean up and update?',
      confirmLabel: 'Clean up & update',
      workingLabel: 'Starting…',
      successLabel: 'Started',
      tone: 'neutral',
      errorTitle: 'Could not start the download',
    });
  });

  it.each<[string, Partial<CatalogBlocker>, string | null, string]>([
    [
      'a link to model files',
      { kind: 'symlink', holdsModel: true },
      null,
      `${LINK_NOTICE} The old files keep their disk space.`,
    ],
    ['a link to other files', { kind: 'symlink' }, null, LINK_NOTICE],
    ['a link that lists, which stays', { kind: 'symlink', listed: true }, null, LINK_NOTICE],
    ['a folder', {}, null, TRASH_NOTICE],
    [
      'a folder Local models lists',
      { listed: true, holdsModel: true },
      null,
      `Local models lists this folder as a model. Make sure that you do not need it. ${TRASH_NOTICE}`,
    ],
    [
      'a short disk',
      {},
      'Needs about 18.7 GB. 10.0 GB is free on this disk.',
      `${TRASH_NOTICE} Needs about 18.7 GB. 10.0 GB is free on this disk.`,
    ],
  ])('notices: %s', (_name, overrides, spaceNote, notice) => {
    const content = cleanupContent({ slug: SLUG, blocker: blocker(overrides) }, { ...WHAT, spaceNote }, handlers);
    expect(textOf(content.notice)).toBe(notice);
  });

  it('says a download of 20 GB or more can take an hour, and not one of 19.9 GB', () => {
    const at = (sizeGb: number) =>
      textOf(cleanupContent({ slug: SLUG, blocker: BLOCKER }, { ...WHAT, sizeGb }, handlers).description);
    expect(at(20)).toContain('20 GB, into a new folder. A download of this size can take an hour or more.');
    expect(at(19.9)).not.toContain('an hour');
  });

  it('leads with the server sentence of a refused Update, then the whole model', () => {
    const sentence = `"x.gguf" in "${SLUG}" is not the catalog file. Move it out of the folder, then use Update again.`;
    const content = cleanupContent({ slug: SLUG, blocker: HELD }, WHAT, { ...handlers, companion: true }, sentence);
    expect(content.title).toBe('Clean up and update companion weights?');
    expect(textOf(content.description)).toBe(
      `Update cannot use ${SLUG} as it is. ${sentence} Clean up & update does this instead: To the Trash The folder ${SLUG}: 4 files (7.1 GB). Downloaded The whole model: Qwen3.8-27B MXFP4, 18.7 GB, into a new folder.`,
    );
  });

  it('truncates a long file name in the middle and keeps the full name on hover', () => {
    const long = `${'x'.repeat(30)}-middle-${'y'.repeat(30)}.gguf`;
    const html = renderToStaticMarkup(
      createElement(
        'div',
        null,
        cleanupContent({ slug: SLUG, blocker: blocker({ fileCount: 1, names: [long] }) }, WHAT, handlers).description,
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
        cleanupContent({ slug: SLUG, blocker: blocker({ fileCount: 4, sizeBytes: 2048 }) }, WHAT, handlers).description,
      ),
    );
    expect(html).toContain(`The folder ${SLUG}: 4 files (2.0 KB).`);
  });

  /**
   * Serve the catalog routes plus a POST /downloads answered by `post` and a POST /downloads/adopt-plan answered by
   * `plan`. The route stub matches by path alone, and a start needs the GET and the POST of `/downloads` to answer
   * differently.
   */
  function serveJobs(
    routes: Record<string, unknown>,
    post: (body: unknown) => ApiResponse,
    plan?: (body: unknown) => ApiResponse | Promise<ApiResponse>,
    get?: (path: string) => Promise<ApiResponse> | undefined,
    locatePlan?: (body: unknown) => ApiResponse | Promise<ApiResponse>,
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
          if (call.method === 'POST' && path === '/downloads/adopt-plan' && plan !== undefined) return plan(call.body);
          if (call.method === 'POST' && path === '/downloads/locate-plan' && locatePlan !== undefined) {
            return locatePlan(call.body);
          }
          const answer = call.method === 'GET' ? get?.(path) : undefined;
          if (answer !== undefined) return answer;
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
    connectDashboardApi(bindEventTargetPort(port1));
    restoreApi = () => {
      disconnectDashboardApi();
      dispose();
      port1.close();
      port2.close();
    };
    return calls;
  }

  function confirmDialog(title = 'Clean up and update?'): HTMLElement | undefined {
    return [...document.querySelectorAll<HTMLElement>('[role="dialog"]')].find((el) => el.textContent?.includes(title));
  }

  function buttonNamed(scope: ParentNode, label: string): HTMLButtonElement {
    const button = [...scope.querySelectorAll<HTMLButtonElement>('button')].find(
      (b) => (b.textContent ?? '').trim() === label,
    );
    expect(button, label).toBeTruthy();
    return button!;
  }

  it.each<[string, CatalogBlocker, string]>([
    ['a folder', BLOCKER, 'Moves the old folder to the Trash when the download is complete.'],
    [
      'a link',
      { ...BLOCKER, kind: 'symlink', fileCount: 0, sizeBytes: 0, names: [], linkTarget: '/Volumes/x/m' },
      'Removes the link when the download is complete. Its folder is not changed.',
    ],
  ])(
    'confirms %s, posts the numbers the user saw and nothing else, then follows the job',
    async (_name, seen, line) => {
      const calls = serveJobs(catalogRoutes({ blockedByForeignDir: true, blocker: seen }), () => ({
        ok: true,
        status: 202,
        body: { id: 'job-cleanup', slug: SLUG, repo: REPO, kind: 'cleanup' },
      }));
      await mountModels();
      await act(async () => buttonNamed(mounted!.container, CLEANUP).click());
      const dialog = confirmDialog()!;
      expect(dialog).toBeTruthy();
      expect(calls.filter((call) => call.method === 'POST')).toEqual([]);

      await act(async () => buttonNamed(dialog, 'Clean up & update').click());
      await waitFor(() => openedStreams.includes('job-cleanup'));
      const posts = calls.filter((call) => call.method === 'POST');
      const { kind, fileCount, sizeBytes, truncated, linkTarget } = seen;
      expect(posts.map((call) => call.body)).toEqual([
        { slug: SLUG, replaceBlocker: { kind, fileCount, sizeBytes, truncated, linkTarget } },
      ]);
      const card = mounted!.container.querySelector(`[data-model="${SLUG}"]`)!;
      // The bar's text for assistive tech, also its tooltip.
      await waitFor(() => (card.textContent ?? '').includes(line));
      expect(buttonNamed(card, 'Cancel')).toBeTruthy();
      expect(card.textContent).not.toContain('Needs cleanup');
    },
  );

  it('names the confirmed entry in the job line and the done toast once the reloaded catalog no longer has it', async () => {
    const success = vi.spyOn(toast, 'success');
    onTestFinished(() => success.mockRestore());
    const seen: CatalogBlocker = { ...BLOCKER, kind: 'file', sizeBytes: 4096, names: [] };
    const after = catalogRoutes({}, [downloadJob({ id: 'job-cleanup', kind: 'cleanup' })]);
    let started = false;
    serveJobs(
      catalogRoutes({ blockedByForeignDir: true, blocker: seen }),
      () => {
        started = true;
        return { ok: true, status: 202, body: { id: 'job-cleanup', slug: SLUG, repo: REPO, kind: 'cleanup' } };
      },
      undefined,
      (path) =>
        started && (path === '/catalog' || path === '/downloads')
          ? Promise.resolve({ ok: true as const, status: 200, body: after[path] })
          : undefined,
    );
    await mountModels();
    await act(async () => buttonNamed(mounted!.container, CLEANUP).click());
    await act(async () => buttonNamed(confirmDialog()!, 'Clean up & update').click());
    await waitFor(() => openedStreams.includes('job-cleanup'));
    await act(async () => reloadMountedQueries());
    await settle();
    const card = mounted!.container.querySelector(`[data-model="${SLUG}"]`)!;
    await waitFor(() =>
      (card.textContent ?? '').includes('Moves the old file to the Trash when the download is complete.'),
    );
    emitDownload('done', { id: 'job-cleanup', outputDir: `/models/${SLUG}`, removed: 'trash' });
    await waitFor(() => success.mock.calls.length > 0);
    expect(success.mock.calls).toEqual([
      ['Download complete', { description: `The old file ${SLUG} is in the Trash.` }],
    ]);
  });

  it('keeps a refused Clean up & update in the confirmation, reloads the catalog, and raises no toast', async () => {
    const error = vi.spyOn(toast, 'error');
    const success = vi.spyOn(toast, 'success');
    onTestFinished(() => {
      error.mockRestore();
      success.mockRestore();
    });
    const calls = serveJobs(catalogRoutes({ blockedByForeignDir: true, blocker: BLOCKER }), () => ({
      ok: false,
      status: 409,
      code: 'E_CONFLICT',
      message: `"${SLUG}" changed on disk. Check it, then try again.`,
    }));
    await mountModels();
    const catalogGets = calls.filter((call) => call.method === 'GET' && call.path === '/catalog').length;
    await act(async () => buttonNamed(mounted!.container, CLEANUP).click());
    await act(async () => buttonNamed(confirmDialog()!, 'Clean up & update').click());
    await waitFor(() => confirmDialog()?.querySelector('[role="alert"]') !== null);
    const alert = confirmDialog()!.querySelector('[role="alert"]')!;
    expect(alert.textContent).toContain('Could not start the download');
    expect(alert.textContent).toContain('changed on disk. Check it, then try again.');
    await waitFor(() => calls.filter((call) => call.method === 'GET' && call.path === '/catalog').length > catalogGets);
    expect(error).not.toHaveBeenCalled();
    expect(success).not.toHaveBeenCalled();
    expect(openedStreams).toEqual([]);
  });

  /** The props of a card over `blocker` whose last job settled and whose catalog has not been reloaded yet. */
  function settlingProps(
    blocker: CatalogBlocker,
    onPlanUpdate: () => Promise<AdoptPlanResponse> = () => Promise.reject(new Error('no plan')),
  ): Parameters<typeof CatalogDownloadAction>[0] {
    return {
      buildName: 'MXFP4',
      modelName: LABEL,
      cleanupName: `${LABEL} MXFP4`,
      host: undefined,
      item: {
        hfRepo: REPO,
        slug: SLUG,
        sizeGb: 23.3,
        installed: false,
        present: false,
        heldBy: null,
        blockedByForeignDir: true,
        blocker,
        localRevision: null,
        localAssetsRepo: null,
        localAssetsRevision: null,
        located: null,
      },
      updateAvailable: false,
      settling: true,
      updateSettling: false,
      job: undefined,
      refused: false,
      refusedSentence: undefined,
      onInstall: () => Promise.resolve(),
      onCleanup: () => Promise.resolve(),
      onConfirmStarted: () => {},
      onPlanUpdate,
      onUpdateRefused: () => {},
      onUpdateInPlace: () => Promise.resolve(),
      onPlanLocate: () => Promise.reject(new Error('no plan')),
      onLocate: () => Promise.resolve(),
      onRecheck: () => {},
      onReplaceLocated: () => Promise.resolve(),
      onDone: () => {},
      onError: () => {},
      onCancelled: () => {},
      onCancel: () => {},
    };
  }

  it('turns Clean up & update off while the last download is being checked', async () => {
    const props = settlingProps(BLOCKER);
    mounted = await renderPage(createElement(CatalogDownloadAction, props), (text) => text.includes(CLEANUP));
    const trigger = buttonNamed(mounted.container, CLEANUP);
    expect(trigger.getAttribute('aria-disabled')).toBe('true');
    await act(async () => trigger.click());
    expect(confirmDialog()).toBeUndefined();
  });

  it.each<[string, Partial<CatalogDownloadProps>, string, string]>([
    [
      'held',
      { item: { ...settlingProps(BLOCKER).item, heldBy: 'UD-Q4_K_M', blocker: null } },
      'Holds UD-Q4_K_M',
      `${SLUG} holds UD-Q4_K_M. It is a working model, so this card does not remove it. Delete it in Local models to install MXFP4 here.`,
    ],
    [
      'locked',
      { item: { ...settlingProps(BLOCKER).item, blocker: { ...BLOCKER, locked: true } } },
      'No access',
      `The app has no write permission on ${SLUG}`,
    ],
    ['cleanup', {}, 'Needs cleanup', `${SLUG} is in the way of this model.`],
  ])('Onboarding names the %s state in the title of its alert', async (_state, fields, title, sentence) => {
    const props = { ...settlingProps(BLOCKER), settling: false, ...fields };
    mounted = await renderPage(createElement(CatalogDownloadAction, props), (text) => text.includes(title));
    expect(mounted.text()).toContain(sentence);
    expect(buttonLabels().includes(CLEANUP)).toBe(title === 'Needs cleanup');
  });

  it('offers the same Clean up & update for a blocked companion, with its own title', async () => {
    const draft = {
      ...draftItem(false),
      blockedByForeignDir: true,
      blocker: { ...BLOCKER, fileCount: 0, sizeBytes: 0, names: [] },
    };
    serveJobs(catalogRoutes({ present: true, installed: true, draft }), () => ({
      ok: true,
      status: 202,
      body: { id: 'job-draft', slug: draft.slug, repo: draft.hfRepo, kind: 'cleanup' },
    }));
    await mountModels();
    const companion = mounted!.container
      .querySelector(`a[href="https://huggingface.co/${draft.hfRepo}"]`)!
      .closest('[role="group"]')!;
    expect(companion.textContent).toContain('needs cleanup');
    await act(async () => buttonNamed(companion, CLEANUP).click());
    const dialog = dialogText('Clean up and update companion weights?');
    expect(dialog).toContain(`${draft.slug} is a folder that the app did not create.`);
    expect(dialog).toContain(`To the Trash The empty folder ${draft.slug}.`);
    expect(dialog).toContain('Downloaded DFlash2, 3.85 GB, into a new folder.');
  });

  const PLAN: AdoptPlanResponse = {
    repo: REPO,
    slug: SLUG,
    token: 'a1'.repeat(32),
    revision: 'b'.repeat(40),
    assetsRevision: 'c'.repeat(40),
    check: { count: 5, bytes: 7_612_907_561 },
    download: { count: 6, bytes: 12_800_000 },
    other: { count: 1, bytes: 1765, names: ['README.md'] },
  };
  const KEEPS =
    'Update deletes nothing and moves nothing. If a file is not the catalog file, Update stops and changes nothing.';

  function updateDialog(title = 'Update model?'): HTMLElement | undefined {
    return confirmDialog(title);
  }

  /** A plan reply that waits for `release`, so a test can look at the card while the plan loads. */
  function heldPlan(reply: ApiResponse): { answer: () => Promise<ApiResponse>; release: () => void } {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    return { answer: () => gate.then(() => reply), release };
  }

  function posts(calls: Array<{ method: string; path: string; body: unknown }>, path: string): unknown[] {
    return calls.filter((call) => call.method === 'POST' && call.path === path).map((call) => call.body);
  }

  it.each<[string, Partial<AdoptPlanResponse>, string]>([
    [
      'checks, downloads and leaves',
      {},
      `Update checks 5 files (7.1 GB) in ${SLUG} against the catalog. It downloads 6 files (12.2 MB) and adds them to the folder. 1 other file stays where it is: README.md. ${KEEPS}`,
    ],
    [
      'nothing to check',
      { check: { count: 0, bytes: 0 } },
      `Update downloads 6 files (12.2 MB) and adds them to the folder. 1 other file stays where it is: README.md. ${KEEPS}`,
    ],
    [
      'nothing to download',
      { download: { count: 0, bytes: 0 } },
      `Update checks 5 files (7.1 GB) in ${SLUG} against the catalog. 1 other file stays where it is: README.md. ${KEEPS}`,
    ],
    [
      'no other file',
      { other: { count: 0, bytes: 0, names: [] } },
      `Update checks 5 files (7.1 GB) in ${SLUG} against the catalog. It downloads 6 files (12.2 MB) and adds them to the folder. ${KEEPS}`,
    ],
    [
      'one of each, and more other files than names',
      {
        check: { count: 1, bytes: 2048 },
        download: { count: 1, bytes: 2048 },
        other: { count: 5, bytes: 10, names: ['.DS_Store', 'README.md', 'notes.txt'] },
      },
      `Update checks 1 file (2.0 KB) in ${SLUG} against the catalog. It downloads 1 file (2.0 KB) and adds it to the folder. 5 other files stay where they are: .DS_Store, README.md, notes.txt. ${KEEPS}`,
    ],
  ])(
    'loads the plan on the click, then says in numbers what Update checks, downloads and leaves: %s',
    async (_name, overrides, expected) => {
      const held = heldPlan({ ok: true, status: 200, body: { ...PLAN, ...overrides } });
      const calls = serveJobs(
        catalogRoutes({ blockedByForeignDir: true, blocker: HELD }),
        () => ({ ok: true, status: 202, body: { id: 'job-update', slug: SLUG, repo: REPO, kind: 'update' } }),
        held.answer,
      );
      await mountModels();
      expect(posts(calls, '/downloads/adopt-plan')).toEqual([]);
      await act(async () => buttonNamed(mounted!.container, 'Update…').click());
      await waitFor(() => posts(calls, '/downloads/adopt-plan').length === 1);
      expect(posts(calls, '/downloads/adopt-plan')).toEqual([{ slug: SLUG }]);
      expect(buttonNamed(mounted!.container, 'Checking…').getAttribute('aria-disabled')).toBe('true');
      await settle();
      expect(updateDialog()).toBeUndefined();

      held.release();
      await waitFor(() => updateDialog() !== undefined);
      const description = updateDialog()!.querySelector('[id]:not(h2)')!;
      expect(splitWhitespace(description.textContent ?? '').join(' ')).toBe(
        `${SLUG} holds a model that the app did not create. ${expected}`,
      );
      expect(posts(calls, '/downloads')).toEqual([]);
    },
  );

  it('drops the plan when the confirmation closes, so the next click loads a new one', async () => {
    const calls = serveJobs(
      catalogRoutes({ blockedByForeignDir: true, blocker: HELD }),
      () => ({ ok: true, status: 202, body: { id: 'job-update', slug: SLUG, repo: REPO, kind: 'update' } }),
      () => ({ ok: true, status: 200, body: PLAN }),
    );
    await mountModels();
    await act(async () => buttonNamed(mounted!.container, 'Update…').click());
    await waitFor(() => updateDialog() !== undefined);
    await act(async () => buttonNamed(updateDialog()!, 'Cancel').click());
    await waitFor(() => updateDialog() === undefined);
    expect(posts(calls, '/downloads/adopt-plan')).toHaveLength(1);
    await act(async () => buttonNamed(mounted!.container, 'Update…').click());
    await waitFor(() => posts(calls, '/downloads/adopt-plan').length === 2);
    await waitFor(() => updateDialog() !== undefined);
    expect(posts(calls, '/downloads')).toEqual([]);
  });

  it('says that the app manages the folder after Update', async () => {
    const content = updateContent(SLUG, PLAN, {
      onUpdateInPlace: () => Promise.resolve(),
      onConfirmStarted: () => {},
    });
    expect(content).toMatchObject({
      title: 'Update model?',
      notice:
        'After Update the app manages this folder. A later update replaces it. Files that you add or change in it are then moved to .staging in the models folder. The app does not delete them.',
      confirmLabel: 'Update',
      workingLabel: 'Starting…',
      successLabel: 'Started',
      tone: 'neutral',
      errorTitle: 'Could not start the update',
    });
    const companion = updateContent(SLUG, PLAN, {
      companion: true,
      onUpdateInPlace: () => Promise.resolve(),
      onConfirmStarted: () => {},
    });
    expect(companion.title).toBe('Update companion weights?');
  });

  it('says a refused plan in a toast and keeps the card as it was', async () => {
    const error = vi.spyOn(toast, 'error');
    onTestFinished(() => error.mockRestore());
    const refusal = `"README.gguf" in "${SLUG}" is a model file that is not part of the catalog model. Move it out of the folder, then use Update again.`;
    let plans = 0;
    const refused = heldPlan({ ok: false, status: 409, code: 'E_CONFLICT', message: refusal });
    const calls = serveJobs(
      catalogRoutes({ blockedByForeignDir: true, blocker: HELD }),
      () => ({ ok: true, status: 202, body: { id: 'job-update', slug: SLUG, repo: REPO, kind: 'update' } }),
      () => (++plans === 1 ? refused.answer() : { ok: true, status: 200, body: PLAN }),
    );
    await mountModels();
    const update = buttonNamed(mounted!.container, 'Update…');
    update.focus();
    await act(async () => update.click());
    await waitFor(() => posts(calls, '/downloads/adopt-plan').length === 1);
    // Mutation: disable the button while the plan loads → a browser moves focus to <body> → red.
    expect(update.textContent).toBe('Checking…');
    expect(update.disabled).toBe(false);
    expect(update.getAttribute('aria-disabled')).toBe('true');
    refused.release();
    await waitFor(() => error.mock.calls.length > 0);
    expect(error.mock.calls).toEqual([['Could not check the folder', { description: refusal }]]);
    expect(document.activeElement).toBe(update);
    const card = mounted!.container.querySelector(`[data-model="${SLUG}"]`)!;
    expect(card.textContent).not.toContain(refusal);
    expect(card.querySelector('[role="alert"]')).toBeNull();
    expect(updateDialog()).toBeUndefined();
    expect(liveButtonLabels()).toContain('Update…');

    await act(async () => buttonNamed(mounted!.container, 'Update…').click());
    await waitFor(() => posts(calls, '/downloads/adopt-plan').length === 2);
    await waitFor(() => updateDialog() !== undefined);
    expect(posts(calls, '/downloads')).toEqual([]);
  });

  it('turns a plan refused for a reason a cleanup solves into Clean up & update, open with the server sentence', async () => {
    const error = vi.spyOn(toast, 'error');
    onTestFinished(() => error.mockRestore());
    const refusal = `"UD-Q4_K_XL/x-00002-of-00004.gguf" in "${SLUG}" is not the catalog file. Its size is different. Move it out of the folder, then use Update again.`;
    const calls = serveJobs(
      catalogRoutes({ blockedByForeignDir: true, blocker: HELD }),
      () => ({ ok: true, status: 202, body: { id: 'job-cleanup', slug: SLUG, repo: REPO, kind: 'cleanup' } }),
      () => ({ ok: false, status: 409, code: 'E_CONFLICT', message: refusal, reason: 'needs-cleanup' }),
    );
    await mountModels();
    await act(async () => buttonNamed(mounted!.container, 'Update…').click());
    // Mutation: ignore the reason → a toast and Update… again → red.
    await waitFor(() => confirmDialog() !== undefined);
    expect(error).not.toHaveBeenCalled();
    const card = mounted!.container.querySelector(`[data-model="${SLUG}"]`)!;
    expect(card.querySelector('[data-row="action"]')?.textContent).toContain('Needs cleanup');
    expect(dialogText()).toContain(
      `Update cannot use ${SLUG} as it is. ${refusal} Clean up & update does this instead:`,
    );
    expect(dialogText()).toContain(`To the Trash The folder ${SLUG}: 4 files (7.1 GB).`);
    expect(dialogText()).toContain('Downloaded The whole model: Qwen3.8-27B MXFP4, 23.3 GB, into a new folder.');
    await act(async () => buttonNamed(confirmDialog()!, 'Clean up & update').click());
    await waitFor(() => openedStreams.includes('job-cleanup'));
    expect(posts(calls, '/downloads')).toEqual([
      {
        slug: SLUG,
        replaceBlocker: {
          kind: 'directory',
          fileCount: 4,
          sizeBytes: 7_612_909_326,
          truncated: false,
          linkTarget: null,
        },
      },
    ]);
  });

  it('offers Update again once the folder changed after a refusal', async () => {
    const refusal = `"x.gguf" in "${SLUG}" is not the catalog file. Move it out of the folder, then use Update again.`;
    const routes = catalogRoutes({ blockedByForeignDir: true, blocker: HELD });
    const fixed = catalogRoutes({
      blockedByForeignDir: true,
      blocker: { ...HELD, fileCount: 3, sizeBytes: 7_000_000_000 },
    });
    let catalogBody = routes['/catalog'];
    serveJobs(
      routes,
      () => ({ ok: true, status: 202, body: { id: 'job-x', slug: SLUG, repo: REPO, kind: 'cleanup' } }),
      () => ({ ok: false, status: 409, code: 'E_CONFLICT', message: refusal, reason: 'needs-cleanup' }),
      (path) => (path === '/catalog' ? Promise.resolve({ ok: true, status: 200, body: catalogBody }) : undefined),
    );
    await mountModels();
    await act(async () => buttonNamed(mounted!.container, 'Update…').click());
    await waitFor(() => confirmDialog() !== undefined);
    await act(async () => buttonNamed(confirmDialog()!, 'Cancel').click());
    await waitFor(() => confirmDialog() === undefined);
    expect(liveButtonLabels()).toContain(CLEANUP);
    // The user moved the file out by hand: the catalog reads other numbers, so the kept refusal no longer applies.
    catalogBody = fixed['/catalog'];
    await act(async () => reloadMountedQueries());
    await waitFor(() => liveButtonLabels().includes('Update…'));
    expect(buttonLabels()).not.toContain(CLEANUP);
  });

  it('posts the token and follows the job after the success pill', async () => {
    const success = vi.spyOn(toast, 'success');
    onTestFinished(() => success.mockRestore());
    const calls = serveJobs(
      catalogRoutes({ blockedByForeignDir: true, blocker: HELD }),
      () => ({ ok: true, status: 202, body: { id: 'job-update', slug: SLUG, repo: REPO, kind: 'update' } }),
      () => ({ ok: true, status: 200, body: PLAN }),
    );
    await mountModels();
    await act(async () => buttonNamed(mounted!.container, 'Update…').click());
    await waitFor(() => updateDialog() !== undefined);
    await act(async () => buttonNamed(updateDialog()!, 'Update').click());
    await waitFor(() => openedStreams.includes('job-update'));
    expect(posts(calls, '/downloads')).toEqual([{ slug: SLUG, adopt: PLAN.token }]);
    const card = mounted!.container.querySelector(`[data-model="${SLUG}"]`)!;
    await waitFor(() => (card.textContent ?? '').includes(`Checks the files in ${SLUG}. Nothing is deleted.`));
    expect(buttonNamed(card, 'Cancel')).toBeTruthy();
    expect(card.textContent).not.toContain('Needs update');

    emitDownload('start', { id: 'job-update', repo: REPO, totalBytes: 6_429_716_416, fileCount: 11 });
    emitDownload('progress', {
      id: 'job-update',
      file: 'Clairic-Flash-UD-Q4_K_XL.gguf',
      receivedBytes: 1024,
      jobReceivedBytes: 1024,
      totalBytes: 6_429_716_416,
      fileIndex: 0,
      fileCount: 11,
      phase: 'check',
    });
    await waitFor(() => (card.textContent ?? '').includes('Checking files · 1.0 kB / 6.4 GB'));
    expect(card.querySelector('[role="progressbar"]')?.getAttribute('aria-label')).toBe(
      'Clairic-Flash-UD-Q4_K_XL.gguf · File 1 of 11',
    );
    emitDownload('done', { id: 'job-update', outputDir: `/models/${SLUG}` });
    await waitFor(() => success.mock.calls.length > 0);
    expect(success.mock.calls).toEqual([['Update complete', { description: SLUG }]]);
  });

  it('keeps a refused Update in the confirmation, reloads the catalog, raises no toast', async () => {
    const error = vi.spyOn(toast, 'error');
    const success = vi.spyOn(toast, 'success');
    onTestFinished(() => {
      error.mockRestore();
      success.mockRestore();
    });
    const calls = serveJobs(
      catalogRoutes({ blockedByForeignDir: true, blocker: HELD }),
      () => ({
        ok: false,
        status: 409,
        code: 'E_CONFLICT',
        message: `"${SLUG}" changed on disk. Close this, then use Update again.`,
      }),
      () => ({ ok: true, status: 200, body: PLAN }),
    );
    await mountModels();
    const catalogGets = calls.filter((call) => call.method === 'GET' && call.path === '/catalog').length;
    await act(async () => buttonNamed(mounted!.container, 'Update…').click());
    await waitFor(() => updateDialog() !== undefined);
    await act(async () => buttonNamed(updateDialog()!, 'Update').click());
    await waitFor(() => updateDialog()?.querySelector('[role="alert"]') !== null);
    const alert = updateDialog()!.querySelector('[role="alert"]')!;
    expect(alert.textContent).toContain('Could not start the update');
    expect(alert.textContent).toContain('changed on disk. Close this, then use Update again.');
    await waitFor(() => calls.filter((call) => call.method === 'GET' && call.path === '/catalog').length > catalogGets);
    expect(error).not.toHaveBeenCalled();
    expect(success).not.toHaveBeenCalled();
    expect(openedStreams).toEqual([]);
  });

  it('turns Update off while the last download is being checked', async () => {
    const plan = vi.fn(() => Promise.resolve(PLAN));
    mounted = await renderPage(createElement(CatalogDownloadAction, settlingProps(HELD, plan)), (text) =>
      text.includes('Needs update'),
    );
    const trigger = buttonNamed(mounted.container, 'Update');
    expect(trigger.getAttribute('aria-disabled')).toBe('true');
    await act(async () => trigger.click());
    await settle();
    expect(plan).not.toHaveBeenCalled();
    expect(updateDialog()).toBeUndefined();
  });

  it('the System One drawer shows the same Update', async () => {
    const routes = catalogRoutes({ blockedByForeignDir: true, blocker: HELD, category: 'systemone' });
    serveJobs(
      routes,
      () => ({ ok: true, status: 202, body: { id: 'job-update', slug: SLUG, repo: REPO, kind: 'update' } }),
      () => ({ ok: true, status: 200, body: PLAN }),
    );
    function DrawerPage(): ReactNode {
      const models = useJson<ModelsResponse>('/models');
      const catalog = useJson<CatalogResponse>('/catalog');
      const host = useHost();
      const updates = useJson<CatalogUpdatesResponse>('/catalog/updates');
      const downloads = useModelDownloads(models, catalog, updates, host);
      return createElement(SystemOneModelsDrawer, {
        open: true,
        onOpenChange: () => {},
        catalog,
        host,
        updates,
        downloads,
      });
    }
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(DrawerPage)), () =>
      (document.body.textContent ?? '').includes('Needs update'),
    );
    await settle();
    const card = document.body.querySelector<HTMLElement>(`[data-model="${SLUG}"]`)!;
    expect(card.querySelector('[data-row="action"]')?.textContent).toContain('Needs update');
    await act(async () => buttonNamed(card, 'Update…').click());
    await waitFor(() => updateDialog() !== undefined);
    expect(updateDialog()!.textContent).toContain(`${SLUG} holds a model that the app did not create.`);
    expect(updateDialog()!.textContent).toContain('README.md');
    expect(updateDialog()!.textContent).toContain('After Update the app manages this folder.');
  });

  it('names a failed Update "Update failed"', async () => {
    const error = vi.spyOn(toast, 'error');
    onTestFinished(() => error.mockRestore());
    recordRequests({
      ...catalogRoutes({ blockedByForeignDir: true, blocker: HELD }, [
        downloadJob({ id: 'job-update', kind: 'update' }),
      ]),
      '/downloads/job-update': { cancelled: true, id: 'job-update' },
    });
    await mountModels();
    const message = `"${SLUG}" changed on disk during the update. Nothing was changed.`;
    emitDownload('error', { id: 'job-update', message });
    await waitFor(() => error.mock.calls.length > 0);
    expect(error.mock.calls).toEqual([['Update failed', { description: message }]]);
  });

  it('names an Update that failed while the page was away "Update failed"', async () => {
    const error = vi.spyOn(toast, 'error');
    onTestFinished(() => error.mockRestore());
    recordRequests({
      ...catalogRoutes({ blockedByForeignDir: true, blocker: HELD }, [
        downloadJob({ id: 'job-update', kind: 'update', state: 'error' }),
      ]),
      '/downloads/job-update': { cancelled: true, id: 'job-update' },
    });
    await mountModels();
    await waitFor(() => error.mock.calls.length > 0);
    expect(error.mock.calls).toEqual([['Update failed', { description: SLUG }]]);
  });

  it('names a job that failed while the page was away by its kind, before the catalog answers', async () => {
    const error = vi.spyOn(toast, 'error');
    onTestFinished(() => error.mockRestore());
    const routes = catalogRoutes({ blockedByForeignDir: true, blocker: BLOCKER }, [
      downloadJob({ id: 'job-update', kind: 'update', state: 'error' }),
      downloadJob({ id: 'job-cleanup', slug: `${SLUG}-other`, kind: 'cleanup', state: 'error' }),
    ]);
    // `/catalog` runs on a worker and `/downloads` on the main thread: the job list can land first.
    serveJobs(
      routes,
      () => ({ ok: true, status: 202, body: { id: 'job-x', slug: SLUG, repo: REPO, kind: 'install' } }),
      undefined,
      (path) => (path === '/catalog' ? new Promise<ApiResponse>(() => {}) : undefined),
    );
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Models)), (text) =>
      text.includes('Recommended models'),
    );
    await waitFor(() => error.mock.calls.length === 2);
    expect(error.mock.calls).toEqual([
      ['Update failed', { description: SLUG }],
      ['Download failed', { description: `${SLUG}-other` }],
    ]);
  });

  it('tells where the kept files are when a later update is done', async () => {
    const success = vi.spyOn(toast, 'success');
    onTestFinished(() => success.mockRestore());
    const keptDir = `/models/.staging/${SLUG}.kept-0d6f`;
    recordRequests({
      ...catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40) }, [downloadJob({})], {
        items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }],
      }),
      '/downloads/job-1': { cancelled: true, id: 'job-1' },
    });
    await mountModels();
    emitDownload('done', { id: 'job-1', outputDir: `/models/${SLUG}`, keptDir });
    await waitFor(() => success.mock.calls.length > 0);
    expect(success.mock.calls).toEqual([
      ['Download complete', { description: `Files that were added or changed in the old folder are in ${keptDir}.` }],
    ]);
  });

  it.each<[string, Partial<CatalogBlocker>, Record<string, unknown>, string]>([
    [
      'a link',
      { kind: 'symlink', fileCount: 0 },
      { removed: 'link' },
      `The link ${SLUG} was removed. Its folder was not changed.`,
    ],
    [
      'a link to a file',
      { kind: 'symlink', fileCount: 0, linkToFile: true },
      { removed: 'link' },
      `The link ${SLUG} was removed.`,
    ],
    [
      'a link the app could not remove',
      { kind: 'symlink', fileCount: 0 },
      { keptDir: `/models/.staging/${SLUG}.kept-0d6f` },
      `The link ${SLUG} could not be removed. It is in /models/.staging/${SLUG}.kept-0d6f. Its folder was not changed.`,
    ],
    ['a folder', {}, { removed: 'trash' }, `The old folder ${SLUG} is in the Trash.`],
    ['a file', { kind: 'file' }, { removed: 'trash' }, `The old file ${SLUG} is in the Trash.`],
    [
      'an entry the Trash refused',
      {},
      { keptDir: `/models/.staging/${SLUG}.kept-0d6f` },
      `The old ${SLUG} could not go to the Trash. It is in /models/.staging/${SLUG}.kept-0d6f. Nothing was deleted.`,
    ],
  ])('says where the old entry went when a cleanup of %s is done', async (_name, overrides, done, description) => {
    const success = vi.spyOn(toast, 'success');
    onTestFinished(() => success.mockRestore());
    recordRequests({
      ...catalogRoutes({ blockedByForeignDir: true, blocker: { ...BLOCKER, ...overrides } }, [
        downloadJob({ kind: 'cleanup' }),
      ]),
      '/downloads/job-1': { cancelled: true, id: 'job-1' },
    });
    await mountModels();
    emitDownload('done', { id: 'job-1', outputDir: `/models/${SLUG}`, ...done });
    await waitFor(() => success.mock.calls.length > 0);
    expect(success.mock.calls).toEqual([['Download complete', { description }]]);
  });

  it('titles a failed job by its kind, not by what was in the way', async () => {
    const error = vi.spyOn(toast, 'error');
    onTestFinished(() => error.mockRestore());
    recordRequests({
      ...catalogRoutes({ blockedByForeignDir: true, blocker: HELD }, [downloadJob({ kind: 'cleanup' })]),
      '/downloads/job-1': { cancelled: true, id: 'job-1' },
    });
    await mountModels();
    const message = 'The download failed. Nothing was removed.';
    emitDownload('error', { id: 'job-1', message });
    await waitFor(() => error.mock.calls.length > 0);
    expect(error.mock.calls).toEqual([['Download failed', { description: message }]]);
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
    // The card still says Installed beside the Update button.
    expect(buttonLabels()).toEqual(expect.arrayContaining(['Update', 'Installed']));
    expect(liveButtonLabels()).toContain('Update');
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
    expect(buttonLabels()).not.toContain('Update');
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
    expect(buttonLabels()).toContain('Update');
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
    expect(buttonLabels()).not.toContain('Update');
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
    expect(buttonLabels()).not.toContain('Update');
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
    expect(buttonLabels()).toContain('Update');
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
          located: null,
          listed: false,
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
    expect(buttonLabels()).toContain('Update');
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
    expect(buttonLabels()).not.toContain('Update');
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
    expect(buttonLabels()).not.toContain('Update');
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
    expect(buttonLabels()).not.toContain('Update');
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
    expect(buttonLabels()).not.toContain('Update');
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
      expect(buttonLabels()).not.toContain('Update');
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
    expect(buttonLabels()).toContain('Update');
    expect(liveButtonLabels()).not.toContain('Update');
    expect(paletteLabels).not.toContain(`Update ${LABEL}`);

    held.release();
    await settle();
    expect(liveButtonLabels()).toContain('Update');
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
    expect(buttonLabels()).toContain('Update');
    expect(liveButtonLabels()).not.toContain('Update');

    held.release();
    await settle();
    expect(liveButtonLabels()).toContain('Update');
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
    expect(liveButtonLabels()).toContain('Update');
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
    expect(buttonLabels()).toContain('Update');
    expect(liveButtonLabels()).not.toContain('Update');

    held.release();
    await settle();
    expect(liveButtonLabels()).toContain('Update');
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
      expect(buttonLabels()).toContain('Update');
      expect(liveButtonLabels()).not.toContain('Update');

      held.release();
      await settle();
      expect(liveButtonLabels()).toContain('Update');
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
            return { ok: true as const, status: 202, body: { id: 'job-b', slug: SLUG, repo: REPO, kind: 'install' } };
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
    connectDashboardApi(bindEventTargetPort(port1));
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
      expect(liveButtonLabels()).toContain('Update');
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
    expect(buttonLabels()).not.toContain('Update');
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
      ).toEqual([['Download failed', SLUG]]);
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
            return {
              ok: true as const,
              status: 202,
              body: { id: 'job-local', slug: SLUG, repo: REPO, kind: 'install' },
            };
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
    connectDashboardApi(bindEventTargetPort(port1));
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

  describe('Locate', () => {
    const FOLDER = '/Users/me/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF';
    const LOCATE_PLAN: LocatePlanResponse = {
      slug: SLUG,
      repo: REPO,
      token: 'e5'.repeat(32),
      folder: FOLDER,
      mode: 'shared',
      byRecord: false,
      revision: 'f'.repeat(40),
      use: { count: 1, bytes: 17.6 * 1024 ** 3 },
      read: { count: 1, bytes: 17.6 * 1024 ** 3 },
      download: { count: 5, bytes: 11.4 * 1024 ** 2 },
    };

    /** The native dialog: records each request and answers it with `answer()` when that is not undefined. */
    function desktopDialog(answer: () => string | null | undefined): unknown[] {
      const requests: unknown[] = [];
      const onMessage = (event: MessageEvent): void => {
        const data = event.data as { type?: unknown; requestId?: unknown };
        if (data?.type !== PICK_FOLDER_MESSAGE) return;
        requests.push(data);
        const path = answer();
        if (path === undefined) return;
        window.dispatchEvent(
          new MessageEvent('message', {
            data: { type: FOLDER_PICKED_MESSAGE, requestId: data.requestId, path },
            source: window,
          }),
        );
      };
      window.addEventListener('message', onMessage);
      onTestFinished(() => window.removeEventListener('message', onMessage));
      return requests;
    }

    function locateDialog(): HTMLElement | undefined {
      return [...document.querySelectorAll<HTMLElement>('[role="dialog"]')].find((el) =>
        el.textContent?.includes('Use these files?'),
      );
    }

    function card(): Element {
      return mounted!.container.querySelector(`[data-model="${SLUG}"]`)!;
    }

    function textOf(node: ParentNode | undefined): string {
      return splitWhitespace(node?.textContent ?? '').join(' ');
    }

    it('picks a model folder, checks it, confirms in the words of the plan, then posts the slug and the token', async () => {
      const success = vi.spyOn(toast, 'success');
      onTestFinished(() => success.mockRestore());
      const held = heldPlan({ ok: true, status: 200, body: LOCATE_PLAN });
      const requests = desktopDialog(() => FOLDER);
      const calls = serveJobs(
        catalogRoutes({}),
        () => ({ ok: true, status: 202, body: { id: 'job-locate', slug: SLUG, repo: REPO, kind: 'locate' } }),
        undefined,
        undefined,
        held.answer,
      );
      await mountModels();
      const locate = buttonNamed(card(), 'Locate…');
      await act(async () => locate.click());
      await waitFor(() => posts(calls, '/downloads/locate-plan').length === 1);
      // Mutation: send the Work purpose → MAIN opens the Work dialog → red.
      expect(requests).toEqual([{ type: PICK_FOLDER_MESSAGE, requestId: expect.any(String), purpose: 'model' }]);
      expect(posts(calls, '/downloads/locate-plan')).toEqual([{ slug: SLUG, path: FOLDER }]);
      const checking = buttonNamed(card(), 'Checking…');
      expect(checking.getAttribute('aria-disabled')).toBe('true');
      // A second click while the check runs does nothing.
      await act(async () => checking.click());
      await settle();
      expect(requests).toHaveLength(1);
      expect(posts(calls, '/downloads/locate-plan')).toHaveLength(1);
      expect(buttonNamed(card(), 'Install').disabled).toBe(true);
      expect(locateDialog()).toBeUndefined();

      held.release();
      await waitFor(() => locateDialog() !== undefined);
      const text = textOf(locateDialog());
      expect(text).toContain(
        `The app found 1 file (17.6 GB) of ${LABEL} in ${truncateMiddle(FOLDER, 48)}. So far it checked names and sizes. Next it reads each file once to check its content. This can take a few minutes. Nothing is copied: the models folder shares these files on disk, so they take no extra space. It downloads 5 files (11.4 MB) into the models folder.`,
      );
      expect(text).toContain('Nothing in your folder is changed, moved or deleted.');
      expect(posts(calls, '/downloads')).toEqual([]);

      await act(async () => buttonNamed(locateDialog()!, 'Use files').click());
      await waitFor(() => openedStreams.includes('job-locate'));
      // Mutation: post the repository or drop the token → red.
      expect(posts(calls, '/downloads')).toEqual([{ slug: SLUG, locate: LOCATE_PLAN.token }]);
      await waitFor(() => buttonLabels().includes('Cancel'));
      emitDownload('done', { id: 'job-locate', outputDir: `/models/${SLUG}` });
      await waitFor(() => success.mock.calls.length > 0);
      expect(success.mock.calls).toEqual([['Model located', { description: `Uses the files in ${FOLDER}.` }]]);
    });

    it('does nothing when the dialog is cancelled', async () => {
      const requests = desktopDialog(() => null);
      const calls = serveJobs(catalogRoutes({}), () => ({
        ok: false,
        status: 500,
        code: 'E_INTERNAL',
        message: 'no',
      }));
      await mountModels();
      await act(async () => buttonNamed(card(), 'Locate…').click());
      await waitFor(() => requests.length === 1);
      await settle();
      expect(calls.filter((call) => call.method === 'POST')).toEqual([]);
      expect(liveButtonLabels()).toContain('Locate…');
      expect(card().querySelector('[data-row="build"]')?.textContent).toContain('23.3 GB');
      expect(locateDialog()).toBeUndefined();
    });

    it('says a refusal on the build line of the card, raises no toast and offers Locate again', async () => {
      const error = vi.spyOn(toast, 'error');
      onTestFinished(() => error.mockRestore());
      const refusal = `This folder holds UD-Q4_K_M. Choose UD-Q4_K_M on the card, then use Locate again.`;
      desktopDialog(() => FOLDER);
      let plans = 0;
      const calls = serveJobs(
        catalogRoutes({}),
        () => ({ ok: true, status: 202, body: { id: 'job-locate', slug: SLUG, repo: REPO, kind: 'locate' } }),
        undefined,
        undefined,
        () =>
          ++plans === 1
            ? { ok: false, status: 409, code: 'E_CONFLICT', message: refusal }
            : { ok: true, status: 200, body: LOCATE_PLAN },
      );
      await mountModels();
      const before = card().getBoundingClientRect().height;
      const locate = buttonNamed(card(), 'Locate…');
      locate.focus();
      await act(async () => locate.click());
      await waitFor(() => (card().querySelector('[data-row="build"]')?.textContent ?? '').includes(refusal));
      expect(error).not.toHaveBeenCalled();
      expect(card().querySelector('[role="alert"], [role="status"]')).toBeNull();
      // The sentence is read out, and the line opens it whole.
      expect(card().querySelector('[aria-live="polite"]')?.textContent).toBe(refusal);
      const rows = [...card().querySelectorAll('[data-row]')].map((cell) => cell.getAttribute('data-row'));
      expect(rows).toEqual(['title', 'description', 'build', 'extras', 'action']);
      expect(card().getBoundingClientRect().height).toBe(before);
      expect(document.activeElement).toBe(buttonNamed(card(), 'Locate…'));

      await act(async () => buttonNamed(card(), 'Locate…').click());
      await waitFor(() => locateDialog() !== undefined);
      expect(posts(calls, '/downloads/locate-plan')).toHaveLength(2);
      expect(card().querySelector('[data-row="build"]')?.textContent).not.toContain(refusal);
    });

    it('says the other disk and the install record in the confirmation, and leaves out an empty download', () => {
      const handlers = { onLocate: () => Promise.resolve(), onStarted: () => {} };
      const linked = locateContent('Hy4-flash', { ...LOCATE_PLAN, mode: 'linked' }, handlers);
      const record = locateContent(
        'Qwen3.8-27B UD-Q4_K_XL',
        {
          ...LOCATE_PLAN,
          byRecord: true,
          read: { count: 0, bytes: 0 },
          download: { count: 0, bytes: 0 },
        },
        handlers,
      );
      const words = (node: ReactNode): string =>
        splitWhitespace(
          renderToStaticMarkup(createElement('p', null, node))
            .replace('<p>', '')
            .replace('</p>', ''),
        )
          .join(' ')
          .replaceAll(`<span class="font-medium text-foreground" title="${FOLDER}">`, '')
          .replaceAll('</span>', '');
      expect(words(linked.description)).toBe(
        `The app found 1 file (17.6 GB) of Hy4-flash in ${truncateMiddle(FOLDER, 48)}. So far it checked names and sizes. Next it reads each file once to check its content. This can take a few minutes. Nothing is copied: the app links to these files. The model works only while this disk is connected. It downloads 5 files (11.4 MB) into the models folder.`,
      );
      expect(words(record.description)).toBe(
        `The app found 1 file (17.6 GB) of Qwen3.8-27B UD-Q4_K_XL in ${truncateMiddle(FOLDER, 48)}. This folder has this app&#x27;s install record for Qwen3.8-27B UD-Q4_K_XL. The app trusts the record: it checks that the files are there and does not read them. Nothing is copied: the models folder shares these files on disk, so they take no extra space.`,
      );
      expect(record).toMatchObject({
        title: 'Use these files?',
        notice: 'Nothing in your folder is changed, moved or deleted.',
        confirmLabel: 'Use files',
        workingLabel: 'Starting…',
        successLabel: 'Started',
        tone: 'neutral',
        errorTitle: 'Could not start Locate',
      });
    });

    it('names the folder a located build uses, and asks before an update downloads the whole build', async () => {
      const located = { from: FOLDER, mode: 'shared' as const, filesAvailable: true };
      const calls = serveJobs(
        catalogRoutes({ present: true, installed: true, localRevision: 'a'.repeat(40), located }, [], {
          items: [{ hfRepo: REPO, remoteRevision: 'b'.repeat(40) }],
        } as CatalogUpdatesResponse),
        () => ({ ok: true, status: 202, body: { id: 'job-new', slug: SLUG, repo: REPO, kind: 'install' } }),
      );
      await mountModels();
      expect(card().querySelector('[data-row="build"] p')?.textContent).toBe(
        `Uses files in ${truncateMiddle(FOLDER, 48)}`,
      );
      expect(card().querySelector('[data-row="action"] [data-status="installed"]')).not.toBeNull();
      expect(buttonLabels()).not.toContain('Locate…');
      await act(async () => buttonNamed(card(), 'Update').click());
      const dialog = [...document.querySelectorAll<HTMLElement>('[role="dialog"]')].find((el) =>
        el.textContent?.includes('Download the new version?'),
      );
      // Mutation: start the update on the click → a POST before the confirmation → red.
      expect(posts(calls, '/downloads')).toEqual([]);
      expect(textOf(dialog)).toContain(
        `The app downloads it (about 23.3 GB) into the models folder. The files in ${truncateMiddle(FOLDER, 48)} are not changed, and the app stops using them.`,
      );
      await act(async () => buttonNamed(dialog!, 'Download').click());
      await waitFor(() => openedStreams.includes('job-new'));
      expect(posts(calls, '/downloads')).toEqual([{ slug: SLUG }]);
    });

    it('shows Files not found in the five rows with Check again, Locate… and Download instead', async () => {
      const away = { from: '/Volumes/Models/qwen', mode: 'linked' as const, filesAvailable: false };
      const calls = serveJobs(catalogRoutes({ located: away }), () => ({
        ok: true,
        status: 202,
        body: { id: 'job-instead', slug: SLUG, repo: REPO, kind: 'install' },
      }));
      await mountModels();
      const rows = [...card().querySelectorAll('[data-row]')].map((cell) => cell.getAttribute('data-row'));
      expect(rows).toEqual(['title', 'description', 'build', 'extras', 'action']);
      expect(card().querySelector('[data-row="build"] p')?.textContent).toBe('Files not found · /Volumes/Models/qwen');
      // Mutation: ignore filesAvailable → a bare Install shows → red.
      expect(buttonLabels()).not.toContain('Install');
      const action = card().querySelector('[data-row="action"]')!;
      expect([...action.querySelectorAll('button')].map((button) => button.textContent?.trim())).toEqual([
        'Check again',
        'Locate…',
        'Download instead',
      ]);
      await act(async () => buttonNamed(card(), 'Files not found · /Volumes/Models/qwen').click());
      expect(document.body.textContent).toContain(
        '/Volumes/Models/qwen is not available. Connect the disk, or locate the files again.',
      );
      const catalogGets = calls.filter((call) => call.method === 'GET' && call.path === '/catalog').length;
      await act(async () => buttonNamed(card(), 'Check again').click());
      await waitFor(
        () => calls.filter((call) => call.method === 'GET' && call.path === '/catalog').length > catalogGets,
      );
      expect(posts(calls, '/downloads')).toEqual([]);
      await act(async () => buttonNamed(card(), 'Download instead').click());
      await waitFor(() => openedStreams.includes('job-instead'));
      expect(posts(calls, '/downloads')).toEqual([{ slug: SLUG }]);
    });

    it('offers no Locate on the DFlash2 line', async () => {
      serveJobs(
        catalogRoutes({
          draft: {
            label: 'DFlash2',
            hfRepo: 'z-lab/Qwen3.8-27B-DFlash2',
            sizeGb: 3.85,
            slug: 'qwen3.8-27b-dflash2',
            installed: false,
            present: false,
            blockedByForeignDir: false,
            blocker: null,
            localRevision: null,
          },
        }),
        () => ({ ok: false, status: 500, code: 'E_INTERNAL', message: 'no' }),
      );
      await mountModels();
      const companion = mounted!.container.querySelector('[role="group"][aria-label="DFlash2 add-on"]')!;
      // Mutation: render Locate in the draft line → red.
      expect([...companion.querySelectorAll('button')].map((button) => button.textContent?.trim())).toEqual(['Add']);
      expect(buttonLabels().filter((label) => label === 'Locate…')).toHaveLength(1);
    });
  });

  describe('the card with more than one build', () => {
    const XL_SLUG = 'inkling-2-mini-gguf';
    const M_SLUG = 'inkling-2-mini-gguf-ud-q4_k_m';
    const ORIGINAL_SLUG = 'inkling-2-mini';
    const MODEL = 'Inkling-2-Mini';

    function inklingBuild(fields: Partial<CatalogBuildItem> & Pick<CatalogBuildItem, 'id' | 'slug'>): CatalogBuildItem {
      return {
        hfRepo: 'unsloth/Inkling-2-Mini-GGUF',
        sizeGb: 18.7,
        source: 'Unsloth GGUF',
        included: 'Vision and DSpark draft included',
        installed: false,
        present: false,
        heldBy: null,
        blockedByForeignDir: false,
        blocker: null,
        localRevision: null,
        localAssetsRepo: null,
        localAssetsRevision: null,
        located: null,
        listed: false,
        ...fields,
      };
    }

    /** The routes of a page with Inkling-2-Mini in three builds, each build's state overridden by its folder. */
    function inklingRoutes(
      builds: Partial<Record<string, Partial<CatalogBuildItem>>> = {},
      jobs: DownloadJob[] = [],
      updates: CatalogUpdatesResponse = { items: [] },
    ): Record<string, unknown> {
      const routes = catalogRoutes({}, jobs, updates);
      routes['/catalog'] = {
        items: [
          {
            label: MODEL,
            description: 'Looped MoE with vision and DSpark drafts, 1M context',
            builds: [
              inklingBuild({ id: 'UD-Q4_K_XL', slug: XL_SLUG, ...builds[XL_SLUG] }),
              inklingBuild({
                id: 'UD-Q4_K_M',
                slug: M_SLUG,
                sizeGb: 17.7,
                note: '1 GB smaller',
                ...builds[M_SLUG],
              }),
              inklingBuild({
                id: 'Original',
                slug: ORIGINAL_SLUG,
                hfRepo: 'thinkingmachines/Inkling-2-Mini',
                sizeGb: 20.2,
                source: 'Thinking Machines',
                note: 'NVFP4 + MXFP8',
                ...builds[ORIGINAL_SLUG],
              }),
            ],
          },
        ],
      };
      return routes;
    }

    async function mountInkling(): Promise<HTMLElement> {
      mounted = await renderPage(
        createElement(MemoryRouter, null, createElement(Models), createElement(PaletteProbe)),
        (text) => text.includes(MODEL),
      );
      await settle();
      return mounted.container.querySelector<HTMLElement>(`[data-model="${XL_SLUG}"]`)!;
    }

    function segment(card: Element, build: string): HTMLButtonElement {
      const found = [...card.querySelectorAll<HTMLButtonElement>('[role="group"] button[data-value]')].find(
        (button) => button.getAttribute('data-value') === build,
      );
      expect(found, build).toBeTruthy();
      return found!;
    }

    function row(card: Element, name: string): string {
      return splitWhitespace(card.querySelector(`[data-row="${name}"]`)?.textContent ?? '').join(' ');
    }

    it('shows the first build and names the switch by the model', async () => {
      recordRequests(inklingRoutes());
      const card = await mountInkling();
      expect(card.querySelector('[role="group"]')?.getAttribute('aria-label')).toBe(`Build of ${MODEL}`);
      expect(segment(card, XL_SLUG).getAttribute('aria-pressed')).toBe('true');
      expect(row(card, 'build')).toContain('18.7 GB · Unsloth GGUF');
      expect(row(card, 'extras')).toBe('Vision and DSpark draft included');
      expect(buttonLabels()).toContain('Install UD-Q4_K_XL');
    });

    it('choosing a build starts nothing and changes the link, the build line and the button', async () => {
      const calls = recordRequests(inklingRoutes());
      const card = await mountInkling();
      const before = calls.length;
      await act(async () => segment(card, ORIGINAL_SLUG).click());
      await settle();
      // Mutation: start a job on choose → a POST → red.
      expect(calls.slice(before).filter((call) => call.method !== 'GET')).toEqual([]);
      expect(card.getAttribute('data-build')).toBe(ORIGINAL_SLUG);
      expect(card.querySelector('[data-row="title"] a')?.getAttribute('href')).toBe(
        'https://huggingface.co/thinkingmachines/Inkling-2-Mini',
      );
      expect(row(card, 'build')).toContain('20.2 GB · Thinking Machines · NVFP4 + MXFP8');
      expect(buttonLabels()).toContain('Install Original');
      expect(buttonLabels()).not.toContain('Install UD-Q4_K_XL');
    });

    it('Install posts the chosen build folder', async () => {
      const calls = serveJobs(inklingRoutes(), (body) => ({
        ok: true,
        status: 202,
        body: {
          id: 'job-m',
          slug: (body as { slug: string }).slug,
          repo: 'unsloth/Inkling-2-Mini-GGUF',
          kind: 'install',
        },
      }));
      const card = await mountInkling();
      await act(async () => segment(card, M_SLUG).click());
      await act(async () => buttonNamed(card, 'Install UD-Q4_K_M').click());
      await waitFor(() => openedStreams.includes('job-m'));
      // Mutation: Install posts builds[0].slug → red.
      expect(posts(calls, '/downloads')).toEqual([{ slug: M_SLUG }]);
      expect(segment(card, M_SLUG).querySelector('[data-mark="downloading"]')).not.toBeNull();
    });

    it('marks every build that is not chosen with its state, in words', async () => {
      recordRequests(
        inklingRoutes(
          {
            [XL_SLUG]: { blockedByForeignDir: true, blocker: BLOCKER },
            [M_SLUG]: { present: true, installed: true, localRevision: 'a'.repeat(40) },
          },
          [downloadJob({ id: 'job-original', slug: ORIGINAL_SLUG })],
          { items: [{ hfRepo: 'unsloth/Inkling-2-Mini-GGUF', remoteRevision: 'b'.repeat(40) }] },
        ),
      );
      const card = await mountInkling();
      // The build with a job is the one the card shows first.
      expect(card.getAttribute('data-build')).toBe(ORIGINAL_SLUG);
      // Mutation: mark only the chosen segment → red.
      expect(segment(card, XL_SLUG).textContent).toBe('UD-Q4_K_XLneeds attention');
      expect(segment(card, M_SLUG).textContent).toBe('UD-Q4_K_Mupdate available');
      expect(segment(card, ORIGINAL_SLUG).textContent).toBe('Originaldownloading');
    });

    it('an installed build that is not first is the one the card shows, with its mark', async () => {
      recordRequests(inklingRoutes({ [ORIGINAL_SLUG]: { present: true, installed: true } }));
      const card = await mountInkling();
      expect(card.getAttribute('data-build')).toBe(ORIGINAL_SLUG);
      expect(card.querySelector('[data-status="installed"]')).not.toBeNull();
      await act(async () => segment(card, XL_SLUG).click());
      expect(segment(card, ORIGINAL_SLUG).textContent).toBe('Originalinstalled');
      expect(buttonLabels()).toContain('Install UD-Q4_K_XL');
      // A model counts as installed when one of its builds is.
      expect(mounted!.text()).toContain('1 of 1 installed');
    });

    it('counts a model with two builds on disk once', async () => {
      recordRequests(
        inklingRoutes({
          [M_SLUG]: { present: true, installed: true },
          [ORIGINAL_SLUG]: { present: true, installed: true },
        }),
      );
      await mountInkling();
      // Mutation: count builds on disk in place of models → '2 of 1 installed' → red.
      expect(mounted!.text()).toContain('1 of 1 installed');
    });

    it('keeps the build whose job ended on the card until the catalog says it is on disk', async () => {
      const stale = inklingRoutes({}, [downloadJob({ id: 'job-original', slug: ORIGINAL_SLUG })]);
      const fresh = inklingRoutes({ [ORIGINAL_SLUG]: { present: true, installed: true } });
      const held = deferred(fresh['/catalog']);
      recordRequests({
        ...stale,
        '/catalog': sequence(stale['/catalog'], held.body),
        '/downloads/job-original': { cancelled: true, id: 'job-original' },
      });
      await mountInkling();
      const card = (): Element => mounted!.container.querySelector(`[data-model="${XL_SLUG}"]`)!;
      expect(card().getAttribute('data-build')).toBe(ORIGINAL_SLUG);

      emitDownload('done', { id: 'job-original', outputDir: '/models/x' });
      await settle();
      // Mutation: shownBuild leaves out the settling build → the card jumps to UD-Q4_K_XL with a live Install → red.
      expect(card().getAttribute('data-build')).toBe(ORIGINAL_SLUG);
      expect(liveButtonLabels().filter((label) => label.startsWith('Install'))).toEqual([]);

      held.release();
      await settle();
      expect(card().getAttribute('data-build')).toBe(ORIGINAL_SLUG);
      expect(card().querySelector('[data-status="installed"]')).not.toBeNull();
    });

    it('lists one palette command per build, by folder, and a command chooses its build', async () => {
      const calls = serveJobs(inklingRoutes(), (body) => ({
        ok: true,
        status: 202,
        body: {
          id: 'job-p',
          slug: (body as { slug: string }).slug,
          repo: 'thinkingmachines/Inkling-2-Mini',
          kind: 'install',
        },
      }));
      const card = await mountInkling();
      // Mutation: ids by hfRepo → two ids collide → red.
      expect(paletteIds.filter((id) => id.startsWith('models:install:'))).toEqual([
        `models:install:${XL_SLUG}`,
        `models:install:${M_SLUG}`,
        `models:install:${ORIGINAL_SLUG}`,
      ]);
      expect(paletteLabels).toContain(`Install ${MODEL} (UD-Q4_K_M)`);
      expect(paletteLabels).toContain(`Install ${MODEL} (Original)`);
      await act(async () => paletteCommands.find((command) => command.id === `models:install:${ORIGINAL_SLUG}`)!.run());
      await settle();
      expect(posts(calls, '/downloads')).toEqual([{ slug: ORIGINAL_SLUG }]);
      expect(card.getAttribute('data-build')).toBe(ORIGINAL_SLUG);
    });
  });

  describe('the five rows', () => {
    const ENTRY: CatalogItem = {
      label: LABEL,
      description: 'Best tool use',
      builds: [
        {
          id: 'MXFP4',
          hfRepo: REPO,
          sizeGb: 23.3,
          source: 'MLX',
          slug: SLUG,
          installed: false,
          present: false,
          heldBy: null,
          blockedByForeignDir: false,
          blocker: null,
          localRevision: null,
          localAssetsRepo: null,
          localAssetsRevision: null,
          located: null,
          listed: false,
        },
      ],
    };

    function props(item: Partial<CatalogBuildItem>, fields: Partial<CatalogDownloadProps> = {}): CatalogDownloadProps {
      return {
        ...settlingProps(BLOCKER),
        settling: false,
        item: { ...ENTRY.builds[0], ...item },
        ...fields,
      };
    }

    function view(download: CatalogDownloadProps, extra: Partial<CatalogCardViewProps> = {}) {
      return createElement(
        MemoryRouter,
        null,
        createElement(CatalogCardView, {
          entry: { ...ENTRY, builds: [{ ...ENTRY.builds[0], ...download.item }] },
          chosen: SLUG,
          onChoose: () => {},
          downloads: [download],
          progress: INITIAL_PROGRESS,
          starting: [],
          onStart: () => {},
          host: undefined,
          ...extra,
        }),
      );
    }

    const job = {
      id: 'job-1',
      kind: 'install' as const,
      committing: false,
      cancelling: false,
      connection: 0,
      source: 'local' as const,
      blocker: null,
    };

    it.each<[string, CatalogDownloadProps, string]>([
      ['install', props({}), 'Install'],
      ['job', props({}, { job }), 'Cancel'],
      ['installed', props({ present: true, installed: true }), 'Installed'],
      ['update', props({ present: true, installed: true }, { updateAvailable: true }), 'Update'],
      ['held', props({ heldBy: 'UD-Q4_K_M' }), 'Holds UD-Q4_K_M'],
      ['locked', props({ blockedByForeignDir: true, blocker: { ...HELD, locked: true } }), 'No access'],
      ['adopt', props({ blockedByForeignDir: true, blocker: HELD }), 'Needs update'],
      ['cleanup', props({ blockedByForeignDir: true, blocker: { ...HELD, truncated: true } }), 'Needs cleanup'],
      ['cleanup', props({ blockedByForeignDir: true, blocker: BLOCKER }), 'Needs cleanup'],
      ['cleanup', props({ blockedByForeignDir: true, blocker: HELD }, { refused: true }), 'Needs cleanup'],
    ])('has five rows and no alert in the %s state', async (state, download, action) => {
      expect(catalogActionState(download.item, download)).toBe(state);
      mounted = await renderPage(view(download), (text) => text.includes(LABEL));
      const card = mounted.container.querySelector('[data-model]')!;
      const rows = [...card.querySelectorAll('[data-row]')].map((cell) => cell.getAttribute('data-row'));
      // Mutation: render the old Alert in the action row → a sixth block or an alert → red.
      expect(rows).toEqual(['title', 'description', 'build', 'extras', 'action']);
      // An Alert is role="alert" (danger) or role="status" (warning).
      expect(card.querySelector('[role="alert"], [role="status"]')).toBeNull();
      expect(card.querySelector('[data-row="action"]')?.textContent).toContain(action);
    });

    it.each<[string, CatalogDownloadProps, string, string]>([
      [
        'held',
        props({ heldBy: 'UD-Q4_K_M', listed: true }),
        'Show in Local models',
        `${SLUG} holds UD-Q4_K_M. It is a working model, so this card does not remove it. Delete it in Local models to install MXFP4 here.`,
      ],
      [
        'locked',
        props({ blockedByForeignDir: true, blocker: { ...HELD, locked: true } }),
        'Show why',
        `The app has no write permission on ${SLUG}, so it cannot move it or add files to it. Change its permissions (Finder, Get Info), then open this page again.`,
      ],
      [
        'cleanup',
        props({ blockedByForeignDir: true, blocker: BLOCKER }),
        'Clean up & update…',
        `${SLUG} is in the way of this model.`,
      ],
    ])(
      'gives a keyboard and screen reader user the reason of the %s state beside %s',
      async (_state, download, control, sentence) => {
        mounted = await renderPage(view(download), (text) => text.includes(LABEL));
        const action = mounted.container.querySelector('[data-row="action"]')!;
        expect(action.textContent).toContain(control);
        // Mutation: keep the sentence in the hover Tooltip only → red.
        expect([...action.querySelectorAll('.sr-only')].map((node) => node.textContent)).toContain(sentence);
      },
    );

    it('links Details and Show in Local models from the build, not from the local model list', async () => {
      mounted = await renderPage(view(props({ present: true, installed: true, listed: true })), (text) =>
        text.includes(LABEL),
      );
      expect(mounted.container.querySelector('[data-row="action"] a')?.getAttribute('href')).toBe(
        `/models?focus=${SLUG}`,
      );
      mounted.unmount();
      mounted = await renderPage(view(props({ present: true, installed: true })), (text) => text.includes(LABEL));
      expect(mounted.container.querySelector('[data-row="action"] a')).toBeNull();
      mounted.unmount();
      mounted = await renderPage(view(props({ heldBy: 'UD-Q4_K_M' })), (text) => text.includes(LABEL));
      expect(liveButtonLabels()).toContain('Show why');
    });

    it('marks the installed text installed when an update is available too', async () => {
      mounted = await renderPage(view(props({ present: true, installed: true }, { updateAvailable: true })), (text) =>
        text.includes(LABEL),
      );
      // Mutation: data-status names the action state ('update') → red.
      expect(mounted.container.querySelector('[data-row="action"] [data-status="installed"]')).not.toBeNull();
    });

    it('takes the first match: job, update, installed, missing, held, locked, adopt, cleanup, install', () => {
      const all = {
        present: true,
        heldBy: 'UD-Q4_K_M',
        blocker: HELD,
        located: null,
      };
      expect(catalogActionState(all, { job, updateAvailable: true })).toBe('job');
      expect(catalogActionState(all, { job: undefined, updateAvailable: true })).toBe('update');
      expect(catalogActionState(all, { job: undefined, updateAvailable: false })).toBe('installed');
      const away = { from: '/Volumes/Models/qwen', mode: 'linked' as const, filesAvailable: false };
      expect(catalogActionState({ ...all, located: away }, { job: undefined, updateAvailable: false })).toBe(
        'installed',
      );
      expect(
        catalogActionState({ ...all, present: false, located: away }, { job: undefined, updateAvailable: true }),
      ).toBe('missing');
      const absent = { ...all, present: false };
      expect(catalogActionState(absent, { job: undefined, updateAvailable: true })).toBe('held');
      const free = { ...absent, heldBy: null };
      expect(
        catalogActionState({ ...free, blocker: { ...HELD, locked: true } }, { job: undefined, updateAvailable: false }),
      ).toBe('locked');
      expect(catalogActionState(free, { job: undefined, updateAvailable: false })).toBe('adopt');
      expect(catalogActionState(free, { job: undefined, updateAvailable: false, refused: true })).toBe('cleanup');
      expect(
        catalogActionState(
          { ...free, blocker: { ...HELD, kind: 'symlink' } },
          { job: undefined, updateAvailable: false },
        ),
      ).toBe('cleanup');
      expect(catalogActionState({ ...free, blocker: BLOCKER }, { job: undefined, updateAvailable: false })).toBe(
        'cleanup',
      );
      expect(
        catalogActionState({ ...absent, heldBy: null, blocker: null }, { job: undefined, updateAvailable: false }),
      ).toBe('install');
    });

    function control(view: LocateControl['view'] = { step: 'idle' }): LocateControl & { starts: number } {
      const made = { view, starts: 0, start: () => (made.starts += 1), close: () => {} };
      return made;
    }

    it('says Starting download… while the request is in flight, with Locate… off', async () => {
      const download = props({});
      mounted = await renderPage(view(download, { starting: [SLUG], locate: control() }), (text) =>
        text.includes(LABEL),
      );
      expect(buttonNamed(mounted.container, 'Starting download…').disabled).toBe(true);
      expect(buttonNamed(mounted.container, 'Locate…').disabled).toBe(true);
    });

    it('renders Locate… only with the prop, beside Install and in Files not found', async () => {
      const locate = control();
      mounted = await renderPage(view(props({})), (text) => text.includes(LABEL));
      expect(liveButtonLabels()).not.toContain('Locate…');
      mounted.unmount();
      mounted = await renderPage(view(props({}), { locate }), (text) => text.includes(LABEL));
      await act(async () => buttonNamed(mounted!.container, 'Locate…').click());
      expect(locate.starts).toBe(1);
      const away = { from: '/Volumes/Models/qwen', mode: 'linked' as const, filesAvailable: false };
      for (const [state, item, fields] of [
        ['installed', { present: true, installed: true }, {}],
        ['update', { present: true, installed: true }, { updateAvailable: true }],
        ['job', {}, { job }],
        ['held', { heldBy: 'UD-Q4_K_M' }, {}],
        ['adopt', { blockedByForeignDir: true, blocker: HELD }, {}],
        ['locked', { blockedByForeignDir: true, blocker: { ...HELD, locked: true } }, {}],
        ['cleanup', { blockedByForeignDir: true, blocker: BLOCKER }, {}],
        ['missing', { located: away }, {}],
      ] as const) {
        mounted.unmount();
        const download = props(item, fields);
        expect(catalogActionState(download.item, download)).toBe(state);
        mounted = await renderPage(view(download, { locate }), (text) => text.includes(LABEL));
        // Mutation: render Locate in every state → red.
        expect([state, buttonLabels().includes('Locate…')]).toEqual([state, state === 'missing']);
      }
    });

    it('says Checking… while the folder is checked, and a refusal on the build line', async () => {
      mounted = await renderPage(view(props({}), { locate: control({ step: 'plan' }) }), (text) =>
        text.includes(LABEL),
      );
      const checking = buttonNamed(mounted.container, 'Checking…');
      expect(checking.getAttribute('aria-disabled')).toBe('true');
      expect(buttonNamed(mounted.container, 'Install').disabled).toBe(true);
      mounted.unmount();
      const message = 'The app cannot read this folder. Pick the folder that holds the model files.';
      mounted = await renderPage(view(props({}), { locate: control({ step: 'refused', message }) }), (text) =>
        text.includes(LABEL),
      );
      expect(mounted.container.querySelector('[data-row="build"] button')?.textContent).toBe(message);
      expect(mounted.container.querySelector('[role="alert"], [role="status"]')).toBeNull();
    });

    it('says a disk that is too small on the build line, with the full sentence for the tooltip', async () => {
      const host = { totalMemBytes: 128 * 1024 ** 3, diskFreeBytes: 10 * 1024 ** 3 };
      mounted = await renderPage(view(props({}), { host }), (text) => text.includes(LABEL));
      expect(mounted.container.querySelector('[data-row="build"]')?.textContent).toContain(
        '23.3 GB · only 10.0 GB free on this disk',
      );
      expect(
        hostLine({ sizeGb: 23.3, present: false }, undefined, { totalMemBytes: 16 * 1024 ** 3, diskFreeBytes: null }),
      ).toBe("23.3 GB · more than this Mac's 16.0 GB");
    });

    it('starts the draft from Add with the draft folder, beside a model with no build on disk', async () => {
      const calls = serveJobs(catalogRoutes({ draft: draftItem(false) }), (body) => ({
        ok: true,
        status: 202,
        body: {
          id: 'job-draft',
          slug: (body as { slug: string }).slug,
          repo: 'z-lab/Qwen3.8-27B-DFlash2',
          kind: 'install',
        },
      }));
      await mountModels();
      const companion = mounted!.container.querySelector('[role="group"][aria-label="DFlash2 add-on"]')!;
      await act(async () => buttonNamed(companion, 'Add').click());
      await waitFor(() => openedStreams.includes('job-draft'));
      expect(posts(calls, '/downloads')).toEqual([{ slug: 'qwen3.8-27b-dflash2' }]);
      // The model keeps its own Install beside the draft's job.
      expect(liveButtonLabels()).toContain('Install');
    });

    it('says the draft waits for the model when no build of it is on disk', async () => {
      await mount(
        createElement(MemoryRouter, null, createElement(Models)),
        catalogRoutes({ draft: draftItem(true) }),
        LABEL,
      );
      const companion = mounted!.container.querySelector('[role="group"][aria-label="DFlash2 add-on"]')!;
      expect(companion.textContent).toContain('added · waits for the model');
    });
  });
});

describe('Models page — local models and the drawer', () => {
  const NAME = 'qwen3.8-27b-gguf';
  const SERVED = 'Qwen3.8-27B-UD-Q4_K_XL';

  function modelsRoutes(
    extra: Record<string, unknown> = {},
    located: ModelsResponse['models'][number]['located'] = null,
  ): Record<string, unknown> {
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
          located,
        },
      ],
      warnings: [],
      dir: '/models',
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

  describe('a model that Locate made', () => {
    const FROM = '/Volumes/Models/lmstudio/unsloth/Qwen3.8-27B-GGUF';

    function dialogWith(title: string): HTMLElement | undefined {
      return [...document.querySelectorAll<HTMLElement>('[role="dialog"]')].find((el) =>
        el.textContent?.includes(title),
      );
    }

    it.each<[string, ModelsResponse['models'][number]['located'], string | null]>([
      ['linked', { from: FROM, mode: 'linked', filesAvailable: true, filesStay: true }, 'Linked'],
      ['away', { from: FROM, mode: 'linked', filesAvailable: false, filesStay: true }, 'Not connected'],
      ['shared', { from: FROM, mode: 'shared', filesAvailable: true, filesStay: true }, null],
    ])('badges the row of a %s model', async (_name, located, badge) => {
      await mountModels(modelsRoutes({}, located));
      const cell = modelRow().querySelector('td')!;
      const badges = [...cell.querySelectorAll('[title]')].filter((node) => node.getAttribute('title') === FROM);
      expect(badges.map((node) => node.textContent)).toEqual(badge === null ? [] : [badge]);
    });

    it('says Remove when the model files stay on disk, and the delete calls the same route', async () => {
      const success = vi.spyOn(toast, 'success');
      const located = { from: FROM, mode: 'shared' as const, filesAvailable: true, filesStay: true };
      const routes = modelsRoutes({ [`/models/${NAME}`]: { deleted: true, name: NAME } }, located);
      routes['/models'] = sequence(routes['/models'], { ...(routes['/models'] as ModelsResponse), models: [] });
      await mountModels(routes);
      expect(modelRow().querySelector(`button[aria-label="Delete ${NAME}"]`)).toBeNull();
      await act(async () =>
        modelRow().querySelector<HTMLButtonElement>(`button[aria-label="Remove ${NAME}"]`)!.click(),
      );
      const dialog = dialogWith('Remove model?')!;
      expect(splitWhitespace(dialog.textContent ?? '').join(' ')).toContain(
        `This removes ${NAME} from the app. Its model files stay on disk; you located them in ${truncateMiddle(FROM, 48)}.`,
      );
      expect(dialog.textContent).not.toContain('permanently');
      await act(async () => buttonIn(dialog, 'Remove').click());
      await waitFor(() => success.mock.calls.length > 0);
      expect(success).toHaveBeenCalledWith('Model removed', { description: NAME });
      expect(calls.filter((call) => call.method === 'DELETE').map((call) => call.path)).toEqual([
        `/api/models/${NAME}`,
      ]);
    });

    it('says the danger Delete when the folder holds the last name of a model file', async () => {
      const located = { from: FROM, mode: 'shared' as const, filesAvailable: true, filesStay: false };
      await mountModels(modelsRoutes({}, located));
      // Mutation: read `located` alone, not `filesStay` → the Remove text over the last copy → red.
      await act(async () =>
        modelRow().querySelector<HTMLButtonElement>(`button[aria-label="Delete ${NAME}"]`)!.click(),
      );
      expect(popover().textContent).toContain('permanently removes');
      expect(dialogWith('Remove model?')).toBeUndefined();
    });

    const shown = truncateMiddle(FROM, 42);
    it.each<[string, 'shared' | 'linked', boolean, string]>([
      ['shared', 'shared', true, `Shares files with${shown}`],
      ['linked', 'linked', true, `Linked to${shown}`],
      ['away', 'linked', false, `Linked to${shown} · not available`],
    ])('names the located folder in the drawer of a %s model', async (_name, mode, filesAvailable, row) => {
      await mountModels(
        modelsRoutes({}, { from: FROM, mode, filesAvailable, filesStay: true }),
        `/models?focus=${NAME}`,
      );
      await waitFor(() => byRole(document, 'dialog', NAME) !== null);
      const disk = byRole(document, 'region', 'On disk') ?? byRole(document, 'dialog', NAME)!;
      expect(splitWhitespace(disk.textContent ?? '').join(' ')).toContain(splitWhitespace(row).join(' '));
      expect(buttonIn(byRole(document, 'dialog', NAME)!, 'Remove model')).toBeTruthy();
    });
  });

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

describe('the gallery card forms', () => {
  it('pairs every "Needs cleanup" with Clean up & update…, and no form says Reinstall', async () => {
    const { CatalogCardSection } = await import('../scripts/dev/gallery/catalog-sections.js');
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(CatalogCardSection)), (text) =>
      text.includes('Needs cleanup'),
    );
    const forms = [...mounted.container.querySelectorAll<HTMLElement>('[data-gallery-card]')];
    let cleanups = 0;
    for (const form of forms) {
      const name = form.getAttribute('data-gallery-card');
      const text = form.textContent ?? '';
      expect([name, text.includes('Reinstall')]).toEqual([name, false]);
      for (const row of form.querySelectorAll<HTMLElement>('[data-row="action"], [role="group"]')) {
        if (!(row.textContent ?? '').toLowerCase().includes('needs cleanup')) continue;
        cleanups += 1;
        const labels = [...row.querySelectorAll('button')].map((button) => (button.textContent ?? '').trim());
        expect([name, labels.includes('Clean up & update…'), labels.includes('Show why')]).toEqual([name, true, false]);
      }
    }
    expect(cleanups).toBeGreaterThanOrEqual(8);
  });
});
