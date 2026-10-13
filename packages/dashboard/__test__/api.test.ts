/**
 * The behavioural gate for the whole dashboard API.
 *
 * Every assertion here drives `runtime.call` — the same entry point the Electron
 * MessagePort bridge feeds — so it tests API behaviour and not transport
 * plumbing. The one block that deliberately goes through the real RPC transport
 * is at the bottom: it holds the port bridge to the identical error-model
 * contract end to end, over a real structured clone.
 */

import {
  chmodSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  realpathSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MessageChannel } from 'node:worker_threads';

import { canonicalCacheRoot, coldTierRestoreFamilyList } from '@mlx-node/agent/catalog';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import type { ApiContext } from '../src/api/context.js';
import { dispatch, isApiPath } from '../src/api/dispatch.js';
import { CodingAgentsService } from '../src/coding-agents.js';
import { DownloadsClosedError } from '../src/download.js';
import { TRACE_RETENTION_DAYS } from '../src/ingest/traces.js';
import { createRpcClient } from '../src/rpc/client.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import { createDashboardRuntime, type DashboardRuntime } from '../src/runtime.js';
import { createTestClient, type TestClient } from './helpers/api-client.js';

const FIXTURE_TRACES = fileURLToPath(new URL('./fixtures/traces', import.meta.url));

const MODEL_CONFIG = JSON.stringify({ model_type: 'qwen3', max_position_embeddings: 40960 });

function ggufHeader(architecture: string): Buffer {
  const string = (text: string): Buffer => {
    const bytes = Buffer.from(text);
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(bytes.length));
    return Buffer.concat([length, bytes]);
  };
  const header = Buffer.alloc(24);
  header.write('GGUF');
  header.writeUInt32LE(3, 4);
  header.writeBigUInt64LE(1n, 16);
  const stringType = Buffer.alloc(4);
  stringType.writeUInt32LE(8);
  return Buffer.concat([header, string('general.architecture'), stringType, string(architecture)]);
}

/** 64-char lowercase-hex cold-cache block filename. */
function hexBlock(index: number): string {
  return `${index.toString(16).padStart(64, '0')}.safetensors`;
}

let base: string;
let tracesDir: string;
let modelsDir: string;
let cacheRoot: string;
let runtime: DashboardRuntime;
let api: TestClient;

/** Run a full incremental ingest and wait for it to land in the index. */
async function ingest(): Promise<void> {
  const res = await api.fetch('/api/ingest', { method: 'POST' });
  expect(res.status).toBe(200);
}

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'dash-api-'));
  tracesDir = join(base, 'traces');
  cpSync(FIXTURE_TRACES, tracesDir, { recursive: true });

  modelsDir = join(base, 'models');
  mkdirSync(join(modelsDir, 'model-a'), { recursive: true });
  writeFileSync(join(modelsDir, 'model-a', 'config.json'), MODEL_CONFIG);
  writeFileSync(join(modelsDir, 'model-a', 'model.safetensors'), Buffer.alloc(2048));

  cacheRoot = join(base, 'cache');
  mkdirSync(cacheRoot, { recursive: true });
  writeFileSync(join(cacheRoot, hexBlock(1)), Buffer.alloc(100));
  writeFileSync(join(cacheRoot, hexBlock(2)), Buffer.alloc(200));

  runtime = createDashboardRuntime({
    dbPath: ':memory:',
    tracesDir,
    modelsDir,
    cacheRoot,
  });
  api = createTestClient(runtime);
});

afterEach(async () => {
  await runtime.close();
  rmSync(base, { recursive: true, force: true });
});

describe('dashboard api — models & catalog', () => {
  it.each(
    ['qwen35', 'gemma4', 'muse-glimmer'].flatMap((family) =>
      ['top-level', 'nested'].map((layout) => ({ family, layout })),
    ),
  )('keeps the persisted $layout $family GGUF default available for setup', async ({ family, layout }) => {
    rmSync(join(modelsDir, 'model-a'), { recursive: true });
    const repo = layout === 'top-level' ? modelsDir : join(modelsDir, 'downloaded-repo');
    mkdirSync(repo, { recursive: true });
    if (family !== 'qwen35' || layout === 'nested') {
      writeFileSync(
        join(repo, 'config.json'),
        JSON.stringify({ model_type: family === 'qwen35' ? 'qwen3_5' : family.replace('-', '_') }),
      );
      writeFileSync(join(repo, 'tokenizer.json'), '{}');
    }
    const names = [`${family}-Q4_K_XL`, `${family}-Q6_K_XL`];
    for (const name of [...names, `${family}-mmproj-Q4_K_XL`, `${family}-draft-Q4_K_XL`]) {
      writeFileSync(join(repo, `${name}.gguf`), ggufHeader(family));
    }
    const listModels = async (): Promise<string[]> => {
      const response = await api.fetch('/api/coding-agents/models');
      expect(response.status).toBe(200);
      const body = (await response.json()) as { models: { name: string }[] };
      return body.models.map((model) => model.name);
    };
    expect(await listModels()).toEqual(names);
    const home = join(base, 'home');
    mkdirSync(join(home, '.mlx-agent'), { recursive: true });
    writeFileSync(join(home, '.mlx-agent', 'config.toml'), `[models]\ndefault = "${names[1]}"\n`);
    const service = new CodingAgentsService({
      prepareCommand: async () => join(home, '.mlx-node', 'bin', 'mlx'),
      home,
      env: {},
      listModels,
      connect: async () => {
        throw new Error('Discovery must not start inference');
      },
    });
    try {
      expect(await service.state()).toMatchObject({ model: names[1], available: true, unavailableReason: null });
      expect(await service.start('detect')).toMatchObject({ available: true });
    } finally {
      await service.close();
    }
  });

  it.each(['qwen3_5', 'muse_glimmer'])('keeps a %s weights.safetensors default available for setup', async (family) => {
    const name = 'native-saved-model';
    const dir = join(modelsDir, name);
    mkdirSync(dir);
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: family }));
    writeFileSync(join(dir, 'tokenizer.json'), '{}');
    writeFileSync(join(dir, 'weights.safetensors'), Buffer.alloc(128));
    const listModels = async (): Promise<string[]> => {
      const response = await api.fetch('/api/coding-agents/models');
      expect(response.status).toBe(200);
      const body = (await response.json()) as { models: { name: string }[] };
      return body.models.map((model) => model.name);
    };
    expect(await listModels()).toContain(name);
    const home = join(base, 'native-home');
    mkdirSync(join(home, '.mlx-agent'), { recursive: true });
    writeFileSync(join(home, '.mlx-agent', 'config.toml'), `[models]\ndefault = "${name}"\n`);
    const service = new CodingAgentsService({
      home,
      env: {},
      listModels,
      prepareCommand: async () => join(home, '.mlx-node', 'bin', 'mlx'),
      connect: async () => {
        throw new Error('Discovery must not start inference');
      },
    });
    try {
      expect(await service.state()).toMatchObject({ model: name, available: true, unavailableReason: null });
      expect(await service.start('detect')).toMatchObject({ available: true });
    } finally {
      await service.close();
    }
  });

  it('only offers complete generative models for local prompt detection', async () => {
    for (const [name, type, weights] of [
      ['partial', 'qwen3', false],
      ['embedding', 'harrier', true],
      ['unknown', 'unsupported-type', true],
    ] as const) {
      mkdirSync(join(modelsDir, name));
      writeFileSync(join(modelsDir, name, 'config.json'), JSON.stringify({ model_type: type }));
      if (weights) writeFileSync(join(modelsDir, name, 'model.safetensors'), Buffer.alloc(2048));
    }
    const response = await api.fetch('/api/coding-agents/models');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ models: [{ name: 'model-a' }] });
  });

  it('returns draft weights separately and allows their deletion without deleting the target', async () => {
    const name = 'qwen3.8-27b-dflash2';
    const config = JSON.stringify({ model_type: 'qwen3', architectures: ['DFlash2DraftModel'] });
    mkdirSync(join(modelsDir, name));
    writeFileSync(join(modelsDir, name, 'config.json'), config);
    writeFileSync(join(modelsDir, name, 'model.safetensors'), Buffer.alloc(128));
    const res = await api.fetch('/api/models');
    const body = (await res.json()) as {
      models: Array<{ name: string }>;
      companions: Array<{ name: string; sizeBytes: number }>;
    };
    expect(body.models.map((model) => model.name)).toEqual(['model-a']);
    expect(body.companions).toMatchObject([{ name, sizeBytes: Buffer.byteLength(config) + 128 }]);
    const removed = await api.fetch(`/api/models/${name}`, { method: 'DELETE' });
    expect(removed.status).toBe(200);
    expect(existsSync(join(modelsDir, name))).toBe(false);
    expect(existsSync(join(modelsDir, 'model-a', 'model.safetensors'))).toBe(true);
    const after = (await (await api.fetch('/api/models')).json()) as typeof body;
    expect(after.models).toHaveLength(1);
    expect(after.companions).toEqual([]);
  });

  it('lists local models', async () => {
    const res = await api.fetch('/api/models');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { models: Array<{ name: string; modelType: string }>; dir: string };
    expect(body.models.map((m) => m.name)).toContain('model-a');
    expect(body.models.find((m) => m.name === 'model-a')?.modelType).toBe('qwen3');
    // The directory is configurable, so the response names where it looked.
    expect(body.dir).toBe(modelsDir);
  });

  // The delete route distinguishes an unknown name from every other refusal ONLY
  // by sniffing `/not found/i` on the store's error message: an unknown checkpoint
  // is a 404, a refused-but-present one (reserved dir, symlink, not a checkpoint,
  // escaping path) is a 400. Nothing else in the store carries that distinction,
  // so the sniff is load-bearing.
  it('deletes a local model, 404s an unknown name and 400s a refused one', async () => {
    mkdirSync(join(modelsDir, 'not-a-checkpoint'), { recursive: true });

    const ok = await api.fetch('/api/models/model-a', { method: 'DELETE' });
    expect(ok.status).toBe(200);
    expect(await ok.json()).toEqual({ deleted: true, name: 'model-a' });
    expect(existsSync(join(modelsDir, 'model-a'))).toBe(false);

    const unknown = await api.fetch('/api/models/ghost-model', { method: 'DELETE' });
    expect(unknown.status).toBe(404);
    expect(((await unknown.json()) as { error: string }).error).toMatch(/not found/i);

    // Present on disk but not a checkpoint → a refusal, not a 404.
    const refused = await api.fetch('/api/models/not-a-checkpoint', { method: 'DELETE' });
    expect(refused.status).toBe(400);
    expect(((await refused.json()) as { error: string }).error).toMatch(/not a model checkpoint/i);
    expect(existsSync(join(modelsDir, 'not-a-checkpoint'))).toBe(true);
  });

  it('serves the catalog with install state', async () => {
    const res = await api.fetch('/api/catalog');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { items: Array<{ slug: string; installed: boolean; hfRepo: string }> };
    expect(body.items.length).toBeGreaterThan(0);
    for (const item of body.items) expect(typeof item.installed).toBe('boolean');
  });
});

describe('dashboard api — ingest fault isolation', () => {
  // chmod 000 is a no-op for root, so the unreadable root would still list — skip.
  const canTestUnreadable = (process.getuid?.() ?? 0) !== 0;

  (canTestUnreadable ? it : it.skip)('degrades an unreadable trace root to a warning, not a failure', async () => {
    // The trace ingest must degrade to a warning instead of throwing: a throw out
    // of `doIngest` would report the whole pass as failed while the rows stay
    // stale, and an empty listing read as "every source file gone" would delete
    // the index on a fault that says nothing about the files.
    await ingest();

    chmodSync(tracesDir, 0o000);
    try {
      const res = await api.fetch('/api/ingest', { method: 'POST' });
      expect(res.status).toBe(200);
      const body = (await res.json()) as {
        traces: { files: number; records: number; pruned: number; warnings: string[] };
      };
      expect(body.traces.warnings.some((w) => w.includes('scan skipped'))).toBe(true);

      // An unreadable root is unknown, not empty: the indexed traces survive it.
      const overview = (await (await api.fetch('/api/metrics/overview')).json()) as {
        totals: { traces: number };
      };
      expect(overview.totals.traces).toBe(3);
    } finally {
      chmodSync(tracesDir, 0o755); // restore so afterEach cleanup can remove it
    }
  });
});

describe('dashboard api — metrics overview', () => {
  it('returns aggregate arrays', async () => {
    await ingest();
    const res = await api.fetch('/api/metrics/overview');
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      tokensByDay: unknown[];
      throughputByModel: unknown[];
      modelShare: unknown[];
      totals: { turns: number; outputTokens: number };
    };
    expect(Array.isArray(body.tokensByDay)).toBe(true);
    expect(body.tokensByDay.length).toBeGreaterThan(0);
    expect(Array.isArray(body.throughputByModel)).toBe(true);
    expect(Array.isArray(body.modelShare)).toBe(true);
    expect(body.totals.turns).toBeGreaterThan(0);
    expect(body.totals.outputTokens).toBeGreaterThan(0);
  });

  // Finding 9-query: the overview must expose a day-bucketed throughput/TTFT trend
  // per model — one row per (model, day) with that day's averages — not just the
  // single range-wide average per model in `throughputByModel`.
  it('returns a per-model, per-day throughput trend, not one range-wide average', async () => {
    const day1 = Date.parse('2026-07-01T12:00:00.000Z');
    const day2 = Date.parse('2026-07-02T12:00:00.000Z');
    writeFileSync(
      join(tracesDir, '2026-07-02-trend.jsonl'),
      `${JSON.stringify({ v: 1, traceId: 'trend-d1', ts: day1, model: 'trend-model', decodeTps: 100, prefillTps: 500, ttftMs: 20 })}\n${JSON.stringify({ v: 1, traceId: 'trend-d2', ts: day2, model: 'trend-model', decodeTps: 200, prefillTps: 700, ttftMs: 40 })}\n`,
    );
    await ingest();

    const body = (await (await api.fetch('/api/metrics/overview')).json()) as {
      throughputTrend: Array<{
        model: string;
        day: string;
        decodeTps: number;
        prefillTps: number;
        ttftMs: number;
        samples: number;
      }>;
    };
    const rows = body.throughputTrend.filter((r) => r.model === 'trend-model');
    // One row per day (not a single averaged 150), ordered by day.
    expect(rows).toHaveLength(2);
    expect(rows[0].day).toBe('2026-07-01');
    expect(rows[1].day).toBe('2026-07-02');
    const d1 = rows.find((r) => r.day === '2026-07-01');
    const d2 = rows.find((r) => r.day === '2026-07-02');
    expect(d1?.decodeTps).toBe(100);
    expect(d1?.prefillTps).toBe(500);
    expect(d1?.ttftMs).toBe(20);
    expect(d1?.samples).toBe(1);
    expect(d2?.decodeTps).toBe(200);
    expect(d2?.samples).toBe(1);
  });

  // Finding F4: the trace perf columns (ttft_ms/prefill_tps/decode_tps) are
  // nullable and legitimately null for a valid trace recorded without timing.
  // A bucket whose traces carry no perf → AVG over an all-NULL group is NULL and
  // MUST surface as `null`, not a fabricated `0` (which would plot a fake
  // `0 tok/s` point and defeat the UI's `Number.isFinite` empty-state guard). A
  // sibling bucket that DOES carry perf must be unchanged (finite values).
  it('reports null (not 0) throughput for a bucket whose traces carry no perf', async () => {
    const day = Date.parse('2026-07-02T12:00:00.000Z');
    writeFileSync(
      join(tracesDir, '2026-07-02-nullperf.jsonl'),
      `${JSON.stringify({ v: 1, traceId: 'np-perf', ts: day, model: 'perf-model', decodeTps: 150, prefillTps: 600, ttftMs: 30 })}\n${JSON.stringify({ v: 1, traceId: 'np-null', ts: day, model: 'nullperf-model' })}\n`,
    );
    await ingest();

    const body = (await (await api.fetch('/api/metrics/overview')).json()) as {
      throughputTrend: Array<{
        model: string;
        day: string;
        decodeTps: number | null;
        prefillTps: number | null;
        ttftMs: number | null;
        samples: number;
      }>;
    };

    // The bucket WITH perf keeps its finite averages, unchanged.
    const perf = body.throughputTrend.find((r) => r.model === 'perf-model');
    expect(perf?.decodeTps).toBe(150);
    expect(perf?.prefillTps).toBe(600);
    expect(perf?.ttftMs).toBe(30);
    expect(perf?.samples).toBe(1);

    // The bucket with only null-perf traces: COUNT(*) still counts the row (the
    // bucket exists), but every AVG is NULL → surfaced as `null`, never `0`.
    const nullPerf = body.throughputTrend.find((r) => r.model === 'nullperf-model');
    expect(nullPerf).toBeDefined();
    expect(nullPerf?.samples).toBe(1);
    expect(nullPerf?.decodeTps).toBeNull();
    expect(nullPerf?.prefillTps).toBeNull();
    expect(nullPerf?.ttftMs).toBeNull();
    // Guard against the old `?? 0` regression re-appearing.
    expect(nullPerf?.decodeTps).not.toBe(0);
    expect(nullPerf?.prefillTps).not.toBe(0);
    expect(nullPerf?.ttftMs).not.toBe(0);
  });
});

describe('dashboard api — cache', () => {
  it('scans and clears the cold cache with an explicit {all:true}', async () => {
    const res = await api.fetch('/api/cache');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { disk: { entryCount: number; totalBytes: number } };
    expect(body.disk.entryCount).toBe(2);
    expect(body.disk.totalBytes).toBe(300);

    const del = await api.fetch('/api/cache', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ all: true }),
    });
    expect(del.status).toBe(200);
    const cleared = (await del.json()) as { removed: number; freedBytes: number };
    expect(cleared.removed).toBe(2);
    expect(cleared.freedBytes).toBe(300);

    const after = (await (await api.fetch('/api/cache')).json()) as { disk: { entryCount: number } };
    expect(after.disk.entryCount).toBe(0);
  });

  // Finding I: an ambiguous body must 400, never fall through to a whole wipe.
  it('rejects an ambiguous clear body instead of wiping the whole cache', async () => {
    const bodies: Array<string | undefined> = [
      undefined,
      '{}',
      JSON.stringify({ olderThanDays: '7' }),
      JSON.stringify({ olderThanDays: 0 }),
      JSON.stringify({ olderThanDays: -1 }),
    ];
    for (const b of bodies) {
      const del = await api.fetch('/api/cache', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        ...(b === undefined ? {} : { body: b }),
      });
      expect(del.status).toBe(400);
      // Nothing was cleared — both blocks survive.
      const after = (await (await api.fetch('/api/cache')).json()) as { disk: { entryCount: number } };
      expect(after.disk.entryCount).toBe(2);
    }
  });

  it('evicts only blocks older than a positive olderThanDays', async () => {
    // Age one block past the 7-day cutoff; the other stays recent.
    const oldBlock = join(cacheRoot, hexBlock(1));
    const oldSec = (Date.now() - 10 * 86_400_000) / 1000;
    utimesSync(oldBlock, oldSec, oldSec);

    const del = await api.fetch('/api/cache', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ olderThanDays: 7 }),
    });
    expect(del.status).toBe(200);
    const evicted = (await del.json()) as { removed: number; freedBytes: number };
    expect(evicted.removed).toBe(1);
    expect(evicted.freedBytes).toBe(100);

    const after = (await (await api.fetch('/api/cache')).json()) as { disk: { entryCount: number } };
    expect(after.disk.entryCount).toBe(1);
  });
});

/**
 * F1 — the Cache page's hit rate and trend must describe THE CACHE IT IS
 * SHOWING.
 *
 * Before the root filter, `disk` was a filesystem scan of one root while
 * `trend` was an unfiltered `SUM()` over every trace row ever ingested: a
 * dashboard pointed at an empty cache dir reported `exists: false,
 * entryCount: 0` beside a hit rate summed across every cache directory the
 * machine had ever used. Nothing in the suite could have caught it — every
 * existing /api/cache assertion read `body.disk.*` and none had ever touched
 * `trend`.
 *
 * These fixtures deliberately make the WRONG answer much larger than the right
 * one, so a regression cannot hide inside a plausible-looking number.
 */
describe('dashboard api — cache trend scoping', () => {
  /** Canonical spelling of the fixture cache root, resolved independently of the
   *  implementation (`/var/folders/...` → `/private/var/folders/...` on macOS). */
  let canonicalCacheRootPath: string;

  /** Append one trace JSONL line shaped like `MetricsTrace.record()` output. */
  function writeTraceFile(name: string, records: Array<Record<string, unknown>>): void {
    const lines = records.map((r) => JSON.stringify({ v: 1, model: 'm', durationMs: 1, finishReason: 'stop', ...r }));
    writeFileSync(join(tracesDir, name), `${lines.join('\n')}\n`);
  }

  interface CacheBody {
    disk: { root: string; entryCount: number };
    trend: Array<{ day: string; hits: number; misses: number; bytesWritten: number; bytesRestored: number }>;
    scope: {
      root: string;
      trendWindowDays: number;
      legacy: { turns: number; hits: number; misses: number };
      otherRoots: { turns: number; hits: number; misses: number };
      unattributed: { turns: number; hits: number; misses: number };
      disabledTurns: number;
      unrootedSidecarCaptures: number;
    };
    health: {
      enqueued: number;
      queueDrops: number;
      evictions: number;
      corruptions: number;
      corruptionsTotal: number;
      queueDropsTotal: number;
      writeErrors: number;
      writeErrorsTotal: number;
      restoreDeclines: number;
      restoreSuppressed: number;
      sidecarCaptureReached: number;
      sidecarChainEmpty: number;
      sidecarBoundarySkips: number;
      sidecarAlreadyPersisted: number;
      sidecarEnqueued: number;
      sidecarQueueDrops: number;
      sidecarInstalled: number;
    };
    restoreFamilies: string[];
  }

  function totals(body: CacheBody): { hits: number; misses: number } {
    return body.trend.reduce((a, r) => ({ hits: a.hits + r.hits, misses: a.misses + r.misses }), {
      hits: 0,
      misses: 0,
    });
  }

  beforeEach(async () => {
    canonicalCacheRootPath = realpathSync(cacheRoot);
    // Start from an empty trace dir: the shared fixtures carry their own
    // (legitimately unattributed) cold rows, and this suite asserts exact
    // totals so a fixture edit can never quietly move them.
    for (const n of readdirSync(tracesDir)) if (n.endsWith('.jsonl')) unlinkSync(join(tracesDir, n));
    // 11 hits / 3 misses belong to THIS cache; 9999 hits belong to another cache
    // dir; 5000 hits predate attribution entirely. Anything unscoped reads 15010.
    writeTraceFile('2026-07-20-1.jsonl', [
      {
        traceId: 'mine-1',
        ts: Date.UTC(2026, 6, 20, 10),
        coldHits: 7,
        coldMisses: 2,
        coldBytesWritten: 100,
        coldBytesRestored: 700,
        coldRoot: canonicalCacheRootPath,
        coldEnabled: true,
        coldEnqueued: 5,
        coldQueueDrops: 3,
        coldEvictions: 2,
        coldCorruptions: 0,
        // DISTINCT non-zero cumulative totals across the two attributed rows so
        // MAX (5) and SUM (7) disagree. With 0 and 2 they were equal and the
        // reducer the acceptance bar depends on could be swapped for SUM with
        // the suite still green.
        coldCorruptionsTotal: 2,
        coldQueueDropsTotal: 4,
        // Write errors follow the corruption shape (SUM the deltas, MAX the
        // totals); declines follow neither — they are summed, because a
        // decline has a legitimate non-zero steady state.
        coldWriteErrors: 3,
        coldWriteErrorsTotal: 6,
        coldRestoreDeclines: 2,
        coldSidecarRestoreSuppressed: 1,
        // Every sidecar counter is a per-turn DELTA, so all eight SUM. The two
        // attributed rows carry different values for each so a reducer swapped
        // to MAX changes a number in the assertions below, and the eight sums
        // are pairwise distinct so a column aliased to a neighbour does too.
        // `coldSidecarEnqueued`/`coldSidecarQueueDrops` stay under the
        // object-scoped `coldEnqueued`/`coldQueueDrops` above, which they are a
        // subset of — the same admission bumps both.
        coldSidecarCaptureReached: 6,
        coldSidecarChainEmpty: 1,
        coldSidecarBoundarySkips: 2,
        coldSidecarAlreadyPersisted: 7,
        coldSidecarEnqueued: 2,
        coldSidecarQueueDrops: 2,
        coldSidecarInstalled: 4,
      },
      {
        traceId: 'mine-2',
        ts: Date.UTC(2026, 6, 21, 10),
        coldHits: 4,
        coldMisses: 1,
        coldBytesWritten: 50,
        coldBytesRestored: 400,
        coldRoot: canonicalCacheRootPath,
        coldEnabled: true,
        coldEnqueued: 3,
        coldQueueDrops: 5,
        coldEvictions: 1,
        coldCorruptions: 2,
        coldCorruptionsTotal: 5,
        coldQueueDropsTotal: 4,
        coldWriteErrors: 4,
        // Distinct from the other row's 6 so MAX (9) and SUM (15) disagree —
        // the same reason `coldCorruptionsTotal` carries 2 and 5.
        coldWriteErrorsTotal: 9,
        coldRestoreDeclines: 5,
        coldSidecarRestoreSuppressed: 2,
        coldSidecarCaptureReached: 4,
        coldSidecarChainEmpty: 3,
        coldSidecarBoundarySkips: 9,
        coldSidecarAlreadyPersisted: 9,
        coldSidecarEnqueued: 3,
        coldSidecarQueueDrops: 4,
        coldSidecarInstalled: 8,
      },
      {
        traceId: 'other-root-1',
        ts: Date.UTC(2026, 6, 20, 11),
        coldHits: 9999,
        coldMisses: 7,
        coldRoot: '/some/other/cache/mlx-paged-v1',
        coldEnabled: true,
        coldCorruptions: 55,
        coldCorruptionsTotal: 55,
        // Another root's faults must never leak into this cache's health.
        coldWriteErrors: 500,
        coldWriteErrorsTotal: 900,
        coldRestoreDeclines: 400,
        coldSidecarRestoreSuppressed: 300,
        // Another root's sidecar traffic, an order of magnitude larger, so a
        // projection that lost the `cold_root = ?` filter cannot look plausible.
        coldSidecarCaptureReached: 900,
        coldSidecarChainEmpty: 600,
        coldSidecarBoundarySkips: 500,
        coldSidecarAlreadyPersisted: 700,
        coldSidecarEnqueued: 400,
        coldSidecarQueueDrops: 350,
        coldSidecarInstalled: 800,
      },
      // No coldRoot AND no coldEnabled: written by a build that predates
      // attribution. Not "the tier was off" — genuinely unknown.
      { traceId: 'legacy-1', ts: Date.UTC(2026, 6, 19, 10), coldHits: 5000, coldMisses: 11 },
      // Tier explicitly off this turn: known, attributable, and zero — for
      // every BLOCK counter. The sidecar capture is the exception and the
      // reason this row carries one: `record_capture_reached()` is the first
      // statement of `capture_gemma4_sliding_cold_sidecar`, above its
      // `adapter.cold_tier()` guard, so a hybrid turn under
      // `--no-persist-cache` reaches the capture, counts it, and returns with
      // no tier to name a root from.
      {
        traceId: 'disabled-1',
        ts: Date.UTC(2026, 6, 20, 12),
        coldHits: 0,
        coldMisses: 0,
        coldEnabled: false,
        coldSidecarCaptureReached: 13,
      },
      // The partition hole: tier ON, root NULL. Matches none of `cold_root = ?`,
      // `cold_root IS NULL AND cold_enabled IS NULL`, `cold_root <> ?`, or
      // `cold_enabled = 0` — so before the fourth bucket these 777 hits were
      // reported by nothing at all.
      {
        traceId: 'orphan-1',
        ts: Date.UTC(2026, 6, 20, 13),
        coldHits: 777,
        coldMisses: 5,
        coldEnabled: true,
        coldSidecarCaptureReached: 21,
      },
    ]);
    await ingest();
  });

  it('scopes the hit/miss trend to the cache root being shown', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    // ONLY this root's rows. 15010 would mean the filter is gone.
    expect(totals(body)).toEqual({ hits: 11, misses: 3 });
    expect(body.scope.root).toBe(canonicalCacheRootPath);
    // Two days, two rows — neither the other root's day nor the legacy day.
    expect(body.trend.map((r) => r.day)).toEqual(['2026-07-20', '2026-07-21']);
    expect(body.trend.reduce((a, r) => a + r.bytesRestored, 0)).toBe(1100);
    expect(body.trend.reduce((a, r) => a + r.bytesWritten, 0)).toBe(150);
  });

  // The decision, pinned: a NULL-root row is EXCLUDED from the shown cache's
  // rate (counting it is the original defect) but REPORTED in its own bucket
  // (dropping it silently would make a real history vanish next to a root path
  // the user is staring at). It is bounded — trace JSONL retention prunes it —
  // so the bucket empties on its own.
  it('excludes legacy NULL-root rows from the rate but reports them as their own bucket', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(totals(body).hits).toBe(11);
    expect(body.scope.legacy).toEqual({ turns: 1, hits: 5000, misses: 11 });
    expect(body.scope.trendWindowDays).toBe(TRACE_RETENTION_DAYS);
    // A turn that ran with the tier OFF is a different story from an
    // unattributable one, and must not be folded into `legacy`.
    expect(body.scope.disabledTurns).toBe(1);
  });

  /**
   * The buckets must PARTITION the table, not merely be disjoint. A row with
   * `cold_enabled = 1` and a NULL `cold_root` matched no arm, so its 777 hits
   * were reported by nothing: trend 0, legacy 0, otherRoots 0, disabledTurns 0.
   * That is the silent-zero the whole scope design exists to refuse, in the one
   * shape it could still take.
   *
   * Asserted as a CONSERVATION law against the table's own total rather than by
   * enumerating the arms, so any future fifth shape has to be accounted for too.
   */
  it('accounts for every trace row: the buckets partition the table, they do not just avoid overlap', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.scope.unattributed).toEqual({ turns: 1, hits: 777, misses: 5 });
    const shown = totals(body);
    const accounted = {
      hits: shown.hits + body.scope.legacy.hits + body.scope.otherRoots.hits + body.scope.unattributed.hits,
      misses: shown.misses + body.scope.legacy.misses + body.scope.otherRoots.misses + body.scope.unattributed.misses,
    };
    // Every cold lookup the fixture wrote: 11 + 5000 + 9999 + 777 hits.
    expect(accounted).toEqual({ hits: 15_787, misses: 26 });
  });

  it('reports another cache directory as otherRoots rather than merging it in', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.scope.otherRoots).toEqual({ turns: 1, hits: 9999, misses: 7 });
    expect(totals(body).hits).toBe(11);
  });

  // The writer and the reader are different PROCESSES resolving the root from
  // their own environments. A raw string compare is a silent-zero trap: on macOS
  // `/tmp` → `/private/tmp` alone makes two identical-looking spellings unequal.
  it('matches a symlinked cache root through canonicalization', async () => {
    const linkRoot = join(base, 'cache-link');
    symlinkSync(cacheRoot, linkRoot);
    const linked = createDashboardRuntime({
      dbPath: ':memory:',
      tracesDir,
      modelsDir,
      cacheRoot: linkRoot,
    });
    const linkedApi = createTestClient(linked);
    try {
      expect((await linkedApi.fetch('/api/ingest', { method: 'POST' })).status).toBe(200);
      const body = (await (await linkedApi.fetch('/api/cache')).json()) as CacheBody;
      // Rows were written under the REAL path; the dashboard was pointed at the
      // symlink. A lexical compare would report a flat zero here.
      expect(body.scope.root).toBe(canonicalCacheRootPath);
      expect(totals(body)).toEqual({ hits: 11, misses: 3 });
    } finally {
      await linked.close();
    }
  });

  // The four counters `coldCacheStats()` always exposed and the agent always
  // dropped. "corruptions must be 0" is the stated acceptance bar for admitting
  // a family to the restore allowlist and was previously uncheckable from
  // anything the user runs.
  it('surfaces the cold-tier counters the agent used to drop, scoped to this cache', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.health.enqueued).toBe(8);
    // Object-scoped: 3 + 5 across the two attributed rows. Blocks and state
    // sidecars share this queue, so it is the SUPERSET of `sidecarQueueDrops`.
    expect(body.health.queueDrops).toBe(8);
    expect(body.health.evictions).toBe(3);
    expect(body.health.corruptions).toBe(2);
    // Cumulative totals reduce with MAX, not SUM (each process restarts at 0),
    // and are scoped — the other root's 55 corruptions must not leak in.
    //
    // The two attributed rows carry 2 and 5, so MAX is 5 and SUM would be 7:
    // asserting 5 is what makes the reducer, not just the scope, load-bearing.
    // Three processes that each saw 2 corruptions must report 2 under a label
    // that says "cumulative max", never 6.
    expect(body.health.corruptionsTotal).toBe(5);
    // Same shape for drops: both rows carry 4, so MAX is 4 and SUM would be 8.
    expect(body.health.queueDropsTotal).toBe(4);
  });

  /**
   * The two failure modes that had no number anywhere.
   *
   * A write that the queue accepted and the disk refused was reported by
   * nothing: the native writer is fail-open and swallows the error, so a root
   * that was read-only, full or unmounted produced `queueDrops 0`,
   * `corruptions 0` and a turn that exited 0 — a dashboard showing a perfectly
   * healthy cache holding nothing. A refused RESTORE was equally silent for
   * the opposite reason: it performs no lookup, so it moved neither hits nor
   * misses and rendered as "no lookups recorded", which reads as "nothing ran"
   * rather than "reuse was refused".
   *
   * The two reducers are asserted to DISAGREE with the alternative on purpose:
   * `writeErrorsTotal` is MAX over per-process totals (6 and 9 → 9, never 15),
   * while `restoreDeclines` is a SUM of deltas (2 + 5 → 7, never 5). Swapping
   * either reducer changes a number here.
   */
  it('surfaces write errors and restore declines, scoped and reduced correctly', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    // Deltas SUM across this cache's rows: 3 + 4. The other root's 500 must not
    // appear — a broken cache elsewhere is not this cache's alarm.
    expect(body.health.writeErrors).toBe(7);
    // Cumulative totals reduce with MAX: 6 and 9 → 9. SUM would say 15.
    expect(body.health.writeErrorsTotal).toBe(9);
    // Declines are SUMMED and have no MAX'd total, because unlike a corruption
    // a decline is normal on any prompt's first turn — a "did it ever happen"
    // latch would be pinned on from the first minute and mean nothing.
    expect(body.health.restoreDeclines).toBe(7);
    expect(body.health.restoreSuppressed).toBe(3);
  });

  /**
   * The other seven `coldSidecar*` counters, which the ingest mapping dropped
   * on the floor while only `restoreSuppressed` reached a column.
   *
   * `sidecarInstalled` is the reason this matters: every
   * `install_*_cold_sidecar` early-return falls through to a full O(prefix)
   * replay that produces CORRECT state, so a regression from "restored and
   * INSTALLED" to "restored and silently re-derived" leaves hits, cached
   * tokens, corruptions and the emitted text all unchanged. Nothing else in
   * the payload moves.
   *
   * All eight are per-turn DELTAS — none has a per-process cumulative twin —
   * so every one SUMS. The fixture's two attributed rows carry different
   * values for each counter, and the eight sums are pairwise distinct, so
   * swapping any reducer to MAX or aliasing any column to a neighbour changes
   * a number here. The other root's much larger traffic pins the scope.
   */
  it('projects every cold sidecar counter, summed per turn and scoped to this cache', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    // 6 + 4. MAX would say 6; the other root's 900 would say the scope is gone.
    expect(body.health.sidecarCaptureReached).toBe(10);
    // 1 + 3 (MAX 3), 2 + 9 (MAX 9), 7 + 9 (MAX 9).
    expect(body.health.sidecarChainEmpty).toBe(4);
    expect(body.health.sidecarBoundarySkips).toBe(11);
    expect(body.health.sidecarAlreadyPersisted).toBe(16);
    // Subsets of the object-scoped counters asserted above (8 enqueued, 8
    // dropped) — the same admission bumps both, so these are never added to them.
    expect(body.health.sidecarEnqueued).toBe(5);
    expect(body.health.sidecarQueueDrops).toBe(6);
    // 4 + 8. MAX would say 8, and the unscoped total would say 812.
    expect(body.health.sidecarInstalled).toBe(12);
  });

  /**
   * `captureReached` is the one sidecar counter that moves with NO tier in
   * hand. `record_capture_reached()` is the first statement of every family's
   * capture — `crates/mlx-core/src/models/gemma4/model.rs:2839` sits above the
   * `adapter.cold_tier()` guard at `:2843`, and qwen3_5 (`:2118` vs `:2122`)
   * and qwen3_5_moe (`:1497` vs `:1501`) are identical — so a hybrid turn run
   * under `--no-persist-cache`, or one whose tier failed to open, counts the
   * capture and then returns with no tier to name a root from. The writer only
   * emits `coldRoot` for a tier that was open
   * (`packages/agent/src/provider/index.ts:182`), so those turns land with a
   * NULL root and the root-scoped `health` query drops them.
   *
   * Reported as its own figure rather than folded into `health`: the scoping
   * exists so another cache's numbers never bleed into the shown one, and a
   * rootless row has no cache to bleed FROM. Dropping it silently is what makes
   * the page say "not reached — dense families" while the hybrid capture ran on
   * every turn and failed before acquiring a tier — the exact state these
   * counters were added to record.
   */
  it('reports sidecar captures that ran with no tier instead of dropping them', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    // 13 (tier off) + 21 (tier on, root unrecorded). The rooted rows' 10 must
    // NOT appear here, and the other root's 900 must not appear at all: a
    // rootless bucket that leaked either would read 23 / 34+10 / 934.
    expect(body.scope.unrootedSidecarCaptures).toBe(34);
    // The mirror assertion, and the one that pins the fix as ADDITIVE: the
    // shown cache's own figure stays exactly its two rooted rows.
    expect(body.health.sidecarCaptureReached).toBe(10);
  });

  // TRACE_RETENTION_DAYS is exported precisely so the window the UI prints and
  // the window the pruner enforces can never drift. Asserting the literal 30
  // passes whether `api.ts` imports the constant or hardcodes it, which defeats
  // the point — so pin the COUPLING: whatever the constant says, the payload
  // says.
  it('reports the trend window as the trace retention constant, not a copy of it', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.scope.trendWindowDays).toBe(TRACE_RETENTION_DAYS);
    // Guard the guard: a constant that went 0/NaN would make the assert vacuous.
    expect(TRACE_RETENTION_DAYS).toBeGreaterThan(0);
  });

  // F5: the UI must not carry a fifth hardcoded copy of the allowlist, so the
  // server sends it. The browser bundle cannot import @mlx-node/agent (it
  // transitively loads the native addon), and a hardcoded array in cache.tsx
  // would sit outside the native drift guard.
  it('serves the cold-tier restore allowlist so the SPA never hardcodes it', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.restoreFamilies).toEqual(coldTierRestoreFamilyList());
    expect(body.restoreFamilies.length).toBeGreaterThan(1);
  });
});

/**
 * The join key a real Control Panel window run actually uses.
 *
 * Every other cache test hands the runtime an explicit `cacheRoot`, but nothing
 * in production supplies one — the app's Control Panel entry constructs the runtime with
 * no cache root at all. So in production `ctx.cacheRoot` is `undefined` and the
 * scope root comes from
 * `coldCacheRoot(undefined)` → `defaultColdCacheDir()` → the `mlx-paged-v1`
 * child of `MLX_COLD_CACHE_DIR`, canonicalized. Whether THAT string equals what
 * the agent recorded was asserted nowhere — both sides were literals the tests
 * chose. A one-segment disagreement shows the Cache page a flat 0/0 trend beside
 * a populated disk scan: quieter than the 16189 the scoping fix was written to
 * kill, and just as wrong.
 *
 * `MLX_COLD_CACHE_DIR` is pointed at a SYMLINK here so the canonicalization step
 * is load-bearing on every platform, not only where `tmpdir()` happens to sit
 * under one.
 */
describe('dashboard api — production cache root (no explicit cacheRoot)', () => {
  interface DefaultCacheBody {
    disk: { root: string; exists: boolean; entryCount: number };
    trend: Array<{ hits: number; misses: number }>;
    scope: { root: string; otherRoots: { turns: number }; legacy: { turns: number } };
  }

  let envBase: string;
  let realColdDir: string;
  let linkedColdDir: string;
  /** Where the native tier actually opens: `<MLX_COLD_CACHE_DIR>/mlx-paged-v1`. */
  let managedRoot: string;
  let previousEnv: string | undefined;
  let defaulted: DashboardRuntime;
  let defaultedApi: TestClient;

  async function cacheBody(): Promise<DefaultCacheBody> {
    return (await (await defaultedApi.fetch('/api/cache')).json()) as DefaultCacheBody;
  }

  beforeEach(async () => {
    envBase = mkdtempSync(join(tmpdir(), 'dash-coldenv-'));
    realColdDir = join(envBase, 'cold');
    managedRoot = join(realColdDir, 'mlx-paged-v1');
    mkdirSync(managedRoot, { recursive: true });
    writeFileSync(join(managedRoot, hexBlock(9)), Buffer.alloc(512));
    linkedColdDir = join(envBase, 'cold-link');
    symlinkSync(realColdDir, linkedColdDir);

    previousEnv = process.env.MLX_COLD_CACHE_DIR;
    process.env.MLX_COLD_CACHE_DIR = linkedColdDir;

    // One trace row, written the way the agent writes it: the root goes through
    // `canonicalCacheRoot` in `provider/index.ts` before `record()` sees it, so
    // the WRITER side of the join is production code here, not a literal.
    for (const n of readdirSync(tracesDir)) if (n.endsWith('.jsonl')) unlinkSync(join(tracesDir, n));
    writeFileSync(
      join(tracesDir, '2026-07-22-default.jsonl'),
      `${JSON.stringify({
        v: 1,
        model: 'm',
        durationMs: 1,
        finishReason: 'stop',
        traceId: 'default-root-1',
        ts: Date.UTC(2026, 6, 22, 9),
        coldHits: 23,
        coldMisses: 5,
        coldEnabled: true,
        coldRoot: canonicalCacheRoot(managedRoot),
      })}\n`,
    );

    defaulted = createDashboardRuntime({
      dbPath: ':memory:',
      tracesDir,
      modelsDir,
      // Exactly what the app's Control Panel entry passes.
      cacheRoot: undefined,
    });
    defaultedApi = createTestClient(defaulted);
    expect((await defaultedApi.fetch('/api/ingest', { method: 'POST' })).status).toBe(200);
  });

  afterEach(async () => {
    await defaulted.close();
    if (previousEnv === undefined) delete process.env.MLX_COLD_CACHE_DIR;
    else process.env.MLX_COLD_CACHE_DIR = previousEnv;
    rmSync(envBase, { recursive: true, force: true });
  });

  it('scans the managed child of MLX_COLD_CACHE_DIR, never the directory itself', async () => {
    const body = await cacheBody();
    expect(body.disk.exists).toBe(true);
    expect(body.disk.entryCount).toBe(1);
    // Asserted structurally rather than against a repeated literal: the tier
    // always nests one managed level under the env var.
    expect(body.disk.root).not.toBe(linkedColdDir);
    expect(dirname(body.disk.root)).toBe(linkedColdDir);
  });

  it('joins the agent-written root when the CLI supplies no cacheRoot', async () => {
    const body = await cacheBody();
    expect(body.scope.root).toBe(canonicalCacheRoot(managedRoot));
    // JOINED, not merely "not crashed": the row's lookups land in the trend and
    // in neither exclusion bucket.
    expect(
      body.trend.reduce((a, r) => ({ hits: a.hits + r.hits, misses: a.misses + r.misses }), { hits: 0, misses: 0 }),
    ).toEqual({
      hits: 23,
      misses: 5,
    });
    expect(body.scope.otherRoots.turns).toBe(0);
    expect(body.scope.legacy.turns).toBe(0);
  });

  it('canonicalizes the derived root, so a symlinked MLX_COLD_CACHE_DIR still joins', async () => {
    const body = await cacheBody();
    // The reader was pointed at the symlink and the writer recorded the real
    // path. A lexical compare reports a flat zero here beside a disk scan that
    // just found a block.
    expect(body.disk.root.startsWith(linkedColdDir)).toBe(true);
    expect(body.scope.root.startsWith(realpathSync(realColdDir))).toBe(true);
    expect(body.scope.root).not.toBe(body.disk.root);
    expect(body.trend.length).toBe(1);
  });
});

describe('dashboard api — downloads', () => {
  it('lists jobs and rejects a non-catalog repo', async () => {
    const list = (await (await api.fetch('/api/downloads')).json()) as { jobs: unknown[] };
    expect(Array.isArray(list.jobs)).toBe(true);

    const bad = await api.fetch('/api/downloads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo: 'someone/not-in-catalog' }),
    });
    expect(bad.status).toBe(400);
  });

  // Finding E: the cancel route is wired end-to-end through the real runtime and
  // its real DownloadManager; an unknown/terminal id is a 404 (nothing to
  // cancel). The known-id → 200 branch is exercised deterministically below
  // without a live download.
  it('404s DELETE /api/downloads/:id for an unknown job', async () => {
    const res = await api.fetch('/api/downloads/no-such-job', { method: 'DELETE' });
    expect(res.status).toBe(404);
  });

  // The progress stream is a transport capability, not a handler: a transport
  // that can stream intercepts this route before dispatch, and one that cannot
  // must reach the placeholder and be TOLD so. A plain `call` — which is exactly
  // what the port bridge issues, since it carries progress over `subscribe`
  // instead — must therefore see 503, never a misleading 404.
  it('answers the streaming route 503, not a misleading 404, for a caller that cannot stream', async () => {
    const res = await api.fetch('/api/downloads/some-job/events');
    expect(res.status).toBe(503);
    expect(((await res.json()) as { error: string }).error).toMatch(/not available over this transport/i);
  });
});

/** Dispatch one request against a stub context, with no runtime or socket. */
function call(ctx: ApiContext, method: string, pathname: string, body?: unknown): ReturnType<typeof dispatch> {
  return dispatch(ctx, { method, pathname, query: new URLSearchParams(), body });
}

// Finding E: the DELETE /api/downloads/:id handler invokes DownloadManager.cancel
// with the decoded id and maps its boolean to 200 (cancelled) / 404 (unknown or
// already-terminal) — verified at the route layer with a stub manager so no live
// download or network is needed.
describe('dashboard api — cancel download route', () => {
  function ctxWith(cancel: (id: string) => boolean): ApiContext {
    return { downloads: { cancel } } as unknown as ApiContext;
  }

  it('returns 200 and cancels a known job id', async () => {
    const seen: string[] = [];
    const ctx = ctxWith((id) => {
      seen.push(id);
      return id === 'known-job';
    });

    expect(isApiPath('/api/downloads/known-job')).toBe(true);
    const res = await call(ctx, 'DELETE', '/api/downloads/known-job');
    expect(seen).toEqual(['known-job']);
    expect(res.status).toBe(200);
    expect(res.ok).toBe(true);
    expect(res.ok ? res.body : null).toEqual({ cancelled: true, id: 'known-job' });
  });

  it('returns 404 for an unknown/terminal job id', async () => {
    const ctx = ctxWith(() => false);

    const res = await call(ctx, 'DELETE', '/api/downloads/ghost');
    expect(res.status).toBe(404);
    expect(res.ok).toBe(false);
    expect(res.ok ? '' : res.code).toBe('E_NOT_FOUND');
    expect(res.ok ? '' : res.message).toContain('ghost');
  });
});

// A route that only ACCEPTS work answers 202, not 200: starting a download queues
// a job that has not run yet. Asserted at the route layer with a stub manager so
// no multi-GB transfer is triggered — a live-repo POST would really download.
describe('dashboard api — download start is 202 Accepted', () => {
  it('answers 202 (not 200) for an accepted download start', async () => {
    const ctx = { downloads: { start: () => 'job-1' } } as unknown as ApiContext;
    const res = await call(ctx, 'POST', '/api/downloads', { repo: 'org/repo' });
    expect(res.status).toBe(202);
    expect(res.ok ? res.body : null).toEqual({ id: 'job-1', repo: 'org/repo' });
  });
});

// A KNOWN path reached with the wrong method must be 405, not 404: the dispatcher
// keeps matching after a method mismatch and remembers that the path shape (with
// its `:param` segments) existed. Losing that collapses every wrong-method call to
// a misleading "no such route".
describe('dashboard api — method-not-allowed vs not-found', () => {
  const ctx = {} as unknown as ApiContext;

  it('405s a known path reached with the wrong method', async () => {
    for (const [method, pathname] of [
      ['PUT', '/api/models'],
      ['PATCH', '/api/models/model-a'],
      ['GET', '/api/ingest'],
      ['PATCH', '/api/downloads/some-id'],
    ] as const) {
      const res = await call(ctx, method, pathname);
      expect(res.status).toBe(405);
      expect(res.ok ? '' : res.code).toBe('E_METHOD_NOT_ALLOWED');
      expect(res.ok ? '' : res.message).toContain(method);
    }
  });

  it('404s a path no route has, whatever the method', async () => {
    for (const method of ['GET', 'POST', 'DELETE'] as const) {
      const res = await call(ctx, method, '/api/nope');
      expect(res.status).toBe(404);
      expect(res.ok ? '' : res.code).toBe('E_NOT_FOUND');
    }
  });

  it('does not treat a non-API path as a route', () => {
    expect(isApiPath('/')).toBe(false);
    expect(isApiPath('/assets/deep/link')).toBe(false);
    expect(isApiPath('/health')).toBe(true);
    expect(isApiPath('/api/models')).toBe(true);
  });
});

// A POST that lands while `close()` is draining downloads is refused by the
// manager. That refusal is a server-lifecycle condition, not a malformed request,
// so it must not be reported as a 400 blaming the caller.
describe('dashboard api — start download route', () => {
  function postDownload(start: () => string): ReturnType<typeof dispatch> {
    const ctx = { downloads: { start } } as unknown as ApiContext;
    return call(ctx, 'POST', '/api/downloads', { repo: 'org/model' });
  }

  it('answers 503 once the download manager is shutting down', async () => {
    const res = await postDownload(() => {
      throw new DownloadsClosedError();
    });
    expect(res.status).toBe(503);
    expect(res.ok).toBe(false);
    expect(res.ok ? '' : res.code).toBe('E_UNAVAILABLE');
    expect(res.ok ? '' : res.message).toMatch(/shutting down/i);
  });

  it('still answers 400 for a repo the catalog does not carry', async () => {
    const res = await postDownload(() => {
      throw new Error('Repo "org/model" is not in the model catalog');
    });
    expect(res.status).toBe(400);
    expect(res.ok ? '' : res.code).toBe('E_BAD_REQUEST');
  });
});

describe('dashboard api — health and unknown paths', () => {
  it('404s an unknown api path', async () => {
    const res = await api.fetch('/api/nope');
    expect(res.status).toBe(404);
  });

  it('reports health', async () => {
    const res = await api.fetch('/health');
    expect(res.status).toBe(200);
    expect(((await res.json()) as { status: string }).status).toBe('ok');
  });

  it('also serves health under /api/health', async () => {
    const res = await api.fetch('/api/health');
    expect(res.status).toBe(200);
    expect(((await res.json()) as { status: string }).status).toBe('ok');
  });
});

// The suite above drives `runtime.call` in-process. This block runs the SAME
// error-model contract through the real RPC transport — two real ports, real
// structured clone — so an envelope the bridge mangles (a lost `code`, a
// re-derived status, a `bodyError` dropped on the floor) fails here rather than
// only in the renderer. Everything else stays in-process because it is API
// behaviour, not transport behaviour.
describe('dashboard api — the RPC transport carries the error model over a port', () => {
  let rpc: TestClient;
  let teardown: () => void;

  beforeEach(() => {
    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    const dispose = serveRuntimeOverPort(runtime, bindEventTargetPort(port2));
    const client = createRpcClient(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
    rpc = createTestClient(client);
    teardown = () => {
      client.close();
      dispose();
    };
  });

  afterEach(() => {
    teardown();
  });

  it('answers a plain read over the port', async () => {
    const res = await rpc.fetch('/api/models');
    expect(res.status).toBe(200);
    expect((await res.json()) as { dir: string }).toMatchObject({ dir: modelsDir });
  });

  it('carries a handler 404 across with its message', async () => {
    const res = await rpc.fetch('/api/models/no-such-model', { method: 'DELETE' });
    expect(res.status).toBe(404);
    expect((await res.json()) as { error: string }).toMatchObject({ error: expect.stringContaining('not found') });
  });

  it('carries a wrong method on a known path as 405, not 404', async () => {
    const res = await rpc.fetch('/api/models', { method: 'PUT' });
    expect(res.status).toBe(405);
    expect(((await res.json()) as { error: string }).error).toContain('PUT');
  });

  it('carries an unparseable request body as a 400 from the handler that needed it', async () => {
    const res = await rpc.fetch('/api/cache', { method: 'DELETE', rawBody: '{ this is not json' });
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: string }).error).toBe('Invalid JSON in request body');
    // Nothing was cleared by the rejected call.
    const after = (await (await api.fetch('/api/cache')).json()) as { disk: { entryCount: number } };
    expect(after.disk.entryCount).toBe(2);
  });

  it('checks the resource id BEFORE the body, so an unknown id still 404s', async () => {
    // The delete handler looks the model up first; an unparseable body must not
    // pre-empt that 404 (the transport records the parse failure as data instead
    // of raising it at read time).
    const res = await rpc.fetch('/api/models/no-such-model', { method: 'DELETE', rawBody: '{ not json either' });
    expect(res.status).toBe(404);
  });

  it('reaches the database worker through the port, not just main-thread routes', async () => {
    // `/api/cache` is worker-owned: a reply proves the port hop and the thread
    // hop compose, which no in-process test can show.
    const res = await rpc.fetch('/api/cache');
    expect(res.status).toBe(200);
    expect((await res.json()) as { disk: { entryCount: number } }).toMatchObject({
      disk: { entryCount: expect.any(Number) },
    });
  });
});
