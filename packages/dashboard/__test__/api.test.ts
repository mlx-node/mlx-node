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
import { quantilesOf } from '../src/api/quantiles.js';
import { CodingAgentsService } from '../src/coding-agents.js';
import { openDashboardDb } from '../src/db/open.js';
import { DownloadsClosedError } from '../src/download.js';
import { ingestTraces, TRACE_RETENTION_DAYS } from '../src/ingest/traces.js';
import { createRpcClient } from '../src/rpc/client.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import { createDashboardRuntime, type DashboardRuntime } from '../src/runtime.js';
import { turnPhases } from '../ui/src/lib/metrics-view.js';
import type { MetricsTotals, TurnRow } from '../ui/src/lib/types.js';
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

/** Start of the local calendar day holding `ms`. */
function localDayStart(ms: number): number {
  const d = new Date(ms);
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
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

/** GET a route that must answer 200, as its JSON body. */
async function get<T>(path: string): Promise<T> {
  const res = await api.fetch(path);
  expect(res.status).toBe(200);
  return (await res.json()) as T;
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
    });
    try {
      expect(await service.state()).toMatchObject({ model: names[1], available: true, unavailableReason: null });
      expect(await service.refresh()).toMatchObject({ available: true });
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
    });
    try {
      expect(await service.state()).toMatchObject({ model: name, available: true, unavailableReason: null });
      expect(await service.refresh()).toMatchObject({ available: true });
    } finally {
      await service.close();
    }
  });

  it('only offers complete generative models for coding agent setup', async () => {
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
    expect(((await unknown.json()) as { error: string }).error).toContain('not found');

    // Present on disk but not a checkpoint → a refusal, not a 404.
    const refused = await api.fetch('/api/models/not-a-checkpoint', { method: 'DELETE' });
    expect(refused.status).toBe(400);
    expect(((await refused.json()) as { error: string }).error).toContain('not a model checkpoint');
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
      const summary = (await (await api.fetch('/api/metrics/summary')).json()) as {
        totals: { turns: number };
      };
      expect(summary.totals.turns).toBe(3);
    } finally {
      chmodSync(tracesDir, 0o755); // restore so afterEach cleanup can remove it
    }
  });
});

describe('dashboard api — metrics', () => {
  /** One v2 trace line with the fields every record carries. */
  function turn(rec: Record<string, unknown>): Record<string, unknown> {
    return {
      v: 2,
      ts: Date.UTC(2026, 6, 10, 12),
      pid: 1,
      model: 'm',
      outcome: 'ok',
      endpoint: 'messages',
      stream: true,
      dispatch: 'start',
      finishReason: 'stop',
      durationMs: 100,
      toolCount: 0,
      imageCount: 0,
      concurrency: 1,
      concurrencyPeak: 1,
      ...rec,
    };
  }

  function writeTurns(name: string, records: Array<Record<string, unknown>>): void {
    writeFileSync(join(tracesDir, name), records.map((r) => `${JSON.stringify(turn(r))}\n`).join(''));
  }

  interface Q {
    n: number;
    p05: number | null;
    p50: number | null;
    p95: number | null;
  }

  beforeEach(() => {
    for (const n of readdirSync(tracesDir)) if (n.endsWith('.jsonl')) unlinkSync(join(tracesDir, n));
  });

  it('summary: nearest-rank p05/p50/p95 for n = 1, 2, 3, 20, 21', async () => {
    const sizes = [1, 2, 3, 20, 21];
    const records: Array<Record<string, unknown>> = [];
    for (const n of sizes) {
      // Written in descending order so a missing sort shows up.
      for (let v = n; v >= 1; v--) records.push({ traceId: `q${n}-${v}`, model: `n${n}`, ttftMs: v });
    }
    writeTurns('2026-07-10-1.jsonl', records);
    await ingest();
    const expected: Record<number, [number, number, number]> = {
      1: [1, 1, 1],
      2: [1, 1, 2],
      3: [1, 2, 3],
      20: [1, 10, 19],
      21: [2, 11, 20],
    };
    for (const n of sizes) {
      const body = await get<{ latency: { ttftMs: Q } }>(`/api/metrics/summary?model=n${n}`);
      const [p05, p50, p95] = expected[n];
      expect(body.latency.ttftMs).toEqual({ n, p05, p50, p95 });
    }
  });

  it('summary: quantiles equal the reference window-function SQL', async () => {
    const values: Record<string, number[]> = {
      a: [5, 3, 9, 1, 7, 2, 8, 4, 6, 10, 11, 12, 13],
      b: [100, 50],
      c: [42],
    };
    const records: Array<Record<string, unknown>> = [];
    for (const [model, list] of Object.entries(values)) {
      list.forEach((v, i) => records.push({ traceId: `ref-${model}-${i}`, model, ttftMs: v }));
    }
    writeTurns('2026-07-10-2.jsonl', records);
    await ingest();

    const ref = openDashboardDb(':memory:');
    try {
      await ingestTraces(ref, tracesDir);
      const rows = ref.sqlite
        .prepare(
          `WITH base AS (
             SELECT model_key AS g, ttft_ms AS v FROM traces WHERE outcome = 'ok' AND ttft_ms IS NOT NULL
           ),
           ranked AS (
             SELECT g, v, ROW_NUMBER() OVER (PARTITION BY g ORDER BY v) AS rn, COUNT(*) OVER (PARTITION BY g) AS n
             FROM base
           )
           SELECT g, n,
                  MAX(CASE WHEN rn = (n *  5 + 99) / 100 THEN v END) AS p05,
                  MAX(CASE WHEN rn = (n * 50 + 99) / 100 THEN v END) AS p50,
                  MAX(CASE WHEN rn = (n * 95 + 99) / 100 THEN v END) AS p95
           FROM ranked GROUP BY g ORDER BY g`,
        )
        .all() as Array<{ g: string; n: number; p05: number; p50: number; p95: number }>;
      expect(rows).toHaveLength(3);
      for (const row of rows) {
        expect(quantilesOf(values[row.g])).toEqual({ n: row.n, p05: row.p05, p50: row.p50, p95: row.p95 });
        const body = await get<{ latency: { ttftMs: Q } }>(`/api/metrics/summary?model=${row.g}`);
        expect(body.latency.ttftMs).toEqual({ n: row.n, p05: row.p05, p50: row.p50, p95: row.p95 });
      }
    } finally {
      ref.close();
    }
  });

  it('summary: percentiles ignore non-ok turns, counts include them', async () => {
    writeTurns('2026-07-10-3.jsonl', [
      { traceId: 'ok-1', ttftMs: 10, outputTokens: 5 },
      { traceId: 'ok-2', ttftMs: 20, outputTokens: 5 },
      {
        traceId: 'err-1',
        outcome: 'error',
        errorKind: 'exception',
        ttftMs: 9000,
        outputTokens: 3,
        finishReason: 'error',
      },
      {
        traceId: 'can-1',
        outcome: 'cancelled',
        failPhase: 'decode',
        ttftMs: 9000,
        outputTokens: 2,
        finishReason: 'cancelled',
      },
    ]);
    await ingest();
    const body = await get<{
      totals: { turns: number; ok: number; error: number; cancelled: number; outputTokens: number };
      latency: { ttftMs: Q };
      failures: Array<{ outcome: string; errorKind: string | null; failPhase: string | null; turns: number }>;
      finishReasons: Array<{ reason: string; turns: number }>;
    }>('/api/metrics/summary');
    expect(body.totals).toMatchObject({ turns: 4, ok: 2, error: 1, cancelled: 1, outputTokens: 15 });
    expect(body.latency.ttftMs).toEqual({ n: 2, p05: 10, p50: 10, p95: 20 });
    expect(body.failures).toEqual(
      expect.arrayContaining([
        { outcome: 'error', errorKind: 'exception', failPhase: null, turns: 1 },
        { outcome: 'cancelled', errorKind: null, failPhase: 'decode', turns: 1 },
      ]),
    );
    expect(body.finishReasons[0]).toEqual({ reason: 'stop', turns: 2 });
  });

  it('summary: previous window is null without both bounds', async () => {
    const t = Date.UTC(2026, 6, 10, 12);
    writeTurns('2026-07-10-4.jsonl', [
      { traceId: 'prev', ts: t - 1500, ttftMs: 1 },
      { traceId: 'cur', ts: t - 500, ttftMs: 2 },
    ]);
    await ingest();
    expect((await get<{ previous: unknown }>(`/api/metrics/summary?from=${t - 1000}`)).previous).toBeNull();
    expect((await get<{ previous: unknown }>(`/api/metrics/summary?to=${t}`)).previous).toBeNull();
    const both = await get<{ totals: { turns: number }; previous: { totals: { turns: number } } }>(
      `/api/metrics/summary?from=${t - 1000}&to=${t}`,
    );
    expect(both.totals.turns).toBe(1);
    expect(both.previous.totals.turns).toBe(1);
    expect((await api.fetch(`/api/metrics/summary?from=${t}&to=${t - 1}`)).status).toBe(400);
  });

  it('summary: a one-token turn with decode_tps 0 is in neither the decode quantiles nor the histogram', async () => {
    writeFileSync(
      join(tracesDir, '2026-07-10-5.jsonl'),
      [
        { v: 1, traceId: 'title', ts: Date.UTC(2026, 6, 10), model: 'm', outputTokens: 1, decodeTps: 0 },
        { v: 1, traceId: 'real', ts: Date.UTC(2026, 6, 10), model: 'm', outputTokens: 50, decodeTps: 42 },
      ]
        .map((r) => `${JSON.stringify(r)}\n`)
        .join(''),
    );
    await ingest();
    const summary = await get<{ latency: { decodeTps: Q } }>('/api/metrics/summary');
    expect(summary.latency.decodeTps).toEqual({ n: 1, p05: 42, p50: 42, p95: 42 });
    const dist = await get<{ series: Array<{ decodeTps: { n: number; counts: number[] } }> }>(
      '/api/metrics/distributions',
    );
    expect(dist.series[0].decodeTps.n).toBe(1);
    expect(dist.series[0].decodeTps.counts[0]).toBe(0);
  });

  it('prefill speed reads rated rows only and the first-token time splits into setup and prefill', async () => {
    const at = Date.UTC(2026, 6, 10, 12);
    writeTurns('2026-07-10-7.jsonl', [
      { traceId: 'A', ts: at, ttftMs: 500, prefillTps: 900, promptTokens: 2000, durationMs: 5000 },
      { traceId: 'B', ts: at + 1, ttftMs: 1200, prefillMs: 1000, prefillTokens: 2000, durationMs: 5000 },
      { traceId: 'C', ts: at + 2, ttftMs: 400, prefillMs: 50, prefillTokens: 100, durationMs: 5000 },
      { traceId: 'D', ts: at + 3, ttftMs: 300, stream: false, durationMs: 5000 },
      { traceId: 'E', ts: at + 4, ttftMs: 700, prefillMs: 800, prefillTokens: 1000, durationMs: 5000 },
    ]);
    await ingest();
    const summary = await get<{
      totals: { prefillMeasured: number; firstTokenWallMs: number; setupWallMs: number; prefillWallMs: number };
      latency: { prefillTps: Q; setupMs: Q };
    }>('/api/metrics/summary');
    expect(summary.latency.prefillTps).toEqual({ n: 2, p05: 1250, p50: 1250, p95: 2000 });
    expect(summary.totals).toMatchObject({
      prefillMeasured: 3,
      firstTokenWallMs: 800,
      setupWallMs: 550,
      prefillWallMs: 1750,
    });
    expect(summary.latency.setupMs).toEqual({ n: 3, p05: 0, p50: 200, p95: 350 });

    const turns = await get<{
      rows: Array<{
        traceId: string;
        prefillMs: number | null;
        prefillTokens: number | null;
        setupMs: number | null;
        prefillTps: number | null;
      }>;
    }>('/api/metrics/turns');
    const byId = new Map(turns.rows.map((r) => [r.traceId, r]));
    for (const id of ['A', 'D']) {
      expect(byId.get(id)).toMatchObject({ prefillMs: null, prefillTokens: null, setupMs: null, prefillTps: null });
    }
    expect(byId.get('B')).toMatchObject({ prefillMs: 1000, prefillTokens: 2000, setupMs: 200, prefillTps: 2000 });
    expect(byId.get('C')).toMatchObject({ prefillMs: 50, prefillTokens: 100, setupMs: 350, prefillTps: null });

    const dist = await get<{ series: Array<{ prefillTps: { n: number } }> }>('/api/metrics/distributions');
    expect(dist.series[0].prefillTps.n).toBe(2);
    const series = await get<{ points: Array<{ prefillTps: Q }> }>(
      `/api/metrics/timeseries?bucket=hour&from=${at - 3_600_000}&to=${at + 3_600_000}`,
    );
    expect(series.points).toHaveLength(1);
    expect(series.points[0].prefillTps.n).toBe(2);
  });

  it('summed phases stay within the wall time when the times of a row disagree', async () => {
    const at = Date.UTC(2026, 6, 10, 12);
    writeTurns('2026-07-10-8.jsonl', [
      {
        traceId: 'over',
        ts: at,
        durationMs: 100,
        ttftMs: 200,
        prefillMs: 150,
        prefillTokens: 1000,
        outputTokens: 11,
        decodeTps: 100,
      },
      { traceId: 'decode', ts: at + 1, durationMs: 1000, ttftMs: 400, outputTokens: 101, decodeTps: 100 },
      { traceId: 'unsplit', ts: at + 2, durationMs: 100, ttftMs: 900 },
      {
        traceId: 'admission',
        ts: at + 3,
        durationMs: 100,
        preDispatchMs: 10,
        queueMs: 500,
        loadWaitMs: 500,
        resolveMs: 500,
      },
      {
        traceId: 'sane',
        ts: at + 4,
        durationMs: 2000,
        preDispatchMs: 50,
        queueMs: 20,
        ttftMs: 600,
        prefillMs: 500,
        prefillTokens: 1000,
        outputTokens: 101,
        decodeTps: 100,
      },
    ]);
    await ingest();
    const phaseSum = (t: MetricsTotals): number =>
      t.queueWallMs +
      t.loadWaitWallMs +
      t.resolveWallMs +
      t.firstTokenWallMs +
      t.setupWallMs +
      t.prefillWallMs +
      t.decodeWallMs +
      t.otherWallMs;
    const summary = await get<{ totals: MetricsTotals }>('/api/metrics/summary');
    const series = await get<{ points: Array<{ totals: MetricsTotals }> }>('/api/metrics/timeseries?bucket=day');
    expect(series.points).toHaveLength(1);
    for (const totals of [summary.totals, series.points[0]!.totals]) {
      expect(phaseSum(totals)).toBeCloseTo(totals.totalWallMs, 6);
      for (const value of Object.values(totals)) expect(value).toBeGreaterThanOrEqual(0);
    }

    const turns = await get<{ rows: TurnRow[] }>('/api/metrics/turns');
    expect(turns.rows).toHaveLength(5);
    const rowSum = (pick: (p: ReturnType<typeof turnPhases>) => number): number =>
      turns.rows.reduce((sum, row) => sum + pick(turnPhases(row)), 0);
    expect(summary.totals).toMatchObject({
      queueWallMs: rowSum((p) => p.queueMs),
      loadWaitWallMs: rowSum((p) => p.loadWaitMs),
      resolveWallMs: rowSum((p) => p.resolveMs),
      firstTokenWallMs: rowSum((p) => p.firstTokenMs),
      setupWallMs: rowSum((p) => p.setupMs),
      prefillWallMs: rowSum((p) => p.prefillMs),
      totalWallMs: rowSum((p) => p.totalMs),
      otherWallMs: rowSum((p) => p.handlingMs + p.tailMs),
    });
    expect(summary.totals.decodeWallMs).toBeCloseTo(
      rowSum((p) => p.decodeMs),
      6,
    );

    writeTurns('2026-07-10-8.jsonl', [
      {
        traceId: 'over',
        ts: at,
        durationMs: 100,
        ttftMs: 200,
        prefillMs: 150,
        prefillTokens: 1000,
        outputTokens: 11,
        decodeTps: 100,
      },
    ]);
    await ingest();
    const one = await get<{ totals: MetricsTotals }>('/api/metrics/summary');
    expect(one.totals).toMatchObject({
      setupWallMs: 0,
      prefillWallMs: 100,
      decodeWallMs: 0,
      otherWallMs: 0,
      totalWallMs: 100,
    });
  });

  it('timeseries: hour and day buckets follow local time', async () => {
    const savedTz = process.env.TZ;
    process.env.TZ = 'Asia/Kolkata';
    try {
      // 10:10 UTC is 15:40 in UTC+05:30.
      const at = Date.UTC(2026, 6, 10, 10, 10);
      writeTurns('2026-07-10-6.jsonl', [{ traceId: 'tz', ts: at }]);
      await ingest();
      const hour = await get<{ points: Array<{ t: number }> }>(
        `/api/metrics/timeseries?bucket=hour&from=${at - 3_600_000}&to=${at}`,
      );
      expect(hour.points.map((p) => p.t)).toEqual([Date.UTC(2026, 6, 10, 9, 30)]);
      const day = await get<{ points: Array<{ t: number }> }>('/api/metrics/timeseries?bucket=day');
      expect(day.points.map((p) => p.t)).toEqual([Date.UTC(2026, 6, 9, 18, 30)]);
    } finally {
      if (savedTz === undefined) delete process.env.TZ;
      else process.env.TZ = savedTz;
    }
  });

  it('timeseries: groupBy none merges models', async () => {
    const at = Date.UTC(2026, 6, 10, 12);
    writeTurns('2026-07-10-7.jsonl', [
      { traceId: 'g1', model: 'x', ts: at, outputTokens: 10, ttftMs: 1 },
      { traceId: 'g2', model: 'y', ts: at + 1, outputTokens: 20, ttftMs: 3 },
    ]);
    await ingest();
    const byModel = await get<{ points: Array<{ model: string }> }>('/api/metrics/timeseries?bucket=day');
    expect(byModel.points.map((p) => p.model)).toEqual(['x', 'y']);
    const merged = await get<{
      groupBy: string;
      points: Array<{ model: string; totals: { turns: number; outputTokens: number }; ttftMs: Q }>;
    }>('/api/metrics/timeseries?bucket=day&groupBy=none');
    expect(merged.groupBy).toBe('none');
    expect(merged.points).toHaveLength(1);
    expect(merged.points[0].model).toBe('*');
    expect(merged.points[0].totals).toMatchObject({ turns: 2, outputTokens: 30 });
    expect(merged.points[0].ttftMs).toEqual({ n: 2, p05: 1, p50: 1, p95: 3 });
    expect((await api.fetch('/api/metrics/timeseries')).status).toBe(400);
    expect((await api.fetch('/api/metrics/timeseries?bucket=5m')).status).toBe(400);
    expect((await api.fetch('/api/metrics/timeseries?bucket=5m&from=0&to=1000000000')).status).toBe(400);
    expect((await api.fetch('/api/metrics/timeseries?bucket=day&groupBy=family')).status).toBe(400);
  });

  it('null perf stays null', async () => {
    writeFileSync(
      join(tracesDir, '2026-07-10-8.jsonl'),
      `${JSON.stringify({ v: 1, traceId: 'np-null', ts: Date.UTC(2026, 6, 10), model: 'nullperf-model' })}\n`,
    );
    await ingest();
    const body = await get<{
      points: Array<{ totals: { turns: number }; ttftMs: Q; decodeTps: Q; mtpMeanAccepted: number | null }>;
    }>('/api/metrics/timeseries?bucket=day');
    expect(body.points).toHaveLength(1);
    expect(body.points[0].totals.turns).toBe(1);
    expect(body.points[0].ttftMs).toEqual({ n: 0, p05: null, p50: null, p95: null });
    expect(body.points[0].decodeTps).toEqual({ n: 0, p05: null, p50: null, p95: null });
    expect(body.points[0].mtpMeanAccepted).toBeNull();
  });

  it('distributions: counts land in the documented bins and the last bin is open-ended', async () => {
    writeTurns('2026-07-10-9.jsonl', [
      { traceId: 'd1', ttftMs: 0 },
      { traceId: 'd2', ttftMs: 99.9 },
      { traceId: 'd3', ttftMs: 100 },
      { traceId: 'd4', ttftMs: 60000 },
      { traceId: 'd5', ttftMs: 999_999 },
    ]);
    await ingest();
    const body = await get<{
      edges: { ttftMs: number[] };
      series: Array<{ model: string; ttftMs: { n: number; counts: number[] } }>;
    }>('/api/metrics/distributions');
    expect(body.edges.ttftMs).toEqual([
      0, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000, 8000, 12000, 20000, 30000, 60000,
    ]);
    const h = body.series[0].ttftMs;
    expect(h.n).toBe(5);
    expect(h.counts).toHaveLength(body.edges.ttftMs.length);
    expect(h.counts[0]).toBe(2);
    expect(h.counts[1]).toBe(1);
    expect(h.counts[h.counts.length - 1]).toBe(2);
  });

  it('speculative: acceptance by position is weighted by cycles', async () => {
    writeTurns('2026-07-10-10.jsonl', [
      { traceId: 's1', mtpCycles: 10, mtpMeanAccepted: 2, mtpAcceptByPos: [1, 0.5], decodeTps: 80, outputTokens: 30 },
      { traceId: 's2', mtpCycles: 30, mtpMeanAccepted: 3, mtpAcceptByPos: [0.5], decodeTps: 90, outputTokens: 30 },
      { traceId: 's3', decodeTps: 40, outputTokens: 30 },
    ]);
    await ingest();
    const body = await get<{
      models: Array<{
        turns: number;
        mtpTurns: number;
        cycles: number;
        meanAccepted: number;
        acceptByPos: Array<{ pos: number; rate: number; cycles: number }>;
        decodeTpsWithMtp: Q;
        decodeTpsWithoutMtp: Q;
      }>;
      trend: Array<{ meanAccepted: number; mtpTurns: number }>;
    }>('/api/metrics/speculative?bucket=day');
    const m = body.models[0];
    expect(m).toMatchObject({ turns: 3, mtpTurns: 2, cycles: 40 });
    expect(m.meanAccepted).toBeCloseTo((2 * 10 + 3 * 30) / 40, 10);
    expect(m.acceptByPos).toEqual([
      { pos: 0, rate: (1 * 10 + 0.5 * 30) / 40, cycles: 40 },
      { pos: 1, rate: 0.5, cycles: 10 },
    ]);
    expect(m.decodeTpsWithMtp.n).toBe(2);
    expect(m.decodeTpsWithoutMtp).toEqual({ n: 1, p05: 40, p50: 40, p95: 40 });
    expect(body.trend[0].mtpTurns).toBe(2);
    expect(body.trend[0].meanAccepted).toBeCloseTo(110 / 40, 10);
  });

  it('turns: cursor pages without gaps or repeats across equal timestamps', async () => {
    const at = Date.UTC(2026, 6, 10, 12);
    const records: Array<Record<string, unknown>> = [];
    for (let i = 0; i < 7; i++) records.push({ traceId: `t-${i}`, ts: i < 5 ? at : at - 1000 * i });
    writeTurns('2026-07-10-11.jsonl', records);
    await ingest();
    const seen: string[] = [];
    let path = '/api/metrics/turns?limit=2';
    for (let page = 0; page < 10; page++) {
      const body = await get<{
        rows: Array<{ traceId: string; ts: number }>;
        next: { beforeTs: number; beforeTraceId: string } | null;
        total: number;
      }>(path);
      expect(body.total).toBe(7);
      seen.push(...body.rows.map((r) => r.traceId));
      if (body.next === null) break;
      path = `/api/metrics/turns?limit=2&beforeTs=${body.next.beforeTs}&beforeTraceId=${body.next.beforeTraceId}`;
    }
    expect(seen).toEqual(['t-4', 't-3', 't-2', 't-1', 't-0', 't-5', 't-6']);
  });

  it('turns: a non-numeric limit or beforeTs is a bad request', async () => {
    const badLimit = await api.fetch('/api/metrics/turns?limit=abc');
    expect(badLimit.status).toBe(400);
    expect(((await badLimit.json()) as { error: string }).error).toBe('limit must be a number');
    const badCursor = await api.fetch('/api/metrics/turns?beforeTs=abc&beforeTraceId=x');
    expect(badCursor.status).toBe(400);
    expect(((await badCursor.json()) as { error: string }).error).toBe('beforeTs must be a number');
  });

  it('turns: traceId returns that one row', async () => {
    writeTurns('2026-07-10-12.jsonl', [
      {
        traceId: 'wanted',
        model: 'alias',
        modelId: 'Checkpoint',
        ctxWindow: 1000,
        promptTokens: 200,
        outputTokens: 51,
        decodeTps: 50,
        preDispatchMs: 10,
      },
      { traceId: 'other' },
    ]);
    await ingest();
    const body = await get<{ rows: Array<Record<string, unknown>>; next: unknown; total: number }>(
      '/api/metrics/turns?traceId=wanted&limit=1&outcome=error',
    );
    expect(body.total).toBe(1);
    expect(body.next).toBeNull();
    expect(body.rows[0]).toMatchObject({
      traceId: 'wanted',
      model: 'Checkpoint',
      alias: 'alias',
      stream: true,
      ctxFillPct: 25.1,
      decodeMs: 1000,
      preDispatchMs: 10,
    });
    expect((await get<{ rows: unknown[] }>('/api/metrics/turns?traceId=missing')).rows).toEqual([]);
  });

  it('models: keys in first-seen order', async () => {
    const at = Date.UTC(2026, 6, 10, 12);
    writeTurns('2026-07-10-13.jsonl', [
      { traceId: 'm1', model: 'late', ts: at + 10 },
      { traceId: 'm2', model: 'alias', modelId: 'early', modelType: 'qwen3', ts: at },
      { traceId: 'm3', model: 'late', ts: at + 20 },
    ]);
    await ingest();
    const body = await get<{ models: Array<{ model: string; modelType: string | null; turns: number }> }>(
      '/api/metrics/models',
    );
    expect(body.models).toEqual([
      { model: 'early', modelType: 'qwen3', firstTs: at, lastTs: at, turns: 1 },
      { model: 'late', modelType: null, firstTs: at + 10, lastTs: at + 20, turns: 2 },
    ]);
  });
});

describe('dashboard api — runtime series', () => {
  const T = Date.UTC(2026, 10, 10, 12, 0, 0);
  let runtimeDir: string;

  function base1(pid: number, startedAt: number, ts: number): Record<string, unknown> {
    return { v: 1, ts, pid, startedAt };
  }

  function sample(pid: number, ts: number, extra: Record<string, unknown> = {}): Record<string, unknown> {
    return {
      ...base1(pid, T - 60_000, ts),
      k: 'sample',
      dtMs: 1000,
      busy: true,
      busyMs: 0,
      health: 'ok',
      memActive: 0,
      memPeak: 0,
      memCache: 0,
      rss: 0,
      inFlight: 0,
      queued: 0,
      resident: 1,
      dOk: 0,
      dError: 0,
      dCancelled: 0,
      dOutTokens: 0,
      dRejected: 0,
      ...extra,
    };
  }

  function sched(ts: number, extra: Record<string, unknown>): Record<string, unknown> {
    return {
      ...base1(7, T - 60_000, ts),
      k: 'sched',
      model: 'M',
      blockCapacity: 100,
      allocatedBlocks: 1,
      freeBlocks: 99,
      reclaimableBlocks: 0,
      watermarkBlocks: 1,
      reservedBlocks: 0,
      memoryCapacityBytes: 1000,
      memoryWatermarkBytes: 10,
      reservedBlockBytes: 0,
      reservedStateBytes: 0,
      globalSteps: 0,
      admitted: 0,
      completed: 0,
      maxBatchOccupancy: 1,
      fusedGreedyEpilogueSteps: 0,
      admissionDeferredBlocks: 0,
      admissionDeferredState: 0,
      rowsAllocEvicted: 0,
      preemptions: 0,
      preemptionsRecompute: 0,
      preemptionsSsd: 0,
      ssdRestoreWaiting: 0,
      ssdRestoreBytes: 0,
      ssdRestoreWaitMs: 0,
      occHist: [],
      ...extra,
    };
  }

  function writeSeries(name: string, records: Array<Record<string, unknown>>): void {
    mkdirSync(runtimeDir, { recursive: true });
    writeFileSync(join(runtimeDir, name), records.map((r) => `${JSON.stringify(r)}\n`).join(''));
  }

  interface Series {
    points: Array<{
      t: number;
      samples: number;
      memActiveAvg: number | null;
      memActiveMax: number | null;
      busyMs: number;
      okTurns: number;
      outTokens: number;
    }>;
    sched: Array<{
      model: string;
      latest: { maxBatchOccupancy: number; ts: number; turns: number | null };
      snapshots: unknown[];
      delta: Record<string, unknown> & { occupancy: Array<{ occupancy: number; steps: number }> };
    }>;
    procs: Array<{ procKey: string }>;
  }

  beforeEach(() => {
    runtimeDir = join(base, 'runtime');
  });

  it('runtime series: two samples 1 s apart in one 5 s bucket give one point with samples 2', async () => {
    writeSeries('2026-11-10-7.jsonl', [sample(7, T + 1000), sample(7, T + 2000)]);
    await ingest();
    const body = await get<Series>(`/api/runtime/series?from=${T}&to=${T + 4999}&bucketMs=5000`);
    expect(body.points).toHaveLength(1);
    expect(body.points[0]).toMatchObject({ t: T, samples: 2 });
  });

  it('runtime series: sums processes after averaging within each', async () => {
    writeSeries('2026-11-10-7.jsonl', [
      sample(7, T + 1000, { memActive: 100 }),
      sample(7, T + 2000, { memActive: 300 }),
    ]);
    writeSeries('2026-11-10-8.jsonl', [sample(8, T + 1500, { memActive: 1000 })]);
    await ingest();
    const body = await get<Series>(`/api/runtime/series?from=${T}&to=${T + 4999}&bucketMs=5000`);
    expect(body.points[0]).toMatchObject({ samples: 3, memActiveAvg: 1200, memActiveMax: 1300 });
    expect(body.procs).toHaveLength(2);
  });

  it("runtime series: busyMs is the sum of the samples' busyMs", async () => {
    writeSeries('2026-11-10-7.jsonl', [
      sample(7, T + 1000, { busyMs: 700, dOk: 2, dOutTokens: 50 }),
      sample(7, T + 61_000, { busy: false, busyMs: 2000, dOk: 1, dOutTokens: 10 }),
    ]);
    await ingest();
    const body = await get<Series>(`/api/runtime/series?from=${T}&to=${T + 3_599_999}&bucketMs=3600000`);
    expect(body.points).toHaveLength(1);
    expect(body.points[0]).toMatchObject({ busyMs: 2700, okTurns: 3, outTokens: 60 });
    expect((await api.fetch(`/api/runtime/series?from=${T}&to=${T + 1000}&bucketMs=1234`)).status).toBe(400);
    expect((await api.fetch(`/api/runtime/series?from=${T}&bucketMs=5000`)).status).toBe(400);
    expect((await api.fetch(`/api/runtime/series?from=0&to=${T}&bucketMs=5000`)).status).toBe(400);
  });

  it("runtime series: maxBatchOccupancy is the latest snapshot's value and is not in delta", async () => {
    writeSeries('2026-11-10-7.jsonl', [
      sched(T - 5000, { maxBatchOccupancy: 2, globalSteps: 10 }),
      sched(T + 1000, { maxBatchOccupancy: 4, globalSteps: 30 }),
      sched(T + 2000, { maxBatchOccupancy: 4, globalSteps: 35 }),
    ]);
    await ingest();
    const body = await get<Series>(`/api/runtime/series?from=${T}&to=${T + 60_000}&bucketMs=5000`);
    expect(body.sched).toHaveLength(1);
    expect(body.sched[0].latest).toMatchObject({ maxBatchOccupancy: 4, ts: T + 2000 });
    expect(body.sched[0].snapshots).toHaveLength(2);
    expect(body.sched[0].delta.globalSteps).toBe(25);
    expect('maxBatchOccupancy' in body.sched[0].delta).toBe(false);
  });

  it('runtime series: the first snapshot of a process and model counts from zero', async () => {
    writeSeries('2026-11-10-7.jsonl', [
      sched(T + 1000, {
        globalSteps: 500,
        admitted: 40,
        completed: 40,
        occHist: [
          [1, 300],
          [2, 200],
        ],
      }),
    ]);
    await ingest();
    const body = await get<Series>(`/api/runtime/series?from=${T}&to=${T + 60_000}&bucketMs=5000`);
    expect(body.sched).toHaveLength(1);
    expect(body.sched[0].delta).toMatchObject({ globalSteps: 500, admitted: 40, completed: 40, preemptions: 0 });
    expect(body.sched[0].delta.occupancy).toEqual([
      { occupancy: 1, steps: 300 },
      { occupancy: 2, steps: 200 },
    ]);
  });

  it('runtime series: scheduler deltas survive a counter reset', async () => {
    writeSeries('2026-11-10-7.jsonl', [
      sched(T - 5000, { admitted: 50, occHist: [[1, 100]] }),
      sched(T + 1000, {
        admitted: 60,
        occHist: [
          [1, 120],
          [2, 5],
        ],
      }),
      // The model was reloaded: counters restart below the previous value.
      sched(T + 2000, { admitted: 4, occHist: [[1, 3]] }),
    ]);
    await ingest();
    const body = await get<Series>(`/api/runtime/series?from=${T}&to=${T + 60_000}&bucketMs=5000`);
    expect(body.sched[0].delta.admitted).toBe(14);
    expect(body.sched[0].delta.occupancy).toEqual([
      { occupancy: 1, steps: 23 },
      { occupancy: 2, steps: 5 },
    ]);
  });

  it('runtime series: turns grow in the range, restart at a reload, and read as unknown when not written', async () => {
    writeSeries('2026-11-10-7.jsonl', [
      sched(T - 5000, { turns: 3 }),
      sched(T + 1000, { turns: 7 }),
      loadRec(T + 2000, true),
      sched(T + 3000, { turns: 2 }),
    ]);
    writeSeries('2026-11-10-8.jsonl', [{ ...sched(T + 1000, {}), pid: 8, model: 'Old' }]);
    await ingest();
    const body = await get<Series>(`/api/runtime/series?from=${T}&to=${T + 60_000}&bucketMs=5000`);
    const byModel = new Map(body.sched.map((entry) => [entry.model, entry]));
    expect(byModel.get('M')?.latest.turns).toBe(2);
    expect(byModel.get('M')?.delta.turns).toBe(6);
    expect(byModel.get('Old')?.latest.turns).toBeNull();
    expect(byModel.get('Old')?.delta.turns).toBe(0);
  });

  function loadRec(ts: number, ok: boolean): Record<string, unknown> {
    return { ...base1(7, T - 60_000, ts), k: 'load', model: 'M', modelType: 'qwen3_5', capability: 'chat', ms: 10, ok };
  }

  it('runtime series: a successful reload between snapshots restarts the counters', async () => {
    writeSeries('2026-11-10-7.jsonl', [
      sched(T + 1000, { admitted: 10, occHist: [[1, 100]] }),
      loadRec(T + 3000, true),
      sched(T + 4000, { admitted: 15, occHist: [[1, 150]] }),
    ]);
    await ingest();
    const body = await get<Series>(`/api/runtime/series?from=${T}&to=${T + 60_000}&bucketMs=5000`);
    expect(body.sched[0].delta.admitted).toBe(25);
    expect(body.sched[0].delta.occupancy).toEqual([{ occupancy: 1, steps: 250 }]);
  });

  it('runtime series: a reload before the range resets and a failed load does not', async () => {
    writeSeries('2026-11-10-7.jsonl', [
      sched(T - 5000, { admitted: 7 }),
      loadRec(T - 1000, true),
      sched(T + 1000, { admitted: 10 }),
      loadRec(T + 2000, false),
      sched(T + 3000, { admitted: 12 }),
    ]);
    await ingest();
    const body = await get<Series>(`/api/runtime/series?from=${T}&to=${T + 60_000}&bucketMs=5000`);
    expect(body.sched[0].delta.admitted).toBe(12);
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
    trend: Array<{ t: number; hits: number; misses: number; bytesWritten: number; bytesRestored: number }>;
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
        // Every sidecar counter is a per-turn DELTA, so all seven SUM. The two
        // attributed rows carry different values for each so a reducer swapped
        // to MAX changes a number in the assertions below, and the seven sums
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
    // Two local days, two rows — neither the other root's day nor the legacy day.
    expect(body.trend.map((r) => r.t)).toEqual([
      localDayStart(Date.UTC(2026, 6, 20, 10)),
      localDayStart(Date.UTC(2026, 6, 21, 10)),
    ]);
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
  it('cache: failed turns are not counted as legacy turns', async () => {
    writeFileSync(
      join(tracesDir, '2026-07-22-failed.jsonl'),
      ['error', 'cancelled']
        .map(
          (outcome, i) =>
            `${JSON.stringify({ v: 2, traceId: `failed-${i}`, ts: Date.UTC(2026, 6, 22), model: 'm', outcome, endpoint: 'messages', dispatch: 'start', stream: true })}\n`,
        )
        .join(''),
    );
    await ingest();
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.scope.legacy).toEqual({ turns: 1, hits: 5000, misses: 11 });
    expect(body.scope.disabledTurns).toBe(1);
  });

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
  });

  /**
   * The seven `coldSidecar*` counters, which the ingest mapping once dropped
   * on the floor.
   *
   * `sidecarInstalled` is the reason this matters: every
   * `install_*_cold_sidecar` early-return falls through to a full O(prefix)
   * replay that produces CORRECT state, so a regression from "restored and
   * INSTALLED" to "restored and silently re-derived" leaves hits, cached
   * tokens, corruptions and the emitted text all unchanged. Nothing else in
   * the payload moves.
   *
   * All seven are per-turn DELTAS — none has a per-process cumulative twin —
   * so every one SUMS. The fixture's two attributed rows carry different
   * values for each counter, and the seven sums are pairwise distinct, so
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
    expect(((await res.json()) as { error: string }).error).toContain('not available over this transport');
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
    expect(res.ok ? '' : res.message).toContain('shutting down');
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
