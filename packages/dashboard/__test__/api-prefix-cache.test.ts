import { closeSync, existsSync, lstatSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import type { RuntimeLiveFile as WriterLiveFile } from '../../server/src/runtime-trace.js';
import type { ApiResponse } from '../src/api/errors.js';
import { prefixCacheRoots } from '../src/api/handlers/prefix-cache.js';
import { LIVE_DEAD_AFTER_MS } from '../src/live.js';
import { takeNamespaceLock, type PrefixCacheResponse } from '../src/prefix-cache.js';
import { createDashboardRuntime, type DashboardRuntime } from '../src/runtime.js';

const NS_A = '0123456789abcdef0123456789abcdef';
const NS_B = 'fedcba9876543210fedcba9876543210';
const DARWIN = process.platform === 'darwin';

let base: string;
let root: string;
let other: string;
let runtimeDir: string;
let runtime: DashboardRuntime;
const held: number[] = [];

function heartbeat(
  ts: number,
  stateCache: WriterLiveFile['stateCache'],
  models: WriterLiveFile['models'] = [],
  pid: number = process.pid,
): void {
  const file: WriterLiveFile = {
    v: 3,
    pid,
    startedAt: ts - 60_000,
    ts,
    intervalMs: 5000,
    host: 'serve',
    version: '0.0.21',
    port: 8080,
    persistPagedCache: true,
    coldCapture: null,
    stateCache,
    health: 'ok',
    mem: { active: 1, peak: 2, cache: 3, wiredLimit: 4, rss: 5 },
    work: { ms: { loading: 0, waiting: 0, prepare: 0, prefill: 0, decode: 0, running: 0 } },
    rates: { windowMs: 10_000, outputTps: null, prefillTps: null },
    loading: [],
    counters: { ok: 0, error: 0, cancelled: 0, outputTokens: 0, loads: 0, rejected: 0 },
    models,
    turns: [],
  };
  mkdirSync(runtimeDir, { recursive: true });
  writeFileSync(join(runtimeDir, `live-${String(pid)}.json`), JSON.stringify(file));
}

function namespace(at: string, name: string, files: Record<string, number>): string {
  const dir = join(at, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'lock'), '');
  for (const [file, bytes] of Object.entries(files)) writeFileSync(join(dir, file), Buffer.alloc(bytes));
  return dir;
}

async function list(): Promise<PrefixCacheResponse> {
  const response = await runtime.call({ method: 'GET', path: '/api/cache/prefix' });
  if (!response.ok) throw new Error(response.message);
  return response.body as PrefixCacheResponse;
}

function remove(name: string, body: unknown): Promise<ApiResponse> {
  return runtime.call({ method: 'DELETE', path: `/api/cache/prefix/${name}`, body });
}

function openedOf(dir: string): number {
  return lstatSync(join(dir, 'lock')).mtimeMs;
}

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'dash-api-prefix-'));
  root = join(base, 'root');
  other = join(base, 'other');
  mkdirSync(root);
  mkdirSync(other);
  const tracesDir = join(base, 'metrics', 'traces');
  mkdirSync(tracesDir, { recursive: true });
  runtimeDir = join(base, 'metrics', 'runtime');
  runtime = createDashboardRuntime({
    chat: { tempRoots: [], root: join(base, 'chat'), homeDir: join(base, 'home') },
    agentHome: join(base, 'agent-home'),
    dbPath: ':memory:',
    tracesDir,
    modelsDir: join(base, 'models'),
    prefixCacheRoot: root,
  });
});

afterEach(async () => {
  for (const fd of held.splice(0)) closeSync(fd);
  await runtime.close();
  rmSync(base, { recursive: true, force: true });
});

describe('GET /api/cache/prefix', () => {
  it("lists the configured root and a live engine's cache dir", async () => {
    namespace(root, NS_A, { a: 10 });
    namespace(other, NS_B, { b: 20 });
    heartbeat(Date.now(), { dir: other, limitBytes: 123 });
    const body = await list();
    expect(body.roots.map((r) => [r.root, r.isDefault, r.exists])).toEqual([
      [root, true, true],
      [other, false, true],
    ]);
    expect(body.roots[0]!.namespaces.map((ns) => [ns.namespace, ns.bytes])).toEqual([[NS_A, 10]]);
    expect(body.roots[0]!.engines).toEqual([]);
    expect(body.roots[1]!.namespaces.map((ns) => [ns.namespace, ns.bytes])).toEqual([[NS_B, 20]]);
    expect(body.roots[1]!.engines).toEqual([{ pid: process.pid, host: 'serve', limitBytes: 123 }]);
    expect(body.counting).toBe(false);
  });

  it('names the model of a cache a running engine reports, and no other', async () => {
    namespace(root, NS_A, { a: 1 });
    namespace(root, NS_B, { b: 1 });
    const row: WriterLiveFile['models'][number] = {
      name: 'Qwen3.6-27B',
      modelType: 'qwen3_5',
      ctxWindow: 1,
      paged: true,
      maxConcurrent: 1,
      loadedAt: null,
      loadMs: null,
      inFlight: 0,
      queued: 0,
      queueLimit: null,
      pool: null,
      stateNamespace: NS_A,
      experts: null,
      images: null,
    };
    heartbeat(Date.now(), { dir: null, limitBytes: null }, [row, { ...row, name: 'Qwen3-8B', stateNamespace: null }]);
    const body = await list();
    expect(body.roots[0]!.namespaces.map((ns) => [ns.namespace, ns.model])).toEqual([
      [NS_A, 'Qwen3.6-27B'],
      [NS_B, null],
    ]);
  });

  it('dead heartbeat files do not hide a live engine', () => {
    const live = 999_999;
    for (let pid = 1; pid <= 40; pid += 1)
      heartbeat(Date.now(), { dir: join(base, 'dead'), limitBytes: null }, [], pid);
    heartbeat(Date.now(), { dir: other, limitBytes: 7 }, [], live);
    const roots = prefixCacheRoots({ prefixCacheRoot: root, runtimeDir }, Date.now(), (pid) => pid === live);
    expect(roots.map((r) => [r.root, r.engines.map((e) => e.pid)])).toEqual([
      [root, []],
      [other, [live]],
    ]);
  });

  it("a dead engine's cache dir is not listed", async () => {
    heartbeat(Date.now() - LIVE_DEAD_AFTER_MS - 1000, { dir: other, limitBytes: 123 });
    const body = await list();
    expect(body.roots.map((r) => r.root)).toEqual([root]);
  });
});

describe('a folder an engine names that holds no model cache', () => {
  it.runIf(DARWIN)('lists no folder without a lock and refuses to delete it', async () => {
    const documents = join(base, 'Documents');
    const folder = join(documents, NS_A);
    mkdirSync(folder, { recursive: true });
    writeFileSync(join(folder, 'report.pdf'), 'keep');
    heartbeat(Date.now(), { dir: documents, limitBytes: null });
    const body = await list();
    expect(body.roots.map((r) => [r.root, r.namespaces.length])).toEqual([
      [root, 0],
      [documents, 0],
    ]);
    const response = await remove(NS_A, { root: documents, openedMs: 0 });
    expect(response).toMatchObject({ ok: false, status: 404, reason: 'not-a-cache' });
    expect(readdirSync(folder)).toEqual(['report.pdf']);
  });
});

describe('DELETE /api/cache/prefix/:namespace', () => {
  it('refuses a root the runtime does not list', async () => {
    const dir = namespace(other, NS_B, { b: 1 });
    const response = await remove(NS_B, { root: other, openedMs: openedOf(dir) });
    expect(response).toMatchObject({ ok: false, status: 403 });
    expect(readdirSync(dir).sort()).toEqual(['b', 'lock']);
  });

  it.runIf(DARWIN)('answers 409 while the lock is held and 404 for a missing cache', async () => {
    const dir = namespace(root, NS_A, { a: 1 });
    held.push(takeNamespaceLock(dir));
    const busy = await remove(NS_A, { root, openedMs: openedOf(dir) });
    expect(busy).toMatchObject({ ok: false, status: 409, reason: 'held' });
    expect(readdirSync(dir).sort()).toEqual(['a', 'lock']);
    const gone = await remove(NS_B, { root, openedMs: 1 });
    expect(gone).toMatchObject({ ok: false, status: 404 });
    closeSync(held.pop()!);
    const changed = await remove(NS_A, { root, openedMs: openedOf(dir) - 1000 });
    expect(changed).toMatchObject({ ok: false, status: 409, reason: 'changed' });
    const done = await remove(NS_A, { root, openedMs: openedOf(dir) });
    expect(done).toMatchObject({ ok: true, body: { removed: 1, done: true, directoryRemoved: true } });
    expect(existsSync(dir)).toBe(false);
  });

  it('refuses a body without the listed open time', async () => {
    const dir = namespace(root, NS_A, { a: 1 });
    for (const body of [{ root }, { root, openedMs: 'x' }, { root, openedMs: null }, { openedMs: 1 }, null]) {
      expect(await remove(NS_A, body)).toMatchObject({ ok: false, status: 400 });
    }
    expect(await remove('not-a-namespace', { root, openedMs: 1 })).toMatchObject({ ok: false, status: 400 });
    expect(readdirSync(dir).sort()).toEqual(['a', 'lock']);
  });
});
