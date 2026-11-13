import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import type { RuntimeLiveFile as WriterLiveFile } from '../../server/src/runtime-trace.js';
import type { IngestSummary } from '../src/api/context.js';
import {
  classifyProcess,
  createLiveWatcher,
  DEAD_LISTED_MS,
  deriveEngineStatus,
  LIVE_DEAD_AFTER_MS,
  parseLiveFile,
  parseSupervisorFile,
  pidAlive,
  type LiveEvent,
  type LiveWatcher,
} from '../src/live.js';

const T = Date.UTC(2026, 10, 10, 12);

/** Typed against the writer's own type, so a field renamed on either side fails the typecheck. */
function liveFile(pid: number, extra: Partial<WriterLiveFile> = {}): WriterLiveFile {
  return {
    v: 1,
    pid,
    startedAt: T - 60_000,
    ts: T,
    intervalMs: 1000,
    host: 'desktop',
    version: '0.0.21',
    port: 8765,
    persistPagedCache: true,
    coldCapture: { blocks: 63_125, ms: 10_000 },
    health: 'ok',
    mem: { active: 1, peak: 2, cache: 3, wiredLimit: 4, rss: 5 },
    decodeTps: 42.5,
    loading: null,
    counters: { ok: 1, error: 0, cancelled: 0, outputTokens: 100, loads: 1, rejected: 0 },
    models: [
      {
        name: 'Qwen3.6-35B-A3B-4bit',
        modelType: 'qwen3_5_moe',
        ctxWindow: 262144,
        paged: true,
        maxConcurrent: 4,
        loadedAt: T - 50_000,
        loadMs: 4000,
        inFlight: 1,
        queued: 0,
        queueLimit: 32,
        pool: { at: T - 1000, blockCapacity: 100, allocatedBlocks: 10, freeBlocks: 80, reclaimableBlocks: 10 },
      },
    ],
    turns: [
      {
        traceId: 'tr-1',
        model: 'claude-haiku-4-5',
        modelId: 'Qwen3.6-35B-A3B-4bit',
        ctxWindow: 262144,
        startedAt: T - 2000,
        stream: true,
        dispatch: 'start',
        phase: 'decode',
        promptTokens: 1200,
        cachedTokens: 1000,
        prefillTotal: 200,
        prefillDone: 200,
        generatedTokens: 30,
      },
    ],
    ...extra,
  };
}

function supervisorFile(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    version: 1,
    writtenAt: T,
    mainPid: 4242,
    state: 'running',
    pid: 777,
    generation: 2,
    streak: 0,
    shortLivedFailures: 0,
    lastExit: { verdict: 'crash', code: null, signal: 'SIGKILL', atMs: T - 9000 },
    healthStatus: 'ok',
    nativeErrorCount: 20,
    nativeErrorsCapped: true,
    lastNativeErrorContext: 'array_eval',
    lastNativeErrorAt: T - 100,
    ...extra,
  };
}

function summary(traces: number, runtime: number): IngestSummary {
  return {
    traces: { files: 0, records: traces, pruned: 0, warnings: [] },
    runtime: { files: 0, records: runtime, pruned: 0, warnings: [] },
  };
}

async function waitFor(check: () => boolean, ms = 3000): Promise<void> {
  const deadline = Date.now() + ms;
  while (!check()) {
    if (Date.now() > deadline) throw new Error('timed out waiting');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

let base: string;
let runtimeDir: string;
let supervisorPath: string;
let watcher: LiveWatcher | undefined;

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'dash-live-'));
  runtimeDir = join(base, 'runtime');
  mkdirSync(runtimeDir, { recursive: true });
  supervisorPath = join(base, 'desktop', 'supervisor.json');
});

afterEach(() => {
  watcher?.close();
  watcher = undefined;
  rmSync(base, { recursive: true, force: true });
});

function writeLive(pid: number, file: unknown): void {
  const path = join(runtimeDir, `live-${pid}.json`);
  writeFileSync(path, typeof file === 'string' ? file : JSON.stringify(file));
  // A fresh mtime each write: the watcher re-reads a file only when its stat changes.
  const t = (Date.now() + Math.random() * 1000) / 1000;
  utimesSync(path, t, t);
}

describe('live: judging a writer', () => {
  it('classifies running, unresponsive and dead writers', () => {
    const alive = (): boolean => true;
    const dead = (): boolean => false;
    const file = { pid: 1, ts: T, intervalMs: 1000 };
    expect(classifyProcess(file, T + 4500, alive)).toBe('running');
    expect(classifyProcess(file, T + 4501, alive)).toBe('unresponsive');
    expect(classifyProcess({ ...file, intervalMs: 5000 }, T + 16_500, alive)).toBe('running');
    expect(classifyProcess(file, T + 100, dead)).toBe('dead');
  });

  it('a stale file with a live pid turns dead after the hard limit', () => {
    const file = { pid: 1, ts: T, intervalMs: 1000 };
    expect(classifyProcess(file, T + LIVE_DEAD_AFTER_MS - 1, () => true)).toBe('unresponsive');
    expect(classifyProcess(file, T + LIVE_DEAD_AFTER_MS, () => true)).toBe('dead');
  });

  it('derives the headline status in precedence order', () => {
    const run = { status: 'running' as const, health: 'ok' as const };
    const loading = { status: 'running' as const, health: 'loading' as const };
    const stuck = { status: 'unresponsive' as const, health: 'ok' as const };
    const gone = { status: 'dead' as const, health: 'ok' as const };
    expect(deriveEngineStatus({ state: 'lying' }, [loading, run])).toBe('untrusted');
    expect(deriveEngineStatus({ state: 'restarting' }, [run, loading])).toBe('loading');
    expect(deriveEngineStatus({ state: 'failed' }, [run])).toBe('running');
    expect(deriveEngineStatus({ state: 'restarting' }, [stuck])).toBe('restarting');
    expect(deriveEngineStatus({ state: 'starting' }, [])).toBe('starting');
    expect(deriveEngineStatus({ state: 'failed' }, [gone])).toBe('failed');
    expect(deriveEngineStatus(null, [stuck, gone])).toBe('unresponsive');
    expect(deriveEngineStatus({ state: 'stopped' }, [gone])).toBe('stopped');
    expect(deriveEngineStatus(null, [])).toBe('stopped');
  });

  it('ignores a supervisor file whose main pid is dead', async () => {
    expect(parseSupervisorFile(supervisorFile(), () => true)).toMatchObject({
      state: 'running',
      nativeErrorsCapped: true,
      lastExit: { verdict: 'crash', signal: 'SIGKILL' },
    });
    expect(parseSupervisorFile(supervisorFile(), () => false)).toBeNull();
    expect(parseSupervisorFile(supervisorFile({ state: 'zombie' }), () => true)).toBeNull();
    expect(parseSupervisorFile(supervisorFile({ streak: undefined }), () => true)).toBeNull();

    mkdirSync(join(base, 'desktop'), { recursive: true });
    writeFileSync(supervisorPath, JSON.stringify(supervisorFile({ state: 'failed' })));
    const live = (mainAlive: boolean): LiveWatcher =>
      createLiveWatcher({
        runtimeDir,
        supervisorPath,
        ingest: async () => summary(0, 0),
        now: () => T,
        isAlive: (pid) => (pid === 4242 ? mainAlive : true),
      });
    const withMain = live(true);
    const withoutMain = live(false);
    try {
      expect((await withMain.readNow()).status).toBe('failed');
      const snapshot = await withoutMain.readNow();
      expect(snapshot.supervisor).toBeNull();
      expect(snapshot.status).toBe('stopped');
    } finally {
      withMain.close();
      withoutMain.close();
    }
  });

  it('the default liveness probe does not import the server package', () => {
    const source = readFileSync(new URL('../src/live.ts', import.meta.url), 'utf8');
    const specifiers: string[] = [];
    for (const line of source.split('\n')) {
      if (!line.startsWith('import ')) continue;
      const open = line.lastIndexOf("'", line.length - 3);
      const close = line.lastIndexOf("'");
      if (open >= 0 && close > open) specifiers.push(line.slice(open + 1, close));
    }
    expect(specifiers.length).toBeGreaterThan(0);
    for (const spec of specifiers) expect(spec.startsWith('@mlx-node/')).toBe(false);
    expect(pidAlive(process.pid)).toBe(true);
    expect(pidAlive(2 ** 30)).toBe(false);
  });
});

describe('live: watcher', () => {
  it('reads a heartbeat into a process row', async () => {
    writeLive(31, liveFile(31));
    watcher = createLiveWatcher({
      runtimeDir,
      supervisorPath,
      ingest: async () => summary(0, 0),
      now: () => T + 500,
      isAlive: () => true,
    });
    const snapshot = await watcher.readNow();
    expect(snapshot.status).toBe('running');
    expect(snapshot.supervisor).toBeNull();
    expect(snapshot.processes).toHaveLength(1);
    const { v: _v, ...fields } = liveFile(31);
    expect(snapshot.processes[0]).toEqual({ ...fields, status: 'running', ageMs: 500 });
  });

  it('reports whether the engine has the disk cache on, and null when the engine does not say', () => {
    expect(parseLiveFile(liveFile(31))?.persistPagedCache).toBe(true);
    expect(parseLiveFile(liveFile(31, { persistPagedCache: false }))?.persistPagedCache).toBe(false);
    const { persistPagedCache: _omitted, ...older } = liveFile(31);
    expect(parseLiveFile(older)?.persistPagedCache).toBeNull();
  });

  it('reads the capture limit, and null when the cache is off, the engine does not say, or the value is malformed', () => {
    expect(parseLiveFile(liveFile(31))?.coldCapture).toEqual({ blocks: 63_125, ms: 10_000 });
    expect(parseLiveFile(liveFile(31, { coldCapture: null }))?.coldCapture).toBeNull();
    const { coldCapture: _omitted, ...older } = liveFile(31);
    expect(parseLiveFile(older)?.coldCapture).toBeNull();
    expect(parseLiveFile({ ...liveFile(31), coldCapture: { blocks: '128', ms: 250 } })?.coldCapture).toBeNull();
  });

  it('replays the latest snapshot to a new subscriber', async () => {
    writeLive(31, liveFile(31));
    watcher = createLiveWatcher({
      runtimeDir,
      supervisorPath,
      ingest: async () => summary(0, 0),
      now: () => T,
      isAlive: () => true,
      pollMs: 10,
    });
    const first: LiveEvent[] = [];
    const stop = watcher.subscribe((e) => first.push(e));
    await waitFor(() => first.length > 0);
    const late: LiveEvent[] = [];
    const stopLate = watcher.subscribe((e) => late.push(e));
    expect(late).toHaveLength(1);
    expect(late[0]).toMatchObject({ type: 'runtime', snapshot: { status: 'running' } });
    stop();
    stopLate();
  });

  it('does not replay an old snapshot after every subscriber left', async () => {
    writeLive(31, liveFile(31));
    let clock = T;
    watcher = createLiveWatcher({
      runtimeDir,
      supervisorPath,
      ingest: async () => summary(0, 0),
      now: () => clock,
      isAlive: () => true,
      pollMs: 10,
    });
    const first: LiveEvent[] = [];
    const stop = watcher.subscribe((e) => first.push(e));
    await waitFor(() => first.length > 0);
    stop();

    clock = T + 600_000;
    const late: LiveEvent[] = [];
    const stopLate = watcher.subscribe((e) => late.push(e));
    expect(late).toEqual([]);
    await waitFor(() => late.length > 0);
    expect(late[0]).toMatchObject({ type: 'runtime', snapshot: { now: T + 600_000 } });
    stopLate();
  });

  it('polls only while subscribed', async () => {
    let ingests = 0;
    let counter = 1;
    writeLive(31, liveFile(31));
    watcher = createLiveWatcher({
      runtimeDir,
      supervisorPath,
      ingest: async () => {
        ingests++;
        return summary(0, 0);
      },
      now: () => T,
      isAlive: () => true,
      pollMs: 10,
    });
    const bump = (): void => {
      counter++;
      writeLive(
        31,
        liveFile(31, { counters: { ok: counter, error: 0, cancelled: 0, outputTokens: 0, loads: 1, rejected: 0 } }),
      );
    };
    bump();
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(ingests).toBe(0);

    const events: LiveEvent[] = [];
    const stop = watcher.subscribe((e) => events.push(e));
    await waitFor(() => events.length > 0);
    stop();
    const seen = events.length;
    bump();
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(events).toHaveLength(seen);
    expect(ingests).toBe(0);
  });

  it('triggers one coalesced ingest when turn counters move', async () => {
    let ingests = 0;
    let counter = 1;
    const write = (): void =>
      writeLive(
        31,
        liveFile(31, { counters: { ok: counter, error: 0, cancelled: 0, outputTokens: 0, loads: 1, rejected: 0 } }),
      );
    write();
    watcher = createLiveWatcher({
      runtimeDir,
      supervisorPath,
      ingest: async () => {
        ingests++;
        return summary(3, 0);
      },
      now: () => Date.now(),
      isAlive: () => true,
      pollMs: 10,
    });
    const events: LiveEvent[] = [];
    const stop = watcher.subscribe((e) => events.push(e));
    await waitFor(() => events.length > 0);
    expect(ingests).toBe(0);

    for (let i = 0; i < 3; i++) {
      counter++;
      write();
      await new Promise((resolve) => setTimeout(resolve, 40));
    }
    await waitFor(() => events.some((e) => e.type === 'data'));
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(ingests).toBe(1);
    expect(events.filter((e) => e.type === 'data')).toEqual([
      { type: 'data', version: 1, traces: true, runtime: false },
    ]);
    stop();
  });

  it('reports an ingest run by someone else only when it wrote records', () => {
    watcher = createLiveWatcher({ runtimeDir, supervisorPath, ingest: async () => summary(0, 0) });
    const events: LiveEvent[] = [];
    const stop = watcher.subscribe((e) => events.push(e));
    watcher.notifyIngested(summary(0, 0));
    watcher.notifyIngested(summary(0, 2));
    expect(events.filter((e) => e.type === 'data')).toEqual([
      { type: 'data', version: 1, traces: false, runtime: true },
    ]);
    stop();
  });

  it("unlinks a dead writer's file after 60 s", async () => {
    let now = T;
    writeLive(31, liveFile(31));
    watcher = createLiveWatcher({
      runtimeDir,
      supervisorPath,
      ingest: async () => summary(0, 0),
      now: () => now,
      isAlive: () => false,
    });
    const first = await watcher.readNow();
    expect(first.processes.map((p) => p.status)).toEqual(['dead']);
    now = T + DEAD_LISTED_MS - 1;
    expect((await watcher.readNow()).processes).toHaveLength(1);
    now = T + DEAD_LISTED_MS;
    expect((await watcher.readNow()).processes).toHaveLength(0);
    expect(existsSync(join(runtimeDir, 'live-31.json'))).toBe(false);
  });

  it('a file that fails to parse is skipped, not treated as gone', async () => {
    writeLive(31, liveFile(31));
    watcher = createLiveWatcher({
      runtimeDir,
      supervisorPath,
      ingest: async () => summary(0, 0),
      now: () => T,
      isAlive: () => true,
    });
    expect((await watcher.readNow()).processes).toHaveLength(1);

    writeLive(31, '{"v":1,"pid":31,"trunc');
    writeLive(32, 'not json');
    writeLive(33, liveFile(33, { host: 'cloud' as never }));
    writeFileSync(join(runtimeDir, 'live-34.json.tmp'), JSON.stringify(liveFile(34)));
    const snapshot = await watcher.readNow();
    expect(snapshot.processes.map((p) => p.pid)).toEqual([31]);
    expect(snapshot.processes[0].counters.ok).toBe(1);
  });

  it('orders running, then unresponsive, then dead; newest first within each', async () => {
    writeLive(1, liveFile(1, { startedAt: T - 3000 }));
    writeLive(2, liveFile(2, { startedAt: T - 1000 }));
    writeLive(3, liveFile(3, { ts: T - 10_000 }));
    writeLive(4, liveFile(4));
    watcher = createLiveWatcher({
      runtimeDir,
      supervisorPath,
      ingest: async () => summary(0, 0),
      now: () => T,
      isAlive: (pid) => pid !== 4,
    });
    const snapshot = await watcher.readNow();
    expect(snapshot.processes.map((p) => [p.pid, p.status])).toEqual([
      [2, 'running'],
      [1, 'running'],
      [3, 'unresponsive'],
      [4, 'dead'],
    ]);
  });
});
