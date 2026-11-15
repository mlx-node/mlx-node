import {
  appendFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  rmSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  RuntimeTrace,
  type RuntimeProcRecord,
  type RuntimeSampleRecord,
  type RuntimeSchedRecord,
} from '../../server/src/runtime-trace.js';
import { openDashboardDb, type DashboardDb } from '../src/db/open.js';
import { runtimeFiles, runtimeLoads, runtimeProcs, runtimeSamples, runtimeSched } from '../src/db/schema.js';
import { ingestRuntime } from '../src/ingest/runtime.js';

const DAY_MS = 86_400_000;
const T0 = Date.UTC(2026, 10, 10, 9, 0, 0);
const PID = 31337;
const STARTED = T0 - 1000;

const PROC: Omit<RuntimeProcRecord, 'ts'> = {
  v: 1,
  k: 'proc',
  pid: PID,
  startedAt: STARTED,
  host: 'desktop',
  version: '0.0.21',
  port: 8765,
  wiredLimitBytes: 100_000_000_000,
  totalMemBytes: 128_000_000_000,
  chip: 'Apple M5 Max',
};

function sample(ts: number, extra: Partial<RuntimeSampleRecord> = {}): RuntimeSampleRecord {
  return {
    v: 1,
    k: 'sample',
    ts,
    pid: PID,
    startedAt: STARTED,
    dtMs: 5000,
    busy: true,
    busyMs: 4000,
    health: 'ok',
    memActive: 10,
    memPeak: 20,
    memCache: 5,
    rss: 300,
    inFlight: 2,
    queued: 1,
    resident: 1,
    dOk: 3,
    dError: 1,
    dCancelled: 0,
    dOutTokens: 400,
    dRejected: 2,
    ...extra,
  };
}

function sched(ts: number): RuntimeSchedRecord {
  return {
    v: 1,
    k: 'sched',
    ts,
    pid: PID,
    startedAt: STARTED,
    model: 'Qwen3.6-35B-A3B-4bit',
    turns: 9,
    blockCapacity: 1000,
    allocatedBlocks: 10,
    freeBlocks: 900,
    reclaimableBlocks: 90,
    watermarkBlocks: 20,
    reservedBlocks: 5,
    memoryCapacityBytes: 8_000_000_000,
    memoryWatermarkBytes: 160_000_000,
    reservedBlockBytes: 40_000_000,
    reservedStateBytes: 1_000_000,
    globalSteps: 500,
    admitted: 7,
    completed: 7,
    maxBatchOccupancy: 3,
    fusedGreedyEpilogueSteps: 0,
    admissionDeferredBlocks: 1,
    admissionDeferredState: 0,
    rowsAllocEvicted: 0,
    preemptions: 2,
    preemptionsRecompute: 1,
    preemptionsSsd: 1,
    ssdRestoreWaiting: 0,
    ssdRestoreBytes: 4096,
    ssdRestoreWaitMs: 12.5,
    occHist: [
      [1, 300],
      [2, 200],
    ],
  };
}

/** The real inference-side sink, so the reader is pinned to the writer's bytes. The suite env turns metrics off. */
function sinkAt(dir: string): RuntimeTrace {
  const saved = process.env.MLX_AGENT_METRICS;
  process.env.MLX_AGENT_METRICS = '1';
  try {
    return new RuntimeTrace({ dir, proc: PROC });
  } finally {
    if (saved === undefined) delete process.env.MLX_AGENT_METRICS;
    else process.env.MLX_AGENT_METRICS = saved;
  }
}

async function writeSeries(dir: string): Promise<void> {
  const sink = sinkAt(dir);
  sink.append(sample(T0));
  sink.append(sched(T0 + 1000));
  sink.append({
    v: 1,
    k: 'load',
    ts: T0 + 2000,
    pid: PID,
    startedAt: STARTED,
    model: 'Qwen3.6-35B-A3B-4bit',
    modelType: 'qwen3_5_moe',
    capability: 'chat',
    ms: 4321.5,
    ok: true,
  });
  sink.append(sample(T0 + 5000, { busy: false, busyMs: 1200 }));
  sink.append({ v: 1, k: 'exit', ts: T0 + 9000, pid: PID, startedAt: STARTED, uptimeMs: 10_000 });
  await sink.close();
}

let dash: DashboardDb;
let base: string;
let dir: string;

beforeEach(() => {
  dash = openDashboardDb(':memory:');
  base = mkdtempSync(join(tmpdir(), 'dash-runtime-'));
  dir = join(base, 'runtime');
});

afterEach(() => {
  dash.close();
  rmSync(base, { recursive: true, force: true });
});

describe('ingestRuntime', () => {
  it('ingests proc, sample, sched, load and exit records', async () => {
    await writeSeries(dir);
    const res = await ingestRuntime(dash, dir);
    expect(res).toMatchObject({ files: 1, records: 6, pruned: 0, warnings: [] });

    const procs = dash.db.select().from(runtimeProcs).all();
    expect(procs).toEqual([
      {
        procKey: `${STARTED}-${PID}`,
        pid: PID,
        startedAt: STARTED,
        host: 'desktop',
        version: '0.0.21',
        port: 8765,
        chip: 'Apple M5 Max',
        wiredLimitBytes: 100_000_000_000,
        totalMemBytes: 128_000_000_000,
        firstTs: T0,
        lastTs: T0 + 9000,
        exitTs: T0 + 9000,
      },
    ]);
    const samples = dash.db.select().from(runtimeSamples).all();
    expect(samples).toHaveLength(2);
    expect(samples[0]).toMatchObject({
      ts: T0,
      dtMs: 5000,
      busy: 1,
      busyMs: 4000,
      health: 'ok',
      memActive: 10,
      memPeak: 20,
      memCache: 5,
      rss: 300,
      inFlight: 2,
      queued: 1,
      resident: 1,
      dOk: 3,
      dError: 1,
      dCancelled: 0,
      dOutTokens: 400,
      dRejected: 2,
    });
    expect(samples[1]).toMatchObject({ busy: 0, busyMs: 1200 });
    const snaps = dash.db.select().from(runtimeSched).all();
    expect(snaps).toHaveLength(1);
    expect(snaps[0]).toMatchObject({
      model: 'Qwen3.6-35B-A3B-4bit',
      blockCapacity: 1000,
      maxBatchOccupancy: 3,
      preemptionsSsd: 1,
      ssdRestoreWaitMs: 12.5,
      turns: 9,
      occHist: '[[1,300],[2,200]]',
    });
    const loads = dash.db.select().from(runtimeLoads).all();
    expect(loads).toHaveLength(1);
    expect(loads[0]).toMatchObject({ model: 'Qwen3.6-35B-A3B-4bit', capability: 'chat', ms: 4321.5, ok: 1 });

    const again = await ingestRuntime(dash, dir);
    expect(again.files).toBe(0);
    expect(dash.db.select().from(runtimeSamples).all()).toHaveLength(2);
  });

  it('stores no turn count for a scheduler snapshot of an engine that did not write one', async () => {
    const sink = sinkAt(dir);
    const old: Partial<RuntimeSchedRecord> = sched(T0 + 1000);
    delete old.turns;
    sink.append(old as RuntimeSchedRecord);
    await sink.close();
    await ingestRuntime(dash, dir);
    expect(
      dash.db
        .select()
        .from(runtimeSched)
        .all()
        .map((row) => row.turns),
    ).toEqual([null]);
  });

  it('appends only new records on the next pass', async () => {
    await writeSeries(dir);
    await ingestRuntime(dash, dir);
    const sink = sinkAt(dir);
    sink.append(sample(T0 + 20_000));
    await sink.close();
    const later = Date.now() / 1000 + 5;
    const [name] = readdirSync(dir);
    utimesSync(join(dir, name), later, later);

    const res = await ingestRuntime(dash, dir);
    // A new sink writes its own proc header before the sample.
    expect(res.records).toBe(2);
    expect(dash.db.select().from(runtimeSamples).all()).toHaveLength(3);
    expect(dash.db.select().from(runtimeProcs).all()).toHaveLength(1);
    expect(dash.db.select().from(runtimeProcs).all()[0].lastTs).toBe(T0 + 20_000);
  });

  it('prunes files older than 14 days', async () => {
    await writeSeries(dir);
    const [name] = readdirSync(dir);
    const file = join(dir, name);
    const old = (Date.now() - 15 * DAY_MS) / 1000;
    utimesSync(file, old, old);

    const res = await ingestRuntime(dash, dir);
    expect(res.pruned).toBe(1);
    expect(existsSync(file)).toBe(false);
    expect(dash.db.select().from(runtimeSamples).all()).toHaveLength(0);
    expect(dash.db.select().from(runtimeFiles).all()).toHaveLength(0);
  });

  it('removes process rows with no remaining data', async () => {
    await writeSeries(dir);
    await ingestRuntime(dash, dir);
    expect(dash.db.select().from(runtimeProcs).all()).toHaveLength(1);

    for (const name of readdirSync(dir)) rmSync(join(dir, name));
    await ingestRuntime(dash, dir);
    expect(dash.db.select().from(runtimeProcs).all()).toHaveLength(0);
    expect(dash.db.select().from(runtimeSched).all()).toHaveLength(0);
    expect(dash.db.select().from(runtimeLoads).all()).toHaveLength(0);
    expect(dash.db.select().from(runtimeFiles).all()).toHaveLength(0);
  });

  it('keeps a process row whose first pass read only the proc header', async () => {
    mkdirSync(dir, { recursive: true });
    const file = join(dir, `2026-11-10-${PID}.jsonl`);
    writeFileSync(file, `${JSON.stringify({ ...PROC, ts: T0 })}\n`);
    expect((await ingestRuntime(dash, dir)).records).toBe(1);

    appendFileSync(file, `${JSON.stringify(sample(T0 + 5000))}\n`);
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);
    expect((await ingestRuntime(dash, dir)).records).toBe(1);

    const procs = dash.db.select().from(runtimeProcs).all();
    expect(procs).toHaveLength(1);
    expect(procs[0]).toMatchObject({
      host: 'desktop',
      version: '0.0.21',
      port: 8765,
      chip: 'Apple M5 Max',
      wiredLimitBytes: 100_000_000_000,
      totalMemBytes: 128_000_000_000,
      firstTs: T0,
      lastTs: T0 + 5000,
    });
  });

  it('keeps a process row whose rewritten file now holds only its proc header', async () => {
    await writeSeries(dir);
    await ingestRuntime(dash, dir);
    const [name] = readdirSync(dir);
    const file = join(dir, name);
    writeFileSync(file, `${JSON.stringify({ ...PROC, ts: T0 + 20_000 })}\n`);
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    expect((await ingestRuntime(dash, dir)).records).toBe(1);
    expect(dash.db.select().from(runtimeSamples).all()).toHaveLength(0);
    const procs = dash.db.select().from(runtimeProcs).all();
    expect(procs).toHaveLength(1);
    expect(procs[0]).toMatchObject({ host: 'desktop', version: '0.0.21', port: 8765, chip: 'Apple M5 Max' });
  });

  it('skips live files and tmp files', async () => {
    mkdirSync(dir, { recursive: true });
    const line = `${JSON.stringify(sample(T0))}\n`;
    writeFileSync(join(dir, `live-${PID}.json`), line);
    writeFileSync(join(dir, `live-${PID}.json.tmp`), line);
    writeFileSync(join(dir, '2026-11-10-1.jsonl.tmp'), line);

    const res = await ingestRuntime(dash, dir);
    expect(res.files).toBe(0);
    expect(dash.db.select().from(runtimeSamples).all()).toHaveLength(0);
    expect(dash.db.select().from(runtimeFiles).all()).toHaveLength(0);
  });

  it('symlinked root skips the pass', async () => {
    const real = join(base, 'elsewhere');
    await writeSeries(real);
    const [name] = readdirSync(real);
    const old = (Date.now() - 30 * DAY_MS) / 1000;
    utimesSync(join(real, name), old, old);
    symlinkSync(real, dir);

    const res = await ingestRuntime(dash, dir);
    expect(res).toEqual({ files: 0, records: 0, pruned: 0, warnings: [] });
    expect(existsSync(join(real, name))).toBe(true);
  });

  it('clears every runtime table when the directory is gone', async () => {
    await writeSeries(dir);
    await ingestRuntime(dash, dir);
    rmSync(dir, { recursive: true });
    await ingestRuntime(dash, dir);
    expect(dash.db.select().from(runtimeProcs).all()).toHaveLength(0);
    expect(dash.db.select().from(runtimeSamples).all()).toHaveLength(0);
    expect(dash.db.select().from(runtimeFiles).all()).toHaveLength(0);
  });
});
