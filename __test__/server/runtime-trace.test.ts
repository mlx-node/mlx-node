import {
  appendFileSync,
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  truncateSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  RUNTIME_FILE_MAX_BYTES,
  RuntimeTrace,
  SINK_QUEUE_MAX,
  type RuntimeProcRecord,
  type RuntimeSampleRecord,
  type RuntimeTraceRecord,
} from '../../packages/server/src/runtime-trace.js';

const DAY1 = Date.UTC(2026, 6, 21, 23, 59, 0);
const DAY2 = Date.UTC(2026, 6, 22, 0, 1, 0);

const PROC: Omit<RuntimeProcRecord, 'ts'> = {
  v: 1,
  k: 'proc',
  pid: process.pid,
  startedAt: DAY1 - 1000,
  host: 'serve',
  version: '0.0.0-test',
  port: 8080,
  wiredLimitBytes: 1,
  totalMemBytes: 2,
  chip: 'Test Chip',
};

function sample(ts: number, overrides: Partial<RuntimeSampleRecord> = {}): RuntimeSampleRecord {
  return {
    v: 1,
    k: 'sample',
    ts,
    pid: process.pid,
    startedAt: PROC.startedAt,
    dtMs: 0,
    busy: false,
    busyMs: 0,
    health: 'ok',
    memActive: 0,
    memPeak: 0,
    memCache: 0,
    rss: 0,
    inFlight: 0,
    queued: 0,
    resident: 0,
    dOk: 0,
    dError: 0,
    dCancelled: 0,
    dOutTokens: 0,
    dRejected: 0,
    ...overrides,
  };
}

function load(ts: number): RuntimeTraceRecord {
  return {
    v: 1,
    k: 'load',
    ts,
    pid: process.pid,
    startedAt: PROC.startedAt,
    model: 'm',
    modelType: 'qwen3_5',
    capability: 'chat',
    ms: 10,
    ok: true,
  };
}

function exit(ts: number): RuntimeTraceRecord {
  return { v: 1, k: 'exit', ts, pid: process.pid, startedAt: PROC.startedAt, uptimeMs: 5 };
}

function readRecords(file: string): RuntimeTraceRecord[] {
  return readFileSync(file, 'utf8')
    .split('\n')
    .filter((line) => line.startsWith('{'))
    .map((line) => JSON.parse(line) as RuntimeTraceRecord);
}

describe('RuntimeTrace', () => {
  let dir: string;
  const savedEnv = process.env.MLX_AGENT_METRICS;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-runtime-trace-'));
    delete process.env.MLX_AGENT_METRICS;
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    if (savedEnv === undefined) delete process.env.MLX_AGENT_METRICS;
    else process.env.MLX_AGENT_METRICS = savedEnv;
  });

  it('appends one line per record and rotates by UTC day with a proc header', async () => {
    const trace = new RuntimeTrace({ dir: join(dir, 'runtime'), proc: PROC });
    trace.append(sample(DAY1, { dtMs: 0 }));
    trace.append(sample(DAY1 + 1000, { dtMs: 1000 }));
    trace.append(sample(DAY2, { dtMs: 120_000 }));
    trace.append(exit(DAY2 + 1));
    await trace.close();

    const first = readRecords(trace.fileFor(DAY1));
    expect(trace.fileFor(DAY1)).toBe(join(dir, 'runtime', `2026-07-21-${process.pid}.jsonl`));
    expect(first.map((r) => r.k)).toEqual(['proc', 'sample', 'sample']);
    expect(first[0]).toEqual({ ...PROC, ts: DAY1 });
    const second = readRecords(trace.fileFor(DAY2));
    expect(second.map((r) => r.k)).toEqual(['proc', 'sample', 'exit']);
    expect(second[0]!.ts).toBe(DAY2);
    for (const line of readFileSync(trace.fileFor(DAY1), 'utf8').split('\n').filter(Boolean)) {
      expect(Buffer.byteLength(line)).toBeLessThan(4096);
    }
  });

  it('drops the oldest samples when the queue is full and reports the count', async () => {
    const trace = new RuntimeTrace({ dir, proc: PROC });
    trace.append(sample(DAY1, { dtMs: 0 }));
    trace.append(load(DAY1));
    const extra = 40;
    for (let i = 1; i <= SINK_QUEUE_MAX + extra; i++) trace.append(sample(DAY1, { dtMs: i }));
    // The first sample was already taken by the writer; the queue held the load plus the newest samples.
    expect(trace.takeDropped()).toBe(extra + 1);
    expect(trace.takeDropped()).toBe(0);
    await trace.close();

    const records = readRecords(trace.fileFor(DAY1));
    expect(records.filter((r) => r.k === 'load')).toHaveLength(1);
    const samples = records.filter((r): r is RuntimeSampleRecord => r.k === 'sample');
    expect(samples).toHaveLength(SINK_QUEUE_MAX);
    // Oldest dropped first: the survivors are the first one plus the newest run.
    expect(samples[0]!.dtMs).toBe(0);
    expect(samples[1]!.dtMs).toBe(extra + 2);
    expect(samples.at(-1)!.dtMs).toBe(SINK_QUEUE_MAX + extra);
  });

  it('stops sample appends past the size cap but keeps load and exit', async () => {
    const trace = new RuntimeTrace({ dir, proc: PROC });
    const file = trace.fileFor(DAY1);
    // A file already at the cap, ending on a line break.
    writeFileSync(file, '');
    truncateSync(file, RUNTIME_FILE_MAX_BYTES - 1);
    appendFileSync(file, '\n');

    trace.append(sample(DAY1));
    trace.append(load(DAY1));
    trace.append(sample(DAY1 + 1));
    trace.append(exit(DAY1 + 2));
    await trace.close();

    expect(readRecords(file).map((r) => r.k)).toEqual(['proc', 'load', 'exit']);
  });

  it('writes a proc header per process when two processes with one pid share a day file', async () => {
    const first = new RuntimeTrace({ dir, proc: PROC });
    first.append(sample(DAY1 - 5000));
    await first.close();
    const second = new RuntimeTrace({ dir, proc: { ...PROC, startedAt: DAY1 - 100, port: 9090 } });
    second.append(sample(DAY1, { startedAt: DAY1 - 100 }));
    second.append(sample(DAY1 + 1, { startedAt: DAY1 - 100 }));
    await second.close();

    const records = readRecords(first.fileFor(DAY1));
    expect(records.map((r) => [r.k, r.startedAt])).toEqual([
      ['proc', PROC.startedAt],
      ['sample', PROC.startedAt],
      ['proc', DAY1 - 100],
      ['sample', DAY1 - 100],
      ['sample', DAY1 - 100],
    ]);
    expect(records[2]).toMatchObject({ port: 9090, ts: DAY1 });
  });

  it('a torn tail does not swallow the next process proc header', async () => {
    const first = new RuntimeTrace({ dir, proc: PROC });
    first.append(sample(DAY1 - 5000));
    await first.close();
    const file = first.fileFor(DAY1);
    appendFileSync(file, JSON.stringify(sample(DAY1 - 4000)).slice(0, 40));

    const second = new RuntimeTrace({ dir, proc: { ...PROC, startedAt: DAY1 - 100, port: 9090 } });
    second.append(sample(DAY1, { startedAt: DAY1 - 100 }));
    await second.close();

    const parsed: Array<[string, number] | 'invalid'> = [];
    for (const line of readFileSync(file, 'utf8').split('\n')) {
      if (line.trim() === '') continue;
      try {
        const rec = JSON.parse(line) as RuntimeTraceRecord;
        parsed.push([rec.k, rec.startedAt]);
      } catch {
        parsed.push('invalid');
      }
    }
    expect(parsed).toEqual([
      ['proc', PROC.startedAt],
      ['sample', PROC.startedAt],
      'invalid',
      ['proc', DAY1 - 100],
      ['sample', DAY1 - 100],
    ]);
  });

  it('writes nothing when MLX_AGENT_METRICS is off', async () => {
    for (const value of ['0', 'false', 'off']) {
      process.env.MLX_AGENT_METRICS = value;
      const trace = new RuntimeTrace({ dir: join(dir, value), proc: PROC });
      expect(trace.enabled).toBe(false);
      trace.append(sample(DAY1));
      trace.append(exit(DAY1));
      await trace.close();
      expect(existsSync(join(dir, value))).toBe(false);
    }
    expect(readdirSync(dir)).toEqual([]);
  });
});
