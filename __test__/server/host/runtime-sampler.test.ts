import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { ChatResult, SchedulerStats } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  EDGE_COALESCE_MS,
  LIVE_DEAD_AFTER_MS,
  shouldSnapshotScheduler,
  startRuntimeSampler,
  type RuntimeSampler,
  type RuntimeSamplerDeps,
} from '../../../packages/server/src/host/runtime-sampler.js';
import type { ResidentModel } from '../../../packages/server/src/registry.js';
import type {
  RuntimeLiveFile,
  RuntimeSampleRecord,
  RuntimeSchedRecord,
  RuntimeTrace,
  RuntimeTraceRecord,
} from '../../../packages/server/src/runtime-trace.js';
import { TurnTraceModel, type LiveTurn } from '../../../packages/server/src/turn-trace.js';

const T0 = Date.UTC(2026, 6, 21, 12, 0, 0);

function schedStats(overrides: Partial<SchedulerStats> = {}): SchedulerStats {
  return {
    globalSteps: 10,
    maxBatchOccupancy: 2,
    decodeBatchOccupancyHist: [
      { occupancy: 1, steps: 6 },
      { occupancy: 2, steps: 4 },
    ],
    fusedGreedyEpilogueSteps: 0,
    admitted: 3,
    completed: 3,
    admissionDeferredBlocks: 0,
    rowsAllocEvicted: 0,
    blockCapacity: 100,
    freeBlocks: 90,
    reclaimableBlocks: 5,
    allocatedBlocks: 10,
    watermarkBlocks: 1,
    reservedBlocks: 0,
    memoryCapacityBytes: 1000,
    memoryWatermarkBytes: 10,
    reservedBlockBytes: 0,
    reservedStateBytes: 0,
    admissionDeferredState: 0,
    ssdRestoreWaiting: 0,
    ssdRestoreBytes: 0,
    ssdRestoreWaitMs: 0,
    preemptions: 0,
    preemptionsRecompute: 0,
    preemptionsSsd: 0,
    ...overrides,
  };
}

interface FakeState {
  live: number;
  begun: number;
  lastSettleAt: number;
  active: number;
  queued: number;
  turns: LiveTurn[];
  /** The registry refuses the idle probe, as when a dispatch entered the lane after the tick read its counters. */
  probeRefused: boolean;
}

/** One resident binding whose admission state the test drives by hand. */
function fakeBinding(stats: () => Promise<SchedulerStats> = async () => schedStats()) {
  const state: FakeState = {
    live: 0,
    begun: 0,
    lastSettleAt: 0,
    active: 0,
    queued: 0,
    turns: [],
    probeRefused: false,
  };
  const schedulerStats = vi.fn(stats);
  const model = {
    schedulerStats,
    hasBlockPagedCache: () => true,
    contextLimits: () => ({
      trainedWindowTokens: 262144,
      effectiveWindowTokens: 65536,
      pagedBlockCapacity: 0,
      pagedBlockSize: 0,
    }),
  };
  const sessions = {
    liveTurns: () => state.turns,
    turnStats: () => ({ live: state.live, begun: state.begun, lastSettleAt: state.lastSettleAt }),
    get activeDispatches() {
      return state.active;
    },
    get queueDepth() {
      return state.queued;
    },
    queueDepthLimit: 16,
    concurrentAdmissionLimit: 4,
    tryIdleProbe: <T>(call: () => Promise<T>): Promise<T> | null => (state.probeRefused ? null : call()),
    identity: { modelId: 'Qwen3.6-27B', modelType: 'qwen3_5', contextWindow: 262144 },
  };
  const entry = { names: ['Qwen3.6-27B', 'claude-haiku-4-5'], model, sessions } as unknown as ResidentModel;
  return { state, schedulerStats, entry };
}

/** A binding whose turns run through a real `TurnTraceModel`, so turn edges reach the sampler. */
function tracedBinding() {
  const lane = { active: 0 };
  const gates: Array<() => void> = [];
  const result = { text: '', toolCalls: [], numTokens: 3, promptTokens: 5, reasoningTokens: 0, finishReason: 'stop' };
  const inner = {
    chatSessionStart: vi.fn(
      () =>
        new Promise<ChatResult>((resolve) => {
          gates.push(() => resolve(result as unknown as ChatResult));
        }),
    ),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;
  const traced = new TurnTraceModel(inner, {
    modelName: 'm',
    trace: { record: () => {} },
    coldStats: () => undefined,
    sidecarStats: () => undefined,
    memory: () => undefined,
    watermarks: { cold: {}, sidecar: {} },
  });
  const entry = {
    names: ['m'],
    model: {},
    sessions: {
      liveTurns: () => traced.liveTurns(),
      turnStats: () => traced.turnStats(),
      get activeDispatches() {
        return lane.active;
      },
      queueDepth: 0,
      queueDepthLimit: undefined,
      concurrentAdmissionLimit: 1,
      identity: undefined,
    },
  } as unknown as ResidentModel;
  return { traced, gates, entry, lane };
}

function captureTrace(): { records: RuntimeTraceRecord[]; trace: RuntimeTrace } {
  const records: RuntimeTraceRecord[] = [];
  const trace = {
    enabled: true,
    append: (rec: RuntimeTraceRecord) => records.push(rec),
    takeDropped: () => 0,
    close: async () => {},
  } as unknown as RuntimeTrace;
  return { records, trace };
}

async function flushIo(): Promise<void> {
  for (let i = 0; i < 20; i++) await new Promise((resolve) => setImmediate(resolve));
}

describe('runtime sampler', () => {
  let dir: string;
  let sampler: RuntimeSampler | null = null;
  const savedEnv = process.env.MLX_AGENT_METRICS;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-runtime-sampler-'));
    delete process.env.MLX_AGENT_METRICS;
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    vi.setSystemTime(T0);
  });

  afterEach(async () => {
    await sampler?.stop();
    sampler = null;
    vi.useRealTimers();
    rmSync(dir, { recursive: true, force: true });
    if (savedEnv === undefined) delete process.env.MLX_AGENT_METRICS;
    else process.env.MLX_AGENT_METRICS = savedEnv;
  });

  function start(entries: ResidentModel[], overrides: Partial<RuntimeSamplerDeps> = {}) {
    const { records, trace } = captureTrace();
    sampler = startRuntimeSampler({
      registry: { listResidentModels: () => entries },
      health: () => ({ status: 'ok' }),
      models: [{ name: 'Qwen3.6-27B', modelType: 'qwen3_5', contextWindow: 262144 }],
      host: { kind: 'serve', port: 8080 },
      trace,
      dir,
      memory: () => ({ activeBytes: 100, peakBytes: 200, cacheBytes: 50 }),
      wiredLimit: () => 1000,
      isAlive: () => true,
      ...overrides,
    });
    return records;
  }

  const samples = (records: RuntimeTraceRecord[]) => records.filter((r): r is RuntimeSampleRecord => r.k === 'sample');

  it('never requests scheduler stats while a dispatch is active or queued', async () => {
    for (const activeDispatches of [0, 1, 2])
      for (const queueDepth of [0, 1])
        for (const live of [0, 1]) {
          const ok = shouldSnapshotScheduler({
            now: T0 + 10_000,
            activeDispatches,
            queueDepth,
            live,
            begun: 5,
            lastSettleAt: T0,
            pending: false,
            snapshotBegun: undefined,
          });
          expect(ok).toBe(activeDispatches === 0 && queueDepth === 0 && live === 0);
        }

    const binding = fakeBinding();
    binding.state.active = 1;
    start([binding.entry]);
    await vi.advanceTimersByTimeAsync(30_000);
    binding.state.active = 0;
    binding.state.queued = 1;
    await vi.advanceTimersByTimeAsync(30_000);
    binding.state.queued = 0;
    binding.state.live = 1;
    await vi.advanceTimersByTimeAsync(30_000);
    expect(binding.schedulerStats).not.toHaveBeenCalled();

    binding.state.live = 0;
    binding.state.lastSettleAt = Date.now();
    await vi.advanceTimersByTimeAsync(5_000);
    expect(binding.schedulerStats).toHaveBeenCalledTimes(1);
  });

  it('requests one scheduler snapshot after a burst and one baseline after load', async () => {
    const binding = fakeBinding();
    const records = start([binding.entry]);
    await vi.advanceTimersByTimeAsync(0);
    expect(binding.schedulerStats).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(binding.schedulerStats).toHaveBeenCalledTimes(1);

    binding.state.begun = 2;
    binding.state.live = 2;
    await vi.advanceTimersByTimeAsync(10_000);
    binding.state.live = 0;
    binding.state.lastSettleAt = Date.now();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(binding.schedulerStats).toHaveBeenCalledTimes(2);

    const sched = records.filter((r) => r.k === 'sched');
    expect(sched).toHaveLength(2);
    expect(sched[0]).toMatchObject({
      model: 'Qwen3.6-27B',
      blockCapacity: 100,
      occHist: [
        [1, 6],
        [2, 4],
      ],
    });
  });

  it('writes the turns sent to the model when the stats call was sent', async () => {
    const waiting: Array<() => void> = [];
    const release = () => waiting.shift()?.();
    const binding = fakeBinding(
      () =>
        new Promise<SchedulerStats>((resolve) => {
          waiting.push(() => resolve(schedStats()));
        }),
    );
    const records = start([binding.entry]);
    const sched = () => records.filter((r): r is RuntimeSchedRecord => r.k === 'sched');
    await vi.advanceTimersByTimeAsync(0);
    expect(binding.schedulerStats).toHaveBeenCalledTimes(1);
    release();
    await vi.advanceTimersByTimeAsync(0);
    expect(sched().map((r) => r.turns)).toEqual([0]);

    binding.state.begun = 3;
    binding.state.lastSettleAt = Date.now();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(binding.schedulerStats).toHaveBeenCalledTimes(2);
    binding.state.begun = 4;
    release();
    await vi.advanceTimersByTimeAsync(0);
    expect(sched().map((r) => r.turns)).toEqual([0, 3]);
  });

  it('scheduler stats go through the registry idle probe and retry after a refusal', async () => {
    const binding = fakeBinding();
    binding.state.probeRefused = true;
    const records = start([binding.entry]);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(binding.schedulerStats).not.toHaveBeenCalled();

    binding.state.probeRefused = false;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(binding.schedulerStats).toHaveBeenCalledTimes(1);
    expect(records.filter((r) => r.k === 'sched')).toHaveLength(1);
  });

  it('skips default-zero scheduler stats', async () => {
    const binding = fakeBinding(async () => schedStats({ blockCapacity: 0, globalSteps: 0 }));
    const records = start([binding.entry]);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(binding.schedulerStats).toHaveBeenCalledTimes(1);
    expect(records.filter((r) => r.k === 'sched')).toEqual([]);
  });

  it('samples every 5 s busy and 60 s idle, plus one on each busy edge', async () => {
    const binding = fakeBinding();
    const records = start([binding.entry]);
    expect(samples(records)).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(59_000);
    expect(samples(records)).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(samples(records)).toHaveLength(2);

    binding.state.live = 1;
    await vi.advanceTimersByTimeAsync(5_000);
    // The busy edge.
    expect(samples(records)).toHaveLength(3);
    expect(samples(records).at(-1)!.busy).toBe(true);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(samples(records)).toHaveLength(5);

    binding.state.live = 0;
    await vi.advanceTimersByTimeAsync(1_000);
    expect(samples(records)).toHaveLength(6);
    expect(samples(records).at(-1)!.busy).toBe(false);
    await vi.advanceTimersByTimeAsync(55_000);
    expect(samples(records)).toHaveLength(6);
  });

  it('pulls the next tick in on a turn edge', async () => {
    const { traced, gates, entry } = tracedBinding();
    const records = start([entry]);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(samples(records)).toHaveLength(1);

    const turn = traced.chatSessionStart([]);
    await vi.advanceTimersByTimeAsync(EDGE_COALESCE_MS);
    const busyEdge = samples(records).at(-1)!;
    expect(busyEdge.busy).toBe(true);
    expect(busyEdge.ts - T0).toBe(1_000 + EDGE_COALESCE_MS);
    gates[0]!();
    await turn;
  });

  it('writes the live file atomically and removes it on stop', async () => {
    const binding = fakeBinding();
    binding.state.live = 1;
    binding.state.turns = [
      {
        traceId: 't1',
        model: 'claude-haiku-4-5',
        modelId: 'Qwen3.6-27B',
        ctxWindow: 65536,
        startedAt: T0,
        stream: true,
        dispatch: 'start',
        progress: { phase: 'decode', prefillTotal: 40, prefillDone: 40, cachedTokens: 60, generatedTokens: 5 },
      },
    ];
    const records = start([binding.entry]);
    await flushIo();
    const liveFile = join(dir, `live-${process.pid}.json`);
    const live = JSON.parse(readFileSync(liveFile, 'utf8')) as RuntimeLiveFile;
    expect(live).toMatchObject({
      v: 1,
      pid: process.pid,
      host: 'serve',
      port: 8080,
      intervalMs: 1000,
      health: 'ok',
      mem: { active: 100, peak: 200, cache: 50, wiredLimit: 1000 },
      decodeTps: null,
      loading: null,
    });
    expect(live.models).toEqual([
      {
        name: 'Qwen3.6-27B',
        modelType: 'qwen3_5',
        ctxWindow: 65536,
        paged: true,
        maxConcurrent: 4,
        loadedAt: null,
        loadMs: null,
        inFlight: 1,
        queued: 0,
        queueLimit: 16,
        pool: null,
      },
    ]);
    expect(live.turns).toEqual([
      {
        traceId: 't1',
        model: 'claude-haiku-4-5',
        modelId: 'Qwen3.6-27B',
        ctxWindow: 65536,
        startedAt: T0,
        stream: true,
        dispatch: 'start',
        phase: 'decode',
        promptTokens: 100,
        cachedTokens: 60,
        prefillTotal: 40,
        prefillDone: 40,
        generatedTokens: 5,
      },
    ]);
    expect(readdirSync(dir).filter((name) => name.endsWith('.tmp'))).toEqual([]);

    await sampler!.stop();
    sampler = null;
    expect(existsSync(liveFile)).toBe(false);
    expect(records.at(-1)).toMatchObject({ k: 'exit' });
    expect(records.at(-2)).toMatchObject({ k: 'sample' });
  });

  it('live decodeTps counts streamed tokens only, not a non-stream turn that settles', async () => {
    let generated = 0;
    let finishStream!: () => void;
    const streamGate = new Promise<void>((resolve) => {
      finishStream = resolve;
    });
    let finishPlain!: () => void;
    const inner = {
      chatSessionStart: vi.fn(
        () =>
          new Promise<ChatResult>((resolve) => {
            finishPlain = () =>
              resolve({
                text: '',
                toolCalls: [],
                numTokens: 2000,
                promptTokens: 5,
                reasoningTokens: 0,
                finishReason: 'stop',
              } as unknown as ChatResult);
          }),
      ),
      chatStreamSessionStart: vi.fn(async function* (
        _messages: unknown,
        _config: unknown,
        _signal: unknown,
        ref?: { read: (() => unknown) | null },
      ) {
        if (ref)
          ref.read = () => ({
            phase: 'decode',
            prefillTotal: 1,
            prefillDone: 1,
            cachedTokens: 0,
            generatedTokens: generated,
          });
        try {
          yield { text: 'a', done: false as const };
          await streamGate;
          yield {
            text: '',
            done: true as const,
            finishReason: 'stop',
            toolCalls: [],
            numTokens: generated,
            promptTokens: 5,
          };
        } finally {
          if (ref) ref.read = null;
        }
      }),
      resetCaches: vi.fn(),
    } as unknown as SessionCapableModel;
    const traced = new TurnTraceModel(inner, {
      modelName: 'm',
      trace: { record: () => {} },
      coldStats: () => undefined,
      sidecarStats: () => undefined,
      memory: () => undefined,
      watermarks: { cold: {}, sidecar: {} },
    });
    const entry = {
      names: ['m'],
      model: {},
      sessions: {
        liveTurns: () => traced.liveTurns(),
        turnStats: () => traced.turnStats(),
        activeDispatches: 0,
        queueDepth: 0,
        queueDepthLimit: undefined,
        concurrentAdmissionLimit: 2,
        identity: undefined,
      },
    } as unknown as ResidentModel;
    start([entry]);
    const liveFile = join(dir, `live-${process.pid}.json`);
    const readLive = () => JSON.parse(readFileSync(liveFile, 'utf8')) as RuntimeLiveFile;

    const stream = traced.chatStreamSessionStart([])[Symbol.asyncIterator]();
    await stream.next();
    const plain = traced.chatSessionStart([]);
    await vi.advanceTimersByTimeAsync(1_000);
    await flushIo();
    for (let i = 0; i < 3; i++) {
      generated += 20;
      await vi.advanceTimersByTimeAsync(1_000);
      await flushIo();
    }
    expect(readLive().decodeTps).toBe(20);

    generated += 20;
    finishPlain();
    await plain;
    await vi.advanceTimersByTimeAsync(1_000);
    await flushIo();
    const live = readLive();
    expect(live.turns).toHaveLength(1);
    expect(live.decodeTps).not.toBeNull();
    expect(live.decodeTps!).toBeLessThanOrEqual(80);

    finishStream();
    while (!(await stream.next()).done) {
      // drains
    }
  });

  it('timedLoad records duration and ok for success and failure', async () => {
    const records = start([]);
    const loaded = await sampler!.timedLoad({ model: 'a', modelType: 'qwen3_5', capability: 'chat' }, async () => {
      vi.setSystemTime(Date.now() + 1234);
      return 'model';
    });
    expect(loaded).toBe('model');
    await expect(
      sampler!.timedLoad({ model: 'tts', modelType: 'qwen3_tts', capability: 'audio' }, async () => {
        vi.setSystemTime(Date.now() + 10);
        throw new Error('load failed: /secret/path');
      }),
    ).rejects.toThrow('load failed');

    const loads = records.filter((r) => r.k === 'load');
    expect(loads).toEqual([
      expect.objectContaining({ model: 'a', modelType: 'qwen3_5', capability: 'chat', ms: 1234, ok: true }),
      expect.objectContaining({ model: 'tts', modelType: 'qwen3_tts', capability: 'audio', ms: 10, ok: false }),
    ]);
    expect(JSON.stringify(records)).not.toContain('secret');
  });

  it('sweeps live files of dead pids, and files older than the hard limit, at start', async () => {
    const write = (name: string, ts: number) => writeFileSync(join(dir, name), JSON.stringify({ v: 1, ts }));
    write('live-111.json', T0);
    write('live-222.json', T0 - LIVE_DEAD_AFTER_MS - 1);
    write('live-333.json', T0 - 1000);
    writeFileSync(join(dir, 'live-111.json.tmp'), '{');
    writeFileSync(join(dir, 'live-333.json.tmp'), '{');
    writeFileSync(join(dir, '2026-07-01-444.jsonl'), '');
    writeFileSync(join(dir, '2026-07-10-444.jsonl'), '');
    writeFileSync(join(dir, 'notes.txt'), '');
    start([], { isAlive: (pid) => pid !== 111 });
    await flushIo();

    const left = readdirSync(dir).sort();
    expect(left).toEqual(
      ['2026-07-10-444.jsonl', 'live-333.json', 'live-333.json.tmp', `live-${process.pid}.json`, 'notes.txt'].sort(),
    );
  });

  it('59 s idle then a 2 s turn gives busyMs of about 2000', async () => {
    const { traced, gates, entry } = tracedBinding();
    const records = start([entry]);
    await vi.advanceTimersByTimeAsync(59_000);
    const turn = traced.chatSessionStart([]);
    await vi.advanceTimersByTimeAsync(2_000);
    gates[0]!();
    await turn;
    await vi.advanceTimersByTimeAsync(1_000);

    const all = samples(records);
    const total = all.reduce((sum, s) => sum + s.busyMs, 0);
    expect(total).toBe(2_000);
    expect(all.at(-1)!.busy).toBe(false);
  });

  it('busyMs ends at the settle edge while the admission lane is still held', async () => {
    const { traced, gates, entry, lane } = tracedBinding();
    const records = start([entry]);
    await vi.advanceTimersByTimeAsync(59_000);
    lane.active = 1;
    const turn = traced.chatSessionStart([]);
    await vi.advanceTimersByTimeAsync(100);
    gates[0]!();
    await turn;
    await vi.advanceTimersByTimeAsync(1);
    lane.active = 0;
    await vi.advanceTimersByTimeAsync(60_000);

    const total = samples(records).reduce((sum, s) => sum + s.busyMs, 0);
    expect(total).toBe(100);
  });

  it('is inert when MLX_AGENT_METRICS is off', async () => {
    process.env.MLX_AGENT_METRICS = '0';
    const records = start([fakeBinding().entry]);
    await vi.advanceTimersByTimeAsync(10_000);
    await flushIo();
    expect(records).toEqual([]);
    expect(readdirSync(dir)).toEqual([]);
  });
});
