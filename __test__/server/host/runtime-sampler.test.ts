import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import type { ChatResult, ChatStreamProgress, SchedulerStats } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  EDGE_COALESCE_MS,
  LIVE_DEAD_AFTER_MS,
  SAMPLE_IDLE_MS,
  shouldSnapshotScheduler,
  startRuntimeSampler,
  stateRootLimit,
  type RuntimeSampler,
  type RuntimeSamplerDeps,
} from '../../../packages/server/src/host/runtime-sampler.js';
import { RESIDENCY_DELTA_FIELDS } from '../../../packages/server/src/metrics-trace.js';
import {
  LOAD_RESIDENCY_FIELDS,
  readResidency,
  type LoadProgress,
  type PipelineStats,
  type WarmupReport,
} from '../../../packages/server/src/native-progress.js';
import type { ResidentModel } from '../../../packages/server/src/registry.js';
import type {
  RuntimeLiveFile,
  RuntimeLoadRecord,
  RuntimeSampleRecord,
  RuntimeSchedRecord,
  RuntimeTrace,
  RuntimeTraceRecord,
} from '../../../packages/server/src/runtime-trace.js';
import { TurnTraceModel, type LiveTurn } from '../../../packages/server/src/turn-trace.js';
import { WorkLedger, type WorkTotals } from '../../../packages/server/src/work-ledger.js';

const T0 = Date.UTC(2026, 6, 21, 12, 0, 0);

/** The ledger the sampler and the traced turns of one test share. */
let ledger = new WorkLedger();

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

/** A `residencyInfo()` report of a partial store, every counter 0. */
function residencyInfo(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  const info: Record<string, unknown> = {
    residency: 'store-plan',
    partialExpertSlots: 412,
    expertCount: 512,
    fallbackReason: 'store plan: experts stream through a 59.2 GiB store, all banks need 73.1 GiB',
  };
  for (const key of LOAD_RESIDENCY_FIELDS) info[key] = 0;
  for (const key of RESIDENCY_DELTA_FIELDS) info[key] = 0;
  return { ...info, ...overrides };
}

const LOAD_RECORD_RESIDENCY_KEYS = ['residency', 'expertSlots', 'expertCount', ...LOAD_RESIDENCY_FIELDS];

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
    servedWindowTokens: 65536,
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
    work: ledger,
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

/** One streamed turn the test drives: its native progress and its end. */
interface DrivenTurn {
  read: () => ChatStreamProgress;
  finish: (numTokens: number) => void;
}

/**
 * A binding whose streamed turns read the progress the test sets, and whose non-streamed turns end when the test
 * says, all through a real `TurnTraceModel` on the test's ledger.
 */
function drivenBinding() {
  const turns: DrivenTurn[] = [];
  const plain: Array<(numTokens: number) => void> = [];
  const inner = {
    chatSessionStart: vi.fn(
      () =>
        new Promise<ChatResult>((resolve) => {
          plain.push((numTokens) =>
            resolve({
              text: '',
              toolCalls: [],
              numTokens,
              promptTokens: 5,
              reasoningTokens: 0,
              finishReason: 'stop',
            } as unknown as ChatResult),
          );
        }),
    ),
    chatStreamSessionStart: vi.fn(async function* (
      _messages: unknown,
      _config: unknown,
      _signal: unknown,
      ref?: { read: (() => ChatStreamProgress | null) | null },
    ) {
      let finish!: (numTokens: number) => void;
      const gate = new Promise<number>((resolve) => {
        finish = resolve;
      });
      const turn: DrivenTurn = { read: () => ({ phase: 'queued', waitMs: 0 }), finish };
      turns.push(turn);
      if (ref) ref.read = () => turn.read();
      try {
        yield { text: 'a', done: false as const };
        const numTokens = await gate;
        yield { text: '', done: true as const, finishReason: 'stop', toolCalls: [], numTokens, promptTokens: 5 };
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
    work: ledger,
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
  /** Start one streamed turn; resolves once it is in flight. */
  const stream = async () => {
    const iterator = traced.chatStreamSessionStart([])[Symbol.asyncIterator]();
    await iterator.next();
    const turn = turns.at(-1)!;
    return {
      turn,
      end: async (numTokens: number) => {
        turn.finish(numTokens);
        while (!(await iterator.next()).done) {
          // drains
        }
      },
    };
  };
  return { traced, entry, stream, plain };
}

/** A one-shot prefill of `total` tokens from `prefillAt`, then decode at `tps` from `decodeAt` (null: still prefilling). */
function phases(total: number, prefillAt: number, decodeAt: number | null, tps = 40): () => ChatStreamProgress {
  return () => {
    const now = Date.now();
    const base = { waitMs: 0, slotMs: 0, prepareMs: 0, cachedTokens: 0, prefillTotal: total };
    if (decodeAt === null || now < decodeAt) {
      return { ...base, phase: 'prefill', prefillDone: 0, prefillElapsedMs: now - prefillAt };
    }
    const decodeElapsedMs = now - decodeAt;
    return {
      ...base,
      phase: 'decode',
      prefillDone: total,
      prefillElapsedMs: decodeAt - prefillAt,
      decodeElapsedMs,
      generatedTokens: 1 + Math.floor((tps * decodeElapsedMs) / 1000),
    };
  };
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

/**
 * Wait until the live body built at time `builtAt` is on disk and the sampler's write has settled. The file shows on
 * disk before the write's promise settles, and a tick that finds a write still in flight writes nothing, so a clock
 * advance in that window leaves a stale file.
 */
async function liveLanded(sampler: RuntimeSampler | null, file: string, builtAt: number | null): Promise<void> {
  const inFlight = () => (sampler as unknown as { liveWrite: Promise<void> | null } | null)?.liveWrite ?? null;
  const deadline = performance.now() + 10_000;
  for (;;) {
    const write = inFlight();
    if (write !== null) {
      await write;
      continue;
    }
    let ts: number | null = null;
    try {
      ts = (JSON.parse(readFileSync(file, 'utf8')) as RuntimeLiveFile).ts;
    } catch {
      // not written yet
    }
    if (ts !== null && ts === builtAt) return;
    if (performance.now() > deadline)
      throw new Error(`live file never reached ts ${String(builtAt)}; last ${String(ts)}`);
    await new Promise((resolve) => setImmediate(resolve));
  }
}

describe('runtime sampler', () => {
  let dir: string;
  let sampler: RuntimeSampler | null = null;
  const savedEnv = process.env.MLX_AGENT_METRICS;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-runtime-sampler-'));
    ledger = new WorkLedger();
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

  /** When the sampler last built a live body: only that build reads `host.persistPagedCache`. */
  let liveBuiltAt: number | null = null;
  const landed = () => liveLanded(sampler, join(dir, `live-${process.pid}.json`), liveBuiltAt);
  const readLive = () => JSON.parse(readFileSync(join(dir, `live-${process.pid}.json`), 'utf8')) as RuntimeLiveFile;

  function start(entries: ResidentModel[], overrides: Partial<RuntimeSamplerDeps> = {}) {
    const { records, trace } = captureTrace();
    liveBuiltAt = null;
    sampler = startRuntimeSampler({
      registry: { listResidentModels: () => entries },
      health: () => ({ status: 'ok' }),
      models: [{ name: 'Qwen3.6-27B', modelType: 'qwen3_5', contextWindow: 262144 }],
      host: {
        kind: 'serve',
        port: 8080,
        get persistPagedCache() {
          liveBuiltAt = Date.now();
          return false;
        },
        coldCapture: { blocksPerTurn: 63_125, budgetMs: 10_000 },
        cacheDir: null,
      },
      trace,
      dir,
      memory: () => ({ activeBytes: 100, peakBytes: 200, cacheBytes: 50 }),
      wiredLimit: () => 1000,
      isAlive: () => true,
      loadProgress: () => null,
      pipelineStats: () => null,
      work: ledger,
      ...overrides,
    });
    return records;
  }

  const samples = (records: RuntimeTraceRecord[]) => records.filter((r): r is RuntimeSampleRecord => r.k === 'sample');

  /** Advance `ms` in steps shorter than any tick gap, letting each live write land before the next tick. */
  async function run(ms: number): Promise<void> {
    for (let left = ms; left > 0; left -= 100) {
      await vi.advanceTimersByTimeAsync(Math.min(100, left));
      await landed();
    }
  }

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

  it('wakes up at SCHED_IDLE_SETTLE_MS so brief idle gaps (< 5 s) still capture scheduler stats', async () => {
    const binding = fakeBinding();
    binding.state.begun = 1;
    binding.state.live = 1;
    const records = start([binding.entry]);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(binding.schedulerStats).not.toHaveBeenCalled();

    // Burst ends right before the next busy tick (750 ms after settle would still beat HEARTBEAT_IDLE_MS = 5 s).
    binding.state.live = 0;
    binding.state.lastSettleAt = Date.now() + 750;
    await vi.advanceTimersByTimeAsync(1_000);
    // At tick time (lastSettleAt + 250 ms < 500 ms), snapshot must wait for settle.
    expect(binding.schedulerStats).not.toHaveBeenCalled();
    const samplesBefore = samples(records).length;

    // 250 ms later (at lastSettleAt + 500 ms, well under the 5 s idle heartbeat), the settle wakeup fires.
    await vi.advanceTimersByTimeAsync(250);
    expect(binding.schedulerStats).toHaveBeenCalledTimes(1);
    expect(records.filter((r) => r.k === 'sched')).toHaveLength(1);
    // The settle tick must not emit an extra sample record inside SAMPLE_IDLE_MS.
    expect(samples(records)).toHaveLength(samplesBefore);
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
        clientSession: 'acp-session-1',
      },
    ];
    const records = start([binding.entry]);
    await landed();
    const liveFile = join(dir, `live-${process.pid}.json`);
    const live = JSON.parse(readFileSync(liveFile, 'utf8')) as RuntimeLiveFile;
    expect(live).toMatchObject({
      v: 3,
      pid: process.pid,
      host: 'serve',
      port: 8080,
      persistPagedCache: false,
      coldCapture: { blocks: 63_125, ms: 10_000 },
      intervalMs: 1000,
      health: 'ok',
      mem: { active: 100, peak: 200, cache: 50, wiredLimit: 1000 },
      work: { ms: { loading: 0, waiting: 0, prepare: 0, prefill: 0, decode: 0, running: 0 } },
      rates: { windowMs: 10_000, outputTps: null, prefillTps: null },
      loading: [],
    });
    expect('decodeTps' in live).toBe(false);
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
        stateNamespace: null,
        experts: null,
        images: false,
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
        waitMs: null,
        slotMs: null,
        prepareMs: null,
        prefillElapsedMs: null,
        decodeElapsedMs: null,
        promptTokens: 100,
        cachedTokens: 60,
        prefillTotal: 40,
        prefillDone: 40,
        generatedTokens: 5,
        clientSession: 'acp-session-1',
      },
    ]);
    expect(readdirSync(dir).filter((name) => name.endsWith('.tmp'))).toEqual([]);

    await sampler!.stop();
    sampler = null;
    expect(existsSync(liveFile)).toBe(false);
    expect(records.at(-1)).toMatchObject({ k: 'exit' });
    expect(records.at(-2)).toMatchObject({ k: 'sample' });
  });

  it('writes a null capture limit when the host has the disk cache off', async () => {
    start([], {
      host: {
        kind: 'serve',
        port: 8080,
        get persistPagedCache() {
          liveBuiltAt = Date.now();
          return false;
        },
        coldCapture: null,
        cacheDir: null,
      },
    });
    await landed();
    const liveFile = join(dir, `live-${process.pid}.json`);
    const live = JSON.parse(readFileSync(liveFile, 'utf8')) as RuntimeLiveFile;
    expect(live.persistPagedCache).toBe(false);
    expect(live.coldCapture).toBeNull();
  });

  it('the heartbeat carries the state cache dir and its limit', async () => {
    const { entry } = fakeBinding();
    start([entry], {
      health: () => ({
        status: 'ok',
        memory_governor: { dflash2_states: { disk: { mode: 'persistent', limit_bytes: 123 } } },
      }),
      host: {
        kind: 'serve',
        port: 8080,
        get persistPagedCache() {
          liveBuiltAt = Date.now();
          return true;
        },
        coldCapture: null,
        cacheDir: '/x',
      },
    });
    await landed();
    expect(readLive().stateCache).toEqual({ dir: '/x', limitBytes: 123 });
  });

  it('the heartbeat carries a relative cache dir as an absolute one', async () => {
    start([], {
      host: {
        kind: 'embedded',
        port: 0,
        get persistPagedCache() {
          liveBuiltAt = Date.now();
          return true;
        },
        coldCapture: null,
        cacheDir: 'rel/states',
      },
    });
    await landed();
    expect(readLive().stateCache.dir).toBe(resolve('rel/states'));
  });

  it('each model row carries the namespace its stateTierNamespace() reports', async () => {
    const named = fakeBinding();
    (named.entry.model as { stateTierNamespace?: () => string | null }).stateTierNamespace = () =>
      '0123456789abcdef0123456789abcdef';
    const closed = fakeBinding();
    (closed.entry.model as { stateTierNamespace?: () => string | null }).stateTierNamespace = () => null;
    const broken = fakeBinding();
    (broken.entry.model as { stateTierNamespace?: () => string | null }).stateTierNamespace = () => {
      throw new Error('gone');
    };
    const plain = fakeBinding();
    start([named.entry, closed.entry, broken.entry, plain.entry]);
    await landed();
    expect(readLive().models.map((model) => model.stateNamespace)).toEqual([
      '0123456789abcdef0123456789abcdef',
      null,
      null,
      null,
    ]);
  });

  // MUTATION: any answer of the method counts as yes -> the 'no' row goes red.
  it("each model row carries the loaded model's own supportsImages() answer", async () => {
    const sees = fakeBinding();
    (sees.entry.model as { supportsImages?: () => boolean }).supportsImages = () => true;
    const blind = fakeBinding();
    (blind.entry.model as { supportsImages?: () => boolean }).supportsImages = () => false;
    const broken = fakeBinding();
    (broken.entry.model as { supportsImages?: () => boolean }).supportsImages = () => {
      throw new Error('gone');
    };
    const plain = fakeBinding();
    start([sees.entry, blind.entry, broken.entry, plain.entry]);
    await landed();
    expect(readLive().models.map((model) => model.images)).toEqual([true, false, null, false]);
  });

  // MUTATION: `streaming: snapshot.residency === 'direct'` -> the partial row says false -> red.
  it('each model row carries its experts: in memory, of the checkpoint, streaming', async () => {
    const partial = fakeBinding();
    (partial.entry.model as { residencyInfo?: () => unknown }).residencyInfo = () => residencyInfo();
    const direct = fakeBinding();
    (direct.entry.model as { residencyInfo?: () => unknown }).residencyInfo = () =>
      residencyInfo({ residency: 'direct', partialExpertSlots: 512 });
    const rows = fakeBinding();
    (rows.entry.model as { residencyInfo?: () => unknown }).residencyInfo = () =>
      residencyInfo({ residency: 'rows', partialExpertSlots: 0 });
    const other = fakeBinding();
    (other.entry.model as { residencyInfo?: () => unknown }).residencyInfo = () => ({ denseBytes: 1, inTurn: false });
    const broken = fakeBinding();
    (broken.entry.model as { residencyInfo?: () => unknown }).residencyInfo = () => {
      throw new Error('gone');
    };
    const plain = fakeBinding();
    start([partial.entry, direct.entry, rows.entry, other.entry, broken.entry, plain.entry]);
    await landed();
    expect(readLive().models.map((model) => model.experts)).toEqual([
      { inMemory: 412, total: 512, streaming: true },
      { inMemory: 512, total: 512, streaming: false },
      { inMemory: 0, total: 512, streaming: true },
      null,
      null,
      null,
    ]);
  });

  it('no limit when two state-tier models are resident', () => {
    const disk = { mode: 'persistent', limit_bytes: 123 };
    expect(stateRootLimit(disk, ['qwen3_5', 'qwen3_5_moe'])).toBeNull();
    expect(stateRootLimit(disk, ['qwen3_5', ''])).toBeNull();
    expect(stateRootLimit(disk, ['qwen3_5', 'qwen3'])).toBe(123);
    expect(stateRootLimit(disk, ['qwen4_exp'])).toBeNull();
    expect(stateRootLimit(undefined, ['qwen3_5'])).toBeNull();
  });

  it('no limit from a temporary tier', () => {
    expect(stateRootLimit({ mode: 'temp', limit_bytes: 123 }, ['qwen3_5'])).toBeNull();
    expect(stateRootLimit({ mode: 'off', limit_bytes: 123 }, ['qwen3_5'])).toBeNull();
    expect(stateRootLimit({ mode: 'persistent', limit_bytes: 0 }, ['qwen3_5'])).toBeNull();
  });

  // MUTATION: credit the final tokens of a non-streamed turn -> 2,000 tokens with no decode time -> the rate jumps -> red.
  it("writes v3: the work counters, the window rates, and each turn's two elapsed clocks, with no decodeTps", async () => {
    const { entry, stream, plain, traced } = drivenBinding();
    start([entry]);
    await landed();

    const { turn, end } = await stream();
    turn.read = phases(1000, T0, T0 + 1500);
    await run(500);
    const other = traced.chatSessionStart([]);
    await run(2000);
    plain[0]!(2000);
    await other;
    await run(500);

    const live = readLive();
    const now = live.ts;
    expect(now).toBeGreaterThan(T0 + 1500);
    expect(live.v).toBe(3);
    expect('decodeTps' in live).toBe(false);
    expect(live.turns).toHaveLength(1);
    expect(live.turns[0]).toMatchObject({ phase: 'decode', prefillElapsedMs: 1500, decodeElapsedMs: now - T0 - 1500 });
    expect(live.work.ms).toEqual({
      loading: 0,
      waiting: 0,
      prepare: 0,
      prefill: 1500,
      decode: now - T0 - 1500,
      running: 2000,
    });
    // 40 tok/s while decoding; the one-shot prefill of 1,000 tokens took 1.5 s; the plain turn's tokens are in neither.
    expect(live.rates).toEqual({ windowMs: 10_000, outputTps: 40, prefillTps: 666.67 });
    await end(1 + 40 * 3);
  });

  // MUTATION: divide by wall time since the anchor -> the rate sags under 40 during the prefill -> red.
  it('holds the Output rate through the next prefill, and writes null once a whole window has no decode', async () => {
    const { entry, stream } = drivenBinding();
    start([entry]);
    await landed();

    const first = await stream();
    first.turn.read = phases(300, T0, T0 + 200);
    await run(3000);
    await first.end(1 + Math.floor((40 * (Date.now() - T0 - 200)) / 1000));
    const second = await stream();
    const prefillAt = Date.now();
    second.turn.read = phases(20_000, prefillAt, null);
    const seen: Array<{ at: number; outputTps: number | null; prefillTps: number | null }> = [];
    for (let i = 0; i < 14; i++) {
      await run(1000);
      const live = readLive();
      seen.push({ at: live.ts - prefillAt, ...live.rates });
    }
    const inWindow = seen.filter((s) => s.at < 10_000);
    const past = seen.filter((s) => s.at > 10_000 + 1000);
    expect(inWindow.length).toBeGreaterThan(5);
    expect(inWindow.every((s) => s.outputTps === 40)).toBe(true);
    expect(past.length).toBeGreaterThan(0);
    expect(past.every((s) => s.outputTps === null)).toBe(true);
    // A one-shot prefill credits nothing until its first token; nothing ever reads 0.
    expect(seen.every((s) => s.prefillTps === null)).toBe(true);
    expect(seen.some((s) => s.outputTps === 0)).toBe(false);

    second.turn.read = phases(20_000, prefillAt, prefillAt + 14_500);
    await run(1500);
    expect(readLive().rates.prefillTps).toBe(Math.round((20_000 * 1000 * 100) / 14_500) / 100);
    await second.end(10);
  });

  // MUTATION: keep `prevWork` at the sampler start, so each sample writes totals instead of deltas -> sums overshoot -> red.
  it('a sample carries the ten work deltas, and they sum to the ledger totals', async () => {
    const { entry, stream, plain, traced } = drivenBinding();
    const records = start([entry]);
    let loaded!: () => void;
    const load = sampler!.timedLoad(
      { model: 'm', modelType: 'qwen3_5', capability: 'chat', path: '/m/m' },
      () => new Promise<string>((resolve) => (loaded = () => resolve('m'))),
    );
    await run(1500);
    loaded();
    await load;

    const { turn, end } = await stream();
    const begin = Date.now();
    turn.read = () => {
      const now = Date.now();
      if (now < begin + 300) return { phase: 'queued', waitMs: now - begin };
      if (now < begin + 500) return { phase: 'slot', waitMs: 300, slotMs: now - begin - 300, prepareMs: 0 };
      if (now < begin + 600) return { phase: 'prepare', waitMs: 300, slotMs: 200, prepareMs: now - begin - 500 };
      return { ...phases(400, begin + 600, begin + 2600)(), waitMs: 300, slotMs: 200, prepareMs: 100 };
    };
    const other = traced.chatSessionStart([]);
    await run(6000);
    plain[0]!(50);
    await other;
    await run(1000);
    await end(1 + Math.floor((40 * (Date.now() - begin - 2600)) / 1000));
    await run(12_000);
    await sampler!.stop();
    sampler = null;

    const totals: WorkTotals = ledger.totals();
    const sum = (key: keyof RuntimeSampleRecord) =>
      samples(records).reduce((acc, rec) => acc + ((rec[key] as number | undefined) ?? 0), 0);
    const expected = {
      loadingMs: totals.ms.loading,
      waitingMs: totals.ms.waiting,
      prepareMs: totals.ms.prepare,
      prefillMs: totals.ms.prefill,
      decodeMs: totals.ms.decode,
      runningMs: totals.ms.running,
      rateOutTokens: totals.outTokens,
      rateOutMs: totals.outMs,
      ratePrefillTokens: totals.prefillTokens,
      ratePrefillMs: totals.prefillMs,
    };
    for (const rec of samples(records)) {
      for (const key of Object.keys(expected)) expect(rec).toHaveProperty(key);
    }
    for (const [key, value] of Object.entries(expected)) {
      expect(value, key).toBeGreaterThan(0);
      expect(sum(key as keyof RuntimeSampleRecord), key).toBe(Math.round(value));
    }
    expect(samples(records).length).toBeGreaterThan(3);
    expect(totals.ms).toMatchObject({ loading: 1500, waiting: 500, prepare: 100, prefill: 2000 });
  });

  it('timedLoad records duration and ok for success and failure', async () => {
    const records = start([]);
    const info = { model: 'a', modelType: 'qwen3_5', capability: 'chat', path: '/m/a' } as const;
    const loaded = await sampler!.timedLoad(info, async () => {
      vi.setSystemTime(Date.now() + 1234);
      return 'model';
    });
    expect(loaded).toBe('model');
    await expect(
      sampler!.timedLoad({ model: 'tts', modelType: 'qwen3_tts', capability: 'audio', path: '/m/tts' }, async () => {
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
    expect(loads[0]).not.toHaveProperty('warmupMs');
  });

  it('two overlapping loads keep their own model and counts in the live file', async () => {
    const progress = new Map<string, LoadProgress | null>([
      ['/m/big', { phase: 'kernels', steps: 50, stepsDone: 3 }],
      ['/m/small', { phase: 'weights', steps: 0, stepsDone: 0 }],
    ]);
    start([], { loadProgress: (path) => progress.get(path) ?? null });
    await landed();
    const gates: Array<() => void> = [];
    const loadOf = (model: string, path: string) =>
      sampler!.timedLoad(
        { model, modelType: 'qwen3_5_moe', capability: 'chat', path },
        () => new Promise<string>((resolve) => gates.push(() => resolve(model))),
      );
    const big = loadOf('Ornith-35B', '/m/big');
    await vi.advanceTimersByTimeAsync(10);
    const small = loadOf('Qwen3.5-4B', '/m/small');
    const other = loadOf('Gemma-4', '/m/other');
    await vi.advanceTimersByTimeAsync(1_000);
    await landed();
    expect(readLive().loading).toEqual([
      { model: 'Ornith-35B', startedAt: T0, phase: 'kernels', stepsDone: 3, steps: 50 },
      { model: 'Qwen3.5-4B', startedAt: T0 + 10, phase: 'weights', stepsDone: 0, steps: 0 },
      { model: 'Gemma-4', startedAt: T0 + 10, phase: 'weights', stepsDone: null, steps: null },
    ]);

    progress.set('/m/big', { phase: 'kernels', steps: 50, stepsDone: 4 });
    gates[1]!();
    await small;
    await vi.advanceTimersByTimeAsync(1_000);
    await landed();
    expect(readLive().loading).toEqual([
      { model: 'Ornith-35B', startedAt: T0, phase: 'kernels', stepsDone: 4, steps: 50 },
      { model: 'Gemma-4', startedAt: T0 + 10, phase: 'weights', stepsDone: null, steps: null },
    ]);
    gates[0]!();
    gates[2]!();
    await Promise.all([big, other]);
    await vi.advanceTimersByTimeAsync(1_000);
    await landed();
    expect(readLive().loading).toEqual([]);
  });

  it('a load record carries the warm-up report, and none without one', async () => {
    const records = start([]);
    const report: WarmupReport = {
      ms: 3_400,
      steps: 50,
      skipped: 1,
      reset: 0,
      pipelines: 212,
      compileMs: 2_900.5,
      peakBytes: 4_096,
    };
    await sampler!.timedLoad(
      { model: 'warm', modelType: 'qwen3_5_moe', capability: 'chat', path: '/m/warm', warmupOf: () => report },
      async () => ({ warmup: report }),
    );
    await sampler!.timedLoad(
      { model: 'flat', modelType: 'qwen3_5', capability: 'chat', path: '/m/flat', warmupOf: () => null },
      async () => ({}),
    );
    const loads = records.filter((r): r is RuntimeLoadRecord => r.k === 'load');
    expect(loads[0]).toMatchObject({
      model: 'warm',
      ok: true,
      warmupMs: 3_400,
      warmupSteps: 50,
      warmupSkipped: 1,
      warmupReset: 0,
      warmupPipelines: 212,
      warmupCompileMs: 2_900.5,
      warmupPeakBytes: 4_096,
    });
    expect(Object.keys(loads[1]!).filter((key) => key.startsWith('warmup'))).toEqual([]);
  });

  // MUTATION: drop the `residencyFields` spread of the load record -> the fields are absent -> red.
  it('a load record carries the residency facts, numbers and the enum only, and none without residencyInfo', async () => {
    const records = start([]);
    const facts = {
      residency: 'host-gate',
      partialExpertSlots: 406,
      expertCount: 512,
      bankBudgetBytes: 61_000_000_000,
      liveNonReclaimableBytes: 30_000_000_000,
      anonymousNeedBytes: 9_000_000_000,
      hostWiredBytes: 12_000_000_000,
      hostCompressorBytes: 3_000_000_000,
      hostAnonymousBytes: 16_000_000_000,
      hostPurgeableBytes: 1_000_000_000,
      selfMlxBytes: 2_000_000_000,
      slotMisses: 77,
    };
    const load = (model: string, result: object) =>
      sampler!.timedLoad(
        { model, modelType: 'qwen4_exp', capability: 'chat', path: `/m/${model}`, residencyOf: readResidency },
        async () => result,
      );
    await load('flash', { residencyInfo: () => residencyInfo(facts) });
    await load('plain', {});
    await load('deepseek', { residencyInfo: () => ({ denseBytes: 1, bankSwitches: 2, inTurn: false }) });
    const loads = records.filter((r): r is RuntimeLoadRecord => r.k === 'load');
    expect(loads[0]).toMatchObject({
      model: 'flash',
      ok: true,
      residency: 'host-gate',
      expertSlots: 406,
      expertCount: 512,
      bankBudgetBytes: 61_000_000_000,
      liveNonReclaimableBytes: 30_000_000_000,
      anonymousNeedBytes: 9_000_000_000,
      hostWiredBytes: 12_000_000_000,
      hostCompressorBytes: 3_000_000_000,
      hostAnonymousBytes: 16_000_000_000,
      hostPurgeableBytes: 1_000_000_000,
      selfMlxBytes: 2_000_000_000,
    });
    expect(Object.keys(loads[0]!)).not.toContain('fallbackReason');
    expect(Object.keys(loads[0]!)).not.toContain('slotMisses');
    for (const later of loads.slice(1)) {
      expect(Object.keys(later).filter((key) => LOAD_RECORD_RESIDENCY_KEYS.includes(key))).toEqual([]);
    }
  });

  it('a sample carries the serving pipeline count only while the counter is alive', async () => {
    let stats: PipelineStats | null = null;
    const records = start([], { pipelineStats: () => stats });
    await vi.advanceTimersByTimeAsync(1);
    expect(samples(records)[0]).not.toHaveProperty('pipelinesCreatedWhileServing');
    stats = { createdLoading: 200, createdServing: 7, loadingMs: 70_000, servingMs: 900 };
    await vi.advanceTimersByTimeAsync(SAMPLE_IDLE_MS);
    expect(samples(records).at(-1)!.pipelinesCreatedWhileServing).toBe(7);
  });

  it('a live turn carries the phase and the three timers, rounded', async () => {
    const binding = fakeBinding();
    binding.state.live = 1;
    binding.state.turns = [
      {
        traceId: 't-prep',
        model: 'm',
        modelId: 'Qwen3.6-27B',
        ctxWindow: 65536,
        startedAt: T0,
        stream: true,
        dispatch: 'continue',
        progress: { phase: 'prepare', waitMs: 75_070.4, slotMs: 0, prepareMs: 300.6, cachedTokens: 19_000 },
        clientSession: null,
      },
      {
        traceId: 't-slot',
        model: 'm',
        modelId: 'Qwen3.6-27B',
        ctxWindow: 65536,
        startedAt: T0 + 1,
        stream: true,
        dispatch: 'start',
        progress: { phase: 'slot', waitMs: 2, slotMs: 1_500 },
        clientSession: 'acp-session-2',
      },
    ];
    start([binding.entry]);
    await landed();
    expect(
      readLive().turns.map((t) => [t.phase, t.waitMs, t.slotMs, t.prepareMs, t.cachedTokens, t.clientSession]),
    ).toEqual([
      ['prepare', 75_070, 0, 301, 19_000, null],
      ['slot', 2, 1_500, null, null, 'acp-session-2'],
    ]);
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
    await landed();
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

  // Finding: a burst that began and settled between two ticks left both busy flags false, so its deltas waited for the
  // 60 s idle sample (at 60 250 ms for a burst at 30 000 ms). MUTATION "write a sample only when a tick sees busy": the
  // burst's sample lands at 60 250 -> red.
  it('writes the sample of a burst no tick saw at the next tick', async () => {
    const { traced, gates, entry } = tracedBinding();
    const records = start([entry]);
    await vi.advanceTimersByTimeAsync(30_000);
    const turn = traced.chatSessionStart([]);
    await vi.advanceTimersByTimeAsync(100);
    gates[0]!();
    await turn;
    await vi.advanceTimersByTimeAsync(EDGE_COALESCE_MS);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(
      samples(records).map((s) => ({ at: s.ts - T0, busy: s.busy, busyMs: s.busyMs, runningMs: s.runningMs })),
    ).toEqual([
      { at: 0, busy: false, busyMs: 0, runningMs: 0 },
      { at: 30_000 + EDGE_COALESCE_MS, busy: false, busyMs: 100, runningMs: 100 },
      { at: 30_000 + EDGE_COALESCE_MS + SAMPLE_IDLE_MS, busy: false, busyMs: 0, runningMs: 0 },
    ]);
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
