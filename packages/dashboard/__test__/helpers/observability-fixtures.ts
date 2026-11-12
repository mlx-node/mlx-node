/** Builders for the API bodies the Overview, Metrics, Engine and Cache pages read. Every field is overridable. */

import type {
  CacheResponse,
  MetricsDistributions,
  MetricsSpeculative,
  MetricsSummary,
  MetricsTimeseries,
  MetricsTotals,
  MetricsTurns,
  Quantiles,
  RuntimeLive,
  RuntimeProcessLive,
  RuntimeSeries,
  SchedSnapshot,
  TurnRow,
} from '../../ui/src/lib/types.js';

export const MIB = 1024 * 1024;
export const GIB = 1024 * MIB;

/** One clock for every builder, so a process and the series rows keyed by its start agree. */
export const T0 = Date.now();

export function quantiles(p50: number | null, n: number, spread = 0.5): Quantiles {
  if (p50 === null || n === 0) return { n, p05: null, p50: null, p95: null };
  return { n, p05: p50 * (1 - spread), p50, p95: p50 * (1 + spread) };
}

export function totals(overrides: Partial<MetricsTotals> = {}): MetricsTotals {
  return {
    turns: 0,
    ok: 0,
    error: 0,
    cancelled: 0,
    prefillMeasured: 0,
    inputTokens: 0,
    cachedTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    queueWallMs: 0,
    loadWaitWallMs: 0,
    resolveWallMs: 0,
    firstTokenWallMs: 0,
    setupWallMs: 0,
    prefillWallMs: 0,
    decodeWallMs: 0,
    otherWallMs: 0,
    totalWallMs: 0,
    ...overrides,
  };
}

/** A busy total: 100 turns, 3 failed, 2 cancelled, every wall-time bucket non-zero. */
export function busyTotals(overrides: Partial<MetricsTotals> = {}): MetricsTotals {
  return totals({
    turns: 100,
    ok: 95,
    error: 3,
    cancelled: 2,
    prefillMeasured: 95,
    inputTokens: 40_000,
    cachedTokens: 160_000,
    outputTokens: 25_000,
    reasoningTokens: 5_000,
    queueWallMs: 2_000,
    loadWaitWallMs: 1_000,
    resolveWallMs: 8_000,
    setupWallMs: 4_000,
    prefillWallMs: 16_000,
    decodeWallMs: 60_000,
    otherWallMs: 1_000,
    totalWallMs: 92_000,
    ...overrides,
  });
}

export function latency(n = 100): MetricsSummary['latency'] {
  return {
    ttftMs: quantiles(400, n),
    setupMs: quantiles(80, n),
    decodeTps: quantiles(42, n),
    prefillTps: quantiles(1800, n),
    totalMs: quantiles(3400, n),
    queueMs: quantiles(2, n),
    ctxFillPct: quantiles(40, n),
  };
}

export function byModelRow(
  model: string,
  turns: number,
  overrides: Partial<MetricsSummary['byModel'][number]> = {},
): MetricsSummary['byModel'][number] {
  return {
    model,
    modelType: 'qwen3_5',
    ctxWindow: 131_072,
    totals: busyTotals({ turns, ok: turns, error: 0, cancelled: 0 }),
    latency: latency(turns),
    mtpTurns: 0,
    lastTs: Date.now() - 60_000,
    ...overrides,
  };
}

export function summary(overrides: Partial<MetricsSummary> = {}): MetricsSummary {
  return {
    range: { from: null, to: null },
    model: null,
    totals: busyTotals(),
    latency: latency(),
    previous: { totals: busyTotals({ turns: 80, ok: 78, error: 2, cancelled: 0 }), latency: latency(80) },
    byModel: [byModelRow('qwen3-8b', 60), byModelRow('gemma-4-e4b', 40, { modelType: 'gemma4' })],
    finishReasons: [
      { reason: 'stop', turns: 80 },
      { reason: 'tool_calls', turns: 15 },
      { reason: 'length', turns: 5 },
    ],
    failures: [
      { outcome: 'cancelled', errorKind: null, failPhase: 'decode', turns: 2 },
      { outcome: 'error', errorKind: 'exception', failPhase: 'prefill', turns: 3 },
    ],
    endpoints: [{ endpoint: '/v1/messages', stream: true, turns: 100 }],
    dispatches: [
      { dispatch: 'start', turns: 20, promptTokens: 20_000, cachedTokens: 2_000 },
      { dispatch: 'continue', turns: 50, promptTokens: 100_000, cachedTokens: 94_000 },
      { dispatch: 'continue_tool', turns: 30, promptTokens: 80_000, cachedTokens: 64_000 },
    ],
    ...overrides,
  };
}

/** An empty window: no turns, every median null. */
export function emptySummary(): MetricsSummary {
  return summary({
    totals: totals(),
    latency: {
      ttftMs: quantiles(null, 0),
      setupMs: quantiles(null, 0),
      decodeTps: quantiles(null, 0),
      prefillTps: quantiles(null, 0),
      totalMs: quantiles(null, 0),
      queueMs: quantiles(null, 0),
      ctxFillPct: quantiles(null, 0),
    },
    previous: null,
    byModel: [],
    finishReasons: [],
    failures: [],
    endpoints: [],
    dispatches: [],
  });
}

type TsPoint = MetricsTimeseries['points'][number];

export function tsPoint(t: number, model: string, overrides: Partial<TsPoint> = {}): TsPoint {
  return {
    t,
    model,
    totals: busyTotals({ turns: 10, ok: 9, error: 1, cancelled: 0 }),
    ttftMs: quantiles(400, 10),
    decodeTps: quantiles(42, 10),
    prefillTps: quantiles(1800, 10),
    totalMs: quantiles(3400, 10),
    mtpMeanAccepted: null,
    mtpTurns: 0,
    coldHits: 0,
    coldMisses: 0,
    concurrencyPeak: 1,
    ...overrides,
  };
}

export function timeseries(points: TsPoint[], groupBy: 'model' | 'none' = 'model'): MetricsTimeseries {
  return { range: { from: null, to: null }, bucket: 'day', groupBy, points };
}

export function distributions(n = 10): MetricsDistributions {
  const hist = (bins: number) => ({ n, counts: Array.from({ length: bins }, (_, i) => (i === 1 ? n : 0)) });
  return {
    range: { from: null, to: null },
    groupBy: 'none',
    edges: {
      ttftMs: [0, 100, 1000],
      decodeTps: [0, 20, 40],
      prefillTps: [0, 500, 1000],
      totalMs: [0, 1000, 10_000],
      ctxFillPct: [0, 25, 50],
      promptTokens: [0, 1000, 10_000],
      outputTokens: [0, 100, 1000],
    },
    series: [
      {
        model: 'all',
        ttftMs: hist(3),
        decodeTps: hist(3),
        prefillTps: hist(3),
        totalMs: hist(3),
        ctxFillPct: hist(3),
        promptTokens: hist(3),
        outputTokens: hist(3),
      },
    ],
  };
}

export function speculative(models: MetricsSpeculative['models'] = []): MetricsSpeculative {
  return { range: { from: null, to: null }, models, trend: [] };
}

export function turnRow(traceId: string, overrides: Partial<TurnRow> = {}): TurnRow {
  return {
    traceId,
    ts: Date.now() - 60_000,
    model: 'qwen3-8b',
    alias: null,
    modelType: 'qwen3',
    outcome: 'ok',
    errorKind: null,
    failPhase: null,
    finishReason: 'stop',
    endpoint: '/v1/messages',
    stream: true,
    dispatch: 'continue',
    promptTokens: 12_000,
    cachedTokens: 9_000,
    outputTokens: 400,
    reasoningTokens: 0,
    ctxWindow: 131_072,
    ctxFillPct: 10,
    preDispatchMs: 50,
    queueMs: 10,
    loadWaitMs: 0,
    resolveMs: 0,
    ttftMs: 300,
    prefillMs: 250,
    prefillTokens: 2_500,
    setupMs: 50,
    decodeMs: 9000,
    durationMs: 9400,
    prefillTps: 10_000,
    decodeTps: 44,
    mtpCycles: null,
    mtpMeanAccepted: null,
    coldHits: null,
    coldMisses: null,
    toolCount: 4,
    toolCallCount: 1,
    imageCount: null,
    concurrency: 1,
    concurrencyPeak: 1,
    memActiveBytes: 8 * GIB,
    pid: 812,
    ...overrides,
  };
}

export function turnsPage(rows: TurnRow[], total = rows.length, next: MetricsTurns['next'] = null): MetricsTurns {
  return { range: { from: null, to: null }, rows, next, total };
}

type LiveModel = RuntimeProcessLive['models'][number];
type LiveTurn = RuntimeProcessLive['turns'][number];

export function liveModel(name: string, overrides: Partial<LiveModel> = {}): LiveModel {
  return {
    name,
    modelType: 'qwen3_5',
    ctxWindow: 131_072,
    paged: true,
    maxConcurrent: 4,
    loadedAt: Date.now() - 3_600_000,
    loadMs: 8_200,
    inFlight: 0,
    queued: 0,
    queueLimit: 16,
    pool: {
      at: Date.now() - 30_000,
      blockCapacity: 4096,
      allocatedBlocks: 1716,
      freeBlocks: 2380,
      reclaimableBlocks: 400,
    },
    ...overrides,
  };
}

export function liveTurn(traceId: string, overrides: Partial<LiveTurn> = {}): LiveTurn {
  return {
    traceId,
    model: 'qwen3-8b',
    modelId: null,
    ctxWindow: 131_072,
    startedAt: Date.now() - 12_000,
    stream: true,
    dispatch: 'continue',
    phase: 'decode',
    promptTokens: 31_000,
    cachedTokens: 30_000,
    prefillTotal: 1_000,
    prefillDone: 1_000,
    generatedTokens: 1_204,
    ...overrides,
  };
}

export function process(overrides: Partial<RuntimeProcessLive> = {}): RuntimeProcessLive {
  return {
    status: 'running',
    ageMs: 400,
    pid: 812,
    startedAt: T0 - 3 * 3_600_000,
    ts: Date.now(),
    host: 'desktop',
    version: '0.0.21',
    port: 52114,
    health: 'ok',
    mem: { active: 18 * GIB, peak: 21 * GIB, cache: 2 * GIB, wiredLimit: 30 * GIB, rss: 1.2 * GIB },
    decodeTps: 42.7,
    loading: null,
    counters: { ok: 1200, error: 3, cancelled: 1, outputTokens: 400_000, loads: 2, rejected: 0 },
    models: [liveModel('qwen3-8b', { inFlight: 1 })],
    turns: [liveTurn('live-1')],
    ...overrides,
  };
}

export function live(overrides: Partial<RuntimeLive> = {}): RuntimeLive {
  return { now: Date.now(), status: 'running', supervisor: null, processes: [process()], ...overrides };
}

export function sched(overrides: Partial<SchedSnapshot> = {}): SchedSnapshot {
  return {
    ts: Date.now() - 30_000,
    blockCapacity: 4096,
    allocatedBlocks: 1716,
    freeBlocks: 2380,
    reclaimableBlocks: 400,
    watermarkBlocks: 200,
    reservedBlocks: 30,
    memoryCapacityBytes: 8 * GIB,
    maxBatchOccupancy: 4,
    turns: 430,
    ...overrides,
  };
}

export function runtimeSeries(overrides: Partial<RuntimeSeries> = {}): RuntimeSeries {
  const bucketMs = 60_000;
  const to = Math.floor(Date.now() / bucketMs) * bucketMs + bucketMs;
  const from = to - 60 * bucketMs;
  const proc = process();
  return {
    range: { from, to },
    bucketMs,
    points: [
      {
        t: from + 50 * bucketMs,
        samples: 6,
        memActiveAvg: 18 * GIB,
        memActiveMax: 19 * GIB,
        memPeakMax: 21 * GIB,
        memCacheAvg: 2 * GIB,
        rssMax: 1.2 * GIB,
        inFlightAvg: 1,
        inFlightMax: 2,
        queuedMax: 1,
        busyMs: 30_000,
        okTurns: 4,
        errorTurns: 0,
        cancelledTurns: 0,
        outTokens: 2_000,
        rejected: 2,
      },
    ],
    procs: [
      {
        procKey: `${String(proc.startedAt)}-${String(proc.pid)}`,
        pid: proc.pid,
        startedAt: proc.startedAt,
        host: 'desktop',
        version: '0.0.21',
        port: 52114,
        chip: 'Apple M5 Max',
        wiredLimitBytes: 30 * GIB,
        totalMemBytes: 128 * GIB,
        firstTs: proc.startedAt,
        lastTs: Date.now(),
        exitTs: null,
      },
    ],
    loads: [
      {
        ts: from + 5 * bucketMs,
        procKey: `${String(proc.startedAt)}-${String(proc.pid)}`,
        model: 'qwen3-8b',
        modelType: 'qwen3',
        capability: 'chat',
        ms: 8_200,
        ok: true,
      },
    ],
    sched: [
      {
        procKey: `${String(proc.startedAt)}-${String(proc.pid)}`,
        model: 'qwen3-8b',
        latest: sched(),
        snapshots: [sched()],
        delta: {
          globalSteps: 81_204,
          admitted: 412,
          completed: 410,
          preemptions: 0,
          preemptionsRecompute: 0,
          preemptionsSsd: 0,
          admissionDeferredBlocks: 3,
          admissionDeferredState: 0,
          rowsAllocEvicted: 0,
          ssdRestoreBytes: 0,
          ssdRestoreWaitMs: 0,
          turns: 415,
          occupancy: [
            { occupancy: 1, steps: 100 },
            { occupancy: 2, steps: 300 },
            { occupancy: 4, steps: 100 },
          ],
        },
      },
    ],
    ...overrides,
  };
}

/** Local midnight `daysAgo` days before today. */
export function dayStart(daysAgo: number): number {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth(), now.getDate() - daysAgo).getTime();
}

export function cacheResponse(overrides: Partial<CacheResponse> = {}): CacheResponse {
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
      newestMtime: Date.now() - 3_600_000,
      ageHistogram: [
        { label: '<1d', count: 100, bytes: 100 * MIB },
        { label: '1-7d', count: 30, bytes: 150 * MIB },
        { label: '7-30d', count: 6, bytes: 40 * MIB },
        { label: '>30d', count: 2, bytes: 10 * MIB },
      ],
    },
    trend: [
      { t: dayStart(2), hits: 16_000, misses: 40, bytesWritten: 8 * MIB, bytesRestored: 64 * MIB },
      { t: dayStart(1), hits: 189, misses: 9, bytesWritten: 2 * MIB, bytesRestored: 16 * MIB },
    ],
    scope: {
      root: '/private/tmp/cold/mlx-paged-v1',
      trendWindowDays: 30,
      legacy: { turns: 0, hits: 0, misses: 0 },
      otherRoots: { turns: 0, hits: 0, misses: 0 },
      unattributed: { turns: 0, hits: 0, misses: 0 },
      disabledTurns: 0,
      unrootedSidecarCaptures: 0,
    },
    health: {
      enqueued: 500,
      queueDrops: 0,
      evictions: 4,
      corruptions: 0,
      corruptionsTotal: 0,
      queueDropsTotal: 0,
      writeErrors: 0,
      writeErrorsTotal: 0,
      restoreDeclines: 0,
      sidecarCaptureReached: 20,
      sidecarChainEmpty: 1,
      sidecarBoundarySkips: 2,
      sidecarAlreadyPersisted: 16,
      sidecarEnqueued: 1,
      sidecarQueueDrops: 0,
      sidecarInstalled: 15,
    },
    restoreFamilies: ['gemma4', 'qwen3', 'qwen3_5', 'qwen3_5_moe'],
    ...overrides,
  };
}
