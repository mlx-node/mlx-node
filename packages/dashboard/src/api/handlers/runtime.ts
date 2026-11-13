/**
 * `GET /api/runtime/series`: the inference runtime series, bucketed.
 *
 * Within one bucket each process is averaged first and the processes are then
 * summed, so two processes alive at once add up while one process sampled
 * twice does not count twice. Scheduler numbers exist only at idle snapshots:
 * counters are reported as growth inside the range, gauges as read.
 */

import type { ApiRequest, WorkerApiContext } from '../context.js';
import { queryInt, toInt, toNum } from '../context.js';
import { ApiError } from '../errors.js';
import { MAX_BUCKETS, runtimeBucketSql } from '../sql.js';

export interface SchedSnapshot {
  ts: number;
  blockCapacity: number;
  allocatedBlocks: number;
  freeBlocks: number;
  reclaimableBlocks: number;
  watermarkBlocks: number;
  reservedBlocks: number;
  memoryCapacityBytes: number;
  /** Largest decode batch since the model was loaded. A high-water mark, not a counter. */
  maxBatchOccupancy: number;
  /** Turns sent to the model since its load, all lanes; null when the engine did not write the count. */
  turns: number | null;
}

const SCHED_COUNTERS = [
  'globalSteps',
  'admitted',
  'completed',
  'preemptions',
  'preemptionsRecompute',
  'preemptionsSsd',
  'admissionDeferredBlocks',
  'admissionDeferredState',
  'rowsAllocEvicted',
  'ssdRestoreBytes',
  'ssdRestoreWaitMs',
  'turns',
] as const;

type SchedCounter = (typeof SCHED_COUNTERS)[number];

export type SchedDelta = Record<SchedCounter, number> & {
  /** Decode steps per batch size. */
  occupancy: Array<{ occupancy: number; steps: number }>;
};

export interface RuntimeSeries {
  range: { from: number; to: number };
  bucketMs: number;
  points: Array<{
    t: number;
    samples: number;
    memActiveAvg: number | null;
    memActiveMax: number | null;
    memPeakMax: number | null;
    memCacheAvg: number | null;
    rssMax: number | null;
    inFlightAvg: number | null;
    inFlightMax: number | null;
    queuedMax: number | null;
    busyMs: number;
    okTurns: number;
    errorTurns: number;
    cancelledTurns: number;
    outTokens: number;
    rejected: number;
  }>;
  procs: Array<{
    procKey: string;
    pid: number;
    startedAt: number;
    host: string | null;
    version: string | null;
    port: number | null;
    chip: string | null;
    wiredLimitBytes: number | null;
    totalMemBytes: number | null;
    firstTs: number;
    lastTs: number;
    exitTs: number | null;
  }>;
  loads: Array<{
    ts: number;
    procKey: string;
    model: string;
    modelType: string | null;
    capability: string | null;
    ms: number | null;
    ok: boolean;
  }>;
  sched: Array<{
    procKey: string;
    model: string;
    latest: SchedSnapshot;
    snapshots: SchedSnapshot[];
    delta: SchedDelta;
  }>;
}

type Row = Record<string, unknown>;

function strOrNull(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function snapshotOf(row: Row): SchedSnapshot {
  return {
    ts: toInt(row.ts),
    blockCapacity: toInt(row.blockCapacity),
    allocatedBlocks: toInt(row.allocatedBlocks),
    freeBlocks: toInt(row.freeBlocks),
    reclaimableBlocks: toInt(row.reclaimableBlocks),
    watermarkBlocks: toInt(row.watermarkBlocks),
    reservedBlocks: toInt(row.reservedBlocks),
    memoryCapacityBytes: toInt(row.memoryCapacityBytes),
    maxBatchOccupancy: toInt(row.maxBatchOccupancy),
    turns: toNum(row.turns),
  };
}

function occupancyOf(text: unknown): Map<number, number> {
  const out = new Map<number, number>();
  if (typeof text !== 'string') return out;
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return out;
  }
  if (!Array.isArray(parsed)) return out;
  for (const pair of parsed) {
    if (!Array.isArray(pair)) continue;
    const occ = toNum(pair[0]);
    const steps = toNum(pair[1]);
    if (occ !== null && steps !== null) out.set(occ, steps);
  }
  return out;
}

/** Growth of a cumulative counter from `prev` to `cur`; a drop means the model was reloaded and `cur` is the growth. */
function growth(prev: number, cur: number): number {
  return cur >= prev ? cur - prev : cur;
}

/**
 * Walk one (process, model) series: `baseline` is the last snapshot before the range, `inRange` the ones inside it.
 * No baseline means no earlier snapshot exists at all; counters start at zero on load, so the walk starts from an
 * all-zero row (the first snapshot comes after the burst that loaded the model). `reloads` are the end times of the
 * successful loads of this model in this process: a snapshot after one belongs to a fresh instance, whose counters
 * also start at zero even when they have already passed the old instance's values.
 */
function deltaOf(baseline: Row | undefined, inRange: Row[], reloads: number[]): SchedDelta {
  const delta = { occupancy: [] } as unknown as SchedDelta;
  for (const key of SCHED_COUNTERS) delta[key] = 0;
  const occ = new Map<number, number>();
  let prev: Row = baseline ?? {};
  for (const cur of inRange) {
    if (prev.ts !== undefined) {
      const prevTs = toInt(prev.ts);
      const curTs = toInt(cur.ts);
      if (reloads.some((ts) => ts > prevTs && ts <= curTs)) prev = {};
    }
    for (const key of SCHED_COUNTERS) delta[key] += growth(toNum(prev[key]) ?? 0, toNum(cur[key]) ?? 0);
    const before = occupancyOf(prev.occHist);
    for (const [size, steps] of occupancyOf(cur.occHist)) {
      occ.set(size, (occ.get(size) ?? 0) + growth(before.get(size) ?? 0, steps));
    }
    prev = cur;
  }
  delta.occupancy = [...occ]
    .filter(([, steps]) => steps > 0)
    .sort((a, b) => a[0] - b[0])
    .map(([occupancy, steps]) => ({ occupancy, steps }));
  return delta;
}

export function handleRuntimeSeries(ctx: WorkerApiContext, req: ApiRequest): RuntimeSeries {
  const from = queryInt(req.query, 'from');
  const to = queryInt(req.query, 'to');
  const bucketMs = queryInt(req.query, 'bucketMs');
  if (from === null || to === null || bucketMs === null) {
    throw ApiError.badRequest('from, to and bucketMs are required');
  }
  if (from > to) throw ApiError.badRequest('from must not be after to');
  const t = runtimeBucketSql(bucketMs);
  if ((to - from) / bucketMs > MAX_BUCKETS)
    throw ApiError.badRequest(`The range spans more than ${MAX_BUCKETS} buckets`);
  const { sqlite } = ctx.dash;

  const points = (
    sqlite
      .prepare(
        `SELECT t, SUM(n) AS samples, SUM(a) AS memActiveAvg, SUM(m) AS memActiveMax, SUM(pk) AS memPeakMax,
                SUM(c) AS memCacheAvg, SUM(r) AS rssMax, SUM(ifa) AS inFlightAvg, SUM(ifm) AS inFlightMax,
                SUM(q) AS queuedMax, SUM(b) AS busyMs, SUM(ok) AS okTurns, SUM(er) AS errorTurns,
                SUM(ca) AS cancelledTurns, SUM(tok) AS outTokens, SUM(rej) AS rejected
         FROM (SELECT ${t} AS t, proc_key, COUNT(*) AS n,
                      AVG(mem_active) AS a, MAX(mem_active) AS m, MAX(mem_peak) AS pk, AVG(mem_cache) AS c,
                      MAX(rss) AS r, AVG(in_flight) AS ifa, MAX(in_flight) AS ifm, MAX(queued) AS q,
                      SUM(COALESCE(busy_ms, 0)) AS b, SUM(COALESCE(d_ok, 0)) AS ok, SUM(COALESCE(d_error, 0)) AS er,
                      SUM(COALESCE(d_cancelled, 0)) AS ca, SUM(COALESCE(d_out_tokens, 0)) AS tok,
                      SUM(COALESCE(d_rejected, 0)) AS rej
               FROM runtime_samples WHERE ts >= ? AND ts <= ? GROUP BY t, proc_key)
         GROUP BY t ORDER BY t`,
      )
      .all(from, to) as Row[]
  ).map((row) => ({
    t: toInt(row.t),
    samples: toInt(row.samples),
    memActiveAvg: toNum(row.memActiveAvg),
    memActiveMax: toNum(row.memActiveMax),
    memPeakMax: toNum(row.memPeakMax),
    memCacheAvg: toNum(row.memCacheAvg),
    rssMax: toNum(row.rssMax),
    inFlightAvg: toNum(row.inFlightAvg),
    inFlightMax: toNum(row.inFlightMax),
    queuedMax: toNum(row.queuedMax),
    busyMs: toNum(row.busyMs) ?? 0,
    okTurns: toInt(row.okTurns),
    errorTurns: toInt(row.errorTurns),
    cancelledTurns: toInt(row.cancelledTurns),
    outTokens: toInt(row.outTokens),
    rejected: toInt(row.rejected),
  }));

  const procs = (
    sqlite
      .prepare(
        `SELECT proc_key AS procKey, pid, started_at AS startedAt, host, version, port, chip,
                wired_limit_bytes AS wiredLimitBytes, total_mem_bytes AS totalMemBytes,
                first_ts AS firstTs, last_ts AS lastTs, exit_ts AS exitTs
         FROM runtime_procs WHERE last_ts >= ? AND first_ts <= ? ORDER BY started_at DESC, proc_key`,
      )
      .all(from, to) as Row[]
  ).map((row) => ({
    procKey: String(row.procKey),
    pid: toInt(row.pid),
    startedAt: toInt(row.startedAt),
    host: strOrNull(row.host),
    version: strOrNull(row.version),
    port: toNum(row.port),
    chip: strOrNull(row.chip),
    wiredLimitBytes: toNum(row.wiredLimitBytes),
    totalMemBytes: toNum(row.totalMemBytes),
    firstTs: toInt(row.firstTs),
    lastTs: toInt(row.lastTs),
    exitTs: toNum(row.exitTs),
  }));

  const loads = (
    sqlite
      .prepare(
        `SELECT ts, proc_key AS procKey, model, model_type AS modelType, capability, ms, ok
         FROM runtime_loads WHERE ts >= ? AND ts <= ? ORDER BY ts`,
      )
      .all(from, to) as Row[]
  ).map((row) => ({
    ts: toInt(row.ts),
    procKey: String(row.procKey),
    model: String(row.model),
    modelType: strOrNull(row.modelType),
    capability: strOrNull(row.capability),
    ms: toNum(row.ms),
    ok: toInt(row.ok) === 1,
  }));

  const reloads = new Map<string, number[]>();
  for (const row of sqlite
    .prepare(`SELECT proc_key AS procKey, model, ts FROM runtime_loads WHERE ok = 1 AND ts <= ? ORDER BY ts`)
    .all(to) as Row[]) {
    const key = `${String(row.procKey)}\u0000${String(row.model)}`;
    const list = reloads.get(key);
    if (list === undefined) reloads.set(key, [toInt(row.ts)]);
    else list.push(toInt(row.ts));
  }

  const snapRows = sqlite
    .prepare(
      `SELECT proc_key AS procKey, model, ts, block_capacity AS blockCapacity, allocated_blocks AS allocatedBlocks,
              free_blocks AS freeBlocks, reclaimable_blocks AS reclaimableBlocks, watermark_blocks AS watermarkBlocks,
              reserved_blocks AS reservedBlocks, memory_capacity_bytes AS memoryCapacityBytes,
              max_batch_occupancy AS maxBatchOccupancy, global_steps AS globalSteps, admitted, completed,
              preemptions, preemptions_recompute AS preemptionsRecompute, preemptions_ssd AS preemptionsSsd,
              admission_deferred_blocks AS admissionDeferredBlocks, admission_deferred_state AS admissionDeferredState,
              rows_alloc_evicted AS rowsAllocEvicted, ssd_restore_bytes AS ssdRestoreBytes,
              ssd_restore_wait_ms AS ssdRestoreWaitMs, turns, occ_hist AS occHist
       FROM runtime_sched WHERE ts <= ? ORDER BY proc_key, model, ts, id`,
    )
    .all(to) as Row[];
  const groups = new Map<string, { procKey: string; model: string; baseline: Row | undefined; inRange: Row[] }>();
  for (const row of snapRows) {
    const key = `${String(row.procKey)}\u0000${String(row.model)}`;
    let group = groups.get(key);
    if (group === undefined) {
      group = { procKey: String(row.procKey), model: String(row.model), baseline: undefined, inRange: [] };
      groups.set(key, group);
    }
    if (toInt(row.ts) < from) group.baseline = row;
    else group.inRange.push(row);
  }
  const sched: RuntimeSeries['sched'] = [];
  for (const group of groups.values()) {
    const latest = group.inRange[group.inRange.length - 1] ?? group.baseline;
    if (latest === undefined) continue;
    sched.push({
      procKey: group.procKey,
      model: group.model,
      latest: snapshotOf(latest),
      snapshots: group.inRange.map(snapshotOf),
      delta: deltaOf(group.baseline, group.inRange, reloads.get(`${group.procKey}\u0000${group.model}`) ?? []),
    });
  }

  return { range: { from, to }, bucketMs, points, procs, loads, sched };
}
