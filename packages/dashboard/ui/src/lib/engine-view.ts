/** Pure transforms for the Engine page and the Overview engine panel. React-free. */

import type { CellSegment } from '../components/arc-x/cell-grid/cell-grid';
import type { ChartPoint } from '../components/arc-x/chart/types';
import type { TickerDelta, TickerItem } from '../components/arc-x/metric-ticker/metric-ticker';
import {
  formatBytes,
  formatBytesParts,
  formatClock,
  formatCount,
  formatCountNoun,
  formatDayLong,
  formatDuration,
  formatNumber,
  formatPercent,
  formatTps,
} from './format';
import type { LiveSample } from './live-history';
import { byteAxis } from './metrics-view';
import type { RuntimeLive, RuntimeProcessLive, RuntimeSeries, SchedSnapshot, SupervisorLive } from './types';

/** The process with that pid, else the first running, else the first unresponsive, else null. */
export function pickProcess(live: RuntimeLive | undefined, pid: number | null): RuntimeProcessLive | null {
  if (live === undefined) return null;
  const alive = live.processes.filter((process) => process.status !== 'dead');
  if (pid !== null) {
    const chosen = alive.find((process) => process.pid === pid);
    if (chosen !== undefined) return chosen;
  }
  return (
    alive.find((process) => process.status === 'running') ??
    alive.find((process) => process.status === 'unresponsive') ??
    null
  );
}

export function procKeyOf(process: { startedAt: number; pid: number }): string {
  return `${String(process.startedAt)}-${String(process.pid)}`;
}

export interface HistoryTotals {
  busyMs: number;
  /** Σ bucketMs of the buckets that hold a sample. */
  coveredMs: number;
  peakInFlight: number | null;
  peakQueued: number | null;
  rejected: number;
  loads: number;
  failedLoads: number;
  slowestLoadMs: number | null;
  starts: number;
  unclean: number;
}

function maxOf(values: readonly (number | null)[]): number | null {
  let best: number | null = null;
  for (const value of values)
    if (value !== null && Number.isFinite(value) && (best === null || value > best)) best = value;
  return best;
}

/**
 * Totals of the History section. `alive` holds the procKeys the live snapshot reports as running or unresponsive:
 * a process without an exit record that is not among them ended without a clean exit.
 */
export function historyTotals(
  series: RuntimeSeries,
  now: number,
  alive: ReadonlySet<string> = new Set(),
): HistoryTotals {
  let busyMs = 0;
  let coveredMs = 0;
  let rejected = 0;
  for (const point of series.points) {
    busyMs += point.busyMs;
    rejected += point.rejected;
    if (point.samples > 0) coveredMs += Math.min(series.bucketMs, Math.max(0, now - point.t));
  }
  const loadMs = series.loads.map((load) => load.ms);
  return {
    busyMs,
    coveredMs,
    peakInFlight: maxOf(series.points.map((point) => point.inFlightMax)),
    peakQueued: maxOf(series.points.map((point) => point.queuedMax)),
    rejected,
    loads: series.loads.length,
    failedLoads: series.loads.filter((load) => !load.ok).length,
    slowestLoadMs: maxOf(loadMs),
    starts: series.procs.filter((proc) => proc.startedAt >= series.range.from).length,
    unclean: series.procs.filter((proc) => proc.exitTs === null && !alive.has(proc.procKey)).length,
  };
}

/** Decode steps per batch size, sorted by size; the mean batch over all steps. */
export function occupancyPoints(delta: RuntimeSeries['sched'][number]['delta']): {
  data: ChartPoint[];
  mean: number | null;
  steps: number;
} {
  const rows = [...delta.occupancy].sort((a, b) => a.occupancy - b.occupancy);
  let steps = 0;
  let weighted = 0;
  const data = rows.map((row): ChartPoint => {
    steps += row.steps;
    weighted += row.occupancy * row.steps;
    return {
      key: String(row.occupancy),
      label: `Batch of ${String(row.occupancy)}`,
      axisLabel: String(row.occupancy),
      values: { steps: row.steps },
    };
  });
  return { data, mean: steps > 0 ? weighted / steps : null, steps };
}

export type BatchUse = 'batched' | 'alone' | 'quiet';

/**
 * How the model's turns ran in the range: `batched` when the scheduler admitted or stepped, `alone` when turns ran but
 * none went through the scheduler, `quiet` when no turn ran or the engine did not write the turn count.
 */
export function batchUse(entry: RuntimeSeries['sched'][number]): BatchUse {
  if (entry.delta.admitted > 0 || entry.delta.globalSteps > 0) return 'batched';
  return entry.latest.turns !== null && entry.delta.turns > 0 ? 'alone' : 'quiet';
}

/** The pool at rest as grid segments: in use = allocated − reclaimable; reclaimable is part of allocated. */
export function poolSegments(snapshot: SchedSnapshot): {
  segments: CellSegment[];
  reserved: number;
  threshold: number;
} {
  const reusable = Math.max(0, Math.min(snapshot.reclaimableBlocks, snapshot.allocatedBlocks));
  return {
    segments: [
      {
        key: 'used',
        label: 'In use',
        value: Math.max(0, snapshot.allocatedBlocks - reusable),
        color: 'var(--phase-decode)',
        pattern: 'solid',
      },
      { key: 'reusable', label: 'Reusable', value: reusable, color: 'var(--phase-decode)', pattern: 'hatch' },
    ],
    reserved: Math.max(0, snapshot.reservedBlocks),
    threshold: Math.max(0, snapshot.watermarkBlocks),
  };
}

const HOUR_MS = 3_600_000;

/** The bucket starts of the window a series answered, so a held answer is drawn on its own grid. */
export function seriesStarts(series: Pick<RuntimeSeries, 'range' | 'bucketMs'>): number[] {
  const starts: number[] = [];
  if (series.bucketMs <= 0) return starts;
  for (let t = series.range.from; t < series.range.to; t += series.bucketMs) starts.push(t);
  return starts;
}

/** Key, tooltip label and axis label of one runtime bucket. Hour buckets span days, so they carry the day. */
export function runtimeBucketPoint(t: number, bucketMs: number): Pick<ChartPoint, 'key' | 'label' | 'axisLabel'> {
  const clock = formatClock(t);
  if (bucketMs >= HOUR_MS) return { key: String(t), label: `${formatDayLong(t)}, ${clock}`, axisLabel: clock };
  return { key: String(t), label: clock, axisLabel: clock };
}

/** `minute`, `5 minutes`, `15 minutes`, `hour`. */
export function runtimeBucketWord(bucketMs: number): string {
  if (bucketMs >= HOUR_MS) return 'hour';
  const minutes = Math.round(bucketMs / 60_000);
  return minutes <= 1 ? 'minute' : `${String(minutes)} minutes`;
}

function pointsByStart(series: RuntimeSeries): Map<number, RuntimeSeries['points'][number]> {
  return new Map(series.points.map((point) => [point.t, point]));
}

/**
 * MLX active, MLX cache and the wired limit per bucket, in one byte unit. Buckets with no sample are gaps. No sample
 * counts: these are gauges read once per sampler tick, not statistics, so a quiet minute is not a low sample.
 */
export function memoryPoints(
  series: RuntimeSeries,
  starts: readonly number[],
  limitBytes: number | null,
): { data: ChartPoint[]; unit: 'MB' | 'GB'; div: number } {
  const byStart = pointsByStart(series);
  const max = maxOf([
    ...series.points.map((point) => point.memActiveAvg),
    ...series.points.map((point) => point.memCacheAvg),
    limitBytes,
  ]);
  const { unit, div } = byteAxis(max ?? 0);
  const scaled = (value: number | null): number | null => (value === null ? null : value / div);
  const data = starts.map((t): ChartPoint => {
    const point = byStart.get(t);
    const samples = point?.samples ?? 0;
    return {
      ...runtimeBucketPoint(t, series.bucketMs),
      values: {
        active: samples > 0 ? scaled(point?.memActiveAvg ?? null) : null,
        cache: samples > 0 ? scaled(point?.memCacheAvg ?? null) : null,
        limit: samples > 0 && limitBytes !== null && limitBytes > 0 ? limitBytes / div : null,
      },
    };
  });
  return { data, unit, div };
}

/** The highest number of turns in flight and queued per bucket; gaps where nothing was sampled. */
export function flightPoints(series: RuntimeSeries, starts: readonly number[]): ChartPoint[] {
  const byStart = pointsByStart(series);
  return starts.map((t): ChartPoint => {
    const point = byStart.get(t);
    const sampled = (point?.samples ?? 0) > 0;
    return {
      ...runtimeBucketPoint(t, series.bucketMs),
      values: {
        inFlight: sampled ? (point?.inFlightMax ?? null) : null,
        queued: sampled ? (point?.queuedMax ?? null) : null,
      },
    };
  });
}

/** '20+' when the supervisor's count sits at its cap, else the formatted count. */
export function nativeErrorText(supervisor: Pick<SupervisorLive, 'nativeErrorCount' | 'nativeErrorsCapped'>): string {
  return supervisor.nativeErrorsCapped ? '20+' : formatNumber(supervisor.nativeErrorCount);
}

const MINUTE_MS = 60_000;
const DELTA_SLACK_MS = 5_000;
const SPARK_WINDOW_MS = 600_000;
const SPARK_BUCKETS = 60;

type Pick1 = (sample: LiveSample) => number | null;

/** Samples of `pid` at or before `now`, oldest first. */
function samplesOf(history: readonly LiveSample[], pid: number, now: number): LiveSample[] {
  return history.filter((sample) => sample.pid === pid && sample.t <= now);
}

/**
 * The newest sample of `pid` minus its sample closest to 60 s earlier; null when no sample lies within
 * 60 s ± 5 s (the first minute, or after a gap), or when either reading is 0: a change from or to nothing is not a
 * trend, the value says it. Nothing is extrapolated.
 */
export function tickerDelta(history: readonly LiveSample[], pick: Pick1, now: number, pid: number): number | null {
  const own = samplesOf(history, pid, now);
  let newest: LiveSample | undefined;
  for (const sample of own) if (newest === undefined || sample.t >= newest.t) newest = sample;
  if (newest === undefined) return null;
  const current = pick(newest);
  if (current === null || current === 0) return null;
  const target = newest.t - MINUTE_MS;
  let best: LiveSample | undefined;
  for (const sample of own) {
    const off = Math.abs(sample.t - target);
    if (off > DELTA_SLACK_MS) continue;
    if (best === undefined || off < Math.abs(best.t - target)) best = sample;
  }
  if (best === undefined) return null;
  const before = pick(best);
  return before === null || before === 0 ? null : current - before;
}

/** The last 600 s of `pid` as 60 buckets of 10 s: each the mean of its samples, null when it holds none. */
export function sparkPoints(history: readonly LiveSample[], pick: Pick1, now: number, pid: number): (number | null)[] {
  const sums = Array.from({ length: SPARK_BUCKETS }, () => 0);
  const counts = Array.from({ length: SPARK_BUCKETS }, () => 0);
  const start = now - SPARK_WINDOW_MS;
  const step = SPARK_WINDOW_MS / SPARK_BUCKETS;
  for (const sample of samplesOf(history, pid, now)) {
    if (sample.t <= start) continue;
    const value = pick(sample);
    if (value === null) continue;
    const index = Math.min(SPARK_BUCKETS - 1, Math.floor((sample.t - start) / step));
    sums[index] += value;
    counts[index] += 1;
  }
  return sums.map((sum, index) => (counts[index] > 0 ? sum / counts[index] : null));
}

/**
 * The last `count` samples of `pid` as exactly `count` slots, oldest first: a null slot where two samples lie more than
 * `gapMs` apart, leading nulls while fewer exist. The fixed length lets a spark reshape per sample instead of remounting.
 */
export function recentSpark(
  history: readonly LiveSample[],
  pick: Pick1,
  pid: number,
  count: number,
  gapMs: number,
): (number | null)[] {
  const slots: (number | null)[] = [];
  let previous: number | null = null;
  for (const sample of history) {
    if (sample.pid !== pid) continue;
    if (previous !== null && sample.t - previous > gapMs) slots.push(null);
    slots.push(pick(sample));
    previous = sample.t;
  }
  const kept = slots.slice(Math.max(0, slots.length - count));
  const pad: (number | null)[] = Array.from({ length: count - kept.length }, () => null);
  return [...pad, ...kept];
}

const BYTE_DIV: Record<string, number> = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4, PB: 1024 ** 5 };

/** A signed change as a ticker chip: `+4.2`, `−1` (U+2212), `0`. */
export function tickerChip(diff: number | null, digits: number, goodWhen: TickerDelta['goodWhen']): TickerDelta | null {
  if (diff === null || !Number.isFinite(diff)) return null;
  const rounded = Number(diff.toFixed(digits));
  if (rounded === 0) return { text: '0', direction: 'flat', goodWhen };
  const magnitude = Math.abs(rounded).toFixed(digits);
  return rounded > 0
    ? { text: `+${magnitude}`, direction: 'up', goodWhen }
    : { text: `−${magnitude}`, direction: 'down', goodWhen };
}

const outTps: Pick1 = (sample) => sample.outTps;
const inFlight: Pick1 = (sample) => sample.inFlight;
const queued: Pick1 = (sample) => sample.queued;
const memActive: Pick1 = (sample) => sample.memActive;

function sum(values: readonly number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

/** The six items of the Engine "Now" ticker. The MLX memory item gets its meter footer from the page. */
export function nowItems(process: RuntimeProcessLive, history: readonly LiveSample[], now: number): TickerItem[] {
  const spark = (pick: Pick1, label: string): TickerItem['spark'] => ({
    data: sparkPoints(history, pick, now, process.pid),
    label: `${label}, last 10 minutes`,
  });
  const flight = sum(process.models.map((model) => model.inFlight));
  const waiting = sum(process.models.map((model) => model.queued));
  const limits = process.models.map((model) => model.queueLimit).filter((limit): limit is number => limit !== null);
  const mem = formatBytesParts(process.mem.active);
  const memDiv = BYTE_DIV[mem.unit] ?? 1;
  const memDelta = tickerDelta(history, memActive, now, process.pid);
  const wired = process.mem.wiredLimit;
  const finished = process.counters.ok + process.counters.error + process.counters.cancelled;
  return [
    {
      key: 'out',
      label: 'Output rate',
      value: formatTps(process.decodeTps ?? 0),
      unit: 'tok/s',
      delta: tickerChip(tickerDelta(history, outTps, now, process.pid), 1, 'up'),
      spark: spark(outTps, 'Output rate'),
      facts: [{ label: 'Turns', value: process.turns.length > 0 ? formatNumber(process.turns.length) : 'Idle' }],
    },
    {
      key: 'flight',
      label: 'In flight',
      value: formatNumber(flight),
      delta: tickerChip(tickerDelta(history, inFlight, now, process.pid), 0, 'neutral'),
      spark: spark(inFlight, 'In flight'),
      facts: [{ label: 'Max at once', value: formatNumber(sum(process.models.map((model) => model.maxConcurrent))) }],
    },
    {
      key: 'queued',
      label: 'Queued',
      value: formatNumber(waiting),
      delta: tickerChip(tickerDelta(history, queued, now, process.pid), 0, 'down'),
      spark: spark(queued, 'Queued'),
      facts: [{ label: 'Queue limit', value: limits.length > 0 ? formatNumber(sum(limits)) : 'none' }],
    },
    {
      key: 'mem',
      label: 'MLX memory',
      value: mem.value === '—' ? null : mem.value,
      unit: mem.unit === '' ? undefined : mem.unit,
      delta: tickerChip(memDelta === null ? null : memDelta / memDiv, 1, 'neutral'),
      spark: spark(memActive, 'MLX memory'),
      facts: [
        {
          label: 'Of the wired limit',
          value: wired > 0 ? formatPercent(process.mem.active / wired) : 'limit unknown',
        },
        { label: 'Wired limit', value: formatBytes(wired > 0 ? wired : null) },
      ],
    },
    {
      key: 'turns',
      label: 'Turns since start',
      value: formatCount(finished),
      facts: [
        { label: 'Failed', value: formatNumber(process.counters.error) },
        { label: 'Cancelled', value: formatNumber(process.counters.cancelled) },
        { label: 'Rejected', value: formatNumber(process.counters.rejected) },
      ],
    },
    {
      key: 'uptime',
      label: 'Uptime',
      value: formatDuration(Math.max(0, now - process.startedAt)),
      facts: [
        { label: 'Pid', value: String(process.pid) },
        { label: 'Host', value: process.host },
        { label: 'Version', value: `v${process.version}` },
      ],
    },
  ];
}

const STATUS_WORD: Record<RuntimeLive['status'], string> = {
  running: 'running',
  loading: 'loading a model',
  starting: 'starting',
  restarting: 'restarting',
  failed: 'failed',
  untrusted: 'not trustworthy',
  unresponsive: 'not responding',
  stopped: 'stopped',
};

/** The one polite sentence of the Engine page. */
export function engineSummary(live: RuntimeLive, process: RuntimeProcessLive | null): string {
  const head = `Inference ${STATUS_WORD[live.status]}.`;
  if (process === null) return head;
  const flight = sum(process.models.map((model) => model.inFlight));
  const wired = process.mem.wiredLimit;
  const memory = wired > 0 ? formatPercent(process.mem.active / wired) : 'unknown';
  return `${head} ${formatCountNoun(flight, 'turn', 'turns')} in flight. ${formatTps(process.decodeTps ?? 0)} tokens per second. Memory ${memory} of the limit.`;
}

/** `start`, `continue`, `after tool`. */
export function dispatchWord(dispatch: RuntimeProcessLive['turns'][number]['dispatch']): string {
  return dispatch === 'continue_tool' ? 'after tool' : dispatch;
}
