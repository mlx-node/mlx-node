/** Pure transforms for the Engine page and the Overview engine panel. React-free. */

import type { StackPoint } from '../components/arc-x/activity-strip/phase-stack';
import type { CellSegment } from '../components/arc-x/cell-grid/cell-grid';
import type { ChartPoint } from '../components/arc-x/chart/types';
import type { EngineStateKey, LaneCounts } from '../components/arc-x/live-turn-row/live-turn-row';
import type { TickerDelta, TickerItem } from '../components/arc-x/metric-ticker/metric-ticker';
import { fill } from './chat-strings';
import {
  formatBytes,
  formatBytesParts,
  formatClock,
  formatCount,
  formatDayLong,
  formatDuration,
  formatNumber,
  formatPercent,
  formatTps,
} from './format';
import { LIVE_GAP_MS, type LiveSample } from './live-history';
import { BANDS, ENGINE_STATE, LANES, METRIC, NOW_FACTS, WINDOW } from './metric-strings';
import { byteAxis } from './metrics-view';
import type {
  HostKind,
  RuntimeLive,
  RuntimeProcessLive,
  RuntimeSeries,
  SchedSnapshot,
  SupervisorLive,
  WorkMs,
} from './types';

export const HOST_WORD: Record<HostKind, string> = {
  desktop: 'Portato',
  serve: 'mlx serve',
  launch: 'mlx launch',
  embedded: 'embedded',
};

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

const outputTps: Pick1 = (sample) => sample.outputTps;
const prefillTps: Pick1 = (sample) => sample.prefillTps;
const inFlight: Pick1 = (sample) => sample.inFlight;
const queued: Pick1 = (sample) => sample.queued;
const memActive: Pick1 = (sample) => sample.memActive;

function sum(values: readonly number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

/** The engine's state word and its lanes per phase, from the newest snapshot. */
export interface EngineState {
  key: EngineStateKey;
  label: string;
  /** Lanes the engine has taken. `other`: preparing, non-streamed turns and System One decisions. */
  lanes: LaneCounts;
}

function stateKey(process: RuntimeProcessLive, lanes: LaneCounts): EngineStateKey {
  if (process.status !== 'running') return 'unresponsive';
  if (lanes.decode > 0) return lanes.prefill > 0 ? 'prefillDecode' : 'decode';
  if (lanes.prefill > 0) return 'prefill';
  if (lanes.other > 0) return 'working';
  if (lanes.waiting > 0) return 'waiting';
  if (process.loading.length > 0) return 'loading';
  return sum(process.models.map((model) => model.queued)) > 0 ? 'queued' : 'idle';
}

/**
 * The engine state from the newest snapshot: a lane in decode or prefill names it; lanes that only prepare or report
 * no progress are `working`; lanes that all wait are `waiting`. With no lane: a load in flight, then the server
 * queue, then idle. A model's `inFlight` beyond its listed turns is a lane without progress (a System One decision).
 */
export function engineState(process: RuntimeProcessLive): EngineState {
  const lanes: LaneCounts = { decode: 0, prefill: 0, other: 0, waiting: 0 };
  for (const turn of process.turns) {
    if (turn.phase === 'decode') lanes.decode += 1;
    else if (turn.phase === 'prefill') lanes.prefill += 1;
    else if (turn.phase === 'queued' || turn.phase === 'slot') lanes.waiting += 1;
    else lanes.other += 1;
  }
  lanes.other += Math.max(0, sum(process.models.map((model) => model.inFlight)) - process.turns.length);
  const key = stateKey(process, lanes);
  return { key, label: ENGINE_STATE[key], lanes };
}

const WORK_KEYS: ReadonlyArray<keyof WorkMs> = ['loading', 'waiting', 'prepare', 'prefill', 'decode', 'running'];

/**
 * Lanes by phase of `pid`, one interval per pair of its consecutive samples: each band's growth in lane-ms ÷ the time
 * between the two writes (`at`). A pair is skipped when no time passed (the same file read twice), when the writes lie
 * more than `LIVE_GAP_MS` apart, or when a counter went down (a new process under the same pid). `other` is
 * preparing, lanes without progress and model loading.
 */
export function stackPoints(history: readonly LiveSample[], pid: number): StackPoint[] {
  const out: StackPoint[] = [];
  let previous: { at: number; work: WorkMs } | null = null;
  for (const sample of history) {
    if (sample.pid !== pid || sample.at === null || sample.work === null) continue;
    const current = { at: sample.at, work: sample.work };
    if (previous !== null) {
      const span = current.at - previous.at;
      const before = previous.work;
      if (span > 0 && span <= LIVE_GAP_MS && WORK_KEYS.every((key) => current.work[key] >= before[key])) {
        const lanes = (key: keyof WorkMs): number => (current.work[key] - before[key]) / span;
        out.push({
          t0: previous.at,
          t1: current.at,
          values: {
            decode: lanes('decode'),
            prefill: lanes('prefill'),
            other: lanes('prepare') + lanes('running') + lanes('loading'),
            waiting: lanes('waiting'),
          },
        });
      }
    }
    previous = current;
  }
  return out;
}

/** A rate as the page prints it, or null (drawn "—") when the server sent none: no 0 is invented. */
export function rateText(tps: number | null): string | null {
  return tps === null ? null : formatTps(tps);
}

/** The seven items of the Engine "Now" ticker. The MLX memory item gets its meter footer from the page. */
export function nowItems(process: RuntimeProcessLive, history: readonly LiveSample[], now: number): TickerItem[] {
  const spark = (pick: Pick1, label: string): TickerItem['spark'] => ({
    data: sparkPoints(history, pick, now, process.pid),
    label: `${label}, last 10 minutes`,
  });
  const lanes = engineState(process).lanes;
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
      label: METRIC.outputRate.label,
      value: rateText(process.rates.outputTps),
      unit: METRIC.outputRate.unit,
      delta: tickerChip(tickerDelta(history, outputTps, now, process.pid), 1, 'up'),
      spark: spark(outputTps, METRIC.outputRate.label),
      facts: [
        { label: NOW_FACTS.lanesDecoding, value: formatNumber(lanes.decode) },
        { label: NOW_FACTS.window, value: WINDOW.last10s },
      ],
    },
    {
      key: 'prefill',
      label: METRIC.prefillSpeed.label,
      value: rateText(process.rates.prefillTps),
      unit: METRIC.prefillSpeed.unit,
      spark: spark(prefillTps, METRIC.prefillSpeed.label),
      facts: [
        { label: NOW_FACTS.lanesInPrefill, value: formatNumber(lanes.prefill) },
        { label: NOW_FACTS.cachedTokens, value: NOW_FACTS.notCounted },
      ],
    },
    {
      key: 'flight',
      label: METRIC.inFlight.label,
      value: formatNumber(flight),
      delta: tickerChip(tickerDelta(history, inFlight, now, process.pid), 0, 'neutral'),
      spark: spark(inFlight, METRIC.inFlight.label),
      facts: [
        { label: 'Max at once', value: formatNumber(sum(process.models.map((model) => model.maxConcurrent))) },
        { label: BANDS.decode, value: formatNumber(lanes.decode) },
        { label: BANDS.prefill, value: formatNumber(lanes.prefill) },
        { label: BANDS.waiting, value: formatNumber(lanes.waiting) },
      ],
    },
    {
      key: 'queued',
      label: METRIC.queued.label,
      value: formatNumber(waiting),
      delta: tickerChip(tickerDelta(history, queued, now, process.pid), 0, 'down'),
      spark: spark(queued, METRIC.queued.label),
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

/** "2 lanes: 1 decode, 1 prefill." With no lane the state word ("Idle."), none when the status already says it. */
function laneSentence(state: EngineState): string | null {
  const parts: Array<[number, string]> = [
    [state.lanes.decode, LANES.decoding],
    [state.lanes.prefill, LANES.prefilling],
    [state.lanes.other, LANES.other],
    [state.lanes.waiting, LANES.waiting],
  ];
  const total = sum(parts.map(([count]) => count));
  if (total <= 0) return state.key === 'unresponsive' ? null : `${state.label}.`;
  const head = total === 1 ? LANES.one : fill(LANES.many, { n: formatNumber(total) });
  const counts = parts.filter(([count]) => count > 0).map(([count, word]) => fill(word, { n: formatNumber(count) }));
  return `${head}: ${counts.join(', ')}.`;
}

/** The one polite sentence of the Engine page. A rate the server did not send is left out. */
export function engineSummary(live: RuntimeLive, process: RuntimeProcessLive | null): string {
  const head = `Inference ${STATUS_WORD[live.status]}.`;
  if (process === null) return head;
  const wired = process.mem.wiredLimit;
  const memory = wired > 0 ? formatPercent(process.mem.active / wired) : 'unknown';
  const output = rateText(process.rates.outputTps);
  const prefill = rateText(process.rates.prefillTps);
  return [
    head,
    laneSentence(engineState(process)),
    output === null ? null : `Output ${output} tokens per second.`,
    prefill === null ? null : `Prefill ${prefill} tokens per second.`,
    `Memory ${memory} of the limit.`,
  ]
    .filter((part): part is string => part !== null)
    .join(' ');
}

/** `start`, `continue`, `after tool`. */
export function dispatchWord(dispatch: RuntimeProcessLive['turns'][number]['dispatch']): string {
  return dispatch === 'continue_tool' ? 'after tool' : dispatch;
}
