/** Time by phase and output rate of the Engine page's History section, from `/api/runtime/series`. React-free. */

import type { ChartPoint } from '../components/arc-x/chart/types';
import { runtimeBucketPoint } from './engine-view';
import type { RuntimePhaseMs, RuntimeSeries } from './types';

/** The four bands of a time-by-phase chart: other = preparing + running + loading. */
export interface PhaseLanes {
  decode: number;
  prefill: number;
  other: number;
  waiting: number;
}

/** `StackedBarChart` draws at most this many columns; longer ranges merge neighbouring buckets. */
export const PHASE_MAX_COLUMNS = 90;

const HOUR_MS = 3_600_000;
const MINUTE_MS = 60_000;

function bandsOf(ms: RuntimePhaseMs): PhaseLanes {
  return { decode: ms.decode, prefill: ms.prefill, other: ms.prepare + ms.running + ms.loading, waiting: ms.waiting };
}

function addBands(sum: PhaseLanes, ms: RuntimePhaseMs): void {
  const bands = bandsOf(ms);
  sum.decode += bands.decode;
  sum.prefill += bands.prefill;
  sum.other += bands.other;
  sum.waiting += bands.waiting;
}

function zeroBands(): PhaseLanes {
  return { decode: 0, prefill: 0, other: 0, waiting: 0 };
}

function totalOf(bands: PhaseLanes): number {
  return bands.decode + bands.prefill + bands.other + bands.waiting;
}

/** True when some bucket of the range holds phase time; false on a range of records that predate it, or all idle. */
export function hasPhaseTime(series: RuntimeSeries): boolean {
  return series.points.some((point) => totalOf(bandsOf(point.phaseMs)) > 0);
}

/**
 * Lane-ms of each band ÷ `observedMs` (the Busy time tile's covered time): 0.69 decode = 69 % of the observed time.
 * Lanes add up, so two lanes decoding the whole time give 2. Null when nothing is observed or no phase time exists.
 */
export function phaseShare(series: RuntimeSeries, observedMs: number): PhaseLanes | null {
  if (!(observedMs > 0)) return null;
  const sum = zeroBands();
  for (const point of series.points) addBands(sum, point.phaseMs);
  if (totalOf(sum) <= 0) return null;
  return {
    decode: sum.decode / observedMs,
    prefill: sum.prefill / observedMs,
    other: sum.other / observedMs,
    waiting: sum.waiting / observedMs,
  };
}

/** `minute`, `30 minutes`, `hour`, `2 hours`. */
export function columnWord(ms: number): string {
  if (ms >= HOUR_MS) {
    const hours = Math.round(ms / HOUR_MS);
    return hours <= 1 ? 'hour' : `${String(hours)} hours`;
  }
  const minutes = Math.round(ms / MINUTE_MS);
  return minutes <= 1 ? 'minute' : `${String(minutes)} minutes`;
}

/**
 * Mean lanes per band per column: Σ lane-ms ÷ the column's span. A column is one bucket, or `per` neighbouring
 * buckets when the range has more than `PHASE_MAX_COLUMNS`. A column with no sample at all is null (a gap); one
 * whose samples carry no phase time is 0 (idle).
 */
export function phasePoints(
  series: RuntimeSeries,
  starts: readonly number[],
): { data: ChartPoint[]; columnMs: number } {
  const per = Math.max(1, Math.ceil(starts.length / PHASE_MAX_COLUMNS));
  const columnMs = series.bucketMs * per;
  const byStart = new Map(series.points.map((point) => [point.t, point]));
  const data: ChartPoint[] = [];
  for (let i = 0; i < starts.length; i += per) {
    const members = starts.slice(i, i + per);
    const sum = zeroBands();
    let sampled = false;
    for (const t of members) {
      const point = byStart.get(t);
      if (point === undefined || point.samples <= 0) continue;
      sampled = true;
      addBands(sum, point.phaseMs);
    }
    const spanMs = members.length * series.bucketMs;
    data.push({
      ...runtimeBucketPoint(members[0], columnMs),
      values: sampled
        ? {
            decode: sum.decode / spanMs,
            prefill: sum.prefill / spanMs,
            other: sum.other / spanMs,
            waiting: sum.waiting / spanMs,
          }
        : { decode: null, prefill: null, other: null, waiting: null },
    });
  }
  return { data, columnMs };
}

/** Output tokens per second of decode time (all lanes, union) per bucket; null where the bucket has no decode. */
export function outputRatePoints(series: RuntimeSeries, starts: readonly number[]): ChartPoint[] {
  const byStart = new Map(series.points.map((point) => [point.t, point]));
  return starts.map((t): ChartPoint => {
    const point = byStart.get(t);
    const rated = point !== undefined && point.rateOutMs > 0 && point.rateOutTokens > 0;
    return {
      ...runtimeBucketPoint(t, series.bucketMs),
      values: { rate: rated ? (point.rateOutTokens * 1000) / point.rateOutMs : null },
    };
  });
}
