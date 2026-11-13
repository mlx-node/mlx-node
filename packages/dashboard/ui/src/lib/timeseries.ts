/** Pure pivots from `/api/metrics/timeseries` rows to chart points: one point per bucket start, gaps filled by rule. */

import type { ChartPoint } from '../components/arc-x/chart/types';
import { formatBucketLabel, formatCountNoun, formatDay, formatDayLong } from './format';
import type { AlignedRange } from './range';
import type { MetricsBucket, MetricsSummary, MetricsTimeseries } from './types';

type TsPoint = MetricsTimeseries['points'][number];

export const OTHER_KEY = 'other';

/** The legend label of the folded series: `Other (1 model)`, `Other (3 models)`. */
export function otherLabel(folded: number): string {
  return `Other (${formatCountNoun(folded, 'model', 'models')})`;
}

/** The footnote of a chart that draws only its top models: `+1 model in the table`. */
export function moreInTableLabel(extra: number): string {
  return `+${formatCountNoun(extra, 'model', 'models')} in the table`;
}

export interface PivotOptions {
  /** Value for a bucket with no row. 0 for sums and counts (no row = nothing happened), null for rates and quantiles. */
  empty: 0 | null;
  /** Sample count behind each value (ChartPoint.samples). */
  samples?: (p: TsPoint) => number;
  /** Models to draw, in draw order. Others are folded per `fold`. */
  models: readonly string[];
  /** 'sum' -> one extra key 'other'; 'drop' -> left out, counted in `dropped`. */
  fold: 'sum' | 'drop';
}

const FIVE_MINUTES_MS = 300_000;
const HOUR_MS = 3_600_000;

function bucketEnd(t: number, bucket: MetricsBucket): number {
  if (bucket === '5m') return t + FIVE_MINUTES_MS;
  if (bucket === 'hour') return t + HOUR_MS;
  const day = new Date(t);
  return new Date(day.getFullYear(), day.getMonth(), day.getDate() + 1).getTime();
}

/** Label helpers: key = String(t); label via formatBucketLabel + ', so far' on the bucket containing now. */
export function bucketPoint(
  t: number,
  bucket: MetricsBucket,
  now: number,
): Pick<ChartPoint, 'key' | 'label' | 'axisLabel'> {
  const base = bucket === 'day' ? formatDayLong(t) : formatBucketLabel(t, bucket);
  const current = t <= now && now < bucketEnd(t, bucket);
  return {
    key: String(t),
    label: current ? `${base}, so far` : base,
    axisLabel: bucket === 'day' ? formatDay(t) : base,
  };
}

/** Index of the bucket of `range` that holds `t`, or -1 outside the window. */
function bucketIndex(starts: readonly number[], to: number, t: number): number {
  if (starts.length === 0 || t < starts[0] || t >= to) return -1;
  let low = 0;
  let high = starts.length - 1;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (starts[mid] <= t) low = mid;
    else high = mid - 1;
  }
  return low;
}

/** The last bucket of an aligned range is the one that held `now` when the range was built. */
function rangePoint(range: AlignedRange, index: number): Pick<ChartPoint, 'key' | 'label' | 'axisLabel'> {
  const t = range.starts[index];
  const last = index === range.starts.length - 1;
  return bucketPoint(t, range.bucket, last ? t : t - 1);
}

function rowsByBucket(points: readonly TsPoint[], range: AlignedRange): TsPoint[][] {
  const rows: TsPoint[][] = range.starts.map(() => []);
  for (const point of points) {
    const index = bucketIndex(range.starts, range.to, point.t);
    if (index !== -1) rows[index].push(point);
  }
  return rows;
}

function sumOrNull(values: readonly (number | null)[]): number | null {
  let total: number | null = null;
  for (const value of values) {
    if (value === null || !Number.isFinite(value)) continue;
    total = (total ?? 0) + value;
  }
  return total;
}

/**
 * One ChartPoint per bucket start; keys = model names (+ 'other'). `dropped` names the models not drawn on their own:
 * left out with `fold: 'drop'`, summed into 'other' with `fold: 'sum'`. 'other' exists only when such a model exists.
 */
export function pivotByModel(
  points: readonly TsPoint[],
  range: AlignedRange,
  pick: (p: TsPoint) => number | null,
  options: PivotOptions,
): { data: ChartPoint[]; dropped: string[] } {
  const drawn = new Set(options.models);
  const others = new Set<string>();
  for (const point of points) if (!drawn.has(point.model)) others.add(point.model);
  const foldOther = options.fold === 'sum' && others.size > 0;
  const data = rowsByBucket(points, range).map((rows, index): ChartPoint => {
    const values: Record<string, number | null> = {};
    const samples: Record<string, number> = {};
    for (const model of options.models) {
      const row = rows.find((item) => item.model === model);
      values[model] = row === undefined ? options.empty : pick(row);
      if (options.samples !== undefined) samples[model] = row === undefined ? 0 : options.samples(row);
    }
    if (foldOther) {
      const rest = rows.filter((row) => !drawn.has(row.model));
      values[OTHER_KEY] = rest.length === 0 ? options.empty : sumOrNull(rest.map(pick));
      if (options.samples !== undefined) {
        const count = options.samples;
        samples[OTHER_KEY] = rest.reduce((sum, row) => sum + count(row), 0);
      }
    }
    return {
      ...rangePoint(range, index),
      values,
      ...(options.samples !== undefined ? { samples } : {}),
    };
  });
  return { data, dropped: [...others].sort() };
}

/** For groupBy=none (or one model): several measures of the same row as series. */
export function pivotMeasures(
  points: readonly TsPoint[],
  range: AlignedRange,
  measures: Readonly<Record<string, (p: TsPoint) => number | null>>,
  empty: 0 | null,
): ChartPoint[] {
  const keys = Object.keys(measures);
  return rowsByBucket(points, range).map((rows, index): ChartPoint => {
    const row = rows[0];
    const values: Record<string, number | null> = {};
    for (const key of keys) values[key] = row === undefined ? empty : measures[key](row);
    return { ...rangePoint(range, index), values };
  });
}

/** Top n model names by turns (ties: name). Input: summary.byModel. */
export function topModels(byModel: MetricsSummary['byModel'], n = 8): string[] {
  return [...byModel]
    .filter((row) => row.totals.turns > 0)
    .sort((a, b) => b.totals.turns - a.totals.turns || (a.model < b.model ? -1 : a.model > b.model ? 1 : 0))
    .slice(0, n)
    .map((row) => row.model);
}
