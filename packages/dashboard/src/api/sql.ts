/**
 * Shared SQL fragments and query-parameter parsing for the metrics and runtime
 * routes.
 *
 * `node:sqlite` binds every JS number as REAL, so a bound number is only ever
 * compared (`ts >= ?`). Arithmetic uses literals or columns: `(ts / ?) * ?`
 * would be a floating-point division and give `ts` back unbucketed.
 */

import { PREFILL_MIN_TOKENS } from '../prefill-rule.js';
import { queryInt } from './context.js';
import { ApiError } from './errors.js';

export type MetricsBucket = '5m' | 'hour' | 'day';
export type MetricsGroupBy = 'model' | 'none';
export type TurnOutcome = 'ok' | 'error' | 'cancelled';

export const BUCKET_MS: Record<MetricsBucket, number> = { '5m': 300_000, hour: 3_600_000, day: 86_400_000 };

/** Most buckets one request may span. */
export const MAX_BUCKETS = 2000;

/** The only `bucketMs` values of `/api/runtime/series`. */
export const RUNTIME_BUCKET_MS: readonly number[] = [5000, 15_000, 60_000, 300_000, 900_000, 3_600_000];

/** Bucket start in epoch ms; hour and day follow local time (DST included). */
export function bucketSql(bucket: MetricsBucket, column = 'ts'): string {
  switch (bucket) {
    case '5m':
      return `((${column} / 300000) * 300000)`;
    case 'hour':
      return `(CAST(strftime('%s', strftime('%Y-%m-%d %H:00:00', ${column} / 1000, 'unixepoch', 'localtime'), 'utc') AS INTEGER) * 1000)`;
    case 'day':
      return `(CAST(strftime('%s', strftime('%Y-%m-%d 00:00:00', ${column} / 1000, 'unixepoch', 'localtime'), 'utc') AS INTEGER) * 1000)`;
  }
}

/** `(ts / N) * N` with `N` inlined as an integer literal. Throws 400 for any value outside {@link RUNTIME_BUCKET_MS}. */
export function runtimeBucketSql(bucketMs: number, column = 'ts'): string {
  if (!RUNTIME_BUCKET_MS.includes(bucketMs)) throw ApiError.badRequest(`Unsupported bucketMs ${bucketMs}`);
  const literal = String(Math.trunc(bucketMs));
  return `((${column} / ${literal}) * ${literal})`;
}

/** Decode wall ms, derived exactly the way native computes `decodeTps`. */
export const DECODE_WALL_SQL =
  'CASE WHEN decode_tps > 0 AND output_tokens > 1 THEN (output_tokens - 1) * 1000.0 / decode_tps END';
/** Handler start to the end of the dispatch. */
export const TOTAL_WALL_SQL = 'COALESCE(pre_dispatch_ms, 0) + duration_ms';
/** Percent of the context window in use at turn end. */
export const CTX_FILL_SQL =
  'CASE WHEN ctx_window > 0 THEN 100.0 * (COALESCE(prompt_tokens, 0) + COALESCE(output_tokens, 0)) / ctx_window END';
/** Prompt tokens net of the cached prefix. */
export const NET_INPUT_SQL = 'MAX(COALESCE(prompt_tokens, 0) - COALESCE(cached_tokens, 0), 0)';
/** Native reports 0.0 for a rate it could not measure; v1 rows still hold those zeros. */
export const DECODE_TPS_SQL = 'CASE WHEN decode_tps > 0 AND output_tokens > 1 THEN decode_tps END';

const MEASURED = 'prefill_ms > 0 AND prefill_tokens > 0';
/** The row carries a measured prefill window. */
export const PREFILL_MEASURED_SQL = MEASURED;
/** Prefill tokens ÷ prefill window, rated rows only. */
export const PREFILL_TPS_SQL = `CASE WHEN prefill_ms > 0 AND prefill_tokens >= ${String(PREFILL_MIN_TOKENS)} THEN prefill_tokens * 1000.0 / prefill_ms END`;
/** Time to first token outside the window. */
export const SETUP_MS_SQL = `CASE WHEN ${MEASURED} AND ttft_ms IS NOT NULL THEN MAX(ttft_ms - prefill_ms, 0) END`;

const nonNegative = (sql: string): string => `MAX(COALESCE(${sql}, 0), 0)`;
const PRE = nonNegative('pre_dispatch_ms');
const DUR = nonNegative('duration_ms');
const QUEUE = `MIN(${nonNegative('queue_ms')}, ${PRE})`;
const LOAD_WAIT = `MIN(${nonNegative('load_wait_ms')}, ${PRE} - ${QUEUE})`;
const RESOLVE = `MIN(${nonNegative('resolve_ms')}, ${PRE} - ${QUEUE} - ${LOAD_WAIT})`;
const TTFT = nonNegative('ttft_ms');
const LEAD = `MIN(CASE WHEN ${TTFT} > 0 THEN ${TTFT} WHEN ${MEASURED} THEN ${nonNegative('prefill_ms')} ELSE 0 END, ${DUR})`;
const PREFILL = `CASE WHEN ${MEASURED} THEN MIN(${nonNegative('prefill_ms')}, ${LEAD}) ELSE 0 END`;

/**
 * One row's phases bounded as `turnPhases` bounds them: admission phases within the pre-dispatch time, the rest
 * within the duration. Their sum never passes {@link PHASE_TOTAL_SQL}, so a summed bar never passes 100 %.
 */
export const PHASE_SQL = {
  queue: QUEUE,
  loadWait: LOAD_WAIT,
  resolve: RESOLVE,
  firstToken: `CASE WHEN ${MEASURED} THEN 0 ELSE ${LEAD} END`,
  setup: `CASE WHEN ${MEASURED} THEN ${LEAD} - ${PREFILL} ELSE 0 END`,
  prefill: PREFILL,
  decode: `MIN(${nonNegative(DECODE_WALL_SQL)}, ${DUR} - ${LEAD})`,
} as const;
/** Pre-dispatch time plus duration, each read as 0 when missing or negative. */
export const PHASE_TOTAL_SQL = `${PRE} + ${DUR}`;

export interface MetricsRange {
  from: number | null;
  to: number | null;
}

/** `from` / `to` (epoch ms, both optional). `from > to` is a 400. */
export function parseRange(query: URLSearchParams): MetricsRange {
  const from = parseNumberParam(query, 'from');
  const to = parseNumberParam(query, 'to');
  if (from !== null && to !== null && from > to) throw ApiError.badRequest('from must not be after to');
  return { from, to };
}

export function parseNumberParam(query: URLSearchParams, name: string): number | null {
  const raw = query.get(name);
  if (raw !== null && raw !== '' && queryInt(query, name) === null) {
    throw ApiError.badRequest(`${name} must be a number`);
  }
  return queryInt(query, name);
}

/** Exact `model_key`, or null for all models. */
export function parseModel(query: URLSearchParams): string | null {
  const raw = query.get('model');
  return raw === null || raw === '' ? null : raw;
}

export function parseBucket(raw: string | null): MetricsBucket {
  if (raw === '5m' || raw === 'hour' || raw === 'day') return raw;
  throw ApiError.badRequest('bucket must be one of 5m, hour, day');
}

export function parseGroupBy(raw: string | null): MetricsGroupBy {
  if (raw === null || raw === '' || raw === 'model') return 'model';
  if (raw === 'none') return 'none';
  throw ApiError.badRequest('groupBy must be model or none');
}

export function parseOutcome(raw: string | null): TurnOutcome | null {
  if (raw === null || raw === '') return null;
  if (raw === 'ok' || raw === 'error' || raw === 'cancelled') return raw;
  throw ApiError.badRequest('outcome must be one of ok, error, cancelled');
}

/** 400 when the range would hold more than {@link MAX_BUCKETS} buckets. A `5m` bucket needs both bounds. */
export function assertBucketCount(range: MetricsRange, bucketMs: number, requireBounds: boolean): void {
  if (range.from === null || range.to === null) {
    if (requireBounds) throw ApiError.badRequest('from and to are both required for this bucket');
    return;
  }
  if ((range.to - range.from) / bucketMs > MAX_BUCKETS) {
    throw ApiError.badRequest(`The range spans more than ${MAX_BUCKETS} buckets`);
  }
}

/** A `WHERE` clause over `ts` (and `model_key` when given), with its arguments. */
export function traceFilter(
  range: MetricsRange,
  model: string | null,
  extra: string[] = [],
): { where: string; args: Array<number | string> } {
  const parts = [...extra];
  const args: Array<number | string> = [];
  if (range.from !== null) {
    parts.push('ts >= ?');
    args.push(range.from);
  }
  if (range.to !== null) {
    parts.push('ts <= ?');
    args.push(range.to);
  }
  if (model !== null) {
    parts.push('model_key = ?');
    args.push(model);
  }
  return { where: parts.length > 0 ? `WHERE ${parts.join(' AND ')}` : '', args };
}
