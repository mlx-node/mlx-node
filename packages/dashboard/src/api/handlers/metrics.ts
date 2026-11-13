/**
 * Per-turn metrics routes: models, summary, timeseries, distributions,
 * speculative, turns.
 *
 * Rules every handler keeps: a missing value is null, never 0 (token and count
 * sums are 0 when empty); throughput and latency read `outcome = 'ok'` rows
 * only, token sums and turn counts read every outcome; empty buckets are never
 * filled; percentiles are nearest rank over one scan per window. Prefill
 * speed reads measured rows only; a row without a prefill window has none.
 */

import type { StatementSync } from 'node:sqlite';

import type { ApiRequest, WorkerApiContext } from '../context.js';
import { toInt, toNum } from '../context.js';
import { ApiError } from '../errors.js';
import { histogramOf, quantilesOf, ValueScan, type Histogram, type Quantiles } from '../quantiles.js';
import {
  assertBucketCount,
  BUCKET_MS,
  bucketSql,
  CTX_FILL_SQL,
  DECODE_TPS_SQL,
  DECODE_WALL_SQL,
  NET_INPUT_SQL,
  parseBucket,
  parseGroupBy,
  parseModel,
  parseNumberParam,
  parseOutcome,
  parseRange,
  PHASE_SQL,
  PHASE_TOTAL_SQL,
  PREFILL_MEASURED_SQL,
  PREFILL_TPS_SQL,
  SETUP_MS_SQL,
  TOTAL_WALL_SQL,
  traceFilter,
  type MetricsBucket,
  type MetricsGroupBy,
  type MetricsRange,
  type TurnOutcome,
} from '../sql.js';

export type { Histogram, Quantiles } from '../quantiles.js';
export type { MetricsBucket, MetricsRange, TurnOutcome } from '../sql.js';

export interface MetricsModels {
  models: Array<{ model: string; modelType: string | null; firstTs: number; lastTs: number; turns: number }>;
}

export interface MetricsTotals {
  turns: number;
  ok: number;
  error: number;
  cancelled: number;
  /** Ok turns that carry a measured prefill window. */
  prefillMeasured: number;
  inputTokens: number;
  cachedTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  queueWallMs: number;
  loadWaitWallMs: number;
  resolveWallMs: number;
  /** Time to first token of turns without a prefill window: setup and prefill together. */
  firstTokenWallMs: number;
  /** Time to first token outside the prefill window, measured turns only. */
  setupWallMs: number;
  /** Sum of measured prefill windows, each capped at its time to first token. */
  prefillWallMs: number;
  decodeWallMs: number;
  /** total − queue − loadWait − resolve − firstToken − setup − prefill − decode, clamped at 0. */
  otherWallMs: number;
  totalWallMs: number;
}

export interface MetricsLatency {
  ttftMs: Quantiles;
  setupMs: Quantiles;
  decodeTps: Quantiles;
  prefillTps: Quantiles;
  totalMs: Quantiles;
  queueMs: Quantiles;
  ctxFillPct: Quantiles;
}

export interface MetricsSummary {
  range: MetricsRange;
  model: string | null;
  totals: MetricsTotals;
  latency: MetricsLatency;
  previous: { totals: MetricsTotals; latency: MetricsLatency } | null;
  byModel: Array<{
    model: string;
    modelType: string | null;
    ctxWindow: number | null;
    totals: MetricsTotals;
    latency: MetricsLatency;
    mtpTurns: number;
    lastTs: number;
  }>;
  finishReasons: Array<{ reason: string; turns: number }>;
  failures: Array<{
    outcome: 'error' | 'cancelled';
    errorKind: 'finish_error' | 'exception' | null;
    failPhase: 'queued' | 'prefill' | 'decode' | null;
    turns: number;
  }>;
  endpoints: Array<{ endpoint: string; stream: boolean; turns: number }>;
  dispatches: Array<{ dispatch: string; turns: number; promptTokens: number; cachedTokens: number }>;
}

export interface MetricsTimeseries {
  range: MetricsRange;
  bucket: MetricsBucket;
  groupBy: MetricsGroupBy;
  points: Array<{
    t: number;
    model: string;
    totals: MetricsTotals;
    ttftMs: Quantiles;
    decodeTps: Quantiles;
    prefillTps: Quantiles;
    totalMs: Quantiles;
    mtpMeanAccepted: number | null;
    mtpTurns: number;
    coldHits: number;
    coldMisses: number;
    concurrencyPeak: number | null;
  }>;
}

type DistributionMetric =
  | 'ttftMs'
  | 'decodeTps'
  | 'prefillTps'
  | 'totalMs'
  | 'ctxFillPct'
  | 'promptTokens'
  | 'outputTokens';

export interface MetricsDistributions {
  range: MetricsRange;
  groupBy: MetricsGroupBy;
  edges: Record<DistributionMetric, number[]>;
  series: Array<{ model: string } & Record<DistributionMetric, Histogram>>;
}

export interface MetricsSpeculative {
  range: MetricsRange;
  models: Array<{
    model: string;
    turns: number;
    mtpTurns: number;
    cycles: number;
    meanAccepted: number | null;
    meanDrafts: number | null;
    meanDepth: number | null;
    acceptByPos: Array<{ pos: number; rate: number; cycles: number }>;
    decodeTpsWithMtp: Quantiles;
    decodeTpsWithoutMtp: Quantiles;
  }>;
  trend: Array<{ t: number; model: string; meanAccepted: number | null; mtpTurns: number }>;
}

export interface TurnRow {
  traceId: string;
  ts: number;
  model: string;
  alias: string | null;
  modelType: string | null;
  outcome: TurnOutcome;
  errorKind: string | null;
  failPhase: string | null;
  finishReason: string | null;
  endpoint: string | null;
  stream: boolean | null;
  dispatch: string | null;
  promptTokens: number | null;
  cachedTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  ctxWindow: number | null;
  ctxFillPct: number | null;
  preDispatchMs: number | null;
  queueMs: number | null;
  loadWaitMs: number | null;
  resolveMs: number | null;
  ttftMs: number | null;
  /** Measured prefill window; null when the turn has none. */
  prefillMs: number | null;
  prefillTokens: number | null;
  setupMs: number | null;
  decodeMs: number | null;
  durationMs: number | null;
  /** Rated turns only (measured, at least `PREFILL_MIN_TOKENS`). */
  prefillTps: number | null;
  decodeTps: number | null;
  mtpCycles: number | null;
  mtpMeanAccepted: number | null;
  coldHits: number | null;
  coldMisses: number | null;
  toolCount: number | null;
  toolCallCount: number | null;
  imageCount: number | null;
  concurrency: number | null;
  concurrencyPeak: number | null;
  memActiveBytes: number | null;
  pid: number | null;
}

export interface MetricsTurns {
  range: MetricsRange;
  rows: TurnRow[];
  next: { beforeTs: number; beforeTraceId: string } | null;
  total: number;
}

/** Fixed bin lower bounds: two models and two ranges stay comparable bin for bin. */
export const HISTOGRAM_EDGES: Record<DistributionMetric, number[]> = {
  ttftMs: [0, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000, 8000, 12000, 20000, 30000, 60000],
  decodeTps: [0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 300],
  prefillTps: [0, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000],
  totalMs: [0, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 30000, 60000, 120000, 300000],
  ctxFillPct: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
  promptTokens: [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
  outputTokens: [0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
};

const TURNS_LIMIT_DEFAULT = 200;
const TURNS_LIMIT_MAX = 2000;
const ALL_MODELS = '*';

type Row = Record<string, unknown>;
type Args = Array<number | string>;

function run(statement: StatementSync, args: Args): Row[] {
  return statement.all(...args) as Row[];
}

function strOrNull(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

/** Aggregate columns of {@link MetricsTotals}; every outcome counts. */
const TOTALS_SELECT = `
  COUNT(*) AS turns,
  SUM(outcome = 'ok') AS ok, SUM(outcome = 'error') AS error, SUM(outcome = 'cancelled') AS cancelled,
  SUM(CASE WHEN outcome = 'ok' AND ${PREFILL_MEASURED_SQL} THEN 1 ELSE 0 END) AS prefillMeasured,
  SUM(${NET_INPUT_SQL}) AS inputTokens, SUM(COALESCE(cached_tokens, 0)) AS cachedTokens,
  SUM(COALESCE(output_tokens, 0)) AS outputTokens, SUM(COALESCE(reasoning_tokens, 0)) AS reasoningTokens,
  SUM(${PHASE_SQL.queue}) AS queueWallMs, SUM(${PHASE_SQL.loadWait}) AS loadWaitWallMs,
  SUM(${PHASE_SQL.resolve}) AS resolveWallMs,
  SUM(${PHASE_SQL.firstToken}) AS firstTokenWallMs, SUM(${PHASE_SQL.setup}) AS setupWallMs,
  SUM(${PHASE_SQL.prefill}) AS prefillWallMs, SUM(${PHASE_SQL.decode}) AS decodeWallMs,
  SUM(${PHASE_TOTAL_SQL}) AS totalWallMs`;

function emptyTotals(): MetricsTotals {
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
  };
}

function withOther(t: MetricsTotals): MetricsTotals {
  const parts =
    t.queueWallMs +
    t.loadWaitWallMs +
    t.resolveWallMs +
    t.firstTokenWallMs +
    t.setupWallMs +
    t.prefillWallMs +
    t.decodeWallMs;
  return { ...t, otherWallMs: Math.max(t.totalWallMs - parts, 0) };
}

function totalsOf(row: Row): MetricsTotals {
  const ms = (value: unknown): number => toNum(value) ?? 0;
  return withOther({
    turns: toInt(row.turns),
    ok: toInt(row.ok),
    error: toInt(row.error),
    cancelled: toInt(row.cancelled),
    prefillMeasured: toInt(row.prefillMeasured),
    inputTokens: toInt(row.inputTokens),
    cachedTokens: toInt(row.cachedTokens),
    outputTokens: toInt(row.outputTokens),
    reasoningTokens: toInt(row.reasoningTokens),
    queueWallMs: ms(row.queueWallMs),
    loadWaitWallMs: ms(row.loadWaitWallMs),
    resolveWallMs: ms(row.resolveWallMs),
    firstTokenWallMs: ms(row.firstTokenWallMs),
    setupWallMs: ms(row.setupWallMs),
    prefillWallMs: ms(row.prefillWallMs),
    decodeWallMs: ms(row.decodeWallMs),
    otherWallMs: 0,
    totalWallMs: ms(row.totalWallMs),
  });
}

function addTotals(a: MetricsTotals, b: MetricsTotals): MetricsTotals {
  const sum = emptyTotals();
  for (const key of Object.keys(sum) as Array<keyof MetricsTotals>) sum[key] = a[key] + b[key];
  return withOther(sum);
}

/** The value columns of one `outcome = 'ok'` scan. */
const SCAN_COLUMNS = `ttft_ms AS ttftMs, ${SETUP_MS_SQL} AS setupMs, ${DECODE_TPS_SQL} AS decodeTps,
  ${PREFILL_TPS_SQL} AS prefillTps,
  ${TOTAL_WALL_SQL} AS totalMs, queue_ms AS queueMs, ${CTX_FILL_SQL} AS ctxFillPct,
  prompt_tokens AS promptTokens, output_tokens AS outputTokens`;

const SCAN_KEYS = [
  'ttftMs',
  'setupMs',
  'decodeTps',
  'prefillTps',
  'totalMs',
  'queueMs',
  'ctxFillPct',
  'promptTokens',
  'outputTokens',
] as const;

function scanInto(scan: ValueScan, group: string, row: Row): void {
  for (const key of SCAN_KEYS) scan.add(group, key, toNum(row[key]));
}

function latencyOf(values: (column: string) => number[]): MetricsLatency {
  return {
    ttftMs: quantilesOf(values('ttftMs')),
    setupMs: quantilesOf(values('setupMs')),
    decodeTps: quantilesOf(values('decodeTps')),
    prefillTps: quantilesOf(values('prefillTps')),
    totalMs: quantilesOf(values('totalMs')),
    queueMs: quantilesOf(values('queueMs')),
    ctxFillPct: quantilesOf(values('ctxFillPct')),
  };
}

interface WindowStats {
  totals: MetricsTotals;
  latency: MetricsLatency;
  models: Array<{ row: Row; totals: MetricsTotals }>;
  scan: ValueScan;
}

function windowStats(ctx: WorkerApiContext, range: MetricsRange, model: string | null): WindowStats {
  const { sqlite } = ctx.dash;
  const all = traceFilter(range, model);
  const models = run(
    sqlite.prepare(
      `SELECT model_key AS model, MAX(model_type) AS modelType, MAX(ctx_window) AS ctxWindow,
              COUNT(mtp_cycles) AS mtpTurns, MAX(ts) AS lastTs, ${TOTALS_SELECT}
       FROM traces ${all.where} GROUP BY model_key ORDER BY turns DESC, model_key`,
    ),
    all.args,
  ).map((row) => ({ row, totals: totalsOf(row) }));
  let totals = emptyTotals();
  for (const m of models) totals = addTotals(totals, m.totals);

  const ok = traceFilter(range, model, ["outcome = 'ok'"]);
  const scan = new ValueScan();
  const statement = sqlite.prepare(`SELECT model_key AS model, ${SCAN_COLUMNS} FROM traces ${ok.where}`);
  for (const row of statement.iterate(...ok.args) as Iterable<Row>) scanInto(scan, String(row.model), row);
  return { totals, latency: latencyOf((c) => scan.all(c)), models, scan };
}

export function handleMetricsModels(ctx: WorkerApiContext): MetricsModels {
  const rows = ctx.dash.sqlite
    .prepare(
      `SELECT model_key AS model, MAX(model_type) AS modelType, MIN(ts) AS firstTs, MAX(ts) AS lastTs,
              COUNT(*) AS turns
       FROM traces GROUP BY model_key ORDER BY firstTs, model_key`,
    )
    .all() as Row[];
  return {
    models: rows.map((row) => ({
      model: String(row.model),
      modelType: strOrNull(row.modelType),
      firstTs: toInt(row.firstTs),
      lastTs: toInt(row.lastTs),
      turns: toInt(row.turns),
    })),
  };
}

export function handleMetricsSummary(ctx: WorkerApiContext, req: ApiRequest): MetricsSummary {
  const range = parseRange(req.query);
  const model = parseModel(req.query);
  const { sqlite } = ctx.dash;
  const current = windowStats(ctx, range, model);

  let previous: MetricsSummary['previous'] = null;
  if (range.from !== null && range.to !== null) {
    const prevTo = range.from - 1;
    const prev = windowStats(ctx, { from: prevTo - (range.to - range.from), to: prevTo }, model);
    previous = { totals: prev.totals, latency: prev.latency };
  }

  const all = traceFilter(range, model);
  const failed = traceFilter(range, model, ["outcome <> 'ok'"]);
  return {
    range,
    model,
    totals: current.totals,
    latency: current.latency,
    previous,
    byModel: current.models.map(({ row, totals }) => {
      const key = String(row.model);
      return {
        model: key,
        modelType: strOrNull(row.modelType),
        ctxWindow: toNum(row.ctxWindow),
        totals,
        latency: latencyOf((c) => current.scan.values(key, c)),
        mtpTurns: toInt(row.mtpTurns),
        lastTs: toInt(row.lastTs),
      };
    }),
    finishReasons: run(
      sqlite.prepare(
        `SELECT COALESCE(finish_reason, 'unknown') AS reason, COUNT(*) AS turns
         FROM traces ${all.where} GROUP BY reason ORDER BY turns DESC, reason`,
      ),
      all.args,
    ).map((row) => ({ reason: String(row.reason), turns: toInt(row.turns) })),
    failures: run(
      sqlite.prepare(
        `SELECT outcome, error_kind AS errorKind, fail_phase AS failPhase, COUNT(*) AS turns
         FROM traces ${failed.where} GROUP BY outcome, error_kind, fail_phase ORDER BY turns DESC`,
      ),
      failed.args,
    ).map((row) => ({
      outcome: row.outcome === 'cancelled' ? 'cancelled' : 'error',
      errorKind: row.errorKind === 'finish_error' || row.errorKind === 'exception' ? row.errorKind : null,
      failPhase:
        row.failPhase === 'queued' || row.failPhase === 'prefill' || row.failPhase === 'decode' ? row.failPhase : null,
      turns: toInt(row.turns),
    })),
    endpoints: run(
      sqlite.prepare(
        `SELECT COALESCE(endpoint, 'unknown') AS endpoint, COALESCE(stream, 0) AS stream, COUNT(*) AS turns
         FROM traces ${all.where} GROUP BY 1, 2 ORDER BY turns DESC`,
      ),
      all.args,
    ).map((row) => ({ endpoint: String(row.endpoint), stream: toInt(row.stream) === 1, turns: toInt(row.turns) })),
    dispatches: run(
      sqlite.prepare(
        `SELECT COALESCE(dispatch, 'unknown') AS dispatch, COUNT(*) AS turns,
                SUM(COALESCE(prompt_tokens, 0)) AS promptTokens, SUM(COALESCE(cached_tokens, 0)) AS cachedTokens
         FROM traces ${all.where} GROUP BY 1 ORDER BY turns DESC`,
      ),
      all.args,
    ).map((row) => ({
      dispatch: String(row.dispatch),
      turns: toInt(row.turns),
      promptTokens: toInt(row.promptTokens),
      cachedTokens: toInt(row.cachedTokens),
    })),
  };
}

/** Group key of a bucketed scan row. */
function pointKey(t: number, model: string): string {
  return `${t}\u0000${model}`;
}

/** Committed tokens per cycle, weighted by cycles; only cycles whose mean is known count. */
const MTP_WEIGHTED_SELECT = `
  SUM(CASE WHEN mtp_mean_accepted IS NOT NULL THEN mtp_mean_accepted * mtp_cycles END) AS mtpAcceptedCycles,
  SUM(CASE WHEN mtp_mean_accepted IS NOT NULL THEN mtp_cycles END) AS mtpCycles,
  COUNT(mtp_cycles) AS mtpTurns`;

function weighted(sum: unknown, weight: unknown): number | null {
  const w = toNum(weight);
  const s = toNum(sum);
  return w !== null && w > 0 && s !== null ? s / w : null;
}

export function handleMetricsTimeseries(ctx: WorkerApiContext, req: ApiRequest): MetricsTimeseries {
  const range = parseRange(req.query);
  const model = parseModel(req.query);
  const bucket = parseBucket(req.query.get('bucket'));
  const groupBy = parseGroupBy(req.query.get('groupBy'));
  assertBucketCount(range, BUCKET_MS[bucket], bucket === '5m');
  const { sqlite } = ctx.dash;
  const t = bucketSql(bucket);
  const modelCol = groupBy === 'model' ? 'model_key' : `'${ALL_MODELS}'`;

  const all = traceFilter(range, model);
  const rows = run(
    sqlite.prepare(
      `SELECT ${t} AS t, ${modelCol} AS model, ${TOTALS_SELECT}, ${MTP_WEIGHTED_SELECT},
              SUM(COALESCE(cold_hits, 0)) AS coldHits, SUM(COALESCE(cold_misses, 0)) AS coldMisses,
              MAX(concurrency_peak) AS concurrencyPeak
       FROM traces ${all.where} GROUP BY 1, 2 ORDER BY 1, 2`,
    ),
    all.args,
  );

  const ok = traceFilter(range, model, ["outcome = 'ok'"]);
  const scan = new ValueScan();
  const statement = sqlite.prepare(`SELECT ${t} AS t, ${modelCol} AS model, ${SCAN_COLUMNS} FROM traces ${ok.where}`);
  for (const row of statement.iterate(...ok.args) as Iterable<Row>) {
    scanInto(scan, pointKey(toInt(row.t), String(row.model)), row);
  }

  return {
    range,
    bucket,
    groupBy,
    points: rows.map((row) => {
      const key = pointKey(toInt(row.t), String(row.model));
      return {
        t: toInt(row.t),
        model: String(row.model),
        totals: totalsOf(row),
        ttftMs: quantilesOf(scan.values(key, 'ttftMs')),
        decodeTps: quantilesOf(scan.values(key, 'decodeTps')),
        prefillTps: quantilesOf(scan.values(key, 'prefillTps')),
        totalMs: quantilesOf(scan.values(key, 'totalMs')),
        mtpMeanAccepted: weighted(row.mtpAcceptedCycles, row.mtpCycles),
        mtpTurns: toInt(row.mtpTurns),
        coldHits: toInt(row.coldHits),
        coldMisses: toInt(row.coldMisses),
        concurrencyPeak: toNum(row.concurrencyPeak),
      };
    }),
  };
}

export function handleMetricsDistributions(ctx: WorkerApiContext, req: ApiRequest): MetricsDistributions {
  const range = parseRange(req.query);
  const model = parseModel(req.query);
  const groupBy = parseGroupBy(req.query.get('groupBy'));
  const ok = traceFilter(range, model, ["outcome = 'ok'"]);
  const modelCol = groupBy === 'model' ? 'model_key' : `'${ALL_MODELS}'`;
  const scan = new ValueScan();
  const statement = ctx.dash.sqlite.prepare(
    `SELECT ${modelCol} AS model, ${SCAN_COLUMNS} FROM traces ${ok.where} ORDER BY model`,
  );
  for (const row of statement.iterate(...ok.args) as Iterable<Row>) scanInto(scan, String(row.model), row);
  const metrics = Object.keys(HISTOGRAM_EDGES) as DistributionMetric[];
  return {
    range,
    groupBy,
    edges: HISTOGRAM_EDGES,
    series: scan.groups().map((group) => {
      const series = { model: group } as { model: string } & Record<DistributionMetric, Histogram>;
      for (const metric of metrics) series[metric] = histogramOf(scan.values(group, metric), HISTOGRAM_EDGES[metric]);
      return series;
    }),
  };
}

export function handleMetricsSpeculative(ctx: WorkerApiContext, req: ApiRequest): MetricsSpeculative {
  const range = parseRange(req.query);
  const model = parseModel(req.query);
  const bucket = parseBucket(req.query.get('bucket'));
  assertBucketCount(range, BUCKET_MS[bucket], bucket === '5m');
  const { sqlite } = ctx.dash;
  const all = traceFilter(range, model);
  const ok = traceFilter(range, model, ["outcome = 'ok'"]);

  const perModel = run(
    sqlite.prepare(
      `SELECT model_key AS model, COUNT(*) AS turns,
              SUM(outcome = 'ok' AND mtp_cycles IS NOT NULL) AS mtpTurns,
              SUM(CASE WHEN outcome = 'ok' THEN mtp_cycles END) AS cycles,
              SUM(CASE WHEN outcome = 'ok' AND mtp_mean_accepted IS NOT NULL THEN mtp_mean_accepted * mtp_cycles END) AS acc,
              SUM(CASE WHEN outcome = 'ok' AND mtp_mean_accepted IS NOT NULL THEN mtp_cycles END) AS accW,
              SUM(CASE WHEN outcome = 'ok' AND mtp_mean_drafts IS NOT NULL THEN mtp_mean_drafts * mtp_cycles END) AS drafts,
              SUM(CASE WHEN outcome = 'ok' AND mtp_mean_drafts IS NOT NULL THEN mtp_cycles END) AS draftsW,
              SUM(CASE WHEN outcome = 'ok' AND mtp_mean_depth IS NOT NULL THEN mtp_mean_depth * mtp_cycles END) AS depth,
              SUM(CASE WHEN outcome = 'ok' AND mtp_mean_depth IS NOT NULL THEN mtp_cycles END) AS depthW
       FROM traces ${all.where} GROUP BY model_key ORDER BY model_key`,
    ),
    all.args,
  );

  const byPos = new Map<string, Array<{ sum: number; cycles: number }>>();
  const posRows = sqlite.prepare(
    `SELECT model_key AS model, mtp_cycles AS cycles, mtp_accept_by_pos AS byPos FROM traces
     ${traceFilter(range, model, ["outcome = 'ok'", 'mtp_accept_by_pos IS NOT NULL']).where}`,
  );
  for (const row of posRows.iterate(...ok.args) as Iterable<Row>) {
    const weight = toNum(row.cycles);
    if (weight === null || weight <= 0 || typeof row.byPos !== 'string') continue;
    let rates: unknown;
    try {
      rates = JSON.parse(row.byPos);
    } catch {
      continue;
    }
    if (!Array.isArray(rates)) continue;
    const key = String(row.model);
    let acc = byPos.get(key);
    if (acc === undefined) {
      acc = [];
      byPos.set(key, acc);
    }
    for (let pos = 0; pos < rates.length; pos++) {
      const rate = toNum(rates[pos]);
      if (rate === null) continue;
      acc[pos] ??= { sum: 0, cycles: 0 };
      acc[pos].sum += rate * weight;
      acc[pos].cycles += weight;
    }
  }

  const tps = new ValueScan();
  const tpsRows = sqlite.prepare(
    `SELECT model_key AS model, mtp_cycles IS NOT NULL AS mtp, ${DECODE_TPS_SQL} AS decodeTps FROM traces ${ok.where}`,
  );
  for (const row of tpsRows.iterate(...ok.args) as Iterable<Row>) {
    tps.add(String(row.model), toInt(row.mtp) === 1 ? 'with' : 'without', toNum(row.decodeTps));
  }

  const trend = run(
    sqlite.prepare(
      `SELECT ${bucketSql(bucket)} AS t, model_key AS model, ${MTP_WEIGHTED_SELECT}
       FROM traces ${ok.where} GROUP BY 1, 2 ORDER BY 1, 2`,
    ),
    ok.args,
  );

  return {
    range,
    models: perModel.map((row) => {
      const key = String(row.model);
      const positions = byPos.get(key) ?? [];
      const acceptByPos: Array<{ pos: number; rate: number; cycles: number }> = [];
      for (let pos = 0; pos < positions.length; pos++) {
        const p = positions[pos];
        if (p !== undefined && p.cycles > 0) acceptByPos.push({ pos, rate: p.sum / p.cycles, cycles: p.cycles });
      }
      return {
        model: key,
        turns: toInt(row.turns),
        mtpTurns: toInt(row.mtpTurns),
        cycles: toInt(row.cycles),
        meanAccepted: weighted(row.acc, row.accW),
        meanDrafts: weighted(row.drafts, row.draftsW),
        meanDepth: weighted(row.depth, row.depthW),
        acceptByPos,
        decodeTpsWithMtp: quantilesOf(tps.values(key, 'with')),
        decodeTpsWithoutMtp: quantilesOf(tps.values(key, 'without')),
      };
    }),
    trend: trend.map((row) => ({
      t: toInt(row.t),
      model: String(row.model),
      meanAccepted: weighted(row.mtpAcceptedCycles, row.mtpCycles),
      mtpTurns: toInt(row.mtpTurns),
    })),
  };
}

const TURN_COLUMNS = `trace_id AS traceId, ts, model_key AS modelKey, model, model_type AS modelType, outcome,
  error_kind AS errorKind, fail_phase AS failPhase, finish_reason AS finishReason, endpoint, stream, dispatch,
  prompt_tokens AS promptTokens, cached_tokens AS cachedTokens, output_tokens AS outputTokens,
  reasoning_tokens AS reasoningTokens, ctx_window AS ctxWindow, ${CTX_FILL_SQL} AS ctxFillPct,
  pre_dispatch_ms AS preDispatchMs, queue_ms AS queueMs, load_wait_ms AS loadWaitMs, resolve_ms AS resolveMs,
  ttft_ms AS ttftMs, CASE WHEN ${PREFILL_MEASURED_SQL} THEN prefill_ms END AS prefillMs,
  CASE WHEN ${PREFILL_MEASURED_SQL} THEN prefill_tokens END AS prefillTokens, ${SETUP_MS_SQL} AS setupMs,
  ${DECODE_WALL_SQL} AS decodeMs, duration_ms AS durationMs, ${PREFILL_TPS_SQL} AS prefillTps,
  decode_tps AS decodeTps, mtp_cycles AS mtpCycles, mtp_mean_accepted AS mtpMeanAccepted, cold_hits AS coldHits,
  cold_misses AS coldMisses, tool_count AS toolCount, tool_call_count AS toolCallCount, image_count AS imageCount,
  concurrency, concurrency_peak AS concurrencyPeak, mem_active_bytes AS memActiveBytes, pid`;

function turnRow(row: Row): TurnRow {
  const key = String(row.modelKey);
  const model = strOrNull(row.model);
  const stream = toNum(row.stream);
  return {
    traceId: String(row.traceId),
    ts: toInt(row.ts),
    model: key,
    alias: model !== null && model !== key ? model : null,
    modelType: strOrNull(row.modelType),
    outcome: row.outcome === 'error' || row.outcome === 'cancelled' ? row.outcome : 'ok',
    errorKind: strOrNull(row.errorKind),
    failPhase: strOrNull(row.failPhase),
    finishReason: strOrNull(row.finishReason),
    endpoint: strOrNull(row.endpoint),
    stream: stream === null ? null : stream === 1,
    dispatch: strOrNull(row.dispatch),
    promptTokens: toNum(row.promptTokens),
    cachedTokens: toNum(row.cachedTokens),
    outputTokens: toNum(row.outputTokens),
    reasoningTokens: toNum(row.reasoningTokens),
    ctxWindow: toNum(row.ctxWindow),
    ctxFillPct: toNum(row.ctxFillPct),
    preDispatchMs: toNum(row.preDispatchMs),
    queueMs: toNum(row.queueMs),
    loadWaitMs: toNum(row.loadWaitMs),
    resolveMs: toNum(row.resolveMs),
    ttftMs: toNum(row.ttftMs),
    prefillMs: toNum(row.prefillMs),
    prefillTokens: toNum(row.prefillTokens),
    setupMs: toNum(row.setupMs),
    decodeMs: toNum(row.decodeMs),
    durationMs: toNum(row.durationMs),
    prefillTps: toNum(row.prefillTps),
    decodeTps: toNum(row.decodeTps),
    mtpCycles: toNum(row.mtpCycles),
    mtpMeanAccepted: toNum(row.mtpMeanAccepted),
    coldHits: toNum(row.coldHits),
    coldMisses: toNum(row.coldMisses),
    toolCount: toNum(row.toolCount),
    toolCallCount: toNum(row.toolCallCount),
    imageCount: toNum(row.imageCount),
    concurrency: toNum(row.concurrency),
    concurrencyPeak: toNum(row.concurrencyPeak),
    memActiveBytes: toNum(row.memActiveBytes),
    pid: toNum(row.pid),
  };
}

export function handleMetricsTurns(ctx: WorkerApiContext, req: ApiRequest): MetricsTurns {
  const { sqlite } = ctx.dash;
  const traceId = req.query.get('traceId');
  if (traceId !== null && traceId !== '') {
    const rows = run(sqlite.prepare(`SELECT ${TURN_COLUMNS} FROM traces WHERE trace_id = ?`), [traceId]).map(turnRow);
    return { range: { from: null, to: null }, rows, next: null, total: rows.length };
  }

  const range = parseRange(req.query);
  const model = parseModel(req.query);
  const outcome = parseOutcome(req.query.get('outcome'));
  const rawLimit = parseNumberParam(req.query, 'limit');
  if (rawLimit !== null && (!Number.isInteger(rawLimit) || rawLimit < 1)) {
    throw ApiError.badRequest('limit must be a positive integer');
  }
  const limit = Math.min(rawLimit ?? TURNS_LIMIT_DEFAULT, TURNS_LIMIT_MAX);
  const beforeTs = parseNumberParam(req.query, 'beforeTs');
  const beforeTraceId = req.query.get('beforeTraceId');
  if ((beforeTs === null) !== (beforeTraceId === null || beforeTraceId === '')) {
    throw ApiError.badRequest('beforeTs and beforeTraceId go together');
  }

  const filter = traceFilter(range, model, outcome !== null ? ['outcome = ?'] : []);
  const filterArgs: Args = outcome !== null ? [outcome, ...filter.args] : filter.args;
  const total = toInt((sqlite.prepare(`SELECT COUNT(*) AS n FROM traces ${filter.where}`).get(...filterArgs) as Row).n);

  let where = filter.where;
  const args: Args = [...filterArgs];
  if (beforeTs !== null && beforeTraceId !== null) {
    where = `${where === '' ? 'WHERE' : `${where} AND`} (ts < ? OR (ts = ? AND trace_id < ?))`;
    args.push(beforeTs, beforeTs, beforeTraceId);
  }
  const rows = run(
    sqlite.prepare(`SELECT ${TURN_COLUMNS} FROM traces ${where} ORDER BY ts DESC, trace_id DESC LIMIT ${limit + 1}`),
    args,
  ).map(turnRow);
  const page = rows.slice(0, limit);
  const last = page[page.length - 1];
  return {
    range,
    rows: page,
    next: rows.length > limit && last !== undefined ? { beforeTs: last.ts, beforeTraceId: last.traceId } : null,
    total,
  };
}
