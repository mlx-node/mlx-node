/**
 * Metrics aggregation: the global overview.
 *
 * The SQL comments ARE the correctness argument for the gross-vs-net token
 * accounting, and must not drift from the query they explain.
 */

import type { ApiRequest, WorkerApiContext } from '../context.js';
import { queryInt, toInt, toNum } from '../context.js';

/**
 * Shape of `GET /api/metrics/overview`. Documented here because the C10 UI is
 * the sole consumer. All token/count fields are non-negative integers; average
 * fields are `number | null` (null when no sample carried that column).
 */
export interface MetricsOverview {
  range: { from: number | null; to: number | null };
  tokensByDay: Array<{ day: string; input: number; output: number; cached: number; reasoning: number }>;
  throughputByModel: Array<{
    model: string;
    avgDecodeTps: number | null;
    avgPrefillTps: number | null;
    avgTtftMs: number | null;
    samples: number;
  }>;
  /**
   * Day-bucketed throughput/TTFT trend per model — the time series the spec
   * promises alongside the range-wide `throughputByModel` averages. Buckets share
   * `tokensByDay`'s `date(ts/1000,'unixepoch')` expression so the UI can align
   * them. Numeric averages are coerced to a number (0 when the bucket carried no
   * sample for that column).
   */
  throughputTrend: Array<{
    model: string;
    day: string;
    decodeTps: number | null;
    prefillTps: number | null;
    ttftMs: number | null;
    samples: number;
  }>;
  mtpByModel: Array<{ model: string; meanAccepted: number | null; avgCycles: number | null; samples: number }>;
  modelShare: Array<{ model: string; turns: number; outputTokens: number }>;
  totals: {
    turns: number;
    traces: number;
    inputTokens: number;
    outputTokens: number;
    cachedTokens: number;
    reasoningTokens: number;
  };
}

function rangeClause(from: number | null, to: number | null, column: string): { sql: string; args: number[] } {
  const parts: string[] = [];
  const args: number[] = [];
  if (from !== null) {
    parts.push(`${column} >= ?`);
    args.push(from);
  }
  if (to !== null) {
    parts.push(`${column} <= ?`);
    args.push(to);
  }
  return { sql: parts.length > 0 ? parts.join(' AND ') : '', args };
}

export function handleMetricsOverview(ctx: WorkerApiContext, req: ApiRequest): unknown {
  const from = queryInt(req.query, 'from');
  const to = queryInt(req.query, 'to');
  const { sqlite } = ctx.dash;

  const tracesRange = rangeClause(from, to, 'ts');
  const tracesWhere = (extra: string): string => {
    const parts = [extra, tracesRange.sql].filter((p) => p !== '');
    return parts.length > 0 ? `WHERE ${parts.join(' AND ')}` : '';
  };

  // A trace row stores GROSS `prompt_tokens`; the `input` figures the UI reports
  // are net-of-cache (what the turn actually paid to prefill). Clamp to the
  // producer's net value: `MAX(prompt - cached, 0)` (SQLite two-arg `MAX(a,b)`
  // is the scalar clamp).
  const traceNetInput = 'MAX(COALESCE(prompt_tokens, 0) - COALESCE(cached_tokens, 0), 0)';

  const tokensByDay = sqlite
    .prepare(
      `SELECT date(ts / 1000, 'unixepoch') AS day,
              COALESCE(SUM(${traceNetInput}), 0) AS input,
              COALESCE(SUM(output_tokens), 0) AS output,
              COALESCE(SUM(cached_tokens), 0) AS cached,
              COALESCE(SUM(reasoning_tokens), 0) AS reasoning
       FROM traces ${tracesWhere('ts > 0')}
       GROUP BY day ORDER BY day`,
    )
    .all(...tracesRange.args)
    .map((row) => ({
      day: String(row.day),
      input: toInt(row.input),
      output: toInt(row.output),
      cached: toInt(row.cached),
      reasoning: toInt(row.reasoning),
    }));

  const throughputByModel = sqlite
    .prepare(
      `SELECT COALESCE(model, 'unknown') AS model,
              AVG(decode_tps) AS avgDecodeTps, AVG(prefill_tps) AS avgPrefillTps,
              AVG(ttft_ms) AS avgTtftMs, COUNT(*) AS samples
       FROM traces ${tracesWhere('')} GROUP BY model ORDER BY samples DESC`,
    )
    .all(...tracesRange.args)
    .map((row) => ({
      model: String(row.model),
      avgDecodeTps: toNum(row.avgDecodeTps),
      avgPrefillTps: toNum(row.avgPrefillTps),
      avgTtftMs: toNum(row.avgTtftMs),
      samples: toInt(row.samples),
    }));

  // Same per-model averages as above, but bucketed per day so the UI can chart a
  // trend. Uses the identical `date(ts/1000,'unixepoch')` bucket and `ts > 0`
  // guard as `tokensByDay` so the two series line up on the same day keys.
  const throughputTrend = sqlite
    .prepare(
      `SELECT COALESCE(model, 'unknown') AS model,
              date(ts / 1000, 'unixepoch') AS day,
              AVG(decode_tps) AS avgDecodeTps, AVG(prefill_tps) AS avgPrefillTps,
              AVG(ttft_ms) AS avgTtftMs, COUNT(*) AS samples
       FROM traces ${tracesWhere('ts > 0')} GROUP BY model, day ORDER BY day, model`,
    )
    .all(...tracesRange.args)
    .map((row) => ({
      model: String(row.model),
      day: String(row.day),
      decodeTps: toNum(row.avgDecodeTps),
      prefillTps: toNum(row.avgPrefillTps),
      ttftMs: toNum(row.avgTtftMs),
      samples: toInt(row.samples),
    }));

  const mtpByModel = sqlite
    .prepare(
      `SELECT COALESCE(model, 'unknown') AS model,
              AVG(mtp_mean_accepted) AS meanAccepted, AVG(mtp_cycles) AS avgCycles,
              COUNT(mtp_mean_accepted) AS samples
       FROM traces ${tracesWhere('mtp_mean_accepted IS NOT NULL')} GROUP BY model ORDER BY samples DESC`,
    )
    .all(...tracesRange.args)
    .map((row) => ({
      model: String(row.model),
      meanAccepted: toNum(row.meanAccepted),
      avgCycles: toNum(row.avgCycles),
      samples: toInt(row.samples),
    }));

  const modelShare = sqlite
    .prepare(
      `SELECT COALESCE(model, 'unknown') AS model, COUNT(*) AS turns,
              COALESCE(SUM(output_tokens), 0) AS outputTokens
       FROM traces ${tracesWhere('')}
       GROUP BY model ORDER BY turns DESC`,
    )
    .all(...tracesRange.args)
    .map((row) => ({ model: String(row.model), turns: toInt(row.turns), outputTokens: toInt(row.outputTokens) }));

  const turnTotals = sqlite
    .prepare(
      `SELECT COUNT(*) AS turns, COALESCE(SUM(${traceNetInput}), 0) AS inputTokens,
              COALESCE(SUM(output_tokens), 0) AS outputTokens,
              COALESCE(SUM(cached_tokens), 0) AS cachedTokens,
              COALESCE(SUM(reasoning_tokens), 0) AS reasoningTokens
       FROM traces ${tracesWhere('')}`,
    )
    .get(...tracesRange.args);

  const overview: MetricsOverview = {
    range: { from, to },
    tokensByDay,
    throughputByModel,
    throughputTrend,
    mtpByModel,
    modelShare,
    totals: {
      // One trace row is one turn of work — both counts read the same column now
      // that session-derived turn rows no longer exist.
      turns: toInt(turnTotals?.turns),
      traces: toInt(turnTotals?.turns),
      inputTokens: toInt(turnTotals?.inputTokens),
      outputTokens: toInt(turnTotals?.outputTokens),
      cachedTokens: toInt(turnTotals?.cachedTokens),
      reasoningTokens: toInt(turnTotals?.reasoningTokens),
    },
  };
  return overview;
}
