/**
 * Metrics aggregation: the per-session turn/trace join and the global overview.
 *
 * Every SQL string and its explanatory comment moved here verbatim — those
 * comments ARE the correctness argument for the fork-dedup, trace-only union and
 * gross-vs-net token accounting, and must not drift from the query they explain.
 */

import type { ApiRequest, WorkerApiContext } from '../context.js';
import { queryInt, toInt, toNum } from '../context.js';
import { requireSession } from './sessions.js';

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

export function handleSessionMetrics(ctx: WorkerApiContext, req: ApiRequest): unknown {
  requireSession(ctx, req.params.id);
  // The per-turn set the SPA charts / counts / token totals are built from is this
  // session's persisted `turns` (LEFT JOINed to their trace) UNIONed with any
  // delegated subagent turns: a child runs on an in-memory session manager (no
  // session JSONL → no `turns` row), yet its shared-provider `traces` row carries
  // the token columns and points back here via `root_session_id`. Without the union
  // the child's tokens/count are dropped from the totals even though the SPA already
  // shows its model badge + folds its throughput in (from the `traces` array below),
  // and it disagrees with the global overview. The union admits ONLY GENUINE children:
  // a child stamps `session_id = <child in-memory id>` (≠ root) and
  // `root_session_id = <root id>` (stream-adapter.ts: `sessionId` from the per-request
  // options, `rootSessionId` from the submit-time cache-owner root), so keying on
  // `root_session_id = ? AND session_id != ?` selects them. Keying on `session_id = ?`
  // too would resurrect an ABANDONED root turn — when a root branches, ingestion drops
  // its assistant turn from `turns` but the trace lingers with `session_id = root`
  // (== root_session_id); it would then pass the trace-only dedup and be miscounted as
  // a delegated turn. The dedup admits ONLY traces with no correlated `turns` row for
  // THIS session (the inner `AND trace_id IS NOT NULL` keeps `NOT IN` from going NULL
  // and dropping the whole set), and — mirroring the global fix in
  // handleMetricsOverview — the trace side stores GROSS `prompt_tokens` so its
  // `inputTokens` is clamped to the producer's net value `MAX(prompt-cached,0)` to
  // avoid a gross-vs-net double-count against the turns side's already-net input.
  const turnRows = ctx.dash.sqlite
    .prepare(
      `SELECT * FROM (
         SELECT t.entry_id AS entryId, t.trace_id AS traceId, t.ts AS ts, t.model AS model,
                t.input_tokens AS inputTokens, t.output_tokens AS outputTokens,
                t.cached_tokens AS cachedTokens, t.reasoning_tokens AS reasoningTokens,
                tr.ttft_ms AS ttftMs, tr.prefill_tps AS prefillTps, tr.decode_tps AS decodeTps,
                tr.mtp_cycles AS mtpCycles, tr.mtp_mean_accepted AS mtpMeanAccepted,
                tr.duration_ms AS durationMs, tr.finish_reason AS finishReason,
                tr.cold_hits AS coldHits, tr.cold_misses AS coldMisses,
                tr.cold_bytes_written AS coldBytesWritten, tr.cold_bytes_restored AS coldBytesRestored
         FROM turns t
         LEFT JOIN traces tr ON tr.trace_id = t.trace_id
         WHERE t.session_id = ?
         UNION ALL
         SELECT NULL AS entryId, tr.trace_id AS traceId, tr.ts AS ts, tr.model AS model,
                MAX(COALESCE(tr.prompt_tokens, 0) - COALESCE(tr.cached_tokens, 0), 0) AS inputTokens,
                tr.output_tokens AS outputTokens, tr.cached_tokens AS cachedTokens,
                tr.reasoning_tokens AS reasoningTokens,
                tr.ttft_ms AS ttftMs, tr.prefill_tps AS prefillTps, tr.decode_tps AS decodeTps,
                tr.mtp_cycles AS mtpCycles, tr.mtp_mean_accepted AS mtpMeanAccepted,
                tr.duration_ms AS durationMs, tr.finish_reason AS finishReason,
                tr.cold_hits AS coldHits, tr.cold_misses AS coldMisses,
                tr.cold_bytes_written AS coldBytesWritten, tr.cold_bytes_restored AS coldBytesRestored
         FROM traces tr
         WHERE tr.root_session_id = ? AND tr.session_id != ?
           AND tr.trace_id NOT IN (SELECT trace_id FROM turns WHERE session_id = ? AND trace_id IS NOT NULL)
       )
       ORDER BY ts`,
    )
    .all(req.params.id, req.params.id, req.params.id, req.params.id);
  // Include this session's own ACTIVE-branch turns AND any subagent turns delegated
  // under it — but NOT an abandoned root turn's lingering trace. The SPA derives its
  // model badges and avg TTFT/decode chips from this `traces` array while the per-turn
  // charts use `turns` (above), so keying on `session_id = ?` alone would readmit the
  // abandoned root trace `turnRows` deliberately excludes (a branched root drops its
  // assistant turn from `turns` but the trace lingers with `session_id = root`), making
  // the chips disagree with the transcript. Mirror `turnRows`: select a root trace only
  // when it correlates to an ACTIVE `turns` row (via `trace_id`), plus genuine delegated
  // children (`root_session_id = ? AND session_id != ?`) not already correlated here. A
  // child (subagent) trace carries no persisted session JSONL, but its `root_session_id`
  // points back here (Finding 11b).
  const traceRows = ctx.dash.sqlite
    .prepare(
      `SELECT trace_id AS traceId, session_id AS sessionId, root_session_id AS rootSessionId,
              ts, model, ttft_ms AS ttftMs, prefill_tps AS prefillTps,
              decode_tps AS decodeTps, mtp_cycles AS mtpCycles, mtp_mean_accepted AS mtpMeanAccepted,
              duration_ms AS durationMs, finish_reason AS finishReason,
              cold_hits AS coldHits, cold_misses AS coldMisses,
              cold_bytes_written AS coldBytesWritten, cold_bytes_restored AS coldBytesRestored
       FROM traces tr
       WHERE tr.trace_id IN (SELECT trace_id FROM turns WHERE session_id = ? AND trace_id IS NOT NULL)
          OR (tr.root_session_id = ? AND tr.session_id != ?
              AND tr.trace_id NOT IN (SELECT trace_id FROM turns WHERE session_id = ? AND trace_id IS NOT NULL))
       ORDER BY ts`,
    )
    .all(req.params.id, req.params.id, req.params.id, req.params.id);
  return { sessionId: req.params.id, turns: turnRows, traces: traceRows };
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

  const turnsRange = rangeClause(from, to, 'ts');
  const tracesRange = rangeClause(from, to, 'ts');
  const turnsWhere = (extra: string): string => {
    const parts = [extra, turnsRange.sql].filter((p) => p !== '');
    return parts.length > 0 ? `WHERE ${parts.join(' AND ')}` : '';
  };
  const tracesWhere = (extra: string): string => {
    const parts = [extra, tracesRange.sql].filter((p) => p !== '');
    return parts.length > 0 ? `WHERE ${parts.join(' AND ')}` : '';
  };

  // A forked session copies inherited turns VERBATIM (same trace_id/entry_id) into
  // a new session file, so the same inference lands as multiple `turns` rows. The
  // per-session views keep every copy (each transcript is correct), but these
  // GLOBAL token sums must count each inference once — collapse copies on their
  // canonical identity `COALESCE(trace_id, entry_id, CAST(id AS TEXT))` (the
  // autoincrement id keeps genuinely-distinct, both-null rows separate).
  const dedupKey = 'COALESCE(trace_id, entry_id, CAST(id AS TEXT))';

  // Subagent turns run on an in-memory session manager (no session JSONL → no
  // `turns` row), yet the shared provider still writes a `traces` row carrying the
  // token columns. UNION those TRACE-ONLY rows into the turns-derived token
  // aggregates below so delegated work is not silently underreported. This guard
  // admits ONLY traces with no correlated `turns` row, so a normal/forked turn's
  // tokens stay sourced from `turns` and are never counted twice. The inner
  // `WHERE trace_id IS NOT NULL` is load-bearing: a NULL in the subquery would make
  // `NOT IN` evaluate to NULL for every row and drop the whole trace-only set. Each
  // trace-only row is one delegated turn of real work, so it also adds 1 to the
  // per-model / overall turn COUNT.
  const traceOnly = 'trace_id NOT IN (SELECT trace_id FROM turns WHERE trace_id IS NOT NULL)';

  // The turns side's `input_tokens` is pi `usage.input`, already NET of cache
  // (`max(0, promptTokens - cacheRead)`; see packages/agent/src/provider/events.ts).
  // A trace row instead stores GROSS `prompt_tokens` (provider/index.ts). Projecting
  // gross into the same `input` column as the turns side double-counts cached tokens
  // for trace-only rows, so clamp the trace side to the producer's net value:
  // `MAX(prompt - cached, 0)` (SQLite two-arg `MAX(a,b)` is the scalar clamp). The
  // other projected trace columns are already apples-to-apples with turns.
  const traceNetInput = 'MAX(COALESCE(prompt_tokens, 0) - COALESCE(cached_tokens, 0), 0)';

  const tokensByDay = sqlite
    .prepare(
      `SELECT date(ts / 1000, 'unixepoch') AS day,
              COALESCE(SUM(input_tokens), 0) AS input,
              COALESCE(SUM(output_tokens), 0) AS output,
              COALESCE(SUM(cached_tokens), 0) AS cached,
              COALESCE(SUM(reasoning_tokens), 0) AS reasoning
       FROM (SELECT input_tokens, output_tokens, cached_tokens, reasoning_tokens, ts
             FROM turns ${turnsWhere('ts > 0')} GROUP BY ${dedupKey}
             UNION ALL
             SELECT ${traceNetInput}, output_tokens, cached_tokens, reasoning_tokens, ts
             FROM traces ${tracesWhere(`ts > 0 AND ${traceOnly}`)})
       GROUP BY day ORDER BY day`,
    )
    .all(...turnsRange.args, ...tracesRange.args)
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
       FROM (SELECT model, output_tokens
             FROM turns ${turnsWhere('')} GROUP BY ${dedupKey}
             UNION ALL
             SELECT model, output_tokens
             FROM traces ${tracesWhere(traceOnly)})
       GROUP BY model ORDER BY turns DESC`,
    )
    .all(...turnsRange.args, ...tracesRange.args)
    .map((row) => ({ model: String(row.model), turns: toInt(row.turns), outputTokens: toInt(row.outputTokens) }));

  const turnTotals = sqlite
    .prepare(
      `SELECT COUNT(*) AS turns, COALESCE(SUM(input_tokens), 0) AS inputTokens,
              COALESCE(SUM(output_tokens), 0) AS outputTokens,
              COALESCE(SUM(cached_tokens), 0) AS cachedTokens,
              COALESCE(SUM(reasoning_tokens), 0) AS reasoningTokens
       FROM (SELECT input_tokens, output_tokens, cached_tokens, reasoning_tokens
             FROM turns ${turnsWhere('')} GROUP BY ${dedupKey}
             UNION ALL
             SELECT ${traceNetInput}, output_tokens, cached_tokens, reasoning_tokens
             FROM traces ${tracesWhere(traceOnly)})`,
    )
    .get(...turnsRange.args, ...tracesRange.args);
  const traceTotals = sqlite
    .prepare(`SELECT COUNT(*) AS traces FROM traces ${tracesWhere('')}`)
    .get(...tracesRange.args);

  const overview: MetricsOverview = {
    range: { from, to },
    tokensByDay,
    throughputByModel,
    throughputTrend,
    mtpByModel,
    modelShare,
    totals: {
      turns: toInt(turnTotals?.turns),
      traces: toInt(traceTotals?.traces),
      inputTokens: toInt(turnTotals?.inputTokens),
      outputTokens: toInt(turnTotals?.outputTokens),
      cachedTokens: toInt(turnTotals?.cachedTokens),
      reasoningTokens: toInt(turnTotals?.reasoningTokens),
    },
  };
  return overview;
}
