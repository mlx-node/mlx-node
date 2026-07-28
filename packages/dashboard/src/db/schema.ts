import { integer, real, sqliteTable, text } from 'drizzle-orm/sqlite-core';

export const sessions = sqliteTable('sessions', {
  id: text('id').primaryKey(),
  path: text('path').notNull(),
  cwd: text('cwd').notNull(),
  name: text('name'),
  created: integer('created').notNull(),
  modified: integer('modified').notNull(),
  messageCount: integer('message_count').notNull().default(0),
  firstMessage: text('first_message'),
  lastIngestedMtime: integer('last_ingested_mtime').notNull().default(0),
  lastIngestedSize: integer('last_ingested_size').notNull().default(0),
});

export const turns = sqliteTable('turns', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  sessionId: text('session_id').notNull(),
  entryId: text('entry_id'),
  traceId: text('trace_id'),
  ts: integer('ts').notNull(),
  model: text('model'),
  inputTokens: integer('input_tokens'),
  outputTokens: integer('output_tokens'),
  cachedTokens: integer('cached_tokens'),
  reasoningTokens: integer('reasoning_tokens'),
});

export const traces = sqliteTable('traces', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  traceId: text('trace_id').notNull().unique(),
  sessionId: text('session_id'),
  /** Root pi session id (subagent turns correlate to their delegating root). */
  rootSessionId: text('root_session_id'),
  ts: integer('ts').notNull(),
  model: text('model'),
  ttftMs: real('ttft_ms'),
  prefillTps: real('prefill_tps'),
  decodeTps: real('decode_tps'),
  mtpCycles: integer('mtp_cycles'),
  mtpMeanAccepted: real('mtp_mean_accepted'),
  durationMs: real('duration_ms'),
  /** Queue + cold-load wait (ms) before native work began this turn. */
  queueMs: integer('queue_ms'),
  finishReason: text('finish_reason'),
  /** 1 when the model was already warm/resident, 0 on a cold load/swap; null if unknown. */
  resident: integer('resident'),
  promptTokens: integer('prompt_tokens'),
  cachedTokens: integer('cached_tokens'),
  outputTokens: integer('output_tokens'),
  reasoningTokens: integer('reasoning_tokens'),
  coldHits: integer('cold_hits'),
  coldMisses: integer('cold_misses'),
  coldBytesWritten: integer('cold_bytes_written'),
  coldBytesRestored: integer('cold_bytes_restored'),
  /**
   * Canonical cold-cache root this turn ran against — the scope key for the
   * Cache page. NULL means "unattributed": either the tier was off (paired with
   * `coldEnabled = 0`) or the record predates attribution (`coldEnabled` NULL
   * too). Never folded into the shown cache's totals; reported separately.
   */
  coldRoot: text('cold_root'),
  /** 1 when the cold tier was open for the turn, 0 when off; NULL if unknown. */
  coldEnabled: integer('cold_enabled'),
  coldEnqueued: integer('cold_enqueued'),
  coldQueueDrops: integer('cold_queue_drops'),
  coldEvictions: integer('cold_evictions'),
  coldCorruptions: integer('cold_corruptions'),
  /** Cumulative (not per-turn) — see `MetricsTraceRecord.coldCorruptionsTotal`. */
  coldCorruptionsTotal: integer('cold_corruptions_total'),
  /** Cumulative (not per-turn) — see `MetricsTraceRecord.coldQueueDropsTotal`. */
  coldQueueDropsTotal: integer('cold_queue_drops_total'),
  /**
   * Writes the queue accepted that never reached disk (read-only / full /
   * unmounted root, failed rename or fsync). The native writer is fail-open
   * and reports the error to nobody, so this is the only column that can tell
   * a cache storing nothing from a cache with nothing to store.
   */
  coldWriteErrors: integer('cold_write_errors'),
  /** Cumulative (not per-turn) — see `MetricsTraceRecord.coldWriteErrorsTotal`. */
  coldWriteErrorsTotal: integer('cold_write_errors_total'),
  /**
   * Restores the walk refused outright. Neither a hit nor a miss, so without
   * this column a refused restore is a row of zeros — read as "nothing ran"
   * rather than "reuse was refused". Summed, never MAX'd: unlike corruptions,
   * a decline is normal on any prompt's first turn, so a "did it ever happen"
   * latch would be pinned on from the first minute.
   */
  coldRestoreDeclines: integer('cold_restore_declines'),
  /**
   * Restores a family threw away after the walk served them. Preceded by real
   * `coldHits`, which is what makes it the harder of the two to spot.
   */
  coldSidecarRestoreSuppressed: integer('cold_sidecar_restore_suppressed'),
  sourceFile: text('source_file'),
});

/**
 * Per-file ingest watermark for the trace JSONL directory. Trace rows are keyed
 * by `traceId` (many per file), so — unlike `sessions` — there is no 1:1
 * file↔row anchor to hang the watermark on; this dedicated table records the
 * `(mtime, size)` of each fully-ingested trace file so a 30s rescan can skip
 * unchanged files instead of re-reading the entire history every pass.
 */
export const traceFiles = sqliteTable('trace_files', {
  name: text('name').primaryKey(),
  lastIngestedMtime: integer('last_ingested_mtime').notNull(),
  lastIngestedSize: integer('last_ingested_size').notNull(),
});
