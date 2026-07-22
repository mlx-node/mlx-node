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
