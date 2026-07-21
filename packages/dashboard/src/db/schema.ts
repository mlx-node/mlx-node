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
  finishReason: text('finish_reason'),
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
