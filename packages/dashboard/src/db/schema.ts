import { integer, real, sqliteTable, text } from 'drizzle-orm/sqlite-core';

export const traces = sqliteTable('traces', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  traceId: text('trace_id').notNull().unique(),
  sessionId: text('session_id'),
  ts: integer('ts').notNull(),
  pid: integer('pid'),
  /** The name the client asked for; may be an alias of `model_key`. */
  model: text('model'),
  modelId: text('model_id'),
  modelType: text('model_type'),
  /** `modelId ?? model ?? 'unknown'`: the grouping key of every per-model query. */
  modelKey: text('model_key').notNull(),
  outcome: text('outcome').notNull(),
  errorKind: text('error_kind'),
  failPhase: text('fail_phase'),
  endpoint: text('endpoint'),
  stream: integer('stream'),
  dispatch: text('dispatch'),
  ttftMs: real('ttft_ms'),
  /** Measured prefill window (`begin_prefill` → first token sampled); null when not measured. */
  prefillMs: real('prefill_ms'),
  /** Tokens that window processed; set with `prefillMs`, never alone. */
  prefillTokens: integer('prefill_tokens'),
  decodeTps: real('decode_tps'),
  mtpCycles: integer('mtp_cycles'),
  mtpMeanAccepted: real('mtp_mean_accepted'),
  mtpMeanDrafts: real('mtp_mean_drafts'),
  mtpMeanDepth: real('mtp_mean_depth'),
  /** JSON array text of at most 16 rates. */
  mtpAcceptByPos: text('mtp_accept_by_pos'),
  durationMs: real('duration_ms'),
  queueMs: real('queue_ms'),
  loadWaitMs: real('load_wait_ms'),
  resolveMs: real('resolve_ms'),
  preDispatchMs: real('pre_dispatch_ms'),
  /** Native creation → pickup; outside `ttft_ms`. NULL from a record that predates it. */
  waitMs: real('wait_ms'),
  /** Native time with no lane or memory to start; inside `ttft_ms`. */
  slotMs: real('slot_ms'),
  /** Native time preparing the prompt before its prefill; inside `ttft_ms`. */
  prepareMs: real('prepare_ms'),
  finishReason: text('finish_reason'),
  promptTokens: integer('prompt_tokens'),
  cachedTokens: integer('cached_tokens'),
  outputTokens: integer('output_tokens'),
  reasoningTokens: integer('reasoning_tokens'),
  prefillDoneTokens: integer('prefill_done_tokens'),
  ctxWindow: integer('ctx_window'),
  toolCount: integer('tool_count'),
  toolCallCount: integer('tool_call_count'),
  imageCount: integer('image_count'),
  concurrency: integer('concurrency'),
  concurrencyPeak: integer('concurrency_peak'),
  memActiveBytes: integer('mem_active_bytes'),
  memPeakBytes: integer('mem_peak_bytes'),
  /** GPU pipelines created while the turn ran (process-wide); NULL while the counter is off. */
  pipelinesCreated: integer('pipelines_created'),
  pipelineCreateMs: real('pipeline_create_ms'),
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
   * The seven `ColdSidecarStats` counters — the recurrent / sliding-window
   * state that lives OUTSIDE the paged pool. Every one is a per-turn DELTA
   * (the trace writer differences the native snapshot across the turn), so
   * every one SUMS. None has a cumulative `*_total` twin, and MAX would not
   * substitute for one: over deltas it reports the busiest single turn, which
   * answers no question anyone asks.
   *
   * Their own `cold_sidecar_*` prefix rather than a merge with the block
   * counters above, because both native structs carry `enqueued` and
   * `queue_drops` and they count different OBJECTS. The sidecar pair is the
   * sidecar SHARE of the block pair (one admission bumps both), so they are
   * read as a subset and never summed with it.
   */
  coldSidecarCaptureReached: integer('cold_sidecar_capture_reached'),
  coldSidecarChainEmpty: integer('cold_sidecar_chain_empty'),
  coldSidecarBoundarySkips: integer('cold_sidecar_boundary_skips'),
  /**
   * Captures that selected a boundary already on disk — the STEADY STATE of a
   * repeated prompt, not a fault. Without it a healthy run (`enqueued = 0`
   * because the first turn already wrote the chain) is indistinguishable from
   * a run whose ladder collapsed.
   */
  coldSidecarAlreadyPersisted: integer('cold_sidecar_already_persisted'),
  coldSidecarEnqueued: integer('cold_sidecar_enqueued'),
  coldSidecarQueueDrops: integer('cold_sidecar_queue_drops'),
  /**
   * Restored sidecars a family INSTALLED as live state. The only column that
   * separates "restored and used" from "restored, then silently re-derived by
   * a full O(prefix) replay": the replay produces correct state, so every
   * other column in this row — `cached_tokens`, `cold_hits`,
   * `cold_corruptions` — reads identically either way.
   */
  coldSidecarInstalled: integer('cold_sidecar_installed'),
  sourceFile: text('source_file'),
});

/**
 * Per-file ingest mark of a JSONL source. `consumed` always ends on a `\n`;
 * `tail` is the base64 of the bytes just before it, which proves on the next
 * pass that the file only grew (append) rather than being rewritten (replace).
 */
export const traceFiles = sqliteTable('trace_files', {
  name: text('name').primaryKey(),
  lastIngestedMtime: integer('last_ingested_mtime').notNull(),
  lastIngestedSize: integer('last_ingested_size').notNull(),
  consumed: integer('consumed').notNull(),
  tail: text('tail').notNull(),
});

export const runtimeFiles = sqliteTable('runtime_files', {
  name: text('name').primaryKey(),
  lastIngestedMtime: integer('last_ingested_mtime').notNull(),
  lastIngestedSize: integer('last_ingested_size').notNull(),
  consumed: integer('consumed').notNull(),
  tail: text('tail').notNull(),
});

/** One inference process, keyed `<startedAt>-<pid>`. */
export const runtimeProcs = sqliteTable('runtime_procs', {
  procKey: text('proc_key').primaryKey(),
  pid: integer('pid').notNull(),
  startedAt: integer('started_at').notNull(),
  host: text('host'),
  version: text('version'),
  port: integer('port'),
  chip: text('chip'),
  wiredLimitBytes: integer('wired_limit_bytes'),
  totalMemBytes: integer('total_mem_bytes'),
  firstTs: integer('first_ts').notNull(),
  lastTs: integer('last_ts').notNull(),
  exitTs: integer('exit_ts'),
});

export const runtimeSamples = sqliteTable('runtime_samples', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  procKey: text('proc_key').notNull(),
  ts: integer('ts').notNull(),
  dtMs: integer('dt_ms'),
  busy: integer('busy').notNull(),
  busyMs: integer('busy_ms'),
  health: text('health'),
  memActive: integer('mem_active'),
  memPeak: integer('mem_peak'),
  memCache: integer('mem_cache'),
  rss: integer('rss'),
  inFlight: integer('in_flight'),
  queued: integer('queued'),
  resident: integer('resident'),
  dOk: integer('d_ok'),
  dError: integer('d_error'),
  dCancelled: integer('d_cancelled'),
  dOutTokens: integer('d_out_tokens'),
  dRejected: integer('d_rejected'),
  sourceFile: text('source_file').notNull(),
});

export const runtimeSched = sqliteTable('runtime_sched', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  procKey: text('proc_key').notNull(),
  ts: integer('ts').notNull(),
  model: text('model').notNull(),
  blockCapacity: integer('block_capacity'),
  allocatedBlocks: integer('allocated_blocks'),
  freeBlocks: integer('free_blocks'),
  reclaimableBlocks: integer('reclaimable_blocks'),
  watermarkBlocks: integer('watermark_blocks'),
  reservedBlocks: integer('reserved_blocks'),
  memoryCapacityBytes: integer('memory_capacity_bytes'),
  memoryWatermarkBytes: integer('memory_watermark_bytes'),
  reservedBlockBytes: integer('reserved_block_bytes'),
  reservedStateBytes: integer('reserved_state_bytes'),
  globalSteps: integer('global_steps'),
  admitted: integer('admitted'),
  completed: integer('completed'),
  maxBatchOccupancy: integer('max_batch_occupancy'),
  fusedGreedyEpilogueSteps: integer('fused_greedy_epilogue_steps'),
  admissionDeferredBlocks: integer('admission_deferred_blocks'),
  admissionDeferredState: integer('admission_deferred_state'),
  rowsAllocEvicted: integer('rows_alloc_evicted'),
  preemptions: integer('preemptions'),
  preemptionsRecompute: integer('preemptions_recompute'),
  preemptionsSsd: integer('preemptions_ssd'),
  ssdRestoreWaiting: integer('ssd_restore_waiting'),
  ssdRestoreBytes: integer('ssd_restore_bytes'),
  ssdRestoreWaitMs: real('ssd_restore_wait_ms'),
  /** Turns sent to the model since its load, all lanes; null in a record of an engine that did not write it. */
  turns: integer('turns'),
  /** JSON `[[occupancy, steps], …]`. */
  occHist: text('occ_hist'),
  sourceFile: text('source_file').notNull(),
});

export const runtimeLoads = sqliteTable('runtime_loads', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  procKey: text('proc_key').notNull(),
  ts: integer('ts').notNull(),
  model: text('model').notNull(),
  modelType: text('model_type'),
  capability: text('capability'),
  ms: real('ms'),
  ok: integer('ok'),
  sourceFile: text('source_file').notNull(),
});

/** Every table of the index. The open-time drift check is built from this list, so a table cannot escape it. */
export const ALL_TABLES = [traces, traceFiles, runtimeFiles, runtimeProcs, runtimeSamples, runtimeSched, runtimeLoads];
