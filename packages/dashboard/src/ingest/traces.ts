import { existsSync, lstatSync, readdirSync, unlinkSync, type Stats } from 'node:fs';
import { join } from 'node:path';

import { eq, isNotNull } from 'drizzle-orm';

import type { DashboardDb } from '../db/open.js';
import { traceFiles, traces } from '../db/schema.js';
import { metricsTraceDir } from '../paths.js';
import {
  liveSourceNames,
  parseJsonObjectLine,
  readJsonlIncrement,
  statJsonlEntries,
  type JsonlMark,
} from './jsonl-tail.js';

export interface TraceIngestResult {
  files: number;
  records: number;
  pruned: number;
  /** Human-readable notes for a pass that was skipped (unreadable trace root). */
  warnings: string[];
}

/** Structural view of a `MetricsTraceRecord` line (v1 or v2). Read defensively. */
type ParsedTrace = Record<string, unknown>;

const OUTCOMES = ['ok', 'error', 'cancelled'];
const ERROR_KINDS = ['finish_error', 'exception'];
const FAIL_PHASES = ['queued', 'slot', 'prepare', 'prefill', 'decode'];
const ENDPOINTS = ['messages', 'responses', 'other'];
const DISPATCHES = ['start', 'continue', 'continue_tool'];
const MTP_ACCEPT_BY_POS_MAX = 16;

const DAY_MS = 86_400_000;

/**
 * How far back trace JSONL (and therefore every row derived from it) is kept.
 * Exported so the API can LABEL the cache trend with its real window instead of
 * letting it read as all-time, and so the two can never drift apart.
 */
export const TRACE_RETENTION_DAYS = 30;

function numOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function strOrNull(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

/** Encode a JSONL boolean into the SQLite 0/1 column; non-booleans → null. */
function boolToInt(value: unknown): number | null {
  return typeof value === 'boolean' ? (value ? 1 : 0) : null;
}

function nonEmptyOrNull(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function enumOrNull(value: unknown, allowed: readonly string[]): string | null {
  return typeof value === 'string' && allowed.includes(value) ? value : null;
}

/** JSON text of an array of at most 16 finite numbers, else null. */
function rateListText(value: unknown): string | null {
  if (!Array.isArray(value) || value.length > MTP_ACCEPT_BY_POS_MAX) return null;
  for (const item of value) if (typeof item !== 'number' || !Number.isFinite(item)) return null;
  return JSON.stringify(value);
}

/** Both measured fields, or neither. */
function measuredPrefill(trace: ParsedTrace): { ms: number; tokens: number } | null {
  const ms = trace.prefillMs;
  const tokens = trace.prefillTokens;
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms <= 0) return null;
  if (typeof tokens !== 'number' || !Number.isInteger(tokens) || tokens <= 0) return null;
  return { ms, tokens };
}

/** The `traces` row of one parsed line, or null when the line carries no trace id. */
function traceRow(trace: ParsedTrace, sourceFile: string): typeof traces.$inferInsert | null {
  if (typeof trace.traceId !== 'string') return null;
  const model = strOrNull(trace.model);
  const measured = measuredPrefill(trace);
  const modelId = nonEmptyOrNull(trace.modelId);
  return {
    traceId: trace.traceId,
    sessionId: strOrNull(trace.sessionId),
    ts: numOrNull(trace.ts) ?? 0,
    pid: numOrNull(trace.pid),
    model,
    modelId,
    modelType: nonEmptyOrNull(trace.modelType),
    modelKey: modelId ?? model ?? 'unknown',
    // v1 wrote successful turns only, and has no outcome field.
    outcome: trace.outcome === undefined ? 'ok' : (enumOrNull(trace.outcome, OUTCOMES) ?? 'ok'),
    errorKind: enumOrNull(trace.errorKind, ERROR_KINDS),
    failPhase: enumOrNull(trace.failPhase, FAIL_PHASES),
    endpoint: enumOrNull(trace.endpoint, ENDPOINTS),
    stream: boolToInt(trace.stream),
    dispatch: enumOrNull(trace.dispatch, DISPATCHES),
    ttftMs: numOrNull(trace.ttftMs),
    prefillMs: measured?.ms ?? null,
    prefillTokens: measured?.tokens ?? null,
    decodeTps: numOrNull(trace.decodeTps),
    mtpCycles: numOrNull(trace.mtpCycles),
    mtpMeanAccepted: numOrNull(trace.mtpMeanAccepted),
    mtpMeanDrafts: numOrNull(trace.mtpMeanDrafts),
    mtpMeanDepth: numOrNull(trace.mtpMeanDepth),
    mtpAcceptByPos: rateListText(trace.mtpAcceptByPos),
    durationMs: numOrNull(trace.durationMs),
    queueMs: numOrNull(trace.queueMs),
    loadWaitMs: numOrNull(trace.loadWaitMs),
    resolveMs: numOrNull(trace.resolveMs),
    preDispatchMs: numOrNull(trace.preDispatchMs),
    waitMs: numOrNull(trace.waitMs),
    slotMs: numOrNull(trace.slotMs),
    prepareMs: numOrNull(trace.prepareMs),
    finishReason: strOrNull(trace.finishReason),
    promptTokens: numOrNull(trace.promptTokens),
    cachedTokens: numOrNull(trace.cachedTokens),
    outputTokens: numOrNull(trace.outputTokens),
    reasoningTokens: numOrNull(trace.reasoningTokens),
    prefillDoneTokens: numOrNull(trace.prefillDoneTokens),
    ctxWindow: numOrNull(trace.ctxWindow),
    toolCount: numOrNull(trace.toolCount),
    toolCallCount: numOrNull(trace.toolCallCount),
    imageCount: numOrNull(trace.imageCount),
    concurrency: numOrNull(trace.concurrency),
    concurrencyPeak: numOrNull(trace.concurrencyPeak),
    memActiveBytes: numOrNull(trace.memActiveBytes),
    memPeakBytes: numOrNull(trace.memPeakBytes),
    pipelinesCreated: numOrNull(trace.pipelinesCreated),
    pipelineCreateMs: numOrNull(trace.pipelineCreateMs),
    coldHits: numOrNull(trace.coldHits),
    coldMisses: numOrNull(trace.coldMisses),
    coldBytesWritten: numOrNull(trace.coldBytesWritten),
    coldBytesRestored: numOrNull(trace.coldBytesRestored),
    // The writer only emits `coldRoot` for a tier that was actually open, so an
    // absent value stays NULL: the "unattributed" bucket of the Cache page.
    coldRoot: strOrNull(trace.coldRoot),
    coldEnabled: boolToInt(trace.coldEnabled),
    coldEnqueued: numOrNull(trace.coldEnqueued),
    coldQueueDrops: numOrNull(trace.coldQueueDrops),
    coldEvictions: numOrNull(trace.coldEvictions),
    coldCorruptions: numOrNull(trace.coldCorruptions),
    coldCorruptionsTotal: numOrNull(trace.coldCorruptionsTotal),
    coldQueueDropsTotal: numOrNull(trace.coldQueueDropsTotal),
    coldWriteErrors: numOrNull(trace.coldWriteErrors),
    coldWriteErrorsTotal: numOrNull(trace.coldWriteErrorsTotal),
    coldRestoreDeclines: numOrNull(trace.coldRestoreDeclines),
    // One line per column: a sidecar counter this list forgets is dropped in
    // silence, and `coldSidecarInstalled` is the one nothing else can stand in for.
    coldSidecarCaptureReached: numOrNull(trace.coldSidecarCaptureReached),
    coldSidecarChainEmpty: numOrNull(trace.coldSidecarChainEmpty),
    coldSidecarBoundarySkips: numOrNull(trace.coldSidecarBoundarySkips),
    coldSidecarAlreadyPersisted: numOrNull(trace.coldSidecarAlreadyPersisted),
    coldSidecarEnqueued: numOrNull(trace.coldSidecarEnqueued),
    coldSidecarQueueDrops: numOrNull(trace.coldSidecarQueueDrops),
    coldSidecarInstalled: numOrNull(trace.coldSidecarInstalled),
    sourceFile,
  };
}

/**
 * Ingest every trace JSONL under `dir` into the SQLite index. Each line is an
 * independent `MetricsTraceRecord`; malformed lines are skipped. Inserts are
 * idempotent on `trace_id`. Files whose mtime is older than `retentionDays` are
 * unlinked AND their rows deleted (JSONL is the source of truth, so the on-disk
 * retention drives it). Rows whose backing file no longer exists are reconciled
 * away so expired telemetry is never stored indefinitely.
 */
export async function ingestTraces(
  dash: DashboardDb,
  dir?: string,
  opts?: { retentionDays?: number },
): Promise<TraceIngestResult> {
  const { db, sqlite } = dash;
  const traceDir = dir ?? metricsTraceDir();
  const retentionDays = opts?.retentionDays ?? TRACE_RETENTION_DAYS;
  const cutoff = Date.now() - retentionDays * DAY_MS;

  let files = 0;
  let records = 0;
  let pruned = 0;
  const warnings: string[] = [];
  if (!existsSync(traceDir)) {
    // A vanished trace directory means zero live source files: run the same
    // retention reconciliation the non-empty path does, against an empty live
    // set, so deleting the dir never leaves ingested rows visible forever. Rows
    // with a NULL source_file predate tracking and are left untouched.
    db.delete(traces).where(isNotNull(traces.sourceFile)).run();
    // No live files means no valid watermarks; clear them so a recreated dir is
    // re-ingested from scratch and the table never tracks vanished files.
    db.delete(traceFiles).run();
    return { files, records, pruned, warnings };
  }

  // No-follow root guard (same lstat stance as `download.ts` `assertRealDirOrAbsent`):
  // the prune below enumerates with `readdirSync`, ages entries with `statSync`, and
  // deletes with `unlinkSync` — all of which FOLLOW symlinks. If `traceDir` itself is
  // a symlink (or any non-directory), a relocated-metrics link could redirect the
  // 30-day prune onto an EXTERNAL target and delete real files there. This is
  // best-effort background work, so rather than throw we skip the whole pass (prune
  // nothing, ingest nothing) and leave everything untouched; a real directory behaves
  // exactly as before.
  //
  // The stat itself can still fail after `existsSync` said yes — another process
  // removing `~/.mlx-node/metrics` mid-pass, or the parent turning unsearchable —
  // so it is guarded too, and skips the pass rather than throwing out of the
  // caller's combined ingest.
  let rootStat: Stats;
  try {
    rootStat = lstatSync(traceDir);
  } catch (err) {
    warnings.push(`${traceDir}: trace root could not be read; scan skipped (${String(err)})`);
    return { files, records, pruned, warnings };
  }
  if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) {
    return { files, records, pruned, warnings };
  }

  // Reconcile rows whose backing file has vanished (renamed or manually deleted)
  // BEFORE the per-file ingest loop. `traces.trace_id` is globally UNIQUE and
  // independent of source_file, so a trace moved from a.jsonl to b.jsonl cannot be
  // inserted under b while a.jsonl still owns its unique row (`onConflictDoNothing`
  // skips it); if a.jsonl's row were only reconciled AFTER the loop, that skipped
  // trace would end up in NO row and never re-insert (the next scan skips b by its
  // watermark). Deleting a.jsonl's rows up front lets b's insert find no conflict.
  // Rows with a NULL source_file predate source tracking and are left untouched.
  //
  // That reconciliation is exactly why an unreadable root (chmod/ACL drift, an
  // external volume returning EACCES, fd exhaustion, a vanish racing another
  // process) must skip the pass instead of falling back to an empty listing: an
  // empty live set reads as "every source file is gone" and deletes the whole
  // index on a fault that says nothing about the files.
  let entries: string[];
  try {
    entries = readdirSync(traceDir);
  } catch (err) {
    warnings.push(`${traceDir}: trace root could not be listed; scan skipped (${String(err)})`);
    return { files, records, pruned, warnings };
  }

  const kinds = statJsonlEntries(traceDir, entries);
  const liveFiles = liveSourceNames(kinds);
  // The tracked set comes from the one-row-per-file mark table: a DISTINCT scan
  // of `traces.source_file` is a full table scan, and a pass can run every 2 s.
  const marks = new Map<string, JsonlMark>();
  for (const row of db.select().from(traceFiles).all()) {
    if (!liveFiles.has(row.name)) {
      db.delete(traces).where(eq(traces.sourceFile, row.name)).run();
      db.delete(traceFiles).where(eq(traceFiles.name, row.name)).run();
      continue;
    }
    marks.set(row.name, {
      mtime: row.lastIngestedMtime,
      size: row.lastIngestedSize,
      consumed: row.consumed,
      tail: row.tail,
    });
  }

  for (const [name, stat] of kinds) {
    // An entry that is not a regular file was left out of `liveFiles` above and
    // must never reach the prune or the read. An unstattable one stays live and
    // is retried next pass.
    if (stat === null || !stat.isFile()) continue;
    const filePath = join(traceDir, name);

    if (stat.mtimeMs < cutoff) {
      try {
        unlinkSync(filePath);
      } catch {
        // A file we cannot remove is still the source of truth: its rows stay.
        continue;
      }
      pruned++;
      db.delete(traces).where(eq(traces.sourceFile, name)).run();
      db.delete(traceFiles).where(eq(traceFiles.name, name)).run();
      continue;
    }

    let increment;
    try {
      increment = readJsonlIncrement(filePath, stat, marks.get(name));
    } catch {
      continue;
    }
    if (increment.mode === 'skip') continue;
    files++;

    // `replace` deletes the file's rows first; both modes stamp the mark in the
    // same transaction as the inserts, so a crash re-reads instead of double-inserting.
    sqlite.exec('BEGIN');
    let fileRecords = 0;
    try {
      if (increment.mode === 'replace') db.delete(traces).where(eq(traces.sourceFile, name)).run();
      for (const line of increment.lines) {
        const parsed = parseJsonObjectLine(line);
        if (parsed === null) continue;
        const row = traceRow(parsed, name);
        if (row === null) continue;
        db.insert(traces).values(row).onConflictDoNothing().run();
        fileRecords++;
      }
      const { mark } = increment;
      const set = {
        lastIngestedMtime: mark.mtime,
        lastIngestedSize: mark.size,
        consumed: mark.consumed,
        tail: mark.tail,
      };
      db.insert(traceFiles)
        .values({ name, ...set })
        .onConflictDoUpdate({ target: traceFiles.name, set })
        .run();
      sqlite.exec('COMMIT');
      records += fileRecords;
    } catch {
      sqlite.exec('ROLLBACK');
      continue;
    }
  }

  return { files, records, pruned, warnings };
}
