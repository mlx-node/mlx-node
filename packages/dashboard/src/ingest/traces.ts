import { existsSync, lstatSync, readFileSync, readdirSync, statSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';

import { eq, isNotNull } from 'drizzle-orm';

import type { DashboardDb } from '../db/open.js';
import { traceFiles, traces } from '../db/schema.js';
import { metricsTraceDir } from '../paths.js';

export interface TraceIngestResult {
  files: number;
  records: number;
  pruned: number;
}

/** Structural view of a `MetricsTraceRecord` line (B1). Read defensively. */
interface ParsedTrace {
  traceId?: unknown;
  sessionId?: unknown;
  rootSessionId?: unknown;
  ts?: unknown;
  model?: unknown;
  ttftMs?: unknown;
  prefillTps?: unknown;
  decodeTps?: unknown;
  mtpCycles?: unknown;
  mtpMeanAccepted?: unknown;
  durationMs?: unknown;
  queueMs?: unknown;
  resident?: unknown;
  finishReason?: unknown;
  promptTokens?: unknown;
  cachedTokens?: unknown;
  outputTokens?: unknown;
  reasoningTokens?: unknown;
  coldHits?: unknown;
  coldMisses?: unknown;
  coldBytesWritten?: unknown;
  coldBytesRestored?: unknown;
}

const DAY_MS = 86_400_000;

function numOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function strOrNull(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

/** Encode a JSONL boolean (`resident`) into the SQLite 0/1 column; non-booleans → null. */
function boolToInt(value: unknown): number | null {
  return typeof value === 'boolean' ? (value ? 1 : 0) : null;
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
  const retentionDays = opts?.retentionDays ?? 30;
  const cutoff = Date.now() - retentionDays * DAY_MS;

  let files = 0;
  let records = 0;
  let pruned = 0;
  if (!existsSync(traceDir)) {
    // A vanished trace directory means zero live source files: run the same
    // retention reconciliation the non-empty path does, against an empty live
    // set, so deleting the dir never leaves ingested rows visible forever. Rows
    // with a NULL source_file predate tracking and are left untouched.
    db.delete(traces).where(isNotNull(traces.sourceFile)).run();
    // No live files means no valid watermarks; clear them so a recreated dir is
    // re-ingested from scratch and the table never tracks vanished files.
    db.delete(traceFiles).run();
    return { files, records, pruned };
  }

  // No-follow root guard (same lstat stance as `download.ts` `assertRealDirOrAbsent`):
  // the prune below enumerates with `readdirSync`, ages entries with `statSync`, and
  // deletes with `unlinkSync` — all of which FOLLOW symlinks. If `traceDir` itself is
  // a symlink (or any non-directory), a relocated-metrics link could redirect the
  // 30-day prune onto an EXTERNAL target and delete real files there. This is
  // best-effort background work, so rather than throw we skip the whole pass (prune
  // nothing, ingest nothing) and leave everything untouched; a real directory behaves
  // exactly as before.
  const rootStat = lstatSync(traceDir);
  if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) {
    return { files, records, pruned };
  }

  for (const name of readdirSync(traceDir)) {
    if (!name.endsWith('.jsonl')) continue;
    const filePath = join(traceDir, name);

    let stat;
    try {
      stat = statSync(filePath);
    } catch {
      continue;
    }
    if (stat.mtimeMs < cutoff) {
      try {
        unlinkSync(filePath);
      } catch {
        // Best-effort prune: a file we cannot remove is left in place, and its
        // rows are kept (the file is still the source of truth).
        continue;
      }
      pruned++;
      db.delete(traces).where(eq(traces.sourceFile, name)).run();
      continue;
    }

    // Incremental skip (mirrors the session ingest watermark): a file whose
    // floored mtime AND size match its stored watermark has already been fully
    // ingested, so skip the read/parse/insert entirely. Without this, every 30s
    // rescan re-reads and re-parses the whole retention window on the event loop.
    const mtime = Math.floor(stat.mtimeMs);
    const size = stat.size;
    const watermark = db
      .select({ mtime: traceFiles.lastIngestedMtime, size: traceFiles.lastIngestedSize })
      .from(traceFiles)
      .where(eq(traceFiles.name, name))
      .all();
    if (watermark.length > 0 && watermark[0].mtime === mtime && watermark[0].size === size) {
      continue;
    }

    let content: string;
    try {
      content = readFileSync(filePath, 'utf8');
    } catch {
      continue;
    }
    files++;

    // A changed file is the authoritative snapshot of its OWN rows, so replace
    // that file's set transactionally instead of appending to it: DELETE this
    // file's prior rows, re-insert every record parsed from the current snapshot,
    // then stamp the watermark — all-or-nothing. Insert-only ingest keyed on the
    // globally-unique trace_id would strand a record dropped from the file
    // ([A,B]→[C] keeps A,B) or keep the OLD field values when a record is
    // rewritten in place (onConflictDoNothing skips the existing trace_id). The
    // disposable index must mirror the JSONL source of truth exactly. Rows for a
    // trace_id owned by a DIFFERENT file are untouched (the delete is scoped to
    // source_file === name); that cross-file case does not arise under the
    // per-turn-UUID + per-(date,pid) append-only writer anyway. onConflictDoNothing
    // stays for intra-file idempotency (a duplicate trace_id within the same file).
    sqlite.exec('BEGIN');
    let fileRecords = 0;
    try {
      db.delete(traces).where(eq(traces.sourceFile, name)).run();
      for (const line of content.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        let rec: unknown;
        try {
          rec = JSON.parse(trimmed);
        } catch {
          continue;
        }
        // A syntactically-valid but non-object line (`null`, a scalar, an array)
        // carries no fields to read. Skip it here — before any field access — so a
        // single bad record cannot throw out of this per-line loop and roll back
        // the rest of the file's valid records.
        if (typeof rec !== 'object' || rec === null || Array.isArray(rec)) continue;
        const trace = rec as ParsedTrace;
        if (typeof trace.traceId !== 'string') continue;
        const ts = numOrNull(trace.ts);
        db.insert(traces)
          .values({
            traceId: trace.traceId,
            sessionId: strOrNull(trace.sessionId),
            rootSessionId: strOrNull(trace.rootSessionId),
            ts: ts ?? 0,
            model: strOrNull(trace.model),
            ttftMs: numOrNull(trace.ttftMs),
            prefillTps: numOrNull(trace.prefillTps),
            decodeTps: numOrNull(trace.decodeTps),
            mtpCycles: numOrNull(trace.mtpCycles),
            mtpMeanAccepted: numOrNull(trace.mtpMeanAccepted),
            durationMs: numOrNull(trace.durationMs),
            queueMs: numOrNull(trace.queueMs),
            resident: boolToInt(trace.resident),
            finishReason: strOrNull(trace.finishReason),
            promptTokens: numOrNull(trace.promptTokens),
            cachedTokens: numOrNull(trace.cachedTokens),
            outputTokens: numOrNull(trace.outputTokens),
            reasoningTokens: numOrNull(trace.reasoningTokens),
            coldHits: numOrNull(trace.coldHits),
            coldMisses: numOrNull(trace.coldMisses),
            coldBytesWritten: numOrNull(trace.coldBytesWritten),
            coldBytesRestored: numOrNull(trace.coldBytesRestored),
            sourceFile: name,
          })
          .onConflictDoNothing()
          .run();
        fileRecords++;
      }

      // Stamp the watermark AFTER every line landed, in the SAME transaction so a
      // crash mid-file re-reads next pass rather than committing a partial replace.
      // A file with only malformed lines still records a watermark, so a
      // permanently garbage file is not re-parsed on every rescan.
      db.insert(traceFiles)
        .values({ name, lastIngestedMtime: mtime, lastIngestedSize: size })
        .onConflictDoUpdate({
          target: traceFiles.name,
          set: { lastIngestedMtime: mtime, lastIngestedSize: size },
        })
        .run();
      sqlite.exec('COMMIT');
      records += fileRecords;
    } catch {
      // Roll back this file's replace so a DB error never leaves a half-deleted or
      // half-inserted set (nor a stamped watermark); the next rescan retries it.
      sqlite.exec('ROLLBACK');
      continue;
    }
  }

  // Reconcile rows whose backing file has vanished (manually deleted, or pruned
  // in an earlier partial run). Rows with a NULL source_file predate source
  // tracking and are left untouched.
  const live = new Set(readdirSync(traceDir).filter((n) => n.endsWith('.jsonl')));
  const tracked = sqlite
    .prepare('SELECT DISTINCT source_file AS sf FROM traces WHERE source_file IS NOT NULL')
    .all() as Array<{ sf: string }>;
  for (const { sf } of tracked) {
    if (!live.has(sf)) db.delete(traces).where(eq(traces.sourceFile, sf)).run();
  }

  // Reconcile watermark rows the same way: drop any whose file has vanished
  // (pruned or manually deleted) so the table tracks only live files and a
  // reused filename is never wrongly skipped.
  const trackedFiles = db.select({ name: traceFiles.name }).from(traceFiles).all();
  for (const { name: tracked } of trackedFiles) {
    if (!live.has(tracked)) db.delete(traceFiles).where(eq(traceFiles.name, tracked)).run();
  }

  return { files, records, pruned };
}
