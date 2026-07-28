import { existsSync, lstatSync, readFileSync, readdirSync, statSync, unlinkSync, type Stats } from 'node:fs';
import { join } from 'node:path';

import { eq, isNotNull } from 'drizzle-orm';

import type { DashboardDb } from '../db/open.js';
import { traceFiles, traces } from '../db/schema.js';
import { metricsTraceDir } from '../paths.js';

export interface TraceIngestResult {
  files: number;
  records: number;
  pruned: number;
  /** Human-readable notes for a pass that was skipped (unreadable trace root). */
  warnings: string[];
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
  coldRoot?: unknown;
  coldEnabled?: unknown;
  coldEnqueued?: unknown;
  coldQueueDrops?: unknown;
  coldEvictions?: unknown;
  coldCorruptions?: unknown;
  coldCorruptionsTotal?: unknown;
  coldQueueDropsTotal?: unknown;
  coldWriteErrors?: unknown;
  coldWriteErrorsTotal?: unknown;
  coldRestoreDeclines?: unknown;
  coldSidecarCaptureReached?: unknown;
  coldSidecarChainEmpty?: unknown;
  coldSidecarBoundarySkips?: unknown;
  coldSidecarAlreadyPersisted?: unknown;
  coldSidecarEnqueued?: unknown;
  coldSidecarQueueDrops?: unknown;
  coldSidecarInstalled?: unknown;
  coldSidecarRestoreSuppressed?: unknown;
}

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

/** Encode a JSONL boolean (`resident`) into the SQLite 0/1 column; non-booleans → null. */
function boolToInt(value: unknown): number | null {
  return typeof value === 'boolean' ? (value ? 1 : 0) : null;
}

/**
 * Stat every `*.jsonl` entry ONCE so liveness is decided from what an entry IS, not
 * from what it is named. Only a regular file is a trace source: a directory or a FIFO
 * can carry the suffix too, and neither is readable as one. A directory fails
 * `readFileSync` with EISDIR (and `unlinkSync` with EPERM), so judged by name it stays
 * "live" forever and keeps its rows and watermark in the index; a FIFO is worse —
 * `readFileSync` BLOCKS on it until a writer closes, and since the server funnels every
 * pass through one serialized ingest chain that wedges the whole dashboard. Neither is
 * reachable from product code (the writer only ever appends to `<date>-<pid>.jsonl`),
 * but the sibling session ingest already stats its candidates, and the same rule belongs
 * here.
 *
 * A stat FAILURE is NOT "gone": a root that is readable but not searchable (mode 0600)
 * lists its entries and then gives EACCES per child. Those names map to `null` and stay
 * live — rows kept, file skipped this pass — the same stance the root guard below takes,
 * so a transient fault never reads as a deletion.
 *
 * `statSync` FOLLOWS symlinks, deliberately unlike `sessions.ts`'s no-follow listing:
 * that one is no-follow so an external Pi transcript cannot surface under the target's
 * id, an identity concern traces do not have. A symlinked trace entry is a legitimate
 * source and ingests today; `unlinkSync` does not follow, so the 30-day prune can only
 * ever remove the link itself, never an external target.
 */
function statTraceEntries(traceDir: string, entries: string[]): Map<string, Stats | null> {
  const kinds = new Map<string, Stats | null>();
  for (const name of entries) {
    if (!name.endsWith('.jsonl')) continue;
    try {
      kinds.set(name, statSync(join(traceDir, name)));
    } catch {
      kinds.set(name, null);
    }
  }
  return kinds;
}

/** Names reconciliation must treat as live: every regular file, plus every unstattable entry. */
function liveSourceNames(kinds: Map<string, Stats | null>): Set<string> {
  const live = new Set<string>();
  for (const [name, stat] of kinds) {
    if (stat === null || stat.isFile()) live.add(name);
  }
  return live;
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

  const kinds = statTraceEntries(traceDir, entries);
  const liveFiles = liveSourceNames(kinds);
  const trackedSources = sqlite
    .prepare('SELECT DISTINCT source_file AS sf FROM traces WHERE source_file IS NOT NULL')
    .all() as Array<{ sf: string }>;
  for (const { sf } of trackedSources) {
    if (!liveFiles.has(sf)) db.delete(traces).where(eq(traces.sourceFile, sf)).run();
  }

  for (const [name, stat] of kinds) {
    // Reuse the one classification above rather than re-deciding here: an entry that
    // is not a regular file was already left out of `liveFiles` (its rows reconciled
    // away), and it must never reach the prune or the read below. An entry we could
    // not stat is skipped too — kept live, retried next pass.
    if (stat === null || !stat.isFile()) continue;
    const filePath = join(traceDir, name);

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
            // The writer only emits `coldRoot` for a tier that was actually
            // open, so an absent value stays NULL here — the "unattributed"
            // bucket the Cache page reports separately rather than folding into
            // the shown cache's hit rate.
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
            // All eight `ColdSidecarStats` counters, not just the suppressed
            // one: this mapping is written a line per column, so a field the
            // agent emits and this list forgets is discarded in silence — and
            // `coldSidecarInstalled` is the one nothing else can stand in for.
            // Every `install_*_cold_sidecar` early-return falls through to a
            // full O(prefix) replay that produces CORRECT state, so a
            // regression from "restored and INSTALLED" to "restored and
            // re-derived" moves no other column in this row.
            coldSidecarCaptureReached: numOrNull(trace.coldSidecarCaptureReached),
            coldSidecarChainEmpty: numOrNull(trace.coldSidecarChainEmpty),
            coldSidecarBoundarySkips: numOrNull(trace.coldSidecarBoundarySkips),
            coldSidecarAlreadyPersisted: numOrNull(trace.coldSidecarAlreadyPersisted),
            coldSidecarEnqueued: numOrNull(trace.coldSidecarEnqueued),
            coldSidecarQueueDrops: numOrNull(trace.coldSidecarQueueDrops),
            coldSidecarInstalled: numOrNull(trace.coldSidecarInstalled),
            coldSidecarRestoreSuppressed: numOrNull(trace.coldSidecarRestoreSuppressed),
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

  // Reconcile watermark rows whose file has vanished (pruned in the loop above, or
  // manually deleted) so the table tracks only live files and a reused filename is
  // never wrongly skipped. Recomputed here — AFTER the loop — so a file just pruned
  // is included in the vanished set. (The trace-row reconciliation runs BEFORE the
  // loop; see the note there — a renamed file's row must be freed before its new
  // name is ingested.)
  //
  // Guarded like the listing above, and for the same reason: the rows committed by
  // the loop stand either way, but reconciling watermarks against a listing we could
  // not take would clear every one of them and force a full re-read next pass.
  let liveNames: string[];
  try {
    liveNames = readdirSync(traceDir);
  } catch (err) {
    warnings.push(`${traceDir}: trace root could not be re-listed; watermark reconciliation skipped (${String(err)})`);
    return { files, records, pruned, warnings };
  }
  // Re-classified, not reused: the loop above may have pruned entries, and the same
  // predicate has to hold here or a row set and its watermark end up disagreeing.
  const live = liveSourceNames(statTraceEntries(traceDir, liveNames));
  const trackedFiles = db.select({ name: traceFiles.name }).from(traceFiles).all();
  for (const { name: tracked } of trackedFiles) {
    if (!live.has(tracked)) db.delete(traceFiles).where(eq(traceFiles.name, tracked)).run();
  }

  return { files, records, pruned, warnings };
}
