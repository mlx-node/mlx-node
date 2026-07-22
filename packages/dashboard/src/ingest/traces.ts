import { existsSync, readFileSync, readdirSync, statSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';

import { eq, isNotNull } from 'drizzle-orm';

import type { DashboardDb } from '../db/open.js';
import { traces } from '../db/schema.js';
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

    let content: string;
    try {
      content = readFileSync(filePath, 'utf8');
    } catch {
      continue;
    }
    files++;

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
      // single bad record cannot throw out of this per-line loop and abort the
      // rest of the file, later files, or the retention reconciliation below.
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
      records++;
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

  return { files, records, pruned };
}
