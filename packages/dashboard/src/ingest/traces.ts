import { existsSync, readFileSync, readdirSync, statSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';

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
  ts?: unknown;
  model?: unknown;
  ttftMs?: unknown;
  prefillTps?: unknown;
  decodeTps?: unknown;
  mtpCycles?: unknown;
  mtpMeanAccepted?: unknown;
  durationMs?: unknown;
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

/**
 * Ingest every trace JSONL under `dir` into the SQLite index. Each line is an
 * independent `MetricsTraceRecord`; malformed lines are skipped. Inserts are
 * idempotent on `trace_id`. Files whose mtime is older than `retentionDays` are
 * unlinked (JSONL is the source of truth, so the on-disk retention drives it).
 */
export async function ingestTraces(
  dash: DashboardDb,
  dir?: string,
  opts?: { retentionDays?: number },
): Promise<TraceIngestResult> {
  const { db } = dash;
  const traceDir = dir ?? metricsTraceDir();
  const retentionDays = opts?.retentionDays ?? 30;
  const cutoff = Date.now() - retentionDays * DAY_MS;

  let files = 0;
  let records = 0;
  let pruned = 0;
  if (!existsSync(traceDir)) return { files, records, pruned };

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
        pruned++;
      } catch {
        // Best-effort prune: a file we cannot remove is left in place.
      }
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
      let rec: ParsedTrace;
      try {
        rec = JSON.parse(trimmed) as ParsedTrace;
      } catch {
        continue;
      }
      if (typeof rec.traceId !== 'string') continue;
      const ts = numOrNull(rec.ts);
      db.insert(traces)
        .values({
          traceId: rec.traceId,
          sessionId: strOrNull(rec.sessionId),
          ts: ts ?? 0,
          model: strOrNull(rec.model),
          ttftMs: numOrNull(rec.ttftMs),
          prefillTps: numOrNull(rec.prefillTps),
          decodeTps: numOrNull(rec.decodeTps),
          mtpCycles: numOrNull(rec.mtpCycles),
          mtpMeanAccepted: numOrNull(rec.mtpMeanAccepted),
          durationMs: numOrNull(rec.durationMs),
          finishReason: strOrNull(rec.finishReason),
          promptTokens: numOrNull(rec.promptTokens),
          cachedTokens: numOrNull(rec.cachedTokens),
          outputTokens: numOrNull(rec.outputTokens),
          reasoningTokens: numOrNull(rec.reasoningTokens),
          coldHits: numOrNull(rec.coldHits),
          coldMisses: numOrNull(rec.coldMisses),
          coldBytesWritten: numOrNull(rec.coldBytesWritten),
          coldBytesRestored: numOrNull(rec.coldBytesRestored),
          sourceFile: name,
        })
        .onConflictDoNothing()
        .run();
      records++;
    }
  }

  return { files, records, pruned };
}
