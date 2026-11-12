/**
 * Ingest of the inference runtime series (`metrics/runtime/<date>-<pid>.jsonl`)
 * into the `runtime_*` tables. Same root guards, retention prune and
 * tail-incremental read as the trace ingest. `live-*.json` heartbeats are not
 * sources here; the live watcher reads them.
 */

import { existsSync, lstatSync, readdirSync, unlinkSync, type Stats } from 'node:fs';
import { join } from 'node:path';
import type { StatementSync } from 'node:sqlite';

import { eq } from 'drizzle-orm';

import type { DashboardDb } from '../db/open.js';
import { runtimeFiles, runtimeLoads, runtimeProcs, runtimeSamples, runtimeSched } from '../db/schema.js';
import { metricsRuntimeDir } from '../paths.js';
import {
  liveSourceNames,
  parseJsonObjectLine,
  readJsonlIncrement,
  statJsonlEntries,
  type JsonlMark,
} from './jsonl-tail.js';

export interface RuntimeIngestResult {
  files: number;
  records: number;
  pruned: number;
  warnings: string[];
}

export const RUNTIME_RETENTION_DAYS = 14;
const DAY_MS = 86_400_000;
const OCC_HIST_MAX = 512;

const HEALTH = ['ok', 'loading', 'degraded', 'error'];
const HOSTS = ['serve', 'launch', 'desktop', 'embedded'];
const CAPABILITIES = ['chat', 'audio', 'rerank'];

type Rec = Record<string, unknown>;
type Bind = number | string | null;

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function nonEmpty(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function member(value: unknown, allowed: readonly string[]): string | null {
  return typeof value === 'string' && allowed.includes(value) ? value : null;
}

function flag(value: unknown): number | null {
  return typeof value === 'boolean' ? (value ? 1 : 0) : null;
}

/** JSON text of `[[occupancy, steps], …]` pairs of finite numbers, else null. */
function occHistText(value: unknown): string | null {
  if (!Array.isArray(value) || value.length > OCC_HIST_MAX) return null;
  for (const pair of value) {
    if (!Array.isArray(pair) || pair.length !== 2) return null;
    if (num(pair[0]) === null || num(pair[1]) === null) return null;
  }
  return JSON.stringify(value);
}

const SCHED_NUMERIC = [
  'blockCapacity',
  'allocatedBlocks',
  'freeBlocks',
  'reclaimableBlocks',
  'watermarkBlocks',
  'reservedBlocks',
  'memoryCapacityBytes',
  'memoryWatermarkBytes',
  'reservedBlockBytes',
  'reservedStateBytes',
  'globalSteps',
  'admitted',
  'completed',
  'maxBatchOccupancy',
  'fusedGreedyEpilogueSteps',
  'admissionDeferredBlocks',
  'admissionDeferredState',
  'rowsAllocEvicted',
  'preemptions',
  'preemptionsRecompute',
  'preemptionsSsd',
  'ssdRestoreWaiting',
  'ssdRestoreBytes',
  'ssdRestoreWaitMs',
  'turns',
] as const;

interface Statements {
  upsertProc: StatementSync;
  procInfo: StatementSync;
  procExit: StatementSync;
  sample: StatementSync;
  sched: StatementSync;
  load: StatementSync;
  fileProcKeys: StatementSync;
  pruneProc: StatementSync;
}

function prepare(dash: DashboardDb): Statements {
  const { sqlite } = dash;
  return {
    upsertProc: sqlite.prepare(
      `INSERT INTO runtime_procs (proc_key, pid, started_at, first_ts, last_ts) VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(proc_key) DO UPDATE SET
         first_ts = MIN(first_ts, excluded.first_ts),
         last_ts = MAX(last_ts, excluded.last_ts)`,
    ),
    procInfo: sqlite.prepare(
      `UPDATE runtime_procs SET host = ?, version = ?, port = ?, chip = ?, wired_limit_bytes = ?, total_mem_bytes = ?
       WHERE proc_key = ?`,
    ),
    procExit: sqlite.prepare('UPDATE runtime_procs SET exit_ts = MAX(COALESCE(exit_ts, 0), ?) WHERE proc_key = ?'),
    sample: sqlite.prepare(
      `INSERT INTO runtime_samples (proc_key, ts, dt_ms, busy, busy_ms, health, mem_active, mem_peak, mem_cache, rss,
         in_flight, queued, resident, d_ok, d_error, d_cancelled, d_out_tokens, d_rejected, source_file)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ),
    sched: sqlite.prepare(
      `INSERT INTO runtime_sched (proc_key, ts, model, block_capacity, allocated_blocks, free_blocks,
         reclaimable_blocks, watermark_blocks, reserved_blocks, memory_capacity_bytes, memory_watermark_bytes,
         reserved_block_bytes, reserved_state_bytes, global_steps, admitted, completed, max_batch_occupancy,
         fused_greedy_epilogue_steps, admission_deferred_blocks, admission_deferred_state, rows_alloc_evicted,
         preemptions, preemptions_recompute, preemptions_ssd, ssd_restore_waiting, ssd_restore_bytes,
         ssd_restore_wait_ms, turns, occ_hist, source_file)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ),
    load: sqlite.prepare(
      `INSERT INTO runtime_loads (proc_key, ts, model, model_type, capability, ms, ok, source_file)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ),
    fileProcKeys: sqlite.prepare(
      `SELECT proc_key FROM runtime_samples WHERE source_file = ?1
       UNION SELECT proc_key FROM runtime_sched WHERE source_file = ?1
       UNION SELECT proc_key FROM runtime_loads WHERE source_file = ?1`,
    ),
    pruneProc: sqlite.prepare(
      `DELETE FROM runtime_procs WHERE proc_key = ?1
         AND NOT EXISTS (SELECT 1 FROM runtime_samples WHERE proc_key = ?1)
         AND NOT EXISTS (SELECT 1 FROM runtime_sched WHERE proc_key = ?1)
         AND NOT EXISTS (SELECT 1 FROM runtime_loads WHERE proc_key = ?1)`,
    ),
  };
}

/** Insert one record's rows. Returns its process key, or null for a line that is not a runtime record. */
function ingestRecord(st: Statements, rec: Rec, sourceFile: string): string | null {
  const ts = num(rec.ts);
  const pid = num(rec.pid);
  const startedAt = num(rec.startedAt);
  if (ts === null || pid === null || startedAt === null) return null;
  const procKey = `${startedAt}-${pid}`;
  const run = (statement: StatementSync, ...values: Bind[]): void => {
    statement.run(...values);
  };
  switch (rec.k) {
    case 'proc':
      run(st.upsertProc, procKey, pid, startedAt, ts, ts);
      run(
        st.procInfo,
        member(rec.host, HOSTS),
        str(rec.version),
        num(rec.port),
        str(rec.chip),
        num(rec.wiredLimitBytes),
        num(rec.totalMemBytes),
        procKey,
      );
      return procKey;
    case 'sample': {
      const busy = flag(rec.busy);
      if (busy === null) return null;
      run(st.upsertProc, procKey, pid, startedAt, ts, ts);
      run(
        st.sample,
        procKey,
        ts,
        num(rec.dtMs),
        busy,
        num(rec.busyMs),
        member(rec.health, HEALTH),
        num(rec.memActive),
        num(rec.memPeak),
        num(rec.memCache),
        num(rec.rss),
        num(rec.inFlight),
        num(rec.queued),
        num(rec.resident),
        num(rec.dOk),
        num(rec.dError),
        num(rec.dCancelled),
        num(rec.dOutTokens),
        num(rec.dRejected),
        sourceFile,
      );
      return procKey;
    }
    case 'sched': {
      const model = nonEmpty(rec.model);
      if (model === null) return null;
      run(st.upsertProc, procKey, pid, startedAt, ts, ts);
      run(
        st.sched,
        procKey,
        ts,
        model,
        ...SCHED_NUMERIC.map((field) => num(rec[field])),
        occHistText(rec.occHist),
        sourceFile,
      );
      return procKey;
    }
    case 'load': {
      const model = nonEmpty(rec.model);
      if (model === null) return null;
      run(st.upsertProc, procKey, pid, startedAt, ts, ts);
      run(
        st.load,
        procKey,
        ts,
        model,
        nonEmpty(rec.modelType),
        member(rec.capability, CAPABILITIES),
        num(rec.ms),
        flag(rec.ok),
        sourceFile,
      );
      return procKey;
    }
    case 'exit':
      run(st.upsertProc, procKey, pid, startedAt, ts, ts);
      run(st.procExit, ts, procKey);
      return procKey;
    default:
      return null;
  }
}

/** Delete one source file's data rows, adding the process keys they held to `procKeys`. */
function deleteFileRows(dash: DashboardDb, st: Statements, name: string, procKeys: Set<string>): void {
  const { db } = dash;
  for (const row of st.fileProcKeys.all(name)) procKeys.add(String(row.proc_key));
  db.delete(runtimeSamples).where(eq(runtimeSamples.sourceFile, name)).run();
  db.delete(runtimeSched).where(eq(runtimeSched.sourceFile, name)).run();
  db.delete(runtimeLoads).where(eq(runtimeLoads.sourceFile, name)).run();
}

function clearAll(dash: DashboardDb): void {
  const { db } = dash;
  db.delete(runtimeSamples).run();
  db.delete(runtimeSched).run();
  db.delete(runtimeLoads).run();
  db.delete(runtimeProcs).run();
  db.delete(runtimeFiles).run();
}

/**
 * Ingest every runtime series file under `dir`. Never throws for a file-level
 * fault: a file that cannot be read is retried next pass, an unreadable root
 * skips the pass with a warning.
 */
export async function ingestRuntime(
  dash: DashboardDb,
  dir?: string,
  opts?: { retentionDays?: number; now?: number },
): Promise<RuntimeIngestResult> {
  const { db, sqlite } = dash;
  const runtimeDir = dir ?? metricsRuntimeDir();
  const cutoff = (opts?.now ?? Date.now()) - (opts?.retentionDays ?? RUNTIME_RETENTION_DAYS) * DAY_MS;
  let files = 0;
  let records = 0;
  let pruned = 0;
  const warnings: string[] = [];

  if (!existsSync(runtimeDir)) {
    clearAll(dash);
    return { files, records, pruned, warnings };
  }
  // A symlinked or non-directory root would redirect the retention prune onto
  // an external target: skip the whole pass instead.
  let rootStat: Stats;
  try {
    rootStat = lstatSync(runtimeDir);
  } catch (err) {
    warnings.push(`${runtimeDir}: runtime root could not be read; scan skipped (${String(err)})`);
    return { files, records, pruned, warnings };
  }
  if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) return { files, records, pruned, warnings };
  // An empty listing taken on a fault would read as "every file is gone".
  let entries: string[];
  try {
    entries = readdirSync(runtimeDir);
  } catch (err) {
    warnings.push(`${runtimeDir}: runtime root could not be listed; scan skipped (${String(err)})`);
    return { files, records, pruned, warnings };
  }

  const kinds = statJsonlEntries(runtimeDir, entries);
  const liveFiles = liveSourceNames(kinds);
  const st = prepare(dash);
  // Process keys whose data rows this pass deleted, and keys a record of this
  // pass wrote: a proc header alone writes no data row, so its process must
  // survive the prune even when its file was replaced.
  const removedKeys = new Set<string>();
  const writtenKeys = new Set<string>();
  const marks = new Map<string, JsonlMark>();
  for (const row of db.select().from(runtimeFiles).all()) {
    if (!liveFiles.has(row.name)) {
      deleteFileRows(dash, st, row.name, removedKeys);
      db.delete(runtimeFiles).where(eq(runtimeFiles.name, row.name)).run();
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
    if (stat === null || !stat.isFile()) continue;
    const filePath = join(runtimeDir, name);
    if (stat.mtimeMs < cutoff) {
      try {
        unlinkSync(filePath);
      } catch {
        continue;
      }
      pruned++;
      deleteFileRows(dash, st, name, removedKeys);
      db.delete(runtimeFiles).where(eq(runtimeFiles.name, name)).run();
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
    sqlite.exec('BEGIN');
    let fileRecords = 0;
    try {
      if (increment.mode === 'replace') deleteFileRows(dash, st, name, removedKeys);
      for (const line of increment.lines) {
        const rec = parseJsonObjectLine(line);
        const procKey = rec === null ? null : ingestRecord(st, rec, name);
        if (procKey === null) continue;
        fileRecords++;
        writtenKeys.add(procKey);
      }
      const { mark } = increment;
      const set = {
        lastIngestedMtime: mark.mtime,
        lastIngestedSize: mark.size,
        consumed: mark.consumed,
        tail: mark.tail,
      };
      db.insert(runtimeFiles)
        .values({ name, ...set })
        .onConflictDoUpdate({ target: runtimeFiles.name, set })
        .run();
      sqlite.exec('COMMIT');
      records += fileRecords;
    } catch {
      sqlite.exec('ROLLBACK');
      continue;
    }
  }

  for (const procKey of removedKeys) {
    if (!writtenKeys.has(procKey)) st.pruneProc.run(procKey);
  }
  return { files, records, pruned, warnings };
}
