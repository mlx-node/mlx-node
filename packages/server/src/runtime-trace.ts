/**
 * Runtime time series of the inference process, appended as JSON Lines to
 * `$HOME/.mlx-node/metrics/runtime/<YYYY-MM-DD>-<pid>.jsonl`, plus the record
 * types of the `live-<pid>.json` heartbeat. Read by the Control Panel ingest
 * (`packages/dashboard/src/ingest/runtime.ts`) and live watcher.
 *
 * Every field is a number, a closed enum or an identifier. The sink is fully
 * async and never throws: telemetry must never stall or break inference.
 */

import { appendFile, mkdir, stat } from 'node:fs/promises';
import { join } from 'node:path';

import { resolveMlxNodeHome } from './host/paths.js';
import { metricsEnvDisabled } from './metrics-trace.js';

export type HostKind = 'serve' | 'launch' | 'desktop' | 'embedded';
export const HOST_KINDS: readonly HostKind[] = ['serve', 'launch', 'desktop', 'embedded'];

export type RuntimeHealth = 'ok' | 'loading' | 'degraded' | 'error';
export type RuntimeCapability = 'chat' | 'audio' | 'rerank';

/** Per file; beyond it only `load` and `exit` records are appended. */
export const RUNTIME_FILE_MAX_BYTES = 16 * 1024 * 1024;
/** Pending appends; the oldest `sample` is dropped first. */
export const SINK_QUEUE_MAX = 256;
/** Days a runtime series file is kept. */
export const RUNTIME_RETENTION_DAYS = 14;
/** One record is one line shorter than this. */
const LINE_MAX_BYTES = 4096;
/** `close()` waits at most this long for the queue to drain. */
const CLOSE_DRAIN_MS = 1000;

/** Directory holding `<YYYY-MM-DD>-<pid>.jsonl` series and `live-<pid>.json` heartbeats. */
export function metricsRuntimeDir(): string {
  return join(resolveMlxNodeHome(), 'metrics', 'runtime');
}

interface RuntimeBase {
  v: 1;
  ts: number;
  pid: number;
  /** Process start (ms since epoch). `${startedAt}-${pid}` is the process key. */
  startedAt: number;
}

/** First line of every file. */
export interface RuntimeProcRecord extends RuntimeBase {
  k: 'proc';
  host: HostKind;
  /** `@mlx-node/server` package version. */
  version: string;
  port: number;
  wiredLimitBytes: number;
  totalMemBytes: number;
  /** `os.cpus()[0].model`, e.g. `Apple M5 Max`. */
  chip: string;
}

export interface RuntimeSampleRecord extends RuntimeBase {
  k: 'sample';
  /** Milliseconds since this process's previous sample (0 for the first). */
  dtMs: number;
  /** Busy state at the moment of the sample. */
  busy: boolean;
  /** Milliseconds the process was busy since its previous sample, measured from busy edges. */
  busyMs: number;
  health: RuntimeHealth;
  memActive: number;
  memPeak: number;
  memCache: number;
  /** `process.memoryUsage.rss()`. */
  rss: number;
  /** Turns in native flight, all models. */
  inFlight: number;
  /** Requests parked in admission lanes, all models. */
  queued: number;
  /** Distinct resident model objects. */
  resident: number;
  dOk: number;
  dError: number;
  dCancelled: number;
  dOutTokens: number;
  /** Requests answered with HTTP 429 since the previous sample. */
  dRejected: number;
  /** Records the sink dropped since the previous sample. Omitted when 0. */
  dropped?: number;
}

/** One idle snapshot of a model's scheduler. Counters are cumulative for the model's lifetime. */
export interface RuntimeSchedRecord extends RuntimeBase {
  k: 'sched';
  model: string;
  /** Turns sent to this model object since its load, read when the stats call was sent. All lanes, not only scheduled. */
  turns: number;
  blockCapacity: number;
  allocatedBlocks: number;
  freeBlocks: number;
  reclaimableBlocks: number;
  watermarkBlocks: number;
  reservedBlocks: number;
  memoryCapacityBytes: number;
  memoryWatermarkBytes: number;
  reservedBlockBytes: number;
  reservedStateBytes: number;
  globalSteps: number;
  admitted: number;
  completed: number;
  maxBatchOccupancy: number;
  fusedGreedyEpilogueSteps: number;
  admissionDeferredBlocks: number;
  admissionDeferredState: number;
  rowsAllocEvicted: number;
  preemptions: number;
  preemptionsRecompute: number;
  preemptionsSsd: number;
  ssdRestoreWaiting: number;
  ssdRestoreBytes: number;
  ssdRestoreWaitMs: number;
  /** `[occupancy, steps]` pairs. */
  occHist: Array<[number, number]>;
}

export interface RuntimeLoadRecord extends RuntimeBase {
  k: 'load';
  model: string;
  modelType: string;
  capability: RuntimeCapability;
  ms: number;
  ok: boolean;
}

/** Clean shutdown only. A killed process leaves none. */
export interface RuntimeExitRecord extends RuntimeBase {
  k: 'exit';
  uptimeMs: number;
}

export type RuntimeTraceRecord =
  | RuntimeProcRecord
  | RuntimeSampleRecord
  | RuntimeSchedRecord
  | RuntimeLoadRecord
  | RuntimeExitRecord;

export interface RuntimeLiveModel {
  name: string;
  modelType: string;
  ctxWindow: number;
  paged: boolean;
  maxConcurrent: number;
  loadedAt: number | null;
  loadMs: number | null;
  inFlight: number;
  queued: number;
  queueLimit: number | null;
  /** Last idle scheduler snapshot; null for models without a scheduler or before the first one. */
  pool: {
    at: number;
    blockCapacity: number;
    allocatedBlocks: number;
    freeBlocks: number;
    reclaimableBlocks: number;
  } | null;
}

export interface RuntimeLiveTurn {
  traceId: string;
  model: string;
  modelId: string | null;
  ctxWindow: number | null;
  startedAt: number;
  stream: boolean;
  dispatch: 'start' | 'continue' | 'continue_tool';
  phase: 'queued' | 'prefill' | 'decode' | null;
  promptTokens: number | null;
  cachedTokens: number | null;
  prefillTotal: number | null;
  prefillDone: number | null;
  generatedTokens: number | null;
}

/** `live-<pid>.json`: replaced atomically (tmp + rename). */
export interface RuntimeLiveFile {
  v: 1;
  pid: number;
  startedAt: number;
  ts: number;
  /** The writer promises the next write within this many ms. */
  intervalMs: number;
  host: HostKind;
  version: string;
  port: number;
  /** Whether this engine's model loads see the paged KV disk cache switched on. */
  persistPagedCache: boolean;
  /** The per-turn capture limits the paged loads read; null when the disk cache is off. */
  coldCapture: { blocks: number; ms: number } | null;
  health: RuntimeHealth;
  mem: { active: number; peak: number; cache: number; wiredLimit: number; rss: number };
  /** Output tokens per second across all in-flight turns since the previous write; null when idle. */
  decodeTps: number | null;
  loading: { model: string; startedAt: number } | null;
  counters: { ok: number; error: number; cancelled: number; outputTokens: number; loads: number; rejected: number };
  models: RuntimeLiveModel[];
  /** At most 32, oldest first. */
  turns: RuntimeLiveTurn[];
}

export interface RuntimeTraceOptions {
  dir?: string;
  /** The `proc` record each new day file starts with, stamped with the write time. */
  proc: Omit<RuntimeProcRecord, 'ts'>;
}

/** UTC day of an epoch-ms instant, `YYYY-MM-DD`. */
function utcDay(ts: number): string {
  return new Date(ts).toISOString().slice(0, 10);
}

/**
 * Async append-only sink: one `appendFile` in flight, FIFO queue capped at
 * {@link SINK_QUEUE_MAX}, day rotation by the record's UTC date, never throws.
 */
export class RuntimeTrace {
  readonly enabled: boolean;
  private readonly dir: string;
  private readonly proc: Omit<RuntimeProcRecord, 'ts'>;
  private readonly queue: RuntimeTraceRecord[] = [];
  private draining: Promise<void> | null = null;
  private dropped = 0;
  private closed = false;
  private file: string | null = null;
  private fileBytes = 0;
  private headerFile: string | null = null;
  private dirReady = false;
  /** The open file may end in a torn line: the next append starts with a line break. */
  private lead = false;

  constructor(opts: RuntimeTraceOptions) {
    this.enabled = !metricsEnvDisabled();
    this.dir = opts.dir ?? metricsRuntimeDir();
    this.proc = opts.proc;
  }

  fileFor(ts: number): string {
    return join(this.dir, `${utcDay(ts)}-${this.proc.pid}.jsonl`);
  }

  /** Queue one record. Never blocks, never throws. */
  append(rec: RuntimeTraceRecord): void {
    if (!this.enabled || this.closed) return;
    this.queue.push(rec);
    if (this.queue.length > SINK_QUEUE_MAX) {
      let victim = -1;
      for (let i = 0; i < this.queue.length; i++) {
        if (this.queue[i]!.k === 'sample') {
          victim = i;
          break;
        }
      }
      this.queue.splice(victim === -1 ? 0 : victim, 1);
      this.dropped += 1;
    }
    this.draining ??= this.drain();
  }

  /** Records dropped since the last call; resets the count. */
  takeDropped(): number {
    const count = this.dropped;
    this.dropped = 0;
    return count;
  }

  /** Drain the queue (at most {@link CLOSE_DRAIN_MS}), then accept nothing more. */
  async close(): Promise<void> {
    if (this.closed) return;
    const pending = this.draining;
    if (pending !== null) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      await Promise.race([
        pending,
        new Promise<void>((resolve) => {
          timer = setTimeout(resolve, CLOSE_DRAIN_MS);
        }),
      ]);
      if (timer !== undefined) clearTimeout(timer);
    }
    this.closed = true;
    this.queue.length = 0;
  }

  private async drain(): Promise<void> {
    try {
      while (this.queue.length > 0) {
        const rec = this.queue.shift()!;
        try {
          await this.write(rec);
        } catch {
          // A failed append is lost; the next one re-creates the directory and re-stats the file.
          this.dirReady = false;
          this.file = null;
        }
      }
    } finally {
      this.draining = null;
    }
  }

  private async write(rec: RuntimeTraceRecord): Promise<void> {
    const file = this.fileFor(rec.ts);
    if (file !== this.file) {
      if (!this.dirReady) {
        await mkdir(this.dir, { recursive: true });
        this.dirReady = true;
      }
      this.file = file;
      this.fileBytes = await stat(file).then(
        (s) => s.size,
        () => 0,
      );
      this.lead = this.fileBytes > 0;
    }
    if (this.fileBytes >= RUNTIME_FILE_MAX_BYTES && rec.k !== 'load' && rec.k !== 'exit') return;
    if (this.headerFile !== file && rec.k !== 'proc') {
      // An earlier process with the same pid may own this day file: every sink writes its own header.
      this.headerFile = file;
      try {
        await this.appendLine(file, { ...this.proc, ts: rec.ts });
      } catch (err) {
        this.headerFile = null;
        throw err;
      }
    }
    await this.appendLine(file, rec);
    if (rec.k === 'proc') this.headerFile = file;
  }

  private async appendLine(file: string, rec: RuntimeTraceRecord): Promise<void> {
    const json = `${JSON.stringify(rec)}\n`;
    if (Buffer.byteLength(json) >= LINE_MAX_BYTES) return;
    if (!this.dirReady) {
      await mkdir(this.dir, { recursive: true });
      this.dirReady = true;
    }
    const line = this.lead ? `\n${json}` : json;
    await appendFile(file, line);
    this.lead = false;
    this.fileBytes += Buffer.byteLength(line);
  }
}
