/**
 * Live engine state for the Control Panel: the inference processes'
 * `live-<pid>.json` heartbeats plus the desktop supervisor file, read on the
 * transport thread with async fs only.
 *
 * The watcher stats the files once per `pollMs` ONLY while someone is
 * subscribed. `fs.watch` is not used: the heartbeat is replaced by rename,
 * which drops a per-file watch. A finished turn moves the writer's counters,
 * which triggers one coalesced ingest and a `data` event.
 *
 * Never import `@mlx-node/server` here: its host entry links the native addon,
 * and this process must stay addon-free (`packages/desktop/__test__/child-entries.test.ts`).
 */

import { readdir, readFile, stat, unlink } from 'node:fs/promises';
import { join } from 'node:path';

import type { IngestSummary } from './api/context.js';

export type HostKind = 'serve' | 'launch' | 'desktop' | 'embedded';
export type EngineStatus =
  | 'running'
  | 'loading'
  | 'starting'
  | 'restarting'
  | 'failed'
  | 'untrusted'
  | 'unresponsive'
  | 'stopped';
export type ProcessStatus = 'running' | 'unresponsive' | 'dead';
export type RuntimeHealth = 'ok' | 'loading' | 'degraded' | 'error';
export type SupervisorState = 'stopped' | 'starting' | 'running' | 'restarting' | 'failed' | 'lying';
export type LiveTurnPhase = 'queued' | 'prefill' | 'decode';
export type LiveDispatch = 'start' | 'continue' | 'continue_tool';

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
  /** Alias the client used. Display and color use `modelId ?? model`. */
  model: string;
  modelId: string | null;
  ctxWindow: number | null;
  startedAt: number;
  stream: boolean;
  dispatch: LiveDispatch;
  phase: LiveTurnPhase | null;
  promptTokens: number | null;
  cachedTokens: number | null;
  prefillTotal: number | null;
  prefillDone: number | null;
  generatedTokens: number | null;
}

/** The validated content of one `live-<pid>.json`. */
export interface RuntimeLiveFile {
  pid: number;
  startedAt: number;
  ts: number;
  intervalMs: number;
  host: HostKind;
  version: string;
  port: number;
  /** Whether the engine's model loads see the paged KV disk cache on; null from an engine that does not say. */
  persistPagedCache: boolean | null;
  /** The per-turn capture limits of the disk cache; null when it is off or the engine does not say. */
  coldCapture: { blocks: number; ms: number } | null;
  health: RuntimeHealth;
  mem: { active: number; peak: number; cache: number; wiredLimit: number; rss: number };
  decodeTps: number | null;
  loading: { model: string; startedAt: number } | null;
  counters: { ok: number; error: number; cancelled: number; outputTokens: number; loads: number; rejected: number };
  models: RuntimeLiveModel[];
  turns: RuntimeLiveTurn[];
}

export interface RuntimeProcessLive extends RuntimeLiveFile {
  status: ProcessStatus;
  /** now − ts. */
  ageMs: number;
}

export interface SupervisorLive {
  state: SupervisorState;
  pid: number | null;
  generation: number;
  streak: number;
  shortLivedFailures: number;
  lastExit: { verdict: 'clean' | 'crash'; code: number | null; signal: string | null; atMs: number } | null;
  healthStatus: RuntimeHealth | null;
  nativeErrorCount: number;
  /** The count is at the supervisor's cap of 20; print "20+". */
  nativeErrorsCapped: boolean;
  lastNativeErrorContext: string | null;
  lastNativeErrorAt: number | null;
  writtenAt: number;
}

export interface RuntimeLive {
  now: number;
  status: EngineStatus;
  /** Desktop-managed sidecar; null when not run by the desktop app. */
  supervisor: SupervisorLive | null;
  /** Running first, then unresponsive, then dead; newest `startedAt` first within each. */
  processes: RuntimeProcessLive[];
}

export type LiveTopic = 'live';

export type LiveEvent =
  | { type: 'runtime'; snapshot: RuntimeLive }
  | { type: 'data'; version: number; traces: boolean; runtime: boolean };

/** A stale file is dead after this long, whatever its pid says: macOS reuses pids. */
export const LIVE_DEAD_AFTER_MS = 300_000;
/** A dead writer stays listed this long before its file is unlinked. */
export const DEAD_LISTED_MS = 60_000;
/** Slack over three missed heartbeats before a writer counts as unresponsive. */
export const STALE_GRACE_MS = 1500;
export const LIVE_POLL_MS = 1000;
/** Two ingests started by the watcher are at least this far apart. */
export const INGEST_COALESCE_MS = 2000;
const LIVE_TURNS_MAX = 32;
const LIVE_MODELS_MAX = 64;

// ---------------------------------------------------------------- parsing

type Obj = Record<string, unknown>;

function isObj(value: unknown): value is Obj {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function fin(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function finOrNull(value: unknown): number | null | undefined {
  if (value === null) return null;
  const n = fin(value);
  return n === null ? undefined : n;
}

function strOrNull(value: unknown): string | null | undefined {
  if (value === null) return null;
  return typeof value === 'string' ? value : undefined;
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[]): T | null {
  for (const candidate of allowed) if (candidate === value) return candidate;
  return null;
}

const HOSTS: readonly HostKind[] = ['serve', 'launch', 'desktop', 'embedded'];
const HEALTH: readonly RuntimeHealth[] = ['ok', 'loading', 'degraded', 'error'];
const STATES: readonly SupervisorState[] = ['stopped', 'starting', 'running', 'restarting', 'failed', 'lying'];
const PHASES: readonly LiveTurnPhase[] = ['queued', 'prefill', 'decode'];
const DISPATCHES: readonly LiveDispatch[] = ['start', 'continue', 'continue_tool'];

/** pid of a `live-<pid>.json` file name, or null. */
export function liveFilePid(name: string): number | null {
  if (!name.startsWith('live-') || !name.endsWith('.json')) return null;
  const digits = name.slice(5, name.length - 5);
  if (digits.length === 0 || digits.length > 10) return null;
  for (let i = 0; i < digits.length; i++) {
    const c = digits.charCodeAt(i);
    if (c < 48 || c > 57) return null;
  }
  const pid = Number(digits);
  return pid > 0 ? pid : null;
}

function parsePool(value: unknown): RuntimeLiveModel['pool'] | undefined {
  if (value === null) return null;
  if (!isObj(value)) return undefined;
  const at = fin(value.at);
  const blockCapacity = fin(value.blockCapacity);
  const allocatedBlocks = fin(value.allocatedBlocks);
  const freeBlocks = fin(value.freeBlocks);
  const reclaimableBlocks = fin(value.reclaimableBlocks);
  if (at === null || blockCapacity === null || allocatedBlocks === null || freeBlocks === null) return undefined;
  if (reclaimableBlocks === null) return undefined;
  return { at, blockCapacity, allocatedBlocks, freeBlocks, reclaimableBlocks };
}

function parseColdCapture(value: unknown): RuntimeLiveFile['coldCapture'] {
  if (!isObj(value)) return null;
  const blocks = fin(value.blocks);
  const ms = fin(value.ms);
  return blocks === null || ms === null ? null : { blocks, ms };
}

function parseModel(value: unknown): RuntimeLiveModel | null {
  if (!isObj(value) || typeof value.name !== 'string' || typeof value.modelType !== 'string') return null;
  const ctxWindow = fin(value.ctxWindow);
  const maxConcurrent = fin(value.maxConcurrent);
  const inFlight = fin(value.inFlight);
  const queued = fin(value.queued);
  const loadedAt = finOrNull(value.loadedAt);
  const loadMs = finOrNull(value.loadMs);
  const queueLimit = finOrNull(value.queueLimit);
  const pool = parsePool(value.pool);
  if (ctxWindow === null || maxConcurrent === null || inFlight === null || queued === null) return null;
  if (loadedAt === undefined || loadMs === undefined || queueLimit === undefined || pool === undefined) return null;
  if (typeof value.paged !== 'boolean') return null;
  return {
    name: value.name,
    modelType: value.modelType,
    ctxWindow,
    paged: value.paged,
    maxConcurrent,
    loadedAt,
    loadMs,
    inFlight,
    queued,
    queueLimit,
    pool,
  };
}

function parseTurn(value: unknown): RuntimeLiveTurn | null {
  if (!isObj(value) || typeof value.traceId !== 'string' || typeof value.model !== 'string') return null;
  const modelId = strOrNull(value.modelId);
  const ctxWindow = finOrNull(value.ctxWindow);
  const startedAt = fin(value.startedAt);
  const dispatch = oneOf(value.dispatch, DISPATCHES);
  const phase = value.phase === null ? null : oneOf(value.phase, PHASES);
  if (modelId === undefined || ctxWindow === undefined || startedAt === null || dispatch === null) return null;
  if (typeof value.stream !== 'boolean' || (value.phase !== null && phase === null)) return null;
  const counts = [
    value.promptTokens,
    value.cachedTokens,
    value.prefillTotal,
    value.prefillDone,
    value.generatedTokens,
  ].map(finOrNull);
  if (counts.some((c) => c === undefined)) return null;
  const [promptTokens, cachedTokens, prefillTotal, prefillDone, generatedTokens] = counts as Array<number | null>;
  return {
    traceId: value.traceId,
    model: value.model,
    modelId,
    ctxWindow,
    startedAt,
    stream: value.stream,
    dispatch,
    phase,
    promptTokens,
    cachedTokens,
    prefillTotal,
    prefillDone,
    generatedTokens,
  };
}

/** Field-by-field validation of a heartbeat; null for anything malformed. */
export function parseLiveFile(value: unknown): RuntimeLiveFile | null {
  if (!isObj(value) || value.v !== 1) return null;
  const pid = fin(value.pid);
  const startedAt = fin(value.startedAt);
  const ts = fin(value.ts);
  const intervalMs = fin(value.intervalMs);
  const port = fin(value.port);
  const host = oneOf(value.host, HOSTS);
  const health = oneOf(value.health, HEALTH);
  const decodeTps = finOrNull(value.decodeTps);
  if (pid === null || startedAt === null || ts === null || intervalMs === null || intervalMs <= 0) return null;
  if (port === null || host === null || health === null || decodeTps === undefined) return null;
  if (typeof value.version !== 'string') return null;
  if (!isObj(value.mem) || !isObj(value.counters)) return null;
  const memKeys = ['active', 'peak', 'cache', 'wiredLimit', 'rss'] as const;
  const mem = {} as RuntimeLiveFile['mem'];
  for (const key of memKeys) {
    const n = fin(value.mem[key]);
    if (n === null) return null;
    mem[key] = n;
  }
  const counterKeys = ['ok', 'error', 'cancelled', 'outputTokens', 'loads', 'rejected'] as const;
  const counters = {} as RuntimeLiveFile['counters'];
  for (const key of counterKeys) {
    const n = fin(value.counters[key]);
    if (n === null) return null;
    counters[key] = n;
  }
  let loading: RuntimeLiveFile['loading'] = null;
  if (value.loading !== null) {
    if (!isObj(value.loading) || typeof value.loading.model !== 'string') return null;
    const since = fin(value.loading.startedAt);
    if (since === null) return null;
    loading = { model: value.loading.model, startedAt: since };
  }
  if (!Array.isArray(value.models) || !Array.isArray(value.turns)) return null;
  const models: RuntimeLiveModel[] = [];
  for (const raw of value.models.slice(0, LIVE_MODELS_MAX)) {
    const model = parseModel(raw);
    if (model === null) return null;
    models.push(model);
  }
  const turns: RuntimeLiveTurn[] = [];
  for (const raw of value.turns.slice(0, LIVE_TURNS_MAX)) {
    const turn = parseTurn(raw);
    if (turn === null) return null;
    turns.push(turn);
  }
  return {
    pid,
    startedAt,
    ts,
    intervalMs,
    host,
    version: value.version,
    port,
    persistPagedCache: typeof value.persistPagedCache === 'boolean' ? value.persistPagedCache : null,
    coldCapture: parseColdCapture(value.coldCapture),
    health,
    mem,
    decodeTps,
    loading,
    counters,
    models,
    turns,
  };
}

/** The supervisor file, or null when malformed or when its main process is gone. */
export function parseSupervisorFile(value: unknown, isAlive: (pid: number) => boolean): SupervisorLive | null {
  if (!isObj(value) || value.version !== 1) return null;
  const mainPid = fin(value.mainPid);
  const writtenAt = fin(value.writtenAt);
  const state = oneOf(value.state, STATES);
  const pid = finOrNull(value.pid);
  const generation = fin(value.generation);
  const streak = fin(value.streak);
  const shortLivedFailures = fin(value.shortLivedFailures);
  const nativeErrorCount = fin(value.nativeErrorCount);
  const lastNativeErrorAt = finOrNull(value.lastNativeErrorAt);
  const lastNativeErrorContext = strOrNull(value.lastNativeErrorContext);
  const healthStatus = value.healthStatus === null ? null : oneOf(value.healthStatus, HEALTH);
  if (mainPid === null || writtenAt === null || state === null || pid === undefined || generation === null) return null;
  if (streak === null || shortLivedFailures === null) return null;
  if (nativeErrorCount === null || lastNativeErrorAt === undefined) return null;
  if (lastNativeErrorContext === undefined || typeof value.nativeErrorsCapped !== 'boolean') return null;
  if (value.healthStatus !== null && healthStatus === null) return null;
  let lastExit: SupervisorLive['lastExit'] = null;
  if (value.lastExit !== null) {
    const exit = value.lastExit;
    if (!isObj(exit)) return null;
    const verdict = oneOf(exit.verdict, ['clean', 'crash'] as const);
    const code = finOrNull(exit.code);
    const signal = strOrNull(exit.signal);
    const atMs = fin(exit.atMs);
    if (verdict === null || code === undefined || signal === undefined || atMs === null) return null;
    lastExit = { verdict, code, signal, atMs };
  }
  if (!isAlive(mainPid)) return null;
  return {
    state,
    pid,
    generation,
    streak,
    shortLivedFailures,
    lastExit,
    healthStatus,
    nativeErrorCount,
    nativeErrorsCapped: value.nativeErrorsCapped,
    lastNativeErrorContext,
    lastNativeErrorAt,
    writtenAt,
  };
}

// ---------------------------------------------------------------- judging

/** `process.kill(pid, 0)`; EPERM means the pid exists under another user. */
export function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === 'EPERM';
  }
}

export function classifyProcess(
  file: Pick<RuntimeLiveFile, 'pid' | 'ts' | 'intervalMs'>,
  now: number,
  isAlive: (pid: number) => boolean,
): ProcessStatus {
  const age = now - file.ts;
  if (age >= LIVE_DEAD_AFTER_MS || !isAlive(file.pid)) return 'dead';
  return age <= 3 * file.intervalMs + STALE_GRACE_MS ? 'running' : 'unresponsive';
}

export function deriveEngineStatus(
  supervisor: Pick<SupervisorLive, 'state'> | null,
  processes: ReadonlyArray<Pick<RuntimeProcessLive, 'status' | 'health'>>,
): EngineStatus {
  if (supervisor?.state === 'lying') return 'untrusted';
  if (processes.some((p) => p.status === 'running' && p.health === 'loading')) return 'loading';
  if (processes.some((p) => p.status === 'running')) return 'running';
  if (
    supervisor !== null &&
    (supervisor.state === 'starting' || supervisor.state === 'restarting' || supervisor.state === 'failed')
  ) {
    return supervisor.state;
  }
  if (processes.some((p) => p.status === 'unresponsive')) return 'unresponsive';
  return 'stopped';
}

const STATUS_ORDER: Record<ProcessStatus, number> = { running: 0, unresponsive: 1, dead: 2 };

// ---------------------------------------------------------------- watcher

export interface LiveWatcherOptions {
  runtimeDir: string;
  supervisorPath: string;
  /** Run one ingest; resolves with what it wrote. */
  ingest: () => Promise<IngestSummary>;
  now?: () => number;
  isAlive?: (pid: number) => boolean;
  pollMs?: number;
}

export interface LiveWatcher {
  subscribe(listener: (event: LiveEvent) => void): () => void;
  /** One read outside the loop, for `GET /api/runtime/live`. */
  readNow(): Promise<RuntimeLive>;
  /** Report an ingest run by someone else (the periodic rescan). */
  notifyIngested(summary: IngestSummary): void;
  close(): void;
}

interface CachedFile {
  mtimeMs: number;
  size: number;
  file: RuntimeLiveFile | null;
}

export function createLiveWatcher(opts: LiveWatcherOptions): LiveWatcher {
  const now = opts.now ?? Date.now;
  const isAlive = opts.isAlive ?? pidAlive;
  const pollMs = opts.pollMs ?? LIVE_POLL_MS;
  const listeners = new Set<(event: LiveEvent) => void>();
  const files = new Map<string, CachedFile>();
  const deadSince = new Map<string, number>();
  let supervisorCache: { mtimeMs: number; size: number; raw: unknown } | null = null;

  let closed = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let ticking = false;
  let latest: RuntimeLive | null = null;
  let latestKey: string | null = null;
  let lastSum: number | null = null;
  let lastProcs: string | null = null;

  let version = 0;
  let ingestRunning = false;
  let ingestPending = false;
  let ingestTimer: ReturnType<typeof setTimeout> | null = null;
  let lastIngestStart = Number.NEGATIVE_INFINITY;

  const emit = (event: LiveEvent): void => {
    for (const listener of Array.from(listeners)) {
      try {
        listener(event);
      } catch {
        // A listener fault must not stop delivery to the others.
      }
    }
  };

  const reportIngest = (summary: IngestSummary): void => {
    const traces = summary.traces.records > 0;
    const runtime = summary.runtime.records > 0;
    if (!traces && !runtime) return;
    version += 1;
    emit({ type: 'data', version, traces, runtime });
  };

  const runIngest = (): void => {
    ingestTimer = null;
    if (closed) return;
    if (ingestRunning) {
      ingestPending = true;
      return;
    }
    ingestRunning = true;
    lastIngestStart = now();
    void opts
      .ingest()
      .then(reportIngest, () => undefined)
      .finally(() => {
        ingestRunning = false;
        if (ingestPending) {
          ingestPending = false;
          requestIngest();
        }
      });
  };

  const requestIngest = (): void => {
    if (closed || ingestTimer !== null) return;
    if (ingestRunning) {
      ingestPending = true;
      return;
    }
    const wait = Math.max(0, lastIngestStart + INGEST_COALESCE_MS - now());
    ingestTimer = setTimeout(runIngest, wait);
    ingestTimer.unref?.();
  };

  async function readLiveFiles(): Promise<void> {
    let names: string[];
    try {
      names = await readdir(opts.runtimeDir);
    } catch (err) {
      // A missing directory means no writer; any other fault keeps what we had.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') files.clear();
      return;
    }
    const present = new Set<string>();
    for (const name of names) {
      if (liveFilePid(name) === null) continue;
      present.add(name);
      const path = join(opts.runtimeDir, name);
      let st;
      try {
        st = await stat(path);
      } catch {
        continue;
      }
      if (!st.isFile()) continue;
      const cached = files.get(name);
      if (cached !== undefined && cached.mtimeMs === st.mtimeMs && cached.size === st.size) continue;
      let parsed: RuntimeLiveFile | null = null;
      try {
        parsed = parseLiveFile(JSON.parse(await readFile(path, 'utf8')));
      } catch {
        parsed = null;
      }
      // An unparsable read keeps the previous content for this tick; it is never "gone".
      if (parsed === null) {
        if (cached === undefined) files.set(name, { mtimeMs: Number.NaN, size: -1, file: null });
        continue;
      }
      files.set(name, { mtimeMs: st.mtimeMs, size: st.size, file: parsed });
    }
    for (const name of files.keys()) {
      if (!present.has(name)) {
        files.delete(name);
        deadSince.delete(name);
      }
    }
  }

  async function readSupervisor(): Promise<SupervisorLive | null> {
    try {
      const st = await stat(opts.supervisorPath);
      if (supervisorCache === null || supervisorCache.mtimeMs !== st.mtimeMs || supervisorCache.size !== st.size) {
        let raw: unknown = null;
        try {
          raw = JSON.parse(await readFile(opts.supervisorPath, 'utf8'));
        } catch {
          raw = null;
        }
        supervisorCache = { mtimeMs: st.mtimeMs, size: st.size, raw };
      }
    } catch {
      supervisorCache = null;
    }
    return supervisorCache === null ? null : parseSupervisorFile(supervisorCache.raw, isAlive);
  }

  async function collect(): Promise<RuntimeLive> {
    await readLiveFiles();
    const supervisor = await readSupervisor();
    const at = now();
    const processes: RuntimeProcessLive[] = [];
    for (const [name, cached] of files) {
      if (cached.file === null) continue;
      const status = classifyProcess(cached.file, at, isAlive);
      if (status === 'dead') {
        const since = deadSince.get(name);
        if (since === undefined) deadSince.set(name, at);
        else if (at - since >= DEAD_LISTED_MS) {
          files.delete(name);
          deadSince.delete(name);
          await unlink(join(opts.runtimeDir, name)).catch(() => undefined);
          continue;
        }
      } else {
        deadSince.delete(name);
      }
      processes.push({ ...cached.file, status, ageMs: at - cached.file.ts });
    }
    processes.sort((a, b) => STATUS_ORDER[a.status] - STATUS_ORDER[b.status] || b.startedAt - a.startedAt);
    return { now: at, status: deriveEngineStatus(supervisor, processes), supervisor, processes };
  }

  /** Everything that matters to a viewer except the clock. */
  function keyOf(snapshot: RuntimeLive): string {
    return JSON.stringify({
      status: snapshot.status,
      supervisor: snapshot.supervisor,
      processes: snapshot.processes.map(({ ageMs: _ageMs, ...rest }) => rest),
    });
  }

  async function tick(): Promise<void> {
    if (ticking || closed) return;
    ticking = true;
    try {
      const snapshot = await collect();
      if (closed) return;
      latest = snapshot;
      const key = keyOf(snapshot);
      if (key !== latestKey) {
        latestKey = key;
        emit({ type: 'runtime', snapshot });
      }
      let sum = 0;
      for (const p of snapshot.processes)
        sum += p.counters.ok + p.counters.error + p.counters.cancelled + p.counters.loads;
      const procs = snapshot.processes.map((p) => `${p.pid}:${p.startedAt}:${p.status}`).join(',');
      if (lastSum !== null && (sum !== lastSum || procs !== lastProcs)) requestIngest();
      lastSum = sum;
      lastProcs = procs;
    } finally {
      ticking = false;
    }
  }

  const schedule = (delay: number): void => {
    if (closed || listeners.size === 0 || timer !== null || ticking) return;
    timer = setTimeout(() => {
      timer = null;
      void tick().finally(() => schedule(pollMs));
    }, delay);
    timer.unref?.();
  };

  const stop = (): void => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    latestKey = null;
  };

  return {
    subscribe(listener) {
      if (closed) return () => {};
      // `latest` outlives the last unsubscribe; an old one must not be replayed, and the next tick must then emit.
      if (latest !== null && now() - latest.now <= 2 * pollMs) listener({ type: 'runtime', snapshot: latest });
      else latestKey = null;
      listeners.add(listener);
      schedule(0);
      return () => {
        listeners.delete(listener);
        if (listeners.size === 0) stop();
      };
    },
    async readNow() {
      const snapshot = await collect();
      latest = snapshot;
      return snapshot;
    },
    notifyIngested(summary) {
      if (!closed) reportIngest(summary);
    },
    close() {
      closed = true;
      stop();
      listeners.clear();
      if (ingestTimer !== null) clearTimeout(ingestTimer);
      ingestTimer = null;
    },
  };
}
