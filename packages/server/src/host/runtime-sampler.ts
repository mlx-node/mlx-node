/**
 * The inference process's own state sampler: a `live-<pid>.json` heartbeat
 * (1 s busy / 5 s idle) and a runtime series (`sample` every 5 s busy / 60 s
 * idle, `sched` idle snapshots, `load`, `exit`) under
 * `~/.mlx-node/metrics/runtime/`.
 *
 * `schedulerStats()` is a scheduler BARRIER: while it is queued the model
 * thread admits no new turn until the running batch drains. It is therefore
 * only ever called on a model that is idle ({@link shouldSnapshotScheduler}).
 * Live concurrency comes from TS-side state instead. No sync fs here.
 */

import { readFile, readdir, mkdir, rename, unlink, writeFile } from 'node:fs/promises';
import { cpus, totalmem } from 'node:os';
import { join } from 'node:path';

import { memoryStats, type SchedulerStats } from '@mlx-node/core';

import pkgJson from '../../package.json' with { type: 'json' };
import type { ServerHealthStatus } from '../health.js';
import { metricsEnvDisabled } from '../metrics-trace.js';
import { queueRejections } from '../queue-rejections.js';
import type { ResidentModel } from '../registry.js';
import {
  metricsRuntimeDir,
  RUNTIME_RETENTION_DAYS,
  RuntimeTrace,
  type HostKind,
  type RuntimeCapability,
  type RuntimeLiveFile,
  type RuntimeLiveModel,
  type RuntimeLiveTurn,
  type RuntimeSampleRecord,
  type RuntimeSchedRecord,
} from '../runtime-trace.js';
import { setTurnEdgeListener, turnCounters, type LiveTurn, type TurnCounters } from '../turn-trace.js';
import type { ColdCaptureBudget } from './persist-cache.js';
import { isProcessAlive } from './temp-root.js';

export const HEARTBEAT_BUSY_MS = 1000;
export const HEARTBEAT_IDLE_MS = 5000;
export const SAMPLE_BUSY_MS = 5000;
export const SAMPLE_IDLE_MS = 60_000;
export const EDGE_COALESCE_MS = 250;
export const SCHED_IDLE_SETTLE_MS = 500;
export const LIVE_TURNS_MAX = 32;
export const LIVE_DEAD_AFTER_MS = 300_000;

const DAY_MS = 86_400_000;

export interface RuntimeSamplerDeps {
  registry: { listResidentModels(): ResidentModel[] };
  health: () => { status: ServerHealthStatus };
  /** Discovered inventory: name → family, window. */
  models: ReadonlyArray<{ name: string; modelType: string; contextWindow: number }>;
  host: { kind: HostKind; port: number; persistPagedCache: boolean; coldCapture: ColdCaptureBudget | null };
  trace?: RuntimeTrace;
  now?: () => number;
  memory?: () => { activeBytes: number; peakBytes: number; cacheBytes: number };
  wiredLimit?: () => number;
  /** Runtime directory; defaults to {@link metricsRuntimeDir}. */
  dir?: string;
  /** Liveness probe for the start-up sweep. */
  isAlive?: (pid: number) => boolean;
}

export interface RuntimeSampler {
  /** Wrap a real load. Records one `load` record and the live `loading` field. */
  timedLoad<T>(
    info: { model: string; modelType: string; capability: RuntimeCapability },
    load: () => Promise<T>,
  ): Promise<T>;
  /** Final sample + `exit` record, flush, remove the live file. Idempotent. */
  stop(): Promise<void>;
}

/** The idle-only rule for `schedulerStats()`. True only for a model with no work at all. */
export function shouldSnapshotScheduler(s: {
  now: number;
  activeDispatches: number;
  queueDepth: number;
  live: number;
  begun: number;
  lastSettleAt: number;
  pending: boolean;
  /** `begun` at the last snapshot of this model object; undefined before the first one. */
  snapshotBegun: number | undefined;
}): boolean {
  if (s.activeDispatches !== 0 || s.queueDepth !== 0 || s.live !== 0) return false;
  if (s.now - s.lastSettleAt < SCHED_IDLE_SETTLE_MS) return false;
  if (s.pending) return false;
  return s.snapshotBegun === undefined || s.snapshotBegun !== s.begun;
}

const INERT: RuntimeSampler = {
  timedLoad: (_info, load) => load(),
  stop: async () => {},
};

let running = false;

function safeMemory(): { activeBytes: number; peakBytes: number; cacheBytes: number } {
  try {
    const m = memoryStats();
    return { activeBytes: m.active, peakBytes: m.peak, cacheBytes: m.cache };
  } catch {
    return { activeBytes: 0, peakBytes: 0, cacheBytes: 0 };
  }
}

function safeWiredLimit(): number {
  try {
    return memoryStats().wiredLimit;
  } catch {
    return 0;
  }
}

function finiteOr0(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function liveColdCapture(budget: ColdCaptureBudget | null): RuntimeLiveFile['coldCapture'] {
  return budget === null ? null : { blocks: budget.blocksPerTurn, ms: budget.budgetMs };
}

function allDigits(text: string): boolean {
  if (text.length === 0) return false;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 48 || c > 57) return false;
  }
  return true;
}

/** pid of a `live-<pid>.json` name, or null. */
export function liveFilePid(name: string): number | null {
  if (!name.startsWith('live-') || !name.endsWith('.json')) return null;
  const digits = name.slice(5, name.length - 5);
  return allDigits(digits) ? Number(digits) : null;
}

/** `YYYY-MM-DD` date of a `<YYYY-MM-DD>-<pid>.jsonl` name, or null. */
function seriesFileDay(name: string): string | null {
  if (!name.endsWith('.jsonl') || name.length < 17) return null;
  const day = name.slice(0, 10);
  if (day.charCodeAt(4) !== 45 || day.charCodeAt(7) !== 45 || name.charCodeAt(10) !== 45) return null;
  if (!allDigits(day.slice(0, 4)) || !allDigits(day.slice(5, 7)) || !allDigits(day.slice(8, 10))) return null;
  if (!allDigits(name.slice(11, name.length - 6))) return null;
  return day;
}

/** Remove live files (and their `.tmp`) of dead writers and series files past retention. Best effort. */
async function sweepRuntimeDir(dir: string, now: number, isAlive: (pid: number) => boolean): Promise<void> {
  let names: string[];
  try {
    names = await readdir(dir);
  } catch {
    return;
  }
  const cutoff = new Date(now - RUNTIME_RETENTION_DAYS * DAY_MS).toISOString().slice(0, 10);
  await Promise.all(
    names.map(async (name) => {
      try {
        if (name.endsWith('.tmp')) {
          const tmpPid = liveFilePid(name.slice(0, name.length - 4));
          if (tmpPid !== null && tmpPid !== process.pid && !isAlive(tmpPid)) await unlink(join(dir, name));
          return;
        }
        const pid = liveFilePid(name);
        if (pid !== null) {
          if (pid === process.pid) return;
          let stale = !isAlive(pid);
          if (!stale) {
            const parsed = JSON.parse(await readFile(join(dir, name), 'utf8')) as { ts?: unknown };
            stale = typeof parsed.ts === 'number' && now - parsed.ts >= LIVE_DEAD_AFTER_MS;
          }
          if (stale) await unlink(join(dir, name));
          return;
        }
        const day = seriesFileDay(name);
        if (day !== null && day < cutoff) await unlink(join(dir, name));
      } catch {
        // Best effort: an unreadable file is left for the next sweep.
      }
    }),
  );
}

interface SchedState {
  pending: boolean;
  snapshotBegun: number | undefined;
  pool: RuntimeLiveModel['pool'];
}

interface ModelFacts {
  ctxWindow: number | undefined;
  paged: boolean;
}

interface BindingView {
  entry: ResidentModel;
  name: string;
  turns: LiveTurn[];
  stats: { live: number; begun: number; lastSettleAt: number };
  active: number;
  queued: number;
}

class Sampler implements RuntimeSampler {
  private readonly now: () => number;
  private readonly trace: RuntimeTrace;
  private readonly dir: string;
  private readonly startedAt: number;
  private readonly memory: () => { activeBytes: number; peakBytes: number; cacheBytes: number };
  private readonly wiredLimit: number;
  private readonly discovered: Map<string, { modelType: string; contextWindow: number }>;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private nextTickAt = 0;
  private stopped = false;
  private stopPromise: Promise<void> | null = null;

  private busy = false;
  private busySince: number | null = null;
  private busyAccumMs = 0;

  private lastSampleAt: number | null = null;
  private lastSampleBusy = false;
  private lastResidentKey = '';
  private prevCounters: TurnCounters;
  private prevRejected: number;

  private loadsActive = 0;
  private loading: { model: string; startedAt: number } | null = null;
  private loads = 0;
  private readonly loaded = new Map<string, { loadedAt: number; loadMs: number }>();

  private readonly sched = new WeakMap<object, SchedState>();
  private readonly facts = new WeakMap<object, ModelFacts>();
  private readonly modelIds = new WeakMap<object, number>();
  private nextModelId = 1;

  private liveWrite: Promise<void> | null = null;
  private liveDirReady = false;
  private lastLiveAt: number | null = null;
  private lastLiveTokens = 0;

  constructor(private readonly deps: RuntimeSamplerDeps) {
    this.now = deps.now ?? Date.now;
    this.startedAt = this.now();
    this.dir = deps.dir ?? metricsRuntimeDir();
    this.memory = deps.memory ?? safeMemory;
    this.wiredLimit = finiteOr0((deps.wiredLimit ?? safeWiredLimit)());
    this.discovered = new Map(
      deps.models.map((m) => [m.name, { modelType: m.modelType, contextWindow: m.contextWindow }]),
    );
    this.trace =
      deps.trace ??
      new RuntimeTrace({
        dir: this.dir,
        proc: {
          v: 1,
          k: 'proc',
          pid: process.pid,
          startedAt: this.startedAt,
          host: deps.host.kind,
          version: String(pkgJson.version),
          port: deps.host.port,
          wiredLimitBytes: this.wiredLimit,
          totalMemBytes: totalmem(),
          chip: (cpus()[0]?.model ?? '').trim(),
        },
      });
    this.prevCounters = turnCounters();
    this.prevRejected = queueRejections();
  }

  start(): void {
    void sweepRuntimeDir(this.dir, this.now(), this.deps.isAlive ?? isProcessAlive);
    setTurnEdgeListener(() => this.onEdge());
    this.tick();
  }

  private livePath(): string {
    return join(this.dir, `live-${process.pid}.json`);
  }

  private onEdge(): void {
    if (this.stopped) return;
    this.noteBusy(this.now(), this.busyNow());
    this.pullIn();
  }

  /**
   * Busy from property reads only; no progress reads. `activeDispatches` stays out: the lane is released
   * after the settle edge with no edge of its own, so counting it stretches every burst to the next tick.
   */
  private busyNow(): boolean {
    if (this.loadsActive > 0) return true;
    for (const entry of this.deps.registry.listResidentModels()) {
      const sessions = entry.sessions;
      if (sessions.turnStats().live > 0 || sessions.queueDepth > 0) return true;
    }
    return false;
  }

  private pullIn(): void {
    if (this.stopped) return;
    const at = this.now() + EDGE_COALESCE_MS;
    if (this.timer !== null && this.nextTickAt <= at) return;
    this.schedule(EDGE_COALESCE_MS);
  }

  private schedule(delayMs: number): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.nextTickAt = this.now() + delayMs;
    const timer = setTimeout(() => {
      this.timer = null;
      this.tick();
    }, delayMs);
    timer.unref?.();
    this.timer = timer;
  }

  private collect(): BindingView[] {
    const out: BindingView[] = [];
    for (const entry of this.deps.registry.listResidentModels()) {
      const sessions = entry.sessions;
      out.push({
        entry,
        name: this.nameOf(entry),
        turns: sessions.liveTurns(),
        stats: sessions.turnStats(),
        active: sessions.activeDispatches,
        queued: sessions.queueDepth,
      });
    }
    return out;
  }

  private computeBusy(views: BindingView[]): boolean {
    if (this.loadsActive > 0) return true;
    for (const view of views) {
      if (view.stats.live > 0 || view.queued > 0) return true;
    }
    return false;
  }

  private noteBusy(now: number, busy: boolean): void {
    if (busy && !this.busy) this.busySince = now;
    else if (!busy && this.busy && this.busySince !== null) {
      this.busyAccumMs += now - this.busySince;
      this.busySince = null;
    }
    this.busy = busy;
  }

  private nameOf(entry: ResidentModel): string {
    const modelId = entry.sessions.identity?.modelId;
    if (modelId !== undefined) return modelId;
    for (const name of entry.names) if (this.discovered.has(name)) return name;
    return entry.names[0] ?? 'unknown';
  }

  private modelKey(model: object): string {
    let id = this.modelIds.get(model);
    if (id === undefined) {
      id = this.nextModelId++;
      this.modelIds.set(model, id);
    }
    return String(id);
  }

  private factsOf(entry: ResidentModel): ModelFacts {
    const model = entry.model as object;
    let facts = this.facts.get(model);
    if (facts === undefined) {
      const ctxWindow = entry.sessions.servedWindowTokens;
      let paged = false;
      try {
        paged = entry.model.hasBlockPagedCache?.() === true;
      } catch {
        // Facts stay at their defaults.
      }
      facts = { ctxWindow, paged };
      this.facts.set(model, facts);
    }
    return facts;
  }

  private tick(): void {
    if (this.stopped) return;
    let busy = this.busy;
    try {
      const now = this.now();
      const views = this.collect();
      busy = this.computeBusy(views);
      this.noteBusy(now, busy);
      this.writeLive(now, views, busy);
      const residentKey = views.map((v) => this.modelKey(v.entry.model as object)).join(',');
      const interval = busy ? SAMPLE_BUSY_MS : SAMPLE_IDLE_MS;
      if (
        this.lastSampleAt === null ||
        now - this.lastSampleAt >= interval ||
        busy !== this.lastSampleBusy ||
        residentKey !== this.lastResidentKey
      ) {
        this.writeSample(now, views, busy);
        this.lastResidentKey = residentKey;
      }
      this.snapshotSchedulers(now, views);
    } catch {
      // A sampler failure must never reach inference; the next tick retries.
    }
    if (!this.stopped) this.schedule(busy ? HEARTBEAT_BUSY_MS : HEARTBEAT_IDLE_MS);
  }

  private health(): RuntimeSampleRecord['health'] {
    try {
      return this.deps.health().status;
    } catch {
      return 'error';
    }
  }

  private writeSample(now: number, views: BindingView[], busy: boolean): void {
    const mem = this.memory();
    const counters = turnCounters();
    const rejected = queueRejections();
    let inFlight = 0;
    let queued = 0;
    for (const view of views) {
      inFlight += view.turns.length;
      queued += view.queued;
    }
    const busyMs = this.busyAccumMs + (this.busySince !== null ? now - this.busySince : 0);
    this.busyAccumMs = 0;
    if (this.busySince !== null) this.busySince = now;
    const rec: RuntimeSampleRecord = {
      v: 1,
      k: 'sample',
      ts: now,
      pid: process.pid,
      startedAt: this.startedAt,
      dtMs: this.lastSampleAt === null ? 0 : now - this.lastSampleAt,
      busy,
      busyMs,
      health: this.health(),
      memActive: finiteOr0(mem.activeBytes),
      memPeak: finiteOr0(mem.peakBytes),
      memCache: finiteOr0(mem.cacheBytes),
      rss: process.memoryUsage.rss(),
      inFlight,
      queued,
      resident: views.length,
      dOk: counters.ok - this.prevCounters.ok,
      dError: counters.error - this.prevCounters.error,
      dCancelled: counters.cancelled - this.prevCounters.cancelled,
      dOutTokens: counters.outputTokens - this.prevCounters.outputTokens,
      dRejected: rejected - this.prevRejected,
    };
    const dropped = this.trace.takeDropped();
    if (dropped > 0) rec.dropped = dropped;
    this.prevCounters = counters;
    this.prevRejected = rejected;
    this.lastSampleAt = now;
    this.lastSampleBusy = busy;
    this.trace.append(rec);
  }

  private snapshotSchedulers(now: number, views: BindingView[]): void {
    for (const view of views) {
      const model = view.entry.model as unknown as { schedulerStats?: () => Promise<SchedulerStats> };
      if (typeof model.schedulerStats !== 'function') continue;
      let state = this.sched.get(model);
      if (state === undefined) {
        state = { pending: false, snapshotBegun: undefined, pool: null };
        this.sched.set(model, state);
      }
      const ok = shouldSnapshotScheduler({
        now,
        activeDispatches: view.active,
        queueDepth: view.queued,
        live: view.stats.live,
        begun: view.stats.begun,
        lastSettleAt: view.stats.lastSettleAt,
        pending: state.pending,
        snapshotBegun: state.snapshotBegun,
      });
      if (!ok) continue;
      const current = state;
      const name = view.name;
      const schedulerStats = model.schedulerStats;
      let request: Promise<SchedulerStats> | null;
      try {
        request = view.entry.sessions.tryIdleProbe(() => schedulerStats.call(model));
      } catch {
        current.snapshotBegun = view.stats.begun;
        continue;
      }
      if (request === null) continue;
      const turns = view.stats.begun;
      current.pending = true;
      current.snapshotBegun = turns;
      request.then(
        (stats) => {
          current.pending = false;
          this.onSchedulerStats(name, current, stats, turns);
        },
        () => {
          current.pending = false;
        },
      );
    }
  }

  private onSchedulerStats(name: string, state: SchedState, stats: SchedulerStats, turns: number): void {
    if (this.stopped) return;
    if (stats.blockCapacity === 0 && stats.globalSteps === 0) return;
    const ts = this.now();
    const rec: RuntimeSchedRecord = {
      v: 1,
      k: 'sched',
      ts,
      pid: process.pid,
      startedAt: this.startedAt,
      model: name,
      turns,
      blockCapacity: finiteOr0(stats.blockCapacity),
      allocatedBlocks: finiteOr0(stats.allocatedBlocks),
      freeBlocks: finiteOr0(stats.freeBlocks),
      reclaimableBlocks: finiteOr0(stats.reclaimableBlocks),
      watermarkBlocks: finiteOr0(stats.watermarkBlocks),
      reservedBlocks: finiteOr0(stats.reservedBlocks),
      memoryCapacityBytes: finiteOr0(stats.memoryCapacityBytes),
      memoryWatermarkBytes: finiteOr0(stats.memoryWatermarkBytes),
      reservedBlockBytes: finiteOr0(stats.reservedBlockBytes),
      reservedStateBytes: finiteOr0(stats.reservedStateBytes),
      globalSteps: finiteOr0(stats.globalSteps),
      admitted: finiteOr0(stats.admitted),
      completed: finiteOr0(stats.completed),
      maxBatchOccupancy: finiteOr0(stats.maxBatchOccupancy),
      fusedGreedyEpilogueSteps: finiteOr0(stats.fusedGreedyEpilogueSteps),
      admissionDeferredBlocks: finiteOr0(stats.admissionDeferredBlocks),
      admissionDeferredState: finiteOr0(stats.admissionDeferredState),
      rowsAllocEvicted: finiteOr0(stats.rowsAllocEvicted),
      preemptions: finiteOr0(stats.preemptions),
      preemptionsRecompute: finiteOr0(stats.preemptionsRecompute),
      preemptionsSsd: finiteOr0(stats.preemptionsSsd),
      ssdRestoreWaiting: finiteOr0(stats.ssdRestoreWaiting),
      ssdRestoreBytes: finiteOr0(stats.ssdRestoreBytes),
      ssdRestoreWaitMs: finiteOr0(stats.ssdRestoreWaitMs),
      occHist: (stats.decodeBatchOccupancyHist ?? []).map((b) => [finiteOr0(b.occupancy), finiteOr0(b.steps)]),
    };
    state.pool = {
      at: ts,
      blockCapacity: rec.blockCapacity,
      allocatedBlocks: rec.allocatedBlocks,
      freeBlocks: rec.freeBlocks,
      reclaimableBlocks: rec.reclaimableBlocks,
    };
    this.trace.append(rec);
  }

  private liveModel(view: BindingView): RuntimeLiveModel {
    const sessions = view.entry.sessions;
    const identity = sessions.identity;
    const discovered = this.discovered.get(view.name);
    const facts = this.factsOf(view.entry);
    const loaded = this.loaded.get(view.name);
    const schedState = this.sched.get(view.entry.model as object);
    return {
      name: view.name,
      modelType: identity?.modelType ?? discovered?.modelType ?? '',
      ctxWindow: facts.ctxWindow ?? identity?.contextWindow ?? discovered?.contextWindow ?? 0,
      paged: facts.paged,
      maxConcurrent: sessions.concurrentAdmissionLimit,
      loadedAt: loaded?.loadedAt ?? null,
      loadMs: loaded?.loadMs ?? null,
      inFlight: view.turns.length,
      queued: view.queued,
      queueLimit: sessions.queueDepthLimit ?? null,
      pool: schedState?.pool ?? null,
    };
  }

  private liveTurns(views: BindingView[]): RuntimeLiveTurn[] {
    const all: LiveTurn[] = [];
    for (const view of views) for (const turn of view.turns) all.push(turn);
    all.sort((a, b) => a.startedAt - b.startedAt);
    return all.slice(0, LIVE_TURNS_MAX).map((turn) => {
      const p = turn.progress;
      const phase = p?.phase === 'queued' || p?.phase === 'prefill' || p?.phase === 'decode' ? p.phase : null;
      const cached = p?.cachedTokens ?? null;
      const prefillTotal = p?.prefillTotal ?? null;
      return {
        traceId: turn.traceId,
        model: turn.model,
        modelId: turn.modelId,
        ctxWindow: turn.ctxWindow,
        startedAt: turn.startedAt,
        stream: turn.stream,
        dispatch: turn.dispatch,
        phase,
        promptTokens: cached !== null && prefillTotal !== null ? cached + prefillTotal : null,
        cachedTokens: cached,
        prefillTotal,
        prefillDone: p?.prefillDone ?? null,
        generatedTokens: p?.generatedTokens ?? null,
      };
    });
  }

  private buildLive(now: number, views: BindingView[], busy: boolean): RuntimeLiveFile {
    const mem = this.memory();
    const counters = turnCounters();
    let inFlight = 0;
    for (const view of views) inFlight += view.turns.length;
    let decodeTps: number | null = null;
    if (inFlight > 0 && this.lastLiveAt !== null && now > this.lastLiveAt) {
      const rate = ((counters.streamedTokens - this.lastLiveTokens) * 1000) / (now - this.lastLiveAt);
      if (Number.isFinite(rate) && rate > 0) decodeTps = Math.round(rate * 100) / 100;
    }
    this.lastLiveAt = now;
    this.lastLiveTokens = counters.streamedTokens;
    return {
      v: 1,
      pid: process.pid,
      startedAt: this.startedAt,
      ts: now,
      intervalMs: busy ? HEARTBEAT_BUSY_MS : HEARTBEAT_IDLE_MS,
      host: this.deps.host.kind,
      version: String(pkgJson.version),
      port: this.deps.host.port,
      persistPagedCache: this.deps.host.persistPagedCache,
      coldCapture: liveColdCapture(this.deps.host.coldCapture),
      health: this.health(),
      mem: {
        active: finiteOr0(mem.activeBytes),
        peak: finiteOr0(mem.peakBytes),
        cache: finiteOr0(mem.cacheBytes),
        wiredLimit: this.wiredLimit,
        rss: process.memoryUsage.rss(),
      },
      decodeTps,
      loading: this.loading,
      counters: {
        ok: counters.ok,
        error: counters.error,
        cancelled: counters.cancelled,
        outputTokens: counters.outputTokens,
        loads: this.loads,
        rejected: queueRejections(),
      },
      models: views.map((view) => this.liveModel(view)),
      turns: this.liveTurns(views),
    };
  }

  private writeLive(now: number, views: BindingView[], busy: boolean): void {
    if (this.liveWrite !== null || !this.trace.enabled) return;
    const body = JSON.stringify(this.buildLive(now, views, busy));
    const path = this.livePath();
    const tmp = `${path}.tmp`;
    this.liveWrite = (async () => {
      try {
        if (!this.liveDirReady) {
          await mkdir(this.dir, { recursive: true });
          this.liveDirReady = true;
        }
        await writeFile(tmp, body);
        await rename(tmp, path);
      } catch {
        this.liveDirReady = false;
      } finally {
        this.liveWrite = null;
      }
    })();
  }

  async timedLoad<T>(
    info: { model: string; modelType: string; capability: RuntimeCapability },
    load: () => Promise<T>,
  ): Promise<T> {
    if (this.stopped) return load();
    const startedAt = this.now();
    this.loadsActive += 1;
    this.loading = { model: info.model, startedAt };
    this.noteBusy(startedAt, true);
    this.pullIn();
    let ok = false;
    try {
      const result = await load();
      ok = true;
      return result;
    } finally {
      const end = this.now();
      const ms = Math.max(0, end - startedAt);
      this.loadsActive -= 1;
      if (this.loadsActive === 0) this.loading = null;
      this.loads += 1;
      if (ok) this.loaded.set(info.model, { loadedAt: end, loadMs: ms });
      if (!this.stopped) {
        this.trace.append({
          v: 1,
          k: 'load',
          ts: end,
          pid: process.pid,
          startedAt: this.startedAt,
          model: info.model,
          modelType: info.modelType,
          capability: info.capability,
          ms,
          ok,
        });
        this.noteBusy(end, this.busyNow());
        this.pullIn();
      }
    }
  }

  stop(): Promise<void> {
    this.stopPromise ??= this.doStop();
    return this.stopPromise;
  }

  private async doStop(): Promise<void> {
    try {
      const now = this.now();
      try {
        const views = this.collect();
        const busy = this.computeBusy(views);
        this.noteBusy(now, busy);
        this.writeSample(now, views, busy);
      } catch {
        // The exit record still goes out.
      }
      this.stopped = true;
      if (this.timer !== null) clearTimeout(this.timer);
      this.timer = null;
      setTurnEdgeListener(null);
      this.trace.append({
        v: 1,
        k: 'exit',
        ts: now,
        pid: process.pid,
        startedAt: this.startedAt,
        uptimeMs: now - this.startedAt,
      });
      await this.trace.close();
      if (this.liveWrite !== null) await this.liveWrite;
      await unlink(this.livePath()).catch(() => undefined);
    } finally {
      running = false;
    }
  }
}

/**
 * Start the sampler. Inert (writes nothing) when `MLX_AGENT_METRICS` is off,
 * and for every call after the first in the same process until it stops.
 */
export function startRuntimeSampler(deps: RuntimeSamplerDeps): RuntimeSampler {
  if (metricsEnvDisabled() || running) return INERT;
  running = true;
  const sampler = new Sampler(deps);
  sampler.start();
  return sampler;
}
