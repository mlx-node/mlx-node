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
import { join, resolve } from 'node:path';

import { memoryStats, type SchedulerStats } from '@mlx-node/core';
import { MODEL_FAMILY_DATA } from '@mlx-node/lm/family-data';

import pkgJson from '../../package.json' with { type: 'json' };
import type { ServerHealthStatus } from '../health.js';
import { metricsEnvDisabled, TURN_FAIL_PHASES } from '../metrics-trace.js';
import {
  expertsLiveOf,
  LOAD_RESIDENCY_FIELDS,
  readLoadProgress,
  readPipelineStats,
  readResidency,
  type LoadProgress,
  type PipelineStats,
  type ResidencySnapshot,
  type WarmupReport,
} from '../native-progress.js';
import { queueRejections } from '../queue-rejections.js';
import type { ResidentModel } from '../registry.js';
import {
  metricsRuntimeDir,
  RUNTIME_RETENTION_DAYS,
  RuntimeTrace,
  type HostKind,
  type RuntimeCapability,
  type RuntimeLiveFile,
  type RuntimeLiveLoad,
  type RuntimeLiveModel,
  type RuntimeLoadRecord,
  type RuntimeLiveTurn,
  type RuntimeSampleRecord,
  type RuntimeSchedRecord,
} from '../runtime-trace.js';
import { setTurnEdgeListener, turnCounters, type LiveTurn, type TurnCounters } from '../turn-trace.js';
import { WORK, type LoadSpan, type WorkLedger, type WorkRates, type WorkTotals } from '../work-ledger.js';
import type { ColdCaptureBudget } from './persist-cache.js';
import { isProcessAlive } from './temp-root.js';

export const HEARTBEAT_BUSY_MS = 1000;
export const HEARTBEAT_IDLE_MS = 5000;
export const SAMPLE_BUSY_MS = 5000;
export const SAMPLE_IDLE_MS = 60_000;
export const EDGE_COALESCE_MS = 250;
const MODEL_META_REFRESH_MS = SAMPLE_BUSY_MS;
export const SCHED_IDLE_SETTLE_MS = 500;
export const LIVE_TURNS_MAX = 32;
export const LIVE_DEAD_AFTER_MS = 300_000;

const DAY_MS = 86_400_000;

/** The part of the server's health the heartbeat reads. */
export interface SamplerHealth {
  status: ServerHealthStatus;
  memory_governor?: { dflash2_states: { disk: { mode: string; limit_bytes: number } } };
}

export interface RuntimeSamplerDeps {
  registry: { listResidentModels(): ResidentModel[] };
  health: () => SamplerHealth;
  /** Discovered inventory: name → family, window. */
  models: ReadonlyArray<{ name: string; modelType: string; contextWindow: number }>;
  host: {
    kind: HostKind;
    port: number;
    persistPagedCache: boolean;
    coldCapture: ColdCaptureBudget | null;
    /** `--cache-dir`; null for the default root. */
    cacheDir: string | null;
  };
  trace?: RuntimeTrace;
  now?: () => number;
  memory?: () => { activeBytes: number; peakBytes: number; cacheBytes: number };
  wiredLimit?: () => number;
  /** Runtime directory; defaults to {@link metricsRuntimeDir}. */
  dir?: string;
  /** Liveness probe for the start-up sweep. */
  isAlive?: (pid: number) => boolean;
  /** Native progress of the load of `path`; defaults to the native addon. */
  loadProgress?: (path: string) => LoadProgress | null;
  /** Process-wide pipeline counter; defaults to the native addon. Null while the counter is off. */
  pipelineStats?: () => PipelineStats | null;
  /** Lane time ledger the sampler advances; defaults to the process-wide one the traced turns credit. */
  work?: WorkLedger;
}

/** What {@link RuntimeSampler.timedLoad} records about one load. */
export interface TimedLoadInfo<T> {
  model: string;
  modelType: string;
  capability: RuntimeCapability;
  /** The exact path the native load receives: the key of its load progress. */
  path: string;
  /** The warm-up report of a loaded model; null when the load ran none. */
  warmupOf?: (result: T) => WarmupReport | null;
  /** The expert residency of a loaded model; null for a model without one. */
  residencyOf?: (result: T) => ResidencySnapshot | null;
}

export interface RuntimeSampler {
  /** Wrap a real load. Records one `load` record and an entry of the live `loading` list. */
  timedLoad<T>(info: TimedLoadInfo<T>, load: () => Promise<T>): Promise<T>;
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

function roundedOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.round(value) : null;
}

function roundedWorkMs(ms: WorkTotals['ms']): WorkTotals['ms'] {
  return {
    loading: Math.round(ms.loading),
    waiting: Math.round(ms.waiting),
    prepare: Math.round(ms.prepare),
    prefill: Math.round(ms.prefill),
    decode: Math.round(ms.decode),
    running: Math.round(ms.running),
  };
}

/** Rounded `now - prev`, taken on the rounded totals so the deltas of all samples sum to the rounded total. */
function workDeltas(
  now: WorkTotals,
  prev: WorkTotals,
): Pick<
  RuntimeSampleRecord,
  | 'loadingMs'
  | 'waitingMs'
  | 'prepareMs'
  | 'prefillMs'
  | 'decodeMs'
  | 'runningMs'
  | 'rateOutTokens'
  | 'rateOutMs'
  | 'ratePrefillTokens'
  | 'ratePrefillMs'
> {
  const d = (a: number, b: number): number => Math.round(a) - Math.round(b);
  return {
    loadingMs: d(now.ms.loading, prev.ms.loading),
    waitingMs: d(now.ms.waiting, prev.ms.waiting),
    prepareMs: d(now.ms.prepare, prev.ms.prepare),
    prefillMs: d(now.ms.prefill, prev.ms.prefill),
    decodeMs: d(now.ms.decode, prev.ms.decode),
    runningMs: d(now.ms.running, prev.ms.running),
    rateOutTokens: d(now.outTokens, prev.outTokens),
    rateOutMs: d(now.outMs, prev.outMs),
    ratePrefillTokens: d(now.prefillTokens, prev.prefillTokens),
    ratePrefillMs: d(now.prefillMs, prev.prefillMs),
  };
}

function livePhase(value: unknown): RuntimeLiveTurn['phase'] {
  for (const phase of TURN_FAIL_PHASES) if (phase === value) return phase;
  return null;
}

function warmupFields(report: WarmupReport): Partial<RuntimeLoadRecord> {
  return {
    warmupMs: report.ms,
    warmupSteps: report.steps,
    warmupSkipped: report.skipped,
    warmupReset: report.reset,
    warmupPipelines: report.pipelines,
    warmupCompileMs: report.compileMs,
    warmupPeakBytes: report.peakBytes,
  };
}

function residencyFields(snapshot: ResidencySnapshot): Partial<RuntimeLoadRecord> {
  const fields: Partial<RuntimeLoadRecord> = {
    residency: snapshot.residency,
    expertSlots: snapshot.expertSlots,
    expertCount: snapshot.expertCount,
  };
  for (const key of LOAD_RESIDENCY_FIELDS) {
    const value = snapshot.load[key];
    if (value !== undefined) fields[key] = value;
  }
  return fields;
}

/**
 * Model types whose loads publish a state tier to `memory_governor.dflash2_states.disk`;
 * `''` is a resident of unknown type (counted conservatively so a summed limit is never
 * attributed to one tier). Families with `acceptsCacheDisk` that omit
 * `publishesGovernorStateDisk` (`qwen4_exp`, `hy_v4_flash`, `inkling2`) do not publish to
 * `dflash2_states` and are intentionally excluded here.
 */
const STATE_TIER_TYPES: ReadonlySet<string> = new Set([
  ...MODEL_FAMILY_DATA.filter(
    (row) => 'publishesGovernorStateDisk' in row && row.publishesGovernorStateDisk === true,
  ).map((row) => row.id),
  '',
]);

/**
 * One persistent state tier's limit, or null when the governor's figure is not one tier's: the governor sums the
 * limits of every state-tier model and reports the largest mode.
 */
export function stateRootLimit(
  disk: { mode: string; limit_bytes: number } | undefined,
  modelTypes: readonly string[],
): number | null {
  if (disk === undefined || disk.mode !== 'persistent') return null;
  let tiers = 0;
  for (const type of modelTypes) if (STATE_TIER_TYPES.has(type)) tiers += 1;
  if (tiers !== 1) return null;
  return Number.isFinite(disk.limit_bytes) && disk.limit_bytes > 0 ? disk.limit_bytes : null;
}

/** The model's `stateTierNamespace()`: a weak read of its tier, no call into the model thread. */
function stateNamespaceOf(model: object): string | null {
  const read = (model as { stateTierNamespace?: () => unknown }).stateTierNamespace;
  if (typeof read !== 'function') return null;
  try {
    const namespace = read.call(model);
    return typeof namespace === 'string' && namespace.length > 0 ? namespace : null;
  } catch {
    return null;
  }
}

function imagesOf(model: object): boolean | null {
  const ask = (model as { supportsImages?: () => unknown }).supportsImages;
  if (typeof ask !== 'function') return false;
  try {
    return ask.call(model) === true;
  } catch {
    return null;
  }
}

function liveExperts(model: object): RuntimeLiveModel['experts'] {
  const snapshot = readResidency(model);
  return snapshot === null ? null : expertsLiveOf(snapshot);
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
  images: boolean | null;
  hasStateNamespace: boolean;
  hasResidency: boolean;
  stateNamespace: string | null;
  stateNamespaceAt: number | null;
  experts: RuntimeLiveModel['experts'];
  expertsAt: number | null;
}

interface TickSnapshot {
  counters: TurnCounters;
  inFlight: number;
  queued: number;
  mem?: { activeBytes: number; peakBytes: number; cacheBytes: number };
  rss?: number;
  rejected?: number;
  health?: SamplerHealth | null;
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
  /** Busy since the last sample, also when no tick saw it: a burst between two ticks still gets its sample. */
  private busySinceSample = false;

  private lastSampleAt: number | null = null;
  private lastSampleBusy = false;
  private lastResidentKey = '';
  private prevCounters: TurnCounters;
  private prevRejected: number;

  /** One entry per load in flight; two loads of one model or one path keep their own entries. */
  private readonly loading = new Set<{ model: string; startedAt: number; path: string }>();
  /** Loads that ended since the last ledger advance, which still folds their time. */
  private readonly endedLoads: LoadSpan[] = [];
  private loads = 0;
  private readonly loaded = new Map<string, { loadedAt: number; loadMs: number }>();

  private readonly sched = new WeakMap<object, SchedState>();
  private readonly facts = new WeakMap<object, ModelFacts>();
  private readonly modelIds = new WeakMap<object, number>();
  private nextModelId = 1;

  private liveWrite: Promise<void> | null = null;
  private liveDirReady = false;
  private readonly work: WorkLedger;
  /** Ledger totals at the previous sample. */
  private prevWork: WorkTotals;

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
    this.work = deps.work ?? WORK;
    this.prevWork = this.work.totals();
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
    if (this.loading.size > 0) return true;
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
    if (this.loading.size > 0) return true;
    for (const view of views) {
      if (view.stats.live > 0 || view.queued > 0) return true;
    }
    return false;
  }

  private noteBusy(now: number, busy: boolean): void {
    if (busy) this.busySinceSample = true;
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
        paged =
          'hasBlockPagedCache' in entry.model && typeof entry.model.hasBlockPagedCache === 'function'
            ? entry.model.hasBlockPagedCache() === true
            : false;
      } catch {
        // Facts stay at their defaults.
      }
      const hasStateNamespace = typeof (model as { stateTierNamespace?: unknown }).stateTierNamespace === 'function';
      const hasResidency = typeof (model as { residencyInfo?: unknown }).residencyInfo === 'function';
      facts = {
        ctxWindow,
        paged,
        images: imagesOf(model),
        hasStateNamespace,
        hasResidency,
        stateNamespace: null,
        stateNamespaceAt: null,
        experts: null,
        expertsAt: null,
      };
      this.facts.set(model, facts);
    }
    return facts;
  }

  private snapshotTick(views: BindingView[]): TickSnapshot {
    let inFlight = 0;
    let queued = 0;
    for (const view of views) {
      inFlight += view.entry.kind === 'decision' ? view.active : view.turns.length;
      queued += view.queued;
    }
    return { counters: turnCounters(), inFlight, queued };
  }

  /** Fold the ledger up to `now`. The views were collected first: that read credits each live turn's progress. */
  private advanceWork(now: number): WorkTotals {
    const loads: LoadSpan[] = [...this.endedLoads];
    for (const load of this.loading) loads.push({ startedAt: load.startedAt, endedAt: null });
    this.endedLoads.length = 0;
    return this.work.advance(now, loads);
  }

  private memOf(snap: TickSnapshot): { activeBytes: number; peakBytes: number; cacheBytes: number } {
    snap.mem ??= this.memory();
    return snap.mem;
  }

  private rssOf(snap: TickSnapshot): number {
    snap.rss ??= process.memoryUsage.rss();
    return snap.rss;
  }

  private rejectedOf(snap: TickSnapshot): number {
    snap.rejected ??= queueRejections();
    return snap.rejected;
  }

  private healthOf(snap: TickSnapshot): SamplerHealth | null {
    if (snap.health === undefined) {
      try {
        snap.health = this.deps.health();
      } catch {
        snap.health = null;
      }
    }
    return snap.health;
  }

  private tick(): void {
    if (this.stopped) return;
    let busy = this.busy;
    let settleDelay: number | null = null;
    try {
      const now = this.now();
      const views = this.collect();
      busy = this.computeBusy(views);
      this.noteBusy(now, busy);
      const snap = this.snapshotTick(views);
      const work = this.advanceWork(now);
      const rates = this.work.rates(now);
      this.writeLive(now, views, busy, snap, work, rates);
      const residentKey = views.map((v) => this.modelKey(v.entry.model as object)).join(',');
      const interval = busy ? SAMPLE_BUSY_MS : SAMPLE_IDLE_MS;
      if (
        this.lastSampleAt === null ||
        now - this.lastSampleAt >= interval ||
        busy !== this.lastSampleBusy ||
        (!busy && this.busySinceSample) ||
        residentKey !== this.lastResidentKey
      ) {
        this.writeSample(now, views, busy, snap, work);
        this.lastResidentKey = residentKey;
      }
      this.snapshotSchedulers(now, views);
      settleDelay = this.nextSchedulerSettleDelayMs(now, views);
    } catch {
      // A sampler failure must never reach inference; the next tick retries.
    }
    if (!this.stopped) {
      let delayMs = busy ? HEARTBEAT_BUSY_MS : HEARTBEAT_IDLE_MS;
      if (settleDelay !== null && settleDelay < delayMs) delayMs = settleDelay;
      this.schedule(delayMs);
    }
  }

  private writeSample(now: number, views: BindingView[], busy: boolean, snap: TickSnapshot, work: WorkTotals): void {
    const mem = this.memOf(snap);
    const counters = snap.counters;
    const rejected = this.rejectedOf(snap);
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
      health: this.healthOf(snap)?.status ?? 'error',
      memActive: finiteOr0(mem.activeBytes),
      memPeak: finiteOr0(mem.peakBytes),
      memCache: finiteOr0(mem.cacheBytes),
      rss: this.rssOf(snap),
      inFlight: snap.inFlight,
      queued: snap.queued,
      resident: views.length,
      dOk: counters.ok - this.prevCounters.ok,
      dError: counters.error - this.prevCounters.error,
      dCancelled: counters.cancelled - this.prevCounters.cancelled,
      dOutTokens: counters.outputTokens - this.prevCounters.outputTokens,
      dRejected: rejected - this.prevRejected,
      ...workDeltas(work, this.prevWork),
    };
    const dropped = this.trace.takeDropped();
    if (dropped > 0) rec.dropped = dropped;
    const pipelines = this.pipelineStats();
    if (pipelines !== null) rec.pipelinesCreatedWhileServing = pipelines.createdServing;
    this.prevCounters = counters;
    this.prevRejected = rejected;
    this.prevWork = work;
    this.lastSampleAt = now;
    this.lastSampleBusy = busy;
    this.busySinceSample = busy;
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

  /**
   * Shortest remaining settle wait for any idle model that still owes a `schedulerStats()`
   * snapshot. Without this, an edge-coalesced idle tick at +250 ms (`< SCHED_IDLE_SETTLE_MS`)
   * sleeps a full `HEARTBEAT_IDLE_MS` (5 s), so models with brief idle gaps never take a reading.
   */
  private nextSchedulerSettleDelayMs(now: number, views: BindingView[]): number | null {
    let soonest: number | null = null;
    for (const view of views) {
      const model = view.entry.model as unknown as { schedulerStats?: () => Promise<SchedulerStats> };
      if (typeof model.schedulerStats !== 'function') continue;
      if (view.active !== 0 || view.queued !== 0 || view.stats.live !== 0) continue;
      const state = this.sched.get(model);
      if (state?.pending) continue;
      if (state !== undefined && state.snapshotBegun !== undefined && state.snapshotBegun === view.stats.begun) {
        continue;
      }
      const remaining = view.stats.lastSettleAt + SCHED_IDLE_SETTLE_MS - now;
      if (remaining <= 0) continue;
      if (soonest === null || remaining < soonest) soonest = remaining;
    }
    return soonest;
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

  private liveModel(view: BindingView, now: number, busy: boolean): RuntimeLiveModel {
    const sessions = view.entry.sessions;
    const identity = sessions.identity;
    const discovered = this.discovered.get(view.name);
    const model = view.entry.model as object;
    const facts = this.factsOf(view.entry);
    if (
      facts.hasStateNamespace &&
      (facts.stateNamespaceAt === null || !busy || now - facts.stateNamespaceAt >= MODEL_META_REFRESH_MS)
    ) {
      facts.stateNamespace = stateNamespaceOf(model);
      facts.stateNamespaceAt = now;
    }
    if (facts.hasResidency && (facts.expertsAt === null || !busy || now - facts.expertsAt >= MODEL_META_REFRESH_MS)) {
      facts.experts = liveExperts(model);
      facts.expertsAt = now;
    }
    const loaded = this.loaded.get(view.name);
    const schedState = this.sched.get(model);
    return {
      name: view.name,
      modelType: identity?.modelType ?? discovered?.modelType ?? '',
      ctxWindow: facts.ctxWindow ?? identity?.contextWindow ?? discovered?.contextWindow ?? 0,
      paged: facts.paged,
      maxConcurrent: sessions.concurrentAdmissionLimit,
      loadedAt: loaded?.loadedAt ?? null,
      loadMs: loaded?.loadMs ?? null,
      inFlight: view.entry.kind === 'decision' ? view.active : view.turns.length,
      queued: view.queued,
      queueLimit: sessions.queueDepthLimit ?? null,
      pool: schedState?.pool ?? null,
      stateNamespace: facts.stateNamespace,
      experts: facts.experts,
      images: facts.images,
    };
  }

  private liveTurns(views: BindingView[]): RuntimeLiveTurn[] {
    const all: LiveTurn[] = [];
    for (const view of views) for (const turn of view.turns) all.push(turn);
    all.sort((a, b) => a.startedAt - b.startedAt);
    return all.slice(0, LIVE_TURNS_MAX).map((turn) => {
      const p = turn.progress;
      const phase = livePhase(p?.phase);
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
        waitMs: roundedOrNull(p?.waitMs),
        slotMs: roundedOrNull(p?.slotMs),
        prepareMs: roundedOrNull(p?.prepareMs),
        prefillElapsedMs: roundedOrNull(p?.prefillElapsedMs),
        decodeElapsedMs: roundedOrNull(p?.decodeElapsedMs),
        promptTokens: cached !== null && prefillTotal !== null ? cached + prefillTotal : null,
        cachedTokens: cached,
        prefillTotal,
        prefillDone: p?.prefillDone ?? null,
        generatedTokens: p?.generatedTokens ?? null,
        clientSession: turn.clientSession,
      };
    });
  }

  private buildLive(
    now: number,
    views: BindingView[],
    busy: boolean,
    snap: TickSnapshot,
    work: WorkTotals,
    rates: WorkRates,
  ): RuntimeLiveFile {
    const mem = this.memOf(snap);
    const counters = snap.counters;
    const health = this.healthOf(snap);
    const models = views.map((view) => this.liveModel(view, now, busy));
    return {
      v: 3,
      pid: process.pid,
      startedAt: this.startedAt,
      ts: now,
      intervalMs: busy ? HEARTBEAT_BUSY_MS : HEARTBEAT_IDLE_MS,
      host: this.deps.host.kind,
      version: String(pkgJson.version),
      port: this.deps.host.port,
      persistPagedCache: this.deps.host.persistPagedCache,
      coldCapture: liveColdCapture(this.deps.host.coldCapture),
      stateCache: {
        dir: this.deps.host.cacheDir === null ? null : resolve(this.deps.host.cacheDir),
        limitBytes: stateRootLimit(
          health?.memory_governor?.dflash2_states.disk,
          models.map((model) => model.modelType),
        ),
      },
      health: health?.status ?? 'error',
      mem: {
        active: finiteOr0(mem.activeBytes),
        peak: finiteOr0(mem.peakBytes),
        cache: finiteOr0(mem.cacheBytes),
        wiredLimit: this.wiredLimit,
        rss: this.rssOf(snap),
      },
      work: { ms: roundedWorkMs(work.ms) },
      rates: { windowMs: rates.windowMs, outputTps: rates.outputTps, prefillTps: rates.prefillTps },
      loading: this.liveLoads(),
      counters: {
        ok: counters.ok,
        error: counters.error,
        cancelled: counters.cancelled,
        outputTokens: counters.outputTokens,
        loads: this.loads,
        rejected: this.rejectedOf(snap),
      },
      models,
      turns: this.liveTurns(views),
    };
  }

  private pipelineStats(): PipelineStats | null {
    try {
      return (this.deps.pipelineStats ?? readPipelineStats)();
    } catch {
      return null;
    }
  }

  private liveLoads(): RuntimeLiveLoad[] {
    const read = this.deps.loadProgress ?? readLoadProgress;
    return [...this.loading]
      .sort((a, b) => a.startedAt - b.startedAt)
      .map((load) => {
        let progress: LoadProgress | null = null;
        try {
          progress = read(load.path);
        } catch {
          progress = null;
        }
        return {
          model: load.model,
          startedAt: load.startedAt,
          phase: progress?.phase ?? 'weights',
          stepsDone: progress?.stepsDone ?? null,
          steps: progress?.steps ?? null,
        };
      });
  }

  private writeLive(
    now: number,
    views: BindingView[],
    busy: boolean,
    snap: TickSnapshot,
    work: WorkTotals,
    rates: WorkRates,
  ): void {
    if (this.liveWrite !== null || !this.trace.enabled) return;
    const body = JSON.stringify(this.buildLive(now, views, busy, snap, work, rates));
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

  async timedLoad<T>(info: TimedLoadInfo<T>, load: () => Promise<T>): Promise<T> {
    if (this.stopped) return load();
    const startedAt = this.now();
    const entry = { model: info.model, startedAt, path: info.path };
    this.loading.add(entry);
    this.noteBusy(startedAt, true);
    this.pullIn();
    let ok = false;
    let warmup: WarmupReport | null = null;
    let residency: ResidencySnapshot | null = null;
    try {
      const result = await load();
      ok = true;
      try {
        warmup = info.warmupOf?.(result) ?? null;
      } catch {
        warmup = null;
      }
      try {
        residency = info.residencyOf?.(result) ?? null;
      } catch {
        residency = null;
      }
      return result;
    } finally {
      const end = this.now();
      const ms = Math.max(0, end - startedAt);
      this.loading.delete(entry);
      if (!this.stopped) this.endedLoads.push({ startedAt, endedAt: end });
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
          ...(warmup === null ? {} : warmupFields(warmup)),
          ...(residency === null ? {} : residencyFields(residency)),
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
        const snap = this.snapshotTick(views);
        this.writeSample(now, views, busy, snap, this.advanceWork(now));
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
