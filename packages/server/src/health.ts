/**
 * Server readiness reporting for supervisors (e.g. an Electron app running
 * the inference server as a child process).
 *
 * The old `/health` returned a constant `{ status: 'ok' }`, which cannot
 * distinguish four states a supervisor genuinely needs to tell apart:
 *
 *   - up, nothing loaded          → keep waiting, this is normal
 *   - up, model resident          → route traffic
 *   - wedged mid-load             → keep waiting, do NOT restart
 *   - wedged behind a full queue  → shed load / warn the user
 *
 * Every input already exists in-process; this module only exposes it. The
 * status ladder itself is a PURE function ({@link deriveHealthStatus}) over a
 * plain record so it can be unit-tested without a server, a registry, or the
 * native addon.
 *
 * IMPORTANT: nothing here may call into `@mlx-node/core`. `/health` is
 * deliberately excluded from the idle sweeper's `beginRequest`/`endRequest`
 * bracket (see `HandlerOptions.idleSweeper`), so a native call from this path
 * would run outside the in-flight accounting the drain timer relies on.
 * Every field below is read from plain JavaScript state.
 */

import type { IdleSweeper } from './idle-sweeper.js';
import type { ModelWorkCoordinator } from './model-work-coordinator.js';
import type { ModelRegistry } from './registry.js';

/**
 * Coarse readiness classification. Ordered by precedence in
 * {@link deriveHealthStatus} — a higher rung wins even when a lower rung's
 * condition also holds.
 */
export type ServerHealthStatus = 'ok' | 'loading' | 'degraded' | 'error';

/**
 * Outcome of the most recent load bracket, recorded by
 * {@link ModelWorkCoordinator} in the `finally` of `withModelLoad` /
 * `withModelLoadInstrumented`.
 *
 * Before this existed, a `resolveModel` failure became an HTTP 500 and was
 * dropped — a supervisor polling after the fact had no way to learn WHY the
 * server had no resident model.
 *
 * Caveat worth knowing when reading this field: the coordinator brackets
 * EVERY resolve attempt, including the no-op fast path taken when the
 * requested model is already resident. A successful no-op therefore
 * overwrites an earlier failure record. That is why the `'error'` rung is
 * additionally gated on "no resident models" — a server that is answering
 * requests is not in an error state regardless of what the last bracket did.
 */
export interface ModelLoadRecord {
  /** Caller-supplied label, normally the model name. `null` when unlabelled. */
  label: string | null;
  /** `Date.now()` at writer-lock acquisition (when the load actually began). */
  startedAt: number;
  /** `Date.now()` when the bracket settled, success or failure. */
  finishedAt: number;
  ok: boolean;
  /** Message of the thrown error, or `null` on success. */
  error: string | null;
}

/** Inputs to the pure status ladder. Deliberately primitive so tests can fixture them. */
export interface HealthStatusInputs {
  /** True while a load holds the coordinator's exclusive writer slot. */
  writerActive: boolean;
  /** Loads parked waiting for the writer slot. */
  waitingWriters: number;
  /** Inference requests currently bracketed by the idle sweeper. */
  inFlight: number;
  /** Distinct model names currently registered. */
  residentModelCount: number;
  /** True when at least one per-model queue is holding its configured max waiters. */
  queueSaturated: boolean;
  /** Most recent load bracket, or `null` if none has settled. */
  lastLoad: ModelLoadRecord | null;
}

/** Full readiness body. `{ status: 'ok' }` remains a strict subset of this. */
export interface ServerHealth {
  status: ServerHealthStatus;
  /** Milliseconds since the reporter was created (≈ server start). */
  uptimeMs: number;
  pid: number;
  models: {
    /** Registered names, including aliases. */
    resident: string[];
    count: number;
  };
  work: {
    inFlight: number;
    /** True while the idle sweeper has a `clearCache()` drain armed. */
    drainPending: boolean;
    writerActive: boolean;
    waitingWriters: number;
  };
  queue: {
    /** Deepest per-model waiter count across every session registry. */
    depth: number;
    /** Configured per-model waiter cap, or `null` when unbounded. */
    limit: number | null;
    saturated: boolean;
  };
  lastLoad: ModelLoadRecord | null;
}

/**
 * The subset served to an UNAUTHENTICATED `/health` poll on a
 * token-protected server. A supervisor must be able to poll before it holds a
 * token, but `models.resident` is user data — model names routinely leak
 * project names and local paths.
 */
export interface ServerHealthMinimal {
  status: ServerHealthStatus;
  uptimeMs: number;
  pid: number;
}

/**
 * Pure status ladder. No I/O, no clock, no native calls — just the four
 * documented rungs, in precedence order:
 *
 *   1. `writerActive`                        → 'loading'
 *   2. last load failed AND nothing resident → 'error'
 *   3. queue saturated, OR a load is parked
 *      behind live inference                 → 'degraded'
 *   4. otherwise                             → 'ok'
 *
 * `'loading'` outranks `'error'` on purpose: a retry that already holds the
 * writer slot means the supervisor should wait, not restart the process.
 */
export function deriveHealthStatus(input: HealthStatusInputs): ServerHealthStatus {
  if (input.writerActive) return 'loading';
  if (input.lastLoad?.ok === false && input.residentModelCount === 0) return 'error';
  if (input.queueSaturated) return 'degraded';
  // A writer parked while readers are still running means the swap cannot
  // proceed until they drain, and every request for the incoming model
  // stalls behind it. Either condition alone is an ordinary transient.
  if (input.waitingWriters > 0 && input.inFlight > 0) return 'degraded';
  return 'ok';
}

/** Project the full body down to the three fields safe to serve without a token. */
export function toMinimalHealth(health: ServerHealth): ServerHealthMinimal {
  return { status: health.status, uptimeMs: health.uptimeMs, pid: health.pid };
}

export interface HealthReporterDeps {
  registry: ModelRegistry;
  /** Optional: supplies `inFlight` / `drainPending`. Absent ⇒ both read as idle. */
  idleSweeper?: IdleSweeper | null;
  /** Optional: supplies writer state + `lastLoad`. Absent ⇒ no load has ever run. */
  modelWorkCoordinator?: ModelWorkCoordinator | null;
  /** Defaults to `Date.now()` at construction. */
  startedAt?: number;
  /** Injectable clock for tests. */
  now?: () => number;
}

/**
 * Build a zero-argument reporter closing over the live server objects. Each
 * call re-reads current state; nothing is cached, so a supervisor polling on
 * an interval always sees the present moment.
 */
export function createHealthReporter(deps: HealthReporterDeps): () => ServerHealth {
  const now = deps.now ?? ((): number => Date.now());
  const startedAt = deps.startedAt ?? now();

  return (): ServerHealth => {
    const resident = deps.registry.list().map((entry) => entry.id);

    // Queue saturation is inherently per-model: one wedged model should
    // surface even while others are idle. We report the WORST case.
    let depth = 0;
    let limit: number | null = null;
    let saturated = false;
    for (const sessionRegistry of deps.registry.listSessionRegistries()) {
      const registryDepth = sessionRegistry.queueDepth;
      if (registryDepth > depth) depth = registryDepth;
      const registryLimit = sessionRegistry.queueDepthLimit;
      if (registryLimit !== undefined) {
        if (limit === null || registryLimit < limit) limit = registryLimit;
        if (registryDepth >= registryLimit) saturated = true;
      }
    }

    const writerActive = deps.modelWorkCoordinator?.writerActive ?? false;
    const waitingWriters = deps.modelWorkCoordinator?.waitingWriters ?? 0;
    const lastLoad = deps.modelWorkCoordinator?.lastLoad ?? null;
    const inFlight = deps.idleSweeper?.inFlight ?? 0;
    const drainPending = deps.idleSweeper?.isPending ?? false;

    const status = deriveHealthStatus({
      writerActive,
      waitingWriters,
      inFlight,
      residentModelCount: resident.length,
      queueSaturated: saturated,
      lastLoad,
    });

    return {
      status,
      // Clamped: a backwards clock step (NTP, fake timers in a sibling test)
      // must not surface a negative uptime a supervisor might read as a wrap.
      uptimeMs: Math.max(0, now() - startedAt),
      pid: process.pid,
      models: { resident, count: resident.length },
      work: { inFlight, drainPending, writerActive, waitingWriters },
      queue: { depth, limit, saturated },
      lastLoad,
    };
  };
}
