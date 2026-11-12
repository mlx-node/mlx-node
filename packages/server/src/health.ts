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

import type { MemoryGovernorStatus } from '@mlx-node/core';

import type { IdleSweeper } from './idle-sweeper.js';
import type { ModelWorkCoordinator } from './model-work-coordinator.js';
import type { ModelRegistry } from './registry.js';
import type { WeightResidency, WeightsSnapshot } from './weight-residency.js';

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
 * Resident HTTP requests bypass the load writer and do not overwrite this
 * record with a no-op. The `'error'` rung is nevertheless gated on "no
 * resident models": a server that is answering requests is not in an error
 * state regardless of what the last load bracket did.
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
    /** True while the weights' idle release is armed. */
    releasePending: boolean;
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
  /**
   * Per resident name with a load-time context: the window a default text
   * turn gets and the format of the K/V it writes (Splash `/status`).
   */
  kv: Record<string, ResidentKvStatus>;
  /**
   * The weights' idle release (Splash `/status` `weights`), when the server
   * can release them. Released weights with nothing resident still read `ok`:
   * the next request loads them.
   */
  weights?: WeightsSnapshot;
  /**
   * The memory governor (Splash `/status` `memory_governor`), while a model
   * on it is resident: as the last poll of the native status left it.
   */
  memory_governor?: MemoryGovernorHealth;
}

interface GapCauseHealth {
  gaps: number;
  ms: number;
}

/** Splash `/status` `scheduler`, one per loaded model. */
export interface SchedulerHealth {
  /** The resident name the model serves under, once it is known. */
  model?: string;
  model_id: number;
  queued: number;
  waiting_resources: number;
  waiting_prefix: number;
  prefilling: number;
  decoding: number;
  prefill_batches: number;
  prefill_rows: number;
  decode_batches: number;
  /** `b1`, `b2`, …: decode batches of that width. */
  decode_batches_by_width: Record<string, number>;
  decode_gap_target_ms: number;
  contended_prefill_ms: { p50: number; p95: number; p99: number; max: number; total: number; samples: number };
  prefill_units: number;
  bound_cuts: number;
  replans: number;
}

/** Splash `/status` `memory_governor`, in its keys. */
export interface MemoryGovernorHealth {
  limit: number;
  charged: number;
  headroom: number;
  pressure: string;
  denied_reservations: number;
  host: {
    measured: boolean;
    available: number;
    spare: number;
    spare_reserve: number;
    headroom: number;
    system_pressure: string;
    growth_allowed: boolean;
  };
  /** Sums to `charged`: "fixed" is what the other accounts leave. */
  accounts: { name: string; current: number }[];
  owed: { bytes: number; transfers: number };
  claims: { waiting: number; bytes: number; oldest_ms: number };
  resource_waits: {
    memory: number;
    concurrency: number;
    suspended: number;
    draining: boolean;
    held_behind_refusal: number;
    oldest_ms: number;
  };
  counters: {
    replans: number;
    resource_suspensions: number;
    resource_resumptions: number;
    resource_timeouts: number;
    capacity_exhausted: number;
    metal_out_of_memory: number;
    publications_refused: number;
  };
  ttft_parts_ms: { own_prefill: number; claim_wait: number; decode: number; other_prefill: number };
  decode_gap_causes: {
    prefill_unit: GapCauseHealth;
    claim_wait: GapCauseHealth;
    ssd_read: GapCauseHealth;
    pool_release: GapCauseHealth;
    replan: GapCauseHealth;
    other: GapCauseHealth;
    prefill_ms: number;
  };
  decode_gap_ms: { p50: number; p99: number; max: number };
  decisions: string[];
  scheduler: SchedulerHealth[];
  /** Splash's DFlash2 state counters and state cache, summed over models. */
  dflash2_states: DFlash2StatesHealth;
}

/** Splash's state publication counters and state-cache snapshot, in its keys. */
export interface DFlash2StatesHealth {
  checkpoint_publications: number;
  replay_state_publications: number;
  junction_materializations: number;
  checkpoint_publication_failures: number;
  replay_state_publication_failures: number;
  junction_materialization_failures: number;
  deduplicated_state_publications: number;
  recycled_state_publications: number;
  checkpoint_retirements: number;
  lazy_junctions: number;
  checkpoint_evictions: number;
  evictions: number;
  entries: number;
  bytes: number;
  checkpoint_entries: number;
  checkpoint_bytes: number;
  in_use: number;
}

/** The native status in `/health`'s keys. */
export function toMemoryGovernorHealth(status: MemoryGovernorStatus): MemoryGovernorHealth {
  const causes = status.decodeGapCauses;
  const states = status.dflash2States;
  return {
    limit: status.limit,
    charged: status.charged,
    headroom: status.headroom,
    pressure: status.pressure,
    denied_reservations: status.deniedReservations,
    host: {
      measured: status.host.measured,
      available: status.host.available,
      spare: status.host.spare,
      spare_reserve: status.host.spareReserve,
      headroom: status.host.headroom,
      system_pressure: status.host.systemPressure,
      growth_allowed: status.host.growthAllowed,
    },
    accounts: status.accounts.map((account) => ({ name: account.name, current: account.current })),
    owed: { bytes: status.owed.bytes, transfers: status.owed.transfers },
    claims: { waiting: status.claims.waiting, bytes: status.claims.bytes, oldest_ms: status.claims.oldestMs },
    resource_waits: {
      memory: status.resourceWaits.memory,
      concurrency: status.resourceWaits.concurrency,
      suspended: status.resourceWaits.suspended,
      draining: status.resourceWaits.draining,
      held_behind_refusal: status.resourceWaits.heldBehindRefusal,
      oldest_ms: status.resourceWaits.oldestMs,
    },
    counters: {
      replans: status.counters.replans,
      resource_suspensions: status.counters.resourceSuspensions,
      resource_resumptions: status.counters.resourceResumptions,
      resource_timeouts: status.counters.resourceTimeouts,
      capacity_exhausted: status.counters.capacityExhausted,
      metal_out_of_memory: status.counters.metalOutOfMemory,
      publications_refused: status.counters.publicationsRefused,
    },
    ttft_parts_ms: {
      own_prefill: status.ttftPartsMs.ownPrefill,
      claim_wait: status.ttftPartsMs.claimWait,
      decode: status.ttftPartsMs.decode,
      other_prefill: status.ttftPartsMs.otherPrefill,
    },
    decode_gap_causes: {
      prefill_unit: { ...causes.prefillUnit },
      claim_wait: { ...causes.claimWait },
      ssd_read: { ...causes.ssdRead },
      pool_release: { ...causes.poolRelease },
      replan: { ...causes.replan },
      other: { ...causes.other },
      prefill_ms: causes.prefillMs,
    },
    decode_gap_ms: { ...status.decodeGapMs },
    decisions: [...status.decisions],
    scheduler: status.scheduler.map((scheduler) => ({
      model_id: scheduler.modelId,
      queued: scheduler.queued,
      waiting_resources: scheduler.waitingResources,
      waiting_prefix: scheduler.waitingPrefix,
      prefilling: scheduler.prefilling,
      decoding: scheduler.decoding,
      prefill_batches: scheduler.prefillBatches,
      prefill_rows: scheduler.prefillRows,
      decode_batches: scheduler.decodeBatches,
      decode_batches_by_width: Object.fromEntries(
        scheduler.decodeBatchesByWidth.map((batches, index) => [`b${index + 1}`, batches]),
      ),
      decode_gap_target_ms: scheduler.decodeGapTargetMs,
      contended_prefill_ms: { ...scheduler.contendedPrefillMs },
      prefill_units: scheduler.prefillUnits,
      bound_cuts: scheduler.boundCuts,
      replans: scheduler.replans,
    })),
    dflash2_states: {
      checkpoint_publications: states.checkpointPublications,
      replay_state_publications: states.replayStatePublications,
      junction_materializations: states.junctionMaterializations,
      checkpoint_publication_failures: states.checkpointPublicationFailures,
      replay_state_publication_failures: states.replayStatePublicationFailures,
      junction_materialization_failures: states.junctionMaterializationFailures,
      deduplicated_state_publications: states.deduplicatedStatePublications,
      recycled_state_publications: states.recycledStatePublications,
      checkpoint_retirements: states.checkpointRetirements,
      lazy_junctions: states.lazyJunctions,
      checkpoint_evictions: states.checkpointEvictions,
      evictions: states.evictions,
      entries: states.entries,
      bytes: states.bytes,
      checkpoint_entries: states.checkpointEntries,
      checkpoint_bytes: states.checkpointBytes,
      in_use: states.inUse,
    },
  };
}

/** Splash's `/status` KV fields for one resident model. */
export interface ResidentKvStatus {
  maximum_context_tokens: number;
  format?: 'int8' | 'bf16';
  quantization?: 'symmetric_int8' | 'none';
  scale_type?: 'float32' | 'none';
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
  /** Optional: the weights' idle release behind `weights` / `releasePending`. */
  weights?: WeightResidency | null;
  /**
   * Optional: the memory governor's status as the server's poller last read
   * it (plain JS state; `null` while no model on it is resident).
   */
  memoryGovernor?: () => MemoryGovernorHealth | null;
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

    if (deps.registry.audio) {
      depth = Math.max(depth, deps.registry.audio.queueDepth, deps.registry.rerank.queueDepth);
      const audioLimit = deps.registry.queueDepthLimit;
      if (audioLimit !== undefined && (deps.registry.audio.pending || deps.registry.rerank.pending)) {
        limit = limit === null ? audioLimit : Math.min(limit, audioLimit);
        saturated ||= Math.max(deps.registry.audio.queueDepth, deps.registry.rerank.queueDepth) >= audioLimit;
      }
    }
    const writerActive = deps.modelWorkCoordinator?.writerActive ?? false;
    const waitingWriters = deps.modelWorkCoordinator?.waitingWriters ?? 0;
    const lastLoad = deps.modelWorkCoordinator?.lastLoad ?? null;
    const inFlight = deps.idleSweeper?.inFlight ?? 0;
    const drainPending = deps.idleSweeper?.isPending ?? false;

    const kv: Record<string, ResidentKvStatus> = {};
    for (const id of resident) {
      const served = deps.registry.servedContext(id);
      if (served === undefined) continue;
      const maximum_context_tokens = served.windowTokens;
      if (served.kvFormat === 'int8') {
        kv[id] = { maximum_context_tokens, format: 'int8', quantization: 'symmetric_int8', scale_type: 'float32' };
      } else if (served.kvFormat === 'bf16') {
        kv[id] = { maximum_context_tokens, format: 'bf16', quantization: 'none', scale_type: 'none' };
      } else {
        kv[id] = { maximum_context_tokens };
      }
    }

    const governed = new Map<number, string>();
    for (const id of resident) {
      const model = deps.registry.servedContext(id)?.governorModelId;
      if (model !== undefined && !governed.has(model)) governed.set(model, id);
    }
    const polled = deps.memoryGovernor?.() ?? null;
    const memoryGovernor =
      polled === null
        ? null
        : {
            ...polled,
            scheduler: polled.scheduler.map((scheduler) => {
              const model = governed.get(scheduler.model_id);
              return model === undefined ? scheduler : { model, ...scheduler };
            }),
          };
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
      work: { inFlight, drainPending, releasePending: deps.weights?.isPending ?? false, writerActive, waitingWriters },
      queue: { depth, limit, saturated },
      lastLoad,
      kv,
      ...(deps.weights ? { weights: deps.weights.snapshot() } : {}),
      ...(memoryGovernor !== null ? { memory_governor: memoryGovernor } : {}),
    };
  };
}
