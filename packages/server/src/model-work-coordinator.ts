import type { ModelLoadRecord } from './health.js';
import type { PreDispatchAdmission, SessionRegistry } from './session-registry.js';

/**
 * Render a thrown value for {@link ModelLoadRecord.error}.
 *
 * Deliberately avoids `String(unknown)`: a rejection carrying a plain object
 * would render as the useless `[object Object]` in the one field a supervisor
 * reads to find out why the model would not load.
 */
function describeLoadFailure(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  if (error == null) return 'unknown error';
  try {
    return JSON.stringify(error) ?? 'unknown error';
  } catch {
    // Circular structure, or a `toJSON` that throws.
    return 'unknown error';
  }
}

/**
 * Result of a `withModelLoad` call that surfaces who actually drove the load
 * vs. who merely parked behind one that was already in flight. Callers use
 * this to split observability between the request that triggered a cold
 * weight-materialize from one that arrived a millisecond later and merely
 * inherited the wait — without the split a 60-second cold-load shows up
 * on every concurrent request as if each one paid for a separate load.
 *
 * `owner` reflects the SYNCHRONOUS state observed at lock acquisition:
 * `true` if the writer lock was free when this caller arrived and the
 * caller itself executed the supplied `fn`; `false` if there was already
 * a writer active (or queued ahead of this caller) when it arrived.
 *
 * `waitMs` and `ownMs` partition the wall-clock interval between when
 * the caller arrived at the coordinator and when its `fn` resolved:
 *  - `waitMs` is time spent blocked inside `acquireWrite()` (zero for a
 *    no-contention owner; ≈ peer's load duration for a follower).
 *  - `ownMs` is time spent inside `fn` once the writer lock was held
 *    (≈ load duration for an owner driving a cold load; near-zero for a
 *    follower whose `fn` is a no-op cache lookup).
 * Both are measured from `Date.now()` and clamped at zero to absorb
 * monotonic-skew. Their sum equals the total elapsed time in the call,
 * so handlers can plumb them into separate observability fields
 * (`server_load_wait_ms` vs. `server_model_resolve_ms`) without
 * double-counting.
 */
export interface ModelLoadOutcome<T> {
  result: T;
  owner: boolean;
  waitMs: number;
  ownMs: number;
}

/** A bounded request admission transferred into the resident FIFO budget. */
export interface ModelLoadAdmission {
  /**
   * Atomically move this request from the cold-load budget into the
   * resident model's ordinary pre-dispatch budget. Idempotent for the same
   * registry; transferring after a hot-swap releases the old reservation
   * before charging the new binding.
   */
  transferToResident(registry: SessionRegistry): PreDispatchAdmission;
  /** Idempotently release this request's current cold or resident unit. */
  release(): void;
}

interface ModelLoadAdmissionState {
  released: boolean;
  coldCounted: boolean;
  residentRegistry?: SessionRegistry;
  residentAdmission?: PreDispatchAdmission;
  transferError?: unknown;
}

/** Raised synchronously when unresolved model-load traffic is over capacity. */
export class ModelLoadQueueFullError extends Error {
  readonly admissionFootprint: number;
  readonly limit: number;

  constructor(admissionFootprint: number, limit: number) {
    super(`Model load queue full: admission footprint ${admissionFootprint} (waiter limit ${limit})`);
    this.name = 'ModelLoadQueueFullError';
    this.admissionFootprint = admissionFootprint;
    this.limit = limit;
  }
}

/**
 * Process-local gate for native MLX work.
 *
 * Individual model instances already have a per-model execution mutex, but a
 * lazy `loadModel()` can still run load-time materialization / warmup Metal
 * work while another model is decoding. MLX's allocator and command queues are
 * process-wide, so model load/swap takes an exclusive writer slot; inference
 * takes shared reader slots.
 */
export class ModelWorkCoordinator {
  private activeReaders = 0;
  private writerHeld = false;
  private queuedWriters = 0;
  private readonly readerWaiters: Array<() => void> = [];
  private readonly writerWaiters: Array<() => void> = [];
  private requestLoadAdmissions = 0;
  /** Outstanding permits grouped by the requested model name. */
  private readonly requestLoadAdmissionsByModel = new Map<string, Set<ModelLoadAdmissionState>>();
  /**
   * Resident bindings published synchronously by `ModelRegistry.register`.
   * Publishing transfers every already-admitted cold request before another
   * request can spend the resident budget independently.
   */
  private readonly residentRegistriesByModel = new Map<string, SessionRegistry>();
  private readonly maxRequestLoadQueueDepth: number | undefined;
  /**
   * Most recent settled load bracket. Retained here because the coordinator
   * is the ONE place that brackets every load: a `resolveModel` failure in
   * `/v1/messages` becomes a 500 and is otherwise dropped on the floor, so a
   * supervisor polling `/health` afterwards had no way to learn what went
   * wrong. See {@link ModelLoadRecord} for the "successful no-op overwrites
   * an earlier failure" caveat.
   */
  private lastLoadRecord: ModelLoadRecord | null = null;

  constructor(maxRequestLoadQueueDepth?: number) {
    this.maxRequestLoadQueueDepth = maxRequestLoadQueueDepth;
  }

  /**
   * Bound requests that arrive before a model has a resident
   * `SessionRegistry`. The capacity mirrors the resident FIFO: `limit`
   * waiters plus one owner/runner. Callers retain the permit through every
   * pre-lock await; registration synchronously transfers it into the resident
   * `SessionRegistry` budget before any later arrival can spend that capacity.
   */
  beginRequestLoadAdmission(modelId: string): ModelLoadAdmission {
    const limit = this.maxRequestLoadQueueDepth;
    const admissions = this.requestLoadAdmissionsByModel.get(modelId) ?? new Set<ModelLoadAdmissionState>();
    let coldFootprint = 0;
    for (const admission of admissions) {
      if (!admission.released && admission.coldCounted) coldFootprint += 1;
    }
    if (limit !== undefined && coldFootprint >= limit + 1) {
      throw new ModelLoadQueueFullError(coldFootprint, limit);
    }
    if (!this.requestLoadAdmissionsByModel.has(modelId)) {
      this.requestLoadAdmissionsByModel.set(modelId, admissions);
    }
    const state: ModelLoadAdmissionState = {
      released: false,
      coldCounted: true,
    };
    admissions.add(state);
    this.requestLoadAdmissions += 1;
    const admission: ModelLoadAdmission = {
      transferToResident: (registry): PreDispatchAdmission => this.transferAdmission(state, registry),
      release: (): void => {
        if (state.released) return;
        state.released = true;
        state.residentAdmission?.release();
        state.residentAdmission = undefined;
        state.residentRegistry = undefined;
        if (state.coldCounted) {
          state.coldCounted = false;
          this.requestLoadAdmissions -= 1;
          if (this.requestLoadAdmissions < 0) this.requestLoadAdmissions = 0;
        }
        const active = this.requestLoadAdmissionsByModel.get(modelId);
        active?.delete(state);
        if (active?.size === 0) this.requestLoadAdmissionsByModel.delete(modelId);
      },
    };
    const resident = this.residentRegistriesByModel.get(modelId);
    if (resident) {
      try {
        this.transferAdmission(state, resident);
      } catch {
        // Preserve the failure on `state`; the handler observes it from its
        // explicit transfer after resolving the binding and returns 429.
      }
    }
    return admission;
  }

  /**
   * Publish a requested model name's resident admission lane. Called
   * synchronously from `ModelRegistry.register`, so every outstanding cold
   * permit moves into the registry before a later arrival can be admitted
   * against an apparently empty resident budget.
   */
  bindRequestLoadAdmissions(modelId: string, registry: SessionRegistry): void {
    this.residentRegistriesByModel.set(modelId, registry);
    const admissions = this.requestLoadAdmissionsByModel.get(modelId);
    if (!admissions) return;
    for (const admission of admissions) {
      if (admission.released) continue;
      try {
        this.transferAdmission(admission, registry);
      } catch {
        // A cold request that cannot fit after an alias/hot-swap transition is
        // marked fail-closed. Its handler will surface the stored QueueFullError
        // rather than running outside either budget.
      }
    }
  }

  /** Forget a name only when it still points at the supplied binding. */
  unbindRequestLoadAdmissions(modelId: string, registry: SessionRegistry): void {
    if (this.residentRegistriesByModel.get(modelId) === registry) {
      this.residentRegistriesByModel.delete(modelId);
    }
  }

  private transferAdmission(state: ModelLoadAdmissionState, registry: SessionRegistry): PreDispatchAdmission {
    if (state.released) {
      throw new Error('Model load admission has already been released');
    }
    if (state.residentRegistry === registry && state.transferError !== undefined) {
      throw state.transferError;
    }
    if (state.residentRegistry === registry && state.residentAdmission) {
      return state.residentAdmission;
    }
    if (state.residentAdmission) {
      state.residentAdmission.release();
      state.residentAdmission = undefined;
      state.residentRegistry = undefined;
    }
    state.transferError = undefined;
    try {
      const residentAdmission = registry.beginPreDispatchAdmission();
      state.residentRegistry = registry;
      state.residentAdmission = residentAdmission;
      if (state.coldCounted) {
        state.coldCounted = false;
        this.requestLoadAdmissions -= 1;
        if (this.requestLoadAdmissions < 0) this.requestLoadAdmissions = 0;
      }
      return residentAdmission;
    } catch (error) {
      // This permit is now a rejected resident transition, not an invisible
      // cold unit. Remove its cold charge; `transferToResident` rethrows the
      // exact resident error when the owning request resumes.
      if (state.coldCounted) {
        state.coldCounted = false;
        this.requestLoadAdmissions -= 1;
        if (this.requestLoadAdmissions < 0) this.requestLoadAdmissions = 0;
      }
      state.transferError = error;
      state.residentRegistry = registry;
      throw error;
    }
  }

  /** Read-only unresolved/pre-FIFO request footprint for diagnostics/tests. */
  get requestLoadAdmissionCount(): number {
    return this.requestLoadAdmissions;
  }

  /** Read-only: `true` while a load holds the exclusive writer slot. */
  get writerActive(): boolean {
    return this.writerHeld;
  }

  /** Read-only: loads parked in `acquireWrite()` waiting for the slot. */
  get waitingWriters(): number {
    return this.queuedWriters;
  }

  /** Read-only: outcome of the most recent settled load bracket, or `null`. */
  get lastLoad(): ModelLoadRecord | null {
    return this.lastLoadRecord;
  }

  /**
   * Record a settled bracket. Called from the `finally` of both load
   * wrappers so a throw is captured just as reliably as a success.
   */
  private recordLoad(label: string | undefined, startedAt: number, error: unknown, ok: boolean): void {
    this.lastLoadRecord = {
      label: label ?? null,
      startedAt,
      finishedAt: Date.now(),
      ok,
      error: ok ? null : describeLoadFailure(error),
    };
  }

  /**
   * @param label Optional identifier (normally the model name) stamped into
   *   {@link lastLoad} so `/health` can name what was being loaded.
   */
  async withModelLoad<T>(fn: () => Promise<T> | T, label?: string): Promise<T> {
    await this.acquireWrite();
    // Measured from lock acquisition, not from arrival: `startedAt` is meant
    // to answer "how long has the actual materialization been running",
    // which is what a supervisor deciding whether to wait needs.
    const startedAt = Date.now();
    let ok = false;
    let failure: unknown;
    try {
      const result = await fn();
      ok = true;
      return result;
    } catch (err) {
      failure = err;
      throw err;
    } finally {
      this.recordLoad(label, startedAt, failure, ok);
      this.releaseWrite();
    }
  }

  /**
   * Like {@link withModelLoad} but reports whether THIS caller owned the
   * load (acquired the writer lock with no contention) or merely waited
   * behind a load that was already in flight when it arrived.
   *
   * Decided at sync-time before any await: if neither a writer is active
   * nor any writer is queued ahead, this caller is the owner; otherwise
   * it is parked behind someone else's load and `owner` is `false`. The
   * distinction is used by `/v1/messages` to split `resolve_ms` (own
   * load + lookup) from `load_wait_ms` (waiting on a peer's load) so a
   * 60-second cold-load does not look like 60 seconds of own work for
   * every concurrent request.
   */
  async withModelLoadInstrumented<T>(fn: () => Promise<T> | T, label?: string): Promise<ModelLoadOutcome<T>> {
    // `owner` MUST be decided synchronously, before any await, so the
    // signal reflects coordinator state at arrival rather than after
    // any peer transition. The wait/own split is measured around the
    // actual phase boundaries (lock acquisition, fn completion) so the
    // two intervals partition cleanly instead of both reporting total
    // elapsed time — see `ModelLoadOutcome` for the contract.
    const owner = !this.writerHeld && this.queuedWriters === 0;
    const arrivedAt = Date.now();
    await this.acquireWrite();
    const lockAcquiredAt = Date.now();
    let ok = false;
    let failure: unknown;
    try {
      const result = await fn();
      ok = true;
      const fnDoneAt = Date.now();
      const waitMs = Math.max(0, lockAcquiredAt - arrivedAt);
      const ownMs = Math.max(0, fnDoneAt - lockAcquiredAt);
      return { result, owner, waitMs, ownMs };
    } catch (err) {
      failure = err;
      throw err;
    } finally {
      this.recordLoad(label, lockAcquiredAt, failure, ok);
      this.releaseWrite();
    }
  }

  async withInference<T>(fn: () => Promise<T> | T): Promise<T> {
    await this.acquireRead();
    try {
      return await fn();
    } finally {
      this.releaseRead();
    }
  }

  private acquireRead(): Promise<void> {
    if (!this.writerHeld && this.queuedWriters === 0) {
      this.activeReaders += 1;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.readerWaiters.push(() => {
        this.activeReaders += 1;
        resolve();
      });
    });
  }

  private acquireWrite(): Promise<void> {
    this.queuedWriters += 1;
    if (!this.writerHeld && this.activeReaders === 0) {
      this.queuedWriters -= 1;
      this.writerHeld = true;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.writerWaiters.push(() => {
        this.queuedWriters -= 1;
        this.writerHeld = true;
        resolve();
      });
    });
  }

  private releaseRead(): void {
    this.activeReaders -= 1;
    if (this.activeReaders < 0) this.activeReaders = 0;
    if (this.activeReaders === 0) this.drain();
  }

  private releaseWrite(): void {
    this.writerHeld = false;
    this.drain();
  }

  private drain(): void {
    if (this.writerHeld) return;
    if (this.activeReaders === 0 && this.writerWaiters.length > 0) {
      this.writerWaiters.shift()?.();
      return;
    }
    if (this.queuedWriters === 0 && this.readerWaiters.length > 0) {
      const readers = this.readerWaiters.splice(0);
      for (const resolve of readers) resolve();
    }
  }
}
