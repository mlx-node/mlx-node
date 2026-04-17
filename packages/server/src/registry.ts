/**
 * ModelRegistry -- maps friendly model names to loaded model instances.
 *
 * All models exposing the chat-session surface (see `SessionCapableModel`
 * from `@mlx-node/lm`) are eligible for serving. Every registered model
 * is paired with a `SessionRegistry` — an LRU+TTL cache of live
 * `ChatSession` instances keyed by server-allocated response id.
 *
 * **Model-instance identity, not name.** Session registries are keyed
 * by MODEL OBJECT identity. The single-warm-session invariant
 * enforced by `SessionRegistry` is a property of the underlying
 * `SessionCapableModel` (one shared native KV cache per instance), so
 * registering the SAME model object under two names MUST yield the
 * SAME `SessionRegistry` — otherwise each alias's local single-warm
 * cache would hand out warm wrappers while the other alias silently
 * stomps them via the shared native state. `register()` looks the
 * model up in an identity-keyed map and reuses the existing registry
 * on alias, or allocates a fresh one on first sight.
 *
 * **Monotonic per-instance ids.** Every distinct model object gets a
 * monotonic `instanceId` on first registration, reused across aliases,
 * and dropped when the binding is fully torn down. The responses
 * endpoint persists this id alongside each stored record and, on a
 * `previous_response_id` continuation, compares the stored id against
 * the live id for `body.model`. This closes two holes a friendly-name
 * check leaves open:
 *
 *  1. A name hot-swap — `register("foo", modelA)` then
 *     `register("foo", modelB)` — would pass a string check, so a
 *     chain produced by `modelA` could be silently replayed through
 *     `modelB`'s tokenizer / chat template / KV layout. With instance
 *     ids the stored id (modelA's) no longer matches the live id
 *     (modelB's) and the continuation is rejected with 400.
 *  2. Two NAMES aliasing the SAME model object would be spuriously
 *     rejected by a string check comparing the stored name against
 *     `body.model`. Instance ids recognise them as the same binding
 *     and the continuation is accepted.
 */

import type { SessionCapableModel } from '@mlx-node/lm';

import { SessionRegistry } from './session-registry.js';

/** Minimal contract for a model that can be served via chat sessions. */
export type ServableModel = SessionCapableModel;

/** Model entry stored in the registry. */
export interface ModelEntry {
  id: string;
  model: ServableModel;
  createdAt: number;
  /**
   * Per-model-instance session cache, shared across every name that
   * points at this exact model object. See the module-level rustdoc.
   */
  sessionRegistry: SessionRegistry;
}

/**
 * Refcounted binding between a `ServableModel` and its shared
 * `SessionRegistry`. One binding per distinct model object currently
 * referenced by at least one registered name.
 *
 * - `refCount` tracks how many names point at this binding so
 *   `unregister()` can drop it once the last alias goes away.
 * - `inFlight` tracks dispatches currently holding the binding via
 *   `acquireDispatchLease()`. Teardown must not tear the binding
 *   down while any lease is held — otherwise an unregister +
 *   re-register of the SAME model mid-dispatch would allocate a
 *   FRESH `SessionRegistry` with an empty `execLock` chain and
 *   concurrent requests would race on the same native model.
 *   Teardown is deferred via `pendingTeardown`; a `register()` that
 *   sees the flag clears it and reuses the still-live binding so
 *   the fresh request's mutex chain serializes behind the in-flight
 *   dispatch on one shared `execLock`.
 * - `pendingPersists` tracks post-commit persist writes still
 *   in-flight under this binding's instance identity. Orthogonal to
 *   `inFlight` so the dispatch lease can be released eagerly once
 *   `withExclusive` returns (a wedged `store.store(...)` cannot pin
 *   the request's abort listeners / lease) while the binding's
 *   `modelInstanceId` still stays valid until every row it stamped
 *   has durably landed. `finalizeBindingTeardown` requires
 *   `pendingPersists === 0` so a same-model unregister + re-register
 *   during a slow persist cannot mint a fresh `modelInstanceId` that
 *   would invalidate the row the persist is about to land.
 */
interface SessionRegistryBinding {
  registry: SessionRegistry;
  refCount: number;
  inFlight: number;
  pendingPersists: number;
  pendingTeardown: boolean;
}

/**
 * Constructor options for {@link ModelRegistry}.
 */
export interface ModelRegistryOptions {
  /**
   * Maximum queue depth (waiters-only) per-model for the session
   * registry's execution mutex. Forwarded into every
   * `SessionRegistry` this registry allocates. See
   * {@link SessionRegistryOptions.maxQueueDepth}. Default: `undefined`
   * (unbounded — current behaviour).
   */
  maxQueueDepth?: number;
}

export class ModelRegistry {
  private readonly maxQueueDepth: number | undefined;
  private readonly models = new Map<string, ModelEntry>();
  /**
   * Identity-keyed (WeakMap semantics, but strong refs because the
   * registry already holds the model through its ModelEntry) map
   * from a model instance to its shared `SessionRegistry` binding.
   * Every name that references the same model object resolves to
   * the same binding — an alias of a registered model shares its
   * session cache and therefore its single-warm invariant.
   */
  private readonly sessionRegistriesByModel = new Map<ServableModel, SessionRegistryBinding>();
  /**
   * Identity-keyed map from a model instance to its monotonic
   * instance id. Entries are allocated on first registration,
   * reused across aliasing, and dropped when the last binding
   * releases (mirrors `sessionRegistriesByModel` lifetime exactly).
   */
  private readonly instanceIds = new Map<ServableModel, number>();
  /** Monotonic counter for `instanceIds`. Never reused. */
  private nextInstanceId = 1;
  /**
   * Tombstone map for instance ids retired by the hard-timeout
   * breaker. When the responses endpoint force-releases a wedged
   * persist's `retainBinding`, the breaker calls
   * `retireInstanceIdForForceRelease(model)` BEFORE dropping the
   * retain so the live id (already stamped into the pending record)
   * is preserved here. A subsequent `register()` of the SAME model
   * object that arrives AFTER the binding has fully torn down
   * inherits the retired id instead of minting a fresh one — so a
   * late-landing persist's row stays chainable. A true hot-swap
   * (different model object) has no tombstone for the new model,
   * so a fresh id is minted and the stale stored row is correctly
   * rejected with 400 instance-mismatch.
   *
   * `WeakMap`-keyed on the model object so entries do not keep the
   * model alive; if the model is GC'd the tombstone is cleaned up
   * automatically.
   *
   * Lifetime is refcounted: store ONE `{ instanceId, outstandingCount }`
   * entry per model. `retireInstanceIdForForceRelease` increments
   * (creating the entry on first retire); `releaseTombstone`
   * decrements and drops the entry when count hits zero. Because
   * `register()` inherits the retired id whenever the tombstone
   * exists, concurrent breakers on the same model all target the
   * SAME numeric `instanceId` — one shared refcount keeps the
   * tombstone alive as long as ANY pending persist still needs it,
   * and memory is bounded at O(1) per model regardless of how many
   * hard-timeouts have fired.
   */
  private readonly retiredInstanceIds = new WeakMap<ServableModel, { instanceId: number; outstandingCount: number }>();

  constructor(opts?: ModelRegistryOptions) {
    this.maxQueueDepth = opts?.maxQueueDepth;
  }

  /**
   * Register a model under a given name.
   *
   * If the name is already registered and the new model is a
   * DIFFERENT instance, the old binding's refcount is decremented
   * (and dropped if no other alias references it) before the new
   * binding is taken. Re-registering with the SAME model instance
   * leaves the binding unchanged.
   *
   * On first sight of a model object a fresh `SessionRegistry` is
   * allocated. On alias the existing registry is reused so the
   * single-warm invariant spans both names.
   *
   * Tombstone-inherit path: if the binding was previously torn down
   * AND the hard-timeout breaker called
   * `retireInstanceIdForForceRelease(model)` before teardown fired,
   * the fresh binding inherits the retired instance id from
   * `retiredInstanceIds` — so a late-landing persist's record stays
   * chainable. A hot-swap (different model object) has no tombstone,
   * so a fresh id is minted and the stale record fails
   * `previous_response_id` with 400. The aliasing path naturally
   * preserves the id because `instanceIds.has(model)` is already
   * true.
   */
  register(name: string, model: ServableModel): void {
    const existing = this.models.get(name);
    if (existing && existing.model === model) {
      // Same name + same model object: leave the binding and refcount
      // alone. Refresh createdAt so `/v1/models` surfaces the most
      // recent registration time.
      existing.createdAt = Math.floor(Date.now() / 1000);
      return;
    }
    if (existing) {
      // Same name, different model: release the old model's refcount
      // before installing the new binding.
      this.dropNameReference(existing.model);
    }

    // Look up or allocate the shared binding. If it is still alive
    // but flagged `pendingTeardown`, clear the flag and reuse it —
    // the fresh registration revives the binding before teardown
    // runs so the shared `SessionRegistry` / mutex chain stays
    // identical and any new dispatch serializes behind the
    // in-flight one.
    let binding = this.sessionRegistriesByModel.get(model);
    if (!binding) {
      binding = {
        registry: new SessionRegistry({ model, maxQueueDepth: this.maxQueueDepth }),
        refCount: 0,
        inFlight: 0,
        pendingPersists: 0,
        pendingTeardown: false,
      };
      this.sessionRegistriesByModel.set(model, binding);
    } else if (binding.pendingTeardown) {
      binding.pendingTeardown = false;
    }
    binding.refCount += 1;
    // Allocate a fresh monotonic instance id on first sight of this
    // model object; reuse the existing id on every alias thereafter.
    // Id lifetime mirrors the binding's — see `finalizeBindingTeardown`.
    // If the binding was fully torn down but the hard-timeout breaker
    // retired the previous id for the same model object, inherit it
    // from the tombstone instead of minting fresh.
    if (!this.instanceIds.has(model)) {
      // Tombstone is refcounted; we do NOT decrement here — the
      // still-pending persists own the outstanding count and balance
      // it via `releaseTombstone` in their own `.finally(...)`.
      const tombstone = this.retiredInstanceIds.get(model);
      if (tombstone) {
        this.instanceIds.set(model, tombstone.instanceId);
      } else {
        this.instanceIds.set(model, this.nextInstanceId);
        this.nextInstanceId += 1;
      }
    }

    this.models.set(name, {
      id: name,
      model,
      createdAt: Math.floor(Date.now() / 1000),
      sessionRegistry: binding.registry,
    });
  }

  /**
   * Unregister a model by name.
   *
   * Drops the name -> ModelEntry mapping and decrements the shared
   * session-registry binding's refcount. When the refcount hits zero
   * (no other alias references this model object) the binding — and
   * the `SessionRegistry` it owns — is dropped entirely so cached
   * sessions for the now-unreferenced model are released.
   *
   * @returns true if the model was removed.
   */
  unregister(name: string): boolean {
    const entry = this.models.get(name);
    if (!entry) return false;
    this.models.delete(name);
    this.dropNameReference(entry.model);
    return true;
  }

  /**
   * Decrement the refcount on a model binding; drop it at zero iff
   * no dispatch holds a lease AND no post-commit persist is still
   * retaining it. When either counter is non-zero the teardown is
   * deferred via `pendingTeardown` so the `SessionRegistry` (and
   * its `execLock` FIFO) stays alive until the last holder releases.
   *
   * A concurrent `register(sameModel)` between `dropNameReference()`
   * and the final release clears `pendingTeardown` and reuses the
   * still-live binding, preserving `modelInstanceId` so any row the
   * pending persist is about to land still resolves to a live id
   * when the next continuation arrives.
   */
  private dropNameReference(model: ServableModel): void {
    const binding = this.sessionRegistriesByModel.get(model);
    if (!binding) return;
    binding.refCount -= 1;
    if (binding.refCount <= 0) {
      if (binding.inFlight > 0 || binding.pendingPersists > 0) {
        // Defer until the last lease AND the last persist retention
        // drop. The binding and its instance id stay in the maps so
        // a same-object re-registration before finalisation can
        // revive it in place.
        binding.pendingTeardown = true;
        return;
      }
      this.finalizeBindingTeardown(model);
    }
  }

  /**
   * Drop `model`'s binding and instance id from the registry.
   * Shared teardown step; a subsequent re-registration usually mints
   * a FRESH instance id — once the last alias, lease, and persist
   * retention all release, any previously stored record referencing
   * this id belongs to a logically dead binding and a continuation
   * against it must fall through to the `currentInstanceId === undefined`
   * rejection path so the stale chain cannot be replayed.
   *
   * Tombstone exception: if the hard-timeout breaker retired the
   * previous id via `retireInstanceIdForForceRelease` before the
   * forced release, a subsequent same-object `register()` inherits
   * the retired id from `retiredInstanceIds` instead of minting
   * fresh — this preserves chain continuity for late-landing
   * persists that crossed the safety breaker.
   */
  private finalizeBindingTeardown(model: ServableModel): void {
    this.sessionRegistriesByModel.delete(model);
    this.instanceIds.delete(model);
  }

  /**
   * Acquire a dispatch lease on the session registry bound to `name`.
   * Returns the live `SessionRegistry` and the binding's instance id,
   * or `undefined` if the name is not registered. Every successful
   * acquisition MUST be balanced with exactly one
   * `releaseDispatchLease(model)` call (typically via try/finally).
   *
   * The lease keeps the binding alive past a concurrent
   * `unregister()` / `register(differentModel)` sequence: the
   * `SessionRegistry` and its `execLock` FIFO remain valid while any
   * lease is outstanding, so a newly registered same-model alias
   * will rebind to the SAME registry and its `withExclusive` will
   * serialize behind the in-flight dispatch.
   *
   * The returned `model` handle is what the caller passes to
   * `releaseDispatchLease()` — the lease binds to the model OBJECT
   * (not the friendly name) because the name can be hot-swapped
   * while the lease is held.
   */
  acquireDispatchLease(
    name: string,
  ): { model: ServableModel; registry: SessionRegistry; instanceId: number } | undefined {
    const entry = this.models.get(name);
    if (!entry) return undefined;
    const binding = this.sessionRegistriesByModel.get(entry.model);
    if (!binding) return undefined;
    const instanceId = this.instanceIds.get(entry.model);
    if (instanceId === undefined) return undefined;
    binding.inFlight += 1;
    return { model: entry.model, registry: binding.registry, instanceId };
  }

  /**
   * Release a dispatch lease previously obtained via
   * `acquireDispatchLease()`. Decrements the binding's in-flight
   * counter and, if the binding has been flagged for teardown (its
   * refcount hit zero while the lease was held), drops it once the
   * last lease releases. Safe to call exactly once per acquired
   * lease; calling it on a model whose binding has already been
   * fully torn down is a no-op.
   */
  releaseDispatchLease(model: ServableModel): void {
    const binding = this.sessionRegistriesByModel.get(model);
    if (!binding) return;
    binding.inFlight -= 1;
    if (binding.inFlight < 0) binding.inFlight = 0;
    if (binding.pendingTeardown && binding.refCount <= 0 && binding.inFlight === 0 && binding.pendingPersists === 0) {
      this.finalizeBindingTeardown(model);
    }
  }

  /**
   * Retain the binding for the duration of a post-commit persist.
   *
   * The responses endpoint starts `store.store(record)` synchronously
   * inside `withExclusive` so the pending-writes tracker observes
   * the in-flight write before the mutex releases, but does NOT
   * await it on the critical path. The write still carries the
   * binding's `modelInstanceId` (stamped into `configJson` by
   * `buildResponseRecord`); without a retention, a same-model
   * unregister + re-register completing while the write is in flight
   * would delete the instance id and the re-registration would mint
   * a fresh one, so the row — when it finally lands — would
   * reference a dead id and the next continuation would be rejected
   * with 400 instance-mismatch.
   *
   * The retention counter is CHECKED in every teardown gate
   * (`dropNameReference`, `releaseDispatchLease`, `releaseBinding`).
   * It is orthogonal to `inFlight` so the dispatch lease can release
   * eagerly after `withExclusive` returns while the binding stays
   * pinned long enough for the backgrounded `store.store(...)` to
   * settle.
   *
   * Safe to call on a model whose binding has already been torn down
   * (no-op). The matching `releaseBinding(model)` MUST still run in
   * the persist's `.finally(...)` so the counter stays balanced.
   */
  retainBinding(model: ServableModel): void {
    const binding = this.sessionRegistriesByModel.get(model);
    if (!binding) return;
    binding.pendingPersists += 1;
  }

  /**
   * Balance a prior `retainBinding()` call. Decrements the persist
   * retention counter and, if the binding has been flagged for
   * teardown (refcount hit zero while the retention was held AND
   * every dispatch lease has already released), drops it once the
   * last retention releases. Safe to call exactly once per retain;
   * calling it on a model whose binding has already been fully torn
   * down is a no-op.
   */
  releaseBinding(model: ServableModel): void {
    const binding = this.sessionRegistriesByModel.get(model);
    if (!binding) return;
    binding.pendingPersists -= 1;
    if (binding.pendingPersists < 0) binding.pendingPersists = 0;
    if (binding.pendingTeardown && binding.refCount <= 0 && binding.inFlight === 0 && binding.pendingPersists === 0) {
      this.finalizeBindingTeardown(model);
    }
  }

  /**
   * Tombstone installer, invoked exclusively by the responses
   * endpoint's hard-timeout breaker (see
   * `getPostCommitPersistHardTimeoutMs` in `endpoints/responses.ts`)
   * when it force-releases the `retainBinding` on a wedged persist.
   * Must be called BEFORE the idempotent
   * `persistRetainBox.release?.()` so `instanceIds.get(model)`
   * still returns the live id that the already-stamped record
   * carries.
   *
   * Returns the retired id so the caller can capture it and scope
   * the tombstone's lifetime to the specific wedged persist via
   * `releaseTombstone(model)` inside that persist's `.finally(...)`.
   * Returns `undefined` when the model has no current instance id
   * assignment (caller raced the natural teardown path).
   *
   * Refcounted: each call increments a shared
   * `{ instanceId, outstandingCount }` entry per model. Overlapping
   * breakers share one slot (they all target the same numeric id
   * because `register()` inherits the retired id whenever the
   * tombstone is present), so memory stays O(1) per model.
   */
  retireInstanceIdForForceRelease(model: ServableModel): { instanceId: number } | undefined {
    const id = this.instanceIds.get(model);
    if (id === undefined) return undefined;
    const existing = this.retiredInstanceIds.get(model);
    if (existing) {
      existing.outstandingCount += 1;
      return { instanceId: existing.instanceId };
    }
    this.retiredInstanceIds.set(model, { instanceId: id, outstandingCount: 1 });
    return { instanceId: id };
  }

  /**
   * Tombstone cleanup. Called from the post-commit persist's
   * `.finally(...)` to balance exactly one prior
   * `retireInstanceIdForForceRelease(model)` call. Decrements the
   * shared refcount and drops the entry at zero so the next natural
   * teardown mints a fresh id.
   *
   * Safe to call on a model whose tombstone has already been drained
   * (no-op). The counter is clamped non-negative so spurious
   * releases cannot underflow and re-enable inheritance.
   */
  releaseTombstone(model: ServableModel): void {
    const entry = this.retiredInstanceIds.get(model);
    if (!entry) return;
    entry.outstandingCount -= 1;
    if (entry.outstandingCount <= 0) {
      this.retiredInstanceIds.delete(model);
    }
  }

  /**
   * Retrieve a model instance by name.
   */
  get(name: string): ServableModel | undefined {
    return this.models.get(name)?.model;
  }

  /**
   * Retrieve the monotonic instance id for the model currently bound
   * to `name`, or `undefined` if the name isn't registered.
   *
   * Two names that alias the same model object return the SAME id
   * (they share a binding), and a name that has been hot-swapped to
   * a different model object returns a DIFFERENT id than before the
   * swap (the prior binding's id was dropped by `dropNameReference`
   * and a fresh id was minted for the new model on re-registration).
   *
   * The responses endpoint uses this to key the
   * `previous_response_id` cross-chain guard on instance identity
   * instead of friendly name, so hot swaps are caught and safe
   * aliases are not spuriously rejected.
   */
  getInstanceId(name: string): number | undefined {
    const entry = this.models.get(name);
    if (!entry) return undefined;
    return this.instanceIds.get(entry.model);
  }

  /**
   * Retrieve the session registry for a given model name, or
   * `undefined` if the name is not registered.
   *
   * Every name that points at the same model instance returns the
   * SAME `SessionRegistry` object. Two aliases `a` and `b` of one
   * model therefore satisfy
   * `registry.getSessionRegistry('a') === registry.getSessionRegistry('b')`,
   * which is what the single-warm invariant requires: any turn
   * through either alias advances the same cache's state, so a later
   * lookup via either alias sees the current warm wrapper (if
   * freshly adopted) or misses and cold-replays (if it was leased
   * out by the other alias) — never a stale wrapper pointing at
   * stomped native state.
   */
  getSessionRegistry(name: string): SessionRegistry | undefined {
    return this.models.get(name)?.sessionRegistry;
  }

  /**
   * Iterate every DISTINCT session registry currently in use.
   *
   * Two aliases of the same model share one `SessionRegistry`, so
   * naively walking every `ModelEntry` would yield duplicates. We
   * walk the identity-keyed bindings instead so each registry
   * appears exactly once, which is what the periodic `sweep()`
   * scheduler in `server.ts` needs to avoid redundantly sweeping
   * the same cache multiple times per tick.
   */
  listSessionRegistries(): SessionRegistry[] {
    const out: SessionRegistry[] = [];
    for (const binding of this.sessionRegistriesByModel.values()) {
      out.push(binding.registry);
    }
    return out;
  }

  /**
   * List all registered models in the OpenAI /v1/models format.
   */
  list(): { id: string; object: string; created: number; owned_by: string }[] {
    const result: { id: string; object: string; created: number; owned_by: string }[] = [];
    for (const entry of this.models.values()) {
      result.push({
        id: entry.id,
        object: 'model',
        created: entry.createdAt,
        owned_by: 'mlx-node',
      });
    }
    return result;
  }

  /**
   * Check whether a model supports streaming.
   *
   * Every `SessionCapableModel` structurally exposes
   * `chatStreamSessionStart`, so this is universally `true` for any
   * properly-typed model registered through the session-capable
   * interface. Kept as a belt-and-suspenders duck-type so a partially
   * stubbed test double (pre-migration or intentionally non-streaming)
   * can still opt out by omitting the method.
   */
  hasStreamSupport(model: ServableModel): boolean {
    const fn = (model as unknown as Record<string, unknown>)['chatStreamSessionStart'];
    return typeof fn === 'function';
  }
}
