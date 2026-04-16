/**
 * ModelRegistry -- maps friendly model names to loaded model instances.
 *
 * All models that expose the chat-session surface (see `SessionCapableModel`
 * from `@mlx-node/lm`) are eligible for serving. Every registered model is
 * paired with a `SessionRegistry` — an LRU+TTL cache of live
 * `ChatSession` instances keyed by the server-allocated response id. The
 * endpoint layer fetches the per-model session registry via
 * {@link ModelRegistry.getSessionRegistry} and routes each request through
 * a session looked up or allocated there.
 *
 * **Model-instance identity, not name.** The session registries are
 * keyed by MODEL OBJECT identity, not by friendly name. The
 * single-warm-session invariant enforced by `SessionRegistry` is a
 * property of the underlying `SessionCapableModel` (one shared native
 * KV cache per instance), so registering the SAME model object under
 * two names MUST yield the SAME `SessionRegistry` — otherwise each
 * alias's local single-warm cache would happily hand out warm
 * wrappers while the other alias silently stomps them via the
 * shared native state. `register()` therefore looks the model
 * object up in an identity-keyed map and reuses the existing
 * registry on alias, or allocates a fresh one on first sight.
 *
 * **Monotonic per-instance ids.** Every distinct model object the
 * registry sees is also assigned a fresh monotonic integer
 * `instanceId` on first registration, reused for every subsequent
 * registration of the same object, and dropped when the binding's
 * refcount falls to zero alongside the `SessionRegistry` itself.
 * The responses endpoint persists the current instance id alongside
 * each stored response record and, on a `previous_response_id`
 * continuation, compares the stored id against the live id for
 * `body.model`. This closes two holes that a friendly-name
 * comparison left open:
 *
 *  1. A name hot-swap — `register("foo", modelA)` followed by
 *     `register("foo", modelB)` — would pass a string check, so a
 *     chain produced by `modelA` could be silently replayed through
 *     `modelB`'s tokenizer / chat template / KV layout. With
 *     instance ids the stored id (modelA's) no longer matches the
 *     live id for `"foo"` (now modelB's) and the continuation is
 *     rejected with a 400.
 *  2. Two NAMES aliasing the SAME model object (already safe thanks
 *     to per-instance `SessionRegistry` sharing) would be spuriously
 *     rejected by a string check comparing the stored name against
 *     `body.model`. Instance ids recognise them as the same binding
 *     and the continuation is accepted.
 *
 * This interface intentionally mirrors `SessionCapableModel` one-to-one —
 * the server always drives models through `ChatSession<M>` wrappers, never
 * the low-level NAPI methods directly.
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
 * `SessionRegistry`. One binding exists per distinct model object
 * currently referenced by at least one registered name; `refCount`
 * tracks how many names currently point at it so `unregister()` can
 * drop the binding once the last alias goes away.
 *
 * `inFlight` tracks how many dispatches are currently holding this
 * binding through `acquireDispatchLease()`. The private name-reference
 * drop path MUST NOT tear the binding (and its `SessionRegistry`, and
 * therefore its FIFO `withExclusive` mutex chain) down while any
 * dispatch still holds a lease — if a caller unregister()s the name
 * mid-dispatch and then re-register()s the SAME model object, the map
 * lookup would otherwise allocate a FRESH `SessionRegistry` with an
 * empty `execLock` chain, and a new concurrent request would race
 * against the in-flight dispatch on the same underlying native model.
 * Teardown is deferred via `pendingTeardown` until the last lease
 * releases; meanwhile a `register()` that sees the pending flag
 * clears it, re-references the still-live binding, and the fresh
 * request's mutex chain naturally serializes behind the in-flight
 * dispatch because they share one `execLock`.
 *
 * `pendingPersists` tracks how many post-commit persist writes are
 * still in flight under this binding's instance identity. It is
 * INTENTIONALLY orthogonal to `inFlight` (dispatch exclusivity) so the
 * iter-39 optimisation — releasing the dispatch lease eagerly after
 * `withExclusive` returns, so a wedged `store.store(...)` cannot pin
 * the request's abort listeners / lease — can still coexist with the
 * iter-40 invariant that the binding's `modelInstanceId` stays valid
 * until every row it stamped has durably landed. When a dispatch
 * enters the post-commit persist window it calls `retainBinding()` to
 * bump this counter; its `.finally(...)` balances that with
 * `releaseBinding()`. `finalizeBindingTeardown` now additionally
 * requires `pendingPersists === 0`, so a same-model unregister +
 * re-register sequence that fires during a slow persist cannot mint
 * a fresh `modelInstanceId` that would invalidate the row the
 * persist is about to land — the row's stored identity still matches
 * the live id when the client's next `previous_response_id`
 * continuation arrives.
 */
interface SessionRegistryBinding {
  registry: SessionRegistry;
  refCount: number;
  inFlight: number;
  pendingPersists: number;
  pendingTeardown: boolean;
}

export class ModelRegistry {
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
   * Register a model under a given name.
   *
   * If a name is already registered and the new model is a DIFFERENT
   * instance, the old binding's refcount is decremented (and dropped
   * if no other alias references it) before the new binding is taken.
   * If a name is re-registered with the SAME model instance the
   * binding is unchanged — the refcount stays stable and no session
   * state is disturbed.
   *
   * When the model object has not been seen before a fresh
   * `SessionRegistry` is allocated for it. When it has (via an
   * existing alias) the existing registry is reused so the
   * single-warm invariant spans both names.
   */
  register(name: string, model: ServableModel): void {
    const existing = this.models.get(name);
    if (existing && existing.model === model) {
      // Re-register under the same name with the same model object —
      // leave the binding and its refcount alone. Refresh createdAt
      // so `/v1/models` surfaces the most recent registration time.
      existing.createdAt = Math.floor(Date.now() / 1000);
      return;
    }
    if (existing) {
      // Same name, different model: release the old model's refcount
      // before installing the new binding. If no other alias still
      // points at the old model, drop its registry entirely.
      this.dropNameReference(existing.model);
    }

    // Look up or allocate the shared binding for this model instance.
    // If the binding is still alive but flagged `pendingTeardown` (its
    // refcount previously hit zero while an in-flight dispatch still
    // held a lease), clear the flag and reuse it — the fresh
    // registration revives the binding before teardown runs, and the
    // shared `SessionRegistry` / mutex chain stays identical so any
    // new dispatch serializes behind the in-flight one instead of
    // racing on a freshly allocated registry.
    let binding = this.sessionRegistriesByModel.get(model);
    if (!binding) {
      binding = {
        registry: new SessionRegistry({ model }),
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
    // The id lifetime mirrors the binding's — see `finalizeBindingTeardown`.
    if (!this.instanceIds.has(model)) {
      this.instanceIds.set(model, this.nextInstanceId);
      this.nextInstanceId += 1;
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
   * Decrement the refcount on a model binding; drop it at zero IFF
   * no dispatch currently holds a lease on it AND no post-commit
   * persist is still retaining it. When either counter is non-zero
   * the teardown is deferred via `pendingTeardown` so the in-flight
   * dispatch's `SessionRegistry` (and therefore its `execLock` FIFO
   * mutex chain) stays alive until the last holder releases. Any
   * concurrent `register(sameModel)` that fires between
   * `dropNameReference()` and the final release will see the
   * still-present binding, clear `pendingTeardown`, and reuse it —
   * so the next request's `withExclusive` naturally serializes
   * behind the in-flight dispatch on one shared mutex, and the
   * binding's `modelInstanceId` is preserved (not re-minted) so
   * any stored row the pending persist is about to land still
   * resolves to a live id when the client's next continuation
   * references it. This is the iter-40 invariant that closes the
   * window where a same-model unregister + re-register during a
   * slow persist would otherwise have produced a row referencing
   * a dead instance id.
   */
  private dropNameReference(model: ServableModel): void {
    const binding = this.sessionRegistriesByModel.get(model);
    if (!binding) return;
    binding.refCount -= 1;
    if (binding.refCount <= 0) {
      if (binding.inFlight > 0 || binding.pendingPersists > 0) {
        // Defer teardown until the last dispatch lease releases AND
        // the last persist retention drops. The binding and its
        // instance id stay in the maps so a same-object
        // re-registration before finalisation can revive it in
        // place.
        binding.pendingTeardown = true;
        return;
      }
      this.finalizeBindingTeardown(model);
    }
  }

  /**
   * Drop `model`'s binding and instance id from the registry.
   * Internal teardown step shared by `dropNameReference()`,
   * `releaseDispatchLease()`, and `releaseBinding()`; see their doc
   * comments for the full lifecycle. On drop, a subsequent
   * re-registration of the SAME model object will mint a FRESH
   * instance id — intentional: once the last alias, the last
   * in-flight lease, AND the last post-commit persist retention
   * all release, any previously persisted stored record that
   * references this id belongs to a logically dead binding, and a
   * later `previous_response_id` continuation against it must fall
   * through to the `currentInstanceId === undefined` rejection path
   * so the stale chain cannot be replayed.
   */
  private finalizeBindingTeardown(model: ServableModel): void {
    this.sessionRegistriesByModel.delete(model);
    this.instanceIds.delete(model);
  }

  /**
   * Acquire a dispatch lease on the session registry bound to `name`.
   *
   * Returns the live `SessionRegistry` and the binding's current
   * instance id, or `undefined` if the name is not registered. The
   * caller MUST balance every successful acquisition with exactly one
   * `releaseDispatchLease(model)` call, regardless of whether the
   * dispatch body threw or returned normally — a typical pattern is
   * `try { ...withExclusive(...) } finally { release(...); }`.
   *
   * The lease keeps the binding alive past a concurrent
   * `unregister()`/`register(differentModel)` sequence: the
   * `SessionRegistry` object and its FIFO `execLock` mutex chain
   * remain valid as long as ANY lease is outstanding, so a newly
   * registered same-model alias that fires a concurrent request
   * will rebind to the SAME registry and its new `withExclusive`
   * will serialize behind the in-flight dispatch instead of racing
   * on a freshly allocated mutex.
   *
   * The returned `model` handle is what the caller passes to
   * `releaseDispatchLease()` — binding the lease to the model
   * OBJECT (not the friendly name) is deliberate: the name can be
   * hot-swapped while the lease is held, but the lease must still
   * release on the original binding.
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
   * The responses endpoint kicks off `store.store(record)` synchronously
   * inside `withExclusive` so the pending-writes tracker observes the
   * in-flight write before the per-model mutex releases. Iter-39
   * moved the eventual error-logging `await` OUT of the request's
   * critical path — so a wedged backend no longer pins abort
   * listeners or the dispatch lease — but the write still carries
   * the binding's `modelInstanceId` as stamped into `configJson`
   * by `buildResponseRecord`. If a same-model unregister +
   * re-register sequence were allowed to run to completion while
   * that write is still in flight, the binding's
   * `finalizeBindingTeardown` step would delete the instance id
   * from the registry, the re-registration would mint a FRESH id,
   * and the row — when it finally lands — would reference a dead
   * id. The very next `previous_response_id` continuation the
   * client issues would then be rejected with a 400
   * "instance-mismatch", even though the client saw a clean
   * `response.completed` for the ancestor.
   *
   * `retainBinding()` increments a separate retention counter that
   * is CHECKED in every teardown gate (see `dropNameReference`,
   * `releaseDispatchLease`, and `releaseBinding`). The counter is
   * orthogonal to `inFlight` so iter-39's eager
   * `releaseDispatchLease` path remains lossless: the dispatch can
   * release, the request can return control to the client, and
   * the binding is still pinned against teardown long enough for
   * the backgrounded `store.store(...)` to settle.
   *
   * Safe to call on a model whose binding has already been fully
   * torn down (no-op) — for example, a caller that retained inside
   * the lock and then lost the race with a forced teardown
   * elsewhere. The matching `releaseBinding(model)` MUST still
   * run in the persist's `.finally(...)` so the counter stays
   * balanced across future re-registrations of the same object.
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
