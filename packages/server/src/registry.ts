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
 */
interface SessionRegistryBinding {
  registry: SessionRegistry;
  refCount: number;
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
      this.releaseBinding(existing.model);
    }

    // Look up or allocate the shared binding for this model instance.
    let binding = this.sessionRegistriesByModel.get(model);
    if (!binding) {
      binding = { registry: new SessionRegistry({ model }), refCount: 0 };
      this.sessionRegistriesByModel.set(model, binding);
    }
    binding.refCount += 1;

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
    this.releaseBinding(entry.model);
    return true;
  }

  /** Decrement the refcount on a model binding; drop it at zero. */
  private releaseBinding(model: ServableModel): void {
    const binding = this.sessionRegistriesByModel.get(model);
    if (!binding) return;
    binding.refCount -= 1;
    if (binding.refCount <= 0) {
      this.sessionRegistriesByModel.delete(model);
    }
  }

  /**
   * Retrieve a model instance by name.
   */
  get(name: string): ServableModel | undefined {
    return this.models.get(name)?.model;
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
