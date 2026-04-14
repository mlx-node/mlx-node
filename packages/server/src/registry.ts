/**
 * ModelRegistry -- maps friendly model names to loaded model instances.
 *
 * All models that expose the chat-session surface (see `SessionCapableModel`
 * from `@mlx-node/lm`) are eligible for serving. Every registered model is
 * paired with its own `SessionRegistry` — an LRU+TTL cache of live
 * `ChatSession` instances keyed by the server-allocated response id. The
 * endpoint layer fetches the per-model session registry via
 * {@link ModelRegistry.getSessionRegistry} and routes each request through
 * a session looked up or allocated there.
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
  /** Per-model LRU+TTL session cache. See `SessionRegistry`. */
  sessionRegistry: SessionRegistry;
}

export class ModelRegistry {
  private readonly models = new Map<string, ModelEntry>();

  /**
   * Register a model under a given name.
   * If a model with the same name already exists, it is replaced —
   * including its SessionRegistry, which means any cached sessions
   * under the old model are dropped.
   */
  register(name: string, model: ServableModel): void {
    this.models.set(name, {
      id: name,
      model,
      createdAt: Math.floor(Date.now() / 1000),
      sessionRegistry: new SessionRegistry({ model }),
    });
  }

  /**
   * Unregister a model by name.
   * @returns true if the model was removed.
   */
  unregister(name: string): boolean {
    return this.models.delete(name);
  }

  /**
   * Retrieve a model instance by name.
   */
  get(name: string): ServableModel | undefined {
    return this.models.get(name)?.model;
  }

  /**
   * Retrieve the session registry for a given model, or `undefined`
   * if the model is not registered.
   */
  getSessionRegistry(name: string): SessionRegistry | undefined {
    return this.models.get(name)?.sessionRegistry;
  }

  /** Iterate every registered session registry. */
  listSessionRegistries(): SessionRegistry[] {
    const out: SessionRegistry[] = [];
    for (const entry of this.models.values()) {
      out.push(entry.sessionRegistry);
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
