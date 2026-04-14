/**
 * ModelRegistry -- maps friendly model names to loaded model instances.
 *
 * All models that expose the chat-session surface (see `SessionCapableModel`
 * from `@mlx-node/lm`) are eligible for serving. Streaming support is
 * detected by the presence of `chatStreamSessionStart` on the model.
 *
 * This interface intentionally mirrors `SessionCapableModel` one-to-one —
 * the server always drives models through `ChatSession<M>` wrappers, never
 * the low-level NAPI methods directly. Step S2 of the chat-session refactor
 * migrates the endpoint layer to use a per-model `SessionRegistry` cache.
 */

import type { SessionCapableModel } from '@mlx-node/lm';

/** Minimal contract for a model that can be served via chat sessions. */
export type ServableModel = SessionCapableModel;

/** Model entry stored in the registry. */
export interface ModelEntry {
  id: string;
  model: ServableModel;
  createdAt: number;
}

export class ModelRegistry {
  private readonly models = new Map<string, ModelEntry>();

  /**
   * Register a model under a given name.
   * If a model with the same name already exists, it is replaced.
   */
  register(name: string, model: ServableModel): void {
    this.models.set(name, {
      id: name,
      model,
      createdAt: Math.floor(Date.now() / 1000),
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
   * Every `SessionCapableModel` is expected to expose
   * `chatStreamSessionStart`, so in practice this is universally true for
   * any properly-typed model. We still duck-type the method so a partially
   * stubbed test double can opt out of streaming by omitting it.
   */
  hasStreamSupport(model: ServableModel): boolean {
    const fn = (model as unknown as Record<string, unknown>)['chatStreamSessionStart'];
    return typeof fn === 'function';
  }
}
