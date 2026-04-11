/**
 * ModelRegistry -- maps friendly model names to loaded model instances.
 *
 * All models that expose a `chat()` method are eligible for serving.
 * Streaming support is detected by duck-typing the `chatStream` method.
 */

/** Minimal contract for a model that can be served. */
export interface ServableModel {
  chat(messages: unknown[], config?: unknown): Promise<unknown>;
}

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
   * Check whether a model supports streaming via `chatStream()`.
   * Uses duck-typing: the model must have a `chatStream` method that returns an
   * async iterable (2 params), not a callback-based stream (3 params).
   */
  hasStreamSupport(model: ServableModel): boolean {
    const fn = (model as unknown as Record<string, unknown>)['chatStream'];
    return typeof fn === 'function' && (fn as Function).length <= 2;
  }
}
