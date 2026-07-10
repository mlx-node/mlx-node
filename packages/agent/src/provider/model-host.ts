/**
 * `MlxModelHost` — single-resident, lazily-loaded model + `ChatSession`
 * owner for the provider bridge.
 *
 * Mirrors the CLI launch-claude swap semantics (drop-then-load, one
 * serialized operation chain) without the registry/alias machinery: the
 * agent process serves exactly one model at a time, and every operation
 * that touches the resident runs on one promise chain. Crucially the
 * resident check/load AND the caller's full inference callback execute
 * inside the SAME serialized closure ({@link MlxModelHost.runWithResident}),
 * so a queued swap to another model can never replace the resident while
 * an earlier caller is still mid-turn on it (stale session handle,
 * overlapping native activity on the compiled-path globals).
 */

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

import type { DiscoveredModelLike } from '../types.js';

export interface MlxModelHostOptions {
  /** Injectable model loader so tests can stub native loading. */
  loadModelFn?: typeof loadModel;
}

interface ResidentModel {
  id: string;
  session: ChatSession;
  /** Kept solely so a swap can explicitly drop the native ref before loading. */
  model: object;
}

export class MlxModelHost {
  private readonly byName = new Map<string, DiscoveredModelLike>();
  private readonly loadModelFn: typeof loadModel;
  private resident: ResidentModel | null = null;
  private chain: Promise<unknown> = Promise.resolve();

  constructor(models: DiscoveredModelLike[], opts: MlxModelHostOptions = {}) {
    for (const model of models) this.byName.set(model.name, model);
    this.loadModelFn = opts.loadModelFn ?? loadModel;
  }

  get residentId(): string | null {
    return this.resident?.id ?? null;
  }

  /**
   * Read-only lookup of the discovery record behind `modelId` (name, path,
   * `ModelType`). Pure map read — never touches the serialized chain or
   * the resident. The stream adapter uses it to pick the launch preset
   * for the model it is about to run.
   */
  modelInfo(modelId: string): DiscoveredModelLike | undefined {
    return this.byName.get(modelId);
  }

  /**
   * Make `modelId` resident (loading or swapping on demand) and run `fn`
   * against its `ChatSession` — both inside one serialized closure, so no
   * other queued operation (in particular a swap to a different model)
   * can touch the resident until `fn` settles. This is the ONLY way to
   * use the resident session; there is deliberately no method that
   * returns a session outside the serialized section.
   *
   * Swaps drop the old session + model refs BEFORE loading the new
   * checkpoint so GC + native destructors can reclaim the old weights
   * during the load. A load failure leaves no resident (next call
   * retries); a failure thrown by `fn` rejects only this call's promise
   * and keeps the resident loaded for later callers.
   */
  runWithResident<T>(modelId: string, fn: (session: ChatSession) => Promise<T>): Promise<T> {
    const entry = this.byName.get(modelId);
    if (!entry) {
      const known = [...this.byName.keys()].join(', ');
      return Promise.reject(new Error(`MlxModelHost: unknown model "${modelId}" (known models: ${known})`));
    }
    return this.runSerialized(async () => {
      let session: ChatSession;
      if (this.resident?.id === modelId) {
        session = this.resident.session;
      } else {
        this.resident = null;
        const model = await this.loadModelFn(entry.path);
        session = new ChatSession(model as unknown as SessionCapableModel);
        this.resident = { id: modelId, session, model };
      }
      return await fn(session);
    });
  }

  /**
   * Run `fn` after every previously queued operation completes. The
   * chain advances regardless of `fn`'s outcome — a rejection reaches
   * only this call's returned promise, never later queued operations.
   */
  private runSerialized<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.chain.then(fn);
    this.chain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }
}
