/**
 * `MlxModelHost` — single-resident, lazily-loaded model + `ChatSession`
 * owner for the provider bridge.
 *
 * Mirrors the CLI launch-claude swap semantics (drop-then-load, one
 * serialized operation chain) without the registry/alias machinery: the
 * agent process serves exactly one model at a time, and every operation
 * that touches the resident (load, swap, inference turn) runs on one
 * promise chain so concurrent pi calls can never race a swap against a
 * decode on the native compiled-path globals.
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
   * Run `fn` after every previously queued operation completes. The
   * chain advances regardless of `fn`'s outcome — a rejection reaches
   * only this call's returned promise, never later queued operations.
   */
  runSerialized<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.chain.then(fn);
    this.chain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  /**
   * Ensure `modelId` is the resident model, loading (or swapping to) it
   * on demand, and resolve with its `ChatSession`. Swaps drop the old
   * session + model refs BEFORE loading the new checkpoint so GC + native
   * destructors can reclaim the old weights during the load.
   */
  ensureResident(modelId: string): Promise<ChatSession> {
    const entry = this.byName.get(modelId);
    if (!entry) {
      const known = [...this.byName.keys()].join(', ');
      return Promise.reject(new Error(`MlxModelHost: unknown model "${modelId}" (known models: ${known})`));
    }
    return this.runSerialized(async () => {
      // Re-check under the serialized section — a queued predecessor may
      // have already made this model resident (or swapped it away).
      if (this.resident?.id === modelId) return this.resident.session;

      this.resident = null;
      const model = await this.loadModelFn(entry.path);
      const session = new ChatSession(model as unknown as SessionCapableModel);
      this.resident = { id: modelId, session, model };
      return session;
    });
  }
}
