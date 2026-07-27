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

import { COLD_TIER_RESTORE_FAMILIES } from '../cold-tier.js';
import type { DiscoveredModelLike } from '../types.js';

/**
 * Re-exported so the HOST consults exactly the symbol the drift guard and the
 * `--no-persist-cache` help text consult. The definition lives in the
 * native-free `../cold-tier.js` leaf (reachable off-package through the
 * `@mlx-node/agent/catalog` subpath, which re-exports it)
 * because this module value-imports `@mlx-node/lm`, which loads the native
 * addon — the dashboard and the CLI help path must be able to read the list
 * without that.
 */
export { COLD_TIER_RESTORE_FAMILIES };

/** Per-load policy handed to {@link MlxModelHostOptions.resolveModelPathFn}. */
export interface ModelLoadPolicy {
  /**
   * Authoritative cold-tier directive for the config overlay
   * (`persist_paged_cache` in the cloned config.json). Present ONLY for loads
   * of a {@link COLD_TIER_RESTORE_FAMILIES} family, carrying the resolved
   * {@link MlxModelHostOptions.persistPagedCache} value as an EXPLICIT
   * boolean: `true` enables the SSD cold tier, `false` authoritatively
   * disables it — overriding any `persist_paged_cache` the checkpoint's own
   * config.json hard-codes, so `mlx agent --no-persist-cache` truly wins.
   * Every other family receives no policy at all, so the overlay never touches
   * the field for them.
   */
  persistPagedCache: boolean;
}

export interface MlxModelHostOptions {
  /** Injectable model loader so tests can stub native loading. */
  loadModelFn?: typeof loadModel;
  /**
   * Optional load-path policy. `mlx agent` uses this to point the loader at
   * an ephemeral config overlay with block-paged attention enabled while
   * leaving the checkpoint directory untouched. The optional per-load policy
   * carries the qwen3 cold-tier opt-in (see {@link ModelLoadPolicy}).
   */
  resolveModelPathFn?: (model: DiscoveredModelLike, policy?: ModelLoadPolicy) => Promise<string>;
  /**
   * Reject a loaded model unless its native paged-cache adapter is active.
   * The agent entrypoint enables this so a model/platform incompatibility
   * fails clearly instead of silently falling back to flat KV cache. Gemma4
   * with an attached external draft is the deliberate exception used only by
   * the agent's explicit draft opt-in: its DSpark / assistant speculative
   * executor is currently flat-cache-only.
   */
  requirePagedCache?: boolean;
  /**
   * Enable the cold tier (persisted paged prefix blocks) by default. `mlx
   * agent` turns this on; `mlx agent --no-persist-cache` sets it false.
   * Applied only to loads of a {@link COLD_TIER_RESTORE_FAMILIES} family —
   * every other family keeps per-layer state outside the paged pool, so its
   * prefix cannot be restored soundly. Defaults true.
   */
  persistPagedCache?: boolean;
}

interface ResidentModel {
  id: string;
  session: ChatSession;
  /** Kept solely so a swap can explicitly drop the native ref before loading. */
  model: object;
  /**
   * Set when the previous turn on this resident ended in a native ERROR
   * terminal. A native error mid-decode can leave the physical KV cache
   * advanced past the committed `cached_token_history`, so the next warm
   * reuse would replay pi's history onto a misaligned prefix (garbled
   * continuation). While dirty, the next turn does a FULL `session.reset()`
   * (cold prefill) instead of the warm-reuse wipe. Cleared on consume; NOT
   * set on abort / stop / length (those leave the cache consistent).
   */
  dirty: boolean;
}

export class MlxModelHost {
  private readonly byName = new Map<string, DiscoveredModelLike>();
  private readonly loadModelFn: typeof loadModel;
  private readonly resolveModelPathFn: (model: DiscoveredModelLike, policy?: ModelLoadPolicy) => Promise<string>;
  private readonly requirePagedCache: boolean;
  private readonly persistPagedCache: boolean;
  private resident: ResidentModel | null = null;
  private chain: Promise<unknown> = Promise.resolve();

  constructor(models: DiscoveredModelLike[], opts: MlxModelHostOptions = {}) {
    for (const model of models) this.byName.set(model.name, model);
    this.loadModelFn = opts.loadModelFn ?? loadModel;
    this.resolveModelPathFn = opts.resolveModelPathFn ?? (async (model) => model.path);
    this.requirePagedCache = opts.requirePagedCache ?? false;
    this.persistPagedCache = opts.persistPagedCache ?? true;
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
   * `fn` also receives a `resident` boolean: `true` when the turn reused
   * the already-loaded model (warm — no load happened), `false` when it had
   * to load or swap the checkpoint first. This is the same warm/cold
   * distinction the branch below already makes; surfacing it lets a metrics
   * consumer separate queue wait from cold-load time without another channel.
   *
   * Swaps drop the old session + model refs BEFORE loading the new
   * checkpoint so GC + native destructors can reclaim the old weights
   * during the load. A load failure leaves no resident (next call
   * retries); a failure thrown by `fn` rejects only this call's promise
   * and keeps the resident loaded for later callers.
   */
  runWithResident<T>(modelId: string, fn: (session: ChatSession, resident: boolean) => Promise<T>): Promise<T> {
    const entry = this.byName.get(modelId);
    if (!entry) {
      const known = [...this.byName.keys()].join(', ');
      return Promise.reject(new Error(`MlxModelHost: unknown model "${modelId}" (known models: ${known})`));
    }
    return this.runSerialized(async () => {
      let session: ChatSession;
      // `true` when this turn reuses the loaded resident (warm), `false` when
      // it loaded/swapped the checkpoint first (cold).
      let resident: boolean;
      if (this.resident?.id === modelId) {
        session = this.resident.session;
        resident = true;
      } else {
        resident = false;
        this.resident = null;
        // Only a COLD_TIER_RESTORE_FAMILIES family has a sound paged cold
        // restore. Hand it an EXPLICIT tri-state directive so the overlay can
        // authoritatively set the flag either way (default-on, or
        // `--no-persist-cache` off — overriding any value in the checkpoint's
        // config.json). Every other family gets no policy at all, so the
        // overlay never touches the field for them.
        const resolvedPath = COLD_TIER_RESTORE_FAMILIES.has(entry.modelType)
          ? await this.resolveModelPathFn(entry, { persistPagedCache: this.persistPagedCache })
          : await this.resolveModelPathFn(entry);
        const model = await this.loadModelFn(resolvedPath);
        const sessionModel = model as unknown as SessionCapableModel;
        const gemmaDraftActive = entry.modelType === 'gemma4' && sessionModel.hasMtpWeights?.() === true;
        if (this.requirePagedCache && sessionModel.hasBlockPagedCache?.() !== true && !gemmaDraftActive) {
          throw new Error(
            `MlxModelHost: model "${modelId}" (${entry.modelType}) loaded without an active ` +
              `PagedAttention cache; this checkpoint, quantization, or platform is not compatible ` +
              `with the mlx agent paged-cache requirement`,
          );
        }
        if (this.requirePagedCache && entry.modelType === 'qwen3_5_moe' && sessionModel.hasMtpWeights?.() === true) {
          throw new Error(
            `MlxModelHost: model "${modelId}" has Qwen3.5 MoE MTP weights, but the native MoE ` +
              `backend cannot combine MTP with PagedAttention yet; refusing to silently downgrade ` +
              `this agent session to paged autoregressive decoding`,
          );
        }
        session = new ChatSession(sessionModel);
        this.resident = { id: modelId, session, model, dirty: false };
      }
      return await fn(session, resident);
    });
  }

  /**
   * Flag the current resident as post-error so the next turn does a full
   * reset instead of a warm reuse. No-op unless `modelId` is the live
   * resident (a load failure or a swap already dropped/replaced it, and a
   * reloaded model starts with a clean cache).
   */
  markResidentDirty(modelId: string): void {
    if (this.resident?.id === modelId) {
      this.resident.dirty = true;
    }
  }

  /**
   * Read-and-clear the resident's post-error `dirty` flag. Returns `true`
   * only when `modelId` is the live resident AND it was dirty — the signal
   * for the caller to run a full `session.reset()` this turn instead of the
   * warm-reuse wipe.
   */
  consumeResidentDirty(modelId: string): boolean {
    if (this.resident?.id !== modelId) {
      return false;
    }
    const wasDirty = this.resident.dirty;
    this.resident.dirty = false;
    return wasDirty;
  }

  /**
   * Drop the current resident so the next `runWithResident` reloads it from
   * scratch. Used when a post-error full reset itself fails and the session
   * can no longer be trusted. No-op unless `modelId` is the live resident.
   */
  invalidateResident(modelId: string): void {
    if (this.resident?.id === modelId) {
      this.resident = null;
    }
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
