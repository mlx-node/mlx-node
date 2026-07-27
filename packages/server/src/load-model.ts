/**
 * Safe out-of-band model load.
 *
 * A supervisor (or a `/model`-picker UI) needs to swap the resident model on a
 * server that is already serving. Doing that naively races the Metal
 * allocator — a corruption/abort bug, not a latency bug — so the bracketing
 * order below is load-bearing:
 *
 * ```text
 * idleSweeper.withSuspendedDrains(          ← OUTSIDE
 *   modelWorkCoordinator.withModelLoad(     ← INSIDE
 *     load() ; registry.register(...)
 *   , label))
 * ```
 *
 * Why the suspension must be OUTSIDE the writer lock: `endRequest()` arms a
 * drain timer at `t + idleClearCacheMs` whose callback runs
 * `__internal__.clearCache()`, walking the process-wide MLX free pool. A load
 * that arrives while another load holds the writer slot parks inside
 * `acquireWrite()` — and if the suspension were taken only after the lock was
 * won, that armed timer would still be live during the wait AND during the
 * hand-off, firing mid-materialization. Suspending first covers the wait, the
 * lock hand-off, `load()`, and `register()` as one interval.
 *
 * Why the writer lock must be INSIDE: `withModelLoad` takes the exclusive
 * writer slot, which excludes inference readers for exactly as long as
 * weights are being materialized — no longer. Hoisting it outside the
 * suspension would gain nothing and would hold readers off during the drain
 * bookkeeping too.
 *
 * See `ServerInstance.withSuspendedDrains` and the `__internal__.clearCache`
 * rustdoc in `packages/core/index.d.cts` for the underlying contracts.
 */

import type { ChatConfig } from '@mlx-node/core';

import type { IdleSweeper } from './idle-sweeper.js';
import type { ModelWorkCoordinator } from './model-work-coordinator.js';
import type { ModelRegistry, RegisterOptions, ServableModel } from './registry.js';

export interface GuardedLoadDeps {
  /** Only `withSuspendedDrains` is used; typed narrowly so tests can double it. */
  idleSweeper: Pick<IdleSweeper, 'withSuspendedDrains'>;
  modelWorkCoordinator: ModelWorkCoordinator;
  registry: ModelRegistry;
}

export interface LoadModelOptions {
  /** Primary registration name. Also used as the `/health` load label. */
  name: string;
  /** Materializes the model. Invoked exactly once, inside both brackets. */
  load: () => Promise<ServableModel>;
  /**
   * Extra names bound to the SAME instance. Aliases share one
   * `SessionRegistry`, preserving the single-warm invariant across names.
   * An alias equal to `name` is ignored rather than re-registered.
   */
  aliases?: string[];
  /** Per-model sampling defaults; forwarded to every alias too. */
  samplingDefaults?: ChatConfig;
  /** Per-model output-token clamp; forwarded to every alias too. */
  maxOutputTokens?: number;
}

/**
 * Load a model and register it under `name` (plus any aliases) with drains
 * suspended and inference excluded for the whole materialization window.
 *
 * Rejects with the underlying error if `load()` (or `register`) throws;
 * both brackets unwind via their own `finally`, so neither the suspend
 * counter nor the writer lock can leak.
 */
export async function runGuardedModelLoad(deps: GuardedLoadDeps, opts: LoadModelOptions): Promise<void> {
  // Build the register options once, omitting keys the caller never set.
  // `ModelRegistry.register` uses `'samplingDefaults' in opts` to decide
  // whether to OVERWRITE an existing binding's defaults, so passing an
  // explicit `undefined` would silently clear them on a re-register.
  const registerOpts: RegisterOptions = {};
  if (opts.samplingDefaults !== undefined) registerOpts.samplingDefaults = opts.samplingDefaults;
  if (opts.maxOutputTokens !== undefined) registerOpts.maxOutputTokens = opts.maxOutputTokens;

  await deps.idleSweeper.withSuspendedDrains(async () => {
    await deps.modelWorkCoordinator.withModelLoad(async () => {
      const instance = await opts.load();
      deps.registry.register(opts.name, instance, registerOpts);
      for (const alias of opts.aliases ?? []) {
        if (alias === opts.name) continue;
        deps.registry.register(alias, instance, registerOpts);
      }
    }, opts.name);
  });
}
