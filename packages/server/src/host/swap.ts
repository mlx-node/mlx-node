/**
 * Single-resident lazy-load policy for an inference host.
 *
 * The host discovers every local model up-front but loads at most one into
 * the `ModelRegistry` at a time. Switching models (e.g. via Claude Code's
 * `/model` picker, or the desktop app's model menu) unregisters the previous
 * instance, disposing it and waiting for native resource release before loading
 * the new one. A target that loads alone (`loadsAlone`) also drops the old
 * model's aliases first when the old model is idle, so its load reads live
 * memory with the old weights gone.
 */

import type { LoadableModel, SessionCapableModel, DecisionModel } from '@mlx-node/lm';

import type { PublicModelEntry } from '../handler.js';
import type { ModelRegistry } from '../registry.js';
import { FLUSH_MS } from '../supervision.js';
import type { ModelIdentity } from '../turn-trace.js';
import type { DiscoveredModel } from './discover.js';

export interface SwapController {
  resolveModel: (name: string) => Promise<void>;
  /**
   * Binds an unknown name to the loaded selected model when `resolveModel`
   * would only add that alias: nothing to load, swap or release, and no
   * resolve or unload in progress. Synchronous; returns whether it bound.
   */
  bindLoadedAlias?: (name: string) => boolean;
  listModels: () => PublicModelEntry[];
  disposeAll?: () => Promise<void>;
  /**
   * {@link disposeAll}, and each unloaded model's `dispose()` frees its
   * weights once its sessions flush; resolves when they are freed. The
   * selection stays, so the next request loads the same model again.
   */
  releaseAll?: () => Promise<void>;
}

function identityOf(entry: DiscoveredModel): ModelIdentity {
  return { modelId: entry.name, modelType: String(entry.modelType), contextWindow: entry.contextWindow };
}

/** A model whose SSD state cache holds its directory's lock until it closes. */
interface CacheClosingModel {
  closeCache(deadlineMs: number): Promise<boolean>;
}

function hasCacheToClose(model: unknown): model is CacheClosingModel {
  return typeof (model as Partial<CacheClosingModel> | undefined)?.closeCache === 'function';
}

/**
 * Build the `resolveModel` + `listModels` callbacks for the handler.
 *
 * `loadModelFn` is injected so tests can stub it without touching native code.
 * The controller serializes every `resolveModel` invocation on a single
 * promise chain so two concurrent requests for different-but-currently-
 * unloaded models cannot race on the native compiled-path globals.
 *
 * `estimatedWindow` is the window an entry's load would serve, priced before
 * it loads; `listModels` shows it between the trained window and the loaded
 * model's own.
 *
 * `loadsAlone` names the entries whose load plans from the memory live at
 * load time. A switch to one, with an old model that holds no dispatch lease,
 * no persist retention and no pre-dispatch permit, also unregisters every
 * alias of the old model, so the awaited disposal ends before the load. If
 * that load fails nothing stays resident and the selection stays the old
 * entry, so the next request loads the old model again.
 */
export function makeSwapController(
  discovered: DiscoveredModel[],
  registry: ModelRegistry,
  loadModelFn: (path: string) => Promise<LoadableModel>,
  defaultName?: string,
  estimatedWindow?: (name: string) => number | undefined,
  loadsAlone?: (entry: DiscoveredModel) => boolean,
): SwapController {
  const byName = new Map<string, DiscoveredModel>();
  for (const entry of discovered) byName.set(entry.name, entry);

  const ordered = [...discovered].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const created = Math.floor(Date.now() / 1000);

  // Which entry unknown names (haiku subagent dispatches, etc.) fall back
  // to before anything is resident. Defaults to discovered[0], but the
  // caller can pin it to the user's `--model` pick so the first haiku
  // title-gen doesn't trigger a load of the alphabetically-first model
  // followed by an immediate swap to the user's real choice.
  const fallbackEntry = (defaultName != null ? byName.get(defaultName) : undefined) ?? discovered[0];

  // Selection survives eviction by another capability; it holds no model resources.
  let selectedEntry = fallbackEntry;
  let resident: { name: string } | null = null;
  // Names we registered as aliases to the current resident (e.g.
  // Claude Code's hardcoded `claude-haiku-*` for subagent dispatches /
  // title generation). Tracked so we can unregister them on `/model`
  // swap — otherwise an alias's refcount would keep the old binding
  // alive past the user's swap.
  const aliases = new Set<string>();
  let currentOp: Promise<unknown> = Promise.resolve();
  // Resolves and unloads not yet finished; while any runs, the resident may be leaving.
  let pending = 0;
  // A dropped model frees its SSD cache only when V8 collects it, and a lease
  // or a persist can keep it for long. A new load of the same model opens
  // the same cache directory and would wait 20 s for its lock, then keep a
  // temporary cache for life: the old one's cache is closed first.
  const dropped = new Map<string, WeakRef<CacheClosingModel>>();
  const drop = (name: string, opts?: { dispose?: boolean }): void => {
    const model = registry.getAny(name);
    if (hasCacheToClose(model)) dropped.set(name, new WeakRef(model));
    if (opts) registry.unregister(name, opts);
    else registry.unregister(name);
  };
  const closeWithin = async (model: CacheClosingModel): Promise<void> => {
    const deadline = Date.now() + FLUSH_MS;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        model.closeCache(deadline),
        new Promise<void>((resolveTimer) => {
          timer = setTimeout(resolveTimer, FLUSH_MS);
          timer.unref?.();
        }),
      ]);
    } catch {
      /* the new load opens a temporary cache if the old one still holds the directory */
    } finally {
      clearTimeout(timer);
    }
  };

  const closeDropped = (name: string): Promise<void> | undefined => {
    const model = dropped.get(name)?.deref();
    dropped.delete(name);
    return model ? closeWithin(model) : undefined;
  };
  async function resolveModel(name: string): Promise<void> {
    // Fast path: already registered under this name (either as a real
    // resident or as an alias we previously installed). Avoid chaining.
    if (registry.getAny(name)) return;

    pending += 1;
    const next = currentOp.then(async () => {
      // Re-check under the serialized section — a prior waiter may have loaded it.
      if (registry.getAny(name)) return;

      // Pick the discovered entry to resolve against. If the requested
      // name matches a discovered model, use it. Otherwise (unknown
      // name — Claude Code's hardcoded small-fast-model, etc.) fall
      // through to the last successfully selected chat model, even if
      // another capability has evicted it in single-resident mode.
      //
      // CRITICAL: this must read `selectedEntry` at RUN time, not QUEUE time.
      // If we capture it before chaining onto `currentOp`, a swap that
      // ran ahead of us will leave us with a stale target — e.g. a haiku
      // alias request that arrived during a `/model a → b` switch would
      // capture `targetEntry = a`, then re-bind itself to `a` and undo
      // the user's switch when its turn finally comes around.
      const knownEntry = byName.get(name);
      const targetEntry = knownEntry ?? selectedEntry;
      const isAlias = targetEntry.name !== name;

      // Swap out any stale resident that isn't the target.
      //
      // We do NOT unregister the aliases here: in-flight messages.ts
      // requests may be microtask-racing between "resolveModel returned"
      // and "registry.getAny(body.model)" and dropping the alias in that
      // window yields a spurious 404. Instead we carry the alias set
      // across the swap and re-point them to the new resident below,
      // so the name always resolves to *some* live instance.
      const oldResident = resident;
      const carriedAliases = new Set(aliases);
      // A busy old model would defer its teardown behind a request that
      // waits for this load, so it keeps its aliases until the new model is
      // in hand. A request between its resolve and its dispatch lease holds
      // a pre-dispatch permit, or (no permit) takes the lease in the
      // microtasks that resume it, which the yield lets run: either one
      // makes the old model busy.
      let freeFirst =
        oldResident !== null &&
        oldResident.name !== targetEntry.name &&
        !registry.getAny(targetEntry.name) &&
        loadsAlone?.(targetEntry) === true;
      if (freeFirst) {
        await new Promise<void>((resume) => setImmediate(resume));
        freeFirst =
          registry.dispatchLeaseCount === 0 &&
          registry.persistRetainCount === 0 &&
          registry.preDispatchAdmitCount === 0;
      }
      if (oldResident && oldResident.name !== targetEntry.name) {
        // Drop our local alias bookkeeping AND the old resident's primary
        // name binding. Unless the target loads alone, leave the alias
        // *names* in the registry pointed at the old model: they hold the
        // only refcount preventing disposal, and an in-flight
        // `registry.getAny(alias)` must keep resolving to *some* live
        // instance until the new model is in hand.
        aliases.clear();
        drop(oldResident.name, { dispose: true });
        if (freeFirst) {
          for (const aliasName of carriedAliases) if (registry.getAny(aliasName)) drop(aliasName, { dispose: true });
        }
        resident = null;
        // With no aliases this joins the old model thread before the new load.
        // Aliased or leased bindings finish through the same tracked disposal path.
        await registry.settleModelDisposals();
      }

      // Ensure the target is resident. If the load throws, restore the
      // pre-swap controller state so future swap attempts know about the
      // aliases we just cleared — otherwise the alias *names* stay bound
      // in the registry to the old model object forever (alias bindings
      // hold their own refcount), but the controller forgets they exist
      // and never repoints them, leaving alias-routed traffic permanently
      // pinned to a stale model.
      let instance = registry.getAny(targetEntry.name);
      if (!instance) {
        const closing = closeDropped(targetEntry.name);
        if (closing) await closing;
        let loaded: LoadableModel;
        try {
          loaded = await loadModelFn(targetEntry.path);
        } catch (err) {
          // Recovery: re-populate the controller's alias set so the next
          // resolveModel call still owns them. The alias→old-model bindings
          // are still live in the registry (we never unregistered them), so
          // we can recover the old model object via any surviving alias and
          // re-bind the old resident's primary name.
          //
          // If `carriedAliases` is empty (no aliases ever existed) AND we
          // unregistered `oldResident.name`, the binding's refcount may have
          // hit zero and the model is gone. There's nothing the controller
          // can do to recover in that case — the user will need to /model-
          // pick again. This is acceptable: alias-less load failures are
          // rare, and the user-facing symptom is "prior model gone, please
          // re-pick", not silently-wrong responses. A target that loaded
          // alone disposed the old model first: nothing is left to restore.
          if (!freeFirst) for (const aliasName of carriedAliases) aliases.add(aliasName);
          if (oldResident && !freeFirst) {
            let oldInstance: SessionCapableModel | DecisionModel | undefined;
            for (const aliasName of carriedAliases) {
              const probe = registry.getAny(aliasName);
              if (probe) {
                oldInstance = probe;
                break;
              }
            }
            if (oldInstance) {
              const oldEntry = byName.get(oldResident.name);
              registry.registerAny(oldResident.name, oldInstance, {
                samplingDefaults: oldEntry?.preset.sampling,
                maxOutputTokens: oldEntry?.preset.maxOutputTokens,
                identity: oldEntry ? identityOf(oldEntry) : undefined,
              });
              resident = oldResident;
            }
          }
          throw err;
        }
        instance = loaded as unknown as SessionCapableModel | DecisionModel;
        registry.registerAny(targetEntry.name, instance, {
          samplingDefaults: targetEntry.preset.sampling,
          maxOutputTokens: targetEntry.preset.maxOutputTokens,
          identity: identityOf(targetEntry),
        });
        resident = { name: targetEntry.name };
      } else if (!resident) {
        resident = { name: targetEntry.name };
      }
      selectedEntry = targetEntry;

      // Re-point any aliases carried across the swap onto the new
      // resident. `registry.registerAny(sameName, differentModel)` drops
      // the old binding's refcount and installs the new one atomically,
      // so any concurrent `registry.getAny(alias)` either sees the old or
      // new instance — never null.
      for (const aliasName of carriedAliases) {
        if (aliasName === targetEntry.name) continue;
        registry.registerAny(aliasName, instance, {
          samplingDefaults: targetEntry.preset.sampling,
          maxOutputTokens: targetEntry.preset.maxOutputTokens,
        });
        aliases.add(aliasName);
      }

      // For unknown names, register an alias on the resident instance so
      // the endpoint's `registry.getAny(name)` lookup succeeds.
      if (isAlias) {
        registry.registerAny(name, instance, {
          samplingDefaults: targetEntry.preset.sampling,
          maxOutputTokens: targetEntry.preset.maxOutputTokens,
        });
        aliases.add(name);
      }

      // Re-pointing the last old alias can start its disposal only now.
      await registry.settleModelDisposals();
    });
    currentOp = next.catch(() => undefined);
    try {
      await next;
    } finally {
      pending -= 1;
    }
  }

  function bindLoadedAlias(name: string): boolean {
    if (pending > 0 || byName.has(name) || registry.getAny(name)) return false;
    if (resident === null || resident.name !== selectedEntry.name) return false;
    const instance = registry.get(resident.name);
    if (instance === undefined) return false;
    registry.registerAny(name, instance, {
      samplingDefaults: selectedEntry.preset.sampling,
      maxOutputTokens: selectedEntry.preset.maxOutputTokens,
    });
    aliases.add(name);
    return true;
  }

  function listModels(): PublicModelEntry[] {
    return ordered.map((entry) => {
      const window =
        registry.servedContext(entry.name)?.windowTokens ?? estimatedWindow?.(entry.name) ?? entry.contextWindow;
      return {
        id: entry.name,
        object: 'model',
        created,
        owned_by: 'mlx-node',
        // Grok Build prefetch contract: it defaults `api_backend` to
        // chat_completions, a route this host does not serve. "messages"
        // (Anthropic wire) is the backend this router fully supports —
        // the same path `mlx launch claude` drives — so always pin it.
        context_window: window,
        max_model_len: window,
        context_length: window,
        api_backend: 'messages',
        model_family: String(entry.modelType),
        supports_images: entry.supportsImages,
      };
    });
  }

  async function unloadAll(dispose: boolean): Promise<void> {
    pending += 1;
    try {
      await currentOp;
      const sessions = registry.listSessionRegistries();
      for (const entry of registry.list()) if (registry.getAny(entry.id)) drop(entry.id, { dispose });
      resident = null;
      aliases.clear();
      await Promise.all(sessions.map((session) => session.flushPendingDisposals()));
      if (dispose) await registry.settleModelDisposals();
    } finally {
      pending -= 1;
    }
  }
  return {
    resolveModel,
    bindLoadedAlias,
    listModels,
    disposeAll: () => unloadAll(false),
    releaseAll: () => unloadAll(true),
  };
}
