/**
 * Single-resident lazy-load policy for `mlx launch claude`.
 *
 * The launch command discovers every local model up-front but loads at most
 * one into the `ModelRegistry` at a time. Switching models (e.g. via Claude
 * Code's `/model` picker) unregisters the previous instance, letting GC +
 * native destructors reclaim memory, before loading the new one.
 */

import type { LoadableModel, SessionCapableModel } from '@mlx-node/lm';
import type { ModelRegistry, PublicModelEntry } from '@mlx-node/server';

import type { DiscoveredModel } from './discover.js';

export interface SwapController {
  resolveModel: (name: string) => Promise<void>;
  listModels: () => PublicModelEntry[];
}

/**
 * Build the `resolveModel` + `listModels` callbacks for the handler.
 *
 * `loadModelFn` is injected so tests can stub it without touching native code.
 * The controller serializes every `resolveModel` invocation on a single
 * promise chain so two concurrent requests for different-but-currently-
 * unloaded models cannot race on the native compiled-path globals.
 */
export function makeSwapController(
  discovered: DiscoveredModel[],
  registry: ModelRegistry,
  loadModelFn: (path: string) => Promise<LoadableModel>,
  defaultName?: string,
): SwapController {
  const byName = new Map<string, DiscoveredModel>();
  for (const entry of discovered) byName.set(entry.name, entry);

  const ordered = [...discovered].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

  // Which entry unknown names (haiku subagent dispatches, etc.) fall back
  // to before anything is resident. Defaults to discovered[0], but the
  // caller can pin it to the user's `--model` pick so the first haiku
  // title-gen doesn't trigger a load of the alphabetically-first model
  // followed by an immediate swap to the user's real choice.
  const fallbackEntry = (defaultName != null ? byName.get(defaultName) : undefined) ?? discovered[0];

  let resident: { name: string } | null = null;
  // Names we registered as aliases to the current resident (e.g.
  // Claude Code's hardcoded `claude-haiku-*` for subagent dispatches /
  // title generation). Tracked so we can unregister them on `/model`
  // swap — otherwise an alias's refcount would keep the old binding
  // alive past the user's swap.
  const aliases = new Set<string>();
  let currentOp: Promise<unknown> = Promise.resolve();

  async function resolveModel(name: string): Promise<void> {
    // Fast path: already registered under this name (either as a real
    // resident or as an alias we previously installed). Avoid chaining.
    if (registry.get(name)) return;

    const next = currentOp.then(async () => {
      // Re-check under the serialized section — a prior waiter may have loaded it.
      if (registry.get(name)) return;

      // Pick the discovered entry to resolve against. If the requested
      // name matches a discovered model, use it. Otherwise (unknown
      // name — Claude Code's hardcoded small-fast-model, etc.) fall
      // through to the current resident so subagent dispatches don't
      // 404, loading discovered[0] on first boot if nothing is resident
      // yet.
      //
      // CRITICAL: this must read `resident` at RUN time, not QUEUE time.
      // If we capture it before chaining onto `currentOp`, a swap that
      // ran ahead of us will leave us with a stale target — e.g. a haiku
      // alias request that arrived during a `/model a → b` switch would
      // capture `targetEntry = a`, then re-bind itself to `a` and undo
      // the user's switch when its turn finally comes around.
      const knownEntry = byName.get(name);
      const targetEntry = knownEntry ?? (resident ? (byName.get(resident.name) ?? fallbackEntry) : fallbackEntry);
      const isAlias = targetEntry.name !== name;

      // Swap out any stale resident that isn't the target.
      //
      // We do NOT unregister the aliases here: in-flight messages.ts
      // requests may be microtask-racing between "resolveModel returned"
      // and "registry.get(body.model)" and dropping the alias in that
      // window yields a spurious 404. Instead we carry the alias set
      // across the swap and re-point them to the new resident below,
      // so the name always resolves to *some* live instance.
      const oldResident = resident;
      const carriedAliases = new Set(aliases);
      if (oldResident && oldResident.name !== targetEntry.name) {
        // Drop our local alias bookkeeping AND the old resident's primary
        // name binding, but leave the alias *names* in the registry pointed
        // at the old model — they hold the only refcount preventing GC,
        // and an in-flight `registry.get(alias)` must keep resolving to
        // *some* live instance until the new model is in hand.
        aliases.clear();
        registry.unregister(oldResident.name);
        resident = null;
      }

      // Ensure the target is resident. If the load throws, restore the
      // pre-swap controller state so future swap attempts know about the
      // aliases we just cleared — otherwise the alias *names* stay bound
      // in the registry to the old model object forever (alias bindings
      // hold their own refcount), but the controller forgets they exist
      // and never repoints them, leaving alias-routed traffic permanently
      // pinned to a stale model.
      let instance = registry.get(targetEntry.name);
      if (!instance) {
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
          // re-pick", not silently-wrong responses.
          for (const aliasName of carriedAliases) aliases.add(aliasName);
          if (oldResident) {
            let oldInstance: SessionCapableModel | undefined;
            for (const aliasName of carriedAliases) {
              const probe = registry.get(aliasName);
              if (probe) {
                oldInstance = probe;
                break;
              }
            }
            if (oldInstance) {
              const oldEntry = byName.get(oldResident.name);
              registry.register(oldResident.name, oldInstance, {
                samplingDefaults: oldEntry?.preset.sampling,
              });
              resident = oldResident;
            }
          }
          throw err;
        }
        instance = loaded as unknown as SessionCapableModel;
        registry.register(targetEntry.name, instance, {
          samplingDefaults: targetEntry.preset.sampling,
        });
        resident = { name: targetEntry.name };
      } else if (!resident) {
        resident = { name: targetEntry.name };
      }

      // Re-point any aliases carried across the swap onto the new
      // resident. `registry.register(sameName, differentModel)` drops
      // the old binding's refcount and installs the new one atomically,
      // so any concurrent `registry.get(alias)` either sees the old or
      // new instance — never null.
      for (const aliasName of carriedAliases) {
        if (aliasName === targetEntry.name) continue;
        registry.register(aliasName, instance, {
          samplingDefaults: targetEntry.preset.sampling,
        });
        aliases.add(aliasName);
      }

      // For unknown names, register an alias on the resident instance so
      // the endpoint's `registry.get(name)` lookup succeeds.
      if (isAlias) {
        registry.register(name, instance, {
          samplingDefaults: targetEntry.preset.sampling,
        });
        aliases.add(name);
      }
    });
    currentOp = next.catch(() => undefined);
    await next;
  }

  function listModels(): PublicModelEntry[] {
    const created = Math.floor(Date.now() / 1000);
    return ordered.map((entry) => ({
      id: entry.name,
      object: 'model',
      created,
      owned_by: 'mlx-node',
    }));
  }

  return { resolveModel, listModels };
}
