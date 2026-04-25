import type { LoadableModel } from '@mlx-node/lm';
import type { ModelRegistry } from '@mlx-node/server';
import { describe, expect, it, vi } from 'vite-plus/test';

import type { DiscoveredModel } from '../../../packages/cli/src/commands/launch-claude/discover.js';
import { makeSwapController } from '../../../packages/cli/src/commands/launch-claude/swap.js';

interface FakeRegistry {
  get: ReturnType<typeof vi.fn>;
  register: ReturnType<typeof vi.fn>;
  unregister: ReturnType<typeof vi.fn>;
}

function fakeRegistry(): FakeRegistry {
  const resident = new Map<string, unknown>();
  const get = vi.fn((name: string) => resident.get(name));
  const register = vi.fn((name: string, model: unknown) => {
    resident.set(name, model);
  });
  const unregister = vi.fn((name: string) => {
    resident.delete(name);
    return true;
  });
  return { get, register, unregister };
}

function discovered(names: string[]): DiscoveredModel[] {
  return names.map((n) => ({
    name: n,
    path: `/fake/${n}`,
    modelType: 'qwen3_5' as const,
    preset: {
      sampling: { temperature: 0.6, topP: 0.95, topK: 20, minP: 0, presencePenalty: 0, repetitionPenalty: 1 },
      maxOutputTokens: 1024,
    },
  }));
}

describe('makeSwapController', () => {
  it('loads a model on first resolve and registers it under the requested name', async () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (p: string) => ({ marker: p }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a']), reg as unknown as ModelRegistry, loader);

    await ctrl.resolveModel('a');
    expect(loader).toHaveBeenCalledTimes(1);
    expect(loader).toHaveBeenCalledWith('/fake/a');
    expect(reg.register).toHaveBeenCalledTimes(1);
    const [name, model, opts] = reg.register.mock.calls[0];
    expect(name).toBe('a');
    expect(model).toEqual({ marker: '/fake/a' });
    expect(opts?.samplingDefaults).toBeDefined();
    expect(opts?.samplingDefaults?.temperature).toBe(0.6);
  });

  it('unregisters the previous resident when switching models', async () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (p: string) => ({ marker: p }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader);

    await ctrl.resolveModel('a');
    await ctrl.resolveModel('b');

    expect(loader.mock.calls.map((c) => c[0])).toEqual(['/fake/a', '/fake/b']);
    expect(reg.unregister).toHaveBeenCalledWith('a');
    expect(reg.register.mock.calls.map((c) => c[0])).toEqual(['a', 'b']);
  });

  it('deduplicates concurrent calls targeting the same name', async () => {
    const reg = fakeRegistry();
    let resolveLoad: ((v: LoadableModel) => void) | null = null;
    const loader = vi.fn(
      () =>
        new Promise<LoadableModel>((resolve) => {
          resolveLoad = resolve;
        }),
    );
    const ctrl = makeSwapController(discovered(['a']), reg as unknown as ModelRegistry, loader);

    const p1 = ctrl.resolveModel('a');
    const p2 = ctrl.resolveModel('a');
    // Give both calls a chance to chain onto currentOp.
    await Promise.resolve();
    resolveLoad!({ marker: '/fake/a' } as unknown as LoadableModel);
    await Promise.all([p1, p2]);

    expect(loader).toHaveBeenCalledTimes(1);
    expect(reg.register).toHaveBeenCalledTimes(1);
  });

  it('serializes concurrent calls for different names in arrival order', async () => {
    const reg = fakeRegistry();
    const order: string[] = [];
    const loader = vi.fn(async (p: string) => {
      order.push(`start:${p}`);
      await new Promise((r) => setTimeout(r, 10));
      order.push(`end:${p}`);
      return { marker: p } as unknown as LoadableModel;
    });
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader);

    const pA = ctrl.resolveModel('a');
    const pB = ctrl.resolveModel('b');
    await Promise.all([pA, pB]);

    expect(loader).toHaveBeenCalledTimes(2);
    expect(order).toEqual(['start:/fake/a', 'end:/fake/a', 'start:/fake/b', 'end:/fake/b']);
  });

  it('aliases an unknown name to discovered[0] on first call', async () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (p: string) => ({ marker: p }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader);

    // Claude Code hardcodes `claude-haiku-*` for subagent dispatches;
    // those should not 404 — we alias them to the resident model.
    await ctrl.resolveModel('claude-haiku-4-5-20251001');

    expect(loader).toHaveBeenCalledTimes(1);
    expect(loader).toHaveBeenCalledWith('/fake/a');
    // Two register calls: one for the real resident, one for the alias.
    expect(reg.register).toHaveBeenCalledTimes(2);
    expect(reg.register.mock.calls.map((c) => c[0])).toEqual(['a', 'claude-haiku-4-5-20251001']);
    // Both point at the same model instance.
    expect(reg.register.mock.calls[0][1]).toBe(reg.register.mock.calls[1][1]);
  });

  it('aliases an unknown name to the current resident, not discovered[0]', async () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (p: string) => ({ marker: p }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader);

    // User /model-picks 'b' first, THEN a subagent dispatch fires.
    await ctrl.resolveModel('b');
    await ctrl.resolveModel('claude-haiku-4-5-20251001');

    // Alias should point at 'b' (resident), not 'a' (discovered[0]).
    const aliasRegister = reg.register.mock.calls.find((c) => c[0] === 'claude-haiku-4-5-20251001');
    expect(aliasRegister).toBeDefined();
    expect(aliasRegister?.[1]).toEqual({ marker: '/fake/b' });
  });

  it('carries aliases across a resident swap instead of unregistering them', async () => {
    // Regression: previously we unregistered all aliases before loading
    // the new resident, which raced with any in-flight messages.ts
    // request whose `resolveModel` had just returned but hadn't yet
    // called `registry.get(alias)`. That request saw a null lookup and
    // returned 404. The fix is to re-point the alias onto the new
    // resident (same-name register with a different model object drops
    // the old binding's refcount atomically) so `registry.get(alias)`
    // always resolves to *some* live instance.
    const reg = fakeRegistry();
    const loader = vi.fn(async (p: string) => ({ marker: p }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader);

    await ctrl.resolveModel('a');
    await ctrl.resolveModel('claude-haiku-4-5-20251001');
    await ctrl.resolveModel('b');

    // Only the old resident's NAME is unregistered; the alias is re-pointed.
    const unregOrder = reg.unregister.mock.calls.map((c) => c[0]);
    expect(unregOrder).toContain('a');
    expect(unregOrder).not.toContain('claude-haiku-4-5-20251001');

    // The alias got a second `register` call, now bound to b's instance.
    const aliasRegs = reg.register.mock.calls.filter((c) => c[0] === 'claude-haiku-4-5-20251001');
    expect(aliasRegs.length).toBe(2);
    expect(aliasRegs[0]?.[1]).toEqual({ marker: '/fake/a' });
    expect(aliasRegs[1]?.[1]).toEqual({ marker: '/fake/b' });
  });

  it('respects defaultName fallback for unknown aliases on the very first call', async () => {
    // With --model flag the user picks a specific discovered entry; we
    // shouldn't load alphabetically-first discovered[0] just because a
    // haiku title-gen arrived before the main chat turn.
    const reg = fakeRegistry();
    const loader = vi.fn(async (p: string) => ({ marker: p }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'b');

    await ctrl.resolveModel('claude-haiku-4-5-20251001');

    // Loader called with b's path (the chosen default), not a's.
    expect(loader).toHaveBeenCalledWith('/fake/b');
    expect(loader).not.toHaveBeenCalledWith('/fake/a');
  });

  it('does not reload the previous resident when an alias request races a /model swap', { timeout: 5000 }, async () => {
    // Regression: previously `targetEntry` and `isAlias` were captured at
    // QUEUE time (i.e. the moment `resolveModel` was called), reading
    // `resident` before chaining onto `currentOp`. A concurrent unknown-name
    // (haiku alias) request that arrived DURING a `/model a → b` switch
    // would capture `targetEntry = a` (the at-queue-time resident), then,
    // when its turn ran AFTER b had become resident, would swap back to a
    // and undo the user's switch.
    //
    // Fix: read `resident` inside the serialized .then() body so the
    // target is computed against the live state at run time.
    const reg = fakeRegistry();

    // Controllable loader: each call returns a promise we can resolve on demand.
    const pending: { path: string; resolve: (v: LoadableModel) => void }[] = [];
    const loader = vi.fn(
      (path: string) =>
        new Promise<LoadableModel>((resolve) => {
          pending.push({ path, resolve });
        }),
    );
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader);

    // 1) Establish resident=a.
    const pA = ctrl.resolveModel('a');
    // Drain microtasks so the loader call lands.
    await Promise.resolve();
    expect(pending).toHaveLength(1);
    expect(pending[0].path).toBe('/fake/a');
    pending[0].resolve({ marker: '/fake/a' } as unknown as LoadableModel);
    await pA;

    // 2) Start a swap to b — DO NOT await; b is mid-load.
    const pB = ctrl.resolveModel('b');
    // Let the .then() body run up to the loader await.
    await Promise.resolve();
    await Promise.resolve();
    expect(pending).toHaveLength(2);
    expect(pending[1].path).toBe('/fake/b');

    // 3) Concurrently fire an alias request while b is still loading.
    // At THIS moment the live `resident` is still a, but by the time the
    // alias's queued body runs, b will be resident. Old code captured
    // targetEntry = a here and would have re-loaded a after b finished.
    const pAlias = ctrl.resolveModel('claude-haiku-4-5-20251001');
    await Promise.resolve();

    // 4) Unblock the b load. The alias body runs next.
    pending[1].resolve({ marker: '/fake/b' } as unknown as LoadableModel);
    await pB;
    await pAlias;

    // 5) Final state: only two real loads (a then b). No third load of a.
    const loadedPaths = loader.mock.calls.map((c) => c[0]);
    expect(loadedPaths).toEqual(['/fake/a', '/fake/b']);
    expect(loadedPaths.filter((p) => p === '/fake/a')).toHaveLength(1);

    // The alias must be registered onto b's instance — not re-registered onto a.
    const aliasRegister = reg.register.mock.calls.find((c) => c[0] === 'claude-haiku-4-5-20251001');
    expect(aliasRegister).toBeDefined();
    expect(aliasRegister?.[1]).toEqual({ marker: '/fake/b' });

    // And `a` must have been unregistered exactly once (during the a → b swap),
    // never re-registered as a fresh resident.
    expect(reg.unregister.mock.calls.map((c) => c[0])).toContain('a');
    expect(reg.register.mock.calls.filter((c) => c[0] === 'a')).toHaveLength(1);
  });

  it('restores alias state when a swap load fails so a later swap can re-point them', async () => {
    // Regression: previously, when `loadModelFn` threw mid-swap, we had
    // already done `aliases.clear()` + `registry.unregister(oldResident)`.
    // The controller forgot the alias names while the registry's alias
    // bindings still held the old model object (alias bindings have their
    // own refcount on the binding). A subsequent successful swap would NOT
    // re-point those aliases (the controller no longer remembered them),
    // leaving alias-routed traffic permanently pinned to the stale model
    // and the old model's memory permanently leaked.
    //
    // Black-box assertion: do the failed swap, then do a successful swap
    // to a *new* model, and verify the haiku alias is repointed onto the
    // new model. If the controller's alias set hadn't been restored, the
    // haiku alias would still point at the original model after the
    // successful swap.
    const reg = fakeRegistry();
    let shouldFail = false;
    const loader = vi.fn(async (p: string) => {
      if (shouldFail) throw new Error(`refusing to load ${p}`);
      return { marker: p } as unknown as LoadableModel;
    });
    const ctrl = makeSwapController(discovered(['a', 'b', 'c']), reg as unknown as ModelRegistry, loader);

    // 1) Load `a` and install a haiku alias on it.
    await ctrl.resolveModel('a');
    await ctrl.resolveModel('claude-haiku-x');
    // Sanity: alias resolves to a's instance.
    const aInstance = reg.register.mock.calls.find((c) => c[0] === 'a')?.[1];
    const probe = (name: string): unknown => (reg.get as unknown as (n: string) => unknown)(name);
    expect(probe('claude-haiku-x')).toBe(aInstance);

    // 2) Attempt a swap to `b` and have it FAIL during load.
    shouldFail = true;
    await expect(ctrl.resolveModel('b')).rejects.toThrow(/refusing to load/);

    // After the failure: the alias name MUST still resolve in the registry
    // (the alias binding kept the old model alive across the failed swap).
    expect(probe('claude-haiku-x')).toBeDefined();

    // 3) Now perform a successful swap to `c`. If alias state was correctly
    // restored, the haiku alias gets repointed onto c. If it wasn't, the
    // haiku alias would still be pinned to a's instance (or worse, to a
    // ghost binding) because the controller forgot it existed.
    shouldFail = false;
    await ctrl.resolveModel('c');

    const cInstance = reg.register.mock.calls.filter((c) => c[0] === 'c').slice(-1)[0]?.[1];
    expect(cInstance).toEqual({ marker: '/fake/c' });
    expect(probe('claude-haiku-x')).toBe(cInstance);
    expect(probe('claude-haiku-x')).not.toBe(aInstance);
  });

  it('lists discovered entries in name order', () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async () => ({}) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['zebra', 'alpha', 'mike']), reg as unknown as ModelRegistry, loader);

    const listed = ctrl.listModels();
    expect(listed.map((e) => e.id)).toEqual(['alpha', 'mike', 'zebra']);
    for (const entry of listed) {
      expect(entry.object).toBe('model');
      expect(entry.owned_by).toBe('mlx-node');
      expect(typeof entry.created).toBe('number');
    }
  });
});
