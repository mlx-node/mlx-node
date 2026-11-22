import type { LoadableModel } from '@mlx-node/lm';
import { MODEL_FAMILY_DATA, familyDataFor } from '@mlx-node/lm/family-data';
import { describe, expect, it, vi } from 'vite-plus/test';

import {
  chatCapability,
  withCapabilityModels,
  type CapabilityAdapter,
} from '../../../packages/server/src/host/capability-models.js';
import type { DiscoveredModel } from '../../../packages/server/src/host/discover.js';
import { makeSwapController } from '../../../packages/server/src/host/swap.js';
import { ModelWorkCoordinator } from '../../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry } from '../../../packages/server/src/registry.js';

interface FakeRegistry {
  getAny: ReturnType<typeof vi.fn<(name: string) => unknown>>;
  registerAny: ReturnType<typeof vi.fn>;
  get: ReturnType<typeof vi.fn>;
  register: ReturnType<typeof vi.fn>;
  unregister: ReturnType<typeof vi.fn>;
  list: () => { id: string }[];
  listSessionRegistries: () => [];
  settleModelDisposals: ReturnType<typeof vi.fn<() => Promise<void>>>;
  servedContext: () => undefined;
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
  return {
    get,
    register,
    getAny: get,
    registerAny: register,
    unregister,
    list: () => [...resident.keys()].map((id) => ({ id })),
    listSessionRegistries: () => [],
    settleModelDisposals: vi.fn(async () => undefined),
    servedContext: () => undefined,
  };
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
    contextWindow: 262144,
    supportsImages: false,
    base: n,
    variant: 'MLX',
    format: 'mlx' as const,
  }));
}

describe('makeSwapController', () => {
  it('registers the target binding with its identity', async () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (path: string) => ({ marker: path }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'a');
    await ctrl.resolveModel('b');
    await ctrl.resolveModel('claude-haiku-4-5');

    const calls = reg.registerAny.mock.calls as Array<[string, unknown, { identity?: unknown }]>;
    const target = calls.find(([name]) => name === 'b');
    expect(target?.[2].identity).toEqual({ modelId: 'b', modelType: 'qwen3_5', contextWindow: 262144 });
    const alias = calls.find(([name]) => name === 'claude-haiku-4-5');
    expect(alias).toBeDefined();
    expect(alias?.[2].identity).toBeUndefined();
  });

  it.each(['asr', 'rerank'] as const)('preserves the selected chat target after %s evicts it', async (kind) => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (path: string) => ({ marker: path }) as unknown as LoadableModel);
    const chat = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'a');
    let otherResident = false;
    const other: CapabilityAdapter = {
      kind,
      listModels: () => [{ id: 'other', object: 'model', created: 0, owned_by: 'test', task: kind }],
      hasResident: () => otherResident,
      resolveModel: async () => {
        otherResident = true;
      },
      disposeAll: async () => {
        otherResident = false;
      },
    };
    const ctrl = withCapabilityModels([chatCapability(chat, reg as unknown as ModelRegistry), other], 'single');
    await ctrl.resolveModel('a');
    await ctrl.resolveModel('b');
    await ctrl.resolveModel('old-alias');
    await ctrl.resolveModel('other');
    expect(reg.list()).toEqual([]);
    expect(otherResident).toBe(true);
    await ctrl.resolveModel('new-alias');
    expect(otherResident).toBe(false);
    expect(loader.mock.calls.map(([path]) => path)).toEqual(['/fake/a', '/fake/b', '/fake/b']);
    expect(reg.getAny('new-alias')).toEqual({ marker: '/fake/b' });
    expect(reg.getAny('old-alias')).toBeUndefined();
  });
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
    expect(reg.unregister).toHaveBeenCalledWith('a', { dispose: true });
    expect(reg.register.mock.calls.map((c) => c[0])).toEqual(['a', 'b']);
  });

  it('waits for native disposal before loading the next resident', async () => {
    const registry = new ModelRegistry();
    const events: string[] = [];
    const a = {
      dispose: vi.fn(async () => {
        events.push('dispose:a:start');
        await Promise.resolve();
        events.push('dispose:a:end');
      }),
    };
    const b = {};
    const loader = vi.fn(async (path: string) => {
      events.push(`load:${path}`);
      return (path.endsWith('/a') ? a : b) as unknown as LoadableModel;
    });
    const ctrl = makeSwapController(discovered(['a', 'b']), registry, loader);

    await ctrl.resolveModel('a');
    await ctrl.resolveModel('b');

    expect(events).toEqual(['load:/fake/a', 'dispose:a:start', 'dispose:a:end', 'load:/fake/b']);
  });

  it('disposes an aliased resident after re-pointing its aliases', async () => {
    const registry = new ModelRegistry();
    const events: string[] = [];
    const a = { dispose: vi.fn(async () => void events.push('dispose:a')) };
    const b = {};
    const loader = vi.fn(async (path: string) => {
      events.push(`load:${path}`);
      return (path.endsWith('/a') ? a : b) as unknown as LoadableModel;
    });
    const ctrl = makeSwapController(discovered(['a', 'b']), registry, loader);

    await ctrl.resolveModel('a');
    await ctrl.resolveModel('claude-haiku-x');
    await ctrl.resolveModel('b');

    expect(events).toEqual(['load:/fake/a', 'load:/fake/b', 'dispose:a']);
    expect(a.dispose).toHaveBeenCalledTimes(1);
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

  // Mutation: list `entry.contextWindow` always → red after the load.
  it('reports the served window after load and the trained one before and after eviction', async () => {
    const registry = new ModelRegistry();
    const model = {
      chatSessionStart: vi.fn(),
      chatStreamSessionStart: vi.fn(),
      hasMtpWeights: () => true,
      contextLimits: () => ({
        trainedWindowTokens: 262144,
        effectiveWindowTokens: 73696,
        pagedBlockCapacity: 0,
        pagedBlockSize: 16,
        kvFormat: 'bf16' as const,
      }),
    };
    const ctrl = makeSwapController(
      discovered(['a']),
      registry,
      vi.fn(async () => model as unknown as LoadableModel),
    );
    const windows = () => {
      const [entry] = ctrl.listModels();
      return [entry!.context_window, entry!.max_model_len, entry!.context_length];
    };
    expect(windows()).toEqual([262144, 262144, 262144]);
    await ctrl.resolveModel('a');
    expect(windows()).toEqual([73696, 73696, 73696]);
    registry.unregister('a');
    expect(windows()).toEqual([262144, 262144, 262144]);
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

  it('disposeAll empties the registry and the aliases but keeps the selection for the next unknown name', async () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (path: string) => ({ marker: path }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'a');
    await ctrl.resolveModel('b');
    await ctrl.resolveModel('mlx-local');
    expect(reg.list().map((e) => e.id)).toEqual(['b', 'mlx-local']);

    await ctrl.disposeAll!();
    expect(reg.list()).toEqual([]);

    await ctrl.resolveModel('mlx-local');
    expect(loader.mock.calls.map(([path]) => path)).toEqual(['/fake/b', '/fake/b']);
    expect(reg.list().map((e) => e.id)).toEqual(['b', 'mlx-local']);
  });

  it("closes a dropped model's SSD cache before the same model loads again", async () => {
    const reg = fakeRegistry();
    const events: string[] = [];
    const deadlines: number[] = [];
    const loader = vi.fn(async (path: string) => {
      events.push(`load ${path}`);
      return {
        closeCache: async (deadlineMs: number) => {
          deadlines.push(deadlineMs - Date.now());
          events.push(`close ${path}`);
          await new Promise((resolve) => setTimeout(resolve, 20));
          events.push(`closed ${path}`);
          return true;
        },
      } as unknown as LoadableModel;
    });
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'a');
    await ctrl.resolveModel('a');
    await ctrl.resolveModel('b');
    await ctrl.resolveModel('a');
    expect(events).toEqual(['load /fake/a', 'load /fake/b', 'close /fake/a', 'closed /fake/a', 'load /fake/a']);

    events.length = 0;
    await ctrl.disposeAll!();
    await ctrl.resolveModel('a');
    expect(events).toEqual(['close /fake/a', 'closed /fake/a', 'load /fake/a']);
    for (const left of deadlines) expect(left).toBeGreaterThan(5000);
  });
});

describe('bindLoadedAlias', () => {
  /** Loads `/fake/b` only once `openB` is called. */
  function gatedLoader(): {
    loader: ReturnType<typeof vi.fn<(path: string) => Promise<LoadableModel>>>;
    openB: () => void;
  } {
    let openB!: () => void;
    const gate = new Promise<void>((resolve) => {
      openB = resolve;
    });
    const loader = vi.fn(async (path: string) => {
      if (path === '/fake/b') await gate;
      return { marker: path } as unknown as LoadableModel;
    });
    return { loader, openB };
  }

  it('binds an unknown name to the loaded model with no load', async () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (path: string) => ({ marker: path }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'a');
    await ctrl.resolveModel('a');

    expect(ctrl.bindLoadedAlias!('mlx-local')).toBe(true);
    expect(reg.getAny('mlx-local')).toBe(reg.getAny('a'));
    expect(ctrl.bindLoadedAlias!('b')).toBe(false);
    expect(loader).toHaveBeenCalledTimes(1);
  });

  // Mutation: the residency guard removed from `bindLoadedAlias` → red.
  it('binds nothing before the first load and after an unload', async () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (path: string) => ({ marker: path }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'a');
    expect(ctrl.bindLoadedAlias!('mlx-local')).toBe(false);
    await ctrl.resolveModel('a');
    await ctrl.disposeAll!();
    expect(ctrl.bindLoadedAlias!('mlx-local')).toBe(false);
    expect(reg.list()).toEqual([]);
  });

  // Mutation: `bindLoadedAlias` does not add the name to the alias set → red.
  it('a bound alias follows the next swap', async () => {
    const reg = fakeRegistry();
    const loader = vi.fn(async (path: string) => ({ marker: path }) as unknown as LoadableModel);
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'a');
    await ctrl.resolveModel('a');
    expect(ctrl.bindLoadedAlias!('mlx-local')).toBe(true);

    await ctrl.resolveModel('b');
    expect(reg.getAny('mlx-local')).toEqual({ marker: '/fake/b' });
  });

  // Mutation: `bindLoadedAlias` ignores `pending` → red.
  it('binds nothing while a swap or an unload is in progress', async () => {
    const reg = fakeRegistry();
    const { loader, openB } = gatedLoader();
    const ctrl = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'a');
    await ctrl.resolveModel('a');

    const swap = ctrl.resolveModel('b');
    expect(ctrl.bindLoadedAlias!('mlx-local')).toBe(false);
    openB();
    await swap;
    expect(ctrl.bindLoadedAlias!('mlx-local')).toBe(true);
    expect(reg.getAny('mlx-local')).toEqual({ marker: '/fake/b' });

    const unload = ctrl.disposeAll!();
    expect(ctrl.bindLoadedAlias!('other-alias')).toBe(false);
    await unload;
    expect(reg.list()).toEqual([]);
  });

  // Mutation: `withCapabilityModels` forwards in 'single' residency → red.
  it('the capability wrapper forwards unknown names only, and never under single residency', async () => {
    for (const residency of ['capability', 'single'] as const) {
      const reg = fakeRegistry();
      const loader = vi.fn(async (path: string) => ({ marker: path }) as unknown as LoadableModel);
      const chat = makeSwapController(discovered(['a', 'b']), reg as unknown as ModelRegistry, loader, 'a');
      const ctrl = withCapabilityModels([chatCapability(chat, reg as unknown as ModelRegistry)], residency);
      await ctrl.resolveModel('a');
      expect(ctrl.bindLoadedAlias!('b')).toBe(false);
      expect(ctrl.bindLoadedAlias!('mlx-local')).toBe(residency === 'capability');
    }
  });
});

describe('makeSwapController, a target that loads alone', () => {
  interface Stub {
    path: string;
    disposed: boolean;
    chatSessionStart: () => void;
    chatStreamSessionStart: () => void;
    dispose: () => Promise<void>;
  }

  function stub(path: string, events: string[]): Stub {
    const model: Stub = {
      path,
      disposed: false,
      chatSessionStart: vi.fn(),
      chatStreamSessionStart: vi.fn(),
      dispose: async () => {
        events.push(`dispose ${path}`);
        await new Promise((resolve) => setTimeout(resolve, 20));
        model.disposed = true;
        events.push(`disposed ${path}`);
      },
    };
    return model;
  }

  /** `old` then `flash` (which loads alone) with the alias `mlx-local` bound on `old`. */
  async function setup(opts: { fail?: (path: string) => boolean; alone?: (name: string) => boolean } = {}) {
    const registry = new ModelRegistry();
    const events: string[] = [];
    const models = new Map<string, Stub>();
    const seen: Array<{ path: string; local: unknown; old: unknown }> = [];
    const asked: string[] = [];
    const loader = vi.fn(async (path: string) => {
      seen.push({ path, local: registry.getAny('mlx-local'), old: registry.getAny('old') });
      events.push(`load ${path}`);
      if (opts.fail?.(path) === true) throw new Error(`refusing to load ${path}`);
      const model = stub(path, events);
      models.set(path, model);
      return model as unknown as LoadableModel;
    });
    const ctrl = makeSwapController(
      discovered(['flash', 'old', 'other']),
      registry,
      loader,
      'old',
      undefined,
      (entry) => {
        asked.push(entry.name);
        return (opts.alone ?? ((name) => name === 'flash'))(entry.name);
      },
    );
    await ctrl.resolveModel('old');
    await ctrl.resolveModel('mlx-local');
    seen.length = 0;
    events.length = 0;
    return { registry, events, models, seen, asked, loader, ctrl, old: models.get('/fake/old')! };
  }

  // MUTATION: skip the wait for `settleModelDisposals` -> the loader runs before the dispose resolved -> red.
  it('disposes the idle old model and every alias, and waits for it, before the load', async () => {
    const { registry, events, seen, ctrl, old } = await setup();
    await ctrl.resolveModel('flash');
    expect(seen).toEqual([{ path: '/fake/flash', local: undefined, old: undefined }]);
    expect(events).toEqual(['dispose /fake/old', 'disposed /fake/old', 'load /fake/flash']);
    expect(old.disposed).toBe(true);
    expect(registry.getAny('flash')).toBeDefined();
  });

  // MUTATION: the carried aliases are not bound to the new model -> `mlx-local` is unregistered -> red.
  it('binds every carried alias to the new model after that load', async () => {
    const { registry, models, ctrl, loader } = await setup();
    await ctrl.resolveModel('flash');
    expect(registry.getAny('mlx-local')).toBe(models.get('/fake/flash'));
    await ctrl.resolveModel('mlx-local');
    expect(loader.mock.calls.map(([path]) => path)).toEqual(['/fake/old', '/fake/flash']);
  });

  // MUTATION: `bindLoadedAlias` ignores `pending` -> the name binds to the old model the swap disposes -> red.
  it('binds no new name while the swap runs, then binds it to the new model', async () => {
    const { registry, events, models, ctrl, loader } = await setup();
    let open!: () => void;
    const gate = new Promise<void>((resolve) => {
      open = resolve;
    });
    let started!: () => void;
    const loading = new Promise<void>((resolve) => {
      started = resolve;
    });
    loader.mockImplementationOnce(async (path: string) => {
      events.push(`load ${path}`);
      started();
      await gate;
      const model = stub(path, events);
      models.set(path, model);
      return model as unknown as LoadableModel;
    });
    const swap = ctrl.resolveModel('flash');
    expect(ctrl.bindLoadedAlias!('claude-haiku')).toBe(false);
    await loading;
    expect(events).toEqual(['dispose /fake/old', 'disposed /fake/old', 'load /fake/flash']);
    expect(ctrl.bindLoadedAlias!('claude-haiku')).toBe(false);
    expect(registry.getAny('claude-haiku')).toBeUndefined();
    open();
    await swap;
    expect(registry.getAny('claude-haiku')).toBeUndefined();
    expect(ctrl.bindLoadedAlias!('claude-haiku')).toBe(true);
    expect(registry.getAny('claude-haiku')).toBe(models.get('/fake/flash'));
  });

  // MUTATION: the selection moves to the target before the load -> the next alias request loads the target -> red.
  it('leaves nothing resident when the load fails, and the next alias request loads the old entry again', async () => {
    let failing = true;
    const { registry, events, ctrl, loader } = await setup({ fail: (path) => failing && path === '/fake/flash' });
    await expect(ctrl.resolveModel('flash')).rejects.toThrow('refusing to load');
    expect(events).toEqual(['dispose /fake/old', 'disposed /fake/old', 'load /fake/flash']);
    expect(registry.list()).toEqual([]);
    expect(registry.getAny('mlx-local')).toBeUndefined();
    failing = false;
    await ctrl.resolveModel('mlx-local');
    expect(loader.mock.calls.map(([path]) => path)).toEqual(['/fake/old', '/fake/flash', '/fake/old']);
    expect(registry.getAny('old')).toBeDefined();
    expect(registry.getAny('mlx-local')).toBe(registry.getAny('old'));
  });

  // MUTATION: the busy check is removed -> the loader sees the alias gone -> red.
  it.each(['dispatch lease', 'persist retention'] as const)(
    'keeps the aliases on the old model through the load while it holds a %s, then disposes it at the release',
    async (hold) => {
      const { registry, events, models, seen, ctrl, old } = await setup();
      const release =
        hold === 'dispatch lease'
          ? (() => {
              const lease = registry.acquireDispatchLease('mlx-local')!;
              return () => registry.releaseDispatchLease(lease.model);
            })()
          : (() => {
              const model = registry.getAny('mlx-local') as Parameters<ModelRegistry['retainBinding']>[0];
              registry.retainBinding(model);
              return () => registry.releaseBinding(model);
            })();
      await ctrl.resolveModel('flash');
      expect(seen).toEqual([{ path: '/fake/flash', local: old, old: undefined }]);
      expect(events).toEqual(['load /fake/flash']);
      expect(old.disposed).toBe(false);
      expect(registry.getAny('mlx-local')).toBe(models.get('/fake/flash'));
      release();
      await registry.settleModelDisposals();
      expect(old.disposed).toBe(true);
    },
  );

  // MUTATION: the predicate is taken as always true -> the loader sees the alias gone -> red.
  it('keeps the aliases on the old model until a target that does not load alone has loaded, then disposes it', async () => {
    const { events, models, seen, ctrl, old, registry } = await setup();
    await ctrl.resolveModel('other');
    expect(seen).toEqual([{ path: '/fake/other', local: old, old: undefined }]);
    expect(events).toEqual(['load /fake/other', 'dispose /fake/old', 'disposed /fake/old']);
    expect(registry.getAny('mlx-local')).toBe(models.get('/fake/other'));
  });

  it('asks nothing and disposes nothing on a first load or a load of the same model', async () => {
    const { events, asked, ctrl, registry, old } = await setup();
    expect(asked).toEqual([]);
    await ctrl.resolveModel('old');
    expect(events).toEqual([]);
    expect(old.disposed).toBe(false);
    await ctrl.releaseAll!();
    events.length = 0;
    await ctrl.resolveModel('mlx-local');
    expect(asked).toEqual([]);
    expect(events).toEqual(['load /fake/old']);
    expect(registry.getAny('mlx-local')).toBe(registry.getAny('old'));
  });

  it('flags qwen4_exp and hy_v4_flash alone among the families', () => {
    const alone = MODEL_FAMILY_DATA.map((family) => family.id).filter((id) => familyDataFor(id)?.loadsAlone === true);
    expect(alone).toEqual(['qwen4_exp', 'hy_v4_flash']);
  });
});

describe('makeSwapController, a request resolved just before a switch to a target that loads alone', () => {
  interface Stub {
    path: string;
    disposed: boolean;
    chatSessionStart: () => void;
    chatStreamSessionStart: () => void;
    dispose: () => Promise<void>;
  }

  const later = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 10));

  /** The host's wiring: a real registry with the coordinator attached; `old` resident unless `cold`. */
  async function setup(cold: boolean) {
    const registry = new ModelRegistry();
    const coordinator = new ModelWorkCoordinator();
    registry.setModelLoadAdmissionCoordinator(coordinator);
    const models = new Map<string, Stub>();
    const seen: Array<{ path: string; alias: unknown }> = [];
    const loader = vi.fn(async (path: string) => {
      seen.push({ path, alias: registry.getAny('claude-haiku') ?? registry.getAny('mlx-local') });
      const model: Stub = {
        path,
        disposed: false,
        chatSessionStart: vi.fn(),
        chatStreamSessionStart: vi.fn(),
        dispose: async () => {
          model.disposed = true;
        },
      };
      models.set(path, model);
      return model as unknown as LoadableModel;
    });
    const ctrl = makeSwapController(
      discovered(['flash', 'old']),
      registry,
      loader,
      'old',
      undefined,
      (entry) => entry.name === 'flash',
    );
    if (!cold) {
      await ctrl.resolveModel('old');
      await ctrl.resolveModel('mlx-local');
    }
    seen.length = 0;
    const load = (name: string): Promise<void> => coordinator.withModelLoad(() => ctrl.resolveModel(name), name);
    return { registry, coordinator, models, seen, ctrl, load };
  }

  /** The lease the request takes: a live model, held until `release`. */
  function lease(registry: ModelRegistry, name: string): { model: Stub; release: () => void } {
    const held = registry.acquireDispatchLease(name);
    expect(held, `${name} resolves to a model`).toBeDefined();
    const model = held!.model as unknown as Stub;
    expect(model.disposed).toBe(false);
    return { model, release: () => registry.releaseDispatchLease(held!.model) };
  }

  // MUTATION swap-no-permit: the pre-dispatch permits are not counted -> the alias is gone -> red.
  it('cold start: an alias request holds its permit past the resolve, then the switch is queued', async () => {
    const { registry, coordinator, models, seen, load } = await setup(true);
    const request = (async () => {
      const admission = coordinator.beginRequestLoadAdmission('claude-haiku');
      await load('claude-haiku');
      await later();
      const held = lease(registry, 'claude-haiku');
      admission.release();
      return held;
    })();
    const swap = load('flash');
    const held = await request;
    await swap;
    held.release();
    expect(seen).toEqual([
      { path: '/fake/old', alias: undefined },
      { path: '/fake/flash', alias: models.get('/fake/old') },
    ]);
    expect(registry.getAny('claude-haiku')).toBe(models.get('/fake/flash'));
  });

  // MUTATION swap-no-permit -> red.
  it('old resident: a bound alias request takes its permit and the switch comes in the same tick', async () => {
    const { registry, models, seen, load } = await setup(false);
    const request = (async () => {
      const admission = registry.getSessionRegistry('mlx-local')!.beginPreDispatchAdmission();
      await later();
      const held = lease(registry, 'mlx-local');
      admission.release();
      return held;
    })();
    const swap = load('flash');
    const held = await request;
    await swap;
    held.release();
    expect(seen).toEqual([{ path: '/fake/flash', alias: models.get('/fake/old') }]);
  });

  // MUTATION swap-no-yield: the switch decides before the resolved request resumes -> red.
  it('old resident, no coordinator: a new alias request takes its lease as it resumes, then the switch', async () => {
    const { registry, models, seen, ctrl } = await setup(false);
    const request = (async () => {
      await ctrl.resolveModel('claude-haiku');
      return lease(registry, 'claude-haiku');
    })();
    const swap = ctrl.resolveModel('flash');
    const held = await request;
    expect(held.model).toBe(models.get('/fake/old'));
    await swap;
    held.release();
    expect(seen).toEqual([{ path: '/fake/flash', alias: models.get('/fake/old') }]);
  });
});
