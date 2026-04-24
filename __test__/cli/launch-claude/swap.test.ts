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
