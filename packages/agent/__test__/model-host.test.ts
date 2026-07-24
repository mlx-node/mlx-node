import type { LoadableModel } from '@mlx-node/lm';
import { ChatSession } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { COLD_TIER_RESTORE_FAMILIES, MlxModelHost } from '../src/provider/model-host.js';
import type { DiscoveredModelLike } from '../src/types.js';

const MODELS: DiscoveredModelLike[] = [
  { name: 'qwen-small', path: '/models/qwen-small', modelType: 'qwen3_5' },
  { name: 'gemma-mid', path: '/models/gemma-mid', modelType: 'gemma4' },
];

/**
 * Stubbed loader: `new ChatSession(model)` only stores the reference and
 * these tests never drive a turn, so a plain object stands in for the
 * native model.
 */
function makeLoader() {
  return vi.fn(async (path: string) => ({ fakeModelFor: path }) as unknown as LoadableModel);
}

/**
 * The exact argument list the host must hand `resolveModelPathFn` for
 * `model`: a cold-restorable family carries the explicit persistence policy,
 * every other family carries nothing at all. Derived from
 * {@link COLD_TIER_RESTORE_FAMILIES} rather than a hard-coded family string,
 * so admitting a new family flips these expectations with it.
 */
function expectedResolveArgs(model: DiscoveredModelLike, persistPagedCache = true): unknown[] {
  return COLD_TIER_RESTORE_FAMILIES.has(model.modelType) ? [model, { persistPagedCache }] : [model];
}

/** Acquire the resident session without doing any work in the callback. */
function getSession(host: MlxModelHost, modelId: string): Promise<ChatSession> {
  return host.runWithResident(modelId, async (session) => session);
}

async function flushMicrotasks(rounds = 8): Promise<void> {
  for (let i = 0; i < rounds; i++) await Promise.resolve();
}

describe('MlxModelHost', () => {
  it('lazily loads a model once and reuses the resident session', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });
    expect(host.residentId).toBeNull();

    const first = await getSession(host, 'qwen-small');
    const second = await getSession(host, 'qwen-small');

    expect(first).toBeInstanceOf(ChatSession);
    expect(second).toBe(first);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(loader).toHaveBeenCalledWith('/models/qwen-small');
    expect(host.residentId).toBe('qwen-small');
  });

  it('forwards the loaded model image capability through ChatSession', async () => {
    const loader = vi.fn(async () => ({ supportsImages: () => true }) as unknown as LoadableModel);
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });

    const session = await getSession(host, 'qwen-small');

    expect(session.supportsImages()).toBe(true);
  });

  it('resolves the model path inside the serialized load before invoking the loader', async () => {
    const events: string[] = [];
    const resolveModelPathFn = vi.fn(async (model: DiscoveredModelLike) => {
      events.push(`resolve:${model.path}`);
      return `/paged${model.path}`;
    });
    const loader = vi.fn(async (path: string) => {
      events.push(`load:${path}`);
      return { fakeModelFor: path } as unknown as LoadableModel;
    });
    const host = new MlxModelHost(MODELS, { loadModelFn: loader, resolveModelPathFn });

    await getSession(host, 'qwen-small');

    expect(events).toEqual(['resolve:/models/qwen-small', 'load:/paged/models/qwen-small']);
    expect(resolveModelPathFn.mock.calls[0]).toEqual(expectedResolveArgs(MODELS[0]));
    expect(loader).toHaveBeenCalledWith('/paged/models/qwen-small');
  });

  it('enforces the agent paged-cache invariant before residentizing a model', async () => {
    const flatLoader = vi.fn(async () => ({ hasBlockPagedCache: () => false }) as unknown as LoadableModel);
    const host = new MlxModelHost(MODELS, {
      loadModelFn: flatLoader,
      requirePagedCache: true,
    });

    await expect(getSession(host, 'qwen-small')).rejects.toThrow(
      /qwen-small.*qwen3_5.*without an active PagedAttention cache/s,
    );
    expect(host.residentId).toBeNull();

    flatLoader.mockResolvedValueOnce({ hasBlockPagedCache: () => true } as unknown as LoadableModel);
    await expect(getSession(host, 'qwen-small')).resolves.toBeInstanceOf(ChatSession);
    expect(host.residentId).toBe('qwen-small');
  });

  it('allows Gemma4 with an attached draft to use its flat speculative executor', async () => {
    const loader = vi.fn(
      async () =>
        ({
          hasBlockPagedCache: () => false,
          hasMtpWeights: () => true,
        }) as unknown as LoadableModel,
    );
    const host = new MlxModelHost(MODELS, {
      loadModelFn: loader,
      requirePagedCache: true,
    });

    await expect(getSession(host, 'gemma-mid')).resolves.toBeInstanceOf(ChatSession);
    expect(host.residentId).toBe('gemma-mid');
  });

  it('rejects Qwen3.5 MoE MTP rather than silently routing a paged session through AR', async () => {
    const loader = vi.fn(
      async () =>
        ({
          hasBlockPagedCache: () => true,
          hasMtpWeights: () => true,
        }) as unknown as LoadableModel,
    );
    const host = new MlxModelHost([{ name: 'qwen-moe-mtp', path: '/models/qwen-moe-mtp', modelType: 'qwen3_5_moe' }], {
      loadModelFn: loader,
      requirePagedCache: true,
    });

    await expect(getSession(host, 'qwen-moe-mtp')).rejects.toThrow(
      /cannot combine MTP with PagedAttention.*refusing to silently downgrade/s,
    );
    expect(host.residentId).toBeNull();
  });

  it('rejects unknown model ids with a clear error listing known ids', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });
    const fn = vi.fn(async (session: ChatSession) => session);
    await expect(host.runWithResident('claude-haiku', fn)).rejects.toThrow(
      /unknown model "claude-haiku".*qwen-small.*gemma-mid/s,
    );
    expect(loader).not.toHaveBeenCalled();
    expect(fn).not.toHaveBeenCalled();
    expect(host.residentId).toBeNull();
  });

  it('serializes same-model callbacks in FIFO order (single load)', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });
    const order: string[] = [];
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const p1 = host.runWithResident('qwen-small', async () => {
      order.push('1-start');
      await gate;
      order.push('1-end');
      return 1;
    });
    const p2 = host.runWithResident('qwen-small', async () => {
      order.push('2-start');
      return 2;
    });

    // Flush microtasks: fn2 must not start while fn1 is parked on its gate.
    await flushMicrotasks();
    expect(order).toEqual(['1-start']);

    release();
    expect(await p1).toBe(1);
    expect(await p2).toBe(2);
    expect(order).toEqual(['1-start', '1-end', '2-start']);
    expect(loader).toHaveBeenCalledTimes(1);
  });

  it('keeps the chain and the resident alive after a callback rejection', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });
    const failing = host.runWithResident('qwen-small', async () => {
      throw new Error('task exploded');
    });
    await expect(failing).rejects.toThrow('task exploded');

    // A callback failure is NOT a load failure: the resident stays warm.
    expect(host.residentId).toBe('qwen-small');
    const after = await host.runWithResident('qwen-small', async () => 'still running');
    expect(after).toBe('still running');
    expect(loader).toHaveBeenCalledTimes(1);
  });

  it('swaps by dropping the old session and loading fresh (old session never reused)', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });

    const qwenSession = await getSession(host, 'qwen-small');
    const gemmaSession = await getSession(host, 'gemma-mid');
    expect(loader).toHaveBeenCalledTimes(2);
    expect(loader).toHaveBeenNthCalledWith(2, '/models/gemma-mid');
    expect(gemmaSession).not.toBe(qwenSession);
    expect(host.residentId).toBe('gemma-mid');

    // Swapping back must reload — the first session's refs were dropped.
    const qwenAgain = await getSession(host, 'qwen-small');
    expect(loader).toHaveBeenCalledTimes(3);
    expect(loader).toHaveBeenNthCalledWith(3, '/models/qwen-small');
    expect(qwenAgain).not.toBe(qwenSession);
    expect(host.residentId).toBe('qwen-small');
  });

  it('serializes concurrent calls for different models (no interleaved loads)', async () => {
    const inFlight: string[] = [];
    const loader = vi.fn(async (path: string) => {
      inFlight.push(`start:${path}`);
      await Promise.resolve();
      inFlight.push(`end:${path}`);
      return { fakeModelFor: path } as unknown as LoadableModel;
    });
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });

    const [a, b] = await Promise.all([getSession(host, 'qwen-small'), getSession(host, 'gemma-mid')]);
    expect(inFlight).toEqual([
      'start:/models/qwen-small',
      'end:/models/qwen-small',
      'start:/models/gemma-mid',
      'end:/models/gemma-mid',
    ]);
    expect(a).not.toBe(b);
    expect(host.residentId).toBe('gemma-mid');
  });

  it('holds the resident for the full callback: a queued swap cannot start mid-inference', async () => {
    const timeline: string[] = [];
    const loader = vi.fn(async (path: string) => {
      timeline.push(`load:${path}`);
      return { fakeModelFor: path } as unknown as LoadableModel;
    });
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });

    let releaseA!: () => void;
    const gateA = new Promise<void>((resolve) => {
      releaseA = resolve;
    });
    let residentDuringA: string | null = 'unset';
    let sessionDuringA: ChatSession | null = null;

    const a = host.runWithResident('qwen-small', async (session) => {
      timeline.push('A:fn-start');
      await gateA;
      // A's model must still be resident and its session identity intact,
      // even though B was requested while A was blocked.
      residentDuringA = host.residentId;
      sessionDuringA = session;
      timeline.push('A:fn-end');
      return session;
    });
    const b = host.runWithResident('gemma-mid', async (session) => {
      timeline.push('B:fn-start');
      return session;
    });

    // Flush microtasks: B's LOAD must not begin while A's fn is blocked.
    await flushMicrotasks();
    expect(timeline).toEqual(['load:/models/qwen-small', 'A:fn-start']);
    expect(host.residentId).toBe('qwen-small');

    releaseA();
    const [sessionA, sessionB] = await Promise.all([a, b]);
    expect(timeline).toEqual([
      'load:/models/qwen-small',
      'A:fn-start',
      'A:fn-end',
      'load:/models/gemma-mid',
      'B:fn-start',
    ]);
    expect(residentDuringA).toBe('qwen-small');
    expect(sessionA).toBe(sessionDuringA);
    expect(sessionA).not.toBe(sessionB);
    expect(host.residentId).toBe('gemma-mid');
  });

  it('tracks the post-error dirty flag: mark then consume-and-clear', async () => {
    const host = new MlxModelHost(MODELS, { loadModelFn: makeLoader() });
    await getSession(host, 'qwen-small');

    // Clean by default.
    expect(host.consumeResidentDirty('qwen-small')).toBe(false);

    host.markResidentDirty('qwen-small');
    // First consume sees it dirty…
    expect(host.consumeResidentDirty('qwen-small')).toBe(true);
    // …and clears it (read-and-clear).
    expect(host.consumeResidentDirty('qwen-small')).toBe(false);
  });

  it('markResidentDirty / consumeResidentDirty are no-ops for a non-resident model', async () => {
    const host = new MlxModelHost(MODELS, { loadModelFn: makeLoader() });
    await getSession(host, 'qwen-small');

    // A different model is not resident: marking it does nothing…
    host.markResidentDirty('gemma-mid');
    expect(host.consumeResidentDirty('gemma-mid')).toBe(false);
    // …and the actual resident stays clean.
    expect(host.consumeResidentDirty('qwen-small')).toBe(false);
  });

  it('invalidateResident drops the resident so the next call reloads', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });
    const first = await getSession(host, 'qwen-small');
    expect(loader).toHaveBeenCalledTimes(1);

    host.invalidateResident('qwen-small');
    expect(host.residentId).toBeNull();

    const reloaded = await getSession(host, 'qwen-small');
    expect(loader).toHaveBeenCalledTimes(2);
    expect(reloaded).not.toBe(first);
    expect(host.residentId).toBe('qwen-small');

    // Invalidating a non-resident model is a no-op.
    host.invalidateResident('gemma-mid');
    expect(host.residentId).toBe('qwen-small');
  });

  describe('persistPagedCache cold-tier policy', () => {
    const QWEN3: DiscoveredModelLike = { name: 'qwen3-dense', path: '/models/qwen3-dense', modelType: 'qwen3' };
    const HOST_MODELS: DiscoveredModelLike[] = [
      QWEN3,
      { name: 'gemma-mid', path: '/models/gemma-mid', modelType: 'gemma4' },
    ];

    /** Records the exact argument list the host hands the config-override seam. */
    function trackingResolver() {
      return vi.fn(async (model: DiscoveredModelLike) => `/paged${model.path}`);
    }

    it('requests qwen3 cold-tier persistence by default', async () => {
      const resolveModelPathFn = trackingResolver();
      const host = new MlxModelHost(HOST_MODELS, { loadModelFn: makeLoader(), resolveModelPathFn });

      await getSession(host, 'qwen3-dense');

      expect(resolveModelPathFn.mock.calls[0]).toEqual([QWEN3, { persistPagedCache: true }]);
    });

    it('passes an authoritative persist-off policy for qwen3 when disabled', async () => {
      const resolveModelPathFn = trackingResolver();
      const host = new MlxModelHost(HOST_MODELS, {
        loadModelFn: makeLoader(),
        resolveModelPathFn,
        persistPagedCache: false,
      });

      await getSession(host, 'qwen3-dense');

      // Tri-state: qwen3 always carries an EXPLICIT policy so the overlay can
      // authoritatively override a checkpoint whose config.json hard-codes
      // persistence on. Disabled must surface as `false`, not as omission.
      expect(resolveModelPathFn.mock.calls[0]).toEqual([QWEN3, { persistPagedCache: false }]);
    });

    it('requests cold-tier persistence by default for a hybrid family too', async () => {
      const resolveModelPathFn = trackingResolver();
      const host = new MlxModelHost(HOST_MODELS, { loadModelFn: makeLoader(), resolveModelPathFn });

      await getSession(host, 'gemma-mid');

      // gemma4 keeps sliding-window state outside the paged pool, but persists
      // it as a cold-tier sidecar and passed its restart-parity gate, so it is
      // on the allowlist and gets the same default-on policy as dense qwen3.
      // Asserted literally rather than through `expectedResolveArgs`, so this
      // still fails if the allowlist and the default ever drift apart.
      expect(resolveModelPathFn.mock.calls[0]).toEqual([HOST_MODELS[1], { persistPagedCache: true }]);
    });

    it('hands no policy at all to a family off the allowlist', async () => {
      const resolveModelPathFn = trackingResolver();
      const lfm2: DiscoveredModelLike = { name: 'lfm2-small', path: '/models/lfm2-small', modelType: 'lfm2' };
      const host = new MlxModelHost([...HOST_MODELS, lfm2], {
        loadModelFn: makeLoader(),
        resolveModelPathFn,
      });

      await getSession(host, 'lfm2-small');

      // lfm2 keeps short-conv state outside the pool with no serialization
      // path, so it is off the allowlist. It must receive NO policy — not
      // `{ persistPagedCache: false }` — so the overlay never touches the
      // field and a checkpoint's own config.json keeps whatever it set.
      expect(resolveModelPathFn.mock.calls[0]).toEqual([lfm2]);
    });
  });

  it('leaves no resident on load failure and allows a retry', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });
    await getSession(host, 'qwen-small');

    loader.mockRejectedValueOnce(new Error('load failed'));
    const fn = vi.fn(async (session: ChatSession) => session);
    await expect(host.runWithResident('gemma-mid', fn)).rejects.toThrow('load failed');
    // Drop-then-load: the old resident was released before the failed load,
    // and the callback never ran against a half-loaded model.
    expect(fn).not.toHaveBeenCalled();
    expect(host.residentId).toBeNull();

    const retried = await getSession(host, 'gemma-mid');
    expect(retried).toBeInstanceOf(ChatSession);
    expect(host.residentId).toBe('gemma-mid');
    expect(loader).toHaveBeenCalledTimes(3);
  });
});
