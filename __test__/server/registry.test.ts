import type { SessionCapableModel } from '@mlx-node/lm';
import { ModelRegistry } from '@mlx-node/server';
import { describe, expect, it, vi } from 'vite-plus/test';

/**
 * Build a minimal session-capable model mock. Every method is a vi.fn() so
 * tests can spy or stub per-method when needed.
 */
function createMockSessionModel(): SessionCapableModel {
  const emptyResult = {
    text: '',
    toolCalls: [],
    thinking: null,
    numTokens: 0,
    promptTokens: 0,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: '',
    performance: undefined,
  };
  // eslint-disable-next-line @typescript-eslint/require-await
  async function* emptyStream(): AsyncGenerator<Record<string, unknown>> {
    yield { done: true, text: '', finishReason: 'stop', toolCalls: [], numTokens: 0, promptTokens: 0 };
  }
  return {
    chatSessionStart: vi.fn().mockResolvedValue(emptyResult),
    chatSessionContinue: vi.fn().mockResolvedValue(emptyResult),
    chatSessionContinueTool: vi.fn().mockResolvedValue(emptyResult),
    chatStreamSessionStart: vi.fn(() => emptyStream()),
    chatStreamSessionContinue: vi.fn(() => emptyStream()),
    chatStreamSessionContinueTool: vi.fn(() => emptyStream()),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;
}

describe('ModelRegistry', () => {
  it('registers and retrieves a model', () => {
    const registry = new ModelRegistry();
    const mockModel = createMockSessionModel();

    registry.register('test-model', mockModel);

    expect(registry.get('test-model')).toBe(mockModel);
  });

  it('returns undefined for unknown model', () => {
    const registry = new ModelRegistry();

    expect(registry.get('nonexistent')).toBeUndefined();
  });

  it('replaces a model when registering with the same name', () => {
    const registry = new ModelRegistry();
    const model1 = createMockSessionModel();
    const model2 = createMockSessionModel();

    registry.register('test-model', model1);
    registry.register('test-model', model2);

    expect(registry.get('test-model')).toBe(model2);
  });

  it('lists all registered models in OpenAI format', () => {
    const registry = new ModelRegistry();
    registry.register('model-a', createMockSessionModel());
    registry.register('model-b', createMockSessionModel());

    const models = registry.list();

    expect(models).toHaveLength(2);
    expect(models[0].id).toBe('model-a');
    expect(models[0].object).toBe('model');
    expect(models[0].owned_by).toBe('mlx-node');
    expect(typeof models[0].created).toBe('number');
    expect(models[1].id).toBe('model-b');
  });

  it('returns empty list when no models registered', () => {
    const registry = new ModelRegistry();
    expect(registry.list()).toEqual([]);
  });

  it('unregisters a model and returns true', () => {
    const registry = new ModelRegistry();
    registry.register('model-a', createMockSessionModel());

    expect(registry.unregister('model-a')).toBe(true);
    expect(registry.get('model-a')).toBeUndefined();
  });

  it('returns false when unregistering a non-existent model', () => {
    const registry = new ModelRegistry();
    expect(registry.unregister('nonexistent')).toBe(false);
  });

  it('hasStreamSupport returns true for session-capable models', () => {
    const registry = new ModelRegistry();
    const streamModel = createMockSessionModel();

    expect(registry.hasStreamSupport(streamModel)).toBe(true);
  });

  it('hasStreamSupport returns false for objects without chatStreamSessionStart method', () => {
    const registry = new ModelRegistry();
    const noStreamModel = {
      chatSessionStart: vi.fn(),
      chatSessionContinue: vi.fn(),
      chatSessionContinueTool: vi.fn(),
      chatStreamSessionContinue: vi.fn(),
      chatStreamSessionContinueTool: vi.fn(),
      resetCaches: vi.fn(),
    } as unknown as SessionCapableModel;

    expect(registry.hasStreamSupport(noStreamModel)).toBe(false);
  });

  it('hasStreamSupport returns false when chatStreamSessionStart is not a function', () => {
    const registry = new ModelRegistry();
    const badStreamModel = {
      ...createMockSessionModel(),
      chatStreamSessionStart: 'not-a-function',
    } as unknown as SessionCapableModel;

    expect(registry.hasStreamSupport(badStreamModel)).toBe(false);
  });

  it('provisions a SessionRegistry alongside every registered model', () => {
    const registry = new ModelRegistry();
    registry.register('sess-model', createMockSessionModel());

    const sessReg = registry.getSessionRegistry('sess-model');
    expect(sessReg).toBeDefined();
    expect(sessReg!.size).toBe(0);
  });

  it('getSessionRegistry returns undefined for unknown model', () => {
    const registry = new ModelRegistry();
    expect(registry.getSessionRegistry('nonexistent')).toBeUndefined();
  });

  it('replaces the SessionRegistry when re-registering a model', () => {
    const registry = new ModelRegistry();
    registry.register('m', createMockSessionModel());
    const firstReg = registry.getSessionRegistry('m');
    registry.register('m', createMockSessionModel());
    const secondReg = registry.getSessionRegistry('m');

    expect(firstReg).toBeDefined();
    expect(secondReg).toBeDefined();
    expect(secondReg).not.toBe(firstReg);
  });

  it('listSessionRegistries returns one registry per registered model', () => {
    const registry = new ModelRegistry();
    registry.register('a', createMockSessionModel());
    registry.register('b', createMockSessionModel());

    const regs = registry.listSessionRegistries();
    expect(regs).toHaveLength(2);
  });

  it('listSessionRegistries returns empty array when no models are registered', () => {
    const registry = new ModelRegistry();
    expect(registry.listSessionRegistries()).toEqual([]);
  });

  describe('dispatch lease (Finding 1)', () => {
    it('defers binding teardown while a dispatch lease is held', () => {
      // The in-flight counter must keep the binding (and its
      // `SessionRegistry`, therefore its `execLock` FIFO mutex chain)
      // alive past an unregister() call that would otherwise fire the
      // final teardown. Without this guard, a
      // subsequent re-registration of the SAME model object before the
      // in-flight dispatch finishes would allocate a FRESH
      // `SessionRegistry` with an empty mutex chain, and a concurrent
      // request would race against the in-flight dispatch on the
      // shared native model.
      const registry = new ModelRegistry();
      const model = createMockSessionModel();
      registry.register('foo', model);

      const lease = registry.acquireDispatchLease('foo');
      expect(lease).toBeDefined();
      const leasedRegistry = lease!.registry;

      // Unregister mid-dispatch — refcount drops to zero but the
      // binding stays alive because a lease is outstanding.
      expect(registry.unregister('foo')).toBe(true);
      // The name is gone (it's unregistered), but the underlying
      // binding is still reachable to balance the lease.
      expect(registry.get('foo')).toBeUndefined();

      // Re-register the SAME model object under the same name: the
      // registry must revive the existing binding instead of
      // allocating a new one, so the session registry returned by
      // `getSessionRegistry` is identical to the one the lease holds.
      registry.register('foo', model);
      const revivedRegistry = registry.getSessionRegistry('foo');
      expect(revivedRegistry).toBe(leasedRegistry);

      // Release the lease — binding stays alive because the
      // re-registration bumped refCount back to 1.
      registry.releaseDispatchLease(lease!.model);
      expect(registry.getSessionRegistry('foo')).toBe(leasedRegistry);
    });

    it('finalizes teardown when the last lease releases after unregister', () => {
      const registry = new ModelRegistry();
      const model = createMockSessionModel();
      registry.register('foo', model);

      const lease = registry.acquireDispatchLease('foo');
      expect(lease).toBeDefined();
      const deferredRegistry = lease!.registry;

      // Unregister — refcount hits zero but teardown is deferred
      // because the lease is outstanding.
      expect(registry.unregister('foo')).toBe(true);

      // No re-registration. Release the lease: the deferred teardown
      // finalises now, and a fresh registration allocates a NEW
      // `SessionRegistry` because the previous binding has been
      // dropped from the identity map.
      registry.releaseDispatchLease(lease!.model);

      registry.register('foo', model);
      const freshRegistry = registry.getSessionRegistry('foo');
      expect(freshRegistry).toBeDefined();
      expect(freshRegistry).not.toBe(deferredRegistry);
    });

    it('shares the execLock mutex chain across lease + re-register so concurrent dispatches serialize', async () => {
      // End-to-end regression: the whole point of deferring teardown
      // is that the `withExclusive` mutex chain stays attached to the
      // SAME `SessionRegistry` object across the
      // unregister/re-register gap. Two overlapping dispatches — one
      // that acquired its lease before unregister, and one that
      // acquired after re-register — must therefore serialize on one
      // shared mutex, not race on two independent chains.
      const registry = new ModelRegistry();
      const model = createMockSessionModel();
      registry.register('foo', model);

      const leaseA = registry.acquireDispatchLease('foo');
      expect(leaseA).toBeDefined();
      const sessionRegA = leaseA!.registry;

      // Park a dispatch on sessionRegA.withExclusive via a gate.
      let releaseFirst!: () => void;
      const firstGate = new Promise<void>((resolve) => {
        releaseFirst = resolve;
      });
      let firstRan = false;
      let firstFinished = false;
      const firstPromise = sessionRegA.withExclusive(async () => {
        firstRan = true;
        await firstGate;
        firstFinished = true;
      });

      // Give the first dispatch a microtask to actually start.
      await Promise.resolve();
      expect(firstRan).toBe(true);

      // Unregister mid-dispatch and re-register the SAME model object.
      // Teardown must be deferred and the binding reused.
      expect(registry.unregister('foo')).toBe(true);
      registry.register('foo', model);
      const leaseB = registry.acquireDispatchLease('foo');
      expect(leaseB).toBeDefined();
      const sessionRegB = leaseB!.registry;
      // Shared binding invariant: both leases must expose the SAME
      // `SessionRegistry` object and therefore the SAME `execLock`.
      expect(sessionRegB).toBe(sessionRegA);

      // Fire a second dispatch on the revived registry. It must park
      // behind the first dispatch, NOT run concurrently.
      let secondStarted = false;
      const secondPromise = sessionRegB.withExclusive(async () => {
        secondStarted = true;
      });

      // Another microtask tick: the second dispatch should STILL be
      // parked because the first is still holding the mutex.
      await Promise.resolve();
      await Promise.resolve();
      expect(secondStarted).toBe(false);
      expect(firstFinished).toBe(false);

      // Release the first dispatch. The second one should run now.
      releaseFirst();
      await firstPromise;
      await secondPromise;
      expect(firstFinished).toBe(true);
      expect(secondStarted).toBe(true);

      // Release leases.
      registry.releaseDispatchLease(leaseA!.model);
      registry.releaseDispatchLease(leaseB!.model);
    });

    it('acquireDispatchLease returns undefined for unknown model', () => {
      const registry = new ModelRegistry();
      expect(registry.acquireDispatchLease('unknown')).toBeUndefined();
    });

    it('releaseDispatchLease is a no-op on an unknown model object', () => {
      const registry = new ModelRegistry();
      const model = createMockSessionModel();
      // Should not throw.
      expect(() => registry.releaseDispatchLease(model)).not.toThrow();
    });
  });

  describe('persist retention (iter-40 Finding 1)', () => {
    it('retainBinding defers teardown across unregister/re-register so the instance id survives', () => {
      // Persist retention is ORTHOGONAL to the dispatch lease: the lease releases
      // eagerly after `withExclusive` returns so a wedged `store.store(...)` cannot
      // pin abort listeners, and retainBinding keeps the binding's `modelInstanceId`
      // alive until every row the persist stamped has landed.
      const registry = new ModelRegistry();
      const model = createMockSessionModel();
      registry.register('foo', model);
      const originalId = registry.getInstanceId('foo');
      expect(typeof originalId).toBe('number');

      // Acquire + release the dispatch lease to model the eager-release path: the
      // dispatch is done, but retainBinding was called from the persist initiation
      // and we still hold a pending persist against the binding.
      const lease = registry.acquireDispatchLease('foo');
      expect(lease).toBeDefined();
      registry.retainBinding(lease!.model);
      registry.releaseDispatchLease(lease!.model);

      // Unregister then re-register the SAME model object.
      // Without retainBinding this would finalise teardown
      // (inFlight==0, refCount<=0) and the re-register would
      // mint a fresh id. With retainBinding teardown is
      // deferred and the re-register reuses the existing id.
      expect(registry.unregister('foo')).toBe(true);
      registry.register('foo', model);
      expect(registry.getInstanceId('foo')).toBe(originalId);

      // Release the persist retention. The binding stays
      // alive because the re-registration bumped refCount
      // back to 1; the id therefore remains stable.
      registry.releaseBinding(lease!.model);
      expect(registry.getInstanceId('foo')).toBe(originalId);
    });

    it('finalizes teardown only after BOTH the lease and the persist retention release', () => {
      const registry = new ModelRegistry();
      const model = createMockSessionModel();
      registry.register('foo', model);
      const originalId = registry.getInstanceId('foo');

      const lease = registry.acquireDispatchLease('foo');
      expect(lease).toBeDefined();
      registry.retainBinding(lease!.model);

      // Unregister with BOTH counters non-zero: teardown is
      // deferred on both axes.
      expect(registry.unregister('foo')).toBe(true);

      // Release the dispatch lease first. Teardown MUST stay
      // deferred because the persist retention is still held
      // — the row the persist is about to land still
      // references this id.
      registry.releaseDispatchLease(lease!.model);
      // Re-registering now must reuse the existing binding's
      // instance id because teardown has NOT finalised.
      registry.register('foo', model);
      expect(registry.getInstanceId('foo')).toBe(originalId);

      // Now unregister again, then drop the persist
      // retention. With refCount==0, inFlight==0, AND
      // pendingPersists reaching 0, teardown finalises and a
      // future registration mints a fresh id.
      expect(registry.unregister('foo')).toBe(true);
      registry.releaseBinding(lease!.model);
      registry.register('foo', model);
      expect(registry.getInstanceId('foo')).not.toBe(originalId);
    });

    it('retainBinding is a no-op on an unknown model object', () => {
      const registry = new ModelRegistry();
      const model = createMockSessionModel();
      expect(() => registry.retainBinding(model)).not.toThrow();
    });

    it('releaseBinding is a no-op on an unknown model object', () => {
      const registry = new ModelRegistry();
      const model = createMockSessionModel();
      expect(() => registry.releaseBinding(model)).not.toThrow();
    });

    it('releaseBinding without a matching retain is clamped to zero', () => {
      // Defensive invariant: a double-release (or an
      // unmatched release) must not drive pendingPersists
      // negative. The floor is clamped so an extra call is
      // a no-op rather than leaving the counter stuck below
      // zero (which would delay future legitimate teardown).
      const registry = new ModelRegistry();
      const model = createMockSessionModel();
      registry.register('foo', model);
      const originalId = registry.getInstanceId('foo');

      // Balanced retain/release — counter returns to 0.
      registry.retainBinding(model);
      registry.releaseBinding(model);
      // Extra releases beyond the matched pair must not
      // drop the counter negative, so a subsequent
      // unregister-only teardown STILL finalises (fresh id
      // on re-register proves teardown actually ran).
      registry.releaseBinding(model);
      registry.releaseBinding(model);
      expect(registry.unregister('foo')).toBe(true);
      registry.register('foo', model);
      expect(registry.getInstanceId('foo')).not.toBe(originalId);
    });
  });
});
