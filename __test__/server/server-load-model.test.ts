import type { AddressInfo } from 'node:net';

import { createServer, type ServerInstance } from '@mlx-node/server';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { runGuardedModelLoad, type GuardedLoadDeps } from '../../packages/server/src/load-model.js';
import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry, type ServableModel } from '../../packages/server/src/registry.js';

/**
 * Out-of-band model load.
 *
 * The nesting order is load-bearing, and getting it wrong is a corruption
 * bug rather than a latency bug:
 *
 *   idleSweeper.withSuspendedDrains(   ← OUTSIDE: also covers the writer-lock WAIT
 *     modelWorkCoordinator.withModelLoad(  ← INSIDE: excludes readers during
 *       load() + registry.register(...)         weight materialization
 *     ))
 *
 * If the suspension were nested INSIDE the writer lock, a load that parks
 * waiting for the lock would leave the post-request drain timer armed. That
 * timer fires `__internal__.clearCache()`, which walks the process-wide Metal
 * free pool — concurrently with the incoming `Model::load()` materializing
 * weights through the same allocator.
 *
 * `runGuardedModelLoad` is exercised directly against a recording sweeper so
 * the ordering is asserted as an ordering, not inferred from a wall clock.
 */

/** Sweeper double that records suspend/release transitions in call order. */
function createRecordingSweeper(): {
  sweeper: GuardedLoadDeps['idleSweeper'];
  events: string[];
  suspendDepth: () => number;
} {
  const events: string[] = [];
  let depth = 0;
  function withSuspendedDrains<T>(fn: () => T | Promise<T>): T | Promise<T> {
    depth += 1;
    events.push('suspend');
    let result: T | Promise<T>;
    try {
      result = fn();
    } catch (err) {
      depth -= 1;
      events.push('release');
      throw err;
    }
    if (result != null && typeof (result as Promise<T>).then === 'function') {
      return Promise.resolve(result).finally(() => {
        depth -= 1;
        events.push('release');
      }) as Promise<T>;
    }
    depth -= 1;
    events.push('release');
    return result;
  }
  return {
    sweeper: { withSuspendedDrains } as GuardedLoadDeps['idleSweeper'],
    events,
    suspendDepth: () => depth,
  };
}

function fakeModel(tag: string): ServableModel {
  return { tag, chatStreamSessionStart(): void {} } as unknown as ServableModel;
}

describe('runGuardedModelLoad ordering', () => {
  it('suspends drains BEFORE waiting on the writer lock and releases only after register', async () => {
    const { sweeper, events } = createRecordingSweeper();
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const registry = new ModelRegistry();

    // Occupy the writer slot so the load under test must park in
    // `acquireWrite()`. This is the exact window the outer suspension exists
    // to cover.
    let releaseIncumbent: (() => void) | undefined;
    const incumbentHeld = new Promise<void>((r) => {
      releaseIncumbent = r;
    });
    const incumbent = modelWorkCoordinator.withModelLoad(() => incumbentHeld, 'incumbent');
    await Promise.resolve();
    expect(modelWorkCoordinator.writerActive).toBe(true);

    const load = vi.fn(async () => {
      events.push('load');
      await Promise.resolve();
      return fakeModel('new');
    });
    const pending = runGuardedModelLoad(
      { idleSweeper: sweeper, modelWorkCoordinator, registry },
      { name: 'newcomer', load },
    );

    // Let the guarded load reach `acquireWrite()` and block there.
    await Promise.resolve();
    await Promise.resolve();
    expect(events).toEqual(['suspend']);
    expect(load).not.toHaveBeenCalled();
    expect(modelWorkCoordinator.waitingWriters).toBe(1);

    releaseIncumbent?.();
    await incumbent;
    await pending;

    expect(events).toEqual(['suspend', 'load', 'release']);
    expect(registry.get('newcomer')).toBeDefined();
  });

  it('holds the suspension across the ENTIRE load, register included', async () => {
    const { sweeper, suspendDepth } = createRecordingSweeper();
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const registry = new ModelRegistry();

    let depthAtLoad = -1;
    let depthAtRegister = -1;
    const originalRegister = registry.register.bind(registry);
    registry.register = (...args: Parameters<typeof originalRegister>) => {
      depthAtRegister = suspendDepth();
      return originalRegister(...args);
    };

    await runGuardedModelLoad(
      { idleSweeper: sweeper, modelWorkCoordinator, registry },
      {
        name: 'm',
        load: async () => {
          depthAtLoad = suspendDepth();
          await Promise.resolve();
          return fakeModel('m');
        },
      },
    );

    expect(depthAtLoad).toBe(1);
    expect(depthAtRegister).toBe(1);
    expect(suspendDepth()).toBe(0);
  });

  it('excludes readers for the whole materialization window', async () => {
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const { sweeper } = createRecordingSweeper();
    const registry = new ModelRegistry();

    const order: string[] = [];
    let releaseLoad: (() => void) | undefined;
    const loadGate = new Promise<void>((r) => {
      releaseLoad = r;
    });

    const load = runGuardedModelLoad(
      { idleSweeper: sweeper, modelWorkCoordinator, registry },
      {
        name: 'm',
        load: async () => {
          order.push('load:start');
          await loadGate;
          order.push('load:end');
          return fakeModel('m');
        },
      },
    );
    await Promise.resolve();
    await Promise.resolve();

    const reader = modelWorkCoordinator.withInference(() => {
      order.push('reader');
    });

    // Give the reader every chance to sneak in while the writer holds the lock.
    await Promise.resolve();
    await new Promise((r) => setImmediate(r));
    expect(order).toEqual(['load:start']);

    releaseLoad?.();
    await load;
    await reader;
    expect(order).toEqual(['load:start', 'load:end', 'reader']);
  });

  it('releases BOTH the suspension and the writer lock when the load throws', async () => {
    const { sweeper, events, suspendDepth } = createRecordingSweeper();
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const registry = new ModelRegistry();

    await expect(
      runGuardedModelLoad(
        { idleSweeper: sweeper, modelWorkCoordinator, registry },
        {
          name: 'doomed',
          load: () => Promise.reject(new Error('weights not found')),
        },
      ),
    ).rejects.toThrow('weights not found');

    expect(suspendDepth()).toBe(0);
    expect(events).toEqual(['suspend', 'release']);
    expect(modelWorkCoordinator.writerActive).toBe(false);
    expect(registry.get('doomed')).toBeUndefined();

    // The coordinator must still be usable — a leaked writer lock would
    // deadlock every subsequent request.
    await expect(modelWorkCoordinator.withInference(() => 'served')).resolves.toBe('served');
    await expect(modelWorkCoordinator.withModelLoad(() => 'loaded', 'retry')).resolves.toBe('loaded');
  });

  it('releases both when register itself throws', async () => {
    const { sweeper, suspendDepth } = createRecordingSweeper();
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const registry = new ModelRegistry();
    registry.register = () => {
      throw new Error('registry exploded');
    };

    await expect(
      runGuardedModelLoad(
        { idleSweeper: sweeper, modelWorkCoordinator, registry },
        { name: 'm', load: () => Promise.resolve(fakeModel('m')) },
      ),
    ).rejects.toThrow('registry exploded');

    expect(suspendDepth()).toBe(0);
    expect(modelWorkCoordinator.writerActive).toBe(false);
  });

  it('records the load under its label for /health', async () => {
    const { sweeper } = createRecordingSweeper();
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const registry = new ModelRegistry();

    await runGuardedModelLoad(
      { idleSweeper: sweeper, modelWorkCoordinator, registry },
      { name: 'labelled-model', load: () => Promise.resolve(fakeModel('m')) },
    );
    expect(modelWorkCoordinator.lastLoad).toMatchObject({ label: 'labelled-model', ok: true, error: null });

    await expect(
      runGuardedModelLoad(
        { idleSweeper: sweeper, modelWorkCoordinator, registry },
        { name: 'bad-model', load: () => Promise.reject(new Error('nope')) },
      ),
    ).rejects.toThrow('nope');
    expect(modelWorkCoordinator.lastLoad).toMatchObject({ label: 'bad-model', ok: false });
    expect(modelWorkCoordinator.lastLoad?.error).toMatch(/nope/);
  });

  it('binds every alias to the same instance inside the same bracket', async () => {
    const { sweeper, events } = createRecordingSweeper();
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const registry = new ModelRegistry();

    await runGuardedModelLoad(
      { idleSweeper: sweeper, modelWorkCoordinator, registry },
      { name: 'primary', aliases: ['alias-a', 'alias-b', 'primary'], load: () => Promise.resolve(fakeModel('m')) },
    );

    const primary = registry.get('primary');
    expect(primary).toBeDefined();
    expect(registry.get('alias-a')).toBe(primary);
    expect(registry.get('alias-b')).toBe(primary);
    // Aliases share the single-warm session binding.
    expect(registry.getSessionRegistry('alias-a')).toBe(registry.getSessionRegistry('primary'));
    expect(events).toEqual(['suspend', 'release']);
  });

  it('forwards samplingDefaults and maxOutputTokens to the binding', async () => {
    const { sweeper } = createRecordingSweeper();
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const registry = new ModelRegistry();

    await runGuardedModelLoad(
      { idleSweeper: sweeper, modelWorkCoordinator, registry },
      {
        name: 'tuned',
        load: () => Promise.resolve(fakeModel('m')),
        samplingDefaults: { temperature: 0.6, topP: 0.95 },
        maxOutputTokens: 4096,
      },
    );

    const sessionReg = registry.getSessionRegistry('tuned');
    expect(sessionReg?.defaultSamplingConfig).toMatchObject({ temperature: 0.6, topP: 0.95 });
    expect(sessionReg?.outputTokenLimit).toBe(4096);
  });
});

describe('ServerInstance.loadModel', () => {
  let servers: ServerInstance[] = [];

  afterEach(async () => {
    const pending = servers;
    servers = [];
    for (const instance of pending) {
      await instance.close({ timeoutMs: 250 }).catch(() => {});
    }
  });

  async function start(config: Parameters<typeof createServer>[0] = {}): Promise<ServerInstance> {
    const instance = await createServer({ port: 0, host: '127.0.0.1', disableStore: true, ...config });
    servers.push(instance);
    return instance;
  }

  it('exposes the coordinator so callers can compose their own brackets', async () => {
    const instance = await start();
    expect(instance.modelWork).toBeInstanceOf(ModelWorkCoordinator);
  });

  it('registers the model and surfaces it on /v1/models and /health', async () => {
    const instance = await start();
    await instance.loadModel({ name: 'served', load: () => Promise.resolve(fakeModel('m')) });

    expect(instance.registry.get('served')).toBeDefined();
    const { port } = instance.server.address() as AddressInfo;
    const listed = (await (await fetch(`http://127.0.0.1:${port}/v1/models`)).json()) as { data: { id: string }[] };
    expect(listed.data.map((m) => m.id)).toContain('served');
    expect(instance.health().models.resident).toContain('served');
    expect(instance.health().lastLoad).toMatchObject({ label: 'served', ok: true });
  });

  it("reports 'loading' on /health for the duration of the load", async () => {
    const instance = await start();
    let releaseLoad: (() => void) | undefined;
    const gate = new Promise<void>((r) => {
      releaseLoad = r;
    });
    const pending = instance.loadModel({
      name: 'slow',
      load: async () => {
        await gate;
        return fakeModel('m');
      },
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(instance.health().status).toBe('loading');
    releaseLoad?.();
    await pending;
    expect(instance.health().status).toBe('ok');
  });

  it("reports 'error' on /health after a failed load with nothing resident", async () => {
    const instance = await start();
    await expect(
      instance.loadModel({ name: 'broken', load: () => Promise.reject(new Error('no weights')) }),
    ).rejects.toThrow('no weights');

    const health = instance.health();
    expect(health.status).toBe('error');
    expect(health.lastLoad).toMatchObject({ label: 'broken', ok: false });
    expect(instance.modelWork.writerActive).toBe(false);
  });

  it('keeps serving after a failed load when a model was already resident', async () => {
    const instance = await start();
    await instance.loadModel({ name: 'good', load: () => Promise.resolve(fakeModel('good')) });
    await expect(instance.loadModel({ name: 'bad', load: () => Promise.reject(new Error('boom')) })).rejects.toThrow(
      'boom',
    );
    expect(instance.health().status).toBe('ok');
    expect(instance.registry.get('good')).toBeDefined();
  });
});
