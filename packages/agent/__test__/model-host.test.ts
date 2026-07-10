import type { LoadableModel } from '@mlx-node/lm';
import { ChatSession } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { MlxModelHost } from '../src/provider/model-host.js';
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

describe('MlxModelHost', () => {
  it('lazily loads a model once and reuses the resident session', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });
    expect(host.residentId).toBeNull();

    const first = await host.ensureResident('qwen-small');
    const second = await host.ensureResident('qwen-small');

    expect(first).toBeInstanceOf(ChatSession);
    expect(second).toBe(first);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(loader).toHaveBeenCalledWith('/models/qwen-small');
    expect(host.residentId).toBe('qwen-small');
  });

  it('rejects unknown model ids with a clear error listing known ids', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });
    await expect(host.ensureResident('claude-haiku')).rejects.toThrow(
      /unknown model "claude-haiku".*qwen-small.*gemma-mid/s,
    );
    expect(loader).not.toHaveBeenCalled();
    expect(host.residentId).toBeNull();
  });

  it('serializes runSerialized calls in FIFO order', async () => {
    const host = new MlxModelHost(MODELS, { loadModelFn: makeLoader() });
    const order: string[] = [];
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const p1 = host.runSerialized(async () => {
      order.push('1-start');
      await gate;
      order.push('1-end');
      return 1;
    });
    const p2 = host.runSerialized(async () => {
      order.push('2-start');
      return 2;
    });

    // Flush microtasks: fn2 must not start while fn1 is parked on its gate.
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(order).toEqual(['1-start']);

    release();
    expect(await p1).toBe(1);
    expect(await p2).toBe(2);
    expect(order).toEqual(['1-start', '1-end', '2-start']);
  });

  it('keeps the chain alive after a rejected fn (rejection reaches only its caller)', async () => {
    const host = new MlxModelHost(MODELS, { loadModelFn: makeLoader() });
    const failing = host.runSerialized(async () => {
      throw new Error('task exploded');
    });
    await expect(failing).rejects.toThrow('task exploded');

    const after = await host.runSerialized(async () => 'still running');
    expect(after).toBe('still running');
  });

  it('swaps by dropping the old session and loading fresh (old session never reused)', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });

    const qwenSession = await host.ensureResident('qwen-small');
    const gemmaSession = await host.ensureResident('gemma-mid');
    expect(loader).toHaveBeenCalledTimes(2);
    expect(loader).toHaveBeenNthCalledWith(2, '/models/gemma-mid');
    expect(gemmaSession).not.toBe(qwenSession);
    expect(host.residentId).toBe('gemma-mid');

    // Swapping back must reload — the first session's refs were dropped.
    const qwenAgain = await host.ensureResident('qwen-small');
    expect(loader).toHaveBeenCalledTimes(3);
    expect(loader).toHaveBeenNthCalledWith(3, '/models/qwen-small');
    expect(qwenAgain).not.toBe(qwenSession);
    expect(host.residentId).toBe('qwen-small');
  });

  it('serializes concurrent ensureResident calls for different models (no interleaved loads)', async () => {
    const inFlight: string[] = [];
    const loader = vi.fn(async (path: string) => {
      inFlight.push(`start:${path}`);
      await Promise.resolve();
      inFlight.push(`end:${path}`);
      return { fakeModelFor: path } as unknown as LoadableModel;
    });
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });

    const [a, b] = await Promise.all([host.ensureResident('qwen-small'), host.ensureResident('gemma-mid')]);
    expect(inFlight).toEqual([
      'start:/models/qwen-small',
      'end:/models/qwen-small',
      'start:/models/gemma-mid',
      'end:/models/gemma-mid',
    ]);
    expect(a).not.toBe(b);
    expect(host.residentId).toBe('gemma-mid');
  });

  it('leaves no resident on load failure and allows a retry', async () => {
    const loader = makeLoader();
    const host = new MlxModelHost(MODELS, { loadModelFn: loader });
    await host.ensureResident('qwen-small');

    loader.mockRejectedValueOnce(new Error('load failed'));
    await expect(host.ensureResident('gemma-mid')).rejects.toThrow('load failed');
    // Drop-then-load: the old resident was released before the failed load.
    expect(host.residentId).toBeNull();

    const retried = await host.ensureResident('gemma-mid');
    expect(retried).toBeInstanceOf(ChatSession);
    expect(host.residentId).toBe('gemma-mid');
    expect(loader).toHaveBeenCalledTimes(3);
  });
});
