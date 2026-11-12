import { __createIdleSweeper as createIdleSweeper } from '@mlx-node/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  createWeightResidency,
  trackWeights,
  WEIGHT_RELEASE_RETRY_MS,
  type WeightResidencyOptions,
} from '../../packages/server/src/weight-residency.js';

/**
 * The weights' idle release on a mock clock (Splash `WeightResidency`,
 * `weight_residency_test.cpp` `testReleaseOnlyHeldWeightsOfAnIdleEngine`):
 * released only after the whole limit without a request, never while a
 * request or a load runs, never for `off`; the 30 s drain keeps its window.
 */
const LIMIT_MS = 600_000;

function harness(over: Partial<WeightResidencyOptions> = {}) {
  const state = {
    resident: true,
    busy: false,
    writer: false,
    releases: 0,
    lines: [] as string[],
    events: [] as string[],
  };
  const weights = createWeightResidency({
    idleReleaseMs: LIMIT_MS,
    release: async () => {
      await Promise.resolve();
      state.releases += 1;
      state.resident = false;
      state.events.push('released');
    },
    drain: () => state.events.push(state.writer ? 'drain in writer' : 'drain outside writer'),
    exclusive: async (fn) => {
      state.writer = true;
      try {
        return await fn();
      } finally {
        state.writer = false;
      }
    },
    busy: () => state.busy,
    resident: () => state.resident,
    log: (line) => state.lines.push(line),
    ...over,
  });
  return { state, weights };
}

describe('weight residency idle release', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  // Mutation M11: the release timer is never armed → red (no release).
  it('releases after the whole limit without a request, once', async () => {
    const { state, weights } = harness();
    weights.beginRequest();
    weights.endRequest();
    await vi.advanceTimersByTimeAsync(LIMIT_MS - 1);
    expect(state.releases).toBe(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(state.releases).toBe(1);
    expect(weights.snapshot()).toMatchObject({ released: true, restoring: false });
    expect(state.lines).toEqual(['Weights released after 600 s without a request; the next request restores them']);
    await vi.advanceTimersByTimeAsync(LIMIT_MS * 3);
    expect(state.releases).toBe(1);
  });

  it('starts the clock at creation, as Splash does at Ready', async () => {
    const { state } = harness();
    await vi.advanceTimersByTimeAsync(LIMIT_MS);
    expect(state.releases).toBe(1);
  });

  it('a request before the limit restarts the clock from when it leaves', async () => {
    const { state, weights } = harness();
    await vi.advanceTimersByTimeAsync(LIMIT_MS - 10);
    weights.beginRequest();
    await vi.advanceTimersByTimeAsync(LIMIT_MS * 2);
    expect(state.releases).toBe(0);
    weights.endRequest();
    await vi.advanceTimersByTimeAsync(LIMIT_MS - 1);
    expect(state.releases).toBe(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(state.releases).toBe(1);
  });

  it('a load holds the clock and restarts it when it ends', async () => {
    const { state, weights } = harness();
    let finish!: () => void;
    const load = weights.withLoad(() => new Promise<void>((resolve) => (finish = resolve)));
    await vi.advanceTimersByTimeAsync(LIMIT_MS * 2);
    expect(state.releases).toBe(0);
    finish();
    await load;
    await vi.advanceTimersByTimeAsync(LIMIT_MS - 1);
    expect(state.releases).toBe(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(state.releases).toBe(1);
  });

  it('off never releases on its own; a release request still does', async () => {
    const { state, weights } = harness({ idleReleaseMs: Infinity });
    await vi.advanceTimersByTimeAsync(LIMIT_MS * 100);
    expect(state.releases).toBe(0);
    expect(weights.isPending).toBe(false);
    const snapshot = await weights.release('request');
    expect(state.releases).toBe(1);
    expect(snapshot).toEqual({
      idleReleaseSeconds: null,
      released: true,
      restoring: false,
      restores: 0,
      restoreFailures: 0,
    });
    expect(state.lines).toEqual(['Weights released on request; the next request restores them']);
  });

  it('refuses while a request is in service or other work holds the model', async () => {
    const { state, weights } = harness();
    weights.beginRequest();
    expect(await weights.release('request')).toMatchObject({ released: false, restoring: false });
    weights.endRequest();
    state.busy = true;
    expect(await weights.release('request')).toMatchObject({ released: false });
    expect(state.releases).toBe(0);
  });

  it('an idle release a lease still blocks is tried again on the 500 ms beat', async () => {
    const { state } = harness();
    state.busy = true;
    await vi.advanceTimersByTimeAsync(LIMIT_MS);
    expect(state.releases).toBe(0);
    state.busy = false;
    await vi.advanceTimersByTimeAsync(WEIGHT_RELEASE_RETRY_MS);
    expect(state.releases).toBe(1);
  });

  it('counts a load after a release as a restore, and a memory refusal as a failure', async () => {
    const { state, weights } = harness();
    await weights.release('request');
    await expect(
      weights.withLoad(async () => {
        throw new Error('qwen3_5 memory plan: the floors of the memory plan require 9 bytes but 1 fit');
      }),
    ).rejects.toThrow('memory plan');
    expect(weights.snapshot()).toMatchObject({ released: true, restores: 0, restoreFailures: 1 });
    const load = weights.withLoad(async () => {
      expect(weights.snapshot()).toMatchObject({ released: true, restoring: true });
      await vi.advanceTimersByTimeAsync(1500);
      state.resident = true;
    });
    await load;
    expect(weights.snapshot()).toMatchObject({ released: false, restoring: false, restores: 1, restoreFailures: 1 });
    expect(state.lines.at(-1)).toBe('Weights restored in 1.50 s');
  });

  // Mutation M15: every release marks a restore pending, a held model or not → red.
  it('an idle release before the first load is no restore', async () => {
    const { state, weights } = harness();
    state.resident = false;
    await vi.advanceTimersByTimeAsync(LIMIT_MS);
    expect(state.releases).toBe(1);
    await weights.withLoad(async () => {
      state.resident = true;
    });
    expect(weights.snapshot()).toMatchObject({ released: false, restores: 0 });
    expect(state.lines).toEqual([]);
  });

  // Mutation M16: the release leaves the allocator's free pool undrained → red.
  it('a release drains the allocator once, after the weights go, in the writer slot', async () => {
    for (const sweepMs of [30_000, 0]) {
      const sweeps: number[] = [];
      const sweeper = createIdleSweeper(sweepMs, () => sweeps.push(Date.now()));
      const { state, weights } = harness();
      const chat = trackWeights(sweeper, weights);
      chat.beginRequest();
      chat.endRequest();
      await vi.advanceTimersByTimeAsync(30_000);
      expect(sweeps).toHaveLength(sweepMs > 0 ? 1 : 0);
      expect(state.events).toEqual([]);
      await vi.advanceTimersByTimeAsync(LIMIT_MS - 30_000);
      expect(state.events).toEqual(['released', 'drain in writer']);
      expect(sweeps).toHaveLength(sweepMs > 0 ? 1 : 0);
      sweeper.close();
      weights.close();
    }
  });

  // Mutation M18: arm the whole limit in one timeout, past Node's 2^31 - 1 ms → red.
  it('a limit past the longest timeout still releases only at its deadline', async () => {
    const limitMs = 3_600_000_000;
    const armed = vi.spyOn(globalThis, 'setTimeout');
    const { state, weights } = harness({ idleReleaseMs: limitMs });
    weights.beginRequest();
    weights.endRequest();
    for (const [, delay] of armed.mock.calls) expect(delay).toBeLessThanOrEqual(2 ** 31 - 1);
    armed.mockRestore();
    await vi.advanceTimersByTimeAsync(1);
    await vi.advanceTimersByTimeAsync(86_400_000);
    expect(state.releases).toBe(0);
    await vi.advanceTimersByTimeAsync(limitMs - 86_400_001 - 1);
    expect(state.releases).toBe(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(state.releases).toBe(1);
  });

  it('the chat sweeper keeps the 30 s drain and feeds the weights clock', async () => {
    const drains: number[] = [];
    const sweeper = createIdleSweeper(30_000, () => drains.push(Date.now()));
    const { state, weights } = harness();
    const chat = trackWeights(sweeper, weights);
    chat.beginRequest();
    expect(weights.inFlight).toBe(1);
    chat.endRequest();
    await vi.advanceTimersByTimeAsync(30_000);
    expect(drains).toHaveLength(1);
    expect(state.releases).toBe(0);
    await vi.advanceTimersByTimeAsync(LIMIT_MS - 30_000);
    expect(state.releases).toBe(1);
    sweeper.close();
  });
});
