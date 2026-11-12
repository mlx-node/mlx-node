import type { ChildProcess } from 'node:child_process';
import { EventEmitter } from 'node:events';

import { FLUSH_MS, SERVER_CLOSE_MS, STOP_MS } from '@mlx-node/server/supervision';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { stopServe } from '../../packages/cli/src/commands/agent/grok-build.js';

/**
 * Splash `supervision`: serve closes its server, then flushes its persistent
 * SSD caches, within what `mlx agent` waits before it kills serve.
 */
describe('stop budgets', () => {
  it('fit Splash: server close and flush end before the stop deadline', () => {
    expect(FLUSH_MS).toBe(6000);
    expect(STOP_MS).toBe(15000);
    expect(SERVER_CLOSE_MS + FLUSH_MS).toBeLessThan(STOP_MS);
  });
});

describe('mlx agent stopping serve', () => {
  afterEach(() => vi.useRealTimers());

  function fakeServe(): { child: ChildProcess; kill: ReturnType<typeof vi.fn> } {
    const emitter = new EventEmitter() as EventEmitter & { exitCode: number | null; signalCode: string | null };
    emitter.exitCode = null;
    emitter.signalCode = null;
    const kill = vi.fn((signal: NodeJS.Signals) => {
      if (signal === 'SIGKILL') {
        emitter.signalCode = 'SIGKILL';
        queueMicrotask(() => emitter.emit('exit', null, 'SIGKILL'));
      }
      return true;
    });
    return { child: Object.assign(emitter, { kill }) as unknown as ChildProcess, kill };
  }

  // Mutation SB1: the default grace back at 10 s → red (SIGKILL at 10 s).
  it('kills a serve that is still flushing only after STOP_MS', async () => {
    vi.useFakeTimers();
    const { child, kill } = fakeServe();
    const stopped = stopServe(child, 'SIGTERM');
    expect(kill.mock.calls).toEqual([['SIGTERM']]);
    await vi.advanceTimersByTimeAsync(STOP_MS - 1);
    expect(kill.mock.calls).toEqual([['SIGTERM']]);
    await vi.advanceTimersByTimeAsync(1);
    await stopped;
    expect(kill.mock.calls).toEqual([['SIGTERM'], ['SIGKILL']]);
  });
});
