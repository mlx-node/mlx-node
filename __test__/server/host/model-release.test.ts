import { describe, expect, it, vi } from 'vite-plus/test';

import { createModelReleaser, resolveCollector } from '../../../packages/server/src/host/model-release.js';

function manualSchedule(): { schedule: (run: () => void) => void; runAll: () => void; queued: () => number } {
  const queue: Array<() => void> = [];
  return {
    schedule: (run) => {
      queue.push(run);
    },
    runAll: () => {
      for (const run of queue.splice(0)) run();
    },
    queued: () => queue.length,
  };
}

describe('createModelReleaser', () => {
  it('collects once per drop, after the dropping call returns; drops before the collection share it', () => {
    const collect = vi.fn();
    const timer = manualSchedule();
    const releaser = createModelReleaser({ resolve: () => collect, schedule: timer.schedule });

    releaser.released();
    releaser.released();
    expect(collect).not.toHaveBeenCalled();
    expect(timer.queued()).toBe(1);
    timer.runAll();
    expect(collect).toHaveBeenCalledTimes(1);

    releaser.released();
    timer.runAll();
    expect(collect).toHaveBeenCalledTimes(2);
  });

  it('with no collector: one log line, no throw, and later drops do nothing', () => {
    const warn = vi.fn();
    const resolve = vi.fn(() => null);
    const timer = manualSchedule();
    const releaser = createModelReleaser({ resolve, schedule: timer.schedule, warn });

    releaser.released();
    expect(() => timer.runAll()).not.toThrow();
    releaser.released();
    expect(() => timer.runAll()).not.toThrow();
    expect(resolve).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]![0]).toContain('no V8 collector');
  });

  it('a collector or a resolver that throws does not throw out of the release', () => {
    const timer = manualSchedule();
    const warn = vi.fn();
    const failing = createModelReleaser({
      resolve: () => () => {
        throw new Error('collector failed');
      },
      schedule: timer.schedule,
      warn,
    });
    failing.released();
    expect(() => timer.runAll()).not.toThrow();

    const broken = createModelReleaser({
      resolve: () => {
        throw new Error('no flags');
      },
      schedule: timer.schedule,
      warn,
    });
    broken.released();
    expect(() => timer.runAll()).not.toThrow();
    expect(warn).toHaveBeenCalledTimes(1);
  });
});

describe('resolveCollector', () => {
  it('finds a working full collector in a process started without --expose-gc', async () => {
    const collect = resolveCollector();
    expect(typeof collect).toBe('function');
    let finalized = false;
    const registry = new FinalizationRegistry(() => {
      finalized = true;
    });
    (() => {
      registry.register({ payload: new Array(1000).fill(0) }, 'dropped');
    })();
    await new Promise((resolve) => setImmediate(resolve));
    collect!();
    await vi.waitFor(() => expect(finalized).toBe(true));
  });
});
