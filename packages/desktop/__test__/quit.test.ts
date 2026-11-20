import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { createQuitHandler } from '../src/main/quit.js';

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('desktop quit', () => {
  it.each(['normal', 'failed', 'deadline'])(
    'preserves normal exit and activation relaunch after %s cleanup',
    async (mode) => {
      const drain = Promise.withResolvers<void>();
      const options = {
        beginShutdown: vi.fn(),
        shutdown: vi.fn(() => drain.promise),
        deadlineMs: 14_000,
        installUpdate: vi.fn(() => false),
        shouldRelaunch: () => true,
        relaunch: vi.fn(),
        quit: vi.fn(),
        report: vi.fn(),
      };
      const quit = createQuitHandler(options);
      const preventDefault = vi.fn();
      quit({ preventDefault });
      quit({ preventDefault });
      expect(options.beginShutdown).toHaveBeenCalledTimes(1);
      expect(options.shutdown).toHaveBeenCalledTimes(1);
      expect(options.quit).not.toHaveBeenCalled();
      if (mode === 'normal') drain.resolve();
      if (mode === 'failed') drain.reject(new Error('child exit failed'));
      await vi.advanceTimersByTimeAsync(mode === 'deadline' ? 14_000 : 0);
      expect(options.report).toHaveBeenCalledTimes(mode === 'normal' ? 0 : 1);
      expect(options.relaunch).toHaveBeenCalledTimes(1);
      expect(options.quit).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(0);
      quit({ preventDefault });
      expect(preventDefault).toHaveBeenCalledTimes(2);
      drain.resolve();
      await vi.advanceTimersByTimeAsync(0);
      expect(options.quit).toHaveBeenCalledTimes(1);
    },
  );

  it.each([false, true])('a never-settling request forces the exit (relaunch requested: %s)', async (relaunch) => {
    let blocked = true;
    const options = {
      beginShutdown: vi.fn(),
      shutdown: vi.fn(() => Promise.resolve()),
      deadlineMs: 14_000,
      installUpdate: vi.fn(() => false),
      shouldRelaunch: () => relaunch,
      relaunch: vi.fn(),
      quit: vi.fn(),
      exitBlocked: () => blocked,
      forceExit: vi.fn(),
      report: vi.fn(),
    };
    createQuitHandler(options)({ preventDefault: vi.fn() });
    await vi.advanceTimersByTimeAsync(0);
    expect(options.forceExit).toHaveBeenCalledTimes(1);
    expect(options.quit).not.toHaveBeenCalled();
    expect(options.relaunch).not.toHaveBeenCalled();
    expect(options.report).toHaveBeenCalledTimes(1);
    expect(String(options.report.mock.calls[0]?.[0])).toContain(
      relaunch ? 'without the requested relaunch' : 'never settled',
    );

    blocked = false;
    const settled = { ...options, forceExit: vi.fn(), quit: vi.fn(), report: vi.fn() };
    createQuitHandler(settled)({ preventDefault: vi.fn() });
    await vi.advanceTimersByTimeAsync(0);
    expect(settled.forceExit).not.toHaveBeenCalled();
    expect(settled.quit).toHaveBeenCalledTimes(1);
    expect(settled.report).not.toHaveBeenCalled();
  });

  it('an update install keeps its own quit even when a request is stuck', async () => {
    const install = vi.fn();
    const options = {
      beginShutdown: vi.fn(),
      shutdown: vi.fn(() => Promise.resolve()),
      deadlineMs: 14_000,
      installUpdate: vi.fn((complete: (install?: (relaunch: boolean) => void) => void) => {
        complete(install);
        return true;
      }),
      shouldRelaunch: () => false,
      relaunch: vi.fn(),
      quit: vi.fn(),
      exitBlocked: () => true,
      forceExit: vi.fn(),
      report: vi.fn(),
    };
    createQuitHandler(options)({ preventDefault: vi.fn() });
    await vi.advanceTimersByTimeAsync(0);
    expect(install).toHaveBeenCalledTimes(1);
    expect(options.forceExit).not.toHaveBeenCalled();
  });
});
