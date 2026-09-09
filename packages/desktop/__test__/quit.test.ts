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
});
