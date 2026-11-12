/**
 * Bounded release of a chat model the registry dropped.
 *
 * Chat models have no dispose: their weights live on a native model thread
 * that exits only when V8 collects the model's JS object. V8 is not told
 * about those bytes, so after a swap or an unload it can keep a dropped
 * model (tens of GB) for minutes. One forced full collection after the drop
 * frees it; the idle sweeper then clears the MLX cache on its usual delay.
 */

import { setFlagsFromString } from 'node:v8';
import { runInNewContext } from 'node:vm';

export type Collector = () => void;

/**
 * The V8 full collector: `globalThis.gc` when the process was started with
 * `--expose-gc`, else the same function from a new context after the flag is
 * set at run time (plain Node and an Electron utilityProcess both allow it).
 */
export function resolveCollector(): Collector | null {
  const exposed: unknown = (globalThis as { gc?: unknown }).gc;
  if (typeof exposed === 'function') return exposed as Collector;
  try {
    setFlagsFromString('--expose-gc');
    const gc: unknown = runInNewContext('gc');
    return typeof gc === 'function' ? (gc as Collector) : null;
  } catch {
    return null;
  }
}

export interface ModelReleaser {
  /** A model binding was dropped and nothing holds it. Never throws. */
  released(): void;
}

export interface ModelReleaserOptions {
  /** Finds the collector on the first release. Default {@link resolveCollector}. */
  resolve?: () => Collector | null;
  /** Default `setImmediate`, so the dropping call stack has unwound. */
  schedule?: (run: () => void) => void;
  warn?: (line: string) => void;
}

/** Drops that arrive before the scheduled collection share it. */
export function createModelReleaser(opts: ModelReleaserOptions = {}): ModelReleaser {
  const resolve = opts.resolve ?? resolveCollector;
  const schedule =
    opts.schedule ??
    ((run: () => void): void => {
      setImmediate(run);
    });
  const warn =
    opts.warn ??
    ((line: string): void => {
      console.warn(line);
    });
  let collector: Collector | null | undefined;
  let pending = false;

  const run = (): void => {
    pending = false;
    if (collector === undefined) {
      try {
        collector = resolve();
      } catch {
        collector = null;
      }
      if (collector === null) {
        warn('[mlx] no V8 collector is available: a dropped model stays in memory until V8 collects it.');
      }
    }
    if (collector === null) return;
    try {
      collector();
    } catch {
      /* the next drop tries again */
    }
  };

  return {
    released(): void {
      if (pending) return;
      pending = true;
      schedule(run);
    },
  };
}
