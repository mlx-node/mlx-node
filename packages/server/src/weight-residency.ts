/**
 * Idle release of the chat weights (Splash `WeightResidency`,
 * `--idle-release`, `POST /v1/memory/release`).
 *
 * After `idleReleaseMs` without a request the weights are released; the next
 * request loads them again. Its own request count, bumped by
 * `/v1/messages`, `/v1/responses` and `/v1/systemone` only, and its own
 * clock: the 30 s drain sweeper keeps its window, and a disabled sweeper
 * leaves this one running. The clock starts at creation (Splash `idleFrom`
 * at Ready), when the last request leaves and when a load ends.
 *
 * mlx-node releases by unloading the chat model, so the KV pool, the prefix
 * cache and the sessions go with the weights (Splash keeps them).
 */

import type { IdleSweeper } from './idle-sweeper.js';

/** Splash `MemoryControl` beat: a release an unfinished lease refused is tried again this often. */
export const WEIGHT_RELEASE_RETRY_MS = 500;

/** Splash `WeightsSnapshot`; `idleReleaseSeconds` is `null` for `off`. */
export interface WeightsSnapshot {
  idleReleaseSeconds: number | null;
  released: boolean;
  restoring: boolean;
  restores: number;
  restoreFailures: number;
}

export interface WeightResidencyOptions {
  /** Milliseconds without a request before the release; `Infinity` = off. */
  idleReleaseMs: number;
  /** Unloads every resident chat model; runs in the coordinator's writer slot. */
  release: () => Promise<void>;
  /** Runs `fn` in the exclusive writer slot. */
  exclusive: <T>(fn: () => Promise<T>) => Promise<T>;
  /** Work this count does not see: a load, a queued writer, a dispatch lease. */
  busy: () => boolean;
  /** Whether a chat model is resident. */
  resident: () => boolean;
  log?: (line: string) => void;
  now?: () => number;
}

export interface WeightResidency {
  beginRequest(): void;
  endRequest(): void;
  /** Bracket a model load: holds the clock and counts a restore after a release. */
  withLoad<T>(fn: () => Promise<T>): Promise<T>;
  withLoad<T>(fn: () => T): T;
  /**
   * Release now unless a request is in service or a load runs; the snapshot
   * says which (Splash `release_weights`).
   */
  release(reason: 'idle' | 'request'): Promise<WeightsSnapshot>;
  snapshot(): WeightsSnapshot;
  close(): void;
  readonly inFlight: number;
  readonly isPending: boolean;
}

/** A load that failed for want of memory: the S4 plan's refusals, its audit, an allocation. */
export function isMemoryRefusal(error: unknown): boolean {
  const message = (error instanceof Error ? error.message : String(error)).toLowerCase();
  return (
    message.includes('memory plan:') ||
    message.includes('exceeds hard budget') ||
    message.includes('exceed planned bytes') ||
    message.includes('out of memory')
  );
}

function isThenable(value: unknown): value is PromiseLike<unknown> {
  if (value == null) return false;
  const kind = typeof value;
  if (kind !== 'object' && kind !== 'function') return false;
  try {
    return typeof (value as { then?: unknown }).then === 'function';
  } catch {
    return false;
  }
}

export function createWeightResidency(options: WeightResidencyOptions): WeightResidency {
  const delayMs = options.idleReleaseMs;
  if (!(delayMs > 0)) throw new Error(`idleReleaseMs must be positive; received ${String(delayMs)}`);
  const log = options.log ?? ((line: string): void => console.log(line));
  const now = options.now ?? ((): number => Date.now());
  const off = delayMs === Infinity;

  let inFlight = 0;
  let loads = 0;
  let released = false;
  /** A release unloaded a resident model: the next load that brings one back is a restore. */
  let restorePending = false;
  let restores = 0;
  let restoreFailures = 0;
  let closed = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let releasing: Promise<WeightsSnapshot> | null = null;

  const cancel = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const arm = (ms: number): void => {
    cancel();
    if (off || closed || inFlight > 0 || loads > 0) return;
    timer = setTimeout(onTimer, ms);
    timer.unref();
  };

  const isReleased = (): boolean => released && !options.resident();

  const snapshot = (): WeightsSnapshot => {
    const isOut = isReleased();
    return {
      idleReleaseSeconds: off ? null : delayMs / 1000,
      released: isOut,
      restoring: isOut && (inFlight > 0 || loads > 0 || options.busy()),
      restores,
      restoreFailures,
    };
  };

  const release = (reason: 'idle' | 'request'): Promise<WeightsSnapshot> => {
    if (releasing !== null) return releasing;
    if (inFlight > 0 || loads > 0 || options.busy()) return Promise.resolve(snapshot());
    cancel();
    const held = options.resident();
    releasing = (async (): Promise<WeightsSnapshot> => {
      try {
        await options.exclusive(() => options.release());
        released = true;
        if (held) {
          restorePending = true;
          log(
            reason === 'idle'
              ? `Weights released after ${delayMs / 1000} s without a request; the next request restores them`
              : 'Weights released on request; the next request restores them',
          );
        }
        return snapshot();
      } finally {
        releasing = null;
      }
    })();
    return releasing;
  };

  function onTimer(): void {
    timer = null;
    if (inFlight > 0 || loads > 0) return;
    if (releasing === null && options.busy()) {
      arm(WEIGHT_RELEASE_RETRY_MS);
      return;
    }
    release('idle').catch((error: unknown) => {
      console.error('[server] idle release of the weights failed:', error);
    });
  }

  function withLoad<T>(fn: () => Promise<T>): Promise<T>;
  function withLoad<T>(fn: () => T): T;
  function withLoad<T>(fn: () => T | Promise<T>): T | Promise<T> {
    const restoring = restorePending && isReleased();
    const startedAt = now();
    loads += 1;
    cancel();
    const settle = (error: unknown, failed: boolean): void => {
      loads -= 1;
      if (!failed && options.resident()) {
        released = false;
        if (restoring && restorePending) {
          restorePending = false;
          restores += 1;
          log(`Weights restored in ${((now() - startedAt) / 1000).toFixed(2)} s`);
        }
      } else if (restoring) {
        if (failed && isMemoryRefusal(error)) {
          restoreFailures += 1;
          log(
            'Weights not restored: the requests waiting for them failed for want of memory; the next request restores them',
          );
        }
      }
      arm(delayMs);
    };
    let result: T | Promise<T>;
    try {
      result = fn();
    } catch (error) {
      settle(error, true);
      throw error;
    }
    if (isThenable(result)) {
      return Promise.resolve(result).then(
        (value) => {
          settle(undefined, false);
          return value;
        },
        (error: unknown) => {
          settle(error, true);
          throw error;
        },
      );
    }
    settle(undefined, false);
    return result;
  }

  arm(delayMs);

  return {
    beginRequest(): void {
      inFlight += 1;
      cancel();
    },
    endRequest(): void {
      if (inFlight === 0) return;
      inFlight -= 1;
      if (inFlight === 0) arm(delayMs);
    },
    withLoad,
    release,
    snapshot,
    close(): void {
      closed = true;
      cancel();
    },
    get inFlight(): number {
      return inFlight;
    },
    get isPending(): boolean {
      return timer !== null;
    },
  };
}

/**
 * The sweeper the chat endpoints use: every begin, end and load bracket
 * reaches the drain sweeper and the weights' clock alike.
 */
export function trackWeights(sweeper: IdleSweeper, weights: WeightResidency): IdleSweeper {
  function withSuspendedDrains<T>(fn: () => Promise<T>): Promise<T>;
  function withSuspendedDrains<T>(fn: () => T): T;
  function withSuspendedDrains<T>(fn: () => T | Promise<T>): T | Promise<T> {
    return sweeper.withSuspendedDrains(() => weights.withLoad(fn as () => T));
  }
  return {
    beginRequest(): void {
      sweeper.beginRequest();
      weights.beginRequest();
    },
    endRequest(): void {
      sweeper.endRequest();
      weights.endRequest();
    },
    close(): void {
      sweeper.close();
    },
    withSuspendedDrains,
    suspendDrains(): () => void {
      return sweeper.suspendDrains();
    },
    get isPending(): boolean {
      return sweeper.isPending;
    },
    get inFlight(): number {
      return sweeper.inFlight;
    },
  };
}
