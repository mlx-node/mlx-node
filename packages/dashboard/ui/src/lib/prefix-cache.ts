import { mutate } from './api';
import type { PrefixCacheDeleteResult, PrefixCacheRoot, PrefixLockState } from './types';

export interface PrefixTotals {
  bytes: number;
  files: number;
  caches: number;
  held: number;
  idle: number;
  unknown: number;
  /** Every row's walk reached its end. */
  complete: boolean;
  /** The smallest limit an engine of this root reports: the first one any engine enforces. Null when none does. */
  limitBytes: number | null;
  /** Engines that report a limit. */
  limitEngines: number;
}

export function prefixTotals(root: Pick<PrefixCacheRoot, 'namespaces' | 'engines'>): PrefixTotals {
  const totals: PrefixTotals = {
    bytes: 0,
    files: 0,
    caches: root.namespaces.length,
    held: 0,
    idle: 0,
    unknown: 0,
    complete: true,
    limitBytes: null,
    limitEngines: 0,
  };
  for (const ns of root.namespaces) {
    totals.bytes += ns.bytes;
    totals.files += ns.files;
    if (ns.lock === 'held') totals.held += 1;
    else if (ns.lock === 'free') totals.idle += 1;
    else totals.unknown += 1;
    if (!ns.complete) totals.complete = false;
  }
  for (const engine of root.engines) {
    if (engine.limitBytes === null) continue;
    totals.limitEngines += 1;
    totals.limitBytes = totals.limitBytes === null ? engine.limitBytes : Math.min(totals.limitBytes, engine.limitBytes);
  }
  return totals;
}

export function lockWord(state: PrefixLockState): string {
  return state === 'held' ? 'In use' : state === 'free' ? 'Idle' : 'Unknown';
}

/**
 * The free time between two delete steps. An engine that opens the same cache tries its lock every 200 ms
 * (`open_persistent_dir`) for 20 s; a longer gap lets one of its tries land, so it does not lose its wait to a long
 * delete. The next step then answers `held`.
 */
export const STEP_GAP_MS = 300;

function pause(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** The delete in steps: each call removes up to a few thousand files under the lock, until the directory is gone. */
export async function deleteNamespace(
  root: string,
  namespace: string,
  openedMs: number,
  wait: (ms: number) => Promise<void> = pause,
): Promise<PrefixCacheDeleteResult> {
  const sum: PrefixCacheDeleteResult = { removed: 0, freedBytes: 0, done: false, left: 0, directoryRemoved: false };
  for (let first = true; ; first = false) {
    if (!first) await wait(STEP_GAP_MS);
    const step = await mutate<PrefixCacheDeleteResult>('DELETE', `/cache/prefix/${namespace}`, { root, openedMs });
    sum.removed += step.removed;
    sum.freedBytes += step.freedBytes;
    if (step.done) {
      return { ...sum, done: true, left: step.left, directoryRemoved: step.directoryRemoved };
    }
    if (step.removed === 0) throw new Error('The cache did not shrink.');
  }
}
