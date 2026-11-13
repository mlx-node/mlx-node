/**
 * A ring buffer of pushed runtime snapshots, one snapshot per second at most, kept for the whole app session
 * (module state: route changes keep it). Fed by `live.ts` for every `runtime` event; a reconnect empties it.
 * One sample per running process, never a sum: every reader filters by pid and shows it beside that process's value.
 */

import { useSyncExternalStore } from 'react';

import { getConnectionGeneration, subscribeConnection } from './connection';
import type { RuntimeLive } from './types';

export interface LiveSample {
  /** snapshot.now */
  t: number;
  /** The process these values belong to; null = no process was running. */
  pid: number | null;
  /** 0 = running and idle. */
  outTps: number | null;
  memActive: number | null;
  memCache: number | null;
  inFlight: number | null;
  queued: number | null;
}

export const LIVE_HISTORY_MAX = 600;
/** Consumers draw a gap where two samples are further apart than this. */
export const LIVE_GAP_MS = 20_000;
/** A snapshot this soon after the last sample is dropped. */
const MIN_SPACING_MS = 900;

let samples: readonly LiveSample[] = [];
let generation = getConnectionGeneration();
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) listener();
}

function syncGeneration(): boolean {
  const current = getConnectionGeneration();
  if (current === generation) return false;
  generation = current;
  samples = [];
  return true;
}

subscribeConnection(() => {
  if (syncGeneration()) notify();
});

/** One sample per running process (its own values), or one all-null sample when none runs. */
function samplesOf(snapshot: RuntimeLive): LiveSample[] {
  const out: LiveSample[] = [];
  for (const process of snapshot.processes) {
    if (process.status !== 'running') continue;
    let inFlight = 0;
    let queued = 0;
    for (const model of process.models) {
      inFlight += model.inFlight;
      queued += model.queued;
    }
    out.push({
      t: snapshot.now,
      pid: process.pid,
      outTps: process.decodeTps ?? 0,
      memActive: process.mem.active,
      memCache: process.mem.cache,
      inFlight,
      queued,
    });
  }
  if (out.length > 0) return out;
  return [{ t: snapshot.now, pid: null, outTps: null, memActive: null, memCache: null, inFlight: null, queued: null }];
}

/** Keeps the newest LIVE_HISTORY_MAX snapshot times; the samples of one snapshot share its `t`. */
function capped(next: readonly LiveSample[]): readonly LiveSample[] {
  let times = 0;
  let previous: number | null = null;
  for (let i = next.length - 1; i >= 0; i -= 1) {
    const t = next[i].t;
    if (t === previous) continue;
    previous = t;
    times += 1;
    if (times > LIVE_HISTORY_MAX) return next.slice(i + 1);
  }
  return next;
}

export function recordLiveSnapshot(snapshot: RuntimeLive): void {
  syncGeneration();
  const last = samples[samples.length - 1];
  if (last !== undefined && snapshot.now - last.t < MIN_SPACING_MS) return;
  samples = capped([...samples, ...samplesOf(snapshot)]);
  notify();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSamples(): readonly LiveSample[] {
  return samples;
}

/** The buffer, oldest first; a new array identity per append. */
export function useLiveHistory(): readonly LiveSample[] {
  return useSyncExternalStore(subscribe, getSamples, getSamples);
}

export function resetLiveHistory(): void {
  samples = [];
  notify();
}
