/**
 * The one shared watch on the runtime's live channel, behind `useSyncExternalStore`.
 *
 * Open only while a component reads it AND the document is visible, so a hidden
 * window costs the service nothing (it stats the live files only while a watch
 * exists). A reconnect is a different runtime: the snapshot of the old one is
 * dropped and the watch is reopened on the new connection.
 */

import { useSyncExternalStore } from 'react';

import { subscribeLive, type Subscription } from './api';
import { getConnectionGeneration, getTransportUp, subscribeConnection } from './connection';
import { recordLiveSnapshot } from './live-history';
import type { LiveEvent, RuntimeLive } from './types';

let snapshot: RuntimeLive | undefined;
let dataVersion = 0;
const listeners = new Set<() => void>();
let subscription: Subscription | null = null;
let boundGeneration = getConnectionGeneration();
let detach: (() => void) | null = null;
// The service sends `data` only to open watches: one missed while hidden must still refetch on show.
let missedData = false;

function notify(): void {
  for (const listener of listeners) listener();
}

function visible(): boolean {
  return typeof document === 'undefined' || document.visibilityState === 'visible';
}

function onEvent(event: LiveEvent): void {
  if (event.type === 'runtime') {
    recordLiveSnapshot(event.snapshot);
    snapshot = event.snapshot;
  } else {
    dataVersion += 1;
  }
  notify();
}

function sync(): void {
  const generation = getConnectionGeneration();
  const wanted = listeners.size > 0 && visible();
  if (subscription !== null && (generation !== boundGeneration || !wanted)) {
    subscription.close();
    subscription = null;
  }
  // Hidden with readers means no watch is open, whether one closed or none ever opened.
  if (listeners.size > 0 && !visible()) missedData = true;
  const stale = generation !== boundGeneration || !getTransportUp();
  boundGeneration = generation;
  if (stale && snapshot !== undefined) {
    snapshot = undefined;
    notify();
  }
  if (subscription === null && wanted) {
    try {
      subscription = subscribeLive(onEvent);
      if (missedData) {
        missedData = false;
        dataVersion += 1;
        notify();
      }
    } catch {
      // Not connected yet: the connection bump that hands over a port runs sync again.
      subscription = null;
    }
  }
}

/** The store's subscribe, for hooks that read `getDataVersion` / `getRuntimeLive` themselves. */
export function subscribeLiveStore(listener: () => void): () => void {
  listeners.add(listener);
  if (detach === null) {
    // The generation is bumped before the new client exists; subscribe once `connectDashboardApi` returned.
    const stopConnection = subscribeConnection(() => queueMicrotask(sync));
    const onVisibility = (): void => sync();
    if (typeof document !== 'undefined') document.addEventListener('visibilitychange', onVisibility);
    detach = () => {
      stopConnection();
      if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', onVisibility);
      detach = null;
    };
  }
  sync();
  return () => {
    listeners.delete(listener);
    sync();
    if (listeners.size === 0) detach?.();
  };
}

/** The latest pushed snapshot of the current connection, or undefined before the first push. */
export function getRuntimeLive(): RuntimeLive | undefined {
  return snapshot;
}

/** Bumped once per `data` event (an ingest wrote records). */
export function getDataVersion(): number {
  return dataVersion;
}

export function useRuntimeLive(): RuntimeLive | undefined {
  return useSyncExternalStore(subscribeLiveStore, getRuntimeLive, getRuntimeLive);
}

export function useDataVersion(): number {
  return useSyncExternalStore(subscribeLiveStore, getDataVersion, getDataVersion);
}
