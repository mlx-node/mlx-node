/**
 * The user's home folder, read from `/health` once per connection and shared by
 * every path on screen. Null while unknown or after a failed read, so a path
 * then shows in full.
 */

import { useEffect, useSyncExternalStore } from 'react';

import { getJson } from './api';
import { getConnectionGeneration, subscribeConnection } from './connection';
import { readCache, writeCache } from './json-cache';
import type { HealthResponse } from './types';

interface HomeState {
  connection: number;
  home: string | null;
}

let state: HomeState = { connection: -1, home: null };
const listeners = new Set<() => void>();

function set(next: HomeState): void {
  state = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function homeOf(body: HealthResponse | undefined): string | null {
  const home = body?.homeDir;
  return typeof home === 'string' && home !== '' ? home : null;
}

function load(connection: number): void {
  if (state.connection === connection) return;
  const cached = readCache<HealthResponse>('/health');
  if (cached !== undefined && homeOf(cached.value) !== null) {
    set({ connection, home: homeOf(cached.value) });
    return;
  }
  set({ connection, home: null });
  getJson<HealthResponse>('/health').then(
    (body) => {
      if (connection !== getConnectionGeneration()) return;
      writeCache('/health', body);
      set({ connection, home: homeOf(body) });
    },
    () => {},
  );
}

/** The home folder of the runtime this window is connected to, or null while it is unknown. */
export function useHomeDir(): string | null {
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration, getConnectionGeneration);
  const snapshot = (): string | null => (state.connection === connection ? state.home : null);
  const home = useSyncExternalStore(subscribe, snapshot, snapshot);
  useEffect(() => {
    load(connection);
  }, [connection]);
  return home;
}
