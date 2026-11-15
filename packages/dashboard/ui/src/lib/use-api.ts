/**
 * Fetch-on-mount hook over the typed `api.ts` client, backed by a
 * stale-while-revalidate cache. Refetches when `path` changes or `reload()` is
 * called; ignores late resolutions after unmount so a navigated-away page never
 * sets state.
 *
 * The cache is what makes navigation stop flickering. A page that has been
 * looked at before renders its previous body on the first frame and corrects it
 * when the runtime answers, so `loading` — and therefore the skeletons every
 * page branches on — is reached only when there is genuinely nothing to show.
 * A refetch that has something to show reports {@link AsyncState.refreshing}
 * instead, which deliberately does NOT collapse the layout.
 */

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';

import { endSharedGets, getJson, getMark } from './api';
import { getConnectionGeneration, subscribeConnection } from './connection';
import { cacheEpoch, readCache, writeCache } from './json-cache';
import { getDataVersion, subscribeLiveStore } from './live';

export interface AsyncState<T> {
  data: T | undefined;
  error: Error | undefined;
  /** Nothing to render yet: show a skeleton. Never true once `data` exists. */
  loading: boolean;
  /** A request is in flight over content that is already on screen. */
  refreshing: boolean;
  /** Refetch. It may share a request only if that one started at or after this call, as a manual Refresh does. */
  reload: () => void;
  /**
   * Apply a change the runtime has already confirmed (a delete answered ok) to what is on screen, then refetch.
   * The refetch's body replaces the patched one, and any request started before the patch is ignored, so a slow,
   * failed or out-of-order listing can never put the removed entry back.
   */
  patch(change: (data: T) => T): void;
}

interface Request<T> {
  /** The path and runtime generation this state describes, detected in render. */
  path: string;
  connection: number;
  /** Bumped by `reload()`; keys the effect so the same path can be refetched. */
  nonce: number;
  /** The fetch shares only a request started after this `getMark()`; 0 on a new path or connection. */
  after: number;
  data: T | undefined;
  error: Error | undefined;
  fetching: boolean;
}

export interface LiveOptions {
  /** Lowest time between two refetch starts of this query. Default LIVE_REFETCH_MIN_MS. */
  minIntervalMs?: number;
}
export const LIVE_REFETCH_MIN_MS = 5000;
export const LIVE_REFETCH_FAST_MS = 1000;

export interface JsonOptions {
  /** Refetch when the runtime reports new data, rate-limited per query. `true` is `{}`. */
  live?: boolean | LiveOptions;
}

let reloadGeneration = 0;
const reloadListeners = new Set<() => void>();

function subscribeReload(listener: () => void): () => void {
  reloadListeners.add(listener);
  return () => {
    reloadListeners.delete(listener);
  };
}

function getReloadGeneration(): number {
  return reloadGeneration;
}

/** Refetch every mounted query once, live or not (after a forced ingest). */
export function reloadMountedQueries(): void {
  reloadGeneration += 1;
  for (const listener of reloadListeners) listener();
}

function subscribeNothing(): () => void {
  return () => {};
}

function begin<T>(path: string, nonce: number, connection: number): Request<T> {
  return { path, connection, nonce, after: 0, data: readCache<T>(path)?.value, error: undefined, fetching: true };
}

export function useJson<T>(path: string, options: JsonOptions = {}): AsyncState<T> {
  // A reconnect hands the app a different runtime, so what is on screen
  // describes something that no longer exists. The shared cache is cleared
  // before this ticks, but mounted state must be reinitialized separately.
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration, getConnectionGeneration);
  const [request, setRequest] = useState<Request<T>>(() => begin<T>(path, 0, connection));

  // Adjusting state during render, rather than in an effect keyed on `path`.
  // An effect commits the old path or runtime's data first and repaints, so a
  // filter change would flash the previous query and a reconnect would present
  // retired runtime data as current. React discards this render and immediately
  // re-runs with the new state, so nothing stale reaches the screen in between.
  if (request.path !== path || request.connection !== connection) {
    setRequest(begin<T>(path, request.nonce, connection));
  }

  const reload = useCallback(() => {
    const after = getMark();
    setRequest((current) => ({ ...current, nonce: current.nonce + 1, after, fetching: true }));
  }, []);

  const patch = useCallback((change: (data: T) => T) => {
    endSharedGets();
    const after = getMark();
    setRequest((current) => ({
      ...current,
      data: current.data === undefined ? undefined : change(current.data),
      nonce: current.nonce + 1,
      after,
      fetching: true,
    }));
  }, []);

  const generation = useSyncExternalStore(subscribeReload, getReloadGeneration, getReloadGeneration);
  const seenGeneration = useRef(generation);
  useEffect(() => {
    if (seenGeneration.current === generation) return;
    seenGeneration.current = generation;
    reload();
  }, [generation, reload]);

  const live = options.live === undefined || options.live === false ? null : options.live === true ? {} : options.live;
  const minIntervalMs = live?.minIntervalMs ?? LIVE_REFETCH_MIN_MS;
  const throttle = useRef({ lastStart: 0, lastDuration: 0, inFlight: false, pending: false, run: 0 });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tryRefetch = useRef<() => void>(() => {});
  tryRefetch.current = () => {
    const state = throttle.current;
    if (!state.pending || state.inFlight) return;
    const wait = state.lastStart + Math.max(minIntervalMs, 3 * state.lastDuration) - Date.now();
    if (wait <= 0) {
      state.pending = false;
      if (timer.current !== null) clearTimeout(timer.current);
      timer.current = null;
      reload();
    } else if (timer.current === null) {
      timer.current = setTimeout(() => {
        timer.current = null;
        tryRefetch.current();
      }, wait);
    }
  };

  const dataVersion = useSyncExternalStore(live === null ? subscribeNothing : subscribeLiveStore, getDataVersion);
  const seenVersion = useRef(dataVersion);
  useEffect(() => {
    if (seenVersion.current === dataVersion) return;
    seenVersion.current = dataVersion;
    if (live === null) return;
    throttle.current.pending = true;
    tryRefetch.current();
  });

  useEffect(
    () => () => {
      throttle.current.pending = false;
      if (timer.current !== null) clearTimeout(timer.current);
      timer.current = null;
    },
    [path, connection],
  );

  const { nonce, after } = request;
  useEffect(() => {
    let active = true;
    const run = ++throttle.current.run;
    const startedAt = Date.now();
    throttle.current.lastStart = startedAt;
    throttle.current.inFlight = true;
    const epoch = cacheEpoch();
    const settle = (): void => {
      if (throttle.current.run !== run) return;
      throttle.current.inFlight = false;
      throttle.current.lastDuration = Date.now() - startedAt;
      if (active) tryRefetch.current();
    };
    getJson<T>(path, { after })
      .then((result) => {
        // `RpcClient.close()` deliberately lets an in-flight call return its
        // authoritative result. After a reconnect that result belongs to the
        // retired runtime: it must update neither this hook nor the shared
        // cache that future mounts seed from.
        if (connection !== getConnectionGeneration()) return;
        // Cached before the `active` check, not after. A body that arrives once
        // the page has moved on is still the correct body for the path that
        // asked for it, and that case is the common one: click into a page,
        // click straight back out, and the response lands on an unmounted
        // component. Discarding it there is what makes returning to a tab feel
        // like a first visit.
        writeCache(path, result, epoch);
        if (!active) return;
        setRequest((current) => ({ ...current, data: result, error: undefined, fetching: false }));
      })
      .catch((err: unknown) => {
        if (!active) return;
        // `data` is deliberately left alone. Pages branch on `error` before
        // anything else, so a failed revalidate still surfaces; keeping the
        // last good body means a transient failure does not also discard what
        // the user was reading.
        setRequest((current) => ({
          ...current,
          error: err instanceof Error ? err : new Error(String(err)),
          fetching: false,
        }));
      })
      .finally(settle);
    return () => {
      active = false;
      if (throttle.current.run === run) throttle.current.inFlight = false;
    };
  }, [path, nonce, after, connection]);

  return {
    data: request.data,
    error: request.error,
    loading: request.data === undefined && request.fetching,
    refreshing: request.data !== undefined && request.fetching,
    reload,
    patch,
  };
}
