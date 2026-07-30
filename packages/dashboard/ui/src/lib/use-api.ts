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

import { useCallback, useEffect, useState, useSyncExternalStore } from 'react';

import { getJson } from './api';
import { getConnectionGeneration, subscribeConnection } from './connection';
import { readCache, writeCache } from './json-cache';

export interface AsyncState<T> {
  data: T | undefined;
  error: Error | undefined;
  /** Nothing to render yet: show a skeleton. Never true once `data` exists. */
  loading: boolean;
  /** A request is in flight over content that is already on screen. */
  refreshing: boolean;
  reload: () => void;
}

interface Request<T> {
  /** The path and runtime generation this state describes, detected in render. */
  path: string;
  connection: number;
  /** Bumped by `reload()`; keys the effect so the same path can be refetched. */
  nonce: number;
  data: T | undefined;
  error: Error | undefined;
  fetching: boolean;
}

function begin<T>(path: string, nonce: number, connection: number): Request<T> {
  return { path, connection, nonce, data: readCache<T>(path)?.value, error: undefined, fetching: true };
}

export function useJson<T>(path: string): AsyncState<T> {
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
    setRequest((current) => ({ ...current, nonce: current.nonce + 1, fetching: true }));
  }, []);

  const { nonce } = request;
  useEffect(() => {
    let active = true;
    getJson<T>(path)
      .then((result) => {
        // `RpcClient.close()` deliberately lets an in-flight call return its
        // authoritative result. After a reconnect that result belongs to the
        // retired runtime: it must update neither this hook nor the shared
        // cache that future mounts seed from.
        if (connection !== getConnectionGeneration()) return;
        // Cached before the `active` check, not after. A body that arrives once
        // the page has moved on is still the correct body for the path that
        // asked for it, and that case is the common one: click into a session,
        // click straight back out, and the response lands on an unmounted
        // component. Discarding it there is what makes returning to a tab feel
        // like a first visit.
        writeCache(path, result);
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
      });
    return () => {
      active = false;
    };
  }, [path, nonce, connection]);

  return {
    data: request.data,
    error: request.error,
    loading: request.data === undefined && request.fetching,
    refreshing: request.data !== undefined && request.fetching,
    reload,
  };
}
