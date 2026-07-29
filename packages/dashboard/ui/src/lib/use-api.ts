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
  /** The path this state describes, so a path change can be detected in render. */
  path: string;
  /** Bumped by `reload()`; keys the effect so the same path can be refetched. */
  nonce: number;
  data: T | undefined;
  error: Error | undefined;
  fetching: boolean;
}

function begin<T>(path: string, nonce: number): Request<T> {
  return { path, nonce, data: readCache<T>(path)?.value, error: undefined, fetching: true };
}

export function useJson<T>(path: string): AsyncState<T> {
  const [request, setRequest] = useState<Request<T>>(() => begin<T>(path, 0));

  // Adjusting state during render, rather than in an effect keyed on `path`.
  // An effect commits the old path's data first and repaints, so switching a
  // filter would flash the previous query's rows for a frame before the
  // skeleton — the exact stutter this hook exists to remove. React discards
  // this render and immediately re-runs with the new state, so nothing reaches
  // the screen in between.
  if (request.path !== path) setRequest(begin<T>(path, request.nonce));

  const reload = useCallback(() => {
    setRequest((current) => ({ ...current, nonce: current.nonce + 1, fetching: true }));
  }, []);

  // A reconnect hands the app a different runtime, so what is on screen
  // describes something that no longer exists. Without this in the deps a
  // mounted hook never refetches — and one that errored while CONTROL PANEL was down
  // stays stuck on that error even after the replacement port arrives.
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration, getConnectionGeneration);

  const { nonce } = request;
  useEffect(() => {
    let active = true;
    getJson<T>(path)
      .then((result) => {
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
