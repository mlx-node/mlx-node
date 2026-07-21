/**
 * Minimal fetch-on-mount hook over the typed `api.ts` client. Refetches when
 * `path` changes or `reload()` is called; ignores late resolutions after unmount
 * so a navigated-away page never sets state.
 */

import { useCallback, useEffect, useState } from 'react';

import { getJson } from './api';

export interface AsyncState<T> {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
  reload: () => void;
}

export function useJson<T>(path: string): AsyncState<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<Error | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(undefined);
    getJson<T>(path)
      .then((result) => {
        if (!active) return;
        setData(result);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(err instanceof Error ? err : new Error(String(err)));
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [path, nonce]);

  return { data, error, loading, reload };
}
