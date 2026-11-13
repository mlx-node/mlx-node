import { useLayoutEffect, useRef, useSyncExternalStore } from 'react';

import { getConnectionGeneration, subscribeConnection } from './connection';
import type { AsyncState } from './use-api';

export interface Held<T> {
  data: T | undefined;
  /** True while `data` is the answer to the PREVIOUS question. */
  held: boolean;
}

/**
 * `state.data` when it is defined. While it is undefined because the question changed (state.loading),
 * the last data this hook returned, with held = true. Forgets on an error without data and on a reconnect.
 */
export function useHeld<T>(state: AsyncState<T>): Held<T> {
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration, getConnectionGeneration);
  // Written after commit only: a render React discards (useJson resets its state during render) can still
  // carry the previous question's or the previous runtime's data.
  const last = useRef<{ data: T | undefined; connection: number }>({ data: undefined, connection });
  const { data, error } = state;
  useLayoutEffect(() => {
    if (data !== undefined) last.current = { data, connection };
    else if (error !== undefined || last.current.connection !== connection)
      last.current = { data: undefined, connection };
  }, [data, error, connection]);

  if (data !== undefined) return { data, held: false };
  if (error !== undefined || !state.loading) return { data: undefined, held: false };
  const previous = last.current;
  if (previous.connection !== connection || previous.data === undefined) return { data: undefined, held: false };
  return { data: previous.data, held: true };
}
