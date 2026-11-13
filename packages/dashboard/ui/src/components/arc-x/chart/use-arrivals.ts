import { useCallback, useState } from 'react';

interface ArrivalState {
  signature: string;
  known: ReadonlySet<string>;
  arriving: ReadonlySet<string>;
}

function signatureOf(keys: readonly string[]): string {
  return [...keys].sort().join('\u0000');
}

/**
 * Keys that appeared after the first render (a series, a point, a row that comes with new data). They carry an
 * arrival animation until `settle(key)` (its animationend). The first render and the static branch arrive nothing.
 */
export function useArrivals(
  keys: readonly string[],
  enabled: boolean,
): { arriving: ReadonlySet<string>; settle: (key: string) => void } {
  const [state, setState] = useState<ArrivalState>(() => ({
    signature: signatureOf(keys),
    known: new Set(keys),
    arriving: new Set(),
  }));
  const signature = signatureOf(keys);
  let current = state;
  if (state.signature !== signature) {
    const arriving = new Set<string>();
    if (enabled) {
      for (const key of state.arriving) if (keys.includes(key)) arriving.add(key);
      for (const key of keys) if (!state.known.has(key)) arriving.add(key);
    }
    current = { signature, known: new Set(keys), arriving };
    setState(current);
  }
  const settle = useCallback((key: string) => {
    setState((previous) => {
      if (!previous.arriving.has(key)) return previous;
      const arriving = new Set(previous.arriving);
      arriving.delete(key);
      return { ...previous, arriving };
    });
  }, []);
  return { arriving: current.arriving, settle };
}
