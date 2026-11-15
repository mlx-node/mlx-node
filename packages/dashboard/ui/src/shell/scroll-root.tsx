import { createContext, useCallback, useContext, useRef } from 'react';

/** The `<main data-scroll-root>` element: the only page scroller. Null outside the shell (tests, `/welcome`). */
export const ScrollRootContext = createContext<HTMLElement | null>(null);

export function useScrollRoot(): HTMLElement | null {
  return useContext(ScrollRootContext);
}

export type NavigationKind = 'push' | 'pop';

/**
 * Scroll positions per history entry. `remember(key)` records where the scroll root stands for the entry being left;
 * `settle(kind, key)` puts a new page at the top (push) or where its entry was left (pop).
 */
export function useScrollMemory(root: HTMLElement | null): {
  remember: (key: string) => void;
  settle: (kind: NavigationKind, key: string) => void;
} {
  const saved = useRef(new Map<string, number>());
  const remember = useCallback(
    (key: string) => {
      if (root !== null) saved.current.set(key, root.scrollTop);
    },
    [root],
  );
  const settle = useCallback(
    (kind: NavigationKind, key: string) => {
      if (root === null) return;
      root.scrollTop = kind === 'pop' ? (saved.current.get(key) ?? 0) : 0;
    },
    [root],
  );
  return { remember, settle };
}
