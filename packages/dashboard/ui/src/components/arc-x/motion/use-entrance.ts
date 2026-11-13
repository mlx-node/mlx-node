import {
  createContext,
  createElement,
  useContext,
  useEffect,
  useState,
  type ReactElement,
  type ReactNode,
  type RefObject,
} from 'react';

import { useStaticMotion } from './use-static-motion';

export type EntranceState = 'pending' | 'run' | 'done';

const GateContext = createContext(true);

/** Closes rich motion for a subtree: rows past a cap, items past a list limit. */
export function MotionGate(props: { open: boolean; children: ReactNode }): ReactElement {
  const parentOpen = useContext(GateContext);
  return createElement(GateContext.Provider, { value: parentOpen && props.open }, props.children);
}

/** False under useStaticMotion() or inside a closed MotionGate. */
export function useRichMotion(): boolean {
  const staticMotion = useStaticMotion();
  const open = useContext(GateContext);
  return !staticMotion && open;
}

function canObserve(): boolean {
  return typeof globalThis.IntersectionObserver === 'function';
}

/**
 * 'pending' until the element is in the viewport with a width, 'run' until the mark tagged data-entrance-last
 * fires animationend, then 'done'. Once per mount.
 */
export function useEntrance(ref: RefObject<Element | null>): EntranceState {
  const rich = useRichMotion();
  const [state, setState] = useState<EntranceState>(() => (rich && canObserve() ? 'pending' : 'done'));

  useEffect(() => {
    if (state !== 'pending') return;
    const element = ref.current;
    if (!rich || !canObserve() || element === null) {
      setState('done');
      return;
    }
    // No `root`: the viewport clips through every scrolling ancestor and also sees Drawer / Dialog portals.
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && entry.boundingClientRect.width > 0) {
            observer.disconnect();
            setState('run');
            return;
          }
        }
      },
      { threshold: 0.3 },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [state, rich, ref]);

  useEffect(() => {
    if (state !== 'run') return;
    const element = ref.current;
    if (element === null || element.querySelector('[data-entrance-last]') === null) {
      setState('done');
      return;
    }
    const onEnd = (event: Event): void => {
      const target = event.target;
      if (target instanceof Element && target.hasAttribute('data-entrance-last')) setState('done');
    };
    element.addEventListener('animationend', onEnd);
    return () => element.removeEventListener('animationend', onEnd);
  }, [state, ref]);

  return state;
}
