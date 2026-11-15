/**
 * Follow the end of a growing thread. While pinned, every size change of the content puts the scroll root at its
 * end (a direct write, never smooth). Only the user releases it: a wheel up, PageUp / ArrowUp / Home outside a text
 * field, or a scroll that ends more than `PIN_SLACK` px from the end. Scrolling back near the end pins again.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

export const PIN_SLACK = 48;
/** The jump button shows without new content once the end is this far away. */
export const JUMP_FAR = 400;

export interface ScrollMetrics {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
}

export type ScrollCause = 'self' | 'scroll' | 'wheel-up' | 'key-up';

export function distanceToEnd(m: ScrollMetrics): number {
  return Math.max(0, m.scrollHeight - m.clientHeight - m.scrollTop);
}

/** Pinned after this event? */
export function nextPinned(pinned: boolean, m: ScrollMetrics, cause: ScrollCause): boolean {
  if (cause === 'wheel-up' || cause === 'key-up') return false;
  if (cause === 'self') return pinned;
  return distanceToEnd(m) <= PIN_SLACK;
}

export function jumpVisible(pinned: boolean, fresh: boolean, m: ScrollMetrics): boolean {
  return !pinned && (fresh || distanceToEnd(m) > JUMP_FAR);
}

const RELEASE_KEYS = new Set(['PageUp', 'ArrowUp', 'Home']);

function editable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || target.tagName === 'TEXTAREA' || target.tagName === 'INPUT';
}

/** Scroll place per conversation, for this window only. */
const places = new Map<string, { top: number; pinned: boolean }>();

export interface StickToBottom {
  /** The node whose size changes are followed (the message column). */
  contentRef: (node: HTMLElement | null) => void;
  pinned: boolean;
  /** New content arrived while released. */
  fresh: boolean;
  showJump: boolean;
  /** Scroll to the end (smooth unless `instant`) and pin. */
  jump: (instant?: boolean) => void;
  /** Pin and go to the end at once (a send). */
  pin: () => void;
}

export function useStickToBottom(root: HTMLElement | null, key: string): StickToBottom {
  const [pinned, setPinned] = useState(true);
  const [fresh, setFresh] = useState(false);
  const [far, setFar] = useState(false);
  const pinnedRef = useRef(true);
  const jumping = useRef(false);
  const ownTop = useRef<number | null>(null);
  const [content, setContent] = useState<HTMLElement | null>(null);

  const metrics = useCallback(
    (): ScrollMetrics =>
      root === null
        ? { scrollTop: 0, scrollHeight: 0, clientHeight: 0 }
        : { scrollTop: root.scrollTop, scrollHeight: root.scrollHeight, clientHeight: root.clientHeight },
    [root],
  );

  const apply = useCallback(
    (next: boolean): void => {
      pinnedRef.current = next;
      setPinned(next);
      if (next) setFresh(false);
      setFar(distanceToEnd(metrics()) > JUMP_FAR);
    },
    [metrics],
  );

  const toEnd = useCallback((): void => {
    if (root === null) return;
    const top = Math.max(0, root.scrollHeight - root.clientHeight);
    ownTop.current = top;
    root.scrollTop = top;
  }, [root]);

  useLayoutEffect(() => {
    const place = places.get(key);
    if (place !== undefined && !place.pinned && root !== null) {
      ownTop.current = place.top;
      root.scrollTop = place.top;
      apply(false);
    } else {
      apply(true);
      toEnd();
    }
    return () => {
      if (root !== null) places.set(key, { top: root.scrollTop, pinned: pinnedRef.current });
    };
  }, [key, root, apply, toEnd]);

  useEffect(() => {
    if (root === null) return;
    const release = (): void => {
      jumping.current = false;
      if (pinnedRef.current) apply(false);
    };
    const onWheel = (event: WheelEvent): void => {
      if (event.deltaY < 0) release();
    };
    const onKey = (event: KeyboardEvent): void => {
      if (RELEASE_KEYS.has(event.key) && !editable(event.target)) release();
    };
    const onScroll = (): void => {
      const m = metrics();
      if (ownTop.current !== null && Math.abs(m.scrollTop - ownTop.current) < 1) {
        ownTop.current = null;
        return;
      }
      ownTop.current = null;
      if (jumping.current) {
        if (distanceToEnd(m) <= PIN_SLACK) jumping.current = false;
        return;
      }
      const next = nextPinned(pinnedRef.current, m, 'scroll');
      if (next !== pinnedRef.current) apply(next);
      else setFar(distanceToEnd(m) > JUMP_FAR);
    };
    root.addEventListener('wheel', onWheel, { passive: true });
    root.addEventListener('keydown', onKey);
    root.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      root.removeEventListener('wheel', onWheel);
      root.removeEventListener('keydown', onKey);
      root.removeEventListener('scroll', onScroll);
    };
  }, [root, metrics, apply]);

  useEffect(() => {
    if (content === null || typeof ResizeObserver !== 'function') return;
    const first = content.getBoundingClientRect();
    let last = first.height;
    let lastWidth = first.width;
    const observer = new ResizeObserver(() => {
      const box = content.getBoundingClientRect();
      // A column that got narrower is taller without any new message.
      const grew = box.height > last && box.width === lastWidth;
      last = box.height;
      lastWidth = box.width;
      if (pinnedRef.current) toEnd();
      else if (grew) setFresh(true);
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, [content, toEnd]);

  const jump = useCallback(
    (instant = false): void => {
      if (root === null) return;
      apply(true);
      const top = Math.max(0, root.scrollHeight - root.clientHeight);
      if (instant || typeof root.scrollTo !== 'function') {
        ownTop.current = top;
        root.scrollTop = top;
        return;
      }
      jumping.current = true;
      root.scrollTo({ top, behavior: 'smooth' });
    },
    [root, apply],
  );

  const pin = useCallback((): void => {
    apply(true);
    toEnd();
  }, [apply, toEnd]);

  return {
    contentRef: setContent,
    pinned,
    fresh,
    showJump: !pinned && (fresh || far),
    jump,
    pin,
  };
}
