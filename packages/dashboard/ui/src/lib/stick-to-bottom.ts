/**
 * Follow the end of a growing thread. While pinned, content growth smoothly chases the scroll root to its end via rAF
 * (or writes directly under reduced motion / width reflows). Only the user releases it: a wheel up, an upward touch
 * swipe or PageUp / ArrowUp / Home outside a text field, clicking/dragging the scrollbar backward, or a scroll that
 * ends more than `PIN_SLACK` px from the end. Scrolling back near the end pins again.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

import { useStaticMotion } from '../components/arc-x/motion/use-static-motion';

export const PIN_SLACK = 48;
/** The jump button shows without new content once the end is this far away. */
export const JUMP_FAR = 400;
/** Exponential smoothing time constant for auto-scroll chase (ms). */
export const CHASE_TAU_MS = 85;
/** Snap directly to target and stop the chase loop below this distance (px). */
export const CHASE_SNAP_PX = 0.75;
/** Clamp large frame deltas (e.g. backgrounded tabs) so a single tick never teleports. */
const MAX_DT_MS = 50;

export function nextChaseTop(current: number, target: number, dtMs: number): number {
  const diff = target - current;
  if (Math.abs(diff) <= CHASE_SNAP_PX) return target;
  const dt = Math.max(0, Math.min(dtMs, MAX_DT_MS));
  const alpha = 1 - Math.exp(-dt / CHASE_TAU_MS);
  const next = current + diff * alpha;
  return Math.abs(target - next) <= CHASE_SNAP_PX ? target : next;
}

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
const DOWN_KEYS = new Set(['PageDown', 'ArrowDown', 'End']);

function editable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || target.tagName === 'TEXTAREA' || target.tagName === 'INPUT';
}

function activatesOnSpace(target: EventTarget | null): boolean {
  const el = target instanceof HTMLElement ? target : target instanceof Node ? target.parentElement : null;
  if (el === null) return false;
  return (
    el.closest(
      'button, summary, select, option, [role="button"], [role="checkbox"], [role="radio"], [role="switch"], [role="menuitem"], [role="option"], [role="tab"]',
    ) !== null
  );
}

function ownsVerticalScroll(target: EventTarget | null, root: HTMLElement, direction: 'up' | 'down' = 'up'): boolean {
  let node: HTMLElement | null =
    target instanceof HTMLElement ? target : target instanceof Node ? target.parentElement : null;
  if (node !== null && node.isConnected && !root.contains(node)) return false;
  while (node !== null && node !== root) {
    const style = getComputedStyle(node);
    const overflowY = style.overflowY || style.overflow;
    const scrollsY = overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'overlay';
    const isScrollContainer = scrollsY || overflowY === 'hidden';
    if (isScrollContainer) {
      const y = style.overscrollBehaviorY || style.overscrollBehavior;
      if (y === 'contain' || y === 'none') return false;
    }
    if (scrollsY && node.scrollHeight > node.clientHeight + 1) {
      if (direction === 'up' && node.scrollTop > 0) return false;
      if (direction === 'down' && node.scrollTop + node.clientHeight < node.scrollHeight - 1) return false;
    }
    node = node.parentElement;
  }
  return true;
}

function touchTarget(point: Touch, fallback: EventTarget | null): EventTarget | null {
  return (point.target as EventTarget | null | undefined) ?? fallback;
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

/**
 * Whether a backward jump during an active chase came from the user rather than a layout shrink (scroll anchoring or
 * clamping to a shorter end). A shrink of `shrink` px can explain a drop of up to `shrink` px (plus rounding).
 */
function isUserScrollBack(m: ScrollMetrics, lastWritten: number, lastScrollHeight: number): boolean {
  const drop = lastWritten - m.scrollTop;
  if (drop <= 2) return false;
  const shrink = Math.max(0, lastScrollHeight - m.scrollHeight);
  return drop > shrink + 2 && distanceToEnd(m) > 2;
}

interface ChaseState {
  frame: number;
  pos: number;
  lastTs: number | null;
  lastWritten: number;
  lastScrollHeight: number;
  prevOverflowAnchor: string;
}

export function useStickToBottom(root: HTMLElement | null, key: string): StickToBottom {
  const staticMotion = useStaticMotion();
  const staticMotionRef = useRef(staticMotion);
  staticMotionRef.current = staticMotion;
  useLayoutEffect(() => {
    staticMotionRef.current = staticMotion;
  }, [staticMotion]);

  const [pinned, setPinned] = useState(true);
  const [fresh, setFresh] = useState(false);
  const [far, setFar] = useState(false);
  const pinnedRef = useRef(true);
  const chase = useRef<ChaseState | null>(null);
  const ownTop = useRef<number | null>(null);
  const touchHold = useRef(false);
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

  const stopChase = useCallback((): void => {
    const state = chase.current;
    if (state === null) return;
    if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(state.frame);
    if (root !== null) root.style.overflowAnchor = state.prevOverflowAnchor;
    chase.current = null;
  }, [root]);

  const toEnd = useCallback((): void => {
    stopChase();
    if (root === null) return;
    const top = Math.max(0, root.scrollHeight - root.clientHeight);
    ownTop.current = top;
    root.scrollTop = top;
    ownTop.current = root.scrollTop;
  }, [root, stopChase]);

  const chaseToEnd = useCallback((): void => {
    if (root === null) return;
    if (staticMotionRef.current || typeof requestAnimationFrame !== 'function') {
      toEnd();
      return;
    }
    const target = Math.max(0, root.scrollHeight - root.clientHeight);
    if (Math.abs(target - root.scrollTop) <= CHASE_SNAP_PX) {
      toEnd();
      return;
    }
    if (chase.current !== null) return;

    const startTop = root.scrollTop;
    const state: ChaseState = {
      frame: 0,
      pos: startTop,
      lastTs: null,
      lastWritten: startTop,
      lastScrollHeight: root.scrollHeight,
      prevOverflowAnchor: root.style.overflowAnchor,
    };
    root.style.overflowAnchor = 'none';
    chase.current = state;

    const step = (now: number): void => {
      if (chase.current !== state) return;
      if (staticMotionRef.current) {
        toEnd();
        return;
      }
      const m = metrics();
      const maxTop = Math.max(0, m.scrollHeight - m.clientHeight);

      if (m.scrollTop < state.lastWritten - 2) {
        if (isUserScrollBack(m, state.lastWritten, state.lastScrollHeight)) {
          stopChase();
          apply(false);
          return;
        }
        state.pos = m.scrollTop;
        state.lastWritten = m.scrollTop;
        state.lastScrollHeight = m.scrollHeight;
      } else if (m.scrollTop > state.lastWritten + 2) {
        state.pos = m.scrollTop;
      }

      const dt = state.lastTs === null ? 16 : now - state.lastTs;
      state.lastTs = now;
      const next = nextChaseTop(state.pos, maxTop, dt);
      state.pos = next;
      ownTop.current = next;
      state.lastWritten = next;
      state.lastScrollHeight = m.scrollHeight;
      root.scrollTop = next;
      const applied = root.scrollTop;
      ownTop.current = applied;
      state.lastWritten = applied;
      state.lastScrollHeight = root.scrollHeight;

      if (Math.abs(maxTop - next) <= CHASE_SNAP_PX || Math.abs(maxTop - applied) <= CHASE_SNAP_PX) {
        stopChase();
        if (Math.abs(maxTop - applied) > 0) {
          ownTop.current = maxTop;
          root.scrollTop = maxTop;
          ownTop.current = root.scrollTop;
        }
        return;
      }
      state.frame = requestAnimationFrame(step);
    };

    state.frame = requestAnimationFrame(step);
  }, [root, metrics, apply, toEnd, stopChase]);

  useLayoutEffect(() => {
    stopChase();
    touchHold.current = false;
    const place = places.get(key);
    if (place !== undefined && !place.pinned && root !== null) {
      ownTop.current = place.top;
      root.scrollTop = place.top;
      ownTop.current = root.scrollTop;
      apply(false);
    } else {
      apply(true);
      toEnd();
    }
    return () => {
      stopChase();
      touchHold.current = false;
      if (root !== null) places.set(key, { top: root.scrollTop, pinned: pinnedRef.current });
    };
  }, [key, root, apply, toEnd, stopChase]);

  useEffect(() => {
    if (root === null) return;
    const targetWindow = root.ownerDocument.defaultView ?? window;
    const release = (): void => {
      stopChase();
      if (pinnedRef.current) apply(false);
    };
    const clearTouchHold = (): void => {
      if (!touchHold.current) return;
      touchHold.current = false;
      if (!pinnedRef.current && distanceToEnd(metrics()) <= PIN_SLACK) apply(true);
    };
    const onWheel = (event: WheelEvent): void => {
      if (event.ctrlKey) return;
      if (event.deltaY < 0) {
        if (ownsVerticalScroll(event.target, root, 'up')) release();
      } else if (event.deltaY > 0) {
        if (ownsVerticalScroll(event.target, root, 'down')) clearTouchHold();
      }
    };
    const onKey = (event: KeyboardEvent): void => {
      if (event.defaultPrevented) return;
      if (!(event.target instanceof Node) || !root.contains(event.target)) return;
      if (editable(event.target)) return;
      if (event.key === ' ' && activatesOnSpace(event.target)) return;
      if (RELEASE_KEYS.has(event.key) || (event.key === ' ' && event.shiftKey)) {
        if (ownsVerticalScroll(event.target, root, 'up')) release();
      } else if (DOWN_KEYS.has(event.key) || (event.key === ' ' && !event.shiftKey)) {
        if (ownsVerticalScroll(event.target, root, 'down')) clearTouchHold();
      }
    };
    const onPointerDown = (event: PointerEvent): void => {
      if (event.target !== root) return;
      const rect = root.getBoundingClientRect();
      const onScrollbar =
        event.clientX >= rect.left + root.clientLeft + root.clientWidth || event.clientX < rect.left + root.clientLeft;
      if (onScrollbar) {
        touchHold.current = false;
        if (chase.current !== null) stopChase();
      }
    };
    const touch: {
      active: boolean;
      id: number;
      minY: number;
      anchorX: number;
      maxY: number;
      downAnchorX: number;
      target: EventTarget | null;
    } = {
      active: false,
      id: 0,
      minY: 0,
      anchorX: 0,
      maxY: 0,
      downAnchorX: 0,
      target: null,
    };
    const canTrackTouch = (target: EventTarget | null, direction: 'up' | 'down' = 'up'): boolean =>
      !editable(target) && ownsVerticalScroll(target, root, direction);
    const canStartTouch = (target: EventTarget | null): boolean =>
      !editable(target) && (ownsVerticalScroll(target, root, 'up') || ownsVerticalScroll(target, root, 'down'));
    const resetTouch = (point: Touch, target: EventTarget | null): void => {
      touch.active = true;
      touch.id = point.identifier;
      touch.minY = point.clientY;
      touch.anchorX = point.clientX;
      touch.maxY = point.clientY;
      touch.downAnchorX = point.clientX;
      touch.target = target;
    };
    const findTouch = (list: TouchList, id: number): Touch | undefined => {
      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        if (item !== undefined && item.identifier === id) return item;
      }
      return undefined;
    };
    const onTouchStart = (event: TouchEvent): void => {
      const first = event.touches[0];
      if (first === undefined || !canStartTouch(event.target)) {
        touch.active = false;
        touch.target = null;
        return;
      }
      const target = touchTarget(first, event.target);
      if (target !== event.target && !canStartTouch(target)) {
        touch.active = false;
        touch.target = null;
        return;
      }
      resetTouch(first, target);
    };
    const onTouchMove = (event: TouchEvent): void => {
      if (!touch.active) return;
      const point = findTouch(event.touches, touch.id);
      if (point === undefined) {
        const fallback = event.touches[0];
        const fallbackTarget = fallback === undefined ? null : touchTarget(fallback, event.target);
        if (fallback === undefined || !canStartTouch(fallbackTarget)) {
          touch.active = false;
          touch.target = null;
          return;
        }
        resetTouch(fallback, fallbackTarget);
        return;
      }
      if (point.clientY > touch.maxY) {
        touch.maxY = point.clientY;
        touch.downAnchorX = point.clientX;
      }
      if (point.clientY < touch.minY) {
        touch.minY = point.clientY;
        touch.anchorX = point.clientX;
      }
      const dyDown = touch.maxY - point.clientY;
      const dxDown = Math.abs(point.clientX - touch.downAnchorX);
      if (dyDown > 2 && dyDown >= dxDown) {
        const currentTarget = touchTarget(point, touch.target);
        if (!canTrackTouch(currentTarget, 'down')) {
          touch.maxY = point.clientY;
          touch.downAnchorX = point.clientX;
          return;
        }
        clearTouchHold();
        touch.minY = point.clientY;
        touch.anchorX = point.clientX;
        touch.maxY = point.clientY;
        touch.downAnchorX = point.clientX;
        return;
      }
      const dy = point.clientY - touch.minY;
      const dx = Math.abs(point.clientX - touch.anchorX);
      if (dy > 2 && dy >= dx) {
        const currentTarget = touchTarget(point, touch.target);
        if (!canTrackTouch(currentTarget, 'up')) {
          touch.minY = point.clientY;
          touch.anchorX = point.clientX;
          return;
        }
        touchHold.current = true;
        release();
      }
    };
    const onTouchEnd = (event: TouchEvent): void => {
      if (!touch.active) return;
      if (findTouch(event.touches, touch.id) !== undefined) return;
      const remaining = event.touches[0];
      const remainingTarget = remaining === undefined ? null : touchTarget(remaining, event.target);
      if (remaining !== undefined && canStartTouch(remainingTarget)) {
        resetTouch(remaining, remainingTarget);
      } else {
        touch.active = false;
        touch.target = null;
      }
    };
    const onScroll = (): void => {
      const m = metrics();
      const activeChase = chase.current;
      if (activeChase !== null) {
        if (Math.abs(m.scrollTop - activeChase.lastWritten) <= 1.5) return;
        if (m.scrollTop < activeChase.lastWritten - 2) {
          if (isUserScrollBack(m, activeChase.lastWritten, activeChase.lastScrollHeight)) {
            stopChase();
            apply(false);
          } else {
            activeChase.pos = m.scrollTop;
            activeChase.lastWritten = m.scrollTop;
            activeChase.lastScrollHeight = m.scrollHeight;
          }
          return;
        }
        activeChase.pos = Math.max(activeChase.pos, m.scrollTop);
        activeChase.lastWritten = m.scrollTop;
        activeChase.lastScrollHeight = m.scrollHeight;
        return;
      }
      if (ownTop.current !== null && Math.abs(m.scrollTop - ownTop.current) <= 1.5) {
        ownTop.current = null;
        return;
      }
      ownTop.current = null;
      if (touchHold.current) {
        setFar(distanceToEnd(m) > JUMP_FAR);
        return;
      }
      const next = nextPinned(pinnedRef.current, m, 'scroll');
      if (next !== pinnedRef.current) apply(next);
      else setFar(distanceToEnd(m) > JUMP_FAR);
    };
    root.addEventListener('wheel', onWheel, { passive: true });
    targetWindow.addEventListener('keydown', onKey);
    root.addEventListener('pointerdown', onPointerDown, { passive: true });
    root.addEventListener('touchstart', onTouchStart, { passive: true });
    root.addEventListener('touchmove', onTouchMove, { passive: true });
    root.addEventListener('touchend', onTouchEnd, { passive: true });
    root.addEventListener('touchcancel', onTouchEnd, { passive: true });
    root.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      root.removeEventListener('wheel', onWheel);
      targetWindow.removeEventListener('keydown', onKey);
      root.removeEventListener('pointerdown', onPointerDown);
      root.removeEventListener('touchstart', onTouchStart);
      root.removeEventListener('touchmove', onTouchMove);
      root.removeEventListener('touchend', onTouchEnd);
      root.removeEventListener('touchcancel', onTouchEnd);
      root.removeEventListener('scroll', onScroll);
    };
  }, [root, metrics, apply, stopChase]);

  useEffect(() => {
    if (content === null || typeof ResizeObserver !== 'function') return;
    const first = content.getBoundingClientRect();
    let last = first.height;
    let lastWidth = first.width;
    const observer = new ResizeObserver(() => {
      const box = content.getBoundingClientRect();
      // A column that got narrower is taller without any new message.
      const widthChanged = box.width !== lastWidth;
      const grew = box.height > last && !widthChanged;
      last = box.height;
      lastWidth = box.width;
      if (pinnedRef.current) {
        if (widthChanged) toEnd();
        else chaseToEnd();
      } else if (grew) setFresh(true);
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, [content, toEnd, chaseToEnd]);

  const jump = useCallback(
    (instant = false): void => {
      touchHold.current = false;
      apply(true);
      if (instant || staticMotionRef.current || typeof requestAnimationFrame !== 'function') toEnd();
      else chaseToEnd();
    },
    [apply, toEnd, chaseToEnd],
  );

  const pin = useCallback((): void => {
    touchHold.current = false;
    apply(true);
    chaseToEnd();
  }, [apply, chaseToEnd]);

  return {
    contentRef: setContent,
    pinned,
    fresh,
    showJump: !pinned && (fresh || far),
    jump,
    pin,
  };
}
