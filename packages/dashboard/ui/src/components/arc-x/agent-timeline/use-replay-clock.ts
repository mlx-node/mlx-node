import { useEffect, useRef, useSyncExternalStore } from 'react';

import type { AgentClock } from './agent-timeline-lib';

/** Largest step one frame may take, so a frame after a stall does not jump. */
export const MAX_FRAME_MS = 100;

export const browserClock: AgentClock = {
  now: () => Date.now(),
  requestFrame: (callback) => requestAnimationFrame(callback),
  cancelFrame: (handle) => cancelAnimationFrame(handle),
};

function windowHidden(): boolean {
  return typeof document !== 'undefined' && document.documentElement.hasAttribute('data-hidden');
}

function subscribeHidden(listener: () => void): () => void {
  if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') return () => {};
  const observer = new MutationObserver(listener);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-hidden'] });
  return () => observer.disconnect();
}

/**
 * The one frame loop of an AgentTimeline. While `running` and the window is shown (`:root` without `data-hidden`),
 * calls `onFrame(dtMs)` once per frame with the time since the previous frame, capped at MAX_FRAME_MS. A hidden window
 * requests no frame; the first frame after it advances by 0.
 */
export function useReplayClock(running: boolean, clock: AgentClock, onFrame: (dtMs: number) => void): void {
  const frame = useRef(onFrame);
  frame.current = onFrame;
  const hidden = useSyncExternalStore(subscribeHidden, windowHidden, () => false);
  const active = running && !hidden;
  useEffect(() => {
    if (!active) return;
    let last: number | null = null;
    let handle = 0;
    const tick = (time: number): void => {
      const dt = last === null ? 0 : Math.min(MAX_FRAME_MS, Math.max(0, time - last));
      last = time;
      frame.current(dt);
      handle = clock.requestFrame(tick);
    };
    handle = clock.requestFrame(tick);
    return () => clock.cancelFrame(handle);
  }, [active, clock]);
}
