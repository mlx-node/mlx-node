import * as React from 'react';

/**
 * Returns a ref to attach to a chapter's result panel. Every time `running`
 * transitions `true -> false` (a run just completed), the hook restarts a
 * brief border-flash animation on that element via the classic
 * `remove class -> force reflow -> add class` trick.
 *
 * Why this exists: chapter Run buttons are deterministic on the same prompt
 * (greedy sampling, no temperature), so a second click produces byte-for-byte
 * identical output. Without an animated "yes, this just refreshed" signal,
 * users click Run and feel like nothing happened.
 *
 * The CSS class `.run-flash-on-mount` and the `@keyframes runFlash` rule
 * live in `styles.css`. Chapters 3, 4, 6, 7, 8, 9 all use this.
 */
export function useRunFlash<T extends HTMLElement = HTMLDivElement>(running: boolean): React.RefObject<T | null> {
  const ref = React.useRef<T>(null);
  const prevRunningRef = React.useRef(running);

  React.useEffect(() => {
    const wasRunning = prevRunningRef.current;
    prevRunningRef.current = running;
    // Only flash on the true -> false edge (a run just completed). Skip the
    // initial mount (false -> false) and the start of a run (false -> true).
    if (!wasRunning || running) return;
    const el = ref.current;
    if (!el) return;
    el.classList.remove('run-flash-on-mount');
    // Force a reflow so removing + re-adding the class restarts the
    // animation. Reading offsetWidth is the standard cross-browser trick.
    void el.offsetWidth;
    el.classList.add('run-flash-on-mount');
  }, [running]);

  return ref;
}
