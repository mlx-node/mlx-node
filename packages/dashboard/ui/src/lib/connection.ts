/**
 * A counter that ticks every time the app is handed a new RPC port.
 *
 * When CONTROL PANEL crashes, `broker.ts` respawns it and hands the SAME live page a
 * replacement port — it goes to real trouble to do this, because a dashboard
 * that stays blank until the user thinks to reload is the failure that whole
 * restart path exists to prevent. But re-rendering the tree does not remount
 * it: `root.render` on an existing root matches on element type and key, so the
 * fibers are preserved, effects do not re-run, and every mounted hook stays
 * bound to the dead generation.
 *
 * `clearCache()` alone does not fix that. It is read only by `useJson`'s state
 * initializer, so it makes FUTURE mounts fetch fresh and is invisible to
 * anything already on screen. The result was a page that looked fine and was
 * frozen: a download card spinning at 45% forever against a healthy runtime,
 * and cards stuck on `E_UNAVAILABLE` until the user happened to navigate away
 * and back.
 *
 * Subscribing to this instead of keying the whole tree on it is deliberate.
 * Remounting would also discard scroll position, an open confirm dialog, the
 * Models page's in-progress download map and the sessions filter — and on the
 * Models page that state IS the download UI the fix is meant to rescue.
 */

let generation = 0;
const listeners = new Set<() => void>();

/** Called by `connectDashboardApi` — one place decides what a reconnect means. */
export function bumpConnectionGeneration(): void {
  generation += 1;
  for (const listener of listeners) listener();
}

export function subscribeConnection(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getConnectionGeneration(): number {
  return generation;
}
