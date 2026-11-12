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
 * Models page's in-progress download map — and on the
 * Models page that state IS the download UI the fix is meant to rescue.
 */

let generation = 0;
let transportUp = false;
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) listener();
}

/** Called by `connectDashboardApi` — one place decides what a reconnect means. */
export function bumpConnectionGeneration(): void {
  generation += 1;
  transportUp = true;
  notify();
}

/** The current port died or was dropped; the generation stays until a replacement port arrives. */
export function markTransportDown(): void {
  if (!transportUp) return;
  transportUp = false;
  notify();
}

/** Whether the current port can carry calls: the shell's `connected`. */
export function getTransportUp(): boolean {
  return transportUp;
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
