/**
 * A clock that does not move on every render.
 *
 * Pages that ask for "the last N days" build a request path from `Date.now()`,
 * and that path is the {@link import('./json-cache').readCache} key. React
 * Router unmounts a route on navigation, so `useMemo` re-runs on every visit and
 * a raw `Date.now()` mints a NEW key each time — the page can never hit its own
 * cache, however many times it is opened.
 *
 * Measured on the Metrics page before this existed: every other tab settled in
 * 20-48ms from cache, Metrics missed on every visit and took 154ms. That 154ms
 * straddles the delay before a skeleton fades in, so the window sat blank and
 * then flashed grey for one frame — which is what made tab switching feel slow.
 *
 * Rounding down to a whole minute costs at most 60s off the upper bound of the
 * range. Both callers bucket by day, so it is not observable in the chart; a
 * revalidation still runs on every visit, so the data inside the window is
 * fresh. What changes is only how often the window's right edge moves.
 */

/** Rounding interval. Bigger means more cache hits and a staler upper bound. */
export const NOW_QUANTUM_MS = 60_000;

export function stableNow(): number {
  return Math.floor(Date.now() / NOW_QUANTUM_MS) * NOW_QUANTUM_MS;
}
