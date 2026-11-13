/**
 * Last-known body for each GET path, kept for the lifetime of the window.
 *
 * This exists to remove a loading state, not to save a request. Every page is a
 * fresh mount — react-router unmounts the old route's component tree on
 * navigation — so without a cache, returning to a tab you looked at two seconds
 * ago starts again from `data === undefined` and paints the whole page as
 * skeletons before the runtime answers. The data was already correct; the
 * skeleton was pure flicker, and it is the flicker the user sees on every single
 * tab switch.
 *
 * A hit does NOT stop the refetch: {@link useJson} renders the cached body
 * immediately and revalidates in the background, so what you see is at worst one
 * round-trip stale and is corrected without ever collapsing the layout.
 *
 * Bounded because the key space is not: `/metrics/summary?from=…&to=…`
 * mints a new key every time the range
 * selector moves. Insertion order is the recency order — a read reinserts — so
 * evicting from the front is a true LRU.
 */

/** Enough to hold every page's working set several times over; see the note above. */
const MAX_ENTRIES = 48;

const store = new Map<string, unknown>();

/** Bumped by every {@link clearCache}; a body read before a clear describes the state the clear invalidated. */
let epoch = 0;

/** The current cache epoch. A reader takes it before its request starts and hands it to {@link writeCache}. */
export function cacheEpoch(): number {
  return epoch;
}

/**
 * The cached body for `path`, wrapped so a hit is distinguishable from a miss.
 *
 * A bare `T | undefined` would conflate the two the moment a route legitimately
 * answers `null`, and the difference decides whether the page shows content or a
 * skeleton.
 */
export function readCache<T>(path: string): { value: T } | undefined {
  if (!store.has(path)) return undefined;
  const value = store.get(path) as T;
  // Reinsert so the entry counts as recently used.
  store.delete(path);
  store.set(path, value);
  return { value };
}

/**
 * Store `value` for `path`, unless a clear happened after the request that read it started (`startedAt` is the
 * {@link cacheEpoch} taken then): a GET in flight across a delete answers with the deleted row, and seeding the next
 * mount with it would paint the row again.
 */
export function writeCache(path: string, value: unknown, startedAt: number = epoch): void {
  if (startedAt !== epoch) return;
  store.delete(path);
  store.set(path, value);
  while (store.size > MAX_ENTRIES) {
    const oldest = store.keys().next();
    if (oldest.done === true) break;
    store.delete(oldest.value);
  }
}

/**
 * Drop everything.
 *
 * Called on any mutation and on any (dis)connection. Both are coarse on purpose:
 * a write is not confined to the path it addressed — deleting a model changes
 * `/models`, `/catalog` and the Overview tiles that count them — and a
 * per-path invalidation list would be a second map of which route affects which,
 * kept in sync by hand and wrong the first time a route is added. Refetching a
 * handful of local RPCs costs less than that class of bug.
 */
export function clearCache(): void {
  store.clear();
  epoch += 1;
}
