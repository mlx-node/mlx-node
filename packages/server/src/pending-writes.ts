/**
 * PendingResponseWrites — per-store in-memory index of response
 * records whose `ResponseStore.store(...)` promise has been
 * *initiated* but has not yet resolved.
 *
 * ## Why this exists
 *
 * Iter-35 moved the `await store.store(...)` call OUT of the per-model
 * `withExclusive` block so a slow SQLite flush would not pin the
 * model's mutex on the next waiter. Persistence is best-effort and
 * does not touch native model state, so holding the mutex across a
 * disk write is pure throughput loss.
 *
 * But the responses endpoint uses `store.getChain(previous_response_id)`
 * as a *gate* for every continuation: when it returns an empty array
 * the request is rejected with a `404 Previous response "…" not
 * found`. That gate ran AFTER the terminal SSE event
 * (`response.completed`) had already been flushed to the client, so a
 * client that immediately fired a follow-up request carrying
 * `previous_response_id: A` could race the handler's off-lock
 * `store.store()` write for request A and receive a spurious 404.
 *
 * Iter-36 finding 1 closes the window. The endpoint now:
 *
 *   1. STARTS the write synchronously — the `Promise<void>` returned
 *      by `store.store(record)` is registered in this tracker under
 *      the response id BEFORE the per-model mutex releases.
 *   2. Leaves the actual disk flush off the critical path — the
 *      caller does not `await` the promise until the outer handler's
 *      try/finally cleanup block.
 *   3. Has the chain-lookup path (`awaitPending(id)`) consult this
 *      tracker BEFORE treating `getChain(id).length === 0` as a
 *      404. If a write is still in flight for that id, we await it
 *      and retry `getChain`. The retry is guaranteed to see the row
 *      because the promise resolves only after the store's own
 *      serialization queue has accepted the insert.
 *
 * ## Semantics
 *
 * `track(id, promise)` registers `promise` under `id` and removes the
 * entry when the promise settles (either fulfill OR reject — a failed
 * SQLite write leaves the tracker empty and the subsequent
 * `getChain()` returns empty, which is the correct 404 shape).
 *
 * `awaitPending(id)` returns the current in-flight promise for `id`
 * or `undefined` if none is tracked. The returned promise is the
 * SAME promise that was registered — it never throws here because we
 * attach our removal handler with `.finally(...)`, leaving the
 * caller's rejection behaviour untouched. Callers typically swallow
 * rejections with `await …catch(() => {})` before retrying
 * `getChain()`, because the tracker promise's rejection is already
 * surfaced through the registering handler's separate awaiter.
 *
 * ## Scope
 *
 * One tracker per `ResponseStore` instance is attached via a
 * `WeakMap`, so callers never need to thread the tracker through the
 * handler plumbing explicitly. This keeps the same (store, tracker)
 * pair alive for the lifetime of the store and avoids leaks across
 * test suites that recreate the store per describe block.
 */

/**
 * Per-store tracker for in-flight `store.store(...)` writes.
 *
 * Thread-safety: Node.js is single-threaded within one event loop
 * tick, so the internal `Map` is safe against concurrent mutation by
 * design. Every mutation (`track`, `awaitPending`, `.finally(...)`
 * cleanup) runs synchronously within a tick.
 */
export class PendingResponseWrites {
  private readonly pending: Map<string, Promise<void>> = new Map();

  /**
   * Register an in-flight write under `id`. The caller must pass the
   * raw `Promise<void>` returned by `store.store(record)` BEFORE
   * awaiting it — otherwise the race window we are trying to close
   * reopens.
   *
   * The tracker attaches its own `.finally(...)` handler to remove
   * the entry when the promise settles. The caller's own handling of
   * the promise (await / catch / log) is unaffected because
   * `.finally` returns a new promise chain that does not steal the
   * rejection.
   */
  track(id: string, writePromise: Promise<void>): void {
    this.pending.set(id, writePromise);
    // Use `.finally` rather than `then`+`catch` so the registration
    // lifetime is symmetric across fulfill/reject — a failed SQLite
    // write should still clear the tracker so subsequent chain
    // lookups see an empty getChain() result and 404 cleanly.
    //
    // The `.finally(...)` returns a new promise whose settlement
    // mirrors `writePromise`'s; we don't await it here (cleanup is
    // fire-and-forget bookkeeping) but we DO need a catch arm so
    // the returned promise does not trigger unhandled-rejection
    // diagnostics if `writePromise` rejects. The rejection is still
    // surfaced to the caller that awaits `writePromise` directly;
    // this catch only swallows it on the cleanup fork.
    void writePromise
      .finally(() => {
        // Guard against the (unlikely but possible) case where the
        // same id has been re-registered after this write resolved —
        // only remove if WE are still the registered entry.
        if (this.pending.get(id) === writePromise) {
          this.pending.delete(id);
        }
      })
      .catch(() => {
        // Rejection already handled by the caller that awaits the
        // registered write promise; the cleanup fork just needs a
        // terminal handler to silence unhandled-rejection warnings.
      });
  }

  /**
   * Return the in-flight write promise for `id`, or `undefined` if
   * none is tracked. Callers typically await with rejection
   * suppressed (the tracker promise's rejection is already handled
   * by the separate awaiter that initiated the write) and then
   * retry `store.getChain(id)`.
   */
  awaitPending(id: string): Promise<void> | undefined {
    return this.pending.get(id);
  }

  /** Number of writes currently in flight. Primarily for tests. */
  get size(): number {
    return this.pending.size;
  }
}

/**
 * Stable `WeakMap` keyed on `ResponseStore` instances so every
 * caller gets the SAME tracker for a given store without having to
 * thread it through handler options. A `WeakMap` is safe here
 * because neither the store nor the tracker retain strong
 * references into the tracker map's keyset — if the store is GC'd
 * the tracker goes with it.
 */
const STORE_TRACKERS: WeakMap<object, PendingResponseWrites> = new WeakMap();

/**
 * Fetch (or lazily create) the tracker for a given store. Always
 * returns the same tracker for the same store instance.
 */
export function getPendingWritesFor(store: object): PendingResponseWrites {
  let tracker = STORE_TRACKERS.get(store);
  if (tracker === undefined) {
    tracker = new PendingResponseWrites();
    STORE_TRACKERS.set(store, tracker);
  }
  return tracker;
}
