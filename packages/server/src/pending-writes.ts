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
 * ## Hard-timeout marker state (iter-50 / iter-51)
 *
 * Iter-49 added a `PendingResponseWrites.evict(id)` primitive so the
 * hard-timeout breaker in `responses.ts` could drop a wedged persist's
 * tracker entry without waiting for the promise to settle. That bounded
 * tracker memory under a truly stuck store, but it also removed the
 * ONLY signal a `previous_response_id` continuation uses to classify a
 * miss as "retryable storage slowness" vs. "permanent 404": after
 * eviction `awaitPending(id)` returned `undefined`, the continuation
 * took the 404 branch, and slow-but-eventual persists that crossed the
 * hard timeout were misclassified as permanent history loss.
 *
 * Iter-50 replaced the outright eviction with a two-phase pending
 * state. `markHardTimedOut(id)` removed the id from the `pending`
 * promise tracker (so `awaitPending` stopped handing out the stale
 * promise and the closure chain was reclaimable) AND added the id to a
 * lightweight `hardTimedOut` `Set<string>` marker. The continuation
 * path consulted `isHardTimedOut(id)` before falling through to
 * `sendNotFound(...)`: when the marker was set it returned the same
 * retryable 503 `storage_timeout` shape the normal `awaitPending`
 * timeout path already used.
 *
 * Iter-51 (codex's iter-50 HIGH finding 1): the iter-50 marker was
 * cleared ONLY by the underlying write promise's `.finally(...)`
 * handler installed in `track()`. For a truly wedged
 * `store.store(...)` that never settles, that handler never runs, so
 * under sustained traffic against a wedged backend the marker set
 * accumulated forever — one entry per hard-timed-out request — and
 * continuations returned retryable 503 for those ids forever (a
 * steady memory leak and an incorrect eventual classification).
 *
 * The fix is an independent TTL with lazy expiry. Markers are now
 * stored as `Map<string, { expiresAt }>`. `markHardTimedOut(id,
 * ttlMs)` computes `expiresAt = Date.now() + ttlMs` at call time;
 * `isHardTimedOut(id)` treats any entry whose `expiresAt <=
 * Date.now()` as absent and deletes it on the way out. Steady-state
 * marker memory is bounded at O(requestRate × TTL) regardless of how
 * long the store stays wedged. Caller (`responses.ts`) reads the TTL
 * from `MLX_HARD_TIMEOUT_MARKER_TTL_MS` (default 300_000ms) and passes
 * it in explicitly so this module stays env-free.
 *
 * The underlying-promise `.finally(...)` path is preserved: if a
 * wedged write does eventually settle the marker is cleared
 * immediately (fast path), otherwise the TTL closes the retryable
 * window after a bounded wall-clock interval (slow path). Marker
 * lifetime is therefore bounded by min(write settlement, TTL expiry),
 * which is finite in every scenario.
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
   * Ids that crossed the hard-timeout breaker in responses.ts while
   * their `store.store(...)` promise was still unresolved (iter-50,
   * TTL-bounded in iter-51).
   *
   * Each entry records `{ expiresAt }` in epoch-ms. Two cleanup paths:
   *
   *   1. Fast path — the underlying write promise's `.finally(...)`
   *      inside `track()` deletes the entry as soon as the wedged
   *      store unwedges. Observable latency: as fast as the write
   *      itself settles.
   *   2. Slow path — `isHardTimedOut(id)` lazily deletes any entry
   *      whose `expiresAt` has passed. No timers, no orphan state.
   *      Bounds steady-state memory under a permanently wedged store
   *      at O(requestRate × TTL).
   *
   * Invariant: a marker is only meaningful for the SPECIFIC write
   * that was live when `markHardTimedOut` was called. If a later
   * `track(id, newPromise)` re-uses the same id after a marker was
   * set, the original promise's `.finally(...)` will still clear the
   * marker on its settlement (clearing the wrong state for the new
   * promise). Callers that mix hard-timed-out ids with brand-new
   * writes under the same id MUST call `markHardTimedOut(id, ttlMs)`
   * again after the new `track(id, ...)` if they want the marker to
   * persist — in practice the responses endpoint scopes response ids
   * to a single persist each, so this collision cannot arise.
   */
  private readonly hardTimedOut: Map<string, { expiresAt: number }> = new Map();

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
        // Iter-50/51: the hard-timeout marker is also cleared here
        // as the fast path. Once this specific `writePromise`
        // resolves or rejects, any marker set against its id during
        // its lifetime is no longer meaningful — the
        // continuation path's retryable-503 window closes at the
        // moment the wedged store unwedges. Under a truly wedged
        // store that NEVER settles this handler never fires, which
        // is why iter-51 also attaches a TTL expiry path (see
        // `isHardTimedOut`): marker lifetime is bounded by
        // min(write settlement, TTL expiry).
        //
        // We clear unconditionally because the marker is a `Map`
        // keyed on id (not a promise reference), so there is no
        // "entry replaced" concern: whoever later re-registers the
        // id must explicitly re-mark if they want the retryable
        // window open again.
        this.hardTimedOut.delete(id);
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

  /**
   * Transition a pending entry to the hard-timed-out marker state.
   *
   * Called by the hard-timeout breaker in `responses.ts` when an
   * in-flight `store.store(...)` has crossed the hard timeout and
   * is presumed wedged. The `pending` entry is removed so that:
   *
   *   * `awaitPending(id)` stops handing out the stale promise
   *     (continuations no longer block on it).
   *   * The promise closure chain referenced through the tracker is
   *     reclaimable — iter-49's memory bound is preserved.
   *
   * The id is added to the `hardTimedOut` marker map so the
   * `previous_response_id` continuation path can distinguish a
   * hard-timed-out slow persist from a genuinely missing chain:
   * `isHardTimedOut(id) === true` means the write may still land
   * eventually, so the continuation returns retryable 503
   * `storage_timeout` instead of permanent 404.
   *
   * Iter-51: the `ttlMs` argument caps the marker lifetime
   * independently of whether the underlying write settles. The fast
   * cleanup path inside `track()`'s `.finally(...)` still fires when
   * the wedged store eventually unwedges; the TTL is the slow path
   * that guarantees bounded memory under a permanently wedged store.
   * The caller (`responses.ts`) reads the TTL from the
   * `MLX_HARD_TIMEOUT_MARKER_TTL_MS` env var; passing it in here
   * keeps this module env-free.
   *
   * Returns true if the id was an active pending entry (and has
   * been moved into the hard-timed-out marker), false if no pending
   * entry existed at call time. A false return does NOT add the id
   * to the marker — a marker without a backing promise has no
   * cleanup signal (beyond the TTL) and could leak if the caller
   * mis-routes ids. The TTL would eventually drain it, but in the
   * meantime continuations would see a spurious retryable-503
   * signal.
   */
  markHardTimedOut(id: string, ttlMs: number): boolean {
    const wasPending = this.pending.delete(id);
    if (wasPending) {
      this.hardTimedOut.set(id, { expiresAt: Date.now() + ttlMs });
    }
    return wasPending;
  }

  /**
   * Whether `id` is currently flagged as hard-timed-out. Used by the
   * `previous_response_id` continuation path to classify a missing
   * chain as retryable 503 `storage_timeout` vs. permanent 404.
   *
   * Iter-51: lazily expires the marker on read. If an entry's
   * `expiresAt` has passed, it is deleted and the method returns
   * false. This is the slow cleanup path — it bounds steady-state
   * marker memory even if the underlying write promise never settles.
   */
  isHardTimedOut(id: string): boolean {
    const entry = this.hardTimedOut.get(id);
    if (entry === undefined) return false;
    if (entry.expiresAt <= Date.now()) {
      this.hardTimedOut.delete(id);
      return false;
    }
    return true;
  }

  /** Number of writes currently in flight. Primarily for tests. */
  get size(): number {
    return this.pending.size;
  }

  /**
   * Number of ids currently in the hard-timed-out marker state.
   * Primarily for tests — exposes iter-50's bookkeeping so
   * regressions can assert both that the marker is set after a
   * hard-timeout fires AND that it drains when the underlying write
   * eventually settles.
   *
   * Iter-51: lazily drains expired entries on read so the reported
   * count reflects the count a subsequent `isHardTimedOut` sweep
   * would see. This is important for the "sustained-traffic against
   * a wedged store + TTL bound" regression: the test asserts that
   * after the TTL window elapses, the tracker reports zero markers
   * even though no underlying write has settled.
   */
  get hardTimedOutSize(): number {
    const now = Date.now();
    for (const [id, entry] of this.hardTimedOut) {
      if (entry.expiresAt <= now) {
        this.hardTimedOut.delete(id);
      }
    }
    return this.hardTimedOut.size;
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
