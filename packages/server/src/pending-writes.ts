/**
 * PendingResponseWrites — per-store in-memory index of response
 * records whose `ResponseStore.store(...)` promise has been initiated
 * but has not yet resolved.
 *
 * ## Why this exists
 *
 * The responses endpoint starts `store.store(record)` synchronously
 * inside the per-model `withExclusive` block (so the tracker observes
 * the in-flight write before the mutex releases) but does NOT await
 * it on the critical path. Without this tracker, a client that fires
 * a follow-up request carrying `previous_response_id: A` immediately
 * after seeing `response.completed` could race the off-lock
 * `store.store()` for A and be rejected with a spurious
 * `404 Previous response not found` because `getChain()` had not yet
 * seen the row.
 *
 * The chain-lookup path consults `awaitPending(id)` BEFORE treating
 * `getChain(id).length === 0` as a 404. If a write is still in flight,
 * it awaits and retries `getChain`; the retry is guaranteed to see
 * the row because the promise resolves only after the store's own
 * serialization queue has accepted the insert.
 *
 * ## Semantics
 *
 * `track(id, promise)` registers `promise` under `id` and removes the
 * entry when the promise settles (fulfill OR reject — a failed write
 * leaves the tracker empty and the subsequent `getChain()` returns
 * empty, which is the correct 404 shape).
 *
 * `awaitPending(id)` returns the tracked in-flight promise, or
 * undefined. It is the SAME promise that was registered; our removal
 * handler is attached via `.finally(...)` so the caller's rejection
 * behaviour is unaffected. Callers typically swallow rejections with
 * `await …catch(() => {})` before retrying `getChain()` because the
 * rejection is already surfaced through the registering awaiter.
 *
 * ## Hard-timeout marker state
 *
 * When the responses-endpoint breaker decides a pending write is
 * wedged it calls `markHardTimedOut(id, ttlMs, absoluteExpiresAt)`,
 * which removes the id from `pending` (so the closure chain is
 * reclaimable) and adds it to a lightweight `hardTimedOut` map. The
 * continuation path then classifies a missing chain as retryable 503
 * `storage_timeout` instead of permanent 404 while the marker is live.
 *
 * Marker lifetime is bounded by:
 *
 *   min(write settlement, last continuation + TTL, absoluteExpiresAt)
 *
 * Five cleanup/refresh paths keep memory bounded and classification
 * honest:
 *
 *   1. Fast path — `track()`'s `.finally(...)` deletes the marker as
 *      soon as the wedged write settles.
 *   2. Read-refresh path — `isHardTimedOut(id)` slides `expiresAt`
 *      forward by `ttlMs` on every live hit (clamped at
 *      `absoluteExpiresAt`). Actively retried chains stay recoverable
 *      as long as the underlying write might still land.
 *   3. Read-expire path — a full TTL elapse without refresh lazily
 *      deletes the entry and classifies the id as permanent 404.
 *   4. Read-absolute-cap path — once `Date.now() >= absoluteExpiresAt`,
 *      the marker is deleted unconditionally. `ResponseStore.getChain()`
 *      hides the row past its own row TTL, so the retryable-503
 *      classification would be factually wrong.
 *   5. Write-sweep path — `markHardTimedOut()` drains expired entries
 *      before inserting, bounded to `MAX_SWEEP_PER_INSERT` visits per
 *      call to keep the transition O(1) amortized even when the map
 *      is large. `isHardTimedOut()` moves refreshed entries to the
 *      Map tail so the bounded sweep cannot be starved by a stable
 *      head cohort of hot entries (natural LRU behaviour).
 *
 * The caller-side `absoluteExpiresAt` is the MINIMUM `expiresAt`
 * across the whole resolved chain, not just the child record.
 * `ResponseStore.getChain()` walks ancestors and aborts on the first
 * expired link (see `crates/mlx-db/src/response_store/reader.rs:44-59`),
 * so a child whose parent expires sooner is unrecoverable at the
 * parent's expiry — not the child's. This module just receives the
 * min-clamped value.
 *
 * ## Pending-entry earliest-expiry metadata
 *
 * The pre-breaker `awaitPending` timeout/probe path in `responses.ts`
 * would otherwise classify any unresolved pending write as retryable
 * `storage_timeout`, even when the resolved chain's earliest ancestor
 * has already expired. A continuation whose parent is already past
 * its row TTL cannot ever succeed via `getChain()`, so looping the
 * client on 503 until the hard breaker fires is wasted.
 *
 * `track(id, promise, earliestExpiresAtMs?)` records the earliest
 * recoverable expiry alongside the tracked promise in a per-id side
 * map (`earliestExpiresByPending`). `getEarliestExpiresAtMs(id)`
 * exposes it so the endpoint can short-circuit to 404 once
 * `Date.now() >= earliestExpiresAtMs`. The side map is keyed
 * identically to `pending` and cleared on the same `.finally(...)`
 * hook — no extra lifecycle surface.
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
   * Per-pending-entry scalar recording the EARLIEST recoverable
   * wall-clock expiry across (record + resolved ancestor chain) at
   * `track()` time. Keyed identically to `pending` and cleared on
   * the same `.finally(...)` settlement hook.
   *
   * The pre-breaker `awaitPending` timeout/probe path consults this
   * via `getEarliestExpiresAtMs(id)`: once `Date.now()` has passed
   * the earliest ancestor expiry, `getChain()` can never succeed,
   * so the continuation short-circuits to 404 rather than looping
   * the client on retryable 503.
   *
   * Optional: callers that pass `undefined` do not populate the
   * side map; `getEarliestExpiresAtMs(id)` then returns `undefined`
   * and the caller falls back to emitting retryable 503.
   */
  private readonly earliestExpiresByPending: Map<string, number> = new Map();

  /**
   * Ids that crossed the hard-timeout breaker in `responses.ts`
   * while their `store.store(...)` promise was still unresolved.
   *
   * Each entry records `{ expiresAt, ttlMs, absoluteExpiresAt }` in
   * epoch-ms. `expiresAt` is the TTL-based sliding window;
   * `absoluteExpiresAt` is the record row's own wall-clock expiry
   * (`record.expiresAt * 1000`). See the module header for the full
   * cleanup/refresh path inventory.
   *
   * Invariant: a marker is only meaningful for the SPECIFIC write
   * that was live when `markHardTimedOut` was called. If a later
   * `track(id, newPromise)` reuses the same id after a marker was
   * set, the original promise's `.finally(...)` will still clear
   * the marker on its settlement (clearing the wrong state for the
   * new promise). In practice the responses endpoint scopes
   * response ids to a single persist each, so this collision cannot
   * arise.
   */
  private readonly hardTimedOut: Map<string, { expiresAt: number; ttlMs: number; absoluteExpiresAt: number }> =
    new Map();

  /**
   * Per-call visit budget for the opportunistic sweep invoked from
   * `markHardTimedOut()`. Without a cap, refresh-on-read could keep
   * the map arbitrarily large and every transition would pay O(N) on
   * the main event loop (amortized O(N^2) across N wedged writes).
   *
   * Cap of 64 makes each transition O(1) with a small constant.
   * JavaScript `Map` iterates in insertion order, so the sweep
   * naturally drains the oldest markers first — which is where
   * same-TTL expiries cluster. A backlog of K expired markers drains
   * fully across ceil(K / 64) subsequent inserts, adequate because
   * the read-path deletions are the authoritative cleanup signals
   * for ids that actually receive continuation traffic.
   *
   * NOTE: the budget is a VISIT limit, not a delete limit. We stop
   * after visiting MAX_SWEEP_PER_INSERT entries regardless of how
   * many were expired, so cost stays bounded even when none of the
   * first 64 entries are expired.
   */
  private static readonly MAX_SWEEP_PER_INSERT = 64;

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
   *
   * `earliestExpiresAtMs` is the EARLIEST wall-clock expiry (epoch-ms)
   * across the record being persisted AND every resolved ancestor in
   * its chain. When provided, it is stored in
   * `earliestExpiresByPending` so the pre-breaker `awaitPending`
   * timeout/probe path can short-circuit to 404 once
   * `Date.now() >= earliestExpiresAtMs` rather than emit retryable
   * 503 for a chain that cannot be recovered via `getChain()`.
   * `Number.isFinite(...)` guards for rows lacking explicit
   * `expiresAt`.
   */
  track(id: string, writePromise: Promise<void>, earliestExpiresAtMs?: number): void {
    this.pending.set(id, writePromise);
    if (earliestExpiresAtMs !== undefined && Number.isFinite(earliestExpiresAtMs)) {
      this.earliestExpiresByPending.set(id, earliestExpiresAtMs);
    }
    // Use `.finally` rather than `then`+`catch` so registration lifetime
    // is symmetric across fulfill/reject — a failed write should still
    // clear the tracker so subsequent chain lookups see an empty
    // getChain() result and 404 cleanly. The trailing `.catch` on the
    // returned chain only silences unhandled-rejection diagnostics on
    // this cleanup fork; the rejection is still surfaced to whoever
    // awaits `writePromise` directly.
    void writePromise
      .finally(() => {
        // Only remove if WE are still the registered entry — the id
        // may have been re-registered after this write resolved.
        if (this.pending.get(id) === writePromise) {
          this.pending.delete(id);
          // Drop the earliest-expiry side entry in lockstep so we
          // never serve a stale scalar for a freshly-registered id
          // that reuses the same string. When the write crosses the
          // hard-timeout breaker, `markHardTimedOut()` already
          // removed the pending entry synchronously, so this guard
          // is false by the time the wedged write eventually
          // settles — that case is handled authoritatively in
          // `markHardTimedOut()` itself.
          this.earliestExpiresByPending.delete(id);
        }
        // Fast path for the hard-timeout marker: clear unconditionally
        // because the marker is keyed on id (not promise reference),
        // so whoever later re-registers the id must explicitly
        // re-mark if they want the retryable window open again.
        // Under a truly wedged store this handler never fires; the
        // TTL path in `isHardTimedOut` is the slow-path bound.
        this.hardTimedOut.delete(id);
      })
      .catch(() => {
        // Terminal handler to silence unhandled-rejection warnings;
        // the real rejection is handled by the registering awaiter.
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
   * Return the EARLIEST wall-clock expiry (epoch-ms) captured
   * alongside the in-flight write for `id` at `track()` time, or
   * `undefined` if no pending write is tracked and no live marker
   * covers this id.
   *
   * Consulted by the pre-breaker `awaitPending` timeout/probe path in
   * `responses.ts` to distinguish a transient storage slowdown
   * (retryable 503) from an unrecoverable chain where the earliest
   * ancestor has already aged out (permanent 404).
   *
   * Falls back to the marker's `absoluteExpiresAt` when the pending
   * entry has already been drained by `markHardTimedOut()` — otherwise
   * a waiter that straddled the `pending -> hardTimedOut` transition
   * would see `undefined` and fall through to retryable 503 for an
   * unrecoverable chain. Both sites are fed
   * `min(recordExpiresAtMs, chainEarliestExpiresAtMs)` by
   * `initiatePersist` in `responses.ts`, so this is lossless.
   *
   * The fallback is gated on the shared `isMarkerLive` predicate so
   * a marker whose TTL or absolute expiry has already passed cannot
   * hand back a future scalar that contradicts `isHardTimedOut()`.
   * Dead markers return `0` (sentinel meaning "already expired") —
   * the consumer's `Date.now() >= earliestMs` guard always trips for
   * `0`, producing a permanent 404 instead of falling through to the
   * retryable-503 branch. This read path stays side-effect-free;
   * `sweepExpired()` is the authoritative reaper.
   */
  getEarliestExpiresAtMs(id: string): number | undefined {
    const pendingValue = this.earliestExpiresByPending.get(id);
    if (pendingValue !== undefined) return pendingValue;
    const entry = this.hardTimedOut.get(id);
    if (entry === undefined) return undefined;
    if (!PendingResponseWrites.isMarkerLive(entry, Date.now())) {
      return 0;
    }
    return entry.absoluteExpiresAt;
  }

  /**
   * Shared, side-effect-free liveness predicate for hard-timeout
   * markers: live iff `now < expiresAt` AND `now < absoluteExpiresAt`.
   *
   * Extracted so `getEarliestExpiresAtMs()` (read-only) and the
   * mutating `isHardTimedOut()` poll agree on liveness without either
   * invoking the other — `isHardTimedOut()` has refresh + move-to-tail
   * side effects only correct for the polling-side caller.
   */
  private static isMarkerLive(entry: { expiresAt: number; absoluteExpiresAt: number }, nowMs: number): boolean {
    return nowMs < entry.expiresAt && nowMs < entry.absoluteExpiresAt;
  }

  /**
   * Transition a pending entry to the hard-timed-out marker state.
   *
   * Called by the hard-timeout breaker in `responses.ts` when an
   * in-flight `store.store(...)` has crossed the hard timeout and is
   * presumed wedged. The `pending` entry is removed so `awaitPending`
   * stops handing out the stale promise and the closure chain is
   * reclaimable. The id is added to the marker map so the
   * continuation path classifies missing chains as retryable 503
   * `storage_timeout` instead of permanent 404 while the marker is
   * live.
   *
   * `ttlMs` caps the marker lifetime independently of whether the
   * underlying write settles. The caller (`responses.ts`) reads it
   * from `MLX_HARD_TIMEOUT_MARKER_TTL_MS`; passing it in keeps this
   * module env-free.
   *
   * `absoluteExpiresAt` is the response record's row expiry
   * (`record.expiresAt * 1000`). The initial expiry is
   * `min(Date.now() + ttlMs, absoluteExpiresAt)` so a short-lived
   * record cannot have its marker outlive its row.
   * `isHardTimedOut()` also consults `absoluteExpiresAt` on every
   * read and hard-stops at that bound regardless of refreshes.
   *
   * Returns true if the id was an active pending entry and was moved
   * to the marker; false if no pending entry existed at call time.
   * A false return does NOT add the id to the marker — a marker
   * without a backing promise has no fast cleanup signal (beyond
   * TTL / absolute cap) and would produce spurious retryable-503
   * signals in the meantime if the caller mis-routes ids.
   */
  markHardTimedOut(id: string, ttlMs: number, absoluteExpiresAt: number): boolean {
    // Drain expired entries on the write path so bounded memory does
    // not depend on any continuation ever reading this map. Runs
    // BEFORE the insert so the new entry (whose `expiresAt` is in
    // the future by construction) is not considered for expiry.
    this.sweepExpired();
    const wasPending = this.pending.delete(id);
    // Drain the earliest-expiry side map in lockstep with the
    // pending delete above. The `.finally(...)` cleanup inside
    // `track()` guards on `pending.get(id) === writePromise` — false
    // once we remove the pending entry, and for never-settling
    // writes the `.finally` never fires at all. Without this
    // authoritative drain the side map would grow unboundedly under
    // a wedged store.
    this.earliestExpiresByPending.delete(id);
    if (wasPending) {
      // Clamp initial expiry at the row's absolute expiry so we
      // never return retryable-503 for a window past the point
      // where the row could be recovered.
      const expiresAt = Math.min(Date.now() + ttlMs, absoluteExpiresAt);
      this.hardTimedOut.set(id, { expiresAt, ttlMs, absoluteExpiresAt });
    }
    return wasPending;
  }

  /**
   * Drain expired marker entries (`expiresAt <= now` or
   * `absoluteExpiresAt <= now`). Visits at most
   * `MAX_SWEEP_PER_INSERT` entries per call so a caller cannot
   * trigger an unbounded linear walk. `Map` insertion order makes
   * this drain the oldest (most likely expired) markers first. The
   * budget is a VISIT limit, not a delete limit.
   */
  private sweepExpired(): void {
    const now = Date.now();
    let visited = 0;
    for (const [id, entry] of this.hardTimedOut) {
      if (visited >= PendingResponseWrites.MAX_SWEEP_PER_INSERT) break;
      visited += 1;
      if (entry.absoluteExpiresAt <= now || entry.expiresAt <= now) {
        this.hardTimedOut.delete(id);
      }
    }
  }

  /**
   * Whether `id` is currently flagged as hard-timed-out. Used by the
   * `previous_response_id` continuation path to classify a missing
   * chain as retryable 503 `storage_timeout` vs. permanent 404.
   *
   * Read-path cleanup + refresh semantics:
   *
   *   - Absolute cap is authoritative: once `now >= absoluteExpiresAt`
   *     the marker is deleted unconditionally. `ResponseStore.getChain()`
   *     hides the row past its own row TTL, so retryable-503 would
   *     lie to the client.
   *   - TTL-expired: lazy delete + return false.
   *   - Live hit: refresh `expiresAt = min(now + ttlMs, absoluteExpiresAt)`
   *     so actively-retried chains stay recoverable while the write
   *     might still land, without ever outliving the row.
   *   - On every live hit, move the entry to the Map tail (O(1)
   *     `delete` + `set` using insertion-order semantics). Without
   *     the rotation a stable head cohort of hot refreshed entries
   *     could indefinitely block the bounded `sweepExpired()` from
   *     reaching expired markers behind them. LRU rotation lets the
   *     sweep make forward progress.
   */
  isHardTimedOut(id: string): boolean {
    const entry = this.hardTimedOut.get(id);
    if (entry === undefined) return false;
    const now = Date.now();
    if (now >= entry.absoluteExpiresAt) {
      this.hardTimedOut.delete(id);
      return false;
    }
    if (entry.expiresAt <= now) {
      this.hardTimedOut.delete(id);
      return false;
    }
    entry.expiresAt = Math.min(now + entry.ttlMs, entry.absoluteExpiresAt);
    // Move refreshed entry to the tail so the bounded sweep can
    // progress past actively-refreshed ids. `set` on an existing
    // key preserves the entry reference, so no copy is made.
    this.hardTimedOut.delete(id);
    this.hardTimedOut.set(id, entry);
    return true;
  }

  /** Number of writes currently in flight. Primarily for tests. */
  get size(): number {
    return this.pending.size;
  }

  /**
   * Number of ids currently in the hard-timed-out marker state.
   * Primarily for tests. Delegates to the shared `sweepExpired()`
   * helper so read-count and write-sweep stay in lockstep.
   *
   * Caveat: because the sweep is bounded (`MAX_SWEEP_PER_INSERT`
   * visits per call), the reported size may include still-present
   * expired entries that sit past the per-call visit budget.
   * Callers needing exact reclaimed-count semantics should drive
   * further `markHardTimedOut()` inserts (each drains another
   * batch) or call `isHardTimedOut(id)` directly — the read-path
   * deletion is authoritative and unbounded per-id.
   */
  get hardTimedOutSize(): number {
    this.sweepExpired();
    return this.hardTimedOut.size;
  }

  /**
   * Number of ids currently holding a scalar entry in the
   * pending-side earliest-expiry map. Primarily for tests —
   * regressions that need to validate the pending-side map is
   * drained (independent of the marker-map fallback in
   * `getEarliestExpiresAtMs`) require a direct readout.
   */
  get earliestExpiresByPendingSize(): number {
    return this.earliestExpiresByPending.size;
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
