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
 * ## Hard-timeout marker state (iter-50 / iter-51 / iter-52 / iter-53)
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
 * Iter-51 introduced an independent TTL with lazy expiry. Markers are
 * stored as `Map<string, { expiresAt, ttlMs }>`. `markHardTimedOut(id,
 * ttlMs)` computes `expiresAt = Date.now() + ttlMs` at call time;
 * `isHardTimedOut(id)` treats any entry whose `expiresAt <=
 * Date.now()` as absent and deletes it on the way out. The caller
 * (`responses.ts`) reads the TTL from `MLX_HARD_TIMEOUT_MARKER_TTL_MS`
 * (default 300_000ms) and passes it in explicitly so this module
 * stays env-free.
 *
 * Iter-52 (codex's iter-51 two HIGH findings):
 *
 *   * Finding 1 — expired markers only drained on continuation reads.
 *     An id that was hard-timed-out and then never retried by the
 *     client still occupied one entry until something eventually
 *     called `isHardTimedOut(id)` or `hardTimedOutSize`. Under a
 *     wedged backend serving new top-level requests (no continuations
 *     at all) this leaked one marker per hard-timed-out write
 *     indefinitely. Fix: `markHardTimedOut()` now opportunistically
 *     sweeps every expired entry on the map BEFORE inserting the new
 *     one, via `sweepExpired()`. Steady-state memory is now bounded
 *     even when no continuation traffic ever arrives.
 *
 *   * Finding 2 — the fixed TTL turned unresolved writes into
 *     permanent 404s. Once the 300s TTL elapsed, the retryable-503
 *     classification flipped to 404 even though the underlying
 *     `store.store(...)` was still running and might still land.
 *     A backend stall longer than the default TTL became an
 *     irreversible chain break. Fix: `isHardTimedOut(id)` now
 *     REFRESHES `entry.expiresAt = Date.now() + entry.ttlMs` on every
 *     live hit. As long as the client keeps retrying the
 *     continuation, the retryable window slides forward and the
 *     chain remains recoverable if the write eventually lands.
 *     Only genuinely abandoned markers (no continuation traffic AND
 *     no eventual settlement) age out. Each entry stores its
 *     original `ttlMs` so the refresh knows the interval.
 *
 * Iter-53 (codex's iter-52 HIGH finding 1 + MEDIUM finding 2):
 *
 *   * Finding 1 — the iter-52 refresh-on-read extended the marker
 *     expiry unboundedly as long as clients kept retrying, even
 *     though the response record itself expires at a FIXED absolute
 *     wall-clock bound (`RESPONSE_TTL_SECONDS`, currently 30 min).
 *     After that absolute bound passes, `ResponseStore.getChain()`
 *     hides the row — no late persist can ever become visible — but
 *     the refresh-on-read kept classifying the id as retryable
 *     `storage_timeout` indefinitely as long as the client kept
 *     pinging before each marker TTL elapsed. Result: permanent 503
 *     loop for an unrecoverable chain. Fix: the caller threads in
 *     the record's `absoluteExpiresAt` (epoch-ms), and both the
 *     initial insert and every read-refresh clamp to that value.
 *     Once `Date.now() >= absoluteExpiresAt`, `isHardTimedOut(id)`
 *     deletes the marker and returns `false` unconditionally — the
 *     row is unrecoverable regardless of retries.
 *
 *   * Finding 2 — `sweepExpired()` was an UNBOUNDED linear walk of
 *     the full marker map on every new hard-timeout insert. With
 *     iter-52's refresh-on-read keeping actively-retried ids alive
 *     for arbitrarily long, the map could grow to N wedged writes
 *     and every new transition paid O(N) on the main event loop,
 *     amortized O(N²) across N inserts. Fix: a bounded per-insert
 *     visit budget (`MAX_SWEEP_PER_INSERT = 64`). The sweep
 *     iterates at most 64 entries per call and deletes expired
 *     entries it finds. JavaScript `Map` preserves insertion
 *     order, so the sweep naturally drains the oldest markers
 *     first — which is where same-TTL expiries cluster. Steady-
 *     state cost per transition is now O(1) and memory is still
 *     reclaimed amortizedly (full drain across roughly ceil(N/64)
 *     subsequent inserts).
 *
 * Iter-54 (codex's iter-53 MEDIUM finding 2):
 *
 *   * `sweepExpired()` always starts from the Map head and stops
 *     after MAX_SWEEP_PER_INSERT visits, but iter-52's
 *     refresh-on-read updated entries in place — preserving their
 *     original insertion position at the head. A fixed cohort of
 *     64 actively-refreshed markers at the head could indefinitely
 *     block every subsequent sweep from reaching expired markers
 *     behind them, starving reclamation. Fix: `isHardTimedOut(id)`
 *     now moves refreshed entries to the tail (O(1) `delete` +
 *     re-`set` leveraging Map insertion-order semantics). The
 *     sweep then makes forward progress across the whole map as
 *     hot entries rotate back, producing natural LRU-eviction
 *     behaviour under sustained hard-timeout traffic.
 *
 *   * Iter-54 also clamps the marker's `absoluteExpiresAt` at the
 *     MINIMUM `expiresAt` across the whole resolved chain, not
 *     just the child record. Iter-55 refactored the caller-side
 *     helper into an inline scalar (`chainEarliestExpiresAtMs`)
 *     computed once when `getChain()` resolves, and threads that
 *     scalar through both `track()` and `markHardTimedOut()`.
 *     `ResponseStore.getChain()` walks ancestors and aborts on the
 *     first expired link
 *     (`crates/mlx-db/src/response_store/reader.rs:44-59`), so a
 *     child whose parent expires sooner is unrecoverable at the
 *     parent's expiry — not the child's. That change lives on the
 *     caller side; this module just receives the min-clamped value.
 *
 * Cleanup paths in effect after iter-54:
 *
 *   1. Fast path — the underlying write promise's `.finally(...)`
 *      inside `track()` deletes the marker as soon as the wedged
 *      store unwedges.
 *   2. Read-refresh path — live continuations slide the TTL
 *      forward so the retryable-503 window stays open while the
 *      client is actively retrying, CLAMPED at the record's
 *      absolute row expiry (iter-53). The refreshed entry is
 *      MOVED TO THE MAP TAIL on every live hit (iter-54) so the
 *      bounded sweep is not starved by a stable head cohort of
 *      hot entries.
 *   3. Read-expire path — a full TTL elapse without any refresh
 *      lazily deletes the entry and returns `false` (permanent
 *      404 classification).
 *   4. Read-absolute-cap path (iter-53) — once `Date.now() >=
 *      absoluteExpiresAt`, the marker is deleted and `false` is
 *      returned regardless of TTL refreshes. Hard stop for
 *      unrecoverable chains.
 *   5. Write-sweep path — `markHardTimedOut()` drains expired
 *      entries on insert, BOUNDED to MAX_SWEEP_PER_INSERT visits
 *      per call (iter-53). Combined with iter-54's read-move,
 *      the sweep now always makes forward progress — stale
 *      head-of-map entries are reclaimed even under sustained
 *      retry traffic against a fixed hot cohort.
 *
 * Marker lifetime is therefore bounded by:
 *
 *   min(write settlement, last continuation + TTL, absoluteExpiresAt)
 *
 * ## Pending-entry earliest-expiry metadata (iter-55)
 *
 * Iter-54 (codex's iter-54 HIGH finding 1) observed that the
 * chain-earliest cap lives inside the hard-timeout branch — the
 * pre-breaker `awaitPending` timeout/probe path in `responses.ts`
 * was still blindly classifying any unresolved pending write as
 * retryable `storage_timeout`, even when the resolved chain's
 * earliest ancestor had already expired. A continuation whose
 * parent is already past its row TTL cannot ever succeed via
 * `getChain()`, so looping the client on 503 for up to
 * `MLX_POST_COMMIT_PERSIST_HARD_TIMEOUT_MS` (60s default) before
 * the hard breaker converts the entry to a marker was wasted.
 *
 * Iter-55 stores the earliest recoverable expiry ALONGSIDE each
 * tracked promise in a separate per-id side map
 * (`earliestExpiresByPending`). `track(id, promise,
 * earliestExpiresAtMs?)` records the scalar at registration
 * time; `getEarliestExpiresAtMs(id)` exposes it to the pre-
 * breaker `awaitPending` path so the endpoint can short-circuit
 * to 404 once `Date.now() >= earliestExpiresAtMs` rather than
 * keep emitting retryable 503. The side map is keyed identically
 * to `pending` and cleared on the same `.finally(...)` hook, so
 * no extra lifecycle surface is added — only the scalar we need
 * to consult during the timeout branch.
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
   * Iter-55: per-pending-entry scalar `earliestExpiresAtMs`
   * recording the EARLIEST recoverable wall-clock expiry across
   * (record + resolved ancestor chain) at the time `track()` was
   * called. Keyed identically to `pending` and cleared on the same
   * `.finally(...)` settlement hook so no extra lifecycle surface
   * is introduced.
   *
   * The pre-breaker `awaitPending` timeout/probe path in
   * `responses.ts` consults this via `getEarliestExpiresAtMs(id)`:
   * once `Date.now()` has passed the earliest ancestor expiry,
   * `getChain()` can never succeed, so the continuation short-
   * circuits to 404 rather than looping the client on retryable
   * 503 for up to the hard-breaker window.
   *
   * Optional: legacy calls that pass `undefined` (e.g. tests or
   * pre-iter-55 helpers) simply do not populate this side map;
   * `getEarliestExpiresAtMs(id)` returns `undefined` and the caller
   * falls back to the pre-iter-55 behaviour (emit retryable 503).
   */
  private readonly earliestExpiresByPending: Map<string, number> = new Map();

  /**
   * Ids that crossed the hard-timeout breaker in responses.ts while
   * their `store.store(...)` promise was still unresolved (iter-50,
   * TTL-bounded in iter-51, sweep-on-write + refresh-on-read in
   * iter-52, absolute-cap + bounded-sweep in iter-53).
   *
   * Each entry records `{ expiresAt, ttlMs, absoluteExpiresAt }` in
   * epoch-ms. `expiresAt` is the TTL-based sliding window;
   * `absoluteExpiresAt` is the record row's own wall-clock expiry
   * (`record.expiresAt * 1000`). Five cleanup/refresh paths:
   *
   *   1. Fast path — the underlying write promise's `.finally(...)`
   *      inside `track()` deletes the entry as soon as the wedged
   *      store unwedges. Observable latency: as fast as the write
   *      itself settles.
   *   2. Read-refresh path (iter-52) — `isHardTimedOut(id)` pushes
   *      `expiresAt` forward by the original `ttlMs` on every live
   *      hit, CLAMPED at `absoluteExpiresAt` (iter-53).
   *      Continuations that keep retrying keep the retryable-503
   *      window open while the row is still visible to
   *      `ResponseStore.getChain()`.
   *   3. Read-expire path — `isHardTimedOut(id)` lazily deletes any
   *      entry whose `expiresAt` has already passed at call time
   *      (no refresh on a dead entry). Returns false.
   *   4. Read-absolute-cap path (iter-53) — once
   *      `Date.now() >= absoluteExpiresAt`, the entry is deleted
   *      and false is returned unconditionally. Hard stop: the
   *      row has aged out of `getChain()`, so no late persist
   *      can make this id chainable again.
   *   5. Write-sweep path (iter-52, bounded in iter-53) —
   *      `markHardTimedOut()` calls `sweepExpired()` BEFORE
   *      inserting a new marker, but visits at most
   *      `MAX_SWEEP_PER_INSERT` entries so every transition is
   *      O(1) amortized even under sustained wedged-store
   *      traffic.
   *
   * Invariant: a marker is only meaningful for the SPECIFIC write
   * that was live when `markHardTimedOut` was called. If a later
   * `track(id, newPromise)` re-uses the same id after a marker was
   * set, the original promise's `.finally(...)` will still clear the
   * marker on its settlement (clearing the wrong state for the new
   * promise). Callers that mix hard-timed-out ids with brand-new
   * writes under the same id MUST call
   * `markHardTimedOut(id, ttlMs, absoluteExpiresAt)` again after
   * the new `track(id, ...)` if they want the marker to persist —
   * in practice the responses endpoint scopes response ids to a
   * single persist each, so this collision cannot arise.
   */
  private readonly hardTimedOut: Map<string, { expiresAt: number; ttlMs: number; absoluteExpiresAt: number }> =
    new Map();

  /**
   * Iter-53 finding 2 fix: per-call budget for the opportunistic
   * sweep invoked from `markHardTimedOut()`. Iter-52 walked the
   * full marker map on every insert; with iter-52's refresh-on-read
   * keeping actively-retried ids alive for arbitrarily long, that
   * was an O(N) operation on the main event loop per transition,
   * amortized O(N²) across N wedged writes.
   *
   * Capping the per-insert visit count at 64 makes each transition
   * O(1) with a small constant. JavaScript `Map` iterates in
   * insertion order, so the sweep naturally drains the oldest
   * markers first — which is where same-TTL expiries cluster. A
   * backlog of K expired markers drains fully across ceil(K / 64)
   * subsequent inserts, which is adequate because the marker map
   * is a best-effort reclamation layer; the read-path deletions
   * (paths 2-4 above) are the authoritative cleanup signals for
   * ids that actually receive continuation traffic.
   *
   * NOTE: the budget is a VISIT limit, not a delete limit. We stop
   * after visiting MAX_SWEEP_PER_INSERT entries regardless of how
   * many of them were expired, so the cost stays bounded even in
   * the degenerate case where none of the first 64 entries are
   * expired.
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
   * Iter-55: `earliestExpiresAtMs` is the EARLIEST wall-clock expiry
   * (epoch-ms) across the record being persisted AND every resolved
   * ancestor in its chain. When provided, it is stored in a per-id
   * side map (`earliestExpiresByPending`) so the pre-breaker
   * `awaitPending` timeout/probe path can short-circuit to 404 once
   * `Date.now() >= earliestExpiresAtMs` rather than keep emitting
   * retryable 503 for a chain that cannot ever be recovered via
   * `getChain()`. The argument is optional so legacy callers (tests,
   * pre-iter-55 helpers) remain compatible; an unset value leaves
   * the pre-iter-55 behaviour in place.
   *
   * `Number.isFinite(...)` guards for rows lacking explicit
   * `expiresAt` — in that case the caller passes `undefined` and
   * no entry is added to the side map.
   */
  track(id: string, writePromise: Promise<void>, earliestExpiresAtMs?: number): void {
    this.pending.set(id, writePromise);
    if (earliestExpiresAtMs !== undefined && Number.isFinite(earliestExpiresAtMs)) {
      this.earliestExpiresByPending.set(id, earliestExpiresAtMs);
    }
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
          // Iter-55: drop the scalar earliest-expiry side entry in
          // lockstep with the pending promise so we never serve a
          // stale earliest-expiry value for a freshly-registered
          // id that happens to reuse the same string.
          this.earliestExpiresByPending.delete(id);
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
   * Iter-55: return the scalar EARLIEST wall-clock expiry (epoch-ms)
   * that was captured alongside the in-flight write for `id` at
   * `track()` time, or `undefined` if either no pending write is
   * tracked under `id` or the caller did not pass an
   * `earliestExpiresAtMs` argument.
   *
   * Consulted by the pre-breaker `awaitPending` timeout/probe path
   * in `responses.ts` to distinguish a transient storage slowdown
   * (retryable 503) from an unrecoverable chain where the earliest
   * ancestor has already aged out (permanent 404).
   *
   * The returned value is the caller-provided scalar; this module
   * does not interpret it beyond storing it under the entry's
   * lifetime. `ResponseStore.getChain()` evaluates row expiries
   * against its own wall-clock at call time, so the scalar reflects
   * the state observed when the write was initiated, which is the
   * latest honest snapshot the pre-breaker path can check against.
   */
  getEarliestExpiresAtMs(id: string): number | undefined {
    return this.earliestExpiresByPending.get(id);
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
   * Iter-52: this method also calls `sweepExpired()` BEFORE the
   * insert so memory reclamation does not depend on a later
   * continuation read. Under a wedged store serving new top-level
   * traffic (without any retries of old response ids), every new
   * hard-timeout event drains the expired tail of the map. The
   * per-entry `ttlMs` field is persisted alongside `expiresAt` so
   * `isHardTimedOut()` can refresh the window without the caller
   * re-threading the interval.
   *
   * Iter-53 finding 1 fix: the caller also threads in
   * `absoluteExpiresAt` — the response record's row expiry
   * (`record.expiresAt * 1000`). On insert we compute
   * `expiresAt = min(Date.now() + ttlMs, absoluteExpiresAt)` so a
   * short-lived record cannot have its marker outlive its row.
   * `isHardTimedOut()` also consults `absoluteExpiresAt` on every
   * read and hard-stops at that bound regardless of refreshes — once
   * `ResponseStore.getChain()` will hide the row, the retryable
   * `storage_timeout` classification is factually wrong and must
   * flip to 404.
   *
   * Iter-53 finding 2 fix: `sweepExpired()` is now bounded to
   * `MAX_SWEEP_PER_INSERT` visited entries per call so this
   * transition stays O(1) amortized even when the marker map is
   * large (e.g. actively-retried ids that iter-52's refresh-on-read
   * keeps alive).
   *
   * Returns true if the id was an active pending entry (and has
   * been moved into the hard-timed-out marker), false if no pending
   * entry existed at call time. A false return does NOT add the id
   * to the marker — a marker without a backing promise has no
   * cleanup signal (beyond the TTL and absolute cap) and could leak
   * if the caller mis-routes ids. The TTL and absolute cap would
   * eventually drain it, but in the meantime continuations would
   * see a spurious retryable-503 signal.
   */
  markHardTimedOut(id: string, ttlMs: number, absoluteExpiresAt: number): boolean {
    // Iter-52 finding 1 fix: drain expired entries on the write path
    // so bounded memory does not depend on any continuation ever
    // reading this map. Runs BEFORE the insert so the new entry is
    // not considered for expiry (its `expiresAt` is in the future by
    // construction). Iter-53 bounds this sweep's cost — see
    // `sweepExpired()`.
    this.sweepExpired();
    const wasPending = this.pending.delete(id);
    if (wasPending) {
      // Iter-53 finding 1 fix: clamp initial expiry at the record's
      // absolute row expiry. If the row will disappear from
      // `getChain()` before `now + ttlMs`, shrink the marker to
      // match so we never return retryable-503 for a window past
      // the point where the row could be recovered.
      const expiresAt = Math.min(Date.now() + ttlMs, absoluteExpiresAt);
      this.hardTimedOut.set(id, { expiresAt, ttlMs, absoluteExpiresAt });
    }
    return wasPending;
  }

  /**
   * Drain expired marker entries (entries whose `expiresAt` is at
   * or before `Date.now()` OR whose `absoluteExpiresAt` is at or
   * before `Date.now()`).
   *
   * Iter-53 finding 2 fix: visits at most
   * `MAX_SWEEP_PER_INSERT` entries per call so a caller cannot
   * trigger an unbounded linear walk. JavaScript `Map` iterates in
   * insertion order, so the sweep naturally drains the oldest
   * markers first — which is where same-TTL expiries cluster. The
   * budget is a VISIT limit, not a delete limit (we count each
   * entry we examine, not just the ones we delete).
   *
   * Private helper used by `markHardTimedOut()` and
   * `hardTimedOutSize` — see the class-level docstring for the full
   * cleanup-path inventory.
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
   * Iter-51: lazily expires the marker on read. If an entry's
   * `expiresAt` has already passed, it is deleted and the method
   * returns false. This is the read-expire cleanup path — it bounds
   * steady-state marker memory even if the underlying write promise
   * never settles (paired with the write-sweep path in iter-52).
   *
   * Iter-52 finding 2 fix: on a live hit (entry present AND
   * `expiresAt > Date.now()`) this method REFRESHES the expiry by
   * pushing `expiresAt = Date.now() + entry.ttlMs` before returning
   * true. Semantic: as long as the client keeps retrying the
   * continuation, the retryable-503 window slides forward and the
   * chain stays recoverable if the underlying write eventually
   * lands. Only genuinely abandoned markers (no read traffic in a
   * full TTL window AND no eventual settlement) age out. Without
   * this refresh a slow-but-eventual persist that crossed the fixed
   * 300s default TTL would flip to a permanent 404 even though the
   * write was still running — an irreversible chain break.
   *
   * Iter-53 finding 1 fix: on every read we first check the
   * ABSOLUTE cap (`entry.absoluteExpiresAt` — the record row's
   * wall-clock expiry). If that has passed, the marker is deleted
   * and false is returned unconditionally — further refreshes would
   * extend the retryable-503 window into a period where
   * `ResponseStore.getChain()` hides the row, so the classification
   * must flip to 404 at that bound. Every refresh is also clamped
   * at the absolute cap so the marker can never outlive the row.
   *
   * Iter-54 (codex's iter-53 MEDIUM finding 2): move refreshed
   * live entries to the Map tail so the bounded write-path sweep
   * can progress past them on subsequent `markHardTimedOut()`
   * calls. Before iter-54, `sweepExpired()` always started from
   * the Map head and stopped after MAX_SWEEP_PER_INSERT visits;
   * because `isHardTimedOut()` refreshed entries in place, the
   * oldest-by-insertion hot entries stayed at the head forever
   * and the sweep could be starved — a fixed set of 64 live-but-
   * refreshed entries at the head would block every subsequent
   * sweep from reaching expired entries behind them. JavaScript
   * `Map` preserves insertion order; `delete` + re-`set` is O(1)
   * and re-inserts at the tail. Natural LRU-eviction semantics:
   * hot entries rotate to the tail as they are refreshed,
   * leaving stale entries at the head available for reclamation.
   */
  isHardTimedOut(id: string): boolean {
    const entry = this.hardTimedOut.get(id);
    if (entry === undefined) return false;
    const now = Date.now();
    // Iter-53 finding 1 fix: absolute cap is authoritative. Once
    // the row's own TTL has passed, refreshing the marker would
    // lie to the client (retryable 503 for an unrecoverable chain).
    // Delete unconditionally and let the continuation path fall
    // through to the real 404.
    if (now >= entry.absoluteExpiresAt) {
      this.hardTimedOut.delete(id);
      return false;
    }
    if (entry.expiresAt <= now) {
      this.hardTimedOut.delete(id);
      return false;
    }
    // Iter-52 finding 2 fix + iter-53 finding 1 clamp: refresh the
    // TTL-based expiry on every live hit so active continuations
    // slide the retryable window forward, but clamp at the
    // absolute cap so the marker can never outlive the row that
    // backs it. The original `ttlMs` and `absoluteExpiresAt` were
    // both persisted on the entry at insertion time so the caller
    // doesn't need to re-thread either one here.
    entry.expiresAt = Math.min(now + entry.ttlMs, entry.absoluteExpiresAt);
    // Iter-54 finding 2 fix: move this refreshed entry to the
    // tail of the insertion-ordered Map so the bounded sweep
    // (which always starts from the head) can make progress past
    // actively-refreshed ids. O(1) — Map `delete` + `set` is
    // constant-time. `set` on an existing key with an existing
    // entry preserves the entry reference (we set the SAME
    // entry object, which is already mutated with the refreshed
    // `expiresAt` above), so no copy is made.
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
   *
   * Iter-52: delegates to the shared `sweepExpired()` helper so
   * read-count and write-sweep stay in lockstep.
   *
   * Iter-53 caveat: because the sweep is now bounded
   * (`MAX_SWEEP_PER_INSERT` visits per call), the reported size may
   * include still-live entries whose `expiresAt` has passed but
   * that sit past the per-call visit budget. Callers that need
   * exact reclaimed-count semantics should drive subsequent
   * `markHardTimedOut()` inserts (each of which drains another
   * batch) or call `isHardTimedOut(id)` directly on the ids of
   * interest — the read-path deletion is authoritative and
   * unbounded per-id.
   */
  get hardTimedOutSize(): number {
    this.sweepExpired();
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
