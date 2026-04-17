/**
 * SessionRegistry -- per-model cache holding AT MOST one live
 * `ChatSession` whose native KV state is currently valid.
 *
 * Design notes:
 *
 *   - **One registry per model.** Composed alongside each registered
 *     `ServableModel` in `ModelRegistry`. Sessions are keyed purely
 *     by response id — no secondary keying on model name because the
 *     registry is already scoped per model.
 *
 *   - **Single-warm-session invariant.** `ChatSession<M>` is a thin
 *     JS wrapper — it does NOT own any native KV cache. The cache
 *     lives on the underlying `SessionCapableModel` (one shared
 *     `cached_token_history` / `caches` vector per model instance).
 *     Any call that runs a turn overwrites that shared native state,
 *     silently invalidating every other `ChatSession` wrapper
 *     pointing at the same model. Caching multiple wrappers per
 *     model is therefore an illusion: at most ONE matches real
 *     native state (whichever ran most recently). To prevent
 *     cross-session corruption this registry holds at most ONE
 *     entry — both `getOrCreate` and `adopt` clear the map before
 *     returning or inserting.
 *
 *   - **Lease semantics on hit.** Clear-on-hit also gives single-
 *     flight lease semantics: two overlapping requests referencing
 *     the same `previous_response_id` cannot share the same live
 *     `ChatSession`. The first wins the cleared entry; the second
 *     finds the map empty and cold-replays from `ResponseStore` on
 *     a fresh session. Without this, the second would hit
 *     `ChatSession`'s single-flight "concurrent send() not allowed"
 *     guard.
 *
 *   - **Instructions / prefix-state change also misses.** Each entry
 *     records the `instructions` string used to adopt it.
 *     `getOrCreate` compares the caller's `requestedInstructions`
 *     against the cached value; mismatch forces cold replay so the
 *     new prefix state is re-primed instead of silently reusing a
 *     stale warmed prompt. The OpenAI `instructions` field and the
 *     Anthropic `system` field both flow through the same parameter
 *     — the registry does not care which is which.
 *
 *   - **Cache miss fallback.** On a miss (eviction, interleaved turn
 *     on a different chain, restart, lease-on-hit) the endpoint
 *     layer reconstructs the conversation from the `ResponseStore`
 *     history, primes a fresh `ChatSession` via `primeHistory()`,
 *     and resumes through `startFromHistory()` /
 *     `startFromHistoryStream()`. That pair dispatches one
 *     `chatSessionStart*` call that rebuilds the full KV cache and
 *     atomically appends the new user turn, so cold replay is
 *     indistinguishable from a hot hit.
 *
 *   - **TTL.** Default 1800 seconds mirrors `RESPONSE_TTL_SECONDS`
 *     in `packages/server/src/endpoints/responses.ts` so the cached
 *     entry ages out alongside its stored response metadata. With
 *     at most one entry there is no LRU bookkeeping — just a single
 *     expiry check on lookup.
 *
 *   - **Thread safety.** Node.js is single-threaded within one
 *     event-loop tick, so the internal `Map` is safe against
 *     concurrent mutation by design. `sweep()` can be scheduled
 *     via `setInterval` without colliding with in-flight calls.
 *
 *   - **Per-model execution mutex.** A dispatch that spans multiple
 *     awaits (map -> prefill -> decode -> persist -> adopt) is NOT
 *     atomic from the registry's POV. Two requests against the
 *     same model would both receive a `ChatSession` pointing at
 *     the same native model; even though the lease-on-hit clear
 *     prevents sharing one `ChatSession` object, the native KV
 *     cache is a single mutable resource and two parallel
 *     `primeHistory()` / `send*()` calls would race. Whichever
 *     finished last would win `adopt()`, poisoning the hot path
 *     for every subsequent chained turn.
 *
 *     `withExclusive(fn)` serializes every per-model dispatch via
 *     a FIFO `execLock` chain. `/v1/responses` and `/v1/messages`
 *     wrap the full `getOrCreate -> run -> adopt/drop` span in one
 *     `withExclusive` so at most one request holds the model at a
 *     time. A weaker epoch-token scheme would let the losing
 *     `adopt()` no-op but the native KV would already be wrong.
 */

import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';

/** Constructor options for {@link SessionRegistry}. */
export interface SessionRegistryOptions {
  /** The model that every session in this registry wraps. Single-model per registry. */
  model: SessionCapableModel;
  /** TTL in seconds before an unused session is evicted. Default: 1800 (30 min). */
  ttlSec?: number;
  /**
   * Maximum number of requests that may be WAITING for the
   * per-model execution mutex at the same time (the in-flight holder
   * does NOT count toward this). When set and the cap is exceeded at
   * `withExclusive` entry, the call throws {@link QueueFullError}
   * synchronously so the endpoint layer can emit HTTP 429 and the
   * client can retry later.
   *
   * Default: `undefined` (unbounded — current behaviour). Opt-in per
   * {@link ServerConfig.maxQueueDepthPerModel} or the
   * `MLX_MAX_QUEUE_DEPTH_PER_MODEL` env var.
   */
  maxQueueDepth?: number;
}

/**
 * Thrown synchronously by {@link SessionRegistry.withExclusive} when
 * the per-model queue cap (`maxQueueDepth`) is exceeded. The error is
 * raised BEFORE awaiting the previous lock holder so endpoint handlers
 * can reliably catch it without racing the chain.
 */
export class QueueFullError extends Error {
  readonly queuedCount: number;
  readonly limit: number;

  constructor(queuedCount: number, limit: number) {
    super(`Model queue full: ${queuedCount} waiting (limit ${limit})`);
    this.name = 'QueueFullError';
    this.queuedCount = queuedCount;
    this.limit = limit;
  }
}

/**
 * Result of {@link SessionRegistry.getOrCreate}. `hit` reflects whether
 * the call consumed a live warm entry (single-use lease) or returned a
 * fresh `ChatSession` on a miss. The endpoint layer uses `hit` to
 * classify the per-request session-cache status emitted to clients via
 * the `X-Session-Cache` observability header.
 */
export interface SessionLookupResult {
  session: ChatSession<SessionCapableModel>;
  hit: boolean;
}

interface SessionEntry {
  session: ChatSession<SessionCapableModel>;
  /**
   * The `instructions` / `system` string the caller adopted this
   * session with. `null` if the caller did not supply any. Compared
   * byte-for-byte against the caller's `requestedInstructions` in
   * `getOrCreate` to detect prefix/system-state changes that would
   * otherwise let a hit silently reuse a stale warmed prompt.
   */
  instructions: string | null;
  /** Unix seconds at which this entry becomes eligible for eviction. */
  expiresAt: number;
}

/** Current time in unix seconds. Kept as a helper so tests can patch `Date.now` via fake timers. */
function nowSec(): number {
  return Math.floor(Date.now() / 1000);
}

export class SessionRegistry {
  private readonly model: SessionCapableModel;
  private readonly ttlSec: number;
  private readonly maxQueueDepth: number | undefined;
  /**
   * Number of callers that are currently WAITING for the per-model
   * execution mutex — i.e. have entered `withExclusive` but have not
   * yet started running their closure. The caller that is actively
   * running inside `fn()` is NOT counted here, so a cap of
   * `maxQueueDepth = N` means "1 running + up to N waiting".
   *
   * Mutated strictly inside `withExclusive`: the admitting caller is
   * counted as a waiter ONLY when the execution chain is already
   * non-idle (i.e. some earlier caller still holds the mutex). The
   * first caller into an idle chain is admitted directly as the
   * runner slot and never contributes to `queuedCount`. Waiters
   * decrement exactly once as they transition from waiting to
   * running (after `await prev`). The counter is intentionally
   * NEVER touched on cap-reject paths (the caller never queued) so
   * the cap check is stable across concurrent entries, and runner-
   * slot admissions leave it alone so a synchronous burst
   * (e.g. `Promise.all([fn, fn])`) does not spuriously bill the
   * runner-slot caller against the waiter cap.
   */
  private queuedCount = 0;
  /**
   * Holds AT MOST ONE entry under the single-warm invariant (see the
   * module-level rustdoc). `getOrCreate` and `adopt` both clear the
   * map as part of their contract so a later lookup cannot hand out
   * a wrapper whose assumed native state has been overwritten by a
   * turn on another cached entry.
   */
  private readonly entries: Map<string, SessionEntry> = new Map();
  /**
   * Shared sentinel representing "the execution chain is idle" — a
   * pre-resolved promise. `execLock` starts at this value and is
   * reset to it whenever the last holder releases without a
   * successor chained behind it. `withExclusive` uses reference
   * equality against this sentinel (`execLock === initialLock`) to
   * tell "I am the runner slot on an idle chain" apart from "I am a
   * waiter behind someone else", which is how the burst
   * (`Promise.all([fn, fn])`) admission bug is avoided.
   */
  private readonly initialLock: Promise<void> = Promise.resolve();
  /**
   * Tail of the per-model execution FIFO. Every `withExclusive` call
   * captures this value as its predecessor, then overwrites it with
   * its own pending promise so the next waiter chains after it. The
   * chain is resolved only when the current holder's `fn` has
   * settled (success or failure), guaranteeing that at most one
   * dispatch runs through this registry's native model at a time.
   * Initialized to `initialLock` so the first caller proceeds
   * without waiting AND is recognised as the runner slot (no waiter
   * increment). When a holder releases as the current chain tail it
   * restores `execLock` to `initialLock` so the next burst starts
   * cleanly from the idle state.
   */
  private execLock: Promise<void> = this.initialLock;

  constructor(opts: SessionRegistryOptions) {
    this.model = opts.model;
    this.ttlSec = opts.ttlSec ?? 1800;
    this.maxQueueDepth = opts.maxQueueDepth;
  }

  /**
   * Number of requests currently WAITING to acquire the per-model
   * execution mutex. Does NOT include the one actively running inside
   * `fn`. Primarily for tests and diagnostics.
   */
  get queueDepth(): number {
    return this.queuedCount;
  }

  /** Number of sessions currently cached. Primarily for tests and diagnostics. Always 0 or 1. */
  get size(): number {
    return this.entries.size;
  }

  /**
   * Look up or allocate a session for the given previous response id.
   * Always returns a `SessionLookupResult` and always leaves the cache
   * empty after return (single-warm invariant).
   *
   * On a null id, missing key, expired entry, or prefix-state
   * mismatch: clear and return `{ session: new ChatSession(model), hit: false }`.
   * The caller primes / cold-replays from the `ResponseStore` and
   * re-adopts after the turn commits.
   *
   * On a hit: the entry is removed and its live session is returned
   * alongside `hit: true`. Overlapping requests against the same
   * `previous_response_id` cannot share the same live `ChatSession` —
   * the first wins, the second misses and cold-replays.
   *
   * `requestedInstructions` is the caller's prefix/system state
   * (OpenAI `instructions`, Anthropic `system`, or `null`); byte-for-
   * byte mismatch against the cached entry forces cold replay so
   * the new prefix is re-primed.
   *
   * The `hit` flag drives the `X-Session-Cache` observability header
   * emitted by both `/v1/responses` and `/v1/messages`: when the caller
   * supplied a `previous_response_id`, `hit === true` yields `hit` and
   * `hit === false` yields `cold_replay` (the endpoint then rebuilds
   * from the `ResponseStore` on a fresh session). Requests with no
   * `previous_response_id` (or the stateless `/v1/messages` endpoint,
   * which always passes `null`) yield `fresh` regardless of this flag.
   */
  getOrCreate(previousResponseId: string | null, requestedInstructions: string | null): SessionLookupResult {
    // Every call is about to overwrite native KV state, so drop any
    // other cached entry now — a later `getOrCreate` must not hand
    // out a wrapper whose assumed state has been stomped. Under the
    // single-warm invariant the map holds at most one entry, so the
    // common case is either "the entry we want" or "nothing".
    if (previousResponseId === null) {
      this.entries.clear();
      return { session: new ChatSession(this.model), hit: false };
    }
    const entry = this.entries.get(previousResponseId);
    if (entry === undefined) {
      this.entries.clear();
      return { session: new ChatSession(this.model), hit: false };
    }
    if (entry.expiresAt < nowSec()) {
      this.entries.clear();
      return { session: new ChatSession(this.model), hit: false };
    }
    // Prefix-state mismatch forces cold replay so the new
    // instructions are re-primed; without this guard, output would
    // silently depend on cache state instead of request contents.
    if (entry.instructions !== requestedInstructions) {
      this.entries.clear();
      return { session: new ChatSession(this.model), hit: false };
    }
    // Hit: clear and hand the session out as a single-use lease so
    // a concurrent second request against the same id cold-replays
    // instead of sharing this live ChatSession.
    this.entries.clear();
    return { session: entry.session, hit: true };
  }

  /**
   * Insert a session under a newly allocated response id. Clears the
   * map before inserting to keep the single-warm invariant explicit
   * regardless of caller ordering.
   *
   * `instructions` is the prefix/system state used for this turn;
   * stored on the entry and compared on the next `getOrCreate` to
   * detect prefix changes that must force a cold replay.
   */
  adopt(responseId: string, session: ChatSession<SessionCapableModel>, instructions: string | null): void {
    this.entries.clear();
    this.entries.set(responseId, {
      session,
      instructions,
      expiresAt: nowSec() + this.ttlSec,
    });
  }

  /**
   * Remove a session by response id. No-op if the key is not present.
   */
  drop(responseId: string): void {
    this.entries.delete(responseId);
  }

  /**
   * Walk the map and drop the entry if its TTL has expired.
   * Intended for periodic cleanup via `setInterval`. Under the
   * single-warm invariant the map holds at most one entry.
   */
  sweep(): void {
    const cutoff = nowSec();
    for (const [key, entry] of this.entries) {
      if (entry.expiresAt < cutoff) {
        this.entries.delete(key);
      }
    }
  }

  /** Empty the registry. Useful at shutdown and in tests. */
  clear(): void {
    this.entries.clear();
  }

  /**
   * Serialize `fn` against every other dispatch through this
   * registry's model. The caller must hold the lock across the
   * entire per-model dispatch span — `getOrCreate` ->
   * `primeHistory`/`send*` -> `adopt`/`drop`. Without it, two
   * concurrent `primeHistory()` / `send*()` calls would race on
   * the single mutable native KV cache and whichever finished last
   * would corrupt the other's chain.
   *
   * FIFO chaining via a rolling `execLock` promise: each caller
   * captures the current tail, publishes a fresh pending promise as
   * the new tail, awaits the old tail, then runs `fn`. The
   * `finally` releases regardless of whether `fn` threw.
   *
   * **Admission control.** When `maxQueueDepth` is configured and the
   * current number of waiters (`queuedCount`, excluding the active
   * holder) is already at or above the cap, the call throws
   * {@link QueueFullError} synchronously — SYNCHRONOUSLY from the
   * caller's perspective, not merely before `await prev`. The wrapper
   * is deliberately NOT declared `async` so the admission gate
   * throws on the caller's stack frame, letting endpoint handlers
   * wrap the call site in a plain try/catch without racing promise
   * microtasks. On acceptance the async body takes over via the
   * returned `Promise<T>`.
   *
   * The cap is "waiters-only" — a cap of N permits one running
   * dispatch plus N queued ones, rejecting the (N+1)th waiter. The
   * default (undefined) preserves the original unbounded behaviour.
   *
   * **Runner-slot admission.** Whether a given caller counts as the
   * runner slot or as a waiter is decided up front by comparing
   * `execLock` against the idle sentinel `initialLock`. If they are
   * identical, nobody is currently in-flight and this caller wins
   * the runner slot: it is not counted against the waiter cap and
   * never touches `queuedCount`. Otherwise it is a waiter and the
   * normal cap check / increment / decrement cycle applies. This is
   * what keeps a synchronous burst such as `Promise.all([fn, fn])`
   * admissible under `maxQueueDepth = 1` — Call 1 is the runner,
   * Call 2 is the one allowed waiter, Call 3 would throw.
   */
  withExclusive<T>(fn: () => Promise<T>): Promise<T> {
    // Distinguish runner-slot from waiter admission. If the chain is
    // idle (`execLock === initialLock`) the current caller is about
    // to become the active holder on its very first `await prev`
    // microtask — it must NOT be billed against the waiter cap and
    // must NOT touch `queuedCount`. Only chained callers (someone
    // else still holds or is ahead in the FIFO) count as waiters.
    const asWaiter = this.execLock !== this.initialLock;

    // Admission check — raised synchronously so endpoint handlers
    // can reliably catch `QueueFullError` without racing any
    // `await`. Only waiters can trip the cap; the runner slot is
    // always admitted. The counter is NOT mutated on the reject
    // path; the request never queued.
    if (asWaiter && this.maxQueueDepth !== undefined && this.queuedCount >= this.maxQueueDepth) {
      throw new QueueFullError(this.queuedCount, this.maxQueueDepth);
    }

    const prev = this.execLock;
    let release!: () => void;
    const myLock = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.execLock = myLock;
    if (asWaiter) {
      this.queuedCount += 1;
    }
    return this._runExclusive(prev, myLock, release, fn, asWaiter);
  }

  /**
   * Async tail of {@link withExclusive}. Kept separate so the public
   * wrapper stays a plain (non-async) function whose admission-gate
   * throw lands on the caller's stack synchronously. This helper
   * owns the post-acceptance bookkeeping: awaiting the predecessor
   * lock, transitioning from waiter to holder (`queuedCount`
   * decrement for waiters only), running `fn`, releasing the FIFO
   * tail, and resetting the chain to `initialLock` when this caller
   * is still the tail (so a future burst admits its first entry as
   * a runner-slot rather than as a waiter).
   */
  private async _runExclusive<T>(
    prev: Promise<void>,
    myLock: Promise<void>,
    release: () => void,
    fn: () => Promise<T>,
    asWaiter: boolean,
  ): Promise<T> {
    // Track whether the waiting-counter has already been balanced so
    // an error raised by `await prev` (should never happen today but
    // is cheap to defend against) cannot double-decrement via the
    // outer `finally` below. Runner-slot admissions never touch the
    // counter, so the flag starts already-balanced for them.
    let waitingDecremented = !asWaiter;
    try {
      try {
        await prev;
      } finally {
        // Transition from "waiting" to "running" — the counter must
        // drop exactly here regardless of whether `prev` fulfilled
        // or rejected, because from this point forward the caller is
        // the active holder and no longer part of the queue depth.
        if (!waitingDecremented) {
          this.queuedCount -= 1;
          if (this.queuedCount < 0) this.queuedCount = 0;
          waitingDecremented = true;
        }
      }
      return await fn();
    } finally {
      // Belt-and-suspenders: if `await prev` managed to throw before
      // reaching the inner `finally` (extremely unlikely given the
      // chain is always resolved with `undefined`), still balance the
      // queued counter so a future cap check doesn't drift upward.
      if (!waitingDecremented) {
        this.queuedCount -= 1;
        if (this.queuedCount < 0) this.queuedCount = 0;
        waitingDecremented = true;
      }
      release();
      // Reset the chain to the idle sentinel ONLY when this caller
      // is still the tail — if someone else has already extended the
      // FIFO behind us, leave their tail in place. Reference-equality
      // gate here is what lets the next burst see `execLock ===
      // initialLock` and admit its first caller as a runner slot.
      if (this.execLock === myLock) {
        this.execLock = this.initialLock;
      }
    }
  }
}
