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
  /**
   * Holds AT MOST ONE entry under the single-warm invariant (see the
   * module-level rustdoc). `getOrCreate` and `adopt` both clear the
   * map as part of their contract so a later lookup cannot hand out
   * a wrapper whose assumed native state has been overwritten by a
   * turn on another cached entry.
   */
  private readonly entries: Map<string, SessionEntry> = new Map();
  /**
   * Tail of the per-model execution FIFO. Every `withExclusive` call
   * captures this value as its predecessor, then overwrites it with
   * its own pending promise so the next waiter chains after it. The
   * chain is resolved only when the current holder's `fn` has
   * settled (success or failure), guaranteeing that at most one
   * dispatch runs through this registry's native model at a time.
   * Initialized to `Promise.resolve()` so the first caller proceeds
   * without waiting.
   */
  private execLock: Promise<void> = Promise.resolve();

  constructor(opts: SessionRegistryOptions) {
    this.model = opts.model;
    this.ttlSec = opts.ttlSec ?? 1800;
  }

  /** Number of sessions currently cached. Primarily for tests and diagnostics. Always 0 or 1. */
  get size(): number {
    return this.entries.size;
  }

  /**
   * Look up or allocate a session for the given previous response id.
   * Always returns a `ChatSession` and always leaves the cache empty
   * after return (single-warm invariant).
   *
   * On a null id, missing key, expired entry, or prefix-state
   * mismatch: clear and return `new ChatSession(model)`. The caller
   * primes / cold-replays from the `ResponseStore` and re-adopts
   * after the turn commits.
   *
   * On a hit: the entry is removed and its live session is returned.
   * Overlapping requests against the same `previous_response_id`
   * cannot share the same live `ChatSession` — the first wins, the
   * second misses and cold-replays.
   *
   * `requestedInstructions` is the caller's prefix/system state
   * (OpenAI `instructions`, Anthropic `system`, or `null`); byte-for-
   * byte mismatch against the cached entry forces cold replay so
   * the new prefix is re-primed.
   */
  getOrCreate(
    previousResponseId: string | null,
    requestedInstructions: string | null,
  ): ChatSession<SessionCapableModel> {
    // Every call is about to overwrite native KV state, so drop any
    // other cached entry now — a later `getOrCreate` must not hand
    // out a wrapper whose assumed state has been stomped. Under the
    // single-warm invariant the map holds at most one entry, so the
    // common case is either "the entry we want" or "nothing".
    if (previousResponseId === null) {
      this.entries.clear();
      return new ChatSession(this.model);
    }
    const entry = this.entries.get(previousResponseId);
    if (entry === undefined) {
      this.entries.clear();
      return new ChatSession(this.model);
    }
    if (entry.expiresAt < nowSec()) {
      this.entries.clear();
      return new ChatSession(this.model);
    }
    // Prefix-state mismatch forces cold replay so the new
    // instructions are re-primed; without this guard, output would
    // silently depend on cache state instead of request contents.
    if (entry.instructions !== requestedInstructions) {
      this.entries.clear();
      return new ChatSession(this.model);
    }
    // Hit: clear and hand the session out as a single-use lease so
    // a concurrent second request against the same id cold-replays
    // instead of sharing this live ChatSession.
    this.entries.clear();
    return entry.session;
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
   */
  async withExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const prev = this.execLock;
    let release!: () => void;
    this.execLock = new Promise<void>((resolve) => {
      release = resolve;
    });
    try {
      await prev;
      return await fn();
    } finally {
      release();
    }
  }
}
