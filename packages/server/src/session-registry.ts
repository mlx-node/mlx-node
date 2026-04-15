/**
 * SessionRegistry -- per-model cache holding AT MOST one live
 * `ChatSession` whose native KV state is currently valid.
 *
 * Design notes:
 *
 *   - **One registry per model.** The server composes a fresh
 *     `SessionRegistry` alongside each registered `ServableModel` in
 *     the `ModelRegistry`. Sessions are keyed purely by the response
 *     id allocated after a turn completes — there is no secondary
 *     keying on model name because the registry itself is already
 *     scoped.
 *
 *   - **Single-warm-session invariant.** `ChatSession<M>` is a thin
 *     JS wrapper — it does NOT own any native KV cache. The cache
 *     lives on the underlying `SessionCapableModel` (a single
 *     mutable resource: one `cached_token_history` / one `caches`
 *     vector per model instance). Any call that runs a turn through
 *     a `ChatSession` overwrites that shared native state, which
 *     silently invalidates every other `ChatSession` wrapper pointing
 *     at the same model.
 *
 *     Caching multiple wrappers per model is therefore an illusion:
 *     at most ONE of them ever matches the real native state
 *     (whichever wrapper ran most recently), and every other cached
 *     wrapper is stale and will produce wrong output on the next
 *     `send`. To prevent that cross-session corruption, this registry
 *     guarantees it holds at most ONE entry at a time:
 *
 *     * `getOrCreate` clears the map on every call. The caller is
 *       about to run a turn that overwrites native state, so any
 *       OTHER cached entry is about to become stale — we drop it
 *       now so a later lookup cannot accidentally hand out a
 *       wrapper whose assumed state has been stomped. On a hit we
 *       return the single existing entry (whose native state is
 *       still valid) and the clear is a no-op; on a miss we return
 *       a fresh `ChatSession` after evicting whatever was left.
 *     * `adopt` also clears before inserting. Under the invariant
 *       there is at most one leftover entry at that point, but the
 *       belt-and-braces clear keeps the invariant explicit
 *       regardless of caller ordering.
 *
 *   - **Lease semantics on hit.** The clear-on-hit also gives us
 *     single-flight lease semantics for free: two overlapping
 *     requests that both reference the same `previous_response_id`
 *     cannot share the same live `ChatSession`. The first caller
 *     wins the cleared entry; the second caller finds the map empty
 *     and cold-replays from the `ResponseStore` on a fresh session.
 *     Without this, the second caller would hit `ChatSession`'s
 *     single-flight "concurrent send() not allowed" guard.
 *
 *   - **Instructions / prefix-state change also misses.** Each entry
 *     records the `instructions` string that was used to adopt it.
 *     On `getOrCreate` we compare the caller's `requestedInstructions`
 *     against the cached entry's value. On mismatch we clear and
 *     return a fresh session — the caller falls through to the
 *     cold-replay path, which re-primes the new instructions via
 *     the full message list. Without this check, a hit would
 *     silently keep using the stale warmed system context while a
 *     cold miss would replay the new one, making output depend on
 *     cache state instead of request contents.
 *
 *     The Anthropic endpoint passes its `system` field through the
 *     same parameter, and the OpenAI endpoint passes its
 *     `instructions` field. The registry does not care which is
 *     which — it is the caller's single "prefix/system state"
 *     identity field.
 *
 *   - **Cache miss fallback.** If a client request arrives with a
 *     `previous_response_id` that the server does not have a cached
 *     session for (eviction, interleaved turn on a different chain,
 *     restart, or the lease-on-hit semantics above), the endpoint
 *     layer falls back to reconstructing the conversation from the
 *     `ResponseStore` history, priming it onto a fresh
 *     `ChatSession` via `primeHistory()`, and then resuming the turn
 *     through `startFromHistory()` (or `startFromHistoryStream()`
 *     for streaming). That pair internally dispatches one
 *     `chatSessionStart*` call that rebuilds the full KV cache and
 *     atomically appends the new user turn, so cold replay is
 *     indistinguishable from a hot hit as far as the model is
 *     concerned.
 *
 *   - **TTL.** The default TTL of 1800 seconds mirrors
 *     `RESPONSE_TTL_SECONDS` in `packages/server/src/endpoints/responses.ts`,
 *     so the cached entry ages out alongside its stored response
 *     metadata. Because the registry holds at most one entry at a
 *     time, there is no LRU bookkeeping — only a single expiry
 *     check on lookup.
 *
 *   - **Thread safety.** Node.js is single-threaded within one event
 *     loop tick, so the internal `Map` is safe against concurrent
 *     mutation by design. `sweep()` can be scheduled via
 *     `setInterval` and will synchronously walk the map (which holds
 *     at most one entry) without colliding with in-flight
 *     `getOrCreate`/`adopt`/`drop` calls.
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
   *
   * Always returns a `ChatSession` and always leaves the cache
   * empty after return (see the module-level rustdoc for the
   * single-warm invariant).
   *
   * On a null id, a missing key, an expired entry, or a prefix-state
   * mismatch: clears the cache and returns a fresh
   * `new ChatSession(model)`. The caller is responsible for priming
   * / cold-replaying from the `ResponseStore` and then adopting the
   * session back under the freshly allocated response id after the
   * turn commits.
   *
   * On a hit: the entry is removed from the cache and its live
   * session is returned. Overlapping requests against the same
   * `previous_response_id` therefore cannot share the same live
   * `ChatSession`: the first caller wins the entry, the second
   * misses and cold-replays. This is required because `ChatSession`
   * is single-flight and concurrent dispatch throws.
   *
   * `requestedInstructions` is the caller's current prefix/system
   * state (OpenAI `instructions`, Anthropic `system`, or `null` if
   * the caller did not supply any). It is compared byte-for-byte
   * against the cached entry's `instructions`; mismatch is treated
   * as a miss so the caller falls through to cold replay and the
   * new instructions are re-primed into the fresh session.
   */
  getOrCreate(
    previousResponseId: string | null,
    requestedInstructions: string | null,
  ): ChatSession<SessionCapableModel> {
    // The single-warm invariant: every call is about to hand a
    // `ChatSession` to the caller, who will run a turn that overwrites
    // the model's shared native KV cache. Any OTHER cached entry is
    // therefore about to become stale — drop the whole map now so a
    // later `getOrCreate` cannot accidentally hand out a wrapper
    // whose assumed native state has been stomped. Under the
    // invariant there is at most ONE entry in the map at this point,
    // so the common case is either "the entry we want" or "nothing".
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
    // Instructions / prefix-state mismatch: evict and force cold
    // replay so the new instructions are re-primed. Without this
    // guard, a hit would silently keep using the stale warmed
    // system context while a cold miss would replay the new one,
    // making output depend on cache state instead of request
    // contents.
    if (entry.instructions !== requestedInstructions) {
      this.entries.clear();
      return new ChatSession(this.model);
    }
    // Hit: clear the entry (under the invariant it is the only
    // entry) and hand the session out as a single-use lease. A
    // concurrent second request against the same id will miss on
    // the now-empty map and cold-replay, so we never hand the same
    // live ChatSession to two overlapping callers.
    this.entries.clear();
    return entry.session;
  }

  /**
   * Insert a session under a newly allocated response id.
   *
   * Under the single-warm invariant there is at most one leftover
   * entry in the map at call time (from a prior `adopt` that has not
   * been leased out again). We clear the map before inserting to
   * keep the invariant explicit regardless of caller ordering — it
   * is a no-op in the common case but guards against future
   * refactors that might allow more than one entry to linger.
   *
   * `instructions` is the prefix/system state the caller used for
   * this turn (OpenAI `instructions`, Anthropic `system`, or `null`).
   * It is stored on the entry and compared on the next
   * `getOrCreate` to detect prefix-state changes that must force a
   * cold replay.
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
   *
   * Intended to be scheduled via `setInterval` for periodic cleanup.
   * Under the single-warm invariant the map holds at most one entry,
   * so this is effectively "check if the one entry is stale and
   * drop it if so". Kept as a for-loop for shape symmetry with the
   * previous bounded-cache implementation. Safe to call at any time
   * — everything is synchronous within one event loop tick so there
   * is no concurrent-mutation hazard.
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
}
