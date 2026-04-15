/**
 * SessionRegistry -- per-model TTL cache of live `ChatSession` instances.
 *
 * Design notes:
 *
 *   - **One registry per model.** The server composes a fresh
 *     `SessionRegistry` alongside each registered `ServableModel` in the
 *     `ModelRegistry`. Sessions are keyed purely by the response id
 *     allocated after a turn completes — there is no secondary keying on
 *     model name because the registry itself is already scoped.
 *
 *   - **Single-use / lease semantics on hit.**
 *     `getOrCreate(previousResponseId, requestedInstructions)` is how
 *     endpoints look up a session mid-request. Every successful hit
 *     **removes the entry from the cache** and returns the live
 *     session to the caller as a lease. The caller then either adopts
 *     it back under the freshly allocated response id via `adopt()`
 *     after the turn commits, or drops it on the floor if the turn
 *     did not commit (streaming error, iterator exhaustion, etc.).
 *
 *     This is deliberate. `ChatSession` is single-flight: concurrent
 *     `send` / `sendStream` / `reset` calls throw. Two overlapping
 *     requests that reference the same `previous_response_id` would
 *     otherwise race on the same live object and one of them would
 *     fail with a 500. Leasing on hit means the second request misses
 *     on the now-empty slot and safely cold-replays from the
 *     `ResponseStore` onto a fresh session — both requests succeed
 *     independently under separate allocated ids.
 *
 *     As a consequence, the endpoint layer's post-turn
 *     `sessionReg.drop(previousResponseId)` is effectively a no-op
 *     (the entry was already leased out in `getOrCreate`) but is kept
 *     for belt-and-braces clarity.
 *
 *   - **Instructions / prefix-state change also misses.** Each entry
 *     records the `instructions` string that was used to adopt it. On
 *     `getOrCreate` we compare the caller's `requestedInstructions`
 *     against the cached entry's value. On mismatch the entry is
 *     evicted and a fresh `ChatSession` is returned — so the caller
 *     falls through to the cold-replay path, which re-primes the new
 *     instructions via the full message list. Without this check, a
 *     hot hit would silently keep using the stale warmed system
 *     context while a cold miss would replay the new one, making
 *     output depend on LRU state instead of request contents.
 *
 *     This is the single "prefix/system state" identity check — the
 *     Anthropic endpoint passes its `system` field through the same
 *     parameter, and the OpenAI endpoint passes its `instructions`
 *     field. The registry does not care which is which.
 *
 *   - **Cache miss fallback.** If a client request arrives with a
 *     `previous_response_id` that the server does not have a cached
 *     session for (eviction, restart, cross-node scale-out, or the
 *     lease-on-hit semantics above), the endpoint layer falls back to
 *     reconstructing the conversation from the `ResponseStore`
 *     history, priming it onto a fresh `ChatSession` via
 *     `primeHistory()`, and then resuming the turn through
 *     `startFromHistory()` (or `startFromHistoryStream()` for
 *     streaming). That pair internally dispatches one
 *     `chatSessionStart*` call that rebuilds the full KV cache and
 *     atomically appends the new user turn, so cold replay is
 *     indistinguishable from a hot hit as far as the model is
 *     concerned. The fallback is wired at the endpoint layer (step S2
 *     of the chat-session refactor), not here — `SessionRegistry`
 *     simply returns a bare `new ChatSession(model)` on miss.
 *
 *   - **Eviction matches `ResponseStore`.** The default TTL of 1800
 *     seconds mirrors `RESPONSE_TTL_SECONDS` in
 *     `packages/server/src/endpoints/responses.ts`, so sessions age out
 *     alongside their stored response metadata. LRU eviction kicks in
 *     when the number of cached sessions exceeds `maxEntries` (default
 *     128). LRU ordering is driven by `adopt()` — there is no in-place
 *     promotion on hit because a hit removes the entry entirely.
 *
 *   - **Thread safety.** Node.js is single-threaded within one event
 *     loop tick, so the internal `Map` is safe against concurrent
 *     mutation by design. `sweep()` can be scheduled via `setInterval`
 *     and will synchronously walk the map without colliding with
 *     in-flight `getOrCreate/adopt/drop` calls.
 */

import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';

/** Constructor options for {@link SessionRegistry}. */
export interface SessionRegistryOptions {
  /** The model that every session in this registry wraps. Single-model per registry. */
  model: SessionCapableModel;
  /** TTL in seconds before an unused session is evicted. Default: 1800 (30 min). */
  ttlSec?: number;
  /** Max number of cached sessions before LRU eviction kicks in. Default: 128. */
  maxEntries?: number;
}

interface SessionEntry {
  session: ChatSession<SessionCapableModel>;
  /**
   * The `instructions` / `system` string the caller adopted this
   * session with. `null` if the caller did not supply any. Compared
   * byte-for-byte against the caller's `requestedInstructions` in
   * `getOrCreate` to detect prefix/system-state changes that would
   * otherwise let a hot hit silently reuse a stale warmed prompt.
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
  private readonly maxEntries: number;
  /**
   * Map insertion order == LRU order. `adopt()` inserts at the tail
   * (MRU); the LRU victim is the map's first key
   * (`entries.keys().next().value`). There is no in-place promotion
   * on hit: a hit removes the entry entirely (lease semantics), so
   * ordering is only ever driven by `adopt()`.
   */
  private readonly entries: Map<string, SessionEntry> = new Map();

  constructor(opts: SessionRegistryOptions) {
    this.model = opts.model;
    this.ttlSec = opts.ttlSec ?? 1800;
    this.maxEntries = opts.maxEntries ?? 128;
  }

  /** Number of sessions currently cached. Primarily for tests and diagnostics. */
  get size(): number {
    return this.entries.size;
  }

  /**
   * Look up or allocate a session for the given previous response id.
   *
   * On a null id, a missing key, or an expired entry, returns a fresh
   * `new ChatSession(model)` without caching it — the endpoint layer
   * is responsible for adopting the session back under the newly
   * allocated response id after the turn commits.
   *
   * On a hit, the entry is **removed from the cache (single-use
   * lease)** and its live session is returned. Overlapping requests
   * against the same `previous_response_id` therefore cannot share
   * the same live `ChatSession`: the first caller wins the lease, the
   * second misses and cold-replays from the `ResponseStore`. This is
   * required because `ChatSession` is single-flight and concurrent
   * dispatch throws.
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
    if (previousResponseId === null) {
      return new ChatSession(this.model);
    }
    const entry = this.entries.get(previousResponseId);
    if (entry === undefined) {
      return new ChatSession(this.model);
    }
    if (entry.expiresAt < nowSec()) {
      this.entries.delete(previousResponseId);
      return new ChatSession(this.model);
    }
    // Instructions / prefix-state mismatch: evict and force cold
    // replay so the new instructions are re-primed. Without this
    // guard, a hot hit would silently keep using the stale warmed
    // system context while a cold miss would replay the new one,
    // making output depend on LRU state instead of request contents.
    if (entry.instructions !== requestedInstructions) {
      this.entries.delete(previousResponseId);
      return new ChatSession(this.model);
    }
    // Lease semantics: remove the entry from the cache before handing
    // it back. A concurrent second request against the same id will
    // miss here and cold-replay, so we never hand the same live
    // ChatSession to two overlapping callers.
    this.entries.delete(previousResponseId);
    return entry.session;
  }

  /**
   * Insert a session under a newly allocated response id.
   *
   * Refreshes the TTL expiry for the new entry. If an entry already
   * exists under the same key it is overwritten and treated as an
   * update (still promoted to MRU via delete + reinsert).
   *
   * When adding a new entry would push `size` past `maxEntries`, the
   * least-recently-used entry is evicted before insertion. (An update
   * to an existing key does not trigger eviction — it does not grow
   * the map.)
   *
   * `instructions` is the prefix/system state the caller used for
   * this turn (OpenAI `instructions`, Anthropic `system`, or `null`).
   * It is stored on the entry and compared on the next
   * `getOrCreate` to detect prefix-state changes that must force a
   * cold replay.
   */
  adopt(responseId: string, session: ChatSession<SessionCapableModel>, instructions: string | null): void {
    const existed = this.entries.delete(responseId);
    if (!existed && this.entries.size >= this.maxEntries) {
      const victim = this.entries.keys().next().value;
      if (victim !== undefined) {
        this.entries.delete(victim);
      }
    }
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
   * Walk the map and drop every entry whose TTL has expired.
   *
   * Intended to be scheduled via `setInterval` for periodic cleanup.
   * Safe to call at any time — everything is synchronous within one
   * event loop tick so there is no concurrent-mutation hazard.
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
