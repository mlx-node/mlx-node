/**
 * SessionRegistry -- per-model LRU+TTL cache of live `ChatSession` instances.
 *
 * Design notes:
 *
 *   - **One registry per model.** The server composes a fresh
 *     `SessionRegistry` alongside each registered `ServableModel` in the
 *     `ModelRegistry`. Sessions are keyed purely by the response id
 *     allocated after a turn completes — there is no secondary keying on
 *     model name because the registry itself is already scoped.
 *
 *   - **Cache hit semantics.** `getOrCreate(previousResponseId)` is how
 *     endpoints look up a session mid-request. It returns a fresh
 *     `ChatSession` on miss (no caching) and the existing session on hit
 *     (LRU-promoted). The caller adopts the session under the new
 *     allocated response id via `adopt()` once the turn completes.
 *
 *   - **Cache miss fallback.** If a client request arrives with a
 *     `previous_response_id` that the server does not have a cached
 *     session for (eviction, restart, or cross-node scale-out), the
 *     endpoint layer falls back to reconstructing the conversation from
 *     the `ResponseStore` history and calling
 *     `chatSessionStart(history)` on a fresh `ChatSession`. This
 *     fallback is wired at the endpoint layer (step S2 of the
 *     chat-session refactor), not here — `SessionRegistry` simply
 *     returns a bare `new ChatSession(model)` on miss.
 *
 *   - **Eviction matches `ResponseStore`.** The default TTL of 1800
 *     seconds mirrors `RESPONSE_TTL_SECONDS` in
 *     `packages/server/src/endpoints/responses.ts`, so sessions age out
 *     alongside their stored response metadata. LRU eviction kicks in
 *     when the number of cached sessions exceeds `maxEntries` (default
 *     128).
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
   * Map insertion order == LRU order. On every hit we `delete` + re-set
   * the entry to move it to the MRU position at the tail of the map.
   * The LRU victim is the map's first key (`entries.keys().next().value`).
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
   * On cache miss (null id, unknown id, or expired entry) this returns a
   * fresh `new ChatSession(model)` without caching it — the endpoint
   * layer is responsible for adopting the session under the newly
   * allocated response id after the turn completes.
   *
   * On cache hit the entry is LRU-promoted (deleted and re-inserted so
   * it moves to the tail of the map's insertion order) and the live
   * session is returned.
   */
  getOrCreate(previousResponseId: string | null): ChatSession<SessionCapableModel> {
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
    // LRU promotion: move to MRU position by delete + reinsert.
    this.entries.delete(previousResponseId);
    this.entries.set(previousResponseId, entry);
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
   */
  adopt(responseId: string, session: ChatSession<SessionCapableModel>): void {
    const existed = this.entries.delete(responseId);
    if (!existed && this.entries.size >= this.maxEntries) {
      const victim = this.entries.keys().next().value;
      if (victim !== undefined) {
        this.entries.delete(victim);
      }
    }
    this.entries.set(responseId, {
      session,
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
