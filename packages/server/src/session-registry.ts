/**
 * SessionRegistry -- per-model cache holding AT MOST one live
 * `ChatSession` whose native KV state is currently valid.
 *
 * **Tier-2 `prompt_cache_key` reuse is ON by default** so the server
 * is compatible with any stateless LLM agent that sends the full
 * conversation history each turn. The key is caller-controlled and
 * HMAC-scoped with a boot-time nonce (raw value never stored on the
 * entry), but two clients that pick the same raw key will still
 * lease the same warm session — a session-hijack surface in
 * multi-tenant settings. For multi-tenant deployments, opt out via
 * `MLX_DISABLE_PROMPT_CACHE_KEY=1` or front the server with an auth
 * proxy that rewrites or namespaces `prompt_cache_key` per tenant
 * before it reaches this process. See also the comments on
 * {@link scopePromptCacheKey}.
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

import { createHash, createHmac, randomBytes } from 'node:crypto';

import type { ChatConfig } from '@mlx-node/core';
import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';

/**
 * Tier-2 `prompt_cache_key` reuse is **ON by default** so the server
 * is immediately compatible with any stateless LLM agent (pi-mono,
 * Aider, Codex CLI, Claude Code, Cline, Continue) that sends the full
 * transcript every turn. Agents that set OpenAI's standard
 * `prompt_cache_key` field — which most do — get automatic KV cache
 * reuse across turns of the same logical session.
 *
 * Opt out via `MLX_DISABLE_PROMPT_CACHE_KEY=1` for **multi-tenant
 * deployments**, where the tier-2 lookup becomes unsafe: two clients
 * that pick the same raw `prompt_cache_key` (by accident or on
 * purpose) would share a warm `ChatSession`, leaking conversation
 * history and sampling state across principals.
 *
 * Multi-tenant isolation is out of scope for this registry. The
 * HMAC-scoping applied below only hides the raw key from memory /
 * dumps — it does NOT protect against two clients sharing the same
 * raw input. Operators who need multi-tenant isolation must either
 * disable the feature or front the server with an auth proxy that
 * rewrites `prompt_cache_key` per-tenant before it reaches the
 * process.
 *
 * Read at call time (not cached at module load) so tests can flip the
 * env via `vi.stubEnv()` between cases without re-importing the module.
 * The check is a single env-var read plus a string compare — negligible
 * against the rest of the lookup work.
 */
function isPromptCacheKeyEnabled(): boolean {
  return process.env.MLX_DISABLE_PROMPT_CACHE_KEY !== '1';
}

/**
 * Minimum length accepted for a caller-supplied `prompt_cache_key`
 * before tier-2 scoping. Short keys make trivial guessing collisions
 * plausible; reject anything shorter than this as if the caller had
 * not supplied a key at all. Chosen to reject one- / two- / few-byte
 * values a client might accidentally pass through while still allowing
 * any reasonable opaque id (UUID prefix, short client token, etc.).
 */
const PROMPT_CACHE_KEY_MIN_LENGTH = 8;

/**
 * Lazily-initialized boot-time nonce used to HMAC every caller-supplied
 * `prompt_cache_key` before it is stored or looked up. Held in memory
 * only — never persisted to disk. A process restart invalidates every
 * tier-2 entry because the next module instance produces a fresh
 * nonce.
 *
 * The nonce makes pre-existing entries unmatchable from outside the
 * process: an attacker who knows a victim's raw `prompt_cache_key` but
 * cannot read the nonce from the server's memory also cannot craft a
 * lookup that collides with the stored HMAC'd key. Combined with the
 * opt-in gate above, the tier-2 surface is off-by-default and bound to
 * a server-instance secret when enabled.
 *
 * Populated lazily on first use so the cost is not paid when tier-2 is
 * disabled. Module-scope so it is shared across every `SessionRegistry`
 * in the process (each per-model registry's HMAC'd keys are still
 * distinct via their per-registry `entries` map — there is no
 * cross-model leakage).
 */
let cachedNonce: Buffer | null = null;

/** Lazily obtain the module-scoped HMAC nonce. */
function getNonce(): Buffer {
  if (cachedNonce === null) {
    cachedNonce = randomBytes(32);
  }
  return cachedNonce;
}

/**
 * Test-only hook used by the scoping unit tests to simulate a
 * server restart: resets the module-scoped HMAC nonce (so every
 * previously stored tier-2 key misses) and clears the silent-miss
 * dedupe cache so tests can re-exercise the once-per-key diagnostic
 * path.
 *
 * **Not exported from the package's public `index.ts` surface** —
 * exporting it there would let downstream consumers nuke tier-2
 * state in production (every stored entry would go unreachable).
 * Tests reach the function via the deep path
 * `packages/server/src/session-registry.ts` instead; that import is
 * deliberately noisy to signal "test-only, do not use from app
 * code". The `__` prefix is a loud-enough convention for the
 * ergonomic test path but NOT sufficient for a public package
 * export.
 */
export function __resetPromptCacheKeyNonceForTests(): void {
  cachedNonce = null;
  loggedSilentMissKeys.clear();
}

/**
 * Test-only probe for the silent-miss dedupe Map's live size. Same
 * "do not use from app code" rationale as
 * {@link __resetPromptCacheKeyNonceForTests}; the `__` prefix and the
 * test-only deep-import path are the load-bearing signals. Exists so
 * the flooding regression test can assert the FIFO cap actually bounds
 * the stored set — an invariant that is otherwise unobservable from
 * outside the module and is NOT implied by the warning count (each
 * call emits at most one warning regardless of whether the cap works).
 */
export function __loggedSilentMissKeysSizeForTests(): number {
  return loggedSilentMissKeys.size;
}

/**
 * Normalize and HMAC-scope a caller-supplied `prompt_cache_key` before
 * it is stored or used for lookup.
 *
 * **Single-tenant trust boundary.** HMAC-scoping hides the raw key
 * from memory dumps and keeps one process instance's stored keys
 * unreachable from another instance (a restart rerolls the nonce),
 * but it does NOT protect against two clients supplying the same raw
 * key — by construction both lookups HMAC to the same scoped key and
 * share the entry. Multi-tenant isolation is out of scope; see the
 * module docstring.
 *
 * Returns `null` when:
 *   - The tier-2 feature is disabled
 *     (`MLX_DISABLE_PROMPT_CACHE_KEY` is set to `"1"`).
 *   - `rawKey` is `null`, `undefined`, or the empty string (callers
 *     that forget to thread the key must not accidentally opt into
 *     tier-2 reuse).
 *   - `rawKey` is shorter than {@link PROMPT_CACHE_KEY_MIN_LENGTH}
 *     characters, which keeps trivial guessing collisions off the
 *     table.
 *
 * Otherwise returns the first 32 hex chars of
 * `HMAC-SHA256(cachedNonce, rawKey)` — opaque, server-instance-scoped,
 * and long enough to preserve the 64-bit entropy floor that the
 * pre-scoped path relied on for key uniqueness.
 */
function scopePromptCacheKey(rawKey: string | null | undefined): string | null {
  if (!isPromptCacheKeyEnabled()) return null;
  if (rawKey == null || rawKey.length < PROMPT_CACHE_KEY_MIN_LENGTH) return null;
  return createHmac('sha256', getNonce()).update(rawKey).digest('hex').slice(0, 32);
}

/**
 * Bounded set of SHA-256-digest prefixes tracking which
 * `prompt_cache_key` values have already had a silent-miss warning
 * emitted in this process. Stored as 64-bit hex digests (not raw
 * strings) so an attacker flooding the endpoint with distinct
 * attacker-controlled keys cannot drive unbounded memory growth.
 * FIFO-evicted at {@link LOGGED_SILENT_MISS_KEYS_MAX} entries via the
 * Map insertion-order guarantee. Reset alongside the nonce / warning
 * flag in {@link __resetPromptCacheKeyNonceForTests} so unit tests
 * can re-exercise the once-per-key path.
 */
const LOGGED_SILENT_MISS_KEYS_MAX = 256;
const loggedSilentMissKeys = new Map<string, true>();

/** Hash a raw key to a bounded digest for dedupe storage. */
function digestSilentMissKey(rawKey: string): string {
  // 64-bit prefix is enough for dedupe across a 256-entry window.
  return createHash('sha256').update(rawKey).digest('hex').slice(0, 16);
}

/**
 * Emit a once-per-raw-key stderr debug warning when the caller
 * supplies a non-empty `prompt_cache_key` but at least one tier-2
 * prerequisite is missing — the env gate is off, or the key is
 * shorter than {@link PROMPT_CACHE_KEY_MIN_LENGTH}. The silent-miss
 * fallback (cold-start as if no key were supplied) is the documented
 * behaviour but is easy to miss during integration; this nudge
 * surfaces the cause once per distinct key so operators don't have
 * to grep source to diagnose a flat `X-Session-Cache: fresh`.
 *
 * No-op when `rawKey` is null / undefined / empty (the caller did
 * not ask for tier-2 at all) or when scoping would succeed (the
 * gate already accepted the key). Called by the endpoint layer
 * right after it has decided `effectivePromptCacheKey`.
 */
export function maybeWarnPromptCacheKeyIneligible(rawKey: string | null | undefined): void {
  if (rawKey == null || rawKey.length === 0) return;
  // Happy-path branch: scoping would succeed, no nudge needed.
  if (isPromptCacheKeyEnabled() && rawKey.length >= PROMPT_CACHE_KEY_MIN_LENGTH) return;
  const digest = digestSilentMissKey(rawKey);
  if (loggedSilentMissKeys.has(digest)) return;
  // FIFO eviction via Map insertion order — bounds memory under
  // adversarial key flooding while preserving once-per-key semantics
  // within the recent-key window.
  if (loggedSilentMissKeys.size >= LOGGED_SILENT_MISS_KEYS_MAX) {
    const oldest = loggedSilentMissKeys.keys().next().value;
    if (oldest !== undefined) loggedSilentMissKeys.delete(oldest);
  }
  loggedSilentMissKeys.set(digest, true);
  if (!isPromptCacheKeyEnabled()) {
    console.warn(
      `[mlx-node] prompt_cache_key supplied but tier-2 reuse is disabled ` +
        `(MLX_DISABLE_PROMPT_CACHE_KEY=1). The key will be ignored and this ` +
        `turn will cold-start. This message is logged once per distinct key.`,
    );
    return;
  }
  console.warn(
    `[mlx-node] prompt_cache_key is shorter than ${PROMPT_CACHE_KEY_MIN_LENGTH} chars; tier-2 reuse requires ` +
      `at least ${PROMPT_CACHE_KEY_MIN_LENGTH} characters. The key will be ignored and this turn will ` +
      `cold-start. This message is logged once per distinct key.`,
  );
}

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
  /**
   * Optional sampling defaults applied to every `ChatSession` this
   * registry allocates. Forwarded verbatim into `new ChatSession(model,
   * { defaultConfig })` so the session's `mergeConfig(overlay)` shallow-
   * merges per-call config on top. Intended for server operators who
   * want to pin per-model sampling knobs (temperature, topK, penalties,
   * etc.) without client cooperation — per-request values from the
   * OpenAI `/v1/responses` or Anthropic `/v1/messages` body still win
   * where present because `ChatSession` treats them as an overlay.
   *
   * When `undefined`, behaviour is unchanged from the pre-defaults era
   * (each `new ChatSession(model)` uses an empty `defaultConfig`).
   */
  samplingDefaults?: ChatConfig;
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
  /**
   * Stable caller-supplied key identifying the logical conversation
   * chain for warm-session reuse across stateless turns that do NOT
   * carry a `previous_response_id`. `null` when the adopting caller
   * supplied no key (or when the request came through an endpoint
   * that does not honor the key, e.g. the chain terminated on an
   * in-`previous_response_id` hop). See
   * {@link SessionRegistry.getOrCreate} for the tier-2 lookup
   * semantics.
   *
   * The distinction between `null` (no key supplied) and the empty
   * string `""` (key explicitly set to empty) is load-bearing — tier-2
   * lookup treats them as different keys so a client that forgets to
   * thread the key does not accidentally collide with another client
   * that did set it to empty.
   */
  promptCacheKey: string | null;
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
   * Per-model sampling defaults forwarded into every new `ChatSession`
   * via its `defaultConfig` constructor option. `undefined` preserves
   * the pre-defaults behaviour (empty `defaultConfig`). See
   * {@link SessionRegistryOptions.samplingDefaults}.
   */
  private samplingDefaults: ChatConfig | undefined;
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
    this.samplingDefaults = opts.samplingDefaults;
  }

  /**
   * Construct a fresh `ChatSession` bound to this registry's model and
   * pre-seeded with the operator-configured `samplingDefaults` (if any).
   * Centralized so every cache-miss branch of `getOrCreate` produces a
   * session whose per-call overlay will merge on top of the same
   * defaults — clients cannot accidentally stray from the server's
   * pinned sampling knobs by picking a cold-replay path.
   */
  private newSession(): ChatSession<SessionCapableModel> {
    if (this.samplingDefaults === undefined) {
      return new ChatSession(this.model);
    }
    return new ChatSession(this.model, { defaultConfig: this.samplingDefaults });
  }

  /**
   * Number of requests currently WAITING to acquire the per-model
   * execution mutex. Does NOT include the one actively running inside
   * `fn`. Primarily for tests and diagnostics.
   */
  get queueDepth(): number {
    return this.queuedCount;
  }

  /**
   * Current sampling defaults applied to every new `ChatSession` this
   * registry allocates. Exposed primarily for tests and diagnostics.
   */
  get defaultSamplingConfig(): ChatConfig | undefined {
    return this.samplingDefaults;
  }

  /**
   * Replace the sampling defaults forwarded into every future
   * `ChatSession` this registry allocates. Called by `ModelRegistry`
   * on a `register(name, model, { samplingDefaults })` refresh so a
   * fresh registration's defaults immediately apply to the next
   * cache-miss cold-start. Sessions already cached at call time keep
   * the defaults they were constructed with — they settle naturally
   * through the single-warm cache rotation.
   */
  setSamplingDefaults(defaults: ChatConfig | undefined): void {
    this.samplingDefaults = defaults;
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
   * Lookup proceeds in two tiers:
   *
   *   1. **Tier 1 — `previousResponseId`.** The existing hot path:
   *      exact id match on a live, non-expired entry whose stored
   *      `instructions` are byte-equal to `requestedInstructions`. On
   *      a match the entry is leased out (single-use: removed from the
   *      map so a concurrent second request cannot share the live
   *      `ChatSession`). On a miss — unknown id, expired, or
   *      instructions drift — the method falls through to a FRESH
   *      session regardless of whether tier 2 would have hit.
   *
   *      `previousResponseId` wins unconditionally when supplied. The
   *      two keys could legitimately identify different conversation
   *      branches (e.g. a client fork where one arm chose the prev-id
   *      path and the other arm chose to set `prompt_cache_key`
   *      without one), so routing the prev-id branch through tier 2
   *      on miss risks splicing the wrong warm state into the wrong
   *      chain. Cold-replay is the safe default.
   *
   *   2. **Tier 2 — `promptCacheKey`.** Only runs when
   *      `previousResponseId` is `null`. Stateless agent clients
   *      (pi-mono, Aider, Codex CLI, Continue, etc.) never use
   *      `previous_response_id` — they own the conversation history
   *      client-side and resend the full transcript on every turn —
   *      so the only way to reuse a warm session across those turns
   *      is to key on the client-supplied `prompt_cache_key`. Scans
   *      for any live, non-expired entry whose stored
   *      `promptCacheKey` is non-null AND byte-equal to the caller's
   *      `promptCacheKey` AND whose stored `instructions` are byte-
   *      equal. Empty string is treated as a distinct key from
   *      `null` — an opt-out sentinel from a client that forgot to
   *      thread the key must NOT collide with another client that
   *      did set it to empty. On a match the entry is leased out
   *      (same single-use semantics as tier 1). On a miss, fall
   *      through to a fresh session.
   *
   * The `hit` flag drives the `X-Session-Cache` observability header
   * emitted by both `/v1/responses` and `/v1/messages`: when the caller
   * supplied a `previous_response_id`, `hit === true` yields `hit` and
   * `hit === false` yields `cold_replay` (the endpoint then rebuilds
   * from the `ResponseStore` on a fresh session). Requests with no
   * `previous_response_id` yield either `fresh` (tier-2 miss) or
   * `prefix_hit` (tier-2 hit — only classified as such once the
   * native `cachedTokens > 0` confirms the prefix-cache machinery
   * actually reused the cached tokens).
   */
  getOrCreate(
    previousResponseId: string | null,
    requestedInstructions: string | null,
    promptCacheKey: string | null = null,
  ): SessionLookupResult {
    // Tier 1: previousResponseId exact match.
    //
    // Every call is about to overwrite native KV state, so drop any
    // other cached entry now — a later `getOrCreate` must not hand
    // out a wrapper whose assumed state has been stomped. Under the
    // single-warm invariant the map holds at most one entry, so the
    // common case is either "the entry we want" or "nothing".
    if (previousResponseId !== null) {
      const entry = this.entries.get(previousResponseId);
      if (entry === undefined) {
        this.entries.clear();
        return { session: this.newSession(), hit: false };
      }
      if (entry.expiresAt < nowSec()) {
        this.entries.clear();
        return { session: this.newSession(), hit: false };
      }
      // Prefix-state mismatch forces cold replay so the new
      // instructions are re-primed; without this guard, output would
      // silently depend on cache state instead of request contents.
      if (entry.instructions !== requestedInstructions) {
        this.entries.clear();
        return { session: this.newSession(), hit: false };
      }
      // Tier-1 hit: clear and hand the session out as a single-use
      // lease so a concurrent second request against the same id
      // cold-replays instead of sharing this live ChatSession. Note
      // that even on a prev-id tier-1 MISS we do NOT fall through to
      // tier 2 — see the docstring above for the precedence
      // rationale.
      this.entries.clear();
      return { session: entry.session, hit: true };
    }

    // Tier 2: promptCacheKey scan (only reached when previousResponseId is null).
    //
    // The registry holds at most one entry under the single-warm
    // invariant, so the "scan" is actually a single lookup — walk the
    // map, check the one entry if present, hit or miss. A non-null
    // scoped key on both the request and the entry plus byte-equal
    // instructions is the match condition.
    //
    // SECURITY: raw caller-supplied keys never touch the map. They
    // are run through {@link scopePromptCacheKey}, which (a) returns
    // `null` when the tier-2 opt-in env var is unset, (b) enforces a
    // minimum length, and (c) HMACs the key with a boot-time nonce
    // held only in this process's memory. Without the opt-in every
    // tier-2 lookup below immediately misses; with the opt-in,
    // attackers who cannot read the process-local nonce cannot craft
    // a lookup that matches a stored entry by guessing the raw key.
    const scopedKey = scopePromptCacheKey(promptCacheKey);
    if (scopedKey !== null) {
      for (const entry of this.entries.values()) {
        if (entry.expiresAt < nowSec()) continue;
        if (entry.promptCacheKey === null) continue;
        if (entry.promptCacheKey !== scopedKey) continue;
        if (entry.instructions !== requestedInstructions) continue;
        // Tier-2 hit: clear and lease (same single-warm / single-use
        // semantics as tier 1).
        this.entries.clear();
        return { session: entry.session, hit: true };
      }
    }

    // Fall through: fresh session. Clear any leftover entry so a
    // later lookup cannot hand out a wrapper whose assumed state has
    // been overwritten by this dispatch.
    this.entries.clear();
    return { session: this.newSession(), hit: false };
  }

  /**
   * Insert a session under a newly allocated response id. Clears the
   * map before inserting to keep the single-warm invariant explicit
   * regardless of caller ordering.
   *
   * `instructions` is the prefix/system state used for this turn;
   * stored on the entry and compared on the next `getOrCreate` to
   * detect prefix changes that must force a cold replay.
   *
   * `promptCacheKey` is the client-supplied conversation-chain key
   * that enables the registry's tier-2 lookup for stateless agent
   * turns that do not carry a `previous_response_id`. `null` /
   * `undefined` means "no key supplied" — stored verbatim so a
   * subsequent stateless lookup that also omits the key does NOT
   * accidentally pick up this entry (only explicit non-null
   * key-equality on both sides can hit tier 2). See
   * {@link SessionRegistry.getOrCreate} for the precedence rules.
   */
  adopt(
    responseId: string,
    session: ChatSession<SessionCapableModel>,
    instructions: string | null,
    promptCacheKey: string | null | undefined = null,
  ): void {
    // Scope the caller-supplied key BEFORE storing so a later
    // `getOrCreate` can only resolve entries via the same opt-in +
    // HMAC path. When tier-2 reuse is disabled (or the key is too
    // short / absent) `scopePromptCacheKey` returns `null`, which
    // disables this entry from ever matching a tier-2 lookup — the
    // raw caller-supplied key is NEVER stored.
    this.entries.clear();
    this.entries.set(responseId, {
      session,
      instructions,
      promptCacheKey: scopePromptCacheKey(promptCacheKey ?? null),
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
