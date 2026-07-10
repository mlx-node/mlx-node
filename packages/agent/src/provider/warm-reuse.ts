/**
 * Agent-private `ChatSession` warm-reuse helper.
 *
 * Port of `packages/server/src/chat-session-warm-reuse.ts` — that module
 * is deliberately kept off the server's export map, so the agent package
 * carries its own copy with identical runtime behavior. Keep the two in
 * sync.
 *
 * Why this helper exists at all: `ChatSession.reset()` is the safe
 * public wipe — it always calls `model.resetCaches()` because the
 * underlying `SessionCapableModel` may be shared across session
 * lifetimes. The agent provider bridge, however, owns exactly one
 * session per model process and replays pi's full message history on
 * every LLM call, so the native KV cache always belongs to the chain
 * being replayed. A JS-state-only reset that preserves the native
 * cache is correct there: the next `primeHistory()` +
 * `startFromHistoryStream()` lets the native prefix verifier recover
 * the reused prefix and skip the corresponding re-prefill.
 *
 * Fields accessed: `inFlight`, `history`, `lastImagesKey`, `turnCount`,
 * `unresolvedOkToolCallCount`. These are TypeScript `private` fields on
 * `ChatSession` (compile-time only) — at runtime they are ordinary
 * properties. The cast through {@link ChatSessionWarmReuseInternals}
 * gives this helper a typed view of the instance without relaxing the
 * class's `private` declarations. The field names MUST stay in sync
 * with `packages/lm/src/chat-session.ts`; a mismatch would silently
 * skip the intended state wipe — the drift test in
 * `packages/agent/__test__/warm-reuse.test.ts` checks every name in
 * {@link WARM_REUSE_TOUCHED_FIELDS} against a real `ChatSession`
 * instance.
 */

import type { ChatSession, SessionCapableModel } from '@mlx-node/lm';

/**
 * Private structural view of the `ChatSession` JS-side state that the
 * warm-reuse helper needs to wipe. Mirrors the internal state
 * documented on `ChatSession` itself — field names are load-bearing:
 * they must byte-match the concrete class's private fields or the
 * cast-based mutation below silently no-ops.
 */
interface ChatSessionWarmReuseInternals {
  inFlight: boolean;
  history: unknown[];
  lastImagesKey: string | null;
  turnCount: number;
  unresolvedOkToolCallCount: number | null;
}

/**
 * `Record<keyof ..., true>` forces this map to list EXACTLY the fields of
 * {@link ChatSessionWarmReuseInternals}: adding/removing/renaming a field
 * in the interface without updating the map is a compile error, so the
 * runtime list below can never drift from the fields the helper touches.
 */
const WARM_REUSE_TOUCHED_FIELD_SET: Record<keyof ChatSessionWarmReuseInternals, true> = {
  inFlight: true,
  history: true,
  lastImagesKey: true,
  turnCount: true,
  unresolvedOkToolCallCount: true,
};

/**
 * Runtime list of the `ChatSession` private field names this module
 * reads or writes — exported solely so the drift test can assert each
 * one still exists on a real `ChatSession` instance.
 */
export const WARM_REUSE_TOUCHED_FIELDS = Object.keys(WARM_REUSE_TOUCHED_FIELD_SET) as ReadonlyArray<
  keyof ChatSessionWarmReuseInternals
>;

/**
 * JS-state-only reset that DELIBERATELY preserves the underlying
 * model's native KV cache and `cached_token_history`.
 *
 * @internal agent-private — used only by the provider bridge's
 * per-call warm replay (`resetPreservingNativeCacheForWarmReuse` →
 * `primeHistory` → `startFromHistoryStream`). Never export from this
 * package's `index.ts`.
 *
 * Wipes ONLY the JS-side session state (history array, image key, turn
 * counter, tool-call fan-out guard). With this function, the JS session
 * is fresh enough for `ChatSession.primeHistory()` (which requires
 * `turnCount === 0`) while the native prefix verifier can still recover
 * the reused prefix on the next `chatSessionStart` and skip the
 * corresponding re-prefill.
 */
export async function resetPreservingNativeCacheForWarmReuse<M extends SessionCapableModel>(
  session: ChatSession<M>,
): Promise<void> {
  // TypeScript `private` fields are only compile-time checks; at
  // runtime they are ordinary properties. The cast through
  // `ChatSessionWarmReuseInternals` preserves full static typing for
  // this helper's mutations while bypassing the `private` gate — which
  // is correct here because this helper is the designated agent-side
  // friend accessor. The cast is funneled through `unknown` because TS
  // correctly rejects a direct `ChatSession → Internals` cast when the
  // concrete class has other non-internals fields.
  const internals = session as unknown as ChatSessionWarmReuseInternals;
  if (internals.inFlight) {
    throw new Error(
      'ChatSession: cannot resetPreservingNativeCacheForWarmReuse() while a send() is in flight; await the previous call first',
    );
  }
  internals.history = [];
  internals.lastImagesKey = null;
  internals.turnCount = 0;
  internals.unresolvedOkToolCallCount = null;
}
