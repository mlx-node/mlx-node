/**
 * Server-private `ChatSession` warm-reuse helper.
 *
 * Kept out of the `@mlx-node/lm` package exports entirely so downstream
 * consumers cannot reach it: the module lives inside
 * `@mlx-node/server`, its file path is not on the server's export map,
 * and nothing re-exports it. The only caller is
 * `endpoints/responses.ts`, which invokes it exclusively under a
 * `SessionRegistry` HIT gate.
 *
 * Why this helper exists at all: `ChatSession.reset()` is the safe
 * public wipe — it always calls `model.resetCaches()` because the
 * underlying `SessionCapableModel` is shared across every session
 * lifetime via the native `ModelRegistry`. A partial wipe that leaves
 * the shared native KV cache intact without the HIT gate would leak a
 * previous (unrelated) request's cached prefix into the next
 * `chat_session_start_sync` call. The server's warm-lease replay path
 * DOES have that HIT gate — the registry's own `getOrCreate` hit
 * signal is the authoritative proof that the native cache genuinely
 * belongs to this chain — so a JS-state-only reset that preserves the
 * native cache is correct there and only there.
 *
 * Fields accessed: `inFlight`, `history`, `lastImagesKey`, `turnCount`,
 * `unresolvedOkToolCallCount`. These are TypeScript `private` fields on
 * `ChatSession` (compile-time only) — at runtime they are ordinary
 * properties. The cast through {@link ChatSessionWarmReuseInternals}
 * gives this helper a typed view of the instance without relaxing the
 * class's `private` declarations. The field names MUST stay in sync
 * with `packages/lm/src/chat-session.ts`; a mismatch would silently
 * skip the intended state wipe and is covered by the chat-session
 * unit tests that exercise this path through the endpoint handler.
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
 * JS-state-only reset that DELIBERATELY preserves the underlying
 * model's native KV cache and `cached_token_history`.
 *
 * @internal server-private — used only by the `SessionRegistry` HIT
 * branch in `endpoints/responses.ts`. Never export from this package's
 * `index.ts`.
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
  // is correct here because this helper is the designated server-side
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
