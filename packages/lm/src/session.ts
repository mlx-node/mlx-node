/**
 * Session-based chat wrapper for Qwen3.5 Dense.
 *
 * Replaces the broken legacy prefix-matching path for multi-round chat.
 * The underlying model exposes two native entry points:
 *
 *   - `chatSessionStart(messages, config)` — text-only jinja render with
 *     `<|im_end|>` as eos. Establishes a clean ChatML boundary in the KV
 *     cache. Used for turn 1.
 *   - `chatSessionContinue(userMessage, config)` — builds a raw ChatML
 *     delta on top of the existing cache, tokenizes it, and prefills the
 *     delta. Used for turns 2..N.
 *
 * `Qwen35Session` hides the turn-1-vs-later distinction from callers so
 * typical usage is:
 *
 * ```typescript
 * import { Qwen35Model, Qwen35Session } from '@mlx-node/lm';
 *
 * const model = await Qwen35Model.load('./models/qwen3.5-0.8b');
 * const session = new Qwen35Session(model, { system: 'Be concise.' });
 * const r1 = await session.send('Say hi in one word.');
 * const r2 = await session.send('Another word?');
 * await session.reset();
 * ```
 */
import type { ChatConfig, ChatMessage, ChatResult } from '@mlx-node/core';

import type { ChatStreamEvent, Qwen35Model } from './stream.js';

export interface Qwen35SessionOptions {
  /**
   * Optional system prompt injected as the first message on turn 1.
   * Ignored on subsequent turns (the cache already includes it).
   */
  system?: string;
  /**
   * Default `ChatConfig` applied to every `send()` call. Per-call
   * config passed to `send()` is shallow-merged on top of this, and
   * `reuseCache` is always forced on (the session path is a
   * session-reuse operation by definition).
   */
  defaultConfig?: ChatConfig;
}

/**
 * Server-side chat session holding a single multi-round conversation
 * against a {@link Qwen35Model} instance.
 *
 * The session tracks its own turn counter and routes the first turn
 * through `chatSessionStart` and subsequent turns through
 * `chatSessionContinue`. Callers MUST NOT mix direct `model.chat(...)`
 * calls on the same model while a session is live — the legacy chat
 * path would attempt to prefix-match against a cache that ends on
 * `<|im_end|>`, which no jinja template renders.
 *
 * Concurrent `send()` calls on the same session are rejected at the
 * class level because the native path relies on the model's single
 * worker thread to serialize cache mutation, and a second in-flight
 * `send()` would race with the first's cache-save step.
 */
export class Qwen35Session {
  private readonly model: Qwen35Model;
  private readonly system: string | undefined;
  private readonly defaultConfig: ChatConfig;
  private turnCount = 0;
  private inFlight = false;

  constructor(model: Qwen35Model, options: Qwen35SessionOptions = {}) {
    this.model = model;
    this.system = options.system;
    this.defaultConfig = options.defaultConfig ?? {};
  }

  /**
   * Send a user message and resolve with the assistant reply.
   *
   * Turn 1 dispatches to `chatSessionStart` with the optional system
   * prompt + the user message. Subsequent turns dispatch to
   * `chatSessionContinue` with just the user string — the server-side
   * cache already holds the prior history.
   */
  async send(userMessage: string, config?: ChatConfig): Promise<ChatResult> {
    if (this.inFlight) {
      throw new Error('Qwen35Session: concurrent send() not allowed; await the previous call first');
    }
    this.inFlight = true;
    try {
      const mergedConfig: ChatConfig = {
        ...this.defaultConfig,
        ...config,
        // The session path is a session-reuse operation by construction.
        // Force `reuseCache` on regardless of what the caller passed —
        // reuseCache=false on the continue path would wipe the cache
        // the delta depends on (see `chat_tokens_delta_sync`).
        reuseCache: true,
      };

      if (this.turnCount === 0) {
        const messages: ChatMessage[] = [];
        if (this.system != null) {
          messages.push({ role: 'system', content: this.system });
        }
        messages.push({ role: 'user', content: userMessage });
        const result = await this.model.chatSessionStart(messages, mergedConfig);
        this.turnCount++;
        return result;
      }
      // `null` for the new `images` guard parameter — this wrapper is
      // text-only; Step T2 will surface an image-aware session API
      // that threads image changes through a fresh session restart.
      const result = await this.model.chatSessionContinue(userMessage, null, mergedConfig);
      this.turnCount++;
      return result;
    } finally {
      this.inFlight = false;
    }
  }

  /**
   * Streaming variant of {@link Qwen35Session#send}.
   *
   * Returns an `AsyncGenerator<ChatStreamEvent>` that yields delta
   * chunks followed by a single final `done: true` chunk.
   *
   * Turn routing matches `send()`:
   *   - Turn 1 → `chatStreamSessionStart` with the optional system
   *     prompt + user message.
   *   - Turn 2+ → `chatStreamSessionContinue` with just the user
   *     message string.
   *
   * `turnCount` is incremented ONLY when the final `done: true` chunk
   * has been yielded. If the caller breaks out of the `for await` loop
   * before that, the turn is considered abandoned — the partial
   * assistant reply is still in the cache (saved by the Rust side) but
   * `turnCount` stays at the previous value so the next `sendStream()`
   * / `send()` call routes as if the aborted stream never happened:
   * turn 0 aborts re-route through `chatStreamSessionStart` which
   * resets the caches again, and turn N (N>0) aborts stay on the
   * `chatStreamSessionContinue` path. In both cases the session state
   * stays consistent.
   *
   * The `inFlight` flag is cleared in ALL termination paths (normal
   * completion, caller break, exception) via the generator's
   * `finally` block.
   */
  async *sendStream(userMessage: string, config?: ChatConfig): AsyncGenerator<ChatStreamEvent> {
    if (this.inFlight) {
      throw new Error('Qwen35Session: concurrent send() not allowed; await the previous call first');
    }
    this.inFlight = true;
    try {
      const mergedConfig: ChatConfig = {
        ...this.defaultConfig,
        ...config,
        // Same as `send()`: force reuseCache on regardless of the
        // caller's input, or the post-decode save_cache_state step
        // would wipe the very cache the next turn depends on.
        reuseCache: true,
      };

      // `sawFinal` is set ONLY when the stream emits a successful
      // terminal chunk — i.e. `done: true` with a non-error
      // `finishReason`. Guard-violation errors from the native side
      // now arrive as thrown exceptions (see `send_stream_error` in
      // Rust), so the natural `try { ... } finally` flow already
      // prevents the counter from advancing on failure. This extra
      // `finishReason !== 'error'` check is belt-and-suspenders: if
      // any future code path were to emit a `done: true` chunk with
      // `finishReason: 'error'` as a regular stream event (instead of
      // throwing), we still refuse to advance the session and brick
      // subsequent turns.
      let sawFinal = false;
      if (this.turnCount === 0) {
        const messages: ChatMessage[] = [];
        if (this.system != null) {
          messages.push({ role: 'system', content: this.system });
        }
        messages.push({ role: 'user', content: userMessage });
        for await (const event of this.model.chatStreamSessionStart(messages, mergedConfig)) {
          if (event.done && event.finishReason !== 'error') sawFinal = true;
          yield event;
        }
      } else {
        // Pass `null` for `images` — the legacy `Qwen35Session` is
        // text-only and doesn't expose an image-change path. T6 deletes
        // this shim in favour of the generic `ChatSession<M>` wrapper.
        for await (const event of this.model.chatStreamSessionContinue(userMessage, null, mergedConfig)) {
          if (event.done && event.finishReason !== 'error') sawFinal = true;
          yield event;
        }
      }

      // Only advance the turn counter when the stream completed
      // normally (the caller drained all chunks including a
      // successful `done: true`). Caller-break, mid-stream
      // exceptions, or error-finish chunks leave `turnCount`
      // untouched — see JSDoc above for the rationale.
      if (sawFinal) {
        this.turnCount++;
      }
    } finally {
      this.inFlight = false;
    }
  }

  /**
   * Reset the session state.
   *
   * Clears the underlying model's KV caches (via `resetCaches()`) and
   * resets the turn counter so the next `send()` goes through
   * `chatSessionStart` again.
   *
   * Returns a `Promise<void>` even though the current native
   * `resetCaches()` is synchronous — this locks in an async-friendly
   * signature so future async implementations don't break callers.
   */
  async reset(): Promise<void> {
    this.model.resetCaches();
    this.turnCount = 0;
  }

  /**
   * Number of completed turns in this session.
   *
   * 0 before the first `send()`, 1 after the first successful send, etc.
   * Does NOT count in-flight sends.
   */
  get turns(): number {
    return this.turnCount;
  }
}
