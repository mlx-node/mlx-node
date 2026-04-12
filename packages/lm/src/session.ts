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

import type { Qwen35Model } from './stream.js';

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
      const result = await this.model.chatSessionContinue(userMessage, mergedConfig);
      this.turnCount++;
      return result;
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
