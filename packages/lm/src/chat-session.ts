/**
 * Generic server-side chat session wrapper.
 *
 * `ChatSession<M>` is the cross-model chat-session wrapper. It works
 * against any model that exposes the uniform chat-session NAPI
 * surface — `chatSessionStart`,
 * `chatSessionContinue`, `chatSessionContinueTool`, and their
 * streaming variants plus `resetCaches`. See `SessionCapableModel`
 * below.
 *
 * Design notes:
 *
 *   - The session tracks its own `ChatMessage[]` history on the
 *     TypeScript side. In the common text-continue case the history
 *     is only appended to and never read back — each `send()` on
 *     turn >= 1 issues a cheap `chatSessionContinue` delta against
 *     the live KV cache. The history is kept purely so the
 *     image-change mid-session path can call `chatSessionStart` with
 *     the full rebuilt history for a clean re-prefill.
 *
 *   - An image hash (`lastImagesKey`) tracks the images bound to the
 *     current cache. A `send()` call whose image set has changed
 *     (different bytes or different ordering) triggers a full
 *     restart: `resetCaches()` → push the new user message (with
 *     images) to history → `chatSessionStart(history)`.
 *
 *   - Text-only `send()` on turn >= 1 takes the cheap delta path.
 *
 *   - `sendToolResult` always dispatches `chatSessionContinueTool`,
 *     since tool turns never change image state.
 *
 *   - `sawFinal` gates `turnCount` advance on the streaming path, so
 *     the session refuses to advance when the stream throws
 *     mid-decode or yields a final chunk with
 *     `finishReason: 'error'`.
 *
 *   - The `inFlight` guard rejects concurrent `send()` /
 *     `sendStream()` calls at the class level. The native side
 *     serializes cache mutation on a single worker thread, so a
 *     second in-flight call would race the first's cache-save step.
 *
 * ## Typical usage
 *
 * ```typescript
 * import { Qwen35Model, ChatSession } from '@mlx-node/lm';
 *
 * const model = await Qwen35Model.load('./models/qwen3.5-0.8b');
 * const session = new ChatSession(model, { system: 'Be concise.' });
 * const r1 = await session.send('Say hi in one word.');
 * const r2 = await session.send('Another word?');
 * await session.reset();
 * ```
 */
import type { ChatConfig, ChatMessage, ChatResult } from '@mlx-node/core';

import type { ChatStreamEvent } from './stream.js';

/**
 * Structural interface matched by every generative model wrapper
 * (`Qwen35Model`, `Qwen35MoeModel`, `Lfm2Model`, `Gemma4Model`,
 * `Qwen3Model`, and the Qianfan-OCR VLM wrapper). `ChatSession<M>` is
 * generic over `M extends SessionCapableModel` so each session
 * instance statically binds to a specific model's concrete type
 * (handy for IDE autocomplete) while the implementation remains
 * fully structural.
 */
export interface SessionCapableModel {
  chatSessionStart(messages: ChatMessage[], config?: ChatConfig | null): Promise<ChatResult>;
  chatSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    config?: ChatConfig | null,
  ): Promise<ChatResult>;
  chatSessionContinueTool(toolCallId: string, content: string, config?: ChatConfig | null): Promise<ChatResult>;
  chatStreamSessionStart(messages: ChatMessage[], config?: ChatConfig | null): AsyncGenerator<ChatStreamEvent>;
  chatStreamSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    config?: ChatConfig | null,
  ): AsyncGenerator<ChatStreamEvent>;
  chatStreamSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
  ): AsyncGenerator<ChatStreamEvent>;
  resetCaches(): void;
}

/** Per-call options for {@link ChatSession#send} / `sendStream`. */
export interface SendOptions {
  /**
   * Optional image bytes attached to this user turn. When the image
   * set differs from the session's current `lastImagesKey`, the
   * session forcibly restarts via `chatSessionStart`.
   */
  images?: Uint8Array[];
  /**
   * Per-call `ChatConfig` overlay applied on top of the session's
   * `defaultConfig`. `reuseCache` is always forced on regardless of
   * what the caller passes.
   */
  config?: ChatConfig;
}

/** Constructor options for {@link ChatSession}. */
export interface ChatSessionOptions {
  /**
   * Optional system prompt prepended as the first message on turn 1.
   * Subsequent turns don't re-inject the system prompt — the cache
   * already holds it.
   */
  system?: string;
  /**
   * Default `ChatConfig` applied to every `send()` / `sendStream()`
   * / `sendToolResult()` call. Per-call config is shallow-merged on
   * top of this, and `reuseCache` is forced on.
   */
  defaultConfig?: ChatConfig;
}

/**
 * Compute a stable hex-encoded identity key for a list of image
 * byte buffers.
 *
 * Returns `null` when no images are provided so `send()` can
 * distinguish "no-images" from "image set changed". The key is
 * order-sensitive: `[A, B]` and `[B, A]` produce different keys,
 * matching the positional semantics of the underlying VLM chat
 * template.
 *
 * This is a byte-identity check — callers use the key solely to
 * decide whether to restart the server-side session, so a
 * non-cryptographic hash is sufficient. We use FNV-1a 64-bit with a
 * length-prefixed framing so different image counts and different
 * byte lengths cannot collide by accident.
 *
 * Implementation note: kept fully sync + self-contained so
 * `send()` can stay synchronous in its routing decision and so the
 * module has no external runtime dependencies beyond `@mlx-node/core`
 * and the existing stream bridge.
 */
function computeImagesKey(images: Uint8Array[] | undefined): string | null {
  if (!images || images.length === 0) return null;
  // FNV-1a 64-bit. Split into two 32-bit halves because JavaScript
  // doesn't have a native 64-bit integer type and BigInt ops are
  // slow on large byte streams. This emulates 64-bit FNV-1a using
  // paired 32-bit lo/hi halves — the standard JS idiom.
  const FNV_OFFSET_LO = 0x84222325 >>> 0;
  const FNV_OFFSET_HI = 0xcbf29ce4 >>> 0;
  const FNV_PRIME_LO = 0x000001b3 >>> 0;
  const FNV_PRIME_HI = 0x00000100 >>> 0;

  let lo = FNV_OFFSET_LO;
  let hi = FNV_OFFSET_HI;

  function mix(byte: number): void {
    lo = (lo ^ byte) >>> 0;
    // Multiply (hi:lo) by (FNV_PRIME_HI:FNV_PRIME_LO) mod 2^64.
    // Break 32-bit halves into 16-bit quarters to keep intermediate
    // products inside the safe-integer range.
    const loLo = lo & 0xffff;
    const loHi = lo >>> 16;
    const hiLo = hi & 0xffff;
    const hiHi = hi >>> 16;

    const pLo = FNV_PRIME_LO & 0xffff;
    const pLoH = FNV_PRIME_LO >>> 16;
    const pHi = FNV_PRIME_HI & 0xffff;
    const pHiH = FNV_PRIME_HI >>> 16;

    const r0 = loLo * pLo;
    const r1 = loLo * pLoH + loHi * pLo;
    const r2 = loLo * pHi + loHi * pLoH + hiLo * pLo;
    const r3 = loLo * pHiH + loHi * pHi + hiLo * pLoH + hiHi * pLo;

    const newLo0 = r0 & 0xffff;
    const carry1 = r0 >>> 16;
    const sum1 = r1 + carry1;
    const newLo1 = sum1 & 0xffff;
    const carry2 = Math.floor(sum1 / 0x10000);
    const sum2 = r2 + carry2;
    const newHi0 = sum2 & 0xffff;
    const carry3 = Math.floor(sum2 / 0x10000);
    const sum3 = r3 + carry3;
    const newHi1 = sum3 & 0xffff;

    lo = ((newLo1 << 16) | newLo0) >>> 0;
    hi = ((newHi1 << 16) | newHi0) >>> 0;
  }

  // Frame each image with a 4-byte little-endian length prefix so
  // `[ab, c]` and `[a, bc]` hash to distinct values.
  mix(images.length & 0xff);
  mix((images.length >>> 8) & 0xff);
  mix((images.length >>> 16) & 0xff);
  mix((images.length >>> 24) & 0xff);
  for (const img of images) {
    mix(img.byteLength & 0xff);
    mix((img.byteLength >>> 8) & 0xff);
    mix((img.byteLength >>> 16) & 0xff);
    mix((img.byteLength >>> 24) & 0xff);
    for (let i = 0; i < img.byteLength; i++) {
      mix(img[i]!);
    }
  }
  return hi.toString(16).padStart(8, '0') + lo.toString(16).padStart(8, '0');
}

/**
 * Cross-model chat session. See module docstring for design notes.
 *
 * The generic parameter `M` statically captures the concrete model
 * type so the structural interface stays as expressive as the
 * concrete one. Internally the class only uses the
 * `SessionCapableModel` surface.
 */
export class ChatSession<M extends SessionCapableModel = SessionCapableModel> {
  private readonly model: M;
  private readonly system: string | undefined;
  private readonly defaultConfig: ChatConfig;

  /**
   * Full conversation history tracked on the TS side. Appended to on
   * every successful turn. Only read back when the image-change path
   * triggers a restart — normal text continues use the server-side
   * cache, not this array.
   */
  private history: ChatMessage[] = [];

  /**
   * Hex-encoded byte-identity key of the image set currently bound
   * to the server's KV cache (FNV-1a 64-bit; see `computeImagesKey`).
   * `null` when no images are cached. A `send()` whose new key
   * differs triggers a full `chatSessionStart` restart.
   */
  private lastImagesKey: string | null = null;

  private turnCount = 0;
  private inFlight = false;

  constructor(model: M, options: ChatSessionOptions = {}) {
    this.model = model;
    this.system = options.system;
    this.defaultConfig = options.defaultConfig ?? {};
  }

  /**
   * Number of completed turns. Increments only after a successful
   * round-trip — in-flight or failed calls leave this untouched.
   */
  get turns(): number {
    return this.turnCount;
  }

  /** Whether the session currently has images bound to its cache. */
  get hasImages(): boolean {
    return this.lastImagesKey !== null;
  }

  /**
   * Send a user message and resolve with the assistant reply.
   *
   * Turn 0 and any turn whose image set has changed dispatch through
   * `chatSessionStart` with the full history. All other turns use
   * the cheap `chatSessionContinue` delta path.
   */
  async send(userMessage: string, opts: SendOptions = {}): Promise<ChatResult> {
    if (this.inFlight) {
      throw new Error('ChatSession: concurrent send() not allowed; await the previous call first');
    }
    this.inFlight = true;
    try {
      const mergedConfig = this.mergeConfig(opts.config);
      const newImagesKey = computeImagesKey(opts.images);
      // Only an explicit NEW image set can trigger a restart. Omitting
      // `images` (newImagesKey === null) is interpreted as "keep the
      // current image cache state" — the server-side cache already
      // holds any prior image context, so a text-only follow-up like
      // "what about the top-right?" can stay on the cheap delta path
      // even after an image turn.
      const imageChanged = newImagesKey !== null && newImagesKey !== this.lastImagesKey;
      const isFirstTurn = this.turnCount === 0;

      if (isFirstTurn || imageChanged) {
        return await this.runStartPath(userMessage, opts.images, newImagesKey, imageChanged, isFirstTurn, mergedConfig);
      }

      // Delta continue: text-only, images always null. The server
      // cache already holds all prior turns (including any images
      // from an earlier restart), so we only need to ship the new
      // user string.
      const result = await this.model.chatSessionContinue(userMessage, null, mergedConfig);
      this.history.push({ role: 'user', content: userMessage });
      this.history.push({ role: 'assistant', content: result.text });
      this.turnCount++;
      return result;
    } finally {
      this.inFlight = false;
    }
  }

  /**
   * Streaming variant of {@link ChatSession#send}.
   *
   * Routing matches `send()`. The assistant reply is accumulated
   * from stream deltas and pushed to `history` only after a
   * successful terminal chunk (`done: true` with non-error
   * `finishReason`). Caller break, mid-stream exceptions, and error
   * finishes all leave `turnCount` untouched and the history
   * un-appended for the turn so the next call re-routes through the
   * start path.
   */
  async *sendStream(userMessage: string, opts: SendOptions = {}): AsyncGenerator<ChatStreamEvent> {
    if (this.inFlight) {
      throw new Error('ChatSession: concurrent send() not allowed; await the previous call first');
    }
    this.inFlight = true;
    try {
      const mergedConfig = this.mergeConfig(opts.config);
      const newImagesKey = computeImagesKey(opts.images);
      // Only an explicit NEW image set can trigger a restart. Omitting
      // `images` (newImagesKey === null) is interpreted as "keep the
      // current image cache state" — the server-side cache already
      // holds any prior image context, so a text-only follow-up like
      // "what about the top-right?" can stay on the cheap delta path
      // even after an image turn.
      const imageChanged = newImagesKey !== null && newImagesKey !== this.lastImagesKey;
      const isFirstTurn = this.turnCount === 0;

      if (isFirstTurn || imageChanged) {
        yield* this.runStartStreamPath(userMessage, opts.images, newImagesKey, imageChanged, isFirstTurn, mergedConfig);
        return;
      }

      // Delta continue stream: text-only.
      let sawFinal = false;
      let accumulated = '';
      let finalRaw: string | null = null;
      try {
        for await (const event of this.model.chatStreamSessionContinue(userMessage, null, mergedConfig)) {
          if (event.done) {
            if (event.finishReason !== 'error') {
              sawFinal = true;
              finalRaw = event.text;
            }
          } else {
            accumulated += event.text;
          }
          yield event;
        }
      } finally {
        // finally runs for normal completion, mid-stream throw,
        // caller `break` (which calls `iterator.return()` and
        // short-circuits the suspended yield), and error-finish
        // chunks alike. The delta path doesn't push to history until
        // commit, so the rollback branch is a no-op: nothing to
        // undo, and the native cache state is managed by the Rust
        // save_cache_state path on its own.
        if (sawFinal) {
          this.history.push({ role: 'user', content: userMessage });
          this.history.push({ role: 'assistant', content: finalRaw ?? accumulated });
          this.turnCount++;
        }
      }
    } finally {
      this.inFlight = false;
    }
  }

  /**
   * Send a tool-result turn. Always dispatches
   * `chatSessionContinueTool` — tool turns never change image state,
   * so there is no restart path here.
   *
   * Appends a `{ role: 'tool', ... }` message to history on success.
   */
  async sendToolResult(toolCallId: string, content: string, opts: { config?: ChatConfig } = {}): Promise<ChatResult> {
    if (this.inFlight) {
      throw new Error('ChatSession: concurrent send() not allowed; await the previous call first');
    }
    this.inFlight = true;
    try {
      const mergedConfig = this.mergeConfig(opts.config);
      const result = await this.model.chatSessionContinueTool(toolCallId, content, mergedConfig);
      this.history.push({ role: 'tool', content, toolCallId });
      this.history.push({ role: 'assistant', content: result.text });
      this.turnCount++;
      return result;
    } finally {
      this.inFlight = false;
    }
  }

  /** Streaming variant of {@link ChatSession#sendToolResult}. */
  async *sendToolResultStream(
    toolCallId: string,
    content: string,
    opts: { config?: ChatConfig } = {},
  ): AsyncGenerator<ChatStreamEvent> {
    if (this.inFlight) {
      throw new Error('ChatSession: concurrent send() not allowed; await the previous call first');
    }
    this.inFlight = true;
    try {
      const mergedConfig = this.mergeConfig(opts.config);
      let sawFinal = false;
      let accumulated = '';
      let finalRaw: string | null = null;
      try {
        for await (const event of this.model.chatStreamSessionContinueTool(toolCallId, content, mergedConfig)) {
          if (event.done) {
            if (event.finishReason !== 'error') {
              sawFinal = true;
              finalRaw = event.text;
            }
          } else {
            accumulated += event.text;
          }
          yield event;
        }
      } finally {
        // finally runs for normal completion, mid-stream throw,
        // caller `break` (iterator.return() short-circuits the yield),
        // and error-finish chunks alike. Tool turns never touch
        // history until commit, so the rollback branch is a no-op.
        if (sawFinal) {
          this.history.push({ role: 'tool', content, toolCallId });
          this.history.push({ role: 'assistant', content: finalRaw ?? accumulated });
          this.turnCount++;
        }
      }
    } finally {
      this.inFlight = false;
    }
  }

  /**
   * Reset the session state.
   *
   * Clears the underlying model's KV caches and wipes local history,
   * image key, and turn counter so the next `send()` goes through
   * `chatSessionStart` again.
   *
   * Returns `Promise<void>` for an async-friendly signature even
   * though `resetCaches()` is currently synchronous.
   */
  async reset(): Promise<void> {
    if (this.inFlight) {
      throw new Error('ChatSession: cannot reset() while a send() is in flight; await the previous call first');
    }
    this.model.resetCaches();
    this.history = [];
    this.lastImagesKey = null;
    this.turnCount = 0;
  }

  // -------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------

  /**
   * Merge default + per-call config and force `reuseCache: true`.
   * The session path is a session-reuse operation by construction —
   * `reuseCache: false` on the continue path would wipe the very
   * cache the delta depends on.
   */
  private mergeConfig(overlay: ChatConfig | undefined): ChatConfig {
    return {
      ...this.defaultConfig,
      ...overlay,
      reuseCache: true,
    };
  }

  /**
   * Shared start-path logic for `send()`. Handles both the turn-0
   * first-ever-send case and the image-change mid-session restart
   * case. The image-change restart preserves prior history so the
   * native side gets the full conversation re-rendered with the new
   * image set.
   */
  private async runStartPath(
    userMessage: string,
    images: Uint8Array[] | undefined,
    newImagesKey: string | null,
    imageChanged: boolean,
    isFirstTurn: boolean,
    config: ChatConfig,
  ): Promise<ChatResult> {
    // Capture pre-state so the restart can be rolled back if the
    // native call fails. The image-change branch resets caches BEFORE
    // we know whether the new prefill will succeed, so on failure we
    // also have to drop turnCount + lastImagesKey to force the next
    // call to re-route through the start path (rather than a delta
    // continue against wiped caches).
    const wasImageChangeRestart = imageChanged && !isFirstTurn;
    const historyLenBefore = this.history.length;

    this.prepareStartPath(imageChanged, isFirstTurn);
    const userMsg = this.buildUserMessage(userMessage, images);
    this.history.push(userMsg);
    try {
      // Pass a shallow snapshot so later pushes to `this.history`
      // (e.g. the assistant reply below) don't retroactively mutate
      // what the native side / any mock observed as its `messages`
      // argument.
      const result = await this.model.chatSessionStart(this.history.slice(), config);
      this.history.push({ role: 'assistant', content: result.text });
      this.turnCount++;
      this.lastImagesKey = newImagesKey;
      return result;
    } catch (err) {
      // Roll back: drop the tentative user push so history stays
      // consistent with turnCount.
      this.history.length = historyLenBefore;
      if (wasImageChangeRestart) {
        // Caches were wiped by prepareStartPath() but the new prefill
        // failed. Force the next call to re-route through the start
        // path with the (preserved) prior history.
        this.turnCount = 0;
        this.lastImagesKey = null;
      }
      throw err;
    }
  }

  /** Streaming counterpart to {@link runStartPath}. */
  private async *runStartStreamPath(
    userMessage: string,
    images: Uint8Array[] | undefined,
    newImagesKey: string | null,
    imageChanged: boolean,
    isFirstTurn: boolean,
    config: ChatConfig,
  ): AsyncGenerator<ChatStreamEvent> {
    // Capture pre-state so any non-successful exit can roll back.
    // See `runStartPath` for the full rationale.
    const wasImageChangeRestart = imageChanged && !isFirstTurn;
    const historyLenBefore = this.history.length;

    this.prepareStartPath(imageChanged, isFirstTurn);
    const userMsg = this.buildUserMessage(userMessage, images);
    // Stage the user message on the pending history BEFORE the
    // stream starts — the native call reads it synchronously via
    // `model.chatStreamSessionStart(history, config)`.
    this.history.push(userMsg);

    let sawFinal = false;
    let accumulated = '';
    let finalRaw: string | null = null;
    // Snapshot the history before dispatch — see `runStartPath` for
    // the rationale.
    const historySnapshot = this.history.slice();
    try {
      for await (const event of this.model.chatStreamSessionStart(historySnapshot, config)) {
        if (event.done) {
          if (event.finishReason !== 'error') {
            sawFinal = true;
            finalRaw = event.text;
          }
        } else {
          accumulated += event.text;
        }
        yield event;
      }
    } finally {
      // finally runs in ALL termination paths: normal completion,
      // mid-stream throw, caller `break` (which calls
      // `iterator.return()` on the generator and short-circuits the
      // suspended `yield`, skipping any post-loop code), and
      // error-finish chunks. The unified commit-or-rollback below
      // makes restart fully transactional regardless of how the
      // generator was wound down. Mid-stream throws still propagate
      // naturally — finally runs first, then the error continues up.
      if (sawFinal) {
        this.history.push({ role: 'assistant', content: finalRaw ?? accumulated });
        this.turnCount++;
        this.lastImagesKey = newImagesKey;
      } else {
        // Roll back: drop the tentative user push so history stays
        // consistent with turnCount.
        this.history.length = historyLenBefore;
        if (wasImageChangeRestart) {
          // Caches were wiped by prepareStartPath() but the new
          // prefill never reached a successful done:true. Force the
          // next call to re-route through the start path with the
          // preserved prior history.
          this.turnCount = 0;
          this.lastImagesKey = null;
        }
      }
    }
  }

  /**
   * Shared pre-start bookkeeping for both `send()` and `sendStream()`:
   *
   *   - On an image-change restart (turn >= 1), reset the native KV
   *     caches so the new image set gets a fresh prefill. History is
   *     intentionally preserved — `chatSessionStart` receives the full
   *     accumulated conversation plus the new user turn so the jinja
   *     render walks every prior turn and every prior image again
   *     (see plan's Turn 3 example: "full jinja on 3-turn history +
   *     image B"). `lastImagesKey` will be overwritten by the
   *     successful start path right after, and `turnCount` is
   *     incremented by the start path the same way as for any other
   *     turn.
   *   - On a fresh / reset history, re-inject the system prompt.
   */
  private prepareStartPath(imageChanged: boolean, isFirstTurn: boolean): void {
    if (imageChanged && !isFirstTurn) {
      this.model.resetCaches();
    }
    if (this.history.length === 0 && this.system != null) {
      this.history.push({ role: 'system', content: this.system });
    }
  }

  /** Build a user `ChatMessage` with or without attached images. */
  private buildUserMessage(userMessage: string, images: Uint8Array[] | undefined): ChatMessage {
    if (images && images.length > 0) {
      return { role: 'user', content: userMessage, images };
    }
    return { role: 'user', content: userMessage };
  }
}
