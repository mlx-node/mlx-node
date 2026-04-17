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
 *     since tool turns never change image state. The session enforces
 *     a strict unresolved-ok-tool-call contract at runtime, driven by
 *     `unresolvedOkToolCallCount` (derived from `ChatResult.toolCalls`
 *     after each turn via `countOkToolCalls` /
 *     `computeTrailingAssistantUnresolvedToolCallCount`):
 *
 *       * `null` — the trailing assistant turn has no outstanding ok
 *         tool call. Plain `send()` / `sendStream()` are the only
 *         valid entry points; `sendToolResult*()` throws because
 *         there is nothing for the result to resolve.
 *       * `1` — exactly one outstanding ok tool call. Plain `send()` /
 *         `sendStream()` throw (they would orphan the call);
 *         `sendToolResult*()` is the sole valid forward step and
 *         dispatches the tool result through the native session.
 *       * `>1` — a multi-tool-call fan-out that the chat-session API
 *         cannot progress incrementally (each `sendToolResult*` would
 *         re-open the assistant turn and weave new replies between
 *         the sibling results). Both `send()` / `sendStream()` and
 *         `sendToolResult*()` throw. The only valid recovery is
 *         `reset()` or `primeHistory()` + `startFromHistory*()` with
 *         a fully-resolved conversation — there is no "advance past
 *         the broken turn" path. This mirrors the native ChatML delta
 *         format which would otherwise silently corrupt multi-call
 *         conversations.
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
 *   - **Cold-restart primitives.** `primeHistory()` plus
 *     `startFromHistory()` / `startFromHistoryStream()` let a caller
 *     seed a fresh session with an externally-reconstructed history
 *     (e.g. a server `ResponseStore` chain) and replay it through the
 *     native `chatSessionStart` path without going through `send()`.
 *     These are intended for server-side `SessionRegistry` cache-miss
 *     cold-start; normal usage stays on `send` / `sendStream` /
 *     `sendToolResult` / `reset`.
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
import type { ChatConfig, ChatMessage, ChatResult, ToolCall, ToolCallResult } from '@mlx-node/core';

import type { ChatStreamEvent } from './stream.js';

/**
 * Convert the parsed `ToolCallResult[]` emitted by the native chat
 * pipeline into the `ToolCall[]` shape expected by
 * `ChatMessage.toolCalls` (and, by extension, the jinja chat
 * templates on cold replay).
 *
 * Two shape differences to bridge:
 *
 *   1. `ToolCallResult.arguments` is `Record<string, unknown> | string`
 *      (already parsed by the native parser when status is "ok",
 *      preserved as the original string on parse failure). The
 *      `ChatMessage.toolCalls` contract is `arguments: string`, and
 *      the native tokenizer's `render_chat_template` pre-parses that
 *      string back into a `serde_json::Value` before handing it to
 *      jinja. We therefore `JSON.stringify` any non-string argument
 *      so the round-trip is lossless. Strings are passed through
 *      verbatim so a failed-to-parse payload retains its original
 *      bytes (the template then sees it as a quoted string, which is
 *      the safest available fallback).
 *   2. Only `status === "ok"` calls carry a well-formed
 *      `(name, arguments)` pair — the other statuses (`invalid_json`,
 *      `missing_name`, `parse_error`) are informational diagnostics
 *      that the native parser emits for observability and that the
 *      downstream chat template has no way to render. Preserving them
 *      on the replay path would inject garbage tool-call tags into
 *      the jinja output. We filter to `ok` entries only — matching the
 *      filter every other consumer (server response mapper, tool-use
 *      examples, README guidance) already applies.
 *
 * Returns `undefined` when the input is absent or yields no `ok`
 * entries so the assistant `ChatMessage` stays minimal (no empty
 * `toolCalls: []` field polluting the history).
 */
function toAssistantToolCalls(toolCalls: readonly ToolCallResult[] | undefined): ToolCall[] | undefined {
  if (!toolCalls || toolCalls.length === 0) return undefined;
  const out: ToolCall[] = [];
  for (const tc of toolCalls) {
    if (tc.status !== 'ok') continue;
    const argsStr = typeof tc.arguments === 'string' ? tc.arguments : JSON.stringify(tc.arguments);
    out.push({ id: tc.id, name: tc.name, arguments: argsStr });
  }
  return out.length > 0 ? out : undefined;
}

/**
 * Build an assistant `ChatMessage` from a just-completed turn's
 * decoded text + tool-call list. The assistant entry is appended to
 * `this.history` after every successful turn and is later read back
 * by the native `chatSessionStart` cold-replay path (image-change
 * mid-session restart, `startFromHistory*`, server-side
 * `SessionRegistry` cache-miss rebuild). Dropping the `toolCalls`
 * field here would orphan any subsequent `{role: 'tool', ...}`
 * entries on replay — the jinja template would render a
 * `<tool_response>` for a call that was never declared on the
 * preceding assistant turn, corrupting the conversation structure
 * and changing model behavior after a restart.
 */
function buildAssistantMessage(text: string, toolCalls: readonly ToolCallResult[] | undefined): ChatMessage {
  const calls = toAssistantToolCalls(toolCalls);
  if (calls) {
    return { role: 'assistant', content: text, toolCalls: calls };
  }
  return { role: 'assistant', content: text };
}

/**
 * Count the `ok`-status tool calls in a `ChatResult.toolCalls` /
 * terminal stream chunk. Used to detect the unsupported multi-call
 * fan-out pattern — the chat-session API only serves one tool call
 * per assistant turn because each `sendToolResult` dispatch
 * immediately re-opens the assistant turn, so a second result would
 * land after a new assistant reply and corrupt the conversation
 * structure. Non-`ok` entries (`parse_error`, `invalid_json`, etc.)
 * are ignored because the caller cannot respond to them anyway.
 */
function countOkToolCalls(toolCalls: readonly ToolCallResult[] | undefined): number {
  if (!toolCalls || toolCalls.length === 0) return 0;
  let n = 0;
  for (const c of toolCalls) {
    if (c.status === 'ok') n++;
  }
  return n;
}

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
  /**
   * The optional `signal` parameter on every streaming entry point is
   * plumbed into the `_runChatStream` fast-abort path in the wrapper
   * implementations. Callers that need client-disconnect-aware
   * cancellation (e.g. HTTP endpoints flipping an AbortController on
   * socket close) can attach one here and the native decode unwinds
   * at the next safepoint via `ChatStreamHandle.cancel()`. Non-signal
   * callers (the common direct-use path) just omit it.
   */
  chatStreamSessionStart(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent>;
  chatStreamSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent>;
  chatStreamSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
    signal?: AbortSignal,
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
  /**
   * Optional AbortSignal plumbed into the streaming fast-abort path.
   *
   * Only honored by the streaming entry points (`sendStream`,
   * `sendToolResultStream`, `startFromHistoryStream`) — the
   * non-streaming `send` / `sendToolResult` / `startFromHistory`
   * calls have NO native cancel surface, so a signal passed to them
   * is ignored. Pass one here and the inner `_runChatStream`
   * adapter wakes from `waitForItem()` on abort, calls
   * `handle.cancel()` on the native handle, and unwinds the stream
   * without throwing an AbortError — the outer consumer's `for await`
   * just ends early. Intended for HTTP endpoints that flip a
   * controller on `res.once('close', …)` so client disconnect stops
   * the native decode at the next safepoint rather than running it
   * to completion under the per-model mutex.
   */
  signal?: AbortSignal;
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

  /**
   * Count of `ok` tool calls emitted by the prior assistant turn, or
   * `null` when the prior turn produced none. Gates every continuation
   * entry point on the tool-call resolution invariant because each
   * native `chat_session_continue*` dispatch re-opens the assistant
   * turn:
   *
   *   - A plain text `send` / `sendStream` after ANY outstanding tool
   *     call would orphan the call(s) by weaving a fresh user turn
   *     between the assistant's `tool_call` and any response.
   *   - A `sendToolResult` / `sendToolResultStream` is only servable
   *     when exactly one tool call is outstanding. A multi-call
   *     fan-out (`> 1`) cannot be resolved one result at a time — the
   *     siblings would be separated by fresh assistant replies — so
   *     those entry points also reject.
   *
   * Cleared on every successful commit whose new turn emits zero `ok`
   * tool calls, and on `reset()`. See `assertCanSendPlain` /
   * `assertCanSendToolResult` for the per-entry-point gate logic.
   */
  private unresolvedOkToolCallCount: number | null = null;

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
   * Count of `ok` tool calls from the most recent assistant turn, or
   * `null` when the trailing turn produced none. Non-null means the
   * session is parked on an unresolved tool-call turn and the only
   * forward-progress move is `sendToolResult*()` against one of the
   * outstanding ids — and only when the count is exactly 1. A
   * multi-call fan-out (`> 1`) cannot be served by the chat-session
   * API at all; server endpoints should pre-check this getter and
   * route around a fan-out via `reset()` + `primeHistory()` +
   * `startFromHistory()` cold replay that resolves every sibling in
   * one atomic jinja render.
   *
   * The flag updates after every successful `send` / `sendStream` /
   * `sendToolResult` / `sendToolResultStream` / `startFromHistory*`
   * commit, and after `primeHistory()` (from the trailing assistant
   * message's `toolCalls.length`). `reset()` clears it.
   */
  get pendingUnresolvedToolCallCount(): number | null {
    return this.unresolvedOkToolCallCount;
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
    this.assertCanSendPlain('send');
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
      this.history.push(buildAssistantMessage(result.text, result.toolCalls));
      this.turnCount++;
      this.recordToolCallFanout(result.toolCalls);
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
    this.assertCanSendPlain('sendStream');
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
        yield* this.runStartStreamPath(
          userMessage,
          opts.images,
          newImagesKey,
          imageChanged,
          isFirstTurn,
          mergedConfig,
          opts.signal,
        );
        return;
      }

      // Delta continue stream: text-only.
      let sawFinal = false;
      let accumulated = '';
      let finalRaw: string | null = null;
      let finalToolCalls: readonly ToolCallResult[] | undefined;
      try {
        for await (const event of this.model.chatStreamSessionContinue(userMessage, null, mergedConfig, opts.signal)) {
          if (event.done) {
            if (event.finishReason !== 'error') {
              sawFinal = true;
              finalRaw = event.text;
              finalToolCalls = event.toolCalls;
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
          this.history.push(buildAssistantMessage(finalRaw ?? accumulated, finalToolCalls));
          this.turnCount++;
          this.recordToolCallFanout(finalToolCalls);
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
   * Rejects if the prior assistant turn emitted more than one `ok`
   * tool call: the chat-session API only supports exactly one tool
   * call per assistant turn because each `sendToolResult` dispatch
   * immediately re-opens the assistant turn, so responding to the
   * remaining calls would interleave new assistant replies between
   * the results and corrupt the conversation structure. Callers that
   * hit this must tighten the prompt / tool spec or reset the
   * session.
   *
   * Appends a `{ role: 'tool', ... }` message to history on success.
   */
  async sendToolResult(toolCallId: string, content: string, opts: { config?: ChatConfig } = {}): Promise<ChatResult> {
    if (this.inFlight) {
      throw new Error('ChatSession: concurrent send() not allowed; await the previous call first');
    }
    this.assertCanSendToolResult('sendToolResult');
    this.inFlight = true;
    try {
      const mergedConfig = this.mergeConfig(opts.config);
      const result = await this.model.chatSessionContinueTool(toolCallId, content, mergedConfig);
      this.history.push({ role: 'tool', content, toolCallId });
      this.history.push(buildAssistantMessage(result.text, result.toolCalls));
      this.turnCount++;
      this.recordToolCallFanout(result.toolCalls);
      return result;
    } finally {
      this.inFlight = false;
    }
  }

  /** Streaming variant of {@link ChatSession#sendToolResult}. */
  async *sendToolResultStream(
    toolCallId: string,
    content: string,
    opts: { config?: ChatConfig; signal?: AbortSignal } = {},
  ): AsyncGenerator<ChatStreamEvent> {
    if (this.inFlight) {
      throw new Error('ChatSession: concurrent send() not allowed; await the previous call first');
    }
    this.assertCanSendToolResult('sendToolResultStream');
    this.inFlight = true;
    try {
      const mergedConfig = this.mergeConfig(opts.config);
      let sawFinal = false;
      let accumulated = '';
      let finalRaw: string | null = null;
      let finalToolCalls: readonly ToolCallResult[] | undefined;
      try {
        for await (const event of this.model.chatStreamSessionContinueTool(
          toolCallId,
          content,
          mergedConfig,
          opts.signal,
        )) {
          if (event.done) {
            if (event.finishReason !== 'error') {
              sawFinal = true;
              finalRaw = event.text;
              finalToolCalls = event.toolCalls;
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
          this.history.push(buildAssistantMessage(finalRaw ?? accumulated, finalToolCalls));
          this.turnCount++;
          this.recordToolCallFanout(finalToolCalls);
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
    this.unresolvedOkToolCallCount = null;
  }

  /**
   * Prime the session history without running inference.
   *
   * Used by the server-side `SessionRegistry` cold-start fallback: when
   * a request arrives with a `previous_response_id` that the cache has
   * missed, the endpoint reconstructs the full conversation from the
   * `ResponseStore` and primes a fresh session with it, then calls
   * `startFromHistory()` to replay it through the native KV cache.
   *
   * Rejects if the session is in flight or has already taken a turn.
   * Replaces the internal history with a shallow copy of `messages`.
   */
  primeHistory(messages: ChatMessage[]): void {
    if (this.inFlight) {
      throw new Error('ChatSession: cannot primeHistory() while a send() is in flight');
    }
    if (this.turnCount > 0) {
      throw new Error('ChatSession: primeHistory() can only be called on a fresh session (turn 0)');
    }
    this.history = messages.slice();
    // Derive the unresolved-tool-call guard from the trailing assistant
    // turn in the primed history so an immediately-post-prime session
    // exposes the same `pendingUnresolvedToolCallCount` state a live
    // session would have been in at that point of the conversation.
    // This lets the server endpoint layer (and any other caller)
    // pre-check the guard before starting cold replay and route around
    // unresolved turns instead of letting `startFromHistory*()` blindly
    // advance past them. The flag is reset on commit in both sync and
    // streaming start-from-history paths based on the new assistant
    // reply, which is the correct semantics for the post-replay current
    // position.
    this.unresolvedOkToolCallCount = this.computeTrailingAssistantUnresolvedToolCallCount();
    // lastImagesKey stays null until startFromHistory() / send() runs —
    // the trailing-images hydration happens at commit time.
  }

  /**
   * Run a cold-start `chatSessionStart` using the currently primed
   * history.
   *
   * Intended pairing with {@link primeHistory}: call
   * `primeHistory(fullHistory)` first, then `startFromHistory()` to
   * replay the conversation through the native chat-session API. The
   * final history entry must be a user or tool turn — this is what the
   * native side treats as the "current input" to generate against.
   *
   * Pushes the assistant reply onto the history, advances `turnCount`
   * to 1, and computes `lastImagesKey` from the most recent user
   * message that carries images (so subsequent text-only continues
   * stay on the delta path, and subsequent image turns correctly
   * trigger restart).
   */
  async startFromHistory(config?: ChatConfig): Promise<ChatResult> {
    if (this.inFlight) {
      throw new Error('ChatSession: cannot startFromHistory() while a send() is in flight');
    }
    if (this.turnCount > 0) {
      throw new Error('ChatSession: startFromHistory() can only be called on a fresh session');
    }
    if (this.history.length === 0) {
      throw new Error('ChatSession: startFromHistory() requires a primed history');
    }
    this.inFlight = true;
    try {
      const mergedConfig = this.mergeConfig(config);
      const result = await this.model.chatSessionStart(this.history.slice(), mergedConfig);
      this.history.push(buildAssistantMessage(result.text, result.toolCalls));
      this.turnCount++;
      this.lastImagesKey = this.computeTrailingImagesKey();
      this.recordToolCallFanout(result.toolCalls);
      return result;
    } finally {
      this.inFlight = false;
    }
  }

  /**
   * Streaming counterpart to {@link startFromHistory}.
   *
   * Iterates `model.chatStreamSessionStart(history.slice(), config)`,
   * accumulates text, and only commits history + `turnCount` +
   * `lastImagesKey` in the `finally` block when a successful terminal
   * chunk was observed (`done: true` with non-error finishReason).
   * Because history is primed (not appended to), rollback on failure
   * is a no-op: the primed state stays intact so the caller can retry.
   */
  async *startFromHistoryStream(config?: ChatConfig, signal?: AbortSignal): AsyncGenerator<ChatStreamEvent> {
    if (this.inFlight) {
      throw new Error('ChatSession: cannot startFromHistoryStream() while a send() is in flight');
    }
    if (this.turnCount > 0) {
      throw new Error('ChatSession: startFromHistoryStream() can only be called on a fresh session');
    }
    if (this.history.length === 0) {
      throw new Error('ChatSession: startFromHistoryStream() requires a primed history');
    }
    this.inFlight = true;
    try {
      const mergedConfig = this.mergeConfig(config);
      const historySnapshot = this.history.slice();
      let sawFinal = false;
      let accumulated = '';
      let finalRaw: string | null = null;
      let finalToolCalls: readonly ToolCallResult[] | undefined;
      try {
        for await (const event of this.model.chatStreamSessionStart(historySnapshot, mergedConfig, signal)) {
          if (event.done) {
            if (event.finishReason !== 'error') {
              sawFinal = true;
              finalRaw = event.text;
              finalToolCalls = event.toolCalls;
            }
          } else {
            accumulated += event.text;
          }
          yield event;
        }
      } finally {
        // finally runs on normal completion, mid-stream throw, caller
        // `break` (iterator.return() short-circuits the yield), and
        // error-finish chunks alike. The primed history is only
        // mutated on a successful commit — on any non-success exit,
        // the primed state is left intact so the caller can retry.
        if (sawFinal) {
          this.history.push(buildAssistantMessage(finalRaw ?? accumulated, finalToolCalls));
          this.turnCount++;
          this.lastImagesKey = this.computeTrailingImagesKey();
          this.recordToolCallFanout(finalToolCalls);
        }
      }
    } finally {
      this.inFlight = false;
    }
  }

  // -------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------

  /**
   * Gate plain-text continuation entry points (`send`, `sendStream`)
   * on the tool-call resolution invariant. Any outstanding `ok` tool
   * call from the prior assistant turn — single or multi — makes a
   * plain text continuation unsafe: the native chat-session API
   * re-opens the assistant turn on each continue, so a new user delta
   * would weave a fresh user message between the assistant's
   * `tool_call` and any response, orphaning the call. Callers must
   * resolve outstanding calls via `sendToolResult*()` (single-call
   * case) or re-enter via `reset()` + `primeHistory()` +
   * `startFromHistory()` with a resolved conversation (multi-call
   * fan-out). `reset()` clears the flag and `startFromHistory*`
   * overwrites it via `recordToolCallFanout` on the new response, so
   * legitimate recovery paths are unaffected.
   */
  private assertCanSendPlain(entryPoint: string): void {
    const n = this.unresolvedOkToolCallCount;
    if (n !== null) {
      const plural = n === 1 ? '' : 's';
      const followUp =
        n > 1
          ? `multi-call fan-outs cannot be served one result at a time — re-enter through reset() + primeHistory() + startFromHistory() with a conversation that resolves every sibling in one atomic replay`
          : `resolve the outstanding call via sendToolResult()`;
      throw new Error(
        `ChatSession.${entryPoint}: previous assistant turn has ${n} unresolved ok tool call${plural}; ` +
          `a plain text continuation would orphan the call${plural} by weaving a new user turn between the ` +
          `assistant's tool_call and any response. ${followUp}, reset() the session, or re-enter through ` +
          `primeHistory() + startFromHistory() with a resolved conversation.`,
      );
    }
  }

  /**
   * Gate tool-result entry points (`sendToolResult`,
   * `sendToolResultStream`) on the single-tool-call-per-turn
   * invariant. Exactly one outstanding tool call is servable — that
   * is the case these methods exist for.
   *
   * Zero outstanding calls (`null`) is also unservable: without a
   * preceding assistant turn that emitted a tool call, a tool-result
   * dispatch would synthesize a `<tool_response>` delta for a call
   * that never existed, corrupting the conversation structure. The
   * native backends do not authenticate `tool_call_id` against prior
   * state — several simply append the tool-response delta verbatim —
   * so rejecting here is the only gate that prevents forged tool
   * state from reaching the model. Callers that want to start a
   * conversation on a resolved tool turn must prime an unresolved
   * single-call assistant turn via `primeHistory()` +
   * `startFromHistory()` first.
   *
   * A multi-call fan-out (`> 1`) cannot be resolved one result at a
   * time because each `sendToolResult` dispatch immediately re-opens
   * the assistant turn, so responding to the siblings would
   * interleave new assistant replies between the results.
   */
  private assertCanSendToolResult(entryPoint: string): void {
    const n = this.unresolvedOkToolCallCount;
    if (n === null) {
      throw new Error(
        `ChatSession.${entryPoint}: no outstanding ok tool call on the previous assistant turn. ` +
          `Tool-result entry points can only be called when the model has just emitted exactly one ` +
          `ok tool call that has not yet been resolved — dispatching a tool result against an empty ` +
          `or already-resolved turn would synthesize a <tool_response> delta for a call that never ` +
          `existed and corrupt the conversation structure. Call send() / sendStream() for plain user ` +
          `turns, or re-enter through primeHistory() + startFromHistory() with a conversation that ` +
          `ends on an unresolved single-call assistant turn.`,
      );
    }
    if (n > 1) {
      throw new Error(
        `ChatSession.${entryPoint}: previous assistant turn emitted ${n} ok tool calls; ` +
          `the chat-session API only supports exactly one tool call per assistant turn because each tool-result ` +
          `call immediately re-opens the assistant turn — responding to the siblings would interleave new assistant ` +
          `replies between the results. Tighten the prompt / tool spec so the model produces at most one call per ` +
          `turn, reset() the session, or re-enter through primeHistory() + startFromHistory() with a resolved ` +
          `conversation.`,
      );
    }
  }

  /**
   * Inspect a just-committed turn's tool calls and store the count of
   * `ok` entries in `unresolvedOkToolCallCount`. Any non-zero count
   * parks the session on an unresolved tool-call turn, which gates
   * the next entry point:
   *
   *   - count === 0 → flag is `null`: `send`/`sendStream` ok,
   *     `sendToolResult*` throws (no outstanding call to resolve)
   *   - count === 1 → `send`/`sendStream` throw; `sendToolResult*` ok
   *   - count >  1 → every entry point throws (fan-out unservable)
   *
   * See `assertCanSendPlain` / `assertCanSendToolResult` for the full
   * rationale.
   */
  private recordToolCallFanout(toolCalls: readonly ToolCallResult[] | undefined): void {
    const n = countOkToolCalls(toolCalls);
    this.unresolvedOkToolCallCount = n > 0 ? n : null;
  }

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
      this.history.push(buildAssistantMessage(result.text, result.toolCalls));
      this.turnCount++;
      this.lastImagesKey = newImagesKey;
      this.recordToolCallFanout(result.toolCalls);
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
    signal: AbortSignal | undefined,
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
    let finalToolCalls: readonly ToolCallResult[] | undefined;
    // Snapshot the history before dispatch — see `runStartPath` for
    // the rationale.
    const historySnapshot = this.history.slice();
    try {
      for await (const event of this.model.chatStreamSessionStart(historySnapshot, config, signal)) {
        if (event.done) {
          if (event.finishReason !== 'error') {
            sawFinal = true;
            finalRaw = event.text;
            finalToolCalls = event.toolCalls;
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
        this.history.push(buildAssistantMessage(finalRaw ?? accumulated, finalToolCalls));
        this.turnCount++;
        this.lastImagesKey = newImagesKey;
        this.recordToolCallFanout(finalToolCalls);
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

  /**
   * Walk the history backward to find the most recent user message
   * with images and return its FNV-1a key. Used by
   * {@link startFromHistory} and {@link startFromHistoryStream} to
   * hydrate `lastImagesKey` after a cold replay, so subsequent delta
   * continues correctly detect image changes.
   */
  private computeTrailingImagesKey(): string | null {
    for (let i = this.history.length - 1; i >= 0; i--) {
      const msg = this.history[i];
      if (msg?.role === 'user' && msg.images && msg.images.length > 0) {
        return computeImagesKey(msg.images);
      }
    }
    return null;
  }

  /**
   * Derive the post-prime value of `unresolvedOkToolCallCount` from
   * the primed history. Walks backward to the most recent assistant
   * turn, then walks forward from that assistant to the end of history
   * subtracting any `tool:` message that references one of the turn's
   * `call_id`s. A fully-resolved history (every outstanding id matched
   * by a sibling `tool:` message) returns `null`; any leftover count is
   * the number of still-unresolved tool calls.
   *
   * Matches the runtime `recordToolCallFanout` semantics on the hot
   * path: zero unresolved → `null` (no pending obligation); one →
   * `1` (servable via `sendToolResult*()` only); two or more → the
   * count itself (unservable fan-out — must be resolved via cold
   * replay). The distinction between "ok" vs. other statuses only
   * exists in the live `ToolCallResult[]` emitted by the native side —
   * the persisted `ChatMessage.toolCalls` on an assistant message only
   * carries successfully parsed calls (i.e. what would have been "ok"
   * in the original live turn), so counting the array length is
   * equivalent. Tool calls whose `id` is missing or empty can't be
   * matched against subsequent `tool_call_id`s, so in that case we
   * fall back to returning the raw `calls.length` (err safe).
   */
  private computeTrailingAssistantUnresolvedToolCallCount(): number | null {
    let assistantIdx = -1;
    for (let i = this.history.length - 1; i >= 0; i--) {
      if (this.history[i]?.role === 'assistant') {
        assistantIdx = i;
        break;
      }
    }
    if (assistantIdx === -1) return null;

    const assistant = this.history[assistantIdx]!;
    const calls = assistant.toolCalls ?? [];
    if (calls.length === 0) return null;

    const outstanding = new Set<string>();
    let missingIdCount = 0;
    for (const tc of calls) {
      if (typeof tc.id === 'string' && tc.id.length > 0) {
        outstanding.add(tc.id);
      } else {
        missingIdCount++;
      }
    }
    // Untracked calls (no id) can't be matched against resolutions —
    // err safe by reporting the raw count.
    if (missingIdCount > 0) return calls.length;

    for (let j = assistantIdx + 1; j < this.history.length; j++) {
      const msg = this.history[j];
      if (msg?.role === 'tool' && typeof msg.toolCallId === 'string' && msg.toolCallId.length > 0) {
        outstanding.delete(msg.toolCallId);
      }
    }
    return outstanding.size > 0 ? outstanding.size : null;
  }
}
