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
import { createHash } from 'node:crypto';

import type { ChatConfig, ChatMessage, ChatResult, ToolCall, ToolCallResult, ToolDefinition } from '@mlx-node/core';

import type { ChatStreamEvent } from './stream.js';

/**
 * Typed prefix the native delta path uses to reject a text-only
 * continuation while the session still holds image/audio KV state
 * (gemma4 raises this after a media turn). The native session refuses
 * to advance the cheap delta on top of media KV, so the session layer
 * recognizes this exact prefix and transparently replays the whole
 * conversation through the cold start path instead of surfacing the
 * raw error to the caller.
 *
 * MUST stay byte-for-byte identical to the Rust constant
 * `IMAGE_CHANGE_RESTART_PREFIX` in
 * `crates/mlx-core/src/engine/cache.rs` — it is not exported across the
 * NAPI boundary, so the two literals are kept in sync by hand. The
 * native message starts with this prefix and is delivered as the
 * `Error.message`: on the sync delta path as a rejected promise, and on
 * the streaming delta path as a thrown error on the generator's first
 * iteration (the native worker-thread sink error is re-thrown by the
 * `packages/lm/src/stream.ts` bridge before any chunk is yielded).
 */
const IMAGE_CHANGE_RESTART_PREFIX = 'IMAGE_CHANGE_REQUIRES_SESSION_RESTART:';

/**
 * Whether `err` is the native media-held delta rejection (see
 * {@link IMAGE_CHANGE_RESTART_PREFIX}). The native message begins with
 * the literal prefix and reaches both the sync and streaming bridges
 * unwrapped (NAPI surfaces `Error.from_reason` as `Error.message`
 * verbatim), so a `startsWith` match is exact.
 */
function isMediaHeldRestartError(err: unknown): boolean {
  return err instanceof Error && err.message.startsWith(IMAGE_CHANGE_RESTART_PREFIX);
}

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
  /**
   * Optional non-generating chat-template tokenizer. Exposed by
   * wrappers that can count prompt tokens without running inference
   * (used by Anthropic `/v1/messages/count_tokens`).
   */
  applyChatTemplate?(
    messages: ChatMessage[],
    addGenerationPrompt?: boolean | null,
    tools?: ToolDefinition[] | null,
    enableThinking?: boolean | null,
  ): Promise<Uint32Array> | Uint32Array;
  chatSessionStart(messages: ChatMessage[], config?: ChatConfig | null): Promise<ChatResult>;
  chatSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    audio: Uint8Array[] | null,
    config?: ChatConfig | null,
  ): Promise<ChatResult>;
  chatSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
    isError?: boolean | null,
  ): Promise<ChatResult>;
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
    audio: Uint8Array[] | null,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent>;
  chatStreamSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
    signal?: AbortSignal,
    isError?: boolean | null,
  ): AsyncGenerator<ChatStreamEvent>;
  resetCaches(): void;
  /**
   * Whether the underlying native model has the block-paged KV cache
   * adapter (`PagedKVCacheAdapter` + `BlockAllocator` + `LayerKVPool`)
   * active.
   *
   * `true` iff the adapter was successfully constructed at load time
   * (driven by the per-model `use_block_paged_cache` config flag, which
   * defaults to ON for Qwen3 + LFM2 after parity verification and OFF
   * for Gemma4 + Qwen3.5 + Qwen3.5 MoE pending parity validation; also
   * always `false` on Qwen3.5 VLM checkpoints where
   * `set_vision_encoder` rejects when the adapter is populated).
   *
   * When `true`, the native cache reuses SYS blocks across requests via
   * content-addressing in the `BlockAllocator`'s prefix-hash table —
   * the JS-side warm slot in
   * `SessionRegistry.getOrCreateWarmAny(requestedSystem)` becomes
   * redundant for stateless `/v1/messages` traffic. The server
   * endpoint reads this getter to decide whether to allocate a fresh
   * `ChatSession` per request (paged-active) or to lease the warm slot
   * (non-paged); see `packages/server/src/endpoints/messages.ts`.
   *
   * Optional on the structural interface so models that pre-date the
   * NAPI getter (notably `QianfanOCRModel` from `@mlx-node/vlm`, which
   * has no paged-adapter wiring) still satisfy the type contract — a
   * missing getter is treated as `false` (not paged) by callers.
   * Surfaced as a synchronous method on every native wrapper that DOES
   * support paged so the routing decision in the server doesn't need a
   * model-thread roundtrip per request — the value is captured at load
   * time and never changes for a given model instance.
   */
  hasBlockPagedCache?(): boolean;
  /**
   * MTP: whether the underlying native model can run speculative
   * decoding. Surfaced by `Qwen3_5Model` / `Qwen3_5MoeModel` (an MTP
   * head shipped in the checkpoint and loaded by persistence) and by
   * `Gemma4Model` (an external draft model — DSpark or Google gemma-4
   * assistant, auto-detected from the draft's config.json — attached
   * via `loadModel` / `loadSession` `draftModelPath`; NOT in-checkpoint
   * MTP heads); all other native wrappers omit the method, in
   * which case callers treat a missing getter as `false` (no MTP).
   *
   * When `true`, {@link ChatSession#mergeConfig} auto-defaults the
   * per-request `enableMtp` flag to `true` — the speculative-decode
   * path takes over unless the caller explicitly opts out by passing
   * `enableMtp: false` in their `ChatConfig` overlay. When `false`
   * (or the method is missing), `enableMtp` is left untouched.
   *
   * Synchronous on every supporting wrapper so the auto-default check
   * doesn't need a model-thread roundtrip per call — the value is
   * captured at load time and never changes for a given model
   * instance.
   *
   * ## Companion `ChatConfig` knobs (only meaningful when `enableMtp` is on)
   *
   * Two related `ChatConfig` fields tune the speculative-decode loop.
   * They are forwarded verbatim to the native side via
   * {@link SendOptions.config} (per-call overlay) or
   * {@link ChatSessionOptions.defaultConfig} (session default), and
   * `mergeConfig` shallow-merges per-call over per-session so an
   * explicit per-send value always wins over the session default for
   * the same field.
   *
   * - **`mtpDepth`** — pins the MTP draft depth per speculative cycle.
   *   On Qwen3.5 native MTP heads it is clamped to `[1, 5]` by the
   *   verify FFI contract, and when unset native code currently pins
   *   depth 1. Setting `mtpDepth` explicitly pins that value unless
   *   the caller also passes `mtpAdaptiveDepth: true` to opt into
   *   adaptive depth with the supplied maximum/seed. Gemma4 external
   *   drafts resolve the field per draft variant instead — see the
   *   Gemma4 section below.
   * - **`mtpAdaptiveDepth`** — toggles the adaptive depth policy.
   *   Defaults to OFF. When ON, the default mode runs a 5-state machine
   *   (`Explore` → `Full` → {`NeighborProbe` | `Reduced` → `Probe`})
   *   with per-depth EMA tracking of
   *   `accepted_tokens / cycle_wall_ns` and picks the depth that
   *   maximizes that rate (DFlash-style, EMA decay α=0.3, drop-back
   *   threshold 0.75). `MLX_MTP_ADAPTIVE_DEPTH_MODE=expected-value`
   *   instead uses the MTPLX-style intra-cycle expected-value gate; by
   *   default it stops at its base depth, with deeper expansion kept
   *   research-only behind `MLX_MTP_EV_ALLOW_DEEPEN=1`.
   *   An explicit `false` always wins, pinning the chosen `mtpDepth` for
   *   every cycle. When `enableMtp` is false (or the model has no MTP
   *   head) the field is ignored.
   *
   * The defaults for both `mtpDepth` and `mtpAdaptiveDepth` are
   * applied on the native side (see
   * `crates/mlx-core/src/models/qwen3_5/chat_common.rs` MTP runtime
   * flag inventory), so omitting them from `defaultConfig` /
   * `SendOptions.config` is the recommended path for callers that
   * just want speculative decoding "on with sensible defaults".
   *
   * ## Gemma4 external drafts (`draftModelPath`)
   *
   * Gemma4 reinterprets the knobs per draft variant (resolved in
   * `gemma4/model.rs` `resolve_params`, always from the RAW config
   * value — the engine's central `[1, 5]` clamp is an MTP-head
   * contract that does not apply to external drafts):
   *
   * - **DSpark**: an unset `mtpDepth` runs full draft blocks (the
   *   draft checkpoint's block size — 7 tokens on
   *   `dspark_gemma4_12b_block7`), and an explicit `mtpDepth` acts as
   *   a CAP on that block (clamped to `[1, blockSize]`).
   * - **Assistant** (Google `gemma-4-*-it-assistant`): chained AR
   *   drafting has no checkpoint-pinned block size — an unset
   *   `mtpDepth` drafts 3 tokens per cycle (`ASSISTANT_DEFAULT_DEPTH`,
   *   a quality/latency tradeoff, not a checkpoint contract), and an
   *   explicit `mtpDepth` clamps to `[1, 8]` (`ASSISTANT_MAX_DEPTH`).
   *
   * `mtpAdaptiveDepth` is ignored for BOTH variants — neither external
   * draft loop has an adaptive depth policy. Qwen3.5 native-MTP
   * semantics above are unchanged.
   */
  hasMtpWeights?(): boolean;
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
   * Optional audio bytes (encoded WAV) attached to this user turn. When
   * the audio set differs from the session's current `lastAudioKey`, the
   * session forcibly restarts via `chatSessionStart` (mirrors `images`).
   * Only the unified Gemma 4 audio checkpoint consumes this.
   */
  audio?: Uint8Array[];
  /**
   * Per-call `ChatConfig` overlay applied on top of the session's
   * `defaultConfig`. `reuseCache` is always forced on regardless of
   * what the caller passes. The overlay is shallow-merged on top of
   * the session default, so per-call values always win over per-session
   * values for the same field — including the speculative-decode
   * knobs `enableMtp`, `mtpDepth`, and `mtpAdaptiveDepth`. See
   * {@link SessionCapableModel.hasMtpWeights} for the full MTP knob
   * surface and default-resolution rules.
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
   * top of this, and `reuseCache` is forced on. Speculative-decode
   * knobs (`enableMtp`, `mtpDepth`, `mtpAdaptiveDepth`) can be parked
   * here as session-wide defaults and overridden per call via
   * {@link SendOptions.config}; see
   * {@link SessionCapableModel.hasMtpWeights} for the MTP knob surface
   * and default-resolution rules.
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
 * decide whether to restart the server-side session, so any
 * collision-resistant digest is sufficient. We use SHA-256 (native
 * `node:crypto`) with a length-prefixed framing so different image
 * counts and different byte lengths cannot collide by accident.
 *
 * Implementation note: kept fully sync + self-contained so
 * `send()` can stay synchronous in its routing decision. `node:crypto`
 * is a Node built-in, so this adds no external runtime dependency
 * beyond `@mlx-node/core` and the existing stream bridge.
 */
function computeImagesKey(images: Uint8Array[] | undefined): string | null {
  return computeByteListKey(images);
}

/**
 * Audio counterpart of {@link computeImagesKey}: a stable, order-sensitive
 * byte-identity key for a list of encoded audio buffers. Used by `send()` /
 * `sendStream()` to decide whether a new audio set must cold-restart the
 * server-side session. Shares the exact SHA-256 framing as the image key.
 */
function computeAudioKey(audio: Uint8Array[] | undefined): string | null {
  return computeByteListKey(audio);
}

/**
 * SHA-256 byte-identity key for a length-framed list of byte buffers.
 * Returns `null` for an empty/absent list so callers can distinguish
 * "no media" from "media changed". Shared by the image and audio keys.
 *
 * Uses `node:crypto`'s native SHA-256 rather than a hand-rolled JS hash
 * loop: hashing large image/audio buffers byte-at-a-time in JS is
 * 25-60x slower than the native digest (measured: 5MB ~105ms JS loop
 * vs ~1.8ms native) and this runs synchronously on the event loop
 * before any `await` in `send()`/`sendStream()`, so the JS loop's cost
 * was a real head-of-line-blocking stall for every other request
 * handled by the same process.
 */
function computeByteListKey(buffers: Uint8Array[] | undefined): string | null {
  if (!buffers || buffers.length === 0) return null;
  const hash = createHash('sha256');
  // Frame each buffer with a 4-byte little-endian length prefix (and a
  // leading count prefix) so `[ab, c]` and `[a, bc]` — and different
  // buffer counts — hash to distinct values.
  const prefix = new Uint8Array(4);
  const prefixView = new DataView(prefix.buffer);
  prefixView.setUint32(0, buffers.length, true);
  hash.update(prefix);
  for (const buf of buffers) {
    prefixView.setUint32(0, buf.byteLength, true);
    hash.update(prefix);
    hash.update(buf);
  }
  return hash.digest('hex');
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
   * to the server's KV cache (SHA-256; see `computeImagesKey`).
   * `null` when no images are cached. A `send()` whose new key
   * differs triggers a full `chatSessionStart` restart.
   */
  private lastImagesKey: string | null = null;

  /**
   * Hex-encoded byte-identity key of the audio set currently bound to the
   * server's KV cache (see {@link computeAudioKey}). `null` when no audio is
   * cached. A `send()` whose new key differs triggers a full
   * `chatSessionStart` restart — the audio counterpart of `lastImagesKey`.
   */
  private lastAudioKey: string | null = null;

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
      const newAudioKey = computeAudioKey(opts.audio);
      // Only an explicit NEW image/audio set can trigger a restart. Omitting
      // `images`/`audio` (key === null) is interpreted as "keep the current
      // media cache state" — the server-side cache already holds any prior
      // media context, so a text-only follow-up like "what about the
      // top-right?" can stay on the cheap delta path even after a media turn.
      const imageChanged = newImagesKey !== null && newImagesKey !== this.lastImagesKey;
      const audioChanged = newAudioKey !== null && newAudioKey !== this.lastAudioKey;
      const isFirstTurn = this.turnCount === 0;

      if (isFirstTurn || imageChanged || audioChanged) {
        return await this.runStartPath(
          userMessage,
          opts.images,
          opts.audio,
          imageChanged || audioChanged,
          isFirstTurn,
          mergedConfig,
        );
      }

      // Delta continue: text-only, images/audio always null. The server
      // cache already holds all prior turns (including any media from an
      // earlier restart), so we only need to ship the new user string.
      let result: ChatResult;
      try {
        result = await this.model.chatSessionContinue(userMessage, null, null, mergedConfig);
      } catch (err) {
        if (!isMediaHeldRestartError(err)) {
          throw err;
        }
        // The native session holds media KV (gemma4 after an image/audio
        // turn) and refused the text delta. Transparently replay the full
        // conversation through the cold start path. The earlier media turn
        // already lives in `this.history`, so the start path re-renders it;
        // the trailing-media keys keep `lastImagesKey`/`lastAudioKey`
        // consistent across the replay. The delta path has NOT pushed
        // `userMessage` yet, so `runStartPath` pushing it adds no duplicate.
        return await this.runStartPath(userMessage, undefined, undefined, true, false, mergedConfig);
      }
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
      const newAudioKey = computeAudioKey(opts.audio);
      // Only an explicit NEW image/audio set can trigger a restart. Omitting
      // `images`/`audio` (key === null) is interpreted as "keep the current
      // media cache state" — the server-side cache already holds any prior
      // media context, so a text-only follow-up like "what about the
      // top-right?" can stay on the cheap delta path even after a media turn.
      const imageChanged = newImagesKey !== null && newImagesKey !== this.lastImagesKey;
      const audioChanged = newAudioKey !== null && newAudioKey !== this.lastAudioKey;
      const isFirstTurn = this.turnCount === 0;

      if (isFirstTurn || imageChanged || audioChanged) {
        yield* this.runStartStreamPath(
          userMessage,
          opts.images,
          opts.audio,
          imageChanged || audioChanged,
          isFirstTurn,
          mergedConfig,
          opts.signal,
        );
        return;
      }

      // Delta continue stream: text-only.
      let sawFinal = false;
      let accumulated = '';
      let accumulatedVisible = '';
      let finalRaw: string | null = null;
      let finalToolCalls: readonly ToolCallResult[] | undefined;
      // Set when the media-held rejection re-routes this turn through the
      // cold start stream. The replay path owns the history push, turnCount
      // increment, and media-key rehydration, so the commit `finally` below
      // must NOT also fire.
      let delegated = false;
      try {
        try {
          for await (const event of this.model.chatStreamSessionContinue(
            userMessage,
            null,
            null,
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
              if (event.isReasoning !== true) {
                accumulatedVisible += event.text;
              }
            }
            yield event;
          }
        } catch (err) {
          // The native session holds media KV (gemma4 after an image/audio
          // turn) and refused the text delta. The streaming bridge re-throws
          // that rejection on the first iteration, BEFORE any chunk is
          // emitted — the native guard fires ahead of any prefill, so
          // `!sawFinal && accumulated === ''` is guaranteed here. Replay the
          // full conversation through the cold start stream. Any non-prefix
          // error, or any error after tokens were already emitted, must
          // propagate unchanged.
          if (!isMediaHeldRestartError(err) || sawFinal || accumulated !== '') {
            throw err;
          }
          delegated = true;
          yield* this.runStartStreamPath(userMessage, undefined, undefined, true, false, mergedConfig, opts.signal);
          return;
        }
      } finally {
        // finally runs for normal completion, mid-stream throw,
        // caller `break` (which calls `iterator.return()` and
        // short-circuits the suspended yield), and error-finish
        // chunks alike. The delta path doesn't push to history until
        // commit, so the rollback branch is a no-op: nothing to
        // undo, and the native cache state is managed by the Rust
        // save_cache_state path on its own. When the media-held
        // rejection delegated to the replay stream, that path already
        // committed (or rolled back) — so this commit must stay off.
        if (sawFinal && !delegated) {
          this.history.push({ role: 'user', content: userMessage });
          this.history.push(buildAssistantMessage(finalRaw || accumulatedVisible, finalToolCalls));
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
   * `isError` is the structured tool-error signal. When `true`, the
   * native renderer prepends a short, model-facing error marker to
   * `content` inside the wire-format tool block so the model
   * receives a clear text-level cue that the tool result represents
   * a failure. The structured field is stored verbatim on the
   * appended `{ role: 'tool', ... }` history entry so cold-replay
   * (image-change restart, `startFromHistory*`, server-side
   * `SessionRegistry` cache-miss rebuild) re-renders the marker
   * consistently with the live turn. Defaults to `undefined` (no
   * marker). Pass through verbatim — we do NOT infer error from
   * `content`.
   *
   * Appends a `{ role: 'tool', ... }` message to history on success.
   */
  async sendToolResult(
    toolCallId: string,
    content: string,
    opts: { isError?: boolean; config?: ChatConfig } = {},
  ): Promise<ChatResult> {
    if (this.inFlight) {
      throw new Error('ChatSession: concurrent send() not allowed; await the previous call first');
    }
    this.assertCanSendToolResult('sendToolResult');
    this.inFlight = true;
    try {
      const { isError, config } = opts;
      const mergedConfig = this.mergeConfig(config);
      const toolMsg: ChatMessage = { role: 'tool', content, toolCallId, isError };
      // A cold native session (turnCount===0) has no live KV to delta
      // against — the typical cause is an interrupted media-held replay
      // whose rollback wiped the cache and reset the counter while
      // leaving the unresolved tool-call flag set. Mirror `send()`'s
      // turn-0 routing: replay the preserved history through the cold
      // start path instead of dispatching a delta that the native side
      // would reject with an un-prefixed "requires an initialized
      // session" error. A normal tool result always follows a prior
      // tool-call turn (turnCount>=1), so this never fires on the happy
      // path.
      if (this.turnCount === 0) {
        return await this.replayToolResultThroughStartPath(toolMsg, mergedConfig);
      }
      try {
        const result = await this.model.chatSessionContinueTool(toolCallId, content, mergedConfig, isError ?? null);
        this.history.push({ role: 'tool', content, toolCallId, isError });
        this.history.push(buildAssistantMessage(result.text, result.toolCalls));
        this.turnCount++;
        this.recordToolCallFanout(result.toolCalls);
        return result;
      } catch (err) {
        if (!isMediaHeldRestartError(err)) {
          throw err;
        }
        // The native session holds media KV (gemma4 after an image/audio
        // turn) and refused the tool-result delta. Transparently replay
        // the full conversation through the cold start path. The prior
        // media turn already lives in `this.history`, so the start path
        // re-renders it; the trailing-media keys keep
        // `lastImagesKey`/`lastAudioKey` consistent across the replay.
        // The delta path threw before pushing the tool message, so the
        // restart core pushes it — `isError` rides on that message so the
        // wire-format error marker is re-rendered, and a tool result
        // always follows >=1 prior turn so `isFirstTurn` is false.
        return await this.replayToolResultThroughStartPath(toolMsg, mergedConfig);
      }
    } finally {
      this.inFlight = false;
    }
  }

  /**
   * Cold-replay a tool result through the start path: re-render the
   * full preserved history (including the prior media turn and the
   * unresolved tool-call assistant turn) plus this tool message. Used
   * when the native session is cold (turnCount===0 — e.g. after an
   * interrupted media-held replay rolled the cache back) and by the
   * media-held rejection catch. `mediaChanged=true` forces a
   * resetCaches so the prefill always starts from a guaranteed-clean
   * cache; `isFirstTurn=false` because a tool result always follows a
   * prior tool-call turn.
   */
  private async replayToolResultThroughStartPath(toolMsg: ChatMessage, config: ChatConfig): Promise<ChatResult> {
    return await this.runStartPathWithMessage(toolMsg, true, false, config);
  }

  /**
   * Streaming variant of {@link ChatSession#sendToolResult}.
   *
   * `isError` mirrors the non-streaming entry point — when `true`,
   * the native renderer prepends a short, model-facing error marker
   * to `content` inside the wire-format tool block. The structured
   * field is stored verbatim on the appended `{ role: 'tool', ... }`
   * history entry so cold-replay re-renders the marker consistently
   * with the live streaming turn.
   */
  async *sendToolResultStream(
    toolCallId: string,
    content: string,
    opts: { isError?: boolean; config?: ChatConfig; signal?: AbortSignal } = {},
  ): AsyncGenerator<ChatStreamEvent> {
    if (this.inFlight) {
      throw new Error('ChatSession: concurrent send() not allowed; await the previous call first');
    }
    this.assertCanSendToolResult('sendToolResultStream');
    this.inFlight = true;
    try {
      const { isError, config, signal } = opts;
      const mergedConfig = this.mergeConfig(config);
      const toolMsg: ChatMessage = { role: 'tool', content, toolCallId, isError };
      // A cold native session (turnCount===0) has no live KV to delta
      // against — typically the residue of an interrupted media-held
      // replay whose rollback wiped the cache and reset the counter
      // while leaving the unresolved tool-call flag set. Mirror
      // `sendStream()`'s turn-0 routing: replay the preserved history
      // through the cold start stream and return before the
      // delta/commit machinery so the start path owns the history push,
      // turnCount increment, and media-key rehydration. A normal tool
      // result always follows a prior tool-call turn (turnCount>=1), so
      // this never fires on the happy path.
      if (this.turnCount === 0) {
        yield* this.replayToolResultThroughStartStreamPath(toolMsg, mergedConfig, signal);
        return;
      }
      let sawFinal = false;
      let accumulated = '';
      let accumulatedVisible = '';
      let finalRaw: string | null = null;
      let finalToolCalls: readonly ToolCallResult[] | undefined;
      // Set when the media-held rejection re-routes this tool turn
      // through the cold start stream. The replay path owns the history
      // push, turnCount increment, and media-key rehydration, so the
      // commit `finally` below must NOT also fire.
      let delegated = false;
      try {
        try {
          for await (const event of this.model.chatStreamSessionContinueTool(
            toolCallId,
            content,
            mergedConfig,
            signal,
            isError ?? null,
          )) {
            if (event.done) {
              if (event.finishReason !== 'error') {
                sawFinal = true;
                finalRaw = event.text;
                finalToolCalls = event.toolCalls;
              }
            } else {
              accumulated += event.text;
              if (event.isReasoning !== true) {
                accumulatedVisible += event.text;
              }
            }
            yield event;
          }
        } catch (err) {
          // The native session holds media KV (gemma4 after an image/audio
          // turn) and refused the tool-result delta. The streaming bridge
          // re-throws that rejection on the first iteration, BEFORE any
          // chunk is emitted — the native guard fires ahead of any
          // prefill, so `!sawFinal && accumulated === ''` is guaranteed
          // here. Replay the full conversation through the cold start
          // stream with the pending tool message; `isError` rides on it so
          // the wire-format error marker is re-rendered. Any non-prefix
          // error, or any error after tokens were already emitted, must
          // propagate unchanged.
          if (!isMediaHeldRestartError(err) || sawFinal || accumulated !== '') {
            throw err;
          }
          delegated = true;
          yield* this.replayToolResultThroughStartStreamPath(toolMsg, mergedConfig, signal);
          return;
        }
      } finally {
        // finally runs for normal completion, mid-stream throw,
        // caller `break` (iterator.return() short-circuits the yield),
        // and error-finish chunks alike. Tool turns never touch
        // history until commit, so the rollback branch is a no-op. When
        // the media-held rejection delegated to the replay stream, that
        // path already committed (or rolled back), so this commit stays
        // off.
        if (sawFinal && !delegated) {
          this.history.push({ role: 'tool', content, toolCallId, isError });
          this.history.push(buildAssistantMessage(finalRaw || accumulatedVisible, finalToolCalls));
          this.turnCount++;
          this.recordToolCallFanout(finalToolCalls);
        }
      }
    } finally {
      this.inFlight = false;
    }
  }

  /**
   * Streaming counterpart of {@link replayToolResultThroughStartPath}:
   * cold-replay a tool result through the start stream. Used by the
   * turn-0 precheck and the media-held rejection catch in
   * {@link sendToolResultStream}. The start stream owns the history
   * push, turnCount increment, and media-key rehydration; callers keep
   * `delegated`/early-return semantics so the commit `finally` stays
   * off.
   */
  private async *replayToolResultThroughStartStreamPath(
    toolMsg: ChatMessage,
    config: ChatConfig,
    signal: AbortSignal | undefined,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* this.runStartStreamPathWithMessage(toolMsg, true, false, config, signal);
  }

  /**
   * Reset the session state.
   *
   * Clears the underlying model's KV caches and wipes local history,
   * image key, and turn counter so the next `send()` goes through
   * `chatSessionStart` again.
   *
   * This is a full wipe — safe default for public callers. It always
   * calls `model.resetCaches()`, which is the ONLY behavior exposed
   * on the public API because the underlying `SessionCapableModel`
   * is shared across every `ChatSession` lifetime via the native
   * `ModelRegistry`: a partial wipe that leaves the shared native
   * KV cache intact would leak a previous (unrelated) request's
   * cached prefix into the next `chat_session_start_sync` call. The
   * server-side warm-lease replay path (where preserving the native
   * cache is correct) uses its own server-private helper gated by
   * the `SessionRegistry` HIT signal — the only authoritative proof
   * that the native cache genuinely belongs to this chain. That
   * helper lives inside `@mlx-node/server`, never touches the
   * `@mlx-node/lm` export map, and is not reachable from downstream
   * consumers. Public consumers of `@mlx-node/lm` have no such HIT
   * signal, so the public API intentionally offers only the full-wipe
   * option.
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
    this.lastAudioKey = null;
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
      this.lastAudioKey = this.computeTrailingAudioKey();
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
      let accumulatedVisible = '';
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
            if (event.isReasoning !== true) {
              accumulatedVisible += event.text;
            }
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
          this.history.push(buildAssistantMessage(finalRaw || accumulatedVisible, finalToolCalls));
          this.turnCount++;
          this.lastImagesKey = this.computeTrailingImagesKey();
          this.lastAudioKey = this.computeTrailingAudioKey();
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
   *
   * MTP auto-default: if neither `defaultConfig` nor `overlay`
   * sets `enableMtp` AND the underlying model exposes
   * `hasMtpWeights()` returning `true`, set `enableMtp = true` so the
   * speculative-decode path runs out of the box on MTP-capable
   * checkpoints. An explicit `false` from either source wins (the
   * undefined-check below preserves it). This duck-typed check also
   * covers Gemma4 with an external draft attached — DSpark or Google
   * assistant (`hasMtpWeights()` reports the external draft there,
   * not in-checkpoint MTP heads).
   */
  private mergeConfig(overlay: ChatConfig | undefined): ChatConfig {
    const merged: ChatConfig = {
      ...this.defaultConfig,
      ...overlay,
      reuseCache: true,
    };
    if (
      merged.enableMtp === undefined &&
      typeof this.model.hasMtpWeights === 'function' &&
      this.model.hasMtpWeights()
    ) {
      merged.enableMtp = true;
    }
    return merged;
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
    audio: Uint8Array[] | undefined,
    mediaChanged: boolean,
    isFirstTurn: boolean,
    config: ChatConfig,
  ): Promise<ChatResult> {
    const userMsg = this.buildUserMessage(userMessage, images, audio);
    return await this.runStartPathWithMessage(userMsg, mediaChanged, isFirstTurn, config);
  }

  /**
   * Core of {@link runStartPath} that takes a PRE-BUILT pending
   * `ChatMessage` (user or tool) instead of building a user message
   * itself. The cold-restart catch in `sendToolResult()` replays the
   * conversation through this core with a pending `{ role: 'tool', ... }`
   * message so the tool-result turn is re-rendered against the full
   * history without duplicating the start-path bookkeeping.
   */
  private async runStartPathWithMessage(
    pendingMessage: ChatMessage,
    mediaChanged: boolean,
    isFirstTurn: boolean,
    config: ChatConfig,
  ): Promise<ChatResult> {
    // Capture pre-state so the restart can be rolled back if the
    // native call fails. The media-change branch resets caches BEFORE
    // we know whether the new prefill will succeed, so on failure we
    // also have to drop turnCount + lastImagesKey/lastAudioKey to force
    // the next call to re-route through the start path (rather than a
    // delta continue against wiped caches).
    const wasMediaChangeRestart = mediaChanged && !isFirstTurn;
    const historyLenBefore = this.history.length;

    this.prepareStartPath(mediaChanged, isFirstTurn);
    this.history.push(pendingMessage);
    try {
      // Pass a shallow snapshot so later pushes to `this.history`
      // (e.g. the assistant reply below) don't retroactively mutate
      // what the native side / any mock observed as its `messages`
      // argument.
      const result = await this.model.chatSessionStart(this.history.slice(), config);
      this.history.push(buildAssistantMessage(result.text, result.toolCalls));
      this.turnCount++;
      // The start path always re-renders the FULL preserved history, so the
      // post-restart sticky keys are the trailing media keys of that history,
      // not the single-turn literal args. A restart driven by a change in only
      // one modality (e.g. an audio-only turn after an earlier image turn)
      // would otherwise null the untouched modality's key even though that
      // media is still live in the native cache, causing a later same-media
      // turn to be mis-detected as a change and replayed twice.
      this.lastImagesKey = this.computeTrailingImagesKey();
      this.lastAudioKey = this.computeTrailingAudioKey();
      this.recordToolCallFanout(result.toolCalls);
      return result;
    } catch (err) {
      // Roll back: drop the tentative user push so history stays
      // consistent with turnCount.
      this.history.length = historyLenBefore;
      if (wasMediaChangeRestart) {
        // Caches were wiped by prepareStartPath() but the new prefill
        // failed. Force the next call to re-route through the start
        // path with the (preserved) prior history.
        this.turnCount = 0;
        this.lastImagesKey = null;
        this.lastAudioKey = null;
      }
      throw err;
    }
  }

  /** Streaming counterpart to {@link runStartPath}. */
  private async *runStartStreamPath(
    userMessage: string,
    images: Uint8Array[] | undefined,
    audio: Uint8Array[] | undefined,
    mediaChanged: boolean,
    isFirstTurn: boolean,
    config: ChatConfig,
    signal: AbortSignal | undefined,
  ): AsyncGenerator<ChatStreamEvent> {
    const userMsg = this.buildUserMessage(userMessage, images, audio);
    yield* this.runStartStreamPathWithMessage(userMsg, mediaChanged, isFirstTurn, config, signal);
  }

  /**
   * Streaming counterpart to {@link runStartPathWithMessage}: replays
   * through the cold start stream from a PRE-BUILT pending
   * `ChatMessage`. The cold-restart catch in `sendToolResultStream()`
   * delegates here with a pending `{ role: 'tool', ... }` message.
   */
  private async *runStartStreamPathWithMessage(
    pendingMessage: ChatMessage,
    mediaChanged: boolean,
    isFirstTurn: boolean,
    config: ChatConfig,
    signal: AbortSignal | undefined,
  ): AsyncGenerator<ChatStreamEvent> {
    // Capture pre-state so any non-successful exit can roll back.
    // See `runStartPath` for the full rationale.
    const wasMediaChangeRestart = mediaChanged && !isFirstTurn;
    const historyLenBefore = this.history.length;

    this.prepareStartPath(mediaChanged, isFirstTurn);
    // Stage the pending message on the pending history BEFORE the
    // stream starts — the native call reads it synchronously via
    // `model.chatStreamSessionStart(history, config)`.
    this.history.push(pendingMessage);

    let sawFinal = false;
    let accumulated = '';
    let accumulatedVisible = '';
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
          if (event.isReasoning !== true) {
            accumulatedVisible += event.text;
          }
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
        this.history.push(buildAssistantMessage(finalRaw || accumulatedVisible, finalToolCalls));
        this.turnCount++;
        // The start path always re-renders the FULL preserved history, so the
        // post-restart sticky keys are the trailing media keys of that history,
        // not the single-turn literal args. A restart driven by a change in only
        // one modality (e.g. an audio-only turn after an earlier image turn)
        // would otherwise null the untouched modality's key even though that
        // media is still live in the native cache, causing a later same-media
        // turn to be mis-detected as a change and replayed twice.
        this.lastImagesKey = this.computeTrailingImagesKey();
        this.lastAudioKey = this.computeTrailingAudioKey();
        this.recordToolCallFanout(finalToolCalls);
      } else {
        // Roll back: drop the tentative user push so history stays
        // consistent with turnCount.
        this.history.length = historyLenBefore;
        if (wasMediaChangeRestart) {
          // Caches were wiped by prepareStartPath() but the new
          // prefill never reached a successful done:true. Force the
          // next call to re-route through the start path with the
          // preserved prior history.
          this.turnCount = 0;
          this.lastImagesKey = null;
          this.lastAudioKey = null;
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
  private prepareStartPath(mediaChanged: boolean, isFirstTurn: boolean): void {
    if (mediaChanged && !isFirstTurn) {
      this.model.resetCaches();
    }
    if (this.history.length === 0 && this.system != null) {
      this.history.push({ role: 'system', content: this.system });
    }
  }

  /** Build a user `ChatMessage` with or without attached images/audio. */
  private buildUserMessage(
    userMessage: string,
    images: Uint8Array[] | undefined,
    audio: Uint8Array[] | undefined,
  ): ChatMessage {
    const msg: ChatMessage = { role: 'user', content: userMessage };
    if (images && images.length > 0) msg.images = images;
    if (audio && audio.length > 0) msg.audio = audio;
    return msg;
  }

  /**
   * Walk the history backward to find the most recent user message
   * with images and return its SHA-256 key. Used by
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
   * Audio counterpart of {@link computeTrailingImagesKey}: walk history
   * backward to the most recent user message carrying audio and return its
   * SHA-256 key, so a cold replay hydrates `lastAudioKey` correctly.
   */
  private computeTrailingAudioKey(): string | null {
    for (let i = this.history.length - 1; i >= 0; i--) {
      const msg = this.history[i];
      if (msg?.role === 'user' && msg.audio && msg.audio.length > 0) {
        return computeAudioKey(msg.audio);
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
