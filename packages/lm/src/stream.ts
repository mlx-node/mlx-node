import {
  Gemma4Model as Gemma4ModelNative,
  Lfm2Model as Lfm2ModelNative,
  Qwen3Model as Qwen3ModelNative,
  Qwen35Model as Qwen35ModelNative,
  Qwen35MoeModel as Qwen35MoeModelNative,
} from '@mlx-node/core';
import type {
  ChatConfig,
  ChatMessage,
  ChatStreamChunk,
  ChatStreamHandle,
  PerformanceMetrics,
  ToolCallResult,
} from '@mlx-node/core';

import type { SessionCapableModel } from './chat-session.js';

export interface ChatStreamDelta {
  text: string;
  done: false;
  isReasoning?: boolean;
}

export interface ChatStreamFinal {
  text: string;
  done: true;
  finishReason: string;
  toolCalls: ToolCallResult[];
  thinking: string | null;
  numTokens: number;
  promptTokens: number;
  reasoningTokens: number;
  rawText: string;
  performance?: PerformanceMetrics;
}

export type ChatStreamEvent = ChatStreamDelta | ChatStreamFinal;

// Save references to the native callback-based session streaming methods
// before we override them. The legacy `chatStream` surface was removed in
// the chat-session refactor; the remaining session entry points below
// drive all streaming via the `ChatSession` API.
//
// Each wrapper class re-declares these three methods as
// `AsyncGenerator<ChatStreamEvent>` overrides that delegate through the
// shared `_runChatStream` bridge, so the wrapper structurally satisfies
// `SessionCapableModel` and can be passed to `ChatSession<M>`.

// Dense
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeDenseChatStreamSessionStart = Qwen35ModelNative.prototype.chatStreamSessionStart;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeDenseChatStreamSessionContinue = Qwen35ModelNative.prototype.chatStreamSessionContinue;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeDenseChatStreamSessionContinueTool = Qwen35ModelNative.prototype.chatStreamSessionContinueTool;

// MoE
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeMoeChatStreamSessionStart = Qwen35MoeModelNative.prototype.chatStreamSessionStart;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeMoeChatStreamSessionContinue = Qwen35MoeModelNative.prototype.chatStreamSessionContinue;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeMoeChatStreamSessionContinueTool = Qwen35MoeModelNative.prototype.chatStreamSessionContinueTool;

// LFM2
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeLfm2ChatStreamSessionStart = Lfm2ModelNative.prototype.chatStreamSessionStart;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeLfm2ChatStreamSessionContinue = Lfm2ModelNative.prototype.chatStreamSessionContinue;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeLfm2ChatStreamSessionContinueTool = Lfm2ModelNative.prototype.chatStreamSessionContinueTool;

// Gemma4
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeGemma4ChatStreamSessionStart = Gemma4ModelNative.prototype.chatStreamSessionStart;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeGemma4ChatStreamSessionContinue = Gemma4ModelNative.prototype.chatStreamSessionContinue;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeGemma4ChatStreamSessionContinueTool = Gemma4ModelNative.prototype.chatStreamSessionContinueTool;

// Qwen3 (legacy, text-only)
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeQwen3ChatStreamSessionStart = Qwen3ModelNative.prototype.chatStreamSessionStart;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeQwen3ChatStreamSessionContinue = Qwen3ModelNative.prototype.chatStreamSessionContinue;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeQwen3ChatStreamSessionContinueTool = Qwen3ModelNative.prototype.chatStreamSessionContinueTool;

/**
 * Shared AsyncGenerator adapter for callback-based native streaming methods.
 *
 * Takes a `startCall` closure that, given the JS-side callback, dispatches
 * the underlying native stream (whatever method signature that is — the
 * closure captures `messages` / `config` / `userMessage` etc) and resolves
 * with a `ChatStreamHandle`. The generator pumps the resulting chunk queue,
 * transforms each chunk into a `ChatStreamEvent`, and calls `handle.cancel()`
 * in a `finally` block so early termination (user `break`, exception) still
 * cleans up native state.
 *
 * ## Signal-driven fast-abort
 *
 * When an optional `AbortSignal` is supplied and fires, the adapter:
 *   1. Calls `handle.cancel()` immediately so the native decode stops
 *      on the next safepoint rather than running to completion.
 *   2. Wakes the pending `waitForItem()` await by pushing a synthetic
 *      "aborted" marker into the queue and calling `notify()`. Without
 *      this wake-up the generator would stay parked on the `await`
 *      until the next native chunk arrived, which on a fast-abort
 *      path (client disconnect before first token) never happens.
 *   3. The generator sees the marker, breaks out of its loop, and the
 *      finally block runs `cancelOnce()` — which is a no-op because
 *      `triggerAbort` already flipped the `cancelled` flag. Some
 *      backends throw on double-cancel, so routing every cancel site
 *      through `cancelOnce` keeps abort behavior deterministic.
 *
 * The finally block is also the landing site for the consumer calling
 * `.return()` on the outer generator — the existing `yield` cleanup
 * covers that case. Signal-driven abort covers the window where the
 * consumer cannot reach `.return()` because they are blocked waiting
 * for the very `yield` that `waitForItem()` is gating.
 *
 * @internal Exported so the VLM wrapper (`@mlx-node/vlm`) can reuse the
 * exact same bridge without duplicating the plumbing. Not part of the
 * public API — may change without notice.
 */
export async function* _runChatStream(
  startCall: (callback: (err: Error | null, chunk: ChatStreamChunk) => void) => Promise<ChatStreamHandle>,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const queue: Array<{ chunk?: ChatStreamChunk; error?: Error; aborted?: boolean }> = [];
  let resolve: (() => void) | null = null;

  const waitForItem = () =>
    queue.length > 0
      ? Promise.resolve()
      : new Promise<void>((r) => {
          resolve = r;
        });

  const notify = () => {
    if (resolve) {
      const r = resolve;
      resolve = null;
      r();
    }
  };

  const callback = (err: Error | null, chunk: ChatStreamChunk) => {
    queue.push(err ? { error: err } : { chunk });
    notify();
  };

  const handle = await startCall(callback);

  // Guard against double-cancel. Some native backends throw on a
  // second `cancel()`; we route every cancel site through this
  // helper so the abort path (via `triggerAbort`) and the unwind
  // path (via the `finally` block) don't cancel twice and so any
  // backend that does throw is swallowed rather than escaping as
  // an error out of an otherwise-clean early termination.
  let cancelled = false;
  const cancelOnce = (): void => {
    if (cancelled) return;
    cancelled = true;
    try {
      handle.cancel();
    } catch {
      // Native backend threw on cancel — nothing actionable here.
      // Swallow so aborted streams still surface as a clean early
      // termination via the synthetic `aborted` marker rather than
      // as an unexpected error out of the generator.
    }
  };

  // Signal-driven fast-abort. If the signal is already aborted at
  // attach time we still arm the listener so the synchronous abort
  // dispatch path runs below (calling `handle.cancel()` after the
  // handle is in hand, not before — there is nothing to cancel
  // pre-start). For an already-fired signal Node delivers the
  // `'abort'` event on the next microtask.
  let onAbort: (() => void) | null = null;
  if (signal != null) {
    const triggerAbort = (): void => {
      // Cancel the native side first so any work-in-flight winds
      // down ASAP. `cancelOnce` is idempotent — the finally block
      // will invoke it again after the generator unwinds, but the
      // second call becomes a no-op.
      cancelOnce();
      // Push a synthetic abort marker so the consumer-visible
      // generator breaks out of its loop at the next iteration
      // rather than waiting for a native chunk that will never
      // arrive (e.g. client disconnect before first token).
      queue.push({ aborted: true });
      notify();
    };
    if (signal.aborted) {
      // Signal already fired before we could attach — dispatch
      // synchronously so the waitForItem below resolves immediately.
      triggerAbort();
    } else {
      onAbort = triggerAbort;
      signal.addEventListener('abort', onAbort, { once: true });
    }
  }

  try {
    loop: while (true) {
      await waitForItem();
      while (queue.length > 0) {
        const item = queue.shift()!;
        if (item.aborted) {
          // Consumer asked us to stop before the next chunk landed.
          // Break out so the finally block runs `handle.cancel()`
          // (idempotent — `triggerAbort` already called it) and the
          // consumer-side `for await` unblocks cleanly. We do NOT
          // throw an AbortError: callers (e.g. server endpoints that
          // flag client disconnect) have already decided this is not
          // an error from their perspective, just an early stop.
          break loop;
        }
        if (item.error) throw item.error;
        const chunk = item.chunk!;
        if (chunk.done) {
          yield {
            text: chunk.text,
            done: true,
            finishReason: chunk.finishReason!,
            toolCalls: chunk.toolCalls ?? [],
            thinking: chunk.thinking ?? null,
            numTokens: chunk.numTokens!,
            promptTokens: chunk.promptTokens ?? 0,
            reasoningTokens: chunk.reasoningTokens ?? 0,
            rawText: chunk.rawText!,
            performance: chunk.performance ?? undefined,
          } as ChatStreamFinal;
          return;
        }
        yield { text: chunk.text, done: false, isReasoning: chunk.isReasoning ?? undefined } as ChatStreamDelta;
      }
    }
  } finally {
    if (signal != null && onAbort != null) {
      try {
        signal.removeEventListener('abort', onAbort);
      } catch {
        // removeEventListener shouldn't throw, but stay defensive —
        // a misbehaving signal must not leak out of the finally.
      }
    }
    cancelOnce();
  }
}

/**
 * Qwen3.5 dense model with AsyncGenerator-based session streaming.
 *
 * Streaming is driven through the session API — `chatStreamSessionStart`,
 * `chatStreamSessionContinue`, and `chatStreamSessionContinueTool` below —
 * which adapt the callback-based native methods to
 * `AsyncGenerator<ChatStreamEvent>` so the wrapper structurally satisfies
 * `SessionCapableModel` and can be passed to `ChatSession<Qwen35Model>`.
 */
export class Qwen35Model extends Qwen35ModelNative {
  static override async load(modelPath: string): Promise<Qwen35Model> {
    const instance = await Qwen35ModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Qwen35Model.prototype);
    return instance as unknown as Qwen35Model;
  }

  /**
   * Streaming variant of {@link Qwen35Model#chatSessionStart}.
   *
   * Resets the KV caches, runs the jinja chat template, prefills on
   * top of the fresh caches, and streams the decoded reply token-by-
   * token. Stops on `<|im_end|>` so the cached history ends on a
   * clean ChatML boundary that subsequent `chatStreamSessionContinue`
   * deltas can append to. Text-only.
   *
   * The optional `signal` parameter wires an AbortSignal into the
   * `_runChatStream` adapter's fast-abort path. Callers that need
   * client-disconnect-aware cancellation (e.g. HTTP endpoints) pass
   * one here and the native decode winds down at the next safepoint.
   */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionStart(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeDenseChatStreamSessionStart.call(this, messages, config ?? null, callback),
      signal,
    );
  }

  /**
   * Streaming variant of {@link Qwen35Model#chatSessionContinue}.
   *
   * Builds a raw ChatML delta on top of the live session caches,
   * tokenizes it, prefills the delta, and streams the decoded reply.
   * Requires a live session started via `chatSessionStart` or
   * `chatStreamSessionStart`. Stops on `<|im_end|>`.
   *
   * `images` is the native opt-in guard parameter — callers that
   * attach a new image set must restart the session via
   * `chatStreamSessionStart` with the full history. The high-level
   * `ChatSession` wrapper handles that routing; callers that drive
   * the wrapper directly should pass `null` for text-only continues.
   */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeDenseChatStreamSessionContinue.call(this, userMessage, images, config ?? null, callback),
      signal,
    );
  }

  /**
   * Streaming variant of {@link Qwen35Model#chatSessionContinueTool}.
   *
   * Builds a ChatML `<tool_response>` delta on top of the live
   * session caches and streams the decoded assistant reply. Requires
   * a live session started via `chatSessionStart` /
   * `chatStreamSessionStart`.
   */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeDenseChatStreamSessionContinueTool.call(this, toolCallId, content, config ?? null, callback),
      signal,
    );
  }
}

/**
 * Qwen3.5 MoE model wrapper.
 *
 * Streaming is driven through the `ChatSession` API — overrides below
 * adapt the callback-based native methods to
 * `AsyncGenerator<ChatStreamEvent>` so the wrapper structurally
 * satisfies `SessionCapableModel`.
 */
export class Qwen35MoeModel extends Qwen35MoeModelNative {
  static override async load(modelPath: string): Promise<Qwen35MoeModel> {
    const instance = await Qwen35MoeModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Qwen35MoeModel.prototype);
    return instance as unknown as Qwen35MoeModel;
  }

  /** Streaming variant of {@link Qwen35MoeModel#chatSessionStart}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionStart(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeMoeChatStreamSessionStart.call(this, messages, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link Qwen35MoeModel#chatSessionContinue}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeMoeChatStreamSessionContinue.call(this, userMessage, images, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link Qwen35MoeModel#chatSessionContinueTool}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeMoeChatStreamSessionContinueTool.call(this, toolCallId, content, config ?? null, callback),
      signal,
    );
  }
}

/**
 * LFM2 model wrapper.
 *
 * Streaming is driven through the `ChatSession` API — overrides below
 * adapt the callback-based native methods to
 * `AsyncGenerator<ChatStreamEvent>` so the wrapper structurally
 * satisfies `SessionCapableModel`. LFM2 is text-only; the native
 * `images` guard rejects non-empty image sets with an
 * `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` prefix.
 */
export class Lfm2Model extends Lfm2ModelNative {
  static override async load(modelPath: string): Promise<Lfm2Model> {
    const instance = await Lfm2ModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Lfm2Model.prototype);
    return instance as unknown as Lfm2Model;
  }

  /** Streaming variant of {@link Lfm2Model#chatSessionStart}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionStart(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeLfm2ChatStreamSessionStart.call(this, messages, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link Lfm2Model#chatSessionContinue}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeLfm2ChatStreamSessionContinue.call(this, userMessage, images, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link Lfm2Model#chatSessionContinueTool}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeLfm2ChatStreamSessionContinueTool.call(this, toolCallId, content, config ?? null, callback),
      signal,
    );
  }
}

/**
 * Gemma4 model wrapper.
 *
 * Streaming is driven through the `ChatSession` API — overrides below
 * adapt the callback-based native methods to
 * `AsyncGenerator<ChatStreamEvent>` so the wrapper structurally
 * satisfies `SessionCapableModel`. Gemma4 is text-only in the
 * current refactor scope; the native `images` guard rejects non-empty
 * image sets with an `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` prefix.
 */
export class Gemma4Model extends Gemma4ModelNative {
  static override async load(modelPath: string): Promise<Gemma4Model> {
    const instance = await Gemma4ModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Gemma4Model.prototype);
    return instance as unknown as Gemma4Model;
  }

  /** Streaming variant of {@link Gemma4Model#chatSessionStart}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionStart(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeGemma4ChatStreamSessionStart.call(this, messages, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link Gemma4Model#chatSessionContinue}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeGemma4ChatStreamSessionContinue.call(this, userMessage, images, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link Gemma4Model#chatSessionContinueTool}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) =>
        _nativeGemma4ChatStreamSessionContinueTool.call(this, toolCallId, content, config ?? null, callback),
      signal,
    );
  }
}

/**
 * Qwen3 (legacy) model wrapper.
 *
 * Streaming is driven through the `ChatSession` API — overrides below
 * adapt the callback-based native methods to
 * `AsyncGenerator<ChatStreamEvent>` so the wrapper structurally
 * satisfies `SessionCapableModel`. Qwen3 legacy is text-only; the
 * native `images` guard rejects non-empty image sets with an
 * `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` prefix.
 */
export class Qwen3Model extends Qwen3ModelNative {
  static override async load(modelPath: string): Promise<Qwen3Model> {
    const instance = await Qwen3ModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Qwen3Model.prototype);
    return instance as unknown as Qwen3Model;
  }

  /** Streaming variant of {@link Qwen3Model#chatSessionStart}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionStart(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeQwen3ChatStreamSessionStart.call(this, messages, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link Qwen3Model#chatSessionContinue}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeQwen3ChatStreamSessionContinue.call(this, userMessage, images, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link Qwen3Model#chatSessionContinueTool}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeQwen3ChatStreamSessionContinueTool.call(this, toolCallId, content, config ?? null, callback),
      signal,
    );
  }
}

// -------------------------------------------------------------------
// Compile-time conformance check
// -------------------------------------------------------------------
//
// Ensures each wrapper class structurally satisfies
// `SessionCapableModel` so `ChatSession<XxxModel>` will type-check in
// downstream code. The assignments are compile-only — the
// `null as unknown as T` placeholder never runs. If a wrapper's
// override signature drifts away from the interface, TypeScript will
// fail to compile this block, surfacing the regression at build time.
function _assertSessionCapable(): void {
  const _qwen35: SessionCapableModel = null as unknown as Qwen35Model;
  const _moe: SessionCapableModel = null as unknown as Qwen35MoeModel;
  const _lfm2: SessionCapableModel = null as unknown as Lfm2Model;
  const _gemma4: SessionCapableModel = null as unknown as Gemma4Model;
  const _qwen3: SessionCapableModel = null as unknown as Qwen3Model;
  void _qwen35;
  void _moe;
  void _lfm2;
  void _gemma4;
  void _qwen3;
}
void _assertSessionCapable;
