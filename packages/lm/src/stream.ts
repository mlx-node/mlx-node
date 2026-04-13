import {
  Gemma4Model as Gemma4ModelNative,
  Lfm2Model as Lfm2ModelNative,
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
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeDenseChatStreamSessionStart = Qwen35ModelNative.prototype.chatStreamSessionStart;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeDenseChatStreamSessionContinue = Qwen35ModelNative.prototype.chatStreamSessionContinue;

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
 */
async function* _runChatStream(
  startCall: (callback: (err: Error | null, chunk: ChatStreamChunk) => void) => Promise<ChatStreamHandle>,
): AsyncGenerator<ChatStreamEvent> {
  const queue: Array<{ chunk?: ChatStreamChunk; error?: Error }> = [];
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

  try {
    while (true) {
      await waitForItem();
      while (queue.length > 0) {
        const item = queue.shift()!;
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
    handle.cancel();
  }
}

/**
 * Legacy `_createChatStream` shape kept for the existing tests at
 * `__test__/models/qwen35-stream.test.ts`. New code should use
 * `_runChatStream` directly with a bound `startCall` closure.
 *
 * @internal Exported for testing only.
 */
export async function* _createChatStream(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  nativeMethod: (
    messages: ChatMessage[],
    config: any,
    callback: (err: Error | null, chunk: ChatStreamChunk) => void,
  ) => Promise<ChatStreamHandle>,
  self: unknown,
  messages: ChatMessage[],
  config: unknown,
): AsyncGenerator<ChatStreamEvent> {
  yield* _runChatStream((callback) => nativeMethod.call(self, messages, config ?? null, callback));
}

/**
 * Qwen3.5 dense model with AsyncGenerator-based session streaming.
 *
 * Streaming is driven through the session API — `chatStreamSessionStart`
 * and `chatStreamSessionContinue` below — which adapt the callback-based
 * native methods to `AsyncGenerator<ChatStreamEvent>`.
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
   */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionStart(messages: ChatMessage[], config?: ChatConfig | null): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream((callback) =>
      _nativeDenseChatStreamSessionStart.call(this, messages, config ?? null, callback),
    );
  }

  /**
   * Streaming variant of {@link Qwen35Model#chatSessionContinue}.
   *
   * Builds a raw ChatML delta on top of the live session caches,
   * tokenizes it, prefills the delta, and streams the decoded reply.
   * Requires a live session started via `chatSessionStart` or
   * `chatStreamSessionStart`. Stops on `<|im_end|>`.
   */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinue(userMessage: string, config?: ChatConfig | null): AsyncGenerator<ChatStreamEvent> {
    // `null` for the new `images` guard parameter — this wrapper is
    // text-only; Step T2 will expose a higher-level image-aware
    // ChatSession API that plumbs image changes through a fresh
    // session restart.
    yield* _runChatStream((callback) =>
      _nativeDenseChatStreamSessionContinue.call(this, userMessage, null, config ?? null, callback),
    );
  }
}

/**
 * Qwen3.5 MoE model wrapper.
 *
 * Streaming is driven through the `ChatSession` API — the legacy
 * `chatStream()` surface was removed in the chat-session refactor.
 */
export class Qwen35MoeModel extends Qwen35MoeModelNative {
  static override async load(modelPath: string): Promise<Qwen35MoeModel> {
    const instance = await Qwen35MoeModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Qwen35MoeModel.prototype);
    return instance as unknown as Qwen35MoeModel;
  }
}

/**
 * LFM2 model wrapper.
 *
 * Streaming is driven through the `ChatSession` API — the legacy
 * `chatStream()` surface was removed in the chat-session refactor.
 */
export class Lfm2Model extends Lfm2ModelNative {
  static override async load(modelPath: string): Promise<Lfm2Model> {
    const instance = await Lfm2ModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Lfm2Model.prototype);
    return instance as unknown as Lfm2Model;
  }
}

/**
 * Gemma4 model wrapper.
 *
 * Streaming is driven through the `ChatSession` API — the legacy
 * `chatStream()` surface was removed in the chat-session refactor.
 */
export class Gemma4Model extends Gemma4ModelNative {
  static override async load(modelPath: string): Promise<Gemma4Model> {
    const instance = await Gemma4ModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Gemma4Model.prototype);
    return instance as unknown as Gemma4Model;
  }
}
