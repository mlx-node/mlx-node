import { Qwen35Model as Qwen35ModelNative, Qwen35MoeModel as Qwen35MoeModelNative } from '@mlx-node/core';
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
}

export interface ChatStreamFinal {
  text: string;
  done: true;
  finishReason: string;
  toolCalls: ToolCallResult[];
  thinking: string | null;
  numTokens: number;
  rawText: string;
  performance?: PerformanceMetrics;
}

export type ChatStreamEvent = ChatStreamDelta | ChatStreamFinal;

// Save references to the native callback-based methods before we override them
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeDenseChatStream = Qwen35ModelNative.prototype.chatStream;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeMoeChatStream = Qwen35MoeModelNative.prototype.chatStream;

/**
 * Shared AsyncGenerator implementation that wraps a native callback-based
 * chatStream into a `for await...of`-compatible stream.
 *
 * Cancellation is automatic via the generator's `finally` block.
 */
/** @internal Exported for testing only. */
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

  const handle = await nativeMethod.call(self, messages, config ?? null, callback);

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
            rawText: chunk.rawText!,
            performance: chunk.performance ?? undefined,
          } as ChatStreamFinal;
          return;
        }
        yield { text: chunk.text, done: false } as ChatStreamDelta;
      }
    }
  } finally {
    handle.cancel();
  }
}

/**
 * Qwen3.5 dense model with AsyncGenerator-based `chatStream()`.
 *
 * @example
 * ```typescript
 * const model = await Qwen35Model.load('./models/qwen3.5-3b');
 * for await (const event of model.chatStream(messages)) {
 *   if (!event.done) process.stdout.write(event.text);
 * }
 * ```
 */
export class Qwen35Model extends Qwen35ModelNative {
  static override async load(modelPath: string): Promise<Qwen35Model> {
    const instance = await Qwen35ModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Qwen35Model.prototype);
    return instance as unknown as Qwen35Model;
  }

  // @ts-expect-error — override callback-based chatStream with AsyncGenerator
  async *chatStream(messages: ChatMessage[], config?: ChatConfig | null): AsyncGenerator<ChatStreamEvent> {
    yield* _createChatStream(_nativeDenseChatStream, this, messages, config);
  }
}

/**
 * Qwen3.5 MoE model with AsyncGenerator-based `chatStream()`.
 *
 * @example
 * ```typescript
 * const model = await Qwen35MoeModel.load('./models/qwen3.5-moe');
 * for await (const event of model.chatStream(messages)) {
 *   if (!event.done) process.stdout.write(event.text);
 * }
 * ```
 */
export class Qwen35MoeModel extends Qwen35MoeModelNative {
  static override async load(modelPath: string): Promise<Qwen35MoeModel> {
    const instance = await Qwen35MoeModelNative.load(modelPath);
    Object.setPrototypeOf(instance, Qwen35MoeModel.prototype);
    return instance as unknown as Qwen35MoeModel;
  }

  // @ts-expect-error — override callback-based chatStream with AsyncGenerator
  async *chatStream(messages: ChatMessage[], config?: ChatConfig | null): AsyncGenerator<ChatStreamEvent> {
    yield* _createChatStream(_nativeMoeChatStream, this, messages, config);
  }
}
