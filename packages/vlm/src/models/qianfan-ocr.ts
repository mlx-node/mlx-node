import { QianfanOCRModel as QianfanOCRModelNative } from '@mlx-node/core';
import type { ChatConfig, ChatMessage } from '@mlx-node/core';
import { _createChatStream, type ChatStreamEvent } from '@mlx-node/lm';

// Save reference to native callback-based method before we override it
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeChatStream = QianfanOCRModelNative.prototype.chatStream;

/**
 * Qianfan-OCR Vision-Language Model with AsyncGenerator-based `chatStream()`.
 *
 * Wraps the native NAPI class to provide a `for await...of`-compatible
 * streaming interface, matching the pattern used by Qwen35Model.
 *
 * @example
 * ```typescript
 * const model = await QianfanOCRModel.load('./models/qianfan-ocr');
 * for await (const event of model.chatStream(messages, config)) {
 *   if (!event.done) process.stdout.write(event.text);
 * }
 * ```
 */
export class QianfanOCRModel extends QianfanOCRModelNative {
  static override async load(modelPath: string): Promise<QianfanOCRModel> {
    const instance = await QianfanOCRModelNative.load(modelPath);
    Object.setPrototypeOf(instance, QianfanOCRModel.prototype);
    return instance as unknown as QianfanOCRModel;
  }

  // @ts-expect-error — override callback-based chatStream with AsyncGenerator
  async *chatStream(messages: ChatMessage[], config?: ChatConfig | null): AsyncGenerator<ChatStreamEvent> {
    yield* _createChatStream(_nativeChatStream, this, messages, config);
  }
}
