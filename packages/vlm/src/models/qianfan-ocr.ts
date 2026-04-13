import { QianfanOCRModel as QianfanOCRModelNative } from '@mlx-node/core';

/**
 * Qianfan-OCR Vision-Language Model wrapper.
 *
 * Streaming is driven through the `ChatSession` API — the legacy
 * `chatStream()` surface was removed in the chat-session refactor.
 */
export class QianfanOCRModel extends QianfanOCRModelNative {
  static override async load(modelPath: string): Promise<QianfanOCRModel> {
    const instance = await QianfanOCRModelNative.load(modelPath);
    Object.setPrototypeOf(instance, QianfanOCRModel.prototype);
    return instance as unknown as QianfanOCRModel;
  }
}
