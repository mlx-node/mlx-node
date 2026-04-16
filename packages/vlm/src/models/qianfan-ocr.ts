import { QianfanOCRModel as QianfanOCRModelNative } from '@mlx-node/core';
import type { ChatConfig, ChatMessage } from '@mlx-node/core';
import { _runChatStream } from '@mlx-node/lm';
import type { ChatStreamEvent, SessionCapableModel } from '@mlx-node/lm';

// Save references to the native callback-based session streaming methods
// before we override them. Captured at module load time so the subclass
// overrides can delegate without recursing into themselves. Each wrapper
// method below bridges the callback API to `AsyncGenerator<ChatStreamEvent>`
// so the wrapper structurally satisfies `SessionCapableModel` and can be
// passed to `ChatSession<QianfanOCRModel>` from `@mlx-node/lm`.
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeQianfanOcrChatStreamSessionStart = QianfanOCRModelNative.prototype.chatStreamSessionStart;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeQianfanOcrChatStreamSessionContinue = QianfanOCRModelNative.prototype.chatStreamSessionContinue;
// oxlint-disable-next-line @typescript-eslint/unbound-method
const _nativeQianfanOcrChatStreamSessionContinueTool = QianfanOCRModelNative.prototype.chatStreamSessionContinueTool;

/**
 * Qianfan-OCR Vision-Language Model wrapper.
 *
 * Streaming is driven through the `ChatSession` API — the overrides
 * below adapt the callback-based native methods to
 * `AsyncGenerator<ChatStreamEvent>` so the wrapper structurally
 * satisfies `SessionCapableModel` from `@mlx-node/lm`.
 *
 * Qianfan-OCR is a VLM (InternViT + Qwen3 language model). The
 * continue path cannot splice new vision features into a live KV
 * cache — image changes always require a fresh session start, which
 * the high-level `ChatSession` wrapper handles via its
 * `lastImagesKey` check.
 */
export class QianfanOCRModel extends QianfanOCRModelNative {
  static override async load(modelPath: string): Promise<QianfanOCRModel> {
    const instance = await QianfanOCRModelNative.load(modelPath);
    Object.setPrototypeOf(instance, QianfanOCRModel.prototype);
    return instance as unknown as QianfanOCRModel;
  }

  /** Streaming variant of {@link QianfanOCRModel#chatSessionStart}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionStart(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) => _nativeQianfanOcrChatStreamSessionStart.call(this, messages, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link QianfanOCRModel#chatSessionContinue}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinue(
    userMessage: string,
    images: Uint8Array[] | null,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) =>
        _nativeQianfanOcrChatStreamSessionContinue.call(this, userMessage, images, config ?? null, callback),
      signal,
    );
  }

  /** Streaming variant of {@link QianfanOCRModel#chatSessionContinueTool}. */
  // @ts-expect-error — override callback-based native method with AsyncGenerator
  async *chatStreamSessionContinueTool(
    toolCallId: string,
    content: string,
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    yield* _runChatStream(
      (callback) =>
        _nativeQianfanOcrChatStreamSessionContinueTool.call(this, toolCallId, content, config ?? null, callback),
      signal,
    );
  }
}

// -------------------------------------------------------------------
// Compile-time conformance check
// -------------------------------------------------------------------
//
// Ensures the wrapper structurally satisfies `SessionCapableModel` so
// `ChatSession<QianfanOCRModel>` will type-check in downstream code.
// The assignment is compile-only — the `null as unknown as T`
// placeholder never runs.
function _assertSessionCapable(): void {
  const _qianfan: SessionCapableModel = null as unknown as QianfanOCRModel;
  void _qianfan;
}
void _assertSessionCapable;
