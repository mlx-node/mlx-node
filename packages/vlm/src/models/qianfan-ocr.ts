import { QianfanOCRModel as QianfanOCRModelNative } from '@mlx-node/core';
import { makeStreamingModel } from '@mlx-node/lm';
import type { SessionCapableModel } from '@mlx-node/lm';

/**
 * Qianfan-OCR Vision-Language Model wrapper.
 *
 * Built from the shared {@link makeStreamingModel} factory in
 * `@mlx-node/lm` — the empty `extends` inherits the AsyncGenerator
 * session-streaming overrides (`chatStreamSessionStart` /
 * `chatStreamSessionContinue` / `chatStreamSessionContinueTool`) so the
 * wrapper structurally satisfies `SessionCapableModel` and can be
 * passed to `ChatSession<QianfanOCRModel>`. Importing the factory from
 * `@mlx-node/lm` is one-directional (vlm → lm), matching the existing
 * `_runChatStream` dependency, so it introduces no circular dependency.
 *
 * Qianfan-OCR is a VLM (InternViT + Qwen3 language model). The continue
 * path cannot splice new vision features into a live KV cache — image
 * changes always require a fresh session start, which the high-level
 * `ChatSession` wrapper handles via its `lastImagesKey` check.
 *
 * Unlike the LM families, Qianfan-OCR does NOT record its model path
 * (`recordModelPath: false`) and therefore exposes no
 * `applyChatTemplate` — preserving its historical surface.
 */
export class QianfanOCRModel extends makeStreamingModel(QianfanOCRModelNative, { recordModelPath: false }) {}

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
