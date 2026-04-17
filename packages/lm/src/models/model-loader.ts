/**
 * Model loading utilities for Qwen3 models
 *
 * Handles loading pretrained weights from MLX format or converting from HuggingFace.
 */

import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { HarrierModel, QianfanOCRModel } from '@mlx-node/core';

import { ChatSession, type SessionCapableModel } from '../chat-session.js';
import type { LoadableModel, TrainableModel } from '../interfaces.js';
import { Gemma4Model, Lfm2Model, Qwen3Model, Qwen35Model, Qwen35MoeModel } from '../stream.js';

export type ModelType =
  | 'qwen3'
  | 'qwen3_5'
  | 'qwen3_5_moe'
  | 'internvl_chat'
  | 'qianfan-ocr'
  | 'harrier'
  | 'gemma4'
  | 'lfm2';

const SUPPORTED_MODEL_TYPES = new Set<ModelType>([
  'qwen3',
  'qwen3_5',
  'qwen3_5_moe',
  'internvl_chat',
  'qianfan-ocr',
  'harrier',
  'gemma4',
  'lfm2',
]);

/**
 * Load a model from disk, auto-detecting architecture from config.json.
 *
 * Supports both language models (Qwen3, Qwen3.5) and vision-language models
 * (Qianfan-OCR / InternVL). Use `instanceof` to narrow the returned type.
 */
export async function loadModel(modelPath: string): Promise<LoadableModel> {
  const modelType = await detectModelType(modelPath);

  switch (modelType) {
    case 'qwen3_5_moe':
      return Qwen35MoeModel.load(modelPath) as unknown as Promise<TrainableModel>;
    case 'qwen3_5':
      return Qwen35Model.load(modelPath) as unknown as Promise<TrainableModel>;
    case 'qwen3':
      return Qwen3Model.load(modelPath) as unknown as Promise<TrainableModel>;
    case 'harrier':
      return HarrierModel.load(modelPath) as unknown as Promise<LoadableModel>;
    case 'internvl_chat':
    case 'qianfan-ocr':
      return QianfanOCRModel.load(modelPath) as unknown as Promise<LoadableModel>;
    case 'gemma4':
      return Gemma4Model.load(modelPath) as unknown as Promise<LoadableModel>;
    case 'lfm2':
      return Lfm2Model.load(modelPath) as unknown as Promise<LoadableModel>;
  }
}

/**
 * Load a model and wrap it in a {@link ChatSession} for multi-turn chat.
 *
 * Convenience around `loadModel()` + `new ChatSession(model)` for the
 * common case where a caller just wants an ergonomic session handle.
 *
 * Rejects models that cannot be driven by a `ChatSession`:
 *   - Embedding models (`HarrierModel`) have no chat surface.
 *   - The native `QianfanOCRModel` exposes callback-based streaming
 *     methods that do not structurally satisfy `SessionCapableModel`'s
 *     `AsyncGenerator` overloads. The VLM AsyncGenerator wrapper lives
 *     in `@mlx-node/vlm` (importing it here would create a circular
 *     package dependency), so callers who want a Qianfan-OCR session
 *     must import `QianfanOCRModel` from `@mlx-node/vlm` and construct
 *     `new ChatSession(model)` directly.
 */
export async function loadSession(modelPath: string): Promise<ChatSession<SessionCapableModel>> {
  const m = await loadModel(modelPath);
  if (m instanceof HarrierModel) {
    throw new Error('loadSession: embedding models (Harrier) cannot be wrapped in a ChatSession');
  }
  if (m instanceof QianfanOCRModel) {
    throw new Error(
      'loadSession: Qianfan-OCR / InternVL session support lives in @mlx-node/vlm. Import QianfanOCRModel from @mlx-node/vlm and construct ChatSession(model) directly.',
    );
  }
  return new ChatSession(m as unknown as SessionCapableModel);
}

export async function detectModelType(modelPath: string): Promise<ModelType> {
  try {
    const raw = await readFile(join(modelPath, 'config.json'), 'utf-8');
    const config = JSON.parse(raw);
    const rawModelType: string = config.model_type ?? 'qwen3';

    // Normalize model_type: gemma4_text → gemma4
    let modelType: ModelType = (rawModelType === 'gemma4_text' ? 'gemma4' : rawModelType) as ModelType;

    // Detect embedding models: Qwen3 backbone with base architecture (no ForCausalLM)
    if (modelType === 'qwen3') {
      const architectures: string[] = config.architectures ?? [];
      if (architectures.includes('Qwen3Model') && !architectures.includes('Qwen3ForCausalLM')) {
        modelType = 'harrier';
      }
    }

    if (!SUPPORTED_MODEL_TYPES.has(modelType)) {
      throw new Error(`Unsupported model_type "${modelType}" in ${modelPath}/config.json`);
    }
    return modelType;
  } catch (e) {
    if (e instanceof Error && e.message.startsWith('Unsupported model_type')) throw e;
    throw new Error(`Cannot detect model type: config.json not found in ${modelPath}`);
  }
}
