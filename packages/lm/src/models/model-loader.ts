/**
 * Model loading utilities for Qwen3 models
 *
 * Handles loading pretrained weights from MLX format or converting from HuggingFace.
 */

import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { Qwen3Model, QianfanOCRModel } from '@mlx-node/core';

import type { LoadableModel, TrainableModel } from '../interfaces.js';
import { Qwen35Model, Qwen35MoeModel } from '../stream.js';

export type ModelType = 'qwen3' | 'qwen3_5' | 'qwen3_5_moe' | 'internvl_chat' | 'qianfan-ocr';

const SUPPORTED_MODEL_TYPES = new Set<ModelType>(['qwen3', 'qwen3_5', 'qwen3_5_moe', 'internvl_chat', 'qianfan-ocr']);

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
      return Qwen3Model.load(modelPath);
    case 'internvl_chat':
    case 'qianfan-ocr':
      return QianfanOCRModel.load(modelPath) as unknown as Promise<LoadableModel>;
  }
}

export async function detectModelType(modelPath: string): Promise<ModelType> {
  try {
    const raw = await readFile(join(modelPath, 'config.json'), 'utf-8');
    const config = JSON.parse(raw);
    const modelType = config.model_type ?? 'qwen3';
    if (!SUPPORTED_MODEL_TYPES.has(modelType)) {
      throw new Error(`Unsupported model_type "${modelType}" in ${modelPath}/config.json`);
    }
    return modelType;
  } catch (e) {
    if (e instanceof Error && e.message.startsWith('Unsupported model_type')) throw e;
    throw new Error(`Cannot detect model type: config.json not found in ${modelPath}`);
  }
}
