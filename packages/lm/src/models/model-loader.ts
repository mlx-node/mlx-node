/**
 * Model loading utilities for Qwen3 models
 *
 * Handles loading pretrained weights from MLX format or converting from HuggingFace.
 */

import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Qwen3Model } from '@mlx-node/core';
import { Qwen35Model, Qwen35MoeModel } from '../stream';
import type { TrainableModel } from '../interfaces';

export type ModelType = 'qwen3' | 'qwen3_5' | 'qwen3_5_moe';

const SUPPORTED_MODEL_TYPES = new Set<ModelType>(['qwen3', 'qwen3_5', 'qwen3_5_moe']);

/**
 * Load a language model from disk, auto-detecting architecture from config.json.
 */
export async function loadModel(modelPath: string): Promise<TrainableModel> {
  const modelType = await detectModelType(modelPath);

  switch (modelType) {
    case 'qwen3_5_moe':
      // Cast: stream wrapper extends native — safe for instanceof and engine factories
      return Qwen35MoeModel.load(modelPath) as unknown as Promise<TrainableModel>;
    case 'qwen3_5':
      // load() auto-detects vision weights and loads encoder if present
      return Qwen35Model.load(modelPath) as unknown as Promise<TrainableModel>;
    case 'qwen3':
      return Qwen3Model.load(modelPath);
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
