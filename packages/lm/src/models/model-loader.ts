/**
 * Model loader utilities for Qwen3 models
 *
 * Handles loading pretrained weights from MLX format or converting from HuggingFace.
 */

import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Qwen3Model } from '@mlx-node/core';
import type { Qwen3_5Model, Qwen3_5MoeModel } from '@mlx-node/core';
import { Qwen35Model, Qwen35MoeModel } from '../stream';

/**
 * Model loader for Qwen3 and Qwen3.5 models
 */
export class ModelLoader {
  /**
   * Load a pretrained Qwen3 model from disk.
   *
   * For Qwen3.5 models, use {@link loadQwen35} instead.
   * If the config.json indicates a qwen3_5 model, this method will
   * automatically delegate to loadQwen35.
   *
   * @param modelPath - Path to the model directory or file
   * @returns Loaded model
   */
  static async loadPretrained(modelPath: string): Promise<Qwen3Model | Qwen3_5Model | Qwen3_5MoeModel> {
    const modelType = await detectModelType(modelPath);

    if (modelType === 'qwen3_5_moe') {
      return (await Qwen35MoeModel.loadPretrained(modelPath)) as unknown as Qwen3_5MoeModel;
    }

    if (modelType === 'qwen3_5') {
      // load_pretrained() auto-detects vision weights and loads encoder if present
      return (await Qwen35Model.loadPretrained(modelPath)) as unknown as Qwen3_5Model;
    }

    return await Qwen3Model.loadPretrained(modelPath);
  }

  /**
   * Load a pretrained Qwen3.5 dense model from disk
   *
   * @param modelPath - Path to the model directory
   * @returns Loaded Qwen3.5 model
   */
  static async loadQwen35(modelPath: string): Promise<Qwen3_5Model> {
    return (await Qwen35Model.loadPretrained(modelPath)) as unknown as Qwen3_5Model;
  }

  /**
   * Load a pretrained Qwen3.5 MoE model from disk
   *
   * @param modelPath - Path to the model directory
   * @returns Loaded Qwen3.5 MoE model
   */
  static async loadQwen35Moe(modelPath: string): Promise<Qwen3_5MoeModel> {
    return (await Qwen35MoeModel.loadPretrained(modelPath)) as unknown as Qwen3_5MoeModel;
  }

  /**
   * Save model configuration and metadata to disk
   *
   * This delegates to the Rust implementation which efficiently handles
   * model saving without running into JavaScript memory/array size limits.
   *
   * Note: This saves configuration and parameter metadata only.
   * For full model weight serialization, use safetensors or binary format.
   */
  static saveModel(model: Qwen3Model, savePath: string): Promise<void> {
    // Delegate to Rust implementation for efficient saving
    return model.saveModel(savePath);
  }
}

async function detectModelType(modelPath: string): Promise<string> {
  try {
    const raw = await readFile(join(modelPath, 'config.json'), 'utf-8');
    const config = JSON.parse(raw);
    return config.model_type ?? 'qwen3';
  } catch {
    return 'qwen3';
  }
}
