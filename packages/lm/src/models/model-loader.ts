/**
 * Model loader utilities for Qwen3 models
 *
 * Handles loading pretrained weights from MLX format or converting from HuggingFace.
 */

import { Qwen3Model, Qwen35Model } from '@mlx-node/core';
import type { Qwen3_5Model } from '@mlx-node/core';

/**
 * Model loader for Qwen3 models
 */
export class ModelLoader {
  /**
   * Load a pretrained Qwen3 model from disk
   *
   * Delegates to Rust implementation for efficient loading without JavaScript memory limits.
   *
   * @param modelPath - Path to the model directory or file
   * @param _deviceMap - Device placement (not used in MLX, kept for compatibility)
   * @returns Loaded model
   */
  static async loadPretrained(modelPath: string, _deviceMap: string = 'auto'): Promise<Qwen3Model> {
    // Delegate to Rust implementation for efficient loading
    return await Qwen3Model.loadPretrained(modelPath);
  }

  /**
   * Load a pretrained Qwen3.5 model from disk
   *
   * Supports both dense and MoE variants.
   *
   * @param modelPath - Path to the model directory
   * @returns Loaded Qwen3.5 model
   */
  static async loadQwen35(modelPath: string): Promise<Qwen3_5Model> {
    return await Qwen35Model.loadPretrained(modelPath);
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
