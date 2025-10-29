/**
 * Entropy filtering utilities for GRPO training
 *
 * Reference: trl/trl/trainer/grpo_trainer.py:get_high_entropy_mask
 *
 * Implements selective training on high-entropy (uncertain) tokens,
 * which is a key optimization in GRPO to focus learning on challenging predictions.
 *
 * This module re-exports the Rust implementation for optimal performance.
 * All entropy filtering operations are implemented in Rust (node/src/grpo_entropy.rs).
 */

// Note: getHighEntropyMask and computeEntropy are exported from the main index.ts via NAPI
// They are not re-exported here to avoid duplicate export errors

/**
 * Configuration for entropy-based filtering in GRPO training
 */
export interface EntropyFilteringConfig {
  /**
   * Whether to enable entropy filtering (default: false)
   */
  enabled: boolean;

  /**
   * Quantile threshold for selecting high-entropy tokens (default: 0.8)
   * - 0.0: all non-pad tokens
   * - 0.5: top 50% highest entropy
   * - 0.8: top 20% highest entropy (recommended)
   * - 1.0: only highest entropy token
   */
  topEntropyQuantile: number;
}

/**
 * Default entropy filtering configuration
 */
export const DEFAULT_ENTROPY_CONFIG: EntropyFilteringConfig = {
  enabled: false,
  topEntropyQuantile: 0.8,
};
