/**
 * MLX-Node - High-Performance ML Framework for Node.js
 *
 * This is the backward-compatibility entry point that re-exports
 * everything from all packages.
 *
 * For new projects, prefer importing directly from the packages:
 * - @mlx-node/core - Core array operations, NN layers, transformers, native bindings
 * - @mlx-node/lm - Model loading, configs (aligned with mlx-lm)
 * - @mlx-node/trl - GRPO training utilities (aligned with TRL)
 */

// Re-export everything from all packages for backward compatibility
export * from '@mlx-node/core';
export * from '@mlx-node/lm';
export * from '@mlx-node/trl';
