/**
 * @mlx-node/browser - MLX inference in the browser via WebGPU + WASM
 *
 * Usage:
 *   import { initMLX } from '@mlx-node/browser';
 *   const mlx = await initMLX({ wasmUrl: '/mlx-core.wasm32-wasi.wasm' });
 *   // mlx.exports has the same API as @mlx-node/core
 */

export { initMLX } from './wasm-loader.js';
export type { MLXBrowserOptions } from './wasm-loader.js';
export { createWebGPUBridge } from './webgpu-bridge.js';
export type { WebGPUBridge } from './webgpu-bridge.js';
