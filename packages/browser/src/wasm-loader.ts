/**
 * WASM Loader for MLX Browser
 *
 * Initializes the WASM module with WebGPU bridge imports injected
 * via NAPI-RS's `overwriteImports` mechanism.
 */

import { createWebGPUBridge } from './webgpu-bridge.js';

export interface MLXBrowserOptions {
  /** URL of the .wasm file */
  wasmUrl: string;
  /** Optional: pre-created GPUAdapter (auto-created if not provided) */
  adapter?: GPUAdapter;
  /** Optional: pre-created GPUDevice (auto-created if not provided) */
  device?: GPUDevice;
  /** Optional: device limits to request */
  requiredLimits?: Record<string, number>;
}

/**
 * Initialize the MLX WASM module with WebGPU support.
 *
 * Must be called from a Web Worker (requires SharedArrayBuffer).
 * Requires Cross-Origin-Isolated headers on the page.
 */
export async function initMLX(options: MLXBrowserOptions) {
  // 1. Check WebGPU availability
  if (!navigator.gpu) {
    throw new Error('WebGPU is not available in this browser');
  }

  // 2. Create or use provided adapter/device
  const adapter =
    options.adapter ?? (await navigator.gpu.requestAdapter());
  if (!adapter) {
    throw new Error('Failed to get WebGPU adapter');
  }

  const defaultLimits: Record<string, number> = {
    maxStorageBuffersPerShaderStage: 16,
    maxBufferSize: 1 << 30, // 1GB
    maxStorageBufferBindingSize: 1 << 30,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupsPerDimension: 65535,
    maxBindGroups: 4,
    maxBindingsPerBindGroup: 16,
  };

  const device =
    options.device ??
    (await adapter.requestDevice({
      requiredLimits: options.requiredLimits ?? defaultLimits,
    }));

  // 3. Create the WebGPU bridge
  const bridge = createWebGPUBridge(adapter, device);

  // 4. Load WASM with bridge injected
  // The NAPI-RS generated loader handles emnapi initialization.
  // We inject our WebGPU imports via overwriteImports.

  // Dynamic import of the generated WASM loader
  // (path will be configured at build time)
  const { instantiateNapiModule } = await import('@napi-rs/wasm-runtime');

  const wasmResponse = fetch(options.wasmUrl);

  const module = await instantiateNapiModule(wasmResponse, {
    asyncWorkPoolSize: 4,
    overwriteImports(importObject: Record<string, Record<string, unknown>>) {
      // Inject WebGPU bridge functions into the env namespace
      importObject.env = {
        ...importObject.env,
        ...importObject.napi,
        ...importObject.emnapi,
        ...bridge.imports,
      };
      return importObject;
    },
  });

  return {
    /** The NAPI module exports (same API as @mlx-node/core) */
    exports: module.exports,
    /** The WebGPU device used */
    device,
    /** The WebGPU adapter used */
    adapter,
  };
}
