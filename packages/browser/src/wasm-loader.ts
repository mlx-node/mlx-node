/**
 * WASM Loader for MLX Browser
 *
 * Initializes the WASM module with WebGPU bridge imports injected
 * via NAPI-RS's `overwriteImports` mechanism.
 */

import {
  instantiateNapiModule,
  getDefaultContext,
  WASI,
} from '@napi-rs/wasm-runtime';

import { createWebGPUBridge } from './webgpu-bridge.js';

export interface MLXBrowserOptions {
  /** URL of the .wasm file (debug or optimized) */
  wasmUrl: string;
  /** Optional: pre-created GPUAdapter */
  adapter?: GPUAdapter;
  /** Optional: pre-created GPUDevice */
  device?: GPUDevice;
}

/**
 * Initialize the MLX WASM module with WebGPU support.
 *
 * Requires Cross-Origin-Isolated headers (COOP/COEP) for SharedArrayBuffer.
 */
export async function initMLX(options: MLXBrowserOptions) {
  // 1. Create or use provided adapter/device
  if (!navigator.gpu) {
    throw new Error('WebGPU is not available in this browser');
  }

  const adapter = options.adapter ?? (await navigator.gpu.requestAdapter());
  if (!adapter) throw new Error('Failed to get WebGPU adapter');

  const device =
    options.device ??
    (await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30),
      },
    }));

  // 2. Create the WebGPU bridge
  const bridge = createWebGPUBridge(adapter, device);

  // 3. Load WASM with bridge injected
  const wasi = new WASI({ version: 'preview1' });
  const context = getDefaultContext();
  const sharedMemory = new WebAssembly.Memory({
    initial: 16000,
    maximum: 65536,
    shared: true,
  });

  const wasmFile = await fetch(options.wasmUrl).then((res) => res.arrayBuffer());

  // C++ exception stubs (libc++abi leaves 3 symbols as imports)
  const cxxStubs = {
    __cxa_allocate_exception: (size: number) =>
      (wasmExports?.malloc as Function)?.(size) ?? 0,
    __cxa_throw: () => {
      throw new Error('C++ exception thrown in WASM');
    },
    __cxa_init_primary_exception: (ptr: number) => ptr,
    // mlx::core::gpu::init() — GPU is pre-initialized via bridge
    _ZN3mlx4core3gpu4initEv: () => {},
  };

  let wasmExports: WebAssembly.Exports | null = null;

  const { napiModule } = await instantiateNapiModule(wasmFile, {
    context,
    asyncWorkPoolSize: 4,
    wasi,
    onCreateWorker() {
      // Workers get real WebGPU bridge (each creates its own GPUDevice)
      return new Worker(
        new URL('./webgpu-worker.mjs', import.meta.url),
        { type: 'module' },
      );
    },
    overwriteImports(importObject: Record<string, Record<string, unknown>>) {
      importObject.env = {
        ...importObject.env,
        ...importObject.napi,
        ...importObject.emnapi,
        memory: sharedMemory,
        ...cxxStubs,
        // Real WebGPU bridge (not stubs)
        ...bridge.imports,
      };
      return importObject;
    },
    beforeInit({ instance }) {
      wasmExports = instance.exports;
      // Give bridge access to WASM memory, function table, malloc/free
      bridge.setInstance(instance);
      // Register NAPI modules
      for (const name of Object.keys(instance.exports)) {
        if (name.startsWith('__napi_register__')) {
          (instance.exports[name] as Function)();
        }
      }
    },
  });

  return {
    exports: napiModule.exports,
    device,
    adapter,
    bridge,
  };
}
