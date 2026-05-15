/**
 * WASM Loader for MLX Browser
 *
 * Initializes the WASM module with WebGPU bridge imports injected
 * via NAPI-RS's `overwriteImports` mechanism.
 */

import { instantiateNapiModule, getDefaultContext, WASI } from '@napi-rs/wasm-runtime';

import webgpuWorkerUrl from './webgpu-worker.mjs?worker&url';
import { createWebGPUBridge } from './webgpu-bridge.js';
import { workerAssetUrl } from './worker-asset-url.js';

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

  // WASM exception tag for C++ exceptions (used by -fwasm-exceptions)
  const cppExceptionTag = new WebAssembly.Tag({ parameters: ['i32'] });

  const cxxStubs = {
    __cpp_exception: cppExceptionTag,
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
      return new Worker(workerAssetUrl(webgpuWorkerUrl), { type: 'module' });
    },
    overwriteImports(importObject: Record<string, Record<string, unknown>>) {
      importObject.env = {
        ...importObject.env,
        ...importObject.napi,
        ...importObject.emnapi,
        memory: sharedMemory,
        ...cxxStubs,
        // Task 4's SabSink declares __wasm_i32_atomic_wait /
        // __wasm_atomic_notify as extern "C" — wasm-ld emits them as host
        // imports, so provide JS stubs wrapping Atomics.wait / notify.
        // Re-read sharedMemory.buffer each call because memory.grow() replaces
        // the buffer object; a cached view would point at the old, smaller
        // range and throw RangeError on indices beyond the old length.
        //
        // initMLX() runs on the main/page thread, where Atomics.wait() is
        // disallowed — calling it throws TypeError. The shim is only
        // reachable from SAB-streaming paths (chat-stream-sab), which must
        // run inside a Worker. Rather than silently throwing from deep
        // inside the WASM, fail loud with an actionable message so the
        // caller knows to move their MLX session into a worker. See codex
        // review baz8oy567 P2.
        __wasm_i32_atomic_wait: (ptr: number, expected: number, timeoutNs: bigint) => {
          const isWorker =
            typeof WorkerGlobalScope !== 'undefined' &&
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (self as any) instanceof (WorkerGlobalScope as any);
          if (!isWorker) {
            throw new Error(
              'MLX SAB streaming (chat-stream-sab) requires a Worker context — ' +
                'Atomics.wait() is disallowed on the main thread. Run initMLX() ' +
                'inside a Web Worker or use the mlx-worker.ts entry point.',
            );
          }
          const view = new Int32Array(sharedMemory.buffer);
          const timeoutMs = timeoutNs === -1n ? Infinity : Number(timeoutNs / 1_000_000n);
          const result = Atomics.wait(view, ptr >>> 2, expected, timeoutMs);
          return result === 'ok' ? 0 : result === 'not-equal' ? 1 : 2;
        },
        __wasm_atomic_notify: (ptr: number, count: number) => {
          const view = new Int32Array(sharedMemory.buffer);
          return Atomics.notify(view, ptr >>> 2, count);
        },
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
