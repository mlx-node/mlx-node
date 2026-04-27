/**
 * MLX Web Worker — child thread for model inference
 *
 * Uses the shared RPC bridge (Atomics.wait → gpu-worker) for all WebGPU calls.
 * The direct bridge (own GPUDevice per child worker) deadlocks because
 * wgpuBufferMapAsync registers a JS Promise .then() callback, but raw_ptr()'s
 * C++ polling loop calls poll_instance() which is a no-op on WASM — the event
 * loop never runs, the callback never fires, infinite loop.
 *
 * The RPC bridge avoids this: wgpuBufferMapAsync blocks via Atomics.wait,
 * the gpu-worker (whose event loop is free) processes the async mapAsync,
 * writes the callback result back, and drainCallbacks fires it synchronously
 * before the RPC call returns.
 */
import { instantiateNapiModuleSync, MessageHandler, WASI } from '@napi-rs/wasm-runtime';

import { createBridgeStub } from './webgpu-bridge-stub.js';

// RPC config received from mlx-worker before emnapi's init message
let rpcConfig = null;

const handler = new MessageHandler({
  async onLoad({ wasmModule, wasmMemory }) {
    // Wait for RPC config (arrives before emnapi's init message)
    while (!rpcConfig) {
      await new Promise((r) => setTimeout(r, 1));
    }

    const bridge = createBridgeStub(
      rpcConfig.cmdBuffer,
      wasmMemory,
      rpcConfig.handles,
      rpcConfig.readbackBuffer,
      rpcConfig.features,
      rpcConfig.poolStatsBuffer, // may be undefined — guarded in stub
      rpcConfig.dispatchBatchBuffer, // Phase 2 — may be undefined
      rpcConfig.dispatchBatch === true,
      rpcConfig.batchBufferId ?? 0, // Phase 2b — per-worker batch buffer ID
      rpcConfig.bufferMetadataBuffer, // Task 3 — shared (size, usage) table
      rpcConfig.statsBuffer, // JS-F010 — shared GET_STATS histogram SAB
      {
        fusionEnabled: rpcConfig.fusionEnabled !== false,
        passCachingEnabled: rpcConfig.passCachingEnabled !== false,
        optimizationControlBuffer: rpcConfig.optimizationControlBuffer,
      },
    );

    const wasi = new WASI({
      print: function () {
        console.log.apply(console, arguments);
      },
      printErr: function () {
        console.error.apply(console, arguments);
      },
    });

    return instantiateNapiModuleSync(wasmModule, {
      childThread: true,
      wasi,
      overwriteImports(importObject) {
        importObject.env = {
          ...importObject.env,
          ...importObject.napi,
          ...importObject.emnapi,
          memory: wasmMemory,
          __cpp_exception: new WebAssembly.Tag({ parameters: ['i32'] }),
          _ZN3mlx4core3gpu4initEv: () => {},
          // Task 4's SabSink declares __wasm_i32_atomic_wait /
          // __wasm_atomic_notify as extern "C" — wasm-ld emits them as host
          // imports, so provide JS stubs wrapping Atomics.wait / notify.
          // Re-read wasmMemory.buffer each call because memory.grow() replaces
          // the buffer object; a cached view would point at the old, smaller
          // range and throw RangeError on indices beyond the old length.
          __wasm_i32_atomic_wait: (ptr, expected, timeoutNs) => {
            const view = new Int32Array(wasmMemory.buffer);
            const timeoutMs = timeoutNs === -1n ? Infinity : Number(timeoutNs / 1_000_000n);
            const result = Atomics.wait(view, ptr >>> 2, expected, timeoutMs);
            return result === 'ok' ? 0 : result === 'not-equal' ? 1 : 2;
          },
          __wasm_atomic_notify: (ptr, count) => {
            const view = new Int32Array(wasmMemory.buffer);
            return Atomics.notify(view, ptr >>> 2, count);
          },
          ...bridge.imports,
        };
      },
      beforeInit({ instance }) {
        bridge.setInstance(instance);
      },
    });
  },
});

globalThis.onmessage = function (e) {
  // Intercept RPC config before emnapi's handler sees it
  if (e.data?.type === '__mlx_rpc_config') {
    rpcConfig = e.data;
    return;
  }
  handler.handle(e);
};
