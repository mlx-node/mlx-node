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
      await new Promise(r => setTimeout(r, 1));
    }

    const bridge = createBridgeStub(
      rpcConfig.cmdBuffer,
      wasmMemory,
      rpcConfig.handles,
      rpcConfig.readbackBuffer,
      rpcConfig.features,
    );

    const wasi = new WASI({
      print: function () { console.log.apply(console, arguments); },
      printErr: function () { console.error.apply(console, arguments); },
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
          // Stream stubs — streaming is handled by chatStream's NAPI callback
          mlx_stream_write: () => {},
          mlx_stream_reset: () => {},
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
