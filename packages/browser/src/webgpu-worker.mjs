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
import {
  STREAM_BUFFER_SIZE,
  STREAM_HEADER_SIZE,
  STREAM_TEXT_OFFSET,
} from './rpc-protocol.js';

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

    // Stream channel stubs — write decoded tokens to SharedArrayBuffer
    const streamBuffer = rpcConfig.streamBuffer;
    const streamView = streamBuffer ? new Uint8Array(streamBuffer) : null;
    const streamI32 = streamBuffer ? new Int32Array(streamBuffer) : null;

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
          // Stream channel for token streaming (same impl as mlx-worker.ts)
          mlx_stream_write: (ptr, len) => {
            if (!streamView || !streamI32) return;
            const maxLen = STREAM_BUFFER_SIZE - STREAM_TEXT_OFFSET;
            if (len > maxLen) len = maxLen;
            const wasmMem = new Uint8Array(wasmMemory.buffer);
            const text = wasmMem.slice(ptr, ptr + len);
            streamView.set(text, STREAM_TEXT_OFFSET);
            Atomics.store(streamI32, 0, len);
            Atomics.add(streamI32, 1, 1);
            Atomics.notify(streamI32, 1);
          },
          mlx_stream_reset: () => {
            if (!streamI32) return;
            Atomics.store(streamI32, 0, 0);
            Atomics.store(streamI32, 1, 0);
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
