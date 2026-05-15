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

// Revision marker: changes worker asset URLs after enabling COOP/COEP headers
// on Void, avoiding stale immutable edge-cache entries without those headers.
globalThis.__MLX_VOID_COEP_WORKER_ASSET_REV = '2026-05-15';

// RPC config received from mlx-worker before emnapi's init message
let rpcConfig = null;
let wasmInstanceRef = null;
let wasmMemoryRef = null;
const cppExceptionTag = new WebAssembly.Tag({ parameters: ['i32'] });

function extractWasmExceptionMessage(err) {
  if (!(err instanceof WebAssembly.Exception) || !wasmMemoryRef) return null;
  try {
    const ptr = err.getArg(cppExceptionTag, 0) >>> 0;
    const mem = new Uint8Array(wasmMemoryRef.buffer);
    const view = new DataView(mem.buffer);
    const found = [];
    for (const off of [4, 8, 12, 16, 20, 24, 28, 32]) {
      if (ptr + off + 4 >= mem.byteLength) continue;
      const strPtr = view.getUint32(ptr + off, true);
      if (strPtr <= 1024 || strPtr >= mem.byteLength - 4) continue;
      let s = '';
      for (let i = strPtr; i < mem.byteLength && mem[i] !== 0 && i - strPtr < 512; i++) {
        const ch = mem[i];
        if (ch >= 32 && ch < 127) s += String.fromCharCode(ch);
        else {
          s = '';
          break;
        }
      }
      if (s.length > 4) found.push(`@${off}:${s}`);
    }
    return found.length > 0 ? `C++ exception ptr=${ptr} ${found.join(' | ')}` : `C++ exception ptr=${ptr}`;
  } catch (decodeErr) {
    return `C++ exception (decode failed: ${decodeErr instanceof Error ? decodeErr.message : String(decodeErr)})`;
  }
}

function formatWorkerError(err) {
  if (err instanceof ErrorEvent) {
    const inner = formatWorkerError(err.error);
    const location =
      err.filename || err.lineno || err.colno
        ? `${err.filename}:${err.lineno}:${err.colno}`
        : '';
    return {
      message: err.message || inner.message,
      stack: inner.stack || location,
    };
  }
  const wasmMessage = extractWasmExceptionMessage(err);
  if (wasmMessage) return { message: wasmMessage, stack: err?.stack };
  if (err instanceof Error) return { message: err.message || String(err), stack: err.stack };
  return { message: typeof err === 'string' ? err : String(err), stack: undefined };
}

function reportWorkerError(prefix, err) {
  const details = formatWorkerError(err);
  const message = `${prefix}: ${details.message}`;
  console.error(`[WASM child] ${message}`, details.stack || '');
  try {
    globalThis.postMessage({
      type: '__mlx_child_error',
      message,
      stack: details.stack,
    });
  } catch {
    // Ignore postMessage failures while the emnapi worker is tearing down.
  }
}

globalThis.addEventListener?.('error', (event) => {
  reportWorkerError('Worker error', event.error ?? event);
});

globalThis.addEventListener?.('unhandledrejection', (event) => {
  reportWorkerError('Unhandled rejection', event.reason);
});

const handler = new MessageHandler({
  async onLoad({ wasmModule, wasmMemory }) {
    wasmMemoryRef = wasmMemory;
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
        const line = Array.from(arguments).map(String).join(' ');
        if (line.includes('[MLX-KERNEL]')) return;
        console.log.apply(console, arguments);
      },
      printErr: function () {
        const line = Array.from(arguments).map(String).join(' ');
        if (line.includes('[MLX-KERNEL]')) return;
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
          __cpp_exception: cppExceptionTag,
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
        wasmInstanceRef = instance;
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
