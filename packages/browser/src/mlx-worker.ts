/**
 * MLX Inference Worker
 *
 * Runs the entire MLX pipeline on a dedicated Web Worker:
 * - Creates its own GPUDevice (navigator.gpu works in workers)
 * - Loads WASM with real WebGPU bridge
 * - Fetches + loads model weights directly into GPU buffers
 * - Runs inference (chat/generate) without blocking the main thread
 *
 * The main thread communicates via postMessage only.
 */
import { Buffer } from 'buffer';
(globalThis as any).Buffer = Buffer;

// Patch TextDecoder to handle SharedArrayBuffer views.
// WASM memory is a SharedArrayBuffer (for threads support), but TextDecoder
// rejects shared views. Copy to a non-shared buffer before decoding.
const _origDecode = TextDecoder.prototype.decode;
TextDecoder.prototype.decode = function (input?: BufferSource | null, options?: TextDecodeOptions) {
  if (input && ArrayBuffer.isView(input) && (input.buffer as any) instanceof SharedArrayBuffer) {
    input = (input as Uint8Array).slice();
  }
  return _origDecode.call(this, input, options);
};

import { instantiateNapiModule, getDefaultContext, WASI } from '@napi-rs/wasm-runtime';

import { createSabRing } from './chat-stream-sab.js';
import { CMD_OFFSET, READBACK_BUFFER_SIZE } from './rpc-protocol.js';
import { parseSafeTensorsHeader, dtypeToCode } from './safetensors.js';
import { createBridgeStub } from './webgpu-bridge-stub.js';

let model: any = null;
let mlxExports: any = null;
let wasmInst: WebAssembly.Instance | null = null;
let cppTag: any = null;

function post(msg: any) {
  (self as any).postMessage(msg);
}

async function handleInit(data: { wasmUrl: string; modelUrl: string; packBf16?: boolean; sdpaFallback?: boolean }) {
  const packBf16 = data.packBf16 === true;
  const sdpaFallback = data.sdpaFallback === true;
  try {
    // 1. Spawn gpu-worker (owns GPUDevice, event loop free for GPU callbacks)
    post({ type: 'progress', step: 'gpu', message: 'Initializing WebGPU...' });
    const cmdBuffer = new SharedArrayBuffer(CMD_OFFSET.TOTAL);
    const readbackBuffer = new SharedArrayBuffer(READBACK_BUFFER_SIZE);
    // WASM module requires min 1002 pages (~66 MB). We use 4096 (~268 MB)
    // for headroom during WASM init (thread stacks, emnapi, etc.) and let
    // memory.grow expand as needed — keeps total well under the 2 GB JS
    // pointer limit (i32 > 2^31 wraps negative in TypedArray constructors).
    // Old value of 30000 pages (~1.97 GB) left almost no room for model data.
    const sharedMemory = new WebAssembly.Memory({
      initial: 4096,
      maximum: 65536,
      shared: true,
    });

    const gpuWorker = new Worker(new URL('./gpu-worker.ts', import.meta.url), { type: 'module' });

    // Wait for gpu-worker to create GPUDevice and be ready
    const gpuReady = await new Promise<any>((resolve, reject) => {
      gpuWorker.onmessage = (e) => {
        if (e.data.type === 'ready') resolve(e.data);
        else if (e.data.type === 'error') reject(new Error(e.data.message));
      };
      gpuWorker.postMessage({
        type: 'init',
        cmdBuffer,
        readbackBuffer,
        wasmMemory: sharedMemory, // Send Memory object, not .buffer — .buffer getter always returns current byteLength after grow()
      });
    });
    post({ type: 'progress', step: 'gpu', message: 'WebGPU ready' });

    // 2. Create bridge stub (RPC via Atomics.wait to gpu-worker)
    const bridge = createBridgeStub(
      cmdBuffer,
      sharedMemory,
      {
        instanceHandle: gpuReady.instanceHandle,
        adapterHandle: gpuReady.adapterHandle,
        deviceHandle: gpuReady.deviceHandle,
        queueHandle: gpuReady.queueHandle,
      },
      readbackBuffer,
      gpuReady.features,
    );

    // 3. Load WASM with bridge stub
    post({ type: 'progress', step: 'wasm', message: 'Loading WASM module...' });
    // Forward lines starting with [MLX-KERNEL] to the main thread so we can
    // verify which matmul kernel variants fire. The WASM stderr lands on
    // the mlx-worker's own devtools target which isn't readable from the
    // main frame console.
    const wasi = new WASI({
      version: 'preview1',
      print: function (...args: unknown[]) {
        const line = args.map(String).join(' ');
        if (line.includes('[MLX-KERNEL]')) {
          post({ type: 'log', message: line });
        } else {
          console.log(...args);
        }
      },
      printErr: function (...args: unknown[]) {
        const line = args.map(String).join(' ');
        if (line.includes('[MLX-KERNEL]')) {
          post({ type: 'log', message: line });
        } else {
          console.error(...args);
        }
      },
    } as any);
    const context = getDefaultContext();
    const wasmFile = await fetch(data.wasmUrl).then((r) => r.arrayBuffer());

    const cppExceptionTag = new WebAssembly.Tag({ parameters: ['i32'] });
    cppTag = cppExceptionTag;

    const cxxStubs = {
      __cpp_exception: cppExceptionTag,
      _ZN3mlx4core3gpu4initEv: () => {},
    };

    const { napiModule } = await instantiateNapiModule(wasmFile, {
      context,
      asyncWorkPoolSize: 4,
      wasi,
      onCreateWorker() {
        // Child workers (model thread) share the gpu-worker's GPUDevice via
        // the same RPC bridge as the main WASM thread. The direct bridge
        // (own GPUDevice) deadlocks because wgpuBufferMapAsync registers a
        // JS Promise .then() callback, but raw_ptr()'s polling loop calls
        // poll_instance() which is a no-op on WASM — the event loop never
        // runs, the callback never fires, infinite loop.
        const w = new Worker(new URL('./webgpu-worker.mjs', import.meta.url), { type: 'module' });
        w.postMessage({
          type: '__mlx_rpc_config',
          cmdBuffer,
          readbackBuffer,
          handles: {
            instanceHandle: gpuReady.instanceHandle,
            adapterHandle: gpuReady.adapterHandle,
            deviceHandle: gpuReady.deviceHandle,
            queueHandle: gpuReady.queueHandle,
          },
          features: gpuReady.features,
        });
        return w;
      },
      // getTable runs AFTER WebAssembly.instantiate but BEFORE _initialize.
      // The C++ Device constructor runs during _initialize and busy-waits for
      // adapter/device callbacks that need wasmTable to resolve function pointers.
      getTable(exports: WebAssembly.Exports) {
        const table = exports.__indirect_function_table as WebAssembly.Table;
        bridge.setInstance({ exports } as unknown as WebAssembly.Instance);
        return table;
      },
      overwriteImports(importObject: Record<string, Record<string, unknown>>) {
        importObject.env = {
          ...importObject.env,
          ...importObject.napi,
          ...importObject.emnapi,
          memory: sharedMemory,
          ...cxxStubs,
          ...bridge.imports,
        };
        return importObject;
      },
      beforeInit({ instance }) {
        wasmInst = instance;
        bridge.setInstance(instance);
        for (const name of Object.keys(instance.exports)) {
          if (name.startsWith('__napi_register__')) {
            (instance.exports[name] as Function)();
          }
        }
      },
    });

    mlxExports = napiModule.exports;
    // Flip the backend's packed-bf16 flag BEFORE any model/weight work runs.
    // wgpu_buffer() checks this flag on first upload to decide whether an
    // eligible bf16 weight gets flipped into StorageMode::PackedBf16. Toggling
    // after the fact is a no-op for already-uploaded buffers.
    if (typeof mlxExports.wgpuSetPackedBf16Enabled === 'function') {
      mlxExports.wgpuSetPackedBf16Enabled(packBf16);
    }
    // Flip the SDPA fallback kill-switch. When true, the WebGPU backend's
    // fast/tile SDPA kernels are bypassed in favor of the decomposed
    // matmul→softmax→matmul path — used by the demo to A/B the fused
    // kernels against the baseline without a rebuild.
    if (typeof mlxExports.wgpuSetSdpaFallbackForced === 'function') {
      mlxExports.wgpuSetSdpaFallbackForced(sdpaFallback);
    }
    const flagSuffix = (packBf16 ? ' pack_bf16=1' : '') + (sdpaFallback ? ' sdpa_fallback=1' : '');
    post({
      type: 'progress',
      step: 'wasm',
      message: `WASM loaded${flagSuffix ? ' (' + flagSuffix.trim() + ')' : ''}`,
    });

    // 3. Fetch model files
    post({ type: 'progress', step: 'model', message: 'Fetching config...' });
    const configJson = await fetch(`${data.modelUrl}/config.json`).then((r) => r.text());

    post({ type: 'progress', step: 'model', message: 'Fetching tokenizer...' });
    const tokenizerJson = await fetch(`${data.modelUrl}/tokenizer.json`).then((r) => r.text());
    // Fetch tokenizer_config.json for the Jinja2 chat template
    let tokenizerConfigJson: string | undefined;
    try {
      const resp = await fetch(`${data.modelUrl}/tokenizer_config.json`);
      if (resp.ok) {
        tokenizerConfigJson = await resp.text();
      }
    } catch (e) {
      console.warn('tokenizer_config.json not available, using default chat template');
    }

    post({ type: 'progress', step: 'model', message: 'Fetching weights...' });
    const weightsResponse = await fetch(`${data.modelUrl}/model.safetensors`);
    const totalSize = Number(weightsResponse.headers.get('content-length') || 0);
    const reader = weightsResponse.body!.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (totalSize > 0 && loaded % (50 * 1024 * 1024) < value.length) {
        const pct = ((loaded / totalSize) * 100).toFixed(0);
        post({ type: 'progress', step: 'download', message: `Downloading weights... ${pct}%`, pct: Number(pct) });
      }
    }

    const weightsBuffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      weightsBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    // 4. Per-tensor CPU data path
    //
    // For each tensor: malloc small WASM buffer → copy bytes → call addCpuTensor
    // (Rust creates MLX array, C++ copies data) → free WASM buffer immediately.
    //
    // This keeps peak WASM pointer well under 2 GB (only one tensor buffer
    // alive at a time), avoiding the i32 offset overflow that kills bulk loading.
    // After all tensors, buildModelFromCpuTensors drains the Rust accumulator.
    post({ type: 'progress', step: 'init_model', message: 'Parsing safetensors header...' });
    const wasmMalloc = wasmInst!.exports.malloc as (size: number) => number;
    const wasmFree = wasmInst!.exports.free as (ptr: number) => void;

    const { tensors, dataOffset } = parseSafeTensorsHeader(weightsBuffer.buffer);
    post({ type: 'log', message: `[CPU] Parsed ${tensors.length} tensors, dataOffset=${dataOffset}` });

    const Qwen35Model = mlxExports.Qwen35Model || mlxExports.Qwen3_5Model;

    // Store config + tokenizer in Rust BEFORE tensor accumulation.
    // Must happen while WASM memory is still small to avoid emnapi DataView
    // bounds errors (the ~5.7 MB tokenizer JSON triggers memory writes that
    // fail if emnapi's DataView is stale after memory.grow).
    Qwen35Model.setCpuModelConfig(configJson, tokenizerJson, tokenizerConfigJson ?? null);

    post({ type: 'progress', step: 'init_model', message: `Loading ${tensors.length} tensors...` });
    for (let i = 0; i < tensors.length; i++) {
      const t = tensors[i];
      if (t.byteSize === 0) continue;

      // Malloc a small WASM buffer for this single tensor
      const ptr = wasmMalloc(t.byteSize);
      if (ptr === 0) {
        throw new Error(`Failed to malloc ${t.byteSize} bytes for tensor "${t.name}" (${i}/${tensors.length})`);
      }
      const uptr = ptr >>> 0;

      // Copy tensor bytes from download buffer into WASM memory
      // Re-create view each iteration (memory.grow may have changed bounds)
      const src = new Uint8Array(weightsBuffer.buffer, dataOffset + t.byteOffset, t.byteSize);
      new Uint8Array(sharedMemory.buffer).set(src, uptr);

      // Create MLX array in Rust (C++ copies from pointer) then free WASM buffer
      Qwen35Model.addCpuTensor(t.name, uptr, t.byteSize, t.shape, dtypeToCode(t.dtype));
      wasmFree(ptr);

      if (i % 50 === 0) {
        post({ type: 'progress', step: 'init_model', message: `Loaded ${i}/${tensors.length} tensors...` });
      }
    }
    post({ type: 'log', message: `[CPU] All ${tensors.length} tensors accumulated in Rust` });

    // Build model from accumulated tensors (config already stored pre-loop)
    post({ type: 'progress', step: 'init_model', message: 'Building model...' });
    const t1 = performance.now();
    model = await Qwen35Model.buildModelFromCpuTensors();
    post({ type: 'progress', step: 'init_model', message: `Model ready (${(performance.now() - t1).toFixed(0)}ms)` });

    // 6. Pipeline warmup — first inference warms GPU pipelines + shader compilation
    post({ type: 'progress', step: 'warmup', message: 'Warming up...' });
    await model.chat([{ role: 'user', content: 'hi' }], { maxNewTokens: 2, temperature: 0 });
    post({ type: 'progress', step: 'warmup', message: 'Warmup complete' });

    post({ type: 'ready' });
  } catch (e) {
    post({ type: 'error', message: String(e), stack: e instanceof Error ? e.stack : undefined });
  }
}

async function handleChat(data: { messages: any[]; config?: any }) {
  try {
    const chatConfig = {
      ...data.config,
      enableThinking: true,
      reportPerformance: true,
    };

    // Allocate a 256 KiB SAB ring. Keep sabRing alive for the duration of the
    // stream so the SAB is not GC'd before the WASM producer finishes writing.
    const sabRing = createSabRing(262_144);
    const { sab } = sabRing;

    // Convert the SharedArrayBuffer to a Buffer for the NAPI call.
    // The buffer polyfill supports SharedArrayBuffer at lines 134-137 of its
    // index.js — Buffer.from(sab) creates a Uint8Array view over the SAB.
    const sabBuf = Buffer.from(sab as unknown as ArrayBuffer);

    const t0 = performance.now();

    // Start the SAB ring reader before launching the stream so that no tokens
    // are missed. onChunk/onError mirror the existing chatStream callback body.
    let abortController: AbortController | null = null;

    abortController = sabRing.reader(
      (chunk) => {
        if (chunk.done) {
          const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
          post({ type: 'progress', step: 'chat', message: `chatStream completed in ${elapsed}s` });
          post({
            type: 'result',
            text: chunk.text,
            rawText: chunk.rawText,
            numTokens: chunk.numTokens,
            finishReason: chunk.finishReason,
            toolCalls: chunk.toolCalls,
            thinking: chunk.thinking,
            performance: chunk.performance
              ? {
                  ttftMs: chunk.performance.ttftMs,
                  prefillTokensPerSecond: chunk.performance.prefillTokensPerSecond,
                  decodeTokensPerSecond: chunk.performance.decodeTokensPerSecond,
                }
              : null,
          });
          // Stop the waitAsync loop now that the stream is done.
          abortController!.abort();
        } else {
          post({
            type: 'chunk',
            text: chunk.text,
            isReasoning: chunk.isReasoning ?? false,
          });
        }
      },
      (e) => {
        post({ type: 'error', message: e.message, stack: e.stack });
      },
    );

    // Use chatStreamSab — writes directly into the SAB ring, no TSFN on the
    // hot decode path. handle.cancel() + abortController.abort() tear down both
    // sides cleanly if cancellation is needed in the future.
    const handle = await model.chatStreamSab(data.messages, chatConfig, sabBuf);
    // handle.cancel() available for cancellation if needed
    void handle;
  } catch (e) {
    let message = String(e);
    let stack: string | undefined;
    if (e instanceof Error) {
      message = e.message;
      stack = e.stack;
    }
    post({ type: 'error', message, stack });
  }
}

// Catch unhandled errors/rejections on this worker
self.addEventListener('error', (e) => {
  post({ type: 'error', message: `Worker error: ${e.message}`, stack: (e as ErrorEvent).filename });
});
self.addEventListener('unhandledrejection', (e: PromiseRejectionEvent) => {
  post({ type: 'error', message: `Unhandled rejection: ${String(e.reason)}` });
});

self.onmessage = (e: MessageEvent) => {
  switch (e.data.type) {
    case 'init':
      handleInit(e.data);
      break;
    case 'chat':
      handleChat(e.data);
      break;
  }
};
