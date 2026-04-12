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

import {
  CMD_OFFSET,
  READBACK_BUFFER_SIZE,
  STREAM_BUFFER_SIZE,
  STREAM_HEADER_SIZE,
  STREAM_TEXT_OFFSET,
} from './rpc-protocol.js';
import { parseSafeTensorsHeader, dtypeToCode } from './safetensors.js';
import { createBridgeStub } from './webgpu-bridge-stub.js';

let model: any = null;
let mlxExports: any = null;
let wasmInst: WebAssembly.Instance | null = null;
let cppTag: any = null;

function post(msg: any) {
  (self as any).postMessage(msg);
}

async function handleInit(data: {
  wasmUrl: string;
  modelUrl: string;
  packBf16?: boolean;
  sdpaFallback?: boolean;
}) {
  const packBf16 = data.packBf16 === true;
  const sdpaFallback = data.sdpaFallback === true;
  try {
    // 1. Spawn gpu-worker (owns GPUDevice, event loop free for GPU callbacks)
    post({ type: 'progress', step: 'gpu', message: 'Initializing WebGPU...' });
    const cmdBuffer = new SharedArrayBuffer(CMD_OFFSET.TOTAL);
    const readbackBuffer = new SharedArrayBuffer(READBACK_BUFFER_SIZE);
    // Streaming text channel — WASM writes decoded tokens here, main thread polls
    const streamBuffer = new SharedArrayBuffer(STREAM_BUFFER_SIZE);
    post({ type: 'stream_buffer', buffer: streamBuffer });
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
        if (line.includes('[MLX-KERNEL]') || line.includes('[CPU-BUILD]')) {
          post({ type: 'log', message: line });
        } else {
          console.log(...args);
        }
      },
      printErr: function (...args: unknown[]) {
        const line = args.map(String).join(' ');
        if (line.includes('[MLX-KERNEL]') || line.includes('[CPU-BUILD]')) {
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

    // Stream channel: WASM calls mlx_stream_write(ptr, len) to push decoded text
    const streamView = new Uint8Array(streamBuffer);
    const streamI32 = new Int32Array(streamBuffer);
    const cxxStubs = {
      __cpp_exception: cppExceptionTag,
      _ZN3mlx4core3gpu4initEv: () => {},
      // Called from Rust with a pointer into WASM memory and byte length
      mlx_stream_write: (ptr: number, len: number) => {
        const maxLen = STREAM_BUFFER_SIZE - STREAM_TEXT_OFFSET;
        if (len > maxLen) len = maxLen; // Truncate to prevent overflow
        const wasmMem = new Uint8Array(sharedMemory.buffer);
        const text = wasmMem.slice(ptr, ptr + len);
        const writePos = STREAM_TEXT_OFFSET;
        streamView.set(text, writePos);
        // Update header: [0]=byte length, [1]=sequence counter
        Atomics.store(streamI32, 0, len);
        Atomics.add(streamI32, 1, 1);
        // Wake main thread's Atomics.waitAsync
        Atomics.notify(streamI32, 1);
      },
      // Reset stream buffer (called before chat starts)
      mlx_stream_reset: () => {
        Atomics.store(streamI32, 0, 0);
        Atomics.store(streamI32, 1, 0);
      },
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
          streamBuffer,
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
    const flagSuffix =
      (packBf16 ? ' pack_bf16=1' : '') +
      (sdpaFallback ? ' sdpa_fallback=1' : '');
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
      Qwen35Model.addCpuTensor(
        t.name,
        uptr,
        t.byteSize,
        t.shape,
        dtypeToCode(t.dtype),
      );
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
    await model.chat(
      [{ role: 'user', content: 'hi' }],
      { maxNewTokens: 2, temperature: 0 },
    );
    post({ type: 'progress', step: 'warmup', message: 'Warmup complete' });

    post({ type: 'ready' });
  } catch (e) {
    post({ type: 'error', message: String(e), stack: e instanceof Error ? e.stack : undefined });
  }
}

async function handleChat(data: { messages: any[]; config?: any }) {
  try {
    const chatConfig = {
      ...(data.config ?? {}),
      enableThinking: true, // Let model think — thinking block gets stripped from final output
      reportPerformance: true,
    };

    // Async chat — tokio_unstable + asyncWorkPoolSize > 0 enables async NAPI
    // on WASM. The Rust model thread handles inference; the JS side awaits.
    post({ type: 'progress', step: 'chat', message: 'Calling model.chat()...' });
    const t0 = performance.now();
    let result: any;
    try {
      result = await model.chat(data.messages, chatConfig);
    } catch (e: any) {
      // Get WASM stack trace and try to extract C++ exception message
      const stack = e.stack || '';
      let detail = `${e.constructor?.name}: ${e.message}`;
      if (e instanceof WebAssembly.Exception && wasmInst) {
        try {
          const ptr = e.getArg(cppTag, 0);
          if (typeof ptr === 'number') {
            const mem = new Uint8Array((wasmInst.exports.memory as WebAssembly.Memory).buffer);
            const view = new DataView(mem.buffer);
            // Scan exception object fields to find readable error message
            for (const off of [4, 8, 12, 16, 20]) {
              if (ptr + off + 4 < mem.length) {
                const strPtr = view.getUint32(ptr + off, true);
                if (strPtr > 1024 && strPtr < mem.length - 4) {
                  let s = '';
                  for (let i = strPtr; i < mem.length && mem[i] !== 0 && i - strPtr < 500; i++) {
                    const ch = mem[i];
                    if (ch >= 32 && ch < 127) s += String.fromCharCode(ch);
                    else {
                      s = '';
                      break;
                    }
                  }
                  if (s.length > 5) {
                    detail = `C++ exception: ${s}`;
                    break;
                  }
                }
              }
            }
          }
        } catch {
          /* extraction failed */
        }
      }
      post({ type: 'progress', step: 'chat', message: `chatFn THREW: ${detail}\nStack: ${stack}` });
      console.error('[WASM Error]', detail, '\nStack:', stack);
      throw e;
    }
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    post({ type: 'progress', step: 'chat', message: `chatFn returned in ${elapsed}s` });

    post({
      type: 'result',
      text: result.text,
      rawText: result.rawText,
      numTokens: result.numTokens,
      finishReason: result.finishReason,
      toolCalls: result.toolCalls,
      thinking: result.thinking,
      performance: result.performance
        ? {
            ttftMs: result.performance.ttftMs,
            prefillTokensPerSecond: result.performance.prefillTokensPerSecond,
            decodeTokensPerSecond: result.performance.decodeTokensPerSecond,
          }
        : null,
    });
  } catch (e) {
    let message = String(e);
    let stack: string | undefined;
    if (e instanceof Error) {
      message = e.message;
      stack = e.stack;
    } else if (e instanceof WebAssembly.Exception) {
      // Try to extract C++ exception message via getArg + WASM memory
      let cppMsg = '';
      try {
        const cppExceptionTag = new WebAssembly.Tag({ parameters: ['i32'] });
        if ((e as any).is?.(cppExceptionTag)) {
          const exnPtr = (e as any).getArg(cppExceptionTag, 0) as number;
          // The exception ptr points to the thrown object. For std::runtime_error,
          // the what() string is at a known offset. Try to read via __cxa_begin_catch.
          if (wasmInst?.exports?.__cxa_begin_catch && wasmInst?.exports?.__cxa_end_catch) {
            const beginCatch = wasmInst.exports.__cxa_begin_catch as Function;
            const endCatch = wasmInst.exports.__cxa_end_catch as Function;
            const objPtr = beginCatch(exnPtr);
            // Read the vtable to find what() — offset 0 is vtable ptr, what() is at vtable[2]
            // For now, just try to read a string at a common offset
            try {
              const mem = new Uint8Array((wasmInst.exports.memory as WebAssembly.Memory).buffer);
              // std::exception vtable: [0]=typeinfo, [4]=destructor, [8]=what()
              // what() returns a const char*. Call it via indirect call.
              const view = new DataView(mem.buffer);
              const vtablePtr = view.getUint32(objPtr, true);
              const whatFnPtr = view.getUint32(vtablePtr + 8, true);
              // Call what() via wasm table
              const table = wasmInst.exports.__indirect_function_table as WebAssembly.Table;
              if (table && whatFnPtr > 0) {
                const whatFn = table.get(whatFnPtr) as Function;
                const strPtr = whatFn(objPtr) as number;
                // Read null-terminated string from WASM memory
                let str = '';
                for (let i = strPtr; i < mem.length && mem[i] !== 0; i++) {
                  str += String.fromCharCode(mem[i]);
                }
                if (str) cppMsg = str;
              }
            } catch (readErr) {
              cppMsg = `(could not read what(): ${readErr})`;
            }
            endCatch();
          }
        }
      } catch (tagErr) {
        // getArg failed — different tag or format
      }
      message = cppMsg
        ? `C++ exception: ${cppMsg}`
        : `WebAssembly.Exception (C++ exception escaped to JS — likely an unimplemented op or runtime error in MLX)`;
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
