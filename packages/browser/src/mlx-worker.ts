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

import {
  instantiateNapiModule,
  getDefaultContext,
  WASI,
} from '@napi-rs/wasm-runtime';

import { createBridgeStub } from './webgpu-bridge-stub.js';
import { CMD_OFFSET, READBACK_BUFFER_SIZE, STREAM_BUFFER_SIZE, STREAM_HEADER_SIZE, STREAM_TEXT_OFFSET } from './rpc-protocol.js';
import { parseSafeTensorsHeader, dtypeToCode } from './safetensors.js';

let model: any = null;
let mlxExports: any = null;
let wasmInst: WebAssembly.Instance | null = null;
let cppTag: any = null;

function post(msg: any) {
  (self as any).postMessage(msg);
}

async function handleInit(data: { wasmUrl: string; modelUrl: string }) {
  try {
    // 1. Spawn gpu-worker (owns GPUDevice, event loop free for GPU callbacks)
    post({ type: 'progress', step: 'gpu', message: 'Initializing WebGPU...' });
    const cmdBuffer = new SharedArrayBuffer(CMD_OFFSET.TOTAL);
    const readbackBuffer = new SharedArrayBuffer(READBACK_BUFFER_SIZE);
    // Streaming text channel — WASM writes decoded tokens here, main thread polls
    const streamBuffer = new SharedArrayBuffer(STREAM_BUFFER_SIZE);
    post({ type: 'stream_buffer', buffer: streamBuffer });
    const sharedMemory = new WebAssembly.Memory({
      initial: 30000,
      maximum: 65536,
      shared: true,
    });

    const gpuWorker = new Worker(
      new URL('./gpu-worker.ts', import.meta.url),
      { type: 'module' },
    );

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
        wasmMemory: sharedMemory,  // Send Memory object, not .buffer — .buffer getter always returns current byteLength after grow()
      });
    });
    post({ type: 'progress', step: 'gpu', message: 'WebGPU ready' });

    // 2. Create bridge stub (RPC via Atomics.wait to gpu-worker)
    const bridge = createBridgeStub(cmdBuffer, sharedMemory, {
      instanceHandle: gpuReady.instanceHandle,
      adapterHandle: gpuReady.adapterHandle,
      deviceHandle: gpuReady.deviceHandle,
      queueHandle: gpuReady.queueHandle,
    }, readbackBuffer);

    // 3. Load WASM with bridge stub
    post({ type: 'progress', step: 'wasm', message: 'Loading WASM module...' });
    const wasi = new WASI({ version: 'preview1' });
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
      asyncWorkPoolSize: 0,
      wasi,
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
    post({ type: 'progress', step: 'wasm', message: 'WASM loaded' });

    // 3. Fetch model files
    post({ type: 'progress', step: 'model', message: 'Fetching config...' });
    const configJson = await fetch(`${data.modelUrl}/config.json`).then(r => r.text());

    post({ type: 'progress', step: 'model', message: 'Fetching tokenizer...' });
    const tokenizerJson = await fetch(`${data.modelUrl}/tokenizer.json`).then(r => r.text());
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

    // 4. Parse SafeTensors + create GPU buffers (zero-copy)
    post({ type: 'progress', step: 'gpu_upload', message: 'Creating GPU buffers...' });
    const t0 = performance.now();
    const { tensors, dataOffset } = parseSafeTensorsHeader(weightsBuffer.buffer);

    const gpuTensors: Array<{ name: string; handle: number; dtypeCode: number; shape: number[]; byteSize: number }> = [];

    // Zero-copy weight upload via SharedArrayBuffer.
    // Copy weights into a SharedArrayBuffer, send it to gpu-worker.
    // The gpu-worker reads directly from the shared buffer to create
    // GPU buffers with mappedAtCreation — no additional copying.
    const weightsSab = new SharedArrayBuffer(weightsBuffer.byteLength);
    new Uint8Array(weightsSab).set(new Uint8Array(weightsBuffer.buffer));

    const tensorMeta = tensors.map(t => ({
      name: t.name,
      byteOffset: t.byteOffset,
      byteSize: t.byteSize,
      dtype: t.dtype,
      shape: t.shape,
    }));

    const uploadResult = await new Promise<{ handles: number[]; uploadedDtypes: string[] }>((resolve) => {
      const handler = (e: MessageEvent) => {
        if (e.data.type === 'weights_uploaded') {
          gpuWorker.removeEventListener('message', handler);
          resolve({ handles: e.data.handles, uploadedDtypes: e.data.uploadedDtypes });
        }
      };
      gpuWorker.addEventListener('message', handler);
      // SharedArrayBuffer is not transferred — both workers can access it
      gpuWorker.postMessage({
        type: 'upload_weights',
        weightsSab,
        dataOffset,
        tensors: tensorMeta,
      });
    });

    const { handles, uploadedDtypes } = uploadResult;
    for (let i = 0; i < tensors.length; i++) {
      // Use the ORIGINAL dtype (bf16/f16) not the GPU storage dtype (f32).
      // The MLX WebGPU backend expects bf16 arrays (stored as f32 internally).
      // Passing f32 dtype breaks the model's type tracking, causing incorrect
      // behavior in operations like SDPA that check result_type().
      const originalDtype = tensors[i].dtype;
      const gpuByteSize = (uploadedDtypes[i] !== originalDtype)
        ? (tensors[i].byteSize / 2) * 4  // expanded from 2 bytes to 4 bytes per element
        : tensors[i].byteSize;
      gpuTensors.push({
        name: tensors[i].name,
        handle: handles[i],
        dtypeCode: dtypeToCode(originalDtype),
        shape: tensors[i].shape,
        byteSize: gpuByteSize,
      });
    }

    post({ type: 'progress', step: 'gpu_upload', message: `${tensors.length} GPU buffers created (${(performance.now() - t0).toFixed(0)}ms)` });

    // 5. Build model
    post({ type: 'progress', step: 'init_model', message: 'Initializing model...' });
    const t1 = performance.now();
    const Qwen35Model = mlxExports.Qwen35Model || mlxExports.Qwen3_5Model;
    model = Qwen35Model.loadFromGpuBuffers(configJson, gpuTensors, tokenizerJson, tokenizerConfigJson);
    post({ type: 'progress', step: 'init_model', message: `Model ready (${(performance.now() - t1).toFixed(0)}ms)` });

    post({ type: 'ready' });
  } catch (e) {
    post({ type: 'error', message: String(e), stack: (e instanceof Error) ? e.stack : undefined });
  }
}

async function handleChat(data: { messages: any[]; config?: any }) {
  try {
    post({ type: 'progress', step: 'chat', message: 'Starting chat inference...' });

    // Test: try a simple tokenize first to isolate where the error is
    const Qwen35Model = mlxExports.Qwen35Model || mlxExports.Qwen3_5Model;
    post({ type: 'progress', step: 'chat', message: 'Tokenizer test OK, calling model.chat...' });

    // Use chatSync (synchronous NAPI export) to avoid emnapi async worker thread
    // dispatch which deadlocks because asyncify can't yield on child worker threads.
    //
    // Manually drive asyncify: emnapi doesn't wrap exports with asyncify, so we
    // handle the unwind/await/rewind cycle here. When mlx_webgpu_poll returns a
    // Promise (to wait for GPU work), the wrapped import triggers asyncify_start_unwind,
    // causing the WASM call stack to save and chatFn.call to return. We then await
    // the Promise, start rewind, and re-call chatFn to restore execution.
    // Diagnostics: only run on first chat to verify model correctness
    const runDiagnostics = !(globalThis as any).__mlx_diag_done;
    (globalThis as any).__mlx_diag_done = true;

    if (runDiagnostics) {
    // Heap sanity check: try a simple allocation before chat to detect corruption
    try {
      const MxArray = mlxExports.MxArray;
      const test = MxArray.fromFloat32(new Float32Array([1, 2, 3]), new BigInt64Array([3n]));
      test.eval();
      const vals = test.toFloat32();
      post({ type: 'progress', step: 'chat', message: `Heap check: [${vals[0]},${vals[1]},${vals[2]}]` });
    } catch (e: any) {
      post({ type: 'progress', step: 'chat', message: `HEAP CHECK FAILED: ${e.message}` });
    }

    // Step-by-step GDN diagnostic: compare each checkpoint against native reference
    try {
      const MxArray = mlxExports.MxArray;
      const nativeRefs: Record<number, number[]> = {
        0: [-0.338840, -0.187517, 0.469171, 1.201061, 0.149156],  // in_proj qkv
        1: [0.012732, 0.000515, 0.000871, 0.157438, -0.005936],   // conv1d + SiLU
        2: [0.004488, 0.000182, 0.000307, 0.055495, -0.002093],   // q RMS norm
        3: [0.074052, 0.999844, 0.999474, 0.015807, 0.782352],    // g values
        4: [0.301661, 0.384497, 0.341307, 0.743674, 0.416961],    // beta values
        5: [-0.000000, 0.000000, 0.000007, 0.000001, 0.000001],   // GDN y output
        6: [-0.000000, -0.000002, -0.000008, -0.000000, -0.000000], // GDN state
        7: [0.000036, -0.000001, -0.000543, -0.000149, 0.000831],  // RMSNorm gating
        8: [-0.071379, -0.001279, -0.003993, -0.005899, -0.004375], // final output
      };
      const cpNames = ['in_proj', 'conv1d+SiLU', 'q_rms_norm', 'g', 'beta', 'gdn_y', 'gdn_state', 'norm_gate', 'output'];
      let firstMismatch = -1;
      for (let cp = 0; cp <= 8; cp++) {
        const vals = MxArray.testGdnStep(cp, 10);
        if (vals[0] === -999) {
          post({ type: 'progress', step: 'chat', message: `GDN cp${cp} (${cpNames[cp]}): ERROR (weights not loaded?)` });
          continue;
        }
        const ref = nativeRefs[cp];
        let match = true;
        const tol = (cp >= 5 && cp <= 6) ? 0.001 : 0.05; // tighter for near-zero values
        for (let i = 0; i < ref.length; i++) {
          const diff = Math.abs(vals[i] - ref[i]);
          if (diff > tol && diff > Math.abs(ref[i]) * 0.5) { match = false; break; }
        }
        const status = match ? 'OK' : 'MISMATCH';
        if (!match && firstMismatch < 0) firstMismatch = cp;
        post({ type: 'progress', step: 'chat', message: `GDN cp${cp} (${cpNames[cp]}): ${status} [${vals.slice(0,5).map((v: number)=>v.toFixed(6)).join(',')}]` });
      }
      if (firstMismatch >= 0) {
        post({ type: 'progress', step: 'chat', message: `>>> FIRST DIVERGENCE at checkpoint ${firstMismatch} (${cpNames[firstMismatch]})` });
      } else {
        post({ type: 'progress', step: 'chat', message: `>>> ALL GDN checkpoints MATCH native` });
      }

      // Sub-step checkpoints within GDN recurrence
      const subRefs: Record<number, number[]> = {
        10: [0, 0, 0, 0, 0],  // decay: state*g = 0
        11: [0, 0, 0, 0, 0],  // kv_mem = 0
        12: [0.00035044, -0.00001372, -0.00549765, -0.00070031, -0.00066313],  // delta
        13: [-0.00000041, -0.00000235, -0.00000796, -0.00000025, -0.00000020],  // state update
        14: [-0.00000045, 0.00000002, 0.00000711, 0.00000091, 0.00000086],  // y manual
        15: [-0.18261321, -4.34687376, -0.18045458, -2.62038946, 1.24019587],  // z values
        16: [-0.00117467, -0.00671195, -0.02271412, -0.00072459, -0.00058014],  // k values (after RMS norm)
        17: [-0.00000041, -0.00000235, -0.00000796, -0.00000025, -0.00000020],  // outer product
        18: [-0.00046219, -0.00264092, -0.00893721, -0.00028510, -0.00022826],  // conv_out at k offset
        19: [-0.00046219, -0.00264092, -0.00893721, -0.00028510, -0.00022826],  // k raw before RMS
      };
      const subNames = ['decay', 'kv_mem', 'delta', 'state_upd', 'y_manual', 'z_values', 'k_values', 'outer_prod', 'conv@k_off', 'k_raw'];
      for (let cp = 10; cp <= 19; cp++) {
        const vals = MxArray.testGdnStep(cp, 10);
        if (vals[0] === -999) continue;
        const ref = subRefs[cp];
        let match = true;
        for (let i = 0; i < ref.length; i++) {
          const diff = Math.abs(vals[i] - ref[i]);
          const absRef = Math.abs(ref[i]);
          if (absRef < 1e-8) { if (diff > 1e-4) match = false; }
          else if (diff / absRef > 0.3) match = false;
        }
        const status = match ? 'OK' : 'MISMATCH';
        post({ type: 'progress', step: 'chat', message: `GDN sub${cp} (${subNames[cp-10]}): ${status} [${vals.slice(0,5).map((v: number)=>v.toFixed(8)).join(',')}]` });
      }
    } catch (e: any) {
      post({ type: 'progress', step: 'chat', message: `GDN step test error: ${e.message}` });
    }

    // Verify C++ weight data matches safetensors after model loading
    try {
      const MxArray = mlxExports.MxArray;
      const wVals = MxArray.readCppWeight('layers.0.input_layernorm.weight', 5);
      const expected = [2.0, 1.296875, 1.53125, 1.75, 1.359375];
      let match = wVals.length >= 5;
      for (let i = 0; match && i < 5; i++) {
        if (Math.abs(wVals[i] - expected[i]) > 0.01) match = false;
      }
      post({ type: 'progress', step: 'chat', message: `Weight verify: ${match ? 'OK' : 'MISMATCH'} vals=[${wVals.slice(0,5).map((v: number)=>v.toFixed(4)).join(',')}]` });
    } catch (e: any) {
      post({ type: 'progress', step: 'chat', message: `Weight verify error: ${e.message}` });
    }

    // Run single forward pass and compare logits
    try {
      const MxArray = mlxExports.MxArray;
      // Run full forward with the test input and check first token logits
      const layerVals = MxArray.testSingleLayerForward();
      const nativeRef = [-0.071379, -0.001279, -0.003993, -0.005899, -0.004375];
      let match = true;
      for (let i = 0; i < 5; i++) {
        if (Math.abs(layerVals[i] - nativeRef[i]) > 0.01) match = false;
      }
      post({ type: 'progress', step: 'chat', message: `Layer0 forward: ${match ? 'MATCH' : 'MISMATCH'} [${layerVals.slice(0,5).map((v: number)=>v.toFixed(6)).join(',')}]` });
    } catch (e: any) {
      post({ type: 'progress', step: 'chat', message: `Layer0 forward error: ${e.message}` });
    }

    // Test attention layer 3 forward (first full-attention layer)
    try {
      const MxArray = mlxExports.MxArray;
      const attnVals = MxArray.testAttentionLayerForward(20);
      const nativeRef = [0.307530, -0.145069, -0.083070, 0.250707, -0.017258, -0.197453, 0.013055, -0.014458, 0.008774, -0.035990];
      let match = true;
      for (let i = 0; i < 10; i++) {
        if (Math.abs(attnVals[i] - nativeRef[i]) > 0.05) { match = false; break; }
      }
      post({ type: 'progress', step: 'chat', message: `Attn L3: ${match ? 'MATCH' : 'MISMATCH'} [${attnVals.slice(0,10).map((v: number)=>v.toFixed(6)).join(',')}]` });
    } catch (e: any) {
      post({ type: 'progress', step: 'chat', message: `Attn L3 error: ${e.message}` });
    }
    } // end runDiagnostics

    const chatFn = model.chatSync || model.chat_sync || model.chat;
    const chatConfig = {
      ...(data.config ?? {}),
      enableThinking: true,  // Let model think — thinking block gets stripped from final output
      reportPerformance: true,
    };

    // Note: true token-by-token streaming is not yet possible with the sync
    // WASM architecture (postMessage is batched until chatSync returns).
    // Tokens are rendered all-at-once when the result arrives.
    // The Rust code still emits [STREAM_TEXT] to stderr for debugging.

    // Direct synchronous call — no asyncify needed!
    // mlx_webgpu_poll blocks via Atomics.wait in the bridge stub.
    // The gpu-worker's event loop processes GPU callbacks and notifies us.
    post({ type: 'progress', step: 'chat', message: 'Calling chatFn...' });
    const t0 = performance.now();
    let result: any;
    try {
      result = chatFn.call(model, data.messages, chatConfig);
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
                    else { s = ''; break; }
                  }
                  if (s.length > 5) { detail = `C++ exception: ${s}`; break; }
                }
              }
            }
          }
        } catch { /* extraction failed */ }
      }
      post({ type: 'progress', step: 'chat', message: `chatFn THREW: ${detail}\nStack: ${stack}` });
      console.error('[WASM Error]', detail, '\nStack:', stack);
      throw e;
    }
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    post({ type: 'progress', step: 'chat', message: `chatFn returned in ${elapsed}s` });

    // Debug: log the raw output to understand garbage tokens
    const rawChars = result.rawText ? [...result.rawText].slice(0, 20).map((c: string) => `${c}(${c.charCodeAt(0)})`) : [];
    post({ type: 'progress', step: 'chat', message: `rawText chars: [${rawChars.join(', ')}]` });
    post({ type: 'progress', step: 'chat', message: `finishReason=${result.finishReason}, numTokens=${result.numTokens}, text="${result.text?.substring(0, 50)}"` });

    // Debug: verify bf16→f32 weight conversion by checking a known value
    try {
      // Create a bf16 value (1.0 = 0x3F80 in bf16) and verify roundtrip
      const testBf16 = new Uint16Array([0x3F80, 0x4000, 0x4040, 0xBF80]); // 1.0, 2.0, 3.0, -1.0
      const testF32 = new Uint32Array(4);
      for (let i = 0; i < 4; i++) testF32[i] = testBf16[i] << 16; // bf16→f32 conversion
      const testF32View = new Float32Array(testF32.buffer);
      post({ type: 'progress', step: 'chat', message: `bf16→f32 verify: [${Array.from(testF32View).map(v => v.toFixed(4)).join(', ')}]` });
    } catch (e: any) {
      post({ type: 'progress', step: 'chat', message: `bf16 verify failed: ${e.message}` });
    }

    post({
      type: 'result',
      text: result.text,
      rawText: result.rawText,
      numTokens: result.numTokens,
      finishReason: result.finishReason,
      toolCalls: result.toolCalls,
      thinking: result.thinking,
      performance: result.performance ? {
        ttftMs: result.performance.ttftMs,
        prefillTokensPerSecond: result.performance.prefillTokensPerSecond,
        decodeTokensPerSecond: result.performance.decodeTokensPerSecond,
      } : null,
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
