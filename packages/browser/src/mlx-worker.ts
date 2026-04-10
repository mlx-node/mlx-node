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

async function handleInit(data: { wasmUrl: string; modelUrl: string; packBf16?: boolean }) {
  const packBf16 = data.packBf16 === true;
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
    // Flip the backend's packed-bf16 flag BEFORE any model/weight work runs.
    // wgpu_buffer() checks this flag on first upload to decide whether an
    // eligible bf16 weight gets flipped into StorageMode::PackedBf16. Toggling
    // after the fact is a no-op for already-uploaded buffers.
    if (typeof mlxExports.wgpuSetPackedBf16Enabled === 'function') {
      mlxExports.wgpuSetPackedBf16Enabled(packBf16);
    }
    post({ type: 'progress', step: 'wasm', message: `WASM loaded${packBf16 ? ' (pack_bf16=1)' : ''}` });

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

    // 4. Parse SafeTensors + create GPU buffers (zero-copy)
    post({ type: 'progress', step: 'gpu_upload', message: 'Creating GPU buffers...' });
    const t0 = performance.now();
    const parsed = parseSafeTensorsHeader(weightsBuffer.buffer);
    let { tensors } = parsed;
    const { dataOffset } = parsed;

    const gpuTensors: Array<{
      name: string;
      handle: number;
      dtypeCode: number;
      shape: number[];
      byteSize: number;
      packedBf16?: boolean;
    }> = [];

    // Zero-copy weight upload via SharedArrayBuffer.
    // Copy weights into a SharedArrayBuffer, send it to gpu-worker.
    // The gpu-worker reads directly from the shared buffer to create
    // GPU buffers with mappedAtCreation — no additional copying.
    const weightsSab = new SharedArrayBuffer(weightsBuffer.byteLength);
    new Uint8Array(weightsSab).set(new Uint8Array(weightsBuffer.buffer));

    // ------------------------------------------------------------------
    // Step B: pre-merge linear_attn.in_proj_qkv + in_proj_z → in_proj_qkvz
    //          and        linear_attn.in_proj_b   + in_proj_a → in_proj_ba
    //
    // Both projections in Qwen3.5 store their operands row-contiguous
    // ([rows_src, hidden]) with a constant dtype. axis-0 concat is therefore
    // a straight byte append: bytes(src0) ++ bytes(src1). Doing this in JS
    // BEFORE upload means the merged tensor enters the gpu-worker upload
    // loop as a single buffer with its OWN PackedBf16 opt-in, instead of
    // going through the Rust fallback in persistence.rs::merge_split_projections
    // that produces a fresh buffer with no storage_mode propagation.
    //
    // The Rust merge code is a fallback: it scans for `.in_proj_qkv.weight`
    // and `.in_proj_b.weight` keys and only runs when present. Once we
    // remove the split `.weight` entries here, it becomes a no-op. If we
    // encounter a quantized checkpoint (any `.scales` or `.biases` sibling
    // to the split .weight) we SKIP the JS pre-merge for that prefix so
    // the Rust code still handles the paired quantization metadata.
    // ------------------------------------------------------------------
    const tensorByName = new Map(tensors.map((t) => [t.name, t]));
    const mergeSpecs = [
      { outSuffix: '.linear_attn.in_proj_qkvz.weight', aSuffix: '.linear_attn.in_proj_qkv.weight', bSuffix: '.linear_attn.in_proj_z.weight' },
      { outSuffix: '.linear_attn.in_proj_ba.weight',   aSuffix: '.linear_attn.in_proj_b.weight',   bSuffix: '.linear_attn.in_proj_a.weight' },
    ];
    type MergePlan = { outName: string; aTensor: typeof tensors[number]; bTensor: typeof tensors[number]; outShape: number[]; outByteSize: number };
    const mergePlans: MergePlan[] = [];
    const removedNames = new Set<string>();
    for (const tensor of tensors) {
      for (const spec of mergeSpecs) {
        if (!tensor.name.endsWith(spec.aSuffix)) continue;
        const prefix = tensor.name.slice(0, -spec.aSuffix.length);
        const bName = `${prefix}${spec.bSuffix}`;
        const bTensor = tensorByName.get(bName);
        if (!bTensor) break;
        // Skip if either side has a quant scale/bias sibling; leave those
        // prefixes for the Rust merge code to handle the full triple.
        const hasQuantSibling = (name: string) => {
          const base = name.replace(/\.weight$/, '');
          return tensorByName.has(`${base}.scales`) || tensorByName.has(`${base}.biases`);
        };
        if (hasQuantSibling(tensor.name) || hasQuantSibling(bName)) break;
        // Both must be the same dtype; shapes must be 2-D row-contiguous
        // with matching trailing dimension (hidden_size). Safetensors
        // tensors are always dense row-major, so this holds for bf16/f32.
        if (tensor.dtype !== bTensor.dtype) break;
        if (tensor.shape.length !== 2 || bTensor.shape.length !== 2) break;
        if (tensor.shape[1] !== bTensor.shape[1]) break;
        const outShape = [tensor.shape[0] + bTensor.shape[0], tensor.shape[1]];
        mergePlans.push({
          outName: `${prefix}${spec.outSuffix}`,
          aTensor: tensor,
          bTensor,
          outShape,
          outByteSize: tensor.byteSize + bTensor.byteSize,
        });
        removedNames.add(tensor.name);
        removedNames.add(bName);
        break;
      }
    }

    // Build a merged-bytes SAB that the gpu-worker will read from for
    // synthetic (pre-merged) tensors. Byte layout: plans[0] bytes ||
    // plans[1] bytes || ... . Offsets are recorded in the synthetic tensor
    // meta so gpu-worker can slice without any extra copies.
    let mergedSab: SharedArrayBuffer | undefined;
    const syntheticTensors: Array<{ name: string; dtype: string; shape: number[]; byteOffset: number; byteSize: number; fromMerged: true }> = [];
    if (mergePlans.length > 0) {
      const totalBytes = mergePlans.reduce((s, p) => s + p.outByteSize, 0);
      mergedSab = new SharedArrayBuffer(totalBytes);
      const mergedU8 = new Uint8Array(mergedSab);
      const srcU8 = new Uint8Array(weightsSab);
      let cursor = 0;
      for (const plan of mergePlans) {
        // axis-0 concat = bytes(a) ++ bytes(b)
        mergedU8.set(
          srcU8.subarray(dataOffset + plan.aTensor.byteOffset, dataOffset + plan.aTensor.byteOffset + plan.aTensor.byteSize),
          cursor,
        );
        cursor += plan.aTensor.byteSize;
        mergedU8.set(
          srcU8.subarray(dataOffset + plan.bTensor.byteOffset, dataOffset + plan.bTensor.byteOffset + plan.bTensor.byteSize),
          cursor,
        );
        cursor += plan.bTensor.byteSize;
        syntheticTensors.push({
          name: plan.outName,
          dtype: plan.aTensor.dtype,
          shape: plan.outShape,
          byteOffset: cursor - plan.outByteSize,
          byteSize: plan.outByteSize,
          fromMerged: true,
        });
      }
    }

    // Drop the split source tensors; append synthetic merged tensors.
    tensors = tensors.filter((t) => !removedNames.has(t.name));
    post({
      type: 'log',
      message: `[PACKBF16] pre-merged ${syntheticTensors.length} linear_attn split projections (removed ${removedNames.size} source tensors)`,
    });

    const tensorMeta: Array<{ name: string; byteOffset: number; byteSize: number; dtype: string; shape: number[]; fromMerged?: boolean }> = [
      ...tensors.map((t) => ({
        name: t.name,
        byteOffset: t.byteOffset,
        byteSize: t.byteSize,
        dtype: t.dtype,
        shape: t.shape,
      })),
      ...syntheticTensors.map((t) => ({
        name: t.name,
        byteOffset: t.byteOffset,
        byteSize: t.byteSize,
        dtype: t.dtype,
        shape: t.shape,
        fromMerged: true as const,
      })),
    ];
    // Keep tensors list in sync with tensorMeta so the post-upload loop
    // that builds gpuTensors iterates over the same entries we uploaded.
    const tensorsForGpu: Array<{ name: string; byteSize: number; dtype: string; shape: number[] }> = tensorMeta.map((t) => ({
      name: t.name,
      byteSize: t.byteSize,
      dtype: t.dtype,
      shape: t.shape,
    }));

    const uploadResult = await new Promise<{
      handles: number[];
      uploadedDtypes: string[];
      packedBf16Flags?: boolean[];
    }>((resolve) => {
      const handler = (e: MessageEvent) => {
        if (e.data.type === 'weights_uploaded') {
          gpuWorker.removeEventListener('message', handler);
          // Forward debug info from the GPU worker to the main page console
          // (worker console.log doesn't always reach the page DevTools tab).
          const dbgPacked = e.data.debugPackedNames ?? [];
          const dbgUnpacked = e.data.debugUnpackedBf16Names ?? [];
          const dbgPackedTotal = e.data.debugPackedTotal ?? 0;
          const dbgUnpackedTotal = e.data.debugUnpackedBf16Total ?? 0;
          (self as any).postMessage({
            type: 'log',
            message: `[PACKBF16] packed=${dbgPackedTotal} unpacked-large-bf16=${dbgUnpackedTotal}`,
          });
          // Print one tensor name per log line so the full lists fit in the
          // message buffer instead of being truncated.
          for (const n of dbgPacked) {
            (self as any).postMessage({ type: 'log', message: `[PACKBF16] PACK ${n}` });
          }
          for (const n of dbgUnpacked) {
            (self as any).postMessage({ type: 'log', message: `[PACKBF16] SKIP ${n}` });
          }
          resolve({
            handles: e.data.handles,
            uploadedDtypes: e.data.uploadedDtypes,
            packedBf16Flags: e.data.packedBf16Flags,
          });
        }
      };
      gpuWorker.addEventListener('message', handler);
      // SharedArrayBuffer is not transferred — both workers can access it
      gpuWorker.postMessage({
        type: 'upload_weights',
        weightsSab,
        mergedSab,
        dataOffset,
        tensors: tensorMeta,
        packBf16,
      });
    });

    const { handles, uploadedDtypes, packedBf16Flags } = uploadResult;
    for (let i = 0; i < tensorsForGpu.length; i++) {
      // Use the ORIGINAL dtype (bf16/f16) not the GPU storage dtype (f32).
      // The MLX WebGPU backend expects bf16 arrays (stored as f32 internally).
      // Passing f32 dtype breaks the model's type tracking, causing incorrect
      // behavior in operations like SDPA that check result_type().
      const originalDtype = tensorsForGpu[i].dtype;
      const isPacked = packedBf16Flags?.[i] === true;
      // GPU byte size:
      //   - packed bf16: 2 bytes/elem, padded to u32 (same as source bf16 size, rounded up)
      //   - expanded (bf16→f32 or f16→f32): 4 bytes/elem
      //   - native: original byte size
      let gpuByteSize: number;
      if (isPacked) {
        const numElements = tensorsForGpu[i].shape.reduce((a, b) => a * b, 1);
        gpuByteSize = Math.ceil(numElements / 2) * 4;
      } else if (uploadedDtypes[i] !== originalDtype) {
        gpuByteSize = (tensorsForGpu[i].byteSize / 2) * 4; // expanded from 2 to 4 bytes per element
      } else {
        gpuByteSize = tensorsForGpu[i].byteSize;
      }
      gpuTensors.push({
        name: tensorsForGpu[i].name,
        handle: handles[i],
        dtypeCode: dtypeToCode(originalDtype),
        shape: tensorsForGpu[i].shape,
        byteSize: gpuByteSize,
        packedBf16: isPacked,
      });
    }

    post({
      type: 'progress',
      step: 'gpu_upload',
      message: `${tensorsForGpu.length} GPU buffers created (${(performance.now() - t0).toFixed(0)}ms)`,
    });

    // 5. Build model
    post({ type: 'progress', step: 'init_model', message: 'Initializing model...' });
    const t1 = performance.now();
    const Qwen35Model = mlxExports.Qwen35Model || mlxExports.Qwen3_5Model;
    model = Qwen35Model.loadFromGpuBuffers(configJson, gpuTensors, tokenizerJson, tokenizerConfigJson);
    post({ type: 'progress', step: 'init_model', message: `Model ready (${(performance.now() - t1).toFixed(0)}ms)` });

    // 6. Pipeline warmup: run a minimal inference to force WebGPU shader
    // compilation for every kernel used in the decode hot path. Without this,
    // the first user-visible inference pays 100-300ms of pipeline compilation
    // on the critical path (TTFT). Warmup uses a 1-token prompt and 2 output
    // tokens — enough to exercise prompt+decode code paths once.
    try {
      post({ type: 'progress', step: 'warmup', message: 'Warming up pipelines...' });
      const t2 = performance.now();
      const warmupFn = model.chatSync || model.chat_sync || model.chat;
      warmupFn.call(model, [{ role: 'user', content: 'hi' }], {
        enableThinking: false,
        reportPerformance: false,
        maxNewTokens: 2,
        max_new_tokens: 2,
        temperature: 0,
      });
      post({ type: 'progress', step: 'warmup', message: `Warmup done (${(performance.now() - t2).toFixed(0)}ms)` });
    } catch (warmupErr) {
      // Non-fatal: if warmup fails, first user inference just pays the
      // compilation cost instead. Log and continue.
      post({ type: 'progress', step: 'warmup', message: `Warmup skipped: ${String(warmupErr)}` });
    }

    post({ type: 'ready' });
  } catch (e) {
    post({ type: 'error', message: String(e), stack: e instanceof Error ? e.stack : undefined });
  }
}

async function handleChat(data: { messages: any[]; config?: any }) {
  try {
    const chatFn = model.chatSync || model.chat_sync || model.chat;
    const chatConfig = {
      ...(data.config ?? {}),
      enableThinking: true, // Let model think — thinking block gets stripped from final output
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
