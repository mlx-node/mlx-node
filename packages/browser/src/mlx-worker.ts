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
import { Buffer } from "buffer";
(globalThis as any).Buffer = Buffer;

// Patch TextDecoder to handle SharedArrayBuffer views.
// WASM memory is a SharedArrayBuffer (for threads support), but TextDecoder
// rejects shared views. Copy to a non-shared buffer before decoding.
const _origDecode = TextDecoder.prototype.decode;
TextDecoder.prototype.decode = function (
  input?: BufferSource | null,
  options?: TextDecodeOptions,
) {
  if (
    input &&
    ArrayBuffer.isView(input) &&
    (input.buffer as any) instanceof SharedArrayBuffer
  ) {
    input = (input as Uint8Array).slice();
  }
  return _origDecode.call(this, input, options);
};

import {
  instantiateNapiModule,
  getDefaultContext,
  WASI,
} from "@napi-rs/wasm-runtime";

import {
  CMD_OFFSET,
  READBACK_BUFFER_SIZE,
  DISPATCH_BATCH_BUFFER_SIZE,
  STATS_BUFFER_SIZE,
} from "./rpc-protocol.js";
import {
  parseSafeTensorsHeader,
  dtypeToCode,
  type TensorInfo,
} from "./safetensors.js";
import {
  createBridgeStub,
  POOL_STATS_SIZE_BYTES,
  BUFFER_METADATA_SIZE_BYTES,
  type BridgeStub,
} from "./webgpu-bridge-stub.js";
import { sanitizeAssistantText } from "./generated-text.js";

let model: any = null;
let mlxExports: any = null;
let wasmInst: WebAssembly.Instance | null = null;
let cppTag: any = null;
let wasmMemory: WebAssembly.Memory | null = null;
let wasmMalloc: ((size: number) => number) | null = null;
let wasmFree: ((ptr: number) => void) | null = null;
let modelSupportsImages = false;
let bridgeOptimizationControl: Int32Array | null = null;
let configuredFusionEnabled = true;
let configuredDispatchBatchEnabled = false;

// GPU-buffer import currently wraps JS-created WGPUBuffer handles before Rust
// builds Linear::weight_t transpose views. Until packed-bf16 metadata is proven
// to survive that path end-to-end, keep JS-side packed weight upload disabled:
// otherwise raw bf16 pairs can be read by normal f32 kernels and corrupt logits.
const PACKED_GPU_WEIGHT_UPLOAD_ENABLED = false;

type UploadTensorInfo = TensorInfo & { fromMerged?: boolean };
type ReasoningEffort = "off" | "low" | "medium" | "high";

const REASONING_TOKEN_BUDGET: Record<
  Exclude<ReasoningEffort, "off">,
  number
> = {
  low: 32,
  medium: 128,
  high: 256,
};

function normalizeReasoningEffort(
  value: unknown,
  enableThinking?: boolean,
): ReasoningEffort {
  if (value === "off" || value === "none") return "off";
  if (value === "low" || value === "medium" || value === "high") return value;
  return enableThinking === true ? "high" : "off";
}

function buildChatConfig(data: {
  config?: any;
  enableThinking?: boolean;
  reasoningEffort?: unknown;
}) {
  const effort = normalizeReasoningEffort(
    data.reasoningEffort,
    data.enableThinking,
  );
  if (effort === "off") {
    return {
      ...data.config,
      reasoningEffort: "none",
      includeReasoning: false,
      reportPerformance: true,
    };
  }

  return {
    ...data.config,
    // Current Qwen3.5 wasm treats "low" as no-thinking. Keep the playground
    // control Codex-shaped by enabling the template and enforcing the low
    // budget here.
    reasoningEffort: effort === "low" ? "medium" : effort,
    thinkingTokenBudget: REASONING_TOKEN_BUDGET[effort],
    includeReasoning: true,
    reportPerformance: true,
  };
}

function isVisionTensorName(name: string): boolean {
  return (
    name.startsWith("vision_tower.") ||
    name.startsWith("visual.") ||
    name.startsWith("model.visual.")
  );
}

function hasMessageImages(messages: any[]): boolean {
  return messages.some(
    (message) => Array.isArray(message?.images) && message.images.length > 0,
  );
}

function rejectUnsupportedImages(messages: any[]): boolean {
  if (modelSupportsImages || !hasMessageImages(messages)) return false;
  post({
    type: "error",
    message: "Image input is unavailable for the loaded text-only model.",
  });
  return true;
}

function setBridgeOptimizations(
  fusionEnabled: boolean,
  passCachingEnabled: boolean,
  dispatchBatchEnabled: boolean,
) {
  if (!bridgeOptimizationControl) return;
  Atomics.store(bridgeOptimizationControl, 0, fusionEnabled ? 1 : 0);
  Atomics.store(bridgeOptimizationControl, 1, passCachingEnabled ? 1 : 0);
  Atomics.store(bridgeOptimizationControl, 2, dispatchBatchEnabled ? 1 : 0);
}

async function withBridgeModeForMessages<T>(
  messages: any[],
  run: () => Promise<T>,
): Promise<T> {
  const imageRequest = hasMessageImages(messages);
  if (!bridgeOptimizationControl) return run();

  const previousFusion = Atomics.load(bridgeOptimizationControl, 0) !== 0;
  const previousPassCaching = Atomics.load(bridgeOptimizationControl, 1) !== 0;
  const previousDispatchBatch = Atomics.load(bridgeOptimizationControl, 2) !== 0;
  if (imageRequest) {
    // VLM prefill still needs the conservative bridge path because it has
    // cross-worker pass ownership and copy/restart patterns that are not yet
    // safe with the text decode fusion stack.
    setBridgeOptimizations(false, false, false);
  } else {
    setBridgeOptimizations(configuredFusionEnabled, true, configuredDispatchBatchEnabled);
  }

  try {
    return await run();
  } finally {
    setBridgeOptimizations(
      previousFusion,
      previousPassCaching,
      previousDispatchBatch,
    );
  }
}

function buildMergedLinearAttentionTensors(
  tensors: TensorInfo[],
  weights: Uint8Array,
  dataOffset: number,
): {
  tensors: UploadTensorInfo[];
  mergedSab?: SharedArrayBuffer;
  mergedCount: number;
} {
  const byName = new Map<string, TensorInfo>();
  for (const tensor of tensors) byName.set(tensor.name, tensor);

  const consumed = new Set<string>();
  const merged: UploadTensorInfo[] = [];
  const mergedChunks: Uint8Array[] = [];
  let mergedBytes = 0;

  const tryMergePair = (
    leftSuffix: string,
    rightSuffix: string,
    mergedSuffix: string,
  ) => {
    for (const left of tensors) {
      if (!left.name.endsWith(leftSuffix) || consumed.has(left.name)) continue;
      const prefix = left.name.slice(0, -leftSuffix.length);
      const right = byName.get(`${prefix}${rightSuffix}`);
      if (!right || consumed.has(right.name)) continue;
      if (
        left.dtype !== right.dtype ||
        left.shape.length !== right.shape.length
      )
        continue;
      let compatible = true;
      for (let i = 1; i < left.shape.length; i++) {
        if (left.shape[i] !== right.shape[i]) {
          compatible = false;
          break;
        }
      }
      if (!compatible) continue;

      const byteOffset = mergedBytes;
      const byteSize = left.byteSize + right.byteSize;
      const shape = [left.shape[0] + right.shape[0], ...left.shape.slice(1)];
      const chunk = new Uint8Array(byteSize);
      chunk.set(
        new Uint8Array(
          weights.buffer,
          dataOffset + left.byteOffset,
          left.byteSize,
        ),
        0,
      );
      chunk.set(
        new Uint8Array(
          weights.buffer,
          dataOffset + right.byteOffset,
          right.byteSize,
        ),
        left.byteSize,
      );
      mergedChunks.push(chunk);
      mergedBytes += byteSize;
      consumed.add(left.name);
      consumed.add(right.name);
      merged.push({
        name: `${prefix}${mergedSuffix}`,
        dtype: left.dtype,
        shape,
        byteOffset,
        byteSize,
        fromMerged: true,
      });
    }
  };

  for (const suffix of ["weight", "scales", "biases"]) {
    tryMergePair(
      `.linear_attn.in_proj_qkv.${suffix}`,
      `.linear_attn.in_proj_z.${suffix}`,
      `.linear_attn.in_proj_qkvz.${suffix}`,
    );
    tryMergePair(
      `.linear_attn.in_proj_b.${suffix}`,
      `.linear_attn.in_proj_a.${suffix}`,
      `.linear_attn.in_proj_ba.${suffix}`,
    );
  }

  const uploadTensors: UploadTensorInfo[] = tensors.filter(
    (tensor) => !consumed.has(tensor.name),
  );
  uploadTensors.push(...merged);
  if (mergedBytes === 0) return { tensors: uploadTensors, mergedCount: 0 };

  const mergedSab = new SharedArrayBuffer(mergedBytes);
  const mergedView = new Uint8Array(mergedSab);
  let cursor = 0;
  for (const chunk of mergedChunks) {
    mergedView.set(chunk, cursor);
    cursor += chunk.byteLength;
  }
  return { tensors: uploadTensors, mergedSab, mergedCount: merged.length };
}

function uploadWeightsToGpu(
  gpuWorker: Worker,
  weightsSab: SharedArrayBuffer,
  dataOffset: number,
  tensors: UploadTensorInfo[],
  mergedSab: SharedArrayBuffer | undefined,
  packBf16: boolean,
): Promise<{
  handles: number[];
  uploadedDtypes: string[];
  uploadedByteSizes: number[];
  packedBf16Flags: boolean[];
  debugPackedTotal?: number;
  debugUnpackedBf16Total?: number;
}> {
  return new Promise((resolve, reject) => {
    const onMessage = (ev: MessageEvent) => {
      const msg = ev.data;
      if (msg?.type === "weights_uploaded") {
        cleanup();
        resolve(msg);
      } else if (msg?.type === "error" || msg?.type === "rpc-error") {
        cleanup();
        reject(new Error(msg.message || "GPU weight upload failed"));
      }
    };
    const onError = (ev: ErrorEvent) => {
      cleanup();
      reject(new Error(ev.message || "GPU worker error during weight upload"));
    };
    const cleanup = () => {
      gpuWorker.removeEventListener("message", onMessage);
      gpuWorker.removeEventListener("error", onError);
    };
    gpuWorker.addEventListener("message", onMessage);
    gpuWorker.addEventListener("error", onError);
    gpuWorker.postMessage({
      type: "upload_weights",
      weightsSab,
      mergedSab,
      dataOffset,
      tensors,
      packBf16,
    });
  });
}

// Phase 0 (?profile=1) state. When enabled, handleChat* wraps each
// generation with a reset + readback of:
//   - wasm-worker rpcCall histogram (bridge.getBridgeStats)
//   - gpu-worker rpcCount histogram (bridge.fetchGpuWorkerStats)
//   - C++ WebGPU dispatch counters (mlxExports.wgpuGetDispatchStats)
// Then posts a {type:'profile', stats:...} message to the main thread.
let profileEnabled = false;
let bridgeRef: BridgeStub | null = null;
// Per-worker DISPATCH_BATCH index counter. Child workers (created by emnapi's
// asyncWorkPoolSize) receive batch buffer IDs 1, 2, 3, ...
let nextWorkerBatchIndex = 1;

function post(msg: any) {
  (self as any).postMessage(msg);
}

async function handleInit(data: {
  wasmUrl: string;
  modelUrl: string;
  packBf16?: boolean;
  sdpaFallback?: boolean;
  profile?: boolean;
  dispatchBatch?: boolean;
  compileMlp?: boolean;
  compileGdnPre?: boolean;
  compileGdnPost?: boolean;
  compileGdnG?: boolean;
  enableVlm?: boolean;
  fuseDispatch?: boolean;
}) {
  const packBf16 = data.packBf16 === true;
  const sdpaFallback = data.sdpaFallback === true;
  profileEnabled = data.profile === true;
  const dispatchBatch = data.dispatchBatch === true;
  // Phase 6b: SwiGLU MLP compile fast path. Default ON; disable via
  // ?compile_mlp=0 on the demo URL. Plumbed through handleInit so the
  // backend flag flips BEFORE any model forward pass runs.
  const compileMlp = data.compileMlp !== false;
  // Phase 6c: GDN pre-recurrence compile fast path. Default ON; disable
  // via ?compile_gdn_pre=0 on the demo URL. Same plumbing pattern as the
  // Phase 6b MLP flag — process-wide atomic flipped before any model
  // forward pass runs.
  const compileGdnPre = data.compileGdnPre !== false;
  // Phase 6d: GDN post-recurrence compile fast path. Default ON; disable
  // via ?compile_gdn_post=0 on the demo URL. Same pattern as 6b/6c —
  // process-wide atomic flipped before any forward pass runs.
  const compileGdnPost = data.compileGdnPost !== false;
  // Phase 6e: GDN decay-gate (compute_g) compile fast path. Default ON;
  // disable via ?compile_gdn_g=0 on the demo URL. Same pattern as
  // 6b/6c/6d — process-wide atomic flipped before any forward pass runs.
  const compileGdnG = data.compileGdnG !== false;
  const enableVlm = data.enableVlm !== false;
  const fuseDispatch = data.fuseDispatch !== false;
  configuredFusionEnabled = fuseDispatch;
  configuredDispatchBatchEnabled = dispatchBatch;
  try {
    // 1. Spawn gpu-worker (owns GPUDevice, event loop free for GPU callbacks)
    post({ type: "progress", step: "gpu", message: "Initializing WebGPU..." });
    const cmdBuffer = new SharedArrayBuffer(CMD_OFFSET.TOTAL);
    const readbackBuffer = new SharedArrayBuffer(READBACK_BUFFER_SIZE);
    const poolStatsBuffer = profileEnabled
      ? new SharedArrayBuffer(POOL_STATS_SIZE_BYTES)
      : undefined;
    // Phase 2: dedicated dispatch-batch SABs — one per worker (main + up to
    // asyncWorkPoolSize child workers). Each worker gets its own SAB so
    // DISPATCH_BATCH can be safely enabled on child pthread workers without
    // races on a shared batch cursor. The gpu-worker receives all buffers in
    // init and looks up the right one by batchBufferId.
    const NUM_BATCH_BUFFERS = 5; // 0 = main, 1..4 = child workers
    const batchBuffers: SharedArrayBuffer[] = [];
    for (let i = 0; i < NUM_BATCH_BUFFERS; i++) {
      batchBuffers.push(new SharedArrayBuffer(DISPATCH_BATCH_BUFFER_SIZE));
    }
    const dispatchBatchBuffer = batchBuffers[0]!;
    // Task 3: shared buffer-metadata SAB. Every bridge stub spawned against
    // this wasmMemory (main + child pthread workers created by emnapi's
    // asyncWorkPoolSize) reads and writes the same (size, usage) table,
    // fixing the `unknownHandle=100%` pathology where handles created on
    // one stub were released from another and the release-side metadata
    // lookup came up empty. Sized for ~35 back-to-back Qwen3.5-0.8B
    // decodes; see webgpu-bridge-stub.ts for the capacity calculation.
    const bufferMetadataBuffer = new SharedArrayBuffer(
      BUFFER_METADATA_SIZE_BYTES,
    );
    // JS-F010: dedicated 1 KiB stats SAB for the gpu-worker's per-opcode
    // RPC histogram. Replaces the old scheme that striped the histogram
    // across the cmd SAB's CALLBACK_BASE / UNIFORM_DATA / reserved regions
    // (which was "safe by serialization" — one loosening of cmd-SAB
    // single-slot and a fused dispatch would silently lose its UNIFORM_DATA
    // bytes). Plumbed into both the gpu-worker init message and every
    // createBridgeStub call so main + child workers share the same view.
    const statsBuffer = profileEnabled
      ? new SharedArrayBuffer(STATS_BUFFER_SIZE)
      : undefined;
    const optimizationControlBuffer = new SharedArrayBuffer(12);
    bridgeOptimizationControl = new Int32Array(optimizationControlBuffer);
    setBridgeOptimizations(
      configuredFusionEnabled,
      true,
      configuredDispatchBatchEnabled,
    );
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
    wasmMemory = sharedMemory;

    const gpuWorker = new Worker(new URL("./gpu-worker.ts", import.meta.url), {
      type: "module",
    });

    // Wait for gpu-worker to create GPUDevice and be ready
    const gpuReady = await new Promise<any>((resolve, reject) => {
      gpuWorker.onmessage = (e) => {
        if (e.data.type === "ready") resolve(e.data);
        else if (e.data.type === "error") reject(new Error(e.data.message));
      };
      gpuWorker.postMessage({
        type: "init",
        cmdBuffer,
        readbackBuffer,
        wasmMemory: sharedMemory, // Send Memory object, not .buffer — .buffer getter always returns current byteLength after grow()
        batchBuffers,
        statsBuffer,
      });
    });
    post({ type: "progress", step: "gpu", message: "WebGPU ready" });

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
      poolStatsBuffer,
      dispatchBatchBuffer,
      dispatchBatch,
      0, // batchBufferId = 0 for main worker
      bufferMetadataBuffer,
      statsBuffer,
      {
        fusionEnabled: configuredFusionEnabled,
        passCachingEnabled: true,
        optimizationControlBuffer,
      },
    );
    // Retain for ?profile=1 readback (resetBridgeStats / fetchGpuWorkerStats
    // need the same BridgeStub instance that owns the SAB cmdBuffer view).
    bridgeRef = bridge;

    // 3. Load WASM with bridge stub
    post({ type: "progress", step: "wasm", message: "Loading WASM module..." });
    // Forward lines starting with [MLX-KERNEL] to the main thread so we can
    // verify which matmul kernel variants fire. The WASM stderr lands on
    // the mlx-worker's own devtools target which isn't readable from the
    // main frame console.
    const wasi = new WASI({
      version: "preview1",
      print: function (...args: unknown[]) {
        const line = args.map(String).join(" ");
        if (line.includes("[MLX-KERNEL]")) {
          post({ type: "log", message: line });
        } else {
          console.log(...args);
        }
      },
      printErr: function (...args: unknown[]) {
        const line = args.map(String).join(" ");
        if (line.includes("[MLX-KERNEL]")) {
          post({ type: "log", message: line });
        } else {
          console.error(...args);
        }
      },
    } as any);
    const context = getDefaultContext();
    const wasmFile = await fetch(data.wasmUrl).then((r) => r.arrayBuffer());

    const cppExceptionTag = new WebAssembly.Tag({ parameters: ["i32"] });
    cppTag = cppExceptionTag;

    // Task 4's SabSink declares extern "C" fn __wasm_i32_atomic_wait /
    // __wasm_atomic_notify expecting LLVM compiler-rt builtins, but wasm-ld
    // emits them as host imports on wasm32-wasip1-threads. Provide JS stubs
    // that wrap Atomics.wait / Atomics.notify over the shared memory.
    // Cache the Int32Array view once (shared memory never detaches, so the
    // view stays valid for the lifetime of the worker). Avoids per-call
    // TypedArray construction on the hot atomic notify path.
    const sharedI32 = new Int32Array(sharedMemory.buffer);
    const cxxStubs = {
      __cpp_exception: cppExceptionTag,
      _ZN3mlx4core3gpu4initEv: () => {},
      __wasm_i32_atomic_wait: (
        ptr: number,
        expected: number,
        timeoutNs: bigint,
      ): number => {
        const index = ptr >>> 2;
        const timeoutMs =
          timeoutNs === -1n ? Infinity : Number(timeoutNs / 1_000_000n);
        const result = Atomics.wait(sharedI32, index, expected, timeoutMs);
        return result === "ok" ? 0 : result === "not-equal" ? 1 : 2;
      },
      __wasm_atomic_notify: (ptr: number, count: number): number => {
        return Atomics.notify(sharedI32, ptr >>> 2, count);
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
        const w = new Worker(new URL("./webgpu-worker.mjs", import.meta.url), {
          type: "module",
        });
        // Per-worker DISPATCH_BATCH: each child worker gets its own batch SAB
        // (indexed 1..4) so batching can be enabled safely without races.
        const workerBatchIndex = nextWorkerBatchIndex++;
        const workerBatchBuffer =
          batchBuffers[workerBatchIndex] ?? batchBuffers[0]!;
        // Child pthreads share the optimization-control SAB with the main
        // worker, so text decode keeps the fast bridge path while VLM image
        // prefill can temporarily force immediate bridge calls.
        w.postMessage({
          type: "__mlx_rpc_config",
          cmdBuffer,
          readbackBuffer,
          poolStatsBuffer,
          dispatchBatchBuffer: workerBatchBuffer,
          batchBufferId: workerBatchIndex,
          dispatchBatch,
          fusionEnabled: configuredFusionEnabled,
          passCachingEnabled: true,
          optimizationControlBuffer,
          bufferMetadataBuffer,
          statsBuffer,
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
          if (name.startsWith("__napi_register__")) {
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
    if (typeof mlxExports.wgpuSetPackedBf16Enabled === "function") {
      mlxExports.wgpuSetPackedBf16Enabled(packBf16);
    }
    // Flip the SDPA fallback kill-switch. When true, the WebGPU backend's
    // fast/tile SDPA kernels are bypassed in favor of the decomposed
    // matmul→softmax→matmul path — used by the demo to A/B the fused
    // kernels against the baseline without a rebuild.
    if (typeof mlxExports.wgpuSetSdpaFallbackForced === "function") {
      mlxExports.wgpuSetSdpaFallbackForced(sdpaFallback);
    }
    // Phase 6b SwiGLU MLP compile fast path. The setter flips a process-wide
    // std::atomic<bool> in mlx_fused_ops.cpp so the next call to
    // mlx_swiglu_mlp_forward routes the element-wise tail through
    // mlx::core::compile. Default ON; disable via ?compile_mlp=0.
    if (typeof mlxExports.wgpuSetSwigluCompileEnabled === "function") {
      mlxExports.wgpuSetSwigluCompileEnabled(compileMlp);
    }
    // Phase 6c GDN pre-recurrence compile fast path. Same pattern as Phase
    // 6b — process-wide atomic flag, flipped before any model forward pass
    // runs. Default ON; disable via ?compile_gdn_pre=0.
    if (typeof mlxExports.wgpuSetGdnPreCompileEnabled === "function") {
      mlxExports.wgpuSetGdnPreCompileEnabled(compileGdnPre);
    }
    // Phase 6d GDN post-recurrence compile fast path. Same pattern as
    // Phase 6b/6c — process-wide atomic flag, flipped before any forward
    // pass runs. Default ON; disable via ?compile_gdn_post=0.
    if (typeof mlxExports.wgpuSetGdnPostCompileEnabled === "function") {
      mlxExports.wgpuSetGdnPostCompileEnabled(compileGdnPost);
    }
    // Phase 6e GDN decay-gate (compute_g) compile fast path. Same pattern
    // as 6b/6c/6d — process-wide atomic flag, flipped before any forward
    // pass runs. Default ON; disable via ?compile_gdn_g=0.
    if (typeof mlxExports.wgpuSetGdnGCompileEnabled === "function") {
      mlxExports.wgpuSetGdnGCompileEnabled(compileGdnG);
    }
    const flagSuffix =
      (packBf16 ? " pack_bf16=1" : "") +
      (sdpaFallback ? " sdpa_fallback=1" : "") +
      (compileMlp ? " compile_mlp=1" : "") +
      (compileGdnPre ? " compile_gdn_pre=1" : "") +
      (compileGdnPost ? " compile_gdn_post=1" : "") +
      (compileGdnG ? " compile_gdn_g=1" : "") +
      (fuseDispatch ? "" : " fuse_dispatch=0");
    post({
      type: "progress",
      step: "wasm",
      message: `WASM loaded${flagSuffix ? " (" + flagSuffix.trim() + ")" : ""}`,
    });

    // 3. Fetch model files
    post({ type: "progress", step: "model", message: "Fetching config..." });
    const configJson = await fetch(`${data.modelUrl}/config.json`).then((r) =>
      r.text(),
    );

    post({ type: "progress", step: "model", message: "Fetching tokenizer..." });
    const tokenizerJson = await fetch(`${data.modelUrl}/tokenizer.json`).then(
      (r) => r.text(),
    );
    // Fetch tokenizer_config.json for the Jinja2 chat template
    let tokenizerConfigJson: string | undefined;
    try {
      const resp = await fetch(`${data.modelUrl}/tokenizer_config.json`);
      if (resp.ok) {
        tokenizerConfigJson = await resp.text();
      }
    } catch (e) {
      console.warn(
        "tokenizer_config.json not available, using default chat template",
      );
    }

    post({ type: "progress", step: "model", message: "Fetching weights..." });
    const weightsResponse = await fetch(`${data.modelUrl}/model.safetensors`);
    const totalSize = Number(
      weightsResponse.headers.get("content-length") || 0,
    );
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
        post({
          type: "progress",
          step: "download",
          message: `Downloading weights... ${pct}%`,
          pct: Number(pct),
        });
      }
    }

    const weightsSab = new SharedArrayBuffer(loaded);
    const weightsBuffer = new Uint8Array(weightsSab);
    let offset = 0;
    for (const chunk of chunks) {
      weightsBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    // 4. Parse safetensors and build the model. Prefer the GPU-buffer path:
    // JS uploads raw weights directly into WebGPU buffers, then Rust wraps the
    // handles with loadFromGpuBuffers. This is the only active load path that
    // can preserve PackedBf16 storage for decode weights.
    post({
      type: "progress",
      step: "init_model",
      message: "Parsing safetensors header...",
    });
    wasmMalloc = wasmInst!.exports.malloc as (size: number) => number;
    wasmFree = wasmInst!.exports.free as (ptr: number) => void;
    const localWasmMalloc = wasmMalloc;
    const localWasmFree = wasmFree;

    const { tensors, dataOffset } = parseSafeTensorsHeader(
      weightsBuffer.buffer,
    );
    const visionTensorCount = tensors.filter((tensor) =>
      isVisionTensorName(tensor.name),
    ).length;
    modelSupportsImages = enableVlm && visionTensorCount > 0;
    const tensorsForModel = modelSupportsImages
      ? tensors
      : tensors.filter((tensor) => !isVisionTensorName(tensor.name));
    post({
      type: "log",
      message: `[MODEL] Parsed ${tensors.length} tensors, dataOffset=${dataOffset}`,
    });
    post({
      type: "log",
      message: modelSupportsImages
        ? `[MODEL] Vision input enabled (${visionTensorCount} vision tensors)`
        : visionTensorCount > 0
          ? `[MODEL] Vision tensors present (${visionTensorCount}) but VLM input is disabled by ?disable_vlm=1`
          : "[MODEL] Vision input unavailable for this model",
    });

    const Qwen35Model = mlxExports.Qwen35Model || mlxExports.Qwen3_5Model;

    if (typeof Qwen35Model.loadFromGpuBuffers === "function") {
      const prepared = buildMergedLinearAttentionTensors(
        tensorsForModel,
        weightsBuffer,
        dataOffset,
      );
      post({
        type: "progress",
        step: "init_model",
        message: `Uploading ${prepared.tensors.length} GPU tensors${prepared.mergedCount ? ` (${prepared.mergedCount} merged)` : ""}...`,
      });
      const uploadPackedBf16 = packBf16 && PACKED_GPU_WEIGHT_UPLOAD_ENABLED;
      if (packBf16 && !uploadPackedBf16) {
        post({
          type: "log",
          message:
            "[GPU] Packed bf16 GPU-buffer upload disabled for correctness; using f32-expanded weights",
        });
      }
      const uploaded = await uploadWeightsToGpu(
        gpuWorker,
        weightsSab,
        dataOffset,
        prepared.tensors,
        prepared.mergedSab,
        uploadPackedBf16,
      );
      if (uploaded.handles.length !== prepared.tensors.length) {
        throw new Error(
          `GPU upload returned ${uploaded.handles.length} handles for ${prepared.tensors.length} tensors`,
        );
      }
      const gpuTensors = prepared.tensors.map((tensor, i) => ({
        name: tensor.name,
        handle: uploaded.handles[i],
        dtypeCode: dtypeToCode(uploaded.uploadedDtypes[i] ?? tensor.dtype),
        shape: tensor.shape,
        byteSize: uploaded.uploadedByteSizes[i] ?? tensor.byteSize,
        packedBf16: uploaded.packedBf16Flags[i] === true,
      }));
      post({
        type: "log",
        message:
          `[GPU] Uploaded ${gpuTensors.length} tensors` +
          ` packed=${uploaded.debugPackedTotal ?? 0}` +
          ` unpacked_bf16=${uploaded.debugUnpackedBf16Total ?? 0}`,
      });

      post({
        type: "progress",
        step: "init_model",
        message: "Building model from GPU buffers...",
      });
      const t1 = performance.now();
      model = await Qwen35Model.loadFromGpuBuffers(
        configJson,
        gpuTensors,
        tokenizerJson,
        tokenizerConfigJson ?? null,
      );
      post({
        type: "progress",
        step: "init_model",
        message: `Model ready (${(performance.now() - t1).toFixed(0)}ms)`,
      });
    } else {
      // Fallback per-tensor CPU data path. For each tensor: malloc small WASM
      // buffer → copy bytes → call addCpuTensor (Rust creates MLX array, C++
      // copies data) → free WASM buffer immediately.
      //
      // This keeps peak WASM pointer under 2 GB, avoiding the i32 offset
      // overflow that kills bulk CPU loading.
      post({
        type: "log",
        message: "[CPU] loadFromGpuBuffers unavailable; using CPU tensor path",
      });
      // Store config + tokenizer in Rust BEFORE tensor accumulation.
      // Must happen while WASM memory is still small to avoid emnapi DataView
      // bounds errors (the ~5.7 MB tokenizer JSON triggers memory writes that
      // fail if emnapi's DataView is stale after memory.grow).
      Qwen35Model.setCpuModelConfig(
        configJson,
        tokenizerJson,
        tokenizerConfigJson ?? null,
      );

      post({
        type: "progress",
        step: "init_model",
        message: `Loading ${tensorsForModel.length} tensors...`,
      });
      for (let i = 0; i < tensorsForModel.length; i++) {
        const t = tensorsForModel[i];
        if (t.byteSize === 0) continue;

        const ptr = localWasmMalloc(t.byteSize);
        if (ptr === 0) {
          throw new Error(
            `Failed to malloc ${t.byteSize} bytes for tensor "${t.name}" (${i}/${tensors.length})`,
          );
        }
        const uptr = ptr >>> 0;

        const src = new Uint8Array(
          weightsBuffer.buffer,
          dataOffset + t.byteOffset,
          t.byteSize,
        );
        new Uint8Array(sharedMemory.buffer).set(src, uptr);

        Qwen35Model.addCpuTensor(
          t.name,
          uptr,
          t.byteSize,
          t.shape,
          dtypeToCode(t.dtype),
        );
        localWasmFree(ptr);

        if (i % 50 === 0) {
          post({
            type: "progress",
            step: "init_model",
            message: `Loaded ${i}/${tensorsForModel.length} tensors...`,
          });
        }
      }
      post({
        type: "log",
        message: `[CPU] All ${tensorsForModel.length} tensors accumulated in Rust`,
      });

      post({
        type: "progress",
        step: "init_model",
        message: "Building model...",
      });
      const t1 = performance.now();
      model = await Qwen35Model.buildModelFromCpuTensors();
      post({
        type: "progress",
        step: "init_model",
        message: `Model ready (${(performance.now() - t1).toFixed(0)}ms)`,
      });
    }

    // 6. Pipeline warmup — first inference warms GPU pipelines + shader compilation.
    // The VLM-capable Qwen3.5 checkpoint uses a vision-aware chat template; a
    // text-only warmup can wedge the worker before the UI becomes usable. Let
    // the first real image request warm the VLM path instead.
    if (modelSupportsImages) {
      post({
        type: "log",
        message: "[WARMUP] skipped for VLM-capable model",
      });
      post({ type: "progress", step: "warmup", message: "Warmup skipped" });
    } else {
      post({ type: "progress", step: "warmup", message: "Warming up..." });
      const warmupResult = await model.chat([{ role: "user", content: "hi" }], {
        maxNewTokens: 2,
        temperature: 0,
        reasoningEffort: "none",
        includeReasoning: false,
        reuseCache: false,
      });
      post({
        type: "log",
        message: `[WARMUP] rawText=${warmupResult.rawText} text=${warmupResult.text} finish=${warmupResult.finishReason}`,
      });
      post({ type: "progress", step: "warmup", message: "Warmup complete" });
    }

    // Ship the shared WASM memory to the main thread so it can mount the
    // SAB chat-stream reader directly over the WASM heap. A shared
    // WebAssembly.Memory is structured-cloneable and the clone refers to
    // the SAME underlying SharedArrayBuffer, so main-thread Int32Array
    // views over memory.buffer see every byte the Rust SabSink writes.
    // This is required to avoid delivering tokens in a single burst at
    // end-of-decode: WASM-side chatStreamSab synchronously blocks this
    // mlx-worker on Atomics.wait round-trips to gpu-worker, so a reader
    // whose Atomics.waitAsync promise resolves on this thread cannot
    // actually run its microtask until chatStreamSab returns. The main
    // thread is never blocked during decode, so the reader drains live.
    post({ type: "ready", sharedMemory, supportsImages: modelSupportsImages });
  } catch (e) {
    post({
      type: "error",
      message: String(e),
      stack: e instanceof Error ? e.stack : undefined,
    });
  }
}

const SAB_RING_SIZE = 262_144;

// Phase 0 (?profile=1) helpers. These are no-ops unless profileEnabled is
// true — each chat/chatStream handler calls resetProfileCounters() before
// generation and postProfileSnapshot() after. See the 'profile' message
// handler in demo/app.ts for the display side.
function resetProfileCounters(): void {
  if (!profileEnabled) return;
  try {
    bridgeRef?.resetBridgeStats();
  } catch (e) {
    console.warn("[mlx-worker] resetBridgeStats failed:", e);
  }
  try {
    if (mlxExports && typeof mlxExports.wgpuResetDispatchStats === "function") {
      mlxExports.wgpuResetDispatchStats();
    }
  } catch (e) {
    console.warn("[mlx-worker] wgpuResetDispatchStats failed:", e);
  }
  try {
    // Zero the gpu-worker histogram too (reset=1 on the RPC).
    bridgeRef?.fetchGpuWorkerStats(true);
  } catch (e) {
    console.warn("[mlx-worker] fetchGpuWorkerStats(reset=true) failed:", e);
  }
}

function postProfileSnapshot(numTokens: number): void {
  if (!profileEnabled) return;
  try {
    const bridgeStats =
      bridgeRef?.getBridgeStats() ??
      ({
        rpcCount: 0,
        byFn: {},
        poolHits: 0,
        poolMisses: 0,
        diagCreateAll: 0,
        diagCreateMappedCopyDst: 0,
        diagCreateMappedNoCopyDst: 0,
        diagReleaseAll: 0,
        diagReleaseUnknownHandle: 0,
        diagReleaseUnpoolable: 0,
        poisonedRpcCount: 0,
      } as any);
    const gpuStats = bridgeRef?.fetchGpuWorkerStats(false) ?? {
      totalRpcs: 0,
      byFn: {},
      gpuPoolHits: 0,
      gpuPoolMisses: 0,
      bindGroupCacheHits: 0,
      bindGroupCacheMisses: 0,
      uniformHotHits: 0,
      uniformHotMisses: 0,
      spinHits: 0,
      spinMisses: 0,
      spinBudget: 0,
    };
    let totalDispatches = 0;
    let totalPassEnds = 0;
    if (mlxExports && typeof mlxExports.wgpuGetDispatchStats === "function") {
      const arr = mlxExports.wgpuGetDispatchStats() as number[];
      totalDispatches = arr[0] ?? 0;
      totalPassEnds = arr[1] ?? 0;
    }
    post({
      type: "profile",
      stats: {
        numTokens,
        totalDispatches,
        totalPassEnds,
        bridgeRpcCount: bridgeStats.rpcCount,
        bridgeByFn: bridgeStats.byFn,
        gpuRpcCount: gpuStats.totalRpcs,
        gpuByFn: gpuStats.byFn,
        gpuPoolHits: gpuStats.gpuPoolHits,
        gpuPoolMisses: gpuStats.gpuPoolMisses,
        bindGroupCacheHits: gpuStats.bindGroupCacheHits,
        bindGroupCacheMisses: gpuStats.bindGroupCacheMisses,
        uniformHotHits: gpuStats.uniformHotHits,
        uniformHotMisses: gpuStats.uniformHotMisses,
        spinHits: gpuStats.spinHits,
        spinMisses: gpuStats.spinMisses,
        spinBudget: gpuStats.spinBudget,
        poolHits: bridgeStats.poolHits,
        poolMisses: bridgeStats.poolMisses,
        diagCreateAll: (bridgeStats as any).diagCreateAll ?? 0,
        diagCreateMappedCopyDst:
          (bridgeStats as any).diagCreateMappedCopyDst ?? 0,
        diagCreateMappedNoCopyDst:
          (bridgeStats as any).diagCreateMappedNoCopyDst ?? 0,
        diagReleaseAll: (bridgeStats as any).diagReleaseAll ?? 0,
        diagReleaseUnknownHandle:
          (bridgeStats as any).diagReleaseUnknownHandle ?? 0,
        diagReleaseUnpoolable: (bridgeStats as any).diagReleaseUnpoolable ?? 0,
        diagBatchAttempt: (bridgeStats as any).diagBatchAttempt ?? 0,
        diagBatchStaged: (bridgeStats as any).diagBatchStaged ?? 0,
        diagBatchDeferredBlock:
          (bridgeStats as any).diagBatchDeferredBlock ?? 0,
        diagBatchStageRefused: (bridgeStats as any).diagBatchStageRefused ?? 0,
        // JS-F008: RPCs dropped because the bridge was poisoned by an
        // earlier BUFFER_RELEASE_BATCH F&F drain timeout. Non-zero here
        // means the demo should treat the bridge as dead and reload.
        poisonedRpcCount: (bridgeStats as any).poisonedRpcCount ?? 0,
      },
    });
  } catch (e) {
    console.warn("[mlx-worker] postProfileSnapshot failed:", e);
  }
}

// Pending stream-finalize resolver. The main-thread SAB reader posts
// {type:'stream-finalize', numTokens} when it has seen the done-chunk;
// handleChatStreamSab awaits this so it can call postProfileSnapshot with
// the correct token count before freeing the ring.
let streamFinalizeResolve: ((numTokens: number) => void) | null = null;

async function handleChat(data: {
  messages: any[];
  config?: any;
  useSab?: boolean;
  mode?: "sab" | "tsfn" | "baseline";
  enableThinking?: boolean;
  reasoningEffort?: ReasoningEffort;
}) {
  if (!wasmMemory || !wasmMalloc || !wasmFree) {
    post({
      type: "error",
      message: "handleChat called before WASM init complete",
    });
    return;
  }
  if (rejectUnsupportedImages(data.messages)) return;
  const mode = data.mode ?? (data.useSab === false ? "tsfn" : "sab");

  if (mode === "tsfn") {
    await handleChatTsfn(data);
    return;
  }
  if (mode === "baseline") {
    await handleChatBaseline(data);
    return;
  }
  const memory = wasmMemory;
  const mallocFn = wasmMalloc;
  const freeFn = wasmFree;

  // Allocate the ring region INSIDE the WASM heap so that the Buffer we hand to
  // napi-rs shares storage with the Rust SabSink. If we allocated a separate
  // SharedArrayBuffer and wrapped it in Buffer.from, emnapi/napi-rs would copy
  // the bytes into the WASM heap on every call, so Rust would write to a heap
  // copy that this JS reader would never see.
  const ringOffset = mallocFn(SAB_RING_SIZE);
  if (ringOffset === 0) {
    post({
      type: "error",
      message: `Failed to malloc ${SAB_RING_SIZE} bytes for SAB ring`,
    });
    return;
  }

  let freed = false;
  const freeRing = () => {
    if (freed) return;
    freed = true;
    freeFn(ringOffset);
  };

  // Register the finalize-resolver BEFORE we tell main to start reading so the
  // main-thread reader's 'stream-finalize' message can't land before we listen.
  const finalizePromise = new Promise<number>((resolve) => {
    streamFinalizeResolve = resolve;
  });

  try {
    const chatConfig = buildChatConfig(data);

    // Buffer.from(arrayBuffer, offset, length) returns a zero-copy VIEW over
    // the WASM heap. Because the underlying ArrayBuffer IS the WASM memory,
    // napi-rs can pass the pointer through without copying — Rust's SabSink
    // and the JS reader see the same bytes.
    const sabBuf = Buffer.from(memory.buffer, ringOffset, SAB_RING_SIZE);

    // Zero the 32-byte SAB header so seq/write_cur/read_cur/cancelled start
    // at 0 for this stream. Previously createSabRingOverHeap did this as a
    // side-effect of its constructor; now the reader runs on the main
    // thread so we must zero it ourselves before the producer writes.
    new Uint8Array(memory.buffer, ringOffset, 32).fill(0);

    const t0 = performance.now();

    // Phase 0: zero all counters before the generation starts. We reset
    // here rather than at the top of handleChat so the SAB-ring alloc
    // doesn't skew the numbers.
    resetProfileCounters();

    // Tell the main thread where the SAB ring lives. The main thread mounts
    // a reader over the WASM heap at this offset and decodes records as
    // SabSink writes them. Main thread NEVER blocks during WASM decode (this
    // worker does), so its Atomics.waitAsync microtasks run immediately and
    // tokens render live. See the 'stream-finalize' handler below.
    post({ type: "stream-sab-open", ringOffset, size: SAB_RING_SIZE });

    await withBridgeModeForMessages(data.messages, async () => {
      // chatStreamSab writes directly into the heap-backed ring — no TSFN on the
      // hot decode path. It returns a handle once the stream is running and
      // resolves when the producer has written the done record.
      const handle = await model.chatStreamSab(
        data.messages,
        chatConfig,
        sabBuf,
      );
      void handle;

      // Wait for the main-thread reader to drain the done-chunk and send us
      // back its numTokens so we can emit the profile snapshot with correct
      // ratios. This also serialises the ring free — we don't free until the
      // reader has consumed everything.
      const numTokens = await finalizePromise;
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      post({
        type: "progress",
        step: "chat",
        message: `chatStream completed in ${elapsed}s`,
      });
      postProfileSnapshot(numTokens);
    });
  } catch (e) {
    let message = String(e);
    let stack: string | undefined;
    if (e instanceof Error) {
      message = e.message;
      stack = e.stack;
    }
    post({ type: "error", message, stack });
  } finally {
    streamFinalizeResolve = null;
    freeRing();
  }
}

async function handleChatBaseline(data: {
  messages: any[];
  config?: any;
  enableThinking?: boolean;
  reasoningEffort?: ReasoningEffort;
}) {
  try {
    if (rejectUnsupportedImages(data.messages)) return;
    const chatConfig = buildChatConfig(data);
    resetProfileCounters();
    const t0 = performance.now();
    const result = await withBridgeModeForMessages(data.messages, () =>
      model.chat(data.messages, chatConfig),
    );
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    post({
      type: "progress",
      step: "chat",
      message: `chat (non-stream) completed in ${elapsed}s`,
    });
    postProfileSnapshot(result.numTokens ?? 0);
    post({
      type: "result",
      text: sanitizeAssistantText(result.text),
      rawText: sanitizeAssistantText(result.rawText),
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
    }
    post({ type: "error", message, stack });
  }
}

async function handleChatTsfn(data: {
  messages: any[];
  config?: any;
  enableThinking?: boolean;
  reasoningEffort?: ReasoningEffort;
}) {
  try {
    if (rejectUnsupportedImages(data.messages)) return;
    const chatConfig = buildChatConfig(data);
    resetProfileCounters();
    const t0 = performance.now();
    let doneResolve: (() => void) | null = null;
    const doneP = new Promise<void>((r) => {
      doneResolve = r;
    });
    await withBridgeModeForMessages(data.messages, async () => {
      const handle = await model.chatStream(
        data.messages,
        chatConfig,
        (err: Error | null, chunk: any) => {
          if (err) {
            post({ type: "error", message: err.message, stack: err.stack });
            doneResolve?.();
            return;
          }
          if (chunk.done) {
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            post({
              type: "progress",
              step: "chat",
              message: `chatStream completed in ${elapsed}s`,
            });
            postProfileSnapshot(chunk.numTokens ?? 0);
            post({
              type: "result",
              text: sanitizeAssistantText(chunk.text),
              rawText: sanitizeAssistantText(chunk.rawText),
              numTokens: chunk.numTokens,
              finishReason: chunk.finishReason,
              toolCalls: chunk.toolCalls,
              thinking: chunk.thinking,
              performance: chunk.performance
                ? {
                    ttftMs: chunk.performance.ttftMs,
                    prefillTokensPerSecond:
                      chunk.performance.prefillTokensPerSecond,
                    decodeTokensPerSecond:
                      chunk.performance.decodeTokensPerSecond,
                  }
                : null,
            });
            doneResolve?.();
          } else {
            post({
              type: "chunk",
              text: chunk.text,
              isReasoning: chunk.isReasoning ?? false,
            });
          }
        },
      );
      void handle;
      await doneP;
    });
  } catch (e) {
    let message = String(e);
    let stack: string | undefined;
    if (e instanceof Error) {
      message = e.message;
      stack = e.stack;
    }
    post({ type: "error", message, stack });
  }
}

// Catch unhandled errors/rejections on this worker
self.addEventListener("error", (e) => {
  post({
    type: "error",
    message: `Worker error: ${e.message}`,
    stack: (e as ErrorEvent).filename,
  });
});
self.addEventListener("unhandledrejection", (e: PromiseRejectionEvent) => {
  post({ type: "error", message: `Unhandled rejection: ${String(e.reason)}` });
});

self.onmessage = (e: MessageEvent) => {
  switch (e.data.type) {
    case "init":
      handleInit(e.data);
      break;
    case "chat":
      handleChat(e.data);
      break;
    case "stream-finalize":
      // Main-thread SAB reader saw the done-chunk and is handing us the
      // token count so we can emit the profile snapshot with correct
      // ratios before freeing the ring. See handleChat above.
      if (streamFinalizeResolve) {
        streamFinalizeResolve(e.data.numTokens ?? 0);
      }
      break;
  }
};
