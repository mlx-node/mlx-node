/**
 * GPU Worker — Owns GPUDevice, processes WebGPU RPC commands
 *
 * This worker's event loop is always free (never blocked by WASM), so GPU
 * async callbacks (onSubmittedWorkDone, mapAsync) resolve naturally.
 *
 * Communication with wasm-worker:
 *   - SharedArrayBuffer command channel (512 bytes, see rpc-protocol.ts)
 *   - SharedArrayBuffer WASM memory (for reading struct descriptors + data)
 *   - Atomics.waitAsync / Atomics.notify for synchronization
 *
 * The handle table, memory reading helpers, and WebGPU call implementations
 * are ported from webgpu-bridge.ts. The key difference is that callbacks
 * (onSubmittedWorkDone, mapAsync) are written to the callback ring in the
 * command buffer instead of being invoked via wasmTable.get().
 */

import { roundUpBucket } from './buffer-bucket.js';
import {
  RpcFn,
  CMD_OFFSET,
  STATUS,
  STATUS_INDEX,
  MAX_CALLBACKS_PER_CALL,
  CALLBACK_ENTRY_SIZE,
  STATS_OPCODE_SLOTS,
  STATS_CALLBACK_SLOTS,
  STATS_INLINE_OFFSET,
  STATS_INLINE_SLOTS,
  STATS_RESERVED_OFFSET,
  STATS_RESERVED_SLOTS,
  DISPATCH_BATCH_HEADER_BYTES,
} from './rpc-protocol.js';

// ---------- Phase 0 RPC Histogram (?profile=1 observability) ----------
//
// Per-opcode count of RpcFn dispatches handled by this worker. Bumped at
// the very top of processCommand's switch (before any case body runs) and
// read + optionally reset by the GET_STATS handler. The array length is
// STATS_OPCODE_SLOTS so the readback handler can assume a fixed layout
// regardless of how many RpcFn values exist.
const gpuRpcCounts = new Uint32Array(STATS_OPCODE_SLOTS);
let gpuTotalRpcs = 0;

// ---------- Handle Table (1H: sparse array for O(1) index lookup) ----------

const handles: any[] = [null]; // index 0 unused — handles start at 1
// Physical (bucketed) buffer size. This is what the pool keys on and what
// device.createBuffer actually allocated. Used by the pool so that a buffer
// released from bucket B rejoins bucket B on the next lookup.
const bufferSizesArr: (number | undefined)[] = [undefined];
// Logical buffer size as originally requested by the C++ caller (pre-bucket).
// This is what BUFFER_GET_SIZE must return so C++ code that does
// `elems = size / sizeof(dtype)` reads only the valid, initialized range.
// When bucketing is not applied (mappedAtCreation, MAP_READ/MAP_WRITE), this
// equals the physical size.
const bufferLogicalSizesArr: (number | undefined)[] = [undefined];
const bufferUsagesArr: (number | undefined)[] = [undefined];
let nextHandle = 1;

// Buffer pool: (usage, size) → stack of free GPUBuffer handles. MLX's allocator
// churns ~870 transient buffers/tok; reusing them eliminates Chrome's
// createBuffer validation cost on the hot path.
const BUFFER_POOL_CAP_PER_KEY = 32;
const bufferPool = new Map<string, number[]>();
let bufferPoolHitCount = 0;
let bufferPoolMissCount = 0;
// WebGPU usage flags — keep in sync with webgpu-bridge-stub.ts.
const WGPU_USAGE_MAP_READ = 0x0001;
const WGPU_USAGE_MAP_WRITE = 0x0002;
function isPoolable(usage: number): boolean {
  // Mapped buffers cannot be reused without unmapping, and their state is
  // fragile — exclude them from the pool.
  return (usage & (WGPU_USAGE_MAP_READ | WGPU_USAGE_MAP_WRITE)) === 0;
}
function poolKey(usage: number, size: number): string {
  return `${usage}:${size}`;
}

function addHandle(obj: any): number {
  const id = nextHandle++;
  handles[id] = obj;
  return id;
}

function getHandle<T>(id: number): T {
  const obj = handles[id];
  if (!obj) throw new Error(`[GPU Worker] Invalid handle: ${id}`);
  return obj as T;
}

function releaseHandle(id: number): void {
  handles[id] = undefined;
  bufferSizesArr[id] = undefined;
  bufferLogicalSizesArr[id] = undefined;
  bufferUsagesArr[id] = undefined;
}

// Pool-aware buffer release — shared by BUFFER_RELEASE and BUFFER_RELEASE_BATCH.
// Never actually destroys GPUBuffers (see comment in BUFFER_RELEASE handler):
// pooling is just handle-table bookkeeping.
function releaseBufferHandle(handle: number): void {
  const size = bufferSizesArr[handle];
  const usage = bufferUsagesArr[handle];
  if (size !== undefined && usage !== undefined && size > 0 && isPoolable(usage)) {
    const key = poolKey(usage, size);
    let stack = bufferPool.get(key);
    if (stack === undefined) {
      stack = [];
      bufferPool.set(key, stack);
    }
    if (stack.length < BUFFER_POOL_CAP_PER_KEY) {
      stack.push(handle);
      // Keep handles[handle] / bufferSizesArr[handle] / bufferUsagesArr[handle]
      // populated so a subsequent pool hit can return this handle with its
      // metadata intact.
      return;
    }
  }
  releaseHandle(handle);
}

// ---------- Memory Helpers (1J: cached WASM DataView) ----------

let cachedWasmBuffer: ArrayBuffer | null = null;
let cachedWasmView: DataView | null = null;
let cachedWasmU8: Uint8Array | null = null;

function getWasmView(): DataView {
  const buf = wasmMemoryObj.buffer;
  if (buf !== cachedWasmBuffer) {
    cachedWasmBuffer = buf;
    cachedWasmView = new DataView(buf);
    cachedWasmU8 = new Uint8Array(buf);
  }
  return cachedWasmView!;
}

function getWasmBytes(): Uint8Array {
  const buf = wasmMemoryObj.buffer;
  if (buf !== cachedWasmBuffer) {
    cachedWasmBuffer = buf;
    cachedWasmView = new DataView(buf);
    cachedWasmU8 = new Uint8Array(buf);
  }
  return cachedWasmU8!;
}

function readString(ptr: number): string {
  if (ptr === 0) return '';
  const bytes = getWasmBytes();
  let end = ptr;
  const maxLen = bytes.byteLength;
  while (end < maxLen && bytes[end] !== 0) end++;
  return new TextDecoder().decode(bytes.slice(ptr, end));
}

// ---------- GPU State ----------

let device: GPUDevice;
let queue: GPUQueue;
let adapter: GPUAdapter;
let hasShaderF16 = false;

// Track pass→encoder association for end+begin after each dispatch.
// WebGPU disallows buffer aliasing within a single compute pass, so each
// dispatch must get its own pass. The C++ backend calls end_compute_pass()
// after every dispatch, but the bridge stub caches the pass for reuse.
// We fix this by ending+restarting the pass in the gpu-worker after every dispatch.
const passEncoderMap = new Map<number, number>(); // passHandle → encoderHandle

function endAndRestartPass(passHandle: number): void {
  const encoderHandle = passEncoderMap.get(passHandle);
  if (encoderHandle === undefined) return;
  const pass = handles[passHandle] as GPUComputePassEncoder;
  pass.end();
  const encoder = handles[encoderHandle] as GPUCommandEncoder;
  const newPass = encoder.beginComputePass();
  handles[passHandle] = newPass; // Replace in-place — bridge stub's cached handle stays valid
}

// Pre-registered handles (set during init)
let instanceHandle: number;
let adapterHandle: number;
let deviceHandle: number;
let queueHandle: number;

// ---------- Shared Memory ----------

let cmdBuffer: SharedArrayBuffer; // Raw SharedArrayBuffer for typed array views into command data
let cmdView: Int32Array;
let cmdDataView: DataView;
let cmdU32: Uint32Array; // Fast unsigned reads (avoids DataView overhead in hot path)
let wasmMemoryObj: WebAssembly.Memory; // Memory object — .buffer always reflects current size after grow()
let readbackView: Uint8Array;

// Phase 2: dispatch batch SAB (optional — only present when the mlx-worker
// passes `dispatchBatchBuffer` in the init message). Used by DISPATCH_BATCH.
let dispatchBatchU32: Uint32Array | null = null;
let dispatchBatchU8: Uint8Array | null = null;
let dispatchBatchBuffer: SharedArrayBuffer | null = null;

// Pending callbacks: accumulated during async operations (mapAsync, adapter/device request).
// Written to the callback ring in the command buffer when the current RPC call completes.
interface PendingCallback {
  fnPtr: number;
  status: number;
  userdataPtr: number;
}
// 1I: Ring buffer for pending callbacks — avoids splice(0, count) overhead
const cbRing: PendingCallback[] = [];
let cbHead = 0;
let cbTail = 0;
function pushCallback(cb: PendingCallback): void {
  cbRing[cbTail++] = cb;
}

// GPU-done callbacks: accumulated from QUEUE_ON_SUBMITTED_WORK_DONE (fn=22).
// These must NOT fire until GPU work actually completes. They are moved to
// pendingCallbacks only during POLL (fn=80) after queue.onSubmittedWorkDone() resolves.
const gpuDoneCallbacks: PendingCallback[] = [];

// Track mapped buffer ranges for the shadow-copy pattern.
// Key = buffer handle, value = { jsRange, offset, size }
// The gpu-worker holds the JS ArrayBuffer; on unmap, it copies WASM->JS (write path).
interface MappedRangeInfo {
  jsRange: ArrayBuffer;
  wasmPtr: number; // WASM pointer allocated by wasm-worker (passed back as RESULT)
  size: number;
  writeBack: boolean;
}
const activeMappings = new Map<number, MappedRangeInfo>();

// ---------- Worker Init ----------

self.onmessage = async (e: MessageEvent) => {
  if (e.data.type === 'init') {
    cmdBuffer = e.data.cmdBuffer;
    wasmMemoryObj = e.data.wasmMemory; // WebAssembly.Memory object
    readbackView = new Uint8Array(e.data.readbackBuffer);

    cmdView = new Int32Array(cmdBuffer);
    cmdDataView = new DataView(cmdBuffer);
    cmdU32 = new Uint32Array(cmdBuffer);

    if (e.data.dispatchBatchBuffer) {
      dispatchBatchBuffer = e.data.dispatchBatchBuffer as SharedArrayBuffer;
      dispatchBatchU32 = new Uint32Array(dispatchBatchBuffer);
      dispatchBatchU8 = new Uint8Array(dispatchBatchBuffer);
    }

    // Create GPU device
    const gpu = navigator.gpu;
    if (!gpu) {
      self.postMessage({ type: 'error', message: 'WebGPU not available in gpu-worker' });
      return;
    }

    const _adapter = await gpu.requestAdapter();
    if (!_adapter) {
      self.postMessage({ type: 'error', message: 'No WebGPU adapter available' });
      return;
    }
    adapter = _adapter;

    // 1A: Runtime feature detection
    hasShaderF16 = adapter.features.has('shader-f16');
    const hasSubgroups = adapter.features.has('subgroups');
    const hasTimestampQuery = adapter.features.has('timestamp-query');
    const requiredFeatures: GPUFeatureName[] = [];
    if (hasShaderF16) requiredFeatures.push('shader-f16');
    if (hasSubgroups) requiredFeatures.push('subgroups');
    if (hasTimestampQuery) requiredFeatures.push('timestamp-query');
    console.log(
      `[GPU Worker] Detected features: shader-f16=${hasShaderF16}, subgroups=${hasSubgroups}, timestamp-query=${hasTimestampQuery}`,
    );

    device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 16),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30),
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
        maxComputeWorkgroupSizeZ: 64,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupsPerDimension: 65535,
        // D=256 tile SDPA needs ~25 KiB shared memory (BQ*D + BK*D + extras).
        // The WebGPU minimum is 16384 (16 KiB); request the adapter's max
        // (32768 on Apple M3) so D=256 kernels can compile and run.
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxBindGroups: 4,
        maxBindingsPerBindGroup: Math.min(adapter.limits.maxBindingsPerBindGroup, 16),
      },
    });
    queue = device.queue;

    device.onuncapturederror = (event) => {
      const error = (event as GPUUncapturedErrorEvent).error;
      console.error('[GPU Worker] Uncaptured error:', error.constructor.name, '-', error.message);
    };

    // Pre-register handles for pre-created objects
    instanceHandle = addHandle({ __brand: 'instance' });
    adapterHandle = addHandle(adapter);
    deviceHandle = addHandle(device);
    queueHandle = addHandle(queue);

    self.postMessage({
      type: 'ready',
      instanceHandle,
      adapterHandle,
      deviceHandle,
      queueHandle,
      features: { shaderF16: hasShaderF16, subgroups: hasSubgroups, timestampQuery: hasTimestampQuery },
    });

    // Start command processing loop
    commandLoop();
  }

  if (e.data.type === 'upload_weights') {
    // Bulk weight upload: read directly from SharedArrayBuffer, create GPU buffers.
    // Zero-copy from shared memory → GPU via mappedAtCreation.
    //
    // Step B: mergedSab holds pre-concatenated linear_attn.in_proj_qkvz /
    // in_proj_ba bytes produced by the mlx-worker before upload. Tensors
    // tagged `fromMerged: true` read their bytes from mergedSab with NO
    // additional dataOffset adjustment (the merged SAB is a bare byte blob;
    // dataOffset is the safetensors data region offset and only applies to
    // weightsSab).
    const { weightsSab, mergedSab, dataOffset, tensors, packBf16 } = e.data as {
      weightsSab: SharedArrayBuffer;
      mergedSab?: SharedArrayBuffer;
      dataOffset: number;
      tensors: Array<{
        name: string;
        byteOffset: number;
        byteSize: number;
        dtype: string;
        shape: number[];
        fromMerged?: boolean;
      }>;
      packBf16?: boolean;
    };

    // Minimum element count for the packed-bf16 path. Matches the threshold
    // used by fromBfloat16Bytes so matmul-consumed weights use the same
    // opt-in policy. Step C extends this with a lower per-norm threshold
    // so 1-D norm weight vectors (hidden_size = 1024 on Qwen3.5-0.8B) also
    // opt into the packed layout and the packed normalization kernel.
    const PACKED_MIN_ELEMENTS = 4096;
    // Norm weights are tiny (hidden_size bf16, typically 512–8192 elements);
    // the packed layout saves trivial DRAM bandwidth (norms are L2-resident
    // on M3) but routes them through the same `_bf16p` kernel variant the
    // rest of the backend uses, which keeps the code paths uniform and
    // avoids a special "bf16 norm weight" exception in the dispatcher.
    // 256 is above any scalar broadcast case and well below the smallest
    // real norm (1024) so the threshold only matters for synthetic tests.
    const NORM_PACKED_MIN_ELEMENTS = 256;
    const usePackBf16 = packBf16 === true;

    // Only tensors consumed by matmul (Step-A GEMV b_transposed=true hot path)
    // are eligible for the packed-bf16 layout. Other ops — notably embedding
    // lookup via `take()` at forward entry, and RMSNorm/LayerNorm elementwise
    // multiplies — still read the buffer as array<f32> / array<bf16> and
    // would produce garbage if fed packed u32 pairs. The Qwen3.5 compiled
    // decode path routes every `*_proj.weight` (q/k/v/o_proj, gate/up/down_proj,
    // in_proj_qkvz, in_proj_ba, out_proj, ...) and `lm_head.weight` through
    // `matmul(x, transpose(weight))` in `linear_proj()`, so those are the
    // only safe names. Embedding (`model.embed_tokens.weight`), norms
    // (`*norm*.weight`), GDN scalars (A_log, dt_bias), and conv1d kernels
    // all stay upconverted. Step B/C will extend this allowlist once the
    // corresponding packed kernels land.
    const isMatmulConsumedWeight = (name: string): boolean => {
      if (!name.endsWith('.weight')) return false;
      // Linear projections — all routed through linear_proj() →
      // matmul(x, transpose(weight)) in mlx_qwen35_common.h. Both decode (GEMV)
      // and prefill (GEMM) of these weights consume the buffer with
      // b_transposed=true, which is the only path that has a packed kernel.
      // MLP (SwiGLU) projections.
      if (name.endsWith('.mlp.gate_proj.weight')) return true;
      if (name.endsWith('.mlp.up_proj.weight')) return true;
      if (name.endsWith('.mlp.down_proj.weight')) return true;
      // Self-attention projections.
      if (name.endsWith('.self_attn.q_proj.weight')) return true;
      if (name.endsWith('.self_attn.k_proj.weight')) return true;
      if (name.endsWith('.self_attn.v_proj.weight')) return true;
      if (name.endsWith('.self_attn.o_proj.weight')) return true;
      // Linear (gated-delta) attention projections. in_proj_qkvz /
      // in_proj_ba are PRE-MERGED in JS (mlx-worker.ts) before upload so
      // the merged buffer takes its OWN PackedBf16 opt-in on first upload
      // — no Rust concat, no storage_mode propagation needed. out_proj is
      // already a single weight. All three land on matmul(x, W.T) via
      // linear_proj() in mlx_qwen35_common.h → b_transposed=true packed
      // kernel.
      if (name.endsWith('.linear_attn.out_proj.weight')) return true;
      if (name.endsWith('.linear_attn.in_proj_qkvz.weight')) return true;
      if (name.endsWith('.linear_attn.in_proj_ba.weight')) return true;
      return false;
    };

    // Step C: norm weights consumed by fast::RMSNorm / fast::LayerNorm.
    // These flow through `make_rmsnorm_kernel` / `make_layernorm_kernel`
    // in backend/webgpu/normalization.cpp which (as of Step C) bind the
    // weight as `array<u32>` and unpack via unpack_bf16_pair when the
    // buffer's storage_mode is PackedBf16. gpu-worker.ts sees the raw
    // safetensors names (pre-rename) so the suffixes match what lands
    // in qwen3_5/persistence.rs:norm_suffixes before prefix stripping
    // and renaming.
    //
    // EXACT allowlist — derived from the Qwen3.5-0.8B VL safetensors
    // header (see packed_bf16_norm_allowlist investigation). Four
    // per-layer text-model norms plus the final LM-head norm are the
    // only RMSNorm weights we pack; vision_tower.* and the per-layer
    // `.linear_attn.norm.weight` MUST be rejected by name (not just by
    // the below-threshold fallback or the bf16 guard — this is the
    // primary safety net).
    //
    // Actual names in the header (prefix-stripped forms shown with the
    // `language_model.model.` prefix that the safetensors loader sees
    // prior to persistence.rs's rename pass):
    //   - language_model.model.layers.{i}.input_layernorm.weight         [1024] bf16
    //   - language_model.model.layers.{i}.post_attention_layernorm.weight [1024] bf16
    //   - language_model.model.layers.{i}.self_attn.q_norm.weight        [ 256] bf16
    //   - language_model.model.layers.{i}.self_attn.k_norm.weight        [ 256] bf16
    //   - language_model.model.norm.weight                               [1024] bf16
    //
    // Rejected norm-like names in the same header:
    //   - language_model.model.layers.{i}.linear_attn.norm.weight       [ 128] bf16
    //       (128 is below NORM_PACKED_MIN_ELEMENTS but the name guard
    //        here is the primary rejection, not the threshold.)
    //   - vision_tower.blocks.{i}.norm1.weight / .norm2.weight          [ 768] bf16
    //   - vision_tower.blocks.{i}.norm1.bias   / .norm2.bias            [ 768] bf16
    //   - vision_tower.merger.norm.weight / .bias                       [ 768] bf16
    //
    // Use exact endsWith tails to rule out any future tensor that might
    // happen to end with `.norm.weight`. The final-norm check uses a
    // dedicated endsWith('.model.norm.weight') so `model.norm.weight`
    // (text-only rename) and `language_model.model.norm.weight` (VL)
    // both match without a loose `.norm.weight` suffix that also
    // accidentally grabs `linear_attn.norm.weight` or future vision
    // norms.
    const isNormConsumedWeight = (name: string): boolean => {
      if (!name.endsWith('.weight')) return false;
      // Reject vision-tower tensors explicitly — belt and suspenders in
      // case a future refactor drops the endsWith-specific check below.
      if (name.startsWith('vision_tower.')) return false;
      // Per-layer pre-attention RMSNorm.
      if (name.endsWith('.input_layernorm.weight')) return true;
      // Per-layer post-attention RMSNorm (feeds the MLP).
      if (name.endsWith('.post_attention_layernorm.weight')) return true;
      // Per-layer full-attention q/k RMSNorm (GQA head normalization).
      if (name.endsWith('.self_attn.q_norm.weight')) return true;
      if (name.endsWith('.self_attn.k_norm.weight')) return true;
      // Final LM-head norm. Matches both the text-only (`model.norm.weight`)
      // and VL (`language_model.model.norm.weight`) names via the shared
      // `.model.norm.weight` tail, without leaking matches to
      // `.linear_attn.norm.weight` or any `vision_tower.*.norm.weight`.
      if (name === 'model.norm.weight') return true;
      if (name.endsWith('.model.norm.weight')) return true;
      return false;
    };

    const handles: number[] = [];
    const uploadedDtypes: string[] = [];
    const packedBf16Flags: boolean[] = [];
    const packedNames: string[] = [];
    const unpackedBf16Names: string[] = [];
    for (const tensor of tensors) {
      const isBf16 = tensor.dtype === 'BF16';
      const isF16 = tensor.dtype === 'F16';
      const numElements = tensor.byteSize / (isBf16 || isF16 ? 2 : 4);
      // Synthetic (pre-merged) tensors live in mergedSab starting at
      // tensor.byteOffset (a raw offset into the merged blob, NOT offset by
      // the safetensors dataOffset). Real safetensors tensors live in
      // weightsSab at dataOffset + tensor.byteOffset.
      const srcSab: SharedArrayBuffer = tensor.fromMerged ? (mergedSab as SharedArrayBuffer) : weightsSab;
      const srcByteOffset = tensor.fromMerged ? tensor.byteOffset : dataOffset + tensor.byteOffset;
      if (tensor.fromMerged && !mergedSab) {
        throw new Error(`Tensor ${tensor.name} marked fromMerged but mergedSab missing`);
      }

      // Packed bf16 path: reinterpret consecutive bf16 pairs as u32 slots.
      // Each slot holds two bf16 elements (low | hi << 16) — exactly the
      // layout the WebGPU backend's upload_with_conversion path produces.
      // Odd element counts pad the trailing slot with zero; the kernel
      // masks out-of-range loads so the pad is harmless.
      //
      // Two separate allowlists:
      //   matmul-consumed weights (large, K >= 4096) → packed GEMV/GEMM
      //   norm-consumed weights (small, hidden_size) → packed RMSNorm/LayerNorm
      // Each has its own min-elements threshold because the motivating
      // kernels have different feasibility floors.
      const matmulConsumed =
        usePackBf16 && isBf16 && numElements >= PACKED_MIN_ELEMENTS && isMatmulConsumedWeight(tensor.name);
      const normConsumed =
        usePackBf16 && isBf16 && numElements >= NORM_PACKED_MIN_ELEMENTS && isNormConsumedWeight(tensor.name);
      const packedEligible = matmulConsumed || normConsumed;
      if (
        isBf16 &&
        usePackBf16 &&
        (numElements >= PACKED_MIN_ELEMENTS ||
          (numElements >= NORM_PACKED_MIN_ELEMENTS && isNormConsumedWeight(tensor.name)))
      ) {
        if (packedEligible) {
          packedNames.push(tensor.name);
        } else {
          unpackedBf16Names.push(tensor.name);
        }
      }
      // bf16 expands to f32 for legacy storage; f16 stays native when shader-f16 is available.
      const needsExpand = (isBf16 && !packedEligible) || (isF16 && !hasShaderF16);

      let gpuByteSize: number;
      if (packedEligible) {
        // Packed: 2 bf16 per u32 = same byte count as the raw bf16 source.
        gpuByteSize = Math.ceil(numElements / 2) * 4;
      } else if (needsExpand) {
        gpuByteSize = numElements * 4;
      } else {
        gpuByteSize = tensor.byteSize;
      }
      const alignedSize = Math.max(4, Math.ceil(gpuByteSize / 4) * 4);

      const gpuBuffer = device.createBuffer({
        size: alignedSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });

      const mapped = gpuBuffer.getMappedRange();
      if (packedEligible) {
        // bf16 is already 2 bytes per element and contiguous in the source
        // buffer; a u32 view of the mapped range stores pairs directly.
        // Note: both the source SAB and the mapped buffer are little-endian
        // wasm targets, so a raw byte copy preserves the (lo, hi) layout
        // that unpack_bf16_pair() expects in WGSL.
        const src = new Uint8Array(srcSab, srcByteOffset, tensor.byteSize);
        const dst = new Uint8Array(mapped, 0, tensor.byteSize);
        dst.set(src);
        // Pad the trailing odd slot with zero bytes (already zero-initialized
        // via mappedAtCreation) — explicit comment for future maintainers.
        uploadedDtypes.push('BF16');
        packedBf16Flags.push(true);
      } else if (needsExpand) {
        // Convert bf16/f16 → f32 in the mapped buffer
        const src16 = new Uint16Array(srcSab, srcByteOffset, numElements);
        const dst32 = new Uint32Array(mapped);
        if (isBf16) {
          // bf16 → f32: shift left by 16 (bf16 is upper 16 bits of f32)
          for (let j = 0; j < numElements; j++) {
            dst32[j] = src16[j] << 16;
          }
        } else {
          // f16 → f32: proper IEEE 754 conversion
          const dstF32 = new Float32Array(mapped);
          for (let j = 0; j < numElements; j++) {
            const h = src16[j];
            const sign = (h >> 15) & 1;
            const exp = (h >> 10) & 0x1f;
            const mant = h & 0x3ff;
            if (exp === 0) {
              dstF32[j] = (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
            } else if (exp === 31) {
              dstF32[j] = mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
            } else {
              dstF32[j] = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
            }
          }
        }
        uploadedDtypes.push('F32');
        packedBf16Flags.push(false);
      } else {
        const mappedU8 = new Uint8Array(mapped);
        const src = new Uint8Array(srcSab, srcByteOffset, tensor.byteSize);
        mappedU8.set(src);
        uploadedDtypes.push(tensor.dtype);
        packedBf16Flags.push(false);
      }
      gpuBuffer.unmap();

      const handle = addHandle(gpuBuffer);
      bufferSizesArr[handle] = alignedSize;
      bufferUsagesArr[handle] = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      handles.push(handle);
    }

    self.postMessage({
      type: 'weights_uploaded',
      handles,
      uploadedDtypes,
      packedBf16Flags,
      debugPackedNames: packedNames,
      debugUnpackedBf16Names: unpackedBf16Names,
      debugPackedTotal: packedNames.length,
      debugUnpackedBf16Total: unpackedBf16Names.length,
    });
  }
};

// ---------- Command Processing Loop ----------

// ---------- Inlined hot-path handlers ----------
// These bypass processCommand entirely: no async overhead, no closure
// allocation, no switch dispatch, no flushCallbacks (hot paths produce
// no callbacks). Called directly from the command loop.

/**
 * Core fused dispatch body shared by single-shot FUSED_FULL_DISPATCH /
 * FUSED_DISPATCH_WITH_UNIFORM and batched DISPATCH_BATCH. Reads entries and
 * (optionally) inline uniform bytes from `entrySrc` + `uniformSrc`, which
 * may alias either the main cmdBuffer or the dedicated batch SAB.
 *
 * Returns the new buffer handle if the record triggered inline deferred
 * buffer creation (FUSED_DISPATCH_WITH_UNIFORM with entries[uniformEntryIdx]=0),
 * or 0 otherwise.
 */
function dispatchFusedRecord(
  opcode: number,
  passHandle: number,
  pipelineHandle: number,
  layoutHandle: number,
  x: number,
  y: number,
  z: number,
  entryCount: number,
  entrySrc: Uint32Array,
  entrySrcBase: number,
  uniformEntryIdx: number,
  uniformDataSize: number,
  uniformBuffer: ArrayBuffer | SharedArrayBuffer | null,
  uniformByteOffset: number,
  uniformUsage: number,
): number {
  const layout = handles[layoutHandle] as GPUBindGroupLayout;
  const entries: GPUBindGroupEntry[] = [];
  let newBufferHandle = 0;

  // Stage 1: optional uniform write (only for FUSED_DISPATCH_WITH_UNIFORM).
  // Must run BEFORE createBindGroup so the patched handle (for deferred
  // creation) lands in entries[uniformEntryIdx].
  if (opcode === RpcFn.FUSED_DISPATCH_WITH_UNIFORM && uniformDataSize > 0 && uniformBuffer !== null) {
    const uniformBufHandle = entrySrc[entrySrcBase + uniformEntryIdx * 3];
    const uniformData = new Uint8Array(uniformBuffer, uniformByteOffset, uniformDataSize);
    if (uniformBufHandle === 0) {
      // Deferred buffer creation (inline): create buffer + write data.
      const usage = uniformUsage || 0x0080 | 0x0008; // STORAGE | COPY_DST fallback
      const poolable = isPoolable(usage);
      // Only bucket for poolable usages (see CREATE_BUFFER_FROM_DATA).
      const physicalSize = poolable ? roundUpBucket(uniformDataSize) : uniformDataSize;
      let handle: number;
      const stack = poolable ? bufferPool.get(poolKey(usage, physicalSize)) : undefined;
      if (stack !== undefined && stack.length > 0) {
        handle = stack.pop()!;
        if (stack.length === 0) bufferPool.delete(poolKey(usage, physicalSize));
        bufferPoolHitCount++;
        const pooled = handles[handle] as GPUBuffer;
        queue.writeBuffer(pooled, 0, uniformData);
        // Update the logical size for this new owner — the previous owner's
        // logical size could differ even though they share a bucket.
        bufferLogicalSizesArr[handle] = uniformDataSize;
      } else {
        if (poolable) bufferPoolMissCount++;
        const buffer = device.createBuffer({ size: physicalSize, usage });
        queue.writeBuffer(buffer, 0, uniformData);
        handle = addHandle(buffer);
        bufferSizesArr[handle] = physicalSize;
        bufferLogicalSizesArr[handle] = uniformDataSize;
        bufferUsagesArr[handle] = usage;
      }
      newBufferHandle = handle;
      // Patch the entry handle so the bind group build below picks up the
      // new GPUBuffer. When `entrySrc === cmdU32` this mutates the main
      // cmdBuffer in place (as the single-dispatch path used to do);
      // otherwise it mutates the batch SAB (harmless — the bridge stub
      // only reuses that SAB from offset 0 on the next batch).
      entrySrc[entrySrcBase + uniformEntryIdx * 3] = handle;
    } else {
      const existing = handles[uniformBufHandle] as GPUBuffer;
      queue.writeBuffer(existing, 0, uniformData);
    }
  }

  // Stage 2: build bind group entries from the inline entry array.
  let eIdx = entrySrcBase;
  for (let i = 0; i < entryCount; i++) {
    const buffer = handles[entrySrc[eIdx]] as GPUBuffer;
    const sizeLo = entrySrc[eIdx + 1];
    const sizeHi = entrySrc[eIdx + 2];
    eIdx += 3;
    const size = sizeLo + sizeHi * 0x100000000;
    const resource: GPUBufferBinding = { buffer, offset: 0 };
    if (size !== 0 && size < 2 ** 53) resource.size = size;
    entries.push({ binding: i, resource });
  }

  // Stage 3: issue the dispatch on the cached compute pass.
  const bindGroup = device.createBindGroup({ layout, entries });
  const pass = handles[passHandle] as GPUComputePassEncoder;
  pass.setPipeline(handles[pipelineHandle] as GPUComputePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y, z);
  endAndRestartPass(passHandle);
  return newBufferHandle;
}

function handleFusedFullDispatch(): void {
  const passHandle = cmdU32[I_ARG0];
  const pipelineHandle = cmdU32[I_ARG0 + 1];
  const layoutHandle = cmdU32[I_ARG0 + 2];
  const x = cmdU32[I_ARG0 + 3];
  const y = cmdU32[I_ARG0 + 4];
  const z = cmdU32[I_ARG0 + 5];
  const entryCount = cmdU32[I_CB_COUNT];
  dispatchFusedRecord(
    RpcFn.FUSED_FULL_DISPATCH,
    passHandle,
    pipelineHandle,
    layoutHandle,
    x,
    y,
    z,
    entryCount,
    cmdU32,
    I_CB_BASE,
    0,
    0,
    null,
    0,
    0,
  );
  cmdU32[I_RESULT] = 0;
}

function handleFusedDispatchWithUniform(): void {
  const passHandle = cmdU32[I_ARG0];
  const pipelineHandle = cmdU32[I_ARG0 + 1];
  const layoutHandle = cmdU32[I_ARG0 + 2];
  const x = cmdU32[I_ARG0 + 3];
  const y = cmdU32[I_ARG0 + 4];
  const z = cmdU32[I_ARG0 + 5];
  const uniformEntryIdx = cmdU32[I_ARG0 + 6];
  const entryCount = cmdU32[I_CB_COUNT];
  const uniformDataSize = cmdU32[I_UNIFORM_SIZE];
  const uniformUsage = cmdU32[I_ARG0 + 7];
  const newBufferHandle = dispatchFusedRecord(
    RpcFn.FUSED_DISPATCH_WITH_UNIFORM,
    passHandle,
    pipelineHandle,
    layoutHandle,
    x,
    y,
    z,
    entryCount,
    cmdU32,
    I_CB_BASE,
    uniformEntryIdx,
    uniformDataSize,
    cmdBuffer,
    CMD_OFFSET.UNIFORM_DATA,
    uniformUsage,
  );
  cmdU32[I_RESULT] = newBufferHandle;
}

/**
 * Phase 2: DISPATCH_BATCH handler. Walks the dispatch batch SAB and issues
 * each packed FUSED_*_DISPATCH record via the shared dispatchFusedRecord
 * helper. The dispatch batch SAB is populated by the bridge stub; see
 * DISPATCH_BATCH in rpc-protocol.ts for the per-record layout.
 */
function handleDispatchBatch(): void {
  if (dispatchBatchU32 === null || dispatchBatchU8 === null || dispatchBatchBuffer === null) {
    cmdU32[I_RESULT] = 0;
    return;
  }
  const batchCount = cmdU32[I_ARG0];
  const batchBytes = cmdU32[I_ARG0 + 1];
  const sab = dispatchBatchU32;
  const sabBuffer = dispatchBatchBuffer;
  let cursor = 0;
  for (let r = 0; r < batchCount; r++) {
    if (cursor >= batchBytes) break;
    const base = cursor >>> 2;
    const opcode = sab[base + 0];
    const passHandle = sab[base + 1];
    const pipelineHandle = sab[base + 2];
    const layoutHandle = sab[base + 3];
    const x = sab[base + 4];
    const y = sab[base + 5];
    const z = sab[base + 6];
    const uniformEntryIdx = sab[base + 7];
    const uniformSize = sab[base + 8];
    const entryCount = sab[base + 9];
    const entryBase = base + (DISPATCH_BATCH_HEADER_BYTES >>> 2);
    const uniformByteOffset = cursor + DISPATCH_BATCH_HEADER_BYTES + entryCount * 12;
    dispatchFusedRecord(
      opcode,
      passHandle,
      pipelineHandle,
      layoutHandle,
      x,
      y,
      z,
      entryCount,
      sab,
      entryBase,
      uniformEntryIdx,
      uniformSize,
      uniformSize > 0 ? sabBuffer : null,
      uniformByteOffset,
      0,
    );
    const uniformPadded = (uniformSize + 3) & ~3;
    cursor += DISPATCH_BATCH_HEADER_BYTES + entryCount * 12 + uniformPadded;
  }
  cmdU32[I_RESULT] = 0;
}

function handleFusedSubmit(): void {
  const encoderHandle = cmdU32[I_ARG0];
  const passHandle = cmdU32[I_ARG0 + 1];
  if (passHandle > 0) {
    (handles[passHandle] as GPUComputePassEncoder).end();
    passEncoderMap.delete(passHandle);
    releaseHandle(passHandle);
  }
  const encoder = handles[encoderHandle] as GPUCommandEncoder;
  const cmdBuf = encoder.finish();
  queue.submit([cmdBuf]);
  releaseHandle(encoderHandle);

  cmdU32[I_RESULT] = 0;
}

function handleFusedCopyBuffer(): void {
  const encoderHandle = cmdU32[I_ARG0];
  const passHandle = cmdU32[I_ARG0 + 1];
  const srcHandle = cmdU32[I_ARG0 + 2];
  const srcOffset = cmdU32[I_ARG0 + 3];
  const dstHandle = cmdU32[I_ARG0 + 4];
  const dstOffset = cmdU32[I_ARG0 + 5];
  const size = cmdU32[I_ARG0 + 6];

  const encoder = handles[encoderHandle] as GPUCommandEncoder;

  if (passHandle !== 0) {
    (handles[passHandle] as GPUComputePassEncoder).end();
    passEncoderMap.delete(passHandle);
    releaseHandle(passHandle);
  }

  encoder.copyBufferToBuffer(
    handles[srcHandle] as GPUBuffer,
    srcOffset,
    handles[dstHandle] as GPUBuffer,
    dstOffset,
    size,
  );

  const newPass = encoder.beginComputePass();
  const newPassHandle = addHandle(newPass);
  passEncoderMap.set(newPassHandle, encoderHandle);
  cmdU32[I_RESULT] = newPassHandle;
}

async function commandLoop(): Promise<void> {
  // Adaptive spin: after dispatch commands (likely followed by another dispatch),
  // spin longer to catch back-to-back commands without V8 event loop overhead.
  // After submit/readback, skip spin (long gap expected).
  const SPIN_DISPATCH = 2000; // optimal: 500→13.3, 1000→14.0, 1500→16.4, 2000→17.5, 4000→17.6
  const SPIN_NONE = 0;
  let spinBudget = SPIN_NONE; // set after each command based on type

  let currentStatus = Atomics.load(cmdView, STATUS_INDEX);

  while (true) {
    // Wait for wasm-worker to set STATUS = PENDING.
    if (currentStatus !== STATUS.PENDING) {
      // Spin-check based on previous command's expected follow-up.
      // Must use Atomics.load (not raw read) — V8 JIT will hoist raw reads
      // out of loops, returning stale cached values from SharedArrayBuffer.
      let spun = false;
      for (let s = 0; s < spinBudget; s++) {
        currentStatus = Atomics.load(cmdView, STATUS_INDEX);
        if (currentStatus === STATUS.PENDING) {
          spun = true;
          break;
        }
      }

      if (!spun) {
        // Fall back to async wait
        let waitLoops = 0;
        while (currentStatus !== STATUS.PENDING) {
          const result = Atomics.waitAsync(cmdView, STATUS_INDEX, currentStatus);
          if (result.async) {
            await result.value;
          } else {
            await new Promise((r) => setTimeout(r, 0));
          }
          currentStatus = Atomics.load(cmdView, STATUS_INDEX);
          waitLoops++;
          if (waitLoops % 1000 === 0) {
            console.log(`[GPU Worker] waiting for PENDING, status=${currentStatus}, loops=${waitLoops}`);
          }
        }
      }
    }

    const fnId = cmdU32[0]; // CMD_OFFSET.FN_ID / 4 = 0

    // Phase 0 observability: bump per-opcode histogram here — BEFORE the
    // fast-path dispatch — so fused-dispatch opcodes (which short-circuit
    // processCommand) are counted too. GET_STATS is excluded so a profile
    // readback does not pollute its own numbers.
    if (fnId !== RpcFn.GET_STATS) {
      gpuTotalRpcs++;
      if (fnId >= 0 && fnId < STATS_OPCODE_SLOTS) {
        gpuRpcCounts[fnId]++;
      }
    }

    // Fast path: hot dispatch functions bypass processCommand entirely.
    if (fnId === RpcFn.FUSED_DISPATCH_WITH_UNIFORM) {
      try {
        handleFusedDispatchWithUniform();
      } catch (err) {
        console.error(`[GPU Worker] Error in FUSED_DISPATCH_WITH_UNIFORM:`, err);
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = SPIN_DISPATCH;
    } else if (fnId === RpcFn.FUSED_FULL_DISPATCH) {
      try {
        handleFusedFullDispatch();
      } catch (err) {
        console.error(`[GPU Worker] Error in FUSED_FULL_DISPATCH:`, err);
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = SPIN_DISPATCH;
    } else if (fnId === RpcFn.DISPATCH_BATCH) {
      try {
        handleDispatchBatch();
      } catch (err) {
        console.error(`[GPU Worker] Error in DISPATCH_BATCH:`, err);
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = SPIN_DISPATCH;
    } else if (fnId === RpcFn.FUSED_COPY_BUFFER) {
      try {
        handleFusedCopyBuffer();
      } catch (err) {
        console.error(`[GPU Worker] Error in FUSED_COPY_BUFFER:`, err);
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = SPIN_DISPATCH; // copy is followed by dispatches
    } else if (fnId === RpcFn.FUSED_SUBMIT) {
      try {
        handleFusedSubmit();
      } catch (err) {
        console.error(`[GPU Worker] Error in FUSED_SUBMIT:`, err);
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = SPIN_NONE;
    } else {
      try {
        await processCommand(fnId);
      } catch (err) {
        console.error(`[GPU Worker] Error processing fn ${fnId}:`, err);
        // Phase 6c diag: forward RPC errors to parent so they surface in
        // test output (browser console isn't piped into the test runner).
        try {
          const msg = err instanceof Error ? err.message : String(err);
          self.postMessage({ type: 'rpc-error', fnId, message: msg });
        } catch {
          // ignore postMessage failures
        }
        cmdU32[I_RESULT] = 0;
      }
      flushCallbacks();
      spinBudget = SPIN_NONE;
    }

    // Signal completion
    Atomics.store(cmdView, STATUS_INDEX, STATUS.DONE);
    Atomics.notify(cmdView, STATUS_INDEX);

    // Pre-check for next command
    currentStatus = Atomics.load(cmdView, STATUS_INDEX);
  }
}

function flushCallbacks(): void {
  const pending = cbTail - cbHead;
  const count = Math.min(pending, MAX_CALLBACKS_PER_CALL);
  cmdU32[I_CB_COUNT] = count;

  for (let i = 0; i < count; i++) {
    const cb = cbRing[cbHead + i];
    const base = CMD_OFFSET.CALLBACK_BASE + i * CALLBACK_ENTRY_SIZE;
    cmdDataView.setUint32(base, cb.fnPtr, true);
    cmdDataView.setUint32(base + 4, cb.status, true);
    cmdDataView.setUint32(base + 8, cb.userdataPtr, true);
    cmdDataView.setUint32(base + 12, 0, true); // pad
  }

  // Advance head; compact when fully drained
  if (count > 0) {
    cbHead += count;
    if (cbHead === cbTail) {
      cbHead = 0;
      cbTail = 0;
      cbRing.length = 0;
    }
  }
}

// ---------- Command Dispatch ----------

// Uint32Array indices for direct reads (element index = byte offset / 4)
const I_ARG0 = CMD_OFFSET.ARG0 >>> 2; // 4
const I_RESULT = CMD_OFFSET.RESULT >>> 2; // 2
const I_RESULT_HI = CMD_OFFSET.RESULT_HI >>> 2; // 3
const I_CB_COUNT = CMD_OFFSET.CALLBACK_COUNT >>> 2;
const I_CB_BASE = CMD_OFFSET.CALLBACK_BASE >>> 2;
const I_UNIFORM_SIZE = CMD_OFFSET.UNIFORM_DATA_SIZE >>> 2;

async function processCommand(fnId: number): Promise<void> {
  // Argument readers via Uint32Array (faster than DataView — no endianness check)
  const arg0 = () => cmdU32[I_ARG0];
  const arg1 = () => cmdU32[I_ARG0 + 1];
  const arg2 = () => cmdU32[I_ARG0 + 2];
  const arg3 = () => cmdU32[I_ARG0 + 3];
  const arg4 = () => cmdU32[I_ARG0 + 4];
  const arg5 = () => cmdU32[I_ARG0 + 5];
  const arg6 = () => cmdU32[I_ARG0 + 6];
  const arg7 = () => cmdU32[I_ARG0 + 7];
  const arg0Hi = () => cmdU32[CMD_OFFSET.ARG0_HI >>> 2];
  const arg1Hi = () => cmdU32[CMD_OFFSET.ARG1_HI >>> 2];
  const arg2Hi = () => cmdU32[CMD_OFFSET.ARG2_HI >>> 2];

  const setResult = (v: number) => {
    cmdU32[I_RESULT] = v;
  };
  const setResultBig = (lo: number, hi: number) => {
    cmdU32[I_RESULT] = lo;
    cmdU32[I_RESULT_HI] = hi;
  };

  // 1J: Use cached DataView/Uint8Array over WASM memory.
  // Cache invalidates automatically when buffer identity changes (memory.grow).
  const wasm = getWasmView;
  const wasmBytes = getWasmBytes;

  // (Histogram bump lives at the top of the dispatch loop so that
  // fast-path opcodes which bypass processCommand are still counted.)

  switch (fnId) {
    // ================================================================
    // Instance (pre-created)
    // ================================================================
    case RpcFn.CREATE_INSTANCE: {
      setResult(instanceHandle);
      break;
    }

    case RpcFn.INSTANCE_REQUEST_ADAPTER: {
      // Args: _instance, _optsPtr, callbackPtr, userdataPtr
      // Adapter is pre-created. Queue callback for wasm-worker to invoke.
      const callbackPtr = arg2();
      const userdataPtr = arg3();
      // Callback signature: (WGPURequestAdapterStatus status, WGPUAdapter adapter, char* message, void* userdata)
      // We push the adapter handle as a "deferred callback with 3 args" but our ring only
      // stores fnPtr + status + userdata. The wasm-worker stub will invoke:
      //   callCallback(fnPtr, 0 /*success*/, adapterHandle, 0 /*null msg*/, userdataPtr)
      // So we store the adapterHandle in the status field (overloaded) and the wasm-worker
      // will reconstruct the 4-arg call. Actually, let's keep it simpler:
      // The callback ring has fnPtr + status + userdata. For request-adapter, "status" is
      // the WGPURequestAdapterStatus (0 = success). The wasm-worker needs the adapter handle.
      // We return adapterHandle as the RESULT so the wasm-worker can read it.
      pushCallback({ fnPtr: callbackPtr, status: 0, userdataPtr });
      setResult(adapterHandle);
      break;
    }

    case RpcFn.INSTANCE_RELEASE: {
      // No-op (instance is pre-created, not ref-counted here)
      setResult(0);
      break;
    }

    // ================================================================
    // Adapter (pre-created)
    // ================================================================
    case RpcFn.ADAPTER_REQUEST_DEVICE: {
      // Args: _adapter, _descPtr, callbackPtr, userdataPtr
      const callbackPtr = arg2();
      const userdataPtr = arg3();
      pushCallback({ fnPtr: callbackPtr, status: 0, userdataPtr });
      setResult(deviceHandle);
      break;
    }

    case RpcFn.ADAPTER_RELEASE: {
      setResult(0);
      break;
    }

    case RpcFn.ADAPTER_GET_PROPERTIES: {
      // Args: _adapter, propsPtr
      const propsPtr = arg1();
      // Zero out 256 bytes at propsPtr in WASM memory
      const view = wasm();
      for (let i = 0; i < 256; i++) view.setUint8(propsPtr + i, 0);
      setResult(0);
      break;
    }

    // ================================================================
    // Device
    // ================================================================
    case RpcFn.DEVICE_CREATE_BUFFER: {
      // Args:
      //   arg0 = descPtr
      //   arg1 = physicalSizeOverride (optional; 0 means "derive from descPtr")
      //          When the bridge-stub already bucketed the request, it passes
      //          bucketSize here so the gpu-worker allocates exactly that much.
      //          This keeps the stub-side pool and gpu-worker-side pool in
      //          agreement on physical sizes — a buffer released from bucket
      //          B has physical >= B, so reuses from bucket B are safe.
      const descPtr = arg0();
      const physicalOverride = arg1();
      const view = wasm();
      // WGPUBufferDescriptor (WASM32, 28 bytes):
      //   0: nextInChain (ptr4)
      //   4: label (ptr4)
      //   8: usage (uint64) -- WGPUBufferUsageFlags
      //  16: size (uint64)
      //  24: mappedAtCreation (uint32/bool)
      const usage = view.getUint32(descPtr + 8, true); // low word suffices
      const sizeLo = view.getUint32(descPtr + 16, true);
      const sizeHi = view.getUint32(descPtr + 20, true);
      const size = sizeLo + sizeHi * 0x100000000;
      const mappedAtCreation = view.getUint32(descPtr + 24, true) !== 0;

      // Pool fast path: reuse a previously released GPUBuffer with the same
      // (usage, bucketSize). Skip mappedAtCreation — we can't redeliver a
      // pre-mapped state to the caller, and the caller will write exactly
      // `size` bytes into the mapped range, so the physical allocation on
      // that path MUST stay at logical `size` (a bucket-sized shadow would
      // cause an overrun on unmap copy-back).
      if (!mappedAtCreation && size > 0 && isPoolable(usage)) {
        // Prefer the stub-supplied override (already bucketed by the stub's
        // pool logic). Fall back to our own bucketing when the stub didn't
        // pass one (older callers, tests, or non-pooled create paths).
        const bucketSize = physicalOverride > 0 ? physicalOverride : roundUpBucket(size);
        const stack = bufferPool.get(poolKey(usage, bucketSize));
        if (stack !== undefined && stack.length > 0) {
          const reused = stack.pop()!;
          if (stack.length === 0) bufferPool.delete(poolKey(usage, bucketSize));
          bufferPoolHitCount++;
          // Update the logical size so a subsequent BUFFER_GET_SIZE on the
          // reused handle returns the NEW caller's logical size, not a
          // stale value from the previous owner.
          bufferLogicalSizesArr[reused] = size;
          setResult(reused);
          break;
        }
        bufferPoolMissCount++;
        // Pool miss: allocate the physical (bucketed) size so this buffer
        // can later rejoin its bucket on release. Every buffer in bucket B
        // has physical size B — that invariant is what makes bucket-reuse
        // correctness-safe for later popped handles.
        try {
          const buffer = device.createBuffer({ size: bucketSize, usage });
          const handle = addHandle(buffer);
          bufferSizesArr[handle] = bucketSize;
          bufferLogicalSizesArr[handle] = size;
          bufferUsagesArr[handle] = usage;
          setResult(handle);
        } catch (e) {
          console.error(
            `[GPU Worker] createBuffer failed: size=${size} bucket=${bucketSize} usage=0x${usage.toString(16)} mapped=false`,
            e,
          );
          setResult(0);
        }
        break;
      }

      // mappedAtCreation OR non-poolable (MAP_READ/MAP_WRITE) OR size<=0:
      // use logical size verbatim. These buffers never enter the pool, so
      // bucketing buys nothing and would break mappedAtCreation semantics
      // (getMappedRange must return exactly `size` bytes).
      try {
        const buffer = device.createBuffer({ size, usage, mappedAtCreation });
        const handle = addHandle(buffer);
        bufferSizesArr[handle] = size;
        bufferLogicalSizesArr[handle] = size;
        bufferUsagesArr[handle] = usage;
        setResult(handle);
      } catch (e) {
        console.error(
          `[GPU Worker] createBuffer failed: size=${size} usage=0x${usage.toString(16)} mapped=${mappedAtCreation}`,
          e,
        );
        setResult(0);
      }
      break;
    }

    case RpcFn.DEVICE_CREATE_SHADER_MODULE: {
      // Args: descPtr
      const descPtr = arg0();
      const view = wasm();
      // WGPUShaderModuleDescriptor (8 bytes):
      //   0: nextInChain (ptr4) -> WGPUShaderModuleWGSLDescriptor
      //   4: label (ptr4)
      const nextInChainPtr = view.getUint32(descPtr, true);
      if (nextInChainPtr === 0) {
        console.error('[GPU Worker] ShaderModule descriptor has no nextInChain');
        setResult(0);
        break;
      }
      // WGPUShaderModuleWGSLDescriptor (12 bytes):
      //   0: chain.next (ptr4)
      //   4: chain.sType (uint32) = 5
      //   8: code (ptr4)
      const codePtr = view.getUint32(nextInChainPtr + 8, true);
      let code = readString(codePtr);
      const module = device.createShaderModule({ code });
      setResult(addHandle(module));
      break;
    }

    case RpcFn.DEVICE_CREATE_COMPUTE_PIPELINE: {
      // Args: descPtr
      const descPtr = arg0();
      const view = wasm();
      // WGPUComputePipelineDescriptor (32 bytes):
      //   0: nextInChain (ptr4)
      //   4: label (ptr4)
      //   8: layout (ptr4) -- 0 = auto
      //  12: compute.nextInChain (ptr4)
      //  16: compute.module (ptr4 = handle)
      //  20: compute.entryPoint (ptr4)
      //  24: compute.constantCount (uint32)
      //  28: compute.constants (ptr4)
      const moduleHandle = view.getUint32(descPtr + 16, true);
      const entryPointPtr = view.getUint32(descPtr + 20, true);
      const module = getHandle<GPUShaderModule>(moduleHandle);
      const entryPoint = entryPointPtr ? readString(entryPointPtr) : 'main';
      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint },
      });
      setResult(addHandle(pipeline));
      break;
    }

    case RpcFn.DEVICE_CREATE_BIND_GROUP: {
      // Args: descPtr
      const descPtr = arg0();
      const view = wasm();
      // WGPUBindGroupDescriptor (20 bytes):
      //   0: nextInChain (ptr4)
      //   4: label (ptr4)
      //   8: layout (ptr4 = handle)
      //  12: entryCount (uint32)
      //  16: entries (ptr4)
      const layoutHandle = view.getUint32(descPtr + 8, true);
      const entryCount = view.getUint32(descPtr + 12, true);
      const entriesPtr = view.getUint32(descPtr + 16, true);

      const layout = getHandle<GPUBindGroupLayout>(layoutHandle);
      const entries: GPUBindGroupEntry[] = [];

      // WGPUBindGroupEntry (40 bytes on WASM32):
      //   0: nextInChain (ptr4)
      //   4: binding (uint32)
      //   8: buffer (ptr4 = handle)
      //  12: padding (4 bytes for uint64 alignment)
      //  16: offset (uint64)
      //  24: size (uint64)
      //  32: sampler (ptr4)
      //  36: textureView (ptr4)
      const ENTRY_SIZE = 40;
      for (let i = 0; i < entryCount; i++) {
        const ePtr = entriesPtr + i * ENTRY_SIZE;
        const binding = view.getUint32(ePtr + 4, true);
        const bufferHandle = view.getUint32(ePtr + 8, true);
        const offsetLo = view.getUint32(ePtr + 16, true);
        const offsetHi = view.getUint32(ePtr + 20, true);
        const offset = offsetLo + offsetHi * 0x100000000;
        const sizeLo = view.getUint32(ePtr + 24, true);
        const sizeHi = view.getUint32(ePtr + 28, true);
        const size = sizeLo + sizeHi * 0x100000000;

        if (bufferHandle !== 0) {
          const resource: GPUBufferBinding = { buffer: getHandle<GPUBuffer>(bufferHandle), offset };
          // size=0 means "whole buffer from offset" in the C API;
          // 0xFFFFFFFFFFFFFFFF (WGPU_WHOLE_SIZE) means the same.
          // In JS WebGPU, we omit size to get that behavior.
          if (size !== 0 && size < 2 ** 53) {
            resource.size = size;
          }
          entries.push({ binding, resource });
        }
      }

      const bindGroup = device.createBindGroup({ layout, entries });
      setResult(addHandle(bindGroup));
      break;
    }

    case RpcFn.DEVICE_CREATE_COMMAND_ENCODER: {
      const encoder = device.createCommandEncoder();
      setResult(addHandle(encoder));
      break;
    }

    case RpcFn.DEVICE_GET_QUEUE: {
      setResult(queueHandle);
      break;
    }

    case RpcFn.DEVICE_GET_LIMITS: {
      // Args: limitsPtr
      const limitsPtr = arg0();
      const view = wasm();
      // WGPUSupportedLimits (WASM32): nextInChain (ptr4) + WGPULimits
      // WGPULimits starts at offset 4 (after nextInChain pointer)
      const L = device.limits;
      const base = limitsPtr + 4;
      let off = 0;
      const w32 = (v: number) => {
        view.setUint32(base + off, v, true);
        off += 4;
      };
      const w64 = (v: number) => {
        view.setUint32(base + off, v, true);
        view.setUint32(base + off + 4, 0, true);
        off += 8;
      };
      w32(L.maxTextureDimension1D);
      w32(L.maxTextureDimension2D);
      w32(L.maxTextureDimension3D);
      w32(L.maxTextureArrayLayers);
      w32(L.maxBindGroups);
      w32(L.maxBindGroupsPlusVertexBuffers ?? 0);
      w32(L.maxBindingsPerBindGroup);
      w32(L.maxDynamicUniformBuffersPerPipelineLayout);
      w32(L.maxDynamicStorageBuffersPerPipelineLayout);
      w32(L.maxSampledTexturesPerShaderStage);
      w32(L.maxSamplersPerShaderStage);
      w32(L.maxStorageBuffersPerShaderStage);
      w32(L.maxStorageTexturesPerShaderStage);
      w32(L.maxUniformBuffersPerShaderStage);
      w64(L.maxUniformBufferBindingSize);
      w64(L.maxStorageBufferBindingSize);
      w32(L.minUniformBufferOffsetAlignment);
      w32(L.minStorageBufferOffsetAlignment);
      w32(L.maxVertexBuffers);
      w64(L.maxBufferSize);
      w32(L.maxVertexAttributes);
      w32(L.maxVertexBufferArrayStride);
      w32(L.maxInterStageShaderComponents);
      w32(L.maxInterStageShaderVariables);
      w32(L.maxColorAttachments);
      w32(L.maxColorAttachmentBytesPerSample ?? 0);
      w32(L.maxComputeWorkgroupStorageSize);
      w32(L.maxComputeInvocationsPerWorkgroup);
      w32(L.maxComputeWorkgroupSizeX);
      w32(L.maxComputeWorkgroupSizeY);
      w32(L.maxComputeWorkgroupSizeZ);
      w32(L.maxComputeWorkgroupsPerDimension);
      setResult(1); // success
      break;
    }

    case RpcFn.DEVICE_SET_ERROR_CALLBACK: {
      // Already set in init. No-op for RPC.
      setResult(0);
      break;
    }

    case RpcFn.DEVICE_SET_LOST_CALLBACK: {
      device.lost.then((info) => {
        console.error('[GPU Worker] Device lost:', info.message);
      });
      setResult(0);
      break;
    }

    case RpcFn.DEVICE_RELEASE: {
      // No-op (device is pre-created)
      setResult(0);
      break;
    }

    // ================================================================
    // Queue
    // ================================================================
    case RpcFn.QUEUE_SUBMIT: {
      // Args: count, cmdBufArrayPtr
      const count = arg0();
      const cmdBufArrayPtr = arg1();
      const view = wasm();
      const commandBuffers: GPUCommandBuffer[] = [];
      for (let i = 0; i < count; i++) {
        const handle = view.getUint32(cmdBufArrayPtr + i * 4, true);
        commandBuffers.push(getHandle<GPUCommandBuffer>(handle));
      }
      queue.submit(commandBuffers);
      setResult(0);
      break;
    }

    case RpcFn.QUEUE_WRITE_BUFFER: {
      // Args: bufferHandle, offsetLo, offsetHi, dataPtr, size
      const bufferHandle = arg0();
      const offsetLo = arg1();
      const offsetHi = arg2();
      const dataPtr = arg3();
      const size = arg4();
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      const offset = offsetLo + offsetHi * 0x100000000;
      // Copy data from WASM memory using DataView (growable SAB safe)
      const data = wasmBytes().slice(dataPtr, dataPtr + size);
      queue.writeBuffer(buffer, offset, data);
      setResult(0);
      break;
    }

    case RpcFn.QUEUE_ON_SUBMITTED_WORK_DONE: {
      // Args: callbackPtr, userdataPtr
      // Store in gpuDoneCallbacks — NOT pendingCallbacks. These callbacks must
      // only fire AFTER queue.onSubmittedWorkDone() resolves during POLL.
      // Firing them immediately (before GPU work completes) causes Worker tasks
      // to execute prematurely, releasing temporaries and notifying the scheduler
      // before the GPU has finished using the data.
      const callbackPtr = arg0();
      const userdataPtr = arg1();
      gpuDoneCallbacks.push({ fnPtr: callbackPtr, status: 0, userdataPtr });
      setResult(0);
      break;
    }

    case RpcFn.QUEUE_RELEASE: {
      setResult(0);
      break;
    }

    // ================================================================
    // Command Encoder
    // ================================================================
    case RpcFn.CMD_ENCODER_BEGIN_COMPUTE_PASS: {
      // Args: encoderHandle, _descPtr
      const encoderHandle = arg0();
      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      const pass = encoder.beginComputePass();
      const ph = addHandle(pass);
      passEncoderMap.set(ph, encoderHandle);
      setResult(ph);
      break;
    }

    case RpcFn.CMD_ENCODER_COPY_BUFFER: {
      // Args: encoderHandle, srcHandle, srcOffsetLo, srcOffsetHi, dstHandle, dstOffsetLo
      // Extended args: dstOffsetHi (ARG6), sizeLo (ARG7), sizeHi (ARG0_HI)
      const encoderHandle = arg0();
      const srcHandle = arg1();
      const srcOffsetLo = arg2();
      const srcOffsetHi = arg3();
      const dstHandle = arg4();
      const dstOffsetLo = arg5();
      const dstOffsetHi = arg6();
      const sizeLo = arg7();
      const sizeHi = arg0Hi();

      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      const src = getHandle<GPUBuffer>(srcHandle);
      const dst = getHandle<GPUBuffer>(dstHandle);
      const srcOffset = srcOffsetLo + srcOffsetHi * 0x100000000;
      const dstOffset = dstOffsetLo + dstOffsetHi * 0x100000000;
      const size = sizeLo + sizeHi * 0x100000000;

      encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
      setResult(0);
      break;
    }

    case RpcFn.CMD_ENCODER_FINISH: {
      // Args: encoderHandle, _descPtr
      const encoderHandle = arg0();
      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      const cmdBuf = encoder.finish();
      setResult(addHandle(cmdBuf));
      break;
    }

    case RpcFn.CMD_ENCODER_RELEASE: {
      const handle = arg0();
      releaseHandle(handle);
      setResult(0);
      break;
    }

    // ================================================================
    // Command Buffer
    // ================================================================
    case RpcFn.CMD_BUFFER_RELEASE: {
      const handle = arg0();
      releaseHandle(handle);
      setResult(0);
      break;
    }

    // ================================================================
    // Compute Pass Encoder
    // ================================================================
    case RpcFn.COMPUTE_PASS_SET_PIPELINE: {
      // Args: passHandle, pipelineHandle
      const passHandle = arg0();
      const pipelineHandle = arg1();
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      const pipeline = getHandle<GPUComputePipeline>(pipelineHandle);
      pass.setPipeline(pipeline);
      setResult(0);
      break;
    }

    case RpcFn.COMPUTE_PASS_SET_BIND_GROUP: {
      // Args: passHandle, groupIndex, bgHandle, dynamicOffsetCount, dynamicOffsetsPtr
      const passHandle = arg0();
      const groupIndex = arg1();
      const bgHandle = arg2();
      const dynamicOffsetCount = arg3();
      const dynamicOffsetsPtr = arg4();

      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      const bindGroup = getHandle<GPUBindGroup>(bgHandle);

      if (dynamicOffsetCount > 0 && dynamicOffsetsPtr !== 0) {
        const view = wasm();
        const offsets: number[] = [];
        for (let i = 0; i < dynamicOffsetCount; i++) {
          offsets.push(view.getUint32(dynamicOffsetsPtr + i * 4, true));
        }
        pass.setBindGroup(groupIndex, bindGroup, offsets);
      } else {
        pass.setBindGroup(groupIndex, bindGroup);
      }
      setResult(0);
      break;
    }

    case RpcFn.COMPUTE_PASS_DISPATCH: {
      // Args: passHandle, x, y, z
      const passHandle = arg0();
      const x = arg1();
      const y = arg2();
      const z = arg3();
      getHandle<GPUComputePassEncoder>(passHandle).dispatchWorkgroups(x, y, z);
      endAndRestartPass(passHandle);
      setResult(0);
      break;
    }

    case RpcFn.COMPUTE_PASS_END: {
      // Args: passHandle
      const passHandle = arg0();
      getHandle<GPUComputePassEncoder>(passHandle).end();
      passEncoderMap.delete(passHandle);
      setResult(0);
      break;
    }

    case RpcFn.COMPUTE_PASS_RELEASE: {
      const handle = arg0();
      passEncoderMap.delete(handle);
      releaseHandle(handle);
      setResult(0);
      break;
    }

    // ================================================================
    // Buffer
    // ================================================================
    case RpcFn.BUFFER_GET_SIZE: {
      // Args: bufferHandle
      // Returns: u64 size via RESULT + RESULT_HI
      //
      // MUST return the LOGICAL size (as originally requested by the C++
      // caller), NOT the bucketed/physical allocation size. C++ code does
      // `elems = size / sizeof(dtype)` to iterate the buffer — returning
      // bucket size would cause it to read past the initialized range
      // into the bucket's zero-padded tail (correctness bug).
      const bufferHandle = arg0();
      const size = bufferLogicalSizesArr[bufferHandle] ?? bufferSizesArr[bufferHandle] ?? 0;
      setResultBig(size & 0xffffffff, Math.floor(size / 0x100000000));
      break;
    }

    case RpcFn.BUFFER_GET_MAPPED_RANGE: {
      // Args: bufferHandle, offset, size, wasmPtr
      //
      // Pattern: wasm-worker calls wasmMalloc(size), passes the resulting pointer
      // as wasmPtr. gpu-worker calls getMappedRange, stores the mapping, returns
      // wasmPtr back so C code can use it. On unmap, gpu-worker copies WASM->JS.
      //
      // For mappedAtCreation (write path): C writes to wasmPtr, we copy to jsRange on unmap.
      // For mapAsync read path: handled by BUFFER_GET_CONST_MAPPED_RANGE.
      const bufferHandle = arg0();
      const offset = arg1();
      let size = arg2();
      const wasmPtr = arg3();

      const buffer = getHandle<GPUBuffer>(bufferHandle);
      if (size === 0) size = (bufferSizesArr[bufferHandle] ?? 0) - offset;

      const jsRange = buffer.getMappedRange(offset, size);
      activeMappings.set(bufferHandle, { jsRange, wasmPtr, size, writeBack: true });

      setResult(wasmPtr);
      break;
    }

    case RpcFn.BUFFER_GET_CONST_MAPPED_RANGE: {
      // Args: bufferHandle, offset, size, wasmPtr
      //
      // Read path: copies GPU data into WASM memory immediately.
      const bufferHandle = arg0();
      const offset = arg1();
      let size = arg2();
      const wasmPtr = arg3();

      const buffer = getHandle<GPUBuffer>(bufferHandle);
      if (size === 0) size = (bufferSizesArr[bufferHandle] ?? 0) - offset;

      const jsRange = buffer.getMappedRange(offset, size);

      // Copy GPU data into the dedicated readback buffer (NOT wasmMemory).
      // The wasm-worker will copy from readbackBuffer to its WASM heap.
      // This avoids the growable SharedArrayBuffer issue where byteLength
      // on this worker doesn't reflect wasm-worker's memory growth.
      const src = new Uint8Array(jsRange);
      if (readbackView && size <= readbackView.byteLength) {
        readbackView.set(src, 0);
      }

      activeMappings.set(bufferHandle, { jsRange, wasmPtr, size, writeBack: false });

      setResult(wasmPtr);
      break;
    }

    case RpcFn.BUFFER_UNMAP: {
      // Args: bufferHandle
      const bufferHandle = arg0();
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      const mapping = activeMappings.get(bufferHandle);

      if (mapping?.writeBack) {
        // Write path: copy from WASM shadow into JS ArrayBuffer before unmap
        const src = wasmBytes().slice(mapping.wasmPtr, mapping.wasmPtr + mapping.size);
        const dst = new Uint8Array(mapping.jsRange);
        dst.set(src);
      }
      if (mapping) activeMappings.delete(bufferHandle);
      // Note: wasm-worker is responsible for calling wasmFree(wasmPtr)

      buffer.unmap();
      setResult(0);
      break;
    }

    case RpcFn.BUFFER_MAP_ASYNC: {
      // Args: bufferHandle, mode, offset, size, callbackPtr, userdataPtr
      //
      // This is async -- the event loop is free here so mapAsync resolves.
      // The callback is queued for the wasm-worker to invoke after Atomics.wait returns.
      const bufferHandle = arg0();
      const mode = arg1();
      const offset = arg2();
      const size = arg3();
      const callbackPtr = arg4();
      const userdataPtr = arg5();

      const buffer = getHandle<GPUBuffer>(bufferHandle);
      const gpuMode = mode === 1 ? GPUMapMode.READ : GPUMapMode.WRITE;

      try {
        await buffer.mapAsync(gpuMode, offset, size);
        pushCallback({ fnPtr: callbackPtr, status: 0, userdataPtr });
      } catch (err) {
        console.error('[GPU Worker] mapAsync failed:', err);
        pushCallback({ fnPtr: callbackPtr, status: 1, userdataPtr });
      }
      setResult(0);
      break;
    }

    case RpcFn.BUFFER_DESTROY: {
      // NO-OP: Don't actually destroy buffers. The old uniform buffer pattern
      // destroys buffers before queue.submit, causing "Buffer used in submit
      // while destroyed" validation errors. Buffers are cleaned up on release
      // (handle table removal) and eventually GC'd.
      setResult(0);
      break;
    }

    case RpcFn.BUFFER_RELEASE: {
      // Pool fast path: park the GPUBuffer for reuse instead of releasing.
      // Safe because gpu-worker never actually destroys GPUBuffers on release
      // (destroying them before queue.submit completes triggers "Buffer used
      // in submit while destroyed" validation errors) — so pooling is just
      // handle-table bookkeeping.
      releaseBufferHandle(arg0());
      setResult(0);
      break;
    }

    case RpcFn.BUFFER_RELEASE_BATCH: {
      // Phase 1' batch release: up to MAX_RELEASE_BATCH (64) handles packed
      // as u32s into the UNIFORM_DATA region by the bridge stub. ARG0 holds
      // the count. Each handle flows through the same pool logic as the
      // single-release path.
      const count = arg0();
      const base = CMD_OFFSET.UNIFORM_DATA >>> 2;
      for (let i = 0; i < count; i++) {
        releaseBufferHandle(cmdU32[base + i]);
      }
      setResult(0);
      break;
    }

    // ================================================================
    // Pipeline
    // ================================================================
    case RpcFn.PIPELINE_GET_BIND_GROUP_LAYOUT: {
      // Args: pipelineHandle, index
      const pipelineHandle = arg0();
      const index = arg1();
      const pipeline = getHandle<GPUComputePipeline>(pipelineHandle);
      const layout = pipeline.getBindGroupLayout(index);
      setResult(addHandle(layout));
      break;
    }

    case RpcFn.PIPELINE_RELEASE: {
      const handle = arg0();
      releaseHandle(handle);
      setResult(0);
      break;
    }

    // ================================================================
    // Release (bind group, bind group layout, shader module)
    // ================================================================
    case RpcFn.BIND_GROUP_RELEASE: {
      const handle = arg0();
      releaseHandle(handle);
      setResult(0);
      break;
    }

    case RpcFn.BIND_GROUP_LAYOUT_RELEASE: {
      const handle = arg0();
      releaseHandle(handle);
      setResult(0);
      break;
    }

    case RpcFn.SHADER_MODULE_RELEASE: {
      const handle = arg0();
      releaseHandle(handle);
      setResult(0);
      break;
    }

    // ================================================================
    // Polling -- THE KEY FUNCTION
    // ================================================================
    case RpcFn.POLL: {
      // Wait for all submitted GPU work to complete.
      // This is the whole reason for the two-worker architecture:
      // the event loop is free here, so onSubmittedWorkDone() resolves.
      await queue.onSubmittedWorkDone();
      // NOW it's safe to fire GPU-done callbacks. Move them to the ring
      // so flushCallbacks() writes them to the callback ring for the wasm-worker.
      if (gpuDoneCallbacks.length > 0) {
        for (let i = 0; i < gpuDoneCallbacks.length; i++) {
          pushCallback(gpuDoneCallbacks[i]);
        }
        gpuDoneCallbacks.length = 0;
      }
      setResult(cbTail - cbHead);
      break;
    }

    // ================================================================
    // Special: register externally-created GPU buffer
    // ================================================================
    case RpcFn.FUSED_DISPATCH: {
      // Fused: setPipeline + setBindGroup(0) + dispatch in one RPC call
      // Args: passHandle, pipelineHandle, bindGroupHandle, x, y, z
      const passHandle = arg0();
      const pipelineHandle = arg1();
      const bgHandle = arg2();
      const x = arg3();
      const y = arg4();
      const z = arg5();
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      pass.setPipeline(getHandle<GPUComputePipeline>(pipelineHandle));
      pass.setBindGroup(0, getHandle<GPUBindGroup>(bgHandle));
      pass.dispatchWorkgroups(x, y, z);
      endAndRestartPass(passHandle);
      setResult(0);
      break;
    }

    case RpcFn.FUSED_DISPATCH_2BG: {
      // Fused: setPipeline + setBindGroup(0,1) + dispatch in one RPC call
      // Args: passHandle, pipelineHandle, bg0Handle, bg1Handle, x, y, z
      const passHandle = arg0();
      const pipelineHandle = arg1();
      const bg0Handle = arg2();
      const bg1Handle = arg3();
      const x = arg4();
      const y = arg5();
      const z = arg6();
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      pass.setPipeline(getHandle<GPUComputePipeline>(pipelineHandle));
      pass.setBindGroup(0, getHandle<GPUBindGroup>(bg0Handle));
      pass.setBindGroup(1, getHandle<GPUBindGroup>(bg1Handle));
      pass.dispatchWorkgroups(x, y, z);
      endAndRestartPass(passHandle);
      setResult(0);
      break;
    }

    case RpcFn.FUSED_SUBMIT: {
      // Fused: [pass_end + pass_release +] finish + submit + release
      // Replaces up to 6 separate RPCs with 1. Args: encoderHandle, passHandle (0=no pass)
      const encoderHandle = arg0();
      const passHandle = arg1();
      if (passHandle > 0) {
        getHandle<GPUComputePassEncoder>(passHandle).end();
        passEncoderMap.delete(passHandle);
        releaseHandle(passHandle);
      }
      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      const cmdBuf = encoder.finish();
      queue.submit([cmdBuf]);
      releaseHandle(encoderHandle);
      setResult(0);
      break;
    }

    case RpcFn.FUSED_FULL_DISPATCH: {
      // Fused: inline createBindGroup + setPipeline + setBindGroup(0) + dispatch
      // Direct cmdU32 reads for hot-path performance (no DataView/closure overhead)
      const passHandle = cmdU32[I_ARG0];
      const pipelineHandle = cmdU32[I_ARG0 + 1];
      const layoutHandle = cmdU32[I_ARG0 + 2];
      const x = cmdU32[I_ARG0 + 3];
      const y = cmdU32[I_ARG0 + 4];
      const z = cmdU32[I_ARG0 + 5];
      const entryCount = cmdU32[I_CB_COUNT];

      const layout = getHandle<GPUBindGroupLayout>(layoutHandle);
      const entries: GPUBindGroupEntry[] = [];
      let eIdx = I_CB_BASE; // Uint32Array index for entry data (stride = 3 u32s = 12 bytes)
      for (let i = 0; i < entryCount; i++) {
        const buffer = getHandle<GPUBuffer>(cmdU32[eIdx]);
        const sizeLo = cmdU32[eIdx + 1];
        const sizeHi = cmdU32[eIdx + 2];
        eIdx += 3;
        const size = sizeLo + sizeHi * 0x100000000;
        const resource: GPUBufferBinding = { buffer, offset: 0 };
        if (size !== 0 && size < 2 ** 53) resource.size = size;
        entries.push({ binding: i, resource });
      }

      const bindGroup = device.createBindGroup({ layout, entries });
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      pass.setPipeline(getHandle<GPUComputePipeline>(pipelineHandle));
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(x, y, z);
      endAndRestartPass(passHandle);
      cmdU32[I_RESULT] = 0;
      break;
    }

    case RpcFn.FUSED_DISPATCH_WITH_UNIFORM: {
      // Like FUSED_FULL_DISPATCH but also writes inline uniform data to one buffer.
      // Direct cmdU32 reads for hot-path performance.
      const passHandle = cmdU32[I_ARG0];
      const pipelineHandle = cmdU32[I_ARG0 + 1];
      const layoutHandle = cmdU32[I_ARG0 + 2];
      const x = cmdU32[I_ARG0 + 3];
      const y = cmdU32[I_ARG0 + 4];
      const z = cmdU32[I_ARG0 + 5];
      const uniformEntryIdx = cmdU32[I_ARG0 + 6];
      const entryCount = cmdU32[I_CB_COUNT];
      const uniformDataSize = cmdU32[I_UNIFORM_SIZE];

      const layout = getHandle<GPUBindGroupLayout>(layoutHandle);
      const entries: GPUBindGroupEntry[] = [];

      // Write uniform data BEFORE the entry loop (avoids per-iteration conditional)
      if (uniformDataSize > 0) {
        const uniformBufHandle = cmdU32[I_CB_BASE + uniformEntryIdx * 3];
        const uniformBuffer = getHandle<GPUBuffer>(uniformBufHandle);
        const uniformData = new Uint8Array(cmdBuffer, CMD_OFFSET.UNIFORM_DATA, uniformDataSize);
        queue.writeBuffer(uniformBuffer, 0, uniformData);
      }

      let eIdx = I_CB_BASE;
      for (let i = 0; i < entryCount; i++) {
        const buffer = getHandle<GPUBuffer>(cmdU32[eIdx]);
        const sizeLo = cmdU32[eIdx + 1];
        const sizeHi = cmdU32[eIdx + 2];
        eIdx += 3;
        const size = sizeLo + sizeHi * 0x100000000;
        const resource: GPUBufferBinding = { buffer, offset: 0 };
        if (size !== 0 && size < 2 ** 53) resource.size = size;
        entries.push({ binding: i, resource });
      }

      const bindGroup = device.createBindGroup({ layout, entries });
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      pass.setPipeline(getHandle<GPUComputePipeline>(pipelineHandle));
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(x, y, z);
      endAndRestartPass(passHandle);
      cmdU32[I_RESULT] = 0;
      break;
    }

    case RpcFn.FUSED_COPY_BUFFER: {
      // Fused: end compute pass (if active) + copyBufferToBuffer + begin new compute pass
      // This saves 2-3 RPC round-trips per buffer copy during decode.
      // Args: encoderHandle, passHandle (0=no pass), srcHandle, srcOffset, dstHandle, dstOffset, size
      // Returns: new compute pass handle
      const encoderHandle = arg0();
      const passHandle = arg1();
      const srcHandle = arg2();
      const srcOffset = arg3();
      const dstHandle = arg4();
      const dstOffset = arg5();
      const size = arg6();

      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);

      // End active compute pass if provided
      if (passHandle !== 0) {
        const pass = getHandle<GPUComputePassEncoder>(passHandle);
        pass.end();
        releaseHandle(passHandle);
      }

      // Perform the copy
      const src = getHandle<GPUBuffer>(srcHandle);
      const dst = getHandle<GPUBuffer>(dstHandle);
      encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);

      // Begin new compute pass for subsequent dispatches
      const newPass = encoder.beginComputePass();
      const newPassHandle = addHandle(newPass);
      setResult(newPassHandle);
      break;
    }

    case RpcFn.ADD_GPU_BUFFER: {
      // This is handled via postMessage, not RPC, because the GPU buffer
      // object can't be serialized through SharedArrayBuffer.
      // This case should not be reached; it exists for protocol completeness.
      console.warn('[GPU Worker] ADD_GPU_BUFFER via RPC is not supported. Use postMessage.');
      setResult(0);
      break;
    }

    case RpcFn.CREATE_BUFFER_FROM_DATA: {
      // Fused: createBuffer + writeBuffer in one RPC (replaces mappedAtCreation pattern).
      // Args: usage, sizeLo, sizeHi, wasmDataPtr
      const usage = arg0();
      const sizeLo = arg1();
      const sizeHi = arg2();
      const wasmDataPtr = arg3();
      const size = sizeLo + sizeHi * 0x100000000;

      try {
        let handle: number;
        const poolable = isPoolable(usage);
        // Only bucket for poolable usages — non-poolable buffers never
        // re-enter the pool, so bucketing would just waste memory with no
        // reuse payoff.
        const physicalSize = poolable ? roundUpBucket(size) : size;
        const stack = poolable ? bufferPool.get(poolKey(usage, physicalSize)) : undefined;
        if (stack !== undefined && stack.length > 0) {
          handle = stack.pop()!;
          if (stack.length === 0) bufferPool.delete(poolKey(usage, physicalSize));
          bufferPoolHitCount++;
          // Overwrite the pooled buffer's contents with the new data (only
          // the first `size` bytes; the tail of physicalSize is don't-care
          // padding — no read path is allowed to touch beyond logical size).
          const buffer = getHandle<GPUBuffer>(handle);
          const data = wasmBytes().slice(wasmDataPtr, wasmDataPtr + size);
          queue.writeBuffer(buffer, 0, data);
          // Update the logical size for the new owner. Previous owner may
          // have had a smaller logical size within the same bucket.
          bufferLogicalSizesArr[handle] = size;
        } else {
          if (poolable) bufferPoolMissCount++;
          // Allocate physicalSize so poolable buffers can be bucket-reused
          // on release. The data write below copies only logical `size`
          // bytes — any tail padding stays as the initial zero contents.
          const buffer = device.createBuffer({ size: physicalSize, usage });
          const data = wasmBytes().slice(wasmDataPtr, wasmDataPtr + size);
          queue.writeBuffer(buffer, 0, data);
          handle = addHandle(buffer);
          bufferSizesArr[handle] = physicalSize;
          bufferLogicalSizesArr[handle] = size;
          bufferUsagesArr[handle] = usage;
        }
        setResult(handle);
      } catch (e) {
        console.error(`[GPU Worker] CREATE_BUFFER_FROM_DATA failed: size=${size} usage=0x${usage.toString(16)}`, e);
        setResult(0);
      }
      break;
    }

    // ================================================================
    // Phase 0 stats (?profile=1 observability). Write the per-opcode
    // histogram into the SAB so the wasm-worker can post it up to the
    // main thread. Storage layout (see rpc-protocol.ts for the rationale):
    //   - cmdU32[I_RESULT]  := gpuTotalRpcs (u32, since last reset)
    //   - SAB bytes 64..64+STATS_CALLBACK_SLOTS*4:
    //       opcode counts for slots 0..30
    //   - SAB bytes STATS_INLINE_OFFSET..+STATS_INLINE_SLOTS*4:
    //       opcode counts for slots 31..94
    //   - SAB bytes STATS_RESERVED_OFFSET..+STATS_RESERVED_SLOTS*4:
    //       opcode counts for slots 95..110 (covers FUSED_*_DISPATCH=96/97/98)
    //
    // ARG0 = reset flag: 1 = zero the histogram after readback.
    // ================================================================
    case RpcFn.GET_STATS: {
      const resetAfter = arg0();
      // Smuggle pool hit/miss into two unused reserved-region slots so the
      // bridge's existing histogram readback picks them up with no protocol
      // change. Slot 100 = hits, slot 101 = misses. Neither corresponds to
      // a real RpcFn opcode (max opcode is 99 = GET_STATS).
      const POOL_HITS_SLOT = 100;
      const POOL_MISSES_SLOT = 101;
      gpuRpcCounts[POOL_HITS_SLOT] = bufferPoolHitCount;
      gpuRpcCounts[POOL_MISSES_SLOT] = bufferPoolMissCount;
      const cbBase = CMD_OFFSET.CALLBACK_COUNT >>> 2;
      for (let i = 0; i < STATS_CALLBACK_SLOTS; i++) {
        cmdU32[cbBase + i] = gpuRpcCounts[i];
      }
      const inlineBase = STATS_INLINE_OFFSET >>> 2;
      for (let i = 0; i < STATS_INLINE_SLOTS; i++) {
        cmdU32[inlineBase + i] = gpuRpcCounts[STATS_CALLBACK_SLOTS + i];
      }
      const reservedBase = STATS_RESERVED_OFFSET >>> 2;
      for (let i = 0; i < STATS_RESERVED_SLOTS; i++) {
        cmdU32[reservedBase + i] = gpuRpcCounts[STATS_CALLBACK_SLOTS + STATS_INLINE_SLOTS + i];
      }
      setResult(gpuTotalRpcs);
      if (resetAfter) {
        gpuRpcCounts.fill(0);
        gpuTotalRpcs = 0;
        bufferPoolHitCount = 0;
        bufferPoolMissCount = 0;
      }
      break;
    }

    default: {
      console.warn(`[GPU Worker] Unknown function ID: ${fnId}`);
      setResult(0);
    }
  }
}
