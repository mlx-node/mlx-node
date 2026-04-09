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

import {
  RpcFn,
  CMD_OFFSET,
  STATUS,
  STATUS_INDEX,
  MAX_CALLBACKS_PER_CALL,
  CALLBACK_ENTRY_SIZE,
} from './rpc-protocol.js';

// ---------- Handle Table (1H: sparse array for O(1) index lookup) ----------

const handles: any[] = [null]; // index 0 unused — handles start at 1
const bufferSizesArr: (number | undefined)[] = [undefined];
let nextHandle = 1;

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

let cmdBuffer: SharedArrayBuffer;  // Raw SharedArrayBuffer for typed array views into command data
let cmdView: Int32Array;
let cmdDataView: DataView;
let cmdU32: Uint32Array;  // Fast unsigned reads (avoids DataView overhead in hot path)
let wasmMemoryObj: WebAssembly.Memory;  // Memory object — .buffer always reflects current size after grow()
let readbackView: Uint8Array;

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
function pushCallback(cb: PendingCallback): void { cbRing[cbTail++] = cb; }

// GPU-done callbacks: accumulated from QUEUE_ON_SUBMITTED_WORK_DONE (fn=22).
// These must NOT fire until GPU work actually completes. They are moved to
// pendingCallbacks only during POLL (fn=80) after queue.onSubmittedWorkDone() resolves.
const gpuDoneCallbacks: PendingCallback[] = [];

// Track mapped buffer ranges for the shadow-copy pattern.
// Key = buffer handle, value = { jsRange, offset, size }
// The gpu-worker holds the JS ArrayBuffer; on unmap, it copies WASM->JS (write path).
interface MappedRangeInfo {
  jsRange: ArrayBuffer;
  wasmPtr: number;   // WASM pointer allocated by wasm-worker (passed back as RESULT)
  size: number;
  writeBack: boolean;
}
const activeMappings = new Map<number, MappedRangeInfo>();

// ---------- Worker Init ----------

self.onmessage = async (e: MessageEvent) => {
  if (e.data.type === 'init') {
    cmdBuffer = e.data.cmdBuffer;
    wasmMemoryObj = e.data.wasmMemory;  // WebAssembly.Memory object
    readbackView = new Uint8Array(e.data.readbackBuffer);

    cmdView = new Int32Array(cmdBuffer);
    cmdDataView = new DataView(cmdBuffer);
    cmdU32 = new Uint32Array(cmdBuffer);

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
    console.log(`[GPU Worker] Detected features: shader-f16=${hasShaderF16}, subgroups=${hasSubgroups}, timestamp-query=${hasTimestampQuery}`);

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
    const { weightsSab, dataOffset, tensors } = e.data as {
      weightsSab: SharedArrayBuffer;
      dataOffset: number;
      tensors: Array<{ name: string; byteOffset: number; byteSize: number; dtype: string; shape: number[] }>;
    };

    const handles: number[] = [];
    const uploadedDtypes: string[] = [];
    for (const tensor of tensors) {
      const isBf16 = tensor.dtype === 'BF16';
      const isF16 = tensor.dtype === 'F16';
      // bf16 always expands to f32 — WGSL has no bf16 type.
      // f16 stays native when shader-f16 is available (halves upload bandwidth);
      // otherwise expands to f32.
      const needsExpand = isBf16 || (isF16 && !hasShaderF16);
      const numElements = tensor.byteSize / (isBf16 || isF16 ? 2 : 4);
      const gpuByteSize = needsExpand ? numElements * 4 : tensor.byteSize;
      const alignedSize = Math.max(4, Math.ceil(gpuByteSize / 4) * 4);

      const gpuBuffer = device.createBuffer({
        size: alignedSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });

      const mapped = gpuBuffer.getMappedRange();
      if (needsExpand) {
        // Convert bf16/f16 → f32 in the mapped buffer
        const src16 = new Uint16Array(weightsSab, dataOffset + tensor.byteOffset, numElements);
        const dst32 = new Uint32Array(mapped);
        if (isBf16) {
          // bf16 → f32: shift left by 16 (bf16 is upper 16 bits of f32)
          for (let j = 0; j < numElements; j++) {
            dst32[j] = src16[j] << 16;
          }
        } else {
          // f16 → f32: proper IEEE 754 conversion
          const dstF32 = new Float32Array(mapped);
          const tmpU16 = new Uint16Array(1);
          const tmpBuf = new ArrayBuffer(4);
          const tmpU32 = new Uint32Array(tmpBuf);
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
      } else {
        const mappedU8 = new Uint8Array(mapped);
        const src = new Uint8Array(weightsSab, dataOffset + tensor.byteOffset, tensor.byteSize);
        mappedU8.set(src);
        uploadedDtypes.push(tensor.dtype);
      }
      gpuBuffer.unmap();

      const handle = addHandle(gpuBuffer);
      bufferSizesArr[handle] = alignedSize;
      handles.push(handle);
    }

    self.postMessage({ type: 'weights_uploaded', handles, uploadedDtypes });
  }
};

// ---------- Command Processing Loop ----------

// ---------- Standalone RMSNorm GPU test (bypasses C++/WASM pipeline) ----------
async function runRMSNormTest(wgslCode: string): Promise<void> {
  try {
    console.log('[RMSNorm-TEST] Starting standalone GPU test...');

    const entryMatch = wgslCode.match(/fn\s+(rmsnorm_\w+)\s*\(/);
    if (!entryMatch) {
      console.error('[RMSNorm-TEST] Could not find entry point in WGSL');
      return;
    }
    const entryPoint = entryMatch[1];
    console.log(`[RMSNorm-TEST] Entry point: ${entryPoint}`);

    const axisSize = 4;
    const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const weightData = new Float32Array([1.0, 1.0, 1.0, 1.0]);

    // Expected: sum_sq=30, mean_sq=7.5, inv_rms≈0.365148, output≈[0.3651, 0.7303, 1.0954, 1.4606]
    const sumSq = inputData.reduce((s, v) => s + v * v, 0);
    const invRms = 1.0 / Math.sqrt(sumSq / axisSize + 1e-6);
    const expected = Array.from(inputData).map((v, i) => v * invRms * weightData[i]);
    console.log(`[RMSNorm-TEST] Expected: [${expected.map(v => v.toFixed(6)).join(', ')}]`);

    // Create GPU buffers
    const inputBuf = device.createBuffer({
      size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const weightBuf = device.createBuffer({
      size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuf = device.createBuffer({
      size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const stagingBuf = device.createBuffer({
      size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const paramsBuf = device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Pack uniform params: vec4<u32> = [axis_size, w_stride, eps_as_bits, pad]
    const paramsData = new ArrayBuffer(16);
    const pv = new DataView(paramsData);
    pv.setUint32(0, axisSize, true);
    pv.setUint32(4, 1, true);         // w_stride = 1
    pv.setFloat32(8, 1e-6, true);     // eps — bitcast<f32> in shader recovers float
    pv.setUint32(12, 0, true);        // pad

    queue.writeBuffer(inputBuf, 0, inputData);
    queue.writeBuffer(weightBuf, 0, weightData);
    queue.writeBuffer(paramsBuf, 0, new Uint8Array(paramsData));

    const module = device.createShaderModule({ code: wgslCode });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint },
    });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: weightBuf } },
        { binding: 2, resource: { buffer: outputBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);  // 1 row = 1 workgroup
    pass.end();
    enc.copyBufferToBuffer(outputBuf, 0, stagingBuf, 0, 16);
    queue.submit([enc.finish()]);

    await stagingBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuf.getMappedRange().slice(0));
    stagingBuf.unmap();

    console.log(`[RMSNorm-TEST] Actual:   [${Array.from(result).map(v => v.toFixed(6)).join(', ')}]`);
    console.log(`[RMSNorm-TEST] Expected: [${expected.map(v => v.toFixed(6)).join(', ')}]`);

    let maxErr = 0;
    for (let i = 0; i < axisSize; i++) {
      maxErr = Math.max(maxErr, Math.abs(result[i] - expected[i]));
    }
    console.log(`[RMSNorm-TEST] Max error: ${maxErr.toExponential(4)}`);

    if (maxErr < 1e-4) {
      console.log('[RMSNorm-TEST] PASS — Shader correct. Bug is in C++/WASM dispatch.');
    } else {
      console.log('[RMSNorm-TEST] FAIL — Shader produces wrong results!');
    }

    inputBuf.destroy();
    weightBuf.destroy();
    outputBuf.destroy();
    stagingBuf.destroy();
    paramsBuf.destroy();
  } catch (e) {
    console.error('[RMSNorm-TEST] Error:', e);
  }
}

// ---------- Inlined hot-path handlers ----------
// These bypass processCommand entirely: no async overhead, no closure
// allocation, no switch dispatch, no flushCallbacks (hot paths produce
// no callbacks). Called directly from the command loop.

function handleFusedFullDispatch(): void {
  const passHandle = cmdU32[I_ARG0];
  const pipelineHandle = cmdU32[I_ARG0 + 1];
  const layoutHandle = cmdU32[I_ARG0 + 2];
  const x = cmdU32[I_ARG0 + 3];
  const y = cmdU32[I_ARG0 + 4];
  const z = cmdU32[I_ARG0 + 5];
  const entryCount = cmdU32[I_CB_COUNT];

  // Direct handle access (no null check) — hot path, handles always valid
  const layout = handles[layoutHandle] as GPUBindGroupLayout;
  const entries: GPUBindGroupEntry[] = [];
  let eIdx = I_CB_BASE;
  for (let i = 0; i < entryCount; i++) {
    const buffer = handles[cmdU32[eIdx]] as GPUBuffer;
    const sizeLo = cmdU32[eIdx + 1];
    const sizeHi = cmdU32[eIdx + 2];
    eIdx += 3;
    const size = sizeLo + sizeHi * 0x100000000;
    const resource: GPUBufferBinding = { buffer, offset: 0 };
    if (size !== 0 && size < 2 ** 53) resource.size = size;
    entries.push({ binding: i, resource });
  }

  const bindGroup = device.createBindGroup({ layout, entries });
  const pass = handles[passHandle] as GPUComputePassEncoder;
  pass.setPipeline(handles[pipelineHandle] as GPUComputePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y, z);
  endAndRestartPass(passHandle);
  cmdU32[I_RESULT] = 0;
}

let _fusedUniformDbgCount = 0;

// Debug readback: captures bind group buffers from first 4-entry dispatch
// for async GPU readback after next submit. Uses a done flag (not null check)
// so it only fires once across the entire session.
let _debugReadbackDone = false;
let _debugReadbackBuffers: { buf: GPUBuffer; size: number; label: string }[] | null = null;
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

  // DEBUG: log first 5 dispatches with 4 entries (likely RMSNorm)
  if (entryCount === 4 && _fusedUniformDbgCount < 5) {
    _fusedUniformDbgCount++;
    const u32 = new Uint32Array(cmdBuffer, CMD_OFFSET.UNIFORM_DATA, Math.min(uniformDataSize / 4, 4));
    const f32 = new Float32Array(cmdBuffer, CMD_OFFSET.UNIFORM_DATA, Math.min(uniformDataSize / 4, 4));
    // Log entry buffer handles and sizes
    const entries: string[] = [];
    let eDbg = I_CB_BASE;
    for (let i = 0; i < entryCount; i++) {
      const bh = cmdU32[eDbg]; const sL = cmdU32[eDbg+1]; const sH = cmdU32[eDbg+2];
      entries.push(`[${i}] buf=${bh} size=${sL + sH * 0x100000000}`);
      eDbg += 3;
    }
    console.log(`[RMSNorm-DBG #${_fusedUniformDbgCount}] entries=${entryCount} uniformIdx=${uniformEntryIdx} uniformSize=${uniformDataSize} dispatch=(${x},${y},${z})`);
    console.log(`  params: axis_size=${u32[0]} w_stride=${u32[1]} eps_bits=0x${u32[2]?.toString(16)} eps_f32=${f32[2]} pad=${u32[3]}`);
    console.log(`  bindings: ${entries.join(', ')}`);
  }

  // Direct handle access — hot path, handles always valid
  const layout = handles[layoutHandle] as GPUBindGroupLayout;
  const entries: GPUBindGroupEntry[] = [];
  let newBufferHandle = 0;  // set if we create a buffer inline

  // Check if uniform entry needs buffer creation (handle = 0)
  const uniformBufHandle = cmdU32[I_CB_BASE + uniformEntryIdx * 3];

  if (uniformBufHandle === 0 && uniformDataSize > 0) {
    // Deferred buffer creation: create buffer + write data inline
    // ARG7 has the buffer usage flags from the bridge stub
    const usage = cmdU32[I_ARG0 + 7] || (0x0080 | 0x0008); // STORAGE | COPY_DST fallback
    const uniformData = new Uint8Array(cmdBuffer, CMD_OFFSET.UNIFORM_DATA, uniformDataSize);
    const buffer = device.createBuffer({ size: uniformDataSize, usage });
    queue.writeBuffer(buffer, 0, uniformData);
    const handle = addHandle(buffer);
    bufferSizesArr[handle] = uniformDataSize;
    newBufferHandle = handle;
    // Patch the entry handle for bind group creation below
    cmdU32[I_CB_BASE + uniformEntryIdx * 3] = handle;
  } else if (uniformDataSize > 0) {
    // Buffer exists — just write data
    const uniformBuffer = handles[uniformBufHandle] as GPUBuffer;
    const uniformData = new Uint8Array(cmdBuffer, CMD_OFFSET.UNIFORM_DATA, uniformDataSize);
    queue.writeBuffer(uniformBuffer, 0, uniformData);
  }

  let eIdx = I_CB_BASE;
  for (let i = 0; i < entryCount; i++) {
    const bufHandle = cmdU32[eIdx];
    const buffer = handles[bufHandle] as GPUBuffer;
    const sizeLo = cmdU32[eIdx + 1];
    const sizeHi = cmdU32[eIdx + 2];
    eIdx += 3;
    const size = sizeLo + sizeHi * 0x100000000;
    const resource: GPUBufferBinding = { buffer, offset: 0 };
    if (size !== 0 && size < 2 ** 53) resource.size = size;
    entries.push({ binding: i, resource });
  }

  const bindGroup = device.createBindGroup({ layout, entries });
  const pass = handles[passHandle] as GPUComputePassEncoder;
  pass.setPipeline(handles[pipelineHandle] as GPUComputePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y, z);
  endAndRestartPass(passHandle);

  // Capture buffers for async readback — target RMSNorm specifically:
  // RMSNorm signature: 4 entries, uniformIdx=3, uniformSize=16
  if (entryCount === 4 && uniformEntryIdx === 3 && uniformDataSize === 16 && !_debugReadbackDone) {
    _debugReadbackDone = true;
    _debugReadbackBuffers = [];
    let eDbg2 = I_CB_BASE;
    const labels = ['input', 'weight', 'output', 'uniform'];
    for (let i = 0; i < entryCount; i++) {
      const bh = cmdU32[eDbg2];
      const sL = cmdU32[eDbg2 + 1];
      const sH = cmdU32[eDbg2 + 2];
      const bufSize = sL + sH * 0x100000000;
      // Skip uniform buffer (no COPY_SRC usage) — only read data buffers
      if (i !== uniformEntryIdx) {
        _debugReadbackBuffers.push({
          buf: handles[bh] as GPUBuffer,
          size: Math.min(bufSize, 512), // read first 128 f32 values
          label: labels[i] || `entry${i}`,
        });
      }
      eDbg2 += 3;
    }
    console.log('[DBG-READBACK] Captured RMSNorm buffers:', _debugReadbackBuffers.map(b => `${b.label}(${b.size}b)`).join(', '));
  }

  // Return new buffer handle (or 0 if no creation)
  cmdU32[I_RESULT] = newBufferHandle;
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

  // Async readback of captured RMSNorm buffers
  if (_debugReadbackBuffers) {
    const bufs = _debugReadbackBuffers;
    _debugReadbackBuffers = null; // one-shot

    // Create staging buffers and copy in a separate encoder
    const stagings: { staging: GPUBuffer; label: string }[] = [];
    const readEnc = device.createCommandEncoder();
    for (const { buf, size, label } of bufs) {
      const readSize = Math.min(size, 128); // first 32 f32 values
      if (readSize === 0) continue;
      const staging = device.createBuffer({
        size: readSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      readEnc.copyBufferToBuffer(buf, 0, staging, 0, readSize);
      stagings.push({ staging, label });
    }
    queue.submit([readEnc.finish()]);

    // Schedule async map+log (runs when event loop yields)
    queueMicrotask(async () => {
      try {
        await device.queue.onSubmittedWorkDone();
        for (const { staging, label } of stagings) {
          await staging.mapAsync(GPUMapMode.READ);
          const f32 = new Float32Array(staging.getMappedRange());
          const all = Array.from(f32);
          const nonZero = all.filter(v => v !== 0).length;
          const hasNaN = all.some(v => Number.isNaN(v));
          const hasInf = all.some(v => !Number.isFinite(v) && !Number.isNaN(v));
          const min = Math.min(...all);
          const max = Math.max(...all);
          const first16 = all.slice(0, 16).map(v => v.toFixed(6)).join(', ');
          console.log(
            `[DBG-READBACK] ${label} (${all.length} f32): nonZero=${nonZero} min=${min.toFixed(6)} max=${max.toFixed(6)} NaN=${hasNaN} Inf=${hasInf}`,
          );
          console.log(`  first16: [${first16}]`);
          if (all.length > 16) {
            const last16 = all.slice(-16).map(v => v.toFixed(6)).join(', ');
            console.log(`  last16: [${last16}]`);
          }
          staging.unmap();
          staging.destroy();
        }
        console.log('[DBG-READBACK] Done');
      } catch (e) {
        console.error('[DBG-READBACK] Error:', e);
      }
    });
  }

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
    handles[srcHandle] as GPUBuffer, srcOffset,
    handles[dstHandle] as GPUBuffer, dstOffset,
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
  const SPIN_DISPATCH = 2000;  // optimal: 500→13.3, 1000→14.0, 1500→16.4, 2000→17.5, 4000→17.6
  const SPIN_NONE = 0;
  let spinBudget = SPIN_NONE;  // set after each command based on type

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
        if (currentStatus === STATUS.PENDING) { spun = true; break; }
      }

      if (!spun) {
        // Fall back to async wait
        let waitLoops = 0;
        while (currentStatus !== STATUS.PENDING) {
          const result = Atomics.waitAsync(cmdView, STATUS_INDEX, currentStatus);
          if (result.async) {
            await result.value;
          } else {
            await new Promise(r => setTimeout(r, 0));
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
    } else if (fnId === RpcFn.FUSED_COPY_BUFFER) {
      try {
        handleFusedCopyBuffer();
      } catch (err) {
        console.error(`[GPU Worker] Error in FUSED_COPY_BUFFER:`, err);
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = SPIN_DISPATCH;  // copy is followed by dispatches
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
const I_ARG0 = CMD_OFFSET.ARG0 >>> 2;     // 4
const I_RESULT = CMD_OFFSET.RESULT >>> 2;  // 2
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

  const setResult = (v: number) => { cmdU32[I_RESULT] = v; };
  const setResultBig = (lo: number, hi: number) => {
    cmdU32[I_RESULT] = lo;
    cmdU32[I_RESULT_HI] = hi;
  };

  // 1J: Use cached DataView/Uint8Array over WASM memory.
  // Cache invalidates automatically when buffer identity changes (memory.grow).
  const wasm = getWasmView;
  const wasmBytes = getWasmBytes;

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
      // Args: descPtr
      const descPtr = arg0();
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

      try {
        const buffer = device.createBuffer({ size, usage, mappedAtCreation });
        const handle = addHandle(buffer);
        bufferSizesArr[handle] = size;
        setResult(handle);
      } catch (e) {
        console.error(`[GPU Worker] createBuffer failed: size=${size} usage=0x${usage.toString(16)} mapped=${mappedAtCreation}`, e);
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
      // DEBUG: log RMSNorm shader source
      if (code.includes('rmsnorm') || code.includes('inv_rms') || code.includes('inverseSqrt')) {
        console.log(`[WGSL-DUMP] RMSNorm shader (${code.length} chars):\n` + code);
        // Run standalone GPU test with known values
        runRMSNormTest(code);
      }
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
      const w32 = (v: number) => { view.setUint32(base + off, v, true); off += 4; };
      const w64 = (v: number) => { view.setUint32(base + off, v, true); view.setUint32(base + off + 4, 0, true); off += 8; };
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
      const bufferHandle = arg0();
      const size = bufferSizesArr[bufferHandle] ?? 0;
      setResultBig(size & 0xFFFFFFFF, Math.floor(size / 0x100000000));
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
      const handle = arg0();
      releaseHandle(handle);
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
      let eIdx = I_CB_BASE;  // Uint32Array index for entry data (stride = 3 u32s = 12 bytes)
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
        const buffer = device.createBuffer({ size, usage });
        // Copy data from WASM memory into the buffer
        const data = wasmBytes().slice(wasmDataPtr, wasmDataPtr + size);
        queue.writeBuffer(buffer, 0, data);
        const handle = addHandle(buffer);
        bufferSizesArr[handle] = size;
        setResult(handle);
      } catch (e) {
        console.error(`[GPU Worker] CREATE_BUFFER_FROM_DATA failed: size=${size} usage=0x${usage.toString(16)}`, e);
        setResult(0);
      }
      break;
    }

    default: {
      console.warn(`[GPU Worker] Unknown function ID: ${fnId}`);
      setResult(0);
    }
  }
}
