/**
 * GPU Worker — Owns GPUDevice, processes WebGPU RPC commands
 *
 * This worker's event loop is always free (never blocked by WASM), so GPU
 * async callbacks (onSubmittedWorkDone, mapAsync) resolve naturally.
 *
 * Communication with wasm-worker:
 *   - SharedArrayBuffer command channel (256 bytes, see rpc-protocol.ts)
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

// ---------- Handle Table ----------

const handles = new Map<number, any>();
const bufferSizes = new Map<number, number>();
let nextHandle = 1;

function addHandle(obj: any): number {
  const id = nextHandle++;
  handles.set(id, obj);
  return id;
}

function getHandle<T>(id: number): T {
  const obj = handles.get(id);
  if (!obj) throw new Error(`[GPU Worker] Invalid handle: ${id}`);
  return obj as T;
}

function releaseHandle(id: number): void {
  handles.delete(id);
  bufferSizes.delete(id);
}

// ---------- Memory Helpers ----------

function readString(ptr: number): string {
  if (ptr === 0) return '';
  // Read from wasmMemoryObj.buffer — always returns current byteLength after grow()
  const bytes = new Uint8Array(wasmMemoryObj.buffer);
  let end = ptr;
  const maxLen = bytes.byteLength;
  while (end < maxLen && bytes[end] !== 0) end++;
  return new TextDecoder().decode(bytes.slice(ptr, end));
}

// ---------- GPU State ----------

let device: GPUDevice;
let queue: GPUQueue;
let adapter: GPUAdapter;

// Pre-registered handles (set during init)
let instanceHandle: number;
let adapterHandle: number;
let deviceHandle: number;
let queueHandle: number;

// ---------- Shared Memory ----------

let cmdView: Int32Array;
let cmdDataView: DataView;
let wasmMemoryObj: WebAssembly.Memory;  // Memory object — .buffer always reflects current size after grow()
let readbackView: Uint8Array;

// Pending callbacks: accumulated during async operations (mapAsync, adapter/device request).
// Written to the callback ring in the command buffer when the current RPC call completes.
interface PendingCallback {
  fnPtr: number;
  status: number;
  userdataPtr: number;
}
const pendingCallbacks: PendingCallback[] = [];

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
    const cmdBuffer: SharedArrayBuffer = e.data.cmdBuffer;
    wasmMemoryObj = e.data.wasmMemory;  // WebAssembly.Memory object
    readbackView = new Uint8Array(e.data.readbackBuffer);

    cmdView = new Int32Array(cmdBuffer);
    cmdDataView = new DataView(cmdBuffer);

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

    device = await adapter.requestDevice({
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
      // bf16/f16 must be expanded to f32 for WebGPU — WGSL has no bf16 type,
      // and f16 support is optional. Store as f32 on GPU.
      const needsExpand = isBf16 || isF16;
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
      bufferSizes.set(handle, alignedSize);
      handles.push(handle);
    }

    self.postMessage({ type: 'weights_uploaded', handles, uploadedDtypes });
  }
};

// ---------- Command Processing Loop ----------

async function commandLoop(): Promise<void> {
  while (true) {
    // Wait for wasm-worker to set STATUS = PENDING.
    // After we set DONE, the wasm-worker reads the result, sets IDLE,
    // then sets PENDING for the next command. We must wait for PENDING
    // specifically, not just "not IDLE".
    let currentStatus = Atomics.load(cmdView, STATUS_INDEX);
    let waitLoops = 0;
    while (currentStatus !== STATUS.PENDING) {
      const result = Atomics.waitAsync(cmdView, STATUS_INDEX, currentStatus);
      if (result.async) {
        await result.value;
      } else {
        // Yield to prevent blocking the event loop on synchronous not-equal
        await new Promise(r => setTimeout(r, 0));
      }
      currentStatus = Atomics.load(cmdView, STATUS_INDEX);
      waitLoops++;
      if (waitLoops % 1000 === 0) {
        console.log(`[GPU Worker] waiting for PENDING, status=${currentStatus}, loops=${waitLoops}`);
      }
    }

    const fnId = cmdDataView.getUint32(CMD_OFFSET.FN_ID, true);

    try {
      await processCommand(fnId);
    } catch (err) {
      console.error(`[GPU Worker] Error processing fn ${fnId}:`, err);
      // Return 0 (null handle) to signal error
      cmdDataView.setUint32(CMD_OFFSET.RESULT, 0, true);
    }

    // Write pending callbacks into the callback ring
    flushCallbacks();

    // Signal completion
    Atomics.store(cmdView, STATUS_INDEX, STATUS.DONE);
    Atomics.notify(cmdView, STATUS_INDEX);
  }
}

function flushCallbacks(): void {
  const count = Math.min(pendingCallbacks.length, MAX_CALLBACKS_PER_CALL);
  cmdDataView.setUint32(CMD_OFFSET.CALLBACK_COUNT, count, true);

  for (let i = 0; i < count; i++) {
    const cb = pendingCallbacks[i];
    const base = CMD_OFFSET.CALLBACK_BASE + i * CALLBACK_ENTRY_SIZE;
    cmdDataView.setUint32(base, cb.fnPtr, true);
    cmdDataView.setUint32(base + 4, cb.status, true);
    cmdDataView.setUint32(base + 8, cb.userdataPtr, true);
    cmdDataView.setUint32(base + 12, 0, true); // pad
  }

  // Remove flushed callbacks (keep overflow for next call)
  if (count > 0) {
    pendingCallbacks.splice(0, count);
  }
}

// ---------- Command Dispatch ----------

async function processCommand(fnId: number): Promise<void> {
  // Argument readers (lazy, read from shared memory on demand)
  const arg0 = () => cmdDataView.getUint32(CMD_OFFSET.ARG0, true);
  const arg1 = () => cmdDataView.getUint32(CMD_OFFSET.ARG1, true);
  const arg2 = () => cmdDataView.getUint32(CMD_OFFSET.ARG2, true);
  const arg3 = () => cmdDataView.getUint32(CMD_OFFSET.ARG3, true);
  const arg4 = () => cmdDataView.getUint32(CMD_OFFSET.ARG4, true);
  const arg5 = () => cmdDataView.getUint32(CMD_OFFSET.ARG5, true);
  const arg6 = () => cmdDataView.getUint32(CMD_OFFSET.ARG6, true);
  const arg7 = () => cmdDataView.getUint32(CMD_OFFSET.ARG7, true);
  const arg0Hi = () => cmdDataView.getUint32(CMD_OFFSET.ARG0_HI, true);
  const arg1Hi = () => cmdDataView.getUint32(CMD_OFFSET.ARG1_HI, true);
  const arg2Hi = () => cmdDataView.getUint32(CMD_OFFSET.ARG2_HI, true);

  const setResult = (v: number) => cmdDataView.setUint32(CMD_OFFSET.RESULT, v, true);
  const setResultBig = (lo: number, hi: number) => {
    cmdDataView.setUint32(CMD_OFFSET.RESULT, lo, true);
    cmdDataView.setUint32(CMD_OFFSET.RESULT_HI, hi, true);
  };

  // Fresh DataView over WASM memory for reading struct descriptors.
  // Must be created on each call since the underlying SAB can grow.
  // Always read wasmMemoryObj.buffer (not a cached SAB) — .buffer getter returns
  // a fresh SAB with the current byteLength after any memory.grow() on the wasm-worker.
  const wasm = () => new DataView(wasmMemoryObj.buffer);
  const wasmBytes = () => new Uint8Array(wasmMemoryObj.buffer);

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
      pendingCallbacks.push({ fnPtr: callbackPtr, status: 0, userdataPtr });
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
      pendingCallbacks.push({ fnPtr: callbackPtr, status: 0, userdataPtr });
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
        bufferSizes.set(handle, size);
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
      const code = readString(codePtr);
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
      setResult(addHandle(pass));
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
      setResult(0);
      break;
    }

    case RpcFn.COMPUTE_PASS_END: {
      // Args: passHandle
      const passHandle = arg0();
      getHandle<GPUComputePassEncoder>(passHandle).end();
      setResult(0);
      break;
    }

    case RpcFn.COMPUTE_PASS_RELEASE: {
      const handle = arg0();
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
      const size = bufferSizes.get(bufferHandle) ?? 0;
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
      if (size === 0) size = (bufferSizes.get(bufferHandle) ?? 0) - offset;

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
      if (size === 0) size = (bufferSizes.get(bufferHandle) ?? 0) - offset;

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
        pendingCallbacks.push({ fnPtr: callbackPtr, status: 0, userdataPtr });
      } catch (err) {
        console.error('[GPU Worker] mapAsync failed:', err);
        pendingCallbacks.push({ fnPtr: callbackPtr, status: 1, userdataPtr });
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
      // NOW it's safe to fire GPU-done callbacks. Move them to pendingCallbacks
      // so flushCallbacks() writes them to the callback ring for the wasm-worker.
      if (gpuDoneCallbacks.length > 0) {
        pendingCallbacks.push(...gpuDoneCallbacks);
        gpuDoneCallbacks.length = 0;
      }
      setResult(pendingCallbacks.length);
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
      setResult(0);
      break;
    }

    case RpcFn.FUSED_DISPATCH_2BG: {
      // Fused: setPipeline + setBindGroup(0,1) + dispatch in one RPC call
      // Args: passHandle, pipelineHandle, bg0Handle, bg1Handle, x, y
      const passHandle = arg0();
      const pipelineHandle = arg1();
      const bg0Handle = arg2();
      const bg1Handle = arg3();
      const x = arg4();
      const y = arg5();
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      pass.setPipeline(getHandle<GPUComputePipeline>(pipelineHandle));
      pass.setBindGroup(0, getHandle<GPUBindGroup>(bg0Handle));
      pass.setBindGroup(1, getHandle<GPUBindGroup>(bg1Handle));
      pass.dispatchWorkgroups(x, y, 1);
      setResult(0);
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

    default: {
      console.warn(`[GPU Worker] Unknown function ID: ${fnId}`);
      setResult(0);
    }
  }
}
