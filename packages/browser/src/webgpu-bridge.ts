/**
 * WebGPU JavaScript Bridge for MLX WASM
 *
 * Implements the ~35 webgpu.h C API functions that the MLX WebGPU backend calls.
 * These are provided as WASM imports via NAPI-RS's `overwriteImports` mechanism.
 *
 * Pattern: C code calls wgpuDeviceCreateBuffer(device, &desc) → WASM import →
 * this JS function reads the descriptor from WASM memory, calls browser
 * GPUDevice.createBuffer(), stores the result in a handle table, returns the
 * integer handle back to C.
 *
 * Struct layouts verified against crates/mlx-sys/mlx/mlx/backend/webgpu/include/webgpu/webgpu.h
 * Reference: Dawn's emdawnwebgpu library_webgpu.js for patterns.
 */

// ---------- Handle Table ----------

type GPUObject =
  | GPUInstance
  | GPUAdapter
  | GPUDevice
  | GPUQueue
  | GPUBuffer
  | GPUShaderModule
  | GPUComputePipeline
  | GPUBindGroup
  | GPUBindGroupLayout
  | GPUCommandEncoder
  | GPUComputePassEncoder
  | GPUCommandBuffer;

interface GPUInstance {
  __brand: 'instance';
}

const handles = new Map<number, GPUObject>();
const bufferSizes = new Map<number, number>();
let nextHandle = 1;

function addHandle(obj: GPUObject): number {
  const id = nextHandle++;
  handles.set(id, obj);
  return id;
}

function getHandle<T extends GPUObject>(id: number): T {
  const obj = handles.get(id);
  if (!obj) throw new Error(`[WebGPU Bridge] Invalid handle: ${id}`);
  return obj as T;
}

function releaseHandle(id: number): void {
  handles.delete(id);
  bufferSizes.delete(id);
}

// ---------- Memory Helpers ----------

function readString(memory: ArrayBuffer, ptr: number): string {
  if (ptr === 0) return '';
  const bytes = new Uint8Array(memory);
  let end = ptr;
  while (bytes[end] !== 0) end++;
  // .slice() copies to a regular ArrayBuffer — TextDecoder rejects SharedArrayBuffer views
  return new TextDecoder().decode(bytes.slice(ptr, end));
}

// ---------- Bridge Factory ----------

export interface WebGPUBridge {
  imports: Record<string, (...args: number[]) => number | void>;
  setInstance(instance: WebAssembly.Instance): void;
  /** Register an externally-created GPU buffer in the handle table */
  addGPUBuffer(buffer: GPUBuffer, size: number): number;
  adapter: GPUAdapter;
  device: GPUDevice;
}

export function createWebGPUBridge(adapter: GPUAdapter, device: GPUDevice): WebGPUBridge {
  const queue = device.queue;

  // Pre-register handles for pre-created objects
  const instanceHandle = addHandle({ __brand: 'instance' } as GPUInstance);
  const adapterHandle = addHandle(adapter as unknown as GPUObject);
  const deviceHandle = addHandle(device as unknown as GPUObject);
  const queueHandle = addHandle(queue as unknown as GPUObject);

  // WASM exports — set via setInstance() after instantiation
  let wasmMemory: WebAssembly.Memory;
  let wasmTable: WebAssembly.Table;
  let wasmMalloc: (size: number) => number;
  let wasmFree: (ptr: number) => void;

  // GPU synchronization via Atomics.wait (no asyncify needed).
  //
  // Instead of returning Promises from mlx_webgpu_poll (which requires asyncify
  // to yield the WASM thread), we block synchronously via Atomics.wait.
  // A helper worker handles GPU completion callbacks and notifies us.
  //
  // Signal layout in SharedArrayBuffer (4 bytes each):
  //   [0] = done signal: 0 = waiting, 1 = GPU work complete
  //   [1] = submit signal: 0 = idle, 1 = main thread waiting for GPU
  const signalBuffer = new SharedArrayBuffer(8);
  const signalView = new Int32Array(signalBuffer);
  const DONE_IDX = 0;
  const SUBMIT_IDX = 1;

  // Spawn a lightweight helper worker for GPU completion callbacks.
  // It shares the GPUDevice and uses Atomics.notify to wake the main worker.
  const helperCode = `
    let queue = null;
    const DONE_IDX = 0;
    const SUBMIT_IDX = 1;

    self.onmessage = async (e) => {
      if (e.data.type === 'init') {
        queue = e.data.queue;
        pollLoop(new Int32Array(e.data.signalBuffer));
      }
    };

    async function pollLoop(signal) {
      while (true) {
        // Wait for main thread to signal that GPU work was submitted
        const result = Atomics.waitAsync(signal, SUBMIT_IDX, 0);
        if (result.async) await result.value;

        // Reset submit signal
        Atomics.store(signal, SUBMIT_IDX, 0);

        // Wait for all submitted GPU work to complete
        await queue.onSubmittedWorkDone();

        // Signal the main thread that GPU work is done
        Atomics.store(signal, DONE_IDX, 1);
        Atomics.notify(signal, DONE_IDX);
      }
    }
  `;
  const helperBlob = new Blob([helperCode], { type: 'application/javascript' });
  const helperWorker = new Worker(URL.createObjectURL(helperBlob));

  let pendingCallbacks = 0;
  let pendingResolvers: Array<() => void> = [];
  let pollCount = 0;

  // Track mapped buffer ranges for the WASM memory shadow pattern
  interface MappedRange {
    wasmPtr: number;
    jsRange: ArrayBuffer;
    size: number;
  }
  const activeMappings = new Map<number, MappedRange>();

  function setInstance(instance: WebAssembly.Instance): void {
    wasmMemory = instance.exports.memory as WebAssembly.Memory;
    wasmTable = instance.exports.__indirect_function_table as WebAssembly.Table;
    wasmMalloc = instance.exports.malloc as (size: number) => number;
    wasmFree = instance.exports.free as (ptr: number) => void;
  }

  function callCallback(fnPtr: number, ...args: number[]): void {
    const fn = wasmTable.get(fnPtr) as (...a: number[]) => void;
    if (fn) fn(...args);
  }

  const imports: Record<string, (...args: number[]) => number | void> = {
    // ===== Instance (pre-created) =====
    wgpuCreateInstance(_descPtr: number): number {
      return instanceHandle;
    },

    wgpuInstanceRequestAdapter(_instance: number, _optsPtr: number, callbackPtr: number, userdataPtr: number): void {
      callCallback(callbackPtr, 0, adapterHandle, 0, userdataPtr);
    },

    wgpuInstanceRelease(): void {},

    // ===== Adapter (pre-created) =====
    wgpuAdapterRequestDevice(_adapter: number, _descPtr: number, callbackPtr: number, userdataPtr: number): void {
      callCallback(callbackPtr, 0, deviceHandle, 0, userdataPtr);
    },

    wgpuAdapterGetLimits(_adapterHandle: number, limitsPtr: number): number {
      // Write adapter limits to the provided pointer (same struct as device limits).
      // In this direct bridge the adapter + device limits are identical.
      const view = new DataView(wasmMemory.buffer);
      // WGPULimits struct starts at limitsPtr + 8 (past nextInChain + sType).
      const base = limitsPtr + 8;
      view.setUint32(base + 0, device.limits.maxTextureDimension1D, true);
      view.setUint32(base + 4, device.limits.maxTextureDimension2D, true);
      view.setUint32(base + 8, device.limits.maxTextureDimension3D, true);
      view.setUint32(base + 12, device.limits.maxTextureArrayLayers, true);
      view.setUint32(base + 16, device.limits.maxBindGroups, true);
      view.setUint32(base + 20, device.limits.maxBindingsPerBindGroup, true);
      view.setUint32(base + 24, device.limits.maxDynamicUniformBuffersPerPipelineLayout, true);
      view.setUint32(base + 28, device.limits.maxDynamicStorageBuffersPerPipelineLayout, true);
      view.setUint32(base + 32, device.limits.maxSampledTexturesPerShaderStage, true);
      view.setUint32(base + 36, device.limits.maxSamplersPerShaderStage, true);
      view.setUint32(base + 40, device.limits.maxStorageBuffersPerShaderStage, true);
      view.setUint32(base + 44, device.limits.maxStorageTexturesPerShaderStage, true);
      view.setUint32(base + 48, device.limits.maxUniformBuffersPerShaderStage, true);
      // maxUniformBufferBindingSize at +52 (u64)
      const ubbs = device.limits.maxUniformBufferBindingSize;
      view.setUint32(base + 52, ubbs & 0xffffffff, true);
      view.setUint32(base + 56, 0, true);
      // maxStorageBufferBindingSize at +60 (u64)
      const sbbs = device.limits.maxStorageBufferBindingSize;
      view.setUint32(base + 60, sbbs & 0xffffffff, true);
      view.setUint32(base + 64, 0, true);
      return 1; // success
    },

    wgpuAdapterHasFeature(_adapter: number, feature: number): number {
      if (feature === 14 && adapter.features.has('shader-f16')) return 1; // WGPUFeatureName_ShaderF16
      if (feature === 0x3f1 && adapter.features.has('subgroups')) return 1; // WGPUFeatureName_Subgroups
      return 0;
    },

    wgpuAdapterRelease(): void {},

    wgpuAdapterGetProperties(_adapter: number, propsPtr: number): void {
      // Zero out — MLX reads vendor info but handles zeros gracefully
      new Uint8Array(wasmMemory.buffer).fill(0, propsPtr, propsPtr + 256);
    },

    // ===== Device =====
    wgpuDeviceCreateBuffer(_device: number, descPtr: number): number {
      const view = new DataView(wasmMemory.buffer);
      // WGPUBufferDescriptor (WASM32, 28 bytes):
      //   0: nextInChain (ptr4)
      //   4: label (ptr4)
      //   8: usage (uint64) — WGPUBufferUsageFlags
      //  16: size (uint64)
      //  24: mappedAtCreation (uint32/bool)
      const usage = view.getUint32(descPtr + 8, true); // low word suffices (flags < 32 bits)
      const sizeLo = view.getUint32(descPtr + 16, true);
      const sizeHi = view.getUint32(descPtr + 20, true);
      const size = sizeLo + sizeHi * 0x100000000;
      const mappedAtCreation = view.getUint32(descPtr + 24, true) !== 0;

      try {
        const buffer = device.createBuffer({ size, usage, mappedAtCreation });
        const handle = addHandle(buffer as unknown as GPUObject);
        bufferSizes.set(handle, size);
        return handle;
      } catch (e) {
        console.error(
          `[WebGPU Bridge] createBuffer failed: size=${size} usage=0x${usage.toString(16)} mapped=${mappedAtCreation}`,
          e,
        );
        return 0;
      }
    },

    wgpuDeviceCreateShaderModule(_device: number, descPtr: number): number {
      const view = new DataView(wasmMemory.buffer);
      // WGPUShaderModuleDescriptor (8 bytes):
      //   0: nextInChain (ptr4) → WGPUShaderModuleWGSLDescriptor
      //   4: label (ptr4)
      const nextInChainPtr = view.getUint32(descPtr, true);
      if (nextInChainPtr === 0) {
        throw new Error('[WebGPU Bridge] ShaderModule descriptor has no nextInChain');
      }
      // WGPUShaderModuleWGSLDescriptor (12 bytes):
      //   0: chain.next (ptr4)
      //   4: chain.sType (uint32) = 5
      //   8: code (ptr4)
      const codePtr = view.getUint32(nextInChainPtr + 8, true);
      const code = readString(wasmMemory.buffer, codePtr);
      const module = device.createShaderModule({ code });
      return addHandle(module as unknown as GPUObject);
    },

    wgpuDeviceCreateComputePipeline(_device: number, descPtr: number): number {
      const view = new DataView(wasmMemory.buffer);
      // WGPUComputePipelineDescriptor (32 bytes):
      //   0: nextInChain (ptr4)
      //   4: label (ptr4)
      //   8: layout (ptr4) — 0 = auto
      //  12: compute.nextInChain (ptr4)
      //  16: compute.module (ptr4 = handle)
      //  20: compute.entryPoint (ptr4)
      //  24: compute.constantCount (uint32)
      //  28: compute.constants (ptr4)
      const moduleHandle = view.getUint32(descPtr + 16, true);
      const entryPointPtr = view.getUint32(descPtr + 20, true);
      const module = getHandle<GPUShaderModule>(moduleHandle);
      const entryPoint = entryPointPtr ? readString(wasmMemory.buffer, entryPointPtr) : 'main';
      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint },
      });
      return addHandle(pipeline as unknown as GPUObject);
    },

    wgpuDeviceCreateBindGroup(_device: number, descPtr: number): number {
      const view = new DataView(wasmMemory.buffer);
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
          entries.push({
            binding,
            resource: { buffer: getHandle<GPUBuffer>(bufferHandle), offset, size },
          });
        }
      }

      const bindGroup = device.createBindGroup({ layout, entries });
      return addHandle(bindGroup as unknown as GPUObject);
    },

    wgpuDeviceCreateCommandEncoder(_device: number, _descPtr: number): number {
      return addHandle(device.createCommandEncoder() as unknown as GPUObject);
    },

    wgpuDeviceGetQueue(): number {
      return queueHandle;
    },

    wgpuDeviceGetLimits(_device: number, limitsPtr: number): number {
      // WGPUSupportedLimits (WASM32): nextInChain (ptr4) + WGPULimits
      // WGPULimits starts at offset 4 (after nextInChain pointer)
      const view = new DataView(wasmMemory.buffer);
      const L = device.limits;
      const base = limitsPtr + 4; // skip nextInChain
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
      w32(L.maxTextureDimension1D); // +0
      w32(L.maxTextureDimension2D); // +4
      w32(L.maxTextureDimension3D); // +8
      w32(L.maxTextureArrayLayers); // +12
      w32(L.maxBindGroups); // +16
      w32(L.maxBindGroupsPlusVertexBuffers ?? 0); // +20
      w32(L.maxBindingsPerBindGroup); // +24
      w32(L.maxDynamicUniformBuffersPerPipelineLayout); // +28
      w32(L.maxDynamicStorageBuffersPerPipelineLayout); // +32
      w32(L.maxSampledTexturesPerShaderStage); // +36
      w32(L.maxSamplersPerShaderStage); // +40
      w32(L.maxStorageBuffersPerShaderStage); // +44
      w32(L.maxStorageTexturesPerShaderStage); // +48
      w32(L.maxUniformBuffersPerShaderStage); // +52
      w64(L.maxUniformBufferBindingSize); // +56
      w64(L.maxStorageBufferBindingSize); // +64
      w32(L.minUniformBufferOffsetAlignment); // +72
      w32(L.minStorageBufferOffsetAlignment); // +76
      w32(L.maxVertexBuffers); // +80
      w64(L.maxBufferSize); // +84
      w32(L.maxVertexAttributes); // +92
      w32(L.maxVertexBufferArrayStride); // +96
      w32(L.maxInterStageShaderComponents); // +100
      w32(L.maxInterStageShaderVariables); // +104
      w32(L.maxColorAttachments); // +108
      w32(L.maxColorAttachmentBytesPerSample ?? 0); // +112
      w32(L.maxComputeWorkgroupStorageSize); // +116
      w32(L.maxComputeInvocationsPerWorkgroup); // +120
      w32(L.maxComputeWorkgroupSizeX); // +124
      w32(L.maxComputeWorkgroupSizeY); // +128
      w32(L.maxComputeWorkgroupSizeZ); // +132
      w32(L.maxComputeWorkgroupsPerDimension); // +136
      return 1;
    },

    wgpuDeviceSetUncapturedErrorCallback(): void {
      device.onuncapturederror = (event) => {
        console.error('[WebGPU] Uncaptured error:', event.error.message);
      };
    },

    wgpuDeviceSetDeviceLostCallback(): void {
      device.lost.then((info) => {
        console.error('[WebGPU] Device lost:', info.message);
      });
    },

    wgpuDeviceRelease(): void {},

    // ===== Queue =====
    wgpuQueueSubmit(_queue: number, count: number, cmdBufArrayPtr: number): void {
      const view = new DataView(wasmMemory.buffer);
      const commandBuffers: GPUCommandBuffer[] = [];
      for (let i = 0; i < count; i++) {
        const handle = view.getUint32(cmdBufArrayPtr + i * 4, true);
        commandBuffers.push(getHandle<GPUCommandBuffer>(handle));
      }
      queue.submit(commandBuffers);
      console.log(`[WebGPU Bridge] queue.submit(${count} cmd bufs), handles=${handles.size}`);
    },

    wgpuQueueWriteBuffer(
      _queue: number,
      bufferHandle: number,
      bufferOffset: bigint,
      dataPtr: number,
      size: number,
    ): void {
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      if (!buffer) {
        console.error(`[WebGPU Bridge] writeBuffer: invalid handle ${bufferHandle}`);
        return;
      }
      const data = new Uint8Array(wasmMemory.buffer, dataPtr, size);
      queue.writeBuffer(buffer, Number(bufferOffset), data);
    },

    wgpuQueueOnSubmittedWorkDone(_queue: number, callbackPtr: number, userdataPtr: number): void {
      pendingCallbacks++;
      queue.onSubmittedWorkDone().then(() => {
        pendingCallbacks--;
        callCallback(callbackPtr, 0, userdataPtr);
        // Wake any waiting poll — GPU work is done
        if (pendingCallbacks === 0) {
          const resolvers = pendingResolvers;
          pendingResolvers = [];
          for (const resolve of resolvers) resolve();
        }
      });
    },

    wgpuQueueRelease(): void {},

    // ===== Command Encoder =====
    wgpuCommandEncoderBeginComputePass(encoderHandle: number, _descPtr: number): number {
      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      return addHandle(encoder.beginComputePass() as unknown as GPUObject);
    },

    // Signature: (i32 encoder, i32 src, i64 srcOffset, i32 dst, i64 dstOffset, i64 size)
    wgpuCommandEncoderCopyBufferToBuffer(
      encoderHandle: number,
      srcHandle: number,
      srcOffset: bigint,
      dstHandle: number,
      dstOffset: bigint,
      size: bigint,
    ): void {
      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      const src = getHandle<GPUBuffer>(srcHandle);
      const dst = getHandle<GPUBuffer>(dstHandle);
      encoder.copyBufferToBuffer(src, Number(srcOffset), dst, Number(dstOffset), Number(size));
    },

    wgpuCommandEncoderFinish(encoderHandle: number, _descPtr: number): number {
      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      return addHandle(encoder.finish() as unknown as GPUObject);
    },

    wgpuCommandEncoderRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Command Buffer =====
    wgpuCommandBufferRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Compute Pass Encoder =====
    wgpuComputePassEncoderSetPipeline(passHandle: number, pipelineHandle: number): void {
      getHandle<GPUComputePassEncoder>(passHandle).setPipeline(getHandle<GPUComputePipeline>(pipelineHandle));
    },

    wgpuComputePassEncoderSetBindGroup(
      passHandle: number,
      groupIndex: number,
      bgHandle: number,
      dynamicOffsetCount: number,
      dynamicOffsetsPtr: number,
    ): void {
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      const bindGroup = getHandle<GPUBindGroup>(bgHandle);
      if (dynamicOffsetCount > 0 && dynamicOffsetsPtr !== 0) {
        const view = new DataView(wasmMemory.buffer);
        const offsets: number[] = [];
        for (let i = 0; i < dynamicOffsetCount; i++) {
          offsets.push(view.getUint32(dynamicOffsetsPtr + i * 4, true));
        }
        pass.setBindGroup(groupIndex, bindGroup, offsets);
      } else {
        pass.setBindGroup(groupIndex, bindGroup);
      }
    },

    wgpuComputePassEncoderDispatchWorkgroups(passHandle: number, x: number, y: number, z: number): void {
      getHandle<GPUComputePassEncoder>(passHandle).dispatchWorkgroups(x, y, z);
    },

    wgpuComputePassEncoderEnd(passHandle: number): void {
      getHandle<GPUComputePassEncoder>(passHandle).end();
    },

    wgpuComputePassEncoderRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Buffer =====
    wgpuBufferGetSize(bufferHandle: number): bigint {
      return BigInt(bufferSizes.get(bufferHandle) ?? 0);
    },

    // Buffer mapping — WASM memory shadow pattern
    //
    // Problem: GPUBuffer.getMappedRange() returns a JS ArrayBuffer, but C code
    // expects a pointer into WASM linear memory it can memcpy to/from.
    //
    // Solution: Allocate a shadow region in WASM memory via malloc, return that
    // pointer to C. On unmap, copy between shadow and JS ArrayBuffer.
    wgpuBufferGetMappedRange(bufferHandle: number, offset: number, size: number): number {
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      if (size === 0) size = (bufferSizes.get(bufferHandle) ?? 0) - offset;
      const jsRange = buffer.getMappedRange(offset, size);

      // Allocate shadow in WASM linear memory
      const wasmPtr = wasmMalloc(size);
      if (wasmPtr === 0) throw new Error('[WebGPU Bridge] malloc failed for mapped range');

      // For mappedAtCreation (write path): C will write to wasmPtr, we copy to jsRange on unmap
      // For mapAsync read: we copy jsRange into wasmPtr now so C can read it
      // We don't know which case here, so just store the mapping — unmap handles both
      activeMappings.set(bufferHandle, { wasmPtr, jsRange, size });
      return wasmPtr;
    },

    wgpuBufferGetConstMappedRange(bufferHandle: number, offset: number, size: number): number {
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      if (size === 0) size = (bufferSizes.get(bufferHandle) ?? 0) - offset;
      const jsRange = buffer.getMappedRange(offset, size);

      // Allocate shadow and copy GPU data into WASM memory (read path)
      const wasmPtr = wasmMalloc(size);
      if (wasmPtr === 0) throw new Error('[WebGPU Bridge] malloc failed for const mapped range');

      const src = new Uint8Array(jsRange);
      const dst = new Uint8Array(wasmMemory.buffer, wasmPtr, size);
      dst.set(src);

      activeMappings.set(bufferHandle, { wasmPtr, jsRange, size });
      return wasmPtr;
    },

    wgpuBufferUnmap(bufferHandle: number): void {
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      const mapping = activeMappings.get(bufferHandle);

      if (mapping) {
        // Copy from WASM shadow into JS ArrayBuffer (write path for mappedAtCreation)
        const src = new Uint8Array(wasmMemory.buffer, mapping.wasmPtr, mapping.size);
        const dst = new Uint8Array(mapping.jsRange);
        dst.set(src);

        activeMappings.delete(bufferHandle);
        wasmFree(mapping.wasmPtr);
      }

      buffer.unmap();
    },

    wgpuBufferMapAsync(
      bufferHandle: number,
      mode: number,
      offset: number,
      size: number,
      callbackPtr: number,
      userdataPtr: number,
    ): void {
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      const gpuMode = mode === 1 ? GPUMapMode.READ : GPUMapMode.WRITE;
      pendingCallbacks++;
      buffer.mapAsync(gpuMode, offset, size).then(
        () => {
          pendingCallbacks--;
          callCallback(callbackPtr, 0, userdataPtr);
          if (pendingCallbacks === 0) {
            const resolvers = pendingResolvers;
            pendingResolvers = [];
            for (const resolve of resolvers) resolve();
          }
        },
        (err: unknown) => {
          pendingCallbacks--;
          console.error('[WebGPU Bridge] mapAsync failed:', err);
          callCallback(callbackPtr, 1, userdataPtr);
          if (pendingCallbacks === 0) {
            const resolvers = pendingResolvers;
            pendingResolvers = [];
            for (const resolve of resolvers) resolve();
          }
        },
      );
    },

    wgpuBufferDestroy(bufferHandle: number): void {
      getHandle<GPUBuffer>(bufferHandle).destroy();
    },

    wgpuBufferRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Pipeline =====
    wgpuComputePipelineGetBindGroupLayout(pipelineHandle: number, index: number): number {
      const pipeline = getHandle<GPUComputePipeline>(pipelineHandle);
      return addHandle(pipeline.getBindGroupLayout(index) as unknown as GPUObject);
    },

    wgpuComputePipelineRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Release =====
    wgpuBindGroupRelease(handle: number): void {
      releaseHandle(handle);
    },
    wgpuBindGroupLayoutRelease(handle: number): void {
      releaseHandle(handle);
    },
    wgpuShaderModuleRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Polling =====
    mlx_webgpu_poll() {
      // Returns a Promise when GPU work is pending → triggers asyncify yield.
      // With synchronize() removed from the WASI path, this is only called
      // from raw_ptr()'s mapAsync poll loop (once per token readback).
      pollCount++;
      if (pendingCallbacks > 0) {
        return new Promise<void>((resolve) => {
          pendingResolvers.push(resolve);
        });
      }
      return Promise.resolve();
    },
  };

  function addGPUBuffer(buffer: GPUBuffer, size: number): number {
    const handle = addHandle(buffer as unknown as GPUObject);
    bufferSizes.set(handle, size);
    return handle;
  }

  return { imports, adapter, device, setInstance, addGPUBuffer };
}
