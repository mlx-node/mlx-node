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
 * Reference: Dawn's emdawnwebgpu library_webgpu.js for struct layouts and naming.
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

// Sentinel type for our pre-created instance
interface GPUInstance {
  __brand: 'instance';
}

const handles = new Map<number, GPUObject>();
const bufferSizes = new Map<number, number>(); // track buffer sizes for wgpuBufferGetSize
const mappedRanges = new Map<number, ArrayBuffer>(); // track mapped ranges
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
  mappedRanges.delete(id);
}

// ---------- Memory Helpers ----------

/** Read a null-terminated UTF-8 string from WASM memory */
function readString(memory: ArrayBuffer, ptr: number): string {
  if (ptr === 0) return '';
  const bytes = new Uint8Array(memory);
  let end = ptr;
  while (bytes[end] !== 0) end++;
  return new TextDecoder().decode(bytes.slice(ptr, end));
}

/** Write a null-terminated UTF-8 string to WASM memory */
function writeString(memory: ArrayBuffer, ptr: number, str: string, maxBytes: number): void {
  const encoded = new TextEncoder().encode(str);
  const bytes = new Uint8Array(memory);
  const len = Math.min(encoded.length, maxBytes - 1);
  bytes.set(encoded.subarray(0, len), ptr);
  bytes[ptr + len] = 0;
}

// ---------- WebGPU Usage Flag Mapping ----------

function mapBufferUsage(cUsage: number): GPUBufferUsageFlags {
  // webgpu.h WGPUBufferUsage flags match WebGPU spec values
  // MapRead=1, MapWrite=2, CopySrc=4, CopyDst=8, Index=16, Vertex=32,
  // Uniform=64, Storage=128, Indirect=256, QueryResolve=512
  return cUsage as GPUBufferUsageFlags;
}

// ---------- Bridge Factory ----------

export interface WebGPUBridge {
  imports: Record<string, (...args: number[]) => number | void>;
  adapter: GPUAdapter;
  device: GPUDevice;
}

/**
 * Create the WebGPU bridge with pre-created adapter and device.
 * The returned `imports` object is injected into the WASM module via `overwriteImports`.
 */
export function createWebGPUBridge(
  adapter: GPUAdapter,
  device: GPUDevice,
): WebGPUBridge {
  const queue = device.queue;

  // Pre-register handles for pre-created objects
  const instanceHandle = addHandle({ __brand: 'instance' } as GPUInstance);
  const adapterHandle = addHandle(adapter as unknown as GPUObject);
  const deviceHandle = addHandle(device as unknown as GPUObject);
  const queueHandle = addHandle(queue as unknown as GPUObject);

  // WASM memory — set during module instantiation
  let wasmMemory: WebAssembly.Memory;
  let wasmTable: WebAssembly.Table;

  /** Call a C function pointer from JS (for async callbacks) */
  function callCallback(fnPtr: number, ...args: number[]): void {
    const fn = wasmTable.get(fnPtr) as (...a: number[]) => void;
    if (fn) fn(...args);
  }

  const imports: Record<string, (...args: number[]) => number | void> = {
    // ===== Instance =====
    wgpuCreateInstance(_descPtr: number): number {
      return instanceHandle;
    },

    wgpuInstanceRequestAdapter(
      _instance: number, _optsPtr: number, callbackPtr: number, userdataPtr: number,
    ): void {
      // Pre-created: call callback immediately with success
      // WGPURequestAdapterStatus_Success = 0
      callCallback(callbackPtr, 0, adapterHandle, 0, userdataPtr);
    },

    wgpuInstanceRelease(_handle: number): void {
      // No-op for pre-created instance
    },

    // ===== Adapter =====
    wgpuAdapterRequestDevice(
      _adapter: number, _descPtr: number, callbackPtr: number, userdataPtr: number,
    ): void {
      // Pre-created: call callback immediately
      // WGPURequestDeviceStatus_Success = 0
      callCallback(callbackPtr, 0, deviceHandle, 0, userdataPtr);
    },

    wgpuAdapterRelease(_handle: number): void {
      // No-op for pre-created
    },

    wgpuAdapterGetProperties(_adapter: number, propsPtr: number): void {
      // Write minimal adapter properties to WASM memory
      // The struct layout is complex — for now write zeros (MLX only reads a few fields)
      const view = new DataView(wasmMemory.buffer);
      // Zero out the struct (safe default)
      for (let i = 0; i < 256; i++) {
        view.setUint8(propsPtr + i, 0);
      }
    },

    // ===== Device =====
    wgpuDeviceCreateBuffer(_device: number, descPtr: number): number {
      const view = new DataView(wasmMemory.buffer);
      // WGPUBufferDescriptor layout (simplified):
      // offset 0: nextInChain (ptr)
      // offset 4/8: label (ptr) — platform dependent
      // Then: usage (uint32), size (uint64), mappedAtCreation (bool)
      // Exact offsets depend on pointer size (32-bit in WASM)
      const nextInChain = view.getUint32(descPtr, true);
      const labelPtr = view.getUint32(descPtr + 4, true);
      const usage = view.getUint32(descPtr + 8, true);
      // size is uint64 at offset 16 (aligned)
      const sizeLo = view.getUint32(descPtr + 16, true);
      const sizeHi = view.getUint32(descPtr + 20, true);
      const size = sizeLo + sizeHi * 0x100000000;
      const mappedAtCreation = view.getUint32(descPtr + 24, true) !== 0;

      const buffer = device.createBuffer({
        size,
        usage: mapBufferUsage(usage),
        mappedAtCreation,
      });

      const handle = addHandle(buffer as unknown as GPUObject);
      bufferSizes.set(handle, size);

      if (mappedAtCreation) {
        mappedRanges.set(handle, buffer.getMappedRange());
      }

      return handle;
    },

    wgpuDeviceCreateShaderModule(_device: number, descPtr: number): number {
      const view = new DataView(wasmMemory.buffer);
      // WGPUShaderModuleDescriptor has nextInChain which points to
      // WGPUShaderModuleWGSLDescriptor containing the WGSL source
      const nextInChainPtr = view.getUint32(descPtr, true);
      if (nextInChainPtr === 0) {
        throw new Error('[WebGPU Bridge] ShaderModule descriptor has no nextInChain');
      }

      // WGPUShaderModuleWGSLDescriptor:
      // offset 0: chain.next (ptr)
      // offset 4: chain.sType (uint32) — should be WGPUSType_ShaderSourceWGSL
      // offset 8: code (const char*)
      const codePtr = view.getUint32(nextInChainPtr + 8, true);
      const code = readString(wasmMemory.buffer, codePtr);

      const module = device.createShaderModule({ code });
      return addHandle(module as unknown as GPUObject);
    },

    wgpuDeviceCreateComputePipeline(_device: number, descPtr: number): number {
      const view = new DataView(wasmMemory.buffer);
      // WGPUComputePipelineDescriptor:
      // offset 0: nextInChain (ptr)
      // offset 4: label (ptr)
      // offset 8: layout (ptr) — WGPUPipelineLayout, 0 = auto
      // offset 12: compute.nextInChain (ptr)
      // offset 16: compute.module (WGPUShaderModule handle)
      // offset 20: compute.entryPoint (const char*)
      // offset 24: compute.constantCount (size_t)
      // offset 28: compute.constants (ptr)
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
      // WGPUBindGroupDescriptor:
      // offset 0: nextInChain
      // offset 4: label
      // offset 8: layout (WGPUBindGroupLayout handle)
      // offset 12: entryCount (size_t)
      // offset 16: entries (WGPUBindGroupEntry*)
      const layoutHandle = view.getUint32(descPtr + 8, true);
      const entryCount = view.getUint32(descPtr + 12, true);
      const entriesPtr = view.getUint32(descPtr + 16, true);

      const layout = getHandle<GPUBindGroupLayout>(layoutHandle);
      const entries: GPUBindGroupEntry[] = [];

      // WGPUBindGroupEntry: 40 bytes each (WASM32)
      // offset 0: nextInChain
      // offset 4: binding (uint32)
      // offset 8: buffer (WGPUBuffer handle, 0 if none)
      // offset 12: offset (uint64)
      // offset 20: size (uint64)
      // offset 28: sampler (handle)
      // offset 32: textureView (handle)
      const ENTRY_SIZE = 36; // approximate, may vary
      for (let i = 0; i < entryCount; i++) {
        const ePtr = entriesPtr + i * ENTRY_SIZE;
        const binding = view.getUint32(ePtr + 4, true);
        const bufferHandle = view.getUint32(ePtr + 8, true);
        const offsetLo = view.getUint32(ePtr + 12, true);
        const offsetHi = view.getUint32(ePtr + 16, true);
        const offset = offsetLo + offsetHi * 0x100000000;
        const sizeLo = view.getUint32(ePtr + 20, true);
        const sizeHi = view.getUint32(ePtr + 24, true);
        const size = sizeLo + sizeHi * 0x100000000;

        if (bufferHandle !== 0) {
          const buffer = getHandle<GPUBuffer>(bufferHandle);
          const entry: GPUBindGroupEntry = {
            binding,
            resource: { buffer, offset, size },
          };
          entries.push(entry);
        }
      }

      const bindGroup = device.createBindGroup({ layout, entries });
      return addHandle(bindGroup as unknown as GPUObject);
    },

    wgpuDeviceCreateCommandEncoder(_device: number, _descPtr: number): number {
      const encoder = device.createCommandEncoder();
      return addHandle(encoder as unknown as GPUObject);
    },

    wgpuDeviceGetQueue(_device: number): number {
      return queueHandle;
    },

    wgpuDeviceGetLimits(_device: number, limitsPtr: number): number {
      // Write supported limits to WASM memory
      const view = new DataView(wasmMemory.buffer);
      const limits = device.limits;

      // WGPUSupportedLimits: nextInChain + WGPULimits struct
      // Write key limits that MLX checks
      const limitsOffset = limitsPtr + 4; // skip nextInChain

      // These offsets correspond to the WGPULimits struct fields
      // maxBufferSize is at a specific offset — write the most important ones
      view.setUint32(limitsOffset + 0, limits.maxTextureDimension1D ?? 8192, true);
      view.setUint32(limitsOffset + 4, limits.maxTextureDimension2D ?? 8192, true);

      return 1; // success
    },

    wgpuDeviceSetUncapturedErrorCallback(_device: number, _callbackPtr: number, _userdata: number): void {
      device.onuncapturederror = (event) => {
        console.error('[WebGPU]', event.error);
      };
    },

    wgpuDeviceSetDeviceLostCallback(_device: number, _callbackPtr: number, _userdata: number): void {
      device.lost.then((info) => {
        console.error('[WebGPU] Device lost:', info.message, 'reason:', info.reason);
      });
    },

    wgpuDeviceRelease(_handle: number): void {
      // No-op for pre-created device
    },

    // ===== Queue =====
    wgpuQueueSubmit(_queue: number, count: number, cmdBufArrayPtr: number): void {
      const view = new DataView(wasmMemory.buffer);
      const commandBuffers: GPUCommandBuffer[] = [];
      for (let i = 0; i < count; i++) {
        const handle = view.getUint32(cmdBufArrayPtr + i * 4, true);
        commandBuffers.push(getHandle<GPUCommandBuffer>(handle));
      }
      queue.submit(commandBuffers);
    },

    wgpuQueueOnSubmittedWorkDone(
      _queue: number, callbackPtr: number, userdataPtr: number,
    ): void {
      queue.onSubmittedWorkDone().then(() => {
        // WGPUQueueWorkDoneStatus_Success = 0
        callCallback(callbackPtr, 0, userdataPtr);
      });
    },

    wgpuQueueRelease(_handle: number): void {
      // No-op for pre-created queue
    },

    // ===== Command Encoder =====
    wgpuCommandEncoderBeginComputePass(encoderHandle: number, _descPtr: number): number {
      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      const pass = encoder.beginComputePass();
      return addHandle(pass as unknown as GPUObject);
    },

    wgpuCommandEncoderCopyBufferToBuffer(
      encoderHandle: number,
      srcHandle: number, srcOffset: number,
      dstHandle: number, dstOffset: number,
      size: number,
    ): void {
      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      const src = getHandle<GPUBuffer>(srcHandle);
      const dst = getHandle<GPUBuffer>(dstHandle);
      encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
    },

    wgpuCommandEncoderFinish(encoderHandle: number, _descPtr: number): number {
      const encoder = getHandle<GPUCommandEncoder>(encoderHandle);
      const cmdBuf = encoder.finish();
      return addHandle(cmdBuf as unknown as GPUObject);
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
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      const pipeline = getHandle<GPUComputePipeline>(pipelineHandle);
      pass.setPipeline(pipeline);
    },

    wgpuComputePassEncoderSetBindGroup(
      passHandle: number, groupIndex: number, bgHandle: number,
      dynamicOffsetCount: number, dynamicOffsetsPtr: number,
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

    wgpuComputePassEncoderDispatchWorkgroups(
      passHandle: number, x: number, y: number, z: number,
    ): void {
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      pass.dispatchWorkgroups(x, y, z);
    },

    wgpuComputePassEncoderEnd(passHandle: number): void {
      const pass = getHandle<GPUComputePassEncoder>(passHandle);
      pass.end();
    },

    wgpuComputePassEncoderRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Buffer =====
    wgpuBufferGetSize(bufferHandle: number): number {
      return bufferSizes.get(bufferHandle) ?? 0;
    },

    wgpuBufferGetMappedRange(bufferHandle: number, offset: number, size: number): number {
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      const range = buffer.getMappedRange(offset, size);
      mappedRanges.set(bufferHandle, range);
      // Copy mapped data into WASM memory and return the WASM pointer
      // The C code expects a pointer it can memcpy to/from.
      // We allocate space in WASM memory via the exported malloc.
      // For mappedAtCreation, the C code writes to this pointer, then we
      // copy back on unmap.

      // For now, return a sentinel — the actual implementation needs
      // WASM malloc/free exports to allocate memory. This is a TODO.
      // The C code in utils.h create_buffer_with_data uses mappedAtCreation
      // and writes via memcpy, then calls unmap.

      // Simplified approach: return the range's backing store offset within
      // WASM memory. Since GPUBuffer mapped ranges are NOT in WASM memory,
      // we need to shadow them.
      return 0; // TODO: implement proper memory shadowing
    },

    wgpuBufferGetConstMappedRange(bufferHandle: number, offset: number, size: number): number {
      // Same as getMappedRange for read-only access
      return imports.wgpuBufferGetMappedRange(bufferHandle, offset, size) as number;
    },

    wgpuBufferUnmap(bufferHandle: number): void {
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      buffer.unmap();
      mappedRanges.delete(bufferHandle);
    },

    wgpuBufferMapAsync(
      bufferHandle: number, mode: number,
      offset: number, size: number,
      callbackPtr: number, userdataPtr: number,
    ): void {
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      const gpuMode = mode === 1 ? GPUMapMode.READ : GPUMapMode.WRITE;
      buffer.mapAsync(gpuMode, offset, size).then(
        () => {
          // WGPUBufferMapAsyncStatus_Success = 0
          callCallback(callbackPtr, 0, userdataPtr);
        },
        (err: unknown) => {
          console.error('[WebGPU Bridge] mapAsync failed:', err);
          // WGPUBufferMapAsyncStatus_Error = 1
          callCallback(callbackPtr, 1, userdataPtr);
        },
      );
    },

    wgpuBufferDestroy(bufferHandle: number): void {
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      buffer.destroy();
    },

    wgpuBufferRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Pipeline =====
    wgpuComputePipelineGetBindGroupLayout(pipelineHandle: number, index: number): number {
      const pipeline = getHandle<GPUComputePipeline>(pipelineHandle);
      const layout = pipeline.getBindGroupLayout(index);
      return addHandle(layout as unknown as GPUObject);
    },

    wgpuComputePipelineRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Release functions =====
    wgpuBindGroupRelease(handle: number): void {
      releaseHandle(handle);
    },

    wgpuBindGroupLayoutRelease(handle: number): void {
      releaseHandle(handle);
    },

    wgpuShaderModuleRelease(handle: number): void {
      releaseHandle(handle);
    },

    // ===== Polling (asyncify-aware) =====
    mlx_webgpu_poll(): void {
      // This function is the asyncify yield point.
      // When called from WASM (via asyncify), it returns a Promise
      // that resolves on the next event loop tick, allowing WebGPU
      // async callbacks (mapAsync, onSubmittedWorkDone) to fire.
      //
      // The asyncify runtime handles the suspend/resume automatically.
      // From the WASM perspective, this is a synchronous call that
      // blocks until the Promise resolves.
    },
  };

  return {
    imports,
    adapter,
    device,
    /** Called during WASM instantiation to capture the memory and table references */
    setMemory(memory: WebAssembly.Memory, table: WebAssembly.Table) {
      wasmMemory = memory;
      wasmTable = table;
    },
  } as WebGPUBridge & { setMemory: (m: WebAssembly.Memory, t: WebAssembly.Table) => void };
}
