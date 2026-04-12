/**
 * WebGPU Bridge Stub for WASM-Worker
 *
 * Each stub function writes args to the SharedArrayBuffer command channel,
 * wakes the gpu-worker via Atomics.notify, and blocks via Atomics.wait
 * until the gpu-worker signals completion.
 *
 * This eliminates asyncify entirely: WASM calls these synchronous stubs,
 * they block the wasm-worker's thread, and the gpu-worker (whose event loop
 * is free) handles all async GPU operations.
 *
 * After each RPC call, any pending callbacks are read from the callback ring
 * and invoked via wasmTable.get(fnPtr)(...args).
 */

import {
  RpcFn,
  CMD_OFFSET,
  STATUS,
  STATUS_INDEX,
  CALLBACK_ENTRY_SIZE,
  MAX_CALLBACKS_PER_CALL,
} from './rpc-protocol.js';

export interface BridgeStub {
  imports: Record<string, (...args: any[]) => number | bigint | void>;
  setInstance(instance: WebAssembly.Instance): void;
}

export function createBridgeStub(
  cmdBuffer: SharedArrayBuffer,
  wasmMemory: WebAssembly.Memory,
  /** Pre-created handle values from gpu-worker 'ready' message */
  preCreatedHandles: {
    instanceHandle: number;
    adapterHandle: number;
    deviceHandle: number;
    queueHandle: number;
  },
  /** Dedicated readback buffer for GPU→CPU data */
  readbackBuffer?: SharedArrayBuffer,
  /** GPU feature flags for local resolution (no RPC needed) */
  gpuFeatures?: { shaderF16?: boolean; subgroups?: boolean; timestampQuery?: boolean },
): BridgeStub {
  const cmdView = new Int32Array(cmdBuffer);
  const cmdDataView = new DataView(cmdBuffer);
  const cmdU32 = new Uint32Array(cmdBuffer);  // fast unsigned writes (avoids DataView overhead)
  const readbackView = readbackBuffer ? new Uint8Array(readbackBuffer) : null;

  // WASM exports -- set via setInstance() after WASM instantiation
  let wasmTable: WebAssembly.Table;
  let _wasmMalloc: (size: number) => number;
  let wasmFree: (ptr: number) => void;
  // WASM32 pointers are unsigned 32-bit, but JS coerces WASM i32 returns to
  // signed. When the WASM heap exceeds 2GB, pointers above 0x80000000 become
  // negative JS numbers. Use >>> 0 to reinterpret as unsigned.
  function wasmMalloc(size: number): number {
    return _wasmMalloc(size) >>> 0;
  }
  // Get a fresh Uint8Array view of WASM heap. wasmMemory.buffer always reflects
  // current size after memory.grow() (WebAssembly.Memory getter returns fresh SAB).
  function heap(): Uint8Array {
    return new Uint8Array(wasmMemory.buffer);
  }

  function setInstance(instance: WebAssembly.Instance): void {
    wasmTable = instance.exports.__indirect_function_table as WebAssembly.Table;
    _wasmMalloc = instance.exports.malloc as (size: number) => number;
    wasmFree = instance.exports.free as (ptr: number) => void;
    // Replay callbacks that were queued during _initialize
    for (const { fnPtr, args } of pendingCallbacks) {
      const fn = wasmTable.get(fnPtr) as ((...a: number[]) => void) | null;
      if (fn) fn(...args);
    }
    pendingCallbacks.length = 0;
  }

  /**
   * Invoke a WASM callback function via the indirect function table.
   */
  // Callbacks queued during _initialize (before wasmTable is set via setInstance)
  const pendingCallbacks: Array<{ fnPtr: number; args: number[] }> = [];

  function callCallback(fnPtr: number, ...args: number[]): void {
    if (!wasmTable) {
      // Queue callback for replay after setInstance
      pendingCallbacks.push({ fnPtr, args });
      return;
    }
    const fn = wasmTable.get(fnPtr) as ((...a: number[]) => void) | null;
    if (fn) fn(...args);
  }

  /**
   * Core RPC call: write command, wake gpu-worker, block until done.
   * Returns the RESULT field (u32).
   */
  let rpcCount = 0;
  // Ring buffer of last 32 RPC calls for deadlock diagnosis
  const RPC_HISTORY_SIZE = 32;
  const rpcHistory: Array<{ n: number; fn: number }> = [];


  // Pending compute pass state for fusion (setPipeline + setBindGroup → single dispatch RPC)
  let pendingPass = -1;
  let pendingPipeline = -1;
  let pendingBindGroup = -1;
  let pendingBindGroup1 = -1;

  // Active compute pass caching (avoid begin/end per dispatch)
  let activeComputePass = -1;
  let activeComputePassEncoder = -1;

  // Fused submit state: buffer finish → fuse with submit → skip releases
  let pendingFinishEncoder = -1;
  let pendingFinishPass = -1; // compute pass to end+release in FUSED_SUBMIT
  let lastFakeCmdBuf = -1;
  let lastFusedEncoder = -1;
  let fakeHandleCounter = 0x7F000000; // high range to avoid collision with real handles

  // Buffer size cache: use plain object (faster than Map for integer keys)
  // Caches buffer sizes to avoid BUFFER_GET_SIZE RPCs (~2.5x per dispatch)
  const bufSizes: Record<number, number> = {};

  // Deferred mappedAtCreation buffers: skip getMappedRange + unmap RPCs.
  // Instead of sending 3 RPCs (createBuffer+getMappedRange+unmap), we:
  //   1. Return a FAKE handle (no RPC at all)
  //   2. Allocate WASM shadow locally; getMappedRange returns it (0 RPCs)
  //   3. On unmap, send single CREATE_BUFFER_FROM_DATA RPC (1 RPC)
  // Total: 1 RPC instead of 2 per uniform buffer creation.
  interface MappedAtCreationInfo {
    wasmPtr: number; // WASM shadow pointer (allocated by wasmMalloc)
    size: number;    // buffer size in bytes
    usage: number;   // GPUBufferUsageFlags for deferred creation
  }
  const mappedAtCreationBuffers = new Map<number, MappedAtCreationInfo>();

  // Fake→real handle remapping for deferred buffer creation.
  // After wgpuBufferUnmap sends CREATE_BUFFER_FROM_DATA, the real handle is stored here.
  // All subsequent bridge stub functions that take a buffer handle resolve through this map.
  let fakeBufferCounter = 0x7D000000; // distinct range from fakeHandleCounter (0x7F000000)
  const deferredBufferRemap = new Map<number, number>(); // fake → real handle

  // Buffers awaiting GPU creation (deferred from wgpuBufferUnmap).
  // Data lives in WASM shadow memory until dispatch time.
  interface DeferredCreation {
    usage: number;
    size: number;
    wasmPtr: number;
  }
  const deferredCreations = new Map<number, DeferredCreation>();

  /** Resolve a buffer handle: if it's a deferred fake handle, return the real one. */
  function resolveBufferHandle(handle: number): number {
    return deferredBufferRemap.get(handle) ?? handle;
  }

  /** Force-create a deferred buffer (fallback when it can't be inlined in dispatch). */
  function materializeDeferredBuffer(fakeHandle: number): void {
    const def = deferredCreations.get(fakeHandle);
    if (!def) return;
    deferredCreations.delete(fakeHandle);
    const sizeLo = def.size & 0xFFFFFFFF;
    const sizeHi = Math.floor(def.size / 0x100000000);
    const realHandle = rpcCall(RpcFn.CREATE_BUFFER_FROM_DATA, def.usage, sizeLo, sizeHi, def.wasmPtr);
    deferredBufferRemap.set(fakeHandle, realHandle);
    bufSizes[realHandle] = def.size;
    wasmFree(def.wasmPtr);
  }

  // Eagerly-parsed bind group descriptors for FUSED_FULL_DISPATCH.
  // During wgpuDeviceCreateBindGroup, we read the descriptor from WASM memory
  // (while the C++ stack frame is still valid) and store the parsed entry data.
  // This avoids a separate DEVICE_CREATE_BIND_GROUP RPC — the bind group is
  // created inline on the gpu-worker side during dispatch.
  interface BgEntryData {
    bufferHandle: number;  // already resolved via resolveBufferHandle
    sizeLo: number;
    sizeHi: number;
  }
  interface ParsedBgDesc {
    layoutHandle: number;
    entries: BgEntryData[];
  }
  const pendingBgData = new Map<number, ParsedBgDesc>();
  let fakeBgCounter = 0x7E000000;

  // Pipeline bind group layout cache: layouts are lightweight and never released
  const layoutCache: Record<number, number> = {};

  // Deferred small wgpuQueueWriteBuffer: buffer uniform data locally.
  // When wgpuQueueWriteBuffer is called with <=256 bytes at offset 0,
  // we defer the write and pack it inline into FUSED_DISPATCH_WITH_UNIFORM.
  interface PendingWriteBuffer {
    data: Uint8Array;  // copied from WASM memory (slice, not subarray)
  }
  const pendingWriteBuffers = new Map<number, PendingWriteBuffer>();

  /** Flush all deferred writeBuffer calls via individual RPCs. */
  function flushPendingWrites(): void {
    if (pendingWriteBuffers.size === 0) return;
    for (const [handle, pending] of pendingWriteBuffers) {
      // Materialize deferred buffer if needed
      if (deferredCreations.has(handle)) {
        materializeDeferredBuffer(handle);
      }
      const resolved = resolveBufferHandle(handle);
      // Allocate temporary WASM memory to pass data through RPC
      const wasmPtr = wasmMalloc(pending.data.byteLength);
      heap().set(pending.data, wasmPtr);
      rpcCall(RpcFn.QUEUE_WRITE_BUFFER, resolved, 0, 0, wasmPtr, pending.data.byteLength);
      wasmFree(wasmPtr);
    }
    pendingWriteBuffers.clear();
  }

  /** Materialize a fake bind group handle into a real one via RPC. */
  function materializeBg(handle: number): number {
    const bgDesc = pendingBgData.get(handle);
    if (!bgDesc) return handle; // already real
    pendingBgData.delete(handle);
    // Reconstruct the descriptor in WASM scratch memory and send CREATE_BIND_GROUP RPC
    const ENTRY_SIZE = 40; // WGPUBindGroupEntry is 40 bytes on WASM32
    const descSize = 20;   // WGPUBindGroupDescriptor is 20 bytes on WASM32
    const entriesSize = bgDesc.entries.length * ENTRY_SIZE;
    const scratchPtr = wasmMalloc(descSize + entriesSize);
    const view = new DataView(heap().buffer);
    const entriesArrPtr = scratchPtr + descSize;

    // Write WGPUBindGroupDescriptor
    view.setUint32(scratchPtr + 0, 0, true);  // nextInChain
    view.setUint32(scratchPtr + 4, 0, true);  // label
    view.setUint32(scratchPtr + 8, bgDesc.layoutHandle, true);
    view.setUint32(scratchPtr + 12, bgDesc.entries.length, true);
    view.setUint32(scratchPtr + 16, entriesArrPtr, true);

    // Write WGPUBindGroupEntry array
    for (let i = 0; i < bgDesc.entries.length; i++) {
      const e = bgDesc.entries[i];
      const ePtr = entriesArrPtr + i * ENTRY_SIZE;
      view.setUint32(ePtr + 0, 0, true);   // nextInChain
      view.setUint32(ePtr + 4, i, true);   // binding = sequential index
      view.setUint32(ePtr + 8, e.bufferHandle, true);
      view.setUint32(ePtr + 12, 0, true);  // padding
      view.setUint32(ePtr + 16, 0, true);  // offset lo = 0
      view.setUint32(ePtr + 20, 0, true);  // offset hi = 0
      view.setUint32(ePtr + 24, e.sizeLo, true);
      view.setUint32(ePtr + 28, e.sizeHi, true);
      view.setUint32(ePtr + 32, 0, true);  // sampler
      view.setUint32(ePtr + 36, 0, true);  // textureView
    }

    const realHandle = rpcCall(RpcFn.DEVICE_CREATE_BIND_GROUP, scratchPtr);
    wasmFree(scratchPtr);
    return realHandle;
  }

  function flushPendingCompute() {
    // Flush any deferred writeBuffer calls that weren't inlined into a fused dispatch
    flushPendingWrites();
    if (pendingPipeline >= 0) {
      rpcCall(RpcFn.COMPUTE_PASS_SET_PIPELINE, pendingPass, pendingPipeline);
      pendingPipeline = -1;
    }
    if (pendingBindGroup >= 0) {
      const realBg = materializeBg(pendingBindGroup);
      rpcCall(RpcFn.COMPUTE_PASS_SET_BIND_GROUP, pendingPass, 0, realBg, 0, 0);
      pendingBindGroup = -1;
    }
    if (pendingBindGroup1 >= 0) {
      const realBg1 = materializeBg(pendingBindGroup1);
      rpcCall(RpcFn.COMPUTE_PASS_SET_BIND_GROUP, pendingPass, 1, realBg1, 0, 0);
      pendingBindGroup1 = -1;
    }
  }

  // Uint32Array indices for direct arg writes (element index = byte offset / 4)
  const I_FN = CMD_OFFSET.FN_ID >>> 2;          // 0
  const I_RESULT = CMD_OFFSET.RESULT >>> 2;      // 2
  const I_ARG0 = CMD_OFFSET.ARG0 >>> 2;         // 4
  const I_CB_COUNT = CMD_OFFSET.CALLBACK_COUNT >>> 2; // 16

  function rpcCall(
    fnId: number, a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0,
  ): number {
    // Write function ID + all arguments via Uint32Array (no DataView overhead, no array alloc)
    cmdU32[I_FN] = fnId;
    cmdU32[I_ARG0] = a0 >>> 0;
    cmdU32[I_ARG0 + 1] = a1 >>> 0;
    cmdU32[I_ARG0 + 2] = a2 >>> 0;
    cmdU32[I_ARG0 + 3] = a3 >>> 0;
    cmdU32[I_ARG0 + 4] = a4 >>> 0;
    cmdU32[I_ARG0 + 5] = a5 >>> 0;
    cmdU32[I_ARG0 + 6] = a6 >>> 0;
    // Don't clear CALLBACK_COUNT for FUSED_FULL_DISPATCH / FUSED_DISPATCH_WITH_UNIFORM —
    // it holds entryCount (written by the caller before rpcCall).
    if (fnId !== RpcFn.FUSED_FULL_DISPATCH && fnId !== RpcFn.FUSED_DISPATCH_WITH_UNIFORM) {
      cmdU32[I_CB_COUNT] = 0;
    }

    // Signal gpu-worker and block until done
    Atomics.store(cmdView, STATUS_INDEX, STATUS.PENDING);
    Atomics.notify(cmdView, STATUS_INDEX);
    const waitResult = Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING, 10_000);
    if (waitResult === 'timed-out') {
      rpcCount++;
      if (rpcHistory.length >= RPC_HISTORY_SIZE) rpcHistory.shift();
      rpcHistory.push({ n: rpcCount, fn: fnId });
      const msg = `[RPC TIMEOUT] fn=${fnId} (#${rpcCount}) timed out after 10s! STATUS=${Atomics.load(cmdView, STATUS_INDEX)}`;
      console.error(msg);
      try { (self as any).postMessage({ type: 'error', message: msg }); } catch {}
      // Second attempt with 30s timeout (avoid infinite hang during init)
      const retry = Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING, 30_000);
      if (retry === 'timed-out') {
        console.error(`[RPC FATAL] fn=${fnId} total 40s timeout — returning 0`);
        Atomics.store(cmdView, STATUS_INDEX, STATUS.IDLE);
        return 0;
      }
    }

    const result = cmdU32[I_RESULT];

    // Process pending callbacks only when present (95%+ of calls have 0).
    // Skip for FUSED_FULL_DISPATCH / FUSED_DISPATCH_WITH_UNIFORM — CALLBACK_COUNT
    // holds entryCount, not callbacks.
    if (fnId !== RpcFn.FUSED_FULL_DISPATCH && fnId !== RpcFn.FUSED_DISPATCH_WITH_UNIFORM) {
      if (cmdU32[I_CB_COUNT] > 0) {
        try { drainCallbacks(fnId); } catch (e) { console.warn(`[Bridge Stub] callback error for fn=${fnId}:`, e); }
      }
    }

    Atomics.store(cmdView, STATUS_INDEX, STATUS.IDLE);
    return result;
  }

  /**
   * RPC call that also writes high-bits for u64 arguments.
   * hiArgs maps arg index -> high 32 bits.
   */
  function rpcCallWithHi(
    fnId: number,
    args: number[],
    hiArgs: Record<number, number>,
  ): number {
    cmdDataView.setUint32(CMD_OFFSET.FN_ID, fnId, true);

    for (let i = 0; i < args.length && i < 8; i++) {
      cmdDataView.setUint32(CMD_OFFSET.ARG0 + i * 4, args[i] >>> 0, true);
    }

    // Write high bits for u64 args
    if (0 in hiArgs) cmdDataView.setUint32(CMD_OFFSET.ARG0_HI, hiArgs[0] >>> 0, true);
    if (1 in hiArgs) cmdDataView.setUint32(CMD_OFFSET.ARG1_HI, hiArgs[1] >>> 0, true);
    if (2 in hiArgs) cmdDataView.setUint32(CMD_OFFSET.ARG2_HI, hiArgs[2] >>> 0, true);
    if (3 in hiArgs) cmdDataView.setUint32(CMD_OFFSET.ARG3_HI, hiArgs[3] >>> 0, true);

    cmdDataView.setUint32(CMD_OFFSET.CALLBACK_COUNT, 0, true);

    Atomics.store(cmdView, STATUS_INDEX, STATUS.PENDING);
    Atomics.notify(cmdView, STATUS_INDEX);

    const waitResult = Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING, 10_000);
    if (waitResult === 'timed-out') {
      rpcCount++;
      if (rpcHistory.length >= RPC_HISTORY_SIZE) rpcHistory.shift();
      rpcHistory.push({ n: rpcCount, fn: fnId });
      console.error(`[RPC TIMEOUT] fn=${fnId} (#${rpcCount}) timed out after 10s! (rpcCallWithHi)`);
      console.error(`[RPC TIMEOUT] Last ${rpcHistory.length} calls:`,
        rpcHistory.map(h => `#${h.n}:fn=${h.fn}`).join(', '));
      console.error(`[RPC TIMEOUT] Status word:`, Atomics.load(cmdView, STATUS_INDEX));
      // Second attempt with 30s timeout (avoid infinite hang during init)
      const retry = Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING, 30_000);
      if (retry === 'timed-out') {
        console.error(`[RPC FATAL] fn=${fnId} total 40s timeout — returning 0`);
        Atomics.store(cmdView, STATUS_INDEX, STATUS.IDLE);
        return 0;
      }
    }

    const result = cmdDataView.getUint32(CMD_OFFSET.RESULT, true);
    const cbCount = cmdDataView.getUint32(CMD_OFFSET.CALLBACK_COUNT, true);
    if (cbCount > 0) {
      try { drainCallbacks(fnId); } catch (e) { console.warn(`[Bridge Stub] callback error for fn=${fnId}:`, e); }
    }
    Atomics.store(cmdView, STATUS_INDEX, STATUS.IDLE);

    return result;
  }

  /**
   * Read and invoke pending callbacks from the callback ring.
   *
   * Different functions need different callback signatures:
   * - INSTANCE_REQUEST_ADAPTER: callCallback(fnPtr, 0, adapterHandle, 0, userdataPtr)
   * - ADAPTER_REQUEST_DEVICE:   callCallback(fnPtr, 0, deviceHandle, 0, userdataPtr)
   * - QUEUE_ON_SUBMITTED_WORK_DONE: callCallback(fnPtr, status, userdataPtr)
   * - BUFFER_MAP_ASYNC:         callCallback(fnPtr, status, userdataPtr)
   * - POLL:                     callCallback(fnPtr, status, userdataPtr) for each
   */
  function drainCallbacks(fnId: number): void {
    const count = cmdDataView.getUint32(CMD_OFFSET.CALLBACK_COUNT, true);
    if (count === 0) return;

    const result = cmdDataView.getUint32(CMD_OFFSET.RESULT, true);

    for (let i = 0; i < count; i++) {
      const base = CMD_OFFSET.CALLBACK_BASE + i * CALLBACK_ENTRY_SIZE;
      const fnPtr = cmdDataView.getUint32(base, true);
      const status = cmdDataView.getUint32(base + 4, true);
      const userdataPtr = cmdDataView.getUint32(base + 8, true);

      if (fnPtr === 0) continue;

      switch (fnId) {
        case RpcFn.INSTANCE_REQUEST_ADAPTER:
          // WGPURequestAdapterCallback(WGPURequestAdapterStatus, WGPUAdapter, char* msg, void* ud)
          callCallback(fnPtr, status, result /* adapterHandle */, 0, userdataPtr);
          break;

        case RpcFn.ADAPTER_REQUEST_DEVICE:
          // WGPURequestDeviceCallback(WGPURequestDeviceStatus, WGPUDevice, char* msg, void* ud)
          callCallback(fnPtr, status, result /* deviceHandle */, 0, userdataPtr);
          break;

        default:
          // Generic 2-arg callback: (status, userdata)
          // Used by onSubmittedWorkDone, mapAsync, and POLL-drained callbacks
          callCallback(fnPtr, status, userdataPtr);
          break;
      }
    }
  }

  // Track wasmPtr for each getMappedRange/getConstMappedRange so we can
  // call wasmFree on unmap. Key = buffer handle, value = WASM pointer.
  const unmapPtrs = new Map<number, number>();

  // ---- Import Stubs ----
  // Each matches the signature expected by the WASM module (from webgpu.h).
  // The first parameter of most functions is a handle to the "self" object
  // (e.g., WGPUDevice for wgpuDeviceCreateBuffer). In the RPC architecture,
  // the gpu-worker already knows which device/queue/etc. to use, so we only
  // pass the descriptor pointer or relevant args.

  const imports: Record<string, (...args: any[]) => number | bigint | void> = {
    // ===== Instance =====
    wgpuCreateInstance(_descPtr: number): number {
      return rpcCall(RpcFn.CREATE_INSTANCE);
    },

    wgpuInstanceRequestAdapter(
      _instance: number, _optsPtr: number, callbackPtr: number, userdataPtr: number,
    ): void {
      rpcCall(RpcFn.INSTANCE_REQUEST_ADAPTER, 0, 0, callbackPtr, userdataPtr);
    },

    wgpuInstanceRelease(): void {
      rpcCall(RpcFn.INSTANCE_RELEASE);
    },

    // ===== Adapter =====
    wgpuAdapterRequestDevice(
      _adapter: number, _descPtr: number, callbackPtr: number, userdataPtr: number,
    ): void {
      rpcCall(RpcFn.ADAPTER_REQUEST_DEVICE, 0, 0, callbackPtr, userdataPtr);
    },

    wgpuAdapterGetLimits(_adapterHandle: number, limitsPtr: number): number {
      // The GPU worker already requested the device with adapter limits.
      // For WASM, we write adapter limits to the provided pointer.
      // Use the same RPC as device limits (they're the same in practice for our usage).
      return rpcCall(RpcFn.DEVICE_GET_LIMITS, limitsPtr);
    },

    wgpuAdapterHasFeature(_adapter: number, feature: number): number {
      if (feature === 14 && gpuFeatures?.shaderF16) return 1;  // WGPUFeatureName_ShaderF16
      if (feature === 0x3F1 && gpuFeatures?.subgroups) return 1;  // WGPUFeatureName_Subgroups
      return 0;
    },

    wgpuAdapterRelease(): void {
      rpcCall(RpcFn.ADAPTER_RELEASE);
    },

    wgpuAdapterGetProperties(_adapter: number, propsPtr: number): void {
      rpcCall(RpcFn.ADAPTER_GET_PROPERTIES, 0, propsPtr);
    },

    // ===== Device =====
    wgpuDeviceCreateBuffer(_device: number, descPtr: number): number {
      // WGPUBufferDescriptor (WASM32 layout):
      //   +0  nextInChain (ptr, 4 bytes)
      //   +4  label (ptr, 4 bytes)
      //   +8  usage (u32, 4 bytes)
      //   +12 padding (4 bytes, alignment for u64)
      //   +16 size lo (u32)
      //   +20 size hi (u32)
      //   +24 mappedAtCreation (WGPUBool = u32)
      const h = heap();
      if (h[descPtr + 24]) {
        // mappedAtCreation=true: read usage to determine if we can defer.
        // We can only use writeBuffer on unmap if the buffer has COPY_DST
        // (WGPUBufferUsage_CopyDst = 0x0008). Buffers with only CopySrc must use
        // the original approach (createBuffer with mappedAtCreation intact).
        const view = new DataView(h.buffer, h.byteOffset, h.byteLength);
        const usage = view.getUint32(descPtr + 8, true);
        const COPY_DST = 0x0008;
        if (usage & COPY_DST) {
          // Fully deferred: return a FAKE handle (0 RPCs).
          // The real GPU buffer is created during wgpuBufferUnmap via CREATE_BUFFER_FROM_DATA.
          const sizeLo = view.getUint32(descPtr + 16, true);
          const sizeHi = view.getUint32(descPtr + 20, true);
          const size = sizeLo + sizeHi * 0x100000000;
          const fakeHandle = fakeBufferCounter++;
          // Allocate WASM shadow for C++ to write into via getMappedRange
          const wasmPtr = wasmMalloc(size);
          mappedAtCreationBuffers.set(fakeHandle, { wasmPtr, size, usage });
          bufSizes[fakeHandle] = size;
          return fakeHandle;
        }
        // No COPY_DST: fall through to normal RPC (gpu-worker handles mapping)
      }
      return rpcCall(RpcFn.DEVICE_CREATE_BUFFER, descPtr);
    },

    wgpuDeviceCreateShaderModule(_device: number, descPtr: number): number {
      return rpcCall(RpcFn.DEVICE_CREATE_SHADER_MODULE, descPtr);
    },

    wgpuDeviceCreateComputePipeline(_device: number, descPtr: number): number {
      return rpcCall(RpcFn.DEVICE_CREATE_COMPUTE_PIPELINE, descPtr);
    },

    wgpuDeviceCreateBindGroup(_device: number, descPtr: number): number {
      // Eagerly read descriptor from WASM memory (stack is still valid here).
      // Parse entries and store them for FUSED_FULL_DISPATCH — avoids a separate
      // DEVICE_CREATE_BIND_GROUP RPC when the bind group is used immediately.
      const h = heap();
      const view = new DataView(h.buffer, h.byteOffset, h.byteLength);
      const layoutHandle = view.getUint32(descPtr + 8, true);
      const entryCount = view.getUint32(descPtr + 12, true);
      const entriesPtr = view.getUint32(descPtr + 16, true);

      // Only defer if entry count fits in overflow space (max 10 entries, 12 bytes each)
      if (entryCount <= 10) {
        const entries: BgEntryData[] = [];
        const ENTRY_SIZE = 40; // WGPUBindGroupEntry is 40 bytes on WASM32
        for (let i = 0; i < entryCount; i++) {
          const ePtr = entriesPtr + i * ENTRY_SIZE;
          let bufferHandle = view.getUint32(ePtr + 8, true);
          // Resolve deferred buffer handles immediately (while stack is valid)
          bufferHandle = resolveBufferHandle(bufferHandle);
          const sizeLo = view.getUint32(ePtr + 24, true);
          const sizeHi = view.getUint32(ePtr + 28, true);
          entries.push({ bufferHandle, sizeLo, sizeHi });
        }
        const fake = fakeBgCounter++;
        pendingBgData.set(fake, { layoutHandle, entries });
        return fake;
      }

      // Too many entries — fall through to direct RPC.
      // Still need to patch fake buffer handles in entries.
      if (deferredBufferRemap.size > 0) {
        const ENTRY_SIZE = 40;
        for (let i = 0; i < entryCount; i++) {
          const ePtr = entriesPtr + i * ENTRY_SIZE;
          const bufferHandle = view.getUint32(ePtr + 8, true);
          if (bufferHandle !== 0) {
            const real = deferredBufferRemap.get(bufferHandle);
            if (real !== undefined) {
              view.setUint32(ePtr + 8, real, true);
            }
          }
        }
      }
      return rpcCall(RpcFn.DEVICE_CREATE_BIND_GROUP, descPtr);
    },

    wgpuDeviceCreateCommandEncoder(_device: number, _descPtr: number): number {
      return rpcCall(RpcFn.DEVICE_CREATE_COMMAND_ENCODER);
    },

    wgpuDeviceGetQueue(): number {
      return rpcCall(RpcFn.DEVICE_GET_QUEUE);
    },

    wgpuDeviceGetLimits(_device: number, limitsPtr: number): number {
      return rpcCall(RpcFn.DEVICE_GET_LIMITS, limitsPtr);
    },

    wgpuDeviceSetUncapturedErrorCallback(
      _device: number, _cb: number, _ud: number,
    ): void {
      rpcCall(RpcFn.DEVICE_SET_ERROR_CALLBACK);
    },

    wgpuDeviceSetDeviceLostCallback(
      _device: number, _cb: number, _ud: number,
    ): void {
      rpcCall(RpcFn.DEVICE_SET_LOST_CALLBACK);
    },

    wgpuDeviceRelease(): void {
      rpcCall(RpcFn.DEVICE_RELEASE);
    },

    // ===== Queue =====
    wgpuQueueSubmit(_queue: number, count: number, cmdBufArrayPtr: number): void {
      // Flush any deferred writeBuffer calls before submitting
      flushPendingWrites();
      if (pendingFinishEncoder >= 0) {
        // FUSED: pass_end + pass_release + finish + submit + release in 1 RPC (saves 5 round-trips)
        const passHandle = pendingFinishPass >= 0 ? pendingFinishPass : 0;
        rpcCall(RpcFn.FUSED_SUBMIT, pendingFinishEncoder, passHandle);
        lastFusedEncoder = pendingFinishEncoder;
        pendingFinishEncoder = -1;
        pendingFinishPass = -1;
        return;
      }
      rpcCall(RpcFn.QUEUE_SUBMIT, count, cmdBufArrayPtr);
    },

    wgpuQueueWriteBuffer(
      _queue: number, bufferHandle: number, bufferOffset: bigint,
      dataPtr: number, size: number,
    ): void {
      const resolved = resolveBufferHandle(bufferHandle);
      // Defer small writes at offset 0 — will be packed into FUSED_DISPATCH_WITH_UNIFORM
      if (size <= 256 && bufferOffset === 0n) {
        const h = heap();
        const data = h.slice(dataPtr, dataPtr + size);
        pendingWriteBuffers.set(resolved, { data });
        return;
      }
      // Large or offset writes: send immediately
      const offsetLo = Number(bufferOffset & 0xFFFFFFFFn);
      const offsetHi = Number(bufferOffset >> 32n);
      rpcCall(RpcFn.QUEUE_WRITE_BUFFER, resolved, offsetLo, offsetHi, dataPtr, size);
    },

    wgpuQueueOnSubmittedWorkDone(
      _queue: number, callbackPtr: number, userdataPtr: number,
    ): void {
      rpcCall(RpcFn.QUEUE_ON_SUBMITTED_WORK_DONE, callbackPtr, userdataPtr);
    },

    wgpuQueueRelease(): void {
      rpcCall(RpcFn.QUEUE_RELEASE);
    },

    // ===== Command Encoder =====
    // Cache active compute pass to avoid begin/end per dispatch.
    // C++ creates a new pass for each eval, but we reuse the existing one.
    wgpuCommandEncoderBeginComputePass(encoderHandle: number, _descPtr: number): number {
      if (activeComputePass >= 0 && activeComputePassEncoder === encoderHandle) {
        // Reuse existing pass
        return activeComputePass;
      }
      // End previous pass if it was for a different encoder
      if (activeComputePass >= 0 && activeComputePassEncoder !== encoderHandle) {
        rpcCall(RpcFn.COMPUTE_PASS_END, activeComputePass);
        rpcCall(RpcFn.COMPUTE_PASS_RELEASE, activeComputePass);
        activeComputePass = -1;
      }
      const handle = rpcCall(RpcFn.CMD_ENCODER_BEGIN_COMPUTE_PASS, encoderHandle);
      activeComputePass = handle;
      activeComputePassEncoder = encoderHandle;
      return handle;
    },

    wgpuCommandEncoderCopyBufferToBuffer(
      encoderHandle: number,
      srcHandle: number, srcOffset: bigint,
      dstHandle: number, dstOffset: bigint,
      size: bigint,
    ): void {
      flushPendingCompute();
      const resolvedSrc = resolveBufferHandle(srcHandle);
      const resolvedDst = resolveBufferHandle(dstHandle);

      // For 32-bit offsets/sizes (common in WASM), use fused RPC
      const srcOff32 = Number(srcOffset);
      const dstOff32 = Number(dstOffset);
      const size32 = Number(size);

      if (srcOffset <= 0xFFFFFFFF && dstOffset <= 0xFFFFFFFF && size <= 0xFFFFFFFF
          && activeComputePassEncoder === encoderHandle) {
        // FUSED: end pass (if active) + copy + begin new pass = 1 RPC instead of 3-4
        const passHandle = activeComputePass >= 0 ? activeComputePass : 0;
        const newPassHandle = rpcCall(
          RpcFn.FUSED_COPY_BUFFER,
          encoderHandle, passHandle, resolvedSrc, srcOff32, resolvedDst, dstOff32, size32,
        );
        activeComputePass = newPassHandle;
        activeComputePassEncoder = encoderHandle;
      } else {
        // Fallback: 64-bit offsets or different encoder — use separate RPCs
        if (activeComputePass >= 0 && activeComputePassEncoder === encoderHandle) {
          rpcCall(RpcFn.COMPUTE_PASS_END, activeComputePass);
          activeComputePass = -1;
          activeComputePassEncoder = -1;
        }
        const srcOffsetLo = Number(srcOffset & 0xFFFFFFFFn);
        const srcOffsetHi = Number(srcOffset >> 32n);
        const dstOffsetLo = Number(dstOffset & 0xFFFFFFFFn);
        const dstOffsetHi = Number(dstOffset >> 32n);
        const sizeLo = Number(size & 0xFFFFFFFFn);
        const sizeHi = Number(size >> 32n);
        rpcCallWithHi(
          RpcFn.CMD_ENCODER_COPY_BUFFER,
          [encoderHandle, resolvedSrc, srcOffsetLo, srcOffsetHi, resolvedDst, dstOffsetLo, dstOffsetHi, sizeLo],
          { 0: sizeHi },
        );
      }
    },

    wgpuCommandEncoderFinish(encoderHandle: number, _descPtr: number): number {
      // Buffer everything for FUSED_SUBMIT — pass end, finish, submit all in one RPC
      flushPendingCompute();
      if (activeComputePass >= 0 && activeComputePassEncoder === encoderHandle) {
        pendingFinishPass = activeComputePass;
        activeComputePass = -1;
        activeComputePassEncoder = -1;
      } else {
        pendingFinishPass = -1;
      }
      pendingFinishEncoder = encoderHandle;
      lastFakeCmdBuf = fakeHandleCounter++;
      return lastFakeCmdBuf;
    },

    wgpuCommandEncoderRelease(handle: number): void {
      if (handle === lastFusedEncoder) {
        lastFusedEncoder = -1;
        return; // Already released by FUSED_SUBMIT
      }
      rpcCall(RpcFn.CMD_ENCODER_RELEASE, handle);
    },

    // ===== Command Buffer =====
    wgpuCommandBufferRelease(handle: number): void {
      if (handle === lastFakeCmdBuf) {
        lastFakeCmdBuf = -1;
        return; // Already released by FUSED_SUBMIT
      }
      rpcCall(RpcFn.CMD_BUFFER_RELEASE, handle);
    },

    // ===== Compute Pass Encoder (with auto-fusion) =====
    // Buffer setPipeline + setBindGroup calls, flush them in a single fused RPC
    // when dispatch is called. This reduces 3+ RPC roundtrips to 1.
    wgpuComputePassEncoderSetPipeline(passHandle: number, pipelineHandle: number): void {
      pendingPass = passHandle;
      pendingPipeline = pipelineHandle;
      pendingBindGroup = -1; // Reset bind group — new pipeline needs new bind group
      pendingBindGroup1 = -1;
    },

    wgpuComputePassEncoderSetBindGroup(
      passHandle: number, groupIndex: number, bgHandle: number,
      dynamicOffsetCount: number, dynamicOffsetsPtr: number,
    ): void {
      if (groupIndex === 0 && dynamicOffsetCount === 0 && pendingPipeline >= 0) {
        // Buffer for fusion with upcoming dispatch
        pendingBindGroup = bgHandle;
      } else if (groupIndex === 1 && dynamicOffsetCount === 0 && pendingPipeline >= 0) {
        // Buffer bind group 1 for FUSED_DISPATCH_2BG
        pendingBindGroup1 = bgHandle;
      } else {
        // Non-fusable: flush pending and do individual RPC
        flushPendingCompute();
        rpcCall(RpcFn.COMPUTE_PASS_SET_BIND_GROUP, passHandle, groupIndex, bgHandle, dynamicOffsetCount, dynamicOffsetsPtr);
      }
    },

    wgpuComputePassEncoderDispatchWorkgroups(
      passHandle: number, x: number, y: number, z: number,
    ): void {
      if (pendingPipeline >= 0 && pendingBindGroup >= 0 && pendingBindGroup1 >= 0) {
        // 2 bind groups: materialize any fake handles and use FUSED_DISPATCH_2BG
        flushPendingWrites();
        const realBg0 = materializeBg(pendingBindGroup);
        const realBg1 = materializeBg(pendingBindGroup1);
        rpcCall(RpcFn.FUSED_DISPATCH_2BG, pendingPass, pendingPipeline, realBg0, realBg1, x, y, z);
        pendingPipeline = -1;
        pendingBindGroup = -1;
        pendingBindGroup1 = -1;
      } else if (pendingPipeline >= 0 && pendingBindGroup >= 0) {
        const bgDesc = pendingBgData.get(pendingBindGroup);
        if (bgDesc) {
          // FUSED dispatch: inline bind group creation + dispatch in one RPC
          pendingBgData.delete(pendingBindGroup);
          const entryCount = bgDesc.entries.length;

          // Check if any bind group entry has a pending writeBuffer or deferred creation
          let uniformEntryIdx = -1;
          let uniformData: Uint8Array | null = null;
          let deferredFakeHandle = -1;  // track deferred creation to finalize after RPC
          let deferredInfo: DeferredCreation | undefined;

          for (let i = 0; i < entryCount; i++) {
            const bufHandle = bgDesc.entries[i].bufferHandle;
            // Check pending writeBuffer first (takes priority — latest data)
            const pending = pendingWriteBuffers.get(bufHandle);
            if (pending) {
              uniformEntryIdx = i;
              uniformData = pending.data;
              pendingWriteBuffers.delete(bufHandle);
              // Also check if this buffer needs creation
              deferredInfo = deferredCreations.get(bufHandle);
              if (deferredInfo) {
                deferredFakeHandle = bufHandle;
                deferredCreations.delete(bufHandle);
              }
              break;
            }
            // Check deferred buffer creation (data from mappedAtCreation)
            const def = deferredCreations.get(bufHandle);
            if (def && def.size <= 256) {
              uniformEntryIdx = i;
              uniformData = heap().slice(def.wasmPtr, def.wasmPtr + def.size);
              deferredFakeHandle = bufHandle;
              deferredInfo = def;
              deferredCreations.delete(bufHandle);
              break;
            }
          }

          // Materialize any remaining deferred creations in the bind group
          // that weren't handled as the uniform entry
          for (let i = 0; i < entryCount; i++) {
            const bufHandle = bgDesc.entries[i].bufferHandle;
            if (deferredCreations.has(bufHandle)) {
              materializeDeferredBuffer(bufHandle);
              bgDesc.entries[i].bufferHandle = resolveBufferHandle(bufHandle);
            }
          }

          // Write entry data into the callback ring space BEFORE rpcCall
          cmdU32[I_CB_COUNT] = entryCount;
          for (let i = 0; i < entryCount; i++) {
            const e = bgDesc.entries[i];
            const base = (CMD_OFFSET.CALLBACK_BASE + i * 12) >>> 2;
            // Use 0 for deferred creation entry (gpu-worker will create buffer)
            cmdU32[base] = (i === uniformEntryIdx && deferredInfo) ? 0 : resolveBufferHandle(e.bufferHandle);
            cmdU32[base + 1] = e.sizeLo;
            cmdU32[base + 2] = e.sizeHi;
          }

          if (uniformData && uniformData.byteLength <= 256) {
            // FUSED_DISPATCH_WITH_UNIFORM: pack data + bind group + dispatch in one RPC
            // ARG7 = buffer usage flags (for deferred creation, or 0 if buffer exists)
            const usage = deferredInfo ? deferredInfo.usage : 0;
            cmdU32[CMD_OFFSET.UNIFORM_DATA_SIZE >>> 2] = uniformData.byteLength;
            new Uint8Array(cmdBuffer, CMD_OFFSET.UNIFORM_DATA, uniformData.byteLength).set(uniformData);
            const result = rpcCall(RpcFn.FUSED_DISPATCH_WITH_UNIFORM, pendingPass, pendingPipeline, bgDesc.layoutHandle, x, y, z, uniformEntryIdx);

            // Finalize deferred buffer creation: gpu-worker returns real handle
            if (deferredInfo && deferredFakeHandle >= 0) {
              if (result > 0) {
                deferredBufferRemap.set(deferredFakeHandle, result);
                bufSizes[result] = deferredInfo.size;
              }
              wasmFree(deferredInfo.wasmPtr);
            }
          } else {
            // No inline data or too large — flush and use FUSED_FULL_DISPATCH
            if (deferredInfo && deferredFakeHandle >= 0) {
              // Can't inline — fall back to immediate creation
              const sizeLo = deferredInfo.size & 0xFFFFFFFF;
              const sizeHi = Math.floor(deferredInfo.size / 0x100000000);
              const realHandle = rpcCall(RpcFn.CREATE_BUFFER_FROM_DATA, deferredInfo.usage, sizeLo, sizeHi, deferredInfo.wasmPtr);
              deferredBufferRemap.set(deferredFakeHandle, realHandle);
              bufSizes[realHandle] = deferredInfo.size;
              wasmFree(deferredInfo.wasmPtr);
              // Rewrite the entry with the real handle
              const base = (CMD_OFFSET.CALLBACK_BASE + uniformEntryIdx * 12) >>> 2;
              cmdU32[base] = realHandle;
            }
            flushPendingWrites();
            rpcCall(RpcFn.FUSED_FULL_DISPATCH, pendingPass, pendingPipeline, bgDesc.layoutHandle, x, y, z);
          }
        } else {
          // Real bind group handle — use existing FUSED_DISPATCH
          flushPendingWrites();
          rpcCall(RpcFn.FUSED_DISPATCH, pendingPass, pendingPipeline, pendingBindGroup, x, y, z);
        }
        pendingPipeline = -1;
        pendingBindGroup = -1;
      } else {
        // No pending state or incomplete — flush and dispatch separately
        flushPendingCompute();
        rpcCall(RpcFn.COMPUTE_PASS_DISPATCH, passHandle, x, y, z);
      }
    },

    wgpuComputePassEncoderEnd(passHandle: number): void {
      flushPendingCompute();
      // Don't actually end the pass — keep it alive for reuse.
      // It will be ended when a different encoder creates a pass,
      // or when the encoder is finished.
    },

    wgpuComputePassEncoderRelease(handle: number): void {
      // Don't release — the pass is being reused.
      // It will be released when the encoder finishes.
    },

    // ===== Buffer =====
    wgpuBufferGetSize(bufferHandle: number): bigint {
      // Fast path: return cached size (no RPC) — check both fake and real handle
      const cached = bufSizes[bufferHandle];
      if (cached !== undefined) return BigInt(cached);
      // Slow path: RPC to gpu-worker (resolve fake→real first)
      const resolved = resolveBufferHandle(bufferHandle);
      rpcCall(RpcFn.BUFFER_GET_SIZE, resolved);
      const lo = cmdDataView.getUint32(CMD_OFFSET.RESULT, true);
      const hi = cmdDataView.getUint32(CMD_OFFSET.RESULT_HI, true);
      // Cache (assume sizes < 2^32 for typical buffers)
      bufSizes[bufferHandle] = lo;
      return BigInt(lo) | (BigInt(hi) << 32n);
    },

    wgpuBufferGetMappedRange(bufferHandle: number, offset: number, size: number): number {
      // Fast path: deferred mappedAtCreation buffer — return WASM shadow (no RPC)
      const mapped = mappedAtCreationBuffers.get(bufferHandle);
      if (mapped) {
        unmapPtrs.set(bufferHandle, mapped.wasmPtr);
        return mapped.wasmPtr + offset;
      }

      // WASM memory shadow pattern: allocate WASM-side buffer, pass pointer to gpu-worker.
      // gpu-worker calls getMappedRange, stores the mapping. On unmap, gpu-worker copies
      // WASM->JS ArrayBuffer (write path for mappedAtCreation).
      let actualSize = size;
      if (actualSize === 0) {
        // size=0 means "whole buffer minus offset". Query buffer size first.
        rpcCall(RpcFn.BUFFER_GET_SIZE, bufferHandle);
        const lo = cmdDataView.getUint32(CMD_OFFSET.RESULT, true);
        const hi = cmdDataView.getUint32(CMD_OFFSET.RESULT_HI, true);
        actualSize = lo + hi * 0x100000000 - offset;
      }
      const wasmPtr = wasmMalloc(actualSize);
      if (wasmPtr === 0) {
        console.error('[Bridge Stub] malloc failed for getMappedRange');
        return 0;
      }
      // Track for wasmFree on unmap
      unmapPtrs.set(bufferHandle, wasmPtr);
      return rpcCall(RpcFn.BUFFER_GET_MAPPED_RANGE, bufferHandle, offset, size, wasmPtr);
    },

    wgpuBufferGetConstMappedRange(bufferHandle: number, offset: number, size: number): number {
      // Read path: gpu-worker copies GPU data into readback buffer,
      // then wasm-worker copies from readback buffer to WASM heap.
      // This avoids the growable SharedArrayBuffer issue where the gpu-worker
      // can't see the wasm-worker's memory growth.
      let actualSize = size;
      if (actualSize === 0) {
        rpcCall(RpcFn.BUFFER_GET_SIZE, bufferHandle);
        const lo = cmdDataView.getUint32(CMD_OFFSET.RESULT, true);
        const hi = cmdDataView.getUint32(CMD_OFFSET.RESULT_HI, true);
        actualSize = lo + hi * 0x100000000 - offset;
      }
      const wasmPtr = wasmMalloc(actualSize);
      if (wasmPtr === 0) {
        console.error('[Bridge Stub] malloc failed for getConstMappedRange, size=', actualSize);
        return 0;
      }
      unmapPtrs.set(bufferHandle, wasmPtr);
      // gpu-worker writes to readbackBuffer (NOT wasmMemory)
      rpcCall(RpcFn.BUFFER_GET_CONST_MAPPED_RANGE, bufferHandle, offset, size, 0);
      // Copy from readback buffer to WASM heap via fresh Memory.buffer view.
      // heap() always reflects the current memory size after growth.
      if (readbackView && actualSize > 0) {
        const h = heap();
        h.set(readbackView.subarray(0, actualSize), wasmPtr);
      }
      return wasmPtr;
    },

    wgpuBufferUnmap(bufferHandle: number): void {
      // Fast path: deferred mappedAtCreation buffer — defer GPU buffer creation to dispatch
      // time so it can be folded into FUSED_DISPATCH_WITH_UNIFORM (0 RPCs here).
      const mapped = mappedAtCreationBuffers.get(bufferHandle);
      if (mapped) {
        mappedAtCreationBuffers.delete(bufferHandle);
        if (mapped.size <= 256) {
          // Small buffer (uniform): defer creation to dispatch time
          deferredCreations.set(bufferHandle, {
            usage: mapped.usage,
            size: mapped.size,
            wasmPtr: mapped.wasmPtr,
          });
          // DON'T free wasmPtr — data is needed at dispatch time
        } else {
          // Large buffer: create immediately (won't fit in inline dispatch data)
          const sizeLo = mapped.size & 0xFFFFFFFF;
          const sizeHi = Math.floor(mapped.size / 0x100000000);
          const realHandle = rpcCall(RpcFn.CREATE_BUFFER_FROM_DATA, mapped.usage, sizeLo, sizeHi, mapped.wasmPtr);
          deferredBufferRemap.set(bufferHandle, realHandle);
          bufSizes[realHandle] = mapped.size;
          wasmFree(mapped.wasmPtr);
        }
        unmapPtrs.delete(bufferHandle);
        return;
      }
      rpcCall(RpcFn.BUFFER_UNMAP, resolveBufferHandle(bufferHandle));
      // Free the WASM shadow pointer allocated by getMappedRange/getConstMappedRange
      const wasmPtr = unmapPtrs.get(bufferHandle);
      if (wasmPtr !== undefined) {
        wasmFree(wasmPtr);
        unmapPtrs.delete(bufferHandle);
      }
    },

    wgpuBufferMapAsync(
      bufferHandle: number, mode: number,
      offset: number, size: number,
      callbackPtr: number, userdataPtr: number,
    ): void {
      rpcCall(RpcFn.BUFFER_MAP_ASYNC, resolveBufferHandle(bufferHandle), mode, offset, size, callbackPtr, userdataPtr);
    },

    wgpuBufferDestroy(bufferHandle: number): void {
      // No-op: gpu-worker already skips buffer destroy to avoid "destroyed in submit" errors.
      // Clean up deferred state if present.
      const mapped = mappedAtCreationBuffers.get(bufferHandle);
      if (mapped) {
        mappedAtCreationBuffers.delete(bufferHandle);
        wasmFree(mapped.wasmPtr);
      }
      const def = deferredCreations.get(bufferHandle);
      if (def) {
        deferredCreations.delete(bufferHandle);
        wasmFree(def.wasmPtr);
      }
    },

    wgpuBufferRelease(handle: number): void {
      // Clean up deferred creation if buffer is released before dispatch
      const def = deferredCreations.get(handle);
      if (def) {
        deferredCreations.delete(handle);
        wasmFree(def.wasmPtr);
        return; // no GPU buffer was created, nothing to release
      }
      const resolved = resolveBufferHandle(handle);
      if (resolved !== handle) {
        // Clean up remap entry — buffer is being released
        deferredBufferRemap.delete(handle);
      }
      rpcCall(RpcFn.BUFFER_RELEASE, resolved);
    },

    // ===== Pipeline =====
    wgpuComputePipelineGetBindGroupLayout(pipelineHandle: number, index: number): number {
      const key = pipelineHandle * 4 + index;
      const cached = layoutCache[key];
      if (cached !== undefined) return cached;
      const handle = rpcCall(RpcFn.PIPELINE_GET_BIND_GROUP_LAYOUT, pipelineHandle, index);
      layoutCache[key] = handle;
      return handle;
    },

    wgpuComputePipelineRelease(handle: number): void {
      rpcCall(RpcFn.PIPELINE_RELEASE, handle);
    },

    // ===== Release =====
    wgpuBindGroupRelease(handle: number): void {
      // Clean up eagerly-parsed bind group data if it was never dispatched
      pendingBgData.delete(handle);
      // No RPC: bind groups are lightweight JS objects, no GPU resources to free.
      // Skipping this RPC saves ~28K roundtrips per generation.
    },

    wgpuBindGroupLayoutRelease(_handle: number): void {
      // No-op: layout objects are lightweight, cached by pipeline
    },

    wgpuShaderModuleRelease(_handle: number): void {
      // No-op: shader modules are cached by the pipeline cache
    },

    // ===== Polling =====
    mlx_webgpu_poll(): void {
      // Synchronous GPU wait via RPC. Blocks until gpu-worker's
      // queue.onSubmittedWorkDone() resolves. Any accumulated callbacks
      // (from prior QUEUE_ON_SUBMITTED_WORK_DONE or BUFFER_MAP_ASYNC calls)
      // are flushed and invoked here.
      //
      // The callback ring can only hold MAX_CALLBACKS_PER_CALL (8) entries
      // per RPC round-trip. If more callbacks are pending (e.g., many commits
      // during a single eval), we must loop until all are drained. The POLL
      // result indicates how many pendingCallbacks the gpu-worker had BEFORE
      // flushing -- if more than 8, there are leftovers that need another trip.
      //
      // After the first POLL (which awaits queue.onSubmittedWorkDone), the
      // gpu-worker has moved all gpuDoneCallbacks to pendingCallbacks.
      // Subsequent POLL calls just flush the remaining pendingCallbacks
      // (queue.onSubmittedWorkDone resolves immediately since no new work
      // was submitted).
      //
      // Note: drainCallbacks during POLL can invoke C++ callbacks that trigger
      // new commits (e.g., Fence::update), which add NEW gpuDoneCallbacks via
      // nested fn=22 RPCs. The next POLL iteration picks these up. The loop
      // terminates because each iteration drains a finite number of callbacks
      // and the callback chain is bounded by the computation graph depth.
      let remaining = rpcCall(RpcFn.POLL);
      while (remaining > MAX_CALLBACKS_PER_CALL) {
        remaining = rpcCall(RpcFn.POLL);
      }
    },
  };

  return { imports, setInstance };
}
