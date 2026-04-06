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
): BridgeStub {
  const cmdView = new Int32Array(cmdBuffer);
  const cmdDataView = new DataView(cmdBuffer);
  const readbackView = readbackBuffer ? new Uint8Array(readbackBuffer) : null;

  // WASM exports -- set via setInstance() after WASM instantiation
  let wasmTable: WebAssembly.Table;
  let wasmMalloc: (size: number) => number;
  let wasmFree: (ptr: number) => void;
  // Get a fresh Uint8Array view of WASM heap. wasmMemory.buffer always reflects
  // current size after memory.grow() (WebAssembly.Memory getter returns fresh SAB).
  function heap(): Uint8Array {
    return new Uint8Array(wasmMemory.buffer);
  }

  function setInstance(instance: WebAssembly.Instance): void {
    wasmTable = instance.exports.__indirect_function_table as WebAssembly.Table;
    wasmMalloc = instance.exports.malloc as (size: number) => number;
    wasmFree = instance.exports.free as (ptr: number) => void;
  }

  /**
   * Invoke a WASM callback function via the indirect function table.
   */
  function callCallback(fnPtr: number, ...args: number[]): void {
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

  // Active compute pass caching (avoid begin/end per dispatch)
  let activeComputePass = -1;
  let activeComputePassEncoder = -1;

  function flushPendingCompute() {
    if (pendingPipeline >= 0) {
      rpcCall(RpcFn.COMPUTE_PASS_SET_PIPELINE, pendingPass, pendingPipeline);
      pendingPipeline = -1;
    }
    if (pendingBindGroup >= 0) {
      rpcCall(RpcFn.COMPUTE_PASS_SET_BIND_GROUP, pendingPass, 0, pendingBindGroup, 0, 0);
      pendingBindGroup = -1;
    }
  }

  function rpcCall(fnId: number, ...args: number[]): number {
    rpcCount++;
    // Keep last 32 calls in ring buffer
    if (rpcHistory.length >= RPC_HISTORY_SIZE) rpcHistory.shift();
    rpcHistory.push({ n: rpcCount, fn: fnId });

    // Write function ID
    cmdDataView.setUint32(CMD_OFFSET.FN_ID, fnId, true);

    // Write arguments (up to 8 u32 values in ARG0..ARG7)
    for (let i = 0; i < args.length && i < 8; i++) {
      cmdDataView.setUint32(CMD_OFFSET.ARG0 + i * 4, args[i] >>> 0, true);
    }

    // Clear callback count from previous call
    cmdDataView.setUint32(CMD_OFFSET.CALLBACK_COUNT, 0, true);

    // Signal gpu-worker: set PENDING and notify
    Atomics.store(cmdView, STATUS_INDEX, STATUS.PENDING);
    Atomics.notify(cmdView, STATUS_INDEX);

    // Block until gpu-worker sets DONE, with 10s timeout for deadlock detection
    const waitResult = Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING, 10_000);
    if (waitResult === 'timed-out') {
      console.error(`[RPC TIMEOUT] fn=${fnId} (#${rpcCount}) timed out after 10s!`);
      console.error(`[RPC TIMEOUT] Last ${rpcHistory.length} calls:`,
        rpcHistory.map(h => `#${h.n}:fn=${h.fn}`).join(', '));
      console.error(`[RPC TIMEOUT] Status word:`, Atomics.load(cmdView, STATUS_INDEX));
      // Wait indefinitely for recovery (don't break the protocol)
      Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING);
    }

    // Read result
    const result = cmdDataView.getUint32(CMD_OFFSET.RESULT, true);

    // Process pending callbacks (critical for init: adapter/device callbacks)
    try {
      drainCallbacks(fnId);
    } catch (e) {
      console.warn(`[Bridge Stub] callback error for fn=${fnId}:`, e);
    }

    // Reset status for next call
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
    rpcCount++;
    if (rpcHistory.length >= RPC_HISTORY_SIZE) rpcHistory.shift();
    rpcHistory.push({ n: rpcCount, fn: fnId });

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
      console.error(`[RPC TIMEOUT] fn=${fnId} (#${rpcCount}) timed out after 10s! (rpcCallWithHi)`);
      console.error(`[RPC TIMEOUT] Last ${rpcHistory.length} calls:`,
        rpcHistory.map(h => `#${h.n}:fn=${h.fn}`).join(', '));
      console.error(`[RPC TIMEOUT] Status word:`, Atomics.load(cmdView, STATUS_INDEX));
      Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING);
    }

    const result = cmdDataView.getUint32(CMD_OFFSET.RESULT, true);
    try {
      drainCallbacks(fnId);
    } catch (e) {
      console.warn(`[Bridge Stub] callback error for fn=${fnId}:`, e);
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

    wgpuAdapterRelease(): void {
      rpcCall(RpcFn.ADAPTER_RELEASE);
    },

    wgpuAdapterGetProperties(_adapter: number, propsPtr: number): void {
      rpcCall(RpcFn.ADAPTER_GET_PROPERTIES, 0, propsPtr);
    },

    // ===== Device =====
    wgpuDeviceCreateBuffer(_device: number, descPtr: number): number {
      return rpcCall(RpcFn.DEVICE_CREATE_BUFFER, descPtr);
    },

    wgpuDeviceCreateShaderModule(_device: number, descPtr: number): number {
      return rpcCall(RpcFn.DEVICE_CREATE_SHADER_MODULE, descPtr);
    },

    wgpuDeviceCreateComputePipeline(_device: number, descPtr: number): number {
      return rpcCall(RpcFn.DEVICE_CREATE_COMPUTE_PIPELINE, descPtr);
    },

    wgpuDeviceCreateBindGroup(_device: number, descPtr: number): number {
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
      rpcCall(RpcFn.QUEUE_SUBMIT, count, cmdBufArrayPtr);
    },

    wgpuQueueWriteBuffer(
      _queue: number, bufferHandle: number, bufferOffset: bigint,
      dataPtr: number, size: number,
    ): void {
      const offsetLo = Number(bufferOffset & 0xFFFFFFFFn);
      const offsetHi = Number(bufferOffset >> 32n);
      rpcCall(RpcFn.QUEUE_WRITE_BUFFER, bufferHandle, offsetLo, offsetHi, dataPtr, size);
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
      // End compute pass before copy — WebGPU doesn't allow mixing
      if (activeComputePass >= 0 && activeComputePassEncoder === encoderHandle) {
        flushPendingCompute();
        rpcCall(RpcFn.COMPUTE_PASS_END, activeComputePass);
        rpcCall(RpcFn.COMPUTE_PASS_RELEASE, activeComputePass);
        activeComputePass = -1;
        activeComputePassEncoder = -1;
      }
      // 7 u32 args + 1 high-bits word = uses ARG0..ARG7 + ARG0_HI
      const srcOffsetLo = Number(srcOffset & 0xFFFFFFFFn);
      const srcOffsetHi = Number(srcOffset >> 32n);
      const dstOffsetLo = Number(dstOffset & 0xFFFFFFFFn);
      const dstOffsetHi = Number(dstOffset >> 32n);
      const sizeLo = Number(size & 0xFFFFFFFFn);
      const sizeHi = Number(size >> 32n);
      rpcCallWithHi(
        RpcFn.CMD_ENCODER_COPY_BUFFER,
        [encoderHandle, srcHandle, srcOffsetLo, srcOffsetHi, dstHandle, dstOffsetLo, dstOffsetHi, sizeLo],
        { 0: sizeHi }, // ARG0_HI = sizeHi (maps to gpu-worker's arg0Hi)
      );
    },

    wgpuCommandEncoderFinish(encoderHandle: number, _descPtr: number): number {
      // End the cached compute pass before finishing the encoder
      if (activeComputePass >= 0 && activeComputePassEncoder === encoderHandle) {
        flushPendingCompute();
        rpcCall(RpcFn.COMPUTE_PASS_END, activeComputePass);
        rpcCall(RpcFn.COMPUTE_PASS_RELEASE, activeComputePass);
        activeComputePass = -1;
        activeComputePassEncoder = -1;
      }
      return rpcCall(RpcFn.CMD_ENCODER_FINISH, encoderHandle);
    },

    wgpuCommandEncoderRelease(handle: number): void {
      rpcCall(RpcFn.CMD_ENCODER_RELEASE, handle);
    },

    // ===== Command Buffer =====
    wgpuCommandBufferRelease(handle: number): void {
      rpcCall(RpcFn.CMD_BUFFER_RELEASE, handle);
    },

    // ===== Compute Pass Encoder (with auto-fusion) =====
    // Buffer setPipeline + setBindGroup calls, flush them in a single fused RPC
    // when dispatch is called. This reduces 3+ RPC roundtrips to 1.
    wgpuComputePassEncoderSetPipeline(passHandle: number, pipelineHandle: number): void {
      pendingPass = passHandle;
      pendingPipeline = pipelineHandle;
      pendingBindGroup = -1; // Reset bind group — new pipeline needs new bind group
    },

    wgpuComputePassEncoderSetBindGroup(
      passHandle: number, groupIndex: number, bgHandle: number,
      dynamicOffsetCount: number, dynamicOffsetsPtr: number,
    ): void {
      if (groupIndex === 0 && dynamicOffsetCount === 0 && pendingPipeline >= 0) {
        // Buffer for fusion with upcoming dispatch
        pendingBindGroup = bgHandle;
      } else {
        // Non-fusable: flush pending and do individual RPC
        flushPendingCompute();
        rpcCall(RpcFn.COMPUTE_PASS_SET_BIND_GROUP, passHandle, groupIndex, bgHandle, dynamicOffsetCount, dynamicOffsetsPtr);
      }
    },

    wgpuComputePassEncoderDispatchWorkgroups(
      passHandle: number, x: number, y: number, z: number,
    ): void {
      if (pendingPipeline >= 0 && pendingBindGroup >= 0) {
        // Fused: setPipeline + setBindGroup(0) + dispatch in 1 RPC
        rpcCall(RpcFn.FUSED_DISPATCH, pendingPass, pendingPipeline, pendingBindGroup, x, y, z);
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
      // Returns u64 via RESULT + RESULT_HI
      rpcCall(RpcFn.BUFFER_GET_SIZE, bufferHandle);
      // Safe to read after rpcCall: gpu-worker won't touch the buffer until
      // next PENDING, and wasm-worker is single-threaded.
      const lo = cmdDataView.getUint32(CMD_OFFSET.RESULT, true);
      const hi = cmdDataView.getUint32(CMD_OFFSET.RESULT_HI, true);
      return BigInt(lo) | (BigInt(hi) << 32n);
    },

    wgpuBufferGetMappedRange(bufferHandle: number, offset: number, size: number): number {
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
        if (wasmPtr + actualSize > h.byteLength) {
          console.error(`[Bridge Stub] HEAP OVERFLOW: wasmPtr=${wasmPtr} + size=${actualSize} = ${wasmPtr + actualSize} > heap=${h.byteLength}`);
          return wasmPtr; // skip copy to avoid crash
        }
        h.set(readbackView.subarray(0, actualSize), wasmPtr);
      }
      return wasmPtr;
    },

    wgpuBufferUnmap(bufferHandle: number): void {
      rpcCall(RpcFn.BUFFER_UNMAP, bufferHandle);
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
      rpcCall(RpcFn.BUFFER_MAP_ASYNC, bufferHandle, mode, offset, size, callbackPtr, userdataPtr);
    },

    wgpuBufferDestroy(bufferHandle: number): void {
      rpcCall(RpcFn.BUFFER_DESTROY, bufferHandle);
    },

    wgpuBufferRelease(handle: number): void {
      rpcCall(RpcFn.BUFFER_RELEASE, handle);
    },

    // ===== Pipeline =====
    wgpuComputePipelineGetBindGroupLayout(pipelineHandle: number, index: number): number {
      return rpcCall(RpcFn.PIPELINE_GET_BIND_GROUP_LAYOUT, pipelineHandle, index);
    },

    wgpuComputePipelineRelease(handle: number): void {
      rpcCall(RpcFn.PIPELINE_RELEASE, handle);
    },

    // ===== Release =====
    wgpuBindGroupRelease(_handle: number): void {
      // No-op: bind groups are lightweight JS objects, no GPU resources to free.
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
