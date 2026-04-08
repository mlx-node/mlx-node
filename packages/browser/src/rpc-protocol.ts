/**
 * RPC Protocol for GPU-Worker Architecture
 *
 * Defines the shared-memory command channel between wasm-worker and gpu-worker.
 *
 * wasm-worker (runs WASM, blocks on Atomics.wait):
 *   1. Writes FN_ID + args to SharedArrayBuffer
 *   2. Sets STATUS = PENDING, notifies gpu-worker
 *   3. Atomics.wait on STATUS until DONE
 *   4. Reads RESULT, resets STATUS = IDLE
 *
 * gpu-worker (owns GPUDevice, event loop free):
 *   1. Atomics.waitAsync on STATUS for PENDING
 *   2. Reads FN_ID + args, executes WebGPU call
 *   3. Writes RESULT, sets STATUS = DONE, notifies wasm-worker
 *
 * Callback pattern: gpu-worker cannot call WASM functions. Instead it writes
 * pending callback info into CALLBACK_* fields. After Atomics.wait returns,
 * wasm-worker reads and invokes them via wasmTable.get(fnPtr)(...args).
 */

// ---- Function IDs ----
// Each maps to one webgpu.h C API function (or MLX extension).
// Values are arbitrary but grouped by resource type.

export const enum RpcFn {
  // Instance
  CREATE_INSTANCE = 1,
  INSTANCE_REQUEST_ADAPTER = 2,
  INSTANCE_RELEASE = 3,
  // Adapter
  ADAPTER_REQUEST_DEVICE = 4,
  ADAPTER_RELEASE = 5,
  ADAPTER_GET_PROPERTIES = 6,
  // Device
  DEVICE_CREATE_BUFFER = 10,
  DEVICE_CREATE_SHADER_MODULE = 11,
  DEVICE_CREATE_COMPUTE_PIPELINE = 12,
  DEVICE_CREATE_BIND_GROUP = 13,
  DEVICE_CREATE_COMMAND_ENCODER = 14,
  DEVICE_GET_QUEUE = 15,
  DEVICE_GET_LIMITS = 16,
  DEVICE_SET_ERROR_CALLBACK = 17,
  DEVICE_SET_LOST_CALLBACK = 18,
  DEVICE_RELEASE = 19,
  // Queue
  QUEUE_SUBMIT = 20,
  QUEUE_WRITE_BUFFER = 21,
  QUEUE_ON_SUBMITTED_WORK_DONE = 22,
  QUEUE_RELEASE = 23,
  // Command Encoder
  CMD_ENCODER_BEGIN_COMPUTE_PASS = 30,
  CMD_ENCODER_COPY_BUFFER = 31,
  CMD_ENCODER_FINISH = 32,
  CMD_ENCODER_RELEASE = 33,
  // Command Buffer
  CMD_BUFFER_RELEASE = 34,
  // Compute Pass
  COMPUTE_PASS_SET_PIPELINE = 40,
  COMPUTE_PASS_SET_BIND_GROUP = 41,
  COMPUTE_PASS_DISPATCH = 42,
  COMPUTE_PASS_END = 43,
  COMPUTE_PASS_RELEASE = 44,
  // Buffer
  BUFFER_GET_SIZE = 50,
  BUFFER_GET_MAPPED_RANGE = 51,
  BUFFER_GET_CONST_MAPPED_RANGE = 52,
  BUFFER_UNMAP = 53,
  BUFFER_MAP_ASYNC = 54,
  BUFFER_DESTROY = 55,
  BUFFER_RELEASE = 56,
  // Pipeline
  PIPELINE_GET_BIND_GROUP_LAYOUT = 60,
  PIPELINE_RELEASE = 61,
  // Release
  BIND_GROUP_RELEASE = 70,
  BIND_GROUP_LAYOUT_RELEASE = 71,
  SHADER_MODULE_RELEASE = 72,
  // Polling
  POLL = 80,
  // Special: register externally-created GPU buffer (for weight upload)
  ADD_GPU_BUFFER = 90,
  // Fused compute dispatch: setPipeline + setBindGroup(0) + dispatch in one RPC
  // Args: passHandle, pipelineHandle, bindGroupHandle, dispatchX, dispatchY, dispatchZ
  FUSED_DISPATCH = 91,
  // Fused compute dispatch with 2 bind groups
  // Args: passHandle, pipelineHandle, bg0Handle, bg1Handle, dispatchX, dispatchY
  FUSED_DISPATCH_2BG = 92,
  // Fused encoder finish + queue submit + release encoder & cmdBuf in one RPC
  // Args: encoderHandle, passHandle (0 = no pass to end)
  FUSED_SUBMIT = 93,
  // Fused: createBindGroup(from descPtr) + setPipeline + setBindGroup(0) + dispatch
  // Args: passHandle, pipelineHandle, bgDescPtr, dispatchX, dispatchY, dispatchZ
  FUSED_BG_DISPATCH = 94,
  // Fused: create buffer + write initial data in one RPC (replaces mappedAtCreation pattern)
  // Args: usage, sizeLo, sizeHi, wasmDataPtr
  CREATE_BUFFER_FROM_DATA = 95,
  // Fused: inline bind group creation + setPipeline + setBindGroup(0) + dispatch
  // ARG0-5: passHandle, pipelineHandle, layoutHandle, dispatchX, Y, Z
  // CALLBACK_COUNT: entryCount (repurposed — gpu-worker checks fnId to distinguish)
  // CALLBACK_BASE+: entry data (bufHandle:u32, sizeLo:u32, sizeHi:u32) × entryCount
  FUSED_FULL_DISPATCH = 96,
  // Fused: like FUSED_FULL_DISPATCH but also writes inline uniform data to one buffer
  // ARG0-5: passHandle, pipelineHandle, layoutHandle, dispatchX, Y, Z
  // ARG6: uniformEntryIdx (which bind group entry gets the writeBuffer)
  // CALLBACK_COUNT: entryCount, CALLBACK_BASE+: entries (same as FUSED_FULL_DISPATCH)
  // UNIFORM_DATA_SIZE (offset 188): u32 size of inline uniform data
  // UNIFORM_DATA (offset 192+): the uniform data bytes (up to 256 bytes)
  FUSED_DISPATCH_WITH_UNIFORM = 97,
}

// ---- Command Buffer Layout (SharedArrayBuffer) ----
//
// Fixed-size 512-byte command record. The first 64 bytes hold the function ID,
// status flag, return values, and up to 8 u32 arguments (+ high words for u64).
// Bytes 64..187 hold the callback ring / bind group entries.
// Bytes 188..191 hold the inline uniform data size (for FUSED_DISPATCH_WITH_UNIFORM).
// Bytes 192..447 hold inline uniform data (up to 256 bytes).
// Bytes 448..511 are reserved.
//
// All offsets are byte offsets from the start of the SharedArrayBuffer.

export const CMD_OFFSET = {
  // ---- Core command fields (0..63) ----
  FN_ID: 0,       // u32: RpcFn function ID
  STATUS: 4,      // i32: STATUS.IDLE / PENDING / DONE (Atomics target)
  RESULT: 8,      // u32: return value (low 32 bits)
  RESULT_HI: 12,  // u32: high 32 bits for i64/u64 returns
  ARG0: 16,       // u32: argument 0 (or low bits of u64 arg)
  ARG1: 20,       // u32: argument 1
  ARG2: 24,       // u32: argument 2
  ARG3: 28,       // u32: argument 3
  ARG4: 32,       // u32: argument 4
  ARG5: 36,       // u32: argument 5
  ARG6: 40,       // u32: argument 6
  ARG7: 44,       // u32: argument 7
  ARG0_HI: 48,    // u32: high bits for u64 arg0
  ARG1_HI: 52,    // u32: high bits for u64 arg1
  ARG2_HI: 56,    // u32: high bits for u64 arg2
  ARG3_HI: 60,    // u32: high bits for u64 arg3

  // ---- Callback ring (64..187) ----
  // After POLL or BUFFER_MAP_ASYNC, gpu-worker writes pending callbacks here.
  // Each callback entry is 16 bytes: [fnPtr: u32, status: u32, userdataPtr: u32, _pad: u32]
  // CALLBACK_COUNT tells wasm-worker how many entries to process.
  // For FUSED_FULL_DISPATCH / FUSED_DISPATCH_WITH_UNIFORM, this region holds
  // bind group entries (bufHandle:u32, sizeLo:u32, sizeHi:u32) × entryCount.
  CALLBACK_COUNT: 64,  // u32: number of pending callbacks (0..8) or entryCount
  CALLBACK_BASE: 68,   // start of callback entries (each 16 bytes)
  // Entry i: fnPtr at 68 + i*16, status at 72 + i*16, userdata at 76 + i*16

  // ---- Inline uniform data (188..447) ----
  // Used by FUSED_DISPATCH_WITH_UNIFORM to pack writeBuffer data inline.
  UNIFORM_DATA_SIZE: 188,  // u32: size of inline uniform data (0 = none)
  UNIFORM_DATA: 192,       // start of inline uniform data (up to 256 bytes)

  // ---- Total ----
  TOTAL: 512,
} as const;

// Max callbacks per RPC round-trip
export const MAX_CALLBACKS_PER_CALL = 8;
// Each callback entry: fnPtr(4) + status(4) + userdataPtr(4) + pad(4) = 16 bytes
export const CALLBACK_ENTRY_SIZE = 16;

export const STATUS = {
  IDLE: 0,
  PENDING: 1,
  DONE: 2,
} as const;

// Int32Array index for Atomics operations on STATUS field
export const STATUS_INDEX = CMD_OFFSET.STATUS / 4;

// Dedicated readback buffer for GPU→CPU data transfer.
// The gpu-worker writes mapped GPU data here; the wasm-worker reads it.
// This avoids creating views on the growable WASM SharedArrayBuffer from
// the gpu-worker (whose byteLength may not reflect wasm-worker's memory growth).
export const READBACK_BUFFER_SIZE = 4 * 1024 * 1024; // 4MB

// --- Streaming text channel ---
// SharedArrayBuffer layout for streaming decoded text from WASM worker to main thread:
//   [0..3]  u32: write cursor (byte offset where next write goes)
//   [4..7]  u32: sequence number (incremented on each token, so reader detects updates)
//   [8..N]  utf-8 text data (cumulative decoded text so far)
export const STREAM_HEADER_SIZE = 8; // 2 x u32
export const STREAM_BUFFER_SIZE = 256 * 1024; // 256 KB for streamed text
export const STREAM_TEXT_OFFSET = STREAM_HEADER_SIZE;
