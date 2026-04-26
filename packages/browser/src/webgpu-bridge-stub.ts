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

import { roundUpBucket } from './buffer-bucket.js';
import {
  RpcFn,
  CMD_OFFSET,
  STATUS,
  STATUS_INDEX,
  CALLBACK_ENTRY_SIZE,
  MAX_CALLBACKS_PER_CALL,
  MAX_RELEASE_BATCH,
  STATS_OPCODE_SLOTS,
  STATS_BUFFER_SIZE,
  STATS_EXTRA_POOL_HITS,
  STATS_EXTRA_POOL_MISSES,
  STATS_EXTRA_BG_CACHE_HITS,
  STATS_EXTRA_BG_CACHE_MISSES,
  STATS_EXTRA_UNIFORM_HOT_HITS,
  STATS_EXTRA_UNIFORM_HOT_MISSES,
  STATS_EXTRA_SPIN_HITS,
  STATS_EXTRA_SPIN_MISSES,
  STATS_EXTRA_SPIN_BUDGET,
  DISPATCH_BATCH_BUFFER_SIZE,
  MAX_DISPATCH_BATCH,
  MAX_DISPATCH_BATCH_ENTRIES,
  MAX_DISPATCH_BATCH_UNIFORM,
  DISPATCH_BATCH_HEADER_BYTES,
} from './rpc-protocol.js';

export interface BridgeStats {
  /** Total rpcCall invocations observed since the last reset. */
  rpcCount: number;
  /** Histogram of rpcCall invocations keyed by RpcFn numeric value. */
  byFn: Record<number, number>;
  /** Buffer pool hits: creates served from the free-list with no RPC. */
  poolHits: number;
  /** Buffer pool misses: creates that fell through to a real RPC. */
  poolMisses: number;
  /** DIAG: wgpuDeviceCreateBuffer entry count (all paths). */
  diagCreateAll: number;
  /** DIAG: mapped-at-creation + COPY_DST path (fake handle, deferred). */
  diagCreateMappedCopyDst: number;
  /** DIAG: mapped-at-creation WITHOUT COPY_DST (falls through to RPC). */
  diagCreateMappedNoCopyDst: number;
  /** DIAG: wgpuBufferRelease entry count (all paths). */
  diagReleaseAll: number;
  /** DIAG: release of an unknown handle (no size/usage entry). */
  diagReleaseUnknownHandle: number;
  /** DIAG: release rejected by isPoolable (fake handle, MAP_* usage). */
  diagReleaseUnpoolable: number;
  /** DIAG: total FUSED_* dispatch calls that reached the batch gate. */
  diagBatchAttempt: number;
  /** DIAG: dispatches successfully staged into DISPATCH_BATCH. */
  diagBatchStaged: number;
  /** DIAG: dispatches blocked from batching because deferredInfo was non-null. */
  diagBatchDeferredBlock: number;
  /** DIAG: dispatches where stageDispatchBatchRecord returned false. */
  diagBatchStageRefused: number;
  /**
   * JS-F008: RPCs attempted after the bridge was poisoned by a
   * BUFFER_RELEASE_BATCH fire-and-forget drain timeout. Each one threw
   * through assertNotPoisoned; this counter exists so a post-generation
   * profile readout can surface "N dropped after poison" without
   * instrumenting every caller with try/catch.
   */
  poisonedRpcCount: number;
}

export interface GpuWorkerStats {
  /** Total RPCs handled by the gpu-worker since its last reset. */
  totalRpcs: number;
  /** Histogram keyed by RpcFn numeric value (gpu-worker's view). */
  byFn: Record<number, number>;
  /** Cumulative GPU-worker-side buffer pool hits since its last reset. */
  gpuPoolHits: number;
  /** Cumulative GPU-worker-side buffer pool misses since its last reset. */
  gpuPoolMisses: number;
  /** Task C: bind-group cache hits (reused a cached GPUBindGroup for a dispatch). */
  bindGroupCacheHits: number;
  /** Task C: bind-group cache misses (created a fresh GPUBindGroup for a dispatch). */
  bindGroupCacheMisses: number;
  /** Task E: deferred-uniform dispatches that reused the pipeline's dedicated uniform buffer. */
  uniformHotHits: number;
  /** Task E: deferred-uniform dispatches that had to allocate the pipeline's dedicated uniform buffer for the first time. */
  uniformHotMisses: number;
  /** P6: commandLoop spin attempts that caught the next command before falling back to waitAsync. */
  spinHits: number;
  /** P6: commandLoop spin attempts that exhausted their budget and had to waitAsync. */
  spinMisses: number;
  /** P6: current adaptive spin budget at snapshot time (iterations per dispatch-follow-up). */
  spinBudget: number;
}

export interface BridgeStub {
  imports: Record<string, (...args: any[]) => number | bigint | void>;
  setInstance(instance: WebAssembly.Instance): void;
  /**
   * Phase 0: snapshot the per-stub RPC counter + per-opcode histogram
   * maintained inside rpcCall / rpcCallWithHi. Used by the `?profile=1`
   * demo path (see mlx-worker.ts) to attribute wasm-worker → gpu-worker
   * traffic alongside the C++-side dispatch counters.
   */
  getBridgeStats(): BridgeStats;
  /** Phase 0: zero the counters returned by getBridgeStats. */
  resetBridgeStats(): void;
  /**
   * Phase 0: issue a GET_STATS RPC to the gpu-worker and decode the
   * per-opcode histogram from the SAB. If `resetAfter` is true the
   * gpu-worker zeros its histogram after the readback.
   */
  fetchGpuWorkerStats(resetAfter: boolean): GpuWorkerStats;
}

// Shared pool/batch-stats SAB layout (all Atomics.add).
//   0: poolHits
//   1: poolMisses
//   2: diagCreateAll
//   3: diagCreateMappedCopyDst
//   4: diagCreateMappedNoCopyDst
//   5: diagReleaseAll
//   6: diagReleaseUnknownHandle
//   7: diagReleaseUnpoolable
//   8: diagPoolEvictions
//   9: diagBatchAttempt
//  10: diagBatchStaged
//  11: diagBatchDeferredBlock
//  12: diagBatchStageRefused
export const POOL_STATS_SLOTS = 13;
export const POOL_STATS_SIZE_BYTES = POOL_STATS_SLOTS * 4;

// Task 3: Shared buffer metadata SAB, indexed by real (gpu-worker) handle.
// Each stub lives in its own Worker and therefore has its own JS closure —
// per-stub Map<handle, size|usage> breaks when a handle is created in one
// stub (e.g. main mlx-worker on startup) and released in another (pthread
// workers, which actually drive decode — see the note in fetchGpuWorkerStats).
// Moving the size/usage tables into a SAB makes them visible to every stub
// instantiated against the same wasmMemory, so release can always look up
// the original (size, usage) and route the handle into the local pool
// instead of falling through to queueRelease as "unknown".
//
// Layout (interleaved u32 pairs): [size_0, usage_0, size_1, usage_1, ...].
// Handle 0 is unused (gpu-worker's nextHandle starts at 1). Fake handles
// (>= FAKE_HANDLE_BASE, ~2 GB) do NOT fit in the index range; they remain
// stub-local via fallback Maps because they never cross workers anyway.
//
// Sizing: gpu-worker.ts allocates handles monotonically via `nextHandle++`
// across the lifetime of the session (no recycling after release). A full
// decode of 224 tok at ~120 fresh buffer creates/tok (post-pool-fix) is
// ~27k new handles per gen. 2^20 = 1,048,576 handles = 8 MiB of SAB covers
// ~35 back-to-back generations without overflow; the stub gracefully falls
// back to local Maps for any handle beyond the range.
export const BUFFER_METADATA_MAX_HANDLES = 1 << 20;
export const BUFFER_METADATA_SIZE_BYTES = BUFFER_METADATA_MAX_HANDLES * 8;

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
  /** Shared pool-stats SAB for cross-worker aggregation (nullable). */
  poolStatsBuffer?: SharedArrayBuffer,
  /**
   * Phase 2: dedicated 16 KiB SAB used to pack batched FUSED_*_DISPATCH
   * records for DISPATCH_BATCH. When undefined (or when `batchEnabled` is
   * false) the stub falls back to single-dispatch RPCs.
   */
  dispatchBatchBuffer?: SharedArrayBuffer,
  /** Phase 2: gate DISPATCH_BATCH (default: off). */
  batchEnabled?: boolean,
  /** Phase 2b: which batch buffer to use for DISPATCH_BATCH RPC (default: 0). */
  batchBufferId?: number,
  /**
   * Task 3: shared SAB holding (size, usage) metadata for every real
   * gpu-worker buffer handle. When provided, all stubs sharing the same
   * buffer see a consistent view of size+usage on release, fixing the
   * `unknownHandle=100%` pathology caused by per-stub closure isolation
   * across child pthread workers. When undefined, each stub falls back
   * to the legacy per-instance Record-based cache.
   */
  bufferMetadataBuffer?: SharedArrayBuffer,
  /**
   * JS-F010: dedicated stats SAB (1 KiB = 256 u32 slots) the gpu-worker
   * writes its per-opcode RPC histogram into during GET_STATS. When
   * undefined, `fetchGpuWorkerStats` still returns the scalar `totalRpcs`
   * from the cmd SAB's RESULT field but the per-opcode histogram is empty.
   * Must be the same SAB the gpu-worker received in its init message.
   */
  statsBuffer?: SharedArrayBuffer,
  options?: {
    fusionEnabled?: boolean;
    passCachingEnabled?: boolean;
  },
): BridgeStub {
  const cmdView = new Int32Array(cmdBuffer);
  const cmdDataView = new DataView(cmdBuffer);
  const cmdU32 = new Uint32Array(cmdBuffer); // fast unsigned writes (avoids DataView overhead)
  const readbackView = readbackBuffer ? new Uint8Array(readbackBuffer) : null;
  // JS-F010: typed view over the dedicated stats SAB. GET_STATS readback
  // reads the per-opcode histogram from here rather than from the cmd SAB's
  // CALLBACK_BASE / UNIFORM_DATA / reserved regions. Null when the caller
  // didn't supply one (legacy test harnesses); the histogram is empty in
  // that case but `totalRpcs` still comes back via the RESULT field.
  const statsU32: Uint32Array | null =
    statsBuffer && statsBuffer.byteLength >= STATS_BUFFER_SIZE
      ? new Uint32Array(statsBuffer, 0, STATS_OPCODE_SLOTS)
      : null;

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
  const fusionEnabled = options?.fusionEnabled !== false;
  const passCachingEnabled = options?.passCachingEnabled !== false;
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

  // Phase 0 (?profile=1) observability: histogram of rpcCall invocations
  // keyed by RpcFn numeric value. Incremented at the very top of rpcCall /
  // rpcCallWithHi so it reflects the wasm-worker's view of every RPC that
  // crosses into the gpu-worker. Zero-cost when unused (typed-array +
  // single Int32 increment per call).
  const bridgeFnCounts = new Uint32Array(128);
  let bridgeRpcCount = 0;
  function bumpBridgeStats(fnId: number): void {
    bridgeRpcCount++;
    if (fnId >= 0 && fnId < bridgeFnCounts.length) {
      bridgeFnCounts[fnId]++;
    }
  }

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
  let fakeHandleCounter = 0x7f000000; // high range to avoid collision with real handles

  // Buffer metadata (size + usage) for real gpu-worker handles.
  //
  // Previously stored as per-stub `Record<number, number>`, which broke the
  // release path under emnapi's multi-pthread worker pool: handles created
  // in one stub (e.g. main mlx-worker during weight upload) were released
  // from another stub (pthread worker driving decode), and each stub's
  // private Record was empty from the other stub's perspective → every
  // release fell through as `unknownHandle`, skipping the pool entirely.
  //
  // Task 3 moves this table into a SharedArrayBuffer keyed by the real
  // handle index, so every stub against the same wasmMemory sees the same
  // data. Fake handles (>= FAKE_HANDLE_BASE) never cross stubs (they're
  // minted locally and resolved to real handles before any RPC), so they
  // stay in stub-local fallback Maps to avoid having to size the SAB for
  // the ~2 GB fake-handle index range.
  const bufMetaSab =
    bufferMetadataBuffer && bufferMetadataBuffer.byteLength >= BUFFER_METADATA_SIZE_BYTES
      ? new Uint32Array(bufferMetadataBuffer, 0, BUFFER_METADATA_MAX_HANDLES * 2)
      : null;
  // Fallback / fake-handle storage. Used for handles that don't fit in the
  // SAB index range (fakes) and for test paths that don't supply a SAB.
  const fakeBufSizes: Record<number, number> = {};
  const fakeBufUsages: Record<number, number> = {};

  function setBufferMeta(handle: number, size: number, usage: number): void {
    if (bufMetaSab && handle > 0 && handle < BUFFER_METADATA_MAX_HANDLES) {
      // Interleaved layout: [size_i, usage_i] per handle.
      // Write usage first so any racing reader that sees a non-zero size
      // (the validity sentinel) is guaranteed to also see the matching usage.
      const base = handle * 2;
      Atomics.store(bufMetaSab, base + 1, usage >>> 0);
      Atomics.store(bufMetaSab, base, size >>> 0);
      return;
    }
    fakeBufSizes[handle] = size;
    fakeBufUsages[handle] = usage;
  }

  function getBufferSize(handle: number): number | undefined {
    if (bufMetaSab && handle > 0 && handle < BUFFER_METADATA_MAX_HANDLES) {
      const v = Atomics.load(bufMetaSab, handle * 2);
      return v === 0 ? undefined : v;
    }
    return fakeBufSizes[handle];
  }

  function getBufferUsage(handle: number): number | undefined {
    if (bufMetaSab && handle > 0 && handle < BUFFER_METADATA_MAX_HANDLES) {
      // Gate on size as the validity sentinel — a zero size means the slot
      // was never written (or has been cleared), so there is no usage.
      const base = handle * 2;
      if (Atomics.load(bufMetaSab, base) === 0) return undefined;
      return Atomics.load(bufMetaSab, base + 1);
    }
    return fakeBufUsages[handle];
  }

  function clearBufferMeta(handle: number): void {
    if (bufMetaSab && handle > 0 && handle < BUFFER_METADATA_MAX_HANDLES) {
      // Clear size first so any concurrent reader sees "unwritten" rather
      // than a stale (size, 0-usage) pair.
      const base = handle * 2;
      Atomics.store(bufMetaSab, base, 0);
      Atomics.store(bufMetaSab, base + 1, 0);
      return;
    }
    delete fakeBufSizes[handle];
    delete fakeBufUsages[handle];
  }

  // Buffer pool: recycle real gpu-worker handles by usage+size key.
  // Key format: `${usage}:${size}`. Value: stack of live handles whose
  // BUFFER_RELEASE RPC was suppressed — they stay alive in the gpu-worker.
  // Policy: LIFO reuse (pop hot buffers first), FIFO eviction of the oldest
  // on overflow (shift from the front when the bucket is full).
  const bufferPool = new Map<string, number[]>();
  const POOL_CAP_PER_KEY = 32;

  // Phase 1' release batching: accumulate wgpuBufferRelease handles and
  // flush them in one BUFFER_RELEASE_BATCH RPC instead of one-per-release.
  // BUFFER_RELEASE is ~1006/tok on Qwen3.5-0.8B; batching 32-64 per RPC
  // drops that to ~16-32 RPCs/tok.
  //
  // IMPORTANT: this reference is reassigned inside flushPendingReleases
  // via the detach-and-swap pattern (see the comment there for rationale).
  // Every reader MUST go through this `let` binding, not capture the array
  // by reference in a closure — otherwise re-entrant queueRelease calls
  // (fired from callbacks drained inside flushDispatchBatchInner during a
  // flush) would push into the stale pre-swap array and get their handles
  // silently dropped when the swap clears it.
  let pendingReleases: number[] = [];
  const poolStatsArr = poolStatsBuffer ? new Int32Array(poolStatsBuffer) : null;
  const POOL_STAT_HITS = 0;
  const POOL_STAT_MISSES = 1;
  const POOL_STAT_CREATE_ALL = 2;
  const POOL_STAT_CREATE_MAPPED_COPY_DST = 3;
  const POOL_STAT_CREATE_MAPPED_NO_COPY_DST = 4;
  const POOL_STAT_RELEASE_ALL = 5;
  const POOL_STAT_RELEASE_UNKNOWN = 6;
  const POOL_STAT_RELEASE_UNPOOLABLE = 7;
  const POOL_STAT_EVICTIONS = 8;
  const POOL_STAT_BATCH_ATTEMPT = 9;
  const POOL_STAT_BATCH_STAGED = 10;
  const POOL_STAT_BATCH_DEFERRED_BLOCK = 11;
  const POOL_STAT_BATCH_STAGE_REFUSED = 12;
  function bumpPoolStat(slot: number): void {
    if (poolStatsArr) Atomics.add(poolStatsArr, slot, 1);
  }
  let bufferPoolHits = 0;
  let bufferPoolMisses = 0;
  let diagCreateAll = 0;
  let diagCreateMappedCopyDst = 0;
  let diagCreateMappedNoCopyDst = 0;
  let diagReleaseAll = 0;
  let diagReleaseUnknownHandle = 0;
  let diagReleaseUnpoolable = 0;

  function poolKey(usage: number, size: number): string {
    return `${usage}:${size}`;
  }

  // Fake handles (0x7d000000+) must never enter the pool — they have no
  // corresponding gpu-worker object to recycle.
  const FAKE_HANDLE_BASE = 0x7d000000;

  // MAP_READ=1 and MAP_WRITE=2 buffers have externally-observable mapping
  // state that reuse would break. Only pure GPU-resident buffers are safe.
  const UNSAFE_POOL_USAGE_MASK = 0x0001 | 0x0002; // MAP_READ | MAP_WRITE

  function isPoolable(handle: number, usage: number): boolean {
    if (handle >= FAKE_HANDLE_BASE) return false;
    if (handle <= 0) return false;
    if ((usage & UNSAFE_POOL_USAGE_MASK) !== 0) return false;
    return true;
  }

  // Deferred mappedAtCreation buffers: skip getMappedRange + unmap RPCs.
  // Instead of sending 3 RPCs (createBuffer+getMappedRange+unmap), we:
  //   1. Return a FAKE handle (no RPC at all)
  //   2. Allocate WASM shadow locally; getMappedRange returns it (0 RPCs)
  //   3. On unmap, send single CREATE_BUFFER_FROM_DATA RPC (1 RPC)
  // Total: 1 RPC instead of 2 per uniform buffer creation.
  interface MappedAtCreationInfo {
    wasmPtr: number; // WASM shadow pointer (allocated by wasmMalloc)
    size: number; // buffer size in bytes
    usage: number; // GPUBufferUsageFlags for deferred creation
  }
  const mappedAtCreationBuffers = new Map<number, MappedAtCreationInfo>();

  // Fake→real handle remapping for deferred buffer creation.
  // After wgpuBufferUnmap sends CREATE_BUFFER_FROM_DATA, the real handle is stored here.
  // All subsequent bridge stub functions that take a buffer handle resolve through this map.
  let fakeBufferCounter = 0x7d000000; // distinct range from fakeHandleCounter (0x7F000000)
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
    const sizeLo = def.size & 0xffffffff;
    const sizeHi = Math.floor(def.size / 0x100000000);
    const realHandle = rpcCall(RpcFn.CREATE_BUFFER_FROM_DATA, def.usage, sizeLo, sizeHi, def.wasmPtr);
    deferredBufferRemap.set(fakeHandle, realHandle);
    setBufferMeta(realHandle, def.size, def.usage);
    // Scrub the fake handle's shadow size entry (set when the fake was created)
    // so the metadata table doesn't accumulate stale fake-handle keys.
    clearBufferMeta(fakeHandle);
    wasmFree(def.wasmPtr);
  }

  // Eagerly-parsed bind group descriptors for FUSED_FULL_DISPATCH.
  // During wgpuDeviceCreateBindGroup, we read the descriptor from WASM memory
  // (while the C++ stack frame is still valid) and store the parsed entry data.
  // This avoids a separate DEVICE_CREATE_BIND_GROUP RPC — the bind group is
  // created inline on the gpu-worker side during dispatch.
  interface BgEntryData {
    bufferHandle: number; // already resolved via resolveBufferHandle
    sizeLo: number;
    sizeHi: number;
  }
  interface ParsedBgDesc {
    layoutHandle: number;
    entries: BgEntryData[];
  }
  const pendingBgData = new Map<number, ParsedBgDesc>();
  let fakeBgCounter = 0x7e000000;

  // Pipeline bind group layout cache (JS-F017). Map outperforms Record for
  // integer keys and lets us drop entries cleanly on PIPELINE_RELEASE so the
  // cache tracks pipeline lifetime. Pipelines die with the model so steady-
  // state size is bounded, but clearing on release is correctness-cheap and
  // keeps the cache tight if anyone rebuilds the pipeline cache mid-session.
  // Key = pipelineHandle * 4 + layoutIndex (assumes ≤4 layouts per pipeline,
  // matches current codegen).
  const layoutCache = new Map<number, number>();

  // Deferred small wgpuQueueWriteBuffer: buffer uniform data locally.
  // When wgpuQueueWriteBuffer is called with <=256 bytes at offset 0,
  // we defer the write and pack it inline into FUSED_DISPATCH_WITH_UNIFORM.
  interface PendingWriteBuffer {
    data: Uint8Array; // copied from WASM memory (slice, not subarray)
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
    const descSize = 20; // WGPUBindGroupDescriptor is 20 bytes on WASM32
    const entriesSize = bgDesc.entries.length * ENTRY_SIZE;
    const scratchPtr = wasmMalloc(descSize + entriesSize);
    const view = new DataView(heap().buffer);
    const entriesArrPtr = scratchPtr + descSize;

    // Write WGPUBindGroupDescriptor
    view.setUint32(scratchPtr + 0, 0, true); // nextInChain
    view.setUint32(scratchPtr + 4, 0, true); // label
    view.setUint32(scratchPtr + 8, bgDesc.layoutHandle, true);
    view.setUint32(scratchPtr + 12, bgDesc.entries.length, true);
    view.setUint32(scratchPtr + 16, entriesArrPtr, true);

    // Write WGPUBindGroupEntry array
    for (let i = 0; i < bgDesc.entries.length; i++) {
      const e = bgDesc.entries[i];
      if (deferredCreations.has(e.bufferHandle)) {
        materializeDeferredBuffer(e.bufferHandle);
      }
      const bufferHandle = resolveBufferHandle(e.bufferHandle);
      const ePtr = entriesArrPtr + i * ENTRY_SIZE;
      view.setUint32(ePtr + 0, 0, true); // nextInChain
      view.setUint32(ePtr + 4, i, true); // binding = sequential index
      view.setUint32(ePtr + 8, bufferHandle, true);
      view.setUint32(ePtr + 12, 0, true); // padding
      view.setUint32(ePtr + 16, 0, true); // offset lo = 0
      view.setUint32(ePtr + 20, 0, true); // offset hi = 0
      view.setUint32(ePtr + 24, e.sizeLo, true);
      view.setUint32(ePtr + 28, e.sizeHi, true);
      view.setUint32(ePtr + 32, 0, true); // sampler
      view.setUint32(ePtr + 36, 0, true); // textureView
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
  const I_FN = CMD_OFFSET.FN_ID >>> 2; // 0
  const I_RESULT = CMD_OFFSET.RESULT >>> 2; // 2
  const I_ARG0 = CMD_OFFSET.ARG0 >>> 2; // 4
  const I_ARG0_HI = CMD_OFFSET.ARG0_HI >>> 2; // 12 — rpcCallWithHi u64 high bits
  const I_CB_COUNT = CMD_OFFSET.CALLBACK_COUNT >>> 2; // 16
  const I_CB_BASE = CMD_OFFSET.CALLBACK_BASE >>> 2; // 17

  // Phase 2: DISPATCH_BATCH state. Views over the dedicated batch SAB (if
  // provided) plus a byte cursor / record count. Records are appended via
  // stageDispatchBatchRecord() from the two dispatch sender paths and flushed
  // as a single DISPATCH_BATCH RPC either when the batch fills up or when any
  // non-dispatch RPC is about to fire (via the flush-before-any-other-RPC
  // guard at the top of rpcCall).
  //
  // NOTE 2026-04-16: batching is currently only enabled on the main worker
  // (child emnapi workers pass dispatchBatch=false because concurrent stubs
  // would race on this shared SAB). The main worker's bridge is idle during
  // decode (bridgeRPCs=1/tok measured), so in the current architecture this
  // path accumulates zero hits in the hot path. Diagnostic counters below
  // make that measurable.
  const batchActive = batchEnabled === true && dispatchBatchBuffer !== undefined;
  const batchU32 = batchActive ? new Uint32Array(dispatchBatchBuffer!) : null;
  const batchU8 = batchActive ? new Uint8Array(dispatchBatchBuffer!) : null;
  const batchBufferIdVal = batchBufferId ?? 0;
  let batchCount = 0;
  let batchBytes = 0;
  let diagBatchAttempt = 0;
  let diagBatchStaged = 0;
  let diagBatchDeferredBlock = 0;
  let diagBatchStageRefused = 0;
  function bumpBatchAttempt(): void {
    diagBatchAttempt++;
    bumpPoolStat(POOL_STAT_BATCH_ATTEMPT);
  }
  function bumpBatchStaged(): void {
    diagBatchStaged++;
    bumpPoolStat(POOL_STAT_BATCH_STAGED);
  }
  function bumpBatchDeferredBlock(): void {
    diagBatchDeferredBlock++;
    bumpPoolStat(POOL_STAT_BATCH_DEFERRED_BLOCK);
  }
  function bumpBatchStageRefused(): void {
    diagBatchStageRefused++;
    bumpPoolStat(POOL_STAT_BATCH_STAGE_REFUSED);
  }

  function flushDispatchBatchInner(): void {
    if (batchCount === 0) return;
    // Throw fast if a prior F&F drain timed out. rpcCall below has its own
    // assertNotPoisoned guard, but we check here too so re-entrant call sites
    // (e.g. from flushPendingReleases) surface the poisoning even if they
    // never reach the rpcCall inside.
    assertNotPoisoned('flushDispatchBatchInner');
    const count = batchCount;
    const bytes = batchBytes;
    batchCount = 0;
    batchBytes = 0;
    // Write count/bytes as ARG0/ARG1 then fire. Bypass the flush-at-entry
    // guard by calling rpcCall with DISPATCH_BATCH directly — the guard
    // already skips DISPATCH_BATCH to avoid re-entrant recursion.
    rpcCall(RpcFn.DISPATCH_BATCH, count, bytes, batchBufferIdVal);
  }

  /**
   * Pack one FUSED_*_DISPATCH record (opcode 96 or 97) into the batch SAB.
   * Returns true if staged, false if the caller should issue a single RPC
   * instead (batch disabled, or record too large to fit in the buffer).
   * The record is read directly from the main cmdBuffer — the caller must
   * populate the cmdBuffer fields exactly as for a single-dispatch RPC.
   */
  function stageDispatchBatchRecord(opcode: number): boolean {
    if (!batchActive || !batchU32 || !batchU8) return false;
    const entryCount = cmdU32[I_CB_COUNT] >>> 0;
    if (entryCount > MAX_DISPATCH_BATCH_ENTRIES) return false;
    const uniformSize =
      opcode === RpcFn.FUSED_DISPATCH_WITH_UNIFORM ? cmdU32[CMD_OFFSET.UNIFORM_DATA_SIZE >>> 2] >>> 0 : 0;
    if (uniformSize > MAX_DISPATCH_BATCH_UNIFORM) return false;

    // Capture all dispatch args BEFORE any flush that could overwrite cmdBuffer.
    // The shared cmd SAB is single-slot; rpcCall inside flushDispatchBatchInner
    // clobbers ARG0..ARG6 and CB_COUNT, so we must snapshot them now.
    const passHandle = cmdU32[I_ARG0] >>> 0;
    const pipelineHandle = cmdU32[I_ARG0 + 1] >>> 0;
    const layoutHandle = cmdU32[I_ARG0 + 2] >>> 0;
    const x = cmdU32[I_ARG0 + 3] >>> 0;
    const y = cmdU32[I_ARG0 + 4] >>> 0;
    const z = cmdU32[I_ARG0 + 5] >>> 0;
    const uniformEntryIdx = opcode === RpcFn.FUSED_DISPATCH_WITH_UNIFORM ? cmdU32[I_ARG0 + 6] >>> 0 : 0;

    // Copy entries into a local array before flush — rpcCall clears CB_COUNT
    // and another worker could overwrite the entry region during the yield.
    const entries = new Uint32Array(entryCount * 3);
    for (let i = 0; i < entryCount; i++) {
      const src = I_CB_BASE + i * 3;
      entries[i * 3] = cmdU32[src] >>> 0;
      entries[i * 3 + 1] = cmdU32[src + 1] >>> 0;
      entries[i * 3 + 2] = cmdU32[src + 2] >>> 0;
    }

    // Copy uniform data into a local buffer before flush.
    let uniformData: Uint8Array | null = null;
    if (uniformSize > 0) {
      uniformData = new Uint8Array(uniformSize);
      uniformData.set(new Uint8Array(cmdBuffer, CMD_OFFSET.UNIFORM_DATA, uniformSize));
    }

    const uniformPadded = (uniformSize + 3) & ~3;
    const recordSize = DISPATCH_BATCH_HEADER_BYTES + entryCount * 12 + uniformPadded;
    // Flush first if this record would overflow the batch SAB or exceed the
    // per-flush cap. Note: flushing resets batchBytes/batchCount to 0 so the
    // staged record always starts at offset 0 after a flush.
    if (batchCount >= MAX_DISPATCH_BATCH || batchBytes + recordSize > DISPATCH_BATCH_BUFFER_SIZE) {
      flushDispatchBatchInner();
    }
    let cursor = batchBytes;
    const base = cursor >>> 2;
    batchU32[base + 0] = opcode >>> 0;
    batchU32[base + 1] = passHandle;
    batchU32[base + 2] = pipelineHandle;
    batchU32[base + 3] = layoutHandle;
    batchU32[base + 4] = x;
    batchU32[base + 5] = y;
    batchU32[base + 6] = z;
    batchU32[base + 7] = uniformEntryIdx;
    batchU32[base + 8] = uniformSize;
    batchU32[base + 9] = entryCount;
    cursor += DISPATCH_BATCH_HEADER_BYTES;
    // Entries: (bufHandle, sizeLo, sizeHi) × entryCount
    let entryDst = cursor >>> 2;
    for (let i = 0; i < entryCount; i++) {
      batchU32[entryDst] = entries[i * 3];
      batchU32[entryDst + 1] = entries[i * 3 + 1];
      batchU32[entryDst + 2] = entries[i * 3 + 2];
      entryDst += 3;
    }
    cursor += entryCount * 12;
    // Uniform data (raw bytes) — copy from local snapshot into the batch buffer.
    if (uniformData !== null) {
      batchU8.set(uniformData, cursor);
      cursor += uniformSize;
      // Zero-pad the tail (optional but keeps the SAB deterministic for
      // debugging — the receiver only reads uniformSize bytes regardless).
      const pad = uniformPadded - uniformSize;
      for (let i = 0; i < pad; i++) batchU8[cursor + i] = 0;
      cursor += pad;
    }
    batchBytes = cursor;
    batchCount++;
    return true;
  }

  // Release-batch helpers. Packs pending handles into the UNIFORM_DATA region
  // (192..447 = 64 u32 slots) and fires a single BUFFER_RELEASE_BATCH RPC.
  // MUST be called before any non-batch RPC so the gpu-worker processes the
  // releases in the correct order relative to subsequent creates/dispatches.
  const I_UNIFORM = CMD_OFFSET.UNIFORM_DATA >>> 2; // 48

  // Task B: BUFFER_RELEASE_BATCH fire-and-forget (F&F).
  //
  // BUFFER_RELEASE_BATCH has no return value the bridge cares about — the
  // gpu-worker writes setResult(0) and the caller drops it. So we skip the
  // Atomics.wait round-trip: after staging args + flipping STATUS=PENDING we
  // return immediately, leaving the gpu-worker to process the release batch
  // in the background.
  //
  // The single-slot cmd SAB means only ONE F&F can be in flight at a time,
  // and the bridge MUST drain any prior F&F before it touches cmd SAB again
  // (writes FN_ID / ARG* / UNIFORM_DATA / CALLBACK_COUNT, or flips STATUS).
  //
  // `ffPending` tracks that state. `drainFireAndForget()` waits for the
  // gpu-worker to flip STATUS PENDING -> DONE (exactly what a normal rpcCall
  // wait does), resets STATUS=IDLE to match the post-normal-RPC invariant,
  // and clears the flag. It is cheap when no F&F is in flight (single bool
  // check) so we can call it liberally at every cmd-SAB write site.
  //
  // On timeout, the bridge is poisoned — subsequent RPCs throw. The
  // gpu-worker may still write late DONE into the orphaned SAB slot; a
  // poisoned bridge cannot safely reuse that slot. Forcing STATUS=IDLE and
  // clearing ffPending on a stuck gpu-worker would let the next RPC write
  // fresh FN_ID / ARG* bytes into a slot the gpu-worker might later clobber
  // with a late completion for the hung release, corrupting that RPC.
  let ffPending = false;
  let ffPoisoned = false;
  // JS-F008: diagnostic counter for RPCs attempted after poisoning. Incremented
  // by assertNotPoisoned before it throws so the demo can surface
  // "X RPCs dropped after poison" in the profile readout without depending on
  // try/catch counters in every caller.
  let poisonedRpcCount = 0;
  function drainFireAndForget(): void {
    if (!ffPending) return;
    // Same wait pattern as rpcCall: 10s then 30s retry. If STATUS is already
    // DONE, Atomics.wait returns 'not-equal' immediately — no round-trip.
    let waitResult = Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING, 10_000);
    if (waitResult === 'timed-out') {
      const msg = `[RPC TIMEOUT] BUFFER_RELEASE_BATCH (F&F) drain timed out after 10s`;
      console.error(msg);
      try {
        // JS-F008: keep the legacy `error` post so existing demo error
        // handlers still react (chat UI surfaces the message and tears
        // down the stream). The dedicated `bridge-poisoned` post below
        // only fires on FINAL poisoning so consumers can tell a recoverable
        // timeout (the gpu-worker eventually caught up) from a hard hang
        // (bridge is unusable until the worker restarts).
        (self as any).postMessage({ type: 'error', message: msg });
      } catch {}
      waitResult = Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING, 30_000);
      if (waitResult === 'timed-out') {
        // Fail closed: poison the bridge rather than pretending success.
        // Leave ffPending=true and STATUS=PENDING so any further cmd-SAB
        // writer throws via assertNotPoisoned before touching the SAB.
        const fatalMsg = `[RPC FATAL] BUFFER_RELEASE_BATCH (F&F) total 40s timeout — bridge poisoned, gpu-worker likely hung`;
        console.error(fatalMsg);
        try {
          (self as any).postMessage({ type: 'error', message: fatalMsg });
          // JS-F008: dedicated message type so the demo (and tests) can
          // differentiate "gpu-worker hung, restart required" from "chat
          // errored mid-generation" — both previously posted as {type:'error'}.
          // `reason` names the timing source so a future multi-stage
          // poison path can supply a different value.
          (self as any).postMessage({
            type: 'bridge-poisoned',
            reason: 'buffer-release-batch-timeout',
            message: fatalMsg,
          });
        } catch {}
        ffPoisoned = true;
        return;
      }
    }
    // After drain, STATUS is DONE. Reset to IDLE so the next RPC starts
    // from the same invariant every other call site expects.
    Atomics.store(cmdView, STATUS_INDEX, STATUS.IDLE);
    ffPending = false;
  }

  function assertNotPoisoned(context: string): void {
    if (ffPoisoned) {
      // JS-F008: bump the diagnostic counter BEFORE throwing. The bridge
      // stats fetch exposes poisonedRpcCount so the demo can show
      // "N RPCs dropped after poison" without relying on a try/catch chain
      // in every caller. Throwing stays the default behavior because
      // rpcCall / rpcCallWithHi return handles that callers dereference —
      // silently returning 0 would trade a loud crash for a silent corruption.
      poisonedRpcCount++;
      throw new Error(
        `[bridge] poisoned by prior BUFFER_RELEASE_BATCH timeout — gpu-worker likely hung. Restart worker to recover. (context: ${context})`,
      );
    }
  }

  function flushPendingReleases(): void {
    if (pendingReleases.length === 0) return;
    // Throw fast before drainFireAndForget so we don't try to drain a
    // poisoned state.
    assertNotPoisoned('flushPendingReleases');
    // Ordering invariant: BUFFER_RELEASE_BATCH must be observed by the
    // gpu-worker AFTER any staged DISPATCH_BATCH and AFTER any prior F&F
    // release.
    //
    // (1) Drain prior F&F: only one F&F can be in flight at a time, and
    //     we're about to clobber UNIFORM_DATA + FN_ID + ARG0 for the next
    //     one. Cheap no-op when nothing is in flight.
    drainFireAndForget();
    // (2) Flush any staged DISPATCH_BATCH BEFORE publishing the new F&F
    //     release. This preserves the "dispatches that reference a buffer
    //     must submit before the buffer's handle is returned to the pool"
    //     invariant that rpcCall previously provided for us (its top-of-
    //     function flush guard runs before any non-DISPATCH_BATCH RPC).
    //     Since Task B's F&F path bypasses rpcCall, we must do this by
    //     hand — otherwise a release-batch auto-flush (via queueRelease
    //     at MAX_RELEASE_BATCH, or via the threshold flush inside the
    //     dispatch hot path) publishes the release first, the gpu-worker
    //     pools those handles, and the still-unsubmitted DISPATCH_BATCH
    //     later runs against a handle whose underlying GPUBuffer has been
    //     recycled by a subsequent allocation.
    //     flushDispatchBatchInner() itself uses rpcCall, which drains any
    //     F&F at its top before cmd-SAB writes, so ordering vs. the drain
    //     above stays consistent.
    //
    // Re-entrancy hazard: flushDispatchBatchInner → rpcCall(DISPATCH_BATCH)
    // synchronously calls drainCallbacks on return. Those callbacks can
    // invoke bridge imports (e.g. wgpuBufferRelease via a WASM-side free)
    // which re-enter queueRelease. queueRelease pushes onto pendingReleases
    // and, at MAX_RELEASE_BATCH, RECURSIVELY fires flushPendingReleases.
    //
    // Without the swap pattern below, the recursive flush would:
    //   - write > MAX_RELEASE_BATCH u32s into UNIFORM_DATA (overruns the
    //     64-slot / 256-byte region),
    //   - fire its own F&F (STATUS=PENDING), then return,
    //   - and the outer flush would resume with pendingReleases.length
    //     still held in its `count` local, read undefined for every index
    //     (coerces to 0), clobber UNIFORM_DATA back to zeros, and fire a
    //     second PENDING that silently drops the real inner batch.
    //
    // The fix: detach pendingReleases into a local snapshot BEFORE we
    // touch cmd-SAB. Reassign `pendingReleases` to a fresh empty array
    // first so any re-entrant queueRelease after this point lands in the
    // new array (picked up by a subsequent flush), never in ours.
    if (batchActive && batchCount > 0) {
      flushDispatchBatchInner();
    }
    // (3) Second drainFireAndForget: if a callback invoked during
    //     flushDispatchBatchInner triggered a nested flushPendingReleases
    //     (via queueRelease's MAX_RELEASE_BATCH guard), that nested call
    //     published its OWN F&F (ffPending=true) and returned. We must
    //     drain THAT F&F before clobbering cmd SAB for our publish below,
    //     otherwise STATUS=PENDING writes would race the gpu-worker still
    //     reading the nested batch's UNIFORM_DATA. This is a no-op when
    //     no nested flush fired.
    drainFireAndForget();
    // (4) Early-return if the nested flush drained everything we had. In
    //     the common no-re-entrancy case this is always false.
    if (pendingReleases.length === 0) return;
    // (5) Detach & swap: from here on, any re-entrant queueRelease appends
    //     to a fresh array that the next flush will handle. `batchToPublish`
    //     holds the handles WE own and will publish atomically.
    const batchToPublish = pendingReleases;
    pendingReleases = [];
    // (6) Defensive cap. Normally batchToPublish == pendingReleases.length
    //     which queueRelease caps at MAX_RELEASE_BATCH. This path IS
    //     reachable, though rare: drainCallbacks (invoked synchronously by
    //     rpcCall(DISPATCH_BATCH) inside flushDispatchBatchInner above) can
    //     re-enter wgpuBufferRelease → queueRelease. If many handles arrive
    //     during that callback window AFTER the swap below would have run
    //     but before we actually swap, the outer batchToPublish can grow
    //     past the cap on re-entry through the very call sites that call
    //     flushPendingReleases on the same `pendingReleases` reference.
    //     Publishing more than 64 entries would overrun UNIFORM_DATA, so
    //     push the tail back onto the fresh pendingReleases for the next
    //     flush and publish only the first MAX_RELEASE_BATCH here.
    if (batchToPublish.length > MAX_RELEASE_BATCH) {
      for (let i = MAX_RELEASE_BATCH; i < batchToPublish.length; i++) {
        pendingReleases.push(batchToPublish[i]!);
      }
      batchToPublish.length = MAX_RELEASE_BATCH;
    }
    const publishCount = batchToPublish.length;
    // Count the F&F release in the bridge histogram — keeps stats parity
    // with the old rpcCall(BUFFER_RELEASE_BATCH) path so diagnostics don't
    // spuriously drop after this optimization lands.
    bumpBridgeStats(RpcFn.BUFFER_RELEASE_BATCH);
    for (let i = 0; i < publishCount; i++) {
      cmdU32[I_UNIFORM + i] = batchToPublish[i]! >>> 0;
    }
    cmdU32[I_FN] = RpcFn.BUFFER_RELEASE_BATCH;
    cmdU32[I_ARG0] = publishCount >>> 0;
    // Match rpcCall's "don't leak entry counts into next call" invariant —
    // BUFFER_RELEASE_BATCH produces no callbacks, but flushCallbacks() on
    // the gpu-worker writes I_CB_COUNT=0 when done, and a drain that lands
    // on STATUS=DONE will see that value. Clearing here keeps the SAB in a
    // known state even if drain aborts early on timeout.
    cmdU32[I_CB_COUNT] = 0;
    Atomics.store(cmdView, STATUS_INDEX, STATUS.PENDING);
    Atomics.notify(cmdView, STATUS_INDEX);
    ffPending = true;
    // DO NOT wait — fire-and-forget. The next cmd-SAB-write site drains.
  }
  function queueRelease(handle: number): void {
    pendingReleases.push(handle);
    if (pendingReleases.length >= MAX_RELEASE_BATCH) {
      flushPendingReleases();
    }
  }

  function rpcCall(fnId: number, a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0): number {
    // Throw fast before any cmd-SAB writes if a prior F&F drain timed out.
    // Must come BEFORE drainFireAndForget so we don't try to drain a poisoned
    // state.
    assertNotPoisoned(`rpcCall(fn=${fnId})`);
    // Phase 2: flush any staged dispatch batch before *any* other RPC so
    // the gpu-worker observes the dispatches in program order relative to
    // subsequent BUFFER_RELEASE_BATCH / QUEUE_WRITE_BUFFER / FUSED_SUBMIT /
    // mapAsync / new-buffer-create calls. DISPATCH_BATCH itself is exempt
    // (re-entrancy guard). FUSED_FULL_DISPATCH / FUSED_DISPATCH_WITH_UNIFORM
    // are also exempt because the hot dispatch path in
    // wgpuComputePassEncoderDispatchWorkgroups tries to stage into the batch
    // first and only falls through to rpcCall when staging is refused. In
    // that fallback case the hot path must flush any pending batch *before*
    // invoking rpcCall so the fallback dispatch does not reorder ahead of
    // earlier staged dispatches — see the two call sites below.
    if (
      batchActive &&
      batchCount > 0 &&
      fnId !== RpcFn.DISPATCH_BATCH &&
      fnId !== RpcFn.FUSED_FULL_DISPATCH &&
      fnId !== RpcFn.FUSED_DISPATCH_WITH_UNIFORM
    ) {
      flushDispatchBatchInner();
    }
    // Flush any queued releases before *other* RPCs so the gpu-worker
    // observes them before the next dispatch/create. Skip for:
    //   - BUFFER_RELEASE_BATCH itself (re-entrancy guard — flushPendingReleases
    //     calls rpcCall with this fnId)
    //   - FUSED_DISPATCH_WITH_UNIFORM: the caller has already written inline
    //     uniform data into the UNIFORM_DATA region. flushPendingReleases
    //     would trample those bytes, silently corrupting the dispatch.
    //     The dispatch hot path (wgpuComputePassEncoderDispatchWorkgroups)
    //     explicitly flushes at the top instead.
    //   - DISPATCH_BATCH: a batched dispatch RPC ordering-wise sits exactly
    //     where the individual FUSED_*_DISPATCH calls used to. Release
    //     batching already handles inline-data collision via the same
    //     per-dispatch flush gate at the top of the hot path, so this one
    //     piggybacks on that. (DISPATCH_BATCH does not write to UNIFORM_DATA
    //     in the main cmdBuffer — its uniform bytes live in the dedicated
    //     dispatchBatchBuffer.)
    if (
      pendingReleases.length > 0 &&
      fnId !== RpcFn.BUFFER_RELEASE_BATCH &&
      fnId !== RpcFn.FUSED_DISPATCH_WITH_UNIFORM &&
      fnId !== RpcFn.DISPATCH_BATCH
    ) {
      flushPendingReleases();
    }
    // Task B: drain any F&F BUFFER_RELEASE_BATCH before clobbering cmd SAB.
    // flushPendingReleases() above may have fired F&F; flushDispatchBatchInner
    // and earlier call sites may also have left ffPending=true. This drain is
    // a cheap no-op when ffPending=false and must stay BELOW the flushes (so
    // those flushes can themselves issue a new F&F without pre-draining) and
    // ABOVE the cmd-SAB writes (so we don't race the gpu-worker still reading
    // UNIFORM_DATA from the prior release batch).
    drainFireAndForget();
    // Phase 0: exclude GET_STATS from the bridge histogram so a profile
    // readback does not skew its own numbers.
    if (fnId !== RpcFn.GET_STATS) {
      bumpBridgeStats(fnId);
    }
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
    // JS-F010: GET_STATS no longer repurposes CALLBACK_COUNT — histogram
    // data lives in the dedicated stats SAB. Clearing it here is now safe
    // (and matches every other non-fused RPC).
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
      try {
        (self as any).postMessage({ type: 'error', message: msg });
      } catch {}
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
    // JS-F010: GET_STATS used to also be excluded here because it clobbered
    // CALLBACK_COUNT. The dedicated stats SAB makes that moot — CALLBACK_COUNT
    // is zero after the call, so the fast-path check naturally short-circuits.
    if (fnId !== RpcFn.FUSED_FULL_DISPATCH && fnId !== RpcFn.FUSED_DISPATCH_WITH_UNIFORM) {
      if (cmdU32[I_CB_COUNT] > 0) {
        try {
          drainCallbacks(fnId);
        } catch (e) {
          console.warn(`[Bridge Stub] callback error for fn=${fnId}:`, e);
        }
      }
    }

    Atomics.store(cmdView, STATUS_INDEX, STATUS.IDLE);
    return result;
  }

  /**
   * RPC call that also writes high-bits for u64 arguments.
   * hiArgs maps arg index -> high 32 bits.
   */
  function rpcCallWithHi(fnId: number, args: number[], hiArgs: Record<number, number>): number {
    // Throw fast before any cmd-SAB writes if a prior F&F drain timed out.
    // Must come BEFORE drainFireAndForget so we don't try to drain a poisoned
    // state.
    assertNotPoisoned(`rpcCallWithHi(fn=${fnId})`);
    // Phase 2: same ordering guard as rpcCall(). rpcCallWithHi is used by
    // wgpuCommandEncoderCopyBufferToBuffer and friends when an argument
    // needs the u64 high-bits slot — e.g. copies against 64-bit offsets or
    // sizes. Without this flush, a staged dispatch batch followed by such
    // a copy would let the copy RPC overtake the pending dispatches. No
    // opcode exemption is needed: rpcCallWithHi is never used for
    // DISPATCH_BATCH / FUSED_*_DISPATCH / BUFFER_RELEASE_BATCH.
    if (batchActive && batchCount > 0) {
      flushDispatchBatchInner();
    }
    if (pendingReleases.length > 0) {
      flushPendingReleases();
    }
    // Task B: drain F&F before clobbering cmd SAB. See matching comment in rpcCall.
    drainFireAndForget();
    bumpBridgeStats(fnId);
    // JS-F018: use direct Uint32Array writes (cmdU32) to match rpcCall.
    // DataView.setUint32 checks endianness on every call; Uint32Array is a
    // plain integer store. Also keeps both call paths converging so a schema
    // change only has to touch one set of constants.
    cmdU32[I_FN] = fnId;

    const argCount = args.length < 8 ? args.length : 8;
    for (let i = 0; i < argCount; i++) {
      cmdU32[I_ARG0 + i] = args[i] >>> 0;
    }

    // Write high bits for u64 args. I_ARG0_HI through I_ARG0_HI+3 are
    // contiguous element indices matching CMD_OFFSET.ARG0_HI..ARG3_HI.
    if (0 in hiArgs) cmdU32[I_ARG0_HI] = hiArgs[0] >>> 0;
    if (1 in hiArgs) cmdU32[I_ARG0_HI + 1] = hiArgs[1] >>> 0;
    if (2 in hiArgs) cmdU32[I_ARG0_HI + 2] = hiArgs[2] >>> 0;
    if (3 in hiArgs) cmdU32[I_ARG0_HI + 3] = hiArgs[3] >>> 0;

    cmdU32[I_CB_COUNT] = 0;

    Atomics.store(cmdView, STATUS_INDEX, STATUS.PENDING);
    Atomics.notify(cmdView, STATUS_INDEX);

    const waitResult = Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING, 10_000);
    if (waitResult === 'timed-out') {
      rpcCount++;
      if (rpcHistory.length >= RPC_HISTORY_SIZE) rpcHistory.shift();
      rpcHistory.push({ n: rpcCount, fn: fnId });
      console.error(`[RPC TIMEOUT] fn=${fnId} (#${rpcCount}) timed out after 10s! (rpcCallWithHi)`);
      console.error(
        `[RPC TIMEOUT] Last ${rpcHistory.length} calls:`,
        rpcHistory.map((h) => `#${h.n}:fn=${h.fn}`).join(', '),
      );
      console.error(`[RPC TIMEOUT] Status word:`, Atomics.load(cmdView, STATUS_INDEX));
      // Second attempt with 30s timeout (avoid infinite hang during init)
      const retry = Atomics.wait(cmdView, STATUS_INDEX, STATUS.PENDING, 30_000);
      if (retry === 'timed-out') {
        console.error(`[RPC FATAL] fn=${fnId} total 40s timeout — returning 0`);
        Atomics.store(cmdView, STATUS_INDEX, STATUS.IDLE);
        return 0;
      }
    }

    const result = cmdU32[I_RESULT];
    const cbCount = cmdU32[I_CB_COUNT];
    if (cbCount > 0) {
      try {
        drainCallbacks(fnId);
      } catch (e) {
        console.warn(`[Bridge Stub] callback error for fn=${fnId}:`, e);
      }
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

    wgpuInstanceRequestAdapter(_instance: number, _optsPtr: number, callbackPtr: number, userdataPtr: number): void {
      rpcCall(RpcFn.INSTANCE_REQUEST_ADAPTER, 0, 0, callbackPtr, userdataPtr);
    },

    wgpuInstanceRelease(): void {
      rpcCall(RpcFn.INSTANCE_RELEASE);
    },

    // ===== Adapter =====
    wgpuAdapterRequestDevice(_adapter: number, _descPtr: number, callbackPtr: number, userdataPtr: number): void {
      rpcCall(RpcFn.ADAPTER_REQUEST_DEVICE, 0, 0, callbackPtr, userdataPtr);
    },

    wgpuAdapterGetLimits(_adapterHandle: number, limitsPtr: number): number {
      // The GPU worker already requested the device with adapter limits.
      // For WASM, we write adapter limits to the provided pointer.
      // Use the same RPC as device limits (they're the same in practice for our usage).
      return rpcCall(RpcFn.DEVICE_GET_LIMITS, limitsPtr);
    },

    wgpuAdapterHasFeature(_adapter: number, feature: number): number {
      if (feature === 14 && gpuFeatures?.shaderF16) return 1; // WGPUFeatureName_ShaderF16
      if (feature === 0x3f1 && gpuFeatures?.subgroups) return 1; // WGPUFeatureName_Subgroups
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
      diagCreateAll++;
      bumpPoolStat(POOL_STAT_CREATE_ALL);
      // WGPUBufferDescriptor (WASM32 layout):
      //   +0  nextInChain (ptr, 4 bytes)
      //   +4  label (ptr, 4 bytes)
      //   +8  usage (u32, 4 bytes)
      //   +12 padding (4 bytes, alignment for u64)
      //   +16 size lo (u32)
      //   +20 size hi (u32)
      //   +24 mappedAtCreation (WGPUBool = u32)
      //
      // Parse the descriptor ONCE at function entry. Prior revisions parsed
      // it inline in each branch, which drifted — one branch read `size` as
      // a full 64-bit value while another truncated to 32 bits, producing
      // wrong bucketSize RPCs for buffers > 4 GiB (see commit 7c4f451).
      // Read the full 64-bit size (sizeLo + sizeHi); the pool invariant
      // physical >= bucketSize relies on this being accurate. mappedAtCreation
      // is a WGPUBool (u32), so read it through the DataView for consistency.
      const h = heap();
      const view = new DataView(h.buffer, h.byteOffset, h.byteLength);
      const usage = view.getUint32(descPtr + 8, true);
      const sizeLo = view.getUint32(descPtr + 16, true);
      const sizeHi = view.getUint32(descPtr + 20, true);
      const size = sizeLo + sizeHi * 0x100000000;
      const mappedAtCreation = view.getUint32(descPtr + 24, true) !== 0;

      if (mappedAtCreation) {
        // mappedAtCreation=true: check usage to determine if we can defer.
        // We can only use writeBuffer on unmap if the buffer has COPY_DST
        // (WGPUBufferUsage_CopyDst = 0x0008). Buffers with only CopySrc must use
        // the original approach (createBuffer with mappedAtCreation intact).
        const COPY_DST = 0x0008;
        if (usage & COPY_DST) {
          diagCreateMappedCopyDst++;
          bumpPoolStat(POOL_STAT_CREATE_MAPPED_COPY_DST);
          // Fully deferred: return a FAKE handle (0 RPCs).
          // The real GPU buffer is created during wgpuBufferUnmap via CREATE_BUFFER_FROM_DATA.
          const fakeHandle = fakeBufferCounter++;
          // Allocate WASM shadow for C++ to write into via getMappedRange
          const wasmPtr = wasmMalloc(size);
          mappedAtCreationBuffers.set(fakeHandle, { wasmPtr, size, usage });
          // Fake handles (>= FAKE_HANDLE_BASE) are stub-local and outside the
          // SAB index range, so stash size in the fake map directly. They are
          // not poolable, so usage is tracked via mappedAtCreationBuffers.
          fakeBufSizes[fakeHandle] = size;
          return fakeHandle;
        }
        diagCreateMappedNoCopyDst++;
        bumpPoolStat(POOL_STAT_CREATE_MAPPED_NO_COPY_DST);
        // No COPY_DST: fall through to normal RPC (gpu-worker handles mapping)
      }

      if (!mappedAtCreation && size > 0) {
        // Bucket the stub pool so that creates with slightly different
        // logical sizes (e.g., 513 vs 512 bytes) can reuse each other.
        // Use the SAME roundUpBucket as the gpu-worker — invariant: every
        // handle in bucket B has physical allocation >= B (because every
        // MISS below allocates exactly bucketSize via arg1).
        const bucketSize = roundUpBucket(size);
        const key = poolKey(usage, bucketSize);
        const stack = bufferPool.get(key);
        if (stack !== undefined && stack.length > 0) {
          const reused = stack.pop()!;
          // Drop empty buckets so bufferPool doesn't accumulate dead keys.
          if (stack.length === 0) bufferPool.delete(key);
          bufferPoolHits++;
          bumpPoolStat(POOL_STAT_HITS);
          // Publish the caller's LOGICAL size via setBufferMeta — the stub
          // fast path for wgpuBufferGetSize reads this and must return the
          // size the caller actually asked for, not the bucket size.
          setBufferMeta(reused, size, usage);
          return reused;
        }
        bufferPoolMisses++;
        bumpPoolStat(POOL_STAT_MISSES);
        // Pool miss: tell the gpu-worker to allocate the full bucketSize so
        // the resulting handle can later be reused by any request that
        // bucket-maps to the same bin. This is the piece that keeps the
        // stub pool safe to pop-and-return without re-creating.
        const h2 = rpcCall(RpcFn.DEVICE_CREATE_BUFFER, descPtr, bucketSize);
        if (h2 > 0 && h2 < FAKE_HANDLE_BASE) {
          // Eager (size, usage) cache — drops BUFFER_GET_SIZE RPCs and gives
          // the release path the data it needs to pool instead of falling
          // through to queueRelease. Store LOGICAL size in the stub cache
          // because that is what C++ callers expect from wgpuBufferGetSize.
          setBufferMeta(h2, size, usage);
        }
        return h2;
      }

      // mappedAtCreation (with or without COPY_DST falling through to RPC):
      // do NOT bucket. getMappedRange must return exactly `size` bytes —
      // a bucket-sized shadow would either overrun on write-back or leak
      // tail bytes. Pass arg1=0 to tell the gpu-worker to use the size
      // from descPtr verbatim.
      const h2 = rpcCall(RpcFn.DEVICE_CREATE_BUFFER, descPtr, 0);
      if (h2 > 0 && h2 < FAKE_HANDLE_BASE) {
        setBufferMeta(h2, size, usage);
      }
      return h2;
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
      // Before issuing the RPC we must patch any fake buffer handles in
      // the descriptor: both the deferred-created (mappedAtCreation small
      // buffer, not yet materialized) and the already-materialized-but-
      // remapped handles. The fast path (entryCount <= 10 →
      // FUSED_*_DISPATCH) materializes deferredCreations inline, but the
      // fall-through path previously only patched deferredBufferRemap,
      // leaving unmaterialized fake handles in the descriptor. Those
      // would reach the gpu-worker as bogus bufferHandles and fail the
      // createBindGroup validation with no diagnostic. Phase 6c fix.
      const ENTRY_SIZE = 40;
      for (let i = 0; i < entryCount; i++) {
        const ePtr = entriesPtr + i * ENTRY_SIZE;
        const bufferHandle = view.getUint32(ePtr + 8, true);
        if (bufferHandle === 0) continue;
        // If this is a small mappedAtCreation buffer whose CREATE_BUFFER_FROM_DATA
        // was deferred, materialize it now so we have a real handle to send.
        if (deferredCreations.has(bufferHandle)) {
          materializeDeferredBuffer(bufferHandle);
        }
        // Rewrite the entry with the resolved handle (fake → real).
        const real = deferredBufferRemap.get(bufferHandle);
        if (real !== undefined) {
          view.setUint32(ePtr + 8, real, true);
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

    wgpuDeviceSetUncapturedErrorCallback(_device: number, _cb: number, _ud: number): void {
      rpcCall(RpcFn.DEVICE_SET_ERROR_CALLBACK);
    },

    wgpuDeviceSetDeviceLostCallback(_device: number, _cb: number, _ud: number): void {
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
      _queue: number,
      bufferHandle: number,
      bufferOffset: bigint,
      dataPtr: number,
      size: number,
    ): void {
      const resolved = resolveBufferHandle(bufferHandle);
      // Defer small writes at offset 0 — will be packed into FUSED_DISPATCH_WITH_UNIFORM
      if (fusionEnabled && size <= 256 && bufferOffset === 0n) {
        const h = heap();
        const data = h.slice(dataPtr, dataPtr + size);
        pendingWriteBuffers.set(resolved, { data });
        return;
      }
      // Large or offset writes: send immediately
      const offsetLo = Number(bufferOffset & 0xffffffffn);
      const offsetHi = Number(bufferOffset >> 32n);
      rpcCall(RpcFn.QUEUE_WRITE_BUFFER, resolved, offsetLo, offsetHi, dataPtr, size);
    },

    wgpuQueueOnSubmittedWorkDone(_queue: number, callbackPtr: number, userdataPtr: number): void {
      rpcCall(RpcFn.QUEUE_ON_SUBMITTED_WORK_DONE, callbackPtr, userdataPtr);
    },

    wgpuQueueRelease(): void {
      rpcCall(RpcFn.QUEUE_RELEASE);
    },

    // ===== Command Encoder =====
    // Cache active compute pass to avoid begin/end per dispatch.
    // C++ creates a new pass for each eval, but we reuse the existing one.
    wgpuCommandEncoderBeginComputePass(encoderHandle: number, _descPtr: number): number {
      if (!passCachingEnabled) {
        const handle = rpcCall(RpcFn.CMD_ENCODER_BEGIN_COMPUTE_PASS, encoderHandle);
        activeComputePass = handle;
        activeComputePassEncoder = encoderHandle;
        return handle;
      }
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
      srcHandle: number,
      srcOffset: bigint,
      dstHandle: number,
      dstOffset: bigint,
      size: bigint,
    ): void {
      flushPendingCompute();
      const resolvedSrc = resolveBufferHandle(srcHandle);
      const resolvedDst = resolveBufferHandle(dstHandle);
      const srcOffsetLo = Number(srcOffset & 0xffffffffn);
      const srcOffsetHi = Number(srcOffset >> 32n);
      const dstOffsetLo = Number(dstOffset & 0xffffffffn);
      const dstOffsetHi = Number(dstOffset >> 32n);
      const sizeLo = Number(size & 0xffffffffn);
      const sizeHi = Number(size >> 32n);

      if (!passCachingEnabled) {
        if (activeComputePass >= 0 && activeComputePassEncoder === encoderHandle) {
          rpcCall(RpcFn.COMPUTE_PASS_END, activeComputePass);
          activeComputePass = -1;
          activeComputePassEncoder = -1;
        }
        rpcCallWithHi(
          RpcFn.CMD_ENCODER_COPY_BUFFER,
          [encoderHandle, resolvedSrc, srcOffsetLo, srcOffsetHi, resolvedDst, dstOffsetLo, dstOffsetHi, sizeLo],
          { 0: sizeHi },
        );
        return;
      }

      // For 32-bit offsets/sizes (common in WASM), use fused RPC
      const srcOff32 = Number(srcOffset);
      const dstOff32 = Number(dstOffset);
      const size32 = Number(size);

      if (srcOffset <= 0xffffffff && dstOffset <= 0xffffffff && size <= 0xffffffff) {
        // FUSED: end pass (if active) + copy + begin new pass = 1 RPC instead of 3-4
        let passHandle = 0;
        if (activeComputePass >= 0) {
          if (activeComputePassEncoder === encoderHandle) {
            passHandle = activeComputePass;
          } else {
            rpcCall(RpcFn.COMPUTE_PASS_END, activeComputePass);
            rpcCall(RpcFn.COMPUTE_PASS_RELEASE, activeComputePass);
          }
        }
        const newPassHandle = rpcCall(
          RpcFn.FUSED_COPY_BUFFER,
          encoderHandle,
          passHandle,
          resolvedSrc,
          srcOff32,
          resolvedDst,
          dstOff32,
          size32,
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
      if (!fusionEnabled) {
        flushPendingCompute();
        rpcCall(RpcFn.COMPUTE_PASS_SET_PIPELINE, passHandle, pipelineHandle);
        return;
      }
      pendingPass = passHandle;
      pendingPipeline = pipelineHandle;
      pendingBindGroup = -1; // Reset bind group — new pipeline needs new bind group
      pendingBindGroup1 = -1;
    },

    wgpuComputePassEncoderSetBindGroup(
      passHandle: number,
      groupIndex: number,
      bgHandle: number,
      dynamicOffsetCount: number,
      dynamicOffsetsPtr: number,
    ): void {
      if (!fusionEnabled) {
        flushPendingCompute();
        const realBg = materializeBg(bgHandle);
        rpcCall(
          RpcFn.COMPUTE_PASS_SET_BIND_GROUP,
          passHandle,
          groupIndex,
          realBg,
          dynamicOffsetCount,
          dynamicOffsetsPtr,
        );
        return;
      }
      if (groupIndex === 0 && dynamicOffsetCount === 0 && pendingPipeline >= 0) {
        // Buffer for fusion with upcoming dispatch
        pendingBindGroup = bgHandle;
      } else if (groupIndex === 1 && dynamicOffsetCount === 0 && pendingPipeline >= 0) {
        // Buffer bind group 1 for FUSED_DISPATCH_2BG
        pendingBindGroup1 = bgHandle;
      } else {
        // Non-fusable: flush pending and do individual RPC
        flushPendingCompute();
        rpcCall(
          RpcFn.COMPUTE_PASS_SET_BIND_GROUP,
          passHandle,
          groupIndex,
          bgHandle,
          dynamicOffsetCount,
          dynamicOffsetsPtr,
        );
      }
    },

    wgpuComputePassEncoderDispatchWorkgroups(passHandle: number, x: number, y: number, z: number): void {
      if (!fusionEnabled) {
        flushPendingCompute();
        rpcCall(RpcFn.COMPUTE_PASS_DISPATCH, passHandle, x, y, z);
        return;
      }
      // FUSED_DISPATCH_WITH_UNIFORM pre-writes uniform bytes into the SAB
      // UNIFORM_DATA region, so rpcCall's flush-at-entry is disabled for it.
      // Flush here instead, BEFORE any SAB writes, so the gpu-worker sees
      // pending releases before the next dispatch — but only when the batch
      // is large enough to be worth a round-trip. Small-batch flushes on
      // every dispatch defeat the amortization and drop average batch size
      // to ~3 handles. Threshold 32 keeps the release visible-to-pool
      // latency bounded while still amortizing RPCs.
      if (pendingReleases.length >= 32) {
        flushPendingReleases();
      }
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
          let deferredFakeHandle = -1; // track deferred creation to finalize after RPC
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

          // Task B: drain any F&F BUFFER_RELEASE_BATCH before clobbering cmd
          // SAB (UNIFORM_DATA overlaps with the release-batch handle list).
          // The flush at the top of this function may have fired F&F; if no
          // subsequent materialize/flushPendingWrites happened, ffPending is
          // still set here. Cheap no-op when false.
          drainFireAndForget();

          // Write entry data into the callback ring space BEFORE rpcCall
          cmdU32[I_CB_COUNT] = entryCount;
          for (let i = 0; i < entryCount; i++) {
            const e = bgDesc.entries[i];
            const base = (CMD_OFFSET.CALLBACK_BASE + i * 12) >>> 2;
            // Use 0 for deferred creation entry (gpu-worker will create buffer)
            cmdU32[base] = i === uniformEntryIdx && deferredInfo ? 0 : resolveBufferHandle(e.bufferHandle);
            cmdU32[base + 1] = e.sizeLo;
            cmdU32[base + 2] = e.sizeHi;
          }

          if (uniformData && uniformData.byteLength <= 256) {
            // FUSED_DISPATCH_WITH_UNIFORM: pack data + bind group + dispatch in one RPC
            // ARG7 = buffer usage flags (for deferred creation, or 0 if buffer exists)
            const usage = deferredInfo ? deferredInfo.usage : 0;
            cmdU32[CMD_OFFSET.UNIFORM_DATA_SIZE >>> 2] = uniformData.byteLength;
            new Uint8Array(cmdBuffer, CMD_OFFSET.UNIFORM_DATA, uniformData.byteLength).set(uniformData);
            // Write ARG fields (same positions as a regular rpcCall) so
            // stageDispatchBatchRecord can copy straight out of cmdU32.
            cmdU32[I_ARG0] = pendingPass >>> 0;
            cmdU32[I_ARG0 + 1] = pendingPipeline >>> 0;
            cmdU32[I_ARG0 + 2] = bgDesc.layoutHandle >>> 0;
            cmdU32[I_ARG0 + 3] = x >>> 0;
            cmdU32[I_ARG0 + 4] = y >>> 0;
            cmdU32[I_ARG0 + 5] = z >>> 0;
            cmdU32[I_ARG0 + 6] = uniformEntryIdx >>> 0;
            cmdU32[I_ARG0 + 7] = usage >>> 0;
            // Batching path: only safe when the call has NO return value
            // to consume (no deferred buffer creation that expects a new
            // handle back from the gpu-worker).
            bumpBatchAttempt();
            if (!deferredInfo && stageDispatchBatchRecord(RpcFn.FUSED_DISPATCH_WITH_UNIFORM)) {
              bumpBatchStaged();
            } else {
              if (deferredInfo) bumpBatchDeferredBlock();
              else bumpBatchStageRefused();
              // Staging refused (e.g. entryCount > MAX_DISPATCH_BATCH_ENTRIES
              // or uniformSize > MAX_DISPATCH_BATCH_UNIFORM, or a deferred
              // buffer create expects a return value). Flush any pending
              // batch first so the fallback dispatch does NOT reorder ahead
              // of earlier staged dispatches — the flush guard at the top of
              // rpcCall exempts FUSED_* opcodes, so we must flush by hand.
              if (batchActive && batchCount > 0) flushDispatchBatchInner();
              const result = rpcCall(
                RpcFn.FUSED_DISPATCH_WITH_UNIFORM,
                pendingPass,
                pendingPipeline,
                bgDesc.layoutHandle,
                x,
                y,
                z,
                uniformEntryIdx,
              );

              // Finalize deferred buffer creation: gpu-worker returns real handle.
              // JS-F016: cleanup runs unconditionally even on RPC failure
              // (result === 0) so wasmFree and the shadow-size scrub fire in
              // every path. Previously the graceful-failure branch silently
              // leaked the WASM heap alloc plus a stale meta-table entry.
              if (deferredInfo && deferredFakeHandle >= 0) {
                if (result > 0) {
                  deferredBufferRemap.set(deferredFakeHandle, result);
                  setBufferMeta(result, deferredInfo.size, deferredInfo.usage);
                }
                // Scrub the fake handle's shadow size entry (set when the
                // fake was created) so the metadata table doesn't leak
                // stale keys. This ALWAYS runs — including the result === 0
                // graceful-failure branch that previously skipped it.
                clearBufferMeta(deferredFakeHandle);
                wasmFree(deferredInfo.wasmPtr);
              }
            }
          } else {
            // No inline data or too large — flush and use FUSED_FULL_DISPATCH
            if (deferredInfo && deferredFakeHandle >= 0) {
              // Can't inline — fall back to immediate creation
              const sizeLo = deferredInfo.size & 0xffffffff;
              const sizeHi = Math.floor(deferredInfo.size / 0x100000000);
              const realHandle = rpcCall(
                RpcFn.CREATE_BUFFER_FROM_DATA,
                deferredInfo.usage,
                sizeLo,
                sizeHi,
                deferredInfo.wasmPtr,
              );
              // JS-F016: guard the remap against RPC-failure returns (0).
              // Previously we stored `fake → 0`, which made resolveBufferHandle
              // translate every later reference to the null handle — corrupting
              // the dispatch silently.
              if (realHandle > 0) {
                deferredBufferRemap.set(deferredFakeHandle, realHandle);
                setBufferMeta(realHandle, deferredInfo.size, deferredInfo.usage);
                // Rewrite the entry with the real handle. Only valid when we
                // actually got a real handle back.
                const base = (CMD_OFFSET.CALLBACK_BASE + uniformEntryIdx * 12) >>> 2;
                cmdU32[base] = realHandle;
              }
              // Scrub the fake handle's shadow size entry and release the WASM
              // shadow unconditionally — the payload is consumed either way.
              clearBufferMeta(deferredFakeHandle);
              wasmFree(deferredInfo.wasmPtr);
            }
            flushPendingWrites();
            // Write ARG fields for the staging path (same as rpcCall would).
            cmdU32[I_ARG0] = pendingPass >>> 0;
            cmdU32[I_ARG0 + 1] = pendingPipeline >>> 0;
            cmdU32[I_ARG0 + 2] = bgDesc.layoutHandle >>> 0;
            cmdU32[I_ARG0 + 3] = x >>> 0;
            cmdU32[I_ARG0 + 4] = y >>> 0;
            cmdU32[I_ARG0 + 5] = z >>> 0;
            // FUSED_FULL_DISPATCH has no uniform data; ensure the size slot
            // is zero so any concurrent batch staging sees uniformSize=0.
            cmdU32[CMD_OFFSET.UNIFORM_DATA_SIZE >>> 2] = 0;
            bumpBatchAttempt();
            if (stageDispatchBatchRecord(RpcFn.FUSED_FULL_DISPATCH)) {
              bumpBatchStaged();
            } else {
              bumpBatchStageRefused();
              // Staging refused (e.g. entryCount > MAX_DISPATCH_BATCH_ENTRIES
              // or buffer overflow). Flush any pending batch
              // first so the fallback dispatch does NOT reorder ahead of
              // earlier staged dispatches — the flush guard at the top of
              // rpcCall exempts FUSED_* opcodes, so we must flush by hand.
              if (batchActive && batchCount > 0) flushDispatchBatchInner();
              rpcCall(RpcFn.FUSED_FULL_DISPATCH, pendingPass, pendingPipeline, bgDesc.layoutHandle, x, y, z);
            }
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
      if (activeComputePass === passHandle) {
        activeComputePass = -1;
        activeComputePassEncoder = -1;
      }
      rpcCall(RpcFn.COMPUTE_PASS_END, passHandle);
    },

    wgpuComputePassEncoderRelease(handle: number): void {
      if (activeComputePass === handle) {
        activeComputePass = -1;
        activeComputePassEncoder = -1;
      }
      rpcCall(RpcFn.COMPUTE_PASS_RELEASE, handle);
    },

    // ===== Buffer =====
    wgpuBufferGetSize(bufferHandle: number): bigint {
      // Fast path: return cached size (no RPC) — check both fake and real handle
      const cached = getBufferSize(bufferHandle);
      if (cached !== undefined) return BigInt(cached);
      // Slow path: RPC to gpu-worker (resolve fake→real first).
      // Do NOT cache-fill the SAB here: we only know `size` and publishing a
      // non-zero size lane without the matching usage would break the
      // setBufferMeta invariant ("usage-first, then size sentinel") and let
      // getBufferUsage return a stale/zero usage. Real handles normally get
      // their metadata seeded at create time via setBufferMeta, so this slow
      // path is rare in practice.
      const resolved = resolveBufferHandle(bufferHandle);
      rpcCall(RpcFn.BUFFER_GET_SIZE, resolved);
      const lo = cmdDataView.getUint32(CMD_OFFSET.RESULT, true);
      const hi = cmdDataView.getUint32(CMD_OFFSET.RESULT_HI, true);
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
          const sizeLo = mapped.size & 0xffffffff;
          const sizeHi = Math.floor(mapped.size / 0x100000000);
          const realHandle = rpcCall(RpcFn.CREATE_BUFFER_FROM_DATA, mapped.usage, sizeLo, sizeHi, mapped.wasmPtr);
          // JS-F016: guard against RPC failure (realHandle === 0). Storing a
          // 0-valued remap turns every future reference to the fake handle
          // into a null dereference on the gpu-worker side.
          if (realHandle > 0) {
            deferredBufferRemap.set(bufferHandle, realHandle);
            setBufferMeta(realHandle, mapped.size, mapped.usage);
          }
          // Scrub the fake handle's shadow size entry + release the WASM
          // shadow in both paths — the payload is consumed by the RPC either
          // way and re-attempting with the same wasmPtr would double-free.
          clearBufferMeta(bufferHandle);
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
      bufferHandle: number,
      mode: number,
      offset: number,
      size: number,
      callbackPtr: number,
      userdataPtr: number,
    ): void {
      rpcCall(RpcFn.BUFFER_MAP_ASYNC, resolveBufferHandle(bufferHandle), mode, offset, size, callbackPtr, userdataPtr);
    },

    wgpuBufferDestroy(bufferHandle: number): void {
      // Drop any deferred writeBuffer keyed by this raw handle BEFORE any
      // early-return branch — mirrors wgpuBufferRelease (see its comment for
      // the full rationale on fake-handle flushPendingWrites leaks).
      pendingWriteBuffers.delete(bufferHandle);

      // No-op: gpu-worker already skips buffer destroy to avoid "destroyed in submit" errors.
      // Clean up deferred state if present.
      const mapped = mappedAtCreationBuffers.get(bufferHandle);
      if (mapped) {
        mappedAtCreationBuffers.delete(bufferHandle);
        wasmFree(mapped.wasmPtr);
        clearBufferMeta(bufferHandle);
      }
      const def = deferredCreations.get(bufferHandle);
      if (def) {
        deferredCreations.delete(bufferHandle);
        wasmFree(def.wasmPtr);
        // Scrub fake-handle shadow metadata: deferredCreations inherits the
        // fake handle from the mappedAtCreation phase, and the size entry
        // was set when the fake was first minted. Without this, destroying
        // a small deferred buffer before its first dispatch leaks the entry.
        clearBufferMeta(bufferHandle);
      }
      const resolved = resolveBufferHandle(bufferHandle);
      if (resolved !== bufferHandle) {
        deferredBufferRemap.delete(bufferHandle);
        // A deferred writeBuffer could also be keyed on the resolved handle;
        // drop it so a pooled-reuse of `resolved` can't be clobbered.
        pendingWriteBuffers.delete(resolved);
      }
      // Destroy is a no-op on the gpu-worker side, so any pooled buddies at
      // this (usage, size) key remain valid GPUBuffers. Leave the pool alone.
      //
      // Do NOT clear the (size, usage) metadata here: MLX's WebGPU allocator
      // calls wgpuBufferDestroy BEFORE wgpuBufferRelease on the same handle
      // during its normal teardown. If we wipe the SAB slot at destroy time,
      // the release-time lookup finds size=0 and falls through the
      // unknownHandle branch, which skips the client pool entirely. Leaving
      // the metadata alive lets the subsequent release pool the real handle
      // and suppress the BUFFER_RELEASE RPC.
    },

    wgpuBufferRelease(handle: number): void {
      diagReleaseAll++;
      bumpPoolStat(POOL_STAT_RELEASE_ALL);
      // Drop any deferred writeBuffer keyed by this raw handle. MUST run before
      // any early-return branch: for fake mappedAtCreation / deferred-creation
      // handles, resolveBufferHandle() returns the raw fake id (no remap exists),
      // so pendingWriteBuffers is keyed by the fake handle. If we deleted only
      // after the resolve step below, fake-handle early returns would leave the
      // entry behind, and the next flushPendingWrites would issue a bogus
      // QUEUE_WRITE_BUFFER RPC against an id the gpu-worker can't resolve.
      pendingWriteBuffers.delete(handle);

      // Clean up shadow state if this fake mapped-at-creation buffer was never unmapped.
      // Must run before any pool/RPC logic — the fake handle has no gpu-worker object,
      // and the WASM shadow allocation would otherwise leak.
      const mapped = mappedAtCreationBuffers.get(handle);
      if (mapped) {
        mappedAtCreationBuffers.delete(handle);
        wasmFree(mapped.wasmPtr);
        clearBufferMeta(handle);
        return;
      }
      // Clean up deferred creation if buffer is released before dispatch
      const def = deferredCreations.get(handle);
      if (def) {
        deferredCreations.delete(handle);
        wasmFree(def.wasmPtr);
        // Scrub fake-handle shadow metadata: the size entry was set when
        // the fake was first minted, and the small-buffer unmap path
        // transfers ownership into deferredCreations without cleaning it
        // up. Without this, releasing a small deferred buffer before its
        // first dispatch leaks the entry.
        clearBufferMeta(handle);
        return; // no GPU buffer was created, nothing to release
      }
      const resolved = resolveBufferHandle(handle);
      if (resolved !== handle) {
        deferredBufferRemap.delete(handle);
        // A writeBuffer might also have been deferred on the resolved key
        // (unlikely, but safe to handle). If the handle gets pooled and reused,
        // a stale flush would clobber the new buffer's contents.
        pendingWriteBuffers.delete(resolved);
      }

      // Try to pool the real handle instead of releasing to the gpu-worker.
      const size = getBufferSize(resolved);
      const usage = getBufferUsage(resolved);
      if (size === undefined || usage === undefined) {
        diagReleaseUnknownHandle++;
        bumpPoolStat(POOL_STAT_RELEASE_UNKNOWN);
      } else if (!isPoolable(resolved, usage)) {
        diagReleaseUnpoolable++;
        bumpPoolStat(POOL_STAT_RELEASE_UNPOOLABLE);
      }
      if (size !== undefined && usage !== undefined && isPoolable(resolved, usage)) {
        // Bucket the stub pool. `size` here is the LOGICAL size from
        // setBufferMeta; converting through roundUpBucket yields the bucket
        // key used on both create and release paths. Because every create
        // MISS above passes bucketSize via arg1 (so the gpu-worker
        // allocates exactly bucketSize), every buffer in this bucket has
        // physical size >= the bucket — safe to pop and return on a later
        // create for any logical size that maps to the same bucket.
        const bucketSize = roundUpBucket(size);
        const key = poolKey(usage, bucketSize);
        let stack = bufferPool.get(key);
        if (stack === undefined) {
          stack = [];
          bufferPool.set(key, stack);
        }
        if (stack.length < POOL_CAP_PER_KEY) {
          stack.push(resolved);
          return; // suppressed RPC — handle stays alive in the gpu-worker
        }
        // Pool full: evict oldest (FIFO) and release it, then pool the new one.
        const evicted = stack.shift()!;
        bumpPoolStat(POOL_STAT_EVICTIONS);
        queueRelease(evicted);
        clearBufferMeta(evicted);
        stack.push(resolved);
        return;
      }

      queueRelease(resolved);
      clearBufferMeta(resolved);
    },

    // ===== Pipeline =====
    wgpuComputePipelineGetBindGroupLayout(pipelineHandle: number, index: number): number {
      const key = pipelineHandle * 4 + index;
      const cached = layoutCache.get(key);
      if (cached !== undefined) return cached;
      const handle = rpcCall(RpcFn.PIPELINE_GET_BIND_GROUP_LAYOUT, pipelineHandle, index);
      layoutCache.set(key, handle);
      return handle;
    },

    wgpuComputePipelineRelease(handle: number): void {
      // Drop any cached layouts for this pipeline — assumes ≤4 layouts,
      // matching the key scheme. Loop is cheap; pipeline releases are rare.
      for (let i = 0; i < 4; i++) {
        layoutCache.delete(handle * 4 + i);
      }
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

  function getBridgeStats(): BridgeStats {
    const byFn: Record<number, number> = {};
    for (let i = 0; i < bridgeFnCounts.length; i++) {
      if (bridgeFnCounts[i] > 0) {
        byFn[i] = bridgeFnCounts[i];
      }
    }
    // Read cumulative counters from the shared pool-stats SAB (every stub
    // Atomics.adds into this region), not from the per-instance locals —
    // the per-instance counters only cover calls made through *this* stub
    // (e.g. the mlx-worker main thread), which misses the pthread workers
    // that actually do the decode work.
    let ph = bufferPoolHits;
    let pm = bufferPoolMisses;
    let dCreateAll = diagCreateAll;
    let dCreateMCD = diagCreateMappedCopyDst;
    let dCreateMNCD = diagCreateMappedNoCopyDst;
    let dRelAll = diagReleaseAll;
    let dRelUnk = diagReleaseUnknownHandle;
    let dRelUnp = diagReleaseUnpoolable;
    let dBatchAttempt = diagBatchAttempt;
    let dBatchStaged = diagBatchStaged;
    let dBatchDeferredBlock = diagBatchDeferredBlock;
    let dBatchStageRefused = diagBatchStageRefused;
    if (poolStatsArr) {
      ph = Atomics.load(poolStatsArr, POOL_STAT_HITS);
      pm = Atomics.load(poolStatsArr, POOL_STAT_MISSES);
      dCreateAll = Atomics.load(poolStatsArr, POOL_STAT_CREATE_ALL);
      dCreateMCD = Atomics.load(poolStatsArr, POOL_STAT_CREATE_MAPPED_COPY_DST);
      dCreateMNCD = Atomics.load(poolStatsArr, POOL_STAT_CREATE_MAPPED_NO_COPY_DST);
      dRelAll = Atomics.load(poolStatsArr, POOL_STAT_RELEASE_ALL);
      dRelUnk = Atomics.load(poolStatsArr, POOL_STAT_RELEASE_UNKNOWN);
      dRelUnp = Atomics.load(poolStatsArr, POOL_STAT_RELEASE_UNPOOLABLE);
      dBatchAttempt = Atomics.load(poolStatsArr, POOL_STAT_BATCH_ATTEMPT);
      dBatchStaged = Atomics.load(poolStatsArr, POOL_STAT_BATCH_STAGED);
      dBatchDeferredBlock = Atomics.load(poolStatsArr, POOL_STAT_BATCH_DEFERRED_BLOCK);
      dBatchStageRefused = Atomics.load(poolStatsArr, POOL_STAT_BATCH_STAGE_REFUSED);
    }
    return {
      rpcCount: bridgeRpcCount,
      byFn,
      poolHits: ph,
      poolMisses: pm,
      diagCreateAll: dCreateAll,
      diagCreateMappedCopyDst: dCreateMCD,
      diagCreateMappedNoCopyDst: dCreateMNCD,
      diagReleaseAll: dRelAll,
      diagReleaseUnknownHandle: dRelUnk,
      diagReleaseUnpoolable: dRelUnp,
      diagBatchAttempt: dBatchAttempt,
      diagBatchStaged: dBatchStaged,
      diagBatchDeferredBlock: dBatchDeferredBlock,
      diagBatchStageRefused: dBatchStageRefused,
      poisonedRpcCount,
    };
  }

  function resetBridgeStats(): void {
    bridgeRpcCount = 0;
    bridgeFnCounts.fill(0);
    bufferPoolHits = 0;
    bufferPoolMisses = 0;
    diagCreateAll = 0;
    diagCreateMappedCopyDst = 0;
    diagCreateMappedNoCopyDst = 0;
    diagReleaseAll = 0;
    diagReleaseUnknownHandle = 0;
    diagReleaseUnpoolable = 0;
    diagBatchAttempt = 0;
    diagBatchStaged = 0;
    diagBatchDeferredBlock = 0;
    diagBatchStageRefused = 0;
    poisonedRpcCount = 0;
    if (poolStatsArr) {
      for (let i = 0; i < POOL_STATS_SLOTS; i++) Atomics.store(poolStatsArr, i, 0);
    }
  }

  function fetchGpuWorkerStats(resetAfter: boolean): GpuWorkerStats {
    const totalRpcs = rpcCall(RpcFn.GET_STATS, resetAfter ? 1 : 0) >>> 0;
    // JS-F010: flat read from the dedicated stats SAB — no region math, no
    // overlap with UNIFORM_DATA / CALLBACK_BASE. When statsU32 is null
    // (legacy test harness that didn't plumb the SAB), the histogram is
    // empty and callers still get a valid totalRpcs + zeroed counters.
    const byFn: Record<number, number> = {};
    if (statsU32) {
      for (let i = 0; i < STATS_OPCODE_SLOTS; i++) {
        const v = statsU32[i];
        if (v > 0) byFn[i] = v;
      }
    }
    // Synthetic counters live above the real RpcFn opcode range. Pull them
    // out so the demo does not render phantom opcodes while preserving real
    // opcode 102/103 counts.
    const gpuPoolHits = byFn[STATS_EXTRA_POOL_HITS] ?? 0;
    const gpuPoolMisses = byFn[STATS_EXTRA_POOL_MISSES] ?? 0;
    const bindGroupCacheHits = byFn[STATS_EXTRA_BG_CACHE_HITS] ?? 0;
    const bindGroupCacheMisses = byFn[STATS_EXTRA_BG_CACHE_MISSES] ?? 0;
    const uniformHotHits = byFn[STATS_EXTRA_UNIFORM_HOT_HITS] ?? 0;
    const uniformHotMisses = byFn[STATS_EXTRA_UNIFORM_HOT_MISSES] ?? 0;
    const spinHits = byFn[STATS_EXTRA_SPIN_HITS] ?? 0;
    const spinMisses = byFn[STATS_EXTRA_SPIN_MISSES] ?? 0;
    const spinBudget = byFn[STATS_EXTRA_SPIN_BUDGET] ?? 0;
    delete byFn[STATS_EXTRA_POOL_HITS];
    delete byFn[STATS_EXTRA_POOL_MISSES];
    delete byFn[STATS_EXTRA_BG_CACHE_HITS];
    delete byFn[STATS_EXTRA_BG_CACHE_MISSES];
    delete byFn[STATS_EXTRA_UNIFORM_HOT_HITS];
    delete byFn[STATS_EXTRA_UNIFORM_HOT_MISSES];
    delete byFn[STATS_EXTRA_SPIN_HITS];
    delete byFn[STATS_EXTRA_SPIN_MISSES];
    delete byFn[STATS_EXTRA_SPIN_BUDGET];
    return {
      totalRpcs,
      byFn,
      gpuPoolHits,
      gpuPoolMisses,
      bindGroupCacheHits,
      bindGroupCacheMisses,
      uniformHotHits,
      uniformHotMisses,
      spinHits,
      spinMisses,
      spinBudget,
    };
  }

  return {
    imports,
    setInstance,
    getBridgeStats,
    resetBridgeStats,
    fetchGpuWorkerStats,
  };
}
