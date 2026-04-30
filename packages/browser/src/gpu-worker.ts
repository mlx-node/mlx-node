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

import { roundUpBucket } from "./buffer-bucket.js";
import {
  RpcFn,
  CMD_OFFSET,
  STATUS,
  STATUS_INDEX,
  MAX_CALLBACKS_PER_CALL,
  CALLBACK_ENTRY_SIZE,
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
  DISPATCH_BATCH_HEADER_BYTES,
  DISPATCH_BATCH_BUFFER_SIZE,
  MAX_DISPATCH_BATCH,
  MAX_DISPATCH_BATCH_ENTRIES,
  MAX_DISPATCH_BATCH_UNIFORM,
} from "./rpc-protocol.js";

// ---------- Phase 0 RPC Histogram (?profile=1 observability) ----------
//
// Per-opcode count of RpcFn dispatches handled by this worker. Bumped at
// the very top of processCommand's switch (before any case body runs) and
// read + optionally reset by the GET_STATS handler. The array length is
// STATS_OPCODE_SLOTS so the readback handler can assume a fixed layout
// regardless of how many RpcFn values exist.
const gpuRpcCounts = new Uint32Array(STATS_OPCODE_SLOTS);
let gpuTotalRpcs = 0;

// ---------- Adaptive spin budget (P6 / JS-F001 + JS-F002) ----------
//
// commandLoop spins on Atomics.load before falling back to Atomics.waitAsync.
// Spin budget used to be a fixed 2000 — empirically the sweet spot on M3 Chrome
// but it's a perf cliff: a missed spin falls through to a macrotask via
// `setTimeout(0)` which costs milliseconds and is the suspected root cause of
// the bimodal 15-21 tok/s variance.
//
// Fix:
//   1. Never `setTimeout(0)` — loop back on Atomics.waitAsync instead.
//   2. Re-check status atomically before awaiting so a status that flipped
//      between the spin and the waitAsync call short-circuits without a tick.
//   3. Adapt the budget: after SPIN_ADAPT_K consecutive hits, double (up to
//      SPIN_MAX); after SPIN_ADAPT_K consecutive misses, halve (down to
//      SPIN_MIN). Starts at the historical 2000 value.
const SPIN_MIN = 500;
const SPIN_MAX = 8000;
const SPIN_ADAPT_K = 8;
let spinDispatchBudget = 2000;
let spinConsecutiveHits = 0;
let spinConsecutiveMisses = 0;
let spinHitTotal = 0;
let spinMissTotal = 0;

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
let currentFnId = 0;
let currentDispatchContext = "";

// Shared with every wasm-side bridge stub. Handles imported directly in this
// worker, such as local-model weight buffers, never pass through
// wgpuDeviceCreateBuffer in the stub, so seed their metadata here.
const BUFFER_METADATA_MAX_HANDLES = 1 << 20;
let bufferMetaU32: Uint32Array | null = null;

function setSharedBufferMeta(
  handle: number,
  size: number,
  usage: number,
): void {
  if (!bufferMetaU32 || handle <= 0 || handle >= BUFFER_METADATA_MAX_HANDLES)
    return;
  const base = handle * 2;
  Atomics.store(bufferMetaU32, base + 1, usage >>> 0);
  Atomics.store(bufferMetaU32, base, size >>> 0);
}

function clearSharedBufferMeta(handle: number): void {
  if (!bufferMetaU32 || handle <= 0 || handle >= BUFFER_METADATA_MAX_HANDLES)
    return;
  const base = handle * 2;
  Atomics.store(bufferMetaU32, base, 0);
  Atomics.store(bufferMetaU32, base + 1, 0);
}

// JS-F007: handle recycling.
//
// Previously `addHandle` only did `nextHandle++`. Released slots were never
// reused — long sessions marched nextHandle toward the bridge's
// BUFFER_METADATA_MAX_HANDLES=2^20 ceiling. Qwen3.5-0.8B at ~120 fresh
// buffer creates/tok × 224 tokens ≈ 27k handles per generation, so the
// bridge metadata SAB blows its capacity around generation 35 and every
// stub silently falls back to stub-local Maps, losing cross-worker release
// visibility.
//
// The cache invariants that make recycling safe:
//   (a) bindGroupCache is keyed by pipelineHandle. Pipelines are never
//       pooled; on PIPELINE_RELEASE we delete the cache entry before
//       releasing the handle. A recycled pipeline handle starts with no
//       cache entry.
//   (b) passEncoderMap is keyed by passHandle. Passes are released in the
//       dispatch submit cycle (endAndRestartPass deletes the old entry
//       before re-adding the replacement).
//   (c) activeMappings is keyed by bufferHandle. Mapped buffers carry
//       MAP_READ / MAP_WRITE usage which `isPoolable` rejects, so these
//       handles skip the pool stack entirely and hit `releaseHandle`
//       directly. The bridge stub clears bufferMetadata on release.
//   (d) The buffer pool keeps `handles[h]` alive while parked. Only when
//       the pool bucket is full does `releaseHandle` actually run — at
//       which point the handle is gone from every bookkeeping map.
//
// So pure monotonic recycling is correct today. To harden against future
// cache additions that might capture a handle integer and read it across
// a release/recycle boundary, we delay recycling by a small generation
// window: a freed handle only becomes eligible after RECYCLE_DELAY other
// handles have been freed since it. This preserves realistic LRU-cache
// patterns (which would miss on the delayed handle rather than hit a
// stale entry) and gives the bridge stub's metadata SAB time to see the
// release land before the same integer is re-issued.
//
// Versioning (upper-bits generation counter) would be cleaner but is an
// invasive change — ~100 call sites read handles as plain integers and
// index `handles[h]` directly. Tail-delay is a ~20-LOC local change that
// fixes the leak without risking correctness regressions.
const freeHandleList: number[] = [];
const RECYCLE_DELAY = 256;

// Buffer pool: (usage, size) → stack of free GPUBuffer handles. MLX's allocator
// churns ~870 transient buffers/tok; reusing them eliminates Chrome's
// createBuffer validation cost on the hot path.
const bufferPool = new Map<string, number[]>();
const BUFFER_POOL_MAX_BYTES = 512 * 1024 * 1024;
const BUFFER_POOL_MAX_BUFFER_BYTES = 64 * 1024 * 1024;
let bufferPoolHitCount = 0;
let bufferPoolMissCount = 0;
let bufferPoolBytes = 0;
const activeCommandEncoders = new Set<number>();
const pendingCommandBuffers = new Set<number>();
const quarantinedBufferPoolHandles: number[] = [];
const quarantinedBufferPoolSet = new Set<number>();
const quarantinedBufferDestroyHandles: number[] = [];
const quarantinedBufferDestroySet = new Set<number>();
const pendingBufferDestroys: GPUBuffer[] = [];
let pendingBufferDestroyBatchScheduled = false;

function bufferPoolCapForSize(size: number): number {
  if (size <= 4 * 1024) return 512;
  if (size <= 64 * 1024) return 256;
  if (size <= 1024 * 1024) return 128;
  return 64;
}

function isPoolableSize(size: number): boolean {
  return size > 0 && size <= BUFFER_POOL_MAX_BUFFER_BYTES;
}

function removeBufferPoolBytes(size: number): void {
  bufferPoolBytes = Math.max(0, bufferPoolBytes - size);
}

// Task C: single-entry LRU bind group cache per pipeline.
//
// FUSED_FULL_DISPATCH / FUSED_DISPATCH_WITH_UNIFORM / DISPATCH_BATCH create a
// fresh GPUBindGroup per dispatch (~80+/tok on Qwen3.5-0.8B decode). A decode
// step reuses the same pipelines against (mostly) the same weight handles; the
// uniform buffer handle recycles through the buffer pool. When all bound
// handles, sizes, and the layout are unchanged, the prior GPUBindGroup is
// still valid and can be reused.
//
// Safety rests on one invariant: gpu-worker handles are monotonic
// (addHandle = nextHandle++) and never recycled. A pool release keeps
// handles[h] alive pointing at the same GPUBuffer, so identical (handle,
// sizeLo, sizeHi) triples across two dispatches are guaranteed to reference
// the same GPUBuffer with the same binding geometry. Contents may differ
// (writeBuffer re-stamps the uniform bytes per dispatch) but a bind group
// is a binding *descriptor*, not a content snapshot.
//
// A small LRU per pipeline keeps bind groups stable across dispatch batches.
// Decode records many dispatches before releases can recycle uniform-buffer
// handles, so a one-slot cache thrashes even when the same signatures return on
// the next token. Keep the cap modest to avoid retaining unbounded bind groups.
const BIND_GROUP_CACHE_PER_PIPELINE = 32;
interface BindGroupCacheEntry {
  layoutHandle: number;
  entryCount: number;
  // JS-F004: snapshot of entrySrc[0..entryCount*3] captured at insert time.
  // Hit-check compares this element-by-element against the fresh entrySrc;
  // no string concat on the hot dispatch path. Storage is a flat Uint32Array
  // because entryCount ≤ MAX_DISPATCH_BATCH_ENTRIES (10), so max size is
  // 10 × 3 u32s = 120 bytes per cache entry.
  entries: Uint32Array;
  bindGroup: GPUBindGroup;
  // Task E: dedicated 256B uniform buffer owned by this cache entry. On the
  // first FUSED_DISPATCH_WITH_UNIFORM dispatch that would otherwise acquire a
  // uniform buffer from the pool, we allocate one here and reuse it on every
  // subsequent dispatch through queue.writeBuffer — skipping the pool churn
  // and the BUFFER_RELEASE_BATCH round-trip from the bridge stub. Destruction
  // happens in PIPELINE_RELEASE.
  uniformBuffer: GPUBuffer | null;
  uniformHandle: number;
}
const bindGroupCache = new Map<number, BindGroupCacheEntry[]>();
let bindGroupCacheHits = 0;
let bindGroupCacheMisses = 0;
let fatalGpuError: string | null = null;
// Task E: hot-path uniform counters — dispatches that reused the pipeline's
// dedicated uniform buffer vs. had to allocate one for the first time.
let uniformHotHits = 0;
let uniformHotMisses = 0;
// Task E: handles owned by a BindGroupCacheEntry's uniformBuffer. The bridge
// stub enqueues a BUFFER_RELEASE_BATCH for the handle we return from the
// deferred-uniform path; we short-circuit releaseBufferHandle for these so
// the dedicated buffer survives until PIPELINE_RELEASE destroys it.
const dedicatedUniformHandles = new Set<number>();
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
  // JS-F007's original motivation was avoiding the bridge-stub
  // BUFFER_METADATA_MAX_HANDLES = 2^20 ceiling during multi-generation
  // sessions. Recycling aggressively from handle 0 (as the first version
  // did) turned out to break across-chat correctness whenever any downstream
  // cache stashed a handle integer (bind-group cache entries, WASM-side
  // buffer refs captured during a prior chat). We now keep the original
  // monotonic scheme until `nextHandle` actually approaches the ceiling;
  // a handful of chats never gets close to 2^19, so recycling is a no-op
  // in practice — but the leak guard is still in place for truly long
  // sessions.
  const RECYCLE_WATERMARK = 1 << 19;
  if (
    nextHandle >= RECYCLE_WATERMARK &&
    freeHandleList.length >= RECYCLE_DELAY
  ) {
    const id = freeHandleList.shift()!;
    handles[id] = obj;
    return id;
  }
  const id = nextHandle++;
  handles[id] = obj;
  return id;
}

function getHandle<T>(id: number, label = "handle"): T {
  const obj = handles[id];
  if (!obj) {
    const context = currentDispatchContext ? ` ${currentDispatchContext}` : "";
    throw new Error(
      `[GPU Worker] Invalid ${label}: ${id} (fn=${currentFnId}${context})`,
    );
  }
  return obj as T;
}

function postRpcError(fnId: number, err: unknown, context = ""): void {
  const message = err instanceof Error ? err.message : String(err);
  const stack = err instanceof Error ? err.stack : undefined;
  try {
    self.postMessage({ type: "rpc-error", fnId, message, stack, context });
  } catch {
    // ignore postMessage failures during worker teardown
  }
}

function releaseHandle(id: number): void {
  if (id <= 0 || handles[id] === undefined) {
    return;
  }
  invalidateBindGroupCacheForHandle(id);
  // Clear every bookkeeping slot BEFORE adding to the free list so a racing
  // read via getHandle observes "undefined" (which throws "Invalid handle")
  // rather than a stale object. The free list is FIFO with a delay window
  // of RECYCLE_DELAY (see freeHandleList comment), so even if a cache
  // somewhere captured this integer it has the delay to notice the release
  // before the same id reappears.
  handles[id] = undefined;
  bufferSizesArr[id] = undefined;
  bufferLogicalSizesArr[id] = undefined;
  bufferUsagesArr[id] = undefined;
  clearSharedBufferMeta(id);
  freeHandleList.push(id);
}

function invalidateBindGroupCacheForHandle(handle: number): void {
  for (const [pipelineHandle, entries] of bindGroupCache) {
    const kept = entries.filter((cached) => {
      for (let i = 0; i < cached.entryCount; i++) {
        if (cached.entries[i * 3] === handle) {
          return false;
        }
      }
      return true;
    });
    if (kept.length === 0) {
      bindGroupCache.delete(pipelineHandle);
    } else if (kept.length !== entries.length) {
      bindGroupCache.set(pipelineHandle, kept);
    }
  }
}

function hasUnsubmittedGpuWork(): boolean {
  return (
    activeCommandEncoders.size > 0 ||
    pendingCommandBuffers.size > 0 ||
    passEncoderMap.size > 0
  );
}

function poolBufferHandleNow(
  handle: number,
  size: number,
  usage: number,
): void {
  if (!isPoolableSize(size)) {
    destroyBufferHandleAfterSubmittedWork(handle);
    return;
  }
  const key = poolKey(usage, size);
  let stack = bufferPool.get(key);
  if (stack === undefined) {
    stack = [];
    bufferPool.set(key, stack);
  }
  if (stack.length < bufferPoolCapForSize(size)) {
    stack.push(handle);
    bufferPoolBytes += size;
    while (bufferPoolBytes > BUFFER_POOL_MAX_BYTES) {
      if (!evictOnePooledBuffer()) break;
    }
    return;
  }
  destroyBufferHandleAfterSubmittedWork(handle);
}

function evictOnePooledBuffer(): boolean {
  for (const [key, stack] of bufferPool) {
    const evicted = stack.shift();
    if (evicted === undefined) {
      bufferPool.delete(key);
      continue;
    }
    const sep = key.indexOf(":");
    const size = Number(key.slice(sep + 1));
    removeBufferPoolBytes(size);
    destroyBufferHandleAfterSubmittedWork(evicted);
    if (stack.length === 0) bufferPool.delete(key);
    return true;
  }
  return false;
}

function destroyBufferHandleAfterSubmittedWork(handle: number): void {
  const buffer = handles[handle] as GPUBuffer | undefined;
  releaseHandle(handle);
  if (buffer !== undefined) {
    pendingBufferDestroys.push(buffer);
    schedulePendingBufferDestroyBatch();
  }
}

function drainPendingBufferDestroys(): void {
  if (pendingBufferDestroys.length === 0) return;
  const buffers = pendingBufferDestroys.splice(0);
  for (const buffer of buffers) {
    try {
      buffer.destroy();
    } catch (err) {
      console.error("[GPU Worker] GPUBuffer destroy failed:", err);
    }
  }
}

function schedulePendingBufferDestroyBatch(): void {
  if (pendingBufferDestroyBatchScheduled) return;
  pendingBufferDestroyBatchScheduled = true;
  queueMicrotask(() => {
    pendingBufferDestroyBatchScheduled = false;
    if (pendingBufferDestroys.length === 0) return;
    const buffers = pendingBufferDestroys.splice(0);
    void queue.onSubmittedWorkDone().then(
      () => {
        for (const buffer of buffers) {
          try {
            buffer.destroy();
          } catch (err) {
            console.error("[GPU Worker] deferred GPUBuffer destroy failed:", err);
          }
        }
      },
      (err) => {
        console.error("[GPU Worker] deferred GPUBuffer destroy failed:", err);
      },
    );
  });
}

function promoteQuarantinedBufferPoolHandles(): void {
  if (quarantinedBufferPoolHandles.length === 0) return;
  const handlesToPool = quarantinedBufferPoolHandles.splice(0);
  for (const handle of handlesToPool) {
    quarantinedBufferPoolSet.delete(handle);
    if (quarantinedBufferDestroySet.has(handle)) {
      continue;
    }
    const size = bufferSizesArr[handle];
    const usage = bufferUsagesArr[handle];
    if (handles[handle] === undefined) {
      continue;
    }
    if (size !== undefined && usage !== undefined && isPoolable(usage)) {
      poolBufferHandleNow(handle, size, usage);
    } else {
      destroyBufferHandleAfterSubmittedWork(handle);
    }
  }
}

function promoteQuarantinedBufferDestroyHandles(): void {
  if (quarantinedBufferDestroyHandles.length === 0) return;
  const handlesToDestroy = quarantinedBufferDestroyHandles.splice(0);
  for (const handle of handlesToDestroy) {
    quarantinedBufferDestroySet.delete(handle);
    if (handles[handle] !== undefined) {
      destroyBufferHandleAfterSubmittedWork(handle);
    }
  }
}

function promoteQuarantinedBufferHandles(): void {
  promoteQuarantinedBufferPoolHandles();
  promoteQuarantinedBufferDestroyHandles();
}

// Pool-aware buffer release — shared by BUFFER_RELEASE and BUFFER_RELEASE_BATCH.
// Small/medium buffers are parked for reuse; oversized or evicted buffers have
// their handles released immediately and their GPUBuffer.destroy() deferred until
// submitted queue work has completed.
function releaseBufferHandle(handle: number): void {
  // Task E: dedicated uniform buffers attached to a BindGroupCacheEntry are
  // owned by the pipeline's lifecycle, not by a per-dispatch pool release.
  // The bridge stub emits a BUFFER_RELEASE_BATCH entry for the monotonic
  // handle we returned from the deferred-uniform HOT MISS path; swallow it
  // here so the GPUBuffer survives until PIPELINE_RELEASE.
  if (dedicatedUniformHandles.has(handle)) return;
  if (quarantinedBufferDestroySet.has(handle)) return;
  const size = bufferSizesArr[handle];
  const usage = bufferUsagesArr[handle];
  if (size !== undefined && usage !== undefined) {
    if (hasUnsubmittedGpuWork()) {
      if (!quarantinedBufferPoolSet.has(handle)) {
        quarantinedBufferPoolSet.add(handle);
        quarantinedBufferPoolHandles.push(handle);
      }
      return;
    }
    if (isPoolable(usage)) {
      poolBufferHandleNow(handle, size, usage);
    } else {
      destroyBufferHandleAfterSubmittedWork(handle);
    }
    return;
  }
  if (hasUnsubmittedGpuWork()) {
    if (!quarantinedBufferPoolSet.has(handle)) {
      quarantinedBufferPoolSet.add(handle);
      quarantinedBufferPoolHandles.push(handle);
    }
    return;
  }
  destroyBufferHandleAfterSubmittedWork(handle);
}

// ---------- Memory Helpers (1J: cached WASM DataView) ----------

let cachedWasmBuffer: ArrayBuffer | null = null;
let cachedWasmByteLength = 0;
let cachedWasmView: DataView | null = null;
let cachedWasmU8: Uint8Array | null = null;

// JS-F019: rebuild views when either the buffer identity changes (non-shared
// memory grow path) OR the byteLength changes (Chromium's shared-memory grow
// preserves identity but still grows byteLength; a Uint8Array constructed
// against the pre-grow SAB captures the old length and becomes undersized).
// Without the length tiebreaker, accesses past the old end read undefined on
// U8 or throw on DataView.
function refreshCacheIfGrew(buf: ArrayBuffer): void {
  const len = buf.byteLength;
  if (buf === cachedWasmBuffer && len === cachedWasmByteLength) return;
  cachedWasmBuffer = buf;
  cachedWasmByteLength = len;
  cachedWasmView = new DataView(buf);
  cachedWasmU8 = new Uint8Array(buf);
}

function getWasmView(): DataView {
  refreshCacheIfGrew(wasmMemoryObj.buffer);
  return cachedWasmView!;
}

function getWasmBytes(): Uint8Array {
  refreshCacheIfGrew(wasmMemoryObj.buffer);
  return cachedWasmU8!;
}

function readString(ptr: number): string {
  if (ptr === 0) return "";
  const bytes = getWasmBytes();
  let end = ptr;
  const maxLen = bytes.byteLength;
  while (end < maxLen && bytes[end] !== 0) end++;
  return new TextDecoder().decode(bytes.slice(ptr, end));
}

// JS-F005: zero-copy source for queue.writeBuffer on engines that accept
// Uint8Array over SharedArrayBuffer (Chromium today). Falls back to the
// slice path on engines that reject shared views (historically Safari).
// `ptr`/`size` are WASM byte offsets. Callers must NOT mutate WASM memory
// until the next Atomics.waitAsync yield — queue.writeBuffer snapshots
// the bytes synchronously inside the call, after which the shared view
// is no longer load-bearing.
function wasmBytesForWrite(ptr: number, size: number): Uint8Array {
  const base = getWasmBytes();
  if (queueWriteBufferSupportsSharedView) {
    return new Uint8Array(base.buffer, base.byteOffset + ptr, size);
  }
  return base.slice(ptr, ptr + size);
}

// ---------- GPU State ----------

let device: GPUDevice;
let queue: GPUQueue;
let adapter: GPUAdapter;
let hasShaderF16 = false;

// Track pass→encoder association. Normal operation lets the C++ backend emit
// pass boundaries from semantic read/write array tracking. Keep the forced
// restart switch as a local validation fallback while tuning the tracker.
const passEncoderMap = new Map<number, number>(); // passHandle → encoderHandle

const RESTART_PASS_AFTER_EACH_DISPATCH = false;

function endAndRestartPassIfForced(passHandle: number): void {
  if (!RESTART_PASS_AFTER_EACH_DISPATCH) return;
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

// Phase 2: dispatch batch SABs — one per worker (main + child pthreads).
// The mlx-worker passes an array of buffers in init; we map them by ID.
// Fallback: a single buffer from older inits is stored at ID 0.
interface BatchBuffer {
  sab: SharedArrayBuffer;
  u32: Uint32Array;
  u8: Uint8Array;
}
const batchBuffers = new Map<number, BatchBuffer>();
// Legacy fallback for single-buffer inits (test harnesses, etc.)
let dispatchBatchU32: Uint32Array | null = null;
let dispatchBatchU8: Uint8Array | null = null;
let dispatchBatchBuffer: SharedArrayBuffer | null = null;

// JS-F010: dedicated stats SAB. When the mlx-worker passes `statsBuffer` in
// the init message, GET_STATS writes the per-opcode histogram there instead
// of overlapping the cmd-SAB's CALLBACK_BASE / UNIFORM_DATA regions. Falls
// back to null on older inits (e.g. test harnesses that don't plumb it) —
// GET_STATS then degrades to writing only cmdU32[I_RESULT] = gpuTotalRpcs
// with no per-opcode histogram, which is harmless for tests that don't
// inspect the histogram.
let statsU32: Uint32Array | null = null;

// JS-F005: runtime-probed flag — is this WebGPU implementation happy to
// accept a Uint8Array over SharedArrayBuffer as the data argument to
// queue.writeBuffer? Chromium-based browsers (Chrome, Edge, Opera) do;
// Safari's WebKit has historically rejected shared views, forcing a slice
// into a fresh non-shared ArrayBuffer. Set once in init after the device
// is up (see the probe block) and read by QUEUE_WRITE_BUFFER /
// CREATE_BUFFER_FROM_DATA / BUFFER_UNMAP on the hot path.
let queueWriteBufferSupportsSharedView = false;

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
//
// JS-F011: backing storage is a fixed Uint32Array (3 u32s per slot) instead
// of an Array<PendingCallback>. Push is 3 u32 writes + cursor bump — no
// object alloc. Drain still allocates a PendingCallback per entry to go
// through pushCallback's existing contract; that is cheaper than pushing
// objects on every QUEUE_ON_SUBMITTED_WORK_DONE in the first place.
const GPU_DONE_CAPACITY = 512;
const gpuDoneBuf = new Uint32Array(GPU_DONE_CAPACITY * 3);
let gpuDoneCount = 0;

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
  if (e.data.type === "init") {
    cmdBuffer = e.data.cmdBuffer;
    wasmMemoryObj = e.data.wasmMemory; // WebAssembly.Memory object
    readbackView = new Uint8Array(e.data.readbackBuffer);

    cmdView = new Int32Array(cmdBuffer);
    cmdDataView = new DataView(cmdBuffer);
    cmdU32 = new Uint32Array(cmdBuffer);

    if (e.data.batchBuffers && Array.isArray(e.data.batchBuffers)) {
      for (let i = 0; i < e.data.batchBuffers.length; i++) {
        const buf = e.data.batchBuffers[i] as SharedArrayBuffer;
        batchBuffers.set(i, {
          sab: buf,
          u32: new Uint32Array(buf),
          u8: new Uint8Array(buf),
        });
      }
      // Also set legacy fallback to ID 0 for any code that still reads it
      const first = batchBuffers.get(0);
      if (first) {
        dispatchBatchBuffer = first.sab;
        dispatchBatchU32 = first.u32;
        dispatchBatchU8 = first.u8;
      }
    } else if (e.data.dispatchBatchBuffer) {
      // Legacy single-buffer init (test harnesses)
      const buf = e.data.dispatchBatchBuffer as SharedArrayBuffer;
      dispatchBatchBuffer = buf;
      dispatchBatchU32 = new Uint32Array(buf);
      dispatchBatchU8 = new Uint8Array(buf);
      batchBuffers.set(0, {
        sab: buf,
        u32: dispatchBatchU32,
        u8: dispatchBatchU8,
      });
    }

    // JS-F010: dedicated stats SAB (1 KiB = 256 u32 slots). Optional —
    // when absent, GET_STATS still returns the scalar total via RESULT but
    // the per-opcode histogram is a no-op. Every production init (mlx-worker)
    // supplies this; test harnesses may not.
    if (
      e.data.statsBuffer &&
      (e.data.statsBuffer as SharedArrayBuffer).byteLength >= STATS_BUFFER_SIZE
    ) {
      statsU32 = new Uint32Array(
        e.data.statsBuffer as SharedArrayBuffer,
        0,
        STATS_OPCODE_SLOTS,
      );
    }
    if (
      e.data.bufferMetadataBuffer &&
      (e.data.bufferMetadataBuffer as SharedArrayBuffer).byteLength >=
        BUFFER_METADATA_MAX_HANDLES * 8
    ) {
      bufferMetaU32 = new Uint32Array(
        e.data.bufferMetadataBuffer as SharedArrayBuffer,
        0,
        BUFFER_METADATA_MAX_HANDLES * 2,
      );
    }

    // Create GPU device
    const gpu = navigator.gpu;
    if (!gpu) {
      self.postMessage({
        type: "error",
        message: "WebGPU not available in gpu-worker",
      });
      return;
    }

    const _adapter = await gpu.requestAdapter();
    if (!_adapter) {
      self.postMessage({
        type: "error",
        message: "No WebGPU adapter available",
      });
      return;
    }
    adapter = _adapter;

    // 1A: Runtime feature detection
    hasShaderF16 = adapter.features.has("shader-f16");
    const hasSubgroups = adapter.features.has("subgroups");
    const hasTimestampQuery = adapter.features.has("timestamp-query");
    const requiredFeatures: GPUFeatureName[] = [];
    if (hasShaderF16) requiredFeatures.push("shader-f16");
    if (hasSubgroups) requiredFeatures.push("subgroups");
    if (hasTimestampQuery) requiredFeatures.push("timestamp-query");
    console.log(
      `[GPU Worker] Detected features: shader-f16=${hasShaderF16}, subgroups=${hasSubgroups}, timestamp-query=${hasTimestampQuery}`,
    );

    const maxBufferSize = adapter.limits.maxBufferSize;
    const maxStorageBufferBindingSize = Math.min(
      adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize,
    );

    device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(
          adapter.limits.maxStorageBuffersPerShaderStage,
          16,
        ),
        maxBufferSize,
        maxStorageBufferBindingSize,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
        maxComputeWorkgroupSizeZ: 64,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupsPerDimension: 65535,
        // D=256 tile SDPA needs ~25 KiB shared memory (BQ*D + BK*D + extras).
        // The WebGPU minimum is 16384 (16 KiB); request the adapter's max
        // (32768 on Apple M3) so D=256 kernels can compile and run.
        maxComputeWorkgroupStorageSize:
          adapter.limits.maxComputeWorkgroupStorageSize,
        maxBindGroups: 4,
        maxBindingsPerBindGroup: Math.min(
          adapter.limits.maxBindingsPerBindGroup,
          16,
        ),
      },
    });
    queue = device.queue;

    device.onuncapturederror = (event) => {
      const error = (event as GPUUncapturedErrorEvent).error;
      const message = `${error.constructor.name}: ${error.message}`;
      fatalGpuError = message;
      console.error("[GPU Worker] Uncaptured error:", message);
      try {
        self.postMessage({ type: "gpu-error", message });
      } catch {
        // ignore postMessage failures
      }
    };

    // JS-F005: feature-probe `queue.writeBuffer` for shared-memory view support.
    //
    // QUEUE_WRITE_BUFFER (~146/tok on Qwen3.5-0.8B decode) and
    // CREATE_BUFFER_FROM_DATA (~37/tok) used to do
    //   `wasmBytes().slice(dataPtr, dataPtr + size)`
    // which fully copies source bytes into a fresh non-shared ArrayBuffer
    // just to hand a writable view to queue.writeBuffer. Chromium accepts
    // a Uint8Array over SharedArrayBuffer directly (so the slice is pure
    // overhead); Safari / Firefox historically rejected shared views.
    //
    // Probe once at init: create a tiny GPU buffer and try writing a
    // 4-byte shared view to it. If the call lands without throwing, cache
    // the result and let the hot-path sites pass shared views directly.
    // If it throws, keep the slice path — the probe itself is cheap (one
    // 4-byte createBuffer + one queue.writeBuffer that Chromium elides
    // when no compute work references the buffer).
    try {
      const probeBuf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const sharedU8 = new Uint8Array(wasmMemoryObj.buffer, 0, 4);
      queue.writeBuffer(probeBuf, 0, sharedU8);
      queueWriteBufferSupportsSharedView = true;
      probeBuf.destroy();
    } catch (err) {
      console.log(
        `[GPU Worker] queue.writeBuffer probe: shared-view path disabled (${err instanceof Error ? err.message : String(err)})`,
      );
      queueWriteBufferSupportsSharedView = false;
    }

    // Pre-register handles for pre-created objects
    instanceHandle = addHandle({ __brand: "instance" });
    adapterHandle = addHandle(adapter);
    deviceHandle = addHandle(device);
    queueHandle = addHandle(queue);

    self.postMessage({
      type: "ready",
      instanceHandle,
      adapterHandle,
      deviceHandle,
      queueHandle,
      features: {
        shaderF16: hasShaderF16,
        subgroups: hasSubgroups,
        timestampQuery: hasTimestampQuery,
      },
    });

    // Start command processing loop
    commandLoop();
  }

  if (e.data.type === "upload_weights") {
    // Bulk weight upload: read directly from a transferred ArrayBuffer (or
    // SharedArrayBuffer for older callers), create GPU buffers. Browser model
    // loading uses transferable ArrayBuffers for transient upload batches so
    // large sharded checkpoints do not exhaust the SharedArrayBuffer quota.
    //
    // Step B: mergedSab holds pre-concatenated linear_attn.in_proj_qkvz /
    // in_proj_ba bytes produced by the mlx-worker before upload. Tensors
    // tagged `fromMerged: true` read their bytes from mergedSab with NO
    // additional dataOffset adjustment (the merged SAB is a bare byte blob;
    // dataOffset is the safetensors data region offset and only applies to
    // weightsBuffer).
    const {
      weightsBuffer = e.data.weightsSab,
      mergedBuffer = e.data.mergedSab,
      dataOffset,
      tensors,
      packBf16,
    } = e.data as {
      weightsBuffer?: ArrayBuffer | SharedArrayBuffer;
      weightsSab?: ArrayBuffer | SharedArrayBuffer;
      mergedBuffer?: ArrayBuffer | SharedArrayBuffer;
      mergedSab?: ArrayBuffer | SharedArrayBuffer;
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
      if (!name.endsWith(".weight")) return false;
      // Linear projections — all routed through linear_proj() →
      // matmul(x, transpose(weight)) in mlx_qwen35_common.h. Both decode (GEMV)
      // and prefill (GEMM) of these weights consume the buffer with
      // b_transposed=true, which is the only path that has a packed kernel.
      // MLP (SwiGLU) projections.
      if (name.endsWith(".mlp.gate_proj.weight")) return true;
      if (name.endsWith(".mlp.up_proj.weight")) return true;
      if (name.endsWith(".mlp.down_proj.weight")) return true;
      // Self-attention projections.
      if (name.endsWith(".self_attn.q_proj.weight")) return true;
      if (name.endsWith(".self_attn.k_proj.weight")) return true;
      if (name.endsWith(".self_attn.v_proj.weight")) return true;
      if (name.endsWith(".self_attn.o_proj.weight")) return true;
      // Linear (gated-delta) attention projections. in_proj_qkvz /
      // in_proj_ba are PRE-MERGED in JS (mlx-worker.ts) before upload so
      // the merged buffer takes its OWN PackedBf16 opt-in on first upload
      // — no Rust concat, no storage_mode propagation needed. out_proj is
      // already a single weight. All three land on matmul(x, W.T) via
      // linear_proj() in mlx_qwen35_common.h → b_transposed=true packed
      // kernel.
      if (name.endsWith(".linear_attn.out_proj.weight")) return true;
      if (name.endsWith(".linear_attn.in_proj_qkvz.weight")) return true;
      if (name.endsWith(".linear_attn.in_proj_ba.weight")) return true;
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
      if (!name.endsWith(".weight")) return false;
      // Reject vision-tower tensors explicitly — belt and suspenders in
      // case a future refactor drops the endsWith-specific check below.
      if (name.startsWith("vision_tower.")) return false;
      // Per-layer pre-attention RMSNorm.
      if (name.endsWith(".input_layernorm.weight")) return true;
      // Per-layer post-attention RMSNorm (feeds the MLP).
      if (name.endsWith(".post_attention_layernorm.weight")) return true;
      // Per-layer full-attention q/k RMSNorm (GQA head normalization).
      if (name.endsWith(".self_attn.q_norm.weight")) return true;
      if (name.endsWith(".self_attn.k_norm.weight")) return true;
      // Final LM-head norm. Matches both the text-only (`model.norm.weight`)
      // and VL (`language_model.model.norm.weight`) names via the shared
      // `.model.norm.weight` tail, without leaking matches to
      // `.linear_attn.norm.weight` or any `vision_tower.*.norm.weight`.
      if (name === "model.norm.weight") return true;
      if (name.endsWith(".model.norm.weight")) return true;
      return false;
    };

    const handles: number[] = [];
    const uploadedDtypes: string[] = [];
    const uploadedByteSizes: number[] = [];
    const packedBf16Flags: boolean[] = [];
    const packedNames: string[] = [];
    const unpackedBf16Names: string[] = [];
    const MAX_WRITE_CHUNK_BYTES = 16 * 1024 * 1024;
    const writeBufferChunked = (buffer: GPUBuffer, data: Uint8Array): void => {
      for (
        let offset = 0;
        offset < data.byteLength;
        offset += MAX_WRITE_CHUNK_BYTES
      ) {
        const size = Math.min(MAX_WRITE_CHUNK_BYTES, data.byteLength - offset);
        queue.writeBuffer(buffer, offset, data, offset, size);
      }
    };
    for (const tensor of tensors) {
      const isBf16 = tensor.dtype === "BF16";
      const isF16 = tensor.dtype === "F16";
      const numElements = tensor.byteSize / (isBf16 || isF16 ? 2 : 4);
      // Synthetic (pre-merged) tensors live in mergedBuffer starting at
      // tensor.byteOffset (a raw offset into the merged blob, NOT offset by
      // the safetensors dataOffset). Real safetensors tensors live in
      // weightsBuffer at dataOffset + tensor.byteOffset.
      const srcBuffer = tensor.fromMerged ? mergedBuffer : weightsBuffer;
      const srcByteOffset = tensor.fromMerged
        ? tensor.byteOffset
        : dataOffset + tensor.byteOffset;
      if (tensor.fromMerged && !mergedBuffer) {
        throw new Error(
          `Tensor ${tensor.name} marked fromMerged but mergedBuffer missing`,
        );
      }
      if (!srcBuffer) {
        throw new Error(`Upload buffer missing for tensor ${tensor.name}`);
      }

      // Packed bf16 path: reinterpret consecutive bf16 pairs as u32 slots.
      // Each slot holds two bf16 elements (low | hi << 16) — exactly the
      // layout the WebGPU backend's upload_with_conversion path produces.
      // Odd element counts pad the trailing slot with zero; the kernel
      // masks out-of-range loads so the pad is harmless.
      //
      // Only matmul-consumed weights are packed during JS-side upload. Norm
      // weights can require loader-side sanitization before use, so keep them
      // on the legacy path until the post-sanitized GPU load path can pack them.
      const matmulConsumed =
        usePackBf16 &&
        isBf16 &&
        numElements >= PACKED_MIN_ELEMENTS &&
        isMatmulConsumedWeight(tensor.name);
      const normConsumed = false;
      const packedEligible = matmulConsumed || normConsumed;
      if (
        isBf16 &&
        usePackBf16 &&
        (numElements >= PACKED_MIN_ELEMENTS ||
          (numElements >= NORM_PACKED_MIN_ELEMENTS &&
            isNormConsumedWeight(tensor.name)))
      ) {
        if (packedEligible) {
          packedNames.push(tensor.name);
        } else {
          unpackedBf16Names.push(tensor.name);
        }
      }
      // bf16 expands to f32 for legacy storage; f16 stays native when shader-f16 is available.
      const needsExpand =
        (isBf16 && !packedEligible) || (isF16 && !hasShaderF16);

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

      const uploadBytes = new Uint8Array(alignedSize);
      if (packedEligible) {
        // bf16 is already 2 bytes per element and contiguous in the source
        // buffer; a u32 view of the destination stores pairs directly.
        // Note: both the source SAB and the mapped buffer are little-endian
        // wasm targets, so a raw byte copy preserves the (lo, hi) layout
        // that unpack_bf16_pair() expects in WGSL.
        const src = new Uint8Array(srcBuffer, srcByteOffset, tensor.byteSize);
        uploadBytes.set(src);
        // Pad the trailing odd slot with zero bytes (already zero-initialized
        // in uploadBytes) — explicit comment for future maintainers.
        uploadedDtypes.push("BF16");
        packedBf16Flags.push(true);
      } else if (needsExpand) {
        // Convert bf16/f16 → f32 in a CPU-side upload buffer.
        const src16 = new Uint16Array(srcBuffer, srcByteOffset, numElements);
        const dst32 = new Uint32Array(uploadBytes.buffer);
        if (isBf16) {
          // bf16 → f32: shift left by 16 (bf16 is upper 16 bits of f32)
          for (let j = 0; j < numElements; j++) {
            dst32[j] = src16[j] << 16;
          }
        } else {
          // f16 → f32: proper IEEE 754 conversion
          const dstF32 = new Float32Array(uploadBytes.buffer);
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
              dstF32[j] =
                (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
            }
          }
        }
        // BF16 tensors are stored as f32 lanes on WebGPU, but the MLX array
        // must remain logically BF16 so downstream ops pick the same dtypes
        // as native/common loading. The WebGPU backend tracks the widened
        // physical storage separately.
        uploadedDtypes.push(isBf16 ? "BF16" : "F32");
        packedBf16Flags.push(false);
      } else {
        const src = new Uint8Array(srcBuffer, srcByteOffset, tensor.byteSize);
        uploadBytes.set(src);
        uploadedDtypes.push(tensor.dtype);
        packedBf16Flags.push(false);
      }

      const gpuBuffer = device.createBuffer({
        size: alignedSize,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
      });
      writeBufferChunked(gpuBuffer, uploadBytes);

      const handle = addHandle(gpuBuffer);
      bufferSizesArr[handle] = alignedSize;
      bufferLogicalSizesArr[handle] = alignedSize;
      bufferUsagesArr[handle] =
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST;
      setSharedBufferMeta(
        handle,
        alignedSize,
        GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      );
      handles.push(handle);
      uploadedByteSizes.push(alignedSize);
    }

    self.postMessage({
      type: "weights_uploaded",
      handles,
      uploadedDtypes,
      uploadedByteSizes,
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
  const layout = getHandle<GPUBindGroupLayout>(
    layoutHandle,
    "bind-group layout",
  );
  let newBufferHandle = 0;
  // Task E: if this dispatch's deferred-uniform path allocated a dedicated
  // uniform buffer (HOT MISS), remember it so the bind-group-cache insert
  // below can attach it to the new BindGroupCacheEntry. HOT HIT reuses the
  // already-attached buffer and skips the pool entirely.
  let hotNewUniformBuffer: GPUBuffer | null = null;
  let hotNewUniformHandle = 0;

  // Stage 1: optional uniform write (only for FUSED_DISPATCH_WITH_UNIFORM).
  // Must run BEFORE createBindGroup so the patched handle (for deferred
  // creation) lands in entries[uniformEntryIdx].
  if (
    opcode === RpcFn.FUSED_DISPATCH_WITH_UNIFORM &&
    uniformDataSize > 0 &&
    uniformBuffer !== null
  ) {
    const uniformBufHandle = entrySrc[entrySrcBase + uniformEntryIdx * 3];
    const uniformData = new Uint8Array(
      uniformBuffer,
      uniformByteOffset,
      uniformDataSize,
    );
    if (uniformBufHandle === 0) {
      // This dispatch needs an inline parameter buffer. Do not reuse one buffer
      // across recorded dispatches: command buffers keep a reference to the
      // GPUBuffer, not a snapshot of the bytes written before each dispatch.
      const usage = uniformUsage || 0x0080 | 0x0008; // STORAGE | COPY_DST fallback
      const buf = device.createBuffer({ size: 256, usage });
      const h = addHandle(buf);
      bufferSizesArr[h] = 256;
      bufferLogicalSizesArr[h] = uniformDataSize;
      bufferUsagesArr[h] = usage;
      queue.writeBuffer(buf, 0, uniformData);
      uniformHotMisses++;
      entrySrc[entrySrcBase + uniformEntryIdx * 3] = h;
      entrySrc[entrySrcBase + uniformEntryIdx * 3 + 1] = 256;
      entrySrc[entrySrcBase + uniformEntryIdx * 3 + 2] = 0;
      newBufferHandle = h;
    } else {
      const existing = getHandle<GPUBuffer>(uniformBufHandle, "uniform buffer");
      queue.writeBuffer(existing, 0, uniformData);
    }
  }

  // Stage 2: probe the bind group cache. We compare *after* Stage 1 so the
  // deferred-uniform handle patch at entries[uniformEntryIdx] is included —
  // otherwise every pool-hit dispatch would spuriously cache-miss because the
  // compare would see "0" while the post-patch handle is the real pooled one.
  //
  // JS-F004: compare-by-integer against a cached Uint32Array snapshot instead
  // of constructing a fresh string signature every dispatch. The per-dispatch
  // string alloc (60-100 chars × 3-4 concat temporaries) is the dominant
  // young-gen GC pressure on the hot path — hundreds of dispatches/token. The
  // compare loop below is at most entryCount * 3 integer compares (≤ 30).
  const cachedEntriesForPipeline = bindGroupCache.get(pipelineHandle);
  const compareLen = entryCount * 3;
  let bindGroup: GPUBindGroup;
  let hit = false;
  let hitIndex = -1;
  if (cachedEntriesForPipeline !== undefined) {
    for (let c = cachedEntriesForPipeline.length - 1; c >= 0; c--) {
      const cached = cachedEntriesForPipeline[c]!;
      if (
        cached.layoutHandle !== layoutHandle ||
        cached.entryCount !== entryCount
      ) {
        continue;
      }
      hit = true;
      const cachedEntries = cached.entries;
      for (let i = 0; i < compareLen; i++) {
        if (cachedEntries[i] !== entrySrc[entrySrcBase + i]) {
          hit = false;
          break;
        }
      }
      if (hit) {
        hitIndex = c;
        break;
      }
    }
  }
  if (hit) {
    const cached = cachedEntriesForPipeline![hitIndex]!;
    bindGroup = cached.bindGroup;
    if (hitIndex !== cachedEntriesForPipeline!.length - 1) {
      cachedEntriesForPipeline!.splice(hitIndex, 1);
      cachedEntriesForPipeline!.push(cached);
    }
    bindGroupCacheHits++;
  } else {
    const entries: GPUBindGroupEntry[] = [];
    let eIdx = entrySrcBase;
    for (let i = 0; i < entryCount; i++) {
      const buffer = getHandle<GPUBuffer>(
        entrySrc[eIdx],
        `bind-group buffer[${i}]`,
      );
      const sizeLo = entrySrc[eIdx + 1];
      const sizeHi = entrySrc[eIdx + 2];
      eIdx += 3;
      const size = sizeLo + sizeHi * 0x100000000;
      const resource: GPUBufferBinding = { buffer, offset: 0 };
      if (size !== 0 && size < 2 ** 53) resource.size = size;
      entries.push({ binding: i, resource });
    }
    bindGroup = device.createBindGroup({ layout, entries });
    const newUniformBuffer: GPUBuffer | null = null;
    const newUniformHandle = 0;
    // Snapshot the entries so the next dispatch can compare element-by-element.
    // Using subarray + new Uint32Array copies just the live slice (entryCount*3)
    // and is cheap relative to the createBindGroup above.
    const snapshot = new Uint32Array(compareLen);
    snapshot.set(entrySrc.subarray(entrySrcBase, entrySrcBase + compareLen));
    let cacheList = bindGroupCache.get(pipelineHandle);
    if (cacheList === undefined) {
      cacheList = [];
      bindGroupCache.set(pipelineHandle, cacheList);
    }
    cacheList.push({
      layoutHandle,
      entryCount,
      entries: snapshot,
      bindGroup,
      uniformBuffer: newUniformBuffer,
      uniformHandle: newUniformHandle,
    });
    if (cacheList.length > BIND_GROUP_CACHE_PER_PIPELINE) {
      cacheList.shift();
    }
    bindGroupCacheMisses++;
  }

  // Stage 3: issue the dispatch on the cached compute pass.
  const pass = getHandle<GPUComputePassEncoder>(passHandle, "compute pass");
  pass.setPipeline(
    getHandle<GPUComputePipeline>(pipelineHandle, "compute pipeline"),
  );
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y, z);
  endAndRestartPassIfForced(passHandle);
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
  const batchBufferId = cmdU32[I_ARG0 + 2] ?? 0;
  const buf = batchBuffers.get(batchBufferId);
  if (!buf) {
    throw new Error(
      `[GPU Worker] Missing DISPATCH_BATCH buffer id=${batchBufferId}`,
    );
  }
  const batchCount = cmdU32[I_ARG0];
  const batchBytes = cmdU32[I_ARG0 + 1];
  if (batchCount > MAX_DISPATCH_BATCH) {
    throw new Error(
      `[GPU Worker] Invalid DISPATCH_BATCH count=${batchCount} max=${MAX_DISPATCH_BATCH}`,
    );
  }
  if (batchCount === 0 && batchBytes > 0) {
    throw new Error(
      `[GPU Worker] Invalid DISPATCH_BATCH bytes=${batchBytes} with count=0`,
    );
  }
  if (batchBytes > DISPATCH_BATCH_BUFFER_SIZE) {
    throw new Error(
      `[GPU Worker] Invalid DISPATCH_BATCH bytes=${batchBytes} max=${DISPATCH_BATCH_BUFFER_SIZE}`,
    );
  }
  if ((batchBytes & 3) !== 0) {
    throw new Error(
      `[GPU Worker] Invalid DISPATCH_BATCH bytes=${batchBytes}; expected 4-byte alignment`,
    );
  }
  const sab = buf.u32;
  const sabBuffer = buf.sab;
  let cursor = 0;
  for (let r = 0; r < batchCount; r++) {
    if (cursor >= batchBytes) {
      throw new Error(
        `[GPU Worker] DISPATCH_BATCH ended early record=${r} count=${batchCount} cursor=${cursor} bytes=${batchBytes}`,
      );
    }
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
    if (
      opcode !== RpcFn.FUSED_FULL_DISPATCH &&
      opcode !== RpcFn.FUSED_DISPATCH_WITH_UNIFORM
    ) {
      throw new Error(
        `[GPU Worker] Invalid DISPATCH_BATCH opcode=${opcode} record=${r}`,
      );
    }
    if (entryCount > MAX_DISPATCH_BATCH_ENTRIES) {
      throw new Error(
        `[GPU Worker] Invalid DISPATCH_BATCH entryCount=${entryCount} record=${r} max=${MAX_DISPATCH_BATCH_ENTRIES}`,
      );
    }
    if (uniformSize > MAX_DISPATCH_BATCH_UNIFORM) {
      throw new Error(
        `[GPU Worker] Invalid DISPATCH_BATCH uniformSize=${uniformSize} record=${r} max=${MAX_DISPATCH_BATCH_UNIFORM}`,
      );
    }
    if (uniformSize > 0 && uniformEntryIdx >= entryCount) {
      throw new Error(
        `[GPU Worker] Invalid DISPATCH_BATCH uniformEntryIdx=${uniformEntryIdx} entryCount=${entryCount} record=${r}`,
      );
    }
    const entryBase = base + (DISPATCH_BATCH_HEADER_BYTES >>> 2);
    const uniformByteOffset =
      cursor + DISPATCH_BATCH_HEADER_BYTES + entryCount * 12;
    const uniformPadded = (uniformSize + 3) & ~3;
    const nextCursor =
      cursor + DISPATCH_BATCH_HEADER_BYTES + entryCount * 12 + uniformPadded;
    if (nextCursor > batchBytes) {
      throw new Error(
        `[GPU Worker] Truncated DISPATCH_BATCH record=${r} cursor=${cursor} next=${nextCursor} bytes=${batchBytes}`,
      );
    }
    let newBufferHandle = 0;
    currentDispatchContext = `batchRecord=${r} opcode=${opcode} pass=${passHandle} pipeline=${pipelineHandle} layout=${layoutHandle}`;
    try {
      newBufferHandle = dispatchFusedRecord(
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
    } finally {
      currentDispatchContext = "";
    }
    if (newBufferHandle !== 0) {
      throw new Error(
        `[GPU Worker] DISPATCH_BATCH unexpectedly created deferred uniform handle=${newBufferHandle} record=${r}`,
      );
    }
    cursor = nextCursor;
  }
  if (cursor !== batchBytes) {
    throw new Error(
      `[GPU Worker] DISPATCH_BATCH trailing bytes cursor=${cursor} bytes=${batchBytes} count=${batchCount}`,
    );
  }
  cmdU32[I_RESULT] = 0;
}

function handleFusedSubmit(): void {
  const encoderHandle = cmdU32[I_ARG0];
  const passHandle = cmdU32[I_ARG0 + 1];
  if (passHandle > 0) {
    getHandle<GPUComputePassEncoder>(passHandle, "fused submit pass").end();
    passEncoderMap.delete(passHandle);
    releaseHandle(passHandle);
  }
  const encoder = getHandle<GPUCommandEncoder>(
    encoderHandle,
    "fused submit encoder",
  );
  const cmdBuf = encoder.finish();
  queue.submit([cmdBuf]);
  activeCommandEncoders.delete(encoderHandle);
  promoteQuarantinedBufferHandles();
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

  const encoder = getHandle<GPUCommandEncoder>(
    encoderHandle,
    "fused copy encoder",
  );

  if (passHandle !== 0) {
    getHandle<GPUComputePassEncoder>(passHandle, "fused copy pass").end();
    passEncoderMap.delete(passHandle);
  }

  encoder.copyBufferToBuffer(
    getHandle<GPUBuffer>(srcHandle, "fused copy src"),
    srcOffset,
    getHandle<GPUBuffer>(dstHandle, "fused copy dst"),
    dstOffset,
    size,
  );

  const newPass = encoder.beginComputePass();
  // copyBufferToBuffer is void in the WebGPU API, so the C++ side keeps using
  // the same pass handle. Replace the ended pass in-place to preserve that
  // cached handle across the fused copy/restart sequence.
  const newPassHandle = passHandle !== 0 ? passHandle : addHandle(newPass);
  handles[newPassHandle] = newPass;
  passEncoderMap.set(newPassHandle, encoderHandle);
  cmdU32[I_RESULT] = newPassHandle;
}

async function commandLoop(): Promise<void> {
  // Adaptive spin: after dispatch commands (likely followed by another
  // dispatch), spin up to spinDispatchBudget iterations before falling back
  // to Atomics.waitAsync. After submit/readback, skip spin (long gap expected).
  // Budget adapts — see SPIN_MIN/SPIN_MAX/SPIN_ADAPT_K at module scope.
  let spinBudget = 0; // set after each command based on expected follow-up

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

      // Adapt the budget only when we actually attempted a dispatch-style spin.
      if (spinBudget > 0) {
        if (spun) {
          spinHitTotal++;
          spinConsecutiveHits++;
          spinConsecutiveMisses = 0;
          if (
            spinConsecutiveHits >= SPIN_ADAPT_K &&
            spinDispatchBudget < SPIN_MAX
          ) {
            spinDispatchBudget = Math.min(SPIN_MAX, spinDispatchBudget * 2);
            spinConsecutiveHits = 0;
          }
        } else {
          spinMissTotal++;
          spinConsecutiveMisses++;
          spinConsecutiveHits = 0;
          if (
            spinConsecutiveMisses >= SPIN_ADAPT_K &&
            spinDispatchBudget > SPIN_MIN
          ) {
            spinDispatchBudget = Math.max(SPIN_MIN, spinDispatchBudget >>> 1);
            spinConsecutiveMisses = 0;
          }
        }
      }

      if (!spun) {
        // Fall back to async wait. Never setTimeout(0) — that's a macrotask
        // hop costing 4-10ms; loop on Atomics.waitAsync instead so a late
        // wake-up between spin and wait collapses to a fresh atomic re-read.
        let waitLoops = 0;
        while (currentStatus !== STATUS.PENDING) {
          // Re-check atomically in case the wasm-worker flipped STATUS during
          // the spin-adapt work above. Cheap and avoids a waitAsync round-trip.
          currentStatus = Atomics.load(cmdView, STATUS_INDEX);
          if (currentStatus === STATUS.PENDING) break;

          const result = Atomics.waitAsync(
            cmdView,
            STATUS_INDEX,
            currentStatus,
          );
          if (result.async) {
            // Real wait — resolves when notify fires or timeout elapses.
            await result.value;
          }
          // result.async === false means the cell already moved off
          // `currentStatus` — re-read below and re-try without yielding.
          currentStatus = Atomics.load(cmdView, STATUS_INDEX);
          waitLoops++;
          if (waitLoops % 1000 === 0) {
            console.log(
              `[GPU Worker] waiting for PENDING, status=${currentStatus}, loops=${waitLoops}`,
            );
          }
        }
      }
    }

    const fnId = cmdU32[0]; // CMD_OFFSET.FN_ID / 4 = 0
    currentFnId = fnId;
    currentDispatchContext = "";

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
    if (fatalGpuError !== null) {
      const err = new Error(fatalGpuError);
      postRpcError(fnId, err, "uncaptured-gpu-error");
      cmdU32[I_ERROR] = 1;
      cmdU32[I_RESULT] = 0;
      spinBudget = 0;
    } else if (fnId === RpcFn.FUSED_DISPATCH_WITH_UNIFORM) {
      try {
        handleFusedDispatchWithUniform();
      } catch (err) {
        console.error(
          `[GPU Worker] Error in FUSED_DISPATCH_WITH_UNIFORM:`,
          err,
        );
        postRpcError(fnId, err, "FUSED_DISPATCH_WITH_UNIFORM");
        cmdU32[I_ERROR] = 1;
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = spinDispatchBudget;
    } else if (fnId === RpcFn.FUSED_FULL_DISPATCH) {
      try {
        handleFusedFullDispatch();
      } catch (err) {
        console.error(`[GPU Worker] Error in FUSED_FULL_DISPATCH:`, err);
        postRpcError(fnId, err, "FUSED_FULL_DISPATCH");
        cmdU32[I_ERROR] = 1;
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = spinDispatchBudget;
    } else if (fnId === RpcFn.DISPATCH_BATCH) {
      try {
        handleDispatchBatch();
      } catch (err) {
        console.error(`[GPU Worker] Error in DISPATCH_BATCH:`, err);
        postRpcError(fnId, err, "DISPATCH_BATCH");
        cmdU32[I_ERROR] = 1;
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = spinDispatchBudget;
    } else if (fnId === RpcFn.FUSED_COPY_BUFFER) {
      try {
        handleFusedCopyBuffer();
      } catch (err) {
        console.error(`[GPU Worker] Error in FUSED_COPY_BUFFER:`, err);
        postRpcError(fnId, err, "FUSED_COPY_BUFFER");
        cmdU32[I_ERROR] = 1;
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = spinDispatchBudget; // copy is followed by dispatches
    } else if (fnId === RpcFn.FUSED_SUBMIT) {
      try {
        handleFusedSubmit();
      } catch (err) {
        console.error(`[GPU Worker] Error in FUSED_SUBMIT:`, err);
        postRpcError(fnId, err, "FUSED_SUBMIT");
        cmdU32[I_ERROR] = 1;
        cmdU32[I_RESULT] = 0;
      }
      spinBudget = 0;
    } else {
      try {
        await processCommand(fnId);
      } catch (err) {
        console.error(`[GPU Worker] Error processing fn ${fnId}:`, err);
        // Phase 6c diag: forward RPC errors to parent so they surface in
        // test output (browser console isn't piped into the test runner).
        try {
          const msg = err instanceof Error ? err.message : String(err);
          self.postMessage({ type: "rpc-error", fnId, message: msg });
        } catch {
          // ignore postMessage failures
        }
        cmdU32[I_ERROR] = 1;
        cmdU32[I_RESULT] = 0;
      }
      flushCallbacks();
      spinBudget = 0;
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
const I_ERROR = CMD_OFFSET.ERROR >>> 2;

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
      if (!mappedAtCreation && isPoolable(usage) && isPoolableSize(size)) {
        // Prefer the stub-supplied override (already bucketed by the stub's
        // pool logic). Fall back to our own bucketing when the stub didn't
        // pass one (older callers, tests, or non-pooled create paths).
        const bucketSize =
          physicalOverride > 0 ? physicalOverride : roundUpBucket(size);
        const stack = bufferPool.get(poolKey(usage, bucketSize));
        if (stack !== undefined && stack.length > 0) {
          const reused = stack.pop()!;
          if (stack.length === 0) bufferPool.delete(poolKey(usage, bucketSize));
          removeBufferPoolBytes(bucketSize);
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
          throw e;
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
        throw e;
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
        throw new Error(
          "[GPU Worker] ShaderModule descriptor has no nextInChain",
        );
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
      const entryPoint = entryPointPtr ? readString(entryPointPtr) : "main";
      const pipeline = device.createComputePipeline({
        layout: "auto",
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
          const resource: GPUBufferBinding = {
            buffer: getHandle<GPUBuffer>(bufferHandle),
            offset,
          };
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
      const handle = addHandle(encoder);
      activeCommandEncoders.add(handle);
      setResult(handle);
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
        console.error("[GPU Worker] Device lost:", info.message);
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
        pendingCommandBuffers.delete(handle);
      }
      queue.submit(commandBuffers);
      promoteQuarantinedBufferHandles();
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
      // JS-F005: zero-copy shared view when supported; slice fallback
      // otherwise. Caller cannot mutate WASM between this call and
      // queue.writeBuffer returning — fine here, we're single-threaded
      // inside the dispatch switch.
      queue.writeBuffer(buffer, offset, wasmBytesForWrite(dataPtr, size));
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
      if (gpuDoneCount >= GPU_DONE_CAPACITY) {
        console.error(
          `[GPU Worker] gpuDoneBuf overflow at ${gpuDoneCount}; dropping callback. ` +
            `This means queue.onSubmittedWorkDone() is being registered faster than POLL drains it.`,
        );
      } else {
        const base = gpuDoneCount * 3;
        gpuDoneBuf[base] = callbackPtr;
        gpuDoneBuf[base + 1] = 0; // status
        gpuDoneBuf[base + 2] = userdataPtr;
        gpuDoneCount++;
      }
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
      activeCommandEncoders.delete(encoderHandle);
      const handle = addHandle(cmdBuf);
      pendingCommandBuffers.add(handle);
      setResult(handle);
      break;
    }

    case RpcFn.CMD_ENCODER_RELEASE: {
      const handle = arg0();
      activeCommandEncoders.delete(handle);
      releaseHandle(handle);
      setResult(0);
      break;
    }

    // ================================================================
    // Command Buffer
    // ================================================================
    case RpcFn.CMD_BUFFER_RELEASE: {
      const handle = arg0();
      pendingCommandBuffers.delete(handle);
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
      endAndRestartPassIfForced(passHandle);
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
      // The bridge keeps compute pass handles stable across backend wrapper
      // lifetimes and fused copy restarts. Releasing the JS handle here can
      // invalidate a pass that another worker still references by handle, so
      // only free handles that are no longer registered as active passes.
      if (!passEncoderMap.has(handle)) {
        releaseHandle(handle);
      }
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
      getHandle<GPUBuffer>(bufferHandle, "buffer for getSize");
      const size =
        bufferLogicalSizesArr[bufferHandle] ?? bufferSizesArr[bufferHandle];
      if (size === undefined) {
        throw new Error(
          `[GPU Worker] Missing size metadata for buffer ${bufferHandle}`,
        );
      }
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
      activeMappings.set(bufferHandle, {
        jsRange,
        wasmPtr,
        size,
        writeBack: true,
      });

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
      if (!readbackView) {
        throw new Error(
          `readback buffer is not initialized for ${size} byte mapped range`,
        );
      }
      if (size > readbackView.byteLength) {
        throw new Error(
          `readback ${size} bytes exceeds shared readback buffer ${readbackView.byteLength} bytes`,
        );
      }
      readbackView.set(src, 0);

      activeMappings.set(bufferHandle, {
        jsRange,
        wasmPtr,
        size,
        writeBack: false,
      });

      setResult(wasmPtr);
      break;
    }

    case RpcFn.BUFFER_UNMAP: {
      // Args: bufferHandle
      const bufferHandle = arg0();
      const buffer = getHandle<GPUBuffer>(bufferHandle);
      const mapping = activeMappings.get(bufferHandle);

      if (mapping?.writeBack) {
        // Write path: copy from WASM shadow into JS ArrayBuffer before unmap.
        // JS-F005: `dst.set(src)` already copies byte-by-byte, so there's no
        // benefit from slicing the source first — hand it a shared view and
        // skip one full memcpy. Unlike queue.writeBuffer, TypedArray.set is
        // spec-required to work across SharedArrayBuffer sources, so no
        // feature probe is needed here.
        const base = getWasmBytes();
        const src = new Uint8Array(
          base.buffer,
          base.byteOffset + mapping.wasmPtr,
          mapping.size,
        );
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
        console.error("[GPU Worker] mapAsync failed:", err);
        pushCallback({ fnPtr: callbackPtr, status: 1, userdataPtr });
      }
      setResult(0);
      break;
    }

    case RpcFn.BUFFER_DESTROY: {
      // Keep destroy as a lifecycle no-op in the browser bridge. MLX's WebGPU
      // allocator commonly calls destroy before release; release is the point
      // where we either pool or safely defer GPUBuffer.destroy().
      setResult(0);
      break;
    }

    case RpcFn.BUFFER_RELEASE: {
      // Pool fast path: park eligible GPUBuffers for reuse. Buffers that exceed
      // the byte budget, are too large, or have unpoolable usage are released
      // from the handle table and destroyed only after submitted work completes.
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
      // Browser/WASI dispatch recording is deliberately more asynchronous
      // than native WebGPU: every WASM worker has its own bridge stub and may
      // hold staged DISPATCH_BATCH records that reference a pipeline handle.
      // A pipeline release from one stub cannot flush sibling stubs' batches.
      // Deleting the GPU-worker handle here therefore lets a later staged
      // dispatch skip setPipeline and silently corrupt decode output.
      //
      // Pipelines are session-level cached objects in this backend, so keep
      // them alive for the worker lifetime. This mirrors the already no-op
      // bind-group-layout release path and is correctness-critical for
      // batched/fused browser inference.
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
      drainPendingBufferDestroys();
      // NOW it's safe to fire GPU-done callbacks. Move them to the ring
      // so flushCallbacks() writes them to the callback ring for the wasm-worker.
      if (gpuDoneCount > 0) {
        for (let i = 0; i < gpuDoneCount; i++) {
          const base = i * 3;
          pushCallback({
            fnPtr: gpuDoneBuf[base],
            status: gpuDoneBuf[base + 1],
            userdataPtr: gpuDoneBuf[base + 2],
          });
        }
        gpuDoneCount = 0;
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
      endAndRestartPassIfForced(passHandle);
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
      endAndRestartPassIfForced(passHandle);
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
      activeCommandEncoders.delete(encoderHandle);
      promoteQuarantinedBufferHandles();
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
      endAndRestartPassIfForced(passHandle);
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
        const uniformData = new Uint8Array(
          cmdBuffer,
          CMD_OFFSET.UNIFORM_DATA,
          uniformDataSize,
        );
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
      endAndRestartPassIfForced(passHandle);
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
        passEncoderMap.delete(passHandle);
      }

      // Perform the copy
      const src = getHandle<GPUBuffer>(srcHandle);
      const dst = getHandle<GPUBuffer>(dstHandle);
      encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);

      // Begin new compute pass for subsequent dispatches
      const newPass = encoder.beginComputePass();
      // Preserve the pass handle for callers that cache it across the void
      // copyBufferToBuffer boundary.
      const newPassHandle = passHandle !== 0 ? passHandle : addHandle(newPass);
      handles[newPassHandle] = newPass;
      passEncoderMap.set(newPassHandle, encoderHandle);
      setResult(newPassHandle);
      break;
    }

    case RpcFn.ADD_GPU_BUFFER: {
      // This is handled via postMessage, not RPC, because the GPU buffer
      // object can't be serialized through SharedArrayBuffer.
      // This case should not be reached; it exists for protocol completeness.
      console.warn(
        "[GPU Worker] ADD_GPU_BUFFER via RPC is not supported. Use postMessage.",
      );
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
        const poolable = isPoolable(usage) && isPoolableSize(size);
        // Only bucket for poolable usages — non-poolable buffers never
        // re-enter the pool, so bucketing would just waste memory with no
        // reuse payoff.
        const physicalSize = poolable ? roundUpBucket(size) : size;
        const stack = poolable
          ? bufferPool.get(poolKey(usage, physicalSize))
          : undefined;
        if (stack !== undefined && stack.length > 0) {
          handle = stack.pop()!;
          if (stack.length === 0)
            bufferPool.delete(poolKey(usage, physicalSize));
          removeBufferPoolBytes(physicalSize);
          bufferPoolHitCount++;
          // Overwrite the pooled buffer's contents with the new data (only
          // the first `size` bytes; the tail of physicalSize is don't-care
          // padding — no read path is allowed to touch beyond logical size).
          // JS-F005: shared view on Chromium, slice fallback elsewhere.
          const buffer = getHandle<GPUBuffer>(handle);
          queue.writeBuffer(buffer, 0, wasmBytesForWrite(wasmDataPtr, size));
          // Update the logical size for the new owner. Previous owner may
          // have had a smaller logical size within the same bucket.
          bufferLogicalSizesArr[handle] = size;
        } else {
          if (poolable) bufferPoolMissCount++;
          // Allocate physicalSize so poolable buffers can be bucket-reused
          // on release. The data write below copies only logical `size`
          // bytes — any tail padding stays as the initial zero contents.
          // JS-F005: shared view on Chromium, slice fallback elsewhere.
          const buffer = device.createBuffer({ size: physicalSize, usage });
          queue.writeBuffer(buffer, 0, wasmBytesForWrite(wasmDataPtr, size));
          handle = addHandle(buffer);
          bufferSizesArr[handle] = physicalSize;
          bufferLogicalSizesArr[handle] = size;
          bufferUsagesArr[handle] = usage;
        }
        setResult(handle);
      } catch (e) {
        console.error(
          `[GPU Worker] CREATE_BUFFER_FROM_DATA failed: size=${size} usage=0x${usage.toString(16)}`,
          e,
        );
        setResult(0);
      }
      break;
    }

    // ================================================================
    // Phase 0 stats (?profile=1 observability). Write the per-opcode
    // histogram into the dedicated stats SAB so the wasm-worker can post
    // it up to the main thread.
    //
    // JS-F010: storage moved out of the cmd SAB's CALLBACK_BASE /
    // UNIFORM_DATA / reserved regions and into a dedicated 1 KiB SAB
    // (statsU32, 256 u32 slots) plumbed through the init message. The cmd
    // SAB only carries the scalar total via cmdU32[I_RESULT] now. This
    // removes the "safe by serialization" tripwire where any loosening of
    // cmd-SAB single-slot would let GET_STATS silently clobber a fused
    // dispatch's UNIFORM_DATA bytes.
    //
    // ARG0 = reset flag: 1 = zero the histogram after readback.
    // ================================================================
    case RpcFn.GET_STATS: {
      const resetAfter = arg0();
      // Synthetic counters live in high stats slots so real RpcFn opcodes
      // 102/103 (BUFFER_RELEASE_BATCH/DISPATCH_BATCH) remain visible.
      gpuRpcCounts[STATS_EXTRA_POOL_HITS] = bufferPoolHitCount;
      gpuRpcCounts[STATS_EXTRA_POOL_MISSES] = bufferPoolMissCount;
      gpuRpcCounts[STATS_EXTRA_BG_CACHE_HITS] = bindGroupCacheHits;
      gpuRpcCounts[STATS_EXTRA_BG_CACHE_MISSES] = bindGroupCacheMisses;
      gpuRpcCounts[STATS_EXTRA_UNIFORM_HOT_HITS] = uniformHotHits;
      gpuRpcCounts[STATS_EXTRA_UNIFORM_HOT_MISSES] = uniformHotMisses;
      gpuRpcCounts[STATS_EXTRA_SPIN_HITS] = spinHitTotal;
      gpuRpcCounts[STATS_EXTRA_SPIN_MISSES] = spinMissTotal;
      gpuRpcCounts[STATS_EXTRA_SPIN_BUDGET] = spinDispatchBudget;
      if (statsU32) {
        // Flat copy into the dedicated stats SAB — no region math. Both
        // arrays are sized to STATS_OPCODE_SLOTS.
        statsU32.set(gpuRpcCounts);
      }
      setResult(gpuTotalRpcs);
      if (resetAfter) {
        gpuRpcCounts.fill(0);
        gpuTotalRpcs = 0;
        bufferPoolHitCount = 0;
        bufferPoolMissCount = 0;
        bindGroupCacheHits = 0;
        bindGroupCacheMisses = 0;
        uniformHotHits = 0;
        uniformHotMisses = 0;
        spinHitTotal = 0;
        spinMissTotal = 0;
        // Budget intentionally preserved — it's adaptive session state,
        // not a per-window counter. Resetting it would throw away the
        // equilibrium we just learned.
      }
      break;
    }

    default: {
      console.warn(`[GPU Worker] Unknown function ID: ${fnId}`);
      setResult(0);
    }
  }
}
