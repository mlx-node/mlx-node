# WebGPU Backend Architecture

The WebGPU backend enables MLX to run on any platform with a WebGPU implementation — including browsers (via WASM), desktop (via Dawn or wgpu-native), and embedded systems. This document covers the full architecture: the C++ kernel layer, the browser bridge, the WASM build system, and the pitfalls we encountered along the way.

## Table of Contents

- [Overview](#overview)
- [Two-Worker Architecture (Browser)](#two-worker-architecture-browser)
  - [Buffer Pool and Release Batching](#buffer-pool-and-release-batching)
  - [Per-Dispatch Pass Restart (Aliasing Workaround)](#per-dispatch-pass-restart-aliasing-workaround)
  - [Observability (`?profile=1`)](#observability--profile1)
- [C++ Backend Architecture](#c-backend-architecture)
- [Type System: CPU vs GPU Widths](#type-system-cpu-vs-gpu-widths)
- [Packed bf16 Storage](#packed-bf16-storage)
- [Fused SDPA Kernels](#fused-sdpa-kernels)
- [Uniform Buffer Pooling](#uniform-buffer-pooling)
- [Memory Management](#memory-management)
- [WASM Build System](#wasm-build-system)
- [Pitfalls and Bugs Found](#pitfalls-and-bugs-found)
- [Performance Characteristics](#performance-characteristics)
- [Operations Not Yet Implemented](#operations-not-yet-implemented)
- [Adding a New Kernel](#adding-a-new-kernel)
- [Debugging Tips](#debugging-tips)

---

## Overview

The backend is split across two repositories:

- **mlx** (`mlx/backend/webgpu/`) — C++ GPU kernels that generate WGSL shader source at runtime, compile and cache pipelines, and dispatch compute work via the standard `webgpu.h` C API. 36 source files, ~13,574 lines.
- **mlx-node-browser** (`packages/browser/`) — TypeScript bridge that implements `webgpu.h` functions as JavaScript calls into the browser's `GPUDevice`, connected to the WASM-compiled MLX via SharedArrayBuffer + Atomics RPC. ~11,237 lines across 13 source files (plus demo).

Three `webgpu.h` implementations are supported:

| Backend       | Description                              |
| ------------- | ---------------------------------------- |
| `DAWN`        | Google's reference WebGPU (desktop)      |
| `WGPU`        | wgpu-native pre-built library (desktop)  |
| `WASI_IMPORT` | WASM target — functions imported from JS |

For the browser target (`WASI_IMPORT`), all `wgpu*` function calls become unresolved WASM imports, satisfied at runtime by the TypeScript bridge in mlx-node-browser.

---

## Two-Worker Architecture (Browser)

The browser deployment uses a two-worker architecture to work around the constraint that WebGPU's `GPUDevice` can only be used on the thread that created it, while WASM needs synchronous blocking (`Atomics.wait`) that is forbidden on the main thread:

```
Main Thread (UI)          wasm-worker                   gpu-worker
    |                        |                              |
    | <- postMessage <-      | <- cmdBuffer SAB ----------> |
    |   (chunks + result)    |   (Atomics.wait RPC)         |
    |                        |   Atomics.waitAsync          | <- GPUDevice
    |                        |   drains stream SAB ring     |
    |                        |   (inside WASM heap)         |
```

The main thread only receives `postMessage` events (`type: 'chunk'`, `type: 'result'`, etc. — see `mlx-worker.ts:517-543` and `demo/app.ts:136-143`). It does not call `Atomics.waitAsync` itself. The decoded-token `Atomics.waitAsync` loop runs inside the wasm-worker (`chat-stream-sab.ts:165-177,183-230`, `mlx-worker.ts:513-555`) — the Rust `SabSink` producer and the JS reader both execute on the wasm-worker thread and share a SAB ring that lives in the WASM heap.

**wasm-worker** (`src/mlx-worker.ts`, 686 lines) — Runs the compiled MLX WASM module. When MLX C++ code calls a WebGPU function (e.g., `wgpuDeviceCreateComputePipeline`), the bridge stub encodes the call into a SharedArrayBuffer command region and wakes the gpu-worker via `Atomics.notify`. Then it blocks on `Atomics.wait` until the gpu-worker writes back the result.

**gpu-worker** (`src/gpu-worker.ts`, 1,881 lines) — Owns the `GPUDevice`. Sits in an `Atomics.waitAsync` loop. When woken, it reads the command from SharedArrayBuffer, executes the corresponding WebGPU API call, writes the result back, and notifies the wasm-worker.

### SharedArrayBuffer RPC Protocol

Defined in `src/rpc-protocol.ts` (229 lines). The command buffer is a fixed-size 512-byte SharedArrayBuffer:

```
Offset  Field              Size     Description
------  -----              ----     -----------
0       FN_ID              4 bytes   RPC function ID (54 function IDs defined, max id 102)
4       STATUS             4 bytes   Atomics wait/notify flag (Int32Array)
8       RESULT             4 bytes   Return value (low 32 bits)
12      RESULT_HI          4 bytes   High 32 bits for u64 returns
16      ARG0..ARG7         32 bytes  Up to 8 u32 arguments
48      ARG0_HI..ARG3_HI   16 bytes  High bits for u64 args
64      CALLBACK_COUNT     4 bytes   Pending callback count (or bind group entry count)
68      CALLBACK_BASE      120 bytes Callback payloads in 68..188; `CALLBACK_ENTRY_SIZE=16` bytes,
                                     so 7 entries fit cleanly. `MAX_CALLBACKS_PER_CALL=8` is the
                                     producer cap (`rpc-protocol.ts:212-214`, `gpu-worker.ts:850-861`);
                                     an 8th entry spans 180..196 and overlaps `UNIFORM_DATA_SIZE`
                                     plus the first 4 bytes of `UNIFORM_DATA`, so it can only coexist
                                     with RPCs that do not use the inline-uniform region (callback-
                                     returning opcodes like POLL / BUFFER_MAP_ASYNC, never with
                                     FUSED_DISPATCH_WITH_UNIFORM).
188     UNIFORM_DATA_SIZE  4 bytes   Inline uniform buffer size (FUSED_DISPATCH_WITH_UNIFORM)
192     UNIFORM_DATA       256 bytes Inline uniform buffer data (up to 256 bytes)
448     Stats tail         64 bytes  16 u32 slots reused by GET_STATS for opcode histogram slots 95..110
                                     (slots 100/101 also repurposed by the gpu-worker as pool-hit/pool-miss counters)
```

A dedicated 4 MiB `readbackBuffer` SharedArrayBuffer sits alongside `cmdBuffer` (`rpc-protocol.ts` `READBACK_BUFFER_SIZE`; `mlx-worker.ts:67-69`, `gpu-worker.ts:1451-1454`, `webgpu-bridge-stub.ts:1250-1257`). GPU→CPU mapped reads are written into this fixed buffer by the gpu-worker; using a separate SAB avoids the growable-WASM-memory hazard where the wasm-worker could grow `memory.buffer` while the gpu-worker holds a stale view.

Each RPC call:

1. wasm-worker writes `FN_ID` and `ARG0..ARGn`
2. wasm-worker sets `STATUS = PENDING` and calls `Atomics.notify`
3. wasm-worker calls `Atomics.wait(STATUS, PENDING)` (blocks)
4. gpu-worker wakes, reads command, executes, writes `RESULT`
5. gpu-worker sets `STATUS = DONE` and calls `Atomics.notify`
6. wasm-worker wakes and reads the result

**Fused dispatch optimizations**: To reduce RPC round-trips, the bridge batches multiple WebGPU calls into single RPC commands:

| RPC ID | Name                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------ | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 91     | `FUSED_DISPATCH`              | setPipeline + setBindGroup + dispatch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 92     | `FUSED_DISPATCH_2BG`          | setPipeline + 2x setBindGroup + dispatch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 93     | `FUSED_SUBMIT`                | endPass + finish + submit + release                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 94     | `FUSED_BG_DISPATCH`           | **Dead opcode.** Declared in `rpc-protocol.ts:92-95` but never emitted by the bridge hot path (`webgpu-bridge-stub.ts:1036-1165` chooses `FUSED_DISPATCH_2BG` / `FUSED_DISPATCH_WITH_UNIFORM` / `FUSED_FULL_DISPATCH` / `FUSED_DISPATCH` instead) and has no `case RpcFn.FUSED_BG_DISPATCH` in the gpu-worker switch (`gpu-worker.ts:1651-1704`). It was an early prototype of inline bind-group fusion, superseded by `FUSED_FULL_DISPATCH` (96), which packs entries into the callback ring (see "Inline bind-group packing in the callback region" below). Kept in the enum only to avoid renumbering the live opcodes. |
| 95     | `CREATE_BUFFER_FROM_DATA`     | createBuffer + writeBuffer (replaces mappedAtCreation)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 96     | `FUSED_FULL_DISPATCH`         | inline createBindGroup + setPipeline + setBindGroup + dispatch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 97     | `FUSED_DISPATCH_WITH_UNIFORM` | FUSED_FULL_DISPATCH + inline uniform buffer write                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 98     | `FUSED_COPY_BUFFER`           | endPass + copyBuffer + beginPass                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 99     | `GET_STATS`                   | Readback of **gpu-worker** RPC histogram + pool-hit / pool-miss counters (bridge-side stats come from local stub counters, not this RPC — see [Observability](#observability--profile1))                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 102    | `BUFFER_RELEASE_BATCH`        | Batched `wgpuBufferRelease` (up to `MAX_RELEASE_BATCH=64` handles packed into the UNIFORM_DATA region)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

`FUSED_DISPATCH_WITH_UNIFORM` (ID 97) is the hot-path workhorse. It does **not** create a GPU buffer (the gpu-worker case at `gpu-worker.ts:1704-1747` looks up an existing buffer and calls `queue.writeBuffer` against it, and always sets `RESULT = 0`). The buffer for the inline uniform must already exist — either via the immediate `CREATE_BUFFER_FROM_DATA` for `> 256`-byte unmap (`webgpu-bridge-stub.ts:1275-1286`) or via `materializeDeferredBuffer()` for the small-uniform path (`webgpu-bridge-stub.ts:300-314`). With those preconditions met, the RPC then collapses what would otherwise be 4 separate calls (`writeBuffer`, `setPipeline`, inline `createBindGroup`, `dispatch`) into a single round-trip. See [Fake `mappedAtCreation` and Deferred Buffer Creation](#fake-mappedatcreation-and-deferred-buffer-creation) for the broader create-and-write flow.

**Inline bind-group packing in the callback region.** `FUSED_FULL_DISPATCH` (ID 96) and `FUSED_DISPATCH_WITH_UNIFORM` (ID 97) reuse the callback-ring region (bytes 68..188) as a **second wire layout**: instead of 16-byte callback records, the bridge packs 12-byte `(bufferHandle:u32, sizeLo:u32, sizeHi:u32)` tuples starting at `CALLBACK_BASE` and writes the entry count into `CALLBACK_COUNT`. Because the dedicated callback area is only 120 bytes (7 full 16-byte callback entries as discussed below), the inline-bind-group fusion has its own cap: `entryCount <= 10` (10 × 12 = 120 bytes). Bind groups with more than 10 entries fall back to a standalone `DEVICE_CREATE_BIND_GROUP` RPC followed by a separate dispatch, losing the fusion win. See `webgpu-bridge-stub.ts:762-788` and `webgpu-bridge-stub.ts:1096-1105` for the gate and packer.

`BUFFER_RELEASE_BATCH` (ID 102) was introduced after Phase 0 profiling revealed that `BUFFER_RELEASE` alone was ~30% of decode-time RPCs. The bridge (`webgpu-bridge-stub.ts:220-223,428-440`) accumulates handles in a `pendingReleases` list, flushes when the batch fills or at the top of `wgpuComputePassEncoderDispatchWorkgroups`, and the gpu-worker (`gpu-worker.ts:1532-1543`) routes each handle through the buffer pool (see [Buffer Pool](#buffer-pool-and-release-batching)).

### Handle Management (gpu-worker)

GPU objects (buffers, pipelines, bind groups, etc.) are tracked by integer handles in a sparse array on the gpu-worker side. When the wasm-worker calls `wgpuDeviceCreateBuffer`, the gpu-worker creates the real `GPUBuffer`, stores it in the handle table, and returns the integer handle. All subsequent references use this handle.

### Buffer Pool and Release Batching

Transient GPU buffers (K-projection workspace, attention intermediates, reduction scratch) dominate decode-time allocations. Two mechanisms amortize them:

**Buffer pool** (`gpu-worker.ts:50-109`). `bufferPool: Map<"usage:size", handle[]>` keyed on `(usage, size)`. On `wgpuDeviceCreateBuffer`, the gpu-worker first probes the pool for a matching key; on hit it reuses the existing handle and bumps the local `bufferPoolHitCount` counter without calling `device.createBuffer` (`gpu-worker.ts:998-1007`). On miss it creates a fresh buffer and bumps `bufferPoolMissCount`. Those two local counters are then surfaced into SAB stats slots 100 / 101 only during a `GET_STATS` readback (`gpu-worker.ts:1846-1864`), so the slots reflect "hits since the last `GET_STATS`" rather than being incremented per buffer create. `isPoolable(usage)` excludes any buffer whose usage contains `MAP_READ | MAP_WRITE` so readback buffers never land in the pool. `BUFFER_POOL_CAP_PER_KEY = 32` bounds the reuse queue per key — handles beyond the cap are actually released.

**Release batching** (`webgpu-bridge-stub.ts:428-447,449-465,1024-1035`, `gpu-worker.ts:1532-1543`). Instead of issuing one RPC per `wgpuBufferRelease`, the bridge pushes handles onto `pendingReleases` and flushes them through `BUFFER_RELEASE_BATCH` (RPC 102). Flush triggers are:

- The batch hits `MAX_RELEASE_BATCH = 64` handles (`queueRelease` at `webgpu-bridge-stub.ts:442-447`).
- Any non-release RPC is about to fire and the batch is non-empty (guard at the top of `rpcCall`, `webgpu-bridge-stub.ts:449-465`).
- `wgpuComputePassEncoderDispatchWorkgroups` runs with `pendingReleases.length >= 32` (`webgpu-bridge-stub.ts:1024-1035`). The 32-handle threshold is a deliberate tuning choice: flushing on every dispatch dropped the average batch size to ~3 and destroyed the amortization; the threshold bounds release-visible-to-pool latency while still letting batches grow.
- `FUSED_DISPATCH_WITH_UNIFORM` is excluded from the `rpcCall` entry flush because its caller has already written uniform bytes into `CMD_OFFSET.UNIFORM_DATA` (offsets 192..447); a flush would trample them. The dispatch hot path's explicit threshold-gated flush covers this case instead.

The gpu-worker's batch handler (`gpu-worker.ts` case `RpcFn.BUFFER_RELEASE_BATCH`) loops over the handles packed into the `UNIFORM_DATA` region starting at offset 192 and routes each through `releaseBufferHandle`, which applies the pool-return logic for poolable usages and falls through to `releaseHandle` otherwise.

The bridge itself lives inside the emnapi async-thread-pool workers, not only in the main wasm-worker. To aggregate buffer-pool and diagnostic counters across those parallel stubs, `mlx-worker.ts:69` allocates a dedicated `poolStatsBuffer` SharedArrayBuffer once per session and threads it through **two** distinct hand-offs:

1. **Main wasm-worker stub.** `mlx-worker.ts:99-112` passes `poolStatsBuffer` directly as the sixth positional argument to `createBridgeStub()`, alongside `cmdBuffer`, `sharedMemory`, the GPU handles, `readbackBuffer`, and `features`. There is no `rpcConfig` object on this path — the call site is plain positional.
2. **Child-worker stubs.** When emnapi spins up an async-thread-pool worker, `mlx-worker.ts:182-193` `postMessage`s an `{ type: '__mlx_rpc_config', cmdBuffer, readbackBuffer, poolStatsBuffer, handles, features }` envelope to the new worker (`features` lands at line 193, the closing brace of the envelope). `webgpu-worker.mjs:29-35` waits for that envelope, then calls `createBridgeStub(rpcConfig.cmdBuffer, wasmMemory, rpcConfig.handles, rpcConfig.readbackBuffer, rpcConfig.features, rpcConfig.poolStatsBuffer)` — same positional layout as the main worker, just sourced from the postMessage envelope. The `poolStatsBuffer` parameter is guarded as optional in the stub, so a child worker that omits it would simply skip the cross-worker aggregation.

Every stub then atomically `Atomics.add`s into shared counter slots (`webgpu-bridge-stub.ts:225-237`). `getBridgeStats()` `Atomics.load`s the aggregated values for the `?profile=1` snapshot (`webgpu-bridge-stub.ts:1516-1525`).

### Per-Dispatch Pass Restart (Aliasing Workaround)

WebGPU forbids a buffer from appearing as both a read-only binding and a read-write binding within the same compute pass. MLX does not track bind-group buffer aliasing, so `CommandEncoder::dispatch_compute` in `mlx/backend/webgpu/device.cpp:709-734,737-753` conservatively calls `end_compute_pass()` after **every** dispatch in both the single-bind-group and vector overloads. The bridge-side `endAndRestartPass()` in `gpu-worker.ts:153-168` is **not** about preventing C++ from keeping one pass open across an `eval()` — C++ already ends it.

Its job is to keep the bridge stub's cached `passHandle` valid while replacing the underlying `GPUComputePassEncoder`. The bridge stub holds onto the pass handle returned by `BEGIN_COMPUTE_PASS` and reuses it across subsequent fused-dispatch RPCs. If the gpu-worker just called `pass.end()` and then let the next begin allocate a brand-new handle slot, the bridge stub's cached pointer would go stale. Instead, `endAndRestartPass()` looks up the `passHandle → encoderHandle` mapping, calls `pass.end()`, calls `encoder.beginComputePass()`, and **overwrites the existing `passHandle` slot in-place**. There is no `wgpuComputePassEncoderRelease` call on this path — the old pass object is just dropped when the slot is replaced. The cost is still real (one extra JS function call plus WebGPU work per dispatch), but it is not an RPC round-trip. The [plan](../../.claude/plans/synthetic-plotting-dahl.md) has a dedicated Phase 1 to replace the per-dispatch `end_compute_pass()` in C++ with per-buffer read/write aliasing tracking so the pass can stay open across non-aliasing dispatches, which would also eliminate the need for this JS-side restart.

### Fake `mappedAtCreation` and Deferred Buffer Creation

The bridge dodges the standard WebGPU "create + map + write + unmap" 3-RPC dance by minting **fake** buffer handles that never reach the gpu-worker until the bridge has actually seen the data the caller intends to write. This is what makes `CREATE_BUFFER_FROM_DATA` (RPC 95) the workhorse for small-uniform creation.

**Fake handle minting** (`webgpu-bridge-stub.ts:683-722`). `wgpuDeviceCreateBuffer` reads the descriptor and triggers the fast path **only** when both `mappedAtCreation == true` _and_ `usage & WGPUBufferUsage_CopyDst (0x0008)` — see lines 696-704 and the corresponding `WGPUBufferUsage` flags at `mlx/backend/webgpu/utils.h:153-158`. (`COPY_SRC`-only mapped buffers fall through to a normal `DEVICE_CREATE_BUFFER` RPC because `writeBuffer` requires `COPY_DST`.) When the pattern matches the bridge:

1. `wasmMalloc`s a shadow region inside WASM memory the same size as the requested buffer.
2. Increments `fakeBufferCounter` (starts at `0x7d000000`, distinct from the `0x7e000000` fake bind-group range and `0x7f000000` general fake handle range) to mint a fresh handle.
3. Records `MappedAtCreationInfo { wasmPtr, size, usage }` in the `mappedAtCreationBuffers` map (`webgpu-bridge-stub.ts:715`).
4. Seeds **only** `bufSizes[fakeHandle] = size` (line 716). It does **not** populate `bufferUsages` — fake mapped handles deliberately have no `bufferUsages` entry, see the explicit "no `bufferUsages` entry" comment at `webgpu-bridge-stub.ts:1369-1372`. The consequence is that fake handles cannot enter the `(usage, size)`-keyed buffer pool until they are materialised into a real buffer; releasing a fake handle hits the unknown / unpoolable counter path instead.

**Mapped range write.** A subsequent `wgpuBufferGetMappedRange` against the fake handle returns a pointer into the WASM shadow region — also zero RPCs. The C++ caller writes its uniform bytes into that shadow.

**Unmap.** `wgpuBufferUnmap` on a fake handle (`webgpu-bridge-stub.ts:1261-1289`) splits on size:

- `mapped.size <= 256`: **no RPC yet.** The entry is moved into `deferredCreations: Map<fakeHandle, DeferredCreation { usage, size, wasmPtr }>` (lines 1267-1273). The WASM shadow ptr is _not_ freed because the bytes are still needed.
- `mapped.size > 256`: **immediate RPC.** The bridge issues `CREATE_BUFFER_FROM_DATA` right here (lines 1275-1286), stores `fakeHandle → realHandle` in `deferredBufferRemap`, populates `bufSizes[realHandle]` / `bufferUsages[realHandle]` (the real handle finally gets a `bufferUsages` entry so it can pool), and frees the WASM shadow ptr.

**Materialisation of `deferredCreations`.** Once a buffer is in `deferredCreations` (the small-uniform branch), there are two paths to materialise it:

- **Forced materialisation via `materializeDeferredBuffer()`** (`webgpu-bridge-stub.ts:300-314`, called from `webgpu-bridge-stub.ts:344-352, 1086-1094`). This issues an explicit `CREATE_BUFFER_FROM_DATA` RPC, stores the result in `deferredBufferRemap`, and frees the WASM shadow. It is invoked when (a) `flushPendingWrites()` needs to write to a deferred buffer, or (b) the dispatch packer has already chosen a _different_ entry as the inline-uniform slot but a deferred fake still appears in the bind group.
- **Inline data path through `FUSED_DISPATCH_WITH_UNIFORM`** (`webgpu-bridge-stub.ts:1107-1135`). When the bridge picks a deferred entry as `uniformEntryIdx`, it copies the shadow bytes into `CMD_OFFSET.UNIFORM_DATA` (≤ 256 bytes), writes `0` into the entry's handle slot in the callback ring as a sentinel, and fires `FUSED_DISPATCH_WITH_UNIFORM`. **The branch then expects the gpu-worker to return the freshly-created buffer handle in `RESULT`** so the bridge can populate `deferredBufferRemap`. This expectation is **not met** by the current gpu-worker — see "Stub/worker mismatch" below — so the deferred-via-RPC-97 branch is effectively dead code; in practice every small `mappedAtCreation+COPY_DST` buffer hits `materializeDeferredBuffer()` first.

**Fake → real remap.** Every bridge function that takes a buffer handle calls `resolveBufferHandle(handle)` (`webgpu-bridge-stub.ts:295-297`), which probes `deferredBufferRemap` and returns the real handle if one exists. This is why `wgpuBufferRelease` and `wgpuBufferDestroy` first scrub the fake-handle entries, then call `resolveBufferHandle` to find the real handle (if any) — see `webgpu-bridge-stub.ts:1349-1433` and `webgpu-bridge-stub.ts:1310-1347` for the cleanup ordering, which has to handle the "released before first dispatch" case (no real buffer ever existed) without leaking the WASM shadow allocation.

**Stub/worker mismatch on `FUSED_DISPATCH_WITH_UNIFORM`.** The gpu-worker case for RPC 97 (`gpu-worker.ts:1704-1747`) does **not** create a buffer. It looks up an existing buffer handle from the entry slot at `cmdU32[I_CB_BASE + uniformEntryIdx * 3]`, calls `queue.writeBuffer(uniformBuffer, 0, uniformData)` against it, builds the bind group from the supplied entries, and dispatches. It always sets `cmdU32[I_RESULT] = 0`. The "creates a buffer" semantics that the bridge's deferred-finalize block (`webgpu-bridge-stub.ts:1124-1135`) is coded against do not exist — `result > 0` is never true on this path, the deferred remap never gets populated through here, and the sentinel `0` handle written into the entry slot would crash `getHandle<GPUBuffer>(0)` if the path ever fired. The path stays alive only because every reachable code flow currently routes deferred small-uniform buffers through `materializeDeferredBuffer()` (and thus a real `CREATE_BUFFER_FROM_DATA` RPC) before the dispatch packer ever picks a deferred entry as `uniformEntryIdx`. The branch should either be deleted or the gpu-worker should be taught to honour the deferred-create-and-return contract.

The aggregate effect of this whole mechanism is that the standard 3-RPC create/map/unmap pattern collapses to either **0 RPCs** (if the buffer is released before unmap) or **1 RPC** (`CREATE_BUFFER_FROM_DATA` at unmap time), with `FUSED_DISPATCH_WITH_UNIFORM` writing into the now-real buffer. It is the single biggest reason the bridge stays under ~3-4 RPCs per kernel dispatch.

### Suppressed Release / Destroy Semantics

Several WebGPU C-API "release/destroy" entry points are intentionally turned into **no-ops** (or local-only cleanups) on the bridge — they never reach the gpu-worker. This is safe because the underlying GPU resources are either lightweight, cached for reuse, or owned by another lifetime that the bridge already manages.

| C-API call                      | Bridge behavior                                                                                                                                                                                                                                 | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `wgpuBindGroupRelease`          | Drops `pendingBgData` for the handle (in case a fused dispatch never consumed the eagerly-parsed descriptor); **no RPC**. (`webgpu-bridge-stub.ts:1450-1455`)                                                                                   | Bind groups are lightweight JS wrappers on the gpu-worker side. Releasing them used to fire ~28K RPCs/generation; suppressing them costs nothing because the gpu-worker reuses the bind-group slot via the handle table's GC-driven path.                                                                                                                                                                                                                                |
| `wgpuBindGroupLayoutRelease`    | Pure no-op. (`webgpu-bridge-stub.ts:1457-1459`)                                                                                                                                                                                                 | Bind-group-layout handles are cached in the stub's `layoutCache` (declared at `webgpu-bridge-stub.ts:333-334`, consumed by `wgpuComputePipelineGetBindGroupLayout` at `:1436-1441`) and reused across dispatches. Releasing them prematurely would invalidate the cache.                                                                                                                                                                                                 |
| `wgpuShaderModuleRelease`       | Pure no-op. (`webgpu-bridge-stub.ts:1461-1463`)                                                                                                                                                                                                 | Shader modules are cached by the pipeline cache keyed on the WGSL hash. Discarding them would force re-compilation on the next pipeline create, which is hundreds of milliseconds.                                                                                                                                                                                                                                                                                       |
| `wgpuBufferDestroy`             | Local cleanup of `mappedAtCreationBuffers` / `deferredCreations` / `deferredBufferRemap` / `bufSizes` / `bufferUsages` / `pendingWriteBuffers`; **no RPC, no GPU-side destroy**. (`webgpu-bridge-stub.ts:1310-1347`, `gpu-worker.ts:1512-1529`) | The gpu-worker has historically hit `"buffer destroyed during submit"` errors when a buffer was destroyed before the queue finished consuming it. Suppressing destroy lets the gpu-worker hold onto the GPU buffer until it is actually released (which routes through the buffer pool, see [Buffer Pool](#buffer-pool-and-release-batching)). The bridge's own shadow tables still need scrubbing to avoid leaks of the WASM-side metadata for fake / deferred handles. |
| `wgpuComputePassEncoderEnd`     | Drops the local `pending*` state but does **not** call `pass.end()`; the pass stays alive in the gpu-worker for the next `BEGIN_COMPUTE_PASS` to reuse. (`webgpu-bridge-stub.ts:1176-1181`)                                                     | Pass-end is cheap, but the cached `passHandle` keeps `endAndRestartPass()` valid (see [Per-Dispatch Pass Restart](#per-dispatch-pass-restart-aliasing-workaround)).                                                                                                                                                                                                                                                                                                      |
| `wgpuComputePassEncoderRelease` | Pure no-op. (`webgpu-bridge-stub.ts:1183-1186`)                                                                                                                                                                                                 | The pass object is dropped when the gpu-worker reuses its handle slot or when the encoder is finished.                                                                                                                                                                                                                                                                                                                                                                   |

The aggregate effect is that the bridge issues exactly **one** "destructive" RPC family (`BUFFER_RELEASE_BATCH`, see [Buffer Pool](#buffer-pool-and-release-batching)), and even that one routes through the buffer pool rather than actually destroying the underlying `GPUBuffer`.

### Observability (`?profile=1`)

The demo (`demo/app.ts:185-246,404-417`) honors `?profile=1` to surface per-generation dispatch/RPC counters. There is no dedicated panel — each profile snapshot is rendered as **five separate `log()` lines** appended to the existing log panel at generation end (`demo/app.ts:223-246`, headers: `[profile] Decode …`, `[profile] opcodes: …`, `[profile] pool: …`, `[profile] gpu-pool: …`, `[profile] diag: …`). The flow is **one snapshot per generation**, not per decode step:

1. The demo parses `?profile=1` during init and forwards `profile: true` into the wasm-worker init message (`demo/app.ts:426-433`, `mlx-worker.ts:54-63`).
2. The wasm-worker calls `resetProfileCounters()` immediately before each chat/chatStream/baseline generation (`mlx-worker.ts:511-518,579-584,619-635`), which atomically zeroes the bridge counters, the C++ dispatch stats, and the gpu-worker histogram.
3. Generation runs normally; every bridge-side `rpcCall` increments its per-opcode slot, every gpu-worker dispatch case increments its own histogram, and every C++ dispatch is tallied inside `device.cpp`.
4. When generation finishes the wasm-worker calls `postProfileSnapshot(numTokens)` (`mlx-worker.ts:388-443`), which merges **three** separate reads into one `{ type: 'profile', stats: {...} }` message:
   - `getBridgeStats()` — bridge-side per-opcode histogram + `poolStatsBuffer` atomics (hits/misses/create/release diagnostics).
   - `fetchGpuWorkerStats(false)` — a `RpcFn.GET_STATS` round-trip that reads the gpu-worker's per-opcode histogram and its internal pool-hit/pool-miss counters (`webgpu-bridge-stub.ts:1556-1584`).
   - `wgpuGetDispatchStats()` — a C++ FFI export (`mlx_stream.cpp:234-265`, `device.h:193-199`) returning the total compute dispatches and compute-pass-end counts maintained by `CommandEncoder` in the WASM module itself.
5. The demo receives the `profile` message and appends the five log lines above to the log panel.

The striped SAB histogram used by `GET_STATS` totals `STATS_OPCODE_SLOTS = 111` u32 slots (444 bytes), split across three regions because no single contiguous free region in the 512-byte command record can hold them all:

- **Callback-ring region** — `STATS_CALLBACK_SLOTS = 31` u32 slots starting at `CALLBACK_COUNT` (offset 64, **not** `CALLBACK_BASE` 68), covering slots 0..30.
- **Inline-uniform region** — `STATS_INLINE_SLOTS = 64` u32 slots at `STATS_INLINE_OFFSET = 192` (offset 192..447), covering slots 31..94.
- **Stats tail** — `STATS_RESERVED_SLOTS = 16` u32 slots at `STATS_RESERVED_OFFSET = 448` (offset 448..511), covering slots 95..110. The gpu-worker also repurposes **slots 100 and 101** in this region as pool-hit / pool-miss counters on every `GET_STATS` readback.

`?profile=1` is the primary reason for the existence of `GET_STATS`, the `poolStatsBuffer` SAB aggregation, and the C++ dispatch counters. It is the measurement harness for the decode-throughput plan — every optimization phase ships its before/after numbers through this path.

### Weight Upload Flow

The current production init path is **per-tensor CPU upload via `addCpuTensor`** (`mlx-worker.ts:291-335`). The gpu-worker's `upload_weights` handler (`gpu-worker.ts:238-513`) still exists as a prebuilt GPU-buffer path but is **not** called from the current init — it is kept for alternate loaders that want to hand MLX pre-resident GPU buffers.

Active flow (mlx-worker → addCpuTensor → MLX C++):

1. mlx-worker downloads `model.safetensors` from `modelUrl` and parses the header via `parseSafeTensorsHeader`, yielding `{ tensors, dataOffset }`.
2. `Qwen35Model.setCpuModelConfig(configJson, tokenizerJson, tokenizerConfigJson)` stores config/tokenizer in Rust **before** the tensor loop runs — the ~5.7 MB tokenizer string must land while WASM memory is still small, otherwise emnapi's stale `DataView` bounds throw after a later `memory.grow`.
3. For each tensor: `malloc(t.byteSize)` inside the WASM heap, `Uint8Array.set` copies the source bytes into the WASM memory at that pointer, then `Qwen35Model.addCpuTensor(name, uptr, byteSize, shape, dtypeCode)` is invoked. Rust immediately builds an `mlx::array` from the pointer (C++ copies the data into a fresh CPU-resident allocation), after which JS `free`s the WASM buffer. Peak pointer stays well under 2 GiB because only one tensor is alive at a time.
4. After the loop, `await Qwen35Model.buildModelFromCpuTensors()` drains the Rust accumulator, constructs the compiled Qwen3.5 model on its dedicated thread, and returns the handle. GPU buffers are materialised lazily by MLX itself on first compute — `upload_with_conversion` in `device.cpp` (lines 448-507) handles **bf16 → f32 (or packed-bf16) and bool → u32** widening at that point. `float16` is passed through unchanged.

The bf16/bool conversions therefore live in `device.cpp` (`upload_with_conversion`) for the current path, not in the JS `upload_weights` handler. The gpu-worker's unused `upload_weights` branch has its own copy of the conversion logic (bf16 expand, `pack_bf16` packing, plus an IEEE 754 f16→f32 widening that does not match any kernel in the current build) and could be re-enabled by swapping init to `postMessage({ type: 'upload_weights', ... })` instead of `addCpuTensor`.

### Token Streaming

The SAB stream ring is an **intra-wasm-worker** transport between the Rust `SabSink` producer and a JS reader loop that both run on the wasm-worker thread. Decoded tokens cross the Rust → JS boundary through that ring with zero copies; the wasm-worker → main-thread hop is still plain `postMessage` in both SAB and TSFN modes. The SAB path simply replaces the per-token NAPI `ThreadsafeFunction` call that the earlier design used on the Rust → JS hop — it is not a main-thread-visible SAB.

This is the default path (`?stream_sab=0` or `?mode=tsfn` selects the legacy TSFN fallback, `?mode=baseline` forces the non-streaming `chat()` path). The SAB ring lives inside the WASM heap itself so the Rust `SabSink` producer and the JS reader share the exact same bytes (see `chat-stream-sab.ts`).

Ring layout (matches `crates/mlx-core/src/chat_stream/wire.rs`):

- 32-byte header: `seq` (i32, producer bumps on every write and notifies), `write_cur`, `read_cur`, `cancelled`.
- Record stream: each record is `[len u16 LE, kind u8, flags u8, payload]` where `kind` is `KIND_TEXT` (0), `KIND_JSON` (1), or `KIND_ERROR` (2). `KIND_TEXT` payloads are raw UTF-8 tokens on the hot decode path; `flags & FLAG_IS_REASONING` marks thinking tokens. `KIND_JSON` carries the structured final chunk (full text, thinking, tool calls, performance metrics).

Dispatch (`mlx-worker.ts` `handleChat` at lines 445-500):

1. `mode` defaults to `'sab'` (unless caller passes `mode='tsfn' | 'baseline'` or `useSab: false`).
2. The worker `malloc`s a 256 KiB region **inside the WASM heap**, wraps it as `Buffer.from(memory.buffer, ringOffset, SAB_RING_SIZE)` (a zero-copy view), and builds a reader via `createSabRingOverHeap(memory, ringOffset, SAB_RING_SIZE)`.
3. It calls `model.chatStreamSab(messages, config, sabBuf)` — Rust writes tokens directly into the ring via `SabSink`.
4. The JS reader loop (inside the wasm-worker) uses `Atomics.waitAsync(seqView, SEQ_IDX, lastSeq)` to wake on each write, then drains as many records as are ready, and **forwards each record to the main thread** via `post({ type: 'chunk', text, isReasoning })` for text records (`mlx-worker.ts:538-543`) and `post({ type: 'result', ... })` for the terminal JSON chunk (`mlx-worker.ts:519-534`).
5. On completion or error, the reader calls `abortController.abort()` which flips `cancelled = 1` so the producer stops writing, and the worker `free`s the heap region.

The TSFN fallback still lives in `handleChatTsfn` (`mlx-worker.ts:612-655`) and calls `model.chatStream(messages, config, callback)`; from the main thread's perspective both modes look identical — they both receive `{type: 'chunk'}` and `{type: 'result'}` `postMessage` events. The TSFN fallback is kept for environments where `Atomics` is unavailable and for A/B measurements, but it is no longer the default.

### Browser Source Files

| File                        | Lines | Purpose                                                                                                                       |
| --------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| `src/test-worker.ts`        | 4,784 | Browser test suite (186 test cases)                                                                                           |
| `src/gpu-worker.ts`         | 1,881 | GPU thread: WebGPU API calls, handle table, fused dispatch, buffer pool, BUFFER_RELEASE_BATCH, GET_STATS                      |
| `src/webgpu-bridge-stub.ts` | 1,593 | C-API stubs: encodes wgpu\* calls into RPC, pending-release batching, GET_STATS readback                                      |
| `src/webgpu-bridge.ts`      | 682   | Bridge setup and initialization                                                                                               |
| `src/mlx-worker.ts`         | 686   | WASM thread: model loading, inference orchestration, `?profile=1` stats forwarding                                            |
| `demo/app.ts`               | 512   | Demo app: Qwen3.5 chat UI + `?profile=1` log-panel output                                                                     |
| `src/chat-stream-sab.ts`    | 410   | SAB ring reader for token streaming (default path)                                                                            |
| `src/rpc-protocol.ts`       | 229   | SharedArrayBuffer layout, RPC function IDs, GET_STATS slot layout                                                             |
| `src/cxx-stubs.ts`          | 131   | C++ standard library stubs for WASM                                                                                           |
| `src/safetensors.ts`        | 123   | SafeTensors file parsing                                                                                                      |
| `src/wasm-loader.ts`        | 109   | WASM module loading and instantiation                                                                                         |
| `src/webgpu-worker.mjs`     | 89    | Child-worker RPC bridge bootstrap (emnapi thread pool)                                                                        |
| `src/index.ts`              | 8     | Package entry point                                                                                                           |
| `demo/index.html`           | —     | Demo HTML shell + `<script type="module" src="./app.ts">` loader (COOP/COEP are set by `vite.config.ts`, not the HTML itself) |

### URL Query Parameters

The demo app (`demo/index.html`) supports runtime feature toggles via URL query params, parsed in `demo/app.ts:410-417`:

| Param                       | Default | Effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `?pack_bf16=0`              | on      | Forwarded into the worker init message and flips the runtime `wgpuSetPackedBf16Enabled()` flag, but on the **current** demo init path it is a no-op for live inference: `addCpuTensor` → `array_from_cpu_data` → `mlx_array_from_cpu_data` (`mlx-worker.ts:291-335`, `persistence.rs:1425-1444`, `safetensors.rs:685-700`, `mlx_array_ops.cpp:51-83`) routes every weight through `wgpu_buffer()`, which leaves `StorageMode::Upconverted` per the `device.cpp:521-528` no-auto-opt-in note. The flag only takes effect on the dormant `gpu-worker.ts` `upload_weights` path or on the test harness via `MxArray::from_bfloat16_bytes` — see [Packed bf16 Storage](#packed-bf16-storage) for the live opt-in entry points. |
| `?sdpa_fallback=1`          | off     | Force SDPA onto the decomposed matmul+softmax+matmul fallback                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `?stream_sab=0`             | on      | Disable the SAB token-stream ring and fall back to the TSFN `chatStream` path                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `?mode=baseline\|sab\|tsfn` | `sab`   | Override the streaming dispatch: `baseline` runs non-streaming `chat()`, `sab` is the default ring-buffer path, `tsfn` forces the legacy ThreadsafeFunction path                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

`packBf16` and `sdpaFallback` are forwarded into the worker `init` message and then into the WASM backend via `wgpuSetPackedBf16Enabled()` / `wgpuSetSdpaFallbackForced()` before any model load runs. `useSab` / `mode` flow through the per-`chat` message and are consumed by `handleChat` in `mlx-worker.ts`.

---

## C++ Backend Architecture

All kernel code lives in `mlx/backend/webgpu/`. Each `.cpp` file typically implements one or more MLX primitives by:

1. Defining a C++ params struct (uniform buffer layout)
2. Generating WGSL shader source as a string
3. Caching the compiled pipeline via `device().get_or_create_shader_module()`
4. Setting up bind groups and dispatching compute work

### Source Files

| File                | Lines | Operations                                                               |
| ------------------- | ----- | ------------------------------------------------------------------------ |
| `sdpa.cpp`          | 1,627 | Fused SDPA: vector (single-pass), 2-pass split-L vector, tile (prefill)  |
| `reduce.cpp`        | 1,135 | Sum, Prod, Max, Min, And, Or (3 strategies)                              |
| `indexing.cpp`      | 1,114 | Gather, GatherAxis, Scatter, ScatterAxis, SliceUpdate                    |
| `matmul.cpp`        | 1,065 | GEMV (split-K), multi-col GEMV, GEMM (tiled 16x16), packed bf16 variants |
| `device.cpp`        | 803   | Device init, feature detection, pipeline caching, packed bf16 upload     |
| `compiled.cpp`      | 616   | Fused element-wise kernel JIT (Compiled primitive)                       |
| `normalization.cpp` | 590   | RMSNorm (2-phase) and LayerNorm (3-phase), packed bf16 weight variants   |
| `allocator.cpp`     | 525   | Buffer allocation, page cache, readback, storage mode management         |
| `rope.cpp`          | 469   | RoPE: single-token and general variants                                  |
| `scan.cpp`          | 461   | Prefix sum/prod/max/min/logaddexp                                        |
| `sort.cpp`          | 390   | Bitonic sort and argsort in shared memory                                |
| `binary.cpp`        | 377   | 18 ops: Add, Multiply, Power, Maximum, comparisons, etc.                 |
| `conv.cpp`          | 339   | Depthwise and grouped conv1d                                             |
| `unary.cpp`         | 327   | 29 ops: Abs, Exp, Log, Sin, Cos, Sqrt, Sigmoid, Erf, etc.                |
| `copy.cpp`          | 301   | Copy with 6 dtype conversion modes                                       |
| `ternary.cpp`       | 283   | Select (where/conditional)                                               |
| `random.cpp`        | 276   | RandomBits via Threefry-2x32-20 PRNG                                     |
| `quantized.cpp`     | 274   | QuantizedMatmul (int2/int4/int8 on-the-fly dequant)                      |
| `softmax.cpp`       | 222   | Online softmax (3-phase: max, sum-exp, normalize)                        |
| `arg_reduce.cpp`    | 218   | ArgMax, ArgMin                                                           |
| `logsumexp.cpp`     | 202   | LogSumExp (2-phase: max, log-sum-exp)                                    |
| `arange.cpp`        | 158   | Integer and float range generation                                       |
| `worker.cpp`        | 138   | Background worker pump + completion handlers                             |
| `slicing.cpp`       | 136   | Concatenate, dynamic slice offset                                        |
| `primitives.cpp`    | 115   | `NO_GPU` / `NO_GPU_MULTI` stubs and cross-reference comments             |
| `device_info.cpp`   | 98    | Device info queries (adapter properties, limits)                         |
| `fence.cpp`         | 83    | Fence primitive for cross-stream synchronisation                         |
| `event.cpp`         | 79    | Event wait/signal wiring                                                 |
| `eval.cpp`          | 62    | Backend `eval` entry + scheduler glue                                    |
| `no_webgpu.cpp`     | 15    | Stub shim for builds with WebGPU disabled                                |
| `utils.h`           | 382   | Type helpers, bind group creation, WGSL codegen, packed bf16 helpers     |
| `op_exprs.h`        | 310   | Shared unary/binary expression builders for compiled.cpp                 |
| `device.h`          | 201   | WebGPUDevice class, UniformBufferPool, pipeline cache                    |
| `allocator.h`       | 112   | WebGPUBuffer struct, StorageMode enum, allocator interface               |
| `worker.h`          | 50    | Background worker for async GPU work                                     |
| `webgpu_backend.h`  | 21    | Public backend entry header                                              |

**Total: 36 files, ~13,574 lines** (30 `.cpp` + 6 headers).

### WGSL Code Generation Pattern

Every kernel follows the same pattern. Here is a simplified example for a unary operation:

```cpp
std::string make_unary_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& op_expr) {
  std::ostringstream s;

  if (in_type == "f16") s << "enable f16;\n";

  s << "const WORKGROUP_SIZE: u32 = 256u;\n"
    << "const N_READS: u32 = 4u;\n\n"
    << "@group(0) @binding(0) var<storage, read> input: array<" << in_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<" << in_type << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: Params;\n\n"
    << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let base = gid.x * N_READS;\n"
    << "  for (var i = base; i < min(base + N_READS, params.size); i++) {\n"
    << "    let in_val = input[i + params.in_offset];\n"
    << "    output[i + params.out_offset] = " << op_expr << ";\n"
    << "  }\n}\n";

  return s.str();
}
```

Kernels are cached by name:

```cpp
std::string entry = "abs_f32_v";
WGPUShaderModule shader = dev.get_or_create_shader_module(
    entry, [&]() { return make_unary_kernel(entry, "f32", "abs(in_val)"); });
auto pe = dev.get_or_create_pipeline(entry, shader, entry.c_str());
```

Each kernel variant gets a unique name encoding the dtype, operation, and variant (`v` for contiguous, `g` for general/strided, `ss`/`sv`/`vs`/`vv` for binary scalar/vector combinations, `_bf16p` for packed bf16).

### Uniform Buffer Layout

All uniform parameters must be vec4-aligned (16 bytes). A typical params struct:

```cpp
struct UnaryParams {
  uint32_t size_ndim[4];   // [size, ndim, pad, pad]
  uint32_t offsets[4];     // [in_offset, out_offset, pad, pad]
  uint32_t shape_0[4];     // shape[0..3]
  uint32_t shape_1[4];     // shape[4..7]
  int32_t strides_0[4];    // in_strides[0..3]
  int32_t strides_1[4];    // in_strides[4..7]
};
```

Shape and strides are split across two vec4s to support up to `MAX_NDIM = 8` dimensions. The WGSL side accesses them via helper functions like `get_shape(i)` and `get_stride(i)` that index into the appropriate vec4.

### Matmul Params (96 bytes)

The matmul uniform buffer supports multi-dimensional batch strides for GQA broadcasting:

```cpp
struct MatmulParams {
  uint32_t M, N, K;
  uint32_t lda, ldb, ldc;
  uint32_t batch_size;
  uint32_t batch_ndim;
  uint32_t batch_shape[4];      // vec4<u32> in WGSL
  uint32_t batch_stride_a[4];   // vec4<u32> in WGSL
  uint32_t batch_stride_b[4];   // vec4<u32> in WGSL
  uint32_t batch_stride_c;      // output is always contiguous
  uint32_t offset_a;
  uint32_t offset_b;
  uint32_t _pad;                // pad to 96 bytes
};
```

The WGSL `elem_to_loc_broadcast()` helper decomposes a flat batch index into per-operand offsets, with a fast path for `batch_ndim==1` (the common case).

### Reduction Strategy

Reductions use one of two algorithms depending on device capabilities:

**Tree reduction** (fallback) — The `emit_unrolled_reduction` helper generates an unrolled shared-memory tree:

```wgsl
// Generated WGSL for workgroup_size=256:
if (tid < 128u) { shared[tid] = op(shared[tid], shared[tid + 128u]); }
workgroupBarrier();
if (tid < 64u) { shared[tid] = op(shared[tid], shared[tid + 64u]); }
workgroupBarrier();
// ... down to stride 1
```

**Subgroup reduction** (when `device().has_subgroups()`) — The `emit_subgroup_reduction` helper emits a two-phase pattern:

```wgsl
enable subgroups;
var sg_val = subgroupAdd(acc);           // Phase 1: hardware reduction
if (subgroupElect()) {
  shared[subgroup_id] = sg_val;          // One value per subgroup
}
workgroupBarrier();
// Phase 2: tree-reduce across ~8 subgroup results
if (tid < 4u) { shared[tid] = op(shared[tid], shared[tid + 4u]); }
workgroupBarrier();
// ... down to stride 1
```

The `prefix` parameter avoids WGSL variable name collisions when a kernel needs multiple reductions (e.g., softmax uses "mx" for max and "sm" for sum).

---

## Type System: CPU vs GPU Widths

A central challenge of the WebGPU backend is that some MLX dtypes have different sizes on CPU and GPU:

| Dtype    | CPU size | GPU storage | GPU size | Shader type               |
| -------- | -------- | ----------- | -------- | ------------------------- |
| float32  | 4 bytes  | `f32`       | 4 bytes  | `f32`                     |
| float16  | 2 bytes  | `f16` (raw) | 2 bytes  | `f16` or `f32` (see note) |
| bfloat16 | 2 bytes  | `f32`       | 4 bytes  | `f32`                     |
| bool     | 1 byte   | `u32`       | 4 bytes  | `u32`                     |
| int32    | 4 bytes  | `i32`       | 4 bytes  | `i32`                     |
| uint32   | 4 bytes  | `u32`       | 4 bytes  | `u32`                     |

This mismatch requires careful handling throughout:

**Allocation**: `wgpu_alloc_size(arr)` computes GPU buffer size using `wgpu_itemsize()` (`utils.h:32-38`), which returns 4 for `bool` and `bfloat16` and **passes every other dtype (including `float16`) through at its CPU itemsize**. Packed bf16 bypasses this helper entirely and uses `wgpu_packed_alloc_size` instead.

**Upload conversion**: When data is uploaded to the GPU (`device.cpp` `upload_with_conversion`, lines 448-507), the only widening conversions are **bfloat16 → f32** (`bf16_bits << 16`) and \*\*bool → u32`. When `StorageMode::PackedBf16`is active, bf16 values are instead packed as u32 pairs (see [Packed bf16 Storage](#packed-bf16-storage)).`float16` data is copied straight through with no conversion — the 2-byte IEEE 754 half layout is preserved byte-for-byte in the GPU buffer.

**Shader-side `f16` fallback**: WGSL's native `f16` type requires the `shader-f16` feature (available in most Chromium builds but not everywhere). When the feature is missing, `dtype_to_wgsl_safe()` (`utils.h:273-277`) returns `"f32"` instead of `"f16"` and kernels like `make_unary_kernel` (`unary.cpp:46-74,197-198`) splice that value directly into the storage declarations they generate — e.g. `@group(0) @binding(0) var<storage, read> input: array<f32>`. In other words the fallback **changes the WGSL storage type, not just local-register math**. This is a latent aliasing hazard: the current init path DOES accept `F16` tensors from SafeTensors and forwards them to Rust as `DType::Float16` (`packages/browser/src/safetensors.ts:17,64-73`, `packages/browser/src/mlx-worker.ts:305-335`, `crates/mlx-core/src/models/qwen3_5/persistence.rs:1178-1185`), and `wgpu_itemsize()` keeps those tensors at 2 bytes on the GPU. Any kernel that splices `dtype_to_wgsl_safe()` directly into its storage declarations — e.g. the generated unary (`unary.cpp:70,197-198`) and binary (`binary.cpp:77,249-250`) kernels — will, on a device without `shader-f16`, read the 2-byte f16 buffer through an `array<f32>` binding and produce wrong results. The `copy` primitive is unaffected because `copy.cpp` loads a fixed WGSL module that declares both storage buffers as `array<u32>` (`kernels/copy.wgsl:41-42`, `copy.cpp:40-45,77-97`) and reconstructs dtype-width loads inside the shader. In practice Qwen3.5-0.8B ships as bf16 so no native f16 weights reach the affected kernels today, but a model with real f16 tensors would expose the gap and needs to be flagged if it comes online.

A second, dormant f16→f32 widening path lives in the JS gpu-worker `upload_weights` handler (`gpu-worker.ts:489-547`) from an earlier architecture. It is no longer reached by the production init flow (see [Weight Upload Flow](#weight-upload-flow)) and exists only for alternate loaders that need to hand MLX pre-resident GPU buffers.

**Download conversion**: When data is read back (`allocator.cpp` `raw_ptr()`), float32 values are truncated back to bfloat16 and uint32 values are truncated to bool.

**Offset calculation**: Array offsets are always in bytes on the CPU side. For the **upconverted** path, each CPU element maps 1:1 to a GPU element, so:

```cpp
uint32_t gpu_elem_idx = arr.offset() / arr.itemsize();  // bytes -> elements
```

works regardless of the size difference (bf16 → f32 widens, but every element still has its own slot).

**Caveat for `StorageMode::PackedBf16`**: the 1:1 mapping no longer holds. Two bf16 elements share one u32 slot, so a kernel that addresses the packed buffer as `array<u32>` must convert offsets and strides into u32 units, not element units. `matmul.cpp:791-807` handles this explicitly: when the B operand is packed it divides `batch_stride_b` by 2 and computes `b.offset() / 4` (rather than `/ arr.itemsize()`) to land at the correct u32 word. Anywhere else that touches a `PackedBf16` buffer must do the same conversion — there is no helper that hides it.

**`dtype_to_wgsl_safe()`** (`utils.h:273-277`): Returns `"f32"` instead of `"f16"` when the device lacks `shader-f16` support. This is _not_ a graceful fallback — `wgpu_itemsize()` (`utils.h:32-37`) still allocates f16 buffers at 2 bytes/element, so any generated kernel that splices the returned string into a storage binding (e.g. `unary.cpp:70-74,197-198`, `binary.cpp:77-83,249-250`) will then read the 2-byte f16 bytes through an `array<f32>` view and produce wrong results. See the hazard discussion above. The `copy` primitive is the only fast-path carve-out because it uses fixed `array<u32>` storage bindings at `kernels/copy.wgsl:41-42` and `copy.cpp:40-45` and reconstructs dtype-width loads inside the shader.

---

## Packed bf16 Storage

The default behavior expands bf16 (2B) to f32 (4B) on GPU upload, doubling memory bandwidth for memory-bound GEMVs. The packed bf16 path stores bf16 values as raw u32 pairs (2 elements per u32), halving GPU read bandwidth for weight-dominated operations.

### Storage Mode

```cpp
// allocator.h
enum class StorageMode : uint8_t {
  Upconverted,  // bf16 -> f32 on upload (default)
  PackedBf16,   // bf16 stored raw, 2 elements per u32
};
```

`StorageMode` is a field on `WebGPUBuffer`. The C++ runtime flag `packed_bf16_enabled()` defaults to **off** in the backend itself (`mlx/backend/webgpu/allocator.cpp:25` initialises the underlying static to `false`); the demo only appears to default it on because `packages/browser/demo/app.ts:411` parses `?pack_bf16` as enabled-unless-`=0` and `packages/browser/src/mlx-worker.ts:61,232-233` forwards that value into `wgpuSetPackedBf16Enabled(...)` before any model load. Even when the flag _is_ on it only **gates** the opt-in helpers — it does not by itself promote any buffer. Opt-in happens **only** through explicit weight-upload entry points that set `StorageMode::PackedBf16` at the buffer's birth — `device.cpp:521-528` has an explicit `NOTE: there is deliberately no auto-opt-in to StorageMode::PackedBf16` so `wgpu_buffer()` (the general-purpose allocator for intermediate tensors and for any weight uploaded via the active CPU-tensor path) never auto-promotes based on shape or size. As a result, on the **current demo init path** every Qwen3.5-0.8B weight ends up in `StorageMode::Upconverted` regardless of the `?pack_bf16` flag — see [Upload Paths](#upload-paths) for the full chain and the dormant JS path that does honor the flag.

The two supported birth points are:

- **CPU-bytes path (test harness only).** `MxArray::from_bfloat16_bytes()` (`crates/mlx-core/src/array/creation.rs:108-139`) calls `sys::mlx_array_from_bfloat16` to build the array, then calls `sys::mlx_wgpu_try_opt_in_packed_bf16(arr, threshold)`. That FFI is implemented at `crates/mlx-sys/src/mlx_array_ops.cpp:120-123`, which forwards to `mlx::core::wgpu::try_opt_in_packed_bf16(arr, min_elements)` in the C++ backend. The forwarded helper checks `packed_bf16_enabled()` and the element-count threshold (`PACKED_BF16_DEFAULT_MIN_ELEMENTS = 4096`, overridable for the small-norm test cases) before flipping the buffer's `storage_mode` to `PackedBf16`. This path is exercised by the unit-test suite but is **not** wired into the demo loader.
- **GPU handle import path.** `array_from_gpu_buffer()` (`crates/mlx-core/src/utils/safetensors.rs:706-733`) wraps an externally-supplied `GPUBuffer` handle into an `MxArray` _without_ changing its storage mode. To mark it packed the caller must additionally call `sys::mlx_wgpu_mark_buffer_packed_bf16()`, implemented at `crates/mlx-sys/src/mlx_array_ops.cpp:126-133` (which forwards to `mlx::core::wgpu::mark_buffer_packed_bf16(arr)`). The dormant `gpu-worker.ts` `upload_weights` handler is the only consumer that does this — it pre-packs SafeTensors weights into u32-pair buffers, hands the handle to Rust through `persistence.rs:1230-1244`, and then calls the mark helper. There is **no** `mlx_array_from_gpu_buffer(..., packed_bf16=true)` overload — the mark step is a separate FFI call.

The element-count thresholds `PACKED_MIN_ELEMENTS = 4096` and `NORM_PACKED_MIN_ELEMENTS = 256` live inside the dormant JS `upload_weights` handler at `gpu-worker.ts:329,337` and are applied at `gpu-worker.ts:472-476`:

- **Matmul-consumed weights** (linear projections routed through `matmul(x, W.T)`): `PACKED_MIN_ELEMENTS = 4096`. Keeps GEMV/GEMM B-operand loads at half bandwidth.
- **Norm-consumed weights** (RMSNorm / LayerNorm weight vectors): `NORM_PACKED_MIN_ELEMENTS = 256`. Routes tiny 1-D norm weights through the same `_bf16p` kernel variant for uniform code paths, even though the DRAM savings are trivial (norms are L2-resident on M3).

Each allowlist is name-based (see `isMatmulConsumedWeight` / `isNormConsumedWeight`): vision-tower tensors, `linear_attn.norm.weight`, embedding, GDN scalars, and conv1d kernels all stay upconverted regardless of size. Because `upload_weights` is dormant in the current production init path, **these thresholds are not hit during normal inference**; they only run when a caller explicitly switches init to the JS weight-upload flow.

### Upload Path

In `device.cpp` `upload_with_conversion()`, when `storage_mode == PackedBf16`:

- Allocates `ceil(n_elems / 2) * 4` bytes on GPU
- Packs raw bf16 pairs into u32 (`lo | (hi << 16)`)
- Odd element counts pad the trailing u32 slot with zero

### WGSL Unpacking

All kernels that read packed bf16 use a shared helper from `utils.h`:

```wgsl
fn unpack_bf16_pair(p: u32) -> vec2<f32> {
  let lo_bits = (p & 0x0000FFFFu) << 16u;
  let hi_bits = p & 0xFFFF0000u;
  return vec2<f32>(bitcast<f32>(lo_bits), bitcast<f32>(hi_bits));
}
```

### Supported Kernels

Packed bf16 variants exist for:

- **GEMV** (split-K and multi-col): B operand bound as `array<u32>`, inner K-loop reads two weights per load. Only `b_transposed=true` (Qwen3.5 hot path) — the `b_transposed=false` variant is not yet implemented (assertions enforce this in `matmul.cpp:154,328`).
- **GEMM** (tiled 16x16): B operand packed, same unpack helper.
- **RMSNorm / LayerNorm**: Weight (and bias for LayerNorm) read from packed storage.

Cache key suffix: `_bf16p` (e.g., `matmul_f32_bf16p_NT_mc4_aligned`).

### Upload Paths

1. **C++ path** (`upload_with_conversion` in `device.cpp`, active branch only for buffers that were already born with `StorageMode::PackedBf16`). Packs bf16 CPU data into u32 pairs lazily when MLX first materialises the GPU buffer. In the current production init this branch is **not reached** — the CPU-tensor `addCpuTensor` loader (`mlx-worker.ts:317-335` → `persistence.rs:1425-1444` → `safetensors.rs:685-700` → `mlx_array_ops.cpp:51-83`) routes every weight through `wgpu_buffer()`, which leaves `storage_mode == Upconverted` per the `device.cpp:521-528` note. So the current default for Qwen3.5-0.8B decode on this init path is **upconverted bf16 → f32**, not packed, even with `?pack_bf16=1`.
2. **JS path** (`gpu-worker.ts` `upload_weights` handler + `persistence.rs` GPU-buffer init flow) — **dormant**. Packs bf16 SafeTensors weights directly into GPU buffers and calls `mlx_wgpu_mark_buffer_packed_bf16()` to flip the storage mode flag. This is the only way the packed path is actually exercised today, and it is not wired into the demo init; both the handler and its persistence.rs consumer still compile and are kept for alternate loaders that want to hand MLX pre-resident GPU buffers. The CPU-tensor path superseded it to avoid the emnapi WASM-pointer overflow, but the CPU-tensor path trades back the packed-bf16 memory savings in the process.

### Performance Notes

On M3 with Qwen3.5-0.8B, packed bf16 gives ~3-4% decode speedup (not the initially projected +10%) because individual MLP weights fit in L2 cache. The win is concentrated in the large `linear_attn.in_proj_qkvz` weight (16 MiB/layer) that exceeds L2.

---

## Fused SDPA Kernels

The `ScaledDotProductAttention` primitive (`sdpa.cpp`, 1,627 lines) implements three kernel variants, replacing the decomposed matmul+softmax+matmul fallback:

### Kernel Variants

| Variant                   | Dispatch                                  | When Used                                                 |
| ------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| **Vector (single-pass)**  | B\*H workgroups, 128 threads each         | Tq=1 decode when `nblocks <= 1` **or** `B*H*nblocks < 64` |
| **Vector 2-pass split-L** | Pass 1: B*H*nblocks WGs; Pass 2: B\*H WGs | Tq=1 decode when `nblocks > 1` and `B*H*nblocks >= 64`    |
| **Tile**                  | ceil(Tq/BQ)*H*B workgroups                | Tq > 1 prefill                                            |

**Supported head dims**: D in {64, 96, 128, 256}. At D=256, the vector kernel uses `D_PER_THREAD=2` (each of 128 threads owns 2 output lanes).

### Vector Kernel (Tq=1 Decode)

`make_sdpa_vector_kernel()`: One workgroup per (batch, head). 128 threads cooperatively process the full K/V sequence. Uses shared memory for Q vector and workgroup reduction. Subgroup reduction used when available.

### 2-Pass Vector Split-L (Long-Context Decode)

Mirrors Metal's `sdpa_vector_2pass_1` / `sdpa_vector_2pass_2`:

- **Pass 1** (`make_sdpa_vector_2pass_1_kernel`): Splits L into blocks of `SDPA_BLOCK_L=128`. Each workgroup handles one (batch, head, L-block), producing partial `(O[D], max, sum)`.
- **Pass 2** (`make_sdpa_vector_2pass_2_kernel`): One workgroup per (batch, head) merges `nblocks` partial results via online softmax reweighting.

Gating (`sdpa.cpp:1354-1358`): the 2-pass split is taken only when both `nblocks > 1` **and** `B*H*nblocks >= 64u`. When splitting L would not buy enough pass-1 workgroups to saturate the GPU (e.g. Qwen3.5-0.8B decode has `B*H = 1*8 = 8` via `q.shape(1)` at `sdpa.cpp:1272-1275`, so an 8-block split dispatches just 64 WGs), the single-pass kernel is used instead to avoid the dispatch + intermediate-buffer overhead.

### Tile Kernel (Prefill)

`make_sdpa_tile_kernel()`: FlashAttention-2-style online softmax tiled attention. Hand-coded because WGSL lacks `simdgroup_matrix`.

- Tile shape: BQ=16 rows of Q, BK=8 columns of K/V per inner tile
- Workgroup layout: (BK, BQ, 1) = (8, 16, 1), 128 threads total
- Dispatch: `(ceil(Tq/BQ), H_q, B)`
- Supports: causal masking (via `q_offset`), additive float mask, no mask
- D=256 fits within Chromium's 32 KiB shared memory budget

### Fallback Conditions

`use_fallback()` (sdpa.cpp:1117-1222) returns `true` (decomposed matmul→softmax→matmul path) when:

- `?sdpa_fallback=1` runtime flag is set.
- Training or logsumexp output is requested.
- Q/K/V ndim sanity check fails (any of Q/K/V has `ndim() < 2`; sdpa.cpp:1136-1139).
- D is not in {64, 96, 128, 256}, or K/V last-dim mismatches Q.
- K and V differ in sequence length.
- Any of Q/K/V dtype is not float32 or bfloat16.
- Scalar bool mask (non-causal, non-array) — the kernel cannot plumb a scalar mask.
- **GPU occupancy gate** (sdpa.cpp:1197-1205): `Tq == 1` **and** `B*H < 32`. For small-H decode models like Qwen3.5-0.8B (`B*H = 1*8 = 8` from `q.shape(0)*q.shape(1)`) the fused Tq=1 path dispatches too few workgroups — even the 2-pass split only reaches ~32 WGs at L=512 — so the decomposed fallback's hundreds of GEMV workgroups win by occupancy (+93% on Qwen3.5-0.8B, per the comment at `sdpa.cpp:1189-1196`).

All remaining cases use the fused kernels. Once past `use_fallback`, `eval_gpu` routes Tq>1 to the tile kernel and Tq==1 to either the single-pass or 2-pass vector kernel as described above. Note: attention sinks (`has_sinks_`) are not checked in `use_fallback()` (the flag is not available at the static call site) but are rejected with a throw in `eval_gpu()` at sdpa.cpp:1247-1255.

### Performance

For Qwen3.5-0.8B decode on Chrome (M3), the **current production path is the decomposed matmul→softmax→matmul fallback**, not the 2-pass fused vector kernel. The occupancy gate at `sdpa.cpp:1197-1205` routes `Tq == 1 && B*H < 32` to the decomposed path, and Qwen3.5-0.8B has `B*H = 1*8 = 8`, which trips the gate on every decode step.

Historical per-commit measurements (not all on the current path):

| Configuration                      | Decode tok/s | Status                                                                                           |
| ---------------------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| Decomposed fallback (3 dispatches) | 26.5         | **current production path for Qwen3.5-0.8B decode**                                              |
| 2-pass fused vector (2 dispatches) | 30-31        | historical, predates the occupancy gate; still the active path for decode models with `B*H ≥ 32` |

The single-row number in the overview at the top of this doc refers to the current production (decomposed) path; the fused-vector numbers persist here as a reference for larger-H models and for tracking Phase 1+ optimizations that may change the gate. The fused tile kernel for prefill (`Tq > 1`) is always active and significantly reduces TTFT by eliminating the softmax HBM roundtrip.

---

## Uniform Buffer Pooling

Creating and destroying uniform buffers per dispatch is expensive. The `UniformBufferPool` class in `device.h` manages a free list of reusable buffers, organized by 256-byte-aligned size:

1. `pool.acquire(queue, data, size)` — finds a free buffer of the right size (or creates one), writes data via `wgpuQueueWriteBuffer`
2. After GPU work completes, the buffer is returned via `encoder.add_completed_handler([buf]() { pool.release(buf); })`

---

## Memory Management

The `WebGPUAllocator` uses a `BufferCache` with 16 KB page granularity. Each allocation is represented by a `WebGPUBuffer` struct:

```cpp
struct WebGPUBuffer {
  WGPUBuffer buffer;         // GPU-side storage
  size_t size;               // Allocated bytes
  void* cpu_ptr;             // Non-null after readback
  size_t cpu_bytes;          // CPU allocation size (for full-buffer uploads)
  bool cpu_dirty;            // CPU data not yet uploaded
  bool gpu_has_data;         // GPU buffer has meaningful data
  Dtype::Val dtype_val;      // For conversion on readback
  StorageMode storage_mode;  // Upconverted or PackedBf16
};
```

The `cpu_ptr` field is lazily allocated on first `raw_ptr()` call (GPU readback). It is **invalidated** after GPU writes to prevent stale data from being returned — this was a critical bug fix (see [Pitfalls](#stale-cpu_ptr-after-gpu-compute)).

The `cpu_bytes` field tracks the full owning buffer size so that `upload_with_conversion()` can upload the whole buffer even when triggered by a slice view.

---

## WASM Build System

The WASM build involves multiple stages:

1. **Cargo + NAPI-RS** compiles Rust code targeting `wasm32-wasip1-threads`
2. **cmake** (invoked by `build.rs`) cross-compiles MLX C++ with WASI-SDK
3. **wasm-ld** links everything into a single `.wasm` binary (~20 MB)
4. The binary is copied to `packages/core/mlx-core.wasm32-wasi.opt.wasm`

**Build-flag ownership**: the WASM build configuration is split across two files, each owning the flags it actually controls:

- **`packages/core/build.ts`** is the WASM build entry point invoked by the package-local `build:wasm` script (`packages/core/package.json:31`: `oxnode ./build.ts --target wasm32-wasip1-threads --profile wasi`). The root `yarn build` / `yarn build:native` path runs `@mlx-node/core`'s plain `build` script (`packages/core/package.json:30`), which is the **native** darwin-arm64 build and does not produce the WASM artifact; producing the browser `.wasm` is a separate `build:wasm` invocation. In its WASM mode, `build.ts` owns the **env vars that must be set before `cargo`/`napi` launches**: `RUSTFLAGS='--cfg tokio_unstable'` and `TARGET_CXXFLAGS='-fwasm-exceptions -fexceptions'` (lines 24-36). `TARGET_CXXFLAGS` matters because `esaxx-rs` (pulled in by the `tokenizers` crate) has its own `cc::Build` that reads it from the environment — `mlx-sys/build.rs` cannot reach that crate directly.
- **`crates/mlx-sys/build.rs`** owns everything inside the `mlx-sys` compile step: the `cmake` invocation that builds `libmlx.a` for WASI (lines 48-100, including `-DMLX_BUILD_WEBGPU=ON`, `-DWEBGPU_BACKEND=WASI_IMPORT`, C/C++ flags, linker flags) **and** the `cc::Build` for the FFI bridge files that links the MLX visibility defines (lines 146-168).

Key flags by source file:

```bash
# Env — packages/core/build.ts:24-36 (before cargo invocation)
RUSTFLAGS='--cfg tokio_unstable'                    # Multi-thread tokio runtime on WASM
TARGET_CXXFLAGS='-fwasm-exceptions -fexceptions'    # Reaches esaxx-rs cc::Build via env

# cmake for libmlx.a — crates/mlx-sys/build.rs:48-100
-DCMAKE_CXX_FLAGS='… -fwasm-exceptions -DMLX_WGPU_LOG_KERNELS=1'
-DCMAKE_C_FLAGS='--target=wasm32-wasip1-threads --sysroot=… -pthread -fPIC -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_SIGNAL'
-DCMAKE_EXE_LINKER_FLAGS='--target=wasm32-wasip1-threads --sysroot=… -pthread -Wl,--allow-undefined'
-DMLX_BUILD_WEBGPU=ON
-DWEBGPU_BACKEND=WASI_IMPORT
-DMLX_BUILD_METAL=OFF
-DMLX_BUILD_CUDA=OFF
-DMLX_BUILD_CPU=ON
-DMLX_NO_BLAS=ON

# cc::Build for FFI bridge — crates/mlx-sys/build.rs:146-168
.define("MLX_USE_WEBGPU", None)
.define("WEBGPU_BACKEND_WASI_IMPORT", None)
.flag("-fexceptions") / .flag("-fwasm-exceptions")
.flag("-fvisibility=hidden") / .flag("-fvisibility-inlines-hidden")

# Rustc link — crates/mlx-sys/build.rs:137
cargo:rustc-link-lib=static:+whole-archive=mlx    # Prevents vtable method GC

# Stack size — set inside the napi-build crate (not the local build files)
# napi-build/src/wasi.rs:21 emits `cargo:rustc-link-arg=-zstack-size=64000000`
# for any wasm32-wasip1-threads target. crates/mlx-core/build.rs just calls
# napi_build::setup(); the flag itself lives in the external crate.
-zstack-size=64000000                               # ~64 MB stack (set via napi-build)
```

**tokio_unstable**: Enables `tokio::runtime::Builder::new_multi_thread()` in napi-rs for WASM targets. Requires `asyncWorkPoolSize > 0` in emnapi instantiation so `wasi_thread_spawn` can create thread pool workers. Without this, napi-rs falls back to `new_current_thread()` which cannot run `spawn_blocking` or parallel async tasks.

**Critical**: The MLX source at `crates/mlx-sys/mlx` is a **symlink** to `/Users/brooklyn/workspace/github/mlx`, not a git submodule. All C++ changes are made in the mlx repo and automatically picked up by WASM builds. Never re-init this as a submodule.

**No Asyncify**: The architecture uses `Atomics.wait` for synchronous blocking instead of Binaryen's asyncify transform. This avoids a ~60 MB binary size increase and emnapi corruption issues.

**No wasm-opt**: Binaryen v129 doesn't support the WASM exception handling proposal used by the build. The raw cargo output is used directly as the production binary.

---

## Pitfalls and Bugs Found

This section documents every significant bug and pitfall encountered during development, organized by category.

### Type Width Mismatches

#### bf16 offset calculation in copy kernel

**Symptom**: bf16 arrays with non-zero offsets read from wrong buffer positions.

**Root cause**: The copy kernel computed GPU offsets by dividing byte offsets by 4 (u32 size). For bf16 (CPU itemsize=2, GPU itemsize=4), an element at byte offset 64 became GPU index 16 instead of the correct 32.

**Fix**: Convert byte offset to element offset first (`bytes / itemsize`), then use as GPU index directly.

#### Bool/bf16 allocation size

**Symptom**: GPU buffer too small for bool and bf16 arrays.

**Root cause**: Allocation used `arr.nbytes()` (CPU bytes) but GPU needs 4 bytes per element for bool (1 byte on CPU) and bf16 (2 bytes on CPU).

**Fix**: `wgpu_alloc_size()` uses `arr.size() * wgpu_itemsize(dtype)` for GPU-sized allocation. `ensure_wgpu_size()` reallocates if needed.

### Buffer Coherence

#### Stale cpu_ptr after GPU compute

**Symptom**: After GPU computation, reading array data returned the original (pre-compute) values.

**Root cause**: `raw_ptr()` returned the cached `cpu_ptr` without checking whether the GPU had written new data.

**Fix**: Invalidate `cpu_ptr` (set to `nullptr`) in `set_output_array()` and after upload in `wgpu_buffer()`.

#### WebGPU buffer aliasing violation

**Symptom**: Validation errors when input and output arrays share the same buffer.

**Root cause**: WebGPU requires that a buffer bound as `storage, read` and `storage, read_write` in the same dispatch must be different buffers.

**Fix**: `ensure_no_alias(out, in)` forces a fresh allocation for the output if they share a buffer.

### Kernel Correctness

#### QuantizedMatmul transposed weight indexing

**Symptom**: Garbage output from all quantized linear projections.

**Root cause**: The transposed branch used column-major indexing, but quantized weights are always stored row-major as `[N, K_packed]`.

**Fix**: Always use row-major indexing: `w[n * w_cols + packed_idx]`.

#### RMSNorm/LayerNorm offset bug

**Symptom**: Garbage normalization for arrays with non-zero offsets (views into larger buffers).

**Root cause**: Contiguity check allowed arrays with non-zero offsets. The kernel reads from buffer position 0.

**Fix**: Changed to `!x.flags().row_contiguous || x.offset() != 0`, forcing a contiguous copy.

#### Subgroup reduction WGSL name collisions

**Symptom**: Shader compilation failure when a kernel uses subgroup reductions twice.

**Root cause**: Same variable names generated for both reductions.

**Fix**: Added a `prefix` parameter to `emit_subgroup_reduction`.

#### Matmul missing offset support

**Symptom**: Incorrect results when matmul inputs are views into larger buffers.

**Fix**: Added `offset_a` and `offset_b` fields to `MatmulParams`, computed as `arr.offset() / arr.itemsize()`.

#### Matmul multi-dim batch stride

**Symptom**: Decomposed SDPA fallback produced garbage for GQA models with multi-dim batch shapes.

**Root cause**: `dispatch_matmul()` took `.back()` of batch strides, only working for single-dim batch shapes. GQA broadcasting creates multi-dim batch shapes that require full stride arrays.

**Fix**: Extended `MatmulParams` with `batch_ndim`, `batch_shape[4]`, `batch_stride_a[4]`, `batch_stride_b[4]` (vec4 in WGSL). Added `elem_to_loc_broadcast()` WGSL helper with `ndim==1` fast path.

#### Subgroup ops in non-uniform control flow

**Symptom**: Undefined behavior when `subgroupAdd` called inside `if (sg_id == 0u)`.

**Root cause**: WGSL subgroup operations require **workgroup-uniform** control flow, not subgroup-uniform.

**Fix**: Keep subgroup reductions at the top of the workgroup body, outside any lane-specific branches.

### Build System

#### WASM vtable method garbage collection

**Symptom**: "function signature mismatch" errors at runtime.

**Root cause**: `wasm-ld`'s GC discards functions whose only references are vtable DATA relocations. MLX's `eval_gpu` virtual methods are only called through vtable dispatch.

**Fix**: `-Wl,--whole-archive=mlx` prevents any archive member from being discarded.

#### LTO corrupts C++ vtables in WASM

**Symptom**: Random crashes and wrong function dispatch after enabling LTO.

**Fix**: `lto = false` in the WASM build profile. LTO remains enabled for native builds.

#### Vtable visibility mismatch

**Symptom**: `call_indirect` signature mismatch at runtime.

**Root cause**: Without matching visibility flags between libmlx.a and the FFI bridge, inline virtual overrides get different vtable thunks.

**Fix**: Both cmake and cc::Build use `-fvisibility=hidden -fvisibility-inlines-hidden`.

#### cmake cache invalidation for WGSL changes

**Symptom**: WGSL kernel changes not picked up after rebuild.

**Root cause**: Most WGSL is generated at runtime as C++ strings. However, cmake does embed a few `.wgsl` files from `kernels/` via `file(GLOB)` + `file(READ)` (CMakeLists.txt:108-123). Changes to runtime-generated WGSL (the vast majority) require touching build files to force recompilation.

**Fix**: Touch `CMakeLists.txt` to force cmake reconfigure. Touch `build.rs` to force Cargo rebuild. Never clear the cmake build cache entirely.

#### esaxx-rs C++ exception incompatibility

**Symptom**: `cannot use 'try' with exceptions disabled` during WASM build.

**Root cause**: The `tokenizers` crate pulls in `esaxx-rs` which uses C++ `try`/`catch`. `esaxx-rs` has its own `cc::Build` that reads `TARGET_CXXFLAGS` from the environment — `mlx-sys/build.rs` can only set flags for its own cmake/cc::Build, not for other crates.

**Fix**: Set `TARGET_CXXFLAGS='-fwasm-exceptions -fexceptions'` in `build.ts` (the single source of truth for WASM build flags). This propagates to all crates' `cc::Build` invocations via the environment.

#### Lazy static init for compiled ops

**Symptom**: GPU device creation triggered at WASM module load, before WebGPU bridge is ready.

**Root cause**: File-scope `static auto compiled_* = mlx::core::compile(...)` in C++ bridge files trigger GPU device creation at static init time.

**Fix**: Convert to lazy function-local statics: `static auto& get_compiled_fn() { static auto fn = mlx::core::compile(...); return fn; }`.

### Browser Environment

#### SharedArrayBuffer requires Cross-Origin-Isolation

`Atomics.wait` and `SharedArrayBuffer` require the page to be served with COOP/COEP headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without these, the two-worker architecture cannot function.

#### Atomics.waitAsync requires Int32Array

**Symptom**: `Atomics.waitAsync` silently fails.

**Root cause**: `Atomics.waitAsync` only works with `Int32Array`, not `Uint32Array`.

#### BigInt in hot paths is very slow

**Symptom**: Unexpectedly low throughput in the TypeScript bridge.

**Fix**: Use plain 32-bit numbers wherever possible. Split 64-bit values into two 32-bit halves.

#### WASM child worker deadlock on `wgpuBufferMapAsync`

**Symptom**: Child WASM workers (emnapi thread pool) hang indefinitely during `raw_ptr()` — the first GPU→CPU readback after a compute operation never completes.

**Root cause**: Three factors combine into an unrecoverable deadlock:

1. `Buffer::raw_ptr()` in `allocator.cpp` calls `wgpuBufferMapAsync()` with a C callback, then enters a polling loop: `while (!map_data.done) { poll_instance(dev.gpu_instance()); }`
2. On WASM (`__wasm__`), `poll_instance()` is a **no-op** — the JS bridge handles async operations, not a native polling mechanism
3. With the **direct bridge** (each worker gets its own `GPUDevice`), `wgpuBufferMapAsync` registers a JS `Promise.then()` callback. But the polling loop blocks the thread's event loop, so the Promise callback never fires → infinite loop

This only manifests in **child workers** (emnapi thread pool), not the main wasm-worker. The main wasm-worker uses the RPC bridge where `wgpuBufferMapAsync` translates to `Atomics.wait` → gpu-worker handles the async map → writes callback info back → `drainCallbacks()` fires the C callback synchronously before the RPC returns.

**Fix**: Child workers must use the **RPC bridge** (shared `cmdBuffer` + `Atomics.wait` → gpu-worker), not the direct bridge. The gpu-worker's event loop is free to process the `mapAsync` Promise, writes the result back via the callback ring in SharedArrayBuffer, and the wasm-worker's `drainCallbacks()` invokes the C callback synchronously. See `webgpu-worker.mjs` for the child worker RPC bridge setup.

---

## Performance Characteristics

As of the latest measurements (April 2026) on Chrome desktop (M3):

| Metric              | Value                      |
| ------------------- | -------------------------- |
| Decode throughput   | ~30 tok/s                  |
| Time to first token | ~196 ms                    |
| Test suite cases    | 186 (see `test-worker.ts`) |

Model: Qwen 3.5 0.8B (bf16, 24 layers, GDN linear attention)

All 186 cases are registered as normal tests — none are gated behind a `skip`/`xfail` wrapper. The `stream channel SharedArrayBuffer write` case throws if the native `testStreamChannel` export is missing, so it functions as a "WASM-build-has-stream-channel-export" sanity check rather than an expected failure. The trailing `SDPA fallback GQA correctness (Tq=8 D=128 Hq=8 Hkv=2)` case at `test-worker.ts:4747-4772` exercises the decomposed fallback against the fused tile kernel and is exported in the full list via `post({ type: 'ready', tests: tests.map((t) => t.name) })` at `test-worker.ts:4783`.

### Key Optimizations (in order of impact)

| Optimization                     | Speedup            | Description                                                               |
| -------------------------------- | ------------------ | ------------------------------------------------------------------------- |
| Fast-path dispatch handlers      | 10.8 -> 17.5 tok/s | Inlined gpu-worker command loop; adaptive spin before Atomics.waitAsync   |
| Fused SDPA vector kernel         | 17.5 -> 20 tok/s   | Q*K^T -> softmax -> *V in one dispatch for Tq=1 decode                    |
| Pipeline warmup                  | 20 -> 25.7 tok/s   | 2-token dummy inference at load forces kernel compile before real request |
| Split-K GEMV                     | 25.7 -> 28.1 tok/s | One workgroup per output element, 256 threads cooperatively reduce K      |
| Multi-col GEMV (COLS_PER_WG=4/8) | 28.1 -> 30.6 tok/s | Amortize A-vector loads across 4-8 output columns                         |
| N-aligned fast path              | 30.6 -> 31.3 tok/s | Drop has1..hasN-1 guards when N % cols_per_wg == 0                        |
| 2-pass split-L SDPA              | Matched decomposed | 2 dispatches with B*H*nblocks WGs; needed for long contexts               |

### What Didn't Work

- **mc16 for N>=32768** — 16 accumulators/thread exceeded register pressure
- **WG_SIZE 128->256** — K=896 already covered in 7 iterations
- **vec4 B reads (non-transposed)** — gated on `!b_transposed`, but Qwen3.5 linear layers all land on `b_transposed` via `check_transpose`, so the kernel never dispatched
- **Subgroup-only final reduce** — WGSL subgroup ops require workgroup-uniform control flow

---

## Operations Not Yet Implemented

These operations throw or fall back to CPU decomposition:

| Operation                       | Status          | Notes                                                                                    |
| ------------------------------- | --------------- | ---------------------------------------------------------------------------------------- |
| `ArgPartition`                  | NO_GPU (throws) | Rarely used                                                                              |
| `BitwiseBinary`                 | NO_GPU          | Simple to implement if needed                                                            |
| `BlockMaskedMM`                 | NO_GPU          | Sparse matmul variant                                                                    |
| `FFT`                           | NO_GPU          | Not needed for transformer inference                                                     |
| `GatherMM`                      | NO_GPU          | Used by some MoE implementations                                                         |
| `GatherQMM`                     | NO_GPU          | Quantized variant of GatherMM                                                            |
| `QQMatmul`                      | throws          | Quantized-quantized matmul; explicit throw at `mlx/backend/webgpu/quantized.cpp:268-271` |
| `Hadamard`                      | NO_GPU          | Rarely used                                                                              |
| `Inverse` / `Cholesky`          | NO_GPU          | Linear algebra decompositions                                                            |
| `SVD` / `Eigh` / `Eig`          | NO_GPU_MULTI    | Eigenvalue decompositions                                                                |
| `LUF` / `QRF`                   | NO_GPU_MULTI    | Matrix factorizations                                                                    |
| `SegmentedMM`                   | NO_GPU          | Segmented matmul                                                                         |
| `MaskedScatter`                 | NO_GPU          | Masked scatter variant                                                                   |
| `Conjugate`                     | NO_GPU          | Complex conjugate                                                                        |
| `DivMod`                        | NO_GPU_MULTI    | Integer division + modulo (multi-output)                                                 |
| `Imag`                          | NO_GPU          | Imaginary part of complex                                                                |
| `Load`                          | NO_GPU          | Load from file                                                                           |
| `Partition`                     | NO_GPU          | Partial sort / partition                                                                 |
| `Real`                          | NO_GPU          | Real part of complex                                                                     |
| `CustomKernel`                  | NO_GPU_MULTI    | User-defined custom kernels (fast:: namespace)                                           |
| `SDPA VJP`                      | Fallback        | Training backward pass (uses decomposed path)                                            |
| `LayerNorm VJP` / `RMSNorm VJP` | NO_GPU_MULTI    | Normalization backward passes                                                            |
| `ConvertFP8` / `Quantize`       | NO_GPU_MULTI    | Quantization ops                                                                         |
| `distributed::AllReduce`        | NO_GPU_MULTI    | Distributed primitive (multi-output stub)                                                |
| `distributed::AllGather`        | NO_GPU_MULTI    | Distributed primitive (multi-output stub)                                                |
| `distributed::Send`             | NO_GPU_MULTI    | Distributed primitive (multi-output stub)                                                |
| `distributed::Recv`             | NO_GPU_MULTI    | Distributed primitive (multi-output stub)                                                |
| `distributed::ReduceScatter`    | NO_GPU_MULTI    | Distributed primitive (multi-output stub)                                                |

**Fully implemented** (fused GPU kernels, not fallback): `Compiled` (fused element-wise JIT), `Matmul`/`AddMM`, `QuantizedMatmul`, `RMSNorm`, `LayerNorm`, `RoPE`, `Softmax`, `LogSumExp`, `Reduce`, `Scan`, `Sort`/`ArgSort`, `ArgReduce`, `RandomBits`, `Convolution`, `Gather`/`Scatter`, all unary (29 ops) and binary (18 ops), `Select`, `Arange`, `Copy`.

**`ScaledDotProductAttention` (forward)** — fused GPU kernels (vector, 2-pass split-L vector, tile) are implemented in `sdpa.cpp`, **but the primitive still routes through `use_fallback()` (`sdpa.cpp:1117-1221`) on every invocation** and falls back to the decomposed matmul→softmax→matmul path when any of the following holds:

- `?sdpa_fallback=1` is set (`sdpa.cpp:1129-1131`).
- `is_training || output_logsumexp` (`sdpa.cpp:1133-1135`).
- Q/K/V shape sanity: any of `q.ndim() < 2`, `k.ndim() < 2`, `v.ndim() < 2` (`sdpa.cpp:1136-1139`).
- Head dim restriction: `q.shape(-1)` ∉ {64, 96, 128, 256} (`sdpa.cpp:1146-1149`); plus `k.shape(-1) == q.shape(-1)` and `v.shape(-1) == q.shape(-1)` (`sdpa.cpp:1150-1152`).
- Sequence-length match: `k.shape(-2) == v.shape(-2)` (`sdpa.cpp:1154-1156`).
- Dtype restriction: `q.dtype()` ∉ {f32, bf16}, or K/V dtype mismatch (`sdpa.cpp:1159-1166`).
- Scalar bool mask without `do_causal` (`sdpa.cpp:1185-1187`).
- **Decode occupancy gate**: `Tq == 1 && B*H < 32` (`sdpa.cpp:1197-1205`). Qwen3.5-0.8B has B=1, H=8, so its decode (Tq=1) hits this gate and runs on the decomposed path — see [Fused SDPA Kernels](#fused-sdpa-kernels) for the full routing table.

The fused tile kernel is the live path for prefill (Tq>1).

---

## Adding a New Kernel

To add a new WebGPU kernel implementation:

1. Create `mlx/backend/webgpu/<op_name>.cpp`
2. Define a params struct (vec4-aligned)
3. Write a `make_<op>_kernel()` function that generates WGSL source
4. Implement `<Primitive>::eval_gpu()` following the standard pattern:
   - Ensure inputs are contiguous (or handle strides)
   - Allocate output with `wgpu_alloc_size()`
   - Get or create shader module and pipeline
   - Set up bind group and dispatch
   - Use uniform pool for params, release in completed handler
5. Remove the `NO_GPU` / `NO_GPU_USE_FALLBACK` entry from `primitives.cpp` and add a comment pointing to the new file
6. Add the file to `CMakeLists.txt`
7. If the op was `NO_GPU_USE_FALLBACK`, implement `use_fallback()` returning `false`

Standard dispatch pattern:

```cpp
void MyOp::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& in = inputs[0];

  // 1. Ensure contiguous (if needed)
  array in_contig = in;
  if (!in.flags().row_contiguous || in.offset() != 0) {
    in_contig = contiguous_copy_gpu(in, s);
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.add_temporary(in_contig);
  }

  // 2. Allocate output
  out.set_data(
      allocator::malloc(wgpu::wgpu_alloc_size(in_contig)),
      in_contig.data_size(),
      in_contig.strides(),
      in_contig.flags());

  // 3. Get/create pipeline
  auto& dev = wgpu::device();
  std::string entry = "my_op_" + std::string(wgpu::dtype_to_wgsl_safe(in.dtype()));
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry, [&]() { return make_my_op_kernel(entry, ...); });
  auto pe = dev.get_or_create_pipeline(entry, shader, entry.c_str());

  // 4. Set up params + bind group
  MyOpParams params{};
  params.data[0] = ...;
  auto& pool = dev.uniform_pool();
  WGPUBuffer ubuf = pool.acquire(dev.gpu_queue(), &params, sizeof(params));

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(in_contig);
  encoder.set_output_array(out);

  WGPUBindGroup bg = wgpu::create_bind_group(pe.layout, {
      {wgpu::wgpu_buffer(in_contig), wgpuBufferGetSize(wgpu::wgpu_buffer(in_contig))},
      {wgpu::wgpu_buffer(out), wgpuBufferGetSize(wgpu::wgpu_buffer(out))},
      {ubuf, sizeof(params)}});

  // 5. Dispatch
  uint32_t n_workgroups = (out.size() + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, n_workgroups);

  // 6. Cleanup
  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([ubuf]() {
    wgpu::device().uniform_pool().release(ubuf);
  });
}
```

---

## Debugging Tips

- **WGSL compilation errors**: The generated WGSL source is a runtime string. Add a `fprintf(stderr, "%s\n", wgsl.c_str())` before `get_or_create_shader_module` to see the exact shader source.

- **Buffer content inspection**: Call `synchronize()` on the stream, then `raw_ptr()` on the array to force a GPU readback. Print the first few elements to verify correctness.

- **Offset bugs**: Always check `arr.offset()` when debugging wrong results. If the kernel doesn't account for offsets, views into larger buffers will read from the wrong location. Remember: `offset()` returns **bytes**, not elements. Convert via `offset() / itemsize()`.

- **Type promotion**: Remember that bf16 is stored as f32 on GPU (unless packed). When comparing GPU output to CPU reference, account for the precision difference (bf16 has ~3 decimal digits of precision).

- **Browser console**: Check for WebGPU validation errors in the browser console. Common issues: buffer aliasing (same buffer in read and read_write slots), buffer too small, workgroup size mismatch.

- **WASM crashes**: Deep model call stacks can overflow the WASM stack (~64 MB). Symptoms: silent crash or `RuntimeError: unreachable`. The stack size comes from `-zstack-size=64000000` which is emitted by the **external napi-build crate** (`napi-build/src/wasi.rs:21`) when `napi_build::setup()` runs from `crates/mlx-core/build.rs`. The flag is not visible in any checked-in build file in this repo. Do not add redundant `-zstack-size` flags elsewhere.

- **WASM build flags**: env vars (`RUSTFLAGS`, `TARGET_CXXFLAGS`) are set by `packages/core/build.ts` before cargo runs; cmake flags and the FFI-bridge `cc::Build` live in `crates/mlx-sys/build.rs`; the WASM stack size is emitted by the external `napi-build` crate (reached via `napi_build::setup()` in `crates/mlx-core/build.rs`). Before adding a new flag, pick the file that already controls the stage you need — do not duplicate into `.cargo/config.toml` or shell scripts.

- **WASM rebuild not picking up changes**: Touch `CMakeLists.txt` to force cmake reconfigure. Touch `build.rs` to force Cargo to re-run the build script. Never delete the cmake build cache.

- **Kernel variant verification**: After adding a new kernel variant (e.g., packed bf16), add a runtime first-dispatch log to confirm the variant actually fires for production shapes. The `check_transpose` dispatcher materializes `b_transposed` views for the common `y = x @ W.T` pattern — kernels gated on `!b_transposed` almost never fire for Qwen3.5 shapes.

- **Benchmark noise**: The noise floor is ~5% per sample. Single-digit % deltas need at least 5 warm samples after a pipeline warmup run. Cold samples (TTFT > 300ms) must be excluded.
