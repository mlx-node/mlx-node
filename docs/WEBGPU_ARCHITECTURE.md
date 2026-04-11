# WebGPU Backend Architecture

The WebGPU backend enables MLX to run on any platform with a WebGPU implementation — including browsers (via WASM), desktop (via Dawn or wgpu-native), and embedded systems. This document covers the full architecture: the C++ kernel layer, the browser bridge, the WASM build system, and the pitfalls we encountered along the way.

## Table of Contents

- [Overview](#overview)
- [Two-Worker Architecture (Browser)](#two-worker-architecture-browser)
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

- **mlx** (`mlx/backend/webgpu/`) — C++ GPU kernels that generate WGSL shader source at runtime, compile and cache pipelines, and dispatch compute work via the standard `webgpu.h` C API. 36 source files, ~13,500 lines.
- **mlx-node-browser** (`packages/browser/`) — TypeScript bridge that implements `webgpu.h` functions as JavaScript calls into the browser's `GPUDevice`, connected to the WASM-compiled MLX via SharedArrayBuffer + Atomics RPC. ~8,500 lines across 10 source files.

Three `webgpu.h` implementations are supported:

| Backend       | Description                                    |
|---------------|------------------------------------------------|
| `DAWN`        | Google's reference WebGPU (desktop)            |
| `WGPU`        | wgpu-native pre-built library (desktop)        |
| `WASI_IMPORT` | WASM target — functions imported from JS       |

For the browser target (`WASI_IMPORT`), all `wgpu*` function calls become unresolved WASM imports, satisfied at runtime by the TypeScript bridge in mlx-node-browser.

---

## Two-Worker Architecture (Browser)

The browser deployment uses a two-worker architecture to work around the constraint that WebGPU's `GPUDevice` can only be used on the thread that created it, while WASM needs synchronous blocking (`Atomics.wait`) that is forbidden on the main thread:

```
Main Thread (UI)          wasm-worker              gpu-worker
    |                        |                         |
    | <- postMessage <-      | <- SharedArrayBuffer -> |
    |   (results)            |   (Atomics RPC)         |
    |                        |                         |
    | <- Atomics.waitAsync < | -> Atomics.wait ->      |
    |   (stream tokens)      |   (blocks for GPU)      | <- GPUDevice
```

**wasm-worker** (`src/mlx-worker.ts`, 675 lines) — Runs the compiled MLX WASM module. When MLX C++ code calls a WebGPU function (e.g., `wgpuDeviceCreateComputePipeline`), the bridge stub encodes the call into a SharedArrayBuffer command region and wakes the gpu-worker via `Atomics.notify`. Then it blocks on `Atomics.wait` until the gpu-worker writes back the result.

**gpu-worker** (`src/gpu-worker.ts`, 1,679 lines) — Owns the `GPUDevice`. Sits in an `Atomics.waitAsync` loop. When woken, it reads the command from SharedArrayBuffer, executes the corresponding WebGPU API call, writes the result back, and notifies the wasm-worker.

### SharedArrayBuffer RPC Protocol

Defined in `src/rpc-protocol.ts` (193 lines). The command buffer is a fixed-size 512-byte SharedArrayBuffer:

```
Offset  Field              Size     Description
------  -----              ----     -----------
0       FN_ID              4 bytes  RPC function ID (51 function IDs defined)
4       STATUS             4 bytes  Atomics wait/notify flag (Int32Array)
8       RESULT             4 bytes  Return value (low 32 bits)
12      RESULT_HI          4 bytes  High 32 bits for u64 returns
16      ARG0..ARG7         32 bytes Up to 8 u32 arguments
48      ARG0_HI..ARG3_HI   16 bytes High bits for u64 args
64      CALLBACK_COUNT     4 bytes  Pending callback count (or bind group entry count)
68      CALLBACK_BASE      120 bytes Callback payloads (8 entries x 16 bytes)
188     UNIFORM_DATA_SIZE  4 bytes  Inline uniform buffer size
192     UNIFORM_DATA       256 bytes Inline uniform buffer data
448     Reserved           64 bytes
```

Each RPC call:

1. wasm-worker writes `FN_ID` and `ARG0..ARGn`
2. wasm-worker sets `STATUS = PENDING` and calls `Atomics.notify`
3. wasm-worker calls `Atomics.wait(STATUS, PENDING)` (blocks)
4. gpu-worker wakes, reads command, executes, writes `RESULT`
5. gpu-worker sets `STATUS = DONE` and calls `Atomics.notify`
6. wasm-worker wakes and reads the result

**Fused dispatch optimizations**: To reduce RPC round-trips, the bridge batches multiple WebGPU calls into single RPC commands:

| RPC ID | Name | Description |
|--------|------|-------------|
| 91 | `FUSED_DISPATCH` | setPipeline + setBindGroup + dispatch |
| 92 | `FUSED_DISPATCH_2BG` | setPipeline + 2x setBindGroup + dispatch |
| 93 | `FUSED_SUBMIT` | endPass + finish + submit + release |
| 94 | `FUSED_BG_DISPATCH` | createBindGroup + setPipeline + setBindGroup + dispatch |
| 95 | `CREATE_BUFFER_FROM_DATA` | createBuffer + writeBuffer (replaces mappedAtCreation) |
| 96 | `FUSED_FULL_DISPATCH` | inline createBindGroup + setPipeline + setBindGroup + dispatch |
| 97 | `FUSED_DISPATCH_WITH_UNIFORM` | FUSED_FULL_DISPATCH + inline uniform buffer write |
| 98 | `FUSED_COPY_BUFFER` | endPass + copyBuffer + beginPass |

`FUSED_DISPATCH_WITH_UNIFORM` (ID 97) is the hot-path workhorse — it turns what would be 5+ RPCs (createBuffer, getMappedRange, unmap, setPipeline, setBindGroup, dispatch) into 1.

### Handle Management (gpu-worker)

GPU objects (buffers, pipelines, bind groups, etc.) are tracked by integer handles in a sparse array on the gpu-worker side. When the wasm-worker calls `wgpuDeviceCreateBuffer`, the gpu-worker creates the real `GPUBuffer`, stores it in the handle table, and returns the integer handle. All subsequent references use this handle.

### Weight Upload Flow

1. mlx-worker downloads SafeTensors weights into a `SharedArrayBuffer`
2. `postMessage({ type: 'upload_weights', weightsSab, tensors })` to gpu-worker
3. gpu-worker creates GPU buffers with `mappedAtCreation: true`
4. bf16 weights: either expanded to f32 inline (`dst32[j] = src16[j] << 16`) or kept packed as u32 pairs when `pack_bf16=1` is enabled
5. f16 weights: IEEE 754 conversion to f32 (or kept as f16 if `shader-f16` available)
6. `gpuBuffer.unmap()`, return handle to wasm-worker
7. wasm-worker passes handles to C++ via `mlx_array_from_gpu_buffer()` FFI

### Token Streaming

Decoded tokens are streamed to the main thread via a separate SharedArrayBuffer channel (256 KB):

```
[0..3]   u32: write cursor (byte offset)
[4..7]   u32: sequence number (incremented per token)
[8..N]   utf-8 text data (cumulative)
```

The WASM module writes tokens via `mlx_stream_write()`, increments the sequence counter with `Atomics.add`, and wakes the main thread with `Atomics.notify`. The main thread uses `Atomics.waitAsync` for non-blocking per-token rendering.

### Browser Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/gpu-worker.ts` | 1,679 | GPU thread: WebGPU API calls, handle table, fused dispatch |
| `src/webgpu-bridge-stub.ts` | 1,091 | C-API stubs: encodes wgpu* calls into RPC |
| `src/webgpu-bridge.ts` | 620 | Bridge setup and initialization |
| `src/mlx-worker.ts` | 675 | WASM thread: model loading, inference orchestration |
| `src/test-worker.ts` | 3,913 | Browser test suite (178 test cases) |
| `src/rpc-protocol.ts` | 193 | SharedArrayBuffer layout and RPC function IDs |
| `src/cxx-stubs.ts` | 131 | C++ standard library stubs for WASM |
| `src/wasm-loader.ts` | 116 | WASM module loading and instantiation |
| `src/safetensors.ts` | 117 | SafeTensors file parsing |
| `src/index.ts` | 6 | Package entry point |
| `demo/app.ts` | 385 | Demo app: Qwen3.5 chat UI |
| `demo/test-ops-runner.ts` | 62 | Test runner HTML page logic |

### URL Query Parameters

The demo app (`demo/index.html`) and test page (`demo/test-ops.html`) support runtime feature toggles via URL query params:

| Param | Effect |
|-------|--------|
| `?pack_bf16=1` | Enable packed bf16 weight storage (2 bf16 per u32 on GPU) |
| `?sdpa_fallback=1` | Force SDPA onto decomposed matmul+softmax+matmul fallback |

These are parsed in `demo/app.ts` (lines 296-297), passed into the worker `init` message, and forwarded to the WASM backend as runtime config flags via `mlx_wgpu_set_packed_bf16_enabled()` and `mlx_wgpu_set_sdpa_fallback_forced()`. Note: `demo/test-ops-runner.ts` does not parse URL query params — it only spawns the test worker and displays results.

---

## C++ Backend Architecture

All kernel code lives in `mlx/backend/webgpu/`. Each `.cpp` file typically implements one or more MLX primitives by:

1. Defining a C++ params struct (uniform buffer layout)
2. Generating WGSL shader source as a string
3. Caching the compiled pipeline via `device().get_or_create_shader_module()`
4. Setting up bind groups and dispatching compute work

### Source Files

| File | Lines | Operations |
|------|-------|------------|
| `sdpa.cpp` | 1,608 | Fused SDPA: vector (decode), 2-pass vector (long-context decode), tile (prefill) |
| `reduce.cpp` | 1,135 | Sum, Prod, Max, Min, And, Or (3 strategies) |
| `indexing.cpp` | 1,114 | Gather, GatherAxis, Scatter, ScatterAxis, SliceUpdate |
| `matmul.cpp` | 1,065 | GEMV (split-K), multi-col GEMV, GEMM (tiled 16x16), packed bf16 variants |
| `device.cpp` | 763 | Device init, feature detection, pipeline caching, packed bf16 upload |
| `compiled.cpp` | 616 | Fused element-wise kernel JIT (Compiled primitive) |
| `normalization.cpp` | 590 | RMSNorm (2-phase) and LayerNorm (3-phase), packed bf16 weight variants |
| `allocator.cpp` | 525 | Buffer allocation, page cache, readback, storage mode management |
| `rope.cpp` | 469 | RoPE: single-token and general variants |
| `scan.cpp` | 461 | Prefix sum/prod/max/min/logaddexp |
| `sort.cpp` | 390 | Bitonic sort and argsort in shared memory |
| `binary.cpp` | 377 | 18 ops: Add, Multiply, Power, Maximum, comparisons, etc. |
| `conv.cpp` | 339 | Depthwise and grouped conv1d |
| `unary.cpp` | 327 | 29 ops: Abs, Exp, Log, Sin, Cos, Sqrt, Sigmoid, Erf, etc. |
| `copy.cpp` | 301 | Copy with 6 dtype conversion modes |
| `ternary.cpp` | 283 | Select (where/conditional) |
| `random.cpp` | 276 | RandomBits via Threefry-2x32-20 PRNG |
| `quantized.cpp` | 274 | QuantizedMatmul (int2/int4/int8 on-the-fly dequant) |
| `arg_reduce.cpp` | 218 | ArgMax, ArgMin |
| `softmax.cpp` | 222 | Online softmax (3-phase: max, sum-exp, normalize) |
| `logsumexp.cpp` | 202 | LogSumExp (2-phase: max, log-sum-exp) |
| `arange.cpp` | 158 | Integer and float range generation |
| `slicing.cpp` | 136 | Concatenate, dynamic slice offset |
| `primitives.cpp` | 115 | NO_GPU stubs and cross-reference comments |
| `op_exprs.h` | 310 | Shared unary/binary expression builders for compiled.cpp |
| `utils.h` | 382 | Type helpers, bind group creation, WGSL codegen, packed bf16 helpers |
| `device.h` | 193 | WebGPUDevice class, UniformBufferPool, pipeline cache |
| `allocator.h` | 112 | WebGPUBuffer struct, StorageMode enum, allocator interface |
| `worker.h` | 50 | Background worker for async GPU work |

**Total: 36 files, ~13,500 lines.**

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

| Dtype    | CPU size | GPU type       | GPU size |
|----------|----------|----------------|----------|
| float32  | 4 bytes  | `f32`          | 4 bytes  |
| float16  | 2 bytes  | `f16` or `f32` | 2 or 4   |
| bfloat16 | 2 bytes  | `f32`          | 4 bytes  |
| bool     | 1 byte   | `u32`          | 4 bytes  |
| int32    | 4 bytes  | `i32`          | 4 bytes  |
| uint32   | 4 bytes  | `u32`          | 4 bytes  |

This mismatch requires careful handling throughout:

**Allocation**: `wgpu_alloc_size(arr)` computes GPU buffer size using `wgpu_itemsize()` which returns 4 for bool and bfloat16.

**Upload conversion**: When data is uploaded to the GPU (`device.cpp` `upload_with_conversion`), bfloat16 values are expanded to float32 (`bf16_bits << 16`) and bool values are expanded to uint32. When `StorageMode::PackedBf16` is active, bf16 values are instead packed as u32 pairs (see [Packed bf16 Storage](#packed-bf16-storage)).

**Download conversion**: When data is read back (`allocator.cpp` `raw_ptr()`), float32 values are truncated back to bfloat16 and uint32 values are truncated to bool.

**Offset calculation**: Array offsets are always in bytes on the CPU side. To get the GPU element index:

```cpp
uint32_t gpu_elem_idx = arr.offset() / arr.itemsize();  // bytes -> elements
```

This works because each CPU element maps 1:1 to a GPU element, regardless of size difference.

**`dtype_to_wgsl_safe()`**: Returns `"f32"` instead of `"f16"` when the device lacks `shader-f16` support, ensuring graceful fallback.

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

`StorageMode` is a field on `WebGPUBuffer`. The runtime flag `packed_bf16_enabled()` (set via `?pack_bf16=1` URL param) controls opt-in. Eligible bf16 weight buffers (>= 4096 elements) are kept packed.

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

### Two Upload Paths

1. **C++ path** (`upload_with_conversion`): packs bf16 CPU data into u32 pairs during upload. Used when MLX allocates the buffer.
2. **JS path** (`gpu-worker.ts`): the gpu-worker packs bf16 SafeTensors weights directly into GPU buffers at upload time, then calls `mlx_wgpu_mark_buffer_packed_bf16()` to flip the storage mode flag. This is the hot path for model weight loading.

### Performance Notes

On M3 with Qwen3.5-0.8B, packed bf16 gives ~3-4% decode speedup (not the initially projected +10%) because individual MLP weights fit in L2 cache. The win is concentrated in the large `linear_attn.in_proj_qkvz` weight (16 MiB/layer) that exceeds L2.

---

## Fused SDPA Kernels

The `ScaledDotProductAttention` primitive (`sdpa.cpp`, 1,608 lines) implements three kernel variants, replacing the decomposed matmul+softmax+matmul fallback:

### Kernel Variants

| Variant | Dispatch | When Used |
|---------|----------|-----------|
| **Vector (single-pass)** | B*H workgroups, 128 threads each | Tq=1 decode, L <= 128 |
| **Vector 2-pass** | Pass 1: B*H*nblocks WGs; Pass 2: B*H WGs | Tq=1 decode, L > 128 |
| **Tile** | ceil(Tq/BQ)*H*B workgroups | Tq > 1 prefill |

**Supported head dims**: D in {64, 96, 128, 256}. At D=256, the vector kernel uses `D_PER_THREAD=2` (each of 128 threads owns 2 output lanes).

### Vector Kernel (Tq=1 Decode)

`make_sdpa_vector_kernel()`: One workgroup per (batch, head). 128 threads cooperatively process the full K/V sequence. Uses shared memory for Q vector and workgroup reduction. Subgroup reduction used when available.

### 2-Pass Vector (Long-Context Decode)

Mirrors Metal's `sdpa_vector_2pass_1` / `sdpa_vector_2pass_2`:

- **Pass 1** (`make_sdpa_vector_2pass_1_kernel`): Splits L into blocks of `SDPA_BLOCK_L=128`. Each workgroup handles one (batch, head, L-block), producing partial `(O[D], max, sum)`.
- **Pass 2** (`make_sdpa_vector_2pass_2_kernel`): One workgroup per (batch, head) merges nblocks partial results via online softmax reweighting.

Gating: dispatched when `nblocks = ceil(L / 128) > 1`.

### Tile Kernel (Prefill)

`make_sdpa_tile_kernel()`: FlashAttention-2-style online softmax tiled attention. Hand-coded because WGSL lacks `simdgroup_matrix`.

- Tile shape: BQ=16 rows of Q, BK=8 columns of K/V per inner tile
- Workgroup layout: (BK, BQ, 1) = (8, 16, 1), 128 threads total
- Dispatch: `(ceil(Tq/BQ), H_q, B)`
- Supports: causal masking (via `q_offset`), additive float mask, no mask
- D=256 fits within Chromium's 32 KiB shared memory budget

### Fallback Conditions

`use_fallback()` returns `true` (decomposed path) when:
- `?sdpa_fallback=1` runtime flag is set
- Training or logsumexp output requested
- D not in {64, 96, 128, 256}
- K/V shape mismatch
- Non-float32/bfloat16 dtype
- Scalar bool mask (non-causal, non-array)

All other cases (including Tq > 1 prefill) use the fused kernels. Note: attention sinks (`has_sinks_`) are not checked in `use_fallback()` but are rejected with a throw in `eval_gpu()` (sdpa.cpp:1232-1240) since the flag is not available in the static fallback check.

### Performance

On Qwen3.5-0.8B in Chrome (M3):

| Configuration | Decode tok/s |
|---|---|
| Decomposed fallback (3 dispatches) | 26.5 |
| 2-pass fused vector (2 dispatches) | 30-31 |

The fused tile kernel for prefill significantly reduces TTFT by eliminating the softmax HBM roundtrip.

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

**Single source of truth**: `packages/core/build.ts` is the canonical build entry point for WASM. It sets all environment variables (`RUSTFLAGS`, `TARGET_CXXFLAGS`) and build options. The shell script `packages/browser/build-wasm.sh` mirrors these flags for convenience but `build.ts` is authoritative. Do not add build flags in other locations (`.cargo/config.toml`, `mlx-sys/build.rs`, etc.).

Key build flags:

```bash
# Environment (set by build.ts)
RUSTFLAGS='--cfg tokio_unstable'                    # Multi-thread tokio runtime on WASM
TARGET_CXXFLAGS='-fwasm-exceptions -fexceptions'    # Required by esaxx-rs (C++ try/catch)

# C++ compilation (via cmake, set in mlx-sys/build.rs)
-fwasm-exceptions          # WASM native exception handling
-femit-all-decls           # Forces emission of inline virtual methods
-fvisibility=hidden        # Consistent vtable visibility
-fvisibility-inlines-hidden

# C++ compilation (bridge files via cc::Build in mlx-sys/build.rs)
-fexceptions
-fwasm-exceptions
-fvisibility=hidden
-fvisibility-inlines-hidden

# Linking
-Wl,--whole-archive=mlx    # Prevents vtable method GC
-zstack-size=64000000      # 64 MB stack (set by napi_build::setup() in crates/mlx-core/build.rs)

# cmake defines
-DMLX_BUILD_WEBGPU=ON
-DWEBGPU_BACKEND=WASI_IMPORT
-DMLX_USE_WEBGPU
-DWEBGPU_BACKEND_WASI_IMPORT
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

---

## Performance Characteristics

As of the latest measurements (April 2026) on Chrome desktop (M3):

| Metric              | Value      |
|---------------------|------------|
| Decode throughput   | ~30 tok/s  |
| Time to first token | ~196 ms    |
| Test suite pass rate| 177/178 (1 expected failure) |

Model: Qwen 3.5 0.8B (bf16, 24 layers, GDN linear attention)

The 1 remaining test failure is expected:
- Stream channel test (1 test) — expects native environment

### Key Optimizations (in order of impact)

| Optimization | Speedup | Description |
|---|---|---|
| Fast-path dispatch handlers | 10.8 -> 17.5 tok/s | Inlined gpu-worker command loop; adaptive spin before Atomics.waitAsync |
| Fused SDPA vector kernel | 17.5 -> 20 tok/s | Q*K^T -> softmax -> *V in one dispatch for Tq=1 decode |
| Pipeline warmup | 20 -> 25.7 tok/s | 2-token dummy inference at load forces kernel compile before real request |
| Split-K GEMV | 25.7 -> 28.1 tok/s | One workgroup per output element, 256 threads cooperatively reduce K |
| Multi-col GEMV (COLS_PER_WG=4/8) | 28.1 -> 30.6 tok/s | Amortize A-vector loads across 4-8 output columns |
| N-aligned fast path | 30.6 -> 31.3 tok/s | Drop has1..hasN-1 guards when N % cols_per_wg == 0 |
| 2-pass split-L SDPA | Matched decomposed | 2 dispatches with B*H*nblocks WGs; needed for long contexts |

### What Didn't Work

- **mc16 for N>=32768** — 16 accumulators/thread exceeded register pressure
- **WG_SIZE 128->256** — K=896 already covered in 7 iterations
- **vec4 B reads (non-transposed)** — gated on `!b_transposed`, but Qwen3.5 linear layers all land on `b_transposed` via `check_transpose`, so the kernel never dispatched
- **Subgroup-only final reduce** — WGSL subgroup ops require workgroup-uniform control flow

---

## Operations Not Yet Implemented

These operations throw or fall back to CPU decomposition:

| Operation                     | Status          | Notes                                          |
|-------------------------------|-----------------|------------------------------------------------|
| `ArgPartition`                | NO_GPU (throws) | Rarely used                                    |
| `BitwiseBinary`               | NO_GPU          | Simple to implement if needed                  |
| `BlockMaskedMM`               | NO_GPU          | Sparse matmul variant                          |
| `FFT`                         | NO_GPU          | Not needed for transformer inference           |
| `GatherMM`                    | NO_GPU          | Used by some MoE implementations               |
| `GatherQMM`                   | NO_GPU          | Quantized variant of GatherMM                  |
| `Hadamard`                    | NO_GPU          | Rarely used                                    |
| `Inverse` / `Cholesky`        | NO_GPU          | Linear algebra decompositions                  |
| `SVD` / `Eigh` / `Eig`        | NO_GPU          | Eigenvalue decompositions                      |
| `LUF` / `QRF`                 | NO_GPU          | Matrix factorizations                          |
| `SegmentedMM`                 | NO_GPU          | Segmented matmul                               |
| `MaskedScatter`               | NO_GPU          | Masked scatter variant                         |
| `Conjugate`                   | NO_GPU          | Complex conjugate                              |
| `DivMod`                      | NO_GPU          | Integer division + modulo (multi-output)        |
| `Imag`                        | NO_GPU          | Imaginary part of complex                      |
| `Load`                        | NO_GPU          | Load from file                                 |
| `Partition`                   | NO_GPU          | Partial sort / partition                       |
| `Real`                        | NO_GPU          | Real part of complex                           |
| `CustomKernel`                | NO_GPU          | User-defined custom kernels (fast:: namespace) |
| `SDPA VJP`                    | Fallback        | Training backward pass (uses decomposed path)  |
| `LayerNorm VJP` / `RMSNorm VJP` | NO_GPU       | Normalization backward passes                  |
| `ConvertFP8` / `Quantize`     | NO_GPU          | Quantization ops                               |

**Fully implemented** (fused GPU kernels, not fallback): `ScaledDotProductAttention` (vector, 2-pass, tile), `Compiled` (fused element-wise JIT), `Matmul`/`AddMM`, `QuantizedMatmul`, `RMSNorm`, `LayerNorm`, `RoPE`, `Softmax`, `LogSumExp`, `Reduce`, `Scan`, `Sort`/`ArgSort`, `ArgReduce`, `RandomBits`, `Convolution`, `Gather`/`Scatter`, all unary (29 ops) and binary (18 ops), `Select`, `Arange`, `Copy`.

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

- **WASM crashes**: Deep model call stacks can overflow the WASM stack (64 MB, set by `napi_build::setup()` in `crates/mlx-core/build.rs`). Symptoms: silent crash or `RuntimeError: unreachable`. The stack size is managed solely by the napi-build crate — do not add redundant `-zstack-size` flags elsewhere.

- **WASM build flags**: All WASM build configuration lives in `packages/core/build.ts` (single source of truth). `RUSTFLAGS` (tokio_unstable) and `TARGET_CXXFLAGS` (exception flags for esaxx-rs) are set there. Do not duplicate these in `.cargo/config.toml`, `mlx-sys/build.rs`, or shell scripts.

- **WASM rebuild not picking up changes**: Touch `CMakeLists.txt` to force cmake reconfigure. Touch `build.rs` to force Cargo to re-run the build script. Never delete the cmake build cache.

- **Kernel variant verification**: After adding a new kernel variant (e.g., packed bf16), add a runtime first-dispatch log to confirm the variant actually fires for production shapes. The `check_transpose` dispatcher materializes `b_transposed` views for the common `y = x @ W.T` pattern — kernels gated on `!b_transposed` almost never fire for Qwen3.5 shapes.

- **Benchmark noise**: The noise floor is ~5% per sample. Single-digit % deltas need at least 5 warm samples after a pipeline warmup run. Cold samples (TTFT > 300ms) must be excluded.
