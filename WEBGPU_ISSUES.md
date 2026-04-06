# WebGPU Backend Issues — Status Update

## Current State: 78/78 WebGPU op tests pass, 0 GPU errors

### Remaining Blocker: Model Code Assertion in GDN slice_axis

The WebGPU backend is working correctly (all 78 unit tests pass). The model crash is now an **assertion failure in the Rust model code**, not a WebGPU issue:

```
Assertion failed: size() > index (mlx/small_vector.h: operator[]: 315)
  at mlx_array_slice_axis
  at MxArray::slice_axis
  at gated_delta_ops          ← Rust model code
  at gated_delta_update
  at GatedDeltaNet::forward
  at DecoderLayer::forward
```

The GDN (Gated Delta Net) layer's `gated_delta_ops` function calls `slice_axis` with an invalid axis/index, causing a SmallVector out-of-bounds access. This is a model logic bug in:
- `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs`

This needs investigation in the Rust model code to fix the slice parameters for the WebGPU/f32 path.

---

# WebGPU Backend Issues — Help Needed (Updated)

## Context

We're building a WebGPU compute backend for the [MLX](https://github.com/ml-explore/mlx) machine learning framework, targeting browser-based inference via WASM. The backend implements ~30 GPU compute operations (matmul, reduce, binary ops, etc.) as WGSL compute shaders dispatched through the `webgpu.h` C API.

**Current state**: 47/53 unit tests pass. 6 failures remain, blocking Qwen3.5 0.8B inference in the browser.

---

## Issue 1 (FIXED): `mean(x, axis)`, bf16, slice — all fixed by WebGPU engineer

All three original issues are resolved. 58/58 unit tests pass with 0 GPU errors.

---

## Issue 4: Dispatch Y=124160 clamped during model inference (CRITICAL)

### Status
Only appears during full Qwen3.5 0.8B inference, NOT in any unit test. The model generates `!!!!!!` (garbage) instead of real text. Performance: TTFT ~5s, decode 1.2 tok/s, 0 GPU validation errors.

### Console Error (during inference only)
```
[WebGPU] WARNING: dispatch Y=124160 clamped to 65535 (original x=1 z=1)
```

This means something directly dispatches `(1, 124160, 1)`. The clamping skips ~half the computation.

### Analysis
- 124160 = 248320 / 2, where 248320 = vocab_size
- The dispatch has `x=1, z=1`, so it's NOT from tiled matmul (which has `wg_x > 1`)
- It's NOT from 1D redistribution (which sets y from x when y==1)
- Tiled matmul Y-chunking code IS working (verified — no warning from that path)
- Unit tests for vocab-sized operations pass: copy(248320), matmul(1x64 * 64x248320), argmax(248320)

### Root Cause Found — bf16 roundtrip (astype + readback) corrupted at ALL sizes

**Minimal repro** (in test-worker.ts):
```js
const BF16 = 3;
const a = MxArray.fromFloat32(new Float32Array([1, 2, 3]), [3]).astype(BF16);
a.eval();
a.toFloat32();  // [0] = 1.636 (WRONG, expected 1.0)
```

Fails at every size (3, 16, 64, ..., 8192). The first element is consistently wrong.

**What works**: bf16 binary ops that return f32 output (e.g., `bf16_a.add(bf16_b)` outputs f32 and reads back correctly). The issue is specifically in reading back a bf16-typed array via `toFloat32()`.

**Root cause**: `toFloat32()` calls `copy_to_buffer` in `mlx_common.h:220` which does:
1. `astype(flat_bf16, float32)` → triggers `copy_gpu` with bf16 input → f32 output
2. The bf16 input's GPU buffer has f32 data (correct — stored as f32)
3. The astype shader is `cast_f32_f32` (identity, since `dtype_to_wgsl(bf16)="f32"`)
4. But something in the GPU→CPU readback path corrupts the data

The value `1.636` = bf16 `0x3FD1`. The correct bf16 for `1.0` is `0x3F80`. The bits are shifted/corrupted, suggesting the readback is not converting f32→bf16→f32 correctly, or the buffer offsets are wrong during the readback mapAsync.

**Key files**:
- `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/src/mlx_common.h:220-240` — `copy_to_buffer(float*)` readback
- `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/utils.h:166-195` — `wgpu_buffer()` CPU→GPU sync (has bf16→f32 conversion on upload)
- GPU→CPU readback path (wherever `data<T>()` resolves for WebGPU arrays after eval)

### Likely Source
The dispatch `(1, 124160, 1)` comes from a copy/binary operation on a **bf16 tensor** with 248320 elements. The bf16 data is 248320 * 2 = 496640 bytes = 124160 u32 elements. If the copy shader uses `array<u32>` indexing with elem_count=124160 and dispatches `(1, elem_count, 1)` instead of the standard 1D `(workgroups, 1, 1)` pattern, this would explain it.

**The bf16→f32 storage conversion should eliminate raw bf16 u32 copies.** Check if any copy path still uses the old bf16 byte counting (N*2 bytes) instead of the expanded f32 size (N*4 bytes).

### Key Files to Check
| File | What to Look For |
|------|-----------------|
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/copy.cpp` | `elem_count` calculation — is it using `nbytes/4` which gives wrong count for bf16? |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/binary.cpp` | Any dispatch that uses array size in bytes/4 as element count |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/gpu/primitives.cpp` | Shared GPU primitives (Full, Copy, etc.) that might bypass the WebGPU-specific allocation |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/device.cpp:439-457` | `clamp_dispatch_dims` — where the warning is printed |

### Also Fixed
- `writeBuffer` bytes not multiple of 4: `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/utils.h` — aligned to 4 bytes

### Reproduce
```bash
cd /Users/brooklyn/workspace/github/mlx-node-browser/packages/browser
npx vite --port 5173
# Open http://localhost:5173/ → type any message → see !!!!! output
# Console shows: dispatch Y=124160 clamped to 65535
```

## Issue 1: `mean(x, axis)` returns all zeros (CRITICAL)

### Reproduction

```js
// In test-worker.ts (run via http://localhost:5173/test-ops.html)
const x = MxArray.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const m = x.mean(axis=[1]);  // Expected: [2, 5], Got: [0, 0]
```

**Works**: `x.mean()` (global mean, no axis) → correct  
**Works**: `x.mean(axis=[0])` on 1D array → correct  
**Works**: `x.sum(axis=[1])` → `[6, 15]` (correct)  
**Works**: `x.sum(axis=[1]).mul(scalar(1/3))` → `[2, 5]` (correct, manually decomposed)  
**Fails**: `x.mean(axis=[1])` → `[0, 0]` (MLX internally does `sum(x, axis) * (1/N)`)

### Analysis

MLX's `mean` is implemented as `multiply(sum(x, axes), array(1.0/N, float32))` in:
- `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/ops.cpp:2117-2119`

The manual decomposition works, so the individual ops (Reduce, Multiply) are correct. The issue is specific to how MLX's internal graph connects them.

We already added per-dispatch compute pass isolation (`end_compute_pass()` after every dispatch in `device.cpp:475,494`) to prevent cross-dispatch buffer aliasing. This fixed ~30 aliasing errors but 1-2 remain.

The remaining error is a **single-dispatch aliasing** — within ONE dispatch, the same `WGPUBuffer` handle is bound to both a `storage, read` binding and a `storage, read_write` binding. This happens when MLX's allocator returns the same buffer for the multiply's output that was used for one of its inputs.

### GPU Validation Error (from Chrome DevTools)

```
[Buffer (unlabeled)] usage (Storage(read-write)|Storage(read-only)) includes 
writable usage and another usage in the same synchronization scope.
 - While validating compute pass usage.
 - While finishing [CommandEncoder (unlabeled)].
```

### Key Source Files

| File | What |
|------|------|
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/binary.cpp:292-394` | `binary_op_gpu_dispatch` — creates bind group with `in_a(read)`, `in_b(read)`, `out(read_write)`. If `in_a_buf == out_buf` or `in_b_buf == out_buf`, WebGPU rejects it. |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/binary.cpp:370-387` | Bind group creation — `wgpu::wgpu_buffer(a)`, `wgpu::wgpu_buffer(b)`, `wgpu::wgpu_buffer(out)` |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/copy.cpp:122-131` | Same pattern in copy dispatch |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/utils.h:92-117` | `create_bind_group` helper |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/device.cpp:409-496` | `dispatch_compute` with per-dispatch pass isolation |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/allocator.cpp` | WebGPU buffer allocator with cache/pool |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/ops.cpp:2097-2119` | MLX `mean()` implementation |

### Possible Fix Approaches

1. **Detect aliasing in bind group creation**: Before creating the bind group, check if any `read` buffer handle equals the `read_write` buffer handle. If so, allocate a temp buffer, `copyBufferToBuffer` the data, and use the temp for the read binding.

2. **Prevent aliasing in the allocator**: When `allocator::malloc` is called for the output buffer, ensure it doesn't return a buffer that's currently in use by any input of the same primitive. This may require tracking "in-flight" buffers.

3. **Detect and re-allocate in eval_gpu**: In each primitive's `eval_gpu`, after `out.set_data(allocator::malloc(...))`, check if `wgpu_buffer(out) == wgpu_buffer(any_input)`. If so, re-allocate.

---

## Issue 2: bfloat16 buffer misalignment

### Reproduction

```js
const BF16 = 3; // DType::BFloat16 enum value
const a = MxArray.fromFloat32([1, 2, 3]).astype(BF16);
const b = MxArray.fromFloat32([4, 5, 6]).astype(BF16);
const c = a.add(b);  // Expected: [5, 7, 9], Got: [5, 7, 7] (partial corruption)
```

### Root Cause

WGSL has no `bfloat16` type. We map `dtype_to_wgsl(bfloat16) → "f32"` so shaders use `array<f32>` (4 bytes/element). But the GPU buffer is allocated for bf16 data: `N * 2` bytes. The shader reads 4 bytes per element from a 2-byte-per-element buffer → misaligned reads, out-of-bounds access.

### Key Source Files

| File | What |
|------|------|
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/utils.h:182-183` | `dtype_to_wgsl(bfloat16)` returns `"f32"` |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/allocator.cpp:30-42` | `malloc(size)` — allocates `size` bytes, doesn't know dtype |

### Fix Approach

For bf16 on WebGPU, we must store data as f32 on the GPU:
- Allocate `N * 4` bytes instead of `N * 2` for bf16 arrays
- Convert bf16→f32 on CPU→GPU sync (`wgpu_buffer()` in `utils.h:135-147`)
- Convert f32→bf16 on GPU→CPU readback
- bf16→f32 conversion: `f32_bits = bf16_bits << 16` (bf16 is upper 16 bits of IEEE 754 f32)

The allocator doesn't know the dtype (only receives total bytes). Options:
- Add a `wgpu_alloc_size(array)` helper: `max(itemsize, 4) * data_size`
- Change all `out.set_data(allocator::malloc(out.nbytes()))` calls to use this helper
- Or modify the allocator to accept dtype info

**Note**: We already handle bf16→f32 conversion in the JavaScript weight upload path (`gpu-worker.ts:178-220`). The C++ fix is needed for intermediate activations created during inference.

---

## Issue 3: Slice readback offset (LOW PRIORITY)

### Reproduction

```js
const a = MxArray.fromFloat32([10, 20, 30, 40, 50], [5]);
const s = a.slice([1], [4]);  // View with offset=1, shape=[3]
s.eval();
s.toFloat32();  // Expected: [20, 30, 40], Got: [50, ?, ?]
```

### Root Cause

`toFloat32()` reads `data_size * itemsize` bytes from the GPU buffer starting at byte 0, but the sliced view has a non-zero element offset (`array::offset() = 1`). The readback doesn't account for this offset.

The readback path goes through `copy_to_buffer` in:
- `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/src/mlx_common.h` — `arr.data<uint8_t>()` should include offset, but GPU buffer mapping may not.

---

## Environment

- **Browser**: Chrome (WebGPU via Dawn)
- **WASM**: wasm32-wasip1-threads, compiled with wasi-sdk + wasm-exceptions
- **Architecture**: Two-worker model (wasm-worker + gpu-worker communicating via SharedArrayBuffer + Atomics)
- **Test page**: `http://localhost:5173/test-ops.html` (run `npx vite --port 5173` from `packages/browser/`)
- **Test file**: `/Users/brooklyn/workspace/github/mlx-node-browser/packages/browser/src/test-worker.ts`

## How to Build & Test

```bash
cd /Users/brooklyn/workspace/github/mlx-node-browser
bash packages/browser/build-wasm.sh
cp packages/browser/dist/index.wasm packages/core/mlx-core.wasm32-wasi.opt.wasm
cd packages/browser && npx vite --port 5173
# Open http://localhost:5173/test-ops.html in Chrome
```
