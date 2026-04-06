# WebGPU Debug Status — Model Runs, Output Quality Remaining

## MILESTONE: Model runs without crashing!

**100/100 unit tests pass.** Model loads, runs 24-layer forward pass, generates tokens at 2.1 tok/s, TTFT 2.8s. Zero GPU errors. Zero crashes.

### Fix: `alignas(8)` on ArrayDesc (heap corruption root cause)

**File**: `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/array.h` line 468

```cpp
struct alignas(8) MLX_API ArrayDesc {
```

**Root cause**: On WASM 32-bit, `std::make_shared<ArrayDesc>` places the object after a 12-byte control block (2×4 ref counts + 4 pointer). This puts `ArrayDesc` at a 4-byte-aligned address. But `ArrayDesc` contains `Strides = SmallVector<int64_t>` which has `alignas(int64_t)` (8-byte) inline storage. Unaligned `int64_t` access traps in WASM.

The `alignas(8)` forces `make_shared` to use an 8-byte-aligned allocation, ensuring all `int64_t` accesses within the struct are properly aligned.

### Remaining Issue: Output is `!!!!!!` (garbage)

The model generates token `!` (char 33) repeatedly, stopped by repetition detection. All individual ops pass with synthetic data. The issue is specific to the real model's forward pass with loaded weights.

# WebGPU Debug Status — Need Senior Engineer Help

## Current State

- **85/85 unit tests pass**, 0 GPU validation errors
- Model loads, runs inference to completion, generates tokens at ~1.3 tok/s
- **But output is always `!!!!!!`** — the model generates the same token repeatedly
- No crashes, no exceptions, no dispatch warnings

## What We Fixed This Session

1. **Conv1d WebGPU kernel** (`webgpu/conv.cpp`) — new WGSL compute shader for GDN's depthwise conv1d
2. **Compiled primitive bypass** (`no_cpu/compiled.cpp`) — disabled `mlx::core::compile` for WASI since WebGPU can't generate fused kernels
3. **Dispatch workgroup overflow** (`device.cpp`) — clamping + matmul Y-offset chunking for WebGPU's 65535 limit
4. **Compute pass isolation** (`device.cpp`) — end pass after each dispatch to prevent buffer aliasing
5. **bf16→f32 weight conversion** (`gpu-worker.ts`) — JS-side conversion during model weight upload
6. **bf16 readback corruption** (`mlx_common.h`) — `copy_to_buffer` now converts to f32 before materialization to avoid bf16 raw_ptr readback cycle
7. **bf16 allocation in shared GPU primitives** (`gpu/primitives.cpp`) — `DynamicSlice` and `View` now use `wgpu::malloc_for_array` for bf16-safe allocation
8. **RandomBits dispatch overflow** (`random.cpp`) — split Y into Y*Z when half_size > 65535 (vocab_size/2 = 124160)
9. **GDN custom kernel bypass** (`gated_delta.rs`) — `#[cfg(not(target_family = "wasm"))]` to skip Metal-only `gated_delta_chunked`/`gated_delta_kernel` and `fused_gdn_gating`, which create lazy graphs via CustomKernel that fail at eval time on WebGPU
10. **writeBuffer alignment** (`utils.h`) — align to 4 bytes for WebGPU requirement

## The Remaining Problem

The model generates `!!!!!!` (same token repeated). This means:
- The forward pass computes without errors
- But the logits are degenerate — argmax always picks the same token
- With `temperature: 0.7` (random sampling), still produces `!` — so logits are likely all-same-value or NaN

## What We've Ruled Out (via unit tests)

All these operations work correctly with synthetic data at model-scale dimensions:

| Operation | Test | Status |
|-----------|------|--------|
| Matmul 1x1024 * 1024x248320 | `large matmul` | PASS |
| Matmul 1024→3584→1024 chain | `model-scale embed + linear` | PASS |
| GDN step (4D broadcast mul+sum) | `GDN step model-sized` (Hv=16, Dv=128, Dk=128) | PASS |
| Conv state pattern (concat+slice) | `conv state pattern` | PASS |
| RMSNorm (x^2 → mean → sqrt) | `rmsnorm pattern` | PASS |
| Softmax | `bf16 softmax` | PASS |
| Argmax over vocab (248320) | `argmax on vocab-sized` | PASS |
| compute_g (exp, softplus chain) | `compute_g pattern` | PASS |
| bf16 roundtrip at all sizes | `bf16 roundtrip sizes` | PASS |
| Squeeze, reshape on bf16 | multiple tests | PASS |
| Large copy (248320 elements) | `large array copy` | PASS |

## What We Suspect

Since all individual ops pass with synthetic data, the issue is likely:

### 1. Conv1d kernel bug with real model dimensions
Our conv1d WGSL shader (`webgpu/conv.cpp`) was written from scratch. It may have an indexing bug for the specific GDN depthwise convolution parameters:
- `groups = conv_dim = 384` (key_dim=128*2 + value_dim=128*2... check actual value)
- `kernel_size = 4`, `stride = 1`, `padding = 0`
- `C_per_group = 1` (depthwise)

**How to test**: Compare conv1d output between Metal native and WebGPU for the same input+weights.

### 2. Weight loading byte order or alignment
The JS bf16→f32 conversion `dst32[j] = src16[j] << 16` might have an issue with certain bf16 values. The conversion is mathematically correct for IEEE 754 but maybe the SafeTensors byte layout is different than expected.

**How to test**: Load one weight tensor, read it back, compare bytes with the original SafeTensors file.

### 3. The `compute_g` / `fused_gdn_gating` fallback path
On WASM, we skip `fused_gdn_gating` and use the ops-based fallback:
```rust
let beta = Activations::sigmoid(b)?;
let g = compute_g(a_log, a, dt_bias)?;  // calls mlx_fused_compute_g → compiled_compute_g
let g_log = g.log()?;
```
`compute_g` calls `compiled_compute_g()` which uses `mlx::core::compile`. We disabled compile, so it falls through to `compute_g_impl` which uses standard ops. BUT: does `g.log()` produce the right values? If `g` contains zeros, `log(0) = -inf`, corrupting the entire chain.

**How to test**: Print intermediate values of `g` after `compute_g` to verify they're in a valid range (0, 1).

### 4. The GDN ops-based loop accumulates numerical errors
`gated_delta_ops` iterates over all timesteps sequentially. The ops-based path may accumulate errors differently than the Metal kernel, producing degenerate output for long sequences (prefill with ~20 tokens from system + user message).

**How to test**: Run the model with a single-token input to minimize prefill, check if decode tokens are still `!`.

## Key Files

| File | What |
|------|------|
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/conv.cpp` | Conv1d WGSL kernel (our implementation) |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-core/src/models/qwen3_5/gated_delta.rs` | GDN Rust code with `#[cfg(wasm)]` skips |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs` | GDN forward pass (conv1d + split + reshape) |
| `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/src/mlx_qwen35_common.h` | C++ `compute_g_impl`, `softplus`, `compiled_compute_g` |
| `/Users/brooklyn/workspace/github/mlx-node-browser/packages/browser/src/gpu-worker.ts:178-220` | JS bf16→f32 weight conversion |
| `/Users/brooklyn/workspace/github/mlx-node-browser/packages/browser/src/test-worker.ts` | 85 unit tests |

## How to Reproduce

```bash
cd /Users/brooklyn/workspace/github/mlx-node-browser
bash packages/browser/build-wasm.sh
cp packages/browser/dist/index.wasm packages/core/mlx-core.wasm32-wasi.opt.wasm
cd packages/browser && npx vite --port 5173

# Tests: http://localhost:5173/test-ops.html (85/85 pass)
# Chat:  http://localhost:5173/ (generates !!!!!! instead of real text)
```

## Latest Finding (after compute_g fix)

With `compute_g` reimplemented in pure Rust (bypassing C++ FFI compiled kernel), the crash moved from `compute_g` to `sum()` inside `gated_delta_ops_inner`:

```
dlmalloc → malloc → operator new → mlx::core::sum() → 
gated_delta_ops_inner → gated_delta_update_inner → 
GatedDeltaNet::forward → DecoderLayer::forward
```

This is **heap corruption** — the `sum()` call is just graph construction (no GPU), but `operator new` fails because `dlmalloc`'s metadata is corrupted from an earlier GPU buffer overflow.

The overflow happens during eval of GPU ops in the preceding part of the GDN forward (conv1d → SiLU → split → reshape → gating computation). One of these ops writes beyond its allocated buffer.

**The debug instrumentation added by the senior engineer triggers the same crash** because it calls `eval_safe` on intermediate tensors, which runs GPU ops that overflow. Disabling debug logging (`should_log` returns false) also crashes at the same point, confirming the overflow is in the model's own ops, not debug code.

**All 89 unit tests pass** including model-scale patterns (broadcast [16]*[1,20,16], compute_g chain, GDN step with Hv=16/Dv=128/Dk=128). The overflow only happens with the REAL model's specific weights and tensor shapes.

## BREAKTHROUGH: Model running without crashes!

With two fixes, the model now runs through the full forward pass without any crashes or errors:

### Fix 1: WebGPUBuffer struct initialization (ROOT CAUSE of heap corruption)
**File**: `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/src/mlx_nn_ops.cpp:967-975`

The `mlx_array_from_gpu_buffer` function had a **stale forward declaration** of `WebGPUBuffer` with only 3 fields (buffer, size, cpu_ptr). The real struct has 5 fields (buffer, size, host_size, cpu_ptr, stores_bfloat16_as_f32). The missing `host_size` and `stores_bfloat16_as_f32` fields were UNINITIALIZED, causing `raw_ptr()` to use garbage values during GPU→CPU readback, corrupting the heap.

### Fix 2: WASM memory increased to 1.5GB
**File**: `/Users/brooklyn/workspace/github/mlx-node-browser/packages/browser/src/mlx-worker.ts:51`

Changed `initial: 16000` (1GB) to `initial: 24000` (1.5GB). The model's lazy graph construction for 24 decoder layers requires more than 1GB of WASM heap.

### Current Status
- Model runs for 80+ minutes without errors (prefill in progress)
- Zero GPU validation errors, zero crashes
- The ops-based GDN recurrence is EXTREMELY slow (~7200 sequential GPU dispatches for prefill)
- Each dispatch goes through Atomics.wait RPC to gpu-worker, adding ~1ms per op
- Total prefill time: estimated 1-2 hours for ~20 tokens

### Performance Optimization Needed
The ops-based fallback (`gated_delta_ops`) processes each timestep sequentially. For prefill with T tokens across 24 layers, this creates ~T × 24 × 15 ≈ 7200 sequential GPU dispatches. Each dispatch has RPC overhead.

**To make this practical**, need one of:
1. Implement `gated_delta_kernel` for WebGPU (fused WGSL compute shader)
2. Batch multiple timesteps in a single dispatch
3. Reduce RPC overhead (batch multiple WebGPU calls per RPC)

## Latest: Heap corruption narrowed to chatSync initialization

### Key findings:
1. **95/95 unit tests pass** — all WebGPU ops work correctly in isolation, even at model-scale dimensions
2. **Heap is clean after model load** — `MxArray.fromFloat32([1,2,3]).eval().toFloat32()` returns `[1,2,3]` correctly right before chatSync
3. **Crash happens during chatSync's FIRST decoder layer graph construction** — not during GPU eval
4. **Even with only 1 layer**, it crashes
5. **Stack is 25MB** (data at 25MB, stack pointer at 64MB) — NOT a stack overflow
6. **1.5GB WASM heap** — NOT an OOM
7. **Disabling GPU readback (raw_ptr returns calloc zeros)** still crashes → NOT from GPU readback
8. **The crash is in `dlmalloc` accessing corrupted metadata** — something writes beyond a heap allocation between the warm-up eval and the first matmul

### What corrupts the heap:
The corruption happens in ONE of these steps inside `chatSync`:
- Warm-up eval (`a + a` → GPU dispatch)
- Chat template rendering (minijinja Jinja2 engine, 7755-char template)
- Tokenization (HuggingFace tokenizers crate, 248K vocab)
- Cache creation (24 `Qwen3_5LayerCache` objects)
- Embedding weight transpose
- Embedding lookup (Gather on [248320, 1024] weight)

### ROOT CAUSE FOUND: Unaligned memory access in MLX array construction

When skipping decoder layers (embedding + RMSNorm + lm_head matmul only), a DIFFERENT error appears:

```
RuntimeError: operation does not support unaligned accesses
  at construct_at<mlx::core::array, SmallVector<int,10>, Dtype const&, shared_ptr...>
```

**This is a WASM alignment trap!** The `array` objects created from GPU buffer handles (`mlx_array_from_gpu_buffer`) have memory that isn't properly aligned for `SmallVector<int, 10>`. When the framework constructs array objects (during matmul's unflatten/broadcast), the `construct_at` requires aligned memory access, but the WASM heap allocation returns insufficiently aligned memory.

**The "memory access out of bounds" crash in the full model** is caused by the SAME alignment issue — unaligned accesses to SmallVector data corrupt the dlmalloc metadata, which then causes the out-of-bounds trap on the next allocation.

**File**: `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/src/mlx_nn_ops.cpp` line 1009
**Fix needed**: Ensure `new array(...)` uses aligned allocation, or add `alignas(16)` to the SmallVector type, or use `aligned_alloc` instead of `operator new` for MLX array objects.

### CRITICAL BUG FOUND: `greater()` with broadcast returns all zeros

**Repro** (in test-worker.ts):
```js
const x = MxArray.fromFloat32([1, 5, 10, 25, 50, 100], [6]);
const thresh = MxArray.fromFloat32([20], [1]);
const cond = x.greater(thresh);  // Expected: [F,F,F,T,T,T]
cond.addScalar(0).eval().toFloat32();  // Returns: [0,0,0,0,0,0] — ALL zeros!
```

**Same-shape `greater` works**: `[1,25,50].greater([20,20,20])` → `[0,1,1]` ✓
**Broadcast `greater` fails**: `[1,5,10,25,50,100].greater([20])` → `[0,0,0,0,0,0]` ✗

This causes `softplus(x) = where(x > 20, x, log(1+exp(x)))` to ALWAYS compute `log(1+exp(x))` even for large x → `Inf` → NaN propagation → garbage model output.

**Impact**: ALL comparison ops with broadcast (Greater, Less, Equal, etc.) are broken. This affects `softplus`, `compute_g`, and any conditional logic in the model.

**Likely cause**: The VectorScalar binary shader variant writes to the output buffer correctly, but the bool output buffer allocation or offset calculation has a bug. The `Greater` shader outputs `u32(0)/u32(1)` to `array<u32>`, but the bool dtype has `itemsize()=1`. Our fix to `wgpu_allocation_size` expanded bool to 4 bytes per element, and `wgpu_offset` now divides by `wgpu_storage_itemsize(bool_)=4`, but something in the chain still uses the old 1-byte-per-element math.

**Files to check**:
- `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/binary.cpp` lines 357-393 — offset/size calculation for VectorScalar
- `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/webgpu/utils.h` — `wgpu_offset`, `wgpu_data_size`, `wgpu_storage_itemsize` for bool
- `/Users/brooklyn/workspace/github/mlx-node-browser/crates/mlx-sys/mlx/mlx/backend/common/binary.h` — `set_binary_op_output_data` — does the VectorScalar path allocate with the lambda or bypass it?

### Most likely candidates:
1. **The tokenizers crate on WASM** — uses `oniguruma` C library internally for regex. C library buffer overflows on WASM are hard to detect.
2. **The warm-up eval** — creates+evals a tiny array. The GPU dispatch goes through the WebGPU RPC bridge. If the readback writes beyond bounds...
3. **The embedding Gather** — the embedding weight is ~1GB as f32. The Gather creates an output array referencing this weight. If the weight's buffer metadata is wrong...

### Suggested approach:
Binary search within chatSync by adding `return Ok(dummy_result)` at different points to find the exact line that triggers the corruption. The eprintln! checkpoints we added don't show in the browser console (WASI stderr not connected).

## Suggested Next Step

**Add logging to the Rust model forward pass** to dump intermediate tensor statistics (mean, std, min, max, dtype, shape) at each layer. This would pinpoint where the values become degenerate. The logging should be gated behind `#[cfg(target_family = "wasm")]` and print to stderr (which shows in the Chrome console for WASM).

Specifically, add logging in:
1. `GatedDeltaNet::forward` — after conv1d, after SiLU, after split
2. `gated_delta_update` — beta, g_log values
3. `gated_delta_ops` — q_t, k_t, v_t at first timestep
4. `DecoderLayer::forward` — hidden state after attention + after MLP
5. `Qwen3_5Model::chat_sync` — logits before argmax
