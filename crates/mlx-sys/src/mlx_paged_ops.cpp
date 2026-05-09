// Phase 2 of the paged-attention compile integration.
//
// Implements the `PagedKVWrite` and `PagedAttention` MLX `Custom`
// primitives. Their `eval_gpu` paths now dispatch onto MLX's own
// `metal::CommandEncoder` via `crates/mlx-sys/src/mlx_paged_dispatch.cpp`
// (Phase 2). The Phase 1 extern-C shim into `crates/mlx-paged-attn/
// src/extern_c.rs` is no longer used by `eval_gpu`, but the shim
// itself is left in place because the existing smoke-test suite
// (`crates/mlx-paged-attn/tests/paged_ops_smoke.rs`) drives the Rust
// dispatcher directly and the broader Rust kernel-dispatch code is
// scheduled for removal in Phase 11.
//
// Because the dispatch is now on MLX's command queue, MLX's dependency
// tracking is correct: callers no longer need to `eval()` ancestor
// arrays before invoking `paged_kv_write` / `paged_attention`, nor
// `eval()` the outputs before reading them — the standard MLX evaluation
// order guarantees correctness.

#include "mlx_paged_ops.h"
#include "mlx_common.h"
#include "mlx_paged_dispatch.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/compile.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::fast {

namespace {

// Derive `x_pack` (the inner-axis vLLM K-pool packing factor) from the
// on-cache KV dtype. `x_pack` is `16 / sizeof(scalar)`:
//   - Fp16 / Bf16 (2 bytes) → x_pack = 8
//   - Fp8 (1 byte)          → x_pack = 16
// Used by the public factories to validate `k_pool.shape(2)` and
// `k_pool.shape(4)` against the dtype-derived layout.
int x_pack_for(KvDtype kv_dtype) {
  switch (kv_dtype) {
    case KvDtype::Fp16:
    case KvDtype::Bf16:
      return 8;
    case KvDtype::Fp8:
      return 16;
  }
  // Unreachable — quiet the warning.
  return 8;
}

// Map the `KvDtype` enum to the matching MLX `Dtype` for io tensors.
// Phase 1 contract (must match the Rust shim's
// `mlx_paged_attn_reshape_and_cache_dispatch` derivation in
// `crates/mlx-paged-attn/src/extern_c.rs`):
//   - non-FP8 cache → io dtype == cache dtype
//   - FP8 cache     → io dtype == bfloat16 (BF16-default for production;
//                     matches `MetalState::reshape_and_cache_kernel_name`'s
//                     `(bfloat16_t, uchar)` instantiation)
mlx::core::Dtype io_dtype_for_kv_dtype(KvDtype kv_dtype) {
  switch (kv_dtype) {
    case KvDtype::Fp16:
      return mlx::core::float16;
    case KvDtype::Bf16:
      return mlx::core::bfloat16;
    case KvDtype::Fp8:
      return mlx::core::bfloat16;
  }
  // Unreachable; quiet the warning.
  return mlx::core::bfloat16;
}

// Map the `KvDtype` enum to the matching MLX `Dtype` for the on-cache
// (K/V pool) storage. Phase 1 contract:
//   - Fp16 cache → float16
//   - Bf16 cache → bfloat16
//   - Fp8 cache  → uint8 (FP8 stored opaquely as bytes; the kernel
//                  reinterprets via `to_cache<KV_T, uchar>` below)
mlx::core::Dtype cache_dtype_for_kv_dtype(KvDtype kv_dtype) {
  switch (kv_dtype) {
    case KvDtype::Fp16:
      return mlx::core::float16;
    case KvDtype::Bf16:
      return mlx::core::bfloat16;
    case KvDtype::Fp8:
      return mlx::core::uint8;
  }
  // Unreachable; quiet the warning.
  return mlx::core::bfloat16;
}

// Translate the public `KvDtype` enum into the dispatch-internal
// `paged::KvDtype` so we can call `mlx::core::fast::paged::dispatch_*`.
// Phase 2: both enums have the same wire values; the cast just
// satisfies the type system.
mlx::core::fast::paged::KvDtype to_paged_dtype(KvDtype d) {
  switch (d) {
    case KvDtype::Fp16:
      return mlx::core::fast::paged::KvDtype::Fp16;
    case KvDtype::Bf16:
      return mlx::core::fast::paged::KvDtype::Bf16;
    case KvDtype::Fp8:
      return mlx::core::fast::paged::KvDtype::Fp8;
  }
  return mlx::core::fast::paged::KvDtype::Bf16;
}

// Reject non-row-contiguous or nonzero-offset views at the factory.
//
// Rationale (Phase 1 review round 8): the Metal kernels (and the Rust
// shim) read each input as a dense row-major buffer starting at the
// raw `MTLBuffer` pointer. For pool-side inputs (k_pool/v_pool the
// kernel writes into) the dispatch carries the BUFFER ONLY — no
// offset / strides — so a sliced or transposed view would silently
// alias to offset 0 in the backing allocation and either:
//   - clobber unrelated regions of the pool (writes), or
//   - read from the wrong start of a non-zero-offset slice (reads).
//
// For read-only inputs (q, new_k/new_v, block_table, seq_lens,
// slot_mapping) the q-side path forwards `q.offset()` and `out.offset()`
// to the dispatcher, but never k_pool/v_pool/block_table/seq_lens
// offsets, and never any strides. Different inputs receive different
// treatment, which is fragile. Phase-2 dispatch will eventually thread
// strides and offsets through; Phase 1's contract is simpler: refuse
// non-row-contiguous / nonzero-offset views and let the caller
// materialize a `mlx::core::contiguous(...)` copy explicitly.
//
// This check passes on:
//   - tracer arrays from `mlx::core::compile()` (default flags
//     `{true, true, true}`, default offset 0; see `array.h:495`)
//   - shape-only arrays from negative-test helpers
//     (`array(Shape{...}, dtype, nullptr, {})`)
//   - normal `eval()`-ed arrays before any slice/transpose op
//
// It fails on:
//   - results of `mlx::core::slice(...)` with nonzero start (offset != 0
//     and/or row_contiguous == false)
//   - results of `mlx::core::transpose(...)` (row_contiguous == false)
//   - any view returned by `array::copy_shared_buffer(...)` that
//     adjusted strides or set a non-default offset
void require_row_contiguous_zero_offset(
    const array& arr,
    const char* op_name,
    const char* input_name) {
  if (!arr.flags().row_contiguous) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] " << input_name
        << " must be row-contiguous; got non-row-contiguous view "
        << "(probably from a transpose / non-trivial slice). "
        << "Materialize a contiguous copy first via "
        << "`mlx::core::contiguous(arr)` before passing to the primitive.";
    throw std::invalid_argument(msg.str());
  }
  if (arr.offset() != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] " << input_name
        << " must have offset 0; got offset " << arr.offset()
        << " (probably from a sliced view starting partway into the "
        << "backing buffer). Materialize a contiguous copy first via "
        << "`mlx::core::contiguous(arr)` before passing to the primitive.";
    throw std::invalid_argument(msg.str());
  }
}

// =============================================================================
// Phase 1 review-round-10 finding: unify factory + eval_gpu validation.
//
// Rounds 3-9 each surfaced another factory-only check that
// `mlx::core::compile`'s cache-hit replay bypassed because the cache
// key only compares rank/shape/dtype — `compile_replace` rebuilds the
// cached primitive with real inputs WITHOUT re-running the factory.
// Symptoms ranged from sliced views silently aliasing the wrong region
// of memory to bad scalar state divide-by-zero in eval_gpu's own
// arithmetic.
//
// This helper closes the entire class of "factory-only check bypassed
// by compile cache" hazards in one swoop: every structural / scalar /
// shape / dtype / contiguity check lives in a single helper that the
// factory AND eval_gpu BOTH call. Future factory checks added to this
// helper are automatically mirrored on the eval_gpu side too.
//
// The helper does NOT perform runtime data-dependent bounds checks
// (slot_mapping max < pool capacity for PagedKVWrite; seq_lens /
// block_table content checks for PagedAttention) — those have a
// trace-vs-runtime asymmetry that requires materialized data and stays
// in the dedicated runtime check immediately following the helper call.
//
// `op_name` is prefixed onto every error message so the caller knows
// which path raised (e.g. "[paged_kv_write]" for the factory,
// "PagedKVWrite::eval_gpu:" for the eval path).
// =============================================================================

void validate_paged_kv_write_inputs(
    const array& k_pool,
    const array& v_pool,
    const array& new_k,
    const array& new_v,
    const array& slot_mapping,
    const array& k_scale,
    const array& v_scale,
    int block_size,
    int num_kv_heads,
    int head_size,
    int x_pack,
    KvDtype kv_dtype,
    const char* op_name) {
  // 1. Contiguity + zero offset on every input (round-8/9).
  require_row_contiguous_zero_offset(k_pool, op_name, "k_pool");
  require_row_contiguous_zero_offset(v_pool, op_name, "v_pool");
  require_row_contiguous_zero_offset(new_k, op_name, "new_k");
  require_row_contiguous_zero_offset(new_v, op_name, "new_v");
  require_row_contiguous_zero_offset(slot_mapping, op_name, "slot_mapping");
  require_row_contiguous_zero_offset(k_scale, op_name, "k_scale");
  require_row_contiguous_zero_offset(v_scale, op_name, "v_scale");

  // 2. Scalar-state positivity (round-7) + x_pack/dtype agreement
  //    (round-3 derivation). Run BEFORE shape arithmetic so a zero
  //    scalar can never reach divide-by-zero arithmetic below.
  if (num_kv_heads <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] num_kv_heads (" << num_kv_heads
        << ") must be > 0.";
    throw std::invalid_argument(msg.str());
  }
  if (head_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] head_size (" << head_size
        << ") must be > 0.";
    throw std::invalid_argument(msg.str());
  }
  if (block_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_size (" << block_size
        << ") must be > 0.";
    throw std::invalid_argument(msg.str());
  }
  if (x_pack <= 0 || head_size % x_pack != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] x_pack (" << x_pack
        << ") must be positive and divide head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  // Caller's `x_pack` must agree with the dtype-derived value
  // (`16 / sizeof(kv_dtype)`). A mismatch means caller and dispatcher
  // disagree on the K-pool layout — guaranteed garbage on read.
  {
    int x_pack_expected = x_pack_for(kv_dtype);
    if (x_pack != x_pack_expected) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] x_pack (" << x_pack
          << ") disagrees with the dtype-derived x_pack ("
          << x_pack_expected << ") for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
  }

  // 3. Dtype-vs-kv_dtype validation (round-4). Fires BEFORE the
  //    pairwise k_pool==v_pool / new_k==new_v checks so each input
  //    slot has its own dedicated rejection path.
  {
    auto expected_cache_dtype = cache_dtype_for_kv_dtype(kv_dtype);
    if (k_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] k_pool dtype " << k_pool.dtype()
          << " disagrees with the expected cache dtype "
          << expected_cache_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
    if (v_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] v_pool dtype " << v_pool.dtype()
          << " disagrees with the expected cache dtype "
          << expected_cache_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
  }
  {
    auto expected_io_dtype = io_dtype_for_kv_dtype(kv_dtype);
    if (new_k.dtype() != expected_io_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] new_k dtype " << new_k.dtype()
          << " disagrees with the expected io dtype "
          << expected_io_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype)
          << " (FP8 io dtype is bfloat16 per Phase 1 contract)";
      throw std::invalid_argument(msg.str());
    }
    if (new_v.dtype() != expected_io_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] new_v dtype " << new_v.dtype()
          << " disagrees with the expected io dtype "
          << expected_io_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype)
          << " (FP8 io dtype is bfloat16 per Phase 1 contract)";
      throw std::invalid_argument(msg.str());
    }
  }
  // Defense-in-depth pairwise checks (tripwire for future kv_dtype additions).
  if (k_pool.dtype() != v_pool.dtype()) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool dtype " << k_pool.dtype()
        << " disagrees with v_pool dtype " << v_pool.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (new_k.dtype() != new_v.dtype()) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k dtype " << new_k.dtype()
        << " disagrees with new_v dtype " << new_v.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (k_scale.dtype() != mlx::core::float32 ||
      v_scale.dtype() != mlx::core::float32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_scale / v_scale must be float32";
    throw std::invalid_argument(msg.str());
  }
  if (k_scale.size() != 1 || v_scale.size() != 1) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_scale / v_scale must be 1-element arrays";
    throw std::invalid_argument(msg.str());
  }

  // 4. Pool shape validation. K layout (vLLM):
  //      [num_blocks, num_kv_heads, head_size/x, block_size, x]
  //    V layout: [num_blocks, num_kv_heads, head_size, block_size]
  if (k_pool.ndim() != 5) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool must be rank 5 "
        << "[num_blocks, num_kv_heads, head_size/x, block_size, x]; got rank "
        << k_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.ndim() != 4) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool must be rank 4 "
        << "[num_blocks, num_kv_heads, head_size, block_size]; got rank "
        << v_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(1) (" << k_pool.shape(1)
        << ") disagrees with num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(2) != head_size / x_pack) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(2) (" << k_pool.shape(2)
        << ") disagrees with head_size/x_pack (" << head_size / x_pack << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(3) != block_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(3) (" << k_pool.shape(3)
        << ") disagrees with block_size (" << block_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(4) != x_pack) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(4) (" << k_pool.shape(4)
        << ") disagrees with x_pack (" << x_pack << ")";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool.shape(1) (" << v_pool.shape(1)
        << ") disagrees with num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool.shape(2) (" << v_pool.shape(2)
        << ") disagrees with head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(3) != block_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool.shape(3) (" << v_pool.shape(3)
        << ") disagrees with block_size (" << block_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(0) != v_pool.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool num_blocks (" << k_pool.shape(0)
        << ") disagrees with v_pool num_blocks (" << v_pool.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }

  // 5. new_k / new_v rank + per-dim agreement.
  if (new_k.ndim() != 3) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k must be rank 3 "
        << "[num_tokens, num_kv_heads, head_size]; got rank " << new_k.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (new_v.ndim() != 3) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_v must be rank 3 "
        << "[num_tokens, num_kv_heads, head_size]; got rank " << new_v.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (new_k.shape(1) != num_kv_heads || new_v.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k/new_v shape(1) ("
        << new_k.shape(1) << "/" << new_v.shape(1)
        << ") must equal num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (new_k.shape(2) != head_size || new_v.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k/new_v shape(2) ("
        << new_k.shape(2) << "/" << new_v.shape(2)
        << ") must equal head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (new_k.shape(0) != new_v.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k tokens (" << new_k.shape(0)
        << ") disagrees with new_v tokens (" << new_v.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }

  // 6. slot_mapping rank/dtype/length.
  if (slot_mapping.ndim() != 1) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] slot_mapping must be rank 1 [num_tokens]; "
        << "got rank " << slot_mapping.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (slot_mapping.dtype() != mlx::core::int64) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] slot_mapping must be int64 (kernel reads it as "
        << "`int64_t*`); got dtype " << slot_mapping.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (slot_mapping.shape(0) != new_k.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] slot_mapping length ("
        << slot_mapping.shape(0) << ") disagrees with new_k tokens ("
        << new_k.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
}

// Sister helper for `paged_attention(...)`. Same defense-in-depth
// rationale as `validate_paged_kv_write_inputs`: every structural
// check lives in one place and runs on BOTH factory and eval_gpu
// paths so compile-cache replay can never bypass any of them.
//
// `paged_attention` does not take an `x_pack` parameter — it derives
// from `kv_dtype` via `x_pack_for(...)`. We compute it locally and
// validate it the same way as the write-side helper.
void validate_paged_attention_inputs(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& k_scale,
    const array& v_scale,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int sliding_window,
    KvDtype kv_dtype,
    const char* op_name) {
  // 1. Contiguity + zero offset.
  require_row_contiguous_zero_offset(q, op_name, "q");
  require_row_contiguous_zero_offset(k_pool, op_name, "k_pool");
  require_row_contiguous_zero_offset(v_pool, op_name, "v_pool");
  require_row_contiguous_zero_offset(block_table, op_name, "block_table");
  require_row_contiguous_zero_offset(seq_lens, op_name, "seq_lens");
  require_row_contiguous_zero_offset(k_scale, op_name, "k_scale");
  require_row_contiguous_zero_offset(v_scale, op_name, "v_scale");

  // 2. Phase 7 lifts the Phase 1 sliding_window=0 restriction. Negative
  // values are still nonsensical (the only "no mask" sentinel is 0) so
  // reject them up-front; the Metal kernel masks K positions older than
  // `context_len - sliding_window` when sliding_window > 0.
  if (sliding_window < 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] sliding_window (" << sliding_window
        << ") must be >= 0 (use 0 to disable the sliding mask).";
    throw std::invalid_argument(msg.str());
  }

  // 3. Scalar-state positivity + GQA divisibility (rounds 6 + 7).
  if (num_kv_heads <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] num_kv_heads (" << num_kv_heads
        << ") must be > 0; the Metal kernel computes "
        << "num_queries_per_kv = num_heads / num_kv_heads, which would "
        << "divide by zero.";
    throw std::invalid_argument(msg.str());
  }
  if (num_q_heads <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] num_q_heads (" << num_q_heads
        << ") must be > 0.";
    throw std::invalid_argument(msg.str());
  }
  if (num_q_heads % num_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] GQA grouping invalid: num_q_heads="
        << num_q_heads << " must be divisible by num_kv_heads="
        << num_kv_heads << " (got remainder "
        << (num_q_heads % num_kv_heads) << "). When num_q_heads < "
        << "num_kv_heads the kernel divides by zero; when "
        << "num_q_heads % num_kv_heads != 0 later heads index past "
        << "the KV-head dimension and read out-of-pool memory.";
    throw std::invalid_argument(msg.str());
  }
  if (block_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_size (" << block_size
        << ") must be > 0; eval_gpu's bounds check divides by it "
        << "(`(s + block_size - 1) / block_size`) and the Metal kernel "
        << "uses it as a grid extent.";
    throw std::invalid_argument(msg.str());
  }
  if (head_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] head_size (" << head_size
        << ") must be > 0; the Metal kernel uses it as a grid extent "
        << "and indexing stride.";
    throw std::invalid_argument(msg.str());
  }

  // Derive x_pack from kv_dtype and verify head_size divisibility.
  // Round-7 / round-9 hazard: a future divergence between the dtype
  // and the structural pool check (k_pool.shape(2)/(4)) would otherwise
  // surface as a kernel buffer-size mismatch.
  int x_pack_expected = x_pack_for(kv_dtype);
  if (x_pack_expected <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] dtype-derived x_pack (" << x_pack_expected
        << ") must be > 0 (kv_dtype " << static_cast<int>(kv_dtype) << ").";
    throw std::invalid_argument(msg.str());
  }
  if (head_size % x_pack_expected != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] head_size (" << head_size
        << ") must be divisible by dtype-derived x_pack ("
        << x_pack_expected << ") for kv_dtype "
        << static_cast<int>(kv_dtype);
    throw std::invalid_argument(msg.str());
  }

  // 4. Scale dtype + 1-element shape.
  if (k_scale.dtype() != mlx::core::float32 ||
      v_scale.dtype() != mlx::core::float32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_scale / v_scale must be float32";
    throw std::invalid_argument(msg.str());
  }
  if (k_scale.size() != 1 || v_scale.size() != 1) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_scale / v_scale must be 1-element arrays";
    throw std::invalid_argument(msg.str());
  }

  // 5. q rank + dtype + per-dim shape.
  if (q.ndim() != 3) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] q must be rank 3 "
        << "[num_seqs, num_q_heads, head_size]; got rank " << q.ndim();
    throw std::invalid_argument(msg.str());
  }
  {
    auto expected_io_dtype = io_dtype_for_kv_dtype(kv_dtype);
    if (q.dtype() != expected_io_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] q dtype " << q.dtype()
          << " disagrees with the expected io dtype "
          << expected_io_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype)
          << " (FP8 io dtype is bfloat16 per Phase 1 contract)";
      throw std::invalid_argument(msg.str());
    }
    auto expected_cache_dtype = cache_dtype_for_kv_dtype(kv_dtype);
    if (k_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] k_pool dtype " << k_pool.dtype()
          << " disagrees with the expected cache dtype "
          << expected_cache_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
    if (v_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] v_pool dtype " << v_pool.dtype()
          << " disagrees with the expected cache dtype "
          << expected_cache_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
  }
  if (q.shape(1) != num_q_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] q.shape(1) (" << q.shape(1)
        << ") disagrees with num_q_heads (" << num_q_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (q.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] q.shape(2) (" << q.shape(2)
        << ") disagrees with head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }

  // 6. Pool rank/shape validation.
  if (k_pool.ndim() != 5) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool must be rank 5 "
        << "[num_blocks, num_kv_heads, head_size/x, block_size, x]; got rank "
        << k_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.ndim() != 4) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool must be rank 4 "
        << "[num_blocks, num_kv_heads, head_size, block_size]; got rank "
        << v_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(0) != v_pool.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool num_blocks (" << k_pool.shape(0)
        << ") disagrees with v_pool num_blocks (" << v_pool.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(1) != num_kv_heads || v_pool.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool/v_pool num_kv_heads ("
        << k_pool.shape(1) << "/" << v_pool.shape(1)
        << ") disagrees with num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(3) != block_size || v_pool.shape(3) != block_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool/v_pool block_size ("
        << k_pool.shape(3) << "/" << v_pool.shape(3)
        << ") disagrees with block_size (" << block_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool.shape(2) (" << v_pool.shape(2)
        << ") disagrees with head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(2) != head_size / x_pack_expected) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(2) (" << k_pool.shape(2)
        << ") disagrees with head_size/x_pack ("
        << head_size / x_pack_expected
        << ") (head_size=" << head_size << ", x_pack=" << x_pack_expected
        << " for kv_dtype " << static_cast<int>(kv_dtype) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(4) != x_pack_expected) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(4) (" << k_pool.shape(4)
        << ") disagrees with dtype-derived x_pack (" << x_pack_expected
        << ") for kv_dtype " << static_cast<int>(kv_dtype);
    throw std::invalid_argument(msg.str());
  }

  // 7. block_table rank / shape / dtype.
  if (block_table.ndim() != 2) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_table must be rank 2 "
        << "[num_seqs, max_blocks_per_seq]; got rank " << block_table.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (block_table.shape(0) != q.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_table.shape(0) ("
        << block_table.shape(0) << ") disagrees with q num_seqs ("
        << q.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (block_table.dtype() != mlx::core::int32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_table dtype must be int32 (kernel "
        << "reads it as 32-bit indices); got dtype " << block_table.dtype();
    throw std::invalid_argument(msg.str());
  }

  // 8. seq_lens rank / shape / dtype.
  if (seq_lens.ndim() != 1) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] seq_lens must be rank 1 [num_seqs]; got rank "
        << seq_lens.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (seq_lens.shape(0) != q.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] seq_lens.shape(0) (" << seq_lens.shape(0)
        << ") disagrees with q num_seqs (" << q.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (seq_lens.dtype() != mlx::core::int32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] seq_lens dtype must be int32 (kernel reads "
        << "it as 32-bit lengths); got dtype " << seq_lens.dtype();
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

// =============================================================================
// PagedKVWrite implementation
// =============================================================================

void PagedKVWrite::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 7) {
    throw std::runtime_error(
        "PagedKVWrite: expected 7 inputs (k_pool, v_pool, new_k, new_v, "
        "slot_mapping, k_scale, v_scale)");
  }
  if (outputs.size() != 2) {
    throw std::runtime_error(
        "PagedKVWrite: expected 2 outputs (k_pool', v_pool')");
  }

  const array& k_pool = inputs[0];
  const array& v_pool = inputs[1];
  const array& new_k = inputs[2];
  const array& new_v = inputs[3];
  const array& slot_mapping = inputs[4];
  const array& k_scale = inputs[5];
  const array& v_scale = inputs[6];

  // Phase 1 review-round-10 contract: every structural / scalar /
  // shape / dtype / contiguity check lives in
  // `validate_paged_kv_write_inputs`, which the public
  // `paged_kv_write(...)` factory ALSO calls. Running it here closes
  // the entire class of "factory-only check bypassed by compile cache"
  // hazards (rounds 3-9) in a single call: `mlx::core::compile`'s
  // cached re-traces rebuild the primitive with real inputs via
  // `compile_replace` WITHOUT re-running the factory, so any check
  // that lives only in the factory is silently skipped by the cached
  // path. The helper uses the primitive's stored scalar state — if a
  // future code path ever constructs the primitive directly with
  // mismatched scalars, the eval_gpu validation rejects before the
  // dispatch path can touch the buffers.
  validate_paged_kv_write_inputs(
      k_pool,
      v_pool,
      new_k,
      new_v,
      slot_mapping,
      k_scale,
      v_scale,
      block_size_,
      num_kv_heads_,
      head_size_,
      x_pack_,
      kv_dtype_,
      "PagedKVWrite::eval_gpu");

  // Output arrays semantically alias the input pools (in-place write).
  // `copy_shared_buffer` makes the output `array` point at the same
  // backing buffer + offset / strides as the input pool. We do this
  // BEFORE the dispatch so the encoder's `set_output_array` call sees
  // the right buffer (the input pool's, shared into the output).
  outputs[0].copy_shared_buffer(k_pool);
  outputs[1].copy_shared_buffer(v_pool);

  // Determine num_tokens from new_k's leading dimension. The kernel
  // expects shape [num_tokens, num_kv_heads, head_size]. The validator
  // already enforced rank 3.
  int num_tokens = static_cast<int>(new_k.shape(0));

  // Slot-id bounds check (Phase 2 safety; runs on EVERY runtime call,
  // including the compile-cached path). Stays out of the validator
  // because it requires materialized data that tracer arrays do not
  // have. The Metal kernel does NOT bounds-check `slot_idx`; a value
  // `>= num_blocks * block_size` writes past the K/V pool. MLX
  // evaluates all primitive inputs before invoking `eval_gpu`, so
  // `slot_mapping.data<int64_t>()` is materialized and safe to read
  // host-side.
  if (slot_mapping.shape(0) > 0) {
    const int32_t num_blocks_runtime = static_cast<int32_t>(k_pool.shape(0));
    const int64_t pool_capacity = static_cast<int64_t>(num_blocks_runtime) *
        static_cast<int64_t>(block_size_);
    const int64_t* slot_data = slot_mapping.data<int64_t>();
    int64_t max_slot = -1;
    for (size_t i = 0; i < slot_mapping.size(); ++i) {
      // Negative slot ids are sentinels for "skip this token"; the
      // kernel handles them with an early `return`. Don't include them
      // in the max — only valid (non-negative) ids must stay in range.
      if (slot_data[i] >= 0 && slot_data[i] > max_slot) {
        max_slot = slot_data[i];
      }
    }
    if (max_slot >= pool_capacity) {
      std::ostringstream msg;
      msg << "[runtime] PagedKVWrite::eval_gpu: slot_mapping max ("
          << max_slot << ") exceeds pool capacity (num_blocks="
          << num_blocks_runtime << " * block_size=" << block_size_ << " = "
          << pool_capacity
          << "). The kernel does not bounds-check slot_idx; an out-of-range "
          << "slot would write past the K/V pool. This check fires on every "
          << "runtime call (including the compile-cached path).";
      throw std::invalid_argument(msg.str());
    }
  }

  // Phase 2: dispatch onto MLX's command encoder. MLX's dependency
  // tracking handles ordering against any preceding/following ops.
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  auto& compute_encoder = mlx::core::metal::get_command_encoder(s);

  mlx::core::fast::paged::dispatch_reshape_and_cache(
      compute_encoder,
      d,
      new_k,
      new_v,
      outputs[0],
      outputs[1],
      slot_mapping,
      k_scale,
      v_scale,
      num_tokens,
      num_kv_heads_,
      head_size_,
      block_size_,
      x_pack_,
      to_paged_dtype(kv_dtype_));
}

std::vector<array> PagedKVWrite::vjp(
    const std::vector<array>& /*primals*/,
    const std::vector<array>& /*cotangents*/,
    const std::vector<int>& /*argnums*/,
    const std::vector<array>& /*outputs*/) {
  throw std::runtime_error("PagedKVWrite is inference-only");
}

std::vector<Shape> PagedKVWrite::output_shapes(
    const std::vector<array>& inputs) {
  if (inputs.size() < 2) {
    throw std::runtime_error(
        "PagedKVWrite::output_shapes: expected at least 2 inputs");
  }
  return {inputs[0].shape(), inputs[1].shape()};
}

bool PagedKVWrite::is_equivalent(const Primitive& other) const {
  const PagedKVWrite& o = static_cast<const PagedKVWrite&>(other);
  return block_size_ == o.block_size_ && num_kv_heads_ == o.num_kv_heads_ &&
      head_size_ == o.head_size_ && x_pack_ == o.x_pack_ &&
      kv_dtype_ == o.kv_dtype_;
}

// =============================================================================
// PagedAttention implementation
// =============================================================================

void PagedAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 7) {
    throw std::runtime_error(
        "PagedAttention: expected 7 inputs (q, k_pool, v_pool, block_table, "
        "seq_lens, k_scale, v_scale)");
  }
  if (outputs.size() != 1) {
    throw std::runtime_error("PagedAttention: expected 1 output");
  }

  const array& q = inputs[0];
  const array& k_pool = inputs[1];
  const array& v_pool = inputs[2];
  const array& block_table = inputs[3];
  const array& seq_lens = inputs[4];
  const array& k_scale = inputs[5];
  const array& v_scale = inputs[6];

  array& out = outputs[0];

  // Phase 1 review-round-10 contract: every structural / scalar /
  // shape / dtype / contiguity check lives in
  // `validate_paged_attention_inputs`, which the public
  // `paged_attention(...)` factory ALSO calls. Running it here closes
  // the entire class of "factory-only check bypassed by compile cache"
  // hazards (rounds 3-9) in a single call: `mlx::core::compile`'s
  // cached re-traces rebuild the primitive with real inputs via
  // `compile_replace` WITHOUT re-running the factory, so any check
  // that lives only in the factory is silently skipped by the cached
  // path. The helper uses the primitive's stored scalar state — if a
  // future code path ever constructs the primitive directly with
  // mismatched scalars, the eval_gpu validation rejects before the
  // dispatch path can touch the buffers. Run BEFORE any host reads
  // (seq_lens.data<int32_t>(), block_table.data<int32_t>() etc.) and
  // BEFORE the encoder dispatch so the throw fires before we touch
  // the buffers.
  validate_paged_attention_inputs(
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      k_scale,
      v_scale,
      block_size_,
      num_q_heads_,
      num_kv_heads_,
      head_size_,
      sliding_window_,
      kv_dtype_,
      "PagedAttention::eval_gpu");

  uint32_t num_seqs = static_cast<uint32_t>(q.shape(0));
  uint32_t max_blocks_per_seq = static_cast<uint32_t>(block_table.shape(1));

  // The validator already enforced seq_lens/block_table dtype == int32.
  // MLX evaluates all primitive inputs before invoking `eval_gpu`, so
  // `seq_lens.data<int32_t>()` is materialized and safe to read host-side.
  // Calling `mlx::core::eval(...)` here would recurse into the scheduler
  // and deadlock.
  const int32_t* seq_lens_data = seq_lens.data<int32_t>();

  // Per-runtime bounds check on `seq_lens` and `block_table` contents
  // (Phase 1 safety; runs on EVERY runtime call, including the
  // compile-cached path).
  //
  // The Metal kernel does NOT bounds-check either input. For each
  // sequence the kernel:
  //   1. reads `context_lens[seq_idx]` as `uint32_t` (so a negative
  //      int32 sneaks in as a huge unsigned context length),
  //   2. derives `num_context_blocks = ceil(context_len / block_size)`,
  //      and reads `block_tables[seq_idx * max_blocks_per_seq + j]`
  //      for each j in [0, num_context_blocks),
  //   3. uses that value directly as `physical_block_number *
  //      kv_block_stride` for K/V pool reads.
  //
  // Without per-call validation, a too-large `seq_lens[i]` reads past
  // that row's block-table region, and a negative or `>= num_blocks`
  // entry in `block_table` addresses outside the K/V pool. Mirror the
  // pattern used immediately above for `PagedKVWrite::eval_gpu`'s
  // slot-mapping check: the factory's structural validation only
  // covers shape/dtype, and `mlx::core::compile`'s cached re-traces
  // bypass the factory entirely, so the runtime-content check must
  // live in `eval_gpu`.
  //
  // Cost: O(num_seqs * max_blocks_per_seq) host-side reads per call.
  // Acceptable for Phase 1 correctness; Phase 2 may push this
  // kernel-side or gate it on a configured trust mode.
  //
  // Run BEFORE the max-context computation below so a malformed input
  // surfaces the descriptive `std::invalid_argument` error instead of
  // the coarser "max_context_len > 0" runtime_error.
  if (num_seqs > 0 && max_blocks_per_seq > 0) {
    const int32_t num_blocks = static_cast<int32_t>(k_pool.shape(0));
    const int64_t max_seq_len_bound =
        static_cast<int64_t>(max_blocks_per_seq) *
        static_cast<int64_t>(block_size_);
    const int32_t* block_table_data = block_table.data<int32_t>();
    for (uint32_t i = 0; i < num_seqs; ++i) {
      const int32_t s = seq_lens_data[i];
      if (s < 0 || static_cast<int64_t>(s) > max_seq_len_bound) {
        std::ostringstream msg;
        msg << "[runtime] PagedAttention::eval_gpu: seq_lens[" << i << "] ("
            << s << ") out of range [0, max_blocks_per_seq * block_size = "
            << max_blocks_per_seq << " * " << block_size_ << " = "
            << max_seq_len_bound
            << "]. The kernel reads context_lens[seq_idx] as uint32_t and "
            << "derives num_context_blocks from it; an out-of-range value "
            << "either reads past the row's block-table region (positive "
            << "overflow) or is interpreted as a huge unsigned value "
            << "(negative). This check fires on every runtime call "
            << "(including the compile-cached path).";
        throw std::invalid_argument(msg.str());
      }
      // ceil(s / block_size_); s >= 0 so integer division is fine.
      const int32_t num_used_blocks =
          (s + block_size_ - 1) / block_size_;
      const size_t row_offset =
          static_cast<size_t>(i) * static_cast<size_t>(max_blocks_per_seq);
      for (int32_t j = 0; j < num_used_blocks; ++j) {
        const int32_t blk = block_table_data[row_offset + static_cast<size_t>(j)];
        if (blk < 0 || blk >= num_blocks) {
          std::ostringstream msg;
          msg << "[runtime] PagedAttention::eval_gpu: block_table[" << i
              << ", " << j << "] (" << blk
              << ") out of range [0, num_blocks = " << num_blocks
              << "). The kernel uses block_table entries directly as "
              << "physical_block_number * kv_block_stride for K/V reads; "
              << "a value outside [0, num_blocks) addresses arbitrary GPU "
              << "memory. This check fires on every runtime call "
              << "(including the compile-cached path).";
          throw std::invalid_argument(msg.str());
        }
      }
    }
  }

  // Determine max_context_len from seq_lens (max element). This is
  // the value that drives the V1/V2 branch. By the time we get here
  // every entry has already been validated as `>= 0` above.
  int32_t max_context_len = 0;
  for (size_t i = 0; i < seq_lens.size(); ++i) {
    if (seq_lens_data[i] > max_context_len) {
      max_context_len = seq_lens_data[i];
    }
  }
  if (max_context_len <= 0) {
    throw std::runtime_error(
        "PagedAttention: max_context_len from seq_lens must be > 0");
  }

  // Allocate the output buffer via MLX's allocator BEFORE dispatch —
  // the encoder's `set_output_array` reads `out.buffer().ptr()` so
  // the buffer must exist by the time the dispatch runs.
  out.set_data(allocator::malloc(out.nbytes()));

  // Phase 2: dispatch onto MLX's command encoder. MLX's dependency
  // tracking handles ordering against any preceding/following ops,
  // so callers no longer need to `eval()` ancestors before invoking
  // `paged_attention` nor `eval()` the output before reading.
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  auto& compute_encoder = mlx::core::metal::get_command_encoder(s);

  mlx::core::fast::paged::dispatch_paged_attention_auto(
      compute_encoder,
      d,
      s,
      out,
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      k_scale,
      v_scale,
      static_cast<int>(num_seqs),
      num_q_heads_,
      num_kv_heads_,
      head_size_,
      block_size_,
      max_context_len,
      static_cast<int>(max_blocks_per_seq),
      scale_,
      softcap_,
      sliding_window_,
      to_paged_dtype(kv_dtype_));
}

std::vector<array> PagedAttention::vjp(
    const std::vector<array>& /*primals*/,
    const std::vector<array>& /*cotangents*/,
    const std::vector<int>& /*argnums*/,
    const std::vector<array>& /*outputs*/) {
  throw std::runtime_error("PagedAttention is inference-only");
}

std::vector<Shape> PagedAttention::output_shapes(
    const std::vector<array>& inputs) {
  if (inputs.empty()) {
    throw std::runtime_error("PagedAttention::output_shapes: empty inputs");
  }
  const auto& q = inputs[0];
  if (q.ndim() != 3) {
    throw std::runtime_error(
        "PagedAttention::output_shapes: q must be rank 3");
  }
  // Spec: output shape is {q_num_tokens, num_q_heads, head_size} from
  // scalar state. We DO NOT echo q.shape() — if q's trailing dims
  // disagree with our scalar state, we'd allocate a buffer of the
  // wrong size, which the kernel would then under- or over-write.
  // Validation against q.shape() lives in `eval_gpu` and the public
  // `paged_attention` factory; this method just reports the
  // shape MLX should allocate.
  return {Shape{q.shape(0), num_q_heads_, head_size_}};
}

bool PagedAttention::is_equivalent(const Primitive& other) const {
  const PagedAttention& o = static_cast<const PagedAttention&>(other);
  return scale_ == o.scale_ && softcap_ == o.softcap_ &&
      block_size_ == o.block_size_ && num_q_heads_ == o.num_q_heads_ &&
      num_kv_heads_ == o.num_kv_heads_ && head_size_ == o.head_size_ &&
      sliding_window_ == o.sliding_window_ && kv_dtype_ == o.kv_dtype_;
}

// =============================================================================
// Public free functions
// =============================================================================

std::pair<array, array> paged_kv_write(
    const array& k_pool,
    const array& v_pool,
    const array& new_k,
    const array& new_v,
    const array& slot_mapping,
    const array& k_scale,
    const array& v_scale,
    int block_size,
    int num_kv_heads,
    int head_size,
    int x_pack,
    KvDtype kv_dtype,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  // Fallback: Phase 1 has no pure-MLX implementation of paged write
  // (the kernel is the implementation). The fallback is only invoked
  // by VJP/JVP transformations, which we throw on. Provide a stub
  // that raises so unintended fallback paths surface immediately.
  auto fallback = [](std::vector<array> /*inputs*/) -> std::vector<array> {
    throw std::runtime_error(
        "paged_kv_write has no fallback implementation (inference-only)");
  };

  // Phase 1 review-round-10 contract: structural / scalar / shape /
  // dtype / contiguity checks live in the shared helper that
  // `PagedKVWrite::eval_gpu` ALSO calls. A factory-only check is
  // bypassed by `mlx::core::compile`'s cached re-traces (which
  // rebuild the cached primitive with real inputs via
  // `compile_replace` WITHOUT re-running the factory), so the helper
  // is the single source of truth for both paths.
  validate_paged_kv_write_inputs(
      k_pool,
      v_pool,
      new_k,
      new_v,
      slot_mapping,
      k_scale,
      v_scale,
      block_size,
      num_kv_heads,
      head_size,
      x_pack,
      kv_dtype,
      "paged_kv_write");

  // Slot-id range check (Phase 1 safety; eval-based, factory-only).
  //
  // The Metal kernel does NOT bounds-check `slot_idx`; a value
  // `>= num_blocks * block_size` writes past the K/V pool. Phase 2
  // will move this check kernel-side using a constant uniform.
  //
  // Phase 1 contract:
  //   - When called with concrete data (most production callers), the
  //     factory evals `max(slot_mapping)` here and throws on overflow.
  //   - When called inside `mlx::core::compile`'s trace
  //     (`mlx::core::detail::in_tracing()` is true), we SKIP the eval
  //     because tracer arrays have no backing data and `eval()` would
  //     fail. The compile-path caller is responsible for a separate
  //     bounds check on the runtime slot_mapping before invoking the
  //     compiled lambda. This matches the documented Phase 1 contract:
  //     callers `eval()` inputs before invoking the primitive.
  //
  // The eval is a one-shot host read of an int64 reduction — small
  // enough to be acceptable next to the Metal dispatch cost.
  if (!mlx::core::detail::in_tracing() && slot_mapping.shape(0) > 0) {
    array max_slot = mlx::core::max(slot_mapping);
    mlx::core::eval(max_slot);
    int64_t max_slot_v = max_slot.item<int64_t>();
    int64_t pool_capacity =
        static_cast<int64_t>(k_pool.shape(0)) * static_cast<int64_t>(block_size);
    if (max_slot_v >= pool_capacity) {
      // Runtime data-dependent guard: `[runtime]` prefix matches the
      // `[runtime] PagedKVWrite::eval_gpu` marker used by the same
      // bounds check inside `PagedKVWrite::eval_gpu`. Tagging both
      // throw sites with the same `[runtime]` prefix keeps slot_mapping
      // bounds errors uniformly distinguishable from `[validator]`
      // structural rejections, regardless of which path (factory vs.
      // compile-cached eval_gpu) caught the bad data.
      std::ostringstream msg;
      msg << "[runtime] [paged_kv_write] slot_mapping max (" << max_slot_v
          << ") exceeds pool capacity (num_blocks=" << k_pool.shape(0)
          << " * block_size=" << block_size << " = " << pool_capacity << ")";
      throw std::invalid_argument(msg.str());
    }
  }

  std::vector<array> inputs = {
      k_pool, v_pool, new_k, new_v, slot_mapping, k_scale, v_scale};

  auto primitive = std::make_shared<PagedKVWrite>(
      s,
      std::move(fallback),
      block_size,
      num_kv_heads,
      head_size,
      x_pack,
      kv_dtype);

  auto results = array::make_arrays(
      {k_pool.shape(), v_pool.shape()},
      {k_pool.dtype(), v_pool.dtype()},
      primitive,
      inputs);

  return {std::move(results[0]), std::move(results[1])};
}

array paged_attention(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& k_scale,
    const array& v_scale,
    float scale,
    float softcap,
    int sliding_window,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    KvDtype kv_dtype,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  auto fallback = [](std::vector<array> /*inputs*/) -> std::vector<array> {
    throw std::runtime_error(
        "paged_attention has no fallback implementation (inference-only)");
  };

  // Phase 1 review-round-10 contract: structural / scalar / shape /
  // dtype / contiguity checks live in the shared helper that
  // `PagedAttention::eval_gpu` ALSO calls. A factory-only check is
  // bypassed by `mlx::core::compile`'s cached re-traces (which
  // rebuild the cached primitive with real inputs via
  // `compile_replace` WITHOUT re-running the factory), so the helper
  // is the single source of truth for both paths.
  validate_paged_attention_inputs(
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      k_scale,
      v_scale,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window,
      kv_dtype,
      "paged_attention");

  // Output shape and dtype
  auto out_dtype = io_dtype_for_kv_dtype(kv_dtype);
  // Spec: output shape is {q.shape(0), num_q_heads, head_size} from
  // scalar state. Even though we just verified q.shape(1)/(2) agree
  // with state, we use state explicitly so this matches
  // PagedAttention::output_shapes — they MUST report the same shape
  // or MLX will allocate a buffer of the wrong size during compile
  // replay.
  Shape out_shape = {q.shape(0), num_q_heads, head_size};

  std::vector<array> inputs = {
      q, k_pool, v_pool, block_table, seq_lens, k_scale, v_scale};

  auto primitive = std::make_shared<PagedAttention>(
      s,
      std::move(fallback),
      scale,
      softcap,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window,
      kv_dtype);

  return array(std::move(out_shape), out_dtype, primitive, std::move(inputs));
}

} // namespace mlx::core::fast

extern "C" {

/// Production FFI: emit a `PagedAttention` MLX Custom primitive and return
/// the resulting on-device array. This is intentionally a tiny bridge over
/// the C++ factory so Rust model code can use the same MLX-graph paged
/// attention primitive as the compiled Qwen path without copying attention
/// outputs through host memory.
///
/// Returns nullptr on bridge/factory validation errors; Rust callers keep the
/// existing read_kv_range + SDPA path as fallback. The returned array is still
/// lazy, so GPU dispatch errors surface later when MLX evaluates the graph.
mlx_array* mlx_paged_attention_forward(
    mlx_array* q_ptr,
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* block_table_ptr,
    mlx_array* seq_lens_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    float scale,
    float softcap,
    int sliding_window,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    uint8_t kv_dtype_raw) {
  if (!q_ptr || !k_pool_ptr || !v_pool_ptr || !block_table_ptr ||
      !seq_lens_ptr || !k_scale_ptr || !v_scale_ptr) {
    return nullptr;
  }

  try {
    using namespace mlx::core;
    using namespace mlx::core::fast;

    auto& q_raw           = *reinterpret_cast<array*>(q_ptr);
    auto& k_pool          = *reinterpret_cast<array*>(k_pool_ptr);
    auto& v_pool          = *reinterpret_cast<array*>(v_pool_ptr);
    auto& block_table_raw = *reinterpret_cast<array*>(block_table_ptr);
    auto& seq_lens_raw    = *reinterpret_cast<array*>(seq_lens_ptr);
    auto& k_scale         = *reinterpret_cast<array*>(k_scale_ptr);
    auto& v_scale         = *reinterpret_cast<array*>(v_scale_ptr);

    auto kv_dtype = static_cast<KvDtype>(kv_dtype_raw);

    // The public factory rejects non-row-contiguous/nonzero-offset inputs.
    // `q` is often produced by reshape/slice/rope chains in Rust model code,
    // so materialize q explicitly at this bridge.
    //
    // Metadata is different: `PagedAttention::eval_gpu` performs host-side
    // content checks on block_table/seq_lens before dispatch. Wrapping those
    // arrays in lazy `contiguous(...)` nodes can make the validator read lazy
    // metadata copies rather than the adapter-owned materialized arrays. Keep
    // the metadata contract strict here: Rust owns materialized, row-contiguous,
    // zero-offset metadata arrays, and the factory validates that contract.
    auto q = contiguous(q_raw);

    auto out = paged_attention(
        q,
        k_pool,
        v_pool,
        block_table_raw,
        seq_lens_raw,
        k_scale,
        v_scale,
        scale,
        softcap,
        sliding_window,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_size,
        kv_dtype);

    return reinterpret_cast<mlx_array*>(new array(std::move(out)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("paged_attention_forward", e.what());
    return nullptr;
  } catch (...) {
    mlx_trace_native_error("paged_attention_forward", "unknown exception");
    return nullptr;
  }
}

/// Production FFI: emit a `PagedKVWrite` MLX Custom primitive and return the
/// dependency-carrying K/V pool output arrays. The returned arrays share the
/// same Metal buffers as the input pools, but they also carry the MLX graph edge
/// from `new_k/new_v` -> cache write. Rust callers must feed these arrays into
/// subsequent MLX paged-attention reads, or explicitly eval them before raw
/// Metal readers touch the pool.
///
/// Returns false on bridge/factory validation errors. GPU dispatch errors still
/// surface later when MLX evaluates the returned arrays.
bool mlx_paged_kv_write_forward(
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* new_k_ptr,
    mlx_array* new_v_ptr,
    mlx_array* slot_mapping_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    int block_size,
    int num_kv_heads,
    int head_size,
    uint8_t kv_dtype_raw,
    mlx_array** out_k_pool_ptr,
    mlx_array** out_v_pool_ptr) {
  if (out_k_pool_ptr) {
    *out_k_pool_ptr = nullptr;
  }
  if (out_v_pool_ptr) {
    *out_v_pool_ptr = nullptr;
  }
  if (!k_pool_ptr || !v_pool_ptr || !new_k_ptr || !new_v_ptr ||
      !slot_mapping_ptr || !k_scale_ptr || !v_scale_ptr || !out_k_pool_ptr ||
      !out_v_pool_ptr) {
    return false;
  }

  try {
    using namespace mlx::core;
    using namespace mlx::core::fast;

    auto& k_pool       = *reinterpret_cast<array*>(k_pool_ptr);
    auto& v_pool       = *reinterpret_cast<array*>(v_pool_ptr);
    auto& new_k_raw    = *reinterpret_cast<array*>(new_k_ptr);
    auto& new_v_raw    = *reinterpret_cast<array*>(new_v_ptr);
    auto& slot_mapping = *reinterpret_cast<array*>(slot_mapping_ptr);
    auto& k_scale      = *reinterpret_cast<array*>(k_scale_ptr);
    auto& v_scale      = *reinterpret_cast<array*>(v_scale_ptr);

    auto kv_dtype = static_cast<KvDtype>(kv_dtype_raw);

    // Rust model code commonly produces K/V via transpose/reshape chains.
    // Keep these as lazy MLX `contiguous(...)` nodes instead of extracting
    // Metal buffers on the Rust side; this is the point of the bridge.
    auto new_k = contiguous(new_k_raw);
    auto new_v = contiguous(new_v_raw);

    auto out = paged_kv_write(
        k_pool,
        v_pool,
        new_k,
        new_v,
        slot_mapping,
        k_scale,
        v_scale,
        block_size,
        num_kv_heads,
        head_size,
        x_pack_for(kv_dtype),
        kv_dtype);

    *out_k_pool_ptr = reinterpret_cast<mlx_array*>(
        new array(std::move(out.first)));
    *out_v_pool_ptr = reinterpret_cast<mlx_array*>(
        new array(std::move(out.second)));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("paged_kv_write_forward", e.what());
    return false;
  } catch (...) {
    mlx_trace_native_error("paged_kv_write_forward", "unknown exception");
    return false;
  }
}

// =============================================================================
// FFI test helpers (Phase 1 only)
//
// These give the Rust unit tests in
// `crates/mlx-paged-attn/tests/paged_ops_smoke.rs` enough surface to
// exercise the C++ primitives' `is_equivalent` and `vjp` behaviour
// without standing up a full C++ test runner. Phase 2 may delete
// these once the dispatch path is C++-native and unit tests live in
// mlx-sys directly.
// =============================================================================

/// Construct two `PagedKVWrite` primitives with the supplied scalar
/// state and return whether `lhs.is_equivalent(rhs)` is true.
///
/// The fallback closure is a stub (throws if invoked) — the
/// `is_equivalent` check never invokes it.
///
/// `kv_dtype_lhs` / `kv_dtype_rhs` follow the C++ enum's u8 layout.
bool mlx_paged_kv_write_is_equivalent(
    int block_size_lhs,
    int num_kv_heads_lhs,
    int head_size_lhs,
    int x_pack_lhs,
    uint8_t kv_dtype_lhs,
    int block_size_rhs,
    int num_kv_heads_rhs,
    int head_size_rhs,
    int x_pack_rhs,
    uint8_t kv_dtype_rhs) {
  using namespace mlx::core::fast;

  auto stub_fallback = [](std::vector<mlx::core::array> /*ignored*/)
      -> std::vector<mlx::core::array> {
    throw std::runtime_error("is_equivalent test should not invoke fallback");
  };

  // Default stream is fine — `is_equivalent` does not depend on it.
  auto s = mlx::core::default_stream(mlx::core::default_device());

  PagedKVWrite lhs(
      s,
      stub_fallback,
      block_size_lhs,
      num_kv_heads_lhs,
      head_size_lhs,
      x_pack_lhs,
      static_cast<KvDtype>(kv_dtype_lhs));
  PagedKVWrite rhs(
      s,
      stub_fallback,
      block_size_rhs,
      num_kv_heads_rhs,
      head_size_rhs,
      x_pack_rhs,
      static_cast<KvDtype>(kv_dtype_rhs));

  return lhs.is_equivalent(rhs);
}

/// Invoke `PagedKVWrite::vjp` with empty argument vectors. Returns
/// `1` if the call threw a `std::runtime_error` containing the
/// expected message; `0` if the call returned without throwing or
/// threw a different error.
int mlx_paged_kv_write_vjp_throws() {
  using namespace mlx::core::fast;

  auto stub_fallback = [](std::vector<mlx::core::array> /*ignored*/)
      -> std::vector<mlx::core::array> { return {}; };
  auto s = mlx::core::default_stream(mlx::core::default_device());

  PagedKVWrite p(s, stub_fallback, 16, 4, 64, 8, KvDtype::Bf16);
  std::vector<mlx::core::array> empty_arrays;
  std::vector<int> empty_argnums;

  try {
    p.vjp(empty_arrays, empty_arrays, empty_argnums, empty_arrays);
  } catch (const std::runtime_error& e) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// Construct two `PagedAttention` primitives with the supplied scalar
/// state and return whether `lhs.is_equivalent(rhs)` is true.
bool mlx_paged_attention_is_equivalent(
    float scale_lhs,
    float softcap_lhs,
    int block_size_lhs,
    int num_q_heads_lhs,
    int num_kv_heads_lhs,
    int head_size_lhs,
    int sliding_window_lhs,
    uint8_t kv_dtype_lhs,
    float scale_rhs,
    float softcap_rhs,
    int block_size_rhs,
    int num_q_heads_rhs,
    int num_kv_heads_rhs,
    int head_size_rhs,
    int sliding_window_rhs,
    uint8_t kv_dtype_rhs) {
  using namespace mlx::core::fast;

  auto stub_fallback = [](std::vector<mlx::core::array> /*ignored*/)
      -> std::vector<mlx::core::array> {
    throw std::runtime_error("is_equivalent test should not invoke fallback");
  };
  auto s = mlx::core::default_stream(mlx::core::default_device());

  PagedAttention lhs(
      s,
      stub_fallback,
      scale_lhs,
      softcap_lhs,
      block_size_lhs,
      num_q_heads_lhs,
      num_kv_heads_lhs,
      head_size_lhs,
      sliding_window_lhs,
      static_cast<KvDtype>(kv_dtype_lhs));
  PagedAttention rhs(
      s,
      stub_fallback,
      scale_rhs,
      softcap_rhs,
      block_size_rhs,
      num_q_heads_rhs,
      num_kv_heads_rhs,
      head_size_rhs,
      sliding_window_rhs,
      static_cast<KvDtype>(kv_dtype_rhs));

  return lhs.is_equivalent(rhs);
}

/// Same idea for `PagedAttention`.
int mlx_paged_attention_vjp_throws() {
  using namespace mlx::core::fast;

  auto stub_fallback = [](std::vector<mlx::core::array> /*ignored*/)
      -> std::vector<mlx::core::array> { return {}; };
  auto s = mlx::core::default_stream(mlx::core::default_device());

  PagedAttention p(
      s,
      stub_fallback,
      /*scale=*/0.125f,
      /*softcap=*/0.0f,
      /*block_size=*/16,
      /*num_q_heads=*/8,
      /*num_kv_heads=*/4,
      /*head_size=*/64,
      /*sliding_window=*/0,
      KvDtype::Bf16);
  std::vector<mlx::core::array> empty_arrays;
  std::vector<int> empty_argnums;

  try {
    p.vjp(empty_arrays, empty_arrays, empty_argnums, empty_arrays);
  } catch (const std::runtime_error& e) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// Verify that `PagedAttention::output_shapes` ignores q's trailing
/// dims and instead uses the primitive's scalar state. Constructs a
/// `PagedAttention` with the supplied scalar state and a tracer-only
/// q array of shape `[q_num_tokens, q_dim1_actual, q_dim2_actual]`,
/// then calls `output_shapes` and copies the returned shape to
/// `out_shape` (caller must size to 3 elements). Returns the number
/// of dimensions in the returned shape (should always be 3 for
/// well-formed input).
int mlx_paged_attention_test_output_shapes(
    int q_num_tokens,
    int q_dim1_actual,
    int q_dim2_actual,
    float scale,
    float softcap,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int sliding_window,
    uint8_t kv_dtype_raw,
    int32_t* out_shape) {
  using namespace mlx::core::fast;

  auto stub_fallback = [](std::vector<mlx::core::array> /*ignored*/)
      -> std::vector<mlx::core::array> {
    throw std::runtime_error("output_shapes test should not invoke fallback");
  };
  auto s = mlx::core::default_stream(mlx::core::default_device());

  PagedAttention prim(
      s,
      stub_fallback,
      scale,
      softcap,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window,
      static_cast<KvDtype>(kv_dtype_raw));

  // Tracer-only q array — shape encodes potentially-mismatched
  // dimensions to verify output_shapes doesn't echo them.
  mlx::core::Shape q_shape{q_num_tokens, q_dim1_actual, q_dim2_actual};
  mlx::core::array q(std::move(q_shape), mlx::core::bfloat16, nullptr, {});

  std::vector<mlx::core::array> inputs{q};
  auto shapes = prim.output_shapes(inputs);
  if (shapes.size() != 1) {
    return -1;
  }
  const auto& out = shapes[0];
  out_shape[0] = static_cast<int32_t>(out[0]);
  out_shape[1] = static_cast<int32_t>(out[1]);
  out_shape[2] = static_cast<int32_t>(out[2]);
  return static_cast<int>(out.size());
}

/// Returns 1 iff the public `paged_attention(...)` factory throws
/// `std::invalid_argument` when called with sliding_window=-1.
/// Returns 0 if it doesn't throw or throws a different exception.
///
/// Phase 7 lifts the Phase 1 sliding_window=0 restriction; the factory
/// now accepts any non-negative value. Negative values remain illegal
/// because the only "no mask" sentinel is 0, so we test a NEGATIVE
/// value here to keep the test exercising the rejection path. The
/// pool/q arrays use tracer-only construction (no backing data needed —
/// the throw fires before eval_gpu).
int mlx_paged_attention_factory_rejects_sliding_window() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  // Build well-formed tracer arrays so only sliding_window triggers
  // the throw.
  // q: [num_seqs=1, num_q_heads=8, head_size=64]
  // k_pool: [num_blocks=4, num_kv_heads=4, head_size/x=8, block_size=16, x=8]
  // v_pool: [num_blocks=4, num_kv_heads=4, head_size=64, block_size=16]
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/-1,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*kv_dtype=*/KvDtype::Bf16,
        /*s=*/StreamOrDevice{});
  } catch (const std::invalid_argument& /*e*/) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// Verify the public `paged_attention(...)` factory rejects q whose
/// trailing dims disagree with the primitive's scalar state.
/// Returns 1 iff `std::invalid_argument` was thrown, 0 otherwise.
int mlx_paged_attention_factory_rejects_q_shape_mismatch() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  // q.shape(2) deliberately disagrees with head_size=64 (we pass 32).
  array q(Shape{1, 8, 32}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        0.125f,
        0.0f,
        /*sliding_window=*/0,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

// =============================================================================
// Phase 1 review-round-3 negative-validation test helpers.
//
// Each helper constructs `paged_attention` / `paged_kv_write` factory
// inputs that are well-formed EXCEPT for one specific dim or dtype.
// The factory MUST reject by throwing `std::invalid_argument`. Any
// other outcome is a regression.
//
// Returns: 1 if `std::invalid_argument` was thrown, 0 otherwise.
// =============================================================================

} // extern "C"

namespace {

// Helper: invoke `paged_attention(...)` with a custom (q, k_pool,
// v_pool, block_table, seq_lens) and well-formed scalar state. Returns
// 1 on `std::invalid_argument`, 0 otherwise. Used by the negative-test
// helpers below to keep them concise.
//
// `kv_dtype` defaults to `Bf16`, matching the original Phase 1 review
// callers. Round-4 review adds Fp8 cases that pass `kv_dtype = Fp8`.
int call_paged_attention_expecting_throw(
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    mlx::core::fast::KvDtype kv_dtype = mlx::core::fast::KvDtype::Bf16) {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);
  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/0,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        kv_dtype,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

// Helper: invoke `paged_kv_write(...)` with custom inputs and the
// supplied (kv_dtype, x_pack) pair. Round-4 review uses this with
// `KvDtype::Fp8 + x_pack=16` to validate FP8 dtype rejection.
int call_paged_kv_write_expecting_throw(
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& new_k,
    const mlx::core::array& new_v,
    const mlx::core::array& slot_mapping,
    mlx::core::fast::KvDtype kv_dtype = mlx::core::fast::KvDtype::Bf16,
    int x_pack = 8) {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);
  try {
    paged_kv_write(
        k_pool,
        v_pool,
        new_k,
        new_v,
        slot_mapping,
        k_scale,
        v_scale,
        /*block_size=*/16,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        x_pack,
        kv_dtype,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

} // namespace

extern "C" {

/// q with rank 2 (not 3) must be rejected.
int mlx_paged_attention_factory_rejects_q_rank_not_3() {
  using namespace mlx::core;
  array q(Shape{8, 64}, bfloat16, nullptr, {}); // rank 2
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// block_table.shape(0) != q.shape(0) must be rejected.
int mlx_paged_attention_factory_rejects_block_table_batch_mismatch() {
  using namespace mlx::core;
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  // block_table batch=2, q batch=1 → mismatch
  array block_table(Shape{2, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// block_table dtype != int32 must be rejected.
int mlx_paged_attention_factory_rejects_block_table_dtype() {
  using namespace mlx::core;
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  // block_table dtype is uint32 (kernel expects int32)
  array block_table(Shape{1, 4}, uint32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// seq_lens.shape(0) != q.shape(0) must be rejected.
int mlx_paged_attention_factory_rejects_seq_lens_batch_mismatch() {
  using namespace mlx::core;
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  // seq_lens length=2, q batch=1 → mismatch
  array seq_lens(Shape{2}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// k_pool.shape(2) != head_size / x_pack must be rejected.
int mlx_paged_attention_factory_rejects_k_pool_inner_dim() {
  using namespace mlx::core;
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  // k_pool.shape(2) = 4 but expected = head_size/x_pack = 64/8 = 8
  array k_pool(Shape{4, 4, 4, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// k_pool.shape(4) != x_pack must be rejected.
int mlx_paged_attention_factory_rejects_k_pool_x_pack() {
  using namespace mlx::core;
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  // k_pool.shape(4) = 16 but expected x_pack for Bf16 = 8
  array k_pool(Shape{4, 4, 8, 16, 16}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// v_pool.shape(2) != head_size must be rejected.
int mlx_paged_attention_factory_rejects_v_pool_head_dim() {
  using namespace mlx::core;
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  // v_pool.shape(2) = 32 but expected = head_size = 64
  array v_pool(Shape{4, 4, 32, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// k_pool.shape(0) != v_pool.shape(0) must be rejected (num_blocks mismatch).
int mlx_paged_attention_factory_rejects_num_blocks_mismatch() {
  using namespace mlx::core;
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  // k_pool num_blocks=4, v_pool num_blocks=8 → mismatch
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{8, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// slot_mapping with rank 2 (not 1) must be rejected.
int mlx_paged_kv_write_factory_rejects_slot_mapping_rank() {
  using namespace mlx::core;
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array new_k(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array new_v(Shape{2, 4, 64}, bfloat16, nullptr, {});
  // slot_mapping with rank 2 — must be rank 1
  array slot_mapping(Shape{1, 2}, int64, nullptr, {});
  return call_paged_kv_write_expecting_throw(
      k_pool, v_pool, new_k, new_v, slot_mapping);
}

/// slot_mapping with int32 dtype must be rejected (kernel reads int64).
int mlx_paged_kv_write_factory_rejects_slot_mapping_dtype() {
  using namespace mlx::core;
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array new_k(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array new_v(Shape{2, 4, 64}, bfloat16, nullptr, {});
  // slot_mapping dtype is int32 — must be int64
  array slot_mapping(Shape{2}, int32, nullptr, {});
  return call_paged_kv_write_expecting_throw(
      k_pool, v_pool, new_k, new_v, slot_mapping);
}

/// slot_mapping length != new_k.shape(0) must be rejected.
int mlx_paged_kv_write_factory_rejects_slot_mapping_length() {
  using namespace mlx::core;
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array new_k(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array new_v(Shape{2, 4, 64}, bfloat16, nullptr, {});
  // slot_mapping length=3, new_k tokens=2 → mismatch
  array slot_mapping(Shape{3}, int64, nullptr, {});
  return call_paged_kv_write_expecting_throw(
      k_pool, v_pool, new_k, new_v, slot_mapping);
}

/// slot_mapping with a max value >= num_blocks * block_size must be
/// rejected (Phase 1 safety check).
///
/// This case requires REAL data — the eval-based bounds check only
/// fires when slot_mapping has a backing buffer that can be evaluated.
/// On hosts without Metal, the eval still works for small int64
/// arrays because MLX can evaluate scalar reductions on CPU. Returns
/// 1 if the throw fired, 0 if not, -1 on construction error.
int mlx_paged_kv_write_factory_rejects_slot_mapping_out_of_range() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  // pool with 4 blocks × block_size=16 → capacity = 64 slots.
  // We pass slot_mapping = [0, 64] which has max=64 == capacity →
  // out of range (slot 64 doesn't exist; valid slots are 0..63).
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});

  // For new_k/new_v we use REAL data so the factory has data to
  // validate against. Two tokens of zero-filled BF16.
  std::vector<uint16_t> new_kv_zeros(2 * 4 * 64, 0);
  auto* bf16_p = reinterpret_cast<const bfloat16_t*>(new_kv_zeros.data());
  array new_k(bf16_p, Shape{2, 4, 64}, bfloat16);
  array new_v(bf16_p, Shape{2, 4, 64}, bfloat16);

  // slot_mapping: [0, 64] — max=64 == 4*16 = pool capacity → REJECTED.
  std::vector<int64_t> slot_mapping_host = {0, 64};
  array slot_mapping(slot_mapping_host.data(), Shape{2}, int64);
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  try {
    paged_kv_write(
        k_pool,
        v_pool,
        new_k,
        new_v,
        slot_mapping,
        k_scale,
        v_scale,
        /*block_size=*/16,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*x_pack=*/8,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// Round-13 finding: assert the factory-side slot_mapping bounds
/// guard's `std::invalid_argument` message contains the `[runtime]`
/// marker. The factory and the compile-cached eval_gpu path BOTH guard
/// the same data-dependent property (slot value < num_blocks *
/// block_size); both must use the `[runtime]` prefix so callers / test
/// infrastructure can distinguish runtime-content guards from the
/// `[validator]`-tagged structural validators that produce different
/// throw classes.
///
/// Returns:
///   1   — factory threw `std::invalid_argument` AND the message
///         contains the `[runtime]` substring (pass).
///   0   — factory did NOT throw (regression — out-of-range
///         slot_mapping reached the kernel).
///  -2   — factory threw `std::invalid_argument` but the message did
///         NOT contain `[runtime]` (the prefix regressed; the runtime
///         guard is no longer uniformly tagged).
///  -1   — factory threw a non-`std::invalid_argument` exception, or
///         a setup step threw (internal helper bug).
///
/// The companion Rust test `paged_kv_write_factory_runtime_guard_marker`
/// asserts rc=1.
int mlx_paged_kv_write_factory_slot_mapping_out_of_range_runtime_marker() {
  try {
    using namespace mlx::core;
    using namespace mlx::core::fast;

    // Same configuration as `_factory_rejects_slot_mapping_out_of_range`:
    // pool with 4 blocks × block_size=16 → capacity = 64 slots, and
    // slot_mapping=[0, 64] so max_slot=64 == capacity → out of range.
    array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
    array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});

    std::vector<uint16_t> new_kv_zeros(2 * 4 * 64, 0);
    auto* bf16_p = reinterpret_cast<const bfloat16_t*>(new_kv_zeros.data());
    array new_k(bf16_p, Shape{2, 4, 64}, bfloat16);
    array new_v(bf16_p, Shape{2, 4, 64}, bfloat16);

    std::vector<int64_t> slot_mapping_host = {0, 64};
    array slot_mapping(slot_mapping_host.data(), Shape{2}, int64);
    array k_scale(1.0f, float32);
    array v_scale(1.0f, float32);

    static constexpr const char* kRuntimeTag = "[runtime]";
    try {
      paged_kv_write(
          k_pool,
          v_pool,
          new_k,
          new_v,
          slot_mapping,
          k_scale,
          v_scale,
          /*block_size=*/16,
          /*num_kv_heads=*/4,
          /*head_size=*/64,
          /*x_pack=*/8,
          KvDtype::Bf16,
          StreamOrDevice{});
    } catch (const std::invalid_argument& e) {
      const std::string what = e.what();
      if (what.find(kRuntimeTag) != std::string::npos) {
        return 1;
      }
      fprintf(
          stderr,
          "[mlx_paged_kv_write_factory_slot_mapping_out_of_range_runtime_"
          "marker] factory threw std::invalid_argument but message did NOT "
          "contain '%s' marker — runtime-content guard prefix regressed. "
          "Got: %s\n",
          kRuntimeTag,
          what.c_str());
      return -2;
    } catch (const std::exception& e) {
      fprintf(
          stderr,
          "[mlx_paged_kv_write_factory_slot_mapping_out_of_range_runtime_"
          "marker] factory threw non-invalid_argument: %s\n",
          e.what());
      return -1;
    }
    fprintf(
        stderr,
        "[mlx_paged_kv_write_factory_slot_mapping_out_of_range_runtime_"
        "marker] factory did NOT throw on out-of-range slot_mapping\n");
    return 0;
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_factory_slot_mapping_out_of_range_runtime_"
        "marker] FFI boundary caught uncontained C++ exception "
        "(extern-C catch-all): %s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_factory_slot_mapping_out_of_range_runtime_"
        "marker] FFI boundary caught uncontained non-std exception "
        "(extern-C catch-all)\n");
    return -1;
  }
}

} // extern "C"

/// Counter used by `mlx_paged_kv_write_compile_trace_*` helpers. Each
/// trace inside the compiled graph increments this counter (a cache
/// MISS in `compiler_cache().find` triggers a re-trace, which calls
/// `paged_kv_write_trace_fn` once). Cache HITs do not call the fn.
///
/// Callers must reset to 0 before exercising a fresh test.
namespace {
std::atomic<int> g_paged_kv_write_trace_count{0};
} // namespace

extern "C" {

/// Reset the trace counter so a test can exercise compile-cache
/// behavior in isolation.
void mlx_paged_kv_write_trace_count_reset() {
  g_paged_kv_write_trace_count.store(0, std::memory_order_seq_cst);
}

/// Read the current trace counter.
int mlx_paged_kv_write_trace_count_get() {
  return g_paged_kv_write_trace_count.load(std::memory_order_seq_cst);
}

} // extern "C"

namespace {

/// The function we hand to `mlx::core::compile`. MLX traces it ONCE
/// per (input shapes, dtypes, constants) tuple — the first call
/// drains through `compile_trace`, which invokes this function with
/// tracer inputs. Subsequent calls with the same shapes/dtypes hit
/// the compile cache and DO NOT call this function. The counter
/// increment is the canonical "did this compile re-trace?" signal.
///
/// The fn itself just emits a `paged_kv_write` primitive on the trace
/// inputs. We don't actually evaluate; the test's purpose is to count
/// trace invocations, not to dispatch GPU.
///
/// Inputs (positional):
///   0: k_pool, 1: v_pool, 2: new_k, 3: new_v, 4: slot_mapping,
///   5: k_scale, 6: v_scale
///
/// Outputs: [k_pool', v_pool'] (semantic in-place aliases).
std::vector<mlx::core::array> paged_kv_write_trace_fn(
    const std::vector<mlx::core::array>& inputs) {
  using namespace mlx::core::fast;
  if (inputs.size() != 7) {
    throw std::runtime_error("paged_kv_write_trace_fn: expected 7 inputs");
  }
  g_paged_kv_write_trace_count.fetch_add(1, std::memory_order_seq_cst);

  // Hard-coded scalar state matches the test inputs in
  // `paged_ops_smoke.rs::compile_trace_paged_kv_write_caches_one_trace`.
  // Block_size=16, num_kv_heads=4, head_size=64, x_pack=8, Bf16.
  auto out = paged_kv_write(
      inputs[0],
      inputs[1],
      inputs[2],
      inputs[3],
      inputs[4],
      inputs[5],
      inputs[6],
      /*block_size=*/16,
      /*num_kv_heads=*/4,
      /*head_size=*/64,
      /*x_pack=*/8,
      KvDtype::Bf16,
      /*s=*/{});
  return {out.first, out.second};
}

} // namespace

extern "C" {

/// Build a `mlx::core::compile`-wrapped function around
/// `paged_kv_write_trace_fn`, call it twice with REAL data-backed
/// inputs that share shapes/dtypes but differ in contents, and return
/// how many times the inner trace ran. The caller asserts the count
/// is exactly 1 after two calls (i.e., the second call hit the
/// compile cache and did NOT re-trace). Returns the trace count
/// (typically 1 on success), or a negative code on error.
///
/// Layout used (matches paged_kv_write_trace_fn's hardcoded scalars):
///   block_size=16, num_kv_heads=4, head_size=64, x_pack=8, Bf16.
///   k_pool: [4, 4, 8, 16, 8] bf16  (shared across both calls;
///                                  in-place writes accumulate)
///   v_pool: [4, 4, 64, 16] bf16
///   new_k:  [num_tokens, 4, 64] bf16
///   new_v:  [num_tokens, 4, 64] bf16
///   slot_mapping: [num_tokens] int64
///
/// `num_tokens` is fixed across both calls (otherwise the cache key
/// would diverge on shape).
///
/// Beyond the trace count, the helper EVALUATES each call's outputs
/// and inspects the second call's K-pool slots after the second eval.
/// If the compile cache wrongly threaded the FIRST call's traced
/// inputs into the second invocation (a `compile_replace` bug), the
/// second call's slots would still hold the first call's K values
/// (or zero, if the second call effectively ran on first-call inputs
/// against the same pool). The test asserts the second call's slots
/// hold the SECOND call's K bytes, which proves both
/// (a) cache HIT (counter==1) AND
/// (b) runtime contents flow through `compile_replace` correctly.
///
/// Return codes:
///   `count` (>=0) — trace counter at end (1 on success).
///   -1            — internal/setup error.
///   -2            — second-call slots did NOT contain second-call K
///                   values (compile_replace runtime-thread bug).
///   -3            — Metal not available; eval-based verification
///                   skipped. The trace-count check still ran.
int mlx_paged_kv_write_compile_trace_smoke(int num_tokens) {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (num_tokens <= 0) {
    return -1;
  }

  // Reset counter so caller sees a clean slate (also protects against
  // earlier tests in the same process having compiled a graph that
  // happened to share fun_id).
  g_paged_kv_write_trace_count.store(0, std::memory_order_seq_cst);

  // Compile our trace function. This wraps it in MLX's compile cache
  // — subsequent calls with the same input shapes/dtypes hit the
  // cache and skip re-tracing.
  auto compiled = mlx::core::compile(&paged_kv_write_trace_fn);

  // Build REAL data-backed inputs. Shapes match the layout above.
  // The K/V pools are shared across both calls so we can verify the
  // second call's writes overlay the same buffer in distinct slot
  // ranges from the first call's writes.
  const int kBlockSize = 16;
  const int kNumKvHeads = 4;
  const int kHeadSize = 64;
  const int kXPack = 8;
  const int kNumBlocks = 4;

  const size_t k_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      (kHeadSize / kXPack) * kBlockSize * kXPack;
  const size_t v_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      kHeadSize * kBlockSize;
  const size_t per_token_elems = static_cast<size_t>(kNumKvHeads) * kHeadSize;
  const size_t new_kv_elems = static_cast<size_t>(num_tokens) * per_token_elems;

  // Initialize pools to zero (sentinel: any nonzero we read after
  // the dispatch must have come from a kernel write).
  std::vector<uint16_t> k_pool_host(k_pool_elems, 0);
  std::vector<uint16_t> v_pool_host(v_pool_elems, 0);

  // K_VAL_A / V_VAL_A: first call fills new_k/new_v with these.
  // We pick two distinct nonzero BF16 representations so the byte
  // pattern in the pool unambiguously identifies which call's input
  // landed there.
  //
  // BF16 of 1.5 = 0x3FC0; BF16 of 3.5 = 0x4060.
  const uint16_t kKValA = 0x3FC0; // bf16(1.5)
  const uint16_t kVValA = 0x4040; // bf16(3.0)
  const uint16_t kKValB = 0x4060; // bf16(3.5)
  const uint16_t kVValB = 0x40A0; // bf16(5.0)

  std::vector<uint16_t> new_k_host_a(new_kv_elems, kKValA);
  std::vector<uint16_t> new_v_host_a(new_kv_elems, kVValA);
  std::vector<uint16_t> new_k_host_b(new_kv_elems, kKValB);
  std::vector<uint16_t> new_v_host_b(new_kv_elems, kVValB);

  // First call's slots: 0..num_tokens-1 (block 0).
  // Second call's slots: kBlockSize..kBlockSize+num_tokens-1 (block 1).
  // num_tokens <= block_size = 16 keeps slot ranges within
  // their respective blocks for clean verification.
  if (num_tokens > kBlockSize) {
    return -1;
  }
  std::vector<int64_t> slot_mapping_host_a(num_tokens);
  std::vector<int64_t> slot_mapping_host_b(num_tokens);
  for (int i = 0; i < num_tokens; ++i) {
    slot_mapping_host_a[i] = static_cast<int64_t>(i);
    slot_mapping_host_b[i] = static_cast<int64_t>(kBlockSize + i);
  }

  // Build BF16 pool arrays once and SHARE across both calls. We do
  // this by binding shared `array` instances and passing them by
  // value (MLX `array` is a refcounted handle).
  Shape k_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize / kXPack, kBlockSize,
      kXPack};
  Shape v_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize, kBlockSize};
  Shape new_kv_shape{num_tokens, kNumKvHeads, kHeadSize};

  // Construct arrays using the iterator-template constructor (copies
  // the data into MLX's allocator, returning real-data-backed arrays).
  auto bf16_arr = [](const std::vector<uint16_t>& src, Shape shape) {
    auto* p = reinterpret_cast<const bfloat16_t*>(src.data());
    return array(p, std::move(shape), bfloat16);
  };
  auto i64_arr = [](const std::vector<int64_t>& src, Shape shape) {
    return array(src.data(), std::move(shape), int64);
  };

  array k_pool = bf16_arr(k_pool_host, k_pool_shape);
  array v_pool = bf16_arr(v_pool_host, v_pool_shape);
  array new_k_a = bf16_arr(new_k_host_a, new_kv_shape);
  array new_v_a = bf16_arr(new_v_host_a, new_kv_shape);
  array slot_mapping_a = i64_arr(slot_mapping_host_a, Shape{num_tokens});
  array k_scale_a(1.0f, float32);
  array v_scale_a(1.0f, float32);

  std::vector<array> inputs1{
      k_pool, v_pool, new_k_a, new_v_a, slot_mapping_a, k_scale_a, v_scale_a};

  std::vector<array> outputs1;
  try {
    outputs1 = compiled(inputs1);
    if (outputs1.size() != 2) {
      return -1;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[compile_trace_smoke] first call threw: %s\n", e.what());
    return -1;
  }

  int count_after_first =
      g_paged_kv_write_trace_count.load(std::memory_order_seq_cst);
  if (count_after_first != 1) {
    fprintf(
        stderr,
        "[compile_trace_smoke] expected 1 trace after first call, got %d\n",
        count_after_first);
    return -1;
  }

  // Build the SECOND set of inputs. Shapes/dtypes match (so the cache
  // is hit) but contents DIFFER (different K/V values, different slot
  // mapping). The cache hit must still substitute these new arrays as
  // the primitive's inputs at eval-time.
  array new_k_b = bf16_arr(new_k_host_b, new_kv_shape);
  array new_v_b = bf16_arr(new_v_host_b, new_kv_shape);
  array slot_mapping_b = i64_arr(slot_mapping_host_b, Shape{num_tokens});
  array k_scale_b(1.0f, float32);
  array v_scale_b(1.0f, float32);

  // Re-use the SAME k_pool / v_pool arrays — both calls share storage,
  // and the second call's writes should overlay the first call's at
  // a different slot range. Caveat: since `paged_kv_write` outputs
  // alias their input pools via `copy_shared_buffer`, the second
  // call's `inputs2[0/1]` must be the SAME array instances as the
  // first call's output aliases for MLX's graph machinery to thread
  // correctly. Using the original `k_pool` / `v_pool` is the
  // standard pattern: outputs and inputs share the same allocation
  // (read the test's verification of pool bytes for proof).
  std::vector<array> inputs2{
      k_pool, v_pool, new_k_b, new_v_b, slot_mapping_b, k_scale_b, v_scale_b};

  std::vector<array> outputs2;
  try {
    outputs2 = compiled(inputs2);
    if (outputs2.size() != 2) {
      return -1;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[compile_trace_smoke] second call threw: %s\n", e.what());
    return -1;
  }

  int count_after_second =
      g_paged_kv_write_trace_count.load(std::memory_order_seq_cst);
  if (count_after_second != 1) {
    return count_after_second;
  }

  // Eval-based verification (Phase 1 second-call-contents check).
  //
  // Skip if Metal isn't available — the dispatch path is GPU-only,
  // so on a non-Metal host we can only verify the trace-count
  // semantics. Return -3 to signal "Metal-skip" so the test caller
  // can mark it as a no-op-success.
  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  try {
    mlx::core::eval(outputs1[0], outputs1[1]);
    mlx::core::eval(outputs2[0], outputs2[1]);
  } catch (const std::exception& e) {
    fprintf(stderr, "[compile_trace_smoke] eval threw: %s\n", e.what());
    return -1;
  }

  // Verify the second call's K bytes landed at slot kBlockSize+0..
  // We read pool[block=1, head=0, x_idx=0, t=0, x_offset=0] which
  // corresponds to the first head/dim of slot kBlockSize.
  //
  // K layout strides (in elements): same as the round-trip test.
  const size_t head_per_block_k = static_cast<size_t>(kHeadSize / kXPack) *
      kBlockSize * kXPack;
  const size_t stride_block_k =
      static_cast<size_t>(kNumKvHeads) * head_per_block_k;
  const size_t stride_head_k = head_per_block_k;
  const size_t stride_xidx_k = static_cast<size_t>(kBlockSize) * kXPack;
  const size_t stride_blockoff_k = kXPack;

  const bfloat16_t* k_pool_bf16 = k_pool.data<bfloat16_t>();
  if (k_pool_bf16 == nullptr) {
    fprintf(stderr, "[compile_trace_smoke] k_pool data ptr is null\n");
    return -1;
  }
  // bfloat16_t is layout-compatible with uint16_t (16-bit `bits_`).
  const uint16_t* k_pool_data =
      reinterpret_cast<const uint16_t*>(k_pool_bf16);

  // Check first-call slots (block 0): expect kKValA.
  // Check second-call slots (block 1): expect kKValB.
  // (Sentinel: if compile_replace threaded inputs1 instead of
  // inputs2, block 1 would either be 0 — never written — or kKValA
  // — first call's data written to first call's slots in pool, but
  // not into block 1.)
  for (int t = 0; t < num_tokens; ++t) {
    const size_t block0_offset = stride_block_k * 0 + // block 0
        stride_head_k * 0 + // head 0
        stride_xidx_k * 0 + // x_idx 0 (j=0..7)
        stride_blockoff_k * static_cast<size_t>(t) + // block_offset = t
        0; // x_offset 0 (j=0)
    const size_t block1_offset = stride_block_k * 1 + // block 1
        stride_head_k * 0 + stride_xidx_k * 0 +
        stride_blockoff_k * static_cast<size_t>(t) + 0;

    if (k_pool_data[block0_offset] != kKValA) {
      fprintf(
          stderr,
          "[compile_trace_smoke] block0 slot t=%d: expected kKValA=0x%04x, "
          "got 0x%04x (first-call K should be at first-call slots)\n",
          t,
          kKValA,
          k_pool_data[block0_offset]);
      return -2;
    }
    if (k_pool_data[block1_offset] != kKValB) {
      fprintf(
          stderr,
          "[compile_trace_smoke] block1 slot t=%d: expected kKValB=0x%04x, "
          "got 0x%04x (second-call K must be at second-call slots; if you "
          "see kKValA=0x%04x or 0x0000 the compile_replace runtime-thread "
          "is broken)\n",
          t,
          kKValB,
          k_pool_data[block1_offset],
          kKValA);
      return -2;
    }
  }

  // V-pool byte verification (Phase 1 review-round-4 addition).
  //
  // K-only verification can pass even if input 3 (`new_v`) was left
  // stale by a `compile_replace` bug that correctly threaded inputs
  // 0/1/2/4 (k_pool, v_pool, new_k, slot_mapping) but not input 3
  // (new_v). In that case `kVValA` would land in block 1 V-slots
  // instead of `kVValB`. Read the V pool back and assert each test
  // slot holds the expected V value to close that hole.
  //
  // V layout (vLLM, distinct from K's `_x` packing):
  //   v_pool[block_idx, num_kv_heads, head_size, block_size]
  // Strides (in elements):
  const size_t head_per_block_v =
      static_cast<size_t>(kHeadSize) * kBlockSize;
  const size_t stride_block_v =
      static_cast<size_t>(kNumKvHeads) * head_per_block_v;
  const size_t stride_head_v = head_per_block_v;
  const size_t stride_j_v = static_cast<size_t>(kBlockSize);

  const bfloat16_t* v_pool_bf16 = v_pool.data<bfloat16_t>();
  if (v_pool_bf16 == nullptr) {
    fprintf(stderr, "[compile_trace_smoke] v_pool data ptr is null\n");
    return -1;
  }
  const uint16_t* v_pool_data =
      reinterpret_cast<const uint16_t*>(v_pool_bf16);

  // Check first-call slots (block 0, head=0, j=0): expect kVValA.
  // Check second-call slots (block 1, head=0, j=0): expect kVValB.
  for (int t = 0; t < num_tokens; ++t) {
    const size_t v_block0_offset = stride_block_v * 0 + // block 0
        stride_head_v * 0 + // head 0
        stride_j_v * 0 + // j = 0
        static_cast<size_t>(t); // block_offset = t
    const size_t v_block1_offset = stride_block_v * 1 + // block 1
        stride_head_v * 0 + // head 0
        stride_j_v * 0 + // j = 0
        static_cast<size_t>(t);

    if (v_pool_data[v_block0_offset] != kVValA) {
      fprintf(
          stderr,
          "[compile_trace_smoke] V slot at block 0, position t=%d, head=0, j=0: "
          "expected kVValA=0x%04x got 0x%04x (first-call V should be at first-call slots)\n",
          t,
          kVValA,
          v_pool_data[v_block0_offset]);
      return -2;
    }
    if (v_pool_data[v_block1_offset] != kVValB) {
      fprintf(
          stderr,
          "[compile_trace_smoke] V slot at block 1, position t=%d, head=0, j=0: "
          "expected kVValB=0x%04x got 0x%04x — did compile_replace fail to thread input 3 (new_v)? "
          "If you see kVValA=0x%04x the second call inherited the first call's new_v.\n",
          t,
          kVValB,
          v_pool_data[v_block1_offset],
          kVValA);
      return -2;
    }
  }

  return count_after_second;
}

/// Verify the public `paged_kv_write(...)` factory rejects k_pool
/// whose interior dims disagree with the primitive's scalar state.
int mlx_paged_kv_write_factory_rejects_pool_shape_mismatch() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  // k_pool: [num_blocks=4, num_kv_heads=8 (WRONG, expects 4), 8, 16, 8]
  array k_pool(Shape{4, 8, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array new_k(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array new_v(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array slot_mapping(Shape{2}, int64, nullptr, {});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  try {
    paged_kv_write(
        k_pool,
        v_pool,
        new_k,
        new_v,
        slot_mapping,
        k_scale,
        v_scale,
        /*block_size=*/16,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*x_pack=*/8,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

// =============================================================================
// Phase 1 review-round-4 dtype-mismatch negative tests for paged_kv_write.
//
// Round 4 finding B: the factory previously checked only pairwise dtype
// equality (k_pool == v_pool, new_k == new_v) — it did NOT verify the
// dtype matched the cache/io dtype implied by `kv_dtype`. These helpers
// each construct factory inputs that are well-formed EXCEPT for one
// dtype slot, then assert `std::invalid_argument` is thrown.
// =============================================================================

/// k_pool dtype is uint8 instead of the expected bfloat16 for Bf16.
int mlx_paged_kv_write_factory_rejects_k_pool_dtype_bf16() {
  using namespace mlx::core;
  // Both pools must agree pairwise to reach the kv_dtype check.
  array k_pool(Shape{4, 4, 8, 16, 8}, uint8, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, uint8, nullptr, {});
  // new_k/new_v keep the right io dtype so only k_pool dtype triggers
  // the throw.
  array new_k(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array new_v(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array slot_mapping(Shape{2}, int64, nullptr, {});
  return call_paged_kv_write_expecting_throw(
      k_pool, v_pool, new_k, new_v, slot_mapping);
}

/// v_pool dtype is uint8 instead of the expected bfloat16 for Bf16.
/// k_pool stays at the right dtype so the k_pool kv_dtype check
/// passes; v_pool is the only mismatch, exercising the v_pool-vs-
/// kv_dtype check directly (the kv_dtype checks fire BEFORE the
/// pairwise check now, see the factory's reordered validation).
int mlx_paged_kv_write_factory_rejects_v_pool_dtype_bf16() {
  using namespace mlx::core;
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, uint8, nullptr, {});
  array new_k(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array new_v(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array slot_mapping(Shape{2}, int64, nullptr, {});
  return call_paged_kv_write_expecting_throw(
      k_pool, v_pool, new_k, new_v, slot_mapping);
}

/// new_k dtype is float32 instead of the expected bfloat16 for Bf16.
int mlx_paged_kv_write_factory_rejects_new_k_dtype_bf16() {
  using namespace mlx::core;
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  // new_k float32, new_v float32 — pair agrees, but disagrees with
  // kv_dtype's expected io dtype (bfloat16).
  array new_k(Shape{2, 4, 64}, float32, nullptr, {});
  array new_v(Shape{2, 4, 64}, float32, nullptr, {});
  array slot_mapping(Shape{2}, int64, nullptr, {});
  return call_paged_kv_write_expecting_throw(
      k_pool, v_pool, new_k, new_v, slot_mapping);
}

/// new_v dtype is float32 instead of the expected bfloat16 for Bf16.
/// new_k stays at the right dtype so the new_k kv_dtype check passes;
/// new_v is the only mismatch, exercising the new_v-vs-kv_dtype check
/// directly (the kv_dtype checks fire BEFORE the pairwise check now,
/// see the factory's reordered validation).
int mlx_paged_kv_write_factory_rejects_new_v_dtype_bf16() {
  using namespace mlx::core;
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array new_k(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array new_v(Shape{2, 4, 64}, float32, nullptr, {});
  array slot_mapping(Shape{2}, int64, nullptr, {});
  return call_paged_kv_write_expecting_throw(
      k_pool, v_pool, new_k, new_v, slot_mapping);
}

/// k_pool dtype is bfloat16 instead of the expected uint8 for Fp8.
int mlx_paged_kv_write_factory_rejects_k_pool_dtype_fp8() {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  // FP8 expects k_pool/v_pool dtype = uint8, x_pack = 16, head_size/x = 4.
  // We pass bfloat16 for the pools to trigger the kv_dtype check.
  array k_pool(Shape{4, 4, 4, 16, 16}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array new_k(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array new_v(Shape{2, 4, 64}, bfloat16, nullptr, {});
  array slot_mapping(Shape{2}, int64, nullptr, {});
  return call_paged_kv_write_expecting_throw(
      k_pool, v_pool, new_k, new_v, slot_mapping, KvDtype::Fp8, /*x_pack=*/16);
}

/// new_k dtype is float16 instead of the expected bfloat16 for Fp8.
int mlx_paged_kv_write_factory_rejects_new_k_dtype_fp8() {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  array k_pool(Shape{4, 4, 4, 16, 16}, uint8, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, uint8, nullptr, {});
  // new_k/new_v float16 — pair agrees, but Phase 1 contract demands
  // bfloat16 io for FP8 cache.
  array new_k(Shape{2, 4, 64}, float16, nullptr, {});
  array new_v(Shape{2, 4, 64}, float16, nullptr, {});
  array slot_mapping(Shape{2}, int64, nullptr, {});
  return call_paged_kv_write_expecting_throw(
      k_pool, v_pool, new_k, new_v, slot_mapping, KvDtype::Fp8, /*x_pack=*/16);
}

// =============================================================================
// Phase 1 review-round-4 dtype-mismatch negative tests for paged_attention.
//
// Round 4 finding C: the factory validated scale dtype, q rank/shape,
// pool layout, and index-buffer dtypes, but never validated `q.dtype()`
// or `k_pool`/`v_pool` dtypes against `kv_dtype`. These helpers
// construct factory inputs that are well-formed EXCEPT for one dtype
// slot, then assert `std::invalid_argument` is thrown.
// =============================================================================

/// q dtype is float32 instead of the expected bfloat16 for Bf16.
int mlx_paged_attention_factory_rejects_q_dtype_bf16() {
  using namespace mlx::core;
  // q float32 — disagrees with the bf16 io dtype implied by Bf16 kv_dtype.
  array q(Shape{1, 8, 64}, float32, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// q dtype is float16 instead of the expected bfloat16 for Fp8.
int mlx_paged_attention_factory_rejects_q_dtype_fp8() {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  // FP8 cache: q must be bfloat16 (Phase 1 contract). Pass float16 to
  // trigger the throw. Pools at FP8-correct (uint8, x_pack=16) so the
  // q-dtype check is the only mismatch.
  array q(Shape{1, 8, 64}, float16, nullptr, {});
  array k_pool(Shape{4, 4, 4, 16, 16}, uint8, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, uint8, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens, KvDtype::Fp8);
}

/// k_pool dtype is uint8 instead of the expected bfloat16 for Bf16.
int mlx_paged_attention_factory_rejects_k_pool_dtype_bf16() {
  using namespace mlx::core;
  // q correct, k_pool/v_pool wrong → kv_dtype check rejects.
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, uint8, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, uint8, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens);
}

/// k_pool dtype is bfloat16 instead of the expected uint8 for Fp8.
int mlx_paged_attention_factory_rejects_k_pool_dtype_fp8() {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  // FP8 expects pools = uint8, x_pack = 16. Pass bfloat16 to trigger.
  array k_pool(Shape{4, 4, 4, 16, 16}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  return call_paged_attention_expecting_throw(
      q, k_pool, v_pool, block_table, seq_lens, KvDtype::Fp8);
}

// =============================================================================
// Phase 1 review-round-6 finding: GQA head-group divisibility.
//
// The Metal kernel computes
//   num_queries_per_kv = num_heads / num_kv_heads
//   kv_head_idx        = head_idx / num_queries_per_kv
// (paged_attention.metal:839-840). Passing num_kv_heads=0,
// num_q_heads<num_kv_heads, or num_q_heads not divisible by
// num_kv_heads turns a structurally shape-consistent call into a GPU
// fault (division by zero) or an out-of-pool K/V read. The factory now
// rejects these cases up front. Each helper builds shape-consistent
// well-formed inputs that match the supplied (num_q_heads, num_kv_heads)
// scalar state, and asserts `std::invalid_argument` is thrown.
// =============================================================================

/// num_kv_heads = 0 must be rejected (division-by-zero risk in
/// `num_queries_per_kv = num_heads / num_kv_heads`).
int mlx_paged_attention_factory_rejects_zero_kv_heads() {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  // Build q/k_pool/v_pool whose shapes match num_q_heads=8 and
  // num_kv_heads=0. We let the factory throw on the GQA check before
  // any other shape check fires (the GQA validation runs early).
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 0, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 0, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);
  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/0,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/0,
        /*head_size=*/64,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// num_q_heads (2) < num_kv_heads (4) must be rejected. The kernel
/// would compute `num_queries_per_kv = 2 / 4 = 0` (integer division)
/// and then divide head_idx by zero on the kv_head_idx line.
int mlx_paged_attention_factory_rejects_q_heads_less_than_kv_heads() {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  // Build inputs consistent with num_q_heads=2, num_kv_heads=4 so the
  // GQA check is the only mismatch; q.shape(1)=2, pool dim 1 = 4.
  array q(Shape{1, 2, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);
  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/0,
        /*block_size=*/16,
        /*num_q_heads=*/2,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// num_q_heads (6) not divisible by num_kv_heads (4) must be rejected.
/// 6 % 4 = 2, so the last two q-heads would compute kv_head_idx = 4, 5
/// — past the end of the 4-entry KV-head dimension, reading out-of-pool
/// memory.
int mlx_paged_attention_factory_rejects_indivisible_grouping() {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  array q(Shape{1, 6, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);
  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/0,
        /*block_size=*/16,
        /*num_q_heads=*/6,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// block_size = 0 must be rejected. Without the factory check, the
/// pool shape equality (`k_pool.shape(3) == block_size`) accepts a
/// zero-sized pool block dimension when `block_size=0`, and on
/// `eval_gpu` the runtime bounds check would compute `(s + block_size -
/// 1) / block_size` and divide by zero in host code BEFORE the later
/// `max_context_len <= 0` guard could reject. Build a structurally
/// consistent zero-block-size graph and assert the factory throws.
int mlx_paged_attention_factory_rejects_zero_block_size() {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  // Pool's block_size dim (k_pool.shape(3) and v_pool.shape(3)) is 0 to
  // satisfy the structural shape-equality check (`pool.shape(3) ==
  // block_size`) — that is precisely the path the bug exploits.
  array q(Shape{1, 8, 64}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 8, 0, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 64, 0}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);
  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/0,
        /*block_size=*/0,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// head_size = 0 must be rejected. The Metal kernel uses head_size as
/// a grid extent and indexing stride; a zero-sized inner dim would set
/// up a degenerate Metal launch. Mirrors `paged_kv_write`'s identical
/// `head_size > 0` check for symmetry.
int mlx_paged_attention_factory_rejects_zero_head_size() {
  using namespace mlx::core;
  using namespace mlx::core::fast;
  // q.shape(2) and v_pool.shape(2) are 0 to keep the structural
  // shape-equality checks satisfied — head_size flows through both
  // q's trailing dim and v_pool's interior dim.
  array q(Shape{1, 8, 0}, bfloat16, nullptr, {});
  array k_pool(Shape{4, 4, 0, 16, 8}, bfloat16, nullptr, {});
  array v_pool(Shape{4, 4, 0, 16}, bfloat16, nullptr, {});
  array block_table(Shape{1, 4}, int32, nullptr, {});
  array seq_lens(Shape{1}, int32, nullptr, {});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);
  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/0,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/0,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

} // extern "C"

// =============================================================================
// Phase 1 review-round-4 finding A: compile-cached path slot-bounds test.
//
// `paged_kv_write`'s factory eval-checks slot bounds, but
// `mlx::core::compile` skips the factory on cache hits and routes
// runtime `slot_mapping` straight into `eval_gpu`. The fix is to also
// check bounds inside `PagedKVWrite::eval_gpu` (which fires on every
// runtime call, including the compile-cached path). This helper
// exercises that path: compile a function, call it once with a valid
// slot_mapping (cache miss → factory check passes), then call it again
// with an out-of-range slot_mapping. The second call MUST throw from
// `eval_gpu` because the cached graph bypasses the factory.
// =============================================================================

namespace {

/// Trace function for the compile-cached out-of-bounds slot test.
/// Mirrors `paged_kv_write_trace_fn` but is its own static so the
/// compile cache key is independent of the existing test's cache.
std::vector<mlx::core::array> paged_kv_write_oob_trace_fn(
    const std::vector<mlx::core::array>& inputs) {
  using namespace mlx::core::fast;
  if (inputs.size() != 7) {
    throw std::runtime_error("paged_kv_write_oob_trace_fn: expected 7 inputs");
  }
  auto out = paged_kv_write(
      inputs[0],
      inputs[1],
      inputs[2],
      inputs[3],
      inputs[4],
      inputs[5],
      inputs[6],
      /*block_size=*/16,
      /*num_kv_heads=*/4,
      /*head_size=*/64,
      /*x_pack=*/8,
      KvDtype::Bf16,
      /*s=*/{});
  return {out.first, out.second};
}

} // namespace

extern "C" {

/// Compiled-path slot-bounds regression test.
///
/// 1. Compile `paged_kv_write_oob_trace_fn`.
/// 2. Call it once with valid slot_mapping = [0, 16] — cache miss
///    triggers the factory's eval-based bounds check (passes).
/// 3. Call it again with the SAME shapes/dtypes but slot_mapping =
///    [0, num_blocks * block_size] = [0, 64] — out of range. Cache HIT
///    bypasses the factory. The bounds check inside
///    `PagedKVWrite::eval_gpu` MUST throw `std::invalid_argument`
///    during the second eval.
///
/// Layout matches the existing compile-trace smoke helper:
///   block_size=16, num_kv_heads=4, head_size=64, x_pack=8, Bf16.
///   k_pool: [4, 4, 8, 16, 8] bf16
///   v_pool: [4, 4, 64, 16] bf16
///   new_k:  [2, 4, 64] bf16
///   new_v:  [2, 4, 64] bf16
///   slot_mapping: [2] int64
///
/// Return codes:
///   1   — second-call eval threw `std::invalid_argument` (fix
///         working — the compile-cached path is bounds-checked).
///   0   — second-call eval did NOT throw (regression — the
///         out-of-range slot reached the kernel, which would write
///         past the K/V pool).
///  -1   — internal/setup error (first call failed unexpectedly,
///         compile produced 0 outputs, etc.).
///  -3   — Metal not available; eval-based verification skipped.
int mlx_paged_kv_write_compile_cached_oob_throws() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  // Build REAL data-backed inputs. Layout matches the existing
  // compile-trace smoke test.
  const int kBlockSize = 16;
  const int kNumKvHeads = 4;
  const int kHeadSize = 64;
  const int kXPack = 8;
  const int kNumBlocks = 4;
  const int kNumTokens = 2;

  const size_t k_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      (kHeadSize / kXPack) * kBlockSize * kXPack;
  const size_t v_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      kHeadSize * kBlockSize;
  const size_t new_kv_elems =
      static_cast<size_t>(kNumTokens) * kNumKvHeads * kHeadSize;

  std::vector<uint16_t> k_pool_host(k_pool_elems, 0);
  std::vector<uint16_t> v_pool_host(v_pool_elems, 0);
  std::vector<uint16_t> new_k_host(new_kv_elems, 0x3FC0); // bf16(1.5)
  std::vector<uint16_t> new_v_host(new_kv_elems, 0x4040); // bf16(3.0)

  auto bf16_arr = [](const std::vector<uint16_t>& src, Shape shape) {
    auto* p = reinterpret_cast<const bfloat16_t*>(src.data());
    return array(p, std::move(shape), bfloat16);
  };
  auto i64_arr = [](const std::vector<int64_t>& src, Shape shape) {
    return array(src.data(), std::move(shape), int64);
  };

  Shape k_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize / kXPack, kBlockSize,
      kXPack};
  Shape v_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize, kBlockSize};
  Shape new_kv_shape{kNumTokens, kNumKvHeads, kHeadSize};

  array k_pool = bf16_arr(k_pool_host, k_pool_shape);
  array v_pool = bf16_arr(v_pool_host, v_pool_shape);
  array new_k = bf16_arr(new_k_host, new_kv_shape);
  array new_v = bf16_arr(new_v_host, new_kv_shape);
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  auto compiled = mlx::core::compile(&paged_kv_write_oob_trace_fn);

  // First call: valid slot_mapping = [0, 16]. Both slots within pool
  // capacity (4*16 = 64).
  std::vector<int64_t> good_slots = {0, 16};
  array slot_mapping_good = i64_arr(good_slots, Shape{kNumTokens});

  std::vector<array> inputs1{
      k_pool, v_pool, new_k, new_v, slot_mapping_good, k_scale, v_scale};

  std::vector<array> outputs1;
  try {
    outputs1 = compiled(inputs1);
    if (outputs1.size() != 2) {
      return -1;
    }
    mlx::core::eval(outputs1[0], outputs1[1]);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[compile_cached_oob] first call (valid slots) threw unexpectedly: %s\n",
        e.what());
    return -1;
  }

  // Second call: slot_mapping = [0, 64]. Pool capacity = 4*16 = 64,
  // so slot 64 is out of range (valid slots are 0..63).
  // Cache HIT: factory's eval-based check is BYPASSED. eval_gpu's
  // own bounds check MUST throw.
  std::vector<int64_t> bad_slots = {
      0, static_cast<int64_t>(kNumBlocks) * static_cast<int64_t>(kBlockSize)};
  array slot_mapping_bad = i64_arr(bad_slots, Shape{kNumTokens});

  std::vector<array> inputs2{
      k_pool, v_pool, new_k, new_v, slot_mapping_bad, k_scale, v_scale};

  std::vector<array> outputs2;
  try {
    outputs2 = compiled(inputs2);
    if (outputs2.size() != 2) {
      return -1;
    }
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[compile_cached_oob] second call (compile) threw early: %s\n",
        e.what());
    return -1;
  }

  // The throw should fire on eval, not on the compiled() lambda call
  // itself (the lambda just stitches the graph; eval invokes
  // `eval_gpu`).
  try {
    mlx::core::eval(outputs2[0], outputs2[1]);
  } catch (const std::invalid_argument& /*e*/) {
    return 1; // SUCCESS — eval_gpu caught the out-of-range slot.
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[compile_cached_oob] second-call eval threw a non-invalid_argument: %s\n",
        e.what());
    return 0;
  }
  fprintf(
      stderr,
      "[compile_cached_oob] second-call eval did NOT throw — the "
      "compile-cached path is missing its slot-bounds check\n");
  return 0;
}

} // extern "C"

// =============================================================================
// Phase 1 review-round-5 finding: PagedAttention::eval_gpu must
// runtime-bounds-check `seq_lens` and `block_table` contents.
//
// The Metal kernel reinterprets seq_lens as `uint32_t*` (so a
// negative int32 value sneaks in as a huge unsigned context length)
// and uses block_table entries directly as `physical_block_number *
// kv_block_stride` for K/V pool reads. The factory does only structural
// validation (rank/shape/dtype), and `mlx::core::compile`'s cached
// re-traces bypass the factory entirely. The fix puts the runtime
// check in `PagedAttention::eval_gpu` so it fires on every replay.
// These helpers exercise that check end-to-end.
// =============================================================================

namespace {

/// Layout shared by the eval_gpu rejection helpers. Matches the
/// `paged_kv_write` compile-trace test: block_size=16, num_kv_heads=4,
/// head_size=64, x_pack=8, num_q_heads=8, kv_dtype=Bf16, with a
/// `[4, 4, 8, 16, 8]` K-pool / `[4, 4, 64, 16]` V-pool / `[1, 4]`
/// block_table / `[1]` seq_lens layout. Builds REAL data-backed arrays
/// from the supplied seq_lens and block_table contents and invokes
/// `paged_attention(...)` then evals the result. Returns 1 if eval
/// throws `std::invalid_argument`, 0 if no throw, -1 on setup error,
/// -3 if Metal is unavailable.
int call_paged_attention_eval_expecting_throw(
    const std::vector<int32_t>& seq_lens_host,
    const std::vector<int32_t>& block_table_host,
    int max_blocks_per_seq) {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  const int kBlockSize = 16;
  const int kNumKvHeads = 4;
  const int kHeadSize = 64;
  const int kXPack = 8;
  const int kNumBlocks = 4;
  const int kNumQHeads = 8;
  const int kNumSeqs = 1;

  if (seq_lens_host.size() != static_cast<size_t>(kNumSeqs)) {
    return -1;
  }
  if (block_table_host.size() !=
      static_cast<size_t>(kNumSeqs) * static_cast<size_t>(max_blocks_per_seq)) {
    return -1;
  }

  const size_t k_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      (kHeadSize / kXPack) * kBlockSize * kXPack;
  const size_t v_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      kHeadSize * kBlockSize;
  const size_t q_elems =
      static_cast<size_t>(kNumSeqs) * kNumQHeads * kHeadSize;

  std::vector<uint16_t> k_pool_host(k_pool_elems, 0);
  std::vector<uint16_t> v_pool_host(v_pool_elems, 0);
  std::vector<uint16_t> q_host(q_elems, 0x3F80); // bf16(1.0)

  auto bf16_arr = [](const std::vector<uint16_t>& src, Shape shape) {
    auto* p = reinterpret_cast<const bfloat16_t*>(src.data());
    return array(p, std::move(shape), bfloat16);
  };
  auto i32_arr = [](const std::vector<int32_t>& src, Shape shape) {
    return array(src.data(), std::move(shape), int32);
  };

  array q = bf16_arr(q_host, Shape{kNumSeqs, kNumQHeads, kHeadSize});
  array k_pool = bf16_arr(
      k_pool_host,
      Shape{kNumBlocks, kNumKvHeads, kHeadSize / kXPack, kBlockSize, kXPack});
  array v_pool = bf16_arr(
      v_pool_host, Shape{kNumBlocks, kNumKvHeads, kHeadSize, kBlockSize});
  array block_table =
      i32_arr(block_table_host, Shape{kNumSeqs, max_blocks_per_seq});
  array seq_lens = i32_arr(seq_lens_host, Shape{kNumSeqs});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  array attn_out = paged_attention(
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      k_scale,
      v_scale,
      /*scale=*/0.125f,
      /*softcap=*/0.0f,
      /*sliding_window=*/0,
      /*block_size=*/kBlockSize,
      /*num_q_heads=*/kNumQHeads,
      /*num_kv_heads=*/kNumKvHeads,
      /*head_size=*/kHeadSize,
      KvDtype::Bf16,
      StreamOrDevice{});

  try {
    mlx::core::eval(attn_out);
  } catch (const std::invalid_argument& /*e*/) {
    return 1;
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_attention_eval] eval threw a non-invalid_argument: %s\n",
        e.what());
    return 0;
  }
  return 0;
}

/// Trace function for the compile-cached `paged_attention` OOB test.
/// Hard-coded scalars match `call_paged_attention_eval_expecting_throw`
/// so the cache key is consistent across compile+replay.
std::vector<mlx::core::array> paged_attention_oob_trace_fn(
    const std::vector<mlx::core::array>& inputs) {
  using namespace mlx::core::fast;
  if (inputs.size() != 7) {
    throw std::runtime_error("paged_attention_oob_trace_fn: expected 7 inputs");
  }
  auto out = paged_attention(
      inputs[0], // q
      inputs[1], // k_pool
      inputs[2], // v_pool
      inputs[3], // block_table
      inputs[4], // seq_lens
      inputs[5], // k_scale
      inputs[6], // v_scale
      /*scale=*/0.125f,
      /*softcap=*/0.0f,
      /*sliding_window=*/0,
      /*block_size=*/16,
      /*num_q_heads=*/8,
      /*num_kv_heads=*/4,
      /*head_size=*/64,
      KvDtype::Bf16,
      /*s=*/{});
  return {out};
}

} // namespace

extern "C" {

/// seq_lens with a negative entry must be rejected by eval_gpu's
/// runtime bounds check.
int mlx_paged_attention_eval_gpu_rejects_negative_seq_len() {
  // max_blocks_per_seq=4. seq_lens=[-1] is an invalid (negative)
  // context length; the kernel would reinterpret it as a huge
  // unsigned value.
  std::vector<int32_t> seq_lens_host = {-1};
  std::vector<int32_t> block_table_host = {0, 0, 0, 0};
  return call_paged_attention_eval_expecting_throw(
      seq_lens_host, block_table_host, /*max_blocks_per_seq=*/4);
}

/// seq_lens larger than `max_blocks_per_seq * block_size` must be
/// rejected by eval_gpu's runtime bounds check.
int mlx_paged_attention_eval_gpu_rejects_oversized_seq_len() {
  // max_blocks_per_seq=4, block_size=16 → bound = 64. seq_lens=[65]
  // exceeds the bound; the kernel would read past the row's
  // block-table region.
  std::vector<int32_t> seq_lens_host = {4 * 16 + 1};
  std::vector<int32_t> block_table_host = {0, 0, 0, 0};
  return call_paged_attention_eval_expecting_throw(
      seq_lens_host, block_table_host, /*max_blocks_per_seq=*/4);
}

/// block_table with a negative entry (within the "used" region for
/// that row) must be rejected by eval_gpu's runtime bounds check.
int mlx_paged_attention_eval_gpu_rejects_negative_block_id() {
  // seq_lens=[16] → num_used_blocks = ceil(16/16) = 1 → only j=0 is
  // checked. block_table[0,0]=-2 is out of [0, num_blocks=4).
  std::vector<int32_t> seq_lens_host = {16};
  std::vector<int32_t> block_table_host = {-2, 0, 0, 0};
  return call_paged_attention_eval_expecting_throw(
      seq_lens_host, block_table_host, /*max_blocks_per_seq=*/4);
}

/// block_table with an entry == num_blocks (one past valid) must be
/// rejected by eval_gpu's runtime bounds check.
int mlx_paged_attention_eval_gpu_rejects_oob_block_id() {
  // seq_lens=[16] → num_used_blocks = 1 → only j=0 is checked.
  // num_blocks=4, so block_table[0,0]=4 is one past valid.
  std::vector<int32_t> seq_lens_host = {16};
  std::vector<int32_t> block_table_host = {4, 0, 0, 0};
  return call_paged_attention_eval_expecting_throw(
      seq_lens_host, block_table_host, /*max_blocks_per_seq=*/4);
}

/// Compile a `paged_attention`-emitting function, call it once with
/// valid inputs (cache miss → factory check passes, eval_gpu check
/// passes on valid block ids), then call it again with the SAME shapes
/// but an out-of-range block id. Cache HIT bypasses the factory; the
/// eval_gpu runtime bounds check MUST throw on the second eval.
///
/// Layout matches `mlx_paged_attention_eval_gpu_rejects_*`.
///
/// Return codes:
///   1   — second-call eval threw `std::invalid_argument` (fix
///         working — the compile-cached path is bounds-checked).
///   0   — second-call eval did NOT throw (regression — the
///         out-of-range block id reached the kernel).
///  -1   — internal/setup error (first call failed unexpectedly).
///  -3   — Metal not available; eval-based verification skipped.
int mlx_paged_attention_compile_cached_oob_throws() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  const int kBlockSize = 16;
  const int kNumKvHeads = 4;
  const int kHeadSize = 64;
  const int kXPack = 8;
  const int kNumBlocks = 4;
  const int kNumQHeads = 8;
  const int kNumSeqs = 1;
  const int kMaxBlocksPerSeq = 4;

  const size_t k_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      (kHeadSize / kXPack) * kBlockSize * kXPack;
  const size_t v_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      kHeadSize * kBlockSize;
  const size_t q_elems =
      static_cast<size_t>(kNumSeqs) * kNumQHeads * kHeadSize;

  std::vector<uint16_t> k_pool_host(k_pool_elems, 0);
  std::vector<uint16_t> v_pool_host(v_pool_elems, 0);
  std::vector<uint16_t> q_host(q_elems, 0x3F80); // bf16(1.0)

  auto bf16_arr = [](const std::vector<uint16_t>& src, Shape shape) {
    auto* p = reinterpret_cast<const bfloat16_t*>(src.data());
    return array(p, std::move(shape), bfloat16);
  };
  auto i32_arr = [](const std::vector<int32_t>& src, Shape shape) {
    return array(src.data(), std::move(shape), int32);
  };

  Shape k_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize / kXPack, kBlockSize,
      kXPack};
  Shape v_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize, kBlockSize};
  Shape q_shape{kNumSeqs, kNumQHeads, kHeadSize};

  array q = bf16_arr(q_host, q_shape);
  array k_pool = bf16_arr(k_pool_host, k_pool_shape);
  array v_pool = bf16_arr(v_pool_host, v_pool_shape);
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  auto compiled = mlx::core::compile(&paged_attention_oob_trace_fn);

  // First call: valid block_table=[0, 0, 0, 0], seq_lens=[16]. Block
  // id 0 is in [0, num_blocks=4) and seq_len=16 is within
  // max_blocks_per_seq * block_size = 64.
  std::vector<int32_t> good_seq_lens = {16};
  std::vector<int32_t> good_block_table = {0, 0, 0, 0};
  array seq_lens_good = i32_arr(good_seq_lens, Shape{kNumSeqs});
  array block_table_good = i32_arr(
      good_block_table, Shape{kNumSeqs, kMaxBlocksPerSeq});

  std::vector<array> inputs1{
      q, k_pool, v_pool, block_table_good, seq_lens_good, k_scale, v_scale};

  std::vector<array> outputs1;
  try {
    outputs1 = compiled(inputs1);
    if (outputs1.size() != 1) {
      return -1;
    }
    mlx::core::eval(outputs1[0]);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_attention_compile_cached_oob] first call (valid inputs) "
        "threw unexpectedly: %s\n",
        e.what());
    return -1;
  }

  // Second call: same shapes, but block_table[0,0] = num_blocks = 4
  // (one past valid). Cache HIT bypasses the factory; eval_gpu's
  // bounds check MUST throw.
  std::vector<int32_t> bad_block_table = {kNumBlocks, 0, 0, 0};
  array block_table_bad = i32_arr(
      bad_block_table, Shape{kNumSeqs, kMaxBlocksPerSeq});
  std::vector<array> inputs2{
      q, k_pool, v_pool, block_table_bad, seq_lens_good, k_scale, v_scale};

  std::vector<array> outputs2;
  try {
    outputs2 = compiled(inputs2);
    if (outputs2.size() != 1) {
      return -1;
    }
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_attention_compile_cached_oob] second call (compile) threw "
        "early: %s\n",
        e.what());
    return -1;
  }

  // The throw should fire on eval, not on the compiled() lambda call
  // itself.
  try {
    mlx::core::eval(outputs2[0]);
  } catch (const std::invalid_argument& /*e*/) {
    return 1; // SUCCESS — eval_gpu caught the out-of-range block id.
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_attention_compile_cached_oob] second-call eval threw a "
        "non-invalid_argument: %s\n",
        e.what());
    return 0;
  }
  fprintf(
      stderr,
      "[paged_attention_compile_cached_oob] second-call eval did NOT throw "
      "— the compile-cached path is missing its block-id bounds check\n");
  return 0;
}

} // extern "C"

// =============================================================================
// Phase 1 review-round-8 finding: factory must reject non-row-contiguous
// or nonzero-offset views for ALL inputs.
//
// MLX arrays support strided / sliced / transposed views with valid
// logical shapes/dtypes. The Metal kernels (and the Rust shim) read
// each input as a dense row-major buffer, and the dispatch path only
// forwards a single offset for q/out (and slot_mapping/new_k/new_v on
// the write side). Pool-side and metadata-side inputs (k_pool, v_pool,
// block_table, seq_lens) are passed as bare buffers — a sliced view
// would silently alias to offset 0 in the backing allocation.
//
// Reject early at the factory; tests below assert the rejection fires
// for transposed q / sliced k_pool / sliced block_table / sliced
// seq_lens. They use real data-backed source arrays (so MLX's
// `slice` / `transpose` ops actually produce non-row-contiguous /
// nonzero-offset views) and are gated on `metal::is_available()` because
// `eval()` is needed to materialize the slice/transpose results.
// =============================================================================

namespace {

/// Build a small bf16 array of zeros with the given shape, eval it, and
/// return it. The `eval()` is needed because `slice` and `transpose`
/// inspect the input's flags / strides during their own eval — but the
/// FACTORY checks `flags()` / `offset()` which are propagated by the
/// downstream primitive's output construction. To get a row_contiguous=
/// false view with offset != 0 we must actually evaluate the slice (so
/// it sets up its descriptor). That requires Metal.
mlx::core::array make_bf16_zeros(mlx::core::Shape shape) {
  using namespace mlx::core;
  size_t size = 1;
  for (auto d : shape) {
    size *= static_cast<size_t>(d);
  }
  std::vector<uint16_t> host(size, 0);
  auto* p = reinterpret_cast<const bfloat16_t*>(host.data());
  return array(p, std::move(shape), bfloat16);
}

mlx::core::array make_int32_zeros(mlx::core::Shape shape) {
  using namespace mlx::core;
  size_t size = 1;
  for (auto d : shape) {
    size *= static_cast<size_t>(d);
  }
  std::vector<int32_t> host(size, 0);
  return array(host.data(), std::move(shape), int32);
}

} // namespace

extern "C" {

/// Verify the public `paged_kv_write(...)` factory rejects a k_pool
/// that is non-row-contiguous (taken via `slice(..., {1,...}, {3,...})`
/// — a sub-region of the original 4-block pool). Returns 1 on
/// `std::invalid_argument`, 0 otherwise. Returns -3 if Metal is
/// unavailable (slice eval requires it).
int mlx_paged_kv_write_factory_rejects_non_contiguous_k_pool() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  // Build a 5-block pool, then slice off the leading block. The result
  // has shape [4, ...] but its `offset()` is nonzero AND its
  // row_contiguous flag is false (slicing along axis 0 of a 5-block
  // pool produces a strided view starting at offset = 1 *
  // strides[0] in the backing buffer).
  array k_pool_full = make_bf16_zeros(Shape{5, 4, 8, 16, 8});
  // Force the source array to be evaluated so the slice has data to
  // anchor on.
  eval(k_pool_full);
  array k_pool_sliced = mlx::core::slice(
      k_pool_full,
      Shape{1, 0, 0, 0, 0},
      Shape{5, 4, 8, 16, 8},
      Shape{1, 1, 1, 1, 1});
  eval(k_pool_sliced);

  // Sanity: confirm the slice is actually non-trivial. We don't return
  // a hard failure here (the factory check is what's under test) — but
  // log if MLX returned a row-contiguous view (e.g. on some future
  // optimization), because then the test would not exercise the
  // intended path.
  if (k_pool_sliced.flags().row_contiguous && k_pool_sliced.offset() == 0) {
    fprintf(
        stderr,
        "[non_contig_k_pool] WARNING: slice produced a row-contiguous "
        "view at offset 0; test will not exercise the intended factory "
        "rejection path.\n");
  }

  // Well-formed companions for the slice we want to reject.
  array v_pool = make_bf16_zeros(Shape{4, 4, 64, 16});
  array new_k = make_bf16_zeros(Shape{2, 4, 64});
  array new_v = make_bf16_zeros(Shape{2, 4, 64});
  array slot_mapping(std::vector<int64_t>{0, 1}.data(), Shape{2}, int64);
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  try {
    paged_kv_write(
        k_pool_sliced,
        v_pool,
        new_k,
        new_v,
        slot_mapping,
        k_scale,
        v_scale,
        /*block_size=*/16,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*x_pack=*/8,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// Verify the public `paged_attention(...)` factory rejects a q that
/// has been transposed (row_contiguous == false). Transposes the
/// `[1, 8, 64]` q to `[8, 1, 64]` then transposes back to `[1, 8, 64]`
/// using a non-trivial axis order, producing a logically correct
/// shape but a strided view of the original buffer.
int mlx_paged_attention_factory_rejects_non_contiguous_q() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  // Build q with shape [1, 8, 64], then transpose to [8, 64, 1] then
  // back to [1, 8, 64] — the result is a non-row-contiguous view. We
  // can't simply transpose to [1, 8, 64] (that's the identity) so we
  // round-trip via swapping middle/last axes. Equivalent: swap axes
  // (1, 2) and back, but use a non-trivial permutation that's
  // observable as non-row-contiguous after the second swap.
  //
  // Simpler: build q' with shape [64, 8, 1] (row-contiguous) and
  // transpose to [1, 8, 64]. The result has the right logical shape
  // but the underlying buffer is in [64, 8, 1] order → non-row-
  // contiguous when viewed as [1, 8, 64].
  array q_alt = make_bf16_zeros(Shape{64, 8, 1});
  eval(q_alt);
  // transpose([2, 1, 0]) yields shape [1, 8, 64]. The view's strides
  // mirror the input buffer, NOT the new logical layout, so
  // row_contiguous == false.
  array q_t = mlx::core::transpose(q_alt, std::vector<int>{2, 1, 0});
  eval(q_t);

  if (q_t.flags().row_contiguous) {
    fprintf(
        stderr,
        "[non_contig_q] WARNING: transpose produced a row-contiguous "
        "view; test will not exercise the intended factory rejection "
        "path.\n");
  }

  array k_pool = make_bf16_zeros(Shape{4, 4, 8, 16, 8});
  array v_pool = make_bf16_zeros(Shape{4, 4, 64, 16});
  array block_table = make_int32_zeros(Shape{1, 4});
  array seq_lens = make_int32_zeros(Shape{1});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  try {
    paged_attention(
        q_t,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/0,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// Verify the public `paged_attention(...)` factory rejects a
/// block_table sliced along dim 0 (e.g. `block_table[1:3]` from a
/// `[4, 4]` original). The slice has shape `[2, 4]` but starts partway
/// into the backing buffer — the kernel would read from the wrong
/// region.
int mlx_paged_attention_factory_rejects_non_contiguous_block_table() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  // Build a [4, 4] int32 block_table, slice off rows [1:3] (yielding
  // [2, 4] with offset != 0). Eval to materialize.
  array bt_full = make_int32_zeros(Shape{4, 4});
  eval(bt_full);
  array bt_sliced = mlx::core::slice(
      bt_full,
      Shape{1, 0},
      Shape{3, 4},
      Shape{1, 1});
  eval(bt_sliced);

  if (bt_sliced.flags().row_contiguous && bt_sliced.offset() == 0) {
    fprintf(
        stderr,
        "[non_contig_block_table] WARNING: slice produced a "
        "row-contiguous view at offset 0; test will not exercise the "
        "intended factory rejection path.\n");
  }

  // q must agree with bt_sliced.shape(0) = 2. Everything else is
  // well-formed.
  array q = make_bf16_zeros(Shape{2, 8, 64});
  array k_pool = make_bf16_zeros(Shape{4, 4, 8, 16, 8});
  array v_pool = make_bf16_zeros(Shape{4, 4, 64, 16});
  array seq_lens = make_int32_zeros(Shape{2});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        bt_sliced,
        seq_lens,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/0,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// Verify the public `paged_attention(...)` factory rejects a seq_lens
/// taken via `seq_lens[1:]` — a 1-element slice with nonzero offset.
int mlx_paged_attention_factory_rejects_non_contiguous_seq_lens() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  // Build a [2] int32 seq_lens, take the trailing element.
  array sl_full = make_int32_zeros(Shape{2});
  eval(sl_full);
  array sl_sliced = mlx::core::slice(
      sl_full, Shape{1}, Shape{2}, Shape{1});
  eval(sl_sliced);

  if (sl_sliced.flags().row_contiguous && sl_sliced.offset() == 0) {
    fprintf(
        stderr,
        "[non_contig_seq_lens] WARNING: slice produced a "
        "row-contiguous view at offset 0; test will not exercise the "
        "intended factory rejection path.\n");
  }

  // Everything else well-formed; q matches sl_sliced.shape(0) = 1.
  array q = make_bf16_zeros(Shape{1, 8, 64});
  array k_pool = make_bf16_zeros(Shape{4, 4, 8, 16, 8});
  array v_pool = make_bf16_zeros(Shape{4, 4, 64, 16});
  array block_table = make_int32_zeros(Shape{1, 4});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  try {
    paged_attention(
        q,
        k_pool,
        v_pool,
        block_table,
        sl_sliced,
        k_scale,
        v_scale,
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*sliding_window=*/0,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

/// Regression for the production FFI bridge: it must not hide invalid
/// metadata views by wrapping block_table/seq_lens in `contiguous(...)`.
/// Metadata is host-read inside `PagedAttention::eval_gpu`, so the bridge
/// preserves the adapter-owned metadata arrays and lets the factory reject
/// non-row-contiguous / nonzero-offset metadata. Returns 1 when the bridge
/// rejects the sliced block_table, 0 if it incorrectly returns a lazy output,
/// and -3 if Metal is unavailable for slice materialization.
int mlx_paged_attention_forward_rejects_non_contiguous_metadata() {
  using namespace mlx::core;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  array bt_full = make_int32_zeros(Shape{2, 4});
  eval(bt_full);
  array bt_sliced = mlx::core::slice(
      bt_full,
      Shape{1, 0},
      Shape{2, 4},
      Shape{1, 1});
  eval(bt_sliced);

  array q = make_bf16_zeros(Shape{1, 8, 64});
  array k_pool = make_bf16_zeros(Shape{4, 4, 8, 16, 8});
  array v_pool = make_bf16_zeros(Shape{4, 4, 64, 16});
  std::vector<int32_t> seq_host = {1};
  array seq_lens(seq_host.data(), Shape{1}, int32);
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  auto* out_ptr = mlx_paged_attention_forward(
      reinterpret_cast<mlx_array*>(&q),
      reinterpret_cast<mlx_array*>(&k_pool),
      reinterpret_cast<mlx_array*>(&v_pool),
      reinterpret_cast<mlx_array*>(&bt_sliced),
      reinterpret_cast<mlx_array*>(&seq_lens),
      reinterpret_cast<mlx_array*>(&k_scale),
      reinterpret_cast<mlx_array*>(&v_scale),
      /*scale=*/0.125f,
      /*softcap=*/0.0f,
      /*sliding_window=*/0,
      /*block_size=*/16,
      /*num_q_heads=*/8,
      /*num_kv_heads=*/4,
      /*head_size=*/64,
      /*kv_dtype_raw=*/1);

  if (out_ptr == nullptr) {
    return 1;
  }
  delete reinterpret_cast<array*>(out_ptr);
  return 0;
}

/// Valid bridge smoke: materialized, row-contiguous metadata should survive
/// the FFI wrapper and evaluate later as a lazy paged-attention output.
/// Returns 1 on success, 0 on eval-time metadata validation failure, -1 on
/// internal setup/dispatch errors, and -3 if Metal is unavailable.
int mlx_paged_attention_forward_eval_accepts_materialized_metadata() {
  using namespace mlx::core;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  array q = make_bf16_zeros(Shape{2, 8, 64});
  array k_pool = make_bf16_zeros(Shape{4, 4, 8, 16, 8});
  array v_pool = make_bf16_zeros(Shape{4, 4, 64, 16});
  std::vector<int32_t> blocks = {
      0, -1, -1, -1,
      0,  1, -1, -1,
  };
  std::vector<int32_t> seqs = {1, 17};
  array block_table(blocks.data(), Shape{2, 4}, int32);
  array seq_lens(seqs.data(), Shape{2}, int32);
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  auto* out_ptr = mlx_paged_attention_forward(
      reinterpret_cast<mlx_array*>(&q),
      reinterpret_cast<mlx_array*>(&k_pool),
      reinterpret_cast<mlx_array*>(&v_pool),
      reinterpret_cast<mlx_array*>(&block_table),
      reinterpret_cast<mlx_array*>(&seq_lens),
      reinterpret_cast<mlx_array*>(&k_scale),
      reinterpret_cast<mlx_array*>(&v_scale),
      /*scale=*/0.125f,
      /*softcap=*/0.0f,
      /*sliding_window=*/0,
      /*block_size=*/16,
      /*num_q_heads=*/8,
      /*num_kv_heads=*/4,
      /*head_size=*/64,
      /*kv_dtype_raw=*/1);

  if (out_ptr == nullptr) {
    return -1;
  }

  auto* out = reinterpret_cast<array*>(out_ptr);
  try {
    eval(*out);
  } catch (const std::invalid_argument& e) {
    fprintf(stderr, "[paged_attention_forward_valid_metadata] %s\n", e.what());
    delete out;
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "[paged_attention_forward_valid_metadata] %s\n", e.what());
    delete out;
    return -1;
  } catch (...) {
    delete out;
    return -1;
  }

  delete out;
  return 1;
}

/// Valid bridge smoke: materialized metadata and lazy K/V inputs should emit
/// dependency-carrying pool outputs and evaluate successfully.
/// Returns 1 on success, 0 when the production bridge rejects the setup, -1 on
/// eval-time failure, and -3 if Metal is unavailable.
int mlx_paged_kv_write_forward_eval_smoke() {
  using namespace mlx::core;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  try {
    array k_pool = make_bf16_zeros(Shape{4, 4, 8, 16, 8});
    array v_pool = make_bf16_zeros(Shape{4, 4, 64, 16});
    array new_k = make_bf16_zeros(Shape{2, 4, 64});
    array new_v = make_bf16_zeros(Shape{2, 4, 64});
    std::vector<int64_t> slots = {0, 1};
    array slot_mapping(slots.data(), Shape{2}, int64);
    array k_scale(1.0f, float32);
    array v_scale(1.0f, float32);
    eval(slot_mapping);

    mlx_array* out_k = nullptr;
    mlx_array* out_v = nullptr;
    bool ok = mlx_paged_kv_write_forward(
        reinterpret_cast<mlx_array*>(&k_pool),
        reinterpret_cast<mlx_array*>(&v_pool),
        reinterpret_cast<mlx_array*>(&new_k),
        reinterpret_cast<mlx_array*>(&new_v),
        reinterpret_cast<mlx_array*>(&slot_mapping),
        reinterpret_cast<mlx_array*>(&k_scale),
        reinterpret_cast<mlx_array*>(&v_scale),
        /*block_size=*/16,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*kv_dtype_raw=*/1,
        &out_k,
        &out_v);
    if (!ok || out_k == nullptr || out_v == nullptr) {
      if (out_k) {
        delete reinterpret_cast<array*>(out_k);
      }
      if (out_v) {
        delete reinterpret_cast<array*>(out_v);
      }
      return 0;
    }

    auto* k_out = reinterpret_cast<array*>(out_k);
    auto* v_out = reinterpret_cast<array*>(out_v);
    eval(*k_out, *v_out);
    delete k_out;
    delete v_out;
    return 1;
  } catch (...) {
    return -1;
  }
}

} // extern "C"

// =============================================================================
// Phase 1 review-round-9 finding: PagedKVWrite::eval_gpu and
// PagedAttention::eval_gpu must mirror the row-contiguous / zero-offset
// check that the factories already perform, because `mlx::core::compile`
// cache hits rebuild the cached primitive with real inputs via
// `compile_replace` WITHOUT re-running the factory — the cache key only
// compares rank/shape/dtype. A graph first traced with contiguous inputs
// can later be replayed with a same-shape sliced/transposed view.
//
// These helpers verify the eval_gpu mirrored check fires on the second
// (compile-cached) eval. Layout is shared with the round-3/round-4 OOB
// tests so the cache keys remain independent of those existing tests.
// =============================================================================

namespace {

/// Trace function for the compile-cached non-contiguous `paged_kv_write`
/// test. Hard-coded scalars match the second-call invocation so the
/// cache key is consistent across compile+replay. Independent of
/// `paged_kv_write_oob_trace_fn` to keep the per-test compile cache
/// disjoint.
std::vector<mlx::core::array> paged_kv_write_non_contig_trace_fn(
    const std::vector<mlx::core::array>& inputs) {
  using namespace mlx::core::fast;
  if (inputs.size() != 7) {
    throw std::runtime_error(
        "paged_kv_write_non_contig_trace_fn: expected 7 inputs");
  }
  auto out = paged_kv_write(
      inputs[0],
      inputs[1],
      inputs[2],
      inputs[3],
      inputs[4],
      inputs[5],
      inputs[6],
      /*block_size=*/16,
      /*num_kv_heads=*/4,
      /*head_size=*/64,
      /*x_pack=*/8,
      KvDtype::Bf16,
      /*s=*/{});
  return {out.first, out.second};
}

/// Trace function for the compile-cached non-contiguous
/// `paged_attention` test. Independent of
/// `paged_attention_oob_trace_fn` to keep the per-test compile cache
/// disjoint.
std::vector<mlx::core::array> paged_attention_non_contig_trace_fn(
    const std::vector<mlx::core::array>& inputs) {
  using namespace mlx::core::fast;
  if (inputs.size() != 7) {
    throw std::runtime_error(
        "paged_attention_non_contig_trace_fn: expected 7 inputs");
  }
  auto out = paged_attention(
      inputs[0], // q
      inputs[1], // k_pool
      inputs[2], // v_pool
      inputs[3], // block_table
      inputs[4], // seq_lens
      inputs[5], // k_scale
      inputs[6], // v_scale
      /*scale=*/0.125f,
      /*softcap=*/0.0f,
      /*sliding_window=*/0,
      /*block_size=*/16,
      /*num_q_heads=*/8,
      /*num_kv_heads=*/4,
      /*head_size=*/64,
      KvDtype::Bf16,
      /*s=*/{});
  return {out};
}

} // namespace

extern "C" {

/// Compile a `paged_kv_write`-emitting function, call it once with
/// contiguous inputs (cache miss → factory + eval_gpu both pass), then
/// call it again with the SAME shapes / dtypes but with `new_k`
/// substituted by a non-row-contiguous transposed view. Cache HIT
/// bypasses the factory; `PagedKVWrite::eval_gpu`'s mirrored
/// `require_row_contiguous_zero_offset` check MUST throw on the second
/// eval.
///
/// Layout matches `mlx_paged_kv_write_compile_cached_oob_throws`:
///   block_size=16, num_kv_heads=4, head_size=64, x_pack=8, Bf16.
///   k_pool: [4, 4, 8, 16, 8] bf16
///   v_pool: [4, 4, 64, 16] bf16
///   new_k:  [2, 4, 64] bf16
///   new_v:  [2, 4, 64] bf16
///   slot_mapping: [2] int64
///
/// The non-contiguous `new_k` is built by transposing a `[64, 4, 2]`
/// row-contiguous source via permutation `(2, 1, 0)`, yielding a
/// `[2, 4, 64]` view whose strides do not match its logical layout
/// (row_contiguous == false).
///
/// Return codes:
///   1   — second-call eval threw `std::invalid_argument` (fix
///         working — the compile-cached path mirrors the contiguity
///         check).
///   0   — second-call eval did NOT throw (regression — a non-contiguous
///         view reached the kernel, which would alias the wrong region
///         of memory).
///  -1   — internal/setup error (first call failed unexpectedly).
///  -3   — Metal not available; eval-based verification skipped (slice/
///         transpose materialization needs Metal).
int mlx_paged_kv_write_compile_cached_non_contiguous_throws() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  const int kBlockSize = 16;
  const int kNumKvHeads = 4;
  const int kHeadSize = 64;
  const int kXPack = 8;
  const int kNumBlocks = 4;
  const int kNumTokens = 2;

  const size_t k_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      (kHeadSize / kXPack) * kBlockSize * kXPack;
  const size_t v_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      kHeadSize * kBlockSize;
  const size_t new_kv_elems =
      static_cast<size_t>(kNumTokens) * kNumKvHeads * kHeadSize;

  std::vector<uint16_t> k_pool_host(k_pool_elems, 0);
  std::vector<uint16_t> v_pool_host(v_pool_elems, 0);
  std::vector<uint16_t> new_k_host(new_kv_elems, 0x3FC0); // bf16(1.5)
  std::vector<uint16_t> new_v_host(new_kv_elems, 0x4040); // bf16(3.0)

  auto bf16_arr = [](const std::vector<uint16_t>& src, Shape shape) {
    auto* p = reinterpret_cast<const bfloat16_t*>(src.data());
    return array(p, std::move(shape), bfloat16);
  };
  auto i64_arr = [](const std::vector<int64_t>& src, Shape shape) {
    return array(src.data(), std::move(shape), int64);
  };

  Shape k_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize / kXPack, kBlockSize,
      kXPack};
  Shape v_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize, kBlockSize};
  Shape new_kv_shape{kNumTokens, kNumKvHeads, kHeadSize};

  array k_pool = bf16_arr(k_pool_host, k_pool_shape);
  array v_pool = bf16_arr(v_pool_host, v_pool_shape);
  array new_k_good = bf16_arr(new_k_host, new_kv_shape);
  array new_v = bf16_arr(new_v_host, new_kv_shape);
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  // First call: contiguous inputs, valid slot_mapping. Cache miss →
  // factory + eval_gpu both pass.
  std::vector<int64_t> good_slots = {0, 16};
  array slot_mapping = i64_arr(good_slots, Shape{kNumTokens});

  auto compiled = mlx::core::compile(&paged_kv_write_non_contig_trace_fn);

  std::vector<array> inputs1{
      k_pool, v_pool, new_k_good, new_v, slot_mapping, k_scale, v_scale};

  std::vector<array> outputs1;
  try {
    outputs1 = compiled(inputs1);
    if (outputs1.size() != 2) {
      return -1;
    }
    mlx::core::eval(outputs1[0], outputs1[1]);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_kv_write_compile_cached_non_contig] first call (contiguous "
        "inputs) threw unexpectedly: %s\n",
        e.what());
    return -1;
  }

  // Second call: substitute a non-row-contiguous view of `new_k` while
  // keeping every other input identical. Build a [64, 4, 2] source and
  // transpose with permutation (2, 1, 0) → [2, 4, 64]; the result has
  // the right logical shape but the strides mirror the [64, 4, 2]
  // layout, so row_contiguous == false.
  std::vector<uint16_t> new_k_alt_host(new_kv_elems, 0x3FC0);
  array new_k_alt = bf16_arr(new_k_alt_host, Shape{kHeadSize, kNumKvHeads,
      kNumTokens});
  eval(new_k_alt);
  array new_k_bad = mlx::core::transpose(new_k_alt, std::vector<int>{2, 1, 0});
  eval(new_k_bad);

  if (new_k_bad.flags().row_contiguous && new_k_bad.offset() == 0) {
    fprintf(
        stderr,
        "[paged_kv_write_compile_cached_non_contig] WARNING: transpose "
        "produced a row-contiguous view at offset 0; the test will not "
        "exercise the intended eval_gpu rejection path.\n");
  }

  std::vector<array> inputs2{
      k_pool, v_pool, new_k_bad, new_v, slot_mapping, k_scale, v_scale};

  std::vector<array> outputs2;
  try {
    outputs2 = compiled(inputs2);
    if (outputs2.size() != 2) {
      return -1;
    }
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_kv_write_compile_cached_non_contig] second call (compile) "
        "threw early: %s\n",
        e.what());
    return -1;
  }

  // The throw must fire on eval, not on the compiled() lambda call
  // itself (the lambda just stitches the graph; eval invokes
  // `eval_gpu`).
  try {
    mlx::core::eval(outputs2[0], outputs2[1]);
  } catch (const std::invalid_argument& /*e*/) {
    return 1; // SUCCESS — eval_gpu caught the non-contiguous view.
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_kv_write_compile_cached_non_contig] second-call eval threw "
        "a non-invalid_argument: %s\n",
        e.what());
    return 0;
  }
  fprintf(
      stderr,
      "[paged_kv_write_compile_cached_non_contig] second-call eval did NOT "
      "throw — the compile-cached path is missing its mirrored "
      "row-contiguous / zero-offset check\n");
  return 0;
}

/// Compile a `paged_attention`-emitting function, call it once with
/// contiguous inputs (cache miss → factory + eval_gpu both pass), then
/// call it again with the SAME shapes / dtypes but with `block_table`
/// substituted by a sliced (nonzero-offset) view of a wider table.
/// Cache HIT bypasses the factory; `PagedAttention::eval_gpu`'s mirrored
/// `require_row_contiguous_zero_offset` check MUST throw on the second
/// eval.
///
/// Layout matches `mlx_paged_attention_compile_cached_oob_throws`:
///   block_size=16, num_kv_heads=4, head_size=64, x_pack=8, Bf16,
///   num_q_heads=8, num_seqs=1, max_blocks_per_seq=4.
///
/// The non-contiguous `block_table` is built by slicing rows [0:1] of a
/// [3, 4] row-contiguous source. The slice has shape [1, 4] (matching
/// the first call's `block_table` shape), but rows beyond row 0 of the
/// underlying [3, 4] backing buffer make the view... actually, for
/// `block_table[0:1]` on a [3, 4] backing buffer, the slice IS
/// row-contiguous at offset 0. To force a nonzero offset, slice rows
/// [1:2] of a [3, 4] backing buffer: shape [1, 4], offset != 0.
///
/// Return codes:
///   1   — second-call eval threw `std::invalid_argument` (fix
///         working — the compile-cached path mirrors the contiguity
///         check).
///   0   — second-call eval did NOT throw (regression — a sliced
///         view with nonzero offset reached the kernel, which would
///         read from the wrong region of the backing allocation).
///  -1   — internal/setup error (first call failed unexpectedly).
///  -3   — Metal not available; eval-based verification skipped (slice
///         materialization needs Metal).
int mlx_paged_attention_compile_cached_non_contiguous_throws() {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  const int kBlockSize = 16;
  const int kNumKvHeads = 4;
  const int kHeadSize = 64;
  const int kXPack = 8;
  const int kNumBlocks = 4;
  const int kNumQHeads = 8;
  const int kNumSeqs = 1;
  const int kMaxBlocksPerSeq = 4;

  const size_t k_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      (kHeadSize / kXPack) * kBlockSize * kXPack;
  const size_t v_pool_elems = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
      kHeadSize * kBlockSize;
  const size_t q_elems =
      static_cast<size_t>(kNumSeqs) * kNumQHeads * kHeadSize;

  std::vector<uint16_t> k_pool_host(k_pool_elems, 0);
  std::vector<uint16_t> v_pool_host(v_pool_elems, 0);
  std::vector<uint16_t> q_host(q_elems, 0x3F80); // bf16(1.0)

  auto bf16_arr = [](const std::vector<uint16_t>& src, Shape shape) {
    auto* p = reinterpret_cast<const bfloat16_t*>(src.data());
    return array(p, std::move(shape), bfloat16);
  };
  auto i32_arr = [](const std::vector<int32_t>& src, Shape shape) {
    return array(src.data(), std::move(shape), int32);
  };

  Shape k_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize / kXPack, kBlockSize,
      kXPack};
  Shape v_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize, kBlockSize};
  Shape q_shape{kNumSeqs, kNumQHeads, kHeadSize};

  array q = bf16_arr(q_host, q_shape);
  array k_pool = bf16_arr(k_pool_host, k_pool_shape);
  array v_pool = bf16_arr(v_pool_host, v_pool_shape);
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  auto compiled = mlx::core::compile(&paged_attention_non_contig_trace_fn);

  // First call: contiguous block_table = [0, 0, 0, 0], seq_lens = [16].
  // Cache miss → factory + eval_gpu both pass.
  std::vector<int32_t> good_seq_lens = {16};
  std::vector<int32_t> good_block_table = {0, 0, 0, 0};
  array seq_lens_good = i32_arr(good_seq_lens, Shape{kNumSeqs});
  array block_table_good = i32_arr(
      good_block_table, Shape{kNumSeqs, kMaxBlocksPerSeq});

  std::vector<array> inputs1{
      q, k_pool, v_pool, block_table_good, seq_lens_good, k_scale, v_scale};

  std::vector<array> outputs1;
  try {
    outputs1 = compiled(inputs1);
    if (outputs1.size() != 1) {
      return -1;
    }
    mlx::core::eval(outputs1[0]);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_attention_compile_cached_non_contig] first call (contiguous "
        "inputs) threw unexpectedly: %s\n",
        e.what());
    return -1;
  }

  // Second call: substitute a non-row-contiguous view of `block_table`
  // while keeping every other input identical. Build a [3, 4] backing
  // buffer with all-zero entries (so the bounds check would still
  // pass), then take rows [1:2] → shape [1, 4] at nonzero offset.
  std::vector<int32_t> bt_full_host(3 * kMaxBlocksPerSeq, 0);
  array bt_full = i32_arr(bt_full_host, Shape{3, kMaxBlocksPerSeq});
  eval(bt_full);
  array block_table_bad = mlx::core::slice(
      bt_full,
      Shape{1, 0},
      Shape{2, kMaxBlocksPerSeq},
      Shape{1, 1});
  eval(block_table_bad);

  if (block_table_bad.flags().row_contiguous &&
      block_table_bad.offset() == 0) {
    fprintf(
        stderr,
        "[paged_attention_compile_cached_non_contig] WARNING: slice "
        "produced a row-contiguous view at offset 0; the test will not "
        "exercise the intended eval_gpu rejection path.\n");
  }

  std::vector<array> inputs2{
      q, k_pool, v_pool, block_table_bad, seq_lens_good, k_scale, v_scale};

  std::vector<array> outputs2;
  try {
    outputs2 = compiled(inputs2);
    if (outputs2.size() != 1) {
      return -1;
    }
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_attention_compile_cached_non_contig] second call (compile) "
        "threw early: %s\n",
        e.what());
    return -1;
  }

  // The throw must fire on eval, not on the compiled() lambda call
  // itself.
  try {
    mlx::core::eval(outputs2[0]);
  } catch (const std::invalid_argument& /*e*/) {
    return 1; // SUCCESS — eval_gpu caught the non-contiguous view.
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[paged_attention_compile_cached_non_contig] second-call eval "
        "threw a non-invalid_argument: %s\n",
        e.what());
    return 0;
  }
  fprintf(
      stderr,
      "[paged_attention_compile_cached_non_contig] second-call eval did "
      "NOT throw — the compile-cached path is missing its mirrored "
      "row-contiguous / zero-offset check\n");
  return 0;
}

} // extern "C"

// =============================================================================
// Phase 1 review-round-10 finding: factory + eval_gpu validation must be
// unified.
//
// `mlx::core::compile`'s cached re-traces rebuild the cached primitive
// with real inputs via `compile_replace` WITHOUT re-running the
// factory. To prove eval_gpu now catches bad scalar state on its own
// (without relying on the factory having already rejected the inputs),
// these helpers construct a primitive DIRECTLY with deliberately bad
// scalar state, wire it into an MLX graph via `array::make_arrays(...)`,
// and call `mlx::core::eval(...)` to invoke `eval_gpu`. The factory is
// never called. The throw must come from the validator inside
// `eval_gpu` itself, demonstrating that the defense-in-depth helper
// closes the entire class of "factory-only check bypassed by compile
// cache" hazards.
//
// All input arrays are tracer-only (built via `array(Shape{...}, dtype,
// nullptr, {})`). `eval_gpu` must reject BEFORE attempting to read
// data, so this is sufficient to drive the rejection without Metal
// being available.
// =============================================================================

namespace {

/// Build a primitive with deliberately bad scalar state, wire it into
/// the MLX graph via `array::make_arrays`, and `eval()` the result.
///
/// Return-code contract (round-12 tightening): the helper must
/// strictly prove the throw came from `PagedKVWrite::eval_gpu`'s
/// validator — not from the graph-construction step (`make_arrays`),
/// not from a different code path that happens to throw the same
/// exception class, and crucially NOT from a later runtime-content
/// guard inside `PagedKVWrite::eval_gpu` itself (slot_mapping bounds
/// check, etc.). The runtime guards in the same `eval_gpu` body emit
/// messages tagged "[runtime] PagedKVWrite::eval_gpu" — those would
/// otherwise pass a substring match on the operation tag alone and
/// falsely report rc=1 even if the scalar validator regressed.
///
/// To distinguish, the validator helper `validate_paged_kv_write_inputs`
/// (and its sister `require_row_contiguous_zero_offset`) prepend the
/// unique marker "[validator]" to every throw. The runtime-content
/// guards in `PagedKVWrite::eval_gpu` use "[runtime]" instead. The
/// helper requires BOTH "[validator]" AND the operation tag
/// "PagedKVWrite::eval_gpu" to count as a validator-driven rejection.
///
/// Codes:
///   * `1`  — eval threw `std::invalid_argument` AND the message
///            contains BOTH "[validator]" AND "PagedKVWrite::eval_gpu".
///            TEST SUCCESS: the eval_gpu validator (not a runtime
///            guard) is the throw site.
///   * `0`  — eval did not throw at all. TEST FAILURE: bad inputs were
///            silently accepted.
///   * `2`  — `make_arrays` threw `std::invalid_argument` BEFORE eval
///            ran. TEST FAILURE: this is an internal helper bug — the
///            graph-construction step should never reject these
///            structurally-valid inputs. Without this code, a pre-eval
///            rejection would masquerade as eval_gpu rejection.
///   * `-1` — any other exception type fired (from either step). TEST
///            FAILURE: internal error.
///   * `-2` — eval threw `std::invalid_argument` but the message did
///            NOT satisfy BOTH the "[validator]" marker AND the
///            "PagedKVWrite::eval_gpu" operation tag. TEST FAILURE:
///            either a non-eval_gpu layer threw, or a runtime-content
///            guard inside eval_gpu threw (which would mean the scalar
///            validator missed the bad state — the very regression
///            this test is here to catch).
int eval_paged_kv_write_with_bad_state(
    int block_size,
    int num_kv_heads,
    int head_size,
    int x_pack,
    mlx::core::fast::KvDtype kv_dtype,
    bool benign_slot_mapping = false) {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  auto stub_fallback = [](std::vector<array> /*ignored*/) -> std::vector<array> {
    throw std::runtime_error("eval_paged_kv_write_with_bad_state fallback");
  };

  // Build well-formed structural inputs for kv_dtype=Bf16 with the
  // canonical dimensions (block_size=16, num_kv_heads=4, head_size=64,
  // x_pack=8). The primitive's bad scalar state is what we want to
  // catch — the inputs themselves are kept structurally valid so we
  // exercise the scalar-state check, not a shape-mismatch check.
  //
  // Use REAL data-backed inputs (zero-filled) so MLX's evaluator
  // actually invokes eval_gpu. With tracer-only `array(shape, dtype,
  // nullptr, {})` inputs, MLX may treat the primitive as part of a
  // trace graph that never reaches eval_gpu.
  //
  // `benign_slot_mapping` controls whether the slot_mapping is also
  // designed to bypass the runtime bounds guard inside `eval_gpu`.
  // The default `{0, 16}` is fine for cases where the validator's
  // own scalar reject (num_kv_heads=0, x_pack mismatch, etc.) fires
  // BEFORE the runtime guard could see the data, but for cases like
  // `block_size=0` the runtime guard would also reject (pool_capacity=
  // num_blocks * block_size = 0, max_slot=16 >= 0). Setting
  // `benign_slot_mapping=true` swaps in `{-1, -1}` (all "skip"
  // sentinels), which the runtime guard explicitly excludes from
  // its max-slot reduction (`if (slot_data[i] >= 0 && ...)`), so
  // ONLY the scalar validator can possibly throw — proving the
  // validator (not the runtime guard) is the throw site.
  std::vector<uint16_t> k_pool_host(4 * 4 * 8 * 16 * 8, 0);
  std::vector<uint16_t> v_pool_host(4 * 4 * 64 * 16, 0);
  std::vector<uint16_t> new_k_host(2 * 4 * 64, 0);
  std::vector<uint16_t> new_v_host(2 * 4 * 64, 0);
  std::vector<int64_t> slot_mapping_host =
      benign_slot_mapping ? std::vector<int64_t>{-1, -1}
                          : std::vector<int64_t>{0, 16};

  auto bf16_arr = [](const std::vector<uint16_t>& src, Shape shape) {
    auto* p = reinterpret_cast<const bfloat16_t*>(src.data());
    return array(p, std::move(shape), bfloat16);
  };

  array k_pool = bf16_arr(k_pool_host, Shape{4, 4, 8, 16, 8});
  array v_pool = bf16_arr(v_pool_host, Shape{4, 4, 64, 16});
  array new_k = bf16_arr(new_k_host, Shape{2, 4, 64});
  array new_v = bf16_arr(new_v_host, Shape{2, 4, 64});
  array slot_mapping(slot_mapping_host.data(), Shape{2}, int64);
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  auto s = default_stream(default_device());
  auto primitive = std::make_shared<PagedKVWrite>(
      s, stub_fallback, block_size, num_kv_heads, head_size, x_pack, kv_dtype);

  std::vector<array> inputs{
      k_pool, v_pool, new_k, new_v, slot_mapping, k_scale, v_scale};

  // Step 1: graph construction. This MUST succeed — `make_arrays`
  // does not invoke the primitive's `eval_gpu` and only rejects on
  // structural problems with the supplied inputs/outputs (rank/dtype
  // mismatches with the requested output shape/dtype, etc.). The
  // structural inputs above are intentionally well-formed for the
  // kv_dtype=Bf16 canonical layout, so any throw here is a helper
  // bug — NOT proof that eval_gpu rejected.
  std::vector<array> results;
  try {
    results = array::make_arrays(
        {k_pool.shape(), v_pool.shape()},
        {k_pool.dtype(), v_pool.dtype()},
        primitive,
        inputs);
  } catch (const std::invalid_argument& e) {
    fprintf(
        stderr,
        "[eval_paged_kv_write_with_bad_state] INTERNAL HELPER BUG: "
        "make_arrays threw std::invalid_argument BEFORE eval (this "
        "is a graph-construction rejection, NOT proof of eval_gpu "
        "rejection): %s\n",
        e.what());
    return 2;
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[eval_paged_kv_write_with_bad_state] make_arrays threw "
        "non-invalid_argument: %s\n",
        e.what());
    return -1;
  }

  // Step 2: eval. ONLY this call should be capable of producing the
  // success exception. The validator helpers
  // (`validate_paged_kv_write_inputs` /
  // `require_row_contiguous_zero_offset`) tag every throw with BOTH
  // "[validator]" AND the operation context "PagedKVWrite::eval_gpu",
  // while later runtime-content guards inside the same `eval_gpu`
  // body (slot_mapping bounds check) tag their throws with
  // "[runtime] PagedKVWrite::eval_gpu" instead. We require BOTH the
  // validator marker AND the operation tag to count as success — a
  // `std::invalid_argument` lacking either substring means the
  // throw is not coming from the scalar validator we are exercising
  // (it is either from a different code layer or from the runtime
  // guard, the latter of which would mean the validator regressed).
  static constexpr const char* kValidatorTag = "[validator]";
  static constexpr const char* kEvalGpuTag = "PagedKVWrite::eval_gpu";
  try {
    mlx::core::eval(results[0], results[1]);
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    const bool has_validator_tag = what.find(kValidatorTag) != std::string::npos;
    const bool has_eval_gpu_tag = what.find(kEvalGpuTag) != std::string::npos;
    if (has_validator_tag && has_eval_gpu_tag) {
      return 1;
    }
    fprintf(
        stderr,
        "[eval_paged_kv_write_with_bad_state] eval threw "
        "std::invalid_argument but message did NOT satisfy both "
        "expected markers (validator='%s' present=%d, op='%s' "
        "present=%d) — throw site is NOT eval_gpu's scalar validator "
        "(it is either a different code layer, or a runtime-content "
        "guard inside eval_gpu, which would mean the validator "
        "regressed). Got: %s\n",
        kValidatorTag,
        static_cast<int>(has_validator_tag),
        kEvalGpuTag,
        static_cast<int>(has_eval_gpu_tag),
        what.c_str());
    return -2;
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[eval_paged_kv_write_with_bad_state] eval threw "
        "non-invalid_argument: %s\n",
        e.what());
    return 0;
  }
  fprintf(
      stderr,
      "[eval_paged_kv_write_with_bad_state] eval did NOT throw — "
      "validator inside eval_gpu missed bad scalar state\n");
  return 0;
}

/// Build a PagedAttention primitive with deliberately bad scalar
/// state, wire it into an MLX graph, and `eval()` the result. See
/// `eval_paged_kv_write_with_bad_state` above for the full
/// return-code contract — same semantics, with the operation context
/// tag being "PagedAttention::eval_gpu". The validator marker
/// "[validator]" stays the same; runtime-content guards in
/// `PagedAttention::eval_gpu` (seq_lens bounds, block_table bounds)
/// emit "[runtime] PagedAttention::eval_gpu", and rc=1 still requires
/// BOTH "[validator]" AND the operation tag to be present.
int eval_paged_attention_with_bad_state(
    float scale,
    float softcap,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int sliding_window,
    mlx::core::fast::KvDtype kv_dtype) {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  auto stub_fallback = [](std::vector<array> /*ignored*/) -> std::vector<array> {
    throw std::runtime_error("eval_paged_attention_with_bad_state fallback");
  };

  // REAL data-backed inputs so MLX's evaluator actually invokes
  // eval_gpu. See `eval_paged_kv_write_with_bad_state` for the
  // rationale.
  std::vector<uint16_t> q_host(1 * 8 * 64, 0);
  std::vector<uint16_t> k_pool_host(4 * 4 * 8 * 16 * 8, 0);
  std::vector<uint16_t> v_pool_host(4 * 4 * 64 * 16, 0);
  std::vector<int32_t> block_table_host = {0, 0, 0, 0};
  std::vector<int32_t> seq_lens_host = {16};

  auto bf16_arr = [](const std::vector<uint16_t>& src, Shape shape) {
    auto* p = reinterpret_cast<const bfloat16_t*>(src.data());
    return array(p, std::move(shape), bfloat16);
  };
  auto i32_arr = [](const std::vector<int32_t>& src, Shape shape) {
    return array(src.data(), std::move(shape), int32);
  };

  array q = bf16_arr(q_host, Shape{1, 8, 64});
  array k_pool = bf16_arr(k_pool_host, Shape{4, 4, 8, 16, 8});
  array v_pool = bf16_arr(v_pool_host, Shape{4, 4, 64, 16});
  array block_table = i32_arr(block_table_host, Shape{1, 4});
  array seq_lens = i32_arr(seq_lens_host, Shape{1});
  array k_scale(1.0f, float32);
  array v_scale(1.0f, float32);

  auto s = default_stream(default_device());
  auto primitive = std::make_shared<PagedAttention>(
      s,
      stub_fallback,
      scale,
      softcap,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window,
      kv_dtype);

  std::vector<array> inputs{
      q, k_pool, v_pool, block_table, seq_lens, k_scale, v_scale};

  // Output shape from primitive's scalar state (matches
  // PagedAttention::output_shapes). Avoid the negative or zero shapes
  // that some bad scalar states would produce — clamp to a safe
  // 1-element shape so the array constructor itself can't reject
  // before eval_gpu runs.
  int safe_out_dim1 = num_q_heads > 0 ? num_q_heads : 1;
  int safe_out_dim2 = head_size > 0 ? head_size : 1;
  Shape out_shape{q.shape(0), safe_out_dim1, safe_out_dim2};

  // Step 1: graph construction. The `array(shape, dtype, primitive,
  // inputs)` constructor only validates structural input/output
  // shapes — it does NOT invoke `eval_gpu`. Wrap it in its own try
  // so a pre-eval rejection is surfaced as rc=2 (internal helper
  // bug) instead of being conflated with eval_gpu rejection.
  std::unique_ptr<array> result_holder;
  try {
    result_holder = std::make_unique<array>(
        std::move(out_shape), bfloat16, primitive, std::move(inputs));
  } catch (const std::invalid_argument& e) {
    fprintf(
        stderr,
        "[eval_paged_attention_with_bad_state] INTERNAL HELPER BUG: "
        "array constructor threw std::invalid_argument BEFORE eval "
        "(this is a graph-construction rejection, NOT proof of "
        "eval_gpu rejection): %s\n",
        e.what());
    return 2;
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[eval_paged_attention_with_bad_state] array constructor threw "
        "non-invalid_argument: %s\n",
        e.what());
    return -1;
  }

  // Step 2: eval. Only this call should produce the success
  // exception, and the message must contain BOTH the validator-only
  // marker "[validator]" injected by `validate_paged_attention_inputs`
  // (and `require_row_contiguous_zero_offset`) AND the operation tag
  // "PagedAttention::eval_gpu". A bare `std::invalid_argument` lacking
  // either substring means the throw came from a different code layer
  // (no op tag) or from a runtime-content guard inside eval_gpu (op
  // tag present but no validator marker — those guards use
  // "[runtime] PagedAttention::eval_gpu"). Either case fails to prove
  // the scalar validator is intact, so we return -2.
  static constexpr const char* kValidatorTag = "[validator]";
  static constexpr const char* kEvalGpuTag = "PagedAttention::eval_gpu";
  try {
    mlx::core::eval(*result_holder);
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    const bool has_validator_tag = what.find(kValidatorTag) != std::string::npos;
    const bool has_eval_gpu_tag = what.find(kEvalGpuTag) != std::string::npos;
    if (has_validator_tag && has_eval_gpu_tag) {
      return 1;
    }
    fprintf(
        stderr,
        "[eval_paged_attention_with_bad_state] eval threw "
        "std::invalid_argument but message did NOT satisfy both "
        "expected markers (validator='%s' present=%d, op='%s' "
        "present=%d) — throw site is NOT eval_gpu's scalar validator "
        "(it is either a different code layer, or a runtime-content "
        "guard inside eval_gpu, which would mean the validator "
        "regressed). Got: %s\n",
        kValidatorTag,
        static_cast<int>(has_validator_tag),
        kEvalGpuTag,
        static_cast<int>(has_eval_gpu_tag),
        what.c_str());
    return -2;
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[eval_paged_attention_with_bad_state] eval threw "
        "non-invalid_argument: %s\n",
        e.what());
    return 0;
  }
  fprintf(
      stderr,
      "[eval_paged_attention_with_bad_state] eval did NOT throw — "
      "validator inside eval_gpu missed bad scalar state\n");
  return 0;
}

} // namespace

extern "C" {

// Round-13 hardening: every extern-C eval_gpu test helper below wraps
// its body in a catch-all so NO C++ exception ever crosses into Rust.
// The inner helpers (`eval_paged_kv_write_with_bad_state` /
// `eval_paged_attention_with_bad_state`) already catch known throw
// sites (graph construction + eval), but a stray exception from any
// other code path inside this boundary (array destructors, MLX
// internals, allocator failures, etc.) would otherwise propagate
// through the FFI boundary and abort the test harness with "Rust
// cannot catch foreign exceptions". Returning `-1` here keeps the
// existing rc contract (-1 == internal helper error) so the Rust
// `assert_eval_gpu_rejects_bad_state` matcher reports a clean
// failure with the underlying message on stderr.
//
// All helpers must satisfy: `try { ... return rc; } catch (...) {
// return -1; }`. Do NOT add code that can throw outside the inner
// `try { ... }` block of these wrappers.

/// Primitive directly constructed with `num_kv_heads_=0`. eval_gpu's
/// validator must reject before the dispatch path can touch the
/// buffers (the kernel would otherwise compute a degenerate
/// `num_queries_per_kv` and divide by zero).
int mlx_paged_kv_write_eval_gpu_rejects_zero_kv_heads() {
  try {
    using namespace mlx::core::fast;
    return eval_paged_kv_write_with_bad_state(
        /*block_size=*/16,
        /*num_kv_heads=*/0,
        /*head_size=*/64,
        /*x_pack=*/8,
        KvDtype::Bf16);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_rejects_zero_kv_heads] FFI boundary "
        "caught uncontained C++ exception (extern-C catch-all): %s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_rejects_zero_kv_heads] FFI boundary "
        "caught uncontained non-std exception (extern-C catch-all)\n");
    return -1;
  }
}

/// Primitive directly constructed with `block_size_=0`. eval_gpu's
/// validator must reject; otherwise the runtime bounds check would
/// compute `(s + block_size_ - 1) / block_size_` and divide by zero.
int mlx_paged_kv_write_eval_gpu_rejects_zero_block_size() {
  try {
    using namespace mlx::core::fast;
    return eval_paged_kv_write_with_bad_state(
        /*block_size=*/0,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*x_pack=*/8,
        KvDtype::Bf16);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_rejects_zero_block_size] FFI boundary "
        "caught uncontained C++ exception (extern-C catch-all): %s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_rejects_zero_block_size] FFI boundary "
        "caught uncontained non-std exception (extern-C catch-all)\n");
    return -1;
  }
}

/// Round-12 regression: prove the scalar validator (NOT the runtime
/// slot_mapping bounds guard) is the throw site for `block_size=0`.
/// The previous helper used slot_mapping={0,16}, which means a
/// regressed validator could still surface rc=1 because the runtime
/// guard would fire (pool_capacity = num_blocks*block_size = 0,
/// max_slot=16 >= 0). With `benign_slot_mapping=true` the
/// slot_mapping is `{-1,-1}` — the runtime guard excludes negative
/// sentinels from its max-slot reduction, so it can NEVER fire on
/// this input. The ONLY throw site capable of producing an
/// `std::invalid_argument` here is `validate_paged_kv_write_inputs`
/// rejecting `block_size <= 0`; if its `[validator]` marker is
/// missing, the helper returns rc=-2 by contract.
int mlx_paged_kv_write_eval_gpu_validator_proof_zero_block_size() {
  try {
    using namespace mlx::core::fast;
    return eval_paged_kv_write_with_bad_state(
        /*block_size=*/0,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*x_pack=*/8,
        KvDtype::Bf16,
        /*benign_slot_mapping=*/true);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_validator_proof_zero_block_size] FFI "
        "boundary caught uncontained C++ exception (extern-C catch-all): "
        "%s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_validator_proof_zero_block_size] FFI "
        "boundary caught uncontained non-std exception (extern-C "
        "catch-all)\n");
    return -1;
  }
}

/// Primitive directly constructed with `head_size_=0`. eval_gpu's
/// validator must reject; the Metal kernel uses `head_size` as a grid
/// extent and indexing stride — a zero-sized inner dim is a degenerate
/// launch.
int mlx_paged_kv_write_eval_gpu_rejects_zero_head_size() {
  try {
    using namespace mlx::core::fast;
    return eval_paged_kv_write_with_bad_state(
        /*block_size=*/16,
        /*num_kv_heads=*/4,
        /*head_size=*/0,
        /*x_pack=*/8,
        KvDtype::Bf16);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_rejects_zero_head_size] FFI boundary "
        "caught uncontained C++ exception (extern-C catch-all): %s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_rejects_zero_head_size] FFI boundary "
        "caught uncontained non-std exception (extern-C catch-all)\n");
    return -1;
  }
}

/// Primitive directly constructed with `x_pack_=16` (Fp8 value)
/// while `kv_dtype_=Bf16` (which expects x_pack=8). eval_gpu's
/// validator must reject; otherwise the K-pool layout assumed by
/// the kernel would disagree with what the dispatch path encodes.
int mlx_paged_kv_write_eval_gpu_rejects_x_pack_dtype_mismatch() {
  try {
    using namespace mlx::core::fast;
    return eval_paged_kv_write_with_bad_state(
        /*block_size=*/16,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*x_pack=*/16,
        KvDtype::Bf16);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_rejects_x_pack_dtype_mismatch] FFI "
        "boundary caught uncontained C++ exception (extern-C catch-all): "
        "%s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_kv_write_eval_gpu_rejects_x_pack_dtype_mismatch] FFI "
        "boundary caught uncontained non-std exception (extern-C "
        "catch-all)\n");
    return -1;
  }
}

/// Primitive directly constructed with `num_kv_heads_=0`. eval_gpu's
/// validator must reject before the dispatch path. The Metal kernel
/// would otherwise compute `num_queries_per_kv = num_q_heads_ / 0` and
/// divide by zero.
int mlx_paged_attention_eval_gpu_rejects_zero_kv_heads() {
  try {
    using namespace mlx::core::fast;
    return eval_paged_attention_with_bad_state(
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/0,
        /*head_size=*/64,
        /*sliding_window=*/0,
        KvDtype::Bf16);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_attention_eval_gpu_rejects_zero_kv_heads] FFI boundary "
        "caught uncontained C++ exception (extern-C catch-all): %s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_attention_eval_gpu_rejects_zero_kv_heads] FFI boundary "
        "caught uncontained non-std exception (extern-C catch-all)\n");
    return -1;
  }
}

/// Primitive directly constructed with `num_q_heads_ % num_kv_heads_
/// != 0`. eval_gpu's validator must reject; otherwise later q-heads
/// would compute kv_head_idx beyond the KV-head dimension and read
/// out-of-pool memory.
int mlx_paged_attention_eval_gpu_rejects_indivisible_grouping() {
  try {
    using namespace mlx::core::fast;
    return eval_paged_attention_with_bad_state(
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*block_size=*/16,
        /*num_q_heads=*/6,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*sliding_window=*/0,
        KvDtype::Bf16);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_attention_eval_gpu_rejects_indivisible_grouping] FFI "
        "boundary caught uncontained C++ exception (extern-C catch-all): "
        "%s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_attention_eval_gpu_rejects_indivisible_grouping] FFI "
        "boundary caught uncontained non-std exception (extern-C "
        "catch-all)\n");
    return -1;
  }
}

/// Primitive directly constructed with `sliding_window_<0`. eval_gpu's
/// validator must reject — Phase 7 lifts the Phase 1 sliding_window=0
/// restriction but negative values are still illegal (the only "no
/// mask" sentinel is 0).
int mlx_paged_attention_eval_gpu_rejects_sliding_window() {
  try {
    using namespace mlx::core::fast;
    return eval_paged_attention_with_bad_state(
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*block_size=*/16,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*sliding_window=*/-1,
        KvDtype::Bf16);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_attention_eval_gpu_rejects_sliding_window] FFI boundary "
        "caught uncontained C++ exception (extern-C catch-all): %s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_attention_eval_gpu_rejects_sliding_window] FFI boundary "
        "caught uncontained non-std exception (extern-C catch-all)\n");
    return -1;
  }
}

/// Primitive directly constructed with `block_size_=0`. eval_gpu's
/// validator must reject; otherwise the runtime bounds check would
/// divide `(s + block_size_ - 1) / block_size_` by zero in host code,
/// crashing the process before the `max_context_len <= 0` guard runs.
int mlx_paged_attention_eval_gpu_rejects_zero_block_size() {
  try {
    using namespace mlx::core::fast;
    return eval_paged_attention_with_bad_state(
        /*scale=*/0.125f,
        /*softcap=*/0.0f,
        /*block_size=*/0,
        /*num_q_heads=*/8,
        /*num_kv_heads=*/4,
        /*head_size=*/64,
        /*sliding_window=*/0,
        KvDtype::Bf16);
  } catch (const std::exception& e) {
    fprintf(
        stderr,
        "[mlx_paged_attention_eval_gpu_rejects_zero_block_size] FFI boundary "
        "caught uncontained C++ exception (extern-C catch-all): %s\n",
        e.what());
    return -1;
  } catch (...) {
    fprintf(
        stderr,
        "[mlx_paged_attention_eval_gpu_rejects_zero_block_size] FFI boundary "
        "caught uncontained non-std exception (extern-C catch-all)\n");
    return -1;
  }
}

/// Phase 2 stress test (mixed paged + non-paged ops, correctness +
/// determinism + V1/V2 coverage).
///
/// Builds a small graph that:
///   1. computes `q_offset = q + bias` (a non-paged op before the write)
///   2. writes new K/V into the K-pool/V-pool via `paged_kv_write`
///   3. computes `attn = paged_attention(q_offset, k_pool', v_pool', ...)`
///   4. computes `out = attn + attn` (a non-paged op after the read)
///
/// This exercises the dispatch's interaction with MLX's command-encoder
/// dependency tracking — `paged_kv_write` mutates k_pool/v_pool in
/// place (registered via `set_output_array`), which `paged_attention`
/// then reads (registered via `set_input_array`). MLX's encoder must
/// fence these correctly. Phase 1's separate-queue dispatch
/// fundamentally cannot get this right; Phase 2 does.
///
/// Correctness + determinism criteria:
///
///   1. SYNCHRONOUS REFERENCE: build the same write+attention graph
///      with an explicit `eval()` between the write and the attention
///      read. The `eval()` boundary forces a true synchronization
///      barrier, so this output is provably correct (the read sees
///      the fully-completed write). Every stress-loop run must be
///      byte-equal to this reference.
///
///   2. NO-WRITE BASELINE: build an attention graph reading a
///      ZERO-INITIALIZED pool (no preceding write). Every stress-loop
///      run's output must DIFFER from this baseline — otherwise a
///      race could deterministically schedule the read before the
///      write and pass the determinism check while reading stale
///      zeros.
///
/// `seq_len` selects V1 vs V2 of the underlying paged-attention
/// kernel: max_context_len <= 512 picks V1 (no partitioning), > 512
/// picks V2 (partitioning + reduce). Coverage of both paths is
/// required because V2 allocates auxiliary buffers
/// (`exp_sums`/`max_logits`/`tmp_out`) and chains a second kernel —
/// the encoder must fence both phases plus any preceding write.
///
/// Returns:
///   0     — success (all `iterations` runs byte-equal to the
///           synchronous reference AND differ from the no-write
///           baseline)
///   -1    — internal/setup error
///   -2    — a stress-loop run diverged from the synchronous reference
///   -3    — Metal not available; test skipped
///   -4    — a stress-loop run matched the no-write baseline (the
///           write didn't actually land in time, but the loop happened
///           to be deterministic about it)
extern "C" int mlx_paged_phase2_stress_mixed_graph_v(
    int iterations,
    int seq_len) {
  using namespace mlx::core;
  using namespace mlx::core::fast;

  if (iterations <= 0 || seq_len <= 0) {
    return -1;
  }
  if (!mlx::core::metal::is_available()) {
    return -3;
  }

  // Block layout: head_size must be in the kernel's instantiation list
  // (32, 64, 80, 96, 112, 120, 128, 192, 256, 512) and divisible by
  // x_pack (8 for Bf16). block_size in {8, 16, 32}.
  const int kBlockSize = 16;
  const int kNumQHeads = 4;
  const int kNumKvHeads = 4;
  const int kHeadSize = 64;
  const int kXPack = 8;
  const int kNumSeqs = 1;
  const int kSeqLen = seq_len;
  // ceil(seq_len / block_size). Pad by one extra block to keep the
  // pool/block-table comfortably sized.
  const int kMaxBlocksPerSeq = (kSeqLen + kBlockSize - 1) / kBlockSize + 1;
  const int kNumBlocks = kMaxBlocksPerSeq;

  try {
    Shape k_pool_shape{
        kNumBlocks, kNumKvHeads, kHeadSize / kXPack, kBlockSize, kXPack};
    Shape v_pool_shape{kNumBlocks, kNumKvHeads, kHeadSize, kBlockSize};
    Shape new_kv_shape{kSeqLen, kNumKvHeads, kHeadSize};
    Shape q_shape{kNumSeqs, kNumQHeads, kHeadSize};

    // Build BF16 array from a deterministic host-side function. We
    // synthesize values via sin(seed_a * i + seed_b) * 0.25 then
    // truncate to bf16 (top 16 bits of the f32 representation).
    auto bf16_arr = [](size_t n, float seed_a, float seed_b, Shape shape) {
      std::vector<uint16_t> host(n);
      for (size_t i = 0; i < n; ++i) {
        float f =
            std::sin(seed_a * static_cast<float>(i) + seed_b) * 0.25f;
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        host[i] = static_cast<uint16_t>(bits >> 16);
      }
      auto* p = reinterpret_cast<const bfloat16_t*>(host.data());
      return array(p, std::move(shape), bfloat16);
    };

    // Build a fresh zero-initialized BF16 array of the given shape.
    auto bf16_zeros = [](size_t n, Shape shape) {
      std::vector<uint16_t> host(n, 0);
      auto* p = reinterpret_cast<const bfloat16_t*>(host.data());
      return array(p, std::move(shape), bfloat16);
    };

    // Sizes used for the host-side allocations.
    const size_t k_pool_n = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
        (kHeadSize / kXPack) * kBlockSize * kXPack;
    const size_t v_pool_n = static_cast<size_t>(kNumBlocks) * kNumKvHeads *
        kHeadSize * kBlockSize;
    const size_t new_kv_n =
        static_cast<size_t>(kSeqLen) * kNumKvHeads * kHeadSize;
    const size_t q_n =
        static_cast<size_t>(kNumSeqs) * kNumQHeads * kHeadSize;
    const size_t bias_n = q_n;

    auto build_block_table = [&]() {
      std::vector<int32_t> block_table_host(kNumSeqs * kMaxBlocksPerSeq);
      for (int s = 0; s < kNumSeqs; ++s) {
        for (int b = 0; b < kMaxBlocksPerSeq; ++b) {
          block_table_host[s * kMaxBlocksPerSeq + b] = b;
        }
      }
      return array(
          block_table_host.data(),
          Shape{kNumSeqs, kMaxBlocksPerSeq},
          int32);
    };

    auto build_seq_lens = [&]() {
      std::vector<int32_t> seq_lens_host(kNumSeqs, kSeqLen);
      return array(seq_lens_host.data(), Shape{kNumSeqs}, int32);
    };

    auto build_slot_mapping = [&]() {
      std::vector<int64_t> slot_host(kSeqLen);
      for (int i = 0; i < kSeqLen; ++i) {
        slot_host[i] = i;
      }
      return array(slot_host.data(), Shape{kSeqLen}, int64);
    };

    auto k_pool_input = [&]() { return bf16_zeros(k_pool_n, k_pool_shape); };
    auto v_pool_input = [&]() { return bf16_zeros(v_pool_n, v_pool_shape); };
    auto new_k_input = [&]() {
      return bf16_arr(new_kv_n, 0.13f, 0.7f, new_kv_shape);
    };
    auto new_v_input = [&]() {
      return bf16_arr(new_kv_n, 0.17f, 1.3f, new_kv_shape);
    };
    auto q_input = [&]() {
      return bf16_arr(q_n, 0.21f, 2.1f, q_shape);
    };
    auto bias_input = [&]() {
      return bf16_arr(bias_n, 0.05f, 0.0f, q_shape);
    };

    // ---------------------------------------------------------------
    // Step 0a: SYNCHRONOUS REFERENCE.
    //
    // Build write+attention with an explicit `eval()` between the
    // write and the attention read. The eval boundary is a hard sync
    // barrier — Metal's queues drain to completion before host code
    // continues, so the attention here MUST see the just-written
    // K/V. This output is the known-good answer.
    // ---------------------------------------------------------------
    std::vector<uint16_t> reference_bytes;
    size_t expected_n_elems = 0;
    {
      array k_pool_ref = k_pool_input();
      array v_pool_ref = v_pool_input();
      array new_k_ref = new_k_input();
      array new_v_ref = new_v_input();
      array slot_mapping_ref = build_slot_mapping();
      array q_ref = q_input();
      array bias_ref = bias_input();
      array block_table_ref = build_block_table();
      array seq_lens_ref = build_seq_lens();
      array k_scale_ref(1.0f, float32);
      array v_scale_ref(1.0f, float32);

      array q_offset_ref = mlx::core::add(q_ref, bias_ref);

      auto [k_pool_written, v_pool_written] = paged_kv_write(
          k_pool_ref,
          v_pool_ref,
          new_k_ref,
          new_v_ref,
          slot_mapping_ref,
          k_scale_ref,
          v_scale_ref,
          kBlockSize,
          kNumKvHeads,
          kHeadSize,
          kXPack,
          KvDtype::Bf16);

      // Force the write to complete before the read graph is built.
      // This is the "true reference" guarantee — eval() drains the
      // command queue, so any subsequent dispatch sees fully-written
      // K/V regardless of any write-to-read fence.
      mlx::core::eval({k_pool_written, v_pool_written});

      array attn_ref = paged_attention(
          q_offset_ref,
          k_pool_written,
          v_pool_written,
          block_table_ref,
          seq_lens_ref,
          k_scale_ref,
          v_scale_ref,
          /*scale=*/0.125f,
          /*softcap=*/0.0f,
          /*sliding_window=*/0,
          kBlockSize,
          kNumQHeads,
          kNumKvHeads,
          kHeadSize,
          KvDtype::Bf16);

      array out_ref = mlx::core::add(attn_ref, attn_ref);
      mlx::core::eval(out_ref);

      const bfloat16_t* data = out_ref.data<bfloat16_t>();
      expected_n_elems = out_ref.size();
      const uint16_t* bits = reinterpret_cast<const uint16_t*>(data);
      reference_bytes.assign(bits, bits + expected_n_elems);
    }

    // ---------------------------------------------------------------
    // Step 0b: NO-WRITE BASELINE.
    //
    // Run the attention graph against a ZERO pool with no preceding
    // write. This is what a broken write→read fence could
    // deterministically read if it scheduled the read before the
    // write completed. Stress-loop outputs MUST DIFFER from this.
    // ---------------------------------------------------------------
    std::vector<uint16_t> nowrite_bytes;
    {
      array k_pool_zero = k_pool_input();
      array v_pool_zero = v_pool_input();
      array q_zero = q_input();
      array bias_zero = bias_input();
      array block_table_zero = build_block_table();
      array seq_lens_zero = build_seq_lens();
      array k_scale_zero(1.0f, float32);
      array v_scale_zero(1.0f, float32);

      array q_offset_zero = mlx::core::add(q_zero, bias_zero);

      array attn_zero = paged_attention(
          q_offset_zero,
          k_pool_zero,
          v_pool_zero,
          block_table_zero,
          seq_lens_zero,
          k_scale_zero,
          v_scale_zero,
          /*scale=*/0.125f,
          /*softcap=*/0.0f,
          /*sliding_window=*/0,
          kBlockSize,
          kNumQHeads,
          kNumKvHeads,
          kHeadSize,
          KvDtype::Bf16);

      array out_zero = mlx::core::add(attn_zero, attn_zero);
      mlx::core::eval(out_zero);

      const bfloat16_t* data = out_zero.data<bfloat16_t>();
      const size_t n_elems = out_zero.size();
      const uint16_t* bits = reinterpret_cast<const uint16_t*>(data);
      nowrite_bytes.assign(bits, bits + n_elems);

      if (n_elems != expected_n_elems) {
        fprintf(
            stderr,
            "[phase2_stress] no-write baseline output size %zu != "
            "reference size %zu\n",
            n_elems,
            expected_n_elems);
        return -1;
      }
    }

    // Reference and no-write baselines must differ. If they happen
    // to be byte-equal — almost impossible with non-zero K/V — the
    // "nowrite" assertion below is meaningless. (A single byte of
    // disagreement is enough.)
    {
      bool any_differ = false;
      for (size_t i = 0; i < expected_n_elems; ++i) {
        if (reference_bytes[i] != nowrite_bytes[i]) {
          any_differ = true;
          break;
        }
      }
      if (!any_differ) {
        fprintf(
            stderr,
            "[phase2_stress] reference and no-write baselines are "
            "byte-equal; pick non-zero K/V to make this test "
            "discriminating\n");
        return -1;
      }
    }

    // ---------------------------------------------------------------
    // Step 1: stress loop. Build the mixed graph N times, identical
    // inputs each time. Each output must equal the synchronous
    // reference and differ from the no-write baseline.
    // ---------------------------------------------------------------
    for (int run = 0; run < iterations; ++run) {
      array k_pool = k_pool_input();
      array v_pool = v_pool_input();
      array new_k = new_k_input();
      array new_v = new_v_input();
      array slot_mapping = build_slot_mapping();
      array q = q_input();
      array bias = bias_input();
      array block_table = build_block_table();
      array seq_lens = build_seq_lens();
      array k_scale(1.0f, float32);
      array v_scale(1.0f, float32);

      // Step 1a: a non-paged op BEFORE the paged write so MLX's
      // dependency graph has to fence between us and another
      // command. (`add` produces a fresh allocation; the write below
      // doesn't depend on it.)
      array q_offset = mlx::core::add(q, bias);

      // Step 1b: paged_kv_write — fills slots 0..kSeqLen-1.
      auto [k_pool_after, v_pool_after] = paged_kv_write(
          k_pool,
          v_pool,
          new_k,
          new_v,
          slot_mapping,
          k_scale,
          v_scale,
          kBlockSize,
          kNumKvHeads,
          kHeadSize,
          kXPack,
          KvDtype::Bf16);

      // Step 1c: paged_attention reads from the just-written pools.
      // The encoder must fence between the write (Step 1b) and this
      // read; if Phase 2 dispatch is correct, MLX's `set_output_array`
      // → `set_input_array` chain handles it. NOTE: no `eval()`
      // between the write and this read in the stress runs — the
      // whole point is to test the encoder's automatic fencing.
      array attn = paged_attention(
          q_offset,
          k_pool_after,
          v_pool_after,
          block_table,
          seq_lens,
          k_scale,
          v_scale,
          /*scale=*/0.125f,
          /*softcap=*/0.0f,
          /*sliding_window=*/0,
          kBlockSize,
          kNumQHeads,
          kNumKvHeads,
          kHeadSize,
          KvDtype::Bf16);

      // Step 1d: a non-paged op AFTER the read (depends on attn).
      array out = mlx::core::add(attn, attn);

      mlx::core::eval(out);

      const bfloat16_t* data = out.data<bfloat16_t>();
      const size_t n_elems = out.size();
      const uint16_t* bits = reinterpret_cast<const uint16_t*>(data);

      if (n_elems != expected_n_elems) {
        fprintf(
            stderr,
            "[phase2_stress] run %d output size mismatch: expected %zu "
            "got %zu\n",
            run,
            expected_n_elems,
            n_elems);
        return -2;
      }

      // Compare to the synchronous reference (the known-good answer).
      // A race in the write→read fence would surface as a divergence
      // here.
      for (size_t i = 0; i < n_elems; ++i) {
        if (bits[i] != reference_bytes[i]) {
          fprintf(
              stderr,
              "[phase2_stress] run %d byte %zu diverged from synchronous "
              "reference: ref=0x%04x current=0x%04x (race or stale read)\n",
              run,
              i,
              reference_bytes[i],
              bits[i]);
          return -2;
        }
      }

      // Compare to the no-write baseline. A run that schedules the
      // read before the write would read zeros and match this
      // baseline; that pattern would also match the previous loop's
      // determinism check, so we have to assert non-equality
      // explicitly. (At least one byte must differ.)
      {
        bool any_differ = false;
        for (size_t i = 0; i < n_elems; ++i) {
          if (bits[i] != nowrite_bytes[i]) {
            any_differ = true;
            break;
          }
        }
        if (!any_differ) {
          fprintf(
              stderr,
              "[phase2_stress] run %d output is byte-equal to the "
              "no-write baseline. The write→read fence either failed "
              "or never ran; the read saw zeros. (race detected)\n",
              run);
          return -4;
        }
      }
    }

    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "[phase2_stress] threw: %s\n", e.what());
    return -1;
  } catch (...) {
    fprintf(stderr, "[phase2_stress] threw non-std exception\n");
    return -1;
  }
}

/// Backward-compatible wrapper: defaults to V1 (seq_len=8). Kept so
/// any external caller that still binds the original symbol gets the
/// hardened logic without re-binding. The Rust test calls the `_v`
/// variant directly so it can drive both V1 and V2.
extern "C" int mlx_paged_phase2_stress_mixed_graph(int iterations) {
  return mlx_paged_phase2_stress_mixed_graph_v(iterations, /*seq_len=*/8);
}

} // extern "C"
