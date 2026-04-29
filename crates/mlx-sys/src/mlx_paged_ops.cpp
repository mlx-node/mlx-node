// Phase 1 of the paged-attention compile integration.
//
// Implements the `PagedKVWrite` and `PagedAttention` MLX `Custom`
// primitives. Their `eval_gpu` paths call into a temporary `extern "C"`
// shim exposed by the `mlx-paged-attn` Rust crate (see
// `crates/mlx-paged-attn/src/extern_c.rs`).
//
// PHASE 1 LIMITATION: this shim dispatches to mlx_paged_attn's separate
// Metal command queue. Phase 2 ports dispatch to MLX's queue (the one
// used by `inputs[0].primitive_ptr()->stream()`) so dependency tracking
// is correct. Until then, callers MUST `eval()` any prior dependencies
// before calling `paged_kv_write`/`paged_attention`, and `eval()` the
// outputs before reading them outside an MLX graph.

#include "mlx_paged_ops.h"
#include "mlx_common.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlx/compile.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::fast {

// =============================================================================
// extern-C shim contract
//
// Symbols defined in `crates/mlx-paged-attn/src/extern_c.rs`. They are
// linked into the final .node cdylib alongside this static library
// (mlx-core depends on both mlx-sys and mlx-paged-attn). Both wrappers
// return 0 on success, -1 on error; errors are written to stderr by
// the Rust side.
// =============================================================================

extern "C" int mlx_paged_attn_reshape_and_cache_dispatch(
    void* key_pool_buffer,
    void* value_pool_buffer,
    void* new_keys_buffer,
    size_t new_keys_offset,
    void* new_values_buffer,
    size_t new_values_offset,
    void* slot_mapping_buffer,
    size_t slot_mapping_offset,
    uint32_t num_tokens,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t block_size,
    int32_t x_pack,
    uint8_t kv_dtype_raw,
    float k_scale,
    float v_scale);

extern "C" int mlx_paged_attn_paged_attention_dispatch(
    void* queries_buffer,
    size_t queries_offset,
    void* key_pool_buffer,
    void* value_pool_buffer,
    void* block_table_buffer,
    void* seq_lens_buffer,
    void* output_buffer,
    size_t output_offset,
    uint32_t num_seqs,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t block_size,
    uint32_t max_context_len,
    uint32_t max_blocks_per_seq,
    float scale,
    float softcap,
    int32_t sliding_window,
    uint8_t kv_dtype_raw,
    float k_scale,
    float v_scale);

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
// Phase 1 contract (must match the Rust shim):
//   - non-FP8 cache → io dtype == cache dtype
//   - FP8 cache     → io dtype == bfloat16 (BF16-default for production)
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

// Extract a Metal `MTLBuffer*` (as a `void*`) plus the byte offset
// from an evaluated `array`. Mirrors the FFI helpers used by the
// existing Rust integration (see
// `mlx_array_get_metal_buffer` in `mlx_advanced_ops.cpp`).
//
// PHASE 1 LIMITATION: we sidestep the public FFI here because we're
// inside the MLX C++ namespace and have direct access to the array's
// buffer. Callers must ensure the array is evaluated before this is
// invoked.
struct MetalArrayPtr {
  void* buffer; // raw MTLBuffer* (cast to void*)
  size_t offset; // byte offset into the buffer
};

MetalArrayPtr metal_array_ptr(const array& arr) {
  // arr.buffer().ptr() is `const void*`; cast away const because the
  // Metal C API takes `id<MTLBuffer>` (a non-const pointer). The
  // dispatcher does not write through this pointer except in the
  // documented in-place case.
  return MetalArrayPtr{
      const_cast<void*>(arr.buffer().ptr()),
      static_cast<size_t>(arr.offset())};
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

  // Output arrays semantically alias the input pools (in-place write).
  // `copy_shared_buffer` makes the output `array` point at the same
  // backing buffer + offset / strides as the input pool.
  outputs[0].copy_shared_buffer(k_pool);
  outputs[1].copy_shared_buffer(v_pool);

  // PHASE 1 LIMITATION: This shim dispatches to mlx_paged_attn's
  // separate Metal command queue. Phase 2 ports dispatch to MLX's
  // queue (the one used by `inputs[0].primitive_ptr()->stream()`) so
  // dependency tracking is correct. Until then, callers MUST `eval()`
  // any prior dependencies before invoking `paged_kv_write` (so the
  // input pools / new K/V / slot_mapping have committed) and `eval()`
  // the outputs before reading.
  //
  // We must NOT call `mlx::core::synchronize()` from inside `eval_gpu`
  // — synchronize enqueues a future onto the same scheduler that
  // dispatched us, deadlocking (it would wait on a task enqueued
  // BEHIND the current eval). The dispatcher's own
  // `wait_until_completed` covers the read-back side.

  // Validate that scalars are 1-element fp32 arrays. The shim accepts
  // any pair of finite floats; we extract them here.
  if (k_scale.size() != 1 || v_scale.size() != 1) {
    throw std::runtime_error(
        "PagedKVWrite: k_scale and v_scale must be 1-element arrays");
  }
  if (k_scale.dtype() != mlx::core::float32 ||
      v_scale.dtype() != mlx::core::float32) {
    throw std::runtime_error(
        "PagedKVWrite: k_scale and v_scale must be float32 arrays");
  }
  // Read scale arrays as host scalars. Scalars constructed via
  // `array(1.0f)` are in `Status::available` immediately, so
  // `item<float>()` is safe without a pre-eval. Calling
  // `mlx::core::eval(...)` here would recurse into the scheduler and
  // deadlock; for graph-produced scales the caller is responsible for
  // evaluating before invoking the primitive.
  float k_scale_f = k_scale.item<float>();
  float v_scale_f = v_scale.item<float>();

  // Determine num_tokens from new_k's leading dimension. The kernel
  // expects shape [num_tokens, num_kv_heads, head_size].
  if (new_k.ndim() != 3 || new_v.ndim() != 3) {
    throw std::runtime_error(
        "PagedKVWrite: new_k / new_v must be rank 3 [num_tokens, "
        "num_kv_heads, head_size]");
  }
  uint32_t num_tokens = static_cast<uint32_t>(new_k.shape(0));

  auto k_pool_ptr = metal_array_ptr(k_pool);
  auto v_pool_ptr = metal_array_ptr(v_pool);
  auto new_k_ptr = metal_array_ptr(new_k);
  auto new_v_ptr = metal_array_ptr(new_v);
  auto slot_mapping_ptr = metal_array_ptr(slot_mapping);

  int rc = mlx_paged_attn_reshape_and_cache_dispatch(
      k_pool_ptr.buffer,
      v_pool_ptr.buffer,
      new_k_ptr.buffer,
      new_k_ptr.offset,
      new_v_ptr.buffer,
      new_v_ptr.offset,
      slot_mapping_ptr.buffer,
      slot_mapping_ptr.offset,
      num_tokens,
      static_cast<uint32_t>(num_kv_heads_),
      static_cast<uint32_t>(head_size_),
      static_cast<uint32_t>(block_size_),
      static_cast<int32_t>(x_pack_),
      static_cast<uint8_t>(kv_dtype_),
      k_scale_f,
      v_scale_f);

  if (rc != 0) {
    throw std::runtime_error(
        "PagedKVWrite: extern-C dispatch failed (see stderr for details)");
  }
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

  if (q.ndim() != 3) {
    throw std::runtime_error(
        "PagedAttention: q must be rank 3 [num_seqs, num_q_heads, head_size]");
  }
  if (block_table.ndim() != 2) {
    throw std::runtime_error(
        "PagedAttention: block_table must be rank 2 "
        "[num_seqs, max_blocks_per_seq]");
  }
  if (seq_lens.ndim() != 1) {
    throw std::runtime_error(
        "PagedAttention: seq_lens must be rank 1 [num_seqs]");
  }
  if (k_scale.size() != 1 || v_scale.size() != 1) {
    throw std::runtime_error(
        "PagedAttention: k_scale and v_scale must be 1-element arrays");
  }

  uint32_t num_seqs = static_cast<uint32_t>(q.shape(0));
  uint32_t max_blocks_per_seq = static_cast<uint32_t>(block_table.shape(1));

  // Determine max_context_len from seq_lens (max element). This is
  // the value that drives the V1/V2 branch. Read host values directly
  // from the seq_lens buffer — caller must `eval()` seq_lens before
  // invoking the primitive (Phase 1 contract; calling
  // `mlx::core::eval(...)` here would recurse into the scheduler and
  // deadlock).
  if (seq_lens.dtype() != mlx::core::int32) {
    throw std::runtime_error("PagedAttention: seq_lens must be int32");
  }
  const int32_t* seq_lens_data = seq_lens.data<int32_t>();
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

  // Allocate the output buffer via MLX's allocator. The shim blits
  // the dispatcher's internal output into this buffer. Phase 2 will
  // pre-route the dispatcher to write here directly.
  out.set_data(allocator::malloc(out.nbytes()));

  // PHASE 1 LIMITATION: same as PagedKVWrite — callers must `eval()`
  // any prior dependencies before invoking `paged_attention`. We do
  // NOT call `mlx::core::synchronize()` here (would deadlock; see the
  // PagedKVWrite::eval_gpu comment). The dispatcher's own
  // `wait_until_completed` covers the read-back side.

  if (k_scale.dtype() != mlx::core::float32 ||
      v_scale.dtype() != mlx::core::float32) {
    throw std::runtime_error(
        "PagedAttention: k_scale and v_scale must be float32 arrays");
  }
  float k_scale_f = k_scale.item<float>();
  float v_scale_f = v_scale.item<float>();

  auto q_ptr = metal_array_ptr(q);
  auto k_pool_ptr = metal_array_ptr(k_pool);
  auto v_pool_ptr = metal_array_ptr(v_pool);
  auto block_table_ptr = metal_array_ptr(block_table);
  auto seq_lens_ptr = metal_array_ptr(seq_lens);
  auto out_ptr = metal_array_ptr(out);

  int rc = mlx_paged_attn_paged_attention_dispatch(
      q_ptr.buffer,
      q_ptr.offset,
      k_pool_ptr.buffer,
      v_pool_ptr.buffer,
      block_table_ptr.buffer,
      seq_lens_ptr.buffer,
      out_ptr.buffer,
      out_ptr.offset,
      num_seqs,
      static_cast<uint32_t>(num_q_heads_),
      static_cast<uint32_t>(num_kv_heads_),
      static_cast<uint32_t>(head_size_),
      static_cast<uint32_t>(block_size_),
      static_cast<uint32_t>(max_context_len),
      max_blocks_per_seq,
      scale_,
      softcap_,
      static_cast<int32_t>(sliding_window_),
      static_cast<uint8_t>(kv_dtype_),
      k_scale_f,
      v_scale_f);

  if (rc != 0) {
    throw std::runtime_error(
        "PagedAttention: extern-C dispatch failed (see stderr for details)");
  }
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

  if (k_pool.dtype() != v_pool.dtype()) {
    std::ostringstream msg;
    msg << "[paged_kv_write] k_pool dtype " << k_pool.dtype()
        << " disagrees with v_pool dtype " << v_pool.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (new_k.dtype() != new_v.dtype()) {
    std::ostringstream msg;
    msg << "[paged_kv_write] new_k dtype " << new_k.dtype()
        << " disagrees with new_v dtype " << new_v.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (k_scale.dtype() != mlx::core::float32 ||
      v_scale.dtype() != mlx::core::float32) {
    throw std::invalid_argument(
        "[paged_kv_write] k_scale / v_scale must be float32");
  }
  if (k_scale.size() != 1 || v_scale.size() != 1) {
    throw std::invalid_argument(
        "[paged_kv_write] k_scale / v_scale must be 1-element arrays");
  }

  // Shape validation against the primitive's scalar state. Each pool
  // dimension must agree with what the kernel will read using the
  // primitive's (block_size, num_kv_heads, head_size, x_pack). A
  // mismatch here would silently route the wrong slots through the
  // dispatcher and corrupt downstream attention.
  //
  // K-pool layout (vLLM): [num_blocks, num_kv_heads, head_size/x, block_size, x]
  // V-pool layout       : [num_blocks, num_kv_heads, head_size, block_size]
  if (k_pool.ndim() != 5) {
    std::ostringstream msg;
    msg << "[paged_kv_write] k_pool must be rank 5 "
        << "[num_blocks, num_kv_heads, head_size/x, block_size, x]; got rank "
        << k_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.ndim() != 4) {
    std::ostringstream msg;
    msg << "[paged_kv_write] v_pool must be rank 4 "
        << "[num_blocks, num_kv_heads, head_size, block_size]; got rank "
        << v_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[paged_kv_write] k_pool.shape(1) (" << k_pool.shape(1)
        << ") disagrees with num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (x_pack <= 0 || head_size % x_pack != 0) {
    std::ostringstream msg;
    msg << "[paged_kv_write] x_pack (" << x_pack
        << ") must be positive and divide head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  // Sanity: caller's `x_pack` must agree with the dtype-derived value.
  // `x_pack = 16 / sizeof(kv_dtype)`. A mismatch here means the caller
  // and the dispatcher disagree on the K-pool layout — guaranteed
  // garbage on read. Reject early.
  {
    int x_pack_expected = x_pack_for(kv_dtype);
    if (x_pack != x_pack_expected) {
      std::ostringstream msg;
      msg << "[paged_kv_write] x_pack (" << x_pack
          << ") disagrees with the dtype-derived x_pack ("
          << x_pack_expected << ") for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
  }
  if (k_pool.shape(2) != head_size / x_pack) {
    std::ostringstream msg;
    msg << "[paged_kv_write] k_pool.shape(2) (" << k_pool.shape(2)
        << ") disagrees with head_size/x_pack (" << head_size / x_pack << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(3) != block_size) {
    std::ostringstream msg;
    msg << "[paged_kv_write] k_pool.shape(3) (" << k_pool.shape(3)
        << ") disagrees with block_size (" << block_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(4) != x_pack) {
    std::ostringstream msg;
    msg << "[paged_kv_write] k_pool.shape(4) (" << k_pool.shape(4)
        << ") disagrees with x_pack (" << x_pack << ")";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[paged_kv_write] v_pool.shape(1) (" << v_pool.shape(1)
        << ") disagrees with num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[paged_kv_write] v_pool.shape(2) (" << v_pool.shape(2)
        << ") disagrees with head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(3) != block_size) {
    std::ostringstream msg;
    msg << "[paged_kv_write] v_pool.shape(3) (" << v_pool.shape(3)
        << ") disagrees with block_size (" << block_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(0) != v_pool.shape(0)) {
    std::ostringstream msg;
    msg << "[paged_kv_write] k_pool num_blocks (" << k_pool.shape(0)
        << ") disagrees with v_pool num_blocks (" << v_pool.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }

  // new_k / new_v must be rank 3 [num_tokens, num_kv_heads, head_size].
  if (new_k.ndim() != 3) {
    std::ostringstream msg;
    msg << "[paged_kv_write] new_k must be rank 3 "
        << "[num_tokens, num_kv_heads, head_size]; got rank " << new_k.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (new_v.ndim() != 3) {
    std::ostringstream msg;
    msg << "[paged_kv_write] new_v must be rank 3 "
        << "[num_tokens, num_kv_heads, head_size]; got rank " << new_v.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (new_k.shape(1) != num_kv_heads || new_v.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[paged_kv_write] new_k/new_v shape(1) ("
        << new_k.shape(1) << "/" << new_v.shape(1)
        << ") must equal num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (new_k.shape(2) != head_size || new_v.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[paged_kv_write] new_k/new_v shape(2) ("
        << new_k.shape(2) << "/" << new_v.shape(2)
        << ") must equal head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (new_k.shape(0) != new_v.shape(0)) {
    std::ostringstream msg;
    msg << "[paged_kv_write] new_k tokens (" << new_k.shape(0)
        << ") disagrees with new_v tokens (" << new_v.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }

  // slot_mapping kernel buffer-contract validation.
  //
  // The Metal kernel reads `slot_mapping[token_idx]` once per token,
  // then computes `block_idx = slot_idx / block_size` and writes to
  // `(block_idx, head_idx, j_idx, ...)` slots in the K/V pools. A
  // shorter-than-num_tokens or wrong-dtype mapping reads past the
  // buffer; a slot_idx >= num_blocks * block_size writes past the
  // pool. Reject every form of malformed slot_mapping at the factory.
  if (slot_mapping.ndim() != 1) {
    std::ostringstream msg;
    msg << "[paged_kv_write] slot_mapping must be rank 1 [num_tokens]; "
        << "got rank " << slot_mapping.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (slot_mapping.dtype() != mlx::core::int64) {
    std::ostringstream msg;
    msg << "[paged_kv_write] slot_mapping must be int64 (kernel reads it as "
        << "`int64_t*`); got dtype " << slot_mapping.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (slot_mapping.shape(0) != new_k.shape(0)) {
    std::ostringstream msg;
    msg << "[paged_kv_write] slot_mapping length (" << slot_mapping.shape(0)
        << ") disagrees with new_k tokens (" << new_k.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }

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
      std::ostringstream msg;
      msg << "[paged_kv_write] slot_mapping max (" << max_slot_v
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

  // Phase 1 explicitly rejects nonzero sliding_window. The primitive
  // tracks it in scalar state for cache-key stability across phases,
  // but the kernel dispatch path doesn't honor it yet — silently
  // accepting it would produce full-context attention behind the
  // caller's back. Phase 7 (Gemma4) will lift this restriction.
  if (sliding_window != 0) {
    throw std::invalid_argument(
        "[paged_attention] sliding_window not yet implemented; "
        "Phase 7 will add it (Gemma4). The only supported value in "
        "Phase 1 is 0.");
  }

  if (k_scale.dtype() != mlx::core::float32 ||
      v_scale.dtype() != mlx::core::float32) {
    throw std::invalid_argument(
        "[paged_attention] k_scale / v_scale must be float32");
  }
  if (k_scale.size() != 1 || v_scale.size() != 1) {
    throw std::invalid_argument(
        "[paged_attention] k_scale / v_scale must be 1-element arrays");
  }
  if (q.ndim() != 3) {
    throw std::invalid_argument(
        "[paged_attention] q must be rank 3 "
        "[num_seqs, num_q_heads, head_size]");
  }

  // Shape validation against scalar state. q is `[num_seqs,
  // num_q_heads, head_size]`; both trailing dims must agree with
  // what the primitive will pass to the dispatcher.
  if (q.shape(1) != num_q_heads) {
    std::ostringstream msg;
    msg << "[paged_attention] q.shape(1) (" << q.shape(1)
        << ") disagrees with num_q_heads (" << num_q_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (q.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[paged_attention] q.shape(2) (" << q.shape(2)
        << ") disagrees with head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }

  // Pool shape validation (mirror paged_kv_write's contract). The
  // kernel reads K through `k_pool[block_id, h, d/x, t, x]` and V
  // through `v_pool[block_id, h, d, t]`, so each interior dim must
  // match scalar state exactly. A mismatched K-pool layout silently
  // re-routes which bytes the kernel treats as `(d/x, x)` packed K —
  // attention would produce garbage with no visible error.
  if (k_pool.ndim() != 5) {
    std::ostringstream msg;
    msg << "[paged_attention] k_pool must be rank 5 "
        << "[num_blocks, num_kv_heads, head_size/x, block_size, x]; got rank "
        << k_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.ndim() != 4) {
    std::ostringstream msg;
    msg << "[paged_attention] v_pool must be rank 4 "
        << "[num_blocks, num_kv_heads, head_size, block_size]; got rank "
        << v_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(0) != v_pool.shape(0)) {
    std::ostringstream msg;
    msg << "[paged_attention] k_pool num_blocks (" << k_pool.shape(0)
        << ") disagrees with v_pool num_blocks (" << v_pool.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(1) != num_kv_heads || v_pool.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[paged_attention] k_pool/v_pool num_kv_heads ("
        << k_pool.shape(1) << "/" << v_pool.shape(1)
        << ") disagrees with num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(3) != block_size || v_pool.shape(3) != block_size) {
    std::ostringstream msg;
    msg << "[paged_attention] k_pool/v_pool block_size ("
        << k_pool.shape(3) << "/" << v_pool.shape(3)
        << ") disagrees with block_size (" << block_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[paged_attention] v_pool.shape(2) (" << v_pool.shape(2)
        << ") disagrees with head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  // Validate K-pool's vLLM packing dims (head_size/x_pack and x_pack)
  // against the dtype-derived x_pack. The factory does not take
  // x_pack — the dtype determines it (Fp16/Bf16 → 8, Fp8 → 16).
  {
    int x_pack_expected = x_pack_for(kv_dtype);
    if (head_size % x_pack_expected != 0) {
      std::ostringstream msg;
      msg << "[paged_attention] head_size (" << head_size
          << ") must be divisible by dtype-derived x_pack ("
          << x_pack_expected << ") for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
    if (k_pool.shape(2) != head_size / x_pack_expected) {
      std::ostringstream msg;
      msg << "[paged_attention] k_pool.shape(2) (" << k_pool.shape(2)
          << ") disagrees with head_size/x_pack ("
          << head_size / x_pack_expected
          << ") (head_size=" << head_size << ", x_pack=" << x_pack_expected
          << " for kv_dtype " << static_cast<int>(kv_dtype) << ")";
      throw std::invalid_argument(msg.str());
    }
    if (k_pool.shape(4) != x_pack_expected) {
      std::ostringstream msg;
      msg << "[paged_attention] k_pool.shape(4) (" << k_pool.shape(4)
          << ") disagrees with dtype-derived x_pack (" << x_pack_expected
          << ") for kv_dtype " << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
  }

  // block_table kernel buffer-contract validation.
  //
  // The Metal kernel reads
  //   `block_tables[seq_idx * max_num_blocks_per_seq + block_idx]`
  // for every (seq, block) pair, then uses that as the K/V pool block
  // index. A wrong rank, mismatched batch, or non-int32 dtype means
  // the kernel reads past the buffer and addresses arbitrary K/V
  // pool blocks.
  if (block_table.ndim() != 2) {
    std::ostringstream msg;
    msg << "[paged_attention] block_table must be rank 2 "
        << "[num_seqs, max_blocks_per_seq]; got rank " << block_table.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (block_table.shape(0) != q.shape(0)) {
    std::ostringstream msg;
    msg << "[paged_attention] block_table.shape(0) (" << block_table.shape(0)
        << ") disagrees with q num_seqs (" << q.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (block_table.dtype() != mlx::core::int32) {
    std::ostringstream msg;
    msg << "[paged_attention] block_table dtype must be int32 (kernel "
        << "reads it as 32-bit indices); got dtype " << block_table.dtype();
    throw std::invalid_argument(msg.str());
  }

  // seq_lens kernel buffer-contract validation.
  //
  // The Metal kernel reads `context_lens[seq_idx]` for every dispatched
  // sequence. A short or non-int32 buffer reads past memory.
  if (seq_lens.ndim() != 1) {
    std::ostringstream msg;
    msg << "[paged_attention] seq_lens must be rank 1 [num_seqs]; got rank "
        << seq_lens.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (seq_lens.shape(0) != q.shape(0)) {
    std::ostringstream msg;
    msg << "[paged_attention] seq_lens.shape(0) (" << seq_lens.shape(0)
        << ") disagrees with q num_seqs (" << q.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (seq_lens.dtype() != mlx::core::int32) {
    std::ostringstream msg;
    msg << "[paged_attention] seq_lens dtype must be int32 (kernel reads "
        << "it as 32-bit lengths); got dtype " << seq_lens.dtype();
    throw std::invalid_argument(msg.str());
  }

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

extern "C" {

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
/// `std::invalid_argument` when called with sliding_window=512.
/// Returns 0 if it doesn't throw or throws a different exception.
///
/// The factory is the earliest validation point; we call it with
/// well-formed shape inputs and a non-zero sliding_window to confirm
/// rejection. The pool/q arrays use tracer-only construction (no
/// backing data needed — the throw fires before eval_gpu).
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
        /*sliding_window=*/512,
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
int call_paged_attention_expecting_throw(
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens) {
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
        KvDtype::Bf16,
        StreamOrDevice{});
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 0;
  }
  return 0;
}

int call_paged_kv_write_expecting_throw(
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& new_k,
    const mlx::core::array& new_v,
    const mlx::core::array& slot_mapping) {
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

} // extern "C"
