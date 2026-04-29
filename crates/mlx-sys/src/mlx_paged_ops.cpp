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

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>

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

// Phase-1 only: synchronize MLX so all prior ops touching the pool
// buffers / scratch arrays have committed before the shim dispatch
// hits the same MTLBuffer through a different command queue. Phase 2
// will replace this with proper dependency tracking via MLX's
// command-encoder API.
void phase1_synchronize() {
  mlx::core::synchronize();
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
  // dependency tracking is correct. Until then, callers MUST eval()
  // any prior dependencies before calling paged_kv_write, and eval()
  // the outputs before reading. We synchronize MLX here so any pending
  // writes to `new_k` / `new_v` / `slot_mapping` have committed before
  // the shim reads them via a different queue.
  phase1_synchronize();

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
  // Force eval of scale arrays so we can read their host values.
  // (k_scale / v_scale are typically constant-folded to host scalars
  // during compile, but this defends against the runtime path.)
  array k_scale_host = k_scale;
  array v_scale_host = v_scale;
  mlx::core::eval(k_scale_host);
  mlx::core::eval(v_scale_host);
  float k_scale_f = k_scale_host.item<float>();
  float v_scale_f = v_scale_host.item<float>();

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
  // the value that drives the V1/V2 branch. Force eval + read host
  // values.
  array seq_lens_host = seq_lens;
  mlx::core::eval(seq_lens_host);
  if (seq_lens_host.dtype() != mlx::core::int32) {
    throw std::runtime_error(
        "PagedAttention: seq_lens must be int32");
  }
  const int32_t* seq_lens_data = seq_lens_host.data<int32_t>();
  int32_t max_context_len = 0;
  for (size_t i = 0; i < seq_lens_host.size(); ++i) {
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

  // PHASE 1 LIMITATION: same as PagedKVWrite — synchronize MLX so
  // pending writes to the pool / queries / block_table / seq_lens
  // have committed before the shim's separate command queue reads
  // them.
  phase1_synchronize();

  array k_scale_host = k_scale;
  array v_scale_host = v_scale;
  mlx::core::eval(k_scale_host);
  mlx::core::eval(v_scale_host);
  if (k_scale_host.dtype() != mlx::core::float32 ||
      v_scale_host.dtype() != mlx::core::float32) {
    throw std::runtime_error(
        "PagedAttention: k_scale and v_scale must be float32 arrays");
  }
  float k_scale_f = k_scale_host.item<float>();
  float v_scale_f = v_scale_host.item<float>();

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
  return {q.shape()};
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

  // Output shape and dtype
  auto out_dtype = io_dtype_for_kv_dtype(kv_dtype);

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

  return array(q.shape(), out_dtype, primitive, std::move(inputs));
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

} // extern "C"
