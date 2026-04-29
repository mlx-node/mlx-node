// Phase 1 of the paged-attention compile integration.
//
// This header declares two MLX `Custom` primitive subclasses
// (`PagedKVWrite`, `PagedAttention`) and the matching public free
// functions that emit them. Their `eval_gpu` implementations call into
// a temporary `extern "C"` shim exposed by the `mlx-paged-attn` Rust
// crate (see `crates/mlx-paged-attn/src/extern_c.rs`).
//
// PHASE 1 LIMITATION: this shim dispatches to mlx_paged_attn's separate
// Metal command queue. Phase 2 ports dispatch to MLX's queue (the one
// used by `inputs[0].primitive_ptr()->stream()`) so dependency tracking
// is correct. Until then, `eval_gpu` synchronizes MLX before the call
// and the dispatcher's own `wait_until_completed` covers the read-back.
// In particular, callers MUST `eval()` any prior dependencies before
// invoking `paged_kv_write`/`paged_attention`, and `eval()` the outputs
// before reading them outside an MLX graph.

#pragma once

#include <cstdint>

#include "mlx/fast_primitives.h"
#include "mlx/utils.h" // StreamOrDevice

namespace mlx::core::fast {

/// On-cache storage element type. Must match
/// `crates/mlx-paged-attn/src/extern_c.rs::KvDtypeC` value-by-value.
enum class KvDtype : uint8_t {
  Fp16 = 0,
  Bf16 = 1,
  Fp8 = 2,
};

/// `PagedKVWrite` writes a chunk of new K/V tokens into the per-layer
/// block-paged K/V pool at positions specified by `slot_mapping`.
///
/// Inputs (in order):
///   0: `k_pool`    — `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
///   1: `v_pool`    — `[num_blocks, num_kv_heads, head_size, block_size]`
///   2: `new_k`     — `[num_tokens, num_kv_heads, head_size]`
///   3: `new_v`     — `[num_tokens, num_kv_heads, head_size]`
///   4: `slot_mapping` — `[num_tokens]` of int64
///   5: `k_scale`   — `[1]` fp32 (placeholder for non-FP8)
///   6: `v_scale`   — `[1]` fp32 (placeholder for non-FP8)
///
/// Outputs (in order):
///   0: `k_pool'`   — semantically the same buffer as `k_pool` (in-place
///                    write; output array shares the input's allocation)
///   1: `v_pool'`   — same, for the value pool
///
/// The primitive's scalar state participates in the compile cache key:
/// re-tracing with different `block_size` / `kv_dtype` etc. yields a
/// new compiled graph; re-tracing with the same scalars but different
/// runtime tensor contents reuses the cached graph.
class PagedKVWrite : public Custom {
 public:
  PagedKVWrite(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int block_size,
      int num_kv_heads,
      int head_size,
      int x_pack,
      KvDtype kv_dtype)
      : Custom(stream, std::move(fallback)),
        block_size_(block_size),
        num_kv_heads_(num_kv_heads),
        head_size_(head_size),
        x_pack_(x_pack),
        kv_dtype_(kv_dtype) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("PagedKVWrite CPU NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  // PagedKVWrite is inference-only. Override vjp so the diagnostic
  // points at this primitive rather than falling through to the
  // generic `Custom::vjp` (which would silently re-run the fallback
  // for gradient computation).
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;

  bool is_equivalent(const Primitive& other) const override;

  DEFINE_NAME(PagedKVWrite);

  auto state() const {
    return std::make_tuple(
        nullptr,
        block_size_,
        num_kv_heads_,
        head_size_,
        x_pack_,
        static_cast<uint8_t>(kv_dtype_));
  }

 private:
  int block_size_;
  int num_kv_heads_;
  int head_size_;
  int x_pack_;
  KvDtype kv_dtype_;
};

/// `PagedAttention` computes attention with K/V gathered from
/// block-paged storage via `block_table` + `seq_lens`. Auto-picks
/// V1/V2 kernels based on the runtime `max_context_len` (see
/// `dispatch_paged_attention_auto` in the Rust shim).
///
/// Inputs (in order):
///   0: `q`          — `[num_seqs, num_q_heads, head_size]`
///   1: `k_pool`     — same as PagedKVWrite
///   2: `v_pool`     — same as PagedKVWrite
///   3: `block_table` — `[num_seqs, max_blocks_per_seq]` int32
///   4: `seq_lens`   — `[num_seqs]` int32
///   5: `k_scale`    — `[1]` fp32 (placeholder for non-FP8)
///   6: `v_scale`    — `[1]` fp32 (placeholder for non-FP8)
///
/// Outputs (in order):
///   0: `attn_out`   — `[num_seqs, num_q_heads, head_size]` in io dtype
///                     (Fp16/Bf16 for non-FP8; Bf16 default for FP8)
class PagedAttention : public Custom {
 public:
  PagedAttention(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float scale,
      float softcap,
      int block_size,
      int num_q_heads,
      int num_kv_heads,
      int head_size,
      int sliding_window,
      KvDtype kv_dtype)
      : Custom(stream, std::move(fallback)),
        scale_(scale),
        softcap_(softcap),
        block_size_(block_size),
        num_q_heads_(num_q_heads),
        num_kv_heads_(num_kv_heads),
        head_size_(head_size),
        sliding_window_(sliding_window),
        kv_dtype_(kv_dtype) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("PagedAttention CPU NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;

  bool is_equivalent(const Primitive& other) const override;

  DEFINE_NAME(PagedAttention);

  auto state() const {
    return std::make_tuple(
        nullptr,
        scale_,
        softcap_,
        block_size_,
        num_q_heads_,
        num_kv_heads_,
        head_size_,
        sliding_window_,
        static_cast<uint8_t>(kv_dtype_));
  }

 private:
  float scale_;
  float softcap_;
  int block_size_;
  int num_q_heads_;
  int num_kv_heads_;
  int head_size_;
  int sliding_window_;
  KvDtype kv_dtype_;
};

// =============================================================================
// Public free functions (the user-facing API for emitting these primitives)
// =============================================================================

/// Emit a `PagedKVWrite` primitive. Returns `(k_pool', v_pool')` —
/// arrays that semantically alias the input pools (the primitive
/// writes in-place). `k_scale` / `v_scale` MUST be `[1]` fp32 arrays
/// even when `kv_dtype != Fp8` (callers pass `array(1.0f)`
/// placeholders so the FP8 calibration path can flow through compile
/// naturally without a separate variant).
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
    StreamOrDevice s = {});

/// Emit a `PagedAttention` primitive. Returns the attention output.
/// `softcap = 0.0` disables soft-capping (the Rust shim translates to
/// the kernel's `softcapping = 1.0` "disabled" sentinel).
/// `sliding_window`: 0 = disabled (only supported value in Phase 1;
/// Phase 7 adds support for nonzero values for Gemma4). The factory
/// throws `std::invalid_argument` if a nonzero value is passed.
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
    StreamOrDevice s = {});

} // namespace mlx::core::fast
