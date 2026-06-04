// MLX `Custom` primitives for block-paged KV cache write + attention.
//
// Declares two MLX `Custom` primitive subclasses (`PagedKVWrite`,
// `PagedAttention`) and the matching public free functions that emit
// them. Their `eval_gpu` paths dispatch onto MLX's own command encoder,
// so MLX's dependency tracking is correct: callers do not need to
// `eval()` ancestors before invoking the primitives nor `eval()` the
// outputs before reading them inside an MLX graph.

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

  // Inference-only. Override vjp so the diagnostic points at this
  // primitive rather than falling through to the generic `Custom::vjp`
  // (which would silently re-run the fallback for gradient computation).
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
/// `dispatch_paged_attention_auto`).
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
///
/// Strict input contract — every dimension and dtype is validated at
/// the factory and a mismatch throws `std::invalid_argument`:
///
///   - `k_pool` rank 5 `[num_blocks, num_kv_heads, head_size/x_pack,
///     block_size, x_pack]`. `x_pack = 16 / sizeof(kv_dtype)`
///     (Fp16/Bf16 → 8, Fp8 → 16). `x_pack` arg must equal that.
///   - `v_pool` rank 4 `[num_blocks, num_kv_heads, head_size, block_size]`.
///   - `k_pool.shape(0) == v_pool.shape(0)` (num_blocks parity).
///   - `new_k`/`new_v` rank 3 `[num_tokens, num_kv_heads, head_size]`,
///     identical shape, identical dtype.
///   - `slot_mapping` rank 1 `[num_tokens]`, dtype `int64`,
///     `shape(0) == new_k.shape(0)`. The kernel reads
///     `slot_mapping[token_idx]` as `int64_t*` and uses
///     `block_idx = slot_idx / block_size`; a mismatch reads/writes
///     past the K/V allocation.
///   - `slot_mapping`'s max value MUST be `< num_blocks * block_size`.
///     The factory eval-checks this (skipped during MLX tracing).
///   - `k_scale`/`v_scale` rank 0/1 of size 1, dtype `float32`.
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
/// `softcap = 0.0` disables soft-capping (translated to the kernel's
/// `softcapping = 1.0` "disabled" sentinel).
/// `sliding_window`: 0 = disabled; nonzero masks K positions older than
/// `context_len - sliding_window`. The factory throws
/// `std::invalid_argument` for negative values.
///
/// Strict input contract — every dimension and dtype is validated at
/// the factory and a mismatch throws `std::invalid_argument`:
///
///   - `q` rank 3 `[num_seqs, num_q_heads, head_size]`.
///   - `k_pool` rank 5 `[num_blocks, num_kv_heads, head_size/x_pack,
///     block_size, x_pack]`, with `x_pack` derived from `kv_dtype`
///     (Fp16/Bf16 → 8, Fp8 → 16). `head_size` must be divisible by
///     `x_pack`.
///   - `v_pool` rank 4 `[num_blocks, num_kv_heads, head_size, block_size]`.
///   - `k_pool.shape(0) == v_pool.shape(0)` (num_blocks parity).
///   - `block_table` rank 2 `[num_seqs, max_blocks_per_seq]`, dtype
///     `int32`, with `shape(0) == q.shape(0)`. The kernel addresses
///     `block_tables + seq_idx * max_blocks_per_seq` and reads each
///     entry as a 32-bit block index.
///   - `seq_lens` rank 1 `[num_seqs]`, dtype `int32`, with
///     `shape(0) == q.shape(0)`. The kernel reads
///     `context_lens[seq_idx]` for every dispatched sequence.
///   - `k_scale`/`v_scale` size 1, dtype `float32`.
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

/// Ragged-Q sibling of `PagedAttention`. Accepts a flat
/// `[total_queries, num_q_heads, head_size]` Q tensor and a
/// `[num_seqs + 1]` int32 `cu_seqlens_q` array (FlashAttention-style
/// cumulative query lengths). Each query token's effective causal
/// context is `context_lens[seq] - q_len[seq] + q_pos_in_seq + 1`,
/// which is what the MTP verify path needs: draft token `t` sees the
/// committed prefix plus `t` ancestors but NOT the speculative tail.
///
/// Used by the compile-traceable MTP verify forward. Kernel-name
/// selection mirrors the single-row `PagedAttention` primitive but
/// routes to the `paged_attention_varlen[_v2_reduce]` kernels in
/// `paged_attn.metallib`.
class PagedAttentionVarlen : public Custom {
 public:
  PagedAttentionVarlen(
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
    throw std::runtime_error("PagedAttentionVarlen CPU NYI");
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

  DEFINE_NAME(PagedAttentionVarlen);

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

/// Emit a `PagedAttentionVarlen` primitive. Returns the attention
/// output of shape `[total_queries, num_q_heads, head_size]`.
///
/// Strict input contract:
///   - `q` rank 3 `[total_queries, num_q_heads, head_size]`.
///   - `k_pool` / `v_pool` follow the same layout as `paged_attention`.
///   - `block_table` rank 2 `[num_seqs, max_blocks_per_seq]` int32.
///   - `seq_lens` rank 1 `[num_seqs]` int32.
///   - `cu_seqlens_q` rank 1 `[num_seqs + 1]` int32 with
///     `cu_seqlens_q[0] == 0` and `cu_seqlens_q[num_seqs] == total_queries`.
///   - `k_scale` / `v_scale` size 1 float32.
array paged_attention_varlen(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& cu_seqlens_q,
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
