// Pure-C++ kernel dispatch wrappers that the `PagedKVWrite` /
// `PagedAttention` `Custom` primitives call from inside `eval_gpu`. They
// encode kernels onto MLX's own `metal::CommandEncoder`, so dependency
// tracking is correct without manual synchronization.
//
// Kernel names + threadgroup-memory math + V1/V2 selection mirror the
// Rust dispatcher in `crates/mlx-paged-attn/src/metal/{state,
// reshape_and_cache, paged_attention}.rs`. The Rust dispatcher remains
// in place: it is the active production paged path for Qwen3, LFM2, and
// Gemma4 (whose paged forward goes through `PagedKVCacheAdapter` →
// `LayerKVPool` → Rust `dispatch_*`, NOT these C++ Custom primitives).
// The C++ port here is exclusively for the compile-traceable primitives
// used inside the Qwen3.5 dense / MoE C++ compile graphs
// (`mlx_qwen35_init_paged` / `mlx_qwen35_moe_init_paged`).
//
// Notes:
//   - The .metallib for these kernels is compiled by `mlx-sys/build.rs`
//     from `crates/mlx-paged-attn/metal/*.metal`. It ships colocated
//     with `mlx.metallib` (e.g. in `packages/core/`) and is loaded via
//     MLX's `Device::get_library(name, path)` path overload.
//   - `Device::get_library` and `::get_kernel` cache by name, so we get
//     pipeline reuse for free.
//   - The Custom primitive's `eval_gpu` is responsible for argument
//     validation, output allocation (`array::set_data`), and any
//     pre-dispatch host-side checks. These functions only do the kernel
//     dispatch.

#pragma once

#include <cstdint>

#include "mlx/array.h"

// These dispatch wrappers take MLX's Metal `CommandEncoder` / `Device` by
// reference, so the whole declaration block is Metal-only. The two TUs that
// include this header (`mlx_paged_dispatch.cpp`, `mlx_paged_ops.cpp`) are
// excluded from the non-Apple build, but guard the include + decls anyway so
// the header is safe to parse on any host.
#if defined(__APPLE__)

#include "mlx/backend/metal/device.h"

namespace mlx::core::fast::paged {

/// On-cache element type. Matches the Rust `MetalDtype` enum used in
/// kernel names: Fp16 → "half", Bf16 → "bfloat16_t", Fp8 → "uchar".
/// (The C++ `KvDtype` enum in `mlx_paged_ops.h` is the public-facing
/// version; this internal enum is equivalent but lives in its own
/// namespace so we don't crowd the public header.)
enum class KvDtype : uint8_t {
  Fp16 = 0,
  Bf16 = 1,
  Fp8 = 2,
};

/// Dispatch the `reshape_and_cache` kernel onto the MLX command
/// encoder. Writes `num_tokens` of `(new_k, new_v)` into the per-layer
/// block-paged K/V pools at the slot ids supplied by `slot_mapping`.
///
/// All array references must already be evaluated (MLX guarantees this
/// before invoking `eval_gpu`). The K-pool layout the kernel expects
/// (with `x = 16 / sizeof(KV_T)`):
///   - K-pool: `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
///   - V-pool: `[num_blocks, num_kv_heads, head_size, block_size]`
///   - new_k/new_v: `[num_tokens, num_kv_heads, head_size]`
///   - slot_mapping: `[num_tokens]` of int64
///
/// The K/V pools are written in-place; the caller's `Custom::eval_gpu`
/// has already arranged for the output arrays to share the input
/// pools' buffers via `copy_shared_buffer`. We pass them as outputs to
/// the encoder so MLX correctly tracks the in-place mutation when
/// scheduling subsequent commands.
///
/// `k_scale` / `v_scale` are FP32 scalar arrays of size 1, only
/// consumed by the kernel for FP8 caches (kernel ignores them
/// otherwise) — but they participate in the compile graph either way.
void dispatch_reshape_and_cache(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    const mlx::core::array& new_k,
    const mlx::core::array& new_v,
    mlx::core::array& k_pool,
    mlx::core::array& v_pool,
    const mlx::core::array& slot_mapping,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int block_size,
    int x_pack,
    KvDtype kv_dtype);

/// Dispatch the paged-attention kernel (V1 if `max_context_len <=
/// PARTITION_SIZE`, else V2 in two phases) onto the MLX command
/// encoder. Writes attention output to `out` (an array MLX has already
/// `set_data`-allocated to shape `[num_seqs, num_q_heads, head_size]`).
///
/// V2 needs three temporary arrays (`exp_sums`, `max_logits`,
/// `tmp_out`) which we allocate and hand to MLX via
/// `encoder.add_temporaries(...)` so MLX manages their lifetime. They
/// are not returned.
///
/// `softcap == 0.0` is the disabled sentinel (no soft-capping). The
/// underlying kernel uses `1.0` as its disabled sentinel; this function
/// translates.
///
/// `sliding_window`: 0 = disabled, nonzero masks older K positions;
/// negative is rejected.
void dispatch_paged_attention_auto(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    float scale,
    float softcap,
    int sliding_window,
    KvDtype kv_dtype);

/// Ragged-Q dispatcher. Writes `[total_queries, num_q_heads,
/// head_size]` attention output to `out`. `cu_seqlens_q` carries the
/// per-sequence query slice boundaries. V1/V2 selection mirrors the
/// single-row dispatcher (`max_context_len <= PARTITION_SIZE` → V1).
///
/// `block_table` and `seq_lens` keep the same layouts as the
/// single-row entrypoint.
void dispatch_paged_attention_varlen_auto(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& cu_seqlens_q,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_seqs,
    int total_queries,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    float scale,
    float softcap,
    int sliding_window,
    KvDtype kv_dtype);

} // namespace mlx::core::fast::paged

#endif // defined(__APPLE__)
