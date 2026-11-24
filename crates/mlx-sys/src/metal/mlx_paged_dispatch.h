// Pure-C++ kernel dispatch wrappers that the `PagedKVWrite` /
// `PagedAttention` `Custom` primitives call from inside `eval_gpu`. They
// encode kernels onto MLX's own `metal::CommandEncoder`, so dependency
// tracking is correct without manual synchronization.
//
// Kernel names + threadgroup-memory math + V1/V2 selection mirror the
// Rust dispatcher in `crates/mlx-paged-attn/src/metal/{state,
// reshape_and_cache, paged_attention}.rs`. The Rust dispatcher remains
// in place as the synchronous fallback path: the adapter's raw-Metal
// `PagedKVCacheAdapter::update_keys_values` goes through `LayerKVPool` →
// Rust `dispatch_*`, outside MLX's graph scheduler. The `Custom`
// primitives dispatched here are the graph-native default: every paged
// family's pure-Rust forward (qwen3, qwen3.5 dense/MoE, lfm2, gemma4)
// emits them via `PagedKVCacheAdapter::update_keys_values_native` /
// `gather_kv_for_decode_graph` (`gather_kv_for_prefill_chunk` is
// family-dependent: opt-in default-OFF on lfm2, absent on qwen3), so the
// K/V write and attention read ride MLX's lazy graph with no per-layer
// host sync. They also stay compile-traceable (usable inside
// `mlx::core::compile`), covered by the compile-trace smoke tests in
// `mlx_paged_ops.cpp`.
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
#include <string>
#include <vector>

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
/// kernel names: Fp16 → "half", Bf16 → "bfloat16_t", Int8 → "char".
/// (The C++ `KvDtype` enum in `mlx_paged_ops.h` is the public-facing
/// version; this internal enum is equivalent but lives in its own
/// namespace so we don't crowd the public header.)
enum class KvDtype : uint8_t {
  Fp16 = 0,
  Bf16 = 1,
  Int8 = 3,
};

/// Per-dispatch compute-route hint. This changes only which kernel reads the
/// authoritative paged K/V pool; it never changes cache layout or ownership.
enum class PagedAttentionRouteHint : uint8_t {
  Auto = 0,
  // Legacy ABI name for the canonical grouped D512 production route.
  ForceD512Staged = 1,
  ForceGeneric = 2,
  ForceD128 = 3,
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
/// `k_scale` / `v_scale` are the primitive's scale outputs (sharing the
/// scale inputs' buffers). Int8 writes one fp32 scale per stored row into
/// `[num_blocks, num_kv_heads, block_size]` (Splash
/// `splash_q8_scale_index`); Fp16/Bf16 ignore them.
///
/// Int8 runs `reshape_and_cache_int8_kv_bfloat16_t_hd{128,256}`: one
/// threadgroup of `head_size` threads per K or V row, `2 * num_tokens *
/// num_kv_heads` threadgroups (Splash `PagedAttention::addPrefillStore`).
///
/// `values_token_major` runs the `_vt` instances: K and V both
/// `[num_blocks, num_kv_heads, block_size, head_size]` (Splash
/// `splash_kv_key_element`), `x_pack == head_size`.
void dispatch_reshape_and_cache(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    const mlx::core::array& new_k,
    const mlx::core::array& new_v,
    mlx::core::array& k_pool,
    mlx::core::array& v_pool,
    const mlx::core::array& slot_mapping,
    mlx::core::array& k_scale,
    mlx::core::array& v_scale,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int block_size,
    int x_pack,
    KvDtype kv_dtype,
    bool values_token_major);

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
///
/// `live_blocks`: the distinct pool blocks the block tables reach. The
/// pool and scale planes are charged to the command buffer for that many
/// blocks, not the whole pool.
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
    int64_t live_blocks,
    float scale,
    float softcap,
    int sliding_window,
    KvDtype kv_dtype,
    PagedAttentionRouteHint route_hint,
    uint32_t grouped_stripes = 0,
    const mlx::core::array* score_scales = nullptr);

// `score_scales` (INT8 pages only, may be null): f32 [query rows, query
// heads], each the score scale of its row's queries in place of `scale`.
// `paged_query_codes` writes it beside the INT8 query codes of rows at or
// past the position rule, and `scale` for rows it passes through.

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
    int64_t live_blocks,
    float scale,
    float softcap,
    int sliding_window,
    KvDtype kv_dtype,
    const mlx::core::array* score_scales = nullptr);

/// Parameters of the in-place prefill / verify attention (Splash
/// SplashPrefillAttentionParams over the mlx-node pool). Mirrored by
/// `PagedPrefillParams` in `paged_prefill_attention.metal`.
struct PagedPrefillParams {
  uint32_t committed_tokens;
  uint32_t rows;
  uint32_t chunk_stride;
  uint32_t table_entries;
  uint32_t split_count;
  uint32_t max_splits;
  uint32_t block_size;
  uint32_t kv_heads;
};
static_assert(sizeof(PagedPrefillParams) == 32, "matches the Metal struct");

/// Query rows of a tile (Splash SPLASH_PREFILL_ATTENTION_TILE_ROWS).
constexpr int kPagedPrefillTileRows = 8;
/// Rows of one chunk (Splash SPLASH_PREFILL_TOKEN_BUDGET).
constexpr int kPagedPrefillMaxRows = 2048;
/// Splits of a prefill / verify launch (Splash
/// SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS / kVerifyMaximumSplits).
constexpr int kPagedPrefillMaxSplits = 32;
constexpr int kPagedVerifyMaxSplits = 128;
/// Rows of a verify launch (Splash SPLASH_TARGET_VERIFY_ROWS, up to 16 for verify trees).
constexpr int kPagedVerifyRows = 16;

/// Whether a (query heads per KV head, head dim) pair has an instance.
bool paged_prefill_attention_geometry(int group, int head_size);

/// The one geometry with a keys-token-major split entry (INT8, block 32):
/// dense Qwen3.5 heads 24/4 at D 256.
constexpr int kPagedPrefillKeysTokenMajorGroup = 6;
constexpr int kPagedPrefillKeysTokenMajorHeadDim = 256;
bool paged_prefill_keys_token_major_geometry(int group, int head_size);

/// Every split and reduce function the dispatch can request.
std::vector<std::string> paged_prefill_attention_kernel_names();

/// The pipeline of split or reduce function `name`. A pipeline that does not
/// build throws, naming the kernel and the device's Apple GPU family.
MTL::ComputePipelineState* paged_prefill_pipeline(
    mlx::core::metal::Device& device,
    const std::string& name);

/// Builds the split and reduce pipelines of every instance this device runs
/// (the q8q entries only where `paged_prefill_int8_queries_supported`).
int64_t build_paged_prefill_attention_pipelines(mlx::core::metal::Device& device);

/// Whether the device runs the q8q (INT8-query) split entries: Apple10 and
/// later (Splash `AttentionCalibration::int8QueriesFromPages`).
bool paged_prefill_int8_queries_supported(mlx::core::metal::Device& device);

/// Floats of the workspace one launch writes: partials then statistics
/// (Splash PagedAttention::prefillPlan, exact).
int64_t paged_prefill_workspace_floats(
    int rows, int splits, int num_q_heads, int head_size);

/// Encodes the split then the reduce. `q_staged` is
/// [kv heads][chunk_stride][group][D] bf16, `out` [rows][query heads][D]
/// bf16, `workspace` the f32 partials + statistics the split writes and the
/// reduce reads. `int8_queries` (INT8 pages only) takes the q8q split. A
/// rank-4 `k_pool` is the keys-token-major page and takes its `_ktm` split.
void dispatch_paged_prefill_attention(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::array& out,
    mlx::core::array& workspace,
    const mlx::core::array& q_staged,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    const mlx::core::array& block_table,
    const PagedPrefillParams& params,
    int num_q_heads,
    int head_size,
    KvDtype cache_dtype,
    bool int8_queries,
    int64_t live_blocks);

/// Parameters of `paged_prefill_attention_query_codes_d{128,256}`. Mirrored by the Metal
/// struct of the same name.
struct PagedQueryCodesParams {
  uint32_t rows;
  uint32_t query_heads;
  uint32_t from;
  float scale;
};
static_assert(sizeof(PagedQueryCodesParams) == 16, "matches the Metal struct");

/// Encodes the decode kernels' INT8 query codes: `q` [rows, query heads, D]
/// bf16 at `positions` [rows] int32; rows at or past `params.from` write
/// codes (bf16 integers) and the tile's score scale, rows below copy `q`
/// and `params.scale`.
void dispatch_paged_query_codes(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    const mlx::core::array& q,
    const mlx::core::array& positions,
    mlx::core::array& codes,
    mlx::core::array& score_scales,
    const PagedQueryCodesParams& params,
    int head_size);

// =============================================================================
// Inkling-2-Mini Paged Attention (relative bias + LSE + optional gate/sink, KTM INT8 & BF16)
// =============================================================================

/// Extra params for Inkling2 decode/varlen kernels. Mirrored by
/// `Inkling2AttnExtraParams` in `paged_attention_inkling2.h`.
struct Inkling2AttnExtraParams {
  uint32_t has_gate_sink;
  uint32_t gate_row_stride;
  uint32_t gate_is_f32;
  uint32_t sink_row_stride;
  uint32_t sink_is_f32;
  uint32_t write_lse;
  uint32_t rel_mode; // 0 = none, 1 = precomputed [T,H,rel_extent], 2 = on-the-fly r [T,H,16] x w [rel_extent,16]
  uint32_t has_tree; // 0 = linear causal chain, 1 = verify tree (tree_rows [total_q, 2] u32)
};
static_assert(sizeof(Inkling2AttnExtraParams) == 32, "matches the Metal struct");

/// Parameters of `inkling2_paged_prefill_*`. Mirrored by
/// `Inkling2PagedPrefillParams` in `paged_prefill_attention_inkling2.h`.
struct Inkling2PagedPrefillParams {
  uint32_t committed_tokens;
  uint32_t rows;
  uint32_t chunk_stride;
  uint32_t table_entries;
  uint32_t split_count;
  uint32_t max_splits;
  uint32_t block_size;
  uint32_t kv_heads;
  uint32_t sliding_window;
  uint32_t rel_extent;
  uint32_t rel_bias_is_f32;
  uint32_t rel_mode;
  uint32_t has_gate_sink;
  uint32_t gate_row_stride;
  uint32_t gate_is_f32;
  uint32_t sink_row_stride;
  uint32_t sink_is_f32;
  uint32_t write_lse;
  uint32_t has_tree;
  uint32_t _pad_tree;
};
static_assert(sizeof(Inkling2PagedPrefillParams) == 80, "matches the Metal struct");

/// Dispatches Inkling2 relative-bias + LSE varlen paged attention (V1 ps=0 or
/// V2 ps=512 + shared FP32 reducer) for INT8 32-token KTM or BF16 vLLM pools,
/// with optional on-the-fly `r @ rel_bias_w^T`, optional fused gate+sink epilogue,
/// and optional verify-tree metadata (`tree_rows [total_q, 2]` u32).
void dispatch_inkling2_paged_attention_auto(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    mlx::core::array* lse_out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& cu_seqlens_q,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    const mlx::core::array* rel_bias,
    const mlx::core::array* rel_bias_w,
    const mlx::core::array* gate,
    const mlx::core::array* sink,
    const Inkling2AttnExtraParams& extra,
    int num_seqs,
    int total_queries,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int64_t live_blocks,
    float scale,
    int sliding_window,
    int rel_extent,
    KvDtype kv_dtype,
    const mlx::core::array* score_scales = nullptr,
    const mlx::core::array* tree_rows = nullptr);

/// Whether the device can run the G=3, D=128 Inkling2 MPP prefill pipelines.
bool inkling2_paged_prefill_mpp_supported(mlx::core::metal::Device& device);

/// Every Inkling2 paged-attention kernel name in `paged_attn.metallib` (decode & prefill).
std::vector<std::string> inkling2_paged_kernel_names();

/// Clears this thread's persistent V2 scratch buffers for Inkling2 paged attention.
void clear_inkling2_v2_scratch();

/// Dispatches Inkling2 G=3, D=128 in-place MPP paged prefill (split + reduce),
/// writing BF16 `out [rows, num_q_heads, 128]` (optionally gate+sink fused) and
/// optional F32 `lse_out [rows, num_q_heads]`. Uses single-tile `_r16` for 9..=16 rows.
void dispatch_inkling2_paged_prefill_attention(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::array& out,
    mlx::core::array* lse_out,
    mlx::core::array& workspace,
    const mlx::core::array& q_staged,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    const mlx::core::array& block_table,
    const mlx::core::array* rel_bias,
    const mlx::core::array* rel_bias_w,
    const mlx::core::array* gate,
    const mlx::core::array* sink,
    const Inkling2PagedPrefillParams& params,
    int num_q_heads,
    int head_size,
    KvDtype cache_dtype,
    bool int8_queries,
    int64_t live_blocks,
    const mlx::core::array* tree_rows = nullptr);

} // namespace mlx::core::fast::paged

#endif // defined(__APPLE__)
