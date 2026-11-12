// Chunked prefill and verify attention in place over the block-paged KV pool
// (Splash kernels/common/paged_attention_tile.h and
// kernels/prefill/paged_attention.metal): the chunk's rows are stored into
// the pool first, then every query row reads its whole history, its own row
// included, back from the pool pages (INT8 rows with their f32 scales, or
// BF16).
//
// A threadgroup owns one KV head, one 8-row query tile (or a part of one,
// tile_part) and one history split; the GQA group is fused, M = rows x
// query heads per KV head <= 64. A page is 32 tokens (Splash
// SplashKvPageTokens): 32 / block_size pool blocks. Each split writes fp32
// partials and {max, sum} statistics for a reduce that combines the splits
// in their order.
//
// Pool layout (vLLM): K [blocks][KV][D / X][block_size][X] with X 16 bytes of
// elements, V [blocks][KV][D][block_size], INT8 scales [blocks][KV][block_size].
// K is staged into threadgroup memory token-major per page; V is a device
// operand of the PV product read in place, 16 tokens at a time.
//
// Queries are KV-head-major [KV][chunk_stride][G][D] (Splash Staging), so a
// tile's fused rows form one contiguous M x D tensor. The reduce writes the
// output token-major [rows][query heads][D].

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

#define PAGED_PREFILL_PAGE_TOKENS 32u
#define PAGED_PREFILL_TILE_ROWS 8u
#define PAGED_PREFILL_MAX_ROWS 2048u
#define PAGED_PREFILL_MAX_SPLITS 128u
#define PAGED_PREFILL_HALF_TOKENS 16u
#define PAGED_PREFILL_THREADS 256u

// Mirrored by the host's PagedPrefillParams (mlx_paged_dispatch.h).
struct PagedPrefillParams {
  uint committed_tokens;
  uint rows;
  uint chunk_stride;
  uint table_entries;
  uint split_count;
  uint max_splits;
  uint block_size;
  uint kv_heads;
};

// Splash SPLASH_ATTENTION_TILE_PARTS: the fewest parts of the 8-row tile with
// at most 64 fused rows each.
constexpr uint paged_prefill_tile_parts(uint group) {
  return group <= 8u ? 1u : group <= 16u ? 2u : group <= 32u ? 4u : 8u;
}

template <uint HeadDim> constexpr float paged_prefill_scale() {
  static_assert(HeadDim == 128 || HeadDim == 256, "a score scale for each compiled head dimension");
  return HeadDim == 256 ? 0.0625f : 0.08838834764831845f;
}

// Splash splash_prefill_attention_contract_valid; an invalid contract makes
// every tile inactive.
inline bool paged_prefill_contract_valid(constant PagedPrefillParams &params) {
  return params.rows > 0 && params.rows <= PAGED_PREFILL_MAX_ROWS &&
         params.chunk_stride >= params.rows &&
         params.chunk_stride <= PAGED_PREFILL_MAX_ROWS &&
         params.chunk_stride % PAGED_PREFILL_PAGE_TOKENS == 0 &&
         (params.block_size == 16 || params.block_size == 32) &&
         ulong(params.table_entries) * params.block_size >=
             ulong(params.committed_tokens) + params.rows &&
         params.split_count > 0 && params.split_count <= params.max_splits &&
         params.max_splits <= PAGED_PREFILL_MAX_SPLITS;
}

// Splash SplashAttentionTile, plus the chunk row of its first row for the
// token-major output.
struct PagedPrefillTile {
  ulong offset;
  ulong first_slot;
  uint first_position;
  uint active_rows;
  uint first_row;
  uint splits;
  bool active;
};

template <uint Group, uint HeadDim>
inline PagedPrefillTile paged_prefill_tile(constant PagedPrefillParams &params,
                                           uint kv_head, uint tile) {
  const uint start = tile * PAGED_PREFILL_TILE_ROWS;
  if (!paged_prefill_contract_valid(params) || kv_head >= params.kv_heads ||
      start >= params.rows)
    return {};
  return {(ulong(kv_head) * params.chunk_stride + start) * Group * HeadDim,
          (ulong(tile) * params.kv_heads + kv_head) * params.split_count,
          params.committed_tokens + start,
          min(PAGED_PREFILL_TILE_ROWS, params.rows - start),
          start,
          params.split_count,
          true};
}

// Splash splash_attention_tile_part: part `part` of a tile run as Parts parts
// of 8 / Parts rows, each a tile of its own rows.
template <uint Group, uint Parts, uint HeadDim>
inline PagedPrefillTile paged_prefill_tile_part(PagedPrefillTile tile, uint part,
                                                uint slot_stride) {
  static_assert(Parts > 0 && PAGED_PREFILL_TILE_ROWS % Parts == 0, "parts of whole rows");
  constexpr uint Rows = PAGED_PREFILL_TILE_ROWS / Parts;
  const uint first = part * Rows;
  if (!tile.active || first >= tile.active_rows)
    return {};
  return {tile.offset + ulong(first) * Group * HeadDim,
          tile.first_slot * Parts + ulong(part) * slot_stride,
          tile.first_position + first,
          min(Rows, tile.active_rows - first),
          tile.first_row + first,
          tile.splits,
          true};
}

inline uint paged_prefill_pages(uint visible_tokens) {
  return (visible_tokens + PAGED_PREFILL_PAGE_TOKENS - 1) / PAGED_PREFILL_PAGE_TOKENS;
}

inline uint paged_prefill_pages_per_split(uint pages, uint splits) {
  return (pages + splits - 1) / splits;
}

// Splash splash_attention_page_softmax over one 32-token page. An INT8
// lane's eight tokens lie in one pool block; a block past the tile's visible
// tokens is never looked up in the table, its lanes read the page's first
// block's scales and the select masks them.
template <uint Group, uint Rows, uint HeadDim, bool Quantized>
inline void paged_prefill_page_softmax(
    threadgroup const float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const int *table, device const float *key_scales,
    device const float *value_scales, constant PagedPrefillParams &params,
    uint kv_head, uint token_start, uint visible_tokens, uint visible_blocks,
    uint committed_tokens, uint active_rows, uint thread_index) {
  constexpr uint N = PAGED_PREFILL_PAGE_TOKENS;
  constexpr uint FusedRows = Rows * Group;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = N / TokensPerLane;
  constexpr float Scale = paged_prefill_scale<HeadDim>();
  static_assert(LanesPerRow * FusedRows <= PAGED_PREFILL_THREADS,
                "one softmax lane per thread of the 256-thread tile");
  if (thread_index >= LanesPerRow * FusedRows)
    return;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  const uint query_row = fused_row / Group;
  const uint causal_end = committed_tokens + min(query_row, active_rows - 1) + 1;
  const uint limit = min(visible_tokens, causal_end);
  const uint token = token_start + column;
  device const float4 *key_scales4 = nullptr;
  device const float4 *value_scales4 = nullptr;
  if constexpr (Quantized) {
    const uint block_size = params.block_size;
    uint logical = token / block_size;
    if (logical >= visible_blocks)
      logical = token_start / block_size;
    const ulong slot =
        (ulong(uint(table[logical])) * params.kv_heads + kv_head) * block_size +
        token % block_size;
    key_scales4 = reinterpret_cast<device const float4 *>(key_scales + slot);
    value_scales4 = reinterpret_cast<device const float4 *>(value_scales + slot);
  }
  threadgroup const float4 *scores4 =
      reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N + column);
  float score[TokensPerLane];
  {
    float4 low = scores4[0], high = scores4[1];
    if constexpr (Quantized) {
      low *= key_scales4[0];
      high *= key_scales4[1];
    }
    low *= Scale;
    high *= Scale;
    score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
    score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = token + j < limit ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = token + j < limit ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == previous_max
                            ? 1.0f
                            : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f)
      atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  // Masked tokens stay exactly zero whatever their stored value scale holds.
  float4 low_scales(1.0f), high_scales(1.0f);
  if constexpr (Quantized) {
    low_scales = value_scales4[0];
    high_scales = value_scales4[1];
  }
  const float4 low(token + 0 < limit ? probability[0] * low_scales.x : 0.0f,
                   token + 1 < limit ? probability[1] * low_scales.y : 0.0f,
                   token + 2 < limit ? probability[2] * low_scales.z : 0.0f,
                   token + 3 < limit ? probability[3] * low_scales.w : 0.0f);
  const float4 high(token + 4 < limit ? probability[4] * high_scales.x : 0.0f,
                    token + 5 < limit ? probability[5] * high_scales.y : 0.0f,
                    token + 6 < limit ? probability[6] * high_scales.z : 0.0f,
                    token + 7 < limit ? probability[7] * high_scales.w : 0.0f);
  threadgroup bfloat4 *probabilities4 =
      reinterpret_cast<threadgroup bfloat4 *>(probabilities + fused_row * N + column);
  probabilities4[0] = bfloat4(low);
  probabilities4[1] = bfloat4(high);
}

// Splash splash_paged_attention_tile over the pool: one tile (or part) over
// its split's pages. Only the blocks holding the tile's visible tokens are
// looked up and read; the K stage of a page's unread block is zero and its
// half of PV is skipped (its probabilities are zero).
template <uint Group, uint Rows, uint HeadDim, typename CacheElement>
inline void paged_prefill_page_loop(
    device bfloat *tile_queries, device const int *table,
    device CacheElement *key_pool, device CacheElement *value_pool,
    device const float *key_scales, device const float *value_scales,
    constant PagedPrefillParams &params, uint kv_head, uint committed_tokens,
    uint active_rows, uint splits, uint split, device float *partials,
    device float *statistics, ulong slot, threadgroup float *scores,
    threadgroup bfloat *probabilities, threadgroup CacheElement *keys,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint thread_index) {
  constexpr ushort M = Rows * Group;
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr ushort N = PAGED_PREFILL_PAGE_TOKENS;
  constexpr ushort Half = PAGED_PREFILL_HALF_TOKENS;
  constexpr ushort D = HeadDim;
  constexpr uint X = 16 / sizeof(CacheElement);
  constexpr uint Chunks = D / X;
  const uint block_size = params.block_size;
  const uint kv_heads = params.kv_heads;
  const uint visible_tokens = committed_tokens + active_rows;
  const uint visible_blocks = (visible_tokens + block_size - 1) / block_size;
  const uint pages = paged_prefill_pages(visible_tokens);
  const uint per_split = paged_prefill_pages_per_split(pages, splits);
  const uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  const uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto kt = tensor(keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto value_type = tensor(static_cast<device CacheElement *>(nullptr),
                           dextents<int, 2>{Half, D}, array<int, 2>{1, int(block_size)});
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = kt.template slice<D, N>(0, 0);
  auto p0 = pt.template slice<Half, M>(0, 0);
  auto v0 = value_type.template slice<Half, D>(0, 0);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, Half, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  const bool running_full =
      uint(running.get_capacity()) * PAGED_PREFILL_THREADS == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  const uint blocks_per_page = N / block_size;
  const uint vectors = Chunks * block_size;
  threadgroup uint4 *staged = reinterpret_cast<threadgroup uint4 *>(keys);
  for (uint page = page_begin; page < page_end; ++page) {
    const uint token_start = page * N;
    // Stage the page's keys token-major: a block's (chunk, token) vectors are
    // contiguous in the pool, vector v = chunk * block_size + token.
    for (uint j = 0; j < blocks_per_page; ++j) {
      const uint logical = page * blocks_per_page + j;
      if (logical < visible_blocks) {
        device const uint4 *source = reinterpret_cast<device const uint4 *>(
            key_pool + (ulong(uint(table[logical])) * kv_heads + kv_head) * D * block_size);
        for (uint v = thread_index; v < vectors; v += PAGED_PREFILL_THREADS)
          staged[(j * block_size + v % block_size) * Chunks + v / block_size] = source[v];
      } else {
        for (uint v = thread_index; v < vectors; v += PAGED_PREFILL_THREADS)
          staged[(j * block_size + v % block_size) * Chunks + v / block_size] = uint4(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), float>();
    qk.run(q0, k0, page_scores);
    page_scores.store(st.slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    paged_prefill_page_softmax<Group, Rows, HeadDim, Quantized>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale, table,
        key_scales, value_scales, params, kv_head, token_start, visible_tokens,
        visible_blocks, committed_tokens, active_rows, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    for (uint step = 0; step < N / Half; ++step) {
      const uint first = token_start + step * Half;
      const uint logical = first / block_size;
      if (logical >= visible_blocks)
        continue;
      device CacheElement *values =
          value_pool + (ulong(uint(table[logical])) * kv_heads + kv_head) * D * block_size +
          first % block_size;
      auto vt = tensor(values, dextents<int, 2>{Half, D}, array<int, 2>{1, int(block_size)});
      auto vs = vt.template slice<Half, D>(0, 0);
      auto ps = pt.template slice<Half, M>(step * Half, 0);
      pv.run(ps, vs, running);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

// Splash splash_prefill_attention_split_phase: KV heads (and each one's
// tile parts) vary first, then query tiles, then history splits.
template <uint Group, uint HeadDim, typename CacheElement>
inline void paged_prefill_split_phase(
    device bfloat *queries, device float *partials, device float *statistics,
    device const int *table, device CacheElement *key_pool,
    device CacheElement *value_pool, device const float *key_scales,
    device const float *value_scales, constant PagedPrefillParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup CacheElement *keys, threadgroup float *row_max,
    threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, uint3 group, uint thread_index) {
  constexpr uint Parts = paged_prefill_tile_parts(Group);
  const uint split = group.z;
  const uint kv_head = group.x / Parts;
  const PagedPrefillTile tile = paged_prefill_tile_part<Group, Parts, HeadDim>(
      paged_prefill_tile<Group, HeadDim>(params, kv_head, group.y), group.x % Parts,
      params.split_count);
  if (!tile.active || split >= tile.splits)
    return;
  paged_prefill_page_loop<Group, PAGED_PREFILL_TILE_ROWS / Parts, HeadDim, CacheElement>(
      queries + tile.offset, table, key_pool, value_pool, key_scales, value_scales,
      params, kv_head, tile.first_position, tile.active_rows, tile.splits, split,
      partials, statistics, tile.first_slot + split, scores, probabilities, keys,
      row_max, row_sum, previous_scale, rescale, thread_index);
}

// Splash splash_attention_combine_splits: one fused row's first `written`
// splits in split order, a threadgroup of HeadDim threads.
template <uint Group, uint Rows, uint HeadDim>
inline void paged_prefill_combine_splits(
    device const float *partials, device const float *statistics,
    device bfloat *row_output, uint active_rows, uint written, ulong head_slot,
    uint fused_row, uint thread_index, threadgroup float *weights,
    threadgroup float *group_values) {
  constexpr uint M = Rows * Group;
  constexpr uint D = HeadDim;
  static_assert(D >= PAGED_PREFILL_MAX_SPLITS, "a thread per split");
  if (fused_row / Group >= active_rows) {
    row_output[thread_index] = bfloat(0.0f);
    return;
  }
  const uint lane = thread_index % 32, sg = thread_index / 32;
  const ulong stat = ((head_slot + thread_index) * M + fused_row) * 2;
  const float maximum = thread_index < written ? statistics[stat] : -INFINITY;
  const float group_maximum = simd_max(maximum);
  if (lane == 0)
    group_values[sg] = group_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float row_maximum = simd_max(lane < D / 32 ? group_values[lane] : -INFINITY);
  if (thread_index < written)
    weights[thread_index] = fast::exp(maximum - row_maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  // Keep both accumulations in the original split order. A parallel sum of
  // the denominator changes speculative acceptance (Splash).
  for (uint split = 0; split < written; ++split) {
    const float weight = weights[split];
    const ulong slot = (head_slot + split) * M + fused_row;
    numerator += weight * partials[slot * D + thread_index];
    denominator += weight * statistics[slot * 2 + 1];
  }
  row_output[thread_index] = bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
}

// Splash splash_prefill_attention_reduce_phase + splash_attention_reduce_row:
// a threadgroup of HeadDim threads per fused row of a whole tile; the row
// finds its part and recomputes that part's split partition.
template <uint Group, uint HeadDim>
inline void paged_prefill_reduce_phase(device const float *partials,
                                       device const float *statistics,
                                       device bfloat *output,
                                       constant PagedPrefillParams &params,
                                       uint3 group, uint thread_index,
                                       threadgroup float *weights,
                                       threadgroup float *group_values) {
  constexpr uint M = PAGED_PREFILL_TILE_ROWS * Group;
  constexpr uint Parts = paged_prefill_tile_parts(Group);
  constexpr uint PartRows = PAGED_PREFILL_TILE_ROWS / Parts;
  constexpr uint PartM = PartRows * Group;
  const uint fused_row = group.y;
  const PagedPrefillTile whole = paged_prefill_tile<Group, HeadDim>(params, group.x, group.z);
  if (!whole.active || fused_row >= M || thread_index >= HeadDim)
    return;
  const uint row = whole.first_row + fused_row / Group;
  if (row >= params.rows)
    return;
  const uint query_heads = params.kv_heads * Group;
  device bfloat *row_output =
      output + (ulong(row) * query_heads + group.x * Group + fused_row % Group) * HeadDim;
  const PagedPrefillTile tile = paged_prefill_tile_part<Group, Parts, HeadDim>(
      whole, fused_row / PartM, params.split_count);
  if (!tile.active) {
    row_output[thread_index] = bfloat(0.0f);
    return;
  }
  const uint pages = paged_prefill_pages(tile.first_position + tile.active_rows);
  const uint per_split = paged_prefill_pages_per_split(pages, tile.splits);
  const uint written = (pages + per_split - 1) / per_split;
  paged_prefill_combine_splits<Group, PartRows, HeadDim>(
      partials, statistics, row_output, tile.active_rows, written, tile.first_slot,
      fused_row % PartM, thread_index, weights, group_values);
}

#define PAGED_PREFILL_SPLIT(Name, Group, HeadDim, CacheElement)                 \
  [[kernel, max_total_threads_per_threadgroup(256)]] void Name(                \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const int *table [[buffer(3)]],                                   \
      device CacheElement *key_pool [[buffer(4)]],                             \
      device CacheElement *value_pool [[buffer(5)]],                           \
      device const float *key_scales [[buffer(6)]],                            \
      device const float *value_scales [[buffer(7)]],                          \
      constant PagedPrefillParams &params [[buffer(8)]],                       \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * PAGED_PREFILL_TILE_ROWS /                       \
                       paged_prefill_tile_parts(Group);                        \
    constexpr uint N = PAGED_PREFILL_PAGE_TOKENS;                              \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    alignas(16) threadgroup CacheElement keys[N * HeadDim];                    \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    paged_prefill_split_phase<Group, HeadDim, CacheElement>(                   \
        queries, partials, statistics, table, key_pool, value_pool,            \
        key_scales, value_scales, params, scores, probabilities, keys,         \
        row_max, row_sum, previous_scale, &rescale, group, thread_index);      \
  }

#define PAGED_PREFILL_REDUCE(Name, Group, HeadDim)                              \
  [[kernel, max_total_threads_per_threadgroup(HeadDim)]] void Name(            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      constant PagedPrefillParams &params [[buffer(3)]],                       \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[PAGED_PREFILL_MAX_SPLITS];                       \
    threadgroup float group_values[HeadDim / 32];                              \
    paged_prefill_reduce_phase<Group, HeadDim>(partials, statistics, output,   \
                                               params, group, thread_index,    \
                                               weights, group_values);         \
  }

#define PAGED_PREFILL_INSTANCES(Group, HeadDim)                                 \
  PAGED_PREFILL_SPLIT(paged_prefill_attention_q8_split_g##Group##_d##HeadDim,   \
                      Group, HeadDim, int8_t)                                  \
  PAGED_PREFILL_SPLIT(paged_prefill_attention_bf16_split_g##Group##_d##HeadDim, \
                      Group, HeadDim, bfloat)                                  \
  PAGED_PREFILL_REDUCE(paged_prefill_attention_reduce_g##Group##_d##HeadDim,    \
                       Group, HeadDim)

PAGED_PREFILL_INSTANCES(4, 128)
PAGED_PREFILL_INSTANCES(4, 256)
PAGED_PREFILL_INSTANCES(6, 128)
PAGED_PREFILL_INSTANCES(6, 256)
PAGED_PREFILL_INSTANCES(8, 128)
PAGED_PREFILL_INSTANCES(8, 256)
PAGED_PREFILL_INSTANCES(16, 128)
PAGED_PREFILL_INSTANCES(16, 256)
