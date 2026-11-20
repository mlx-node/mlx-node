// The in-place prefill / verify tile over the keys-token-major page (Splash
// 1.6.1 runtime/metal/kernels/common/paged_attention_tile.h
// splash_paged_attention_tile, kv_extent.h SplashKvAddressing; Apache-2.0):
// a page is one 32-token pool block, K [blocks][KV][32][D] and V
// [blocks][KV][D][32] are device operands read in place, the INT8 scales
// [blocks][KV][32] are one row per page. QK and PV run once per page; the
// tail page's rows past the tile's visible tokens are masked by the softmax.

template <uint Group, uint Rows, uint HeadDim, typename Score>
inline void paged_prefill_ktm_page_softmax(
    threadgroup const Score *scores, float score_scale,
    threadgroup bfloat *probabilities, threadgroup float *row_max,
    threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, device const float4 *key_scales4,
    device const float4 *value_scales4, uint token_start, uint visible_tokens,
    uint committed_tokens, uint active_rows, uint thread_index) {
  constexpr uint N = PAGED_PREFILL_PAGE_TOKENS;
  constexpr uint FusedRows = Rows * Group;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = PAGED_PREFILL_LANES_PER_ROW;
  static_assert(N == LanesPerRow * TokensPerLane, "four lanes of eight tokens span one page");
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
  const uint vector = column / 4;
  threadgroup const vec<Score, 4> *scores4 =
      reinterpret_cast<threadgroup const vec<Score, 4> *>(scores + fused_row * N + column);
  float score[TokensPerLane];
  {
    float4 low = float4(scores4[0]), high = float4(scores4[1]);
    low *= key_scales4[vector];
    high *= key_scales4[vector + 1];
    low *= score_scale;
    high *= score_scale;
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
  // Masked tokens stay exactly zero whatever their stored scale and value hold.
  const float4 low_scales = value_scales4[vector];
  const float4 high_scales = value_scales4[vector + 1];
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

template <uint Group, uint Rows, uint HeadDim, bool Int8Queries>
inline void paged_prefill_ktm_page_loop(
    device bfloat *tile_queries, device const int *table, device int8_t *key_pool,
    device int8_t *value_pool, device const float *key_scales,
    device const float *value_scales, constant PagedPrefillParams &params,
    uint kv_head, uint committed_tokens, uint active_rows, uint splits, uint split,
    device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint thread_index) {
  constexpr ushort M = Rows * Group;
  constexpr ushort N = PAGED_PREFILL_PAGE_TOKENS;
  constexpr ushort D = HeadDim;
  // A page that is not one pool block would read another page's rows.
  if (params.block_size != N)
    return;
  const uint kv_heads = params.kv_heads;
  const uint visible_tokens = committed_tokens + active_rows;
  const uint pages = paged_prefill_pages(visible_tokens);
  const uint per_split = paged_prefill_pages_per_split(pages, splits);
  const uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  const uint page_end = min(pages, page_begin + per_split);
  using QueryElement = conditional_t<Int8Queries, int8_t, bfloat>;
  using Score = conditional_t<Int8Queries, int, float>;
  device QueryElement *queries;
  float score_scale = paged_prefill_scale<HeadDim>();
  if constexpr (Int8Queries) {
    queries = reinterpret_cast<device int8_t *>(partials + slot * M * D);
    score_scale = paged_int8_score_scale<HeadDim>(
        paged_prefill_query_codes<M, HeadDim>(tile_queries, queries, thread_index));
  } else {
    queries = tile_queries;
  }
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  // The query codes are device writes the whole threadgroup reads.
  threadgroup_barrier(Int8Queries ? mem_flags::mem_threadgroup | mem_flags::mem_device
                                  : mem_flags::mem_threadgroup);

  threadgroup Score *page_score_tile = reinterpret_cast<threadgroup Score *>(scores);
  auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(page_score_tile, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto key_type = tensor(static_cast<device int8_t *>(nullptr), dextents<int, 2>{D, N},
                         array<int, 2>{1, D});
  auto value_type = tensor(static_cast<device int8_t *>(nullptr), dextents<int, 2>{N, D},
                           array<int, 2>{1, int(N)});
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = key_type.template slice<D, N>(0, 0);
  auto p0 = pt.template slice<N, M>(0, 0);
  auto v0 = value_type.template slice<N, D>(0, 0);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
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

  for (uint page = page_begin; page < page_end; ++page) {
    const uint token_start = page * N;
    const ulong first = (ulong(uint(table[page])) * kv_heads + kv_head) * N;
    auto kt = tensor(key_pool + first * D, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto vt = tensor(value_pool + first * D, dextents<int, 2>{N, D}, array<int, 2>{1, int(N)});
    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), Score>();
    auto ks = kt.template slice<D, N>(0, 0);
    qk.run(q0, ks, page_scores);
    page_scores.store(st.template slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    paged_prefill_ktm_page_softmax<Group, Rows, HeadDim, Score>(
        page_score_tile, score_scale, probabilities, row_max, row_sum, previous_scale,
        rescale, reinterpret_cast<device const float4 *>(key_scales + first),
        reinterpret_cast<device const float4 *>(value_scales + first), token_start,
        visible_tokens, committed_tokens, active_rows, thread_index);
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
    auto vs = vt.template slice<N, D>(0, 0);
    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

#define PAGED_PREFILL_KTM_SPLIT(Name, Group, HeadDim, Int8Queries)               \
  [[kernel, max_total_threads_per_threadgroup(256)]] void Name(                \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const int *table [[buffer(3)]],                                   \
      device int8_t *key_pool [[buffer(4)]],                                   \
      device int8_t *value_pool [[buffer(5)]],                                 \
      device const float *key_scales [[buffer(6)]],                            \
      device const float *value_scales [[buffer(7)]],                          \
      constant PagedPrefillParams &params [[buffer(8)]],                       \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Parts = paged_prefill_tile_parts(Group);                    \
    constexpr uint M = Group * PAGED_PREFILL_TILE_ROWS / Parts;                \
    constexpr uint N = PAGED_PREFILL_PAGE_TOKENS;                              \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    const uint split = group.z;                                                \
    const uint kv_head = group.x / Parts;                                      \
    const PagedPrefillTile tile = paged_prefill_tile_part<Group, Parts, HeadDim>( \
        paged_prefill_tile<Group, HeadDim>(params, kv_head, group.y),          \
        group.x % Parts, params.split_count);                                  \
    if (!tile.active || split >= tile.splits)                                  \
      return;                                                                  \
    paged_prefill_ktm_page_loop<Group, PAGED_PREFILL_TILE_ROWS / Parts, HeadDim, \
                                Int8Queries>(                                  \
        queries + tile.offset, table, key_pool, value_pool, key_scales,        \
        value_scales, params, kv_head, tile.first_position, tile.active_rows,  \
        tile.splits, split, partials, statistics, tile.first_slot + split,     \
        scores, probabilities, row_max, row_sum, previous_scale, &rescale,     \
        thread_index);                                                         \
  }

PAGED_PREFILL_KTM_SPLIT(paged_prefill_attention_q8_split_g6_d256_ktm, 6, 256, false)
PAGED_PREFILL_KTM_SPLIT(paged_prefill_attention_q8q_split_g6_d256_ktm, 6, 256, true)
