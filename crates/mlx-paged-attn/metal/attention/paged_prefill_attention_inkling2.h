// Inkling-2-Mini in-place MPP paged prefill / verify kernels (G=3, D=128):
// - INT8 32-token keys-token-major (`q8` BF16 queries + `q8q` INT8 queries on M5+)
// - BF16 vLLM pool (`block_size = 16` or `32`)
// - Sliding-window page range skip (splits only active pages `[first_page, last_page)`)
// - Relative bias (`R @ W_rel`) for tiles overlapping `[q_pos - rel_extent + 1, q_pos]`,
//   and zero-bias fast path for all distant tiles (`token_start + 32 <= q_tile_min + 1 - rel_extent`)
// - Emits both BF16 `output [rows, num_q_heads, 128]` and natural-log F32 `lse [rows, num_q_heads]`
//
// Kernel prefix is `inkling2_paged_prefill_` (separate from `paged_prefill_attention_`).

struct Inkling2PagedPrefillParams {
  uint committed_tokens;
  uint rows;
  uint chunk_stride;
  uint table_entries;
  uint split_count;
  uint max_splits;
  uint block_size;
  uint kv_heads;
  uint sliding_window;
  uint rel_extent;
  uint rel_bias_is_f32;
  uint rel_mode; // 0 = none, 1 = precomputed [rows,H,rel_extent], 2 = on-the-fly r [rows,H,16] x w [rel_extent,16]
  uint has_gate_sink;
  uint gate_row_stride;
  uint gate_is_f32;
  uint sink_row_stride;
  uint sink_is_f32;
  uint write_lse;
};

inline bool inkling2_prefill_contract_valid(
    constant Inkling2PagedPrefillParams &params) {
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

template <uint Group, uint HeadDim>
inline PagedPrefillTile inkling2_prefill_tile(
    constant Inkling2PagedPrefillParams &params, uint kv_head, uint tile) {
  const uint start = tile * PAGED_PREFILL_TILE_ROWS;
  if (!inkling2_prefill_contract_valid(params) || kv_head >= params.kv_heads ||
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

#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
inline float inkling2_prefill_rel_dot16(
    float4 r0, float4 r1, float4 r2, float4 r3, const device float4 *w4) {
  const float d0 = r0.x * w4[0].x + r0.y * w4[0].y + r0.z * w4[0].z + r0.w * w4[0].w;
  const float d1 = r1.x * w4[1].x + r1.y * w4[1].y + r1.z * w4[1].z + r1.w * w4[1].w;
  const float d2 = r2.x * w4[2].x + r2.y * w4[2].y + r2.z * w4[2].z + r2.w * w4[2].w;
  const float d3 = r3.x * w4[3].x + r3.y * w4[3].y + r3.z * w4[3].z + r3.w * w4[3].w;
  return (d0 + d1) + (d2 + d3);
}
#pragma clang fp reassociate(on)
#pragma clang fp contract(fast)

inline float inkling2_prefill_read_bias(device const uchar *rel_bias_raw,
                                        ulong idx, uint is_f32) {
  if (is_f32 != 0u)
    return reinterpret_cast<device const float *>(rel_bias_raw)[idx];
  return float(reinterpret_cast<device const bfloat *>(rel_bias_raw)[idx]);
}

// Compute active 32-token page range `[first_page, last_page_excl)` for a tile.
// For SWA (`sliding_window > 0`), skips all evicted pages before `q_tile_min + 1 - sliding_window`.
inline void inkling2_active_page_range(uint first_position, uint active_rows,
                                       uint sliding_window,
                                       thread uint &first_page,
                                       thread uint &last_page_excl) {
  const uint visible_tokens = first_position + active_rows;
  const uint tile_min_k =
      (sliding_window > 0u && first_position + 1u > sliding_window)
          ? (first_position + 1u - sliding_window)
          : 0u;
  first_page = tile_min_k / PAGED_PREFILL_PAGE_TOKENS;
  last_page_excl = paged_prefill_pages(visible_tokens);
}

// Unified page softmax for Inkling2 KTM INT8 and BF16 vLLM pages.
// `HasBias == false` is the zero-bias fast path (distant tiles on global layers).
template <uint Group, uint Rows, uint HeadDim, typename Score, bool HasBias,
          bool IsInt8Ktm>
inline void inkling2_prefill_page_softmax(
    threadgroup const Score *scores, float score_scale,
    threadgroup bfloat *probabilities, threadgroup float *row_max,
    threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, device const float4 *key_scales4,
    device const float4 *value_scales4, device const uchar *rel_bias_raw,
    device const float *rel_bias_w, uint rel_extent, uint rel_bias_is_f32,
    uint rel_mode, uint sliding_window, uint kv_head, uint num_q_heads,
    uint tile_first_row, uint token_start, uint visible_tokens,
    uint committed_tokens, uint active_rows, uint thread_index) {
  constexpr uint N = PAGED_PREFILL_PAGE_TOKENS;
  constexpr uint FusedRows = Rows * Group;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = PAGED_PREFILL_LANES_PER_ROW;
  if (thread_index >= LanesPerRow * FusedRows)
    return;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  const uint query_row = min(fused_row / Group, active_rows - 1u);
  const uint q_pos = committed_tokens + query_row;
  const uint limit = min(visible_tokens, q_pos + 1u);
  const uint row_lower =
      (sliding_window > 0u && q_pos + 1u > sliding_window)
          ? (q_pos + 1u - sliding_window)
          : 0u;
  const uint token = token_start + column;
  const uint vector = column / 4u;
  threadgroup const vec<Score, 4> *scores4 =
      reinterpret_cast<threadgroup const vec<Score, 4> *>(
          scores + fused_row * N + column);
  float score[TokensPerLane];
  {
    float4 low = float4(scores4[0]), high = float4(scores4[1]);
    if constexpr (IsInt8Ktm) {
      low *= key_scales4[vector];
      high *= key_scales4[vector + 1u];
    }
    low *= score_scale;
    high *= score_scale;
    score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
    score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
  }

  ulong bias_row_base = 0;
  float4 r0(0.0f), r1(0.0f), r2(0.0f), r3(0.0f);
  if constexpr (HasBias) {
    const uint global_q_row = tile_first_row + query_row;
    const uint global_q_head = kv_head * Group + (fused_row % Group);
    if (rel_mode == 2u) {
      const device float4 *r4 =
          reinterpret_cast<const device float4 *>(rel_bias_raw) +
          (ulong(global_q_row) * num_q_heads + global_q_head) * 4ul;
      r0 = r4[0];
      r1 = r4[1];
      r2 = r4[2];
      r3 = r4[3];
    } else {
      bias_row_base =
          (ulong(global_q_row) * num_q_heads + global_q_head) * rel_extent;
    }
  }

  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    const uint k_pos = token + j;
    const bool valid = k_pos >= row_lower && k_pos < limit;
    if (valid) {
      if constexpr (HasBias) {
        const uint dist = q_pos - k_pos;
        if (dist < rel_extent) {
          if (rel_mode == 2u) {
            const device float4 *w4 =
                reinterpret_cast<const device float4 *>(rel_bias_w) +
                ulong(dist) * 4ul;
            score[j] += inkling2_prefill_rel_dot16(r0, r1, r2, r3, w4);
          } else {
            score[j] += inkling2_prefill_read_bias(
                rel_bias_raw, bias_row_base + dist, rel_bias_is_f32);
          }
        }
      }
      local_max = max(local_max, score[j]);
    } else {
      score[j] = -INFINITY;
    }
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  const bool finite_max = next_max > -INFINITY;
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    const uint k_pos = token + j;
    const bool valid = k_pos >= row_lower && k_pos < limit;
    probability[j] =
        (valid && finite_max) ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0u) {
    const float scale = (!finite_max || previous_max == -INFINITY ||
                         next_max == previous_max)
                            ? 1.0f
                            : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f)
      atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  float4 low, high;
  if constexpr (IsInt8Ktm) {
    const float4 low_scales = value_scales4[vector];
    const float4 high_scales = value_scales4[vector + 1u];
    low = float4(
        (token + 0u >= row_lower && token + 0u < limit) ? probability[0] * low_scales.x : 0.0f,
        (token + 1u >= row_lower && token + 1u < limit) ? probability[1] * low_scales.y : 0.0f,
        (token + 2u >= row_lower && token + 2u < limit) ? probability[2] * low_scales.z : 0.0f,
        (token + 3u >= row_lower && token + 3u < limit) ? probability[3] * low_scales.w : 0.0f);
    high = float4(
        (token + 4u >= row_lower && token + 4u < limit) ? probability[4] * high_scales.x : 0.0f,
        (token + 5u >= row_lower && token + 5u < limit) ? probability[5] * high_scales.y : 0.0f,
        (token + 6u >= row_lower && token + 6u < limit) ? probability[6] * high_scales.z : 0.0f,
        (token + 7u >= row_lower && token + 7u < limit) ? probability[7] * high_scales.w : 0.0f);
  } else {
    low = float4(probability[0], probability[1], probability[2], probability[3]);
    high = float4(probability[4], probability[5], probability[6], probability[7]);
  }
  threadgroup bfloat4 *probabilities4 =
      reinterpret_cast<threadgroup bfloat4 *>(probabilities + fused_row * N + column);
  probabilities4[0] = bfloat4(low);
  probabilities4[1] = bfloat4(high);
}

// Unified page loop for Inkling2 KTM INT8 and BF16 vLLM pools (G=3, D=128, M=24).
template <uint Group, uint Rows, uint HeadDim, bool IsInt8Ktm, bool Int8Queries>
inline void inkling2_prefill_page_loop(
    device bfloat *tile_queries, device const int *table,
    device conditional_t<IsInt8Ktm, int8_t, bfloat> *key_pool,
    device conditional_t<IsInt8Ktm, int8_t, bfloat> *value_pool,
    device const float *key_scales, device const float *value_scales,
    device const uchar *rel_bias_raw, device const float *rel_bias_w,
    constant Inkling2PagedPrefillParams &params, uint kv_head,
    uint committed_tokens, uint tile_first_row, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup conditional_t<IsInt8Ktm, int8_t, bfloat> *keys,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint thread_index) {
  using CacheElement = conditional_t<IsInt8Ktm, int8_t, bfloat>;
  static_assert(IsInt8Ktm || !Int8Queries, "INT8 queries require KTM INT8 pages");
  constexpr ushort M = Rows * Group;
  constexpr ushort N = PAGED_PREFILL_PAGE_TOKENS;
  constexpr ushort Half = PAGED_PREFILL_HALF_TOKENS;
  constexpr ushort PvK = IsInt8Ktm ? N : Half;
  constexpr ushort D = HeadDim;
  constexpr uint X = 16u / sizeof(CacheElement);
  constexpr uint Chunks = D / X;
  if constexpr (IsInt8Ktm) {
    if (params.block_size != N)
      return;
  }
  const uint block_size = params.block_size;
  const uint kv_heads = params.kv_heads;
  const uint num_q_heads = kv_heads * Group;
  const uint visible_tokens = committed_tokens + active_rows;
  const uint visible_blocks =
      IsInt8Ktm ? 0u : (visible_tokens + block_size - 1u) / block_size;
  uint first_page = 0u, last_page_excl = 0u;
  inkling2_active_page_range(committed_tokens, active_rows,
                             params.sliding_window, first_page, last_page_excl);
  const uint active_pages =
      last_page_excl > first_page ? last_page_excl - first_page : 0u;
  if (active_pages == 0u)
    return;
  const uint per_split = paged_prefill_pages_per_split(active_pages, splits);
  const uint page_begin = first_page + split * per_split;
  if (page_begin >= last_page_excl)
    return;
  const uint page_end = min(last_page_excl, page_begin + per_split);

  // Earliest key position that can ever receive a non-zero rel_bias in this tile.
  const uint bias_min_k =
      (params.rel_extent > 0u && committed_tokens + 1u > params.rel_extent)
          ? (committed_tokens + 1u - params.rel_extent)
          : 0u;

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
  threadgroup_barrier(Int8Queries ? mem_flags::mem_threadgroup | mem_flags::mem_device
                                  : mem_flags::mem_threadgroup);

  threadgroup Score *page_score_tile = reinterpret_cast<threadgroup Score *>(scores);
  auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(page_score_tile, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto value_type = tensor(
      static_cast<device CacheElement *>(nullptr), dextents<int, 2>{PvK, D},
      array<int, 2>{1, IsInt8Ktm ? int(N) : int(block_size)});
  auto q0 = qt.template slice<D, M>(0, 0);
  auto p0 = pt.template slice<PvK, M>(0, 0);
  auto v0 = value_type.template slice<PvK, D>(0, 0);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, PvK, false, true, true,
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
    ulong ktm_first = 0;
    if constexpr (IsInt8Ktm) {
      ktm_first = (ulong(uint(table[page])) * kv_heads + kv_head) * N;
      auto kt = tensor(key_pool + ktm_first * D, dextents<int, 2>{D, N},
                       array<int, 2>{1, D});
      auto ks = kt.template slice<D, N>(0, 0);
      auto page_scores = qk.template get_destination_cooperative_tensor<
          decltype(q0), decltype(ks), Score>();
      qk.run(q0, ks, page_scores);
      page_scores.store(st.template slice<N, M>(0, 0));
    } else {
      const uint blocks_per_page = N / block_size;
      const uint vectors = Chunks * block_size;
      threadgroup uint4 *staged = reinterpret_cast<threadgroup uint4 *>(keys);
      for (uint j = 0; j < blocks_per_page; ++j) {
        const uint logical = page * blocks_per_page + j;
        if (logical < visible_blocks) {
          device const uint4 *source = reinterpret_cast<device const uint4 *>(
              key_pool +
              (ulong(uint(table[logical])) * kv_heads + kv_head) * D * block_size);
          for (uint v = thread_index; v < vectors; v += PAGED_PREFILL_THREADS)
            staged[(j * block_size + v % block_size) * Chunks + v / block_size] =
                source[v];
        } else {
          for (uint v = thread_index; v < vectors; v += PAGED_PREFILL_THREADS)
            staged[(j * block_size + v % block_size) * Chunks + v / block_size] =
                uint4(0);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto kt = tensor(keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
      auto k0 = kt.template slice<D, N>(0, 0);
      auto page_scores = qk.template get_destination_cooperative_tensor<
          decltype(q0), decltype(k0), Score>();
      qk.run(q0, k0, page_scores);
      page_scores.store(st.template slice<N, M>(0, 0));
    }
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Zero-bias fast path for all distant pages where `token_start + 32 <= bias_min_k`.
    const bool page_has_bias =
        (params.rel_extent > 0u) && (params.rel_mode != 0u) &&
        (token_start + N > bias_min_k);
    device const float4 *ks4 =
        IsInt8Ktm ? reinterpret_cast<device const float4 *>(key_scales + ktm_first)
                  : nullptr;
    device const float4 *vs4 =
        IsInt8Ktm ? reinterpret_cast<device const float4 *>(value_scales + ktm_first)
                  : nullptr;
    if (page_has_bias) {
      inkling2_prefill_page_softmax<Group, Rows, HeadDim, Score, true,
                                    IsInt8Ktm>(
          page_score_tile, score_scale, probabilities, row_max, row_sum,
          previous_scale, rescale, ks4, vs4, rel_bias_raw, rel_bias_w,
          params.rel_extent, params.rel_bias_is_f32, params.rel_mode,
          params.sliding_window, kv_head, num_q_heads, tile_first_row,
          token_start, visible_tokens, committed_tokens, active_rows,
          thread_index);
    } else {
      inkling2_prefill_page_softmax<Group, Rows, HeadDim, Score, false,
                                    IsInt8Ktm>(
          page_score_tile, score_scale, probabilities, row_max, row_sum,
          previous_scale, rescale, ks4, vs4, rel_bias_raw, rel_bias_w,
          params.rel_extent, params.rel_bias_is_f32, params.rel_mode,
          params.sliding_window, kv_head, num_q_heads, tile_first_row,
          token_start, visible_tokens, committed_tokens, active_rows,
          thread_index);
    }
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
    if constexpr (IsInt8Ktm) {
      auto vt = tensor(value_pool + ktm_first * D, dextents<int, 2>{N, D},
                       array<int, 2>{1, int(N)});
      auto vs = vt.template slice<N, D>(0, 0);
      pv.run(p0, vs, running);
    } else {
      for (uint step = 0; step < N / Half; ++step) {
        const uint first = token_start + step * Half;
        const uint logical = first / block_size;
        if (logical >= visible_blocks)
          continue;
        device CacheElement *values =
            value_pool +
            (ulong(uint(table[logical])) * kv_heads + kv_head) * D * block_size +
            first % block_size;
        auto vt = tensor(values, dextents<int, 2>{Half, D},
                         array<int, 2>{1, int(block_size)});
        auto vs = vt.template slice<Half, D>(0, 0);
        auto ps = pt.template slice<Half, M>(step * Half, 0);
        pv.run(ps, vs, running);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
inline float inkling2_prefill_precise_sigmoid(float x) {
  return precise::divide(1.0f, 1.0f + precise::exp(-x));
}

inline float inkling2_prefill_apply_gate_sink(
    float o_f, float g_val, float lse_val, float sink_val) {
  const float o_bf16 = float(bfloat(o_f));
  const float sink_scale = inkling2_prefill_precise_sigmoid(lse_val - sink_val);
  const float g_sig = inkling2_prefill_precise_sigmoid(g_val);
  return (o_bf16 * g_sig) * sink_scale;
}
#pragma clang fp reassociate(on)
#pragma clang fp contract(fast)

// Inkling2 prefill reduce: emits BF16 `output` (with optional fused gate+sink) + optional F32 `lse`.
template <uint Group, uint HeadDim>
inline void inkling2_prefill_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output, device float *lse_out,
    device const uchar *gate_raw, device const uchar *sink_raw,
    constant Inkling2PagedPrefillParams &params, uint3 group, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values) {
  constexpr uint M = PAGED_PREFILL_TILE_ROWS * Group;
  constexpr uint D = HeadDim;
  const uint fused_row = group.y;
  const PagedPrefillTile tile =
      inkling2_prefill_tile<Group, HeadDim>(params, group.x, group.z);
  if (!tile.active || fused_row >= M || thread_index >= D)
    return;
  const uint row = tile.first_row + fused_row / Group;
  if (row >= params.rows)
    return;
  const uint query_heads = params.kv_heads * Group;
  const uint q_head = group.x * Group + fused_row % Group;
  const ulong out_head_idx = ulong(row) * query_heads + q_head;
  device bfloat *row_output = output + out_head_idx * D;

  uint first_page = 0u, last_page_excl = 0u;
  inkling2_active_page_range(tile.first_position, tile.active_rows,
                             params.sliding_window, first_page, last_page_excl);
  const uint active_pages =
      last_page_excl > first_page ? last_page_excl - first_page : 0u;
  if (active_pages == 0u) {
    row_output[thread_index] = bfloat(0.0f);
    if (thread_index == 0u && params.write_lse != 0u)
      lse_out[out_head_idx] = -INFINITY;
    return;
  }
  const uint per_split = paged_prefill_pages_per_split(active_pages, tile.splits);
  const uint written = (active_pages + per_split - 1u) / per_split;

  const uint lane = thread_index % 32u, sg = thread_index / 32u;
  const ulong stat = ((tile.first_slot + thread_index) * M + fused_row) * 2u;
  const float split_sum = thread_index < written ? statistics[stat + 1u] : 0.0f;
  const float maximum =
      (thread_index < written && split_sum > 0.0f) ? statistics[stat] : -INFINITY;
  const float group_maximum = simd_max(maximum);
  if (lane == 0u)
    group_values[sg] = group_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float row_maximum =
      simd_max(lane < D / 32u ? group_values[lane] : -INFINITY);
  if (thread_index < written) {
    weights[thread_index] = (split_sum > 0.0f && row_maximum > -INFINITY)
                                ? fast::exp(maximum - row_maximum)
                                : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  for (uint split = 0u; split < written; ++split) {
    const float weight = weights[split];
    if (weight > 0.0f) {
      const ulong slot = (tile.first_slot + split) * M + fused_row;
      numerator += weight * partials[slot * D + thread_index];
      denominator += weight * statistics[slot * 2u + 1u];
    }
  }
  float o_f = denominator > 0.0f ? numerator / denominator : 0.0f;
  const float lse_val =
      (denominator > 0.0f) ? (row_maximum + precise::log(denominator)) : -INFINITY;
  if (params.has_gate_sink != 0u) {
    const ulong sink_idx = ulong(row) * ulong(params.sink_row_stride) + ulong(q_head);
    const float sink_val = inkling2_prefill_read_bias(
        sink_raw, sink_idx, params.sink_is_f32);
    const ulong gate_idx = ulong(row) * ulong(params.gate_row_stride) +
                           ulong(q_head) * D + ulong(thread_index);
    const float g_val =
        inkling2_prefill_read_bias(gate_raw, gate_idx, params.gate_is_f32);
    o_f = inkling2_prefill_apply_gate_sink(o_f, g_val, lse_val, sink_val);
  }
  row_output[thread_index] = bfloat(o_f);
  if (thread_index == 0u && params.write_lse != 0u) {
    lse_out[out_head_idx] = lse_val;
  }
}

#define INKLING2_PAGED_PREFILL_KTM_SPLIT(Name, Group, HeadDim, Int8Queries)    \
  [[kernel, max_total_threads_per_threadgroup(256)]] void Name(                \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const int *table [[buffer(3)]],                                   \
      device int8_t *key_pool [[buffer(4)]],                                   \
      device int8_t *value_pool [[buffer(5)]],                                 \
      device const float *key_scales [[buffer(6)]],                            \
      device const float *value_scales [[buffer(7)]],                          \
      device const uchar *rel_bias_raw [[buffer(8)]],                          \
      constant Inkling2PagedPrefillParams &params [[buffer(9)]],               \
      device const float *rel_bias_w [[buffer(10)]],                           \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * PAGED_PREFILL_TILE_ROWS;                        \
    constexpr uint N = PAGED_PREFILL_PAGE_TOKENS;                              \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    const uint split = group.z;                                                \
    const uint kv_head = group.x;                                              \
    const PagedPrefillTile tile =                                              \
        inkling2_prefill_tile<Group, HeadDim>(params, kv_head, group.y);       \
    if (!tile.active || split >= tile.splits)                                  \
      return;                                                                  \
    inkling2_prefill_page_loop<Group, PAGED_PREFILL_TILE_ROWS, HeadDim, true,  \
                               Int8Queries>(                                   \
        queries + tile.offset, table, key_pool, value_pool, key_scales,        \
        value_scales, rel_bias_raw, rel_bias_w, params, kv_head,               \
        tile.first_position, tile.first_row, tile.active_rows, tile.splits,    \
        split, partials, statistics, tile.first_slot + split, scores,          \
        probabilities, static_cast<threadgroup int8_t *>(nullptr), row_max,    \
        row_sum, previous_scale, &rescale, thread_index);                      \
  }

#define INKLING2_PAGED_PREFILL_BF16_SPLIT(Name, Group, HeadDim)                \
  [[kernel, max_total_threads_per_threadgroup(256)]] void Name(                \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const int *table [[buffer(3)]],                                   \
      device bfloat *key_pool [[buffer(4)]],                                   \
      device bfloat *value_pool [[buffer(5)]],                                 \
      device const uchar *rel_bias_raw [[buffer(6)]],                          \
      constant Inkling2PagedPrefillParams &params [[buffer(7)]],               \
      device const float *rel_bias_w [[buffer(8)]],                            \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * PAGED_PREFILL_TILE_ROWS;                        \
    constexpr uint N = PAGED_PREFILL_PAGE_TOKENS;                              \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    alignas(16) threadgroup bfloat keys[N * HeadDim];                          \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    const uint split = group.z;                                                \
    const uint kv_head = group.x;                                              \
    const PagedPrefillTile tile =                                              \
        inkling2_prefill_tile<Group, HeadDim>(params, kv_head, group.y);       \
    if (!tile.active || split >= tile.splits)                                  \
      return;                                                                  \
    inkling2_prefill_page_loop<Group, PAGED_PREFILL_TILE_ROWS, HeadDim, false, \
                               false>(                                         \
        queries + tile.offset, table, key_pool, value_pool, nullptr, nullptr,  \
        rel_bias_raw, rel_bias_w, params, kv_head, tile.first_position,        \
        tile.first_row, tile.active_rows, tile.splits, split, partials,        \
        statistics, tile.first_slot + split, scores, probabilities, keys,      \
        row_max, row_sum, previous_scale, &rescale, thread_index);             \
  }

#define INKLING2_PAGED_PREFILL_REDUCE(Name, Group, HeadDim)                    \
  [[kernel, max_total_threads_per_threadgroup(HeadDim)]] void Name(            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      device float *lse_out [[buffer(3)]],                                     \
      constant Inkling2PagedPrefillParams &params [[buffer(4)]],               \
      device const uchar *gate_raw [[buffer(5)]],                              \
      device const uchar *sink_raw [[buffer(6)]],                              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[PAGED_PREFILL_MAX_SPLITS];                       \
    threadgroup float group_values[HeadDim / 32];                              \
    inkling2_prefill_reduce_phase<Group, HeadDim>(                             \
        partials, statistics, output, lse_out, gate_raw, sink_raw, params,     \
        group, thread_index, weights, group_values);                           \
  }

INKLING2_PAGED_PREFILL_KTM_SPLIT(inkling2_paged_prefill_q8_split_g3_d128_ktm, 3, 128, false)
INKLING2_PAGED_PREFILL_KTM_SPLIT(inkling2_paged_prefill_q8q_split_g3_d128_ktm, 3, 128, true)
INKLING2_PAGED_PREFILL_BF16_SPLIT(inkling2_paged_prefill_bf16_split_g3_d128, 3, 128)
INKLING2_PAGED_PREFILL_REDUCE(inkling2_paged_prefill_reduce_g3_d128, 3, 128)
