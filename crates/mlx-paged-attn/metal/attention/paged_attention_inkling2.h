// Inkling2 relative-bias + natural-log LSE paged attention decode / varlen
// kernels (HEAD_SIZE = 128):
//
// - `paged_attention_inkling2_ktm`: INT8 32-token keys-token-major page
//   (K char [blocks][KV][32][128], V char [blocks][KV][128][32],
//   k_scale/v_scale f32 [blocks][KV][32]), V1 (ps=0) and V2 (ps=512).
// - `paged_attention_inkling2_bf16`: BF16 vLLM page
//   (K bfloat16_t [blocks][KV][16][bs][8], V bfloat16_t [blocks][KV][128][bs])
//   for block_size = 16 and 32, V1 (ps=0) and V2 (ps=512).
// - `paged_attention_inkling2_v2_reduce`: shared FP32 partition reducer for
//   both KTM INT8 and BF16 V2, emitting both BF16 `out` and F32 `lse`.
//
// All decode kernels natively support varlen (`cu_seqlens_q` + `num_seqs`),
// covering single-token decode, batched `[N, 1]` decode, 8-row DSpark verify,
// and mixed multi-sequence batches in a single path.
//
// For each query row `q_token_idx` in sequence `seq_idx`:
//   effective_context_len = context_lens[seq_idx] - q_len + q_pos_in_seq + 1
//   q_pos = effective_context_len - 1
//   dist  = q_pos - token_idx
//   valid iff 0 <= dist && (sliding_window == 0 || dist < sliding_window)
//   if valid && rel_extent > 0 && dist < rel_extent:
//     qk += rel_bias[q_token_idx, head_idx, dist] (rel_mode == 1)
//        or dot(r[q_token_idx, head_idx, 0..16], rel_bias_w[dist, 0..16]) (rel_mode == 2)
//
// Optional fused gate+sink epilogue (when `extra.has_gate_sink != 0`):
//   o_bf16 = float(bfloat16_t(o_raw))
//   y = bfloat16_t((o_bf16 * precise_sigmoid(gate)) * precise_sigmoid(lse - sink))

struct Inkling2AttnExtraParams {
  uint32_t has_gate_sink;
  uint32_t gate_row_stride;
  uint32_t gate_is_f32;
  uint32_t sink_row_stride;
  uint32_t sink_is_f32;
  uint32_t write_lse;
  uint32_t rel_mode; // 0 = none, 1 = precomputed [T,H,rel_extent], 2 = on-the-fly r [T,H,16] x w [rel_extent,16]
  uint32_t _pad;
};

inline float inkling2_read_rel_bias(const device uchar *rel_bias_raw,
                                    int64_t idx, int rel_bias_is_f32) {
  if (rel_bias_is_f32 != 0) {
    return reinterpret_cast<const device float *>(rel_bias_raw)[idx];
  }
  return float(reinterpret_cast<const device bfloat16_t *>(rel_bias_raw)[idx]);
}

inline float inkling2_read_epilogue_f32(const device uchar *raw, int64_t idx,
                                        uint32_t is_f32) {
  if (is_f32 != 0u) {
    return reinterpret_cast<const device float *>(raw)[idx];
  }
  return float(reinterpret_cast<const device bfloat16_t *>(raw)[idx]);
}

#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
inline float inkling2_rel_dot8(float4 ra, float4 rb, float4 wa, float4 wb) {
  const float da = ra.x * wa.x + ra.y * wa.y + ra.z * wa.z + ra.w * wa.w;
  const float db = rb.x * wb.x + rb.y * wb.y + rb.z * wb.z + rb.w * wb.w;
  return da + db;
}

inline float inkling2_precise_sigmoid(float x) {
  return precise::divide(1.0f, 1.0f + precise::exp(-x));
}

inline float inkling2_apply_gate_sink(float o_f, float g_val, float sink_scale) {
  const float o_bf16 = float(bfloat16_t(o_f));
  const float g_sig = inkling2_precise_sigmoid(g_val);
  return (o_bf16 * g_sig) * sink_scale;
}
#pragma clang fp reassociate(on)
#pragma clang fp contract(fast)

// ============================================================================
// Unified Inkling2 decode / varlen implementation (KTM INT8 & BF16 vLLM pool)
// Both modes write unnormalized FP32 partials in V2 so they share the exact
// same FP32 reducer (`paged_attention_inkling2_v2_reduce`).
// ============================================================================
template <int HEAD_SIZE, int BLOCK_SIZE, int PARTITION_SIZE, bool IsInt8Ktm>
inline void paged_attention_inkling2_impl(
    device float *exp_sums,
    device float *max_logits,
    device typename PagedOut<bfloat16_t, char, (PARTITION_SIZE > 0)>::Type *out,
    device const bfloat16_t *q,
    device const conditional_t<IsInt8Ktm, char, bfloat16_t> *k_cache,
    device const conditional_t<IsInt8Ktm, char, bfloat16_t> *v_cache,
    const device float *__restrict__ k_scale,
    const device float *__restrict__ v_scale,
    const constant int &num_kv_heads,
    const constant float &scale,
    device const uint32_t *block_tables,
    device const uint32_t *context_lens,
    const constant int &max_num_blocks_per_seq,
    const constant int &q_stride,
    const constant int &kv_block_stride,
    const constant int &kv_head_stride,
    const constant int &sliding_window,
    device const int32_t *cu_seqlens_q,
    const constant int &num_seqs,
    const device uchar *rel_bias_raw,
    const constant int &rel_extent,
    const constant int &rel_bias_is_f32,
    const device float *score_scales,
    const int has_score_scales,
    device float *lse_out,
    const device uchar *gate_raw,
    const device uchar *sink_raw,
    const device float *rel_bias_w,
    const constant Inkling2AttnExtraParams &extra,
    threadgroup char *shared_mem,
    threadgroup typename PagedAttentionLayout<
        bfloat16_t, HEAD_SIZE, BLOCK_SIZE, 256, 32>::Q_vec (
        &q_vecs)[PagedAttentionLayout<bfloat16_t, HEAD_SIZE, BLOCK_SIZE, 256,
                                      32>::THREAD_GROUP_SIZE]
                [PagedAttentionLayout<bfloat16_t, HEAD_SIZE, BLOCK_SIZE, 256,
                                      32>::NUM_VECS_PER_THREAD],
    threadgroup float *red_smem,
    uint3 threadgroup_position_in_grid,
    uint3 threadgroups_per_grid,
    uint3 thread_position_in_threadgroup,
    uint simd_tid,
    uint simd_lid) {
  typedef bfloat16_t T;
  using CACHE_T = conditional_t<IsInt8Ktm, char, bfloat16_t>;
  constexpr int PAGE = IsInt8Ktm ? 32 : BLOCK_SIZE;
  constexpr int NUM_THREADS = 256;
  constexpr int NUM_SIMD_LANES = 32;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  constexpr int x = 16 / sizeof(T); // 8 for BF16

  const int head_idx = int(threadgroup_position_in_grid.x);
  const int num_heads = int(threadgroups_per_grid.x);
  const int q_token_idx = int(threadgroup_position_in_grid.y);
  const int partition_idx = int(threadgroup_position_in_grid.z);
  const int max_num_partitions = int(threadgroups_per_grid.z);
  const int thread_idx = int(thread_position_in_threadgroup.x);

  const int seq_idx = find_seq_idx_varlen(cu_seqlens_q, q_token_idx, num_seqs);
  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  const int q_pos_in_seq = q_token_idx - q_seq_start;
  const int effective_context_len =
      int(context_lens[seq_idx]) - q_len + q_pos_in_seq + 1;
  if (effective_context_len <= 0) {
    if (!USE_PARTITIONING && thread_idx == 0 && extra.write_lse != 0u) {
      lse_out[q_token_idx * num_heads + head_idx] = -INFINITY;
    }
    return;
  }
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= effective_context_len) {
    return;
  }

  const int q_pos = effective_context_len - 1;
  const int sliding_lower =
      (sliding_window > 0 && effective_context_len > sliding_window)
          ? (effective_context_len - sliding_window)
          : 0;

  const int num_context_blocks =
      DIVIDE_ROUND_UP(effective_context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, effective_context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  // Fast skip for partitions entirely before the sliding window lower bound.
  if (USE_PARTITIONING && end_token_idx <= sliding_lower) {
    if (thread_idx == 0) {
      const int64_t stat_idx =
          (int64_t(q_token_idx) * num_heads + head_idx) * max_num_partitions +
          partition_idx;
      max_logits[stat_idx] = -INFINITY;
      exp_sums[stat_idx] = 0.0f;
    }
    return;
  }

  constexpr int THREAD_GROUP_SIZE = MAX(NUM_SIMD_LANES / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, NUM_SIMD_LANES);
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = int(simd_tid);
  const int lane = int(simd_lid);

  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float score_scale =
      (IsInt8Ktm && has_score_scales)
          ? score_scales[q_token_idx * num_heads + head_idx]
          : scale;

  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using K_vec = typename Vec<T, VEC_SIZE>::Type;
  using Q_vec = typename Vec<T, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<CACHE_T, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  const device T *q_ptr = q + int64_t(q_token_idx) * q_stride + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const device Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float *logits = reinterpret_cast<threadgroup float *>(shared_mem);

  float qk_max = -FLT_MAX;
  const int64_t rel_bias_row_base =
      (int64_t(q_token_idx) * num_heads + head_idx) * int64_t(rel_extent);
  float4 r0(0.0f), r1(0.0f), r2(0.0f), r3(0.0f);
  if (extra.rel_mode == 2u && rel_extent > 0) {
    const device float4 *r4 =
        reinterpret_cast<const device float4 *>(rel_bias_raw) +
        (int64_t(q_token_idx) * num_heads + head_idx) * 4;
    if constexpr (THREAD_GROUP_SIZE == 2) {
      r0 = r4[thread_group_offset * 2];
      r1 = r4[thread_group_offset * 2 + 1];
    } else {
      r0 = r4[0];
      r1 = r4[1];
      r2 = r4[2];
      r3 = r4[3];
    }
  }

  const device uint32_t *block_table =
      block_tables + int64_t(seq_idx) * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    if ((block_idx + 1) * BLOCK_SIZE <= sliding_lower) {
      continue;
    }
    const int page_half = IsInt8Ktm ? (block_idx & 1) : 0;
    const int64_t physical_block = static_cast<int64_t>(
        IsInt8Ktm ? block_table[block_idx >> 1] : block_table[block_idx]);

    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * NUM_SIMD_LANES) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      const bool valid =
          token_idx < effective_context_len && token_idx >= sliding_lower;

      float qk = -INFINITY;
      if (valid) {
        K_vec k_vecs[NUM_VECS_PER_THREAD];
        if constexpr (IsInt8Ktm) {
          const device CACHE_T *k_ptr =
              k_cache + physical_block * int64_t(kv_block_stride) +
              int64_t(kv_head_idx) * kv_head_stride +
              (page_half * BLOCK_SIZE + physical_block_offset) * HEAD_SIZE;
#pragma unroll
          for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
            const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
            Quant_vec k_vec_quant = *reinterpret_cast<const device Quant_vec *>(
                k_ptr + vec_idx * VEC_SIZE);
            k_vecs[j] = char_to<K_vec, Quant_vec>(k_vec_quant);
          }

          qk = Qk_dot<T, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset],
                                                 k_vecs);
          qk *= k_scale[(physical_block * num_kv_heads + kv_head_idx) * PAGE +
                        page_half * BLOCK_SIZE + physical_block_offset];
          qk *= score_scale;
        } else {
          const device T *k_ptr =
              k_cache + physical_block * int64_t(kv_block_stride) +
              int64_t(kv_head_idx) * kv_head_stride + physical_block_offset * x;
#pragma unroll
          for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
            const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
            const int offset1 = (vec_idx * VEC_SIZE) / x;
            const int offset2 = (vec_idx * VEC_SIZE) % x;
            k_vecs[j] = *reinterpret_cast<const device K_vec *>(
                k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          }

          qk = scale * Qk_dot<T, THREAD_GROUP_SIZE>::dot(
                           q_vecs[thread_group_offset], k_vecs);
        }

        if (rel_extent > 0) {
          const int dist = q_pos - token_idx;
          if (dist < rel_extent) {
            if (extra.rel_mode == 2u) {
              const device float4 *w4 =
                  reinterpret_cast<const device float4 *>(rel_bias_w) +
                  int64_t(dist) * 4;
              if constexpr (THREAD_GROUP_SIZE == 2) {
                float rb = inkling2_rel_dot8(
                    r0, r1, w4[thread_group_offset * 2],
                    w4[thread_group_offset * 2 + 1]);
                rb += simd_shuffle_xor(rb, 1);
                if (thread_group_offset == 0) {
                  qk += rb;
                }
              } else {
                const float rb = inkling2_rel_dot8(r0, r1, w4[0], w4[1]) +
                                 inkling2_rel_dot8(r2, r3, w4[2], w4[3]);
                qk += rb;
              }
            } else if (thread_group_offset == 0 && extra.rel_mode == 1u) {
              qk += inkling2_read_rel_bias(
                  rel_bias_raw, rel_bias_row_base + dist, rel_bias_is_f32);
            }
          }
        }
      }

      if (thread_group_offset == 0 && token_idx < end_token_idx) {
        logits[token_idx - start_token_idx] = valid ? qk : -INFINITY;
        if (valid) {
          qk_max = max(qk_max, qk);
        }
      }
    }
  }

#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  qk_max = simd_shuffle(qk_max, 0);

  float exp_sum = 0.0f;
  const bool has_valid = qk_max > -FLT_MAX;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    const int tok = start_token_idx + i;
    const bool valid = tok >= sliding_lower && tok < effective_context_len;
    const float val = (has_valid && valid) ? fast::exp(logits[i] - qk_max) : 0.0f;
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(&red_smem[NUM_WARPS], exp_sum,
                                                 simd_tid, simd_lid);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (USE_PARTITIONING && thread_idx == 0) {
    const int64_t stat_idx =
        (int64_t(q_token_idx) * num_heads + head_idx) * max_num_partitions +
        partition_idx;
    max_logits[stat_idx] = (exp_sum > 0.0f) ? qk_max : -INFINITY;
    exp_sums[stat_idx] = exp_sum;
  }
  if (!USE_PARTITIONING && thread_idx == 0 && extra.write_lse != 0u) {
    lse_out[q_token_idx * num_heads + head_idx] =
        (exp_sum > 0.0f) ? (qk_max + precise::log(exp_sum)) : -INFINITY;
  }

  constexpr int V_VEC_SIZE = MIN(16 / sizeof(T), BLOCK_SIZE);
  using V_vec = typename Vec<T, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<T, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;
  using V_quant_vec = typename Vec<CACHE_T, V_VEC_SIZE>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = NUM_SIMD_LANES / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.0f;
  }

  T zero_value = 0;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    if ((block_idx + 1) * BLOCK_SIZE <= sliding_lower) {
      continue;
    }
    const int page_half = IsInt8Ktm ? (block_idx & 1) : 0;
    const int64_t physical_block = static_cast<int64_t>(
        IsInt8Ktm ? block_table[block_idx >> 1] : block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    Float_L_vec logits_float_vec = *reinterpret_cast<threadgroup Float_L_vec *>(
        logits + token_idx - start_token_idx);
    {
      thread float *p = reinterpret_cast<thread float *>(&logits_float_vec);
      if constexpr (IsInt8Ktm) {
        const device float *value_scales =
            v_scale + (physical_block * num_kv_heads + kv_head_idx) * PAGE +
            page_half * BLOCK_SIZE + physical_block_offset;
#pragma unroll
        for (int j = 0; j < V_VEC_SIZE; j++) {
          const bool valid = token_idx + j < effective_context_len &&
                             token_idx + j >= sliding_lower;
          p[j] = valid ? p[j] * value_scales[j] : 0.0f;
        }
      } else {
#pragma unroll
        for (int j = 0; j < V_VEC_SIZE; j++) {
          const bool valid = token_idx + j < effective_context_len &&
                             token_idx + j >= sliding_lower;
          if (!valid) {
            p[j] = 0.0f;
          }
        }
      }
    }
    L_vec logits_vec;
    from_float(logits_vec, logits_float_vec);

    const device CACHE_T *v_ptr = v_cache +
                                  physical_block * int64_t(kv_block_stride) +
                                  int64_t(kv_head_idx) * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset =
            row_idx * PAGE + page_half * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;
        if constexpr (IsInt8Ktm) {
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const device V_quant_vec *>(v_ptr + offset);
          v_vec = char_to<V_vec, V_quant_vec>(v_quant_vec);
        } else {
          v_vec = *reinterpret_cast<const device V_vec *>(v_ptr + offset);
        }
        if (block_idx == num_context_blocks - 1) {
          thread T *v_vec_ptr = reinterpret_cast<thread T *>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < effective_context_len
                               ? v_vec_ptr[j]
                               : zero_value;
          }
        }
        accs[i] += dot<Float_L_vec>(logits_vec, v_vec);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += simd_shuffle_xor(acc, mask);
    }
    accs[i] = acc;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float *out_smem =
      reinterpret_cast<threadgroup float *>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    if (warp_idx >= mid && warp_idx < i) {
      threadgroup float *dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int r = 0; r < NUM_ROWS_PER_THREAD; r++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + r * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[r];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_idx < mid) {
      const threadgroup float *src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int r = 0; r < NUM_ROWS_PER_THREAD; r++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + r * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[r] += src[row_idx];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (warp_idx == 0) {
    using Out_T = typename PagedOut<T, char, USE_PARTITIONING>::Type;
    device Out_T *out_ptr =
        out +
        (int64_t(q_token_idx) * num_heads + head_idx) * max_num_partitions *
            HEAD_SIZE +
        int64_t(partition_idx) * HEAD_SIZE;
    float sink_scale = 1.0f;
    if constexpr (!USE_PARTITIONING) {
      if (extra.has_gate_sink != 0u) {
        const float lse_val =
            (exp_sum > 0.0f) ? (qk_max + precise::log(exp_sum)) : -INFINITY;
        const float sink_val = inkling2_read_epilogue_f32(
            sink_raw,
            int64_t(q_token_idx) * int64_t(extra.sink_row_stride) + head_idx,
            extra.sink_is_f32);
        sink_scale = inkling2_precise_sigmoid(lse_val - sink_val);
      }
    }
    const int64_t gate_row_base =
        int64_t(q_token_idx) * int64_t(extra.gate_row_stride) +
        int64_t(head_idx) * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        if constexpr (!USE_PARTITIONING) {
          float o_f = (exp_sum > 0.0f) ? (accs[i] / exp_sum) : 0.0f;
          if (extra.has_gate_sink != 0u) {
            const float g_val = inkling2_read_epilogue_f32(
                gate_raw, gate_row_base + row_idx, extra.gate_is_f32);
            o_f = inkling2_apply_gate_sink(o_f, g_val, sink_scale);
          }
          *(out_ptr + row_idx) = Out_T(o_f);
        } else {
          *(out_ptr + row_idx) = Out_T(accs[i]);
        }
      }
    }
  }
}

// ============================================================================
// INT8 32-token KTM Inkling2 decode / varlen entry (buffers 0..28)
// ============================================================================
template <int HEAD_SIZE, int PARTITION_SIZE>
[[kernel]] void paged_attention_inkling2_ktm(
    device float *exp_sums [[buffer(0)]],
    device float *max_logits [[buffer(1)]],
    device typename PagedOut<bfloat16_t, char, (PARTITION_SIZE > 0)>::Type *out
    [[buffer(2)]],
    device const bfloat16_t *q [[buffer(3)]],
    device const char *k_cache [[buffer(4)]],
    device const char *v_cache [[buffer(5)]],
    const device float *__restrict__ k_scale [[buffer(6)]],
    const device float *__restrict__ v_scale [[buffer(7)]],
    const constant int &num_kv_heads [[buffer(8)]],
    const constant float &scale [[buffer(9)]],
    device const uint32_t *block_tables [[buffer(10)]],
    device const uint32_t *context_lens [[buffer(11)]],
    const constant int &max_num_blocks_per_seq [[buffer(12)]],
    const constant int &q_stride [[buffer(13)]],
    const constant int &kv_block_stride [[buffer(14)]],
    const constant int &kv_head_stride [[buffer(15)]],
    const constant int &sliding_window [[buffer(16)]],
    device const int32_t *cu_seqlens_q [[buffer(17)]],
    const constant int &num_seqs [[buffer(18)]],
    const device uchar *rel_bias_raw [[buffer(19)]],
    const constant int &rel_extent [[buffer(20)]],
    const constant int &rel_bias_is_f32 [[buffer(21)]],
    const device float *score_scales [[buffer(22)]],
    const constant int &has_score_scales [[buffer(23)]],
    device float *lse_out [[buffer(24)]],
    const device uchar *gate_raw [[buffer(25)]],
    const device uchar *sink_raw [[buffer(26)]],
    const device float *rel_bias_w [[buffer(27)]],
    const constant Inkling2AttnExtraParams &extra [[buffer(28)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using Layout = PagedAttentionLayout<bfloat16_t, HEAD_SIZE, 16, 256, 32>;
  threadgroup typename Layout::Q_vec
      q_vecs[Layout::THREAD_GROUP_SIZE][Layout::NUM_VECS_PER_THREAD];
  threadgroup float red_smem[2 * Layout::NUM_WARPS];
  paged_attention_inkling2_impl<HEAD_SIZE, 16, PARTITION_SIZE, true>(
      exp_sums, max_logits, out, q, k_cache, v_cache, k_scale, v_scale,
      num_kv_heads, scale, block_tables, context_lens, max_num_blocks_per_seq,
      q_stride, kv_block_stride, kv_head_stride, sliding_window, cu_seqlens_q,
      num_seqs, rel_bias_raw, rel_extent, rel_bias_is_f32, score_scales,
      has_score_scales, lse_out, gate_raw, sink_raw, rel_bias_w, extra,
      shared_mem, q_vecs, red_smem, threadgroup_position_in_grid,
      threadgroups_per_grid, thread_position_in_threadgroup, simd_tid,
      simd_lid);
}

// ============================================================================
// BF16 vLLM-pool Inkling2 decode / varlen entry (buffers 0..24)
// ============================================================================
template <int HEAD_SIZE, int BLOCK_SIZE, int PARTITION_SIZE>
[[kernel]] void paged_attention_inkling2_bf16(
    device float *exp_sums [[buffer(0)]],
    device float *max_logits [[buffer(1)]],
    device typename PagedOut<bfloat16_t, char, (PARTITION_SIZE > 0)>::Type *out
    [[buffer(2)]],
    device const bfloat16_t *q [[buffer(3)]],
    device const bfloat16_t *k_cache [[buffer(4)]],
    device const bfloat16_t *v_cache [[buffer(5)]],
    const constant int &num_kv_heads [[buffer(6)]],
    const constant float &scale [[buffer(7)]],
    device const uint32_t *block_tables [[buffer(8)]],
    device const uint32_t *context_lens [[buffer(9)]],
    const constant int &max_num_blocks_per_seq [[buffer(10)]],
    const constant int &q_stride [[buffer(11)]],
    const constant int &kv_block_stride [[buffer(12)]],
    const constant int &kv_head_stride [[buffer(13)]],
    const constant int &sliding_window [[buffer(14)]],
    device const int32_t *cu_seqlens_q [[buffer(15)]],
    const constant int &num_seqs [[buffer(16)]],
    const device uchar *rel_bias_raw [[buffer(17)]],
    const constant int &rel_extent [[buffer(18)]],
    const constant int &rel_bias_is_f32 [[buffer(19)]],
    device float *lse_out [[buffer(20)]],
    const device uchar *gate_raw [[buffer(21)]],
    const device uchar *sink_raw [[buffer(22)]],
    const device float *rel_bias_w [[buffer(23)]],
    const constant Inkling2AttnExtraParams &extra [[buffer(24)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using Layout =
      PagedAttentionLayout<bfloat16_t, HEAD_SIZE, BLOCK_SIZE, 256, 32>;
  threadgroup typename Layout::Q_vec
      q_vecs[Layout::THREAD_GROUP_SIZE][Layout::NUM_VECS_PER_THREAD];
  threadgroup float red_smem[2 * Layout::NUM_WARPS];
  paged_attention_inkling2_impl<HEAD_SIZE, BLOCK_SIZE, PARTITION_SIZE, false>(
      exp_sums, max_logits, out, q, k_cache, v_cache, nullptr, nullptr,
      num_kv_heads, scale, block_tables, context_lens, max_num_blocks_per_seq,
      q_stride, kv_block_stride, kv_head_stride, sliding_window, cu_seqlens_q,
      num_seqs, rel_bias_raw, rel_extent, rel_bias_is_f32, nullptr, 0, lse_out,
      gate_raw, sink_raw, rel_bias_w, extra, shared_mem, q_vecs, red_smem,
      threadgroup_position_in_grid, threadgroups_per_grid,
      thread_position_in_threadgroup, simd_tid, simd_lid);
}

// ============================================================================
// Shared FP32 V2 partition reducer for Inkling2 (emits BF16 `out` + F32 `lse`)
// ============================================================================
template <int HEAD_SIZE, int NUM_THREADS, int PARTITION_SIZE>
[[kernel]] void paged_attention_inkling2_v2_reduce(
    device bfloat16_t *out [[buffer(0)]],
    device float *lse_out [[buffer(1)]],
    const device float *exp_sums [[buffer(2)]],
    const device float *max_logits [[buffer(3)]],
    const device float *tmp_out [[buffer(4)]],
    const device uint32_t *context_lens [[buffer(5)]],
    const constant int &max_num_partitions [[buffer(6)]],
    device const int32_t *cu_seqlens_q [[buffer(7)]],
    const constant int &num_seqs [[buffer(8)]],
    const device uchar *gate_raw [[buffer(9)]],
    const device uchar *sink_raw [[buffer(10)]],
    const constant Inkling2AttnExtraParams &extra [[buffer(11)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
  const int num_heads = int(threadgroups_per_grid.x);
  const int head_idx = int(threadgroup_position_in_grid.x);
  const int q_token_idx = int(threadgroup_position_in_grid.y);
  const int thread_idx = int(thread_position_in_threadgroup.x);

  const int seq_idx = find_seq_idx_varlen(cu_seqlens_q, q_token_idx, num_seqs);
  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  const int effective_context_len =
      int(context_lens[seq_idx]) - q_len + (q_token_idx - q_seq_start) + 1;
  const int64_t row = int64_t(q_token_idx) * num_heads + head_idx;

  if (effective_context_len <= 0) {
    if (thread_idx == 0 && extra.write_lse != 0u) {
      lse_out[row] = -INFINITY;
    }
    for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
      out[row * HEAD_SIZE + i] = bfloat16_t(0.0f);
    }
    return;
  }

  const int count = DIVIDE_ROUND_UP(effective_context_len, PARTITION_SIZE);
  const device float *sums = exp_sums + row * max_num_partitions;
  const device float *maxima = max_logits + row * max_num_partitions;
  const device float *partials =
      tmp_out + row * int64_t(max_num_partitions) * HEAD_SIZE;
  threadgroup float *weights = reinterpret_cast<threadgroup float *>(shared_mem);

  float maximum = -INFINITY;
  for (int p = 0; p < count; ++p) {
    if (sums[p] > 0.0f) {
      maximum = max(maximum, maxima[p]);
    }
  }
  for (int p = thread_idx; p < count; p += NUM_THREADS) {
    weights[p] = (sums[p] > 0.0f && maximum > -INFINITY)
                     ? fast::exp(maxima[p] - maximum)
                     : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int64_t gate_row_base =
      int64_t(q_token_idx) * int64_t(extra.gate_row_stride) +
      int64_t(head_idx) * HEAD_SIZE;
  for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int p = 0; p < count; ++p) {
      const float w = weights[p];
      if (w > 0.0f) {
        numerator += w * partials[int64_t(p) * HEAD_SIZE + i];
        denominator += w * sums[p];
      }
    }
    float o_f = (denominator > 0.0f) ? (numerator / denominator) : 0.0f;
    const float lse_val =
        (denominator > 0.0f) ? (maximum + precise::log(denominator)) : -INFINITY;
    if (extra.has_gate_sink != 0u) {
      const float sink_val = inkling2_read_epilogue_f32(
          sink_raw,
          int64_t(q_token_idx) * int64_t(extra.sink_row_stride) + head_idx,
          extra.sink_is_f32);
      const float sink_scale = inkling2_precise_sigmoid(lse_val - sink_val);
      const float g_val = inkling2_read_epilogue_f32(
          gate_raw, gate_row_base + i, extra.gate_is_f32);
      o_f = inkling2_apply_gate_sink(o_f, g_val, sink_scale);
    }
    out[row * HEAD_SIZE + i] = bfloat16_t(o_f);
    if (i == 0 && extra.write_lse != 0u) {
      lse_out[row] = lse_val;
    }
  }
}

// ============================================================================
// Explicit instantiations (HEAD_SIZE = 128)
// ============================================================================
#define instantiate_paged_attention_inkling2_ktm(name, out_type, ps)           \
  template [[host_name(name)]] [[kernel]] void                                 \
  paged_attention_inkling2_ktm<128, ps>(                                       \
      device float *exp_sums [[buffer(0)]],                                    \
      device float *max_logits [[buffer(1)]],                                  \
      device out_type *out [[buffer(2)]],                                      \
      device const bfloat16_t *q [[buffer(3)]],                                \
      device const char *k_cache [[buffer(4)]],                                \
      device const char *v_cache [[buffer(5)]],                                \
      const device float *__restrict__ k_scale [[buffer(6)]],                  \
      const device float *__restrict__ v_scale [[buffer(7)]],                  \
      const constant int &num_kv_heads [[buffer(8)]],                          \
      const constant float &scale [[buffer(9)]],                               \
      device const uint32_t *block_tables [[buffer(10)]],                      \
      device const uint32_t *context_lens [[buffer(11)]],                      \
      const constant int &max_num_blocks_per_seq [[buffer(12)]],               \
      const constant int &q_stride [[buffer(13)]],                             \
      const constant int &kv_block_stride [[buffer(14)]],                      \
      const constant int &kv_head_stride [[buffer(15)]],                       \
      const constant int &sliding_window [[buffer(16)]],                       \
      device const int32_t *cu_seqlens_q [[buffer(17)]],                       \
      const constant int &num_seqs [[buffer(18)]],                             \
      const device uchar *rel_bias_raw [[buffer(19)]],                         \
      const constant int &rel_extent [[buffer(20)]],                           \
      const constant int &rel_bias_is_f32 [[buffer(21)]],                      \
      const device float *score_scales [[buffer(22)]],                         \
      const constant int &has_score_scales [[buffer(23)]],                     \
      device float *lse_out [[buffer(24)]],                                    \
      const device uchar *gate_raw [[buffer(25)]],                             \
      const device uchar *sink_raw [[buffer(26)]],                             \
      const device float *rel_bias_w [[buffer(27)]],                           \
      const constant Inkling2AttnExtraParams &extra [[buffer(28)]],            \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup                                     \
          [[thread_position_in_threadgroup]],                                  \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

instantiate_paged_attention_inkling2_ktm(
    "paged_attention_inkling2_ktm_hs128_ps0", bfloat16_t, 0);
instantiate_paged_attention_inkling2_ktm(
    "paged_attention_inkling2_ktm_hs128_ps512", float, 512);

#define instantiate_paged_attention_inkling2_bf16(name, out_type, bs, ps)      \
  template [[host_name(name)]] [[kernel]] void                                 \
  paged_attention_inkling2_bf16<128, bs, ps>(                                  \
      device float *exp_sums [[buffer(0)]],                                    \
      device float *max_logits [[buffer(1)]],                                  \
      device out_type *out [[buffer(2)]],                                      \
      device const bfloat16_t *q [[buffer(3)]],                                \
      device const bfloat16_t *k_cache [[buffer(4)]],                          \
      device const bfloat16_t *v_cache [[buffer(5)]],                          \
      const constant int &num_kv_heads [[buffer(6)]],                          \
      const constant float &scale [[buffer(7)]],                               \
      device const uint32_t *block_tables [[buffer(8)]],                       \
      device const uint32_t *context_lens [[buffer(9)]],                       \
      const constant int &max_num_blocks_per_seq [[buffer(10)]],               \
      const constant int &q_stride [[buffer(11)]],                             \
      const constant int &kv_block_stride [[buffer(12)]],                      \
      const constant int &kv_head_stride [[buffer(13)]],                       \
      const constant int &sliding_window [[buffer(14)]],                       \
      device const int32_t *cu_seqlens_q [[buffer(15)]],                       \
      const constant int &num_seqs [[buffer(16)]],                             \
      const device uchar *rel_bias_raw [[buffer(17)]],                         \
      const constant int &rel_extent [[buffer(18)]],                           \
      const constant int &rel_bias_is_f32 [[buffer(19)]],                      \
      device float *lse_out [[buffer(20)]],                                    \
      const device uchar *gate_raw [[buffer(21)]],                             \
      const device uchar *sink_raw [[buffer(22)]],                             \
      const device float *rel_bias_w [[buffer(23)]],                           \
      const constant Inkling2AttnExtraParams &extra [[buffer(24)]],            \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup                                     \
          [[thread_position_in_threadgroup]],                                  \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

instantiate_paged_attention_inkling2_bf16(
    "paged_attention_inkling2_bf16_hs128_bs16_ps0", bfloat16_t, 16, 0);
instantiate_paged_attention_inkling2_bf16(
    "paged_attention_inkling2_bf16_hs128_bs16_ps512", float, 16, 512);
instantiate_paged_attention_inkling2_bf16(
    "paged_attention_inkling2_bf16_hs128_bs32_ps0", bfloat16_t, 32, 0);
instantiate_paged_attention_inkling2_bf16(
    "paged_attention_inkling2_bf16_hs128_bs32_ps512", float, 32, 512);

template [[host_name("paged_attention_inkling2_v2_reduce_hs128_ps512")]]
[[kernel]] void paged_attention_inkling2_v2_reduce<128, 256, 512>(
    device bfloat16_t *out [[buffer(0)]],
    device float *lse_out [[buffer(1)]],
    const device float *exp_sums [[buffer(2)]],
    const device float *max_logits [[buffer(3)]],
    const device float *tmp_out [[buffer(4)]],
    const device uint32_t *context_lens [[buffer(5)]],
    const constant int &max_num_partitions [[buffer(6)]],
    device const int32_t *cu_seqlens_q [[buffer(7)]],
    const constant int &num_seqs [[buffer(8)]],
    const device uchar *gate_raw [[buffer(9)]],
    const device uchar *sink_raw [[buffer(10)]],
    const constant Inkling2AttnExtraParams &extra [[buffer(11)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]);
