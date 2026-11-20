// Decode reads of the keys-token-major page: K char [blocks][KV][32][D], V
// char [blocks][KV][D][32], key and value scales f32 [blocks][KV][32]. Each
// kernel is a block-16 int8 kernel of paged_attention.metal with its
// arithmetic and sum order unchanged: it walks 16-token logical blocks,
// logical block b being half b & 1 of page b >> 1, so its bits are the
// block-16 kernel's over the same rows.

// paged_attention_grouped_int8_bs16_striped over the keys-token-major page.
template <int HEAD_SIZE>
[[kernel]] void paged_attention_grouped_int8_ktm_striped(
    device float *exp_sums [[buffer(0)]],
    device float *max_logits [[buffer(1)]],
    device float *tmp_out [[buffer(2)]],
    device const bfloat16_t *q [[buffer(3)]],
    device const char *k_cache [[buffer(4)]],
    device const char *v_cache [[buffer(5)]],
    const device float *__restrict__ k_scale [[buffer(6)]],
    const device float *__restrict__ v_scale [[buffer(7)]],
    const constant int &num_kv_heads [[buffer(8)]],
    const constant float &scale [[buffer(9)]],
    const constant float &softcapping [[buffer(10)]],
    device const uint32_t *block_tables [[buffer(11)]],
    device const uint32_t *context_lens [[buffer(12)]],
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    device const float *alibi_slopes [[buffer(14)]],
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_block_stride [[buffer(16)]],
    const constant int &kv_head_stride [[buffer(17)]],
    const constant int &sliding_window [[buffer(18)]],
    const device float *score_scales [[buffer(21)]],
    const constant int &has_score_scales [[buffer(22)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BLOCK_SIZE = 16;
  constexpr int PAGE = 32;
  constexpr int PACKS_PER_HALF_HEAD = HEAD_SIZE / 16 / 2;
  constexpr int OUTPUTS_PER_LANE = HEAD_SIZE / 32;

  const int kv_head_idx = int(threadgroup_position_in_grid.x);
  const int seq_idx = 0;
  const int stripe_idx = int(threadgroup_position_in_grid.z);
  const int num_stripes = int(threadgroups_per_grid.z);
  const int local_q_head = int(thread_position_in_threadgroup.y);
  const int q_pos_in_seq = int(threadgroup_position_in_grid.y);
  const int q_len = int(threadgroups_per_grid.y);
  const int lane = int(thread_position_in_threadgroup.x);
  const int gqa_factor = int(threads_per_threadgroup.y);
  const int head_idx = kv_head_idx * gqa_factor + local_q_head;
  const int num_heads = q_stride / HEAD_SIZE;

  const uint32_t context_len = context_lens[seq_idx];
  const int effective_context_len =
      int(context_len) - q_len + q_pos_in_seq + 1;
  const int sliding_lower =
      (sliding_window > 0 && effective_context_len > sliding_window)
          ? (effective_context_len - sliding_window)
          : 0;
  const int q_token_idx = q_pos_in_seq;
  const device bfloat16_t *q_ptr =
      q + q_token_idx * q_stride + head_idx * HEAD_SIZE;
  const float score_scale =
      has_score_scales ? score_scales[q_token_idx * num_heads + head_idx]
                       : scale;

  float acc[OUTPUTS_PER_LANE];
#pragma unroll
  for (int i = 0; i < OUTPUTS_PER_LANE; ++i) {
    acc[i] = 0.0f;
  }
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  const device uint32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;

  const int token_in_block = lane / 2;
  const int half_head = lane & 1;
  const int num_context_blocks =
      DIVIDE_ROUND_UP(int(context_len), BLOCK_SIZE);
  const int per_stripe = DIVIDE_ROUND_UP(num_context_blocks, num_stripes);
  const int first_block = stripe_idx * per_stripe;
  const int end_block = min(first_block + per_stripe, num_context_blocks);

  for (int logical_block = first_block; logical_block < end_block;
       ++logical_block) {
    const int block_start_token = logical_block * BLOCK_SIZE;
    if (block_start_token + BLOCK_SIZE <= sliding_lower) {
      continue;
    }
    const int page_half = logical_block & 1;
    const int64_t physical_page =
        static_cast<int64_t>(block_table[logical_block >> 1]);
    const int64_t scale_base =
        (physical_page * num_kv_heads + kv_head_idx) * PAGE +
        page_half * BLOCK_SIZE;
    const device char *k_page = k_cache +
        physical_page * int64_t(kv_block_stride) +
        kv_head_idx * kv_head_stride;
    const device char *k_token =
        k_page + (page_half * BLOCK_SIZE + token_in_block) * HEAD_SIZE;
    float partial = 0.0f;
#pragma unroll
    for (int pack = 0; pack < PACKS_PER_HALF_HEAD; ++pack) {
      const int head_pack = half_head * PACKS_PER_HALF_HEAD + pack;
      const device bfloat16_t *q_pack = q_ptr + head_pack * 16;
      const device char *k_pack = k_token + head_pack * 16;
      const Char8_ k0 = *reinterpret_cast<const device Char8_ *>(k_pack);
      const Char8_ k1 = *reinterpret_cast<const device Char8_ *>(k_pack + 8);
      partial += sum(mul<Float8_, Bfloat8_, Bfloat8_>(
          *reinterpret_cast<const device Bfloat8_ *>(q_pack),
          char_to<Bfloat8_, Char8_>(k0)));
      partial += sum(mul<Float8_, Bfloat8_, Bfloat8_>(
          *reinterpret_cast<const device Bfloat8_ *>(q_pack + 8),
          char_to<Bfloat8_, Char8_>(k1)));
    }
    float score = partial + simd_shuffle_xor(partial, 1);
    score *= k_scale[scale_base + token_in_block];
    score *= score_scale;
    if (softcapping != 1.0f) {
      score = precise::tanh(score / softcapping) * softcapping;
    }

    const int token_idx = block_start_token + token_in_block;
    const bool valid =
        token_idx < effective_context_len && token_idx >= sliding_lower;
    const float owned_score = ((lane & 1) == 0 && valid) ? score : -FLT_MAX;
    const float block_max = simd_max(owned_score);
    const float new_max = max(running_max, block_max);
    const float old_factor =
        running_sum > 0.0f ? fast::exp(running_max - new_max) : 0.0f;
    const float owned_weight =
        ((lane & 1) == 0 && valid) ? fast::exp(score - new_max) : 0.0f;
    const float block_sum = simd_sum(owned_weight);

    const device float *value_scales = v_scale + scale_base;
    float weights[BLOCK_SIZE];
#pragma unroll
    for (int token = 0; token < BLOCK_SIZE; ++token) {
      const float p = simd_shuffle(owned_weight, token * 2);
      const int t = block_start_token + token;
      const bool token_valid = t < effective_context_len && t >= sliding_lower;
      weights[token] = token_valid ? p * value_scales[token] : 0.0f;
    }
    Float8_ weights0;
    weights0.x = float4(weights[0], weights[1], weights[2], weights[3]);
    weights0.y = float4(weights[4], weights[5], weights[6], weights[7]);
    Float8_ weights1;
    weights1.x = float4(weights[8], weights[9], weights[10], weights[11]);
    weights1.y = float4(weights[12], weights[13], weights[14], weights[15]);
    Bfloat8_ p0;
    Bfloat8_ p1;
    from_float(p0, weights0);
    from_float(p1, weights1);

    const device char *v_page = v_cache +
        physical_page * int64_t(kv_block_stride) +
        kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < OUTPUTS_PER_LANE; ++i) {
      const int dim = lane * OUTPUTS_PER_LANE + i;
      const device char *v_row = v_page + dim * PAGE + page_half * BLOCK_SIZE;
      const Char8_ v0 = *reinterpret_cast<const device Char8_ *>(v_row);
      const Char8_ v1 = *reinterpret_cast<const device Char8_ *>(v_row + 8);
      const Bfloat8_ values0 = char_to<Bfloat8_, Char8_>(v0);
      const Bfloat8_ values1 = char_to<Bfloat8_, Char8_>(v1);
      const float block_acc =
          sum(mul<Float8_, Bfloat8_, Bfloat8_>(p0, values0)) +
          sum(mul<Float8_, Bfloat8_, Bfloat8_>(p1, values1));
      acc[i] = acc[i] * old_factor + block_acc;
    }
    running_max = new_max;
    running_sum = running_sum * old_factor + block_sum;
  }

  const int row = q_token_idx * num_heads + head_idx;
  const int stats_offset = row * num_stripes + stripe_idx;
  if (simd_lid == 0) {
    exp_sums[stats_offset] = running_sum;
    max_logits[stats_offset] = running_max;
  }
  device float *out_ptr =
      tmp_out + stats_offset * HEAD_SIZE + lane * OUTPUTS_PER_LANE;
#pragma unroll
  for (int i = 0; i < OUTPUTS_PER_LANE; ++i) {
    out_ptr[i] = acc[i];
  }
  (void)alibi_slopes;
}

// The int8 branch of paged_attention<bfloat16_t, char, HEAD_SIZE, 16, 256, 32,
// PARTITION_SIZE> over the keys-token-major page.
template <int HEAD_SIZE, int PARTITION_SIZE>
[[kernel]] void paged_attention_ktm(
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
    const constant float &softcapping [[buffer(10)]],
    device const uint32_t *block_tables [[buffer(11)]],
    device const uint32_t *context_lens [[buffer(12)]],
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    device const float *alibi_slopes [[buffer(14)]],
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_block_stride [[buffer(16)]],
    const constant int &kv_head_stride [[buffer(17)]],
    const constant int &sliding_window [[buffer(18)]],
    const device float *score_scales [[buffer(21)]],
    const constant int &has_score_scales [[buffer(22)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  typedef bfloat16_t T;
  typedef char CACHE_T;
  constexpr int BLOCK_SIZE = 16;
  constexpr int PAGE = 32;
  constexpr int NUM_THREADS = 256;
  constexpr int NUM_SIMD_LANES = 32;
  const int seq_idx = threadgroup_position_in_grid.y;
  const int partition_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int thread_idx = thread_position_in_threadgroup.x;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const uint32_t context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(NUM_SIMD_LANES / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, NUM_SIMD_LANES);
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  const int head_idx = threadgroup_position_in_grid.x;
  const int num_heads = threadgroups_per_grid.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float score_scale =
      has_score_scales ? score_scales[seq_idx * num_heads + head_idx] : scale;

  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using K_vec = typename Vec<T, VEC_SIZE>::Type;
  using Q_vec = typename Vec<T, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<CACHE_T, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  const device T *q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  threadgroup Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const device Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float *logits = reinterpret_cast<threadgroup float *>(shared_mem);
  threadgroup float red_smem[2 * NUM_WARPS];

  float qk_max = -FLT_MAX;

  const device uint32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const int page_half = block_idx & 1;
    const int64_t physical_page =
        static_cast<int64_t>(block_table[block_idx >> 1]);

    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * NUM_SIMD_LANES) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

      const device CACHE_T *k_ptr =
          k_cache + physical_page * kv_block_stride +
          kv_head_idx * kv_head_stride +
          (page_half * BLOCK_SIZE + physical_block_offset) * HEAD_SIZE;
#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        Quant_vec k_vec_quant = *reinterpret_cast<const device Quant_vec *>(
            k_ptr + vec_idx * VEC_SIZE);
        k_vecs[j] = char_to<K_vec, Quant_vec>(k_vec_quant);
      }

      float qk = Qk_dot<T, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset],
                                                   k_vecs);
      qk *= k_scale[(physical_page * num_kv_heads + kv_head_idx) * PAGE +
                    page_half * BLOCK_SIZE + physical_block_offset];
      qk *= score_scale;

      if (softcapping != 1.0) {
        qk = precise::tanh(qk / softcapping) * softcapping;
      }

      if (thread_group_offset == 0) {
        const int sw = sliding_window;
        const uint32_t sliding_lower =
            (sw > 0 && context_len > uint32_t(sw))
                ? (context_len - uint32_t(sw))
                : 0u;
        const bool sliding_evicted =
            sw > 0 && uint32_t(token_idx) < sliding_lower;
        const bool out_of_context = token_idx >= context_len;
        float stored_logit;
        if (out_of_context) {
          stored_logit = 0.f;
        } else if (sliding_evicted) {
          stored_logit = -INFINITY;
        } else {
          stored_logit = qk;
        }
        logits[token_idx - start_token_idx] = stored_logit;
        const bool any_mask = out_of_context || sliding_evicted;
        qk_max = any_mask ? qk_max : max(qk_max, qk);
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

  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = exp(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(&red_smem[NUM_WARPS], exp_sum,
                                                 simd_tid, simd_lid);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (USE_PARTITIONING && thread_idx == 0) {
    device float *max_logits_ptr =
        max_logits + seq_idx * num_heads * max_num_partitions +
        head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    device float *exp_sums_ptr = exp_sums +
                                 seq_idx * num_heads * max_num_partitions +
                                 head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
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
    accs[i] = 0.f;
  }

  T zero_value = 0;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const int page_half = block_idx & 1;
    const int64_t physical_page =
        static_cast<int64_t>(block_table[block_idx >> 1]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    Float_L_vec logits_float_vec = *reinterpret_cast<threadgroup Float_L_vec *>(
        logits + token_idx - start_token_idx);
    {
      const int limit = int(context_len);
      const int lower = (sliding_window > 0 && limit > sliding_window)
                            ? limit - sliding_window
                            : 0;
      const device float *value_scales =
          v_scale + (physical_page * num_kv_heads + kv_head_idx) * PAGE +
          page_half * BLOCK_SIZE + physical_block_offset;
      thread float *p = reinterpret_cast<thread float *>(&logits_float_vec);
#pragma unroll
      for (int j = 0; j < V_VEC_SIZE; j++) {
        const bool valid = token_idx + j < limit && token_idx + j >= lower;
        p[j] = valid ? p[j] * value_scales[j] : 0.0f;
      }
    }
    from_float(logits_vec, logits_float_vec);

    const device CACHE_T *v_ptr = v_cache + physical_page * kv_block_stride +
                                  kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset =
            row_idx * PAGE + page_half * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;
        V_quant_vec v_quant_vec =
            *reinterpret_cast<const device V_quant_vec *>(v_ptr + offset);
        v_vec = char_to<V_vec, V_quant_vec>(v_quant_vec);

        if (block_idx == num_context_blocks - 1) {
          thread T *v_vec_ptr = reinterpret_cast<thread T *>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] =
                token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
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
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_idx < mid) {
      const threadgroup float *src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (warp_idx == 0) {
    using Out_T = typename PagedOut<T, CACHE_T, USE_PARTITIONING>::Type;
    device Out_T *out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        if constexpr (!USE_PARTITIONING) {
          *(out_ptr + row_idx) =
              Out_T(exp_sum > 0.0f ? accs[i] / exp_sum : 0.0f);
        } else {
          *(out_ptr + row_idx) = Out_T(accs[i]);
        }
      }
    }
  }
  (void)alibi_slopes;
}

template [[host_name("paged_attention_grouped_int8_hs256_striped_ktm")]]
[[kernel]] void paged_attention_grouped_int8_ktm_striped<256>(
    device float *exp_sums [[buffer(0)]],
    device float *max_logits [[buffer(1)]],
    device float *tmp_out [[buffer(2)]],
    device const bfloat16_t *q [[buffer(3)]],
    device const char *k_cache [[buffer(4)]],
    device const char *v_cache [[buffer(5)]],
    const device float *__restrict__ k_scale [[buffer(6)]],
    const device float *__restrict__ v_scale [[buffer(7)]],
    const constant int &num_kv_heads [[buffer(8)]],
    const constant float &scale [[buffer(9)]],
    const constant float &softcapping [[buffer(10)]],
    device const uint32_t *block_tables [[buffer(11)]],
    device const uint32_t *context_lens [[buffer(12)]],
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    device const float *alibi_slopes [[buffer(14)]],
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_block_stride [[buffer(16)]],
    const constant int &kv_head_stride [[buffer(17)]],
    const constant int &sliding_window [[buffer(18)]],
    const device float *score_scales [[buffer(21)]],
    const constant int &has_score_scales [[buffer(22)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_ktm(name, out_type, partition_size)       \
  template [[host_name(name)]] [[kernel]] void                                \
  paged_attention_ktm<256, partition_size>(                                   \
      device float *exp_sums [[buffer(0)]],                                   \
      device float *max_logits [[buffer(1)]],                                 \
      device out_type *out [[buffer(2)]],                                     \
      device const bfloat16_t *q [[buffer(3)]],                               \
      device const char *k_cache [[buffer(4)]],                               \
      device const char *v_cache [[buffer(5)]],                               \
      const device float *__restrict__ k_scale [[buffer(6)]],                 \
      const device float *__restrict__ v_scale [[buffer(7)]],                 \
      const constant int &num_kv_heads [[buffer(8)]],                         \
      const constant float &scale [[buffer(9)]],                              \
      const constant float &softcapping [[buffer(10)]],                       \
      device const uint32_t *block_tables [[buffer(11)]],                     \
      device const uint32_t *context_lens [[buffer(12)]],                     \
      const constant int &max_num_blocks_per_seq [[buffer(13)]],              \
      device const float *alibi_slopes [[buffer(14)]],                        \
      const constant int &q_stride [[buffer(15)]],                            \
      const constant int &kv_block_stride [[buffer(16)]],                     \
      const constant int &kv_head_stride [[buffer(17)]],                      \
      const constant int &sliding_window [[buffer(18)]],                      \
      const device float *score_scales [[buffer(21)]],                        \
      const constant int &has_score_scales [[buffer(22)]],                    \
      threadgroup char *shared_mem [[threadgroup(0)]],                        \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],    \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                  \
      uint3 thread_position_in_threadgroup                                    \
          [[thread_position_in_threadgroup]],                                 \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]);

instantiate_paged_attention_ktm(
    "paged_attention_bfloat16_t_cache_char_hs256_nt256_nsl32_ps0_ktm",
    bfloat16_t, 0);
instantiate_paged_attention_ktm(
    "paged_attention_bfloat16_t_cache_char_hs256_nt256_nsl32_ps512_ktm",
    float, 512);
