// Vector SDPA over two logically adjacent K/V segments (prefix cache rows,
// then the rows the verify block just produced). build.rs prebuilds the BF16
// D=256 instantiations at the end of this file into paged_attn.metallib.
//
// The score traversal and online-softmax update order equal MLX's
// sdpa_vector / sdpa_vector_2pass_1 (sdpa_vector.h); only the address
// calculation chooses prefix or new K/V. The two-pass kernels write partials
// in sdpa_vector_2pass_2's layout, and the caller reduces them with MLX's own
// kernel. Changing any statement order here breaks bit-identity with
// concatenated K/V through MLX's vector SDPA.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

constant bool do_causal [[function_constant(22)]];
constant int blocks [[function_constant(26)]];
constant int verify_gqa [[function_constant(27)]];
constant int verify_rows [[function_constant(28)]];

constant constexpr float kFiniteMin = -metal::numeric_limits<float>::max();

template <typename T, int D, int V = D>
[[kernel]] void segmented_sdpa_one_pass(
    const device T* queries [[buffer(0)]],
    const device T* prefix_keys [[buffer(1)]],
    const device T* prefix_values [[buffer(2)]],
    const device T* new_keys [[buffer(3)]],
    const device T* new_values [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& gqa_factor [[buffer(6)]],
    const constant int& prefix_n [[buffer(7)]],
    const constant int& new_n [[buffer(8)]],
    const constant long* strides [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const constant int& num_q_heads [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  // q batch/head/sequence; then prefix K/V and new K/V batch/head/sequence.
  const long q_batch_stride = strides[0];
  const long q_head_stride = strides[1];
  const long q_seq_stride = strides[2];
  const long pk_batch_stride = strides[3];
  const long pk_head_stride = strides[4];
  const long pk_seq_stride = strides[5];
  const long pv_batch_stride = strides[6];
  const long pv_head_stride = strides[7];
  const long pv_seq_stride = strides[8];
  const long nk_batch_stride = strides[9];
  const long nk_head_stride = strides[10];
  const long nk_seq_stride = strides[11];
  const long nv_batch_stride = strides[12];
  const long nv_head_stride = strides[13];
  const long nv_seq_stride = strides[14];

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];
  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int batch_idx = q_batch_head_idx / num_q_heads;
  const int q_head_idx = q_batch_head_idx - batch_idx * num_q_heads;
  const int kv_head_idx = q_head_idx / gqa_factor;
  const int n = prefix_n + new_n;
  queries += batch_idx * q_batch_stride + q_head_idx * q_head_stride +
      q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  out += (q_batch_head_idx * tpg.y + q_seq_idx) * V + simd_gid * v_per_thread;

  for (int j = 0; j < qk_per_thread; ++j) {
    q[j] = static_cast<U>(scale) * queries[j];
  }
  for (int j = 0; j < v_per_thread; ++j) {
    o[j] = 0;
  }

  U max_score = kFiniteMin;
  U sum_exp_score = 0;
  // Every prefix row precedes every query. Traverse that segment without
  // a per-row segment selection or causal branch, then continue the same
  // strided score sequence through the visible new rows.
  int i = simd_gid;
  const device T* key = prefix_keys + batch_idx * pk_batch_stride +
      kv_head_idx * pk_head_stride + i * pk_seq_stride +
      simd_lid * qk_per_thread;
  const device T* value = prefix_values + batch_idx * pv_batch_stride +
      kv_head_idx * pv_head_stride + i * pv_seq_stride +
      simd_lid * v_per_thread;
  // The host guarantees the row steps fit in 32 bits (faster than 64-bit).
  const int key_step = BN * int(pk_seq_stride);
  const int value_step = BN * int(pv_seq_stride);
  for (; i < prefix_n; i += BN) {
    for (int j = 0; j < qk_per_thread; ++j) {
      k[j] = key[j];
    }
    U score = 0;
    for (int j = 0; j < qk_per_thread; ++j) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; ++j) {
      o[j] = o[j] * factor + exp_score * value[j];
    }
    key += key_step;
    value += value_step;
  }
  const device T* nk = new_keys + batch_idx * nk_batch_stride +
      kv_head_idx * nk_head_stride + simd_lid * qk_per_thread;
  const device T* nv = new_values + batch_idx * nv_batch_stride +
      kv_head_idx * nv_head_stride + simd_lid * v_per_thread;
  const int visible_n = do_causal ? n - int(tpg.y) + q_seq_idx + 1 : n;
  for (; i < visible_n; i += BN) {
    const device T* key = nk + (i - prefix_n) * nk_seq_stride;
    const device T* value = nv + (i - prefix_n) * nv_seq_stride;
    for (int j = 0; j < qk_per_thread; ++j) {
      k[j] = key[j];
    }
    U score = 0;
    for (int j = 0; j < qk_per_thread; ++j) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; ++j) {
      o[j] = o[j] * factor + exp_score * value[j];
    }
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int j = 0; j < v_per_thread; ++j) {
    outputs[simd_lid * BD + simd_gid] = o[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[j] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[j] = sum_exp_score == 0 ? o[j] : (o[j] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int j = 0; j < v_per_thread; ++j) {
      out[j] = static_cast<T>(o[j]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void segmented_sdpa_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* prefix_keys [[buffer(1)]],
    const device T* prefix_values [[buffer(2)]],
    const device T* new_keys [[buffer(3)]],
    const device T* new_values [[buffer(4)]],
    device T* out [[buffer(5)]],
    device float* sums [[buffer(6)]],
    device float* maxs [[buffer(7)]],
    const constant int& prefix_n [[buffer(8)]],
    const constant int& new_n [[buffer(9)]],
    const constant long* strides [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_q_heads = tpg.x * gqa_factor;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int n = prefix_n + new_n;

  queries += batch_idx * strides[0] + q_head_idx * strides[1] +
      q_seq_idx * strides[2] + simd_lid * qk_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  for (int j = 0; j < qk_per_thread; ++j) {
    q[j] = static_cast<U>(scale) * queries[j];
  }
  U max_score = kFiniteMin;
  U sum_exp_score = 0;
  int i = block_idx;
  const device T* key = prefix_keys + batch_idx * strides[3] +
      kv_head_idx * strides[4] + i * strides[5] + simd_lid * qk_per_thread;
  const device T* value = prefix_values + batch_idx * strides[6] +
      kv_head_idx * strides[7] + i * strides[8] + simd_lid * v_per_thread;
  const int key_step = blocks * int(strides[5]);
  const int value_step = blocks * int(strides[8]);
  for (; i < prefix_n; i += blocks) {
    U score = 0;
    for (int j = 0; j < qk_per_thread; ++j) {
      score += q[j] * key[j];
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; ++j) {
      o[j] = o[j] * factor + exp_score * value[j];
    }
    key += key_step;
    value += value_step;
  }
  const device T* nk = new_keys + batch_idx * strides[9] +
      kv_head_idx * strides[10] + simd_lid * qk_per_thread;
  const device T* nv = new_values + batch_idx * strides[12] +
      kv_head_idx * strides[13] + simd_lid * v_per_thread;
  const int visible_n = do_causal ? n - q_seq_len + q_seq_idx + 1 : n;
  for (; i < visible_n; i += blocks) {
    const device T* key = nk + (i - prefix_n) * strides[11];
    const device T* value = nv + (i - prefix_n) * strides[14];
    U score = 0;
    for (int j = 0; j < qk_per_thread; ++j) {
      score += q[j] * key[j];
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; ++j) {
      o[j] = o[j] * factor + exp_score * value[j];
    }
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int j = 0; j < v_per_thread; ++j) {
    out[j] = static_cast<T>(o[j]);
  }
}

template <typename U, typename T, int QK, int VP>
METAL_FUNC void sdpa_segmented_verify_update(
    thread const U* q,
    thread U* o,
    thread U& max_score,
    thread U& sum_exp_score,
    thread const T* key,
    thread const T* value) {
  U score = 0;
  for (int j = 0; j < QK; ++j) {
    score += q[j] * key[j];
  }
  score = simd_sum(score);
  U new_max = max(max_score, score);
  U factor = fast::exp(max_score - new_max);
  U exp_score = fast::exp(score - new_max);
  max_score = new_max;
  sum_exp_score = sum_exp_score * factor + exp_score;
  for (int j = 0; j < VP; ++j) {
    o[j] = o[j] * factor + exp_score * value[j];
  }
}

// Causal first pass for a whole verify block: one threadgroup serves every
// (query head, row) pair of one KV head, and each simdgroup applies each K/V
// row it loads to two pairs. Per pair the key order, statements and causal
// limit equal segmented_sdpa_2pass_1 with q_seq_len = verify_rows, so
// the partials are bit-identical to that kernel at the same `blocks`.
template <typename T, int D, int V = D>
[[kernel]] void segmented_sdpa_verify_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* prefix_keys [[buffer(1)]],
    const device T* prefix_values [[buffer(2)]],
    const device T* new_keys [[buffer(3)]],
    const device T* new_values [[buffer(4)]],
    device T* out [[buffer(5)]],
    device float* sums [[buffer(6)]],
    device float* maxs [[buffer(7)]],
    const constant int& prefix_n [[buffer(8)]],
    const constant int& new_n [[buffer(9)]],
    const constant long* strides [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Four pairs per simdgroup spill registers.
  constexpr int PAIRS = 2;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int num_q_heads = tpg.x * verify_gqa;
  const int simdgroups = verify_gqa * verify_rows / PAIRS;
  const int n = prefix_n + new_n;

  thread U q[PAIRS][qk_per_thread];
  thread U o[PAIRS][v_per_thread];
  thread U max_score[PAIRS];
  thread U sum_exp_score[PAIRS];
  thread int visible_n[PAIRS];
  thread int o_offset[PAIRS];
  for (int k = 0; k < PAIRS; ++k) {
    const int pair = int(simd_gid) + k * simdgroups;
    const int q_seq_idx = pair / verify_gqa;
    const int q_head_idx =
        verify_gqa * kv_head_idx + (pair - q_seq_idx * verify_gqa);
    o_offset[k] =
        (batch_idx * num_q_heads + q_head_idx) * verify_rows + q_seq_idx;
    const device T* query = queries + batch_idx * strides[0] +
        q_head_idx * strides[1] + q_seq_idx * strides[2] +
        simd_lid * qk_per_thread;
    for (int j = 0; j < qk_per_thread; ++j) {
      q[k][j] = static_cast<U>(scale) * query[j];
    }
    for (int j = 0; j < v_per_thread; ++j) {
      o[k][j] = 0;
    }
    max_score[k] = kFiniteMin;
    sum_exp_score[k] = 0;
    visible_n[k] = n - verify_rows + q_seq_idx + 1;
  }

  int i = block_idx;
  const device T* key = prefix_keys + batch_idx * strides[3] +
      kv_head_idx * strides[4] + i * strides[5] + simd_lid * qk_per_thread;
  const device T* value = prefix_values + batch_idx * strides[6] +
      kv_head_idx * strides[7] + i * strides[8] + simd_lid * v_per_thread;
  const int key_step = blocks * int(strides[5]);
  const int value_step = blocks * int(strides[8]);
  for (; i < prefix_n; i += blocks) {
    thread T k_row[qk_per_thread];
    thread T v_row[v_per_thread];
    for (int j = 0; j < qk_per_thread; ++j) {
      k_row[j] = key[j];
    }
    for (int j = 0; j < v_per_thread; ++j) {
      v_row[j] = value[j];
    }
    for (int k = 0; k < PAIRS; ++k) {
      sdpa_segmented_verify_update<U, T, qk_per_thread, v_per_thread>(
          q[k], o[k], max_score[k], sum_exp_score[k], k_row, v_row);
    }
    key += key_step;
    value += value_step;
  }
  const device T* nk = new_keys + batch_idx * strides[9] +
      kv_head_idx * strides[10] + simd_lid * qk_per_thread;
  const device T* nv = new_values + batch_idx * strides[12] +
      kv_head_idx * strides[13] + simd_lid * v_per_thread;
  for (; i < n; i += blocks) {
    const device T* new_key = nk + (i - prefix_n) * strides[11];
    const device T* new_value = nv + (i - prefix_n) * strides[14];
    thread T k_row[qk_per_thread];
    thread T v_row[v_per_thread];
    for (int j = 0; j < qk_per_thread; ++j) {
      k_row[j] = new_key[j];
    }
    for (int j = 0; j < v_per_thread; ++j) {
      v_row[j] = new_value[j];
    }
    for (int k = 0; k < PAIRS; ++k) {
      if (i < visible_n[k]) {
        sdpa_segmented_verify_update<U, T, qk_per_thread, v_per_thread>(
            q[k], o[k], max_score[k], sum_exp_score[k], k_row, v_row);
      }
    }
  }

  for (int k = 0; k < PAIRS; ++k) {
    device T* partial = out + o_offset[k] * blocks * V + block_idx * V +
        simd_lid * v_per_thread;
    if (simd_lid == 0) {
      sums[o_offset[k] * blocks + block_idx] = sum_exp_score[k];
      maxs[o_offset[k] * blocks + block_idx] = max_score[k];
    }
    for (int j = 0; j < v_per_thread; ++j) {
      partial[j] = static_cast<T>(o[k][j]);
    }
  }
}

// Every kernel mlx_segmented_sdpa.cpp can request. The host names must match
// its kernel_name() character for character; bridge_metallib_names
// checks both directions. The function constants above are specialized when
// the dispatcher builds each pipeline.
template [[host_name("mlx_node_sdpa_segmented_bf16_256")]] [[kernel]]
decltype(segmented_sdpa_one_pass<bfloat, 256, 256>)
    segmented_sdpa_one_pass<bfloat, 256, 256>;
template [[host_name("mlx_node_sdpa_segmented_2pass_1_bf16_256")]] [[kernel]]
decltype(segmented_sdpa_2pass_1<bfloat, 256, 256>)
    segmented_sdpa_2pass_1<bfloat, 256, 256>;
template [[host_name("mlx_node_sdpa_segmented_verify_2pass_1_bf16_256")]] [[kernel]]
decltype(segmented_sdpa_verify_2pass_1<bfloat, 256, 256>)
    segmented_sdpa_verify_2pass_1<bfloat, 256, 256>;
