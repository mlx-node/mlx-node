// Vector SDPA over two logically adjacent K/V segments (prefix cache rows,
// then the rows the verify block just produced). build.rs prebuilds the BF16
// D=256 instantiations at the end of this file into paged_attn.metallib.
//
// The score traversal and online-softmax update order of the vector kernels
// equal MLX's sdpa_vector / sdpa_vector_2pass_1 (sdpa_vector.h); only the
// address calculation chooses prefix or new K/V. The two-pass kernels write
// partials in sdpa_vector_2pass_2's layout, and the caller reduces them with
// MLX's own kernel. Changing any statement order there breaks bit-identity
// with concatenated K/V through MLX's vector SDPA. The simdgroup-matrix
// verify kernel at the end is the exception: it writes the same partial
// layout but reduces in tile order (fp32, not bit-identical). Its tensor-op
// sibling for gen-17+ GPUs is sdpa_segmented_nax.metal.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

constant bool do_causal [[function_constant(22)]];
constant int blocks [[function_constant(26)]];
constant int verify_gqa [[function_constant(27)]];
constant int verify_rows [[function_constant(28)]];
// Keys per K/V tile of the simdgroup-matrix verify kernel (16 or 32).
constant int tile_n [[function_constant(29)]];

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

// 8x8 simdgroup_matrix fragment coordinates, as MLX steel's
// BaseMMAFrag::get_coord: a lane holds row `fm`, columns `fn` and `fn + 1`.
// The four lanes of one row are lane, lane ^ 1, lane ^ 8 and lane ^ 9.
struct FragCoord {
  int fm;
  int fn;
};

METAL_FUNC FragCoord frag_coord(uint simd_lid) {
  const int qid = int(simd_lid) / 4;
  return {(qid & 4) + ((int(simd_lid) / 2) % 4),
          (qid & 2) * 2 + (int(simd_lid) % 2) * 2};
}

template <bool B>
struct BoolTag {
  static constant constexpr bool value = B;
};

// One K or V tile (rows [k0, k0 + 8 * frags) of the prefix / new segment
// pair) into the [tile_n][LD] tile, 16 bytes per thread per step. MASKED
// selects the segment per row and zero-fills rows at or past `kend`; the
// unmasked form reads prefix rows only. Device latency is hidden by the
// other resident threadgroups, not by register prefetch (measured slower).
template <typename T, int D, int LD>
struct SdpaTileLoader {
  static constant constexpr int VEC = 16 / sizeof(T);
  static constant constexpr int VPR = D / VEC;

  const device T* prefix;
  long prefix_seq_stride;
  const device T* fresh;
  long fresh_seq_stride;
  int prefix_n;
  int kend;
  uint lid;
  uint threads;

  template <bool MASKED>
  METAL_FUNC void load(threadgroup T* tile, int k0, int frags) const {
    const int total = frags * 8 * VPR;
    for (int v = int(lid); v < total; v += int(threads)) {
      const int r = v / VPR;
      const int c = (v - r * VPR) * VEC;
      const int key = k0 + r;
      uint4 bits = uint4(0);
      if (!MASKED || key < kend) {
        const device T* src = (!MASKED || key < prefix_n)
            ? prefix + long(key) * prefix_seq_stride
            : fresh + long(key - prefix_n) * fresh_seq_stride;
        bits = *reinterpret_cast<const device uint4*>(src + c);
      }
      *reinterpret_cast<threadgroup uint4*>(tile + r * LD + c) = bits;
    }
  }
};

// Fragments live in registers as 2-element vectors (MLX steel's MMATile);
// simdgroup_matrix values exist only inside the MMA call.
template <typename T>
METAL_FUNC void sdpa_tile_mma(
    thread float2& acc,
    thread const vec<T, 2>& a,
    thread const simdgroup_matrix<T, 8, 8>& b) {
  simdgroup_matrix<T, 8, 8> am;
  simdgroup_matrix<float, 8, 8> cm;
  reinterpret_cast<thread vec<T, 2>&>(am.thread_elements()) = a;
  reinterpret_cast<thread float2&>(cm.thread_elements()) = acc;
  simdgroup_multiply_accumulate(cm, am, b, cm);
  acc = reinterpret_cast<thread float2&>(cm.thread_elements());
}

// First pass of a verify block with simdgroup-matrix MMAs. One threadgroup
// serves all M = gqa * rows (head, row) queries of one KV head over the keys
// of one contiguous partition, 2 simdgroups per 8 queries (one per half of
// D, as MLX steel attention at D=256). Per tile_n keys: K tile -> S = Q K^T
// (BF16 MMA, fp32 accumulate, halves summed through threadgroup memory),
// online softmax per query, P (BF16) V tile -> O. Only tiles past the last
// full prefix tile select segments per row, mask (causal and `kend`) and
// skip the key fragments past `kend`.
// Threadgroup memory, set by the host: the K/V tile (tile_n * (D + 8) BF16)
// followed by the score exchange (simdgroups * 32 * tile_n / 4 floats).
template <typename T, int D>
[[kernel]] void segmented_sdpa_verify_tile_2pass_1(
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
    const constant int& gqa [[buffer(12)]],
    const constant int& rows [[buffer(13)]],
    const constant int& partitions [[buffer(14)]],
    threadgroup T* kv_tile [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int WN = 2;
  constexpr int DH = D / WN;
  constexpr int LD = D + 16 / sizeof(T);
  constexpr int FD = DH / 8;
  // Score fragments per lane at the largest tile (32 keys).
  constexpr int MAX_FT = 4;
  typedef simdgroup_matrix<T, 8, 8> mat_in;
  typedef vec<T, 2> frag_in;
  const int FT = tile_n / 8;

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int threads = tptg.x;
  const int row_block = int(simd_gid) / WN;
  const int d_half = int(simd_gid) % WN;
  const int num_q_heads = tpg.x * gqa;
  const int n = prefix_n + new_n;
  const FragCoord fc = frag_coord(simd_lid);

  // This lane's query: fragment row fm of the simdgroup's 8-query block.
  const int m = row_block * 8 + fc.fm;
  const int q_seq_idx = m / gqa;
  const int q_head_idx = gqa * kv_head_idx + (m - q_seq_idx * gqa);
  const int visible_n = n - rows + q_seq_idx + 1;
  const int o_offset =
      (batch_idx * num_q_heads + q_head_idx) * rows + q_seq_idx;

  const int chunk = (n + partitions - 1) / partitions;
  const int kstart = block_idx * chunk;
  const int kend = min(n, kstart + chunk);
  const int limit = min(kend, visible_n);

  // Q fragments are re-read per tile (L1 resident): keeping them in
  // registers lowered occupancy and measured slower.
  const device frag_in* q = reinterpret_cast<const device frag_in*>(
      queries + batch_idx * strides[0] + q_head_idx * strides[1] +
      q_seq_idx * strides[2] + d_half * DH + fc.fn);
  float2 o[FD];
#pragma clang loop unroll(full)
  for (int k = 0; k < FD; ++k) {
    o[k] = float2(0);
  }
  // Scores in log2 units: exp2 of a kFiniteMin-masked score is 0.
  const float scale2 = scale * M_LOG2E_F;
  float max_score = kFiniteMin;
  float sum_exp_score = 0;

  const SdpaTileLoader<T, D, LD> keys{
      prefix_keys + batch_idx * strides[3] + kv_head_idx * strides[4],
      strides[5],
      new_keys + batch_idx * strides[9] + kv_head_idx * strides[10],
      strides[11],
      prefix_n,
      kend,
      lid,
      uint(threads)};
  const SdpaTileLoader<T, D, LD> values{
      prefix_values + batch_idx * strides[6] + kv_head_idx * strides[7],
      strides[8],
      new_values + batch_idx * strides[12] + kv_head_idx * strides[13],
      strides[14],
      prefix_n,
      kend,
      lid,
      uint(threads)};

  threadgroup float* xchg =
      reinterpret_cast<threadgroup float*>(kv_tile + tile_n * LD);
  const int xchg_slot = int(simd_gid) * 32 * 2 * FT;
  const int peer_slot = (int(simd_gid) ^ 1) * 32 * 2 * FT;

  // Full tiles inside the prefix need no segment select or mask.
  const int prefix_end = min(kend, prefix_n);
  const int full_end = prefix_end > kstart
      ? kstart + ((prefix_end - kstart) / tile_n) * tile_n
      : kstart;

  auto process = [&](int k0, auto tag) {
    constexpr bool masked = decltype(tag)::value;
    const int frags = masked ? min(FT, (kend - k0 + 7) / 8) : FT;
    // The previous tile's PV reads are done.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    keys.template load<masked>(kv_tile, k0, frags);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 s[MAX_FT];
#pragma clang loop unroll(full)
    for (int f = 0; f < MAX_FT; ++f) {
      s[f] = float2(0);
      if (f < frags) {
#pragma clang loop unroll(full)
        for (int k = 0; k < FD; ++k) {
          mat_in kf;
          simdgroup_load(kf, kv_tile + f * 8 * LD + d_half * DH + k * 8, LD,
                         ulong2(0, 0), true);
          const frag_in qk = q[4 * k];
          sdpa_tile_mma(s[f], qk, kf);
        }
      }
    }

    // Sum the two D halves of S. The barrier also ends every simdgroup's K
    // reads, so the V rows can land in the tile afterwards.
#pragma clang loop unroll(full)
    for (int f = 0; f < MAX_FT; ++f) {
      if (f < frags) {
        xchg[xchg_slot + (2 * f) * 32 + simd_lid] = s[f].x;
        xchg[xchg_slot + (2 * f + 1) * 32 + simd_lid] = s[f].y;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values.template load<masked>(kv_tile, k0, frags);
    float tile_max = kFiniteMin;
#pragma clang loop unroll(full)
    for (int f = 0; f < MAX_FT; ++f) {
      if (f < frags) {
        float2 v = s[f] +
            float2(xchg[peer_slot + (2 * f) * 32 + simd_lid],
                   xchg[peer_slot + (2 * f + 1) * 32 + simd_lid]);
        v *= scale2;
        if (masked) {
          const int key = k0 + 8 * f + fc.fn;
          v.x = key >= limit ? kFiniteMin : v.x;
          v.y = key + 1 >= limit ? kFiniteMin : v.y;
        }
        s[f] = v;
        tile_max = max(tile_max, max(v.x, v.y));
      }
    }
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(8)));
    const float new_max = max(max_score, tile_max);
    // A query with no visible key yet: keep exp2 arguments finite.
    const float ref = new_max == kFiniteMin ? 0.0f : new_max;
    const float factor = fast::exp2(max_score - ref);
    // P in BF16 for the MMA; the row sum uses the same rounded weights.
    frag_in p[MAX_FT];
    float row_sum = 0;
#pragma clang loop unroll(full)
    for (int f = 0; f < MAX_FT; ++f) {
      p[f] = frag_in(0);
      if (f < frags) {
        p[f] = frag_in(fast::exp2(s[f] - ref));
        row_sum += float(p[f].x) + float(p[f].y);
      }
    }
    row_sum += simd_shuffle_xor(row_sum, ushort(1));
    row_sum += simd_shuffle_xor(row_sum, ushort(8));
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + row_sum;
#pragma clang loop unroll(full)
    for (int k = 0; k < FD; ++k) {
      o[k] *= factor;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (int k = 0; k < FD; ++k) {
#pragma clang loop unroll(full)
      for (int f = 0; f < MAX_FT; ++f) {
        if (f < frags) {
          mat_in vf;
          simdgroup_load(vf, kv_tile + f * 8 * LD + d_half * DH + k * 8, LD);
          sdpa_tile_mma(o[k], p[f], vf);
        }
      }
    }
  };

  for (int k0 = kstart; k0 < full_end; k0 += tile_n) {
    process(k0, BoolTag<false>{});
  }
  for (int k0 = full_end; k0 < kend; k0 += tile_n) {
    process(k0, BoolTag<true>{});
  }

  device T* partial =
      out + o_offset * partitions * D + block_idx * D + d_half * DH + fc.fn;
#pragma clang loop unroll(full)
  for (int k = 0; k < FD; ++k) {
    *reinterpret_cast<device frag_in*>(partial + 8 * k) = frag_in(o[k]);
  }
  if (d_half == 0 && fc.fn == 0) {
    sums[o_offset * partitions + block_idx] = sum_exp_score;
    // Back to natural units for sdpa_vector_2pass_2's exp().
    maxs[o_offset * partitions + block_idx] =
        max_score == kFiniteMin ? kFiniteMin : max_score * M_LN2_F;
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
template [[host_name("mlx_node_sdpa_segmented_verify_tile_2pass_1_bf16_256")]] [[kernel]]
decltype(segmented_sdpa_verify_tile_2pass_1<bfloat, 256>)
    segmented_sdpa_verify_tile_2pass_1<bfloat, 256>;
