R"METAL(
constexpr int D_K = D_V + D_ROPE;
constexpr int V_PER_LANE = D_V / 32;
constexpr int R_PER_LANE = D_ROPE / 32;
constexpr int K_PER_LANE = V_PER_LANE + R_PER_LANE;
constexpr int HALF = D_K / 2;
constexpr int K_LANES = HALF / V_PER_LANE;
constexpr float MAX_FLOOR = -1e30f;

constexpr int TG_Q_ELEMS = HEADS * K_PER_LANE * 32;
constexpr int TG_O_ELEMS = BN * 32 * 4;
constexpr int TG_ELEMS = (TG_Q_ELEMS > TG_O_ELEMS) ? TG_Q_ELEMS : TG_O_ELEMS;

const int S = q_shape[0];
const int H = q_shape[1];
const int KMAX = idx_shape[1];
const int N_KEYS = k0_shape[0];
const int n_splits = split_cfg[0];
const int k_split = split_cfg[1];

const int y_id = int(threadgroup_position_in_grid.y);
const int head_group = y_id / n_splits;
const int sp = y_id - head_group * n_splits;
const int s = int(threadgroup_position_in_grid.z);
const int first_head = head_group * HEADS;
if (s >= S || first_head >= H) {
  return;
}

const uint tid = thread_index_in_threadgroup;
const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;

threadgroup float tg_scratch[TG_ELEMS];
threadgroup float max_scores[BN];
threadgroup float sum_exp_scores[BN];
threadgroup float sg_factors[BN];

const float sc = scale[0];
for (int iq = int(tid); iq < HEADS * D_K; iq += BN * 32) {
  const int h = iq / D_K;
  const int c = iq - h * D_K;
  const size_t q_off =
      (size_t(s) * size_t(H) + size_t(first_head + h)) * size_t(D_K) + size_t(c);
  const float val = q[q_off];
  int lane_dst;
  int loc_dst;
  if (c < D_V) {
    lane_dst = c / V_PER_LANE;
    loc_dst = c - lane_dst * V_PER_LANE;
  } else {
    const int r = c - D_V;
    lane_dst = r / R_PER_LANE;
    loc_dst = V_PER_LANE + (r - lane_dst * R_PER_LANE);
  }
  tg_scratch[(h * K_PER_LANE + loc_dst) * 32 + lane_dst] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float o[HEADS][V_PER_LANE];
float max_score[HEADS];
float sum_exp_score[HEADS];
for (int h = 0; h < HEADS; ++h) {
  for (int j = 0; j < V_PER_LANE; ++j) {
    o[h][j] = 0.0f;
  }
  max_score[h] = MAX_FLOOR;
  sum_exp_score[h] = 0.0f;
}

const int k_start = sp * k_split;
const int k_end =
    (sp + 1 < n_splits) ? metal::min(k_start + k_split, KMAX) : KMAX;

const size_t idx_base = size_t(s) * size_t(KMAX);
for (int k_idx = k_start + int(simd_gid); k_idx < k_end; k_idx += BN) {
  const int slot = idx[idx_base + size_t(k_idx)];
  // Skip unlisted or out-of-range slots before any device load so a non-finite
  // row at slot 0 cannot taint the accumulator.
  if (slot < 0 || slot >= N_KEYS) {
    continue;
  }

  float kreg[V_PER_LANE];
  float rreg[R_PER_LANE];
  if constexpr (SPLIT) {
    const size_t base = size_t(slot) * size_t(HALF);
    const device half4* kp = (int(simd_lid) < K_LANES)
        ? reinterpret_cast<const device half4*>(
              k0 + base + size_t(simd_lid) * size_t(V_PER_LANE))
        : reinterpret_cast<const device half4*>(
              k1 + base + size_t(int(simd_lid) - K_LANES) * size_t(V_PER_LANE));
#pragma unroll
    for (int g = 0; g < V_PER_LANE / 4; ++g) {
      const float4 vf = float4(kp[g]);
      kreg[g * 4 + 0] = vf.x;
      kreg[g * 4 + 1] = vf.y;
      kreg[g * 4 + 2] = vf.z;
      kreg[g * 4 + 3] = vf.w;
    }
    const size_t r_off =
        base + size_t(D_V - HALF) + size_t(simd_lid) * size_t(R_PER_LANE);
    const float2 rf = float2(*reinterpret_cast<const device half2*>(k1 + r_off));
    rreg[0] = rf.x;
    rreg[1] = rf.y;
  } else {
    const size_t base = size_t(slot) * size_t(D_K);
    const device half4* kp = reinterpret_cast<const device half4*>(
        k0 + base + size_t(simd_lid) * size_t(V_PER_LANE));
#pragma unroll
    for (int g = 0; g < V_PER_LANE / 4; ++g) {
      const float4 vf = float4(kp[g]);
      kreg[g * 4 + 0] = vf.x;
      kreg[g * 4 + 1] = vf.y;
      kreg[g * 4 + 2] = vf.z;
      kreg[g * 4 + 3] = vf.w;
    }
    const size_t r_off =
        base + size_t(D_V) + size_t(simd_lid) * size_t(R_PER_LANE);
    const float2 rf = float2(*reinterpret_cast<const device half2*>(k0 + r_off));
    rreg[0] = rf.x;
    rreg[1] = rf.y;
  }

  for (int h = 0; h < HEADS; ++h) {
    float score = 0.0f;
    const threadgroup float* qh = tg_scratch + h * K_PER_LANE * 32 + int(simd_lid);
    for (int j = 0; j < V_PER_LANE; ++j) {
      score = metal::fma(qh[j * 32], kreg[j], score);
    }
    for (int j = 0; j < R_PER_LANE; ++j) {
      score = metal::fma(qh[(V_PER_LANE + j) * 32], rreg[j], score);
    }
    score = simd_sum(score) * sc;

    if (score > max_score[h]) {
      const float factor = metal::exp(max_score[h] - score);
      max_score[h] = score;
      sum_exp_score[h] = metal::fma(sum_exp_score[h], factor, 1.0f);
      for (int j = 0; j < V_PER_LANE; ++j) {
        o[h][j] = metal::fma(o[h][j], factor, kreg[j]);
      }
    } else {
      const float exp_score = metal::exp(score - max_score[h]);
      sum_exp_score[h] += exp_score;
      for (int j = 0; j < V_PER_LANE; ++j) {
        o[h][j] = metal::fma(exp_score, kreg[j], o[h][j]);
      }
    }
  }
}

for (int h = 0; h < HEADS; ++h) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score[h];
    sum_exp_scores[simd_gid] = sum_exp_score[h];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float lane_max = (int(simd_lid) < BN) ? max_scores[simd_lid] : MAX_FLOOR;
  const float global_max = simd_max(lane_max);
  const float lane_factor =
      (int(simd_lid) < BN) ? metal::exp(lane_max - global_max) : 0.0f;
  const float lane_sum = (int(simd_lid) < BN) ? sum_exp_scores[simd_lid] : 0.0f;
  const float total_sum = simd_sum(lane_sum * lane_factor);

  if (simd_gid == 0 && int(simd_lid) < BN) {
    sg_factors[simd_lid] = lane_factor;
  }

  const size_t mp_idx =
      (size_t(s) * size_t(H) + size_t(first_head + h)) * size_t(n_splits) +
      size_t(sp);
  if (tid == 0) {
    max_part[mp_idx] = global_max;
    sum_part[mp_idx] = total_sum;
  }

  device float* out_h = o_part + mp_idx * size_t(D_V);

  for (int base_j = 0; base_j < V_PER_LANE; base_j += 4) {
    if (base_j > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int k = 0; k < 4; ++k) {
      tg_scratch[(int(simd_gid) * 4 + k) * 32 + int(simd_lid)] = o[h][base_j + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 128) {
      const int k = int(tid >> 5);
      const int orig_lane = int(tid & 31);
      float acc = 0.0f;
      for (int sg = 0; sg < BN; ++sg) {
        acc = metal::fma(
            tg_scratch[(sg * 4 + k) * 32 + orig_lane], sg_factors[sg], acc);
      }
      out_h[orig_lane * V_PER_LANE + base_j + k] =
          (total_sum > 0.0f) ? acc : 0.0f;
    }
  }
}
)METAL"
