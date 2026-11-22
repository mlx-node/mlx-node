R"METAL(
constexpr int D_V = 512;
constexpr int D_ROPE = 64;
constexpr int D_K = D_V + D_ROPE;
constexpr int HALF_DK = D_K / 2;
constexpr int HALF_DK8 = HALF_DK / 8;
constexpr int kU = 8;
constexpr int HG = 2;
constexpr int HT = HG * kU;
constexpr int TR = 16;
constexpr int TK = TR / kU;
constexpr int WN = 8;
constexpr int NT = WN * 32;
constexpr int KV_STRIDE = HALF_DK + 8;
constexpr int DV_PER_SG = D_V / WN;
constexpr int TDh = DV_PER_SG / kU;
constexpr float MAX_FLOOR = -1e30f;

#define HY4_M3_MMA_MAT(ACC, A_VAL, B_MAT)                              \
  do {                                                                 \
    const half2 _a_val = (A_VAL);                                      \
    simdgroup_half8x8 _a_mat;                                          \
    simdgroup_float8x8 _c_mat;                                         \
    _a_mat.thread_elements()[0] = _a_val.x;                            \
    _a_mat.thread_elements()[1] = _a_val.y;                            \
    _c_mat.thread_elements()[0] = (ACC).x;                             \
    _c_mat.thread_elements()[1] = (ACC).y;                             \
    simdgroup_multiply_accumulate(_c_mat, _a_mat, (B_MAT), _c_mat);    \
    (ACC).x = _c_mat.thread_elements()[0];                             \
    (ACC).y = _c_mat.thread_elements()[1];                             \
  } while (0)

#define HY4_M3_MMA_REG(ACC, A_VAL, B_VAL)                              \
  do {                                                                 \
    const half2 _a_val = (A_VAL);                                      \
    const half2 _b_val = (B_VAL);                                      \
    simdgroup_half8x8 _a_mat;                                          \
    simdgroup_half8x8 _b_mat;                                          \
    simdgroup_float8x8 _c_mat;                                         \
    _a_mat.thread_elements()[0] = _a_val.x;                            \
    _a_mat.thread_elements()[1] = _a_val.y;                            \
    _b_mat.thread_elements()[0] = _b_val.x;                            \
    _b_mat.thread_elements()[1] = _b_val.y;                            \
    _c_mat.thread_elements()[0] = (ACC).x;                             \
    _c_mat.thread_elements()[1] = (ACC).y;                             \
    simdgroup_multiply_accumulate(_c_mat, _a_mat, _b_mat, _c_mat);     \
    (ACC).x = _c_mat.thread_elements()[0];                             \
    (ACC).y = _c_mat.thread_elements()[1];                             \
  } while (0)

const int S = q_shape[0];
const int H = q_shape[1];
const int KMAX = idx_shape[1];
const int N_KEYS = k0_shape[0];
const int n_splits = split_cfg[0];
const int k_split = split_cfg[1];

const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
const uint tid = simd_gid * 32u + simd_lid;
const int d_group = int(simd_gid);

const int y_id = int(threadgroup_position_in_grid.y);
const int head_group = SINGLE_SPLIT ? y_id : (y_id / n_splits);
const int sp = SINGLE_SPLIT ? 0 : (y_id - head_group * n_splits);
const int head_base = head_group * HT;
const int s = int(threadgroup_position_in_grid.z);
if (s >= S || head_base >= H) {
  return;
}

// 16 x 296 half (9,472 B) + 4-slot score tree buffer (4,096 B) + s_slots (64 B)
// = 13,632 B (13.3 KiB <= 16 KiB) for 2+ resident TGs per core.
alignas(16) threadgroup half s_kv[TR * KV_STRIDE];
alignas(16) threadgroup float s_xchg[4 * HG * TK * 2 * 32];
alignas(16) threadgroup int s_slots[TR];

const uint qid = simd_lid >> 2;
const int fm = int((qid & 4u) + ((simd_lid >> 1) & 3u));
const int fn = int(((qid & 2u) << 1) + ((simd_lid & 1u) << 1));

const int h_idx0 = head_base + fm;
const int h_idx1 = head_base + kU + fm;
const bool h_ok0 = (h_idx0 < H);
const bool h_ok1 = (h_idx1 < H);

const device float* q_row0 =
    q + (size_t(s) * size_t(H) + size_t(h_ok0 ? h_idx0 : 0)) * size_t(D_K);
const device float* q_row1 =
    q + (size_t(s) * size_t(H) + size_t(h_ok1 ? h_idx1 : 0)) * size_t(D_K);

// Each SG owns 64 V columns across HT=16 heads; SGs 4..7 also own 16 RoPE cols each.
const int v_base = d_group * DV_PER_SG;
half2 Q_v0[TDh];
half2 Q_v1[TDh];
#pragma unroll
for (short vi = 0; vi < TDh; ++vi) {
  const int col = v_base + int(vi) * kU + fn;
  Q_v0[vi] = h_ok0 ? half2(half(q_row0[col]), half(q_row0[col + 1])) : half2(0.0h);
  Q_v1[vi] = h_ok1 ? half2(half(q_row1[col]), half(q_row1[col + 1])) : half2(0.0h);
}

half2 Q_r0[2];
half2 Q_r1[2];
if (d_group >= 4) {
  const int r_base = D_V + (d_group - 4) * 16;
  #pragma unroll
  for (short ri = 0; ri < 2; ++ri) {
    const int col = r_base + int(ri) * kU + fn;
    Q_r0[ri] = h_ok0 ? half2(half(q_row0[col]), half(q_row0[col + 1])) : half2(0.0h);
    Q_r1[ri] = h_ok1 ? half2(half(q_row1[col]), half(q_row1[col + 1])) : half2(0.0h);
  }
}

float2 O_acc0[TDh];
float2 O_acc1[TDh];
#pragma unroll
for (short vi = 0; vi < TDh; ++vi) {
  O_acc0[vi] = float2(0.0f);
  O_acc1[vi] = float2(0.0f);
}

float max_score0 = MAX_FLOOR;
float max_score1 = MAX_FLOOR;
float sum_score0 = 0.0f;
float sum_score1 = 0.0f;

const float scale2 = scale[0] * M_LOG2E_F;
const int k_start = sp * k_split;
const int k_end =
    (sp + 1 < n_splits) ? metal::min(k_start + k_split, KMAX) : KMAX;
const size_t idx_base = size_t(s) * size_t(KMAX);

threadgroup float4* s_xchg4 = reinterpret_cast<threadgroup float4*>(s_xchg);
const int lid = int(simd_lid);

bool prev_skipped = false;

for (int rt = k_start; rt < k_end; rt += TR) {
  if (prev_skipped) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid < uint(TR)) {
    const int kp = rt + int(tid);
    const int raw = (kp < k_end) ? idx[idx_base + size_t(kp)] : -1;
    s_slots[tid] = (raw >= 0 && raw < N_KEYS) ? raw : -1;
  }
  // Dual-purpose barrier: RAW for s_slots and WAR for s_kv/s_xchg from prior iteration.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint col_mask = 0u;
  #pragma unroll
  for (short ik = 0; ik < TK; ++ik) {
    #pragma unroll
    for (short jj = 0; jj < 2; ++jj) {
      if (s_slots[int(ik) * kU + fn + int(jj)] >= 0) {
        col_mask |= (1u << (int(ik) * 2 + int(jj)));
      }
    }
  }
  if (!simd_any(col_mask != 0u)) {
    prev_skipped = true;
    continue;
  }
  prev_skipped = false;
  const bool tile_all_valid = simd_all(col_mask == 0xfu);

  // First 288 cols (0..287, always from k0).
  for (int i = int(tid); i < TR * HALF_DK8; i += NT) {
    const int rr = i / HALF_DK8;
    const int c8 = i - rr * HALF_DK8;
    const int slot = s_slots[rr];
    threadgroup float4* dst =
        reinterpret_cast<threadgroup float4*>(s_kv + rr * KV_STRIDE + c8 * 8);
    if (slot >= 0) {
      constexpr int stride0 = SPLIT ? HALF_DK : D_K;
      *dst = *reinterpret_cast<const device float4*>(
          k0 + size_t(slot) * size_t(stride0) + size_t(c8 * 8));
    } else {
      *dst = float4(0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float2 S_acc0[TK];
  float2 S_acc1[TK];
  S_acc0[0] = float2(0.0f);
  S_acc0[1] = float2(0.0f);
  S_acc1[0] = float2(0.0f);
  S_acc1[1] = float2(0.0f);

  // SGs 0..3 save all 8 V tiles in registers; SG 4 saves its first 4 V tiles (256..287).
  half2 mv_reg0[TDh];
  half2 mv_reg1[TDh];

  if (d_group < 4) {
    #pragma unroll
    for (short vi = 0; vi < TDh; ++vi) {
      const int col_off = v_base + int(vi) * kU;
      simdgroup_half8x8 mk0;
      simdgroup_half8x8 mk1;
      simdgroup_load(mk0, s_kv + col_off, KV_STRIDE, ulong2(0), true);
      simdgroup_load(mk1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), true);
      HY4_M3_MMA_MAT(S_acc0[0], Q_v0[vi], mk0);
      HY4_M3_MMA_MAT(S_acc0[1], Q_v0[vi], mk1);
      HY4_M3_MMA_MAT(S_acc1[0], Q_v1[vi], mk0);
      HY4_M3_MMA_MAT(S_acc1[1], Q_v1[vi], mk1);
      mv_reg0[vi] = *reinterpret_cast<const threadgroup half2*>(
          s_kv + fm * KV_STRIDE + col_off + fn);
      mv_reg1[vi] = *reinterpret_cast<const threadgroup half2*>(
          s_kv + (fm + kU) * KV_STRIDE + col_off + fn);
    }
    s_xchg4[(d_group * 2 + 0) * 32 + lid] =
        float4(S_acc0[0].x, S_acc0[0].y, S_acc0[1].x, S_acc0[1].y);
    s_xchg4[(d_group * 2 + 1) * 32 + lid] =
        float4(S_acc1[0].x, S_acc1[0].y, S_acc1[1].x, S_acc1[1].y);
  } else if (d_group == 4) {
    #pragma unroll
    for (short vi = 0; vi < 4; ++vi) {
      const int col_off = 256 + int(vi) * kU;
      simdgroup_half8x8 mk0;
      simdgroup_half8x8 mk1;
      simdgroup_load(mk0, s_kv + col_off, KV_STRIDE, ulong2(0), true);
      simdgroup_load(mk1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), true);
      HY4_M3_MMA_MAT(S_acc0[0], Q_v0[vi], mk0);
      HY4_M3_MMA_MAT(S_acc0[1], Q_v0[vi], mk1);
      HY4_M3_MMA_MAT(S_acc1[0], Q_v1[vi], mk0);
      HY4_M3_MMA_MAT(S_acc1[1], Q_v1[vi], mk1);
      mv_reg0[vi] = *reinterpret_cast<const threadgroup half2*>(
          s_kv + fm * KV_STRIDE + col_off + fn);
      mv_reg1[vi] = *reinterpret_cast<const threadgroup half2*>(
          s_kv + (fm + kU) * KV_STRIDE + col_off + fn);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Second 288 cols (288..575; kept live in s_kv for P @ V on SGs 4..7).
  for (int i = int(tid); i < TR * HALF_DK8; i += NT) {
    const int rr = i / HALF_DK8;
    const int c8 = i - rr * HALF_DK8;
    const int slot = s_slots[rr];
    threadgroup float4* dst =
        reinterpret_cast<threadgroup float4*>(s_kv + rr * KV_STRIDE + c8 * 8);
    if (slot >= 0) {
      if (SPLIT) {
        *dst = *reinterpret_cast<const device float4*>(
            k1 + size_t(slot) * size_t(HALF_DK) + size_t(c8 * 8));
      } else {
        *dst = *reinterpret_cast<const device float4*>(
            k0 + size_t(slot) * size_t(D_K) + size_t(HALF_DK + c8 * 8));
      }
    } else {
      *dst = float4(0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (d_group == 4) {
    #pragma unroll
    for (short vi = 0; vi < 4; ++vi) {
      const int col_off = int(vi) * kU;
      simdgroup_half8x8 mk0;
      simdgroup_half8x8 mk1;
      simdgroup_load(mk0, s_kv + col_off, KV_STRIDE, ulong2(0), true);
      simdgroup_load(mk1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), true);
      HY4_M3_MMA_MAT(S_acc0[0], Q_v0[4 + vi], mk0);
      HY4_M3_MMA_MAT(S_acc0[1], Q_v0[4 + vi], mk1);
      HY4_M3_MMA_MAT(S_acc1[0], Q_v1[4 + vi], mk0);
      HY4_M3_MMA_MAT(S_acc1[1], Q_v1[4 + vi], mk1);
    }
    #pragma unroll
    for (short ri = 0; ri < 2; ++ri) {
      const int col_off = (D_V - HALF_DK) + int(ri) * kU;
      simdgroup_half8x8 mk0;
      simdgroup_half8x8 mk1;
      simdgroup_load(mk0, s_kv + col_off, KV_STRIDE, ulong2(0), true);
      simdgroup_load(mk1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), true);
      HY4_M3_MMA_MAT(S_acc0[0], Q_r0[ri], mk0);
      HY4_M3_MMA_MAT(S_acc0[1], Q_r0[ri], mk1);
      HY4_M3_MMA_MAT(S_acc1[0], Q_r1[ri], mk0);
      HY4_M3_MMA_MAT(S_acc1[1], Q_r1[ri], mk1);
    }
    s_xchg4[0 * 32 + lid] +=
        float4(S_acc0[0].x, S_acc0[0].y, S_acc0[1].x, S_acc0[1].y);
    s_xchg4[1 * 32 + lid] +=
        float4(S_acc1[0].x, S_acc1[0].y, S_acc1[1].x, S_acc1[1].y);
  } else if (d_group >= 5) {
    const int k1_base = v_base - HALF_DK;
    #pragma unroll
    for (short vi = 0; vi < TDh; ++vi) {
      const int col_off = k1_base + int(vi) * kU;
      simdgroup_half8x8 mk0;
      simdgroup_half8x8 mk1;
      simdgroup_load(mk0, s_kv + col_off, KV_STRIDE, ulong2(0), true);
      simdgroup_load(mk1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), true);
      HY4_M3_MMA_MAT(S_acc0[0], Q_v0[vi], mk0);
      HY4_M3_MMA_MAT(S_acc0[1], Q_v0[vi], mk1);
      HY4_M3_MMA_MAT(S_acc1[0], Q_v1[vi], mk0);
      HY4_M3_MMA_MAT(S_acc1[1], Q_v1[vi], mk1);
    }
    const int r_base = (D_V - HALF_DK) + (d_group - 4) * 16;
    #pragma unroll
    for (short ri = 0; ri < 2; ++ri) {
      const int col_off = r_base + int(ri) * kU;
      simdgroup_half8x8 mk0;
      simdgroup_half8x8 mk1;
      simdgroup_load(mk0, s_kv + col_off, KV_STRIDE, ulong2(0), true);
      simdgroup_load(mk1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), true);
      HY4_M3_MMA_MAT(S_acc0[0], Q_r0[ri], mk0);
      HY4_M3_MMA_MAT(S_acc0[1], Q_r0[ri], mk1);
      HY4_M3_MMA_MAT(S_acc1[0], Q_r1[ri], mk0);
      HY4_M3_MMA_MAT(S_acc1[1], Q_r1[ri], mk1);
    }
    const int slot_idx = d_group - 4;
    s_xchg4[(slot_idx * 2 + 0) * 32 + lid] +=
        float4(S_acc0[0].x, S_acc0[0].y, S_acc0[1].x, S_acc0[1].y);
    s_xchg4[(slot_idx * 2 + 1) * 32 + lid] +=
        float4(S_acc1[0].x, S_acc1[0].y, S_acc1[1].x, S_acc1[1].y);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Deterministic binary-tree sum across WN=8: ((dg0+dg4)+(dg1+dg5))+((dg2+dg6)+(dg3+dg7)).
  const float4 tot0 =
      (s_xchg4[(0 * 2 + 0) * 32 + lid] + s_xchg4[(1 * 2 + 0) * 32 + lid]) +
      (s_xchg4[(2 * 2 + 0) * 32 + lid] + s_xchg4[(3 * 2 + 0) * 32 + lid]);
  const float4 tot1 =
      (s_xchg4[(0 * 2 + 1) * 32 + lid] + s_xchg4[(1 * 2 + 1) * 32 + lid]) +
      (s_xchg4[(2 * 2 + 1) * 32 + lid] + s_xchg4[(3 * 2 + 1) * 32 + lid]);
  S_acc0[0] = tot0.xy;
  S_acc0[1] = tot0.zw;
  S_acc1[0] = tot1.xy;
  S_acc1[1] = tot1.zw;

  // In-register online softmax for both head groups (hg = 0, 1).
  float new_max0 = max_score0;
  float new_max1 = max_score1;
  if (tile_all_valid) {
    S_acc0[0] *= scale2;
    S_acc0[1] *= scale2;
    S_acc1[0] *= scale2;
    S_acc1[1] *= scale2;

    float rm0 = metal::max(
        metal::max(S_acc0[0].x, S_acc0[0].y),
        metal::max(S_acc0[1].x, S_acc0[1].y));
    rm0 = metal::max(rm0, simd_shuffle_xor(rm0, ushort(1)));
    rm0 = metal::max(rm0, simd_shuffle_xor(rm0, ushort(8)));
    new_max0 = metal::max(new_max0, rm0);

    float rm1 = metal::max(
        metal::max(S_acc1[0].x, S_acc1[0].y),
        metal::max(S_acc1[1].x, S_acc1[1].y));
    rm1 = metal::max(rm1, simd_shuffle_xor(rm1, ushort(1)));
    rm1 = metal::max(rm1, simd_shuffle_xor(rm1, ushort(8)));
    new_max1 = metal::max(new_max1, rm1);

    S_acc0[0].x = metal::exp2(S_acc0[0].x - new_max0);
    S_acc0[0].y = metal::exp2(S_acc0[0].y - new_max0);
    S_acc0[1].x = metal::exp2(S_acc0[1].x - new_max0);
    S_acc0[1].y = metal::exp2(S_acc0[1].y - new_max0);

    S_acc1[0].x = metal::exp2(S_acc1[0].x - new_max1);
    S_acc1[0].y = metal::exp2(S_acc1[0].y - new_max1);
    S_acc1[1].x = metal::exp2(S_acc1[1].x - new_max1);
    S_acc1[1].y = metal::exp2(S_acc1[1].y - new_max1);
  } else {
    const bool ok00 = (col_mask & 1u) != 0u;
    const bool ok01 = (col_mask & 2u) != 0u;
    const bool ok10 = (col_mask & 4u) != 0u;
    const bool ok11 = (col_mask & 8u) != 0u;

    S_acc0[0].x = ok00 ? (S_acc0[0].x * scale2) : MAX_FLOOR;
    S_acc0[0].y = ok01 ? (S_acc0[0].y * scale2) : MAX_FLOOR;
    S_acc0[1].x = ok10 ? (S_acc0[1].x * scale2) : MAX_FLOOR;
    S_acc0[1].y = ok11 ? (S_acc0[1].y * scale2) : MAX_FLOOR;

    S_acc1[0].x = ok00 ? (S_acc1[0].x * scale2) : MAX_FLOOR;
    S_acc1[0].y = ok01 ? (S_acc1[0].y * scale2) : MAX_FLOOR;
    S_acc1[1].x = ok10 ? (S_acc1[1].x * scale2) : MAX_FLOOR;
    S_acc1[1].y = ok11 ? (S_acc1[1].y * scale2) : MAX_FLOOR;

    float rm0 = metal::max(
        metal::max(S_acc0[0].x, S_acc0[0].y),
        metal::max(S_acc0[1].x, S_acc0[1].y));
    rm0 = metal::max(rm0, simd_shuffle_xor(rm0, ushort(1)));
    rm0 = metal::max(rm0, simd_shuffle_xor(rm0, ushort(8)));
    new_max0 = metal::max(new_max0, rm0);

    float rm1 = metal::max(
        metal::max(S_acc1[0].x, S_acc1[0].y),
        metal::max(S_acc1[1].x, S_acc1[1].y));
    rm1 = metal::max(rm1, simd_shuffle_xor(rm1, ushort(1)));
    rm1 = metal::max(rm1, simd_shuffle_xor(rm1, ushort(8)));
    new_max1 = metal::max(new_max1, rm1);

    S_acc0[0].x = ok00 ? metal::exp2(S_acc0[0].x - new_max0) : 0.0f;
    S_acc0[0].y = ok01 ? metal::exp2(S_acc0[0].y - new_max0) : 0.0f;
    S_acc0[1].x = ok10 ? metal::exp2(S_acc0[1].x - new_max0) : 0.0f;
    S_acc0[1].y = ok11 ? metal::exp2(S_acc0[1].y - new_max0) : 0.0f;

    S_acc1[0].x = ok00 ? metal::exp2(S_acc1[0].x - new_max1) : 0.0f;
    S_acc1[0].y = ok01 ? metal::exp2(S_acc1[0].y - new_max1) : 0.0f;
    S_acc1[1].x = ok10 ? metal::exp2(S_acc1[1].x - new_max1) : 0.0f;
    S_acc1[1].y = ok11 ? metal::exp2(S_acc1[1].y - new_max1) : 0.0f;
  }

  const float factor0 = metal::exp2(max_score0 - new_max0);
  const float factor1 = metal::exp2(max_score1 - new_max1);
  max_score0 = new_max0;
  max_score1 = new_max1;

  float rs0 = (S_acc0[0].x + S_acc0[0].y) + (S_acc0[1].x + S_acc0[1].y);
  rs0 += simd_shuffle_xor(rs0, ushort(1));
  rs0 += simd_shuffle_xor(rs0, ushort(8));
  sum_score0 = metal::fma(sum_score0, factor0, rs0);

  float rs1 = (S_acc1[0].x + S_acc1[0].y) + (S_acc1[1].x + S_acc1[1].y);
  rs1 += simd_shuffle_xor(rs1, ushort(1));
  rs1 += simd_shuffle_xor(rs1, ushort(8));
  sum_score1 = metal::fma(sum_score1, factor1, rs1);

  // Register-only P cast.
  const half2 p0_0 = half2(S_acc0[0]);
  const half2 p0_1 = half2(S_acc0[1]);
  const half2 p1_0 = half2(S_acc1[0]);
  const half2 p1_1 = half2(S_acc1[1]);

  // Direct scalar O rescale + P @ V.
  if (d_group < 4) {
    #pragma unroll
    for (short vi = 0; vi < TDh; ++vi) {
      O_acc0[vi] *= factor0;
      O_acc1[vi] *= factor1;
      HY4_M3_MMA_REG(O_acc0[vi], p0_0, mv_reg0[vi]);
      HY4_M3_MMA_REG(O_acc0[vi], p0_1, mv_reg1[vi]);
      HY4_M3_MMA_REG(O_acc1[vi], p1_0, mv_reg0[vi]);
      HY4_M3_MMA_REG(O_acc1[vi], p1_1, mv_reg1[vi]);
    }
  } else if (d_group == 4) {
    #pragma unroll
    for (short vi = 0; vi < 4; ++vi) {
      O_acc0[vi] *= factor0;
      O_acc1[vi] *= factor1;
      HY4_M3_MMA_REG(O_acc0[vi], p0_0, mv_reg0[vi]);
      HY4_M3_MMA_REG(O_acc0[vi], p0_1, mv_reg1[vi]);
      HY4_M3_MMA_REG(O_acc1[vi], p1_0, mv_reg0[vi]);
      HY4_M3_MMA_REG(O_acc1[vi], p1_1, mv_reg1[vi]);
    }
    #pragma unroll
    for (short vi = 0; vi < 4; ++vi) {
      const int col_off = int(vi) * kU;
      simdgroup_half8x8 mv0;
      simdgroup_half8x8 mv1;
      simdgroup_load(mv0, s_kv + col_off, KV_STRIDE, ulong2(0), false);
      simdgroup_load(mv1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), false);
      O_acc0[4 + vi] *= factor0;
      O_acc1[4 + vi] *= factor1;
      HY4_M3_MMA_MAT(O_acc0[4 + vi], p0_0, mv0);
      HY4_M3_MMA_MAT(O_acc0[4 + vi], p0_1, mv1);
      HY4_M3_MMA_MAT(O_acc1[4 + vi], p1_0, mv0);
      HY4_M3_MMA_MAT(O_acc1[4 + vi], p1_1, mv1);
    }
  } else {
    const int k1_base = v_base - HALF_DK;
    #pragma unroll
    for (short vi = 0; vi < TDh; ++vi) {
      const int col_off = k1_base + int(vi) * kU;
      simdgroup_half8x8 mv0;
      simdgroup_half8x8 mv1;
      simdgroup_load(mv0, s_kv + col_off, KV_STRIDE, ulong2(0), false);
      simdgroup_load(mv1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), false);
      O_acc0[vi] *= factor0;
      O_acc1[vi] *= factor1;
      HY4_M3_MMA_MAT(O_acc0[vi], p0_0, mv0);
      HY4_M3_MMA_MAT(O_acc0[vi], p0_1, mv1);
      HY4_M3_MMA_MAT(O_acc1[vi], p1_0, mv0);
      HY4_M3_MMA_MAT(O_acc1[vi], p1_1, mv1);
    }
  }
}

#undef HY4_M3_MMA_MAT
#undef HY4_M3_MMA_REG

const bool nz0 = (sum_score0 > 0.0f);
const bool nz1 = (sum_score1 > 0.0f);

if (SINGLE_SPLIT) {
  // Match mla_decode_merge's exact single-active-split formula bit-for-bit.
  if (h_ok0) {
    const float max_nat0 = nz0 ? (max_score0 * M_LN2_F) : MAX_FLOOR;
    const float gm0 = nz0 ? metal::max(MAX_FLOOR, max_nat0) : MAX_FLOOR;
    const float sink0 = sinks[h_idx0];
    const float m_all0 = metal::max(gm0, sink0);
    const float k_scale0 = metal::exp(gm0 - m_all0);
    const float s_term0 = metal::exp(sink0 - m_all0);
    const float den0 = sum_score0 * k_scale0 + s_term0;
    const bool ok0 = nz0 && (den0 > 0.0f);
    device float* out0 =
        o_part + (size_t(s) * size_t(H) + size_t(h_idx0)) * size_t(D_V);
    #pragma unroll
    for (short vi = 0; vi < TDh; ++vi) {
      const int col = v_base + int(vi) * kU + fn;
      const float2 val = ok0 ? ((O_acc0[vi] * k_scale0) / den0) : float2(0.0f);
      out0[col] = val.x;
      out0[col + 1] = val.y;
    }
  }
  if (h_ok1) {
    const float max_nat1 = nz1 ? (max_score1 * M_LN2_F) : MAX_FLOOR;
    const float gm1 = nz1 ? metal::max(MAX_FLOOR, max_nat1) : MAX_FLOOR;
    const float sink1 = sinks[h_idx1];
    const float m_all1 = metal::max(gm1, sink1);
    const float k_scale1 = metal::exp(gm1 - m_all1);
    const float s_term1 = metal::exp(sink1 - m_all1);
    const float den1 = sum_score1 * k_scale1 + s_term1;
    const bool ok1 = nz1 && (den1 > 0.0f);
    device float* out1 =
        o_part + (size_t(s) * size_t(H) + size_t(h_idx1)) * size_t(D_V);
    #pragma unroll
    for (short vi = 0; vi < TDh; ++vi) {
      const int col = v_base + int(vi) * kU + fn;
      const float2 val = ok1 ? ((O_acc1[vi] * k_scale1) / den1) : float2(0.0f);
      out1[col] = val.x;
      out1[col + 1] = val.y;
    }
  }
} else {
  if (h_ok0) {
    const size_t mp0 =
        (size_t(s) * size_t(H) + size_t(h_idx0)) * size_t(n_splits) + size_t(sp);
    device float* out0 = o_part + mp0 * size_t(D_V);
    #pragma unroll
    for (short vi = 0; vi < TDh; ++vi) {
      const int col = v_base + int(vi) * kU + fn;
      const float2 val = nz0 ? O_acc0[vi] : float2(0.0f);
      out0[col] = val.x;
      out0[col + 1] = val.y;
    }
    if (d_group == 0 && fn == 0) {
      max_part[mp0] = nz0 ? (max_score0 * M_LN2_F) : MAX_FLOOR;
      sum_part[mp0] = sum_score0;
    }
  }
  if (h_ok1) {
    const size_t mp1 =
        (size_t(s) * size_t(H) + size_t(h_idx1)) * size_t(n_splits) + size_t(sp);
    device float* out1 = o_part + mp1 * size_t(D_V);
    #pragma unroll
    for (short vi = 0; vi < TDh; ++vi) {
      const int col = v_base + int(vi) * kU + fn;
      const float2 val = nz1 ? O_acc1[vi] : float2(0.0f);
      out1[col] = val.x;
      out1[col + 1] = val.y;
    }
    if (d_group == 0 && fn == 0) {
      max_part[mp1] = nz1 ? (max_score1 * M_LN2_F) : MAX_FLOOR;
      sum_part[mp1] = sum_score1;
    }
  }
}
)METAL"