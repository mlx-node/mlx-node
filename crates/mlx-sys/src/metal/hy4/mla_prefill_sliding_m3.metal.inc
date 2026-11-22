R"metal(
static_assert(WN == 4, "mla_prefill_sliding_m3 requires WN == 4");
static_assert(WM == 2 || WM == 4, "mla_prefill_sliding_m3 requires WM == 2 or 4");
static_assert(
    D_V % (WN * 8) == 0 && D_ROPE % (WN * 8) == 0,
    "mla_prefill_sliding_m3 requires D_V and D_ROPE divisible by WN * 8");

constexpr int D_K = D_V + D_ROPE;
constexpr int kU = 8;
constexpr int BQ = kU;
constexpr int BH = WM;
constexpr int BK = 16;
constexpr int TK = BK / kU;
constexpr int KV_STRIDE = D_V + 8;
constexpr int ROPE_STRIDE = D_ROPE + 8;
constexpr int DV_PER_SG = D_V / WN;
constexpr int DR_PER_SG = D_ROPE / WN;
constexpr int TDh = DV_PER_SG / kU;
constexpr int TRh = DR_PER_SG / kU;
constexpr int kEPF = 2;
constexpr int XCHG_STRIDE = TK * kEPF * 32;
constexpr int NUM_THREADS = WM * WN * 32;
constexpr int VEC8_V = D_V / 8;
constexpr int VEC8_ROPE = D_ROPE / 8;
constexpr int V_ROWS_PER_PASS = NUM_THREADS / VEC8_V;
constexpr int V_PASSES = BK / V_ROWS_PER_PASS;
constexpr float MAX_FLOOR = -1e30f;
constexpr bool USE_Q_LO = !FAST_1MMA && (sizeof(Q_T) == 4);

static_assert(
    NUM_THREADS % VEC8_V == 0 && BK % V_ROWS_PER_PASS == 0,
    "mla_prefill_sliding_m3 cooperative V load geometry mismatch");
static_assert(
    NUM_THREADS >= BK * VEC8_ROPE,
    "mla_prefill_sliding_m3 requires enough threads for single-pass RoPE load");

#define HY4_M3_MMA(ACC, A_VAL, B_MAT)                                  \
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

const int S = q_shape[0];
const int H = q_shape[1];
const int start_pos = int(params[0]);
const int ring_cap = int(params[1]);
const int window_size = int(params[2]);
const int first_key = int(params[3]);

const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
const uint tid = simd_gid * 32u + simd_lid;
const int row_group = int(simd_gid) / WN;
const int d_group = int(simd_gid) % WN;

// Each TG shares loaded K/V across BH = WM heads for the same BQ = 8 query tokens.
const int q_tile_base = int(threadgroup_position_in_grid.y) * BQ;
const int first_head = int(threadgroup_position_in_grid.z) * BH;
const int head = first_head + row_group;
if (q_tile_base >= S || first_head >= H) {
  return;
}

// 1-pass Full-V TG layout (16B-aligned, 4-bank skew):
// s_kv (16,640 B) + s_rope (2,304 B) + s_xchg (4,096 B for WM=4 / 2,048 B for WM=2).
alignas(16) threadgroup half s_kv[BK * KV_STRIDE];
alignas(16) threadgroup half s_rope[BK * ROPE_STRIDE];
alignas(16) threadgroup float s_xchg[WM * 2 * XCHG_STRIDE];

// Exact Metal 3 8x8 simdgroup_matrix lane mapping:
// lane owns query row fm and columns (fn, fn + 1).
const uint qid = simd_lid >> 2;
const int fm = int((qid & 4u) + ((simd_lid >> 1) & 3u));
const int fn = int(((qid & 2u) << 1) + ((simd_lid & 1u) << 1));

const int valid_q = (head < H)
    ? metal::max(0, metal::min(BQ, S - q_tile_base))
    : 0;
const int q_idx = q_tile_base + fm;
const int q_pos = start_pos + q_idx;
const bool q_ok = (head < H) && (q_idx < S);

const int v_col_base = d_group * DV_PER_SG;
const int r_col_base = d_group * DR_PER_SG;

const int q_v_dim = SPLIT_Q ? D_V : D_K;
const device Q_T* qv_row =
    q + (size_t(q_ok ? q_idx : 0) * size_t(H) + size_t(q_ok ? head : 0)) *
            size_t(q_v_dim);
const device Q_T* qr_row = SPLIT_Q
    ? (q_pe +
       (size_t(q_ok ? q_idx : 0) * size_t(H) + size_t(q_ok ? head : 0)) *
           size_t(D_ROPE))
    : (qv_row + D_V);

half2 Qv_hi[TDh];
half2 Qv_lo[USE_Q_LO ? TDh : 1];
#pragma unroll
for (short iv = 0; iv < TDh; ++iv) {
  const int col = v_col_base + int(iv) * kU + fn;
  if (q_ok) {
    if (USE_Q_LO) {
      const float v0 = float(qv_row[col]);
      const float v1 = float(qv_row[col + 1]);
      const half h0 = half(v0);
      const half h1 = half(v1);
      Qv_hi[iv] = half2(h0, h1);
      Qv_lo[USE_Q_LO ? iv : 0] = half2(half(v0 - float(h0)), half(v1 - float(h1)));
    } else {
      Qv_hi[iv] = half2(half(qv_row[col]), half(qv_row[col + 1]));
    }
  } else {
    Qv_hi[iv] = half2(0.0h);
    if (USE_Q_LO) {
      Qv_lo[USE_Q_LO ? iv : 0] = half2(0.0h);
    }
  }
}

half2 Qr_hi[TRh];
half2 Qr_lo[USE_Q_LO ? TRh : 1];
#pragma unroll
for (short ir = 0; ir < TRh; ++ir) {
  const int col = r_col_base + int(ir) * kU + fn;
  if (q_ok) {
    if (USE_Q_LO) {
      const float v0 = float(qr_row[col]);
      const float v1 = float(qr_row[col + 1]);
      const half h0 = half(v0);
      const half h1 = half(v1);
      Qr_hi[ir] = half2(h0, h1);
      Qr_lo[USE_Q_LO ? ir : 0] = half2(half(v0 - float(h0)), half(v1 - float(h1)));
    } else {
      Qr_hi[ir] = half2(half(qr_row[col]), half(qr_row[col + 1]));
    }
  } else {
    Qr_hi[ir] = half2(0.0h);
    if (USE_Q_LO) {
      Qr_lo[USE_Q_LO ? ir : 0] = half2(0.0h);
    }
  }
}

float2 O_acc[TDh];
#pragma unroll
for (short iv = 0; iv < TDh; ++iv) {
  O_acc[iv] = float2(0.0f);
}

float max_score = MAX_FLOOR;
float sum_score = 0.0f;

const float scale2 = scale[0] * M_LOG2E_F;

const int q_tg_lo = start_pos + q_tile_base;
const int q_tg_hi = start_pos + metal::min(S, q_tile_base + BQ) - 1;
const int k_lo = metal::max(first_key, q_tg_lo + 1 - window_size);
const int k_hi = q_tg_hi;

const int v_row0 = int(tid) / VEC8_V;
const int v_c8 = int(tid) % VEC8_V;
const int rope_row = int(tid) / VEC8_ROPE;
const int rope_c8 = int(tid) % VEC8_ROPE;

threadgroup float4* s_xchg4 = reinterpret_cast<threadgroup float4*>(s_xchg);
const int xidx0 = (row_group * 2) * 32 + int(simd_lid);
const int xidx1 = xidx0 + 32;
const int my_xidx = ((d_group & 1) == 0) ? xidx0 : xidx1;

for (int k_base = k_lo; k_base <= k_hi; k_base += BK) {
  if (k_base > k_lo) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Cooperative 1-pass load of full V (0..511) into s_kv and RoPE (512..575) into s_rope.
  #pragma unroll
  for (int pass = 0; pass < V_PASSES; ++pass) {
    const int row = v_row0 + pass * V_ROWS_PER_PASS;
    const int kp = k_base + row;
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_kv + row * KV_STRIDE + v_c8 * 8);
    if (kp <= k_hi) {
      const int slot =
          IS_RING ? int(uint(kp) % uint(ring_cap)) : (kp - first_key);
      *dst = *reinterpret_cast<const device float4*>(
          k + size_t(slot) * size_t(D_K) + size_t(v_c8 * 8));
    } else {
      *dst = float4(0.0f);
    }
  }
  if (rope_row < BK) {
    const int kp = k_base + rope_row;
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_rope + rope_row * ROPE_STRIDE + rope_c8 * 8);
    if (kp <= k_hi) {
      const int slot =
          IS_RING ? int(uint(kp) % uint(ring_cap)) : (kp - first_key);
      *dst = *reinterpret_cast<const device float4*>(
          k + size_t(slot) * size_t(D_K) + size_t(D_V + rope_c8 * 8));
    } else {
      *dst = float4(0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Q @ K^T across this simdgroup's D_V/WN + D_ROPE/WN slice.
  float2 S_acc[TK];
  S_acc[0] = float2(0.0f);
  S_acc[1] = float2(0.0f);

  if (valid_q > 0) {
    #pragma unroll
    for (short iv = 0; iv < TDh; ++iv) {
      const int col_off = v_col_base + int(iv) * kU;
      simdgroup_half8x8 mk0;
      simdgroup_half8x8 mk1;
      simdgroup_load(mk0, s_kv + col_off, KV_STRIDE, ulong2(0), true);
      simdgroup_load(
          mk1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), true);
      HY4_M3_MMA(S_acc[0], Qv_hi[iv], mk0);
      HY4_M3_MMA(S_acc[1], Qv_hi[iv], mk1);
      if (USE_Q_LO) {
        HY4_M3_MMA(S_acc[0], Qv_lo[USE_Q_LO ? iv : 0], mk0);
        HY4_M3_MMA(S_acc[1], Qv_lo[USE_Q_LO ? iv : 0], mk1);
      }
    }

    #pragma unroll
    for (short ir = 0; ir < TRh; ++ir) {
      const int col_off = r_col_base + int(ir) * kU;
      simdgroup_half8x8 mk0;
      simdgroup_half8x8 mk1;
      simdgroup_load(mk0, s_rope + col_off, ROPE_STRIDE, ulong2(0), true);
      simdgroup_load(
          mk1, s_rope + kU * ROPE_STRIDE + col_off, ROPE_STRIDE, ulong2(0), true);
      HY4_M3_MMA(S_acc[0], Qr_hi[ir], mk0);
      HY4_M3_MMA(S_acc[1], Qr_hi[ir], mk1);
      if (USE_Q_LO) {
        HY4_M3_MMA(S_acc[0], Qr_lo[USE_Q_LO ? ir : 0], mk0);
        HY4_M3_MMA(S_acc[1], Qr_lo[USE_Q_LO ? ir : 0], mk1);
      }
    }
  }

  // Deterministic 2-slot tree reduction across WN=4 in s_xchg: (dg0+dg2)+(dg1+dg3).
  const float4 my_s = float4(S_acc[0].x, S_acc[0].y, S_acc[1].x, S_acc[1].y);
  if (valid_q > 0 && d_group >= 2) {
    s_xchg4[my_xidx] = my_s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (valid_q > 0 && d_group < 2) {
    s_xchg4[my_xidx] = my_s + s_xchg4[my_xidx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (valid_q > 0) {
    const float4 s_sum = s_xchg4[xidx0] + s_xchg4[xidx1];
    S_acc[0] = s_sum.xy;
    S_acc[1] = s_sum.zw;

    const bool tile_all_valid =
        (valid_q == BQ) &&
        ((k_base + BK - 1) <= q_tg_lo) &&
        ((q_tg_hi - k_base) < window_size);

    // Exact F32 online softmax (exp2 + scale * M_LOG2E_F).
    float new_max = max_score;
    if (tile_all_valid) {
      S_acc[0] *= scale2;
      S_acc[1] *= scale2;
      float row_max = metal::max(
          metal::max(S_acc[0].x, S_acc[0].y),
          metal::max(S_acc[1].x, S_acc[1].y));
      row_max = metal::max(row_max, simd_shuffle_xor(row_max, ushort(1)));
      row_max = metal::max(row_max, simd_shuffle_xor(row_max, ushort(8)));
      new_max = metal::max(new_max, row_max);

      S_acc[0].x = metal::exp2(S_acc[0].x - new_max);
      S_acc[0].y = metal::exp2(S_acc[0].y - new_max);
      S_acc[1].x = metal::exp2(S_acc[1].x - new_max);
      S_acc[1].y = metal::exp2(S_acc[1].y - new_max);
    } else {
      const int kp00 = k_base + fn;
      const int kp01 = kp00 + 1;
      const int kp10 = k_base + kU + fn;
      const int kp11 = kp10 + 1;

      const bool ok00 = q_ok && (kp00 <= q_pos) && ((q_pos - kp00) < window_size);
      const bool ok01 = q_ok && (kp01 <= q_pos) && ((q_pos - kp01) < window_size);
      const bool ok10 = q_ok && (kp10 <= q_pos) && ((q_pos - kp10) < window_size);
      const bool ok11 = q_ok && (kp11 <= q_pos) && ((q_pos - kp11) < window_size);

      S_acc[0].x = ok00 ? (S_acc[0].x * scale2) : MAX_FLOOR;
      S_acc[0].y = ok01 ? (S_acc[0].y * scale2) : MAX_FLOOR;
      S_acc[1].x = ok10 ? (S_acc[1].x * scale2) : MAX_FLOOR;
      S_acc[1].y = ok11 ? (S_acc[1].y * scale2) : MAX_FLOOR;

      float row_max = metal::max(
          metal::max(S_acc[0].x, S_acc[0].y),
          metal::max(S_acc[1].x, S_acc[1].y));
      row_max = metal::max(row_max, simd_shuffle_xor(row_max, ushort(1)));
      row_max = metal::max(row_max, simd_shuffle_xor(row_max, ushort(8)));
      new_max = metal::max(new_max, row_max);

      S_acc[0].x = ok00 ? metal::exp2(S_acc[0].x - new_max) : 0.0f;
      S_acc[0].y = ok01 ? metal::exp2(S_acc[0].y - new_max) : 0.0f;
      S_acc[1].x = ok10 ? metal::exp2(S_acc[1].x - new_max) : 0.0f;
      S_acc[1].y = ok11 ? metal::exp2(S_acc[1].y - new_max) : 0.0f;
    }

    const float factor = metal::exp2(max_score - new_max);
    max_score = new_max;
    float row_sum = (S_acc[0].x + S_acc[0].y) + (S_acc[1].x + S_acc[1].y);
    row_sum += simd_shuffle_xor(row_sum, ushort(1));
    row_sum += simd_shuffle_xor(row_sum, ushort(8));
    sum_score = metal::fma(sum_score, factor, row_sum);

    // Register-only P cast (hi + optional lo residual for 2MMA).
    half2 p_hi[TK];
    half2 p_lo[FAST_1MMA ? 1 : TK];
    #pragma unroll
    for (short ik = 0; ik < TK; ++ik) {
      const float2 pf = S_acc[ik];
      const half2 hi = half2(pf);
      p_hi[ik] = hi;
      if (!FAST_1MMA) {
        p_lo[FAST_1MMA ? 0 : ik] = half2(pf - float2(hi));
      }
    }

    // Rescale O and accumulate P @ V directly from resident s_kv.
    #pragma unroll
    for (short iv = 0; iv < TDh; ++iv) {
      const int col_off = v_col_base + int(iv) * kU;
      simdgroup_half8x8 mv0;
      simdgroup_half8x8 mv1;
      simdgroup_load(mv0, s_kv + col_off, KV_STRIDE, ulong2(0), false);
      simdgroup_load(
          mv1, s_kv + kU * KV_STRIDE + col_off, KV_STRIDE, ulong2(0), false);
      O_acc[iv] *= factor;
      HY4_M3_MMA(O_acc[iv], p_hi[0], mv0);
      HY4_M3_MMA(O_acc[iv], p_hi[1], mv1);
      if (!FAST_1MMA) {
        HY4_M3_MMA(O_acc[iv], p_lo[0], mv0);
        HY4_M3_MMA(O_acc[iv], p_lo[FAST_1MMA ? 0 : 1], mv1);
      }
    }
  }
}

#undef HY4_M3_MMA

if (q_ok) {
  const float sink_log2 = sinks[head] * M_LOG2E_F;
  const float m_h = max_score;
  const float l_h = sum_score;
  const float m_all = metal::max(m_h, sink_log2);
  const float k_scale = metal::exp2(m_h - m_all);
  const float s_term = metal::exp2(sink_log2 - m_all);
  const float den = metal::fma(l_h, k_scale, s_term);
  const bool valid_out = (l_h > 0.0f && den > 0.0f);
  const float out_scale = valid_out ? (k_scale / den) : 0.0f;

  device OUT_T* out_row =
      out + (size_t(q_idx) * size_t(H) + size_t(head)) * size_t(D_V);
  #pragma unroll
  for (short iv = 0; iv < TDh; ++iv) {
    const int col = v_col_base + int(iv) * kU + fn;
    const float2 o = valid_out ? (O_acc[iv] * out_scale) : float2(0.0f);
    out_row[col] = OUT_T(o.x);
    out_row[col + 1] = OUT_T(o.y);
  }
}
)metal"