R"METAL(
constexpr int D_V = 512;
constexpr int D_ROPE = 64;
constexpr int D_K = D_V + D_ROPE;
constexpr int HALF_DK = D_K / 2;
constexpr int TR = 32;
constexpr short kU = 16;
constexpr int WN = 8;
constexpr int NT = WN * 32;
constexpr int TK = TR / kU;
constexpr int KV_STRIDE = HALF_DK + 8;
constexpr int HALF_DK8 = HALF_DK / 8;
constexpr int TDh = D_V / (WN * kU);
constexpr int S_KV_U32 = (TR * KV_STRIDE) / 2;
constexpr int XCHG_STRIDE = TK * BaseNAXFrag::kElemsPerFrag * 32;
constexpr float MAX_FLOOR = -1e30f;

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
const int head_base = head_group * int(kU);
const int s = int(threadgroup_position_in_grid.z);
if (s >= S || head_base >= H) {
  return;
}

using otile_t = NAXTile<float, 1, TDh>;
using stile_t = NAXTile<float, 1, TK>;
using hfrag_t = typename BaseNAXFrag::template dtype_frag_t<half>;
constexpr short kEPF = BaseNAXFrag::kElemsPerFrag;

// 32 x 296 half staging (18.5 KiB) + 4-slot score reduction buffer (8 KiB)
// keeps total TG memory under 27 KiB and eliminates score-aliasing barriers.
alignas(16) threadgroup uint s_buf[S_KV_U32];
threadgroup half* s_kv = reinterpret_cast<threadgroup half*>(s_buf);
threadgroup float s_xchg[4 * XCHG_STRIDE];
threadgroup int s_slots[TR];

const short2 coord = BaseNAXFrag::get_coord();
const short fm = coord.y;
const short fn = coord.x;

const device float* q_base =
    q + (size_t(s) * size_t(H) + size_t(head_base)) * size_t(D_K);

auto load_q_half = [&](int col) -> hfrag_t {
  const float4 qa = *reinterpret_cast<const device float4*>(
      q_base + size_t(fm) * size_t(D_K) + size_t(col + int(fn)));
  const float4 qb = *reinterpret_cast<const device float4*>(
      q_base + size_t(fm + 8) * size_t(D_K) + size_t(col + int(fn)));
  const half4 ha = half4(qa);
  const half4 hb = half4(qb);
  return hfrag_t(ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w);
};

// Each SG owns its own 64 V columns for both Q @ K^T and P @ V so all 32 key
// rows' V fragments stay in registers across softmax with zero re-reads.
const int v_base = d_group * (D_V / WN);
hfrag_t Q_v[4];
STEEL_PRAGMA_UNROLL
for (short k = 0; k < 4; ++k) {
  Q_v[k] = load_q_half(v_base + int(k) * kU);
}
hfrag_t Q_rope;
if (d_group >= 4) {
  Q_rope = load_q_half(D_V + (d_group - 4) * kU);
}

otile_t Otile;
Otile.clear();

metal::vec<float, 2> max_score(MAX_FLOOR);
metal::vec<float, 2> sum_score(0.0f);

const float scale2 = scale[0] * M_LOG2E_F;
const int k_start = sp * k_split;
const int k_end =
    (sp + 1 < n_splits) ? metal::min(k_start + k_split, KMAX) : KMAX;
const size_t idx_base = size_t(s) * size_t(KMAX);

for (int rt = k_start; rt < k_end; rt += TR) {
  if (tid < uint(TR)) {
    const int kp = rt + int(tid);
    const int raw = (kp < k_end) ? idx[idx_base + size_t(kp)] : -1;
    s_slots[tid] = (raw >= 0 && raw < N_KEYS) ? raw : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  bool col_valid[TK][4];
  bool lane_any = false;
  bool lane_all = true;
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < 4; ++jj) {
      const bool ok = s_slots[int(ik) * kU + int(fn) + int(jj)] >= 0;
      col_valid[ik][jj] = ok;
      lane_any = lane_any || ok;
      lane_all = lane_all && ok;
    }
  }
  if (!simd_any(lane_any)) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    continue;
  }
  const bool tile_all_valid = simd_all(lane_all);

  stile_t Stile;
  Stile.clear();
  hfrag_t v_reg0[4];
  hfrag_t v_reg1[4];

  // First 288 cols (always from k0).
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

  if (d_group < 4) {
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < 4; ++k) {
      const int col_off = v_base + int(k) * kU;
      hy4_load_kv_frag_tg<half, KV_STRIDE>(v_reg0[k], s_kv, col_off, fm, fn);
      hy4_load_kv_frag_tg<half, KV_STRIDE>(
          v_reg1[k], s_kv + kU * KV_STRIDE, col_off, fm, fn);
      BaseNAXFrag::mma(
          Stile.frag_at(0, 0),
          Stile.frag_at(0, 1),
          Q_v[k],
          metal::false_type{},
          v_reg0[k],
          v_reg1[k],
          metal::true_type{});
    }
    const int base = d_group * XCHG_STRIDE + int(simd_lid);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      const thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; ++i) {
        s_xchg[base + (int(ik) * kEPF + int(i)) * 32] = sf[i];
      }
    }
  } else if (d_group == 4) {
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < 2; ++k) {
      const int col_off = 256 + int(k) * kU;
      hy4_load_kv_frag_tg<half, KV_STRIDE>(v_reg0[k], s_kv, col_off, fm, fn);
      hy4_load_kv_frag_tg<half, KV_STRIDE>(
          v_reg1[k], s_kv + kU * KV_STRIDE, col_off, fm, fn);
      BaseNAXFrag::mma(
          Stile.frag_at(0, 0),
          Stile.frag_at(0, 1),
          Q_v[k],
          metal::false_type{},
          v_reg0[k],
          v_reg1[k],
          metal::true_type{});
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Second 288 cols (k1 when SPLIT, k0 + 288 otherwise).
  for (int i = int(tid); i < TR * HALF_DK8; i += NT) {
    const int rr = i / HALF_DK8;
    const int c8 = i - rr * HALF_DK8;
    const int slot = s_slots[rr];
    threadgroup float4* dst =
        reinterpret_cast<threadgroup float4*>(s_kv + rr * KV_STRIDE + c8 * 8);
    if (slot >= 0) {
      if constexpr (SPLIT) {
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
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < 2; ++k) {
      const int col_off = int(k) * kU;
      hy4_load_kv_frag_tg<half, KV_STRIDE>(v_reg0[2 + k], s_kv, col_off, fm, fn);
      hy4_load_kv_frag_tg<half, KV_STRIDE>(
          v_reg1[2 + k], s_kv + kU * KV_STRIDE, col_off, fm, fn);
      BaseNAXFrag::mma(
          Stile.frag_at(0, 0),
          Stile.frag_at(0, 1),
          Q_v[2 + k],
          metal::false_type{},
          v_reg0[2 + k],
          v_reg1[2 + k],
          metal::true_type{});
    }
    hfrag_t kr0, kr1;
    hy4_load_kv_frag_tg<half, KV_STRIDE>(kr0, s_kv, D_V - HALF_DK, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        kr1, s_kv + kU * KV_STRIDE, D_V - HALF_DK, fm, fn);
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Q_rope,
        metal::false_type{},
        kr0,
        kr1,
        metal::true_type{});
    const int base = int(simd_lid);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      const thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; ++i) {
        const int off = base + (int(ik) * kEPF + int(i)) * 32;
        s_xchg[off] = s_xchg[off] + sf[i];
      }
    }
  } else if (d_group >= 5) {
    const int k1_base = v_base - HALF_DK;
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < 4; ++k) {
      const int col_off = k1_base + int(k) * kU;
      hy4_load_kv_frag_tg<half, KV_STRIDE>(v_reg0[k], s_kv, col_off, fm, fn);
      hy4_load_kv_frag_tg<half, KV_STRIDE>(
          v_reg1[k], s_kv + kU * KV_STRIDE, col_off, fm, fn);
      BaseNAXFrag::mma(
          Stile.frag_at(0, 0),
          Stile.frag_at(0, 1),
          Q_v[k],
          metal::false_type{},
          v_reg0[k],
          v_reg1[k],
          metal::true_type{});
    }
    hfrag_t kr0, kr1;
    const int r_off = (D_V - HALF_DK) + (d_group - 4) * kU;
    hy4_load_kv_frag_tg<half, KV_STRIDE>(kr0, s_kv, r_off, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        kr1, s_kv + kU * KV_STRIDE, r_off, fm, fn);
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Q_rope,
        metal::false_type{},
        kr0,
        kr1,
        metal::true_type{});
    const int base = (d_group - 4) * XCHG_STRIDE + int(simd_lid);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      const thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; ++i) {
        const int off = base + (int(ik) * kEPF + int(i)) * 32;
        s_xchg[off] = s_xchg[off] + sf[i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Deterministic binary-tree sum across WN=8: ((dg0+dg4)+(dg1+dg5))+((dg2+dg6)+(dg3+dg7)).
  {
    const int lid = int(simd_lid);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; ++i) {
        const int off = (int(ik) * kEPF + int(i)) * 32 + lid;
        const float a = s_xchg[0 * XCHG_STRIDE + off];
        const float b = s_xchg[1 * XCHG_STRIDE + off];
        const float c = s_xchg[2 * XCHG_STRIDE + off];
        const float d = s_xchg[3 * XCHG_STRIDE + off];
        sf[i] = (a + b) + (c + d);
      }
    }
  }

  if (tile_all_valid) {
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
      Stile.elems()[ii] *= scale2;
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < 2; ++ii) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 4; ++jj) {
          const short loc = ii * 4 + jj;
          sf[loc] = col_valid[ik][jj] ? (sf[loc] * scale2) : MAX_FLOOR;
        }
      }
    }
  }

  metal::vec<float, 2> new_max = max_score;
  Stile.template row_reduce<Hy4MaxOp>(new_max);

  if (tile_all_valid) {
    Stile.template row_bin_op<Hy4Exp2SubOp>(new_max);
  } else {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < 2; ++ii) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 4; ++jj) {
          const short loc = ii * 4 + jj;
          sf[loc] = col_valid[ik][jj] ? metal::exp2(sf[loc] - new_max[ii]) : 0.0f;
        }
      }
    }
  }

  metal::vec<float, 2> factor;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 2; ++i) {
    factor[i] = metal::exp2(max_score[i] - new_max[i]);
    max_score[i] = new_max[i];
    sum_score[i] *= factor[i];
  }
  Stile.template row_reduce<Hy4SumOp>(sum_score);
  Otile.template row_bin_op<Hy4MulOp>(factor);

  hfrag_t p_h[TK];
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    const thread auto& pf = Stile.frag_at(0, ik);
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kEPF; ++e) {
      p_h[ik][e] = half(pf[e]);
    }
  }

  BaseNAXFrag::mma(
      Otile.frag_at(0, 0),
      Otile.frag_at(0, 1),
      p_h[0],
      metal::false_type{},
      v_reg0[0],
      v_reg0[1],
      metal::false_type{});
  BaseNAXFrag::mma(
      Otile.frag_at(0, 2),
      Otile.frag_at(0, 3),
      p_h[0],
      metal::false_type{},
      v_reg0[2],
      v_reg0[3],
      metal::false_type{});
  BaseNAXFrag::mma(
      Otile.frag_at(0, 0),
      Otile.frag_at(0, 1),
      p_h[1],
      metal::false_type{},
      v_reg1[0],
      v_reg1[1],
      metal::false_type{});
  BaseNAXFrag::mma(
      Otile.frag_at(0, 2),
      Otile.frag_at(0, 3),
      p_h[1],
      metal::false_type{},
      v_reg1[2],
      v_reg1[3],
      metal::false_type{});
}

const size_t mp_base_a =
    (size_t(s) * size_t(H) + size_t(head_base + int(fm))) * size_t(n_splits) +
    size_t(sp);
const size_t mp_base_b =
    (size_t(s) * size_t(H) + size_t(head_base + int(fm) + 8)) * size_t(n_splits) +
    size_t(sp);
device float* out_a = o_part + mp_base_a * size_t(D_V);
device float* out_b = o_part + mp_base_b * size_t(D_V);
const bool nz_a = (sum_score[0] > 0.0f);
const bool nz_b = (sum_score[1] > 0.0f);

if constexpr (SINGLE_SPLIT) {
  // Match mla_decode_merge's single-active-split formula bit-for-bit so
  // trailing -1 padding crossing k_split preserves exact bitwise equality.
  const float max_nat_a = nz_a ? (max_score[0] * M_LN2_F) : MAX_FLOOR;
  const float gm_a = nz_a ? metal::max(MAX_FLOOR, max_nat_a) : MAX_FLOOR;
  const float sink_a = sinks[head_base + int(fm)];
  const float m_all_a = metal::max(gm_a, sink_a);
  const float k_scale_a = metal::exp(gm_a - m_all_a);
  const float s_term_a = metal::exp(sink_a - m_all_a);
  const float den_a = sum_score[0] * k_scale_a + s_term_a;
  const bool ok_a = nz_a && (den_a > 0.0f);

  const float max_nat_b = nz_b ? (max_score[1] * M_LN2_F) : MAX_FLOOR;
  const float gm_b = nz_b ? metal::max(MAX_FLOOR, max_nat_b) : MAX_FLOOR;
  const float sink_b = sinks[head_base + int(fm) + 8];
  const float m_all_b = metal::max(gm_b, sink_b);
  const float k_scale_b = metal::exp(gm_b - m_all_b);
  const float s_term_b = metal::exp(sink_b - m_all_b);
  const float den_b = sum_score[1] * k_scale_b + s_term_b;
  const bool ok_b = nz_b && (den_b > 0.0f);

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < TDh; ++i) {
    const int col = v_base + int(i) * kU + int(fn);
    const thread auto& of = Otile.frag_at(0, i);
    *reinterpret_cast<device float4*>(out_a + col) =
        ok_a ? ((float4(of[0], of[1], of[2], of[3]) * k_scale_a) / den_a)
             : float4(0.0f);
    *reinterpret_cast<device float4*>(out_b + col) =
        ok_b ? ((float4(of[4], of[5], of[6], of[7]) * k_scale_b) / den_b)
             : float4(0.0f);
  }
} else {
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < TDh; ++i) {
    const int col = v_base + int(i) * kU + int(fn);
    const thread auto& of = Otile.frag_at(0, i);
    *reinterpret_cast<device float4*>(out_a + col) =
        nz_a ? float4(of[0], of[1], of[2], of[3]) : float4(0.0f);
    *reinterpret_cast<device float4*>(out_b + col) =
        nz_b ? float4(of[4], of[5], of[6], of[7]) : float4(0.0f);
  }

  // fn == 0 selects 8 lanes covering fm = 0..7 bijectively.
  if (d_group == 0 && fn == 0) {
    max_part[mp_base_a] = nz_a ? (max_score[0] * M_LN2_F) : MAX_FLOOR;
    sum_part[mp_base_a] = sum_score[0];
    max_part[mp_base_b] = nz_b ? (max_score[1] * M_LN2_F) : MAX_FLOOR;
    sum_part[mp_base_b] = sum_score[1];
  }
}
)METAL"
