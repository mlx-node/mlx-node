R"METAL(
constexpr int D_V = 512;
constexpr int D_ROPE = 64;
constexpr int D_K = D_V + D_ROPE;
constexpr int HALF_DK = D_K / 2;
constexpr int TR = 32;
constexpr short kU = 16;
constexpr int WN = 4;
constexpr int NT = WN * 32;
constexpr int TK = TR / kU;
constexpr int KV_STRIDE = HALF_DK + 8;
constexpr int HALF_DK8 = HALF_DK / 8;
constexpr int TDh = D_V / (WN * kU);
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
const int head_group = y_id / n_splits;
const int sp = y_id - head_group * n_splits;
const int head_base = head_group * int(kU);
const int s = int(threadgroup_position_in_grid.z);
if (s >= S || head_base >= H) {
  return;
}

using otile_t = NAXTile<float, 1, TDh>;
using stile_t = NAXTile<float, 1, TK>;
using hfrag_t = typename BaseNAXFrag::template dtype_frag_t<half>;
constexpr short kEPF = BaseNAXFrag::kElemsPerFrag;

// 32 x 296 half staging (18.5 KiB) + separate 2-slot score reduction (4 KiB)
// keeps TG <= 23 KiB and preserves cols 288..512 in s_kv for P @ V.
alignas(16) threadgroup uint s_buf[(TR * KV_STRIDE) / 2];
threadgroup half* s_kv = reinterpret_cast<threadgroup half*>(s_buf);
threadgroup float s_xchg[2 * XCHG_STRIDE];
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

// Partition the 36 16-col chunks of D_K=576 evenly across WN=4 (9 chunks/SG):
// 4 from [0,256), 4 from [288,544), plus 1 extra from {256, 272} (d_group < 2)
// or {544, 560} (d_group >= 2).
hfrag_t Q0[4];
hfrag_t Q1[4];
STEEL_PRAGMA_UNROLL
for (short k = 0; k < 4; ++k) {
  Q0[k] = load_q_half(d_group * 64 + int(k) * kU);
  Q1[k] = load_q_half(HALF_DK + d_group * 64 + int(k) * kU);
}
const int extra_kv_off = 256 + (d_group & 1) * kU;
const int extra_q_col = (d_group < 2) ? extra_kv_off : (HALF_DK + extra_kv_off);
const hfrag_t Q_extra = load_q_half(extra_q_col);

otile_t Otile;
Otile.clear();

metal::vec<float, 2> max_score(MAX_FLOOR);
metal::vec<float, 2> sum_score(0.0f);

const float scale2 = scale[0] * M_LOG2E_F;
const int k_start = sp * k_split;
const int k_end =
    (sp + 1 < n_splits) ? metal::min(k_start + k_split, KMAX) : KMAX;
const size_t idx_base = size_t(s) * size_t(KMAX);
const int v_base = d_group * (D_V / WN);

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

  STEEL_PRAGMA_UNROLL
  for (short k = 0; k < 4; ++k) {
    const int col_off = d_group * 64 + int(k) * kU;
    hfrag_t kf0, kf1;
    hy4_load_kv_frag_tg<half, KV_STRIDE>(kf0, s_kv, col_off, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        kf1, s_kv + kU * KV_STRIDE, col_off, fm, fn);
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Q0[k],
        metal::false_type{},
        kf0,
        kf1,
        metal::true_type{});
  }
  if (d_group < 2) {
    hfrag_t kf0, kf1;
    hy4_load_kv_frag_tg<half, KV_STRIDE>(kf0, s_kv, extra_kv_off, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        kf1, s_kv + kU * KV_STRIDE, extra_kv_off, fm, fn);
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Q_extra,
        metal::false_type{},
        kf0,
        kf1,
        metal::true_type{});
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

  STEEL_PRAGMA_UNROLL
  for (short k = 0; k < 4; ++k) {
    const int col_off = d_group * 64 + int(k) * kU;
    hfrag_t kf0, kf1;
    hy4_load_kv_frag_tg<half, KV_STRIDE>(kf0, s_kv, col_off, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        kf1, s_kv + kU * KV_STRIDE, col_off, fm, fn);
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Q1[k],
        metal::false_type{},
        kf0,
        kf1,
        metal::true_type{});
  }
  if (d_group >= 2) {
    hfrag_t kf0, kf1;
    hy4_load_kv_frag_tg<half, KV_STRIDE>(kf0, s_kv, extra_kv_off, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        kf1, s_kv + kU * KV_STRIDE, extra_kv_off, fm, fn);
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Q_extra,
        metal::false_type{},
        kf0,
        kf1,
        metal::true_type{});
  }

  // Deterministic 2-slot tree reduction across WN=4: (dg0+dg2)+(dg1+dg3).
  const int xbase0 = int(simd_lid) * (TK * kEPF);
  const int xbase1 = xbase0 + XCHG_STRIDE;
  const int my_xbase = ((d_group & 1) == 0) ? xbase0 : xbase1;
  if (d_group >= 2) {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      const thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; ++i) {
        s_xchg[my_xbase + int(ik) * kEPF + i] = sf[i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (d_group < 2) {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      const thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; ++i) {
        s_xchg[my_xbase + int(ik) * kEPF + i] =
            sf[i] + s_xchg[my_xbase + int(ik) * kEPF + i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    thread auto& sf = Stile.frag_at(0, ik);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kEPF; ++i) {
      sf[i] = s_xchg[xbase0 + int(ik) * kEPF + i] +
          s_xchg[xbase1 + int(ik) * kEPF + i];
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

  const int v_slot0a = s_slots[int(fm)];
  const int v_slot0b = s_slots[int(fm) + 8];
  const int v_slot1a = s_slots[kU + int(fm)];
  const int v_slot1b = s_slots[kU + int(fm) + 8];

  STEEL_PRAGMA_UNROLL
  for (short p = 0; p < TDh / 2; ++p) {
    const int col0 = v_base + int(p) * 32;
    const int col1 = col0 + kU;
    const short id0 = p * 2;
    const short id1 = id0 + 1;
    hfrag_t vf0, vf1;
    if (col0 >= HALF_DK) {
      const int tc0 = col0 - HALF_DK;
      const int tc1 = col1 - HALF_DK;
      hy4_load_kv_frag_tg<half, KV_STRIDE>(vf0, s_kv, tc0, fm, fn);
      hy4_load_kv_frag_tg<half, KV_STRIDE>(vf1, s_kv, tc1, fm, fn);
      BaseNAXFrag::mma(
          Otile.frag_at(0, id0),
          Otile.frag_at(0, id1),
          p_h[0],
          metal::false_type{},
          vf0,
          vf1,
          metal::false_type{});

      hy4_load_kv_frag_tg<half, KV_STRIDE>(
          vf0, s_kv + kU * KV_STRIDE, tc0, fm, fn);
      hy4_load_kv_frag_tg<half, KV_STRIDE>(
          vf1, s_kv + kU * KV_STRIDE, tc1, fm, fn);
      BaseNAXFrag::mma(
          Otile.frag_at(0, id0),
          Otile.frag_at(0, id1),
          p_h[1],
          metal::false_type{},
          vf0,
          vf1,
          metal::false_type{});
    } else {
      hy4_load_kv_frag_dev<SPLIT, D_K, HALF_DK>(
          vf0, k0, k1, v_slot0a, v_slot0b, col0, fn);
      hy4_load_kv_frag_dev<SPLIT, D_K, HALF_DK>(
          vf1, k0, k1, v_slot0a, v_slot0b, col1, fn);
      BaseNAXFrag::mma(
          Otile.frag_at(0, id0),
          Otile.frag_at(0, id1),
          p_h[0],
          metal::false_type{},
          vf0,
          vf1,
          metal::false_type{});

      hy4_load_kv_frag_dev<SPLIT, D_K, HALF_DK>(
          vf0, k0, k1, v_slot1a, v_slot1b, col0, fn);
      hy4_load_kv_frag_dev<SPLIT, D_K, HALF_DK>(
          vf1, k0, k1, v_slot1a, v_slot1b, col1, fn);
      BaseNAXFrag::mma(
          Otile.frag_at(0, id0),
          Otile.frag_at(0, id1),
          p_h[1],
          metal::false_type{},
          vf0,
          vf1,
          metal::false_type{});
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
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
// Convert log2 max_score to natural-log units for mla_decode_merge.
if (d_group == 0 && fn == 0) {
  max_part[mp_base_a] = nz_a ? (max_score[0] * M_LN2_F) : MAX_FLOOR;
  sum_part[mp_base_a] = sum_score[0];
  max_part[mp_base_b] = nz_b ? (max_score[1] * M_LN2_F) : MAX_FLOOR;
  sum_part[mp_base_b] = sum_score[1];
}
)METAL"
