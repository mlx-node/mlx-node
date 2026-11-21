R"METAL(
constexpr int D_K = D_V + D_ROPE;
constexpr int HALF = D_K / 2;
constexpr short kU = 16;
constexpr int BH = WM * kU;
constexpr int BDh = D_V / WN;
constexpr int BRh = D_ROPE / WN;
constexpr int TDh = BDh / kU;
constexpr int TRh = BRh / kU;
constexpr int KV_STRIDE = D_K + 8;
constexpr int VEC8_PER_ROW = D_K / 8;
constexpr int VEC8_HALF = HALF / 8;
constexpr float MAX_FLOOR = -1e30f;

const int S = q_shape[0];
const int H = q_shape[1];
const int KMAX = idx_shape[1];
const int N_KEYS = k0_shape[0];

const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
const uint tid = simd_gid * 32u + simd_lid;
const int row_group = int(simd_gid) / WN;
const int d_group = int(simd_gid) % WN;

const int first_head = int(threadgroup_position_in_grid.y) * BH;
const int head_base = first_head + row_group * kU;
const int s = int(threadgroup_position_in_grid.z);
if (s >= S || head_base >= H) {
  return;
}

using otile_t = NAXTile<float, 1, TDh>;
using stile_t = NAXTile<float, 1, 1>;
using qfrag_t = typename BaseNAXFrag::template dtype_frag_t<C>;
constexpr short kEPF = BaseNAXFrag::kElemsPerFrag;

alignas(16) threadgroup half s_kv[BK * KV_STRIDE];
threadgroup float s_xchg[WM][WN][kEPF * 32];
threadgroup int s_slots[BK];

const short2 coord = BaseNAXFrag::get_coord();
const short fm = coord.y;
const short fn = coord.x;

const int v_col_base = d_group * BDh;
const int r_col_base = D_V + d_group * BRh;

const device float* q_base =
    q + (size_t(s) * size_t(H) + size_t(head_base)) * size_t(D_K);

NAXTile<C, 1, 1> Qv[TDh];
STEEL_PRAGMA_UNROLL
for (short id = 0; id < TDh; ++id) {
  Qv[id].load(q_base + v_col_base + int(id) * kU, D_K);
}

NAXTile<C, 1, 1> Qr[TRh];
STEEL_PRAGMA_UNROLL
for (short ir = 0; ir < TRh; ++ir) {
  Qr[ir].load(q_base + r_col_base + int(ir) * kU, D_K);
}

otile_t Otile;
Otile.clear();

metal::vec<float, 2> max_score(MAX_FLOOR);
metal::vec<float, 2> sum_score(0.0f);

const float scale2 = scale[0] * M_LOG2E_F;
const size_t idx_base = size_t(s) * size_t(KMAX);

for (int k_base = 0; k_base < KMAX; k_base += BK) {
  if (tid < uint(BK)) {
    const int kp = k_base + int(tid);
    const int raw = (kp < KMAX) ? idx[idx_base + size_t(kp)] : -1;
    s_slots[tid] = (raw >= 0 && raw < N_KEYS) ? raw : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  bool col_valid[4];
  bool lane_any = false;
  bool lane_all = true;
  STEEL_PRAGMA_UNROLL
  for (short jj = 0; jj < 4; ++jj) {
    const bool ok = s_slots[int(fn) + int(jj)] >= 0;
    col_valid[jj] = ok;
    lane_any = lane_any || ok;
    lane_all = lane_all && ok;
  }
  // Every simdgroup covers all BK=16 columns identically across its 32 lanes,
  // so simd_any is uniform across the threadgroup before any barrier.
  if (!simd_any(lane_any)) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    continue;
  }
  const bool tile_all_valid = simd_all(lane_all);

  for (int idx8 = int(tid); idx8 < BK * VEC8_PER_ROW; idx8 += WM * WN * 32) {
    const int row = idx8 / VEC8_PER_ROW;
    const int col8 = idx8 % VEC8_PER_ROW;
    const int slot = s_slots[row];
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_kv + row * KV_STRIDE + col8 * 8);
    if (slot >= 0) {
      if constexpr (SPLIT) {
        const device half* plane = (col8 >= VEC8_HALF) ? k1 : k0;
        const int pcol8 = (col8 >= VEC8_HALF) ? (col8 - VEC8_HALF) : col8;
        *dst = *reinterpret_cast<const device float4*>(
            plane + size_t(slot) * size_t(HALF) + size_t(pcol8 * 8));
      } else {
        *dst = *reinterpret_cast<const device float4*>(
            k0 + size_t(slot) * size_t(D_K) + size_t(col8 * 8));
      }
    } else {
      *dst = float4(0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  stile_t Stile;
  Stile.clear();

#pragma clang loop unroll_count(2)
  for (short id = 0; id < TDh; ++id) {
    qfrag_t kf;
    hy4_load_kv_frag_tg<C, KV_STRIDE>(
        kf, s_kv, v_col_base + int(id) * kU, fm, fn);
    // Cached K is FP16 (exact in TF32); splitting F32 Q into top-10-bit TF32
    // + residual recovers FP32 score accuracy with zero extra memory loads.
    if constexpr (metal::is_same_v<C, float>) {
      qfrag_t qhi, qlo;
      const thread auto& qf = Qv[id].frag_at(0, 0);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kEPF; ++e) {
        const float v = qf[e];
        const float hi = as_type<float>(as_type<uint>(v) & 0xffffe000u);
        qhi[e] = hi;
        qlo[e] = v - hi;
      }
      hy4_nax_mma_16x16x32(
          Stile.frag_at(0, 0),
          qhi,
          qlo,
          metal::false_type{},
          kf,
          kf,
          metal::true_type{});
    } else {
      hy4_nax_mma_16x16x16(
          Stile.frag_at(0, 0),
          Qv[id].frag_at(0, 0),
          metal::false_type{},
          kf,
          metal::true_type{});
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short ir = 0; ir < TRh; ++ir) {
    qfrag_t kf;
    hy4_load_kv_frag_tg<C, KV_STRIDE>(
        kf, s_kv, r_col_base + int(ir) * kU, fm, fn);
    if constexpr (metal::is_same_v<C, float>) {
      qfrag_t qhi, qlo;
      const thread auto& qf = Qr[ir].frag_at(0, 0);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kEPF; ++e) {
        const float v = qf[e];
        const float hi = as_type<float>(as_type<uint>(v) & 0xffffe000u);
        qhi[e] = hi;
        qlo[e] = v - hi;
      }
      hy4_nax_mma_16x16x32(
          Stile.frag_at(0, 0),
          qhi,
          qlo,
          metal::false_type{},
          kf,
          kf,
          metal::true_type{});
    } else {
      hy4_nax_mma_16x16x16(
          Stile.frag_at(0, 0),
          Qr[ir].frag_at(0, 0),
          metal::false_type{},
          kf,
          metal::true_type{});
    }
  }

  // Exchange partial Q@K^T scores across D-groups in fixed order 0..WN-1 so
  // every D-group obtains identical softmax weights.
  threadgroup float* xslot = s_xchg[row_group][d_group];
  const short xbase = short(simd_lid) * kEPF;
  {
    thread auto& sf = Stile.frag_at(0, 0);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kEPF; ++i) {
      xslot[xbase + i] = sf[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  {
    thread auto& sf = Stile.frag_at(0, 0);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kEPF; ++i) {
      float acc = s_xchg[row_group][0][xbase + i];
      STEEL_PRAGMA_UNROLL
      for (short peer = 1; peer < WN; ++peer) {
        acc += s_xchg[row_group][peer][xbase + i];
      }
      sf[i] = acc;
    }

    if (tile_all_valid) {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < kEPF; ++ii) {
        sf[ii] *= scale2;
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < 2; ++ii) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 4; ++jj) {
          const short loc = ii * 4 + jj;
          sf[loc] = col_valid[jj] ? (sf[loc] * scale2) : MAX_FLOOR;
        }
      }
    }

    metal::vec<float, 2> new_max = max_score;
    Stile.template row_reduce<Hy4MaxOp>(new_max);

    if (tile_all_valid) {
      Stile.template row_bin_op<Hy4Exp2SubOp>(new_max);
    } else {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < 2; ++ii) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 4; ++jj) {
          const short loc = ii * 4 + jj;
          sf[loc] = col_valid[jj] ? metal::exp2(sf[loc] - new_max[ii]) : 0.0f;
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

    qfrag_t p_hi;
    qfrag_t p_lo;
    if constexpr (metal::is_same_v<C, float>) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kEPF; ++e) {
        const float v = sf[e];
        const float hi = as_type<float>(as_type<uint>(v) & 0xffffe000u);
        p_hi[e] = hi;
        p_lo[e] = v - hi;
      }
    } else {
      p_hi = qfrag_t(sf);
    }

    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      const int col0 = v_col_base + int(id) * kU;
      const int col1 = col0 + kU;
      qfrag_t vf0, vf1;
      hy4_load_kv_frag_tg<C, KV_STRIDE>(vf0, s_kv, col0, fm, fn);
      hy4_load_kv_frag_tg<C, KV_STRIDE>(vf1, s_kv, col1, fm, fn);
      BaseNAXFrag::mma(
          Otile.frag_at(0, id),
          Otile.frag_at(0, id + 1),
          p_hi,
          metal::false_type{},
          vf0,
          vf1,
          metal::false_type{});
      // Cached V is FP16 (exact in TF32); split F32 P into TF32 hi + residual.
      if constexpr (metal::is_same_v<C, float>) {
        BaseNAXFrag::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            p_lo,
            metal::false_type{},
            vf0,
            vf1,
            metal::false_type{});
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

metal::vec<float, 2> out_scale;
STEEL_PRAGMA_UNROLL
for (short i = 0; i < 2; ++i) {
  const int h_idx = head_base + int(fm) + int(i) * 8;
  const float sink_log2 = sinks[h_idx] * M_LOG2E_F;
  const float m_h = max_score[i];
  const float l_h = sum_score[i];
  const float m_all = metal::max(m_h, sink_log2);
  const float k_scale = metal::exp2(m_h - m_all);
  const float s_term = metal::exp2(sink_log2 - m_all);
  const float den = metal::fma(l_h, k_scale, s_term);
  out_scale[i] = (l_h > 0.0f && den > 0.0f) ? (k_scale / den) : 0.0f;
}
Otile.template row_bin_op<Hy4MulOp>(out_scale);

device float* out_base =
    out + (size_t(s) * size_t(H) + size_t(head_base)) * size_t(D_V) +
    size_t(v_col_base);
Otile.store(out_base, D_V);
)METAL"
