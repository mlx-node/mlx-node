R"METAL(
constexpr int D_K = D_V + D_ROPE;
constexpr int HALF = D_K / 2;
constexpr short kU = 16;
constexpr int BH = WM * kU;
constexpr int BDh = D_V / WN;
constexpr int BRh = D_ROPE / WN;
constexpr int TDh = BDh / kU;
constexpr int TRh = BRh / kU;
constexpr int TK = BK / kU;
constexpr float MAX_FLOOR = -1e30f;

const int S = q_shape[0];
const int H = q_shape[1];
const int KMAX = idx_shape[1];
const int N_KEYS = k0_shape[0];

const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
const int row_group = int(simd_gid) / WN;
const int d_group = int(simd_gid) % WN;

const int first_head = int(threadgroup_position_in_grid.y) * BH;
const int head_base = first_head + row_group * kU;
const int s = int(threadgroup_position_in_grid.z);
if (s >= S || head_base >= H) {
  return;
}

using otile_t = NAXTile<float, 1, TDh>;
using stile_t = NAXTile<float, 1, TK>;
using qfrag_t = typename BaseNAXFrag::template dtype_frag_t<C>;
constexpr short kEPF = BaseNAXFrag::kElemsPerFrag;

threadgroup float s_xchg[WM][WN][TK * kEPF * 32];

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
  bool col_valid[TK][4];
  bool lane_any = false;
  bool lane_all = true;
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < 4; ++jj) {
      const int kp = k_base + int(ik) * kU + int(fn) + int(jj);
      const int raw = (kp < KMAX) ? idx[idx_base + size_t(kp)] : -1;
      const bool ok = (raw >= 0 && raw < N_KEYS);
      col_valid[ik][jj] = ok;
      lane_any = lane_any || ok;
      lane_all = lane_all && ok;
    }
  }
  // Every simdgroup covers all BK columns identically across its 32 lanes, so
  // simd_any is uniform across the threadgroup before any barrier.
  if (!simd_any(lane_any)) {
    continue;
  }
  const bool tile_all_valid = simd_all(lane_all);

  int k_slots[TK][2];
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    const int kpa = k_base + int(ik) * kU + int(fm);
    const int kpb = kpa + 8;
    const int ra = (kpa < KMAX) ? idx[idx_base + size_t(kpa)] : -1;
    const int rb = (kpb < KMAX) ? idx[idx_base + size_t(kpb)] : -1;
    k_slots[ik][0] = (ra >= 0 && ra < N_KEYS) ? ra : -1;
    k_slots[ik][1] = (rb >= 0 && rb < N_KEYS) ? rb : -1;
  }

  stile_t Stile;
  Stile.clear();

  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ik += 2) {
    const int s0a = k_slots[ik][0];
    const int s0b = k_slots[ik][1];
    const int s1a = k_slots[ik + 1][0];
    const int s1b = k_slots[ik + 1][1];

#pragma clang loop unroll_count(2)
    for (short id = 0; id < TDh; ++id) {
      const int col_off = v_col_base + int(id) * kU;
      qfrag_t kf0, kf1;
      hy4_load_kv_frag<C, SPLIT, D_K, HALF>(kf0, k0, k1, s0a, s0b, col_off, fn);
      hy4_load_kv_frag<C, SPLIT, D_K, HALF>(kf1, k0, k1, s1a, s1b, col_off, fn);
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
        BaseNAXFrag::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            qhi,
            metal::false_type{},
            kf0,
            kf1,
            metal::true_type{});
        BaseNAXFrag::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            qlo,
            metal::false_type{},
            kf0,
            kf1,
            metal::true_type{});
      } else {
        BaseNAXFrag::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qv[id].frag_at(0, 0),
            metal::false_type{},
            kf0,
            kf1,
            metal::true_type{});
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ir = 0; ir < TRh; ++ir) {
      const int col_off = r_col_base + int(ir) * kU;
      qfrag_t kf0, kf1;
      hy4_load_kv_frag<C, SPLIT, D_K, HALF>(kf0, k0, k1, s0a, s0b, col_off, fn);
      hy4_load_kv_frag<C, SPLIT, D_K, HALF>(kf1, k0, k1, s1a, s1b, col_off, fn);
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
        BaseNAXFrag::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            qhi,
            metal::false_type{},
            kf0,
            kf1,
            metal::true_type{});
        BaseNAXFrag::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            qlo,
            metal::false_type{},
            kf0,
            kf1,
            metal::true_type{});
      } else {
        BaseNAXFrag::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qr[ir].frag_at(0, 0),
            metal::false_type{},
            kf0,
            kf1,
            metal::true_type{});
      }
    }
  }

  // Exchange partial Q@K^T scores across D-groups in fixed order 0..WN-1 so
  // every D-group obtains identical softmax weights.
  threadgroup float* xslot = s_xchg[row_group][d_group];
  const short xbase = short(simd_lid) * (TK * kEPF);
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    thread auto& sf = Stile.frag_at(0, ik);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kEPF; ++i) {
      xslot[xbase + ik * kEPF + i] = sf[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    thread auto& sf = Stile.frag_at(0, ik);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kEPF; ++i) {
      float acc = s_xchg[row_group][0][xbase + ik * kEPF + i];
      STEEL_PRAGMA_UNROLL
      for (short peer = 1; peer < WN; ++peer) {
        acc += s_xchg[row_group][peer][xbase + ik * kEPF + i];
      }
      sf[i] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

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

  qfrag_t p_hi[TK];
  qfrag_t p_lo[TK];
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    if constexpr (metal::is_same_v<C, float>) {
      const thread auto& pf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kEPF; ++e) {
        const float v = pf[e];
        const float hi = as_type<float>(as_type<uint>(v) & 0xffffe000u);
        p_hi[ik][e] = hi;
        p_lo[ik][e] = v - hi;
      }
    } else {
      p_hi[ik] = qfrag_t(Stile.frag_at(0, ik));
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id += 2) {
    const int col0 = v_col_base + int(id) * kU;
    const int col1 = col0 + kU;
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      qfrag_t vf0, vf1;
      hy4_load_kv_frag<C, SPLIT, D_K, HALF>(
          vf0, k0, k1, k_slots[ik][0], k_slots[ik][1], col0, fn);
      hy4_load_kv_frag<C, SPLIT, D_K, HALF>(
          vf1, k0, k1, k_slots[ik][0], k_slots[ik][1], col1, fn);
      BaseNAXFrag::mma(
          Otile.frag_at(0, id),
          Otile.frag_at(0, id + 1),
          p_hi[ik],
          metal::false_type{},
          vf0,
          vf1,
          metal::false_type{});
      // Cached V is FP16 (exact in TF32); split F32 P into TF32 hi + residual.
      if constexpr (metal::is_same_v<C, float>) {
        BaseNAXFrag::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            p_lo[ik],
            metal::false_type{},
            vf0,
            vf1,
            metal::false_type{});
      }
    }
  }
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
