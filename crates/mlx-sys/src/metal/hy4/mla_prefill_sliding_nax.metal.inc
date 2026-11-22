R"METAL(
constexpr int D_K = D_V + D_ROPE;
constexpr short kU = 16;
constexpr int BQ = kU;
constexpr int BH = WM;
constexpr int BK = 32;
constexpr int TK = BK / kU;
constexpr int D_CHUNK = 256;
constexpr int ROPE_STRIDE = D_ROPE + 8;
constexpr int VEC8_STRIP = D_CHUNK / 8;
constexpr int VEC8_ROPE = D_ROPE / 8;
constexpr int N_STRIPS = D_V / D_CHUNK;
constexpr int SUBS_PER_STRIP = D_CHUNK / (WN * kU);
constexpr int TDh = D_V / (WN * kU);
constexpr int TRh = D_ROPE / (WN * kU);
constexpr int XCHG_STRIDE = TK * BaseNAXFrag::kElemsPerFrag * 32;
constexpr float MAX_FLOOR = -1e30f;
constexpr bool USE_Q_LO = !FAST_1MMA && !Q_F16;

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

// Each TG shares loaded K/V strips across BH = WM heads for the same 16 query tokens.
const int q_tile_base = int(threadgroup_position_in_grid.y) * BQ;
const int head = int(threadgroup_position_in_grid.z) * BH + row_group;
if (q_tile_base >= S) {
  return;
}

using otile_t = NAXTile<float, 1, TDh>;
using stile_t = NAXTile<float, 1, TK>;
using hfrag_t = typename BaseNAXFrag::template dtype_frag_t<half>;
constexpr short kEPF = BaseNAXFrag::kElemsPerFrag;

// Dedicated s_kv (16,384 B) + s_rope (4,608 B) + s_xchg (8,192 B for WM=2) = 29,184 B <= 32 KiB.
// Keeps s_rope disjoint from s_xchg so reduction needs no WAR barrier against RoPE reads.
alignas(16) threadgroup half s_kv[BK * D_CHUNK];
alignas(16) threadgroup half s_rope[BK * ROPE_STRIDE];
alignas(16) threadgroup float s_xchg[WM * 2 * XCHG_STRIDE];

const short2 coord = BaseNAXFrag::get_coord();
const short fm = coord.y;
const short fn = coord.x;

const short valid_q = (head < H)
    ? short(metal::max(0, metal::min(int(kU), S - q_tile_base)))
    : short(0);

const int q_v_dim = SPLIT_Q ? D_V : D_K;
const int q_v_stride = H * q_v_dim;
const device Q_T* qv_base =
    q + (size_t(q_tile_base) * size_t(H) + size_t(metal::min(head, H - 1))) *
        size_t(q_v_dim);

const int q_r_dim = SPLIT_Q ? D_ROPE : D_K;
const int q_r_stride = H * q_r_dim;
const device Q_T* qr_base = SPLIT_Q
    ? (q_pe +
       (size_t(q_tile_base) * size_t(H) + size_t(metal::min(head, H - 1))) *
           size_t(D_ROPE))
    : (qv_base + D_V);

hfrag_t Qv_hi[TDh];
hfrag_t Qv_lo[USE_Q_LO ? TDh : 1];
STEEL_PRAGMA_UNROLL
for (short strip = 0; strip < N_STRIPS; ++strip) {
  STEEL_PRAGMA_UNROLL
  for (short sub = 0; sub < SUBS_PER_STRIP; ++sub) {
    const int col =
        int(strip) * D_CHUNK + d_group * (D_CHUNK / WN) + int(sub) * kU;
    const short id = strip * SUBS_PER_STRIP + sub;
    if constexpr (USE_Q_LO) {
      NAXTile<float, 1, 1> qt;
      if (valid_q == kU) {
        qt.load(qv_base + col, q_v_stride);
      } else if (valid_q > 0) {
        qt.load_rows(qv_base + col, q_v_stride, valid_q);
      } else {
        qt.clear();
      }
      const thread auto& qf = qt.frag_at(0, 0);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kEPF; ++e) {
        const float v = qf[e];
        const half hi = half(v);
        Qv_hi[id][e] = hi;
        Qv_lo[id][e] = half(v - float(hi));
      }
    } else {
      NAXTile<half, 1, 1> qt;
      if (valid_q == kU) {
        qt.load(qv_base + col, q_v_stride);
      } else if (valid_q > 0) {
        qt.load_rows(qv_base + col, q_v_stride, valid_q);
      } else {
        qt.clear();
      }
      Qv_hi[id] = qt.frag_at(0, 0);
    }
  }
}

hfrag_t Qr_hi[TRh];
hfrag_t Qr_lo[USE_Q_LO ? TRh : 1];
STEEL_PRAGMA_UNROLL
for (short ir = 0; ir < TRh; ++ir) {
  const int col = d_group * (D_ROPE / WN) + int(ir) * kU;
  if constexpr (USE_Q_LO) {
    NAXTile<float, 1, 1> qt;
    if (valid_q == kU) {
      qt.load(qr_base + col, q_r_stride);
    } else if (valid_q > 0) {
      qt.load_rows(qr_base + col, q_r_stride, valid_q);
    } else {
      qt.clear();
    }
    const thread auto& qf = qt.frag_at(0, 0);
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kEPF; ++e) {
      const float v = qf[e];
      const half hi = half(v);
      Qr_hi[ir][e] = hi;
      Qr_lo[ir][e] = half(v - float(hi));
    }
  } else {
    NAXTile<half, 1, 1> qt;
    if (valid_q == kU) {
      qt.load(qr_base + col, q_r_stride);
    } else if (valid_q > 0) {
      qt.load_rows(qr_base + col, q_r_stride, valid_q);
    } else {
      qt.clear();
    }
    Qr_hi[ir] = qt.frag_at(0, 0);
  }
}

otile_t Otile;
Otile.clear();

metal::vec<float, 2> max_score(MAX_FLOOR);
metal::vec<float, 2> sum_score(0.0f);

const float scale2 = scale[0] * M_LOG2E_F;

const int q_tg_lo = start_pos + q_tile_base;
const int q_tg_hi = start_pos + metal::min(S, q_tile_base + BQ) - 1;
const int k_lo = metal::max(first_key, q_tg_lo + 1 - window_size);
const int k_hi = q_tg_hi;

const int q_pos0 = start_pos + q_tile_base + int(fm);
const int q_pos1 = q_pos0 + 8;
const bool q_ok0 = (head < H) && ((q_tile_base + int(fm)) < S);
const bool q_ok1 = (head < H) && ((q_tile_base + int(fm) + 8) < S);

constexpr int NUM_THREADS = WM * WN * 32;
constexpr int ROWS_PER_PASS = NUM_THREADS / VEC8_STRIP;
constexpr int STRIP_PASSES = BK / ROWS_PER_PASS;
const int strip_row = int(tid) / VEC8_STRIP;
const int strip_c8 = int(tid) % VEC8_STRIP;
const int rope_row = int(tid) / VEC8_ROPE;
const int rope_c8 = int(tid) % VEC8_ROPE;

for (int k_base = k_lo; k_base <= k_hi; k_base += BK) {
  if (k_base > k_lo) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  stile_t Stile;
  Stile.clear();

  // Strip 0 (0..255) into XOR-swizzled s_kv, kept in registers for P@V.
  STEEL_PRAGMA_UNROLL
  for (int pass = 0; pass < STRIP_PASSES; ++pass) {
    const int row = strip_row + pass * ROWS_PER_PASS;
    const int kp = k_base + row;
    const int sw_c8 = hy4_kv_swz_c8(strip_c8, row);
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_kv + row * D_CHUNK + sw_c8 * 8);
    if (kp <= k_hi) {
      const int slot = IS_RING ? (kp % ring_cap) : (kp - first_key);
      *dst = *reinterpret_cast<const device float4*>(
          k + size_t(slot) * size_t(D_K) + size_t(strip_c8 * 8));
    } else {
      *dst = float4(0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  hfrag_t k0_s0[SUBS_PER_STRIP];
  hfrag_t k1_s0[SUBS_PER_STRIP];
  if (valid_q > 0) {
    STEEL_PRAGMA_UNROLL
    for (short sub = 0; sub < SUBS_PER_STRIP; ++sub) {
      const int col_off = d_group * (D_CHUNK / WN) + int(sub) * kU;
      hy4_load_kv_frag_tg_swz<half, D_CHUNK>(k0_s0[sub], s_kv, col_off, fm, fn);
      hy4_load_kv_frag_tg_swz<half, D_CHUNK>(
          k1_s0[sub], s_kv + kU * D_CHUNK, col_off, fm, fn);
      BaseNAXFrag::mma(
          Stile.frag_at(0, 0),
          Stile.frag_at(0, 1),
          Qv_hi[sub],
          metal::false_type{},
          k0_s0[sub],
          k1_s0[sub],
          metal::true_type{});
      if constexpr (USE_Q_LO) {
        BaseNAXFrag::mma(
            Stile.frag_at(0, 0),
            Stile.frag_at(0, 1),
            Qv_lo[sub],
            metal::false_type{},
            k0_s0[sub],
            k1_s0[sub],
            metal::true_type{});
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Strip 1 (256..511) into s_kv (stays alive for P@V) + RoPE (512..575) into s_rope.
  STEEL_PRAGMA_UNROLL
  for (int pass = 0; pass < STRIP_PASSES; ++pass) {
    const int row = strip_row + pass * ROWS_PER_PASS;
    const int kp = k_base + row;
    const int sw_c8 = hy4_kv_swz_c8(strip_c8, row);
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_kv + row * D_CHUNK + sw_c8 * 8);
    if (kp <= k_hi) {
      const int slot = IS_RING ? (kp % ring_cap) : (kp - first_key);
      *dst = *reinterpret_cast<const device float4*>(
          k + size_t(slot) * size_t(D_K) + size_t(D_CHUNK + strip_c8 * 8));
    } else {
      *dst = float4(0.0f);
    }
  }
  if (rope_row < BK) {
    const int kp = k_base + rope_row;
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_rope + rope_row * ROPE_STRIDE + rope_c8 * 8);
    if (kp <= k_hi) {
      const int slot = IS_RING ? (kp % ring_cap) : (kp - first_key);
      *dst = *reinterpret_cast<const device float4*>(
          k + size_t(slot) * size_t(D_K) + size_t(D_V + rope_c8 * 8));
    } else {
      *dst = float4(0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (valid_q > 0) {
    STEEL_PRAGMA_UNROLL
    for (short sub = 0; sub < SUBS_PER_STRIP; ++sub) {
      const int col_off = d_group * (D_CHUNK / WN) + int(sub) * kU;
      hfrag_t kf0, kf1;
      hy4_load_kv_frag_tg_swz<half, D_CHUNK>(kf0, s_kv, col_off, fm, fn);
      hy4_load_kv_frag_tg_swz<half, D_CHUNK>(
          kf1, s_kv + kU * D_CHUNK, col_off, fm, fn);
      const short id = SUBS_PER_STRIP + sub;
      BaseNAXFrag::mma(
          Stile.frag_at(0, 0),
          Stile.frag_at(0, 1),
          Qv_hi[id],
          metal::false_type{},
          kf0,
          kf1,
          metal::true_type{});
      if constexpr (USE_Q_LO) {
        BaseNAXFrag::mma(
            Stile.frag_at(0, 0),
            Stile.frag_at(0, 1),
            Qv_lo[id],
            metal::false_type{},
            kf0,
            kf1,
            metal::true_type{});
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short ir = 0; ir < TRh; ++ir) {
      const int col_off = d_group * (D_ROPE / WN) + int(ir) * kU;
      hfrag_t kf0, kf1;
      hy4_load_kv_frag_tg<half, ROPE_STRIDE>(kf0, s_rope, col_off, fm, fn);
      hy4_load_kv_frag_tg<half, ROPE_STRIDE>(
          kf1, s_rope + kU * ROPE_STRIDE, col_off, fm, fn);
      BaseNAXFrag::mma(
          Stile.frag_at(0, 0),
          Stile.frag_at(0, 1),
          Qr_hi[ir],
          metal::false_type{},
          kf0,
          kf1,
          metal::true_type{});
      if constexpr (USE_Q_LO) {
        BaseNAXFrag::mma(
            Stile.frag_at(0, 0),
            Stile.frag_at(0, 1),
            Qr_lo[ir],
            metal::false_type{},
            kf0,
            kf1,
            metal::true_type{});
      }
    }
  }

  // Deterministic 2-slot tree reduction across WN=4 in s_xchg: (dg0+dg2)+(dg1+dg3).
  const int xbase0 = (row_group * 2) * XCHG_STRIDE + int(simd_lid) * (TK * kEPF);
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

  if (valid_q > 0) {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; ++i) {
        sf[i] = s_xchg[xbase0 + int(ik) * kEPF + i] +
            s_xchg[xbase1 + int(ik) * kEPF + i];
      }
    }

    const bool tile_all_valid =
        (valid_q == kU) &&
        ((k_base + BK - 1) <= q_tg_lo) &&
        ((q_tg_hi - k_base) < window_size);

    metal::vec<float, 2> new_max = max_score;
    if (tile_all_valid) {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
        Stile.elems()[ii] *= scale2;
      }
      Stile.template row_reduce<Hy4MaxOp>(new_max);
      Stile.template row_bin_op<Hy4Exp2SubOp>(new_max);
    } else {
      bool elem_ok[TK][2][4];
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ++ik) {
        thread auto& sf = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < 2; ++ii) {
          const int qp = (ii == 0) ? q_pos0 : q_pos1;
          const bool qok = (ii == 0) ? q_ok0 : q_ok1;
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < 4; ++jj) {
            const int kp = k_base + int(ik) * kU + int(fn) + int(jj);
            const bool ok = qok && (kp <= qp) && ((qp - kp) < window_size);
            elem_ok[ik][ii][jj] = ok;
            const short loc = ii * 4 + jj;
            sf[loc] = ok ? (sf[loc] * scale2) : MAX_FLOOR;
          }
        }
      }
      Stile.template row_reduce<Hy4MaxOp>(new_max);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ++ik) {
        thread auto& sf = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < 2; ++ii) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < 4; ++jj) {
            const short loc = ii * 4 + jj;
            sf[loc] = elem_ok[ik][ii][jj] ? metal::exp2(sf[loc] - new_max[ii]) : 0.0f;
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

    hfrag_t p_hi[TK];
    hfrag_t p_lo[FAST_1MMA ? 1 : TK];
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      const thread auto& pf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kEPF; ++e) {
        const float v = pf[e];
        const half hi = half(v);
        p_hi[ik][e] = hi;
        if constexpr (!FAST_1MMA) {
          p_lo[ik][e] = half(v - float(hi));
        }
      }
    }

    // P @ V Strip 0: reuse k0_s0/k1_s0 already in registers.
    STEEL_PRAGMA_UNROLL
    for (short pair = 0; pair < SUBS_PER_STRIP / 2; ++pair) {
      const short id0 = pair * 2;
      const short id1 = id0 + 1;
      BaseNAXFrag::mma(
          Otile.frag_at(0, id0),
          Otile.frag_at(0, id1),
          p_hi[0],
          metal::false_type{},
          k0_s0[id0],
          k0_s0[id1],
          metal::false_type{});
      if constexpr (!FAST_1MMA) {
        BaseNAXFrag::mma(
            Otile.frag_at(0, id0),
            Otile.frag_at(0, id1),
            p_lo[0],
            metal::false_type{},
            k0_s0[id0],
            k0_s0[id1],
            metal::false_type{});
      }
      BaseNAXFrag::mma(
          Otile.frag_at(0, id0),
          Otile.frag_at(0, id1),
          p_hi[1],
          metal::false_type{},
          k1_s0[id0],
          k1_s0[id1],
          metal::false_type{});
      if constexpr (!FAST_1MMA) {
        BaseNAXFrag::mma(
            Otile.frag_at(0, id0),
            Otile.frag_at(0, id1),
            p_lo[1],
            metal::false_type{},
            k1_s0[id0],
            k1_s0[id1],
            metal::false_type{});
      }
    }

    // P @ V Strip 1: read directly from TG-staged s_kv.
    STEEL_PRAGMA_UNROLL
    for (short pair = 0; pair < SUBS_PER_STRIP / 2; ++pair) {
      const int col0 = d_group * (D_CHUNK / WN) + int(pair * 2) * kU;
      const int col1 = col0 + kU;
      const short id0 = SUBS_PER_STRIP + pair * 2;
      const short id1 = id0 + 1;
      hfrag_t vf0, vf1;
      hy4_load_kv_frag_tg_swz<half, D_CHUNK>(vf0, s_kv, col0, fm, fn);
      hy4_load_kv_frag_tg_swz<half, D_CHUNK>(vf1, s_kv, col1, fm, fn);
      BaseNAXFrag::mma(
          Otile.frag_at(0, id0),
          Otile.frag_at(0, id1),
          p_hi[0],
          metal::false_type{},
          vf0,
          vf1,
          metal::false_type{});
      if constexpr (!FAST_1MMA) {
        BaseNAXFrag::mma(
            Otile.frag_at(0, id0),
            Otile.frag_at(0, id1),
            p_lo[0],
            metal::false_type{},
            vf0,
            vf1,
            metal::false_type{});
      }

      hy4_load_kv_frag_tg_swz<half, D_CHUNK>(
          vf0, s_kv + kU * D_CHUNK, col0, fm, fn);
      hy4_load_kv_frag_tg_swz<half, D_CHUNK>(
          vf1, s_kv + kU * D_CHUNK, col1, fm, fn);
      BaseNAXFrag::mma(
          Otile.frag_at(0, id0),
          Otile.frag_at(0, id1),
          p_hi[1],
          metal::false_type{},
          vf0,
          vf1,
          metal::false_type{});
      if constexpr (!FAST_1MMA) {
        BaseNAXFrag::mma(
            Otile.frag_at(0, id0),
            Otile.frag_at(0, id1),
            p_lo[1],
            metal::false_type{},
            vf0,
            vf1,
            metal::false_type{});
      }
    }
  }
}

if (valid_q > 0) {
  const float sink_log2 = sinks[head] * M_LOG2E_F;
  metal::vec<float, 2> out_scale;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 2; ++i) {
    const float m_h = max_score[i];
    const float l_h = sum_score[i];
    const float m_all = metal::max(m_h, sink_log2);
    const float k_scale = metal::exp2(m_h - m_all);
    const float s_term = metal::exp2(sink_log2 - m_all);
    const float den = metal::fma(l_h, k_scale, s_term);
    out_scale[i] = (l_h > 0.0f && den > 0.0f) ? (k_scale / den) : 0.0f;
  }
  Otile.template row_bin_op<Hy4MulOp>(out_scale);

  device OUT_T* out_base =
      out + (size_t(q_tile_base) * size_t(H) + size_t(head)) * size_t(D_V);
  const int out_stride = H * D_V;
  STEEL_PRAGMA_UNROLL
  for (short strip = 0; strip < N_STRIPS; ++strip) {
    STEEL_PRAGMA_UNROLL
    for (short sub = 0; sub < SUBS_PER_STRIP; ++sub) {
      const int col0 =
          int(strip) * D_CHUNK + d_group * (D_CHUNK / WN) + int(sub) * kU;
      const short id = strip * SUBS_PER_STRIP + sub;
      if (valid_q == kU) {
        BaseNAXFrag::store(
            Otile.frag_at(0, id), out_base + col0, out_stride, Int<1>{});
      } else {
        BaseNAXFrag::store_rows(
            Otile.frag_at(0, id), out_base + col0, out_stride, Int<1>{}, valid_q);
      }
    }
  }
}
)METAL"