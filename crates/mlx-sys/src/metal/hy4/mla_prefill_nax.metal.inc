R"METAL(
constexpr int D_K = D_V + D_ROPE;
constexpr int HALF = D_K / 2;
constexpr short kU = 16;
constexpr int BH = WM * kU;
constexpr int TK = BK / kU;
constexpr int D_CHUNK = 256;
constexpr int KV_STRIDE = D_CHUNK + 8;
constexpr int ROPE_STRIDE = D_ROPE + 8;
constexpr int VEC8_STRIP = D_CHUNK / 8;
constexpr int VEC8_ROPE = D_ROPE / 8;
constexpr int VEC8_HALF = HALF / 8;
constexpr int N_STRIPS = D_V / D_CHUNK;
constexpr int SUBS_PER_STRIP = D_CHUNK / (WN * kU);
constexpr int TDh = D_V / (WN * kU);
constexpr int TRh = D_ROPE / (WN * kU);
constexpr int XCHG_STRIDE = TK * BaseNAXFrag::kElemsPerFrag * 32;
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
using stile_t = NAXTile<float, 1, TK>;
using hfrag_t = typename BaseNAXFrag::template dtype_frag_t<half>;
constexpr short kEPF = BaseNAXFrag::kElemsPerFrag;

// Separate s_kv (16,896 B) + s_rope (4,608 B) + s_xchg (8,192 B) = 29,696 B <= 32 KiB.
// Eliminates WAR barrier before s_xchg and preserves Strip 1 in s_kv for P@V.
alignas(16) threadgroup half s_kv[BK * KV_STRIDE];
alignas(16) threadgroup half s_rope[BK * ROPE_STRIDE];
alignas(16) threadgroup float s_xchg[WM * 2 * XCHG_STRIDE];

const short2 coord = BaseNAXFrag::get_coord();
const short fm = coord.y;
const short fn = coord.x;

const int q_v_dim = SPLIT_Q ? D_V : D_K;
const device float* qv_base =
    q + (size_t(s) * size_t(H) + size_t(head_base)) * size_t(q_v_dim);
const int q_r_dim = SPLIT_Q ? D_ROPE : D_K;
const device float* qr_base = SPLIT_Q
    ? (q_pe + (size_t(s) * size_t(H) + size_t(head_base)) * size_t(D_ROPE))
    : (qv_base + D_V);

hfrag_t Qv_hi[TDh];
hfrag_t Qv_lo[TDh];
STEEL_PRAGMA_UNROLL
for (short strip = 0; strip < N_STRIPS; ++strip) {
  STEEL_PRAGMA_UNROLL
  for (short sub = 0; sub < SUBS_PER_STRIP; ++sub) {
    const int col =
        int(strip) * D_CHUNK + d_group * (D_CHUNK / WN) + int(sub) * kU;
    NAXTile<float, 1, 1> qt;
    qt.load(qv_base + col, q_v_dim);
    const thread auto& qf = qt.frag_at(0, 0);
    const short id = strip * SUBS_PER_STRIP + sub;
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kEPF; ++e) {
      const float v = qf[e];
      const half hi = half(v);
      Qv_hi[id][e] = hi;
      Qv_lo[id][e] = half(v - float(hi));
    }
  }
}

hfrag_t Qr_hi[TRh];
hfrag_t Qr_lo[TRh];
STEEL_PRAGMA_UNROLL
for (short ir = 0; ir < TRh; ++ir) {
  const int col = d_group * (D_ROPE / WN) + int(ir) * kU;
  NAXTile<float, 1, 1> qt;
  qt.load(qr_base + col, q_r_dim);
  const thread auto& qf = qt.frag_at(0, 0);
  STEEL_PRAGMA_UNROLL
  for (short e = 0; e < kEPF; ++e) {
    const float v = qf[e];
    const half hi = half(v);
    Qr_hi[ir][e] = hi;
    Qr_lo[ir][e] = half(v - float(hi));
  }
}

otile_t Otile;
Otile.clear();

metal::vec<float, 2> max_score(MAX_FLOOR);
metal::vec<float, 2> sum_score(0.0f);

const float scale2 = scale[0] * M_LOG2E_F;
const size_t idx_base = size_t(s) * size_t(KMAX);
constexpr int NUM_THREADS = WM * WN * 32;
constexpr int ROWS_PER_PASS = NUM_THREADS / VEC8_STRIP;
constexpr int STRIP_PASSES = BK / ROWS_PER_PASS;
const int strip_row = int(tid) / VEC8_STRIP;
const int strip_c8 = int(tid) % VEC8_STRIP;
const int rope_row = int(tid) / VEC8_ROPE;
const int rope_c8 = int(tid) % VEC8_ROPE;

bool need_war = false;

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
  // Every simdgroup checks the same BK=32 idx entries, so simd_any is TG-uniform.
  if (!simd_any(lane_any)) {
    continue;
  }
  const bool tile_all_valid = simd_all(lane_all);

  if (need_war) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  need_war = true;

  int my_slots[STRIP_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int pass = 0; pass < STRIP_PASSES; ++pass) {
    const int kp = k_base + strip_row + pass * ROWS_PER_PASS;
    const int raw = (kp < KMAX) ? idx[idx_base + size_t(kp)] : -1;
    my_slots[pass] = (raw >= 0 && raw < N_KEYS) ? raw : -1;
  }

  stile_t Stile;
  Stile.clear();

  // Strip 0 (0..255) staged into s_kv and kept in registers for P@V.
  STEEL_PRAGMA_UNROLL
  for (int pass = 0; pass < STRIP_PASSES; ++pass) {
    const int row = strip_row + pass * ROWS_PER_PASS;
    const int slot = my_slots[pass];
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_kv + row * KV_STRIDE + strip_c8 * 8);
    if (slot >= 0) {
      if constexpr (SPLIT) {
        *dst = *reinterpret_cast<const device float4*>(
            k0 + size_t(slot) * size_t(HALF) + size_t(strip_c8 * 8));
      } else {
        *dst = *reinterpret_cast<const device float4*>(
            k0 + size_t(slot) * size_t(D_K) + size_t(strip_c8 * 8));
      }
    } else {
      *dst = float4(0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  hfrag_t k0_s0[SUBS_PER_STRIP];
  hfrag_t k1_s0[SUBS_PER_STRIP];
  STEEL_PRAGMA_UNROLL
  for (short sub = 0; sub < SUBS_PER_STRIP; ++sub) {
    const int col_off = d_group * (D_CHUNK / WN) + int(sub) * kU;
    hy4_load_kv_frag_tg<half, KV_STRIDE>(k0_s0[sub], s_kv, col_off, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        k1_s0[sub], s_kv + kU * KV_STRIDE, col_off, fm, fn);
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Qv_hi[sub],
        metal::false_type{},
        k0_s0[sub],
        k1_s0[sub],
        metal::true_type{});
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Qv_lo[sub],
        metal::false_type{},
        k0_s0[sub],
        k1_s0[sub],
        metal::true_type{});
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Strip 1 (256..511) into s_kv (kept alive for P@V) + RoPE (512..575) into s_rope.
  STEEL_PRAGMA_UNROLL
  for (int pass = 0; pass < STRIP_PASSES; ++pass) {
    const int row = strip_row + pass * ROWS_PER_PASS;
    const int slot = my_slots[pass];
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_kv + row * KV_STRIDE + strip_c8 * 8);
    if (slot >= 0) {
      const int g_c8 = VEC8_STRIP + strip_c8;
      if constexpr (SPLIT) {
        const device half* plane = (g_c8 >= VEC8_HALF) ? k1 : k0;
        const int p_c8 = (g_c8 >= VEC8_HALF) ? (g_c8 - VEC8_HALF) : g_c8;
        *dst = *reinterpret_cast<const device float4*>(
            plane + size_t(slot) * size_t(HALF) + size_t(p_c8 * 8));
      } else {
        *dst = *reinterpret_cast<const device float4*>(
            k0 + size_t(slot) * size_t(D_K) + size_t(g_c8 * 8));
      }
    } else {
      *dst = float4(0.0f);
    }
  }
  {
    const int kp = k_base + rope_row;
    const int raw = (kp < KMAX) ? idx[idx_base + size_t(kp)] : -1;
    const int slot = (raw >= 0 && raw < N_KEYS) ? raw : -1;
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_rope + rope_row * ROPE_STRIDE + rope_c8 * 8);
    if (slot >= 0) {
      const int g_c8 = (D_V / 8) + rope_c8;
      if constexpr (SPLIT) {
        const int p_c8 = g_c8 - VEC8_HALF;
        *dst = *reinterpret_cast<const device float4*>(
            k1 + size_t(slot) * size_t(HALF) + size_t(p_c8 * 8));
      } else {
        *dst = *reinterpret_cast<const device float4*>(
            k0 + size_t(slot) * size_t(D_K) + size_t(g_c8 * 8));
      }
    } else {
      *dst = float4(0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  STEEL_PRAGMA_UNROLL
  for (short sub = 0; sub < SUBS_PER_STRIP; ++sub) {
    const int col_off = d_group * (D_CHUNK / WN) + int(sub) * kU;
    hfrag_t kf0, kf1;
    hy4_load_kv_frag_tg<half, KV_STRIDE>(kf0, s_kv, col_off, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        kf1, s_kv + kU * KV_STRIDE, col_off, fm, fn);
    const short id = SUBS_PER_STRIP + sub;
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Qv_hi[id],
        metal::false_type{},
        kf0,
        kf1,
        metal::true_type{});
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Qv_lo[id],
        metal::false_type{},
        kf0,
        kf1,
        metal::true_type{});
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
    BaseNAXFrag::mma(
        Stile.frag_at(0, 0),
        Stile.frag_at(0, 1),
        Qr_lo[ir],
        metal::false_type{},
        kf0,
        kf1,
        metal::true_type{});
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

  hfrag_t p_hi[TK];
  hfrag_t p_lo[TK];
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    const thread auto& pf = Stile.frag_at(0, ik);
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kEPF; ++e) {
      const float v = pf[e];
      const half hi = half(v);
      p_hi[ik][e] = hi;
      p_lo[ik][e] = half(v - float(hi));
    }
  }

  // P @ V Strip 0: reuse k0_s0/k1_s0 already in registers (zero device reads).
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
    BaseNAXFrag::mma(
        Otile.frag_at(0, id0),
        Otile.frag_at(0, id1),
        p_lo[0],
        metal::false_type{},
        k0_s0[id0],
        k0_s0[id1],
        metal::false_type{});
    BaseNAXFrag::mma(
        Otile.frag_at(0, id0),
        Otile.frag_at(0, id1),
        p_hi[1],
        metal::false_type{},
        k1_s0[id0],
        k1_s0[id1],
        metal::false_type{});
    BaseNAXFrag::mma(
        Otile.frag_at(0, id0),
        Otile.frag_at(0, id1),
        p_lo[1],
        metal::false_type{},
        k1_s0[id0],
        k1_s0[id1],
        metal::false_type{});
  }

  // P @ V Strip 1: read directly from TG-staged s_kv (zero scattered DRAM reads).
  STEEL_PRAGMA_UNROLL
  for (short pair = 0; pair < SUBS_PER_STRIP / 2; ++pair) {
    const int col0 = d_group * (D_CHUNK / WN) + int(pair * 2) * kU;
    const int col1 = col0 + kU;
    const short id0 = SUBS_PER_STRIP + pair * 2;
    const short id1 = id0 + 1;
    hfrag_t vf0, vf1;
    hy4_load_kv_frag_tg<half, KV_STRIDE>(vf0, s_kv, col0, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(vf1, s_kv, col1, fm, fn);
    BaseNAXFrag::mma(
        Otile.frag_at(0, id0),
        Otile.frag_at(0, id1),
        p_hi[0],
        metal::false_type{},
        vf0,
        vf1,
        metal::false_type{});
    BaseNAXFrag::mma(
        Otile.frag_at(0, id0),
        Otile.frag_at(0, id1),
        p_lo[0],
        metal::false_type{},
        vf0,
        vf1,
        metal::false_type{});

    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        vf0, s_kv + kU * KV_STRIDE, col0, fm, fn);
    hy4_load_kv_frag_tg<half, KV_STRIDE>(
        vf1, s_kv + kU * KV_STRIDE, col1, fm, fn);
    BaseNAXFrag::mma(
        Otile.frag_at(0, id0),
        Otile.frag_at(0, id1),
        p_hi[1],
        metal::false_type{},
        vf0,
        vf1,
        metal::false_type{});
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
    out + (size_t(s) * size_t(H) + size_t(head_base)) * size_t(D_V);
STEEL_PRAGMA_UNROLL
for (short strip = 0; strip < N_STRIPS; ++strip) {
  STEEL_PRAGMA_UNROLL
  for (short sub = 0; sub < SUBS_PER_STRIP; ++sub) {
    const int col0 =
        int(strip) * D_CHUNK + d_group * (D_CHUNK / WN) + int(sub) * kU;
    const short id = strip * SUBS_PER_STRIP + sub;
    BaseNAXFrag::store(Otile.frag_at(0, id), out_base + col0, D_V, Int<1>{});
  }
}
)METAL"