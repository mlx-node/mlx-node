R"METAL(
constexpr int D_K = D_V + D_ROPE;
constexpr int HALF = D_K / 2;
constexpr short kU = 16;
constexpr int BH = WM * kU;
constexpr int TK = BK / kU;
constexpr int D_CHUNK = 256;
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
constexpr bool USE_Q_LO = !FAST_1MMA && !Q_F16;

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

// XOR-swizzled s_kv (16,384 B) + s_rope (4,608 B) + s_xchg (8,192 B) + s_slots (128 B) = 29,312 B <= 32 KiB.
alignas(16) threadgroup half s_kv[BK * D_CHUNK];
alignas(16) threadgroup half s_rope[BK * ROPE_STRIDE];
alignas(16) threadgroup float s_xchg[WM * 2 * XCHG_STRIDE];
alignas(16) threadgroup int s_slots[BK];

const short2 coord = BaseNAXFrag::get_coord();
const short fm = coord.y;
const short fn = coord.x;

const int q_v_dim = SPLIT_Q ? D_V : D_K;
const device Q_T* qv_base =
    q + (size_t(s) * size_t(H) + size_t(head_base)) * size_t(q_v_dim);
const int q_r_dim = SPLIT_Q ? D_ROPE : D_K;
const device Q_T* qr_base = SPLIT_Q
    ? (q_pe + (size_t(s) * size_t(H) + size_t(head_base)) * size_t(D_ROPE))
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
      qt.load(qv_base + col, q_v_dim);
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
      qt.load(qv_base + col, q_v_dim);
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
    qt.load(qr_base + col, q_r_dim);
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
    qt.load(qr_base + col, q_r_dim);
    Qr_hi[ir] = qt.frag_at(0, 0);
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

bool prev_skipped = false;

for (int k_base = 0; k_base < KMAX; k_base += BK) {
  // Skipped iterations return without a subsequent barrier; guard s_slots WAR before reload.
  // Non-skipped iterations already executed barriers after their last s_slots read.
  if (prev_skipped) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid < uint(BK)) {
    const int kp = k_base + int(tid);
    const int raw = (kp < KMAX) ? idx[idx_base + size_t(kp)] : -1;
    s_slots[tid] = (raw >= 0 && raw < N_KEYS) ? raw : -1;
  }
  // Dual-purpose barrier: RAW for s_slots and WAR for s_kv from prior iteration's P@V Strip 1.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint8_t col_mask = 0u;
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ++ik) {
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < 4; ++jj) {
      if (s_slots[int(ik) * kU + int(fn) + int(jj)] >= 0) {
        col_mask |= uint8_t(1u << (int(ik) * 4 + int(jj)));
      }
    }
  }
  // Every simdgroup checks the same BK=32 s_slots entries, so simd_any is TG-uniform.
  if (!simd_any(col_mask != 0u)) {
    prev_skipped = true;
    continue;
  }
  prev_skipped = false;
  const bool tile_all_valid = simd_all(col_mask == 0xffu);

  stile_t Stile;
  Stile.clear();

  // Strip 0 (0..255) into XOR-swizzled s_kv, kept in registers for P@V.
  STEEL_PRAGMA_UNROLL
  for (int pass = 0; pass < STRIP_PASSES; ++pass) {
    const int row = strip_row + pass * ROWS_PER_PASS;
    const int slot = s_slots[row];
    const int sw_c8 = hy4_kv_swz_c8(strip_c8, row);
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_kv + row * D_CHUNK + sw_c8 * 8);
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
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Strip 1 (256..511) into s_kv (kept alive for P@V) + RoPE (512..575) into s_rope.
  STEEL_PRAGMA_UNROLL
  for (int pass = 0; pass < STRIP_PASSES; ++pass) {
    const int row = strip_row + pass * ROWS_PER_PASS;
    const int slot = s_slots[row];
    const int sw_c8 = hy4_kv_swz_c8(strip_c8, row);
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_kv + row * D_CHUNK + sw_c8 * 8);
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
  if (rope_row < BK) {
    const int slot = s_slots[rope_row];
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

  metal::vec<float, 2> new_max = max_score;
  if (tile_all_valid) {
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
      Stile.elems()[ii] *= scale2;
    }
    Stile.template row_reduce<Hy4MaxOp>(new_max);
    Stile.template row_bin_op<Hy4Exp2SubOp>(new_max);
  } else {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ++ik) {
      thread auto& sf = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < 4; ++jj) {
        const bool ok = ((col_mask >> (int(ik) * 4 + int(jj))) & 1u) != 0u;
        sf[jj] = ok ? (sf[jj] * scale2) : MAX_FLOOR;
        sf[4 + jj] = ok ? (sf[4 + jj] * scale2) : MAX_FLOOR;
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
          const bool ok = ((col_mask >> (int(ik) * 4 + int(jj))) & 1u) != 0u;
          const short loc = ii * 4 + jj;
          sf[loc] = ok ? metal::exp2(sf[loc] - new_max[ii]) : 0.0f;
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

  // P @ V Strip 1: read directly from XOR-swizzled TG-staged s_kv.
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

device OUT_T* out_base =
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