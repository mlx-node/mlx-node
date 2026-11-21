R"METAL(
constexpr int D_K = D_V + D_ROPE;
constexpr short kU = 16;
constexpr int BQ = kU;
constexpr int BH = WM;
constexpr int BK = 32;
constexpr int TK = BK / kU;
constexpr int D_CHUNK = 128;
constexpr int KV_STRIDE = D_CHUNK + 8;
constexpr int VEC8_STRIP = D_CHUNK / 8;
constexpr int VEC8_ROPE = D_ROPE / 8;
constexpr int N_STRIPS = D_V / D_CHUNK;
constexpr int TDh = D_V / (WN * kU);
constexpr int TRh = D_ROPE / (WN * kU);
constexpr int XCHG_STRIDE = TK * BaseNAXFrag::kElemsPerFrag * 32;
constexpr float MAX_FLOOR = -1e30f;

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

// Single 8,704 B buffer serves as s_kv (32 x 136 half) and s_xchg (8,192 B
// 2-slot tree reduction across WN=4), keeping TG <= 8.5 KiB (2+ TGs/core).
alignas(16) threadgroup uint s_buf[(BK * KV_STRIDE) / 2];
threadgroup half* s_kv = reinterpret_cast<threadgroup half*>(s_buf);
threadgroup float* s_xchg = reinterpret_cast<threadgroup float*>(s_buf);

const short2 coord = BaseNAXFrag::get_coord();
const short fm = coord.y;
const short fn = coord.x;

const short valid_q = (head < H)
    ? short(metal::max(0, metal::min(int(kU), S - q_tile_base)))
    : short(0);

const int q_stride = H * D_K;
const device float* q_base =
    q + (size_t(q_tile_base) * size_t(H) + size_t(metal::min(head, H - 1))) * size_t(D_K);

hfrag_t Qv_hi[TDh];
hfrag_t Qv_lo[TDh];
STEEL_PRAGMA_UNROLL
for (short strip = 0; strip < N_STRIPS; ++strip) {
  STEEL_PRAGMA_UNROLL
  for (short sub = 0; sub < 2; ++sub) {
    const int col = int(strip) * D_CHUNK + d_group * 32 + int(sub) * kU;
    NAXTile<float, 1, 1> qt;
    if (valid_q == kU) {
      qt.load(q_base + col, q_stride);
    } else if (valid_q > 0) {
      qt.load_rows(q_base + col, q_stride, valid_q);
    } else {
      qt.clear();
    }
    const thread auto& qf = qt.frag_at(0, 0);
    const short id = strip * 2 + sub;
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
  const int col = D_V + d_group * (D_ROPE / WN) + int(ir) * kU;
  NAXTile<float, 1, 1> qt;
  if (valid_q == kU) {
    qt.load(q_base + col, q_stride);
  } else if (valid_q > 0) {
    qt.load_rows(q_base + col, q_stride, valid_q);
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

const int strip_row = int(tid) / VEC8_STRIP;
const int strip_c8 = int(tid) % VEC8_STRIP;
const int rope_row = int(tid) / VEC8_ROPE;
const int rope_c8 = int(tid) % VEC8_ROPE;

for (int k_base = k_lo; k_base <= k_hi; k_base += BK) {
  stile_t Stile;
  Stile.clear();

  // Stage K in D_CHUNK=128 strips into s_kv (32 x 136 half) shared across BH heads.
  STEEL_PRAGMA_UNROLL
  for (short strip = 0; strip < N_STRIPS; ++strip) {
    const int base_c8 = int(strip) * VEC8_STRIP;
    STEEL_PRAGMA_UNROLL
    for (int pass = 0; pass < 2; ++pass) {
      const int row = strip_row + pass * 16;
      const int kp = k_base + row;
      threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
          s_kv + row * KV_STRIDE + strip_c8 * 8);
      if (kp <= k_hi) {
        const int slot = IS_RING ? (kp % ring_cap) : (kp - first_key);
        *dst = *reinterpret_cast<const device float4*>(
            k + size_t(slot) * size_t(D_K) + size_t((base_c8 + strip_c8) * 8));
      } else {
        *dst = float4(0.0f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_q > 0) {
      STEEL_PRAGMA_UNROLL
      for (short sub = 0; sub < 2; ++sub) {
        const int col_off = d_group * 32 + int(sub) * kU;
        hfrag_t kf0, kf1;
        hy4_load_kv_frag_tg<half, KV_STRIDE>(kf0, s_kv, col_off, fm, fn);
        hy4_load_kv_frag_tg<half, KV_STRIDE>(
            kf1, s_kv + kU * KV_STRIDE, col_off, fm, fn);
        const short id = strip * 2 + sub;
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
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // RoPE strip (64 cols).
  {
    const int kp = k_base + rope_row;
    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
        s_kv + rope_row * KV_STRIDE + rope_c8 * 8);
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
    for (short ir = 0; ir < TRh; ++ir) {
      const int col_off = d_group * (D_ROPE / WN) + int(ir) * kU;
      hfrag_t kf0, kf1;
      hy4_load_kv_frag_tg<half, KV_STRIDE>(kf0, s_kv, col_off, fm, fn);
      hy4_load_kv_frag_tg<half, KV_STRIDE>(
          kf1, s_kv + kU * KV_STRIDE, col_off, fm, fn);
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
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

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

    bool elem_ok[TK][2][4];
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

    const int kp0a = k_base + int(fm);
    const int kp0b = kp0a + 8;
    const int kp1a = k_base + kU + int(fm);
    const int kp1b = kp1a + 8;
    const int s0a = (kp0a <= k_hi) ? (IS_RING ? (kp0a % ring_cap) : (kp0a - first_key)) : -1;
    const int s0b = (kp0b <= k_hi) ? (IS_RING ? (kp0b % ring_cap) : (kp0b - first_key)) : -1;
    const int s1a = (kp1a <= k_hi) ? (IS_RING ? (kp1a % ring_cap) : (kp1a - first_key)) : -1;
    const int s1b = (kp1b <= k_hi) ? (IS_RING ? (kp1b % ring_cap) : (kp1b - first_key)) : -1;

    STEEL_PRAGMA_UNROLL
    for (short strip = 0; strip < N_STRIPS; ++strip) {
      const int col0 = int(strip) * D_CHUNK + d_group * 32;
      const int col1 = col0 + kU;
      const short id0 = strip * 2;
      const short id1 = id0 + 1;
      hfrag_t vf0, vf1;
      hy4_load_kv_frag_dev<false, D_K, D_K>(vf0, k, k, s0a, s0b, col0, fn);
      hy4_load_kv_frag_dev<false, D_K, D_K>(vf1, k, k, s0a, s0b, col1, fn);
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

      hy4_load_kv_frag_dev<false, D_K, D_K>(vf0, k, k, s1a, s1b, col0, fn);
      hy4_load_kv_frag_dev<false, D_K, D_K>(vf1, k, k, s1a, s1b, col1, fn);
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
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

  device float* out_base =
      out + (size_t(q_tile_base) * size_t(H) + size_t(head)) * size_t(D_V);
  const int out_stride = H * D_V;
  STEEL_PRAGMA_UNROLL
  for (short strip = 0; strip < N_STRIPS; ++strip) {
    const int col0 = int(strip) * D_CHUNK + d_group * 32;
    if (valid_q == kU) {
      BaseNAXFrag::store(
          Otile.frag_at(0, strip * 2), out_base + col0, out_stride, Int<1>{});
      BaseNAXFrag::store(
          Otile.frag_at(0, strip * 2 + 1), out_base + col0 + kU, out_stride, Int<1>{});
    } else {
      BaseNAXFrag::store_rows(
          Otile.frag_at(0, strip * 2), out_base + col0, out_stride, Int<1>{}, valid_q);
      BaseNAXFrag::store_rows(
          Otile.frag_at(0, strip * 2 + 1), out_base + col0 + kU, out_stride, Int<1>{}, valid_q);
    }
  }
}
)METAL"