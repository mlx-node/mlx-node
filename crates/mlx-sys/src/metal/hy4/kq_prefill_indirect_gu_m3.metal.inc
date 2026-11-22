R"METAL(
constexpr int BM = 32;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int PAD = 40;

const int R = token_rows_shape[0];
const int N_TOK = x_shape[0];
const int K = x_shape[1];
const int S = wg_shape[0];
const int M = wg_shape[1];

const uint tile = threadgroup_position_in_grid.y;
const uint begin = tiles[3u * tile];
const uint end = tiles[3u * tile + 1u];
const uint slot = tiles[3u * tile + 2u];
if (begin >= end || end > uint(R)) return;

const uint col = threadgroup_position_in_grid.x * uint(BN);
if (col >= uint(M)) return;

const uint sg_idx = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg_idx * 32u + lane;
const short tm = short((sg_idx / 2u) * 16u);
const short tn = short((sg_idx % 2u) * 32u);
const short valid = short(max(0, min(16, int(end - begin) - int(tm))));
const ushort2 coord = hy4_m3_sg_coord(lane);
const ushort fn = coord.x;
const ushort fm = coord.y;

// As (2,560 B) + Gs (5,120 B) + Us (5,120 B) = 12,800 B (12.5 KiB).
alignas(16) threadgroup half As[BM * PAD];
alignas(16) threadgroup half Gs[BN * PAD];
alignas(16) threadgroup half Us[BN * PAD];

simdgroup_float8x8 cg[2][4];
simdgroup_float8x8 cu[2][4];
STEEL_PRAGMA_UNROLL
for (short mi = 0; mi < 2; ++mi) {
  STEEL_PRAGMA_UNROLL
  for (short ni = 0; ni < 4; ++ni) {
    cg[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    cu[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
  }
}

if (slot < uint(S)) {
  const uint ar = tid >> 2;
  const uint ac = (tid & 3u) * 8u;
  bool live = (begin + ar) < end;
  const uint token = live ? token_rows[begin + ar] : 0u;
  live = live && (token < uint(N_TOK));
  half4 av[2];
  if (live) {
    const device float4* src = reinterpret_cast<const device float4*>(
        x + size_t(token) * size_t(K) + size_t(ac));
    av[0] = half4(src[0]);
    av[1] = half4(src[1]);
  } else {
    threadgroup half4* dst =
        reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
    dst[0] = half4(0.0h);
    dst[1] = half4(0.0h);
  }

  // SGs 0..1 own gate weights, SGs 2..3 own up weights (64 threads each -> 1 row/thread).
  const bool is_up = (sg_idx >= 2u);
  const uint wr_local = tid & 63u;
  const size_t wr = size_t(slot) * size_t(M) + size_t(col + wr_local);
  const device uint4* wp =
      reinterpret_cast<const device uint4*>(is_up ? wu : wg) +
      wr * size_t(K >> 5);
  KQScales<float, 4, 8, true, KQ_LINEAR, 0> wscale(
      is_up ? su : sg, is_up ? bu : bg, wr * size_t(K >> 5));
  threadgroup half4* w_dst =
      reinterpret_cast<threadgroup half4*>((is_up ? Us : Gs) + wr_local * PAD);

  uint4 qw = wp[0];
  float sc, bi;
  wscale.at(0, sc, bi);

  for (int base = 0; base < K; base += BK) {
    if (base > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    hy4_kq_unpack_q4_group32(qw, sc, bi, w_dst);
    if (live) {
      threadgroup half4* dst =
          reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
      dst[0] = av[0];
      dst[1] = av[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int next = base + BK;
    if (next < K) {
      if (live) {
        const device float4* src = reinterpret_cast<const device float4*>(
            x + size_t(token) * size_t(K) + size_t(next) + size_t(ac));
        av[0] = half4(src[0]);
        av[1] = half4(src[1]);
      }
      const size_t next_g = size_t(next >> 5);
      qw = wp[next_g];
      wscale.at(next_g, sc, bi);
    }

    if (valid > 0) {
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < BK; kk += 8) {
        simdgroup_half8x8 ma0, ma1;
        simdgroup_load(ma0, As + (tm + 0) * PAD + kk, PAD);
        simdgroup_load(ma1, As + (tm + 8) * PAD + kk, PAD);
        STEEL_PRAGMA_UNROLL
        for (short ni = 0; ni < 4; ++ni) {
          simdgroup_half8x8 mbg;
          simdgroup_load(
              mbg, Gs + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
          simdgroup_multiply_accumulate(cg[0][ni], ma0, mbg, cg[0][ni]);
          simdgroup_multiply_accumulate(cg[1][ni], ma1, mbg, cg[1][ni]);
        }
        STEEL_PRAGMA_UNROLL
        for (short ni = 0; ni < 4; ++ni) {
          simdgroup_half8x8 mbu;
          simdgroup_load(
              mbu, Us + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
          simdgroup_multiply_accumulate(cu[0][ni], ma0, mbu, cu[0][ni]);
          simdgroup_multiply_accumulate(cu[1][ni], ma1, mbu, cu[1][ni]);
        }
      }
    }
  }
}

if (valid > 0) {
  const float lim = limit[0];
  STEEL_PRAGMA_UNROLL
  for (short mi = 0; mi < 2; ++mi) {
    const short r_off = mi * 8 + short(fm);
    if (r_off < valid) {
      device float* row_base =
          out + size_t(begin + uint(tm) + uint(r_off)) * size_t(M) +
          size_t(col + uint(tn));
      STEEL_PRAGMA_UNROLL
      for (short ni = 0; ni < 4; ++ni) {
        const short c_off = ni * 8 + short(fn);
        float2 gv = hy4_m3_te(cg[mi][ni]);
        float2 uv = hy4_m3_te(cu[mi][ni]);
        // Asymmetric clamp: never lower-clamp gate.
        if (lim > 0.0f) {
          gv = metal::min(gv, float2(lim));
          uv = metal::clamp(uv, float2(-lim), float2(lim));
        }
        const float2 res = (gv / (1.0f + metal::exp(-gv))) * uv;
        row_base[c_off] = res.x;
        row_base[c_off + 1] = res.y;
      }
    }
  }
}
)METAL"
