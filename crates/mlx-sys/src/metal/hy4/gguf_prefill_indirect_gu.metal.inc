R"METAL(
constexpr int BM = 32;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int PAD = 36;

const int R = token_rows_shape[0];
const int N_TOK = x_shape[0];
const int K = x_shape[1];
const int S = dims[0];
const int M = dims[1];

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
const short valid_m = short(max(0, min(16, int(end - begin) - int(tm))));
const short valid_n = short(max(0, min(32, int(M) - int(col + uint(tn)))));

// BM=32, BN=64, BK=32, PAD=36 keeps threadgroup memory at 23,040 B (<= 32 KB).
alignas(16) threadgroup float As[BM * PAD];
alignas(16) threadgroup float Gs[BN * PAD];
alignas(16) threadgroup float Us[BN * PAD];

constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto ca = op.template get_left_input_cooperative_tensor<float, float, float>();
auto cb = op.template get_right_input_cooperative_tensor<float, float, float>();
using AT = metal::remove_addrspace_t<decltype(ca)>;
using BT = metal::remove_addrspace_t<decltype(cb)>;
auto cg = op.template get_destination_cooperative_tensor<AT, BT, float>();
auto cu = op.template get_destination_cooperative_tensor<AT, BT, float>();
constexpr int FE = NAXTile<float, 1, 2>::kElemsPerFrag;
STEEL_PRAGMA_UNROLL
for (int e = 0; e < 2 * FE; ++e) {
  cg[e] = 0.0f;
  cu[e] = 0.0f;
}

if (slot < uint(S)) {
  const uint ar = tid / 4u;
  const uint ac = (tid % 4u) * 8u;
  bool live = (begin + ar) < end;
  const uint token = live ? token_rows[begin + ar] : 0u;
  live = live && (token < uint(N_TOK));
  float4 av[2];
  if (live) {
    const device float4* src = reinterpret_cast<const device float4*>(
        x + size_t(token) * size_t(K) + size_t(ac));
    av[0] = src[0];
    av[1] = src[1];
  } else {
    threadgroup float4* dst =
        reinterpret_cast<threadgroup float4*>(As + ar * PAD + ac);
    dst[0] = float4(0.0f);
    dst[1] = float4(0.0f);
  }

  const bool is_up = (sg_idx >= 2u);
  const uint wr_local = tid & 63u;
  const bool w_live = (col + wr_local) < uint(M);
  const size_t row_bytes = size_t(K / 256) * 144u;
  const device uint8_t* w_row =
      (is_up ? (up_buf + size_t(offsets[1])) : (gate_buf + size_t(offsets[0]))) +
      (size_t(slot) * size_t(M) + size_t(col + wr_local)) * row_bytes;
  threadgroup float4* w_dst4 =
      reinterpret_cast<threadgroup float4*>((is_up ? Us : Gs) + wr_local * PAD);

  float blk_d = 0.0f;
  float blk_dmin = 0.0f;
  uint32_t sc_w[3] = {0u, 0u, 0u};
  uint32_t qs_w[8];

  if (w_live) {
    const half2 dm = *reinterpret_cast<const device half2*>(w_row);
    blk_d = float(dm.x);
    blk_dmin = float(dm.y);
    const device uint32_t* scp = reinterpret_cast<const device uint32_t*>(w_row + 4);
    sc_w[0] = scp[0];
    sc_w[1] = scp[1];
    sc_w[2] = scp[2];
    const device uint32_t* qsp = reinterpret_cast<const device uint32_t*>(w_row + 16);
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      qs_w[i] = qsp[i];
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      w_dst4[i] = float4(0.0f);
    }
  }

  for (int base = 0; base < K; base += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live) {
      threadgroup float4* dst =
          reinterpret_cast<threadgroup float4*>(As + ar * PAD + ac);
      dst[0] = av[0];
      dst[1] = av[1];
    }
    if (w_live) {
      const int g = (base >> 5) & 7;
      float d_g, m_g;
      hy4_gguf_q4k_group_scale(sc_w, blk_d, blk_dmin, g, d_g, m_g);
      const int shift = (g & 1) * 4;
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < 8; ++i) {
        const uint32_t w = qs_w[i] >> shift;
        w_dst4[i] = float4(
            fma(float(w & 0x0Fu), d_g, m_g),
            fma(float((w >> 8) & 0x0Fu), d_g, m_g),
            fma(float((w >> 16) & 0x0Fu), d_g, m_g),
            fma(float((w >> 24) & 0x0Fu), d_g, m_g));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int next = base + BK;
    if (next < K) {
      if (live) {
        const device float4* src = reinterpret_cast<const device float4*>(
            x + size_t(token) * size_t(K) + size_t(next) + size_t(ac));
        av[0] = src[0];
        av[1] = src[1];
      }
      if (w_live && ((next & 32) == 0)) {
        const device uint8_t* nblk = w_row + size_t(next >> 8) * 144u;
        if ((next & 255) == 0) {
          const half2 dm = *reinterpret_cast<const device half2*>(nblk);
          blk_d = float(dm.x);
          blk_dmin = float(dm.y);
          const device uint32_t* scp =
              reinterpret_cast<const device uint32_t*>(nblk + 4);
          sc_w[0] = scp[0];
          sc_w[1] = scp[1];
          sc_w[2] = scp[2];
        }
        const int next_c = (next >> 6) & 3;
        const device uint32_t* qsp = reinterpret_cast<const device uint32_t*>(
            nblk + 16 + size_t(next_c) * 32u);
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
          qs_w[i] = qsp[i];
        }
      }
    }

    if (valid_m > 0 && valid_n > 0) {
      NAXTile<float, 1, 2> A;
      NAXTile<float, 2, 2> B;
      A.template load<float, PAD, 1>(As + tm * PAD);
      B.template load<float, PAD, 1>(Gs + tn * PAD);
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < 2; ++k) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < FE; ++e) {
          ca[e] = A.frag_at(0, k)[e];
          cb[e] = B.frag_at(0, k)[e];
          cb[FE + e] = B.frag_at(1, k)[e];
        }
        op.run(ca, cb, cg);
      }
      B.template load<float, PAD, 1>(Us + tn * PAD);
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < 2; ++k) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < FE; ++e) {
          ca[e] = A.frag_at(0, k)[e];
          cb[e] = B.frag_at(0, k)[e];
          cb[FE + e] = B.frag_at(1, k)[e];
        }
        op.run(ca, cb, cu);
      }
    }
  }
}

if (valid_m > 0 && valid_n > 0) {
  const float lim = limit[0];
  NAXTile<float, 1, 2> D;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < 2; ++f) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < FE; ++e) {
      float gv = cg[f * FE + e];
      float uv = cu[f * FE + e];
      // Asymmetric clamp: never lower-clamp gate.
      if (lim > 0.0f) {
        gv = metal::min(gv, lim);
        uv = metal::clamp(uv, -lim, lim);
      }
      D.val_frags[f][e] = (gv / (1.0f + metal::exp(-gv))) * uv;
    }
  }
  device float* out_ptr =
      out + size_t(begin + uint(tm)) * size_t(M) + size_t(col + uint(tn));
  if (valid_m == 16 && valid_n == 32) {
    D.store(out_ptr, M);
  } else {
    D.store_slice(out_ptr, M, short2(0, 0), short2(valid_n, valid_m));
  }
}
)METAL"
