R"METAL(
constexpr int BM = 32;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int PAD = 36;

const int R = hidden_shape[0];
const int M = hidden_shape[1];
const int S = dims[0];
const int H = dims[1];

const uint tile = threadgroup_position_in_grid.y;
const uint begin = tiles[3u * tile];
const uint end = tiles[3u * tile + 1u];
const uint slot = tiles[3u * tile + 2u];
if (begin >= end || end > uint(R)) return;

const uint col = threadgroup_position_in_grid.x * uint(BN);
if (col >= uint(H)) return;

const uint sg_idx = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg_idx * 32u + lane;
const short tm = short((sg_idx / 2u) * 16u);
const short tn = short((sg_idx % 2u) * 32u);
const short valid_m = short(max(0, min(16, int(end - begin) - int(tm))));
const short valid_n = short(max(0, min(32, int(H) - int(col + uint(tn)))));

// BM=32, BN=64, BK=32, PAD=36 keeps threadgroup memory at 13,824 B (<= 32 KB).
alignas(16) threadgroup float As[BM * PAD];
alignas(16) threadgroup float Bs[BN * PAD];

constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto ca = op.template get_left_input_cooperative_tensor<float, float, float>();
auto cb = op.template get_right_input_cooperative_tensor<float, float, float>();
using AT = metal::remove_addrspace_t<decltype(ca)>;
using BT = metal::remove_addrspace_t<decltype(cb)>;
auto cd = op.template get_destination_cooperative_tensor<AT, BT, float>();
constexpr int FE = NAXTile<float, 1, 2>::kElemsPerFrag;
STEEL_PRAGMA_UNROLL
for (int e = 0; e < 2 * FE; ++e) {
  cd[e] = 0.0f;
}

if (slot < uint(S)) {
  const uint ar = tid / 4u;
  const uint ac = (tid % 4u) * 8u;
  const bool live = (begin + ar) < end;
  float4 av[2];
  if (live) {
    const device float4* src = reinterpret_cast<const device float4*>(
        hidden + size_t(begin + ar) * size_t(M) + size_t(ac));
    av[0] = src[0];
    av[1] = src[1];
  } else {
    threadgroup float4* dst =
        reinterpret_cast<threadgroup float4*>(As + ar * PAD + ac);
    dst[0] = float4(0.0f);
    dst[1] = float4(0.0f);
  }

  // 2 threads per weight row: half_id=0 covers 0..15, half_id=1 covers 16..31.
  const uint wr_local = tid >> 1;
  const uint half_id = tid & 1u;
  const bool w_live = (col + wr_local) < uint(H);
  constexpr size_t BLK_BYTES = (BITS == 6) ? 210u : 144u;
  const size_t row_bytes = size_t(M / 256) * BLK_BYTES;
  const device uint8_t* w_row =
      down_buf + size_t(down_off[0]) +
      (size_t(slot) * size_t(H) + size_t(col + wr_local)) * row_bytes;
  threadgroup float4* w_dst4 = reinterpret_cast<threadgroup float4*>(
      Bs + wr_local * PAD + half_id * 16u);

  if (!w_live) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      w_dst4[i] = float4(0.0f);
    }
  }

  if constexpr (BITS == 4) {
    float blk_d = 0.0f;
    float blk_dmin = 0.0f;
    uint32_t sc_w[3] = {0u, 0u, 0u};
    uint32_t qs_w[4];

    if (w_live) {
      const half2 dm = *reinterpret_cast<const device half2*>(w_row);
      blk_d = float(dm.x);
      blk_dmin = float(dm.y);
      const device uint32_t* scp =
          reinterpret_cast<const device uint32_t*>(w_row + 4);
      sc_w[0] = scp[0];
      sc_w[1] = scp[1];
      sc_w[2] = scp[2];
      const device uint32_t* qsp = reinterpret_cast<const device uint32_t*>(
          w_row + 16 + size_t(half_id) * 16u);
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i) {
        qs_w[i] = qsp[i];
      }
    }

    for (int base = 0; base < M; base += BK) {
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
        for (int i = 0; i < 4; ++i) {
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
      if (next < M) {
        if (live) {
          const device float4* src = reinterpret_cast<const device float4*>(
              hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
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
              nblk + 16 + size_t(next_c) * 32u + size_t(half_id) * 16u);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 4; ++i) {
            qs_w[i] = qsp[i];
          }
        }
      }

      if (valid_m > 0 && valid_n > 0) {
        NAXTile<float, 1, 2> A;
        NAXTile<float, 2, 2> B;
        A.template load<float, PAD, 1>(As + tm * PAD);
        B.template load<float, PAD, 1>(Bs + tn * PAD);
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < 2; ++k) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < FE; ++e) {
            ca[e] = A.frag_at(0, k)[e];
            cb[e] = B.frag_at(0, k)[e];
            cb[FE + e] = B.frag_at(1, k)[e];
          }
          op.run(ca, cb, cd);
        }
      }
    }
  } else {
    // block_q6_K is 210 bytes (2-byte aligned only); use uint16_t/int8_t/half only.
    uint16_t ql0[8];
    uint16_t ql1[8];
    uint16_t qh[8];
    float sc_half[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const size_t l_off = size_t(half_id) * 16u;

    if (w_live) {
      const device uint16_t* p_ql0 =
          reinterpret_cast<const device uint16_t*>(w_row + l_off);
      const device uint16_t* p_ql1 = p_ql0 + 16;
      const device uint16_t* p_qh =
          reinterpret_cast<const device uint16_t*>(w_row + 128 + l_off);
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < 8; ++i) {
        ql0[i] = p_ql0[i];
        ql1[i] = p_ql1[i];
        qh[i] = p_qh[i];
      }
      const float d = float(*reinterpret_cast<const device half*>(w_row + 208));
      const device int8_t* scp =
          reinterpret_cast<const device int8_t*>(w_row + 192) + half_id;
      sc_half[0] = d * float(scp[0]);
      sc_half[1] = d * float(scp[2]);
      sc_half[2] = d * float(scp[4]);
      sc_half[3] = d * float(scp[6]);
    }

    for (int base = 0; base < M; base += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (live) {
        threadgroup float4* dst =
            reinterpret_cast<threadgroup float4*>(As + ar * PAD + ac);
        dst[0] = av[0];
        dst[1] = av[1];
      }
      if (w_live) {
        const int q_quad = (base >> 5) & 3;
        const int ql_shift = (q_quad >> 1) * 4;
        const int qh_shift = q_quad * 2;
        const float sc_eff = sc_half[q_quad];
        const bool use_ql1 = (q_quad & 1) != 0;
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          const uint16_t qa = use_ql1 ? ql1[2 * i] : ql0[2 * i];
          const uint16_t qb = use_ql1 ? ql1[2 * i + 1] : ql0[2 * i + 1];
          const uint16_t ha = qh[2 * i];
          const uint16_t hb = qh[2 * i + 1];

          const uint8_t ql_b0 = uint8_t(qa & 0xFFu);
          const uint8_t ql_b1 = uint8_t(qa >> 8);
          const uint8_t ql_b2 = uint8_t(qb & 0xFFu);
          const uint8_t ql_b3 = uint8_t(qb >> 8);
          const uint8_t qh_b0 = uint8_t(ha & 0xFFu);
          const uint8_t qh_b1 = uint8_t(ha >> 8);
          const uint8_t qh_b2 = uint8_t(hb & 0xFFu);
          const uint8_t qh_b3 = uint8_t(hb >> 8);

          const int c0 = int(((ql_b0 >> ql_shift) & 0x0Fu) | (((qh_b0 >> qh_shift) & 0x03u) << 4)) - 32;
          const int c1 = int(((ql_b1 >> ql_shift) & 0x0Fu) | (((qh_b1 >> qh_shift) & 0x03u) << 4)) - 32;
          const int c2 = int(((ql_b2 >> ql_shift) & 0x0Fu) | (((qh_b2 >> qh_shift) & 0x03u) << 4)) - 32;
          const int c3 = int(((ql_b3 >> ql_shift) & 0x0Fu) | (((qh_b3 >> qh_shift) & 0x03u) << 4)) - 32;
          w_dst4[i] = float4(float(c0) * sc_eff, float(c1) * sc_eff, float(c2) * sc_eff, float(c3) * sc_eff);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const int next = base + BK;
      if (next < M) {
        if (live) {
          const device float4* src = reinterpret_cast<const device float4*>(
              hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
          av[0] = src[0];
          av[1] = src[1];
        }
        if (w_live && ((next & 127) == 0)) {
          const device uint8_t* nblk = w_row + size_t(next >> 8) * 210u;
          const size_t nip = size_t((next >> 7) & 1);
          const device uint16_t* p_ql0 = reinterpret_cast<const device uint16_t*>(
              nblk + nip * 64u + l_off);
          const device uint16_t* p_ql1 = p_ql0 + 16;
          const device uint16_t* p_qh = reinterpret_cast<const device uint16_t*>(
              nblk + 128 + nip * 32u + l_off);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 8; ++i) {
            ql0[i] = p_ql0[i];
            ql1[i] = p_ql1[i];
            qh[i] = p_qh[i];
          }
          const float d = float(*reinterpret_cast<const device half*>(nblk + 208));
          const device int8_t* scp =
              reinterpret_cast<const device int8_t*>(nblk + 192) + nip * 8u + half_id;
          sc_half[0] = d * float(scp[0]);
          sc_half[1] = d * float(scp[2]);
          sc_half[2] = d * float(scp[4]);
          sc_half[3] = d * float(scp[6]);
        }
      }

      if (valid_m > 0 && valid_n > 0) {
        NAXTile<float, 1, 2> A;
        NAXTile<float, 2, 2> B;
        A.template load<float, PAD, 1>(As + tm * PAD);
        B.template load<float, PAD, 1>(Bs + tn * PAD);
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < 2; ++k) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < FE; ++e) {
            ca[e] = A.frag_at(0, k)[e];
            cb[e] = B.frag_at(0, k)[e];
            cb[FE + e] = B.frag_at(1, k)[e];
          }
          op.run(ca, cb, cd);
        }
      }
    }
  }
}

if (valid_m > 0 && valid_n > 0) {
  NAXTile<float, 1, 2> D;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < 2; ++f) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < FE; ++e) {
      D.val_frags[f][e] = cd[f * FE + e];
    }
  }
  device float* out_ptr =
      out + size_t(begin + uint(tm)) * size_t(H) + size_t(col + uint(tn));
  if (valid_m == 16 && valid_n == 32) {
    D.store(out_ptr, H);
  } else {
    D.store_slice(out_ptr, H, short2(0, 0), short2(valid_n, valid_m));
  }
}
)METAL"
