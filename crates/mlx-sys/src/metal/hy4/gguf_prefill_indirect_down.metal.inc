R"METAL(
constexpr int BM = 32;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int PAD = 72;

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

// As (4,608 B) + Bs (9,216 B) = 13,824 B <= 16 KiB (2 TGs/core).
alignas(16) threadgroup half As[BM * PAD];
alignas(16) threadgroup half Bs[BN * PAD];

NAXTile<float, 1, 2> cd;
cd.clear();

if (slot < uint(S)) {
  const uint ar = tid >> 2;
  const uint ac = (tid & 3u) * 16u;
  const bool live = (begin + ar) < end;
  half4 av[4];
  if (live) {
    if constexpr (IN_F16) {
      const device half4* src = reinterpret_cast<const device half4*>(
          hidden + size_t(begin + ar) * size_t(M) + size_t(ac));
      av[0] = src[0];
      av[1] = src[1];
      av[2] = src[2];
      av[3] = src[3];
    } else {
      const device float4* src = reinterpret_cast<const device float4*>(
          hidden + size_t(begin + ar) * size_t(M) + size_t(ac));
      av[0] = half4(src[0]);
      av[1] = half4(src[1]);
      av[2] = half4(src[2]);
      av[3] = half4(src[3]);
    }
  } else {
    threadgroup half4* dst =
        reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
    dst[0] = half4(0.0h);
    dst[1] = half4(0.0h);
    dst[2] = half4(0.0h);
    dst[3] = half4(0.0h);
  }

  // 4 threads per weight row cooperatively load 8 consecutive bytes each = 32 B.
  const uint row0 = tid >> 2;
  const uint lane4 = tid & 3u;
  const ushort quad_base = ushort(lane & ~3u);
  const bool w_live[2] = {
      (col + row0) < uint(H),
      (col + row0 + 32u) < uint(H),
  };
  constexpr size_t BLK_BYTES = (BITS == 6) ? 210u : 144u;
  const size_t row_bytes = size_t(M / 256) * BLK_BYTES;
  const device uint8_t* w_row[2];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < 2; ++p) {
    const uint wr = row0 + uint(p) * 32u;
    w_row[p] = down_buf + size_t(down_off[0]) +
        (size_t(slot) * size_t(H) + size_t(col + wr)) * row_bytes;
    if (!w_live[p]) {
      threadgroup half4* d_lo =
          reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + lane4 * 8u);
      threadgroup half4* d_hi =
          reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + 32u + lane4 * 8u);
      d_lo[0] = half4(0.0h);
      d_lo[1] = half4(0.0h);
      d_hi[0] = half4(0.0h);
      d_hi[1] = half4(0.0h);
    }
  }

  if constexpr (BITS == 4) {
    float blk_d[2], blk_dmin[2];
    uint32_t sc_w[2][3];
    uint32_t qs_w[2][2];

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      const uint32_t hdr = w_live[p]
          ? reinterpret_cast<const device uint32_t*>(w_row[p])[lane4]
          : 0u;
      const half2 dm = as_type<half2>(simd_shuffle(hdr, quad_base));
      blk_d[p] = float(dm.x);
      blk_dmin[p] = float(dm.y);
      sc_w[p][0] = simd_shuffle(hdr, quad_base | 1u);
      sc_w[p][1] = simd_shuffle(hdr, quad_base | 2u);
      sc_w[p][2] = simd_shuffle(hdr, quad_base | 3u);
      if (w_live[p]) {
        const device uint32_t* qsp = reinterpret_cast<const device uint32_t*>(
            w_row[p] + 16u + size_t(lane4) * 8u);
        qs_w[p][0] = qsp[0];
        qs_w[p][1] = qsp[1];
      }
    }

    for (int base = 0; base < M; base += BK) {
      if (base > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (live) {
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
        dst[0] = av[0];
        dst[1] = av[1];
        dst[2] = av[2];
        dst[3] = av[3];
      }
      const int c = (base >> 6) & 3;
      const int g0 = 2 * c;
      const int g1 = g0 + 1;
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        if (w_live[p]) {
          const uint wr = row0 + uint(p) * 32u;
          float d0, m0, d1, m1;
          hy4_gguf_q4k_group_scale(sc_w[p], blk_d[p], blk_dmin[p], g0, d0, m0);
          hy4_gguf_q4k_group_scale(sc_w[p], blk_d[p], blk_dmin[p], g1, d1, m1);
          threadgroup half4* d_lo =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + lane4 * 8u);
          threadgroup half4* d_hi =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + 32u + lane4 * 8u);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            const uint32_t w = qs_w[p][i];
            d_lo[i] = half4(
                fma(float(w & 0x0Fu), d0, m0),
                fma(float((w >> 8) & 0x0Fu), d0, m0),
                fma(float((w >> 16) & 0x0Fu), d0, m0),
                fma(float((w >> 24) & 0x0Fu), d0, m0));
            const uint32_t wh = w >> 4;
            d_hi[i] = half4(
                fma(float(wh & 0x0Fu), d1, m1),
                fma(float((wh >> 8) & 0x0Fu), d1, m1),
                fma(float((wh >> 16) & 0x0Fu), d1, m1),
                fma(float((wh >> 24) & 0x0Fu), d1, m1));
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const int next = base + BK;
      if (next < M) {
        if (live) {
          if constexpr (IN_F16) {
            const device half4* src = reinterpret_cast<const device half4*>(
                hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
            av[0] = src[0];
            av[1] = src[1];
            av[2] = src[2];
            av[3] = src[3];
          } else {
            const device float4* src = reinterpret_cast<const device float4*>(
                hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
            av[0] = half4(src[0]);
            av[1] = half4(src[1]);
            av[2] = half4(src[2]);
            av[3] = half4(src[3]);
          }
        }
        const size_t nblk_off = size_t(next >> 8) * 144u;
        const size_t nq_off =
            16u + size_t((next >> 6) & 3) * 32u + size_t(lane4) * 8u;
        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < 2; ++p) {
          const device uint8_t* nblk = w_row[p] + nblk_off;
          if ((next & 255) == 0) {
            const uint32_t hdr = w_live[p]
                ? reinterpret_cast<const device uint32_t*>(nblk)[lane4]
                : 0u;
            const half2 dm = as_type<half2>(simd_shuffle(hdr, quad_base));
            blk_d[p] = float(dm.x);
            blk_dmin[p] = float(dm.y);
            sc_w[p][0] = simd_shuffle(hdr, quad_base | 1u);
            sc_w[p][1] = simd_shuffle(hdr, quad_base | 2u);
            sc_w[p][2] = simd_shuffle(hdr, quad_base | 3u);
          }
          if (w_live[p]) {
            const device uint32_t* qsp =
                reinterpret_cast<const device uint32_t*>(nblk + nq_off);
            qs_w[p][0] = qsp[0];
            qs_w[p][1] = qsp[1];
          }
        }
      }

      if (valid_m > 0 && valid_n > 0) {
        NAXTile<half, 1, 4> A;
        NAXTile<half, 2, 4> B;
        A.template load<half, PAD, 1>(As + tm * PAD);
        B.template load<half, PAD, 1>(Bs + tn * PAD);
        tile_matmad_nax(cd, A, metal::false_type{}, B, metal::true_type{});
      }
    }
  } else {
    // block_q6_K is 210 bytes (2-byte aligned only); use uint16_t/int8_t/half only.
    uint16_t ql0[2][4];
    uint16_t ql1[2][4];
    uint16_t qh[2][4];
    float sc[2][4];
    const size_t l_off = size_t(lane4) * 8u;
    const size_t sc_off = size_t(lane4 >> 1);

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      if (w_live[p]) {
        const device uint16_t* p_ql0 =
            reinterpret_cast<const device uint16_t*>(w_row[p] + l_off);
        const device uint16_t* p_ql1 = p_ql0 + 16;
        const device uint16_t* p_qh =
            reinterpret_cast<const device uint16_t*>(w_row[p] + 128u + l_off);
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          ql0[p][i] = p_ql0[i];
          ql1[p][i] = p_ql1[i];
          qh[p][i] = p_qh[i];
        }
        const float d =
            float(*reinterpret_cast<const device half*>(w_row[p] + 208u));
        const device int8_t* scp =
            reinterpret_cast<const device int8_t*>(w_row[p] + 192u) + sc_off;
        sc[p][0] = d * float(scp[0]);
        sc[p][1] = d * float(scp[2]);
        sc[p][2] = d * float(scp[4]);
        sc[p][3] = d * float(scp[6]);
      }
    }

    for (int base = 0; base < M; base += BK) {
      if (base > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (live) {
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
        dst[0] = av[0];
        dst[1] = av[1];
        dst[2] = av[2];
        dst[3] = av[3];
      }
      const int hi_half = (base >> 6) & 1;
      const int ql_sh = hi_half ? 4 : 0;
      const int qh_sh_a = hi_half ? 4 : 0;
      const int qh_sh_b = hi_half ? 6 : 2;
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        if (w_live[p]) {
          const uint wr = row0 + uint(p) * 32u;
          const float sc_a = hi_half ? sc[p][2] : sc[p][0];
          const float sc_b = hi_half ? sc[p][3] : sc[p][1];
          threadgroup half4* d_lo =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + lane4 * 8u);
          threadgroup half4* d_hi =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + 32u + lane4 * 8u);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            const uint16_t q0a = ql0[p][2 * i];
            const uint16_t q0b = ql0[p][2 * i + 1];
            const uint16_t q1a = ql1[p][2 * i];
            const uint16_t q1b = ql1[p][2 * i + 1];
            const uint16_t ha = qh[p][2 * i];
            const uint16_t hb = qh[p][2 * i + 1];

            const uint8_t h0 = uint8_t(ha & 0xFFu);
            const uint8_t h1 = uint8_t(ha >> 8);
            const uint8_t h2 = uint8_t(hb & 0xFFu);
            const uint8_t h3 = uint8_t(hb >> 8);

            const int a0 = int((((q0a & 0xFFu) >> ql_sh) & 0x0Fu) | (((h0 >> qh_sh_a) & 0x03u) << 4)) - 32;
            const int a1 = int((((q0a >> 8) >> ql_sh) & 0x0Fu) | (((h1 >> qh_sh_a) & 0x03u) << 4)) - 32;
            const int a2 = int((((q0b & 0xFFu) >> ql_sh) & 0x0Fu) | (((h2 >> qh_sh_a) & 0x03u) << 4)) - 32;
            const int a3 = int((((q0b >> 8) >> ql_sh) & 0x0Fu) | (((h3 >> qh_sh_a) & 0x03u) << 4)) - 32;
            d_lo[i] = half4(float(a0) * sc_a, float(a1) * sc_a, float(a2) * sc_a, float(a3) * sc_a);

            const int b0 = int((((q1a & 0xFFu) >> ql_sh) & 0x0Fu) | (((h0 >> qh_sh_b) & 0x03u) << 4)) - 32;
            const int b1 = int((((q1a >> 8) >> ql_sh) & 0x0Fu) | (((h1 >> qh_sh_b) & 0x03u) << 4)) - 32;
            const int b2 = int((((q1b & 0xFFu) >> ql_sh) & 0x0Fu) | (((h2 >> qh_sh_b) & 0x03u) << 4)) - 32;
            const int b3 = int((((q1b >> 8) >> ql_sh) & 0x0Fu) | (((h3 >> qh_sh_b) & 0x03u) << 4)) - 32;
            d_hi[i] = half4(float(b0) * sc_b, float(b1) * sc_b, float(b2) * sc_b, float(b3) * sc_b);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const int next = base + BK;
      if (next < M) {
        if (live) {
          if constexpr (IN_F16) {
            const device half4* src = reinterpret_cast<const device half4*>(
                hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
            av[0] = src[0];
            av[1] = src[1];
            av[2] = src[2];
            av[3] = src[3];
          } else {
            const device float4* src = reinterpret_cast<const device float4*>(
                hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
            av[0] = half4(src[0]);
            av[1] = half4(src[1]);
            av[2] = half4(src[2]);
            av[3] = half4(src[3]);
          }
        }
        if ((next & 127) == 0) {
          const size_t nblk_off = size_t(next >> 8) * 210u;
          const size_t nip = size_t((next >> 7) & 1);
          STEEL_PRAGMA_UNROLL
          for (int p = 0; p < 2; ++p) {
            if (w_live[p]) {
              const device uint8_t* nblk = w_row[p] + nblk_off;
              const device uint16_t* p_ql0 =
                  reinterpret_cast<const device uint16_t*>(
                      nblk + nip * 64u + l_off);
              const device uint16_t* p_ql1 = p_ql0 + 16;
              const device uint16_t* p_qh =
                  reinterpret_cast<const device uint16_t*>(
                      nblk + 128u + nip * 32u + l_off);
              STEEL_PRAGMA_UNROLL
              for (int i = 0; i < 4; ++i) {
                ql0[p][i] = p_ql0[i];
                ql1[p][i] = p_ql1[i];
                qh[p][i] = p_qh[i];
              }
              const float d =
                  float(*reinterpret_cast<const device half*>(nblk + 208u));
              const device int8_t* scp =
                  reinterpret_cast<const device int8_t*>(nblk + 192u) +
                  nip * 8u + sc_off;
              sc[p][0] = d * float(scp[0]);
              sc[p][1] = d * float(scp[2]);
              sc[p][2] = d * float(scp[4]);
              sc[p][3] = d * float(scp[6]);
            }
          }
        }
      }

      if (valid_m > 0 && valid_n > 0) {
        NAXTile<half, 1, 4> A;
        NAXTile<half, 2, 4> B;
        A.template load<half, PAD, 1>(As + tm * PAD);
        B.template load<half, PAD, 1>(Bs + tn * PAD);
        tile_matmad_nax(cd, A, metal::false_type{}, B, metal::true_type{});
      }
    }
  }
}

if (valid_m > 0 && valid_n > 0) {
  device float* out_ptr =
      out + size_t(begin + uint(tm)) * size_t(H) + size_t(col + uint(tn));
  if (valid_m == 16 && valid_n == 32) {
    cd.store(out_ptr, H);
  } else {
    cd.store_slice(out_ptr, H, short2(0, 0), short2(valid_n, valid_m));
  }
}
)METAL"