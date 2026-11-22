R"METAL(
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int PAD = 64;

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
const short tm = short((sg_idx / 2u) * 32u);
const short tn = short((sg_idx % 2u) * 32u);
const short valid_m = short(max(0, min(32, int(end - begin) - int(tm))));
const short valid_n = short(max(0, min(32, int(H) - int(col + uint(tn)))));

// As (8 KiB) + Bs (8 KiB) = 16 KiB (2 TGs/core).
alignas(16) threadgroup half As[BM * PAD];
alignas(16) threadgroup half Bs[BN * PAD];

NAXTile<float, 2, 2> cd;
cd.clear();

if (slot < uint(S)) {
  const uint ar0 = tid >> 2;
  const uint ac4 = tid & 3u;
  const uint ac = ac4 * 16u;
  const bool a_live[2] = {
      (begin + ar0) < end,
      (begin + ar0 + 32u) < end,
  };
  const uint sw_a0 = uint(hy4_swz_c8(int(ac4 * 2u), int(ar0)));
  const uint sw_a1 = uint(hy4_swz_c8(int(ac4 * 2u + 1u), int(ar0)));

  half4 av[2][4];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < 2; ++p) {
    const uint ar = ar0 + uint(p) * 32u;
    if (a_live[p]) {
      if constexpr (IN_F16) {
        const device float4* src = reinterpret_cast<const device float4*>(
            hidden + size_t(begin + ar) * size_t(M) + size_t(ac));
        const float4 r0 = src[0];
        const float4 r1 = src[1];
        av[p][0] = as_type<half4>(r0.xy);
        av[p][1] = as_type<half4>(r0.zw);
        av[p][2] = as_type<half4>(r1.xy);
        av[p][3] = as_type<half4>(r1.zw);
      } else {
        const device float4* src = reinterpret_cast<const device float4*>(
            hidden + size_t(begin + ar) * size_t(M) + size_t(ac));
        av[p][0] = half4(src[0]);
        av[p][1] = half4(src[1]);
        av[p][2] = half4(src[2]);
        av[p][3] = half4(src[3]);
      }
    } else {
      threadgroup half4* d0 =
          reinterpret_cast<threadgroup half4*>(As + ar * PAD + sw_a0 * 8u);
      threadgroup half4* d1 =
          reinterpret_cast<threadgroup half4*>(As + ar * PAD + sw_a1 * 8u);
      d0[0] = half4(0.0h);
      d0[1] = half4(0.0h);
      d1[0] = half4(0.0h);
      d1[1] = half4(0.0h);
    }
  }

  // 4 threads per weight row cooperatively load 8 consecutive bytes each = 32 B.
  const uint row0 = tid >> 2;
  const uint lane4 = tid & 3u;
  const ushort quad_base = ushort(lane & ~3u);
  const uint sw_lo = uint(hy4_swz_c8(int(lane4), int(row0)));
  const uint sw_hi = uint(hy4_swz_c8(int(4u + lane4), int(row0)));
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
          reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + sw_lo * 8u);
      threadgroup half4* d_hi =
          reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + sw_hi * 8u);
      d_lo[0] = half4(0.0h);
      d_lo[1] = half4(0.0h);
      d_hi[0] = half4(0.0h);
      d_hi[1] = half4(0.0h);
    }
  }

  const bool active = (valid_m > 0 && valid_n > 0);
  const short2 coord = BaseNAXFrag::get_coord();
  const short fm = coord.y;
  const short fn = coord.x;

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
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        if (a_live[p]) {
          const uint ar = ar0 + uint(p) * 32u;
          threadgroup half4* d0 =
              reinterpret_cast<threadgroup half4*>(As + ar * PAD + sw_a0 * 8u);
          threadgroup half4* d1 =
              reinterpret_cast<threadgroup half4*>(As + ar * PAD + sw_a1 * 8u);
          d0[0] = av[p][0];
          d0[1] = av[p][1];
          d1[0] = av[p][2];
          d1[1] = av[p][3];
        }
      }
      const int c = (base >> 6) & 3;
      const int g0 = 2 * c;
      const int g1 = g0 + 1;
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        float d0 = 0.0f, m0 = 0.0f, d1 = 0.0f, m1 = 0.0f;
        if (w_live[p] && lane4 == 0u) {
          hy4_gguf_q4k_group_scale(sc_w[p], blk_d[p], blk_dmin[p], g0, d0, m0);
          hy4_gguf_q4k_group_scale(sc_w[p], blk_d[p], blk_dmin[p], g1, d1, m1);
        }
        const float d0_f = simd_shuffle(d0, quad_base);
        const float m0_f = simd_shuffle(m0, quad_base);
        const float d1_f = simd_shuffle(d1, quad_base);
        const float m1_f = simd_shuffle(m1, quad_base);
        if (w_live[p]) {
          const uint wr = row0 + uint(p) * 32u;
          threadgroup half4* d_lo =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + sw_lo * 8u);
          threadgroup half4* d_hi =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + sw_hi * 8u);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            const uchar4 b = as_type<uchar4>(qs_w[p][i]);
            d_lo[i] = half4(metal::fma(float4(b & uchar4(0x0Fu)), d0_f, m0_f));
            d_hi[i] = half4(metal::fma(float4(b >> 4), d1_f, m1_f));
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const int next = base + BK;
      if (next < M) {
        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < 2; ++p) {
          if (a_live[p]) {
            const uint ar = ar0 + uint(p) * 32u;
            if constexpr (IN_F16) {
              const device float4* src = reinterpret_cast<const device float4*>(
                  hidden + size_t(begin + ar) * size_t(M) + size_t(next) +
                  size_t(ac));
              const float4 r0 = src[0];
              const float4 r1 = src[1];
              av[p][0] = as_type<half4>(r0.xy);
              av[p][1] = as_type<half4>(r0.zw);
              av[p][2] = as_type<half4>(r1.xy);
              av[p][3] = as_type<half4>(r1.zw);
            } else {
              const device float4* src = reinterpret_cast<const device float4*>(
                  hidden + size_t(begin + ar) * size_t(M) + size_t(next) +
                  size_t(ac));
              av[p][0] = half4(src[0]);
              av[p][1] = half4(src[1]);
              av[p][2] = half4(src[2]);
              av[p][3] = half4(src[3]);
            }
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

      if (active) {
        NAXTile<half, 2, 4> A;
        NAXTile<half, 2, 4> B;
        hy4_load_tile_swz64(A, As + size_t(tm) * PAD, fm, fn);
        hy4_load_tile_swz64(B, Bs + size_t(tn) * PAD, fm, fn);
        tile_matmad_nax(cd, A, metal::false_type{}, B, metal::true_type{});
      }
    }
  } else {
    // block_q6_K is 210 bytes (2-byte aligned ONLY); use packed_ushort4/half only.
    uchar4 ql0_v[2][2];
    uchar4 ql1_v[2][2];
    uchar4 qh_v[2][2];
    float4 sc[2];
    const size_t l_off = size_t(lane4) * 8u;

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      float d_val = 0.0f;
      uint2 sc_u2 = uint2(0u);
      if (w_live[p]) {
        const ushort4 v0 = ushort4(*reinterpret_cast<const device packed_ushort4*>(
            w_row[p] + l_off));
        const ushort4 v1 = ushort4(*reinterpret_cast<const device packed_ushort4*>(
            w_row[p] + 32u + l_off));
        const ushort4 vh = ushort4(*reinterpret_cast<const device packed_ushort4*>(
            w_row[p] + 128u + l_off));
        ql0_v[p][0] = as_type<uchar4>(v0.xy);
        ql0_v[p][1] = as_type<uchar4>(v0.zw);
        ql1_v[p][0] = as_type<uchar4>(v1.xy);
        ql1_v[p][1] = as_type<uchar4>(v1.zw);
        qh_v[p][0] = as_type<uchar4>(vh.xy);
        qh_v[p][1] = as_type<uchar4>(vh.zw);
        if (lane4 == 0u) {
          d_val = float(*reinterpret_cast<const device half*>(w_row[p] + 208u));
          const ushort4 sc_raw = ushort4(
              *reinterpret_cast<const device packed_ushort4*>(w_row[p] + 192u));
          sc_u2 = uint2(as_type<uint>(sc_raw.xy), as_type<uint>(sc_raw.zw));
        }
      }
      d_val = simd_shuffle(d_val, quad_base);
      sc_u2 = simd_shuffle(sc_u2, quad_base);
      if (w_live[p]) {
        const char4 c0 = as_type<char4>(sc_u2.x);
        const char4 c1 = as_type<char4>(sc_u2.y);
        const float4 sv = ((lane4 & 2u) == 0u)
            ? float4(float(c0.x), float(c0.z), float(c1.x), float(c1.z))
            : float4(float(c0.y), float(c0.w), float(c1.y), float(c1.w));
        sc[p] = sv * d_val;
      }
    }

    for (int base = 0; base < M; base += BK) {
      if (base > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        if (a_live[p]) {
          const uint ar = ar0 + uint(p) * 32u;
          threadgroup half4* d0 =
              reinterpret_cast<threadgroup half4*>(As + ar * PAD + sw_a0 * 8u);
          threadgroup half4* d1 =
              reinterpret_cast<threadgroup half4*>(As + ar * PAD + sw_a1 * 8u);
          d0[0] = av[p][0];
          d0[1] = av[p][1];
          d1[0] = av[p][2];
          d1[1] = av[p][3];
        }
      }
      const int hi_half = (base >> 6) & 1;
      const uchar ql_sh = hi_half ? 4u : 0u;
      const uchar qh_sh_a = hi_half ? 4u : 0u;
      const uchar qh_sh_b = hi_half ? 6u : 2u;
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        if (w_live[p]) {
          const uint wr = row0 + uint(p) * 32u;
          const float sc_a = hi_half ? sc[p].z : sc[p].x;
          const float sc_b = hi_half ? sc[p].w : sc[p].y;
          threadgroup half4* d_lo =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + sw_lo * 8u);
          threadgroup half4* d_hi =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + sw_hi * 8u);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            const uchar4 q0 =
                ((ql0_v[p][i] >> ql_sh) & uchar4(0x0Fu)) |
                (((qh_v[p][i] >> qh_sh_a) & uchar4(0x03u)) << 4);
            d_lo[i] = half4(float4(char4(q0) - char4(32)) * sc_a);
            const uchar4 q1 =
                ((ql1_v[p][i] >> ql_sh) & uchar4(0x0Fu)) |
                (((qh_v[p][i] >> qh_sh_b) & uchar4(0x03u)) << 4);
            d_hi[i] = half4(float4(char4(q1) - char4(32)) * sc_b);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const int next = base + BK;
      if (next < M) {
        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < 2; ++p) {
          if (a_live[p]) {
            const uint ar = ar0 + uint(p) * 32u;
            if constexpr (IN_F16) {
              const device float4* src = reinterpret_cast<const device float4*>(
                  hidden + size_t(begin + ar) * size_t(M) + size_t(next) +
                  size_t(ac));
              const float4 r0 = src[0];
              const float4 r1 = src[1];
              av[p][0] = as_type<half4>(r0.xy);
              av[p][1] = as_type<half4>(r0.zw);
              av[p][2] = as_type<half4>(r1.xy);
              av[p][3] = as_type<half4>(r1.zw);
            } else {
              const device float4* src = reinterpret_cast<const device float4*>(
                  hidden + size_t(begin + ar) * size_t(M) + size_t(next) +
                  size_t(ac));
              av[p][0] = half4(src[0]);
              av[p][1] = half4(src[1]);
              av[p][2] = half4(src[2]);
              av[p][3] = half4(src[3]);
            }
          }
        }
        if ((next & 127) == 0) {
          const size_t nblk_off = size_t(next >> 8) * 210u;
          const size_t nip = size_t((next >> 7) & 1);
          STEEL_PRAGMA_UNROLL
          for (int p = 0; p < 2; ++p) {
            float d_val = 0.0f;
            uint2 sc_u2 = uint2(0u);
            if (w_live[p]) {
              const device uint8_t* nblk = w_row[p] + nblk_off;
              const ushort4 v0 = ushort4(*reinterpret_cast<const device packed_ushort4*>(
                  nblk + nip * 64u + l_off));
              const ushort4 v1 = ushort4(*reinterpret_cast<const device packed_ushort4*>(
                  nblk + nip * 64u + 32u + l_off));
              const ushort4 vh = ushort4(*reinterpret_cast<const device packed_ushort4*>(
                  nblk + 128u + nip * 32u + l_off));
              ql0_v[p][0] = as_type<uchar4>(v0.xy);
              ql0_v[p][1] = as_type<uchar4>(v0.zw);
              ql1_v[p][0] = as_type<uchar4>(v1.xy);
              ql1_v[p][1] = as_type<uchar4>(v1.zw);
              qh_v[p][0] = as_type<uchar4>(vh.xy);
              qh_v[p][1] = as_type<uchar4>(vh.zw);
              if (lane4 == 0u) {
                d_val = float(*reinterpret_cast<const device half*>(nblk + 208u));
                const ushort4 sc_raw = ushort4(
                    *reinterpret_cast<const device packed_ushort4*>(
                        nblk + 192u + nip * 8u));
                sc_u2 = uint2(as_type<uint>(sc_raw.xy), as_type<uint>(sc_raw.zw));
              }
            }
            d_val = simd_shuffle(d_val, quad_base);
            sc_u2 = simd_shuffle(sc_u2, quad_base);
            if (w_live[p]) {
              const char4 c0 = as_type<char4>(sc_u2.x);
              const char4 c1 = as_type<char4>(sc_u2.y);
              const float4 sv = ((lane4 & 2u) == 0u)
                  ? float4(float(c0.x), float(c0.z), float(c1.x), float(c1.z))
                  : float4(float(c0.y), float(c0.w), float(c1.y), float(c1.w));
              sc[p] = sv * d_val;
            }
          }
        }
      }

      if (active) {
        NAXTile<half, 2, 4> A;
        NAXTile<half, 2, 4> B;
        hy4_load_tile_swz64(A, As + size_t(tm) * PAD, fm, fn);
        hy4_load_tile_swz64(B, Bs + size_t(tn) * PAD, fm, fn);
        tile_matmad_nax(cd, A, metal::false_type{}, B, metal::true_type{});
      }
    }
  }
}

if (valid_m > 0 && valid_n > 0) {
  auto out_ptr =
      out + size_t(begin + uint(tm)) * size_t(H) + size_t(col + uint(tn));
  if (valid_m == 32 && valid_n == 32) {
    cd.store(out_ptr, H);
  } else {
    cd.store_slice(out_ptr, H, short2(0, 0), short2(valid_n, valid_m));
  }
}
)METAL"