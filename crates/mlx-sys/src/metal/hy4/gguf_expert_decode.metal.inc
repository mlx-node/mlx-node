R"METAL(
struct Hy4GgufQ4kRow {
  half2 dm;
  uint16_t sc16[4];
  uint32_t q1;
  uint32_t q2;
};

inline Hy4GgufQ4kRow hy4_gguf_q4k_load(
    const device uint8_t* blk,
    short iq,
    short ir) {
  Hy4GgufQ4kRow w;
  w.dm = *reinterpret_cast<const device half2*>(blk);
  const device uint16_t* sc =
      reinterpret_cast<const device uint16_t*>(blk + 4) + iq;
  const uint16_t s0 = sc[0];
  const uint16_t s2 = sc[2];
  const uint16_t s4 = sc[4];
  w.sc16[0] = s0 & 0x3f3fu;
  w.sc16[1] = s2 & 0x3f3fu;
  w.sc16[2] = (s4 & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2);
  w.sc16[3] = ((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2);
  const device uint32_t* q32 =
      reinterpret_cast<const device uint32_t*>(blk + 16) + 8 * iq + ir;
  w.q1 = q32[0];
  w.q2 = q32[16];
  return w;
}

inline float hy4_gguf_q4k_dot(
    thread const Hy4GgufQ4kRow& w,
    float4 v0,
    float4 v1,
    float4 v2,
    float4 v3,
    float4 sumy) {
  float4 a1 = float4(0.0f);
  float4 a2 = float4(0.0f);
  {
    const uint16_t p1 = uint16_t(w.q1 & 0xFFFFu);
    const uint16_t p2 = uint16_t(w.q2 & 0xFFFFu);
    a1[0] += v0[0] * float(p1 & 0x000Fu);
    a1[1] += v0[1] * float(p1 & 0x0F00u);
    a1[2] += v1[0] * float(p1 & 0x00F0u);
    a1[3] += v1[1] * float(p1 & 0xF000u);
    a2[0] += v2[0] * float(p2 & 0x000Fu);
    a2[1] += v2[1] * float(p2 & 0x0F00u);
    a2[2] += v3[0] * float(p2 & 0x00F0u);
    a2[3] += v3[1] * float(p2 & 0xF000u);
  }
  {
    const uint16_t p1 = uint16_t(w.q1 >> 16);
    const uint16_t p2 = uint16_t(w.q2 >> 16);
    a1[0] += v0[2] * float(p1 & 0x000Fu);
    a1[1] += v0[3] * float(p1 & 0x0F00u);
    a1[2] += v1[2] * float(p1 & 0x00F0u);
    a1[3] += v1[3] * float(p1 & 0xF000u);
    a2[0] += v2[2] * float(p2 & 0x000Fu);
    a2[1] += v2[3] * float(p2 & 0x0F00u);
    a2[2] += v3[2] * float(p2 & 0x00F0u);
    a2[3] += v3[3] * float(p2 & 0xF000u);
  }
  const float s0 = float(w.sc16[0] & 0xFFu);
  const float s1 = float(w.sc16[0] >> 8);
  const float m0 = float(w.sc16[1] & 0xFFu);
  const float m1 = float(w.sc16[1] >> 8);
  const float s4 = float(w.sc16[2] & 0xFFu);
  const float s5 = float(w.sc16[2] >> 8);
  const float m4 = float(w.sc16[3] & 0xFFu);
  const float m5 = float(w.sc16[3] >> 8);
  return float(w.dm.x) * (
      (a1[0] + (1.0f / 256.0f) * a1[1]) * s0 +
      (a1[2] + (1.0f / 256.0f) * a1[3]) * s1 * (1.0f / 16.0f) +
      (a2[0] + (1.0f / 256.0f) * a2[1]) * s4 +
      (a2[2] + (1.0f / 256.0f) * a2[3]) * s5 * (1.0f / 16.0f))
    - float(w.dm.y) * (
      sumy[0] * m0 + sumy[1] * m1 + sumy[2] * m4 + sumy[3] * m5);
}

struct Hy4GgufQ6kRow {
  half d;
  uint16_t q1[2];
  uint16_t q2[2];
  uint16_t qh[2];
  int8_t sc[4];
};

// block_q6_K is 210 bytes (2-byte aligned only); never cast to uint32_t*.
inline Hy4GgufQ6kRow hy4_gguf_q6k_load(
    const device uint8_t* blk,
    short ip,
    short l0,
    short is) {
  Hy4GgufQ6kRow w;
  const device uint16_t* q1_16 =
      reinterpret_cast<const device uint16_t*>(blk + 64 * ip + l0);
  const device uint16_t* q2_16 = q1_16 + 16;
  const device uint16_t* qh_16 =
      reinterpret_cast<const device uint16_t*>(blk + 128 + 32 * ip + l0);
  const device uint16_t* sc16 =
      reinterpret_cast<const device uint16_t*>(blk + 192 + (is & ~1));
  const uint16_t sc_shift = uint16_t((is & 1) * 8);
  w.q1[0] = q1_16[0];
  w.q1[1] = q1_16[1];
  w.q2[0] = q2_16[0];
  w.q2[1] = q2_16[1];
  w.qh[0] = qh_16[0];
  w.qh[1] = qh_16[1];
  w.sc[0] = int8_t((sc16[0] >> sc_shift) & 0xFFu);
  w.sc[1] = int8_t((sc16[1] >> sc_shift) & 0xFFu);
  w.sc[2] = int8_t((sc16[2] >> sc_shift) & 0xFFu);
  w.sc[3] = int8_t((sc16[3] >> sc_shift) & 0xFFu);
  w.d = *reinterpret_cast<const device half*>(blk + 208);
  return w;
}

inline float hy4_gguf_q6k_dot(
    thread const Hy4GgufQ6kRow& w,
    float4 y0,
    float4 y1,
    float4 y2,
    float4 y3,
    float4 sumy) {
  float4 s = float4(0.0f);
#pragma unroll
  for (short p = 0; p < 2; ++p) {
    const uint16_t w1 = w.q1[p];
    const uint16_t w2 = w.q2[p];
    const uint16_t wh = w.qh[p];

    const uint b1_0 = uint(w1 & 0xFFu);
    const uint b2_0 = uint(w2 & 0xFFu);
    const uint bh_0 = uint(wh & 0xFFu);
    s[0] += y0[2 * p + 0] * float((b1_0 & 0x0Fu) | ((bh_0 & 0x03u) << 4));
    s[1] += y1[2 * p + 0] * float((b2_0 & 0x0Fu) | ((bh_0 & 0x0Cu) << 2));
    s[2] += y2[2 * p + 0] * float((b1_0 >> 4) | (bh_0 & 0x30u));
    s[3] += y3[2 * p + 0] * float((b2_0 >> 4) | ((bh_0 & 0xC0u) >> 2));

    const uint b1_1 = uint(w1 >> 8);
    const uint b2_1 = uint(w2 >> 8);
    const uint bh_1 = uint(wh >> 8);
    s[0] += y0[2 * p + 1] * float((b1_1 & 0x0Fu) | ((bh_1 & 0x03u) << 4));
    s[1] += y1[2 * p + 1] * float((b2_1 & 0x0Fu) | ((bh_1 & 0x0Cu) << 2));
    s[2] += y2[2 * p + 1] * float((b1_1 >> 4) | (bh_1 & 0x30u));
    s[3] += y3[2 * p + 1] * float((b2_1 >> 4) | ((bh_1 & 0xC0u) >> 2));
  }
  s = metal::fma(sumy, float4(-32.0f), s);
  return float(w.d) * (
      s[0] * float(w.sc[0]) + s[1] * float(w.sc[1]) +
      s[2] * float(w.sc[2]) + s[3] * float(w.sc[3]));
}

template <
    bool FAST,
    bool DEDUP,
    int MAX_MATCH,
    int NSG,
    int RPS,
    typename RowIdsPtr,
    typename SlotsPtr>
inline void hy4_gguf_gate_up_swiglu_impl(
    const device float* x,
    RowIdsPtr row_ids,
    SlotsPtr slots,
    const device uint* dedup,
    const device uint8_t* gate_buf,
    uint64_t gate_off,
    const device uint8_t* up_buf,
    uint64_t up_off,
    float lim,
    device float* out,
    int ROWS,
    int N_ASSIGN,
    int K,
    size_t x_row_stride,
    int M,
    int S,
    uint assign,
    uint tile,
    uint simd,
    uint lane) {
  const int first = int(tile) * (NSG * RPS) + int(simd) * RPS;
  if (assign >= uint(N_ASSIGN) || first >= M) return;

  const int nb = K / 256;
  const size_t row_bytes = size_t(nb) * 144u;
  const short ix = short(lane >> 4);
  const short it = short(lane & 15u);
  const short iq = it >> 3;
  const short ir = it & 7;
  const size_t y_lane_off = size_t(64 * iq + 4 * ir);

  if constexpr (DEDUP) {
    // dedup4[0].x = n_uniq; dedup4[1 + u] = {expert, count(1..4), packed_assign, packed_row}.
    const device uint4* dedup4 = reinterpret_cast<const device uint4*>(dedup);
    const uint n_uniq = dedup4[0].x;
    if (assign >= n_uniq) return;

    const uint4 ent = dedup4[1u + assign];
    const uint slot = ent.x;
    const uint cnt = ent.y;
    const uint packed_a = ent.z;
    const uint packed_r = ent.w;

    if (slot >= uint(S)) {
      if (lane == 0u) {
        for (uint m = 0u; m < cnt; ++m) {
          const uint dst_a = (packed_a >> (m * 8u)) & 0xFFu;
#pragma unroll
          for (int r = 0; r < RPS; ++r) {
            if (FAST || first + r < M) {
              out[size_t(dst_a) * size_t(M) + size_t(first + r)] = 0.0f;
            }
          }
        }
      }
      return;
    }

    const size_t base_off = (size_t(slot) * size_t(M) + size_t(first)) * row_bytes;
    const device uint8_t* g_base = gate_buf + size_t(gate_off) + base_off;
    const device uint8_t* u_base = up_buf + size_t(up_off) + base_off;

    if (cnt == 1u) {
      const uint dst_a = packed_a & 0xFFu;
      const uint rid = packed_r & 0xFFu;
      float g[RPS];
      float u_acc[RPS];
#pragma unroll
      for (int r = 0; r < RPS; ++r) {
        g[r] = 0.0f;
        u_acc[r] = 0.0f;
      }

      if (rid < uint(ROWS)) {
        const device float* x_row = x + size_t(rid) * x_row_stride + y_lane_off;
        for (int ib = int(ix); ib < nb; ib += 2) {
          const device float* y4 = x_row + size_t(ib) * 256u;
          const float4 v0 = *reinterpret_cast<const device float4*>(y4 + 0);
          const float4 v1 = *reinterpret_cast<const device float4*>(y4 + 32);
          const float4 v2 = *reinterpret_cast<const device float4*>(y4 + 128);
          const float4 v3 = *reinterpret_cast<const device float4*>(y4 + 160);
          const float4 sumy = float4(
              v0[0] + v0[1] + v0[2] + v0[3],
              v1[0] + v1[1] + v1[2] + v1[3],
              v2[0] + v2[1] + v2[2] + v2[3],
              v3[0] + v3[1] + v3[2] + v3[3]);
          const size_t blk_off = size_t(ib) * 144u;
#pragma unroll
          for (int r = 0; r < RPS; ++r) {
            if (FAST || first + r < M) {
              const Hy4GgufQ4kRow gw =
                  hy4_gguf_q4k_load(g_base + size_t(r) * row_bytes + blk_off, iq, ir);
              g[r] += hy4_gguf_q4k_dot(gw, v0, v1, v2, v3, sumy);
              const Hy4GgufQ4kRow uw =
                  hy4_gguf_q4k_load(u_base + size_t(r) * row_bytes + blk_off, iq, ir);
              u_acc[r] += hy4_gguf_q4k_dot(uw, v0, v1, v2, v3, sumy);
            }
          }
        }
      }

#pragma unroll
      for (int r = 0; r < RPS; ++r) {
        if (FAST || first + r < M) {
          float gv = simd_sum(g[r]);
          float uv = simd_sum(u_acc[r]);
          if (lane == 0u) {
            // Asymmetric clamp: never lower-clamp gate.
            if (lim > 0.0f) {
              gv = metal::min(gv, lim);
              uv = metal::clamp(uv, -lim, lim);
            }
            const float sig = 1.0f / (1.0f + metal::exp(-gv));
            out[size_t(dst_a) * size_t(M) + size_t(first + r)] = (sig * gv) * uv;
          }
        }
      }
      return;
    }

    // Load one weight row at a time so gw/uw rows never overlap in registers.
    float g[MAX_MATCH][RPS];
    float u_acc[MAX_MATCH][RPS];
#pragma unroll
    for (int m = 0; m < MAX_MATCH; ++m) {
#pragma unroll
      for (int r = 0; r < RPS; ++r) {
        g[m][r] = 0.0f;
        u_acc[m][r] = 0.0f;
      }
    }

    for (int ib = int(ix); ib < nb; ib += 2) {
      const size_t blk_off = size_t(ib) * 144u;
      const size_t x_ib_off = size_t(ib) * 256u + y_lane_off;
#pragma unroll
      for (int r = 0; r < RPS; ++r) {
        if (FAST || first + r < M) {
          {
            const Hy4GgufQ4kRow gw =
                hy4_gguf_q4k_load(g_base + size_t(r) * row_bytes + blk_off, iq, ir);
#pragma unroll
            for (int m = 0; m < MAX_MATCH; ++m) {
              if (uint(m) < cnt) {
                const uint rid = (packed_r >> (uint(m) * 8u)) & 0xFFu;
                if (rid < uint(ROWS)) {
                  const device float* y4 = x + size_t(rid) * x_row_stride + x_ib_off;
                  const float4 v0 = *reinterpret_cast<const device float4*>(y4 + 0);
                  const float4 v1 = *reinterpret_cast<const device float4*>(y4 + 32);
                  const float4 v2 = *reinterpret_cast<const device float4*>(y4 + 128);
                  const float4 v3 = *reinterpret_cast<const device float4*>(y4 + 160);
                  const float4 sumy = float4(
                      v0[0] + v0[1] + v0[2] + v0[3],
                      v1[0] + v1[1] + v1[2] + v1[3],
                      v2[0] + v2[1] + v2[2] + v2[3],
                      v3[0] + v3[1] + v3[2] + v3[3]);
                  g[m][r] += hy4_gguf_q4k_dot(gw, v0, v1, v2, v3, sumy);
                }
              }
            }
          }
          {
            const Hy4GgufQ4kRow uw =
                hy4_gguf_q4k_load(u_base + size_t(r) * row_bytes + blk_off, iq, ir);
#pragma unroll
            for (int m = 0; m < MAX_MATCH; ++m) {
              if (uint(m) < cnt) {
                const uint rid = (packed_r >> (uint(m) * 8u)) & 0xFFu;
                if (rid < uint(ROWS)) {
                  const device float* y4 = x + size_t(rid) * x_row_stride + x_ib_off;
                  const float4 v0 = *reinterpret_cast<const device float4*>(y4 + 0);
                  const float4 v1 = *reinterpret_cast<const device float4*>(y4 + 32);
                  const float4 v2 = *reinterpret_cast<const device float4*>(y4 + 128);
                  const float4 v3 = *reinterpret_cast<const device float4*>(y4 + 160);
                  const float4 sumy = float4(
                      v0[0] + v0[1] + v0[2] + v0[3],
                      v1[0] + v1[1] + v1[2] + v1[3],
                      v2[0] + v2[1] + v2[2] + v2[3],
                      v3[0] + v3[1] + v3[2] + v3[3]);
                  u_acc[m][r] += hy4_gguf_q4k_dot(uw, v0, v1, v2, v3, sumy);
                }
              }
            }
          }
        }
      }
    }

#pragma unroll
    for (int m = 0; m < MAX_MATCH; ++m) {
      if (uint(m) < cnt) {
        const uint dst_a = (packed_a >> (uint(m) * 8u)) & 0xFFu;
#pragma unroll
        for (int r = 0; r < RPS; ++r) {
          if (FAST || first + r < M) {
            float gv = simd_sum(g[m][r]);
            float uv = simd_sum(u_acc[m][r]);
            if (lane == 0u) {
              // Asymmetric clamp: never lower-clamp gate.
              if (lim > 0.0f) {
                gv = metal::min(gv, lim);
                uv = metal::clamp(uv, -lim, lim);
              }
              const float sig = 1.0f / (1.0f + metal::exp(-gv));
              out[size_t(dst_a) * size_t(M) + size_t(first + r)] = (sig * gv) * uv;
            }
          }
        }
      }
    }
    return;
  }

  const uint slot = slots[assign];
  const uint row_id = row_ids[assign];
  float g[RPS];
  float u[RPS];
#pragma unroll
  for (int r = 0; r < RPS; ++r) {
    g[r] = 0.0f;
    u[r] = 0.0f;
  }

  if (slot < uint(S) && row_id < uint(ROWS)) {
    const size_t base_off = (size_t(slot) * size_t(M) + size_t(first)) * row_bytes;
    const device uint8_t* g_base = gate_buf + size_t(gate_off) + base_off;
    const device uint8_t* u_base = up_buf + size_t(up_off) + base_off;
    const device float* x_row = x + size_t(row_id) * x_row_stride + y_lane_off;

    for (int ib = int(ix); ib < nb; ib += 2) {
      const device float* y4 = x_row + size_t(ib) * 256u;
      const float4 v0 = *reinterpret_cast<const device float4*>(y4 + 0);
      const float4 v1 = *reinterpret_cast<const device float4*>(y4 + 32);
      const float4 v2 = *reinterpret_cast<const device float4*>(y4 + 128);
      const float4 v3 = *reinterpret_cast<const device float4*>(y4 + 160);
      const float4 sumy = float4(
          v0[0] + v0[1] + v0[2] + v0[3],
          v1[0] + v1[1] + v1[2] + v1[3],
          v2[0] + v2[1] + v2[2] + v2[3],
          v3[0] + v3[1] + v3[2] + v3[3]);
      const size_t blk_off = size_t(ib) * 144u;
#pragma unroll
      for (int r = 0; r < RPS; r += 2) {
        Hy4GgufQ4kRow gw0, uw0, gw1, uw1;
        if (FAST || first + r < M) {
          gw0 = hy4_gguf_q4k_load(g_base + size_t(r) * row_bytes + blk_off, iq, ir);
          uw0 = hy4_gguf_q4k_load(u_base + size_t(r) * row_bytes + blk_off, iq, ir);
        }
        if (FAST || first + r + 1 < M) {
          gw1 = hy4_gguf_q4k_load(g_base + size_t(r + 1) * row_bytes + blk_off, iq, ir);
          uw1 = hy4_gguf_q4k_load(u_base + size_t(r + 1) * row_bytes + blk_off, iq, ir);
        }
        if (FAST || first + r < M) {
          g[r] += hy4_gguf_q4k_dot(gw0, v0, v1, v2, v3, sumy);
          u[r] += hy4_gguf_q4k_dot(uw0, v0, v1, v2, v3, sumy);
        }
        if (FAST || first + r + 1 < M) {
          g[r + 1] += hy4_gguf_q4k_dot(gw1, v0, v1, v2, v3, sumy);
          u[r + 1] += hy4_gguf_q4k_dot(uw1, v0, v1, v2, v3, sumy);
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RPS; ++r) {
    if (FAST || first + r < M) {
      float gv = simd_sum(g[r]);
      float uv = simd_sum(u[r]);
      if (lane == 0) {
        // Asymmetric clamp: never lower-clamp gate.
        if (lim > 0.0f) {
          gv = metal::min(gv, lim);
          uv = metal::clamp(uv, -lim, lim);
        }
        const float sig = 1.0f / (1.0f + metal::exp(-gv));
        out[size_t(assign) * size_t(M) + size_t(first + r)] = (sig * gv) * uv;
      }
    }
  }
}

template <
    int BITS,
    int TOP,
    bool FAST,
    bool DEDUP_ROWS,
    int MAX_MATCH,
    typename SlotsPtr,
    typename WeightsPtr>
inline void hy4_gguf_down_combine_impl(
    const device float* hidden,
    SlotsPtr slots,
    WeightsPtr weights,
    const device uint* dedup,
    const device uint8_t* down_buf,
    uint64_t down_off,
    device float* out,
    threadgroup float* products,
    int ROWS,
    int M,
    int H,
    int S,
    uint row,
    uint tile,
    uint simd,
    uint lane) {
  constexpr int RPS = 4;
  constexpr int KSG = 8;

  const int first = int(tile) * RPS;
  if (first >= H) return;

  const int nb = M / 256;
  constexpr size_t BLK_BYTES = (BITS == 6) ? 210u : 144u;
  const size_t row_bytes = size_t(nb) * BLK_BYTES;

  if constexpr (DEDUP_ROWS) {
    const device uint4* dedup4 = reinterpret_cast<const device uint4*>(dedup);
    const uint n_uniq = dedup4[0].x;
    for (uint u = simd; u < n_uniq; u += uint(KSG)) {
      const uint4 ent = dedup4[1u + u];
      const uint slot = ent.x;
      const uint cnt = ent.y;
      const uint packed_a = ent.z;

      if (slot >= uint(S)) {
        if (lane == 0u) {
          for (uint m = 0u; m < cnt; ++m) {
            const uint dst_a = (packed_a >> (m * 8u)) & 0xFFu;
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              products[dst_a * uint(RPS) + uint(r)] = 0.0f;
            }
          }
        }
        continue;
      }

      const device uint8_t* d_base =
          down_buf + size_t(down_off) +
          (size_t(slot) * size_t(H) + size_t(first)) * row_bytes;

      if (cnt == 1u) {
        const uint a0 = packed_a & 0xFFu;
        const device float* h_row = hidden + size_t(a0) * size_t(M);
        float acc[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};
        if constexpr (BITS == 4) {
          const short ix = short(lane >> 4);
          const short it = short(lane & 15u);
          const short iq = it >> 3;
          const short ir = it & 7;
          const device float* x_lane = h_row + size_t(64 * iq + 4 * ir);
          for (int ib = int(ix); ib < nb; ib += 2) {
            const device float* y4 = x_lane + size_t(ib) * 256u;
            const float4 v0 = *reinterpret_cast<const device float4*>(y4 + 0);
            const float4 v1 = *reinterpret_cast<const device float4*>(y4 + 32);
            const float4 v2 = *reinterpret_cast<const device float4*>(y4 + 128);
            const float4 v3 = *reinterpret_cast<const device float4*>(y4 + 160);
            const float4 sumy = float4(
                v0[0] + v0[1] + v0[2] + v0[3],
                v1[0] + v1[1] + v1[2] + v1[3],
                v2[0] + v2[1] + v2[2] + v2[3],
                v3[0] + v3[1] + v3[2] + v3[3]);
            const size_t blk_off = size_t(ib) * 144u;
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST || first + r < H) {
                const Hy4GgufQ4kRow dw =
                    hy4_gguf_q4k_load(d_base + size_t(r) * row_bytes + blk_off, iq, ir);
                acc[r] += hy4_gguf_q4k_dot(dw, v0, v1, v2, v3, sumy);
              }
            }
          }
        } else {
          const short ix = short(lane >> 4);
          const short tid_l = short(lane & 15u);
          const short ip = tid_l >> 3;
          const short il = tid_l & 7;
          const short l0 = 4 * il;
          const short is = 8 * ip + (l0 >> 4);
          const device float* x_lane = h_row + size_t(128 * ip + l0);
          for (int ib = int(ix); ib < nb; ib += 2) {
            const device float* y = x_lane + size_t(ib) * 256u;
            const float4 y0 = *reinterpret_cast<const device float4*>(y + 0);
            const float4 y1 = *reinterpret_cast<const device float4*>(y + 32);
            const float4 y2 = *reinterpret_cast<const device float4*>(y + 64);
            const float4 y3 = *reinterpret_cast<const device float4*>(y + 96);
            const float4 sumy = float4(
                y0[0] + y0[1] + y0[2] + y0[3],
                y1[0] + y1[1] + y1[2] + y1[3],
                y2[0] + y2[1] + y2[2] + y2[3],
                y3[0] + y3[1] + y3[2] + y3[3]);
            const size_t blk_off = size_t(ib) * 210u;
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST || first + r < H) {
                const Hy4GgufQ6kRow dw =
                    hy4_gguf_q6k_load(d_base + size_t(r) * row_bytes + blk_off, ip, l0, is);
                acc[r] += hy4_gguf_q6k_dot(dw, y0, y1, y2, y3, sumy);
              }
            }
          }
        }
        const float weight = weights[a0];
#pragma unroll
        for (int r = 0; r < RPS; ++r) {
          if (FAST || first + r < H) {
            const float val = simd_sum(acc[r]);
            if (lane == 0u) {
              products[a0 * uint(RPS) + uint(r)] = val * weight;
            }
          }
        }
      } else {
        // Load one weight row at a time so multi-match accumulation stays spill-free.
        float acc[MAX_MATCH][RPS];
#pragma unroll
        for (int m = 0; m < MAX_MATCH; ++m) {
#pragma unroll
          for (int r = 0; r < RPS; ++r) {
            acc[m][r] = 0.0f;
          }
        }
        if constexpr (BITS == 4) {
          const short ix = short(lane >> 4);
          const short it = short(lane & 15u);
          const short iq = it >> 3;
          const short ir = it & 7;
          const size_t x_lane_off = size_t(64 * iq + 4 * ir);
          for (int ib = int(ix); ib < nb; ib += 2) {
            const size_t blk_off = size_t(ib) * 144u;
            const size_t y_ib_off = size_t(ib) * 256u + x_lane_off;
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST || first + r < H) {
                const Hy4GgufQ4kRow dw =
                    hy4_gguf_q4k_load(d_base + size_t(r) * row_bytes + blk_off, iq, ir);
#pragma unroll
                for (int m = 0; m < MAX_MATCH; ++m) {
                  if (uint(m) < cnt) {
                    const uint a_m = (packed_a >> (uint(m) * 8u)) & 0xFFu;
                    const device float* y4 = hidden + size_t(a_m) * size_t(M) + y_ib_off;
                    const float4 v0 = *reinterpret_cast<const device float4*>(y4 + 0);
                    const float4 v1 = *reinterpret_cast<const device float4*>(y4 + 32);
                    const float4 v2 = *reinterpret_cast<const device float4*>(y4 + 128);
                    const float4 v3 = *reinterpret_cast<const device float4*>(y4 + 160);
                    const float4 sumy = float4(
                        v0[0] + v0[1] + v0[2] + v0[3],
                        v1[0] + v1[1] + v1[2] + v1[3],
                        v2[0] + v2[1] + v2[2] + v2[3],
                        v3[0] + v3[1] + v3[2] + v3[3]);
                    acc[m][r] += hy4_gguf_q4k_dot(dw, v0, v1, v2, v3, sumy);
                  }
                }
              }
            }
          }
        } else {
          const short ix = short(lane >> 4);
          const short tid_l = short(lane & 15u);
          const short ip = tid_l >> 3;
          const short il = tid_l & 7;
          const short l0 = 4 * il;
          const short is = 8 * ip + (l0 >> 4);
          const size_t x_lane_off = size_t(128 * ip + l0);
          for (int ib = int(ix); ib < nb; ib += 2) {
            const size_t blk_off = size_t(ib) * 210u;
            const size_t y_ib_off = size_t(ib) * 256u + x_lane_off;
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST || first + r < H) {
                const Hy4GgufQ6kRow dw =
                    hy4_gguf_q6k_load(d_base + size_t(r) * row_bytes + blk_off, ip, l0, is);
#pragma unroll
                for (int m = 0; m < MAX_MATCH; ++m) {
                  if (uint(m) < cnt) {
                    const uint a_m = (packed_a >> (uint(m) * 8u)) & 0xFFu;
                    const device float* y = hidden + size_t(a_m) * size_t(M) + y_ib_off;
                    const float4 y0 = *reinterpret_cast<const device float4*>(y + 0);
                    const float4 y1 = *reinterpret_cast<const device float4*>(y + 32);
                    const float4 y2 = *reinterpret_cast<const device float4*>(y + 64);
                    const float4 y3 = *reinterpret_cast<const device float4*>(y + 96);
                    const float4 sumy = float4(
                        y0[0] + y0[1] + y0[2] + y0[3],
                        y1[0] + y1[1] + y1[2] + y1[3],
                        y2[0] + y2[1] + y2[2] + y2[3],
                        y3[0] + y3[1] + y3[2] + y3[3]);
                    acc[m][r] += hy4_gguf_q6k_dot(dw, y0, y1, y2, y3, sumy);
                  }
                }
              }
            }
          }
        }
#pragma unroll
        for (int m = 0; m < MAX_MATCH; ++m) {
          if (uint(m) < cnt) {
            const uint dst_a = (packed_a >> (uint(m) * 8u)) & 0xFFu;
            const float weight = weights[dst_a];
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST || first + r < H) {
                const float val = simd_sum(acc[m][r]);
                if (lane == 0u) {
                  products[dst_a * uint(RPS) + uint(r)] = val * weight;
                }
              }
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Match MLX col_reduce_small's 8-lane modulo fold.
    for (uint r_idx = simd; r_idx < uint(ROWS); r_idx += uint(KSG)) {
      if (lane < uint(RPS)) {
        if (FAST || first + int(lane) < H) {
          constexpr int LANES = (TOP < 8) ? TOP : 8;
          const uint row_base = r_idx * uint(TOP);
          float totals[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
          for (int slot_idx = 0; slot_idx < TOP; ++slot_idx) {
            totals[slot_idx % 8] += products[(row_base + uint(slot_idx)) * uint(RPS) + lane];
          }
          float total = totals[0];
          for (int i = 1; i < LANES; ++i) {
            total += totals[i];
          }
          out[size_t(r_idx) * size_t(H) + size_t(first) + size_t(lane)] = total;
        }
      }
    }
    return;
  }

  if (row >= uint(ROWS)) return;

  for (int slot_idx = int(simd); slot_idx < TOP; slot_idx += KSG) {
    const size_t assign = size_t(row) * size_t(TOP) + size_t(slot_idx);
    const uint slot = slots[assign];
    float acc[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (slot < uint(S)) {
      const device uint8_t* d_base =
          down_buf + size_t(down_off) +
          (size_t(slot) * size_t(H) + size_t(first)) * row_bytes;
      const device float* h_row = hidden + assign * size_t(M);

      if constexpr (BITS == 4) {
        const short ix = short(lane >> 4);
        const short it = short(lane & 15u);
        const short iq = it >> 3;
        const short ir = it & 7;
        const device float* x_lane = h_row + size_t(64 * iq + 4 * ir);
        for (int ib = int(ix); ib < nb; ib += 2) {
          const device float* y4 = x_lane + size_t(ib) * 256u;
          const float4 v0 = *reinterpret_cast<const device float4*>(y4 + 0);
          const float4 v1 = *reinterpret_cast<const device float4*>(y4 + 32);
          const float4 v2 = *reinterpret_cast<const device float4*>(y4 + 128);
          const float4 v3 = *reinterpret_cast<const device float4*>(y4 + 160);
          const float4 sumy = float4(
              v0[0] + v0[1] + v0[2] + v0[3],
              v1[0] + v1[1] + v1[2] + v1[3],
              v2[0] + v2[1] + v2[2] + v2[3],
              v3[0] + v3[1] + v3[2] + v3[3]);
          const size_t blk_off = size_t(ib) * 144u;
#pragma unroll
          for (int r = 0; r < RPS; r += 2) {
            Hy4GgufQ4kRow dw0, dw1;
            if (FAST || first + r < H) {
              dw0 = hy4_gguf_q4k_load(d_base + size_t(r) * row_bytes + blk_off, iq, ir);
            }
            if (FAST || first + r + 1 < H) {
              dw1 = hy4_gguf_q4k_load(d_base + size_t(r + 1) * row_bytes + blk_off, iq, ir);
            }
            if (FAST || first + r < H) {
              acc[r] += hy4_gguf_q4k_dot(dw0, v0, v1, v2, v3, sumy);
            }
            if (FAST || first + r + 1 < H) {
              acc[r + 1] += hy4_gguf_q4k_dot(dw1, v0, v1, v2, v3, sumy);
            }
          }
        }
      } else {
        const short ix = short(lane >> 4);
        const short tid = short(lane & 15u);
        const short ip = tid >> 3;
        const short il = tid & 7;
        const short l0 = 4 * il;
        const short is = 8 * ip + (l0 >> 4);
        const device float* x_lane = h_row + size_t(128 * ip + l0);
        for (int ib = int(ix); ib < nb; ib += 2) {
          const device float* y = x_lane + size_t(ib) * 256u;
          const float4 y0 = *reinterpret_cast<const device float4*>(y + 0);
          const float4 y1 = *reinterpret_cast<const device float4*>(y + 32);
          const float4 y2 = *reinterpret_cast<const device float4*>(y + 64);
          const float4 y3 = *reinterpret_cast<const device float4*>(y + 96);
          const float4 sumy = float4(
              y0[0] + y0[1] + y0[2] + y0[3],
              y1[0] + y1[1] + y1[2] + y1[3],
              y2[0] + y2[1] + y2[2] + y2[3],
              y3[0] + y3[1] + y3[2] + y3[3]);
          const size_t blk_off = size_t(ib) * 210u;
#pragma unroll
          for (int r = 0; r < RPS; r += 2) {
            Hy4GgufQ6kRow dw0, dw1;
            if (FAST || first + r < H) {
              dw0 = hy4_gguf_q6k_load(d_base + size_t(r) * row_bytes + blk_off, ip, l0, is);
            }
            if (FAST || first + r + 1 < H) {
              dw1 = hy4_gguf_q6k_load(d_base + size_t(r + 1) * row_bytes + blk_off, ip, l0, is);
            }
            if (FAST || first + r < H) {
              acc[r] += hy4_gguf_q6k_dot(dw0, y0, y1, y2, y3, sumy);
            }
            if (FAST || first + r + 1 < H) {
              acc[r + 1] += hy4_gguf_q6k_dot(dw1, y0, y1, y2, y3, sumy);
            }
          }
        }
      }
    }

    const float weight = weights[assign];
#pragma unroll
    for (int r = 0; r < RPS; ++r) {
      if (FAST || first + r < H) {
        const float val = simd_sum(acc[r]);
        if (lane == 0) {
          products[slot_idx * RPS + r] = val * weight;
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Match MLX col_reduce_small's 8-lane modulo fold.
  if (simd == 0 && lane < uint(RPS)) {
    if (FAST || first + int(lane) < H) {
      constexpr int LANES = (TOP < 8) ? TOP : 8;
      float totals[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int slot_idx = 0; slot_idx < TOP; ++slot_idx) {
        totals[slot_idx % 8] += products[slot_idx * RPS + int(lane)];
      }
      float total = totals[0];
      for (int i = 1; i < LANES; ++i) {
        total += totals[i];
      }
      out[size_t(row) * size_t(H) + size_t(first) + size_t(lane)] = total;
    }
  }
}
)METAL"