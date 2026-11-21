R"METAL(
inline float hy4_aff4_dot8_prescaled(
    uint32_t w,
    float4 x0s,
    float4 x1s,
    float xsum,
    float scale,
    float bias) {
  const uint16_t lo = uint16_t(w & 0xFFFFu);
  const uint16_t hi = uint16_t(w >> 16);
  const float acc =
      x0s[0] * float(lo & 0x000Fu) +
      x0s[1] * float(lo & 0x00F0u) +
      x0s[2] * float(lo & 0x0F00u) +
      x0s[3] * float(lo & 0xF000u) +
      x1s[0] * float(hi & 0x000Fu) +
      x1s[1] * float(hi & 0x00F0u) +
      x1s[2] * float(hi & 0x0F00u) +
      x1s[3] * float(hi & 0xF000u);
  return metal::fma(scale, acc, bias * xsum);
}

inline float hy4_aff4_dot16_prescaled(
    uint2 w2,
    float4 x0s,
    float4 x1s,
    float4 x2s,
    float4 x3s,
    float xsum,
    float scale,
    float bias) {
  const uint16_t w0l = uint16_t(w2.x & 0xFFFFu);
  const uint16_t w0h = uint16_t(w2.x >> 16);
  const uint16_t w1l = uint16_t(w2.y & 0xFFFFu);
  const uint16_t w1h = uint16_t(w2.y >> 16);
  const float acc =
      (x0s[0] * float(w0l & 0x000Fu) +
       x0s[1] * float(w0l & 0x00F0u) +
       x0s[2] * float(w0l & 0x0F00u) +
       x0s[3] * float(w0l & 0xF000u)) +
      (x1s[0] * float(w0h & 0x000Fu) +
       x1s[1] * float(w0h & 0x00F0u) +
       x1s[2] * float(w0h & 0x0F00u) +
       x1s[3] * float(w0h & 0xF000u)) +
      (x2s[0] * float(w1l & 0x000Fu) +
       x2s[1] * float(w1l & 0x00F0u) +
       x2s[2] * float(w1l & 0x0F00u) +
       x2s[3] * float(w1l & 0xF000u)) +
      (x3s[0] * float(w1h & 0x000Fu) +
       x3s[1] * float(w1h & 0x00F0u) +
       x3s[2] * float(w1h & 0x0F00u) +
       x3s[3] * float(w1h & 0xF000u));
  return metal::fma(scale, acc, bias * xsum);
}

inline float4 hy4_aff4_prescale4(float4 v) {
  return float4(v[0], v[1] * (1.0f / 16.0f), v[2] * (1.0f / 256.0f), v[3] * (1.0f / 4096.0f));
}

template <
    bool FAST_K,
    bool FAST_M,
    bool DEDUP,
    int MAX_MATCH,
    int NSG,
    int RPS,
    typename RowIdsPtr,
    typename SlotsPtr>
inline void hy4_affine4_gate_up_swiglu_impl(
    const device float* x,
    RowIdsPtr row_ids,
    SlotsPtr slots,
    const device uint* dedup,
    const device uint32_t* wg,
    const device bfloat* sg,
    const device bfloat* bg,
    const device uint32_t* wu,
    const device bfloat* su,
    const device bfloat* bu,
    float lim,
    device float* out,
    int ROWS,
    int N_ASSIGN,
    int K,
    size_t x_row_stride,
    int M,
    int E,
    uint assign,
    uint tile,
    uint simd,
    uint lane) {
  constexpr int VP = FAST_K ? 16 : 8;
  constexpr int BLOCK = VP * 32;
  const int first = int(tile) * (NSG * RPS) + int(simd) * RPS;
  if (assign >= uint(N_ASSIGN) || first >= M) return;

  const size_t stride_w = size_t(K / 8);
  const size_t stride_g = size_t(K / 64);

  if constexpr (DEDUP) {
    const device uint4* dedup4 = reinterpret_cast<const device uint4*>(dedup);
    const uint n_uniq = dedup4[0].x;
    if (assign >= n_uniq) return;

    const uint4 ent = dedup4[1u + assign];
    const uint exp_id = ent.x;
    const uint cnt = ent.y;
    const uint packed_a = ent.z;
    const uint packed_r = ent.w;

    if (exp_id >= uint(E)) {
      if (lane == 0u) {
        for (uint m = 0u; m < cnt; ++m) {
          const uint dst_a = (packed_a >> (m * 8u)) & 0xFFu;
#pragma unroll
          for (int r = 0; r < RPS; ++r) {
            if (FAST_M || first + r < M) {
              out[size_t(dst_a) * size_t(M) + size_t(first + r)] = 0.0f;
            }
          }
        }
      }
      return;
    }

    const size_t base_row = size_t(exp_id) * size_t(M) + size_t(first);
    const device uint32_t* wg_base = wg + base_row * stride_w;
    const device bfloat* sg_base = sg + base_row * stride_g;
    const device bfloat* bg_base = bg + base_row * stride_g;
    const device uint32_t* wu_base = wu + base_row * stride_w;
    const device bfloat* su_base = su + base_row * stride_g;
    const device bfloat* bu_base = bu + base_row * stride_g;

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
        const device float* x_row = x + size_t(rid) * x_row_stride;
        if constexpr (FAST_K) {
          for (int k = 0; k < K; k += BLOCK) {
            const int col = k + int(lane) * 16;
            const device float4* xp = reinterpret_cast<const device float4*>(x_row + col);
            const float4 x0 = xp[0];
            const float4 x1 = xp[1];
            const float4 x2 = xp[2];
            const float4 x3 = xp[3];
            const float xsum =
                (x0[0] + x0[1] + x0[2] + x0[3]) +
                (x1[0] + x1[1] + x1[2] + x1[3]) +
                (x2[0] + x2[1] + x2[2] + x2[3]) +
                (x3[0] + x3[1] + x3[2] + x3[3]);
            const float4 x0s = hy4_aff4_prescale4(x0);
            const float4 x1s = hy4_aff4_prescale4(x1);
            const float4 x2s = hy4_aff4_prescale4(x2);
            const float4 x3s = hy4_aff4_prescale4(x3);
            const size_t w_off = size_t(col >> 3);
            const size_t g_off = size_t(col >> 6);
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST_M || first + r < M) {
                const uint2 gw = *reinterpret_cast<const device uint2*>(
                    wg_base + size_t(r) * stride_w + w_off);
                const float gsc = float(sg_base[size_t(r) * stride_g + g_off]);
                const float gbi = float(bg_base[size_t(r) * stride_g + g_off]);
                g[r] += hy4_aff4_dot16_prescaled(gw, x0s, x1s, x2s, x3s, xsum, gsc, gbi);
                const uint2 uw = *reinterpret_cast<const device uint2*>(
                    wu_base + size_t(r) * stride_w + w_off);
                const float usc = float(su_base[size_t(r) * stride_g + g_off]);
                const float ubi = float(bu_base[size_t(r) * stride_g + g_off]);
                u_acc[r] += hy4_aff4_dot16_prescaled(uw, x0s, x1s, x2s, x3s, xsum, usc, ubi);
              }
            }
          }
        } else {
          for (int k = 0; k < K; k += BLOCK) {
            const int col = k + int(lane) * 8;
            if (col < K) {
              const device float4* xp = reinterpret_cast<const device float4*>(x_row + col);
              const float4 x0 = xp[0];
              const float4 x1 = xp[1];
              const float xsum =
                  (x0[0] + x0[1] + x0[2] + x0[3]) +
                  (x1[0] + x1[1] + x1[2] + x1[3]);
              const float4 x0s = hy4_aff4_prescale4(x0);
              const float4 x1s = hy4_aff4_prescale4(x1);
              const size_t w_off = size_t(col >> 3);
              const size_t g_off = size_t(col >> 6);
#pragma unroll
              for (int r = 0; r < RPS; ++r) {
                if (FAST_M || first + r < M) {
                  const uint32_t gw = wg_base[size_t(r) * stride_w + w_off];
                  const float gsc = float(sg_base[size_t(r) * stride_g + g_off]);
                  const float gbi = float(bg_base[size_t(r) * stride_g + g_off]);
                  g[r] += hy4_aff4_dot8_prescaled(gw, x0s, x1s, xsum, gsc, gbi);
                  const uint32_t uw = wu_base[size_t(r) * stride_w + w_off];
                  const float usc = float(su_base[size_t(r) * stride_g + g_off]);
                  const float ubi = float(bu_base[size_t(r) * stride_g + g_off]);
                  u_acc[r] += hy4_aff4_dot8_prescaled(uw, x0s, x1s, xsum, usc, ubi);
                }
              }
            }
          }
        }
      }

#pragma unroll
      for (int r = 0; r < RPS; ++r) {
        if (FAST_M || first + r < M) {
          float gv = simd_sum(g[r]);
          float uv = simd_sum(u_acc[r]);
          if (lane == 0u) {
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

    // Load one weight row at a time so multi-match accumulation stays spill-free.
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

    if constexpr (FAST_K) {
      for (int k = 0; k < K; k += BLOCK) {
        const int col = k + int(lane) * 16;
        const size_t w_off = size_t(col >> 3);
        const size_t g_off = size_t(col >> 6);
#pragma unroll
        for (int r = 0; r < RPS; ++r) {
          if (FAST_M || first + r < M) {
            const uint2 gw = *reinterpret_cast<const device uint2*>(
                wg_base + size_t(r) * stride_w + w_off);
            const float gsc = float(sg_base[size_t(r) * stride_g + g_off]);
            const float gbi = float(bg_base[size_t(r) * stride_g + g_off]);
            const uint2 uw = *reinterpret_cast<const device uint2*>(
                wu_base + size_t(r) * stride_w + w_off);
            const float usc = float(su_base[size_t(r) * stride_g + g_off]);
            const float ubi = float(bu_base[size_t(r) * stride_g + g_off]);
#pragma unroll
            for (int m = 0; m < MAX_MATCH; ++m) {
              if (uint(m) < cnt) {
                const uint rid = (packed_r >> (uint(m) * 8u)) & 0xFFu;
                if (rid < uint(ROWS)) {
                  const device float4* xp = reinterpret_cast<const device float4*>(
                      x + size_t(rid) * x_row_stride + col);
                  const float4 x0 = xp[0];
                  const float4 x1 = xp[1];
                  const float4 x2 = xp[2];
                  const float4 x3 = xp[3];
                  const float xsum =
                      (x0[0] + x0[1] + x0[2] + x0[3]) +
                      (x1[0] + x1[1] + x1[2] + x1[3]) +
                      (x2[0] + x2[1] + x2[2] + x2[3]) +
                      (x3[0] + x3[1] + x3[2] + x3[3]);
                  const float4 x0s = hy4_aff4_prescale4(x0);
                  const float4 x1s = hy4_aff4_prescale4(x1);
                  const float4 x2s = hy4_aff4_prescale4(x2);
                  const float4 x3s = hy4_aff4_prescale4(x3);
                  g[m][r] += hy4_aff4_dot16_prescaled(gw, x0s, x1s, x2s, x3s, xsum, gsc, gbi);
                  u_acc[m][r] += hy4_aff4_dot16_prescaled(uw, x0s, x1s, x2s, x3s, xsum, usc, ubi);
                }
              }
            }
          }
        }
      }
    } else {
      for (int k = 0; k < K; k += BLOCK) {
        const int col = k + int(lane) * 8;
        if (col < K) {
          const size_t w_off = size_t(col >> 3);
          const size_t g_off = size_t(col >> 6);
#pragma unroll
          for (int r = 0; r < RPS; ++r) {
            if (FAST_M || first + r < M) {
              const uint32_t gw = wg_base[size_t(r) * stride_w + w_off];
              const float gsc = float(sg_base[size_t(r) * stride_g + g_off]);
              const float gbi = float(bg_base[size_t(r) * stride_g + g_off]);
              const uint32_t uw = wu_base[size_t(r) * stride_w + w_off];
              const float usc = float(su_base[size_t(r) * stride_g + g_off]);
              const float ubi = float(bu_base[size_t(r) * stride_g + g_off]);
#pragma unroll
              for (int m = 0; m < MAX_MATCH; ++m) {
                if (uint(m) < cnt) {
                  const uint rid = (packed_r >> (uint(m) * 8u)) & 0xFFu;
                  if (rid < uint(ROWS)) {
                    const device float4* xp = reinterpret_cast<const device float4*>(
                        x + size_t(rid) * x_row_stride + col);
                    const float4 x0 = xp[0];
                    const float4 x1 = xp[1];
                    const float xsum =
                        (x0[0] + x0[1] + x0[2] + x0[3]) +
                        (x1[0] + x1[1] + x1[2] + x1[3]);
                    const float4 x0s = hy4_aff4_prescale4(x0);
                    const float4 x1s = hy4_aff4_prescale4(x1);
                    g[m][r] += hy4_aff4_dot8_prescaled(gw, x0s, x1s, xsum, gsc, gbi);
                    u_acc[m][r] += hy4_aff4_dot8_prescaled(uw, x0s, x1s, xsum, usc, ubi);
                  }
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
          if (FAST_M || first + r < M) {
            float gv = simd_sum(g[m][r]);
            float uv = simd_sum(u_acc[m][r]);
            if (lane == 0u) {
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

  const uint exp_id = slots[assign];
  const uint row_id = row_ids[assign];
  float g[RPS];
  float u_acc[RPS];
#pragma unroll
  for (int r = 0; r < RPS; ++r) {
    g[r] = 0.0f;
    u_acc[r] = 0.0f;
  }

  if (exp_id < uint(E) && row_id < uint(ROWS)) {
    const size_t base_row = size_t(exp_id) * size_t(M) + size_t(first);
    const device uint32_t* wg_base = wg + base_row * stride_w;
    const device bfloat* sg_base = sg + base_row * stride_g;
    const device bfloat* bg_base = bg + base_row * stride_g;
    const device uint32_t* wu_base = wu + base_row * stride_w;
    const device bfloat* su_base = su + base_row * stride_g;
    const device bfloat* bu_base = bu + base_row * stride_g;
    const device float* x_row = x + size_t(row_id) * x_row_stride;

    if constexpr (FAST_K) {
      for (int k = 0; k < K; k += BLOCK) {
        const int col = k + int(lane) * 16;
        const device float4* xp = reinterpret_cast<const device float4*>(x_row + col);
        const float4 x0 = xp[0];
        const float4 x1 = xp[1];
        const float4 x2 = xp[2];
        const float4 x3 = xp[3];
        const float xsum =
            (x0[0] + x0[1] + x0[2] + x0[3]) +
            (x1[0] + x1[1] + x1[2] + x1[3]) +
            (x2[0] + x2[1] + x2[2] + x2[3]) +
            (x3[0] + x3[1] + x3[2] + x3[3]);
        const float4 x0s = hy4_aff4_prescale4(x0);
        const float4 x1s = hy4_aff4_prescale4(x1);
        const float4 x2s = hy4_aff4_prescale4(x2);
        const float4 x3s = hy4_aff4_prescale4(x3);
        const size_t w_off = size_t(col >> 3);
        const size_t g_off = size_t(col >> 6);
#pragma unroll
        for (int r = 0; r < RPS; ++r) {
          if (FAST_M || first + r < M) {
            const uint2 gw = *reinterpret_cast<const device uint2*>(
                wg_base + size_t(r) * stride_w + w_off);
            const float gsc = float(sg_base[size_t(r) * stride_g + g_off]);
            const float gbi = float(bg_base[size_t(r) * stride_g + g_off]);
            g[r] += hy4_aff4_dot16_prescaled(gw, x0s, x1s, x2s, x3s, xsum, gsc, gbi);
            const uint2 uw = *reinterpret_cast<const device uint2*>(
                wu_base + size_t(r) * stride_w + w_off);
            const float usc = float(su_base[size_t(r) * stride_g + g_off]);
            const float ubi = float(bu_base[size_t(r) * stride_g + g_off]);
            u_acc[r] += hy4_aff4_dot16_prescaled(uw, x0s, x1s, x2s, x3s, xsum, usc, ubi);
          }
        }
      }
    } else {
      for (int k = 0; k < K; k += BLOCK) {
        const int col = k + int(lane) * 8;
        if (col < K) {
          const device float4* xp = reinterpret_cast<const device float4*>(x_row + col);
          const float4 x0 = xp[0];
          const float4 x1 = xp[1];
          const float xsum =
              (x0[0] + x0[1] + x0[2] + x0[3]) +
              (x1[0] + x1[1] + x1[2] + x1[3]);
          const float4 x0s = hy4_aff4_prescale4(x0);
          const float4 x1s = hy4_aff4_prescale4(x1);
          const size_t w_off = size_t(col >> 3);
          const size_t g_off = size_t(col >> 6);
#pragma unroll
          for (int r = 0; r < RPS; ++r) {
            if (FAST_M || first + r < M) {
              const uint32_t gw = wg_base[size_t(r) * stride_w + w_off];
              const float gsc = float(sg_base[size_t(r) * stride_g + g_off]);
              const float gbi = float(bg_base[size_t(r) * stride_g + g_off]);
              g[r] += hy4_aff4_dot8_prescaled(gw, x0s, x1s, xsum, gsc, gbi);
              const uint32_t uw = wu_base[size_t(r) * stride_w + w_off];
              const float usc = float(su_base[size_t(r) * stride_g + g_off]);
              const float ubi = float(bu_base[size_t(r) * stride_g + g_off]);
              u_acc[r] += hy4_aff4_dot8_prescaled(uw, x0s, x1s, xsum, usc, ubi);
            }
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RPS; ++r) {
    if (FAST_M || first + r < M) {
      float gv = simd_sum(g[r]);
      float uv = simd_sum(u_acc[r]);
      if (lane == 0u) {
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
    int TOP,
    bool FAST_M,
    bool FAST_H,
    bool DEDUP_ROWS,
    int MAX_MATCH,
    typename SlotsPtr,
    typename WeightsPtr>
inline void hy4_affine4_down_combine_impl(
    const device float* hidden,
    SlotsPtr slots,
    WeightsPtr weights,
    const device uint* dedup,
    const device uint32_t* wd,
    const device bfloat* sd,
    const device bfloat* bd,
    device float* out,
    threadgroup float* products,
    int ROWS,
    int M,
    int H,
    int E,
    uint row,
    uint tile,
    uint simd,
    uint lane) {
  constexpr int RPS = 4;
  constexpr int KSG = 8;
  constexpr int VP = FAST_M ? 16 : 8;
  constexpr int BLOCK = VP * 32;

  const int first = int(tile) * RPS;
  if (first >= H) return;

  const size_t stride_w = size_t(M / 8);
  const size_t stride_g = size_t(M / 64);

  if constexpr (DEDUP_ROWS) {
    const device uint4* dedup4 = reinterpret_cast<const device uint4*>(dedup);
    const uint n_uniq = dedup4[0].x;
    for (uint u = simd; u < n_uniq; u += uint(KSG)) {
      const uint4 ent = dedup4[1u + u];
      const uint exp_id = ent.x;
      const uint cnt = ent.y;
      const uint packed_a = ent.z;

      if (exp_id >= uint(E)) {
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

      const size_t base_row = size_t(exp_id) * size_t(H) + size_t(first);
      const device uint32_t* wd_base = wd + base_row * stride_w;
      const device bfloat* sd_base = sd + base_row * stride_g;
      const device bfloat* bd_base = bd + base_row * stride_g;

      if (cnt == 1u) {
        const uint a0 = packed_a & 0xFFu;
        const device float* h_row = hidden + size_t(a0) * size_t(M);
        float acc[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};
        if constexpr (FAST_M) {
          for (int k = 0; k < M; k += BLOCK) {
            const int col = k + int(lane) * 16;
            const device float4* hp = reinterpret_cast<const device float4*>(h_row + col);
            const float4 x0 = hp[0];
            const float4 x1 = hp[1];
            const float4 x2 = hp[2];
            const float4 x3 = hp[3];
            const float xsum =
                (x0[0] + x0[1] + x0[2] + x0[3]) +
                (x1[0] + x1[1] + x1[2] + x1[3]) +
                (x2[0] + x2[1] + x2[2] + x2[3]) +
                (x3[0] + x3[1] + x3[2] + x3[3]);
            const float4 x0s = hy4_aff4_prescale4(x0);
            const float4 x1s = hy4_aff4_prescale4(x1);
            const float4 x2s = hy4_aff4_prescale4(x2);
            const float4 x3s = hy4_aff4_prescale4(x3);
            const size_t w_off = size_t(col >> 3);
            const size_t g_off = size_t(col >> 6);
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST_H || first + r < H) {
                const uint2 dw = *reinterpret_cast<const device uint2*>(
                    wd_base + size_t(r) * stride_w + w_off);
                const float dsc = float(sd_base[size_t(r) * stride_g + g_off]);
                const float dbi = float(bd_base[size_t(r) * stride_g + g_off]);
                acc[r] += hy4_aff4_dot16_prescaled(dw, x0s, x1s, x2s, x3s, xsum, dsc, dbi);
              }
            }
          }
        } else {
          for (int k = 0; k < M; k += BLOCK) {
            const int col = k + int(lane) * 8;
            if (col < M) {
              const device float4* hp = reinterpret_cast<const device float4*>(h_row + col);
              const float4 x0 = hp[0];
              const float4 x1 = hp[1];
              const float xsum =
                  (x0[0] + x0[1] + x0[2] + x0[3]) +
                  (x1[0] + x1[1] + x1[2] + x1[3]);
              const float4 x0s = hy4_aff4_prescale4(x0);
              const float4 x1s = hy4_aff4_prescale4(x1);
              const size_t w_off = size_t(col >> 3);
              const size_t g_off = size_t(col >> 6);
#pragma unroll
              for (int r = 0; r < RPS; ++r) {
                if (FAST_H || first + r < H) {
                  const uint32_t dw = wd_base[size_t(r) * stride_w + w_off];
                  const float dsc = float(sd_base[size_t(r) * stride_g + g_off]);
                  const float dbi = float(bd_base[size_t(r) * stride_g + g_off]);
                  acc[r] += hy4_aff4_dot8_prescaled(dw, x0s, x1s, xsum, dsc, dbi);
                }
              }
            }
          }
        }
        const float weight = weights[a0];
#pragma unroll
        for (int r = 0; r < RPS; ++r) {
          if (FAST_H || first + r < H) {
            const float val = simd_sum(acc[r]);
            if (lane == 0u) {
              products[a0 * uint(RPS) + uint(r)] = val * weight;
            }
          }
        }
      } else {
        float acc[MAX_MATCH][RPS];
#pragma unroll
        for (int m = 0; m < MAX_MATCH; ++m) {
#pragma unroll
          for (int r = 0; r < RPS; ++r) {
            acc[m][r] = 0.0f;
          }
        }
        if constexpr (FAST_M) {
          for (int k = 0; k < M; k += BLOCK) {
            const int col = k + int(lane) * 16;
            const size_t w_off = size_t(col >> 3);
            const size_t g_off = size_t(col >> 6);
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST_H || first + r < H) {
                const uint2 dw = *reinterpret_cast<const device uint2*>(
                    wd_base + size_t(r) * stride_w + w_off);
                const float dsc = float(sd_base[size_t(r) * stride_g + g_off]);
                const float dbi = float(bd_base[size_t(r) * stride_g + g_off]);
#pragma unroll
                for (int m = 0; m < MAX_MATCH; ++m) {
                  if (uint(m) < cnt) {
                    const uint a_m = (packed_a >> (uint(m) * 8u)) & 0xFFu;
                    const device float4* hp = reinterpret_cast<const device float4*>(
                        hidden + size_t(a_m) * size_t(M) + col);
                    const float4 x0 = hp[0];
                    const float4 x1 = hp[1];
                    const float4 x2 = hp[2];
                    const float4 x3 = hp[3];
                    const float xsum =
                        (x0[0] + x0[1] + x0[2] + x0[3]) +
                        (x1[0] + x1[1] + x1[2] + x1[3]) +
                        (x2[0] + x2[1] + x2[2] + x2[3]) +
                        (x3[0] + x3[1] + x3[2] + x3[3]);
                    const float4 x0s = hy4_aff4_prescale4(x0);
                    const float4 x1s = hy4_aff4_prescale4(x1);
                    const float4 x2s = hy4_aff4_prescale4(x2);
                    const float4 x3s = hy4_aff4_prescale4(x3);
                    acc[m][r] += hy4_aff4_dot16_prescaled(dw, x0s, x1s, x2s, x3s, xsum, dsc, dbi);
                  }
                }
              }
            }
          }
        } else {
          for (int k = 0; k < M; k += BLOCK) {
            const int col = k + int(lane) * 8;
            if (col < M) {
              const size_t w_off = size_t(col >> 3);
              const size_t g_off = size_t(col >> 6);
#pragma unroll
              for (int r = 0; r < RPS; ++r) {
                if (FAST_H || first + r < H) {
                  const uint32_t dw = wd_base[size_t(r) * stride_w + w_off];
                  const float dsc = float(sd_base[size_t(r) * stride_g + g_off]);
                  const float dbi = float(bd_base[size_t(r) * stride_g + g_off]);
#pragma unroll
                  for (int m = 0; m < MAX_MATCH; ++m) {
                    if (uint(m) < cnt) {
                      const uint a_m = (packed_a >> (uint(m) * 8u)) & 0xFFu;
                      const device float4* hp = reinterpret_cast<const device float4*>(
                          hidden + size_t(a_m) * size_t(M) + col);
                      const float4 x0 = hp[0];
                      const float4 x1 = hp[1];
                      const float xsum =
                          (x0[0] + x0[1] + x0[2] + x0[3]) +
                          (x1[0] + x1[1] + x1[2] + x1[3]);
                      const float4 x0s = hy4_aff4_prescale4(x0);
                      const float4 x1s = hy4_aff4_prescale4(x1);
                      acc[m][r] += hy4_aff4_dot8_prescaled(dw, x0s, x1s, xsum, dsc, dbi);
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
            const float weight = weights[dst_a];
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST_H || first + r < H) {
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

    // Match MLX col_reduce_small's 8-lane modulo fold for row-invariance.
    for (uint r_idx = simd; r_idx < uint(ROWS); r_idx += uint(KSG)) {
      if (lane < uint(RPS)) {
        if (FAST_H || first + int(lane) < H) {
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
    const uint exp_id = slots[assign];
    float acc[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (exp_id < uint(E)) {
      const size_t base_row = size_t(exp_id) * size_t(H) + size_t(first);
      const device uint32_t* wd_base = wd + base_row * stride_w;
      const device bfloat* sd_base = sd + base_row * stride_g;
      const device bfloat* bd_base = bd + base_row * stride_g;
      const device float* h_row = hidden + assign * size_t(M);

      if constexpr (FAST_M) {
        for (int k = 0; k < M; k += BLOCK) {
          const int col = k + int(lane) * 16;
          const device float4* hp = reinterpret_cast<const device float4*>(h_row + col);
          const float4 x0 = hp[0];
          const float4 x1 = hp[1];
          const float4 x2 = hp[2];
          const float4 x3 = hp[3];
          const float xsum =
              (x0[0] + x0[1] + x0[2] + x0[3]) +
              (x1[0] + x1[1] + x1[2] + x1[3]) +
              (x2[0] + x2[1] + x2[2] + x2[3]) +
              (x3[0] + x3[1] + x3[2] + x3[3]);
          const float4 x0s = hy4_aff4_prescale4(x0);
          const float4 x1s = hy4_aff4_prescale4(x1);
          const float4 x2s = hy4_aff4_prescale4(x2);
          const float4 x3s = hy4_aff4_prescale4(x3);
          const size_t w_off = size_t(col >> 3);
          const size_t g_off = size_t(col >> 6);
#pragma unroll
          for (int r = 0; r < RPS; ++r) {
            if (FAST_H || first + r < H) {
              const uint2 dw = *reinterpret_cast<const device uint2*>(
                  wd_base + size_t(r) * stride_w + w_off);
              const float dsc = float(sd_base[size_t(r) * stride_g + g_off]);
              const float dbi = float(bd_base[size_t(r) * stride_g + g_off]);
              acc[r] += hy4_aff4_dot16_prescaled(dw, x0s, x1s, x2s, x3s, xsum, dsc, dbi);
            }
          }
        }
      } else {
        for (int k = 0; k < M; k += BLOCK) {
          const int col = k + int(lane) * 8;
          if (col < M) {
            const device float4* hp = reinterpret_cast<const device float4*>(h_row + col);
            const float4 x0 = hp[0];
            const float4 x1 = hp[1];
            const float xsum =
                (x0[0] + x0[1] + x0[2] + x0[3]) +
                (x1[0] + x1[1] + x1[2] + x1[3]);
            const float4 x0s = hy4_aff4_prescale4(x0);
            const float4 x1s = hy4_aff4_prescale4(x1);
            const size_t w_off = size_t(col >> 3);
            const size_t g_off = size_t(col >> 6);
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
              if (FAST_H || first + r < H) {
                const uint32_t dw = wd_base[size_t(r) * stride_w + w_off];
                const float dsc = float(sd_base[size_t(r) * stride_g + g_off]);
                const float dbi = float(bd_base[size_t(r) * stride_g + g_off]);
                acc[r] += hy4_aff4_dot8_prescaled(dw, x0s, x1s, xsum, dsc, dbi);
              }
            }
          }
        }
      }
    }

    const float weight = weights[assign];
#pragma unroll
    for (int r = 0; r < RPS; ++r) {
      if (FAST_H || first + r < H) {
        const float val = simd_sum(acc[r]);
        if (lane == 0u) {
          products[slot_idx * RPS + r] = val * weight;
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd == 0u && lane < uint(RPS)) {
    if (FAST_H || first + int(lane) < H) {
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
