R"METAL(
template <
    int HC,
    bool HEAD_ONLY,
    bool VEC4,
    typename XPtr,
    typename WPtr,
    typename ScalePtr,
    typename BasePtr,
    typename NormWPtr,
    typename ParamsPtr>
inline void hy4_hc_mix_reduce_norm_impl(
    XPtr x,
    WPtr weight,
    ScalePtr scale,
    BasePtr base,
    NormWPtr norm_w,
    ParamsPtr params,
    device float* h_norm,
    device float* post,
    threadgroup float* tg_red,
    threadgroup float* tg_pre,
    threadgroup float* tg_h2,
    int M,
    int D,
    uint m,
    uint tid,
    uint lane,
    uint sg) {
  if (int(m) >= M) return;

  constexpr int N_GATES = HEAD_ONLY ? HC : (2 * HC);
  constexpr int STRIDE = 1 + N_GATES;
  const size_t hd = size_t(HC) * size_t(D);
  const size_t x_row_off = size_t(m) * hd;
  const size_t h_row_off = size_t(m) * size_t(D);

  float sq_acc = 0.0f;
  float dot_acc[N_GATES];
  for (int g = 0; g < N_GATES; ++g) {
    dot_acc[g] = 0.0f;
  }

  if constexpr (VEC4) {
    const size_t hd4 = hd >> 2;
    const device float4* x4 =
        reinterpret_cast<const device float4*>(x + x_row_off);
    for (size_t idx4 = tid; idx4 < hd4; idx4 += 256) {
      const float4 xv = x4[idx4];
      sq_acc += dot(xv, xv);
      for (int g = 0; g < N_GATES; ++g) {
        const device float4* w4 =
            reinterpret_cast<const device float4*>(weight + size_t(g) * hd);
        dot_acc[g] += dot(xv, w4[idx4]);
      }
    }
  } else {
    const auto xr = x + x_row_off;
    for (size_t idx = tid; idx < hd; idx += 256) {
      const float xv = xr[idx];
      sq_acc += xv * xv;
      for (int g = 0; g < N_GATES; ++g) {
        dot_acc[g] += xv * weight[size_t(g) * hd + idx];
      }
    }
  }

  sq_acc = simd_sum(sq_acc);
  if (lane == 0) {
    tg_red[sg * STRIDE] = sq_acc;
  }
  for (int g = 0; g < N_GATES; ++g) {
    const float d_sum = simd_sum(dot_acc[g]);
    if (lane == 0) {
      tg_red[sg * STRIDE + 1 + g] = d_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    float sq_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      sq_total += tg_red[s * STRIDE];
    }
    const float hc_norm_eps = params[0];
    const float hc_eps = params[1];
    const float magnitude = params[2];
    const float rs = 1.0f / metal::sqrt(sq_total / float(hd) + hc_norm_eps);
    const float s_pre = scale[0];
    for (int c = 0; c < HC; ++c) {
      float dot_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        dot_total += tg_red[s * STRIDE + 1 + c];
      }
      const float z_pre = (dot_total * rs) * s_pre + base[c];
      tg_pre[c] = 1.0f / (1.0f + metal::exp(-z_pre)) + hc_eps;
    }
    if constexpr (!HEAD_ONLY) {
      const float s_post = scale[1];
      for (int c = 0; c < HC; ++c) {
        float dot_total = 0.0f;
        for (int s = 0; s < 8; ++s) {
          dot_total += tg_red[s * STRIDE + 1 + HC + c];
        }
        const float z_post = (dot_total * rs) * s_post + base[HC + c];
        post[size_t(m) * size_t(HC) + size_t(c)] =
            (1.0f / (1.0f + metal::exp(-z_post))) * magnitude + hc_eps;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float pre_reg[HC];
  for (int c = 0; c < HC; ++c) {
    pre_reg[c] = tg_pre[c];
  }

  const float out_norm_eps = params[3];

  if constexpr (VEC4) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* nw4 = reinterpret_cast<const device float4*>(norm_w);
    device float4* out4 = reinterpret_cast<device float4*>(h_norm + h_row_off);
    if (d4 <= 1024) {
      float4 h_reg[4];
      float h2 = 0.0f;
      for (int k = 0; k < 4; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256;
        if (j4 < d4) {
          float4 acc = float4(0.0f);
          for (int c = 0; c < HC; ++c) {
            const device float4* xc4 = reinterpret_cast<const device float4*>(
                x + x_row_off + size_t(c) * size_t(D));
            acc += pre_reg[c] * xc4[j4];
          }
          h_reg[k] = acc;
          h2 += dot(acc, acc);
        }
      }
      h2 = simd_sum(h2);
      if (lane == 0) {
        tg_h2[sg] = h2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float h2_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        h2_total += tg_h2[s];
      }
      const float rs_out =
          metal::precise::rsqrt(h2_total / float(D) + out_norm_eps);
      for (int k = 0; k < 4; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256;
        if (j4 < d4) {
          out4[j4] = h_reg[k] * rs_out * nw4[j4];
        }
      }
    } else {
      float h2 = 0.0f;
      for (size_t j4 = tid; j4 < d4; j4 += 256) {
        float4 acc = float4(0.0f);
        for (int c = 0; c < HC; ++c) {
          const device float4* xc4 = reinterpret_cast<const device float4*>(
              x + x_row_off + size_t(c) * size_t(D));
          acc += pre_reg[c] * xc4[j4];
        }
        h2 += dot(acc, acc);
      }
      h2 = simd_sum(h2);
      if (lane == 0) {
        tg_h2[sg] = h2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float h2_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        h2_total += tg_h2[s];
      }
      const float rs_out =
          metal::precise::rsqrt(h2_total / float(D) + out_norm_eps);
      for (size_t j4 = tid; j4 < d4; j4 += 256) {
        float4 acc = float4(0.0f);
        for (int c = 0; c < HC; ++c) {
          const device float4* xc4 = reinterpret_cast<const device float4*>(
              x + x_row_off + size_t(c) * size_t(D));
          acc += pre_reg[c] * xc4[j4];
        }
        out4[j4] = acc * rs_out * nw4[j4];
      }
    }
  } else {
    float h2 = 0.0f;
    for (size_t j = tid; j < size_t(D); j += 256) {
      float acc = 0.0f;
      for (int c = 0; c < HC; ++c) {
        acc += pre_reg[c] * x[x_row_off + size_t(c) * size_t(D) + j];
      }
      h2 += acc * acc;
    }
    h2 = simd_sum(h2);
    if (lane == 0) {
      tg_h2[sg] = h2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float h2_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      h2_total += tg_h2[s];
    }
    const float rs_out =
        metal::precise::rsqrt(h2_total / float(D) + out_norm_eps);
    device float* out_r = h_norm + h_row_off;
    for (size_t j = tid; j < size_t(D); j += 256) {
      float acc = 0.0f;
      for (int c = 0; c < HC; ++c) {
        acc += pre_reg[c] * x[x_row_off + size_t(c) * size_t(D) + j];
      }
      out_r[j] = acc * rs_out * norm_w[j];
    }
  }
}

template <
    int HC,
    bool VEC4,
    typename YPtr,
    typename ResPtr,
    typename PostPtr,
    typename PostNormPtr,
    typename EpsPtr>
inline void hy4_post_norm_hc_post_impl(
    YPtr y,
    ResPtr res,
    PostPtr post,
    PostNormPtr post_norm_w,
    EpsPtr post_eps,
    device float* out,
    threadgroup float* tg_y2,
    int M,
    int D,
    uint m,
    uint tid,
    uint lane,
    uint sg) {
  if (int(m) >= M) return;

  const size_t y_row_off = size_t(m) * size_t(D);
  const size_t x_row_off = size_t(m) * size_t(HC) * size_t(D);
  const size_t p_row_off = size_t(m) * size_t(HC);

  float post_reg[HC];
  for (int c = 0; c < HC; ++c) {
    post_reg[c] = post[p_row_off + size_t(c)];
  }

  if constexpr (VEC4) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* y4 =
        reinterpret_cast<const device float4*>(y + y_row_off);
    const device float4* pnw4 =
        reinterpret_cast<const device float4*>(post_norm_w);
    if (d4 <= 1024) {
      float4 y_reg[4];
      float y2 = 0.0f;
      for (int k = 0; k < 4; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256;
        if (j4 < d4) {
          y_reg[k] = y4[j4];
          y2 += dot(y_reg[k], y_reg[k]);
        }
      }
      y2 = simd_sum(y2);
      if (lane == 0) {
        tg_y2[sg] = y2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float y2_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        y2_total += tg_y2[s];
      }
      const float rs = metal::precise::rsqrt(y2_total / float(D) + post_eps[0]);
      for (int k = 0; k < 4; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256;
        if (j4 < d4) {
          const float4 yn = y_reg[k] * rs * pnw4[j4];
          for (int c = 0; c < HC; ++c) {
            const device float4* r4 = reinterpret_cast<const device float4*>(
                res + x_row_off + size_t(c) * size_t(D));
            device float4* o4 = reinterpret_cast<device float4*>(
                out + x_row_off + size_t(c) * size_t(D));
            o4[j4] = r4[j4] + post_reg[c] * yn;
          }
        }
      }
    } else {
      float y2 = 0.0f;
      for (size_t j4 = tid; j4 < d4; j4 += 256) {
        const float4 yv = y4[j4];
        y2 += dot(yv, yv);
      }
      y2 = simd_sum(y2);
      if (lane == 0) {
        tg_y2[sg] = y2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float y2_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        y2_total += tg_y2[s];
      }
      const float rs = metal::precise::rsqrt(y2_total / float(D) + post_eps[0]);
      for (size_t j4 = tid; j4 < d4; j4 += 256) {
        const float4 yn = y4[j4] * rs * pnw4[j4];
        for (int c = 0; c < HC; ++c) {
          const device float4* r4 = reinterpret_cast<const device float4*>(
              res + x_row_off + size_t(c) * size_t(D));
          device float4* o4 = reinterpret_cast<device float4*>(
              out + x_row_off + size_t(c) * size_t(D));
          o4[j4] = r4[j4] + post_reg[c] * yn;
        }
      }
    }
  } else {
    float y2 = 0.0f;
    for (size_t j = tid; j < size_t(D); j += 256) {
      const float yv = y[y_row_off + j];
      y2 += yv * yv;
    }
    y2 = simd_sum(y2);
    if (lane == 0) {
      tg_y2[sg] = y2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y2_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      y2_total += tg_y2[s];
    }
    const float rs = metal::precise::rsqrt(y2_total / float(D) + post_eps[0]);
    for (size_t j = tid; j < size_t(D); j += 256) {
      const float yn = y[y_row_off + j] * rs * post_norm_w[j];
      for (int c = 0; c < HC; ++c) {
        const size_t idx = x_row_off + size_t(c) * size_t(D) + j;
        out[idx] = res[idx] + post_reg[c] * yn;
      }
    }
  }
}

template <
    int HC,
    bool VEC4,
    bool SKIP_NEXT_NORM,
    bool HEAD_ONLY,
    bool USE_REG,
    bool ROW2,
    typename YPtr,
    typename ResPtr,
    typename PostPrevPtr,
    typename PostNormPtr,
    typename WPtr,
    typename ScalePtr,
    typename BasePtr,
    typename NextNormPtr,
    typename ParamsPtr>
inline void hy4_hc_bridge_impl(
    YPtr y,
    ResPtr res,
    PostPrevPtr post_prev,
    PostNormPtr post_norm_w,
    WPtr next_weight,
    ScalePtr next_scale,
    BasePtr next_base,
    NextNormPtr next_norm_w,
    ParamsPtr params,
    device float* x_out,
    device float* h_out,
    device float* post_next,
    threadgroup float* tg_red,
    threadgroup float* tg_pre,
    threadgroup float* tg_y2,
    threadgroup float* tg_h2,
    int M,
    int D,
    uint m_blk,
    uint tid,
    uint lane,
    uint sg) {
  constexpr int N_GATES = HEAD_ONLY ? HC : (2 * HC);
  constexpr int STRIDE = 1 + N_GATES;

  const float post_eps = params[0];
  const float hc_norm_eps = params[1];
  const float hc_eps = params[2];
  const float magnitude = params[3];
  const float next_eps = params[4];

  if constexpr (USE_REG && ROW2) {
    const uint m0 = m_blk * 2u;
    const uint m1 = m0 + 1u;
    if (int(m0) >= M) return;
    const bool has1 = (int(m1) < M);

    const size_t d4 = size_t(D) >> 2;
    const size_t hd = size_t(HC) * size_t(D);
    const size_t x_off0 = size_t(m0) * hd;
    const size_t y_off0 = size_t(m0) * size_t(D);
    const size_t p_off0 = size_t(m0) * size_t(HC);
    const size_t x_off1 = size_t(m1) * hd;
    const size_t y_off1 = size_t(m1) * size_t(D);
    const size_t p_off1 = size_t(m1) * size_t(HC);

    float pp0[HC], pp1[HC];
    for (int c = 0; c < HC; ++c) {
      pp0[c] = post_prev[p_off0 + size_t(c)];
      pp1[c] = has1 ? post_prev[p_off1 + size_t(c)] : 0.0f;
    }

    const device float4* y4_0 = reinterpret_cast<const device float4*>(y + y_off0);
    const device float4* y4_1 = reinterpret_cast<const device float4*>(y + y_off1);
    const device float4* pnw4 = reinterpret_cast<const device float4*>(post_norm_w);

    float4 ynv0[3], ynv1[3];
    float y2_0 = 0.0f;
    float y2_1 = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        ynv0[k] = y4_0[j4];
        y2_0 += dot(ynv0[k], ynv0[k]);
        if (has1) {
          ynv1[k] = y4_1[j4];
          y2_1 += dot(ynv1[k], ynv1[k]);
        }
      }
    }
    y2_0 = simd_sum(y2_0);
    y2_1 = simd_sum(y2_1);
    if (lane == 0u) {
      tg_y2[sg] = y2_0;
      tg_y2[8u + sg] = y2_1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y2_tot0 = 0.0f;
    float y2_tot1 = 0.0f;
    for (int s = 0; s < 8; ++s) {
      y2_tot0 += tg_y2[s];
      y2_tot1 += tg_y2[8 + s];
    }
    const float rs_post0 = metal::precise::rsqrt(y2_tot0 / float(D) + post_eps);
    const float rs_post1 = metal::precise::rsqrt(y2_tot1 / float(D) + post_eps);
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        const float4 pnw = pnw4[j4];
        ynv0[k] = ynv0[k] * rs_post0 * pnw;
        if (has1) {
          ynv1[k] = ynv1[k] * rs_post1 * pnw;
        }
      }
    }

    const device float4* res4_0 = reinterpret_cast<const device float4*>(res + x_off0);
    const device float4* res4_1 = reinterpret_cast<const device float4*>(res + x_off1);
    float4 xc0[HC][3];
    float4 xc1[HC][3];
    float sq0 = 0.0f;
    float sq1 = 0.0f;
    for (int c = 0; c < HC; ++c) {
      const size_t c_off = size_t(c) * d4;
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          const size_t idx4 = c_off + j4;
          const float4 v0 = res4_0[idx4] + pp0[c] * ynv0[k];
          xc0[c][k] = v0;
          if constexpr (!HEAD_ONLY) {
            reinterpret_cast<device float4*>(x_out + x_off0)[idx4] = v0;
          }
          sq0 += dot(v0, v0);
          if (has1) {
            const float4 v1 = res4_1[idx4] + pp1[c] * ynv1[k];
            xc1[c][k] = v1;
            if constexpr (!HEAD_ONLY) {
              reinterpret_cast<device float4*>(x_out + x_off1)[idx4] = v1;
            }
            sq1 += dot(v1, v1);
          }
        }
      }
    }
    sq0 = simd_sum(sq0);
    sq1 = simd_sum(sq1);
    if (lane == 0u) {
      tg_red[sg * STRIDE] = sq0;
      tg_red[(8u + sg) * STRIDE] = sq1;
    }

    // Outer loop over gates streams each weight row sequentially once for both
    // rows while keeping only two scalar dot accumulators live.
    for (int g = 0; g < N_GATES; ++g) {
      const device float4* w4 =
          reinterpret_cast<const device float4*>(next_weight + size_t(g) * hd);
      float d0 = 0.0f;
      float d1 = 0.0f;
      for (int c = 0; c < HC; ++c) {
        const size_t c_off = size_t(c) * d4;
        for (int k = 0; k < 3; ++k) {
          const size_t j4 = size_t(tid) + size_t(k) * 256u;
          if (j4 < d4) {
            const float4 wv = w4[c_off + j4];
            d0 += dot(xc0[c][k], wv);
            if (has1) {
              d1 += dot(xc1[c][k], wv);
            }
          }
        }
      }
      const float s0 = simd_sum(d0);
      const float s1 = simd_sum(d1);
      if (lane == 0u) {
        tg_red[sg * STRIDE + 1 + g] = s0;
        tg_red[(8u + sg) * STRIDE + 1 + g] = s1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u || (tid == 1u && has1)) {
      const uint r = tid;
      const int red_off = int(r) * 8 * STRIDE;
      const size_t p_off = (r == 0u) ? p_off0 : p_off1;
      float sq_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        sq_total += tg_red[red_off + s * STRIDE];
      }
      const float rs_hc =
          1.0f / metal::sqrt(sq_total / float(hd) + hc_norm_eps);
      const float s_pre = next_scale[0];
      for (int c = 0; c < HC; ++c) {
        float dot_total = 0.0f;
        for (int s = 0; s < 8; ++s) {
          dot_total += tg_red[red_off + s * STRIDE + 1 + c];
        }
        const float z_pre = (dot_total * rs_hc) * s_pre + next_base[c];
        tg_pre[int(r) * HC + c] = 1.0f / (1.0f + metal::exp(-z_pre)) + hc_eps;
      }
      if constexpr (!HEAD_ONLY) {
        const float s_post = next_scale[1];
        for (int c = 0; c < HC; ++c) {
          float dot_total = 0.0f;
          for (int s = 0; s < 8; ++s) {
            dot_total += tg_red[red_off + s * STRIDE + 1 + HC + c];
          }
          const float z_post = (dot_total * rs_hc) * s_post + next_base[HC + c];
          post_next[p_off + size_t(c)] =
              (1.0f / (1.0f + metal::exp(-z_post))) * magnitude + hc_eps;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pre0[HC], pre1[HC];
    for (int c = 0; c < HC; ++c) {
      pre0[c] = tg_pre[c];
      pre1[c] = has1 ? tg_pre[HC + c] : 0.0f;
    }

    device float4* hout4_0 = reinterpret_cast<device float4*>(h_out + y_off0);
    device float4* hout4_1 = reinterpret_cast<device float4*>(h_out + y_off1);
    float4 h0[3], h1[3];
    float h2_0 = 0.0f;
    float h2_1 = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        for (int c = 0; c < HC; ++c) {
          acc0 += pre0[c] * xc0[c][k];
          if (has1) {
            acc1 += pre1[c] * xc1[c][k];
          }
        }
        h0[k] = acc0;
        h1[k] = acc1;
        if constexpr (!SKIP_NEXT_NORM) {
          h2_0 += dot(acc0, acc0);
          if (has1) {
            h2_1 += dot(acc1, acc1);
          }
        }
      }
    }
    if constexpr (SKIP_NEXT_NORM) {
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          hout4_0[j4] = h0[k];
          if (has1) {
            hout4_1[j4] = h1[k];
          }
        }
      }
    } else {
      h2_0 = simd_sum(h2_0);
      h2_1 = simd_sum(h2_1);
      if (lane == 0u) {
        tg_h2[sg] = h2_0;
        tg_h2[8u + sg] = h2_1;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float h2_tot0 = 0.0f;
      float h2_tot1 = 0.0f;
      for (int s = 0; s < 8; ++s) {
        h2_tot0 += tg_h2[s];
        h2_tot1 += tg_h2[8 + s];
      }
      const float rs_next0 =
          metal::precise::rsqrt(h2_tot0 / float(D) + next_eps);
      const float rs_next1 =
          metal::precise::rsqrt(h2_tot1 / float(D) + next_eps);
      const device float4* nnw4 =
          reinterpret_cast<const device float4*>(next_norm_w);
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          const float4 nnw = nnw4[j4];
          hout4_0[j4] = h0[k] * rs_next0 * nnw;
          if (has1) {
            hout4_1[j4] = h1[k] * rs_next1 * nnw;
          }
        }
      }
    }
    return;
  }

  const uint m = m_blk;
  if (int(m) >= M) return;

  const size_t hd = size_t(HC) * size_t(D);
  const size_t x_row_off = size_t(m) * hd;
  const size_t y_row_off = size_t(m) * size_t(D);
  const size_t p_row_off = size_t(m) * size_t(HC);

  float post_prev_reg[HC];
  for (int c = 0; c < HC; ++c) {
    post_prev_reg[c] = post_prev[p_row_off + size_t(c)];
  }

  if constexpr (USE_REG) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* y4 =
        reinterpret_cast<const device float4*>(y + y_row_off);
    const device float4* pnw4 =
        reinterpret_cast<const device float4*>(post_norm_w);
    const device float4* res4 =
        reinterpret_cast<const device float4*>(res + x_row_off);

    float4 ynv[3];
    float y2 = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        ynv[k] = y4[j4];
        y2 += dot(ynv[k], ynv[k]);
      }
    }
    y2 = simd_sum(y2);
    if (lane == 0u) {
      tg_y2[sg] = y2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y2_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      y2_total += tg_y2[s];
    }
    const float rs_post = metal::precise::rsqrt(y2_total / float(D) + post_eps);
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        ynv[k] = ynv[k] * rs_post * pnw4[j4];
      }
    }

    float4 xc_reg[HC][3];
    float sq_acc = 0.0f;
    for (int c = 0; c < HC; ++c) {
      const size_t c_off = size_t(c) * d4;
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          const size_t idx4 = c_off + j4;
          const float4 xc = res4[idx4] + post_prev_reg[c] * ynv[k];
          xc_reg[c][k] = xc;
          if constexpr (!HEAD_ONLY) {
            reinterpret_cast<device float4*>(x_out + x_row_off)[idx4] = xc;
          }
          sq_acc += dot(xc, xc);
        }
      }
    }
    sq_acc = simd_sum(sq_acc);
    if (lane == 0u) {
      tg_red[sg * STRIDE] = sq_acc;
    }
    for (int g = 0; g < N_GATES; ++g) {
      const device float4* w4 =
          reinterpret_cast<const device float4*>(next_weight + size_t(g) * hd);
      float d_acc = 0.0f;
      for (int c = 0; c < HC; ++c) {
        const size_t c_off = size_t(c) * d4;
        for (int k = 0; k < 3; ++k) {
          const size_t j4 = size_t(tid) + size_t(k) * 256u;
          if (j4 < d4) {
            d_acc += dot(xc_reg[c][k], w4[c_off + j4]);
          }
        }
      }
      const float d_sum = simd_sum(d_acc);
      if (lane == 0u) {
        tg_red[sg * STRIDE + 1 + g] = d_sum;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
      float sq_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        sq_total += tg_red[s * STRIDE];
      }
      const float rs_hc =
          1.0f / metal::sqrt(sq_total / float(hd) + hc_norm_eps);
      const float s_pre = next_scale[0];
      for (int c = 0; c < HC; ++c) {
        float dot_total = 0.0f;
        for (int s = 0; s < 8; ++s) {
          dot_total += tg_red[s * STRIDE + 1 + c];
        }
        const float z_pre = (dot_total * rs_hc) * s_pre + next_base[c];
        tg_pre[c] = 1.0f / (1.0f + metal::exp(-z_pre)) + hc_eps;
      }
      if constexpr (!HEAD_ONLY) {
        const float s_post = next_scale[1];
        for (int c = 0; c < HC; ++c) {
          float dot_total = 0.0f;
          for (int s = 0; s < 8; ++s) {
            dot_total += tg_red[s * STRIDE + 1 + HC + c];
          }
          const float z_post = (dot_total * rs_hc) * s_post + next_base[HC + c];
          post_next[p_row_off + size_t(c)] =
              (1.0f / (1.0f + metal::exp(-z_post))) * magnitude + hc_eps;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pre_reg[HC];
    for (int c = 0; c < HC; ++c) {
      pre_reg[c] = tg_pre[c];
    }

    device float4* hout4 = reinterpret_cast<device float4*>(h_out + y_row_off);
    float4 h_reg[3];
    float h2 = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        float4 acc = float4(0.0f);
        for (int c = 0; c < HC; ++c) {
          acc += pre_reg[c] * xc_reg[c][k];
        }
        h_reg[k] = acc;
        if constexpr (!SKIP_NEXT_NORM) {
          h2 += dot(acc, acc);
        }
      }
    }
    if constexpr (SKIP_NEXT_NORM) {
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          hout4[j4] = h_reg[k];
        }
      }
    } else {
      h2 = simd_sum(h2);
      if (lane == 0u) {
        tg_h2[sg] = h2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float h2_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        h2_total += tg_h2[s];
      }
      const float rs_next =
          metal::precise::rsqrt(h2_total / float(D) + next_eps);
      const device float4* nnw4 =
          reinterpret_cast<const device float4*>(next_norm_w);
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          hout4[j4] = h_reg[k] * rs_next * nnw4[j4];
        }
      }
    }
    return;
  }

  float rs_post = 0.0f;
  if constexpr (VEC4) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* y4 =
        reinterpret_cast<const device float4*>(y + y_row_off);
    float y2 = 0.0f;
    for (size_t j4 = tid; j4 < d4; j4 += 256) {
      const float4 yv = y4[j4];
      y2 += dot(yv, yv);
    }
    y2 = simd_sum(y2);
    if (lane == 0) {
      tg_y2[sg] = y2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y2_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      y2_total += tg_y2[s];
    }
    rs_post = metal::precise::rsqrt(y2_total / float(D) + post_eps);
  } else {
    float y2 = 0.0f;
    for (size_t j = tid; j < size_t(D); j += 256) {
      const float yv = y[y_row_off + j];
      y2 += yv * yv;
    }
    y2 = simd_sum(y2);
    if (lane == 0) {
      tg_y2[sg] = y2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y2_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      y2_total += tg_y2[s];
    }
    rs_post = metal::precise::rsqrt(y2_total / float(D) + post_eps);
  }

  float sq_acc = 0.0f;
  float dot_acc[N_GATES];
  for (int g = 0; g < N_GATES; ++g) {
    dot_acc[g] = 0.0f;
  }

  if constexpr (VEC4) {
    const size_t d4 = size_t(D) >> 2;
    const size_t hd4 = hd >> 2;
    const device float4* y4 =
        reinterpret_cast<const device float4*>(y + y_row_off);
    const device float4* pnw4 =
        reinterpret_cast<const device float4*>(post_norm_w);
    const device float4* res4 =
        reinterpret_cast<const device float4*>(res + x_row_off);
    for (size_t idx4 = tid; idx4 < hd4; idx4 += 256) {
      const int c = int(idx4 / d4);
      const size_t j4 = idx4 - size_t(c) * d4;
      const float4 ynv = y4[j4] * rs_post * pnw4[j4];
      const float4 xc = res4[idx4] + post_prev_reg[c] * ynv;
      if constexpr (!HEAD_ONLY) {
        reinterpret_cast<device float4*>(x_out + x_row_off)[idx4] = xc;
      }
      sq_acc += dot(xc, xc);
      for (int g = 0; g < N_GATES; ++g) {
        const device float4* w4 = reinterpret_cast<const device float4*>(
            next_weight + size_t(g) * hd);
        dot_acc[g] += dot(xc, w4[idx4]);
      }
    }
  } else {
    for (size_t idx = tid; idx < hd; idx += 256) {
      const int c = int(idx / size_t(D));
      const size_t j = idx - size_t(c) * size_t(D);
      const float ynv = y[y_row_off + j] * rs_post * post_norm_w[j];
      const float xc = res[x_row_off + idx] + post_prev_reg[c] * ynv;
      if constexpr (!HEAD_ONLY) {
        x_out[x_row_off + idx] = xc;
      }
      sq_acc += xc * xc;
      for (int g = 0; g < N_GATES; ++g) {
        dot_acc[g] += xc * next_weight[size_t(g) * hd + idx];
      }
    }
  }

  sq_acc = simd_sum(sq_acc);
  if (lane == 0) {
    tg_red[sg * STRIDE] = sq_acc;
  }
  for (int g = 0; g < N_GATES; ++g) {
    const float d_sum = simd_sum(dot_acc[g]);
    if (lane == 0) {
      tg_red[sg * STRIDE + 1 + g] = d_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    float sq_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      sq_total += tg_red[s * STRIDE];
    }
    const float rs_hc =
        1.0f / metal::sqrt(sq_total / float(hd) + hc_norm_eps);
    const float s_pre = next_scale[0];
    for (int c = 0; c < HC; ++c) {
      float dot_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        dot_total += tg_red[s * STRIDE + 1 + c];
      }
      const float z_pre = (dot_total * rs_hc) * s_pre + next_base[c];
      tg_pre[c] = 1.0f / (1.0f + metal::exp(-z_pre)) + hc_eps;
    }
    if constexpr (!HEAD_ONLY) {
      const float s_post = next_scale[1];
      for (int c = 0; c < HC; ++c) {
        float dot_total = 0.0f;
        for (int s = 0; s < 8; ++s) {
          dot_total += tg_red[s * STRIDE + 1 + HC + c];
        }
        const float z_post = (dot_total * rs_hc) * s_post + next_base[HC + c];
        post_next[p_row_off + size_t(c)] =
            (1.0f / (1.0f + metal::exp(-z_post))) * magnitude + hc_eps;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float pre_reg[HC];
  for (int c = 0; c < HC; ++c) {
    pre_reg[c] = tg_pre[c];
  }

  if constexpr (VEC4) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* y4 =
        reinterpret_cast<const device float4*>(y + y_row_off);
    const device float4* pnw4 =
        reinterpret_cast<const device float4*>(post_norm_w);
    const device float4* res4 =
        reinterpret_cast<const device float4*>(res + x_row_off);
    device float4* hout4 = reinterpret_cast<device float4*>(h_out + y_row_off);
    if constexpr (SKIP_NEXT_NORM) {
      for (size_t j4 = tid; j4 < d4; j4 += 256) {
        const float4 ynv = y4[j4] * rs_post * pnw4[j4];
        float4 acc = float4(0.0f);
        for (int c = 0; c < HC; ++c) {
          const float4 xc = res4[size_t(c) * d4 + j4] + post_prev_reg[c] * ynv;
          acc += pre_reg[c] * xc;
        }
        hout4[j4] = acc;
      }
    } else {
      float h2 = 0.0f;
      for (size_t j4 = tid; j4 < d4; j4 += 256) {
        const float4 ynv = y4[j4] * rs_post * pnw4[j4];
        float4 acc = float4(0.0f);
        for (int c = 0; c < HC; ++c) {
          const float4 xc = res4[size_t(c) * d4 + j4] + post_prev_reg[c] * ynv;
          acc += pre_reg[c] * xc;
        }
        h2 += dot(acc, acc);
      }
      h2 = simd_sum(h2);
      if (lane == 0) {
        tg_h2[sg] = h2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float h2_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        h2_total += tg_h2[s];
      }
      const float rs_next =
          metal::precise::rsqrt(h2_total / float(D) + next_eps);
      const device float4* nnw4 =
          reinterpret_cast<const device float4*>(next_norm_w);
      for (size_t j4 = tid; j4 < d4; j4 += 256) {
        const float4 ynv = y4[j4] * rs_post * pnw4[j4];
        float4 acc = float4(0.0f);
        for (int c = 0; c < HC; ++c) {
          const float4 xc = res4[size_t(c) * d4 + j4] + post_prev_reg[c] * ynv;
          acc += pre_reg[c] * xc;
        }
        hout4[j4] = acc * rs_next * nnw4[j4];
      }
    }
  } else {
    device float* hout = h_out + y_row_off;
    if constexpr (SKIP_NEXT_NORM) {
      for (size_t j = tid; j < size_t(D); j += 256) {
        const float ynv = y[y_row_off + j] * rs_post * post_norm_w[j];
        float acc = 0.0f;
        for (int c = 0; c < HC; ++c) {
          const float xc =
              res[x_row_off + size_t(c) * size_t(D) + j] + post_prev_reg[c] * ynv;
          acc += pre_reg[c] * xc;
        }
        hout[j] = acc;
      }
    } else {
      float h2 = 0.0f;
      for (size_t j = tid; j < size_t(D); j += 256) {
        const float ynv = y[y_row_off + j] * rs_post * post_norm_w[j];
        float acc = 0.0f;
        for (int c = 0; c < HC; ++c) {
          const float xc =
              res[x_row_off + size_t(c) * size_t(D) + j] + post_prev_reg[c] * ynv;
          acc += pre_reg[c] * xc;
        }
        h2 += acc * acc;
      }
      h2 = simd_sum(h2);
      if (lane == 0) {
        tg_h2[sg] = h2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float h2_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        h2_total += tg_h2[s];
      }
      const float rs_next =
          metal::precise::rsqrt(h2_total / float(D) + next_eps);
      for (size_t j = tid; j < size_t(D); j += 256) {
        const float ynv = y[y_row_off + j] * rs_post * post_norm_w[j];
        float acc = 0.0f;
        for (int c = 0; c < HC; ++c) {
          const float xc =
              res[x_row_off + size_t(c) * size_t(D) + j] + post_prev_reg[c] * ynv;
          acc += pre_reg[c] * xc;
        }
        hout[j] = acc * rs_next * next_norm_w[j];
      }
    }
  }
}

template <
    int HC,
    int NSPLITS,
    int TG_X,
    int M_TILE,
    bool HEAD_ONLY,
    typename YPtr,
    typename ResPtr,
    typename PostPrevPtr,
    typename PostNormPtr,
    typename WPtr,
    typename ParamsPtr>
inline void hy4_hc_bridge_pass1_impl(
    YPtr y,
    ResPtr res,
    PostPrevPtr post_prev,
    PostNormPtr post_norm_w,
    WPtr next_weight,
    ParamsPtr params,
    device float* x_out,
    device float* partial,
    threadgroup float* tg_y2,
    threadgroup float* tg_red,
    int M,
    int D,
    uint s_idx,
    uint m_blk,
    uint tid,
    uint lane,
    uint sg) {
  const int m_base = int(m_blk) * M_TILE;
  if (m_base >= M || int(s_idx) >= NSPLITS) {
    return;
  }
  const int m_cnt = metal::min(M_TILE, M - m_base);
  constexpr int N_GATES = HEAD_ONLY ? HC : (2 * HC);
  constexpr int STRIDE = 1 + N_GATES;
  constexpr int PART_COLS = STRIDE + (HEAD_ONLY ? 1 : 0);
  constexpr int NSG = TG_X >> 5;

  const size_t d4 = size_t(D) >> 2;
  const size_t hd = size_t(HC) * size_t(D);
  const size_t hd4 = hd >> 2;
  const float post_eps = params[0];

  float y2[M_TILE];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    y2[r] = 0.0f;
  }
  for (size_t j4 = tid; j4 < d4; j4 += uint(TG_X)) {
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      if (r < m_cnt) {
        const device float4* y4 = reinterpret_cast<const device float4*>(
            y + size_t(m_base + r) * size_t(D));
        const float4 yv = y4[j4];
        y2[r] += dot(yv, yv);
      }
    }
  }

  float rs_post[M_TILE];
  if constexpr (NSG == 1) {
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const float y2_total = simd_sum(y2[r]);
      rs_post[r] = metal::precise::rsqrt(y2_total / float(D) + post_eps);
    }
  } else {
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const float s = simd_sum(y2[r]);
      if (lane == 0u) {
        tg_y2[r * NSG + int(sg)] = s;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      float y2_total = 0.0f;
      for (int i = 0; i < NSG; ++i) {
        y2_total += tg_y2[r * NSG + i];
      }
      rs_post[r] = metal::precise::rsqrt(y2_total / float(D) + post_eps);
    }
  }

  float post_prev_reg[M_TILE][HC];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    const size_t p_off = size_t(metal::min(m_base + r, M - 1)) * size_t(HC);
    for (int c = 0; c < HC; ++c) {
      post_prev_reg[r][c] = post_prev[p_off + size_t(c)];
    }
  }

  const device float4* pnw4 =
      reinterpret_cast<const device float4*>(post_norm_w);
  const device float4* w4_base =
      reinterpret_cast<const device float4*>(next_weight);

  const size_t chunk = (hd4 + size_t(NSPLITS) - 1u) / size_t(NSPLITS);
  const size_t idx4_lo = size_t(s_idx) * chunk;
  const size_t idx4_hi = metal::min(idx4_lo + chunk, hd4);

  float sq_acc[M_TILE];
  float dot_acc[M_TILE][N_GATES];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    sq_acc[r] = 0.0f;
    for (int g = 0; g < N_GATES; ++g) {
      dot_acc[r][g] = 0.0f;
    }
  }

  const uint d4_u = uint(d4);
  for (size_t idx4 = idx4_lo + size_t(tid); idx4 < idx4_hi; idx4 += uint(TG_X)) {
    uint c = 0u;
    uint j4_u = uint(idx4);
    while (j4_u >= d4_u) {
      j4_u -= d4_u;
      ++c;
    }
    const size_t j4 = size_t(j4_u);
    const float4 pnw = pnw4[j4];

    float4 xc[M_TILE];
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      if (r < m_cnt) {
        const int mr = m_base + r;
        const size_t y_off = size_t(mr) * size_t(D);
        const size_t x_off = size_t(mr) * hd;
        const device float4* y4 =
            reinterpret_cast<const device float4*>(y + y_off);
        const device float4* res4 =
            reinterpret_cast<const device float4*>(res + x_off);
        const float4 ynv = y4[j4] * rs_post[r] * pnw;
        const float4 v = res4[idx4] + post_prev_reg[r][c] * ynv;
        xc[r] = v;
        if constexpr (!HEAD_ONLY) {
          reinterpret_cast<device float4*>(x_out + x_off)[idx4] = v;
        }
        sq_acc[r] += dot(v, v);
      }
    }
    for (int g = 0; g < N_GATES; ++g) {
      const float4 wv = w4_base[size_t(g) * hd4 + idx4];
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        if (r < m_cnt) {
          dot_acc[r][g] += dot(xc[r], wv);
        }
      }
    }
  }

  if constexpr (NSG == 1) {
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      if (r < m_cnt) {
        const size_t po =
            (size_t(m_base + r) * size_t(NSPLITS) + size_t(s_idx)) *
            size_t(PART_COLS);
        const float sqs = simd_sum(sq_acc[r]);
        if (lane == 0u) {
          partial[po] = sqs;
        }
        for (int g = 0; g < N_GATES; ++g) {
          const float ds = simd_sum(dot_acc[r][g]);
          if (lane == 0u) {
            partial[po + 1u + size_t(g)] = ds;
          }
        }
        if constexpr (HEAD_ONLY) {
          if (lane == 0u) {
            partial[po + size_t(STRIDE)] = rs_post[r];
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      if (r < m_cnt) {
        const int rbase = (r * NSG + int(sg)) * STRIDE;
        const float sqs = simd_sum(sq_acc[r]);
        if (lane == 0u) {
          tg_red[rbase] = sqs;
        }
        for (int g = 0; g < N_GATES; ++g) {
          const float ds = simd_sum(dot_acc[r][g]);
          if (lane == 0u) {
            tg_red[rbase + 1 + g] = ds;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(m_cnt * STRIDE)) {
      const int r = int(tid) / STRIDE;
      const int col = int(tid) - r * STRIDE;
      float acc = 0.0f;
      for (int i = 0; i < NSG; ++i) {
        acc += tg_red[(r * NSG + i) * STRIDE + col];
      }
      const size_t po =
          (size_t(m_base + r) * size_t(NSPLITS) + size_t(s_idx)) *
          size_t(PART_COLS);
      partial[po + size_t(col)] = acc;
    }
    if constexpr (HEAD_ONLY) {
      if (tid < uint(m_cnt)) {
        const int r = int(tid);
        const size_t po =
            (size_t(m_base + r) * size_t(NSPLITS) + size_t(s_idx)) *
            size_t(PART_COLS);
        partial[po + size_t(STRIDE)] = rs_post[r];
      }
    }
  }
}

template <
    int HC,
    int NSPLITS,
    bool SKIP_NEXT_NORM,
    bool HEAD_ONLY,
    typename XOutPtr,
    typename YPtr,
    typename ResPtr,
    typename PostPrevPtr,
    typename PostNormPtr,
    typename PartialPtr,
    typename ScalePtr,
    typename BasePtr,
    typename NextNormPtr,
    typename ParamsPtr>
inline void hy4_hc_bridge_pass2_impl(
    XOutPtr x_out,
    YPtr y,
    ResPtr res,
    PostPrevPtr post_prev,
    PostNormPtr post_norm_w,
    PartialPtr partial,
    ScalePtr next_scale,
    BasePtr next_base,
    NextNormPtr next_norm_w,
    ParamsPtr params,
    device float* h_out,
    device float* post_next,
    threadgroup float* tg_pre,
    threadgroup float* tg_h2,
    int M,
    int D,
    uint m,
    uint tid,
    uint lane,
    uint sg) {
  if (int(m) >= M) {
    return;
  }
  constexpr int N_GATES = HEAD_ONLY ? HC : (2 * HC);
  constexpr int STRIDE = 1 + N_GATES;
  constexpr int PART_COLS = STRIDE + (HEAD_ONLY ? 1 : 0);

  const size_t d4 = size_t(D) >> 2;
  const size_t hd = size_t(HC) * size_t(D);
  const size_t x_row_off = size_t(m) * hd;
  const size_t y_row_off = size_t(m) * size_t(D);
  const size_t p_row_off = size_t(m) * size_t(HC);
  const size_t part_base = size_t(m) * size_t(NSPLITS) * size_t(PART_COLS);

  const float hc_norm_eps = params[1];
  const float hc_eps = params[2];
  const float magnitude = params[3];
  const float next_eps = params[4];

  if (tid == 0u) {
    float sq_total = 0.0f;
    for (int s = 0; s < NSPLITS; ++s) {
      sq_total += partial[part_base + size_t(s) * size_t(PART_COLS)];
    }
    const float rs_hc =
        1.0f / metal::sqrt(sq_total / float(hd) + hc_norm_eps);
    const float s_pre = next_scale[0];
    for (int c = 0; c < HC; ++c) {
      float dot_total = 0.0f;
      for (int s = 0; s < NSPLITS; ++s) {
        dot_total +=
            partial[part_base + size_t(s) * size_t(PART_COLS) + 1u + size_t(c)];
      }
      const float z_pre = (dot_total * rs_hc) * s_pre + next_base[c];
      tg_pre[c] = 1.0f / (1.0f + metal::exp(-z_pre)) + hc_eps;
    }
    if constexpr (!HEAD_ONLY) {
      const float s_post = next_scale[1];
      for (int c = 0; c < HC; ++c) {
        float dot_total = 0.0f;
        for (int s = 0; s < NSPLITS; ++s) {
          dot_total += partial[part_base + size_t(s) * size_t(PART_COLS) + 1u +
                               size_t(HC + c)];
        }
        const float z_post = (dot_total * rs_hc) * s_post + next_base[HC + c];
        post_next[p_row_off + size_t(c)] =
            (1.0f / (1.0f + metal::exp(-z_post))) * magnitude + hc_eps;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float pre_reg[HC];
  for (int c = 0; c < HC; ++c) {
    pre_reg[c] = tg_pre[c];
  }

  device float4* hout4 = reinterpret_cast<device float4*>(h_out + y_row_off);
  const float rs_post = HEAD_ONLY ? partial[part_base + size_t(STRIDE)] : 0.0f;
  float post_prev_reg[HC];
  if constexpr (HEAD_ONLY) {
    for (int c = 0; c < HC; ++c) {
      post_prev_reg[c] = post_prev[p_row_off + size_t(c)];
    }
  }

  if (d4 <= 768u) {
    float4 h_reg[3];
    float h2 = 0.0f;
    if constexpr (!HEAD_ONLY) {
      const device float4* xo4 =
          reinterpret_cast<const device float4*>(x_out + x_row_off);
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          float4 acc = float4(0.0f);
          for (int c = 0; c < HC; ++c) {
            acc += pre_reg[c] * xo4[size_t(c) * d4 + j4];
          }
          h_reg[k] = acc;
          if constexpr (!SKIP_NEXT_NORM) {
            h2 += dot(acc, acc);
          }
        }
      }
    } else {
      const device float4* y4 =
          reinterpret_cast<const device float4*>(y + y_row_off);
      const device float4* pnw4 =
          reinterpret_cast<const device float4*>(post_norm_w);
      const device float4* res4 =
          reinterpret_cast<const device float4*>(res + x_row_off);
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          const float4 ynv = y4[j4] * rs_post * pnw4[j4];
          float4 acc = float4(0.0f);
          for (int c = 0; c < HC; ++c) {
            const float4 xc = res4[size_t(c) * d4 + j4] + post_prev_reg[c] * ynv;
            acc += pre_reg[c] * xc;
          }
          h_reg[k] = acc;
          if constexpr (!SKIP_NEXT_NORM) {
            h2 += dot(acc, acc);
          }
        }
      }
    }

    if constexpr (SKIP_NEXT_NORM) {
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          hout4[j4] = h_reg[k];
        }
      }
    } else {
      h2 = simd_sum(h2);
      if (lane == 0u) {
        tg_h2[sg] = h2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float h2_total = 0.0f;
      for (int s = 0; s < 8; ++s) {
        h2_total += tg_h2[s];
      }
      const float rs_next =
          metal::precise::rsqrt(h2_total / float(D) + next_eps);
      const device float4* nnw4 =
          reinterpret_cast<const device float4*>(next_norm_w);
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          hout4[j4] = h_reg[k] * rs_next * nnw4[j4];
        }
      }
    }
  } else {
    float h2 = 0.0f;
    if constexpr (!HEAD_ONLY) {
      const device float4* xo4 =
          reinterpret_cast<const device float4*>(x_out + x_row_off);
      for (size_t j4 = tid; j4 < d4; j4 += 256u) {
        float4 acc = float4(0.0f);
        for (int c = 0; c < HC; ++c) {
          acc += pre_reg[c] * xo4[size_t(c) * d4 + j4];
        }
        if constexpr (SKIP_NEXT_NORM) {
          hout4[j4] = acc;
        } else {
          h2 += dot(acc, acc);
        }
      }
      if constexpr (!SKIP_NEXT_NORM) {
        h2 = simd_sum(h2);
        if (lane == 0u) {
          tg_h2[sg] = h2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float h2_total = 0.0f;
        for (int s = 0; s < 8; ++s) {
          h2_total += tg_h2[s];
        }
        const float rs_next =
            metal::precise::rsqrt(h2_total / float(D) + next_eps);
        const device float4* nnw4 =
            reinterpret_cast<const device float4*>(next_norm_w);
        for (size_t j4 = tid; j4 < d4; j4 += 256u) {
          float4 acc = float4(0.0f);
          for (int c = 0; c < HC; ++c) {
            acc += pre_reg[c] * xo4[size_t(c) * d4 + j4];
          }
          hout4[j4] = acc * rs_next * nnw4[j4];
        }
      }
    } else {
      const device float4* y4 =
          reinterpret_cast<const device float4*>(y + y_row_off);
      const device float4* pnw4 =
          reinterpret_cast<const device float4*>(post_norm_w);
      const device float4* res4 =
          reinterpret_cast<const device float4*>(res + x_row_off);
      for (size_t j4 = tid; j4 < d4; j4 += 256u) {
        const float4 ynv = y4[j4] * rs_post * pnw4[j4];
        float4 acc = float4(0.0f);
        for (int c = 0; c < HC; ++c) {
          const float4 xc = res4[size_t(c) * d4 + j4] + post_prev_reg[c] * ynv;
          acc += pre_reg[c] * xc;
        }
        if constexpr (SKIP_NEXT_NORM) {
          hout4[j4] = acc;
        } else {
          h2 += dot(acc, acc);
        }
      }
      if constexpr (!SKIP_NEXT_NORM) {
        h2 = simd_sum(h2);
        if (lane == 0u) {
          tg_h2[sg] = h2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float h2_total = 0.0f;
        for (int s = 0; s < 8; ++s) {
          h2_total += tg_h2[s];
        }
        const float rs_next =
            metal::precise::rsqrt(h2_total / float(D) + next_eps);
        const device float4* nnw4 =
            reinterpret_cast<const device float4*>(next_norm_w);
        for (size_t j4 = tid; j4 < d4; j4 += 256u) {
          const float4 ynv = y4[j4] * rs_post * pnw4[j4];
          float4 acc = float4(0.0f);
          for (int c = 0; c < HC; ++c) {
            const float4 xc =
                res4[size_t(c) * d4 + j4] + post_prev_reg[c] * ynv;
            acc += pre_reg[c] * xc;
          }
          hout4[j4] = acc * rs_next * nnw4[j4];
        }
      }
    }
  }
}
)METAL"
