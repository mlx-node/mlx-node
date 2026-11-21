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
    bool USE_TG_YNORM,
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
    threadgroup float4* tg_yn4,
    threadgroup float* tg_red,
    threadgroup float* tg_pre,
    threadgroup float* tg_y2,
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
  const size_t y_row_off = size_t(m) * size_t(D);
  const size_t p_row_off = size_t(m) * size_t(HC);

  const float post_eps = params[0];
  const float hc_norm_eps = params[1];
  const float hc_eps = params[2];
  const float magnitude = params[3];
  const float next_eps = params[4];

  float post_prev_reg[HC];
  for (int c = 0; c < HC; ++c) {
    post_prev_reg[c] = post_prev[p_row_off + size_t(c)];
  }

  float rs_post = 0.0f;
  if constexpr (USE_TG_YNORM) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* y4 =
        reinterpret_cast<const device float4*>(y + y_row_off);
    const device float4* pnw4 =
        reinterpret_cast<const device float4*>(post_norm_w);
    float4 y_reg[3];
    float y2 = 0.0f;
    for (int k = 0; k < 3; ++k) {
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
    rs_post = metal::precise::rsqrt(y2_total / float(D) + post_eps);
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256;
      if (j4 < d4) {
        tg_yn4[j4] = y_reg[k] * rs_post * pnw4[j4];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else if constexpr (VEC4) {
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

  if constexpr (USE_TG_YNORM) {
    const size_t d4 = size_t(D) >> 2;
    const size_t hd4 = hd >> 2;
    const device float4* res4 =
        reinterpret_cast<const device float4*>(res + x_row_off);
    for (size_t idx4 = tid; idx4 < hd4; idx4 += 256) {
      const int c = int(idx4 / d4);
      const size_t j4 = idx4 - size_t(c) * d4;
      const float4 xc = res4[idx4] + post_prev_reg[c] * tg_yn4[j4];
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
  } else if constexpr (VEC4) {
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

  if constexpr (USE_TG_YNORM) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* res4 =
        reinterpret_cast<const device float4*>(res + x_row_off);
    device float4* hout4 = reinterpret_cast<device float4*>(h_out + y_row_off);
    float4 h_reg[3];
    float h2 = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256;
      if (j4 < d4) {
        const float4 ynv = tg_yn4[j4];
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
    if constexpr (SKIP_NEXT_NORM) {
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256;
        if (j4 < d4) {
          hout4[j4] = h_reg[k];
        }
      }
    } else {
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
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256;
        if (j4 < d4) {
          hout4[j4] = h_reg[k] * rs_next * nnw4[j4];
        }
      }
    }
  } else if constexpr (VEC4) {
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
)METAL"
