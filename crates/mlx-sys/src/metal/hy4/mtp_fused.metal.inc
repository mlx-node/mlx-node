R"METAL(
template <
    bool VEC4,
    bool USE_REG,
    typename EPtr,
    typename HPtr,
    typename WPtr,
    typename EpsPtr>
inline void hy4_mtp_eh_norm_concat_impl(
    EPtr e_in,
    HPtr h_in,
    WPtr enorm,
    WPtr hnorm,
    EpsPtr eps_ptr,
    device float* out,
    threadgroup float* tg_sq,
    int M,
    int D,
    uint m,
    uint half_idx,
    uint tid,
    uint lane,
    uint sg) {
  if (int(m) >= M || half_idx >= 2u) {
    return;
  }
  const float eps = eps_ptr[0];
  const size_t in_off = size_t(m) * size_t(D);
  const size_t out_off = size_t(m) * size_t(2 * D) + size_t(half_idx) * size_t(D);
  WPtr nw = (half_idx == 0u) ? enorm : hnorm;

  if constexpr (VEC4 && USE_REG) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* nw4 = reinterpret_cast<const device float4*>(nw);
    device float4* out4 = reinterpret_cast<device float4*>(out + out_off);
    float4 reg[3];
    float sq = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        const size_t j = j4 << 2;
        float4 v;
        if (half_idx == 0u) {
          v = float4(
              float(e_in[in_off + j + 0u]),
              float(e_in[in_off + j + 1u]),
              float(e_in[in_off + j + 2u]),
              float(e_in[in_off + j + 3u]));
        } else {
          v = float4(
              float(h_in[in_off + j + 0u]),
              float(h_in[in_off + j + 1u]),
              float(h_in[in_off + j + 2u]),
              float(h_in[in_off + j + 3u]));
        }
        reg[k] = v;
        sq += dot(v, v);
      }
    }
    sq = simd_sum(sq);
    if (lane == 0u) {
      tg_sq[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sq_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      sq_total += tg_sq[s];
    }
    const float rs = metal::precise::rsqrt(sq_total / float(D) + eps);
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        out4[j4] = reg[k] * rs * nw4[j4];
      }
    }
  } else {
    float sq = 0.0f;
    for (size_t j = tid; j < size_t(D); j += 256u) {
      const float v =
          (half_idx == 0u) ? float(e_in[in_off + j]) : float(h_in[in_off + j]);
      sq += v * v;
    }
    sq = simd_sum(sq);
    if (lane == 0u) {
      tg_sq[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sq_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      sq_total += tg_sq[s];
    }
    const float rs = metal::precise::rsqrt(sq_total / float(D) + eps);
    for (size_t j = tid; j < size_t(D); j += 256u) {
      const float v =
          (half_idx == 0u) ? float(e_in[in_off + j]) : float(h_in[in_off + j]);
      out[out_off + j] = v * rs * nw[j];
    }
  }
}

template <
    bool VEC4,
    bool USE_REG,
    bool NEED_X_OUT,
    typename ResPtr,
    typename YPtr,
    typename PostWPtr,
    typename NextWPtr,
    typename EpsPtr>
inline void hy4_post_norm_add_rms_norm_impl(
    ResPtr res,
    YPtr y,
    PostWPtr post_w,
    NextWPtr next_w,
    EpsPtr eps_ptr,
    device float* x_out,
    device float* h_out,
    threadgroup float* tg_y2,
    threadgroup float* tg_x2,
    int M,
    int D,
    uint m,
    uint tid,
    uint lane,
    uint sg) {
  if (int(m) >= M) {
    return;
  }
  const float eps = eps_ptr[0];
  const size_t row_off = size_t(m) * size_t(D);

  if constexpr (VEC4 && USE_REG) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* res4 =
        reinterpret_cast<const device float4*>(res + row_off);
    const device float4* y4 =
        reinterpret_cast<const device float4*>(y + row_off);
    const device float4* pw4 = reinterpret_cast<const device float4*>(post_w);
    const device float4* nw4 = reinterpret_cast<const device float4*>(next_w);
    device float4* hout4 = reinterpret_cast<device float4*>(h_out + row_off);

    float4 reg[3];
    float y2 = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        const float4 yv = y4[j4];
        reg[k] = yv;
        y2 += dot(yv, yv);
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
    const float rs_post = metal::precise::rsqrt(y2_total / float(D) + eps);

    float x2 = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        const float4 xv = res4[j4] + reg[k] * rs_post * pw4[j4];
        reg[k] = xv;
        if constexpr (NEED_X_OUT) {
          reinterpret_cast<device float4*>(x_out + row_off)[j4] = xv;
        }
        x2 += dot(xv, xv);
      }
    }
    x2 = simd_sum(x2);
    if (lane == 0u) {
      tg_x2[sg] = x2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float x2_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      x2_total += tg_x2[s];
    }
    const float rs_next = metal::precise::rsqrt(x2_total / float(D) + eps);
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        hout4[j4] = reg[k] * rs_next * nw4[j4];
      }
    }
  } else {
    float y2 = 0.0f;
    for (size_t j = tid; j < size_t(D); j += 256u) {
      const float yv = y[row_off + j];
      y2 += yv * yv;
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
    const float rs_post = metal::precise::rsqrt(y2_total / float(D) + eps);

    float x2 = 0.0f;
    for (size_t j = tid; j < size_t(D); j += 256u) {
      const float xv = res[row_off + j] + y[row_off + j] * rs_post * post_w[j];
      if constexpr (NEED_X_OUT) {
        x_out[row_off + j] = xv;
      }
      x2 += xv * xv;
    }
    x2 = simd_sum(x2);
    if (lane == 0u) {
      tg_x2[sg] = x2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float x2_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      x2_total += tg_x2[s];
    }
    const float rs_next = metal::precise::rsqrt(x2_total / float(D) + eps);
    for (size_t j = tid; j < size_t(D); j += 256u) {
      const float xv = res[row_off + j] + y[row_off + j] * rs_post * post_w[j];
      h_out[row_off + j] = xv * rs_next * next_w[j];
    }
  }
}
)METAL"
