R"METAL(
template <typename RowsPtr, typename NormPtr, typename EpsPtr>
inline void hy4_engram_norm_rows_impl(
    RowsPtr rows,
    NormPtr lookup_norm,
    EpsPtr eps_ptr,
    device float* out,
    threadgroup float* tg_sq,
    int M,
    int COLS,
    int HD,
    uint col,
    uint m,
    uint tid,
    uint lane,
    uint sg) {
  if (int(m) >= M || int(col) >= COLS) {
    return;
  }
  const size_t base = (size_t(m) * size_t(COLS) + size_t(col)) * size_t(HD);
  const size_t nw_base = size_t(col) * size_t(HD);
  const float eps = eps_ptr[0];

  if (HD <= 128) {
    float v = 0.0f;
    if (int(tid) < HD) {
      v = rows[base + size_t(tid)];
    }
    float sq = simd_sum(v * v);
    if (lane == 0u) {
      tg_sq[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sq_tot = tg_sq[0] + tg_sq[1] + tg_sq[2] + tg_sq[3];
    const float rs = metal::precise::rsqrt(sq_tot / float(HD) + eps);
    if (int(tid) < HD) {
      out[base + size_t(tid)] = v * rs * lookup_norm[nw_base + size_t(tid)];
    }
  } else {
    float sq = 0.0f;
    for (size_t j = tid; j < size_t(HD); j += 128u) {
      const float v = rows[base + j];
      sq += v * v;
    }
    sq = simd_sum(sq);
    if (lane == 0u) {
      tg_sq[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sq_tot = tg_sq[0] + tg_sq[1] + tg_sq[2] + tg_sq[3];
    const float rs = metal::precise::rsqrt(sq_tot / float(HD) + eps);
    for (size_t j = tid; j < size_t(HD); j += 128u) {
      out[base + j] = rows[base + j] * rs * lookup_norm[nw_base + j];
    }
  }
}

template <
    int HC,
    bool VEC4,
    bool USE_REG,
    bool HAS_MASK,
    typename XPtr,
    typename KvRawPtr,
    typename KvBPtr,
    typename GateWPtr,
    typename PreGammaPtr,
    typename HcScalePtr,
    typename MaskPtr,
    typename EpsPtr>
inline void hy4_engram_gate_add_impl(
    XPtr x,
    KvRawPtr kv_raw,
    KvBPtr kv_b,
    GateWPtr gate_w,
    PreGammaPtr pre_gamma,
    HcScalePtr hc_scale,
    MaskPtr keep_mask,
    EpsPtr eps_ptr,
    device float* out,
    threadgroup float* tg_red,
    int M,
    int D,
    uint m,
    uint tid,
    uint lane,
    uint sg) {
  if (int(m) >= M) {
    return;
  }
  const size_t hd = size_t(HC) * size_t(D);
  const size_t x_row_off = size_t(m) * hd;

  if constexpr (HAS_MASK) {
    if (keep_mask[m] == 0u) {
      if constexpr (VEC4) {
        const size_t hd4 = hd >> 2;
        const device float4* x4 =
            reinterpret_cast<const device float4*>(x + x_row_off);
        device float4* o4 =
            reinterpret_cast<device float4*>(out + x_row_off);
        for (size_t i = tid; i < hd4; i += 256u) {
          o4[i] = x4[i];
        }
      } else {
        for (size_t i = tid; i < hd; i += 256u) {
          out[x_row_off + i] = x[x_row_off + i];
        }
      }
      return;
    }
  }

  const float eps = eps_ptr[0];
  const float inv_sqrt_d = metal::precise::rsqrt(float(D));
  const size_t kv_row_off = size_t(m) * size_t(HC + 1) * size_t(D);
  const size_t val_off = size_t(HC) * size_t(D);

  if constexpr (USE_REG) {
    const size_t d4 = size_t(D) >> 2;
    const device float4* kv_val4 =
        reinterpret_cast<const device float4*>(kv_raw + kv_row_off + val_off);
    const device float4* kvb_val4 =
        reinterpret_cast<const device float4*>(kv_b + val_off);
    const device float4* pg4 =
        reinterpret_cast<const device float4*>(pre_gamma);

    float4 sig_reg[3];
    float v2 = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        const float4 vv = kv_val4[j4] + kvb_val4[j4];
        sig_reg[k] = vv;
        v2 += metal::dot(vv, vv);
      }
    }
    v2 = simd_sum(v2);
    if (lane == 0u) {
      tg_red[sg] = v2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v2_tot = 0.0f;
    for (int s = 0; s < 8; ++s) {
      v2_tot += tg_red[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rs_val = metal::precise::rsqrt(v2_tot / float(D) + eps);
    for (int k = 0; k < 3; ++k) {
      const size_t j4 = size_t(tid) + size_t(k) * 256u;
      if (j4 < d4) {
        sig_reg[k] = sig_reg[k] * rs_val * pg4[j4];
      }
    }

    for (int c = 0; c < HC; ++c) {
      const size_t c_off = size_t(c) * size_t(D);
      const device float4* xc4 =
          reinterpret_cast<const device float4*>(x + x_row_off + c_off);
      const device float4* kc4 =
          reinterpret_cast<const device float4*>(kv_raw + kv_row_off + c_off);
      const device float4* kb4 =
          reinterpret_cast<const device float4*>(kv_b + c_off);
      const device float4* gw4 =
          reinterpret_cast<const device float4*>(gate_w + c_off);
      const device float4* hs4 =
          reinterpret_cast<const device float4*>(hc_scale + c_off);
      device float4* oc4 =
          reinterpret_cast<device float4*>(out + x_row_off + c_off);

      float4 x_reg[3];
      float4 k_reg[3];
      float x2 = 0.0f;
      float k2 = 0.0f;
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          x_reg[k] = xc4[j4];
          k_reg[k] = kc4[j4] + kb4[j4];
          x2 += metal::dot(x_reg[k], x_reg[k]);
          k2 += metal::dot(k_reg[k], k_reg[k]);
        }
      }
      x2 = simd_sum(x2);
      k2 = simd_sum(k2);
      if (lane == 0u) {
        tg_red[sg] = x2;
        tg_red[8u + sg] = k2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float x2_tot = 0.0f;
      float k2_tot = 0.0f;
      for (int s = 0; s < 8; ++s) {
        x2_tot += tg_red[s];
        k2_tot += tg_red[8 + s];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float rs_x = metal::precise::rsqrt(x2_tot / float(D) + eps);
      const float rs_k = metal::precise::rsqrt(k2_tot / float(D) + eps);

      float dot_acc = 0.0f;
      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          const float4 xn = x_reg[k] * rs_x;
          const float4 kn = k_reg[k] * rs_k;
          dot_acc += metal::dot(xn * kn, gw4[j4]);
        }
      }
      dot_acc = simd_sum(dot_acc);
      if (lane == 0u) {
        tg_red[sg] = dot_acc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float dot_tot = 0.0f;
      for (int s = 0; s < 8; ++s) {
        dot_tot += tg_red[s];
      }
      const float dot_val = dot_tot * inv_sqrt_d;
      const float a = metal::sqrt(metal::max(metal::abs(dot_val), 1e-6f));
      const float signed_a = (dot_val < 0.0f) ? -a : a;
      const float g_c = 1.0f / (1.0f + metal::precise::exp(-signed_a));

      for (int k = 0; k < 3; ++k) {
        const size_t j4 = size_t(tid) + size_t(k) * 256u;
        if (j4 < d4) {
          oc4[j4] = x_reg[k] + sig_reg[k] * (g_c + hs4[j4]);
        }
      }
      if (c + 1 < HC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  } else {
    float v2 = 0.0f;
    for (size_t j = tid; j < size_t(D); j += 256u) {
      const float vv = kv_raw[kv_row_off + val_off + j] + kv_b[val_off + j];
      v2 += vv * vv;
    }
    v2 = simd_sum(v2);
    if (lane == 0u) {
      tg_red[sg] = v2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v2_tot = 0.0f;
    for (int s = 0; s < 8; ++s) {
      v2_tot += tg_red[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rs_val = metal::precise::rsqrt(v2_tot / float(D) + eps);

    for (int c = 0; c < HC; ++c) {
      const size_t c_off = size_t(c) * size_t(D);
      float x2 = 0.0f;
      float k2 = 0.0f;
      for (size_t j = tid; j < size_t(D); j += 256u) {
        const float xv = x[x_row_off + c_off + j];
        const float kv = kv_raw[kv_row_off + c_off + j] + kv_b[c_off + j];
        x2 += xv * xv;
        k2 += kv * kv;
      }
      x2 = simd_sum(x2);
      k2 = simd_sum(k2);
      if (lane == 0u) {
        tg_red[sg] = x2;
        tg_red[8u + sg] = k2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float x2_tot = 0.0f;
      float k2_tot = 0.0f;
      for (int s = 0; s < 8; ++s) {
        x2_tot += tg_red[s];
        k2_tot += tg_red[8 + s];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float rs_x = metal::precise::rsqrt(x2_tot / float(D) + eps);
      const float rs_k = metal::precise::rsqrt(k2_tot / float(D) + eps);

      float dot_acc = 0.0f;
      for (size_t j = tid; j < size_t(D); j += 256u) {
        const float xn = x[x_row_off + c_off + j] * rs_x;
        const float kn =
            (kv_raw[kv_row_off + c_off + j] + kv_b[c_off + j]) * rs_k;
        dot_acc += (xn * kn) * gate_w[c_off + j];
      }
      dot_acc = simd_sum(dot_acc);
      if (lane == 0u) {
        tg_red[sg] = dot_acc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float dot_tot = 0.0f;
      for (int s = 0; s < 8; ++s) {
        dot_tot += tg_red[s];
      }
      const float dot_val = dot_tot * inv_sqrt_d;
      const float a = metal::sqrt(metal::max(metal::abs(dot_val), 1e-6f));
      const float signed_a = (dot_val < 0.0f) ? -a : a;
      const float g_c = 1.0f / (1.0f + metal::precise::exp(-signed_a));

      for (size_t j = tid; j < size_t(D); j += 256u) {
        const float sig = (kv_raw[kv_row_off + val_off + j] + kv_b[val_off + j]) *
                          rs_val * pre_gamma[j];
        out[x_row_off + c_off + j] =
            x[x_row_off + c_off + j] + sig * (g_c + hc_scale[c_off + j]);
      }
      if (c + 1 < HC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
}
)METAL"
