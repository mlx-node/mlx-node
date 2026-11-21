R"METAL(
template <bool OUT_F16, typename OutT>
inline void hy4_store_kv4(device OutT* dst, size_t off, float4 v) {
  if constexpr (OUT_F16) {
    *reinterpret_cast<device half4*>(dst + off) = half4(v);
  } else {
    *reinterpret_cast<device float4*>(dst + off) = v;
  }
}

template <
    bool VEC4,
    bool OUT_F16,
    bool SPLIT_OUT,
    typename AllPtr,
    typename QNormPtr,
    typename KvNormPtr,
    typename PosPtr,
    typename InvFreqPtr,
    typename EpsPtr,
    typename OutKvT>
inline void hy4_mla_prep_q_kv_impl(
    AllPtr all,
    QNormPtr q_norm_w,
    KvNormPtr kv_norm_w,
    PosPtr pos,
    InvFreqPtr inv_freq,
    EpsPtr norm_eps_ptr,
    device float* qr,
    device OutKvT* kv0,
    device OutKvT* kv1,
    threadgroup float* tg_red,
    int M,
    int q_lora,
    int kv_lora,
    int half_rd,
    size_t all_s0,
    uint m,
    uint tid,
    uint lane,
    uint sg) {
  if (int(m) >= M) {
    return;
  }
  const int rd = half_rd * 2;
  const int kv_row = kv_lora + rd;
  const int half_kv = kv_row >> 1;
  const float norm_eps = norm_eps_ptr[0];
  const size_t row_off = size_t(m) * all_s0;

  if constexpr (VEC4) {
    const size_t q_lora4 = size_t(q_lora) >> 2;
    const size_t kv_lora4 = size_t(kv_lora) >> 2;
    const device float4* q_in4 =
        reinterpret_cast<const device float4*>(all + row_off);
    const device float4* kv_in4 =
        reinterpret_cast<const device float4*>(all + row_off + size_t(q_lora));
    const device float4* qnw4 =
        reinterpret_cast<const device float4*>(q_norm_w);
    const device float4* kvnw4 =
        reinterpret_cast<const device float4*>(kv_norm_w);
    device float4* qr4 =
        reinterpret_cast<device float4*>(qr + size_t(m) * size_t(q_lora));

    if (q_lora4 <= 256u && kv_lora4 <= 256u) {
      float4 qv = float4(0.0f);
      float sq_q = 0.0f;
      if (tid < q_lora4) {
        qv = q_in4[tid];
        sq_q = metal::dot(qv, qv);
      }
      float4 kvv = float4(0.0f);
      float sq_kv = 0.0f;
      if (tid < kv_lora4) {
        kvv = kv_in4[tid];
        sq_kv = metal::dot(kvv, kvv);
      }
      sq_q = simd_sum(sq_q);
      sq_kv = simd_sum(sq_kv);
      if (lane == 0u) {
        tg_red[sg] = sq_q;
        tg_red[8u + sg] = sq_kv;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float sum_q = 0.0f;
      float sum_kv = 0.0f;
      for (int s = 0; s < 8; ++s) {
        sum_q += tg_red[s];
        sum_kv += tg_red[8 + s];
      }
      const float rs_q =
          metal::precise::rsqrt(sum_q / float(q_lora) + norm_eps);
      const float rs_kv =
          metal::precise::rsqrt(sum_kv / float(kv_lora) + norm_eps);

      if (tid < q_lora4) {
        qr4[tid] = qv * rs_q * qnw4[tid];
      }
      if (tid < kv_lora4) {
        const float4 kvn = kvv * rs_kv * kvnw4[tid];
        const size_t col = size_t(tid) * 4u;
        if constexpr (SPLIT_OUT) {
          if (col < size_t(half_kv)) {
            hy4_store_kv4<OUT_F16>(kv0, size_t(m) * size_t(half_kv) + col, kvn);
          } else {
            hy4_store_kv4<OUT_F16>(
                kv1, size_t(m) * size_t(half_kv) + (col - size_t(half_kv)), kvn);
          }
        } else {
          hy4_store_kv4<OUT_F16>(kv0, size_t(m) * size_t(kv_row) + col, kvn);
        }
      }
    } else {
      float sq_q = 0.0f;
      for (size_t j4 = tid; j4 < q_lora4; j4 += 256u) {
        const float4 qv = q_in4[j4];
        sq_q += metal::dot(qv, qv);
      }
      float sq_kv = 0.0f;
      for (size_t j4 = tid; j4 < kv_lora4; j4 += 256u) {
        const float4 kvv = kv_in4[j4];
        sq_kv += metal::dot(kvv, kvv);
      }
      sq_q = simd_sum(sq_q);
      sq_kv = simd_sum(sq_kv);
      if (lane == 0u) {
        tg_red[sg] = sq_q;
        tg_red[8u + sg] = sq_kv;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float sum_q = 0.0f;
      float sum_kv = 0.0f;
      for (int s = 0; s < 8; ++s) {
        sum_q += tg_red[s];
        sum_kv += tg_red[8 + s];
      }
      const float rs_q =
          metal::precise::rsqrt(sum_q / float(q_lora) + norm_eps);
      const float rs_kv =
          metal::precise::rsqrt(sum_kv / float(kv_lora) + norm_eps);

      for (size_t j4 = tid; j4 < q_lora4; j4 += 256u) {
        qr4[j4] = q_in4[j4] * rs_q * qnw4[j4];
      }
      for (size_t j4 = tid; j4 < kv_lora4; j4 += 256u) {
        const float4 kvn = kv_in4[j4] * rs_kv * kvnw4[j4];
        const size_t col = j4 * 4u;
        if constexpr (SPLIT_OUT) {
          if (col < size_t(half_kv)) {
            hy4_store_kv4<OUT_F16>(kv0, size_t(m) * size_t(half_kv) + col, kvn);
          } else {
            hy4_store_kv4<OUT_F16>(
                kv1, size_t(m) * size_t(half_kv) + (col - size_t(half_kv)), kvn);
          }
        } else {
          hy4_store_kv4<OUT_F16>(kv0, size_t(m) * size_t(kv_row) + col, kvn);
        }
      }
    }
  } else {
    float sq_q = 0.0f;
    for (size_t j = tid; j < size_t(q_lora); j += 256u) {
      const float v = all[row_off + j];
      sq_q += v * v;
    }
    float sq_kv = 0.0f;
    for (size_t j = tid; j < size_t(kv_lora); j += 256u) {
      const float v = all[row_off + size_t(q_lora) + j];
      sq_kv += v * v;
    }
    sq_q = simd_sum(sq_q);
    sq_kv = simd_sum(sq_kv);
    if (lane == 0u) {
      tg_red[sg] = sq_q;
      tg_red[8u + sg] = sq_kv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_q = 0.0f;
    float sum_kv = 0.0f;
    for (int s = 0; s < 8; ++s) {
      sum_q += tg_red[s];
      sum_kv += tg_red[8 + s];
    }
    const float rs_q = metal::precise::rsqrt(sum_q / float(q_lora) + norm_eps);
    const float rs_kv =
        metal::precise::rsqrt(sum_kv / float(kv_lora) + norm_eps);

    device float* qr_row = qr + size_t(m) * size_t(q_lora);
    for (size_t j = tid; j < size_t(q_lora); j += 256u) {
      qr_row[j] = all[row_off + j] * rs_q * q_norm_w[j];
    }
    for (size_t j = tid; j < size_t(kv_lora); j += 256u) {
      const float kvn =
          all[row_off + size_t(q_lora) + j] * rs_kv * kv_norm_w[j];
      if constexpr (SPLIT_OUT) {
        if (j < size_t(half_kv)) {
          kv0[size_t(m) * size_t(half_kv) + j] = OutKvT(kvn);
        } else {
          kv1[size_t(m) * size_t(half_kv) + (j - size_t(half_kv))] = OutKvT(kvn);
        }
      } else {
        kv0[size_t(m) * size_t(kv_row) + j] = OutKvT(kvn);
      }
    }
  }

  const float pos_m = float(pos[m]);
  const size_t kpe_off = row_off + size_t(q_lora) + size_t(kv_lora);
  for (uint p = tid; p < uint(half_rd); p += 256u) {
    const float theta = pos_m * inv_freq[p];
    const float c = metal::precise::cos(theta);
    const float s = metal::precise::sin(theta);
    const float x0 = all[kpe_off + size_t(2u * p)];
    const float x1 = all[kpe_off + size_t(2u * p + 1u)];
    const float y0 = x0 * c - x1 * s;
    const float y1 = x0 * s + x1 * c;
    const size_t c0 = size_t(kv_lora) + size_t(2u * p);
    const size_t c1 = c0 + 1u;
    if constexpr (SPLIT_OUT) {
      if (c0 < size_t(half_kv)) {
        kv0[size_t(m) * size_t(half_kv) + c0] = OutKvT(y0);
      } else {
        kv1[size_t(m) * size_t(half_kv) + (c0 - size_t(half_kv))] = OutKvT(y0);
      }
      if (c1 < size_t(half_kv)) {
        kv0[size_t(m) * size_t(half_kv) + c1] = OutKvT(y1);
      } else {
        kv1[size_t(m) * size_t(half_kv) + (c1 - size_t(half_kv))] = OutKvT(y1);
      }
    } else {
      kv0[size_t(m) * size_t(kv_row) + c0] = OutKvT(y0);
      kv0[size_t(m) * size_t(kv_row) + c1] = OutKvT(y1);
    }
  }
}
)METAL"
