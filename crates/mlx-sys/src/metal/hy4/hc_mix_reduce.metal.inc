R"METAL(
const int M = x_shape[0];
const int D = x_shape[2];
const uint m = threadgroup_position_in_grid.y;
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
if (int(m) >= M) return;

constexpr int N_GATES = HAS_POST ? (2 * HC) : HC;
constexpr int STRIDE = 1 + N_GATES;
const size_t hd = size_t(HC) * size_t(D);
const size_t x_row_off = size_t(m) * hd;

threadgroup float tg_red[8 * STRIDE];
threadgroup float tg_pre[HC];

float sq_acc = 0.0f;
float dot_acc[N_GATES];
for (int g = 0; g < N_GATES; ++g) {
  dot_acc[g] = 0.0f;
}

if (VEC4) {
  const size_t hd4 = hd >> 2;
  const device float4* x4 = reinterpret_cast<const device float4*>(x + x_row_off);
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
  const device float* xr = x + x_row_off;
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
  const float norm_eps = params[0];
  const float hc_eps = params[1];
  const float magnitude = params[2];
  const float rs = 1.0f / metal::sqrt(sq_total / float(hd) + norm_eps);
  const float s_pre = scale[0];
  for (int c = 0; c < HC; ++c) {
    float dot_total = 0.0f;
    for (int s = 0; s < 8; ++s) {
      dot_total += tg_red[s * STRIDE + 1 + c];
    }
    const float z_pre = (dot_total * rs) * s_pre + base[c];
    // metal::exp preserves full f32 sigmoid precision across row batches.
    tg_pre[c] = 1.0f / (1.0f + metal::exp(-z_pre)) + hc_eps;
  }
  if (HAS_POST) {
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

if (VEC4) {
  const size_t d4 = size_t(D) >> 2;
  device float4* h4 = reinterpret_cast<device float4*>(h + size_t(m) * size_t(D));
  for (size_t j4 = tid; j4 < d4; j4 += 256) {
    float4 acc = float4(0.0f);
    for (int c = 0; c < HC; ++c) {
      const device float4* xc4 = reinterpret_cast<const device float4*>(
          x + x_row_off + size_t(c) * size_t(D));
      acc += pre_reg[c] * xc4[j4];
    }
    h4[j4] = acc;
  }
} else {
  device float* hr = h + size_t(m) * size_t(D);
  for (size_t j = tid; j < size_t(D); j += 256) {
    float acc = 0.0f;
    for (int c = 0; c < HC; ++c) {
      acc += pre_reg[c] * x[x_row_off + size_t(c) * size_t(D) + j];
    }
    hr[j] = acc;
  }
}
)METAL"
