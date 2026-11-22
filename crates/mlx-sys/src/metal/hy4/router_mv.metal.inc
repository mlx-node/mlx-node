R"METAL(
constexpr int EXPERTS_PER_TG = NUM_SG * RPS;

const int M = x_shape[0];
const int K = x_shape[1];
const int E = w_shape[0];

const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int n_etiles = (E + EXPERTS_PER_TG - 1) / EXPERTS_PER_TG;
const int z = int(threadgroup_position_in_grid.z);
const int e_tile = z % n_etiles;
const int m_tile = z / n_etiles;

const int e0 = e_tile * EXPERTS_PER_TG + int(sg) * RPS;
const int row0 = m_tile * M_TILE;
if (e0 >= E || row0 >= M) return;

const int n_f4 = K >> 2;
const device float4* x_base = reinterpret_cast<const device float4*>(x);
const device float4* w_f4 = reinterpret_cast<const device float4*>(w);
const device half4* w_h4 = reinterpret_cast<const device half4*>(w);

uint x_off[M_TILE];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const int mr = metal::min(row0 + r, M - 1);
  x_off[r] = uint(size_t(mr) * size_t(n_f4));
}

float acc[M_TILE][RPS];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
#pragma unroll
  for (int p = 0; p < RPS; ++p) {
    acc[r][p] = 0.0f;
  }
}

const int k_main_f4 = FAST_K ? n_f4 : (n_f4 & ~127);

for (int base_f4 = 0; base_f4 < k_main_f4; base_f4 += 128) {
  const size_t i0 = size_t(base_f4 + int(lane));
  const size_t i1 = i0 + 32u;
  const size_t i2 = i0 + 64u;
  const size_t i3 = i0 + 96u;
  float4 w0[RPS], w1[RPS], w2[RPS], w3[RPS];
#pragma unroll
  for (int p = 0; p < RPS; ++p) {
    const size_t wo = size_t(metal::min(e0 + p, E - 1)) * size_t(n_f4);
    if constexpr (W_F16) {
      w0[p] = float4(w_h4[wo + i0]);
      w1[p] = float4(w_h4[wo + i1]);
      w2[p] = float4(w_h4[wo + i2]);
      w3[p] = float4(w_h4[wo + i3]);
    } else {
      w0[p] = w_f4[wo + i0];
      w1[p] = w_f4[wo + i1];
      w2[p] = w_f4[wo + i2];
      w3[p] = w_f4[wo + i3];
    }
  }
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    const size_t xo = size_t(x_off[r]);
    const float4 x0 = x_base[xo + i0];
    const float4 x1 = x_base[xo + i1];
    const float4 x2 = x_base[xo + i2];
    const float4 x3 = x_base[xo + i3];
#pragma unroll
    for (int p = 0; p < RPS; ++p) {
      acc[r][p] += metal::dot(x0, w0[p]);
      acc[r][p] += metal::dot(x1, w1[p]);
      acc[r][p] += metal::dot(x2, w2[p]);
      acc[r][p] += metal::dot(x3, w3[p]);
    }
  }
}

if constexpr (!FAST_K) {
  for (int base_f4 = k_main_f4; base_f4 < n_f4; base_f4 += 32) {
    const int idx_i = base_f4 + int(lane);
    if (idx_i < n_f4) {
      const size_t idx = size_t(idx_i);
      float4 wv[RPS];
#pragma unroll
      for (int p = 0; p < RPS; ++p) {
        const size_t wo = size_t(metal::min(e0 + p, E - 1)) * size_t(n_f4);
        if constexpr (W_F16) {
          wv[p] = float4(w_h4[wo + idx]);
        } else {
          wv[p] = w_f4[wo + idx];
        }
      }
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const float4 xv = x_base[size_t(x_off[r]) + idx];
#pragma unroll
        for (int p = 0; p < RPS; ++p) {
          acc[r][p] += metal::dot(xv, wv[p]);
        }
      }
    }
  }
}

#pragma unroll
for (int p = 0; p < RPS; ++p) {
  const int ep = e0 + p;
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    const float v0 = simd_sum(acc[r][p]);
    const int mr = row0 + r;
    if (lane == 0u && mr < M && ep < E) {
      out[size_t(mr) * size_t(E) + size_t(ep)] = v0;
    }
  }
}
)METAL"