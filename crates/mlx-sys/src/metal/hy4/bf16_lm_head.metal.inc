R"METAL(
constexpr int VP = FAST_K ? 8 : 4;
constexpr int BLOCK = VP * 32;
const int M = x_shape[0];
const int K = x_shape[1];
const int V = w_shape[0];
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int first = int(threadgroup_position_in_grid.y) * (2 * RPS) + int(sg) * RPS;
const int row0 = int(threadgroup_position_in_grid.z) * R_TILE;
if (first >= V || row0 >= M) return;

float acc[R_TILE][RPS];
for (int r = 0; r < R_TILE; ++r) {
  for (int o = 0; o < RPS; ++o) {
    acc[r][o] = 0.0f;
  }
}

if (FAST_K) {
  for (int base = 0; base < K; base += BLOCK) {
    const int col = base + int(lane) * 8;
    float4 x0[R_TILE];
    float4 x1[R_TILE];
    for (int r = 0; r < R_TILE; ++r) {
      const int row = row0 + r;
      if (row < M) {
        const device float4* xp = reinterpret_cast<const device float4*>(
            x + size_t(row) * size_t(K) + col);
        x0[r] = xp[0];
        x1[r] = xp[1];
      } else {
        x0[r] = float4(0.0f);
        x1[r] = float4(0.0f);
      }
    }
    for (int o = 0; o < RPS; ++o) {
      const int vrow = first + o;
      if (vrow < V) {
        const uint4 w_raw = *reinterpret_cast<const device uint4*>(
            w + size_t(vrow) * size_t(K) + col);
        const float4 w0 = float4(
            as_type<float>(w_raw.x << 16),
            as_type<float>(w_raw.x & 0xFFFF0000u),
            as_type<float>(w_raw.y << 16),
            as_type<float>(w_raw.y & 0xFFFF0000u));
        const float4 w1 = float4(
            as_type<float>(w_raw.z << 16),
            as_type<float>(w_raw.z & 0xFFFF0000u),
            as_type<float>(w_raw.w << 16),
            as_type<float>(w_raw.w & 0xFFFF0000u));
        for (int r = 0; r < R_TILE; ++r) {
          acc[r][o] += dot(x0[r], w0);
          acc[r][o] += dot(x1[r], w1);
        }
      }
    }
  }
} else {
  for (int base = 0; base < K; base += BLOCK) {
    const int col = base + int(lane) * 4;
    if (col < K) {
      float4 x0[R_TILE];
      for (int r = 0; r < R_TILE; ++r) {
        const int row = row0 + r;
        if (row < M) {
          x0[r] = *reinterpret_cast<const device float4*>(
              x + size_t(row) * size_t(K) + col);
        } else {
          x0[r] = float4(0.0f);
        }
      }
      for (int o = 0; o < RPS; ++o) {
        const int vrow = first + o;
        if (vrow < V) {
          const uint2 w_raw = *reinterpret_cast<const device uint2*>(
              w + size_t(vrow) * size_t(K) + col);
          const float4 w0 = float4(
              as_type<float>(w_raw.x << 16),
              as_type<float>(w_raw.x & 0xFFFF0000u),
              as_type<float>(w_raw.y << 16),
              as_type<float>(w_raw.y & 0xFFFF0000u));
          for (int r = 0; r < R_TILE; ++r) {
            acc[r][o] += dot(x0[r], w0);
          }
        }
      }
    }
  }
}

for (int r = 0; r < R_TILE; ++r) {
  const int row = row0 + r;
  if (row < M) {
    for (int o = 0; o < RPS; ++o) {
      const int vrow = first + o;
      if (vrow < V) {
        const float val = simd_sum(acc[r][o]);
        if (lane == 0) {
          out[size_t(row) * size_t(V) + size_t(vrow)] = val;
        }
      }
    }
  }
}
)METAL"
