R"METAL(
constexpr int VP = FAST_K ? 8 : 4;
constexpr int BLOCK = VP * 32;
const int M = x_shape[0];
const int K = x_shape[1];
const int V = w_shape[0];
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int first = int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(sg) * RPS;
const int row0 = int(threadgroup_position_in_grid.z) * R_TILE;
if (first >= V || row0 >= M) return;

float acc[R_TILE][RPS];
#pragma unroll
for (int r = 0; r < R_TILE; ++r) {
#pragma unroll
  for (int o = 0; o < RPS; ++o) {
    acc[r][o] = 0.0f;
  }
}

if constexpr (FAST_K) {
  for (int base = 0; base < K; base += BLOCK) {
    const int col = base + int(lane) * 8;
    if constexpr (STREAM_X) {
#pragma unroll
      for (int o = 0; o < RPS; ++o) {
        const int vrow = first + o;
        if (vrow < V) {
          const device bfloat4* wp = reinterpret_cast<const device bfloat4*>(
              w + size_t(vrow) * size_t(K) + col);
          const float4 w0 = float4(wp[0]);
          const float4 w1 = float4(wp[1]);
          for (int r = 0; r < R_TILE; ++r) {
            const int row = row0 + r;
            if (row < M) {
              const device float4* xp = reinterpret_cast<const device float4*>(
                  x + size_t(row) * size_t(K) + col);
              acc[r][o] += dot(xp[0], w0);
              acc[r][o] += dot(xp[1], w1);
            }
          }
        }
      }
    } else {
      float4 x0[R_TILE];
      float4 x1[R_TILE];
#pragma unroll
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
#pragma unroll
      for (int o = 0; o < RPS; ++o) {
        const int vrow = first + o;
        if (vrow < V) {
          const device bfloat4* wp = reinterpret_cast<const device bfloat4*>(
              w + size_t(vrow) * size_t(K) + col);
          const float4 w0 = float4(wp[0]);
          const float4 w1 = float4(wp[1]);
#pragma unroll
          for (int r = 0; r < R_TILE; ++r) {
            acc[r][o] += dot(x0[r], w0);
            acc[r][o] += dot(x1[r], w1);
          }
        }
      }
    }
  }
} else {
  for (int base = 0; base < K; base += BLOCK) {
    const int col = base + int(lane) * 4;
    if (col < K) {
      if constexpr (STREAM_X) {
#pragma unroll
        for (int o = 0; o < RPS; ++o) {
          const int vrow = first + o;
          if (vrow < V) {
            const float4 w0 = float4(*reinterpret_cast<const device bfloat4*>(
                w + size_t(vrow) * size_t(K) + col));
            for (int r = 0; r < R_TILE; ++r) {
              const int row = row0 + r;
              if (row < M) {
                const float4 xv = *reinterpret_cast<const device float4*>(
                    x + size_t(row) * size_t(K) + col);
                acc[r][o] += dot(xv, w0);
              }
            }
          }
        }
      } else {
        float4 x0[R_TILE];
#pragma unroll
        for (int r = 0; r < R_TILE; ++r) {
          const int row = row0 + r;
          if (row < M) {
            x0[r] = *reinterpret_cast<const device float4*>(
                x + size_t(row) * size_t(K) + col);
          } else {
            x0[r] = float4(0.0f);
          }
        }
#pragma unroll
        for (int o = 0; o < RPS; ++o) {
          const int vrow = first + o;
          if (vrow < V) {
            const float4 w0 = float4(*reinterpret_cast<const device bfloat4*>(
                w + size_t(vrow) * size_t(K) + col));
#pragma unroll
            for (int r = 0; r < R_TILE; ++r) {
              acc[r][o] += dot(x0[r], w0);
            }
          }
        }
      }
    }
  }
}

#pragma unroll
for (int r = 0; r < R_TILE; ++r) {
  const int row = row0 + r;
  if (row < M) {
#pragma unroll
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