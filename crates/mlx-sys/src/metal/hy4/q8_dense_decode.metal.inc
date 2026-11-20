R"METAL(
constexpr int VP = FAST ? 8 : 4;
constexpr int BLOCK = VP * 32;
constexpr int K_AXIS = GROUPED ? 2 : 1;

const int M = x_shape[0];
const int G = GROUPED ? x_shape[1] : 1;
const int K = x_shape[K_AXIS];
const int N = w_shape[0] / G;

const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int first = int(threadgroup_position_in_grid.y) * (2 * RPS) + int(sg) * RPS;
const int n_rtiles = (M + R_TILE - 1) / R_TILE;
const int z = int(threadgroup_position_in_grid.z);
const int g = GROUPED ? (z / n_rtiles) : 0;
const int row0 = (GROUPED ? (z % n_rtiles) : z) * R_TILE;

if (first >= N || row0 >= M || g >= G) return;

float acc[R_TILE][RPS];
for (int r = 0; r < R_TILE; ++r) {
  for (int o = 0; o < RPS; ++o) {
    acc[r][o] = 0.0f;
  }
}

const device uint8_t* w_bytes = reinterpret_cast<const device uint8_t*>(w);
const size_t k_stride = size_t(K);
const size_t g_stride = size_t(K / 32);

for (int base = 0; base < K; base += BLOCK) {
  const int col = base + int(lane) * VP;
  if (FAST || col < K) {
    float values[R_TILE][VP];
    float sums[R_TILE];
    for (int r = 0; r < R_TILE; ++r) {
      const int row = row0 + r;
      if (row < M) {
        const size_t x_off = (size_t(row) * size_t(G) + size_t(g)) * k_stride + size_t(col);
        sums[r] = load_vector<float, float, VP, 8>(x + x_off, values[r]);
      }
    }
    const size_t side_col = size_t(col / 32);
    for (int o = 0; o < RPS; ++o) {
      const int n = first + o;
      if (FAST || n < N) {
        const size_t wrow = size_t(g) * size_t(N) + size_t(n);
        const device uint8_t* codes = w_bytes + wrow * k_stride + size_t(col);
        const size_t side = wrow * g_stride + side_col;
        const float scale = scales[side];
        const float bias = biases[side];
        for (int r = 0; r < R_TILE; ++r) {
          if (row0 + r < M) {
            acc[r][o] += qdot<float, VP, 8, false>(codes, values[r], scale, bias, sums[r]);
          }
        }
      }
    }
  }
}

for (int r = 0; r < R_TILE; ++r) {
  const int row = row0 + r;
  if (row < M) {
    const size_t out_row = (size_t(row) * size_t(G) + size_t(g)) * size_t(N);
    for (int o = 0; o < RPS; ++o) {
      const int n = first + o;
      if (FAST || n < N) {
        const float val = simd_sum(acc[r][o]);
        if (lane == 0) {
          out[out_row + size_t(n)] = val;
        }
      }
    }
  }
}
)METAL"
