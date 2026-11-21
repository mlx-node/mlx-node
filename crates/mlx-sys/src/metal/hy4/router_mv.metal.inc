R"METAL(
constexpr int NUM_SG = 4;
constexpr int EXPERTS_PER_SG = 2;
constexpr int EXPERTS_PER_TG = NUM_SG * EXPERTS_PER_SG;

const int M = x_shape[0];
const int K = x_shape[1];
const int E = w_shape[0];

const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int n_etiles = (E + EXPERTS_PER_TG - 1) / EXPERTS_PER_TG;
const int z = int(threadgroup_position_in_grid.z);
const int e_tile = z % n_etiles;
const int m_tile = z / n_etiles;

const int e0 = e_tile * EXPERTS_PER_TG + int(sg) * EXPERTS_PER_SG;
const int row0 = m_tile * M_TILE;
if (e0 >= E || row0 >= M) return;

const int e1 = metal::min(e0 + 1, E - 1);
const int n_f4 = K >> 2;

const device float4* w4_0 =
    reinterpret_cast<const device float4*>(w) + size_t(e0) * size_t(n_f4);
const device float4* w4_1 =
    reinterpret_cast<const device float4*>(w) + size_t(e1) * size_t(n_f4);
const device float4* x_base = reinterpret_cast<const device float4*>(x);

size_t x_off[M_TILE];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const int mr = metal::min(row0 + r, M - 1);
  x_off[r] = size_t(mr) * size_t(n_f4);
}

float acc[M_TILE][EXPERTS_PER_SG];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  acc[r][0] = 0.0f;
  acc[r][1] = 0.0f;
}

const int k_main_f4 = FAST_K ? n_f4 : (n_f4 & ~31);
for (int base_f4 = 0; base_f4 < k_main_f4; base_f4 += 32) {
  const int idx = base_f4 + int(lane);
  const float4 w0 = w4_0[idx];
  const float4 w1 = w4_1[idx];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    const float4 xv = x_base[x_off[r] + size_t(idx)];
    acc[r][0] += metal::dot(xv, w0);
    acc[r][1] += metal::dot(xv, w1);
  }
}

if constexpr (!FAST_K) {
  const int idx = k_main_f4 + int(lane);
  if (idx < n_f4) {
    const float4 w0 = w4_0[idx];
    const float4 w1 = w4_1[idx];
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const float4 xv = x_base[x_off[r] + size_t(idx)];
      acc[r][0] += metal::dot(xv, w0);
      acc[r][1] += metal::dot(xv, w1);
    }
  }
}

#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const float v0 = simd_sum(acc[r][0]);
  const float v1 = simd_sum(acc[r][1]);
  const int mr = row0 + r;
  if (lane == 0 && mr < M) {
    const size_t obase = size_t(mr) * size_t(E);
    out[obase + size_t(e0)] = v0;
    if (e0 + 1 < E) {
      out[obase + size_t(e0 + 1)] = v1;
    }
  }
}
)METAL"
