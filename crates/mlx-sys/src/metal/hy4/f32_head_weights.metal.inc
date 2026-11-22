R"METAL(
constexpr int NUM_SG = 4;
constexpr int TG_THREADS = NUM_SG * 32;

const int M = x_shape[0];
const int K = x_shape[1];
const int H = w_shape[0];

const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int h = int(threadgroup_position_in_grid.y);
const int m_tile = int(threadgroup_position_in_grid.z);
const int row0 = m_tile * M_TILE;
if (h >= H || row0 >= M) {
  return;
}

const int k_f4 = K >> 2;
const device float4* x4 = reinterpret_cast<const device float4*>(x);
const device float4* w4 = reinterpret_cast<const device float4*>(w);
const size_t w_row = size_t(h) * size_t(k_f4);

size_t x_row[M_TILE];
float acc[M_TILE];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const int mr = metal::min(row0 + r, M - 1);
  x_row[r] = size_t(mr) * size_t(k_f4);
  acc[r] = 0.0f;
}

for (int k4 = int(tid); k4 < k_f4; k4 += TG_THREADS) {
  const float4 wv = w4[w_row + size_t(k4)];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    acc[r] += metal::dot(x4[x_row[r] + size_t(k4)], wv);
  }
}

threadgroup float tg_red[M_TILE * NUM_SG];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const float s = simd_sum(acc[r]);
  if (lane == 0u) {
    tg_red[r * NUM_SG + int(sg)] = s;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < uint(M_TILE)) {
  const int r = int(tid);
  const int mr = row0 + r;
  if (mr < M) {
    const int base = r * NUM_SG;
    const float total =
        (tg_red[base] + tg_red[base + 1]) + (tg_red[base + 2] + tg_red[base + 3]);
    out[size_t(mr) * size_t(H) + size_t(h)] = total * scale[0];
  }
}
)METAL"
