R"METAL(
constexpr int NUM_SG = 4;
constexpr int RPS = 2;
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

const int tg_e0 = e_tile * EXPERTS_PER_TG;
const int row0 = m_tile * M_TILE;
// Threadgroup-uniform early exit so all 4 simdgroups hit barriers together.
if (tg_e0 >= E || row0 >= M) return;

const int e0 = tg_e0 + int(sg) * RPS;
const int n_f4 = K >> 2;
const device float4* x_base = reinterpret_cast<const device float4*>(x);

size_t w_off[RPS];
#pragma unroll
for (int p = 0; p < RPS; ++p) {
  const int ep = metal::min(e0 + p, E - 1);
  w_off[p] = size_t(ep) * size_t(n_f4);
}

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

const int k_main_f4 = FAST_K ? n_f4 : (n_f4 & ~31);

if constexpr (USE_TG_X) {
  // Planar [M_TILE][4][32] layout avoids 4-way float4 bank conflicts.
  threadgroup float tg_x[M_TILE][4][32];
  for (int base_f4 = 0; base_f4 < k_main_f4; base_f4 += 32) {
    if (base_f4 > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = int(sg); r < M_TILE; r += NUM_SG) {
      const float4 xv = x_base[size_t(x_off[r]) + size_t(base_f4 + int(lane))];
      tg_x[r][0][lane] = xv.x;
      tg_x[r][1][lane] = xv.y;
      tg_x[r][2][lane] = xv.z;
      tg_x[r][3][lane] = xv.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t idx = size_t(base_f4 + int(lane));
#pragma unroll
    for (int p = 0; p < RPS; ++p) {
      float4 wv;
      if constexpr (W_F16) {
        wv = float4(reinterpret_cast<const device half4*>(w)[w_off[p] + idx]);
      } else {
        wv = reinterpret_cast<const device float4*>(w)[w_off[p] + idx];
      }
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const float4 xv = float4(
            tg_x[r][0][lane], tg_x[r][1][lane],
            tg_x[r][2][lane], tg_x[r][3][lane]);
        acc[r][p] += metal::dot(xv, wv);
      }
    }
  }
} else {
  for (int base_f4 = 0; base_f4 < k_main_f4; base_f4 += 32) {
    const size_t idx = size_t(base_f4 + int(lane));
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const float4 xv = x_base[size_t(x_off[r]) + idx];
#pragma unroll
      for (int p = 0; p < RPS; ++p) {
        float4 wv;
        if constexpr (W_F16) {
          wv = float4(reinterpret_cast<const device half4*>(w)[w_off[p] + idx]);
        } else {
          wv = reinterpret_cast<const device float4*>(w)[w_off[p] + idx];
        }
        acc[r][p] += metal::dot(xv, wv);
      }
    }
  }
}

if constexpr (!FAST_K) {
  const int idx_i = k_main_f4 + int(lane);
  if (idx_i < n_f4) {
    const size_t idx = size_t(idx_i);
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const float4 xv = x_base[size_t(x_off[r]) + idx];
#pragma unroll
      for (int p = 0; p < RPS; ++p) {
        float4 wv;
        if constexpr (W_F16) {
          wv = float4(reinterpret_cast<const device half4*>(w)[w_off[p] + idx]);
        } else {
          wv = reinterpret_cast<const device float4*>(w)[w_off[p] + idx];
        }
        acc[r][p] += metal::dot(xv, wv);
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