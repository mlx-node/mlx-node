R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ float gdn_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ nv_bfloat16 gdn_bf16_mul(nv_bfloat16 a, nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}

extern "C" __global__ __launch_bounds__(1024, 1) void mlx_gdn_fused_step_cuda_kernel(
    const nv_bfloat16* q, const nv_bfloat16* k, const nv_bfloat16* v,
    const float* g, const nv_bfloat16* beta, const float* state,
    const nv_bfloat16* z, const nv_bfloat16* w, nv_bfloat16* out,
    float* state_out, int T, int Hk, int Hv, long long z_stride_b,
    long long z_stride_t, float eps, int tiled_heads) {
  constexpr int D = 128;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int col = warp * 4 + lane / 8;
  const int lane_in_col = lane % 8;
  const int hk = tiled_heads ? h % Hk : h / (Hv / Hk);
  const long long state_offset = ((long long)b * Hv + h) * D * D + col * D;
  float s[16];
  for (int r = 0; r < 16; ++r) {
    s[r] = state[state_offset + r * 8 + lane_in_col];
  }

  __shared__ nv_bfloat16 y_shared[32 * D];
  __shared__ float norm_parts[16][2];
  __shared__ float norm_totals[16];
  for (int t = 0; t < T; ++t) {
    const long long qk_offset = ((long long)b * T + t) * Hk * D + hk * D;
    const long long input_offset = ((long long)b * T + t) * Hv * D + h * D;
    const long long gate_offset = ((long long)b * T + t) * Hv + h;
    float k_reg[16];
    float q_reg[16];
    float kv = 0.0f;
    for (int r = 0; r < 16; ++r) {
      const int i = r * 8 + lane_in_col;
      k_reg[r] = __bfloat162float(k[qk_offset + i]);
      q_reg[r] = __bfloat162float(q[qk_offset + i]);
      kv += s[r] * k_reg[r];
    }
    for (int shift = 4; shift > 0; shift >>= 1) {
      kv += __shfl_xor_sync(0xffffffffu, kv, shift, 8);
    }
    const float decay = g[gate_offset];
    const float delta = (__bfloat162float(v[input_offset + col]) - decay * kv) *
        __bfloat162float(beta[gate_offset]);
    float y = 0.0f;
    for (int r = 0; r < 16; ++r) {
      s[r] = decay * s[r] + k_reg[r] * delta;
      y += s[r] * q_reg[r];
    }
    for (int shift = 4; shift > 0; shift >>= 1) {
      y += __shfl_xor_sync(0xffffffffu, y, shift, 8);
    }
    if (lane_in_col == 0) {
      y_shared[t * D + col] = __float2bfloat16(y);
    }
  }
  for (int r = 0; r < 16; ++r) {
    state_out[state_offset + r * 8 + lane_in_col] = s[r];
  }
  __syncthreads();

  // Match CUDA rms_norm_small: a 64-thread block covers 16 eight-value
  // vectors; inactive lanes contribute zero to its two-warp XOR reduction.
  // Loop uniformly in multiples of 16 so every thread hits every __syncthreads().
  const int group = tid / 64;
  const int sub = tid % 64;
  const int sub_lane = sub % 32;
  const int sub_warp = sub / 32;
  for (int t_base = 0; t_base < T; t_base += 16) {
    const int t = t_base + group;
    const bool active = t < T;
    float values[8];
    float acc = 0.0f;
    if (active && sub < D / 8) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        values[i] = __bfloat162float(y_shared[t * D + sub * 8 + i]);
        acc += values[i] * values[i];
      }
    }
    acc = gdn_warp_sum(acc);
    if (sub_lane == 0) {
      norm_parts[group][sub_warp] = acc;
    }
    __syncthreads();
    float total = sub_lane < 2 ? norm_parts[group][sub_lane] : 0.0f;
    total = gdn_warp_sum(total);
    if (sub_lane == 0) {
      norm_totals[group] = total;
    }
    __syncthreads();
    if (active && sub < D / 8) {
      const float inv = rsqrtf(norm_totals[group] / D + eps);
      const long long z_offset = (long long)b * z_stride_b +
          (long long)t * z_stride_t + h * D + sub * 8;
      const long long out_offset =
          ((long long)b * T + t) * Hv * D + h * D + sub * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const nv_bfloat16 scaled = __float2bfloat16(values[i] * inv);
        const nv_bfloat16 normed = gdn_bf16_mul(w[sub * 8 + i], scaled);
        const nv_bfloat16 zt = z[z_offset + i];
        const nv_bfloat16 one = __float2bfloat16(1.0f);
        const nv_bfloat16 sigmoid_y = one / (one + hexp(__habs(zt)));
        const nv_bfloat16 sig = zt < __float2bfloat16(0.0f)
            ? sigmoid_y
            : one - sigmoid_y;
        const nv_bfloat16 gated = gdn_bf16_mul(sig, zt);
        out[out_offset + i] = gdn_bf16_mul(gated, normed);
      }
    }
    __syncthreads();
  }
}

extern "C" __global__ __launch_bounds__(1024, 1) void mlx_gdn_replay_cuda_kernel(
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    const float* __restrict__ g,
    const nv_bfloat16* __restrict__ beta,
    const float* __restrict__ state,
    float* __restrict__ state_out,
    int accepted_steps, int window_stride, int Hk, int Hv, int tiled_heads) {
  constexpr int D = 128;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int col = warp * 4 + lane / 8;
  const int lane_in_col = lane % 8;
  const int hk = tiled_heads ? h % Hk : h / (Hv / Hk);
  const long long state_offset = ((long long)b * Hv + h) * D * D + col * D;
  float s[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    s[r] = state[state_offset + r * 8 + lane_in_col];
  }

  for (int t = 0; t < accepted_steps; ++t) {
    const long long k_offset = ((long long)b * window_stride + t) * Hk * D + hk * D;
    const long long v_offset = ((long long)b * window_stride + t) * Hv * D + h * D;
    const long long gate_offset = ((long long)b * window_stride + t) * Hv + h;
    float k_reg[16];
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int i = r * 8 + lane_in_col;
      k_reg[r] = __bfloat162float(k[k_offset + i]);
      kv += s[r] * k_reg[r];
    }
    for (int shift = 4; shift > 0; shift >>= 1) {
      kv += __shfl_xor_sync(0xffffffffu, kv, shift, 8);
    }
    const float decay = g[gate_offset];
    const float delta = (__bfloat162float(v[v_offset + col]) - decay * kv) *
        __bfloat162float(beta[gate_offset]);
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      s[r] = decay * s[r] + k_reg[r] * delta;
    }
  }
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    state_out[state_offset + r * 8 + lane_in_col] = s[r];
  }
}
)cuda"
