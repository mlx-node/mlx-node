R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ float gdn_prep_warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ nv_bfloat16 gdn_prep_bf16_silu(nv_bfloat16 raw) {
  const float f = __bfloat162float(raw);
  const float e = expf(fabsf(f));
  const float y = 1.0f / (1.0f + e);
  const float sig = (f < 0.0f) ? y : (1.0f - y);
  const nv_bfloat16 sig_bf = __float2bfloat16(sig);
  return __float2bfloat16(__bfloat162float(raw) * __bfloat162float(sig_bf));
}

__device__ __forceinline__ float gdn_prep_precise_sigmoid_f32(float x) {
  const float y = 1.0f / (1.0f + expf(fabsf(x)));
  return (x < 0.0f) ? y : (1.0f - y);
}

// Fused Qwen3.5/3.8 GDN preparation: 4-tap depthwise conv1d + SiLU + q|k|v split
// + q/k RMS/L2 norm+scale + decay & beta gating + next conv history.
// Geometry: W=10240 (16*128 q + 16*128 k + 48*128 v = 80 vectors of 128 channels),
// B=1, T in 1..=1024, K=4.
// Grid: dim3(20, T, 1), Block: dim3(32, 4, 1) -> 80 warps cover vec=0..79.
template <bool A_BF16, bool B_BF16, bool BETA_BF16>
__global__ __launch_bounds__(128, 4) void mlx_gdn_prepare_cuda_kernel(
    const nv_bfloat16* __restrict__ qkv,
    const void* __restrict__ a_ptr,
    const void* __restrict__ b_ptr,
    const float* __restrict__ conv,
    const nv_bfloat16* __restrict__ history,
    const float* __restrict__ scale,
    const float* __restrict__ dt,
    nv_bfloat16* __restrict__ q_out,
    nv_bfloat16* __restrict__ k_out,
    nv_bfloat16* __restrict__ v_out,
    float* __restrict__ decay_out,
    void* __restrict__ beta_out,
    nv_bfloat16* __restrict__ next_history,
    int T,
    long long qkv_s1, long long qkv_s2,
    long long a_s1, long long a_s2,
    long long b_s1, long long b_s2,
    long long hist_s0, long long hist_s1,
    int mean_eps) {
  constexpr int W = 10240;
  const int lane = threadIdx.x;
  const int vec = blockIdx.x * 4 + threadIdx.y; // 0..79
  const int token = blockIdx.y;                 // 0..T-1
  if (vec >= 80 || token >= T) {
    return;
  }

  float values[4];
  float squared = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int ch = vec * 128 + lane * 4 + i;
    float total = 0.0f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      const int row = token + tap;
      const nv_bfloat16 src = (row < 3)
          ? history[row * hist_s0 + ch * hist_s1]
          : qkv[(row - 3) * qkv_s1 + ch * qkv_s2];
      total += __bfloat162float(src) * conv[ch * 4 + tap];
    }
    const nv_bfloat16 raw = __float2bfloat16(total);
    const nv_bfloat16 activated = gdn_prep_bf16_silu(raw);
    const float vf = __bfloat162float(activated);
    values[i] = vf;
    squared += vf * vf;

    if (token == 0) {
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        const int source = T + row;
        next_history[row * W + ch] = (source < 3)
            ? history[source * hist_s0 + ch * hist_s1]
            : qkv[(source - 3) * qkv_s1 + ch * qkv_s2];
      }
    }
  }

  if (vec < 32) {
    const float sum_sq = gdn_prep_warp_sum(squared);
    const float eps_term = mean_eps ? (128.0f * 1e-6f) : 1e-6f;
    const float divisor = sqrtf(sum_sq + eps_term);
    const nv_bfloat16 q_scale = __float2bfloat16(0.08838834764831845f);
    const float q_scale_f = __bfloat162float(q_scale);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int ch = (vec & 15) * 128 + lane * 4 + i;
      const nv_bfloat16 normalized = __float2bfloat16(values[i] / divisor);
      if (vec < 16) {
        q_out[token * 2048 + ch] = __float2bfloat16(__bfloat162float(normalized) * q_scale_f);
      } else {
        k_out[token * 2048 + ch] = normalized;
      }
    }
  } else {
    const int base = token * 6144 + (vec - 32) * 128 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v_out[base + i] = __float2bfloat16(values[i]);
    }
  }

  if (vec < 48 && lane == 0) {
    const int h = token * 48 + vec;
    const float a_val = A_BF16
        ? __bfloat162float(reinterpret_cast<const nv_bfloat16*>(a_ptr)[token * a_s1 + vec * a_s2])
        : reinterpret_cast<const float*>(a_ptr)[token * a_s1 + vec * a_s2];
    const float b_val = B_BF16
        ? __bfloat162float(reinterpret_cast<const nv_bfloat16*>(b_ptr)[token * b_s1 + vec * b_s2])
        : reinterpret_cast<const float*>(b_ptr)[token * b_s1 + vec * b_s2];
    const float ax = a_val + dt[vec];
    const float sp = fmaxf(ax, 0.0f) + log1pf(expf(-fabsf(ax)));
    decay_out[h] = expf(sp * scale[vec]);
    const float sig_b = gdn_prep_precise_sigmoid_f32(b_val);
    if (BETA_BF16) {
      reinterpret_cast<nv_bfloat16*>(beta_out)[h] = __float2bfloat16(sig_b);
    } else {
      reinterpret_cast<float*>(beta_out)[h] = sig_b;
    }
  }
}
)cuda"
R"cuda(
// Tiled prefill GDN prepare (TILE_T = 32): 3-token sliding register window cuts
// qkv DRAM reads 4x, conv weight reads 32x, and uses coalesced uint2 loads/stores!
// 100% bit-exact with mlx_gdn_prepare_cuda_kernel.
// Grid: dim3(20, (T + 31) / 32, 1), Block: dim3(32, 4, 1).
template <bool A_BF16, bool B_BF16, bool BETA_BF16>
__global__ __launch_bounds__(128, 4) void mlx_gdn_prepare_tiled_cuda_kernel(
    const nv_bfloat16* __restrict__ qkv,
    const void* __restrict__ a_ptr,
    const void* __restrict__ b_ptr,
    const float* __restrict__ conv,
    const nv_bfloat16* __restrict__ history,
    const float* __restrict__ scale,
    const float* __restrict__ dt,
    nv_bfloat16* __restrict__ q_out,
    nv_bfloat16* __restrict__ k_out,
    nv_bfloat16* __restrict__ v_out,
    float* __restrict__ decay_out,
    void* __restrict__ beta_out,
    nv_bfloat16* __restrict__ next_history,
    int T,
    long long qkv_s1, long long qkv_s2,
    long long a_s1, long long a_s2,
    long long b_s1, long long b_s2,
    long long hist_s0, long long hist_s1,
    int mean_eps) {
  constexpr int W = 10240;
  constexpr int TILE_T = 32;
  const int lane = threadIdx.x;
  const int vec = blockIdx.x * 4 + threadIdx.y; // 0..79
  const int t0 = blockIdx.y * TILE_T;
  if (vec >= 80 || t0 >= T) return;
  const int t_end = min(t0 + TILE_T, T);
  const int ch0 = vec * 128 + lane * 4;

  // Load 4x4 conv weights once per 32-token tile.
  float4 cw[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    cw[i] = *reinterpret_cast<const float4*>(conv + (ch0 + i) * 4);
  }

  const bool qkv_contig = (qkv_s2 == 1) &&
      (((reinterpret_cast<unsigned long long>(qkv) |
         static_cast<unsigned long long>(qkv_s1 * 2)) & 7ull) == 0ull);
  const bool hist_contig = (hist_s1 == 1) &&
      (((reinterpret_cast<unsigned long long>(history) |
         static_cast<unsigned long long>(hist_s0 * 2)) & 7ull) == 0ull);

  auto load_row_bf16 = [&](int row, nv_bfloat16 dst[4]) {
    if (row < 3) {
      if (hist_contig) {
        const uint2 raw = *reinterpret_cast<const uint2*>(history + row * hist_s0 + ch0);
        *reinterpret_cast<uint2*>(dst) = raw;
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = history[row * hist_s0 + (ch0 + i) * hist_s1];
      }
    } else {
      const long long qrow = row - 3;
      if (qkv_contig) {
        const uint2 raw = *reinterpret_cast<const uint2*>(qkv + qrow * qkv_s1 + ch0);
        *reinterpret_cast<uint2*>(dst) = raw;
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = qkv[qrow * qkv_s1 + (ch0 + i) * qkv_s2];
      }
    }
  };

  auto to_f4 = [](const nv_bfloat16 src[4]) -> float4 {
    return make_float4(
        __bfloat162float(src[0]), __bfloat162float(src[1]),
        __bfloat162float(src[2]), __bfloat162float(src[3]));
  };

  nv_bfloat16 b_prev[3][4];
  load_row_bf16(t0 + 0, b_prev[0]);
  load_row_bf16(t0 + 1, b_prev[1]);
  load_row_bf16(t0 + 2, b_prev[2]);
  float4 r0 = to_f4(b_prev[0]);
  float4 r1 = to_f4(b_prev[1]);
  float4 r2 = to_f4(b_prev[2]);

  const float eps_term = mean_eps ? (128.0f * 1e-6f) : 1e-6f;
  const float q_scale_f = __bfloat162float(__float2bfloat16(0.08838834764831845f));
  const float dt_v = (vec < 48 && lane == 0) ? dt[vec] : 0.0f;
  const float sc_v = (vec < 48 && lane == 0) ? scale[vec] : 0.0f;

  for (int token = t0; token < t_end; ++token) {
    nv_bfloat16 b_cur[4];
    load_row_bf16(token + 3, b_cur);
    const float4 r3 = to_f4(b_cur);

    float values[4];
    float squared = 0.0f;
    const float t_f[4] = {
        r0.x * cw[0].x + r1.x * cw[0].y + r2.x * cw[0].z + r3.x * cw[0].w,
        r0.y * cw[1].x + r1.y * cw[1].y + r2.y * cw[1].z + r3.y * cw[1].w,
        r0.z * cw[2].x + r1.z * cw[2].y + r2.z * cw[2].z + r3.z * cw[2].w,
        r0.w * cw[3].x + r1.w * cw[3].y + r2.w * cw[3].z + r3.w * cw[3].w,
    };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const nv_bfloat16 act = gdn_prep_bf16_silu(__float2bfloat16(t_f[i]));
      const float vf = __bfloat162float(act);
      values[i] = vf;
      squared += vf * vf;
    }

    if (vec < 32) {
      const float sum_sq = gdn_prep_warp_sum(squared);
      const float divisor = sqrtf(sum_sq + eps_term);
      nv_bfloat16 out4[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const nv_bfloat16 normalized = __float2bfloat16(values[i] / divisor);
        out4[i] = (vec < 16)
            ? __float2bfloat16(__bfloat162float(normalized) * q_scale_f)
            : normalized;
      }
      const int ch_out = (vec & 15) * 128 + lane * 4;
      if (vec < 16) {
        *reinterpret_cast<uint2*>(q_out + static_cast<size_t>(token) * 2048 + ch_out) =
            *reinterpret_cast<const uint2*>(out4);
      } else {
        *reinterpret_cast<uint2*>(k_out + static_cast<size_t>(token) * 2048 + ch_out) =
            *reinterpret_cast<const uint2*>(out4);
      }
    } else {
      nv_bfloat16 out4[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) out4[i] = __float2bfloat16(values[i]);
      const size_t base = static_cast<size_t>(token) * 6144 + (vec - 32) * 128 + lane * 4;
      *reinterpret_cast<uint2*>(v_out + base) = *reinterpret_cast<const uint2*>(out4);
    }

    if (vec < 48 && lane == 0) {
      const size_t h = static_cast<size_t>(token) * 48 + vec;
      const float a_val = A_BF16
          ? __bfloat162float(reinterpret_cast<const nv_bfloat16*>(a_ptr)[token * a_s1 + vec * a_s2])
          : reinterpret_cast<const float*>(a_ptr)[token * a_s1 + vec * a_s2];
      const float b_val = B_BF16
          ? __bfloat162float(reinterpret_cast<const nv_bfloat16*>(b_ptr)[token * b_s1 + vec * b_s2])
          : reinterpret_cast<const float*>(b_ptr)[token * b_s1 + vec * b_s2];
      const float ax = a_val + dt_v;
      const float sp = fmaxf(ax, 0.0f) + log1pf(expf(-fabsf(ax)));
      decay_out[h] = expf(sp * sc_v);
      const float sig_b = gdn_prep_precise_sigmoid_f32(b_val);
      if constexpr (BETA_BF16) {
        reinterpret_cast<nv_bfloat16*>(beta_out)[h] = __float2bfloat16(sig_b);
      } else {
        reinterpret_cast<float*>(beta_out)[h] = sig_b;
      }
    }

    r0 = r1;
    r1 = r2;
    r2 = r3;
    if (token == T - 1) {
      // Last tile writes next_history[0..2] = rows [T, T+1, T+2].
      nv_bfloat16 nh[4];
#pragma unroll
      for (int hr = 0; hr < 3; ++hr) {
        load_row_bf16(T + hr, nh);
        *reinterpret_cast<uint2*>(next_history + hr * W + ch0) =
            *reinterpret_cast<const uint2*>(nh);
      }
    }
  }
}
)cuda"
