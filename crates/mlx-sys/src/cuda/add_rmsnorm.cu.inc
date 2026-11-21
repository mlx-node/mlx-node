R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ float fused_to_f32(T v);
template <>
__device__ __forceinline__ float fused_to_f32<nv_bfloat16>(nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float fused_to_f32<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float fused_to_f32<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T fused_from_f32(float v);
template <>
__device__ __forceinline__ nv_bfloat16 fused_from_f32<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ __half fused_from_f32<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ float fused_from_f32<float>(float v) {
  return v;
}

__device__ __forceinline__ float fused_warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

// Fused residual-add + RMSNorm:
//   h = T(float(x) + float(res))
//   normed = T(float(w) * float(T(float(h) * rsqrtf(mean_sq + eps))))
// Matches MLX CUDA rms_norm.cu per-op rounding.
template <typename T, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void mlx_fused_add_rmsnorm_cuda_kernel(
    const T* __restrict__ x,
    const T* __restrict__ res,
    const T* __restrict__ w,
    const float* __restrict__ eps_ptr,
    T* __restrict__ h_out,
    T* __restrict__ normed_out,
    int rows, int axis) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const size_t base = (size_t)row * axis;
  const T* xr = x + base;
  const T* rr = res + base;
  T* hr = h_out + base;
  T* nr = normed_out + base;

  float sq = 0.0f;
  const bool vec8 =
      (sizeof(T) == 2) && ((axis & 7) == 0) &&
      (((reinterpret_cast<unsigned long long>(xr) |
         reinterpret_cast<unsigned long long>(rr) |
         reinterpret_cast<unsigned long long>(w) |
         reinterpret_cast<unsigned long long>(hr) |
         reinterpret_cast<unsigned long long>(nr)) & 15ull) == 0);

  if (vec8) {
    const int nvec = axis >> 3;
    const uint4* xv = reinterpret_cast<const uint4*>(xr);
    const uint4* rv = reinterpret_cast<const uint4*>(rr);
    uint4* hv = reinterpret_cast<uint4*>(hr);
    uint4 h0, h1;
    const bool use_reg2 = (nvec <= 2 * BLOCK_SIZE);
    if (use_reg2) {
      if (tid < nvec) {
        const uint4 xa = xv[tid];
        const uint4 ra = rv[tid];
        const T* x_e = reinterpret_cast<const T*>(&xa);
        const T* r_e = reinterpret_cast<const T*>(&ra);
        T* h_e = reinterpret_cast<T*>(&h0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const T hv_j = fused_from_f32<T>(fused_to_f32<T>(x_e[j]) + fused_to_f32<T>(r_e[j]));
          h_e[j] = hv_j;
          const float hf = fused_to_f32<T>(hv_j);
          sq += hf * hf;
        }
        hv[tid] = h0;
      }
      if (tid + BLOCK_SIZE < nvec) {
        const int i1 = tid + BLOCK_SIZE;
        const uint4 xa = xv[i1];
        const uint4 ra = rv[i1];
        const T* x_e = reinterpret_cast<const T*>(&xa);
        const T* r_e = reinterpret_cast<const T*>(&ra);
        T* h_e = reinterpret_cast<T*>(&h1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const T hv_j = fused_from_f32<T>(fused_to_f32<T>(x_e[j]) + fused_to_f32<T>(r_e[j]));
          h_e[j] = hv_j;
          const float hf = fused_to_f32<T>(hv_j);
          sq += hf * hf;
        }
        hv[i1] = h1;
      }
    } else {
      for (int i = tid; i < nvec; i += BLOCK_SIZE) {
        const uint4 xa = xv[i];
        const uint4 ra = rv[i];
        uint4 ha;
        const T* x_e = reinterpret_cast<const T*>(&xa);
        const T* r_e = reinterpret_cast<const T*>(&ra);
        T* h_e = reinterpret_cast<T*>(&ha);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const T hv_j = fused_from_f32<T>(fused_to_f32<T>(x_e[j]) + fused_to_f32<T>(r_e[j]));
          h_e[j] = hv_j;
          const float hf = fused_to_f32<T>(hv_j);
          sq += hf * hf;
        }
        hv[i] = ha;
      }
    }

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float smem[NUM_WARPS];
    sq = fused_warp_sum(sq);
    if (lane == 0) {
      smem[warp] = sq;
    }
    __syncthreads();
    float total = 0.0f;
    if (warp == 0) {
      total = lane < NUM_WARPS ? smem[lane] : 0.0f;
      total = fused_warp_sum(total);
      if (lane == 0) {
        smem[0] = rsqrtf(total / static_cast<float>(axis) + eps_ptr[0]);
      }
    }
    __syncthreads();
    const float inv = smem[0];

    const uint4* wv = reinterpret_cast<const uint4*>(w);
    uint4* nv = reinterpret_cast<uint4*>(nr);
    if (use_reg2) {
      if (tid < nvec) {
        const uint4 wa = wv[tid];
        uint4 na;
        const T* h_e = reinterpret_cast<const T*>(&h0);
        const T* w_e = reinterpret_cast<const T*>(&wa);
        T* n_e = reinterpret_cast<T*>(&na);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const T scaled = fused_from_f32<T>(fused_to_f32<T>(h_e[j]) * inv);
          n_e[j] = fused_from_f32<T>(fused_to_f32<T>(w_e[j]) * fused_to_f32<T>(scaled));
        }
        nv[tid] = na;
      }
      if (tid + BLOCK_SIZE < nvec) {
        const int i1 = tid + BLOCK_SIZE;
        const uint4 wa = wv[i1];
        uint4 na;
        const T* h_e = reinterpret_cast<const T*>(&h1);
        const T* w_e = reinterpret_cast<const T*>(&wa);
        T* n_e = reinterpret_cast<T*>(&na);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const T scaled = fused_from_f32<T>(fused_to_f32<T>(h_e[j]) * inv);
          n_e[j] = fused_from_f32<T>(fused_to_f32<T>(w_e[j]) * fused_to_f32<T>(scaled));
        }
        nv[i1] = na;
      }
    } else {
      for (int i = tid; i < nvec; i += BLOCK_SIZE) {
        const uint4 ha = hv[i];
        const uint4 wa = wv[i];
        uint4 na;
        const T* h_e = reinterpret_cast<const T*>(&ha);
        const T* w_e = reinterpret_cast<const T*>(&wa);
        T* n_e = reinterpret_cast<T*>(&na);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const T scaled = fused_from_f32<T>(fused_to_f32<T>(h_e[j]) * inv);
          n_e[j] = fused_from_f32<T>(fused_to_f32<T>(w_e[j]) * fused_to_f32<T>(scaled));
        }
        nv[i] = na;
      }
    }
    return;
  }

  for (int i = tid; i < axis; i += BLOCK_SIZE) {
    const T hv_i = fused_from_f32<T>(fused_to_f32<T>(xr[i]) + fused_to_f32<T>(rr[i]));
    hr[i] = hv_i;
    const float hf = fused_to_f32<T>(hv_i);
    sq += hf * hf;
  }

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem[NUM_WARPS];
  sq = fused_warp_sum(sq);
  if (lane == 0) {
    smem[warp] = sq;
  }
  __syncthreads();
  float total = 0.0f;
  if (warp == 0) {
    total = lane < NUM_WARPS ? smem[lane] : 0.0f;
    total = fused_warp_sum(total);
    if (lane == 0) {
      smem[0] = rsqrtf(total / static_cast<float>(axis) + eps_ptr[0]);
    }
  }
  __syncthreads();
  const float inv = smem[0];

  for (int i = tid; i < axis; i += BLOCK_SIZE) {
    const T scaled = fused_from_f32<T>(fused_to_f32<T>(hr[i]) * inv);
    nr[i] = fused_from_f32<T>(fused_to_f32<T>(w[i]) * fused_to_f32<T>(scaled));
  }
}

template <typename T>
__device__ __forceinline__ T fused_sigmoid(T x) {
  const float f = fused_to_f32<T>(x);
  const float y = 1.0f / (1.0f + expf(fabsf(f)));
  return fused_from_f32<T>(f < 0.0f ? y : (1.0f - y));
}

// Fused SwiGLU: out = T(float(T(float(gate) * float(sigmoid(gate)))) * float(up))
// Exact per-op rounding matching MLX's unfused sigmoid + mul + mul chain.
template <typename T>
__global__ __launch_bounds__(256) void mlx_swiglu_cuda_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ out,
    long long n) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const bool vec8 =
      (sizeof(T) == 2) && ((n & 7) == 0) &&
      (((reinterpret_cast<unsigned long long>(gate) |
         reinterpret_cast<unsigned long long>(up) |
         reinterpret_cast<unsigned long long>(out)) & 15ull) == 0);
  if (vec8) {
    const long long nvec = n >> 3;
    if (idx >= nvec) {
      return;
    }
    const uint4 gv = reinterpret_cast<const uint4*>(gate)[idx];
    const uint4 uv = reinterpret_cast<const uint4*>(up)[idx];
    uint4 ov;
    const T* ge = reinterpret_cast<const T*>(&gv);
    const T* ue = reinterpret_cast<const T*>(&uv);
    T* oe = reinterpret_cast<T*>(&ov);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const T sig = fused_sigmoid<T>(ge[j]);
      const T silu = fused_from_f32<T>(fused_to_f32<T>(ge[j]) * fused_to_f32<T>(sig));
      oe[j] = fused_from_f32<T>(fused_to_f32<T>(silu) * fused_to_f32<T>(ue[j]));
    }
    reinterpret_cast<uint4*>(out)[idx] = ov;
    return;
  }
  if (idx >= n) {
    return;
  }
  const T g = gate[idx];
  const T sig = fused_sigmoid<T>(g);
  const T silu = fused_from_f32<T>(fused_to_f32<T>(g) * fused_to_f32<T>(sig));
  out[idx] = fused_from_f32<T>(fused_to_f32<T>(silu) * fused_to_f32<T>(up[idx]));
}

// Fused SigmoidMul: out = T(float(value) * float(sigmoid(gate)))
template <typename T>
__global__ __launch_bounds__(256) void mlx_sigmoid_mul_cuda_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ value,
    T* __restrict__ out,
    long long n) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const bool vec8 =
      (sizeof(T) == 2) && ((n & 7) == 0) &&
      (((reinterpret_cast<unsigned long long>(gate) |
         reinterpret_cast<unsigned long long>(value) |
         reinterpret_cast<unsigned long long>(out)) & 15ull) == 0);
  if (vec8) {
    const long long nvec = n >> 3;
    if (idx >= nvec) {
      return;
    }
    const uint4 gv = reinterpret_cast<const uint4*>(gate)[idx];
    const uint4 vv = reinterpret_cast<const uint4*>(value)[idx];
    uint4 ov;
    const T* ge = reinterpret_cast<const T*>(&gv);
    const T* ve = reinterpret_cast<const T*>(&vv);
    T* oe = reinterpret_cast<T*>(&ov);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const T sig = fused_sigmoid<T>(ge[j]);
      oe[j] = fused_from_f32<T>(fused_to_f32<T>(ve[j]) * fused_to_f32<T>(sig));
    }
    reinterpret_cast<uint4*>(out)[idx] = ov;
    return;
  }
  if (idx >= n) {
    return;
  }
  const T sig = fused_sigmoid<T>(gate[idx]);
  out[idx] = fused_from_f32<T>(fused_to_f32<T>(value[idx]) * fused_to_f32<T>(sig));
}

// 2D-strided SwiGLU / SigmoidMul (last dim contiguous, arbitrary row stride).
// Avoids contiguous_copy_gpu on split outputs of merged gate_up / q_gate GEMMs.
template <typename T>
__global__ __launch_bounds__(256) void mlx_swiglu_strided_cuda_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ out,
    long long rows,
    long long cols,
    long long g_rs,
    long long u_rs) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const bool vec8 =
      (sizeof(T) == 2) && ((cols & 7) == 0) && ((g_rs & 7) == 0) && ((u_rs & 7) == 0) &&
      (((reinterpret_cast<unsigned long long>(gate) |
         reinterpret_cast<unsigned long long>(up) |
         reinterpret_cast<unsigned long long>(out)) & 15ull) == 0);
  if (vec8) {
    const long long nvec_row = cols >> 3;
    const long long total_vec = rows * nvec_row;
    if (idx >= total_vec) {
      return;
    }
    const long long r = idx / nvec_row;
    const long long c8 = idx % nvec_row;
    const uint4 gv = *reinterpret_cast<const uint4*>(gate + r * g_rs + (c8 << 3));
    const uint4 uv = *reinterpret_cast<const uint4*>(up + r * u_rs + (c8 << 3));
    uint4 ov;
    const T* ge = reinterpret_cast<const T*>(&gv);
    const T* ue = reinterpret_cast<const T*>(&uv);
    T* oe = reinterpret_cast<T*>(&ov);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const T sig = fused_sigmoid<T>(ge[j]);
      const T silu = fused_from_f32<T>(fused_to_f32<T>(ge[j]) * fused_to_f32<T>(sig));
      oe[j] = fused_from_f32<T>(fused_to_f32<T>(silu) * fused_to_f32<T>(ue[j]));
    }
    *reinterpret_cast<uint4*>(out + r * cols + (c8 << 3)) = ov;
    return;
  }
  const long long total = rows * cols;
  if (idx >= total) {
    return;
  }
  const long long r = idx / cols;
  const long long c = idx % cols;
  const T g = gate[r * g_rs + c];
  const T sig = fused_sigmoid<T>(g);
  const T silu = fused_from_f32<T>(fused_to_f32<T>(g) * fused_to_f32<T>(sig));
  out[idx] = fused_from_f32<T>(fused_to_f32<T>(silu) * fused_to_f32<T>(up[r * u_rs + c]));
}

template <typename T>
__global__ __launch_bounds__(256) void mlx_sigmoid_mul_strided_cuda_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ value,
    T* __restrict__ out,
    long long rows,
    long long cols,
    long long g_rs,
    long long v_rs) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const bool vec8 =
      (sizeof(T) == 2) && ((cols & 7) == 0) && ((g_rs & 7) == 0) && ((v_rs & 7) == 0) &&
      (((reinterpret_cast<unsigned long long>(gate) |
         reinterpret_cast<unsigned long long>(value) |
         reinterpret_cast<unsigned long long>(out)) & 15ull) == 0);
  if (vec8) {
    const long long nvec_row = cols >> 3;
    const long long total_vec = rows * nvec_row;
    if (idx >= total_vec) {
      return;
    }
    const long long r = idx / nvec_row;
    const long long c8 = idx % nvec_row;
    const uint4 gv = *reinterpret_cast<const uint4*>(gate + r * g_rs + (c8 << 3));
    const uint4 vv = *reinterpret_cast<const uint4*>(value + r * v_rs + (c8 << 3));
    uint4 ov;
    const T* ge = reinterpret_cast<const T*>(&gv);
    const T* ve = reinterpret_cast<const T*>(&vv);
    T* oe = reinterpret_cast<T*>(&ov);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const T sig = fused_sigmoid<T>(ge[j]);
      oe[j] = fused_from_f32<T>(fused_to_f32<T>(ve[j]) * fused_to_f32<T>(sig));
    }
    *reinterpret_cast<uint4*>(out + r * cols + (c8 << 3)) = ov;
    return;
  }
  const long long total = rows * cols;
  if (idx >= total) {
    return;
  }
  const long long r = idx / cols;
  const long long c = idx % cols;
  const T sig = fused_sigmoid<T>(gate[r * g_rs + c]);
  out[idx] = fused_from_f32<T>(fused_to_f32<T>(value[r * v_rs + c]) * fused_to_f32<T>(sig));
}
)cuda"
