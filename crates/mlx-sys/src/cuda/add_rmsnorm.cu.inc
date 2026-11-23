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

// Fused residual-add + RMSNorm (and standalone RMSNorm when HAS_RES == false):
//   HAS_RES == true:  h = T(float(x) + float(res)) -> written to h_out
//   HAS_RES == false: h = x (res and h_out are nullptr)
//   PRECISE_F32 == false:
//     normed = T(float(w_t) * float(T(float(h) * rsqrtf(mean_sq + eps))))
//     Matches MLX CUDA rms_norm.cu per-op rounding.
//   PRECISE_F32 == true:
//     w is float[axis]; normed = T((float(h) * rsqrtf(mean_sq + eps)) * w_f32)
//     No intermediate BF16/F16 rounding; bit-identical to
//     rms_norm(h.astype(f32), w_f32, eps).astype(T).
template <typename T, bool HAS_RES>
__device__ __forceinline__ uint4 fused_add_load_vec8(
    const uint4* __restrict__ xv,
    const uint4* __restrict__ rv,
    uint4* __restrict__ hv,
    int idx,
    float& sq) {
  const uint4 xa = xv[idx];
  uint4 ha;
  if constexpr (HAS_RES) {
    const uint4 ra = rv[idx];
    const T* x_e = reinterpret_cast<const T*>(&xa);
    const T* r_e = reinterpret_cast<const T*>(&ra);
    T* h_e = reinterpret_cast<T*>(&ha);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const T hv_j =
          fused_from_f32<T>(fused_to_f32<T>(x_e[j]) + fused_to_f32<T>(r_e[j]));
      h_e[j] = hv_j;
      const float hf = fused_to_f32<T>(hv_j);
      sq += hf * hf;
    }
    hv[idx] = ha;
  } else {
    ha = xa;
    const T* h_e = reinterpret_cast<const T*>(&ha);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float hf = fused_to_f32<T>(h_e[j]);
      sq += hf * hf;
    }
  }
  return ha;
}

template <typename T, bool PRECISE_F32>
__device__ __forceinline__ void fused_norm_store_vec8(
    uint4 ha,
    const void* __restrict__ w_raw,
    uint4* __restrict__ nv,
    int idx,
    float inv) {
  uint4 na;
  const T* h_e = reinterpret_cast<const T*>(&ha);
  T* n_e = reinterpret_cast<T*>(&na);
  if constexpr (PRECISE_F32) {
    const float4* wv4 = reinterpret_cast<const float4*>(w_raw);
    const float4 w0 = wv4[idx * 2];
    const float4 w1 = wv4[idx * 2 + 1];
    n_e[0] = fused_from_f32<T>((fused_to_f32<T>(h_e[0]) * inv) * w0.x);
    n_e[1] = fused_from_f32<T>((fused_to_f32<T>(h_e[1]) * inv) * w0.y);
    n_e[2] = fused_from_f32<T>((fused_to_f32<T>(h_e[2]) * inv) * w0.z);
    n_e[3] = fused_from_f32<T>((fused_to_f32<T>(h_e[3]) * inv) * w0.w);
    n_e[4] = fused_from_f32<T>((fused_to_f32<T>(h_e[4]) * inv) * w1.x);
    n_e[5] = fused_from_f32<T>((fused_to_f32<T>(h_e[5]) * inv) * w1.y);
    n_e[6] = fused_from_f32<T>((fused_to_f32<T>(h_e[6]) * inv) * w1.z);
    n_e[7] = fused_from_f32<T>((fused_to_f32<T>(h_e[7]) * inv) * w1.w);
  } else {
    const uint4 wa = reinterpret_cast<const uint4*>(w_raw)[idx];
    const T* w_e = reinterpret_cast<const T*>(&wa);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const T scaled = fused_from_f32<T>(fused_to_f32<T>(h_e[j]) * inv);
      n_e[j] = fused_from_f32<T>(fused_to_f32<T>(w_e[j]) * fused_to_f32<T>(scaled));
    }
  }
  nv[idx] = na;
}

template <typename T, int BLOCK_SIZE, bool PRECISE_F32, bool HAS_RES>
__global__ __launch_bounds__(BLOCK_SIZE) void mlx_fused_add_rmsnorm_cuda_kernel(
    const T* __restrict__ x,
    const T* __restrict__ res,
    const void* __restrict__ w_raw,
    const float* __restrict__ eps_ptr,
    float eps_val,
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
  const size_t base = (size_t)row * (size_t)axis;
  const T* xr = x + base;
  const T* rr = HAS_RES ? (res + base) : nullptr;
  T* hr = HAS_RES ? (h_out + base) : nullptr;
  T* nr = normed_out + base;

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem[NUM_WARPS];
  float sq = 0.0f;

  if constexpr (sizeof(T) == 2) {
    const bool vec8 =
        ((axis & 7) == 0) &&
        (((reinterpret_cast<unsigned long long>(xr) |
           (HAS_RES ? reinterpret_cast<unsigned long long>(rr) : 0ull) |
           reinterpret_cast<unsigned long long>(w_raw) |
           (HAS_RES ? reinterpret_cast<unsigned long long>(hr) : 0ull) |
           reinterpret_cast<unsigned long long>(nr)) & 15ull) == 0);

    if (vec8) {
      const int nvec = axis >> 3;
      const uint4* xv = reinterpret_cast<const uint4*>(xr);
      const uint4* rv = HAS_RES ? reinterpret_cast<const uint4*>(rr) : nullptr;
      uint4* hv = HAS_RES ? reinterpret_cast<uint4*>(hr) : nullptr;
      uint4 h0, h1;
      const bool use_reg2 = (nvec <= 2 * BLOCK_SIZE);
      if (use_reg2) {
        if (tid < nvec) {
          h0 = fused_add_load_vec8<T, HAS_RES>(xv, rv, hv, tid, sq);
        }
        if (tid + BLOCK_SIZE < nvec) {
          h1 = fused_add_load_vec8<T, HAS_RES>(xv, rv, hv, tid + BLOCK_SIZE, sq);
        }
      } else {
        for (int i = tid; i < nvec; i += BLOCK_SIZE) {
          (void)fused_add_load_vec8<T, HAS_RES>(xv, rv, hv, i, sq);
        }
      }

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
          const float eps = eps_ptr ? eps_ptr[0] : eps_val;
          smem[0] = rsqrtf(total / static_cast<float>(axis) + eps);
        }
      }
      __syncthreads();
      const float inv = smem[0];

      uint4* nv = reinterpret_cast<uint4*>(nr);
      if (use_reg2) {
        if (tid < nvec) {
          fused_norm_store_vec8<T, PRECISE_F32>(h0, w_raw, nv, tid, inv);
        }
        if (tid + BLOCK_SIZE < nvec) {
          fused_norm_store_vec8<T, PRECISE_F32>(
              h1, w_raw, nv, tid + BLOCK_SIZE, inv);
        }
      } else {
        for (int i = tid; i < nvec; i += BLOCK_SIZE) {
          uint4 ha;
          if constexpr (HAS_RES) {
            ha = hv[i];
          } else {
            ha = xv[i];
          }
          fused_norm_store_vec8<T, PRECISE_F32>(ha, w_raw, nv, i, inv);
        }
      }
      return;
    }
  }

  for (int i = tid; i < axis; i += BLOCK_SIZE) {
    float hf;
    if constexpr (HAS_RES) {
      const T hv_i =
          fused_from_f32<T>(fused_to_f32<T>(xr[i]) + fused_to_f32<T>(rr[i]));
      hr[i] = hv_i;
      hf = fused_to_f32<T>(hv_i);
    } else {
      hf = fused_to_f32<T>(xr[i]);
    }
    sq += hf * hf;
  }

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
      const float eps = eps_ptr ? eps_ptr[0] : eps_val;
      smem[0] = rsqrtf(total / static_cast<float>(axis) + eps);
    }
  }
  __syncthreads();
  const float inv = smem[0];

  for (int i = tid; i < axis; i += BLOCK_SIZE) {
    T hv_i;
    if constexpr (HAS_RES) {
      hv_i = hr[i];
    } else {
      hv_i = xr[i];
    }
    if constexpr (PRECISE_F32) {
      const float wf = reinterpret_cast<const float*>(w_raw)[i];
      nr[i] = fused_from_f32<T>((fused_to_f32<T>(hv_i) * inv) * wf);
    } else {
      const T scaled = fused_from_f32<T>(fused_to_f32<T>(hv_i) * inv);
      const T wt = reinterpret_cast<const T*>(w_raw)[i];
      nr[i] = fused_from_f32<T>(fused_to_f32<T>(wt) * fused_to_f32<T>(scaled));
    }
  }
}
)cuda"
R"cuda(

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
