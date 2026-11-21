R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Bit-exact CUDA twin of metal/common/dflash2_conv.metal.inc:
//   out[b,l,h] = sum_k (base[side,k,h] + dyn[b,l,side,k,g(h)]) * x[b,l-k,h]
// Every add/mul rounds through InT, matching the elementwise chain bit-for-bit.
template <typename T>
__device__ __forceinline__ float d2c_to_f32(T v);

template <>
__device__ __forceinline__ float d2c_to_f32<nv_bfloat16>(nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float d2c_to_f32<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float d2c_to_f32<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T d2c_from_f32(float v);

template <>
__device__ __forceinline__ nv_bfloat16 d2c_from_f32<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ __half d2c_from_f32<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ float d2c_from_f32<float>(float v) {
  return v;
}

template <typename T>
__global__ __launch_bounds__(256) void mlx_dflash2_conv_cuda_kernel(
    const T* __restrict__ x,
    const T* __restrict__ dyn,
    const T* __restrict__ base,
    T* __restrict__ out,
    int B, int L, int H, int K, int G, int side) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int l = blockIdx.y;
  const int b = blockIdx.z;
  if (h >= H || l >= L || b >= B) {
    return;
  }
  const int S = H / G;
  const int g = h / S;
  T acc = d2c_from_f32<T>(0.0f);
  const size_t dyn_row = (((size_t)b * L + l) * 2 + side) * (size_t)K * G;
  const size_t base_side = (size_t)side * K * H;
  for (int k = 0; k < K; ++k) {
    const int lk = l - k;
    const T xv = lk >= 0
        ? x[((size_t)b * L + lk) * H + h]
        : d2c_from_f32<T>(0.0f);
    const T w = d2c_from_f32<T>(
        d2c_to_f32<T>(base[base_side + (size_t)k * H + h]) +
        d2c_to_f32<T>(dyn[dyn_row + (size_t)k * G + g]));
    const T prod = d2c_from_f32<T>(d2c_to_f32<T>(w) * d2c_to_f32<T>(xv));
    acc = d2c_from_f32<T>(d2c_to_f32<T>(acc) + d2c_to_f32<T>(prod));
  }
  out[((size_t)b * L + l) * H + h] = acc;
}
)cuda"
