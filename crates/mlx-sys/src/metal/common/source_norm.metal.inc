R"SOURCE_NORM(
// Adapted from Splash 1482f6bbc2abf12375332725302019594150b6cb,
// common/rms_inverse.h and shared/normalization.metal (Apache-2.0).
#include <metal_stdlib>
using namespace metal;

inline float source_norm_inverse(float sum, threadgroup float* reductions,
                                 uint tid, uint lane, uint sg) {
#pragma clang fp reassociate(off)
  sum = simd_sum(sum);
  if (lane == 0) reductions[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += reductions[i];
    reductions[0] = rsqrt(total / 2048 + 1.0e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return reductions[0];
}

template <bool AddResidual, bool Staged>
kernel void source_norm_2048(
    device const bfloat* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const bfloat* residual [[buffer(2)]],
    device bfloat* normalized [[buffer(3)]],
    device bfloat* hidden [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp reassociate(off)
  threadgroup float reductions[Staged ? 32 : 8];
  threadgroup bfloat stage[Staged ? 2048 : 1];
  const ulong base = ulong(row) * 2048;
  float sum = 0.0f;
  if constexpr (Staged) {
    // Scalar addresses retain correctness for any BF16/F32 element offset.
    // Four contiguous values per active thread match the source staged layout.
    if (tid < 512) {
#pragma unroll
      for (uint j = 0; j < 4; ++j) {
        const uint col = tid * 4 + j;
        bfloat x = input[base + col];
        if constexpr (AddResidual) {
          x = bfloat(float(x) + float(residual[base + col]));
          hidden[base + col] = x;
        }
        stage[col] = x;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 256) {
      for (uint col = tid; col < 2048; col += 256) {
        const float x = float(stage[col]);
        sum += x * x;
      }
    }
  } else {
    for (uint col = tid; col < 2048; col += 256) {
      bfloat x = input[base + col];
      if constexpr (AddResidual) {
        x = bfloat(float(x) + float(residual[base + col]));
        hidden[base + col] = x;
      }
      const float value = float(x);
      sum += value * value;
    }
  }
  const float inverse = source_norm_inverse(sum, reductions, tid, lane, sg);
  if constexpr (Staged) {
    if (tid < 512) {
#pragma unroll
      for (uint j = 0; j < 4; ++j) {
        const uint col = tid * 4 + j;
        normalized[base + col] = bfloat((float(stage[col]) * inverse) * weight[col]);
      }
    }
  } else {
    for (uint col = tid; col < 2048; col += 256) {
      bfloat x = input[base + col];
      if constexpr (AddResidual) x = bfloat(float(x) + float(residual[base + col]));
      normalized[base + col] = bfloat((float(x) * inverse) * weight[col]);
    }
  }
}
template [[host_name("source_norm_2048_stream")]] kernel decltype(source_norm_2048<false,false>) source_norm_2048<false,false>;
template [[host_name("source_norm_2048_stage")]] kernel decltype(source_norm_2048<false,true>) source_norm_2048<false,true>;
template [[host_name("source_norm_2048_add_stream")]] kernel decltype(source_norm_2048<true,false>) source_norm_2048<true,false>;
template [[host_name("source_norm_2048_add_stage")]] kernel decltype(source_norm_2048<true,true>) source_norm_2048<true,true>;
)SOURCE_NORM"
