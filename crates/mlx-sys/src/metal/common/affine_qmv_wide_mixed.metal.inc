// BF16 x and output, F32 affine scales/biases, 8-bit group-32 weights,
// transposed, 2 <= M rows. A whole library source (not a metal_kernel body):
// mlx_affine_mixed_qmm.cpp appends one instantiation per vecs_per_tg.
//
// The arithmetic is MLX's affine qmv_wide (k_lanes 8) run on F32 x: each lane
// walks every 8th group, decodes it in 8-value sub-chunks (scale * q + bias),
// sums each sub-chunk from zero before adding it to the row total, and the
// 8 lanes reduce with a shuffle ladder. BF16 -> F32 is exact, so only the
// single rounding at the store differs from the F32 kernel, and that rounding
// is the cast the promoted path applies to its F32 output. Changing the order
// of any sum here breaks bit-identity with that path.
R"(
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <int vecs_per_tg>
[[kernel]] void affine_qmv_wide_mixed_q8g32(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device float* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 32;
  constexpr int bits = 8;
  constexpr int k_lanes = 8;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 32 / k_lanes;
  constexpr int sub = 8;

  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;

  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +
      results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;

  const int row = min(out_row, out_vec_size - 1);

  const int in_vec_size_w = in_vec_size * bits / 8;
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
  const device float* srow = scales + row * in_vec_size_g;
  const device float* brow = biases + row * in_vec_size_g;

  const device bfloat* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) {
    xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
  }

  float result[vecs_per_tg] = {0};

  for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {
    float scale = srow[g];
    float bias = brow[g];
#pragma unroll
    for (int sc = 0; sc < group_size / sub; sc++) {
      const int k0 = g * group_size + sc * sub;
      const device uint8_t* wc = wrow + k0 * bits / 8;
      float w_dq[sub];
      for (int i = 0; i < sub; i++) {
        w_dq[i] = scale * wc[i] + bias;
      }
#pragma unroll
      for (int v = 0; v < vecs_per_tg; v++) {
        const device bfloat* xc = xv[v] + k0;
        float acc = 0;
#pragma unroll
        for (int i = 0; i < sub; i++) {
          acc += static_cast<float>(xc[i]) * w_dq[i];
        }
        result[v] += acc;
      }
    }
  }

  for (int v = 0; v < vecs_per_tg; v++) {
    result[v] += simd_shuffle_down(result[v], 4);
    result[v] += simd_shuffle_down(result[v], 2);
    result[v] += simd_shuffle_down(result[v], 1);
  }

  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) {
        y[(vec0 + v) * out_vec_size + out_row] = static_cast<bfloat>(result[v]);
      }
    }
  }
}
)"
