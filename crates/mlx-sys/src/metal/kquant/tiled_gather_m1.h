// Copyright © 2023-2024 Apple Inc.

#pragma once
// Baseline gather_qmv_fast arithmetic, with only packed-weight addresses
// changed to T64. One physical row per selected expert; no expert-stack copy.
template <int bits>
[[kernel]] void kquant_tiled_gather_m1(const device uint32_t* weights [[buffer(0)]],
                                       const device uint8_t* all_scales [[buffer(1)]],
                                       const device float16_t* all_biases [[buffer(2)]],
                                       const device bfloat16_t* input [[buffer(3)]],
                                       const device uint32_t* indices [[buffer(4)]],
                                       device bfloat16_t* output [[buffer(5)]],
                                       constant int* p [[buffer(6)]],
                                       uint3 tid [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  constexpr int group_size = bits == 6 ? 16 : 32;
  constexpr int super_ratio = 256 / group_size;
  constexpr bool has_min = bits != 6;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int values_per_thread = pack_factor * 2;
  constexpr int block_size = values_per_thread * 32;
  static_assert(values_per_thread <= group_size);
  const int K = p[0], N = p[1], E = p[2], input_rows = p[3];
  const uint expert = indices[tid.z];
  const int row0 = int(tid.y) * 8 + int(sg) * 4;
  if (expert >= uint(E)) {
    if (lane == 0)
      for (int row = 0; row < 4; ++row) output[tid.z * N + row0 + row] = bfloat16_t(0);
    return;
  }
  const device uint32_t* w = weights + size_t(expert) * N * K * bits / 32;
  const device uint8_t* scales = all_scales + size_t(expert) * N * K / 16;
  const device float16_t* biases =
      all_biases + size_t(expert) * N * K / (has_min ? 128 : 256);
  const device bfloat16_t* x =
      input + (input_rows == 1 ? 0 : tid.z) * K + lane * values_per_thread;
  float x_thread[values_per_thread];
  float result[4] = {0};
  for (int k = 0; k < K; k += block_size) {
    float sum = load_vector<bfloat16_t, float, values_per_thread, bits>(x, x_thread);
    const int column = k + int(lane) * values_per_thread;
    for (int row = 0; row < 4; ++row) {
      const device uint8_t* wl = kq_tiled_row<bits>(w, row0 + row, K / 32) +
                                 size_t(column / 32) * kq_tiled_unit_stride<bits>() +
                                 (column % 32) * bits / 8;
      KQScalesTiled<float, bits, super_ratio, has_min, KQ_LINEAR, 0> srow(
          scales, biases, K / group_size, row0 + row);
      float s, b;
      srow.at(column / group_size, s, b);
      result[row] +=
          qdot<float, values_per_thread, bits, false>(wl, x_thread, s, b, sum);
    }
    x += block_size;
  }
  for (int row = 0; row < 4; ++row) {
    result[row] = simd_sum(result[row]);
    if (lane == 0) output[tid.z * N + row0 + row] = bfloat16_t(result[row]);
  }
}
