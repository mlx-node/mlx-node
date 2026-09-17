R"metal(
// TrackFastMoE.routerGemv (MLXFAST-ROUTERRPS1): one output row per
// SIMD group. Retain the native BF16 GEMV's four-value K walk, F32
// accumulation, shuffle-down reduction and final BF16 boundary.
const uint lane = thread_index_in_simdgroup;
const uint row = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
float result = 0;
for (int base = 0; base < 2560; base += 128) {
  float values[4];
  T weights[4];
  for (int j = 0; j < 4; ++j) {
    values[j] = float(x[base + lane * 4 + j]);
    weights[j] = w[size_t(row) * 2560 + base + lane * 4 + j];
  }
  for (int j = 0; j < 4; ++j) result += weights[j] * values[j];
}
for (ushort step = 16; step >= 1; step >>= 1) {
  result += simd_shuffle_down(result, step);
}
if (lane == 0) out[row] = T(result);
)metal"
