R"metal(
// TrackFastMixer.downInjectSource: one down row per SIMD group plus two
// injection tiles. Preserve the local affine8 GEMV walks and BF16 boundaries.
const uint tile = threadgroup_position_in_grid.y;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
constexpr int K = 10240;
float result = 0;
if (tile < 160) {
  const uint row = tile * 2 + sg;
  for (int base = 0; base < K; base += 256) {
    const int col = base + lane * 8;
    float values[8];
    const float sum = load_vector<T,float,8,8>(x + col, values);
    const auto codes = reinterpret_cast<const device uchar*>(wd) + row * K + col;
    const size_t side = row * (K / 32) + col / 32;
    result += qdot<float,8,8,false>(codes, values, float(sd[side]), float(bd[side]), sum);
  }
  const T value = T(simd_sum(result));
  if (lane == 0) act[row] = silu_table[as_type<ushort>(T(value * T(0.25f)))];
} else {
  const uint row = (tile - 160) * 2 + sg;
  int base = 0;
  for (; base < K - 128; base += 128) {
    const int col = base + lane * 4;
    float values[4];
    const float sum = load_vector<T,float,4,8>(x + col, values);
    const auto codes = reinterpret_cast<const device uchar*>(wi) + row * K + col;
    const size_t side = row * (K / 32) + col / 32;
    result += qdot<float,4,8,false>(codes, values, float(si[side]), float(bi[side]), sum);
  }
  const int col = base + lane * 4;
  float values[4];
  const float sum = load_vector_safe<T,float,4,8>(x + col, values, 4);
  const auto codes = reinterpret_cast<const device uchar*>(wi) + row * K + col;
  const size_t side = row * (K / 32) + col / 32;
  result += qdot_safe<float,4,8,false>(codes, values, float(si[side]), float(bi[side]), sum, 4);
  const float value = simd_sum(result);
  if (lane == 0) injection[row] = T(value);
}
)metal"
