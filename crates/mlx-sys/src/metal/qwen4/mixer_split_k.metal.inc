R"metal(
// TrackFastMixerSplitK (cb5b4578): parallelize dot-product blocks, then
// restore each lane's ascending block fold before the original SIMD sum.
// Local affine8 packs, FP16 companions and BF16 boundaries are unchanged.
const uint tile = threadgroup_position_in_grid.y;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
constexpr int K = 10240;
constexpr int SPLIT = 4;
threadgroup float scratch[80 * 32];
if (tile < 160) {
  constexpr int RPS = 2, VP = 8, BLOCK = VP * 32, NB = K / BLOCK;
  for (int block = int(sg) * (NB / SPLIT);
       block < (int(sg) + 1) * (NB / SPLIT); ++block) {
    const int col = block * BLOCK + lane * VP;
    float values[VP];
    const float sum = load_vector<T,float,VP,8>(x + col, values);
    for (int r = 0; r < RPS; ++r) {
      const uint row = tile * RPS + r;
      const auto codes = reinterpret_cast<const device uchar*>(wd) + row * K + col;
      const size_t side = row * (K / 32) + col / 32;
      scratch[(block * RPS + r) * 32 + lane] =
          qdot<float,VP,8,false>(codes, values, float(sd[side]), float(bd[side]), sum);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float result[RPS] = {0};
    for (int block = 0; block < NB; ++block)
      for (int r = 0; r < RPS; ++r)
        result[r] += scratch[(block * RPS + r) * 32 + lane];
    for (int r = 0; r < RPS; ++r) {
      const T value = T(simd_sum(result[r]));
      if (lane == 0)
        act[tile * RPS + r] = silu_table[as_type<ushort>(T(value * T(0.25f)))];
    }
  }
} else {
  constexpr int VP = 4, BLOCK = VP * 32, NB = K / BLOCK;
  const uint row = tile - 160;
  for (int block = int(sg) * (NB / SPLIT);
       block < (int(sg) + 1) * (NB / SPLIT); ++block) {
    const int col = block * BLOCK + lane * VP;
    float values[VP];
    const float sum = load_vector<T,float,VP,8>(x + col, values);
    const auto codes = reinterpret_cast<const device uchar*>(wi) + row * K + col;
    const size_t side = row * (K / 32) + col / 32;
    scratch[block * 32 + lane] =
        qdot<float,VP,8,false>(codes, values, float(si[side]), float(bi[side]), sum);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float result = 0;
    for (int block = 0; block < NB; ++block) result += scratch[block * 32 + lane];
    const T value = T(simd_sum(result));
    if (lane == 0) injection[row] = value;
  }
}
)metal"
