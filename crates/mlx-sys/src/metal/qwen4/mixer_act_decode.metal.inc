R"metal(
// The mixer activation's 40-block FAST walk over K = 10240, split across the
// eight simdgroups of a threadgroup: each owns five blocks and stores every
// block's qdot term. Simdgroup 0 adds the 40 terms in ascending block order,
// the order of the one-simdgroup chain `result += qdot(...)`, then the same
// simd_sum, so each output keeps that chain's bits. A partial sum per
// simdgroup would reassociate the chain and move the outputs.
// SUMS writes the F32 simd_sum instead of the BF16 SiLU epilogue.
constexpr int K = 10240;
constexpr int N = 320;
constexpr int VP = 8;
constexpr int BLOCK = VP * 32;
constexpr int NB = K / BLOCK;
constexpr int SPLIT = 8;
constexpr int PER = NB / SPLIT;
static_assert(NB % SPLIT == 0, "whole blocks per simdgroup");
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint first = threadgroup_position_in_grid.y;
threadgroup float terms[ROWS * NB * 32];
for (int j = 0; j < PER; ++j) {
  const int blk = int(sg) * PER + j;
  const int col = blk * BLOCK + lane * VP;
  float values[ROWS][VP];
  float sum[ROWS];
  for (int row = 0; row < ROWS; ++row) sum[row] = load_vector<T,float,VP,8>(x + row * K + col, values[row]);
  const auto codes = reinterpret_cast<const device uchar*>(w) + first * K + col;
  const size_t side = first * (K / 32) + col / 32;
  const float scale = float(scales[side]), bias = float(biases[side]);
  for (int row = 0; row < ROWS; ++row)
    terms[(row * NB + blk) * 32 + lane] = qdot<float,VP,8,false>(codes, values[row], scale, bias, sum[row]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg != 0) return;
for (int row = 0; row < ROWS; ++row) {
  float result = 0;
  for (int blk = 0; blk < NB; ++blk) result += terms[(row * NB + blk) * 32 + lane];
  const float value = simd_sum(result);
  if (lane == 0) {
    using O = metal::remove_cv_t<metal::remove_reference_t<decltype(out[0])>>;
    if constexpr (SUMS) {
      out[row * N + first] = O(value);
    } else {
      // TrackFastMixer's SiLU epilogue: the local mixer divides its BF16
      // projection by four before native SiLU.
      const T v = T(value);
      out[row * N + first] = O(sigmoid_table[as_type<ushort>(T(v * T(0.25f)))]);
    }
  }
}
)metal"
