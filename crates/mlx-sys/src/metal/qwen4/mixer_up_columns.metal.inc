R"metal(
// TrackFastMixer.upMixSource: two columns per group, retaining the
// existing affine8 row layout and every per-row accumulation boundary.

// Keep MLX's affine8 qmv/qmv_fast lane ownership, K walk and F32
// accumulation. Only the storage dtypes differ: the original GGUF FP16
// sidecars and BF16 input are converted in registers instead of allocating
// three promoted arrays. The projection still rounds to BF16 once at output.
constexpr int VP = FAST ? 8 : 4;
constexpr int BLOCK = VP * 32;
const uint lane = thread_index_in_simdgroup;
const uint tile = threadgroup_position_in_grid.y;
const uint sg = simdgroup_index_in_threadgroup;
const uint d0 = tile * 2;
const uint first = d0;
threadgroup T products[8];

float result[RPS] = {0};
int base = 0;
for (; base < (FAST ? K : K - BLOCK); base += BLOCK) {
  float values[VP];
  const int col = base + lane * VP;
  const float sum = load_vector<T,float,VP,8>(x + col, values);
  for (int r = 0; r < RPS; ++r) {
    const uint slot = sg * 4 + r;
    const size_t row = d0 + (slot & 1) + (N / 4) * (slot >> 1);
    const auto codes = reinterpret_cast<const device uchar*>(w) + row * K + col;
    const size_t side = row * (K / 32) + col / 32;
    result[r] += qdot<float,VP,8,false>(codes, values, float(scales[side]), float(biases[side]), sum);
  }
}
if constexpr (!FAST) {
  const int col = base + lane * VP;
  if (col < K) {
    float values[VP];
    const float sum = load_vector<T,float,VP,8>(x + col, values);
    for (int r = 0; r < RPS; ++r) {
      const uint slot = sg * 4 + r;
    const size_t row = d0 + (slot & 1) + (N / 4) * (slot >> 1);
      const auto codes = reinterpret_cast<const device uchar*>(w) + row * K + col;
      const size_t side = row * (K / 32) + col / 32;
      result[r] += qdot<float,VP,8,false>(codes, values, float(scales[side]), float(biases[side]), sum);
    }
  }
}
for (int r = 0; r < 4; ++r) result[r] = simd_sum(result[r]);
if (lane < 4) {
  const float low = metal::select(result[0], result[1], (lane & 1u) != 0);
  const float high = metal::select(result[2], result[3], (lane & 1u) != 0);
  const T value = T(metal::select(low, high, (lane & 2u) != 0));
  const uint slot = sg * 4 + lane;
  const uint row = d0 + (slot & 1) + (N / 4) * (slot >> 1);
  products[slot] = T(sigmoid_table[as_type<ushort>(value)] * normed[row]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint t = sg * 32 + lane;
if (t < 2) {
  T total = T(0);
  for (int s = 0; s < 4; ++s) total = T(total + products[s * 2 + t]);
  out[d0 + t] = T(total * T(0.25f));
}
)metal"
