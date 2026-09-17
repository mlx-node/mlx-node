R"metal(
// Keep MLX's affine8 qmv/qmv_fast lane ownership, K walk and F32
// accumulation. Only the storage dtypes differ: the original GGUF FP16
// sidecars and BF16 input are converted in registers instead of allocating
// three promoted arrays. The projection still rounds to BF16 once at output.
constexpr int VP = FAST ? 8 : 4;
constexpr int BLOCK = VP * 32;
const uint lane = thread_index_in_simdgroup;
const uint first = MIX
    ? threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup
    : threadgroup_position_in_grid.y * (2 * RPS) + simdgroup_index_in_threadgroup * RPS;
if (first >= N) return;
float result[RPS] = {0};
int base = 0;
for (; base < (FAST ? K : K - BLOCK); base += BLOCK) {
  float values[VP];
  const int col = base + lane * VP;
  const float sum = load_vector<T,float,VP,8>(x + col, values);
  for (int r = 0; r < RPS; ++r) {
    const size_t row = first + r * (MIX ? N / 4 : 1);
    const auto codes = reinterpret_cast<const device uchar*>(w) + row * K + col;
    const size_t side = row * (K / 32) + col / 32;
    result[r] += qdot<float,VP,8,false>(codes, values, float(scales[side]), float(biases[side]), sum);
  }
}
if constexpr (!FAST) {
  // TrackFastMoE's FULLTAIL: admitted K values are multiples of 32, so
  // every live tail lane owns a complete pack. Use the identical fixed-size
  // arithmetic instead of runtime-indexed helpers that can spill registers.
  static_assert(K % VP == 0, "dense tail lanes must own complete packs");
  const int col = base + lane * VP;
  if (col < K) {
    float values[VP];
    const float sum = load_vector<T,float,VP,8>(x + col, values);
    for (int r = 0; r < RPS; ++r) {
      const size_t row = first + r * (MIX ? N / 4 : 1);
      const auto codes = reinterpret_cast<const device uchar*>(w) + row * K + col;
      const size_t side = row * (K / 32) + col / 32;
      result[r] += qdot<float,VP,8,false>(codes, values, float(scales[side]), float(biases[side]), sum);
    }
  }
}
if constexpr (MIX) {
  T total = T(0);
  if constexpr (MIX == 2) {
    // TrackFastMixer.upMixSource: independent stream products run in four
    // lanes. Keep this kernel's one-column SIMD ownership and BF16 fold.
    static_assert(RPS == 4, "four mixer streams");
    for (int r = 0; r < RPS; ++r) result[r] = simd_sum(result[r]);
    float product = 0;
    if (lane < 4) {
      const float low = metal::select(result[0], result[1], (lane & 1u) != 0);
      const float high = metal::select(result[2], result[3], (lane & 1u) != 0);
      const T value = T(metal::select(low, high, (lane & 2u) != 0));
      product = float(T(sigmoid_table[as_type<ushort>(value)] * normed[first + lane * (N / 4)]));
    }
    for (int r = 0; r < RPS; ++r) total = T(total + T(simd_shuffle(product, r)));
  } else {
    for (int r = 0; r < RPS; ++r) {
      const T value = T(simd_sum(result[r]));
      const T product = T(sigmoid_table[as_type<ushort>(value)] * normed[first + r * (N / 4)]);
      total = T(total + product);
    }
  }
  if (lane == 0) out[first] = T(total * T(0.25f));
} else {
  for (int r = 0; r < RPS; ++r) {
    const float value = simd_sum(result[r]);
    if (lane == 0) {
      T v = T(value);
      // TrackFastMixer's one-row schedule and SiLU epilogue. The local
      // mixer divides its BF16 projection by four before native SiLU.
      if constexpr (ACT) v = sigmoid_table[as_type<ushort>(T(v * T(0.25f)))];
      out[first + r] = v;
    }
  }
}
)metal"
