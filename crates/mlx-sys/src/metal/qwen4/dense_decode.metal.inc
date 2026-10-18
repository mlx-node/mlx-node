R"metal(
// Keep MLX's affine8 qmv/qmv_fast lane ownership, K walk and F32
// accumulation. Only the storage dtypes differ: the original GGUF FP16
// sidecars and BF16 input are converted in registers instead of allocating
// three promoted arrays. The projection still rounds to BF16 once at output.
// ROWS activation rows share every weight read; each row keeps the single-row
// lane assignment, ascending K order and its own accumulator, so row j of an
// R-row call carries the bits of a one-row call on row j.
constexpr int VP = FAST ? 8 : 4;
constexpr int BLOCK = VP * 32;
const uint lane = thread_index_in_simdgroup;
const uint first = MIX
    ? threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup
    : threadgroup_position_in_grid.y * (2 * RPS) + simdgroup_index_in_threadgroup * RPS;
if (first >= N) return;
float result[ROWS][RPS] = {};
int base = 0;
for (; base < (FAST ? K : K - BLOCK); base += BLOCK) {
  float values[ROWS][VP];
  float sum[ROWS];
  const int col = base + lane * VP;
  for (int row = 0; row < ROWS; ++row) sum[row] = load_vector<T,float,VP,8>(x + row * K + col, values[row]);
  for (int r = 0; r < RPS; ++r) {
    const size_t wrow = first + r * (MIX ? N / 4 : 1);
    const auto codes = reinterpret_cast<const device uchar*>(w) + wrow * K + col;
    const size_t side = wrow * (K / 32) + col / 32;
    const float scale = float(scales[side]), bias = float(biases[side]);
    for (int row = 0; row < ROWS; ++row)
      result[row][r] += qdot<float,VP,8,false>(codes, values[row], scale, bias, sum[row]);
  }
}
if constexpr (!FAST) {
  // TrackFastMoE's FULLTAIL: admitted K values are multiples of 32, so
  // every live tail lane owns a complete pack. Use the identical fixed-size
  // arithmetic instead of runtime-indexed helpers that can spill registers.
  static_assert(K % VP == 0, "dense tail lanes must own complete packs");
  const int col = base + lane * VP;
  if (col < K) {
    float values[ROWS][VP];
    float sum[ROWS];
    for (int row = 0; row < ROWS; ++row) sum[row] = load_vector<T,float,VP,8>(x + row * K + col, values[row]);
    for (int r = 0; r < RPS; ++r) {
      const size_t wrow = first + r * (MIX ? N / 4 : 1);
      const auto codes = reinterpret_cast<const device uchar*>(w) + wrow * K + col;
      const size_t side = wrow * (K / 32) + col / 32;
      const float scale = float(scales[side]), bias = float(biases[side]);
      for (int row = 0; row < ROWS; ++row)
        result[row][r] += qdot<float,VP,8,false>(codes, values[row], scale, bias, sum[row]);
    }
  }
}
for (int row = 0; row < ROWS; ++row) {
  if constexpr (MIX) {
    T total = T(0);
    if constexpr (MIX == 2) {
      // TrackFastMixer.upMixSource: independent stream products run in four
      // lanes. Keep this kernel's one-column SIMD ownership and BF16 fold.
      static_assert(RPS == 4, "four mixer streams");
      float reduced[RPS];
      for (int r = 0; r < RPS; ++r) reduced[r] = simd_sum(result[row][r]);
      float product = 0;
      if (lane < 4) {
        const float low = metal::select(reduced[0], reduced[1], (lane & 1u) != 0);
        const float high = metal::select(reduced[2], reduced[3], (lane & 1u) != 0);
        const T value = T(metal::select(low, high, (lane & 2u) != 0));
        product = float(T(sigmoid_table[as_type<ushort>(value)] * normed[row * N + first + lane * (N / 4)]));
      }
      for (int r = 0; r < RPS; ++r) total = T(total + T(simd_shuffle(product, r)));
    } else {
      for (int r = 0; r < RPS; ++r) {
        const T value = T(simd_sum(result[row][r]));
        const T product = T(sigmoid_table[as_type<ushort>(value)] * normed[row * N + first + r * (N / 4)]);
        total = T(total + product);
      }
    }
    if (lane == 0) out[row * (N / 4) + first] = T(total * T(0.25f));
  } else {
    for (int r = 0; r < RPS; ++r) {
      const float value = simd_sum(result[row][r]);
      if (lane == 0) {
        T v = T(value);
        // TrackFastMixer's one-row schedule and SiLU epilogue. The local
        // mixer divides its BF16 projection by four before native SiLU.
        if constexpr (ACT) v = sigmoid_table[as_type<ushort>(T(v * T(0.25f)))];
        out[row * N + first + r] = v;
      }
    }
  }
}
)metal"
