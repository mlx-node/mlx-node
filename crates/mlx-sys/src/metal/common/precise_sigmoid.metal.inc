// Keep the precise exponential: the default intrinsic can change BF16 ties.
R"metal(
template <typename T> METAL_FUNC T precise_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
)metal"
