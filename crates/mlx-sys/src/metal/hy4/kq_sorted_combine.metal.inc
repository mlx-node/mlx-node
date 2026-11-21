R"METAL(
const int R = down_shape[0];
const int H = down_shape[1];
const int N_TOK = weights_shape[0];
const uint tok = thread_position_in_grid.y;
const uint c = thread_position_in_grid.x;
if constexpr (VEC4) {
  const uint cols4 = uint(H / 4);
  if (tok >= uint(N_TOK) || c >= cols4) return;
  device float4* out4 = reinterpret_cast<device float4*>(out);
  const size_t base = size_t(tok) * size_t(TOP);
  float4 acc = float4(0.0f);
  if constexpr (IN_F16) {
    const device half4* down4 = reinterpret_cast<const device half4*>(down);
    for (int slot = 0; slot < TOP; ++slot) {
      const uint row = inverse[base + size_t(slot)];
      if (row < uint(R)) {
        const float w = weights[base + size_t(slot)];
        acc = metal::fma(
            float4(down4[size_t(row) * size_t(cols4) + size_t(c)]),
            float4(w),
            acc);
      }
    }
  } else {
    const device float4* down4 = reinterpret_cast<const device float4*>(down);
    for (int slot = 0; slot < TOP; ++slot) {
      const uint row = inverse[base + size_t(slot)];
      if (row < uint(R)) {
        const float w = weights[base + size_t(slot)];
        acc = metal::fma(
            down4[size_t(row) * size_t(cols4) + size_t(c)],
            float4(w),
            acc);
      }
    }
  }
  out4[size_t(tok) * size_t(cols4) + size_t(c)] = acc;
} else {
  if (tok >= uint(N_TOK) || c >= uint(H)) return;
  const size_t base = size_t(tok) * size_t(TOP);
  float acc = 0.0f;
  for (int slot = 0; slot < TOP; ++slot) {
    const uint row = inverse[base + size_t(slot)];
    if (row < uint(R)) {
      const float w = weights[base + size_t(slot)];
      acc = metal::fma(float(down[size_t(row) * size_t(H) + size_t(c)]), w, acc);
    }
  }
  out[size_t(tok) * size_t(H) + size_t(c)] = acc;
}
)METAL"