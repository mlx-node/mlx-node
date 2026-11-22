R"METAL(
const int M = res_shape[0];
const int D = res_shape[2];
const uint col = thread_position_in_grid.x;
const uint m = thread_position_in_grid.y;
if (int(m) >= M) return;

const size_t base_off = size_t(m) * size_t(HC) * size_t(D);
const size_t p_off = size_t(m) * size_t(HC);

if (VEC4) {
  const uint d4 = uint(D) >> 2;
  if (col >= d4) return;
  const device float4* y4 =
      reinterpret_cast<const device float4*>(y + size_t(m) * size_t(D));
  const float4 yv = y4[col];
  for (int c = 0; c < HC; ++c) {
    const float pv = post[p_off + size_t(c)];
    const device float4* r4 = reinterpret_cast<const device float4*>(
        res + base_off + size_t(c) * size_t(D));
    device float4* o4 = reinterpret_cast<device float4*>(
        out + base_off + size_t(c) * size_t(D));
    o4[col] = r4[col] + pv * yv;
  }
} else {
  if (int(col) >= D) return;
  const float yv = y[size_t(m) * size_t(D) + col];
  for (int c = 0; c < HC; ++c) {
    const float pv = post[p_off + size_t(c)];
    const size_t idx = base_off + size_t(c) * size_t(D) + col;
    out[idx] = res[idx] + pv * yv;
  }
}
)METAL"
