// Fuse target attention's native BF16 sigmoid gate with native SG8 input
// preparation. Splash uses producer-prepared inputs too, but this preserves
// mlx-node's operand packing and row-sum association (not source Table16).
R"(
template <typename T>
inline T attention_sg8_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
inline float attention_sg8_fm_reduce(float x) {
  x += simd_shuffle_xor(x, ushort(2));
  x += simd_shuffle_xor(x, ushort(4));
  x += simd_shuffle_xor(x, ushort(16));
  return x;
}
[[kernel]] void mlx_node_attention_sg8_prepare(
    const device bfloat16_t* gate [[buffer(0)]],
    const device bfloat16_t* attention [[buffer(1)]],
    device bfloat16_t* values [[buffer(2)]],
    device uint4* packed [[buffer(3)]],
    device float* sums [[buffer(4)]],
    constant long* strides [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint K = 4096, D = 256;
  const uint unit = group * 4 + sg;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4u) | ((lane >> 1) & 3u);
  const uint fn = ((qid & 2u) << 1) | ((lane & 1u) << 1);
  const uint column = unit * 32 + fm * 4;
  const uint head = column / D, dim = column % D;
  bfloat16_t a[4], b[4];
  uint pair[4];
#pragma unroll
  for (uint i = 0; i < 4; ++i) {
    const bfloat16_t ga = gate[long(fn) * strides[0] + column + i];
    const bfloat16_t gb = gate[long(fn + 1) * strides[0] + column + i];
    const bfloat16_t va = attention[long(head) * strides[1] + long(fn) * strides[2] + dim + i];
    const bfloat16_t vb = attention[long(head) * strides[1] + long(fn + 1) * strides[2] + dim + i];
    // Same BF16 sigmoid boundary and BF16 product as sigmoid_mul_compiled.
    a[i] = va * attention_sg8_sigmoid(ga);
    b[i] = vb * attention_sg8_sigmoid(gb);
    values[fn * K + column + i] = a[i];
    values[(fn + 1) * K + column + i] = b[i];
    pair[i] = uint(as_type<ushort>(a[i])) | (uint(as_type<ushort>(b[i])) << 16);
  }
  packed[unit * 32 + lane] = uint4(pair[0], pair[1], pair[2], pair[3]);
  const float sa = attention_sg8_fm_reduce((float(a[0]) + float(a[1])) + (float(a[2]) + float(a[3])));
  const float sb = attention_sg8_fm_reduce((float(b[0]) + float(b[1])) + (float(b[2]) + float(b[3])));
  if (fm == 0) {
    sums[unit * 8 + fn] = sa;
    sums[unit * 8 + fn + 1] = sb;
  }
}
)"
