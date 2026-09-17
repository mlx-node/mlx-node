R"METAL(
static_assert(D == 256, "eight values per lane in a single SIMD group");
// Adapted from TrackFastKernels2.attnPrepSource at 8981cef5 (MIT).
// See ../MLXFAST-LICENSE.txt. Local F32 scales and BF16 rotation boundaries
// are preserved. Unlike the reference RMSNorm reduction, the local compiled
// mean uses one SIMD group with two successive four-element reads per lane
// for a 256-element row (MLX reduce_row.h), then sqrt and division.
const uint lid = thread_position_in_threadgroup.x;
const uint row = threadgroup_position_in_grid.y;

// Read the transposed Q/gate view directly; avoid a full contiguous copy.
const long src = (row / S) * x_strides[1] + (row % S) * x_strides[2];
threadgroup T normalized[D];
float kept[8];
float acc = 0;
for (uint j=0; j<8; ++j) {
  kept[j] = float(x[src + (lid * 4 + (j % 4) + (j / 4) * 128) * x_strides[3]]);
  volatile float sq = kept[j] * kept[j];
  acc = sq + acc;
}
float total = simd_sum(acc);
volatile float mean = total * (1.0f / float(D));
float denom = metal::precise::sqrt(mean + eps);
for (uint j=0; j<8; ++j) {
  uint d = lid * 4 + (j % 4) + (j / 4) * 128;
  float n = kept[j] / denom;
  float scale = w[d * w_strides[0]];
  normalized[d] = T(n * scale);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j=0; j<8; ++j) {
  uint d = lid * 4 + (j % 4) + (j / 4) * 128;
  T result = normalized[d];
  if (d < ROT) {
    uint k = d % (ROT/2);
    uint token = row % S;
    T a = normalized[k], b = normalized[k+ROT/2];
    T c = cosb[token * cosb_strides[2] + k * cosb_strides[3]], sn = sinb[token * sinb_strides[2] + k * sinb_strides[3]];
    T p = (d < ROT/2 ? a : b) * c;
    T q = (d < ROT/2 ? b : a) * sn;
    result = d < ROT/2 ? T(p - q) : T(p + q);
  }
  out[row * D + d] = result;
}
)METAL"
