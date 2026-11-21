R"cuda(
// (I + A)^-1 for contiguous batches of strictly lower triangular 64x64 A.
// Each warp owns independent output rows and keeps two output columns in registers.
extern "C" __global__ void mlx_node_gdn_invert64(const float* a, float* out) {
  __shared__ float sa[64 * 64];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const float* batch = a + blockIdx.x * 64 * 64;
  float* result = out + blockIdx.x * 64 * 64;

  for (int i = tid; i < 64 * 64; i += 256) {
    sa[i] = batch[i];
  }
  __syncthreads();

  for (int row = warp; row < 64; row += 8) {
    const int col0 = lane;
    const int col1 = lane + 32;
    float x0 = col0 == row ? 1.0f : (col0 < row ? -sa[row * 64 + col0] : 0.0f);
    float x1 = col1 == row ? 1.0f : (col1 < row ? -sa[row * 64 + col1] : 0.0f);
    for (int pivot = row - 1; pivot > 0; --pivot) {
      const float xp = __shfl_sync(0xffffffffu, pivot < 32 ? x0 : x1, pivot & 31, 32);
      if (col0 < pivot) {
        x0 -= xp * sa[pivot * 64 + col0];
      }
      if (col1 < pivot) {
        x1 -= xp * sa[pivot * 64 + col1];
      }
    }
    result[row * 64 + col0] = x0;
    result[row * 64 + col1] = x1;
  }
}
)cuda"
