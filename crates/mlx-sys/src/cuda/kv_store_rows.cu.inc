R"cuda(
extern "C" __global__ void mlx_node_kv_store_rows_cuda(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    int B, int H, int rows, int D, int itemsize, int offset,
    long long ds0, long long ds1, long long ds2, long long ds3,
    long long ss0, long long ss1, long long ss2, long long ss3) {
  const long long total = (long long)B * H * rows * D;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int d = (int)(idx % D);
  long long rem = idx / D;
  const int r = (int)(rem % rows);
  rem /= rows;
  const int h = (int)(rem % H);
  const int b = (int)(rem / H);
  const long long doff =
      (b * ds0 + h * ds1 + (long long)(offset + r) * ds2 + d * ds3) * itemsize;
  const long long soff =
      (b * ss0 + h * ss1 + (long long)r * ss2 + d * ss3) * itemsize;
  if (itemsize == 2) {
    *reinterpret_cast<unsigned short*>(dst + doff) =
        *reinterpret_cast<const unsigned short*>(src + soff);
  } else if (itemsize == 4) {
    *reinterpret_cast<unsigned int*>(dst + doff) =
        *reinterpret_cast<const unsigned int*>(src + soff);
  } else {
    dst[doff] = src[soff];
  }
}

// Fast 16-byte vectorized KV store for contiguous last-dim BF16/FP16 rows
// (itemsize == 2, D % 8 == 0, ds3 == 1, ss3 == 1, 16-byte aligned).
// Grid: dim3(rows, H, B), Block: dim3(D / 8, 1, 1) — zero 64-bit div/mod.
extern "C" __global__ void mlx_node_kv_store_rows_vec16_cuda(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    int offset,
    long long ds0_b, long long ds1_b, long long ds2_b,
    long long ss0_b, long long ss1_b, long long ss2_b) {
  const int r = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int v = threadIdx.x;
  const long long doff = b * ds0_b + h * ds1_b + (long long)(offset + r) * ds2_b;
  const long long soff = b * ss0_b + h * ss1_b + (long long)r * ss2_b;
  reinterpret_cast<uint4*>(dst + doff)[v] =
      reinterpret_cast<const uint4*>(src + soff)[v];
}
)cuda"
