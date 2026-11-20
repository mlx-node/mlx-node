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
)cuda"
