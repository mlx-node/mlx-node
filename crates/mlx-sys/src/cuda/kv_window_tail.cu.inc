R"cuda(
#include <cuda_bf16.h>

struct KvTailCopyParams {
  long long src_strides[4];
  long long dst_strides[4];
  int elems;
};

extern "C" __global__ void mlx_kv_window_tail_cuda_kernel(
    const nv_bfloat16* src_k, const nv_bfloat16* src_v,
    nv_bfloat16* dst_k, nv_bfloat16* dst_v,
    KvTailCopyParams params_k, KvTailCopyParams params_v, int rows, int offset,
    int dims) {
  const int tensor = blockIdx.y;
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const auto& params = tensor == 0 ? params_k : params_v;
  const int elems = params.elems;
  if (flat >= elems) {
    return;
  }
  const int d = flat % dims;
  const int t = (flat / dims) % rows;
  const int h = flat / (dims * rows);
  const long long src_offset = h * params.src_strides[1] +
      t * params.src_strides[2] + d * params.src_strides[3];
  const long long dst_offset = h * params.dst_strides[1] +
      (t + offset) * params.dst_strides[2] + d * params.dst_strides[3];
  if (tensor == 0) {
    dst_k[dst_offset] = src_k[src_offset];
  } else {
    dst_v[dst_offset] = src_v[src_offset];
  }
}
)cuda"
