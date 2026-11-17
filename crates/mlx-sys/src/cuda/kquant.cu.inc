R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Windows NVRTC has no host C standard library headers in its include path.

// Matrix dimensions are excluded from the batch descriptors used by matmul.
struct ShapeInfo {
  long long strides[10];
  int shape[10];
  int ndim;
};

__device__ long long offset(long long index, const ShapeInfo& info) {
  long long result = 0;
  for (int i = info.ndim - 1; i >= 0; --i) {
    result += (index % info.shape[i]) * info.strides[i];
    index /= info.shape[i];
  }
  return result;
}

struct Quant {
  int bits, group, ratio, scale_bytes, bias_entries;
  bool minimum, codebook, affine, tiled;
};

__device__ float decode(
    const unsigned char* w, const unsigned char* scales, const unsigned short* biases,
    int row, int col, int width, long long w_stride, long long s_stride,
    long long b_stride, Quant q) {
  const int group = col / q.group;
  const int super = group / q.ratio;
  const int group_in_super = group % q.ratio;
  int bit = col * q.bits;
  if (q.tiled) {
    const long long tile = row / 64;
    const int lane = row % 64;
    const int super_count = width / (q.group * q.ratio);
    const long long unit = (tile * (width / 32) + col / 32) * 64 + lane;
    w += unit * q.bits * 4;
    bit = (col % 32) * q.bits;
    const long long companion = (tile * super_count + super) * 64 + lane;
    scales += (companion * q.ratio + group_in_super) * q.scale_bytes;
    biases += companion * q.bias_entries;
  } else {
    w += row * w_stride;
    scales += row * s_stride + group * q.scale_bytes;
    biases += row * b_stride + super * q.bias_entries;
  }
  unsigned int code = w[bit / 8] >> (bit % 8);
  if (bit % 8 + q.bits > 8)
    code |= static_cast<unsigned int>(w[bit / 8 + 1]) << (8 - bit % 8);
  code &= (1u << q.bits) - 1;

  if (q.affine) {
    const unsigned int scale_bits = scales[0] | (scales[1] << 8);
    const float scale = __uint_as_float(scale_bits << 16);
    const float bias = __uint_as_float(static_cast<unsigned int>(biases[group_in_super]) << 16);
    return __fadd_rn(__fmul_rn(scale, code), bias);
  }
  const float d = __half2float(__ushort_as_half(biases[0]));
  if (q.minimum) {
    const float scale = d * scales[0];
    const float minimum = __half2float(__ushort_as_half(biases[1])) * scales[1];
    return __fsub_rn(__fmul_rn(scale, code), minimum);
  }
  const float scale = d * static_cast<signed char>(scales[0]);
  if (q.codebook) {
    constexpr signed char values[] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return scale * values[code];
  }
  return scale * (static_cast<int>(code) - (1 << (q.bits - 1)));
}

template <typename T>
__global__ void mlx_node_kquant_dequantize(
    const unsigned char* w, const unsigned char* scales, const unsigned short* biases, T* out,
    long long count, int width, ShapeInfo w_rows, ShapeInfo s_rows,
    ShapeInfo b_rows, Quant q) {
  for (long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += gridDim.x * blockDim.x) {
    const long long row = i / width;
    out[i] = static_cast<T>(decode(
        w + 4 * offset(row, w_rows), scales + offset(row, s_rows) * (q.affine ? 2 : 1),
        biases + offset(row, b_rows), 0, i % width, width, 0, 0, 0, q));
  }
}

template <typename T>
__global__ void mlx_node_kquant_matmul(
    const T* x, const unsigned char* w, const unsigned char* scales, const unsigned short* biases,
    const unsigned int* lhs, const unsigned int* rhs, T* out, long long count,
    int M, int N, int K, long long x_row, long long x_col, long long w_row,
    long long s_row, long long b_row, ShapeInfo x_batch, ShapeInfo w_batch,
    ShapeInfo s_batch, ShapeInfo b_batch, ShapeInfo lhs_shape, ShapeInfo rhs_shape,
    Quant q, bool transpose) {
  const int lane = threadIdx.x % 32;
  const int warps = blockDim.x / 32;
  const long long matrix_size = static_cast<long long>(M) * N;
  for (long long i = blockIdx.x * warps + threadIdx.x / 32;
       i < count; i += gridDim.x * warps) {
    const long long batch = i / matrix_size;
    const int m = i / N % M;
    const int n = i % N;
    const long long xi = lhs ? lhs[offset(batch, lhs_shape)] : batch;
    const long long wi = rhs ? rhs[offset(batch, rhs_shape)] : batch;
    const T* x_data = x + offset(xi, x_batch) + m * x_row;
    const unsigned char* w_data = w + 4 * offset(wi, w_batch);
    const unsigned char* s_data = scales + offset(wi, s_batch) * (q.affine ? 2 : 1);
    const unsigned short* b_data = biases + offset(wi, b_batch);
    float sum = 0;
    for (int k = lane; k < K; k += 32) {
      const float value = decode(w_data, s_data, b_data,
          transpose ? n : k, transpose ? k : n, transpose ? K : N,
          w_row, s_row, b_row, q);
      sum = fmaf(static_cast<float>(x_data[k * x_col]), value, sum);
    }
    for (int shift = 16; shift > 0; shift /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, shift);
    if (lane == 0) out[i] = static_cast<T>(sum);
  }
}
)cuda"
