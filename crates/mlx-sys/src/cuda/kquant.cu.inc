R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Matrix dimensions are excluded from the batch descriptors used by matmul.
struct ShapeInfo {
  int64_t strides[10];
  int32_t shape[10];
  int32_t ndim;
};

__device__ int64_t offset(int64_t index, const ShapeInfo& info) {
  int64_t result = 0;
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
    const uint8_t* w, const uint8_t* scales, const uint16_t* biases,
    int row, int col, int width, int64_t w_stride, int64_t s_stride,
    int64_t b_stride, Quant q) {
  const int group = col / q.group;
  const int super = group / q.ratio;
  const int group_in_super = group % q.ratio;
  int bit = col * q.bits;
  if (q.tiled) {
    const int64_t tile = row / 64;
    const int lane = row % 64;
    const int super_count = width / (q.group * q.ratio);
    const int64_t unit = (tile * (width / 32) + col / 32) * 64 + lane;
    w += unit * q.bits * 4;
    bit = (col % 32) * q.bits;
    const int64_t companion = (tile * super_count + super) * 64 + lane;
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
  const float scale = d * static_cast<int8_t>(scales[0]);
  if (q.codebook) {
    constexpr int8_t values[] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return scale * values[code];
  }
  return scale * (static_cast<int>(code) - (1 << (q.bits - 1)));
}

template <typename T>
__global__ void mlx_node_kquant_dequantize(
    const uint8_t* w, const uint8_t* scales, const uint16_t* biases, T* out,
    int64_t count, int width, ShapeInfo w_rows, ShapeInfo s_rows,
    ShapeInfo b_rows, Quant q) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += gridDim.x * blockDim.x) {
    const int64_t row = i / width;
    out[i] = static_cast<T>(decode(
        w + 4 * offset(row, w_rows), scales + offset(row, s_rows) * (q.affine ? 2 : 1),
        biases + offset(row, b_rows), 0, i % width, width, 0, 0, 0, q));
  }
}

template <typename T>
__global__ void mlx_node_kquant_matmul(
    const T* x, const uint8_t* w, const uint8_t* scales, const uint16_t* biases,
    const uint32_t* lhs, const uint32_t* rhs, T* out, int64_t count,
    int M, int N, int K, int64_t x_row, int64_t x_col, int64_t w_row,
    int64_t s_row, int64_t b_row, ShapeInfo x_batch, ShapeInfo w_batch,
    ShapeInfo s_batch, ShapeInfo b_batch, ShapeInfo lhs_shape, ShapeInfo rhs_shape,
    Quant q, bool transpose) {
  const int lane = threadIdx.x % 32;
  const int warps = blockDim.x / 32;
  const int64_t matrix_size = static_cast<int64_t>(M) * N;
  for (int64_t i = blockIdx.x * warps + threadIdx.x / 32;
       i < count; i += gridDim.x * warps) {
    const int64_t batch = i / matrix_size;
    const int m = i / N % M;
    const int n = i % N;
    const int64_t xi = lhs ? lhs[offset(batch, lhs_shape)] : batch;
    const int64_t wi = rhs ? rhs[offset(batch, rhs_shape)] : batch;
    const T* x_data = x + offset(xi, x_batch) + m * x_row;
    const uint8_t* w_data = w + 4 * offset(wi, w_batch);
    const uint8_t* s_data = scales + offset(wi, s_batch) * (q.affine ? 2 : 1);
    const uint16_t* b_data = biases + offset(wi, b_batch);
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
