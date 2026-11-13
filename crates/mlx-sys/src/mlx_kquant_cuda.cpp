#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_kquant.h"

#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace mlx::core::kquant {
namespace {

constexpr char kernel_source[] =
#include "cuda/kquant.cu.inc"
    ;

struct ShapeInfo {
  int64_t strides[10]{};
  int32_t shape[10]{};
  int32_t ndim{};
};

struct Quant {
  int bits, group, ratio, scale_bytes, bias_entries;
  bool minimum, codebook, affine, tiled;
};

ShapeInfo shape_info(const array& a, int trailing_dims) {
  ShapeInfo info;
  const auto ndim = a.ndim() - trailing_dims;
  if (ndim > std::size(info.shape)) {
    throw std::invalid_argument("CUDA K-quant supports at most 10 batch dimensions");
  }
  info.ndim = static_cast<int32_t>(ndim);
  for (int i = 0; i < info.ndim; ++i) {
    info.shape[i] = a.shape(i);
    info.strides[i] = a.strides()[i];
  }
  return info;
}

Quant quant_info(Mode mode, Layout layout) {
  if (is_grid(mode)) {
    throw std::invalid_argument(
        std::string("CUDA K-quant does not support mode ") + mode_name(mode));
  }
  return {default_bits(mode), default_group_size(mode), super_ratio(mode),
          scale_bytes_per_group(mode), bias_entries_per_super_block(mode),
          has_sub_min(mode), uses_iq4nl_grid(mode), is_affine(mode),
          layout == Layout::Tiled64};
}

// Packed codes and their companions must have contiguous elements within each row.
array packed_rows(const array& a, const Stream& stream, cu::CommandEncoder& encoder) {
  if (a.shape(-1) <= 1 || a.strides().back() == 1) {
    return a;
  }
  auto copy = contiguous_copy_gpu(a, stream);
  encoder.add_temporary(copy);
  return copy;
}

void launch(cu::CommandEncoder& encoder, const char* name, const array& out,
            int work_per_block, cu::KernelArgs& args) {
  if (out.dtype() != float32 && out.dtype() != float16 && out.dtype() != bfloat16) {
    throw std::invalid_argument("CUDA K-quant requires float32, float16, or bfloat16");
  }
  const auto entry = std::string(name) + "<" + dtype_to_cuda_type(out.dtype()) + ">";
  const auto& device = encoder.device();
  // Invalidate cached kernels when either their source or GPU architecture changes.
  static const auto source_hash = std::hash<std::string_view>{}(kernel_source);
  const auto module_name =
      std::string(name) + "_" + std::to_string(source_hash) + "_" +
      dtype_to_string(out.dtype()) + "_" +
      std::to_string(device.compute_capability_major()) + "_" +
      std::to_string(device.compute_capability_minor());
  auto& module = cu::get_jit_module(encoder.device(), module_name, [&] {
    return std::make_tuple(false, std::string(kernel_source),
                           std::vector<std::string>{entry});
  });
  const auto blocks = std::min<size_t>((out.size() - 1) / work_per_block + 1, 65535);
  encoder.add_kernel_node_raw(module.get_kernel(entry), dim3(blocks), dim3(128),
                              {}, 0, args.args());
}

void matmul_cuda(const std::vector<array>& inputs, array& out, const Stream& stream,
                 Mode mode, Layout layout, bool transpose) {
  auto& encoder = cu::get_command_encoder(stream);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }
  const auto q = quant_info(mode, layout);
  const auto& x = inputs[0];
  const auto w = packed_rows(inputs[1], stream, encoder);
  const auto scales = packed_rows(inputs[2], stream, encoder);
  const auto biases = packed_rows(inputs[3], stream, encoder);
  const bool gather = inputs.size() == 6;
  const auto x_batch = shape_info(x, std::min<size_t>(2, x.ndim()));
  const auto w_batch = shape_info(w, 2);
  const auto s_batch = shape_info(scales, 2);
  const auto b_batch = shape_info(biases, 2);
  const auto lhs_shape = gather ? shape_info(inputs[4], 0) : ShapeInfo{};
  const auto rhs_shape = gather ? shape_info(inputs[5], 0) : ShapeInfo{};
  const int M = x.ndim() > 1 ? x.shape(-2) : 1;
  const auto x_row = x.ndim() > 1 ? x.strides()[x.ndim() - 2] : 0;
  auto row_stride = [](const array& a) { return a.strides()[a.ndim() - 2]; };

  cu::KernelArgs args;
  for (const auto& input : {x, w, scales, biases}) {
    encoder.set_input_array(input);
    args.append(input);
  }
  for (int i = 4; i < 6; ++i) {
    if (gather) {
      encoder.set_input_array(inputs[i]);
      args.append(inputs[i]);
    } else {
      args.append<CUdeviceptr>(0);
    }
  }
  encoder.set_output_array(out);
  args.append(out);
  args.append<int64_t>(out.size());
  args.append<int32_t>(M);
  args.append<int32_t>(out.shape(-1));
  args.append<int32_t>(x.shape(-1));
  args.append<int64_t>(x_row);
  args.append<int64_t>(x.strides().back());
  args.append<int64_t>(row_stride(w) * w.itemsize());
  args.append<int64_t>(row_stride(scales) * scales.itemsize());
  args.append<int64_t>(row_stride(biases));
  for (const auto* info : {&x_batch, &w_batch, &s_batch, &b_batch, &lhs_shape, &rhs_shape}) {
    args.append_ptr(info);
  }
  args.append_ptr(&q);
  args.append(transpose);
  launch(encoder, "mlx_node_kquant_matmul", out, 4, args);
}

} // namespace

void KQuantMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  matmul_cuda(inputs, out, stream(), mode_, layout_, transpose_);
}

void KQuantGatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  matmul_cuda(inputs, out, stream(), mode_, Layout::RowMajor, transpose_);
}

void KQuantDequantize::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& encoder = cu::get_command_encoder(stream());
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }
  const auto q = quant_info(mode_, Layout::RowMajor);
  const auto w = packed_rows(inputs[0], stream(), encoder);
  const auto scales = packed_rows(inputs[1], stream(), encoder);
  const auto biases = packed_rows(inputs[2], stream(), encoder);
  const auto w_rows = shape_info(w, 1);
  const auto s_rows = shape_info(scales, 1);
  const auto b_rows = shape_info(biases, 1);
  cu::KernelArgs args;
  for (const auto& input : {w, scales, biases}) {
    encoder.set_input_array(input);
    args.append(input);
  }
  encoder.set_output_array(out);
  args.append(out);
  args.append<int64_t>(out.size());
  args.append<int32_t>(out.shape(-1));
  args.append_ptr(&w_rows);
  args.append_ptr(&s_rows);
  args.append_ptr(&b_rows);
  args.append_ptr(&q);
  launch(encoder, "mlx_node_kquant_dequantize", out, 128, args);
}

} // namespace mlx::core::kquant

#endif
