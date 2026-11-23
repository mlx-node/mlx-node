#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_kquant.h"
#include "mlx_cuda_ops.h"

#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mlx::core::kquant {
namespace {

constexpr char kernel_source[] =
#include "cuda/kquant.cu.inc"
    ;

constexpr char kquant_few_source[] =
#include "cuda/kquant_few.cu.inc"
    ;

constexpr char kquant_prefill_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/kquant_prefill.cu.inc"
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

void kquant_few_dispatch(
    int mode, bool tiled, bool m16, int N, int units,
    const char*& entry, int& num_warps) {
  static constexpr const char* kEntries[6][2][2][3] = {
      {{{"mlx_node_kquant_matmul_few<0,false,4,false>",
         "mlx_node_kquant_matmul_few<0,false,8,false>",
         "mlx_node_kquant_matmul_few<0,false,16,false>"},
        {"mlx_node_kquant_matmul_few<0,false,4,true>",
         "mlx_node_kquant_matmul_few<0,false,8,true>",
         "mlx_node_kquant_matmul_few<0,false,16,true>"}},
       {{"mlx_node_kquant_matmul_few<0,true,4,false>",
         "mlx_node_kquant_matmul_few<0,true,8,false>",
         "mlx_node_kquant_matmul_few<0,true,16,false>"},
        {"mlx_node_kquant_matmul_few<0,true,4,true>",
         "mlx_node_kquant_matmul_few<0,true,8,true>",
         "mlx_node_kquant_matmul_few<0,true,16,true>"}}},
      {{{"mlx_node_kquant_matmul_few<1,false,4,false>",
         "mlx_node_kquant_matmul_few<1,false,8,false>",
         "mlx_node_kquant_matmul_few<1,false,16,false>"},
        {"mlx_node_kquant_matmul_few<1,false,4,true>",
         "mlx_node_kquant_matmul_few<1,false,8,true>",
         "mlx_node_kquant_matmul_few<1,false,16,true>"}},
       {{"mlx_node_kquant_matmul_few<1,true,4,false>",
         "mlx_node_kquant_matmul_few<1,true,8,false>",
         "mlx_node_kquant_matmul_few<1,true,16,false>"},
        {"mlx_node_kquant_matmul_few<1,true,4,true>",
         "mlx_node_kquant_matmul_few<1,true,8,true>",
         "mlx_node_kquant_matmul_few<1,true,16,true>"}}},
      {{{"mlx_node_kquant_matmul_few<2,false,4,false>",
         "mlx_node_kquant_matmul_few<2,false,8,false>",
         "mlx_node_kquant_matmul_few<2,false,16,false>"},
        {"mlx_node_kquant_matmul_few<2,false,4,true>",
         "mlx_node_kquant_matmul_few<2,false,8,true>",
         "mlx_node_kquant_matmul_few<2,false,16,true>"}},
       {{"mlx_node_kquant_matmul_few<2,true,4,false>",
         "mlx_node_kquant_matmul_few<2,true,8,false>",
         "mlx_node_kquant_matmul_few<2,true,16,false>"},
        {"mlx_node_kquant_matmul_few<2,true,4,true>",
         "mlx_node_kquant_matmul_few<2,true,8,true>",
         "mlx_node_kquant_matmul_few<2,true,16,true>"}}},
      {{{"mlx_node_kquant_matmul_few<3,false,4,false>",
         "mlx_node_kquant_matmul_few<3,false,8,false>",
         "mlx_node_kquant_matmul_few<3,false,16,false>"},
        {"mlx_node_kquant_matmul_few<3,false,4,true>",
         "mlx_node_kquant_matmul_few<3,false,8,true>",
         "mlx_node_kquant_matmul_few<3,false,16,true>"}},
       {{"mlx_node_kquant_matmul_few<3,true,4,false>",
         "mlx_node_kquant_matmul_few<3,true,8,false>",
         "mlx_node_kquant_matmul_few<3,true,16,false>"},
        {"mlx_node_kquant_matmul_few<3,true,4,true>",
         "mlx_node_kquant_matmul_few<3,true,8,true>",
         "mlx_node_kquant_matmul_few<3,true,16,true>"}}},
      {{{"mlx_node_kquant_matmul_few<4,false,4,false>",
         "mlx_node_kquant_matmul_few<4,false,8,false>",
         "mlx_node_kquant_matmul_few<4,false,16,false>"},
        {"mlx_node_kquant_matmul_few<4,false,4,true>",
         "mlx_node_kquant_matmul_few<4,false,8,true>",
         "mlx_node_kquant_matmul_few<4,false,16,true>"}},
       {{"mlx_node_kquant_matmul_few<4,true,4,false>",
         "mlx_node_kquant_matmul_few<4,true,8,false>",
         "mlx_node_kquant_matmul_few<4,true,16,false>"},
        {"mlx_node_kquant_matmul_few<4,true,4,true>",
         "mlx_node_kquant_matmul_few<4,true,8,true>",
         "mlx_node_kquant_matmul_few<4,true,16,true>"}}},
      {{{"mlx_node_kquant_matmul_few<5,false,4,false>",
         "mlx_node_kquant_matmul_few<5,false,8,false>",
         "mlx_node_kquant_matmul_few<5,false,16,false>"},
        {"mlx_node_kquant_matmul_few<5,false,4,true>",
         "mlx_node_kquant_matmul_few<5,false,8,true>",
         "mlx_node_kquant_matmul_few<5,false,16,true>"}},
       {{"mlx_node_kquant_matmul_few<5,true,4,false>",
         "mlx_node_kquant_matmul_few<5,true,8,false>",
         "mlx_node_kquant_matmul_few<5,true,16,false>"},
        {"mlx_node_kquant_matmul_few<5,true,4,true>",
         "mlx_node_kquant_matmul_few<5,true,8,true>",
         "mlx_node_kquant_matmul_few<5,true,16,true>"}}},
  };
  static constexpr int kWarps[3] = {4, 8, 16};
  int widx = 0;
  if (!tiled) {
    widx = (units >= 8 && mode != 0) ? 1 : 0;
  } else if (units < 8) {
    widx = 0;
  } else if ((units % 8 != 0) && (units % 4 == 0) && N >= 2048) {
    // Eliminate W=8 warp tail imbalance (e.g., 27B K=5120 -> units=20, K=17408 -> units=68)
    // when N >= 2048 (>= 128 CTAs) already saturates SM occupancy at W=4 with half the SMEM.
    widx = 0;
  } else {
    // 128-bit cp.async.cg double-buffered SMEM saturates DRAM at W=8 for both
    // M <= 8 and M in 9..=16 across all K-quant formats.
    widx = 1;
  }
  num_warps = kWarps[widx];
  entry = kEntries[mode][tiled ? 1 : 0][m16 ? 1 : 0][widx];
}

class KQuantFew : public UnaryPrimitive {
public:
  explicit KQuantFew(Stream stream, int mode, bool tiled)
      : UnaryPrimitive(stream), mode_(mode), tiled_(tiled) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("KQuantFew requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<4>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array scales = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array biases = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);

    if (x.ndim() < 2) {
      throw std::invalid_argument("KQuantFew: unsupported replay shape");
    }
    const int K = x.shape(-1);
    const int64_t M64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    const int N = w.shape(0);
    if (M64 < 1 || M64 > 16 || N <= 0 || K < 256 || K % 256 != 0) {
      throw std::invalid_argument("KQuantFew: unsupported replay shape");
    }
    const int M = static_cast<int>(M64);
    const bool m16 = M > 8;
    const int units = K / 256;
    array act(Shape{units * (m16 ? 5120 : 2560)}, uint8, nullptr, {});
    act.set_data(cu::malloc_async(act.nbytes(), encoder));
    encoder.add_temporary(act);

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    // Bind only x -> act for quantize so weight tensors are not counted twice
    // in CUDA Graph bytes_in_graph_.
    encoder.set_input_array(x);
    encoder.set_output_array(act);
    auto quantize = cu::jit_kernel(
        encoder.device(), kquant_few_source,
        m16 ? "mlx_node_kquant_quantize_few_16" : "mlx_node_kquant_quantize_few");
    cu::KernelArgs qargs;
    qargs.append(x);
    qargs.append(act);
    qargs.append<int32_t>(M);
    qargs.append<int32_t>(K);
    const int q_groups = units * (m16 ? 128 : 64);
    encoder.add_kernel_node_raw(
        quantize, dim3((q_groups + 15) / 16), dim3(128), {}, 0, qargs.args());

    const char* entry = nullptr;
    int num_warps = 4;
    kquant_few_dispatch(mode_, tiled_, m16, N, units, entry, num_warps);
    auto matmul = cu::jit_kernel(encoder.device(), kquant_few_source, entry);
    encoder.set_input_array(w);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(act);
    encoder.set_output_array(out);
    cu::KernelArgs args;
    args.append(w);
    args.append(scales);
    args.append(biases);
    args.append(act);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    encoder.add_kernel_node_raw(
        matmul, dim3((N + 15) / 16), dim3(32, num_warps), {}, 0, args.args());
  }

  DEFINE_NAME(KQuantFew)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const KQuantFew*>(&other);
    return rhs && rhs->mode_ == mode_ && rhs->tiled_ == tiled_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }

private:
  int mode_;
  bool tiled_;
};

class KQuantPrefill : public UnaryPrimitive {
public:
  explicit KQuantPrefill(Stream stream, int mode)
      : UnaryPrimitive(stream), mode_(mode) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("KQuantPrefill requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array scales = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array biases = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);

    if (x.ndim() < 2) {
      throw std::invalid_argument("KQuantPrefill: unsupported replay shape");
    }
    const int K = x.shape(-1);
    const int64_t M64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    const int N = w.shape(0);
    if (M64 < 9 || M64 > (std::numeric_limits<int32_t>::max() - 127) ||
        N <= 0 || (N % 64) != 0 || K < 256 || (K % 256) != 0) {
      throw std::invalid_argument("KQuantPrefill: unsupported replay shape");
    }
    const int M = static_cast<int>(M64);
    const int M_pad = ((M + 127) / 128) * 128;
    const int units = K / 256;
    const int64_t total_groups = static_cast<int64_t>(M_pad) * units;

    array xq(Shape{K / 64, M_pad, 64}, uint8, nullptr, {});
    array x_scales(Shape{units, M_pad}, float32, nullptr, {});
    array x_qsums(Shape{units, M_pad, 8}, int16, nullptr, {});
    xq.set_data(cu::malloc_async(xq.nbytes(), encoder));
    x_scales.set_data(cu::malloc_async(x_scales.nbytes(), encoder));
    x_qsums.set_data(cu::malloc_async(x_qsums.nbytes(), encoder));
    encoder.add_temporary(xq);
    encoder.add_temporary(x_scales);
    encoder.add_temporary(x_qsums);

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    encoder.set_input_array(x);
    encoder.set_output_array(xq);
    encoder.set_output_array(x_scales);
    encoder.set_output_array(x_qsums);
    auto quantize = cu::jit_kernel(
        encoder.device(), kquant_prefill_source, "mlx_node_kquant_quantize_prefill");
    cu::KernelArgs qargs;
    qargs.append(x);
    qargs.append(xq);
    qargs.append(x_scales);
    qargs.append(x_qsums);
    qargs.append<int32_t>(M);
    qargs.append<int32_t>(M_pad);
    qargs.append<int32_t>(K);
    const uint32_t q_grid = static_cast<uint32_t>((total_groups + 7) / 8);
    encoder.add_kernel_node_raw(
        quantize, dim3(q_grid), dim3(256), {}, 0, qargs.args());

    static constexpr const char* kPrefillEntries[6] = {
        "mlx_node_kquant_prefill_t64<0>",
        "mlx_node_kquant_prefill_t64<1>",
        "mlx_node_kquant_prefill_t64<2>",
        "mlx_node_kquant_prefill_t64<3>",
        "mlx_node_kquant_prefill_t64<4>",
        "mlx_node_kquant_prefill_t64<5>",
    };
    auto matmul = cu::jit_kernel(
        encoder.device(), kquant_prefill_source, kPrefillEntries[mode_]);
    encoder.set_input_array(w);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(xq);
    encoder.set_input_array(x_scales);
    encoder.set_input_array(x_qsums);
    encoder.set_output_array(out);
    cu::KernelArgs args;
    args.append(w);
    args.append(scales);
    args.append(biases);
    args.append(xq);
    args.append(x_scales);
    args.append(x_qsums);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(M_pad);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    const uint32_t num_tiles =
        static_cast<uint32_t>((M_pad / 128) * (N / 64));
    encoder.add_kernel_node_raw(
        matmul, dim3(num_tiles), dim3(256), {}, 0, args.args());
  }

  DEFINE_NAME(KQuantPrefill)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const KQuantPrefill*>(&other);
    return rhs && rhs->mode_ == mode_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }

private:
  int mode_;
};

class KQuantFewMulti : public Primitive {
public:
  KQuantFewMulti(Stream stream, std::vector<int> modes, std::vector<bool> tiled)
      : Primitive(stream), modes_(std::move(modes)), tiled_(std::move(tiled)) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("KQuantFewMulti requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    const size_t count = modes_.size();
    array x = cu::ensure_row_contiguous_aligned<4>(inputs[0], stream(), encoder);
    if (x.ndim() < 2) {
      throw std::invalid_argument("KQuantFewMulti: unsupported replay shape");
    }
    const int K = x.shape(-1);
    const int64_t M64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    if (M64 < 1 || M64 > 16 || K < 256 || K % 256 != 0) {
      throw std::invalid_argument("KQuantFewMulti: unsupported replay shape");
    }
    const int M = static_cast<int>(M64);
    const bool m16 = M > 8;
    const int units = K / 256;
    array act(Shape{units * (m16 ? 5120 : 2560)}, uint8, nullptr, {});
    act.set_data(cu::malloc_async(act.nbytes(), encoder));
    encoder.add_temporary(act);

    encoder.set_input_array(x);
    encoder.set_output_array(act);
    auto quantize = cu::jit_kernel(
        encoder.device(), kquant_few_source,
        m16 ? "mlx_node_kquant_quantize_few_16" : "mlx_node_kquant_quantize_few");
    cu::KernelArgs qargs;
    qargs.append(x);
    qargs.append(act);
    qargs.append<int32_t>(M);
    qargs.append<int32_t>(K);
    const int q_groups = units * (m16 ? 128 : 64);
    encoder.add_kernel_node_raw(
        quantize, dim3((q_groups + 15) / 16), dim3(128), {}, 0, qargs.args());

    for (size_t p = 0; p < count; ++p) {
      array w = cu::ensure_row_contiguous_aligned<16>(inputs[1 + p * 3], stream(), encoder);
      array scales = cu::ensure_row_contiguous_aligned<16>(inputs[2 + p * 3], stream(), encoder);
      array biases = cu::ensure_row_contiguous_aligned<4>(inputs[3 + p * 3], stream(), encoder);
      const int N = w.shape(0);
      auto& out = outputs[p];
      out.set_data(cu::malloc_async(out.nbytes(), encoder));
      const char* entry = nullptr;
      int num_warps = 4;
      kquant_few_dispatch(modes_[p], tiled_[p], m16, N, units, entry, num_warps);
      auto matmul = cu::jit_kernel(encoder.device(), kquant_few_source, entry);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      encoder.set_input_array(biases);
      encoder.set_input_array(act);
      encoder.set_output_array(out);
      cu::KernelArgs args;
      args.append(w);
      args.append(scales);
      args.append(biases);
      args.append(act);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      encoder.add_kernel_node_raw(
          matmul, dim3((N + 15) / 16), dim3(32, num_warps), {}, 0, args.args());
    }
  }

  DEFINE_NAME(KQuantFewMulti)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const KQuantFewMulti*>(&other);
    return rhs && rhs->modes_ == modes_ && rhs->tiled_ == tiled_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    std::vector<Shape> shapes;
    shapes.reserve(modes_.size());
    for (size_t p = 0; p < modes_.size(); ++p) {
      auto shape = inputs[0].shape();
      shape.back() = inputs[1 + p * 3].shape(0);
      shapes.push_back(std::move(shape));
    }
    return shapes;
  }

private:
  std::vector<int> modes_;
  std::vector<bool> tiled_;
};

class KQuantPrefillMulti : public Primitive {
public:
  KQuantPrefillMulti(Stream stream, std::vector<int> modes)
      : Primitive(stream), modes_(std::move(modes)) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("KQuantPrefillMulti requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    const size_t count = modes_.size();
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    if (x.ndim() < 2) {
      throw std::invalid_argument("KQuantPrefillMulti: unsupported replay shape");
    }
    const int K = x.shape(-1);
    const int64_t M64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    if (M64 < 9 || M64 > (std::numeric_limits<int32_t>::max() - 127) ||
        K < 256 || (K % 256) != 0) {
      throw std::invalid_argument("KQuantPrefillMulti: unsupported replay shape");
    }
    const int M = static_cast<int>(M64);
    const int M_pad = ((M + 127) / 128) * 128;
    const int units = K / 256;
    const int64_t total_groups = static_cast<int64_t>(M_pad) * units;

    array xq(Shape{K / 64, M_pad, 64}, uint8, nullptr, {});
    array x_scales(Shape{units, M_pad}, float32, nullptr, {});
    array x_qsums(Shape{units, M_pad, 8}, int16, nullptr, {});
    xq.set_data(cu::malloc_async(xq.nbytes(), encoder));
    x_scales.set_data(cu::malloc_async(x_scales.nbytes(), encoder));
    x_qsums.set_data(cu::malloc_async(x_qsums.nbytes(), encoder));
    encoder.add_temporary(xq);
    encoder.add_temporary(x_scales);
    encoder.add_temporary(x_qsums);

    encoder.set_input_array(x);
    encoder.set_output_array(xq);
    encoder.set_output_array(x_scales);
    encoder.set_output_array(x_qsums);
    auto quantize = cu::jit_kernel(
        encoder.device(), kquant_prefill_source, "mlx_node_kquant_quantize_prefill");
    cu::KernelArgs qargs;
    qargs.append(x);
    qargs.append(xq);
    qargs.append(x_scales);
    qargs.append(x_qsums);
    qargs.append<int32_t>(M);
    qargs.append<int32_t>(M_pad);
    qargs.append<int32_t>(K);
    const uint32_t q_grid = static_cast<uint32_t>((total_groups + 7) / 8);
    encoder.add_kernel_node_raw(
        quantize, dim3(q_grid), dim3(256), {}, 0, qargs.args());

    static constexpr const char* kPrefillEntries[6] = {
        "mlx_node_kquant_prefill_t64<0>",
        "mlx_node_kquant_prefill_t64<1>",
        "mlx_node_kquant_prefill_t64<2>",
        "mlx_node_kquant_prefill_t64<3>",
        "mlx_node_kquant_prefill_t64<4>",
        "mlx_node_kquant_prefill_t64<5>",
    };
    for (size_t p = 0; p < count; ++p) {
      array w = cu::ensure_row_contiguous_aligned<16>(inputs[1 + p * 3], stream(), encoder);
      array scales = cu::ensure_row_contiguous_aligned<16>(inputs[2 + p * 3], stream(), encoder);
      array biases = cu::ensure_row_contiguous_aligned<4>(inputs[3 + p * 3], stream(), encoder);
      const int N = w.shape(0);
      auto& out = outputs[p];
      out.set_data(cu::malloc_async(out.nbytes(), encoder));
      auto matmul = cu::jit_kernel(
          encoder.device(), kquant_prefill_source, kPrefillEntries[modes_[p]]);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      encoder.set_input_array(biases);
      encoder.set_input_array(xq);
      encoder.set_input_array(x_scales);
      encoder.set_input_array(x_qsums);
      encoder.set_output_array(out);
      cu::KernelArgs args;
      args.append(w);
      args.append(scales);
      args.append(biases);
      args.append(xq);
      args.append(x_scales);
      args.append(x_qsums);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(M_pad);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      const uint32_t num_tiles =
          static_cast<uint32_t>((M_pad / 128) * (N / 64));
      encoder.add_kernel_node_raw(
          matmul, dim3(num_tiles), dim3(256), {}, 0, args.args());
    }
  }

  DEFINE_NAME(KQuantPrefillMulti)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const KQuantPrefillMulti*>(&other);
    return rhs && rhs->modes_ == modes_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    std::vector<Shape> shapes;
    shapes.reserve(modes_.size());
    for (size_t p = 0; p < modes_.size(); ++p) {
      auto shape = inputs[0].shape();
      shape.back() = inputs[1 + p * 3].shape(0);
      shapes.push_back(std::move(shape));
    }
    return shapes;
  }

private:
  std::vector<int> modes_;
};

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

extern "C" mlx_array* mlx_kquant_few(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle, const char* mode_c) {
  try {
    using namespace mlx::core;
    if (!x_handle || !w_handle || !scales_handle || !biases_handle || !mode_c ||
        default_device() != Device::gpu) {
      return nullptr;
    }
    const auto parsed = kquant::parse_mode_layout(mode_c);
    if (!parsed) {
      return nullptr;
    }
    const int mode = [&] {
      switch (parsed->mode) {
        case kquant::Mode::Q4K: return 0;
        case kquant::Mode::Q5K: return 1;
        case kquant::Mode::Q6K: return 2;
        case kquant::Mode::IQ4XS: return 3;
        case kquant::Mode::Q3K: return 4;
        case kquant::Mode::IQ4NL: return 5;
        default: return -1;
      }
    }();
    if (mode < 0 || cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const bool tiled = parsed->layout == kquant::Layout::Tiled64;
    std::vector<array> inputs = {
        *reinterpret_cast<array*>(x_handle), *reinterpret_cast<array*>(w_handle),
        *reinterpret_cast<array*>(scales_handle),
        *reinterpret_cast<array*>(biases_handle)};
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& scales = inputs[2];
    const auto& biases = inputs[3];
    if (x.dtype() != bfloat16 || x.ndim() < 2) {
      return nullptr;
    }
    const int K = x.shape(-1);
    const int64_t M64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    const int bits = mode == 1 ? 5 : (mode == 2 ? 6 : (mode == 4 ? 3 : 4));
    const int scale_cols = (mode == 3 || mode == 5) ? K / 32 : K / 16;
    const int bias_cols = mode == 5 ? K / 32 : ((mode == 0 || mode == 1) ? K / 128 : K / 256);
    const Dtype scale_dtype = (mode == 0 || mode == 1) ? uint8 : int8;
    if (M64 < 1 || M64 > 16 || K < 2048 || K % 256 != 0 ||
        w.ndim() != 2 || w.shape(0) <= 0 || w.dtype() != uint32 ||
        w.shape(1) != K * bits / 32 ||
        scales.ndim() != 2 || scales.dtype() != scale_dtype ||
        scales.shape(0) != w.shape(0) || scales.shape(1) != scale_cols ||
        biases.ndim() != 2 || biases.dtype() != float16 ||
        biases.shape(0) != w.shape(0) || biases.shape(1) != bias_cols ||
        (tiled && w.shape(0) % 64 != 0)) {
      return nullptr;
    }
    auto shape = x.shape();
    shape.back() = w.shape(0);
    auto result = array(
        std::move(shape), bfloat16,
        std::make_shared<kquant::KQuantFew>(
            default_stream(Device::gpu), mode, tiled),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_kquant_few error: " << e.what() << std::endl;
  }
  return nullptr;
}

extern "C" mlx_array* mlx_kquant_prefill(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle, const char* mode_c) {
  try {
    using namespace mlx::core;
    if (!x_handle || !w_handle || !scales_handle || !biases_handle || !mode_c ||
        default_device() != Device::gpu) {
      return nullptr;
    }
    const auto parsed = kquant::parse_mode_layout(mode_c);
    if (!parsed || parsed->layout != kquant::Layout::Tiled64) {
      return nullptr;
    }
    const int mode = [&] {
      switch (parsed->mode) {
        case kquant::Mode::Q4K: return 0;
        case kquant::Mode::Q5K: return 1;
        case kquant::Mode::Q6K: return 2;
        case kquant::Mode::IQ4XS: return 3;
        case kquant::Mode::Q3K: return 4;
        case kquant::Mode::IQ4NL: return 5;
        default: return -1;
      }
    }();
    if (mode < 0 || cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    std::vector<array> inputs = {
        *reinterpret_cast<array*>(x_handle), *reinterpret_cast<array*>(w_handle),
        *reinterpret_cast<array*>(scales_handle),
        *reinterpret_cast<array*>(biases_handle)};
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& scales = inputs[2];
    const auto& biases = inputs[3];
    if (x.dtype() != bfloat16 || x.ndim() < 2) {
      return nullptr;
    }
    const int K = x.shape(-1);
    const int64_t M64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    const int bits = mode == 1 ? 5 : (mode == 2 ? 6 : (mode == 4 ? 3 : 4));
    const int scale_cols = (mode == 3 || mode == 5) ? K / 32 : K / 16;
    const int bias_cols = mode == 5 ? K / 32 : ((mode == 0 || mode == 1) ? K / 128 : K / 256);
    const Dtype scale_dtype = (mode == 0 || mode == 1) ? uint8 : int8;
    if (M64 < 9 || M64 > (std::numeric_limits<int32_t>::max() - 127) ||
        K < 2048 || K % 256 != 0 ||
        w.ndim() != 2 || w.shape(0) <= 0 || w.shape(0) % 64 != 0 ||
        w.dtype() != uint32 || w.shape(1) != K * bits / 32 ||
        scales.ndim() != 2 || scales.dtype() != scale_dtype ||
        scales.shape(0) != w.shape(0) || scales.shape(1) != scale_cols ||
        biases.ndim() != 2 || biases.dtype() != float16 ||
        biases.shape(0) != w.shape(0) || biases.shape(1) != bias_cols) {
      return nullptr;
    }
    auto shape = x.shape();
    shape.back() = w.shape(0);
    auto result = array(
        std::move(shape), bfloat16,
        std::make_shared<kquant::KQuantPrefill>(
            default_stream(Device::gpu), mode),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_kquant_prefill error: " << e.what() << std::endl;
  }
  return nullptr;
}

extern "C" bool mlx_kquant_few_multi(
    mlx_array* x_handle,
    uint32_t count,
    mlx_array* const* weights,
    mlx_array* const* scales_arr,
    mlx_array* const* biases_arr,
    const char* const* modes_c,
    mlx_array** out_handles) {
  if (out_handles) {
    for (uint32_t i = 0; i < count; ++i) out_handles[i] = nullptr;
  }
  try {
    using namespace mlx::core;
    if (!x_handle || count < 2 || count > 4 || !weights || !scales_arr ||
        !biases_arr || !modes_c || !out_handles ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_handle);
    if (x.dtype() != bfloat16 || x.ndim() < 2) {
      return false;
    }
    const int K = x.shape(-1);
    const int64_t M64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    if (M64 < 1 || M64 > 16 || K < 2048 || K % 256 != 0) {
      return false;
    }
    std::vector<array> inputs;
    inputs.reserve(1 + count * 3);
    inputs.push_back(x);
    std::vector<int> modes;
    std::vector<bool> tiled_flags;
    std::vector<Shape> shapes;
    std::vector<Dtype> dtypes;
    modes.reserve(count);
    tiled_flags.reserve(count);
    shapes.reserve(count);
    dtypes.reserve(count);
    for (uint32_t p = 0; p < count; ++p) {
      if (!weights[p] || !scales_arr[p] || !biases_arr[p] || !modes_c[p]) {
        return false;
      }
      const auto parsed = kquant::parse_mode_layout(modes_c[p]);
      if (!parsed) return false;
      const int mode = [&] {
        switch (parsed->mode) {
          case kquant::Mode::Q4K: return 0;
          case kquant::Mode::Q5K: return 1;
          case kquant::Mode::Q6K: return 2;
          case kquant::Mode::IQ4XS: return 3;
          case kquant::Mode::Q3K: return 4;
          case kquant::Mode::IQ4NL: return 5;
          default: return -1;
        }
      }();
      if (mode < 0) return false;
      const bool tiled = parsed->layout == kquant::Layout::Tiled64;
      const auto& w = *reinterpret_cast<array*>(weights[p]);
      const auto& sc = *reinterpret_cast<array*>(scales_arr[p]);
      const auto& bi = *reinterpret_cast<array*>(biases_arr[p]);
      const int bits = mode == 1 ? 5 : (mode == 2 ? 6 : (mode == 4 ? 3 : 4));
      const int scale_cols = (mode == 3 || mode == 5) ? K / 32 : K / 16;
      const int bias_cols = mode == 5 ? K / 32 : ((mode == 0 || mode == 1) ? K / 128 : K / 256);
      const Dtype scale_dtype = (mode == 0 || mode == 1) ? uint8 : int8;
      if (w.ndim() != 2 || w.shape(0) <= 0 || w.dtype() != uint32 ||
          w.shape(1) != K * bits / 32 ||
          sc.ndim() != 2 || sc.dtype() != scale_dtype ||
          sc.shape(0) != w.shape(0) || sc.shape(1) != scale_cols ||
          bi.ndim() != 2 || bi.dtype() != float16 ||
          bi.shape(0) != w.shape(0) || bi.shape(1) != bias_cols ||
          (tiled && w.shape(0) % 64 != 0)) {
        return false;
      }
      inputs.push_back(w);
      inputs.push_back(sc);
      inputs.push_back(bi);
      modes.push_back(mode);
      tiled_flags.push_back(tiled);
      auto shape = x.shape();
      shape.back() = w.shape(0);
      shapes.push_back(std::move(shape));
      dtypes.push_back(bfloat16);
    }
    auto outs = array::make_arrays(
        std::move(shapes), std::move(dtypes),
        std::make_shared<kquant::KQuantFewMulti>(
            default_stream(Device::gpu), std::move(modes), std::move(tiled_flags)),
        std::move(inputs));
    for (uint32_t p = 0; p < count; ++p) {
      out_handles[p] = reinterpret_cast<mlx_array*>(new array(std::move(outs[p])));
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mlx_kquant_few_multi error: " << e.what() << std::endl;
  }
  return false;
}

extern "C" bool mlx_kquant_prefill_multi(
    mlx_array* x_handle,
    uint32_t count,
    mlx_array* const* weights,
    mlx_array* const* scales_arr,
    mlx_array* const* biases_arr,
    const char* const* modes_c,
    mlx_array** out_handles) {
  if (out_handles) {
    for (uint32_t i = 0; i < count; ++i) out_handles[i] = nullptr;
  }
  try {
    using namespace mlx::core;
    if (!x_handle || count < 2 || count > 4 || !weights || !scales_arr ||
        !biases_arr || !modes_c || !out_handles ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_handle);
    if (x.dtype() != bfloat16 || x.ndim() < 2) {
      return false;
    }
    const int K = x.shape(-1);
    const int64_t M64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    if (M64 < 9 || M64 > (std::numeric_limits<int32_t>::max() - 127) ||
        K < 2048 || K % 256 != 0) {
      return false;
    }
    std::vector<array> inputs;
    inputs.reserve(1 + count * 3);
    inputs.push_back(x);
    std::vector<int> modes;
    std::vector<Shape> shapes;
    std::vector<Dtype> dtypes;
    modes.reserve(count);
    shapes.reserve(count);
    dtypes.reserve(count);
    for (uint32_t p = 0; p < count; ++p) {
      if (!weights[p] || !scales_arr[p] || !biases_arr[p] || !modes_c[p]) {
        return false;
      }
      const auto parsed = kquant::parse_mode_layout(modes_c[p]);
      if (!parsed || parsed->layout != kquant::Layout::Tiled64) return false;
      const int mode = [&] {
        switch (parsed->mode) {
          case kquant::Mode::Q4K: return 0;
          case kquant::Mode::Q5K: return 1;
          case kquant::Mode::Q6K: return 2;
          case kquant::Mode::IQ4XS: return 3;
          case kquant::Mode::Q3K: return 4;
          case kquant::Mode::IQ4NL: return 5;
          default: return -1;
        }
      }();
      if (mode < 0) return false;
      const auto& w = *reinterpret_cast<array*>(weights[p]);
      const auto& sc = *reinterpret_cast<array*>(scales_arr[p]);
      const auto& bi = *reinterpret_cast<array*>(biases_arr[p]);
      const int bits = mode == 1 ? 5 : (mode == 2 ? 6 : (mode == 4 ? 3 : 4));
      const int scale_cols = (mode == 3 || mode == 5) ? K / 32 : K / 16;
      const int bias_cols = mode == 5 ? K / 32 : ((mode == 0 || mode == 1) ? K / 128 : K / 256);
      const Dtype scale_dtype = (mode == 0 || mode == 1) ? uint8 : int8;
      if (w.ndim() != 2 || w.shape(0) <= 0 || w.shape(0) % 64 != 0 ||
          w.dtype() != uint32 || w.shape(1) != K * bits / 32 ||
          sc.ndim() != 2 || sc.dtype() != scale_dtype ||
          sc.shape(0) != w.shape(0) || sc.shape(1) != scale_cols ||
          bi.ndim() != 2 || bi.dtype() != float16 ||
          bi.shape(0) != w.shape(0) || bi.shape(1) != bias_cols) {
        return false;
      }
      inputs.push_back(w);
      inputs.push_back(sc);
      inputs.push_back(bi);
      modes.push_back(mode);
      auto shape = x.shape();
      shape.back() = w.shape(0);
      shapes.push_back(std::move(shape));
      dtypes.push_back(bfloat16);
    }
    auto outs = array::make_arrays(
        std::move(shapes), std::move(dtypes),
        std::make_shared<kquant::KQuantPrefillMulti>(
            default_stream(Device::gpu), std::move(modes)),
        std::move(inputs));
    for (uint32_t p = 0; p < count; ++p) {
      out_handles[p] = reinterpret_cast<mlx_array*>(new array(std::move(outs[p])));
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mlx_kquant_prefill_multi error: " << e.what() << std::endl;
  }
  return false;
}

#else
extern "C" mlx_array* mlx_kquant_few(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle, const char* mode_c) {
  (void)x_handle;
  (void)w_handle;
  (void)scales_handle;
  (void)biases_handle;
  (void)mode_c;
  return nullptr;
}

extern "C" mlx_array* mlx_kquant_prefill(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle, const char* mode_c) {
  (void)x_handle;
  (void)w_handle;
  (void)scales_handle;
  (void)biases_handle;
  (void)mode_c;
  return nullptr;
}

extern "C" bool mlx_kquant_few_multi(
    mlx_array*, uint32_t, mlx_array* const*, mlx_array* const*,
    mlx_array* const*, const char* const*, mlx_array**) {
  return false;
}

extern "C" bool mlx_kquant_prefill_multi(
    mlx_array*, uint32_t, mlx_array* const*, mlx_array* const*,
    mlx_array* const*, const char* const*, mlx_array**) {
  return false;
}
#endif
