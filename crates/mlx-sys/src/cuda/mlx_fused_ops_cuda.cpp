#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace {

constexpr char add_rmsnorm_cuda_source[] =
#include "cuda/add_rmsnorm.cu.inc"
    ;

class FusedAddRmsNormCuda final : public mlx::core::Primitive {
 public:
  explicit FusedAddRmsNormCuda(mlx::core::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("FusedAddRmsNormCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    using namespace mlx::core;
    auto& encoder = cu::get_command_encoder(stream());
    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    const auto& x = owned[0];
    const auto& res = owned[1];
    const auto& w = owned[2];
    const auto& eps = owned[3];
    auto& h_out = outputs[0];
    auto& normed_out = outputs[1];
    h_out.set_data(cu::malloc_async(h_out.nbytes(), encoder));
    normed_out.set_data(cu::malloc_async(normed_out.nbytes(), encoder));
    encoder.set_input_array(x);
    encoder.set_input_array(res);
    encoder.set_input_array(w);
    encoder.set_input_array(eps);
    encoder.set_output_array(h_out);
    encoder.set_output_array(normed_out);

    const int axis = x.shape(-1);
    const int rows = static_cast<int>(x.size() / axis);
    const bool use_256 = axis <= 2048;
    const int block_size = use_256 ? 256 : 512;
    const char* entry = nullptr;
    if (x.dtype() == bfloat16) {
      entry = use_256 ? "mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,256>"
                      : "mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,512>";
    } else if (x.dtype() == float16) {
      entry = use_256 ? "mlx_fused_add_rmsnorm_cuda_kernel<__half,256>"
                      : "mlx_fused_add_rmsnorm_cuda_kernel<__half,512>";
    } else {
      entry = use_256 ? "mlx_fused_add_rmsnorm_cuda_kernel<float,256>"
                      : "mlx_fused_add_rmsnorm_cuda_kernel<float,512>";
    }
    auto kernel = cu::jit_kernel(encoder.device(), add_rmsnorm_cuda_source, entry);
    cu::KernelArgs args;
    args.append(x);
    args.append(res);
    args.append(w);
    args.append(eps);
    args.append(h_out);
    args.append(normed_out);
    args.append<int32_t>(rows);
    args.append<int32_t>(axis);
    encoder.add_kernel_node_raw(
        kernel, dim3(rows, 1, 1), dim3(block_size, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(FusedAddRmsNormCuda)
  bool is_equivalent(const mlx::core::Primitive&) const override { return true; }
  std::vector<mlx::core::Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape(), inputs[0].shape()};
  }
};

class FusedGateBinaryCuda final : public mlx::core::UnaryPrimitive {
 public:
  FusedGateBinaryCuda(mlx::core::Stream stream, bool is_swiglu)
      : UnaryPrimitive(stream), is_swiglu_(is_swiglu) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("FusedGateBinaryCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    using namespace mlx::core;
    auto& encoder = cu::get_command_encoder(stream());
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    const int64_t n = static_cast<int64_t>(inputs[0].size());
    if (n <= 0) {
      return;
    }

    auto is_2d_row_strided = [](const array& arr, int64_t& cols, int64_t& rs) -> bool {
      const int nd = arr.ndim();
      if (nd < 2 || arr.strides(nd - 1) != 1) {
        return false;
      }
      cols = arr.shape(nd - 1);
      rs = arr.strides(nd - 2);
      if (cols <= 0 || rs < 0) {
        return false;
      }
      for (int d = nd - 3; d >= 0; --d) {
        if (arr.shape(d) > 1 && arr.strides(d) != arr.strides(d + 1) * arr.shape(d + 1)) {
          return false;
        }
      }
      return true;
    };

    const bool both_contig =
        inputs[0].flags().row_contiguous && inputs[1].flags().row_contiguous;
    int64_t a_cols = 0, a_rs = 0, b_cols = 0, b_rs = 0;
    if (!both_contig &&
        is_2d_row_strided(inputs[0], a_cols, a_rs) &&
        is_2d_row_strided(inputs[1], b_cols, b_rs) &&
        a_cols == b_cols) {
      const array& a = inputs[0];
      const array& b = inputs[1];
      const int64_t cols = a_cols;
      const int64_t rows = n / cols;
      encoder.set_input_array(a);
      encoder.set_input_array(b);
      encoder.set_output_array(out);
      const char* entry = nullptr;
      if (is_swiglu_) {
        entry = a.dtype() == bfloat16 ? "mlx_swiglu_strided_cuda_kernel<nv_bfloat16>"
            : a.dtype() == float16    ? "mlx_swiglu_strided_cuda_kernel<__half>"
                                      : "mlx_swiglu_strided_cuda_kernel<float>";
      } else {
        entry = a.dtype() == bfloat16 ? "mlx_sigmoid_mul_strided_cuda_kernel<nv_bfloat16>"
            : a.dtype() == float16    ? "mlx_sigmoid_mul_strided_cuda_kernel<__half>"
                                      : "mlx_sigmoid_mul_strided_cuda_kernel<float>";
      }
      const bool vec8 =
          (a.itemsize() == 2) && ((cols & 7) == 0) && ((a_rs & 7) == 0) && ((b_rs & 7) == 0) &&
          (((reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) |
             reinterpret_cast<uintptr_t>(gpu_ptr<void>(b)) |
             reinterpret_cast<uintptr_t>(gpu_ptr<void>(out))) & 15u) == 0);
      const int64_t work = vec8 ? (rows * (cols >> 3)) : n;
      auto kernel = cu::jit_kernel(encoder.device(), add_rmsnorm_cuda_source, entry);
      cu::KernelArgs args;
      args.append(a);
      args.append(b);
      args.append(out);
      args.append<int64_t>(rows);
      args.append<int64_t>(cols);
      args.append<int64_t>(a_rs);
      args.append<int64_t>(b_rs);
      encoder.add_kernel_node_raw(
          kernel, dim3(static_cast<uint32_t>((work + 255) / 256), 1, 1),
          dim3(256, 1, 1), {}, 0, args.args());
      return;
    }

    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    const auto& a = owned[0];
    const auto& b = owned[1];
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    const char* entry = nullptr;
    if (is_swiglu_) {
      entry = a.dtype() == bfloat16 ? "mlx_swiglu_cuda_kernel<nv_bfloat16>"
          : a.dtype() == float16    ? "mlx_swiglu_cuda_kernel<__half>"
                                    : "mlx_swiglu_cuda_kernel<float>";
    } else {
      entry = a.dtype() == bfloat16 ? "mlx_sigmoid_mul_cuda_kernel<nv_bfloat16>"
          : a.dtype() == float16    ? "mlx_sigmoid_mul_cuda_kernel<__half>"
                                    : "mlx_sigmoid_mul_cuda_kernel<float>";
    }
    const bool vec8 =
        (a.itemsize() == 2) && ((n & 7) == 0) &&
        (((reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) |
           reinterpret_cast<uintptr_t>(gpu_ptr<void>(b)) |
           reinterpret_cast<uintptr_t>(gpu_ptr<void>(out))) & 15u) == 0);
    const int64_t work = vec8 ? (n >> 3) : n;
    auto kernel = cu::jit_kernel(encoder.device(), add_rmsnorm_cuda_source, entry);
    cu::KernelArgs args;
    args.append(a);
    args.append(b);
    args.append(out);
    args.append<int64_t>(n);
    encoder.add_kernel_node_raw(
        kernel, dim3(static_cast<uint32_t>((work + 255) / 256), 1, 1),
        dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(FusedGateBinaryCuda)
  bool is_equivalent(const mlx::core::Primitive& other) const override {
    const auto* rhs = dynamic_cast<const FusedGateBinaryCuda*>(&other);
    return rhs && rhs->is_swiglu_ == is_swiglu_;
  }
  std::vector<mlx::core::Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  bool is_swiglu_;
};

}  // namespace
#endif

namespace mlx_node_internal {

#ifdef MLX_NODE_CUDA_ENABLED
std::optional<mlx::core::array> fused_gate_binary_cuda(
    const mlx::core::array& gate,
    const mlx::core::array& value,
    bool is_swiglu) {
  if (mlx::core::default_device() == mlx::core::Device::gpu &&
      gate.shape() == value.shape() && gate.dtype() == value.dtype() &&
      (gate.dtype() == mlx::core::bfloat16 ||
       gate.dtype() == mlx::core::float16 ||
       gate.dtype() == mlx::core::float32)) {
    return mlx::core::array(
        gate.shape(),
        gate.dtype(),
        std::make_shared<FusedGateBinaryCuda>(
            mlx::core::default_stream(mlx::core::Device::gpu), is_swiglu),
        {gate, value});
  }
  return std::nullopt;
}

std::optional<std::pair<mlx::core::array, mlx::core::array>>
fused_add_rmsnorm_cuda(
    const mlx::core::array& x,
    const mlx::core::array& res,
    const mlx::core::array& w,
    const mlx::core::array& eps) {
  if (mlx::core::default_device() != mlx::core::Device::gpu) {
    return std::nullopt;
  }
  const auto dt = x.dtype();
  auto outs = mlx::core::array::make_arrays(
      {x.shape(), x.shape()},
      {dt, dt},
      std::make_shared<FusedAddRmsNormCuda>(
          mlx::core::default_stream(mlx::core::Device::gpu)),
      {x, res, w, eps});
  return std::make_pair(std::move(outs[0]), std::move(outs[1]));
}
#else
std::optional<mlx::core::array> fused_gate_binary_cuda(
    const mlx::core::array&,
    const mlx::core::array&,
    bool) {
  return std::nullopt;
}

std::optional<std::pair<mlx::core::array, mlx::core::array>>
fused_add_rmsnorm_cuda(
    const mlx::core::array&,
    const mlx::core::array&,
    const mlx::core::array&,
    const mlx::core::array&) {
  return std::nullopt;
}
#endif

}  // namespace mlx_node_internal
