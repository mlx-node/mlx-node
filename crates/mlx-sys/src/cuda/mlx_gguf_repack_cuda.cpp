// CUDA backend for GPU repack of GGUF raw blocks into runtime packed arrays.

#include "mlx_common.h"
#include "mlx_gguf_repack.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx/backend/cuda/cuda.h"
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace {

using namespace mlx::core;
using namespace mlx_node::gguf_repack;

constexpr char qwen4_gguf_repack_source[] =
#include "cuda/qwen4_gguf_repack.cu.inc"
    ;

class GgufRepackBandCuda final : public Primitive {
 public:
  GgufRepackBandCuda(Stream stream, int kernel, GgufRepackParams params)
      : Primitive(stream), kernel_(kernel), params_(params) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GgufRepackBandCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto& dst = inputs[1 + i];
      if (!dst.flags().row_contiguous || dst.data_size() != dst.size()) {
        throw std::runtime_error(
            "GgufRepackBandCuda: destination array is not row-contiguous");
      }
      outputs[i].copy_shared_buffer(dst);
    }
    if ((reinterpret_cast<uintptr_t>(gpu_ptr<void>(outputs[0])) & 15u) != 0u) {
      throw std::runtime_error(
          "GgufRepackBandCuda: weight destination is not 16-byte aligned");
    }
    if (kernel_ <= 3) {
      if ((reinterpret_cast<uintptr_t>(gpu_ptr<void>(outputs[1])) & 15u) != 0u ||
          (reinterpret_cast<uintptr_t>(gpu_ptr<void>(outputs[2])) & 3u) != 0u) {
        throw std::runtime_error(
            "GgufRepackBandCuda: K-quant scales/biases destination alignment error");
      }
    }
    array st = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    encoder.set_input_array(st);
    for (const auto& out : outputs) {
      encoder.set_output_array(out);
    }

    cu::KernelArgs args;
    args.append(st);
    for (const auto& out : outputs) {
      args.append(out);
    }
    args.append<uint64_t>(params_.src_off);
    args.append<uint32_t>(params_.nb);
    args.append<uint32_t>(params_.kb);

    uint32_t grid_x = 1;
    if (kKernels[kernel_].t64) {
      args.append<uint32_t>(params_.ku);
      args.append<uint32_t>(params_.tile0);
      grid_x = static_cast<uint32_t>((uint64_t(params_.nb) * 8ull + 255ull) / 256ull);
    } else {
      args.append<uint32_t>(params_.row0);
      if (kernel_ == 0 || kernel_ == 2) {
        grid_x = (params_.nb + 31u) / 32u;
      } else {
        grid_x = (params_.nb + 255u) / 256u;
      }
    }

    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_gguf_repack_source, kKernels[kernel_].name);
    encoder.add_kernel_node_raw(
        fn, dim3(grid_x, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GgufRepackBandCuda)

  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GgufRepackBandCuda*>(&other);
    return rhs && kernel_ == rhs->kernel_ &&
        std::memcmp(&params_, &rhs->params_, sizeof(params_)) == 0;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    std::vector<Shape> shapes;
    shapes.reserve(inputs.size() - 1);
    for (size_t i = 1; i < inputs.size(); ++i) {
      shapes.push_back(inputs[i].shape());
    }
    return shapes;
  }

 private:
  int kernel_;
  GgufRepackParams params_;
};

}  // namespace
#endif

namespace mlx_node::gguf_repack {

bool cuda_repack_available() {
#ifdef MLX_NODE_CUDA_ENABLED
  return mlx::core::cu::is_available();
#else
  return false;
#endif
}

bool cuda_repack_supports_kernel(int kernel) noexcept {
#ifdef MLX_NODE_CUDA_ENABLED
  switch (kernel) {
    case 0:   // mlx_node_gguf_repack_q4k
    case 1:   // mlx_node_gguf_repack_q4k_t64
    case 2:   // mlx_node_gguf_repack_q5k
    case 3:   // mlx_node_gguf_repack_q5k_t64
    case 14:  // mlx_node_gguf_repack_q8_0
    case 17:  // mlx_node_gguf_repack_q5_1
    case 18:  // mlx_node_gguf_repack_q8_0_bias
      return true;
    default:
      return false;
  }
#else
  (void)kernel;
  return false;
#endif
}

void cuda_repack_validate(int kernel) {
#ifdef MLX_NODE_CUDA_ENABLED
  expect(mlx::core::cu::is_available(), "requires the CUDA device");
  expect(kernel >= 0 && kernel < kKernelCount &&
             cuda_repack_supports_kernel(kernel),
         "CUDA repack does not support kernel id " + std::to_string(kernel));
#else
  (void)kernel;
  throw std::runtime_error("requires the CUDA device");
#endif
}

std::vector<mlx::core::array> cuda_repack_band(
    mlx::core::Stream stream,
    int kernel,
    const GgufRepackParams& params,
    std::vector<mlx::core::Shape> shapes,
    const std::vector<mlx::core::Dtype>& dtypes,
    std::vector<mlx::core::array> inputs) {
#ifdef MLX_NODE_CUDA_ENABLED
  auto primitive = std::make_shared<GgufRepackBandCuda>(stream, kernel, params);
  return mlx::core::array::make_arrays(std::move(shapes), dtypes, primitive,
                                       std::move(inputs));
#else
  (void)stream;
  (void)kernel;
  (void)params;
  (void)shapes;
  (void)dtypes;
  (void)inputs;
  throw std::runtime_error("GgufRepackBandCuda requires CUDA");
#endif
}

}  // namespace mlx_node::gguf_repack
