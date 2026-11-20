#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace mlx::core {
namespace {

constexpr char greedy_path_source[] =
#include "cuda/dflash2_greedy_path.cu.inc"
    ;

class DFlash2GreedyPathCuda : public Primitive {
public:
  DFlash2GreedyPathCuda(Stream stream, bool confidence)
      : Primitive(stream), confidence_(confidence) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DFlash2 greedy path CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    for (const auto& input : inputs) {
      encoder.set_input_array(input);
    }
    for (auto& output : outputs) {
      output.set_data(cu::malloc_async(output.nbytes(), encoder));
      encoder.set_output_array(output);
    }
    const int length = inputs[0].shape(1);
    const int top_k = inputs[0].shape(2);
    cu::KernelArgs args;
    args.append(inputs[0]);
    args.append(inputs[1]);
    args.append(outputs[0]);
    if (confidence_) {
      args.append(outputs[1]);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append<int32_t>(length);
    args.append<int32_t>(top_k);
    for (int axis = 1; axis < 3; ++axis) {
      args.append<int64_t>(inputs[0].strides(axis));
    }
    for (int axis = 0; axis < 3; ++axis) {
      args.append<int64_t>(inputs[1].strides(axis));
    }
    auto kernel = cu::jit_kernel(
        encoder.device(), greedy_path_source, "mlx_dflash2_greedy_path_cuda_kernel");
    encoder.add_kernel_node_raw(kernel, dim3(1), dim3(1), {}, 0, args.args());
  }

  DEFINE_NAME(DFlash2GreedyPathCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const DFlash2GreedyPathCuda&>(other);
    return confidence_ == rhs.confidence_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape shape{inputs[0].shape(1)};
    return std::vector<Shape>(confidence_ ? 2 : 1, std::move(shape));
  }

private:
  bool confidence_;
};

bool greedy_path_cuda(
    mlx_array* candidates, mlx_array* scores, mlx_array** out_path,
    mlx_array** out_confidence) {
  if (out_path) {
    *out_path = nullptr;
  }
  if (out_confidence) {
    *out_confidence = nullptr;
  }
  try {
    if (!candidates || !scores || !out_path || default_device() != Device::gpu) {
      return false;
    }
    const auto& c = *reinterpret_cast<array*>(candidates);
    const auto& s = *reinterpret_cast<array*>(scores);
    if (c.dtype() != int32 || s.dtype() != float32 || c.ndim() != 3 ||
        s.ndim() != 3 || c.shape(0) != 1) {
      return false;
    }
    const int length = c.shape(1);
    const int top_k = c.shape(2);
    if (length < 1 || top_k < 1 || top_k > 64 ||
        s.shape() != Shape{length, top_k, top_k}) {
      return false;
    }
    const bool with_confidence = out_confidence != nullptr;
    auto shapes = with_confidence
        ? std::vector<Shape>{{length}, {length}}
        : std::vector<Shape>{{length}};
    auto dtypes = with_confidence
        ? std::vector<Dtype>{int32, float32}
        : std::vector<Dtype>{int32};
    auto outputs = array::make_arrays(
        std::move(shapes), std::move(dtypes),
        std::make_shared<DFlash2GreedyPathCuda>(
            default_stream(Device::gpu), with_confidence),
        {c, s});
    *out_path = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    if (with_confidence) {
      *out_confidence = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    }
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_dflash2_greedy_path_cuda: %s\n", error.what());
    return false;
  }
}

} // namespace
} // namespace mlx::core
#endif

extern "C" bool mlx_dflash2_greedy_path_cuda(
    mlx_array* candidates, mlx_array* scores, mlx_array** out_path) {
#ifdef MLX_NODE_CUDA_ENABLED
  return mlx::core::greedy_path_cuda(candidates, scores, out_path, nullptr);
#else
  (void)candidates;
  (void)scores;
  (void)out_path;
  return false;
#endif
}

extern "C" bool mlx_dflash2_greedy_path_confidence_cuda(
    mlx_array* candidates, mlx_array* scores, mlx_array** out_path,
    mlx_array** out_confidence) {
#ifdef MLX_NODE_CUDA_ENABLED
  return mlx::core::greedy_path_cuda(candidates, scores, out_path, out_confidence);
#else
  (void)candidates;
  (void)scores;
  (void)out_path;
  (void)out_confidence;
  return false;
#endif
}
