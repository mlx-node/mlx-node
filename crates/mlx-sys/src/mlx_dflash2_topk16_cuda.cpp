#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

#include <limits>

namespace mlx::core {
namespace {

constexpr char dflash2_topk16_source[] =
#include "cuda/dflash2_topk16.cu.inc"
    ;

class DFlash2Topk16Cuda : public Primitive {
public:
  explicit DFlash2Topk16Cuda(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DFlash2 topk16 CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    const auto& input = owned[0];
    auto& ids = outputs[0];
    auto& values = outputs[1];
    ids.set_data(cu::malloc_async(ids.nbytes(), encoder));
    values.set_data(cu::malloc_async(values.nbytes(), encoder));
    encoder.set_input_array(input);
    encoder.set_output_array(ids);
    encoder.set_output_array(values);

    const int vocab = input.shape(-1);
    const int rows = input.size() / vocab;
    const int dtype = input.dtype() == bfloat16 ? 0 : input.dtype() == float16 ? 1 : 2;
    cu::KernelArgs args;
    args.append(input);
    args.append<int32_t>(dtype);
    args.append<int32_t>(vocab);
    args.append(ids);
    args.append(values);
    auto kernel = cu::jit_kernel(
        encoder.device(), dflash2_topk16_source, "mlx_dflash2_topk16_cuda_kernel");
    encoder.add_kernel_node_raw(
        kernel, dim3(rows, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(DFlash2Topk16Cuda)
  bool is_equivalent(const Primitive&) const override {
    return true;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int rows = inputs[0].size() / inputs[0].shape(-1);
    return {{rows, 16}, {rows, 16}};
  }
};

} // namespace
} // namespace mlx::core
#endif

extern "C" int mlx_dflash2_topk16_cuda(
    mlx_array* logits, mlx_array** out_ids, mlx_array** out_values) {
#ifdef MLX_NODE_CUDA_ENABLED
  if (out_ids) {
    *out_ids = nullptr;
  }
  if (out_values) {
    *out_values = nullptr;
  }
  try {
    if (!logits || !out_ids || !out_values) {
      return 0;
    }
    using namespace mlx::core;
    const auto& input = *reinterpret_cast<array*>(logits);
    if (default_device() != Device::gpu || input.ndim() < 2) {
      return 0;
    }
    const int vocab = input.shape(-1);
    if (vocab < 16 || (input.dtype() != bfloat16 && input.dtype() != float16 &&
                       input.dtype() != float32)) {
      return 0;
    }
    const int64_t rows = input.size() / vocab;
    if (rows < 1 || rows > std::numeric_limits<int>::max() ||
        input.size() != rows * vocab) {
      return 0;
    }
    auto outputs = array::make_arrays(
        {{static_cast<int>(rows), 16}, {static_cast<int>(rows), 16}},
        {int32, float32},
        std::make_shared<DFlash2Topk16Cuda>(default_stream(Device::gpu)),
        {input});
    *out_ids = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_values = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_dflash2_topk16_cuda: %s\n", error.what());
    return -1;
  }
#else
  (void)logits;
  (void)out_ids;
  (void)out_values;
  return 0;
#endif
}
