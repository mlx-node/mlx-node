#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace mlx::core {
namespace {

constexpr char kv_window_tail_source[] =
#include "cuda/kv_window_tail.cu.inc"
    ;

struct KvTailCopyParams {
  int64_t src_strides[4];
  int64_t dst_strides[4];
  int32_t elems;
};

class KvStoreWindowTailCuda : public Primitive {
public:
  explicit KvStoreWindowTailCuda(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("KV window tail CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    for (int i = 0; i < 2; ++i) {
      const auto& full = in[i];
      const auto& view = in[2 + i];
      const auto& block = in[4 + i];
      if (full.ndim() != 4 || view.ndim() != 4 || block.ndim() != 4) {
        throw std::invalid_argument("KV window tail: replay rank changed");
      }
      const auto row_bytes = size_t(full.shape(3)) * full.itemsize();
      if (full.dtype() != bfloat16 || view.dtype() != bfloat16 ||
          block.dtype() != bfloat16 || full.shape(0) != 1 ||
          !full.flags().row_contiguous || full.shape(0) != view.shape(0) ||
          full.shape(1) != view.shape(1) || full.shape(3) != view.shape(3) ||
          full.shape(0) != block.shape(0) || full.shape(1) != block.shape(1) ||
          full.shape(3) != block.shape(3) || block.shape(2) < 1 ||
          block.shape(2) > view.shape(2) ||
          full.buffer().ptr() != view.buffer().ptr() ||
          view.strides() != full.strides() || view.offset() < full.offset() ||
          row_bytes == 0 || (view.offset() - full.offset()) % row_bytes != 0 ||
          (view.offset() - full.offset()) / row_bytes + view.shape(2) >
              size_t(full.shape(2)) ||
          full.buffer().ptr() == block.buffer().ptr()) {
        throw std::invalid_argument("KV window tail: invalid window alias or block");
      }
    }
    if (in[0].buffer().ptr() == in[1].buffer().ptr() ||
        in[0].buffer().ptr() == in[5].buffer().ptr() ||
        in[1].buffer().ptr() == in[4].buffer().ptr() ||
        in[2].shape() != in[3].shape() || in[4].shape() != in[5].shape()) {
      throw std::invalid_argument("KV window tail: K/V geometry or ownership differs");
    }

    for (int i = 0; i < 4; ++i) {
      out[i].copy_shared_buffer(in[i]);
    }

    auto& encoder = cu::get_command_encoder(stream());
    // Only bind the new K/V blocks (`in[4..5]`) as inputs; `out[0..3]` alias
    // `in[0..3]` via copy_shared_buffer and set_output_array already tracks them
    // without inflating CUDA Graph bytes_in_graph_ by the full ring buffers.
    encoder.set_input_array(in[4]);
    encoder.set_input_array(in[5]);
    for (const auto& output : out) {
      encoder.set_output_array(output);
    }

    KvTailCopyParams params[2]{};
    int max_elems = 0;
    for (int i = 0; i < 2; ++i) {
      const auto& block = in[4 + i];
      const auto& view = in[2 + i];
      params[i].elems = static_cast<int32_t>(block.size());
      max_elems = std::max(max_elems, params[i].elems);
      for (int axis = 0; axis < 4; ++axis) {
        params[i].src_strides[axis] = block.strides(axis);
        params[i].dst_strides[axis] = view.strides(axis);
      }
    }
    const int rows = in[4].shape(2);
    const int offset = in[2].shape(2) - rows;
    const int dims = in[4].shape(3);

    cu::KernelArgs args;
    args.append(in[4]);
    args.append(in[5]);
    args.append(out[2]);
    args.append(out[3]);
    args.append_ptr(&params[0]);
    args.append_ptr(&params[1]);
    args.append<int32_t>(rows);
    args.append<int32_t>(offset);
    args.append<int32_t>(dims);
    auto kernel = cu::jit_kernel(
        encoder.device(), kv_window_tail_source, "mlx_kv_window_tail_cuda_kernel");
    encoder.add_kernel_node_raw(
        kernel, dim3((max_elems + 255) / 256, 2), dim3(256), {}, 0,
        args.args());
  }

  DEFINE_NAME(KvStoreWindowTailCuda)
  bool is_equivalent(const Primitive&) const override {
    return true;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& in) override {
    return {in[0].shape(), in[1].shape(), in[2].shape(), in[3].shape()};
  }
};

} // namespace
} // namespace mlx::core
#endif

extern "C" bool mlx_kv_store_window_tail_cuda(mlx_array* const* args, mlx_array** out) {
  for (int i = 0; out && i < 4; ++i) {
    out[i] = nullptr;
  }
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!args || !out || default_device() != Device::gpu) {
      return false;
    }
    std::vector<array> inputs;
    for (int i = 0; i < 6; ++i) {
      if (!args[i]) {
        return false;
      }
      const auto& a = *reinterpret_cast<array*>(args[i]);
      if (a.ndim() != 4 || a.dtype() != bfloat16 || a.shape(0) != 1) {
        return false;
      }
      inputs.push_back(a);
    }
    std::vector<Shape> shapes;
    for (int i = 0; i < 4; ++i) {
      shapes.push_back(inputs[i].shape());
    }
    auto outputs = array::make_arrays(
        std::move(shapes), std::vector<Dtype>(4, bfloat16),
        std::make_shared<KvStoreWindowTailCuda>(default_stream(Device::gpu)),
        std::move(inputs));
    for (int i = 0; i < 4; ++i) {
      out[i] = reinterpret_cast<mlx_array*>(new array(std::move(outputs[i])));
    }
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_kv_store_window_tail_cuda: %s\n", error.what());
    return false;
  }
#else
  (void)args;
  return false;
#endif
}
