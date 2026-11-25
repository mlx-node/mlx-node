#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace mlx::core {
namespace {

constexpr char dflash2_topk16_source[] =
#include "cuda/dflash2_topk16.cu.inc"
    ;

constexpr char dflash2_selector_edges_source[] =
#include "cuda/dflash2_selector_edges.cu.inc"
    ;

bool try_uniform_row_stride(const array& a, int64_t& row_stride) noexcept {
  const int ndim = static_cast<int>(a.ndim());
  if (ndim < 2 || a.strides(ndim - 1) != 1) {
    return false;
  }
  const int64_t vocab = a.shape(ndim - 1);
  bool has_row_dim = false;
  int64_t expected_stride = 0;
  row_stride = vocab;
  for (int d = ndim - 2; d >= 0; --d) {
    const int64_t dim = a.shape(d);
    if (dim <= 1) {
      continue;
    }
    const int64_t st = a.strides(d);
    if (!has_row_dim) {
      if (st < 0 || st > std::numeric_limits<int64_t>::max() / dim) {
        return false;
      }
      row_stride = st;
      expected_stride = st * dim;
      has_row_dim = true;
    } else {
      if (st != expected_stride ||
          expected_stride > std::numeric_limits<int64_t>::max() / dim) {
        return false;
      }
      expected_stride *= dim;
    }
  }
  return true;
}

class DFlash2Topk16Cuda : public Primitive {
public:
  explicit DFlash2Topk16Cuda(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DFlash2 topk16 CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    int64_t row_stride = 0;
    array input = inputs[0];
    if (!try_uniform_row_stride(input, row_stride)) {
      auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
      input = std::move(owned[0]);
      row_stride = input.shape(-1);
    }
    auto& ids = outputs[0];
    auto& values = outputs[1];
    ids.set_data(cu::malloc_async(ids.nbytes(), encoder));
    values.set_data(cu::malloc_async(values.nbytes(), encoder));
    const int vocab = input.shape(-1);
    const int rows = static_cast<int>(input.size() / vocab);
    const int dtype = input.dtype() == bfloat16 ? 0 : input.dtype() == float16 ? 1 : 2;
    constexpr int kSliceSize = 8192;
    if (vocab > kSliceSize) {
      const int num_slices = (vocab + kSliceSize - 1) / kSliceSize;
      const int total_candidates = num_slices * 16;
      array part_ids(Shape{rows, total_candidates}, int32, nullptr, {});
      array part_values(Shape{rows, total_candidates}, float32, nullptr, {});
      part_ids.set_data(cu::malloc_async(part_ids.nbytes(), encoder));
      part_values.set_data(cu::malloc_async(part_values.nbytes(), encoder));
      encoder.add_temporary(part_ids);
      encoder.add_temporary(part_values);

      encoder.set_input_array(input);
      encoder.set_output_array(part_ids);
      encoder.set_output_array(part_values);
      static constexpr const char* kSliceEntries[3] = {
          "mlx_dflash2_topk16_slice<0>",
          "mlx_dflash2_topk16_slice<1>",
          "mlx_dflash2_topk16_slice<2>",
      };
      auto slice_kernel = cu::jit_kernel(
          encoder.device(), dflash2_topk16_source, kSliceEntries[dtype]);
      cu::KernelArgs sargs;
      sargs.append(input);
      sargs.append<int32_t>(vocab);
      sargs.append<int64_t>(row_stride);
      sargs.append<int32_t>(kSliceSize);
      sargs.append<int32_t>(num_slices);
      sargs.append(part_ids);
      sargs.append(part_values);
      encoder.add_kernel_node_raw(
          slice_kernel, dim3(num_slices, rows, 1), dim3(256, 1, 1), {}, 0,
          sargs.args());

      encoder.set_input_array(part_ids);
      encoder.set_input_array(part_values);
      encoder.set_output_array(ids);
      encoder.set_output_array(values);
      auto merge_kernel = cu::jit_kernel(
          encoder.device(), dflash2_topk16_source, "mlx_dflash2_topk16_merge");
      cu::KernelArgs margs;
      margs.append(part_ids);
      margs.append(part_values);
      margs.append<int32_t>(total_candidates);
      margs.append(ids);
      margs.append(values);
      encoder.add_kernel_node_raw(
          merge_kernel, dim3(rows, 1, 1), dim3(256, 1, 1), {}, 0, margs.args());
      return;
    }
    encoder.set_input_array(input);
    encoder.set_output_array(ids);
    encoder.set_output_array(values);
    cu::KernelArgs args;
    args.append(input);
    args.append<int32_t>(dtype);
    args.append<int32_t>(vocab);
    args.append<int64_t>(row_stride);
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
    const int rows = static_cast<int>(
        inputs[0].size() / static_cast<size_t>(inputs[0].shape(-1)));
    return {{rows, 16}, {rows, 16}};
  }
};

class DFlash2SelectorEdgesCuda : public UnaryPrimitive {
public:
  explicit DFlash2SelectorEdgesCuda(Stream stream, float softcap = 0.0f)
      : UnaryPrimitive(stream), softcap_(softcap) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("DFlash2SelectorEdgesCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    const auto& candidates = owned[0];
    const auto& unary = owned[1];
    const auto& hidden = owned[2];
    const auto& predecessor = owned[3];
    const auto& successor = owned[4];
    const auto& anchor = owned[5];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    for (const auto& arr : owned) {
      encoder.set_input_array(arr);
    }
    encoder.set_output_array(out);
    auto kernel = cu::jit_kernel(
        encoder.device(), dflash2_selector_edges_source,
        "mlx_dflash2_selector_edges_cuda_kernel");
    const int length = candidates.shape(1);
    const int vocab = predecessor.shape(0);
    const int unary_is_bf16 = unary.dtype() == bfloat16 ? 1 : 0;
    cu::KernelArgs args;
    args.append(candidates);
    args.append(unary);
    args.append(hidden);
    args.append(predecessor);
    args.append(successor);
    args.append(anchor);
    args.append(out);
    args.append<int32_t>(vocab);
    args.append<int32_t>(unary_is_bf16);
    args.append<float>(softcap_);
    encoder.add_kernel_node_raw(
        kernel, dim3(length, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(DFlash2SelectorEdgesCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const DFlash2SelectorEdgesCuda*>(&other);
    if (!rhs) {
      return false;
    }
    uint32_t lhs_bits = 0, rhs_bits = 0;
    std::memcpy(&lhs_bits, &softcap_, sizeof(lhs_bits));
    std::memcpy(&rhs_bits, &rhs->softcap_, sizeof(rhs_bits));
    return lhs_bits == rhs_bits;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {{inputs[0].shape(1), 16, 16}};
  }

private:
  float softcap_;
};

} // namespace
} // namespace mlx::core
#endif

extern "C" mlx_array* mlx_dflash2_selector_edges_cuda_softcap(
    mlx_array* candidates, mlx_array* unary, mlx_array* hidden,
    mlx_array* predecessor, mlx_array* successor, mlx_array* anchor,
    float softcap) {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!candidates || !unary || !hidden || !predecessor || !successor || !anchor ||
        default_device() != Device::gpu || !(softcap >= 0.0f && std::isfinite(softcap))) {
      return nullptr;
    }
    const auto& c = *reinterpret_cast<array*>(candidates);
    const auto& u = *reinterpret_cast<array*>(unary);
    const auto& h = *reinterpret_cast<array*>(hidden);
    const auto& p = *reinterpret_cast<array*>(predecessor);
    const auto& s = *reinterpret_cast<array*>(successor);
    const auto& a = *reinterpret_cast<array*>(anchor);
    if (c.dtype() != int32 || (u.dtype() != float32 && u.dtype() != bfloat16) ||
        h.dtype() != bfloat16 || p.dtype() != bfloat16 || s.dtype() != bfloat16 ||
        a.dtype() != int32 || c.ndim() != 3 || c.shape(0) != 1 ||
        c.shape(2) != 16 || c.shape(1) < 1 || c.shape(1) > 7 ||
        u.shape() != c.shape() || h.shape() != Shape{c.shape(1), 1, 256} ||
        p.ndim() != 2 || p.shape(0) < 16 || p.shape(1) != 256 ||
        s.shape() != p.shape() || a.shape() != Shape{1}) {
      return nullptr;
    }
    auto result = array(
        Shape{c.shape(1), 16, 16}, float32,
        std::make_shared<DFlash2SelectorEdgesCuda>(default_stream(Device::gpu), softcap),
        {c, u, h, p, s, a});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_dflash2_selector_edges_cuda_softcap: %s\n", error.what());
    return nullptr;
  }
#else
  (void)candidates; (void)unary; (void)hidden; (void)predecessor; (void)successor; (void)anchor;
  (void)softcap;
  return nullptr;
#endif
}

extern "C" mlx_array* mlx_dflash2_selector_edges_cuda(
    mlx_array* candidates, mlx_array* unary, mlx_array* hidden,
    mlx_array* predecessor, mlx_array* successor, mlx_array* anchor) {
  return mlx_dflash2_selector_edges_cuda_softcap(
      candidates, unary, hidden, predecessor, successor, anchor, 0.0f);
}

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
    const int64_t rows = static_cast<int64_t>(input.size() / static_cast<size_t>(vocab));
    if (rows < 1 || rows > std::numeric_limits<int>::max() ||
        input.size() != static_cast<size_t>(rows) * static_cast<size_t>(vocab)) {
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
