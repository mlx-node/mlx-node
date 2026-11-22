#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace mlx::core {
namespace {

constexpr char dflash2_select_tree_source[] =
#include "cuda/dflash2_select_tree.cu.inc"
    ;

class DFlash2SelectTreeCuda : public Primitive {
public:
  DFlash2SelectTreeCuda(
      Stream stream, uint32_t anchor, int32_t lookup_len, int32_t max_nodes,
      int32_t max_depth, float budget_slope)
      : Primitive(stream),
        anchor_(anchor),
        lookup_len_(lookup_len),
        max_nodes_(max_nodes),
        max_depth_(max_depth),
        budget_slope_(budget_slope) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DFlash2SelectTreeCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    const auto& candidates = owned[0];
    const auto& scores = owned[1];
    const auto& lookup_tokens = owned[2];
    const auto& lookup_rates = owned[3];
    const auto& budget_intercepts = owned[4];

    for (const auto& arr : owned) {
      encoder.set_input_array(arr);
    }
    for (auto& out : outputs) {
      out.set_data(cu::malloc_async(out.nbytes(), encoder));
      encoder.set_output_array(out);
    }

    const int32_t slots = static_cast<int32_t>(candidates.shape(1));
    const int32_t rates_by_agreement =
        (lookup_rates.ndim() == 2 && lookup_rates.shape(1) == 4) ? 1 : 0;

    auto kernel = cu::jit_kernel(
        encoder.device(), dflash2_select_tree_source,
        "mlx_dflash2_select_tree_cuda_kernel");
    cu::KernelArgs args;
    args.append(candidates);
    args.append(scores);
    args.append(lookup_tokens);
    args.append(lookup_rates);
    args.append(budget_intercepts);
    args.append(outputs[0]);
    args.append(outputs[1]);
    args.append(outputs[2]);
    args.append(outputs[3]);
    args.append(outputs[4]);
    args.append(outputs[5]);
    args.append<int32_t>(slots);
    args.append<uint32_t>(anchor_);
    args.append<int32_t>(lookup_len_);
    args.append<int32_t>(max_nodes_);
    args.append<int32_t>(max_depth_);
    args.append<float>(budget_slope_);
    args.append<int32_t>(rates_by_agreement);
    encoder.add_kernel_node_raw(
        kernel, dim3(1), dim3(128), {}, 0, args.args());
  }

  DEFINE_NAME(DFlash2SelectTreeCuda)

  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const DFlash2SelectTreeCuda*>(&other);
    if (!rhs) {
      return false;
    }
    uint32_t lhs_bits = 0, rhs_bits = 0;
    std::memcpy(&lhs_bits, &budget_slope_, sizeof(lhs_bits));
    std::memcpy(&rhs_bits, &rhs->budget_slope_, sizeof(rhs_bits));
    return anchor_ == rhs->anchor_ &&
        lookup_len_ == rhs->lookup_len_ &&
        max_nodes_ == rhs->max_nodes_ &&
        max_depth_ == rhs->max_depth_ &&
        lhs_bits == rhs_bits;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {
        {1, max_nodes_},
        {max_nodes_},
        {max_nodes_},
        {max_nodes_},
        {max_nodes_, 4},
        {max_nodes_, 4},
    };
  }

private:
  uint32_t anchor_;
  int32_t lookup_len_;
  int32_t max_nodes_;
  int32_t max_depth_;
  float budget_slope_;
};

} // namespace
} // namespace mlx::core
#endif

extern "C" bool mlx_dflash2_select_tree_cuda(
    mlx_array* candidates,
    mlx_array* scores,
    mlx_array* lookup_tokens,
    mlx_array* lookup_rates,
    mlx_array* budget_intercepts,
    uint32_t anchor,
    int32_t lookup_len,
    int32_t max_nodes,
    int32_t max_depth,
    float budget_slope,
    mlx_array** outputs) {
  if (outputs) {
    for (int i = 0; i < 6; ++i) {
      outputs[i] = nullptr;
    }
  }
#ifdef MLX_NODE_CUDA_ENABLED
  if (!outputs) {
    return false;
  }
  try {
    using namespace mlx::core;
    if (!candidates || !scores || !lookup_tokens || !lookup_rates ||
        !budget_intercepts || default_device() != Device::gpu) {
      return false;
    }
    if (lookup_len < 0 || lookup_len > 15 || max_nodes < 1 || max_nodes > 16 ||
        max_depth < 0 || max_depth > 15 || !std::isfinite(budget_slope)) {
      return false;
    }
    const auto& c = *reinterpret_cast<array*>(candidates);
    const auto& s = *reinterpret_cast<array*>(scores);
    const auto& lt = *reinterpret_cast<array*>(lookup_tokens);
    const auto& lr = *reinterpret_cast<array*>(lookup_rates);
    const auto& bi = *reinterpret_cast<array*>(budget_intercepts);

    if (c.dtype() != int32 || c.ndim() != 3 || c.shape(0) != 1 ||
        c.shape(1) < 1 || c.shape(1) > 7 || c.shape(2) != 16) {
      return false;
    }
    const int L = c.shape(1);
    if (s.dtype() != float32 || s.shape() != Shape{L, 16, 16}) {
      return false;
    }
    if ((lt.dtype() != int32 && lt.dtype() != uint32) || lt.shape() != Shape{16}) {
      return false;
    }
    const bool valid_lr = lr.dtype() == float32 &&
        (lr.shape() == Shape{16} || lr.shape() == Shape{16, 4});
    if (!valid_lr) {
      return false;
    }
    if (bi.dtype() != float32 || (bi.shape() != Shape{8} && bi.shape() != Shape{7})) {
      return false;
    }

    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        {{1, max_nodes},
         {max_nodes},
         {max_nodes},
         {max_nodes},
         {max_nodes, 4},
         {max_nodes, 4}},
        {uint32, int32, int32, uint32, int32, float32},
        std::make_shared<DFlash2SelectTreeCuda>(
            stream, anchor, lookup_len, max_nodes, max_depth, budget_slope),
        {c, s, lt, lr, bi});
    for (int i = 0; i < 6; ++i) {
      outputs[i] = reinterpret_cast<mlx_array*>(new array(std::move(results[i])));
    }
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_dflash2_select_tree_cuda: %s\n", error.what());
    return false;
  }
#else
  (void)candidates; (void)scores; (void)lookup_tokens; (void)lookup_rates;
  (void)budget_intercepts; (void)anchor; (void)lookup_len; (void)max_nodes;
  (void)max_depth; (void)budget_slope;
  return false;
#endif
}
