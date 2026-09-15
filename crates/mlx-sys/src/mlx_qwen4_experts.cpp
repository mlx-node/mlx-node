#include "mlx_common.h"
#ifdef MLX_NODE_METAL_ENABLED
namespace mlx::core::qwen4_preamble {
const char *gemm();
const char *quantized_utils();
const char *kquant();
} // namespace mlx::core::qwen4_preamble
namespace {
const std::string &expert_header() {
  static const std::string header =
      std::string(mlx::core::qwen4_preamble::gemm()) +
      mlx::core::qwen4_preamble::quantized_utils() +
      mlx::core::qwen4_preamble::kquant() +
#include "metal/qwen4_expert_decode.metal.inc"
      ;
  return header;
}

// All changing weights, slot mappings, scores and activations are inputs.
// Shapes specialize the two supported gate formats and two down formats.
std::vector<array> routed_experts(const std::vector<array> &a) {
  int tokens = a[0].shape(1), experts = a[3].shape(0) / 640;
  int gate_bits = a[3].shape(1) * 32 / 2560;
  int down_bits = a[9].shape(1) * 32 / 640;
  static auto gu = mlx::core::fast::metal_kernel(
      "qwen4_gate_up_decode", {"x", "ids", "wg", "sg", "bg", "wu", "su", "bu"},
      {"out"}, R"(
    q4_gate_up<T,BITS,2560,640,E,10>(x,ids,wg,sg,bg,wu,su,bu,out,
      threadgroup_position_in_grid.z,threadgroup_position_in_grid.y,
      simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
  )",
      expert_header());
  auto hidden = gu(
      {a[0], a[1], a[3], a[4], a[5], a[6], a[7], a[8]}, {{tokens * 10, 1, 640}},
      {mlx::core::float32}, {32, 160, tokens * 10}, {32, 2, 1},
      {{"T", mlx::core::bfloat16}, {"BITS", gate_bits}, {"E", experts}},
      std::nullopt, false, mlx::core::Device::gpu)[0];
  static auto down = mlx::core::fast::metal_kernel(
      "qwen4_down_combine_decode",
      {"x", "ids", "scores", "w", "scales", "biases"}, {"out"}, R"(
    threadgroup P products[10*4];
    q4_down_combine<T,P,BITS,2560,640,E,10,4,5>(x,ids,scores,w,scales,biases,out,products,
      threadgroup_position_in_grid.z,threadgroup_position_in_grid.y,
      simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
  )",
      expert_header());
  auto result =
      down({hidden, a[1], a[2], a[9], a[10], a[11]}, {{1, tokens, 2560}},
           {a[2].dtype()}, {32, 640 * 5, tokens}, {32, 5, 1},
           {{"T", mlx::core::bfloat16},
            {"P", a[2].dtype()},
            {"BITS", down_bits},
            {"E", experts}},
           std::nullopt, false, mlx::core::Device::gpu)[0];
  return {result};
}
} // namespace
#endif

extern "C" mlx_array *mlx_qwen4_routed_experts(mlx_array *x, mlx_array *ids,
                                               mlx_array *scores, mlx_array *wg,
                                               mlx_array *sg, mlx_array *bg,
                                               mlx_array *wu, mlx_array *su,
                                               mlx_array *bu, mlx_array *wd,
                                               mlx_array *sd, mlx_array *bd) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    std::vector<array> a;
    for (auto p : {x, ids, scores, wg, sg, bg, wu, su, bu, wd, sd, bd}) {
      if (!p)
        return nullptr;
      a.push_back(*reinterpret_cast<array *>(p));
    }
    if (a[0].ndim() != 3 || a[0].shape(0) != 1 || a[0].shape(1) < 1 ||
        a[0].shape(1) > 8 || a[0].shape(2) != 2560 ||
        a[0].dtype() != mlx::core::bfloat16 || a[1].ndim() != 1 ||
        a[1].size() != a[0].shape(1) * 10 ||
        a[1].dtype() != mlx::core::uint32 || a[2].size() != a[1].size() ||
        (a[2].dtype() != mlx::core::float32 &&
         a[2].dtype() != mlx::core::bfloat16))
      return nullptr;
    if (a[3].ndim() != 2 || a[3].shape(0) % 640)
      return nullptr;
    int experts = a[3].shape(0) / 640, bits = a[3].shape(1) * 32 / 2560;
    if (experts < 1 || experts > 512 || (bits != 4 && bits != 5))
      return nullptr;
    for (int base : {3, 6}) {
      if (a[base].shape() != Shape{experts * 640, 2560 * bits / 32} ||
          a[base].dtype() != mlx::core::uint32 ||
          a[base + 1].shape() != Shape{experts * 640, 160} ||
          a[base + 1].dtype() != mlx::core::uint8 ||
          a[base + 2].shape() != Shape{experts * 640, 20} ||
          a[base + 2].dtype() != mlx::core::float16)
        return nullptr;
    }
    if (a[9].ndim() != 2)
      return nullptr;
    int down_bits = a[9].shape(1) * 32 / 640;
    if ((down_bits != 5 && down_bits != 8) ||
        a[9].shape() != Shape{experts * 2560, 640 * down_bits / 32} ||
        a[9].dtype() != mlx::core::uint32 ||
        a[10].shape() != Shape{experts * 2560, 20} ||
        a[10].dtype() != mlx::core::float16 || a[11].shape() != a[10].shape() ||
        a[11].dtype() != mlx::core::float16)
      return nullptr;
    static auto fn = mlx::core::compile(routed_experts);
    auto out = fn(a)[0];
    // Opt-in evidence that a real decode reaches this specialization. All
    // changing inputs remain graph arguments; no model arrays are captured.
    static const bool trace =
        std::getenv("MLX_QWEN4_TRACE_FUSED_EXPERTS") != nullptr;
    static thread_local bool traced = false;
    if (trace && !traced) {
      std::cerr << "QWEN4_FUSED_EXPERTS tokens=" << a[0].shape(1)
                << " score_dtype=" << a[2].dtype() << " experts=" << experts
                << std::endl;
      traced = true;
    }
    return reinterpret_cast<mlx_array *>(new array(std::move(out)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 routed experts: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}
