#include "mlx_common.h"
#include "mlx_qwen4_flags.h"
#ifdef MLX_NODE_METAL_ENABLED
#include "metal/common/native_activations.h"
#include "metal/common/quantized.h"
namespace {
using mlx::core::metal::native_bf16_silu_table;
using mlx::core::quantized_preamble::qmv_header;

const std::string &expert_header() {
  static const std::string header = qmv_header() +
#include "metal/common/precise_sigmoid.metal.inc"
#include "metal/common/quantized_expert_decode.metal.inc"
      ;
  return header;
}

// A native sigmoid can choose a different exponential approximation from
// metal::precise::exp at a BF16 halfway point (for example -6.84375).
// Generate all 65536 BF16 results with the same compiled primitive used by
// sigmoid_mul, then reuse this immutable 128 KiB table in the fused kernels.
const array &native_bf16_sigmoid_table() {
  static const array table = [] {
    std::vector<uint16_t> bits(65536);
    for (size_t i = 0; i < bits.size(); ++i)
      bits[i] = static_cast<uint16_t>(i);
    auto values = mlx::core::view(
        array(bits.data(), {65536}, mlx::core::uint16), mlx::core::bfloat16);
    // A lone sigmoid can remain a precompiled unary kernel. Include the
    // dynamic multiplier, as sigmoid_mul does, to use its fused JIT kernel.
    static auto fn = mlx::core::compile(
        [](const std::vector<array> &a) {
          return std::vector<array>{a[1] * mlx::core::sigmoid(a[0])};
        },
        true);
    auto result =
        fn({values, mlx::core::ones({65536}, mlx::core::bfloat16)})[0];
    mlx::core::eval({result});
    return result;
  }();
  return table;
}

// The injection path uses the uncompiled sigmoid after its BF16 division.
// Include both scalar boundaries in the lookup; the compiled mixer sigmoid
// table above intentionally represents a different native operation.
const array &native_bf16_inject_table() {
  static const array table = [] {
    std::vector<uint16_t> bits(65536);
    for (size_t i = 0; i < bits.size(); ++i)
      bits[i] = uint16_t(i);
    auto values = mlx::core::view(
        array(bits.data(), {65536}, mlx::core::uint16), mlx::core::bfloat16);
    auto result = mlx::core::sigmoid(values / array(4, mlx::core::bfloat16)) *
                  array(2, mlx::core::bfloat16);
    mlx::core::eval({result});
    return result;
  }();
  return table;
}

// All changing weights, slot mappings, scores and activations are inputs.
// Shapes specialize the two supported gate formats and two down formats.
template <bool LANES>
std::vector<array> routed_experts(const std::vector<array> &a) {
  int tokens = a[0].shape(1), experts = a[3].shape(0) / 640;
  int gate_bits = a[3].shape(1) * 32 / 2560;
  int down_bits = a[9].shape(1) * 32 / 640;
  static auto gu = mlx::core::fast::metal_kernel(
      "qwen4_gate_up_decode", {"x", "ids", "wg", "sg", "bg", "wu", "su", "bu"},
      {"out"}, R"(
    quantized_expert_gate_up<T,BITS,2560,640,E,10,LANES>(x,ids,wg,sg,bg,wu,su,bu,out,
      threadgroup_position_in_grid.z,threadgroup_position_in_grid.y,
      simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
  )",
      expert_header());
  auto hidden = gu({a[0], a[1], a[3], a[4], a[5], a[6], a[7], a[8]},
                   {{tokens * 10, 1, 640}}, {mlx::core::float32},
                   {32, 160, tokens * 10}, {32, 2, 1},
                   {{"T", mlx::core::bfloat16},
                    {"BITS", gate_bits},
                    {"E", experts},
                    {"LANES", LANES}},
                   std::nullopt, false, mlx::core::Device::gpu)[0];
  static auto down = mlx::core::fast::metal_kernel(
      "qwen4_down_combine_decode",
      {"x", "ids", "scores", "w", "scales", "biases"}, {"out"}, R"(
    threadgroup P products[10*4];
    affine_expert_down_combine<T,P,BITS,2560,640,E,10,4,5,LANES>(x,ids,scores,w,scales,biases,out,products,
      threadgroup_position_in_grid.z,threadgroup_position_in_grid.y,
      simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
  )",
      expert_header());
  auto result =
      down({hidden, a[1], a[2], a[9], a[10], a[11]}, {{1, tokens, 2560}},
           {a[2].dtype()}, {32, 640 * 5, tokens}, {32, 5, 1},
           {{"T", mlx::core::bfloat16},
            {"P", a[2].dtype()},
            {"LANES", LANES},
            {"BITS", down_bits},
            {"E", experts}},
           std::nullopt, false, mlx::core::Device::gpu)[0];
  return {result};
}

const std::string &shared_expert_header() {
  static const std::string header = expert_header() +
#include "metal/qwen4/shared_expert_decode.metal.inc"
      ;
  return header;
}

template <int RPS, bool STAGE, bool LANES>
std::vector<array> routed_shared_experts(const std::vector<array> &a) {
  const int experts = a[3].shape(0) / 640;
  const int gate_bits = a[3].shape(1) * 32 / 2560;
  const int down_bits = a[9].shape(1) * 32 / 640;
  static auto gu = mlx::core::fast::metal_kernel(
      "qwen4_routed_shared_gu",
      {"x", "ids", "wg", "sg", "bg", "wu", "su", "bu", "swg", "ssg", "sbg",
       "swu", "ssu", "sbu", "sigmoid_table"},
      {"out"}, R"(
    if (threadgroup_position_in_grid.z < 10) {
      quantized_expert_gate_up<T,BITS,2560,640,E,10,LANES>(x,ids,wg,sg,bg,wu,su,bu,out,
        threadgroup_position_in_grid.z,threadgroup_position_in_grid.y,
        simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
    } else {
      q4_shared_gu<T,LANES>(x,swg,ssg,sbg,swu,ssu,sbu,sigmoid_table,out,
        threadgroup_position_in_grid.y,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
    }
  )",
      shared_expert_header());
  auto hidden =
      gu({a[0], a[1], a[3], a[4], a[5], a[6], a[7], a[8], a[12], a[13], a[14],
          a[15], a[16], a[17], a[22]},
         {{11, 1, 640}}, {mlx::core::float32}, {32, 160, 11}, {32, 2, 1},
         {{"T", mlx::core::bfloat16},
          {"BITS", gate_bits},
          {"E", experts},
          {"LANES", LANES}},
         std::nullopt, false, mlx::core::Device::gpu)[0];
  static auto down = mlx::core::fast::metal_kernel(
      "qwen4_routed_shared_down",
      {"x", "ids", "scores", "w", "scales", "biases", "sw", "ss", "sb", "gate",
       "sigmoid_table"},
      {"out"}, R"(
    threadgroup P products[10*RPS];
    threadgroup T shared_products[RPS];
    q4_down_combine_shared<T,P,BITS,2560,640,E,10,RPS,5,STAGE>(
      x,ids,scores,w,scales,biases,sw,ss,sb,gate[0],sigmoid_table,out,products,shared_products,
      0,threadgroup_position_in_grid.y,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
  )",
      shared_expert_header());
  return down({hidden, a[1], a[2], a[9], a[10], a[11], a[18], a[19], a[20],
               a[21], a[22]},
              {{1, 1, 2560}}, {mlx::core::bfloat16}, {32, (2560 / RPS) * 6, 1},
              {32, 6, 1},
              {{"T", mlx::core::bfloat16},
               {"P", a[2].dtype()},
               {"RPS", RPS},
               {"STAGE", STAGE || LANES},
               {"BITS", down_bits},
               {"E", experts}},
              std::nullopt, false, mlx::core::Device::gpu);
}
} // namespace
#endif

// Port of TrackFastKernels2.mixSource at mlxfast 8981cef5. The GGUF
// checkpoint keeps the four-stream mean outside its weights, so retain the
// final divide here. Use this build's native sigmoid table at BF16 boundaries.
extern "C" mlx_array *mlx_qwen4_prefill_hc_mix(mlx_array *up,
                                               mlx_array *normed) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!up || !normed)
      return nullptr;
    const auto &w = *reinterpret_cast<array *>(up);
    const auto &n = *reinterpret_cast<array *>(normed);
    if (w.ndim() != 3 || w.shape(0) != 1 || w.shape(1) <= 8 ||
        w.shape(1) > 1024 || w.shape(2) != 10240 ||
        w.dtype() != mlx::core::bfloat16 || n.shape() != w.shape() ||
        n.dtype() != w.dtype())
      return nullptr;
    static auto fn = mlx::core::compile([](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_prefill_hc_mix", {"w", "normed", "sigmoid_table"}, {"out"},
          R"(
        const uint d = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (d >= 2560) return;
        T acc = T(0);
        for (int s = 0; s < 4; ++s) {
          const uint i = row * 10240 + s * 2560 + d;
          T sg = sigmoid_table[as_type<ushort>(w[i])];
          T p = sg * normed[i];
          acc = acc + p;
        }
        out[row * 2560 + d] = acc / T(4);
      )");
      return kernel(a, {{1, a[0].shape(1), 2560}}, {a[0].dtype()},
                    {2560, a[0].shape(1), 1}, {256, 1, 1},
                    {{"T", a[0].dtype()}}, std::nullopt, false,
                    mlx::core::default_stream(mlx::core::Device::gpu));
    });
    auto result = fn({w, n, native_bf16_sigmoid_table()});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 prefill mixer combine: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

extern "C" mlx_array *mlx_qwen4_dense_decode(mlx_array *input,
                                             mlx_array *weight,
                                             mlx_array *scales,
                                             mlx_array *biases) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !weight || !scales || !biases)
      return nullptr;
    const auto &x = *reinterpret_cast<array *>(input);
    if (x.ndim() < 2 || x.dtype() != mlx::core::bfloat16 || x.shape(-1) < 256 ||
        x.size() != x.shape(-1))
      return nullptr;
    const auto &w = *reinterpret_cast<array *>(weight);
    const auto &s = *reinterpret_cast<array *>(scales);
    const auto &b = *reinterpret_cast<array *>(biases);
    const int k = x.shape(-1);
    if (w.ndim() != 2 || w.dtype() != mlx::core::uint32 || k % 32 ||
        w.shape(0) < 4 || w.shape(0) % 4 || w.shape(1) != k / 4 ||
        s.dtype() != mlx::core::float16 || b.dtype() != mlx::core::float16 ||
        s.shape() != Shape{w.shape(0), k / 32} || b.shape() != s.shape())
      return nullptr;
    // Shapes/dtypes specialize the graph; every mutable tensor is an input.
    // Reuse the custom primitive instead of rebuilding its Metal source for
    // every projection. This never caches a weight value or activation.
    static auto graph = [](const std::vector<array> &in) {
      const auto &x = in[0], &w = in[1], &s = in[2], &b = in[3];
      const int k = x.shape(-1), n = w.shape(0);
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_dense_decode",
          {"x", "w", "scales", "biases", "normed", "sigmoid_table"}, {"out"},
#include "metal/qwen4/dense_decode.metal.inc"
          , qmv_header());
      auto shape = x.shape();
      shape.back() = n;
      return kernel({x, w, s, b, x, x}, {shape}, {mlx::core::bfloat16},
                    {32, 2 * ((n + 7) / 8), 1}, {32, 2, 1},
                    {{"T", mlx::core::bfloat16},
                     {"K", k},
                     {"N", n},
                     {"FAST", n % 8 == 0 && k % 512 == 0},
                     {"RPS", 4},
                     {"ACT", false},
                     {"MIX", false}},
                    std::nullopt, false, mlx::core::Device::gpu);
    };
    static auto compiled = mlx::core::compile(graph);
    auto setting = qwen4_env("MLX_QWEN4_CACHED_KERNEL_GRAPHS");
    const bool cached = (!setting || std::string(setting) != "0");
    auto result = cached ? compiled({x, w, s, b}) : graph({x, w, s, b});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 compact dense decode: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

extern "C" mlx_array *mlx_qwen4_decode_mixer_act(mlx_array *input,
                                                 mlx_array *weight,
                                                 mlx_array *scales,
                                                 mlx_array *biases) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !weight || !scales || !biases)
      return nullptr;
    const auto &x = *reinterpret_cast<array *>(input);
    const auto &w = *reinterpret_cast<array *>(weight);
    const auto &s = *reinterpret_cast<array *>(scales);
    const auto &b = *reinterpret_cast<array *>(biases);
    if (x.shape() != Shape{1, 1, 10240} || x.dtype() != mlx::core::bfloat16 ||
        w.shape() != Shape{320, 2560} || w.dtype() != mlx::core::uint32 ||
        s.shape() != Shape{320, 320} || s.dtype() != mlx::core::float16 ||
        b.shape() != s.shape() || b.dtype() != s.dtype())
      return nullptr;
    static auto graph = [](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_decode_mixer_act",
          {"x", "w", "scales", "biases", "normed", "sigmoid_table"}, {"out"},
#include "metal/qwen4/dense_decode.metal.inc"
          , qmv_header());
      return kernel({a[0], a[1], a[2], a[3], a[0], a[4]}, {{1, 1, 320}},
                    {mlx::core::bfloat16}, {32, 320, 1}, {32, 2, 1},
                    {{"T", mlx::core::bfloat16},
                     {"K", 10240},
                     {"N", 320},
                     {"FAST", true},
                     {"MIX", false},
                     {"RPS", 1},
                     {"ACT", true}},
                    std::nullopt, false, mlx::core::Device::gpu);
    };
    static auto compiled = mlx::core::compile(graph);
    const auto &table = native_bf16_silu_table();
    auto result = compiled({x, w, s, b, table});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 reference decode mixer activation: " << e.what()
              << std::endl;
  }
#endif
  return nullptr;
}

// Fuse the two projections without concatenating or retaining another bank.
extern "C" bool mlx_qwen4_mixer_down_inject(
    mlx_array *input, mlx_array *down, mlx_array *down_scales,
    mlx_array *down_biases, mlx_array *inject, mlx_array *inject_scales,
    mlx_array *inject_biases, mlx_array **out, mlx_array **out_inject) {
  if (!out || !out_inject)
    return false;
  *out = nullptr;
  *out_inject = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !down || !down_scales || !down_biases || !inject ||
        !inject_scales || !inject_biases)
      return false;
    const auto &x = *reinterpret_cast<array *>(input);
    const auto &wd = *reinterpret_cast<array *>(down);
    const auto &sd = *reinterpret_cast<array *>(down_scales);
    const auto &bd = *reinterpret_cast<array *>(down_biases);
    const auto &wi = *reinterpret_cast<array *>(inject);
    const auto &si = *reinterpret_cast<array *>(inject_scales);
    const auto &bi = *reinterpret_cast<array *>(inject_biases);
    auto valid = [](const array &w, const array &s, const array &b, int n) {
      return w.shape() == Shape{n, 2560} && w.dtype() == mlx::core::uint32 &&
             s.shape() == Shape{n, 320} && s.dtype() == mlx::core::float16 &&
             b.shape() == s.shape() && b.dtype() == s.dtype();
    };
    if (x.shape() != Shape{1, 1, 10240} || x.dtype() != mlx::core::bfloat16 ||
        !valid(wd, sd, bd, 320) || !valid(wi, si, bi, 4))
      return false;
    static auto compiled = mlx::core::compile([](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_mixer_down_inject",
          {"x", "wd", "sd", "bd", "wi", "si", "bi", "silu_table"},
          {"act", "injection"},
#include "metal/qwen4/mixer_down_inject.metal.inc"
          , qmv_header());
      return kernel(a, {{1, 1, 320}, {1, 1, 4}},
                    {mlx::core::bfloat16, mlx::core::bfloat16}, {32, 324, 1},
                    {32, 2, 1}, {{"T", mlx::core::bfloat16}}, std::nullopt,
                    false, mlx::core::Device::gpu);
    });
    static auto split_compiled = mlx::core::compile([](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_mixer_split_k",
          {"x", "wd", "sd", "bd", "wi", "si", "bi", "silu_table"},
          {"act", "injection"},
#include "metal/qwen4/mixer_split_k.metal.inc"
          , qmv_header());
      return kernel(a, {{1, 1, 320}, {1, 1, 4}},
                    {mlx::core::bfloat16, mlx::core::bfloat16}, {32, 656, 1},
                    {32, 4, 1}, {{"T", mlx::core::bfloat16}}, std::nullopt,
                    false, mlx::core::Device::gpu);
    });
    const auto split = qwen4_env("MLX_QWEN4_MIXER_SPLIT_K");
    auto result = (split && std::string(split) == "1" ? split_compiled : compiled)(
        {x, wd, sd, bd, wi, si, bi, native_bf16_silu_table()});
    auto activation = std::make_unique<array>(std::move(result[0]));
    auto injection = std::make_unique<array>(std::move(result[1]));
    *out = reinterpret_cast<mlx_array *>(activation.release());
    *out_inject = reinterpret_cast<mlx_array *>(injection.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 mixer down/injection: " << e.what() << std::endl;
  }
#endif
  return false;
}

extern "C" mlx_array *mlx_qwen4_hyper_up(mlx_array *input, mlx_array *weight,
                                         mlx_array *scales, mlx_array *biases,
                                         mlx_array *normed) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !weight || !scales || !biases || !normed)
      return nullptr;
    const auto &x = *reinterpret_cast<array *>(input);
    const auto &w = *reinterpret_cast<array *>(weight);
    const auto &s = *reinterpret_cast<array *>(scales);
    const auto &b = *reinterpret_cast<array *>(biases);
    const auto &n = *reinterpret_cast<array *>(normed);
    if (x.shape() != Shape{1, 1, 320} || x.dtype() != mlx::core::bfloat16 ||
        n.shape() != Shape{1, 1, 10240} || n.dtype() != x.dtype() ||
        w.shape() != Shape{10240, 80} || w.dtype() != mlx::core::uint32 ||
        s.shape() != Shape{10240, 10} || s.dtype() != mlx::core::float16 ||
        b.shape() != s.shape() || b.dtype() != s.dtype())
      return nullptr;
    static auto graph = [](const std::vector<array> &in, bool parallel) {
      const auto &x = in[0], &w = in[1], &s = in[2], &b = in[3], &n = in[4];
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_hyper_up",
          {"x", "w", "scales", "biases", "normed", "sigmoid_table"}, {"out"},
#include "metal/qwen4/dense_decode.metal.inc"
          , qmv_header());
      return kernel({x, w, s, b, n, in[5]}, {{1, 1, 2560}}, {x.dtype()},
                    {32, 2560, 1}, {32, 2, 1},
                    {{"T", x.dtype()},
                     {"K", 320},
                     {"N", 10240},
                     {"FAST", false},
                     {"RPS", 4},
                     {"ACT", false},
                     {"MIX", parallel ? 2 : 1}},
                    std::nullopt, false, mlx::core::Device::gpu);
    };
    static auto compiled = mlx::core::compile(
        [](const std::vector<array> &a) { return graph(a, false); });
    static auto parallel_compiled = mlx::core::compile(
        [](const std::vector<array> &a) { return graph(a, true); });
    const auto lanes = qwen4_env("MLX_QWEN4_MIXER_LANE_PRODUCTS");
    const bool parallel = lanes && std::string(lanes) == "1";
    auto setting = qwen4_env("MLX_QWEN4_CACHED_KERNEL_GRAPHS");
    const bool cached = (!setting || std::string(setting) != "0");
    const auto &table = native_bf16_sigmoid_table();
    auto result = cached ? (parallel ? parallel_compiled
                                     : compiled)({x, w, s, b, n, table})
                         : graph({x, w, s, b, n, table}, parallel);
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 hyper up: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

// TrackFastMixer.upMix emits the injection gate in the same dispatch. Keep
// the local Q8 projection and both sigmoid arithmetic paths unchanged.
extern "C" bool mlx_qwen4_hyper_up_inject(
    mlx_array *input, mlx_array *weight, mlx_array *scales, mlx_array *biases,
    mlx_array *normed, mlx_array *injection, mlx_array **out,
    mlx_array **out_gate) {
  if (!out || !out_gate) return false;
  *out = nullptr;
  *out_gate = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !weight || !scales || !biases || !normed || !injection)
      return false;
    const auto &x = *reinterpret_cast<array *>(input);
    const auto &w = *reinterpret_cast<array *>(weight);
    const auto &s = *reinterpret_cast<array *>(scales);
    const auto &b = *reinterpret_cast<array *>(biases);
    const auto &n = *reinterpret_cast<array *>(normed);
    const auto &g = *reinterpret_cast<array *>(injection);
    if (x.shape() != Shape{1, 1, 320} || x.dtype() != mlx::core::bfloat16 ||
        n.shape() != Shape{1, 1, 10240} || n.dtype() != x.dtype() ||
        g.shape() != Shape{1, 1, 4} || g.dtype() != x.dtype() ||
        w.shape() != Shape{10240, 80} || w.dtype() != mlx::core::uint32 ||
        s.shape() != Shape{10240, 10} || s.dtype() != mlx::core::float16 ||
        b.shape() != s.shape() || b.dtype() != s.dtype())
      return false;
    static auto graph = [](const std::vector<array> &a, int mode) {
      static const std::string epilogue = R"(
        if (threadgroup_position_in_grid.y == 0 &&
            simdgroup_index_in_threadgroup == 0 && lane < 4) {
          inject_out[lane] = inject_table[as_type<ushort>(inject_projection[lane])];
        }
      )";
      static const std::string source = std::string(
#include "metal/qwen4/dense_decode.metal.inc"
          ) + epilogue;
      static const std::string columns_source = std::string(
#include "metal/qwen4/mixer_up_columns.metal.inc"
          ) + epilogue;
      static auto columns_kernel = mlx::core::fast::metal_kernel(
          "qwen4_mixer_up_columns",
          {"x", "w", "scales", "biases", "normed", "sigmoid_table",
           "inject_projection", "inject_table"},
          {"out", "inject_out"}, columns_source, qmv_header());
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_hyper_up_inject",
          {"x", "w", "scales", "biases", "normed", "sigmoid_table",
           "inject_projection", "inject_table"},
          {"out", "inject_out"}, source, qmv_header());
      auto &selected = mode == 2 ? columns_kernel : kernel;
      return selected(a, {{1, 1, 2560}, {1, 1, 4}},
                    {mlx::core::bfloat16, mlx::core::bfloat16}, {32, 2560, 1},
                    {32, 2, 1},
                    {{"T", mlx::core::bfloat16},
                     {"K", 320},
                     {"N", 10240},
                     {"FAST", false},
                     {"RPS", 4},
                     {"ACT", false},
                     {"MIX", mode == 1 ? 2 : 1}},
                    std::nullopt, false, mlx::core::Device::gpu);
    };
    static auto compiled = mlx::core::compile(
        [](const std::vector<array> &a) { return graph(a, 0); });
    static auto parallel_compiled = mlx::core::compile(
        [](const std::vector<array> &a) { return graph(a, 1); });
    static auto columns_compiled = mlx::core::compile(
        [](const std::vector<array> &a) { return graph(a, 2); });
    const auto columns = qwen4_env("MLX_QWEN4_MIXER_UP_COLUMNS");
    const bool two_columns = columns && std::string(columns) == "1";
    const auto lanes = qwen4_env("MLX_QWEN4_MIXER_LANE_PRODUCTS");
    const bool parallel = lanes && std::string(lanes) == "1";
    auto result = (two_columns ? columns_compiled
                               : (parallel ? parallel_compiled : compiled))(
        {x, w, s, b, n, native_bf16_sigmoid_table(), g,
         native_bf16_inject_table()});
    auto mixed = std::make_unique<array>(std::move(result[0]));
    auto gate = std::make_unique<array>(std::move(result[1]));
    *out = reinterpret_cast<mlx_array *>(mixed.release());
    *out_gate = reinterpret_cast<mlx_array *>(gate.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 mixer injection: " << e.what() << std::endl;
  }
#endif
  return false;
}

extern "C" mlx_array *mlx_qwen4_sorted_shared_combine(
    mlx_array *values, mlx_array *scores, mlx_array *inverse,
    mlx_array *shared, mlx_array *gate, int top) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!values || !scores || !inverse || !shared || !gate || top != 10)
      return nullptr;
    const auto &v = *reinterpret_cast<array *>(values);
    const auto &s = *reinterpret_cast<array *>(scores);
    const auto &order = *reinterpret_cast<array *>(inverse);
    const auto &sh = *reinterpret_cast<array *>(shared);
    const auto &g = *reinterpret_cast<array *>(gate);
    if (v.ndim() != 2 || v.shape(1) != 2560 || v.shape(0) % top ||
        v.dtype() != mlx::core::bfloat16 ||
        s.size() != v.shape(0) || order.size() != v.shape(0) ||
        (s.dtype() != mlx::core::bfloat16 && s.dtype() != mlx::core::float32) ||
        (order.dtype() != mlx::core::uint32 && order.dtype() != mlx::core::int32))
      return nullptr;
    const int tokens = v.shape(0) / top;
    if (tokens <= 8 || tokens > 1024 || sh.shape() != Shape{1, tokens, 2560} ||
        g.shape() != Shape{1, tokens, 1} || sh.dtype() != v.dtype() ||
        g.dtype() != v.dtype())
      return nullptr;
    static auto graph = [](const std::vector<array> &a) {
      const int tokens = a[0].shape(0) / 10;
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_sorted_shared_combine",
          {"values", "scores", "inverse", "shared", "gate", "sigmoid_table"},
          {"out"},
#include "metal/common/sorted_expert_shared_combine.metal.inc"
          );
      return kernel(a, {{1, tokens, 2560}}, {mlx::core::bfloat16},
                    {tokens * 2560, 1, 1}, {256, 1, 1},
                    {{"T", mlx::core::bfloat16}, {"P", a[1].dtype()},
                     {"TOKENS", tokens}, {"H", 2560},
                     {"ROWS", tokens * 10}, {"TOP", 10}},
                    std::nullopt, false, mlx::core::Device::gpu);
    };
    static auto compiled = mlx::core::compile(graph);
    auto result = compiled({v, s, order, sh, g, native_bf16_sigmoid_table()});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 sorted shared combine: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

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
    const auto lanes = qwen4_env("MLX_QWEN4_EXPERT_LANE_STAGING");
    auto out = [&] {
      if (lanes && std::string(lanes) == "1") {
        static auto fn = mlx::core::compile(routed_experts<true>);
        return fn(a)[0];
      }
      static auto fn = mlx::core::compile(routed_experts<false>);
      return fn(a)[0];
    }();
    // Opt-in evidence that a real decode reaches this specialization. All
    // changing inputs remain graph arguments; no model arrays are captured.
    static const bool trace =
        qwen4_env("MLX_QWEN4_TRACE_FUSED_EXPERTS") != nullptr;
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

// Read SDPA's head-major result and the interleaved Q/gate projection without
// materializing the two token-major reshape buffers before the pointwise op.
extern "C" mlx_array *mlx_qwen4_attention_gate(mlx_array *attention,
                                              mlx_array *projection) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!attention || !projection) return nullptr;
    const auto &att = *reinterpret_cast<array *>(attention);
    const auto &qg = *reinterpret_cast<array *>(projection);
    if (att.ndim() != 4 || att.shape(0) != 1 || att.shape(1) < 1 ||
        att.shape(1) > 128 || att.shape(2) < 1 || att.shape(2) > 1024 ||
        att.shape(3) != 256 || att.dtype() != mlx::core::bfloat16 ||
        qg.shape() != Shape{1, att.shape(2), att.shape(1), 512} ||
        qg.dtype() != att.dtype()) return nullptr;
    static auto graph = [](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_attention_gate", {"att", "qg", "sigmoid_table"}, {"out"},
#include "metal/common/attention_gate_bf16.metal.inc"
          , "", false);
      const int heads = a[0].shape(1), tokens = a[0].shape(2);
      return kernel(a, {{1, tokens, heads * 256}}, {a[0].dtype()},
                    {heads * 256, tokens, 1}, {256, 1, 1},
                    {{"T", a[0].dtype()}, {"H", heads}, {"D", 256}},
                    std::nullopt, false, mlx::core::Device::gpu);
    };
    static auto compiled = mlx::core::compile(graph);
    auto result = compiled({att, qg, native_bf16_sigmoid_table()});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 attention output gate: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

// 0..11 are the existing routed-expert inputs, 12..20 the shared banks,
// and 21 the BF16 shared gate. All arrays remain graph inputs on replay.
extern "C" mlx_array *mlx_qwen4_routed_shared_experts(mlx_array *const *inputs,
                                                      size_t count) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!inputs || count != 22)
      return nullptr;
    std::vector<array> a;
    for (size_t i = 0; i < count; ++i) {
      if (!inputs[i])
        return nullptr;
      a.push_back(*reinterpret_cast<array *>(inputs[i]));
    }
    if (a[0].shape() != Shape{1, 1, 2560} ||
        a[0].dtype() != mlx::core::bfloat16 || a[1].shape() != Shape{10} ||
        a[1].dtype() != mlx::core::uint32 || a[2].size() != 10 ||
        (a[2].dtype() != mlx::core::bfloat16 &&
         a[2].dtype() != mlx::core::float32) ||
        a[21].shape() != Shape{1, 1, 1} ||
        a[21].dtype() != mlx::core::bfloat16 || a[3].ndim() != 2 ||
        a[3].shape(0) % 640 || a[9].ndim() != 2)
      return nullptr;
    const int experts = a[3].shape(0) / 640, bits = a[3].shape(1) * 32 / 2560;
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
    const int db = a[9].shape(1) * 32 / 640;
    if ((db != 5 && db != 8) ||
        a[9].shape() != Shape{experts * 2560, 640 * db / 32} ||
        a[9].dtype() != mlx::core::uint32 ||
        a[10].shape() != Shape{experts * 2560, 20} ||
        a[10].dtype() != mlx::core::float16 || a[11].shape() != a[10].shape() ||
        a[11].dtype() != mlx::core::float16)
      return nullptr;
    for (int base : {12, 15, 18}) {
      const int n = base == 18 ? 2560 : 640, k = base == 18 ? 640 : 2560;
      if (a[base].shape() != Shape{n, k / 4} ||
          a[base].dtype() != mlx::core::uint32 ||
          a[base + 1].shape() != Shape{n, k / 32} ||
          a[base + 1].dtype() != mlx::core::float16 ||
          a[base + 2].shape() != a[base + 1].shape() ||
          a[base + 2].dtype() != mlx::core::float16)
        return nullptr;
    }
    a.push_back(native_bf16_sigmoid_table());
    // TrackFastMoE's singleton two-row tiles and distributed staging.
    // Keep the four-row schedule available for a paired whole-model check.
    const auto schedule = qwen4_env("MLX_QWEN4_REFERENCE_DOWN_SCHEDULE");
    const auto lanes = qwen4_env("MLX_QWEN4_EXPERT_LANE_STAGING");
    array out = [&] {
      if (lanes && std::string(lanes) == "1") {
        if (schedule && std::string(schedule) == "1") {
          static auto compiled =
              mlx::core::compile(routed_shared_experts<2, true, true>);
          return compiled(a)[0];
        }
        static auto compiled =
            mlx::core::compile(routed_shared_experts<4, false, true>);
        return compiled(a)[0];
      }
      if (schedule && std::string(schedule) == "1") {
        static auto compiled =
            mlx::core::compile(routed_shared_experts<2, true, false>);
        return compiled(a)[0];
      }
      static auto compiled =
          mlx::core::compile(routed_shared_experts<4, false, false>);
      return compiled(a)[0];
    }();
    return reinterpret_cast<mlx_array *>(new array(std::move(out)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 routed/shared experts: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}
