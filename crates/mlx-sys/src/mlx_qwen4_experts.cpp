#include "mlx_common.h"
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

// All changing weights, slot mappings, scores and activations are inputs.
// Shapes specialize the two supported gate formats and two down formats.
std::vector<array> routed_experts(const std::vector<array> &a) {
  int tokens = a[0].shape(1), experts = a[3].shape(0) / 640;
  int gate_bits = a[3].shape(1) * 32 / 2560;
  int down_bits = a[9].shape(1) * 32 / 640;
  static auto gu = mlx::core::fast::metal_kernel(
      "qwen4_gate_up_decode", {"x", "ids", "wg", "sg", "bg", "wu", "su", "bu"},
      {"out"}, R"(
    quantized_expert_gate_up<T,BITS,2560,640,E,10>(x,ids,wg,sg,bg,wu,su,bu,out,
      threadgroup_position_in_grid.z,threadgroup_position_in_grid.y,
      simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
  )",
      expert_header());
  auto hidden = gu({a[0], a[1], a[3], a[4], a[5], a[6], a[7], a[8]},
                   {{tokens * 10, 1, 640}}, {mlx::core::float32},
                   {32, 160, tokens * 10}, {32, 2, 1},
                   {{"T", mlx::core::bfloat16},
                    {"BITS", gate_bits},
                    {"E", experts}},
                   std::nullopt, false, mlx::core::Device::gpu)[0];
  static auto down = mlx::core::fast::metal_kernel(
      "qwen4_down_combine_decode",
      {"x", "ids", "scores", "w", "scales", "biases"}, {"out"}, R"(
    threadgroup P products[10*4];
    affine_expert_down_combine<T,P,BITS,2560,640,E,10,4,5>(x,ids,scores,w,scales,biases,out,products,
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

const std::string &shared_expert_header() {
  static const std::string header = expert_header() +
#include "metal/qwen4/shared_expert_decode.metal.inc"
      ;
  return header;
}

std::vector<array> routed_shared_experts(const std::vector<array> &a) {
  constexpr int RPS = 4;
  const int rows = a[0].shape(1);
  const int experts = a[3].shape(0) / 640;
  const int gate_bits = a[3].shape(1) * 32 / 2560;
  const int down_bits = a[9].shape(1) * 32 / 640;
  // Threadgroup z < 10 * ROWS is the routed assignment row * 10 + slot; the
  // last z serves the shared expert. A threadgroup holds 2 * ROWS simdgroups,
  // one per (output half, activation row). Shared hidden rows follow the
  // 10 * ROWS routed rows.
  static auto gu = mlx::core::fast::metal_kernel(
      "qwen4_routed_shared_gu",
      {"x", "ids", "wg", "sg", "bg", "wu", "su", "bu", "swg", "ssg", "sbg",
       "swu", "ssu", "sbu", "sigmoid_table"},
      {"out"}, R"(
    const uint z = threadgroup_position_in_grid.z;
    const uint simd = simdgroup_index_in_threadgroup;
    if (z < 10 * ROWS) {
      routed_gu_rows<T,BITS,2560,640,E,10,ROWS>(x,ids,wg,sg,bg,wu,su,bu,out,
        z,threadgroup_position_in_grid.y,simd,thread_index_in_simdgroup);
    } else {
      const uint row = simd % ROWS;
      q4_shared_gu<T>(x + row * 2560,swg,ssg,sbg,swu,ssu,sbu,sigmoid_table,
        out + (10 * ROWS + row) * 640,
        threadgroup_position_in_grid.y * 8 + (simd / ROWS) * 4,thread_index_in_simdgroup);
    }
  )",
      shared_expert_header());
  auto hidden =
      gu({a[0], a[1], a[3], a[4], a[5], a[6], a[7], a[8], a[12], a[13], a[14],
          a[15], a[16], a[17], a[22]},
         {{11 * rows, 1, 640}}, {mlx::core::float32},
         {32, 160 * rows, 10 * rows + 1}, {32, 2 * rows, 1},
         {{"T", mlx::core::bfloat16},
          {"BITS", gate_bits},
          {"E", experts},
          {"ROWS", rows}},
         std::nullopt, false, mlx::core::Device::gpu)[0];
  // One threadgroup per RPS-column output tile serves every row with
  // 6 * ROWS simdgroups, one per (role, activation row).
  static auto down = mlx::core::fast::metal_kernel(
      "qwen4_routed_shared_down",
      {"x", "ids", "scores", "w", "scales", "biases", "sw", "ss", "sb", "gate",
       "sigmoid_table"},
      {"out"}, R"(
    threadgroup P products[10*RPS*ROWS];
    threadgroup T shared_products[RPS*ROWS];
    q4_down_combine_shared<T,P,BITS,2560,640,E,10,RPS,5,ROWS>(
      x,ids,scores,w,scales,biases,sw,ss,sb,gate,sigmoid_table,out,products,shared_products,
      threadgroup_position_in_grid.y,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
  )",
      shared_expert_header());
  return down({hidden, a[1], a[2], a[9], a[10], a[11], a[18], a[19], a[20],
               a[21], a[22]},
              {{1, rows, 2560}}, {mlx::core::bfloat16},
              {32, (2560 / RPS) * 6 * rows, 1}, {32, 6 * rows, 1},
              {{"T", mlx::core::bfloat16},
               {"P", a[2].dtype()},
               {"RPS", RPS},
               {"BITS", down_bits},
               {"E", experts},
               {"ROWS", rows}},
              std::nullopt, false, mlx::core::Device::gpu);
}

// 0..11 are the routed-expert inputs, 12..20 the shared banks, 21 the BF16
// shared gate and 22 the sigmoid table. One schedule serves both the
// single-row entry and the R-row entry.
const std::function<std::vector<array>(const std::vector<array> &)> &
routed_shared_default() {
  static auto compiled = mlx::core::compile(routed_shared_experts);
  return compiled;
}

bool routed_shared_admits(const std::vector<array> &a, int rows_max) {
  if (a.size() != 22 || a[0].ndim() != 3 || a[0].shape(0) != 1 ||
      a[0].shape(1) < 1 || a[0].shape(1) > rows_max || a[0].shape(2) != 2560 ||
      a[0].dtype() != mlx::core::bfloat16)
    return false;
  const int rows = a[0].shape(1);
  if (a[1].shape() != Shape{10 * rows} || a[1].dtype() != mlx::core::uint32 ||
      a[2].size() != 10 * rows ||
      (a[2].dtype() != mlx::core::bfloat16 &&
       a[2].dtype() != mlx::core::float32) ||
      a[21].shape() != Shape{1, rows, 1} ||
      a[21].dtype() != mlx::core::bfloat16 || a[3].ndim() != 2 ||
      a[3].shape(0) % 640 || a[9].ndim() != 2)
    return false;
  const int experts = a[3].shape(0) / 640, bits = a[3].shape(1) * 32 / 2560;
  if (experts < 1 || experts > 512 || (bits != 4 && bits != 5))
    return false;
  for (int base : {3, 6}) {
    if (a[base].shape() != Shape{experts * 640, 2560 * bits / 32} ||
        a[base].dtype() != mlx::core::uint32 ||
        a[base + 1].shape() != Shape{experts * 640, 160} ||
        a[base + 1].dtype() != mlx::core::uint8 ||
        a[base + 2].shape() != Shape{experts * 640, 20} ||
        a[base + 2].dtype() != mlx::core::float16)
      return false;
  }
  const int db = a[9].shape(1) * 32 / 640;
  if ((db != 5 && db != 8) ||
      a[9].shape() != Shape{experts * 2560, 640 * db / 32} ||
      a[9].dtype() != mlx::core::uint32 ||
      a[10].shape() != Shape{experts * 2560, 20} ||
      a[10].dtype() != mlx::core::float16 || a[11].shape() != a[10].shape() ||
      a[11].dtype() != mlx::core::float16)
    return false;
  for (int base : {12, 15, 18}) {
    const int n = base == 18 ? 2560 : 640, k = base == 18 ? 640 : 2560;
    if (a[base].shape() != Shape{n, k / 4} ||
        a[base].dtype() != mlx::core::uint32 ||
        a[base + 1].shape() != Shape{n, k / 32} ||
        a[base + 1].dtype() != mlx::core::float16 ||
        a[base + 2].shape() != a[base + 1].shape() ||
        a[base + 2].dtype() != mlx::core::float16)
      return false;
  }
  return true;
}

bool collect_arrays(mlx_array *const *inputs, size_t count,
                    std::vector<array> &a) {
  if (!inputs)
    return false;
  for (size_t i = 0; i < count; ++i) {
    if (!inputs[i])
      return false;
    a.push_back(*reinterpret_cast<array *>(inputs[i]));
  }
  return true;
}

// Shapes/dtypes specialize the graphs below; every mutable tensor is an
// input. One custom primitive per projection serves the single-row decode
// entry and the R-row entry; nothing caches a weight value or activation.
bool dense_decode_admits(const array &x, const array &w, const array &s,
                         const array &b) {
  const int k = x.shape(-1);
  return x.ndim() >= 2 && x.dtype() == mlx::core::bfloat16 && k >= 256 &&
         w.ndim() == 2 && w.dtype() == mlx::core::uint32 && k % 32 == 0 &&
         w.shape(0) >= 4 && w.shape(0) % 4 == 0 && w.shape(1) == k / 4 &&
         s.dtype() == mlx::core::float16 && b.dtype() == mlx::core::float16 &&
         s.shape() == Shape{w.shape(0), k / 32} && b.shape() == s.shape();
}

std::vector<array> dense_decode_graph(const std::vector<array> &in) {
  const auto &x = in[0], &w = in[1], &s = in[2], &b = in[3];
  const int k = x.shape(-1), n = w.shape(0), rows = x.size() / k;
  static auto kernel = mlx::core::fast::metal_kernel(
      "qwen4_dense_decode",
      {"x", "w", "scales", "biases", "normed", "sigmoid_table"}, {"out"},
#include "metal/qwen4/dense_decode.metal.inc"
      , qmv_header());
  auto shape = x.shape();
  shape.back() = n;
  // Rows per simdgroup only schedule the work: every output keeps its lane
  // partials and K order, so the bits do not depend on it. One activation
  // row keeps MLX's four rows per simdgroup; narrow projections with several
  // rows need the extra threadgroups to fill the GPU.
  const int rps = rows == 1 || n >= 8192 ? 4 : 2;
  return kernel({x, w, s, b, x, x}, {shape}, {mlx::core::bfloat16},
                {32, 2 * ((n + 2 * rps - 1) / (2 * rps)), 1}, {32, 2, 1},
                {{"T", mlx::core::bfloat16},
                 {"K", k},
                 {"N", n},
                 {"FAST", n % 8 == 0 && k % 512 == 0},
                 {"RPS", rps},
                 {"ACT", false},
                 {"MIX", false},
                 {"ROWS", rows}},
                std::nullopt, false, mlx::core::Device::gpu);
}

mlx_array *dense_decode_run(const array &x, const array &w, const array &s,
                            const array &b) {
  static auto compiled = mlx::core::compile(dense_decode_graph);
  auto result = compiled({x, w, s, b});
  return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
}

bool mixer_act_admits(const array &x, const array &w, const array &s,
                      const array &b, int rows_max) {
  return x.ndim() == 3 && x.shape(0) == 1 && x.shape(1) >= 1 &&
         x.shape(1) <= rows_max && x.shape(2) == 10240 &&
         x.dtype() == mlx::core::bfloat16 && w.shape() == Shape{320, 2560} &&
         w.dtype() == mlx::core::uint32 && s.shape() == Shape{320, 320} &&
         s.dtype() == mlx::core::float16 && b.shape() == s.shape() &&
         b.dtype() == s.dtype();
}

std::vector<array> mixer_act_graph(const std::vector<array> &a) {
  const int rows = a[0].shape(1);
  static auto kernel = mlx::core::fast::metal_kernel(
      "qwen4_decode_mixer_act",
      {"x", "w", "scales", "biases", "normed", "sigmoid_table"}, {"out"},
#include "metal/qwen4/dense_decode.metal.inc"
      , qmv_header());
  return kernel({a[0], a[1], a[2], a[3], a[0], a[4]}, {{1, rows, 320}},
                {mlx::core::bfloat16}, {32, 320, 1}, {32, 2, 1},
                {{"T", mlx::core::bfloat16},
                 {"K", 10240},
                 {"N", 320},
                 {"FAST", true},
                 {"MIX", false},
                 {"RPS", 1},
                 {"ACT", true},
                 {"ROWS", rows}},
                std::nullopt, false, mlx::core::Device::gpu);
}

mlx_array *mixer_act_run(const array &x, const array &w, const array &s,
                         const array &b) {
  static auto compiled = mlx::core::compile(mixer_act_graph);
  auto result = compiled({x, w, s, b, native_bf16_silu_table()});
  return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
}

bool hyper_up_admits(const array &x, const array &w, const array &s,
                     const array &b, const array &n, int rows_max) {
  return x.ndim() == 3 && x.shape(0) == 1 && x.shape(1) >= 1 &&
         x.shape(1) <= rows_max && x.shape(2) == 320 &&
         x.dtype() == mlx::core::bfloat16 &&
         n.shape() == Shape{1, x.shape(1), 10240} && n.dtype() == x.dtype() &&
         w.shape() == Shape{10240, 80} && w.dtype() == mlx::core::uint32 &&
         s.shape() == Shape{10240, 10} && s.dtype() == mlx::core::float16 &&
         b.shape() == s.shape() && b.dtype() == s.dtype();
}

std::vector<array> hyper_up_graph(const std::vector<array> &in) {
  const auto &x = in[0], &w = in[1], &s = in[2], &b = in[3], &n = in[4];
  const int rows = x.shape(1);
  static auto kernel = mlx::core::fast::metal_kernel(
      "qwen4_hyper_up",
      {"x", "w", "scales", "biases", "normed", "sigmoid_table"}, {"out"},
#include "metal/qwen4/dense_decode.metal.inc"
      , qmv_header());
  return kernel({x, w, s, b, n, in[5]}, {{1, rows, 2560}}, {x.dtype()},
                {32, 2560, 1}, {32, 2, 1},
                {{"T", x.dtype()},
                 {"K", 320},
                 {"N", 10240},
                 {"FAST", false},
                 {"RPS", 4},
                 {"ACT", false},
                 {"MIX", 1},
                 {"ROWS", rows}},
                std::nullopt, false, mlx::core::Device::gpu);
}

const std::function<std::vector<array>(const std::vector<array> &)> &
hyper_up_compiled() {
  static auto compiled = mlx::core::compile(hyper_up_graph);
  return compiled;
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
    const auto &w = *reinterpret_cast<array *>(weight);
    const auto &s = *reinterpret_cast<array *>(scales);
    const auto &b = *reinterpret_cast<array *>(biases);
    if (x.ndim() < 2 || x.size() != x.shape(-1) ||
        !dense_decode_admits(x, w, s, b))
      return nullptr;
    return dense_decode_run(x, w, s, b);
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 compact dense decode: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

// Up to four activation rows share one weight stream; row j carries the bits
// of mlx_qwen4_dense_decode on row j alone.
extern "C" mlx_array *mlx_qwen4_dense_decode_rows(mlx_array *input,
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
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(1) < 1 || x.shape(1) > 4 ||
        !dense_decode_admits(x, w, s, b))
      return nullptr;
    return dense_decode_run(x, w, s, b);
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 row-batched dense decode: " << e.what() << std::endl;
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
    if (!mixer_act_admits(x, w, s, b, 1))
      return nullptr;
    return mixer_act_run(x, w, s, b);
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 reference decode mixer activation: " << e.what()
              << std::endl;
  }
#endif
  return nullptr;
}

extern "C" mlx_array *mlx_qwen4_decode_mixer_act_rows(mlx_array *input,
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
    if (!mixer_act_admits(x, w, s, b, 4))
      return nullptr;
    return mixer_act_run(x, w, s, b);
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 row-batched mixer activation: " << e.what()
              << std::endl;
  }
#endif
  return nullptr;
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
    if (!hyper_up_admits(x, w, s, b, n, 1))
      return nullptr;
    auto result =
        hyper_up_compiled()({x, w, s, b, n, native_bf16_sigmoid_table()});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 hyper up: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

extern "C" mlx_array *mlx_qwen4_hyper_up_rows(mlx_array *input,
                                              mlx_array *weight,
                                              mlx_array *scales,
                                              mlx_array *biases,
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
    if (!hyper_up_admits(x, w, s, b, n, 4))
      return nullptr;
    auto result =
        hyper_up_compiled()({x, w, s, b, n, native_bf16_sigmoid_table()});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 row-batched hyper up: " << e.what() << std::endl;
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
    static auto fn = mlx::core::compile(routed_experts);
    return reinterpret_cast<mlx_array *>(new array(fn(a)[0]));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 routed experts: " << e.what() << std::endl;
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
    std::vector<array> a;
    if (count != 22 || !collect_arrays(inputs, count, a) ||
        !routed_shared_admits(a, 1))
      return nullptr;
    a.push_back(native_bf16_sigmoid_table());
    return reinterpret_cast<mlx_array *>(
        new array(routed_shared_default()(a)[0]));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 routed/shared experts: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

// Same inputs for up to four rows: x [1,R,2560], ids [10R], scores 10R,
// gate [1,R,1]. Row j carries the bits of the single-row entry on row j.
extern "C" mlx_array *
mlx_qwen4_routed_shared_experts_rows(mlx_array *const *inputs, size_t count) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    std::vector<array> a;
    if (count != 22 || !collect_arrays(inputs, count, a) ||
        !routed_shared_admits(a, 4))
      return nullptr;
    a.push_back(native_bf16_sigmoid_table());
    return reinterpret_cast<mlx_array *>(
        new array(routed_shared_default()(a)[0]));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 row-batched routed/shared experts: " << e.what()
              << std::endl;
  }
#endif
  return nullptr;
}
