#include "mlx_common.h"
#include "mlx_qwen4_flags.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string_view>
#ifdef MLX_NODE_METAL_ENABLED
#include "metal/common/quantized.h"
#include "mlx/primitives.h"
#endif

namespace {
struct ForwardFlags {
  size_t depth = 0;
  bool enabled = false;
  std::map<std::string, std::optional<std::string>, std::less<>> values;
};
thread_local ForwardFlags forward_flags;
} // namespace

const char *qwen4_env(const char *name) noexcept {
  if (!name)
    return nullptr;
  if (!forward_flags.depth || !forward_flags.enabled)
    return std::getenv(name);
  try {
    auto found = forward_flags.values.find(std::string_view(name));
    if (found == forward_flags.values.end()) {
      const auto raw = std::getenv(name);
      found = forward_flags.values
                  .emplace(name, raw ? std::optional<std::string>(raw)
                                     : std::nullopt)
                  .first;
    }
    return found->second ? found->second->c_str() : nullptr;
  } catch (...) {
    // Allocation failure must not cross the C ABI or change flag semantics.
    return std::getenv(name);
  }
}

extern "C" {

void mlx_qwen4_flags_begin() noexcept {
  if (forward_flags.depth++ == 0) {
    const auto setting = std::getenv("MLX_QWEN4_CACHE_FLAGS");
    forward_flags.enabled = !setting || std::strcmp(setting, "0") != 0;
  }
}

void mlx_qwen4_flags_end() noexcept {
  if (forward_flags.depth && --forward_flags.depth == 0) {
    forward_flags.values.clear();
    forward_flags.enabled = false;
  }
}

bool mlx_qwen4_flag_equals(const char *name, const char *value) noexcept {
  const auto setting = qwen4_env(name);
  return setting && value && std::strcmp(setting, value) == 0;
}

mlx_array *mlx_qwen4_router_decode(mlx_array *input, mlx_array *weight) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !weight)
      return nullptr;
    const auto &x = *reinterpret_cast<array *>(input);
    const auto &w = *reinterpret_cast<array *>(weight);
    if (x.shape() != mlx::core::Shape{1, 1, 2560} ||
        w.shape() != mlx::core::Shape{512, 2560} ||
        x.dtype() != mlx::core::bfloat16 || w.dtype() != x.dtype())
      return nullptr;
    static auto graph = [](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_router_decode", {"x", "w"}, {"out"},
#include "metal/qwen4/router_decode.metal.inc"
      );
      return kernel(a, {{1, 1, 512}}, {mlx::core::bfloat16}, {32 * 128, 1, 4},
                    {32, 1, 4}, {{"T", mlx::core::bfloat16}}, std::nullopt,
                    false, mlx::core::Device::gpu);
    };
    static auto compiled = mlx::core::compile(graph);
    auto result = compiled({x, w});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 router decode: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

// Singleton only: wider reductions use a different operation schedule.
bool mlx_qwen4_singleton_routes(mlx_array *logits, mlx_array **ids,
                                mlx_array **scores) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!logits || !ids || !scores)
      return false;
    const auto &x = *reinterpret_cast<array *>(logits);
    if (x.shape() != mlx::core::Shape{1, 1, 512} ||
        (x.dtype() != mlx::core::bfloat16 && x.dtype() != mlx::core::float32))
      return false;
    static auto make_graph = [](bool registers) {
      return [registers](const std::vector<array> &in) {
        const auto &x = in[0];
        static auto kernel = mlx::core::fast::metal_kernel(
            "qwen4_singleton_routes", {"logits"}, {"ids", "scores"},
#include "metal/qwen4/singleton_routes.metal.inc"
        );
        return kernel({x}, {{1, 1, 10}, {1, 1, 10}},
                      {mlx::core::uint32, x.dtype()}, {128, 1, 1}, {128, 1, 1},
                      {{"T", x.dtype()}, {"REG", registers}}, std::nullopt, false,
                      mlx::core::default_stream(mlx::core::Device::gpu));
      };
    };
    static auto graph = make_graph(false);
    static auto register_graph = make_graph(true);
    static auto compiled = mlx::core::compile(graph);
    static auto register_compiled = mlx::core::compile(register_graph);
    auto setting = qwen4_env("MLX_QWEN4_CACHED_KERNEL_GRAPHS");
    const bool cached = (!setting || std::string(setting) != "0");
    const auto reg = qwen4_env("MLX_QWEN4_ROUTE_REGISTER_RESULTS");
    const bool registers = reg && std::string(reg) == "1";
    auto result = registers
                      ? (cached ? register_compiled({x}) : register_graph({x}))
                      : (cached ? compiled({x}) : graph({x}));
    auto i = std::make_unique<array>(std::move(result[0]));
    auto p = std::make_unique<array>(std::move(result[1]));
    *ids = reinterpret_cast<mlx_array *>(i.release());
    *scores = reinterpret_cast<mlx_array *>(p.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 singleton routing: " << e.what() << std::endl;
    return false;
  }
#else
  return false;
#endif
}
// Keep the shared-gate projection alongside route selection, as in
// TrackFastMoE. Every changing tensor remains a compiled-graph argument.
bool mlx_qwen4_routes_shared_gate(mlx_array *logits, mlx_array *input,
                                 mlx_array *weight, mlx_array **ids,
                                 mlx_array **scores, mlx_array **gate) {
  if (!ids || !scores || !gate) return false;
  *ids = nullptr;
  *scores = nullptr;
  *gate = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!logits || !input || !weight) return false;
    const auto &l = *reinterpret_cast<array *>(logits);
    const auto &x = *reinterpret_cast<array *>(input);
    const auto &w = *reinterpret_cast<array *>(weight);
    if (l.shape() != mlx::core::Shape{1, 1, 512} ||
        x.shape() != mlx::core::Shape{1, 1, 2560} ||
        w.shape() != mlx::core::Shape{1, 2560} ||
        l.dtype() != mlx::core::bfloat16 || x.dtype() != l.dtype() ||
        w.dtype() != x.dtype()) return false;
    static auto graph = [](const std::vector<array> &a, bool registers) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_routes_shared_gate", {"logits", "x", "wg"},
          {"ids", "scores", "gate"},
#include "metal/qwen4/routes_shared_gate.metal.inc"
      );
      return kernel(a, {{1, 1, 10}, {1, 1, 10}, {1, 1, 1}},
                    {mlx::core::uint32, mlx::core::bfloat16, mlx::core::bfloat16},
                    {384, 1, 1}, {384, 1, 1},
                    {{"T", mlx::core::bfloat16}, {"REG", registers}},
                    std::nullopt, false, mlx::core::Device::gpu);
    };
    static auto compiled = mlx::core::compile(
        [](const std::vector<array> &a) { return graph(a, false); });
    static auto register_compiled = mlx::core::compile(
        [](const std::vector<array> &a) { return graph(a, true); });
    const bool registers = mlx_qwen4_flag_equals("MLX_QWEN4_ROUTE_REGISTER_RESULTS", "1");
    const bool cached = !mlx_qwen4_flag_equals("MLX_QWEN4_CACHED_KERNEL_GRAPHS", "0");
    auto result = cached ? (registers ? register_compiled : compiled)({l, x, w})
                         : graph({l, x, w}, registers);
    auto i = std::make_unique<array>(std::move(result[0]));
    auto s = std::make_unique<array>(std::move(result[1]));
    auto g = std::make_unique<array>(std::move(result[2]));
    *ids = reinterpret_cast<mlx_array *>(i.release());
    *scores = reinterpret_cast<mlx_array *>(s.release());
    *gate = reinterpret_cast<mlx_array *>(g.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 routes/shared gate: " << e.what() << std::endl;
  }
#endif
  return false;
}
// TrackFastMoE.route: one independent threadgroup per prompt row.
// Preserve this checkpoint's probability-first selection and normalization.
bool mlx_qwen4_prefill_routes(mlx_array *logits, mlx_array **ids,
                              mlx_array **scores) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!logits || !ids || !scores)
      return false;
    const auto &x = *reinterpret_cast<array *>(logits);
    if ((x.ndim() != 3 || x.shape(0) != 1 || x.shape(1) < 9 ||
         x.shape(1) > 1024 || x.shape(2) != 512) ||
        (x.dtype() != mlx::core::bfloat16 && x.dtype() != mlx::core::float32))
      return false;
    static auto make_graph = [](bool registers) {
      return [registers](const std::vector<array> &in) {
        const auto &x = in[0];
        static auto kernel = mlx::core::fast::metal_kernel(
            "qwen4_prefill_routes", {"logits"}, {"ids", "scores"},
#include "metal/qwen4/prefill_routes.metal.inc"
        );
        int rows = x.shape(1);
        return kernel({x}, {{1, rows, 10}, {1, rows, 10}},
                      {mlx::core::uint32, x.dtype()}, {128, rows, 1}, {128, 1, 1},
                      {{"T", x.dtype()}, {"REG", registers}}, std::nullopt, false,
                      mlx::core::default_stream(mlx::core::Device::gpu));
      };
    };
    static auto graph = make_graph(false);
    static auto register_graph = make_graph(true);
    static auto compiled = mlx::core::compile(graph);
    static auto register_compiled = mlx::core::compile(register_graph);
    auto setting = qwen4_env("MLX_QWEN4_CACHED_KERNEL_GRAPHS");
    const bool cached = (!setting || std::string(setting) != "0");
    const auto reg = qwen4_env("MLX_QWEN4_ROUTE_REGISTER_RESULTS");
    const bool registers = reg && std::string(reg) == "1";
    auto result = registers
                      ? (cached ? register_compiled({x}) : register_graph({x}))
                      : (cached ? compiled({x}) : graph({x}));
    auto i = std::make_unique<array>(std::move(result[0]));
    auto p = std::make_unique<array>(std::move(result[1]));
    *ids = reinterpret_cast<mlx_array *>(i.release());
    *scores = reinterpret_cast<mlx_array *>(p.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 prefill routing: " << e.what() << std::endl;
    return false;
  }
#else
  return false;
#endif
}
// TrackFastPLE.convSource, adapted to the local F32 tap products and BF16
// boundary. Keep SiLU/residual in their existing graph to preserve rounding.
mlx_array *mlx_qwen4_prefill_ple_conv(mlx_array *full, mlx_array *weight) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!full || !weight)
      return nullptr;
    const auto &x = *reinterpret_cast<array *>(full);
    const auto &w = *reinterpret_cast<array *>(weight);
    if (x.ndim() != 2 || x.shape(0) < 18 || x.shape(0) > 1033 ||
        x.shape(1) != 10240 || x.dtype() != mlx::core::bfloat16 ||
        w.shape() != mlx::core::Shape{10240, 4} ||
        w.dtype() != mlx::core::float32)
      return nullptr;
    static auto graph = [](const std::vector<array> &a) {
      int rows = a[0].shape(0) - 9;
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_prefill_ple_conv", {"full", "weight"}, {"out"},
          R"(
          const uint c=thread_position_in_grid.x;
          const uint t=thread_position_in_grid.y;
          if(c>=10240) return;
          float acc=0.0f;
          for(int j=0;j<4;++j) {
            volatile float product=float(full[size_t(t+j*3)*10240+c])*weight[c*4+j];
            acc=product+acc;
          }
          out[size_t(t)*10240+c]=T(acc);
        )");
      return kernel(a, {{1, rows, 10240}}, {mlx::core::bfloat16},
                    {10240, rows, 1}, {256, 1, 1}, {{"T", mlx::core::bfloat16}},
                    std::nullopt, false,
                    mlx::core::default_stream(mlx::core::Device::gpu));
    };
    static auto compiled = mlx::core::compile(graph);
    auto out = compiled({x, w});
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 reference PLE convolution: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}
// Affine expert GEMV keeps the half precision scale/bias banks in place. The
// generic gather promotes entire banks when BF16 activations meet F16 scales.
// Promote only the small activation, and preserve MLX's per-row accumulation.
mlx_array *mlx_qwen4_affine_expert_gemv(mlx_array *x, mlx_array *ids,
                                        mlx_array *weight, mlx_array *scales,
                                        mlx_array *biases, int experts,
                                        int group, int bits) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!x || !ids || !weight || !scales || !biases || experts <= 0 ||
        group != 32 || (bits != 5 && bits != 8))
      return nullptr;
    auto input = *reinterpret_cast<array *>(x),
         selected = *reinterpret_cast<array *>(ids);
    auto w = *reinterpret_cast<array *>(weight),
         s = *reinterpret_cast<array *>(scales),
         b = *reinterpret_cast<array *>(biases);
    if (input.ndim() != 3 || input.shape(1) != 1 ||
        selected.dtype() != mlx::core::uint32 || w.ndim() != 2 ||
        w.shape(0) % experts || w.dtype() != mlx::core::uint32 ||
        s.ndim() != 2 || s.dtype() != mlx::core::float16 ||
        b.dtype() != s.dtype() || b.shape() != s.shape())
      return nullptr;
    int rows = input.shape(0), k = input.shape(2), n = w.shape(0) / experts;
    if (rows <= 0 || rows > 80 || selected.size() != rows || k <= 0 ||
        k % group || n <= 0 || w.shape(1) != k * bits / 32 ||
        s.shape(0) != w.shape(0) || s.shape(1) != k / group)
      return nullptr;
    auto compute_type = mlx::core::promote_types(input.dtype(), s.dtype());
    static auto kernel = mlx::core::fast::metal_kernel(
        "qwen4_affine_expert_gemv", {"x", "ids", "w", "scales", "biases"},
        {"out"}, R"(
      constexpr int VP = get_pack_factor<BITS,32>();
      constexpr int BP = get_bytes_per_pack<BITS,32>();
      constexpr int BLOCK = VP * 32;
      uint assignment = threadgroup_position_in_grid.z;
      uint expert = ids[assignment];
      uint lane = thread_index_in_simdgroup;
      uint first = threadgroup_position_in_grid.y * 4 + simdgroup_index_in_threadgroup * 2;
      float result[2] = {0,0};
      if (first >= N) return;
      if (expert < E) {
        for (int base = 0; base < K; base += BLOCK) {
          int col = base + lane * VP;
          if (col < K) {
            float values[VP];
            float sum = load_vector<C,float,VP,BITS>(x + size_t(assignment)*K + col,values);
            for (int r = 0; r < 2 && first+r < N; ++r) {
              size_t row = size_t(expert)*N + first+r;
              auto codes = (const device uint8_t*)w + row*(K*BP/VP) + col*BP/VP;
              size_t side = row*(K/GS) + col/GS;
              result[r] += qdot<float,VP,BITS,false>(codes,values,float(scales[side]),float(biases[side]),sum);
            }
          }
        }
      }
      for (int r = 0; r < 2 && first+r < N; ++r) {
        float value = simd_sum(result[r]);
        if (lane == 0) out[size_t(assignment)*N + first+r] = T(C(value));
      }
    )",
        std::string(mlx::core::quantized_preamble::gemm()) +
            mlx::core::quantized_preamble::quantized_utils() +
            mlx::core::quantized_preamble::kquant());
    auto result = kernel(
        {astype(input, compute_type), selected, w, s, b}, {{rows, 1, n}},
        {input.dtype()}, {32, ((n + 3) / 4) * 2, rows}, {32, 2, 1},
        {{"C", compute_type},
         {"T", input.dtype()},
         {"K", k},
         {"N", n},
         {"E", experts},
         {"GS", group},
         {"BITS", bits}},
        std::nullopt, false, mlx::core::default_stream(mlx::core::Device::gpu));
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 affine expert GEMV: " << e.what() << std::endl;
    return nullptr;
  }
#else
  return nullptr;
#endif
}

mlx_array *mlx_qwen4_expert_gemv(mlx_array *x, mlx_array *ids,
                                 mlx_array *weight, mlx_array *scales,
                                 mlx_array *biases, int experts, int group,
                                 int bits, const char *mode) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!x || !ids || !weight || !scales || !biases || !mode || experts <= 0)
      return nullptr;
    auto input = *reinterpret_cast<array *>(x),
         selected = *reinterpret_cast<array *>(ids);
    auto w = *reinterpret_cast<array *>(weight),
         s = *reinterpret_cast<array *>(scales),
         b = *reinterpret_cast<array *>(biases);
    auto quant = mlx::core::string_to_quantization_mode(mode);
    int ratio = mlx::core::quant_super_ratio(quant);
    if (!ratio || input.ndim() != 3 || input.shape(1) != 1 ||
        selected.dtype() != mlx::core::uint32 || w.ndim() != 2 ||
        w.shape(0) % experts || b.dtype() != mlx::core::float16)
      return nullptr;
    int rows = input.shape(0), k = input.shape(2), n = w.shape(0) / experts;
    if (rows > 80 || selected.size() != rows || k % group || n <= 0)
      return nullptr;
    static auto kernel = [] {
      // Reuse the vendored MLX arithmetic. Only dimension arguments change
      // from constant-buffer references to values for template specialization.
      std::string header = std::string(mlx::core::quantized_preamble::gemm()) +
                           mlx::core::quantized_preamble::quantized_utils() +
                           mlx::core::quantized_preamble::kquant();
      for (const char *name : {"kquant_qmv_fast_impl", "kquant_qmv_impl"}) {
        auto start = header.find(std::string("METAL_FUNC void ") + name);
        auto end = header.find('{', start);
        auto signature = header.substr(start, end - start);
        for (size_t p = 0; (p = signature.find("const constant int&", p)) !=
                           std::string::npos;)
          signature.replace(p, std::string("const constant int&").size(),
                            "const int");
        header.replace(start, end - start, signature);
      }
      return mlx::core::fast::metal_kernel(
          "qwen4_expert_gemv", {"x", "ids", "w", "scales", "biases"}, {"out"},
          R"(
        uint row = threadgroup_position_in_grid.z;
        uint expert = ids[row];
        if (expert >= E) return;
        const device uint32_t* weights = w + size_t(expert) * WS;
        auto scale = KQScales<float,BITS,SR,HM>((const device uint8_t*)scales + size_t(expert)*SS,
          biases + size_t(expert)*BS);
        auto input = x + size_t(row)*K;
        auto output = out + size_t(row)*N;
        uint3 tile = uint3(0,threadgroup_position_in_grid.y,0);
        if (FAST) kquant_qmv_fast_impl<T,GS,BITS,SR,HM>(weights,scale,input,output,K,N,tile,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
        else kquant_qmv_impl<T,GS,BITS,SR,HM>(weights,scale,input,output,K,N,tile,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
      )",
          header);
    }();
    auto result = kernel(
        {input, selected, w, s, b}, {{rows, 1, n}}, {input.dtype()},
        {32, ((n + 7) / 8) * 2, rows}, {32, 2, 1},
        {{"T", input.dtype()},
         {"K", k},
         {"N", n},
         {"E", experts},
         {"GS", group},
         {"BITS", bits},
         {"SR", ratio},
         {"HM", mlx::core::quant_has_sub_min(quant)},
         {"FAST", n % 8 == 0 && k % 512 == 0},
         {"WS", int(w.size() / experts)},
         {"SS", int(s.nbytes() / experts)},
         {"BS", int(b.size() / experts)}},
        std::nullopt, false, mlx::core::default_stream(mlx::core::Device::gpu));
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 expert GEMV: " << e.what() << std::endl;
    return nullptr;
  }
#else
  return nullptr;
#endif
}

bool mlx_qwen4_gather_window(mlx_array *keys, mlx_array *values,
                             mlx_array *table, mlx_array *tokens,
                             mlx_array *fresh_k, mlx_array *fresh_v, int base,
                             int block_size, mlx_array **out_k,
                             mlx_array **out_v) {
  try {
    if (!keys || !values || !table || !tokens || !fresh_k || !fresh_v ||
        !out_k || !out_v || base < 0 || block_size <= 0)
      return false;
    auto k = *reinterpret_cast<array *>(keys),
         v = *reinterpret_cast<array *>(values);
    auto blocks = *reinterpret_cast<array *>(table),
         ids = *reinterpret_cast<array *>(tokens);
    auto fk = *reinterpret_cast<array *>(fresh_k),
         fv = *reinterpret_cast<array *>(fresh_v);
    if (fk.ndim() != 4 || fk.shape() != fv.shape() ||
        fk.dtype() != fv.dtype() || ids.ndim() != 1 ||
        ids.dtype() != mlx::core::int32 || blocks.dtype() != mlx::core::int32 ||
        k.dtype() != v.dtype() || k.size() != v.size() || fk.shape(0) != 1 ||
        fk.shape(3) % 8)
      return false;
    int heads = fk.shape(1), dim = fk.shape(3), fresh = fk.shape(2),
        count = ids.size();
    static auto kernel = mlx::core::fast::metal_kernel(
        "qwen4_gather_window", {"keys", "values", "blocks", "ids", "fk", "fv"},
        {"ko", "vo"}, R"(
      uint i = thread_position_in_grid.x;
      if (i >= H * N * D) return;
      uint d = i % D, t = (i / D) % N, h = i / (D * N);
      int token = ids[t];
      if (token < 0 || token >= BASE + FRESH) { ko[i] = 0; vo[i] = 0; return; }
      if (token >= BASE) {
        size_t src = (size_t(h) * FRESH + uint(token - BASE)) * D + d;
        ko[i] = fk[src]; vo[i] = fv[src];
      } else {
        int block = blocks[token / BS];
        if (block < 0 || block >= POOL) { ko[i] = 0; vo[i] = 0; return; }
        uint offset = token % BS;
        size_t ki = (((size_t(block) * H + h) * (D / 8) + d / 8) * BS + offset) * 8 + d % 8;
        size_t vi = ((size_t(block) * H + h) * D + d) * BS + offset;
        ko[i] = keys[ki]; vo[i] = values[vi];
      }
    )");
    auto result = kernel(
        {k, v, blocks, ids, fk, fv},
        {{1, heads, count, dim}, {1, heads, count, dim}},
        {fk.dtype(), fk.dtype()}, {heads * count * dim, 1, 1}, {256, 1, 1},
        {{"H", heads},
         {"D", dim},
         {"N", count},
         {"BASE", base},
         {"FRESH", fresh},
         {"BS", block_size},
         {"POOL", int(k.size() / (heads * dim * block_size))}},
        std::nullopt, false, mlx::core::default_stream(mlx::core::Device::gpu));
    *out_k = reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
    *out_v = reinterpret_cast<mlx_array *>(new array(std::move(result[1])));
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 paged window: " << e.what() << std::endl;
    return false;
  }
}

// Reference partial rotary epilogue with local BF16 operation boundaries.
mlx_array *mlx_qwen4_rotary_window(mlx_array *input, mlx_array *cosine,
                                    mlx_array *sine) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !cosine || !sine)
      return nullptr;
    const auto &x = *reinterpret_cast<array *>(input);
    const auto &c = *reinterpret_cast<array *>(cosine);
    const auto &s = *reinterpret_cast<array *>(sine);
    if (x.ndim() != 4 || x.shape(0) != 1 || x.shape(1) < 1 ||
        x.shape(2) < 1 || x.shape(2) > 1024 || x.shape(3) < 1 ||
        x.size() > std::numeric_limits<int>::max() ||
        x.dtype() != mlx::core::bfloat16 || c.dtype() != x.dtype() ||
        s.dtype() != x.dtype() || c.ndim() != 4 ||
        c.shape(0) != 1 || c.shape(1) != 1 || c.shape(2) != x.shape(2) ||
        c.shape(3) < 1 || c.shape(3) > x.shape(3) / 2 || s.shape() != c.shape())
      return nullptr;
    static auto graph = [](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_rotary_window", {"x", "cosb", "sinb"}, {"out"},
#include "metal/common/rope_split_half.metal.inc"
      );
      const auto &x = a[0];
      return kernel(a, {x.shape()}, {x.dtype()}, {int(x.size()), 1, 1},
                    {256, 1, 1}, {{"T", x.dtype()}, {"S", x.shape(2)},
                                 {"D", x.shape(3)}, {"ROT", 2 * a[1].shape(3)},
                                 {"COUNT", int(x.size())}},
                    std::nullopt, false,
                    mlx::core::default_stream(mlx::core::Device::gpu));
    };
    static auto compiled = mlx::core::compile(graph);
    auto result = compiled({x, c, s});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 rotary window: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

// Reference attention prep, with the GGUF's F32 scales and reduction order.
mlx_array *mlx_qwen4_attention_norm_rotary(mlx_array *input, mlx_array *weight,
                                          mlx_array *cosine, mlx_array *sine,
                                          double eps) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !weight || !cosine || !sine || !std::isfinite(eps) || eps <= 0)
      return nullptr;
    const auto &x = *reinterpret_cast<array *>(input);
    const auto &w = *reinterpret_cast<array *>(weight);
    const auto &c = *reinterpret_cast<array *>(cosine);
    const auto &s = *reinterpret_cast<array *>(sine);
    if (x.ndim() != 4 || x.shape(0) != 1 || x.shape(1) < 1 ||
        x.shape(2) < 1 || x.shape(2) > 1024 || x.shape(3) != 256 ||
        x.size() > std::numeric_limits<int>::max() ||
        x.dtype() != mlx::core::bfloat16 || w.size() != 256 ||
        w.dtype() != mlx::core::float32 || c.dtype() != x.dtype() ||
        s.dtype() != x.dtype() || c.ndim() != 4 ||
        c.shape(0) != 1 || c.shape(1) != 1 || c.shape(2) != x.shape(2) ||
        c.shape(3) < 1 || c.shape(3) > 128 || s.shape() != c.shape())
      return nullptr;
    static auto graph = [](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_attention_norm_rotary", {"x", "w", "cosb", "sinb", "eps"},
          {"out"},
#include "metal/common/rms_norm_rope_256.metal.inc"
          , "", false);
      const auto &x = a[0];
      return kernel(a, {x.shape()}, {x.dtype()},
                    {32, int(x.size() / 256), 1}, {32, 1, 1},
                    {{"T", x.dtype()}, {"S", x.shape(2)}, {"D", 256},
                     {"ROT", 2 * a[2].shape(3)}},
                    std::nullopt, false,
                    mlx::core::default_stream(mlx::core::Device::gpu));
    };
    static auto compiled = mlx::core::compile(graph);
    auto result = compiled({x, reshape(w, {256}), c, s, array(float(eps))});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 attention normalization and rotary: " << e.what()
              << std::endl;
  }
#endif
  return nullptr;
}

// Reference wide normalization schedule, retaining this GGUF's arithmetic.
static mlx_array *qwen4_hyper_norm(mlx_array *x, mlx_array *w, double eps,
                                   bool decode) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!x || !w || !std::isfinite(eps) || eps <= 0)
      return nullptr;
    const auto &input = *reinterpret_cast<array *>(x);
    const auto &weight = *reinterpret_cast<array *>(w);
    if (input.ndim() != 3 || input.shape(0) != 1 ||
        input.shape(1) < (decode ? 1 : 9) ||
        input.shape(1) > (decode ? 8 : 1024) || input.shape(2) != 10240 ||
        input.dtype() != mlx::core::bfloat16 || weight.size() != 10240 ||
        weight.dtype() != mlx::core::float32)
      return nullptr;
    static auto graph = [](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_prefill_norm", {"residual", "scale", "eps"}, {"normed"},
#include "metal/common/group_rms_norm.metal.inc"
      );
      int rows = a[0].shape(1);
      // Reference singleton schedule: 640 threads (20 simdgroups). Wide
      // windows keep four simdgroups and the same 20 virtual partials.
      int sg = rows <= 8 ? 20 : 4;
      return kernel(
          a, {{1, rows, 10240}}, {mlx::core::bfloat16}, {32 * sg, 4, rows},
          {32 * sg, 1, 1},
          {{"InT", mlx::core::bfloat16}, {"H", 2560}, {"W", 10240}, {"SG", sg}},
          std::nullopt, false,
          mlx::core::default_stream(mlx::core::Device::gpu));
    };
    static auto compiled = mlx::core::compile(graph);
    auto result =
        compiled({input, reshape(weight, {10240}), array(float(eps))});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 reference prefill normalization: " << e.what()
              << std::endl;
  }
#endif
  return nullptr;
}

mlx_array *mlx_qwen4_prefill_norm(mlx_array *x, mlx_array *w, double eps) {
  return qwen4_hyper_norm(x, w, eps, false);
}
mlx_array *mlx_qwen4_decode_norm(mlx_array *x, mlx_array *w, double eps) {
  return qwen4_hyper_norm(x, w, eps, true);
}

bool mlx_qwen4_inject_norm(mlx_array *x, mlx_array *y, mlx_array *g,
                           mlx_array *w, double eps, mlx_array **stream,
                           mlx_array **normed) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!stream || !normed)
      return false;
    *stream = nullptr;
    *normed = nullptr;
    if (!x || !y || !g || !w || !std::isfinite(eps) || eps <= 0)
      return false;
    const auto &input = *reinterpret_cast<array *>(x);
    const auto &branch = *reinterpret_cast<array *>(y);
    const auto &gate = *reinterpret_cast<array *>(g);
    const auto &weight = *reinterpret_cast<array *>(w);
    if (input.ndim() != 3 || input.shape(0) != 1 || input.shape(1) < 1 ||
        input.shape(1) > 1024 || input.shape(2) != 10240 ||
        input.dtype() != mlx::core::bfloat16 ||
        branch.dtype() != input.dtype() || gate.dtype() != input.dtype() ||
        branch.shape() != Shape{1, input.shape(1), 2560} ||
        gate.shape() != Shape{1, input.shape(1), 4} || weight.size() != 10240 ||
        weight.dtype() != mlx::core::float32)
      return false;
    static auto graph = [](const std::vector<array> &a) {
      static auto kernel = mlx::core::fast::metal_kernel(
          "qwen4_inject_norm", {"residual", "branch", "inject", "scale", "eps"},
          {"stream", "normed"},
#include "metal/qwen4/inject_norm.metal.inc"
      );
      int rows = a[0].shape(1), sg = rows <= 8 ? 20 : 4;
      return kernel(a, {{1, rows, 10240}, {1, rows, 10240}},
                    {mlx::core::bfloat16, mlx::core::bfloat16},
                    {32 * sg, 4, rows}, {32 * sg, 1, 1},
                    {{"InT", mlx::core::bfloat16},
                     {"H", 2560},
                     {"W", 10240},
                     {"HC", 4},
                     {"SG", sg}},
                    std::nullopt, false,
                    mlx::core::default_stream(mlx::core::Device::gpu));
    };
    static auto compiled = mlx::core::compile(graph);
    auto result = compiled(
        {input, branch, gate, reshape(weight, {10240}), array(float(eps))});
    auto a = std::make_unique<array>(std::move(result[0]));
    auto b = std::make_unique<array>(std::move(result[1]));
    *stream = reinterpret_cast<mlx_array *>(a.release());
    *normed = reinterpret_cast<mlx_array *>(b.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 reference injection and normalization: " << e.what()
              << std::endl;
  }
#endif
  return false;
}

mlx_array *mlx_qwen4_norm(mlx_array *x, mlx_array *w, int group, double eps,
                          bool centered) {
  try {
    if (!x || !w || group <= 0)
      return nullptr;
    auto input = *reinterpret_cast<array *>(x),
         weight = *reinterpret_cast<array *>(w);
    if (input.shape(-1) % group || weight.size() != input.shape(-1))
      return nullptr;
    const auto port = qwen4_env("MLX_QWEN4_PREFILL_NORM");
    if (group == 2560 && !centered && (!port || std::string(port) != "0")) {
      if (auto result = mlx_qwen4_prefill_norm(x, w, eps))
        return result;
    }
    const auto decode_port = qwen4_env("MLX_QWEN4_DECODE_NORM");
    if (group == 2560 && !centered &&
        (!decode_port || std::string(decode_port) != "0")) {
      if (auto result = mlx_qwen4_decode_norm(x, w, eps))
        return result;
    }
    auto shape = input.shape();
    auto grouped = reshape(astype(input, mlx::core::float32),
                           {-1, int(weight.size()) / group, group});
    auto scale = astype(weight, mlx::core::float32);
    if (centered)
      scale = scale + array(1.0f);
    scale = reshape(scale, {int(weight.size()) / group, group});
    static auto fn = mlx::core::compile(
        [](const std::vector<array> &a) {
          auto normalized = a[0] / sqrt(mean(square(a[0]), {-1}, true) + a[2]);
          return std::vector<array>{normalized * a[1]};
        },
        true);
    auto result =
        astype(reshape(fn({grouped, scale, array(float(eps))})[0], shape),
               input.dtype());
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 normalization: " << e.what() << std::endl;
    return nullptr;
  }
}

// Fuse the prefill epilogue without storing intermediate F32 upcasts. Keep
// both BF16 rounding boundaries and the original compiled norm/gate algebra.
static std::vector<array>
qwen4_gdn_epilogue_graph(const std::vector<array> &a) {
  auto y = astype(astype(a[0], mlx::core::bfloat16), mlx::core::float32);
  auto grouped = reshape(y, {-1, 1, 128});
  static auto norm = mlx::core::compile(
      [](const std::vector<array> &v) {
        return std::vector<array>{
            v[0] / sqrt(mean(square(v[0]), {-1}, true) + v[2]) * v[1]};
      },
      true);
  auto n = astype(
      reshape(norm({grouped, reshape(a[2], {1, 128}), a[3]})[0], y.shape()),
      mlx::core::bfloat16);
  static auto gate = mlx::core::compile(
      [](const std::vector<array> &v) {
        return std::vector<array>{v[1] * sigmoid(v[0])};
      },
      true);
  return {astype(gate({astype(a[1], mlx::core::float32),
                       astype(n, mlx::core::float32)})[0],
                 mlx::core::bfloat16)};
}

// Port of mlxfast TrackFastKernels2.gatedRMSSource (8981cef5). Retain our
// checkpoint's F32 norm scale and single BF16 rounding after normalization.
static std::vector<array> qwen4_gdn_norm_port(const std::vector<array> &a) {
  static auto kernel = mlx::core::fast::metal_kernel(
      "qwen4_prefill_gdn_norm", {"y", "proj", "w", "eps"}, {"out"}, R"(
    constexpr int N_READS = 4;
    const uint lid = thread_position_in_threadgroup.x;
    const uint hv = thread_position_in_grid.y;
    const uint row = thread_position_in_grid.z;
    const uint base = (row * HV + hv) * 128;
    float thread_x[N_READS];
    float acc = 0.0f;
    for (int i = 0; i < N_READS; ++i) {
      thread_x[i] = float(T(y[base + lid * N_READS + i]));
      // The unfused source rounds square before the sum; prevent an FMA.
      volatile float squared = thread_x[i] * thread_x[i];
      acc = squared + acc;
    }
    acc = simd_sum(acc);
    const float denom = metal::precise::sqrt(acc / 128.0f + eps);
    for (int i = 0; i < N_READS; ++i) {
      const uint d = lid * N_READS + i;
      T n = T((thread_x[i] / denom) * w[d]);
      const float z = float(proj[base + d]);
      const auto s = 1 / (1 + metal::exp(metal::abs(z)));
      const float g = z < 0 ? s : 1 - s;
      out[base + d] = T(g * float(n));
    }
  )");
  return kernel(a, {a[0].shape()}, {mlx::core::bfloat16},
                {32, a[0].shape(2), a[0].shape(1)}, {32, 1, 1},
                {{"T", mlx::core::bfloat16}, {"HV", a[0].shape(2)}},
                std::nullopt, false,
                mlx::core::default_stream(mlx::core::Device::gpu));
}

mlx_array *mlx_qwen4_gdn_epilogue(mlx_array *out, mlx_array *z, mlx_array *norm,
                                  double eps) {
  try {
    if (!out || !z || !norm || !std::isfinite(eps) || eps <= 0)
      return nullptr;
    auto x = *reinterpret_cast<array *>(out), g = *reinterpret_cast<array *>(z),
         w = *reinterpret_cast<array *>(norm);
    if (x.ndim() != 4 || x.shape(0) != 1 || x.shape(1) < 1 ||
        x.shape(1) > 1024 || x.shape(3) != 128 ||
        x.dtype() != mlx::core::float32 || g.shape() != x.shape() ||
        g.dtype() != mlx::core::bfloat16 || w.size() != 128 ||
        w.dtype() != mlx::core::float32)
      return nullptr;
    std::vector<array> a{x, g, w, array(float(eps))};
    static auto fn = mlx::core::compile(qwen4_gdn_epilogue_graph);
    static auto port = mlx::core::compile(qwen4_gdn_norm_port);
    const auto setting = qwen4_env("MLX_QWEN4_PREFILL_GDN_NORM");
    auto y = (!setting || std::string(setting) != "0") && x.shape(1) > 8
                 ? port(a)[0]
                 : fn(a)[0];
    if (qwen4_env("MLX_QWEN4_EPILOGUE_CHECK")) {
      auto ref = qwen4_gdn_epilogue_graph(a)[0];
      if (mlx::core::any(not_equal(y, ref)).item<bool>()) {
        std::cerr << "QWEN4_EPILOGUE_CHECK_FAILED" << std::endl;
        return nullptr;
      }
    }
    return reinterpret_cast<mlx_array *>(new array(std::move(y)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 GDN epilogue: " << e.what() << std::endl;
    return nullptr;
  }
}

bool mlx_qwen4_gdn_gates(mlx_array *a, mlx_array *b, mlx_array *scale,
                         mlx_array *dt, mlx_array **decay, mlx_array **beta) {
  try {
    if (!a || !b || !scale || !dt || !decay || !beta)
      return false;
    static auto fn = mlx::core::compile(
        [](const std::vector<array> &a) {
          auto z = a[0] + a[3];
          auto softplus = maximum(z, array(0.0f)) + log1p(exp(-abs(z)));
          return std::vector<array>{exp(softplus * a[2]), sigmoid(a[1])};
        },
        true);
    auto result =
        fn({*reinterpret_cast<array *>(a), *reinterpret_cast<array *>(b),
            *reinterpret_cast<array *>(scale), *reinterpret_cast<array *>(dt)});
    *decay = reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
    *beta = reinterpret_cast<mlx_array *>(new array(std::move(result[1])));
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 GDN gates: " << e.what() << std::endl;
    return false;
  }
}

mlx_array *mlx_qwen4_inject(mlx_array *x, mlx_array *y, mlx_array *g) {
  try {
    if (!x || !y || !g)
      return nullptr;
    static auto fn = mlx::core::compile(
        [](const std::vector<array> &a) {
          return std::vector<array>{a[0] + a[1] * a[2]};
        },
        true);
    auto input = *reinterpret_cast<array *>(x),
         branch = *reinterpret_cast<array *>(y),
         gate = *reinterpret_cast<array *>(g);
    auto residual = reshape(input, {input.shape(0), input.shape(1),
                                    gate.shape(-1), branch.shape(-1)});
    auto result = reshape(
        fn({residual, expand_dims(branch, -2), expand_dims(gate, -1)})[0],
        input.shape());
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 injection: " << e.what() << std::endl;
    return nullptr;
  }
}

// Stable bucket ordering inspired by mlxfast's TrackPrefillSort. Equal expert
// ids retain assignment order, and inverse is written during the scatter.
bool mlx_qwen4_route_sort(mlx_array *ids, int experts, mlx_array **order,
                          mlx_array **inverse, mlx_array **sorted_ids) {
  try {
    if (!ids || !order || !inverse || !sorted_ids || experts <= 0 ||
        experts > 1024)
      return false;
    auto input = *reinterpret_cast<array *>(ids);
    if (input.ndim() != 1 || input.dtype() != mlx::core::uint32 ||
        input.size() < 256 || input.size() > 65536)
      return false;
    const int rows = input.size(), blocks = (rows + 255) / 256;
    static auto count =
        mlx::core::fast::metal_kernel("qwen4_route_counts", {"ids"}, {"counts"},
#include "metal/common/route_counts.metal.inc"
        );
    static auto scatter =
        mlx::core::fast::metal_kernel("qwen4_route_scatter", {"ids", "counts"},
                                      {"order", "inverse", "sorted_ids"}, R"(
      threadgroup uint tile[256], prefix[E], scratch[E], earlier[E];
      uint lane = thread_position_in_threadgroup.x, block = threadgroup_position_in_grid.x;
      uint row = block * 256 + lane;
      tile[lane] = row < R ? ids[row] : E;
      for (uint expert = lane; expert < E; expert += 256) {
        uint total = 0, before = 0;
        for (uint b = 0; b < NB; ++b) {
          uint n = counts[b * E + expert]; total += n; if (b < block) before += n;
        }
        prefix[expert] = total; earlier[expert] = before;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint stride = 1; stride < E; stride *= 2) {
        for (uint e = lane; e < E; e += 256) scratch[e] = prefix[e] + (e >= stride ? prefix[e - stride] : 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < E; e += 256) prefix[e] = scratch[e];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (row >= R) return;
      uint expert = tile[lane];
      if (expert >= E) return; // ids are generated by the bounded router.
      uint rank = earlier[expert] + (expert > 0 ? prefix[expert - 1] : 0);
      for (uint j = 0; j < lane; ++j) rank += tile[j] == expert;
      order[rank] = row; inverse[row] = rank; sorted_ids[rank] = expert;
    )");
    auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto counts =
        count({input}, {{blocks * experts}}, {mlx::core::uint32},
              {blocks * 256, 1, 1}, {256, 1, 1},
              {{"E", experts},
               {"R", rows},
               {"BALLOT", experts > 256 && experts <= 512 &&
                              (!qwen4_env("MLX_QWEN4_BALLOT_ROUTE_SORT") ||
                               std::string(qwen4_env(
                                   "MLX_QWEN4_BALLOT_ROUTE_SORT")) != "0")}},
              std::nullopt, false, stream);
    auto result =
        scatter({input, counts[0]}, {{rows}, {rows}, {rows}},
                {mlx::core::uint32, mlx::core::uint32, mlx::core::uint32},
                {blocks * 256, 1, 1}, {256, 1, 1},
                {{"E", experts}, {"R", rows}, {"NB", blocks}}, std::nullopt,
                false, stream);
    *order = reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
    *inverse = reinterpret_cast<mlx_array *>(new array(std::move(result[1])));
    *sorted_ids =
        reinterpret_cast<mlx_array *>(new array(std::move(result[2])));
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 routing: " << e.what() << std::endl;
    return false;
  }
}

// The caller owns these mutable slot banks exclusively and evaluates its last
// bank-dependent output before reusing a slot. Validate all copies first, then
// finish GPU work before touching shared CPU-visible storage. Sources retain
// their arrays throughout the transaction. No caller may publish a new mapping
// until this operation succeeds.
bool mlx_qwen4_copy_weight_rows(mlx_array **destinations, mlx_array **sources,
                                const uint32_t *slots, size_t arrays,
                                size_t updates) {
  try {
    if (!destinations || !sources || !slots || !arrays || !updates ||
        arrays > 12 || updates > 16)
      return false;
    std::vector<array> ready;
    for (size_t a = 0; a < arrays; ++a) {
      if (!destinations[a])
        return false;
      auto &d = *reinterpret_cast<array *>(destinations[a]);
      if (d.ndim() != 2 || !d.flags().row_contiguous)
        return false;
      ready.push_back(d);
      for (size_t u = 0; u < updates; ++u) {
        if (!sources[u * arrays + a])
          return false;
        auto &s = *reinterpret_cast<array *>(sources[u * arrays + a]);
        if (s.ndim() != 2 || !s.flags().row_contiguous ||
            s.dtype() != d.dtype() || s.shape(1) != d.shape(1) ||
            s.shape(0) <= 0)
          return false;
        if ((uint64_t(slots[u]) + 1) * uint64_t(s.shape(0)) >
            uint64_t(d.shape(0)))
          return false;
        ready.push_back(s);
      }
    }
    mlx::core::eval(ready);
    mlx::core::synchronize(mlx::core::default_stream(mlx::core::Device::gpu));
    for (size_t a = 0; a < arrays; ++a) {
      auto &d = *reinterpret_cast<array *>(destinations[a]);
      for (size_t u = 0; u < updates; ++u) {
        auto &s = *reinterpret_cast<array *>(sources[u * arrays + a]);
        std::memcpy(d.data<char>() + size_t(slots[u]) * s.nbytes(),
                    s.data<char>(), s.nbytes());
      }
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 expert slot update: " << e.what() << std::endl;
    return false;
  }
}

bool mlx_qwen4_gather_pages(mlx_array *keys, mlx_array *values,
                            mlx_array *slots, int heads, int dim,
                            int block_size, mlx_array **out_keys,
                            mlx_array **out_values) {
  try {
    if (!keys || !values || !slots || !out_keys || !out_values || heads <= 0 ||
        dim <= 0 || dim % 8 || block_size <= 0)
      return false;
    auto &k = *reinterpret_cast<array *>(keys);
    auto &v = *reinterpret_cast<array *>(values);
    auto &indices = *reinterpret_cast<array *>(slots);
    if (indices.ndim() != 1 || indices.dtype() != mlx::core::int32 ||
        k.dtype() != v.dtype() || k.size() != v.size())
      return false;
    const int count = indices.size();
    static auto kernel = mlx::core::fast::metal_kernel(
        "qwen4_gather_pages", {"keys", "values", "slots"}, {"ko", "vo"}, R"(
      const uint j = thread_position_in_grid.x;
      if (j >= HC * TC * DC) return;
      const uint d = j % DC;
      const uint t = (j / DC) % TC;
      const uint h = j / (DC * TC);
      const uint slot = slots[t];
      const uint block = slot / BS, offset = slot % BS;
      const size_t ki = (((size_t(block) * HC + h) * (DC / 8) + d / 8) * BS + offset) * 8 + d % 8;
      const size_t vi = ((size_t(block) * HC + h) * DC + d) * BS + offset;
      ko[j] = keys[ki]; vo[j] = values[vi];
    )");
    auto result = kernel(
        {k, v, indices}, {{1, heads, count, dim}, {1, heads, count, dim}},
        {k.dtype(), k.dtype()}, {heads * count * dim, 1, 1}, {256, 1, 1},
        {{"HC", heads}, {"TC", count}, {"DC", dim}, {"BS", block_size}},
        std::nullopt, false, mlx::core::default_stream(mlx::core::Device::gpu));
    *out_keys = reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
    *out_values =
        reinterpret_cast<mlx_array *>(new array(std::move(result[1])));
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 page gather: " << e.what() << std::endl;
    return false;
  }
}
}
