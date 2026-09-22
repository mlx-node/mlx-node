#include "mlx_portable_qmm.h"
#include "mlx_common.h"
#include <map>

#ifdef MLX_NODE_METAL_ENABLED
#include "metal/common/quantized.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/transforms_impl.h"
namespace mlx::core::quantized_preamble {
const char *utils();
}
#endif

namespace mlx::core {
#ifdef MLX_NODE_METAL_ENABLED
namespace {
// A/B fragments keep the original BF16/FP16 values; C remains FP32. The
// linked Steel template otherwise widens A/B to FP32 before every MMA.
// This private copy leaves MLX's other GEMM and attention kernels untouched.
const std::string &portable_gemm() {
  static const std::string source = [] {
    std::string text = quantized_preamble::gemm();
    auto replace = [&](const std::string &before, const std::string &after) {
      auto pos = text.find(before);
      if (pos == std::string::npos)
        throw std::runtime_error("Review portable GEMM preamble");
      text.replace(pos, before.size(), after);
    };
    replace("MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;",
            "MMATile<T, TM, 1> Atile;");
    replace("MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;",
            "MMATile<T, 1, TN> Btile;");
    replace(
        "  METAL_FUNC static constexpr void mma(\n      thread frag_type& D,",
        R"(
  template <typename Input>
  METAL_FUNC static constexpr void mma(
      thread frag_type& D, thread metal::vec<Input, 2>& A,
      thread metal::vec<Input, 2>& B, thread frag_type& C) {
    mat_type D_mat, C_mat;
    metal::simdgroup_matrix<Input, 8, 8> A_mat, B_mat;
    reinterpret_cast<thread metal::vec<Input, 2>&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread metal::vec<Input, 2>&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;
    simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }
  METAL_FUNC static constexpr void mma(
      thread frag_type& D,)");
    return text;
  }();
  return source;
}
MTL::ComputePipelineState *portable_pipeline(Dtype dtype, int gs, int bits,
                                             const std::string &mode, int bm,
                                             int bn, int bk) {
  auto &device = metal::device(Device::gpu);
  const auto quant = string_to_quantization_mode(mode);
  const std::string type = dtype == bfloat16 ? "bfloat16_t" : "half";
  const std::string spec =
      type + "," + std::to_string(gs) + "," + std::to_string(bits) + "," +
      std::to_string(quant_super_ratio(quant)) + "," +
      (quant_has_sub_min(quant) ? "true" : "false") + ",false,false," +
      std::to_string(bm) + "," + std::to_string(bk) + "," + std::to_string(bn);
  const std::string name = "mlx_node_portable_" + mode + "_" + type + "_" +
                           std::to_string(bm) + "_" + std::to_string(bn) + "_" +
                           std::to_string(bk);
  auto *library = device.get_library(name, [&] {
    return std::string(quantized_preamble::utils()) + portable_gemm() +
           quantized_preamble::quantized_utils() +
           quantized_preamble::kquant() + "\ntemplate [[host_name(\"" + name +
           "\")]] [[kernel]] decltype(kquant_qmm_t<" + spec +
           ">) kquant_qmm_t<" + spec + ">;\n";
  });
  auto *kernel = device.get_kernel(name, library);
  if (kernel->threadExecutionWidth() != 32 ||
      kernel->maxTotalThreadsPerThreadgroup() < 128 ||
      kernel->staticThreadgroupMemoryLength() >
          device.mtl_device()->maxThreadgroupMemoryLength())
    throw std::runtime_error(
        "portable K-quant tile exceeds device pipeline limits");
  return kernel;
}

class PortableKQuant : public fast::Custom {
public:
  PortableKQuant(Stream stream, int gs, int bits, std::string mode, int bm,
                 int bn, int bk)
      : Custom(stream,
               [=](std::vector<array> inputs) {
                 return std::vector<array>{
                     quantized_matmul(inputs[0], inputs[1], inputs[2],
                                      inputs[3], true, gs, bits, mode, stream)};
               }),
        gs_(gs), bits_(bits), mode_(std::move(mode)), bm_(bm), bn_(bn),
        bk_(bk) {}
  void eval_cpu(const std::vector<array> &, std::vector<array> &) override {
    throw std::runtime_error("PortableKQuant requires Metal");
  }
  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    const auto &x = inputs[0];
    auto &out = outputs[0];
    const int k = x.shape(-1), n = inputs[1].shape(0), m = x.size() / k;
    out.set_data(allocator::malloc(out.nbytes()));
    auto &encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(
        portable_pipeline(x.dtype(), gs_, bits_, mode_, bm_, bn_, bk_));
    encoder.set_input_array(inputs[1], 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(x, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(k, 5);
    encoder.set_bytes(n, 6);
    encoder.set_bytes(m, 7);
    encoder.dispatch_threadgroups(
        MTL::Size((n + bn_ - 1) / bn_, (m + bm_ - 1) / bm_, 1),
        MTL::Size(32, 2, 2));
  }
  DEFINE_NAME(PortableKQuant)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const PortableKQuant &>(other);
    return gs_ == rhs.gs_ && bits_ == rhs.bits_ && mode_ == rhs.mode_ &&
           bm_ == rhs.bm_ && bn_ == rhs.bn_ && bk_ == rhs.bk_;
  }

private:
  int gs_, bits_;
  std::string mode_;
  int bm_, bn_, bk_;
};
} // namespace
#endif

std::optional<array> portable_kquant_matmul(const array &x, const array &w,
                                            const array &scales,
                                            const std::optional<array> &biases,
                                            bool transpose, int group_size,
                                            int bits, const std::string &mode) {
#ifdef MLX_NODE_METAL_ENABLED
  const char *setting = std::getenv("MLX_PORTABLE_KQUANT");
  if (setting && std::string(setting) == "0")
    return std::nullopt;
  constexpr int bm = 64, bn = 64, bk = 32; // 10 KiB of threadgroup scratch.
  if (default_device().type != Device::gpu || !metal::is_available() ||
      detail::in_grad_tracing() || !transpose || !biases || w.ndim() != 2 ||
      x.ndim() < 2 || x.shape(-1) <= 0 || x.shape(-1) % 256 ||
      x.size() / x.shape(-1) < 128 || w.shape(0) < 1024 ||
      !x.flags().row_contiguous || !w.flags().row_contiguous ||
      !scales.flags().row_contiguous || !biases->flags().row_contiguous ||
      (x.dtype() != bfloat16 && x.dtype() != float16) ||
      (mode != "q4k" && mode != "q5k" && mode != "q6k" && mode != "iq4xs"))
    return std::nullopt;
  if ((!setting || std::string(setting) != "1") && metal::is_nax_available())
    return std::nullopt;
  // Preserve stock split-K's arithmetic and small-matrix scheduling. Only
  // replace the unsplit QMM path. Decode and short verification stay stock.
  const size_t m = x.size() / x.shape(-1);
  if (((m + 31) / 32) * ((size_t(w.shape(0)) + 31) / 32) < 512)
    return std::nullopt;
  // The caller first constructs stock QMM, which validates packed geometry.
  // Cache both success and failure: an older compiler/device can keep the
  // established implementation without retrying a failed compile per layer.
  static std::mutex capability_mutex;
  static std::map<std::string, bool> capabilities;
  {
    std::lock_guard<std::mutex> lock(capability_mutex);
    const std::string key = mode + (x.dtype() == bfloat16 ? "_bf16" : "_fp16");
    auto found = capabilities.find(key);
    if (found == capabilities.end()) {
      bool available = false;
      try {
        portable_pipeline(x.dtype(), group_size, bits, mode, bm, bn, bk);
        available = true;
        std::cerr << "[mlx] portable K-quant prefill enabled: " << key
                  << " (64x64x32, 16-bit inputs, FP32 accumulation)\n";
      } catch (const std::exception &error) {
        std::cerr << "[mlx] portable K-quant prefill unavailable: " << key
                  << ": " << error.what() << '\n';
      }
      found = capabilities.emplace(key, available).first;
    }
    if (!found->second)
      return std::nullopt;
  }
  auto shape = x.shape();
  shape.back() = w.shape(0);
  return array(shape, x.dtype(),
               std::make_shared<PortableKQuant>(to_stream({}), group_size, bits,
                                                mode, bm, bn, bk),
               {x, w, scales, *biases});
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  (void)transpose;
  (void)group_size;
  (void)bits;
  (void)mode;
  return std::nullopt;
#endif
}
} // namespace mlx::core
