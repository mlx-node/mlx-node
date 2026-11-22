#include "cuda/qwen4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mlx_node::qwen4_cuda {
namespace {

constexpr char kQwen4Q8PrefillSource[] =
#include "cuda/qwen4_q8_prefill.cu.inc"
    ;

}  // namespace

std::string_view qwen4_q8_prefill_source() {
  return std::string_view(
      kQwen4Q8PrefillSource, sizeof(kQwen4Q8PrefillSource) - 1);
}

}  // namespace mlx_node::qwen4_cuda

namespace mlx::core::qwen4_ops {
namespace {

using namespace mlx_node::qwen4_cuda;

// Persistent per-(device, stream) workspace for direct Q8_0 prefill GEMM.
// Eliminates per-projection cudaMallocAsync churn across 325+ projections/chunk.
struct Qwen4Q8PrefillWorkspace {
  size_t cap_mk = 0;
  size_t cap_sc = 0;
  size_t cap_qs = 0;
  std::optional<array> xq;
  std::optional<array> x_sc;
  std::optional<array> x_qsum;

  static size_t bucket_size(
      size_t need, size_t b0, size_t b1, size_t b2, size_t b3, size_t step) {
    if (need <= b0) return b0;
    if (need <= b1) return b1;
    if (need <= b2) return b2;
    if (need <= b3) return b3;
    return ((need + step - 1) / step) * step;
  }

  void ensure(
      size_t need_mk, size_t need_sc, size_t need_qs,
      cu::CommandEncoder& encoder) {
    auto retire = [&](std::optional<array>& opt) {
      if (opt) {
        encoder.add_temporary(std::move(*opt));
        opt.reset();
      }
    };
    if (need_mk > cap_mk) {
      retire(xq);
      cap_mk = bucket_size(
          need_mk, 4u << 20, 16u << 20, 48u << 20, 96u << 20, 16u << 20);
      xq = alloc_persistent({static_cast<int>(cap_mk)}, int8, encoder);
    }
    if (need_sc > cap_sc) {
      retire(x_sc);
      cap_sc = bucket_size(need_sc, 65536, 262144, 524288, 1048576, 262144);
      x_sc = alloc_persistent({static_cast<int>(cap_sc)}, float32, encoder);
    }
    if (need_qs > cap_qs) {
      retire(x_qsum);
      cap_qs = bucket_size(need_qs, 262144, 1048576, 2097152, 4194304, 1048576);
      x_qsum = alloc_persistent({static_cast<int>(cap_qs)}, int16, encoder);
    }
  }
};

Qwen4Q8PrefillWorkspace& get_q8_prefill_workspace(int dev, int stream_idx) {
  static std::mutex mu;
  // Intentionally leaked so GPU buffers are not destroyed during teardown.
  static auto* map = new std::unordered_map<uint64_t, Qwen4Q8PrefillWorkspace>();
  uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(dev)) << 32) |
      static_cast<uint32_t>(stream_idx);
  std::lock_guard<std::mutex> lk(mu);
  return (*map)[key];
}

// ============================================================================
// Direct Q8_0 Prefill Tensor-Core GEMM (INT8 128x128 / 128x64 or Exact BF16)
// ============================================================================
class Qwen4Q8PrefillCuda final : public Primitive {
 public:
  Qwen4Q8PrefillCuda(Stream stream, int gs_shift, bool general_bias)
      : Primitive(stream), gs_shift_(gs_shift), general_bias_(general_bias) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4Q8PrefillCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array sc = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    std::optional<array> bi;
    if (general_bias_) {
      bi = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    }

    const int N = w.shape(0);
    const int K = w.shape(1) * 4;
    const int M = static_cast<int>(x.size() / static_cast<size_t>(K));
    if (M < 1 || M > 65535 || N < 2 || (N & 1) != 0 || K < 32 || (K & 31) != 0) {
      throw std::invalid_argument("Qwen4Q8PrefillCuda: unsupported shape");
    }

    auto& out = outputs[0];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    if (gs_shift_ == -2) {
      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(sc);
      if (general_bias_) {
        encoder.set_input_array(*bi);
      }
      encoder.set_output_array(out);
      const char* entry = general_bias_ ? "qwen4_q8_prefill_bf16<true>"
                                        : "qwen4_q8_prefill_bf16<false>";
      auto fn = cu::jit_kernel(encoder.device(), qwen4_q8_prefill_source(), entry);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(sc);
      if (general_bias_) {
        args.append(*bi);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      const uint32_t g64 =
          static_cast<uint32_t>(((M + 127) / 128) * ((N + 63) / 64));
      encoder.add_kernel_node_raw(
          fn, dim3(g64, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
      return;
    }

    const size_t need_mk = static_cast<size_t>(M) * static_cast<size_t>(K);
    const size_t need_sc = (gs_shift_ == 0)
        ? static_cast<size_t>(M)
        : static_cast<size_t>(M) * static_cast<size_t>(K >> gs_shift_);
    const size_t need_qs = general_bias_
        ? static_cast<size_t>(M) * static_cast<size_t>(K >> 5)
        : 0;
    auto& ws = get_q8_prefill_workspace(
        encoder.device().cuda_device(), stream().index);
    ws.ensure(need_mk, need_sc, need_qs, encoder);
    const array& xq = *ws.xq;
    const array& x_sc = *ws.x_sc;

    const int gs_idx = (gs_shift_ == 0) ? 0 : (gs_shift_ == 7 ? 1 : 2);
    const int gb_idx = general_bias_ ? 1 : 0;

    // 1. Quantize BF16 activations -> INT8 + per-row / grouped scales (+ optional qsum)
    {
      static constexpr const char* kQuantEntries[3][2] = {
          {"qwen4_q8_quantize_x<0, false>", "qwen4_q8_quantize_x<0, true>"},
          {"qwen4_q8_quantize_x<7, false>", "qwen4_q8_quantize_x<7, true>"},
          {"qwen4_q8_quantize_x<8, false>", "qwen4_q8_quantize_x<8, true>"},
      };
      encoder.set_input_array(x);
      encoder.set_output_array(xq);
      encoder.set_output_array(x_sc);
      if (general_bias_) {
        encoder.set_output_array(*ws.x_qsum);
      }
      auto k_quant = cu::jit_kernel(
          encoder.device(), qwen4_q8_prefill_source(),
          kQuantEntries[gs_idx][gb_idx]);
      cu::KernelArgs args;
      args.append(x);
      args.append(xq);
      args.append(x_sc);
      if (general_bias_) {
        args.append(*ws.x_qsum);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(M);
      args.append<int32_t>(K);
      uint32_t grid_q = 0;
      if (gs_shift_ == 0) {
        grid_q = static_cast<uint32_t>(M);
      } else if (gs_shift_ == 7) {
        grid_q = static_cast<uint32_t>(
            (static_cast<uint64_t>(M) * static_cast<uint64_t>(K >> 7) * 16 + 255) /
            256);
      } else {
        grid_q = static_cast<uint32_t>(
            (static_cast<uint64_t>(M) * static_cast<uint64_t>(K >> 8) * 32 + 255) /
            256);
      }
      encoder.add_kernel_node_raw(
          k_quant, dim3(grid_q, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
    }

    // 2. Direct Q8_0 INT8 Tensor-Core GEMM (128x128 dynamic SMEM vs 128x64 static SMEM)
    {
      const int m_tiles = (M + 127) / 128;
      const int n_tiles_128 = (N + 127) / 128;
      const bool use_128x128 = (K % 128 == 0) && (N >= 512) &&
          ((N & 127) == 0 || N >= 1024) &&
          (m_tiles * n_tiles_128 >= 96);

      static constexpr const char* kGemm128Entries[3][2] = {
          {"qwen4_q8_prefill_int8_128x128<0, false>",
           "qwen4_q8_prefill_int8_128x128<0, true>"},
          {"qwen4_q8_prefill_int8_128x128<7, false>",
           "qwen4_q8_prefill_int8_128x128<7, true>"},
          {"qwen4_q8_prefill_int8_128x128<8, false>",
           "qwen4_q8_prefill_int8_128x128<8, true>"},
      };
      static constexpr const char* kGemm64Entries[3][2] = {
          {"qwen4_q8_prefill_int8_128x64<0, false>",
           "qwen4_q8_prefill_int8_128x64<0, true>"},
          {"qwen4_q8_prefill_int8_128x64<7, false>",
           "qwen4_q8_prefill_int8_128x64<7, true>"},
          {"qwen4_q8_prefill_int8_128x64<8, false>",
           "qwen4_q8_prefill_int8_128x64<8, true>"},
      };

      encoder.set_input_array(xq);
      encoder.set_input_array(x_sc);
      if (general_bias_) {
        encoder.set_input_array(*ws.x_qsum);
      }
      encoder.set_input_array(w);
      encoder.set_input_array(sc);
      if (general_bias_) {
        encoder.set_input_array(*bi);
      }
      encoder.set_output_array(out);
      // Mark persistent workspace buffers as outputs of the GEMM so CUDA Graph
      // dependency tracking enforces strict happens-before before reuse.
      encoder.set_output_array(xq);
      encoder.set_output_array(x_sc);
      if (general_bias_) {
        encoder.set_output_array(*ws.x_qsum);
      }

      cu::KernelArgs args;
      args.append(xq);
      args.append(x_sc);
      if (general_bias_) {
        args.append(*ws.x_qsum);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append(w);
      args.append(sc);
      if (general_bias_) {
        args.append(*bi);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);

      if (use_128x128) {
        constexpr int kSm128Sym =
            2 * (128 * 128 + 128 * 128 + 4 * 128 * 4 + 16);
        constexpr int kSm128Gen =
            2 * (128 * 128 + 128 * 128 + 8 * 128 * 4);
        const int smem_bytes = general_bias_ ? kSm128Gen : kSm128Sym;
        auto fn = cu::jit_kernel(
            encoder.device(), qwen4_q8_prefill_source(),
            kGemm128Entries[gs_idx][gb_idx]);
        ensure_dynamic_smem(fn, smem_bytes);
        const uint32_t g128 = static_cast<uint32_t>(m_tiles * n_tiles_128);
        encoder.add_kernel_node_raw(
            fn, dim3(g128, 1, 1), dim3(256, 1, 1), {},
            static_cast<uint32_t>(smem_bytes), args.args());
      } else {
        auto fn = cu::jit_kernel(
            encoder.device(), qwen4_q8_prefill_source(),
            kGemm64Entries[gs_idx][gb_idx]);
        const uint32_t g64 =
            static_cast<uint32_t>(m_tiles * ((N + 63) / 64));
        encoder.add_kernel_node_raw(
            fn, dim3(g64, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
      }
    }
  }

  DEFINE_NAME(Qwen4Q8PrefillCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4Q8PrefillCuda*>(&other);
    return rhs && rhs->gs_shift_ == gs_shift_ &&
        rhs->general_bias_ == general_bias_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() = inputs[1].shape(0);
    return {std::move(s)};
  }

 private:
  int gs_shift_;
  bool general_bias_;
};

// ============================================================================
// Fused Prefill mHC Up-Mix (BF16 MMA, Zero [M, 10240] DRAM Buffer!)
// ============================================================================
class Qwen4HcUpMixPrefillCuda final : public Primitive {
 public:
  explicit Qwen4HcUpMixPrefillCuda(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4HcUpMixPrefillCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array down_act =
        cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array normed =
        cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array w_up =
        cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array s_up =
        cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);

    const int M = static_cast<int>(down_act.size() / kRank);
    if (M < 1 || M > 65535 ||
        normed.size() != static_cast<size_t>(M) * kW) {
      throw std::invalid_argument("Qwen4HcUpMixPrefillCuda: unsupported shape");
    }

    auto& out = outputs[0];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(down_act);
    encoder.set_input_array(normed);
    encoder.set_input_array(w_up);
    encoder.set_input_array(s_up);
    encoder.set_output_array(out);

    const bool use_v2_i8 = (M > 512);
    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_q8_prefill_source(),
        use_v2_i8 ? "qwen4_hc_up_mix_prefill_v2"
                  : "qwen4_hc_up_mix_prefill<false>");
    cu::KernelArgs args;
    args.append(down_act);
    args.append(w_up);
    args.append(s_up);
    args.append(normed);
    args.append(out);
    args.append<int32_t>(M);
    const uint32_t g_hc = use_v2_i8
        ? static_cast<uint32_t>(((M + 63) / 64) * 80)
        : static_cast<uint32_t>(((M + 127) / 128) * 160);
    encoder.add_kernel_node_raw(
        fn, dim3(g_hc, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Qwen4HcUpMixPrefillCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Qwen4HcUpMixPrefillCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() = kH;
    return {std::move(s)};
  }
};

}  // namespace
}  // namespace mlx::core::qwen4_ops

extern "C" mlx_array* mlx_qwen4_q8_prefill_cuda(
    mlx_array* x_h, mlx_array* w_h, mlx_array* scales_h, mlx_array* biases_h,
    int32_t gs_shift, bool general_bias) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    if (!x_h || !w_h || !scales_h || (general_bias && !biases_h) ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& w = *reinterpret_cast<array*>(w_h);
    const auto& sc = *reinterpret_cast<array*>(scales_h);
    if (x.dtype() != bfloat16 || x.ndim() < 2 || w.dtype() != uint32 ||
        w.ndim() != 2 || sc.dtype() != float16 || sc.ndim() != 2) {
      return nullptr;
    }
    const int N = w.shape(0);
    const int K = w.shape(1) * 4;
    if (N < 2 || (N & 1) != 0 || K < 32 || (K & 31) != 0 ||
        x.shape(-1) != K || sc.shape(0) != N || sc.shape(1) != (K >> 5)) {
      return nullptr;
    }
    const int64_t m64 = static_cast<int64_t>(x.size()) / K;
    if (m64 < 1 || m64 > 65535 ||
        static_cast<uint64_t>(m64) * static_cast<uint64_t>(K) >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return nullptr;
    }

    int eff_gs = gs_shift;
    if (eff_gs == -1) {
      if ((K % 256) == 0) {
        eff_gs = 8;
      } else if ((K % 128) == 0) {
        eff_gs = 7;
      } else if ((K % 64) == 0) {
        eff_gs = 0;
      } else {
        eff_gs = -2;
      }
    }
    if (eff_gs == 0) {
      if ((K % 64) != 0) return nullptr;
    } else if (eff_gs == 7) {
      if ((K % 128) != 0) return nullptr;
    } else if (eff_gs == 8) {
      if ((K % 256) != 0) return nullptr;
    } else if (eff_gs != -2) {
      return nullptr;
    }

    std::vector<array> inputs;
    inputs.reserve(general_bias ? 4 : 3);
    inputs.push_back(x);
    inputs.push_back(w);
    inputs.push_back(sc);
    if (biases_h) {
      const auto& bi = *reinterpret_cast<array*>(biases_h);
      if (bi.dtype() != float16 || bi.ndim() != 2 || bi.shape(0) != N ||
          bi.shape(1) != (K >> 5)) {
        return nullptr;
      }
      if (general_bias) {
        inputs.push_back(bi);
      }
    }

    Shape out_shape = x.shape();
    out_shape.back() = N;
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Qwen4Q8PrefillCuda>(
            default_stream(Device::gpu), eff_gs, general_bias),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_q8_prefill_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_qwen4_hc_up_mix_prefill_cuda(
    mlx_array* down_act_h, mlx_array* normed_h,
    mlx_array* w_up_h, mlx_array* s_up_h, mlx_array* b_up_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    using namespace mlx_node::qwen4_cuda;
    if (!down_act_h || !normed_h || !w_up_h || !s_up_h ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& down_act = *reinterpret_cast<array*>(down_act_h);
    const auto& normed = *reinterpret_cast<array*>(normed_h);
    const auto& w_up = *reinterpret_cast<array*>(w_up_h);
    const auto& s_up = *reinterpret_cast<array*>(s_up_h);
    if (down_act.dtype() != bfloat16 || down_act.ndim() < 2 ||
        down_act.shape(-1) != kRank || normed.dtype() != bfloat16 ||
        normed.ndim() < 2 || normed.shape(-1) != kW) {
      return nullptr;
    }
    const int64_t m64 = static_cast<int64_t>(down_act.size()) / kRank;
    if (m64 < 1 || m64 > 65535 ||
        normed.size() != static_cast<size_t>(m64) * kW) {
      return nullptr;
    }
    if (w_up.dtype() != uint32 || w_up.ndim() != 2 ||
        w_up.shape(0) != kW || w_up.shape(1) != kRank / 4 ||
        s_up.dtype() != float16 || s_up.ndim() != 2 ||
        s_up.shape(0) != kW || s_up.shape(1) != kRank / 32) {
      return nullptr;
    }
    if (b_up_h) {
      const auto& b_up = *reinterpret_cast<array*>(b_up_h);
      if (b_up.dtype() != float16 || b_up.ndim() != 2 ||
          b_up.shape(0) != kW || b_up.shape(1) != kRank / 32) {
        return nullptr;
      }
    }
    Shape out_shape = down_act.shape();
    out_shape.back() = kH;
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Qwen4HcUpMixPrefillCuda>(default_stream(Device::gpu)),
        {down_act, normed, w_up, s_up});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_hc_up_mix_prefill_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" mlx_array* mlx_qwen4_q8_prefill_cuda(
    mlx_array* x_h, mlx_array* w_h, mlx_array* scales_h, mlx_array* biases_h,
    int32_t gs_shift, bool general_bias) {
  (void)x_h; (void)w_h; (void)scales_h; (void)biases_h;
  (void)gs_shift; (void)general_bias;
  return nullptr;
}

extern "C" mlx_array* mlx_qwen4_hc_up_mix_prefill_cuda(
    mlx_array* down_act_h, mlx_array* normed_h,
    mlx_array* w_up_h, mlx_array* s_up_h, mlx_array* b_up_h) {
  (void)down_act_h; (void)normed_h; (void)w_up_h; (void)s_up_h; (void)b_up_h;
  return nullptr;
}

#endif
