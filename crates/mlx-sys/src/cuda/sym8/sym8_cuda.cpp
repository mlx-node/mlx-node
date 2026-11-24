#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_common.h"
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mlx::core::sym8_ops {
namespace {

constexpr char kSym8HadamardSrc[] =
#include "cuda/sym8/sym8_hadamard.cu.inc"
    ;

constexpr char kSym8WeightQuantSrc[] =
#include "cuda/sym8/sym8_weight_quant.cu.inc"
    ;

constexpr char kSym8WeightQuantKQuantSrc[] =
#include "cuda/sym8/sym8_weight_quant_kquant.cu.inc"
    ;

constexpr char kSym8ImmaLargeSrc[] =
#include "cuda/sym8/sym8_imma_large.cu.inc"
    ;

constexpr char kSym8ImmaSmallSrc[] =
#include "cuda/sym8/sym8_imma_small.cu.inc"
    ;

constexpr char kSym8RmsNormHadamardSrc[] =
#include "cuda/sym8/sym8_rmsnorm_hadamard.cu.inc"
    ;

inline std::string_view sym8_hadamard_src() noexcept {
  return std::string_view(kSym8HadamardSrc, sizeof(kSym8HadamardSrc) - 1);
}

inline std::string_view sym8_weight_quant_src() noexcept {
  return std::string_view(kSym8WeightQuantSrc, sizeof(kSym8WeightQuantSrc) - 1);
}

inline std::string_view sym8_weight_quant_kquant_src() noexcept {
  return std::string_view(
      kSym8WeightQuantKQuantSrc, sizeof(kSym8WeightQuantKQuantSrc) - 1);
}

inline std::string_view sym8_imma_large_src() noexcept {
  return std::string_view(kSym8ImmaLargeSrc, sizeof(kSym8ImmaLargeSrc) - 1);
}

inline std::string_view sym8_imma_small_src() noexcept {
  return std::string_view(kSym8ImmaSmallSrc, sizeof(kSym8ImmaSmallSrc) - 1);
}

inline std::string_view sym8_rmsnorm_hadamard_src() noexcept {
  return std::string_view(
      kSym8RmsNormHadamardSrc, sizeof(kSym8RmsNormHadamardSrc) - 1);
}

inline int float_dtype_code(Dtype dt) noexcept {
  if (dt == bfloat16) return 0;
  if (dt == float16) return 1;
  if (dt == float32) return 2;
  return -1;
}

inline int scale_dtype_code(Dtype dt) noexcept {
  if (dt == float32) return 0;
  if (dt == bfloat16) return 1;
  if (dt == float16) return 2;
  return -1;
}

inline int max_optin_smem_bytes(int dev_ordinal) {
  static std::mutex mu;
  static std::unordered_map<int, int> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(dev_ordinal);
  if (it != cache.end()) {
    return it->second;
  }
  int max_optin = 49152;
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, dev_ordinal) == CUDA_SUCCESS) {
    int val = 0;
    if (cuDeviceGetAttribute(
            &val, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev) ==
        CUDA_SUCCESS) {
      max_optin = val;
    }
  }
  cache[dev_ordinal] = max_optin;
  return max_optin;
}

inline int static_smem_bytes(CUfunction fn) {
  static std::mutex mu;
  static std::unordered_map<CUfunction, int> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(fn);
  if (it != cache.end()) {
    return it->second;
  }
  int val = 0;
  CUresult rc =
      cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);
  if (rc != CUDA_SUCCESS || val < 0) {
    throw std::runtime_error(
        "Sym8Cuda: cuFuncGetAttribute(SHARED_SIZE_BYTES) failed");
  }
  cache[fn] = val;
  return val;
}

inline void ensure_dynamic_smem(CUfunction fn, int smem_bytes) {
  if (smem_bytes <= 0) {
    return;
  }
  const int static_smem = static_smem_bytes(fn);
  if (static_cast<int64_t>(smem_bytes) + static_cast<int64_t>(static_smem) <=
      49152) {
    return;
  }
  static std::mutex mu;
  static std::unordered_map<CUfunction, int> configured;
  std::lock_guard<std::mutex> lk(mu);
  auto it = configured.find(fn);
  if (it != configured.end() && it->second >= smem_bytes) {
    return;
  }
  CUresult rc = cuFuncSetAttribute(
      fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);
  if (rc != CUDA_SUCCESS) {
    throw std::runtime_error(
        "Sym8Cuda: cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) failed");
  }
  configured[fn] = smem_bytes;
}

inline uint32_t checked_grid_x(
    int64_t m64, int64_t n64, int64_t bm, int64_t bn) {
  const int64_t num_m_tiles = (m64 + bm - 1) / bm;
  const int64_t num_n_tiles = (n64 + bn - 1) / bn;
  constexpr int64_t kMaxGridX = std::numeric_limits<int32_t>::max();
  if (num_m_tiles < 1 || num_n_tiles < 1 ||
      num_m_tiles > kMaxGridX / num_n_tiles) {
    throw std::invalid_argument(
        "Sym8ImmaGemmCuda: grid_x exceeds INT32_MAX");
  }
  return static_cast<uint32_t>(num_m_tiles * num_n_tiles);
}

// ============================================================================
// 1. Fused (Optional SwiGLU +) Optional Signs+FWHT-1024 + Per-Token INT8 Quant
// ============================================================================
class Sym8HadamardQuantCuda final : public Primitive {
 public:
  Sym8HadamardQuantCuda(Stream stream, bool has_signs, bool fused_swiglu)
      : Primitive(stream), has_signs_(has_signs), fused_swiglu_(fused_swiglu) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Sym8HadamardQuantCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    std::optional<array> signs;
    if (has_signs_) {
      signs = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    }

    const int in_dtype = float_dtype_code(x.dtype());
    if (in_dtype < 0 || x.ndim() < 2) {
      throw std::invalid_argument("Sym8HadamardQuantCuda: unsupported input dtype or rank");
    }
    const int64_t in_k64 = x.shape(-1);
    if (in_k64 < 1 || (fused_swiglu_ && (in_k64 & 1) != 0)) {
      throw std::invalid_argument("Sym8HadamardQuantCuda: invalid K dimension");
    }
    const int64_t k64 = fused_swiglu_ ? (in_k64 / 2) : in_k64;
    const int64_t m64 = static_cast<int64_t>(x.size()) / in_k64;
    if (m64 < 1 || m64 > std::numeric_limits<int32_t>::max() ||
        k64 < 1 || k64 > std::numeric_limits<int32_t>::max() ||
        (has_signs_ && (k64 % 1024) != 0)) {
      throw std::invalid_argument("Sym8HadamardQuantCuda: unsupported shape");
    }
    const int M = static_cast<int>(m64);
    const int K = static_cast<int>(k64);

    auto& out_xq = outputs[0];
    auto& out_sx = outputs[1];
    out_xq.set_data(cu::malloc_async(out_xq.nbytes(), encoder));
    out_sx.set_data(cu::malloc_async(out_sx.nbytes(), encoder));

    encoder.set_input_array(x);
    if (has_signs_) {
      encoder.set_input_array(*signs);
    }
    encoder.set_output_array(out_xq);
    encoder.set_output_array(out_sx);

    static constexpr const char* kFp32Entries[3][2][2] = {
        {{"sym8_hadamard_quant_int8_fp32smem<0, false, false>",
          "sym8_hadamard_quant_int8_fp32smem<0, false, true>"},
         {"sym8_hadamard_quant_int8_fp32smem<0, true, false>",
          "sym8_hadamard_quant_int8_fp32smem<0, true, true>"}},
        {{"sym8_hadamard_quant_int8_fp32smem<1, false, false>",
          "sym8_hadamard_quant_int8_fp32smem<1, false, true>"},
         {"sym8_hadamard_quant_int8_fp32smem<1, true, false>",
          "sym8_hadamard_quant_int8_fp32smem<1, true, true>"}},
        {{"sym8_hadamard_quant_int8_fp32smem<2, false, false>",
          "sym8_hadamard_quant_int8_fp32smem<2, false, true>"},
         {"sym8_hadamard_quant_int8_fp32smem<2, true, false>",
          "sym8_hadamard_quant_int8_fp32smem<2, true, true>"}},
    };
    static constexpr const char* kBf16Entries[3][2][2] = {
        {{"sym8_hadamard_quant_int8_bf16smem<0, false, false>",
          "sym8_hadamard_quant_int8_bf16smem<0, false, true>"},
         {"sym8_hadamard_quant_int8_bf16smem<0, true, false>",
          "sym8_hadamard_quant_int8_bf16smem<0, true, true>"}},
        {{"sym8_hadamard_quant_int8_bf16smem<1, false, false>",
          "sym8_hadamard_quant_int8_bf16smem<1, false, true>"},
         {"sym8_hadamard_quant_int8_bf16smem<1, true, false>",
          "sym8_hadamard_quant_int8_bf16smem<1, true, true>"}},
        {{"sym8_hadamard_quant_int8_bf16smem<2, false, false>",
          "sym8_hadamard_quant_int8_bf16smem<2, false, true>"},
         {"sym8_hadamard_quant_int8_bf16smem<2, true, false>",
          "sym8_hadamard_quant_int8_bf16smem<2, true, true>"}},
    };
    static constexpr const char* kTwoPassEntries[3][2][2] = {
        {{"sym8_hadamard_quant_int8_2pass<0, false, false>",
          "sym8_hadamard_quant_int8_2pass<0, false, true>"},
         {"sym8_hadamard_quant_int8_2pass<0, true, false>",
          "sym8_hadamard_quant_int8_2pass<0, true, true>"}},
        {{"sym8_hadamard_quant_int8_2pass<1, false, false>",
          "sym8_hadamard_quant_int8_2pass<1, false, true>"},
         {"sym8_hadamard_quant_int8_2pass<1, true, false>",
          "sym8_hadamard_quant_int8_2pass<1, true, true>"}},
        {{"sym8_hadamard_quant_int8_2pass<2, false, false>",
          "sym8_hadamard_quant_int8_2pass<2, false, true>"},
         {"sym8_hadamard_quant_int8_2pass<2, true, false>",
          "sym8_hadamard_quant_int8_2pass<2, true, true>"}},
    };

    const int hs = has_signs_ ? 1 : 0;
    const int sw = fused_swiglu_ ? 1 : 0;
    const int wht_bytes = has_signs_ ? (4 * 260 * 4) : 0;
    int smem_bytes = 0;
    CUfunction fn = nullptr;

    if ((K & 7) == 0 && K <= 8192) {
      smem_bytes = wht_bytes + K * 4;
      fn = cu::jit_kernel(
          encoder.device(), sym8_hadamard_src(), kFp32Entries[in_dtype][hs][sw]);
      ensure_dynamic_smem(fn, smem_bytes);
    } else if ((K & 7) == 0 && K <= 46080) {
      const int need_smem = wht_bytes + K * 2;
      const int max_optin = max_optin_smem_bytes(encoder.device().cuda_device());
      if (need_smem <= max_optin) {
        CUfunction bf16_fn = cu::jit_kernel(
            encoder.device(), sym8_hadamard_src(), kBf16Entries[in_dtype][hs][sw]);
        const int static_smem = static_smem_bytes(bf16_fn);
        if (static_cast<int64_t>(need_smem) + static_cast<int64_t>(static_smem) <=
            static_cast<int64_t>(max_optin)) {
          smem_bytes = need_smem;
          fn = bf16_fn;
          ensure_dynamic_smem(fn, smem_bytes);
        }
      }
      if (!fn) {
        smem_bytes = 0;
        fn = cu::jit_kernel(
            encoder.device(), sym8_hadamard_src(), kTwoPassEntries[in_dtype][hs][sw]);
      }
    } else {
      smem_bytes = 0;
      fn = cu::jit_kernel(
          encoder.device(), sym8_hadamard_src(), kTwoPassEntries[in_dtype][hs][sw]);
    }

    cu::KernelArgs args;
    args.append(x);
    if (has_signs_) {
      args.append(*signs);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out_xq);
    args.append(out_sx);
    args.append<int32_t>(M);
    args.append<int32_t>(K);

    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(M), 1, 1), dim3(128, 1, 1), {},
        static_cast<uint32_t>(smem_bytes), args.args());
  }

  DEFINE_NAME(Sym8HadamardQuantCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Sym8HadamardQuantCuda*>(&other);
    return rhs && rhs->has_signs_ == has_signs_ &&
        rhs->fused_swiglu_ == fused_swiglu_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape q_shape = inputs[0].shape();
    if (fused_swiglu_) {
      q_shape.back() /= 2;
    }
    Shape s_shape = inputs[0].shape();
    s_shape.pop_back();
    return {std::move(q_shape), std::move(s_shape)};
  }

 private:
  bool has_signs_;
  bool fused_swiglu_;
};

// ============================================================================
// 2. Fused-Epilogue Sym8 IMMA GEMM (int8 x int8 -> int32 -> BF16)
// ============================================================================
class Sym8ImmaGemmCuda final : public Primitive {
 public:
  explicit Sym8ImmaGemmCuda(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Sym8ImmaGemmCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x_i8 = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w_i8 = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array s_x = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array s_w = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);

    if (x_i8.ndim() < 2 || w_i8.ndim() != 2) {
      throw std::invalid_argument("Sym8ImmaGemmCuda: unsupported rank");
    }
    const int64_t n64 = w_i8.shape(0);
    const int64_t k64 = w_i8.shape(1);
    if (n64 < 1 || n64 > std::numeric_limits<int32_t>::max() ||
        k64 < 1 || k64 > std::numeric_limits<int32_t>::max() ||
        x_i8.shape(-1) != k64) {
      throw std::invalid_argument("Sym8ImmaGemmCuda: unsupported shape");
    }
    const size_t x_elems = x_i8.size();
    const size_t k_sz = static_cast<size_t>(k64);
    if (x_elems == 0 || (x_elems % k_sz) != 0) {
      throw std::invalid_argument("Sym8ImmaGemmCuda: unsupported shape");
    }
    const size_t m_sz = x_elems / k_sz;
    if (m_sz < 1 ||
        m_sz > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        s_x.size() != m_sz ||
        s_w.size() != static_cast<size_t>(n64)) {
      throw std::invalid_argument("Sym8ImmaGemmCuda: unsupported shape");
    }
    const int64_t m64 = static_cast<int64_t>(m_sz);
    const int M = static_cast<int>(m64);
    const int N = static_cast<int>(n64);
    const int K = static_cast<int>(k64);

    const int max_optin = max_optin_smem_bytes(encoder.device().cuda_device());
    std::string_view src;
    const char* entry = nullptr;
    int64_t bm = 128;
    int64_t bn = 128;
    dim3 block(256, 1, 1);
    int smem_bytes = 0;

    if ((K % 64) != 0 || (N % 8) != 0) {
      src = sym8_imma_small_src();
      entry = "sym8_gemm_fallback";
      bm = 16;
      bn = 16;
      block = dim3(16, 16, 1);
    } else if (M <= 512 && (K % 128) == 0) {
      if (M >= 280 && N >= 16384 && max_optin >= 98304) {
        src = sym8_imma_large_src();
        entry = "sym8_imma_128x128_k128_s3_gm8";
        bm = 128;
        bn = 128;
        block = dim3(256, 1, 1);
        smem_bytes = 98304;
      } else if (K <= 4096) {
        src = sym8_imma_small_src();
        entry = "sym8_imma_64x128_k128_s2";
        bm = 64;
        bn = 128;
        block = dim3(128, 1, 1);
      } else if (K >= 12288 && N >= 5120 && M <= 256 && max_optin >= 98304) {
        src = sym8_imma_large_src();
        entry = "sym8_imma_128x128_k128_s3_gm8";
        bm = 128;
        bn = 128;
        block = dim3(256, 1, 1);
        smem_bytes = 98304;
      } else if (K >= 12288 && M >= 258 && max_optin >= 73728) {
        src = sym8_imma_small_src();
        entry = "sym8_imma_128x64_k128_s3";
        bm = 128;
        bn = 64;
        block = dim3(256, 1, 1);
        smem_bytes = 73728;
      } else {
        src = sym8_imma_small_src();
        entry = "sym8_imma_64x64_k128_s3";
        bm = 64;
        bn = 64;
        block = dim3(128, 1, 1);
      }
    } else if (M > 512 && K > 8192 && (K % 128) == 0 && max_optin >= 98304) {
      src = sym8_imma_large_src();
      entry = "sym8_imma_128x128_k128_s3";
      bm = 128;
      bn = 128;
      block = dim3(256, 1, 1);
      smem_bytes = 98304;
    } else {
      src = sym8_imma_large_src();
      entry = "sym8_imma_128x128_k64_s3";
      bm = 128;
      bn = 128;
      block = dim3(256, 1, 1);
    }

    const uint32_t grid_x = checked_grid_x(m64, n64, bm, bn);
    auto fn = cu::jit_kernel(encoder.device(), src, entry);
    if (smem_bytes > 0) {
      ensure_dynamic_smem(fn, smem_bytes);
    }

    auto& out = outputs[0];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    encoder.set_input_array(x_i8);
    encoder.set_input_array(w_i8);
    encoder.set_input_array(s_x);
    encoder.set_input_array(s_w);
    encoder.set_output_array(out);

    cu::KernelArgs args;
    args.append(x_i8);
    args.append(w_i8);
    args.append(s_x);
    args.append(s_w);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(N);
    args.append<int32_t>(K);

    encoder.add_kernel_node_raw(
        fn, dim3(grid_x, 1, 1), block, {},
        static_cast<uint32_t>(smem_bytes), args.args());
  }

  DEFINE_NAME(Sym8ImmaGemmCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Sym8ImmaGemmCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape out_shape = inputs[0].shape();
    out_shape.back() = inputs[1].shape(0);
    return {std::move(out_shape)};
  }
};

// ============================================================================
// 3. Bit-Exact FP32 Weight Quantizer: Packed Affine Q8 -> Sym8
// ============================================================================
class Sym8QuantizeWeightQ8Cuda final : public Primitive {
 public:
  Sym8QuantizeWeightQ8Cuda(Stream stream, bool has_biases, bool has_signs)
      : Primitive(stream), has_biases_(has_biases), has_signs_(has_signs) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Sym8QuantizeWeightQ8Cuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array w_u32 = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w_scales = cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder);
    std::optional<array> w_biases;
    std::optional<array> signs;
    size_t idx = 2;
    if (has_biases_) {
      w_biases = cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    }
    if (has_signs_) {
      signs = cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    }

    const int sb_dtype = scale_dtype_code(w_scales.dtype());
    const int64_t n64 = w_u32.shape(0);
    const int64_t k64 = static_cast<int64_t>(w_u32.shape(1)) * 4;
    if (sb_dtype < 0 || n64 < 1 || n64 > std::numeric_limits<int32_t>::max() ||
        k64 < 32 || k64 > std::numeric_limits<int32_t>::max() ||
        (k64 % 32) != 0 || (has_signs_ && (k64 % 1024) != 0)) {
      throw std::invalid_argument("Sym8QuantizeWeightQ8Cuda: unsupported shape or dtype");
    }
    const int N = static_cast<int>(n64);
    const int K = static_cast<int>(k64);

    auto& out_q = outputs[0];
    auto& out_s = outputs[1];
    out_q.set_data(cu::malloc_async(out_q.nbytes(), encoder));
    out_s.set_data(cu::malloc_async(out_s.nbytes(), encoder));

    encoder.set_input_array(w_u32);
    encoder.set_input_array(w_scales);
    if (has_biases_) encoder.set_input_array(*w_biases);
    if (has_signs_) encoder.set_input_array(*signs);
    encoder.set_output_array(out_q);
    encoder.set_output_array(out_s);

    static constexpr const char* kEntries[3][2][2] = {
        {{"sym8_quantize_weight_q8<0, false, false>",
          "sym8_quantize_weight_q8<0, false, true>"},
         {"sym8_quantize_weight_q8<0, true, false>",
          "sym8_quantize_weight_q8<0, true, true>"}},
        {{"sym8_quantize_weight_q8<1, false, false>",
          "sym8_quantize_weight_q8<1, false, true>"},
         {"sym8_quantize_weight_q8<1, true, false>",
          "sym8_quantize_weight_q8<1, true, true>"}},
        {{"sym8_quantize_weight_q8<2, false, false>",
          "sym8_quantize_weight_q8<2, false, true>"},
         {"sym8_quantize_weight_q8<2, true, false>",
          "sym8_quantize_weight_q8<2, true, true>"}},
    };
    auto fn = cu::jit_kernel(
        encoder.device(), sym8_weight_quant_src(),
        kEntries[sb_dtype][has_biases_ ? 1 : 0][has_signs_ ? 1 : 0]);

    cu::KernelArgs args;
    args.append(w_u32);
    args.append(w_scales);
    if (has_biases_) {
      args.append(*w_biases);
    } else {
      args.append<CUdeviceptr>(0);
    }
    if (has_signs_) {
      args.append(*signs);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out_q);
    args.append(out_s);
    args.append<int32_t>(N);
    args.append<int32_t>(K);

    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(N), 1, 1), dim3(128, 1, 1), {}, 0,
        args.args());
  }

  DEFINE_NAME(Sym8QuantizeWeightQ8Cuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Sym8QuantizeWeightQ8Cuda*>(&other);
    return rhs && rhs->has_biases_ == has_biases_ && rhs->has_signs_ == has_signs_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int N = inputs[0].shape(0);
    const int K = inputs[0].shape(1) * 4;
    return {{N, K}, {N}};
  }

 private:
  bool has_biases_;
  bool has_signs_;
};

// ============================================================================
// 4. Bit-Exact FP32 Weight Quantizer: Dense BF16/FP16/FP32 -> Sym8
// ============================================================================
class Sym8QuantizeWeightDenseCuda final : public Primitive {
 public:
  Sym8QuantizeWeightDenseCuda(Stream stream, bool has_signs)
      : Primitive(stream), has_signs_(has_signs) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Sym8QuantizeWeightDenseCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    std::optional<array> signs;
    if (has_signs_) {
      signs = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    }

    const int in_dtype = float_dtype_code(w.dtype());
    const int64_t n64 = w.shape(0);
    const int64_t k64 = w.shape(1);
    if (in_dtype < 0 || n64 < 1 || n64 > std::numeric_limits<int32_t>::max() ||
        k64 < 1 || k64 > std::numeric_limits<int32_t>::max() ||
        (has_signs_ && (k64 % 1024) != 0)) {
      throw std::invalid_argument("Sym8QuantizeWeightDenseCuda: unsupported shape or dtype");
    }
    const int N = static_cast<int>(n64);
    const int K = static_cast<int>(k64);

    auto& out_q = outputs[0];
    auto& out_s = outputs[1];
    out_q.set_data(cu::malloc_async(out_q.nbytes(), encoder));
    out_s.set_data(cu::malloc_async(out_s.nbytes(), encoder));

    encoder.set_input_array(w);
    if (has_signs_) encoder.set_input_array(*signs);
    encoder.set_output_array(out_q);
    encoder.set_output_array(out_s);

    static constexpr const char* kEntries[3][2] = {
        {"sym8_quantize_weight_dense<0, false>", "sym8_quantize_weight_dense<0, true>"},
        {"sym8_quantize_weight_dense<1, false>", "sym8_quantize_weight_dense<1, true>"},
        {"sym8_quantize_weight_dense<2, false>", "sym8_quantize_weight_dense<2, true>"},
    };
    auto fn = cu::jit_kernel(
        encoder.device(), sym8_weight_quant_src(),
        kEntries[in_dtype][has_signs_ ? 1 : 0]);

    cu::KernelArgs args;
    args.append(w);
    if (has_signs_) {
      args.append(*signs);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out_q);
    args.append(out_s);
    args.append<int32_t>(N);
    args.append<int32_t>(K);

    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(N), 1, 1), dim3(128, 1, 1), {}, 0,
        args.args());
  }

  DEFINE_NAME(Sym8QuantizeWeightDenseCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Sym8QuantizeWeightDenseCuda*>(&other);
    return rhs && rhs->has_signs_ == has_signs_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape(), {inputs[0].shape(0)}};
  }

 private:
  bool has_signs_;
};

// ============================================================================
// 4b. Bit-Exact FP32 Weight Quantizer: Packed K-Quant (Q4_K / Q6_K) -> Sym8
// ============================================================================
class Sym8QuantizeWeightKQuantCuda final : public Primitive {
 public:
  Sym8QuantizeWeightKQuantCuda(
      Stream stream, bool is_q6k, bool tiled, bool has_signs, int N, int K)
      : Primitive(stream),
        is_q6k_(is_q6k),
        tiled_(tiled),
        has_signs_(has_signs),
        N_(N),
        K_(K) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Sym8QuantizeWeightKQuantCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array scales =
        cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array biases =
        cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
    std::optional<array> signs;
    if (has_signs_) {
      signs = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    }

    const int64_t num_sb = static_cast<int64_t>(K_) / 256;
    const int64_t expected_w_cols = num_sb * (is_q6k_ ? 48 : 32);
    const int64_t expected_s_cols = num_sb * 16;
    const int64_t expected_b_cols = is_q6k_ ? num_sb : (num_sb * 2);
    const Dtype expected_s_dt = is_q6k_ ? int8 : uint8;
    if (N_ < 1 || K_ < 256 || (K_ % 256) != 0 || (tiled_ && (N_ % 64) != 0) ||
        (has_signs_ && (K_ % 1024) != 0) ||
        w.dtype() != uint32 || w.ndim() != 2 || w.shape(0) != N_ ||
        w.shape(1) != expected_w_cols ||
        scales.dtype() != expected_s_dt || scales.ndim() != 2 ||
        scales.shape(0) != N_ || scales.shape(1) != expected_s_cols ||
        biases.dtype() != float16 || biases.ndim() != 2 ||
        biases.shape(0) != N_ || biases.shape(1) != expected_b_cols ||
        (has_signs_ &&
         (signs->dtype() != float32 || signs->ndim() != 1 ||
          signs->shape(0) != K_))) {
      throw std::invalid_argument(
          "Sym8QuantizeWeightKQuantCuda: unsupported shape or dtype");
    }

    auto& out_q = outputs[0];
    auto& out_s = outputs[1];
    out_q.set_data(cu::malloc_async(out_q.nbytes(), encoder));
    out_s.set_data(cu::malloc_async(out_s.nbytes(), encoder));

    encoder.set_input_array(w);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    if (has_signs_) encoder.set_input_array(*signs);
    encoder.set_output_array(out_q);
    encoder.set_output_array(out_s);

    static constexpr const char* kEntries[2][2][2] = {
        {{"mlx_sym8_wq_q4k_rm_nosigns", "mlx_sym8_wq_q4k_rm_signs"},
         {"mlx_sym8_wq_q4k_t64_nosigns", "mlx_sym8_wq_q4k_t64_signs"}},
        {{"mlx_sym8_wq_q6k_rm_nosigns", "mlx_sym8_wq_q6k_rm_signs"},
         {"mlx_sym8_wq_q6k_t64_nosigns", "mlx_sym8_wq_q6k_t64_signs"}},
    };
    auto fn = cu::jit_kernel(
        encoder.device(), sym8_weight_quant_kquant_src(),
        kEntries[is_q6k_ ? 1 : 0][tiled_ ? 1 : 0][has_signs_ ? 1 : 0]);

    cu::KernelArgs args;
    args.append(w);
    args.append(scales);
    args.append(biases);
    if (has_signs_) {
      args.append(*signs);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out_q);
    args.append(out_s);
    args.append<int32_t>(N_);
    args.append<int32_t>(K_);

    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(N_), 1, 1), dim3(128, 1, 1), {}, 0,
        args.args());
  }

  DEFINE_NAME(Sym8QuantizeWeightKQuantCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Sym8QuantizeWeightKQuantCuda*>(&other);
    return rhs && rhs->is_q6k_ == is_q6k_ && rhs->tiled_ == tiled_ &&
        rhs->has_signs_ == has_signs_ && rhs->N_ == N_ && rhs->K_ == K_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {{static_cast<ShapeElem>(N_), static_cast<ShapeElem>(K_)},
            {static_cast<ShapeElem>(N_)}};
  }

 private:
  bool is_q6k_;
  bool tiled_;
  bool has_signs_;
  int N_;
  int K_;
};

// ============================================================================
// 5. Fused (Optional Residual Add +) RMSNorm + Optional Hadamard-1024 + INT8 Quant
// ============================================================================
class Sym8AddRmsNormHadQuantCuda final : public Primitive {
 public:
  Sym8AddRmsNormHadQuantCuda(
      Stream stream,
      bool has_delta,
      bool write_normed,
      bool w_is_f32,
      bool has_signs,
      float eps)
      : Primitive(stream),
        has_delta_(has_delta),
        write_normed_(write_normed),
        w_is_f32_(w_is_f32),
        has_signs_(has_signs),
        eps_(eps) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Sym8AddRmsNormHadQuantCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array h = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w_norm = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    std::optional<array> delta;
    std::optional<array> signs;
    size_t idx = 2;
    if (has_delta_) {
      delta = cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    }
    if (has_signs_) {
      signs = cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    }

    if (!std::isfinite(eps_) || eps_ < 0.0f) {
      throw std::invalid_argument(
          "Sym8AddRmsNormHadQuantCuda: eps must be finite and non-negative");
    }
    if (h.dtype() != bfloat16 || h.ndim() < 2) {
      throw std::invalid_argument(
          "Sym8AddRmsNormHadQuantCuda: h must be bfloat16 with rank >= 2");
    }
    const int64_t h64 = h.shape(-1);
    if (h64 < 16 || h64 > 8192 || (h64 % 16) != 0 ||
        (has_signs_ && (h64 % 1024) != 0)) {
      throw std::invalid_argument("Sym8AddRmsNormHadQuantCuda: unsupported H");
    }
    const int64_t m64 = static_cast<int64_t>(h.size()) / h64;
    if (m64 < 1 || m64 > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("Sym8AddRmsNormHadQuantCuda: unsupported M");
    }
    const Dtype expected_w_dt = w_is_f32_ ? float32 : bfloat16;
    if (w_norm.dtype() != expected_w_dt || w_norm.ndim() != 1 ||
        w_norm.shape(0) != h64) {
      throw std::invalid_argument("Sym8AddRmsNormHadQuantCuda: invalid w_norm");
    }
    if (has_delta_ && (delta->dtype() != bfloat16 || delta->shape() != h.shape())) {
      throw std::invalid_argument("Sym8AddRmsNormHadQuantCuda: invalid delta");
    }
    if (has_signs_ &&
        (signs->dtype() != float32 || signs->ndim() != 1 ||
         signs->shape(0) != h64)) {
      throw std::invalid_argument("Sym8AddRmsNormHadQuantCuda: invalid signs");
    }

    const int M = static_cast<int>(m64);
    const int H = static_cast<int>(h64);

    auto& out_xq = outputs[0];
    auto& out_sx = outputs[1];
    out_xq.set_data(cu::malloc_async(out_xq.nbytes(), encoder));
    out_sx.set_data(cu::malloc_async(out_sx.nbytes(), encoder));
    size_t out_idx = 2;
    array* out_h = nullptr;
    array* out_normed = nullptr;
    if (has_delta_) {
      out_h = &outputs[out_idx++];
      out_h->set_data(cu::malloc_async(out_h->nbytes(), encoder));
    }
    if (write_normed_) {
      out_normed = &outputs[out_idx++];
      out_normed->set_data(cu::malloc_async(out_normed->nbytes(), encoder));
    }

    encoder.set_input_array(h);
    encoder.set_input_array(w_norm);
    if (has_delta_) encoder.set_input_array(*delta);
    if (has_signs_) encoder.set_input_array(*signs);
    encoder.set_output_array(out_xq);
    encoder.set_output_array(out_sx);
    if (out_h) encoder.set_output_array(*out_h);
    if (out_normed) encoder.set_output_array(*out_normed);

    static constexpr const char* kEntries[2][2][2][2] = {
        {
            {
                {"sym8_add_rmsnorm_hadamard_quant<false, false, false, false>",
                 "sym8_add_rmsnorm_hadamard_quant<false, false, false, true>"},
                {"sym8_add_rmsnorm_hadamard_quant<false, false, true, false>",
                 "sym8_add_rmsnorm_hadamard_quant<false, false, true, true>"},
            },
            {
                {"sym8_add_rmsnorm_hadamard_quant<false, true, false, false>",
                 "sym8_add_rmsnorm_hadamard_quant<false, true, false, true>"},
                {"sym8_add_rmsnorm_hadamard_quant<false, true, true, false>",
                 "sym8_add_rmsnorm_hadamard_quant<false, true, true, true>"},
            },
        },
        {
            {
                {"sym8_add_rmsnorm_hadamard_quant<true, false, false, false>",
                 "sym8_add_rmsnorm_hadamard_quant<true, false, false, true>"},
                {"sym8_add_rmsnorm_hadamard_quant<true, false, true, false>",
                 "sym8_add_rmsnorm_hadamard_quant<true, false, true, true>"},
            },
            {
                {"sym8_add_rmsnorm_hadamard_quant<true, true, false, false>",
                 "sym8_add_rmsnorm_hadamard_quant<true, true, false, true>"},
                {"sym8_add_rmsnorm_hadamard_quant<true, true, true, false>",
                 "sym8_add_rmsnorm_hadamard_quant<true, true, true, true>"},
            },
        },
    };
    const char* entry = kEntries[has_delta_ ? 1 : 0][write_normed_ ? 1 : 0]
                                [w_is_f32_ ? 1 : 0][has_signs_ ? 1 : 0];
    auto fn = cu::jit_kernel(
        encoder.device(), sym8_rmsnorm_hadamard_src(), entry);

    cu::KernelArgs args;
    args.append(h);
    if (has_delta_) {
      args.append(*delta);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(w_norm);
    if (has_signs_) {
      args.append(*signs);
    } else {
      args.append<CUdeviceptr>(0);
    }
    if (out_h) {
      args.append(*out_h);
    } else {
      args.append<CUdeviceptr>(0);
    }
    if (out_normed) {
      args.append(*out_normed);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out_xq);
    args.append(out_sx);
    args.append<float>(eps_);
    args.append<int32_t>(M);
    args.append<int32_t>(H);

    const uint32_t smem_bytes =
        static_cast<uint32_t>(H) * static_cast<uint32_t>(sizeof(float));
    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(M), 1, 1), dim3(128, 1, 1), {},
        smem_bytes, args.args());
  }

  DEFINE_NAME(Sym8AddRmsNormHadQuantCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Sym8AddRmsNormHadQuantCuda*>(&other);
    return rhs && rhs->has_delta_ == has_delta_ &&
        rhs->write_normed_ == write_normed_ &&
        rhs->w_is_f32_ == w_is_f32_ &&
        rhs->has_signs_ == has_signs_ &&
        rhs->eps_ == eps_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape q_shape = inputs[0].shape();
    Shape s_shape = inputs[0].shape();
    s_shape.pop_back();
    std::vector<Shape> shapes;
    shapes.reserve(2 + (has_delta_ ? 1 : 0) + (write_normed_ ? 1 : 0));
    shapes.push_back(std::move(q_shape));
    shapes.push_back(std::move(s_shape));
    if (has_delta_) {
      shapes.push_back(inputs[0].shape());
    }
    if (write_normed_) {
      shapes.push_back(inputs[0].shape());
    }
    return shapes;
  }

 private:
  bool has_delta_;
  bool write_normed_;
  bool w_is_f32_;
  bool has_signs_;
  float eps_;
};

bool run_hadamard_quant_ffi(
    mlx_array* x_h, mlx_array* signs_h, bool fused_swiglu,
    mlx_array** out_xq, mlx_array** out_sx, const char* fn_name) {
  if (out_xq) *out_xq = nullptr;
  if (out_sx) *out_sx = nullptr;
  try {
    if (!x_h || !out_xq || !out_sx ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    if (float_dtype_code(x.dtype()) < 0 || x.ndim() < 2) {
      return false;
    }
    const int64_t in_k = x.shape(-1);
    if (in_k < 1 || (fused_swiglu && (in_k & 1) != 0)) {
      return false;
    }
    const int64_t k = fused_swiglu ? (in_k / 2) : in_k;
    const int64_t m = static_cast<int64_t>(x.size()) / in_k;
    if (m < 1 || m > std::numeric_limits<int32_t>::max() ||
        k < 1 || k > std::numeric_limits<int32_t>::max()) {
      return false;
    }

    const bool has_signs = (signs_h != nullptr);
    std::vector<array> inputs;
    inputs.reserve(has_signs ? 2 : 1);
    inputs.push_back(x);
    if (has_signs) {
      const auto& s = *reinterpret_cast<array*>(signs_h);
      if (s.dtype() != float32 || s.ndim() != 1 || s.shape(0) != k || (k % 1024) != 0) {
        return false;
      }
      inputs.push_back(s);
    }

    Shape q_shape = x.shape();
    q_shape.back() = static_cast<ShapeElem>(k);
    Shape s_shape = x.shape();
    s_shape.pop_back();

    auto results = array::make_arrays(
        {std::move(q_shape), std::move(s_shape)},
        {int8, float32},
        std::make_shared<Sym8HadamardQuantCuda>(
            default_stream(Device::gpu), has_signs, fused_swiglu),
        std::move(inputs));
    *out_xq = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_sx = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: %s\n", fn_name, e.what());
    if (out_xq && *out_xq) {
      delete reinterpret_cast<array*>(*out_xq);
      *out_xq = nullptr;
    }
    if (out_sx && *out_sx) {
      delete reinterpret_cast<array*>(*out_sx);
      *out_sx = nullptr;
    }
    return false;
  }
}

}  // namespace
}  // namespace mlx::core::sym8_ops

extern "C" bool mlx_cuda_sym8_hadamard_quant(
    mlx_array* x_h, mlx_array* signs_h,
    mlx_array** out_xq, mlx_array** out_sx) {
  return mlx::core::sym8_ops::run_hadamard_quant_ffi(
      x_h, signs_h, false, out_xq, out_sx, "mlx_cuda_sym8_hadamard_quant");
}

extern "C" bool mlx_cuda_sym8_swiglu_hadamard_quant(
    mlx_array* gate_up_h, mlx_array* signs_h,
    mlx_array** out_xq, mlx_array** out_sx) {
  return mlx::core::sym8_ops::run_hadamard_quant_ffi(
      gate_up_h, signs_h, true, out_xq, out_sx,
      "mlx_cuda_sym8_swiglu_hadamard_quant");
}

extern "C" mlx_array* mlx_cuda_sym8_imma_gemm(
    mlx_array* x_i8_h, mlx_array* w_i8_h,
    mlx_array* s_x_h, mlx_array* s_w_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::sym8_ops;
    if (!x_i8_h || !w_i8_h || !s_x_h || !s_w_h ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& x_i8 = *reinterpret_cast<array*>(x_i8_h);
    const auto& w_i8 = *reinterpret_cast<array*>(w_i8_h);
    const auto& s_x = *reinterpret_cast<array*>(s_x_h);
    const auto& s_w = *reinterpret_cast<array*>(s_w_h);
    if (x_i8.dtype() != int8 || x_i8.ndim() < 2 ||
        w_i8.dtype() != int8 || w_i8.ndim() != 2 ||
        s_x.dtype() != float32 ||
        s_w.dtype() != float32 || s_w.ndim() != 1) {
      return nullptr;
    }
    const int64_t N = w_i8.shape(0);
    const int64_t K = w_i8.shape(1);
    if (N < 1 || N > std::numeric_limits<int32_t>::max() ||
        K < 1 || K > std::numeric_limits<int32_t>::max() ||
        x_i8.shape(-1) != K || s_w.shape(0) != N) {
      return nullptr;
    }
    const int64_t M = static_cast<int64_t>(x_i8.size()) / K;
    if (M < 1 || M > std::numeric_limits<int32_t>::max() ||
        s_x.size() != static_cast<size_t>(M)) {
      return nullptr;
    }
    Shape out_shape = x_i8.shape();
    out_shape.back() = static_cast<ShapeElem>(N);
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Sym8ImmaGemmCuda>(default_stream(Device::gpu)),
        {x_i8, w_i8, s_x, s_w});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cuda_sym8_imma_gemm: %s\n", e.what());
    return nullptr;
  }
}

extern "C" bool mlx_cuda_sym8_quantize_weight_q8(
    mlx_array* w_u32_h, mlx_array* w_scales_h,
    mlx_array* w_biases_h, mlx_array* signs_h,
    mlx_array** out_q, mlx_array** out_s) {
  if (out_q) *out_q = nullptr;
  if (out_s) *out_s = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::sym8_ops;
    if (!w_u32_h || !w_scales_h || !out_q || !out_s ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& w_u32 = *reinterpret_cast<array*>(w_u32_h);
    const auto& w_scales = *reinterpret_cast<array*>(w_scales_h);
    if (w_u32.dtype() != uint32 || w_u32.ndim() != 2 ||
        scale_dtype_code(w_scales.dtype()) < 0 || w_scales.ndim() != 2) {
      return false;
    }
    const int64_t N = w_u32.shape(0);
    const int64_t K = static_cast<int64_t>(w_u32.shape(1)) * 4;
    if (N < 1 || N > std::numeric_limits<int32_t>::max() ||
        K < 32 || K > std::numeric_limits<int32_t>::max() ||
        (K % 32) != 0 ||
        w_scales.shape(0) != N || w_scales.shape(1) != (K / 32)) {
      return false;
    }

    const bool has_biases = (w_biases_h != nullptr);
    const bool has_signs = (signs_h != nullptr);
    std::vector<array> inputs;
    inputs.reserve(2 + (has_biases ? 1 : 0) + (has_signs ? 1 : 0));
    inputs.push_back(w_u32);
    inputs.push_back(w_scales);
    if (has_biases) {
      const auto& bi = *reinterpret_cast<array*>(w_biases_h);
      if (scale_dtype_code(bi.dtype()) < 0 || bi.ndim() != 2 ||
          bi.shape(0) != N || bi.shape(1) != (K / 32)) {
        return false;
      }
      if (bi.dtype() != w_scales.dtype()) {
        inputs.push_back(astype(bi, w_scales.dtype(), default_stream(Device::gpu)));
      } else {
        inputs.push_back(bi);
      }
    }
    if (has_signs) {
      const auto& s = *reinterpret_cast<array*>(signs_h);
      if (s.dtype() != float32 || s.ndim() != 1 || s.shape(0) != K || (K % 1024) != 0) {
        return false;
      }
      inputs.push_back(s);
    }

    auto results = array::make_arrays(
        {{static_cast<ShapeElem>(N), static_cast<ShapeElem>(K)},
         {static_cast<ShapeElem>(N)}},
        {int8, float32},
        std::make_shared<Sym8QuantizeWeightQ8Cuda>(
            default_stream(Device::gpu), has_biases, has_signs),
        std::move(inputs));
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_s = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cuda_sym8_quantize_weight_q8: %s\n", e.what());
    if (out_q && *out_q) {
      delete reinterpret_cast<mlx::core::array*>(*out_q);
      *out_q = nullptr;
    }
    if (out_s && *out_s) {
      delete reinterpret_cast<mlx::core::array*>(*out_s);
      *out_s = nullptr;
    }
    return false;
  }
}

extern "C" bool mlx_sym8_quantize_weight_kquant_cuda(
    mlx_array* w_h,
    mlx_array* scales_h,
    mlx_array* biases_h,
    bool is_q6k,
    bool tiled,
    mlx_array* signs_h,
    mlx_array** out_q,
    mlx_array** out_s) {
  if (out_q) *out_q = nullptr;
  if (out_s) *out_s = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::sym8_ops;
    if (!w_h || !scales_h || !biases_h || !out_q || !out_s ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& w = *reinterpret_cast<array*>(w_h);
    const auto& scales = *reinterpret_cast<array*>(scales_h);
    const auto& biases = *reinterpret_cast<array*>(biases_h);
    const Dtype expected_s_dt = is_q6k ? int8 : uint8;
    if (w.dtype() != uint32 || w.ndim() != 2 ||
        scales.dtype() != expected_s_dt || scales.ndim() != 2 ||
        biases.dtype() != float16 || biases.ndim() != 2) {
      return false;
    }
    const int64_t N = w.shape(0);
    const int64_t w_cols = w.shape(1);
    const int64_t words_per_sb = is_q6k ? 48 : 32;
    if (N < 1 || N > std::numeric_limits<int32_t>::max() ||
        (tiled && (N % 64) != 0) ||
        w_cols < words_per_sb || (w_cols % words_per_sb) != 0) {
      return false;
    }
    const int64_t num_sb = w_cols / words_per_sb;
    if (num_sb > std::numeric_limits<int32_t>::max() / 256) {
      return false;
    }
    const int64_t K = num_sb * 256;
    const int64_t expected_s_cols = num_sb * 16;
    const int64_t expected_b_cols = is_q6k ? num_sb : (num_sb * 2);
    if (scales.shape(0) != N || scales.shape(1) != expected_s_cols ||
        biases.shape(0) != N || biases.shape(1) != expected_b_cols) {
      return false;
    }

    const bool has_signs = (signs_h != nullptr);
    std::vector<array> inputs;
    inputs.reserve(has_signs ? 4 : 3);
    inputs.push_back(w);
    inputs.push_back(scales);
    inputs.push_back(biases);
    if (has_signs) {
      const auto& s = *reinterpret_cast<array*>(signs_h);
      if (s.dtype() != float32 || s.ndim() != 1 || s.shape(0) != K ||
          (K % 1024) != 0) {
        return false;
      }
      inputs.push_back(s);
    }

    auto results = array::make_arrays(
        {{static_cast<ShapeElem>(N), static_cast<ShapeElem>(K)},
         {static_cast<ShapeElem>(N)}},
        {int8, float32},
        std::make_shared<Sym8QuantizeWeightKQuantCuda>(
            default_stream(Device::gpu), is_q6k, tiled, has_signs,
            static_cast<int>(N), static_cast<int>(K)),
        std::move(inputs));
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_s = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_sym8_quantize_weight_kquant_cuda: %s\n", e.what());
    if (out_q && *out_q) {
      delete reinterpret_cast<mlx::core::array*>(*out_q);
      *out_q = nullptr;
    }
    if (out_s && *out_s) {
      delete reinterpret_cast<mlx::core::array*>(*out_s);
      *out_s = nullptr;
    }
    return false;
  }
}

extern "C" bool mlx_cuda_sym8_quantize_weight_dense(
    mlx_array* w_h, mlx_array* signs_h,
    mlx_array** out_q, mlx_array** out_s) {
  if (out_q) *out_q = nullptr;
  if (out_s) *out_s = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::sym8_ops;
    if (!w_h || !out_q || !out_s ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& w = *reinterpret_cast<array*>(w_h);
    if (float_dtype_code(w.dtype()) < 0 || w.ndim() != 2) {
      return false;
    }
    const int64_t N = w.shape(0);
    const int64_t K = w.shape(1);
    if (N < 1 || N > std::numeric_limits<int32_t>::max() ||
        K < 1 || K > std::numeric_limits<int32_t>::max()) {
      return false;
    }

    const bool has_signs = (signs_h != nullptr);
    std::vector<array> inputs;
    inputs.reserve(has_signs ? 2 : 1);
    inputs.push_back(w);
    if (has_signs) {
      const auto& s = *reinterpret_cast<array*>(signs_h);
      if (s.dtype() != float32 || s.ndim() != 1 || s.shape(0) != K || (K % 1024) != 0) {
        return false;
      }
      inputs.push_back(s);
    }

    auto results = array::make_arrays(
        {{static_cast<ShapeElem>(N), static_cast<ShapeElem>(K)},
         {static_cast<ShapeElem>(N)}},
        {int8, float32},
        std::make_shared<Sym8QuantizeWeightDenseCuda>(
            default_stream(Device::gpu), has_signs),
        std::move(inputs));
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_s = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cuda_sym8_quantize_weight_dense: %s\n", e.what());
    if (out_q && *out_q) {
      delete reinterpret_cast<mlx::core::array*>(*out_q);
      *out_q = nullptr;
    }
    if (out_s && *out_s) {
      delete reinterpret_cast<mlx::core::array*>(*out_s);
      *out_s = nullptr;
    }
    return false;
  }
}

extern "C" bool mlx_cuda_sym8_add_rmsnorm_hadamard_quant(
    mlx_array* h_h,
    mlx_array* delta_h,
    mlx_array* w_norm_h,
    mlx_array* signs_h,
    float eps,
    bool write_normed,
    mlx_array** out_h,
    mlx_array** out_normed,
    mlx_array** out_xq,
    mlx_array** out_sx) {
  if (out_h) *out_h = nullptr;
  if (out_normed) *out_normed = nullptr;
  if (out_xq) *out_xq = nullptr;
  if (out_sx) *out_sx = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::sym8_ops;
    const bool has_delta = (delta_h != nullptr);
    const bool has_signs = (signs_h != nullptr);
    if (!h_h || !w_norm_h || !out_xq || !out_sx ||
        (has_delta && !out_h) || (write_normed && !out_normed) ||
        !std::isfinite(eps) || eps < 0.0f ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& h = *reinterpret_cast<array*>(h_h);
    const auto& w_norm = *reinterpret_cast<array*>(w_norm_h);
    if (h.dtype() != bfloat16 || h.ndim() < 2) {
      return false;
    }
    const int64_t h_dim = h.shape(-1);
    if (h_dim < 16 || h_dim > 8192 || (h_dim % 16) != 0 ||
        (has_signs && (h_dim % 1024) != 0)) {
      return false;
    }
    const int64_t m_dim = static_cast<int64_t>(h.size()) / h_dim;
    if (m_dim < 1 || m_dim > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    const bool w_is_f32 = (w_norm.dtype() == float32);
    if ((!w_is_f32 && w_norm.dtype() != bfloat16) ||
        w_norm.ndim() != 1 || w_norm.shape(0) != h_dim) {
      return false;
    }

    std::vector<array> inputs;
    inputs.reserve(2 + (has_delta ? 1 : 0) + (has_signs ? 1 : 0));
    inputs.push_back(h);
    inputs.push_back(w_norm);
    if (has_delta) {
      const auto& delta = *reinterpret_cast<array*>(delta_h);
      if (delta.dtype() != bfloat16 || delta.shape() != h.shape()) {
        return false;
      }
      inputs.push_back(delta);
    }
    if (has_signs) {
      const auto& signs = *reinterpret_cast<array*>(signs_h);
      if (signs.dtype() != float32 || signs.ndim() != 1 ||
          signs.shape(0) != h_dim) {
        return false;
      }
      inputs.push_back(signs);
    }

    Shape q_shape = h.shape();
    Shape s_shape = h.shape();
    s_shape.pop_back();
    std::vector<Shape> out_shapes;
    std::vector<Dtype> out_dtypes;
    const size_t n_outs = 2 + (has_delta ? 1 : 0) + (write_normed ? 1 : 0);
    out_shapes.reserve(n_outs);
    out_dtypes.reserve(n_outs);
    out_shapes.push_back(std::move(q_shape));
    out_dtypes.push_back(int8);
    out_shapes.push_back(std::move(s_shape));
    out_dtypes.push_back(float32);
    if (has_delta) {
      out_shapes.push_back(h.shape());
      out_dtypes.push_back(bfloat16);
    }
    if (write_normed) {
      out_shapes.push_back(h.shape());
      out_dtypes.push_back(bfloat16);
    }

    auto results = array::make_arrays(
        std::move(out_shapes),
        std::move(out_dtypes),
        std::make_shared<Sym8AddRmsNormHadQuantCuda>(
            default_stream(Device::gpu),
            has_delta,
            write_normed,
            w_is_f32,
            has_signs,
            eps),
        std::move(inputs));

    *out_xq = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_sx = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    size_t idx = 2;
    if (has_delta) {
      *out_h = reinterpret_cast<mlx_array*>(new array(std::move(results[idx++])));
    }
    if (write_normed) {
      *out_normed =
          reinterpret_cast<mlx_array*>(new array(std::move(results[idx++])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(
        stderr, "mlx_cuda_sym8_add_rmsnorm_hadamard_quant: %s\n", e.what());
    if (out_h && *out_h) {
      delete reinterpret_cast<mlx::core::array*>(*out_h);
      *out_h = nullptr;
    }
    if (out_normed && *out_normed) {
      delete reinterpret_cast<mlx::core::array*>(*out_normed);
      *out_normed = nullptr;
    }
    if (out_xq && *out_xq) {
      delete reinterpret_cast<mlx::core::array*>(*out_xq);
      *out_xq = nullptr;
    }
    if (out_sx && *out_sx) {
      delete reinterpret_cast<mlx::core::array*>(*out_sx);
      *out_sx = nullptr;
    }
    return false;
  }
}

#else

struct mlx_array;

extern "C" bool mlx_cuda_sym8_hadamard_quant(
    mlx_array* x_h, mlx_array* signs_h,
    mlx_array** out_xq, mlx_array** out_sx) {
  (void)x_h;
  (void)signs_h;
  if (out_xq) *out_xq = nullptr;
  if (out_sx) *out_sx = nullptr;
  return false;
}

extern "C" bool mlx_cuda_sym8_swiglu_hadamard_quant(
    mlx_array* gate_up_h, mlx_array* signs_h,
    mlx_array** out_xq, mlx_array** out_sx) {
  (void)gate_up_h;
  (void)signs_h;
  if (out_xq) *out_xq = nullptr;
  if (out_sx) *out_sx = nullptr;
  return false;
}

extern "C" mlx_array* mlx_cuda_sym8_imma_gemm(
    mlx_array* x_i8_h, mlx_array* w_i8_h,
    mlx_array* s_x_h, mlx_array* s_w_h) {
  (void)x_i8_h;
  (void)w_i8_h;
  (void)s_x_h;
  (void)s_w_h;
  return nullptr;
}

extern "C" bool mlx_cuda_sym8_quantize_weight_q8(
    mlx_array* w_u32_h, mlx_array* w_scales_h,
    mlx_array* w_biases_h, mlx_array* signs_h,
    mlx_array** out_q, mlx_array** out_s) {
  (void)w_u32_h;
  (void)w_scales_h;
  (void)w_biases_h;
  (void)signs_h;
  if (out_q) *out_q = nullptr;
  if (out_s) *out_s = nullptr;
  return false;
}

extern "C" bool mlx_sym8_quantize_weight_kquant_cuda(
    mlx_array* w_h,
    mlx_array* scales_h,
    mlx_array* biases_h,
    bool is_q6k,
    bool tiled,
    mlx_array* signs_h,
    mlx_array** out_q,
    mlx_array** out_s) {
  (void)w_h;
  (void)scales_h;
  (void)biases_h;
  (void)is_q6k;
  (void)tiled;
  (void)signs_h;
  if (out_q) *out_q = nullptr;
  if (out_s) *out_s = nullptr;
  return false;
}

extern "C" bool mlx_cuda_sym8_quantize_weight_dense(
    mlx_array* w_h, mlx_array* signs_h,
    mlx_array** out_q, mlx_array** out_s) {
  (void)w_h;
  (void)signs_h;
  if (out_q) *out_q = nullptr;
  if (out_s) *out_s = nullptr;
  return false;
}

extern "C" bool mlx_cuda_sym8_add_rmsnorm_hadamard_quant(
    mlx_array* h_h,
    mlx_array* delta_h,
    mlx_array* w_norm_h,
    mlx_array* signs_h,
    float eps,
    bool write_normed,
    mlx_array** out_h,
    mlx_array** out_normed,
    mlx_array** out_xq,
    mlx_array** out_sx) {
  (void)h_h;
  (void)delta_h;
  (void)w_norm_h;
  (void)signs_h;
  (void)eps;
  (void)write_normed;
  if (out_h) *out_h = nullptr;
  if (out_normed) *out_normed = nullptr;
  if (out_xq) *out_xq = nullptr;
  if (out_sx) *out_sx = nullptr;
  return false;
}

#endif
