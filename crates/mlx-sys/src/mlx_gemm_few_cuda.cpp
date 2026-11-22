#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_cuda_ops.h"

#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace mlx::core::gemm_few {
namespace {

constexpr char affine_q8_few_source[] =
#include "cuda/affine_q8_few.cu.inc"
    ;

constexpr char bf16_gemm_few_source[] =
#include "cuda/bf16_gemm_few.cu.inc"
    ;

int sidecar_dtype_code(Dtype dt) noexcept {
  if (dt == float32) {
    return 0;
  }
  if (dt == bfloat16) {
    return 1;
  }
  if (dt == float16) {
    return 2;
  }
  return -1;
}

class AffineQ8Few : public UnaryPrimitive {
public:
  explicit AffineQ8Few(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("AffineQ8Few requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    const int sb_dtype = sidecar_dtype_code(inputs[2].dtype());
    if (sb_dtype < 0 || inputs[3].dtype() != inputs[2].dtype()) {
      throw std::invalid_argument("AffineQ8Few: unsupported sidecar dtype");
    }
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array scales = (sb_dtype == 0)
        ? cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
    array biases = (sb_dtype == 0)
        ? cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<2>(inputs[3], stream(), encoder);

    const int K = x.shape(-1);
    const int M = (K > 0) ? static_cast<int>(x.size() / K) : 0;
    const int N = w.shape(0);
    if (M < 1 || M > 16 || N <= 0 || K < 32 || K % 32 != 0) {
      throw std::invalid_argument("AffineQ8Few: unsupported replay shape");
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    if (M <= 8) {
      // Single-launch coalesced 512B/warp uint4 streaming FP32 FMA kernel for
      // M in 1..=8 (beats SMEM-staged BF16 MMA + Split-K on both bandwidth and
      // FP32 Q8 precision across N=96, 1024, 5120).
      static constexpr const char* kU4Entries[3][4][2] = {
          {
              {"mlx_node_affine_q8_u4_1row<0,1,4>", "mlx_node_affine_q8_u4_1row<0,1,8>"},
              {"mlx_node_affine_q8_u4_1row<0,2,4>", "mlx_node_affine_q8_u4_1row<0,2,8>"},
              {"mlx_node_affine_q8_u4_1row<0,4,4>", "mlx_node_affine_q8_u4_1row<0,4,8>"},
              {"mlx_node_affine_q8_u4_1row<0,8,4>", "mlx_node_affine_q8_u4_1row<0,8,8>"},
          },
          {
              {"mlx_node_affine_q8_u4_1row<1,1,4>", "mlx_node_affine_q8_u4_1row<1,1,8>"},
              {"mlx_node_affine_q8_u4_1row<1,2,4>", "mlx_node_affine_q8_u4_1row<1,2,8>"},
              {"mlx_node_affine_q8_u4_1row<1,4,4>", "mlx_node_affine_q8_u4_1row<1,4,8>"},
              {"mlx_node_affine_q8_u4_1row<1,8,4>", "mlx_node_affine_q8_u4_1row<1,8,8>"},
          },
          {
              {"mlx_node_affine_q8_u4_1row<2,1,4>", "mlx_node_affine_q8_u4_1row<2,1,8>"},
              {"mlx_node_affine_q8_u4_1row<2,2,4>", "mlx_node_affine_q8_u4_1row<2,2,8>"},
              {"mlx_node_affine_q8_u4_1row<2,4,4>", "mlx_node_affine_q8_u4_1row<2,4,8>"},
              {"mlx_node_affine_q8_u4_1row<2,8,4>", "mlx_node_affine_q8_u4_1row<2,8,8>"},
          },
      };
      const int m_bucket = (M == 1) ? 0 : (M == 2 ? 1 : (M <= 4 ? 2 : 3));
      const bool use_8w = (M <= 4 && N <= 128);
      const int num_warps = use_8w ? 8 : 4;
      auto kernel = cu::jit_kernel(
          encoder.device(), affine_q8_few_source,
          kU4Entries[sb_dtype][m_bucket][use_8w ? 1 : 0]);
      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      encoder.set_input_array(biases);
      encoder.set_output_array(out);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      args.append(biases);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      encoder.add_kernel_node_raw(
          kernel, dim3(N), dim3(32, num_warps), {}, 0, args.args());
      return;
    }

    // Opt-in exact-FP32 u4_1row path for M in 9..=16 (matches M<=8 summation
    // order bit-for-bit, at ~+1.66 ms/step cost over Tensor Core MMA on RTX Spark).
    static const bool exact_m16 = [] {
      const char* v = std::getenv("MLX_CUDA_AQ8_EXACT_M16");
      return v && v[0] == '1';
    }();
    if (exact_m16) {
      static constexpr const char* kU4M16Entries[3] = {
          "mlx_node_affine_q8_u4_1row<0,16,4>",
          "mlx_node_affine_q8_u4_1row<1,16,4>",
          "mlx_node_affine_q8_u4_1row<2,16,4>",
      };
      auto kernel = cu::jit_kernel(
          encoder.device(), affine_q8_few_source, kU4M16Entries[sb_dtype]);
      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      encoder.set_input_array(biases);
      encoder.set_output_array(out);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      args.append(biases);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      encoder.add_kernel_node_raw(
          kernel, dim3(N), dim3(32, 4), {}, 0, args.args());
      return;
    }

    // M in 9..=16: single-launch coalesced SMEM-staged BF16 Tensor Core MMA
    // (avoids Split-K partial buffer allocation + reduce launch overhead).
    static constexpr const char* kMmaEntries[3] = {
        "mlx_node_affine_q8_mma<0,2,false>",
        "mlx_node_affine_q8_mma<1,2,false>",
        "mlx_node_affine_q8_mma<2,2,false>",
    };
    const int row_blocks = (N + 15) / 16;
    auto kernel = cu::jit_kernel(
        encoder.device(), affine_q8_few_source, kMmaEntries[sb_dtype]);
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_output_array(out);
    cu::KernelArgs args;
    args.append(x);
    args.append(w);
    args.append(scales);
    args.append(biases);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    encoder.add_kernel_node_raw(
        kernel, dim3(row_blocks), dim3(32, 8), {}, 0, args.args());
  }

  DEFINE_NAME(AffineQ8Few)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const AffineQ8Few*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }
};

class Bf16GemmFew : public UnaryPrimitive {
public:
  explicit Bf16GemmFew(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Bf16GemmFew requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_output_array(out);

    const int K = x.shape(-1);
    const int M = (K > 0) ? static_cast<int>(x.size() / K) : 0;
    const int N = w.shape(0);
    if (M < 1 || M > 16 || N <= 0 || K < 8 || K % 8 != 0) {
      throw std::invalid_argument("Bf16GemmFew: unsupported replay shape");
    }
    static constexpr const char* kBf16Entries[16] = {
        "mlx_node_bf16_gemm_few<1>",
        "mlx_node_bf16_gemm_few<2>",
        "mlx_node_bf16_gemm_few<3>",
        "mlx_node_bf16_gemm_few<4>",
        "mlx_node_bf16_gemm_few<5>",
        "mlx_node_bf16_gemm_few<6>",
        "mlx_node_bf16_gemm_few<7>",
        "mlx_node_bf16_gemm_few<8>",
        "mlx_node_bf16_gemm_few<9>",
        "mlx_node_bf16_gemm_few<10>",
        "mlx_node_bf16_gemm_few<11>",
        "mlx_node_bf16_gemm_few<12>",
        "mlx_node_bf16_gemm_few<13>",
        "mlx_node_bf16_gemm_few<14>",
        "mlx_node_bf16_gemm_few<15>",
        "mlx_node_bf16_gemm_few<16>",
    };
    auto kernel = cu::jit_kernel(
        encoder.device(), bf16_gemm_few_source, kBf16Entries[M - 1]);
    cu::KernelArgs args;
    args.append(x);
    args.append(w);
    args.append(out);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    encoder.add_kernel_node_raw(
        kernel, dim3((N + 7) / 8), dim3(32, 4), {}, 0, args.args());
  }

  DEFINE_NAME(Bf16GemmFew)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Bf16GemmFew*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }
};

} // namespace
} // namespace mlx::core::gemm_few

extern "C" mlx_array* mlx_affine_q8_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle) {
  try {
    using namespace mlx::core;
    if (!x_handle || !w_handle || !scales_handle || !biases_handle ||
        default_device() != Device::gpu) {
      return nullptr;
    }
    std::vector<array> inputs = {
        *reinterpret_cast<array*>(x_handle), *reinterpret_cast<array*>(w_handle),
        *reinterpret_cast<array*>(scales_handle),
        *reinterpret_cast<array*>(biases_handle)};
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& scales = inputs[2];
    const auto& biases = inputs[3];
    if (x.dtype() != bfloat16 || x.ndim() < 2 || w.ndim() != 2) {
      return nullptr;
    }
    const int K = x.shape(-1);
    if (K < 32 || K % 32 != 0) {
      return nullptr;
    }
    const int64_t rows = static_cast<int64_t>(x.size()) / K;
    const int N = w.shape(0);
    const int sb_dtype = gemm_few::sidecar_dtype_code(scales.dtype());
    if (rows < 1 || rows > 16 || N <= 0 || w.dtype() != uint32 ||
        w.shape(1) != K / 4 || sb_dtype < 0 ||
        scales.ndim() != 2 || scales.shape(0) != N || scales.shape(1) != K / 32 ||
        biases.ndim() != 2 || biases.dtype() != scales.dtype() ||
        biases.shape(0) != N || biases.shape(1) != K / 32) {
      return nullptr;
    }
    auto shape = x.shape();
    shape.back() = N;
    auto result = array(
        std::move(shape), bfloat16,
        std::make_shared<gemm_few::AffineQ8Few>(default_stream(Device::gpu)),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_affine_q8_few_cuda error: " << e.what() << std::endl;
  }
  return nullptr;
}

extern "C" mlx_array* mlx_bf16_gemm_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle) {
  try {
    using namespace mlx::core;
    if (!x_handle || !w_handle || default_device() != Device::gpu) {
      return nullptr;
    }
    std::vector<array> inputs = {
        *reinterpret_cast<array*>(x_handle),
        *reinterpret_cast<array*>(w_handle)};
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    if (x.dtype() != bfloat16 || w.dtype() != bfloat16 ||
        x.ndim() < 2 || w.ndim() != 2) {
      return nullptr;
    }
    const int K = x.shape(-1);
    if (K < 8 || K % 8 != 0 || w.shape(1) != K || w.shape(0) <= 0) {
      return nullptr;
    }
    const int64_t rows = static_cast<int64_t>(x.size()) / K;
    if (rows < 1 || rows > 16) {
      return nullptr;
    }
    auto shape = x.shape();
    shape.back() = w.shape(0);
    auto result = array(
        std::move(shape), bfloat16,
        std::make_shared<gemm_few::Bf16GemmFew>(default_stream(Device::gpu)),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_bf16_gemm_few_cuda error: " << e.what() << std::endl;
  }
  return nullptr;
}

#else

extern "C" mlx_array* mlx_affine_q8_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle) {
  (void)x_handle;
  (void)w_handle;
  (void)scales_handle;
  (void)biases_handle;
  return nullptr;
}

extern "C" mlx_array* mlx_bf16_gemm_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle) {
  (void)x_handle;
  (void)w_handle;
  return nullptr;
}

#endif
