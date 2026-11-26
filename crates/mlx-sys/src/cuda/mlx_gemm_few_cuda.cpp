#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_cuda_ops.h"

#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

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

void bf16_ensure_carveout(CUfunction fn, int carveout_pct);

class AffineQ8Few : public UnaryPrimitive {
public:
  AffineQ8Few(Stream stream, bool symmetric)
      : UnaryPrimitive(stream), symmetric_(symmetric) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("AffineQ8Few requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    if ((symmetric_ && inputs.size() < 3) || (!symmetric_ && inputs.size() < 4)) {
      throw std::invalid_argument("AffineQ8Few: missing input arrays");
    }
    const int sb_dtype = sidecar_dtype_code(inputs[2].dtype());
    if (sb_dtype < 0 || (!symmetric_ && inputs[3].dtype() != inputs[2].dtype())) {
      throw std::invalid_argument("AffineQ8Few: unsupported sidecar dtype");
    }
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array scales = (sb_dtype == 0)
        ? cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
    std::optional<array> biases;
    if (!symmetric_) {
      biases = (sb_dtype == 0)
          ? cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<2>(inputs[3], stream(), encoder);
    }

    const int K = x.shape(-1);
    const int M = (K > 0) ? static_cast<int>(x.size() / K) : 0;
    const int N = w.shape(0);
    if (M < 1 || M > 16 || N <= 0 || K < 32 || K % 32 != 0) {
      throw std::invalid_argument("AffineQ8Few: unsupported replay shape");
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    const int sym_idx = symmetric_ ? 1 : 0;

    auto bind_and_build_args = [&](const array& target_out) {
      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      if (!symmetric_) {
        encoder.set_input_array(*biases);
      }
      encoder.set_output_array(target_out);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      if (!symmetric_) {
        args.append(*biases);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append(target_out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      return args;
    };

    // Opt-in exact-FP32 u4_1row path for all M in 1..=16 (matches M<=3 summation
    // order bit-for-bit when debugging).
    static const bool exact_m16 = [] {
      const char* v = std::getenv("MLX_CUDA_AQ8_EXACT_M16");
      return v && v[0] == '1';
    }();
    const bool has_mma = encoder.device().compute_capability_major() >= 8;

    // Dispatch Tensor Core MMA (cp.async.cg + group stagger) across M in 3..=16 when
    // K >= 256 and N >= 256 (plus M >= 9 for all N on sm_80+); M in 1..=2 uses zero-SMEM
    // 1-warp-per-row streaming uint4 (4 warps/CTA for fine wave scheduling on N=1600..8192).
    const bool use_mma =
        !exact_m16 && has_mma && ((M >= 3 && K >= 256 && N >= 256) || M >= 9);

    if (!use_mma) {
      // M in 1..=2 (and M=3..4 pre-Ampere / narrow-N fallback) at N >= 128, K >= 512:
      // 4-warp/CTA for M<=2 eliminates wave-tail quantization (e.g. q_kv N=1600 -> 400 CTAs);
      // 8-warp/CTA kept for M=3..4 fallback.
      if (!exact_m16 && M <= 4 && N >= 128 && K >= 512) {
        static constexpr const char* kU4WarpEntries[2][3][4] = {
            {
                {"mlx_node_affine_q8_u4_1warp<0,1,4,false>",
                 "mlx_node_affine_q8_u4_1warp<0,2,4,false>",
                 "mlx_node_affine_q8_u4_1warp<0,3,8,false>",
                 "mlx_node_affine_q8_u4_1warp<0,4,8,false>"},
                {"mlx_node_affine_q8_u4_1warp<1,1,4,false>",
                 "mlx_node_affine_q8_u4_1warp<1,2,4,false>",
                 "mlx_node_affine_q8_u4_1warp<1,3,8,false>",
                 "mlx_node_affine_q8_u4_1warp<1,4,8,false>"},
                {"mlx_node_affine_q8_u4_1warp<2,1,4,false>",
                 "mlx_node_affine_q8_u4_1warp<2,2,4,false>",
                 "mlx_node_affine_q8_u4_1warp<2,3,8,false>",
                 "mlx_node_affine_q8_u4_1warp<2,4,8,false>"},
            },
            {
                {"mlx_node_affine_q8_u4_1warp<0,1,4,true>",
                 "mlx_node_affine_q8_u4_1warp<0,2,4,true>",
                 "mlx_node_affine_q8_u4_1warp<0,3,8,true>",
                 "mlx_node_affine_q8_u4_1warp<0,4,8,true>"},
                {"mlx_node_affine_q8_u4_1warp<1,1,4,true>",
                 "mlx_node_affine_q8_u4_1warp<1,2,4,true>",
                 "mlx_node_affine_q8_u4_1warp<1,3,8,true>",
                 "mlx_node_affine_q8_u4_1warp<1,4,8,true>"},
                {"mlx_node_affine_q8_u4_1warp<2,1,4,true>",
                 "mlx_node_affine_q8_u4_1warp<2,2,4,true>",
                 "mlx_node_affine_q8_u4_1warp<2,3,8,true>",
                 "mlx_node_affine_q8_u4_1warp<2,4,8,true>"},
            },
        };
        const int num_warps = (M <= 2) ? 4 : 8;
        auto kernel = cu::jit_kernel(
            encoder.device(), affine_q8_few_source,
            kU4WarpEntries[sym_idx][sb_dtype][M - 1]);
        bf16_ensure_carveout(kernel, 0);
        auto args = bind_and_build_args(out);
        encoder.add_kernel_node_raw(
            kernel, dim3((N + num_warps - 1) / num_warps), dim3(32, num_warps), {}, 0, args.args());
        return;
      }

      // Single-launch coalesced 512B/warp uint4 streaming FP32 FMA kernel for
      // M in 1..=3, narrow N < 256, K < 256, or exact_m16=1.
      static constexpr const char* kU4Entries[2][3][5][2] = {
          {
              {
                  {"mlx_node_affine_q8_u4_1row<0,1,4,false>", "mlx_node_affine_q8_u4_1row<0,1,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<0,2,4,false>", "mlx_node_affine_q8_u4_1row<0,2,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<0,4,4,false>", "mlx_node_affine_q8_u4_1row<0,4,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<0,8,4,false>", "mlx_node_affine_q8_u4_1row<0,8,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<0,16,4,false>", "mlx_node_affine_q8_u4_1row<0,16,4,false>"},
              },
              {
                  {"mlx_node_affine_q8_u4_1row<1,1,4,false>", "mlx_node_affine_q8_u4_1row<1,1,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<1,2,4,false>", "mlx_node_affine_q8_u4_1row<1,2,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<1,4,4,false>", "mlx_node_affine_q8_u4_1row<1,4,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<1,8,4,false>", "mlx_node_affine_q8_u4_1row<1,8,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<1,16,4,false>", "mlx_node_affine_q8_u4_1row<1,16,4,false>"},
              },
              {
                  {"mlx_node_affine_q8_u4_1row<2,1,4,false>", "mlx_node_affine_q8_u4_1row<2,1,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<2,2,4,false>", "mlx_node_affine_q8_u4_1row<2,2,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<2,4,4,false>", "mlx_node_affine_q8_u4_1row<2,4,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<2,8,4,false>", "mlx_node_affine_q8_u4_1row<2,8,8,false>"},
                  {"mlx_node_affine_q8_u4_1row<2,16,4,false>", "mlx_node_affine_q8_u4_1row<2,16,4,false>"},
              },
          },
          {
              {
                  {"mlx_node_affine_q8_u4_1row<0,1,4,true>", "mlx_node_affine_q8_u4_1row<0,1,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<0,2,4,true>", "mlx_node_affine_q8_u4_1row<0,2,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<0,4,4,true>", "mlx_node_affine_q8_u4_1row<0,4,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<0,8,4,true>", "mlx_node_affine_q8_u4_1row<0,8,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<0,16,4,true>", "mlx_node_affine_q8_u4_1row<0,16,4,true>"},
              },
              {
                  {"mlx_node_affine_q8_u4_1row<1,1,4,true>", "mlx_node_affine_q8_u4_1row<1,1,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<1,2,4,true>", "mlx_node_affine_q8_u4_1row<1,2,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<1,4,4,true>", "mlx_node_affine_q8_u4_1row<1,4,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<1,8,4,true>", "mlx_node_affine_q8_u4_1row<1,8,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<1,16,4,true>", "mlx_node_affine_q8_u4_1row<1,16,4,true>"},
              },
              {
                  {"mlx_node_affine_q8_u4_1row<2,1,4,true>", "mlx_node_affine_q8_u4_1row<2,1,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<2,2,4,true>", "mlx_node_affine_q8_u4_1row<2,2,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<2,4,4,true>", "mlx_node_affine_q8_u4_1row<2,4,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<2,8,4,true>", "mlx_node_affine_q8_u4_1row<2,8,8,true>"},
                  {"mlx_node_affine_q8_u4_1row<2,16,4,true>", "mlx_node_affine_q8_u4_1row<2,16,4,true>"},
              },
          },
      };
      const int m_bucket =
          (M == 1) ? 0 : (M == 2 ? 1 : (M <= 4 ? 2 : (M <= 8 ? 3 : 4)));
      const bool use_8w = (M <= 4 && N <= 128);
      const int num_warps = use_8w ? 8 : 4;
      auto kernel = cu::jit_kernel(
          encoder.device(), affine_q8_few_source,
          kU4Entries[sym_idx][sb_dtype][m_bucket][use_8w ? 1 : 0]);
      auto args = bind_and_build_args(out);
      encoder.add_kernel_node_raw(
          kernel, dim3(N), dim3(32, num_warps), {}, 0, args.args());
      return;
    }

    // M in 4..=16: coalesced SMEM-staged BF16 Tensor Core MMA (M_TILES=1 for 4..=8,
    // M_TILES=2 for 9..=16). For wave-quantized narrow N <= 1024 with K >= 2048
    // (e.g. LP_K/V [1024, 2304]: 64 CTAs -> 192 CTAs = 4.0 exact waves on 48 SMs),
    // dispatch deterministic Split-K + vectorized L2 reduce.
    static constexpr const char* kMmaEntries[2][3][2][2] = {
        {
            {
                {"mlx_node_affine_q8_mma<0,1,false,false>", "mlx_node_affine_q8_mma<0,1,true,false>"},
                {"mlx_node_affine_q8_mma<0,2,false,false>", "mlx_node_affine_q8_mma<0,2,true,false>"},
            },
            {
                {"mlx_node_affine_q8_mma<1,1,false,false>", "mlx_node_affine_q8_mma<1,1,true,false>"},
                {"mlx_node_affine_q8_mma<1,2,false,false>", "mlx_node_affine_q8_mma<1,2,true,false>"},
            },
            {
                {"mlx_node_affine_q8_mma<2,1,false,false>", "mlx_node_affine_q8_mma<2,1,true,false>"},
                {"mlx_node_affine_q8_mma<2,2,false,false>", "mlx_node_affine_q8_mma<2,2,true,false>"},
            },
        },
        {
            {
                {"mlx_node_affine_q8_mma<0,1,false,true>", "mlx_node_affine_q8_mma<0,1,true,true>"},
                {"mlx_node_affine_q8_mma<0,2,false,true>", "mlx_node_affine_q8_mma<0,2,true,true>"},
            },
            {
                {"mlx_node_affine_q8_mma<1,1,false,true>", "mlx_node_affine_q8_mma<1,1,true,true>"},
                {"mlx_node_affine_q8_mma<1,2,false,true>", "mlx_node_affine_q8_mma<1,2,true,true>"},
            },
            {
                {"mlx_node_affine_q8_mma<2,1,false,true>", "mlx_node_affine_q8_mma<2,1,true,true>"},
                {"mlx_node_affine_q8_mma<2,2,false,true>", "mlx_node_affine_q8_mma<2,2,true,true>"},
            },
        },
    };
    const int m_tile_idx = (M <= 8) ? 0 : 1;
    const int row_blocks = (N + 15) / 16;
    const int num_tiles = (K >> 5) >> 3;
    int split_k = 1;
    if (N <= 1024 && K >= 2048 && num_tiles >= 6) {
      if ((num_tiles % 3) == 0) {
        split_k = 3;
      } else if (row_blocks <= 48 && (num_tiles % 4) == 0) {
        split_k = 4;
      } else if ((num_tiles % 2) == 0) {
        split_k = 2;
      }
    }

    if (split_k > 1) {
      array partials(Shape{split_k, M, N}, float32, nullptr, {});
      partials.set_data(cu::malloc_async(partials.nbytes(), encoder));
      encoder.add_temporary(partials);

      auto kernel = cu::jit_kernel(
          encoder.device(), affine_q8_few_source,
          kMmaEntries[sym_idx][sb_dtype][m_tile_idx][1]);
      auto args = bind_and_build_args(partials);
      encoder.add_kernel_node_raw(
          kernel, dim3(row_blocks, split_k), dim3(32, 8), {}, 0, args.args());

      encoder.set_input_array(partials);
      encoder.set_output_array(out);
      auto red_kernel = cu::jit_kernel(
          encoder.device(), affine_q8_few_source,
          "mlx_node_affine_q8_g32_reduce");
      cu::KernelArgs red_args;
      red_args.append(partials);
      red_args.append(out);
      red_args.append<int32_t>(M);
      red_args.append<int32_t>(N);
      red_args.append<int32_t>(split_k);
      const int total = M * N;
      const int work = ((total & 3) == 0) ? (total >> 2) : total;
      encoder.add_kernel_node_raw(
          red_kernel, dim3((work + 63) / 64), dim3(64), {}, 0, red_args.args());
      return;
    }

    auto kernel = cu::jit_kernel(
        encoder.device(), affine_q8_few_source,
        kMmaEntries[sym_idx][sb_dtype][m_tile_idx][0]);
    auto args = bind_and_build_args(out);
    encoder.add_kernel_node_raw(
        kernel, dim3(row_blocks), dim3(32, 8), {}, 0, args.args());
  }

  DEFINE_NAME(AffineQ8Few)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const AffineQ8Few*>(&other);
    return rhs != nullptr && rhs->symmetric_ == symmetric_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }

private:
  bool symmetric_;
};

class AffineQ8FewPair : public Primitive {
public:
  AffineQ8FewPair(Stream stream, bool symmetric)
      : Primitive(stream), symmetric_(symmetric) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("AffineQ8FewPair requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    const size_t expected = symmetric_ ? 6u : 8u;
    if (inputs.size() != expected || outputs.size() != 2u) {
      throw std::invalid_argument("AffineQ8FewPair: invalid input/output count");
    }
    const size_t idx_x1 = symmetric_ ? 3u : 4u;
    const size_t idx_w1 = symmetric_ ? 4u : 5u;
    const size_t idx_s1 = symmetric_ ? 5u : 6u;
    const int sb_dtype = sidecar_dtype_code(inputs[2].dtype());
    if (sb_dtype < 0 || inputs[idx_s1].dtype() != inputs[2].dtype() ||
        (!symmetric_ &&
         (inputs[3].dtype() != inputs[2].dtype() ||
          inputs[7].dtype() != inputs[2].dtype()))) {
      throw std::invalid_argument("AffineQ8FewPair: unsupported sidecar dtype");
    }

    array x0 = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w0 = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array s0 = (sb_dtype == 0)
        ? cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
    std::optional<array> b0;
    if (!symmetric_) {
      b0 = (sb_dtype == 0)
          ? cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<2>(inputs[3], stream(), encoder);
    }

    array x1 =
        cu::ensure_row_contiguous_aligned<16>(inputs[idx_x1], stream(), encoder);
    array w1 =
        cu::ensure_row_contiguous_aligned<16>(inputs[idx_w1], stream(), encoder);
    array s1 = (sb_dtype == 0)
        ? cu::ensure_row_contiguous_aligned<4>(inputs[idx_s1], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<2>(inputs[idx_s1], stream(), encoder);
    std::optional<array> b1;
    if (!symmetric_) {
      b1 = (sb_dtype == 0)
          ? cu::ensure_row_contiguous_aligned<4>(inputs[7], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<2>(inputs[7], stream(), encoder);
    }

    const int K = x0.shape(-1);
    const int M = (K > 0) ? static_cast<int>(x0.size() / K) : 0;
    const int N = w0.shape(0);
    if (M < 1 || M > 16 || N < 16 || (N % 16) != 0 || K < 256 || (K % 32) != 0 ||
        x1.shape(-1) != K || static_cast<int>(x1.size() / K) != M ||
        w1.shape(0) != N) {
      throw std::invalid_argument("AffineQ8FewPair: unsupported replay shape");
    }

    auto& out0 = outputs[0];
    auto& out1 = outputs[1];
    out0.set_data(cu::malloc_async(out0.nbytes(), encoder));
    out1.set_data(cu::malloc_async(out1.nbytes(), encoder));

    encoder.set_input_array(x0);
    encoder.set_input_array(w0);
    encoder.set_input_array(s0);
    if (!symmetric_) {
      encoder.set_input_array(*b0);
    }
    encoder.set_input_array(x1);
    encoder.set_input_array(w1);
    encoder.set_input_array(s1);
    if (!symmetric_) {
      encoder.set_input_array(*b1);
    }
    encoder.set_output_array(out0);
    encoder.set_output_array(out1);

    cu::KernelArgs args;
    args.append(x0);
    args.append(w0);
    args.append(s0);
    if (!symmetric_) {
      args.append(*b0);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out0);
    args.append(x1);
    args.append(w1);
    args.append(s1);
    if (!symmetric_) {
      args.append(*b1);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out1);
    args.append<int32_t>(M);
    args.append<int32_t>(N);
    args.append<int32_t>(K);

    const int sym_idx = symmetric_ ? 1 : 0;
    if (M <= 2) {
      static constexpr const char* kU4WarpPairEntries[2][3][2] = {
          {
              {"mlx_node_affine_q8_u4_1warp_pair<0,1,8,false>",
               "mlx_node_affine_q8_u4_1warp_pair<0,2,8,false>"},
              {"mlx_node_affine_q8_u4_1warp_pair<1,1,8,false>",
               "mlx_node_affine_q8_u4_1warp_pair<1,2,8,false>"},
              {"mlx_node_affine_q8_u4_1warp_pair<2,1,8,false>",
               "mlx_node_affine_q8_u4_1warp_pair<2,2,8,false>"},
          },
          {
              {"mlx_node_affine_q8_u4_1warp_pair<0,1,8,true>",
               "mlx_node_affine_q8_u4_1warp_pair<0,2,8,true>"},
              {"mlx_node_affine_q8_u4_1warp_pair<1,1,8,true>",
               "mlx_node_affine_q8_u4_1warp_pair<1,2,8,true>"},
              {"mlx_node_affine_q8_u4_1warp_pair<2,1,8,true>",
               "mlx_node_affine_q8_u4_1warp_pair<2,2,8,true>"},
          },
      };
      auto kernel = cu::jit_kernel(
          encoder.device(), affine_q8_few_source,
          kU4WarpPairEntries[sym_idx][sb_dtype][M - 1]);
      encoder.add_kernel_node_raw(
          kernel, dim3(static_cast<uint32_t>(N / 8), 2, 1), dim3(32, 8, 1), {},
          0, args.args());
      return;
    }

    if (encoder.device().compute_capability_major() < 8) {
      throw std::runtime_error("AffineQ8FewPair M>=3 requires sm_80+");
    }
    static constexpr const char* kMmaPairEntries[2][3][2] = {
        {
            {"mlx_node_affine_q8_mma_pair<0,1,false>",
             "mlx_node_affine_q8_mma_pair<0,2,false>"},
            {"mlx_node_affine_q8_mma_pair<1,1,false>",
             "mlx_node_affine_q8_mma_pair<1,2,false>"},
            {"mlx_node_affine_q8_mma_pair<2,1,false>",
             "mlx_node_affine_q8_mma_pair<2,2,false>"},
        },
        {
            {"mlx_node_affine_q8_mma_pair<0,1,true>",
             "mlx_node_affine_q8_mma_pair<0,2,true>"},
            {"mlx_node_affine_q8_mma_pair<1,1,true>",
             "mlx_node_affine_q8_mma_pair<1,2,true>"},
            {"mlx_node_affine_q8_mma_pair<2,1,true>",
             "mlx_node_affine_q8_mma_pair<2,2,true>"},
        },
    };
    const int m_tile_idx = (M <= 8) ? 0 : 1;
    auto kernel = cu::jit_kernel(
        encoder.device(), affine_q8_few_source,
        kMmaPairEntries[sym_idx][sb_dtype][m_tile_idx]);
    encoder.add_kernel_node_raw(
        kernel, dim3(static_cast<uint32_t>(N / 16), 2, 1), dim3(32, 8, 1), {},
        0, args.args());
  }

  DEFINE_NAME(AffineQ8FewPair)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const AffineQ8FewPair*>(&other);
    return rhs != nullptr && rhs->symmetric_ == symmetric_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const size_t idx_x1 = symmetric_ ? 3u : 4u;
    const size_t idx_w1 = symmetric_ ? 4u : 5u;
    auto s0 = inputs[0].shape();
    s0.back() = inputs[1].shape(0);
    auto s1 = inputs[idx_x1].shape();
    s1.back() = inputs[idx_w1].shape(0);
    return {std::move(s0), std::move(s1)};
  }

private:
  bool symmetric_;
};

int bf16_sm_count(int dev_ordinal) {
  static std::mutex mu;
  static std::unordered_map<int, int> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(dev_ordinal);
  if (it != cache.end()) {
    return it->second;
  }
  int sms = 48;
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, dev_ordinal) == CUDA_SUCCESS) {
    int val = 0;
    if (cuDeviceGetAttribute(
            &val, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) == CUDA_SUCCESS &&
        val > 0) {
      sms = val;
    }
  }
  cache[dev_ordinal] = sms;
  return sms;
}

void bf16_ensure_carveout(CUfunction fn, int carveout_pct) {
  static std::mutex mu;
  static std::unordered_map<CUfunction, int> configured;
  std::lock_guard<std::mutex> lock(mu);
  auto it = configured.find(fn);
  if (it != configured.end() && it->second == carveout_pct) {
    return;
  }
  (void)cuFuncSetAttribute(
      fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, carveout_pct);
  configured[fn] = carveout_pct;
}

int bf16_choose_split_k(
    int grid_ctas, int num_units, int num_sms, int target_ctas_per_sm) noexcept {
  const int target = num_sms * target_ctas_per_sm;
  if (grid_ctas <= 0 || grid_ctas >= target || num_units < 4) {
    return 1;
  }
  const int max_sk = std::min(8, num_units / 2);
  if (max_sk < 2) {
    return 1;
  }
  const int needed = (target + grid_ctas - 1) / grid_ctas;
  if (needed <= 1) {
    return 1;
  }
  if (needed >= max_sk) {
    return max_sk;
  }
  for (int sk = needed; sk <= std::min(needed + 1, max_sk); ++sk) {
    if ((num_units % sk) == 0) {
      return sk;
    }
  }
  return needed;
}

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

    const int K = x.shape(-1);
    const int M = (K > 0) ? static_cast<int>(x.size() / K) : 0;
    const int N = w.shape(0);
    if (M < 1 || M > 16 || N <= 0 || K < 8 || K % 8 != 0) {
      throw std::invalid_argument("Bf16GemmFew: unsupported replay shape");
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    const int num_sms = bf16_sm_count(encoder.device().cuda_device());

    auto bind_and_build_args = [&](const array& target_out) {
      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_output_array(target_out);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(target_out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      return args;
    };

    auto launch_reduce = [&](const array& partials, int split_k) {
      encoder.set_input_array(partials);
      encoder.set_output_array(out);
      auto red_kernel = cu::jit_kernel(
          encoder.device(), bf16_gemm_few_source,
          "mlx_node_bf16_gemm_few_reduce");
      cu::KernelArgs red_args;
      red_args.append(partials);
      red_args.append(out);
      red_args.append<int32_t>(M);
      red_args.append<int32_t>(N);
      red_args.append<int32_t>(split_k);
      const int total = M * N;
      const int work = ((total & 3) == 0) ? (total >> 2) : total;
      encoder.add_kernel_node_raw(
          red_kernel, dim3(static_cast<uint32_t>((work + 127) / 128)),
          dim3(128), {}, 0, red_args.args());
    };

    if (M <= 3) {
      // M in 1..=3: 2-rows-per-warp, 4-warp/CTA (8 rows/CTA, 0 B SMEM) streaming
      // uint4 kernel sharing L1-cached X across 2 weight rows per warp.
      static constexpr const char* kU4TwoRowEntries[3][2] = {
          {"mlx_node_bf16_gemm_few_u4_2row<1, false>",
           "mlx_node_bf16_gemm_few_u4_2row<1, true>"},
          {"mlx_node_bf16_gemm_few_u4_2row<2, false>",
           "mlx_node_bf16_gemm_few_u4_2row<2, true>"},
          {"mlx_node_bf16_gemm_few_u4_2row<3, false>",
           "mlx_node_bf16_gemm_few_u4_2row<3, true>"},
      };
      const int grid_n = (N + 7) / 8;
      const int k_chunks = ((K >> 3) + 31) >> 5;
      const int split_k = bf16_choose_split_k(grid_n, k_chunks, num_sms, 2);

      auto kernel = cu::jit_kernel(
          encoder.device(), bf16_gemm_few_source,
          kU4TwoRowEntries[M - 1][split_k > 1 ? 1 : 0]);
      bf16_ensure_carveout(kernel, 0);

      if (split_k > 1) {
        array partials(Shape{split_k, M, N}, float32, nullptr, {});
        partials.set_data(cu::malloc_async(partials.nbytes(), encoder));
        encoder.add_temporary(partials);
        auto args = bind_and_build_args(partials);
        encoder.add_kernel_node_raw(
            kernel,
            dim3(static_cast<uint32_t>(grid_n), static_cast<uint32_t>(split_k)),
            dim3(32, 4), {}, 0, args.args());
        launch_reduce(partials, split_k);
        return;
      }

      auto args = bind_and_build_args(out);
      encoder.add_kernel_node_raw(
          kernel, dim3(static_cast<uint32_t>(grid_n)), dim3(32, 4), {}, 0,
          args.args());
      return;
    }

    const bool has_mma = encoder.device().compute_capability_major() >= 8;
    const bool use_mma = has_mma && (M >= 9) && (K >= 128) && ((K & 15) == 0);

    if (use_mma) {
      static constexpr const char* kMma2Entries[2] = {
          "mlx_node_bf16_gemm_few_mma<2, false>",
          "mlx_node_bf16_gemm_few_mma<2, true>",
      };
      const int row_blocks = (N + 15) / 16;
      const int num_tiles = K >> 7;
      // MMA uses 8-warp (256-thread) CTAs; >= 2 CTAs/SM already saturates DRAM
      // bandwidth without paying Split-K partial alloc + reduce overhead.
      const int split_k = bf16_choose_split_k(row_blocks, num_tiles, num_sms, 2);

      auto kernel = cu::jit_kernel(
          encoder.device(), bf16_gemm_few_source,
          kMma2Entries[split_k > 1 ? 1 : 0]);
      bf16_ensure_carveout(kernel, 25);

      if (split_k > 1) {
        array partials(Shape{split_k, M, N}, float32, nullptr, {});
        partials.set_data(cu::malloc_async(partials.nbytes(), encoder));
        encoder.add_temporary(partials);
        auto args = bind_and_build_args(partials);
        encoder.add_kernel_node_raw(
            kernel,
            dim3(static_cast<uint32_t>(row_blocks), static_cast<uint32_t>(split_k)),
            dim3(32, 8), {}, 0, args.args());
        launch_reduce(partials, split_k);
        return;
      }

      auto args = bind_and_build_args(out);
      encoder.add_kernel_node_raw(
          kernel, dim3(static_cast<uint32_t>(row_blocks)), dim3(32, 8), {}, 0,
          args.args());
      return;
    }

    // M in 4..=8 (and M in 9..=16 fallback when sm < 80, K < 128, or K % 16 != 0):
    // 1-row-per-warp 4-warp streaming uint4 (0 B SMEM, 0 barriers, sk=1 for N >= 384).
    static constexpr const char* kU4Entries[2][2] = {
        {"mlx_node_bf16_gemm_few_u4<8, 4, false>",
         "mlx_node_bf16_gemm_few_u4<8, 4, true>"},
        {"mlx_node_bf16_gemm_few_u4<16, 4, false>",
         "mlx_node_bf16_gemm_few_u4<16, 4, true>"},
    };
    const int m_idx = (M <= 8) ? 0 : 1;
    const int grid_n = (N + 3) / 4;
    const int k_chunks = ((K >> 3) + 31) >> 5;
    const int split_k = bf16_choose_split_k(grid_n, k_chunks, num_sms, 2);

    auto kernel = cu::jit_kernel(
        encoder.device(), bf16_gemm_few_source,
        kU4Entries[m_idx][split_k > 1 ? 1 : 0]);
    bf16_ensure_carveout(kernel, 0);

    if (split_k > 1) {
      array partials(Shape{split_k, M, N}, float32, nullptr, {});
      partials.set_data(cu::malloc_async(partials.nbytes(), encoder));
      encoder.add_temporary(partials);
      auto args = bind_and_build_args(partials);
      encoder.add_kernel_node_raw(
          kernel,
          dim3(static_cast<uint32_t>(grid_n), static_cast<uint32_t>(split_k)),
          dim3(32, 4), {}, 0, args.args());
      launch_reduce(partials, split_k);
      return;
    }

    auto args = bind_and_build_args(out);
    encoder.add_kernel_node_raw(
        kernel, dim3(static_cast<uint32_t>(grid_n)), dim3(32, 4), {}, 0,
        args.args());
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

class Bf16GemmFewPair : public Primitive {
public:
  explicit Bf16GemmFewPair(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Bf16GemmFewPair requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    if (inputs.size() != 4u || outputs.size() != 2u) {
      throw std::invalid_argument("Bf16GemmFewPair: invalid input/output count");
    }
    array x0 = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w0 = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array x1 = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array w1 = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);

    const int K = x0.shape(-1);
    const int M = (K > 0) ? static_cast<int>(x0.size() / K) : 0;
    const int N = w0.shape(0);
    if (M < 1 || M > 16 || N <= 0 || K < 8 || (K % 8) != 0 ||
        x1.shape(-1) != K || static_cast<int>(x1.size() / K) != M ||
        w1.shape(0) != N) {
      throw std::invalid_argument("Bf16GemmFewPair: unsupported replay shape");
    }

    auto& out0 = outputs[0];
    auto& out1 = outputs[1];
    out0.set_data(cu::malloc_async(out0.nbytes(), encoder));
    out1.set_data(cu::malloc_async(out1.nbytes(), encoder));
    const int num_sms = bf16_sm_count(encoder.device().cuda_device());

    auto bind_and_build_args = [&](const array& dst0, const array& dst1) {
      encoder.set_input_array(x0);
      encoder.set_input_array(w0);
      encoder.set_input_array(x1);
      encoder.set_input_array(w1);
      encoder.set_output_array(dst0);
      encoder.set_output_array(dst1);
      cu::KernelArgs args;
      args.append(x0);
      args.append(w0);
      args.append(dst0);
      args.append(x1);
      args.append(w1);
      args.append(dst1);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      return args;
    };

    auto launch_reduce_pair =
        [&](const array& p0, const array& p1, int split_k) {
          encoder.set_input_array(p0);
          encoder.set_input_array(p1);
          encoder.set_output_array(out0);
          encoder.set_output_array(out1);
          auto red_kernel = cu::jit_kernel(
              encoder.device(), bf16_gemm_few_source,
              "mlx_node_bf16_gemm_few_reduce_pair");
          cu::KernelArgs red_args;
          red_args.append(p0);
          red_args.append(out0);
          red_args.append(p1);
          red_args.append(out1);
          red_args.append<int32_t>(M);
          red_args.append<int32_t>(N);
          red_args.append<int32_t>(split_k);
          const int total = M * N;
          const int work = ((total & 3) == 0) ? (total >> 2) : total;
          encoder.add_kernel_node_raw(
              red_kernel,
              dim3(static_cast<uint32_t>((work + 127) / 128), 2, 1),
              dim3(128), {}, 0, red_args.args());
        };

    if (M <= 3) {
      static constexpr const char* kU4TwoRowPairEntries[3][2] = {
          {"mlx_node_bf16_gemm_few_u4_2row_pair<1, false>",
           "mlx_node_bf16_gemm_few_u4_2row_pair<1, true>"},
          {"mlx_node_bf16_gemm_few_u4_2row_pair<2, false>",
           "mlx_node_bf16_gemm_few_u4_2row_pair<2, true>"},
          {"mlx_node_bf16_gemm_few_u4_2row_pair<3, false>",
           "mlx_node_bf16_gemm_few_u4_2row_pair<3, true>"},
      };
      const int grid_n = (N + 7) / 8;
      const int k_chunks = ((K >> 3) + 31) >> 5;
      const int split_k = bf16_choose_split_k(grid_n * 2, k_chunks, num_sms, 2);

      auto kernel = cu::jit_kernel(
          encoder.device(), bf16_gemm_few_source,
          kU4TwoRowPairEntries[M - 1][split_k > 1 ? 1 : 0]);
      bf16_ensure_carveout(kernel, 0);

      if (split_k > 1) {
        array p0(Shape{split_k, M, N}, float32, nullptr, {});
        array p1(Shape{split_k, M, N}, float32, nullptr, {});
        p0.set_data(cu::malloc_async(p0.nbytes(), encoder));
        p1.set_data(cu::malloc_async(p1.nbytes(), encoder));
        encoder.add_temporary(p0);
        encoder.add_temporary(p1);
        auto args = bind_and_build_args(p0, p1);
        encoder.add_kernel_node_raw(
            kernel,
            dim3(static_cast<uint32_t>(grid_n), static_cast<uint32_t>(split_k), 2),
            dim3(32, 4, 1), {}, 0, args.args());
        launch_reduce_pair(p0, p1, split_k);
        return;
      }

      auto args = bind_and_build_args(out0, out1);
      encoder.add_kernel_node_raw(
          kernel, dim3(static_cast<uint32_t>(grid_n), 1, 2), dim3(32, 4, 1), {},
          0, args.args());
      return;
    }

    const bool has_mma = encoder.device().compute_capability_major() >= 8;
    const bool use_mma = has_mma && (M >= 9) && (K >= 128) && ((K & 15) == 0);

    if (use_mma) {
      static constexpr const char* kMma2PairEntries[2] = {
          "mlx_node_bf16_gemm_few_mma_pair<2, false>",
          "mlx_node_bf16_gemm_few_mma_pair<2, true>",
      };
      const int row_blocks = (N + 15) / 16;
      const int num_tiles = K >> 7;
      const int split_k =
          bf16_choose_split_k(row_blocks * 2, num_tiles, num_sms, 2);

      auto kernel = cu::jit_kernel(
          encoder.device(), bf16_gemm_few_source,
          kMma2PairEntries[split_k > 1 ? 1 : 0]);
      bf16_ensure_carveout(kernel, 25);

      if (split_k > 1) {
        array p0(Shape{split_k, M, N}, float32, nullptr, {});
        array p1(Shape{split_k, M, N}, float32, nullptr, {});
        p0.set_data(cu::malloc_async(p0.nbytes(), encoder));
        p1.set_data(cu::malloc_async(p1.nbytes(), encoder));
        encoder.add_temporary(p0);
        encoder.add_temporary(p1);
        auto args = bind_and_build_args(p0, p1);
        encoder.add_kernel_node_raw(
            kernel,
            dim3(
                static_cast<uint32_t>(row_blocks),
                static_cast<uint32_t>(split_k),
                2),
            dim3(32, 8, 1), {}, 0, args.args());
        launch_reduce_pair(p0, p1, split_k);
        return;
      }

      auto args = bind_and_build_args(out0, out1);
      encoder.add_kernel_node_raw(
          kernel, dim3(static_cast<uint32_t>(row_blocks), 1, 2),
          dim3(32, 8, 1), {}, 0, args.args());
      return;
    }

    static constexpr const char* kU4PairEntries[2][2] = {
        {"mlx_node_bf16_gemm_few_u4_pair<8, 4, false>",
         "mlx_node_bf16_gemm_few_u4_pair<8, 4, true>"},
        {"mlx_node_bf16_gemm_few_u4_pair<16, 4, false>",
         "mlx_node_bf16_gemm_few_u4_pair<16, 4, true>"},
    };
    const int m_idx = (M <= 8) ? 0 : 1;
    const int grid_n = (N + 3) / 4;
    const int k_chunks = ((K >> 3) + 31) >> 5;
    const int split_k = bf16_choose_split_k(grid_n * 2, k_chunks, num_sms, 2);

    auto kernel = cu::jit_kernel(
        encoder.device(), bf16_gemm_few_source,
        kU4PairEntries[m_idx][split_k > 1 ? 1 : 0]);
    bf16_ensure_carveout(kernel, 0);

    if (split_k > 1) {
      array p0(Shape{split_k, M, N}, float32, nullptr, {});
      array p1(Shape{split_k, M, N}, float32, nullptr, {});
      p0.set_data(cu::malloc_async(p0.nbytes(), encoder));
      p1.set_data(cu::malloc_async(p1.nbytes(), encoder));
      encoder.add_temporary(p0);
      encoder.add_temporary(p1);
      auto args = bind_and_build_args(p0, p1);
      encoder.add_kernel_node_raw(
          kernel,
          dim3(static_cast<uint32_t>(grid_n), static_cast<uint32_t>(split_k), 2),
          dim3(32, 4, 1), {}, 0, args.args());
      launch_reduce_pair(p0, p1, split_k);
      return;
    }

    auto args = bind_and_build_args(out0, out1);
    encoder.add_kernel_node_raw(
        kernel, dim3(static_cast<uint32_t>(grid_n), 1, 2), dim3(32, 4, 1), {},
        0, args.args());
  }

  DEFINE_NAME(Bf16GemmFewPair)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Bf16GemmFewPair*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto s0 = inputs[0].shape();
    s0.back() = inputs[1].shape(0);
    auto s1 = inputs[2].shape();
    s1.back() = inputs[3].shape(0);
    return {std::move(s0), std::move(s1)};
  }
};

} // namespace
} // namespace mlx::core::gemm_few

extern "C" mlx_array* mlx_affine_q8_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle) {
  try {
    using namespace mlx::core;
    if (!x_handle || !w_handle || !scales_handle ||
        default_device() != Device::gpu) {
      return nullptr;
    }
    const bool symmetric = (biases_handle == nullptr);
    std::vector<array> inputs;
    inputs.reserve(symmetric ? 3 : 4);
    inputs.push_back(*reinterpret_cast<array*>(x_handle));
    inputs.push_back(*reinterpret_cast<array*>(w_handle));
    inputs.push_back(*reinterpret_cast<array*>(scales_handle));
    if (!symmetric) {
      inputs.push_back(*reinterpret_cast<array*>(biases_handle));
    }
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& scales = inputs[2];
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
        scales.ndim() != 2 || scales.shape(0) != N || scales.shape(1) != K / 32) {
      return nullptr;
    }
    if (!symmetric) {
      const auto& biases = inputs[3];
      if (biases.ndim() != 2 || biases.dtype() != scales.dtype() ||
          biases.shape(0) != N || biases.shape(1) != K / 32) {
        return nullptr;
      }
    }
    auto shape = x.shape();
    shape.back() = N;
    auto result = array(
        std::move(shape), bfloat16,
        std::make_shared<gemm_few::AffineQ8Few>(
            default_stream(Device::gpu), symmetric),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_affine_q8_few_cuda error: " << e.what() << std::endl;
  }
  return nullptr;
}

extern "C" bool mlx_affine_q8_few_pair_cuda(
    mlx_array* x0, mlx_array* w0, mlx_array* s0, mlx_array* b0,
    mlx_array* x1, mlx_array* w1, mlx_array* s1, mlx_array* b1,
    int group_size,
    mlx_array** out0, mlx_array** out1) {
  if (out0) {
    *out0 = nullptr;
  }
  if (out1) {
    *out1 = nullptr;
  }
  try {
    using namespace mlx::core;
    static const bool exact_m16 = [] {
      const char* v = std::getenv("MLX_CUDA_AQ8_EXACT_M16");
      return v && v[0] == '1';
    }();
    if (exact_m16 || group_size != 32 ||
        !x0 || !w0 || !s0 || !x1 || !w1 || !s1 || !out0 || !out1 ||
        ((b0 == nullptr) != (b1 == nullptr)) ||
        default_device() != Device::gpu) {
      return false;
    }
    const bool symmetric = (b0 == nullptr);
    const auto& ax0 = *reinterpret_cast<array*>(x0);
    const auto& aw0 = *reinterpret_cast<array*>(w0);
    const auto& as0 = *reinterpret_cast<array*>(s0);
    const auto& ax1 = *reinterpret_cast<array*>(x1);
    const auto& aw1 = *reinterpret_cast<array*>(w1);
    const auto& as1 = *reinterpret_cast<array*>(s1);
    if (ax0.dtype() != bfloat16 || ax1.dtype() != bfloat16 ||
        ax0.ndim() < 2 || ax1.ndim() < 2 ||
        aw0.ndim() != 2 || aw1.ndim() != 2) {
      return false;
    }
    const int K = ax0.shape(-1);
    if (K < 256 || (K % 32) != 0 || ax1.shape(-1) != K) {
      return false;
    }
    const int64_t rows0 = static_cast<int64_t>(ax0.size()) / K;
    const int64_t rows1 = static_cast<int64_t>(ax1.size()) / K;
    const int N = aw0.shape(0);
    const int sb_dtype = gemm_few::sidecar_dtype_code(as0.dtype());
    if (rows0 < 1 || rows0 > 16 || rows1 != rows0 ||
        N < 16 || (N % 16) != 0 || aw1.shape(0) != N ||
        aw0.dtype() != uint32 || aw1.dtype() != uint32 ||
        aw0.shape(1) != K / 4 || aw1.shape(1) != K / 4 ||
        sb_dtype < 0 || as1.dtype() != as0.dtype() ||
        as0.ndim() != 2 || as0.shape(0) != N || as0.shape(1) != K / 32 ||
        as1.ndim() != 2 || as1.shape(0) != N || as1.shape(1) != K / 32) {
      return false;
    }
    if (rows0 >= 3 && cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    std::vector<array> inputs;
    inputs.reserve(symmetric ? 6u : 8u);
    inputs.push_back(ax0);
    inputs.push_back(aw0);
    inputs.push_back(as0);
    if (!symmetric) {
      const auto& ab0 = *reinterpret_cast<array*>(b0);
      if (ab0.ndim() != 2 || ab0.dtype() != as0.dtype() ||
          ab0.shape(0) != N || ab0.shape(1) != K / 32) {
        return false;
      }
      inputs.push_back(ab0);
    }
    inputs.push_back(ax1);
    inputs.push_back(aw1);
    inputs.push_back(as1);
    if (!symmetric) {
      const auto& ab1 = *reinterpret_cast<array*>(b1);
      if (ab1.ndim() != 2 || ab1.dtype() != as1.dtype() ||
          ab1.shape(0) != N || ab1.shape(1) != K / 32) {
        return false;
      }
      inputs.push_back(ab1);
    }
    auto shape0 = ax0.shape();
    shape0.back() = N;
    auto shape1 = ax1.shape();
    shape1.back() = N;
    auto outputs = array::make_arrays(
        {std::move(shape0), std::move(shape1)}, {bfloat16, bfloat16},
        std::make_shared<gemm_few::AffineQ8FewPair>(
            default_stream(Device::gpu), symmetric),
        std::move(inputs));
    *out0 = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out1 = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mlx_affine_q8_few_pair_cuda error: " << e.what() << std::endl;
  }
  return false;
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

extern "C" bool mlx_bf16_gemm_few_pair_cuda(
    mlx_array* x0_handle, mlx_array* w0_handle,
    mlx_array* x1_handle, mlx_array* w1_handle,
    mlx_array** out0_handle, mlx_array** out1_handle) {
  if (out0_handle) {
    *out0_handle = nullptr;
  }
  if (out1_handle) {
    *out1_handle = nullptr;
  }
  try {
    using namespace mlx::core;
    if (!x0_handle || !w0_handle || !x1_handle || !w1_handle ||
        !out0_handle || !out1_handle || default_device() != Device::gpu) {
      return false;
    }
    const auto& x0 = *reinterpret_cast<array*>(x0_handle);
    const auto& w0 = *reinterpret_cast<array*>(w0_handle);
    const auto& x1 = *reinterpret_cast<array*>(x1_handle);
    const auto& w1 = *reinterpret_cast<array*>(w1_handle);
    if (x0.dtype() != bfloat16 || w0.dtype() != bfloat16 ||
        x1.dtype() != bfloat16 || w1.dtype() != bfloat16 ||
        x0.ndim() < 2 || x1.ndim() < 2 || w0.ndim() != 2 || w1.ndim() != 2) {
      return false;
    }
    const int K = x0.shape(-1);
    const int N = w0.shape(0);
    if (K < 8 || (K % 8) != 0 || x1.shape(-1) != K ||
        w0.shape(1) != K || w1.shape(1) != K || N <= 0 || w1.shape(0) != N) {
      return false;
    }
    const int64_t rows0 = static_cast<int64_t>(x0.size()) / K;
    const int64_t rows1 = static_cast<int64_t>(x1.size()) / K;
    if (rows0 < 1 || rows0 > 16 || rows1 != rows0) {
      return false;
    }
    auto shape0 = x0.shape();
    shape0.back() = N;
    auto shape1 = x1.shape();
    shape1.back() = N;
    auto outputs = array::make_arrays(
        {std::move(shape0), std::move(shape1)}, {bfloat16, bfloat16},
        std::make_shared<gemm_few::Bf16GemmFewPair>(
            default_stream(Device::gpu)),
        {x0, w0, x1, w1});
    *out0_handle = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out1_handle = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mlx_bf16_gemm_few_pair_cuda error: " << e.what() << std::endl;
  }
  return false;
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

extern "C" bool mlx_affine_q8_few_pair_cuda(
    mlx_array* x0, mlx_array* w0, mlx_array* s0, mlx_array* b0,
    mlx_array* x1, mlx_array* w1, mlx_array* s1, mlx_array* b1,
    int group_size,
    mlx_array** out0, mlx_array** out1) {
  (void)x0;
  (void)w0;
  (void)s0;
  (void)b0;
  (void)x1;
  (void)w1;
  (void)s1;
  (void)b1;
  (void)group_size;
  if (out0) {
    *out0 = nullptr;
  }
  if (out1) {
    *out1 = nullptr;
  }
  return false;
}

extern "C" mlx_array* mlx_bf16_gemm_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle) {
  (void)x_handle;
  (void)w_handle;
  return nullptr;
}

extern "C" bool mlx_bf16_gemm_few_pair_cuda(
    mlx_array* x0_handle, mlx_array* w0_handle,
    mlx_array* x1_handle, mlx_array* w1_handle,
    mlx_array** out0_handle, mlx_array** out1_handle) {
  (void)x0_handle;
  (void)w0_handle;
  (void)x1_handle;
  (void)w1_handle;
  if (out0_handle) {
    *out0_handle = nullptr;
  }
  if (out1_handle) {
    *out1_handle = nullptr;
  }
  return false;
}

#endif
