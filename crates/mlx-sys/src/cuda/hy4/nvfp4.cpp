#include "cuda/hy4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx/backend/cuda/quantized/qqmm_impl.h"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mlx::core::hy4_nvfp4_cuda {
namespace {

using namespace mlx_node::hy4_cuda;

constexpr char hy4_nvfp4_few_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_nvfp4_few.cu.inc"
    ;

constexpr char hy4_nvfp4_prefill_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_nvfp4_prefill.cu.inc"
    ;

constexpr char hy4_grouped_q8_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_grouped_q8.cu.inc"
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

int persistent_ctas(CUfunction fn, int block_size, int dev_ordinal, int max_tiles) {
  static std::mutex mu;
  static std::unordered_map<CUfunction, int> active_cache;
  int active_per_sm = 0;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = active_cache.find(fn);
    if (it != active_cache.end()) {
      active_per_sm = it->second;
    } else {
      int blocks = 0;
      if (cuOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, fn, block_size, 0) == CUDA_SUCCESS &&
          blocks > 0) {
        active_per_sm = blocks;
      } else {
        active_per_sm = 4;
      }
      active_cache[fn] = active_per_sm;
    }
  }
  const int num_sms = device_sm_count(dev_ordinal);
  const int target = active_per_sm * num_sms;
  return std::max(1, std::min(max_tiles, target));
}

// ============================================================================
// 1. Pre-Swizzle NVFP4 Weight Scales to 128x4 cuBLASLt Tiled Layout
// ============================================================================
class Hy4Nvfp4SwizzleScalesCuda final : public UnaryPrimitive {
 public:
  explicit Hy4Nvfp4SwizzleScalesCuda(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4Nvfp4SwizzleScalesCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array sc = cu::ensure_row_contiguous_aligned<4>(inputs[0], stream(), encoder);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    mlx::core::swizzle_scales(sc, out, encoder, stream());
  }

  DEFINE_NAME(Hy4Nvfp4SwizzleScalesCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Hy4Nvfp4SwizzleScalesCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto [pad_n, pad_k16] =
        mlx::core::get_padded_scale_dims(inputs[0].shape(0), inputs[0].shape(1));
    return {Shape{pad_n, pad_k16}};
  }
};

// ============================================================================
// 2. NVFP4 Dense Few-Row Linear (M = 1..16, K % 64 == 0, arbitrary N >= 1)
// ============================================================================
class Hy4Nvfp4FewCuda final : public UnaryPrimitive {
 public:
  explicit Hy4Nvfp4FewCuda(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4Nvfp4FewCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array scales = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);

    const int K = x.shape(-1);
    const int M = (K > 0) ? static_cast<int>(x.size() / static_cast<size_t>(K)) : 0;
    const int N = w.shape(0);
    if (M < 1 || M > 16 || N < 1 || K < 64 || (K % 64) != 0) {
      throw std::invalid_argument("Hy4Nvfp4FewCuda: unsupported shape");
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    const bool has_mma = encoder.device().compute_capability_major() >= 8;

    // M == 1 (N >= 192): zero-SMEM 2-row/warp streaming uint4 GEMV (2x x reuse & MLP,
    // 2 warps/CTA for K<=1536, 4 warps/CTA for K>1536) saturates DRAM ceiling.
    if (M == 1 && N >= 192) {
      const bool use_2w = (K <= 1536);
      const int rows_per_cta = use_2w ? 4 : 8;
      auto fn = cu::jit_kernel(
          encoder.device(), hy4_nvfp4_few_source,
          use_2w ? "hy4_nvfp4_few_u4_2row_m1<2>" : "hy4_nvfp4_few_u4_2row_m1<4>");
      ensure_smem_carveout(fn, 0);

      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      encoder.set_output_array(out);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      args.append(out);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      const uint32_t grid_x =
          static_cast<uint32_t>((N + rows_per_cta - 1) / rows_per_cta);
      const dim3 block = use_2w ? dim3(32, 2, 1) : dim3(32, 4, 1);
      encoder.add_kernel_node_raw(
          fn, dim3(grid_x, 1, 1), block, {}, 0, args.args());
      return;
    }

    // Pre-Ampere fallback (cc < 8): streaming uint4 GEMV
    if (!has_mma) {
      static constexpr const char* kU4Entries[6] = {
          "hy4_nvfp4_few_u4<1>",
          "hy4_nvfp4_few_u4<2>",
          "hy4_nvfp4_few_u4<3>",
          "hy4_nvfp4_few_u4<4>",
          "hy4_nvfp4_few_u4<8>",
          "hy4_nvfp4_few_u4<16>",
      };
      static constexpr const char* kU4OneRowEntries[6] = {
          "hy4_nvfp4_few_u4_1row<1>",
          "hy4_nvfp4_few_u4_1row<2>",
          "hy4_nvfp4_few_u4_1row<3>",
          "hy4_nvfp4_few_u4_1row<4>",
          "hy4_nvfp4_few_u4_1row<8>",
          "hy4_nvfp4_few_u4_1row<16>",
      };
      const int m_idx = (M <= 4) ? (M - 1) : (M <= 8 ? 4 : 5);
      const bool use_1row = (N < 192);
      auto fn = cu::jit_kernel(
          encoder.device(), hy4_nvfp4_few_source,
          use_1row ? kU4OneRowEntries[m_idx] : kU4Entries[m_idx]);
      ensure_smem_carveout(fn, 0);

      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      encoder.set_output_array(out);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      const uint32_t grid_x = use_1row
          ? static_cast<uint32_t>(N)
          : static_cast<uint32_t>((N + 7) / 8);
      const dim3 block = use_1row ? dim3(32, 4, 1) : dim3(32, 8, 1);
      encoder.add_kernel_node_raw(
          fn, dim3(grid_x, 1, 1), block, {}, 0, args.args());
      return;
    }

    // M in 2..16 (or narrow N < 192) on sm_80+: 2-stage BK=512 cp.async.cg + BF16 TC MMA (+ Split-K)
    static constexpr const char* kMmaEntries[2][2] = {
        {"hy4_nvfp4_few_mma<1, false>", "hy4_nvfp4_few_mma<1, true>"},
        {"hy4_nvfp4_few_mma<2, false>", "hy4_nvfp4_few_mma<2, true>"},
    };
    const int m_tile_idx = (M <= 8) ? 0 : 1;
    const int row_blocks = (N + 15) / 16;
    const int num_tiles = K >> 9;  // full 512-elem tiles
    const int num_sms = device_sm_count(encoder.device().cuda_device());
    int split_k = 1;
    if (row_blocks < num_sms * 2 && num_tiles >= 4) {
      const int target = (num_sms * 2 + row_blocks - 1) / row_blocks;
      split_k = std::min({target, num_tiles, 8});
      if (split_k < 2) {
        split_k = 1;
      }
    }

    if (split_k > 1) {
      array partials = alloc_temp({split_k, M, N}, float32, encoder);
      auto fn = cu::jit_kernel(
          encoder.device(), hy4_nvfp4_few_source, kMmaEntries[m_tile_idx][1]);
      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      encoder.set_output_array(partials);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      args.append(partials);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      encoder.add_kernel_node_raw(
          fn,
          dim3(static_cast<uint32_t>(row_blocks), static_cast<uint32_t>(split_k), 1),
          dim3(32, 8, 1), {}, 0, args.args());

      auto red_fn = cu::jit_kernel(
          encoder.device(), hy4_nvfp4_few_source, "hy4_splitk_reduce_f32_to_bf16");
      encoder.set_input_array(partials);
      encoder.set_output_array(out);
      cu::KernelArgs red_args;
      red_args.append(partials);
      red_args.append(out);
      red_args.append<int32_t>(M);
      red_args.append<int32_t>(N);
      red_args.append<int32_t>(split_k);
      const int total = M * N;
      const int work = ((total & 3) == 0) ? (total >> 2) : total;
      encoder.add_kernel_node_raw(
          red_fn, dim3(static_cast<uint32_t>((work + 63) / 64), 1, 1),
          dim3(64, 1, 1), {}, 0, red_args.args());
      return;
    }

    auto fn = cu::jit_kernel(
        encoder.device(), hy4_nvfp4_few_source, kMmaEntries[m_tile_idx][0]);
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    encoder.set_input_array(scales);
    encoder.set_output_array(out);
    cu::KernelArgs args;
    args.append(x);
    args.append(w);
    args.append(scales);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(row_blocks), 1, 1), dim3(32, 8, 1),
        {}, 0, args.args());
  }

  DEFINE_NAME(Hy4Nvfp4FewCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Hy4Nvfp4FewCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }
};

// ============================================================================
// 3. NVFP4 Dense Prefill Linear (CublasQQMM FP4xFP4 or BM=64 BN=128 BF16 TC)
// ============================================================================
class Hy4Nvfp4PrefillCuda final : public UnaryPrimitive {
 public:
  Hy4Nvfp4PrefillCuda(Stream stream, int mode, bool has_swizzled_scales)
      : UnaryPrimitive(stream),
        mode_(mode),
        has_swizzled_scales_(has_swizzled_scales) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4Nvfp4PrefillCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array scales = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);

    const int K = x.shape(-1);
    const int64_t rows64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    const int N = w.shape(0);
    if (rows64 < 1 || rows64 > INT32_MAX || N < 1 || K < 64 || (K % 64) != 0) {
      throw std::invalid_argument("Hy4Nvfp4PrefillCuda: unsupported shape");
    }
    const int M = static_cast<int>(rows64);

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    const int cc_major = encoder.device().compute_capability_major();

    const bool use_cublas_fp4 =
        (mode_ == 0 && cc_major >= 10 && M >= 64 && N >= 128 &&
         (N % 128) == 0 && (K % 64) == 0) ||
        (mode_ == 2 && cc_major >= 10 && M >= 2 && N >= 16 &&
         (N % 16) == 0 && (K % 64) == 0);

    if (use_cublas_fp4) {
      // Path A: Blackwell native FP4 x FP4 CublasQQMM
      array x_q = alloc_temp({M, K / 8}, uint32, encoder);
      auto [pad_m, pad_k16] = mlx::core::get_padded_scale_dims(M, K / 16);
      array x_s_swizzled = alloc_temp({pad_m, pad_k16}, uint8, encoder);

      auto qfn = cu::jit_kernel(
          encoder.device(), hy4_nvfp4_prefill_source,
          "hy4_nvfp4_quantize_x_swizzled");
      encoder.set_input_array(x);
      encoder.set_output_array(x_q);
      encoder.set_output_array(x_s_swizzled);
      cu::KernelArgs qargs;
      qargs.append(x);
      qargs.append(x_q);
      qargs.append(x_s_swizzled);
      qargs.append<int32_t>(M);
      qargs.append<int32_t>(K);
      encoder.add_kernel_node_raw(
          qfn,
          dim3(static_cast<uint32_t>(K / 64), static_cast<uint32_t>(pad_m / 128), 1),
          dim3(256, 1, 1), {}, 0, qargs.args());

      array w_s_swizzled = has_swizzled_scales_
          ? cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder)
          : mlx::core::pad_and_swizzle_scales(scales, encoder, stream());

      mlx::core::qqmm_impl(
          encoder,
          M,
          N,
          K,
          /*a_transposed=*/false,
          /*lda=*/K,
          /*b_transposed=*/true,
          /*ldb=*/K,
          out,
          x_q,
          w,
          x_s_swizzled,
          w_s_swizzled,
          QuantizationMode::Nvfp4);
      return;
    }

    // Path B: Exact BF16-activation x NVFP4-weight BM=64 BN=128 BK=64 TC MMA
    auto fn = cu::jit_kernel(
        encoder.device(), hy4_nvfp4_prefill_source,
        "hy4_nvfp4_prefill_bf16_bm64");
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    encoder.set_input_array(scales);
    encoder.set_output_array(out);
    cu::KernelArgs args;
    args.append(x);
    args.append(w);
    args.append(scales);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    const uint32_t num_m_tiles = static_cast<uint32_t>((M + 63) / 64);
    const uint32_t num_n_tiles = static_cast<uint32_t>((N + 127) / 128);
    encoder.add_kernel_node_raw(
        fn, dim3(num_m_tiles * num_n_tiles, 1, 1), dim3(32, 8, 1),
        {}, 0, args.args());
  }

  DEFINE_NAME(Hy4Nvfp4PrefillCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4Nvfp4PrefillCuda*>(&other);
    return rhs && rhs->mode_ == mode_ &&
        rhs->has_swizzled_scales_ == has_swizzled_scales_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }

 private:
  int mode_;
  bool has_swizzled_scales_;
};

// ============================================================================
// 4. Fused LM Head (NVFP4 / SymQ8 / BF16) + 2-Stage Zero-DRAM GreedyArgmax
// ============================================================================
class Hy4LmHeadCuda final : public Primitive {
 public:
  Hy4LmHeadCuda(Stream stream, int weight_mode, int out_mode)
      : Primitive(stream), weight_mode_(weight_mode), out_mode_(out_mode) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Hy4LmHeadCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);

    const int K = x.shape(-1);
    const int64_t rows64 = (K > 0) ? (static_cast<int64_t>(x.size()) / K) : 0;
    const int V = w.shape(0);
    if (rows64 < 1 || rows64 > INT32_MAX || V < 1 || K < 64 || (K % 64) != 0) {
      throw std::invalid_argument("Hy4LmHeadCuda: unsupported shape");
    }
    const int M = static_cast<int>(rows64);

    if (out_mode_ == 0) {
      outputs[0].set_data(cu::malloc_async(outputs[0].nbytes(), encoder));
    } else {
      outputs[0].set_data(cu::malloc_async(outputs[0].nbytes(), encoder));
      outputs[1].set_data(cu::malloc_async(outputs[1].nbytes(), encoder));
    }

    const bool has_mma = encoder.device().compute_capability_major() >= 8;
    const int m_tile_idx = (M <= 8) ? 0 : 1;
    const uint32_t m_blocks_16 = static_cast<uint32_t>((M + 15) / 16);

    std::optional<array> partial_vals;
    std::optional<array> partial_idxs;
    int num_ctas = 0;

    if (weight_mode_ == 0) {
      // NVFP4: 2-row/warp streaming uint4 for M==1 logits (out_mode=0, ~280 GB/s),
      // BK=512 cp.async.cg + BF16 TC MMA for out_mode=1 and M >= 2 (~273-278 GB/s).
      array scales = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);
      static const bool force_u4 = [] {
        const char* v = std::getenv("MLX_HY4_LM_HEAD_U4");
        return v && v[0] == '1';
      }();
      if (M == 1 && out_mode_ == 0 && !force_u4) {
        auto fn = cu::jit_kernel(
            encoder.device(), hy4_nvfp4_few_source, "hy4_lm_head_nvfp4_u4_2row_m1");
        ensure_smem_carveout(fn, 0);
        encoder.set_input_array(x);
        encoder.set_input_array(w);
        encoder.set_input_array(scales);
        encoder.set_output_array(outputs[0]);
        cu::KernelArgs args;
        args.append(x);
        args.append(w);
        args.append(scales);
        args.append(outputs[0]);
        args.append<int32_t>(V);
        args.append<int32_t>(K);
        encoder.add_kernel_node_raw(
            fn, dim3(static_cast<uint32_t>((V + 7) / 8), 1, 1),
            dim3(32, 4, 1), {}, 0, args.args());
        return;
      }
      const bool use_u4 = !has_mma || (force_u4 && M <= 2);

      const char* entry = nullptr;
      uint32_t grid_y = m_blocks_16;
      if (use_u4) {
        static constexpr const char* kNvfp4U4[2][2] = {
            {"hy4_lm_head_nvfp4_u4<1, 0>", "hy4_lm_head_nvfp4_u4<1, 1>"},
            {"hy4_lm_head_nvfp4_u4<2, 0>", "hy4_lm_head_nvfp4_u4<2, 1>"},
        };
        const int cap = (M == 1) ? 1 : 2;
        entry = kNvfp4U4[cap == 1 ? 0 : 1][out_mode_];
        grid_y = static_cast<uint32_t>((M + cap - 1) / cap);
      } else {
        static constexpr const char* kNvfp4Mma[2][2] = {
            {"hy4_lm_head_nvfp4_mma<1, 0>", "hy4_lm_head_nvfp4_mma<1, 1>"},
            {"hy4_lm_head_nvfp4_mma<2, 0>", "hy4_lm_head_nvfp4_mma<2, 1>"},
        };
        entry = kNvfp4Mma[m_tile_idx][out_mode_];
      }

      auto fn = cu::jit_kernel(encoder.device(), hy4_nvfp4_few_source, entry);
      if (use_u4) {
        ensure_smem_carveout(fn, 0);
      }
      const int max_tiles = use_u4 ? ((V + 7) / 8) : ((V + 15) / 16);
      num_ctas = (out_mode_ == 1)
          ? persistent_ctas(fn, 256, encoder.device().cuda_device(), max_tiles)
          : max_tiles;
      if (out_mode_ == 1) {
        partial_vals = alloc_temp({M, num_ctas}, float32, encoder);
        partial_idxs = alloc_temp({M, num_ctas}, int32, encoder);
      }

      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      if (out_mode_ == 1) {
        encoder.set_output_array(*partial_vals);
        encoder.set_output_array(*partial_idxs);
      } else {
        encoder.set_output_array(outputs[0]);
      }
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      if (out_mode_ == 1) {
        args.append(*partial_vals);
        args.append(*partial_idxs);
      } else {
        args.append(outputs[0]);
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(M);
      args.append<int32_t>(V);
      args.append<int32_t>(K);
      encoder.add_kernel_node_raw(
          fn, dim3(static_cast<uint32_t>(num_ctas), grid_y, 1),
          dim3(32, 8, 1), {}, 0, args.args());
    } else if (weight_mode_ == 1) {
      // SymQ8: upgraded 16B cp.async.cg MMA hits ~270+ GB/s across M >= 1 (both logits
      // and contiguous-chunk argmax); streaming uint4 remains as pre-Ampere fallback.
      const int sb_dtype = sidecar_dtype_code(inputs[2].dtype());
      if (sb_dtype < 0) {
        throw std::invalid_argument("Hy4LmHeadCuda: unsupported SymQ8 scales dtype");
      }
      array scales = (sb_dtype == 0)
          ? cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);

      const bool use_u4 = !has_mma;
      const char* entry = nullptr;
      uint32_t grid_y = m_blocks_16;
      if (use_u4) {
        static constexpr const char* kSymQ8U4[3][3][2] = {
            {{"hy4_lm_head_symq8_u4<0, 1, 0>", "hy4_lm_head_symq8_u4<0, 1, 1>"},
             {"hy4_lm_head_symq8_u4<0, 2, 0>", "hy4_lm_head_symq8_u4<0, 2, 1>"},
             {"hy4_lm_head_symq8_u4<0, 4, 0>", "hy4_lm_head_symq8_u4<0, 4, 1>"}},
            {{"hy4_lm_head_symq8_u4<1, 1, 0>", "hy4_lm_head_symq8_u4<1, 1, 1>"},
             {"hy4_lm_head_symq8_u4<1, 2, 0>", "hy4_lm_head_symq8_u4<1, 2, 1>"},
             {"hy4_lm_head_symq8_u4<1, 4, 0>", "hy4_lm_head_symq8_u4<1, 4, 1>"}},
            {{"hy4_lm_head_symq8_u4<2, 1, 0>", "hy4_lm_head_symq8_u4<2, 1, 1>"},
             {"hy4_lm_head_symq8_u4<2, 2, 0>", "hy4_lm_head_symq8_u4<2, 2, 1>"},
             {"hy4_lm_head_symq8_u4<2, 4, 0>", "hy4_lm_head_symq8_u4<2, 4, 1>"}},
        };
        const int cap_idx = (M == 1) ? 0 : (M == 2 ? 1 : 2);
        const int cap = (cap_idx == 0) ? 1 : (cap_idx == 1 ? 2 : 4);
        entry = kSymQ8U4[sb_dtype][cap_idx][out_mode_];
        grid_y = static_cast<uint32_t>((M + cap - 1) / cap);
      } else {
        static constexpr const char* kSymQ8Mma[3][2][2] = {
            {{"hy4_lm_head_symq8_mma<0, 1, 0>", "hy4_lm_head_symq8_mma<0, 1, 1>"},
             {"hy4_lm_head_symq8_mma<0, 2, 0>", "hy4_lm_head_symq8_mma<0, 2, 1>"}},
            {{"hy4_lm_head_symq8_mma<1, 1, 0>", "hy4_lm_head_symq8_mma<1, 1, 1>"},
             {"hy4_lm_head_symq8_mma<1, 2, 0>", "hy4_lm_head_symq8_mma<1, 2, 1>"}},
            {{"hy4_lm_head_symq8_mma<2, 1, 0>", "hy4_lm_head_symq8_mma<2, 1, 1>"},
             {"hy4_lm_head_symq8_mma<2, 2, 0>", "hy4_lm_head_symq8_mma<2, 2, 1>"}},
        };
        entry = kSymQ8Mma[sb_dtype][m_tile_idx][out_mode_];
      }

      auto fn = cu::jit_kernel(encoder.device(), hy4_nvfp4_few_source, entry);
      if (use_u4) {
        ensure_smem_carveout(fn, 0);
      }
      const int max_tiles = use_u4 ? ((V + 7) / 8) : ((V + 15) / 16);
      num_ctas = (out_mode_ == 1)
          ? persistent_ctas(fn, 256, encoder.device().cuda_device(), max_tiles)
          : max_tiles;
      if (out_mode_ == 1) {
        partial_vals = alloc_temp({M, num_ctas}, float32, encoder);
        partial_idxs = alloc_temp({M, num_ctas}, int32, encoder);
      }

      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      if (out_mode_ == 1) {
        encoder.set_output_array(*partial_vals);
        encoder.set_output_array(*partial_idxs);
      } else {
        encoder.set_output_array(outputs[0]);
      }
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      if (out_mode_ == 1) {
        args.append(*partial_vals);
        args.append(*partial_idxs);
      } else {
        args.append(outputs[0]);
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(M);
      args.append<int32_t>(V);
      args.append<int32_t>(K);
      encoder.add_kernel_node_raw(
          fn, dim3(static_cast<uint32_t>(num_ctas), grid_y, 1),
          dim3(32, 8, 1), {}, 0, args.args());
    } else {
      // BF16 dense
      static constexpr const char* kBf16Mma[2][2] = {
          {"hy4_lm_head_bf16_mma<1, 0>", "hy4_lm_head_bf16_mma<1, 1>"},
          {"hy4_lm_head_bf16_mma<2, 0>", "hy4_lm_head_bf16_mma<2, 1>"},
      };
      auto fn = cu::jit_kernel(
          encoder.device(), hy4_nvfp4_few_source, kBf16Mma[m_tile_idx][out_mode_]);
      ensure_smem_carveout(fn, (M <= 8) ? 50 : 25);
      const int max_tiles = (V + 15) / 16;
      num_ctas = (out_mode_ == 1)
          ? persistent_ctas(fn, 256, encoder.device().cuda_device(), max_tiles)
          : max_tiles;
      if (out_mode_ == 1) {
        partial_vals = alloc_temp({M, num_ctas}, float32, encoder);
        partial_idxs = alloc_temp({M, num_ctas}, int32, encoder);
      }

      encoder.set_input_array(x);
      encoder.set_input_array(w);
      if (out_mode_ == 1) {
        encoder.set_output_array(*partial_vals);
        encoder.set_output_array(*partial_idxs);
      } else {
        encoder.set_output_array(outputs[0]);
      }
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      if (out_mode_ == 1) {
        args.append(*partial_vals);
        args.append(*partial_idxs);
      } else {
        args.append(outputs[0]);
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(M);
      args.append<int32_t>(V);
      args.append<int32_t>(K);
      encoder.add_kernel_node_raw(
          fn, dim3(static_cast<uint32_t>(num_ctas), m_blocks_16, 1),
          dim3(32, 8, 1), {}, 0, args.args());
    }

    // Stage 2 argmax reduction when out_mode_ == 1:
    // [M, num_ctas] -> (outputs[0]: i32 [M], outputs[1]: f32 [M])
    if (out_mode_ == 1) {
      auto fn2 = cu::jit_kernel(
          encoder.device(), hy4_nvfp4_few_source, "hy4_lm_head_argmax_stage2");
      encoder.set_input_array(*partial_vals);
      encoder.set_input_array(*partial_idxs);
      encoder.set_output_array(outputs[0]);
      encoder.set_output_array(outputs[1]);
      cu::KernelArgs args2;
      args2.append(*partial_vals);
      args2.append(*partial_idxs);
      args2.append(outputs[0]);
      args2.append(outputs[1]);
      args2.append<int32_t>(M);
      args2.append<int32_t>(num_ctas);
      encoder.add_kernel_node_raw(
          fn2, dim3(static_cast<uint32_t>(std::min(M, 65535)), 1, 1),
          dim3(256, 1, 1), {}, 0, args2.args());
    }
  }

  DEFINE_NAME(Hy4LmHeadCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4LmHeadCuda*>(&other);
    return rhs && rhs->weight_mode_ == weight_mode_ && rhs->out_mode_ == out_mode_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int K = inputs[0].shape(-1);
    const int M = (K > 0) ? static_cast<int>(inputs[0].size() / static_cast<size_t>(K)) : 0;
    if (out_mode_ == 1) {
      return {Shape{M}, Shape{M}};
    }
    return {Shape{M, inputs[1].shape(0)}};
  }

 private:
  int weight_mode_;
  int out_mode_;
};

// Zero-copy strided view acceptor for Hy4GroupedQ8Cuda x [M, H, in_dim] (bf16).
// Avoids MLX Copy on last-dim slices (e.g. q_nope = q_raw[:, :, :192]) whenever
// each (m, h) row is unit-stride and 16B-aligned.
array prepare_grouped_q8_x(
    const array& x_in,
    int M,
    int n_heads,
    int in_dim,
    const Stream& stream,
    cu::CommandEncoder& encoder,
    int64_t& x_m_stride_u4,
    int64_t& x_h_stride_u4) {
  if (x_in.ndim() == 3 && x_in.shape(0) == M && x_in.shape(1) == n_heads &&
      x_in.shape(2) == in_dim && x_in.strides(2) == 1 &&
      (reinterpret_cast<uintptr_t>(gpu_ptr<void>(x_in)) & 15u) == 0) {
    const bool ok_h =
        (n_heads == 1) || (x_in.strides(1) >= 0 && (x_in.strides(1) % 8) == 0);
    const bool ok_m =
        (M == 1) || (x_in.strides(0) >= 0 && (x_in.strides(0) % 8) == 0);
    if (ok_h && ok_m) {
      x_h_stride_u4 = (n_heads == 1) ? 0 : (x_in.strides(1) / 8);
      x_m_stride_u4 = (M == 1) ? 0 : (x_in.strides(0) / 8);
      return x_in;
    }
  }
  array x = cu::ensure_row_contiguous_aligned<16>(x_in, stream, encoder);
  x_h_stride_u4 = static_cast<int64_t>(in_dim) / 8;
  x_m_stride_u4 = static_cast<int64_t>(n_heads) * x_h_stride_u4;
  return x;
}

// Zero-copy strided view acceptor for Hy4GroupedQ8Cuda gate (bf16 [M, H*out_dim] or [M, H, out_dim]).
// Avoids MLX Copy on last-dim slices (e.g. gate = q_kv_gate[:, q_lora+kv_row:]).
array prepare_grouped_q8_gate(
    const array& g_in,
    int M,
    int n_heads,
    int out_dim,
    const Stream& stream,
    cu::CommandEncoder& encoder,
    int64_t& gate_m_stride,
    int64_t& gate_h_stride) {
  const bool aligned2 =
      (reinterpret_cast<uintptr_t>(gpu_ptr<void>(g_in)) & 1u) == 0;
  if (aligned2 && g_in.ndim() == 2 && g_in.shape(0) == M &&
      g_in.shape(1) == static_cast<int64_t>(n_heads) * out_dim &&
      g_in.strides(1) == 1 && (M == 1 || g_in.strides(0) >= 0)) {
    gate_m_stride = (M == 1) ? 0 : g_in.strides(0);
    gate_h_stride = out_dim;
    return g_in;
  }
  if (aligned2 && g_in.ndim() == 3 && g_in.shape(0) == M &&
      g_in.shape(1) == n_heads && g_in.shape(2) == out_dim &&
      g_in.strides(2) == 1 && (n_heads == 1 || g_in.strides(1) >= 0) &&
      (M == 1 || g_in.strides(0) >= 0)) {
    gate_m_stride = (M == 1) ? 0 : g_in.strides(0);
    gate_h_stride = (n_heads == 1) ? 0 : g_in.strides(1);
    return g_in;
  }
  array g = cu::ensure_row_contiguous_aligned<2>(g_in, stream, encoder);
  gate_m_stride = static_cast<int64_t>(n_heads) * out_dim;
  gate_h_stride = out_dim;
  return g;
}

// ============================================================================
// 5. Grouped SymQ8 Attention Projection (+ optional fused * sigmoid(gate))
// ============================================================================
class Hy4GroupedQ8Cuda final : public UnaryPrimitive {
 public:
  Hy4GroupedQ8Cuda(
      Stream stream, int n_heads, int in_dim, int out_dim, bool has_gate)
      : UnaryPrimitive(stream),
        n_heads_(n_heads),
        in_dim_(in_dim),
        out_dim_(out_dim),
        has_gate_(has_gate) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4GroupedQ8Cuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    const int64_t denom = static_cast<int64_t>(n_heads_) * in_dim_;
    const int64_t rows64 = (denom > 0) ? (static_cast<int64_t>(inputs[0].size()) / denom) : 0;
    if (rows64 < 1 || rows64 > INT32_MAX || n_heads_ < 1 || out_dim_ < 1 ||
        in_dim_ < 32 || (in_dim_ % 32) != 0) {
      throw std::invalid_argument("Hy4GroupedQ8Cuda: unsupported shape");
    }
    const int M = static_cast<int>(rows64);

    int64_t x_m_stride_u4 = 0;
    int64_t x_h_stride_u4 = 0;
    array x = prepare_grouped_q8_x(
        inputs[0], M, n_heads_, in_dim_, stream(), encoder, x_m_stride_u4, x_h_stride_u4);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array scales = cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
    std::optional<array> gate;
    int64_t gate_m_stride = 0;
    int64_t gate_h_stride = 0;
    if (has_gate_) {
      gate = prepare_grouped_q8_gate(
          inputs[3], M, n_heads_, out_dim_, stream(), encoder, gate_m_stride, gate_h_stride);
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    const bool has_mma = encoder.device().compute_capability_major() >= 8;
    const int gate_idx = has_gate_ ? 1 : 0;

    auto bind_and_build_args = [&]() {
      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      if (has_gate_) {
        encoder.set_input_array(*gate);
      }
      encoder.set_output_array(out);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      if (has_gate_) {
        args.append(*gate);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(n_heads_);
      args.append<int32_t>(in_dim_);
      args.append<int32_t>(out_dim_);
      args.append<int64_t>(x_m_stride_u4);
      args.append<int64_t>(x_h_stride_u4);
      args.append<int64_t>(gate_m_stride);
      args.append<int64_t>(gate_h_stride);
      return args;
    };

    // Streaming 8-warp uint4 (2x row-pipelined, coalesced epilogue) hits ~260-269 GB/s
    // at M=1..2 and serves as the pre-Ampere fallback; on sm_80+ for M=3..16, BF16 TC MMA
    // beats scalar FMA (especially on k_b M=4 and multi-token verify).
    if (M <= 2 || !has_mma) {
      static constexpr const char* kFewU4[3][2][2][2] = {
          {
              {{"hy4_grouped_q8_few_u4<1, false, false, false>",
                "hy4_grouped_q8_few_u4<1, false, false, true>"},
               {"hy4_grouped_q8_few_u4<1, false, true, false>",
                "hy4_grouped_q8_few_u4<1, false, true, true>"}},
              {{"hy4_grouped_q8_few_u4<1, true, false, false>",
                "hy4_grouped_q8_few_u4<1, true, false, true>"},
               {"hy4_grouped_q8_few_u4<1, true, true, false>",
                "hy4_grouped_q8_few_u4<1, true, true, true>"}},
          },
          {
              {{"hy4_grouped_q8_few_u4<2, false, false, false>",
                "hy4_grouped_q8_few_u4<2, false, false, true>"},
               {"hy4_grouped_q8_few_u4<2, false, true, false>",
                "hy4_grouped_q8_few_u4<2, false, true, true>"}},
              {{"hy4_grouped_q8_few_u4<2, true, false, false>",
                "hy4_grouped_q8_few_u4<2, true, false, true>"},
               {"hy4_grouped_q8_few_u4<2, true, true, false>",
                "hy4_grouped_q8_few_u4<2, true, true, true>"}},
          },
          {
              {{"hy4_grouped_q8_few_u4<4, false, false, false>",
                "hy4_grouped_q8_few_u4<4, false, false, true>"},
               {"hy4_grouped_q8_few_u4<4, false, true, false>",
                "hy4_grouped_q8_few_u4<4, false, true, true>"}},
              {{"hy4_grouped_q8_few_u4<4, true, false, false>",
                "hy4_grouped_q8_few_u4<4, true, false, true>"},
               {"hy4_grouped_q8_few_u4<4, true, true, false>",
                "hy4_grouped_q8_few_u4<4, true, true, true>"}},
          },
      };
      const int m_idx = (M == 1) ? 0 : (M == 2 ? 1 : 2);
      const int cap = (m_idx == 0) ? 1 : (m_idx == 1 ? 2 : 4);
      const int half_warp = (in_dim_ <= 256) ? 1 : 0;
      const int use_smem_x = (in_dim_ <= 512) ? 1 : 0;
      auto fn = cu::jit_kernel(
          encoder.device(), hy4_grouped_q8_source,
          kFewU4[m_idx][half_warp][use_smem_x][gate_idx]);
      auto args = bind_and_build_args();
      encoder.add_kernel_node_raw(
          fn,
          dim3(
              static_cast<uint32_t>((out_dim_ + 31) / 32),
              static_cast<uint32_t>(n_heads_),
              static_cast<uint32_t>((M + cap - 1) / cap)),
          dim3(32, 8, 1), {}, 0, args.args());
      return;
    }

    if (M <= 16) {
      static constexpr const char* kFewMma[2][2] = {
          {"hy4_grouped_q8_few_mma<1, false>", "hy4_grouped_q8_few_mma<1, true>"},
          {"hy4_grouped_q8_few_mma<2, false>", "hy4_grouped_q8_few_mma<2, true>"},
      };
      auto fn = cu::jit_kernel(
          encoder.device(), hy4_grouped_q8_source,
          kFewMma[M <= 8 ? 0 : 1][gate_idx]);
      auto args = bind_and_build_args();
      encoder.add_kernel_node_raw(
          fn,
          dim3(
              static_cast<uint32_t>((out_dim_ + 15) / 16),
              static_cast<uint32_t>(n_heads_), 1),
          dim3(32, 4, 1), {}, 0, args.args());
      return;
    }

    // M >= 17: BM=64, BN=64, BK=64 (or BK=32) double-buffered INT8 TC MMA
    static constexpr const char* kPrefill[2][2] = {
        {"hy4_grouped_q8_prefill_int8<32, false>",
         "hy4_grouped_q8_prefill_int8<32, true>"},
        {"hy4_grouped_q8_prefill_int8<64, false>",
         "hy4_grouped_q8_prefill_int8<64, true>"},
    };
    const int bk64 = ((in_dim_ % 64) == 0) ? 1 : 0;
    auto fn = cu::jit_kernel(
        encoder.device(), hy4_grouped_q8_source, kPrefill[bk64][gate_idx]);
    auto args = bind_and_build_args();
    const uint32_t num_m_tiles = static_cast<uint32_t>((M + 63) / 64);
    const uint32_t num_n_tiles = static_cast<uint32_t>((out_dim_ + 63) / 64);
    encoder.add_kernel_node_raw(
        fn,
        dim3(num_m_tiles * num_n_tiles, static_cast<uint32_t>(n_heads_), 1),
        dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Hy4GroupedQ8Cuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4GroupedQ8Cuda*>(&other);
    return rhs && rhs->n_heads_ == n_heads_ && rhs->in_dim_ == in_dim_ &&
        rhs->out_dim_ == out_dim_ && rhs->has_gate_ == has_gate_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = out_dim_;
    return {std::move(shape)};
  }

 private:
  int n_heads_;
  int in_dim_;
  int out_dim_;
  bool has_gate_;
};

}  // namespace
}  // namespace mlx::core::hy4_nvfp4_cuda

// ============================================================================
// Exported C FFI Entry Points
// ============================================================================

extern "C" mlx_array* mlx_hy4_nvfp4_swizzle_scales_cuda(mlx_array* scales) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_nvfp4_cuda;
    if (!scales || default_device() != Device::gpu) {
      return nullptr;
    }
    const auto& sc = *reinterpret_cast<array*>(scales);
    if (sc.ndim() != 2 || sc.dtype() != uint8 ||
        sc.shape(0) < 1 || sc.shape(1) < 1) {
      return nullptr;
    }
    auto [pad_n, pad_k16] = get_padded_scale_dims(sc.shape(0), sc.shape(1));
    auto result = array(
        Shape{pad_n, pad_k16}, uint8,
        std::make_shared<Hy4Nvfp4SwizzleScalesCuda>(default_stream(Device::gpu)),
        {sc});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_nvfp4_swizzle_scales_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_hy4_nvfp4_swizzle_scales_cuda: unknown exception\n");
    return nullptr;
  }
}

extern "C" mlx_array* mlx_hy4_nvfp4_few_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_nvfp4_cuda;
    if (!x || !w || !scales || default_device() != Device::gpu) {
      return nullptr;
    }
    const auto& xa = *reinterpret_cast<array*>(x);
    const auto& wa = *reinterpret_cast<array*>(w);
    const auto& sa = *reinterpret_cast<array*>(scales);
    if (xa.dtype() != bfloat16 || xa.ndim() < 2 ||
        wa.dtype() != uint32 || wa.ndim() != 2 ||
        sa.dtype() != uint8 || sa.ndim() != 2) {
      return nullptr;
    }
    const int K = xa.shape(-1);
    const int N = wa.shape(0);
    if (K < 64 || (K % 64) != 0 || N < 1 ||
        wa.shape(1) != K / 8 ||
        sa.shape(0) != N || sa.shape(1) != K / 16) {
      return nullptr;
    }
    const int64_t rows = static_cast<int64_t>(xa.size()) / K;
    if (rows < 1 || rows > 16) {
      return nullptr;
    }
    auto out_shape = xa.shape();
    out_shape.back() = N;
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Hy4Nvfp4FewCuda>(default_stream(Device::gpu)),
        {xa, wa, sa});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_nvfp4_few_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_hy4_nvfp4_few_cuda: unknown exception\n");
    return nullptr;
  }
}

extern "C" mlx_array* mlx_hy4_nvfp4_prefill_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* swizzled_scales_opt,
    int32_t mode) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_nvfp4_cuda;
    if (!x || !w || !scales || mode < 0 || mode > 2 ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& xa = *reinterpret_cast<array*>(x);
    const auto& wa = *reinterpret_cast<array*>(w);
    const auto& sa = *reinterpret_cast<array*>(scales);
    if (xa.dtype() != bfloat16 || xa.ndim() < 2 ||
        wa.dtype() != uint32 || wa.ndim() != 2 ||
        sa.dtype() != uint8 || sa.ndim() != 2) {
      return nullptr;
    }
    const int K = xa.shape(-1);
    const int N = wa.shape(0);
    if (K < 64 || (K % 64) != 0 || N < 1 ||
        wa.shape(1) != K / 8 ||
        sa.shape(0) != N || sa.shape(1) != K / 16) {
      return nullptr;
    }
    const int64_t rows = static_cast<int64_t>(xa.size()) / K;
    if (rows < 1 || rows > INT32_MAX) {
      return nullptr;
    }
    std::vector<array> inputs;
    inputs.reserve(swizzled_scales_opt ? 4u : 3u);
    inputs.push_back(xa);
    inputs.push_back(wa);
    inputs.push_back(sa);
    const bool has_swz = (swizzled_scales_opt != nullptr);
    if (has_swz) {
      const auto& swza = *reinterpret_cast<array*>(swizzled_scales_opt);
      auto [pad_n, pad_k16] = get_padded_scale_dims(N, K / 16);
      if (swza.dtype() != uint8 || swza.ndim() != 2 ||
          swza.shape(0) != pad_n || swza.shape(1) != pad_k16) {
        return nullptr;
      }
      inputs.push_back(swza);
    }
    auto out_shape = xa.shape();
    out_shape.back() = N;
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Hy4Nvfp4PrefillCuda>(
            default_stream(Device::gpu), mode, has_swz),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_nvfp4_prefill_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_hy4_nvfp4_prefill_cuda: unknown exception\n");
    return nullptr;
  }
}

extern "C" bool mlx_hy4_lm_head_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales_opt,
    int32_t weight_mode,
    int32_t out_mode,
    mlx_array** out_a,
    mlx_array** out_b) {
  if (out_a) {
    *out_a = nullptr;
  }
  if (out_b) {
    *out_b = nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_nvfp4_cuda;
    if (!x || !w || !out_a || (out_mode == 1 && !out_b) ||
        weight_mode < 0 || weight_mode > 2 ||
        out_mode < 0 || out_mode > 1 ||
        default_device() != Device::gpu) {
      return false;
    }
    const auto& xa = *reinterpret_cast<array*>(x);
    const auto& wa = *reinterpret_cast<array*>(w);
    if (xa.dtype() != bfloat16 || xa.ndim() < 2 || wa.ndim() != 2) {
      return false;
    }
    const int K = xa.shape(-1);
    const int V = wa.shape(0);
    if (K < 64 || (K % 64) != 0 || V < 1) {
      return false;
    }
    const int64_t rows64 = static_cast<int64_t>(xa.size()) / K;
    if (rows64 < 1 || rows64 > INT32_MAX) {
      return false;
    }
    const int M = static_cast<int>(rows64);

    if (weight_mode == 2 && cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }

    std::vector<array> inputs;
    inputs.reserve(3);
    inputs.push_back(xa);
    inputs.push_back(wa);

    if (weight_mode == 0) {
      if (!scales_opt) {
        return false;
      }
      const auto& sa = *reinterpret_cast<array*>(scales_opt);
      if (wa.dtype() != uint32 || wa.shape(1) != K / 8 ||
          sa.dtype() != uint8 || sa.ndim() != 2 ||
          sa.shape(0) != V || sa.shape(1) != K / 16) {
        return false;
      }
      inputs.push_back(sa);
    } else if (weight_mode == 1) {
      if (!scales_opt) {
        return false;
      }
      const auto& sa = *reinterpret_cast<array*>(scales_opt);
      if (wa.dtype() != uint32 || wa.shape(1) != K / 4 ||
          sidecar_dtype_code(sa.dtype()) < 0 || sa.ndim() != 2 ||
          sa.shape(0) != V || sa.shape(1) != K / 32) {
        return false;
      }
      inputs.push_back(sa);
    } else {
      if (scales_opt != nullptr || wa.dtype() != bfloat16 || wa.shape(1) != K) {
        return false;
      }
    }

    if (out_mode == 0) {
      auto outputs = array::make_arrays(
          {Shape{M, V}}, {float32},
          std::make_shared<Hy4LmHeadCuda>(
              default_stream(Device::gpu), weight_mode, out_mode),
          std::move(inputs));
      *out_a = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
      return true;
    }

    auto outputs = array::make_arrays(
        {Shape{M}, Shape{M}}, {int32, float32},
        std::make_shared<Hy4LmHeadCuda>(
            default_stream(Device::gpu), weight_mode, out_mode),
        std::move(inputs));
    *out_a = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_b = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_lm_head_cuda: %s\n", e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "mlx_hy4_lm_head_cuda: unknown exception\n");
    return false;
  }
}

extern "C" mlx_array* mlx_hy4_grouped_q8_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* gate_opt,
    int32_t n_heads,
    int32_t in_dim,
    int32_t out_dim) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_nvfp4_cuda;
    if (!x || !w || !scales || n_heads < 1 || n_heads > 65535 ||
        in_dim < 32 || (in_dim % 32) != 0 || out_dim < 1 ||
        default_device() != Device::gpu) {
      return nullptr;
    }
    const auto& xa = *reinterpret_cast<array*>(x);
    const auto& wa = *reinterpret_cast<array*>(w);
    const auto& sa = *reinterpret_cast<array*>(scales);
    if (xa.dtype() != bfloat16 || xa.ndim() < 3 ||
        xa.shape(-2) != n_heads || xa.shape(-1) != in_dim) {
      return nullptr;
    }
    const int64_t denom = static_cast<int64_t>(n_heads) * in_dim;
    const int64_t rows64 = static_cast<int64_t>(xa.size()) / denom;
    if (rows64 < 1 || rows64 > INT32_MAX) {
      return nullptr;
    }
    const bool w_2d = (wa.ndim() == 2 &&
                       wa.shape(0) == static_cast<int64_t>(n_heads) * out_dim &&
                       wa.shape(1) == in_dim / 4);
    const bool w_3d = (wa.ndim() == 3 && wa.shape(0) == n_heads &&
                       wa.shape(1) == out_dim && wa.shape(2) == in_dim / 4);
    const bool s_2d = (sa.ndim() == 2 &&
                       sa.shape(0) == static_cast<int64_t>(n_heads) * out_dim &&
                       sa.shape(1) == in_dim / 32);
    const bool s_3d = (sa.ndim() == 3 && sa.shape(0) == n_heads &&
                       sa.shape(1) == out_dim && sa.shape(2) == in_dim / 32);
    if (wa.dtype() != uint32 || (!w_2d && !w_3d) ||
        sa.dtype() != float16 || (!s_2d && !s_3d)) {
      return nullptr;
    }

    const bool has_gate = (gate_opt != nullptr);
    std::vector<array> inputs;
    inputs.reserve(has_gate ? 4u : 3u);
    inputs.push_back(xa);
    inputs.push_back(wa);
    inputs.push_back(sa);
    if (has_gate) {
      const auto& ga = *reinterpret_cast<array*>(gate_opt);
      const int64_t expected_elems =
          rows64 * static_cast<int64_t>(n_heads) * out_dim;
      if (ga.dtype() != bfloat16 ||
          static_cast<int64_t>(ga.size()) != expected_elems) {
        return nullptr;
      }
      inputs.push_back(ga);
    }

    auto out_shape = xa.shape();
    out_shape.back() = out_dim;
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Hy4GroupedQ8Cuda>(
            default_stream(Device::gpu), n_heads, in_dim, out_dim, has_gate),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_grouped_q8_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_hy4_grouped_q8_cuda: unknown exception\n");
    return nullptr;
  }
}

#else

// ============================================================================
// Non-CUDA Stubs (macOS / CPU-only builds)
// ============================================================================

extern "C" mlx_array* mlx_hy4_nvfp4_swizzle_scales_cuda(mlx_array* scales) {
  (void)scales;
  return nullptr;
}

extern "C" mlx_array* mlx_hy4_nvfp4_few_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales) {
  (void)x;
  (void)w;
  (void)scales;
  return nullptr;
}

extern "C" mlx_array* mlx_hy4_nvfp4_prefill_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* swizzled_scales_opt,
    int32_t mode) {
  (void)x;
  (void)w;
  (void)scales;
  (void)swizzled_scales_opt;
  (void)mode;
  return nullptr;
}

extern "C" bool mlx_hy4_lm_head_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales_opt,
    int32_t weight_mode,
    int32_t out_mode,
    mlx_array** out_a,
    mlx_array** out_b) {
  (void)x;
  (void)w;
  (void)scales_opt;
  (void)weight_mode;
  (void)out_mode;
  if (out_a) *out_a = nullptr;
  if (out_b) *out_b = nullptr;
  return false;
}

extern "C" mlx_array* mlx_hy4_grouped_q8_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* gate_opt,
    int32_t n_heads,
    int32_t in_dim,
    int32_t out_dim) {
  (void)x;
  (void)w;
  (void)scales;
  (void)gate_opt;
  (void)n_heads;
  (void)in_dim;
  (void)out_dim;
  return nullptr;
}

#endif  // MLX_NODE_CUDA_ENABLED
