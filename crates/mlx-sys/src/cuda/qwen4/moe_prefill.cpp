#include "cuda/qwen4/moe_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace mlx::core::qwen4_moe {
namespace {

using namespace mlx_node::qwen4_cuda;

constexpr char qwen4_moe_prefill_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/qwen4_moe_prefill.cu.inc"
    ;

class Qwen4MoePrefillCuda final : public UnaryPrimitive {
 public:
  Qwen4MoePrefillCuda(
      Stream stream, int mode_gu, int bits_dn, int num_experts, int top_k,
      bool add_shared)
      : UnaryPrimitive(stream),
        mode_gu_(mode_gu),
        bits_dn_(bits_dn),
        num_experts_(num_experts),
        top_k_(top_k),
        add_shared_(add_shared) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Qwen4MoePrefillCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    constexpr int kBmHot = 64;
    constexpr int kColdMax = 32;
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array ids = cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder);
    array scores = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);
    array wg = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    array sg = cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
    array bg = cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder);
    array wu = cu::ensure_row_contiguous_aligned<16>(inputs[6], stream(), encoder);
    array su = cu::ensure_row_contiguous_aligned<16>(inputs[7], stream(), encoder);
    array bu = cu::ensure_row_contiguous_aligned<16>(inputs[8], stream(), encoder);
    array wd = cu::ensure_row_contiguous_aligned<16>(inputs[9], stream(), encoder);
    array sd = cu::ensure_row_contiguous_aligned<16>(inputs[10], stream(), encoder);
    array bd = cu::ensure_row_contiguous_aligned<16>(inputs[11], stream(), encoder);

    const int M = static_cast<int>(x.size() / kH);
    const int P = M * top_k_;
    const int max_tiles = (P + kBmHot - 1) / kBmHot + num_experts_;
    if (M < 1 || M > 65535 || top_k_ < 1 || top_k_ > 16 ||
        max_tiles > 65535 || num_experts_ < 1 || num_experts_ > kMaxExperts) {
      throw std::invalid_argument("Qwen4MoePrefillCuda: unsupported replay shape");
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    auto& ws = get_prefill_workspace(
        encoder.device().cuda_device(), stream().index);
    ws.ensure(M, P, max_tiles, encoder);
    const array& sorted_pairs = *ws.sorted_pairs;
    const array& inv_perm = *ws.inv_perm;
    const array& tile_expert = *ws.tile_expert;
    const array& tile_begin = *ws.tile_begin;
    const array& tile_end = *ws.tile_end;
    const array& num_tiles = *ws.num_tiles;
    const array& xq = *ws.xq;
    const array& x_scales = *ws.x_scales;
    const array& x_qsums = *ws.x_qsums;
    const array& mid_q = *ws.mid_q;
    const array& mid_sc = *ws.mid_sc;
    const array& mid_qsum = *ws.mid_qsum;
    const array& down_out = *ws.down_out;

    // 1. Counting-sort routes + build split hot (BM=64) / cold (BM=32) tile table
    {
      encoder.set_input_array(ids);
      encoder.set_output_array(sorted_pairs);
      encoder.set_output_array(inv_perm);
      encoder.set_output_array(tile_expert);
      encoder.set_output_array(tile_begin);
      encoder.set_output_array(tile_end);
      encoder.set_output_array(num_tiles);
      auto k_sort = cu::jit_kernel(
          encoder.device(), qwen4_moe_prefill_source,
          "qwen4_moe_prefill_route_sort_v2");
      cu::KernelArgs args;
      args.append(ids);
      args.append(sorted_pairs);
      args.append(inv_perm);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append<int32_t>(P);
      args.append<int32_t>(num_experts_);
      args.append<int32_t>(kBmHot);
      args.append<int32_t>(kColdMax);
      encoder.add_kernel_node_raw(
          k_sort, dim3(1, 1, 1), dim3(512, 1, 1), {}, 0, args.args());
    }

    // 2. Super-block (GS=256) per-token INT8 quantizer
    {
      encoder.set_input_array(x);
      encoder.set_output_array(xq);
      encoder.set_output_array(x_scales);
      encoder.set_output_array(x_qsums);
      auto k_quant = cu::jit_kernel(
          encoder.device(), qwen4_moe_prefill_source,
          "qwen4_moe_prefill_quantize_x");
      cu::KernelArgs args;
      args.append(x);
      args.append(xq);
      args.append(x_scales);
      args.append(x_qsums);
      args.append<int32_t>(M);
      args.append<int32_t>(kH);
      const uint32_t grid_x =
          static_cast<uint32_t>((M * (kH / 256) + 7) / 8);
      encoder.add_kernel_node_raw(
          k_quant, dim3(grid_x, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
    }

    // 3A. Grouped prefill gate + up + SwiGLU -- hot tiles [0, num_hot) (BM=64)
    {
      encoder.set_input_array(wg);
      encoder.set_input_array(sg);
      encoder.set_input_array(bg);
      encoder.set_input_array(wu);
      encoder.set_input_array(su);
      encoder.set_input_array(bu);
      encoder.set_input_array(xq);
      encoder.set_input_array(x_scales);
      encoder.set_input_array(x_qsums);
      encoder.set_input_array(sorted_pairs);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(mid_q);
      encoder.set_output_array(mid_sc);
      encoder.set_output_array(mid_qsum);
      auto k_gu_hot = cu::jit_kernel(
          encoder.device(), qwen4_moe_prefill_source,
          mode_gu_ == 0 ? "qwen4_moe_prefill_gate_up_swiglu_bm64<0, true>"
                        : "qwen4_moe_prefill_gate_up_swiglu_bm64<1, true>");
      cu::KernelArgs args;
      args.append(wg);
      args.append(sg);
      args.append(bg);
      args.append(wu);
      args.append(su);
      args.append(bu);
      args.append(xq);
      args.append(x_scales);
      args.append(x_qsums);
      args.append(sorted_pairs);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(mid_q);
      args.append(mid_sc);
      args.append(mid_qsum);
      args.append<int32_t>(kDff);
      args.append<int32_t>(kH);
      args.append<int32_t>(top_k_);
      encoder.add_kernel_node_raw(
          k_gu_hot, dim3(kDff / 64, max_tiles, 1), dim3(32, 4, 1), {}, 0,
          args.args());
    }

    // 3B. Grouped prefill gate + up + SwiGLU -- cold tiles [num_hot, total) (BM=32)
    {
      encoder.set_input_array(wg);
      encoder.set_input_array(sg);
      encoder.set_input_array(bg);
      encoder.set_input_array(wu);
      encoder.set_input_array(su);
      encoder.set_input_array(bu);
      encoder.set_input_array(xq);
      encoder.set_input_array(x_scales);
      encoder.set_input_array(x_qsums);
      encoder.set_input_array(sorted_pairs);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(mid_q);
      encoder.set_output_array(mid_sc);
      encoder.set_output_array(mid_qsum);
      auto k_gu_cold = cu::jit_kernel(
          encoder.device(), qwen4_moe_prefill_source,
          mode_gu_ == 0 ? "qwen4_moe_prefill_gate_up_swiglu_bm32<0, true>"
                        : "qwen4_moe_prefill_gate_up_swiglu_bm32<1, true>");
      cu::KernelArgs args;
      args.append(wg);
      args.append(sg);
      args.append(bg);
      args.append(wu);
      args.append(su);
      args.append(bu);
      args.append(xq);
      args.append(x_scales);
      args.append(x_qsums);
      args.append(sorted_pairs);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(mid_q);
      args.append(mid_sc);
      args.append(mid_qsum);
      args.append<int32_t>(kDff);
      args.append<int32_t>(kH);
      args.append<int32_t>(top_k_);
      encoder.add_kernel_node_raw(
          k_gu_cold, dim3(kDff / 64, num_experts_, 1), dim3(32, 4, 1), {}, 0,
          args.args());
    }

    // 4. Grouped prefill INT8 Tensor-Core down projection (BM=64 Q5_1 or Q8_0, BK=128)
    {
      encoder.set_input_array(wd);
      encoder.set_input_array(sd);
      encoder.set_input_array(bd);
      encoder.set_input_array(mid_q);
      encoder.set_input_array(mid_sc);
      encoder.set_input_array(mid_qsum);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(down_out);
      const char* dn_entry = (bits_dn_ == 5)
          ? "qwen4_moe_prefill_down_bm64_i8<5>"
          : "qwen4_moe_prefill_down_bm64_i8<8>";
      const dim3 dn_block(32, 4, 1);
      auto k_dn = cu::jit_kernel(
          encoder.device(), qwen4_moe_prefill_source, dn_entry);
      cu::KernelArgs args;
      args.append(wd);
      args.append(sd);
      args.append(bd);
      args.append(mid_q);
      args.append(mid_sc);
      args.append(mid_qsum);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(down_out);
      args.append<int32_t>(kH);
      args.append<int32_t>(kDff);
      encoder.add_kernel_node_raw(
          k_dn, dim3(kH / 64, max_tiles, 1), dn_block, {}, 0, args.args());
    }

    // 5. Deterministic vectorized sorted combine v2 (+ optional shared add)
    {
      encoder.set_input_array(down_out);
      encoder.set_input_array(scores);
      encoder.set_input_array(inv_perm);
      cu::KernelArgs args;
      args.append(down_out);
      args.append(scores);
      args.append(inv_perm);
      if (add_shared_) {
        array shared_out =
            cu::ensure_row_contiguous_aligned<16>(inputs[12], stream(), encoder);
        array shared_gate =
            cu::ensure_row_contiguous_aligned<2>(inputs[13], stream(), encoder);
        encoder.set_input_array(shared_out);
        encoder.set_input_array(shared_gate);
        args.append(shared_out);
        args.append(shared_gate);
      } else {
        args.append<CUdeviceptr>(0);
        args.append<CUdeviceptr>(0);
      }
      encoder.set_output_array(out);
      // Mark all persistent workspace buffers as outputs of the final combine
      // kernel so MLX's CUDA Graph dependency tracker enforces a strict
      // happens-before edge before any subsequent Qwen4MoePrefillCuda reuses
      // the same workspace within the same graph.
      encoder.set_output_array(sorted_pairs);
      encoder.set_output_array(inv_perm);
      encoder.set_output_array(tile_expert);
      encoder.set_output_array(tile_begin);
      encoder.set_output_array(tile_end);
      encoder.set_output_array(num_tiles);
      encoder.set_output_array(xq);
      encoder.set_output_array(x_scales);
      encoder.set_output_array(x_qsums);
      encoder.set_output_array(mid_q);
      encoder.set_output_array(mid_sc);
      encoder.set_output_array(mid_qsum);
      encoder.set_output_array(down_out);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(kH);
      args.append<int32_t>(top_k_);
      auto k_comb = cu::jit_kernel(
          encoder.device(), qwen4_moe_prefill_source,
          add_shared_ ? "qwen4_moe_prefill_combine_v2<true>"
                      : "qwen4_moe_prefill_combine_v2<false>");
      encoder.add_kernel_node_raw(
          k_comb, dim3((kH / 8 + 319) / 320, M, 1), dim3(320, 1, 1), {}, 0,
          args.args());
    }
  }

  DEFINE_NAME(Qwen4MoePrefillCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4MoePrefillCuda*>(&other);
    return rhs && rhs->mode_gu_ == mode_gu_ && rhs->bits_dn_ == bits_dn_ &&
        rhs->num_experts_ == num_experts_ && rhs->top_k_ == top_k_ &&
        rhs->add_shared_ == add_shared_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int mode_gu_;
  int bits_dn_;
  int num_experts_;
  int top_k_;
  bool add_shared_;
};

}  // namespace
}  // namespace mlx::core::qwen4_moe

extern "C" mlx_array* mlx_qwen4_moe_prefill_cuda(
    mlx_array* x_h, mlx_array* ids_h, mlx_array* scores_h,
    mlx_array* wg_h, mlx_array* sg_h, mlx_array* bg_h,
    mlx_array* wu_h, mlx_array* su_h, mlx_array* bu_h,
    mlx_array* wd_h, mlx_array* sd_h, mlx_array* bd_h,
    int32_t mode_gu, int32_t bits_dn,
    mlx_array* shared_out_h, mlx_array* shared_gate_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_moe;
    using namespace mlx_node::qwen4_cuda;
    if (!x_h || !ids_h || !scores_h || !wg_h || !sg_h || !bg_h || !wu_h ||
        !su_h || !bu_h || !wd_h || !sd_h || !bd_h ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    if ((shared_out_h != nullptr) != (shared_gate_h != nullptr)) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& ids = *reinterpret_cast<array*>(ids_h);
    const auto& scores = *reinterpret_cast<array*>(scores_h);
    const auto& wg = *reinterpret_cast<array*>(wg_h);
    const auto& sg = *reinterpret_cast<array*>(sg_h);
    const auto& bg = *reinterpret_cast<array*>(bg_h);
    const auto& wu = *reinterpret_cast<array*>(wu_h);
    const auto& su = *reinterpret_cast<array*>(su_h);
    const auto& bu = *reinterpret_cast<array*>(bu_h);
    const auto& wd = *reinterpret_cast<array*>(wd_h);
    const auto& sd = *reinterpret_cast<array*>(sd_h);
    const auto& bd = *reinterpret_cast<array*>(bd_h);

    int M = 0, E = 0, top_k = 0;
    if (!validate_routed_inputs(
            x, ids, scores, wg, sg, bg, wu, su, bu, wd, sd, bd,
            mode_gu, bits_dn, M, E, top_k)) {
      return nullptr;
    }
    if (top_k > 16) {
      return nullptr;
    }
    const int64_t p64 = static_cast<int64_t>(M) * top_k;
    const int64_t max_tiles = (p64 + 63) / 64 + E;
    if (p64 > 131072 || max_tiles > 65535) {
      return nullptr;
    }

    const bool add_shared = (shared_out_h != nullptr);
    std::vector<array> inputs;
    inputs.reserve(add_shared ? 14 : 12);
    inputs.insert(inputs.end(), {x, ids, scores, wg, sg, bg, wu, su, bu, wd, sd, bd});

    if (add_shared) {
      const auto& sout = *reinterpret_cast<array*>(shared_out_h);
      const auto& sgate = *reinterpret_cast<array*>(shared_gate_h);
      if (sout.dtype() != bfloat16 ||
          sout.size() != static_cast<size_t>(M) * kH ||
          sgate.dtype() != bfloat16 ||
          sgate.size() != static_cast<size_t>(M)) {
        return nullptr;
      }
      inputs.push_back(sout);
      inputs.push_back(sgate);
    }

    auto result = array(
        x.shape(), bfloat16,
        std::make_shared<Qwen4MoePrefillCuda>(
            default_stream(Device::gpu), mode_gu, bits_dn, E, top_k,
            add_shared),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_moe_prefill_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" mlx_array* mlx_qwen4_moe_prefill_cuda(
    mlx_array* x_h, mlx_array* ids_h, mlx_array* scores_h,
    mlx_array* wg_h, mlx_array* sg_h, mlx_array* bg_h,
    mlx_array* wu_h, mlx_array* su_h, mlx_array* bu_h,
    mlx_array* wd_h, mlx_array* sd_h, mlx_array* bd_h,
    int32_t mode_gu, int32_t bits_dn,
    mlx_array* shared_out_h, mlx_array* shared_gate_h) {
  (void)x_h; (void)ids_h; (void)scores_h;
  (void)wg_h; (void)sg_h; (void)bg_h;
  (void)wu_h; (void)su_h; (void)bu_h;
  (void)wd_h; (void)sd_h; (void)bd_h;
  (void)mode_gu; (void)bits_dn;
  (void)shared_out_h; (void)shared_gate_h;
  return nullptr;
}

#endif
