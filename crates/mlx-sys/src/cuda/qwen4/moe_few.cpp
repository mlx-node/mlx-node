#include "cuda/qwen4/moe_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mlx::core::qwen4_moe {
namespace {

using namespace mlx_node::qwen4_cuda;

constexpr char qwen4_moe_few_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/qwen4_moe_few.cu.inc"
    ;

class Qwen4MoeFewCuda final : public UnaryPrimitive {
 public:
  Qwen4MoeFewCuda(
      Stream stream, int mode_gu, int bits_dn, int num_experts, int top_k,
      bool has_shared)
      : UnaryPrimitive(stream),
        mode_gu_(mode_gu),
        bits_dn_(bits_dn),
        num_experts_(num_experts),
        top_k_(top_k),
        has_shared_(has_shared) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Qwen4MoeFewCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
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
    if (M < 1 || M > 16 || P > 256 || num_experts_ < 1 ||
        num_experts_ > kMaxExperts) {
      throw std::invalid_argument("Qwen4MoeFewCuda: unsupported replay shape");
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    array xq = alloc_temp({M * kH}, uint8, encoder);
    array x_scales = alloc_temp({M * (kH / 32)}, float32, encoder);
    array x_sums = alloc_temp({M * (kH / 32)}, float32, encoder);
    array mid = alloc_temp({P, kDff}, bfloat16, encoder);
    array down_pairs = alloc_temp({P, kH}, bfloat16, encoder);

    const bool m1_direct_q4k = (M == 1 && mode_gu_ == 0);
    // Only allocate grouping buffers when a grouped kernel runs.
    std::optional<array> unique_experts;
    std::optional<array> bucket_offsets;
    std::optional<array> pair_indices;
    std::optional<array> num_unique;
    if (!m1_direct_q4k) {
      unique_experts = alloc_temp({P}, int32, encoder);
      bucket_offsets = alloc_temp({P + 1}, int32, encoder);
      pair_indices = alloc_temp({P}, int32, encoder);
      num_unique = alloc_temp({1}, int32, encoder);
    }

    // 1. Plan unique experts (when needed) + quantize X[M, 2560] to g32 Q8_1
    if (m1_direct_q4k) {
      encoder.set_input_array(x);
      encoder.set_output_array(xq);
      encoder.set_output_array(x_scales);
      encoder.set_output_array(x_sums);
      auto k_qx = cu::jit_kernel(
          encoder.device(), qwen4_moe_few_source, "qwen4_moe_quantize_x_m1");
      cu::KernelArgs args;
      args.append(x);
      args.append(xq);
      args.append(x_scales);
      args.append(x_sums);
      args.append<int32_t>(kH);
      const uint32_t grid_x =
          static_cast<uint32_t>(((kH / 32) * 8 + 255) / 256);
      encoder.add_kernel_node_raw(
          k_qx, dim3(grid_x, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
    } else {
      encoder.set_input_array(ids);
      encoder.set_input_array(x);
      encoder.set_output_array(*unique_experts);
      encoder.set_output_array(*bucket_offsets);
      encoder.set_output_array(*pair_indices);
      encoder.set_output_array(*num_unique);
      encoder.set_output_array(xq);
      encoder.set_output_array(x_scales);
      encoder.set_output_array(x_sums);
      auto k_plan = cu::jit_kernel(
          encoder.device(), qwen4_moe_few_source,
          "qwen4_moe_plan_and_quantize_few");
      cu::KernelArgs args;
      args.append(ids);
      args.append(x);
      args.append(*unique_experts);
      args.append(*bucket_offsets);
      args.append(*pair_indices);
      args.append(*num_unique);
      args.append(xq);
      args.append(x_scales);
      args.append(x_sums);
      args.append<int32_t>(M);
      args.append<int32_t>(P);
      args.append<int32_t>(num_experts_);
      args.append<int32_t>(kH);
      const uint32_t grid_x =
          1u + static_cast<uint32_t>((M * (kH / 32) * 8 + 255) / 256);
      encoder.add_kernel_node_raw(
          k_plan, dim3(grid_x, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
    }

    const bool allow_hi = (M > 8);

    // 2. Routed gate + up + SwiGLU (Q4_K or Q5_K) -> mid[P, 640]
    if (mode_gu_ == 0 && M == 1) {
      encoder.set_input_array(wg);
      encoder.set_input_array(sg);
      encoder.set_input_array(bg);
      encoder.set_input_array(wu);
      encoder.set_input_array(su);
      encoder.set_input_array(bu);
      encoder.set_input_array(xq);
      encoder.set_input_array(x_scales);
      encoder.set_input_array(x_sums);
      encoder.set_input_array(ids);
      encoder.set_output_array(mid);
      auto k_gu = cu::jit_kernel(
          encoder.device(), qwen4_moe_few_source,
          "qwen4_moe_gate_up_swiglu_q4k_m1");
      cu::KernelArgs args;
      args.append(wg);
      args.append(sg);
      args.append(bg);
      args.append(wu);
      args.append(su);
      args.append(bu);
      args.append(xq);
      args.append(x_scales);
      args.append(x_sums);
      args.append(ids);
      args.append(mid);
      args.append<int32_t>(kDff);
      args.append<int32_t>(kH);
      encoder.add_kernel_node_raw(
          k_gu, dim3(kDff / 16, top_k_, 1), dim3(128, 1, 1), {}, 0,
          args.args());
    } else {
      encoder.set_input_array(wg);
      encoder.set_input_array(sg);
      encoder.set_input_array(bg);
      encoder.set_input_array(wu);
      encoder.set_input_array(su);
      encoder.set_input_array(bu);
      encoder.set_input_array(xq);
      encoder.set_input_array(x_scales);
      encoder.set_input_array(x_sums);
      encoder.set_input_array(*unique_experts);
      encoder.set_input_array(*bucket_offsets);
      encoder.set_input_array(*pair_indices);
      encoder.set_input_array(*num_unique);
      encoder.set_output_array(mid);
      const char* entry = (mode_gu_ == 0)
          ? (allow_hi ? "qwen4_moe_gate_up_swiglu_q4k_few<2, true>"
                      : "qwen4_moe_gate_up_swiglu_q4k_few<2, false>")
          : (allow_hi ? "qwen4_moe_gate_up_swiglu_few<1, true>"
                      : "qwen4_moe_gate_up_swiglu_few<1, false>");
      const uint32_t grid_x = (mode_gu_ == 0) ? (kDff / 32) : (kDff / 16);
      auto k_gu = cu::jit_kernel(encoder.device(), qwen4_moe_few_source, entry);
      cu::KernelArgs args;
      args.append(wg);
      args.append(sg);
      args.append(bg);
      args.append(wu);
      args.append(su);
      args.append(bu);
      args.append(xq);
      args.append(x_scales);
      args.append(x_sums);
      args.append(*unique_experts);
      args.append(*bucket_offsets);
      args.append(*pair_indices);
      args.append(*num_unique);
      args.append(mid);
      args.append<int32_t>(kDff);
      args.append<int32_t>(kH);
      args.append<int32_t>(top_k_);
      encoder.add_kernel_node_raw(
          k_gu, dim3(grid_x, P, 1), dim3(32, 5, 1), {}, 0, args.args());
    }

    // 3. Routed down projection (Q5_1 or Q8_0) -> down_pairs[P, 2560]
    if (M == 1) {
      encoder.set_input_array(wd);
      encoder.set_input_array(sd);
      encoder.set_input_array(bd);
      encoder.set_input_array(mid);
      encoder.set_input_array(ids);
      encoder.set_output_array(down_pairs);
      const char* entry = (bits_dn_ == 5)
          ? "qwen4_moe_down_m1<5>"
          : "qwen4_moe_down_m1<8>";
      auto k_dn = cu::jit_kernel(encoder.device(), qwen4_moe_few_source, entry);
      cu::KernelArgs args;
      args.append(wd);
      args.append(sd);
      args.append(bd);
      args.append(mid);
      args.append(ids);
      args.append(down_pairs);
      args.append<int32_t>(kH);
      args.append<int32_t>(kDff);
      const uint32_t grid_x = (bits_dn_ == 5) ? (kH / 16) : (kH / 4);
      encoder.add_kernel_node_raw(
          k_dn, dim3(grid_x, top_k_, 1), dim3(32, 4, 1), {}, 0, args.args());
    } else {
      encoder.set_input_array(wd);
      encoder.set_input_array(sd);
      encoder.set_input_array(bd);
      encoder.set_input_array(mid);
      encoder.set_input_array(scores);
      encoder.set_input_array(*unique_experts);
      encoder.set_input_array(*bucket_offsets);
      encoder.set_input_array(*pair_indices);
      encoder.set_input_array(*num_unique);
      encoder.set_output_array(down_pairs);
      const char* entry = (bits_dn_ == 5)
          ? (allow_hi ? "qwen4_moe_down_few<5, 4, true, false>"
                      : "qwen4_moe_down_few<5, 4, false, false>")
          : (allow_hi ? "qwen4_moe_down_few<8, 4, true, false>"
                      : "qwen4_moe_down_few<8, 4, false, false>");
      auto k_dn = cu::jit_kernel(encoder.device(), qwen4_moe_few_source, entry);
      cu::KernelArgs args;
      args.append(wd);
      args.append(sd);
      args.append(bd);
      args.append(mid);
      args.append(scores);
      args.append(*unique_experts);
      args.append(*bucket_offsets);
      args.append(*pair_indices);
      args.append(*num_unique);
      args.append(down_pairs);
      args.append<int32_t>(kH);
      args.append<int32_t>(kDff);
      args.append<int32_t>(top_k_);
      encoder.add_kernel_node_raw(
          k_dn, dim3(kH / 64, P, 1), dim3(32, 5, 1), {}, 0, args.args());
    }

    // 4. Shared expert (optional) + deterministic weighted combine -> out[M, 2560]
    if (has_shared_) {
      array swg = cu::ensure_row_contiguous_aligned<16>(inputs[12], stream(), encoder);
      array ssg = cu::ensure_row_contiguous_aligned<16>(inputs[13], stream(), encoder);
      array sbg = cu::ensure_row_contiguous_aligned<16>(inputs[14], stream(), encoder);
      array swu = cu::ensure_row_contiguous_aligned<16>(inputs[15], stream(), encoder);
      array ssu = cu::ensure_row_contiguous_aligned<16>(inputs[16], stream(), encoder);
      array sbu = cu::ensure_row_contiguous_aligned<16>(inputs[17], stream(), encoder);
      array swd = cu::ensure_row_contiguous_aligned<16>(inputs[18], stream(), encoder);
      array ssd = cu::ensure_row_contiguous_aligned<16>(inputs[19], stream(), encoder);
      array sbd = cu::ensure_row_contiguous_aligned<16>(inputs[20], stream(), encoder);
      array shared_gate =
          cu::ensure_row_contiguous_aligned<2>(inputs[21], stream(), encoder);
      array shared_mid = alloc_temp({M, kDff}, bfloat16, encoder);

      if (M == 1) {
        {
          encoder.set_input_array(swg);
          encoder.set_input_array(ssg);
          encoder.set_input_array(sbg);
          encoder.set_input_array(swu);
          encoder.set_input_array(ssu);
          encoder.set_input_array(sbu);
          encoder.set_input_array(x);
          encoder.set_output_array(shared_mid);
          auto k_sgu = cu::jit_kernel(
              encoder.device(), qwen4_moe_few_source,
              "qwen4_shared_gate_up_swiglu_m1");
          cu::KernelArgs args;
          args.append(swg);
          args.append(ssg);
          args.append(sbg);
          args.append(swu);
          args.append(ssu);
          args.append(sbu);
          args.append(x);
          args.append(shared_mid);
          args.append<int32_t>(kDff);
          args.append<int32_t>(kH);
          encoder.add_kernel_node_raw(
              k_sgu, dim3(kDff, 1, 1), dim3(32, 4, 1), {}, 0, args.args());
        }
        {
          encoder.set_input_array(swd);
          encoder.set_input_array(ssd);
          encoder.set_input_array(sbd);
          encoder.set_input_array(shared_mid);
          encoder.set_input_array(shared_gate);
          encoder.set_input_array(down_pairs);
          encoder.set_input_array(scores);
          encoder.set_output_array(out);
          auto k_sdn = cu::jit_kernel(
              encoder.device(), qwen4_moe_few_source,
              "qwen4_shared_down_combine_m1");
          cu::KernelArgs args;
          args.append(swd);
          args.append(ssd);
          args.append(sbd);
          args.append(shared_mid);
          args.append(shared_gate);
          args.append(down_pairs);
          args.append(scores);
          args.append(out);
          args.append<int32_t>(kH);
          args.append<int32_t>(kDff);
          args.append<int32_t>(top_k_);
          encoder.add_kernel_node_raw(
              k_sdn, dim3(kH / 4, 1, 1), dim3(32, 4, 1), {}, 0, args.args());
        }
      } else {
        {
          encoder.set_input_array(swg);
          encoder.set_input_array(ssg);
          encoder.set_input_array(sbg);
          encoder.set_input_array(swu);
          encoder.set_input_array(ssu);
          encoder.set_input_array(sbu);
          encoder.set_input_array(x);
          encoder.set_output_array(shared_mid);
          auto k_sgu = cu::jit_kernel(
              encoder.device(), qwen4_moe_few_source,
              allow_hi ? "qwen4_shared_gate_up_swiglu_few<true>"
                       : "qwen4_shared_gate_up_swiglu_few<false>");
          cu::KernelArgs args;
          args.append(swg);
          args.append(ssg);
          args.append(sbg);
          args.append(swu);
          args.append(ssu);
          args.append(sbu);
          args.append(x);
          args.append(shared_mid);
          args.append<int32_t>(M);
          args.append<int32_t>(kDff);
          args.append<int32_t>(kH);
          encoder.add_kernel_node_raw(
              k_sgu, dim3(kDff / 16, 1, 1), dim3(32, 4, 1), {}, 0, args.args());
        }
        {
          encoder.set_input_array(swd);
          encoder.set_input_array(ssd);
          encoder.set_input_array(sbd);
          encoder.set_input_array(shared_mid);
          encoder.set_input_array(shared_gate);
          encoder.set_input_array(down_pairs);
          encoder.set_input_array(scores);
          encoder.set_output_array(out);
          auto k_sdn = cu::jit_kernel(
              encoder.device(), qwen4_moe_few_source,
              allow_hi ? "qwen4_shared_down_combine_few<2, true>"
                       : "qwen4_shared_down_combine_few<2, false>");
          cu::KernelArgs args;
          args.append(swd);
          args.append(ssd);
          args.append(sbd);
          args.append(shared_mid);
          args.append(shared_gate);
          args.append(down_pairs);
          args.append(scores);
          args.append(out);
          args.append<int32_t>(M);
          args.append<int32_t>(kH);
          args.append<int32_t>(kDff);
          args.append<int32_t>(top_k_);
          encoder.add_kernel_node_raw(
              k_sdn, dim3(kH / 32, 1, 1), dim3(32, 5, 1), {}, 0, args.args());
        }
      }
    } else {
      encoder.set_input_array(down_pairs);
      encoder.set_input_array(scores);
      encoder.set_output_array(out);
      auto k_comb = cu::jit_kernel(
          encoder.device(), qwen4_moe_few_source,
          "qwen4_moe_combine_few<false>");
      cu::KernelArgs args;
      args.append(down_pairs);
      args.append(scores);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(kH);
      args.append<int32_t>(top_k_);
      encoder.add_kernel_node_raw(
          k_comb, dim3((kH / 8 + 255) / 256, M, 1), dim3(256, 1, 1), {}, 0,
          args.args());
    }
  }

  DEFINE_NAME(Qwen4MoeFewCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4MoeFewCuda*>(&other);
    return rhs && rhs->mode_gu_ == mode_gu_ && rhs->bits_dn_ == bits_dn_ &&
        rhs->num_experts_ == num_experts_ && rhs->top_k_ == top_k_ &&
        rhs->has_shared_ == has_shared_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int mode_gu_;
  int bits_dn_;
  int num_experts_;
  int top_k_;
  bool has_shared_;
};

}  // namespace
}  // namespace mlx::core::qwen4_moe

extern "C" mlx_array* mlx_qwen4_moe_routed_shared_few_cuda(
    mlx_array* x_h, mlx_array* ids_h, mlx_array* scores_h,
    mlx_array* wg_h, mlx_array* sg_h, mlx_array* bg_h,
    mlx_array* wu_h, mlx_array* su_h, mlx_array* bu_h,
    mlx_array* wd_h, mlx_array* sd_h, mlx_array* bd_h,
    int32_t mode_gu, int32_t bits_dn,
    mlx_array* swg_h, mlx_array* ssg_h, mlx_array* sbg_h,
    mlx_array* swu_h, mlx_array* ssu_h, mlx_array* sbu_h,
    mlx_array* swd_h, mlx_array* ssd_h, mlx_array* sbd_h,
    mlx_array* shared_gate_h) {
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
    const bool any_shared = swg_h || ssg_h || sbg_h || swu_h || ssu_h ||
        sbu_h || swd_h || ssd_h || sbd_h || shared_gate_h;
    const bool all_shared = swg_h && ssg_h && sbg_h && swu_h && ssu_h &&
        sbu_h && swd_h && ssd_h && sbd_h && shared_gate_h;
    if (any_shared && !all_shared) {
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
    if (M < 1 || M > 16 || top_k > 10 || M * top_k > 256) {
      return nullptr;
    }
    std::vector<array> inputs;
    inputs.reserve(all_shared ? 22 : 12);
    inputs.insert(inputs.end(), {x, ids, scores, wg, sg, bg, wu, su, bu, wd, sd, bd});

    if (all_shared) {
      const auto& swg = *reinterpret_cast<array*>(swg_h);
      const auto& ssg = *reinterpret_cast<array*>(ssg_h);
      const auto& sbg = *reinterpret_cast<array*>(sbg_h);
      const auto& swu = *reinterpret_cast<array*>(swu_h);
      const auto& ssu = *reinterpret_cast<array*>(ssu_h);
      const auto& sbu = *reinterpret_cast<array*>(sbu_h);
      const auto& swd = *reinterpret_cast<array*>(swd_h);
      const auto& ssd = *reinterpret_cast<array*>(ssd_h);
      const auto& sbd = *reinterpret_cast<array*>(sbd_h);
      const auto& sgate = *reinterpret_cast<array*>(shared_gate_h);
      if (!check_2d_bank(swg, uint32, kDff, 640) ||
          !check_2d_bank(ssg, float16, kDff, 80) ||
          !check_2d_bank(sbg, float16, kDff, 80) ||
          !check_2d_bank(swu, uint32, kDff, 640) ||
          !check_2d_bank(ssu, float16, kDff, 80) ||
          !check_2d_bank(sbu, float16, kDff, 80) ||
          !check_2d_bank(swd, uint32, kH, 160) ||
          !check_2d_bank(ssd, float16, kH, 20) ||
          !check_2d_bank(sbd, float16, kH, 20) ||
          sgate.dtype() != bfloat16 ||
          sgate.size() != static_cast<size_t>(M)) {
        return nullptr;
      }
      inputs.insert(
          inputs.end(),
          {swg, ssg, sbg, swu, ssu, sbu, swd, ssd, sbd, sgate});
    }

    auto result = array(
        x.shape(), bfloat16,
        std::make_shared<Qwen4MoeFewCuda>(
            default_stream(Device::gpu), mode_gu, bits_dn, E, top_k,
            all_shared),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_moe_routed_shared_few_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" mlx_array* mlx_qwen4_moe_routed_shared_few_cuda(
    mlx_array* x_h, mlx_array* ids_h, mlx_array* scores_h,
    mlx_array* wg_h, mlx_array* sg_h, mlx_array* bg_h,
    mlx_array* wu_h, mlx_array* su_h, mlx_array* bu_h,
    mlx_array* wd_h, mlx_array* sd_h, mlx_array* bd_h,
    int32_t mode_gu, int32_t bits_dn,
    mlx_array* swg_h, mlx_array* ssg_h, mlx_array* sbg_h,
    mlx_array* swu_h, mlx_array* ssu_h, mlx_array* sbu_h,
    mlx_array* swd_h, mlx_array* ssd_h, mlx_array* sbd_h,
    mlx_array* shared_gate_h) {
  (void)x_h; (void)ids_h; (void)scores_h;
  (void)wg_h; (void)sg_h; (void)bg_h;
  (void)wu_h; (void)su_h; (void)bu_h;
  (void)wd_h; (void)sd_h; (void)bd_h;
  (void)mode_gu; (void)bits_dn;
  (void)swg_h; (void)ssg_h; (void)sbg_h;
  (void)swu_h; (void)ssu_h; (void)sbu_h;
  (void)swd_h; (void)ssd_h; (void)sbd_h;
  (void)shared_gate_h;
  return nullptr;
}

#endif
