#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mlx::core::qwen4_moe {
namespace {

constexpr int kH = 2560;
constexpr int kDff = 640;
constexpr int kMaxExperts = 512;

constexpr char qwen4_moe_few_source[] =
#include "cuda/qwen4_moe_few.cu.inc"
    ;

constexpr char qwen4_moe_prefill_source[] =
#include "cuda/qwen4_moe_prefill.cu.inc"
    ;

array alloc_temp(Shape shape, Dtype dtype, cu::CommandEncoder& encoder) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(cu::malloc_async(a.nbytes(), encoder));
  encoder.add_temporary(a);
  return a;
}

array alloc_persistent(Shape shape, Dtype dtype, cu::CommandEncoder& encoder) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(cu::malloc_async(a.nbytes(), encoder));
  return a;
}

// Persistent per-(device, stream) workspace for Qwen4MoePrefillCuda.
// Eliminates 137 MB/layer x N in-flight WDDM cudaMallocAsync allocations
// whenever prompt length M changes (1504 -> 1695 -> 2048) by reusing one
// bucketed workspace across all 48 sequential layers.
struct Qwen4MoePrefillWorkspace {
  int cap_m = 0;
  int cap_p = 0;
  int cap_tiles = 0;
  std::optional<array> sorted_pairs;
  std::optional<array> inv_perm;
  std::optional<array> tile_expert;
  std::optional<array> tile_begin;
  std::optional<array> tile_end;
  std::optional<array> num_tiles;
  std::optional<array> xq;
  std::optional<array> x_scales;
  std::optional<array> x_qsums;
  std::optional<array> mid_q;
  std::optional<array> mid_sc;
  std::optional<array> mid_qsum;
  std::optional<array> down_out;

  void ensure(int M, int P, int max_tiles, cu::CommandEncoder& encoder) {
    if (M <= cap_m && P <= cap_p && max_tiles <= cap_tiles && down_out) {
      return;
    }
    auto retire = [&](std::optional<array>& opt) {
      if (opt) {
        encoder.add_temporary(std::move(*opt));
        opt.reset();
      }
    };
    retire(sorted_pairs);
    retire(inv_perm);
    retire(tile_expert);
    retire(tile_begin);
    retire(tile_end);
    retire(num_tiles);
    retire(xq);
    retire(x_scales);
    retire(x_qsums);
    retire(mid_q);
    retire(mid_sc);
    retire(mid_qsum);
    retire(down_out);

    int bm = M <= 256 ? 256
        : (M <= 512 ? 512
        : (M <= 2048 ? 2048
        : (M <= 4096 ? 4096
        : (M <= 6144 ? 6144
        : (M <= 8192 ? 8192 : ((M + 1023) / 1024) * 1024)))));
    int bp = std::max(P, bm * 10);
    int bt = std::max(max_tiles, (bp + 63) / 64 + kMaxExperts);
    cap_m = bm;
    cap_p = bp;
    cap_tiles = bt;

    sorted_pairs = alloc_persistent({bp}, int32, encoder);
    inv_perm = alloc_persistent({bp}, int32, encoder);
    tile_expert = alloc_persistent({bt}, int32, encoder);
    tile_begin = alloc_persistent({bt}, int32, encoder);
    tile_end = alloc_persistent({bt}, int32, encoder);
    // [0] = total_tiles, [1] = num_hot_tiles (BM=64 hot + BM=32 cold split)
    num_tiles = alloc_persistent({2}, int32, encoder);
    xq = alloc_persistent({bm * kH}, uint8, encoder);
    x_scales = alloc_persistent({bm * (kH / 256)}, float32, encoder);
    x_qsums = alloc_persistent({bm * (kH / 32)}, int16, encoder);
    mid_q = alloc_persistent({bp, kDff}, uint8, encoder);
    mid_sc = alloc_persistent({bp, kDff / 64}, float32, encoder);
    mid_qsum = alloc_persistent({bp, kDff / 32}, int16, encoder);
    down_out = alloc_persistent({bp, kH}, bfloat16, encoder);
  }
};

Qwen4MoePrefillWorkspace& get_prefill_workspace(int dev, int stream_idx) {
  static std::mutex mu;
  // Intentionally leaked so GPU buffers are not destroyed during process teardown.
  static auto* map = new std::unordered_map<uint64_t, Qwen4MoePrefillWorkspace>();
  uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(dev)) << 32) |
      static_cast<uint32_t>(stream_idx);
  std::lock_guard<std::mutex> lk(mu);
  return (*map)[key];
}

// Validate a routed expert weight/scale/bias bank in either 3D [E, rows, cols]
// or 2D [E * rows, cols] layout. Returns E or -1 on mismatch.
int expert_bank_count(
    const array& a, Dtype expected_dtype, int rows_per_exp, int expected_cols) {
  if (a.dtype() != expected_dtype) {
    return -1;
  }
  if (a.ndim() == 3) {
    if (a.shape(1) != rows_per_exp || a.shape(2) != expected_cols) {
      return -1;
    }
    return a.shape(0);
  }
  if (a.ndim() == 2) {
    if (a.shape(1) != expected_cols || a.shape(0) <= 0 ||
        a.shape(0) % rows_per_exp != 0) {
      return -1;
    }
    return a.shape(0) / rows_per_exp;
  }
  return -1;
}

bool check_2d_bank(const array& a, Dtype dt, int rows, int cols) {
  return a.dtype() == dt && a.ndim() == 2 && a.shape(0) == rows &&
      a.shape(1) == cols;
}

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

// Common validation for routed banks and routing metadata.
bool validate_routed_inputs(
    const array& x, const array& ids, const array& scores,
    const array& wg, const array& sg, const array& bg,
    const array& wu, const array& su, const array& bu,
    const array& wd, const array& sd, const array& bd,
    int mode_gu, int bits_dn,
    int& out_m, int& out_e, int& out_top_k) {
  if (x.dtype() != bfloat16 || x.ndim() < 2 || x.shape(-1) != kH) {
    return false;
  }
  if (mode_gu != 0 && mode_gu != 1) {
    return false;
  }
  if (bits_dn != 5 && bits_dn != 8) {
    return false;
  }
  const int64_t m64 = static_cast<int64_t>(x.size()) / kH;
  if (m64 < 1 || m64 > 65535) {
    return false;
  }
  const int M = static_cast<int>(m64);
  if (ids.dtype() != int32 && ids.dtype() != uint32) {
    return false;
  }
  if (scores.dtype() != float32 || ids.size() != scores.size() ||
      ids.size() == 0 || ids.size() % static_cast<size_t>(M) != 0) {
    return false;
  }
  const int top_k = static_cast<int>(ids.size() / static_cast<size_t>(M));
  if (top_k < 1) {
    return false;
  }

  const int gu_w_cols = (mode_gu == 0) ? 320 : 400;
  const int dn_w_cols = (bits_dn == 5) ? 100 : 160;
  const int e0 = expert_bank_count(wg, uint32, kDff, gu_w_cols);
  if (e0 < 1 || e0 > kMaxExperts) {
    return false;
  }
  if (expert_bank_count(sg, uint8, kDff, 160) != e0 ||
      expert_bank_count(bg, float16, kDff, 20) != e0 ||
      expert_bank_count(wu, uint32, kDff, gu_w_cols) != e0 ||
      expert_bank_count(su, uint8, kDff, 160) != e0 ||
      expert_bank_count(bu, float16, kDff, 20) != e0 ||
      expert_bank_count(wd, uint32, kH, dn_w_cols) != e0 ||
      expert_bank_count(sd, float16, kH, 20) != e0 ||
      expert_bank_count(bd, float16, kH, 20) != e0) {
    return false;
  }
  out_m = M;
  out_e = e0;
  out_top_k = top_k;
  return true;
}

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
