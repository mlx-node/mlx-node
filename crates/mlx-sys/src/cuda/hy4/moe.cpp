#include "cuda/hy4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mlx::core::hy4_moe_cuda {
namespace {

using namespace mlx_node::hy4_cuda;

constexpr int kMaxExperts = 512;
constexpr int kMaxTopK = 16;

constexpr char hy4_moe_router_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_moe_router.cu.inc"
    ;

constexpr char hy4_moe_few_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_moe_few.cu.inc"
    ;

constexpr char hy4_moe_prefill_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_moe_prefill.cu.inc"
    ;

inline array alloc_temp(
    Shape shape, Dtype dtype, cu::CommandEncoder& encoder) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(cu::malloc_async(a.nbytes(), encoder));
  encoder.add_temporary(a);
  return a;
}

inline array alloc_persistent(
    Shape shape, Dtype dtype, cu::CommandEncoder& encoder) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(cu::malloc_async(a.nbytes(), encoder));
  return a;
}

// Accept 2D column-sliced BF16 `z` (e.g. `[M, 1536]` sliced from `[M, 4608]`)
// without copying whenever inner stride is 1, row stride is a non-negative
// multiple of 8 BF16 elements (16 B), and base device pointer is 16 B aligned.
inline array prepare_z_strided_aligned16(
    const array& input, int lat, const Stream& stream,
    cu::CommandEncoder& encoder, int32_t& out_z_row_u4) {
  if (input.ndim() == 2 && input.strides(1) == 1 &&
      input.strides(0) >= 0 && (input.strides(0) % 8) == 0 &&
      input.strides(0) <= INT32_MAX &&
      (reinterpret_cast<uintptr_t>(gpu_ptr<void>(input)) & 15u) == 0) {
    out_z_row_u4 = static_cast<int32_t>(input.strides(0) / 8);
    return input;
  }
  array z = cu::ensure_row_contiguous_aligned<16>(input, stream, encoder);
  out_z_row_u4 = lat >> 3;
  return z;
}

// Persistent bucketed per-(device, stream) workspace for Hy4MoeRoutedPrefillCuda.
// Reuses one bucketed `mid` + `down_out` + route-table allocation across all
// 41 sequential MoE layers during prefill (eliminating 41 x ~110 MB concurrent
// WDDM cudaMallocAsync stalls). Generation-gated stream completion handler
// automatically returns buffers to MLX's buffer_cache_ once the stream goes idle
// so active memory drops to 0 and clear_cache() reclaims VRAM cleanly.
struct Hy4MoePrefillWorkspace {
  std::mutex mu;
  uint64_t use_gen = 0;
  int cap_p = 0;
  int cap_tiles = 0;
  int cap_ff = 0;
  int cap_lat = 0;
  std::optional<array> sorted_pairs;
  std::optional<array> pair_to_sorted;
  std::optional<array> tile_expert;
  std::optional<array> tile_begin;
  std::optional<array> tile_end;
  std::optional<array> num_tiles;
  std::optional<array> mid;
  std::optional<array> down_out;

  void reset_locked() {
    sorted_pairs.reset();
    pair_to_sorted.reset();
    tile_expert.reset();
    tile_begin.reset();
    tile_end.reset();
    num_tiles.reset();
    mid.reset();
    down_out.reset();
    cap_p = 0;
    cap_tiles = 0;
    cap_ff = 0;
    cap_lat = 0;
  }

  void ensure_locked(
      int P, int max_tiles, int ff, int lat, cu::CommandEncoder& encoder) {
    if (P <= cap_p && max_tiles <= cap_tiles && ff <= cap_ff &&
        lat <= cap_lat && down_out) {
      return;
    }
    auto retire = [&](std::optional<array>& opt) {
      if (opt) {
        encoder.add_temporary(std::move(*opt));
        opt.reset();
      }
    };
    retire(sorted_pairs);
    retire(pair_to_sorted);
    retire(tile_expert);
    retire(tile_begin);
    retire(tile_end);
    retire(num_tiles);
    retire(mid);
    retire(down_out);

    // Tight 2048-pair bucket rounding keeps Spark headroom bounded while avoiding
    // churn across prompt-length increments.
    const int bp = std::max(cap_p, std::max(2048, ((P + 2047) / 2048) * 2048));
    const int bt = std::max(
        cap_tiles, std::max(max_tiles, (bp + 63) / 64 + kMaxExperts));
    const int bff = std::max(cap_ff, ff);
    const int blat = std::max(cap_lat, lat);

    sorted_pairs = alloc_persistent({bp}, int32, encoder);
    pair_to_sorted = alloc_persistent({bp}, int32, encoder);
    tile_expert = alloc_persistent({bt}, int32, encoder);
    tile_begin = alloc_persistent({bt}, int32, encoder);
    tile_end = alloc_persistent({bt}, int32, encoder);
    num_tiles = alloc_persistent({2}, int32, encoder);
    mid = alloc_persistent({bp, bff}, bfloat16, encoder);
    down_out = alloc_persistent({bp, blat}, bfloat16, encoder);

    cap_p = bp;
    cap_tiles = bt;
    cap_ff = bff;
    cap_lat = blat;
  }
};

inline Hy4MoePrefillWorkspace& get_prefill_workspace(int dev, int stream_idx) {
  static std::mutex map_mu;
  // Intentionally leaked so GPU buffers are not destroyed during process teardown.
  static auto* map = new std::unordered_map<uint64_t, Hy4MoePrefillWorkspace>();
  const uint64_t key =
      (static_cast<uint64_t>(static_cast<uint32_t>(dev)) << 32) |
      static_cast<uint32_t>(stream_idx);
  std::lock_guard<std::mutex> lk(map_mu);
  return map->try_emplace(key).first->second;
}

// Persistent per-(device, stream) workspace for Hy4MoeRouterTopkCuda (M <= 16).
// Reuses a single 32 KB FP32 logits buffer across all 41 layers; automatically
// returns it to MLX's buffer_cache_ once the stream finishes queued work.
struct Hy4MoeRouterWorkspace {
  std::mutex mu;
  uint64_t use_gen = 0;
  std::optional<array> logits;

  void reset_locked() {
    logits.reset();
  }

  void ensure_locked(cu::CommandEncoder& encoder) {
    if (logits) return;
    logits = alloc_persistent({16, kMaxExperts}, float32, encoder);
  }
};

inline Hy4MoeRouterWorkspace& get_router_workspace(int dev, int stream_idx) {
  static std::mutex map_mu;
  static auto* map = new std::unordered_map<uint64_t, Hy4MoeRouterWorkspace>();
  const uint64_t key =
      (static_cast<uint64_t>(static_cast<uint32_t>(dev)) << 32) |
      static_cast<uint32_t>(stream_idx);
  std::lock_guard<std::mutex> lk(map_mu);
  return map->try_emplace(key).first->second;
}

// Persistent packed per-(device, stream) workspace for Hy4MoeRoutedFewCuda (M <= 16).
// Eliminates 7 cu::malloc_async calls per layer (287/forward -> 1-2 on layer 0,
// 0 across layers 1..40), returning all buffers to MLX's buffer_cache_ on idle.
struct Hy4MoeFewWorkspace {
  std::mutex mu;
  uint64_t use_gen = 0;
  int cap_meta_i32 = 0;
  size_t cap_bf16 = 0;
  std::optional<array> meta;
  std::optional<array> mid_down;

  void reset_locked() {
    meta.reset();
    mid_down.reset();
    cap_meta_i32 = 0;
    cap_bf16 = 0;
  }

  void ensure_locked(
      int need_meta_i32, size_t need_bf16, cu::CommandEncoder& encoder) {
    if (need_meta_i32 > 0 && (need_meta_i32 > cap_meta_i32 || !meta)) {
      if (meta) {
        encoder.add_temporary(std::move(*meta));
        meta.reset();
      }
      const int cap = std::max(cap_meta_i32, std::max(1536, need_meta_i32));
      meta = alloc_persistent({cap}, int32, encoder);
      cap_meta_i32 = cap;
    }
    if (need_bf16 > cap_bf16 || !mid_down) {
      if (mid_down) {
        encoder.add_temporary(std::move(*mid_down));
        mid_down.reset();
      }
      const size_t cap = std::max(cap_bf16, need_bf16);
      mid_down = alloc_persistent({static_cast<int>(cap)}, bfloat16, encoder);
      cap_bf16 = cap;
    }
  }
};

inline Hy4MoeFewWorkspace& get_few_workspace(int dev, int stream_idx) {
  static std::mutex map_mu;
  static auto* map = new std::unordered_map<uint64_t, Hy4MoeFewWorkspace>();
  const uint64_t key =
      (static_cast<uint64_t>(static_cast<uint32_t>(dev)) << 32) |
      static_cast<uint32_t>(stream_idx);
  std::lock_guard<std::mutex> lk(map_mu);
  return map->try_emplace(key).first->second;
}

inline bool is_f16_or_bf16(Dtype dt) {
  return dt == float16 || dt == bfloat16;
}

inline int combine_block_x(int lat) {
  const int lat8 = lat >> 3;
  if (lat8 <= 32) return 32;
  if (lat8 >= 256) return 256;
  return (lat8 + 31) & ~31;
}

struct RoutedShapeInfo {
  int M;
  int E;
  int top_k;
  int lat;
  int ff;
  int gu_sb_bf16;
  int dn_sb_bf16;
};

inline bool validate_routed_banks(
    const array& z,
    const array& topk_ids,
    const array& topk_weights,
    const array& gate_w,
    const array& gate_s,
    const array& up_w,
    const array& up_s,
    const array& down_w,
    const array& down_s,
    int32_t mode,
    bool few_only,
    RoutedShapeInfo& info) {
  if (mode != 0 && mode != 1) return false;
  if (z.dtype() != bfloat16 || z.ndim() < 2) return false;

  const int lat = z.shape(-1);
  if (lat < 64 || (lat % 64) != 0) return false;
  const int64_t m64 = static_cast<int64_t>(z.size()) / lat;
  if (m64 < 1 || (few_only ? (m64 > 16) : (m64 > 65535))) return false;
  const int M = static_cast<int>(m64);

  if (topk_ids.dtype() != uint32 && topk_ids.dtype() != int32) return false;
  if (topk_weights.dtype() != float32) return false;
  if (topk_ids.size() == 0 || topk_ids.size() != topk_weights.size() ||
      (topk_ids.size() % static_cast<size_t>(M)) != 0) {
    return false;
  }
  const int64_t tk64 = static_cast<int64_t>(topk_ids.size() / static_cast<size_t>(M));
  if (tk64 < 1 || tk64 > kMaxTopK) return false;
  const int top_k = static_cast<int>(tk64);

  if (gate_w.dtype() != uint32 || gate_w.ndim() != 3 ||
      up_w.dtype() != uint32 || up_w.ndim() != 3 ||
      down_w.dtype() != uint32 || down_w.ndim() != 3) {
    return false;
  }
  const int E = gate_w.shape(0);
  const int ff = gate_w.shape(1);
  if (E < 1 || E > kMaxExperts || top_k > E) return false;
  if (ff < 64 || (ff % 64) != 0) return false;

  const int gu_w_cols = (mode == 0) ? (lat / 8) : (lat / 4);
  const int dn_w_cols = (mode == 0) ? (ff / 8) : (ff / 4);
  if (gate_w.shape(2) != gu_w_cols ||
      up_w.shape(0) != E || up_w.shape(1) != ff || up_w.shape(2) != gu_w_cols ||
      down_w.shape(0) != E || down_w.shape(1) != lat || down_w.shape(2) != dn_w_cols) {
    return false;
  }

  int gu_sb_bf16 = 0;
  int dn_sb_bf16 = 0;
  if (mode == 0) {
    const int gu_s_cols = lat / 16;
    const int dn_s_cols = ff / 16;
    if (gate_s.dtype() != uint8 || gate_s.ndim() != 3 ||
        gate_s.shape(0) != E || gate_s.shape(1) != ff || gate_s.shape(2) != gu_s_cols ||
        up_s.dtype() != uint8 || up_s.ndim() != 3 ||
        up_s.shape(0) != E || up_s.shape(1) != ff || up_s.shape(2) != gu_s_cols ||
        down_s.dtype() != uint8 || down_s.ndim() != 3 ||
        down_s.shape(0) != E || down_s.shape(1) != lat || down_s.shape(2) != dn_s_cols) {
      return false;
    }
  } else {
    const int gu_s_cols = lat / 32;
    const int dn_s_cols = ff / 32;
    if (!is_f16_or_bf16(gate_s.dtype()) || up_s.dtype() != gate_s.dtype() ||
        !is_f16_or_bf16(down_s.dtype())) {
      return false;
    }
    if (gate_s.ndim() != 3 || gate_s.shape(0) != E || gate_s.shape(1) != ff ||
        gate_s.shape(2) != gu_s_cols ||
        up_s.ndim() != 3 || up_s.shape(0) != E || up_s.shape(1) != ff ||
        up_s.shape(2) != gu_s_cols ||
        down_s.ndim() != 3 || down_s.shape(0) != E || down_s.shape(1) != lat ||
        down_s.shape(2) != dn_s_cols) {
      return false;
    }
    gu_sb_bf16 = (gate_s.dtype() == bfloat16) ? 1 : 0;
    dn_sb_bf16 = (down_s.dtype() == bfloat16) ? 1 : 0;
  }

  info = RoutedShapeInfo{M, E, top_k, lat, ff, gu_sb_bf16, dn_sb_bf16};
  return true;
}

// ============================================================================
// Primitive 1: Hy4MoeRouterTopkCuda
// ============================================================================
class Hy4MoeRouterTopkCuda final : public Primitive {
 public:
  Hy4MoeRouterTopkCuda(
      Stream stream, int E, int D, int top_k, float route_scale,
      bool norm_topk, bool w_is_bf16, bool has_bias)
      : Primitive(stream),
        E_(E),
        D_(D),
        top_k_(top_k),
        route_scale_(route_scale),
        norm_topk_(norm_topk),
        w_is_bf16_(w_is_bf16),
        has_bias_(has_bias) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Hy4MoeRouterTopkCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array router_w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    std::optional<array> bias;
    if (has_bias_) {
      bias = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);
    }

    const int M = static_cast<int>(x.size() / D_);
    if (M < 1 || M > 65535) {
      throw std::invalid_argument("Hy4MoeRouterTopkCuda: unsupported M");
    }

    array& out_ids = outputs[0];
    array& out_weights = outputs[1];
    out_ids.set_data(cu::malloc_async(out_ids.nbytes(), encoder));
    out_weights.set_data(cu::malloc_async(out_weights.nbytes(), encoder));

    const bool use_ws = (M <= 16);
    std::optional<array> logits_ref;
    if (use_ws) {
      auto& ws = get_router_workspace(
          encoder.device().cuda_device(), stream().index);
      uint64_t my_gen = 0;
      {
        std::lock_guard<std::mutex> lk(ws.mu);
        ws.ensure_locked(encoder);
        my_gen = ++ws.use_gen;
        logits_ref = *ws.logits;
      }
      encoder.add_completed_handler([&ws, my_gen]() {
        std::lock_guard<std::mutex> lk(ws.mu);
        if (ws.use_gen == my_gen) {
          ws.reset_locked();
        }
      });
    } else {
      logits_ref = alloc_temp({M, E_}, float32, encoder);
    }
    const array& logits = *logits_ref;

    // 1. Exact-FP32 Router Logits: logits[M, E] = f32(x[M, D]) @ f32(router_w[E, D])^T
    {
      encoder.set_input_array(x);
      encoder.set_input_array(router_w);
      encoder.set_output_array(logits);
      cu::KernelArgs args;
      args.append(x);
      args.append(router_w);
      args.append(logits);
      args.append<int32_t>(M);
      args.append<int32_t>(E_);
      args.append<int32_t>(D_);
      if (M <= 16) {
        const bool aligned = w_is_bf16_ ? ((D_ & 7) == 0) : ((D_ & 3) == 0);
        if (aligned) {
          const char* entry = nullptr;
          if (M == 1) {
            entry = w_is_bf16_ ? "hy4_moe_router_gemv_u4<true, 1>"
                               : "hy4_moe_router_gemv_u4<false, 1>";
          } else if (M == 2) {
            entry = w_is_bf16_ ? "hy4_moe_router_gemv_u4<true, 2>"
                               : "hy4_moe_router_gemv_u4<false, 2>";
          } else if (M <= 4) {
            entry = w_is_bf16_ ? "hy4_moe_router_gemv_u4<true, 4>"
                               : "hy4_moe_router_gemv_u4<false, 4>";
          } else {
            entry = w_is_bf16_ ? "hy4_moe_router_gemv_u4_multi<true>"
                               : "hy4_moe_router_gemv_u4_multi<false>";
          }
          auto k_mv = cu::jit_kernel(
              encoder.device(), hy4_moe_router_source, entry);
          encoder.add_kernel_node_raw(
              k_mv, dim3(E_, 1, 1), dim3(32, 4, 1), {}, 0, args.args());
        } else {
          auto k_mv = cu::jit_kernel(
              encoder.device(), hy4_moe_router_source,
              w_is_bf16_ ? "hy4_moe_router_gemv_unaligned<true>"
                         : "hy4_moe_router_gemv_unaligned<false>");
          encoder.add_kernel_node_raw(
              k_mv, dim3(E_, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
        }
      } else {
        const bool d_align8 = ((D_ & 7) == 0);
        static constexpr const char* kRouterMmEntries[2][2] = {
            {"hy4_moe_router_gemm_prefill<false, false>",
             "hy4_moe_router_gemm_prefill<false, true>"},
            {"hy4_moe_router_gemm_prefill<true, false>",
             "hy4_moe_router_gemm_prefill<true, true>"},
        };
        auto k_mm = cu::jit_kernel(
            encoder.device(), hy4_moe_router_source,
            kRouterMmEntries[w_is_bf16_ ? 1 : 0][d_align8 ? 1 : 0]);
        ensure_smem_carveout(k_mm, 100);
        encoder.add_kernel_node_raw(
            k_mm, dim3((E_ + 63) / 64, (M + 63) / 64, 1), dim3(32, 4, 1),
            {}, 0, args.args());
      }
    }

    // 2. Sigmoid + Optional Bias + Branchless Tree-Bitonic Top-K + Ascending-ID Sort
    {
      encoder.set_input_array(logits);
      if (bias) encoder.set_input_array(*bias);
      encoder.set_output_array(out_ids);
      encoder.set_output_array(out_weights);
      // Mark persistent workspace logits as an output so MLX's CUDA Graph DAG
      // serializes any subsequent layer reusing ws.logits after Top-K finishes.
      if (use_ws) encoder.set_output_array(logits);
      auto k_tk = cu::jit_kernel(
          encoder.device(), hy4_moe_router_source, "hy4_moe_router_topk_rows");
      cu::KernelArgs args;
      args.append(logits);
      if (bias) args.append(*bias);
      else args.append<CUdeviceptr>(0);
      args.append(out_ids);
      args.append(out_weights);
      args.append<int32_t>(M);
      args.append<int32_t>(E_);
      args.append<int32_t>(top_k_);
      args.append<float>(route_scale_);
      args.append<int32_t>(norm_topk_ ? 1 : 0);
      encoder.add_kernel_node_raw(
          k_tk, dim3(M, 1, 1), dim3(512, 1, 1), {}, 0, args.args());
    }
  }

  DEFINE_NAME(Hy4MoeRouterTopkCuda)

  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4MoeRouterTopkCuda*>(&other);
    return rhs && E_ == rhs->E_ && D_ == rhs->D_ && top_k_ == rhs->top_k_ &&
        route_scale_ == rhs->route_scale_ && norm_topk_ == rhs->norm_topk_ &&
        w_is_bf16_ == rhs->w_is_bf16_ && has_bias_ == rhs->has_bias_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() = top_k_;
    return {s, s};
  }

 private:
  int E_;
  int D_;
  int top_k_;
  float route_scale_;
  bool norm_topk_;
  bool w_is_bf16_;
  bool has_bias_;
};

// ============================================================================
// Primitive 2: Hy4MoeRoutedFewCuda (M = 1..=16)
// ============================================================================
class Hy4MoeRoutedFewCuda final : public UnaryPrimitive {
 public:
  Hy4MoeRoutedFewCuda(
      Stream stream, int E, int top_k, int lat, int ff, int mode,
      float swiglu_limit, int gu_sb_bf16, int dn_sb_bf16)
      : UnaryPrimitive(stream),
        E_(E),
        top_k_(top_k),
        lat_(lat),
        ff_(ff),
        mode_(mode),
        swiglu_limit_(swiglu_limit),
        gu_sb_bf16_(gu_sb_bf16),
        dn_sb_bf16_(dn_sb_bf16) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4MoeRoutedFewCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    int32_t z_row_u4 = lat_ >> 3;
    array z = prepare_z_strided_aligned16(
        inputs[0], lat_, stream(), encoder, z_row_u4);
    array topk_ids = cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder);
    array topk_weights = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);
    array gate_w = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    array gate_s = cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);
    array up_w = cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder);
    array up_s = cu::ensure_row_contiguous_aligned<4>(inputs[6], stream(), encoder);
    array down_w = cu::ensure_row_contiguous_aligned<16>(inputs[7], stream(), encoder);
    array down_s = cu::ensure_row_contiguous_aligned<4>(inputs[8], stream(), encoder);

    const int M = static_cast<int>(z.size() / lat_);
    if (M < 1 || M > 16) {
      throw std::invalid_argument("Hy4MoeRoutedFewCuda: M must be in 1..16");
    }
    const int P = M * top_k_;
    const size_t need_bf16 =
        static_cast<size_t>(P) * static_cast<size_t>(ff_ + lat_);

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    auto& ws = get_few_workspace(
        encoder.device().cuda_device(), stream().index);

    if (M == 1) {
      // Direct M==1 path: zero route_plan launch, zero route metadata buffers.
      // Exact even with duplicate topk_ids since each k in 0..top_k owns slot k.
      uint64_t my_gen = 0;
      std::optional<array> md_ref;
      {
        std::lock_guard<std::mutex> lk(ws.mu);
        ws.ensure_locked(/*need_meta_i32=*/0, need_bf16, encoder);
        my_gen = ++ws.use_gen;
        md_ref = *ws.mid_down;
      }
      encoder.add_completed_handler([&ws, my_gen]() {
        std::lock_guard<std::mutex> lk(ws.mu);
        if (ws.use_gen == my_gen) {
          ws.reset_locked();
        }
      });
      const array& mid_down = *md_ref;
      const CUdeviceptr mid_ptr =
          reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(mid_down));
      // ff_ % 64 == 0 => P * ff_ * 2 is 128-byte aligned (>= 16 B uint4 align).
      const CUdeviceptr dn_ptr =
          mid_ptr + static_cast<size_t>(P) * static_cast<size_t>(ff_) * sizeof(uint16_t);

      // 1. Direct Zero-SMEM Gate + Up + Asymmetric Clamped SwiGLU -> mid[top_k, ff]
      {
        encoder.set_input_array(gate_w);
        encoder.set_input_array(gate_s);
        encoder.set_input_array(up_w);
        encoder.set_input_array(up_s);
        encoder.set_input_array(z);
        encoder.set_input_array(topk_ids);
        encoder.set_output_array(mid_down);
        static constexpr const char* kGuM1[2] = {
            "hy4_moe_gate_up_cold<0, true>",
            "hy4_moe_gate_up_cold<1, true>",
        };
        auto k_gu = cu::jit_kernel(
            encoder.device(), hy4_moe_few_source, kGuM1[mode_]);
        cu::KernelArgs args;
        args.append(gate_w);
        args.append(gate_s);
        args.append(up_w);
        args.append(up_s);
        args.append(z);
        args.append(topk_ids);
        args.append<CUdeviceptr>(0);
        args.append<CUdeviceptr>(0);
        args.append<CUdeviceptr>(0);
        args.append<CUdeviceptr>(mid_ptr);
        args.append<int32_t>(lat_);
        args.append<int32_t>(ff_);
        args.append<int32_t>(top_k_);
        args.append<int32_t>(E_);
        args.append<float>(swiglu_limit_);
        args.append<int32_t>(gu_sb_bf16_);
        args.append<int32_t>(z_row_u4);
        encoder.add_kernel_node_raw(
            k_gu, dim3(ff_ / 4, top_k_, 1), dim3(32, 4, 1), {}, 0, args.args());
      }

      // 2. Direct Zero-SMEM Down Projection -> down_out[top_k, lat]
      {
        encoder.set_input_array(down_w);
        encoder.set_input_array(down_s);
        encoder.set_input_array(topk_ids);
        encoder.set_input_array(mid_down);
        encoder.set_output_array(mid_down);
        static constexpr const char* kDnM1[2] = {
            "hy4_moe_down_cold<0, true>",
            "hy4_moe_down_cold<1, true>",
        };
        auto k_dn = cu::jit_kernel(
            encoder.device(), hy4_moe_few_source, kDnM1[mode_]);
        cu::KernelArgs args;
        args.append(down_w);
        args.append(down_s);
        args.append<CUdeviceptr>(mid_ptr);
        args.append(topk_ids);
        args.append<CUdeviceptr>(0);
        args.append<CUdeviceptr>(0);
        args.append<CUdeviceptr>(dn_ptr);
        args.append<int32_t>(lat_);
        args.append<int32_t>(ff_);
        args.append<int32_t>(E_);
        args.append<int32_t>(dn_sb_bf16_);
        encoder.add_kernel_node_raw(
            k_dn, dim3(lat_ / 16, top_k_, 1), dim3(32, 8, 1), {}, 0, args.args());
      }

      // 3. Zero-SMEM Warp-Shuffle Ascending-k Weighted Combine -> out[1, lat]
      {
        encoder.set_input_array(mid_down);
        encoder.set_input_array(topk_weights);
        encoder.set_output_array(out);
        encoder.set_output_array(mid_down);
        auto k_comb = cu::jit_kernel(
            encoder.device(), hy4_moe_few_source, "hy4_moe_combine_few");
        cu::KernelArgs args;
        args.append<CUdeviceptr>(dn_ptr);
        args.append(topk_weights);
        args.append<CUdeviceptr>(0);
        args.append(out);
        args.append<int32_t>(1);
        args.append<int32_t>(lat_);
        args.append<int32_t>(top_k_);
        constexpr int bx = 32;
        const int lat8 = lat_ >> 3;
        encoder.add_kernel_node_raw(
            k_comb, dim3((lat8 + bx - 1) / bx, 1, 1), dim3(bx, 1, 1), {}, 0,
            args.args());
      }
      return;
    }

    // M in 2..=16: Split Cold (cnt <= 2, zero-SMEM streaming) + Hot (cnt >= 3,
    // pure 2-stage cp.async.cg + BF16 TC MMA) using packed persistent workspace.
    const bool allow_hi = (M > 8);
    const int chunk_max = allow_hi ? 16 : 8;
    const int max_cold = std::min(E_, P);
    // Exact bound: every hot expert has c >= 3 and chunk_max >= 8, so
    // ceil(c / chunk_max) <= floor(c / 3) => sum <= P / 3.
    const int max_hot = P / 3;

    const int off_counts = 0;
    const int off_cold_exp = off_counts + 2;
    const int off_cold_meta = off_cold_exp + max_cold;
    const int off_hot_exp = off_cold_meta + max_cold;
    const int off_hot_begin = off_hot_exp + max_hot;
    const int off_hot_end = off_hot_begin + max_hot;
    const int off_sorted_pairs = off_hot_end + max_hot;
    const int off_pair_to_sorted = off_sorted_pairs + P;
    const int need_meta = off_pair_to_sorted + P;

    uint64_t my_gen = 0;
    std::optional<array> meta_ref, md_ref;
    {
      std::lock_guard<std::mutex> lk(ws.mu);
      ws.ensure_locked(need_meta, need_bf16, encoder);
      my_gen = ++ws.use_gen;
      meta_ref = *ws.meta;
      md_ref = *ws.mid_down;
    }
    encoder.add_completed_handler([&ws, my_gen]() {
      std::lock_guard<std::mutex> lk(ws.mu);
      if (ws.use_gen == my_gen) {
        ws.reset_locked();
      }
    });
    const array& meta = *meta_ref;
    const array& mid_down = *md_ref;

    const CUdeviceptr meta_base =
        reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(meta));
    auto meta_ptr = [meta_base](int off) -> CUdeviceptr {
      return meta_base + static_cast<size_t>(off) * sizeof(int32_t);
    };
    const CUdeviceptr counts_ptr = meta_ptr(off_counts);
    const CUdeviceptr cold_exp_ptr = meta_ptr(off_cold_exp);
    const CUdeviceptr cold_meta_ptr = meta_ptr(off_cold_meta);
    const CUdeviceptr hot_exp_ptr = meta_ptr(off_hot_exp);
    const CUdeviceptr hot_begin_ptr = meta_ptr(off_hot_begin);
    const CUdeviceptr hot_end_ptr = meta_ptr(off_hot_end);
    const CUdeviceptr sorted_pairs_ptr = meta_ptr(off_sorted_pairs);
    const CUdeviceptr pair_to_sorted_ptr = meta_ptr(off_pair_to_sorted);

    const CUdeviceptr mid_ptr =
        reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(mid_down));
    const CUdeviceptr dn_ptr =
        mid_ptr + static_cast<size_t>(P) * static_cast<size_t>(ff_) * sizeof(uint16_t);

    // 1. Split Cold/Hot Route Plan (1-CTA deterministic counting sort, zero host sync)
    {
      encoder.set_input_array(topk_ids);
      encoder.set_output_array(meta);
      auto k_rp = cu::jit_kernel(
          encoder.device(), hy4_moe_few_source, "hy4_moe_few_route_plan_split");
      cu::KernelArgs args;
      args.append(topk_ids);
      args.append<CUdeviceptr>(counts_ptr);
      args.append<CUdeviceptr>(cold_exp_ptr);
      args.append<CUdeviceptr>(cold_meta_ptr);
      args.append<CUdeviceptr>(hot_exp_ptr);
      args.append<CUdeviceptr>(hot_begin_ptr);
      args.append<CUdeviceptr>(hot_end_ptr);
      args.append<CUdeviceptr>(sorted_pairs_ptr);
      args.append<CUdeviceptr>(pair_to_sorted_ptr);
      args.append<int32_t>(P);
      args.append<int32_t>(E_);
      args.append<int32_t>(chunk_max);
      encoder.add_kernel_node_raw(
          k_rp, dim3(1, 1, 1), dim3(512, 1, 1), {}, 0, args.args());
    }

    // 2A. Cold Gate + Up + Asymmetric Clamped SwiGLU (cnt <= 2, zero SMEM)
    {
      encoder.set_input_array(gate_w);
      encoder.set_input_array(gate_s);
      encoder.set_input_array(up_w);
      encoder.set_input_array(up_s);
      encoder.set_input_array(z);
      encoder.set_input_array(meta);
      encoder.set_output_array(mid_down);
      static constexpr const char* kGuCold[2] = {
          "hy4_moe_gate_up_cold<0, false>",
          "hy4_moe_gate_up_cold<1, false>",
      };
      auto k_gu_cold = cu::jit_kernel(
          encoder.device(), hy4_moe_few_source, kGuCold[mode_]);
      cu::KernelArgs args;
      args.append(gate_w);
      args.append(gate_s);
      args.append(up_w);
      args.append(up_s);
      args.append(z);
      args.append<CUdeviceptr>(cold_exp_ptr);
      args.append<CUdeviceptr>(cold_meta_ptr);
      args.append<CUdeviceptr>(sorted_pairs_ptr);
      args.append<CUdeviceptr>(counts_ptr);
      args.append<CUdeviceptr>(mid_ptr);
      args.append<int32_t>(lat_);
      args.append<int32_t>(ff_);
      args.append<int32_t>(top_k_);
      args.append<int32_t>(E_);
      args.append<float>(swiglu_limit_);
      args.append<int32_t>(gu_sb_bf16_);
      args.append<int32_t>(z_row_u4);
      encoder.add_kernel_node_raw(
          k_gu_cold, dim3(ff_ / 4, max_cold, 1), dim3(32, 4, 1), {}, 0,
          args.args());
    }

    // 2B. Hot Gate + Up + Asymmetric Clamped SwiGLU (cnt >= 3, BF16 TC MMA)
    if (max_hot > 0) {
      encoder.set_input_array(gate_w);
      encoder.set_input_array(gate_s);
      encoder.set_input_array(up_w);
      encoder.set_input_array(up_s);
      encoder.set_input_array(z);
      encoder.set_input_array(meta);
      encoder.set_output_array(mid_down);
      static constexpr const char* kGuHot[2][2] = {
          {"hy4_moe_gate_up_hot<0, false>", "hy4_moe_gate_up_hot<0, true>"},
          {"hy4_moe_gate_up_hot<1, false>", "hy4_moe_gate_up_hot<1, true>"},
      };
      auto k_gu_hot = cu::jit_kernel(
          encoder.device(), hy4_moe_few_source,
          kGuHot[mode_][allow_hi ? 1 : 0]);
      ensure_smem_carveout(k_gu_hot, 100);
      cu::KernelArgs args;
      args.append(gate_w);
      args.append(gate_s);
      args.append(up_w);
      args.append(up_s);
      args.append(z);
      args.append<CUdeviceptr>(hot_exp_ptr);
      args.append<CUdeviceptr>(hot_begin_ptr);
      args.append<CUdeviceptr>(hot_end_ptr);
      args.append<CUdeviceptr>(sorted_pairs_ptr);
      args.append<CUdeviceptr>(counts_ptr);
      args.append<CUdeviceptr>(mid_ptr);
      args.append<int32_t>(lat_);
      args.append<int32_t>(ff_);
      args.append<int32_t>(top_k_);
      args.append<float>(swiglu_limit_);
      args.append<int32_t>(gu_sb_bf16_);
      args.append<int32_t>(z_row_u4);
      encoder.add_kernel_node_raw(
          k_gu_hot, dim3(ff_ / 16, max_hot, 1), dim3(32, 4, 1), {}, 0,
          args.args());
    }

    // 3A. Cold Down Projection (cnt <= 2, zero SMEM)
    {
      encoder.set_input_array(down_w);
      encoder.set_input_array(down_s);
      encoder.set_input_array(meta);
      encoder.set_input_array(mid_down);
      encoder.set_output_array(mid_down);
      static constexpr const char* kDnCold[2] = {
          "hy4_moe_down_cold<0, false>",
          "hy4_moe_down_cold<1, false>",
      };
      auto k_dn_cold = cu::jit_kernel(
          encoder.device(), hy4_moe_few_source, kDnCold[mode_]);
      cu::KernelArgs args;
      args.append(down_w);
      args.append(down_s);
      args.append<CUdeviceptr>(mid_ptr);
      args.append<CUdeviceptr>(cold_exp_ptr);
      args.append<CUdeviceptr>(cold_meta_ptr);
      args.append<CUdeviceptr>(counts_ptr);
      args.append<CUdeviceptr>(dn_ptr);
      args.append<int32_t>(lat_);
      args.append<int32_t>(ff_);
      args.append<int32_t>(E_);
      args.append<int32_t>(dn_sb_bf16_);
      encoder.add_kernel_node_raw(
          k_dn_cold, dim3(lat_ / 16, max_cold, 1), dim3(32, 8, 1), {}, 0,
          args.args());
    }

    // 3B. Hot Down Projection (cnt >= 3, BF16 TC MMA)
    if (max_hot > 0) {
      encoder.set_input_array(down_w);
      encoder.set_input_array(down_s);
      encoder.set_input_array(meta);
      encoder.set_input_array(mid_down);
      encoder.set_output_array(mid_down);
      static constexpr const char* kDnHot[2][2] = {
          {"hy4_moe_down_hot<0, false>", "hy4_moe_down_hot<0, true>"},
          {"hy4_moe_down_hot<1, false>", "hy4_moe_down_hot<1, true>"},
      };
      auto k_dn_hot = cu::jit_kernel(
          encoder.device(), hy4_moe_few_source,
          kDnHot[mode_][allow_hi ? 1 : 0]);
      ensure_smem_carveout(k_dn_hot, 100);
      cu::KernelArgs args;
      args.append(down_w);
      args.append(down_s);
      args.append<CUdeviceptr>(mid_ptr);
      args.append<CUdeviceptr>(hot_exp_ptr);
      args.append<CUdeviceptr>(hot_begin_ptr);
      args.append<CUdeviceptr>(hot_end_ptr);
      args.append<CUdeviceptr>(counts_ptr);
      args.append<CUdeviceptr>(dn_ptr);
      args.append<int32_t>(lat_);
      args.append<int32_t>(ff_);
      args.append<int32_t>(dn_sb_bf16_);
      encoder.add_kernel_node_raw(
          k_dn_hot, dim3(lat_ / 32, max_hot, 1), dim3(32, 4, 1), {}, 0,
          args.args());
    }

    // 4. Zero-SMEM Warp-Shuffle Ascending-k Weighted Combine -> out[M, lat]
    {
      encoder.set_input_array(mid_down);
      encoder.set_input_array(topk_weights);
      encoder.set_input_array(meta);
      encoder.set_output_array(out);
      // Mark workspace buffers as outputs so MLX's CUDA Graph DAG enforces a
      // happens-before edge before any subsequent layer reuses ws.meta/mid_down.
      encoder.set_output_array(meta);
      encoder.set_output_array(mid_down);
      auto k_comb = cu::jit_kernel(
          encoder.device(), hy4_moe_few_source, "hy4_moe_combine_few");
      cu::KernelArgs args;
      args.append<CUdeviceptr>(dn_ptr);
      args.append(topk_weights);
      args.append<CUdeviceptr>(pair_to_sorted_ptr);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(lat_);
      args.append<int32_t>(top_k_);
      constexpr int bx = 32;
      const int lat8 = lat_ >> 3;
      encoder.add_kernel_node_raw(
          k_comb, dim3((lat8 + bx - 1) / bx, M, 1), dim3(bx, 1, 1), {}, 0,
          args.args());
    }
  }

  DEFINE_NAME(Hy4MoeRoutedFewCuda)

  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4MoeRoutedFewCuda*>(&other);
    return rhs && E_ == rhs->E_ && top_k_ == rhs->top_k_ &&
        lat_ == rhs->lat_ && ff_ == rhs->ff_ && mode_ == rhs->mode_ &&
        swiglu_limit_ == rhs->swiglu_limit_ &&
        gu_sb_bf16_ == rhs->gu_sb_bf16_ && dn_sb_bf16_ == rhs->dn_sb_bf16_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int E_;
  int top_k_;
  int lat_;
  int ff_;
  int mode_;
  float swiglu_limit_;
  int gu_sb_bf16_;
  int dn_sb_bf16_;
};

// ============================================================================
// Primitive 3: Hy4MoeRoutedPrefillCuda (M >= 1, tuned for M = 17..4096)
// ============================================================================
class Hy4MoeRoutedPrefillCuda final : public UnaryPrimitive {
 public:
  Hy4MoeRoutedPrefillCuda(
      Stream stream, int E, int top_k, int lat, int ff, int mode,
      float swiglu_limit, int gu_sb_bf16, int dn_sb_bf16)
      : UnaryPrimitive(stream),
        E_(E),
        top_k_(top_k),
        lat_(lat),
        ff_(ff),
        mode_(mode),
        swiglu_limit_(swiglu_limit),
        gu_sb_bf16_(gu_sb_bf16),
        dn_sb_bf16_(dn_sb_bf16) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4MoeRoutedPrefillCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    int32_t z_row_u4 = lat_ >> 3;
    array z = prepare_z_strided_aligned16(
        inputs[0], lat_, stream(), encoder, z_row_u4);
    array topk_ids = cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder);
    array topk_weights = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);
    array gate_w = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    array gate_s = cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);
    array up_w = cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder);
    array up_s = cu::ensure_row_contiguous_aligned<4>(inputs[6], stream(), encoder);
    array down_w = cu::ensure_row_contiguous_aligned<16>(inputs[7], stream(), encoder);
    array down_s = cu::ensure_row_contiguous_aligned<4>(inputs[8], stream(), encoder);

    const int M = static_cast<int>(z.size() / lat_);
    const int64_t p64 = static_cast<int64_t>(M) * top_k_;
    const int64_t max_tiles64 = (p64 + 63) / 64 + E_;
    if (M < 1 || M > 65535 || p64 > 262144 || max_tiles64 > 65535) {
      throw std::invalid_argument("Hy4MoeRoutedPrefillCuda: unsupported replay shape");
    }
    const int P = static_cast<int>(p64);
    const int max_tiles = static_cast<int>(max_tiles64);
    // Tight proven upper bounds on Zipfian hot (cnt > 32, BM=64) and cold (1..32, BM=32) tiles.
    const int max_hot_tiles = (P >= 33) ? (std::min(E_, P / 33) + P / 64) : 0;
    const int max_cold_tiles = std::min(E_, P);

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    // Reuse one bucketed per-(device, stream) workspace across all sequential
    // MoE layers during prefill; generation-gated completion handler returns
    // buffers to MLX's buffer_cache_ once this stream finishes all queued layers.
    auto& ws = get_prefill_workspace(
        encoder.device().cuda_device(), stream().index);
    uint64_t my_gen = 0;
    std::optional<array> sp_ref, p2s_ref, te_ref, tb_ref, tend_ref, nt_ref,
        mid_ref, dn_ref;
    {
      std::lock_guard<std::mutex> lk(ws.mu);
      ws.ensure_locked(P, max_tiles, ff_, lat_, encoder);
      my_gen = ++ws.use_gen;
      sp_ref = *ws.sorted_pairs;
      p2s_ref = *ws.pair_to_sorted;
      te_ref = *ws.tile_expert;
      tb_ref = *ws.tile_begin;
      tend_ref = *ws.tile_end;
      nt_ref = *ws.num_tiles;
      mid_ref = *ws.mid;
      dn_ref = *ws.down_out;
    }
    encoder.add_completed_handler([&ws, my_gen]() {
      std::lock_guard<std::mutex> lk(ws.mu);
      if (ws.use_gen == my_gen) {
        ws.reset_locked();
      }
    });
    const array& sorted_pairs = *sp_ref;
    const array& pair_to_sorted = *p2s_ref;
    const array& tile_expert = *te_ref;
    const array& tile_begin = *tb_ref;
    const array& tile_end = *tend_ref;
    const array& num_tiles = *nt_ref;
    const array& mid = *mid_ref;
    const array& down_out = *dn_ref;

    // 1. Counting-sort routes + build split hot (BM=64) / cold (BM=32) tile table
    {
      encoder.set_input_array(topk_ids);
      encoder.set_output_array(sorted_pairs);
      encoder.set_output_array(pair_to_sorted);
      encoder.set_output_array(tile_expert);
      encoder.set_output_array(tile_begin);
      encoder.set_output_array(tile_end);
      encoder.set_output_array(num_tiles);
      auto k_sort = cu::jit_kernel(
          encoder.device(), hy4_moe_prefill_source, "hy4_moe_prefill_route_sort");
      cu::KernelArgs args;
      args.append(topk_ids);
      args.append(sorted_pairs);
      args.append(pair_to_sorted);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append<int32_t>(P);
      args.append<int32_t>(E_);
      encoder.add_kernel_node_raw(
          k_sort, dim3(1, 1, 1), dim3(512, 1, 1), {}, 0, args.args());
    }

    // Default BN=64 (4 warps = 128 thr) gives 3-4 CTAs/SM on hot (BM=64) and
    // 4-6 CTAs/SM on cold (BM=32), hiding DRAM/barrier stalls on BW-bound MoE.
    constexpr int gu_bn = 64;
    constexpr int gu_warps = gu_bn / 16;

    // 2A. Fused Gate + Up + Clamped SwiGLU -- hot tiles [0, num_hot) (BM=64)
    if (max_hot_tiles > 0) {
      encoder.set_input_array(gate_w);
      encoder.set_input_array(gate_s);
      encoder.set_input_array(up_w);
      encoder.set_input_array(up_s);
      encoder.set_input_array(z);
      encoder.set_input_array(sorted_pairs);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(mid);
      static constexpr const char* kGuHot[2] = {
          "hy4_moe_prefill_gate_up<0, 64, 64, false>",
          "hy4_moe_prefill_gate_up<1, 64, 64, false>",
      };
      auto k_gu_hot = cu::jit_kernel(
          encoder.device(), hy4_moe_prefill_source, kGuHot[mode_]);
      ensure_smem_carveout(k_gu_hot, 100);
      cu::KernelArgs args;
      args.append(gate_w);
      args.append(gate_s);
      args.append(up_w);
      args.append(up_s);
      args.append(z);
      args.append(sorted_pairs);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(mid);
      args.append<int32_t>(lat_);
      args.append<int32_t>(ff_);
      args.append<int32_t>(top_k_);
      args.append<float>(swiglu_limit_);
      args.append<int32_t>(gu_sb_bf16_);
      args.append<int32_t>(z_row_u4);
      encoder.add_kernel_node_raw(
          k_gu_hot, dim3(ff_ / gu_bn, max_hot_tiles, 1), dim3(32, gu_warps, 1),
          {}, 0, args.args());
    }

    // 2B. Fused Gate + Up + Clamped SwiGLU -- cold tiles [num_hot, total) (BM=32)
    {
      encoder.set_input_array(gate_w);
      encoder.set_input_array(gate_s);
      encoder.set_input_array(up_w);
      encoder.set_input_array(up_s);
      encoder.set_input_array(z);
      encoder.set_input_array(sorted_pairs);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(mid);
      static constexpr const char* kGuCold[2] = {
          "hy4_moe_prefill_gate_up<0, 32, 64, true>",
          "hy4_moe_prefill_gate_up<1, 32, 64, true>",
      };
      auto k_gu_cold = cu::jit_kernel(
          encoder.device(), hy4_moe_prefill_source, kGuCold[mode_]);
      ensure_smem_carveout(k_gu_cold, 100);
      cu::KernelArgs args;
      args.append(gate_w);
      args.append(gate_s);
      args.append(up_w);
      args.append(up_s);
      args.append(z);
      args.append(sorted_pairs);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(mid);
      args.append<int32_t>(lat_);
      args.append<int32_t>(ff_);
      args.append<int32_t>(top_k_);
      args.append<float>(swiglu_limit_);
      args.append<int32_t>(gu_sb_bf16_);
      args.append<int32_t>(z_row_u4);
      encoder.add_kernel_node_raw(
          k_gu_cold, dim3(ff_ / gu_bn, max_cold_tiles, 1), dim3(32, gu_warps, 1),
          {}, 0, args.args());
    }

    constexpr int dn_bn = 64;
    constexpr int dn_warps = dn_bn / 16;

    // 3A. Down Projection -- hot tiles [0, num_hot) (BM=64)
    if (max_hot_tiles > 0) {
      encoder.set_input_array(down_w);
      encoder.set_input_array(down_s);
      encoder.set_input_array(mid);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(down_out);
      static constexpr const char* kDnHot[2] = {
          "hy4_moe_prefill_down<0, 64, 64, false>",
          "hy4_moe_prefill_down<1, 64, 64, false>",
      };
      auto k_dn_hot = cu::jit_kernel(
          encoder.device(), hy4_moe_prefill_source, kDnHot[mode_]);
      ensure_smem_carveout(k_dn_hot, 100);
      cu::KernelArgs args;
      args.append(down_w);
      args.append(down_s);
      args.append(mid);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(down_out);
      args.append<int32_t>(lat_);
      args.append<int32_t>(ff_);
      args.append<int32_t>(dn_sb_bf16_);
      encoder.add_kernel_node_raw(
          k_dn_hot, dim3(lat_ / dn_bn, max_hot_tiles, 1), dim3(32, dn_warps, 1),
          {}, 0, args.args());
    }

    // 3B. Down Projection -- cold tiles [num_hot, total) (BM=32)
    {
      encoder.set_input_array(down_w);
      encoder.set_input_array(down_s);
      encoder.set_input_array(mid);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(down_out);
      static constexpr const char* kDnCold[2] = {
          "hy4_moe_prefill_down<0, 32, 64, true>",
          "hy4_moe_prefill_down<1, 32, 64, true>",
      };
      auto k_dn_cold = cu::jit_kernel(
          encoder.device(), hy4_moe_prefill_source, kDnCold[mode_]);
      ensure_smem_carveout(k_dn_cold, 100);
      cu::KernelArgs args;
      args.append(down_w);
      args.append(down_s);
      args.append(mid);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(down_out);
      args.append<int32_t>(lat_);
      args.append<int32_t>(ff_);
      args.append<int32_t>(dn_sb_bf16_);
      encoder.add_kernel_node_raw(
          k_dn_cold, dim3(lat_ / dn_bn, max_cold_tiles, 1), dim3(32, dn_warps, 1),
          {}, 0, args.args());
    }

    // 4. Deterministic Ascending-k Weighted Combine
    {
      encoder.set_input_array(down_out);
      encoder.set_input_array(topk_weights);
      encoder.set_input_array(pair_to_sorted);
      encoder.set_output_array(out);
      // Mark all workspace buffers as outputs of the final combine kernel so
      // MLX's CUDA Graph dependency tracker enforces a happens-before edge
      // before any subsequent Hy4MoeRoutedPrefillCuda reuses the workspace.
      encoder.set_output_array(sorted_pairs);
      encoder.set_output_array(pair_to_sorted);
      encoder.set_output_array(tile_expert);
      encoder.set_output_array(tile_begin);
      encoder.set_output_array(tile_end);
      encoder.set_output_array(num_tiles);
      encoder.set_output_array(mid);
      encoder.set_output_array(down_out);
      auto k_comb = cu::jit_kernel(
          encoder.device(), hy4_moe_prefill_source, "hy4_moe_prefill_combine");
      cu::KernelArgs args;
      args.append(down_out);
      args.append(topk_weights);
      args.append(pair_to_sorted);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(lat_);
      args.append<int32_t>(top_k_);
      const int bx = (M <= 16) ? 32 : combine_block_x(lat_);
      const int lat8 = lat_ >> 3;
      encoder.add_kernel_node_raw(
          k_comb, dim3((lat8 + bx - 1) / bx, M, 1), dim3(bx, 1, 1), {}, 0,
          args.args());
    }
  }

  DEFINE_NAME(Hy4MoeRoutedPrefillCuda)

  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4MoeRoutedPrefillCuda*>(&other);
    return rhs && E_ == rhs->E_ && top_k_ == rhs->top_k_ &&
        lat_ == rhs->lat_ && ff_ == rhs->ff_ && mode_ == rhs->mode_ &&
        swiglu_limit_ == rhs->swiglu_limit_ &&
        gu_sb_bf16_ == rhs->gu_sb_bf16_ && dn_sb_bf16_ == rhs->dn_sb_bf16_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int E_;
  int top_k_;
  int lat_;
  int ff_;
  int mode_;
  float swiglu_limit_;
  int gu_sb_bf16_;
  int dn_sb_bf16_;
};

}  // namespace
}  // namespace mlx::core::hy4_moe_cuda

extern "C" bool mlx_hy4_moe_router_topk_cuda(
    mlx_array* x_h,
    mlx_array* router_w_h,
    mlx_array* exp_probs_b_opt_h,
    int32_t top_k,
    float route_scale,
    bool norm_topk,
    mlx_array** out_ids,
    mlx_array** out_weights) {
  if (out_ids) *out_ids = nullptr;
  if (out_weights) *out_weights = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_moe_cuda;
    if (!x_h || !router_w_h || !out_ids || !out_weights ||
        !std::isfinite(route_scale) ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& router_w = *reinterpret_cast<array*>(router_w_h);
    if (x.dtype() != bfloat16 || x.ndim() < 2) return false;
    if (router_w.ndim() != 2 ||
        (router_w.dtype() != float32 && router_w.dtype() != bfloat16)) {
      return false;
    }
    const int E = router_w.shape(0);
    const int D = router_w.shape(1);
    if (E < 1 || E > kMaxExperts || D < 32 || x.shape(-1) != D) return false;
    if (top_k < 1 || top_k > kMaxTopK || top_k > E) return false;

    const int64_t m64 = static_cast<int64_t>(x.size()) / D;
    if (m64 < 1 || m64 > 65535) return false;

    std::vector<array> inputs;
    inputs.reserve(exp_probs_b_opt_h ? 3 : 2);
    inputs.push_back(x);
    inputs.push_back(router_w);

    bool has_bias = false;
    if (exp_probs_b_opt_h) {
      const auto& bias = *reinterpret_cast<array*>(exp_probs_b_opt_h);
      if (bias.dtype() != float32 || bias.size() != static_cast<size_t>(E)) {
        return false;
      }
      has_bias = true;
      inputs.push_back(bias);
    }

    Shape out_shape = x.shape();
    out_shape.back() = top_k;
    const bool w_is_bf16 = (router_w.dtype() == bfloat16);
    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        {out_shape, out_shape},
        {uint32, float32},
        std::make_shared<Hy4MoeRouterTopkCuda>(
            stream, E, D, top_k, route_scale, norm_topk, w_is_bf16, has_bias),
        std::move(inputs));

    *out_ids = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_weights = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_moe_router_topk_cuda: %s\n", e.what());
    return false;
  }
}

extern "C" mlx_array* mlx_hy4_moe_routed_few_cuda(
    mlx_array* z_h,
    mlx_array* topk_ids_h,
    mlx_array* topk_weights_h,
    mlx_array* gate_w_h,
    mlx_array* gate_s_h,
    mlx_array* up_w_h,
    mlx_array* up_s_h,
    mlx_array* down_w_h,
    mlx_array* down_s_h,
    int32_t mode,
    float swiglu_limit) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_moe_cuda;
    if (!z_h || !topk_ids_h || !topk_weights_h || !gate_w_h || !gate_s_h ||
        !up_w_h || !up_s_h || !down_w_h || !down_s_h ||
        !std::isfinite(swiglu_limit) ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& z = *reinterpret_cast<array*>(z_h);
    const auto& topk_ids = *reinterpret_cast<array*>(topk_ids_h);
    const auto& topk_weights = *reinterpret_cast<array*>(topk_weights_h);
    const auto& gate_w = *reinterpret_cast<array*>(gate_w_h);
    const auto& gate_s = *reinterpret_cast<array*>(gate_s_h);
    const auto& up_w = *reinterpret_cast<array*>(up_w_h);
    const auto& up_s = *reinterpret_cast<array*>(up_s_h);
    const auto& down_w = *reinterpret_cast<array*>(down_w_h);
    const auto& down_s = *reinterpret_cast<array*>(down_s_h);

    RoutedShapeInfo info{};
    if (!validate_routed_banks(
            z, topk_ids, topk_weights, gate_w, gate_s, up_w, up_s,
            down_w, down_s, mode, /*few_only=*/true, info)) {
      return nullptr;
    }

    auto result = array(
        z.shape(), bfloat16,
        std::make_shared<Hy4MoeRoutedFewCuda>(
            default_stream(Device::gpu), info.E, info.top_k, info.lat, info.ff,
            mode, swiglu_limit, info.gu_sb_bf16, info.dn_sb_bf16),
        {z, topk_ids, topk_weights, gate_w, gate_s, up_w, up_s, down_w, down_s});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_moe_routed_few_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_hy4_moe_routed_prefill_cuda(
    mlx_array* z_h,
    mlx_array* topk_ids_h,
    mlx_array* topk_weights_h,
    mlx_array* gate_w_h,
    mlx_array* gate_s_h,
    mlx_array* up_w_h,
    mlx_array* up_s_h,
    mlx_array* down_w_h,
    mlx_array* down_s_h,
    int32_t mode,
    float swiglu_limit) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_moe_cuda;
    if (!z_h || !topk_ids_h || !topk_weights_h || !gate_w_h || !gate_s_h ||
        !up_w_h || !up_s_h || !down_w_h || !down_s_h ||
        !std::isfinite(swiglu_limit) ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& z = *reinterpret_cast<array*>(z_h);
    const auto& topk_ids = *reinterpret_cast<array*>(topk_ids_h);
    const auto& topk_weights = *reinterpret_cast<array*>(topk_weights_h);
    const auto& gate_w = *reinterpret_cast<array*>(gate_w_h);
    const auto& gate_s = *reinterpret_cast<array*>(gate_s_h);
    const auto& up_w = *reinterpret_cast<array*>(up_w_h);
    const auto& up_s = *reinterpret_cast<array*>(up_s_h);
    const auto& down_w = *reinterpret_cast<array*>(down_w_h);
    const auto& down_s = *reinterpret_cast<array*>(down_s_h);

    RoutedShapeInfo info{};
    if (!validate_routed_banks(
            z, topk_ids, topk_weights, gate_w, gate_s, up_w, up_s,
            down_w, down_s, mode, /*few_only=*/false, info)) {
      return nullptr;
    }
    const int64_t p64 = static_cast<int64_t>(info.M) * info.top_k;
    const int64_t max_tiles = (p64 + 63) / 64 + info.E;
    if (p64 > 262144 || max_tiles > 65535) {
      return nullptr;
    }

    auto result = array(
        z.shape(), bfloat16,
        std::make_shared<Hy4MoeRoutedPrefillCuda>(
            default_stream(Device::gpu), info.E, info.top_k, info.lat, info.ff,
            mode, swiglu_limit, info.gu_sb_bf16, info.dn_sb_bf16),
        {z, topk_ids, topk_weights, gate_w, gate_s, up_w, up_s, down_w, down_s});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_moe_routed_prefill_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" bool mlx_hy4_moe_router_topk_cuda(
    mlx_array* x,
    mlx_array* router_w,
    mlx_array* exp_probs_b_opt,
    int32_t top_k,
    float route_scale,
    bool norm_topk,
    mlx_array** out_ids,
    mlx_array** out_weights) {
  (void)x;
  (void)router_w;
  (void)exp_probs_b_opt;
  (void)top_k;
  (void)route_scale;
  (void)norm_topk;
  if (out_ids) *out_ids = nullptr;
  if (out_weights) *out_weights = nullptr;
  return false;
}

extern "C" mlx_array* mlx_hy4_moe_routed_few_cuda(
    mlx_array* z,
    mlx_array* topk_ids,
    mlx_array* topk_weights,
    mlx_array* gate_w,
    mlx_array* gate_s,
    mlx_array* up_w,
    mlx_array* up_s,
    mlx_array* down_w,
    mlx_array* down_s,
    int32_t mode,
    float swiglu_limit) {
  (void)z;
  (void)topk_ids;
  (void)topk_weights;
  (void)gate_w;
  (void)gate_s;
  (void)up_w;
  (void)up_s;
  (void)down_w;
  (void)down_s;
  (void)mode;
  (void)swiglu_limit;
  return nullptr;
}

extern "C" mlx_array* mlx_hy4_moe_routed_prefill_cuda(
    mlx_array* z,
    mlx_array* topk_ids,
    mlx_array* topk_weights,
    mlx_array* gate_w,
    mlx_array* gate_s,
    mlx_array* up_w,
    mlx_array* up_s,
    mlx_array* down_w,
    mlx_array* down_s,
    int32_t mode,
    float swiglu_limit) {
  (void)z;
  (void)topk_ids;
  (void)topk_weights;
  (void)gate_w;
  (void)gate_s;
  (void)up_w;
  (void)up_s;
  (void)down_w;
  (void)down_s;
  (void)mode;
  (void)swiglu_limit;
  return nullptr;
}

#endif
