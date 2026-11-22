#pragma once

#include "cuda/qwen4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace mlx_node::qwen4_cuda {

constexpr int kDff = 640;
constexpr int kMaxExperts = 512;

// Persistent per-(device, stream) workspace for Qwen4MoePrefillCuda.
// Eliminates 137 MB/layer x N in-flight WDDM cudaMallocAsync allocations
// whenever prompt length M changes (1504 -> 1695 -> 2048) by reusing one
// bucketed workspace across all 48 sequential layers.
struct Qwen4MoePrefillWorkspace {
  int cap_m = 0;
  int cap_p = 0;
  int cap_tiles = 0;
  std::optional<mlx::core::array> sorted_pairs;
  std::optional<mlx::core::array> inv_perm;
  std::optional<mlx::core::array> tile_expert;
  std::optional<mlx::core::array> tile_begin;
  std::optional<mlx::core::array> tile_end;
  std::optional<mlx::core::array> num_tiles;
  std::optional<mlx::core::array> xq;
  std::optional<mlx::core::array> x_scales;
  std::optional<mlx::core::array> x_qsums;
  std::optional<mlx::core::array> mid_q;
  std::optional<mlx::core::array> mid_sc;
  std::optional<mlx::core::array> mid_qsum;
  std::optional<mlx::core::array> down_out;

  void ensure(
      int M, int P, int max_tiles, mlx::core::cu::CommandEncoder& encoder) {
    if (M <= cap_m && P <= cap_p && max_tiles <= cap_tiles && down_out) {
      return;
    }
    auto retire = [&](std::optional<mlx::core::array>& opt) {
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

    sorted_pairs = alloc_persistent({bp}, mlx::core::int32, encoder);
    inv_perm = alloc_persistent({bp}, mlx::core::int32, encoder);
    tile_expert = alloc_persistent({bt}, mlx::core::int32, encoder);
    tile_begin = alloc_persistent({bt}, mlx::core::int32, encoder);
    tile_end = alloc_persistent({bt}, mlx::core::int32, encoder);
    // [0] = total_tiles, [1] = num_hot_tiles (BM=64 hot + BM=32 cold split)
    num_tiles = alloc_persistent({2}, mlx::core::int32, encoder);
    xq = alloc_persistent({bm * kH}, mlx::core::uint8, encoder);
    x_scales = alloc_persistent({bm * (kH / 256)}, mlx::core::float32, encoder);
    x_qsums = alloc_persistent({bm * (kH / 32)}, mlx::core::int16, encoder);
    mid_q = alloc_persistent({bp, kDff}, mlx::core::uint8, encoder);
    mid_sc = alloc_persistent({bp, kDff / 64}, mlx::core::float32, encoder);
    mid_qsum = alloc_persistent({bp, kDff / 32}, mlx::core::int16, encoder);
    down_out = alloc_persistent({bp, kH}, mlx::core::bfloat16, encoder);
  }
};

inline Qwen4MoePrefillWorkspace& get_prefill_workspace(int dev, int stream_idx) {
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
inline int expert_bank_count(
    const mlx::core::array& a,
    mlx::core::Dtype expected_dtype,
    int rows_per_exp,
    int expected_cols) {
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

inline bool check_2d_bank(
    const mlx::core::array& a, mlx::core::Dtype dt, int rows, int cols) {
  return a.dtype() == dt && a.ndim() == 2 && a.shape(0) == rows &&
      a.shape(1) == cols;
}

// Common validation for routed banks and routing metadata.
inline bool validate_routed_inputs(
    const mlx::core::array& x,
    const mlx::core::array& ids,
    const mlx::core::array& scores,
    const mlx::core::array& wg,
    const mlx::core::array& sg,
    const mlx::core::array& bg,
    const mlx::core::array& wu,
    const mlx::core::array& su,
    const mlx::core::array& bu,
    const mlx::core::array& wd,
    const mlx::core::array& sd,
    const mlx::core::array& bd,
    int mode_gu,
    int bits_dn,
    int& out_m,
    int& out_e,
    int& out_top_k) {
  if (x.dtype() != mlx::core::bfloat16 || x.ndim() < 2 || x.shape(-1) != kH) {
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
  if (ids.dtype() != mlx::core::int32 && ids.dtype() != mlx::core::uint32) {
    return false;
  }
  if (scores.dtype() != mlx::core::float32 || ids.size() != scores.size() ||
      ids.size() == 0 || ids.size() % static_cast<size_t>(M) != 0) {
    return false;
  }
  const int top_k = static_cast<int>(ids.size() / static_cast<size_t>(M));
  if (top_k < 1) {
    return false;
  }

  const int gu_w_cols = (mode_gu == 0) ? 320 : 400;
  const int dn_w_cols = (bits_dn == 5) ? 100 : 160;
  const int e0 = expert_bank_count(wg, mlx::core::uint32, kDff, gu_w_cols);
  if (e0 < 1 || e0 > kMaxExperts) {
    return false;
  }
  if (expert_bank_count(sg, mlx::core::uint8, kDff, 160) != e0 ||
      expert_bank_count(bg, mlx::core::float16, kDff, 20) != e0 ||
      expert_bank_count(wu, mlx::core::uint32, kDff, gu_w_cols) != e0 ||
      expert_bank_count(su, mlx::core::uint8, kDff, 160) != e0 ||
      expert_bank_count(bu, mlx::core::float16, kDff, 20) != e0 ||
      expert_bank_count(wd, mlx::core::uint32, kH, dn_w_cols) != e0 ||
      expert_bank_count(sd, mlx::core::float16, kH, 20) != e0 ||
      expert_bank_count(bd, mlx::core::float16, kH, 20) != e0) {
    return false;
  }
  out_m = M;
  out_e = e0;
  out_top_k = top_k;
  return true;
}

}  // namespace mlx_node::qwen4_cuda

#endif  // MLX_NODE_CUDA_ENABLED
