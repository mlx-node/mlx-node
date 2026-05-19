#pragma once
//
// W6.30 — MTPLX `multi3_qmv4` Metal kernel port.
//
// Ports the experimental small-M (M=3) quantized matrix-vector kernel from
// MTPLX's `verify_qmv.py::_multi3_qmv4_kernel` (Python `mx.fast.metal_kernel`)
// to C++ `mlx::fast::metal_kernel`.
//
// The kernel computes `y = x @ w.T` where:
//   - `x`      is `[3, K]` (M=3, contiguous bf16/fp16)
//   - `w`      is affine-quantized 4-bit weights `[N, K/8]` (uint32 packed)
//   - `scales` is `[N, K/group_size]` (bf16/fp16, matches `x` dtype)
//   - `biases` is `[N, K/group_size]` (bf16/fp16, matches `x` dtype)
//   - `y`      is `[3, N]` (output dtype = `x` dtype)
//
// The bandwidth win over stock `mlx::core::quantized_matmul` is that
// dequantized weight loads are reused across the 3 verify rows; stock qmv
// treats each row independently and re-issues weight loads per row.
//
// Eligibility envelope (matches MTPLX `is_multi3_qmv4_eligible`):
//   - `bits == 4`
//   - `group_size ∈ {32, 64, 128}`
//   - `mode == "affine"`
//   - has biases
//   - `K % 512 == 0` and `N % 8 == 0`
//   - M (= `x.shape[-2]`) == 3 exactly
//   - input dtype bf16 or fp16; scales+biases match input dtype
//
// Outside the envelope, callers should dispatch to stock
// `mlx::core::quantized_matmul`. `multi3_qmv4_matmul` falls back internally
// so the wrapper is safe to call as a probe.
//
// Per-(group_size, dtype) kernel compilation is cached via a process-wide
// `std::unordered_map` guarded by a mutex (the MLX equivalent of MTPLX's
// `functools.lru_cache`).
//
// W6.30 scope: pure addition. No existing forward path is rewired by this
// header. The verify-hot-path integration is a follow-up task (W6.37 / W6.31).

#include "mlx_common.h"

namespace qmv_multi3 {

// Returns true iff the (group_size, dtype, K, N) tuple satisfies the eligibility
// envelope above. Mirrors `is_multi3_qmv4_eligible` from MTPLX but is shape-only
// — callers responsible for `bits == 4 && mode == "affine" && has biases`.
inline bool is_eligible(int group_size, mlx::core::Dtype dtype, int K, int N) {
  if (group_size != 32 && group_size != 64 && group_size != 128) return false;
  if (dtype != mlx::core::bfloat16 && dtype != mlx::core::float16) return false;
  if (K <= 0 || N <= 0) return false;
  if (K % 512 != 0) return false;
  if (N % 8 != 0) return false;
  return true;
}

// Compute `y = x @ w.T` with the multi3 qmv4 kernel.
//
// Inputs:
//   x       : [3, K]            (bf16 or fp16, contiguous)
//   w       : [N, K/8]          (uint32 packed affine 4-bit weights)
//   scales  : [N, K/group_size] (matches x dtype)
//   biases  : [N, K/group_size] (matches x dtype)
//   group_size : 32 | 64 | 128
//
// Returns: [3, N] in `x.dtype()` on success.
//
// If the inputs fall outside the kernel envelope (wrong shapes, dtype mismatch,
// missing biases — caller is responsible for affine/4-bit), falls back to
// stock `mlx::core::quantized_matmul` so callers can wire the wrapper into a
// probe without risking invalid runtime behavior.
mlx::core::array multi3_qmv4_matmul(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int group_size);

}  // namespace qmv_multi3

// =====================================================================
// extern "C" FFI surface for the Rust/NAPI microbench harness.
// =====================================================================
extern "C" {

// Run the multi3 qmv4 kernel and return a newly-owned mlx_array handle.
// Returns nullptr on error. Mirrors the C++ wrapper's eligibility/fallback
// semantics — callers can use this as a single entry point that always
// returns a valid `[3, N]` array regardless of whether the kernel path or
// the stock fallback was taken.
struct mlx_array;
mlx_array* mlx_multi3_qmv4_matmul(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int32_t group_size);

// Run stock `mlx::core::quantized_matmul` with affine/4-bit settings.
// Used by the microbench as the baseline. Returns nullptr on error.
mlx_array* mlx_stock_qmv4_matmul(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int32_t group_size);

}  // extern "C"
