// Non-Metal stubs for the block-paged custom primitives.
//
// `mlx_paged_ops.cpp` (which DEFINES `mlx::core::fast::paged_kv_write` /
// `paged_attention` / `paged_attention_varlen`) is excluded from the
// non-Metal build because it `#include`s `mlx/backend/metal/device.h` and
// dispatches Metal kernels. But other FFI translation units reference these
// symbols (the paged dispatch entry points in `mlx_paged_dispatch.cpp` and
// their callers), so the Linux link would fail with undefined references
// without a definition.
//
// On a CUDA host the block-paged Metal path is never actually invoked: the
// Rust loaders gate paged-attention use on `mlx_metal_is_available()` (false
// on CUDA), so every forward takes the non-paged eager path. These stubs
// therefore only need to exist for the linker; reaching one at runtime is a
// bug, so they throw loudly rather than silently mis-compute.
//
// This file is active whenever build.rs did not define
// `MLX_NODE_METAL_ENABLED`, including `MLX_DISABLE_METAL=1` on macOS. In a
// Metal build it is an empty TU and the real definitions in
// `mlx_paged_ops.cpp` are used.

#if !defined(MLX_NODE_METAL_ENABLED)

#include <stdexcept>
#include <utility>

#include "mlx_paged_ops.h"

namespace mlx::core::fast {

std::pair<array, array> paged_kv_write(
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    int,
    int,
    int,
    int,
    KvDtype,
    StreamOrDevice) {
  throw std::runtime_error(
      "paged_kv_write: block-paged KV write requires the Metal backend; "
      "the compiled-paged forward path is unavailable on this host (expected "
      "the eager Rust path via mlx_metal_is_available()==false)");
}

array paged_attention(
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    KvDtype,
    StreamOrDevice) {
  throw std::runtime_error(
      "paged_attention: block-paged attention requires the Metal backend; "
      "the compiled-paged forward path is unavailable on this host (expected "
      "the eager Rust path via mlx_metal_is_available()==false)");
}

array paged_attention_varlen(
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    KvDtype,
    StreamOrDevice) {
  throw std::runtime_error(
      "paged_attention_varlen: block-paged varlen attention requires the "
      "Metal backend; the compiled-paged forward path is unavailable on this "
      "host (expected the eager Rust path via mlx_metal_is_available()==false)");
}

} // namespace mlx::core::fast

// Rust still links the macOS adapter methods in a CPU-only macOS build even
// though runtime `mlx_metal_is_available()` gates them off. Provide the same
// conservative C ABI surface as Linux so those unreachable calls fail closed
// instead of leaving the final binary with undefined symbols.
extern "C" {

mlx_array* mlx_paged_attention_forward(
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    uint8_t) {
  return nullptr;
}

mlx_array* mlx_paged_attention_forward_with_route(
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    uint8_t,
    uint8_t) {
  return nullptr;
}

int mlx_paged_grouped_d512_capability(int, int) {
  return 0;
}

mlx_array* mlx_paged_attention_varlen_forward(
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    uint8_t) {
  return nullptr;
}

bool mlx_paged_kv_write_forward(
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    mlx_array*,
    int,
    int,
    int,
    uint8_t,
    mlx_array** out_k_pool,
    mlx_array** out_v_pool) {
  if (out_k_pool) {
    *out_k_pool = nullptr;
  }
  if (out_v_pool) {
    *out_v_pool = nullptr;
  }
  return false;
}

} // extern "C"

#endif // !defined(MLX_NODE_METAL_ENABLED)
