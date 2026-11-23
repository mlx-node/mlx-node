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

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "mlx_paged_ops.h"

namespace mlx::core::fast {

std::array<array, 4> paged_kv_write(
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
    StreamOrDevice,
    bool) {
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

mlx_array* mlx_paged_attention_forward_with_plan(
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
    uint8_t, uint32_t grouped_stripes, mlx_array*) {
  (void)grouped_stripes;
  return nullptr;
}

uint32_t mlx_paged_grouped_d128_max_stripes(uint32_t, uint32_t) { return 0; }

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
    uint8_t,
    mlx_array*) {
  return nullptr;
}

bool mlx_paged_query_codes_forward(
    mlx_array*, mlx_array*, uint32_t, float, mlx_array** out_codes,
    mlx_array** out_score_scales) {
  if (out_codes) *out_codes = nullptr;
  if (out_score_scales) *out_score_scales = nullptr;
  return false;
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
    bool,
    mlx_array** out_k_pool,
    mlx_array** out_v_pool,
    mlx_array** out_k_scale,
    mlx_array** out_v_scale) {
  for (mlx_array** out : {out_k_pool, out_v_pool, out_k_scale, out_v_scale}) {
    if (out) {
      *out = nullptr;
    }
  }
  return false;
}

bool mlx_paged_prefill_attention_forward(
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
    int,
    int,
    int,
    int,
    uint8_t,
    bool,
    float,
    mlx_array** out_attention,
    mlx_array** out_workspace) {
  if (out_attention) {
    *out_attention = nullptr;
  }
  if (out_workspace) {
    *out_workspace = nullptr;
  }
  return false;
}

bool mlx_paged_prefill_mpp_builds() { return false; }

void mlx_paged_prefill_test_probe_reset() {}

uint64_t mlx_paged_prefill_test_probe_count(int) { return 0; }

uint32_t mlx_paged_prefill_test_last_splits() { return 0; }

uint32_t mlx_paged_prefill_test_last_max_splits() { return 0; }

// CPU-only / non-Metal bridge definitions for the three entry points normally
// supplied by mlx_paged_profile.cpp. That translation unit includes
// Metal/Metal.hpp and is therefore intentionally excluded when
// MLX_DISABLE_METAL=1 (and on non-Apple builds). Keep these stubs in a
// translation unit that consumes only MLX's public, backend-neutral headers so
// non-Metal builds can link.
int32_t mlx_total_system_memory(uint64_t* out_value) {
#if defined(__APPLE__)
  size_t memsize = 0;
  size_t length = sizeof(memsize);
  if (sysctlbyname("hw.memsize", &memsize, &length, nullptr, 0) != 0) {
    return -1;
  }
  if (out_value != nullptr) {
    *out_value = static_cast<uint64_t>(memsize);
  }
  return 0;
#else
  (void)out_value;
  return -1;
#endif
}

int32_t mlx_max_recommended_working_set_size(uint64_t* out_value) {
  if (out_value != nullptr) {
    *out_value = 0;
  }
  return 0;
}

mlx_array* mlx_array_from_metal_buffer_view(
    void* metal_buffer_ptr,
    const int64_t* dims,
    size_t ndim,
    int32_t dtype_code) {
  (void)metal_buffer_ptr;
  (void)dims;
  (void)ndim;
  (void)dtype_code;
  return nullptr;
}

} // extern "C"

#endif // !defined(MLX_NODE_METAL_ENABLED)
