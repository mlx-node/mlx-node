// Non-Apple stubs for the block-paged custom primitives.
//
// `mlx_paged_ops.cpp` (which DEFINES `mlx::core::fast::paged_kv_write` /
// `paged_attention` / `paged_attention_varlen`) is excluded from the
// non-Apple build because it `#include`s `mlx/backend/metal/device.h` and
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
// This file is compiled ONLY on non-Apple hosts (the whole body is under
// `#if !defined(__APPLE__)`); on macOS it is an empty TU and the real
// definitions in `mlx_paged_ops.cpp` are used.

#if !defined(__APPLE__)

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

#endif // !defined(__APPLE__)
