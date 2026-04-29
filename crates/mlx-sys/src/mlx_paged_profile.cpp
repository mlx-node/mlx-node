// Profile-run helpers for the vLLM-style auto-sized block pool (Phase 3).
//
// The profile-run sequence in `crates/mlx-paged-attn/src/profile.rs`:
//   1. reset peak memory counter
//   2. dummy forward at (batch=1, seq=max_position_embeddings)
//   3. eval + read peak memory
//   4. compute kv-cache budget = total_memory * util - peak - safety_margin
//   5. divide by per-block bytes -> num_blocks
//
// Steps 1, 3, 4 need three things from MLX/Metal that the existing FFI in
// `mlx_stream.cpp` already exposes (`mlx_get_peak_memory`,
// `mlx_reset_peak_memory`, `mlx_get_active_memory`). Step 4 also needs the
// total system memory — `device_info()["memory_size"]` — which is what this
// file adds.
//
// On non-Metal hosts `mlx_total_system_memory` falls back to the sysctl
// `hw.memsize` value if available, else returns 0 so the profile.rs caller
// can surface a clear error. This is the same `sysctlbyname("hw.memsize")`
// MLX itself uses inside `device_info.cpp`, so the values agree by
// construction; we just expose it through FFI without going through
// MLX's `device_info()` map (which cannot be called when Metal is
// unavailable).

#include "mlx_common.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

extern "C" {

// Total physical system memory in bytes (Apple Silicon: unified memory
// shared with the GPU). On macOS this matches `device_info()["memory_size"]`
// when Metal is available; we read sysctl directly so the call works on
// pre-MLX-init paths and on no-Metal hosts.
//
// Returns 0 on unsupported platforms or if sysctl fails — the Rust profile
// caller treats 0 as "memory APIs unavailable" and surfaces ProfileError.
size_t mlx_total_system_memory() {
#if defined(__APPLE__)
  size_t memsize = 0;
  size_t length = sizeof(memsize);
  if (sysctlbyname("hw.memsize", &memsize, &length, nullptr, 0) != 0) {
    return 0;
  }
  return memsize;
#else
  return 0;
#endif
}

// Wrap an existing MTL::Buffer (passed as `void*` for FFI) as an MLX
// `array` view — zero-copy, no host roundtrip, ownership stays with the
// caller via a no-op deleter.
//
// Used by `LayerKVPool::{key,value}_cache_array` (Phase 3) to expose the
// per-layer K/V pool buffers as MLX-traceable inputs to the compiled
// forward graph. The pool retains ownership; the array's deleter is a
// no-op so MLX doesn't try to `free()` the buffer when the array is
// dropped.
//
// Inputs:
// - `metal_buffer_ptr`: an `MTL::Buffer*` (the same pointer
//   `mlx_array_get_metal_buffer` returns). Must outlive the resulting
//   array.
// - `dims`/`ndim`: shape of the view (caller is responsible for total
//   element count being consistent with the buffer's byte length / dtype
//   size — we don't sanity-check because the caller already knows the
//   pool's shape).
// - `dtype_code`: `BridgeDType` enum value matching the on-buffer element
//   dtype (Fp16/Bf16 for non-FP8 caches, UChar for FP8).
//
// Returns a heap-allocated `mlx_array*` (caller frees via `mlx_array_free`).
// Returns nullptr on Metal-unavailable hosts, null buffer pointers, or
// invalid dtype codes.
//
// SAFETY: this constructs an `allocator::Buffer{void*}` directly from the
// MTL::Buffer pointer — same shape MLX uses internally for its own
// metal-backed arrays (see `MetalAllocator::make_buffer`). The deleter
// is a no-op, so dropping the array does not release the buffer; the
// caller (LayerKVPool) keeps the metal::Buffer alive for the entire
// pool lifetime.
mlx_array* mlx_array_from_metal_buffer_view(
    void* metal_buffer_ptr,
    const int64_t* dims,
    size_t ndim,
    int32_t dtype_code) {
  if (!metal_buffer_ptr || !mlx::core::metal::is_available()) {
    return nullptr;
  }
  if (!dims || ndim == 0) {
    return nullptr;
  }
  mlx::core::Dtype dtype = to_mlx_dtype(dtype_code);

  // No-op deleter — the buffer is owned by the caller (LayerKVPool's
  // `metal::Buffer`). MLX must NOT call `allocator::free` on it.
  mlx::core::Deleter no_op = [](mlx::core::allocator::Buffer) {};

  // Construct array from existing buffer pointer. `allocator::Buffer{ptr}`
  // wraps the MTL::Buffer* as MLX's allocator::Buffer (same as
  // `MetalAllocator::make_buffer` returns).
  try {
    mlx::core::Shape shape = make_shape(dims, ndim);
    auto buf = mlx::core::allocator::Buffer{metal_buffer_ptr};
    auto* arr = new mlx::core::array(
        buf, std::move(shape), dtype, std::move(no_op));
    return reinterpret_cast<mlx_array*>(arr);
  } catch (...) {
    return nullptr;
  }
}

// MLX's max recommended working-set size in bytes (the GPU-visible budget
// MTLDevice reports as `recommendedMaxWorkingSetSize`). On Apple Silicon
// this is normally ~75% of unified memory — vLLM-style auto-sizers use it
// as the *upper* bound when `MLX_KV_MEMORY_UTILIZATION` would otherwise
// over-commit the system.
//
// The value comes from `mlx::core::gpu::device_info()["max_recommended_
// working_set_size"]` (populated in `device_info.cpp`). Returns 0 if Metal
// is unavailable or device_info is empty (e.g. CPU-only build); the Rust
// caller falls back to the sysctl total in that case.
size_t mlx_max_recommended_working_set_size() {
  if (!mlx::core::metal::is_available()) {
    return 0;
  }
  try {
    auto& info = mlx::core::gpu::device_info(0);
    auto it = info.find("max_recommended_working_set_size");
    if (it == info.end()) {
      return 0;
    }
    if (auto* p = std::get_if<size_t>(&it->second)) {
      return *p;
    }
    return 0;
  } catch (...) {
    return 0;
  }
}

}  // extern "C"
