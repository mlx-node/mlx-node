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

#if defined(__APPLE__)
#include "mlx/backend/metal/device.h"
#endif

extern "C" {

// Total physical system memory in bytes (Apple Silicon: unified memory
// shared with the GPU). On macOS this matches `device_info()["memory_size"]`
// when Metal is available; we read sysctl directly so the call works on
// pre-MLX-init paths and on no-Metal hosts.
//
// Fallible-FFI contract: returns 0 on success and writes the value through
// `out_value`. Returns -1 if sysctl fails or an exception is caught
// (`out_value` is left untouched on failure). The Rust profile caller
// surfaces `ProfileError::TotalMemoryUnavailable` on -1 (or on a successful
// 0 from a non-macOS host).
//
// Wrapped in a catch-all so any unexpected failure (sysctl currently never
// throws, but this is defense-in-depth: future MLX changes could add a
// memory-size helper that does) cannot unwind across the FFI boundary
// into Rust — that would abort the process via "Rust cannot catch
// foreign exceptions".
int32_t mlx_total_system_memory(uint64_t* out_value) {
#if defined(__APPLE__)
  try {
    size_t memsize = 0;
    size_t length = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &length, nullptr, 0) != 0) {
      return -1;
    }
    if (out_value != nullptr) {
      *out_value = static_cast<uint64_t>(memsize);
    }
    return 0;
  } catch (...) {
    return -1;
  }
#else
  (void)out_value;
  return -1;
#endif
}

// Wrap an existing MTL::Buffer (passed as `void*` for FFI) as an MLX
// `array` view — zero-copy, no host roundtrip.
//
// Used by `LayerKVPool::{key,value}_cache_array` (Phase 3) to expose the
// per-layer K/V pool buffers as MLX-traceable inputs to the compiled
// forward graph.
//
// Inputs:
// - `metal_buffer_ptr`: an `MTL::Buffer*` (the same pointer
//   `mlx_array_get_metal_buffer` returns). The function retains the
//   buffer for the array's lifetime, so the caller does NOT need to
//   keep the original holder alive — though typically callers will
//   anyway because the same buffer is shared with their own write path.
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
// SAFETY / lifetime contract:
// `MTL::Buffer` is reference-counted via Apple's `NS::Referencing`
// (retain/release). We bump the refcount BEFORE constructing the
// array, and the array's `Deleter` lambda releases it on Drop. This
// makes the array view INDEPENDENT of the original holder's
// lifetime — dropping the holder (e.g. the Rust `LayerKVPool`'s
// `metal::Buffer`) decrements the refcount, but Metal won't actually
// free the underlying GPU memory until the array view is also
// dropped. Without retain/release the original "no-op deleter"
// version was vulnerable to GPU use-after-free if the holder
// dropped first.
//
// We construct the `allocator::Buffer{void*}` from the MTL::Buffer*
// the same way MLX uses internally for its own metal-backed arrays
// (see `MetalAllocator::make_buffer`). MLX's `allocator::free`
// would do its own retain/release accounting against the MetalAllocator
// pool — but we bypass that path entirely by supplying our own
// deleter that calls `MTL::Buffer::release()` directly. The buffer
// was made by the caller (typically Rust's `metal::Device::new_buffer`),
// not by the MLX allocator, so MLX's pool accounting must NOT be
// invoked on it.
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

#if defined(__APPLE__)
  // Bump the MTL::Buffer refcount so the array's view holds an independent
  // reference, then capture the same pointer in the deleter to release()
  // when the array is dropped. Without this, the array's deleter was a
  // no-op and the caller's holder (Rust LayerKVPool's metal::Buffer) was
  // the sole owner — dropping the holder while keeping the array view
  // would leave the array pointing at a freed Metal buffer (GPU UAF).
  auto* mtl_buffer = static_cast<MTL::Buffer*>(metal_buffer_ptr);
  mtl_buffer->retain();

  // Releases the array's retained reference. The original holder's
  // separate reference is released by the holder's own destructor.
  // Capture by value (raw pointer) — MTL::Buffer is reference-counted
  // and the retain() above guarantees the pointer stays valid until
  // this lambda runs.
  mlx::core::Deleter retain_release = [mtl_buffer](mlx::core::allocator::Buffer) {
    mtl_buffer->release();
  };
#else
  // Non-macOS: there is no MTL::Buffer to retain/release. We still take
  // the no-op path so the file compiles for `cfg(not(target_os = "macos"))`
  // unit tests, but the function returns null above before reaching
  // this point because `mlx::core::metal::is_available()` is false.
  mlx::core::Deleter retain_release = [](mlx::core::allocator::Buffer) {};
#endif

  // Construct array from existing buffer pointer. `allocator::Buffer{ptr}`
  // wraps the MTL::Buffer* as MLX's allocator::Buffer (same as
  // `MetalAllocator::make_buffer` returns).
  try {
    mlx::core::Shape shape = make_shape(dims, ndim);
    auto buf = mlx::core::allocator::Buffer{metal_buffer_ptr};
    auto* arr = new mlx::core::array(
        buf, std::move(shape), dtype, std::move(retain_release));
    return reinterpret_cast<mlx_array*>(arr);
  } catch (...) {
    // Construction failed AFTER we retained — release the bumped
    // refcount so the buffer doesn't leak. Without this, repeated
    // failure on the same buffer would slowly leak refcounts.
#if defined(__APPLE__)
    mtl_buffer->release();
#endif
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
// working_set_size"]` (populated in `device_info.cpp`).
//
// Fallible-FFI contract (split status to disambiguate "missing key" from
// "Metal failure"):
//   - Returns 0 and writes the actual value through `out_value` on success.
//   - Returns 0 and writes `0` through `out_value` when Metal is
//     unavailable, the `device_info` map does not contain the
//     `max_recommended_working_set_size` entry, or the entry exists but
//     has the wrong type. These are the legitimate "no bound published"
//     cases — schema/version drift in `device_info.cpp` should NOT be
//     conflated with a Metal allocator failure.
//   - Returns -1 ONLY if a C++ exception is caught (`out_value` is left
//     untouched on this path). This is the genuine Metal-failure mode.
//
// The Rust auto-sizer caller maps `Ok(0)` to "no working-set bound
// reported" (falls back to the physical-RAM budget alone) and `-1` to
// `ProfileError::MetalUnavailable`.
int32_t mlx_max_recommended_working_set_size(uint64_t* out_value) {
  if (!mlx::core::metal::is_available()) {
    if (out_value != nullptr) {
      *out_value = 0;
    }
    return 0;
  }
  try {
    auto& info = mlx::core::gpu::device_info(0);
    auto it = info.find("max_recommended_working_set_size");
    if (it == info.end()) {
      // Schema/version drift: `device_info.cpp` no longer publishes this
      // key. Surface as success-with-zero so the auto-sizer falls back
      // to the physical-RAM budget rather than aborting profiling.
      if (out_value != nullptr) {
        *out_value = 0;
      }
      return 0;
    }
    if (auto* p = std::get_if<size_t>(&it->second)) {
      if (out_value != nullptr) {
        *out_value = static_cast<uint64_t>(*p);
      }
      return 0;
    }
    // Wrong variant type — also schema/version drift. Same handling as
    // missing key: success-with-zero, NOT a Metal failure.
    if (out_value != nullptr) {
      *out_value = 0;
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

}  // extern "C"
