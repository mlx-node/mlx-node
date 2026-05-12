#include "mlx_common.h"

// ============================================================================
// Stream Operations (extern "C" for FFI)
// ============================================================================

namespace {
// Helper to convert device type to MLX Device
mlx::core::Device to_device_helper(int32_t device_type) {
  return device_type == 0 ? mlx::core::Device::cpu : mlx::core::Device::gpu;
}

// Helper to convert MLX Stream to mlx_stream struct
mlx_stream to_mlx_stream_helper(const mlx::core::Stream& s) {
  mlx_stream result;
  result.index = s.index;
  result.device_type = (s.device == mlx::core::Device::cpu) ? 0 : 1;
  return result;
}

// Helper to convert mlx_stream struct to MLX Stream
mlx::core::Stream from_mlx_stream_helper(mlx_stream s) {
  return mlx::core::Stream(s.index, to_device_helper(s.device_type));
}
}  // End helpers namespace

extern "C" {

// Get the default stream for the given device
mlx_stream mlx_default_stream(int32_t device_type) {
  auto device = to_device_helper(device_type);
  auto stream = mlx::core::default_stream(device);
  return to_mlx_stream_helper(stream);
}

// Create a new stream on the given device
mlx_stream mlx_new_stream(int32_t device_type) {
  auto device = to_device_helper(device_type);
  auto stream = mlx::core::new_stream(device);
  return to_mlx_stream_helper(stream);
}

// Set the default stream
void mlx_set_default_stream(mlx_stream stream) {
  try {
    auto s = from_mlx_stream_helper(stream);
    mlx::core::set_default_stream(s);
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in set_default_stream: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in set_default_stream" << std::endl;
  }
}

// Synchronize with the given stream
void mlx_stream_synchronize(mlx_stream stream) {
  try {
    auto s = from_mlx_stream_helper(stream);
    mlx::core::synchronize(s);
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in stream_synchronize: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in stream_synchronize" << std::endl;
  }
}

// ================================================================================
// Metal Operations (Memory Management)
// ================================================================================
//
// Fallible-FFI contract:
//
// Every memory accessor below is wrapped in catch-all because the underlying
// `mlx::core::*` calls resolve through `metal::allocator()` on macOS, which
// lazily constructs a `MetalAllocator` keyed off `device(Device::gpu)`. On a
// host where Metal is unavailable (CPU-only build, no GPU, sandbox without
// IOAccelerator, virtualized environment), constructing the allocator can
// throw a `std::runtime_error` from MLX. Without the catch-all the C++
// exception unwinds across the FFI boundary into Rust and aborts the
// process with "Rust cannot catch foreign exceptions".
//
// Each shim now returns `int32_t` (0 = success, -1 = caught exception)
// and writes its measurement into a caller-supplied out-pointer. This
// replaces the old "ambiguous sentinel `0`" contract: a real measurement
// of zero bytes is now distinguishable from a caught exception. The Rust
// callers translate `-1` into `ProfileError::MetalUnavailable` (auto-
// sizer) or fall back to a static heuristic (memory stats, cache limit
// coordinator).

// Check if Metal backend is available
bool mlx_metal_is_available() {
  try {
    return mlx::core::metal::is_available();
  } catch (...) {
    return false;
  }
}

// Get Metal device information as JSON string
// Returns a JSON string with device properties like max_recommended_working_set_size
const char* mlx_metal_device_info() {
  // Static buffer to hold the JSON string
  static std::string info_json;

  if (!mlx::core::metal::is_available()) {
    info_json = "{\"available\": false}";
    return info_json.c_str();
  }

  try {
    const auto& device_info = mlx::core::gpu::device_info();

    // Build JSON string manually
    std::ostringstream json;
    json << "{";
    json << "\"available\": true";

    // Get max_recommended_working_set_size (this is the key we need for wired_limit)
    auto it = device_info.find("max_recommended_working_set_size");
    if (it != device_info.end()) {
      // The value is a variant<string, size_t>, extract size_t
      if (const auto* val = std::get_if<size_t>(&it->second)) {
        json << ", \"max_recommended_working_set_size\": " << *val;
      }
    }

    json << "}";

    info_json = json.str();
    return info_json.c_str();
  } catch (const std::exception& e) {
    info_json = "{\"available\": true, \"error\": \"" + std::string(e.what()) + "\"}";
    return info_json.c_str();
  }
}

// Set the wired memory limit. Writes the previous limit through `out_old_limit`
// (which may be null). Wired memory cannot be paged out (important for Metal
// GPU). Uses mlx::core::set_wired_limit (not metal-specific).
//
// Returns 0 on success, -1 on caught exception (out_old_limit is left
// untouched on failure).
int32_t mlx_set_wired_limit(uint64_t limit, uint64_t* out_old_limit) {
  try {
    size_t prev = mlx::core::set_wired_limit(static_cast<size_t>(limit));
    if (out_old_limit != nullptr) {
      *out_old_limit = static_cast<uint64_t>(prev);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in set_wired_limit: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in set_wired_limit" << std::endl;
    return -1;
  }
}

// Get peak memory usage (works with any backend). Writes the result through
// `out_value` on success. Returns 0 on success, -1 on caught exception
// (out_value is left untouched on failure).
int32_t mlx_get_peak_memory(uint64_t* out_value) {
  try {
    size_t v = mlx::core::get_peak_memory();
    if (out_value != nullptr) {
      *out_value = static_cast<uint64_t>(v);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in get_peak_memory: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in get_peak_memory" << std::endl;
    return -1;
  }
}

// Get actively used memory in bytes (excludes cached memory). Returns 0 on
// success, -1 on caught exception. See `mlx_get_peak_memory` for the
// fallible-FFI contract rationale.
int32_t mlx_get_active_memory(uint64_t* out_value) {
  try {
    size_t v = mlx::core::get_active_memory();
    if (out_value != nullptr) {
      *out_value = static_cast<uint64_t>(v);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in get_active_memory: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in get_active_memory" << std::endl;
    return -1;
  }
}

// Get cache memory size in bytes. Returns 0 on success, -1 on caught
// exception. See `mlx_get_peak_memory` for the fallible-FFI contract
// rationale.
int32_t mlx_get_cache_memory(uint64_t* out_value) {
  try {
    size_t v = mlx::core::get_cache_memory();
    if (out_value != nullptr) {
      *out_value = static_cast<uint64_t>(v);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in get_cache_memory: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in get_cache_memory" << std::endl;
    return -1;
  }
}

// Reset peak memory counter to zero. Returns 0 on success, -1 on caught
// exception. See `mlx_get_peak_memory` for the fallible-FFI contract
// rationale. On no-Metal hosts the underlying `mlx::core::reset_peak_memory()`
// can throw while constructing the lazy `metal::allocator()`; the caller is
// expected to gate this behind a `mlx_metal_is_available()` check on no-
// Metal hosts to avoid the cerr spam.
int32_t mlx_reset_peak_memory() {
  try {
    mlx::core::reset_peak_memory();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in reset_peak_memory: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in reset_peak_memory" << std::endl;
    return -1;
  }
}

// Set memory limit (guideline for max memory use). Writes the previous
// limit through `out_old_limit` (which may be null). Returns 0 on success,
// -1 on caught exception.
int32_t mlx_set_memory_limit(uint64_t limit, uint64_t* out_old_limit) {
  try {
    size_t prev = mlx::core::set_memory_limit(static_cast<size_t>(limit));
    if (out_old_limit != nullptr) {
      *out_old_limit = static_cast<uint64_t>(prev);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in set_memory_limit: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in set_memory_limit" << std::endl;
    return -1;
  }
}

// Get current memory limit. Returns 0 on success, -1 on caught exception.
// See `mlx_get_peak_memory` for the fallible-FFI contract rationale.
int32_t mlx_get_memory_limit(uint64_t* out_value) {
  try {
    size_t v = mlx::core::get_memory_limit();
    if (out_value != nullptr) {
      *out_value = static_cast<uint64_t>(v);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in get_memory_limit: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in get_memory_limit" << std::endl;
    return -1;
  }
}

// Set cache limit (controls memory pool/cache size). Writes the previous
// limit through `out_old_limit` (which may be null). This limits how much
// memory MLX pre-allocates for caching. Returns 0 on success, -1 on
// caught exception.
int32_t mlx_set_cache_limit(uint64_t limit, uint64_t* out_old_limit) {
  try {
    size_t prev = mlx::core::set_cache_limit(static_cast<size_t>(limit));
    if (out_old_limit != nullptr) {
      *out_old_limit = static_cast<uint64_t>(prev);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in set_cache_limit: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in set_cache_limit" << std::endl;
    return -1;
  }
}

// Get the number of bytes in an array without evaluating it
// This is much faster than calling shape() which triggers evaluation
size_t mlx_array_nbytes(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  return static_cast<uint64_t>(arr->nbytes());
}

}  // extern "C"
