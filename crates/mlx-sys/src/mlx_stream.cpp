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

// Set the wired memory limit and return the old limit
// Wired memory cannot be paged out (important for Metal GPU)
// Uses mlx::core::set_wired_limit (not metal-specific)
size_t mlx_set_wired_limit(size_t limit) {
  try {
    return mlx::core::set_wired_limit(limit);
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in set_wired_limit: " << e.what() << std::endl;
    return 0;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in set_wired_limit" << std::endl;
    return 0;
  }
}

// Get the current wired memory limit
// Note: MLX doesn't have a get_wired_limit function, so we return 0
// The set_wired_limit function returns the old limit when called
size_t mlx_get_wired_limit() {
  // MLX doesn't provide a getter for wired limit
  // Return 0 to indicate no limit is set
  return 0;
}

// Get peak memory usage (works with any backend)
size_t mlx_get_peak_memory() {
  return mlx::core::get_peak_memory();
}

// Get actively used memory in bytes (excludes cached memory)
size_t mlx_get_active_memory() {
  return mlx::core::get_active_memory();
}

// Get cache memory size in bytes
size_t mlx_get_cache_memory() {
  return mlx::core::get_cache_memory();
}

// Reset peak memory counter to zero
void mlx_reset_peak_memory() {
  mlx::core::reset_peak_memory();
}

// Set memory limit (guideline for max memory use)
// Returns the previous limit
size_t mlx_set_memory_limit(size_t limit) {
  return mlx::core::set_memory_limit(limit);
}

// Get current memory limit
size_t mlx_get_memory_limit() {
  return mlx::core::get_memory_limit();
}

// Set cache limit (controls memory pool/cache size)
// Returns the previous limit
// This limits how much memory MLX pre-allocates for caching
size_t mlx_set_cache_limit(size_t limit) {
  return mlx::core::set_cache_limit(limit);
}

// Get the number of bytes in an array without evaluating it
// This is much faster than calling shape() which triggers evaluation
size_t mlx_array_nbytes(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  return static_cast<uint64_t>(arr->nbytes());
}

}  // extern "C"
