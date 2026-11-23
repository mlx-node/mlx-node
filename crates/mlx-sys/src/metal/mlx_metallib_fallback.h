#pragma once

// A prebuilt libmlx.a bakes the producer's METAL_PATH, which no consumer
// has, so when nothing is colocated with the binary MLX finds no metallib
// and traps in Device::Device(). Cargo >= 1.100 runs test binaries from
// build/<pkg>/<hash>/out/, where build.rs cannot place a copy, so point MLX
// at this build's own <OUT_DIR>/lib/mlx.metallib before any device exists.
#ifdef MLX_NODE_MLX_METALLIB

#include <dlfcn.h>

#include <filesystem>
#include <system_error>

#include "mlx/backend/metal/metal.h"

namespace mlx_node_detail {

inline bool use_built_mlx_metallib_when_none_is_colocated() {
  Dl_info info;
  if (!dladdr(
          reinterpret_cast<void*>(
              &use_built_mlx_metallib_when_none_is_colocated),
          &info)) {
    return false;
  }
  const auto bin_dir = std::filesystem::path(info.dli_fname).parent_path();
  std::error_code ec;
  // MLX's colocated lookups (device.cpp load_default_library) keep priority.
  for (const char* rel :
       {"mlx.metallib", "Resources/mlx.metallib", "Resources/default.metallib"}) {
    if (std::filesystem::exists(bin_dir / rel, ec)) {
      return false;
    }
  }
  const std::filesystem::path built(MLX_NODE_MLX_METALLIB);
  if (!mlx::core::metal::get_metallib_path().empty() ||
      !std::filesystem::exists(built, ec)) {
    return false;
  }
  mlx::core::metal::set_metallib_path(built.string());
  return true;
}

inline const bool built_mlx_metallib_fallback =
    use_built_mlx_metallib_when_none_is_colocated();

} // namespace mlx_node_detail

#endif
