#pragma once

// paged_attn.metallib: the paged-attention kernels plus the prebuilt bridge
// kernels (K-quant, segmented SDPA, mixed affine qmv_wide). Metal-only
// (defined in mlx_paged_dispatch.cpp).

#include <filesystem>
#include <string>

#include "mlx/backend/metal/device.h"

namespace mlx::core::fast::paged {

// MLX_PAGED_ATTN_METALLIB, else next to the loaded binary (or its Resources/
// or parent dir). Throws, listing the tried paths, when none exists.
std::filesystem::path paged_attn_metallib_path();

// The library, loaded once per process and cached by MLX.
MTL::Library* get_paged_attn_library(mlx::core::metal::Device& device);

// A prebuilt bridge kernel from the library. Function-constant kernels pass
// the specialization in `hash_name` / `func_consts`. There is no JIT fallback:
// a kernel that is missing or does not load throws "[<tag>] Cannot load
// <family> kernel <kname> from <path> ...".
MTL::ComputePipelineState* get_prebuilt_kernel(
    mlx::core::metal::Device& device,
    const char* tag,
    const char* family,
    const std::string& kname,
    const std::string& hash_name = "",
    const mlx::core::metal::MTLFCList& func_consts = {});

} // namespace mlx::core::fast::paged
