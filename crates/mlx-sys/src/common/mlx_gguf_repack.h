#pragma once

// Shared kernel table, band-planning helpers, and backend entry points for
// GPU repack of GGUF raw blocks into runtime packed arrays.

#include <cstdint>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/dtype.h"
#include "mlx/stream.h"

namespace mlx_node::gguf_repack {

// Indexed by the kernel id the Rust side passes (`GpuRepackKind` x layout).
// Each id must stay at its index; new kernels are appended.
enum class ScaleType { U8, I8, F16 };

struct KernelInfo {
  const char* name;
  uint32_t block_values;    // values per GGUF block (256 or 32)
  uint32_t block_bytes;     // GGUF bytes per block
  uint32_t weight_words;    // .weight uint32 words per block
  uint32_t scale_bytes;     // .scales bytes per block
  uint32_t bias_bytes;      // .biases bytes per block; 0 = no .biases
  uint32_t offset_rule;     // staging byte offset of a band must be a multiple
  uint32_t threads_per_blk; // grid = blocks * threads_per_blk
  bool t64;
  ScaleType scales;
};

inline constexpr KernelInfo kKernels[] = {
    {"mlx_node_gguf_repack_q4k", 256, 144, 32, 16, 4, 16, 8, false, ScaleType::U8},
    {"mlx_node_gguf_repack_q4k_t64", 256, 144, 32, 16, 4, 16, 8, true, ScaleType::U8},
    {"mlx_node_gguf_repack_q5k", 256, 176, 40, 16, 4, 16, 8, false, ScaleType::U8},
    {"mlx_node_gguf_repack_q5k_t64", 256, 176, 40, 16, 4, 16, 8, true, ScaleType::U8},
    {"mlx_node_gguf_repack_q6k", 256, 210, 48, 16, 2, 2, 8, false, ScaleType::I8},
    {"mlx_node_gguf_repack_q6k_t64", 256, 210, 48, 16, 2, 2, 8, true, ScaleType::I8},
    {"mlx_node_gguf_repack_iq4xs", 256, 136, 32, 8, 2, 8, 8, false, ScaleType::I8},
    {"mlx_node_gguf_repack_iq4xs_t64", 256, 136, 32, 8, 2, 8, 8, true, ScaleType::I8},
    {"mlx_node_gguf_repack_q3k", 256, 110, 24, 16, 2, 2, 2, false, ScaleType::I8},
    {"mlx_node_gguf_repack_q3k_t64", 256, 110, 24, 16, 2, 2, 2, true, ScaleType::I8},
    {"mlx_node_gguf_repack_iq4nl", 32, 18, 4, 1, 2, 2, 1, false, ScaleType::I8},
    {"mlx_node_gguf_repack_iq4nl_t64", 32, 18, 4, 1, 2, 2, 1, true, ScaleType::I8},
    {"mlx_node_gguf_repack_iq3s", 256, 110, 24, 16, 2, 2, 8, false, ScaleType::U8},
    {"mlx_node_gguf_repack_iq3s_t64", 256, 110, 24, 16, 2, 2, 8, true, ScaleType::U8},
    {"mlx_node_gguf_repack_q8_0", 32, 34, 8, 2, 0, 2, 1, false, ScaleType::F16},
    {"mlx_node_gguf_repack_mxfp4", 32, 17, 4, 1, 0, 1, 1, false, ScaleType::U8},
    {"mlx_node_gguf_repack_q4_0", 32, 18, 4, 2, 0, 2, 1, false, ScaleType::F16},
    {"mlx_node_gguf_repack_q5_1", 32, 24, 5, 2, 2, 2, 1, false, ScaleType::F16},
    {"mlx_node_gguf_repack_q8_0_bias", 32, 34, 8, 2, 2, 2, 1, false, ScaleType::F16},
    {"mlx_node_gguf_repack_nvfp4", 64, 36, 8, 4, 0, 4, 1, false, ScaleType::U8},
};
inline constexpr int kKernelCount = int(sizeof(kKernels) / sizeof(kKernels[0]));

inline constexpr uint32_t kValuesPerUnit = 32;
inline constexpr uint32_t kTileRows = 64;

// Mirrors `GgufRepackParams` in gguf_repack.metal.
struct GgufRepackParams {
  uint64_t src_off;
  uint32_t nb;
  uint32_t kb;
  uint32_t ku;
  uint32_t row0;
  uint32_t rows;
  uint32_t tile0;
};
static_assert(sizeof(GgufRepackParams) == 32, "GgufRepackParams layout");

struct BandPlan {
  GgufRepackParams params;
  uint64_t src_bytes;
};

// Shared planning and validation helpers (defined in common/mlx_gguf_repack.cpp).
mlx::core::Dtype scale_dtype(const KernelInfo& info);
void expect(bool ok, const std::string& what);
void expect_dense(const mlx::core::array& a, const char* what);
uint64_t dst_rows(const mlx::core::array& a,
                  mlx::core::Dtype dtype,
                  uint64_t cols,
                  const char* what);
BandPlan plan_band(int kernel,
                   uint32_t k,
                   uint32_t row0,
                   uint32_t rows,
                   uint64_t src_offset);

// Metal backend entry points (defined in metal/mlx_gguf_repack_metal.cpp).
bool metal_repack_available();
void metal_repack_validate(int kernel);
std::vector<mlx::core::array> metal_repack_band(
    mlx::core::Stream stream,
    int kernel,
    const GgufRepackParams& params,
    std::vector<mlx::core::Shape> shapes,
    const std::vector<mlx::core::Dtype>& dtypes,
    std::vector<mlx::core::array> inputs);

// CUDA backend entry points (defined in cuda/mlx_gguf_repack_cuda.cpp).
bool cuda_repack_available();
bool cuda_repack_supports_kernel(int kernel) noexcept;
void cuda_repack_validate(int kernel);
std::vector<mlx::core::array> cuda_repack_band(
    mlx::core::Stream stream,
    int kernel,
    const GgufRepackParams& params,
    std::vector<mlx::core::Shape> shapes,
    const std::vector<mlx::core::Dtype>& dtypes,
    std::vector<mlx::core::array> inputs);

}  // namespace mlx_node::gguf_repack
