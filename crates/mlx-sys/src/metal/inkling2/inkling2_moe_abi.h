#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif

// Shared ABI between crates/mlx-sys/src/mlx_inkling2_moe.cpp and
// crates/mlx-sys/src/metal/inkling2/inkling2_moe.metal.
struct Inkling2MoeTile {
  uint32_t expert;
  uint32_t rows;
};

struct Inkling2MoeTileCount {
  uint32_t total_tiles;
  uint32_t simd_tiles;
  uint32_t nax_tiles;
  uint32_t _pad0;
  uint32_t gather_grid[3];
  uint32_t simd_gu_grid[3];
  uint32_t simd_dn_grid[3];
  uint32_t nax_gu_grid[3];
  uint32_t nax_dn_grid[3];
};

struct Inkling2MoeParams {
  uint32_t rows;         // T
  uint32_t hidden;       // D (e.g. 2304, multiple of 128 for NAX down, 64 otherwise)
  uint32_t intermediate; // I (e.g. 1152, multiple of 64)
  uint32_t experts;      // E (e.g. 128, <= 256)
  uint32_t top_k;        // K (e.g. 8, <= 16)
  uint32_t tile_rows;    // 8 (decode/verify) or 32 (prefill)
  float gate_softcap;    // 4.0f
  float up_softcap;      // 25.0f
  // flags:
  //   bit 0: is_q8 (0 = NVFP4 g16 E4M3, 1 = Q8_0 affine g32)
  //   bit 1: has_gu_outer_scale (NVFP4 per-expert F32 outer scale on gate_up)
  //   bit 2: q8_sidecar_is_f32 (0 = BF16 scales/biases, 1 = F32 scales/biases)
  //   bit 3: gu_round_bf16_before_situ (1 = match MLX gather_qmm BF16 boundary)
  uint32_t flags;
  uint32_t col_cutoff;   // tiles with rows <= col_cutoff (<= 8) use 0-smem SIMD; > use 4-SG NAX
  uint32_t max_tiles;    // stride separating simd_list and nax_list inside tile_lists[2*max_tiles]
  uint32_t _pad1;
};

static_assert(sizeof(Inkling2MoeTile) == 8, "Inkling2MoeTile ABI");
static_assert(sizeof(Inkling2MoeTileCount) == 76, "Inkling2MoeTileCount ABI");
static_assert(sizeof(Inkling2MoeParams) == 48, "Inkling2MoeParams ABI");
