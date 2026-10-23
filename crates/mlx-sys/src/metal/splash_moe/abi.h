#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif

// The grouping contract follows Splash 1482f6bbc2abf12375332725302019594150b6cb
// runtime/metal/abi/MoE.h. See THIRD_PARTY_NOTICES (Apache-2.0).
struct SplashMoeTile { uint32_t expert, rows; };
struct SplashMoeTileCount {
  uint32_t count;
  uint32_t gate_up_grid[3];
  uint32_t down_grid[3];
};
struct SplashMoeParams {
  uint32_t rows, hidden, intermediate, experts, top_k, tile_rows;
  // Bit 0: normalize; bits 1/2: BF16 router/shared; bit 3: device row budget.
  // Bit 4: register expert experiment; bit 6: its hybrid column experiment.
  // Bit 5: paired-stage experiment; bit 7 reserved for host scheduling budget.
  // Bits 8..15: test-only stage view; bits 16..18: research dispatch cutoff.
  uint32_t normalize;
  // Six projections: routed gate/up/down, shared gate/up/down. Each format
  // is 4, 5 or 6 (GGML K-quant); bit 8 denotes the lossless Tiled64 layout.
  uint32_t formats[6];
  // Captured opt-in zero-residual source arithmetic. Eval revalidates M8.
  uint32_t source_math;
  // Fixed output meaning, even when source_math falls back on replay.
  uint32_t source_residual;
};
static_assert(sizeof(SplashMoeTile) == 8, "shared tile ABI");
static_assert(sizeof(SplashMoeTileCount) == 28, "shared indirect grid ABI");
static_assert(sizeof(SplashMoeParams) == 60, "shared parameter ABI");
