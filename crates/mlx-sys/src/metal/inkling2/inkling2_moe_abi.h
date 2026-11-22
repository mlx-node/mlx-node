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
  //   bit 0 (1u):   is_q8 (1 = Q8_0 affine/sym g32)
  //   bit 1 (2u):   has_gu_outer_scale (NVFP4 per-expert F32 outer scale on gate_up)
  //   bit 2 (4u):   q8_sidecar_is_f32 (0 = BF16 scales/biases, 1 = F32 scales/biases)
  //   bit 3 (8u):   gu_round_bf16_before_situ (1 = match MLX gather_qmm BF16 boundary)
  //   bit 4 (16u):  direct_gather (1 = SIMD gate_up reads x[routes[r]] directly, skipping gather)
  //   bit 5 (32u):  interleaved_gate_up (0 = split [gate(0..I), up(I..2I)], 1 = interleaved [2*j, 2*j+1])
  //   bit 6 (64u):  is_mxfp8 (1 = MXFP8 E4M3FN + UE8M0 g32)
  //   bit 7 (128u): q8_symmetric_f16 (1 = Q8_0 symmetric: native Float16 scales, no biases, w = s*(u8^0x80))
  uint32_t flags;
  uint32_t col_cutoff;   // tiles with rows <= col_cutoff (<= 8) use 0-smem SIMD; > use 4-SG NAX
  uint32_t max_tiles;    // stride separating simd_list and nax_list inside tile_lists[2*max_tiles]
  uint32_t _pad1;
};

struct Inkling2MoeRouteGroupParams {
  uint32_t rows;          // T (1 <= T <= 32)
  uint32_t hidden;        // D
  uint32_t intermediate;  // I
  uint32_t experts;       // E (<= 256)
  uint32_t top_k;         // K (<= 16, T * K <= 512)
  uint32_t tile_rows;     // 8
  float route_scale;      // 8.0f
  // flags:
  //   bit 0 (1u):  is_8bit (Q8_0 or MXFP8)
  //   bit 1 (2u):  logits_f32 (1 = F32, 0 = BF16)
  //   bit 2 (4u):  has_bias
  //   bit 3 (8u):  bias_f32
  //   bit 4 (16u): has_down_scale
  //   bit 5 (32u): down_scale_f32
  uint32_t flags;
  uint32_t col_cutoff;
  uint32_t max_tiles;
  uint32_t logits_stride; // row stride in elems
  uint32_t _pad0;
};

struct Inkling2MoePostConvParams {
  uint32_t rows;         // T
  uint32_t hidden;       // D
  uint32_t top_k;        // K (<= 16)
  uint32_t num_seqs;     // N (>= 1)
  uint32_t packed_hist;  // 0 = h0..h3 separate [3, D], 1 = h0 packed [N, 3, D]
  uint32_t res_stride;   // elems per token row in res (D)
  // flags:
  //   bit 0 (1u): w_f32 (1 = post_w is F32, 0 = BF16)
  //   bit 1 (2u): has_res (1 = add FP32 residual)
  //   bit 2 (4u): res_f32 (1 = res is F32, 0 = BF16)
  //   bit 3 (8u): write_tape (1 = write pre-conv moe_out BF16 to tape_out)
  uint32_t flags;
  uint32_t hist_f32_mask; // bit s = 1 if hist[s] is f32
};

static_assert(sizeof(Inkling2MoeTile) == 8, "Inkling2MoeTile ABI");
static_assert(sizeof(Inkling2MoeTileCount) == 72, "Inkling2MoeTileCount ABI");
static_assert(sizeof(Inkling2MoeParams) == 48, "Inkling2MoeParams ABI");
static_assert(sizeof(Inkling2MoeRouteGroupParams) == 48, "Inkling2MoeRouteGroupParams ABI");
static_assert(sizeof(Inkling2MoePostConvParams) == 32, "Inkling2MoePostConvParams ABI");
