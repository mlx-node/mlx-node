// clang-format off
// Keep the source order of float operations (Splash gguf_linear.metal).
#pragma clang fp reassociate(off)
#include "mlx/backend/metal/kernels/utils.h"
#include "kquant_m8_nax.h"

// bfloat16 only, so this file is compiled once (no KQUANT_DTYPE). Every mode
// in the Tiled64 layout ("_t64") at the 8-, 16- and 32-row tiers (M <= 32);
// row-major only the 8-row tier of the modes the dispatcher routes there
// (kernels::m8_nax_row_major_mode: q3k, q2k, iq4nl and the seven grid modes,
// none of which qmv_sg8 decodes).
// (group_size, bits, super_ratio, has_min, kind, scale_shift) per mode as in
// kquant.metal.
#define instantiate_kquant_m8_nax_tiled(mode, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_qmm_m8_nax_t64_bfloat16_t_gs_" #group_size "_b_" #bits, \
      kquant_qmm_m8_nax, bfloat16_t, group_size, bits, super_ratio, has_min, kind, shift, true, 8) \
  instantiate_kernel( \
      #mode "_qmm_m16_nax_t64_bfloat16_t_gs_" #group_size "_b_" #bits, \
      kquant_qmm_m8_nax, bfloat16_t, group_size, bits, super_ratio, has_min, kind, shift, true, 16) \
  instantiate_kernel( \
      #mode "_qmm_m32_nax_t64_bfloat16_t_gs_" #group_size "_b_" #bits, \
      kquant_qmm_m8_nax, bfloat16_t, group_size, bits, super_ratio, has_min, kind, shift, true, 32)

#define instantiate_kquant_m8_nax(mode, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_qmm_m8_nax_bfloat16_t_gs_" #group_size "_b_" #bits, \
      kquant_qmm_m8_nax, bfloat16_t, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_m8_nax_tiled(mode, group_size, bits, super_ratio, has_min, kind, shift)

instantiate_kquant_m8_nax_tiled(q6k, 16, 6, 16, false, KQ_LINEAR, 0)
instantiate_kquant_m8_nax_tiled(q4k, 32, 4, 8, true, KQ_LINEAR, 0)
instantiate_kquant_m8_nax_tiled(q5k, 32, 5, 8, true, KQ_LINEAR, 0)
instantiate_kquant_m8_nax(q3k, 16, 3, 16, false, KQ_LINEAR, 0)
instantiate_kquant_m8_nax(q2k, 16, 2, 16, true, KQ_LINEAR, 0)
instantiate_kquant_m8_nax(iq4nl, 32, 4, 1, false, KQ_CODEBOOK, 0)
instantiate_kquant_m8_nax_tiled(iq4xs, 32, 4, 8, false, KQ_CODEBOOK, 0)
instantiate_kquant_m8_nax_tiled(iq3s8, 32, 8, 8, false, KQ_INT8, 0)
instantiate_kquant_m8_nax(iq2xxs, 32, 1, 8, false, KQ_GRID_IQ2XXS, -3)
instantiate_kquant_m8_nax(iq2xs, 32, 2, 8, false, KQ_GRID_IQ2XS, -3)
instantiate_kquant_m8_nax(iq2s, 32, 2, 8, false, KQ_GRID_IQ2S, -3)
instantiate_kquant_m8_nax(iq3xxs, 32, 2, 8, false, KQ_GRID_IQ3XXS, -2)
instantiate_kquant_m8_nax(iq1s, 32, 1, 8, false, KQ_GRID_IQ1S, -3)
instantiate_kquant_m8_nax(iq1m, 32, 1, 8, false, KQ_GRID_IQ1M, -3)
instantiate_kquant_m8_nax(iq3s, 32, 3, 8, false, KQ_GRID_IQ3S, 0)
// The MLX affine modes exist in the Tiled64 layout only (row-major affine
// stays on MLX's own kernels).
instantiate_kquant_m8_nax_tiled(a4g64, 64, 4, 4, false, KQ_AFFINE, 0)
instantiate_kquant_m8_nax_tiled(a8g64, 64, 8, 4, false, KQ_AFFINE, 0)
// clang-format on
