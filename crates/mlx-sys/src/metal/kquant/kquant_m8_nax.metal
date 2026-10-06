// clang-format off
// Keep the source order of float operations (Splash gguf_linear.metal).
#pragma clang fp reassociate(off)
#include "mlx/backend/metal/kernels/utils.h"
#include "kquant_m8_nax.h"

// M = 8 bfloat16 only, so this file is compiled once (no KQUANT_DTYPE). Every
// mode in the Tiled64 layout ("_t64"); row-major only the modes the
// dispatcher routes there (kernels::m8_nax_row_major_mode: q3k, q2k, iq4nl).
// (group_size, bits, super_ratio, has_min, kind, scale_shift) per mode as in
// kquant.metal.
#define instantiate_kquant_m8_nax_tiled(mode, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_qmm_m8_nax_t64_bfloat16_t_gs_" #group_size "_b_" #bits, \
      kquant_qmm_m8_nax, bfloat16_t, group_size, bits, super_ratio, has_min, kind, shift, true)

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
instantiate_kquant_m8_nax_tiled(iq3s, 32, 8, 8, false, KQ_INT8, 0)
// clang-format on
