// clang-format off
// Keep the source order of float operations (Splash gguf_linear.metal).
#pragma clang fp reassociate(off)
#include "mlx/backend/metal/kernels/utils.h"
#include "kquant_m8_nax.h"

// M = 8 bfloat16 only, so this file is compiled once (no KQUANT_DTYPE). Each
// mode once per weight layout: row-major and Tiled64 ("_t64").
#define instantiate_kquant_m8_nax(mode, group_size, bits, super_ratio, has_min) \
  instantiate_kernel( \
      #mode "_qmm_m8_nax_bfloat16_t_gs_" #group_size "_b_" #bits, \
      kquant_qmm_m8_nax, bfloat16_t, group_size, bits, super_ratio, has_min) \
  instantiate_kernel( \
      #mode "_qmm_m8_nax_t64_bfloat16_t_gs_" #group_size "_b_" #bits, \
      kquant_qmm_m8_nax, bfloat16_t, group_size, bits, super_ratio, has_min, true)

instantiate_kquant_m8_nax(q6k, 16, 6, 16, false)
instantiate_kquant_m8_nax(q4k, 32, 4, 8, true)
instantiate_kquant_m8_nax(q5k, 32, 5, 8, true)
instantiate_kquant_m8_nax(q3k, 16, 3, 16, false)
instantiate_kquant_m8_nax(iq4nl, 32, 4, 1, false)
instantiate_kquant_m8_nax(iq4xs, 32, 4, 8, false)
instantiate_kquant_m8_nax(iq3s, 32, 8, 8, false)
// clang-format on
