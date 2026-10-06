// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "kquant_nax.h"

// Only qmm_t: the one K-quant path that reaches NAX (a transposed weight, the
// prefill shape). build.rs compiles this file per KQUANT_DTYPE like
// kquant.metal, at the deployment target (>= 26.2, the tensor-ops ABI).
//
// super_ratio is 256 / group_size, the number of sub-blocks a super-block's
// (d, dmin) covers; has_min says whether the sub-scales interleave a minimum;
// kind and shift are the kquant_mode.h traits.

#define instantiate_kquant_aligned_batched(mode, name, type, aligned, batched, group_size, bits, super_ratio, has_min, kind, shift, bm, bn, bk, wm, wn) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned "_batch_" #batched, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift, \
      aligned, \
      batched, \
      bm, \
      bk, \
      bn, \
      wm, \
      wn)

// The Tiled64 layout ("_t64"): a 2-D weight with N % 64 == 0, so aligned and
// unbatched only.
#define instantiate_kquant_nax_tiled(mode, name, type, group_size, bits, super_ratio, has_min, kind, shift, bm, bn, bk, wm, wn) \
  instantiate_kernel( \
      #mode "_" #name "_t64_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_true_batch_0", \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift, \
      true, \
      false, \
      bm, \
      bk, \
      bn, \
      wm, \
      wn, \
      true)

#define instantiate_kquant_nax_all(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_aligned_batched(mode, qmm_t_nax, type, true, 1, group_size, bits, super_ratio, has_min, kind, shift, 64, 64, 64, 2, 2) \
  instantiate_kquant_aligned_batched(mode, qmm_t_nax, type, true, 0, group_size, bits, super_ratio, has_min, kind, shift, 64, 64, 64, 2, 2) \
  instantiate_kquant_aligned_batched(mode, qmm_t_nax, type, false, 1, group_size, bits, super_ratio, has_min, kind, shift, 64, 64, 64, 2, 2) \
  instantiate_kquant_aligned_batched(mode, qmm_t_nax, type, false, 0, group_size, bits, super_ratio, has_min, kind, shift, 64, 64, 64, 2, 2) \
  instantiate_kquant_nax_tiled(mode, qmm_t_nax, type, group_size, bits, super_ratio, has_min, kind, shift, 64, 64, 64, 2, 2)

#define instantiate_kquant_nax_types(type) \
  instantiate_kquant_nax_all(q6k, type, 16, 6, 16, false, KQ_LINEAR, 0) \
  instantiate_kquant_nax_all(q4k, type, 32, 4, 8, true, KQ_LINEAR, 0) \
  instantiate_kquant_nax_all(q5k, type, 32, 5, 8, true, KQ_LINEAR, 0) \
  instantiate_kquant_nax_all(q3k, type, 16, 3, 16, false, KQ_LINEAR, 0) \
  instantiate_kquant_nax_all(q2k, type, 16, 2, 16, true, KQ_LINEAR, 0) \
  instantiate_kquant_nax_all(iq4nl, type, 32, 4, 1, false, KQ_CODEBOOK, 0) \
  instantiate_kquant_nax_all(iq4xs, type, 32, 4, 8, false, KQ_CODEBOOK, 0) \
  instantiate_kquant_nax_all(iq3s, type, 32, 8, 8, false, KQ_INT8, 0)

#if KQUANT_DTYPE == 0
instantiate_kquant_nax_types(float)
#elif KQUANT_DTYPE == 1
instantiate_kquant_nax_types(float16_t)
#elif KQUANT_DTYPE == 2
instantiate_kquant_nax_types(bfloat16_t)
#else
#error "KQUANT_DTYPE must be 0, 1 or 2"
#endif
// clang-format on
