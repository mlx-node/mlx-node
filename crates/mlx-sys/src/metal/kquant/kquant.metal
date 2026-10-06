// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "kquant.h"

// Every kernel mlx_kquant_metal.cpp can request, prebuilt into
// paged_attn.metallib. The host names must match kquant::kernels in
// mlx_kquant_metal.cpp character for character; kquant_metallib_names checks
// both directions. build.rs compiles this file once per KQUANT_DTYPE
// (0 float, 1 float16_t, 2 bfloat16_t) to build the three in parallel.
//
// q6k/q3k are symmetric sub-blocks of 16, q2k an asymmetric sub-block of 16,
// q4k/q5k asymmetric sub-blocks of 32; the IQ modes carry a codebook (iq4nl,
// iq4xs) or int8 values (iq3s); the grid modes (iq1s, iq1m, iq2xxs, iq2xs,
// iq2s, iq3xxs) grid indices whose `bits` is the unit's word count and whose
// scale carries the 2^shift of kquant_grid.h. super_ratio is 256 /
// group_size, the number of sub-blocks a super-block's (d, dmin) covers.
//
// There is no quantize: K-quants are only ever consumed, and producing one
// needs ggml's make_qkx2_quants search. There is no qmv_quad either: the
// dispatcher sends K of 64 or 128 to qmv.

#define instantiate_kquant(mode, name, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift)

#define instantiate_kquant_batched(mode, name, type, batched, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift, \
      batched)

#define instantiate_kquant_aligned(mode, name, type, aligned, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift, \
      aligned)

#define instantiate_kquant_aligned_batched(mode, name, type, aligned, batched, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift, \
      aligned, \
      batched)

#define instantiate_kquant_wide(mode, name, type, vecs_per_tg, k_lanes, batched, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift, \
      vecs_per_tg, \
      k_lanes, \
      batched)

#define instantiate_kquant_split_k(mode, name, type, split_k, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift, \
      split_k)

#define instantiate_kquant_rhs(mode, name, type, bm, bn, bk, wm, wn, transpose, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      kquant_gather_qmm_rhs, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift, \
      bm, \
      bn, \
      bk, \
      wm, \
      wn, \
      transpose)

#define instantiate_kquant_batched_wrap(mode, name, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_batched(mode, name, type, 1, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_batched(mode, name, type, 0, group_size, bits, super_ratio, has_min, kind, shift)

#define instantiate_kquant_all_batched(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_batched_wrap(mode, qmv_fast, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_batched_wrap(mode, qmv, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_batched_wrap(mode, qvm, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_batched_wrap(mode, qmm_n, type, group_size, bits, super_ratio, has_min, kind, shift)

#define instantiate_kquant_all_single(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant(mode, dequantize, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant(mode, gather_qmv_fast, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant(mode, gather_qmv, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant(mode, gather_qvm, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant(mode, gather_qmm_n, type, group_size, bits, super_ratio, has_min, kind, shift)

#define instantiate_kquant_all_aligned(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_aligned(mode, gather_qmm_t, type, true, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_aligned(mode, gather_qmm_t, type, false, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_aligned(mode, qmm_t_splitk, type, true, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_aligned(mode, qmm_t_splitk, type, false, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_aligned_batched(mode, qmm_t, type, true, 1, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_aligned_batched(mode, qmm_t, type, true, 0, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_aligned_batched(mode, qmm_t, type, false, 1, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_aligned_batched(mode, qmm_t, type, false, 0, group_size, bits, super_ratio, has_min, kind, shift)

// vecs_per_tg (input-vector tile) 2..8, k_lanes 8 as the affine family uses.
#define instantiate_kquant_wide_wrap(mode, type, vecs_per_tg, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide(mode, qmv_wide, type, vecs_per_tg, 8, 0, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide(mode, qmv_wide, type, vecs_per_tg, 8, 1, group_size, bits, super_ratio, has_min, kind, shift)

#define instantiate_kquant_all_wide(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_wrap(mode, type, 2, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_wrap(mode, type, 3, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_wrap(mode, type, 4, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_wrap(mode, type, 5, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_wrap(mode, type, 6, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_wrap(mode, type, 7, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_wrap(mode, type, 8, group_size, bits, super_ratio, has_min, kind, shift)

#define instantiate_kquant_all_splitk(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_split_k(mode, qvm_split_k, type, 8, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_split_k(mode, qvm_split_k, type, 32, group_size, bits, super_ratio, has_min, kind, shift)

#define instantiate_kquant_all_rhs(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_rhs(mode, gather_qmm_rhs_nt, type, 16, 32, 32, 1, 2, true, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_rhs(mode, gather_qmm_rhs_nn, type, 16, 32, 32, 1, 2, false, group_size, bits, super_ratio, has_min, kind, shift)

// The Tiled64 weight layout ("_t64", mlx_kquant.h): a 2-D weight read as
// x @ w.T with N % 64 == 0, so the transposed, aligned, unbatched kernels
// only. M = 1 takes qmv_t64, so qmv_wide starts at nv_2.
#define instantiate_kquant_wide_tiled(mode, type, vecs_per_tg, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_qmv_wide_t64_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_8_batch_0", \
      kquant_qmv_wide, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      kind, \
      shift, \
      vecs_per_tg, \
      8, \
      0, \
      true)

#define instantiate_kquant_all_tiled(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_qmv_t64_" #type "_gs_" #group_size "_b_" #bits "_ks_8", \
      kquant_qmv_t64, type, group_size, bits, super_ratio, has_min, kind, shift, 8) \
  instantiate_kernel( \
      #mode "_qmv_t64_" #type "_gs_" #group_size "_b_" #bits "_ks_16", \
      kquant_qmv_t64, type, group_size, bits, super_ratio, has_min, kind, shift, 16) \
  instantiate_kquant_wide_tiled(mode, type, 2, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_tiled(mode, type, 3, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_tiled(mode, type, 4, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_tiled(mode, type, 5, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_tiled(mode, type, 6, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_tiled(mode, type, 7, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_wide_tiled(mode, type, 8, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kernel( \
      #mode "_qmm_t_t64_" #type "_gs_" #group_size "_b_" #bits "_alN_true_batch_0", \
      kquant_qmm_t, type, group_size, bits, super_ratio, has_min, kind, shift, true, 0, 32, 32, 32, true) \
  instantiate_kernel( \
      #mode "_qmm_t_splitk_t64_" #type "_gs_" #group_size "_b_" #bits "_alN_true", \
      kquant_qmm_t_splitk, type, group_size, bits, super_ratio, has_min, kind, shift, true, 32, 32, 32, true)

#define instantiate_kquant_modes(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_all_single(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_all_batched(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_all_aligned(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_all_wide(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_all_splitk(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_all_rhs(mode, type, group_size, bits, super_ratio, has_min, kind, shift) \
  instantiate_kquant_all_tiled(mode, type, group_size, bits, super_ratio, has_min, kind, shift)

// (group_size, bits, super_ratio, has_min, kind, scale_shift) per mode:
// kquant_mode.h; must agree with kquant::Mode's traits in mlx_kquant.h.
#define instantiate_kquant_types(type) \
  instantiate_kquant_modes(q6k, type, 16, 6, 16, false, KQ_LINEAR, 0) \
  instantiate_kquant_modes(q4k, type, 32, 4, 8, true, KQ_LINEAR, 0) \
  instantiate_kquant_modes(q5k, type, 32, 5, 8, true, KQ_LINEAR, 0) \
  instantiate_kquant_modes(q3k, type, 16, 3, 16, false, KQ_LINEAR, 0) \
  instantiate_kquant_modes(q2k, type, 16, 2, 16, true, KQ_LINEAR, 0) \
  instantiate_kquant_modes(iq4nl, type, 32, 4, 1, false, KQ_CODEBOOK, 0) \
  instantiate_kquant_modes(iq4xs, type, 32, 4, 8, false, KQ_CODEBOOK, 0) \
  instantiate_kquant_modes(iq3s, type, 32, 8, 8, false, KQ_INT8, 0) \
  instantiate_kquant_modes(iq2xxs, type, 32, 1, 8, false, KQ_GRID_IQ2XXS, -3) \
  instantiate_kquant_modes(iq2xs, type, 32, 2, 8, false, KQ_GRID_IQ2XS, -3) \
  instantiate_kquant_modes(iq2s, type, 32, 2, 8, false, KQ_GRID_IQ2S, -3) \
  instantiate_kquant_modes(iq3xxs, type, 32, 2, 8, false, KQ_GRID_IQ3XXS, -2) \
  instantiate_kquant_modes(iq1s, type, 32, 1, 8, false, KQ_GRID_IQ1S, -3) \
  instantiate_kquant_modes(iq1m, type, 32, 1, 8, false, KQ_GRID_IQ1M, -3)

#if KQUANT_DTYPE == 0
instantiate_kquant_types(float)
#elif KQUANT_DTYPE == 1
instantiate_kquant_types(float16_t)
#elif KQUANT_DTYPE == 2
instantiate_kquant_types(bfloat16_t)

// M = 8 simdgroup-matrix qmv: bfloat16 only, and only the modes kq_sg8::format
// decodes (q2k, q3k, iq4nl and iq3s take qmm_m8_nax or qmv_wide).
instantiate_kquant(q6k, qmv_sg8, bfloat16_t, 16, 6, 16, false, KQ_LINEAR, 0)
instantiate_kquant(q4k, qmv_sg8, bfloat16_t, 32, 4, 8, true, KQ_LINEAR, 0)
instantiate_kquant(q5k, qmv_sg8, bfloat16_t, 32, 5, 8, true, KQ_LINEAR, 0)
instantiate_kquant(iq4xs, qmv_sg8, bfloat16_t, 32, 4, 8, false, KQ_CODEBOOK, 0)
instantiate_kernel("kquant_qmv_sg8_prep_bfloat16_t_gs_16", kquant_qmv_sg8_prep, bfloat16_t, 16)
instantiate_kernel("kquant_qmv_sg8_prep_bfloat16_t_gs_32", kquant_qmv_sg8_prep, bfloat16_t, 32)
#else
#error "KQUANT_DTYPE must be 0, 1 or 2"
#endif
// clang-format on
