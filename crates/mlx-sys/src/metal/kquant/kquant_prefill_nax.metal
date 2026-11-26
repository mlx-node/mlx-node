// Standalone Metal 4 (`-std=metal4.0 -O3`) Tiled64 bfloat16 prefill GEMM
// kernels (BM = 128, BN = 64, BK = 64, 128 threads = 4 simdgroups).
//
// Zero MLX header dependencies and plain non-template `[[kernel]]` entrypoints
// (no `weak_odr` templates), compiled twice into paged_attn.metallib:
//   - plain stores (`kquant_prefill_nax.air`, `-std=metal4.0 -O3`):
//       17 kernels `<mode>_qmm_t_nax_t64_p128_bf16` (15 K-quant/IQ + a4g64/a8g64)
//   - fused store epilogues (`kquant_prefill_nax_ep.air`,
//       `-std=metal4.0 -O3 -fno-fast-math -DKQUANT_PREFILL_EPILOGUES=1`):
//       14 kernels `<mode>_qmm_t_nax_t64_p128_{res,swiglu}_bf16` (7 epilogue modes)
//
// Threadgroup memory is declared strictly per format class so non-codebook /
// non-grid kernels stay at exactly 16,384 B (no dummy `tl[1]`/`grid64[1]`/`grid32[1]`).

#include "kquant_prefill_nax.h"

// clang-format off
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshift-count-negative"

#define KQ_PREFILL_TG_PLAIN(group_size, bits, super_ratio, has_min, kind) \
  threadgroup half stage[2 * kq_prefill::kPrefillStage]; \
  threadgroup half2* tl = nullptr; \
  threadgroup uint64_t* grid64 = nullptr; \
  threadgroup uint32_t* grid32 = nullptr;

#define KQ_PREFILL_TG_CODEBOOK(group_size, bits, super_ratio, has_min, kind) \
  threadgroup half stage[2 * kq_prefill::kPrefillStage]; \
  threadgroup half2 tl[kq_m8::kTableEntries]; \
  threadgroup uint64_t* grid64 = nullptr; \
  threadgroup uint32_t* grid32 = nullptr;

#define KQ_PREFILL_TG_GRID64(group_size, bits, super_ratio, has_min, kind) \
  constexpr kq_m8::Format F = \
      kq_m8::format<group_size, bits, super_ratio, has_min, kind>(); \
  threadgroup half stage[2 * kq_prefill::kPrefillStage]; \
  threadgroup half2* tl = nullptr; \
  threadgroup uint64_t grid64[kq_m8::grid_tl64<F>()]; \
  threadgroup uint32_t* grid32 = nullptr;

#define KQ_PREFILL_TG_GRID32(group_size, bits, super_ratio, has_min, kind) \
  constexpr kq_m8::Format F = \
      kq_m8::format<group_size, bits, super_ratio, has_min, kind>(); \
  threadgroup half stage[2 * kq_prefill::kPrefillStage]; \
  threadgroup half2* tl = nullptr; \
  threadgroup uint64_t* grid64 = nullptr; \
  threadgroup uint32_t grid32[kq_m8::grid_tl32<F>()];

#ifndef KQUANT_PREFILL_EPILOGUES

#define DEFINE_KQ_PREFILL_PLAIN(mode, tg_decl, group_size, bits, super_ratio, has_min, kind, shift) \
  [[kernel, max_total_threads_per_threadgroup(128)]] void \
  mode##_qmm_t_nax_t64_p128_bf16( \
      const device uint32_t* w [[buffer(0)]], \
      const device uint8_t* scales [[buffer(1)]], \
      const device half* biases [[buffer(2)]], \
      const device bfloat* x [[buffer(3)]], \
      device bfloat* y [[buffer(4)]], \
      const constant int& K [[buffer(5)]], \
      const constant int& N [[buffer(6)]], \
      const constant int& M [[buffer(7)]], \
      uint2 group [[threadgroup_position_in_grid]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]) { \
    tg_decl(group_size, bits, super_ratio, has_min, kind) \
    kq_prefill::kquant_prefill128_impl< \
        group_size, bits, super_ratio, has_min, kind, shift, kq_prefill::KQ_EP_NONE>( \
        w, scales, biases, x, y, nullptr, K, N, M, \
        stage, tl, grid64, grid32, group, simd_lid, simd_gid); \
  }

DEFINE_KQ_PREFILL_PLAIN(q6k,    KQ_PREFILL_TG_PLAIN,    16, 6, 16, false, KQ_LINEAR,      0)
DEFINE_KQ_PREFILL_PLAIN(q4k,    KQ_PREFILL_TG_PLAIN,    32, 4,  8, true,  KQ_LINEAR,      0)
DEFINE_KQ_PREFILL_PLAIN(q5k,    KQ_PREFILL_TG_PLAIN,    32, 5,  8, true,  KQ_LINEAR,      0)
DEFINE_KQ_PREFILL_PLAIN(q3k,    KQ_PREFILL_TG_PLAIN,    16, 3, 16, false, KQ_LINEAR,      0)
DEFINE_KQ_PREFILL_PLAIN(q2k,    KQ_PREFILL_TG_PLAIN,    16, 2, 16, true,  KQ_LINEAR,      0)
DEFINE_KQ_PREFILL_PLAIN(iq4nl,  KQ_PREFILL_TG_CODEBOOK, 32, 4,  1, false, KQ_CODEBOOK,    0)
DEFINE_KQ_PREFILL_PLAIN(iq4xs,  KQ_PREFILL_TG_CODEBOOK, 32, 4,  8, false, KQ_CODEBOOK,    0)
DEFINE_KQ_PREFILL_PLAIN(iq3s8,  KQ_PREFILL_TG_PLAIN,    32, 8,  8, false, KQ_INT8,        0)
DEFINE_KQ_PREFILL_PLAIN(iq2xxs, KQ_PREFILL_TG_GRID64,   32, 1,  8, false, KQ_GRID_IQ2XXS, -3)
DEFINE_KQ_PREFILL_PLAIN(iq2xs,  KQ_PREFILL_TG_GRID64,   32, 2,  8, false, KQ_GRID_IQ2XS,  -3)
DEFINE_KQ_PREFILL_PLAIN(iq2s,   KQ_PREFILL_TG_GRID64,   32, 2,  8, false, KQ_GRID_IQ2S,   -3)
DEFINE_KQ_PREFILL_PLAIN(iq3xxs, KQ_PREFILL_TG_GRID32,   32, 2,  8, false, KQ_GRID_IQ3XXS, -2)
DEFINE_KQ_PREFILL_PLAIN(iq1s,   KQ_PREFILL_TG_GRID32,   32, 1,  8, false, KQ_GRID_IQ1S,   -3)
DEFINE_KQ_PREFILL_PLAIN(iq1m,   KQ_PREFILL_TG_GRID32,   32, 1,  8, false, KQ_GRID_IQ1M,   -3)
DEFINE_KQ_PREFILL_PLAIN(iq3s,   KQ_PREFILL_TG_GRID32,   32, 3,  8, false, KQ_GRID_IQ3S,    0)
DEFINE_KQ_PREFILL_PLAIN(a4g64,  KQ_PREFILL_TG_PLAIN,    64, 4,  4, false, KQ_AFFINE,       0)
DEFINE_KQ_PREFILL_PLAIN(a8g64,  KQ_PREFILL_TG_PLAIN,    64, 8,  4, false, KQ_AFFINE,       0)

#else

#define DEFINE_KQ_PREFILL_EP(mode, ep_name, ep, tg_decl, group_size, bits, super_ratio, has_min, kind, shift) \
  [[kernel, max_total_threads_per_threadgroup(128)]] void \
  mode##_qmm_t_nax_t64_p128_##ep_name##_bf16( \
      const device uint32_t* w [[buffer(0)]], \
      const device uint8_t* scales [[buffer(1)]], \
      const device half* biases [[buffer(2)]], \
      const device bfloat* x [[buffer(3)]], \
      device bfloat* y [[buffer(4)]], \
      const constant int& K [[buffer(5)]], \
      const constant int& N [[buffer(6)]], \
      const constant int& M [[buffer(7)]], \
      const device bfloat* aux [[buffer(8)]], \
      uint2 group [[threadgroup_position_in_grid]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]) { \
    tg_decl(group_size, bits, super_ratio, has_min, kind) \
    kq_prefill::kquant_prefill128_impl< \
        group_size, bits, super_ratio, has_min, kind, shift, ep>( \
        w, scales, biases, x, y, aux, K, N, M, \
        stage, tl, grid64, grid32, group, simd_lid, simd_gid); \
  }

#define DEFINE_KQ_PREFILL_EPS(mode, tg_decl, group_size, bits, super_ratio, has_min, kind, shift) \
  DEFINE_KQ_PREFILL_EP(mode, res,    kq_prefill::KQ_EP_RESIDUAL, tg_decl, group_size, bits, super_ratio, has_min, kind, shift) \
  DEFINE_KQ_PREFILL_EP(mode, swiglu, kq_prefill::KQ_EP_SWIGLU,   tg_decl, group_size, bits, super_ratio, has_min, kind, shift)

DEFINE_KQ_PREFILL_EPS(q6k,   KQ_PREFILL_TG_PLAIN,    16, 6, 16, false, KQ_LINEAR,    0)
DEFINE_KQ_PREFILL_EPS(q4k,   KQ_PREFILL_TG_PLAIN,    32, 4,  8, true,  KQ_LINEAR,    0)
DEFINE_KQ_PREFILL_EPS(q5k,   KQ_PREFILL_TG_PLAIN,    32, 5,  8, true,  KQ_LINEAR,    0)
DEFINE_KQ_PREFILL_EPS(q3k,   KQ_PREFILL_TG_PLAIN,    16, 3, 16, false, KQ_LINEAR,    0)
DEFINE_KQ_PREFILL_EPS(iq4nl, KQ_PREFILL_TG_CODEBOOK, 32, 4,  1, false, KQ_CODEBOOK,  0)
DEFINE_KQ_PREFILL_EPS(iq4xs, KQ_PREFILL_TG_CODEBOOK, 32, 4,  8, false, KQ_CODEBOOK,  0)
DEFINE_KQ_PREFILL_EPS(iq3s,  KQ_PREFILL_TG_GRID32,   32, 3,  8, false, KQ_GRID_IQ3S, 0)

#endif
#pragma clang diagnostic pop
// clang-format on
