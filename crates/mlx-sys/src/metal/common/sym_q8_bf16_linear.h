#pragma once

// ============================================================================
// Shared Generic BF16 & Symmetric Q8_0 (`SYM_F16`) Linear Metal Kernels
//
// Hoisted verbatim from `inkling2/inkling2_ops.metal` (Sections 15, 18, 19, 20:
// all 13 BF16 and symmetric Q8_0 linear / fused lm_head+softcap+argmax kernels).
//
// SYM_F16 Q8_0 Wire Layout:
//   - group_size = 32 (K % 32 == 0)
//   - Weights stored as uint8 codes XOR-shifted by 0x80 so
//     `as_type<char4>(w_u32 ^ 0x80808080u)` recovers exact signed int8 in
//     [-128, 127] (0-ULP bit-exact to `int(u8) - 128`, zero weight repacking).
//   - Per-group scales stored as native IEEE Float16 (`half`), ZERO bias array.
//
// FP Contract & Include Requirements:
//   - Must be compiled with `-fno-fast-math` and `#pragma clang fp contract(off)`
//     active in the including translation unit so FP32 accumulation & BF16
//     rounding stay 0-ULP batch-size invariant across M = 1..=32.
//   - Expects `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`,
//     `<metal_stdlib>`, `using namespace metal;`, `using namespace mpp::tensor_ops;`,
//     and `inkling2_softcap(float, float)` declared prior to `#include`.
// ============================================================================

// ============================================================================
// 15. inkling2_bf16_linear + inkling2_bf16_lm_head_softcap_argmax
//     Coalesced single-pass BF16 linear (`1 <= M <= 32`, `K % 4 == 0`) with
//     128-bit `uint4` weight loads and exact 32-lane FP32 reduction order
//     (0-ULP bit-exact batch-size invariance across ALL M in 1..=32!).
// ============================================================================
struct Inkling2Bf16LinearParams {
  uint rows;        // M (1..=32)
  uint out_dim;     // N (or out_cols for lm_head)
  uint in_dim;      // K (% 4 == 0)
  uint num_splits;  // stage1 TG count for lm_head argmax (0 for linear)
  float softcap;    // softcap (> 0) for lm_head argmax
  uint _pad0;
  uint _pad1;
  uint _pad2;
};

inline float inkling2_reduce_32_ordered(float v) {
  v += simd_shuffle_down(v, 16);
  v += simd_shuffle_down(v, 8);
  v += simd_shuffle_down(v, 4);
  v += simd_shuffle_down(v, 2);
  v += simd_shuffle_down(v, 1);
  return v;
}

// Explicit FP32 FMA micro-step (8 ffma ops vs 16 fmul/fadd under fp contract(off),
// identical instruction order across ALL M=1..=32 -> 0-ULP batch-size invariance!).
inline float inkling2_bf16_fma8(float acc, float4 w0, float4 w1, uint4 xr) {
  const float4 x0 = float4(as_type<bfloat4>(xr.xy));
  acc = fma(w0.x, x0.x, acc);
  acc = fma(w0.y, x0.y, acc);
  acc = fma(w0.z, x0.z, acc);
  acc = fma(w0.w, x0.w, acc);
  const float4 x1 = float4(as_type<bfloat4>(xr.zw));
  acc = fma(w1.x, x1.x, acc);
  acc = fma(w1.y, x1.y, acc);
  acc = fma(w1.z, x1.z, acc);
  acc = fma(w1.w, x1.w, acc);
  return acc;
}

inline float inkling2_bf16_fma4(float acc, float4 w0, ushort4 xr) {
  const float4 x0 = float4(as_type<bfloat4>(xr));
  acc = fma(w0.x, x0.x, acc);
  acc = fma(w0.y, x0.y, acc);
  acc = fma(w0.z, x0.z, acc);
  acc = fma(w0.w, x0.w, acc);
  return acc;
}

// Grid: (ceil(N / 4), 1, 1), TG: (128, 1, 1) -> 4 SGs x 1 output row/SG.
// 4x in-flight uint4 loads (16 cachelines/SG) saturates M5 Max DRAM bandwidth while
// preserving the exact sequential k-order of `inkling2_bf16_linear_batched` (0-ULP parity).
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_bf16_linear_m1(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant Inkling2Bf16LinearParams &params [[buffer(3)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint n = tgid_x * 4u + sg;
  if (n >= params.out_dim) {
    return;
  }
  const uint K = params.in_dim;
  const uint K_vec8 = K & ~7u;
  const uint K_unroll = K_vec8 & ~1023u;
  const device bfloat *w_row = w + ulong(n) * ulong(K);

  float acc = 0.0f;
  uint k = lane * 8u;
  for (; k < K_unroll; k += 1024u) {
    const uint4 wr0 = *reinterpret_cast<device const uint4 *>(w_row + k);
    const uint4 wr1 = *reinterpret_cast<device const uint4 *>(w_row + k + 256u);
    const uint4 wr2 = *reinterpret_cast<device const uint4 *>(w_row + k + 512u);
    const uint4 wr3 = *reinterpret_cast<device const uint4 *>(w_row + k + 768u);
    const uint4 xr0 = *reinterpret_cast<device const uint4 *>(x + k);
    const uint4 xr1 = *reinterpret_cast<device const uint4 *>(x + k + 256u);
    const uint4 xr2 = *reinterpret_cast<device const uint4 *>(x + k + 512u);
    const uint4 xr3 = *reinterpret_cast<device const uint4 *>(x + k + 768u);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr0.xy)), float4(as_type<bfloat4>(wr0.zw)), xr0);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr1.xy)), float4(as_type<bfloat4>(wr1.zw)), xr1);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr2.xy)), float4(as_type<bfloat4>(wr2.zw)), xr2);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr3.xy)), float4(as_type<bfloat4>(wr3.zw)), xr3);
  }
  for (; k < K_vec8; k += 256u) {
    const uint4 wr = *reinterpret_cast<device const uint4 *>(w_row + k);
    const uint4 xr = *reinterpret_cast<device const uint4 *>(x + k);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr.xy)), float4(as_type<bfloat4>(wr.zw)), xr);
  }
  if (K_vec8 < K && lane == 0u) {
    const ushort4 wr = *reinterpret_cast<device const ushort4 *>(w_row + K_vec8);
    const ushort4 xr = *reinterpret_cast<device const ushort4 *>(x + K_vec8);
    acc = inkling2_bf16_fma4(acc, float4(as_type<bfloat4>(wr)), xr);
  }

  acc = inkling2_reduce_32_ordered(acc);
  if (lane == 0u) {
    out[n] = bfloat(acc);
  }
}

// Small-M (`1 <= M <= 8`) Apple M5 Max NAX BF16 linear (`TSM=8, TSN=16, BK=64, KS=4`, 1.5 KB smem).
// Uses identical `BK=64, KS=4` NAX accumulation & SG reduction order as `inkling2_bf16_linear_batched`
// -> 0-ULP bit-exact parity across ALL M = 1..=1024 with 4x less smem and 4x smaller NAX tile!
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_bf16_linear_nax_m8(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant Inkling2Bf16LinearParams &params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  (void)lane;
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using dev_tensor_t = metal::tensor<device bfloat, extents_t, metal::tensor_inline>;
  using tg_tensor_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;

  constexpr int TSM = 8;
  constexpr int TSN = 16;
  constexpr int BK = 64;
  constexpr int KS = 4;

  const int M = int(params.rows);
  const int N = int(params.out_dim);
  const int K = int(params.in_dim);
  const int n_off = int(tgid.x) * TSN;
  if (M <= 0 || n_off >= N) {
    return;
  }

  const int rows = min(TSM, M);
  const int cols = min(TSN, N - n_off);
  const bool full_mn = (rows == TSM && cols == TSN);
  const int k_group = int(sg);

  device bfloat *A = const_cast<device bfloat *>(x);
  device bfloat *B = const_cast<device bfloat *>(w) + int64_t(n_off) * K;
  device bfloat *D = out + n_off;

  dev_tensor_t tA(A, extents_t(K, rows), strides_t{1, K});
  dev_tensor_t tB(B, extents_t(K, cols), strides_t{1, K});

  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  constexpr auto desc = matmul2d_descriptor(
      TSM,
      TSN,
      BK,
      false,
      true,
      true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, metal::execution_simdgroup> op;

  auto Dacc = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();

  const int k_iters = ((K + BK - 1) / BK + KS - 1) / KS;
  for (int it = 0; it < k_iters; ++it) {
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + k_group) * BK;
    if (k >= K) {
      continue;
    }
    if (full_mn && k + BK <= K) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op.run(Areg, b, Dacc);
    } else {
      const int kk = min(BK, K - k);
      dev_tensor_t vA(A + k, extents_t(kk, rows), strides_t{1, K});
      dev_tensor_t vB(B + k, extents_t(kk, cols), strides_t{1, K});
      op.run(vA, vB, Dacc);
    }
  }

  constexpr int kBlockSize = TSM * TSN; // 128 floats
  threadgroup float partials[(KS - 1) * kBlockSize]; // 1.5 KB smem
  if (k_group > 0) {
    tg_tensor_t tP(partials + (k_group - 1) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tP);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k_group != 0) {
    return;
  }

  #pragma unroll
  for (int g = 0; g < KS - 1; ++g) {
    tg_tensor_t tP(partials + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    auto part = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    part.load(tP);
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] += part[i];
    }
  }

  auto Dout = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, bfloat>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dout[i] = bfloat(Dacc[i]);
  }
  dev_tensor_t tD(D, extents_t(cols, rows), strides_t{1, N});
  if (full_mn) {
    Dout.store(tD.template slice<TSN, TSM>(0, 0));
  } else {
    Dout.store(tD);
  }
}

// Grid: (ceil(M / 32) * ceil(N / 16), 1, 1), TG: (128, 1, 1) -> 4 SGs split-K (KS=4).
// Apple M5 Max hardware NAX (`mpp::tensor_ops::matmul2d`, TSM=32, TSN=16, BK=64, KS=4, 6 KB smem):
// - Streams W directly from device memory at ~580 GB/s & 60+ TFLOPS in ONE DRAM pass for M=1..=32!
// - Never shifts tile origins at M/N edges -> 0-ULP bit-exact batch-size invariance across ALL
//   M = 1, 2, 4, 5, 6, 7, 8, 16, 20, 32, 38, 678, 2048!
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_bf16_linear_batched(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant Inkling2Bf16LinearParams &params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  (void)lane;
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using dev_tensor_t = metal::tensor<device bfloat, extents_t, metal::tensor_inline>;
  using tg_tensor_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;

  constexpr int TSM = 32;
  constexpr int TSN = 16;
  constexpr int BK = 64;
  constexpr int KS = 4;

  const int M = int(params.rows);
  const int N = int(params.out_dim);
  const int K = int(params.in_dim);
  const int tiles_m = (M + TSM - 1) / TSM;
  const int tgp = int(tgid.x);
  const int tile_m = tgp % tiles_m;
  const int tile_n = tgp / tiles_m;
  const int m_off = tile_m * TSM;
  const int n_off = tile_n * TSN;
  if (m_off >= M || n_off >= N) {
    return;
  }

  const int rows = min(TSM, M - m_off);
  const int cols = min(TSN, N - n_off);
  const bool full_mn = (rows == TSM && cols == TSN);
  const int k_group = int(sg);

  device bfloat *A = const_cast<device bfloat *>(x) + int64_t(m_off) * K;
  device bfloat *B = const_cast<device bfloat *>(w) + int64_t(n_off) * K;
  device bfloat *D = out + int64_t(m_off) * N + n_off;

  dev_tensor_t tA(A, extents_t(K, rows), strides_t{1, K});
  dev_tensor_t tB(B, extents_t(K, cols), strides_t{1, K});

  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  constexpr auto desc = matmul2d_descriptor(
      TSM,
      TSN,
      BK,
      false,
      true,
      true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, metal::execution_simdgroup> op;

  auto Dacc = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();

  const int k_iters = ((K + BK - 1) / BK + KS - 1) / KS;
  for (int it = 0; it < k_iters; ++it) {
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + k_group) * BK;
    if (k >= K) {
      continue;
    }
    if (full_mn && k + BK <= K) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op.run(Areg, b, Dacc);
    } else {
      const int kk = min(BK, K - k);
      dev_tensor_t vA(A + k, extents_t(kk, rows), strides_t{1, K});
      dev_tensor_t vB(B + k, extents_t(kk, cols), strides_t{1, K});
      op.run(vA, vB, Dacc);
    }
  }

  constexpr int kBlockSize = TSM * TSN; // 512 floats
  threadgroup float partials[(KS - 1) * kBlockSize]; // 6 KB smem
  if (k_group > 0) {
    tg_tensor_t tP(partials + (k_group - 1) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tP);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k_group != 0) {
    return;
  }

  #pragma unroll
  for (int g = 0; g < KS - 1; ++g) {
    tg_tensor_t tP(partials + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    auto part = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    part.load(tP);
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] += part[i];
    }
  }

  auto Dout = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, bfloat>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dout[i] = bfloat(Dacc[i]);
  }
  dev_tensor_t tD(D, extents_t(cols, rows), strides_t{1, N});
  if (full_mn) {
    Dout.store(tD.template slice<TSN, TSM>(0, 0));
  } else {
    Dout.store(tD);
  }
}

// Fused NAX BF16 lm_head + softcap + Stage 1 argmax (ZERO [M, V] DRAM write, 1 DRAM pass!).
// Grid: (num_splits = ceil(out_cols / 32), 1, 1), TG: (128, 1, 1) -> 4 SGs split-K (KS=4).
// Uses exact same TSM=32, BK=64, KS=4 NAX reduction as `inkling2_bf16_linear_batched` ->
// 0-ULP bit-exact parity across ALL M = 1..=32 AND ~1.7 ms for 926 MB lm_head!
// Stage 2 reuses `inkling2_lm_head_softcap_argmax_stage2`!
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_bf16_lm_head_softcap_argmax_p1(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device float *partial_vals [[buffer(2)]],
    device int32_t *partial_idxs [[buffer(3)]],
    constant Inkling2Bf16LinearParams &params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using dev_tensor_t = metal::tensor<device bfloat, extents_t, metal::tensor_inline>;
  using tg_tensor_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;

  constexpr int TSM = 32;
  constexpr int TSN = 32;
  constexpr int BK = 64;
  constexpr int KS = 4;

  const uint split = tgid.x;
  if (split >= params.num_splits) {
    return;
  }
  const int M = int(params.rows);
  const int V = int(params.out_dim);
  const int K = int(params.in_dim);
  const int n_off = int(split) * TSN;
  if (n_off >= V || M <= 0) {
    return;
  }
  const int rows = min(TSM, M);
  const int cols = min(TSN, V - n_off);
  const bool full_mn = (rows == TSM && cols == TSN);
  const int k_group = int(sg);

  device bfloat *A = const_cast<device bfloat *>(x);
  device bfloat *B = const_cast<device bfloat *>(w) + int64_t(n_off) * K;

  dev_tensor_t tA(A, extents_t(K, rows), strides_t{1, K});
  dev_tensor_t tB(B, extents_t(K, cols), strides_t{1, K});

  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  constexpr auto desc = matmul2d_descriptor(
      TSM,
      TSN,
      BK,
      false,
      true,
      true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, metal::execution_simdgroup> op;

  auto Dacc = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();

  const int k_iters = ((K + BK - 1) / BK + KS - 1) / KS;
  for (int it = 0; it < k_iters; ++it) {
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + k_group) * BK;
    if (k >= K) {
      continue;
    }
    if (full_mn && k + BK <= K) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op.run(Areg, b, Dacc);
    } else {
      const int kk = min(BK, K - k);
      dev_tensor_t vA(A + k, extents_t(kk, rows), strides_t{1, K});
      dev_tensor_t vB(B + k, extents_t(kk, cols), strides_t{1, K});
      op.run(vA, vB, Dacc);
    }
  }

  constexpr int kBlockSize = TSM * TSN; // 1024 floats
  threadgroup float partials[(KS - 1) * kBlockSize]; // 12 KB smem (reused for final logits!)
  if (k_group > 0) {
    tg_tensor_t tP(partials + (k_group - 1) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tP);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (k_group == 0) {
    #pragma unroll
    for (int g = 0; g < KS - 1; ++g) {
      tg_tensor_t tP(partials + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
      auto part = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
      part.load(tP);
      #pragma unroll
      for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
        Dacc[i] += part[i];
      }
    }
    tg_tensor_t tOut(partials, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tOut);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // All 4 simdgroups cooperatively softcap + argmax `rows` rows x `cols` (<= 32) vocab cols!
  const float cap = params.softcap;
  for (uint r = sg; r < uint(rows); r += 4u) {
    float bv = -INFINITY;
    int32_t bi = -1;
    if (lane < uint(cols)) {
      float val = float(bfloat(partials[r * 32u + lane]));
      if (cap > 0.0f) {
        val = inkling2_softcap(val, cap);
      }
      bv = val;
      bi = int32_t(n_off + int(lane));
    }
    #pragma unroll
    for (ushort off = 16; off >= 1; off >>= 1) {
      const float ov = simd_shuffle_down(bv, off);
      const int32_t oi = simd_shuffle_down(bi, off);
      if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0u) {
      const ulong out_slot = ulong(r) * ulong(params.num_splits) + ulong(split);
      partial_vals[out_slot] = bv;
      partial_idxs[out_slot] = bi;
    }
  }
}

// ============================================================================
// 18. Symmetric Q8_0 (`group_size=32`, native Float16 scales, ZERO biases!)
//     `as_type<char4>(w_u32 ^ 0x80808080u)` is 0-ULP bit-exact to `int(u8) - 128`
//     for every byte 0..255 with zero weight repacking!
//     128-bit `uint4` weight + activation loads (`kLanes = 8`, 256B/warp),
//     exact same k-order & 8-lane tree reduction across ALL `M in 1..=32`
//     (0-ULP batch-size invariance!).
//
//     - `inkling2_lm_head_q8_softcap_argmax_stage1` (reuses `stage2`)
//     - `inkling2_lm_head_q8_bf16` (BF16 base_logits OR FP32+softcap logits)
//     - `inkling2_q8_sym_matmul_bf16` (dense Q8_0 + optional fused BF16 tail!)
// ============================================================================
struct Inkling2Q8SymParams {
  uint rows;        // M (1..=32)
  uint out_dim;     // N (or out_cols for lm_head)
  uint in_dim;      // K (% 32 == 0)
  uint num_splits;  // stage1 TG count for argmax (0 for linear/lm_head_bf16)
  float softcap;    // > 0 applies softcap in lm_head kernels
  uint aux_dim;     // N_aux for fused BF16 tail (0 when none)
  uint flags;       // bit0: out_is_f32, bit1: has_aux
  uint _pad;
};

inline float inkling2_q8_sym_reduce8(float v) {
  v += simd_shuffle_down(v, 4);
  v += simd_shuffle_down(v, 2);
  v += simd_shuffle_down(v, 1);
  return v;
}

inline float inkling2_q8_sym_dot32_x(
    float4 w0, float4 w1, float4 w2, float4 w3,
    float4 w4, float4 w5, float4 w6, float4 w7,
    device const uint4 *xp) {
  const uint4 x01 = xp[0];
  const uint4 x23 = xp[1];
  const uint4 x45 = xp[2];
  const uint4 x67 = xp[3];
  float4 d = w0 * float4(as_type<bfloat4>(x01.xy));
  d = fma(w1, float4(as_type<bfloat4>(x01.zw)), d);
  d = fma(w2, float4(as_type<bfloat4>(x23.xy)), d);
  d = fma(w3, float4(as_type<bfloat4>(x23.zw)), d);
  d = fma(w4, float4(as_type<bfloat4>(x45.xy)), d);
  d = fma(w5, float4(as_type<bfloat4>(x45.zw)), d);
  d = fma(w6, float4(as_type<bfloat4>(x67.xy)), d);
  d = fma(w7, float4(as_type<bfloat4>(x67.zw)), d);
  return (d.x + d.y) + (d.z + d.w);
}

inline float inkling2_q8_sym_row_m1(
    device const uint4 *x_u4,
    device const uint4 *w_row,
    device const half *s_row,
    uint n_groups,
    uint k_lane) {
  float acc = 0.0f;
  uint g = k_lane;
  for (; g + 8u < n_groups; g += 16u) {
    const uint4 p0a = w_row[g * 2u + 0u];
    const uint4 p1a = w_row[g * 2u + 1u];
    const float sca = float(s_row[g]);
    const uint g1 = g + 8u;
    const uint4 p0b = w_row[g1 * 2u + 0u];
    const uint4 p1b = w_row[g1 * 2u + 1u];
    const float scb = float(s_row[g1]);

    const float4 w0a = float4(as_type<char4>(p0a.x ^ 0x80808080u));
    const float4 w1a = float4(as_type<char4>(p0a.y ^ 0x80808080u));
    const float4 w2a = float4(as_type<char4>(p0a.z ^ 0x80808080u));
    const float4 w3a = float4(as_type<char4>(p0a.w ^ 0x80808080u));
    const float4 w4a = float4(as_type<char4>(p1a.x ^ 0x80808080u));
    const float4 w5a = float4(as_type<char4>(p1a.y ^ 0x80808080u));
    const float4 w6a = float4(as_type<char4>(p1a.z ^ 0x80808080u));
    const float4 w7a = float4(as_type<char4>(p1a.w ^ 0x80808080u));
    acc = fma(
        sca,
        inkling2_q8_sym_dot32_x(w0a, w1a, w2a, w3a, w4a, w5a, w6a, w7a, x_u4 + ulong(g) * 4ul),
        acc);

    const float4 w0b = float4(as_type<char4>(p0b.x ^ 0x80808080u));
    const float4 w1b = float4(as_type<char4>(p0b.y ^ 0x80808080u));
    const float4 w2b = float4(as_type<char4>(p0b.z ^ 0x80808080u));
    const float4 w3b = float4(as_type<char4>(p0b.w ^ 0x80808080u));
    const float4 w4b = float4(as_type<char4>(p1b.x ^ 0x80808080u));
    const float4 w5b = float4(as_type<char4>(p1b.y ^ 0x80808080u));
    const float4 w6b = float4(as_type<char4>(p1b.z ^ 0x80808080u));
    const float4 w7b = float4(as_type<char4>(p1b.w ^ 0x80808080u));
    acc = fma(
        scb,
        inkling2_q8_sym_dot32_x(w0b, w1b, w2b, w3b, w4b, w5b, w6b, w7b, x_u4 + ulong(g1) * 4ul),
        acc);
  }
  if (g < n_groups) {
    const uint4 p0 = w_row[g * 2u + 0u];
    const uint4 p1 = w_row[g * 2u + 1u];
    const float sc = float(s_row[g]);
    const float4 w0 = float4(as_type<char4>(p0.x ^ 0x80808080u));
    const float4 w1 = float4(as_type<char4>(p0.y ^ 0x80808080u));
    const float4 w2 = float4(as_type<char4>(p0.z ^ 0x80808080u));
    const float4 w3 = float4(as_type<char4>(p0.w ^ 0x80808080u));
    const float4 w4 = float4(as_type<char4>(p1.x ^ 0x80808080u));
    const float4 w5 = float4(as_type<char4>(p1.y ^ 0x80808080u));
    const float4 w6 = float4(as_type<char4>(p1.z ^ 0x80808080u));
    const float4 w7 = float4(as_type<char4>(p1.w ^ 0x80808080u));
    acc = fma(
        sc,
        inkling2_q8_sym_dot32_x(w0, w1, w2, w3, w4, w5, w6, w7, x_u4 + ulong(g) * 4ul),
        acc);
  }
  return inkling2_q8_sym_reduce8(acc);
}

template <uint M_CAP>
inline void inkling2_q8_sym_row_multi(
    device const uint4 *x_u4,
    device const uint4 *w_row,
    device const half *s_row,
    uint m0,
    uint n_m,
    uint n_groups,
    uint k_lane,
    thread float (&acc)[M_CAP]) {
  const ulong x_row_u4 = ulong(n_groups) * 4ul;
  #pragma unroll
  for (uint i = 0u; i < M_CAP; ++i) {
    acc[i] = 0.0f;
  }
  const device uint4 *x_base = x_u4 + ulong(m0) * x_row_u4;
  for (uint g = k_lane; g < n_groups; g += 8u) {
    const uint4 p0 = w_row[g * 2u + 0u];
    const uint4 p1 = w_row[g * 2u + 1u];
    const float sc = float(s_row[g]);
    const float4 w0 = float4(as_type<char4>(p0.x ^ 0x80808080u));
    const float4 w1 = float4(as_type<char4>(p0.y ^ 0x80808080u));
    const float4 w2 = float4(as_type<char4>(p0.z ^ 0x80808080u));
    const float4 w3 = float4(as_type<char4>(p0.w ^ 0x80808080u));
    const float4 w4 = float4(as_type<char4>(p1.x ^ 0x80808080u));
    const float4 w5 = float4(as_type<char4>(p1.y ^ 0x80808080u));
    const float4 w6 = float4(as_type<char4>(p1.z ^ 0x80808080u));
    const float4 w7 = float4(as_type<char4>(p1.w ^ 0x80808080u));
    const device uint4 *xg = x_base + ulong(g) * 4ul;
    #pragma unroll
    for (uint i = 0u; i < M_CAP; ++i) {
      if (i < n_m) {
        acc[i] = fma(
            sc,
            inkling2_q8_sym_dot32_x(w0, w1, w2, w3, w4, w5, w6, w7, xg + ulong(i) * x_row_u4),
            acc[i]);
      }
    }
  }
  #pragma unroll
  for (uint i = 0u; i < M_CAP; ++i) {
    if (i < n_m) {
      acc[i] = inkling2_q8_sym_reduce8(acc[i]);
    }
  }
}

template <uint M_CAP>
inline void inkling2_bf16_aux_row_multi(
    device const uint4 *x_u4,
    device const uint4 *w_aux_row,
    uint m0,
    uint n_m,
    uint n_groups,
    uint k_lane,
    thread float (&acc)[M_CAP]) {
  const ulong x_row_u4 = ulong(n_groups) * 4ul;
  #pragma unroll
  for (uint i = 0u; i < M_CAP; ++i) {
    acc[i] = 0.0f;
  }
  const device uint4 *x_base = x_u4 + ulong(m0) * x_row_u4;
  for (uint g = k_lane; g < n_groups; g += 8u) {
    const uint4 wp01 = w_aux_row[g * 4u + 0u];
    const uint4 wp23 = w_aux_row[g * 4u + 1u];
    const uint4 wp45 = w_aux_row[g * 4u + 2u];
    const uint4 wp67 = w_aux_row[g * 4u + 3u];
    const float4 w0 = float4(as_type<bfloat4>(wp01.xy));
    const float4 w1 = float4(as_type<bfloat4>(wp01.zw));
    const float4 w2 = float4(as_type<bfloat4>(wp23.xy));
    const float4 w3 = float4(as_type<bfloat4>(wp23.zw));
    const float4 w4 = float4(as_type<bfloat4>(wp45.xy));
    const float4 w5 = float4(as_type<bfloat4>(wp45.zw));
    const float4 w6 = float4(as_type<bfloat4>(wp67.xy));
    const float4 w7 = float4(as_type<bfloat4>(wp67.zw));
    const device uint4 *xg = x_base + ulong(g) * 4ul;
    #pragma unroll
    for (uint i = 0u; i < M_CAP; ++i) {
      if (i < n_m) {
        acc[i] += inkling2_q8_sym_dot32_x(
            w0, w1, w2, w3, w4, w5, w6, w7, xg + ulong(i) * x_row_u4);
      }
    }
  }
  #pragma unroll
  for (uint i = 0u; i < M_CAP; ++i) {
    if (i < n_m) {
      acc[i] = inkling2_q8_sym_reduce8(acc[i]);
    }
  }
}

inline void inkling2_argmax_sg_reduce4(thread float &bv, thread int32_t &bi) {
  {
    const float ov = simd_shuffle_xor(bv, 16);
    const int32_t oi = simd_shuffle_xor(bi, 16);
    if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
      bv = ov;
      bi = oi;
    }
  }
  {
    const float ov = simd_shuffle_xor(bv, 8);
    const int32_t oi = simd_shuffle_xor(bi, 8);
    if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
      bv = ov;
      bi = oi;
    }
  }
}

// (a) Fused Symmetric Q8_0 lm_head + softcap + Stage 1 argmax (ZERO [M, V] DRAM write!).
// Grid: (num_splits, 1, 1), TG: (128, 1, 1) -> 4 SGs.
// Stage 2 reuses `inkling2_lm_head_softcap_argmax_stage2`.
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_lm_head_q8_softcap_argmax_stage1(
    device const uint4 *x_u4 [[buffer(0)]],
    device const uint4 *w_u4 [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *partial_vals [[buffer(3)]],
    device int32_t *partial_idxs [[buffer(4)]],
    constant Inkling2Q8SymParams &params [[buffer(5)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint split = tgid_x;
  if (split >= params.num_splits) {
    return;
  }
  const uint M = params.rows;
  const uint V = params.out_dim;
  const uint n_groups = params.in_dim >> 5u;
  const ulong row_u4_stride = ulong(n_groups) * 2ul;
  const float cap = params.softcap;

  const uint chunk = (V + params.num_splits - 1u) / params.num_splits;
  const uint start = split * chunk;
  const uint end = min(start + chunk, V);

  const uint k_lane = lane & 7u;
  const uint r_id = lane >> 3u; // 0..3

  threadgroup float tg_v[32][4];
  threadgroup int32_t tg_i[32][4];

  if (M == 1u) {
    float best_v = -INFINITY;
    int32_t best_i = -1;
    for (uint v_base = start; v_base < end; v_base += 16u) {
      const uint v = v_base + sg * 4u + r_id;
      if (v < end) {
        const float dot_v = inkling2_q8_sym_row_m1(
            x_u4, w_u4 + ulong(v) * row_u4_stride, scales + ulong(v) * ulong(n_groups),
            n_groups, k_lane);
        if (k_lane == 0u) {
          const float val = cap > 0.0f ? inkling2_softcap(dot_v, cap) : dot_v;
          const int32_t ci = int32_t(v);
          if (best_i < 0 || val > best_v || (val == best_v && ci < best_i)) {
            best_v = val;
            best_i = ci;
          }
        }
      }
    }
    inkling2_argmax_sg_reduce4(best_v, best_i);
    if (lane == 0u) {
      tg_v[0][sg] = best_v;
      tg_i[0][sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
      float bv = tg_v[0][0];
      int32_t bi = tg_i[0][0];
      #pragma unroll
      for (uint s = 1u; s < 4u; ++s) {
        const float ov = tg_v[0][s];
        const int32_t oi = tg_i[0][s];
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
          bv = ov;
          bi = oi;
        }
      }
      partial_vals[split] = bv;
      partial_idxs[split] = bi;
    }
    return;
  }

  if (M <= 8u) {
    float best_v[8];
    int32_t best_i[8];
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
      best_v[i] = -INFINITY;
      best_i[i] = -1;
    }
    for (uint v_base = start; v_base < end; v_base += 16u) {
      const uint v = v_base + sg * 4u + r_id;
      if (v < end) {
        float acc[8];
        inkling2_q8_sym_row_multi<8>(
            x_u4, w_u4 + ulong(v) * row_u4_stride, scales + ulong(v) * ulong(n_groups),
            0u, M, n_groups, k_lane, acc);
        if (k_lane == 0u) {
          const int32_t ci = int32_t(v);
          #pragma unroll
          for (uint i = 0u; i < 8u; ++i) {
            if (i < M) {
              const float val = cap > 0.0f ? inkling2_softcap(acc[i], cap) : acc[i];
              if (best_i[i] < 0 || val > best_v[i] || (val == best_v[i] && ci < best_i[i])) {
                best_v[i] = val;
                best_i[i] = ci;
              }
            }
          }
        }
      }
    }
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
      if (i < M) {
        inkling2_argmax_sg_reduce4(best_v[i], best_i[i]);
        if (lane == 0u) {
          tg_v[i][sg] = best_v[i];
          tg_i[i][sg] = best_i[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < M) {
      float bv = tg_v[lane][0];
      int32_t bi = tg_i[lane][0];
      #pragma unroll
      for (uint s = 1u; s < 4u; ++s) {
        const float ov = tg_v[lane][s];
        const int32_t oi = tg_i[lane][s];
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
          bv = ov;
          bi = oi;
        }
      }
      const ulong out_slot = ulong(lane) * ulong(params.num_splits) + ulong(split);
      partial_vals[out_slot] = bv;
      partial_idxs[out_slot] = bi;
    }
    return;
  }

  if (M <= 16u) {
    float best_v[16];
    int32_t best_i[16];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
      best_v[i] = -INFINITY;
      best_i[i] = -1;
    }
    for (uint v_base = start; v_base < end; v_base += 16u) {
      const uint v = v_base + sg * 4u + r_id;
      if (v < end) {
        float acc[16];
        inkling2_q8_sym_row_multi<16>(
            x_u4, w_u4 + ulong(v) * row_u4_stride, scales + ulong(v) * ulong(n_groups),
            0u, M, n_groups, k_lane, acc);
        if (k_lane == 0u) {
          const int32_t ci = int32_t(v);
          #pragma unroll
          for (uint i = 0u; i < 16u; ++i) {
            if (i < M) {
              const float val = cap > 0.0f ? inkling2_softcap(acc[i], cap) : acc[i];
              if (best_i[i] < 0 || val > best_v[i] || (val == best_v[i] && ci < best_i[i])) {
                best_v[i] = val;
                best_i[i] = ci;
              }
            }
          }
        }
      }
    }
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
      if (i < M) {
        inkling2_argmax_sg_reduce4(best_v[i], best_i[i]);
        if (lane == 0u) {
          tg_v[i][sg] = best_v[i];
          tg_i[i][sg] = best_i[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < M) {
      float bv = tg_v[lane][0];
      int32_t bi = tg_i[lane][0];
      #pragma unroll
      for (uint s = 1u; s < 4u; ++s) {
        const float ov = tg_v[lane][s];
        const int32_t oi = tg_i[lane][s];
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
          bv = ov;
          bi = oi;
        }
      }
      const ulong out_slot = ulong(lane) * ulong(params.num_splits) + ulong(split);
      partial_vals[out_slot] = bv;
      partial_idxs[out_slot] = bi;
    }
    return;
  }

  // 17 <= M <= 32: pairs of SGs share each vocab row in L1 cache while splitting M into [0..16) / [16..M)!
  const uint sg_pair = sg >> 1u; // 0 or 1
  const uint m_half = sg & 1u;   // 0 or 1
  const uint m0 = m_half * 16u;
  const uint n_m = min(16u, M - m0);
  float best_v[16];
  int32_t best_i[16];
  #pragma unroll
  for (uint i = 0u; i < 16u; ++i) {
    best_v[i] = -INFINITY;
    best_i[i] = -1;
  }
  for (uint v_base = start; v_base < end; v_base += 8u) {
    const uint v = v_base + sg_pair * 4u + r_id;
    if (v < end) {
      float acc[16];
      inkling2_q8_sym_row_multi<16>(
          x_u4, w_u4 + ulong(v) * row_u4_stride, scales + ulong(v) * ulong(n_groups),
          m0, n_m, n_groups, k_lane, acc);
      if (k_lane == 0u) {
        const int32_t ci = int32_t(v);
        #pragma unroll
        for (uint i = 0u; i < 16u; ++i) {
          if (i < n_m) {
            const float val = cap > 0.0f ? inkling2_softcap(acc[i], cap) : acc[i];
            if (best_i[i] < 0 || val > best_v[i] || (val == best_v[i] && ci < best_i[i])) {
              best_v[i] = val;
              best_i[i] = ci;
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (uint i = 0u; i < 16u; ++i) {
    if (i < n_m) {
      inkling2_argmax_sg_reduce4(best_v[i], best_i[i]);
      if (lane == 0u) {
        tg_v[m0 + i][sg_pair] = best_v[i];
        tg_i[m0 + i][sg_pair] = best_i[i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane < M) {
    float bv = tg_v[lane][0];
    int32_t bi = tg_i[lane][0];
    const float ov = tg_v[lane][1];
    const int32_t oi = tg_i[lane][1];
    if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
      bv = ov;
      bi = oi;
    }
    const ulong out_slot = ulong(lane) * ulong(params.num_splits) + ulong(split);
    partial_vals[out_slot] = bv;
    partial_idxs[out_slot] = bi;
  }
}

inline void inkling2_q8_sym_store_scalar(
    device uchar *out_raw,
    ulong idx,
    float val,
    uint out_is_f32) {
  if (out_is_f32 != 0u) {
    reinterpret_cast<device float *>(out_raw)[idx] = val;
  } else {
    reinterpret_cast<device bfloat *>(out_raw)[idx] = bfloat(val);
  }
}

// (b) Symmetric Q8_0 lm_head -> `[M, out_cols]` BF16 (DSpark base_logits) OR FP32+softcap (sampled/prefill).
// Grid: (ceil(out_cols / rows_per_tg), 1, 1), TG: (128, 1, 1) -> rows_per_tg = 16 (M<=16) or 8 (M>16).
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_lm_head_q8_bf16(
    device const uint4 *x_u4 [[buffer(0)]],
    device const uint4 *w_u4 [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device uchar *out_raw [[buffer(3)]],
    constant Inkling2Q8SymParams &params [[buffer(4)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint M = params.rows;
  const uint V = params.out_dim;
  const uint n_groups = params.in_dim >> 5u;
  const ulong row_u4_stride = ulong(n_groups) * 2ul;
  const uint out_is_f32 = params.flags & 1u;
  const float cap = params.softcap;

  const uint k_lane = lane & 7u;
  const uint r_id = lane >> 3u; // 0..3

  if (M == 1u) {
    const uint v = tgid_x * 16u + sg * 4u + r_id;
    if (v >= V) {
      return;
    }
    const float dot_v = inkling2_q8_sym_row_m1(
        x_u4, w_u4 + ulong(v) * row_u4_stride, scales + ulong(v) * ulong(n_groups),
        n_groups, k_lane);
    if (k_lane == 0u) {
      const float val = cap > 0.0f ? inkling2_softcap(dot_v, cap) : dot_v;
      inkling2_q8_sym_store_scalar(out_raw, ulong(v), val, out_is_f32);
    }
    return;
  }

  if (M <= 8u) {
    const uint v = tgid_x * 16u + sg * 4u + r_id;
    if (v >= V) {
      return;
    }
    float acc[8];
    inkling2_q8_sym_row_multi<8>(
        x_u4, w_u4 + ulong(v) * row_u4_stride, scales + ulong(v) * ulong(n_groups),
        0u, M, n_groups, k_lane, acc);
    if (k_lane == 0u) {
      #pragma unroll
      for (uint i = 0u; i < 8u; ++i) {
        if (i < M) {
          const float val = cap > 0.0f ? inkling2_softcap(acc[i], cap) : acc[i];
          inkling2_q8_sym_store_scalar(out_raw, ulong(i) * ulong(V) + ulong(v), val, out_is_f32);
        }
      }
    }
    return;
  }

  if (M <= 16u) {
    const uint v = tgid_x * 16u + sg * 4u + r_id;
    if (v >= V) {
      return;
    }
    float acc[16];
    inkling2_q8_sym_row_multi<16>(
        x_u4, w_u4 + ulong(v) * row_u4_stride, scales + ulong(v) * ulong(n_groups),
        0u, M, n_groups, k_lane, acc);
    if (k_lane == 0u) {
      #pragma unroll
      for (uint i = 0u; i < 16u; ++i) {
        if (i < M) {
          const float val = cap > 0.0f ? inkling2_softcap(acc[i], cap) : acc[i];
          inkling2_q8_sym_store_scalar(out_raw, ulong(i) * ulong(V) + ulong(v), val, out_is_f32);
        }
      }
    }
    return;
  }

  const uint sg_pair = sg >> 1u;
  const uint m_half = sg & 1u;
  const uint v = tgid_x * 8u + sg_pair * 4u + r_id;
  if (v >= V) {
    return;
  }
  const uint m0 = m_half * 16u;
  const uint n_m = min(16u, M - m0);
  float acc[16];
  inkling2_q8_sym_row_multi<16>(
      x_u4, w_u4 + ulong(v) * row_u4_stride, scales + ulong(v) * ulong(n_groups),
      m0, n_m, n_groups, k_lane, acc);
  if (k_lane == 0u) {
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
      if (i < n_m) {
        const float val = cap > 0.0f ? inkling2_softcap(acc[i], cap) : acc[i];
        inkling2_q8_sym_store_scalar(
            out_raw, ulong(m0 + i) * ulong(V) + ulong(v), val, out_is_f32);
      }
    }
  }
}

// (c) Symmetric Q8_0 dense projection matmul (`1 <= M <= 32`, `K % 32 == 0`) with optional
//     fused BF16 tail projection (`sink_proj` [24, K] or `k_conv_proj`/`v_conv_proj` [32, K])
//     in the EXACT SAME GPU dispatch!
// Grid: (q8_tgs + aux_tgs, 1, 1), TG: (128, 1, 1).
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_q8_sym_matmul_bf16(
    device const uint4 *x_u4 [[buffer(0)]],
    device const uint4 *w_u4 [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const uint4 *w_aux_u4 [[buffer(3)]],
    device uchar *out_raw [[buffer(4)]],
    device bfloat *out_aux [[buffer(5)]],
    constant Inkling2Q8SymParams &params [[buffer(6)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint M = params.rows;
  const uint N = params.out_dim;
  const uint n_groups = params.in_dim >> 5u;
  const uint out_is_f32 = params.flags & 1u;
  const bool has_aux = ((params.flags >> 1u) & 1u) != 0u;
  const uint rows_per_tg = (M <= 16u) ? 16u : 8u;
  const uint q8_tgs = (N + rows_per_tg - 1u) / rows_per_tg;

  const uint k_lane = lane & 7u;
  const uint r_id = lane >> 3u;

  if (tgid_x < q8_tgs) {
    const ulong row_u4_stride = ulong(n_groups) * 2ul;
    if (M == 1u) {
      const uint n = tgid_x * 16u + sg * 4u + r_id;
      if (n >= N) {
        return;
      }
      const float dot_v = inkling2_q8_sym_row_m1(
          x_u4, w_u4 + ulong(n) * row_u4_stride, scales + ulong(n) * ulong(n_groups),
          n_groups, k_lane);
      if (k_lane == 0u) {
        inkling2_q8_sym_store_scalar(out_raw, ulong(n), dot_v, out_is_f32);
      }
      return;
    }
    if (M <= 8u) {
      const uint n = tgid_x * 16u + sg * 4u + r_id;
      if (n >= N) {
        return;
      }
      float acc[8];
      inkling2_q8_sym_row_multi<8>(
          x_u4, w_u4 + ulong(n) * row_u4_stride, scales + ulong(n) * ulong(n_groups),
          0u, M, n_groups, k_lane, acc);
      if (k_lane == 0u) {
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
          if (i < M) {
            inkling2_q8_sym_store_scalar(
                out_raw, ulong(i) * ulong(N) + ulong(n), acc[i], out_is_f32);
          }
        }
      }
      return;
    }
    if (M <= 16u) {
      const uint n = tgid_x * 16u + sg * 4u + r_id;
      if (n >= N) {
        return;
      }
      float acc[16];
      inkling2_q8_sym_row_multi<16>(
          x_u4, w_u4 + ulong(n) * row_u4_stride, scales + ulong(n) * ulong(n_groups),
          0u, M, n_groups, k_lane, acc);
      if (k_lane == 0u) {
        #pragma unroll
        for (uint i = 0u; i < 16u; ++i) {
          if (i < M) {
            inkling2_q8_sym_store_scalar(
                out_raw, ulong(i) * ulong(N) + ulong(n), acc[i], out_is_f32);
          }
        }
      }
      return;
    }
    const uint sg_pair = sg >> 1u;
    const uint m_half = sg & 1u;
    const uint n = tgid_x * 8u + sg_pair * 4u + r_id;
    if (n >= N) {
      return;
    }
    const uint m0 = m_half * 16u;
    const uint n_m = min(16u, M - m0);
    float acc[16];
    inkling2_q8_sym_row_multi<16>(
        x_u4, w_u4 + ulong(n) * row_u4_stride, scales + ulong(n) * ulong(n_groups),
        m0, n_m, n_groups, k_lane, acc);
    if (k_lane == 0u) {
      #pragma unroll
      for (uint i = 0u; i < 16u; ++i) {
        if (i < n_m) {
          inkling2_q8_sym_store_scalar(
              out_raw, ulong(m0 + i) * ulong(N) + ulong(n), acc[i], out_is_f32);
        }
      }
    }
    return;
  }

  // Optional fused exact-BF16 tail projection (uniform across TG -> 0% divergence!).
  if (!has_aux) {
    return;
  }
  const uint N_aux = params.aux_dim;
  const uint aux_tg = tgid_x - q8_tgs;
  const ulong aux_row_u4 = ulong(n_groups) * 4ul;

  if (M <= 8u) {
    const uint n = aux_tg * 16u + sg * 4u + r_id;
    if (n >= N_aux) {
      return;
    }
    float acc[8];
    inkling2_bf16_aux_row_multi<8>(
        x_u4, w_aux_u4 + ulong(n) * aux_row_u4, 0u, M, n_groups, k_lane, acc);
    if (k_lane == 0u) {
      #pragma unroll
      for (uint i = 0u; i < 8u; ++i) {
        if (i < M) {
          out_aux[ulong(i) * ulong(N_aux) + ulong(n)] = bfloat(acc[i]);
        }
      }
    }
    return;
  }

  if (M <= 16u) {
    const uint n = aux_tg * 16u + sg * 4u + r_id;
    if (n >= N_aux) {
      return;
    }
    float acc[16];
    inkling2_bf16_aux_row_multi<16>(
        x_u4, w_aux_u4 + ulong(n) * aux_row_u4, 0u, M, n_groups, k_lane, acc);
    if (k_lane == 0u) {
      #pragma unroll
      for (uint i = 0u; i < 16u; ++i) {
        if (i < M) {
          out_aux[ulong(i) * ulong(N_aux) + ulong(n)] = bfloat(acc[i]);
        }
      }
    }
    return;
  }

  const uint sg_pair = sg >> 1u;
  const uint m_half = sg & 1u;
  const uint n = aux_tg * 8u + sg_pair * 4u + r_id;
  if (n >= N_aux) {
    return;
  }
  const uint m0 = m_half * 16u;
  const uint n_m = min(16u, M - m0);
  float acc[16];
  inkling2_bf16_aux_row_multi<16>(
      x_u4, w_aux_u4 + ulong(n) * aux_row_u4, m0, n_m, n_groups, k_lane, acc);
  if (k_lane == 0u) {
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
      if (i < n_m) {
        out_aux[ulong(m0 + i) * ulong(N_aux) + ulong(n)] = bfloat(acc[i]);
      }
    }
  }
}

// ============================================================================
// 19. Apple M5 Max NAX Symmetric Q8_0 (`mpp::tensor_ops::matmul2d`)
//     Single-pass 0-ULP exact dequant: EVERY int8 (`[-128, 127]`) is 100% BIT-EXACT
//     in `bfloat16`! 128 threads = 32 rows x 4 groups (1 full 32B Q8_0 group per thread,
//     4 threads = 128B cacheline per row, ZERO branch for scale load!).
//     Computes exact FP32 group dot `Dtmp = A_bf16 @ q_i8_bf16^T` via single NAX
//     `matmul2d(32, 32, 32, mode::multiply)` -> 2x fewer A reads & 2x fewer TGs on lm_head!
//     FMA `Dacc += Dtmp * scale_f16[n, g]`, 4 SGs split-K (KS=4), 12.5 KB overlaid smem,
//     0-ULP batch-size invariant across ALL `M in 1..=32`!
//
//     - `inkling2_q8_sym_matmul_nax_bf16`: dense Q8_0 (+ optional BF16 tail) & `lm_head`
//     - `inkling2_lm_head_q8_nax_softcap_argmax_p1`: zero-DRAM-logits softcap + argmax p1
// ============================================================================
inline void inkling2_stage_sym_q8_16_bf16(uint4 p, threadgroup bfloat *dst) {
  threadgroup bfloat4 *dst4 = reinterpret_cast<threadgroup bfloat4 *>(dst);
  dst4[0] = bfloat4(as_type<char4>(p.x ^ 0x80808080u));
  dst4[1] = bfloat4(as_type<char4>(p.y ^ 0x80808080u));
  dst4[2] = bfloat4(as_type<char4>(p.z ^ 0x80808080u));
  dst4[3] = bfloat4(as_type<char4>(p.w ^ 0x80808080u));
}

inline void inkling2_stage_zero_16_bf16(threadgroup bfloat *dst) {
  threadgroup bfloat4 *dst4 = reinterpret_cast<threadgroup bfloat4 *>(dst);
  const bfloat4 z = bfloat4(bfloat(0.0f));
  dst4[0] = z;
  dst4[1] = z;
  dst4[2] = z;
  dst4[3] = z;
}

inline void inkling2_stage_sym_q8_32_bf16(uint4 p0, uint4 p1, threadgroup bfloat *dst) {
  threadgroup bfloat4 *dst4 = reinterpret_cast<threadgroup bfloat4 *>(dst);
  dst4[0] = bfloat4(as_type<char4>(p0.x ^ 0x80808080u));
  dst4[1] = bfloat4(as_type<char4>(p0.y ^ 0x80808080u));
  dst4[2] = bfloat4(as_type<char4>(p0.z ^ 0x80808080u));
  dst4[3] = bfloat4(as_type<char4>(p0.w ^ 0x80808080u));
  dst4[4] = bfloat4(as_type<char4>(p1.x ^ 0x80808080u));
  dst4[5] = bfloat4(as_type<char4>(p1.y ^ 0x80808080u));
  dst4[6] = bfloat4(as_type<char4>(p1.z ^ 0x80808080u));
  dst4[7] = bfloat4(as_type<char4>(p1.w ^ 0x80808080u));
}

inline void inkling2_stage_zero_32_bf16(threadgroup bfloat *dst) {
  threadgroup bfloat4 *dst4 = reinterpret_cast<threadgroup bfloat4 *>(dst);
  const bfloat4 z = bfloat4(bfloat(0.0f));
  #pragma unroll
  for (uint i = 0u; i < 8u; ++i) {
    dst4[i] = z;
  }
}

// Small-M (`1 <= M <= 8`) Apple M5 Max NAX Symmetric Q8_0 matmul (+ optional BF16 tail) & `lm_head`.
// `TSM=8, TSN=16, BK=32, KS=4`, per-SG staging with register prefetch & `simdgroup_barrier`
// (ZERO K-loop `threadgroup_barrier`s, 2x more TGs -> saturates all 40 M5 Max cores, 4.8 KB smem!).
// Identical `BK=32, KS=4` NAX accumulation & SG reduction order as `inkling2_q8_sym_matmul_nax_bf16`
// -> 0-ULP bit-exact parity across ALL M = 1..=2048!
// Grid: (ceil(N / 16) + (has_aux ? ceil(N_aux / 16) : 0), 1, 1), TG: (128, 1, 1).
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_q8_sym_matmul_nax_m8_bf16(
    device const bfloat *x [[buffer(0)]],
    device const uint4 *w_u4 [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const bfloat *w_aux [[buffer(3)]],
    device uchar *out_raw [[buffer(4)]],
    device bfloat *out_aux [[buffer(5)]],
    constant Inkling2Q8SymParams &params [[buffer(6)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using dev_b_t = metal::tensor<device bfloat, extents_t, metal::tensor_inline>;
  using tg_b_t = metal::tensor<threadgroup bfloat, extents_t, metal::tensor_inline>;
  using tg_f_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;

  constexpr int TSM = 8;
  constexpr int TSN = 16;
  constexpr int BK = 32;
  constexpr int KS = 4;
  constexpr int STAGE_STRIDE = 136; // 128 + 8 bfloat padding -> 0 bank conflicts!
  constexpr uint BUF_ELEMS = uint(TSN * STAGE_STRIDE); // 2,176 bfloats = 4,352 B/buf

  const int M = int(params.rows);
  const int N = int(params.out_dim);
  const int K = int(params.in_dim);
  const uint n_groups = params.in_dim >> 5u;
  const uint q8_tgs = (params.out_dim + 15u) >> 4u;
  const uint out_is_f32 = params.flags & 1u;
  const bool has_aux = ((params.flags >> 1u) & 1u) != 0u;

  // 8,704 B shared buffer: double-buffered `stage[2][16 * 136]` bfloat during K-loop
  // (100% 128B full-cacheline coalesced loads, 1 barrier per 4 groups, overlapped prefetch!)
  // and `partials[3 * 8 * 16]` float (1,536 B) during KS=4 epilogue reduction!
  // Plus `stage_s[2][4][16]` float (512 B) -> 9.0 KB total smem!
  threadgroup float smem_buf[(2 * TSN * STAGE_STRIDE) / 2];
  threadgroup float stage_s[2][KS][TSN];
  threadgroup bfloat *stage = reinterpret_cast<threadgroup bfloat *>(smem_buf);

  if (tgid_x < q8_tgs) {
    constexpr auto desc_mul = matmul2d_descriptor(
        TSM,
        TSN,
        BK,
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc_mul, metal::execution_simdgroup> op_mul;

    const int n_off = int(tgid_x) * TSN;
    if (M <= 0 || n_off >= N) {
      return;
    }
    const int rows = min(TSM, M);
    const int cols = min(TSN, N - n_off);
    const ulong row_u4_stride = ulong(n_groups) * 2ul;

    device bfloat *A = const_cast<device bfloat *>(x);
    dev_b_t tA(A, extents_t(K, rows), strides_t{1, K});

    tg_b_t tB0(stage + int(sg) * BK, extents_t(BK, TSN), strides_t{1, STAGE_STRIDE});
    tg_b_t tB1(stage + BUF_ELEMS + int(sg) * BK, extents_t(BK, TSN), strides_t{1, STAGE_STRIDE});
    auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
    auto b_chunk0 = tB0.template slice<BK, TSN>(0, 0);
    auto b_chunk1 = tB1.template slice<BK, TSN>(0, 0);
    using a_chunk_t = decltype(a_chunk(0));
    using b_chunk_t = decltype(b_chunk0);

    auto Dacc = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    auto Dtmp = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] = 0.0f;
    }

    // 128 threads = 16 rows x 8 threads/row: 8 consecutive uint4 loads = 128B full cacheline!
    const uint r = tid >> 3u;       // 0..15 (16 output rows per TG)
    const uint c8 = tid & 7u;       // 0..7  (8 consecutive uint4s = 4 Q8_0 groups)
    const uint g4 = c8 >> 1u;       // 0..3  (group index within 4-group block)
    const uint half_g = c8 & 1u;    // 0..1  (low/high 16B uint4 of the 32B group)
    const uint row_idx = uint(n_off) + r;
    const bool valid_row = row_idx < params.out_dim;
    const device uint4 *w_row = w_u4 + ulong(row_idx) * row_u4_stride;
    const device half *s_row = scales + ulong(row_idx) * ulong(n_groups);
    const uint dst_off = r * uint(STAGE_STRIDE) + g4 * 32u + half_g * 16u;

    const bool full_m = (rows == TSM);
    if (n_groups > 0u) {
      if (valid_row && g4 < n_groups) {
        const uint4 p0 = w_row[g4 * 2u + half_g];
        inkling2_stage_sym_q8_16_bf16(p0, stage + dst_off);
        if (half_g == 0u) {
          stage_s[0][g4][r] = float(s_row[g4]);
        }
      } else {
        inkling2_stage_zero_16_bf16(stage + dst_off);
        if (half_g == 0u) {
          stage_s[0][g4][r] = 0.0f;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint g_base = 0u, buf = 0u; g_base < n_groups; g_base += 4u, buf ^= 1u) {
      const uint g_next_base = g_base + 4u;
      const uint g_next_load = g_next_base + g4;
      uint4 p_next = uint4(0u);
      float s_next = 0.0f;
      if (valid_row && g_next_load < n_groups) {
        p_next = w_row[g_next_load * 2u + half_g];
        if (half_g == 0u) {
          s_next = float(s_row[g_next_load]);
        }
      }

      const uint g_sg = g_base + sg;
      if (g_sg < n_groups) {
        const int k = int(g_sg * 32u);
        if (full_m) {
          auto a = a_chunk(k);
          if (buf == 0u) {
            op_mul.run(a, b_chunk0, Dtmp);
          } else {
            op_mul.run(a, b_chunk1, Dtmp);
          }
        } else {
          dev_b_t vA(A + k, extents_t(BK, rows), strides_t{1, K});
          if (buf == 0u) {
            op_mul.run(vA, b_chunk0, Dtmp);
          } else {
            op_mul.run(vA, b_chunk1, Dtmp);
          }
        }
        const threadgroup float *sc_ptr = stage_s[buf][sg];
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          const auto idx = Dtmp.get_multidimensional_index(i);
          Dacc[i] = fma(Dtmp[i], sc_ptr[idx[0]], Dacc[i]);
        }
      }

      if (g_next_base < n_groups) {
        const uint next_buf = buf ^ 1u;
        threadgroup bfloat *next_dst = stage + next_buf * BUF_ELEMS + dst_off;
        if (valid_row && g_next_load < n_groups) {
          inkling2_stage_sym_q8_16_bf16(p_next, next_dst);
          if (half_g == 0u) {
            stage_s[next_buf][g4][r] = s_next;
          }
        } else {
          inkling2_stage_zero_16_bf16(next_dst);
          if (half_g == 0u) {
            stage_s[next_buf][g4][r] = 0.0f;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    // Ensure all 4 SGs finish reading `stage` before overlaying `partials` in `smem_buf`.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr int kBlockSize = TSM * TSN; // 128 floats
    if (sg > 0u) {
      tg_f_t tP(smem_buf + (sg - 1u) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
      Dacc.store(tP);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool needs_scalar_epilogue = (out_is_f32 != 0u || params.softcap > 0.0f);
    if (sg == 0u) {
      #pragma unroll
      for (uint g = 0u; g < uint(KS - 1); ++g) {
        tg_f_t tP(smem_buf + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
        auto part = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
        part.load(tP);
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          Dacc[i] += part[i];
        }
      }

      if (!needs_scalar_epilogue) {
        auto Dout = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, bfloat>();
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          Dout[i] = bfloat(Dacc[i]);
        }
        device bfloat *D = reinterpret_cast<device bfloat *>(out_raw) + n_off;
        dev_b_t tD(D, extents_t(cols, rows), strides_t{1, N});
        if (full_m && cols == TSN) {
          Dout.store(tD.template slice<TSN, TSM>(0, 0));
        } else {
          Dout.store(tD);
        }
      } else {
        tg_f_t tOut(smem_buf, extents_t(TSN, TSM), strides_t{1, TSN});
        Dacc.store(tOut);
      }
    }
    if (!needs_scalar_epilogue) {
      return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float cap = params.softcap;
    const uint total = uint(rows) * uint(cols);
    for (uint idx = tid; idx < total; idx += 128u) {
      const uint r_out = idx / uint(cols);
      const uint c_out = idx - r_out * uint(cols);
      float v = smem_buf[r_out * uint(TSN) + c_out];
      if (cap > 0.0f) {
        v = inkling2_softcap(v, cap);
      }
      inkling2_q8_sym_store_scalar(
          out_raw, ulong(r_out) * ulong(N) + ulong(n_off) + ulong(c_out), v, out_is_f32);
    }
    return;
  }

  // Optional fused exact-BF16 tail projection (`TSM=8, TSN=16, BK_AUX=64, KS=4`).
  if (!has_aux) {
    return;
  }
  const int N_aux = int(params.aux_dim);
  const int n_off = int(tgid_x - q8_tgs) * TSN;
  if (M <= 0 || n_off >= N_aux) {
    return;
  }
  const int rows = min(TSM, M);
  const int cols = min(TSN, N_aux - n_off);
  const bool full_mn = (rows == TSM && cols == TSN);
  constexpr int BK_AUX = 64;
  constexpr auto desc_aux = matmul2d_descriptor(
      TSM,
      TSN,
      BK_AUX,
      false,
      true,
      true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_aux, metal::execution_simdgroup> op_aux;

  device bfloat *A = const_cast<device bfloat *>(x);
  device bfloat *B = const_cast<device bfloat *>(w_aux) + int64_t(n_off) * K;
  dev_b_t tA(A, extents_t(K, rows), strides_t{1, K});
  dev_b_t tB(B, extents_t(K, cols), strides_t{1, K});
  auto a_chunk = [&](int k) { return tA.template slice<BK_AUX, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK_AUX, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  auto Dacc = op_aux.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op_aux.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();

  const int k_iters = ((K + BK_AUX - 1) / BK_AUX + KS - 1) / KS;
  for (int it = 0; it < k_iters; ++it) {
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + int(sg)) * BK_AUX;
    if (k >= K) {
      continue;
    }
    if (full_mn && k + BK_AUX <= K) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op_aux.run(Areg, b, Dacc);
    } else {
      const int kk = min(BK_AUX, K - k);
      dev_b_t vA(A + k, extents_t(kk, rows), strides_t{1, K});
      dev_b_t vB(B + k, extents_t(kk, cols), strides_t{1, K});
      op_aux.run(vA, vB, Dacc);
    }
  }

  constexpr int kBlockSize = TSM * TSN; // 128 floats
  if (sg > 0u) {
    tg_f_t tP(smem_buf + (sg - 1u) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tP);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) {
    return;
  }
  #pragma unroll
  for (uint g = 0u; g < uint(KS - 1); ++g) {
    tg_f_t tP(smem_buf + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    auto part = op_aux.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    part.load(tP);
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] += part[i];
    }
  }
  auto Dout = op_aux.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, bfloat>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dout[i] = bfloat(Dacc[i]);
  }
  dev_b_t tD(out_aux + n_off, extents_t(cols, rows), strides_t{1, N_aux});
  if (full_mn) {
    Dout.store(tD.template slice<TSN, TSM>(0, 0));
  } else {
    Dout.store(tD);
  }
}

// Small-M (`1 <= M <= 8`), Wide-N (`N > 3072`, e.g. `qrgkv` [8960, K] & `lm_head` [200704, K])
// Apple M5 Max NAX Symmetric Q8_0 matmul (+ optional BF16 tail) & `lm_head`.
// `TSM=8, TSN=32, BK=32, KS=4`, 128B full-cacheline coalesced loads, 9.2 KB smem (vs 12.8 KB),
// 4x smaller NAX tile than `TSM=32`, half the `A` reads vs `TSN=16`!
// Identical `BK=32, KS=4` NAX accumulation & SG reduction order as `inkling2_q8_sym_matmul_nax_bf16`
// -> 0-ULP bit-exact parity across ALL M = 1..=2048!
// Grid: (ceil(N / 32) + (has_aux ? ceil(N_aux / 32) : 0), 1, 1), TG: (128, 1, 1).
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_q8_sym_matmul_nax_m8_n32_bf16(
    device const bfloat *x [[buffer(0)]],
    device const uint4 *w_u4 [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const bfloat *w_aux [[buffer(3)]],
    device uchar *out_raw [[buffer(4)]],
    device bfloat *out_aux [[buffer(5)]],
    constant Inkling2Q8SymParams &params [[buffer(6)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  (void)lane;
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using dev_b_t = metal::tensor<device bfloat, extents_t, metal::tensor_inline>;
  using tg_b_t = metal::tensor<threadgroup bfloat, extents_t, metal::tensor_inline>;
  using tg_f_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;

  constexpr int TSM = 8;
  constexpr int TSN = 32;
  constexpr int BK = 32;
  constexpr int KS = 4;
  constexpr int STAGE_STRIDE = 136; // 128 + 8 bfloat padding -> 0 bank conflicts!

  const int M = int(params.rows);
  const int N = int(params.out_dim);
  const int K = int(params.in_dim);
  const uint n_groups = params.in_dim >> 5u;
  const uint q8_tgs = (params.out_dim + 31u) >> 5u;
  const uint out_is_f32 = params.flags & 1u;
  const bool has_aux = ((params.flags >> 1u) & 1u) != 0u;

  // 8,704 B shared buffer: holds `stage[32 * 136]` bfloat (8,704 B) during K-loop
  // and `partials[3 * 8 * 32]` float (3,072 B) during KS=4 epilogue reduction!
  // Plus `stage_s[4][32]` float (512 B) -> 9.2 KB total smem!
  threadgroup float smem_buf[(TSN * STAGE_STRIDE) / 2];
  threadgroup float stage_s[KS][TSN];
  threadgroup bfloat *stage = reinterpret_cast<threadgroup bfloat *>(smem_buf);

  if (tgid_x < q8_tgs) {
    constexpr auto desc_mul = matmul2d_descriptor(
        TSM,
        TSN,
        BK,
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc_mul, metal::execution_simdgroup> op_mul;

    const int n_off = int(tgid_x) * TSN;
    if (M <= 0 || n_off >= N) {
      return;
    }
    const int rows = min(TSM, M);
    const int cols = min(TSN, N - n_off);
    const ulong row_u4_stride = ulong(n_groups) * 2ul;

    device bfloat *A = const_cast<device bfloat *>(x);
    dev_b_t tA(A, extents_t(K, rows), strides_t{1, K});

    tg_b_t tB(stage + int(sg) * BK, extents_t(BK, TSN), strides_t{1, STAGE_STRIDE});
    auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
    auto b_chunk = tB.template slice<BK, TSN>(0, 0);
    using a_chunk_t = decltype(a_chunk(0));
    using b_chunk_t = decltype(b_chunk);

    auto Dacc = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    auto Dtmp = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] = 0.0f;
    }

    const uint r = tid >> 2u;   // 0..31 (32 output rows)
    const uint g4 = tid & 3u;   // 0..3  (4 consecutive groups = 128B full cacheline!)
    const uint row_idx = uint(n_off) + r;
    const bool valid_row = row_idx < params.out_dim;
    const device uint4 *w_row = w_u4 + ulong(row_idx) * row_u4_stride;
    const device half *s_row = scales + ulong(row_idx) * ulong(n_groups);
    const uint dst_off = r * uint(STAGE_STRIDE) + g4 * 32u;

    const bool full_m = (rows == TSM);
    uint4 p0_cur = uint4(0u);
    uint4 p1_cur = uint4(0u);
    float s_cur = 0.0f;
    if (valid_row && g4 < n_groups) {
      p0_cur = w_row[g4 * 2u + 0u];
      p1_cur = w_row[g4 * 2u + 1u];
      s_cur = float(s_row[g4]);
    }
    for (uint g_base = 0u; g_base < n_groups; g_base += 4u) {
      const uint g_load = g_base + g4;
      if (valid_row && g_load < n_groups) {
        inkling2_stage_sym_q8_32_bf16(p0_cur, p1_cur, stage + dst_off);
        stage_s[g4][r] = s_cur;
      } else {
        inkling2_stage_zero_32_bf16(stage + dst_off);
        stage_s[g4][r] = 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint g_next = g_load + 4u;
      if (valid_row && g_next < n_groups) {
        p0_cur = w_row[g_next * 2u + 0u];
        p1_cur = w_row[g_next * 2u + 1u];
        s_cur = float(s_row[g_next]);
      }

      const uint g_sg = g_base + sg;
      if (g_sg < n_groups) {
        const int k = int(g_sg * 32u);
        if (full_m) {
          auto a = a_chunk(k);
          op_mul.run(a, b_chunk, Dtmp);
        } else {
          dev_b_t vA(A + k, extents_t(BK, rows), strides_t{1, K});
          op_mul.run(vA, b_chunk, Dtmp);
        }
        const threadgroup float *sc_ptr = stage_s[sg];
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          const auto idx = Dtmp.get_multidimensional_index(i);
          Dacc[i] = fma(Dtmp[i], sc_ptr[idx[0]], Dacc[i]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    constexpr int kBlockSize = TSM * TSN; // 256 floats
    if (sg > 0u) {
      tg_f_t tP(smem_buf + (sg - 1u) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
      Dacc.store(tP);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool needs_scalar_epilogue = (out_is_f32 != 0u || params.softcap > 0.0f);
    if (sg == 0u) {
      #pragma unroll
      for (uint g = 0u; g < uint(KS - 1); ++g) {
        tg_f_t tP(smem_buf + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
        auto part = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
        part.load(tP);
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          Dacc[i] += part[i];
        }
      }

      if (!needs_scalar_epilogue) {
        auto Dout = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, bfloat>();
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          Dout[i] = bfloat(Dacc[i]);
        }
        device bfloat *D = reinterpret_cast<device bfloat *>(out_raw) + n_off;
        dev_b_t tD(D, extents_t(cols, rows), strides_t{1, N});
        if (full_m && cols == TSN) {
          Dout.store(tD.template slice<TSN, TSM>(0, 0));
        } else {
          Dout.store(tD);
        }
      } else {
        tg_f_t tOut(smem_buf, extents_t(TSN, TSM), strides_t{1, TSN});
        Dacc.store(tOut);
      }
    }
    if (!needs_scalar_epilogue) {
      return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float cap = params.softcap;
    const uint total = uint(rows) * uint(cols);
    for (uint idx = tid; idx < total; idx += 128u) {
      const uint r_out = idx / uint(cols);
      const uint c_out = idx - r_out * uint(cols);
      float v = smem_buf[r_out * uint(TSN) + c_out];
      if (cap > 0.0f) {
        v = inkling2_softcap(v, cap);
      }
      inkling2_q8_sym_store_scalar(
          out_raw, ulong(r_out) * ulong(N) + ulong(n_off) + ulong(c_out), v, out_is_f32);
    }
    return;
  }

  // Optional fused exact-BF16 tail projection (`TSM=8, TSN=32, BK_AUX=64, KS=4`).
  if (!has_aux) {
    return;
  }
  const int N_aux = int(params.aux_dim);
  const int n_off = int(tgid_x - q8_tgs) * TSN;
  if (M <= 0 || n_off >= N_aux) {
    return;
  }
  const int rows = min(TSM, M);
  const int cols = min(TSN, N_aux - n_off);
  const bool full_mn = (rows == TSM && cols == TSN);
  constexpr int BK_AUX = 64;
  constexpr auto desc_aux = matmul2d_descriptor(
      TSM,
      TSN,
      BK_AUX,
      false,
      true,
      true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_aux, metal::execution_simdgroup> op_aux;

  device bfloat *A = const_cast<device bfloat *>(x);
  device bfloat *B = const_cast<device bfloat *>(w_aux) + int64_t(n_off) * K;
  dev_b_t tA(A, extents_t(K, rows), strides_t{1, K});
  dev_b_t tB(B, extents_t(K, cols), strides_t{1, K});
  auto a_chunk = [&](int k) { return tA.template slice<BK_AUX, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK_AUX, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  auto Dacc = op_aux.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op_aux.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();

  const int k_iters = ((K + BK_AUX - 1) / BK_AUX + KS - 1) / KS;
  for (int it = 0; it < k_iters; ++it) {
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + int(sg)) * BK_AUX;
    if (k >= K) {
      continue;
    }
    if (full_mn && k + BK_AUX <= K) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op_aux.run(Areg, b, Dacc);
    } else {
      const int kk = min(BK_AUX, K - k);
      dev_b_t vA(A + k, extents_t(kk, rows), strides_t{1, K});
      dev_b_t vB(B + k, extents_t(kk, cols), strides_t{1, K});
      op_aux.run(vA, vB, Dacc);
    }
  }

  constexpr int kBlockSize = TSM * TSN; // 256 floats
  if (sg > 0u) {
    tg_f_t tP(smem_buf + (sg - 1u) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tP);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) {
    return;
  }
  #pragma unroll
  for (uint g = 0u; g < uint(KS - 1); ++g) {
    tg_f_t tP(smem_buf + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    auto part = op_aux.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    part.load(tP);
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] += part[i];
    }
  }
  auto Dout = op_aux.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, bfloat>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dout[i] = bfloat(Dacc[i]);
  }
  dev_b_t tD(out_aux + n_off, extents_t(cols, rows), strides_t{1, N_aux});
  if (full_mn) {
    Dout.store(tD.template slice<TSN, TSM>(0, 0));
  } else {
    Dout.store(tD);
  }
}

// Grid: (ceil(N / 32) + (has_aux ? ceil(N_aux / 32) : 0), 1, 1), TG: (128, 1, 1) -> 4 SGs (KS=4).
// Handles BOTH `inkling2_q8_sym_matmul_bf16` (+ optional fused BF16 tail) AND `inkling2_lm_head_q8_bf16`
// (BF16 OR FP32 + optional softcap).
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_q8_sym_matmul_nax_bf16(
    device const bfloat *x [[buffer(0)]],
    device const uint4 *w_u4 [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const bfloat *w_aux [[buffer(3)]],
    device uchar *out_raw [[buffer(4)]],
    device bfloat *out_aux [[buffer(5)]],
    constant Inkling2Q8SymParams &params [[buffer(6)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  (void)lane;
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using dev_b_t = metal::tensor<device bfloat, extents_t, metal::tensor_inline>;
  using tg_b_t = metal::tensor<threadgroup bfloat, extents_t, metal::tensor_inline>;
  using tg_f_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;

  constexpr int TSM = 32;
  constexpr int TSN = 32;
  constexpr int BK = 32;
  constexpr int KS = 4;
  constexpr int STAGE_STRIDE = 136; // 128 + 8 bfloat padding -> 0 bank conflicts!

  const int M = int(params.rows);
  const int N = int(params.out_dim);
  const int K = int(params.in_dim);
  const uint n_groups = params.in_dim >> 5u;
  const uint tiles_m = (params.rows + 31u) >> 5u;
  const uint tiles_n = (params.out_dim + 31u) >> 5u;
  const uint q8_tgs = tiles_m * tiles_n;
  const uint out_is_f32 = params.flags & 1u;
  const bool has_aux = ((params.flags >> 1u) & 1u) != 0u;

  // 12,288 B shared buffer: holds `stage[32 * 136]` bfloat (8,704 B) during K-loop
  // and `partials[3 * 32 * 32]` float (12,288 B) during KS=4 epilogue reduction!
  // Plus `stage_s[4][32]` float (512 B) -> 12.5 KB total smem!
  threadgroup float smem_buf[(KS - 1) * TSM * TSN];
  threadgroup float stage_s[KS][TSN];
  threadgroup bfloat *stage = reinterpret_cast<threadgroup bfloat *>(smem_buf);

  if (tgid_x < q8_tgs) {
    constexpr auto desc_mul = matmul2d_descriptor(
        TSM,
        TSN,
        BK,
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc_mul, metal::execution_simdgroup> op_mul;

    const int tile_m = int(tgid_x % tiles_m);
    const int tile_n = int(tgid_x / tiles_m);
    const int m_off = tile_m * TSM;
    const int n_off = tile_n * TSN;
    if (m_off >= M || n_off >= N) {
      return;
    }
    const int rows = min(TSM, M - m_off);
    const int cols = min(TSN, N - n_off);
    const ulong row_u4_stride = ulong(n_groups) * 2ul;

    device bfloat *A = const_cast<device bfloat *>(x) + int64_t(m_off) * K;
    dev_b_t tA(A, extents_t(K, rows), strides_t{1, K});

    tg_b_t tB(stage + int(sg) * BK, extents_t(BK, TSN), strides_t{1, STAGE_STRIDE});
    auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
    auto b_chunk = tB.template slice<BK, TSN>(0, 0);
    using a_chunk_t = decltype(a_chunk(0));
    using b_chunk_t = decltype(b_chunk);

    auto Dacc = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    auto Dtmp = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] = 0.0f;
    }

    const uint r = tid >> 2u;   // 0..31 (32 output rows)
    const uint g4 = tid & 3u;   // 0..3  (4 consecutive groups = 128B full cacheline!)
    const uint row_idx = uint(n_off) + r;
    const bool valid_row = row_idx < params.out_dim;
    const device uint4 *w_row = w_u4 + ulong(row_idx) * row_u4_stride;
    const device half *s_row = scales + ulong(row_idx) * ulong(n_groups);
    const uint dst_off = r * uint(STAGE_STRIDE) + g4 * 32u;

    const bool full_m = (rows == TSM);
    for (uint g_base = 0u; g_base < n_groups; g_base += 4u) {
      const uint g_load = g_base + g4;
      if (valid_row && g_load < n_groups) {
        const uint4 p0 = w_row[g_load * 2u + 0u];
        const uint4 p1 = w_row[g_load * 2u + 1u];
        inkling2_stage_sym_q8_32_bf16(p0, p1, stage + dst_off);
        stage_s[g4][r] = float(s_row[g_load]);
      } else {
        inkling2_stage_zero_32_bf16(stage + dst_off);
        stage_s[g4][r] = 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint g_sg = g_base + sg;
      if (g_sg < n_groups) {
        const int k = int(g_sg * 32u);
        if (full_m) {
          auto a = a_chunk(k);
          op_mul.run(a, b_chunk, Dtmp);
        } else {
          dev_b_t vA(A + k, extents_t(BK, rows), strides_t{1, K});
          op_mul.run(vA, b_chunk, Dtmp);
        }
        const threadgroup float *sc_ptr = stage_s[sg];
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          const auto idx = Dtmp.get_multidimensional_index(i);
          Dacc[i] = fma(Dtmp[i], sc_ptr[idx[0]], Dacc[i]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    constexpr int kBlockSize = TSM * TSN; // 1024 floats
    if (sg > 0u) {
      tg_f_t tP(smem_buf + (sg - 1u) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
      Dacc.store(tP);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool needs_scalar_epilogue = (out_is_f32 != 0u || params.softcap > 0.0f);
    if (sg == 0u) {
      #pragma unroll
      for (uint g = 0u; g < uint(KS - 1); ++g) {
        tg_f_t tP(smem_buf + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
        auto part = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
        part.load(tP);
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          Dacc[i] += part[i];
        }
      }

      if (!needs_scalar_epilogue) {
        auto Dout = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, bfloat>();
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          Dout[i] = bfloat(Dacc[i]);
        }
        device bfloat *D = reinterpret_cast<device bfloat *>(out_raw) + int64_t(m_off) * N + n_off;
        dev_b_t tD(D, extents_t(cols, rows), strides_t{1, N});
        if (full_m && cols == TSN) {
          Dout.store(tD.template slice<TSN, TSM>(0, 0));
        } else {
          Dout.store(tD);
        }
      } else {
        // FP32 or softcap epilogue: dump reduced tile to smem for cooperative write.
        tg_f_t tOut(smem_buf, extents_t(TSN, TSM), strides_t{1, TSN});
        Dacc.store(tOut);
      }
    }
    if (!needs_scalar_epilogue) {
      return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float cap = params.softcap;
    const uint total = uint(rows) * uint(cols);
    for (uint idx = tid; idx < total; idx += 128u) {
      const uint r_out = idx / uint(cols);
      const uint c_out = idx - r_out * uint(cols);
      float v = smem_buf[r_out * uint(TSN) + c_out];
      if (cap > 0.0f) {
        v = inkling2_softcap(v, cap);
      }
      inkling2_q8_sym_store_scalar(
          out_raw, ulong(m_off + int(r_out)) * ulong(N) + ulong(n_off) + ulong(c_out), v, out_is_f32);
    }
    return;
  }

  // Optional fused exact-BF16 tail projection (`sink_proj` [24, K], `k_conv_proj`/`v_conv_proj` [32, K])
  // using identical 4-SG NAX `matmul2d(32, 32, 64)` (1 TG per M-tile covers all 24 or 32 rows!).
  if (!has_aux) {
    return;
  }
  const int N_aux = int(params.aux_dim);
  const uint aux_tgp = tgid_x - q8_tgs;
  const int tile_m = int(aux_tgp % tiles_m);
  const int tile_n = int(aux_tgp / tiles_m);
  const int m_off = tile_m * TSM;
  const int n_off = tile_n * TSN;
  if (m_off >= M || n_off >= N_aux) {
    return;
  }
  const int rows = min(TSM, M - m_off);
  constexpr int BK_AUX = 64;
  constexpr auto desc_aux = matmul2d_descriptor(
      TSM,
      TSN,
      BK_AUX,
      false,
      true,
      true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_aux, metal::execution_simdgroup> op_aux;

  const int cols = min(TSN, N_aux - n_off);
  const bool full_mn = (rows == TSM && cols == TSN);
  device bfloat *A = const_cast<device bfloat *>(x) + int64_t(m_off) * K;
  device bfloat *B = const_cast<device bfloat *>(w_aux) + int64_t(n_off) * K;
  dev_b_t tA(A, extents_t(K, rows), strides_t{1, K});
  dev_b_t tB(B, extents_t(K, cols), strides_t{1, K});
  auto a_chunk = [&](int k) { return tA.template slice<BK_AUX, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK_AUX, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  auto Dacc = op_aux.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op_aux.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();

  const int k_iters = ((K + BK_AUX - 1) / BK_AUX + KS - 1) / KS;
  for (int it = 0; it < k_iters; ++it) {
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + int(sg)) * BK_AUX;
    if (k >= K) {
      continue;
    }
    if (full_mn && k + BK_AUX <= K) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op_aux.run(Areg, b, Dacc);
    } else {
      const int kk = min(BK_AUX, K - k);
      dev_b_t vA(A + k, extents_t(kk, rows), strides_t{1, K});
      dev_b_t vB(B + k, extents_t(kk, cols), strides_t{1, K});
      op_aux.run(vA, vB, Dacc);
    }
  }

  constexpr int kBlockSize = TSM * TSN; // 1024 floats
  if (sg > 0u) {
    tg_f_t tP(smem_buf + (sg - 1u) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tP);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) {
    return;
  }
  #pragma unroll
  for (uint g = 0u; g < uint(KS - 1); ++g) {
    tg_f_t tP(smem_buf + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    auto part = op_aux.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    part.load(tP);
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] += part[i];
    }
  }
  auto Dout = op_aux.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, bfloat>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dout[i] = bfloat(Dacc[i]);
  }
  dev_b_t tD(out_aux + int64_t(m_off) * N_aux + n_off, extents_t(cols, rows), strides_t{1, N_aux});
  if (full_mn) {
    Dout.store(tD.template slice<TSN, TSM>(0, 0));
  } else {
    Dout.store(tD);
  }
}

// Fused NAX Symmetric Q8_0 `lm_head` + softcap + Stage 1 argmax (ZERO [M, V] DRAM write!).
// Grid: (num_splits = ceil(out_cols / 64), 1, 1), TG: (128, 1, 1) -> 4 SGs (KS=4), 12.5 KB smem!
// Processes 2 consecutive 32-row tiles per TG with IDENTICAL `matmul2d(32, 32, 32, mode::multiply)`
// & `KS=4` reduction as `inkling2_q8_sym_matmul_nax_bf16` -> 0-ULP bit-exact across ALL M=1..=32!
// Stage 2 reuses `inkling2_lm_head_softcap_argmax_stage2`.
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_lm_head_q8_nax_softcap_argmax_p1(
    device const bfloat *x [[buffer(0)]],
    device const uint4 *w_u4 [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *partial_vals [[buffer(3)]],
    device int32_t *partial_idxs [[buffer(4)]],
    constant Inkling2Q8SymParams &params [[buffer(5)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using dev_b_t = metal::tensor<device bfloat, extents_t, metal::tensor_inline>;
  using tg_b_t = metal::tensor<threadgroup bfloat, extents_t, metal::tensor_inline>;
  using tg_f_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;

  constexpr int TSM = 32;
  constexpr int TSN = 32;
  constexpr int BK = 32;
  constexpr int KS = 4;
  constexpr int TILES_PER_TG = 2; // 64 vocab rows per TG
  constexpr int STAGE_STRIDE = 136;

  const uint split = tgid_x;
  if (split >= params.num_splits) {
    return;
  }
  const int M = int(params.rows);
  const int V = int(params.out_dim);
  const int K = int(params.in_dim);
  const uint n_groups = params.in_dim >> 5u;
  const int base_n = int(split) * (TSN * TILES_PER_TG);
  if (base_n >= V || M <= 0) {
    return;
  }
  const ulong row_u4_stride = ulong(n_groups) * 2ul;

  constexpr auto desc_mul = matmul2d_descriptor(
      TSM,
      TSN,
      BK,
      false,
      true,
      true,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc_mul, metal::execution_simdgroup> op_mul;

  device bfloat *A = const_cast<device bfloat *>(x);
  dev_b_t tA(A, extents_t(K, M), strides_t{1, K});

  threadgroup float smem_buf[(KS - 1) * TSM * TSN];
  threadgroup float stage_s[KS][TSN];
  threadgroup bfloat *stage = reinterpret_cast<threadgroup bfloat *>(smem_buf);

  tg_b_t tB(stage + int(sg) * BK, extents_t(BK, TSN), strides_t{1, STAGE_STRIDE});
  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = tB.template slice<BK, TSN>(0, 0);
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk);

  const uint r = tid >> 2u;
  const uint g4 = tid & 3u;
  const uint dst_off = r * uint(STAGE_STRIDE) + g4 * 32u;
  const bool full_m = (M == TSM);
  const float cap = params.softcap;

  float best_v[8];
  int32_t best_i[8];
  #pragma unroll
  for (uint idx = 0u; idx < 8u; ++idx) {
    best_v[idx] = -INFINITY;
    best_i[idx] = -1;
  }

  for (int tile = 0; tile < TILES_PER_TG; ++tile) {
    const int n_off = base_n + tile * TSN;
    if (n_off >= V) {
      break;
    }
    const int cols = min(TSN, V - n_off);
    const uint row_idx = uint(n_off) + r;
    const bool valid_row = row_idx < params.out_dim;
    const device uint4 *w_row = w_u4 + ulong(row_idx) * row_u4_stride;
    const device half *s_row = scales + ulong(row_idx) * ulong(n_groups);

    auto Dacc = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    auto Dtmp = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] = 0.0f;
    }

    for (uint g_base = 0u; g_base < n_groups; g_base += 4u) {
      const uint g_load = g_base + g4;
      if (valid_row && g_load < n_groups) {
        const uint4 p0 = w_row[g_load * 2u + 0u];
        const uint4 p1 = w_row[g_load * 2u + 1u];
        inkling2_stage_sym_q8_32_bf16(p0, p1, stage + dst_off);
        stage_s[g4][r] = float(s_row[g_load]);
      } else {
        inkling2_stage_zero_32_bf16(stage + dst_off);
        stage_s[g4][r] = 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint g_sg = g_base + sg;
      if (g_sg < n_groups) {
        const int k = int(g_sg * 32u);
        if (full_m) {
          auto a = a_chunk(k);
          op_mul.run(a, b_chunk, Dtmp);
        } else {
          dev_b_t vA(A + k, extents_t(BK, M), strides_t{1, K});
          op_mul.run(vA, b_chunk, Dtmp);
        }
        const threadgroup float *sc_ptr = stage_s[sg];
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          const auto idx = Dtmp.get_multidimensional_index(i);
          Dacc[i] = fma(Dtmp[i], sc_ptr[idx[0]], Dacc[i]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    constexpr int kBlockSize = TSM * TSN; // 1024 floats
    if (sg > 0u) {
      tg_f_t tP(smem_buf + (sg - 1u) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
      Dacc.store(tP);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
      #pragma unroll
      for (uint g = 0u; g < uint(KS - 1); ++g) {
        tg_f_t tP(smem_buf + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
        auto part = op_mul.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
        part.load(tP);
        #pragma unroll
        for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
          Dacc[i] += part[i];
        }
      }
      tg_f_t tOut(smem_buf, extents_t(TSN, TSM), strides_t{1, TSN});
      Dacc.store(tOut);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All 4 SGs cooperatively softcap + reduce all 32 vocab cols (100% warp utilization!).
    for (uint idx = 0u, r_m = sg; r_m < uint(M); r_m += 4u, ++idx) {
      float tv = -INFINITY;
      int32_t ti = -1;
      if (lane < uint(cols)) {
        float val = smem_buf[r_m * uint(TSN) + lane];
        if (cap > 0.0f) {
          val = inkling2_softcap(val, cap);
        }
        tv = val;
        ti = int32_t(n_off + int(lane));
      }
      #pragma unroll
      for (ushort off = 16; off >= 1; off >>= 1) {
        const float ov = simd_shuffle_down(tv, off);
        const int32_t oi = simd_shuffle_down(ti, off);
        if (oi >= 0 && (ti < 0 || ov > tv || (ov == tv && oi < ti))) {
          tv = ov;
          ti = oi;
        }
      }
      if (lane == 0u && ti >= 0) {
        if (best_i[idx] < 0 || tv > best_v[idx] || (tv == best_v[idx] && ti < best_i[idx])) {
          best_v[idx] = tv;
          best_i[idx] = ti;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0u) {
    for (uint idx = 0u, r_m = sg; r_m < uint(M); r_m += 4u, ++idx) {
      const ulong out_slot = ulong(r_m) * ulong(params.num_splits) + ulong(split);
      partial_vals[out_slot] = best_v[idx];
      partial_idxs[out_slot] = best_i[idx];
    }
  }
}

// ============================================================================
// 20. Pure-SIMD 128-bit `uint4` BF16 Linear & Fused `lm_head` + Softcap + Argmax
//     for non-NAX Apple7-Apple9 GPUs (M1 / M2 / M3 / M4, `1 <= M <= 32`, `K % 4 == 0`).
//     - Zero MPP/NAX tensor ops, zero threadgroup memory for `linear_simd`.
//     - 8 lanes per output row (`kLanes = 8`): 8 consecutive `uint4`s = 128B full cacheline
//       per coalesced load, 2x unrolled (256B in-flight) for `K % 8 == 0`, with an aligned
//       `ushort4` fallback when `K % 8 == 4`.
//     - 4 rows/SG, 16 rows/TG (`M <= 16`, 1 DRAM pass) or 8 rows/TG (`17 <= M <= 32`, pairs
//       of SGs share `w` in L1 while splitting `M` into `[0..16)` and `[16..M)`).
//     - Identical accumulation & 8-lane tree reduction order across ALL `M in 1..=32` AND
//       between `linear_simd` and `lm_head_softcap_argmax_simd_stage1` -> 0-ULP batch-size
//       invariance and 100% bit-exact parity with `FullSoftcap`!
// ============================================================================
inline float inkling2_bf16_simd_dot16(
    float4 w0, float4 w1, float4 w2, float4 w3, uint4 xr0, uint4 xr1) {
  float4 d = w0 * float4(as_type<bfloat4>(xr0.xy));
  d = fma(w1, float4(as_type<bfloat4>(xr0.zw)), d);
  d = fma(w2, float4(as_type<bfloat4>(xr1.xy)), d);
  d = fma(w3, float4(as_type<bfloat4>(xr1.zw)), d);
  return (d.x + d.y) + (d.z + d.w);
}

inline float inkling2_bf16_simd_dot8(float4 w0, float4 w1, uint4 xr) {
  float4 d = w0 * float4(as_type<bfloat4>(xr.xy));
  d = fma(w1, float4(as_type<bfloat4>(xr.zw)), d);
  return (d.x + d.y) + (d.z + d.w);
}

inline float inkling2_bf16_simd_dot4(float4 w0, ushort4 xr) {
  const float4 d = w0 * float4(as_type<bfloat4>(xr));
  return (d.x + d.y) + (d.z + d.w);
}

template <uint M_CAP>
inline void inkling2_bf16_simd_row_multi(
    device const bfloat *x_raw,
    device const bfloat *w_row_raw,
    uint m0,
    uint n_m,
    uint K,
    uint k_lane,
    thread float (&acc)[M_CAP]) {
  #pragma unroll
  for (uint i = 0u; i < M_CAP; ++i) {
    acc[i] = 0.0f;
  }

  if ((K & 7u) == 0u) {
    const uint n_vec8 = K >> 3u;
    const ulong x_row_u4 = ulong(n_vec8);
    const device uint4 *w_u4 = reinterpret_cast<device const uint4 *>(w_row_raw);
    const device uint4 *x_base =
        reinterpret_cast<device const uint4 *>(x_raw) + ulong(m0) * x_row_u4;

    uint kv = k_lane;
    for (; kv + 8u < n_vec8; kv += 16u) {
      const uint4 wr0 = w_u4[kv];
      const uint4 wr1 = w_u4[kv + 8u];
      const float4 w0 = float4(as_type<bfloat4>(wr0.xy));
      const float4 w1 = float4(as_type<bfloat4>(wr0.zw));
      const float4 w2 = float4(as_type<bfloat4>(wr1.xy));
      const float4 w3 = float4(as_type<bfloat4>(wr1.zw));
      const device uint4 *xk = x_base + ulong(kv);
      #pragma unroll
      for (uint i = 0u; i < M_CAP; ++i) {
        if (i < n_m) {
          const device uint4 *xr = xk + ulong(i) * x_row_u4;
          acc[i] += inkling2_bf16_simd_dot16(w0, w1, w2, w3, xr[0], xr[8]);
        }
      }
    }
    if (kv < n_vec8) {
      const uint4 wr0 = w_u4[kv];
      const float4 w0 = float4(as_type<bfloat4>(wr0.xy));
      const float4 w1 = float4(as_type<bfloat4>(wr0.zw));
      const device uint4 *xk = x_base + ulong(kv);
      #pragma unroll
      for (uint i = 0u; i < M_CAP; ++i) {
        if (i < n_m) {
          acc[i] += inkling2_bf16_simd_dot8(w0, w1, xk[ulong(i) * x_row_u4]);
        }
      }
    }
  } else {
    // Aligned 8-byte `ushort4` fallback for rare `K % 8 == 4` shapes.
    const uint n_vec4 = K >> 2u;
    const ulong x_row_u2 = ulong(n_vec4);
    const device ushort4 *w_u2 = reinterpret_cast<device const ushort4 *>(w_row_raw);
    const device ushort4 *x_base =
        reinterpret_cast<device const ushort4 *>(x_raw) + ulong(m0) * x_row_u2;
    for (uint kv = k_lane; kv < n_vec4; kv += 8u) {
      const float4 w0 = float4(as_type<bfloat4>(w_u2[kv]));
      const device ushort4 *xk = x_base + ulong(kv);
      #pragma unroll
      for (uint i = 0u; i < M_CAP; ++i) {
        if (i < n_m) {
          acc[i] += inkling2_bf16_simd_dot4(w0, xk[ulong(i) * x_row_u2]);
        }
      }
    }
  }

  #pragma unroll
  for (uint i = 0u; i < M_CAP; ++i) {
    if (i < n_m) {
      acc[i] = inkling2_q8_sym_reduce8(acc[i]);
    }
  }
}

// Pure-SIMD BF16 linear `x [M, K] @ w [N, K]^T -> [M, N] BF16` (`1 <= M <= 32`, `K % 4 == 0`).
// Grid: (ceil(N / rows_per_tg), 1, 1), TG: (128, 1, 1) -> rows_per_tg = 16 (M<=16) or 8 (M>16).
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_bf16_linear_simd(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant Inkling2Bf16LinearParams &params [[buffer(3)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint M = params.rows;
  const uint N = params.out_dim;
  const uint K = params.in_dim;

  const uint k_lane = lane & 7u;
  const uint r_id = lane >> 3u; // 0..3

  if (M == 1u) {
    const uint n = tgid_x * 16u + sg * 4u + r_id;
    if (n >= N) {
      return;
    }
    float acc[1];
    inkling2_bf16_simd_row_multi<1>(x, w + ulong(n) * ulong(K), 0u, 1u, K, k_lane, acc);
    if (k_lane == 0u) {
      out[n] = bfloat(acc[0]);
    }
    return;
  }

  if (M <= 8u) {
    const uint n = tgid_x * 16u + sg * 4u + r_id;
    if (n >= N) {
      return;
    }
    float acc[8];
    inkling2_bf16_simd_row_multi<8>(x, w + ulong(n) * ulong(K), 0u, M, K, k_lane, acc);
    if (k_lane == 0u) {
      #pragma unroll
      for (uint i = 0u; i < 8u; ++i) {
        if (i < M) {
          out[ulong(i) * ulong(N) + ulong(n)] = bfloat(acc[i]);
        }
      }
    }
    return;
  }

  if (M <= 16u) {
    const uint n = tgid_x * 16u + sg * 4u + r_id;
    if (n >= N) {
      return;
    }
    float acc[16];
    inkling2_bf16_simd_row_multi<16>(x, w + ulong(n) * ulong(K), 0u, M, K, k_lane, acc);
    if (k_lane == 0u) {
      #pragma unroll
      for (uint i = 0u; i < 16u; ++i) {
        if (i < M) {
          out[ulong(i) * ulong(N) + ulong(n)] = bfloat(acc[i]);
        }
      }
    }
    return;
  }

  const uint sg_pair = sg >> 1u;
  const uint m_half = sg & 1u;
  const uint n = tgid_x * 8u + sg_pair * 4u + r_id;
  if (n >= N) {
    return;
  }
  const uint m0 = m_half * 16u;
  const uint n_m = min(16u, M - m0);
  float acc[16];
  inkling2_bf16_simd_row_multi<16>(x, w + ulong(n) * ulong(K), m0, n_m, K, k_lane, acc);
  if (k_lane == 0u) {
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
      if (i < n_m) {
        out[ulong(m0 + i) * ulong(N) + ulong(n)] = bfloat(acc[i]);
      }
    }
  }
}

// Pure-SIMD Fused BF16 `lm_head` + softcap + Stage 1 argmax (ZERO `[M, V]` DRAM write!).
// Grid: (num_splits, 1, 1), TG: (128, 1, 1) -> 4 SGs.
// Stage 2 reuses `inkling2_lm_head_softcap_argmax_stage2`.
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_bf16_lm_head_softcap_argmax_simd_stage1(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device float *partial_vals [[buffer(2)]],
    device int32_t *partial_idxs [[buffer(3)]],
    constant Inkling2Bf16LinearParams &params [[buffer(4)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint split = tgid_x;
  if (split >= params.num_splits) {
    return;
  }
  const uint M = params.rows;
  const uint V = params.out_dim;
  const uint K = params.in_dim;
  const float cap = params.softcap;

  const uint chunk = (V + params.num_splits - 1u) / params.num_splits;
  const uint start = split * chunk;
  const uint end = min(start + chunk, V);

  const uint k_lane = lane & 7u;
  const uint r_id = lane >> 3u; // 0..3

  threadgroup float tg_v[32][4];
  threadgroup int32_t tg_i[32][4];

  if (M == 1u) {
    float best_v = -INFINITY;
    int32_t best_i = -1;
    for (uint v_base = start; v_base < end; v_base += 16u) {
      const uint v = v_base + sg * 4u + r_id;
      if (v < end) {
        float acc[1];
        inkling2_bf16_simd_row_multi<1>(x, w + ulong(v) * ulong(K), 0u, 1u, K, k_lane, acc);
        if (k_lane == 0u) {
          float val = float(bfloat(acc[0]));
          if (cap > 0.0f) {
            val = inkling2_softcap(val, cap);
          }
          const int32_t ci = int32_t(v);
          if (best_i < 0 || val > best_v || (val == best_v && ci < best_i)) {
            best_v = val;
            best_i = ci;
          }
        }
      }
    }
    inkling2_argmax_sg_reduce4(best_v, best_i);
    if (lane == 0u) {
      tg_v[0][sg] = best_v;
      tg_i[0][sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
      float bv = tg_v[0][0];
      int32_t bi = tg_i[0][0];
      #pragma unroll
      for (uint s = 1u; s < 4u; ++s) {
        const float ov = tg_v[0][s];
        const int32_t oi = tg_i[0][s];
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
          bv = ov;
          bi = oi;
        }
      }
      partial_vals[split] = bv;
      partial_idxs[split] = bi;
    }
    return;
  }

  if (M <= 8u) {
    float best_v[8];
    int32_t best_i[8];
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
      best_v[i] = -INFINITY;
      best_i[i] = -1;
    }
    for (uint v_base = start; v_base < end; v_base += 16u) {
      const uint v = v_base + sg * 4u + r_id;
      if (v < end) {
        float acc[8];
        inkling2_bf16_simd_row_multi<8>(x, w + ulong(v) * ulong(K), 0u, M, K, k_lane, acc);
        if (k_lane == 0u) {
          const int32_t ci = int32_t(v);
          #pragma unroll
          for (uint i = 0u; i < 8u; ++i) {
            if (i < M) {
              float val = float(bfloat(acc[i]));
              if (cap > 0.0f) {
                val = inkling2_softcap(val, cap);
              }
              if (best_i[i] < 0 || val > best_v[i] || (val == best_v[i] && ci < best_i[i])) {
                best_v[i] = val;
                best_i[i] = ci;
              }
            }
          }
        }
      }
    }
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
      if (i < M) {
        inkling2_argmax_sg_reduce4(best_v[i], best_i[i]);
        if (lane == 0u) {
          tg_v[i][sg] = best_v[i];
          tg_i[i][sg] = best_i[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < M) {
      float bv = tg_v[lane][0];
      int32_t bi = tg_i[lane][0];
      #pragma unroll
      for (uint s = 1u; s < 4u; ++s) {
        const float ov = tg_v[lane][s];
        const int32_t oi = tg_i[lane][s];
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
          bv = ov;
          bi = oi;
        }
      }
      const ulong out_slot = ulong(lane) * ulong(params.num_splits) + ulong(split);
      partial_vals[out_slot] = bv;
      partial_idxs[out_slot] = bi;
    }
    return;
  }

  if (M <= 16u) {
    float best_v[16];
    int32_t best_i[16];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
      best_v[i] = -INFINITY;
      best_i[i] = -1;
    }
    for (uint v_base = start; v_base < end; v_base += 16u) {
      const uint v = v_base + sg * 4u + r_id;
      if (v < end) {
        float acc[16];
        inkling2_bf16_simd_row_multi<16>(x, w + ulong(v) * ulong(K), 0u, M, K, k_lane, acc);
        if (k_lane == 0u) {
          const int32_t ci = int32_t(v);
          #pragma unroll
          for (uint i = 0u; i < 16u; ++i) {
            if (i < M) {
              float val = float(bfloat(acc[i]));
              if (cap > 0.0f) {
                val = inkling2_softcap(val, cap);
              }
              if (best_i[i] < 0 || val > best_v[i] || (val == best_v[i] && ci < best_i[i])) {
                best_v[i] = val;
                best_i[i] = ci;
              }
            }
          }
        }
      }
    }
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
      if (i < M) {
        inkling2_argmax_sg_reduce4(best_v[i], best_i[i]);
        if (lane == 0u) {
          tg_v[i][sg] = best_v[i];
          tg_i[i][sg] = best_i[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < M) {
      float bv = tg_v[lane][0];
      int32_t bi = tg_i[lane][0];
      #pragma unroll
      for (uint s = 1u; s < 4u; ++s) {
        const float ov = tg_v[lane][s];
        const int32_t oi = tg_i[lane][s];
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
          bv = ov;
          bi = oi;
        }
      }
      const ulong out_slot = ulong(lane) * ulong(params.num_splits) + ulong(split);
      partial_vals[out_slot] = bv;
      partial_idxs[out_slot] = bi;
    }
    return;
  }

  // 17 <= M <= 32: pairs of SGs share each vocab row in L1 while splitting M into [0..16) / [16..M).
  const uint sg_pair = sg >> 1u;
  const uint m_half = sg & 1u;
  const uint m0 = m_half * 16u;
  const uint n_m = min(16u, M - m0);
  float best_v[16];
  int32_t best_i[16];
  #pragma unroll
  for (uint i = 0u; i < 16u; ++i) {
    best_v[i] = -INFINITY;
    best_i[i] = -1;
  }
  for (uint v_base = start; v_base < end; v_base += 8u) {
    const uint v = v_base + sg_pair * 4u + r_id;
    if (v < end) {
      float acc[16];
      inkling2_bf16_simd_row_multi<16>(x, w + ulong(v) * ulong(K), m0, n_m, K, k_lane, acc);
      if (k_lane == 0u) {
        const int32_t ci = int32_t(v);
        #pragma unroll
        for (uint i = 0u; i < 16u; ++i) {
          if (i < n_m) {
            float val = float(bfloat(acc[i]));
            if (cap > 0.0f) {
              val = inkling2_softcap(val, cap);
            }
            if (best_i[i] < 0 || val > best_v[i] || (val == best_v[i] && ci < best_i[i])) {
              best_v[i] = val;
              best_i[i] = ci;
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (uint i = 0u; i < 16u; ++i) {
    if (i < n_m) {
      inkling2_argmax_sg_reduce4(best_v[i], best_i[i]);
      if (lane == 0u) {
        tg_v[m0 + i][sg_pair] = best_v[i];
        tg_i[m0 + i][sg_pair] = best_i[i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane < M) {
    float bv = tg_v[lane][0];
    int32_t bi = tg_i[lane][0];
    const float ov = tg_v[lane][1];
    const int32_t oi = tg_i[lane][1];
    if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
      bv = ov;
      bi = oi;
    }
    const ulong out_slot = ulong(lane) * ulong(params.num_splits) + ulong(split);
    partial_vals[out_slot] = bv;
    partial_idxs[out_slot] = bi;
  }
}
