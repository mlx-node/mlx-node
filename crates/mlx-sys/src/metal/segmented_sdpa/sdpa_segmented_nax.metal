// Verify first pass on the Metal 4 tensor op (mpp::tensor_ops::matmul2d,
// "NAX", gen-17+ GPUs). Same contract as segmented_sdpa_verify_tile_2pass_1
// in sdpa_segmented.metal: one threadgroup per (KV head, batch, partition)
// serves the M = gqa * rows queries of one KV head over a contiguous key
// range and writes partials, sums and maxs in sdpa_vector_2pass_2's layout.
// The matmuls read Q, K and V straight from device memory as tensor
// operands (no threadgroup staging); only the fp32 scores, the BF16
// probabilities and the row statistics live in threadgroup memory.
//
// Query order is head-major, m = h * rows + r, so the M rows are one strided
// tensor; the host dispatches this kernel only when q.strides(1) ==
// rows * q.strides(2). The softmax follows Splash's page softmax
// (runtime/metal/kernels/common/paged_attention_tile.h, Apache-2.0; see
// THIRD_PARTY_NOTICES): four lanes own one fused row, the row max and sum are
// two xor shuffles, and a shared flag skips the O rescale when no row max
// moved. Scores are kept in log2 units (exp2), as the simdgroup-matrix tile
// kernel does; `maxs` leave in natural units.
//
// 256 threads = execution_simdgroups<8> is a structural constant: the op is
// undefined for any other count. build.rs prebuilds the instantiations at
// the end of this file into paged_attn.metallib.
//
// INT8 K/V (KV = int8_t, Splash's target KV format): the int8 rows are
// matmul2d operands directly (bfloat x int8 -> float is a supported pair);
// one fp32 scale per (token, head) multiplies the key's column of S before
// the softmax and the value's softmax weight before P V, as Splash's page
// softmax does (paged_attention_tile.h). Both segments carry the same
// element type, so one running P V accumulator serves prefix and new rows.
// The scale buffers are bound for every instantiation; the BF16 ones never
// read them.

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

namespace sdpa_nax {

constant constexpr float kFiniteMin = -numeric_limits<float>::max();
constant constexpr int kThreads = 256;
constant constexpr int kSimdgroups = kThreads / 32;
constant constexpr int kLanesPerRow = 4;

template <bool B>
struct BoolTag {
  static constant constexpr bool value = B;
};

} // namespace sdpa_nax

template <typename T, int D, int M, int N, typename KV = T>
[[kernel, max_total_threads_per_threadgroup(256)]] void
segmented_sdpa_verify_nax_2pass_1(
    device T* queries [[buffer(0)]],
    device KV* prefix_keys [[buffer(1)]],
    device KV* prefix_values [[buffer(2)]],
    device KV* new_keys [[buffer(3)]],
    device KV* new_values [[buffer(4)]],
    device T* out [[buffer(5)]],
    device float* sums [[buffer(6)]],
    device float* maxs [[buffer(7)]],
    const constant int& prefix_n [[buffer(8)]],
    const constant int& new_n [[buffer(9)]],
    const constant long* strides [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const constant int& gqa [[buffer(12)]],
    const constant int& rows [[buffer(13)]],
    const constant int& partitions [[buffer(14)]],
    const device float* prefix_key_scales [[buffer(15)]],
    const device float* prefix_value_scales [[buffer(16)]],
    const device float* new_key_scales [[buffer(17)]],
    const device float* new_value_scales [[buffer(18)]],
    const constant long* scale_strides [[buffer(19)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint lid [[thread_index_in_threadgroup]]) {
  using namespace sdpa_nax;
  static_assert(M % 8 == 0 && M >= 8 && M <= kThreads / kLanesPerRow,
                "M = gqa * rows: a multiple of 8, one softmax lane quartet "
                "per row");
  static_assert(N % (4 * kLanesPerRow) == 0, "keys per lane are float4s");
  constexpr int KPL = N / kLanesPerRow;
  constexpr bool quantized = is_same<KV, int8_t>::value;

  // Threadgroup memory, set by the host (segmented_nax_scratch_bytes): fp32
  // scores [M][N], BF16 probabilities [M][N], fp32 previous scale [M], two
  // rescale flags.
  threadgroup float* st = scratch;
  threadgroup T* pt = reinterpret_cast<threadgroup T*>(st + M * N);
  threadgroup float* prev_scale =
      reinterpret_cast<threadgroup float*>(pt + M * N);
  threadgroup atomic_uint* rescale =
      reinterpret_cast<threadgroup atomic_uint*>(prev_scale + M);

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int num_q_heads = tpg.x * gqa;
  const int n = prefix_n + new_n;
  const int chunk = (n + partitions - 1) / partitions;
  const int kstart = block_idx * chunk;
  const int kend = min(n, kstart + chunk);

  // This lane's fused row (head-major) in the softmax pass.
  const bool softmax_lane = int(lid) < kLanesPerRow * M;
  const int m = int(lid) / kLanesPerRow;
  const int c0 = (int(lid) % kLanesPerRow) * KPL;
  const int q_seq_idx = m % rows;
  const int limit = min(kend, n - rows + q_seq_idx + 1);
  const float scale2 = scale * M_LOG2E_F;
  float row_max = kFiniteMin;
  float row_sum = 0;

  // Q: the gqa heads of this KV head are M = gqa * rows contiguous rows.
  device T* q_base = queries + batch_idx * strides[0] +
      long(gqa * kv_head_idx) * strides[1];
  auto qt = tensor(q_base, dextents<int, 2>{D, M},
                   array<int, 2>{1, int(strides[2])});
  auto q0 = qt.template slice<D, M>(0, 0);
  auto s_tensor = tensor(st, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto s0 = s_tensor.template slice<N, M>(0, 0);
  auto p_tensor = tensor(pt, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = p_tensor.template slice<N, M>(0, 0);
  auto kv_type = tensor(static_cast<device KV*>(nullptr),
                        dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto kv0 = kv_type.template slice<D, N>(0, 0);

  constexpr auto qk_desc = matmul2d_descriptor(
      M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_desc = matmul2d_descriptor(
      M, D, N, false, false, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<kSimdgroups>> qk;
  matmul2d<pv_desc, execution_simdgroups<kSimdgroups>> pv;
  // Zeroed here, not in a helper: a returned initialized cooperative tensor
  // loses its values (Splash gguf_staged_tile.h).
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(kv0), float>();
  const bool running_full =
      uint(running.get_capacity()) * uint(kThreads) == uint(M) * uint(D);
#pragma unroll
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (running_full || running.is_valid_element(i)) {
      running[i] = 0.0f;
    }
  }
  if (lid == 0) {
    atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    atomic_store_explicit(rescale + 1, 0u, memory_order_relaxed);
  }

  int tile = 0;
  // Keys [k0, k0 + rem) of one segment: S = Q K^T into st, softmax into pt,
  // O = O * scale + P V. A full tile (rem == N) takes static slices; a
  // partial one keeps the dynamic extents {D, rem} so the op reads nothing
  // past them (a static slice reads all N rows whatever the extents).
  // `k_scales` / `v_scales` point at the segment's first scale (token stride
  // 1); int8 only.
  auto process = [&](device KV* k_ptr, int k_seq_stride, device KV* v_ptr,
                     int v_seq_stride, const device float* k_scales,
                     const device float* v_scales, int k0, int rem,
                     auto tag) {
    constexpr bool full = decltype(tag)::value;
    auto kt = tensor(k_ptr, dextents<int, 2>{D, rem},
                     array<int, 2>{1, k_seq_stride});
    auto s = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(kv0), float>();
    if constexpr (full) {
      auto ks = kt.template slice<D, N>(0, 0);
      qk.run(q0, ks, s);
    } else {
      auto ks = kt.slice(0, 0);
      qk.run(q0, ks, s);
    }
    s.store(s0);
    threadgroup atomic_uint* flag = rescale + (tile & 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (softmax_lane) {
      threadgroup const float4* s4 =
          reinterpret_cast<threadgroup const float4*>(st + m * N + c0);
      float sc[KPL];
#pragma unroll
      for (int j = 0; j < KPL / 4; ++j) {
        const float4 v = s4[j] * scale2;
        sc[4 * j] = v.x;
        sc[4 * j + 1] = v.y;
        sc[4 * j + 2] = v.z;
        sc[4 * j + 3] = v.w;
      }
      const int visible = min(rem, limit - k0);
      // Key scales: one per column, only where the tile holds a key (the
      // scale array ends with the segment).
      float vs[KPL];
      if (quantized) {
#pragma unroll
        for (int j = 0; j < KPL; ++j) {
          const bool present = c0 + j < rem;
          sc[j] *= present ? k_scales[c0 + j] : 1.0f;
          vs[j] = present ? v_scales[c0 + j] : 0.0f;
        }
      }
      float tile_max = kFiniteMin;
#pragma unroll
      for (int j = 0; j < KPL; ++j) {
        sc[j] = c0 + j < visible ? sc[j] : kFiniteMin;
        tile_max = max(tile_max, sc[j]);
      }
      tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
      tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(2)));
      const float new_max = max(row_max, tile_max);
      // A row with no visible key yet: keep the exp2 arguments finite.
      const float ref = new_max == kFiniteMin ? 0.0f : new_max;
      const float factor = fast::exp2(row_max - ref);
      // P in BF16 for the matmul; the row sum uses the same rounded weights.
      // INT8: the stored P carries the value scale (so P V dequantizes V),
      // while the row sum takes the unscaled fp32 weights, as Splash does.
      float local_sum = 0;
      threadgroup vec<T, 4>* p4 =
          reinterpret_cast<threadgroup vec<T, 4>*>(pt + m * N + c0);
#pragma unroll
      for (int j = 0; j < KPL / 4; ++j) {
        const float4 e = fast::exp2(
            float4(sc[4 * j], sc[4 * j + 1], sc[4 * j + 2], sc[4 * j + 3]) -
            ref);
        if (quantized) {
          p4[j] = vec<T, 4>(
              e * float4(vs[4 * j], vs[4 * j + 1], vs[4 * j + 2], vs[4 * j + 3]));
          local_sum += e.x + e.y + e.z + e.w;
        } else {
          const vec<T, 4> p = vec<T, 4>(e);
          p4[j] = p;
          local_sum += float(p.x) + float(p.y) + float(p.z) + float(p.w);
        }
      }
      local_sum += simd_shuffle_xor(local_sum, ushort(1));
      local_sum += simd_shuffle_xor(local_sum, ushort(2));
      row_sum = row_sum * factor + local_sum;
      row_max = new_max;
      if (c0 == 0) {
        prev_scale[m] = factor;
        if (factor != 1.0f) {
          atomic_store_explicit(flag, 1u, memory_order_relaxed);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (atomic_load_explicit(flag, memory_order_relaxed)) {
#pragma unroll
      for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!running_full && !running.is_valid_element(i)) {
          continue;
        }
        const auto idx = running.get_multidimensional_index(i);
        running[i] *= prev_scale[idx[1]];
      }
    }
    // The other flag is read by nobody now: clear it for the next tile.
    if (lid == 0) {
      atomic_store_explicit(rescale + ((tile + 1) & 1), 0u,
                            memory_order_relaxed);
    }
    auto vt = tensor(v_ptr, dextents<int, 2>{D, rem},
                     array<int, 2>{1, v_seq_stride});
    if constexpr (full) {
      auto vs = vt.template slice<D, N>(0, 0);
      pv.run(p0, vs, running);
    } else {
      auto vs = vt.slice(0, 0);
      pv.run(p0, vs, running);
    }
    ++tile;
  };

  device KV* pk = prefix_keys + batch_idx * strides[3] +
      kv_head_idx * strides[4];
  device KV* pv_ = prefix_values + batch_idx * strides[6] +
      kv_head_idx * strides[7];
  const device float* pks = prefix_key_scales +
      batch_idx * scale_strides[0] + kv_head_idx * scale_strides[1];
  const device float* pvs = prefix_value_scales +
      batch_idx * scale_strides[2] + kv_head_idx * scale_strides[3];
  const int prefix_end = min(kend, prefix_n);
  int k0 = kstart;
  for (; k0 + N <= prefix_end; k0 += N) {
    process(pk + long(k0) * strides[5], int(strides[5]),
            pv_ + long(k0) * strides[8], int(strides[8]), pks + k0, pvs + k0,
            k0, N, BoolTag<true>{});
  }
  if (k0 < prefix_end) {
    process(pk + long(k0) * strides[5], int(strides[5]),
            pv_ + long(k0) * strides[8], int(strides[8]), pks + k0, pvs + k0,
            k0, prefix_end - k0, BoolTag<false>{});
  }
  device KV* nk = new_keys + batch_idx * strides[9] +
      kv_head_idx * strides[10];
  device KV* nv = new_values + batch_idx * strides[12] +
      kv_head_idx * strides[13];
  const device float* nks = new_key_scales + batch_idx * scale_strides[4] +
      kv_head_idx * scale_strides[5];
  const device float* nvs = new_value_scales + batch_idx * scale_strides[6] +
      kv_head_idx * scale_strides[7];
  // At most 8 new rows: one partial tile.
  k0 = max(kstart, prefix_n);
  if (k0 < kend) {
    process(nk + long(k0 - prefix_n) * strides[11], int(strides[11]),
            nv + long(k0 - prefix_n) * strides[14], int(strides[14]),
            nks + (k0 - prefix_n), nvs + (k0 - prefix_n), k0, kend - k0,
            BoolTag<false>{});
  }

  // Partials [B, Hq, rows, partitions, D]: the head-major rows of this KV
  // head are contiguous at stride partitions * D. index[0] is the column,
  // index[1] the row.
  const long o_base = long(batch_idx * num_q_heads + gqa * kv_head_idx) * rows;
  device T* partial = out + (o_base * partitions + block_idx) * D;
  const long row_stride = long(partitions) * D;
#pragma unroll
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (!running_full && !running.is_valid_element(i)) {
      continue;
    }
    const auto idx = running.get_multidimensional_index(i);
    partial[long(idx[1]) * row_stride + idx[0]] = T(running[i]);
  }
  if (softmax_lane && c0 == 0) {
    const long o_offset = o_base + m;
    sums[o_offset * partitions + block_idx] = row_sum;
    // Back to natural units for sdpa_vector_2pass_2's exp().
    maxs[o_offset * partitions + block_idx] =
        row_max == kFiniteMin ? kFiniteMin : row_max * M_LN2_F;
  }
}

// Every (M, N) the host can request (mlx_segmented_sdpa.cpp kernel_name);
// bridge_metallib_names checks both directions. Each shape has a BF16 and an
// INT8 K/V form.
#define instantiate_sdpa_nax(m, n)                                           \
  template [[host_name("mlx_node_sdpa_segmented_verify_nax_2pass_1_bf16_256" \
                       "_m" #m "_n" #n)]] [[kernel]]                         \
  decltype(segmented_sdpa_verify_nax_2pass_1<bfloat, 256, m, n>)             \
      segmented_sdpa_verify_nax_2pass_1<bfloat, 256, m, n>;                  \
  template [[host_name("mlx_node_sdpa_segmented_verify_nax_2pass_1_int8_256" \
                       "_m" #m "_n" #n)]] [[kernel]]                         \
  decltype(segmented_sdpa_verify_nax_2pass_1<bfloat, 256, m, n, int8_t>)     \
      segmented_sdpa_verify_nax_2pass_1<bfloat, 256, m, n, int8_t>;

instantiate_sdpa_nax(8, 32)
instantiate_sdpa_nax(8, 64)
instantiate_sdpa_nax(16, 32)
instantiate_sdpa_nax(16, 64)
instantiate_sdpa_nax(24, 32)
instantiate_sdpa_nax(24, 64)
instantiate_sdpa_nax(32, 32)
instantiate_sdpa_nax(32, 64)
instantiate_sdpa_nax(40, 32)
instantiate_sdpa_nax(40, 64)
instantiate_sdpa_nax(48, 32)
instantiate_sdpa_nax(48, 64)
instantiate_sdpa_nax(56, 32)
instantiate_sdpa_nax(56, 64)
instantiate_sdpa_nax(64, 32)
instantiate_sdpa_nax(64, 64)
