// Adapted from Splash draft.metal (Apache-2.0). Four split tensor-op draft
// attention with FP32 online softmax. Inputs are live MLX KV views, not a
// physical ring: all partial tiles use dynamic tensor extents so no kernel
// reads rows outside a view. The supplied mask preserves absolute-position
// sliding visibility across compiled replays. See THIRD_PARTY_NOTICES.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include "abi.h"
using namespace metal;
using namespace mpp::tensor_ops;

inline void draft_attention_split_phase(
    device bfloat *queries, device bfloat *keys, device bfloat *values,
    device bfloat *query_keys, device bfloat *query_values,
    device float *partial, device const bool *mask, constant DraftAttentionParams &p, uint split,
    threadgroup float *score_storage, threadgroup float *row_max,
    threadgroup float *row_sum, threadgroup float *previous_scale,
    uint thread_index, uint lane, uint simd_group) {
  constexpr ushort M = 32, N = 128,
                   D = 128, TileK = 64;
  constexpr uint Rows = 8,
                 Splits = 4;
  uint old_count = p.key_rows - Rows;
  uint live_tiles = (old_count + N - 1) / N;
  // A split with no tile that is not the last one has nothing to add: it
  // leaves only row maxima of -inf, which the reduce skips, instead of
  // loading Q and storing a zero accumulator at short contexts.
  if (split >= live_tiles && split + 1 != Splits) {
    if (thread_index < M)
      partial[M * D + thread_index] = -INFINITY;
    return;
  }
  auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, int(p.query_row_stride)});
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, TileK, false, true, false);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  auto q0 = qt.slice<TileK, M>(0, 0);
  auto pt = tensor(score_storage, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<TileK, M>(0, 0);
  auto first_v = tensor(values, dextents<int, 2>{D, N},
                        array<int, 2>{1, int(p.value_row_stride)});
  auto first_v0 = first_v.slice<D, TileK>(0, 0);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, TileK, false, false, false);
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(first_v0), float>();
  for (ushort i = 0; i < running.get_capacity(); ++i)
    running[i] = 0.0f;
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint tile = split; tile < live_tiles; tile += Splits) {
    uint slot = tile * N;
    uint remaining = min(uint(N), old_count - slot);
    auto kt =
        tensor(keys + slot * p.key_row_stride, dextents<int, 2>{D, int(remaining)},
               array<int, 2>{1, int(p.key_row_stride)});
    auto k0 = kt.slice(0, 0);
    auto scores =
        qk.template get_destination_cooperative_tensor<decltype(q0),
                                                       decltype(k0), float>();
    for (ushort i = 0; i < scores.get_capacity(); ++i)
      scores[i] = 0.0f;
    for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
      auto qs = qt.slice<TileK, M>(chunk * TileK, 0);
      auto bounded_k = tensor(keys + slot * p.key_row_stride + chunk * TileK,
          dextents<int, 2>{TileK, int(remaining)}, array<int, 2>{1, int(p.key_row_stride)});
      auto ks = bounded_k.slice(0, 0);
      auto partial_scores = qk.template get_destination_cooperative_tensor<
          decltype(qs), decltype(ks), float>();
      qk.run(qs, ks, partial_scores);
      for (ushort i = 0; i < scores.get_capacity(); ++i) {
        scores[i] += partial_scores[i];
      }
    }
    scores.store(pt.slice<N, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort batch = 0; batch < M; batch += 8) {
      ushort matrix_row = batch + simd_group;
      uint proposal_row = matrix_row % Rows;
      float local_scores[4];
      float tile_max = -INFINITY;
      for (ushort i = 0; i < 4; ++i) {
        uint column = lane + i * 32;
        uint key = slot + column;
        bool valid = column < remaining &&
            (!p.has_mask || mask[proposal_row * p.key_rows + key]);
        float score = score_storage[matrix_row * N + column] * p.scale;
        local_scores[i] = valid ? score : -INFINITY;
        tile_max = max(tile_max, local_scores[i]);
      }
      tile_max = simd_max(tile_max);
      float next_max = max(row_max[matrix_row], tile_max);
      // A row with no visible key so far keeps its running state instead of
      // scaling it by exp(-inf - -inf).
      bool empty = next_max == -INFINITY;
      float scale = empty ? 1.0f : fast::exp(row_max[matrix_row] - next_max);
      float tile_sum = 0.0f;
      for (ushort i = 0; i < 4; ++i) {
        local_scores[i] = empty ? 0.0f : fast::exp(local_scores[i] - next_max);
        tile_sum += local_scores[i];
      }
      tile_sum = simd_sum(tile_sum);
      if (lane == 0) {
        previous_scale[matrix_row] = scale;
        row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
        row_max[matrix_row] = next_max;
      }
      for (ushort i = 0; i < 4; ++i) {
        uint column = lane + i * 32;
        score_storage[matrix_row * N + column] = local_scores[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Native V is token-major. Keep the non-transposed MPP operand with
    // unit dimension stride; a transposed view with strided dimension zero
    // does not safely bound dynamic tails on this backend.
    auto vt = tensor(values + slot * p.value_row_stride, dextents<int, 2>{D, int(remaining)},
                     array<int, 2>{1, int(p.value_row_stride)});
    auto v0 = vt.slice(0, 0);
    auto partial_output =
        pv.template get_destination_cooperative_tensor<decltype(p0),
                                                       decltype(v0), float>();
    for (ushort i = 0; i < partial_output.get_capacity(); ++i) {
      partial_output[i] = 0.0f;
    }
    for (ushort chunk = 0; chunk * TileK < remaining; ++chunk) {
      auto ps = pt.slice<TileK, M>(chunk * TileK, 0);
      auto bounded_v = tensor(values + (slot + chunk * TileK) * p.value_row_stride,
          dextents<int, 2>{D, int(min(uint(TileK), remaining - chunk * TileK))},
          array<int, 2>{1, int(p.value_row_stride)});
      auto vs = bounded_v.slice(0, 0);
      auto partial_values = pv.template get_destination_cooperative_tensor<
          decltype(ps), decltype(vs), float>();
      pv.run(ps, vs, partial_values);
      for (ushort i = 0; i < partial_output.get_capacity(); ++i) {
        partial_output[i] += partial_values[i];
      }
    }
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      auto index = running.get_multidimensional_index(i);
      running[i] = running[i] * previous_scale[index[1]] + partial_output[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (split + 1 == Splits) {
    // Persistent query K/V contains exactly eight rows.  The K=16 padding
    // required by the MPP value product exists only in threadgroup memory.
    constexpr ushort CurrentN = Rows;
    constexpr ushort CurrentValueTile = 16;
    threadgroup float *current_raw_scores =
        score_storage + M * CurrentValueTile;
    threadgroup bfloat *current_values =
        reinterpret_cast<threadgroup bfloat *>(
            score_storage + M * CurrentValueTile + M * CurrentN);
    auto current_scores =
        tensor(current_raw_scores, dextents<int, 2>{CurrentN, M},
               array<int, 2>{1, CurrentN});
    constexpr auto current_qk_descriptor =
        matmul2d_descriptor(M, CurrentN, TileK, false, true, false);
    matmul2d<current_qk_descriptor, execution_simdgroups<8>> current_qk;
    auto current_kt = tensor(query_keys, dextents<int, 2>{D, CurrentN},
                             array<int, 2>{1, int(p.key_row_stride)});
    auto current_k0 = current_kt.slice<TileK, CurrentN>(0, 0);
    auto current = current_qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(current_k0), float>();
    for (ushort i = 0; i < current.get_capacity(); ++i)
      current[i] = 0.0f;
    for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
      auto qs = qt.slice<TileK, M>(chunk * TileK, 0);
      auto ks = current_kt.slice<TileK, CurrentN>(chunk * TileK, 0);
      auto partial_scores =
          current_qk.template get_destination_cooperative_tensor<
              decltype(qs), decltype(ks), float>();
      current_qk.run(qs, ks, partial_scores);
      for (ushort i = 0; i < current.get_capacity(); ++i) {
        current[i] += partial_scores[i];
      }
    }
    current.store(current_scores.slice<CurrentN, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort batch = 0; batch < M; batch += 8) {
      ushort matrix_row = batch + simd_group;
      bool valid = lane < CurrentN && (!p.has_mask ||
          mask[(matrix_row % Rows) * p.key_rows + old_count + lane]);
      float score = valid
                        ? current_raw_scores[matrix_row * CurrentN + lane] *
                              p.scale
                        : -INFINITY;
      float tile_max = simd_max(score);
      float next_max = max(row_max[matrix_row], tile_max);
      bool empty = next_max == -INFINITY;
      float scale = empty ? 1.0f : fast::exp(row_max[matrix_row] - next_max);
      float probability = empty ? 0.0f : fast::exp(score - next_max);
      float tile_sum = simd_sum(probability);
      if (lane == 0) {
        previous_scale[matrix_row] = scale;
        row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
        row_max[matrix_row] = next_max;
      }
      if (lane < CurrentValueTile) {
        score_storage[matrix_row * CurrentValueTile + lane] =
            lane < CurrentN ? probability : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto current_probabilities =
        tensor(score_storage, dextents<int, 2>{CurrentValueTile, M},
               array<int, 2>{1, CurrentValueTile});
    auto current_p0 = current_probabilities.slice<CurrentValueTile, M>(0, 0);
    for (uint index = thread_index; index < D * CurrentValueTile;
         index += 256) {
      uint dimension = index / CurrentValueTile;
      uint token = index % CurrentValueTile;
      current_values[index] = token < CurrentN
                                  ? query_values[token * p.value_row_stride + dimension]
                                  : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto current_vt =
        tensor(current_values, dextents<int, 2>{CurrentValueTile, D},
               array<int, 2>{1, int(CurrentValueTile)});
    auto current_v0 = current_vt.slice<CurrentValueTile, D>(0, 0);
    constexpr auto current_pv_descriptor =
        matmul2d_descriptor(M, D, CurrentValueTile, false, true, false);
    matmul2d<current_pv_descriptor, execution_simdgroups<8>> current_pv;
    auto current_output =
        current_pv.template get_destination_cooperative_tensor<
            decltype(current_p0), decltype(current_v0), float>();
    current_pv.run(current_p0, current_v0, current_output);
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      auto index = running.get_multidimensional_index(i);
      running[i] = running[i] * previous_scale[index[1]] + current_output[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  auto accumulator =
      tensor(partial, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(accumulator.slice<D, M>(0, 0));
  if (thread_index < M) {
    partial[M * D + thread_index] = row_max[thread_index];
    partial[M * D + M + thread_index] = row_sum[thread_index];
  }
}

// Fixed-order combine of one (lane, head) pair's split partials into the
// normalized bf16 rows, so the result does not depend on which split
// finished first or on how many lanes ran.
inline void draft_attention_reduce_phase(device const float *partials,
                                         device bfloat *output,
                                         uint thread_index, uint head) {
  constexpr uint M = 32, D = 128,
                 Stride = M * D + 2 * M,
                 Splits = 4;
  constexpr uint Chunk = D / 8;
  uint row = thread_index / 8;
  uint dim = (thread_index % 8) * Chunk;
  float maximum = -INFINITY;
  for (uint split = 0; split < Splits; ++split)
    maximum = max(maximum, partials[split * Stride + M * D + row]);
  float total = 0.0f;
  float accumulated[Chunk];
  for (uint i = 0; i < Chunk; ++i)
    accumulated[i] = 0.0f;
  for (uint split = 0; split < Splits; ++split) {
    device const float *partial = partials + split * Stride;
    float split_max = partial[M * D + row];
    // A split with no visible key contributes nothing rather than
    // exp(-inf - max) times an all-zero accumulator.
    if (split_max == -INFINITY)
      continue;
    float scale = fast::exp(split_max - maximum);
    total += scale * partial[M * D + M + row];
    for (uint i = 0; i < Chunk; ++i)
      accumulated[i] += scale * partial[row * D + dim + i];
  }
  for (uint i = 0; i < Chunk; ++i)
    // Local row is [4 query heads, 8 positions]. Store [position, 32 heads]
    // directly, preserving every reduction and BF16 rounding operation.
    output[((row % 8) * 32 + head * 4 + row / 8) * D + dim + i] =
        bfloat(total > 0.0f ? accumulated[i] / total : 0.0f);
}


[[kernel, max_total_threads_per_threadgroup(256)]]
void dflash2_attention_split(
    device bfloat *queries [[buffer(0)]], device bfloat *keys [[buffer(1)]],
    device bfloat *values [[buffer(2)]], device const bool *mask [[buffer(3)]],
    device float *partials [[buffer(4)]], constant DraftAttentionParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint M = 32, N = 128, D = 128, Splits = 4, Partial = M * D + 2 * M;
  uint head = group.x, split = group.y, old_count = p.key_rows - 8;
  threadgroup float workspace[M * N + 3 * M];
  device bfloat *k = keys + head * p.key_head_stride;
  device bfloat *v = values + head * p.value_head_stride;
  draft_attention_split_phase(
      queries + head * 4 * p.query_head_stride, k, v,
      k + old_count * p.key_row_stride, v + old_count * p.value_row_stride,
      partials + (head * Splits + split) * Partial, mask, p, split,
      workspace, workspace + M * N, workspace + M * N + M,
      workspace + M * N + 2 * M, thread_index, lane, simd_group);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void dflash2_attention_reduce(device const float *partials [[buffer(0)]],
    device bfloat *output [[buffer(1)]], uint head [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint M = 32, D = 128, Splits = 4, Partial = M * D + 2 * M;
  draft_attention_reduce_phase(partials + head * Splits * Partial,
      output, thread_index, head);
}
