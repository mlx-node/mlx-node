// Mixed target projections in one dispatch: Q4_K/Q5_K/Q6_K
// (mixed_projection_m8) and, with IQ4_XS / IQ4_NL tiles and Q8_0 affine
// rows, mixed_projection_m8_joined.
// The tile body below is the existing Splash-derived kquant_qmm_m8_nax
// arithmetic, preserving its per-segment split order and BF16 boundary.
// The Q8_0 rows are affine_qmv_wide_mixed_q8g32's per-thread loop
// (affine_mixed/affine_qmv_wide_mixed.metal, MLX's qmv_wide), statement for
// statement: any change of sum order breaks bit identity with that call.
// See THIRD_PARTY_NOTICES for Apache-2.0 provenance.
#pragma clang fp reassociate(off)
#include "mlx/backend/metal/kernels/utils.h"
#include "../kquant/kquant_m8_nax.h"
#include "abi.h"

// One qmv_wide threadgroup of an 8-bit group-32 affine segment: vecs rows
// of x, 8 output columns. `local` = tile.y * x_tiles + tile.x, the grid of
// mlx_affine_mixed_qmm.cpp's qmv_wide.
template <int vecs_per_tg>
METAL_FUNC void mixed_projection_a8g32(
    const device uint32_t* w, const device float* scales, const device float* biases,
    const device bfloat* x, device bfloat* y,
    int in_vec_size, int out_vec_size, int M, uint local, uint x_tiles,
    uint simd_gid, uint simd_lid) {
  constexpr int group_size = 32;
  constexpr int bits = 8;
  constexpr int k_lanes = 8;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 32 / k_lanes;
  constexpr int sub = 8;
  const uint2 tid(local % x_tiles, local / x_tiles);

  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;

  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +
      results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;

  const int row = min(out_row, out_vec_size - 1);

  const int in_vec_size_w = in_vec_size * bits / 8;
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
  const device float* srow = scales + row * in_vec_size_g;
  const device float* brow = biases + row * in_vec_size_g;

  const device bfloat* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) {
    xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
  }

  float result[vecs_per_tg] = {0};

  for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {
    float scale = srow[g];
    float bias = brow[g];
#pragma unroll
    for (int sc = 0; sc < group_size / sub; sc++) {
      const int k0 = g * group_size + sc * sub;
      const device uint8_t* wc = wrow + k0 * bits / 8;
      float w_dq[sub];
      for (int i = 0; i < sub; i++) {
        w_dq[i] = scale * wc[i] + bias;
      }
#pragma unroll
      for (int v = 0; v < vecs_per_tg; v++) {
        const device bfloat* xc = xv[v] + k0;
        float acc = 0;
#pragma unroll
        for (int i = 0; i < sub; i++) {
          acc += static_cast<float>(xc[i]) * w_dq[i];
        }
        result[v] += acc;
      }
    }
  }

  for (int v = 0; v < vecs_per_tg; v++) {
    result[v] += simd_shuffle_down(result[v], 4);
    result[v] += simd_shuffle_down(result[v], 2);
    result[v] += simd_shuffle_down(result[v], 1);
  }

  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) {
        y[(vec0 + v) * out_vec_size + out_row] = static_cast<bfloat>(result[v]);
      }
    }
  }
}

template <int group_size, int bits, int super_ratio, bool has_min, int kind>
METAL_FUNC void mixed_projection_tile(
    const device uint32_t* w, const device uint8_t* scales, const device half* biases,
    device bfloat16_t* x, device bfloat16_t* y,
    device coherent(device) float* partials, device atomic_uint* counters,
    int in_vec_size, int out_vec_size, int num_splits, int num_rows, uint spread,
    uint2 group, uint simd_lane, uint simd_group,
    threadgroup half* stage, threadgroup bfloat16_t* astage, threadgroup half2* tl,
    threadgroup uint64_t* grid64, threadgroup uint32_t* grid32, threadgroup uint& arrival) {
  using T = bfloat16_t;
  constexpr int rows = 8, scale_shift = 0;
  constexpr bool tiled = true;
  using namespace kq_m8;
  constexpr ushort kRows = rows;
  static_assert(rows == 8 || rows == 16 || rows == 32, "Splash's decode tiers");
  constexpr Format F = format<group_size, bits, super_ratio, has_min, kind>();
  static_assert(F != Unsupported, "no M = 8 NAX decode for this mode");
  static_assert(
      is_affine<F>() ||
          sb_scale_bytes<F>() ==
              (F == IQ4NL ? 8 : super_ratio * kq_scale_bytes_per_group<has_min, kind>()),
      "the super-block companion stride must match kquant_mode.h");
  static_assert(is_same_v<T, bfloat>, "the tensor-op A operand is bfloat16");
  constexpr bool codebook = F == IQ4XS || F == IQ4NL;
  typedef typename Codes<F>::W W;


  const uint tid = simd_group * 32 + simd_lane;
  if constexpr (codebook) {
    for (uint i = tid; i < kTableEntries; i += kThreads) {
      tl[i] = half2(float(kIQ4[i & 15]), float(kIQ4[i >> 4]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if constexpr (is_grid<F>()) {
    kq_grid_stage_table<grid_kind<F>()>(grid64, grid32, tid, kThreads);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const KQGridTgTables grid{grid64, grid32};

  const uint K = in_vec_size;
  const uint N = out_vec_size;
  const uint M = num_rows;
  const uint splits = num_splits;
  const uint per = (K / kStep) / splits;
  const uint step_begin = group.y * per;
  const uint step_end = step_begin + per;
  const uint step_first = staged_first_step(group.x, step_begin, step_end, spread);
  const uint column0 = group.x * kTileCols + simd_group * kCols;
  const uint n = column0 + simd_lane;
  threadgroup half* my = stage + simd_group * 2 * kStage;

  constexpr auto desc = matmul2d_descriptor(
      kRows, kCols, kStep, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  tensor<threadgroup T, dextents<int, 2>, tensor_inline> at0(
      astage, dextents<int, 2>{kStep, kRows}, array<int, 2>{1, kStep});
  tensor<threadgroup T, dextents<int, 2>, tensor_inline> at1(
      astage + kRows * kStep, dextents<int, 2>{kStep, kRows},
      array<int, 2>{1, kStep});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
      my, dextents<int, 2>{kStep, kCols}, array<int, 2>{1, kStep});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
      my + kStage, dextents<int, 2>{kStep, kCols}, array<int, 2>{1, kStep});
  auto a0 = at0.template slice<kStep, kRows>(0, 0);
  auto a1 = at1.template slice<kStep, kRows>(0, 0);
  auto b0 = bt0.template slice<kStep, kCols>(0, 0);
  auto b1 = bt1.template slice<kStep, kCols>(0, 0);
  // Zeroed here, not in a helper: a returned initialized cooperative tensor
  // loses its values (Splash gguf_staged_tile.h).
  auto acc = op.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }

  // The codes and coefficients of a step are loaded during the step before.
  // (Deeper prefetch was measured: 4 steps in flight cost 5-10% on the tiled
  // Qwen3.8 shapes, from the extra registers.)
  const device uint32_t* row = unit_base<bits, tiled>(w, n, K);
  constexpr uint stride = unit_stride<bits, tiled>();
  CoefCursor<F, tiled, scale_shift, group_size> coefs(scales, biases, n, K);
  W cur = Codes<F>::load(row, step_first, stride);
  Coef cc = coefs.at(step_first);
  if constexpr (is_grid<F>()) {
    cur.sc = coefs.companion(step_first);
  }
  ARows<rows, T> xa = ARows<rows, T>::load(x, K, M, step_first * kStep, tid);
  uint step = step_first;
  for (uint walked = 0; walked < per; ++walked) {
    // The two buffers alternate in traversal order, including at the wrap.
    // With spreading disabled this is the original monotonic step parity.
    const uint buffer = (spread ? walked : step) & 1;
    threadgroup half* buf = my + buffer * kStage;
    stage32<F, scale_shift>(cur, cc, tl, buf + simd_lane * kStep, grid);
    xa.store(astage + buffer * kRows * kStep, tid);
    // The A stage is the threadgroup's, so the step barrier is too.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint next = step + 1 < step_end ? step + 1 : step_begin;
    if (walked + 1 < per) {
      cur = Codes<F>::load(row, next, stride);
      cc = coefs.at(next);
      if constexpr (is_grid<F>()) {
        cur.sc = coefs.companion(next);
      }
      xa = ARows<rows, T>::load(x, K, M, next * kStep, tid);
    }
    if (buffer) {
      op.run(a1, b1, acc);
    } else {
      op.run(a0, b0, acc);
    }
    step = next;
  }

  // index[0] is the column, index[1] the row.
  if (splits == 1) {
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) {
        continue;
      }
      const auto idx = acc.get_multidimensional_index(i);
      if (uint(idx[1]) < M) {
        y[uint(idx[1]) * N + column0 + uint(idx[0])] = T(float(acc[i]));
      }
    }
    return;
  }
  const uint split = group.y;
  const auto at = [&](uint s, uint row_, uint col) {
    return (size_t(s) * kRows + row_) * N + column0 + col;
  };
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) {
      continue;
    }
    const auto idx = acc.get_multidimensional_index(i);
    partials[at(split, uint(idx[1]), uint(idx[0]))] = float(acc[i]);
  }
  if (!split_arrive_last(counters + group.x, splits, tid, &arrival)) {
    return;
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) {
      continue;
    }
    const auto idx = acc.get_multidimensional_index(i);
    const uint row_ = uint(idx[1]);
    const uint col = uint(idx[0]);
    if (row_ >= M) {
      continue;
    }
    float total = 0.0f;
    for (uint s = 0; s < splits; ++s) {
      total += s == split ? float(acc[i]) : partials[at(s, row_, col)];
    }
    y[row_ * N + column0 + col] = T(total);
  }
  split_release(counters + group.x, tid);
}

template <bool joined>
[[kernel, max_total_threads_per_threadgroup(64)]] void mixed_projection_m8(
    device bfloat16_t* x [[buffer(0)]],
    const device uint32_t* w0 [[buffer(1)]], const device uint8_t* s0 [[buffer(2)]], const device half* b0 [[buffer(3)]],
    const device uint32_t* w1 [[buffer(4)]], const device uint8_t* s1 [[buffer(5)]], const device half* b1 [[buffer(6)]],
    const device uint32_t* w2 [[buffer(7)]], const device uint8_t* s2 [[buffer(8)]], const device half* b2 [[buffer(9)]],
    device bfloat16_t* y0 [[buffer(10)]], device bfloat16_t* y1 [[buffer(11)]], device bfloat16_t* y2 [[buffer(12)]],
    device coherent(device) float* partials [[buffer(13)]], device atomic_uint* counters [[buffer(14)]],
    constant MixedProjectionParams& p [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  using namespace kq_m8;
  threadgroup half stage[2 * 2 * kStage];
  threadgroup bfloat16_t astage[2 * 8 * kStep];
  threadgroup half2 tl[joined ? kTableEntries : 1];
  threadgroup uint64_t grid64[1];
  threadgroup uint32_t grid32[1];
  threadgroup uint arrival;
  const uint segment = group < p.group_end[0] ? 0 : group < p.group_end[1] ? 1 : 2;
  const uint local = group - (segment ? p.group_end[segment - 1] : 0);
  if constexpr (joined) {
    if (p.format[segment] == kMixedA8G32) {
      const device uint32_t* w = segment == 0 ? w0 : segment == 1 ? w1 : w2;
      const device uint8_t* s = segment == 0 ? s0 : segment == 1 ? s1 : s2;
      const device half* b = segment == 0 ? b0 : segment == 1 ? b1 : b2;
      device bfloat16_t* y = segment == 0 ? y0 : segment == 1 ? y1 : y2;
      const uint vecs = p.vectors[segment];
      const uint x_tiles = (8 + vecs - 1) / vecs;
#define ROWS(V) mixed_projection_a8g32<V>(w, (const device float*)s, (const device float*)b, (const device bfloat*)x, (device bfloat*)y, p.input_size, p.columns[segment], 8, local, x_tiles, sg, lane)
      if (vecs == 8) ROWS(8);
      else ROWS(4);
#undef ROWS
      return;
    }
  }
  const uint tiles = p.columns[segment] / 64;
  const uint2 tile(local % tiles, local / tiles);
  const device uint32_t* w = segment == 0 ? w0 : segment == 1 ? w1 : w2;
  const device uint8_t* s = segment == 0 ? s0 : segment == 1 ? s1 : s2;
  const device half* b = segment == 0 ? b0 : segment == 1 ? b1 : b2;
  device bfloat16_t* y = segment == 0 ? y0 : segment == 1 ? y1 : y2;
  // Do not offset dummy bindings when every segment has a single split.
  auto part = partials + (p.splits[segment] > 1 ? p.partial_offset[segment] : 0);
  auto counter = counters + (p.splits[segment] > 1 ? p.counter_offset[segment] : 0);
#define RUN(G, B, S, MIN, KIND) mixed_projection_tile<G, B, S, MIN, KIND>(w, s, b, x, y, part, counter, p.input_size, p.columns[segment], p.splits[segment], 8, p.spread, tile, lane, sg, stage, astage, tl, grid64, grid32, arrival)
  if (p.format[segment] == kMixedQ4K) RUN(32, 4, 8, true, KQ_LINEAR);
  else if (p.format[segment] == kMixedQ5K) RUN(32, 5, 8, true, KQ_LINEAR);
  else if constexpr (!joined) RUN(16, 6, 16, false, KQ_LINEAR);
  else if (p.format[segment] == kMixedQ6K) RUN(16, 6, 16, false, KQ_LINEAR);
  else if (p.format[segment] == kMixedIQ4XS) RUN(32, 4, 8, false, KQ_CODEBOOK);
  else RUN(32, 4, 1, false, KQ_CODEBOOK);
#undef RUN
}

// Host names: mlx_mixed_projection.cpp's kernel_names().
template [[host_name("mixed_projection_m8")]] [[kernel]]
decltype(mixed_projection_m8<false>) mixed_projection_m8<false>;
template [[host_name("mixed_projection_m8_joined")]] [[kernel]]
decltype(mixed_projection_m8<true>) mixed_projection_m8<true>;
