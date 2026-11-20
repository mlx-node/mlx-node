// Adapted from Splash 1.6.1 (https://github.com/incoai/splash) d8af1054aab0f97de21d043e01aa98af3754f10e,
// runtime/metal/kernels/shared/gguf_linear.metal. Apache-2.0; see THIRD_PARTY_NOTICES. Changed: the staged decode
// kernels only (no prefill, 64-row, a_f32, leading-input or int8 kernels), the eight formats of
// metal/abi/QuantFormat.h; the gate/up pair stores mlx-node's swiglu (kernels/common/gguf_tile.h); the residual,
// gate and pair kernels are instantiated only when GGUF_PLANES_STORE_EPILOGUES is set (built -fno-fast-math), the
// plain and fused kernels only when it is not. With GGUF_PLANES_TEST_KERNELS set, only the test-only kernels (for
// gguf_planes_test.metallib, never shipped): gguf_dequant in both builds (gguf_dequant_epilogues in the epilogue
// build), and Q4_K 8-row residual and gate|up kernels with Splash's fp32 epilogues.
// GGUF decoding follows ggml's Metal dequantize.h / mul_mm.metal (MIT, The ggml authors).
//
// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile, matching llama.cpp Metal
// dequantize.h / mul_mm.metal. Keep activations BF16. Weight planes and meta in the MDGG0001 layout
// (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h. Activations bf16 [rows][K]; weights
// staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output.
// Keep the source order of float operations, which Metal's default fast math lets the compiler reassociate. Set
// before the includes, so it also holds for the shared format and reduction code compiled here.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/split_reduce.h"

// A tile's sums over every K partition, handed to store(row, column, sum). One partition stores its own; more publish
// fp32 partials [split][Rows][destination column] (kernels/common/split_reduce.h) and the last arriving partition adds
// them in split order. `column0` is the simdgroup's first destination column, `counter` its threadgroup's (one per 64
// destination columns: the segments of a projection never share one).
template <ushort Rows, class Acc, class Store>
inline void gguf_store_sums(thread Acc &acc, uint splits, uint split, device coherent(device) float *partials,
                            device atomic_uint *counter, uint stride, uint column0, uint thread_index,
                            threadgroup uint *arrival, Store store) {
  if (splits == 1) { gguf_elements(acc, store); return; }
  const auto at = [&](uint s, uint row, uint column) { return (ulong(s) * Rows + row) * stride + column0 + column; };
  gguf_elements(acc, [&](uint row, uint column, float v) { partials[at(split, row, column)] = v; });
  if (!split_arrive_last(counter, splits, thread_index, arrival)) return;
  gguf_elements(acc, [&](uint row, uint column, float v) {
    store(row, column, split_sum(v, split, splits, [&](uint s) { return partials[at(s, row, column)]; }));
  });
  split_release(counter, thread_index);
}

#if GGUF_PLANES_STORE_EPILOGUES
// A gate/up pair's sums over every K partition, stored as the swiglu of bf16(gate) and bf16(up) (gguf_swiglu). One
// partition stores its own; more publish fp32 partials [split][Rows][2 * stride], the gate's columns then the up's at
// stride on, and the last arriving partition adds each in split order. `counter` is the simdgroup's (one per 32
// destination columns).
template <ushort Rows, bool Fp32, class Acc>
inline void gguf_store_gate_up(thread Acc &gate, thread Acc &up, uint splits, uint split,
                               device coherent(device) float *partials, device atomic_uint *counter, uint stride,
                               uint column0, uint lane, threadgroup uint *arrival, device bfloat *output) {
  const auto value = [](float g, float u) -> bfloat {
    if constexpr (Fp32) return gguf_swiglu_fp32(g, u);
    else return gguf_swiglu(bfloat(g), bfloat(u));
  };
  const auto out = [&](uint row, uint column) { return ulong(row) * stride + column0 + column; };
  if (splits == 1) {
#pragma unroll
    for (ushort i = 0; i < gate.get_capacity(); ++i) {
      if (!gate.is_valid_element(i)) continue;
      const auto index = gate.get_multidimensional_index(i);
      output[out(index[1], index[0])] = value(gate[i], up[i]);
    }
    return;
  }
  const auto at = [&](uint s, uint row, uint column) {
    return (ulong(s) * Rows + row) * (2 * stride) + column0 + column;
  };
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    partials[at(split, index[1], index[0])] = gate[i];
    partials[at(split, index[1], index[0]) + stride] = up[i];
  }
  if (!split_arrive_last(counter, splits, lane, arrival)) return;
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    const uint row = index[1], column = index[0];
    const float g = split_sum(float(gate[i]), split, splits, [&](uint s) { return partials[at(s, row, column)]; });
    const float u = split_sum(float(up[i]), split, splits,
                              [&](uint s) { return partials[at(s, row, column) + stride]; });
    output[out(row, column)] = value(g, u);
  }
  split_release(counter, lane);
}
#endif

// ---------------- decode dispatches over (64-column tiles, K partitions): two simdgroups of 32 columns per
// threadgroup, every request lane in its tile, `splits` partitions of K (grid.y; kernels/common/split_reduce.h).
// MPP computes 16-row fragments, so a tile holds 8, 16 or 32 rows. Rows are independent, so every active row is the
// bits of any other tile height. Each column tile walks its partition from the step staged_first_step gives it
// (kernels/common/gguf_staged_tile.h): in lockstep, or spread when p.spread is set.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device Out *output,
                             device coherent(device) float *partials, device atomic_uint *counters, device bfloat *aux,
                             constant GgufDecodeParams &p, uint2 group, uint simd_lane, uint simd_group,
                             threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits, sb = group.y * per, se = sb + per,
             origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  auto acc = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);
  gguf_zero(acc);
  staged_accumulate<F, Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, sb, se,
                                          staged_first_step(group.x, sb, se, p.spread), acc);
  gguf_store_sums<Rows>(acc, p.splits, group.y, partials, counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, p.out_stride, column0,
                        simd_group * 32 + simd_lane, arrival, [&](uint row, uint column, float v) {
    const ulong o = ulong(row) * p.out_stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
// The staged decode kernels: grid (column tiles, K partitions), two simdgroups.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                        device uchar *meta [[buffer(3)]], device Out *output [[buffer(4)]],
                        device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]],
                        device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                        uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
  threadgroup half stage[kStagedStages];
  threadgroup uint arrival;
  gguf_decode_tile<F, Rows, Ep>(input, w0, w1, meta, output, partials, counters, aux, p, group, simd_lane, simd_group,
                                stage, tl, &arrival);
}
template <class Out>
using GgufDecodeKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device Out *,
                              device coherent(device) float *, device atomic_uint *, device bfloat *,
                              constant GgufDecodeParams &, uint2, uint, uint);
#define GGUF_DECODE(F, f, R, ep, Ep, Out) \
  template [[host_name("gguf_decode_" #f "_m" #R "_" #ep)]] kernel GgufDecodeKernel<Out> gguf_decode<F, R, Ep, Out>;
#define GGUF_DECODE_ROWS(F, f, ep, Ep, Out) \
  GGUF_DECODE(F, f, 8, ep, Ep, Out) GGUF_DECODE(F, f, 16, ep, Ep, Out) GGUF_DECODE(F, f, 32, ep, Ep, Out)
#if GGUF_PLANES_TEST_KERNELS
#if GGUF_PLANES_STORE_EPILOGUES
GGUF_DECODE(FmtQ4K, q4k, 8, r_fp32, EpResidualFp32, bfloat)
#endif
#else
#if GGUF_PLANES_STORE_EPILOGUES
#define GGUF_DECODE_FORMAT(F, f) GGUF_DECODE_ROWS(F, f, r, EpResidual, bfloat) GGUF_DECODE_ROWS(F, f, g, EpUpWithGate, bfloat)
#else
#define GGUF_DECODE_FORMAT(F, f) GGUF_DECODE_ROWS(F, f, a, EpNone, bfloat)
#endif
QUANT_FORMATS(GGUF_DECODE_FORMAT)
#undef GGUF_DECODE_FORMAT
#endif
#undef GGUF_DECODE_ROWS
#undef GGUF_DECODE

#if GGUF_PLANES_STORE_EPILOGUES
// Gate/up in one dispatch, where gate and up share their format: grid (32-column tiles, K partitions), threadgroups
// of one simdgroup that stages the gate's and the up's same 32 columns (gguf_staged_pair_steps), so the per-step
// latency of a tile's chain serves both tensors. Each walks from the step its 64-column tile walks from in the two
// passes, so at the same K split gate and up are the bits of the gate pass's and of the up pass's sums.
template <class F, ushort Rows, bool Fp32>
kernel void gguf_decode_gate_up(device bfloat *input [[buffer(0)]], device uchar *gw0 [[buffer(1)]],
                                device uchar *gw1 [[buffer(2)]], device uchar *gmeta [[buffer(3)]],
                                device uchar *uw0 [[buffer(4)]], device uchar *uw1 [[buffer(5)]],
                                device uchar *umeta [[buffer(6)]], device bfloat *output [[buffer(7)]],
                                device coherent(device) float *partials [[buffer(8)]],
                                device atomic_uint *counters [[buffer(9)]], constant GgufDecodeParams &p [[buffer(10)]],
                                uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, lane, GGUF_GATE_UP_THREADS);
  threadgroup half stage[2 * GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP];
  threadgroup uint arrival;
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits, sb = group.y * per, se = sb + per,
             origin = group.x * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  auto gate = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, stage);
  auto up = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, stage);
  gguf_zero(gate);
  gguf_zero(up);
  gguf_staged_pair_steps<F, Rows>(input, gw0, gw1, gmeta, uw0, uw1, umeta, p.input_size, origin, stage, tl, lane, sb,
                                  se, staged_first_step(group.x / 2, sb, se, p.spread), gate, up);
  gguf_store_gate_up<Rows, Fp32>(gate, up, p.splits, group.y, partials,
                           counters + p.out_offset / GGUF_STAGED_COLUMNS + group.x, p.out_stride, column0, lane,
                           &arrival, output);
}
using GgufDecodeGateUpKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device uchar *,
                                    device uchar *, device uchar *, device bfloat *, device coherent(device) float *,
                                    device atomic_uint *, constant GgufDecodeParams &, uint2, uint);
#if GGUF_PLANES_TEST_KERNELS
template [[host_name("gguf_decode_q4k_m8_gate_up_fp32")]] kernel GgufDecodeGateUpKernel gguf_decode_gate_up<FmtQ4K, 8, true>;
#define GGUF_DEQUANT_KERNEL gguf_dequant_epilogues
#else
#define GGUF_DECODE_GATE_UP(F, f, R) \
  template [[host_name("gguf_decode_" #f "_m" #R "_gate_up")]] kernel GgufDecodeGateUpKernel gguf_decode_gate_up<F, R, false>;
#define GGUF_DECODE_GATE_UP_FORMAT(F, f) GGUF_DECODE_GATE_UP(F, f, 8) GGUF_DECODE_GATE_UP(F, f, 16) GGUF_DECODE_GATE_UP(F, f, 32)
QUANT_FORMATS(GGUF_DECODE_GATE_UP_FORMAT)
#undef GGUF_DECODE_GATE_UP_FORMAT
#undef GGUF_DECODE_GATE_UP
#endif
#elif GGUF_PLANES_TEST_KERNELS
#define GGUF_DEQUANT_KERNEL gguf_dequant
#else
// Fused projections (qkv|z|ab, q|k|v): up to three column segments of any formats in one dispatch, so the small
// segments do not run as dispatches of their own. The threadgroup's tile picks its segment, and the segment's format
// picks the decode; every segment takes the same K splits, and its tiles walk as its projection alone would, from the
// step staged_first_step gives the tile's index within the segment. The fused kernels stay plain kernels: as
// template instantiations (weak_odr) the compiler infers fewer parameter attributes and inlines differently.
#define GGUF_SEGMENT(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_SEGMENTS GGUF_SEGMENT(1, w0a, w1a, ma), GGUF_SEGMENT(4, w0b, w1b, mb), GGUF_SEGMENT(7, w0c, w1c, mc)
// The segment s of column tile `tile` among the segments' tiles (cols, 0 past the last) and the tile's index `local`
// within it; GGUF_SEGMENT_PLANE the segment's `plane` of x##a, x##b and x##c; GGUF_SEGMENT_OF both, with the
// segment's planes w0, w1 and meta.
#define GGUF_SEGMENT_INDEX(tile, cols)                                                                             \
  const uint t0 = (cols)[0] / GGUF_TILE_COLUMNS, t1 = t0 + (cols)[1] / GGUF_TILE_COLUMNS;                          \
  const uint s = (tile) < t0 ? 0 : (tile) < t1 ? 1 : 2, local = (tile) - (s == 0 ? 0 : s == 1 ? t0 : t1)
#define GGUF_SEGMENT_PLANE(plane, x) device uchar *plane = s == 0 ? x##a : s == 1 ? x##b : x##c
#define GGUF_SEGMENT_OF(tile, cols)                                                                                \
  GGUF_SEGMENT_INDEX(tile, cols);                                                                                  \
  GGUF_SEGMENT_PLANE(w0, w0);                                                                                      \
  GGUF_SEGMENT_PLANE(w1, w1);                                                                                      \
  GGUF_SEGMENT_PLANE(meta, m)
// A fused pipeline decodes its dispatch's segment formats alone, function constants 0-2 (each once, the last repeated
// past them): the GPU compiler builds a staged loop for each format a pipeline decodes. fused_formats runs
// body(F()) with the format type of `format`, one body per distinct format; an id the pipeline does not decode
// decodes as its first, so every id runs one body.
constant uint kFusedFormat0 [[function_constant(0)]];
constant uint kFusedFormat1 [[function_constant(1)]];
constant uint kFusedFormat2 [[function_constant(2)]];
struct fused_formats {
  template <class Body> void operator()(uint format, Body body) const {
    if (kFusedFormat2 != kFusedFormat1 && format == kFusedFormat2) quant_format_switch(kFusedFormat2, body);
    else if (kFusedFormat1 != kFusedFormat0 && format == kFusedFormat1) quant_format_switch(kFusedFormat1, body);
    else quant_format_switch(kFusedFormat0, body);
  }
};
#define GGUF_DECODE_FUSED(R)                                                                                       \
  kernel void gguf_decode_fused_m##R(device bfloat *input [[buffer(0)]], GGUF_SEGMENTS,                          \
                                     device bfloat *output [[buffer(10)]],                                        \
                                     device coherent(device) float *partials [[buffer(11)]],                      \
                                     device atomic_uint *counters [[buffer(12)]],                                 \
                                     constant GgufDecodeFusedParams &p [[buffer(13)]],                            \
                                     uint2 group [[threadgroup_position_in_grid]],                                \
                                     uint simd_lane [[thread_index_in_simdgroup]],                                \
                                     uint simd_group [[simdgroup_index_in_threadgroup]]) {                        \
    threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries]; threadgroup uint arrival; \
    GGUF_SEGMENT_OF(group.x, p.cols);                                                                             \
    const uint per = p.input_size / GGUF_STAGED_STEP / p.splits;                                                  \
    const uint sb = group.y * per, se = sb + per;                                                                 \
    const uint origin = local * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.offset[s] + origin; \
    threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;                                            \
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);                                         \
    gguf_zero(acc);                                                                                               \
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(p.fmt[s], input, w0, w1, meta, p.input_size, origin, my, tl, \
                                            simd_group * 32 + simd_lane, simd_lane, sb, se,                       \
                                            staged_first_step(local, sb, se, p.spread), acc, fused_formats());    \
    gguf_store_sums<R>(acc, p.splits, group.y, partials, counters + p.offset[s] / GGUF_TILE_COLUMNS + local,       \
                       p.out_stride, column0, simd_group * 32 + simd_lane, &arrival,                              \
                       [&](uint row, uint column, float v) { output[ulong(row) * p.out_stride + column0 + column] = bfloat(v); }); \
  }
GGUF_DECODE_FUSED(8) GGUF_DECODE_FUSED(16) GGUF_DECODE_FUSED(32)
#undef GGUF_DECODE_FUSED
#endif

#if GGUF_PLANES_TEST_KERNELS
// Test-only: the planes of a [rows, K] tensor decoded to half [rows][K] by the staged tiles' own decode
// (dequant32_chunks), one thread per (row, group of 32), threadgroups of 256.
kernel void GGUF_DEQUANT_KERNEL(device uchar *w0 [[buffer(0)]], device uchar *w1 [[buffer(1)]], device uchar *meta [[buffer(2)]],
                         device half *output [[buffer(3)]], constant GgufDequantParams &p [[buffer(4)]],
                         uint t [[thread_position_in_grid]], uint index [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]]) {
  threadgroup half2 tl[kQuantPairTableEntries];
  const uint G = p.input_size / 32;
  quant_format_switch(p.fmt, [&](auto format) {
    typedef decltype(format) F;
    quant_pair_table<F>(tl, index, threads);
    if (t >= p.rows * G) return;
    const uint n = t / G, g = t % G;
    device uchar *p0 = w0 + quant_tile_index(n, g, G) * F::P0, *p1 = w1 + quant_tile_index(n, g, G) * F::P1;
    device uchar *m = meta + quant_tile_index(n, g / F::MetaGroups, G / F::MetaGroups) * F::MetaBytes;
    device half *row = output + ulong(n) * p.input_size + 32 * g;
    dequant32_chunks<F>(F::load(p0, p1), F::loadMeta(m), g % F::MetaGroups, tl, [&](ushort c, half4 lo, half4 hi) {
      for (ushort i = 0; i < 4; ++i) { row[4 * c + i] = lo[i]; row[16 + 4 * c + i] = hi[i]; }
    });
  });
}
#undef GGUF_DEQUANT_KERNEL
#endif
