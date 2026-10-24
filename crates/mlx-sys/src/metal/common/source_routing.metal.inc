R"METAL(
// Adapted from Splash 1482f6bbc2abf12375332725302019594150b6cb; Apache-2.0.
// BF16 router/shared weights widen exactly to their original GGUF F32 values.
#pragma clang fp reassociate(off)
#include <metal_stdlib>
using namespace metal;
#define GGUF_FLOAT_PROJECTIONS 2u
#define SPLASH_MOE_EXPERT_SLOTS 256u
struct GgufFloatParams { uint rows,input_size,output_size[2],out_stride[2],out_offset[2]; };
struct MoeRouteParams { uint rows,input_size,experts,top_k; };
inline float splash_sigmoid(float x) { return 1.0f / (1.0f + fast::exp2(-1.44269504089f*x)); }
#include <metal_stdlib>
using namespace metal;

// The 8x8 simdgroup_matrix MMA on operands held in plain registers, shared by
// the register matrix kernels (decode/linear_gguf_sgmatrix.metal,
// shared/gguf_float.metal).
namespace sgmatrix {
// simdgroup_matrix lane -> element mapping (verified by the driver's probe):
// a lane's thread_elements() are M[fm][fn] and M[fm][fn + 1].
struct Lane {
  ushort fm;
  ushort fn;
};
inline Lane lane_map(uint lane) {
  const uint qid = lane >> 2;
  Lane l;
  l.fm = ushort((qid & 4) | ((lane >> 1) & 3));
  l.fn = ushort(((qid & 2) << 1) | ((lane & 1) << 1));
  return l;
}

template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}

// One 8x8x8 MMA on plain-register operands: c += a x b. The simdgroup_matrix
// objects live only inside this call (the MLX steel pattern), so the
// persistent accumulator is an ordinary float2 the compiler keeps in
// registers; a persistent simdgroup_matrix read through thread_elements()
// each group was kept in thread memory instead.
template <typename T>
__attribute__((always_inline)) inline void
mma_acc(thread float2 &c, vec<T, 2> a, vec<T, 2> b) {
  simdgroup_matrix<T, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}
} // namespace sgmatrix
// fp32 projections of the GGUF float tensors llama.cpp keeps unquantized: the
// MoE router (ffn_gate_inp, F32, or BF16 as some GGUFs store it), the QSA
// indexer's query and key, which share their input and, on one tile, a
// dispatch, and, when a GGUF stores them as F32, the GDN alpha/beta gates
// (ops/Linear.hpp float segments). out[r][n] = sum_k x[r][k] W[n][k] on
// simdgroup float 8x8 MMA with exact operands (bf16 activations widen to
// fp32, the weights as stored, BF16 ones widened), so a result differs from
// the fp64 product only by fp32 accumulation, and a BF16 tensor's equals that
// of the F32 tensor of its values bit for bit. The reference GGUF
// implementation's Metal matrix multiply stages F32 weights as half above
// eight rows; no phase here rounds them.
#pragma clang fp reassociate(off)






// One projection of a dispatch (GgufFloatParams) and the column tile of a threadgroup within it: a dispatch's column
// tiles are its projections' in projection order.
template <typename W, typename T> struct FloatProjection {
  device const W *weights;
  device T *output;
  uint output_size, out_stride, out_offset, tile;
};
static_assert(GGUF_FLOAT_PROJECTIONS == 2, "a dispatch binds two projections");
// The projection of dispatch column tile `tile`, of `columns` columns.
template <typename W, typename T>
inline FloatProjection<W, T> float_projection(constant GgufFloatParams &p, uint tile, uint columns,
                                              device const W *weights0, device T *output0, device const W *weights1,
                                              device T *output1) {
  const uint tiles = (p.output_size[0] + columns - 1) / columns;
  return tile < tiles
             ? FloatProjection<W, T>{weights0, output0, p.output_size[0], p.out_stride[0], p.out_offset[0], tile}
             : FloatProjection<W, T>{weights1, output1, p.output_size[1], p.out_stride[1], p.out_offset[1],
                                     tile - tiles};
}

namespace gguf_float {

constant constexpr uint kFragments = 4;          // 8-row fragments per threadgroup: 32 rows
constant constexpr uint kRows = 8 * kFragments;
constant constexpr uint kStep = 64;              // inputs per step: eight runs of 8
// K parts, one per simdgroup. A decode dispatch has few threadgroups (64 at Flash-Next's router width, 32 at the
// 35B's); on the M5 Max at 8 to 32 rows, 8 parts are the fastest at those widths, where 16 took up to 29% longer, and
// 16 parts gained at most 19% at the narrow alpha/beta widths.
constant constexpr uint kSplits = 8;

// Eight consecutive values of a row as stored, 16 bytes of bf16 or 32 of fp32, each widened exactly to fp32 where
// read; zeros past the row's last run.
template <typename V> struct Eight;
template <> struct Eight<bfloat> {
  uint4 pairs;
  static Eight load(device const bfloat *values, bool inside) {
    return {inside ? *reinterpret_cast<device const uint4 *>(values) : uint4(0)};
  }
  float operator[](uint i) const { return float(as_type<bfloat2>(pairs[i / 2])[i % 2]); }
};
template <> struct Eight<float> {
  float4 low, high;
  static Eight load(device const float *values, bool inside) {
    device const float4 *quads = reinterpret_cast<device const float4 *>(values);
    return inside ? Eight{quads[0], quads[1]} : Eight{float4(0), float4(0)};
  }
  float operator[](uint i) const { return i < 4 ? low[i] : high[i - 4]; }
};

// One threadgroup: output columns [8 tile, 8 tile + 8) of the projection's rows [32 y, 32 y + 32). K runs in steps of
// kStep inputs, simdgroup s taking part s of the steps for every fragment; simdgroup f then adds fragment f's parts in
// order, so a result depends neither on the grid nor on the row count. MMA j of the step from input k multiplies
// inputs k + 8 i + j over its index i, so each lane's operands are whole runs of 8 inputs: lane (fm, fn) loads
// W[column fn, fn + 1][k + 8 fm, + 8) as B and x[row fm][k + 8 fn, + 16) as A, a column's runs 128 contiguous bytes
// of bf16 over its eight lanes, and holds the destination [row fm][column fn, fn + 1] (sgmatrix::lane_map). The
// weights load a step ahead. Even and odd MMAs accumulate apart, which halves each chain of fp32 sums at no cost in
// time: at the router's K = 2048 the mean error stays 5.5e-9 of sum |x w| (7.4e-9 with one chain per part). Rows of
// K (a multiple of 8) start 16-byte aligned in page-aligned buffers.
template <typename W, typename T>
inline void project(device const bfloat *input, FloatProjection<W, T> projection, constant GgufFloatParams &p, uint y,
                    uint sg, uint lane, threadgroup float2 *parts) {
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint K = p.input_size, runs = K / 8, column = projection.tile * 8 + l.fn, row0 = y * kRows;
  const uint live = min(kRows, p.rows - row0);
  const uint steps = (K + kStep - 1) / kStep, first = sg * steps / kSplits, last = (sg + 1) * steps / kSplits;
  device const W *w = projection.weights + ulong(column) * K;
  const auto load = [&](uint step, thread Eight<W> (&b)[2]) {
    const uint run = step * 8 + l.fm;
    b[0] = Eight<W>::load(w + run * 8, run < runs);
    b[1] = Eight<W>::load(w + K + run * 8, run < runs);
  };
  float2 acc[kFragments][2];
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) acc[f][0] = acc[f][1] = float2(0);
  Eight<W> next[2];
  load(first, next);
  for (uint step = first; step < last; ++step) {
    const Eight<W> b0 = next[0], b1 = next[1];
    if (step + 1 < last) load(step + 1, next);
    const uint run = step * 8 + l.fn;
#pragma unroll
    for (uint f = 0; f < kFragments; ++f) {
      if (f * 8 >= live) break;
      const uint row = f * 8 + l.fm;
      device const bfloat *x = input + ulong(row0 + row) * K + run * 8;
      const Eight<bfloat> a0 = Eight<bfloat>::load(x, row < live && run < runs);
      const Eight<bfloat> a1 = Eight<bfloat>::load(x + 8, row < live && run + 1 < runs);
#pragma unroll
      for (uint j = 0; j < 8; ++j)
        sgmatrix::mma_acc<float>(acc[f][j % 2], float2(a0[j], a1[j]), float2(b0[j], b1[j]));
    }
  }
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) parts[(sg * kFragments + f) * 32 + lane] = acc[f][0] + acc[f][1];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint f = sg, row = f * 8 + l.fm;
  if (f >= kFragments || row >= live) return;
  float2 total = float2(0);
  for (uint s = 0; s < kSplits; ++s) total += parts[(s * kFragments + f) * 32 + lane];
  device T *out = projection.output + ulong(row0 + row) * projection.out_stride + projection.out_offset + column;
  out[0] = T(total.x);
  out[1] = T(total.y);
}

} // namespace gguf_float

static_assert(gguf_float::kSplits >= gguf_float::kFragments, "simdgroup f reduces fragment f");

// Grid (the projections' output_size / 8 summed, ceil(rows / 32)), 32 kSplits threads; named by the weights' and the
// output's types.
#define GGUF_FLOAT_KERNEL(Name, W, T)                                                                          \
  kernel void Name(device const bfloat *input [[buffer(0)]], device const W *weights0 [[buffer(1)]],          \
                   device T *output0 [[buffer(2)]], device const W *weights1 [[buffer(3)]],                   \
                   device T *output1 [[buffer(4)]], constant GgufFloatParams &p [[buffer(5)]],                \
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],     \
                   uint lane [[thread_index_in_simdgroup]]) {                                                 \
    threadgroup float2 parts[gguf_float::kSplits * gguf_float::kFragments * 32];                              \
    gguf_float::project<W, T>(input, float_projection(p, tg.x, 8, weights0, output0, weights1, output1), p,   \
                              tg.y, sg, lane, parts);                                                         \
  }



GGUF_FLOAT_KERNEL(mlx_splash_source_router, bfloat, float)
#undef GGUF_FLOAT_KERNEL

// Routes group rows by expert so each tile streams its weights once. Each row
// carries top_k routed experts plus the shared expert (id `experts`), weighted
// by the sigmoid of its scalar gate.
//
// One row per threadgroup: thread e holds expert e's score, the threadgroup
// reduces the shared expert's F32 scalar gate with a fixed partial-sum order,
// and simdgroup 0 orders the experts, descending score then ascending id,
// which is the order the shape's routing and tie-break contract requires.
// Scores and routing weights are fp32. On 40,960 router rows each of real 35B
// prose, code and chat, top-8 sets chosen from bf16-rounded scores differ from
// the fp64 choice on 12-15% of rows (fp32: at most 0.0024%), and bf16 weights
// round 19-20% of the combine's outputs away from bf16 of its exact sum
// (fp32: 0.008%).
template <uint Slots>
inline void moe_route_select(device const float *scores, device bfloat *input,
                             device const bfloat *shared_gate, device uint *selected,
                             device float *routing_weights,
                             constant MoeRouteParams &params, uint group,
                             uint thread_index, uint simd_lane,
                             uint simd_group, threadgroup float *row_scores,
                             threadgroup float *ordered,
                             threadgroup float *scalar_partials) {
  // Slots expert slots, one per thread.
  constexpr uint Simdgroups = Slots / 32;
  constexpr uint ExpertsPerLane = Slots / 32;
  const uint row = group;
  if (row >= params.rows)
    return;
  row_scores[thread_index] =
      thread_index < params.experts
          ? scores[ulong(row) * Slots + thread_index]
          : -numeric_limits<float>::infinity();

  device bfloat *row_input = input + ulong(row) * params.input_size;
  float scalar = 0.0f;
  for (uint dimension = thread_index; dimension < params.input_size;
       dimension += Slots) {
    scalar += float(row_input[dimension]) *
              float(shared_gate[dimension]);
  }
  scalar = simd_sum(scalar);
  if (simd_lane == 0)
    scalar_partials[simd_group] = scalar;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong row_routes = ulong(row) * (params.top_k + 1);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint partial = 0; partial < Simdgroups; ++partial)
      total += scalar_partials[partial];
    selected[row_routes + params.top_k] = params.experts;
    routing_weights[row_routes + params.top_k] = splash_sigmoid(total);
  }
  if (simd_group != 0)
    return;
  // Lane l holds experts l, l + 32, ... Each rank takes the highest remaining
  // score and the lowest expert id among ties.
  float lane_scores[ExpertsPerLane];
  for (uint slot = 0; slot < ExpertsPerLane; ++slot)
    lane_scores[slot] = row_scores[simd_lane + 32 * slot];
  for (uint rank = 0; rank < params.top_k; ++rank) {
    float best = -numeric_limits<float>::infinity();
    uint best_slot = 0;
    for (uint slot = 0; slot < ExpertsPerLane; ++slot) {
      if (lane_scores[slot] > best) {
        best = lane_scores[slot];
        best_slot = slot;
      }
    }
    const float row_best = simd_max(best);
    const uint candidate =
        best == row_best ? simd_lane + 32 * best_slot : 0xFFFFFFFFu;
    const uint winner = simd_min(candidate);
    if (candidate == winner)
      lane_scores[best_slot] = -numeric_limits<float>::infinity();
    if (simd_lane == 0) {
      ordered[rank] = row_best;
      selected[row_routes + rank] = winner;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint rank = simd_lane; rank < params.top_k; rank += 32) {
    float denominator = 0.0f;
    for (uint other = 0; other < params.top_k; ++other) {
      denominator +=
          fast::exp2((ordered[other] - ordered[0]) * 1.44269504089f);
    }
    routing_weights[row_routes + rank] =
        fast::exp2((ordered[rank] - ordered[0]) * 1.44269504089f) /
        denominator;
  }
}

// The GGUF router: fp32 scores (kernels/shared/gguf_float.metal) in rows of
// Slots expert slots and the F32 shared-expert scalar gate, on
// SPLASH_MOE_EXPERT_SLOTS slots or (_e512) SPLASH_MOE_WIDE_EXPERT_SLOTS.
#define MOE_ROUTE_SELECT_ENTRY(Name, Slots)                                    \
  kernel void Name(                                                            \
      device const float *scores [[buffer(0)]],                                \
      device bfloat *input [[buffer(1)]],                                      \
      device const bfloat *shared_gate [[buffer(2)]],                           \
      device uint *selected [[buffer(3)]],                                     \
      device float *routing_weights [[buffer(4)]],                             \
      constant MoeRouteParams &params [[buffer(5)]],                           \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    threadgroup float row_scores[Slots];                                       \
    threadgroup float ordered[Slots];                                          \
    threadgroup float scalar_partials[Slots / 32];                             \
    moe_route_select<Slots>(scores, input, shared_gate, selected,              \
                            routing_weights, params, group, thread_index,      \
                            simd_lane, simd_group, row_scores, ordered,        \
                            scalar_partials);                                  \
  }
MOE_ROUTE_SELECT_ENTRY(mlx_splash_source_route, SPLASH_MOE_EXPERT_SLOTS)

#undef MOE_ROUTE_SELECT_ENTRY


)METAL"
