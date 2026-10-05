// M = 8 bfloat16 K-quant matmul on the Metal 4 tensor op (NAX).
//
// Ported from Splash (incoai/splash, Apache-2.0; see THIRD_PARTY_NOTICES):
// runtime/metal/kernels/common/gguf_staged.h, gguf_staged_tile.h,
// split_reduce.h and kernels/shared/gguf_linear.metal's decode tile, with the
// per-format decode rewritten for mlx-node's LSB-first code stream and
// (.scales, .biases) companions (crates/mlx-core/src/utils/gguf_kquant.rs).
//
// One threadgroup = 2 simdgroups x 32 output columns = a 64-column tile;
// grid (N / 64, K splits). Per 32-input step every lane dequantizes its
// column's 32 codes to half into a double-buffered threadgroup stage (fp32
// scale * code + bias, one rounding), then the simdgroup runs matmul2d
// (8 x 32 x 32, A = bfloat x in device memory, B = the half stage, fp32
// accumulate) on it while the next step's codes are already in flight.
// Splits > 1 publish fp32 partials; the last-arriving partition of a tile
// adds them in split order and writes the bfloat16 output, so the result does
// not depend on scheduling.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

namespace kq_m8 {

constant constexpr ushort kRows = 8;
constant constexpr ushort kCols = 32;
constant constexpr ushort kStep = 32;
constant constexpr ushort kTileCols = 64;
constant constexpr ushort kThreads = 64;
constant constexpr uint kStage = kCols * kStep;
constant constexpr uint kTableEntries = 256;

enum Format : int { Q4K, Q5K, Q6K, Q3K, IQ4NL, IQ4XS, IQ3S, Unsupported };

template <int group_size, int bits, int super_ratio, bool has_min>
constexpr Format format() {
  if (group_size == 32 && super_ratio == 8 && bits == 4) {
    return has_min ? Q4K : IQ4XS;
  }
  if (group_size == 32 && super_ratio == 8 && bits == 5 && has_min) {
    return Q5K;
  }
  if (group_size == 16 && super_ratio == 16 && bits == 6 && !has_min) {
    return Q6K;
  }
  if (group_size == 16 && super_ratio == 16 && bits == 3 && !has_min) {
    return Q3K;
  }
  if (group_size == 32 && super_ratio == 1 && bits == 4 && !has_min) {
    return IQ4NL;
  }
  if (group_size == 32 && super_ratio == 8 && bits == 8 && !has_min) {
    return IQ3S;
  }
  return Unsupported;
}

constant constexpr int8_t kIQ4[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// .s / .m of the first 16 codes of a unit (.x) and of the last 16 (.y); equal
// for the 32-code groups. Same operand order as KQScales (mlx_kquant.cpp).
struct Coef {
  float2 s;
  float2 m;
};

// A unit is 32 consecutive k of one weight row: `bits` uint32 words.
template <Format F>
struct Codes;

template <>
struct Codes<Q4K> {
  typedef uint4 W;
  static W load(const device uint32_t* row, uint u) {
    return *reinterpret_cast<const device uint4*>(row + u * 4);
  }
};
template <>
struct Codes<IQ4XS> {
  typedef uint4 W;
  static W load(const device uint32_t* row, uint u) {
    return Codes<Q4K>::load(row, u);
  }
};
template <>
struct Codes<IQ4NL> {
  typedef uint4 W;
  static W load(const device uint32_t* row, uint u) {
    return Codes<Q4K>::load(row, u);
  }
};

template <>
struct Codes<Q5K> {
  struct W {
    uint4 a;
    uint b;
  };
  static W load(const device uint32_t* row, uint u) {
    const device uint32_t* s = row + u * 5;
    return {uint4(s[0], s[1], s[2], s[3]), s[4]};
  }
};

template <>
struct Codes<Q6K> {
  struct W {
    uint4 a;
    uint2 b;
  };
  static W load(const device uint32_t* row, uint u) {
    const device uint2* s = reinterpret_cast<const device uint2*>(row + u * 6);
    return {uint4(s[0], s[1]), s[2]};
  }
};

template <>
struct Codes<Q3K> {
  struct W {
    uint2 a;
    uint b;
  };
  static W load(const device uint32_t* row, uint u) {
    const device uint32_t* s = row + u * 3;
    return {uint2(s[0], s[1]), s[2]};
  }
};

template <>
struct Codes<IQ3S> {
  struct W {
    uint4 a;
    uint4 b;
  };
  static W load(const device uint32_t* row, uint u) {
    const device uint4* s = reinterpret_cast<const device uint4*>(row + u * 8);
    return {s[0], s[1]};
  }
};

template <Format F>
METAL_FUNC Coef coef(
    const device uint8_t* scales,
    const device half* biases,
    uint n,
    uint K,
    uint u) {
  Coef c;
  if constexpr (F == Q4K || F == Q5K) {
    const uint sm = *reinterpret_cast<const device ushort*>(
        scales + size_t(n) * (K / 16) + 2 * u);
    const half2 d = *reinterpret_cast<const device half2*>(
        biases + size_t(n) * (K / 128) + 2 * (u >> 3));
    c.s = float2(float(d.x) * float(sm & 0xFFu));
    c.m = float2(-(float(d.y) * float(sm >> 8)));
  } else if constexpr (F == Q6K || F == Q3K) {
    const uint sc = *reinterpret_cast<const device ushort*>(
        scales + size_t(n) * (K / 16) + 2 * u);
    const float d = float(biases[size_t(n) * (K / 256) + (u >> 3)]);
    c.s = float2(
        d * float(as_type<char>(uchar(sc & 0xFFu))),
        d * float(as_type<char>(uchar(sc >> 8))));
    c.m = float(-(F == Q6K ? 32 : 4)) * c.s;
  } else if constexpr (F == IQ4NL) {
    const float d = float(biases[size_t(n) * (K / 32) + u]);
    c.s = float2(d * float(as_type<char>(scales[size_t(n) * (K / 32) + u])));
    c.m = float2(0.0f);
  } else {
    const float d = float(biases[size_t(n) * (K / 256) + (u >> 3)]);
    c.s = float2(d * float(as_type<char>(scales[size_t(n) * (K / 32) + u])));
    c.m = F == IQ3S ? float(-128) * c.s : float2(0.0f);
  }
  return c;
}

// Codes 8j..8j+7 of a unit as bytes: the even ones in .x, the odd in .y.
template <Format F>
METAL_FUNC uint2 bytes8(typename Codes<F>::W w, ushort j);

template <>
METAL_FUNC uint2 bytes8<Q4K>(uint4 w, ushort j) {
  const uint v = w[j];
  return uint2(v & 0x0F0F0F0Fu, (v >> 4) & 0x0F0F0F0Fu);
}

template <>
METAL_FUNC uint2 bytes8<Q5K>(Codes<Q5K>::W w, ushort j) {
  // Codes 8j.. start at bit 40j: word j, shift 8j.
  const uint lo0 = w.a[j];
  const uint hi0 = j < 3 ? w.a[(j + 1) & 3] : w.b;
  const uint sh = 8 * j;
  const uint lo = sh ? (lo0 >> sh) | (hi0 << (32 - sh)) : lo0;
  const uint hi = hi0 >> sh;
  const uint even = (lo & 0x1Fu) | ((lo >> 2) & 0x1F00u) |
      ((lo >> 4) & 0x1F0000u) | ((lo >> 6) & 0x03000000u) | ((hi & 7u) << 26);
  const uint odd = ((lo >> 5) & 0x1Fu) | ((lo >> 7) & 0x1F00u) |
      ((lo >> 9) & 0x1F0000u) | ((hi << 21) & 0x1F000000u);
  return uint2(even, odd);
}

template <>
METAL_FUNC uint2 bytes8<Q6K>(Codes<Q6K>::W w, ushort j) {
  // Codes 8j.. start at bit 48j: words {0, 1}, {1, 2}, {3, 4}, {4, 5}.
  uint lo, hi;
  if (j == 0) {
    lo = w.a.x;
    hi = w.a.y;
  } else if (j == 1) {
    lo = (w.a.y >> 16) | (w.a.z << 16);
    hi = w.a.z >> 16;
  } else if (j == 2) {
    lo = w.a.w;
    hi = w.b.x;
  } else {
    lo = (w.b.x >> 16) | (w.b.y << 16);
    hi = w.b.y >> 16;
  }
  const uint even = (lo & 0x3Fu) | ((lo >> 4) & 0x3F00u) |
      ((lo >> 8) & 0x3F0000u) | ((hi << 20) & 0x3F000000u);
  const uint odd = ((lo >> 6) & 0x3Fu) | ((lo >> 10) & 0x3F00u) |
      ((lo >> 14) & 0x030000u) | ((hi & 15u) << 18) |
      ((hi << 14) & 0x3F000000u);
  return uint2(even, odd);
}

template <>
METAL_FUNC uint2 bytes8<Q3K>(Codes<Q3K>::W w, ushort j) {
  // Codes 8j.. start at bit 24j.
  uint v;
  if (j == 0) {
    v = w.a.x;
  } else if (j == 1) {
    v = (w.a.x >> 24) | (w.a.y << 8);
  } else if (j == 2) {
    v = (w.a.y >> 16) | (w.b << 16);
  } else {
    v = w.b >> 8;
  }
  const uint even = (v & 7u) | ((v << 2) & 0x700u) | ((v << 4) & 0x70000u) |
      ((v << 6) & 0x7000000u);
  const uint odd = ((v >> 3) & 7u) | ((v >> 1) & 0x700u) |
      ((v << 1) & 0x70000u) | ((v << 3) & 0x7000000u);
  return uint2(even, odd);
}

template <>
METAL_FUNC uint2 bytes8<IQ3S>(Codes<IQ3S>::W w, ushort j) {
  const uint4 h = j < 2 ? w.a : w.b;
  const uint w0 = h[2 * (j & 1)];
  const uint w1 = h[2 * (j & 1) + 1];
  const uint even = (w0 & 0xFFu) | ((w0 >> 8) & 0xFF00u) |
      ((w1 << 16) & 0xFF0000u) | ((w1 << 8) & 0xFF000000u);
  const uint odd = ((w0 >> 8) & 0xFFu) | ((w0 >> 16) & 0xFF00u) |
      ((w1 << 8) & 0xFF0000u) | (w1 & 0xFF000000u);
  return uint2(even, odd);
}

// One lane's unit as 32 half values at dst, each rounded once from fp32.
template <Format F>
METAL_FUNC void stage32(
    typename Codes<F>::W w,
    Coef c,
    threadgroup const half2* tl,
    threadgroup half* dst) {
  if constexpr (F == IQ4XS || F == IQ4NL) {
    const float s = c.s.x;
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      const uchar4 b = as_type<uchar4>(w[j]);
      const half2 v0 = half2(float2(tl[b.x]) * s);
      const half2 v1 = half2(float2(tl[b.y]) * s);
      const half2 v2 = half2(float2(tl[b.z]) * s);
      const half2 v3 = half2(float2(tl[b.w]) * s);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j) = half4(v0, v1);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j + 4) = half4(v2, v3);
    }
  } else {
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      const uint2 q = bytes8<F>(w, j);
      const float s = j < 2 ? c.s.x : c.s.y;
      const float m = j < 2 ? c.m.x : c.m.y;
      const half4 e = half4(fma(float4(as_type<uchar4>(q.x)), float4(s), float4(m)));
      const half4 o = half4(fma(float4(as_type<uchar4>(q.y)), float4(s), float4(m)));
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j) = half4(e.x, o.x, e.y, o.y);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j + 4) = half4(e.z, o.z, e.w, o.w);
    }
  }
}

// --- split-K reduction (Splash split_reduce.h) ---------------------------

// True in every thread of the threadgroup that arrives last at `counter`.
METAL_FUNC bool split_arrive_last(
    device atomic_uint* counter,
    uint splits,
    uint thread_index,
    threadgroup uint* arrival) {
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(
        mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    atomic_thread_fence(
        mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  return *arrival == splits - 1;
}

METAL_FUNC void split_release(device atomic_uint* counter, uint thread_index) {
  if (thread_index == 0) {
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
  }
}

} // namespace kq_m8

// Grid (N / 64, splits) of 64 threads. x is [8][K] bfloat16, y [8][N];
// partials [splits][8][N] fp32 and counters [N / 64] (zero before and after
// each dispatch) are read only when splits > 1.
template <typename T, int group_size, int bits, int super_ratio, bool has_min>
[[kernel, max_total_threads_per_threadgroup(64)]] void kquant_qmm_m8_nax(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device half* biases [[buffer(2)]],
    device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    device coherent(device) float* partials [[buffer(5)]],
    device atomic_uint* counters [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& num_splits [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  using namespace kq_m8;
  constexpr Format F = format<group_size, bits, super_ratio, has_min>();
  static_assert(F != Unsupported, "no M = 8 NAX decode for this mode");
  static_assert(is_same_v<T, bfloat>, "the tensor-op A operand is bfloat16");
  constexpr bool codebook = F == IQ4XS || F == IQ4NL;
  typedef typename Codes<F>::W W;

  threadgroup half stage[2 * 2 * kStage];
  threadgroup half2 tl[codebook ? kTableEntries : 1];
  threadgroup uint arrival;

  const uint tid = simd_group * 32 + simd_lane;
  if constexpr (codebook) {
    for (uint i = tid; i < kTableEntries; i += kThreads) {
      tl[i] = half2(float(kIQ4[i & 15]), float(kIQ4[i >> 4]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint K = in_vec_size;
  const uint N = out_vec_size;
  const uint splits = num_splits;
  const uint per = (K / kStep) / splits;
  const uint step_begin = group.y * per;
  const uint step_end = step_begin + per;
  const uint column0 = group.x * kTileCols + simd_group * kCols;
  const uint n = column0 + simd_lane;
  threadgroup half* my = stage + simd_group * 2 * kStage;

  auto a = tensor(x, dextents<int, 2>{int(K), kRows}, array<int, 2>{1, int(K)});
  constexpr auto desc = matmul2d_descriptor(
      kRows, kCols, kStep, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
      my, dextents<int, 2>{kStep, kCols}, array<int, 2>{1, kStep});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
      my + kStage, dextents<int, 2>{kStep, kCols}, array<int, 2>{1, kStep});
  auto b0 = bt0.template slice<kStep, kCols>(0, 0);
  auto b1 = bt1.template slice<kStep, kCols>(0, 0);
  auto a0 = a.template slice<kStep, kRows>(0, 0);
  // Zeroed here, not in a helper: a returned initialized cooperative tensor
  // loses its values (Splash gguf_staged_tile.h).
  auto acc = op.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }

  // The codes and coefficients of a step are loaded during the step before.
  const device uint32_t* row = w + size_t(n) * (K * bits / 32);
  W cur = Codes<F>::load(row, step_begin);
  Coef cc = coef<F>(scales, biases, n, K, step_begin);
  for (uint step = step_begin; step < step_end; ++step) {
    threadgroup half* buf = my + (step & 1) * kStage;
    stage32<F>(cur, cc, tl, buf + simd_lane * kStep);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (step + 1 < step_end) {
      cur = Codes<F>::load(row, step + 1);
      cc = coef<F>(scales, biases, n, K, step + 1);
    }
    auto as = a.template slice<kStep, kRows>(int(step * kStep), 0);
    if (step & 1) {
      op.run(as, b1, acc);
    } else {
      op.run(as, b0, acc);
    }
  }

  // index[0] is the column, index[1] the row.
  if (splits == 1) {
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) {
        continue;
      }
      const auto idx = acc.get_multidimensional_index(i);
      y[uint(idx[1]) * N + column0 + uint(idx[0])] = T(float(acc[i]));
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
    float total = 0.0f;
    for (uint s = 0; s < splits; ++s) {
      total += s == split ? float(acc[i]) : partials[at(s, row_, col)];
    }
    y[row_ * N + column0 + col] = T(total);
  }
  split_release(counters + group.x, tid);
}
