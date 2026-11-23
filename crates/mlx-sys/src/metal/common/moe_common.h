#pragma once
#include <metal_stdlib>

// Shared safe AOT MoE Metal scaffolding for Splash MoE and Inkling-2 MoE.
//
// NOTE: Do NOT place `#pragma clang fp contract(off)` at file scope in this
// header, because `splash_moe/moe.metal` compiles without global FP contraction
// disabled while Inkling-2 kernels require it. Any helper needing strict
// no-FMA semantics scopes `#pragma clang fp contract(off)` inside its body.

// Round a live tile row count up to the smallest supported NAX/MPP matmul
// tile height (8, 16, or `tile_rows`).
inline uint moe_matmul_rows(uint rows, uint tile_rows) {
  return metal::min(tile_rows, rows <= 8u ? 8u : rows <= 16u ? 16u : tile_rows);
}

// Exclusive prefix sum across a 256-thread (8-simdgroup) threadgroup.
inline uint moe_exclusive_prefix_sum_256(
    uint value,
    threadgroup uint (&totals)[8],
    uint lane,
    uint sg) {
  uint start = metal::simd_prefix_exclusive_sum(value);
  if (lane == 31u) {
    totals[sg] = start + value;
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint i = 0u; i < sg; ++i) {
    start += totals[i];
  }
  return start;
}

// Explicit top-K comparison with tie-break polarity baked into the template
// parameter so callers cannot silently mix conventions:
//   TieBreakSmaller = true  -> Inkling-2 (smaller expert id wins ties: `<`)
//   TieBreakSmaller = false -> Splash    (larger  expert id wins ties: `>`)
template <bool TieBreakSmaller>
inline bool moe_topk_better(float cand_v, uint cand_e, float best_v, uint best_e) {
  if constexpr (TieBreakSmaller) {
    return cand_v > best_v || (cand_v == best_v && cand_e < best_e);
  } else {
    return cand_v > best_v || (cand_v == best_v && cand_e > best_e);
  }
}

inline float moe_read_f32(device const uchar *ptr, ulong idx, uint is_f32) {
  if (is_f32 != 0u) {
    return reinterpret_cast<device const float *>(ptr)[idx];
  }
  return float(reinterpret_cast<device const bfloat *>(ptr)[idx]);
}

// Single-simdgroup (32-lane) exact FP32 sigmoid + optional bias + 5-step
// butterfly top-K selection + sum normalization + route_scale + optional
// per-expert down_scale. Calls `emit(k, eid, w)` on `lane == 0u` for each
// `k in 0..K`.
template <uint MaxSlots, bool TieBreakSmaller, typename EmitFn>
inline void moe_sigmoid_bias_topk_row(
    device const uchar *logits_raw,
    ulong row_base,
    device const uchar *bias_raw,
    device const uchar *down_scale_raw,
    uint E,
    uint K,
    float route_scale,
    uint logits_f32,
    bool has_bias,
    uint bias_f32,
    bool has_down_scale,
    uint ds_f32,
    uint lane,
    EmitFn emit) {
  #pragma clang fp contract(off)
  float local_s[MaxSlots];
  float local_p[MaxSlots];
  int32_t local_id[MaxSlots];
  uint count = 0u;

  for (uint e = lane; e < E; e += 32u) {
    const float z = moe_read_f32(logits_raw, row_base + e, logits_f32);
    const float p = 1.0f / (1.0f + metal::exp(-z));
    const float s = has_bias ? (p + moe_read_f32(bias_raw, ulong(e), bias_f32)) : p;
    local_s[count] = s;
    local_p[count] = p;
    local_id[count] = int32_t(e);
    ++count;
  }

  int32_t sel_idx[16];
  float sel_p[16];
  float sum_p = 0.0f;

  for (uint k = 0u; k < K; ++k) {
    float best_s = -INFINITY;
    int32_t best_i = -1;
    float best_p = 0.0f;
    for (uint j = 0u; j < count; ++j) {
      const int32_t cid = local_id[j];
      if (cid >= 0) {
        const float cs = local_s[j];
        if (best_i < 0 ||
            moe_topk_better<TieBreakSmaller>(cs, uint(cid), best_s, uint(best_i))) {
          best_s = cs;
          best_i = cid;
          best_p = local_p[j];
        }
      }
    }

    // 5-step butterfly simdgroup reduction with explicit tie-break polarity.
    for (ushort mask = 16; mask > 0; mask >>= 1) {
      const float os = metal::simd_shuffle_xor(best_s, mask);
      const int32_t oi = metal::simd_shuffle_xor(best_i, mask);
      const float op = metal::simd_shuffle_xor(best_p, mask);
      if (oi >= 0 &&
          (best_i < 0 ||
           moe_topk_better<TieBreakSmaller>(os, uint(oi), best_s, uint(best_i)))) {
        best_s = os;
        best_i = oi;
        best_p = op;
      }
    }

    // Mark winner as picked in owning lane.
    for (uint j = 0u; j < count; ++j) {
      if (local_id[j] == best_i) {
        local_id[j] = -1;
      }
    }

    if (lane == 0u) {
      sel_idx[k] = best_i;
      sel_p[k] = best_p;
      sum_p += best_p;
    }
  }

  if (lane == 0u) {
    const float denom = sum_p + 1e-20f;
    for (uint k = 0u; k < K; ++k) {
      const int32_t eid = sel_idx[k];
      float w = (sel_p[k] / denom) * route_scale;
      if (has_down_scale) {
        w *= moe_read_f32(down_scale_raw, ulong(uint32_t(eid)), ds_f32);
      }
      emit(k, eid, w);
    }
  }
}
