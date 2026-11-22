R"METAL(
const int M = logits_shape[0];
const int E = logits_shape[1];
const uint m = threadgroup_position_in_grid.y;
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
if (int(m) >= M) return;

constexpr int MAX_ITERS = (MAX_E + 127) / 128;
threadgroup float tg_s[MAX_E];
threadgroup float tg_cand_val[4 * K];
threadgroup uint tg_cand_id[4 * K];

const size_t row_off = size_t(m) * size_t(E);

if constexpr (MAX_ITERS == 4) {
  float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY;
  uint i0 = 0xFFFFFFFFu, i1 = 0xFFFFFFFFu, i2 = 0xFFFFFFFFu, i3 = 0xFFFFFFFFu;

  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 0 * 128, v0, i0);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 1 * 128, v1, i1);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 2 * 128, v2, i2);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 3 * 128, v3, i3);

  // 4-element sorting network in registers (best at slot 0).
  hy4_topk_cswap(v0, i0, v1, i1);
  hy4_topk_cswap(v2, i2, v3, i3);
  hy4_topk_cswap(v0, i0, v2, i2);
  hy4_topk_cswap(v1, i1, v3, i3);
  hy4_topk_cswap(v1, i1, v2, i2);

  // Each simdgroup extracts its local top-K in registers with zero TG barriers.
  for (int k = 0; k < K; ++k) {
    const float sg_max = simd_max(v0);
    const uint cand = (i0 != 0xFFFFFFFFu && v0 == sg_max) ? i0 : 0xFFFFFFFFu;
    const uint win_id = simd_min(cand);
    if (lane == 0u) {
      tg_cand_val[sg * uint(K) + uint(k)] = (win_id != 0xFFFFFFFFu) ? sg_max : -INFINITY;
      tg_cand_id[sg * uint(K) + uint(k)] = win_id;
    }
    if (win_id != 0xFFFFFFFFu && i0 == win_id) {
      v0 = v1; i0 = i1;
      v1 = v2; i1 = i2;
      v2 = v3; i2 = i3;
      v3 = -INFINITY; i3 = 0xFFFFFFFFu;
    }
  }
} else {
  float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY;
  float v4 = -INFINITY, v5 = -INFINITY, v6 = -INFINITY, v7 = -INFINITY;
  uint i0 = 0xFFFFFFFFu, i1 = 0xFFFFFFFFu, i2 = 0xFFFFFFFFu, i3 = 0xFFFFFFFFu;
  uint i4 = 0xFFFFFFFFu, i5 = 0xFFFFFFFFu, i6 = 0xFFFFFFFFu, i7 = 0xFFFFFFFFu;

  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 0 * 128, v0, i0);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 1 * 128, v1, i1);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 2 * 128, v2, i2);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 3 * 128, v3, i3);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 4 * 128, v4, i4);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 5 * 128, v5, i5);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 6 * 128, v6, i6);
  hy4_topk_load_one<HAS_BIAS>(logits, bias, tg_s, row_off, E, int(tid) + 7 * 128, v7, i7);

  // 8-element sorting network in registers (best at slot 0).
  hy4_topk_cswap(v0, i0, v1, i1);
  hy4_topk_cswap(v2, i2, v3, i3);
  hy4_topk_cswap(v4, i4, v5, i5);
  hy4_topk_cswap(v6, i6, v7, i7);
  hy4_topk_cswap(v0, i0, v2, i2);
  hy4_topk_cswap(v1, i1, v3, i3);
  hy4_topk_cswap(v4, i4, v6, i6);
  hy4_topk_cswap(v5, i5, v7, i7);
  hy4_topk_cswap(v1, i1, v2, i2);
  hy4_topk_cswap(v5, i5, v6, i6);
  hy4_topk_cswap(v0, i0, v4, i4);
  hy4_topk_cswap(v1, i1, v5, i5);
  hy4_topk_cswap(v2, i2, v6, i6);
  hy4_topk_cswap(v3, i3, v7, i7);
  hy4_topk_cswap(v2, i2, v4, i4);
  hy4_topk_cswap(v3, i3, v5, i5);
  hy4_topk_cswap(v1, i1, v2, i2);
  hy4_topk_cswap(v3, i3, v4, i4);
  hy4_topk_cswap(v5, i5, v6, i6);

  for (int k = 0; k < K; ++k) {
    const float sg_max = simd_max(v0);
    const uint cand = (i0 != 0xFFFFFFFFu && v0 == sg_max) ? i0 : 0xFFFFFFFFu;
    const uint win_id = simd_min(cand);
    if (lane == 0u) {
      tg_cand_val[sg * uint(K) + uint(k)] = (win_id != 0xFFFFFFFFu) ? sg_max : -INFINITY;
      tg_cand_id[sg * uint(K) + uint(k)] = win_id;
    }
    if (win_id != 0xFFFFFFFFu && i0 == win_id) {
      v0 = v1; i0 = i1;
      v1 = v2; i1 = i2;
      v2 = v3; i2 = i3;
      v3 = v4; i3 = i4;
      v4 = v5; i4 = i5;
      v5 = v6; i5 = i6;
      v6 = v7; i6 = i7;
      v7 = -INFINITY; i7 = 0xFFFFFFFFu;
    }
  }
}

threadgroup_barrier(mem_flags::mem_threadgroup);

if (sg == 0u) {
  const uint rev = 31u - lane;

  // Merge SG 0 and SG 1 sorted top-K lists via bitonic half-cleaner + 32-lane merge.
  float a_val = (int(lane) < K) ? tg_cand_val[lane] : -INFINITY;
  uint a_id = (int(lane) < K) ? tg_cand_id[lane] : 0xFFFFFFFFu;
  float b_val = (int(rev) < K) ? tg_cand_val[uint(K) + rev] : -INFINITY;
  uint b_id = (int(rev) < K) ? tg_cand_id[uint(K) + rev] : 0xFFFFFFFFu;
  const bool take_a01 = hy4_topk_better(a_val, a_id, b_val, b_id);
  float m01_val = take_a01 ? a_val : b_val;
  uint m01_id = take_a01 ? a_id : b_id;
  hy4_topk_bitonic_merge32(m01_val, m01_id, lane);

  // Merge SG 2 and SG 3 sorted top-K lists.
  a_val = (int(lane) < K) ? tg_cand_val[2u * uint(K) + lane] : -INFINITY;
  a_id = (int(lane) < K) ? tg_cand_id[2u * uint(K) + lane] : 0xFFFFFFFFu;
  b_val = (int(rev) < K) ? tg_cand_val[3u * uint(K) + rev] : -INFINITY;
  b_id = (int(rev) < K) ? tg_cand_id[3u * uint(K) + rev] : 0xFFFFFFFFu;
  const bool take_a23 = hy4_topk_better(a_val, a_id, b_val, b_id);
  float m23_val = take_a23 ? a_val : b_val;
  uint m23_id = take_a23 ? a_id : b_id;
  hy4_topk_bitonic_merge32(m23_val, m23_id, lane);

  // Merge M01 and reversed M23 to obtain global top-K in exact descending rank order.
  const float b23_val = simd_shuffle(m23_val, ushort(rev));
  const uint b23_id = simd_shuffle(m23_id, ushort(rev));
  const bool take_01 = hy4_topk_better(m01_val, m01_id, b23_val, b23_id);
  float val = take_01 ? m01_val : b23_val;
  uint id = take_01 ? m01_id : b23_id;
  hy4_topk_bitonic_merge32(val, id, lane);

  // Mask lanes >= K before ascending-ID sort so rank >= K never enters top-K.
  float my_s = (int(lane) < K) ? tg_s[id] : 0.0f;
  if (int(lane) >= K) {
    id = 0xFFFFFFFFu;
  }

  // Rank-order summation preserves bitwise parity with host route_select.
  float sum_s = 0.0f;
  for (int i = 0; i < K; ++i) {
    sum_s += simd_shuffle(my_s, ushort(i));
  }
  sum_s += params[1];

  if constexpr (K > 1) {
    constexpr uint N_SORT = (K <= 16) ? 16u : 32u;
    for (uint size = 2u; size <= N_SORT; size <<= 1u) {
      const bool dir_desc = (lane & size) != 0u;
      for (uint stride = size >> 1u; stride >= 1u; stride >>= 1u) {
        const uint oi = simd_shuffle_xor(id, ushort(stride));
        const float os = simd_shuffle_xor(my_s, ushort(stride));
        const bool upper = (lane & stride) != 0u;
        const bool swap = (upper != dir_desc) ? (oi > id) : (oi < id);
        if (swap) {
          id = oi;
          my_s = os;
        }
      }
    }
  }

  if (int(lane) < K) {
    const float scale = params[0];
    const size_t out_off = size_t(m) * size_t(K) + size_t(lane);
    ids[out_off] = id;
    const float w =
        (NORM && K > 1) ? metal::precise::divide(my_s, sum_s) : my_s;
    weights[out_off] = w * scale;
  }
}
)METAL"