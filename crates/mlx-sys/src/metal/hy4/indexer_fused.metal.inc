R"METAL(
template <
    bool PAGED,
    int INDEX_ROWS,
    typename QPtr,
    typename WPtr,
    typename KPtr,
    typename PagePtr,
    typename ParamsPtr>
inline void hy4_indexer_score_decode_impl(
    QPtr q,
    WPtr w,
    KPtr k_pool,
    PagePtr page_ids,
    ParamsPtr params,
    device float* out,
    threadgroup float4* q_tg,
    threadgroup float* w_tg,
    int H,
    uint s,
    uint g_tile,
    uint tid,
    uint lane,
    uint sg) {
  const uint S = params[0];
  const uint N_g = params[1];
  const uint start_g = params[2];
  if (s >= S) {
    return;
  }

  const size_t q_s_off = size_t(s) * size_t(H) * 128u;
  const uint q_vec_total = uint(H) * 32u;
  for (uint i = tid; i < q_vec_total; i += 256u) {
    const uint h = i >> 5;
    const uint c4 = i & 31u;
    const size_t off = q_s_off + size_t(h) * 128u + size_t(c4) * 4u;
    q_tg[i] = float4(q[off], q[off + 1u], q[off + 2u], q[off + 3u]);
  }
  if (tid < uint(H)) {
    w_tg[tid] = w[size_t(s) * size_t(H) + size_t(tid)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint g = g_tile * 8u + sg;
  if (g >= N_g) {
    return;
  }

  const uint g_abs = start_g + g;
  size_t k_row;
  if constexpr (PAGED) {
    const uint page_idx = g_abs / uint(INDEX_ROWS);
    const uint local_row = g_abs - page_idx * uint(INDEX_ROWS);
    const int phys_page = page_ids[page_idx];
    k_row = size_t(phys_page) * size_t(INDEX_ROWS) + size_t(local_row);
  } else {
    k_row = size_t(g_abs);
  }

  const size_t k_off = k_row * 128u + size_t(lane) * 4u;
  const float4 kv = float4(
      float(k_pool[k_off]),
      float(k_pool[k_off + 1u]),
      float(k_pool[k_off + 2u]),
      float(k_pool[k_off + 3u]));

  float acc = 0.0f;
  for (uint h = 0u; h < uint(H); ++h) {
    const float4 qv = q_tg[h * 32u + lane];
    const float d = simd_sum(metal::dot(qv, kv));
    if (lane == 0u) {
      acc = metal::fma(w_tg[h], metal::max(d, 0.0f), acc);
    }
  }
  if (lane == 0u) {
    out[size_t(s) * size_t(N_g) + size_t(g)] = acc;
  }
}

template <
    bool PAGED,
    int INDEX_ROWS,
    typename QPtr,
    typename WPtr,
    typename KPtr,
    typename PagePtr,
    typename ParamsPtr>
inline void hy4_indexer_score_prefill_impl(
    QPtr q,
    WPtr w,
    KPtr k_pool,
    PagePtr page_ids,
    ParamsPtr params,
    device float* out,
    threadgroup float4* k_tile_buf,
    threadgroup float4* q_tile_buf,
    threadgroup float* s_dot,
    threadgroup float* w_tg,
    int H,
    uint g_tile,
    uint s_tile,
    uint tid,
    uint lane,
    uint sg) {
  // Padded stride (132 = 128 + 4) stays 16-byte aligned and reduces TG bank
  // conflicts during transposed simdgroup_load.
  constexpr uint STRIDE = 132u;
  constexpr uint STRIDE4 = STRIDE / 4u;

  const uint S = params[0];
  const uint N_g = params[1];
  const uint start_g = params[2];
  const uint s0 = s_tile * 16u;
  const uint g0 = g_tile * 32u;
  if (s0 >= S || g0 >= N_g) {
    return;
  }

  if (tid < uint(H) * 16u) {
    const uint h = tid >> 4;
    const uint r = tid & 15u;
    const uint s_row = s0 + r;
    w_tg[tid] = (s_row < S) ? w[size_t(s_row) * size_t(H) + size_t(h)] : 0.0f;
  }

  for (uint i = tid; i < 32u * 32u; i += 256u) {
    const uint r = i >> 5;
    const uint c4 = i & 31u;
    const uint g = g0 + r;
    float4 kv = float4(0.0f);
    if (g < N_g) {
      const uint g_abs = start_g + g;
      size_t k_row;
      if constexpr (PAGED) {
        const uint page_idx = g_abs / uint(INDEX_ROWS);
        const uint local_row = g_abs - page_idx * uint(INDEX_ROWS);
        const int phys_page = page_ids[page_idx];
        k_row = size_t(phys_page) * size_t(INDEX_ROWS) + size_t(local_row);
      } else {
        k_row = size_t(g_abs);
      }
      const size_t k_off = k_row * 128u + size_t(c4) * 4u;
      kv = float4(
          float(k_pool[k_off]),
          float(k_pool[k_off + 1u]),
          float(k_pool[k_off + 2u]),
          float(k_pool[k_off + 3u]));
    }
    k_tile_buf[r * STRIDE4 + c4] = kv;
  }

  threadgroup float* k_tile = reinterpret_cast<threadgroup float*>(k_tile_buf);
  threadgroup float* q_tile = reinterpret_cast<threadgroup float*>(q_tile_buf);

  const uint sg_q = sg >> 2;
  const uint sg_k = sg & 3u;
  const uint r0 = lane >> 3;
  const uint c0 = lane & 7u;
  const uint r1 = r0 + 4u;
  const threadgroup float* q_sg = q_tile + (sg_q * 8u) * STRIDE;
  const threadgroup float* k_sg = k_tile + (sg_k * 8u) * STRIDE;
  threadgroup float* sd = s_dot + sg * 64u;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (uint h = 0u; h < uint(H); ++h) {
    for (uint i = tid; i < 16u * 32u; i += 256u) {
      const uint r = i >> 5;
      const uint c4 = i & 31u;
      const uint s_row = s0 + r;
      float4 qv = float4(0.0f);
      if (s_row < S) {
        const size_t q_off =
            (size_t(s_row) * size_t(H) + size_t(h)) * 128u + size_t(c4) * 4u;
        qv = float4(q[q_off], q[q_off + 1u], q[q_off + 2u], q[q_off + 3u]);
      }
      q_tile_buf[r * STRIDE4 + c4] = qv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 dot_h = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint d = 0u; d < 128u; d += 8u) {
      simdgroup_float8x8 mq;
      simdgroup_float8x8 mk;
      simdgroup_load(mq, q_sg + d, STRIDE);
      simdgroup_load(mk, k_sg + d, STRIDE, ulong2(0), true);
      simdgroup_multiply_accumulate(dot_h, mq, mk, dot_h);
    }
    simdgroup_store(dot_h, sd, 8u);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const float w0 = w_tg[h * 16u + sg_q * 8u + r0];
    const float w1 = w_tg[h * 16u + sg_q * 8u + r1];
    acc0 = metal::fma(w0, metal::max(sd[r0 * 8u + c0], 0.0f), acc0);
    acc1 = metal::fma(w1, metal::max(sd[r1 * 8u + c0], 0.0f), acc1);

    if (h + 1u < uint(H)) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  const uint g_col = g0 + sg_k * 8u + c0;
  if (g_col < N_g) {
    const uint s_row0 = s0 + sg_q * 8u + r0;
    if (s_row0 < S) {
      out[size_t(s_row0) * size_t(N_g) + size_t(g_col)] = acc0;
    }
    const uint s_row1 = s0 + sg_q * 8u + r1;
    if (s_row1 < S) {
      out[size_t(s_row1) * size_t(N_g) + size_t(g_col)] = acc1;
    }
  }
}

// Order-preserving IEEE-754 float -> uint32 map. Canonicalizes -0.0f to +0.0f
// so tie-breaking matches IEEE-754 equality (`-0.0f == +0.0f`); maps NaN to 0
// (strictly below -inf).
inline uint hy4_f2ui(float f) {
  if (metal::isnan(f)) {
    return 0u;
  }
  if (f == 0.0f) {
    return 0x80000000u;
  }
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <bool PAGED, typename PagePtr>
inline int hy4_map_cell(uint cell, PagePtr page_ids, uint block_size) {
  if constexpr (PAGED) {
    if (block_size == 32u) {
      return page_ids[cell >> 5] * 32 + int(cell & 31u);
    }
    return page_ids[cell / block_size] * int(block_size) +
        int(cell % block_size);
  } else {
    return int(cell);
  }
}

template <
    bool PAGED,
    typename ScoresPtr,
    typename VisPtr,
    typename TailStartPtr,
    typename TailLenPtr,
    typename PagePtr,
    typename ParamsPtr>
inline void hy4_indexer_topk_cells_impl(
    ScoresPtr scores,
    VisPtr visible_groups,
    TailStartPtr tail_starts,
    TailLenPtr tail_lens,
    PagePtr page_ids,
    ParamsPtr params,
    device int* out,
    threadgroup atomic_uint* hist,
    threadgroup uint* sg_eq,
    threadgroup uint* sg_keep,
    threadgroup uint* sel_groups,
    threadgroup uint* tg_state,
    uint S,
    uint N_g,
    uint r,
    uint tid,
    uint lane,
    uint sg) {
  if (r >= S) {
    return;
  }
  const uint n_top = params[0];
  const uint kpool = params[1];
  const uint kmax = params[2];
  const uint block_size = params[3];

  const int raw_vis = visible_groups[r];
  const uint vis = (raw_vis > 0) ? metal::min(uint(raw_vis), N_g) : 0u;
  const int raw_ts = tail_starts[r];
  const int raw_tl = tail_lens[r];
  const uint tail_start = (raw_ts > 0) ? uint(raw_ts) : 0u;
  const uint tail_len = (raw_tl > 0) ? uint(raw_tl) : 0u;
  const size_t out_base = size_t(r) * size_t(kmax);

  if (vis <= n_top) {
    const uint n_vis_cells = vis * kpool;
    const uint total_valid = n_vis_cells + tail_len;
    for (uint i = tid; i < kmax; i += 256u) {
      int slot = -1;
      if (i < n_vis_cells) {
        slot = hy4_map_cell<PAGED>(i, page_ids, block_size);
      } else if (i < total_valid) {
        slot = hy4_map_cell<PAGED>(
            tail_start + (i - n_vis_cells), page_ids, block_size);
      }
      out[out_base + size_t(i)] = slot;
    }
    return;
  }

  const size_t sc_base = size_t(r) * size_t(N_g);

  if (tid == 0u) {
    tg_state[0] = 0u;
    tg_state[1] = 0u;
    tg_state[2] = n_top;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int p = 3; p >= 0; --p) {
    const uint shift = uint(p) * 8u;
    const uint prefix = tg_state[0];
    const uint mask = tg_state[1];
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint g = tid; g < vis; g += 256u) {
      const uint key = hy4_f2ui(scores[sc_base + size_t(g)]);
      if ((key & mask) == prefix) {
        const uint bucket = (key >> shift) & 0xFFu;
        atomic_fetch_add_explicit(&hist[bucket], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
      const uint rem = tg_state[2];
      uint cum = 0u;
      uint chosen = 0u;
      for (int b = 255; b >= 0; --b) {
        const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (cum + c >= rem) {
          chosen = uint(b);
          break;
        }
        cum += c;
      }
      tg_state[0] = prefix | (chosen << shift);
      tg_state[1] = mask | (0xFFu << shift);
      tg_state[2] = rem - cum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint thr = tg_state[0];
  const uint need_ties = tg_state[2];

  uint seen_eq = 0u;
  uint out_cnt = 0u;
  for (uint base = 0u; base < vis; base += 256u) {
    const uint g = base + tid;
    uint is_gt = 0u;
    uint is_eq = 0u;
    if (g < vis) {
      const uint key = hy4_f2ui(scores[sc_base + size_t(g)]);
      is_gt = (key > thr) ? 1u : 0u;
      is_eq = (key == thr) ? 1u : 0u;
    }

    const uint eq_lane_ex = simd_prefix_exclusive_sum(is_eq);
    const uint eq_sg_tot = simd_sum(is_eq);
    if (lane == 0u) {
      sg_eq[sg] = eq_sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint eq_sg_ex = 0u;
    for (uint i = 0u; i < sg; ++i) {
      eq_sg_ex += sg_eq[i];
    }
    uint eq_chunk_tot = eq_sg_ex;
    for (uint i = sg; i < 8u; ++i) {
      eq_chunk_tot += sg_eq[i];
    }
    const uint eq_rank = seen_eq + eq_sg_ex + eq_lane_ex;
    seen_eq += eq_chunk_tot;

    const uint keep =
        (is_gt != 0u || (is_eq != 0u && eq_rank < need_ties)) ? 1u : 0u;

    const uint keep_lane_ex = simd_prefix_exclusive_sum(keep);
    const uint keep_sg_tot = simd_sum(keep);
    if (lane == 0u) {
      sg_keep[sg] = keep_sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint keep_sg_ex = 0u;
    for (uint i = 0u; i < sg; ++i) {
      keep_sg_ex += sg_keep[i];
    }
    uint keep_chunk_tot = keep_sg_ex;
    for (uint i = sg; i < 8u; ++i) {
      keep_chunk_tot += sg_keep[i];
    }
    if (keep != 0u) {
      const uint pos = out_cnt + keep_sg_ex + keep_lane_ex;
      if (pos < n_top) {
        sel_groups[pos] = g;
      }
    }
    out_cnt += keep_chunk_tot;
    if (out_cnt >= n_top) {
      break;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint n_sel_cells = n_top * kpool;
  const uint total_valid = n_sel_cells + tail_len;
  for (uint i = tid; i < kmax; i += 256u) {
    int slot = -1;
    if (i < n_sel_cells) {
      const uint g_id =
          (kpool == 4u) ? sel_groups[i >> 2] : sel_groups[i / kpool];
      const uint sub = (kpool == 4u) ? (i & 3u) : (i % kpool);
      slot = hy4_map_cell<PAGED>(g_id * kpool + sub, page_ids, block_size);
    } else if (i < total_valid) {
      slot = hy4_map_cell<PAGED>(
          tail_start + (i - n_sel_cells), page_ids, block_size);
    }
    out[out_base + size_t(i)] = slot;
  }
}
)METAL"
