R"METAL(
template <int TILE_M, typename IdsPtr, typename ParamsPtr>
inline void hy4_moe_prefill_sort_single_impl(
    IdsPtr ids,
    ParamsPtr params,
    device uint* token_rows,
    device uint* tiles,
    device uint* inverse,
    device uint* sorted_slots,
    threadgroup atomic_uint* tg_counts,
    threadgroup uint* tg_offsets,
    threadgroup uint* tg_tile_offsets,
    threadgroup uint* sg_c,
    threadgroup uint* sg_t,
    uint tid,
    uint sg,
    uint lane) {
  constexpr uint TM = uint(TILE_M);
  constexpr uint TILE_SHIFT = (TILE_M == 64) ? 6u : 5u;
  constexpr uint TILE_MASK = TM - 1u;

  const uint R = params[0];
  const uint K = params[1];
  const uint E = params[2];
  const uint MAX_TILES = params[3];

  for (uint e = tid; e < E; e += 256u) {
    atomic_store_explicit(&tg_counts[e], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < R; i += 256u) {
    const uint e = ids[i];
    if (e < E) {
      atomic_fetch_add_explicit(&tg_counts[e], 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint carry_c = 0u;
  uint carry_t = 0u;
  for (uint base = 0u; base < E; base += 256u) {
    const uint e = base + tid;
    const uint c = (e < E)
        ? atomic_load_explicit(&tg_counts[e], memory_order_relaxed)
        : 0u;
    const uint tc = (c + TILE_MASK) >> TILE_SHIFT;

    const uint ex_c = simd_prefix_exclusive_sum(c);
    const uint tot_c = simd_sum(c);
    const uint ex_t = simd_prefix_exclusive_sum(tc);
    const uint tot_t = simd_sum(tc);
    if (lane == 0u) {
      sg_c[sg] = tot_c;
      sg_t[sg] = tot_t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_ex_c = 0u;
    uint sg_ex_t = 0u;
    for (uint s = 0u; s < sg; ++s) {
      sg_ex_c += sg_c[s];
      sg_ex_t += sg_t[s];
    }
    uint chunk_c = sg_ex_c;
    uint chunk_t = sg_ex_t;
    for (uint s = sg; s < 8u; ++s) {
      chunk_c += sg_c[s];
      chunk_t += sg_t[s];
    }
    if (e < E) {
      tg_offsets[e] = carry_c + sg_ex_c + ex_c;
      tg_tile_offsets[e] = carry_t + sg_ex_t + ex_t;
    }
    carry_c += chunk_c;
    carry_t += chunk_t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint total_tiles = carry_t;
  for (uint e = tid; e < E; e += 256u) {
    const uint c = atomic_load_explicit(&tg_counts[e], memory_order_relaxed);
    if (c > 0u) {
      const uint off = tg_offsets[e];
      const uint t0 = tg_tile_offsets[e];
      const uint nt = (c + TILE_MASK) >> TILE_SHIFT;
      for (uint j = 0u; j < nt; ++j) {
        const uint b = off + (j << TILE_SHIFT);
        const uint end = min(b + TM, off + c);
        const uint t = t0 + j;
        if (t < MAX_TILES) {
          tiles[3u * t + 0u] = b;
          tiles[3u * t + 1u] = end;
          tiles[3u * t + 2u] = e;
        }
      }
    }
  }
  for (uint t = total_tiles + tid; t < MAX_TILES; t += 256u) {
    tiles[3u * t + 0u] = 0u;
    tiles[3u * t + 1u] = 0u;
    tiles[3u * t + 2u] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint e = tid; e < E; e += 256u) {
    atomic_store_explicit(&tg_counts[e], tg_offsets[e], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < R; i += 256u) {
    const uint e = ids[i];
    if (e < E) {
      const uint pos =
          atomic_fetch_add_explicit(&tg_counts[e], 1u, memory_order_relaxed);
      token_rows[pos] = i / K;
      sorted_slots[pos] = e;
      inverse[i] = pos;
    } else {
      inverse[i] = R;
    }
  }
  for (uint pos = carry_c + tid; pos < R; pos += 256u) {
    token_rows[pos] = 0u;
    sorted_slots[pos] = 0u;
  }
}

template <typename IdsPtr, typename ParamsPtr>
inline void hy4_moe_prefill_sort_hist_impl(
    IdsPtr ids,
    ParamsPtr params,
    device uint* block_counts,
    threadgroup atomic_uint* tg_hist,
    uint block_id,
    uint tid) {
  const uint R = params[0];
  const uint E = params[2];
  const uint chunk = params[4];

  for (uint e = tid; e < E; e += 256u) {
    atomic_store_explicit(&tg_hist[e], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint lo = block_id * chunk;
  const uint hi = min(lo + chunk, R);
  for (uint i = lo + tid; i < hi; i += 256u) {
    const uint e = ids[i];
    if (e < E) {
      atomic_fetch_add_explicit(&tg_hist[e], 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device uint* dst = block_counts + size_t(block_id) * size_t(E);
  for (uint e = tid; e < E; e += 256u) {
    dst[e] = atomic_load_explicit(&tg_hist[e], memory_order_relaxed);
  }
}

template <int TILE_M, typename CountsPtr, typename ParamsPtr>
inline void hy4_moe_prefill_sort_scan_tiles_impl(
    CountsPtr block_counts,
    ParamsPtr params,
    device uint* block_offsets,
    device uint* tiles,
    device uint* meta,
    threadgroup uint* sg_c,
    threadgroup uint* sg_t,
    uint tid,
    uint sg,
    uint lane) {
  constexpr uint TM = uint(TILE_M);
  constexpr uint TILE_SHIFT = (TILE_M == 64) ? 6u : 5u;
  constexpr uint TILE_MASK = TM - 1u;

  const uint E = params[1];
  const uint MAX_TILES = params[2];
  const uint N_TGS = params[3];

  uint carry_c = 0u;
  uint carry_t = 0u;
  for (uint base = 0u; base < E; base += 256u) {
    const uint e = base + tid;
    uint c = 0u;
    if (e < E) {
      for (uint b = 0u; b < N_TGS; ++b) {
        c += block_counts[size_t(b) * size_t(E) + size_t(e)];
      }
    }
    const uint tc = (c + TILE_MASK) >> TILE_SHIFT;

    const uint ex_c = simd_prefix_exclusive_sum(c);
    const uint tot_c = simd_sum(c);
    const uint ex_t = simd_prefix_exclusive_sum(tc);
    const uint tot_t = simd_sum(tc);
    if (lane == 0u) {
      sg_c[sg] = tot_c;
      sg_t[sg] = tot_t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_ex_c = 0u;
    uint sg_ex_t = 0u;
    for (uint s = 0u; s < sg; ++s) {
      sg_ex_c += sg_c[s];
      sg_ex_t += sg_t[s];
    }
    uint chunk_c = sg_ex_c;
    uint chunk_t = sg_ex_t;
    for (uint s = sg; s < 8u; ++s) {
      chunk_c += sg_c[s];
      chunk_t += sg_t[s];
    }

    if (e < E) {
      const uint off = carry_c + sg_ex_c + ex_c;
      uint run = off;
      for (uint b = 0u; b < N_TGS; ++b) {
        const size_t idx = size_t(b) * size_t(E) + size_t(e);
        block_offsets[idx] = run;
        run += block_counts[idx];
      }
      if (c > 0u) {
        const uint t0 = carry_t + sg_ex_t + ex_t;
        for (uint j = 0u; j < tc; ++j) {
          const uint b_row = off + (j << TILE_SHIFT);
          const uint end_row = min(b_row + TM, off + c);
          const uint t = t0 + j;
          if (t < MAX_TILES) {
            tiles[3u * t + 0u] = b_row;
            tiles[3u * t + 1u] = end_row;
            tiles[3u * t + 2u] = e;
          }
        }
      }
    }
    carry_c += chunk_c;
    carry_t += chunk_t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint t = carry_t + tid; t < MAX_TILES; t += 256u) {
    tiles[3u * t + 0u] = 0u;
    tiles[3u * t + 1u] = 0u;
    tiles[3u * t + 2u] = 0u;
  }
  if (tid == 0u) {
    meta[0] = carry_c;
  }
}

template <
    typename IdsPtr,
    typename OffsetsPtr,
    typename MetaPtr,
    typename ParamsPtr>
inline void hy4_moe_prefill_sort_scatter_impl(
    IdsPtr ids,
    OffsetsPtr block_offsets,
    MetaPtr meta,
    ParamsPtr params,
    device uint* token_rows,
    device uint* inverse,
    device uint* sorted_slots,
    threadgroup atomic_uint* tg_cursor,
    uint block_id,
    uint tid) {
  const uint R = params[0];
  const uint K = params[1];
  const uint E = params[2];
  const uint N_TGS = params[3];
  const uint chunk = params[4];

  const size_t boff = size_t(block_id) * size_t(E);
  for (uint e = tid; e < E; e += 256u) {
    atomic_store_explicit(
        &tg_cursor[e], block_offsets[boff + size_t(e)], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint lo = block_id * chunk;
  const uint hi = min(lo + chunk, R);
  for (uint i = lo + tid; i < hi; i += 256u) {
    const uint e = ids[i];
    if (e < E) {
      const uint pos =
          atomic_fetch_add_explicit(&tg_cursor[e], 1u, memory_order_relaxed);
      token_rows[pos] = i / K;
      sorted_slots[pos] = e;
      inverse[i] = pos;
    } else {
      inverse[i] = R;
    }
  }

  const uint total_valid = meta[0];
  const uint gtid = block_id * 256u + tid;
  const uint gstride = N_TGS * 256u;
  for (uint pos = total_valid + gtid; pos < R; pos += gstride) {
    token_rows[pos] = 0u;
    sorted_slots[pos] = 0u;
  }
}
)METAL"