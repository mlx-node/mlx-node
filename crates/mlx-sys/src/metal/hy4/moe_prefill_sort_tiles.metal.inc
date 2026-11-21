R"METAL(
const uint R = params[0];
const uint K = params[1];
const uint E = params[2];
const uint MAX_TILES = params[3];
const uint tid = thread_index_in_threadgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;

threadgroup atomic_uint tg_counts[1024];
threadgroup uint tg_offsets[1024];
threadgroup uint tg_tile_offsets[1024];
threadgroup uint sg_c[8];
threadgroup uint sg_t[8];

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
  const uint tc = (c + 31u) >> 5;

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
    const uint nt = (c + 31u) >> 5;
    for (uint j = 0u; j < nt; ++j) {
      const uint b = off + (j << 5);
      const uint end = min(b + 32u, off + c);
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
)METAL"