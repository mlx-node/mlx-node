// Helpers shared by the dflash2 top-16 scan/merge kernels. Algorithm
// adapted from Splash's draft_select_top16_sharded / top16_merge_shards
// (Apache-2.0 — runtime/metal/kernels/decode/sampling.metal): the total
// order is (value desc, token id asc) so ties resolve identically at every
// level regardless of the thread/shard partition.
R"(
inline bool topk16_beats(float value, uint token, float other, uint other_token) {
  return value > other || (value == other && token < other_token);
}

// Register-resident sorted top-16 insert (descending). Caller pre-checks
// against the last slot; the unrolled shift keeps every index static.
inline void topk16_insert(thread float (&values)[16], thread uint (&ids)[16],
                          float value, uint token) {
#pragma clang loop unroll(full)
  for (int slot = 15; slot > 0; --slot) {
    bool here = topk16_beats(value, token, values[slot], ids[slot]);
    bool above = topk16_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  bool top = topk16_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

template <int Count>
inline void topk16_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (int slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

// The simdgroup's best list head: (value desc, id asc), so ties and the
// empty sentinel (-inf, ~0u) resolve the same way everywhere.
inline void topk16_simd_best_head(float value, uint token, thread float &best,
                                  thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}
)"
