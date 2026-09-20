// Merge body — one simdgroup per row. Lane `l` holds SHARDS*16/32
// consecutive partial entries (all from one shard, still sorted). Sixteen
// rounds pop the simdgroup-wide best; writing to slot (K-1-rank) emits the
// merged list ASCENDING by value, matching argpartition's last-K slice.
// Template params: SHARDS.
R"(
  {
    constexpr uint K = 16;
    constexpr uint Entries = SHARDS * K / 32;
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint origin = row * SHARDS * K + lane * Entries;
    float values[Entries];
    uint ids[Entries];
    for (uint i = 0; i < Entries; ++i) {
      values[i] = partial_values[origin + i];
      ids[i] = partial_ids[origin + i];
    }
    for (uint rank = 0; rank < K; ++rank) {
      float best;
      uint best_id;
      topk16_simd_best_head(values[0], ids[0], best, best_id);
      if (lane == rank) {
        out_ids[row * K + (K - 1 - rank)] = int(best_id);
        out_values[row * K + (K - 1 - rank)] = best;
      }
      if (values[0] == best && ids[0] == best_id)
        topk16_pop<Entries>(values, ids);
    }
  }
)"
