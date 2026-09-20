// Scan body — one threadgroup per (row, shard). Each thread strided-scans
// its share of the shard keeping a register-resident sorted top-16, then
// sixteen rounds pop the group-wide best head into the partial (descending).
// Template params: T (input dtype), VOCAB, SHARDS.
R"(
  {
    constexpr uint K = 16, Threads = 256, SimdGroups = Threads / 32;
    const uint row = threadgroup_position_in_grid.y;
    const uint shard = threadgroup_position_in_grid.x;
    const uint shard_tokens = (VOCAB + SHARDS - 1) / SHARDS;
    const uint begin = min(shard * shard_tokens, (uint)VOCAB);
    const uint end = min(begin + shard_tokens, (uint)VOCAB);
    device const T *row_base = logits + (ulong)row * VOCAB;

    float values[K];
    uint ids[K];
    for (uint i = 0; i < K; ++i) {
      values[i] = -INFINITY;
      ids[i] = 0xffffffffu;
    }
    for (uint token = begin + thread_index_in_threadgroup; token < end;
         token += Threads) {
      float value = float(row_base[token]);
      if (topk16_beats(value, token, values[K - 1], ids[K - 1]))
        topk16_insert(values, ids, value, token);
    }

    // Sixteen rounds pop the group-wide best head; the simdgroup bests
    // alternate between two slots so one barrier per round suffices.
    const uint group = threadgroup_position_in_grid.y * SHARDS +
        threadgroup_position_in_grid.x;
    threadgroup float round_values[2][SimdGroups];
    threadgroup uint round_ids[2][SimdGroups];
    for (uint rank = 0; rank < K; ++rank) {
      float head_value = values[0];
      uint head_id = ids[0];
      float simd_value;
      uint simd_id;
      topk16_simd_best_head(head_value, head_id, simd_value, simd_id);
      uint slot = rank & 1;
      if (thread_index_in_simdgroup == 0) {
        round_values[slot][simdgroup_index_in_threadgroup] = simd_value;
        round_ids[slot][simdgroup_index_in_threadgroup] = simd_id;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float best = round_values[slot][0];
      uint best_id = round_ids[slot][0];
      for (uint other = 1; other < SimdGroups; ++other) {
        if (topk16_beats(round_values[slot][other], round_ids[slot][other],
                         best, best_id)) {
          best = round_values[slot][other];
          best_id = round_ids[slot][other];
        }
      }
      if (thread_index_in_threadgroup == rank) {
        partial_ids[group * K + rank] = best_id;
        partial_values[group * K + rank] = best;
      }
      if (head_value == best && head_id == best_id)
        topk16_pop<K>(values, ids);
    }
  }
)"
