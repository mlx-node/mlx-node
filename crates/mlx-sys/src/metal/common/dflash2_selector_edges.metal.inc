// Adapted from Splash runtime/metal/kernels/decode/sampling.metal,
// draft_select_edges (Apache-2.0; see THIRD_PARTY_NOTICES).
// Fixed K=16, rank=256. One 256-thread group per proposal position; a
// SIMD group scores eight successors of one predecessor together. Products,
// accumulation, SIMD reduction and unary addition remain FP32, including
// predecessor*projected_hidden, unlike the established native BF16 graph.
// The anchor's identical predecessor rows are all initialized, even though
// the walk only reads row zero. Candidate ids use Splash's bounded clamp.
R"(
  {
    constexpr uint K = 16, Rank = 256, Dims = Rank / 32;
    const uint position = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint vocabulary = predecessor_shape[0];
    const uint tasks = (position > 0 ? K : 1) * 2;
    for (uint task = sg; task < tasks; task += 8) {
      const uint previous = task / 2;
      const uint first = (task % 2) * 8;
      const uint predecessor_id = min(uint(position > 0
          ? candidates[(position - 1) * K + previous] : anchor[0]), vocabulary - 1);
      float context[Dims];
      float next[8][Dims];
      for (uint i = 0; i < Dims; ++i) {
        const uint dim = lane + i * 32;
        context[i] = float(predecessor[ulong(predecessor_id) * Rank + dim]) *
                     float(hidden[ulong(position) * Rank + dim]);
      }
      for (uint j = 0; j < 8; ++j) {
        const uint id = min(uint(candidates[position * K + first + j]), vocabulary - 1);
        for (uint i = 0; i < Dims; ++i)
          next[j][i] = float(successor[ulong(id) * Rank + lane + i * 32]);
      }
      for (uint j = 0; j < 8; ++j) {
        float score = 0.0f;
        for (uint i = 0; i < Dims; ++i)
          score += context[i] * next[j][i];
        score = simd_sum(score);
        if (lane == 0) {
          const float combined = score + unary[position * K + first + j];
          if (position > 0) {
            scores[(ulong(position) * K + previous) * K + first + j] = combined;
          } else {
            for (uint p = 0; p < K; ++p)
              scores[p * K + first + j] = combined;
          }
        }
      }
    }
  }
)"
