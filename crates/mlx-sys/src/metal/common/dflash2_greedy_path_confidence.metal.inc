// Confidence formula from Splash sampling.metal (Apache-2.0); path preserves MLX tie semantics.
// Greedy DFlash predecessor walk. The selector has already built a score
// table [L,K,K]: at each position the previously-selected candidate chooses
// one of K rows, whose last maximum selects the next candidate. One GPU
// thread performs the short dependent walk (production L=7, K=16).
//
// This exactly mirrors the existing lazy graph's reversed argmax, using MLX's
// arg_reduce rule (upstream a124ac096):
// - scan candidate slots from K-1 down to 0 with a strict `>` comparison;
// - ties, including +0/-0, keep the highest original index;
// - a NaN beats every number and the first NaN met wins, i.e. the highest
//   original NaN index; if no value exceeds -INFINITY, slot K-1 stays.
//
// candidates: [1,L,K] int32, scores: [L,K,K] float32, path: [L] int32.
// Shape metadata supplies L and K. Grid/threadgroup: one thread.
R"(
  {
    // The builtin also keeps MLX JIT multi-output signatures well-formed when
    // input shape metadata adds buffers before the two output parameters.
    if (thread_position_in_grid.x != 0) return;
    const uint L = candidates_shape[1];
    const uint K = candidates_shape[2];
    uint predecessor = 0;
    for (uint position = 0; position < L; ++position) {
      const ulong row = (ulong(position) * K + predecessor) * K;
      uint selected = K - 1;
      float best = -INFINITY;
      for (uint reverse = 0; reverse < K; ++reverse) {
        const uint candidate = K - 1 - reverse;
        const float value = scores[row + candidate];
        if ((!isnan(best) && isnan(value)) || value > best) {
          best = value;
          selected = candidate;
        }
      }
      path[position] = candidates[ulong(position) * K + selected];
      float normalizer = 0.0f;
      for (uint candidate = 0; candidate < K; ++candidate)
        normalizer += exp(scores[row + candidate] - best);
      confidence[position] = isfinite(best) && isfinite(normalizer) && normalizer > 0.0f
        ? 1.0f / normalizer : 0.0f;
      predecessor = selected;
    }
  }
)"
