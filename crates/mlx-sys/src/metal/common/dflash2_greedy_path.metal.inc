// Greedy DFlash predecessor walk. The selector has already built a score
// table [L,K,K]: at each position the previously-selected candidate chooses
// one of K rows, whose last maximum selects the next candidate. One GPU
// thread performs the short dependent walk (production L=7, K=16).
//
// This exactly mirrors the existing lazy graph's reversed argmax:
// - scan candidate slots from K-1 down to 0 with a strict `>` comparison;
// - ties, including +0/-0, keep the highest original index;
// - NaNs never win; if no value exceeds -INFINITY, slot K-1 remains selected.
//
// candidates: [1,L,K] int32, scores: [L,K,K] float32, path: [L] int32.
// Shape metadata supplies L and K. Grid/threadgroup: one thread.
R"(
  {
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
        if (value > best) {
          best = value;
          selected = candidate;
        }
      }
      path[position] = candidates[ulong(position) * K + selected];
      predecessor = selected;
    }
  }
)"
