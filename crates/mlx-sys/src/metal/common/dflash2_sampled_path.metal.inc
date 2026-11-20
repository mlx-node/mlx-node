// Sampled DFlash predecessor walk, from Splash draft_select_dflash
// (runtime/metal/kernels/decode/sampling.metal, Apache-2.0). At each position
// the previously drawn candidate picks one of K score rows; the draw takes
// softmax(row / temperature) and the host's uniform for that position: the
// first candidate whose cumulative probability exceeds it, else the last one
// with positive probability (rounding). q holds the probabilities drawn from.
// -inf scores have no mass. A row with a NaN or +inf score, or without a
// finite one, writes path -1 and q 0 from there on (the host walk's rule).
//
// candidates: [1,L,K] int32, scores: [L,K,K] float32, uniforms: [L] float32,
// temperature: [1] float32; path: [L] int32, q: [L,K] float32. One thread.
R"(
  {
    if (thread_position_in_grid.x != 0) return;
    const uint L = candidates_shape[1];
    const uint K = candidates_shape[2];
    const float scale = max(temperature[0], FLT_MIN);
    uint predecessor = 0;
    bool failed = false;
    for (uint position = 0; position < L; ++position) {
      const ulong row = (ulong(position) * K + predecessor) * K;
      const ulong out = ulong(position) * K;
      float maximum = -INFINITY;
      bool invalid = false;
      for (uint candidate = 0; candidate < K; ++candidate) {
        const float value = scores[row + candidate];
        if (isfinite(value))
          maximum = max(maximum, value);
        else if (isnan(value) || value > 0.0f)
          invalid = true;
      }
      if (failed || invalid || !isfinite(maximum)) {
        failed = true;
        for (uint candidate = 0; candidate < K; ++candidate)
          q[out + candidate] = 0.0f;
        path[position] = -1;
        continue;
      }
      float sum = 0.0f;
      for (uint candidate = 0; candidate < K; ++candidate) {
        const float value = scores[row + candidate];
        const float weight = isfinite(value) ? exp((value - maximum) / scale) : 0.0f;
        q[out + candidate] = weight;
        sum += weight;
      }
      const float uniform = uniforms[position];
      float cumulative = 0.0f;
      uint selected = K;
      uint last = 0;
      for (uint candidate = 0; candidate < K; ++candidate) {
        const float probability = q[out + candidate] / sum;
        q[out + candidate] = probability;
        if (probability <= 0.0f)
          continue;
        cumulative += probability;
        last = candidate;
        if (selected == K && uniform < cumulative)
          selected = candidate;
      }
      if (selected == K)
        selected = last;
      path[position] = candidates[out + selected];
      predecessor = selected;
    }
  }
)"
