R"metal(
// Adapted from TrackP12Prefill.sortedCombineKernel (Layr Labs, MIT).

      uint i=thread_position_in_grid.x;
      if (i>=TOKENS*H) return;
      uint token=i/H, col=i%H;
      constexpr int LANES=TOP<8?TOP:8;
      P totals[LANES];
      for (int lane=0; lane<LANES; ++lane) totals[lane]=P(0);
      for (int slot=0; slot<TOP; ++slot) {
        uint row=uint(inverse[token*TOP+slot]);
        // The volatile product keeps the original multiply-round-add boundary,
        // including FP32 products, instead of allowing a fused multiply-add.
        volatile float unrounded=row<ROWS ? float(P(values[size_t(row)*H+col]))*float(P(scores[token*TOP+slot])) : 0.0f;
        P product=P(unrounded);
        totals[slot%LANES]=P(product+totals[slot%LANES]);
      }
      P total=totals[0];
      for (int lane=1; lane<LANES; ++lane) total=P(totals[lane]+total);
      T sg = sigmoid_table[as_type<ushort>(gate[token])];
      T sh = T(sg * shared[i]);
      out[i] = T(T(total) + sh);

)metal"
