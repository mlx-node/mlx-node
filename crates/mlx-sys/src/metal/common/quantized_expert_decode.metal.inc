R"metal(
// Same lane ownership and K walk as kquant_qmv_fast_impl. Both projections
// reuse the activation load; each retains its independent F32 accumulator.
template <typename T, int BITS, int H, int M, int E, int TOP, bool LANES>
METAL_FUNC void quantized_expert_gate_up(
    const device T* x, const device uint* ids,
    const device uint* wg, const device uchar* sg, const device half* bg,
    const device uint* wu, const device uchar* su, const device half* bu,
    device float* out, uint assignment, uint tile, uint simd, uint lane) {
  constexpr int VP = 2 * get_pack_factor<BITS,32>();
  constexpr int BP = get_bytes_per_pack<BITS,32>();
  constexpr int PACK = get_pack_factor<BITS,32>();
  constexpr int BLOCK = VP * 32;
  uint expert = ids[assignment];
  uint first = tile * 8 + simd * 4;
  float g[4] = {0}, u[4] = {0};
  if (expert < E) {
    size_t row = size_t(expert) * M + first;
    auto gw = (const device uchar*)wg + row * (H * BP / PACK) + lane * 2 * BP;
    auto uw = (const device uchar*)wu + row * (H * BP / PACK) + lane * 2 * BP;
    auto gs = KQScales<float,BITS,8,true>(sg, bg, row * (H / 32) + lane / (32 / VP));
    auto us = KQScales<float,BITS,8,true>(su, bu, row * (H / 32) + lane / (32 / VP));
    auto input = x + size_t(assignment / TOP) * H + lane * VP;
    for (int k = 0; k < H; k += BLOCK) {
      float values[VP];
      float sum = load_vector<T,float,VP,BITS>(input, values);
      for (int r = 0; r < 4; ++r) {
        float scale, bias;
        gs.at(r * (H / 32), scale, bias);
        g[r] += qdot<float,VP,BITS,false>(gw + r * (H * BP / PACK), values, scale, bias, sum);
        us.at(r * (H / 32), scale, bias);
        u[r] += qdot<float,VP,BITS,false>(uw + r * (H * BP / PACK), values, scale, bias, sum);
      }
      gw += BLOCK * BP / PACK; uw += BLOCK * BP / PACK;
      gs.advance(BLOCK / 32); us.advance(BLOCK / 32); input += BLOCK;
    }
  }
  if constexpr (LANES) {
    // TrackFastMoE gate/up staging (8981cef5): reductions broadcast each
    // complete row to all lanes. Assign one identical epilogue to each lane.
    for (int r = 0; r < 4; ++r) { g[r] = simd_sum(g[r]); u[r] = simd_sum(u[r]); }
    if (lane < 4) {
      T gate = T(g[lane]), up = T(u[lane]);
      T silu = T(precise_sigmoid(gate) * gate);
      out[size_t(assignment) * M + first + lane] = float(T(silu * up));
    }
  } else {
  for (int r = 0; r < 4; ++r) {
    T gate = T(simd_sum(g[r])), up = T(simd_sum(u[r]));
    if (lane == 0) {
      T silu = T(precise_sigmoid(gate) * gate);
      out[size_t(assignment) * M + first + r] = float(T(silu * up));
    }
  }
  }
}

// Affine5/8 qdot with FP16 companions and a group size of 32. Experts are split
// across SIMD groups, then folded in the original eight-lane reduction order.
template <typename T, typename P, int BITS, int H, int M, int E, int TOP, int RPS, int KSG, bool LANES>
METAL_FUNC void affine_expert_down_combine(
    const device float* x, const device uint* ids, const device P* scores,
    const device uint* w, const device half* scales, const device half* biases,
    device P* out, threadgroup P* products,
    uint token, uint tile, uint simd, uint lane) {
  constexpr int VP = get_pack_factor<BITS,32>();
  constexpr int BP = get_bytes_per_pack<BITS,32>();
  constexpr int BLOCK = VP * 32;
  uint first = tile * RPS;
  for (int slot = simd; slot < TOP; slot += KSG) {
    uint assignment = token * TOP + slot, expert = ids[assignment];
    float result[RPS] = {0};
    if (expert < E) {
      for (int base = 0; base < M; base += BLOCK) {
        int col = base + lane * VP;
        if (col < M) {
          float values[VP];
          // Gate/up writes the BF16-rounded activation directly as F32,
          // matching the old affine activation promotion and vector sums.
          float sum = load_vector<float,float,VP,BITS>(x + size_t(assignment) * M + col, values);
          for (int r = 0; r < RPS; ++r) {
            size_t row = size_t(expert) * H + first + r;
            auto codes = (const device uchar*)w + row * (M * BP / VP) + col * BP / VP;
            size_t side = row * (M / 32) + col / 32;
            result[r] += qdot<float,VP,BITS,false>(codes, values, float(scales[side]), float(biases[side]), sum);
          }
        }
      }
    }
    if constexpr (LANES) {
      for (int r = 0; r < RPS; ++r) result[r] = simd_sum(result[r]);
      if (lane < RPS) {
        volatile float product = float(P(T(result[lane]))) * float(scores[assignment]);
        products[slot * RPS + lane] = P(product);
      }
    } else {
    for (int r = 0; r < RPS; ++r) {
      T value = T(simd_sum(result[r]));
      if (lane == 0) {
        volatile float product = float(P(value)) * float(scores[assignment]);
        products[slot * RPS + r] = P(product);
      }
    }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0 && lane < RPS) {
    P totals[8] = {0};
    for (int slot = 0; slot < TOP; ++slot)
      totals[slot % 8] = P(products[slot * RPS + lane] + totals[slot % 8]);
    P total = totals[0];
    for (int i = 1; i < 8; ++i) total = P(totals[i] + total);
    out[size_t(token) * H + first + lane] = total;
  }
}
)metal"
