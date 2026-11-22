R"METAL(
template <int VP>
METAL_FUNC float hy4_qdot4_reg(
    const thread uint16_t* ws,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  for (int i = 0; i < (VP / 4); ++i) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

template <bool FAST, bool DEDUP, typename RowIdsPtr, typename SlotsPtr>
METAL_FUNC void hy4_kq_gate_up_swiglu_impl(
    const device float* x,
    RowIdsPtr row_ids,
    SlotsPtr slots,
    const device uint* wg,
    const device uint8_t* sg,
    const device float16_t* bg,
    const device uint* wu,
    const device uint8_t* su,
    const device float16_t* bu,
    float lim,
    device float* out,
    int ROWS,
    int N_ASSIGN,
    int K,
    int M,
    int S,
    uint assign,
    uint tile,
    uint simd,
    uint lane) {
  constexpr int PACK = get_pack_factor<4, 32>();
  constexpr int BP = get_bytes_per_pack<4, 32>();
  constexpr int PACKS = FAST ? 2 : 1;
  constexpr int VP = PACK * PACKS;
  constexpr int BLOCK = VP * 32;
  constexpr int SCALE_STEP = 32 / VP;

  const int first = int(tile) * 8 + int(simd) * 4;
  if (assign >= uint(N_ASSIGN) || first >= M) return;

  const uint slot = slots[assign];

  if constexpr (DEDUP) {
    if (slot < uint(S)) {
      for (uint b = 0; b < assign; ++b) {
        if (slots[b] == slot) return;
      }

      constexpr int MAX_MATCH = 8;
      constexpr int W16 = VP / 4;
      const size_t row_stride_w = size_t(K) * size_t(BP) / size_t(PACK);
      const size_t row_stride_g = size_t(K) / 32;
      const size_t base_row = size_t(slot) * size_t(M) + size_t(first);

      uint cursor = assign;
      while (cursor < uint(N_ASSIGN)) {
        uint match_assign[MAX_MATCH];
        uint match_row[MAX_MATCH];
        int n_match = 0;
        for (; cursor < uint(N_ASSIGN) && n_match < MAX_MATCH; ++cursor) {
          if (slots[cursor] == slot) {
            match_assign[n_match] = cursor;
            match_row[n_match] = row_ids[cursor];
            ++n_match;
          }
        }
        if (n_match == 0) break;

        float g[MAX_MATCH][4];
        float u[MAX_MATCH][4];
        for (int m = 0; m < n_match; ++m) {
          for (int r = 0; r < 4; ++r) {
            g[m][r] = 0.0f;
            u[m][r] = 0.0f;
          }
        }

        auto gw = reinterpret_cast<const device uint8_t*>(wg) +
            base_row * row_stride_w + size_t(lane) * size_t(PACKS) * size_t(BP);
        auto uw = reinterpret_cast<const device uint8_t*>(wu) +
            base_row * row_stride_w + size_t(lane) * size_t(PACKS) * size_t(BP);
        auto gs = KQScales<float, 4, 8, true, KQ_LINEAR, 0>(
            sg, bg, base_row * row_stride_g + size_t(lane) / size_t(SCALE_STEP));
        auto us = KQScales<float, 4, 8, true, KQ_LINEAR, 0>(
            su, bu, base_row * row_stride_g + size_t(lane) / size_t(SCALE_STEP));

        for (int k = 0; k < K; k += BLOCK) {
          uint16_t gw_reg[4][W16];
          uint16_t uw_reg[4][W16];
          float g_sc[4], g_bi[4], u_sc[4], u_bi[4];
          for (int r = 0; r < 4; ++r) {
            if (FAST || first + r < M) {
              const device uint16_t* gws =
                  reinterpret_cast<const device uint16_t*>(gw + size_t(r) * row_stride_w);
              const device uint16_t* uws =
                  reinterpret_cast<const device uint16_t*>(uw + size_t(r) * row_stride_w);
              for (int i = 0; i < W16; ++i) {
                gw_reg[r][i] = gws[i];
                uw_reg[r][i] = uws[i];
              }
              gs.at(size_t(r) * row_stride_g, g_sc[r], g_bi[r]);
              us.at(size_t(r) * row_stride_g, u_sc[r], u_bi[r]);
            }
          }

          for (int m = 0; m < n_match; ++m) {
            const uint rid = match_row[m];
            if (rid < uint(ROWS)) {
              const device float* input =
                  x + size_t(rid) * size_t(K) + size_t(k) + size_t(lane) * size_t(VP);
              float values[VP];
              const float sum = load_vector<float, float, VP, 4>(input, values);
              for (int r = 0; r < 4; ++r) {
                if (FAST || first + r < M) {
                  g[m][r] += hy4_qdot4_reg<VP>(gw_reg[r], values, g_sc[r], g_bi[r], sum);
                  u[m][r] += hy4_qdot4_reg<VP>(uw_reg[r], values, u_sc[r], u_bi[r], sum);
                }
              }
            }
          }

          gw += size_t(BLOCK) * size_t(BP) / size_t(PACK);
          uw += size_t(BLOCK) * size_t(BP) / size_t(PACK);
          gs.advance(BLOCK / 32);
          us.advance(BLOCK / 32);
        }

        for (int m = 0; m < n_match; ++m) {
          const uint dst_assign = match_assign[m];
          for (int r = 0; r < 4; ++r) {
            if (FAST || first + r < M) {
              float gv = simd_sum(g[m][r]);
              float uv = simd_sum(u[m][r]);
              if (lane == 0) {
                if (lim > 0.0f) {
                  gv = metal::min(gv, lim);
                  uv = metal::clamp(uv, -lim, lim);
                }
                const float sig = 1.0f / (1.0f + metal::exp(-gv));
                out[size_t(dst_assign) * size_t(M) + size_t(first + r)] = (sig * gv) * uv;
              }
            }
          }
        }
      }
      return;
    }
  }

  const uint row_id = row_ids[assign];
  float g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float u[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (slot < uint(S) && row_id < uint(ROWS)) {
    const size_t row_stride_w = size_t(K) * size_t(BP) / size_t(PACK);
    const size_t row_stride_g = size_t(K) / 32;
    const size_t base_row = size_t(slot) * size_t(M) + size_t(first);
    auto gw = reinterpret_cast<const device uint8_t*>(wg) +
        base_row * row_stride_w + size_t(lane) * size_t(PACKS) * size_t(BP);
    auto uw = reinterpret_cast<const device uint8_t*>(wu) +
        base_row * row_stride_w + size_t(lane) * size_t(PACKS) * size_t(BP);
    auto gs = KQScales<float, 4, 8, true, KQ_LINEAR, 0>(
        sg, bg, base_row * row_stride_g + size_t(lane) / size_t(SCALE_STEP));
    auto us = KQScales<float, 4, 8, true, KQ_LINEAR, 0>(
        su, bu, base_row * row_stride_g + size_t(lane) / size_t(SCALE_STEP));
    const device float* input = x + size_t(row_id) * size_t(K) + size_t(lane) * size_t(VP);

    for (int k = 0; k < K; k += BLOCK) {
      float values[VP];
      const float sum = load_vector<float, float, VP, 4>(input, values);
      for (int r = 0; r < 4; ++r) {
        if (FAST || first + r < M) {
          float scale, bias;
          gs.at(size_t(r) * row_stride_g, scale, bias);
          g[r] += qdot<float, VP, 4, false>(gw + size_t(r) * row_stride_w, values, scale, bias, sum);
          us.at(size_t(r) * row_stride_g, scale, bias);
          u[r] += qdot<float, VP, 4, false>(uw + size_t(r) * row_stride_w, values, scale, bias, sum);
        }
      }
      gw += size_t(BLOCK) * size_t(BP) / size_t(PACK);
      uw += size_t(BLOCK) * size_t(BP) / size_t(PACK);
      gs.advance(BLOCK / 32);
      us.advance(BLOCK / 32);
      input += BLOCK;
    }
  }

  for (int r = 0; r < 4; ++r) {
    if (FAST || first + r < M) {
      float gv = simd_sum(g[r]);
      float uv = simd_sum(u[r]);
      if (lane == 0) {
        if (lim > 0.0f) {
          gv = metal::min(gv, lim);
          uv = metal::clamp(uv, -lim, lim);
        }
        const float sig = 1.0f / (1.0f + metal::exp(-gv));
        out[size_t(assign) * size_t(M) + size_t(first + r)] = (sig * gv) * uv;
      }
    }
  }
}

// Q6_K keeps PACKS == 2 (VP == 8) because kquant.h's 6-bit qdot advances
// pointers by 4*i / 3*i inside the loop, which is only valid for i in {0, 1}.
template <
    int BITS,
    int GS,
    int SR,
    bool HM,
    bool FAST,
    int RPS,
    int KSG,
    typename SlotsPtr,
    typename WeightsPtr>
METAL_FUNC void hy4_kq_down_combine_impl(
    const device float* hidden,
    SlotsPtr slots,
    WeightsPtr weights,
    const device uint* wd,
    const device uint8_t* sd,
    const device float16_t* bd,
    device float* out,
    threadgroup float* products,
    int ROWS,
    int TOP,
    int M,
    int H,
    int S,
    uint row,
    uint tile,
    uint simd,
    uint lane) {
  constexpr int PACK = get_pack_factor<BITS, 32>();
  constexpr int BP = get_bytes_per_pack<BITS, 32>();
  constexpr int PACKS = (BITS == 6) ? 2 : (FAST ? 2 : 1);
  constexpr int VP = PACK * PACKS;
  constexpr int BLOCK = VP * 32;
  constexpr int SCALE_STEP = GS / VP;

  const int first = int(tile) * RPS;
  if (row >= uint(ROWS) || first >= H) return;

  const size_t row_stride_w = size_t(M) * size_t(BP) / size_t(PACK);
  const size_t row_stride_g = size_t(M) / size_t(GS);

  for (int slot_idx = int(simd); slot_idx < TOP; slot_idx += KSG) {
    const size_t assign = size_t(row) * size_t(TOP) + size_t(slot_idx);
    const uint slot = slots[assign];
    float acc[RPS];
    for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;

    if (slot < uint(S)) {
      const size_t base_row = size_t(slot) * size_t(H) + size_t(first);
      auto dw = reinterpret_cast<const device uint8_t*>(wd) +
          base_row * row_stride_w + size_t(lane) * size_t(PACKS) * size_t(BP);
      auto ds = KQScales<float, BITS, SR, HM, KQ_LINEAR, 0>(
          sd, bd, base_row * row_stride_g + size_t(lane) / size_t(SCALE_STEP));
      const device float* input = hidden + assign * size_t(M) + size_t(lane) * size_t(VP);

      for (int k = 0; k < M; k += BLOCK) {
        float values[VP];
        const float sum = load_vector<float, float, VP, BITS>(input, values);
        for (int r = 0; r < RPS; ++r) {
          if (FAST || first + r < H) {
            float scale, bias;
            ds.at(size_t(r) * row_stride_g, scale, bias);
            acc[r] += qdot<float, VP, BITS, false>(
                dw + size_t(r) * row_stride_w, values, scale, bias, sum);
          }
        }
        dw += size_t(BLOCK) * size_t(BP) / size_t(PACK);
        ds.advance(BLOCK / GS);
        input += BLOCK;
      }
    }

    const float weight = weights[assign];
    for (int r = 0; r < RPS; ++r) {
      if (FAST || first + r < H) {
        const float val = simd_sum(acc[r]);
        if (lane == 0) {
          products[slot_idx * RPS + r] = val * weight;
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Match MLX col_reduce_small's 8-lane modulo fold so full-hit combine and
  // split-step weighted+sum agree bit-for-bit.
  if (simd == 0 && lane < uint(RPS)) {
    if (FAST || first + int(lane) < H) {
      const int lanes = TOP < 8 ? TOP : 8;
      float totals[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int slot_idx = 0; slot_idx < TOP; ++slot_idx) {
        totals[slot_idx % 8] += products[slot_idx * RPS + int(lane)];
      }
      float total = totals[0];
      for (int i = 1; i < lanes; ++i) {
        total += totals[i];
      }
      out[size_t(row) * size_t(H) + size_t(first) + size_t(lane)] = total;
    }
  }
}

template <int BITS, int GS, int SR, bool HM, bool FAST, int RPS>
METAL_FUNC void hy4_kq_down_weighted_impl(
    const device float* hidden,
    uint slot,
    float weight,
    const device uint* wd,
    const device uint8_t* sd,
    const device float16_t* bd,
    device float* out,
    int N_ASSIGN,
    int M,
    int H,
    int S,
    uint assign,
    uint tile,
    uint simd,
    uint lane) {
  constexpr int PACK = get_pack_factor<BITS, 32>();
  constexpr int BP = get_bytes_per_pack<BITS, 32>();
  constexpr int PACKS = (BITS == 6) ? 2 : (FAST ? 2 : 1);
  constexpr int VP = PACK * PACKS;
  constexpr int BLOCK = VP * 32;
  constexpr int SCALE_STEP = GS / VP;

  const int first = int(tile) * (2 * RPS) + int(simd) * RPS;
  if (assign >= uint(N_ASSIGN) || first >= H) return;

  float acc[RPS];
  for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;

  if (slot < uint(S)) {
    const size_t row_stride_w = size_t(M) * size_t(BP) / size_t(PACK);
    const size_t row_stride_g = size_t(M) / size_t(GS);
    const size_t base_row = size_t(slot) * size_t(H) + size_t(first);
    auto dw = reinterpret_cast<const device uint8_t*>(wd) +
        base_row * row_stride_w + size_t(lane) * size_t(PACKS) * size_t(BP);
    auto ds = KQScales<float, BITS, SR, HM, KQ_LINEAR, 0>(
        sd, bd, base_row * row_stride_g + size_t(lane) / size_t(SCALE_STEP));
    const device float* input = hidden + size_t(assign) * size_t(M) + size_t(lane) * size_t(VP);

    for (int k = 0; k < M; k += BLOCK) {
      float values[VP];
      const float sum = load_vector<float, float, VP, BITS>(input, values);
      for (int r = 0; r < RPS; ++r) {
        if (FAST || first + r < H) {
          float scale, bias;
          ds.at(size_t(r) * row_stride_g, scale, bias);
          acc[r] += qdot<float, VP, BITS, false>(
              dw + size_t(r) * row_stride_w, values, scale, bias, sum);
        }
      }
      dw += size_t(BLOCK) * size_t(BP) / size_t(PACK);
      ds.advance(BLOCK / GS);
      input += BLOCK;
    }
  }

  for (int r = 0; r < RPS; ++r) {
    if (FAST || first + r < H) {
      const float val = simd_sum(acc[r]);
      if (lane == 0) {
        out[size_t(assign) * size_t(H) + size_t(first + r)] = val * weight;
      }
    }
  }
}
)METAL"
