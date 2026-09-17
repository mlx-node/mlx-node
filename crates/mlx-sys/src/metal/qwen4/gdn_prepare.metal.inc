// Wide GGUF preparation: one SIMD group per 128-channel vector. Preserve
// conv_sequence's F32 products, BF16 SiLU/QK boundaries and owned history.
R"(
  {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    const uint token = thread_position_in_grid.z;
    const uint vec = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    constexpr uint W = 10240;
    float values[4], squared = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint ch = vec * 128 + lane * 4 + i;
        float total = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            uint row = token + tap;
            float value = float(row < 3 ? history[row * W + ch] : qkv[(row - 3) * W + ch]);
            float product = value * conv[ch * 4 + tap];
            total = product + total;
        }
        T raw = T(total);
        T activated = raw * precise_sigmoid(raw);
        values[i] = float(activated);
        squared += values[i] * values[i];
        if (token == 0) {
            for (uint row = 0; row < 3; ++row) {
                uint source = TOKENS + row;
                next_history[row * W + ch] = source < 3
                    ? history[source * W + ch] : qkv[(source - 3) * W + ch];
            }
        }
    }
    if (vec < 32) {
        float divisor = metal::precise::sqrt(simd_sum(squared) + 1e-6f);
        for (uint i = 0; i < 4; ++i) {
            uint ch = (vec % 16) * 128 + lane * 4 + i;
            T normalized = T(values[i] / divisor);
            if (vec < 16) q[token * 2048 + ch] = normalized * T(0.08838834764831845f);
            else k[token * 2048 + ch] = normalized;
        }
    } else {
        for (uint i = 0; i < 4; ++i)
            v[token * 6144 + (vec - 32) * 128 + lane * 4 + i] = T(values[i]);
    }
    if (vec < 48 && lane == 0) {
        uint h = token * 48 + vec;
        float ax = float(a[h]) + dt[vec];
        float softplus = metal::max(ax, 0.0f) + log1p(metal::precise::exp(-metal::abs(ax)));
        decay[h] = metal::precise::exp(softplus * scale[vec]);
        beta[h] = precise_sigmoid(float(b[h]));
    }
  }
)"
