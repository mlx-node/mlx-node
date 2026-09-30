// Wide GGUF preparation: one SIMD group per 128-channel vector. Preserve
// conv_sequence's F32 products, BF16 SiLU/QK boundaries and owned history.
// qkv/a/b/history are read through their strides (split views, no copies);
// axis 0 of qkv/a/b has size 1. conv/scale/dt are flat row-contiguous.
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
            float value = float(row < 3
                ? history[row * history_strides[0] + ch * history_strides[1]]
                : qkv[(row - 3) * qkv_strides[1] + ch * qkv_strides[2]]);
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
                    ? history[source * history_strides[0] + ch * history_strides[1]]
                    : qkv[(source - 3) * qkv_strides[1] + ch * qkv_strides[2]];
            }
        }
    }
    if (vec < 32) {
        // MEAN_EPS: RMSNorm semantics — eps inside the mean, i.e.
        // x/sqrt(mean(x^2)+eps) == x/sqrt((sum+d*eps)/d); after the q/k
        // scale fold the divisor is sqrt(sum + d*eps). Off: L2-style
        // x/sqrt(sum + eps) (qwen4's math::l2).
        float divisor = metal::precise::sqrt(
            simd_sum(squared) + (MEAN_EPS ? 128.0f * 1e-6f : 1e-6f));
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
        float ax = float(a[token * a_strides[1] + vec * a_strides[2]]) + dt[vec];
        float softplus = metal::max(ax, 0.0f) + log1p(metal::precise::exp(-metal::abs(ax)));
        decay[h] = metal::precise::exp(softplus * scale[vec]);
        beta[h] = metal::remove_reference_t<decltype(beta[0])>(
            precise_sigmoid(float(b[token * b_strides[1] + vec * b_strides[2]])));
    }
  }
)"
