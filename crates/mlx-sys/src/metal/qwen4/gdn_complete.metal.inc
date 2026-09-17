// Singleton GGUF GDN: one threadgroup per value head, four value rows per
// SIMD group. The modulo key-head mapping and BF16 boundaries follow the local
// decoder. Inputs and outputs never alias, including convolution history.
R"(
    const uint hv = threadgroup_position_in_grid.z;
    const uint hk = hv % 16;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr uint W = 10240;
    threadgroup float q_shared[128], k_shared[128], v_shared[128];
    threadgroup float gates[2];
    threadgroup T y_shared[128];
    float4 prefetched;
    if constexpr (PREFETCH)
        prefetched = *reinterpret_cast<const device float4*>(state + (hv * 128 + sg * 4) * 128 + 4 * lane);
    if (sg < 3) {
        uint vec = sg == 0 ? hk : (sg == 1 ? 16 + hk : 32 + hv);
        float values[4];
        float total = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            uint ch = vec * 128 + lane * 4 + i;
            float c = 0.0f;
            {
                // The eager convolution materializes F32 products before the
                // column sum. Do not contract these products into FMAs.
                #pragma clang fp contract(off)
                #pragma clang fp reassociate(off)
                for (uint j = 0; j < KC; ++j) {
                    float x = float(j + 1 == KC ? qkv[ch] : history[j * W + ch]);
                    float product = x * float(conv[ch * KC + j]);
                    c = product + c;
                }
            }
            T raw = T(c);
            T sig = precise_sigmoid(raw);
            T activated = raw * sig;
            values[i] = float(activated);
            total += values[i] * values[i];
            if (sg == 2 || hv < 16) {
                for (uint j = 0; j + 1 < KC; ++j) {
                    next_history[j * W + ch] = j + 2 == KC ? qkv[ch] : history[(j + 1) * W + ch];
                }
            }
        }
        if (sg < 2) {
            total = simd_sum(total);
            float divisor = metal::precise::sqrt(total + 1e-6f);
            for (uint i = 0; i < 4; ++i) {
                T normalized = T(values[i] / divisor);
                if (sg == 0) q_shared[lane * 4 + i] = float(T(normalized * T(0.08838834764831845f)));
                else k_shared[lane * 4 + i] = float(normalized);
            }
        } else {
            for (uint i = 0; i < 4; ++i) v_shared[lane * 4 + i] = values[i];
        }
    }
    if (sg == 0 && lane == 0) {
        float ax = float(a[hv]) + float(dt[hv]);
        float softplus = metal::max(ax, 0.0f) + log1p(metal::precise::exp(-metal::abs(ax)));
        gates[0] = metal::precise::exp(softplus * float(scale[hv]));
        gates[1] = precise_sigmoid(float(b[hv]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // TrackFastGDNDecode: retain local FP32 arithmetic and modulo heads.
    if constexpr (PREFETCH) {
    const float4 local_q = *reinterpret_cast<const threadgroup float4*>(q_shared + 4 * lane);
    const float4 local_k = *reinterpret_cast<const threadgroup float4*>(k_shared + 4 * lane);
    const float decay = gates[0], beta = gates[1];
    for (uint r = 0; r < 4; ++r) {
        const uint row = (hv * 128 + sg * 4 + r) * 128 + 4 * lane;
        const float4 current = prefetched;
        if (r + 1 < 4) prefetched = *reinterpret_cast<const device float4*>(state + row + 128);
        float s[4], memory = 0;
        for (uint i = 0; i < 4; ++i) {
            s[i] = current[i] * decay;
            memory += s[i] * local_k[i];
        }
        memory = simd_sum(memory);
        const float delta = (v_shared[sg * 4 + r] - memory) * beta;
        float result = 0;
        for (uint i = 0; i < 4; ++i) {
            s[i] = s[i] + local_k[i] * delta;
            result += s[i] * local_q[i];
        }
        *reinterpret_cast<device float4*>(next + row) = float4(s[0], s[1], s[2], s[3]);
        result = simd_sum(result);
        if (lane == 0) y_shared[sg * 4 + r] = T(result);
    }
    } else {
    // Use the same four-row per-lane accumulation as gated_delta_step_4vcol.
    float s[4][4];
    float memory[4] = {0,0,0,0};
    for (uint i = 0; i < 4; ++i) {
        uint k = lane * 4 + i;
        for (uint r = 0; r < 4; ++r) {
            uint v = sg * 4 + r;
            s[r][i] = state[(hv * 128 + v) * 128 + k] * gates[0];
            memory[r] += s[r][i] * k_shared[k];
        }
    }
    float delta[4];
    for (uint r = 0; r < 4; ++r) {
        memory[r] = simd_sum(memory[r]);
        delta[r] = (v_shared[sg * 4 + r] - memory[r]) * gates[1];
    }
    float result[4] = {0,0,0,0};
    for (uint i = 0; i < 4; ++i) {
        uint k = lane * 4 + i;
        for (uint r = 0; r < 4; ++r) {
            uint v = sg * 4 + r;
            s[r][i] = s[r][i] + k_shared[k] * delta[r];
            result[r] += s[r][i] * q_shared[k];
            next[(hv * 128 + v) * 128 + k] = s[r][i];
        }
    }
    for (uint r = 0; r < 4; ++r) {
        result[r] = simd_sum(result[r]);
        if (lane == 0) y_shared[sg * 4 + r] = T(result[r]);
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float values[4], total = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            values[i] = float(y_shared[lane * 4 + i]);
            total += values[i] * values[i];
        }
        total = simd_sum(total);
        float divisor = metal::precise::sqrt(total / 128.0f + eps);
        for (uint i = 0; i < 4; ++i) {
            uint d = lane * 4 + i;
            T normalized = T((values[i] / divisor) * float(norm[d]));
            out[hv * 128 + d] = T(float(normalized) * precise_sigmoid(float(z[hv * 128 + d])));
        }
    }
)"
