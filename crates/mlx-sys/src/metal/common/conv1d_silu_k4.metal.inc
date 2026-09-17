// One thread per channel and token. History is a separate, owned allocation;
// short chunks may retain part of the previous history without retaining x.
R"(
    const uint ch = thread_position_in_grid.x;
    const uint token = thread_position_in_grid.y;
    if (ch >= W || token >= TOKENS) return;
    float total = 0.0f;
    {
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        for (uint tap = 0; tap < 4; ++tap) {
            uint row = token + tap;
            float value = float(row < 3 ? history[row * W + ch] : x[(row - 3) * W + ch]);
            float product = value * weight[ch * 4 + tap];
            total = product + total;
        }
    }
    T raw = T(total);
    T sig = precise_sigmoid(raw);
    out[token * W + ch] = raw * sig;
    // Every output history row must be written even for a one-token chunk.
    if (token == 0) {
        for (uint row = 0; row < 3; ++row) {
            uint source = TOKENS + row;
            next_history[row * W + ch] = source < 3
                ? history[source * W + ch] : x[(source - 3) * W + ch];
        }
    }
)"
