// Fused GDN gating kernel: computes beta and g from raw inputs.
//
//   beta = sigmoid(b)
//   g    = -exp(a_log[head_idx]) * softplus(a + dt_bias[head_idx])
//
// where softplus(x) = x > 20 ? x : log(1 + exp(x))  (numerically stable)
//
// Layout:
//   b:      [total_elements] (InT)  -- raw logit for beta
//   a:      [total_elements] (InT)  -- raw input for gate
//   a_log:  [num_heads]      (f32)  -- per-head log-scale (always f32)
//   dt_bias:[num_heads]      (f32)  -- per-head bias (always f32)
//   beta:   [total_elements] (InT)  -- output: sigmoid(b)
//   g_out:  [total_elements] (f32)  -- output: gate value (f32 for log-space accumulation)
//
// Grid:  (total_elements,)
// Block: (256,) or similar
//
// Template param: InT (bfloat16_t or half)
R"(
    auto idx = thread_position_in_grid.x;
    if (idx >= total_elements) return;

    int head_idx = idx % num_heads;

    // beta = sigmoid(b)
    float b_val = static_cast<float>(b[idx]);
    float beta_val = 1.0f / (1.0f + exp(-b_val));

    // g = -exp(a_log[head]) * softplus(a + dt_bias[head])
    float a_val = static_cast<float>(a[idx]);
    float a_log_val = a_log[head_idx];
    float dt_bias_val = dt_bias[head_idx];

    float sp_input = a_val + dt_bias_val;
    float softplus_val = (sp_input > 20.0f) ? sp_input : log(1.0f + exp(sp_input));
    float g_val = -exp(a_log_val) * softplus_val;

    beta_out[idx] = static_cast<InT>(beta_val);
    g_out[idx] = g_val;
)"
