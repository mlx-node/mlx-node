#include "mlx_common.h"

extern "C" {

// Fused SwiGLU MLP forward pass
// Combines 5 operations into 1 FFI call:
// 1. gate = x @ w_gate.T
// 2. up = x @ w_up.T
// 3. gate_act = silu(gate) = gate * sigmoid(gate)
// 4. gated = gate_act * up
// 5. output = gated @ w_down.T
mlx_array* mlx_swiglu_mlp_forward(mlx_array* x_handle,
                                   mlx_array* w_gate_handle,
                                   mlx_array* w_up_handle,
                                   mlx_array* w_down_handle) {
  auto x = reinterpret_cast<array*>(x_handle);
  auto w_gate = reinterpret_cast<array*>(w_gate_handle);
  auto w_up = reinterpret_cast<array*>(w_up_handle);
  auto w_down = reinterpret_cast<array*>(w_down_handle);

  // Transpose weights: [out, in] -> [in, out] for matmul
  auto w_gate_t = transpose(*w_gate, {1, 0});
  auto w_up_t = transpose(*w_up, {1, 0});
  auto w_down_t = transpose(*w_down, {1, 0});

  // gate = x @ w_gate.T
  auto gate = matmul(*x, w_gate_t);

  // up = x @ w_up.T
  auto up = matmul(*x, w_up_t);

  // silu(gate) = gate * sigmoid(gate)
  auto gate_act = gate * sigmoid(gate);

  // gated = gate_act * up
  auto gated = gate_act * up;

  // output = gated @ w_down.T
  auto output = matmul(gated, w_down_t);

  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused Multi-Head Attention forward pass (without KV cache)
// Reduces ~20 FFI calls to 1 for attention computation
// Parameters:
//   x: input [B, L, D]
//   w_q, w_k, w_v, w_o: projection weights [out, in]
//   q_norm_w, k_norm_w: optional QK norm weights (can be nullptr)
//   n_heads, n_kv_heads, head_dim: attention configuration
//   scale: attention scale factor (usually 1/sqrt(head_dim))
//   rope_base, rope_dims: RoPE parameters
//   qk_norm_eps: epsilon for QK normalization
//   use_causal: whether to use causal masking
//   rope_offset: position offset for RoPE (for KV cache support)
mlx_array* mlx_fused_attention_forward(
    mlx_array* x_handle,
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* w_o_handle,
    mlx_array* q_norm_w_handle,  // Can be nullptr if no QK norm
    mlx_array* k_norm_w_handle,  // Can be nullptr if no QK norm
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale,
    float rope_base,
    int rope_dims,
    float qk_norm_eps,
    bool use_causal,
    int rope_offset) {

  auto x = reinterpret_cast<array*>(x_handle);
  auto w_q = reinterpret_cast<array*>(w_q_handle);
  auto w_k = reinterpret_cast<array*>(w_k_handle);
  auto w_v = reinterpret_cast<array*>(w_v_handle);
  auto w_o = reinterpret_cast<array*>(w_o_handle);

  // Get input shape (cast to int for MLX Shape)
  int batch = static_cast<int>(x->shape()[0]);
  int seq_len = static_cast<int>(x->shape()[1]);

  // 1. Project Q/K/V (transpose weights then matmul)
  auto w_q_t = transpose(*w_q, {1, 0});
  auto w_k_t = transpose(*w_k, {1, 0});
  auto w_v_t = transpose(*w_v, {1, 0});
  auto w_o_t = transpose(*w_o, {1, 0});

  auto queries = matmul(*x, w_q_t);  // [B, L, n_heads * head_dim]
  auto keys = matmul(*x, w_k_t);     // [B, L, n_kv_heads * head_dim]
  auto values = matmul(*x, w_v_t);   // [B, L, n_kv_heads * head_dim]

  // 2. Reshape to multi-head format [B, L, n_heads, head_dim]
  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  // 3. Apply QK normalization if weights provided (before transpose!)
  if (q_norm_w_handle) {
    auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w_handle) {
    auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  // 4. Transpose to attention layout [B, n_heads, L, head_dim]
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  // 5. Apply RoPE
  bool traditional = false;  // MLX uses non-traditional by default
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});

  // 6. Scaled dot-product attention
  std::string mask_mode = use_causal && seq_len > 1 ? "causal" : "";
  auto output = fast::scaled_dot_product_attention(queries, keys, values, scale, mask_mode, {}, std::nullopt, {});
  output.eval();  // Force GPU sync after expensive SDPA to prevent timeout

  // 7. Transpose back to [B, L, n_heads, head_dim]
  output = transpose(output, {0, 2, 1, 3});

  // 8. Reshape to [B, L, n_heads * head_dim]
  output = reshape(output, {batch, seq_len, n_heads * head_dim});

  // 9. Output projection
  output = matmul(output, w_o_t);

  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused attention with KV cache support
// Returns output array and updates cached_keys/cached_values in-place
void mlx_fused_attention_forward_cached(
    mlx_array* x_handle,
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* w_o_handle,
    mlx_array* q_norm_w_handle,
    mlx_array* k_norm_w_handle,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale,
    float rope_base,
    int rope_dims,
    float qk_norm_eps,
    bool use_causal,
    // KV cache (in/out)
    mlx_array** cached_keys_ptr,
    mlx_array** cached_values_ptr,
    int cache_offset,
    // Output
    mlx_array** output_ptr) {

  auto x = reinterpret_cast<array*>(x_handle);
  auto w_q = reinterpret_cast<array*>(w_q_handle);
  auto w_k = reinterpret_cast<array*>(w_k_handle);
  auto w_v = reinterpret_cast<array*>(w_v_handle);
  auto w_o = reinterpret_cast<array*>(w_o_handle);

  // Get input shape (cast to int for MLX Shape)
  int batch = static_cast<int>(x->shape()[0]);
  int seq_len = static_cast<int>(x->shape()[1]);

  // 1. Project Q/K/V
  auto w_q_t = transpose(*w_q, {1, 0});
  auto w_k_t = transpose(*w_k, {1, 0});
  auto w_v_t = transpose(*w_v, {1, 0});
  auto w_o_t = transpose(*w_o, {1, 0});

  auto queries = matmul(*x, w_q_t);
  auto keys = matmul(*x, w_k_t);
  auto values = matmul(*x, w_v_t);

  // 2. Reshape
  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  // 3. QK normalization
  if (q_norm_w_handle) {
    auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w_handle) {
    auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  // 4. Transpose to [B, n_heads, L, head_dim]
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  // 5. Apply RoPE
  bool traditional = false;
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});

  // 6. Update KV cache
  if (*cached_keys_ptr && *cached_values_ptr) {
    auto cached_keys = reinterpret_cast<array*>(*cached_keys_ptr);
    auto cached_values = reinterpret_cast<array*>(*cached_values_ptr);

    // Concatenate new keys/values with cache
    keys = concatenate({*cached_keys, keys}, 2);
    values = concatenate({*cached_values, values}, 2);

    // Update cache pointers
    delete cached_keys;
    delete cached_values;
    *cached_keys_ptr = reinterpret_cast<mlx_array*>(new array(keys));
    *cached_values_ptr = reinterpret_cast<mlx_array*>(new array(values));
  } else {
    // Initialize cache
    *cached_keys_ptr = reinterpret_cast<mlx_array*>(new array(keys));
    *cached_values_ptr = reinterpret_cast<mlx_array*>(new array(values));
  }

  // 7. Scaled dot-product attention
  // For generation (seq_len == 1), no mask needed
  // For prefill (seq_len > 1), use causal mask
  int kv_len = static_cast<int>(keys.shape()[2]);
  std::string mask_mode = "";
  if (use_causal && seq_len > 1 && seq_len == kv_len) {
    mask_mode = "causal";
  }
  auto output = fast::scaled_dot_product_attention(queries, keys, values, scale, mask_mode, {}, std::nullopt, {});
  output.eval();  // Force GPU sync after expensive SDPA to prevent timeout

  // 8. Transpose back
  output = transpose(output, {0, 2, 1, 3});

  // 9. Reshape
  output = reshape(output, {batch, seq_len, n_heads * head_dim});

  // 10. Output projection
  output = matmul(output, w_o_t);

  *output_ptr = reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused Transformer Block forward (without KV cache)
// Combines: norm -> attention -> residual -> norm -> mlp -> residual
// Reduces ~40 FFI calls to 1 per block
mlx_array* mlx_fused_transformer_block_forward(
    mlx_array* x_handle,
    // Layer norm weights
    mlx_array* input_norm_w_handle,
    mlx_array* post_attn_norm_w_handle,
    // Attention weights
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* w_o_handle,
    mlx_array* q_norm_w_handle,  // Can be nullptr
    mlx_array* k_norm_w_handle,  // Can be nullptr
    // MLP weights
    mlx_array* w_gate_handle,
    mlx_array* w_up_handle,
    mlx_array* w_down_handle,
    // Config
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float attn_scale,
    float rope_base,
    int rope_dims,
    float norm_eps,
    float qk_norm_eps,
    bool use_causal,
    int rope_offset) {

  auto x = reinterpret_cast<array*>(x_handle);
  auto input_norm_w = reinterpret_cast<array*>(input_norm_w_handle);
  auto post_attn_norm_w = reinterpret_cast<array*>(post_attn_norm_w_handle);
  auto w_q = reinterpret_cast<array*>(w_q_handle);
  auto w_k = reinterpret_cast<array*>(w_k_handle);
  auto w_v = reinterpret_cast<array*>(w_v_handle);
  auto w_o = reinterpret_cast<array*>(w_o_handle);
  auto w_gate = reinterpret_cast<array*>(w_gate_handle);
  auto w_up = reinterpret_cast<array*>(w_up_handle);
  auto w_down = reinterpret_cast<array*>(w_down_handle);

  // Get input shape
  int batch = static_cast<int>(x->shape()[0]);
  int seq_len = static_cast<int>(x->shape()[1]);

  // === Part 1: Self-Attention ===

  // 1. Input layer norm
  auto normed = fast::rms_norm(*x, std::optional<array>(*input_norm_w), norm_eps, {});

  // 2. Q/K/V projections
  auto w_q_t = transpose(*w_q, {1, 0});
  auto w_k_t = transpose(*w_k, {1, 0});
  auto w_v_t = transpose(*w_v, {1, 0});
  auto w_o_t = transpose(*w_o, {1, 0});

  auto queries = matmul(normed, w_q_t);
  auto keys = matmul(normed, w_k_t);
  auto values = matmul(normed, w_v_t);

  // 3. Reshape to multi-head format
  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  // 4. QK normalization
  if (q_norm_w_handle) {
    auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w_handle) {
    auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  // 5. Transpose to attention layout
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  // 6. Apply RoPE
  bool traditional = false;
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});

  // 7. Scaled dot-product attention
  std::string mask_mode = use_causal && seq_len > 1 ? "causal" : "";
  auto attn_output = fast::scaled_dot_product_attention(queries, keys, values, attn_scale, mask_mode, {}, std::nullopt, {});
  attn_output.eval();  // Force GPU sync after SDPA to prevent timeout

  // 8. Transpose back and reshape
  attn_output = transpose(attn_output, {0, 2, 1, 3});
  attn_output = reshape(attn_output, {batch, seq_len, n_heads * head_dim});

  // 9. Output projection
  attn_output = matmul(attn_output, w_o_t);

  // 10. Attention residual
  auto h = *x + attn_output;

  // === Part 2: MLP ===

  // 11. Post-attention layer norm
  auto mlp_input = fast::rms_norm(h, std::optional<array>(*post_attn_norm_w), norm_eps, {});

  // 12. MLP (SwiGLU)
  auto w_gate_t = transpose(*w_gate, {1, 0});
  auto w_up_t = transpose(*w_up, {1, 0});
  auto w_down_t = transpose(*w_down, {1, 0});

  auto gate = matmul(mlp_input, w_gate_t);
  auto up = matmul(mlp_input, w_up_t);
  auto gate_act = gate * sigmoid(gate);  // SiLU
  auto gated = gate_act * up;
  auto mlp_output = matmul(gated, w_down_t);

  // 13. MLP residual
  auto output = h + mlp_output;

  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused Transformer Block with KV cache support
void mlx_fused_transformer_block_forward_cached(
    mlx_array* x_handle,
    // Layer norm weights
    mlx_array* input_norm_w_handle,
    mlx_array* post_attn_norm_w_handle,
    // Attention weights
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* w_o_handle,
    mlx_array* q_norm_w_handle,
    mlx_array* k_norm_w_handle,
    // MLP weights
    mlx_array* w_gate_handle,
    mlx_array* w_up_handle,
    mlx_array* w_down_handle,
    // Config
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float attn_scale,
    float rope_base,
    int rope_dims,
    float norm_eps,
    float qk_norm_eps,
    bool use_causal,
    // KV cache (in/out)
    mlx_array** cached_keys_ptr,
    mlx_array** cached_values_ptr,
    int cache_offset,
    // Output
    mlx_array** output_ptr) {

  auto x = reinterpret_cast<array*>(x_handle);
  auto input_norm_w = reinterpret_cast<array*>(input_norm_w_handle);
  auto post_attn_norm_w = reinterpret_cast<array*>(post_attn_norm_w_handle);
  auto w_q = reinterpret_cast<array*>(w_q_handle);
  auto w_k = reinterpret_cast<array*>(w_k_handle);
  auto w_v = reinterpret_cast<array*>(w_v_handle);
  auto w_o = reinterpret_cast<array*>(w_o_handle);
  auto w_gate = reinterpret_cast<array*>(w_gate_handle);
  auto w_up = reinterpret_cast<array*>(w_up_handle);
  auto w_down = reinterpret_cast<array*>(w_down_handle);

  int batch = static_cast<int>(x->shape()[0]);
  int seq_len = static_cast<int>(x->shape()[1]);

  // === Part 1: Self-Attention ===

  auto normed = fast::rms_norm(*x, std::optional<array>(*input_norm_w), norm_eps, {});

  auto w_q_t = transpose(*w_q, {1, 0});
  auto w_k_t = transpose(*w_k, {1, 0});
  auto w_v_t = transpose(*w_v, {1, 0});
  auto w_o_t = transpose(*w_o, {1, 0});

  auto queries = matmul(normed, w_q_t);
  auto keys = matmul(normed, w_k_t);
  auto values = matmul(normed, w_v_t);

  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  if (q_norm_w_handle) {
    auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w_handle) {
    auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  bool traditional = false;
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});

  // Update KV cache
  if (*cached_keys_ptr && *cached_values_ptr) {
    auto cached_keys = reinterpret_cast<array*>(*cached_keys_ptr);
    auto cached_values = reinterpret_cast<array*>(*cached_values_ptr);
    keys = concatenate({*cached_keys, keys}, 2);
    values = concatenate({*cached_values, values}, 2);
    delete cached_keys;
    delete cached_values;
    *cached_keys_ptr = reinterpret_cast<mlx_array*>(new array(keys));
    *cached_values_ptr = reinterpret_cast<mlx_array*>(new array(values));
  } else {
    *cached_keys_ptr = reinterpret_cast<mlx_array*>(new array(keys));
    *cached_values_ptr = reinterpret_cast<mlx_array*>(new array(values));
  }

  int kv_len = static_cast<int>(keys.shape()[2]);
  std::string mask_mode = "";
  if (use_causal && seq_len > 1 && seq_len == kv_len) {
    mask_mode = "causal";
  }
  auto attn_output = fast::scaled_dot_product_attention(queries, keys, values, attn_scale, mask_mode, {}, std::nullopt, {});
  attn_output.eval();  // Force GPU sync after SDPA to prevent timeout

  attn_output = transpose(attn_output, {0, 2, 1, 3});
  attn_output = reshape(attn_output, {batch, seq_len, n_heads * head_dim});
  attn_output = matmul(attn_output, w_o_t);

  auto h = *x + attn_output;

  // === Part 2: MLP ===

  auto mlp_input = fast::rms_norm(h, std::optional<array>(*post_attn_norm_w), norm_eps, {});

  auto w_gate_t = transpose(*w_gate, {1, 0});
  auto w_up_t = transpose(*w_up, {1, 0});
  auto w_down_t = transpose(*w_down, {1, 0});

  auto gate = matmul(mlp_input, w_gate_t);
  auto up = matmul(mlp_input, w_up_t);
  auto gate_act = gate * sigmoid(gate);
  auto gated = gate_act * up;
  auto mlp_output = matmul(gated, w_down_t);

  auto output = h + mlp_output;

  *output_ptr = reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused Q/K/V projection with RoPE for cached attention
// Returns Q, K, V in attention layout (B, n_heads, L, head_dim) with RoPE applied
// This fuses: projection -> reshape -> qk_norm -> transpose -> RoPE
void mlx_fused_attention_qkv(
    mlx_array* x_handle,
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* q_norm_w_handle,  // Can be null
    mlx_array* k_norm_w_handle,  // Can be null
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_base,
    int rope_dims,
    float qk_norm_eps,
    int rope_offset,
    mlx_array** q_out,
    mlx_array** k_out,
    mlx_array** v_out
) {
    try {
        auto x = reinterpret_cast<array*>(x_handle);
        auto w_q = reinterpret_cast<array*>(w_q_handle);
        auto w_k = reinterpret_cast<array*>(w_k_handle);
        auto w_v = reinterpret_cast<array*>(w_v_handle);

        int batch = static_cast<int>(x->shape()[0]);
        int seq_len = static_cast<int>(x->shape()[1]);

        // Transpose weights for matmul: (hidden, proj) -> (proj, hidden)
        auto w_q_t = transpose(*w_q);
        auto w_k_t = transpose(*w_k);
        auto w_v_t = transpose(*w_v);

        // 1. Q/K/V projections
        auto queries = matmul(*x, w_q_t);  // (B, L, n_heads * head_dim)
        auto keys = matmul(*x, w_k_t);     // (B, L, n_kv_heads * head_dim)
        auto values = matmul(*x, w_v_t);   // (B, L, n_kv_heads * head_dim)

        // 2. Reshape to multi-head format: (B, L, n_heads, head_dim)
        queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
        keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
        values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

        // 3. Apply QK normalization BEFORE transpose (matching transformers)
        if (q_norm_w_handle) {
            auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
            queries = mlx::core::fast::rms_norm(queries, *q_norm_w, qk_norm_eps);
        }
        if (k_norm_w_handle) {
            auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
            keys = mlx::core::fast::rms_norm(keys, *k_norm_w, qk_norm_eps);
        }

        // 4. Transpose to attention layout: (B, n_heads, L, head_dim)
        queries = transpose(queries, {0, 2, 1, 3});
        keys = transpose(keys, {0, 2, 1, 3});
        values = transpose(values, {0, 2, 1, 3});

        // 5. Apply RoPE
        queries = mlx::core::fast::rope(queries, rope_dims, false, rope_base, 1.0f, rope_offset);
        keys = mlx::core::fast::rope(keys, rope_dims, false, rope_base, 1.0f, rope_offset);

        *q_out = reinterpret_cast<mlx_array*>(new array(std::move(queries)));
        *k_out = reinterpret_cast<mlx_array*>(new array(std::move(keys)));
        *v_out = reinterpret_cast<mlx_array*>(new array(std::move(values)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_fused_attention_qkv error: " << e.what() << std::endl;
        *q_out = nullptr;
        *k_out = nullptr;
        *v_out = nullptr;
    }
}

// Fused SDPA + output projection for cached attention
// Takes Q (B, n_heads, L, head_dim) and full cached K/V (B, n_kv_heads, total_len, head_dim)
// Returns output (B, L, hidden_size)
mlx_array* mlx_fused_attention_output(
    mlx_array* q_handle,
    mlx_array* k_handle,
    mlx_array* v_handle,
    mlx_array* w_o_handle,
    int n_heads,
    int head_dim,
    float attn_scale,
    bool use_causal
) {
    try {
        auto queries = reinterpret_cast<array*>(q_handle);
        auto keys = reinterpret_cast<array*>(k_handle);
        auto values = reinterpret_cast<array*>(v_handle);
        auto w_o = reinterpret_cast<array*>(w_o_handle);

        int batch = static_cast<int>(queries->shape()[0]);
        int q_len = static_cast<int>(queries->shape()[2]);
        int hidden_size = n_heads * head_dim;

        // SDPA - determine mask mode (valid modes: "causal", "array", or "" for none)
        std::string mask_mode = (use_causal && q_len > 1) ? "causal" : "";
        auto attn_output = mlx::core::fast::scaled_dot_product_attention(
            *queries, *keys, *values, attn_scale, mask_mode
        );
        attn_output.eval();  // Force GPU sync after expensive SDPA to prevent timeout

        // Transpose back: (B, n_heads, L, head_dim) -> (B, L, n_heads, head_dim)
        attn_output = transpose(attn_output, {0, 2, 1, 3});

        // Reshape: (B, L, n_heads, head_dim) -> (B, L, hidden_size)
        attn_output = reshape(attn_output, {batch, q_len, hidden_size});

        // Output projection
        auto w_o_t = transpose(*w_o);
        auto output = matmul(attn_output, w_o_t);

        return reinterpret_cast<mlx_array*>(new array(std::move(output)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_fused_attention_output error: " << e.what() << std::endl;
        return nullptr;
    }
}

}  // extern "C"
