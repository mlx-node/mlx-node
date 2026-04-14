#include "mlx_common.h"
#include "mlx_qwen35_common.h"

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

// E39: Stacked SwiGLU MLP. Takes pre-stacked + pre-transposed weights computed
// once at model load (see MLP::finalize_gate_up in transformer/mlp.rs):
//   w_gate_up_t: [hidden, 2*intermediate]  (concatenate then transpose)
//   w_down_t:    [intermediate, hidden]    (transpose)
//
// Saves one of the two MLP matmuls (gate and up fused into one [hidden,
// 2*intermediate] matmul). For Qwen3.5-4B: hidden=2560, intermediate=9216, so
// the fused matmul is x @ W [B,T,2560] @ [2560, 18432]. Lookup of W tile/cache
// is amortized vs the two separate matmuls.
mlx_array* mlx_swiglu_mlp_forward_stacked(mlx_array* x_handle,
                                           mlx_array* w_gate_up_t_handle,
                                           mlx_array* w_down_t_handle) {
  auto x = reinterpret_cast<array*>(x_handle);
  auto w_gate_up_t = reinterpret_cast<array*>(w_gate_up_t_handle);
  auto w_down_t = reinterpret_cast<array*>(w_down_t_handle);

  // x: [B, T, hidden]; w_gate_up_t: [hidden, 2*intermediate]
  auto gate_up = matmul(*x, *w_gate_up_t);  // [B, T, 2*intermediate]

  int intermediate = static_cast<int>(gate_up.shape().back()) / 2;
  int B = static_cast<int>(gate_up.shape()[0]);
  int T = static_cast<int>(gate_up.shape()[1]);

  auto gate = slice(gate_up, {0, 0, 0}, {B, T, intermediate});
  auto up   = slice(gate_up, {0, 0, intermediate}, {B, T, 2 * intermediate});

  // E40: fused swiglu via mlx::core::compile (one Metal kernel for
  // sigmoid(gate)*gate*up). Helper is from mlx_qwen35_common.h.
  auto gated = qwen35_common::swiglu(gate, up);

  // down: gated @ w_down_t
  auto output = matmul(gated, *w_down_t);
  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

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

// Fused GDN pre-recurrence forward pass.
//
// Collapses the pre-recurrence portion of GatedDeltaNet::forward into a single
// FFI call. Replicates the exact op sequence from
// crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs:128-262 so greedy
// decode tokens remain byte-identical to the per-op path.
//
// Fuses:
//   qkvz = x @ w_qkvz.T
//   ba   = x @ w_ba.T
//   b, a = split(ba, num_v_heads, axis=2)
//   qkv, z = split(qkvz, conv_dim, axis=2)
//   if mask: qkv = where(mask[:, :, None], qkv, zeros_of_dtype(qkv))
//   conv_input = concat(conv_state_or_zeros, qkv, axis=1)
//   new_conv_state = conv_input[:, total_len - (kernel-1):, :]
//   conv_out = conv1d(conv_input, w_conv, 1, 0, 1, conv_dim)
//   conv_out = conv_out[:, -T:, :] if longer
//   conv_out = silu(conv_out)
//   q_flat = conv_out[:, :, :key_dim]
//   k_flat = conv_out[:, :, key_dim:2*key_dim]
//   v_flat = conv_out[:, :, 2*key_dim:conv_dim]
//   q = rms_norm_no_weight(q_flat.reshape(B,T,Hk,Dk), 1e-6) * inv_scale^2
//   k = rms_norm_no_weight(k_flat.reshape(B,T,Hk,Dk), 1e-6) * inv_scale
//   v = v_flat.reshape(B,T,Hv,Dv)
//
// z remains flat [B, T, value_dim] matching the slice output (NOT per-head).
//
// Writes 7 outputs via out-pointers. Returns 0 on success, -1 on error.
// Caller owns all returned mlx_array*.
int mlx_gdn_prefusion_forward(
    mlx_array* x_handle,            // [B, T, hidden]
    mlx_array* w_qkvz_handle,       // [key_dim*2 + value_dim*2, hidden]
    mlx_array* w_ba_handle,         // [num_v_heads*2, hidden]
    mlx_array* w_conv_handle,       // [conv_dim, kernel_dim, 1]
    mlx_array* conv_state_handle,   // [B, kernel_dim-1, conv_dim] or nullptr
    mlx_array* mask_handle,         // [B, T] or nullptr
    int num_k_heads,
    int num_v_heads,
    int key_head_dim,
    int value_head_dim,
    int conv_kernel_dim,
    float rms_eps,
    mlx_array** out_q,
    mlx_array** out_k,
    mlx_array** out_v,
    mlx_array** out_z,
    mlx_array** out_a,
    mlx_array** out_b,
    mlx_array** out_new_conv_state) {
    try {
        auto x = reinterpret_cast<array*>(x_handle);
        auto w_qkvz = reinterpret_cast<array*>(w_qkvz_handle);
        auto w_ba = reinterpret_cast<array*>(w_ba_handle);
        auto w_conv = reinterpret_cast<array*>(w_conv_handle);

        int batch = static_cast<int>(x->shape()[0]);
        int seq_len = static_cast<int>(x->shape()[1]);
        int key_dim = num_k_heads * key_head_dim;
        int value_dim = num_v_heads * value_head_dim;
        int conv_dim = key_dim * 2 + value_dim;

        // 1. Projections: qkvz = x @ w_qkvz.T, ba = x @ w_ba.T
        auto w_qkvz_t = transpose(*w_qkvz, {1, 0});
        auto w_ba_t = transpose(*w_ba, {1, 0});
        auto qkvz = matmul(*x, w_qkvz_t);  // [B, T, key_dim*2 + value_dim*2]
        auto ba = matmul(*x, w_ba_t);      // [B, T, num_v_heads*2]

        // 2. Split ba → b, a (each [B, T, num_v_heads]), via slice along axis 2.
        auto b = slice(ba, {0, 0, 0}, {batch, seq_len, num_v_heads});
        auto a = slice(ba, {0, 0, num_v_heads}, {batch, seq_len, num_v_heads * 2});

        // 3. Split qkvz → qkv (conv_dim wide), z (remaining).
        int qkvz_last = key_dim * 2 + value_dim * 2;
        auto qkv = slice(qkvz, {0, 0, 0}, {batch, seq_len, conv_dim});
        auto z = slice(qkvz, {0, 0, conv_dim}, {batch, seq_len, qkvz_last});

        // 4. Optional mask application: qkv = where(mask[:,:,None], qkv, zeros).
        //    Use qkv's dtype to avoid f32 promotion for bf16/f16 models.
        if (mask_handle) {
            auto mask = reinterpret_cast<array*>(mask_handle);
            auto m_3d = reshape(*mask, {batch, seq_len, 1});
            auto zero_scalar = zeros({1}, qkv.dtype());
            qkv = where(m_3d, qkv, zero_scalar);
        }

        // 5. Prepend conv_state (or zero pad) to qkv along the time axis.
        array conv_input = [&]() -> array {
            int pad_len = conv_kernel_dim - 1;
            if (conv_state_handle) {
                auto conv_state = reinterpret_cast<array*>(conv_state_handle);
                return concatenate({*conv_state, qkv}, 1);
            }
            auto zero_pad = zeros({batch, pad_len, conv_dim}, qkv.dtype());
            return concatenate({zero_pad, qkv}, 1);
        }();

        // 6. Save new conv_state = conv_input[:, total_len - (kernel-1):, :].
        int total_len = static_cast<int>(conv_input.shape()[1]);
        int keep = conv_kernel_dim - 1;
        array new_conv_state = (total_len >= keep)
            ? slice(conv_input,
                    {0, total_len - keep, 0},
                    {batch, total_len, conv_dim})
            : conv_input;

        // 7. Depthwise conv1d — replicate Rust Conv1d::forward_depthwise
        //    (crates/mlx-core/src/nn/conv1d.rs) byte-for-byte so both the
        //    per-op WASM path and this fused path build the same lazy graph
        //    and therefore produce identical bf16 values.
        //
        //    Weight: [C, K, 1] → squeeze(axis=2) → [C, K]
        //    For each k in [0..K):
        //      input_slice = conv_input[:, k:k+t_out, :]   (axis=1 slice)
        //      w_k         = weight_2d[:, k:k+1]           ([C, 1])
        //                      .squeeze(axis=1)             ([C])
        //                      .reshape({1, 1, -1})         ([1, 1, C])
        //      term        = input_slice * w_k
        //      result      = (k == 0) ? term : result + term
        int conv_in_len = static_cast<int>(conv_input.shape()[1]);
        int t_out = conv_in_len - conv_kernel_dim + 1;
        if (t_out <= 0) {
            throw std::runtime_error(
                "mlx_gdn_prefusion_forward: input seq_len < kernel_size");
        }
        auto weight_2d = squeeze(*w_conv, std::vector<int>{2});  // [C, K]
        array conv_out_acc = array(0.0f, conv_input.dtype());     // placeholder
        for (int k = 0; k < conv_kernel_dim; ++k) {
            auto input_slice = slice(
                conv_input,
                {0, k, 0},
                {batch, k + t_out, conv_dim});  // [B, t_out, C]
            auto w_col = slice(weight_2d, {0, k}, {conv_dim, k + 1});  // [C, 1]
            auto w_col_1d = squeeze(w_col, std::vector<int>{1});       // [C]
            auto w_k = reshape(w_col_1d, {1, 1, conv_dim});            // [1, 1, C]
            auto term = input_slice * w_k;
            conv_out_acc = (k == 0) ? term : (conv_out_acc + term);
        }
        auto conv_out = conv_out_acc;

        // 8. If conv produced more than seq_len timesteps, take the last seq_len.
        int conv_out_len = static_cast<int>(conv_out.shape()[1]);
        if (conv_out_len > seq_len) {
            conv_out = slice(conv_out,
                             {0, conv_out_len - seq_len, 0},
                             {batch, conv_out_len, conv_dim});
        }

        // 9. SiLU activation.
        conv_out = conv_out * sigmoid(conv_out);

        // 10. Split into q, k, v (flat along channel axis).
        auto q_flat = slice(conv_out, {0, 0, 0}, {batch, seq_len, key_dim});
        auto k_flat = slice(conv_out, {0, 0, key_dim}, {batch, seq_len, key_dim * 2});
        auto v_flat = slice(conv_out, {0, 0, key_dim * 2}, {batch, seq_len, conv_dim});

        // 11. Reshape to [B, T, H, D].
        auto q = reshape(q_flat, {batch, seq_len, num_k_heads, key_head_dim});
        auto k = reshape(k_flat, {batch, seq_len, num_k_heads, key_head_dim});
        auto v = reshape(v_flat, {batch, seq_len, num_v_heads, value_head_dim});

        // 12. RMS norm (no weight) and inv_scale application matching the
        //     Rust per-op path exactly:
        //       inv_scale = key_head_dim^(-0.5)
        //       q = rms_norm_no_weight(q, eps) * (inv_scale * inv_scale) as scalar
        //       k = rms_norm_no_weight(k, eps) * inv_scale as scalar
        //
        //     Multiply by a dtype-matched scalar array so bf16 stays bf16.
        //     Match Rust `(key_head_dim as f64).powf(-0.5)` precision by
        //     computing the scalar in double and casting to float at the
        //     point where mlx_array_mul_scalar builds its scalar array.
        double inv_scale_d = std::pow(static_cast<double>(key_head_dim), -0.5);
        float q_scale_f = static_cast<float>(inv_scale_d * inv_scale_d);
        float k_scale_f = static_cast<float>(inv_scale_d);
        auto q_dtype = q.dtype();
        auto k_dtype = k.dtype();
        q = fast::rms_norm(q, std::nullopt, rms_eps, {});
        k = fast::rms_norm(k, std::nullopt, rms_eps, {});
        q = q * array(q_scale_f, q_dtype);
        k = k * array(k_scale_f, k_dtype);

        *out_q = reinterpret_cast<mlx_array*>(new array(std::move(q)));
        *out_k = reinterpret_cast<mlx_array*>(new array(std::move(k)));
        *out_v = reinterpret_cast<mlx_array*>(new array(std::move(v)));
        *out_z = reinterpret_cast<mlx_array*>(new array(std::move(z)));
        *out_a = reinterpret_cast<mlx_array*>(new array(std::move(a)));
        *out_b = reinterpret_cast<mlx_array*>(new array(std::move(b)));
        *out_new_conv_state =
            reinterpret_cast<mlx_array*>(new array(std::move(new_conv_state)));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "mlx_gdn_prefusion_forward error: " << e.what() << std::endl;
        *out_q = nullptr;
        *out_k = nullptr;
        *out_v = nullptr;
        *out_z = nullptr;
        *out_a = nullptr;
        *out_b = nullptr;
        *out_new_conv_state = nullptr;
        return -1;
    }
}

}  // extern "C"
