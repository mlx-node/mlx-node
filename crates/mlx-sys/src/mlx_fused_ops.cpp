#include "mlx_common.h"
#include "mlx_qwen35_common.h"

namespace {

std::vector<array> sigmoid_mul_impl(const std::vector<array>& inputs) {
  return {inputs[1] * sigmoid(inputs[0])};
}

}  // namespace

extern "C" {

mlx_array* mlx_sigmoid_mul_compiled(mlx_array* gate_handle, mlx_array* value_handle) {
  const auto& gate = *reinterpret_cast<array*>(gate_handle);
  const auto& value = *reinterpret_cast<array*>(value_handle);
  static auto fn = mlx::core::compile(sigmoid_mul_impl, /*shapeless=*/true);
  auto output = fn({gate, value})[0];
  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fuse the activation independently of the projections, which may use
// different native K-quant formats for gate and up.
mlx_array* mlx_swiglu_compiled(mlx_array* gate_handle, mlx_array* up_handle) {
  const auto& gate = *reinterpret_cast<array*>(gate_handle);
  const auto& up = *reinterpret_cast<array*>(up_handle);
  auto output = qwen35_common::swiglu(gate, up);
  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

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

// Fused Q/K/V projection with one RoPE offset per batch row.
// Returns Q, K, V in attention layout (B, n_heads, L, head_dim).
void mlx_fused_attention_qkv_with_offsets(
    mlx_array* x_handle,
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* q_norm_w_handle,
    mlx_array* k_norm_w_handle,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_base,
    int rope_dims,
    float qk_norm_eps,
    mlx_array* rope_offsets_handle,
    mlx_array** q_out,
    mlx_array** k_out,
    mlx_array** v_out
) {
    try {
        auto x = reinterpret_cast<array*>(x_handle);
        auto w_q = reinterpret_cast<array*>(w_q_handle);
        auto w_k = reinterpret_cast<array*>(w_k_handle);
        auto w_v = reinterpret_cast<array*>(w_v_handle);
        auto rope_offsets = reinterpret_cast<array*>(rope_offsets_handle);

        int batch = static_cast<int>(x->shape()[0]);
        int seq_len = static_cast<int>(x->shape()[1]);

        auto queries = matmul(*x, transpose(*w_q));
        auto keys = matmul(*x, transpose(*w_k));
        auto values = matmul(*x, transpose(*w_v));

        queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
        keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
        values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

        if (q_norm_w_handle) {
            auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
            queries = mlx::core::fast::rms_norm(queries, *q_norm_w, qk_norm_eps);
        }
        if (k_norm_w_handle) {
            auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
            keys = mlx::core::fast::rms_norm(keys, *k_norm_w, qk_norm_eps);
        }

        queries = transpose(queries, {0, 2, 1, 3});
        keys = transpose(keys, {0, 2, 1, 3});
        values = transpose(values, {0, 2, 1, 3});

        queries = mlx::core::fast::rope(
            queries, rope_dims, false, std::optional<float>(rope_base),
            1.0f, *rope_offsets, std::nullopt, {});
        keys = mlx::core::fast::rope(
            keys, rope_dims, false, std::optional<float>(rope_base),
            1.0f, *rope_offsets, std::nullopt, {});

        *q_out = reinterpret_cast<mlx_array*>(new array(std::move(queries)));
        *k_out = reinterpret_cast<mlx_array*>(new array(std::move(keys)));
        *v_out = reinterpret_cast<mlx_array*>(new array(std::move(values)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_fused_attention_qkv_with_offsets error: " << e.what() << std::endl;
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

// Fused residual-add + RMSNorm: h = x + res (rounded to the compute dtype),
// normed = rms_norm(h) * w. One dispatch replaces the [Add + RMSNorm] pair at
// every post-attention and post-MLP norm site, and the kernel emits BOTH
// outputs so the residual stream stays exact. The element mapping mirrors
// MLX's rms_single_row/rms_looped partitioning (contiguous N_READS chunks
// striding lsize*N_READS, same simd_sum folds), so the result is bit-identical
// to the separate ops at the sizes those kernels serve.
//
// Contract: `x` and `res` are contiguous [rows, AXIS] (or flattened
// equivalents) of the same dtype T in {f32, f16, bf16}; `w` is [AXIS] in T;
// `eps` is a scalar f32 array. Outputs are `h` and `normed`, both [rows, AXIS]
// in T.
extern "C" bool mlx_fused_add_rmsnorm(mlx_array* x_handle,
                                    mlx_array* res_handle,
                                    mlx_array* w_handle,
                                    mlx_array* eps_handle,
                                    mlx_array** out_h,
                                    mlx_array** out_normed) {
  if (!x_handle || !res_handle || !w_handle || !eps_handle || !out_h ||
      !out_normed)
    return false;
  *out_h = nullptr;
  *out_normed = nullptr;
  try {
    const auto& x = *reinterpret_cast<array*>(x_handle);
    const auto& res = *reinterpret_cast<array*>(res_handle);
    const auto& w = *reinterpret_cast<array*>(w_handle);
    const auto& eps = *reinterpret_cast<array*>(eps_handle);
    if (x.ndim() < 1 || res.shape() != x.shape() || w.size() != x.shape(-1) ||
        eps.size() != 1 || eps.dtype() != mlx::core::float32)
      return false;
    // The kernel reads x/res/w at flat row-major offsets.
    if (!x.flags().row_contiguous || !res.flags().row_contiguous ||
        !w.flags().row_contiguous)
      return false;
    const auto dt = x.dtype();
    if (dt != res.dtype() || dt != w.dtype())
      return false;
    if (dt != mlx::core::bfloat16 && dt != mlx::core::float16 &&
        dt != mlx::core::float32)
      return false;
    const int axis = x.shape(-1);
    const int rows = x.size() / axis;
    if (rows < 1 || rows > 4096 || x.size() != rows * axis ||
        res.size() != rows * axis)
      return false;

    static auto kernel = mlx::core::fast::metal_kernel(
        "add_rmsnorm", {"x", "res", "w", "eps"}, {"h_out", "normed"},
#include "metal/common/add_rmsnorm.metal.inc"
    );
    // Mirror normalization.cpp's threadgroup sizing so the element mapping
    // matches rms_single_row (axis <= 4096) or rms_looped (above) exactly.
    constexpr int n_reads = 4;
    const int lsize = axis <= 4096
        ? ((axis + n_reads - 1) / n_reads + 31) / 32 * 32
        : 1024;
    auto outs = kernel({x, res, w, eps}, {x.shape(), x.shape()}, {dt, dt},
                       {lsize, rows, 1}, {lsize, 1, 1},
                       {{"T", dt}, {"AXIS", axis}}, std::nullopt, false,
                       mlx::core::Device::gpu);
    *out_h = reinterpret_cast<mlx_array*>(new array(std::move(outs[0])));
    *out_normed = reinterpret_cast<mlx_array*>(new array(std::move(outs[1])));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mlx_fused_add_rmsnorm error: " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "mlx_fused_add_rmsnorm error: unknown exception" << std::endl;
    return false;
  }
}

}  // extern "C"
