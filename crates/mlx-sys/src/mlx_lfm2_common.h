#pragma once

// =============================================================================
// LFM2.5 MoE compiled forward path — shared definitions.
//
// Phase 0 (inert scaffold): this header declares the config POD that the
// compiled graph will consume and pulls in the shared MLX includes. The actual
// graph (pure-fns, weight lookups, compiled decode closure) lands in later
// phases, modeled on `mlx_qwen35_common.h`.
//
// The process-global weight registry (`g_weights()`, `get_weight()`,
// `linear_proj()`, `g_active_model_id()`) is process-wide and model-agnostic;
// it is reused from `mlx_qwen35_common.h` rather than duplicated.
// =============================================================================

#include "mlx_common.h"
#include "mlx_qwen35_common.h"  // shared weight registry + linear_proj/swiglu/get_weight

#include <cmath>
#include <string>

namespace lfm2_common {

// Mirrors the fields of Rust `Lfm2Config` that the compiled forward needs.
// POD only (no `mlx::core::array` members — `array` has no default ctor).
// Phase 0: declared for the FFI init signature; populated in Phase 1.
struct Lfm2MoeConfig {
  int num_layers = 0;
  int hidden_size = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  float rope_theta = 0.0f;
  float norm_eps = 0.0f;
  int conv_l_cache = 0;
  int num_experts = 0;
  int num_experts_per_tok = 0;
  int num_dense_layers = 0;
  bool norm_topk_prob = true;
  bool use_expert_bias = true;
  bool tie_embedding = true;
  int max_kv_len = 0;
  int batch_size = 1;
};

// Result of one attention layer: output + the updated KV caches to write back.
struct Lfm2AttnResult {
  array output;
  array keys;
  array values;
};

// =====================================================================
// Single-token decode attention for an lfm2 full_attention layer.
//
// Mirrors `Lfm2Attention::forward` (attention.rs:88) and `lfm2.py:79-109`:
//   - GQA (num_heads q / num_kv_heads kv), head_dim per head
//   - per-head RMSNorm on Q and K (NONE on V), eps = norm_eps
//   - neox RoPE (traditional=false) over the FULL head_dim, base = rope_theta
//   - NO q-gating (unlike qwen3.5's 2x-width q_proj), NO bias on any proj
//   - scale = head_dim^-0.5; output proj key is "out_proj" (not "o_proj")
//
// x:        [B, hidden] (2D decode). kv_keys/kv_values: [B, num_kv_heads,
//           max_kv_len, head_dim]. attn_mask: [1,1,1,max_kv_len] additive bf16
//           (used only when dynamic_kv=false). offset: scalar position.
// dynamic_kv=true slices the KV cache to the valid range [0..offset+1] and
// passes NO mask (numerically identical to the native decode path, which uses
// a freshly-grown cache + no mask); =false uses the fixed-shape padded cache +
// additive mask (required when the fn is wrapped in mlx::core::compile).
// =====================================================================
inline Lfm2AttnResult lfm2_attn_pure_fn(
    const array& x,
    int layer_idx,
    const array& kv_keys,
    const array& kv_values,
    const array& attn_mask,
    int offset,
    const Lfm2MoeConfig& cfg,
    bool dynamic_kv = false) {
  using namespace qwen35_common;
  int B = x.shape(0);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  // 1. Q/K/V projections — NO bias, NO 2x gate width.
  auto queries = linear_proj(x, pfx + "q_proj");
  auto keys = linear_proj(x, pfx + "k_proj");
  auto values = linear_proj(x, pfx + "v_proj");

  // 2. Reshape to [B, 1, H, D] (T=1 for decode).
  queries = reshape(queries, {B, 1, cfg.num_heads, cfg.head_dim});
  keys = reshape(keys, {B, 1, cfg.num_kv_heads, cfg.head_dim});
  values = reshape(values, {B, 1, cfg.num_kv_heads, cfg.head_dim});

  // 3. Per-head RMSNorm on Q and K over head_dim (eps = norm_eps). V: none.
  //    Applied on [B,1,H,D] BEFORE the transpose, matching native
  //    `Lfm2Attention::forward` (attention.rs:105).
  queries =
      mlx::core::fast::rms_norm(queries, get_weight(pfx + "q_layernorm.weight"), cfg.norm_eps);
  keys = mlx::core::fast::rms_norm(keys, get_weight(pfx + "k_layernorm.weight"), cfg.norm_eps);

  // 4. Transpose to [B, H, T, D] FIRST, so RoPE's position axis is T (axis -2),
  //    not the head axis. The native path ropes the already-transposed
  //    [B,H,T,D] (attention.rs:107,129) — roping the pre-transpose [B,1,H,D]
  //    would assign per-HEAD positions and corrupt the rotation.
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  // 5. neox RoPE over the FULL head_dim (no partial dims), base = rope_theta.
  queries =
      mlx::core::fast::rope(queries, cfg.head_dim, false, cfg.rope_theta, 1.0f, offset);
  keys = mlx::core::fast::rope(keys, cfg.head_dim, false, cfg.rope_theta, 1.0f, offset);

  // 6. KV cache update via slice_update at axis 2 (time), array start index.
  auto offset_1d = reshape(array(offset, mlx::core::int32), {1});
  auto new_kv_keys = mlx::core::slice_update(kv_keys, keys, offset_1d, {2});
  auto new_kv_values = mlx::core::slice_update(kv_values, values, offset_1d, {2});

  // 7. SDPA. scale = head_dim^-0.5.
  float scale = std::pow(static_cast<float>(cfg.head_dim), -0.5f);
  array attn_out = [&]() -> array {
    if (dynamic_kv) {
      int valid_len = offset + 1;
      auto vk = slice(new_kv_keys, {0, 0, 0, 0},
                      {B, cfg.num_kv_heads, valid_len, cfg.head_dim});
      auto vv = slice(new_kv_values, {0, 0, 0, 0},
                      {B, cfg.num_kv_heads, valid_len, cfg.head_dim});
      return mlx::core::fast::scaled_dot_product_attention(
          queries, vk, vv, scale, "", std::nullopt, {});
    }
    return mlx::core::fast::scaled_dot_product_attention(
        queries, new_kv_keys, new_kv_values, scale, "", attn_mask, {});
  }();

  // 8. [B,H,T,D] -> [B,T,H,D] -> [B, H*D]. NO gate.
  attn_out = transpose(attn_out, {0, 2, 1, 3});
  attn_out = reshape(attn_out, {B, cfg.num_heads * cfg.head_dim});

  // 9. Output projection — "out_proj", NO bias.
  auto output = linear_proj(attn_out, pfx + "out_proj");

  return {output, new_kv_keys, new_kv_values};
}

// =====================================================================
// Dense SwiGLU MLP for an lfm2 layer (mirrors `MLP::forward` / lfm2.py).
//   down_proj(swiglu(gate_proj(x), up_proj(x))), keys
//   layers.{i}.feed_forward.{gate,up,down}_proj.
// x: [B, hidden] -> [B, hidden].
// =====================================================================
inline array lfm2_dense_mlp(const array& x, int layer_idx) {
  using namespace qwen35_common;
  std::string mp = "layers." + std::to_string(layer_idx) + ".feed_forward.";
  auto gate = linear_proj(x, mp + "gate_proj");
  auto up = linear_proj(x, mp + "up_proj");
  return linear_proj(swiglu(gate, up), mp + "down_proj");
}

// Result of one ShortConv decode step: output + the conv state to write back.
struct Lfm2ConvResult {
  array output;
  array new_state;
};

// =====================================================================
// Single-token decode for an lfm2 ShortConv (gated depthwise Conv1d) layer.
//
// Token-for-token port of the `ShortConv::forward` decode branch
// (short_conv.rs:69-127) / `lfm2.py:134-170`:
//   BCx = in_proj(x)                       [B, 3*hidden]  (+bias iff conv_bias)
//   B,C,x = split into 3 along last axis (ORDER: B, C, x)
//   Bx = B * x                             [B, hidden]
//   bx_3d = reshape(Bx, [B, 1, hidden])
//   conv_in = concatenate(conv_state, bx_3d, axis=1)   [B, l_cache, hidden]
//   new_state = last (l_cache-1) rows of conv_in (axis 1)  [B, l_cache-1, hidden]
//   conv_out = conv1d(conv_in, W[H,K,1], 1,0,1, groups=hidden)  [B, 1, hidden]
//                                          (+conv bias [hidden] iff conv_bias)
//   y = C * conv_out                       [B, 1, hidden]  (C broadcasts over T)
//   out = out_proj(reshape(y, [B,hidden]))              (+bias iff conv_bias)
//
// ASSUMES single-token decode (one token/step, fully-warm cache): the native
// Rust `ShortConv::forward` decode branch makes the same simplification — no SSM
// `conv_mask` / no `cache.lengths`-aware retention (those only matter for ragged
// batched prefill, lfm2.py:143-163). Do NOT reuse this for batched decode with
// ragged lengths without adding the mask + length-aware retention.
//
// x:          [B, hidden] (2D decode input — already operator-normed by caller).
// conv_state: [B, l_cache-1, hidden] (zeros on the first step, prior new_state
//             after). Threaded by the caller across decode steps (slot 0).
// Weight keys (registered under layers.{layer_idx}.conv.*): note the DOUBLED
// `conv.conv` for the depthwise weight — the ShortConv block prefix is
// `...conv.` and the nn.Conv1d submodule inside it is ALSO named `conv`, so the
// real checkpoint key is `layers.{i}.conv.conv.weight` (persistence.rs:907).
// Since `pfx` already ends in `conv.`, the depthwise leaf is `"conv.weight"`.
//   in_proj.weight [3H,H] (+in_proj.bias [3H]), out_proj.weight [H,H]
//   (+out_proj.bias [H]), conv.conv.weight [H, l_cache, 1] (+conv.conv.bias [H]).
// Biases are present iff conv_bias=true (a single config flag gates all three).
// =====================================================================
inline Lfm2ConvResult lfm2_conv_pure_fn(
    const array& x,
    int layer_idx,
    const array& conv_state,
    int l_cache,
    int hidden,
    bool conv_bias) {
  using namespace qwen35_common;
  int B = x.shape(0);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".conv.";

  // 1. in_proj: [B, hidden] -> [B, 3*hidden]. linear_proj does NOT add the
  //    additive bias, so add it manually (broadcasts over [3H]) when present.
  auto bcx = linear_proj(x, pfx + "in_proj");
  if (conv_bias) {
    bcx = add(bcx, get_weight(pfx + "in_proj.bias"));
  }

  // 2. split into B, C, x along the last axis (ORDER: B, C, x — input gate B*x,
  //    output gate C). Each [B, hidden].
  auto b_gate = slice(bcx, {0, 0}, {B, hidden});
  auto c_gate = slice(bcx, {0, hidden}, {B, hidden * 2});
  auto x_val = slice(bcx, {0, hidden * 2}, {B, hidden * 3});

  // 3. input gate Bx = B * x, then reshape to time-major [B, 1, hidden].
  auto bx = b_gate * x_val;
  auto bx_3d = reshape(bx, {B, 1, hidden});

  // 4. conv state: prepend (l_cache-1) cached positions on the time axis.
  //    conv_in length is exactly l_cache; new_state keeps the LAST (l_cache-1).
  //    Same form as the GDN conv path (mlx_qwen35_common.h:521-524).
  auto conv_in = concatenate({conv_state, bx_3d}, 1);  // [B, l_cache, hidden]
  int total_len = l_cache;
  int keep = l_cache - 1;
  auto new_state =
      slice(conv_in, {0, total_len - keep, 0}, {B, total_len, hidden});  // [B, l_cache-1, hidden]

  // 5. depthwise conv1d: weight [hidden, l_cache, 1] (3D, NOT auto-transposed),
  //    stride 1, pad 0, dil 1, groups = hidden (DEPTHWISE). Input length
  //    l_cache, kernel l_cache -> output length 1 -> [B, 1, hidden].
  auto conv_w = get_weight(pfx + "conv.weight");  // -> layers.{i}.conv.conv.weight
  auto conv_out = mlx::core::conv1d(conv_in, conv_w, /*stride=*/1, /*padding=*/0,
                                    /*dilation=*/1, /*groups=*/hidden);  // [B, 1, hidden]
  if (conv_bias) {
    // conv1d has no bias param; add [hidden] manually (broadcasts over [B,1,hidden]).
    conv_out = add(conv_out, get_weight(pfx + "conv.bias"));  // -> layers.{i}.conv.conv.bias
  }

  // 6. output gate y = C * conv_out. c_gate is [B, hidden]; reshape to
  //    [B, 1, hidden] so it broadcasts cleanly against conv_out [B, 1, hidden].
  auto y = reshape(c_gate, {B, 1, hidden}) * conv_out;  // [B, 1, hidden]

  // 7. out_proj: [B, hidden] -> [B, hidden] (+bias iff conv_bias).
  auto out = linear_proj(reshape(y, {B, hidden}), pfx + "out_proj");
  if (conv_bias) {
    out = add(out, get_weight(pfx + "out_proj.bias"));
  }

  return {out, new_state};
}

}  // namespace lfm2_common
