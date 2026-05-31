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
  // Per-expert SwiGLU hidden dim (moe layers) and dense-in-MoE SwiGLU hidden dim.
  // Currently informational — the gather experts infer shapes from the stacked
  // weights — kept for symmetry with the Rust config and future prefill paths.
  int moe_intermediate_size = 0;
  int intermediate_size = 0;
  bool norm_topk_prob = true;
  bool use_expert_bias = true;
  // Router score function: false -> softmax (lfm2_moe native default), true ->
  // sigmoid (qwen3.5-moe style). lfm2_moe `Lfm2SparseMoeBlock` uses SOFTMAX.
  bool use_sigmoid = false;
  // ShortConv bias flag. When true the conv pure-fn adds the three conv biases
  // (conv.in_proj.bias, conv.conv.bias, conv.out_proj.bias). Default false
  // (bias-free checkpoint).
  bool conv_bias = false;
  // Phase 3b quant fields (default 0 = bf16). Declared NOW so the config ABI is
  // stable across 3a->3b. quant_mode: 0=bf16, 1=mxfp8, 2=mxfp4, 3=nvfp4.
  int quant_mode = 0;
  int bits = 0;
  int group_size = 0;
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
// Array-offset variant of lfm2_attn_pure_fn for the COMPILED decode graph.
//
// Identical math to the scalar fn above (same lfm2 specifics: GQA, per-head q/k
// RMSNorm, NO q-gate/bias, neox RoPE over the full head_dim applied AFTER the
// transpose, out_proj), with exactly two substitutions so the graph topology is
// invariant across decode steps (mirrors qwen35 `attn_for_compile`):
//   - fast::rope takes the *array* offset overload (not a baked-in int)
//   - slice_update's KV start index is reshape(offset_arr, {1}) (not array(int))
// Always uses the fixed-shape padded KV cache + static additive mask
// (positions <= offset -> 0, else -inf), the compile-safe path — so this is the
// dynamic_kv=false branch with an array offset. The scalar fn is kept unchanged
// so the Phase-2a operator probes stay green; the decode loop calls THIS one.
// =====================================================================
inline Lfm2AttnResult lfm2_attn_pure_fn_arr(
    const array& x,
    int layer_idx,
    const array& kv_keys,
    const array& kv_values,
    const array& attn_mask,
    const array& offset_arr,
    const Lfm2MoeConfig& cfg) {
  using namespace qwen35_common;
  int B = x.shape(0);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  auto queries = linear_proj(x, pfx + "q_proj");
  auto keys = linear_proj(x, pfx + "k_proj");
  auto values = linear_proj(x, pfx + "v_proj");

  queries = reshape(queries, {B, 1, cfg.num_heads, cfg.head_dim});
  keys = reshape(keys, {B, 1, cfg.num_kv_heads, cfg.head_dim});
  values = reshape(values, {B, 1, cfg.num_kv_heads, cfg.head_dim});

  queries =
      mlx::core::fast::rms_norm(queries, get_weight(pfx + "q_layernorm.weight"), cfg.norm_eps);
  keys = mlx::core::fast::rms_norm(keys, get_weight(pfx + "k_layernorm.weight"), cfg.norm_eps);

  // Transpose to [B,H,T,D] FIRST, then RoPE (lfm2 ordering — position axis = T).
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  // RoPE with the ARRAY offset overload (graph-safe).
  queries =
      mlx::core::fast::rope(queries, cfg.head_dim, false, cfg.rope_theta, 1.0f, offset_arr);
  keys = mlx::core::fast::rope(keys, cfg.head_dim, false, cfg.rope_theta, 1.0f, offset_arr);

  // KV cache update via slice_update at axis 2, ARRAY start index.
  auto offset_1d = reshape(offset_arr, {1});
  auto new_kv_keys = mlx::core::slice_update(kv_keys, keys, offset_1d, {2});
  auto new_kv_values = mlx::core::slice_update(kv_values, values, offset_1d, {2});

  // SDPA over the fixed-shape padded cache + static additive mask.
  float scale = std::pow(static_cast<float>(cfg.head_dim), -0.5f);
  array attn_out = mlx::core::fast::scaled_dot_product_attention(
      queries, new_kv_keys, new_kv_values, scale, "", attn_mask, {});

  attn_out = transpose(attn_out, {0, 2, 1, 3});
  attn_out = reshape(attn_out, {B, cfg.num_heads * cfg.head_dim});
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

// =====================================================================
// Per-expert switch-linear via gather_mm (bf16) or gather_qmm (quant).
//
// `prefix` is the full key prefix WITHOUT a trailing ".weight"/".scales"/
// ".biases" (e.g. "layers.3.feed_forward.switch_mlp.gate_proj").
//   x:       [ne, 1, in]   (or [ne, k, in] for the down stage)
//   indices: [ne, k]       rhs (expert) indices
// Returns [ne, k, out]. The bf16 branch swaps the stored [E, out, in] weight to
// [E, in, out] for gather_mm (matching qwen35 `switch_linear_forward`). The quant
// branch (cfg.quant_mode != 0) is declared for Phase 3b; 3a only exercises bf16.
// =====================================================================
inline array lfm2_switch_linear(
    const array& x,
    const std::string& prefix,
    const array& indices,
    const Lfm2MoeConfig& cfg) {
  using namespace qwen35_common;
  if (cfg.quant_mode != 0) {
    // Quantized experts (Phase 3b). gate/up/down ship stacked .scales (+ optional
    // .biases for affine recipes). mode string per cfg.quant_mode.
    auto w = get_weight(prefix + ".weight");
    auto scales = get_weight(prefix + ".scales");
    std::optional<array> biases = std::nullopt;
    if (g_weights().count(prefix + ".biases") > 0) {
      biases = get_weight(prefix + ".biases");
    }
    const char* mode = "mxfp8";
    if (cfg.quant_mode == 2) {
      mode = "mxfp4";
    } else if (cfg.quant_mode == 3) {
      mode = "nvfp4";
    }
    return mlx::core::gather_qmm(
        x, w, scales, biases,
        std::nullopt,  // lhs_indices (not used)
        indices,       // rhs_indices (expert indices)
        true,          // transpose
        std::optional<int>(cfg.group_size),
        std::optional<int>(cfg.bits),
        std::string(mode),
        /*sorted=*/false);
  }
  // bf16/f16: plain gather_mm over the swapped weight ([E, in, out]).
  auto w = get_weight(prefix + ".weight");  // [E, out, in]
  return mlx::core::gather_mm(x, mlx::core::swapaxes(w, -2, -1), std::nullopt, indices);
}

// =====================================================================
// Sparse top-k MoE FFN for an lfm2 layer (mirrors `Lfm2SparseMoeBlock::forward`
// in sparse_moe.rs and lfm2_moe.py::Lfm2MoeSparseMoeBlock).
//
// Routing (parity-critical — each step matches the native Rust line-for-line):
//   logits  = gate(x).astype(f32)                              # [ne, E]
//   scores  = use_sigmoid ? sigmoid(logits) : softmax(logits, -1)  # lfm2: SOFTMAX
//   routing = use_expert_bias ? scores + expert_bias : scores  # selection gates
//   inds    = argpartition(routing, E-k-1, -1)[..., E-k:E]      # last k
//   weights = take_along_axis(routing, inds, -1)               # from the SAME
//                                                              #   (post-bias) gates
//                                                              #   the native code
//                                                              #   argpartitions
//   weights = norm_topk_prob ? weights / (sum(weights,-1) + 1e-20) : weights
//   weights = weights.astype(x.dtype)
//   y       = SwiGLU experts via gather_mm/gather_qmm           # [ne, k, hidden]
//   out     = sum(y * weights[...,None], -2)                    # [ne, hidden]
//
// Native `sparse_moe.rs` mutates `gates` in place by adding the bias, then BOTH
// argpartitions AND take_along_axis on that post-bias tensor. We replicate that
// by argpartition + take_along_axis BOTH on `routing`. When use_expert_bias is
// false, routing == scores, so the two are identical. NO shared expert, NO
// routed_scaling. The renorm epsilon (1e-20, f32) matches sparse_moe.rs.
//
// x: [ne, hidden] (2D decode) -> [ne, hidden].
// =====================================================================
inline array lfm2_moe_ffn(const array& x, int layer_idx, const Lfm2MoeConfig& cfg) {
  using namespace qwen35_common;
  int E = cfg.num_experts;
  int k = cfg.num_experts_per_tok;
  std::string pfx = "layers." + std::to_string(layer_idx) + ".feed_forward.";

  // ---- Router ----
  auto logits = linear_proj(x, pfx + "gate");  // [ne, E]; gate.weight is [E, hidden]
  auto sf32 = astype(logits, mlx::core::float32);
  array scores =
      cfg.use_sigmoid ? mlx::core::sigmoid(sf32) : mlx::core::softmax(sf32, std::vector<int>{-1});

  // expert_bias is a selection gate add (post-softmax, pre-topk). Native gathers
  // the routing weights from these same biased gates, so `routing` is also the
  // gather source. When use_expert_bias is false, routing == scores.
  array routing = scores;
  if (cfg.use_expert_bias) {
    routing = add(scores, astype(get_weight(pfx + "expert_bias"), mlx::core::float32));
  }

  // ---- Top-k: mirror native argpartition(routing, E-k-1, -1)[.., E-k:E] ----
  // Slice the last axis [E-k:E] rank-agnostically: `routing`/`part` may be 2D
  // ([ne, E]) or 3D ([B, ne, E]) depending on the caller's batch shape, so build
  // the per-axis start/stop from `part.shape()` rather than hard-coding rank 2.
  auto part = mlx::core::argpartition(routing, E - k - 1, -1);
  mlx::core::Shape slc_start(part.ndim(), 0);
  mlx::core::Shape slc_stop = part.shape();
  slc_start.back() = E - k;  // last axis lower bound
  slc_stop.back() = E;       // last axis upper bound
  auto inds = slice(part, slc_start, slc_stop);  // [..., k]

  // weights gathered from the (biased) routing gates; renorm adds 1e-20 (native).
  array weights = take_along_axis(routing, inds, -1);  // [ne, k]
  if (cfg.norm_topk_prob) {
    auto denom = sum(weights, std::vector<int>{-1}, /*keepdims=*/true);
    denom = add(denom, array(1e-20f, mlx::core::float32));
    weights = divide(weights, denom);
  }
  weights = astype(weights, x.dtype());

  // ---- Experts via gather_mm/gather_qmm (SwiGLU) ----
  // gather_mm requires the lhs to carry a leading expert-broadcast axis: expand
  // x [ne, hidden] -> [ne, 1, 1, hidden] (matches the qwen3.5-MoE switch convention).
  // gather_mm(x4, swapaxes(w), nullopt, inds[ne,k]) -> [ne, 1, k, out]; the SwiGLU
  // and down_proj keep that rank; squeeze the penultimate (1) axis afterward.
  auto x4 = reshape(x, {-1, 1, 1, x.shape(-1)});  // [ne, 1, 1, hidden]
  auto gproj = lfm2_switch_linear(x4, pfx + "switch_mlp.gate_proj", inds, cfg);
  auto uproj = lfm2_switch_linear(x4, pfx + "switch_mlp.up_proj", inds, cfg);
  auto act = swiglu(gproj, uproj);  // SiLU(gate) * up  -> [ne, 1, k, moe_inter]
  auto y4 = lfm2_switch_linear(act, pfx + "switch_mlp.down_proj", inds, cfg);  // [ne, 1, k, hidden]
  auto y = squeeze(y4, -2);  // drop the penultimate broadcast axis -> [ne, k, hidden]

  // weighted sum over k: (y * weights[...,None]).sum(-2) -> [ne, hidden]
  auto w_exp = expand_dims(weights, -1);  // [ne, k, 1]
  return sum(multiply(y, w_exp), std::vector<int>{-2}, /*keepdims=*/false);
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
