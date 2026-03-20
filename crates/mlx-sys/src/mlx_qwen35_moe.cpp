#include "mlx_qwen35_common.h"
#include <unordered_set>
#include <cstdlib>

using namespace qwen35_common;

// =============================================================================
// Qwen3.5 MoE Forward Pass — Compiled via mlx::core::compile
//
// The entire 40-layer MoE decode (single-token) is compiled into a cached
// computation graph. Despite MoE routing being data-dependent (different
// expert indices each step), ALL array shapes are fixed for B=1, k=8:
//   - argpartition output: always [1, num_experts]
//   - top-k indices: always [1, k]
//   - gather_qmm output: always [1, 1, 1, moe_intermediate_size]
//   - do_sort threshold (size >= 64): always FALSE for B=1, k=8
//
// compile() caches the graph topology and pre-encodes Metal command buffers.
// The offset is passed as an input array (not a C++ int constant) so the
// graph structure is identical across steps.
//
// Attention uses static additive mask over pre-allocated KV caches (fixed
// shape) instead of dynamic KV slicing (variable shape). The SDPA overhead
// from processing padding tokens is negligible (~0.02ms/step for 10 layers).
//
// Set MLX_NO_COMPILE=1 to fall back to the non-compiled path for A/B testing.
//
// Weights are shared with the dense path via g_weights() in the common header.
// =============================================================================

namespace {

// MoE-specific config (extends BaseConfig)
struct MoeConfig : BaseConfig {
  int num_experts;
  int num_experts_per_tok;
  bool norm_topk_prob;
  int decoder_sparse_step;
};

// Per-layer quantization info (detected at init time by probing g_weights)
struct LayerQuantInfo {
  // switch_mlp (expert) quantization
  bool sw_quant;
  int sw_gs, sw_bits;
  std::string sw_mode;
  // shared_expert quantization
  bool sh_quant;
  int sh_gs, sh_bits;
  std::string sh_mode;
  // router gate quantization
  bool g_quant;
  int g_gs, g_bits;
  std::string g_mode;
  // shared_expert_gate quantization
  bool sg_quant;
  int sg_gs, sg_bits;
  std::string sg_mode;
};

// Dense MLP quantization info
struct DenseMLPQuantInfo {
  bool quant;
  int gs, bits;
  std::string mode;
};

static MoeConfig g_moe_config{};
static std::vector<array> g_moe_caches;     // num_layers * 2 arrays
static int g_moe_offset_int = 0;
static bool g_moe_inited = false;
static std::vector<LayerQuantInfo> g_layer_quant;     // per-layer MoE quant info
static std::vector<DenseMLPQuantInfo> g_dense_quant;  // per-layer dense MLP quant info

// Cached 3D transposes for expert weights [E,out,in] → [E,in,out]
static std::unordered_map<std::string, array> g_weight_transposes_3d;

const array& get_weight_t3d(const std::string& name) {
  auto it = g_weight_transposes_3d.find(name);
  if (it != g_weight_transposes_3d.end()) {
    return it->second;
  }
  const auto& w = get_weight(name);
  // [E, out, in] → [E, in, out]
  auto wt = transpose(w, {0, 2, 1});
  auto [inserted_it, _] = g_weight_transposes_3d.emplace(name, std::move(wt));
  return inserted_it->second;
}

// =====================================================================
// Quantized linear helpers (call MLX quantized_matmul / gather_qmm directly)
// =====================================================================

// Standard quantized linear: quantized_matmul(x, w, scales, biases)
array quantized_linear_forward(
    const array& x,
    const std::string& prefix,
    int gs, int bits,
    const std::string& mode) {
  const auto& w = get_weight(prefix + ".weight");
  const auto& scales = get_weight(prefix + ".scales");
  std::optional<array> biases = std::nullopt;
  if (has_weight(prefix + ".biases")) {
    biases = get_weight(prefix + ".biases");
  }
  return mlx::core::quantized_matmul(
      x, w, scales, biases,
      true, // transpose
      std::optional<int>(gs),
      std::optional<int>(bits),
      mode);
}

// Linear forward: dense matmul or quantized
array linear_forward(
    const array& x,
    const std::string& prefix,
    bool is_quant, int gs, int bits, const std::string& mode) {
  if (is_quant && has_weight(prefix + ".scales")) {
    return quantized_linear_forward(x, prefix, gs, bits, mode);
  }
  return matmul(x, get_weight_t(prefix + ".weight"));
}

// Switch linear forward: gather_mm (non-quantized)
array switch_linear_forward(
    const array& x,         // [N, 1, 1, D] or sorted [N*k, 1, 1, D]
    const std::string& key, // e.g. "layers.0.mlp.switch_mlp.gate_proj"
    const array& indices,   // expert indices
    bool sorted) {
  // gather_mm(x, W^T, nullopt, indices, sorted)
  // W is [E, out, in], W^T is [E, in, out]
  return mlx::core::gather_mm(
      x,
      get_weight_t3d(key + ".weight"),
      std::nullopt,  // no scales (dense)
      indices,
      sorted);
}

// Switch linear forward (auto-dispatch quantized vs dense, auto-detect bits)
array switch_linear_fwd(
    const array& x,
    const std::string& prefix,
    const array& indices,
    bool sorted,
    bool is_quant, int /*gs_hint*/, int /*bits_hint*/, const std::string& /*mode_hint*/) {
  if (is_quant && has_weight(prefix + ".scales")) {
    const auto& w = get_weight(prefix + ".weight");
    const auto& scales = get_weight(prefix + ".scales");
    std::optional<array> biases = std::nullopt;
    if (has_weight(prefix + ".biases")) {
      biases = get_weight(prefix + ".biases");
    }
    // Infer bits from weight/scales shape ratio (supports mixed-bit recipes)
    // For 3D switch weights: w=[E, out, in_packed], scales=[E, out, in/gs]
    int w_cols = w.shape(-1);
    int s_cols = scales.shape(-1);
    int gs = 64;
    int original_cols = s_cols * gs;
    int bits = (w_cols * 32) / original_cols;
    std::string mode = biases.has_value() ? "affine" : "mxfp8";
    if (!biases.has_value()) { gs = 32; bits = 8; }
    return mlx::core::gather_qmm(
        x, w, scales, biases,
        std::nullopt,  // lhs_indices (not used)
        indices,       // rhs_indices (expert indices)
        true,          // transpose
        std::optional<int>(gs),
        std::optional<int>(bits),
        mode,
        sorted);
  }
  return switch_linear_forward(x, prefix, indices, sorted);
}

// =====================================================================
// Gather sort / scatter unsort for efficient expert routing
// =====================================================================

struct GatherSortResult {
  array x_sorted;
  array idx_sorted;
  array inv_order;
};

GatherSortResult gather_sort(const array& x, const array& indices) {
  // indices: [ne, k], x: [ne, 1, 1, D]
  auto idx_shape = indices.shape();
  int m = idx_shape.back();

  auto flat_indices = reshape(indices, {-1});
  auto order = argsort(flat_indices, -1);
  auto inv_order = argsort(order, -1);
  auto idx_sorted = take(flat_indices, order, 0);

  auto x_shape = x.shape();
  int d = x_shape.back();
  auto x_flat = reshape(x, {-1, 1, d});
  auto m_arr = array(m, mlx::core::int32);
  auto token_indices = mlx::core::floor_divide(order, m_arr);
  auto x_sorted = take(x_flat, token_indices, 0);

  return {x_sorted, idx_sorted, inv_order};
}

array scatter_unsort(const array& x, const array& inv_order, const Shape& orig_shape) {
  auto unsorted = take(x, inv_order, 0);
  auto x_shape = unsorted.shape();
  Shape new_shape(orig_shape.begin(), orig_shape.end());
  for (size_t i = 1; i < x_shape.size(); i++) {
    new_shape.push_back(x_shape[i]);
  }
  return reshape(unsorted, new_shape);
}

// =====================================================================
// Sparse MoE block forward
// =====================================================================

array sparse_moe_fn(
    const array& x,        // [B, hidden] — 2D (single-token decode)
    int layer_idx,
    const MoeConfig& cfg,
    const LayerQuantInfo& qi) {
  int B = x.shape(0);
  int hidden = cfg.hidden_size;
  int ne = B;  // single-token decode: seq_len=1, so ne = B
  int k = cfg.num_experts_per_tok;
  int num_exp = cfg.num_experts;

  std::string pfx = "layers." + std::to_string(layer_idx) + ".mlp.";

  // x is already 2D [B, hidden] — no reshape needed for single-token
  auto x_flat = x;

  // Router
  auto router_logits = linear_forward(x_flat, pfx + "gate",
      qi.g_quant, qi.g_gs, qi.g_bits, qi.g_mode);
  auto routing_weights = mlx::core::softmax(router_logits, {-1}, /*precise=*/true);

  // Top-k selection via argpartition
  auto top_indices_full = argpartition(routing_weights, -k, -1);
  auto top_indices = slice(top_indices_full, {0, num_exp - k}, {ne, num_exp});
  auto top_weights = mlx::core::take_along_axis(routing_weights, top_indices, -1);

  // Normalize weights
  if (cfg.norm_topk_prob) {
    auto wsum = sum(top_weights, {-1}, true);
    top_weights = top_weights / wsum;
  }

  // Expand x for gather_mm: [ne, D] → [ne, 1, 1, D]
  auto x_expanded = reshape(x_flat, {ne, 1, 1, hidden});

  std::string sw_pfx = pfx + "switch_mlp.";

  // Threshold for sorted dispatch (matches Rust/Python)
  bool do_sort = (top_indices.size() >= 64);

  array expert_out = zeros({}, mlx::core::bfloat16);
  if (do_sort) {
    auto sorted = gather_sort(x_expanded, top_indices);
    const auto& idx = sorted.idx_sorted;

    auto gate_out = switch_linear_fwd(sorted.x_sorted, sw_pfx + "gate_proj", idx, true,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    auto up_out = switch_linear_fwd(sorted.x_sorted, sw_pfx + "up_proj", idx, true,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    auto activated = swiglu(gate_out, up_out);
    auto result = switch_linear_fwd(activated, sw_pfx + "down_proj", idx, true,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    expert_out = scatter_unsort(result, sorted.inv_order, top_indices.shape());
  } else {
    auto gate_out = switch_linear_fwd(x_expanded, sw_pfx + "gate_proj", top_indices, false,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    auto up_out = switch_linear_fwd(x_expanded, sw_pfx + "up_proj", top_indices, false,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    auto activated = swiglu(gate_out, up_out);
    expert_out = switch_linear_fwd(activated, sw_pfx + "down_proj", top_indices, false,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
  }

  // Squeeze the penultimate dim (from gather_mm output)
  expert_out = squeeze(expert_out, {-2});

  // Weight experts: [ne, k, D] * [ne, k, 1] → sum → [ne, D]
  auto weights_expanded = reshape(top_weights, {ne, k, 1});
  auto weighted = expert_out * weights_expanded;
  auto expert_output = sum(weighted, {1});

  // Shared expert — use linear_proj (auto-detects bits per tensor) since
  // down_proj may have different bits than gate_proj/up_proj (e.g. unsloth recipe)
  std::string se_pfx = pfx + "shared_expert.";
  auto se_gate_in = linear_proj(x_flat, se_pfx + "gate_proj");
  auto se_up_in = linear_proj(x_flat, se_pfx + "up_proj");
  auto se_activated = swiglu(se_gate_in, se_up_in);
  auto shared_out = linear_proj(se_activated, se_pfx + "down_proj");

  // Shared expert gate: sigmoid
  auto shared_gate = linear_forward(x_flat, pfx + "shared_expert_gate",
      qi.sg_quant, qi.sg_gs, qi.sg_bits, qi.sg_mode);
  shared_gate = sigmoid(shared_gate);

  auto shared_contribution = shared_out * shared_gate;
  return expert_output + shared_contribution;
}

// =====================================================================
// Dense MLP forward (for non-MoE layers in MoE model)
// =====================================================================

array dense_mlp_fn(
    const array& x,
    int layer_idx,
    const MoeConfig& cfg,
    const DenseMLPQuantInfo& qi) {
  // Use linear_proj (auto-detects bits per tensor) since down_proj may
  // have different bits than gate_proj/up_proj (e.g. unsloth recipe)
  std::string mp = "layers." + std::to_string(layer_idx) + ".mlp.";
  auto gate = linear_proj(x, mp + "gate_proj");
  auto up   = linear_proj(x, mp + "up_proj");
  auto mlp_out = linear_proj(swiglu(gate, up), mp + "down_proj");
  return mlp_out;
}

// =====================================================================
// Full MoE decode function (40 layers, GDN/attn + MoE/dense MLP)
// =====================================================================

// Determine if layer is MoE (same logic as Rust config.is_moe_layer)
bool is_moe_layer(int layer_idx, const MoeConfig& cfg) {
  if (cfg.num_experts <= 0) return false;
  if (cfg.decoder_sparse_step <= 0) return false;
  // mlp_only_layers check not needed here — handled at init time in layer_quant
  return ((layer_idx + 1) % cfg.decoder_sparse_step) == 0;
}

// =====================================================================
// Attention for compiled path — array offset + static mask
//
// Like attn_pure_fn but:
//   1. fast::rope uses array offset overload (not int)
//   2. slice_update uses array offset (not constant)
//   3. Always uses static additive mask (no dynamic_kv)
// This ensures the graph topology is identical across decode steps.
// =====================================================================

AttnPureResult attn_for_compile(
    const array& x,          // [B, hidden] — 2D
    int layer_idx,
    const array& kv_keys,    // [B, Hkv, max_kv_len, D]
    const array& kv_values,  // [B, Hkv, max_kv_len, D]
    const array& attn_mask,  // [1, 1, 1, max_kv_len] additive mask
    const array& offset_arr, // scalar int32
    const BaseConfig& cfg) {
  int B = x.shape(0);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  // Q projection (2x width for per-head gating)
  auto q_proj = linear_proj(x, pfx + "q_proj");
  if (has_weight(pfx + "q_proj.bias")) q_proj = q_proj + get_weight(pfx + "q_proj.bias");

  auto qph    = reshape(q_proj, {B, 1, cfg.num_heads, cfg.head_dim * 2});
  auto queries = slice(qph, {0, 0, 0, 0},           {B, 1, cfg.num_heads, cfg.head_dim});
  auto gate    = slice(qph, {0, 0, 0, cfg.head_dim}, {B, 1, cfg.num_heads, cfg.head_dim * 2});
  gate = reshape(gate, {B, cfg.num_heads * cfg.head_dim});

  // K, V projections
  auto keys   = linear_proj(x, pfx + "k_proj");
  auto values = linear_proj(x, pfx + "v_proj");
  if (has_weight(pfx + "k_proj.bias")) keys   = keys   + get_weight(pfx + "k_proj.bias");
  if (has_weight(pfx + "v_proj.bias")) values = values + get_weight(pfx + "v_proj.bias");

  keys   = reshape(keys,   {B, 1, cfg.num_kv_heads, cfg.head_dim});
  values = reshape(values, {B, 1, cfg.num_kv_heads, cfg.head_dim});

  // QK norm
  queries = fast::rms_norm(queries, get_weight(pfx + "q_norm.weight"), cfg.rms_norm_eps);
  keys    = fast::rms_norm(keys,    get_weight(pfx + "k_norm.weight"), cfg.rms_norm_eps);

  // RoPE with array offset (graph-safe — no baked-in constant)
  queries = fast::rope(queries, cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset_arr);
  keys    = fast::rope(keys,    cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset_arr);

  // Transpose for SDPA
  queries = transpose(queries, {0, 2, 1, 3});
  keys    = transpose(keys,    {0, 2, 1, 3});
  values  = transpose(values,  {0, 2, 1, 3});

  // KV cache update with array offset
  auto offset_1d = reshape(offset_arr, {1});
  auto new_kv_keys   = mlx::core::slice_update(kv_keys,   keys,   offset_1d, {2});
  auto new_kv_values = mlx::core::slice_update(kv_values, values, offset_1d, {2});

  // SDPA with static additive mask (fixed shapes for compile)
  float scale = std::pow((float)cfg.head_dim, -0.5f);
  auto attn_out = fast::scaled_dot_product_attention(
      queries, new_kv_keys, new_kv_values, scale, "", attn_mask, {});

  // Transpose back + reshape
  attn_out = transpose(attn_out, {0, 2, 1, 3});
  attn_out = reshape(attn_out, {B, cfg.num_heads * cfg.head_dim});

  // Gate
  attn_out = compiled_attn_gate()({attn_out, gate})[0];

  // Output projection
  auto output = linear_proj(attn_out, pfx + "o_proj");
  if (has_weight(pfx + "o_proj.bias")) output = output + get_weight(pfx + "o_proj.bias");

  return {output, new_kv_keys, new_kv_values};
}

// =====================================================================
// Compilable MoE decode function
//
// inputs:  [h, offset_arr, cache[0].a, cache[0].b, ..., cache[N-1].a, cache[N-1].b]
// outputs: [logits, new_offset, new_cache[0].a, ..., new_cache[N-1].b]
//
// All offset-dependent ops use offset_arr (input array, not C++ int),
// so the graph topology is identical across decode steps.
// =====================================================================

static std::vector<array> moe_compiled_decode_fn(const std::vector<array>& inputs) {
  const auto& cfg = g_moe_config;
  auto h = inputs[0];
  auto offset_arr = inputs[1]; // scalar int32

  // Attention mask: [1, 1, 1, max_kv_len]
  // positions <= offset → valid (0.0), positions > offset → masked (-inf)
  int first_fa = cfg.full_attention_interval - 1;
  int max_kv_len = inputs[2 + first_fa * 2].shape(2);
  auto positions = arange(0, max_kv_len, mlx::core::int32);
  auto valid_mask = less_equal(positions, offset_arr);
  auto attn_mask = where(valid_mask,
      array(0.0f, mlx::core::bfloat16),
      array(-std::numeric_limits<float>::infinity(), mlx::core::bfloat16));
  attn_mask = reshape(attn_mask, {1, 1, 1, max_kv_len});

  // Pre-allocate new_caches (placeholders overwritten in loop)
  std::vector<array> new_caches;
  new_caches.reserve(cfg.num_layers * 2);
  for (int i = 0; i < cfg.num_layers * 2; i++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  for (int i = 0; i < cfg.num_layers; i++) {
    bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
    std::string lp = "layers." + std::to_string(i);

    // Pre-norm
    auto normed = fast::rms_norm(h, get_weight(lp + ".input_layernorm.weight"), cfg.rms_norm_eps);

    // Attention (GDN or full)
    if (is_linear) {
      const auto& cs = inputs[2 + i * 2];
      const auto& rs = inputs[2 + i * 2 + 1];
      auto res = gdn_pure_fn(normed, i, cs, rs, cfg);
      h = h + res.output;
      new_caches[i * 2]     = std::move(res.conv_state);
      new_caches[i * 2 + 1] = std::move(res.recurrent_state);
    } else {
      const auto& kk = inputs[2 + i * 2];
      const auto& kv = inputs[2 + i * 2 + 1];
      auto res = attn_for_compile(normed, i, kk, kv, attn_mask, offset_arr, cfg);
      h = h + res.output;
      new_caches[i * 2]     = std::move(res.keys);
      new_caches[i * 2 + 1] = std::move(res.values);
    }

    // Post-norm + MLP
    auto mlp_in = fast::rms_norm(h, get_weight(lp + ".post_attention_layernorm.weight"), cfg.rms_norm_eps);

    bool moe = is_moe_layer(i, cfg);
    if (moe) {
      h = h + sparse_moe_fn(mlp_in, i, cfg, g_layer_quant[i]);
    } else {
      h = h + dense_mlp_fn(mlp_in, i, cfg, g_dense_quant[i]);
    }
  }

  // Final norm + LM head
  h = fast::rms_norm(h, get_weight("final_norm.weight"), cfg.rms_norm_eps);
  if (cfg.tie_word_embeddings) {
    h = linear_proj(h, "embedding");
  } else {
    h = linear_proj(h, "lm_head");
  }

  auto new_offset = offset_arr + array(1, mlx::core::int32);

  std::vector<array> result;
  result.reserve(2 + cfg.num_layers * 2);
  result.push_back(std::move(h));
  result.push_back(std::move(new_offset));
  for (auto& c : new_caches) result.push_back(std::move(c));
  return result;
}

static auto& compiled_moe_decode() {
  static auto fn = mlx::core::compile(moe_compiled_decode_fn);
  return fn;
}

}  // namespace

// =============================================================================
// Public FFI functions
// =============================================================================

extern "C" {

// Initialize MoE forward pass from post-prefill caches.
// moe_params: [num_experts, num_experts_per_tok, norm_topk_prob, decoder_sparse_step]
void mlx_qwen35_moe_init_from_prefill(
    // BaseConfig params
    int num_layers,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rope_theta,
    int rope_dims,
    float rms_norm_eps,
    int full_attention_interval,
    int linear_num_k_heads,
    int linear_num_v_heads,
    int linear_key_head_dim,
    int linear_value_head_dim,
    int linear_conv_kernel_dim,
    int tie_word_embeddings,
    int max_kv_len,
    int batch_size,
    // MoE-specific params
    int num_experts,
    int num_experts_per_tok,
    int norm_topk_prob,
    int decoder_sparse_step,
    // mlp_only_layers (comma-separated indices, or null)
    const int* mlp_only_layers,
    int mlp_only_layers_len,
    // Cache arrays and offset
    mlx_array** cache_arrays,
    int prefill_offset
) {
  try {
    g_moe_config = MoeConfig{{
      num_layers, hidden_size, num_heads, num_kv_heads, head_dim,
      rope_theta, rope_dims, rms_norm_eps, full_attention_interval,
      linear_num_k_heads, linear_num_v_heads, linear_key_head_dim,
      linear_value_head_dim, linear_conv_kernel_dim,
      tie_word_embeddings != 0,
      max_kv_len, batch_size
    }, num_experts, num_experts_per_tok, norm_topk_prob != 0, decoder_sparse_step};

    // Build mlp_only set
    std::unordered_set<int> mlp_only_set;
    if (mlp_only_layers && mlp_only_layers_len > 0) {
      for (int i = 0; i < mlp_only_layers_len; i++) {
        mlp_only_set.insert(mlp_only_layers[i]);
      }
    }

    // Import caches (same pattern as dense init)
    g_moe_caches.clear();
    g_moe_caches.reserve(num_layers * 2);

    for (int i = 0; i < num_layers; i++) {
      bool is_linear = !((i + 1) % full_attention_interval == 0);

      if (is_linear) {
        if (!cache_arrays[i * 2] || !cache_arrays[i * 2 + 1]) {
          g_moe_inited = false;
          return;
        }
        g_moe_caches.push_back(*reinterpret_cast<array*>(cache_arrays[i * 2]));
        g_moe_caches.push_back(*reinterpret_cast<array*>(cache_arrays[i * 2 + 1]));
      } else {
        if (!cache_arrays[i * 2] || !cache_arrays[i * 2 + 1]) {
          g_moe_inited = false;
          return;
        }
        auto& kk = *reinterpret_cast<array*>(cache_arrays[i * 2]);
        auto& kv = *reinterpret_cast<array*>(cache_arrays[i * 2 + 1]);
        int current_cap = kk.shape(2);
        if (current_cap < max_kv_len) {
          int pad_len = max_kv_len - current_cap;
          auto kpad = zeros({batch_size, num_kv_heads, pad_len, head_dim}, kk.dtype());
          auto vpad = zeros({batch_size, num_kv_heads, pad_len, head_dim}, kv.dtype());
          g_moe_caches.push_back(concatenate({kk, kpad}, 2));
          g_moe_caches.push_back(concatenate({kv, vpad}, 2));
        } else {
          g_moe_caches.push_back(kk);
          g_moe_caches.push_back(kv);
        }
      }
    }

    // Detect quantization per-layer by probing for .scales keys
    g_layer_quant.clear();
    g_layer_quant.reserve(num_layers);
    g_dense_quant.clear();
    g_dense_quant.reserve(num_layers);

    auto detect_quant = [](const std::string& prefix) -> std::tuple<bool, int, int, std::string> {
      if (!has_weight(prefix + ".scales")) {
        return {false, 0, 0, ""};
      }
      // Check for MXFP8: no biases, infer from scales shape
      bool has_biases = has_weight(prefix + ".biases");
      if (!has_biases) {
        // MXFP8: group_size=32, bits=8
        return {true, 32, 8, "mxfp8"};
      }
      // Affine: infer bits from weight/scales shape ratio
      const auto& w = get_weight(prefix + ".weight");
      const auto& scales = get_weight(prefix + ".scales");
      int bits = infer_affine_bits(w, scales, 64);
      return {true, 64, bits, "affine"};
    };

    auto detect_gate_quant = [](const std::string& prefix) -> std::tuple<bool, int, int, std::string> {
      if (!has_weight(prefix + ".scales")) {
        return {false, 0, 0, ""};
      }
      // Router gates always use 8-bit affine with group_size=64
      return {true, 64, 8, "affine"};
    };

    for (int i = 0; i < num_layers; i++) {
      std::string pfx = "layers." + std::to_string(i) + ".mlp.";

      // Check if this layer is MoE
      bool moe = (num_experts > 0 && decoder_sparse_step > 0 &&
                  ((i + 1) % decoder_sparse_step) == 0 &&
                  mlp_only_set.count(i) == 0);

      if (moe) {
        auto [sw_q, sw_gs, sw_bits, sw_mode] = detect_quant(pfx + "switch_mlp.gate_proj");
        auto [sh_q, sh_gs, sh_bits, sh_mode] = detect_quant(pfx + "shared_expert.gate_proj");
        auto [g_q, g_gs, g_bits, g_mode] = detect_gate_quant(pfx + "gate");
        auto [sg_q, sg_gs, sg_bits, sg_mode] = detect_gate_quant(pfx + "shared_expert_gate");

        g_layer_quant.push_back(LayerQuantInfo{
          sw_q, sw_gs, sw_bits, sw_mode,
          sh_q, sh_gs, sh_bits, sh_mode,
          g_q, g_gs, g_bits, g_mode,
          sg_q, sg_gs, sg_bits, sg_mode,
        });
        g_dense_quant.push_back(DenseMLPQuantInfo{false, 0, 0, ""});
      } else {
        g_layer_quant.push_back(LayerQuantInfo{});
        auto [dq, dgs, dbits, dmode] = detect_quant(pfx + "gate_proj");
        g_dense_quant.push_back(DenseMLPQuantInfo{dq, dgs, dbits, dmode});
      }
    }

    g_moe_offset_int = prefill_offset;
    g_moe_inited = true;

    // Clear 3D transpose cache
    g_weight_transposes_3d.clear();

    // Break the lazy RNG split chain
    auto rng_key = mlx::core::random::KeySequence::default_().next();
    mlx::core::eval({rng_key});
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_qwen35_moe_init_from_prefill: " << e.what() << std::endl;
    g_moe_inited = false;
  }
}

// MoE single-token decode step (compiled path by default)
void mlx_qwen35_moe_forward(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    mlx_array** output_logits,
    int* cache_offset_out
) {
  if (!g_moe_inited) {
    *output_logits = nullptr;
    return;
  }
  const auto& cfg = g_moe_config;

  try {
    auto& input_ids      = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    // Embedding lookup: [B, 1] → [B, hidden] (2D)
    auto flat_ids = reshape(input_ids, {-1});
    auto h = take(embedding_weight, flat_ids, 0);

    // Build inputs for the compilable function
    std::vector<array> fn_inputs;
    fn_inputs.reserve(2 + cfg.num_layers * 2);
    fn_inputs.push_back(std::move(h));
    fn_inputs.push_back(array(g_moe_offset_int, mlx::core::int32));
    for (const auto& c : g_moe_caches) {
      fn_inputs.push_back(c);
    }

    // MLX_NO_COMPILE=1 disables compilation for A/B testing
    static bool no_compile = std::getenv("MLX_NO_COMPILE") != nullptr;
    auto outputs = no_compile
        ? moe_compiled_decode_fn(fn_inputs)
        : compiled_moe_decode()(fn_inputs);

    // Extract outputs: [logits, new_offset, new_caches...]
    *output_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    g_moe_offset_int++;
    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_moe_caches[i] = outputs[2 + i];
    }

    if (cache_offset_out) {
      *cache_offset_out = g_moe_offset_int;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_forward: %s\n", e.what());
    fflush(stderr);
    *output_logits = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_moe_forward\n");
    fflush(stderr);
    *output_logits = nullptr;
  }
}

// Eval token (+ caches implicitly via dependency graph).
//
// The compiled forward returns [logits, offset, caches...] as outputs of a
// single Compiled primitive. Evaluating the token (which depends on logits)
// triggers the entire compiled graph, materializing all cache arrays too.
// This matches Python's mx.async_eval(y, logprobs) pattern (2 arrays, not 81).
//
// Set MLX_EVAL_ALL_CACHES=1 to revert to eval'ing token + all 80 caches
// explicitly (previous behavior, slightly slower due to scheduling overhead).
void mlx_qwen35_moe_eval_token_and_caches(mlx_array* next_token_ptr) {
  try {
    static bool eval_all = std::getenv("MLX_EVAL_ALL_CACHES") != nullptr;
    if (eval_all) {
      std::vector<array> to_eval;
      to_eval.reserve(1 + g_moe_caches.size());
      to_eval.push_back(*reinterpret_cast<array*>(next_token_ptr));
      for (const auto& c : g_moe_caches) {
        to_eval.push_back(c);
      }
      mlx::core::async_eval(std::move(to_eval));
    } else {
      mlx::core::async_eval({*reinterpret_cast<array*>(next_token_ptr)});
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_eval_token_and_caches: %s\n", e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_moe_eval_token_and_caches\n");
    fflush(stderr);
  }
}

void mlx_qwen35_moe_sync_eval_caches() {
  try {
    if (g_moe_caches.empty()) return;
    std::vector<array> to_eval;
    to_eval.reserve(g_moe_caches.size());
    for (const auto& c : g_moe_caches) {
      to_eval.push_back(c);
    }
    mlx::core::eval(std::move(to_eval));
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in moe_sync_eval_caches: %s\n", e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in moe_sync_eval_caches\n");
    fflush(stderr);
  }
}

// Reset MoE state
void mlx_qwen35_moe_reset() {
  g_moe_caches.clear();
  g_moe_offset_int = 0;
  g_moe_inited = false;
  g_layer_quant.clear();
  g_dense_quant.clear();
  g_weight_transposes_3d.clear();
}

}  // extern "C"
