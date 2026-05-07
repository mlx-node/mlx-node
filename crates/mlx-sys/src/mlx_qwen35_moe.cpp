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

// =====================================================================
// Phase 4 piece 1: paged-decode globals.
//
// Independent from the flat-path globals above so both compile graphs can
// coexist while piece 2 wires the Rust caller. Sized at init-time from
// the active `MoeConfig.num_layers`.
//
// `g_paged_inited` gates the new `mlx_qwen35_moe_forward_paged` FFI. Until
// piece 2 swaps Rust callers over, `mlx_qwen35_moe_init_paged` is the only
// way for any of these to flip true.
//
// Layout (size = num_layers; one entry per layer indexed by `layer_idx`):
//   - g_k_pools / g_v_pools / g_k_scales / g_v_scales: meaningful only for
//     full-attention layers. Linear-layer slots hold a small placeholder
//     `zeros({}, bf16)` array — they're never read by the paged graph.
//   - g_paged_linear_caches: size = num_layers * 2. Slot `2i` and `2i+1`
//     hold conv_state and recurrent_state for layer `i` when that layer is
//     linear-attention. Full-attn slots hold placeholders.
//
// This indexed-by-layer design keeps the per-layer dispatch in the
// compile graph as a simple `inputs[base + i*4 + k]` lookup with a single
// `is_linear` branch.
// =====================================================================
static MoeConfig g_paged_config{};
static std::vector<array> g_k_pools;          // [num_layers]
static std::vector<array> g_v_pools;          // [num_layers]
static std::vector<array> g_k_scales;         // [num_layers]
static std::vector<array> g_v_scales;         // [num_layers]
static std::vector<array> g_paged_linear_caches;  // [num_layers * 2]
static int g_paged_offset_int = 0;
static bool g_paged_inited = false;

// Cached 3D transposes for expert weights [E,out,in] → [E,in,out]
static std::unordered_map<std::string, array> g_weight_transposes_3d;

// Pure read — 3D transposes are pre-computed in mlx_qwen35_moe_init_from_prefill.
// Returns by VALUE so the caller's copy survives concurrent map mutations.
array get_weight_t3d(const std::string& name) {
  auto it = g_weight_transposes_3d.find(name);
  if (it != g_weight_transposes_3d.end()) {
    return it->second;  // copy (refcount bump)
  }
  throw std::runtime_error("3D transpose not found for weight: " + name);
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
  auto w = get_weight(prefix + ".weight");
  auto scales = get_weight(prefix + ".scales");
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
    auto w = get_weight(prefix + ".weight");
    auto scales = get_weight(prefix + ".scales");
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

// =====================================================================
// Paged MoE decode function — Phase 4 piece 1.
//
// Mirrors `moe_compiled_decode_fn` structurally but routes full-attention
// layers through `attn_for_compile_paged` (paged_kv_write + paged_attention)
// instead of the flat slice_update + masked SDPA path. Linear (GDN) layers
// take the same `gdn_pure_fn` path the flat graph uses — paged storage only
// applies to full-attention K/V.
//
// Input vector layout (all arrays — order matters for the compile cache):
//   [0]                  h:                  embedded input
//   [1]                  offset_arr:         [1] int32
//   [2]                  block_table:        [1, max_blocks_per_seq] int32
//   [3]                  slot_mapping:       [chunk_size_max] int64
//   [4]                  num_valid_tokens:   [1] int32
//   [5]                  num_valid_blocks:   [1] int32
//   [6]                  seq_lens:           [1] int32
//   For each layer i in [0, num_layers):
//     If linear:
//       [7 + i*4 + 0]    conv_state:         layer's conv state
//       [7 + i*4 + 1]    recurrent_state:    layer's recurrent state
//       [7 + i*4 + 2]    placeholder         (unused — keeps stride uniform)
//       [7 + i*4 + 3]    placeholder         (unused — keeps stride uniform)
//     If full-attention:
//       [7 + i*4 + 0]    k_pool:             paged K storage
//       [7 + i*4 + 1]    v_pool:             paged V storage
//       [7 + i*4 + 2]    k_scale:            [1] f32
//       [7 + i*4 + 3]    v_scale:            [1] f32
//
// Output vector layout:
//   [0]                  logits
//   [1]                  new_offset:         offset_arr + 1
//   For each layer i:
//     If linear:
//       [2 + i*2 + 0]    new_conv_state
//       [2 + i*2 + 1]    new_recurrent_state
//     If full-attention:
//       [2 + i*2 + 0]    new_k_pool          (post-write pool tensor)
//       [2 + i*2 + 1]    new_v_pool          (post-write pool tensor)
//
// The uniform 4-input-per-layer stride keeps the compile cache key stable
// regardless of which layers are linear vs. full-attention; the
// `is_linear` switch is a no-op for the cache because all input shapes
// are fixed at compile time. Output stride is 2-per-layer because scales
// are inputs only — they don't mutate per step.
// =====================================================================
static std::vector<array> moe_compiled_decode_fn_paged(const std::vector<array>& inputs) {
  const auto& cfg = g_paged_config;
  auto h          = inputs[0];
  auto offset_arr = inputs[1];   // [1] int32
  auto block_table      = inputs[2];
  auto slot_mapping     = inputs[3];
  auto num_valid_tokens = inputs[4];
  auto num_valid_blocks = inputs[5];
  auto seq_lens         = inputs[6];

  // Phase 4 piece 1 hard-coded contract.
  constexpr int BLOCK_SIZE = 16;

  constexpr int kHeader = 7;
  constexpr int kPerLayer = 4;

  // Pre-allocate new_caches with placeholders. Output stride = 2 per layer,
  // matching the flat graph (scales are NOT mutated by the forward).
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

    int base = kHeader + i * kPerLayer;
    if (is_linear) {
      const auto& cs = inputs[base + 0];
      const auto& rs = inputs[base + 1];
      auto res = gdn_pure_fn(normed, i, cs, rs, cfg);
      h = h + res.output;
      new_caches[i * 2]     = std::move(res.conv_state);
      new_caches[i * 2 + 1] = std::move(res.recurrent_state);
    } else {
      const auto& k_pool  = inputs[base + 0];
      const auto& v_pool  = inputs[base + 1];
      const auto& k_scale = inputs[base + 2];
      const auto& v_scale = inputs[base + 3];
      auto res = attn_for_compile_paged(
          normed, i,
          k_pool, v_pool,
          k_scale, v_scale,
          offset_arr,
          block_table, slot_mapping,
          num_valid_tokens, num_valid_blocks,
          seq_lens,
          BLOCK_SIZE,
          cfg);
      h = h + res.output;
      // attn_for_compile_paged stashes new_k_pool/new_v_pool in keys/values.
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

static auto& compiled_moe_decode_paged() {
  static auto fn = mlx::core::compile(moe_compiled_decode_fn_paged);
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
      auto w = get_weight(prefix + ".weight");
      auto scales = get_weight(prefix + ".scales");
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

    // Pre-compute 3D transposes for all expert weights [E,out,in] → [E,in,out].
    // This eliminates lazy mutation in get_weight_t3d() during inference.
    g_weight_transposes_3d.clear();
    {
      std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
      for (const auto& [name, w] : g_weights()) {
        if (w.ndim() == 3) {
          g_weight_transposes_3d.insert_or_assign(name, transpose(w, {0, 2, 1}));
        }
      }
    }

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

// Reset MoE state — clears BOTH the legacy flat globals AND the
// Phase 4 piece 1 paged globals. Keeping these symmetric is required
// because `mlx_qwen35_moe_init_paged` flips `g_paged_inited` to true
// independently of `g_moe_inited`; without clearing the paged side here,
// a later `mlx_qwen35_moe_forward_paged()` would pass the init guard
// and reuse stale KV pools / scales / linear caches / offset from a
// previous request or model.
void mlx_qwen35_moe_reset() {
  // Legacy flat-path globals.
  g_moe_caches.clear();
  g_moe_offset_int = 0;
  g_moe_inited = false;
  g_layer_quant.clear();
  g_dense_quant.clear();
  g_weight_transposes_3d.clear();

  // Phase 4 piece 1 paged-path globals.
  g_paged_config = MoeConfig{};
  g_k_pools.clear();
  g_v_pools.clear();
  g_k_scales.clear();
  g_v_scales.clear();
  g_paged_linear_caches.clear();
  g_paged_offset_int = 0;
  g_paged_inited = false;
}

// Export MoE caches for PromptCache reuse.
// Copies cache arrays to caller-provided output pointers (heap-allocated).
// Returns number of arrays exported, or 0 if not initialized.
int mlx_qwen35_moe_export_caches(mlx_array** out_ptrs, int max_count) {
  if (!g_moe_inited || g_moe_caches.empty()) return 0;
  int count = std::min((int)g_moe_caches.size(), max_count);
  for (int i = 0; i < count; i++) {
    // Heap-allocate a copy — MLX arrays are ref-counted internally,
    // so the underlying Metal buffer is shared (not duplicated).
    out_ptrs[i] = reinterpret_cast<mlx_array*>(new array(g_moe_caches[i]));
  }
  return count;
}

// Export paged linear-attention caches for live-session continuation.
//
// The block-paged path keeps full-attention K/V in the Rust adapter pools, but
// compiled paged decode advances GDN conv/recurrent state in
// `g_paged_linear_caches`. Rust needs those arrays back before
// `mlx_qwen35_moe_reset()` clears the globals, otherwise the next turn has to
// replay the whole cached prefix through GDN.
int mlx_qwen35_moe_export_paged_linear_caches(mlx_array** out_ptrs, int max_count) {
  if (!g_paged_inited || g_paged_linear_caches.empty()) return 0;
  int expected = static_cast<int>(g_paged_linear_caches.size());
  int count = std::min(expected, max_count);
  for (int i = 0; i < count; i++) {
    out_ptrs[i] = nullptr;
  }
  for (int layer = 0; layer < g_paged_config.num_layers; layer++) {
    int base = layer * 2;
    if (base + 1 >= count) break;
    bool is_linear = !((layer + 1) % g_paged_config.full_attention_interval == 0);
    if (!is_linear) continue;
    out_ptrs[base] = reinterpret_cast<mlx_array*>(new array(g_paged_linear_caches[base]));
    out_ptrs[base + 1] = reinterpret_cast<mlx_array*>(new array(g_paged_linear_caches[base + 1]));
  }
  return count;
}

int mlx_qwen35_moe_get_paged_cache_offset() {
  return g_paged_offset_int;
}

// Get current MoE cache offset (number of tokens processed).
int mlx_qwen35_moe_get_cache_offset() {
  return g_moe_offset_int;
}

// Adjust MoE cache offset by delta (for VLM M-RoPE position correction).
void mlx_qwen35_moe_adjust_offset(int delta) {
  g_moe_offset_int += delta;
}

// =============================================================================
// Phase 4 piece 1: paged forward FFI.
//
// These coexist alongside `mlx_qwen35_moe_forward` / `_init_from_prefill`
// while piece 2 swaps the Rust callers. Piece 3 deletes the legacy pair.
// =============================================================================

// Initialize the paged MoE forward graph from per-layer pool / scale handles.
//
// Layout contract:
//   - `k_pool_handles[i]`: pointer to a `[num_blocks, num_kv_heads,
//     head_size/x_pack=8, block_size=16, x_pack=8]` bf16 array view.
//     Phase 4 piece 1 uses bf16 (`KvDtype::Bf16`) and `block_size = 16`.
//   - `v_pool_handles[i]`: pointer to a `[num_blocks, num_kv_heads,
//     head_size, block_size=16]` bf16 array view.
//   - `k_scale_handles[i]` / `v_scale_handles[i]`: pointer to `[1]` f32
//     scale placeholders (1.0 in Phase 4; FP8 calibration in Phase 10).
//   - For linear-attention layers (those satisfying
//     `(i + 1) % full_attention_interval != 0`), the corresponding pool /
//     scale slots may be null — they're stored as bf16 zero placeholders
//     and never read by the compiled graph.
//
// `linear_cache_arrays` mirrors `cache_arrays` in `mlx_qwen35_moe_init_from_prefill`
// for linear layers only: pairs of `(conv_state, recurrent_state)` indexed
// by layer. Full-attn slots are ignored. Pass null for the entire array
// to skip linear-cache seeding (e.g. for the smoke test, which uses
// placeholder zeros).
//
// `prefill_offset` becomes the initial `g_paged_offset_int`. `mlp_only_layers`
// follows the same convention as the flat init.
//
// Compile-graph configuration (encoded into the new FFI's signature):
//   - block_size       = 16
//   - kv_dtype         = Bf16
//   - x_pack           = 8
//   - sliding_window   = 0
//
// `max_blocks_per_seq` and `chunk_size_max` define the fixed-shape input
// dimensions threaded into `PagedAttentionInputs` (see `mlx_common.h`).
// They become part of the compile-cache key — re-tracing with different
// values yields a new compiled graph.
//
// Returns 0 on success; -1 on failure. On failure `g_paged_inited` is
// cleared and a stderr diagnostic is emitted; the Rust caller MUST
// inspect the return value and fall back to the pure-Rust paged path
// rather than entering the compiled paged decode (which would dispatch
// against uninitialized globals).
int32_t mlx_qwen35_moe_init_paged(
    // BaseConfig params (mirrors the flat init)
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
    // MoE-specific params (mirrors the flat init)
    int num_experts,
    int num_experts_per_tok,
    int norm_topk_prob,
    int decoder_sparse_step,
    const int* mlp_only_layers,
    int mlp_only_layers_len,
    // Per-layer paged storage. Each pointer array is sized `num_layers`.
    // Linear-layer slots may be null (stored as bf16 zero placeholders).
    mlx_array** k_pool_handles,
    mlx_array** v_pool_handles,
    mlx_array** k_scale_handles,
    mlx_array** v_scale_handles,
    // Per-layer linear-attention caches: 2 entries per layer
    // (conv_state, recurrent_state). Full-attn slots are ignored. May be
    // null entirely to skip seeding (then linear-layer slots are
    // placeholder zeros — graph builds but produces meaningless
    // recurrence output).
    mlx_array** linear_cache_arrays,
    int prefill_offset
) {
  try {
    g_paged_config = MoeConfig{{
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

    // Reset the paged globals.
    g_k_pools.clear();
    g_v_pools.clear();
    g_k_scales.clear();
    g_v_scales.clear();
    g_paged_linear_caches.clear();
    g_k_pools.reserve(num_layers);
    g_v_pools.reserve(num_layers);
    g_k_scales.reserve(num_layers);
    g_v_scales.reserve(num_layers);
    g_paged_linear_caches.reserve(num_layers * 2);

    auto bf16_placeholder = []() { return zeros({}, mlx::core::bfloat16); };
    auto f32_placeholder  = []() { return array(1.0f, mlx::core::float32); };

    for (int i = 0; i < num_layers; i++) {
      bool is_linear = !((i + 1) % full_attention_interval == 0);

      // Pool / scale slots: meaningful for full-attn layers only.
      if (!is_linear) {
        if (!k_pool_handles || !v_pool_handles ||
            !k_scale_handles || !v_scale_handles ||
            !k_pool_handles[i] || !v_pool_handles[i] ||
            !k_scale_handles[i] || !v_scale_handles[i]) {
          g_paged_inited = false;
          std::cerr << "[MLX] mlx_qwen35_moe_init_paged: missing pool/scale handle for full-attn layer " << i << std::endl;
          return -1;
        }
        g_k_pools.push_back(*reinterpret_cast<array*>(k_pool_handles[i]));
        g_v_pools.push_back(*reinterpret_cast<array*>(v_pool_handles[i]));
        g_k_scales.push_back(*reinterpret_cast<array*>(k_scale_handles[i]));
        g_v_scales.push_back(*reinterpret_cast<array*>(v_scale_handles[i]));
      } else {
        // Linear layer: stash placeholders so per-layer indexing works.
        g_k_pools.push_back(bf16_placeholder());
        g_v_pools.push_back(bf16_placeholder());
        g_k_scales.push_back(f32_placeholder());
        g_v_scales.push_back(f32_placeholder());
      }

      // Linear caches: meaningful for linear-attn layers only.
      if (is_linear && linear_cache_arrays &&
          linear_cache_arrays[i * 2] && linear_cache_arrays[i * 2 + 1]) {
        g_paged_linear_caches.push_back(*reinterpret_cast<array*>(linear_cache_arrays[i * 2]));
        g_paged_linear_caches.push_back(*reinterpret_cast<array*>(linear_cache_arrays[i * 2 + 1]));
      } else {
        g_paged_linear_caches.push_back(bf16_placeholder());
        g_paged_linear_caches.push_back(bf16_placeholder());
      }
    }

    // Detect quantization per-layer (same logic as flat init). The paged
    // graph re-uses the existing g_layer_quant / g_dense_quant arrays
    // populated by the flat init (or any compatible paged seeding); we
    // re-detect here so the paged FFI can be called standalone.
    g_layer_quant.clear();
    g_layer_quant.reserve(num_layers);
    g_dense_quant.clear();
    g_dense_quant.reserve(num_layers);

    auto detect_quant = [](const std::string& prefix) -> std::tuple<bool, int, int, std::string> {
      if (!has_weight(prefix + ".scales")) {
        return {false, 0, 0, ""};
      }
      bool has_biases = has_weight(prefix + ".biases");
      if (!has_biases) {
        return {true, 32, 8, "mxfp8"};
      }
      auto w = get_weight(prefix + ".weight");
      auto scales = get_weight(prefix + ".scales");
      int bits = infer_affine_bits(w, scales, 64);
      return {true, 64, bits, "affine"};
    };

    auto detect_gate_quant = [](const std::string& prefix) -> std::tuple<bool, int, int, std::string> {
      if (!has_weight(prefix + ".scales")) {
        return {false, 0, 0, ""};
      }
      return {true, 64, 8, "affine"};
    };

    for (int i = 0; i < num_layers; i++) {
      std::string pfx = "layers." + std::to_string(i) + ".mlp.";
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

    g_paged_offset_int = prefill_offset;

    // Pre-compute 3D transposes for expert weights (same as flat init).
    g_weight_transposes_3d.clear();
    {
      std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
      for (const auto& [name, w] : g_weights()) {
        if (w.ndim() == 3) {
          g_weight_transposes_3d.insert_or_assign(name, transpose(w, {0, 2, 1}));
        }
      }
    }

    // Defense-in-depth: surface layout / dtype / Metal-availability
    // failures HERE (init time) rather than letting them blow up inside
    // the first `mlx_qwen35_moe_forward_paged` call where the Rust
    // caller's `record_tokens` has already mutated adapter state. We
    // force-eval every full-attn pool / scale handle so the bf16 / f32
    // layouts are materialized on the Metal queue and any underlying
    // allocation failure raises a c++ exception we catch below.
    {
      std::vector<array> probe;
      probe.reserve(num_layers * 4 + 1);
      for (int i = 0; i < num_layers; i++) {
        bool is_linear = !((i + 1) % full_attention_interval == 0);
        if (is_linear) continue;
        // Validate dtype contract: pools must be bf16, scales must be f32.
        if (g_k_pools[i].dtype() != mlx::core::bfloat16) {
          std::cerr << "[MLX] mlx_qwen35_moe_init_paged: layer " << i
                    << " k_pool dtype != bf16" << std::endl;
          g_paged_inited = false;
          return -1;
        }
        if (g_v_pools[i].dtype() != mlx::core::bfloat16) {
          std::cerr << "[MLX] mlx_qwen35_moe_init_paged: layer " << i
                    << " v_pool dtype != bf16" << std::endl;
          g_paged_inited = false;
          return -1;
        }
        if (g_k_scales[i].dtype() != mlx::core::float32) {
          std::cerr << "[MLX] mlx_qwen35_moe_init_paged: layer " << i
                    << " k_scale dtype != f32" << std::endl;
          g_paged_inited = false;
          return -1;
        }
        if (g_v_scales[i].dtype() != mlx::core::float32) {
          std::cerr << "[MLX] mlx_qwen35_moe_init_paged: layer " << i
                    << " v_scale dtype != f32" << std::endl;
          g_paged_inited = false;
          return -1;
        }
        probe.push_back(g_k_pools[i]);
        probe.push_back(g_v_pools[i]);
        probe.push_back(g_k_scales[i]);
        probe.push_back(g_v_scales[i]);
      }
      // Break the lazy RNG split chain, and force-eval the pool / scale
      // layout in the same batch so any Metal-allocation or layout error
      // throws here.
      auto rng_key = mlx::core::random::KeySequence::default_().next();
      probe.push_back(rng_key);
      mlx::core::eval(std::move(probe));
    }

    g_paged_inited = true;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_qwen35_moe_init_paged: " << e.what() << std::endl;
    g_paged_inited = false;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] mlx_qwen35_moe_init_paged: unknown exception" << std::endl;
    g_paged_inited = false;
    return -1;
  }
}

// Single-token paged decode step. Inputs (PagedAttentionInputs) come from
// the Rust adapter's `build_paged_attention_inputs` (Phase 3); the per-
// layer pool/scale globals come from `mlx_qwen35_moe_init_paged`.
//
// PHASE 4 PIECE 1 CONTRACT: This FFI is decode-only — `input_ids` MUST
// have exactly one element and `slot_mapping` MUST be `[1]`. Chunked
// prefill (multi-token) goes through the legacy flat path until piece 2
// lands a separate paged-prefill helper. The contract is enforced
// explicitly: violating it returns null logits without modifying global
// state, so the Rust caller can fall back cleanly. See the docstring on
// `attn_for_compile_paged` in `mlx_qwen35_common.h` for the full
// rationale.
//
// `output_logits` receives a heap-allocated `mlx_array*` (caller owns).
// `cache_offset_out` receives the post-step offset (== prefill_offset + 1 + n
// after n successful calls).
void mlx_qwen35_moe_forward_paged(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    mlx_array* offset_arr_ptr,
    mlx_array* block_table_ptr,
    mlx_array* slot_mapping_ptr,
    mlx_array* num_valid_tokens_ptr,
    mlx_array* num_valid_blocks_ptr,
    mlx_array* seq_lens_ptr,
    mlx_array** output_logits,
    int* cache_offset_out
) {
  if (!g_paged_inited) {
    if (output_logits) *output_logits = nullptr;
    return;
  }
  if (!input_ids_ptr || !embedding_weight_ptr || !output_logits ||
      !offset_arr_ptr || !block_table_ptr || !slot_mapping_ptr ||
      !num_valid_tokens_ptr || !num_valid_blocks_ptr || !seq_lens_ptr) {
    if (output_logits) *output_logits = nullptr;
    return;
  }
  const auto& cfg = g_paged_config;

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);
    auto& offset_arr       = *reinterpret_cast<array*>(offset_arr_ptr);
    auto& block_table      = *reinterpret_cast<array*>(block_table_ptr);
    auto& slot_mapping     = *reinterpret_cast<array*>(slot_mapping_ptr);
    auto& num_valid_tokens = *reinterpret_cast<array*>(num_valid_tokens_ptr);
    auto& num_valid_blocks = *reinterpret_cast<array*>(num_valid_blocks_ptr);
    auto& seq_lens         = *reinterpret_cast<array*>(seq_lens_ptr);

    // Phase 4 piece 1 contract: single-token decode only.
    //
    // `attn_for_compile_paged` builds new_k / new_v with shape
    // `[1, num_kv_heads, head_size]` and feeds `slot_mapping` directly
    // into `paged_kv_write`, which requires
    // `slot_mapping.shape(0) == new_k.shape(0)`. Chunked-prefill (B > 1)
    // is reserved for piece 2's transfer step; until then, reject any
    // caller that tries to push more than one token through the paged
    // FFI. Returning null here matches the existing error contract and
    // lets the Rust caller fall back to the flat path cleanly.
    if (input_ids.size() != 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_moe_forward_paged: phase 4 piece 1 contract "
              "violated — input_ids.size() = %lld, expected 1 (decode-only)\n",
              static_cast<long long>(input_ids.size()));
      fflush(stderr);
      *output_logits = nullptr;
      return;
    }
    if (slot_mapping.ndim() != 1 || slot_mapping.shape(0) != 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_moe_forward_paged: phase 4 piece 1 contract "
              "violated — slot_mapping shape must be [1], got ndim=%d, "
              "shape[0]=%lld\n",
              slot_mapping.ndim(),
              slot_mapping.ndim() >= 1 ? static_cast<long long>(slot_mapping.shape(0)) : -1LL);
      fflush(stderr);
      *output_logits = nullptr;
      return;
    }

    // Embedding lookup: [B, 1] → [B, hidden] (2D)
    auto flat_ids = reshape(input_ids, {-1});
    auto h = take(embedding_weight, flat_ids, 0);

    // Build inputs for the compilable paged function.
    std::vector<array> fn_inputs;
    fn_inputs.reserve(7 + cfg.num_layers * 4);
    fn_inputs.push_back(std::move(h));
    fn_inputs.push_back(offset_arr);
    fn_inputs.push_back(block_table);
    fn_inputs.push_back(slot_mapping);
    fn_inputs.push_back(num_valid_tokens);
    fn_inputs.push_back(num_valid_blocks);
    fn_inputs.push_back(seq_lens);
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      if (is_linear) {
        fn_inputs.push_back(g_paged_linear_caches[i * 2]);
        fn_inputs.push_back(g_paged_linear_caches[i * 2 + 1]);
        fn_inputs.push_back(g_k_scales[i]);   // unused placeholder
        fn_inputs.push_back(g_v_scales[i]);   // unused placeholder
      } else {
        fn_inputs.push_back(g_k_pools[i]);
        fn_inputs.push_back(g_v_pools[i]);
        fn_inputs.push_back(g_k_scales[i]);
        fn_inputs.push_back(g_v_scales[i]);
      }
    }

    static bool no_compile = std::getenv("MLX_NO_COMPILE") != nullptr;
    auto outputs = no_compile
        ? moe_compiled_decode_fn_paged(fn_inputs)
        : compiled_moe_decode_paged()(fn_inputs);

    // Extract: [logits, new_offset, new_caches...]
    *output_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    g_paged_offset_int++;
    // Stash post-step caches back into the per-layer slots. Linear layers
    // get conv/recurrent state; full-attn layers get the post-write pool
    // tensor (functionally aliased to the input pool but we update the
    // slot anyway so the next call's dependency edge flows correctly).
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      auto& a = outputs[2 + i * 2];
      auto& b = outputs[2 + i * 2 + 1];
      if (is_linear) {
        g_paged_linear_caches[i * 2]     = a;
        g_paged_linear_caches[i * 2 + 1] = b;
      } else {
        g_k_pools[i] = a;
        g_v_pools[i] = b;
      }
    }

    if (cache_offset_out) {
      *cache_offset_out = g_paged_offset_int;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_forward_paged: %s\n", e.what());
    fflush(stderr);
    *output_logits = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_moe_forward_paged\n");
    fflush(stderr);
    *output_logits = nullptr;
  }
}

// =============================================================================
// Phase 4 piece 1 test helper: build the paged-attention graph in isolation
// and force-evaluate it.
//
// Unlike `mlx_qwen35_moe_forward_paged`, which traces the full MoE decode
// graph (all 40 layers + MoE routing + LM head) and therefore needs the
// FULL weight set registered, this helper exercises ONLY
// `attn_for_compile_paged` for layer 0. It self-registers a minimal
// synthetic weight set (q/k/v/o + q_norm/k_norm), constructs synthetic
// inputs at the contract shapes, calls the helper, eval()s the output to
// force kernel dispatch (proving paged_kv_write + paged_attention bind
// and run), then clears the synthetic weights.
//
// Return codes:
//   0  — success (graph built, eval() succeeded, kernels dispatched).
//  -1  — Metal not available on this host. The Rust test treats this as a
//        clean skip (no Metal device → can't synthesize the dispatch).
//  -2  — any other failure (graph construction, eval, weight registration,
//        unknown exception). The Rust test treats this as a HARD FAILURE.
//
// Splitting -1 (no-Metal skip) from -2 (real failure) prevents a broken
// `paged_kv_write`/`paged_attention` binding from silently passing on a
// Metal-equipped host: cargo hides passing-test stderr by default, so a
// single "non-zero return" code lumped both cases together and the
// originally weakened test would have accepted any failure as success.
//
// This is the "graph build smoke" coverage that Codex's Finding 3 asked
// for — the existing forward_paged smoke test fails inside the LM-head /
// embedding lookup BEFORE reaching `attn_for_compile_paged`, so without
// this helper the paged graph itself is never exercised.
//
// IMPORTANT: This helper writes to `g_weights()`, so callers MUST invoke
// `mlx_clear_weights()` before/after if any other model state is loaded.
// The Rust test wrapper does both explicitly.
int mlx_qwen35_moe_trace_paged_attn_helper() {
  // Fast path: if Metal isn't available, the paged kernels can't dispatch
  // at all. Surface a distinct return code so the Rust test skips cleanly
  // instead of conflating this with a graph/eval failure.
  if (!mlx::core::metal::is_available()) {
    return -1;
  }
  // Hard-coded contract shapes mirror the smoke-test config.
  constexpr int B               = 1;
  constexpr int NUM_HEADS       = 16;
  constexpr int NUM_KV_HEADS    = 2;
  constexpr int HEAD_DIM        = 128;
  constexpr int HIDDEN          = NUM_HEADS * HEAD_DIM;
  constexpr int Q_OUT           = NUM_HEADS * HEAD_DIM * 2;  // 2x for gating
  constexpr int KV_OUT          = NUM_KV_HEADS * HEAD_DIM;
  constexpr int BLOCK_SIZE      = 16;
  constexpr int X_PACK          = 8;
  constexpr int NUM_BLOCKS      = 4;
  constexpr int MAX_BLOCKS_PER_SEQ = NUM_BLOCKS;
  constexpr int LAYER_IDX       = 0;
  constexpr float ROPE_THETA    = 100000.0f;
  constexpr int   ROPE_DIMS     = 32;
  constexpr float RMS_NORM_EPS  = 1e-6f;

  try {
    // ---- Self-register synthetic weights for layer 0 self-attention ----
    //
    // Weight shapes follow the standard MLX layout `[out_features, in_features]`
    // (mlx_store_weight auto-transposes 2D weights for matmul use).
    auto rms_w = [](int dim) {
      return mlx::core::ones({dim}, mlx::core::bfloat16);
    };
    auto proj_w = [](int out_features, int in_features) {
      // Small constants keep the result finite; exact values irrelevant —
      // we're testing graph wiring + kernel dispatch, not numerical
      // correctness.
      return mlx::core::full({out_features, in_features}, 0.01f, mlx::core::bfloat16);
    };

    // Acquire the writer lock once for the whole synthetic-weight bundle.
    {
      std::unique_lock<std::shared_mutex> lock(g_weights_mutex());
      auto store = [](const std::string& name, array w) {
        g_weights().insert_or_assign(name, w);
        if (w.ndim() == 2) {
          g_weight_transposes().insert_or_assign(name, transpose(w));
        }
      };
      const std::string pfx = "layers." + std::to_string(LAYER_IDX) + ".self_attn.";
      store(pfx + "q_proj.weight", proj_w(Q_OUT,  HIDDEN));
      store(pfx + "k_proj.weight", proj_w(KV_OUT, HIDDEN));
      store(pfx + "v_proj.weight", proj_w(KV_OUT, HIDDEN));
      store(pfx + "o_proj.weight", proj_w(HIDDEN, NUM_HEADS * HEAD_DIM));
      store(pfx + "q_norm.weight", rms_w(HEAD_DIM));
      store(pfx + "k_norm.weight", rms_w(HEAD_DIM));
    }

    // ---- Synthetic inputs at the piece-1 contract shapes ----
    auto x = mlx::core::zeros({B, HIDDEN}, mlx::core::bfloat16);

    auto k_pool = mlx::core::zeros(
        {NUM_BLOCKS, NUM_KV_HEADS, HEAD_DIM / X_PACK, BLOCK_SIZE, X_PACK},
        mlx::core::bfloat16);
    auto v_pool = mlx::core::zeros(
        {NUM_BLOCKS, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE},
        mlx::core::bfloat16);
    auto k_scale = mlx::core::array(1.0f, mlx::core::float32);
    auto v_scale = mlx::core::array(1.0f, mlx::core::float32);

    // RoPE offset, block_table, slot_mapping, valid counts, seq_lens —
    // all at the piece-1 single-token contract shapes.
    int32_t offset_buf[1]      = {0};
    int32_t block_table_buf[MAX_BLOCKS_PER_SEQ];
    for (int i = 0; i < MAX_BLOCKS_PER_SEQ; i++) block_table_buf[i] = (i == 0) ? 0 : -1;
    int64_t slot_mapping_buf[1]   = {0};
    int32_t num_valid_tokens_buf[1] = {1};
    int32_t num_valid_blocks_buf[1] = {1};
    int32_t seq_lens_buf[1]         = {1};

    auto offset_arr = mlx::core::array(
        offset_buf, mlx::core::Shape{1}, mlx::core::int32);
    auto block_table = mlx::core::array(
        block_table_buf, mlx::core::Shape{1, MAX_BLOCKS_PER_SEQ}, mlx::core::int32);
    auto slot_mapping = mlx::core::array(
        slot_mapping_buf, mlx::core::Shape{1}, mlx::core::int64);
    auto num_valid_tokens = mlx::core::array(
        num_valid_tokens_buf, mlx::core::Shape{1}, mlx::core::int32);
    auto num_valid_blocks = mlx::core::array(
        num_valid_blocks_buf, mlx::core::Shape{1}, mlx::core::int32);
    auto seq_lens = mlx::core::array(
        seq_lens_buf, mlx::core::Shape{1}, mlx::core::int32);

    BaseConfig cfg{};
    cfg.num_layers              = 1;
    cfg.hidden_size             = HIDDEN;
    cfg.num_heads               = NUM_HEADS;
    cfg.num_kv_heads            = NUM_KV_HEADS;
    cfg.head_dim                = HEAD_DIM;
    cfg.rope_theta              = ROPE_THETA;
    cfg.rope_dims               = ROPE_DIMS;
    cfg.rms_norm_eps            = RMS_NORM_EPS;
    cfg.full_attention_interval = 1;
    cfg.linear_num_k_heads      = 0;
    cfg.linear_num_v_heads      = 0;
    cfg.linear_key_head_dim     = 0;
    cfg.linear_value_head_dim   = 0;
    cfg.linear_conv_kernel_dim  = 0;
    cfg.tie_word_embeddings     = false;
    cfg.max_kv_len              = 0;
    cfg.batch_size              = B;

    // ---- Build the paged-attention graph ----
    auto res = attn_for_compile_paged(
        x, LAYER_IDX,
        k_pool, v_pool, k_scale, v_scale,
        offset_arr,
        block_table, slot_mapping,
        num_valid_tokens, num_valid_blocks,
        seq_lens,
        BLOCK_SIZE,
        cfg);

    // ---- Force evaluation so paged_kv_write + paged_attention actually
    // dispatch on the Metal queue. Graph construction alone is lazy;
    // without eval() this could pass even if a kernel binding broke.
    mlx::core::eval({res.output, res.keys, res.values});

    // ---- Clean up synthetic weights so concurrent state stays clean ----
    {
      std::unique_lock<std::shared_mutex> lock(g_weights_mutex());
      const std::string pfx = "layers." + std::to_string(LAYER_IDX) + ".self_attn.";
      for (const auto& suffix : {
               "q_proj.weight", "k_proj.weight", "v_proj.weight",
               "o_proj.weight", "q_norm.weight", "k_norm.weight",
           }) {
        std::string key = pfx + suffix;
        g_weights().erase(key);
        g_weight_transposes().erase(key);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_moe_trace_paged_attn_helper: %s\n",
            e.what());
    fflush(stderr);
    return -2;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_moe_trace_paged_attn_helper\n");
    fflush(stderr);
    return -2;
  }
}

}  // extern "C"
