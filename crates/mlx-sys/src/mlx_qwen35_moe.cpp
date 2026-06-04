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

// MTP — last post-final-norm hidden of the LAST committed token,
// stashed at the LM-head boundary inside `moe_compiled_decode_fn` BEFORE
// the `lm_head` linear runs. Shape is `[B, hidden_size]` after the
// per-row final_norm (decode batch is `[1, 1]` → `[1, hidden]`).
// The MTP seeding path consumes this via
// `mlx_qwen35_moe_export_last_hidden` to get the first draft step's
// `prev_hidden` without re-running the main MoE forward.
//
// Cleared on `mlx_qwen35_moe_reset` so cross-turn stale handles never
// leak. The `std::optional<array>` distinguishes "never populated" from
// "populated with zeros".
static std::optional<array> g_moe_last_hidden;

// MTP — snapshot of the GDN linear-attention caches (conv_state +
// recurrent_state) plus the decode offset taken BEFORE the verify FFI
// runs its D+1 sequential MoE forwards. The verify loop mutates
// `g_moe_caches` in place; for linear-attention layers the recurrent /
// conv state advances through D+1 tokens that may or may not all be
// accepted. On rejection we restore this snapshot so the linear state
// matches the pre-verify "after Step A" state; the caller then replays
// exactly K accepted drafts via the main MoE forward FFI to bring the
// linear state forward through the committed drafts only. Mirrors the
// dense-path snapshot (`g_compiled_linear_snapshot` in mlx_qwen35.cpp).
//
// Cleared on `mlx_qwen35_moe_reset`.
static std::vector<array> g_moe_linear_snapshot;
static int g_moe_linear_snapshot_offset = 0;
static bool g_moe_linear_snapshot_taken = false;

// =====================================================================
// Paged-decode globals.
//
// Independent from the flat-path globals above so both compile graphs can
// coexist. Sized at init-time from the active `MoeConfig.num_layers`.
//
// `g_paged_inited` gates the `mlx_qwen35_moe_forward_paged` FFI;
// `mlx_qwen35_moe_init_paged` flips it true.
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

// Switch linear forward (auto-dispatch quantized vs dense).
//
// Dispatch order:
//   1. Registry hit → use Rust-authoritative (mode, bits, group_size).
//   2. Registry miss → infer from companion-tensor presence (legacy
//      heuristic). The hint args are kept for ABI stability with
//      existing callers but ignored — the registry is the source of truth.
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

    int gs;
    int bits;
    std::string mode;
    if (auto info = lookup_quant_info(prefix)) {
      gs = info->group_size;
      bits = info->bits;
      mode = info->mode;
    } else {
      // Legacy heuristic. Shape-based bit inference handles mixed-bit affine
      // recipes; no-biases means MXFP8 by convention.
      int w_cols = w.shape(-1);
      int s_cols = scales.shape(-1);
      gs = 64;
      int original_cols = s_cols * gs;
      bits = (w_cols * 32) / original_cols;
      mode = biases.has_value() ? "affine" : "mxfp8";
      if (!biases.has_value()) { gs = 32; bits = 8; }
    }

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
  // MTP — emit the post-final-norm hidden of the last decoded token
  // alongside logits so `mlx_qwen35_moe_forward` can stash it into
  // `g_moe_last_hidden`. This function is wrapped in `mlx::core::compile`,
  // so a global assignment inside the body would only fire at trace time;
  // returning the hidden as an extra output threads it through the
  // compiled graph on every call. Shape after the per-row final_norm is
  // `[1, hidden_size]` for the flat-path decode (batch=1, single token).
  array last_hidden = h;
  if (cfg.tie_word_embeddings) {
    h = linear_proj(h, "embedding");
  } else {
    h = linear_proj(h, "lm_head");
  }

  auto new_offset = offset_arr + array(1, mlx::core::int32);

  std::vector<array> result;
  result.reserve(3 + cfg.num_layers * 2);
  result.push_back(std::move(h));
  result.push_back(std::move(new_offset));
  result.push_back(std::move(last_hidden));
  for (auto& c : new_caches) result.push_back(std::move(c));
  return result;
}

// Dispatch-table type for the compiled MoE graphs. `BatchedVerifyFn` itself
// lives in mlx_qwen35.cpp's anonymous namespace (a separate translation
// unit), so we re-declare the identical signature here. All three MoE
// compiled bodies share `std::vector<array>(const std::vector<array>&)`.
using MoeCompiledFn = std::function<std::vector<array>(const std::vector<array>&)>;

// Resettable file-scope globals. These compiled MoE graphs read expert +
// attention weights via `get_weight(...)` INSIDE the traced closure, so
// `mlx::core::compile(...)` bakes the captured weight arrays into a
// process-lifetime cache. On a model RELOAD they would be stale, so a
// second same-shape model would decode/verify with the FIRST model's
// weights (silent corruption). The slots start empty and are lazily
// assigned `compile(...)` on first call (on the inference thread, under the
// Rust `COMPILED_WEIGHTS_RWLOCK.read()`); `mlx_qwen35_moe_invalidate_compiled_graphs()`
// nulls them under the write lock so the next call re-traces against the live
// registry. (Empty-function default is well-formed; `array` has no default
// ctor but `std::function<>` does.)
static MoeCompiledFn g_moe_decode_compiled;
static MoeCompiledFn g_moe_verify_batched_compiled;
static MoeCompiledFn g_moe_decode_paged_compiled;

// Resettable global cleared by `mlx_qwen35_moe_invalidate_compiled_graphs()`
// on reload — see the `g_moe_decode_compiled` declaration above. This is the
// MoE AR-decode graph.
static MoeCompiledFn& compiled_moe_decode() {
  if (!g_moe_decode_compiled) {
    g_moe_decode_compiled = compile_resettable_weight_graph(moe_compiled_decode_fn);
  }
  return g_moe_decode_compiled;
}

// =====================================================================
// Paged MoE decode function.
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

  // Hard-coded contract.
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

// =====================================================================
// MoE batched verify decode graph.
//
// Direct MoE mirror of `qwen35_verify_batched_decode_fn<false>` in
// `mlx_qwen35.cpp` (the MoE path doesn't ship tape-replay). One forward
// over `T = depth + 1` tokens replacing D+1 sequential
// `mlx_qwen35_moe_forward` calls.
//
// Input vector layout (matches the per-step MoE graph plus the 3D h):
//   [0]                  h_3d         [B, T, hidden]  bf16
//   [1]                  offset_arr   [1]             int32
//   For each layer i in [0, num_layers):
//     [2 + i*2 + 0]      cache_a      (k for full-attn, conv_state for linear)
//     [2 + i*2 + 1]      cache_b      (v for full-attn, recurrent_state for linear)
//
// Output vector layout:
//   [0]                  logits       [B, T, vocab]
//   [1]                  new_offset_arr — `offset_arr + T`
//   [2]                  hiddens      [B, T, hidden]  pre-lm_head hidden
//   For each layer i:
//     [3 + i*2 + 0]      new_cache_a
//     [3 + i*2 + 1]      new_cache_b
//
// `last_hidden` is kept at slot 2 to mirror the per-step graph layout —
// the FFI consumes it independently of the cache stride.
// =====================================================================
static std::vector<array> moe_verify_batched_decode_fn(
    const std::vector<array>& inputs) {
  const auto& cfg = g_moe_config;
  auto h          = inputs[0];        // [B, T, hidden]
  auto offset_arr = inputs[1];        // [1] int32

  int B = h.shape(0);
  int T = h.shape(1);
  int hidden = h.shape(2);

  // Tail-causal mask `[1, 1, T, max_kv_len]`: at query row `t`, valid
  // keys are `[0..offset + t]`. Built once for all full-attention layers.
  int first_fa = cfg.full_attention_interval - 1;
  int max_kv_len = inputs[2 + first_fa * 2].shape(2);
  auto col_positions = arange(0, max_kv_len, mlx::core::int32);   // [K]
  auto row_idx       = arange(0, T, mlx::core::int32);            // [T]
  auto col_row = reshape(col_positions, {1, max_kv_len});         // [1, K]
  auto row_col = reshape(row_idx, {T, 1}) + offset_arr;           // [T, 1]
  auto valid_2d = less_equal(col_row, row_col);                   // [T, K]
  auto attn_mask = where(valid_2d,
      array(0.0f, mlx::core::bfloat16),
      array(-std::numeric_limits<float>::infinity(), mlx::core::bfloat16));
  attn_mask = reshape(attn_mask, {1, 1, T, max_kv_len});

  std::vector<array> new_caches;
  new_caches.reserve(cfg.num_layers * 2);
  for (int i = 0; i < cfg.num_layers * 2; i++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  for (int i = 0; i < cfg.num_layers; i++) {
    bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
    std::string lp = "layers." + std::to_string(i);

    auto normed = fast::rms_norm(h, get_weight(lp + ".input_layernorm.weight"),
                                 cfg.rms_norm_eps);

    array layer_out = zeros({}, mlx::core::bfloat16);
    if (is_linear) {
      const auto& cs = inputs[2 + i * 2];
      const auto& rs = inputs[2 + i * 2 + 1];
      auto res = gdn_batched_verify_fn(normed, i, cs, rs, cfg);
      layer_out = std::move(res.output);
      new_caches[i * 2]     = std::move(res.conv_state);
      new_caches[i * 2 + 1] = std::move(res.recurrent_state);
    } else {
      const auto& kk = inputs[2 + i * 2];
      const auto& kv = inputs[2 + i * 2 + 1];
      auto res = attn_batched_verify_fn(normed, i, kk, kv, attn_mask, offset_arr, cfg);
      layer_out = std::move(res.output);
      new_caches[i * 2]     = std::move(res.keys);
      new_caches[i * 2 + 1] = std::move(res.values);
    }
    h = h + layer_out;

    // Post-attn norm + MoE/Dense MLP. The MLP helpers expect 2D input
    // `[B*T, hidden]` (they internally treat `B` as the token count
    // for routing / expert dispatch). Flatten time into batch.
    auto h_flat = reshape(h, {B * T, hidden});
    auto mlp_in = fast::rms_norm(h_flat, get_weight(lp + ".post_attention_layernorm.weight"),
                                 cfg.rms_norm_eps);

    array mlp_out = zeros({}, mlx::core::bfloat16);
    bool moe = is_moe_layer(i, cfg);
    if (moe) {
      mlp_out = sparse_moe_fn(mlp_in, i, cfg, g_layer_quant[i]);
    } else {
      mlp_out = dense_mlp_fn(mlp_in, i, cfg, g_dense_quant[i]);
    }
    h = h + reshape(mlp_out, {B, T, hidden});
  }

  // Final norm + LM head on flat 2D, then reshape outputs.
  auto h_flat = reshape(h, {B * T, hidden});
  h_flat = fast::rms_norm(h_flat, get_weight("final_norm.weight"), cfg.rms_norm_eps);
  array last_hidden = reshape(h_flat, {B, T, hidden});

  auto logits_flat = cfg.tie_word_embeddings
      ? linear_proj(h_flat, "embedding")
      : linear_proj(h_flat, "lm_head");
  int vocab = logits_flat.shape(-1);
  auto logits = reshape(logits_flat, {B, T, vocab});

  auto new_offset = offset_arr + array(T, mlx::core::int32);

  std::vector<array> result;
  result.reserve(3 + cfg.num_layers * 2);
  result.push_back(std::move(logits));
  result.push_back(std::move(new_offset));
  result.push_back(std::move(last_hidden));
  for (auto& c : new_caches) result.push_back(std::move(c));
  return result;
}

// Resettable global cleared by `mlx_qwen35_moe_invalidate_compiled_graphs()`
// on reload — see the `g_moe_decode_compiled` declaration above. This is the
// MoE MTP-verify graph.
static MoeCompiledFn& compiled_moe_verify_batched() {
  if (!g_moe_verify_batched_compiled) {
    g_moe_verify_batched_compiled =
        compile_resettable_weight_graph(moe_verify_batched_decode_fn);
  }
  return g_moe_verify_batched_compiled;
}

// Resettable global cleared by `mlx_qwen35_moe_invalidate_compiled_graphs()`
// on reload — see the `g_moe_decode_compiled` declaration above. This is the
// paged MoE AR-decode graph.
static MoeCompiledFn& compiled_moe_decode_paged() {
  if (!g_moe_decode_paged_compiled) {
    g_moe_decode_paged_compiled =
        compile_resettable_weight_graph(moe_compiled_decode_fn_paged);
  }
  return g_moe_decode_paged_compiled;
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

    // Per-layer quant detection dispatches through detect_layer_quant /
    // detect_router_gate_quant in mlx_qwen35_common.h (registry-first with a
    // heuristic fallback). Keeping the bodies in the header guarantees the
    // flat and paged init paths stay in lockstep.
    for (int i = 0; i < num_layers; i++) {
      std::string pfx = "layers." + std::to_string(i) + ".mlp.";

      // Check if this layer is MoE
      bool moe = (num_experts > 0 && decoder_sparse_step > 0 &&
                  ((i + 1) % decoder_sparse_step) == 0 &&
                  mlp_only_set.count(i) == 0);

      if (moe) {
        auto [sw_q, sw_gs, sw_bits, sw_mode] = detect_layer_quant(pfx + "switch_mlp.gate_proj");
        auto [sh_q, sh_gs, sh_bits, sh_mode] = detect_layer_quant(pfx + "shared_expert.gate_proj");
        auto [g_q, g_gs, g_bits, g_mode] = detect_router_gate_quant(pfx + "gate");
        auto [sg_q, sg_gs, sg_bits, sg_mode] = detect_router_gate_quant(pfx + "shared_expert_gate");

        g_layer_quant.push_back(LayerQuantInfo{
          sw_q, sw_gs, sw_bits, sw_mode,
          sh_q, sh_gs, sh_bits, sh_mode,
          g_q, g_gs, g_bits, g_mode,
          sg_q, sg_gs, sg_bits, sg_mode,
        });
        g_dense_quant.push_back(DenseMLPQuantInfo{false, 0, 0, ""});
      } else {
        g_layer_quant.push_back(LayerQuantInfo{});
        auto [dq, dgs, dbits, dmode] = detect_layer_quant(pfx + "gate_proj");
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

    // Extract outputs: [logits, new_offset, last_hidden, new_caches...].
    // `last_hidden` is at index 2 (caches start at index 3); stashed for
    // `mlx_qwen35_moe_export_last_hidden`. Non-MTP callers pay nothing
    // beyond one extra array copy in the result vector.
    const size_t expected_outputs = 3 + static_cast<size_t>(cfg.num_layers) * 2;
    if (outputs.size() != expected_outputs) {
      fprintf(stderr,
              "[MLX] moe_compiled_decode_fn returned %zu outputs, expected %zu "
              "(layout: [logits, offset, hidden, caches...]). Stride drift.\n",
              outputs.size(), expected_outputs);
      fflush(stderr);
      *output_logits = nullptr;
      return;
    }
    *output_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    g_moe_offset_int++;
    g_moe_last_hidden = outputs[2];
    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_moe_caches[i] = outputs[3 + i];
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

// -----------------------------------------------------------------------------
// MoE batched verify forward (MoE twin of
// `mlx_qwen35_forward_batched_verify`).
//
// Runs ONE compiled forward over `T = depth + 1` tokens, replacing D+1
// sequential `mlx_qwen35_moe_forward` calls performed inside the per-depth
// MoE verify closure. Emits `[1, T, vocab]` logits + `[1, T, hidden]`
// post-final-norm hiddens in one dispatch. Does NOT support tape-replay.
//
// Inputs:
//   - input_ids:        `[1, T]` int32 tokens (T = depth + 1).
//   - embedding_weight: model's embedding table (or LM-head if untied).
//   - depth:            T = depth + 1; depth ∈ [1, 5] enforced by caller.
// Outputs (heap-allocated, caller owns):
//   - out_logits:       `[1, T, vocab]` bf16 logits.
//   - out_hiddens:      `[1, T, hidden_size]` bf16 post-final-norm.
// Side effects:
//   - `g_moe_offset_int` += T
//   - `g_moe_caches[]` updated in place with the post-verify state.
//   - `g_moe_last_hidden` is NOT touched here; the batched FFI returns the
//     full `[1, T, hidden]` directly so the legacy per-step stash isn't used.
//
// Caller MUST hold `MOE_COMPILED_MUTEX` and the weights read lock.
// -----------------------------------------------------------------------------
void mlx_qwen35_moe_forward_batched_verify(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_logits,
    mlx_array** out_hiddens
) {
  if (out_logits) *out_logits = nullptr;
  if (out_hiddens) *out_hiddens = nullptr;
  if (!input_ids_ptr || !embedding_weight_ptr || !out_logits || !out_hiddens) {
    return;
  }
  if (!g_moe_inited) return;
  if (depth < 1 || depth > 5) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_moe_forward_batched_verify: depth %d outside [1, 5]\n",
            depth);
    fflush(stderr);
    return;
  }
  const auto& cfg = g_moe_config;
  int T = depth + 1;

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != T) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_moe_forward_batched_verify: input_ids shape must be "
              "[1, %d], got ndim=%d shape=[%lld,%lld]\n",
              T, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    // Embed: `[1, T]` int32 → `[1, T, hidden]` bf16.
    auto flat_ids = reshape(input_ids, {-1});
    auto emb_flat = take(embedding_weight, flat_ids, 0);
    auto h_3d = reshape(emb_flat, {1, T, cfg.hidden_size});

    std::vector<array> fn_inputs;
    fn_inputs.reserve(2 + cfg.num_layers * 2);
    fn_inputs.push_back(std::move(h_3d));
    fn_inputs.push_back(reshape(array(g_moe_offset_int, mlx::core::int32), {1}));
    for (const auto& c : g_moe_caches) {
      fn_inputs.push_back(c);
    }

    static bool no_compile = std::getenv("MLX_NO_COMPILE") != nullptr;
    auto outputs = no_compile
        ? moe_verify_batched_decode_fn(fn_inputs)
        : compiled_moe_verify_batched()(fn_inputs);

    // outputs[0]: logits  [1, T, vocab]
    // outputs[1]: new_offset_arr (unused — we maintain `g_moe_offset_int` here)
    // outputs[2]: hiddens [1, T, hidden]
    // outputs[3..]: updated caches
    const size_t expected = 3 + static_cast<size_t>(cfg.num_layers) * 2;
    if (outputs.size() != expected) {
      fprintf(stderr,
              "[MLX] moe_verify_batched_decode_fn returned %zu outputs, expected %zu\n",
              outputs.size(), expected);
      fflush(stderr);
      return;
    }
    // Stage allocations into locals first: if `new array(...)` throws on
    // the SECOND call (`std::bad_alloc` under OOM) we'd otherwise leak the
    // first heap `array` already written to `*out_logits`. Only commit to
    // the out-pointers after both allocations succeed.
    array* logits_alloc  = new array(outputs[0]);
    array* hiddens_alloc = nullptr;
    try {
      hiddens_alloc = new array(outputs[2]);
    } catch (...) {
      delete logits_alloc;
      throw;
    }
    *out_logits  = reinterpret_cast<mlx_array*>(logits_alloc);
    *out_hiddens = reinterpret_cast<mlx_array*>(hiddens_alloc);

    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_moe_caches[i] = outputs[3 + i];
    }
    g_moe_offset_int += T;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_forward_batched_verify: %s\n",
            e.what());
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_moe_forward_batched_verify\n");
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
  }
}

// -----------------------------------------------------------------------------
// Eagerly compile the MoE batched verify graph for ALL depths in {1..5}.
// MoE has only ONE variant (no tape-replay) so this prewarms 5 graph
// shapes total.
//
// `compiled_moe_verify_batched()` lazily compiles its graph on first call,
// so without prewarm the first verify cycle of each MoE prompt pays the
// trace+compile cost for its depth-D shape. This entry point runs ONE
// dummy verify forward per depth to force `mlx::core::eval` of the
// compiled-graph outputs, populating MLX's internal shape-keyed compile
// cache so subsequent verifies at the same shape hit the cache.
//
// State preservation:
//   - Snapshots `g_moe_caches[]` and `g_moe_offset_int` before any dummy
//     call and restores them afterward, so the main MoE path's state is
//     unchanged.
//
// Preconditions:
//   - `g_moe_inited` is true.
//   - Embedding weight (`"embedding"` or `"lm_head"`) is registered.
//
// Failure handling: best-effort; exceptions are logged and swallowed.
// -----------------------------------------------------------------------------
void mlx_qwen35_moe_prewarm_verify_compiled() {
  if (!g_moe_inited) {
    return;
  }
  const auto& cfg = g_moe_config;
  if (g_moe_caches.empty()) {
    return;
  }

  // Weights are registered under e.g. `"embedding.weight"` /
  // `"lm_head.weight"` — match the production fetch via the `.weight`
  // suffix.
  array embedding_weight = zeros({}, mlx::core::bfloat16);
  try {
    if (cfg.tie_word_embeddings && has_weight("embedding.weight")) {
      embedding_weight = get_weight("embedding.weight");
    } else if (has_weight("lm_head.weight")) {
      embedding_weight = get_weight("lm_head.weight");
    } else if (has_weight("embedding.weight")) {
      embedding_weight = get_weight("embedding.weight");
    } else {
      fprintf(stderr,
              "[MLX] moe_prewarm_verify_compiled: no embedding.weight/lm_head.weight "
              "registered; skipping prewarm.\n");
      fflush(stderr);
      return;
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] moe_prewarm_verify_compiled: failed to fetch embedding "
            "weight: %s\n", e.what());
    fflush(stderr);
    return;
  }

  std::vector<array> saved_caches = g_moe_caches;
  int saved_offset_int = g_moe_offset_int;

  auto run_one = [&](int depth) {
    int T = depth + 1;
    g_moe_caches = saved_caches;
    g_moe_offset_int = saved_offset_int;

    array dummy_ids = zeros({1, T}, mlx::core::int32);
    array emb_copy = embedding_weight;
    mlx_array* logits_ptr = nullptr;
    mlx_array* hidden_ptr = nullptr;
    mlx_qwen35_moe_forward_batched_verify(
        reinterpret_cast<mlx_array*>(&dummy_ids),
        reinterpret_cast<mlx_array*>(&emb_copy),
        depth,
        &logits_ptr,
        &hidden_ptr);
    if (!logits_ptr || !hidden_ptr) {
      if (logits_ptr) delete reinterpret_cast<array*>(logits_ptr);
      if (hidden_ptr) delete reinterpret_cast<array*>(hidden_ptr);
      throw std::runtime_error(
          "mlx_qwen35_moe_prewarm_verify_compiled: batched verify returned null");
    }
    array logits = *reinterpret_cast<array*>(logits_ptr);
    delete reinterpret_cast<array*>(logits_ptr);
    array hiddens = *reinterpret_cast<array*>(hidden_ptr);
    delete reinterpret_cast<array*>(hidden_ptr);
    mlx::core::eval({logits, hiddens});
  };

  try {
    for (int d = 1; d <= 5; d++) run_one(d);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_prewarm_verify_compiled: %s\n",
            e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_moe_prewarm_verify_compiled\n");
    fflush(stderr);
  }

  g_moe_caches = std::move(saved_caches);
  g_moe_offset_int = saved_offset_int;
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

// Same as `mlx_qwen35_moe_eval_token_and_caches` but also folds an
// arbitrary `extra` array into the async_eval dispatch. Used by the
// chained-cycles MTP path on the MoE twin to fuse the `verify_hidden[K]`
// slice with the next-cycle draft eval. Mirrors
// `mlx_qwen35_eval_token_caches_and_extra` on the dense side; see
// that comment block for the full rationale.
//
// Honours `MLX_EVAL_ALL_CACHES` for parity with the non-chained
// helper: when set, the dispatch batches token + extra + all MoE
// caches; when unset, only token + extra (matching the default
// behaviour that token-only eval implicitly drains the compiled
// graph). `extra_ptr` MAY be null.
void mlx_qwen35_moe_eval_token_caches_and_extra(
    mlx_array* next_token_ptr, mlx_array* extra_ptr) {
  try {
    static bool eval_all = std::getenv("MLX_EVAL_ALL_CACHES") != nullptr;
    if (eval_all) {
      std::vector<array> to_eval;
      to_eval.reserve(2 + g_moe_caches.size());
      to_eval.push_back(*reinterpret_cast<array*>(next_token_ptr));
      if (extra_ptr) {
        to_eval.push_back(*reinterpret_cast<array*>(extra_ptr));
      }
      for (const auto& c : g_moe_caches) {
        to_eval.push_back(c);
      }
      mlx::core::async_eval(std::move(to_eval));
    } else {
      if (extra_ptr) {
        mlx::core::async_eval({*reinterpret_cast<array*>(next_token_ptr),
                               *reinterpret_cast<array*>(extra_ptr)});
      } else {
        mlx::core::async_eval({*reinterpret_cast<array*>(next_token_ptr)});
      }
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_moe_eval_token_caches_and_extra: %s\n",
            e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_moe_eval_token_caches_and_extra\n");
    fflush(stderr);
  }
}

// Reset MoE state — clears BOTH the flat globals AND the paged globals.
// Keeping these symmetric is required because `mlx_qwen35_moe_init_paged`
// flips `g_paged_inited` to true independently of `g_moe_inited`; without
// clearing the paged side here, a later `mlx_qwen35_moe_forward_paged()`
// would pass the init guard and reuse stale KV pools / scales / linear
// caches / offset from a previous request or model.
void mlx_qwen35_moe_reset() {
  // Flat-path globals.
  g_moe_caches.clear();
  g_moe_offset_int = 0;
  g_moe_inited = false;
  g_layer_quant.clear();
  g_dense_quant.clear();
  g_weight_transposes_3d.clear();

  // MTP — clear the stashed last-hidden so a subsequent reset → re-init
  // turn doesn't see a stale handle whose underlying buffer is no longer
  // valid.
  g_moe_last_hidden = std::nullopt;

  // MTP — drop any pending linear-cache snapshot so it can't leak across
  // turns / model loads.
  g_moe_linear_snapshot.clear();
  g_moe_linear_snapshot_offset = 0;
  g_moe_linear_snapshot_taken = false;

  // Paged-path globals.
  g_paged_config = MoeConfig{};
  g_k_pools.clear();
  g_v_pools.clear();
  g_k_scales.clear();
  g_v_scales.clear();
  g_paged_linear_caches.clear();
  g_paged_offset_int = 0;
  g_paged_inited = false;
}

// MTP — export a heap-allocated deep copy of the post-final-norm hidden
// state of the last decoded token, captured by the most recent
// `mlx_qwen35_moe_forward` invocation. Returns nullptr
// if no forward has run since the last reset / the stash is
// unpopulated. Caller owns the returned `mlx_array*`
// (use `mlx_array_delete`).
//
// The returned handle is a lazy MLX array whose graph references the
// final_norm output; the caller MUST `eval()` it before reading any
// element, and MUST NOT call `mlx_qwen35_moe_reset()` between export
// and eval (the reset would clear `g_moe_caches` whose inputs the
// hidden may still depend on via the cached graph). Mirrors the dense
// `mlx_qwen35_export_last_hidden` contract.
void mlx_qwen35_moe_export_last_hidden(mlx_array** out) {
  if (!out) return;
  *out = nullptr;
  if (!g_moe_inited || !g_moe_last_hidden.has_value()) {
    return;
  }
  try {
    *out = reinterpret_cast<mlx_array*>(new array(*g_moe_last_hidden));
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_export_last_hidden: %s\n", e.what());
    fflush(stderr);
    *out = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_moe_export_last_hidden\n");
    fflush(stderr);
    *out = nullptr;
  }
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

// MTP — snapshot the GDN linear-attention caches plus the decode offset.
// Mirrors the dense-path `mlx_qwen35_compiled_snapshot_linear_caches`.
// Called by the MTP cycle macro AFTER Step A and BEFORE verify.
//
// Only linear-attention layer slots are populated; full-attention
// slots are stored as bf16 zero placeholders so per-layer indexing
// stays uniform.
void mlx_qwen35_moe_compiled_snapshot_linear_caches() {
  if (!g_moe_inited) {
    g_moe_linear_snapshot_taken = false;
    return;
  }
  const auto& cfg = g_moe_config;
  try {
    g_moe_linear_snapshot.clear();
    g_moe_linear_snapshot.reserve(cfg.num_layers * 2);
    auto placeholder = []() { return zeros({}, mlx::core::bfloat16); };
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      if (is_linear) {
        // `array(other)` = refcount bump; underlying recurrent/conv
        // buffer stays alive while the global slot mutates inside the
        // verify loop.
        g_moe_linear_snapshot.push_back(array(g_moe_caches[i * 2]));
        g_moe_linear_snapshot.push_back(array(g_moe_caches[i * 2 + 1]));
      } else {
        g_moe_linear_snapshot.push_back(placeholder());
        g_moe_linear_snapshot.push_back(placeholder());
      }
    }
    g_moe_linear_snapshot_offset = g_moe_offset_int;
    g_moe_linear_snapshot_taken = true;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_compiled_snapshot_linear_caches: %s\n", e.what());
    fflush(stderr);
    g_moe_linear_snapshot.clear();
    g_moe_linear_snapshot_taken = false;
  }
}

// MTP — restore the GDN linear caches AND the decode offset from the most
// recent snapshot. Mirrors the dense-path
// `mlx_qwen35_compiled_restore_linear_caches`. Called on rejection
// before replaying K accepted drafts via the main MoE forward FFI.
//
// No-op if no snapshot has been taken since the last reset.
void mlx_qwen35_moe_compiled_restore_linear_caches() {
  if (!g_moe_inited || !g_moe_linear_snapshot_taken) return;
  const auto& cfg = g_moe_config;
  try {
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      if (!is_linear) continue;
      g_moe_caches[i * 2]     = array(g_moe_linear_snapshot[i * 2]);
      g_moe_caches[i * 2 + 1] = array(g_moe_linear_snapshot[i * 2 + 1]);
    }
    g_moe_offset_int = g_moe_linear_snapshot_offset;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_compiled_restore_linear_caches: %s\n", e.what());
    fflush(stderr);
  }
}

// Exposed for `mlx_qwen35_moe_mtp_compiled_init_from_main` so the MTP init
// can fail loudly if the main MoE compiled path hasn't been initialised yet.
// Mirrors `mlx_qwen35_is_compile_inited` on the dense side — without this
// guard the MTP path would mirror a phantom prefix offset into its own
// `g_mtp_offset_int` from a fresh-zero `g_moe_offset_int`.
int mlx_qwen35_moe_is_compile_inited() {
  return g_moe_inited ? 1 : 0;
}

// Test-only helper: forcibly mark the main MoE compiled path as initialised
// (or not) without going through the full `init_from_prefill` flow that
// requires real per-layer KV cache arrays. Used by the MoE MTP FFI smoke
// tests in `crates/mlx-core/src/models/qwen3_5_moe/mtp.rs` so they can
// satisfy the new `is_compile_inited` precondition in
// `mlx_qwen35_moe_mtp_compiled_init_from_main` without standing up a full
// MoE decoder cache. Production code MUST NOT call this — use
// `mlx_qwen35_moe_init_from_prefill` instead.
void mlx_qwen35_moe_compiled_test_force_inited(int inited) {
  g_moe_inited = (inited != 0);
}

// =============================================================================
// Paged forward FFI.
//
// Coexists alongside `mlx_qwen35_moe_forward` / `_init_from_prefill`.
// =============================================================================

// Initialize the paged MoE forward graph from per-layer pool / scale handles.
//
// Layout contract:
//   - `k_pool_handles[i]`: pointer to a `[num_blocks, num_kv_heads,
//     head_size/x_pack=8, block_size=16, x_pack=8]` bf16 array view.
//     Uses bf16 (`KvDtype::Bf16`) and `block_size = 16`.
//   - `v_pool_handles[i]`: pointer to a `[num_blocks, num_kv_heads,
//     head_size, block_size=16]` bf16 array view.
//   - `k_scale_handles[i]` / `v_scale_handles[i]`: pointer to `[1]` f32
//     scale placeholders (1.0; reserved for FP8 calibration).
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
// Compile-graph configuration (encoded into the FFI's signature):
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

    // Same per-layer detect helpers as the flat path (see comment there).
    for (int i = 0; i < num_layers; i++) {
      std::string pfx = "layers." + std::to_string(i) + ".mlp.";
      bool moe = (num_experts > 0 && decoder_sparse_step > 0 &&
                  ((i + 1) % decoder_sparse_step) == 0 &&
                  mlp_only_set.count(i) == 0);

      if (moe) {
        auto [sw_q, sw_gs, sw_bits, sw_mode] = detect_layer_quant(pfx + "switch_mlp.gate_proj");
        auto [sh_q, sh_gs, sh_bits, sh_mode] = detect_layer_quant(pfx + "shared_expert.gate_proj");
        auto [g_q, g_gs, g_bits, g_mode] = detect_router_gate_quant(pfx + "gate");
        auto [sg_q, sg_gs, sg_bits, sg_mode] = detect_router_gate_quant(pfx + "shared_expert_gate");
        g_layer_quant.push_back(LayerQuantInfo{
          sw_q, sw_gs, sw_bits, sw_mode,
          sh_q, sh_gs, sh_bits, sh_mode,
          g_q, g_gs, g_bits, g_mode,
          sg_q, sg_gs, sg_bits, sg_mode,
        });
        g_dense_quant.push_back(DenseMLPQuantInfo{false, 0, 0, ""});
      } else {
        g_layer_quant.push_back(LayerQuantInfo{});
        auto [dq, dgs, dbits, dmode] = detect_layer_quant(pfx + "gate_proj");
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
// the Rust adapter's `build_paged_attention_inputs`; the per-layer
// pool/scale globals come from `mlx_qwen35_moe_init_paged`.
//
// CONTRACT: This FFI is decode-only — `input_ids` MUST have exactly one
// element and `slot_mapping` MUST be `[1]`. Chunked prefill (multi-token)
// goes through the flat path. The contract is enforced explicitly:
// violating it returns null logits without modifying global state, so the
// Rust caller can fall back cleanly. See the docstring on
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

    // Single-token decode only.
    //
    // `attn_for_compile_paged` builds new_k / new_v with shape
    // `[1, num_kv_heads, head_size]` and feeds `slot_mapping` directly
    // into `paged_kv_write`, which requires
    // `slot_mapping.shape(0) == new_k.shape(0)`. Reject any caller that
    // tries to push more than one token through the paged FFI. Returning
    // null here matches the existing error contract and lets the Rust
    // caller fall back to the flat path cleanly.
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
// Test helper: build the paged-attention graph in isolation and
// force-evaluate it.
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
// single "non-zero return" code lumping both cases together would accept
// any failure as success.
//
// Graph-build smoke coverage — the forward_paged smoke test fails inside
// the LM-head / embedding lookup BEFORE reaching `attn_for_compile_paged`,
// so without this helper the paged graph itself is never exercised.
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

// Invalidate ALL compiled MoE dispatch graphs (the MTP-verify graph plus
// the flat + paged AR-decode graphs) so the next call re-traces against
// the CURRENT weight registry.
//
// The compiled MoE bodies read expert + attention weights via `get_weight(...)`
// INSIDE the traced closure (NOT as compile inputs), so `mlx::core::compile(...)`
// bakes the captured weight arrays into the cached tape and reuses them verbatim
// on every later call. These globals are PROCESS-WIDE and deliberately survive
// the per-turn `mlx_qwen35_moe_reset()` (cross-turn reuse). On a model RELOAD
// (weights swapped in the registry) the baked weights are stale, so a second
// same-shape MoE model loaded in the same process would decode/verify with the
// FIRST model's weights — silent corruption.
//
// Each graph is compiled through `compile_resettable_weight_graph` (see
// `mlx_qwen35_common.h`), which wraps the free function in a CAPTURING lambda so
// MLX hands it a heap-allocated, UNIQUE `fun_id` whose shared_ptr deleter calls
// `compile_erase()`. Assigning an empty `std::function{}` therefore destroys the
// wrapper, ERASES its compile-cache entry (freeing the baked tape), and forces a
// NEW wrapper (new `fun_id`) to be built on the next call — which re-traces
// against the live registry on whatever thread next calls `compiled_moe_*`.
// (A plain `mlx::core::compile(free_fn)` would NOT work here: it keys on the
// function's stable code address with no eviction hook, so nulling the
// std::function would silently replay the stale tape.) This is correct
// regardless of thread because the re-trace is lazy on the inference thread; we
// do NOT rely on cross-thread `compile_clear_cache()` (MLX's CompilerCache is
// thread_local).
//
// Called by the Rust MoE loader (`register_moe_weights_with_cpp`) on EVERY model
// reload, immediately after the dense `mlx_qwen35_invalidate_compiled_graphs()`
// call and INSIDE the `COMPILED_WEIGHTS_RWLOCK` write critical section, so it is
// serialized against in-flight compiled reads. This mirrors the dense
// `mlx_qwen35_invalidate_compiled_graphs()` (which nulls the dense bucket/paged
// verify tables that the MoE path does not use). Also transitively invalidates
// the MoE MTP draft graph in `mlx_qwen35_moe_mtp_compiled.cpp`.
void mlx_qwen35_moe_invalidate_compiled_graphs() {
  g_moe_decode_compiled = MoeCompiledFn{};
  g_moe_verify_batched_compiled = MoeCompiledFn{};
  g_moe_decode_paged_compiled = MoeCompiledFn{};
  mlx_qwen35_moe_mtp_invalidate_compiled_graphs();
}

}  // extern "C"
