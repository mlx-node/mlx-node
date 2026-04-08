#include "mlx_qwen35_common.h"
#include <cstdlib>

using namespace qwen35_common;

// =============================================================================
// Gemma4 26B-a4b-it Forward Pass -- Compiled via mlx::core::compile
//
// Complete implementation matching Python mlx-vlm forward pass 1:1.
//
// Architecture:
//   30 layers: 25 sliding_attention + 5 full_attention (global)
//   MoE on ALL layers: dense MLP + parallel sparse MoE (128 experts, top_k=8)
//   sliding_window=1024, head_dim=256 (sliding), global_head_dim=512 (global)
//   num_heads=16, num_kv_heads=8 (sliding), global_num_kv_heads=2 (global)
//   attention_k_eq_v=true on global layers
//   partial_rotary_factor=0.25 on global layers (ProportionalRoPE)
//   final_logit_softcapping=30.0
//
// Key design decisions matching Qwen3.5 MoE (36 tok/s):
//   1. Pre-allocated caches with FIXED shapes for compile() graph caching
//   2. SDPA receives full padded caches + additive mask (not sliced caches)
//   3. ALL offset-dependent ops use offset_arr (input array, not C++ int)
//   4. Masks built ONCE before the layer loop, shared by all same-type layers
//   5. MoE uses gather_mm with sorted=false for decode (indices.size < 64)
//
// Set MLX_NO_COMPILE=1 to fall back to the non-compiled path for A/B testing.
// =============================================================================

namespace {

// Gemma4-specific config
struct Gemma4Config {
  int num_layers;
  int hidden_size;
  int num_heads;
  // Sliding attention params
  int num_kv_heads;       // 8
  int head_dim;           // 256
  float rope_local_base_freq; // 10K
  // Global attention params
  int global_num_kv_heads;   // 2
  int global_head_dim;       // 512
  float rope_theta;          // 1M
  float partial_rotary_factor; // 0.25
  // Common
  float rms_norm_eps;
  int sliding_window;     // 1024
  bool tie_word_embeddings;
  int max_kv_len;
  int batch_size;
  // MoE
  int num_experts;
  int top_k_experts;
  int moe_intermediate_size;
  int intermediate_size;  // dense MLP
  // Logit softcapping
  float final_logit_softcapping;
  // Per-layer type: true = global (full attention), false = sliding
  std::vector<bool> is_global_layer;
};

static Gemma4Config g_gemma4_config{};
static std::vector<array> g_gemma4_caches;     // num_layers * 2 arrays (keys, values)
static int g_gemma4_offset_int = 0;
static bool g_gemma4_inited = false;

// Pre-computed freqs for fast::rope on global layers: 1-D [rotary_dim/2]
// This is what Python ProportionalRoPE stores as self._freqs.
// fast::rope internally computes reciprocal(freqs) to get inv_freq, then
// theta = positions * inv_freq, cos/sin applied on GPU via optimized kernel.
static std::vector<array> g_rope_freqs_storage;
static int g_rotated_dims = 0;  // 2 * rope_angles (e.g., 128 for partial_rotary_factor=0.25, head_dim=512)

// Cached 3D transposes for expert weights [E,out,in] -> [E,in,out]
static std::unordered_map<std::string, array> g_weight_transposes_3d;

// Pure read -- 3D transposes are pre-computed in init.
array get_weight_t3d(const std::string& name) {
  auto it = g_weight_transposes_3d.find(name);
  if (it != g_weight_transposes_3d.end()) {
    return it->second;  // copy (refcount bump)
  }
  throw std::runtime_error("3D transpose not found for weight: " + name);
}

// =====================================================================
// GeGLU activation (GELU-gated linear unit) -- compiled for kernel fusion
//
// Python: nn.gelu_approx(gate) * up
// gelu_approx = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Computed in f32 to avoid bf16 overflow in tanh intermediate values.
// =====================================================================

static std::vector<array> geglu_impl(const std::vector<array>& inputs) {
  const auto& gate = inputs[0];
  const auto& up = inputs[1];
  auto gate_f32 = astype(gate, mlx::core::float32);
  auto c = array(0.7978845608028654f, mlx::core::float32); // sqrt(2/pi)
  auto inner = c * (gate_f32 + array(0.044715f, mlx::core::float32) * gate_f32 * gate_f32 * gate_f32);
  auto activated = array(0.5f, mlx::core::float32) * gate_f32 * (array(1.0f, mlx::core::float32) + tanh(inner));
  return {astype(activated, gate.dtype()) * up};
}

static auto& compiled_geglu() {
  static auto fn = mlx::core::compile(geglu_impl, /*shapeless=*/true);
  return fn;
}

static array geglu(const array& gate, const array& up) {
  return compiled_geglu()({gate, up})[0];
}

// =====================================================================
// Proportional RoPE for global layers via fast::rope with precomputed freqs
//
// Matches Python mlx-vlm ProportionalRoPE.__call__:
//   1. Split head into left/right halves
//   2. Gather rotated portions from each half
//   3. Apply mx.fast.rope(rotated, rotated_dims, freqs=self._freqs)
//   4. Scatter rotated portions back, reassemble head
//
// IMPORTANT FOR compile():
//   All slice operations use the integer constants g_rotated_dims/2 and
//   global_head_dim/2 which are fixed at init time. The only dynamic value
//   is offset_arr which is passed through to fast::rope as an input array.
//   This keeps the graph topology identical across decode steps.
// =====================================================================

static array proportional_rope(const array& x, const array& offset_arr) {
  // x is [B, H, 1, D] where D = global_head_dim (e.g. 512)
  // After transpose to [B, H, L, D] format for SDPA
  int D = g_gemma4_config.global_head_dim;
  int half = D / 2;                         // 256
  int rot_half = g_rotated_dims / 2;        // 64

  // Split head into left/right halves along last axis
  // Using integer constants (not shape-dependent) for compile stability
  auto left  = slice(x, {0, 0, 0, 0},    {x.shape(0), x.shape(1), x.shape(2), half});
  auto right = slice(x, {0, 0, 0, half},  {x.shape(0), x.shape(1), x.shape(2), D});

  // Gather rotated portions: first rot_half dims from each half
  auto left_rot  = slice(left,  {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), rot_half});
  auto right_rot = slice(right, {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), rot_half});

  // rotated = concat([left[:rot_half], right[:rot_half]], axis=-1)  -> [..., rotated_dims]
  auto rotated = concatenate({left_rot, right_rot}, -1);

  // Apply fast::rope with precomputed freqs (single fused Metal kernel)
  rotated = fast::rope(
      rotated,
      g_rotated_dims,        // dims = rotated_dims (e.g. 128)
      false,                 // traditional = false
      std::nullopt,          // base = nullopt (using freqs)
      1.0f,                  // scale
      offset_arr,            // array offset for compile compatibility
      g_rope_freqs_storage[0]  // precomputed freqs [rotated_dims/2]
  );

  // Scatter back: left = concat([rotated[:rot_half], left[rot_half:]], axis=-1)
  auto rotated_left  = slice(rotated, {0, 0, 0, 0},        {x.shape(0), x.shape(1), x.shape(2), rot_half});
  auto rotated_right = slice(rotated, {0, 0, 0, rot_half},  {x.shape(0), x.shape(1), x.shape(2), g_rotated_dims});
  auto left_rest  = slice(left,  {0, 0, 0, rot_half}, {x.shape(0), x.shape(1), x.shape(2), half});
  auto right_rest = slice(right, {0, 0, 0, rot_half}, {x.shape(0), x.shape(1), x.shape(2), half});

  left  = concatenate({rotated_left, left_rest}, -1);
  right = concatenate({rotated_right, right_rest}, -1);

  return concatenate({left, right}, -1);
}

// =====================================================================
// Gemma4 attention -- matches Python Attention.__call__ exactly
//
// Key differences from Qwen3.5:
//   1. No gating (standard attention, not gated)
//   2. V gets scale-free RMS norm (no learnable weight)
//   3. K=V sharing on global layers (values = keys BEFORE k_norm)
//   4. Attention scale = 1.0
//   5. Sliding layers: circular KV cache via remainder(offset, window)
//   6. Global layers: proportional RoPE, linear KV cache write
// =====================================================================

struct Gemma4AttnResult { array output, keys, values; };

static Gemma4AttnResult gemma4_attention(
    const array& x,          // [B, hidden] -- 2D
    int layer_idx,
    const array& kv_keys,    // [B, Hkv, cache_len, D]
    const array& kv_values,  // [B, Hkv, cache_len, D]
    const array& attn_mask,  // [1, 1, 1, cache_len] additive mask
    const array& offset_arr, // scalar int32
    const Gemma4Config& cfg) {

  int B = x.shape(0);
  bool is_global = cfg.is_global_layer[layer_idx];
  int num_kv_heads = is_global ? cfg.global_num_kv_heads : cfg.num_kv_heads;
  int head_dim = is_global ? cfg.global_head_dim : cfg.head_dim;

  // Python: self.use_k_eq_v = getattr(config, 'attention_k_eq_v', False) and not self.is_sliding
  bool k_is_v = is_global;

  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  // Python: queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
  auto queries = linear_proj(x, pfx + "q_proj");
  queries = reshape(queries, {B, 1, cfg.num_heads, head_dim});

  // Python: queries = self.q_norm(queries)
  queries = fast::rms_norm(queries, get_weight(pfx + "q_norm.weight"), cfg.rms_norm_eps);

  // Python: keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
  auto keys = linear_proj(x, pfx + "k_proj");
  keys = reshape(keys, {B, 1, num_kv_heads, head_dim});

  // Python: if self.use_k_eq_v: values = keys (before k_norm!)
  //         else: values = self.v_proj(x).reshape(...)
  array values = keys;  // default: K=V (before k_norm)
  if (!k_is_v) {
    values = linear_proj(x, pfx + "v_proj");
    values = reshape(values, {B, 1, num_kv_heads, head_dim});
  }

  // Python: keys = self.k_norm(keys)
  keys = fast::rms_norm(keys, get_weight(pfx + "k_norm.weight"), cfg.rms_norm_eps);
  // Python: values = self.v_norm(values) -- scale-free RMSNorm (no learnable weight)
  values = rms_norm_no_weight(values, cfg.rms_norm_eps);

  // Python: values = values.transpose(0, 2, 1, 3)
  values = transpose(values, {0, 2, 1, 3});
  // Python: keys = keys.transpose(0, 2, 1, 3)
  keys = transpose(keys, {0, 2, 1, 3});

  // Python: keys = self.rope(keys, offset=offset)
  if (is_global) {
    keys = proportional_rope(keys, offset_arr);
  } else {
    keys = fast::rope(keys, head_dim, false, cfg.rope_local_base_freq, 1.0f, offset_arr);
  }

  // Python: if cache is not None: keys, values = cache.update_and_fetch(keys, values)
  // We use pre-allocated caches + slice_update (fixed shapes for compile)
  auto offset_1d = reshape(offset_arr, {1});
  auto write_pos = is_global
      ? offset_1d
      : reshape(mlx::core::remainder(offset_arr, array(cfg.sliding_window, mlx::core::int32)), {1});
  auto new_kv_keys   = mlx::core::slice_update(kv_keys,   keys,   write_pos, {2});
  auto new_kv_values = mlx::core::slice_update(kv_values, values, write_pos, {2});

  // Python: queries = queries.transpose(0, 2, 1, 3)
  queries = transpose(queries, {0, 2, 1, 3});
  // Python: queries = self.rope(queries, offset=offset)
  if (is_global) {
    queries = proportional_rope(queries, offset_arr);
  } else {
    queries = fast::rope(queries, head_dim, false, cfg.rope_local_base_freq, 1.0f, offset_arr);
  }

  // Python: output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
  // self.scale = 1.0 -- QKV normalization handles scaling
  // We pass full padded caches + additive mask (Qwen3.5 approach for fixed shapes)
  float scale = 1.0f;
  auto attn_out = fast::scaled_dot_product_attention(
      queries, new_kv_keys, new_kv_values, scale, "", attn_mask, {});

  // Python: output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
  attn_out = transpose(attn_out, {0, 2, 1, 3});
  attn_out = reshape(attn_out, {B, cfg.num_heads * head_dim});

  // Python: return self.o_proj(output)
  auto output = linear_proj(attn_out, pfx + "o_proj");

  return {output, new_kv_keys, new_kv_values};
}

// =====================================================================
// Dense MLP -- matches Python MLP.__call__
//
// output = down_proj(gelu_approx(gate_proj(x)) * up_proj(x))
//
// Python: nn.gelu_approx is the TANH approximation:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// =====================================================================

static array gemma4_dense_mlp(
    const array& x,
    int layer_idx) {
  std::string mp = "layers." + std::to_string(layer_idx) + ".mlp.";
  // Python: return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))
  auto gate = linear_proj(x, mp + "gate_proj");
  auto up   = linear_proj(x, mp + "up_proj");
  return linear_proj(geglu(gate, up), mp + "down_proj");
}

// =====================================================================
// Gemma4 MoE router -- matches Python Router.__call__
//
// Python:
//   x = self.norm(x)                     # RMSNormNoScale
//   x = x * self._root_size              # hidden_size^(-0.5)
//   x = x * self.scale                   # learnable [hidden_size]
//   expert_scores = self.proj(x)          # [B, num_experts]
//   router_probs = softmax(expert_scores) # over all experts
//   top_k_indices = argpartition(-expert_scores, kth=top_k-1)[..., :top_k]
//   top_k_weights = take_along_axis(router_probs, top_k_indices)
//   top_k_weights = top_k_weights / sum(top_k_weights, keepdims=True)
//   top_k_weights = top_k_weights * per_expert_scale[top_k_indices]
//   return top_k_indices, top_k_weights
// =====================================================================

struct RouterResult { array top_k_indices, top_k_weights; };

static RouterResult gemma4_router(
    const array& x,          // [B, hidden] -- 2D
    int layer_idx,
    const Gemma4Config& cfg) {
  std::string pfx = "layers." + std::to_string(layer_idx) + ".router.";
  int k = cfg.top_k_experts;

  // Python: x = self.norm(x) -- RMSNormNoScale (no learnable weight)
  auto normed = rms_norm_no_weight(x, cfg.rms_norm_eps);

  // Python: x = x * self._root_size
  auto root_size = array(std::pow((float)cfg.hidden_size, -0.5f), normed.dtype());
  auto scaled = normed * root_size;

  // Python: x = x * self.scale
  scaled = scaled * get_weight(pfx + "scale");

  // Python: expert_scores = self.proj(x)
  auto expert_scores = linear_proj(scaled, pfx + "proj");

  // Python: router_probs = mx.softmax(expert_scores, axis=-1)
  auto router_probs = mlx::core::softmax(expert_scores, {-1}, /*precise=*/true);

  // Python: top_k_indices = mx.argpartition(-expert_scores, kth=self.config.top_k_experts - 1, axis=-1)
  //                         [..., : self.config.top_k_experts]
  // Note: argpartition(-x, kth=k-1) puts the k SMALLEST of -x (= k LARGEST of x) in positions 0..k-1
  auto neg_scores = negative(expert_scores);
  auto top_indices_full = argpartition(neg_scores, k - 1, -1);
  auto top_k_indices = slice(top_indices_full, {0, 0}, {x.shape(0), k});

  // Python: top_k_weights = mx.take_along_axis(router_probs, top_k_indices, axis=-1)
  auto top_k_weights = mlx::core::take_along_axis(router_probs, top_k_indices, -1);

  // Python: top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)
  auto weight_sum = sum(top_k_weights, {-1}, true);
  top_k_weights = top_k_weights / weight_sum;

  // Python: top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]
  auto per_expert_scale = get_weight(pfx + "per_expert_scale");
  auto flat_idx = reshape(top_k_indices, {-1});
  auto expert_scales = take(per_expert_scale, flat_idx, 0);
  expert_scales = reshape(expert_scales, top_k_weights.shape());
  top_k_weights = top_k_weights * expert_scales;

  return {top_k_indices, top_k_weights};
}

// =====================================================================
// Gemma4 MoE expert dispatch -- matches Python Experts.__call__
//
// Python SwitchGLU flow:
//   x = expand_dims(x, (-2, -3))  -> [B*S, 1, 1, H]
//   do_sort = indices.size >= 64
//   (for decode B=1,S=1,K=8: size=8 < 64 -> no sort)
//   x_up   = self.up_proj(x, indices, sorted=False)
//   x_gate = self.gate_proj(x, indices, sorted=False)
//   x = self.down_proj(self.activation(x_up, x_gate), indices, sorted=False)
//   return x.squeeze(-2)
//
// SwitchLinear.__call__:
//   gather_mm(x, self.weight.swapaxes(-1,-2), rhs_indices=indices, sorted=sorted)
//
// GeGLU activation:
//   return nn.gelu_approx(gate) * x   -- note: gate is x_gate, x is x_up
//
// We use fused gate_up_proj (already transposed [E, hidden, 2*moe_inter] in
// g_weight_transposes_3d). gather_mm returns [B, K, 1, 2*moe_inter], we split,
// apply gelu_approx(gate)*up, then down_proj.
// =====================================================================

static array gemma4_moe_experts(
    const array& x,        // [B, hidden] -- 2D (B*S, single-token decode: B*S=1)
    const array& top_k_indices,   // [B*S, K]
    const array& top_k_weights,   // [B*S, K]
    int layer_idx,
    const Gemma4Config& cfg) {
  int B = x.shape(0);
  int hidden = cfg.hidden_size;
  int k = cfg.top_k_experts;
  int moe_inter = cfg.moe_intermediate_size;

  std::string pfx = "layers." + std::to_string(layer_idx) + ".experts.";

  // Python SwitchGLU: x = mx.expand_dims(x, (-2, -3)) -> [B*S, 1, 1, H]
  auto x_expanded = reshape(x, {B, 1, 1, hidden});

  // Python: do_sort = indices.size >= 64
  // For decode (B=1, K=8): size=8 < 64, so sorted=False
  // We match this exactly -- no sorting for decode.

  // Fused gate_up: [B, K, 1, 2*moe_inter]
  // get_weight_t3d gives [E, hidden, 2*moe_inter] (already transposed at init)
  auto gate_up = mlx::core::gather_mm(
      x_expanded,
      get_weight_t3d(pfx + "gate_up_proj"),
      std::nullopt,  // no lhs_indices
      top_k_indices,
      false);        // sorted=false (decode, size < 64)

  // Split gate and up: each [B, K, 1, moe_inter]
  auto gate = slice(gate_up, {0, 0, 0, 0}, {gate_up.shape(0), gate_up.shape(1), 1, moe_inter});
  auto up   = slice(gate_up, {0, 0, 0, moe_inter}, {gate_up.shape(0), gate_up.shape(1), 1, 2 * moe_inter});

  // Python GeGLU: return nn.gelu_approx(gate) * x  (where x=up, gate=gate)
  auto activated = geglu(gate, up);

  // down_proj: [B, K, 1, hidden]
  auto down = mlx::core::gather_mm(
      activated,
      get_weight_t3d(pfx + "down_proj"),
      std::nullopt,
      top_k_indices,
      false);

  // Squeeze the penultimate dim: [B, K, 1, hidden] -> [B, K, hidden]
  auto expert_out = squeeze(down, {-2});

  // Apply routing weights: [B, K, 1] * [B, K, hidden]
  auto dt = x.dtype();
  auto weights_expanded = astype(reshape(top_k_weights, {B, k, 1}), dt);
  auto weighted = expert_out * weights_expanded;

  // Sum over experts: [B, hidden]
  return sum(weighted, {1});
}

// =====================================================================
// Full compilable decode function
//
// Matches Python DecoderLayer.__call__ and Gemma4TextModel.__call__
//
// inputs:  [h, offset_arr, cache[0].keys, cache[0].values, ..., cache[N-1].keys, cache[N-1].values]
// outputs: [logits, new_offset, new_cache[0].keys, ..., new_cache[N-1].values]
// =====================================================================

static std::vector<array> gemma4_compiled_decode_fn(const std::vector<array>& inputs) {
  const auto& cfg = g_gemma4_config;
  auto h = inputs[0];
  auto offset_arr = inputs[1]; // scalar int32

  // ---------------------------------------------------------------
  // Build attention masks ONCE before the layer loop.
  // Matches Qwen3.5 MoE approach: full padded caches + additive mask.
  //
  // For compile() stability, masks are built with offset_arr (input array),
  // not a C++ integer constant. This keeps graph topology identical across
  // decode steps.
  // ---------------------------------------------------------------

  // Single unified attention mask: [1, 1, 1, max_kv_len]
  // ALL caches are now padded to max_kv_len (matching Qwen3.5 MoE pattern).
  // positions <= offset → valid (0.0), positions > offset → -inf
  // This works for both global and sliding layers because:
  //   - Global layers write linearly at offset, read all valid positions
  //   - Sliding layers write circularly via remainder(offset, window), but the
  //     mask still correctly marks positions <= offset as valid
  int max_kv_len_val = inputs[2].shape(2);  // First cache's seq dimension
  auto positions = arange(0, max_kv_len_val, mlx::core::int32);
  auto valid_mask = less_equal(positions, offset_arr);
  auto attn_mask = where(valid_mask,
      array(0.0f, mlx::core::bfloat16),
      array(-std::numeric_limits<float>::infinity(), mlx::core::bfloat16));
  attn_mask = reshape(attn_mask, {1, 1, 1, max_kv_len_val});

  // Pre-allocate new_caches (placeholders overwritten in loop)
  std::vector<array> new_caches;
  new_caches.reserve(cfg.num_layers * 2);
  for (int i = 0; i < cfg.num_layers * 2; i++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  for (int i = 0; i < cfg.num_layers; i++) {
    bool is_global = cfg.is_global_layer[i];
    std::string lp = "layers." + std::to_string(i);

    // Single unified mask for all layers (all caches are max_kv_len).

    // ---------------------------------------------------------------
    // Python DecoderLayer.__call__:
    //   residual = x
    //   h = self.input_layernorm(x)
    //   h = self.self_attn(h, mask, cache)
    //   h = self.post_attention_layernorm(h)
    //   h = residual + h
    // ---------------------------------------------------------------
    auto residual = h;

    // 1. input_layernorm
    auto normed = fast::rms_norm(h, get_weight(lp + ".input_layernorm.weight"), cfg.rms_norm_eps);

    // 2. Attention
    const auto& kk = inputs[2 + i * 2];
    const auto& kv = inputs[2 + i * 2 + 1];
    auto attn_res = gemma4_attention(normed, i, kk, kv, attn_mask, offset_arr, cfg);

    // 3. post_attention_layernorm + residual
    auto attn_normed = fast::rms_norm(attn_res.output,
        get_weight(lp + ".post_attention_layernorm.weight"), cfg.rms_norm_eps);
    h = residual + attn_normed;

    new_caches[i * 2]     = std::move(attn_res.keys);
    new_caches[i * 2 + 1] = std::move(attn_res.values);

    // ---------------------------------------------------------------
    // Python DecoderLayer.__call__ (FFN):
    //   residual = h
    //   if self.enable_moe:
    //     h1 = self.pre_feedforward_layernorm(h)
    //     h1 = self.mlp(h1)
    //     h1 = self.post_feedforward_layernorm_1(h1)
    //     top_k_indices, top_k_weights = self.router(h)
    //     h2 = self.pre_feedforward_layernorm_2(h)
    //     h2 = self.experts(h2, top_k_indices, top_k_weights)
    //     h2 = self.post_feedforward_layernorm_2(h2)
    //     h = h1 + h2
    //   else:
    //     h = self.pre_feedforward_layernorm(h)
    //     h = self.mlp(h)
    //   h = self.post_feedforward_layernorm(h)
    //   h = residual + h
    // ---------------------------------------------------------------
    residual = h;

    bool has_moe = (cfg.num_experts > 0 && has_weight(lp + ".router.proj.weight"));

    if (has_moe) {
      // Dense MLP branch
      auto h1 = fast::rms_norm(h,
          get_weight(lp + ".pre_feedforward_layernorm.weight"), cfg.rms_norm_eps);
      h1 = gemma4_dense_mlp(h1, i);
      h1 = fast::rms_norm(h1,
          get_weight(lp + ".post_feedforward_layernorm_1.weight"), cfg.rms_norm_eps);

      // MoE branch: router sees h (not normed)
      auto router_res = gemma4_router(h, i, cfg);

      // Experts see pre_feedforward_layernorm_2(h)
      auto h2 = fast::rms_norm(h,
          get_weight(lp + ".pre_feedforward_layernorm_2.weight"), cfg.rms_norm_eps);
      h2 = gemma4_moe_experts(h2, router_res.top_k_indices, router_res.top_k_weights, i, cfg);
      h2 = fast::rms_norm(h2,
          get_weight(lp + ".post_feedforward_layernorm_2.weight"), cfg.rms_norm_eps);

      // Python: h = h1 + h2
      h = h1 + h2;
    } else {
      // Non-MoE layers
      h = fast::rms_norm(h,
          get_weight(lp + ".pre_feedforward_layernorm.weight"), cfg.rms_norm_eps);
      h = gemma4_dense_mlp(h, i);
    }

    // Python: h = self.post_feedforward_layernorm(h)
    h = fast::rms_norm(h,
        get_weight(lp + ".post_feedforward_layernorm.weight"), cfg.rms_norm_eps);
    // Python: h = residual + h
    h = residual + h;

    // ---------------------------------------------------------------
    // Python: if self.layer_scalar is not None: h = h * self.layer_scalar
    // ---------------------------------------------------------------
    if (has_weight(lp + ".layer_scalar")) {
      h = h * get_weight(lp + ".layer_scalar");
    }
  }

  // ---------------------------------------------------------------
  // Python Gemma4TextModel.__call__: return self.norm(h)
  // Python LanguageModel.__call__:
  //   out = self.model(...)
  //   out = self.model.embed_tokens.as_linear(out)  # h @ embed_weight^T
  //   if self.final_logit_softcapping: out = logit_softcap(cap, out)
  // ---------------------------------------------------------------
  h = fast::rms_norm(h, get_weight("norm.weight"), cfg.rms_norm_eps);

  if (cfg.tie_word_embeddings) {
    h = linear_proj(h, "embed_tokens");
  } else {
    h = linear_proj(h, "lm_head");
  }

  // Logit softcapping: tanh(logits / cap) * cap
  if (cfg.final_logit_softcapping > 0.0f) {
    auto cap = array(cfg.final_logit_softcapping, h.dtype());
    h = tanh(h / cap) * cap;
  }

  auto new_offset = offset_arr + array(1, mlx::core::int32);

  std::vector<array> result;
  result.reserve(2 + cfg.num_layers * 2);
  result.push_back(std::move(h));
  result.push_back(std::move(new_offset));
  for (auto& c : new_caches) result.push_back(std::move(c));
  return result;
}

static auto& compiled_gemma4_decode() {
  static auto fn = mlx::core::compile(gemma4_compiled_decode_fn);
  return fn;
}

// Greedy variant: argmax inside compiled graph avoids materializing full logits
static std::vector<array> gemma4_compiled_decode_greedy_fn(
    const std::vector<array>& inputs) {
  const auto& cfg = g_gemma4_config;
  auto outputs = gemma4_compiled_decode_fn(inputs);
  auto next_token = argmax(outputs[0], -1);

  std::vector<array> result;
  result.reserve(2 + cfg.num_layers * 2);
  result.push_back(std::move(next_token));
  result.push_back(std::move(outputs[1]));
  for (size_t i = 2; i < outputs.size(); ++i) {
    result.push_back(std::move(outputs[i]));
  }
  return result;
}

static auto& compiled_gemma4_decode_greedy() {
  static auto fn = mlx::core::compile(gemma4_compiled_decode_greedy_fn);
  return fn;
}

}  // namespace

// =============================================================================
// Public FFI functions
// =============================================================================

extern "C" {

// Initialize Gemma4 forward pass from post-prefill caches.
void mlx_gemma4_init_from_prefill(
    // Config params
    int num_layers,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int global_num_kv_heads,
    int global_head_dim,
    float rope_theta,
    float rope_local_base_freq,
    float partial_rotary_factor,
    float rms_norm_eps,
    int sliding_window,
    int tie_word_embeddings,
    int max_kv_len,
    int batch_size,
    // MoE params
    int num_experts,
    int top_k_experts,
    int moe_intermediate_size,
    int intermediate_size,
    // Logit softcapping
    float final_logit_softcapping,
    // Per-layer type flags: 1 = global, 0 = sliding
    const int* layer_types,
    int layer_types_len,
    // Cache arrays and offset
    mlx_array** cache_arrays,
    int prefill_offset
) {
  try {
    // Build config
    g_gemma4_config = Gemma4Config{};
    g_gemma4_config.num_layers = num_layers;
    g_gemma4_config.hidden_size = hidden_size;
    g_gemma4_config.num_heads = num_heads;
    g_gemma4_config.num_kv_heads = num_kv_heads;
    g_gemma4_config.head_dim = head_dim;
    g_gemma4_config.global_num_kv_heads = global_num_kv_heads;
    g_gemma4_config.global_head_dim = global_head_dim;
    g_gemma4_config.rope_theta = rope_theta;
    g_gemma4_config.rope_local_base_freq = rope_local_base_freq;
    g_gemma4_config.partial_rotary_factor = partial_rotary_factor;
    g_gemma4_config.rms_norm_eps = rms_norm_eps;
    g_gemma4_config.sliding_window = sliding_window;
    g_gemma4_config.tie_word_embeddings = (tie_word_embeddings != 0);
    g_gemma4_config.max_kv_len = max_kv_len;
    g_gemma4_config.batch_size = batch_size;
    g_gemma4_config.num_experts = num_experts;
    g_gemma4_config.top_k_experts = top_k_experts;
    g_gemma4_config.moe_intermediate_size = moe_intermediate_size;
    g_gemma4_config.intermediate_size = intermediate_size;
    g_gemma4_config.final_logit_softcapping = final_logit_softcapping;

    // Build per-layer type flags
    g_gemma4_config.is_global_layer.clear();
    g_gemma4_config.is_global_layer.reserve(num_layers);
    for (int i = 0; i < num_layers; i++) {
      bool is_global = (i < layer_types_len) ? (layer_types[i] != 0) : false;
      g_gemma4_config.is_global_layer.push_back(is_global);
    }

    const auto& cfg = g_gemma4_config;

    // Pre-compute frequencies for proportional RoPE on global layers.
    //
    // Python ProportionalRoPE computes:
    //   exponents = arange(0, rotated_dims, 2) / dims   # dims = global_head_dim
    //   _freqs = factor * base^exponents
    //
    // fast::rope internally does: inv_freqs = reciprocal(freqs)
    //   then theta = positions * inv_freqs, cos/sin applied on GPU.
    {
      int rotary_dim = (int)(global_head_dim * partial_rotary_factor);
      int rope_angles = rotary_dim / 2;
      g_rotated_dims = 2 * rope_angles;  // e.g. 128 for partial_rotary_factor=0.25, head_dim=512

      // g_rope_freqs_storage: 1-D [rope_angles] -- for fast::rope freqs param
      // freqs[i] = factor * base^(2i / dims)  (factor=1.0 for proportional type)
      std::vector<float> freqs_data(rope_angles);
      for (int i = 0; i < rope_angles; i++) {
        float exponent = (2.0f * i) / (float)global_head_dim;
        freqs_data[i] = std::pow(rope_theta, exponent);  // = 1.0 / inv_freq[i]
      }
      g_rope_freqs_storage.clear();
      g_rope_freqs_storage.push_back(array(freqs_data.data(), {rope_angles}, mlx::core::float32));
    }

    // Import caches
    g_gemma4_caches.clear();
    g_gemma4_caches.reserve(num_layers * 2);

    for (int i = 0; i < num_layers; i++) {
      if (!cache_arrays[i * 2] || !cache_arrays[i * 2 + 1]) {
        g_gemma4_inited = false;
        return;
      }
      auto& ck = *reinterpret_cast<array*>(cache_arrays[i * 2]);
      auto& cv = *reinterpret_cast<array*>(cache_arrays[i * 2 + 1]);

      bool is_global = cfg.is_global_layer[i];
      int kv_heads = is_global ? cfg.global_num_kv_heads : cfg.num_kv_heads;
      int hd = is_global ? cfg.global_head_dim : cfg.head_dim;

      // Pad ALL caches to max_kv_len (matching Qwen3.5 MoE's uniform cache strategy).
      // This ensures all layers have the same cache seq_len dimension, which:
      // 1. Allows a SINGLE shared mask for all SDPA calls
      // 2. Enables compile() to cache the Metal command buffer efficiently
      // Sliding layers use circular writes via remainder(offset, window) regardless.
      int target_len = max_kv_len;
      int current_cap = ck.shape(2);
      if (current_cap < target_len) {
        int pad_len = target_len - current_cap;
        auto kpad = zeros({batch_size, kv_heads, pad_len, hd}, ck.dtype());
        auto vpad = zeros({batch_size, kv_heads, pad_len, hd}, cv.dtype());
        g_gemma4_caches.push_back(concatenate({ck, kpad}, 2));
        g_gemma4_caches.push_back(concatenate({cv, vpad}, 2));
      } else {
        g_gemma4_caches.push_back(ck);
        g_gemma4_caches.push_back(cv);
      }
    }

    g_gemma4_offset_int = prefill_offset;
    g_gemma4_inited = true;

    // Pre-compute 3D transposes for all expert weights [E,out,in] -> [E,in,out].
    g_weight_transposes_3d.clear();
    {
      std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
      for (const auto& [name, w] : g_weights()) {
        if (w.ndim() == 3) {
          g_weight_transposes_3d.insert_or_assign(name, transpose(w, {0, 2, 1}));
        }
      }
    }

    // Eval all caches to materialize them AND break any lazy graph chains
    // from the prefill. Without this, each decode step's eval must traverse
    // back through the entire prefill graph for each cache.
    {
      std::vector<array> to_eval;
      to_eval.reserve(g_gemma4_caches.size());
      for (const auto& c : g_gemma4_caches) to_eval.push_back(c);
      mlx::core::eval(std::move(to_eval));
    }

    // Replace each cache with a FRESH contiguous copy that has no graph
    // ancestry. This ensures the decode loop's slice_update creates minimal
    // graph nodes (no ancestry chain from prefill to traverse).
    for (auto& c : g_gemma4_caches) {
      c = mlx::core::copy(c);
      mlx::core::eval({c});
    }

    // Break the lazy RNG split chain
    auto rng_key = mlx::core::random::KeySequence::default_().next();
    mlx::core::eval({rng_key});
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_gemma4_init_from_prefill: " << e.what() << std::endl;
    g_gemma4_inited = false;
  }
}

// Gemma4 single-token decode step (compiled path by default)
void mlx_gemma4_forward(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    mlx_array** output_logits,
    int* cache_offset_out
) {
  if (!g_gemma4_inited) {
    *output_logits = nullptr;
    return;
  }
  const auto& cfg = g_gemma4_config;

  try {
    auto& input_ids      = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    // Python: h = self.embed_tokens(inputs); h = h * self.embed_scale
    auto flat_ids = reshape(input_ids, {-1});
    auto h = take(embedding_weight, flat_ids, 0);
    auto embed_scale = array(std::sqrt((float)cfg.hidden_size), h.dtype());
    h = h * embed_scale;

    // Build inputs for the compilable function
    std::vector<array> fn_inputs;
    fn_inputs.reserve(2 + cfg.num_layers * 2);
    fn_inputs.push_back(std::move(h));
    fn_inputs.push_back(array(g_gemma4_offset_int, mlx::core::int32));
    for (const auto& c : g_gemma4_caches) {
      fn_inputs.push_back(c);
    }

    // MLX_NO_COMPILE=1 disables compilation for A/B testing
    static bool no_compile = std::getenv("MLX_NO_COMPILE") != nullptr;
    auto outputs = no_compile
        ? gemma4_compiled_decode_fn(fn_inputs)
        : compiled_gemma4_decode()(fn_inputs);

    // Extract outputs: [logits, new_offset, new_caches...]
    *output_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    g_gemma4_offset_int++;
    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_gemma4_caches[i] = outputs[2 + i];
    }

    if (cache_offset_out) {
      *cache_offset_out = g_gemma4_offset_int;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_gemma4_forward: %s\n", e.what());
    fflush(stderr);
    *output_logits = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_gemma4_forward\n");
    fflush(stderr);
    *output_logits = nullptr;
  }
}

// Gemma4 single-token greedy decode step.
void mlx_gemma4_forward_greedy(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    mlx_array** output_token,
    int* cache_offset_out
) {
  if (!g_gemma4_inited) {
    *output_token = nullptr;
    return;
  }
  const auto& cfg = g_gemma4_config;

  try {
    auto& input_ids = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    auto flat_ids = reshape(input_ids, {-1});
    auto h = take(embedding_weight, flat_ids, 0);
    auto embed_scale = array(std::sqrt((float)cfg.hidden_size), h.dtype());
    h = h * embed_scale;

    std::vector<array> fn_inputs;
    fn_inputs.reserve(2 + cfg.num_layers * 2);
    fn_inputs.push_back(std::move(h));
    fn_inputs.push_back(array(g_gemma4_offset_int, mlx::core::int32));
    for (const auto& c : g_gemma4_caches) {
      fn_inputs.push_back(c);
    }

    static bool no_compile = std::getenv("MLX_NO_COMPILE") != nullptr;
    auto outputs = no_compile
        ? gemma4_compiled_decode_greedy_fn(fn_inputs)
        : compiled_gemma4_decode_greedy()(fn_inputs);

    *output_token = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    g_gemma4_offset_int++;
    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_gemma4_caches[i] = outputs[2 + i];
    }

    if (cache_offset_out) {
      *cache_offset_out = g_gemma4_offset_int;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_gemma4_forward_greedy: %s\n", e.what());
    fflush(stderr);
    *output_token = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_gemma4_forward_greedy\n");
    fflush(stderr);
    *output_token = nullptr;
  }
}

// Full decode loop entirely in C++ -- no per-step Rust round-trip.
// Generates up to max_tokens, stops at any EOS token.
// Returns number of tokens generated. Token IDs written to out_tokens.
int mlx_gemma4_generate(
    mlx_array* first_token_ptr,
    mlx_array* embedding_weight_ptr,
    int max_tokens,
    float temperature,
    const int* eos_ids,
    int num_eos_ids,
    int* out_tokens
) {
  if (!g_gemma4_inited) return 0;
  const auto& cfg = g_gemma4_config;

  try {
    auto y = *reinterpret_cast<array*>(first_token_ptr);
    auto embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);
    auto embed_scale = array(std::sqrt((float)cfg.hidden_size), mlx::core::bfloat16);

    std::vector<int> eos_vec(eos_ids, eos_ids + num_eos_ids);

    int generated = 0;

    for (int step = 0; step < max_tokens; step++) {
      auto flat_ids = reshape(y, {-1});
      auto h = take(embedding_weight, flat_ids, 0);
      h = h * embed_scale;

      // Build inputs for the compiled function
      std::vector<array> fn_inputs;
      fn_inputs.reserve(2 + cfg.num_layers * 2);
      fn_inputs.push_back(std::move(h));
      fn_inputs.push_back(array(g_gemma4_offset_int, mlx::core::int32));
      for (const auto& c : g_gemma4_caches) {
        fn_inputs.push_back(c);
      }

      static bool no_compile = std::getenv("MLX_NO_COMPILE") != nullptr;
      auto outputs = no_compile
          ? gemma4_compiled_decode_fn(fn_inputs)
          : compiled_gemma4_decode()(fn_inputs);

      auto logits = outputs[0];
      g_gemma4_offset_int++;
      for (int i = 0; i < cfg.num_layers * 2; i++) {
        g_gemma4_caches[i] = outputs[2 + i];
      }

      // Sample
      if (temperature <= 0.0f) {
        y = argmax(logits, -1);
      } else {
        y = mlx::core::random::categorical(logits * array(1.0f / temperature, logits.dtype()));
      }
      y = reshape(y, {-1});

      // Eval token (caches materialize through dependency graph)
      mlx::core::eval({y});
      int token_id = y.item<int>();
      out_tokens[generated++] = token_id;

      // Check EOS
      bool is_eos = false;
      for (int eid : eos_vec) { if (token_id == eid) { is_eos = true; break; } }
      if (is_eos) break;
    }

    return generated;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_gemma4_generate: %s\n", e.what());
    fflush(stderr);
    return 0;
  }
}

// Eval token + caches.
//
// With compile(), evaluating the token (which depends on logits from the compiled
// graph) triggers the entire compiled graph, materializing all cache arrays.
// This matches Python's mx.async_eval(y) pattern.
//
// Without compile (MLX_NO_COMPILE=1), caches are independent graph outputs
// that must be explicitly eval'd. Always eval all caches for safety.
void mlx_gemma4_eval_token_and_caches(mlx_array* next_token_ptr) {
  try {
    static bool no_compile = std::getenv("MLX_NO_COMPILE") != nullptr;
    if (no_compile) {
      // Non-compiled: must eval token + ALL caches explicitly
      std::vector<array> to_eval;
      to_eval.reserve(1 + g_gemma4_caches.size());
      to_eval.push_back(*reinterpret_cast<array*>(next_token_ptr));
      for (const auto& c : g_gemma4_caches) {
        to_eval.push_back(c);
      }
      mlx::core::async_eval(std::move(to_eval));
    } else {
      // Compiled: eval token only -- caches materialize through dependency graph
      mlx::core::async_eval({*reinterpret_cast<array*>(next_token_ptr)});
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_gemma4_eval_token_and_caches: %s\n", e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_gemma4_eval_token_and_caches\n");
    fflush(stderr);
  }
}

// Synchronously eval all caches (for periodic memory management).
void mlx_gemma4_sync_eval_caches() {
  try {
    if (g_gemma4_caches.empty()) return;
    std::vector<array> to_eval;
    to_eval.reserve(g_gemma4_caches.size());
    for (const auto& c : g_gemma4_caches) {
      to_eval.push_back(c);
    }
    mlx::core::eval(std::move(to_eval));
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in gemma4_sync_eval_caches: %s\n", e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in gemma4_sync_eval_caches\n");
    fflush(stderr);
  }
}

// Benchmark: run N decode steps entirely in C++ with per-step eval.
double mlx_gemma4_benchmark(int num_steps) {
  if (!g_gemma4_inited) return -1.0;
  const auto& cfg = g_gemma4_config;

  try {
    auto y = array(100, mlx::core::int32);
    auto embedding_weight = get_weight("embed_tokens.weight");
    auto embed_scale = array(std::sqrt((float)cfg.hidden_size), mlx::core::bfloat16);

    // Warmup: 2 steps
    for (int w = 0; w < 2; w++) {
      auto flat_ids = reshape(y, {-1});
      auto h = take(embedding_weight, flat_ids, 0) * embed_scale;

      std::vector<array> fn_inputs;
      fn_inputs.reserve(2 + cfg.num_layers * 2);
      fn_inputs.push_back(std::move(h));
      fn_inputs.push_back(array(g_gemma4_offset_int, mlx::core::int32));
      for (const auto& c : g_gemma4_caches) fn_inputs.push_back(c);

      auto outputs = compiled_gemma4_decode()(fn_inputs);
      y = argmax(outputs[0], -1);
      g_gemma4_offset_int++;
      for (int i = 0; i < cfg.num_layers * 2; i++) {
        g_gemma4_caches[i] = outputs[2 + i];
      }
      mlx::core::eval({y});
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < num_steps; step++) {
      auto flat_ids = reshape(y, {-1});
      auto h = take(embedding_weight, flat_ids, 0) * embed_scale;

      std::vector<array> fn_inputs;
      fn_inputs.reserve(2 + cfg.num_layers * 2);
      fn_inputs.push_back(std::move(h));
      fn_inputs.push_back(array(g_gemma4_offset_int, mlx::core::int32));
      for (const auto& c : g_gemma4_caches) fn_inputs.push_back(c);

      auto outputs = compiled_gemma4_decode()(fn_inputs);
      y = argmax(outputs[0], -1);
      g_gemma4_offset_int++;
      for (int i = 0; i < cfg.num_layers * 2; i++) {
        g_gemma4_caches[i] = outputs[2 + i];
      }
      mlx::core::eval({y});
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    fprintf(stderr, "[gemma4 C++ bench] %d steps: %.0fms (%.0fms/step = %.1f tok/s)\n",
        num_steps, ms, ms / num_steps, num_steps / (ms / 1000.0));
    return ms;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Benchmark error: %s\n", e.what());
    return -1.0;
  }
}

// Reset Gemma4 state
void mlx_gemma4_reset() {
  g_gemma4_caches.clear();
  g_gemma4_offset_int = 0;
  g_gemma4_inited = false;
  g_weight_transposes_3d.clear();
  g_rope_freqs_storage.clear();
  g_gemma4_config.is_global_layer.clear();
}

// Export caches for PromptCache reuse.
int mlx_gemma4_export_caches(mlx_array** out_ptrs, int max_count) {
  if (!g_gemma4_inited || g_gemma4_caches.empty()) return 0;
  int count = std::min((int)g_gemma4_caches.size(), max_count);
  for (int i = 0; i < count; i++) {
    out_ptrs[i] = reinterpret_cast<mlx_array*>(new array(g_gemma4_caches[i]));
  }
  return count;
}

// Get current cache offset (number of tokens processed).
int mlx_gemma4_get_cache_offset() {
  return g_gemma4_offset_int;
}

// Adjust cache offset by delta (for VLM position correction).
void mlx_gemma4_adjust_offset(int delta) {
  g_gemma4_offset_int += delta;
}

}  // extern "C"
