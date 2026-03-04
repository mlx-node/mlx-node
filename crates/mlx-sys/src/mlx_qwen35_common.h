#pragma once

// =============================================================================
// Shared helpers for Qwen3.5 dense and MoE forward passes.
//
// Contains: global weight storage, BaseConfig, compiled helper functions,
// Metal gated delta kernel, and pure GDN/attention forward functions.
//
// Both mlx_qwen35.cpp (dense) and mlx_qwen35_moe.cpp (MoE) include this.
// All functions are inline to avoid ODR violations across translation units.
// =============================================================================

#include "mlx_common.h"

#include <string>
#include <unordered_map>
#include <mutex>

namespace qwen35_common {

// =====================================================================
// Global weight storage (shared between dense and MoE)
// =====================================================================

inline std::unordered_map<std::string, array>& g_weights() {
  static std::unordered_map<std::string, array> instance;
  return instance;
}

inline std::mutex& g_weights_mutex() {
  static std::mutex instance;
  return instance;
}

inline std::unordered_map<std::string, array>& g_weight_transposes() {
  static std::unordered_map<std::string, array> instance;
  return instance;
}

inline const array& get_weight(const std::string& name) {
  auto it = g_weights().find(name);
  if (it == g_weights().end()) {
    throw std::runtime_error("Weight not found: " + name);
  }
  return it->second;
}

inline const array& get_weight_t(const std::string& name) {
  auto it = g_weight_transposes().find(name);
  if (it != g_weight_transposes().end()) {
    return it->second;
  }
  const auto& w = get_weight(name);
  auto wt = transpose(w);
  auto [inserted_it, _] = g_weight_transposes().emplace(name, std::move(wt));
  return inserted_it->second;
}

inline bool has_weight(const std::string& name) {
  return g_weights().count(name) > 0;
}

// Auto-detecting linear projection: uses quantized_matmul if .scales exists,
// otherwise plain matmul with transposed weight. Safe for both dense (bf16)
// and quantized (MXFP8/affine) weights.
//
// Quant param heuristic: the presence of .biases distinguishes the format:
//   - No .biases → MXFP8 quantization (group_size=32, bits=8, mode="mxfp8")
//   - Has .biases → Affine quantization (group_size=64, bits=4, mode="affine")
// This is correct for all currently supported Qwen3.5 model formats because:
//   - MXFP8 (mlx-community fp8 models) never stores biases
//   - Affine 4-bit (mlx-community 4bit models) always stores biases
// The MoE path uses explicit quant param detection (detect_quant) instead.
inline array linear_proj(const array& x, const std::string& prefix) {
  std::string scales_key = prefix + ".scales";
  if (has_weight(scales_key)) {
    const auto& w = get_weight(prefix + ".weight");
    const auto& scales = get_weight(scales_key);
    std::string biases_key = prefix + ".biases";
    std::optional<array> biases = std::nullopt;
    if (has_weight(biases_key)) {
      biases = get_weight(biases_key);
    }
    // Detect MXFP8 vs affine: no biases → MXFP8 (gs=32, bits=8, mode="mxfp8")
    if (!biases.has_value()) {
      return mlx::core::quantized_matmul(x, w, scales, biases, true, 32, 8, "mxfp8");
    } else {
      return mlx::core::quantized_matmul(x, w, scales, biases, true, 64, 4, "affine");
    }
  }
  return matmul(x, get_weight_t(prefix + ".weight"));
}

// =====================================================================
// Base config (shared fields between dense CompileConfig and MoeConfig)
// =====================================================================

struct BaseConfig {
  int num_layers;
  int hidden_size;
  int num_heads;
  int num_kv_heads;
  int head_dim;
  float rope_theta;
  int rope_dims;
  float rms_norm_eps;
  int full_attention_interval;
  int linear_num_k_heads;
  int linear_num_v_heads;
  int linear_key_head_dim;
  int linear_value_head_dim;
  int linear_conv_kernel_dim;
  bool tie_word_embeddings;
  int max_kv_len;
  int batch_size;
};

// =====================================================================
// Compiled helper functions
// =====================================================================

inline array rms_norm_no_weight(const array& x, float eps) {
  return fast::rms_norm(x, std::nullopt, eps);
}

inline array silu(const array& x) {
  return x * sigmoid(x);
}

// SwiGLU: silu(gate) * up — compiled for kernel fusion
inline std::vector<array> swiglu_impl(const std::vector<array>& inputs) {
  const auto& gate = inputs[0];
  const auto& up = inputs[1];
  return {sigmoid(gate) * gate * up};
}

inline auto& compiled_swiglu() {
  static auto fn = mlx::core::compile(swiglu_impl, /*shapeless=*/true);
  return fn;
}

inline array swiglu(const array& gate, const array& up) {
  return compiled_swiglu()({gate, up})[0];
}

// Numerically stable softplus: where(x > 20, x, log1p(exp(x)))
// Naive log(exp(x)+1) overflows for large x in bf16/f16 (max ~65504).
// Threshold of 20 is well above typical values but well below bf16 overflow.
inline array softplus(const array& x) {
  return mlx::core::where(
      mlx::core::greater(x, array(20.0f, x.dtype())),
      x,
      mlx::core::log1p(exp(x)));
}

// Fused compute_g: g = exp(-exp(A_log) * softplus(a + dt_bias))
// A_log and dt_bias are f32 (from HF checkpoints). Computation is in f32.
// Result is cast back to a.dtype (bf16) to match Python mlx-lm behavior.
inline std::vector<array> compute_g_impl(const std::vector<array>& inputs) {
  const auto& a_log = inputs[0];
  const auto& a = inputs[1];
  const auto& dt_bias = inputs[2];
  auto A = exp(a_log);
  auto sp = softplus(mlx::core::add(a, dt_bias));
  auto result = exp(negative(A * sp));
  return {astype(result, a.dtype())};
}

inline auto& compiled_compute_g() {
  static auto fn = mlx::core::compile(compute_g_impl, /*shapeless=*/true);
  return fn;
}

inline array fused_compute_g(const array& a_log, const array& a, const array& dt_bias) {
  return compiled_compute_g()({a_log, a, dt_bias})[0];
}

// SiLU activation fused with multiply: silu(x) * y = sigmoid(x) * x * y
inline std::vector<array> silu_mul_impl(const std::vector<array>& inputs) {
  const auto& x = inputs[0];
  const auto& y = inputs[1];
  return {sigmoid(x) * x * y};
}
inline auto& compiled_silu_mul() {
  static auto fn = mlx::core::compile(silu_mul_impl, /*shapeless=*/true);
  return fn;
}

// silu(x): sigmoid(x) * x — compiled
inline std::vector<array> silu_impl2(const std::vector<array>& inputs) {
  const auto& x = inputs[0];
  return {sigmoid(x) * x};
}
inline auto& compiled_silu() {
  static auto fn = mlx::core::compile(silu_impl2, /*shapeless=*/true);
  return fn;
}

// attn_gate: attn_out * sigmoid(gate) — compiled
inline std::vector<array> attn_gate_impl(const std::vector<array>& inputs) {
  return {inputs[0] * sigmoid(inputs[1])};
}
inline auto& compiled_attn_gate() {
  static auto fn = mlx::core::compile(attn_gate_impl, /*shapeless=*/true);
  return fn;
}

// =====================================================================
// Metal kernel for gated delta recurrence
// =====================================================================

inline std::mutex& qwen35_kernel_mutex() {
  static std::mutex instance;
  return instance;
}

inline std::unique_ptr<mlx::core::fast::CustomKernelFunction>& qwen35_gd_kernel() {
  static std::unique_ptr<mlx::core::fast::CustomKernelFunction> instance;
  return instance;
}

inline void ensure_gated_delta_kernel() {
  std::lock_guard<std::mutex> lock(qwen35_kernel_mutex());
  if (qwen35_gd_kernel()) return;

  static const char* source =
    #include "metal/gated_delta_step.metal.inc"
  ;

  auto kernel = fast::metal_kernel(
      "gated_delta_step",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out"},
      source
  );
  qwen35_gd_kernel() = std::make_unique<mlx::core::fast::CustomKernelFunction>(std::move(kernel));
}

inline std::pair<array, array> gated_delta_kernel_call(
    const array& q, const array& k, const array& v,
    const array& g, const array& beta_arr,
    const array& state) {
  ensure_gated_delta_kernel();

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = q.shape(2);
  int Dk = q.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);
  auto input_type = q.dtype();
  auto T_arr = array(T, mlx::core::int32);

  auto results = (*qwen35_gd_kernel())(
      {q, k, v, g, beta_arr, state, T_arr},
      {Shape{B, T, Hv, Dv}, state.shape()},
      {input_type, input_type},
      std::make_tuple(32, Dv, B * Hv),
      std::make_tuple(32, 4, 1),
      {{"InT", input_type}, {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
      std::nullopt, false,
      mlx::core::default_stream(mlx::core::Device::gpu)
  );

  return std::pair<array, array>(std::move(results[0]), std::move(results[1]));
}

// =====================================================================
// Pure GDN forward (shared between dense compiled and MoE non-compiled)
// =====================================================================

struct GDNPureResult { array output, conv_state, recurrent_state; };

inline GDNPureResult gdn_pure_fn(
    const array& x,              // [B, hidden] — 2D
    int layer_idx,
    const array& conv_state,      // [B, kernel-1, conv_dim]
    const array& recurrent_state, // [B, Hv, Dv, Dk]
    const BaseConfig& cfg) {
  int B = x.shape(0);
  int key_dim = cfg.linear_num_k_heads * cfg.linear_key_head_dim;
  int value_dim = cfg.linear_num_v_heads * cfg.linear_value_head_dim;
  int conv_dim = key_dim * 2 + value_dim;

  std::string pfx = "layers." + std::to_string(layer_idx) + ".linear_attn.";

  // Projections — auto-detecting quantized vs dense weights
  // Merged projections (bf16): in_proj_qkvz, in_proj_ba
  // Separate projections (quantized): in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
  struct QkvzResult { array qkv, z; };
  auto [qkv, z] = [&]() -> QkvzResult {
    if (has_weight(pfx + "in_proj_qkvz.weight")) {
      auto qkvz = linear_proj(x, pfx + "in_proj_qkvz");
      return {slice(qkvz, {0, 0}, {B, conv_dim}),
              slice(qkvz, {0, conv_dim}, {B, key_dim * 2 + value_dim * 2})};
    }
    return {linear_proj(x, pfx + "in_proj_qkv"),
            linear_proj(x, pfx + "in_proj_z")};
  }();
  struct BaResult { array b, a; };
  auto [b, a] = [&]() -> BaResult {
    if (has_weight(pfx + "in_proj_ba.weight")) {
      auto ba = linear_proj(x, pfx + "in_proj_ba");
      return {slice(ba, {0, 0}, {B, cfg.linear_num_v_heads}),
              slice(ba, {0, cfg.linear_num_v_heads}, {B, cfg.linear_num_v_heads * 2})};
    }
    return {linear_proj(x, pfx + "in_proj_b"),
            linear_proj(x, pfx + "in_proj_a")};
  }();

  // Reshape qkv to 3D for conv1d
  auto qkv_3d = reshape(qkv, {B, 1, conv_dim});

  auto conv_input = concatenate({conv_state, qkv_3d}, 1);

  int total_len = cfg.linear_conv_kernel_dim;
  int keep = cfg.linear_conv_kernel_dim - 1;
  auto new_conv_state = slice(conv_input, {0, total_len - keep, 0}, {B, total_len, conv_dim});

  const auto& conv_w = get_weight(pfx + "conv1d.weight");
  auto conv_out = mlx::core::conv1d(conv_input, conv_w, 1, 0, 1, conv_dim);

  // SiLU — compiled
  conv_out = compiled_silu()({conv_out})[0];

  // Split into q, k, v
  auto q = slice(conv_out, {0, 0, 0},              {B, 1, key_dim});
  auto k = slice(conv_out, {0, 0, key_dim},        {B, 1, key_dim * 2});
  auto v = slice(conv_out, {0, 0, key_dim * 2},    {B, 1, conv_dim});

  // Reshape to heads
  q = reshape(q, {B, 1, cfg.linear_num_k_heads, cfg.linear_key_head_dim});
  k = reshape(k, {B, 1, cfg.linear_num_k_heads, cfg.linear_key_head_dim});
  v = reshape(v, {B, 1, cfg.linear_num_v_heads, cfg.linear_value_head_dim});

  // RMS norm + scaling
  float inv_s = std::pow((float)cfg.linear_key_head_dim, -0.5f);
  auto q_dt = q.dtype();
  q = rms_norm_no_weight(q, 1e-6f) * array(inv_s * inv_s, q_dt);
  k = rms_norm_no_weight(k, 1e-6f) * array(inv_s, q_dt);

  // Beta, g
  auto beta_3d = reshape(sigmoid(b), {B, 1, cfg.linear_num_v_heads});
  const auto& a_log  = get_weight(pfx + "A_log");
  const auto& dt_b   = get_weight(pfx + "dt_bias");
  auto g_3d = reshape(fused_compute_g(a_log, a, dt_b), {B, 1, cfg.linear_num_v_heads});

  // Metal kernel
  auto [y, new_recurrent_state] = gated_delta_kernel_call(q, k, v, g_3d, beta_3d, recurrent_state);

  // RMSNorm gating
  auto z_h = reshape(z, {B, 1, cfg.linear_num_v_heads, cfg.linear_value_head_dim});
  const auto& nw = get_weight(pfx + "norm.weight");
  auto y_normed = fast::rms_norm(y, nw, cfg.rms_norm_eps);
  y_normed = compiled_silu_mul()({z_h, y_normed})[0];

  // Output projection
  auto y_flat = reshape(y_normed, {B, value_dim});
  auto output = linear_proj(y_flat, pfx + "out_proj");

  return {output, new_conv_state, new_recurrent_state};
}

// =====================================================================
// Pure attention forward (shared between dense compiled and MoE non-compiled)
// =====================================================================

struct AttnPureResult { array output, keys, values; };

inline AttnPureResult attn_pure_fn(
    const array& x,          // [B, hidden] — 2D
    int layer_idx,
    const array& kv_keys,    // [B, Hkv, max_kv_len, D]
    const array& kv_values,  // [B, Hkv, max_kv_len, D]
    const array& attn_mask,  // [1, 1, 1, max_kv_len] additive mask (ignored when dynamic_kv=true)
    int offset,
    const BaseConfig& cfg,
    bool dynamic_kv = false) {  // true = slice KV to valid range, skip mask (MoE path)
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

  // RoPE
  queries = fast::rope(queries, cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset);
  keys    = fast::rope(keys,    cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset);

  // Transpose for SDPA
  queries = transpose(queries, {0, 2, 1, 3});
  keys    = transpose(keys,    {0, 2, 1, 3});
  values  = transpose(values,  {0, 2, 1, 3});

  // KV cache update
  auto offset_1d = reshape(array(offset, mlx::core::int32), {1});
  auto new_kv_keys   = mlx::core::slice_update(kv_keys,   keys,   offset_1d, {2});
  auto new_kv_values = mlx::core::slice_update(kv_values, values, offset_1d, {2});

  // SDPA
  float scale = std::pow((float)cfg.head_dim, -0.5f);
  auto attn_out = [&]() -> array {
    if (dynamic_kv) {
      // MoE path: slice KV cache to valid range [0..offset+1], pass no mask → faster SDPA kernel
      int valid_len = offset + 1;
      auto valid_keys   = slice(new_kv_keys,   {0, 0, 0, 0}, {B, cfg.num_kv_heads, valid_len, cfg.head_dim});
      auto valid_values = slice(new_kv_values, {0, 0, 0, 0}, {B, cfg.num_kv_heads, valid_len, cfg.head_dim});
      return fast::scaled_dot_product_attention(
          queries, valid_keys, valid_values, scale, "", std::nullopt, {});
    } else {
      // Dense compiled path: fixed shapes + additive mask (required for mlx::core::compile)
      return fast::scaled_dot_product_attention(
          queries, new_kv_keys, new_kv_values, scale, "", attn_mask, {});
    }
  }();

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

}  // namespace qwen35_common
