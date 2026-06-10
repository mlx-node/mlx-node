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
#include "mlx_paged_ops.h"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

// Cross-TU reload-invalidation hooks for the MTP compiled draft/commit graphs.
// Defined in mlx_qwen35_mtp_compiled.cpp / mlx_qwen35_moe_mtp_compiled.cpp;
// called transitively from the dense / MoE main `*_invalidate_compiled_graphs`
// reload entry points so a model reload re-traces every weight-baking graph.
extern "C" void mlx_qwen35_mtp_invalidate_compiled_graphs();
extern "C" void mlx_qwen35_moe_mtp_invalidate_compiled_graphs();

namespace qwen35_common {

// =====================================================================
// MTP control-flow tracing
// =====================================================================
//
// Gated by `MLX_MTP_TRACE`: set to `1` / `true` / `on` (case-insensitive,
// surrounding whitespace ignored) to emit C++-side ENTER/EXIT traces for
// the MTP draft / verify entrypoints. Default OFF so production decode is
// not flooded. Truthy parsing matches the Rust `MLX_MTP_*` readers so the
// convention is uniform.
inline bool mtp_trace_enabled() {
  static const bool enabled = []() {
    const char* raw = std::getenv("MLX_MTP_TRACE");
    if (!raw) return false;
    std::string v(raw);
    size_t s = 0;
    while (s < v.size() && std::isspace(static_cast<unsigned char>(v[s]))) s++;
    size_t e = v.size();
    while (e > s && std::isspace(static_cast<unsigned char>(v[e - 1]))) e--;
    std::string trimmed = v.substr(s, e - s);
    for (char& c : trimmed) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return trimmed == "1" || trimmed == "true" || trimmed == "on";
  }();
  return enabled;
}

// =====================================================================
// Global weight storage (shared between dense and MoE)
// =====================================================================

inline std::unordered_map<std::string, array>& g_weights() {
  static std::unordered_map<std::string, array> instance;
  return instance;
}

inline std::shared_mutex& g_weights_mutex() {
  static std::shared_mutex instance;
  return instance;
}

inline std::unordered_map<std::string, array>& g_weight_transposes() {
  static std::unordered_map<std::string, array> instance;
  return instance;
}

// Model identity: atomic counter set after all weights are stored.
// Inference checks this against its own model_id to avoid using another model's weights.
// Value 0 means no model has registered weights.
inline std::atomic<uint64_t>& g_active_model_id() {
  static std::atomic<uint64_t> instance{0};
  return instance;
}

// Returns by VALUE (refcount bump) so the caller's copy survives even if a
// concurrent writer clears the map. MLX arrays are refcounted handles — cheap to copy.
inline array get_weight(const std::string& name) {
  std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
  auto it = g_weights().find(name);
  if (it == g_weights().end()) {
    throw std::runtime_error("Weight not found: " + name);
  }
  return it->second;  // copy under lock
}

// Pure read — transposes are pre-computed during weight registration.
// Returns by VALUE for same reason as get_weight().
inline array get_weight_t(const std::string& name) {
  std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
  auto it = g_weight_transposes().find(name);
  if (it != g_weight_transposes().end()) {
    return it->second;  // copy under lock
  }
  throw std::runtime_error("Transpose not found for weight: " + name);
}

inline bool has_weight(const std::string& name) {
  std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
  return g_weights().count(name) > 0;
}

// =====================================================================
// Quantization metadata registry
//
// Sidecar map keyed by projection prefix (NO `.weight`/`.scales`/`.biases`
// suffix), e.g. "layers.3.mlp.switch_mlp.gate_proj". Holds the authoritative
// per-layer (mode, bits, group_size) supplied by Rust at weight-registration
// time so the compiled forward path can dispatch correctly without inferring
// from companion-tensor presence (which conflates MXFP4/MXFP8/NVFP4).
//
// Shares g_weights_mutex() — a single lock at registration covers both maps.
// =====================================================================

struct QuantInfo {
  std::string mode;  // "affine" | "mxfp8" | "mxfp4" | "nvfp4" | "sym8"
  int bits;
  int group_size;
};

inline std::unordered_map<std::string, QuantInfo>& g_quant_info() {
  static std::unordered_map<std::string, QuantInfo> instance;
  return instance;
}

inline std::optional<QuantInfo> lookup_quant_info(const std::string& prefix) {
  std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
  auto it = g_quant_info().find(prefix);
  if (it == g_quant_info().end()) return std::nullopt;
  return it->second;  // copy under lock
}

// Infer affine quantization bits from weight and scales shapes.
// For affine: weight_cols = original_cols * bits / 32,
//             scales_cols = original_cols / group_size
// So: bits = (weight_cols * 32 * group_size) / (scales_cols * 32 * group_size / bits)
//         = weight_cols * group_size / (scales_cols * 32 / bits)
// Simpler: original_cols = scales_cols * group_size,
//          bits = weight_cols * 32 / original_cols
inline int infer_affine_bits(const array& w, const array& scales, int group_size = 64) {
  int w_cols = w.shape(-1);
  int s_cols = scales.shape(-1);
  int original_cols = s_cols * group_size;
  return (w_cols * 32) / original_cols;
}

// Auto-detecting linear projection.
//
// Dispatch order:
//   1. If Rust registered (mode, bits, group_size) for this prefix via
//      `mlx_store_quant_info`, use those values verbatim. This is the
//      authoritative path — Rust knows the true quantization tuple from
//      config.json + per-layer overrides + recipe upgrades.
//   2. Otherwise fall back to a heuristic that infers MXFP8 vs affine
//      from companion-tensor presence. Preserves correctness for any
//      caller that has not (yet) plumbed quant-info registration. A
//      one-shot log fires the first time the fallback runs per process
//      so it's visible when the registry path is unexpectedly bypassed.
//   3. If no `.scales` companion exists at all, the weight is bf16/f16
//      and we use plain matmul against the pre-transposed weight.
// Set + mutex tracking which `linear_proj` prefixes have already
// emitted a one-shot dispatch trace. Lazily initialised so the linker
// instantiates a single shared copy across all TUs that include this
// header.
inline std::unordered_set<std::string>& g_linear_proj_logged_prefixes() {
  static std::unordered_set<std::string> instance;
  return instance;
}

inline std::mutex& g_linear_proj_logged_mutex() {
  static std::mutex instance;
  return instance;
}

// sym8 projection (per-output-channel symmetric int8, W8A8 kernels).
//
// Layout contract (decided at registration, see Rust
// `register_weights_with_cpp` in qwen3_5/persistence.rs): for sym8 prefixes
// Rust stores the CONTIGUOUS [K,N] int8 KERNEL OPERAND as `{prefix}.weight`
// (NOT the checkpoint's [N,K] tensor), the [N,K] CHECKPOINT tensor as
// `{prefix}.weight_nk` (buffer-shared with the params map — no extra copy;
// consumed by the decode QMV's simd_sum kernel which streams [N,K] row-major),
// and the f32 [N] per-output-channel scale as `{prefix}.scales`. There is no
// `.biases` (sym8 is symmetric by construction).
//
// Dispatch mirrors the eager Rust `QuantizedLinear::forward_sym8` EXACTLY
// (same shared `na_int8::*` builders, same M<=2/M>=3 boundary), so compiled
// and eager sym8 forwards are byte-identical for the same inputs. All
// contract violations THROW (fail-loud — a silent fallback to
// quantized_matmul would read the int8 operand as an MXFP8/affine pack and
// emit garbage logits).
inline array sym8_linear_proj(const array& x, const array& w_kn,
                              const array& scales,
                              const std::optional<array>& biases,
                              const std::string& prefix) {
  if (biases.has_value()) {
    throw std::runtime_error(
        "sym8 projection '" + prefix +
        "': unexpected .biases sidecar (sym8 is symmetric — convert never "
        "emits one)");
  }
  if (w_kn.dtype() != mlx::core::int8 || w_kn.ndim() != 2) {
    throw std::runtime_error(
        "sym8 projection '" + prefix +
        "': registered weight is not a 2-D int8 [K,N] kernel operand");
  }
  if (scales.dtype() != mlx::core::float32 || scales.ndim() != 1 ||
      scales.shape(0) != w_kn.shape(1)) {
    throw std::runtime_error(
        "sym8 projection '" + prefix +
        "': expected f32 [N] .scales matching the [K,N] operand");
  }
  const int K = x.shape(-1);
  if (w_kn.shape(0) != K) {
    throw std::runtime_error(
        "sym8 projection '" + prefix + "': x.K=" + std::to_string(K) +
        " != w.K=" + std::to_string(w_kn.shape(0)) +
        " — was the [N,K] checkpoint tensor registered instead of the [K,N] "
        "kernel operand?");
  }
  // Flatten leading dims to M (mirrors the Rust forward's [.., K] -> [M, K]).
  const int64_t m64 = x.size() / static_cast<int64_t>(K);
  const int M = static_cast<int>(m64);
  array x2d = (x.ndim() == 2) ? x : reshape(x, {M, K});
  // DECODE (M<=2) is the W8A16 matvec (bf16 activations read directly — no
  // act quant, activation-exact); PREFILL (M>=3) stays the W8A8 GEMM.
  // INT8_QMV_W8A16=0 (read inside qmv_w8a16_lazy, the shared builder) reroutes
  // decode back to the old W8A8 qmv for same-binary A/B.
  // The decode matvec needs the [N,K] CHECKPOINT orientation alongside the
  // [K,N] operand (its simd_sum kernel streams [N,K] row-major). Registration
  // stores it under `{prefix}.weight_nk`; absence is a registration bug —
  // fail loud rather than silently transposing per call.
  array y2d = [&]() {
    if (M <= 2) {
      const std::string nk_key = prefix + ".weight_nk";
      if (!has_weight(nk_key)) {
        throw std::runtime_error(
            "sym8 projection '" + prefix +
            "': missing registered [N,K] checkpoint tensor '" + nk_key +
            "' (register_weights_with_cpp must store it for the decode QMV)");
      }
      return na_int8::qmv_w8a16_lazy(x2d, w_kn, get_weight(nk_key), scales);
    }
    return na_int8::w8a8_linear_lazy(x2d, w_kn, scales);
  }();
  if (x.ndim() == 2) return y2d;
  auto out_shape = x.shape();
  out_shape.back() = w_kn.shape(1);
  return reshape(y2d, out_shape);
}

inline array linear_proj(const array& x, const std::string& prefix) {
  std::string scales_key = prefix + ".scales";
  bool scales_present = has_weight(scales_key);
  if (scales_present) {
    auto w = get_weight(prefix + ".weight");
    auto scales = get_weight(scales_key);
    std::string biases_key = prefix + ".biases";
    std::optional<array> biases = std::nullopt;
    if (has_weight(biases_key)) {
      biases = get_weight(biases_key);
    }

    // Registry-first dispatch.
    if (auto info = lookup_quant_info(prefix)) {
      // One-shot dispatch trace per NEW prefix. Gated because this fires
      // once per projection and otherwise floods benchmark worker stderr.
      if (mtp_trace_enabled()) {
        std::lock_guard<std::mutex> lk(g_linear_proj_logged_mutex());
        if (g_linear_proj_logged_prefixes().insert(prefix).second) {
          fprintf(stderr,
                  "[MLX] linear_proj dispatch: prefix='%s' scales_present=1 "
                  "registered_quant_mode='%s' bits=%d group_size=%d "
                  "fallback_used=0\n",
                  prefix.c_str(), info->mode.c_str(), info->bits,
                  info->group_size);
        }
      }
      // sym8 has no quantized_matmul pack — route to the int8 W8A8 kernels
      // (same shared builders as the eager Rust path; see sym8_linear_proj).
      if (info->mode == "sym8") {
        return sym8_linear_proj(x, w, scales, biases, prefix);
      }
      return mlx::core::quantized_matmul(
          x, w, scales, biases, /*transpose=*/true,
          info->group_size, info->bits, info->mode);
    }

    // Fallback heuristic: only the registered modes affine/mxfp8/mxfp4/nvfp4
    // are correct callers; MXFP4/NVFP4 paths under the heuristic would silently
    // mislabel as MXFP8 because none of them carry biases. The fallback exists
    // for the dense qwen3_5 path (which does not register quant info) —
    // dense models in production are MXFP8 or affine, so the heuristic remains
    // correct there.
    static std::atomic<bool> warned_fallback{false};
    if (!warned_fallback.exchange(true)) {
      fprintf(stderr,
              "[mlx_qwen35] linear_proj: registry miss for prefix '%s' "
              "(first occurrence; suppressing further warnings). Falling "
              "back to companion-tensor heuristic.\n",
              prefix.c_str());
    }
    // One-shot dispatch trace per NEW prefix (fallback branch).
    if (mtp_trace_enabled()) {
      std::lock_guard<std::mutex> lk(g_linear_proj_logged_mutex());
      if (g_linear_proj_logged_prefixes().insert(prefix).second) {
        const char* mode = biases.has_value() ? "affine(infer)" : "mxfp8(heuristic)";
        fprintf(stderr,
                "[MLX] linear_proj dispatch: prefix='%s' scales_present=1 "
                "registered_quant_mode=NONE fallback_used=1 fallback_mode='%s'\n",
                prefix.c_str(), mode);
      }
    }
    if (!biases.has_value()) {
      return mlx::core::quantized_matmul(x, w, scales, biases, true, 32, 8, "mxfp8");
    } else {
      int bits = infer_affine_bits(w, scales, 64);
      return mlx::core::quantized_matmul(x, w, scales, biases, true, 64, bits, "affine");
    }
  }
  // One-shot dispatch trace per NEW prefix (plain matmul branch).
  if (mtp_trace_enabled()) {
    std::lock_guard<std::mutex> lk(g_linear_proj_logged_mutex());
    if (g_linear_proj_logged_prefixes().insert(prefix).second) {
      fprintf(stderr,
              "[MLX] linear_proj dispatch: prefix='%s' scales_present=0 "
              "registered_quant_mode=NONE fallback_used=0 "
              "fallback_mode='bf16_matmul'\n",
              prefix.c_str());
    }
  }
  return matmul(x, get_weight_t(prefix + ".weight"));
}

// Detect quantization for a layer projection (q_proj, gate_proj, switch_mlp.*,
// shared_expert.*, dense MLP gate_proj, etc).
//
// Returns (is_quantized, group_size, bits, mode). When the prefix has no
// `.scales` companion, returns (false, 0, 0, "").
//
// Dispatch order matches `linear_proj`:
//   1. Registry hit -> use Rust-authoritative (mode, bits, group_size).
//   2. Registry miss -> heuristic: no-biases -> MXFP8 (gs=32, bits=8);
//      has-biases -> affine (gs=64, bits inferred from weight/scales ratio).
inline std::tuple<bool, int, int, std::string>
detect_layer_quant(const std::string& prefix) {
  if (!has_weight(prefix + ".scales")) {
    return {false, 0, 0, ""};
  }
  if (auto info = lookup_quant_info(prefix)) {
    // Every caller feeds this tuple into quantized_matmul/gather_qmm-style
    // ops, which have NO sym8 pack — a sym8 entry reaching here means a
    // registration bug (the MoE/gemma4/lfm2 loaders skip C++ registration
    // for sym8 checkpoints entirely). Fail loud instead of mis-packing.
    if (info->mode == "sym8") {
      throw std::runtime_error(
          "detect_layer_quant: prefix '" + prefix +
          "' is registered as sym8, which this caller cannot dispatch");
    }
    return {true, info->group_size, info->bits, info->mode};
  }
  bool has_biases = has_weight(prefix + ".biases");
  if (!has_biases) {
    return {true, 32, 8, "mxfp8"};
  }
  auto w = get_weight(prefix + ".weight");
  auto scales = get_weight(prefix + ".scales");
  int bits = infer_affine_bits(w, scales, 64);
  return {true, 64, bits, "affine"};
}

// Detect quantization for a router/shared-expert gate. Same return shape as
// `detect_layer_quant`. With the registry, Rust drives the actual values
// (MXFP8 gates under `--q-mxfp --q-bits 8` no-recipe path, 8-bit affine in
// all other cases). Heuristic fallback hardcodes 8-bit affine gs=64 for
// callers that have not been plumbed.
inline std::tuple<bool, int, int, std::string>
detect_router_gate_quant(const std::string& prefix) {
  if (!has_weight(prefix + ".scales")) {
    return {false, 0, 0, ""};
  }
  if (auto info = lookup_quant_info(prefix)) {
    // Same rationale as detect_layer_quant: no caller can dispatch sym8.
    if (info->mode == "sym8") {
      throw std::runtime_error(
          "detect_router_gate_quant: prefix '" + prefix +
          "' is registered as sym8, which this caller cannot dispatch");
    }
    return {true, info->group_size, info->bits, info->mode};
  }
  return {true, 64, 8, "affine"};
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

// Wrap a weight-baking compiled-graph function so MLX assigns it a UNIQUE,
// erasable compile-cache identity. The graph reads model weights via
// get_weight() at trace time, so the weights are baked into the cached tape.
// MLX keys its compile cache on a fun_id: a free function (or captureless
// lambda) yields a STABLE code-address fun_id with NO eviction hook, so
// assigning {} to the std::function does NOT erase the tape — a reloaded model
// silently replays the previous model's baked weights. Capturing `fn` makes the
// closure non-convertible to a function pointer, routing through
// compile(std::function) -> get_function_address()==0 -> a heap shared_ptr
// fun_id whose deleter calls compile_erase(): assigning {} frees the tape and
// the next compile re-traces against the live weights. See compile.cpp:1205-1241.
inline std::function<std::vector<array>(const std::vector<array>&)>
compile_resettable_weight_graph(
    std::vector<array> (*fn)(const std::vector<array>&)) {
  return mlx::core::compile(
      [fn](const std::vector<array>& inputs) { return fn(inputs); });
}

inline array rms_norm_no_weight(const array& x, float eps) {
  return fast::rms_norm(x, std::nullopt, eps);
}

// SwiGLU: sigmoid(gate) * gate * up — compiled for kernel fusion
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
// M-RoPE (Multimodal Rotary Position Embeddings) for 3D position IDs
// =====================================================================

// Applies M-RoPE to queries and keys using 3-channel position IDs.
// Each channel (temporal, height, width) controls a section of rotary dims.
// Non-rotary dimensions (rope_dims..head_dim) are passed through unchanged.
inline std::pair<array, array> apply_mrope(
    const array& queries,      // [B, T, H, D]
    const array& keys,         // [B, T, Hkv, D]
    const array& position_ids, // [3, B, T]
    int rope_dims,             // total rotary dims (e.g. 64)
    float rope_theta,
    const std::vector<int>& mrope_section) {  // e.g. [11, 11, 10], sum=32 half-dims

  auto dt = queries.dtype();
  int B = queries.shape(0);
  int T = queries.shape(1);
  int half_rotary = rope_dims / 2;  // = sum of mrope_section

  // Extract 3 position channels: each [B, T]
  auto pos_t = reshape(slice(position_ids, {0, 0, 0}, {1, B, T}), {B, T});
  auto pos_h = reshape(slice(position_ids, {1, 0, 0}, {2, B, T}), {B, T});
  auto pos_w = reshape(slice(position_ids, {2, 0, 0}, {3, B, T}), {B, T});
  std::vector<array> pos_channels = {pos_t, pos_h, pos_w};

  // Build concatenated frequency tensor [B, T, half_rotary]
  std::vector<array> freq_sections;
  int dim_offset = 0;
  for (int s = 0; s < 3; s++) {
    int section_dims = mrope_section[s];  // number of half-dim pairs in this section
    // inv_freq: theta^(-2*(dim_offset+i)/rope_dims) for i in [0..section_dims)
    // Shape: [section_dims]
    std::vector<float> inv_freq_data(section_dims);
    for (int i = 0; i < section_dims; i++) {
      float exponent = -2.0f * (float)(dim_offset + i) / (float)rope_dims;
      inv_freq_data[i] = std::pow(rope_theta, exponent);
    }
    auto inv_freq = array(inv_freq_data.data(), {1, 1, section_dims}, mlx::core::float32);

    // positions: [B, T, 1] * inv_freq: [1, 1, section_dims] → freqs: [B, T, section_dims]
    auto pos = astype(reshape(pos_channels[s], {B, T, 1}), mlx::core::float32);
    auto freqs = pos * inv_freq;
    freq_sections.push_back(freqs);
    dim_offset += section_dims;
  }

  // Concatenate: [B, T, half_rotary]
  auto freqs = concatenate(freq_sections, 2);

  // cos/sin: [B, T, 1, half_rotary] for broadcasting over heads
  auto cos_f = astype(reshape(cos(freqs), {B, T, 1, half_rotary}), dt);
  auto sin_f = astype(reshape(sin(freqs), {B, T, 1, half_rotary}), dt);

  // Helper: apply rotary embedding to the first rope_dims dims of x [B, T, H, D]
  auto apply_rot = [&](const array& x) -> array {
    int H = x.shape(2);
    int D = x.shape(3);
    // Rotary portion: [B, T, H, rope_dims]
    auto x_rot = slice(x, {0, 0, 0, 0}, {B, T, H, rope_dims});
    // Split into even/odd halves: [B, T, H, half_rotary]
    auto x_even = slice(x_rot, {0, 0, 0, 0},          {B, T, H, rope_dims}, {1, 1, 1, 2});
    auto x_odd  = slice(x_rot, {0, 0, 0, 1},          {B, T, H, rope_dims}, {1, 1, 1, 2});
    // Rotation: [even*cos - odd*sin, even*sin + odd*cos]
    auto r_even = x_even * cos_f - x_odd * sin_f;
    auto r_odd  = x_even * sin_f + x_odd * cos_f;
    // Interleave back: stack [B, T, H, half_rotary, 2] → reshape [B, T, H, rope_dims]
    auto stacked = stack({r_even, r_odd}, 4);  // [B, T, H, half_rotary, 2]
    auto rotated = reshape(stacked, {B, T, H, rope_dims});
    if (rope_dims < D) {
      // Pass-through portion
      auto x_pass = slice(x, {0, 0, 0, rope_dims}, {B, T, H, D});
      return concatenate({rotated, x_pass}, 3);
    }
    return rotated;
  };

  return {apply_rot(queries), apply_rot(keys)};
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
// Tape-emitting GDN kernel for MTP rollback. Like the standard
// gated-delta kernel but emits the per-step innovation `delta` as a
// third output (`tape`, fp32, shape `[B, T, Hv, Dv]`). The MTP verify
// path records `(tape, k, g, qkv)` per layer during the D+1 verify
// forward; on rollback the tape-replay kernel applies the first
// `accepted_steps` innovations to the pre-verify snapshot state without
// re-running the main model forward.
//
// Kernel source: `metal/gated_delta_step_tape.metal.inc`.
// =====================================================================

inline std::unique_ptr<mlx::core::fast::CustomKernelFunction>& qwen35_gd_tape_kernel() {
  static std::unique_ptr<mlx::core::fast::CustomKernelFunction> instance;
  return instance;
}

inline void ensure_gated_delta_tape_kernel() {
  std::lock_guard<std::mutex> lock(qwen35_kernel_mutex());
  if (qwen35_gd_tape_kernel()) return;

  static const char* source =
    #include "metal/gated_delta_step_tape.metal.inc"
  ;

  auto kernel = fast::metal_kernel(
      "gated_delta_step_tape",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out", "innovation_tape"},
      source
  );
  qwen35_gd_tape_kernel() = std::make_unique<mlx::core::fast::CustomKernelFunction>(std::move(kernel));
}

// Returns (y, state_out, tape). `tape` is fp32, shape `[B, T, Hv, Dv]`.
inline std::tuple<array, array, array> gated_delta_kernel_with_tape_call(
    const array& q, const array& k, const array& v,
    const array& g, const array& beta_arr,
    const array& state) {
  ensure_gated_delta_tape_kernel();

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = q.shape(2);
  int Dk = q.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);
  auto input_type = q.dtype();
  auto T_arr = array(T, mlx::core::int32);

  auto results = (*qwen35_gd_tape_kernel())(
      {q, k, v, g, beta_arr, state, T_arr},
      {Shape{B, T, Hv, Dv}, state.shape(), Shape{B, T, Hv, Dv}},
      {input_type, input_type, mlx::core::float32},
      std::make_tuple(32, Dv, B * Hv),
      std::make_tuple(32, 4, 1),
      {{"InT", input_type}, {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
      std::nullopt, false,
      mlx::core::default_stream(mlx::core::Device::gpu)
  );

  return std::tuple<array, array, array>(
      std::move(results[0]), std::move(results[1]), std::move(results[2]));
}

// =====================================================================
// Tape-replay kernel. Takes a snapshot `state`, plus per-step
// `(tape, k, g)` recorded by the tape-emitting forward kernel, and
// returns a new state advanced by `T` steps. The recurrent_state output
// matches what `T` calls to `gated_delta_kernel_call` with the same
// per-step inputs would have produced (modulo the per-step InT
// round-trip the forward path takes between FFI calls in our verify
// loop — see `metal/tape_replay.metal.inc` header for details).
//
// `tape.shape == [B, T, Hv, Dv]` (fp32)
// `k.shape    == [B, T, Hk, Dk]` (model dtype)
// `g.shape    == [B, T, Hv]` (fp32)
// `state.shape == [B, Hv, Dv, Dk]` (model dtype)
// =====================================================================

inline std::unique_ptr<mlx::core::fast::CustomKernelFunction>& qwen35_tape_replay_kernel() {
  static std::unique_ptr<mlx::core::fast::CustomKernelFunction> instance;
  return instance;
}

inline void ensure_tape_replay_kernel() {
  std::lock_guard<std::mutex> lock(qwen35_kernel_mutex());
  if (qwen35_tape_replay_kernel()) return;

  static const char* source =
    #include "metal/tape_replay.metal.inc"
  ;

  auto kernel = fast::metal_kernel(
      "tape_replay",
      {"tape", "k", "g", "state_in", "T"},
      {"state_out"},
      source
  );
  qwen35_tape_replay_kernel() = std::make_unique<mlx::core::fast::CustomKernelFunction>(std::move(kernel));
}

inline array tape_replay_kernel_call(
    const array& tape, const array& k, const array& g,
    const array& state) {
  ensure_tape_replay_kernel();

  int B = k.shape(0);
  int T = k.shape(1);
  int Hk = k.shape(2);
  int Dk = k.shape(3);
  int Hv = tape.shape(2);
  int Dv = tape.shape(3);
  auto input_type = state.dtype();
  auto T_arr = array(T, mlx::core::int32);

  auto results = (*qwen35_tape_replay_kernel())(
      {tape, k, g, state, T_arr},
      {state.shape()},
      {input_type},
      std::make_tuple(32, Dv, B * Hv),
      std::make_tuple(32, 4, 1),
      {{"InT", input_type}, {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
      std::nullopt, false,
      mlx::core::default_stream(mlx::core::Device::gpu)
  );

  return std::move(results[0]);
}

// =====================================================================
// Pure GDN forward (shared between dense compiled and MoE non-compiled)
// =====================================================================

struct GDNPureResult { array output, conv_state, recurrent_state; };

// Extended GDN result that also returns the per-step `(tape, k, g,
// qkv)` tensors needed by the rollback tape-replay path. `tape` is fp32
// `[B, 1, Hv, Dv]`, `k` is `[B, 1, Hk, Dk]` (model dtype), `g` is
// `[B, 1, Hv]` (fp32), `qkv` is `[B, 1, conv_dim]` (model dtype — the
// pre-conv qkv projection before the depthwise-conv state advance).
struct GDNPureResultWithTape {
  array output, conv_state, recurrent_state;
  array tape, k_tape, g_tape, qkv_tape;
};

// GDN forward templated on `WithTape`. When `WithTape=false` returns
// `GDNPureResult`; when `WithTape=true` returns the extended
// `GDNPureResultWithTape` that also carries the per-step `(tape, k, g,
// qkv)` tensors consumed by the rollback tape-replay path. Both
// instantiations are math-identical — only the underlying Metal kernel
// differs (`gated_delta_kernel_call` vs `gated_delta_kernel_with_tape_call`).
template <bool WithTape>
inline std::conditional_t<WithTape, GDNPureResultWithTape, GDNPureResult>
gdn_pure_fn_impl(
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

  // Reshape qkv to 3D for conv1d — when WithTape this is also the tensor
  // we record for conv-state replay.
  auto qkv_3d = reshape(qkv, {B, 1, conv_dim});

  auto conv_input = concatenate({conv_state, qkv_3d}, 1);

  int total_len = cfg.linear_conv_kernel_dim;
  int keep = cfg.linear_conv_kernel_dim - 1;
  auto new_conv_state = slice(conv_input, {0, total_len - keep, 0}, {B, total_len, conv_dim});

  auto conv_w = get_weight(pfx + "conv1d.weight");
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
  auto a_log  = get_weight(pfx + "A_log");
  auto dt_b   = get_weight(pfx + "dt_bias");
  auto g_3d = reshape(fused_compute_g(a_log, a, dt_b), {B, 1, cfg.linear_num_v_heads});

  // Metal kernel — non-tape returns (y, recurrent_state); tape variant
  // additionally returns the per-step delta `tape` [B, 1, Hv, Dv] fp32.
  // `array` has no default ctor, so we emit the kernel call inside a
  // lambda whose return type encodes both variants. Downstream graph
  // (rmsnorm + out_proj) is shared.
  auto kernel_out = [&] {
    if constexpr (WithTape) {
      return gated_delta_kernel_with_tape_call(q, k, v, g_3d, beta_3d, recurrent_state);
    } else {
      return gated_delta_kernel_call(q, k, v, g_3d, beta_3d, recurrent_state);
    }
  }();
  // `kernel_out` is `tuple` (tape) or `pair` (non-tape); `std::get<0/1>`
  // works for both.
  auto& y                   = std::get<0>(kernel_out);
  auto& new_recurrent_state = std::get<1>(kernel_out);

  // RMSNorm gating
  auto z_h = reshape(z, {B, 1, cfg.linear_num_v_heads, cfg.linear_value_head_dim});
  auto nw = get_weight(pfx + "norm.weight");
  auto y_normed = fast::rms_norm(y, nw, cfg.rms_norm_eps);
  y_normed = compiled_silu_mul()({z_h, y_normed})[0];

  // Output projection
  auto y_flat = reshape(y_normed, {B, value_dim});
  auto output = linear_proj(y_flat, pfx + "out_proj");

  if constexpr (WithTape) {
    // k after rmsnorm+scale, g fp32 — captured before the kernel call.
    return GDNPureResultWithTape{
        std::move(output), std::move(new_conv_state), std::move(new_recurrent_state),
        std::move(std::get<2>(kernel_out)), std::move(k), std::move(g_3d), std::move(qkv_3d)};
  } else {
    return GDNPureResult{std::move(output), std::move(new_conv_state),
                         std::move(new_recurrent_state)};
  }
}

// Thin wrappers over the templated `gdn_pure_fn_impl`.
inline GDNPureResult gdn_pure_fn(
    const array& x,
    int layer_idx,
    const array& conv_state,
    const array& recurrent_state,
    const BaseConfig& cfg) {
  return gdn_pure_fn_impl<false>(x, layer_idx, conv_state, recurrent_state, cfg);
}

// Tape-recording variant of `gdn_pure_fn`. Identical math; the
// underlying Metal kernel call also returns the per-step `tape`
// innovation, and we surface `(k, g, qkv)` for tape-replay rollback.
// Used by the dense and MoE main forwards during MTP verify cycles when
// tape-replay is enabled.
inline GDNPureResultWithTape gdn_pure_fn_with_tape(
    const array& x,
    int layer_idx,
    const array& conv_state,
    const array& recurrent_state,
    const BaseConfig& cfg) {
  return gdn_pure_fn_impl<true>(x, layer_idx, conv_state, recurrent_state, cfg);
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
    bool dynamic_kv = false) {  // true = slice KV to valid range, skip mask (T=1 decode)
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
      // Slice KV cache to valid range [0..offset+1], pass no mask
      // → faster SDPA kernel. Mirrors upstream mlx-lm's
      // `create_attention_mask(N=1) → None` + `cache.state` slicing
      // pattern (see ./mlx-lm/mlx_lm/models/base.py:51-52 and
      // ./mlx-lm/mlx_lm/models/cache.py:362-368). Only safe when the
      // caller is NOT inside `mlx::core::compile` — the slice length
      // depends on the C++ `offset` int which changes per decode step,
      // so a compiled graph would mis-cache. Currently used by the
      // dense flat decode path (`qwen35_decode_fn` in mlx_qwen35.cpp).
      int valid_len = offset + 1;
      auto valid_keys   = slice(new_kv_keys,   {0, 0, 0, 0}, {B, cfg.num_kv_heads, valid_len, cfg.head_dim});
      auto valid_values = slice(new_kv_values, {0, 0, 0, 0}, {B, cfg.num_kv_heads, valid_len, cfg.head_dim});
      return fast::scaled_dot_product_attention(
          queries, valid_keys, valid_values, scale, "", std::nullopt, {});
    } else {
      // Compiled / verify paths: fixed shapes + additive mask (required for mlx::core::compile)
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

// =====================================================================
// MTP helper: attention forward with array-valued RoPE offset and
// parameterised weight prefix.
//
// Mirrors `attn_pure_fn` but:
//   - Reads weights from `<prefix>.self_attn.*` (NOT hard-coded
//     `layers.{i}.self_attn.*`), so the MTP compiled draft graph
//     (`mtp.layers.{j}.self_attn.*`) and the per-depth verify graph
//     (`layers.{i}.self_attn.*`) can both share this helper without
//     duplicating the attention body.
//   - Takes the cache write offset as a `[1] int32` array so the closure
//     can be cached by `mlx::core::compile(...)`. The flat-path
//     `attn_pure_fn` captures a C++ `int` which invalidates the compile
//     cache every step; this variant threads the offset through the
//     graph instead.
//
// Behavior, shapes and weight-key conventions otherwise match
// `attn_pure_fn(dynamic_kv=false)` exactly. The function intentionally
// does NOT support `dynamic_kv` — the verify / MTP draft graphs always
// rely on the additive `attn_mask` (compile requires fixed shapes).
// =====================================================================
inline AttnPureResult attn_pure_fn_arr_rope_write_offset(
    const array& x,             // [B, hidden] — 2D
    const std::string& layer_prefix,  // e.g. "layers.3" or "mtp.layers.0"
    const array& kv_keys,       // [B, Hkv, max_kv_len, D]
    const array& kv_values,     // [B, Hkv, max_kv_len, D]
    const array& attn_mask,     // [1, 1, 1, max_kv_len] additive bf16 mask
    const array& rope_offset_arr,   // [1] int32 — RoPE absolute position
    const array& write_offset_arr,  // [1] int32 — local slice_update slot
    const BaseConfig& cfg) {
  int B = x.shape(0);
  std::string pfx = layer_prefix + ".self_attn.";

  auto q_proj = linear_proj(x, pfx + "q_proj");
  if (has_weight(pfx + "q_proj.bias")) q_proj = q_proj + get_weight(pfx + "q_proj.bias");

  auto qph    = reshape(q_proj, {B, 1, cfg.num_heads, cfg.head_dim * 2});
  auto queries = slice(qph, {0, 0, 0, 0},            {B, 1, cfg.num_heads, cfg.head_dim});
  auto gate    = slice(qph, {0, 0, 0, cfg.head_dim}, {B, 1, cfg.num_heads, cfg.head_dim * 2});
  gate = reshape(gate, {B, cfg.num_heads * cfg.head_dim});

  auto keys   = linear_proj(x, pfx + "k_proj");
  auto values = linear_proj(x, pfx + "v_proj");
  if (has_weight(pfx + "k_proj.bias")) keys   = keys   + get_weight(pfx + "k_proj.bias");
  if (has_weight(pfx + "v_proj.bias")) values = values + get_weight(pfx + "v_proj.bias");

  keys   = reshape(keys,   {B, 1, cfg.num_kv_heads, cfg.head_dim});
  values = reshape(values, {B, 1, cfg.num_kv_heads, cfg.head_dim});

  queries = fast::rms_norm(queries, get_weight(pfx + "q_norm.weight"), cfg.rms_norm_eps);
  keys    = fast::rms_norm(keys,    get_weight(pfx + "k_norm.weight"), cfg.rms_norm_eps);

  // Array-valued RoPE offset — graph-stable across decode steps.
  queries = fast::rope(queries, cfg.rope_dims, false, cfg.rope_theta, 1.0f, rope_offset_arr);
  keys    = fast::rope(keys,    cfg.rope_dims, false, cfg.rope_theta, 1.0f, rope_offset_arr);

  queries = transpose(queries, {0, 2, 1, 3});
  keys    = transpose(keys,    {0, 2, 1, 3});
  values  = transpose(values,  {0, 2, 1, 3});

  // slice_update with array start so the verify graph can advance the
  // KV-cache write offset by 0,1,...,depth without re-tracing.
  auto new_kv_keys   = mlx::core::slice_update(kv_keys,   keys,   write_offset_arr, {2});
  auto new_kv_values = mlx::core::slice_update(kv_values, values, write_offset_arr, {2});

  float scale = std::pow((float)cfg.head_dim, -0.5f);
  auto attn_out = fast::scaled_dot_product_attention(
      queries, new_kv_keys, new_kv_values, scale, "", attn_mask, {});

  attn_out = transpose(attn_out, {0, 2, 1, 3});
  attn_out = reshape(attn_out, {B, cfg.num_heads * cfg.head_dim});

  attn_out = compiled_attn_gate()({attn_out, gate})[0];

  auto output = linear_proj(attn_out, pfx + "o_proj");
  if (has_weight(pfx + "o_proj.bias")) output = output + get_weight(pfx + "o_proj.bias");

  return {output, new_kv_keys, new_kv_values};
}

inline AttnPureResult attn_pure_fn_arr_offset(
    const array& x,             // [B, hidden] — 2D
    const std::string& layer_prefix,
    const array& kv_keys,
    const array& kv_values,
    const array& attn_mask,
    const array& offset_arr,    // [1] int32 — RoPE + slice_update start
    const BaseConfig& cfg) {
  return attn_pure_fn_arr_rope_write_offset(
      x, layer_prefix, kv_keys, kv_values, attn_mask, offset_arr, offset_arr, cfg);
}

// =====================================================================
// Attention prefill (3D input, M-RoPE, causal mask)
// =====================================================================

inline AttnPureResult attn_prefill_fn(
    const array& x,              // [B, T, hidden] — 3D
    int layer_idx,
    const array& position_ids,   // [3, B, T]
    const BaseConfig& cfg,
    const std::vector<int>& mrope_section) {
  int B = x.shape(0);
  int T = x.shape(1);
  int hidden = x.shape(2);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  // Flatten to [B*T, hidden] for projections
  auto x_flat = reshape(x, {B * T, hidden});

  // Q projection (2x width for per-head gating)
  auto q_proj = linear_proj(x_flat, pfx + "q_proj");
  if (has_weight(pfx + "q_proj.bias")) q_proj = q_proj + get_weight(pfx + "q_proj.bias");

  auto qph    = reshape(q_proj, {B, T, cfg.num_heads, cfg.head_dim * 2});
  auto queries = slice(qph, {0, 0, 0, 0},             {B, T, cfg.num_heads, cfg.head_dim});
  auto gate    = slice(qph, {0, 0, 0, cfg.head_dim},  {B, T, cfg.num_heads, cfg.head_dim * 2});
  gate = reshape(gate, {B, T, cfg.num_heads * cfg.head_dim});

  // K, V projections
  auto keys   = linear_proj(x_flat, pfx + "k_proj");
  auto values = linear_proj(x_flat, pfx + "v_proj");
  if (has_weight(pfx + "k_proj.bias")) keys   = keys   + get_weight(pfx + "k_proj.bias");
  if (has_weight(pfx + "v_proj.bias")) values = values + get_weight(pfx + "v_proj.bias");

  keys   = reshape(keys,   {B, T, cfg.num_kv_heads, cfg.head_dim});
  values = reshape(values, {B, T, cfg.num_kv_heads, cfg.head_dim});

  // QK norm
  queries = fast::rms_norm(queries, get_weight(pfx + "q_norm.weight"), cfg.rms_norm_eps);
  keys    = fast::rms_norm(keys,    get_weight(pfx + "k_norm.weight"), cfg.rms_norm_eps);

  // M-RoPE (replaces scalar-offset fast::rope)
  auto [q_rope, k_rope] = apply_mrope(queries, keys, position_ids, cfg.rope_dims, cfg.rope_theta, mrope_section);

  // Transpose to [B, H, T, D] for SDPA
  auto q_t = transpose(q_rope, {0, 2, 1, 3});
  auto k_t = transpose(k_rope, {0, 2, 1, 3});
  auto v_t = transpose(values, {0, 2, 1, 3});

  // Causal mask: [1, 1, T, T] — upper-triangular -inf
  auto dt = x.dtype();
  auto mask_col = reshape(arange(0, T, mlx::core::int32), {1, 1, 1, T});
  auto mask_row = reshape(arange(0, T, mlx::core::int32), {1, 1, T, 1});
  auto causal_mask = where(
      greater(mask_col, mask_row),
      array(-std::numeric_limits<float>::infinity(), dt),
      array(0.0f, dt));

  // SDPA
  float scale = std::pow((float)cfg.head_dim, -0.5f);
  auto attn_out = fast::scaled_dot_product_attention(
      q_t, k_t, v_t, scale, "", causal_mask, {});

  // Transpose back to [B, T, H, D] → [B, T, H*D]
  attn_out = transpose(attn_out, {0, 2, 1, 3});
  attn_out = reshape(attn_out, {B, T, cfg.num_heads * cfg.head_dim});

  // Gate
  attn_out = compiled_attn_gate()({attn_out, gate})[0];

  // Output projection: flatten to [B*T, H*D], project, reshape back
  auto out_flat = linear_proj(reshape(attn_out, {B * T, cfg.num_heads * cfg.head_dim}), pfx + "o_proj");
  if (has_weight(pfx + "o_proj.bias")) out_flat = out_flat + get_weight(pfx + "o_proj.bias");
  auto output = reshape(out_flat, {B, T, hidden});

  // Return keys/values in [B, Hkv, T, D] format (for cache initialization)
  return {output, k_t, v_t};
}

// =====================================================================
// Attention prefill (3D input, scalar RoPE offset, causal mask).
// Text-only variant. Differs from attn_prefill_fn only in the RoPE
// step — scalar-offset fast::rope instead of M-RoPE. The caller
// (mlx_qwen35_text_prefill) uses single-chunk prefill so offset=0
// and SDPA can run in "causal" string mode (no explicit mask array).
// Returns keys/values in [B, Hkv, T, D] for cache initialization.
// =====================================================================

inline AttnPureResult attn_prefill_fn_scalar(
    const array& x,    // [B, T, hidden] — 3D
    int layer_idx,
    int offset,        // RoPE start position (0 for first chunk)
    const BaseConfig& cfg) {
  int B = x.shape(0);
  int T = x.shape(1);
  int hidden = x.shape(2);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  auto x_flat = reshape(x, {B * T, hidden});

  // Q projection (2x width for per-head gating)
  auto q_proj = linear_proj(x_flat, pfx + "q_proj");
  if (has_weight(pfx + "q_proj.bias")) q_proj = q_proj + get_weight(pfx + "q_proj.bias");

  auto qph    = reshape(q_proj, {B, T, cfg.num_heads, cfg.head_dim * 2});
  auto queries = slice(qph, {0, 0, 0, 0},            {B, T, cfg.num_heads, cfg.head_dim});
  auto gate    = slice(qph, {0, 0, 0, cfg.head_dim}, {B, T, cfg.num_heads, cfg.head_dim * 2});
  gate = reshape(gate, {B, T, cfg.num_heads * cfg.head_dim});

  // K, V projections
  auto keys   = linear_proj(x_flat, pfx + "k_proj");
  auto values = linear_proj(x_flat, pfx + "v_proj");
  if (has_weight(pfx + "k_proj.bias")) keys   = keys   + get_weight(pfx + "k_proj.bias");
  if (has_weight(pfx + "v_proj.bias")) values = values + get_weight(pfx + "v_proj.bias");

  keys   = reshape(keys,   {B, T, cfg.num_kv_heads, cfg.head_dim});
  values = reshape(values, {B, T, cfg.num_kv_heads, cfg.head_dim});

  // QK norm
  queries = fast::rms_norm(queries, get_weight(pfx + "q_norm.weight"), cfg.rms_norm_eps);
  keys    = fast::rms_norm(keys,    get_weight(pfx + "k_norm.weight"), cfg.rms_norm_eps);

  // Scalar-offset partial RoPE (text path; matches Qwen3_5Attention::forward()).
  queries = fast::rope(queries, cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset);
  keys    = fast::rope(keys,    cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset);

  // [B, T, H, D] → [B, H, T, D]
  auto q_t = transpose(queries, {0, 2, 1, 3});
  auto k_t = transpose(keys,    {0, 2, 1, 3});
  auto v_t = transpose(values,  {0, 2, 1, 3});

  // SDPA in "causal" string mode — uses MLX's fused causal kernel, no
  // explicit O(T^2) mask materialization. Matches the Rust prefill path's
  // scaled_dot_product_attention_causal() (see attention.rs:228).
  float scale = std::pow((float)cfg.head_dim, -0.5f);
  auto attn_out = fast::scaled_dot_product_attention(
      q_t, k_t, v_t, scale, "causal", std::nullopt, std::nullopt);

  attn_out = transpose(attn_out, {0, 2, 1, 3});
  attn_out = reshape(attn_out, {B, T, cfg.num_heads * cfg.head_dim});

  // Gate
  attn_out = compiled_attn_gate()({attn_out, gate})[0];

  // Output projection
  auto out_flat = linear_proj(reshape(attn_out, {B * T, cfg.num_heads * cfg.head_dim}), pfx + "o_proj");
  if (has_weight(pfx + "o_proj.bias")) out_flat = out_flat + get_weight(pfx + "o_proj.bias");
  auto output = reshape(out_flat, {B, T, hidden});

  return {output, k_t, v_t};
}

// =====================================================================
// GDN prefill (3D input, full sequence conv1d, initial recurrent state)
// =====================================================================

inline GDNPureResult gdn_prefill_fn(
    const array& x,  // [B, T, hidden] — 3D
    int layer_idx,
    const BaseConfig& cfg) {
  int B = x.shape(0);
  int T = x.shape(1);
  int hidden = x.shape(2);
  int key_dim = cfg.linear_num_k_heads * cfg.linear_key_head_dim;
  int value_dim = cfg.linear_num_v_heads * cfg.linear_value_head_dim;
  int conv_dim = key_dim * 2 + value_dim;

  std::string pfx = "layers." + std::to_string(layer_idx) + ".linear_attn.";

  // Flatten to [B*T, hidden] for projections
  auto x_flat = reshape(x, {B * T, hidden});

  // Projections — auto-detecting quantized vs dense weights
  struct QkvzResult { array qkv, z; };
  auto [qkv, z] = [&]() -> QkvzResult {
    if (has_weight(pfx + "in_proj_qkvz.weight")) {
      auto qkvz = linear_proj(x_flat, pfx + "in_proj_qkvz");
      return {slice(qkvz, {0, 0}, {B * T, conv_dim}),
              slice(qkvz, {0, conv_dim}, {B * T, key_dim * 2 + value_dim * 2})};
    }
    return {linear_proj(x_flat, pfx + "in_proj_qkv"),
            linear_proj(x_flat, pfx + "in_proj_z")};
  }();
  struct BaResult { array b, a; };
  auto [b, a] = [&]() -> BaResult {
    if (has_weight(pfx + "in_proj_ba.weight")) {
      auto ba = linear_proj(x_flat, pfx + "in_proj_ba");
      return {slice(ba, {0, 0}, {B * T, cfg.linear_num_v_heads}),
              slice(ba, {0, cfg.linear_num_v_heads}, {B * T, cfg.linear_num_v_heads * 2})};
    }
    return {linear_proj(x_flat, pfx + "in_proj_b"),
            linear_proj(x_flat, pfx + "in_proj_a")};
  }();

  // Reshape qkv to 3D for conv1d: [B, T, conv_dim]
  auto qkv_3d = reshape(qkv, {B, T, conv_dim});

  // Conv1d: prepend (kernel_dim-1) zeros instead of using conv_state
  int pad_len = cfg.linear_conv_kernel_dim - 1;
  auto zero_pad = zeros({B, pad_len, conv_dim}, qkv_3d.dtype());
  auto conv_input = concatenate({zero_pad, qkv_3d}, 1);  // [B, pad_len+T, conv_dim]

  // Save conv_state: last (kernel_dim-1) tokens of pre-conv qkv for decode continuation
  int total_len = pad_len + T;
  auto new_conv_state = slice(conv_input, {0, total_len - pad_len, 0}, {B, total_len, conv_dim});

  auto conv_w = get_weight(pfx + "conv1d.weight");
  auto conv_out = mlx::core::conv1d(conv_input, conv_w, 1, 0, 1, conv_dim);  // [B, T, conv_dim]

  // SiLU — compiled
  conv_out = compiled_silu()({conv_out})[0];

  // Split into q, k, v
  auto q = slice(conv_out, {0, 0, 0},              {B, T, key_dim});
  auto k = slice(conv_out, {0, 0, key_dim},        {B, T, key_dim * 2});
  auto v = slice(conv_out, {0, 0, key_dim * 2},    {B, T, conv_dim});

  // Reshape to heads: [B, T, H, D]
  q = reshape(q, {B, T, cfg.linear_num_k_heads, cfg.linear_key_head_dim});
  k = reshape(k, {B, T, cfg.linear_num_k_heads, cfg.linear_key_head_dim});
  v = reshape(v, {B, T, cfg.linear_num_v_heads, cfg.linear_value_head_dim});

  // RMS norm + scaling
  float inv_s = std::pow((float)cfg.linear_key_head_dim, -0.5f);
  auto q_dt = q.dtype();
  q = rms_norm_no_weight(q, 1e-6f) * array(inv_s * inv_s, q_dt);
  k = rms_norm_no_weight(k, 1e-6f) * array(inv_s, q_dt);

  // Beta, g — reshape b/a from [B*T, Hv] to [B, T, Hv]
  auto beta_3d = reshape(sigmoid(b), {B, T, cfg.linear_num_v_heads});
  auto a_log = get_weight(pfx + "A_log");
  auto dt_b  = get_weight(pfx + "dt_bias");
  // fused_compute_g works on flat tensors [B*T, Hv]
  auto g_flat = fused_compute_g(a_log, a, dt_b);
  auto g_3d = reshape(g_flat, {B, T, cfg.linear_num_v_heads});

  // Metal kernel: handles T>1, returns (y [B, T, Hv, Dv], new_state [B, Hv, Dv, Dk])
  // Initial recurrent_state is zeros
  auto init_state = zeros(
      {B, cfg.linear_num_v_heads, cfg.linear_value_head_dim, cfg.linear_key_head_dim},
      q_dt);
  auto [y, new_recurrent_state] = gated_delta_kernel_call(q, k, v, g_3d, beta_3d, init_state);

  // RMSNorm gating — z is [B*T, value_dim], reshape to [B, T, Hv, Dv]
  auto z_h = reshape(z, {B, T, cfg.linear_num_v_heads, cfg.linear_value_head_dim});
  auto nw = get_weight(pfx + "norm.weight");
  auto y_normed = fast::rms_norm(y, nw, cfg.rms_norm_eps);
  y_normed = compiled_silu_mul()({z_h, y_normed})[0];

  // Output projection: flatten to [B*T, value_dim], project, reshape back
  auto y_flat = reshape(y_normed, {B * T, value_dim});
  auto out_flat = linear_proj(y_flat, pfx + "out_proj");
  auto output = reshape(out_flat, {B, T, hidden});

  return {output, new_conv_state, new_recurrent_state};
}

// =====================================================================
// Paged attention for compiled path — uses paged_kv_write + paged_attention
//
// Like attn_for_compile but writes new K/V into a per-layer block-paged pool
// (instead of a flat slice_update over a fixed [B, Hkv, max_kv_len, D] cache)
// and gathers attention K/V via PagedAttention (instead of a static
// additive mask over the flat cache).
//
// CONTRACT: SINGLE-TOKEN DECODE ONLY.
//
// This helper processes exactly ONE new token per call. As a hard invariant:
//
//     B == num_tokens == num_seqs == 1
//     slot_mapping.shape(0) == new_k.shape(0) == new_v.shape(0) == 1
//
// `num_valid_tokens` is therefore implicitly 1 and is NOT consulted here —
// `paged_kv_write` requires `slot_mapping.shape(0) == new_k.shape(0)`, so
// any caller violating the single-token invariant would crash inside the
// kernel validator. The FFI enforces this with an explicit guard before
// graph construction. Chunked prefill (B / num_tokens > 1) would need a
// second variant honoring the `num_valid_tokens` / sentinel-padded
// `slot_mapping` contract (see `crates/mlx-paged-attn/src/inputs.rs`).
//
// Inputs:
//   - x:                 [B=1, hidden] — 2D activation
//   - layer_idx:         transformer layer index (used for weight prefix)
//   - k_pool, v_pool:    per-layer paged K/V storage (shapes per
//                        `mlx_paged_ops.h`)
//   - k_scale, v_scale:  [1] f32 scale placeholders (1.0; reserved for
//                        FP8 calibration)
//   - offset_arr:        [1] int32 — global token position of the new token
//                        (used by RoPE; same role as in `attn_for_compile`)
//   - block_table:       [1, max_blocks_per_seq] int32, sentinel-padded -1
//   - slot_mapping:      [1] int64 — single active slot (NOT sentinel-padded;
//                        chunk_size_max MUST equal 1)
//   - num_valid_tokens:  [1] int32 (UNUSED; reserved for chunked prefill)
//   - num_valid_blocks:  [1] int32 (informational)
//   - seq_lens:          [1] int32 — total context length so far (paged_attn
//                        kernel reads `seq_lens[seq_idx=0]`)
//   - cfg:               BaseConfig with num_heads, num_kv_heads, head_dim,
//                        rope_dims, rope_theta, rms_norm_eps
//
// Configuration (hard-coded contract):
//   - block_size = 16 (passed in via `block_size` parameter for clarity).
//   - kv_dtype  = Bf16.
//   - x_pack    = 8 (= 16 / sizeof(bf16)).
//   - sliding_window = 0.
//
// Returns the layer output (`AttnPureResult.keys/values` are the post-write
// pool tensors so the compile graph dependency edges flow through
// `paged_kv_write` outputs into the next layer's `paged_attention` inputs;
// the caller plumbs them back into the global pool storage).
// =====================================================================
inline AttnPureResult attn_for_compile_paged(
    const array& x,                  // [B, hidden] — 2D
    int layer_idx,
    const array& k_pool,             // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const array& v_pool,             // [num_blocks, num_kv_heads, head_size, block_size]
    const array& k_scale,            // [1] f32 placeholder
    const array& v_scale,            // [1] f32 placeholder
    const array& offset_arr,         // [1] int32
    const array& block_table,        // [1, max_blocks_per_seq] int32
    const array& slot_mapping,       // [chunk_size_max] int64
    const array& /*num_valid_tokens*/, // [1] int32 (informational)
    const array& /*num_valid_blocks*/, // [1] int32 (informational)
    const array& seq_lens,           // [1] int32
    int block_size,
    const BaseConfig& cfg) {
  int B = x.shape(0);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  // Q projection (2x width for per-head gating)
  auto q_proj = linear_proj(x, pfx + "q_proj");
  if (has_weight(pfx + "q_proj.bias")) q_proj = q_proj + get_weight(pfx + "q_proj.bias");

  auto qph     = reshape(q_proj, {B, 1, cfg.num_heads, cfg.head_dim * 2});
  auto queries = slice(qph, {0, 0, 0, 0},            {B, 1, cfg.num_heads, cfg.head_dim});
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

  // RoPE — array offset (graph-safe, no baked-in constant)
  queries = fast::rope(queries, cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset_arr);
  keys    = fast::rope(keys,    cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset_arr);

  // Reshape to the layouts paged_kv_write / paged_attention expect:
  //   new_k / new_v: [num_tokens, num_kv_heads, head_size]
  //   q:             [num_seqs,   num_q_heads,  head_size]
  // For decode B == num_tokens == num_seqs == 1 and the time axis is 1.
  int num_tokens = B;  // single-token decode
  auto new_k = reshape(keys,    {num_tokens, cfg.num_kv_heads, cfg.head_dim});
  auto new_v = reshape(values,  {num_tokens, cfg.num_kv_heads, cfg.head_dim});
  auto q_pa  = reshape(queries, {num_tokens, cfg.num_heads,    cfg.head_dim});

  // Hard-coded contract: bf16 cache, x_pack=8, sliding=0.
  constexpr int X_PACK = 8;
  constexpr int SLIDING_WINDOW = 0;
  const auto kv_dtype = mlx::core::fast::KvDtype::Bf16;

  // 1) Write the new K/V into the paged pool.
  auto write_pair = mlx::core::fast::paged_kv_write(
      k_pool, v_pool,
      new_k, new_v,
      slot_mapping,
      k_scale, v_scale,
      block_size,
      cfg.num_kv_heads,
      cfg.head_dim,
      X_PACK,
      kv_dtype);
  auto new_k_pool = std::move(write_pair.first);
  auto new_v_pool = std::move(write_pair.second);

  // 2) Gather + attend via paged kernel against the just-written pool.
  //    The pool tensors are functionally aliased (paged_kv_write writes
  //    in-place) but using the post-write outputs makes the dependency edge
  //    explicit so MLX schedules the read after the write.
  float scale = std::pow((float)cfg.head_dim, -0.5f);
  auto attn_out = mlx::core::fast::paged_attention(
      q_pa,
      new_k_pool, new_v_pool,
      block_table, seq_lens,
      k_scale, v_scale,
      scale,
      /*softcap=*/0.0f,
      SLIDING_WINDOW,
      block_size,
      cfg.num_heads,
      cfg.num_kv_heads,
      cfg.head_dim,
      kv_dtype);

  // attn_out: [num_tokens, num_q_heads, head_size]. Flatten to [B, H*D] for
  // the gate + output projection (matches the flat-path layout).
  attn_out = reshape(attn_out, {B, cfg.num_heads * cfg.head_dim});

  // Gate
  attn_out = compiled_attn_gate()({attn_out, gate})[0];

  // Output projection
  auto output = linear_proj(attn_out, pfx + "o_proj");
  if (has_weight(pfx + "o_proj.bias")) output = output + get_weight(pfx + "o_proj.bias");

  // We re-purpose AttnPureResult.{keys,values} as the post-write pool
  // tensors so the caller can stash them back into the global pool slot.
  return {output, new_k_pool, new_v_pool};
}

// =====================================================================
// Batched attention forward for the MTP verify graph.
//
// Processes a contiguous chunk of `T` decode tokens in ONE forward (vs the
// flat-path `attn_pure_fn` / `attn_for_compile` helpers which both
// hard-wire T = 1). Used by the per-depth batched verify graph in
// `mlx_qwen35.cpp` / `mlx_qwen35_moe.cpp`.
//
// Inputs:
//   - x:                 `[B, T, hidden]` 3D activation (T = depth + 1).
//   - layer_idx:         transformer layer index — picks weight prefix.
//   - kv_keys/values:    flat-path KV caches `[B, Hkv, max_kv_len, D]`.
//                        T new K/V slots will be written at positions
//                        `[offset, offset+1, ..., offset+T-1]`.
//   - tail_mask:         additive bf16 mask of shape `[1, 1, T, max_kv_len]`
//                        — built once at graph entry so all layers reuse it.
//                        Position `i` (0..T-1) is valid for keys
//                        `[0..offset+i]` and masked out elsewhere.
//   - offset_arr:        `[1]` int32 — first new-slot position. Threaded
//                        through `fast::rope` so the RoPE position vector
//                        is `[offset, offset+1, ..., offset+T-1]`
//                        (intrinsic to RoPE's per-axis advance).
//
// Output:
//   - output:            `[B, T, hidden]` (3D).
//   - keys/values:       new KV caches with T slots written (same dtype
//                        and shape as inputs; only positions
//                        `[offset, offset+T)` have changed).
//
// Mirrors `attn_for_compile` (which the dense flat path also uses for
// T=1) with three changes:
//   1. Input is 3D `[B, T, hidden]` so the linear projections are batched
//      across both `B` and `T`. The reshape uses `T` explicitly.
//   2. `slice_update(kv, keys, offset_arr, {2})` writes T contiguous
//      slots at axis 2 starting at `offset_arr[0]` — the slice-update
//      kernel infers the slot count from `keys.shape(2)`.
//   3. The additive `tail_mask` is per-position (shape `[1, 1, T, B])
//      where B ≤ max_kv_len is the SDPA "bucket" — see `bucket_kv_len`.
//
// `bucket_kv_len` is the static SDPA key-column count baked into the
// compile trace. If `bucket_kv_len < max_kv_len`, the writeback
// still operates on the full `[B, Hkv, max_kv_len, D]` cache (so the
// returned `new_kv_keys/values` are full-size and the caller's
// `g_compiled_caches[]` mutation contract is unchanged), but the K/V
// view fed to SDPA is a static prefix `slice` of length `bucket_kv_len`.
// The bucket dispatcher (in `mlx_qwen35.cpp`) picks the smallest bucket
// ≥ `offset + T`, so all VALID key columns are inside the slice — the
// columns past offset+T-1 inside the slice are still zero-init and
// masked off by the tail mask.
//
// Passing `bucket_kv_len == 0` (default) preserves legacy behavior:
// SDPA sees the full max_kv_len cache. This keeps the legacy/fallback
// graph entry compilable without a separate code path.
//
// Weight keys read: `layers.{layer_idx}.self_attn.{q,k,v,o}_proj{.weight,
// .bias?}`, `.q_norm.weight`, `.k_norm.weight`. Same set as the per-step
// helpers; the batched path doesn't introduce new weights.
// =====================================================================
inline AttnPureResult attn_batched_verify_fn(
    const array& x,            // [B, T, hidden] — 3D
    int layer_idx,
    const array& kv_keys,      // [B, Hkv, max_kv_len, D]
    const array& kv_values,    // [B, Hkv, max_kv_len, D]
    const array& tail_mask,    // [1, 1, T, bucket_kv_len] additive bf16 mask
    const array& offset_arr,   // [1] int32
    const BaseConfig& cfg,
    int bucket_kv_len = 0) {
  int B = x.shape(0);
  int T = x.shape(1);
  int hidden = x.shape(2);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  // Project on flat 2D (matmul broadcasts cleanly across batch+time).
  auto x_flat = reshape(x, {B * T, hidden});

  auto q_proj = linear_proj(x_flat, pfx + "q_proj");
  if (has_weight(pfx + "q_proj.bias")) q_proj = q_proj + get_weight(pfx + "q_proj.bias");

  // Q has 2x width (per-head gating).
  auto qph     = reshape(q_proj, {B, T, cfg.num_heads, cfg.head_dim * 2});
  auto queries = slice(qph, {0, 0, 0, 0},            {B, T, cfg.num_heads, cfg.head_dim});
  auto gate    = slice(qph, {0, 0, 0, cfg.head_dim}, {B, T, cfg.num_heads, cfg.head_dim * 2});
  // gate is consumed by `compiled_attn_gate` which expects 2D `[BT, H*D]`.
  gate = reshape(gate, {B * T, cfg.num_heads * cfg.head_dim});

  auto keys   = linear_proj(x_flat, pfx + "k_proj");
  auto values = linear_proj(x_flat, pfx + "v_proj");
  if (has_weight(pfx + "k_proj.bias")) keys   = keys   + get_weight(pfx + "k_proj.bias");
  if (has_weight(pfx + "v_proj.bias")) values = values + get_weight(pfx + "v_proj.bias");

  keys   = reshape(keys,   {B, T, cfg.num_kv_heads, cfg.head_dim});
  values = reshape(values, {B, T, cfg.num_kv_heads, cfg.head_dim});

  // QK norm (per-head; shape preserved).
  queries = fast::rms_norm(queries, get_weight(pfx + "q_norm.weight"), cfg.rms_norm_eps);
  keys    = fast::rms_norm(keys,    get_weight(pfx + "k_norm.weight"), cfg.rms_norm_eps);

  // RoPE: array offset; queries/keys at axis 1 (time) get positions
  // [offset, offset+1, ..., offset+T-1]. Matches mlx-lm's
  // `Qwen3NextAttention.__call__` (qwen3_next.py:146-147) — the same
  // single-scalar `cache.offset` is used for prefill (T>1) decode batches.
  queries = fast::rope(queries, cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset_arr);
  keys    = fast::rope(keys,    cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset_arr);

  // Transpose for SDPA: [B, T, H, D] -> [B, H, T, D].
  queries = transpose(queries, {0, 2, 1, 3});
  auto keys_for_write   = transpose(keys,   {0, 2, 1, 3});  // [B, Hkv, T, D]
  auto values_for_write = transpose(values, {0, 2, 1, 3});

  // slice_update with array start: writes T contiguous K/V slots at
  // axis 2 starting at `offset_arr[0]`. The kernel infers the slot
  // count from the update tensor's axis-2 length (= T).
  auto new_kv_keys   = mlx::core::slice_update(kv_keys,   keys_for_write,   offset_arr, {2});
  auto new_kv_values = mlx::core::slice_update(kv_values, values_for_write, offset_arr, {2});

  // Bucketed SDPA view. The caller supplies a bucket size that's
  // (a) ≥ `offset + T` (so every valid key column is inside the slice)
  // and (b) a static integer baked into the compile trace (so SDPA sees
  // a smaller, fixed `[B, Hkv, bucket, D]` key tensor). `bucket_kv_len==0`
  // means "no bucketing" — SDPA reads the full max_kv_len cache (legacy).
  int Hkv = cfg.num_kv_heads;
  int max_kv_len = kv_keys.shape(2);
  array sdpa_keys   = new_kv_keys;
  array sdpa_values = new_kv_values;
  if (bucket_kv_len > 0 && bucket_kv_len < max_kv_len) {
    sdpa_keys   = slice(new_kv_keys,   {0, 0, 0, 0}, {B, Hkv, bucket_kv_len, cfg.head_dim});
    sdpa_values = slice(new_kv_values, {0, 0, 0, 0}, {B, Hkv, bucket_kv_len, cfg.head_dim});
  }

  // SDPA over the bucketed kv view with a `[1, 1, T, bucket]` (or
  // `[1, 1, T, max_kv_len]` in the legacy path) tail mask. Position-`t`
  // queries see keys `[0..offset+t]`; the mask construction in the caller
  // graph zeros that range and `-inf`s the rest.
  float scale = std::pow((float)cfg.head_dim, -0.5f);
  auto attn_out = fast::scaled_dot_product_attention(
      queries, sdpa_keys, sdpa_values, scale, "", tail_mask, {});

  // Transpose back to [B, T, H, D] -> [B*T, H*D].
  attn_out = transpose(attn_out, {0, 2, 1, 3});
  attn_out = reshape(attn_out, {B * T, cfg.num_heads * cfg.head_dim});

  // Gate (compiled silu*gate kernel; same kernel as per-step path).
  attn_out = compiled_attn_gate()({attn_out, gate})[0];

  // Output projection -> reshape back to 3D for the residual add upstream.
  auto out_flat = linear_proj(attn_out, pfx + "o_proj");
  if (has_weight(pfx + "o_proj.bias")) out_flat = out_flat + get_weight(pfx + "o_proj.bias");
  auto output = reshape(out_flat, {B, T, hidden});

  return {output, new_kv_keys, new_kv_values};
}

// Paged sibling of `attn_batched_verify_fn`. Replaces
// `slice_update` + SDPA over a BHTD cache with `paged_kv_write` +
// `paged_attention_varlen` over the vLLM-style pool. Q/K/V projection,
// QK norm, and RoPE are identical so kernel parity with the BHTD path
// holds up to the attention math itself (varlen kernel reduces softmax
// per query row independently of context length, so values can differ
// within bf16 noise vs. SDPA bucketed verify). The pool tensors
// returned in `keys` / `values` of `AttnPureResult` are the post-write
// handles produced by `paged_kv_write`; the caller stashes them back
// into the global pool slot for the next forward.
inline AttnPureResult attn_batched_verify_fn_paged(
    const array& x,                  // [B, T, hidden]
    int layer_idx,
    const array& k_pool,             // [num_blocks, num_kv_heads, head_size/x_pack, block_size, x_pack]
    const array& v_pool,             // [num_blocks, num_kv_heads, head_size, block_size]
    const array& k_scale,            // [1] f32
    const array& v_scale,            // [1] f32
    const array& offset_arr,         // [1] int32 — RoPE start position
    const array& block_table,        // [1, max_blocks_per_seq] int32
    const array& slot_mapping,       // [T = depth+1] int64 — exact length
                                     // required by paged_kv_write (its
                                     // shape(0) MUST equal new_k.shape(0)
                                     // = B*T). Caller slices any
                                     // chunk_size_max-padded array down
                                     // before passing it in.
    const array& seq_lens,           // [1] int32 — post-write context
    const array& cu_seqlens_q,       // [2] int32 — [0, T]
    int block_size,
    const BaseConfig& cfg) {
  int B = x.shape(0);
  int T = x.shape(1);
  int hidden = x.shape(2);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  auto x_flat = reshape(x, {B * T, hidden});

  auto q_proj = linear_proj(x_flat, pfx + "q_proj");
  if (has_weight(pfx + "q_proj.bias")) q_proj = q_proj + get_weight(pfx + "q_proj.bias");

  auto qph     = reshape(q_proj, {B, T, cfg.num_heads, cfg.head_dim * 2});
  auto queries = slice(qph, {0, 0, 0, 0},            {B, T, cfg.num_heads, cfg.head_dim});
  auto gate    = slice(qph, {0, 0, 0, cfg.head_dim}, {B, T, cfg.num_heads, cfg.head_dim * 2});
  gate = reshape(gate, {B * T, cfg.num_heads * cfg.head_dim});

  auto keys   = linear_proj(x_flat, pfx + "k_proj");
  auto values = linear_proj(x_flat, pfx + "v_proj");
  if (has_weight(pfx + "k_proj.bias")) keys   = keys   + get_weight(pfx + "k_proj.bias");
  if (has_weight(pfx + "v_proj.bias")) values = values + get_weight(pfx + "v_proj.bias");

  keys   = reshape(keys,   {B, T, cfg.num_kv_heads, cfg.head_dim});
  values = reshape(values, {B, T, cfg.num_kv_heads, cfg.head_dim});

  queries = fast::rms_norm(queries, get_weight(pfx + "q_norm.weight"), cfg.rms_norm_eps);
  keys    = fast::rms_norm(keys,    get_weight(pfx + "k_norm.weight"), cfg.rms_norm_eps);

  queries = fast::rope(queries, cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset_arr);
  keys    = fast::rope(keys,    cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset_arr);

  auto new_k_flat = reshape(keys,    {B * T, cfg.num_kv_heads, cfg.head_dim});
  auto new_v_flat = reshape(values,  {B * T, cfg.num_kv_heads, cfg.head_dim});
  auto q_flat     = reshape(queries, {B * T, cfg.num_heads,    cfg.head_dim});

  constexpr int X_PACK = 8;
  constexpr int SLIDING_WINDOW = 0;
  const auto kv_dtype = mlx::core::fast::KvDtype::Bf16;

  auto write_pair = mlx::core::fast::paged_kv_write(
      k_pool, v_pool,
      new_k_flat, new_v_flat,
      slot_mapping,
      k_scale, v_scale,
      block_size,
      cfg.num_kv_heads,
      cfg.head_dim,
      X_PACK,
      kv_dtype);
  auto new_k_pool = std::move(write_pair.first);
  auto new_v_pool = std::move(write_pair.second);

  float scale = std::pow((float)cfg.head_dim, -0.5f);
  auto attn_out = mlx::core::fast::paged_attention_varlen(
      q_flat,
      new_k_pool, new_v_pool,
      block_table, seq_lens, cu_seqlens_q,
      k_scale, v_scale,
      scale,
      /*softcap=*/0.0f,
      SLIDING_WINDOW,
      block_size,
      cfg.num_heads,
      cfg.num_kv_heads,
      cfg.head_dim,
      kv_dtype);

  attn_out = reshape(attn_out, {B * T, cfg.num_heads * cfg.head_dim});

  attn_out = compiled_attn_gate()({attn_out, gate})[0];

  auto out_flat = linear_proj(attn_out, pfx + "o_proj");
  if (has_weight(pfx + "o_proj.bias")) out_flat = out_flat + get_weight(pfx + "o_proj.bias");
  auto output = reshape(out_flat, {B, T, hidden});

  return {output, new_k_pool, new_v_pool};
}

// =====================================================================
// Batched GDN forward for the MTP verify graph.
//
// Processes `T` decode tokens in ONE Metal kernel call starting from the
// CURRENT recurrent + conv state (not zeros). Equivalent to looping
// `gdn_pure_fn` T times — the gated-delta Metal kernel already handles
// T > 1 internally and advances the recurrent state T steps in a single
// dispatch. Same kernel mlx-lm uses for prefill.
//
// Inputs:
//   - x:                 `[B, T, hidden]` 3D activation.
//   - layer_idx:         linear-attention layer index.
//   - conv_state:        `[B, kernel_dim-1, conv_dim]` — pre-verify state.
//   - recurrent_state:   `[B, Hv, Dv, Dk]` — pre-verify state.
//
// Outputs:
//   - output:            `[B, T, hidden]`.
//   - conv_state:        last `kernel_dim-1` conv-input rows after the T
//                        new qkv rows were appended.
//   - recurrent_state:   state after T recurrent advances.
// =====================================================================
// Batched GDN forward templated on `WithTape`. When `WithTape=false`
// returns `GDNPureResult`; when `WithTape=true` returns
// `GDNPureResultWithTape` carrying the per-step `(tape, k, g, qkv)` of
// shape `[B, T, ...]`. Mirrors `gdn_pure_fn_impl` but operates on T>=1
// tokens in a single kernel dispatch.
template <bool WithTape>
inline std::conditional_t<WithTape, GDNPureResultWithTape, GDNPureResult>
gdn_batched_verify_fn_impl(
    const array& x,                  // [B, T, hidden] — 3D
    int layer_idx,
    const array& conv_state,         // [B, kernel-1, conv_dim]
    const array& recurrent_state,    // [B, Hv, Dv, Dk]
    const BaseConfig& cfg) {
  int B = x.shape(0);
  int T = x.shape(1);
  int hidden = x.shape(2);
  int key_dim = cfg.linear_num_k_heads * cfg.linear_key_head_dim;
  int value_dim = cfg.linear_num_v_heads * cfg.linear_value_head_dim;
  int conv_dim = key_dim * 2 + value_dim;

  std::string pfx = "layers." + std::to_string(layer_idx) + ".linear_attn.";

  // Project on flat 2D — auto-detect merged-vs-split projection layout.
  auto x_flat = reshape(x, {B * T, hidden});
  struct QkvzResult { array qkv, z; };
  auto [qkv, z] = [&]() -> QkvzResult {
    if (has_weight(pfx + "in_proj_qkvz.weight")) {
      auto qkvz = linear_proj(x_flat, pfx + "in_proj_qkvz");
      return {slice(qkvz, {0, 0}, {B * T, conv_dim}),
              slice(qkvz, {0, conv_dim}, {B * T, key_dim * 2 + value_dim * 2})};
    }
    return {linear_proj(x_flat, pfx + "in_proj_qkv"),
            linear_proj(x_flat, pfx + "in_proj_z")};
  }();
  struct BaResult { array b, a; };
  auto [b, a] = [&]() -> BaResult {
    if (has_weight(pfx + "in_proj_ba.weight")) {
      auto ba = linear_proj(x_flat, pfx + "in_proj_ba");
      return {slice(ba, {0, 0}, {B * T, cfg.linear_num_v_heads}),
              slice(ba, {0, cfg.linear_num_v_heads}, {B * T, cfg.linear_num_v_heads * 2})};
    }
    return {linear_proj(x_flat, pfx + "in_proj_b"),
            linear_proj(x_flat, pfx + "in_proj_a")};
  }();

  // Re-3D the qkv for conv1d: `[B, T, conv_dim]`. When WithTape, this is
  // also the tensor recorded for conv-state replay (qkv_tape).
  auto qkv_3d = reshape(qkv, {B, T, conv_dim});

  // Conv1d: prepend the pre-verify conv_state, slide the kernel across
  // (kernel_dim-1 + T) inputs, take T outputs.
  auto conv_input = concatenate({conv_state, qkv_3d}, 1);
  // new_conv_state: last (kernel_dim-1) rows after the T new qkv rows.
  int keep = cfg.linear_conv_kernel_dim - 1;
  int total = keep + T;
  auto new_conv_state = slice(conv_input, {0, total - keep, 0}, {B, total, conv_dim});

  auto conv_w = get_weight(pfx + "conv1d.weight");
  auto conv_out = mlx::core::conv1d(conv_input, conv_w, 1, 0, 1, conv_dim);  // [B, T, conv_dim]

  // SiLU — compiled
  conv_out = compiled_silu()({conv_out})[0];

  // Split into q, k, v
  auto q = slice(conv_out, {0, 0, 0},              {B, T, key_dim});
  auto k = slice(conv_out, {0, 0, key_dim},        {B, T, key_dim * 2});
  auto v = slice(conv_out, {0, 0, key_dim * 2},    {B, T, conv_dim});

  // Reshape to heads: [B, T, H, D]
  q = reshape(q, {B, T, cfg.linear_num_k_heads, cfg.linear_key_head_dim});
  k = reshape(k, {B, T, cfg.linear_num_k_heads, cfg.linear_key_head_dim});
  v = reshape(v, {B, T, cfg.linear_num_v_heads, cfg.linear_value_head_dim});

  // RMS norm + scaling (same constants as gdn_pure_fn / gdn_prefill_fn).
  float inv_s = std::pow((float)cfg.linear_key_head_dim, -0.5f);
  auto q_dt = q.dtype();
  q = rms_norm_no_weight(q, 1e-6f) * array(inv_s * inv_s, q_dt);
  k = rms_norm_no_weight(k, 1e-6f) * array(inv_s, q_dt);

  // Beta, g — reshape b/a from [B*T, Hv] to [B, T, Hv]
  auto beta_3d = reshape(sigmoid(b), {B, T, cfg.linear_num_v_heads});
  auto a_log = get_weight(pfx + "A_log");
  auto dt_b  = get_weight(pfx + "dt_bias");
  auto g_flat = fused_compute_g(a_log, a, dt_b);
  auto g_3d = reshape(g_flat, {B, T, cfg.linear_num_v_heads});

  // Metal kernel — T advances starting from the CURRENT recurrent state.
  // Non-tape variant returns (y, recurrent_state); tape variant also
  // returns `[B, T, Hv, Dv]` fp32 per-step delta tape. `array` has no
  // default ctor — emit the call inside a constexpr lambda so the
  // downstream rmsnorm+out_proj graph can be shared.
  auto kernel_out = [&] {
    if constexpr (WithTape) {
      return gated_delta_kernel_with_tape_call(q, k, v, g_3d, beta_3d, recurrent_state);
    } else {
      return gated_delta_kernel_call(q, k, v, g_3d, beta_3d, recurrent_state);
    }
  }();
  auto& y                   = std::get<0>(kernel_out);
  auto& new_recurrent_state = std::get<1>(kernel_out);

  // RMSNorm gating — z is [B*T, value_dim], reshape to [B, T, Hv, Dv]
  auto z_h = reshape(z, {B, T, cfg.linear_num_v_heads, cfg.linear_value_head_dim});
  auto nw = get_weight(pfx + "norm.weight");
  auto y_normed = fast::rms_norm(y, nw, cfg.rms_norm_eps);
  y_normed = compiled_silu_mul()({z_h, y_normed})[0];

  // Output projection
  auto y_flat = reshape(y_normed, {B * T, value_dim});
  auto out_flat = linear_proj(y_flat, pfx + "out_proj");
  auto output = reshape(out_flat, {B, T, hidden});

  if constexpr (WithTape) {
    // k after rmsnorm+scale, g fp32 — captured before the kernel call.
    return GDNPureResultWithTape{
        std::move(output), std::move(new_conv_state), std::move(new_recurrent_state),
        std::move(std::get<2>(kernel_out)), std::move(k), std::move(g_3d), std::move(qkv_3d)};
  } else {
    return GDNPureResult{std::move(output), std::move(new_conv_state),
                         std::move(new_recurrent_state)};
  }
}

// Thin wrappers over `gdn_batched_verify_fn_impl`.
inline GDNPureResult gdn_batched_verify_fn(
    const array& x,
    int layer_idx,
    const array& conv_state,
    const array& recurrent_state,
    const BaseConfig& cfg) {
  return gdn_batched_verify_fn_impl<false>(x, layer_idx, conv_state, recurrent_state, cfg);
}

// Tape-recording variant of `gdn_batched_verify_fn`. Identical math;
// emits per-step `(tape, k, g, qkv)` tensors of shape `[B, T, ...]`
// directly (the Metal tape kernel emits the T-wide tape in one dispatch).
// The dense main path stashes these into the `g_gdn_*_tape_acc`
// accumulators in ONE assignment; the tape-replay rollback path consumes
// a `slice([:, 0..accepted_steps, ...])`.
inline GDNPureResultWithTape gdn_batched_verify_fn_with_tape(
    const array& x,
    int layer_idx,
    const array& conv_state,
    const array& recurrent_state,
    const BaseConfig& cfg) {
  return gdn_batched_verify_fn_impl<true>(x, layer_idx, conv_state, recurrent_state, cfg);
}

}  // namespace qwen35_common
