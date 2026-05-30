#include "mlx_lfm2_common.h"

#include <cstddef>
#include <cstdint>
#include <optional>

using namespace lfm2_common;

// Weight-registry FFI (defined in mlx_common_weights.cpp). Used by the
// component-parity probes below to register a single layer's weights into the
// shared `g_weights()` map before running the compiled pure-fns.
extern "C" {
void mlx_store_weight(const char* name, mlx_array* weight);
void mlx_clear_weights();
}

// =============================================================================
// LFM2.5 MoE Compiled Forward Pass — Phase 0 scaffold (INERT)
//
// These entry points are wired through FFI and compiled into the addon, but do
// NOTHING yet: `mlx_lfm2_get_model_id()` returns 0, so the Rust dispatcher's
// `mlx_lfm2_get_model_id() == model_id` gate (with model_id >= 1) is NEVER
// satisfied, and the model keeps running its Rust-native forward. The real
// compiled graph (dense attention + ShortConv + sparse MoE, modeled on the
// qwen3.5 compiled path) lands in later phases.
//
// IMPORTANT (Phase 1): before flipping the gate on, reconcile compiled-path
// OWNERSHIP with the qwen3.5 path — both read the SAME process-global weight
// map (`g_weights()` in mlx_qwen35_common.h). Only one model may own it at a
// time, so the active model id must be the single source of truth and the
// per-model id counters must not collide across models. See mlx_qwen35.cpp for
// the ownership/registration pattern.
// =============================================================================

namespace {
// File-local active model id. Phase 0 has no setter, so it stays 0 and the
// Rust gate (`mlx_lfm2_get_model_id() == model_id`, where model_id >= 1) is
// always false.
uint64_t g_lfm2_active_model_id = 0;
}  // namespace

extern "C" {

// GATE source. Returns 0 in Phase 0 (no setter wired) → compiled path OFF.
uint64_t mlx_lfm2_get_model_id() { return g_lfm2_active_model_id; }

// Inert: no weights are registered into a compiled graph yet.
size_t mlx_lfm2_weight_count() { return 0; }

// Inert: accepts the prefill config the real graph will need, does nothing.
// (Phase 1+ builds and seeds the compiled decode graph from these args.)
void mlx_lfm2_moe_init_from_prefill(
    int /*num_layers*/,
    int /*hidden_size*/,
    int /*num_heads*/,
    int /*num_kv_heads*/,
    int /*head_dim*/,
    float /*rope_theta*/,
    float /*norm_eps*/,
    int /*conv_l_cache*/,
    int /*num_experts*/,
    int /*num_experts_per_tok*/,
    int /*num_dense_layers*/,
    int /*norm_topk_prob*/,
    int /*use_expert_bias*/,
    int /*tie_embedding*/,
    int /*max_kv_len*/,
    int /*batch_size*/,
    mlx_array** /*cache_arrays*/,
    int /*prefill_offset*/) {
  // Phase 1+: build and seed the compiled decode graph here.
}

// Inert: ALWAYS returns a null logits pointer so any accidental caller detects
// "compiled path not enabled" and falls back to the native forward.
void mlx_lfm2_moe_forward(
    mlx_array* /*input_ids*/,
    mlx_array* /*embedding_weight*/,
    mlx_array** output_logits,
    int* /*cache_offset_out*/) {
  if (output_logits) {
    *output_logits = nullptr;
  }
}

// Inert: nothing to tear down yet.
void mlx_lfm2_moe_reset() { g_lfm2_active_model_id = 0; }

// =============================================================================
// Component-parity probes (test-only). These register ONE layer's weights into
// the shared weight map, run the compiled pure-fn, and return the output so a
// Rust test can compare it to the native Rust-side forward. They are the
// Phase-1 parity gate (the full compiled forward is not end-to-end runnable
// until the ShortConv operator lands in Phase 2). Each probe `mlx_clear_weights`
// first for a clean slate; in the mlx-core unit-test binary no other code
// touches `g_weights()`, so this is race-free.
// =============================================================================

// Run a SEQUENCE of `T` lfm2 attention decode steps (B=1, offset 0..T-1)
// through `lfm2_attn_pure_fn`, threading the KV cache, and return the LAST
// step's output `[1, num_heads*head_dim]`. Running a sequence (not a single
// step) is what actually exercises multi-key softmax, the RoPE offset, and the
// QK layernorm — a single step's softmax over one key is trivially 1.0.
//
// `x_seq` is `[T, hidden]` (one decode input per row). Weights are natural
// `[out, in]` (q/k/v/out_proj) / `[head_dim]` (q/k_layernorm) — identical to
// what the native `Lfm2Attention` holds. Caller owns the returned array;
// nullptr on error.
mlx_array* mlx_lfm2_probe_attn_seq(
    mlx_array* x_seq_ptr,
    mlx_array* q_w, mlx_array* k_w, mlx_array* v_w, mlx_array* out_w,
    mlx_array* q_norm_w, mlx_array* k_norm_w,
    int num_heads, int num_kv_heads, int head_dim,
    float rope_theta, float norm_eps) {
  try {
    mlx_clear_weights();
    mlx_store_weight("layers.0.self_attn.q_proj.weight", q_w);
    mlx_store_weight("layers.0.self_attn.k_proj.weight", k_w);
    mlx_store_weight("layers.0.self_attn.v_proj.weight", v_w);
    mlx_store_weight("layers.0.self_attn.out_proj.weight", out_w);
    mlx_store_weight("layers.0.self_attn.q_layernorm.weight", q_norm_w);
    mlx_store_weight("layers.0.self_attn.k_layernorm.weight", k_norm_w);

    Lfm2MoeConfig cfg{};
    cfg.num_heads = num_heads;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim = head_dim;
    cfg.rope_theta = rope_theta;
    cfg.norm_eps = norm_eps;

    auto& x_seq = *reinterpret_cast<array*>(x_seq_ptr);
    int T = x_seq.shape(0);
    int hidden = x_seq.shape(1);

    auto kv_keys = zeros({1, num_kv_heads, T, head_dim}, x_seq.dtype());
    auto kv_values = zeros({1, num_kv_heads, T, head_dim}, x_seq.dtype());
    auto dummy_mask = zeros({1, 1, 1, 1}, mlx::core::bfloat16);

    array last_out = zeros({1, num_heads * head_dim}, x_seq.dtype());
    for (int i = 0; i < T; i++) {
      auto x_i = reshape(slice(x_seq, {i, 0}, {i + 1, hidden}), {1, hidden});
      auto res = lfm2_attn_pure_fn(x_i, 0, kv_keys, kv_values, dummy_mask, i,
                                   cfg, /*dynamic_kv=*/true);
      kv_keys = res.keys;
      kv_values = res.values;
      last_out = res.output;
    }
    mlx::core::eval({last_out});
    auto* out = new array(last_out);
    mlx_clear_weights();
    return reinterpret_cast<mlx_array*>(out);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] mlx_lfm2_probe_attn_seq: %s\n", e.what());
    fflush(stderr);
    mlx_clear_weights();
    return nullptr;
  } catch (...) {
    mlx_clear_weights();
    return nullptr;
  }
}

// Run the dense SwiGLU MLP through `lfm2_dense_mlp`. Weights are natural
// [out, in]. Caller owns the returned array. Returns nullptr on error.
mlx_array* mlx_lfm2_probe_dense_mlp(
    mlx_array* x_ptr, mlx_array* gate_w, mlx_array* up_w, mlx_array* down_w) {
  try {
    mlx_clear_weights();
    mlx_store_weight("layers.0.feed_forward.gate_proj.weight", gate_w);
    mlx_store_weight("layers.0.feed_forward.up_proj.weight", up_w);
    mlx_store_weight("layers.0.feed_forward.down_proj.weight", down_w);

    auto& x = *reinterpret_cast<array*>(x_ptr);
    auto res = lfm2_dense_mlp(x, 0);
    mlx::core::eval({res});
    auto* out = new array(res);
    mlx_clear_weights();
    return reinterpret_cast<mlx_array*>(out);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] mlx_lfm2_probe_dense_mlp: %s\n", e.what());
    fflush(stderr);
    mlx_clear_weights();
    return nullptr;
  } catch (...) {
    mlx_clear_weights();
    return nullptr;
  }
}

}  // extern "C"
