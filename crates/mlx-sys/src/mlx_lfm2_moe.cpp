#include "mlx_lfm2_common.h"

#include <cstddef>
#include <cstdint>

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

}  // extern "C"
