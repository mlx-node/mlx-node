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
// when the real graph is wired (Phase 1+) it is reused from
// `mlx_qwen35_common.h` rather than duplicated.
// =============================================================================

#include "mlx_common.h"

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

}  // namespace lfm2_common
