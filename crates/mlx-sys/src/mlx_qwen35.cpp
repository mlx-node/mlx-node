#include "mlx_qwen35_common.h"
#include <cstdlib>

using namespace qwen35_common;

// =============================================================================
// Qwen3.5 Dense Compiled Forward Pass
//
// Implements the entire Qwen3.5 dense model forward pass (single-token decode)
// in one FFI call. Weight storage, helpers, GDN and attention functions are
// shared with the MoE path via mlx_qwen35_common.h.
//
// Phase 5 piece 1 adds a parallel paged-decode graph
// (`dense_compiled_decode_fn_paged`) that routes full-attention layers
// through the paged-attention kernels while keeping linear-attention
// (GDN) layers on the same `gdn_pure_fn` helper as the flat path. The
// paged-path globals (`g_dense_paged_*`) are independent from the flat
// path globals (`g_compiled_*`) so both graphs can coexist; the
// dispatcher chooses one or the other per turn but never mixes them
// within a single decode step.
// =============================================================================

namespace {

// Config for the compiled path (extends BaseConfig with compile-specific state)
struct CompileConfig : BaseConfig {};

static CompileConfig g_compile_config{};
static std::vector<array> g_compiled_caches;         // num_layers * 2 arrays
static std::optional<array> g_compiled_offset;       // scalar int32 (current decode position)
static int g_offset_int = 0;                         // C++ int mirror of g_compiled_offset
static bool g_compile_inited = false;

// =====================================================================
// Phase 5 piece 1: paged-decode globals.
//
// Independent from the flat-path globals above so both compile graphs
// can coexist while the Rust dispatcher decides per-turn whether to
// take the compiled paged path or fall back to the legacy flat path.
//
// Layout (size = num_layers; one entry per layer indexed by `layer_idx`):
//   - g_dense_k_pools / g_dense_v_pools / g_dense_k_scales /
//     g_dense_v_scales: meaningful only for full-attention layers.
//     Linear-layer slots hold a small placeholder array — they're never
//     read by the paged graph.
//   - g_dense_paged_linear_caches: size = num_layers * 2. Slot `2i` and
//     `2i+1` hold conv_state and recurrent_state for layer `i` when
//     that layer is linear-attention. Full-attn slots hold placeholders.
//
// `g_dense_paged_inited` gates the new `mlx_qwen35_forward_paged` FFI.
// `mlx_qwen35_init_paged` is the only way to flip it true; clearing
// happens in `mlx_qwen35_compiled_reset` so a single reset wipes BOTH
// graphs' state.
// =====================================================================
static CompileConfig g_dense_paged_config{};
static std::vector<array> g_dense_k_pools;          // [num_layers]
static std::vector<array> g_dense_v_pools;          // [num_layers]
static std::vector<array> g_dense_k_scales;         // [num_layers]
static std::vector<array> g_dense_v_scales;         // [num_layers]
static std::vector<array> g_dense_paged_linear_caches;  // [num_layers * 2]
static int g_dense_paged_offset_int = 0;
static bool g_dense_paged_inited = false;

static bool dense_paged_is_linear_layer(int layer) {
  int interval = g_dense_paged_config.full_attention_interval;
  return interval <= 0 || ((layer + 1) % interval != 0);
}

// =============================================================================
// The compilable forward function
// inputs: [h, offset_arr, cache[0].a, cache[0].b, ..., cache[N-1].a, cache[N-1].b]
// outputs: [logits, new_offset, new_cache[0].a, new_cache[0].b, ...]
// =============================================================================
static std::vector<array> qwen35_decode_fn(const std::vector<array>& inputs) {
  const auto& cfg = g_compile_config;
  auto h = inputs[0];
  int offset = g_offset_int;

  // Attention mask: [1, 1, 1, max_kv_len]
  int first_fa = cfg.full_attention_interval - 1;
  int max_kv_len = inputs[2 + first_fa * 2].shape(2);
  auto positions = arange(0, max_kv_len, mlx::core::int32);
  auto valid_mask = less_equal(positions, array(offset, mlx::core::int32));
  auto attn_mask = where(valid_mask,
      array(0.0f, mlx::core::bfloat16),
      array(-std::numeric_limits<float>::infinity(), mlx::core::bfloat16));
  attn_mask = reshape(attn_mask, {1, 1, 1, max_kv_len});

  std::vector<array> new_caches;
  new_caches.reserve(cfg.num_layers * 2);
  for (int i = 0; i < cfg.num_layers * 2; i++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  for (int i = 0; i < cfg.num_layers; i++) {
    bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
    std::string lp = "layers." + std::to_string(i);

    auto normed = fast::rms_norm(h, get_weight(lp + ".input_layernorm.weight"), cfg.rms_norm_eps);

    array layer_out = zeros({}, mlx::core::bfloat16);
    if (is_linear) {
      const auto& cs = inputs[2 + i * 2];
      const auto& rs = inputs[2 + i * 2 + 1];
      auto res = gdn_pure_fn(normed, i, cs, rs, cfg);
      layer_out = std::move(res.output);
      new_caches[i * 2]     = std::move(res.conv_state);
      new_caches[i * 2 + 1] = std::move(res.recurrent_state);
    } else {
      const auto& kk = inputs[2 + i * 2];
      const auto& kv = inputs[2 + i * 2 + 1];
      auto res = attn_pure_fn(normed, i, kk, kv, attn_mask, offset, cfg);
      layer_out = std::move(res.output);
      new_caches[i * 2]     = std::move(res.keys);
      new_caches[i * 2 + 1] = std::move(res.values);
    }
    h = h + layer_out;

    // MLP (SwiGLU)
    std::string mp = lp + ".mlp.";
    auto mlp_in  = fast::rms_norm(h, get_weight(lp + ".post_attention_layernorm.weight"), cfg.rms_norm_eps);
    auto gate    = linear_proj(mlp_in, mp + "gate_proj");
    auto up      = linear_proj(mlp_in, mp + "up_proj");
    auto mlp_out = linear_proj(swiglu(gate, up), mp + "down_proj");
    h = h + mlp_out;
  }

  // Final norm + LM head
  h = fast::rms_norm(h, get_weight("final_norm.weight"), cfg.rms_norm_eps);
  if (cfg.tie_word_embeddings) {
    h = linear_proj(h, "embedding");
  } else {
    h = linear_proj(h, "lm_head");
  }

  auto new_offset = array(offset + 1, mlx::core::int32);

  std::vector<array> result;
  result.reserve(2 + cfg.num_layers * 2);
  result.push_back(std::move(h));
  result.push_back(std::move(new_offset));
  for (auto& c : new_caches) result.push_back(std::move(c));
  return result;
}

// Note: mlx::core::compile(qwen35_decode_fn) is NOT used here because the
// compile cache is invalidated every step due to the changing g_offset_int
// captured in qwen35_decode_fn. Instead, we call qwen35_decode_fn directly
// and rely on the inner compiled helpers (compiled_swiglu, compiled_compute_g,
// etc.) for kernel fusion.

// =====================================================================
// Phase 5 piece 1: full-graph paged-decode compile function.
//
// Mirrors `moe_compiled_decode_fn_paged` from `mlx_qwen35_moe.cpp` but
// without expert routing — every layer either runs `gdn_pure_fn`
// (linear) or `attn_for_compile_paged` (full attention) followed by a
// dense SwiGLU MLP.
//
// Unlike the legacy `qwen35_decode_fn` flat path (which captures a
// scalar `g_offset_int` and therefore invalidates the compile cache on
// every step), this graph takes the offset as an array input
// (`offset_arr`). All shapes are fixed at trace time, so
// `mlx::core::compile(...)` produces a cache key that stays valid
// across all decode steps within one turn.
//
// Input vector layout (matches the MoE paged graph):
//   [0]                  h:                  embedding [B, hidden]
//   [1]                  offset_arr:         [1] int32
//   [2]                  block_table:        [1, max_blocks_per_seq] int32
//   [3]                  slot_mapping:       [1] int64
//   [4]                  num_valid_tokens:   [1] int32
//   [5]                  num_valid_blocks:   [1] int32
//   [6]                  seq_lens:           [1] int32
//   For each layer i in [0, num_layers):
//     If linear:
//       [7 + i*4 + 0]    conv_state
//       [7 + i*4 + 1]    recurrent_state
//       [7 + i*4 + 2]    placeholder         (unused — keeps stride uniform)
//       [7 + i*4 + 3]    placeholder         (unused — keeps stride uniform)
//     If full-attention:
//       [7 + i*4 + 0]    k_pool
//       [7 + i*4 + 1]    v_pool
//       [7 + i*4 + 2]    k_scale
//       [7 + i*4 + 3]    v_scale
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
//       [2 + i*2 + 1]    new_v_pool
//
// The 4-input / 2-output stride is identical to the MoE paged graph so
// the same `attn_for_compile_paged` helper plumbs in unchanged.
// =====================================================================
static std::vector<array> dense_compiled_decode_fn_paged(const std::vector<array>& inputs) {
  const auto& cfg = g_dense_paged_config;
  auto h          = inputs[0];
  auto offset_arr = inputs[1];   // [1] int32
  auto block_table      = inputs[2];
  auto slot_mapping     = inputs[3];
  auto num_valid_tokens = inputs[4];
  auto num_valid_blocks = inputs[5];
  auto seq_lens         = inputs[6];

  // Phase 5 piece 1 hard-coded contract (matches the MoE paged graph).
  constexpr int BLOCK_SIZE = 16;

  constexpr int kHeader = 7;
  constexpr int kPerLayer = 4;

  // Pre-allocate new_caches with placeholders. Output stride = 2 per layer.
  std::vector<array> new_caches;
  new_caches.reserve(cfg.num_layers * 2);
  for (int i = 0; i < cfg.num_layers * 2; i++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  for (int i = 0; i < cfg.num_layers; i++) {
    bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
    std::string lp = "layers." + std::to_string(i);

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

    // Dense MLP (SwiGLU) — no MoE routing.
    std::string mp = lp + ".mlp.";
    auto mlp_in  = fast::rms_norm(h, get_weight(lp + ".post_attention_layernorm.weight"), cfg.rms_norm_eps);
    auto gate    = linear_proj(mlp_in, mp + "gate_proj");
    auto up      = linear_proj(mlp_in, mp + "up_proj");
    auto mlp_out = linear_proj(swiglu(gate, up), mp + "down_proj");
    h = h + mlp_out;
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

static auto& compiled_dense_decode_paged() {
  static auto fn = mlx::core::compile(dense_compiled_decode_fn_paged);
  return fn;
}

} // namespace

// =============================================================================
// Public FFI functions
// =============================================================================

extern "C" {

// Weight storage FFI (mlx_store_weight, mlx_clear_weights,
// mlx_weight_count, mlx_set_model_id) moved to
// mlx_common_weights.cpp — shared by all compiled model forward passes.

uint64_t mlx_qwen35_get_model_id() {
  return g_active_model_id().load(std::memory_order_acquire);
}

void mlx_qwen35_compiled_init_from_prefill(
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
    mlx_array** cache_arrays,
    int prefill_offset
) {
  try {
    g_compile_config = CompileConfig{{
      num_layers, hidden_size, num_heads, num_kv_heads, head_dim,
      rope_theta, rope_dims, rms_norm_eps, full_attention_interval,
      linear_num_k_heads, linear_num_v_heads, linear_key_head_dim,
      linear_value_head_dim, linear_conv_kernel_dim,
      tie_word_embeddings != 0,
      max_kv_len, batch_size
    }};

    g_compiled_caches.clear();
    g_compiled_caches.reserve(num_layers * 2);

    for (int i = 0; i < num_layers; i++) {
      bool is_linear = !((i + 1) % full_attention_interval == 0);

      if (is_linear) {
        g_compiled_caches.push_back(*reinterpret_cast<array*>(cache_arrays[i * 2]));
        g_compiled_caches.push_back(*reinterpret_cast<array*>(cache_arrays[i * 2 + 1]));
      } else {
        auto& kk = *reinterpret_cast<array*>(cache_arrays[i * 2]);
        auto& kv = *reinterpret_cast<array*>(cache_arrays[i * 2 + 1]);
        int current_cap = kk.shape(2);
        if (current_cap < max_kv_len) {
          int pad_len = max_kv_len - current_cap;
          auto kpad = zeros({batch_size, num_kv_heads, pad_len, head_dim}, kk.dtype());
          auto vpad = zeros({batch_size, num_kv_heads, pad_len, head_dim}, kv.dtype());
          g_compiled_caches.push_back(concatenate({kk, kpad}, 2));
          g_compiled_caches.push_back(concatenate({kv, vpad}, 2));
        } else {
          g_compiled_caches.push_back(kk);
          g_compiled_caches.push_back(kv);
        }
      }
    }

    g_compiled_offset = array(prefill_offset, mlx::core::int32);
    g_offset_int = prefill_offset;
    g_compile_inited = true;

    // Break the lazy RNG split chain from model initialization.
    auto rng_key = mlx::core::random::KeySequence::default_().next();
    mlx::core::eval({rng_key});
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_qwen35_compiled_init_from_prefill: " << e.what() << std::endl;
    g_compile_inited = false;
  }
}

void mlx_qwen35_forward_compiled(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    mlx_array** output_logits,
    int* cache_offset_out
) {
  if (!input_ids_ptr || !embedding_weight_ptr || !output_logits) {
    if (output_logits) *output_logits = nullptr;
    return;
  }
  if (!g_compile_inited) {
    *output_logits = nullptr;
    return;
  }
  const auto& cfg = g_compile_config;

  try {
    auto& input_ids      = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    auto flat_ids = reshape(input_ids, {-1});
    auto h = take(embedding_weight, flat_ids, 0);

    std::vector<array> inputs;
    inputs.reserve(2 + cfg.num_layers * 2);
    inputs.push_back(std::move(h));
    inputs.push_back(array(g_offset_int, mlx::core::int32));
    for (const auto& c : g_compiled_caches) {
      inputs.push_back(c);
    }

    auto outputs = qwen35_decode_fn(inputs);

    *output_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    g_offset_int++;
    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_compiled_caches[i] = outputs[2 + i];
    }

    if (cache_offset_out) {
      *cache_offset_out = g_offset_int;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_forward_compiled: %s\n", e.what());
    fflush(stderr);
    *output_logits = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_forward_compiled\n");
    fflush(stderr);
    *output_logits = nullptr;
  }
}

void mlx_qwen35_eval_token_and_compiled_caches(mlx_array* next_token_ptr) {
  try {
    std::vector<array> to_eval;
    to_eval.reserve(1 + g_compiled_caches.size());
    to_eval.push_back(*reinterpret_cast<array*>(next_token_ptr));
    for (const auto& c : g_compiled_caches) {
      to_eval.push_back(c);
    }
    mlx::core::async_eval(std::move(to_eval));
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in eval_token_and_compiled_caches: %s\n", e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in eval_token_and_compiled_caches\n");
    fflush(stderr);
  }
}

void mlx_qwen35_compiled_adjust_offset(int delta) {
  g_offset_int += delta;
  g_compiled_offset = array(g_offset_int, mlx::core::int32);
}

// Reset BOTH the legacy flat compiled state AND the Phase 5 piece 1
// paged-path globals. Keeping these symmetric is required because
// `mlx_qwen35_init_paged` flips `g_dense_paged_inited` to true
// independently of `g_compile_inited`; without clearing the paged side
// here, a later `mlx_qwen35_forward_paged()` would pass the init guard
// and reuse stale KV pools / scales / linear caches / offset from a
// previous request or model. Mirrors `mlx_qwen35_moe_reset` from the
// MoE path.
void mlx_qwen35_compiled_reset() {
  // Legacy flat-path compiled globals.
  g_compiled_caches.clear();
  g_compiled_offset = std::nullopt;
  g_offset_int = 0;
  g_compile_inited = false;

  // Phase 5 piece 1 paged-path globals.
  g_dense_paged_config = CompileConfig{};
  g_dense_k_pools.clear();
  g_dense_v_pools.clear();
  g_dense_k_scales.clear();
  g_dense_v_scales.clear();
  g_dense_paged_linear_caches.clear();
  g_dense_paged_offset_int = 0;
  g_dense_paged_inited = false;
}

// Export compiled caches for PromptCache reuse.
int mlx_qwen35_export_caches(mlx_array** out_ptrs, int max_count) {
  if (!g_compile_inited || g_compiled_caches.empty()) return 0;
  int count = std::min((int)g_compiled_caches.size(), max_count);
  for (int i = 0; i < count; i++) {
    out_ptrs[i] = reinterpret_cast<mlx_array*>(new array(g_compiled_caches[i]));
  }
  return count;
}

int mlx_qwen35_get_cache_offset() {
  return g_offset_int;
}

int mlx_qwen35_export_paged_linear_caches(mlx_array** out_ptrs, int max_count) {
  if (!g_dense_paged_inited || g_dense_paged_linear_caches.empty()) return 0;
  int expected = static_cast<int>(g_dense_paged_linear_caches.size());
  int count = std::min(expected, max_count);
  for (int i = 0; i < count; i++) {
    out_ptrs[i] = nullptr;
  }
  for (int layer = 0; layer < g_dense_paged_config.num_layers; layer++) {
    int base = layer * 2;
    if (base + 1 >= count) break;
    if (!dense_paged_is_linear_layer(layer)) continue;
    out_ptrs[base] = reinterpret_cast<mlx_array*>(new array(g_dense_paged_linear_caches[base]));
    out_ptrs[base + 1] = reinterpret_cast<mlx_array*>(new array(g_dense_paged_linear_caches[base + 1]));
  }
  return count;
}

int mlx_qwen35_get_paged_cache_offset() {
  return g_dense_paged_offset_int;
}

// =============================================================================
// Phase 5 piece 1: paged Dense forward FFI.
//
// These coexist alongside `mlx_qwen35_forward_compiled` /
// `_compiled_init_from_prefill` while the Rust dispatcher decides
// per-turn which graph to run. A single `mlx_qwen35_compiled_reset()`
// wipes BOTH graphs' state.
// =============================================================================

// Initialize the paged Dense forward graph from per-layer pool / scale
// handles AND per-layer linear-attention recurrent caches.
//
// Layout contract (mirrors `mlx_qwen35_moe_init_paged`):
//   - `k_pool_handles[i]`: pointer to a `[num_blocks, num_kv_heads,
//     head_size/x_pack=8, block_size=16, x_pack=8]` bf16 array view.
//     Phase 5 piece 1 hard-codes bf16 (`KvDtype::Bf16`) and
//     `block_size = 16`.
//   - `v_pool_handles[i]`: pointer to a `[num_blocks, num_kv_heads,
//     head_size, block_size=16]` bf16 array view.
//   - `k_scale_handles[i]` / `v_scale_handles[i]`: pointer to `[1]` f32
//     scale placeholders (1.0 in Phase 5; FP8 calibration in Phase 10).
//   - For linear-attention layers (those satisfying
//     `(i + 1) % full_attention_interval != 0`), the corresponding pool
//     / scale slots may be null — they're stored as bf16 zero
//     placeholders and never read by the compiled graph.
//
// `linear_cache_arrays` mirrors `cache_arrays` in
// `mlx_qwen35_compiled_init_from_prefill` for linear layers only:
// pairs of `(conv_state, recurrent_state)` indexed by layer.
// Full-attn slots are ignored. Pass null for the entire array to skip
// linear-cache seeding.
//
// `prefill_offset` becomes the initial `g_dense_paged_offset_int`.
//
// Compile-graph configuration:
//   - block_size       = 16
//   - kv_dtype         = Bf16
//   - x_pack           = 8
//   - sliding_window   = 0
//
// Returns 0 on success; -1 on failure. On failure
// `g_dense_paged_inited` is cleared and a stderr diagnostic is emitted;
// the Rust caller MUST inspect the return value and fall back to the
// pure-Rust paged path rather than entering the compiled paged decode
// (which would dispatch against uninitialized globals).
int32_t mlx_qwen35_init_paged(
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
    g_dense_paged_config = CompileConfig{{
      num_layers, hidden_size, num_heads, num_kv_heads, head_dim,
      rope_theta, rope_dims, rms_norm_eps, full_attention_interval,
      linear_num_k_heads, linear_num_v_heads, linear_key_head_dim,
      linear_value_head_dim, linear_conv_kernel_dim,
      tie_word_embeddings != 0,
      max_kv_len, batch_size
    }};

    // Reset the paged globals.
    g_dense_k_pools.clear();
    g_dense_v_pools.clear();
    g_dense_k_scales.clear();
    g_dense_v_scales.clear();
    g_dense_paged_linear_caches.clear();
    g_dense_k_pools.reserve(num_layers);
    g_dense_v_pools.reserve(num_layers);
    g_dense_k_scales.reserve(num_layers);
    g_dense_v_scales.reserve(num_layers);
    g_dense_paged_linear_caches.reserve(num_layers * 2);

    auto bf16_placeholder = []() { return zeros({}, mlx::core::bfloat16); };
    auto f32_placeholder  = []() { return array(1.0f, mlx::core::float32); };

    for (int i = 0; i < num_layers; i++) {
      bool is_linear = dense_paged_is_linear_layer(i);

      // Pool / scale slots: meaningful for full-attn layers only.
      if (!is_linear) {
        if (!k_pool_handles || !v_pool_handles ||
            !k_scale_handles || !v_scale_handles ||
            !k_pool_handles[i] || !v_pool_handles[i] ||
            !k_scale_handles[i] || !v_scale_handles[i]) {
          g_dense_paged_inited = false;
          std::cerr << "[MLX] mlx_qwen35_init_paged: missing pool/scale handle for full-attn layer " << i << std::endl;
          return -1;
        }
        g_dense_k_pools.push_back(*reinterpret_cast<array*>(k_pool_handles[i]));
        g_dense_v_pools.push_back(*reinterpret_cast<array*>(v_pool_handles[i]));
        g_dense_k_scales.push_back(*reinterpret_cast<array*>(k_scale_handles[i]));
        g_dense_v_scales.push_back(*reinterpret_cast<array*>(v_scale_handles[i]));
      } else {
        // Linear layer: stash placeholders so per-layer indexing works.
        g_dense_k_pools.push_back(bf16_placeholder());
        g_dense_v_pools.push_back(bf16_placeholder());
        g_dense_k_scales.push_back(f32_placeholder());
        g_dense_v_scales.push_back(f32_placeholder());
      }

      // Linear caches: meaningful for linear-attn layers only.
      if (is_linear && linear_cache_arrays &&
          linear_cache_arrays[i * 2] && linear_cache_arrays[i * 2 + 1]) {
        g_dense_paged_linear_caches.push_back(*reinterpret_cast<array*>(linear_cache_arrays[i * 2]));
        g_dense_paged_linear_caches.push_back(*reinterpret_cast<array*>(linear_cache_arrays[i * 2 + 1]));
      } else {
        g_dense_paged_linear_caches.push_back(bf16_placeholder());
        g_dense_paged_linear_caches.push_back(bf16_placeholder());
      }
    }

    g_dense_paged_offset_int = prefill_offset;

    // Defense-in-depth: surface layout / dtype / Metal-availability
    // failures HERE (init time) rather than letting them blow up inside
    // the first `mlx_qwen35_forward_paged` call where the Rust caller's
    // `record_tokens` has already mutated adapter state. We force-eval
    // every full-attn pool / scale handle so the bf16 / f32 layouts are
    // materialized on the Metal queue and any underlying allocation
    // failure raises a c++ exception we catch below.
    {
      std::vector<array> probe;
      probe.reserve(num_layers * 4 + 1);
      for (int i = 0; i < num_layers; i++) {
        bool is_linear = dense_paged_is_linear_layer(i);
        if (is_linear) continue;
        // Validate dtype contract: pools must be bf16, scales must be f32.
        if (g_dense_k_pools[i].dtype() != mlx::core::bfloat16) {
          std::cerr << "[MLX] mlx_qwen35_init_paged: layer " << i
                    << " k_pool dtype != bf16" << std::endl;
          g_dense_paged_inited = false;
          return -1;
        }
        if (g_dense_v_pools[i].dtype() != mlx::core::bfloat16) {
          std::cerr << "[MLX] mlx_qwen35_init_paged: layer " << i
                    << " v_pool dtype != bf16" << std::endl;
          g_dense_paged_inited = false;
          return -1;
        }
        if (g_dense_k_scales[i].dtype() != mlx::core::float32) {
          std::cerr << "[MLX] mlx_qwen35_init_paged: layer " << i
                    << " k_scale dtype != f32" << std::endl;
          g_dense_paged_inited = false;
          return -1;
        }
        if (g_dense_v_scales[i].dtype() != mlx::core::float32) {
          std::cerr << "[MLX] mlx_qwen35_init_paged: layer " << i
                    << " v_scale dtype != f32" << std::endl;
          g_dense_paged_inited = false;
          return -1;
        }
        probe.push_back(g_dense_k_pools[i]);
        probe.push_back(g_dense_v_pools[i]);
        probe.push_back(g_dense_k_scales[i]);
        probe.push_back(g_dense_v_scales[i]);
      }
      // Break the lazy RNG split chain from model initialization, and
      // force-eval the pool / scale layout in the same batch so any
      // Metal-allocation or layout error throws here.
      auto rng_key = mlx::core::random::KeySequence::default_().next();
      probe.push_back(rng_key);
      mlx::core::eval(std::move(probe));
    }

    g_dense_paged_inited = true;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_qwen35_init_paged: " << e.what() << std::endl;
    g_dense_paged_inited = false;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] mlx_qwen35_init_paged: unknown exception" << std::endl;
    g_dense_paged_inited = false;
    return -1;
  }
}

// Single-token paged decode step. Inputs (PagedAttentionInputs) come from
// the Rust adapter's `build_paged_attention_inputs`; the per-layer
// pool/scale globals come from `mlx_qwen35_init_paged`.
//
// PHASE 5 PIECE 1 CONTRACT: This FFI is decode-only — `input_ids` MUST
// have exactly one element and `slot_mapping` MUST be `[1]`. Chunked
// prefill (multi-token) is reserved for later phases. The contract is
// enforced explicitly: violating it returns null logits without
// modifying global state, so the Rust caller can fall back cleanly.
//
// `output_logits` receives a heap-allocated `mlx_array*` (caller owns).
// `cache_offset_out` receives the post-step offset (== prefill_offset
// + 1 + n after n successful calls).
void mlx_qwen35_forward_paged(
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
  if (!g_dense_paged_inited) {
    if (output_logits) *output_logits = nullptr;
    return;
  }
  if (!input_ids_ptr || !embedding_weight_ptr || !output_logits ||
      !offset_arr_ptr || !block_table_ptr || !slot_mapping_ptr ||
      !num_valid_tokens_ptr || !num_valid_blocks_ptr || !seq_lens_ptr) {
    if (output_logits) *output_logits = nullptr;
    return;
  }
  const auto& cfg = g_dense_paged_config;

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);
    auto& offset_arr       = *reinterpret_cast<array*>(offset_arr_ptr);
    auto& block_table      = *reinterpret_cast<array*>(block_table_ptr);
    auto& slot_mapping     = *reinterpret_cast<array*>(slot_mapping_ptr);
    auto& num_valid_tokens = *reinterpret_cast<array*>(num_valid_tokens_ptr);
    auto& num_valid_blocks = *reinterpret_cast<array*>(num_valid_blocks_ptr);
    auto& seq_lens         = *reinterpret_cast<array*>(seq_lens_ptr);

    // Phase 5 piece 1 contract: single-token decode only.
    //
    // `attn_for_compile_paged` builds new_k / new_v with shape
    // `[1, num_kv_heads, head_size]` and feeds `slot_mapping` directly
    // into `paged_kv_write`, which requires
    // `slot_mapping.shape(0) == new_k.shape(0)`. Multi-token (B > 1)
    // is reserved for later phases.
    if (input_ids.size() != 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_forward_paged: phase 5 piece 1 contract "
              "violated — input_ids.size() = %lld, expected 1 (decode-only)\n",
              static_cast<long long>(input_ids.size()));
      fflush(stderr);
      *output_logits = nullptr;
      return;
    }
    if (slot_mapping.ndim() != 1 || slot_mapping.shape(0) != 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_forward_paged: phase 5 piece 1 contract "
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
      bool is_linear = dense_paged_is_linear_layer(i);
      if (is_linear) {
        fn_inputs.push_back(g_dense_paged_linear_caches[i * 2]);
        fn_inputs.push_back(g_dense_paged_linear_caches[i * 2 + 1]);
        fn_inputs.push_back(g_dense_k_scales[i]);   // unused placeholder
        fn_inputs.push_back(g_dense_v_scales[i]);   // unused placeholder
      } else {
        fn_inputs.push_back(g_dense_k_pools[i]);
        fn_inputs.push_back(g_dense_v_pools[i]);
        fn_inputs.push_back(g_dense_k_scales[i]);
        fn_inputs.push_back(g_dense_v_scales[i]);
      }
    }

    static bool no_compile = std::getenv("MLX_NO_COMPILE") != nullptr;
    auto outputs = no_compile
        ? dense_compiled_decode_fn_paged(fn_inputs)
        : compiled_dense_decode_paged()(fn_inputs);

    // Extract: [logits, new_offset, new_caches...]
    *output_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    g_dense_paged_offset_int++;
    // Stash post-step caches back into the per-layer slots.
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = dense_paged_is_linear_layer(i);
      auto& a = outputs[2 + i * 2];
      auto& b = outputs[2 + i * 2 + 1];
      if (is_linear) {
        g_dense_paged_linear_caches[i * 2]     = a;
        g_dense_paged_linear_caches[i * 2 + 1] = b;
      } else {
        g_dense_k_pools[i] = a;
        g_dense_v_pools[i] = b;
      }
    }

    if (cache_offset_out) {
      *cache_offset_out = g_dense_paged_offset_int;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_forward_paged: %s\n", e.what());
    fflush(stderr);
    *output_logits = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_forward_paged\n");
    fflush(stderr);
    *output_logits = nullptr;
  }
}

}  // extern "C"
