#include "mlx_qwen35_common.h"

using namespace qwen35_common;

// =============================================================================
// Qwen3.5 Dense Compiled Forward Pass
//
// Implements the entire Qwen3.5 dense model forward pass (single-token decode)
// in one FFI call. Weight storage, helpers, GDN and attention functions are
// shared with the MoE path via mlx_qwen35_common.h.
// =============================================================================

namespace {

// Config for the compiled path (extends BaseConfig with compile-specific state)
struct CompileConfig : BaseConfig {};

static CompileConfig g_compile_config{};
static std::vector<array> g_compiled_caches;         // num_layers * 2 arrays
static std::optional<array> g_compiled_offset;       // scalar int32 (current decode position)
static int g_offset_int = 0;                         // C++ int mirror of g_compiled_offset
static bool g_compile_inited = false;

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
    auto gate    = matmul(mlp_in, get_weight_t(mp + "gate_proj.weight"));
    auto up      = matmul(mlp_in, get_weight_t(mp + "up_proj.weight"));
    auto mlp_out = matmul(swiglu(gate, up), get_weight_t(mp + "down_proj.weight"));
    h = h + mlp_out;
  }

  // Final norm + LM head
  h = fast::rms_norm(h, get_weight("final_norm.weight"), cfg.rms_norm_eps);
  if (cfg.tie_word_embeddings) {
    h = matmul(h, get_weight_t("embedding.weight"));
  } else {
    h = matmul(h, get_weight_t("lm_head.weight"));
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

} // namespace

// =============================================================================
// Public FFI functions
// =============================================================================

extern "C" {

void mlx_qwen35_store_weight(const char* name, mlx_array* weight) {
  std::lock_guard<std::mutex> lock(g_weights_mutex());
  auto& arr = *reinterpret_cast<array*>(weight);
  g_weights().insert_or_assign(std::string(name), arr);
}

void mlx_qwen35_clear_weights() {
  std::lock_guard<std::mutex> lock(g_weights_mutex());
  g_weights().clear();
  g_weight_transposes().clear();
}

size_t mlx_qwen35_weight_count() {
  std::lock_guard<std::mutex> lock(g_weights_mutex());
  return g_weights().size();
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

void mlx_qwen35_compiled_reset() {
  g_compiled_caches.clear();
  g_compiled_offset = std::nullopt;
  g_offset_int = 0;
  g_compile_inited = false;
}

}  // extern "C"
