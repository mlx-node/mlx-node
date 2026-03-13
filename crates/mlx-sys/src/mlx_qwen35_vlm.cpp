#include "mlx_qwen35_common.h"

using namespace qwen35_common;

// =============================================================================
// Qwen3.5 VLM Prefill
//
// Runs the full language model forward pass on a sequence of embeddings
// (text + merged vision features) with M-RoPE position IDs.
// After prefill, caches are stored and accessible via mlx_qwen35_vlm_get_cache
// for transfer to the existing compiled decode path.
// =============================================================================

namespace {

struct VLMConfig : BaseConfig {
  std::vector<int> mrope_section;  // [11, 11, 10] for Qwen3.5-VL
};

static VLMConfig g_vlm_config{};
static std::vector<array> g_vlm_caches;
static int g_vlm_offset = 0;
static bool g_vlm_inited = false;

}  // namespace

extern "C" {

void mlx_qwen35_vlm_prefill(
    mlx_array* inputs_embeds_ptr,
    mlx_array* position_ids_ptr,
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
    const int* mrope_section_ptr,
    int rope_deltas,
    mlx_array** output_logits
) {
  if (!inputs_embeds_ptr || !position_ids_ptr || !output_logits) {
    if (output_logits) *output_logits = nullptr;
    return;
  }

  try {
    auto& inputs_embeds = *reinterpret_cast<array*>(inputs_embeds_ptr);
    auto& position_ids  = *reinterpret_cast<array*>(position_ids_ptr);

    g_vlm_config = VLMConfig{{
      num_layers, hidden_size, num_heads, num_kv_heads, head_dim,
      rope_theta, rope_dims, rms_norm_eps, full_attention_interval,
      linear_num_k_heads, linear_num_v_heads, linear_key_head_dim,
      linear_value_head_dim, linear_conv_kernel_dim,
      tie_word_embeddings != 0,
      max_kv_len, batch_size
    }, {mrope_section_ptr[0], mrope_section_ptr[1], mrope_section_ptr[2]}};

    const auto& cfg = g_vlm_config;
    int B = inputs_embeds.shape(0);
    int T = inputs_embeds.shape(1);

    auto h = inputs_embeds;

    g_vlm_caches.clear();
    g_vlm_caches.reserve(num_layers * 2);
    for (int i = 0; i < num_layers * 2; i++) {
      g_vlm_caches.push_back(zeros({}, mlx::core::bfloat16));
    }

    for (int i = 0; i < num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      std::string lp = "layers." + std::to_string(i);

      // Pre-attention norm: flatten to [B*T, hidden] for rms_norm, reshape back
      auto h_flat = reshape(h, {B * T, hidden_size});
      auto normed = reshape(
          fast::rms_norm(h_flat, get_weight(lp + ".input_layernorm.weight"), cfg.rms_norm_eps),
          {B, T, hidden_size});

      array layer_out = zeros({}, mlx::core::bfloat16);

      if (is_linear) {
        auto res = gdn_prefill_fn(normed, i, cfg);
        layer_out = std::move(res.output);
        g_vlm_caches[i * 2]     = std::move(res.conv_state);
        g_vlm_caches[i * 2 + 1] = std::move(res.recurrent_state);
      } else {
        auto res = attn_prefill_fn(normed, i, position_ids, cfg, cfg.mrope_section);
        layer_out = std::move(res.output);
        if (T < max_kv_len) {
          int pad_len = max_kv_len - T;
          auto kpad = zeros({B, num_kv_heads, pad_len, head_dim}, res.keys.dtype());
          auto vpad = zeros({B, num_kv_heads, pad_len, head_dim}, res.values.dtype());
          g_vlm_caches[i * 2]     = concatenate({res.keys, kpad}, 2);
          g_vlm_caches[i * 2 + 1] = concatenate({res.values, vpad}, 2);
        } else {
          g_vlm_caches[i * 2]     = std::move(res.keys);
          g_vlm_caches[i * 2 + 1] = std::move(res.values);
        }
      }

      h = h + layer_out;

      // Post-attention MLP (SwiGLU)
      std::string mp = lp + ".mlp.";
      auto mlp_flat = reshape(h, {B * T, hidden_size});
      auto mlp_in = fast::rms_norm(mlp_flat, get_weight(lp + ".post_attention_layernorm.weight"), cfg.rms_norm_eps);
      auto gate    = matmul(mlp_in, get_weight_t(mp + "gate_proj.weight"));
      auto up      = matmul(mlp_in, get_weight_t(mp + "up_proj.weight"));
      auto mlp_out = reshape(
          matmul(swiglu(gate, up), get_weight_t(mp + "down_proj.weight")),
          {B, T, hidden_size});
      h = h + mlp_out;
    }

    // Final norm + LM head
    auto h_flat = reshape(h, {B * T, hidden_size});
    h_flat = fast::rms_norm(h_flat, get_weight("final_norm.weight"), cfg.rms_norm_eps);
    if (cfg.tie_word_embeddings) {
      h_flat = matmul(h_flat, get_weight_t("embedding.weight"));
    } else {
      h_flat = matmul(h_flat, get_weight_t("lm_head.weight"));
    }

    // Last-token logits: [1, vocab]
    int vocab = h_flat.shape(1);
    auto logits = slice(h_flat, {(B * T) - 1, 0}, {B * T, vocab});
    logits = reshape(logits, {1, vocab});

    *output_logits = reinterpret_cast<mlx_array*>(new array(logits));

    // Set offset for decode
    g_vlm_offset = T;
    g_vlm_inited = true;

    // Eval logits + all caches. Without this, caches remain as lazy graph nodes
    // from the entire 32-layer prefill. The decode path would then try to eval
    // the full prefill graph during the first decode step, causing Metal exceptions.
    // This matches Python mlx-lm: mx.eval([c.state for c in prompt_cache])
    {
      std::vector<array> to_eval;
      to_eval.reserve(g_vlm_caches.size() + 1);
      to_eval.push_back(logits);
      for (auto& c : g_vlm_caches) {
        to_eval.push_back(c);
      }
      mlx::core::eval(to_eval);
    }

    // Break the lazy RNG split chain from model initialization.
    auto rng_key = mlx::core::random::KeySequence::default_().next();
    mlx::core::eval({rng_key});

  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_vlm_prefill: %s\n", e.what());
    fflush(stderr);
    if (output_logits) *output_logits = nullptr;
    g_vlm_inited = false;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_vlm_prefill\n");
    fflush(stderr);
    if (output_logits) *output_logits = nullptr;
    g_vlm_inited = false;
  }
}

int mlx_qwen35_vlm_cache_count() {
  return static_cast<int>(g_vlm_caches.size());
}

mlx_array* mlx_qwen35_vlm_get_cache(int index) {
  if (index < 0 || index >= static_cast<int>(g_vlm_caches.size())) return nullptr;
  return reinterpret_cast<mlx_array*>(&g_vlm_caches[index]);
}

int mlx_qwen35_vlm_get_offset() {
  return g_vlm_offset;
}

void mlx_qwen35_vlm_reset() {
  g_vlm_caches.clear();
  g_vlm_offset = 0;
  g_vlm_inited = false;
}

}  // extern "C"
