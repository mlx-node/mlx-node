#include "mlx_qwen35_common.h"

using namespace qwen35_common;

// =============================================================================
// Qwen3.5 Dense TEXT Prefill
//
// Text-only sibling of mlx_qwen35_vlm_prefill: takes pre-embedded inputs
// (Rust caller does the embedding lookup), runs the full forward in one FFI
// call using gdn_prefill_fn + attn_prefill_fn_scalar, returns the last-token
// logits. Per-layer caches are stored in g_text_caches for transfer to the
// compiled decode path via mlx_qwen35_compiled_init_from_prefill.
// =============================================================================

namespace {

struct TextPrefillConfig : BaseConfig {};

static TextPrefillConfig g_text_config{};
static std::vector<array> g_text_caches;
static int g_text_offset = 0;
static bool g_text_inited = false;

}  // namespace

extern "C" {

// Runs the full Qwen3.5-dense text prefill in one FFI call.
//
// Inputs:
//   inputs_embeds_ptr: [B, T, hidden_size] bf16 — Rust caller does the
//                      embedding lookup before this call.
//   max_kv_len:        pre-allocation size for the K/V cache (must be
//                      >= T so the per-layer caches can be padded).
//   batch_size:        currently expected to be 1.
//
// Output:
//   output_logits: [B, vocab] containing the last-token logits.
//                  Caller takes ownership of the returned pointer.
//
// Cache state is held in g_text_caches for transfer to the compiled decode path.
void mlx_qwen35_text_prefill(
    mlx_array* inputs_embeds_ptr,
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
    mlx_array** output_logits
) {
  if (!inputs_embeds_ptr || !output_logits) {
    if (output_logits) *output_logits = nullptr;
    return;
  }

  // Last-token extraction below assumes B=1. h_flat is laid out batch-major,
  // and slicing rows [(B*T)-1, B*T) only grabs the last token of the FINAL
  // batch element, silently dropping batches 0..B-2. Reject B != 1 loudly
  // until a per-batch gather is wired in.
  if (batch_size != 1) {
    fprintf(stderr, "[MLX] mlx_qwen35_text_prefill: batch_size=%d not supported (only B=1)\n", batch_size);
    fflush(stderr);
    *output_logits = nullptr;
    return;
  }

  try {
    auto& inputs_embeds = *reinterpret_cast<array*>(inputs_embeds_ptr);

    g_text_config = TextPrefillConfig{{
      num_layers, hidden_size, num_heads, num_kv_heads, head_dim,
      rope_theta, rope_dims, rms_norm_eps, full_attention_interval,
      linear_num_k_heads, linear_num_v_heads, linear_key_head_dim,
      linear_value_head_dim, linear_conv_kernel_dim,
      tie_word_embeddings != 0,
      max_kv_len, batch_size
    }};

    const auto& cfg = g_text_config;
    int B = inputs_embeds.shape(0);
    int T = inputs_embeds.shape(1);

    auto h = inputs_embeds;

    g_text_caches.clear();
    g_text_caches.reserve(num_layers * 2);
    for (int i = 0; i < num_layers * 2; i++) {
      g_text_caches.push_back(zeros({}, mlx::core::bfloat16));
    }

    for (int i = 0; i < num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      std::string lp = "layers." + std::to_string(i);

      auto h_flat = reshape(h, {B * T, hidden_size});
      auto normed = reshape(
          fast::rms_norm(h_flat, get_weight(lp + ".input_layernorm.weight"), cfg.rms_norm_eps),
          {B, T, hidden_size});

      array layer_out = zeros({}, mlx::core::bfloat16);

      if (is_linear) {
        auto res = gdn_prefill_fn(normed, i, cfg);
        layer_out = std::move(res.output);
        g_text_caches[i * 2]     = std::move(res.conv_state);
        g_text_caches[i * 2 + 1] = std::move(res.recurrent_state);
      } else {
        // Scalar-offset RoPE; first chunk so offset=0.
        auto res = attn_prefill_fn_scalar(normed, i, 0, cfg);
        layer_out = std::move(res.output);
        if (T < max_kv_len) {
          int pad_len = max_kv_len - T;
          auto kpad = zeros({B, num_kv_heads, pad_len, head_dim}, res.keys.dtype());
          auto vpad = zeros({B, num_kv_heads, pad_len, head_dim}, res.values.dtype());
          g_text_caches[i * 2]     = concatenate({res.keys, kpad}, 2);
          g_text_caches[i * 2 + 1] = concatenate({res.values, vpad}, 2);
        } else {
          g_text_caches[i * 2]     = std::move(res.keys);
          g_text_caches[i * 2 + 1] = std::move(res.values);
        }
      }

      h = h + layer_out;

      // SwiGLU MLP
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

    int vocab = h_flat.shape(1);
    auto logits = slice(h_flat, {(B * T) - 1, 0}, {B * T, vocab});
    logits = reshape(logits, {1, vocab});

    *output_logits = reinterpret_cast<mlx_array*>(new array(logits));

    g_text_offset = T;
    g_text_inited = true;

    // Eval logits + all caches so subsequent FFI calls (e.g. cache transfer)
    // see materialized arrays, not a dangling lazy graph rooted in the
    // (now-released) inputs.
    {
      std::vector<array> to_eval;
      to_eval.reserve(g_text_caches.size() + 1);
      to_eval.push_back(logits);
      for (auto& c : g_text_caches) to_eval.push_back(c);
      mlx::core::eval(to_eval);
    }

    // Break the lazy RNG split chain (matches VLM path).
    auto rng_key = mlx::core::random::KeySequence::default_().next();
    mlx::core::eval({rng_key});

  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_text_prefill: %s\n", e.what());
    fflush(stderr);
    if (output_logits) *output_logits = nullptr;
    g_text_inited = false;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_text_prefill\n");
    fflush(stderr);
    if (output_logits) *output_logits = nullptr;
    g_text_inited = false;
  }
}

int mlx_qwen35_text_cache_count() {
  return static_cast<int>(g_text_caches.size());
}

mlx_array* mlx_qwen35_text_get_cache(int index) {
  if (index < 0 || index >= static_cast<int>(g_text_caches.size())) return nullptr;
  return reinterpret_cast<mlx_array*>(new array(g_text_caches[index]));
}

int mlx_qwen35_text_get_offset() {
  return g_text_offset;
}

void mlx_qwen35_text_reset() {
  g_text_caches.clear();
  g_text_offset = 0;
  g_text_inited = false;
}

}  // extern "C"
