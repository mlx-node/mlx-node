#include "mlx_lfm2_common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <vector>

using namespace lfm2_common;

// Weight-registry FFI (defined in mlx_common_weights.cpp). Used by the
// component-parity probes below to register a single layer's weights into the
// shared `g_weights()` map before running the compiled pure-fns.
extern "C" {
void mlx_store_weight(const char* name, mlx_array* weight);
void mlx_clear_weights();
size_t mlx_weight_count();
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
// Model-id ownership is the SHARED g_active_model_id atom, read via
// qwen35_common::g_active_model_id() in mlx_lfm2_get_model_id below and published
// by mlx_set_model_id during registration (Phase 2b-2). lfm2 keeps NO private id:
// a separate one over the shared g_weights() map would collide with a co-resident
// qwen3.5 model (see the QWEN35_MODEL_ID_COUNTER invariant in lfm2/model.rs).

// Decode-graph config consumed by `lfm2_decode_fn`. Set by the caller (the
// 2b-1 probe; 2b-2 `init_from_prefill`) before invoking the loop. `g_lfm2_config`
// is the POD shape; `g_lfm2_is_attn` is the per-layer dispatch (1=attn, 0=conv),
// length num_layers — kept OUT of the POD (which must stay copyable per cfg).
lfm2_common::Lfm2MoeConfig g_lfm2_config;
std::vector<int> g_lfm2_is_attn;

// =====================================================================
// Full single-token decode loop over the dense lfm2 backbone, assembled from
// the parity-proven pure-fns (lfm2_attn_pure_fn_arr / lfm2_conv_pure_fn /
// lfm2_dense_mlp). Mirrors the qwen35 `moe_compiled_decode_fn` SHAPE: uniform
// 2N input/output cache stride so the compile-cache key is invariant.
//
//   inputs:  [h([B,hidden]), offset_arr(scalar i32),
//             slot[0].a, slot[0].b, ..., slot[N-1].a, slot[N-1].b]
//   outputs: [logits([B,vocab]), new_offset, new_slot[0].a, ..., new_slot[N-1].b]
//
// Per layer (matching native decoder_layer.rs:151-171):
//   normed = rms_norm(h, operator_norm);  h += op(normed)            (residual 1)
//   ffn_in = rms_norm(h, ffn_norm);        h += dense_mlp(ffn_in)     (residual 2)
// then rms_norm(h, embedding_norm) and the tied `embed_tokens` LM head.
//
// Cache slots (uniform stride 2, indexed by ABSOLUTE layer idx):
//   attn layer i: slot.a = kv_keys, slot.b = kv_values
//                 [B, num_kv_heads, max_kv_len, head_dim]
//   conv layer i: slot.a = conv_state [B, l_cache-1, hidden];
//                 slot.b = UNUSED placeholder (pre-seeded scalar bf16 zero, left
//                 untouched — no input->output identity edge).
//
// INVARIANT: every attention KV cache is padded to the SAME max_kv_len; the
// additive decode mask (positions <= offset -> 0, else -inf) is derived from the
// first attention layer's key cache. (2b-1 calls this EAGERLY via the probe;
// 2b-2 wraps it in mlx::core::compile.)
// =====================================================================
std::vector<array> lfm2_decode_fn(const std::vector<array>& inputs) {
  using namespace lfm2_common;
  using namespace qwen35_common;
  const auto& cfg = g_lfm2_config;
  auto h = inputs[0];           // [B, hidden]
  auto offset_arr = inputs[1];  // scalar int32

  // Static additive mask [1,1,1,max_kv_len] from the first attention layer.
  int first_attn = -1;
  for (int i = 0; i < cfg.num_layers; i++) {
    if (g_lfm2_is_attn[i]) {
      first_attn = i;
      break;
    }
  }
  int max_kv_len = (first_attn >= 0) ? inputs[2 + first_attn * 2].shape(2) : 1;
  auto positions = arange(0, max_kv_len, mlx::core::int32);
  auto valid = less_equal(positions, offset_arr);
  auto attn_mask = reshape(
      where(valid, array(0.0f, mlx::core::bfloat16),
            array(-std::numeric_limits<float>::infinity(), mlx::core::bfloat16)),
      {1, 1, 1, max_kv_len});

  // Pre-seed all output cache slots (conv slot.b stays this scalar zero).
  std::vector<array> new_caches;
  new_caches.reserve(cfg.num_layers * 2);
  for (int i = 0; i < cfg.num_layers * 2; i++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  for (int i = 0; i < cfg.num_layers; i++) {
    std::string lp = "layers." + std::to_string(i);

    // (1) operator_norm BEFORE the op, residual after.
    auto normed =
        mlx::core::fast::rms_norm(h, get_weight(lp + ".operator_norm.weight"), cfg.norm_eps);
    if (g_lfm2_is_attn[i]) {
      const auto& kk = inputs[2 + i * 2];
      const auto& kv = inputs[2 + i * 2 + 1];
      auto res = lfm2_attn_pure_fn_arr(normed, i, kk, kv, attn_mask, offset_arr, cfg);
      h = h + res.output;
      new_caches[i * 2] = res.keys;
      new_caches[i * 2 + 1] = res.values;
    } else {
      const auto& cs = inputs[2 + i * 2];
      auto res = lfm2_conv_pure_fn(normed, i, cs, cfg.conv_l_cache, cfg.hidden_size,
                                   /*conv_bias=*/false);
      h = h + res.output;
      new_caches[i * 2] = res.new_state;
      // slot.b left as the pre-seeded scalar zero (unused for conv layers).
    }

    // (2) ffn_norm BEFORE the FFN, residual after (EVERY layer).
    //
    // DENSE-FFN ONLY (Phase 2 scope). Every layer routes through
    // `lfm2_dense_mlp`, which is CORRECT for the dense `lfm2` backbone
    // (LFM2.5-1.2B: all layers are dense SwiGLU). It is WRONG for `lfm2_moe`,
    // whose layers >= num_dense_layers are `Lfm2SparseMoeBlock` (router +
    // expert_bias + top-k + switch_mlp). Adding that sparse dispatch here is
    // Phase 3. Until it lands, the Phase-2b-2 gate flip MUST gate on the dense
    // model only (`!config.is_moe()`) so an lfm2_moe checkpoint can never
    // silently take this dense-only path and compute the wrong FFN.
    auto ffn_in =
        mlx::core::fast::rms_norm(h, get_weight(lp + ".ffn_norm.weight"), cfg.norm_eps);
    h = h + lfm2_dense_mlp(ffn_in, i);
  }

  // Final norm + tied LM head (linear_proj appends ".weight"; tie reads
  // embed_tokens.weight via get_weight_t, untied reads lm_head.weight).
  h = mlx::core::fast::rms_norm(h, get_weight("embedding_norm.weight"), cfg.norm_eps);
  h = cfg.tie_embedding ? linear_proj(h, "embed_tokens") : linear_proj(h, "lm_head");

  auto new_offset = offset_arr + array(1, mlx::core::int32);
  std::vector<array> out;
  out.reserve(2 + cfg.num_layers * 2);
  out.push_back(h);
  out.push_back(new_offset);
  for (auto& c : new_caches) {
    out.push_back(c);
  }
  return out;
}

// =====================================================================
// Production decode state (2b-2 Stage B/C). Mirrors the qwen35-MoE
// flat-path globals (`g_moe_caches` / `g_moe_offset_int` / `g_moe_inited`).
//
//   g_lfm2_caches      live cache vector, uniform stride 2 by ABSOLUTE layer
//                      idx. attn layer i -> (kv_keys, kv_values) padded to
//                      max_kv_len; conv layer i -> (conv_state, scalar bf16
//                      zero placeholder). Threaded across decode steps.
//   g_lfm2_offset_int  current decode position (next write slot in KV).
//   g_lfm2_inited      true iff init_from_prefill imported caches cleanly.
//   g_lfm2_forward_calls  cumulative forward count (engagement signal; NOT
//                      reset by mlx_lfm2_moe_reset).
// =====================================================================
std::vector<array> g_lfm2_caches;
int g_lfm2_offset_int = 0;
bool g_lfm2_inited = false;
uint64_t g_lfm2_forward_calls = 0;

// Compiled wrapper around lfm2_decode_fn — compiled once, reused per step so
// the compile-cache key stays stable (input shapes are fixed at init time).
static auto& compiled_lfm2_decode() {
  static auto fn = mlx::core::compile(lfm2_decode_fn);
  return fn;
}
}  // namespace

extern "C" {

// GATE source. Returns 0 in Phase 0 (no setter wired) → compiled path OFF.
uint64_t mlx_lfm2_get_model_id() {
  return qwen35_common::g_active_model_id().load(std::memory_order_acquire);
}

// Shared weight count (the lfm2 compiled path owns the SAME g_weights() map).
size_t mlx_lfm2_weight_count() { return mlx_weight_count(); }

// Build + seed the compiled decode graph from post-prefill state.
//
// `is_attn` (length num_layers, 1=attn/0=conv) drives the per-layer dispatch
// and is built dynamically Rust-side from config.is_attention_layer; it is
// NEVER a modulo/hardcoded pattern (lfm2 mixes conv/attn irregularly).
//
// Cache import (uniform stride 2 by ABSOLUTE layer idx, matching the
// lfm2_decode_fn input contract):
//   attn layer i: import cache_arrays[i*2]/[i*2+1] as K/V, PADDED to max_kv_len
//                 via concatenate (mirrors qwen35_moe init); null on either ->
//                 g_lfm2_inited=false, bail. The decode mask is derived from
//                 the FIRST attention layer's padded KV, so this slot MUST be a
//                 real [B,nkv,max_kv_len,head_dim] tensor.
//   conv layer i: import cache_arrays[i*2] as conv_state [B,l_cache-1,hidden];
//                 push a scalar bf16 zero for slot.b. The conv branch never
//                 reads cache_arrays[i*2+1] (Rust passes null there).
void mlx_lfm2_moe_init_from_prefill(
    int num_layers,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rope_theta,
    float norm_eps,
    int conv_l_cache,
    int num_experts,
    int num_experts_per_tok,
    int num_dense_layers,
    int norm_topk_prob,
    int use_expert_bias,
    int tie_embedding,
    int max_kv_len,
    int batch_size,
    const int32_t* is_attn,
    mlx_array** cache_arrays,
    int prefill_offset) {
  try {
    g_lfm2_config = Lfm2MoeConfig{};
    g_lfm2_config.num_layers = num_layers;
    g_lfm2_config.hidden_size = hidden_size;
    g_lfm2_config.num_heads = num_heads;
    g_lfm2_config.num_kv_heads = num_kv_heads;
    g_lfm2_config.head_dim = head_dim;
    g_lfm2_config.rope_theta = rope_theta;
    g_lfm2_config.norm_eps = norm_eps;
    g_lfm2_config.conv_l_cache = conv_l_cache;
    g_lfm2_config.num_experts = num_experts;
    g_lfm2_config.num_experts_per_tok = num_experts_per_tok;
    g_lfm2_config.num_dense_layers = num_dense_layers;
    g_lfm2_config.norm_topk_prob = norm_topk_prob != 0;
    g_lfm2_config.use_expert_bias = use_expert_bias != 0;
    g_lfm2_config.tie_embedding = tie_embedding != 0;
    g_lfm2_config.max_kv_len = max_kv_len;
    g_lfm2_config.batch_size = batch_size;

    // NOTE: Lfm2MoeConfig has NO rope_dims — RoPE is over the full head_dim.

    g_lfm2_is_attn.assign(is_attn, is_attn + num_layers);

    g_lfm2_caches.clear();
    g_lfm2_caches.reserve(num_layers * 2);
    g_lfm2_inited = false;

    for (int i = 0; i < num_layers; i++) {
      if (is_attn[i]) {
        if (!cache_arrays[i * 2] || !cache_arrays[i * 2 + 1]) {
          g_lfm2_caches.clear();
          return;
        }
        auto& kk = *reinterpret_cast<array*>(cache_arrays[i * 2]);
        auto& kv = *reinterpret_cast<array*>(cache_arrays[i * 2 + 1]);
        int current_cap = kk.shape(2);
        if (current_cap < max_kv_len) {
          int pad_len = max_kv_len - current_cap;
          auto kpad = zeros({batch_size, num_kv_heads, pad_len, head_dim}, kk.dtype());
          auto vpad = zeros({batch_size, num_kv_heads, pad_len, head_dim}, kv.dtype());
          g_lfm2_caches.push_back(concatenate({kk, kpad}, 2));
          g_lfm2_caches.push_back(concatenate({kv, vpad}, 2));
        } else {
          g_lfm2_caches.push_back(kk);
          g_lfm2_caches.push_back(kv);
        }
      } else {
        // Conv layer: only slot.a (conv_state) is read. slot.b is an unused
        // scalar placeholder (NEVER reads cache_arrays[i*2+1]).
        if (!cache_arrays[i * 2]) {
          g_lfm2_caches.clear();
          return;
        }
        auto& cs = *reinterpret_cast<array*>(cache_arrays[i * 2]);
        g_lfm2_caches.push_back(cs);
        g_lfm2_caches.push_back(zeros({}, mlx::core::bfloat16));
      }
    }

    g_lfm2_offset_int = prefill_offset;
    g_lfm2_inited = true;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] mlx_lfm2_moe_init_from_prefill: %s\n", e.what());
    fflush(stderr);
    g_lfm2_caches.clear();
    g_lfm2_inited = false;
  } catch (...) {
    g_lfm2_caches.clear();
    g_lfm2_inited = false;
  }
}

// Single-token compiled decode step. Writes a null *output_logits when the
// graph is not initialized (or on error) so the caller falls back to native.
void mlx_lfm2_moe_forward(
    mlx_array* input_ids,
    mlx_array* embedding_weight,
    mlx_array** output_logits,
    int* cache_offset_out) {
  if (!g_lfm2_inited) {
    if (output_logits) {
      *output_logits = nullptr;
    }
    return;
  }

  try {
    g_lfm2_forward_calls++;

    auto& ids = *reinterpret_cast<array*>(input_ids);
    auto& embedding = *reinterpret_cast<array*>(embedding_weight);

    // Embedding lookup: [B,1] -> [B, hidden] (2D, matching lfm2_decode_fn h).
    auto flat_ids = reshape(ids, {-1});
    auto h = take(embedding, flat_ids, 0);

    std::vector<array> fn_inputs;
    fn_inputs.reserve(2 + g_lfm2_caches.size());
    fn_inputs.push_back(std::move(h));
    fn_inputs.push_back(array(g_lfm2_offset_int, mlx::core::int32));
    for (const auto& c : g_lfm2_caches) {
      fn_inputs.push_back(c);
    }

    // MLX_NO_COMPILE=1 disables compilation for A/B testing.
    static bool no_compile = std::getenv("MLX_NO_COMPILE") != nullptr;
    auto outputs = no_compile ? lfm2_decode_fn(fn_inputs) : compiled_lfm2_decode()(fn_inputs);

    if (output_logits) {
      *output_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    }
    g_lfm2_offset_int++;
    for (int i = 0; i < g_lfm2_config.num_layers * 2; i++) {
      g_lfm2_caches[i] = outputs[2 + i];
    }
    if (cache_offset_out) {
      *cache_offset_out = g_lfm2_offset_int;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_lfm2_moe_forward: %s\n", e.what());
    fflush(stderr);
    if (output_logits) {
      *output_logits = nullptr;
    }
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_lfm2_moe_forward\n");
    fflush(stderr);
    if (output_logits) {
      *output_logits = nullptr;
    }
  }
}

// Async-eval the sampled token (+ caches implicitly via the compiled graph's
// dependency edges). MLX_EVAL_ALL_CACHES=1 evals token + every live cache
// explicitly (slower; for debugging). Mirrors mlx_qwen35_moe_eval_token_and_caches.
void mlx_lfm2_moe_eval_token_and_caches(mlx_array* token) {
  try {
    static bool eval_all = std::getenv("MLX_EVAL_ALL_CACHES") != nullptr;
    if (eval_all) {
      std::vector<array> to_eval;
      to_eval.reserve(1 + g_lfm2_caches.size());
      to_eval.push_back(*reinterpret_cast<array*>(token));
      for (const auto& c : g_lfm2_caches) {
        to_eval.push_back(c);
      }
      mlx::core::async_eval(std::move(to_eval));
    } else {
      mlx::core::async_eval({*reinterpret_cast<array*>(token)});
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_lfm2_moe_eval_token_and_caches: %s\n", e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_lfm2_moe_eval_token_and_caches\n");
    fflush(stderr);
  }
}

// Cumulative engagement counter. Intentionally NOT reset by
// mlx_lfm2_moe_reset — it is a process-lifetime "did the compiled decode path
// ever run" signal for the e2e assertion.
uint64_t mlx_lfm2_moe_forward_call_count() { return g_lfm2_forward_calls; }

// Export the live caches for cross-turn reuse. Copies cache arrays to caller-
// provided output pointers (heap-allocated). Returns the number exported (the
// uniform stride-2 vector, including conv scalar placeholders), or 0 if not
// initialized. MLX arrays are ref-counted so the underlying Metal buffer is
// shared, not duplicated. Mirrors mlx_qwen35_moe_export_caches.
int mlx_lfm2_moe_export_caches(mlx_array** out_ptrs, int max_count) {
  if (!g_lfm2_inited || g_lfm2_caches.empty()) {
    return 0;
  }
  int count = std::min(static_cast<int>(g_lfm2_caches.size()), max_count);
  for (int i = 0; i < count; i++) {
    out_ptrs[i] = reinterpret_cast<mlx_array*>(new array(g_lfm2_caches[i]));
  }
  return count;
}

// Current decode offset (number of cached tokens after the last forward).
int mlx_lfm2_moe_get_cache_offset() { return g_lfm2_offset_int; }

// Whether init_from_prefill seeded the decode graph cleanly. The Rust caller
// checks this after seeding because init is `void` but can bail internally
// (null cache slot, or a padding/concatenate exception) — letting Rust fall
// back to the native path instead of treating the first null forward as fatal.
int mlx_lfm2_moe_is_initialized() { return g_lfm2_inited ? 1 : 0; }

// Tear down the decode state. Does NOT touch the shared model-id atom
// (mlx_clear_weights owns it) and does NOT reset g_lfm2_forward_calls.
void mlx_lfm2_moe_reset() {
  g_lfm2_caches.clear();
  g_lfm2_offset_int = 0;
  g_lfm2_inited = false;
}

// =============================================================================
// Component-parity probes (TEST-ONLY). These register ONE layer's weights into
// the shared weight map, run the compiled pure-fn, and return the output so a
// Rust test can compare it to the native Rust-side forward.
//
// CALLER CONTRACT — these are DESTRUCTIVE on the process-global `g_weights()`
// registry: each does `mlx_clear_weights -> store -> run -> mlx_clear_weights`,
// and `mlx_clear_weights` ALSO resets the active model id. That registry is the
// SAME one the production compiled paths (qwen3.5 / qwen3.5-MoE / gemma4, and
// eventually lfm2) own during registration + inference, guarded process-wide by
// the Rust `COMPILED_WEIGHTS_RWLOCK`. A probe call that overlaps a live compiled
// registration/inference would wipe its weights mid-flight. So every caller MUST
// hold `COMPILED_WEIGHTS_RWLOCK` (write) across the whole probe call — the Rust
// parity tests do exactly this. Do NOT call these from any production path; they
// exist solely for the component-parity gate (the full compiled forward is not
// end-to-end runnable until the backbone lands in Phase 2+).
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

// Run a SEQUENCE of `T` lfm2 ShortConv decode steps (B=1, offset 0..T-1)
// through `lfm2_conv_pure_fn`, threading the conv state ([1, l_cache-1, hidden],
// zeros init), and return the LAST step's output `[1, hidden]`. Running a
// sequence (not a single step) is what exercises the causal conv window and the
// state carry-over: on step 0 the state is all-zeros, so only step >=1 mixes
// real history through the depthwise kernel.
//
// `x_seq` is `[T, hidden]` (one decode input per row). Linear weights are
// natural `[out, in]` (in_proj [3H,H], out_proj [H,H]); the depthwise conv
// weight is MLX-layout `[hidden, l_cache, 1]` (3D, NOT transposed) and is stored
// under the DOUBLED key `layers.0.conv.conv.weight` (block prefix `conv.` +
// nn.Conv1d submodule `conv`) to match the real checkpoint (persistence.rs:907)
// — do NOT collapse it to a single `conv.weight`. Biases (iff conv_bias != 0)
// are in_proj `[3H]`, conv `[hidden]`, out_proj `[hidden]`; the bias pointers
// may be null when conv_bias == 0. Caller owns the returned array; null on error.
mlx_array* mlx_lfm2_probe_conv_seq(
    mlx_array* x_seq_ptr,
    mlx_array* in_proj_w, mlx_array* conv_w, mlx_array* out_proj_w,
    mlx_array* in_proj_b, mlx_array* conv_b, mlx_array* out_proj_b,
    int l_cache, int conv_bias) {
  try {
    mlx_clear_weights();
    mlx_store_weight("layers.0.conv.in_proj.weight", in_proj_w);
    mlx_store_weight("layers.0.conv.out_proj.weight", out_proj_w);
    mlx_store_weight("layers.0.conv.conv.weight", conv_w);  // [hidden, l_cache, 1]
    if (conv_bias) {
      mlx_store_weight("layers.0.conv.in_proj.bias", in_proj_b);
      mlx_store_weight("layers.0.conv.conv.bias", conv_b);
      mlx_store_weight("layers.0.conv.out_proj.bias", out_proj_b);
    }

    auto& x_seq = *reinterpret_cast<array*>(x_seq_ptr);
    int T = x_seq.shape(0);
    int hidden = x_seq.shape(1);

    // conv state slot: [B=1, l_cache-1, hidden], zeros, input dtype (bf16).
    auto state = zeros({1, l_cache - 1, hidden}, x_seq.dtype());

    array last_out = zeros({1, hidden}, x_seq.dtype());
    for (int i = 0; i < T; i++) {
      auto x_i = reshape(slice(x_seq, {i, 0}, {i + 1, hidden}), {1, hidden});
      auto res = lfm2_conv_pure_fn(x_i, /*layer_idx=*/0, state, l_cache, hidden,
                                   conv_bias != 0);
      state = res.new_state;
      last_out = res.output;
    }
    mlx::core::eval({last_out});
    auto* out = new array(last_out);
    mlx_clear_weights();
    return reinterpret_cast<mlx_array*>(out);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] mlx_lfm2_probe_conv_seq: %s\n", e.what());
    fflush(stderr);
    mlx_clear_weights();
    return nullptr;
  } catch (...) {
    mlx_clear_weights();
    return nullptr;
  }
}

// Run a SEQUENCE of `T` lfm2 attention decode steps through the ARRAY-OFFSET
// compiled variant `lfm2_attn_pure_fn_arr` (the one the decode loop will use):
// fixed-shape padded KV cache [1, num_kv_heads, T, head_dim] + a per-step static
// additive mask (positions <= offset -> 0, else -inf), array offset = step index.
// Returns the LAST step's output [1, num_heads*head_dim]. Gates the array variant
// (fixed-cache + mask + array RoPE/slice_update) independently of the scalar
// dynamic_kv path, BEFORE it is wired into lfm2_decode_fn.
//
// TEST-ONLY + DESTRUCTIVE on g_weights() — caller MUST hold COMPILED_WEIGHTS_RWLOCK
// (write); see the probe-section contract above. Weights natural [out,in] /
// [head_dim]. Caller owns the returned array; nullptr on error.
mlx_array* mlx_lfm2_probe_attn_arr_seq(
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
    auto positions = arange(0, T, mlx::core::int32);

    array last_out = zeros({1, num_heads * head_dim}, x_seq.dtype());
    for (int i = 0; i < T; i++) {
      auto x_i = reshape(slice(x_seq, {i, 0}, {i + 1, hidden}), {1, hidden});
      auto offset_arr = array(i, mlx::core::int32);
      // Static additive mask [1,1,1,T]: positions <= offset -> 0, else -inf.
      auto valid = less_equal(positions, offset_arr);
      auto mask = reshape(
          where(valid, array(0.0f, mlx::core::bfloat16),
                array(-std::numeric_limits<float>::infinity(), mlx::core::bfloat16)),
          {1, 1, 1, T});
      auto res = lfm2_attn_pure_fn_arr(x_i, 0, kv_keys, kv_values, mask, offset_arr, cfg);
      kv_keys = res.keys;
      kv_values = res.values;
      last_out = res.output;
    }
    mlx::core::eval({last_out});
    auto* out = new array(last_out);
    mlx_clear_weights();
    return reinterpret_cast<mlx_array*>(out);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] mlx_lfm2_probe_attn_arr_seq: %s\n", e.what());
    fflush(stderr);
    mlx_clear_weights();
    return nullptr;
  } catch (...) {
    mlx_clear_weights();
    return nullptr;
  }
}

// Run a SYNTHETIC small dense lfm2 model through the FULL `lfm2_decode_fn`
// assembly for `T` decode steps and return the LAST step's logits [1, vocab].
// This is the 2b-1 end-to-end-SHAPED parity gate: it exercises the per-layer
// conv-vs-attn dispatch (from is_attn[]), the operator_norm->op->+res->ffn_norm->
// mlp->+res order, the conv-state vs KV slot interleaving at uniform stride 2,
// the final embedding_norm, and the tied embed_tokens head — WITHOUT the real
// checkpoint and WITHOUT flipping the production gate (mlx_lfm2_get_model_id
// stays 0). `lfm2_decode_fn` is invoked EAGERLY (un-compiled) so a graph-trace
// bug cannot be masked; the compiled path is validated in 2b-2.
//
// Per-layer weights are passed as arrays-of-pointers indexed by layer; conv
// layers ignore the attn pointers and vice versa (read per is_attn[i]). The
// embedding table is stored under "embed_tokens.weight" (the tied head's
// linear_proj appends ".weight" and reads get_weight_t). Weights natural
// [out,in] / [head_dim]; conv weight [hidden,l_cache,1]. token_ids has length T.
//
// TEST-ONLY + DESTRUCTIVE on g_weights() — caller MUST hold COMPILED_WEIGHTS_RWLOCK
// (write); see the probe-section contract above. Caller owns the returned array;
// nullptr on error.
mlx_array* mlx_lfm2_probe_decode_seq(
    mlx_array* embed_w_ptr, mlx_array* emb_norm_ptr,
    const int* is_attn, int num_layers,
    int hidden, int num_heads, int num_kv_heads, int head_dim,
    int l_cache, float rope_theta, float norm_eps,
    const int* token_ids, int T,
    mlx_array** op_norm_w, mlx_array** ffn_norm_w,
    mlx_array** gate_w, mlx_array** up_w, mlx_array** down_w,
    mlx_array** q_w, mlx_array** k_w, mlx_array** v_w, mlx_array** out_w,
    mlx_array** qn_w, mlx_array** kn_w,
    mlx_array** in_proj_w, mlx_array** conv_w, mlx_array** out_proj_w) {
  try {
    mlx_clear_weights();
    auto& embed_w = *reinterpret_cast<array*>(embed_w_ptr);
    // Tied head: linear_proj(h,"embed_tokens") -> get_weight_t("embed_tokens.weight").
    mlx_store_weight("embed_tokens.weight", embed_w_ptr);
    mlx_store_weight("embedding_norm.weight", emb_norm_ptr);
    for (int i = 0; i < num_layers; i++) {
      std::string lp = "layers." + std::to_string(i);
      mlx_store_weight((lp + ".operator_norm.weight").c_str(), op_norm_w[i]);
      mlx_store_weight((lp + ".ffn_norm.weight").c_str(), ffn_norm_w[i]);
      mlx_store_weight((lp + ".feed_forward.gate_proj.weight").c_str(), gate_w[i]);
      mlx_store_weight((lp + ".feed_forward.up_proj.weight").c_str(), up_w[i]);
      mlx_store_weight((lp + ".feed_forward.down_proj.weight").c_str(), down_w[i]);
      if (is_attn[i]) {
        mlx_store_weight((lp + ".self_attn.q_proj.weight").c_str(), q_w[i]);
        mlx_store_weight((lp + ".self_attn.k_proj.weight").c_str(), k_w[i]);
        mlx_store_weight((lp + ".self_attn.v_proj.weight").c_str(), v_w[i]);
        mlx_store_weight((lp + ".self_attn.out_proj.weight").c_str(), out_w[i]);
        mlx_store_weight((lp + ".self_attn.q_layernorm.weight").c_str(), qn_w[i]);
        mlx_store_weight((lp + ".self_attn.k_layernorm.weight").c_str(), kn_w[i]);
      } else {
        mlx_store_weight((lp + ".conv.in_proj.weight").c_str(), in_proj_w[i]);
        mlx_store_weight((lp + ".conv.conv.weight").c_str(), conv_w[i]);  // [H,l_cache,1]
        mlx_store_weight((lp + ".conv.out_proj.weight").c_str(), out_proj_w[i]);
      }
    }

    g_lfm2_config = Lfm2MoeConfig{};
    g_lfm2_config.num_layers = num_layers;
    g_lfm2_config.hidden_size = hidden;
    g_lfm2_config.num_heads = num_heads;
    g_lfm2_config.num_kv_heads = num_kv_heads;
    g_lfm2_config.head_dim = head_dim;
    g_lfm2_config.conv_l_cache = l_cache;
    g_lfm2_config.rope_theta = rope_theta;
    g_lfm2_config.norm_eps = norm_eps;
    g_lfm2_config.tie_embedding = true;
    g_lfm2_config.max_kv_len = T;
    g_lfm2_is_attn.assign(is_attn, is_attn + num_layers);

    // Local cache vector (uniform stride 2). conv -> (state, scalar placeholder);
    // attn -> (kv_keys, kv_values) padded to T.
    std::vector<array> caches;
    caches.reserve(num_layers * 2);
    for (int i = 0; i < num_layers; i++) {
      if (is_attn[i]) {
        caches.push_back(zeros({1, num_kv_heads, T, head_dim}, mlx::core::bfloat16));
        caches.push_back(zeros({1, num_kv_heads, T, head_dim}, mlx::core::bfloat16));
      } else {
        caches.push_back(zeros({1, l_cache - 1, hidden}, mlx::core::bfloat16));
        caches.push_back(zeros({}, mlx::core::bfloat16));  // unused placeholder
      }
    }

    array last_logits = zeros({1, embed_w.shape(0)}, mlx::core::bfloat16);
    for (int t = 0; t < T; t++) {
      auto idx = reshape(array(token_ids[t], mlx::core::int32), {1});
      auto h = take(embed_w, idx, 0);  // [1, hidden]
      std::vector<array> in;
      in.reserve(2 + num_layers * 2);
      in.push_back(h);
      in.push_back(array(t, mlx::core::int32));  // offset = t
      for (auto& c : caches) {
        in.push_back(c);
      }
      auto outs = lfm2_decode_fn(in);  // EAGER (un-compiled)
      last_logits = outs[0];
      for (int i = 0; i < num_layers * 2; i++) {
        caches[i] = outs[2 + i];
      }
    }
    mlx::core::eval({last_logits});
    auto* out = new array(last_logits);
    mlx_clear_weights();
    return reinterpret_cast<mlx_array*>(out);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] mlx_lfm2_probe_decode_seq: %s\n", e.what());
    fflush(stderr);
    mlx_clear_weights();
    return nullptr;
  } catch (...) {
    mlx_clear_weights();
    return nullptr;
  }
}

}  // extern "C"
