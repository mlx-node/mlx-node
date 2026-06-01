#include "mlx_lfm2_common.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
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
                                   /*conv_bias=*/cfg.conv_bias);
      h = h + res.output;
      new_caches[i * 2] = res.new_state;
      // slot.b left as the pre-seeded scalar zero (unused for conv layers).
    }

    // (2) ffn_norm BEFORE the FFN, residual after (EVERY layer).
    //
    // Per-layer FFN dispatch (Phase 3a): a layer is a MoE layer iff the model has
    // experts (num_experts > 0) AND its index is >= num_dense_layers; otherwise it
    // is dense SwiGLU. A pure-dense lfm2 checkpoint has num_experts == 0, so EVERY
    // layer takes the dense path and behavior is UNCHANGED from Phase 2 (the
    // num_dense_layers default of 0 is irrelevant when num_experts == 0). The MoE
    // arm runs the sparse routing: router softmax + selection-only expert_bias +
    // top-k + switch_mlp SwiGLU + weighted sum (lfm2_moe_ffn). The Phase-2b-2 gate
    // flip still gates on `!config.is_moe()`; the production MoE gate is lifted in
    // Phase 3c.
    auto ffn_in =
        mlx::core::fast::rms_norm(h, get_weight(lp + ".ffn_norm.weight"), cfg.norm_eps);
    bool is_moe_layer = cfg.num_experts > 0 && i >= cfg.num_dense_layers;
    if (is_moe_layer) {
      h = h + lfm2_moe_ffn(ffn_in, i, cfg);
    } else {
      h = h + lfm2_dense_mlp(ffn_in, i);
    }
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
//   g_lfm2_compiled_decode_calls  cumulative count of forwards that took the
//                      TRACED compiled_lfm2_decode() branch (i.e. NOT the eager
//                      MLX_NO_COMPILE arm). Process-lifetime "did the traced
//                      compiled branch run" signal; NOT reset by
//                      mlx_lfm2_moe_reset.
// =====================================================================
std::vector<array> g_lfm2_caches;
int g_lfm2_offset_int = 0;
bool g_lfm2_inited = false;
uint64_t g_lfm2_forward_calls = 0;
uint64_t g_lfm2_compiled_decode_calls = 0;

// ---------------------------------------------------------------------------
// REBUILDABLE compiled-decode closure (Phase 3c hardening).
//
// The compiled `lfm2_decode_fn` graph captures the weight CONSTANTS resolved by
// `get_weight()` at TRACE time. When a NEW lfm2 model registers its weights
// (`register_weights_with_cpp` -> `mlx_clear_weights` + `mlx_store_weight` +
// `mlx_set_model_id`), the previously-compiled closure still references model
// A's frozen constants — so reusing it for model B would silently decode A's
// weights. The model-id gate prevents B from running A's *id*, but a second
// lfm2 model that re-takes the compiled path with the same shapes would reuse
// the stale graph. We therefore make the closure REBUILDABLE and key its
// validity on a monotonically-increasing epoch bumped on every (re)registration
// (and on weight clear); a mismatch forces a recompile that re-captures the
// live constants.
//
// THREAD-SAFETY CONTRACT (the [HIGH] finding):
//   * `g_lfm2_compile_epoch` is an atomic publish counter; ANY thread can bump
//     it via `mlx_lfm2_invalidate_compiled()` (called under the registration
//     write lock).
//   * The epoch-check + rebuild touches the NON-atomic `g_lfm2_compiled` /
//     `g_lfm2_compiled_epoch_built`, so it is guarded by a dedicated
//     `g_lfm2_compiled_mu`. Without it a concurrent invalidate+rebuild would
//     race the read of the optional.
//   * `compiled_lfm2_decode()` returns the std::function BY VALUE (a copy) — NOT
//     a reference into `g_lfm2_compiled`. A later invalidate that reassigns the
//     optional destroys the stored function; a caller holding a reference would
//     then dangle (UB) the moment it invoked it. The std::function copy is cheap
//     (mlx's compiled callable is a shared-handle wrapper, not a graph clone) and
//     negligible next to the per-step forward compute, and it lets the caller own
//     a stable callable that a concurrent swap can never invalidate mid-call.
// ---------------------------------------------------------------------------
std::mutex g_lfm2_compiled_mu;
std::optional<std::function<std::vector<array>(const std::vector<array>&)>> g_lfm2_compiled;
uint64_t g_lfm2_compiled_epoch_built = 0;
std::uintptr_t g_lfm2_compiled_fun_id_built = 0;  // 0 == none installed
bool g_lfm2_has_compiled_fun_id = false;
std::atomic<uint64_t> g_lfm2_compile_epoch{0};

// lfm2-LOCAL fun_id base: the address of a dedicated static. Used to derive a
// UNIQUE compile id per epoch (see compiled_lfm2_decode below).
inline std::uintptr_t lfm2_fun_id_base() {
  static char anchor = 0;
  return reinterpret_cast<std::uintptr_t>(&anchor);
}

// Returns a stable, owned COPY of the compiled decode closure, RE-TRACING it
// under a fresh per-epoch `fun_id` if the registration epoch advanced since it
// was last built.
//
// CRITICAL MLX DETAIL: the public `mlx::core::compile(fun)` keys its trace cache
// on `fun`'s target ADDRESS. `lfm2_decode_fn` is always the same function, so
// re-calling `compile(lfm2_decode_fn)` returns a wrapper that, on the same input
// shapes, REPLAYS the first cached trace — which captured model A's weight
// CONSTANTS at `get_weight()` time. (Verified: an A->B swap that merely
// re-wrapped via the public `compile` kept B's compiled output ≈ A's, not B's
// eager.) We therefore use the lower-level
// `mlx::core::detail::compile(fun, fun_id, shapeless, constants)` with a UNIQUE
// per-epoch `fun_id`, so every registration epoch gets a FRESH trace keyed on a
// distinct id — re-capturing the live constants — and we `detail::compile_erase`
// the prior epoch's id so the global compile cache does not grow per load.
//
// fun_id scheme: `lfm2_fun_id_base() ^ ((epoch << 1) | 1)`. The base is a
// data-segment address unique to this TU (the public API's function-pointer
// `fun_id`s and qwen's own `compile()` ids never collide with it); the `<<1 | 1`
// keeps every epoch's id distinct and odd. This is lfm2-LOCAL — it does not
// touch qwen3.5's independent compiled path.
//
// THREAD-SAFETY (the [HIGH] finding): `g_lfm2_compiled_mu` guards the whole
// epoch-check + (re)compile + erase (all touch the non-atomic optional / id
// state and the MLX compile cache); we return BY VALUE so a concurrent
// invalidate that reassigns the optional cannot dangle the caller's handle.
std::function<std::vector<array>(const std::vector<array>&)> compiled_lfm2_decode() {
  std::lock_guard<std::mutex> lk(g_lfm2_compiled_mu);
  uint64_t cur = g_lfm2_compile_epoch.load(std::memory_order_acquire);
  if (!g_lfm2_compiled.has_value() || g_lfm2_compiled_epoch_built != cur) {
    if (g_lfm2_has_compiled_fun_id) {
      mlx::core::detail::compile_erase(g_lfm2_compiled_fun_id_built);
    }
    std::uintptr_t fun_id = lfm2_fun_id_base() ^ static_cast<std::uintptr_t>((cur << 1) | 1ull);
    g_lfm2_compiled = mlx::core::detail::compile(lfm2_decode_fn, fun_id);
    g_lfm2_compiled_epoch_built = cur;
    g_lfm2_compiled_fun_id_built = fun_id;
    g_lfm2_has_compiled_fun_id = true;
  }
  return *g_lfm2_compiled;
}

// TEST-ONLY: drop the cached compiled closure WITHOUT advancing the compile
// epoch, forcing the next compiled_lfm2_decode() at the CURRENT epoch to
// re-trace against the currently-registered weights. Used by the no-bump warm
// probe so it deterministically freezes the just-built synthetic model rather
// than reusing whatever closure a prior same-epoch probe left cached. Guards the
// non-atomic closure state with g_lfm2_compiled_mu, exactly like
// compiled_lfm2_decode().
//
// CALLER CONTRACT: the target weights MUST already be stored in g_weights()
// before calling this — the re-trace happens on the NEXT compiled_lfm2_decode(),
// which captures whatever constants are live then. Calling this before storing
// the intended weights would re-trace against stale/empty constants.
void lfm2_reset_compiled_closure_same_epoch() {
  std::lock_guard<std::mutex> lk(g_lfm2_compiled_mu);
  if (g_lfm2_has_compiled_fun_id) {
    mlx::core::detail::compile_erase(g_lfm2_compiled_fun_id_built);
    g_lfm2_has_compiled_fun_id = false;
  }
  g_lfm2_compiled.reset();
  g_lfm2_compiled_epoch_built = 0;
}
// =============================================================================
// TEST-ONLY shared synthetic-MoE helpers (used by the compiled-vs-eager probe
// AND the no-bump warm probe). Extracted from `mlx_lfm2_probe_moe_compiled_vs_eager`
// so the warm probe can reuse the IDENTICAL build + decode without copy-paste.
// =============================================================================

// Fixed synthetic-MoE topology shared by both probes (3 layers, hidden 32,
// E=32/k=4, num_dense_layers=1, T=8, is_attn={1,0,1}). Returned to the caller
// so the decode helper can use it for the input lookup; the config dims are
// also published into `g_lfm2_config` / `g_lfm2_is_attn`. Clears + re-stores
// the weight map. `well_separated` controls the expert_bias spread (big gaps
// => decisive routing; tiny gaps => near-tie). Does NOT bump the compile epoch.
struct Lfm2SyntheticMoe {
  array embed;
  int num_layers;
  int hidden;
  int num_kv_heads;
  int head_dim;
  int l_cache;
  int vocab;
  int T;
};

Lfm2SyntheticMoe lfm2_build_synthetic_moe(uint64_t seed, int well_separated) {
  // ---- fixed synthetic config ----
  const int num_layers = 3;
  const int hidden = 32;
  const int num_heads = 4;
  const int num_kv_heads = 2;
  const int head_dim = 8;
  const int l_cache = 4;
  // E=32 / k=4 matches the real lfm2.5-8b-a1b routing fan-out, so the near-tie
  // case has the SAME number of top-k boundary candidates as the 8B model
  // (the regime where a single fused-FP selection flip can occur).
  const int E = 32;
  const int k = 4;
  const int num_dense_layers = 1;
  const int vocab = 48;
  const int T = 8;
  const float rope_theta = 10000.0f;
  const float norm_eps = 1e-5f;
  const int is_attn[3] = {1, 0, 1};  // attn, conv, attn
  const int moe_inter = 24;          // per-expert SwiGLU hidden
  const int dense_inter = 40;        // dense-layer SwiGLU hidden

  mlx_clear_weights();

  // ---- seeded xorshift -> [-1,1) ----
  uint64_t s = seed ? seed : 0x10F23C0Deull;
  auto next = [&]() -> float {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return (static_cast<float>(s >> 40) / static_cast<float>(1u << 23)) - 1.0f;
  };
  auto mk = [&](std::vector<int> shape, float scale) -> array {
    int nelem = 1;
    for (int d : shape) {
      nelem *= d;
    }
    std::vector<float> buf(static_cast<size_t>(nelem));
    for (int i = 0; i < nelem; i++) {
      buf[i] = next() * scale;
    }
    mlx::core::Shape sh(shape.begin(), shape.end());
    return astype(array(buf.data(), sh, mlx::core::float32), mlx::core::bfloat16);
  };

  // ---- register weights (mlx_store_weight copies; free our temp wrapper) ----
  auto store = [&](const std::string& name, const array& a) {
    auto* p = new array(a);
    mlx_store_weight(name.c_str(), reinterpret_cast<mlx_array*>(p));
    delete p;
  };
  auto store_norm = [&](const std::string& name, std::vector<int> shape) {
    // norm weights centered at 1.0 (RMSNorm gain).
    store(name, mk(shape, 0.02f) + array(1.0f, mlx::core::bfloat16));
  };

  auto embed = mk({vocab, hidden}, 0.05f);
  store("embed_tokens.weight", embed);
  store_norm("embedding_norm.weight", {hidden});

  for (int i = 0; i < num_layers; i++) {
    std::string lp = "layers." + std::to_string(i);
    store_norm(lp + ".operator_norm.weight", {hidden});
    store_norm(lp + ".ffn_norm.weight", {hidden});

    bool is_moe = i >= num_dense_layers;  // E>0 always here
    if (is_moe) {
      store(lp + ".feed_forward.gate.weight", mk({E, hidden}, 0.1f));
      // expert_bias is the selection lever (see header).
      std::vector<float> bias(E);
      for (int e = 0; e < E; e++) {
        bias[e] = well_separated ? 4.0f * static_cast<float>(e)
                                 : 1e-4f * static_cast<float>(e);
      }
      mlx::core::Shape bsh{E};
      store(lp + ".feed_forward.expert_bias",
            astype(array(bias.data(), bsh, mlx::core::float32), mlx::core::bfloat16));
      store(lp + ".feed_forward.switch_mlp.gate_proj.weight", mk({E, moe_inter, hidden}, 0.08f));
      store(lp + ".feed_forward.switch_mlp.up_proj.weight", mk({E, moe_inter, hidden}, 0.08f));
      store(lp + ".feed_forward.switch_mlp.down_proj.weight", mk({E, hidden, moe_inter}, 0.08f));
    } else {
      store(lp + ".feed_forward.gate_proj.weight", mk({dense_inter, hidden}, 0.08f));
      store(lp + ".feed_forward.up_proj.weight", mk({dense_inter, hidden}, 0.08f));
      store(lp + ".feed_forward.down_proj.weight", mk({hidden, dense_inter}, 0.08f));
    }

    if (is_attn[i]) {
      store(lp + ".self_attn.q_proj.weight", mk({num_heads * head_dim, hidden}, 0.08f));
      store(lp + ".self_attn.k_proj.weight", mk({num_kv_heads * head_dim, hidden}, 0.08f));
      store(lp + ".self_attn.v_proj.weight", mk({num_kv_heads * head_dim, hidden}, 0.08f));
      store(lp + ".self_attn.out_proj.weight", mk({hidden, num_heads * head_dim}, 0.08f));
      store_norm(lp + ".self_attn.q_layernorm.weight", {head_dim});
      store_norm(lp + ".self_attn.k_layernorm.weight", {head_dim});
    } else {
      store(lp + ".conv.in_proj.weight", mk({3 * hidden, hidden}, 0.08f));
      store(lp + ".conv.conv.weight", mk({hidden, l_cache, 1}, 0.2f));
      store(lp + ".conv.out_proj.weight", mk({hidden, hidden}, 0.08f));
    }
  }

  // ---- config ----
  g_lfm2_config = Lfm2MoeConfig{};
  g_lfm2_config.num_layers = num_layers;
  g_lfm2_config.hidden_size = hidden;
  g_lfm2_config.num_heads = num_heads;
  g_lfm2_config.num_kv_heads = num_kv_heads;
  g_lfm2_config.head_dim = head_dim;
  g_lfm2_config.conv_l_cache = l_cache;
  g_lfm2_config.rope_theta = rope_theta;
  g_lfm2_config.norm_eps = norm_eps;
  g_lfm2_config.num_experts = E;
  g_lfm2_config.num_experts_per_tok = k;
  g_lfm2_config.num_dense_layers = num_dense_layers;
  g_lfm2_config.norm_topk_prob = true;
  g_lfm2_config.use_expert_bias = true;
  g_lfm2_config.use_sigmoid = false;
  g_lfm2_config.tie_embedding = true;
  g_lfm2_config.max_kv_len = T;
  g_lfm2_is_attn.assign(is_attn, is_attn + num_layers);

  return Lfm2SyntheticMoe{embed, num_layers, hidden, num_kv_heads, head_dim,
                          l_cache, vocab, T};
}

// Drive T decode steps over the synthetic MoE built above (CURRENT
// g_weights()/g_lfm2_config state), eager or compiled, returning the last-step
// logits. `compiled` selects compiled_lfm2_decode() vs the eager lfm2_decode_fn.
array lfm2_run_synthetic_decode(const Lfm2SyntheticMoe& m, bool compiled) {
  const int num_layers = m.num_layers;
  const int hidden = m.hidden;
  const int num_kv_heads = m.num_kv_heads;
  const int head_dim = m.head_dim;
  const int l_cache = m.l_cache;
  const int vocab = m.vocab;
  const int T = m.T;
  const int token_ids[8] = {3, 11, 7, 22, 5, 19, 31, 14};
  const auto& embed = m.embed;

  std::vector<array> caches;
  caches.reserve(num_layers * 2);
  for (int i = 0; i < num_layers; i++) {
    if (g_lfm2_is_attn[i]) {
      caches.push_back(zeros({1, num_kv_heads, T, head_dim}, mlx::core::bfloat16));
      caches.push_back(zeros({1, num_kv_heads, T, head_dim}, mlx::core::bfloat16));
    } else {
      caches.push_back(zeros({1, l_cache - 1, hidden}, mlx::core::bfloat16));
      caches.push_back(zeros({}, mlx::core::bfloat16));
    }
  }
  array last_logits = zeros({1, vocab}, mlx::core::bfloat16);
  for (int t = 0; t < T; t++) {
    auto idx = reshape(array(token_ids[t], mlx::core::int32), {1});
    auto h = take(embed, idx, 0);  // [1, hidden]
    std::vector<array> in;
    in.reserve(2 + num_layers * 2);
    in.push_back(h);
    in.push_back(array(t, mlx::core::int32));
    for (auto& c : caches) {
      in.push_back(c);
    }
    std::vector<array> outs;
    if (compiled) {
      // Owned copy of the by-value compiled closure (see contract) before
      // invoking — no dangling reference if a swap races.
      auto fn = compiled_lfm2_decode();
      outs = fn(in);
    } else {
      outs = lfm2_decode_fn(in);
    }
    last_logits = outs[0];
    for (int i = 0; i < num_layers * 2; i++) {
      caches[i] = outs[2 + i];
    }
  }
  mlx::core::eval({last_logits});
  return last_logits;
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
    int conv_bias,
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
    g_lfm2_config.conv_bias = conv_bias != 0;
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
    // Take an OWNED copy of the compiled closure into a local before invoking it.
    // `compiled_lfm2_decode()` returns by value (see its contract), so a
    // concurrent invalidate+recompile that reassigns `g_lfm2_compiled` cannot
    // dangle this handle mid-call. (When no_compile is set we skip it entirely.)
    std::vector<array> outputs;
    if (no_compile) {
      outputs = lfm2_decode_fn(fn_inputs);
    } else {
      auto compiled = compiled_lfm2_decode();
      // Bump the TRACED-branch counter ONLY here (never in the no_compile arm
      // above and never before the branch) — this is the proof that the actual
      // compiled closure ran, not just that the forward FFI was entered.
      g_lfm2_compiled_decode_calls++;
      outputs = compiled(fn_inputs);
    }

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

// Cumulative count of forwards that took the TRACED compiled_lfm2_decode()
// branch (NOT the eager MLX_NO_COMPILE arm). Distinguishes "the forward FFI was
// entered" from "the compiled closure actually ran". Like the count above, it is
// intentionally NOT reset by mlx_lfm2_moe_reset.
uint64_t mlx_lfm2_moe_compiled_decode_call_count() {
  return g_lfm2_compiled_decode_calls;
}

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

// Invalidate the cached compiled-decode closure so the NEXT
// `compiled_lfm2_decode()` recompiles `lfm2_decode_fn`, re-capturing the live
// weight constants from `g_weights()`. Called by `register_weights_with_cpp`
// under `COMPILED_WEIGHTS_RWLOCK.write()` (just before `mlx_set_model_id`), so a
// freshly-loaded lfm2 model can never reuse a prior model's frozen graph.
//
// The bump is a release fetch_add on an atomic; the rebuild that consumes it
// (in `compiled_lfm2_decode()`) acquire-loads under `g_lfm2_compiled_mu`, so the
// epoch publish is ordered ahead of any decode that observes it.
//
// FREED-CONSTANT NOTE ([LOW] finding): `mlx_clear_weights` (mlx_common_weights
// .cpp) frees the weight constants AND resets the active model id to 0, but does
// NOT itself bump this epoch. That is SAFE: after a bare clear, the active model
// id is 0, so `compiled_path_active()` / `mlx_lfm2_get_model_id()` no longer
// match any live `Lfm2Inner::model_id` and decode falls back to native — no
// decode ever runs against the (now-freed) captured constants. A subsequent
// (re)registration calls `mlx_clear_weights` THEN stores new weights THEN
// `mlx_lfm2_invalidate_compiled()`, so by the time the gate re-opens (model id
// republished) the epoch has advanced and the next decode recompiles against the
// live constants. The model-id gate is the primary defense; the epoch bump is
// what makes a SAME-SHAPE A->B swap (which keeps passing the gate under B's id)
// re-capture B's weights instead of reusing A's frozen graph.
void mlx_lfm2_invalidate_compiled() {
  g_lfm2_compile_epoch.fetch_add(1, std::memory_order_acq_rel);
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
    mlx_array** in_proj_w, mlx_array** conv_w, mlx_array** out_proj_w,
    int conv_bias, mlx_array** in_proj_b_w, mlx_array** conv_b_w,
    mlx_array** out_proj_b_w) {
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
        // Phase 4 Piece 1: conv biases under the SAME keys lfm2_conv_pure_fn's
        // get_weight reads (conv.in_proj.bias / conv.conv.bias /
        // conv.out_proj.bias). Only when conv_bias is on, so the conv_bias==0
        // probe call is byte-identical to before.
        if (conv_bias) {
          mlx_store_weight((lp + ".conv.in_proj.bias").c_str(), in_proj_b_w[i]);
          mlx_store_weight((lp + ".conv.conv.bias").c_str(), conv_b_w[i]);
          mlx_store_weight((lp + ".conv.out_proj.bias").c_str(), out_proj_b_w[i]);
        }
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
    g_lfm2_config.conv_bias = conv_bias != 0;
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

// Run a SYNTHETIC small MoE lfm2 model through the FULL `lfm2_decode_fn`
// assembly for `T` decode steps and return the LAST step's logits [1, vocab].
// This is the Phase-3a end-to-end-SHAPED MoE parity gate: it exercises the
// per-layer dense-vs-MoE FFN dispatch (layers >= num_dense_layers route through
// `lfm2_moe_ffn`: router softmax + selection-only expert_bias + top-k +
// switch_mlp SwiGLU + weighted sum), threaded through the same conv/attn
// backbone as the dense probe — WITHOUT the real checkpoint and WITHOUT flipping
// the production gate (mlx_lfm2_get_model_id stays 0). `lfm2_decode_fn` runs
// EAGERLY (un-compiled) so a graph-trace bug cannot be masked.
//
// Layout: dense layers (idx < num_dense_layers OR num_experts == 0) use the
// per-layer gate_w/up_w/down_w arrays; MoE layers use the stacked
// moe_gate_proj/moe_up_proj/moe_down_proj ([E,out,in]) + router moe_router_w
// ([E,hidden]) + moe_bias ([E]). Irrelevant arrays are null per layer (read per
// the is-MoE / is_attn predicate). bf16 experts only (quant_mode = 0).
//
// TEST-ONLY + DESTRUCTIVE on g_weights() — caller MUST hold COMPILED_WEIGHTS_RWLOCK
// (write); see the dense probe contract above. Caller owns the returned array;
// nullptr on error.
mlx_array* mlx_lfm2_probe_moe_decode_seq(
    mlx_array* embed_w_ptr, mlx_array* emb_norm_ptr,
    const int* is_attn, int num_layers,
    int hidden, int num_heads, int num_kv_heads, int head_dim,
    int l_cache, float rope_theta, float norm_eps,
    int num_experts, int num_experts_per_tok, int num_dense_layers,
    int norm_topk_prob, int use_expert_bias, int use_sigmoid,
    const int* token_ids, int T,
    mlx_array** op_norm_w, mlx_array** ffn_norm_w,
    mlx_array** gate_w, mlx_array** up_w, mlx_array** down_w,
    mlx_array** q_w, mlx_array** k_w, mlx_array** v_w, mlx_array** out_w,
    mlx_array** qn_w, mlx_array** kn_w,
    mlx_array** in_proj_w, mlx_array** conv_w, mlx_array** out_proj_w,
    mlx_array** moe_router_w, mlx_array** moe_bias,
    mlx_array** moe_gate_proj, mlx_array** moe_up_proj, mlx_array** moe_down_proj) {
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

      bool is_moe = num_experts > 0 && i >= num_dense_layers;
      if (is_moe) {
        // Router gate [E, hidden] + (optional) expert_bias [E] + stacked experts.
        mlx_store_weight((lp + ".feed_forward.gate.weight").c_str(), moe_router_w[i]);
        if (use_expert_bias) {
          mlx_store_weight((lp + ".feed_forward.expert_bias").c_str(), moe_bias[i]);
        }
        mlx_store_weight((lp + ".feed_forward.switch_mlp.gate_proj.weight").c_str(),
                         moe_gate_proj[i]);
        mlx_store_weight((lp + ".feed_forward.switch_mlp.up_proj.weight").c_str(),
                         moe_up_proj[i]);
        mlx_store_weight((lp + ".feed_forward.switch_mlp.down_proj.weight").c_str(),
                         moe_down_proj[i]);
      } else {
        mlx_store_weight((lp + ".feed_forward.gate_proj.weight").c_str(), gate_w[i]);
        mlx_store_weight((lp + ".feed_forward.up_proj.weight").c_str(), up_w[i]);
        mlx_store_weight((lp + ".feed_forward.down_proj.weight").c_str(), down_w[i]);
      }

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
    g_lfm2_config.num_experts = num_experts;
    g_lfm2_config.num_experts_per_tok = num_experts_per_tok;
    g_lfm2_config.num_dense_layers = num_dense_layers;
    g_lfm2_config.norm_topk_prob = norm_topk_prob != 0;
    g_lfm2_config.use_expert_bias = use_expert_bias != 0;
    g_lfm2_config.use_sigmoid = use_sigmoid != 0;
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
    fprintf(stderr, "[MLX] mlx_lfm2_probe_moe_decode_seq: %s\n", e.what());
    fflush(stderr);
    mlx_clear_weights();
    return nullptr;
  } catch (...) {
    mlx_clear_weights();
    return nullptr;
  }
}

// =============================================================================
// DECISIVE H1/H2 PROBE (TEST-ONLY).
//
// Builds a FIXED 3-layer synthetic MoE lfm2 model entirely in C++ (seeded), runs
// the SAME `lfm2_decode_fn` BOTH eagerly AND through the process-global
// `compiled_lfm2_decode()` over T decode steps, and writes the last-step logit
// max-abs(compiled - eager) into *out_maxabs. The ONLY variable between the two
// runs is mlx::core::compile — so a nonzero diff isolates a compile effect on
// the MoE FFN path.
//
// Router selection is driven ENTIRELY by `expert_bias` (added post-softmax in
// `lfm2_moe_ffn`). softmax(routing) lies in [0,1], so:
//   well_separated != 0 : expert_bias[e] = 4.0f * e  -> gaps of 4.0 dominate the
//                         <1.0 softmax spread, the top-k is the bias ranking,
//                         input-independent, NO near-ties (the H2-negative case:
//                         compiled==eager here => compile() selects correctly).
//   well_separated == 0 : expert_bias[e] = 1e-4f * e (tiny gaps) -> selection is
//                         decided by softmax(routing) near-ties => FP-fusion
//                         sensitive (the H2-positive case: a flip here positively
//                         confirms near-tie selection sensitivity is the
//                         mechanism, NOT a structural bug).
//
// The config is FIXED (3 layers, hidden 32, E=4, k=2, dense_layers=1, T=6) so the
// compiled static topology (input count 2+2*3=8) is stable per process. Caller
// MUST hold COMPILED_WEIGHTS_RWLOCK (write); DESTRUCTIVE on g_weights().
// Returns 0 on success (out_maxabs written), -1 on error.
int mlx_lfm2_probe_moe_compiled_vs_eager(uint64_t seed, int well_separated,
                                         float* out_maxabs) {
  try {
    auto m = lfm2_build_synthetic_moe(seed, well_separated);

    // Order-independence: a prior probe in this process may have left a compiled
    // closure cached at the current epoch; bump so this probe's compiled run
    // re-traces against THESE synthetic constants (mirrors register_weights_with_cpp).
    mlx_lfm2_invalidate_compiled();

    auto eager = lfm2_run_synthetic_decode(m, false);
    auto comp = lfm2_run_synthetic_decode(m, true);
    auto diff = max(abs(subtract(astype(comp, mlx::core::float32),
                                 astype(eager, mlx::core::float32))));
    mlx::core::eval({diff});
    if (out_maxabs) {
      *out_maxabs = diff.item<float>();
    }
    mlx_clear_weights();
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] mlx_lfm2_probe_moe_compiled_vs_eager: %s\n", e.what());
    fflush(stderr);
    mlx_clear_weights();
    return -1;
  } catch (...) {
    mlx_clear_weights();
    return -1;
  }
}

// TEST-ONLY: warm a compiled lfm2 decode closure for the fixed synthetic MoE
// built from `seed` (well_separated=1): build weights/config, drop any closure a
// prior same-epoch probe left (lfm2_reset_compiled_closure_same_epoch), run ONE
// compiled decode over the T steps so a closure is traced + cached at the CURRENT
// epoch against THESE `seed` constants, then clear weights WITHOUT bumping the
// compile epoch. The same-epoch reset is what makes the cached stale closure
// DETERMINISTICALLY `seed`'s model (not whatever a prior probe happened to leave),
// so the A->B swap test can exercise the MODEL-A epoch-bump fix. Caller MUST hold
// COMPILED_WEIGHTS_RWLOCK (write); DESTRUCTIVE on g_weights(). Returns 0 on
// success, -1 on error.
int mlx_lfm2_probe_warm_compiled_no_bump(uint64_t seed) {
  try {
    auto m = lfm2_build_synthetic_moe(seed, /*well_separated=*/1);
    // Deliberately NO mlx_lfm2_invalidate_compiled() here: the whole point of
    // this probe is to leave a compiled closure cached at the current epoch so
    // the A->B swap test's MODEL-A bump has a stale closure to defeat.
    //
    // Force this probe's OWN constants to be the closure cached at the current
    // epoch: drop any closure a prior same-epoch probe left, so the compiled
    // decode below re-traces against THESE freshly-built `seed` weights. Without
    // this, compiled_lfm2_decode() would reuse a stale same-epoch closure and the
    // warm probe would NOT deterministically freeze `seed` (the A->B regression
    // would then be non-load-bearing for reasons unrelated to the MODEL-A bump).
    lfm2_reset_compiled_closure_same_epoch();
    auto comp = lfm2_run_synthetic_decode(m, /*compiled=*/true);
    mlx::core::eval({comp});
    mlx_clear_weights();
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] mlx_lfm2_probe_warm_compiled_no_bump: %s\n", e.what());
    fflush(stderr);
    mlx_clear_weights();
    return -1;
  } catch (...) {
    mlx_clear_weights();
    return -1;
  }
}

// =============================================================================
// A->B MODEL-SWAP REGRESSION PROBE (TEST-ONLY) — locks in the epoch-invalidation
// fix for the stale-compiled-closure (frozen weight-constant) hazard.
//
// Builds two DISTINCT synthetic MoE models A and B with the SAME fixed topology
// (so the compiled decode graph has identical input shapes — exactly the case
// where a stale closure would be silently reused) but DIFFERENT weights (seeded
// from seed_a vs seed_b). In ONE process:
//   1. register A, run it through `compiled_lfm2_decode()`  -> a_comp logits
//      (this TRACES + caches the compiled graph against A's frozen constants).
//   2. register B (clear + store B's weights) and bump the compile epoch via
//      `mlx_lfm2_invalidate_compiled()` — mirroring `register_weights_with_cpp`,
//      which bumps the epoch under the write lock before publishing the id.
//   3. run B through `compiled_lfm2_decode()` -> b_comp, AND run B EAGERLY
//      (`lfm2_decode_fn`, never compiled) -> b_eager.
//
// Writes three max-abs diffs:
//   *out_b_comp_vs_b_eager : b_comp vs b_eager. With the epoch bump this is
//        ~0 (compiled recompiled against B's live constants). WITHOUT the bump
//        the cached closure replays A's frozen constants, so b_comp == a_comp
//        and this diff blows up — that is the regression the test catches.
//   *out_b_comp_vs_a_comp  : b_comp vs a_comp. Proves A and B are genuinely
//        different models — if this were ~0 the test couldn't distinguish a
//        stale-graph reuse from two coincidentally-equal models.
//   *out_a_comp_vs_a_eager : a_comp vs a_eager sanity (compile is faithful for A
//        too; rules out a pre-existing compile bug masking the result).
//
// Both A and B use the WELL-SEPARATED router (bias gaps of 4.0) so neither model
// has near-tie selection jitter — any b_comp-vs-b_eager divergence is the
// stale-closure bug, NOT a fused-FP top-k flip. Caller MUST hold
// COMPILED_WEIGHTS_RWLOCK (write); DESTRUCTIVE on g_weights(). Returns 0 on
// success, -1 on error.
int mlx_lfm2_probe_moe_ab_swap(uint64_t seed_a, uint64_t seed_b,
                               float* out_b_comp_vs_b_eager,
                               float* out_b_comp_vs_a_comp,
                               float* out_a_comp_vs_a_eager) {
  try {
    // ---- fixed synthetic config (identical for A and B) ----
    const int num_layers = 3;
    const int hidden = 32;
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 8;
    const int l_cache = 4;
    const int E = 32;
    const int k = 4;
    const int num_dense_layers = 1;
    const int vocab = 48;
    const int T = 8;
    const float rope_theta = 10000.0f;
    const float norm_eps = 1e-5f;
    const int is_attn[3] = {1, 0, 1};  // attn, conv, attn
    const int moe_inter = 24;
    const int dense_inter = 40;
    const int token_ids[8] = {3, 11, 7, 22, 5, 19, 31, 14};

    // Build the synthetic model for `seed` into the shared g_weights() map +
    // g_lfm2_config, and return its embedding table (needed for the input
    // lookup at run time). Mirrors `mlx_lfm2_probe_moe_compiled_vs_eager`'s
    // construction with the WELL-SEPARATED bias.
    auto build_model = [&](uint64_t seed) -> array {
      uint64_t s = seed ? seed : 0x10F23C0Deull;
      auto next = [&]() -> float {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (static_cast<float>(s >> 40) / static_cast<float>(1u << 23)) - 1.0f;
      };
      auto mk = [&](std::vector<int> shape, float scale) -> array {
        int nelem = 1;
        for (int d : shape) {
          nelem *= d;
        }
        std::vector<float> buf(static_cast<size_t>(nelem));
        for (int i = 0; i < nelem; i++) {
          buf[i] = next() * scale;
        }
        mlx::core::Shape sh(shape.begin(), shape.end());
        return astype(array(buf.data(), sh, mlx::core::float32), mlx::core::bfloat16);
      };
      auto store = [&](const std::string& name, const array& a) {
        auto* p = new array(a);
        mlx_store_weight(name.c_str(), reinterpret_cast<mlx_array*>(p));
        delete p;
      };
      auto store_norm = [&](const std::string& name, std::vector<int> shape) {
        store(name, mk(shape, 0.02f) + array(1.0f, mlx::core::bfloat16));
      };

      mlx_clear_weights();
      auto embed = mk({vocab, hidden}, 0.05f);
      store("embed_tokens.weight", embed);
      store_norm("embedding_norm.weight", {hidden});
      for (int i = 0; i < num_layers; i++) {
        std::string lp = "layers." + std::to_string(i);
        store_norm(lp + ".operator_norm.weight", {hidden});
        store_norm(lp + ".ffn_norm.weight", {hidden});
        bool is_moe = i >= num_dense_layers;
        if (is_moe) {
          store(lp + ".feed_forward.gate.weight", mk({E, hidden}, 0.1f));
          std::vector<float> bias(E);
          for (int e = 0; e < E; e++) {
            bias[e] = 4.0f * static_cast<float>(e);  // well-separated
          }
          mlx::core::Shape bsh{E};
          store(lp + ".feed_forward.expert_bias",
                astype(array(bias.data(), bsh, mlx::core::float32), mlx::core::bfloat16));
          store(lp + ".feed_forward.switch_mlp.gate_proj.weight", mk({E, moe_inter, hidden}, 0.08f));
          store(lp + ".feed_forward.switch_mlp.up_proj.weight", mk({E, moe_inter, hidden}, 0.08f));
          store(lp + ".feed_forward.switch_mlp.down_proj.weight", mk({E, hidden, moe_inter}, 0.08f));
        } else {
          store(lp + ".feed_forward.gate_proj.weight", mk({dense_inter, hidden}, 0.08f));
          store(lp + ".feed_forward.up_proj.weight", mk({dense_inter, hidden}, 0.08f));
          store(lp + ".feed_forward.down_proj.weight", mk({hidden, dense_inter}, 0.08f));
        }
        if (is_attn[i]) {
          store(lp + ".self_attn.q_proj.weight", mk({num_heads * head_dim, hidden}, 0.08f));
          store(lp + ".self_attn.k_proj.weight", mk({num_kv_heads * head_dim, hidden}, 0.08f));
          store(lp + ".self_attn.v_proj.weight", mk({num_kv_heads * head_dim, hidden}, 0.08f));
          store(lp + ".self_attn.out_proj.weight", mk({hidden, num_heads * head_dim}, 0.08f));
          store_norm(lp + ".self_attn.q_layernorm.weight", {head_dim});
          store_norm(lp + ".self_attn.k_layernorm.weight", {head_dim});
        } else {
          store(lp + ".conv.in_proj.weight", mk({3 * hidden, hidden}, 0.08f));
          store(lp + ".conv.conv.weight", mk({hidden, l_cache, 1}, 0.2f));
          store(lp + ".conv.out_proj.weight", mk({hidden, hidden}, 0.08f));
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
      g_lfm2_config.num_experts = E;
      g_lfm2_config.num_experts_per_tok = k;
      g_lfm2_config.num_dense_layers = num_dense_layers;
      g_lfm2_config.norm_topk_prob = true;
      g_lfm2_config.use_expert_bias = true;
      g_lfm2_config.use_sigmoid = false;
      g_lfm2_config.tie_embedding = true;
      g_lfm2_config.max_kv_len = T;
      g_lfm2_is_attn.assign(is_attn, is_attn + num_layers);
      return embed;
    };

    // Drive T decode steps over the CURRENT g_weights()/g_lfm2_config state,
    // eager or compiled, using `embed` for the input lookup.
    auto run = [&](const array& embed, bool compiled) -> array {
      std::vector<array> caches;
      caches.reserve(num_layers * 2);
      for (int i = 0; i < num_layers; i++) {
        if (is_attn[i]) {
          caches.push_back(zeros({1, num_kv_heads, T, head_dim}, mlx::core::bfloat16));
          caches.push_back(zeros({1, num_kv_heads, T, head_dim}, mlx::core::bfloat16));
        } else {
          caches.push_back(zeros({1, l_cache - 1, hidden}, mlx::core::bfloat16));
          caches.push_back(zeros({}, mlx::core::bfloat16));
        }
      }
      array last_logits = zeros({1, vocab}, mlx::core::bfloat16);
      for (int t = 0; t < T; t++) {
        auto idx = reshape(array(token_ids[t], mlx::core::int32), {1});
        auto h = take(embed, idx, 0);
        std::vector<array> in;
        in.reserve(2 + num_layers * 2);
        in.push_back(h);
        in.push_back(array(t, mlx::core::int32));
        for (auto& c : caches) {
          in.push_back(c);
        }
        std::vector<array> outs;
        if (compiled) {
          auto fn = compiled_lfm2_decode();  // owned copy (see contract)
          outs = fn(in);
        } else {
          outs = lfm2_decode_fn(in);
        }
        last_logits = outs[0];
        for (int i = 0; i < num_layers * 2; i++) {
          caches[i] = outs[2 + i];
        }
      }
      mlx::core::eval({last_logits});
      return last_logits;
    };
    auto maxabs = [&](const array& x, const array& y) -> float {
      auto d = max(abs(subtract(astype(x, mlx::core::float32), astype(y, mlx::core::float32))));
      mlx::core::eval({d});
      return d.item<float>();
    };

    // ---- (1) MODEL A: build, compile (caches the graph), capture. ----
    // F3: build_model just did a destructive clear+store (mlx_clear_weights +
    // re-store). Mirror register_weights_with_cpp's pre-decode epoch bump so a
    // compiled closure cached by an EARLIER probe at the current epoch cannot be
    // reused for MODEL A's freshly-stored weights. Without this, MODEL A's first
    // compiled run could replay a prior probe's frozen constants (order-dependent
    // / flaky a_comp_vs_a_eager). MODEL B already bumps below (step 2), so the
    // bump is added ONLY at the MODEL-A site to preserve A/B distinguishability.
    auto embed_a = build_model(seed_a);
    mlx_lfm2_invalidate_compiled();
    auto a_comp = run(embed_a, /*compiled=*/true);
    auto a_eager = run(embed_a, /*compiled=*/false);

    // ---- (2) MODEL B: re-register DIFFERENT weights + bump the epoch. ----
    // build_model clears + re-stores g_weights() (registration's clear+store);
    // invalidate mirrors `register_weights_with_cpp`'s epoch bump under the
    // write lock. WITHOUT this bump the closure cached in step (1) is reused and
    // b_comp replays A's frozen constants.
    auto embed_b = build_model(seed_b);
    mlx_lfm2_invalidate_compiled();

    // ---- (3) run B compiled (must recompile) AND eager. ----
    auto b_comp = run(embed_b, /*compiled=*/true);
    auto b_eager = run(embed_b, /*compiled=*/false);

    if (out_b_comp_vs_b_eager) {
      *out_b_comp_vs_b_eager = maxabs(b_comp, b_eager);
    }
    if (out_b_comp_vs_a_comp) {
      *out_b_comp_vs_a_comp = maxabs(b_comp, a_comp);
    }
    if (out_a_comp_vs_a_eager) {
      *out_a_comp_vs_a_eager = maxabs(a_comp, a_eager);
    }

    mlx_clear_weights();
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] mlx_lfm2_probe_moe_ab_swap: %s\n", e.what());
    fflush(stderr);
    mlx_clear_weights();
    return -1;
  } catch (...) {
    mlx_clear_weights();
    return -1;
  }
}

}  // extern "C"
