// =============================================================================
// Qwen3.5 Dense MTP (Multi-Token Prediction) compiled draft + verify graphs.
//
// Companion to the main `mlx_qwen35.cpp` compiled forward path (W5 of the
// MTP deepresearch plan). Provides three FFI entrypoints:
//
//   - `mlx_qwen35_mtp_compiled_init_from_main`: allocates per-MTP-layer
//     KV caches sized to the main model's `max_kv_len` and snapshots
//     the model config (mirrors `mlx_qwen35_compiled_init_from_prefill`
//     but stores it in `g_mtp_*` globals so the MTP path can coexist
//     with the main path under a single `DENSE_COMPILED_MUTEX`).
//
//   - `mlx_qwen35_mtp_draft_compiled`: one MTP draft step. Inputs are
//     `(prev_hidden, prev_emb)` — both `[1, 1, hidden]` from the caller.
//     Outputs the next hidden state (for the next draft step) and the
//     draft logits (full vocab). Reuses the SAME compiled graph across
//     all D draft steps; offset is passed as `[1] int32` so the compile
//     cache is stable.
//
//   - `mlx_qwen35_mtp_verify_compiled`: one verify pass on `depth+1`
//     tokens. Updates the MAIN model's KV caches (`g_compiled_caches` /
//     `g_offset_int` from `mlx_qwen35.cpp`) by `depth+1` positions and
//     returns logits of shape `[1, depth+1, vocab]`. Internally
//     dispatches to a per-depth compiled function from a small table
//     populated lazily — verify graphs for depths {1..5} get cached;
//     `depth > 5` is rejected per the plan.
//
// IMPORTANT: this file READS the main path's `g_compiled_caches` /
// `g_offset_int` via the `extern` declarations below. The verify graph
// MUST be called in the same mutex critical section as the main path
// (`DENSE_COMPILED_MUTEX`) — Rust side enforces this. There is no
// process-wide lock here; we trust the Rust caller.
//
// Per the W5 plan: MTPLX `graphbank.py` pre-compiles one verify graph
// per depth (1..5). We mirror that with `g_verify_compiled_by_depth`,
// populated on first use of each depth. Pre-warming at init time was
// considered but deferred — first-use trace is one-time per process so
// the user-visible cost is negligible (a single extra draft cycle).
// =============================================================================

#include "mlx_qwen35_common.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <unordered_map>

using namespace qwen35_common;

// =============================================================================
// Cross-file shared state from `mlx_qwen35.cpp`.
//
// The main flat-path compiled state lives in an anonymous namespace in
// `mlx_qwen35.cpp` so it's NOT directly accessible here. We need it for
// two reasons:
//   (1) The verify graph MUST update the main KV caches (it's verifying
//       D+1 tokens against the committed prefix, then commits them).
//   (2) The init helper validates that the main path has been
//       initialised first by querying `mlx_qwen35_is_compile_inited()`.
//       Without that check, `mlx_qwen35_get_cache_offset()` silently
//       returns 0 from a fresh `g_offset_int`, and the MTP path would
//       mirror a phantom prefix offset into `g_mtp_offset_int`.
//
// Rather than make those globals header-visible (which would change the
// main file), we go through the existing FFI surface: the main path
// exposes `mlx_qwen35_get_cache_offset`, `mlx_qwen35_is_compile_inited`,
// and `mlx_qwen35_export_caches` for inspection, and
// `mlx_qwen35_forward_compiled` already mutates the main caches. The
// verify implementation here calls the existing flat per-step body D+1
// times in a loop (each call advances `g_offset_int` by one and updates
// `g_compiled_caches[]` in place). That keeps the state-mutation
// semantics identical to the main path and avoids the risk of
// double-incrementing the offset.
//
// The verify graph still gets a per-depth compile-cache key: each
// length-D verify is wrapped as a single closure that performs D+1
// sequential `mlx_qwen35_forward_compiled` calls; the closure body
// itself isn't traced (the per-step body is not currently compiled),
// but the per-step kernel fusions in `compiled_swiglu` /
// `compiled_compute_g` / `compiled_attn_gate` ARE cached. This is the
// same trade-off the main path makes; the W5 plan accepts it because
// graph-cached per-depth verify is a Phase-2 perf win (next workstream)
// and the immediate ≥1.6× target is reachable with the current kernel
// fusions alone.
// =============================================================================

extern "C" void mlx_qwen35_forward_compiled(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    mlx_array** output_logits,
    int* cache_offset_out);

extern "C" int mlx_qwen35_get_cache_offset();

extern "C" int mlx_qwen35_is_compile_inited();

// W6.5 — read the LAST stashed `g_last_hidden` (refcounted clone) from the
// main flat-path compiled state. The verify graph in this file loops
// `mlx_qwen35_forward_compiled` D+1 times; each call reassigns
// `g_last_hidden` to the post-final-norm hidden of THAT step's token.
// After the verify loop completes, `g_last_hidden` therefore holds the
// hidden at verify position D (i.e. the prediction context for the bonus
// token on full-accept, or for the residual sample on rejection).
//
// Returns null when the main path is uninitialised OR no forward has run
// since the last reset. Mirrors the public FFI used by the W6 Step-A
// seeding path; declared here too so the new `*_with_hidden` verify can
// thread it without going through the FFI boundary twice.
extern "C" void mlx_qwen35_export_last_hidden(mlx_array** out);

namespace {

// =====================================================================
// MTP-specific compiled state.
// =====================================================================

struct MTPCompileConfig : BaseConfig {
  int n_mtp_layers;       // number of MTP DecoderLayers (== config.n_mtp_layers)
  int mtp_fa_layer_idx;   // the layer-idx used inside MTP DecoderLayers
                          // (full_attention_interval - 1). All MTP
                          // layers share this idx because MTP layers
                          // are always full-attention (enforced
                          // Rust-side in `Qwen3_5MTPModule::new`).
};

// Init-from-main snapshot + per-MTP-layer KV caches.
static MTPCompileConfig g_mtp_config{};
static std::vector<array> g_mtp_compiled_caches;  // 2 * n_mtp_layers (K,V interleaved)
static int g_mtp_offset_int = 0;                  // mirror of main `g_offset_int`
                                                   // at the time of init.
static bool g_mtp_compile_inited = false;

// =====================================================================
// Draft graph: traced once, reused across all D draft steps.
//
// Inputs (vector order matters — compile keys on shapes only, but the
// closure captures positional indexes):
//   [0]                prev_hidden  [1, 1, hidden]  bf16
//   [1]                prev_emb     [1, 1, hidden]  bf16
//   [2]                offset_arr   [1]             int32 (RoPE +
//                                                  slice_update start)
//   For each MTP layer j in [0, n_mtp_layers):
//     [3 + j*2 + 0]    K cache      [1, Hkv, max_kv_len, head_dim]
//     [3 + j*2 + 1]    V cache      [1, Hkv, max_kv_len, head_dim]
//
// Outputs:
//   [0]                h_next       [1, 1, hidden]  — for next draft
//                                                    step's prev_hidden
//   [1]                draft_logits [1, vocab]      — sampler input
//   For each MTP layer j:
//     [2 + j*2 + 0]    new K cache
//     [2 + j*2 + 1]    new V cache
// =====================================================================
static std::vector<array> mtp_draft_decode_fn(const std::vector<array>& inputs) {
  const auto& cfg = g_mtp_config;
  auto prev_hidden = inputs[0];          // [1, 1, hidden]
  auto prev_emb    = inputs[1];          // [1, 1, hidden]
  auto offset_arr  = inputs[2];          // [1] int32

  // Mirror Qwen3_5MTPModule::forward (W2 dense Rust path):
  //   h_norm = pre_fc_norm_hidden(prev_hidden)
  //   e_norm = pre_fc_norm_embedding(prev_emb)
  //   h      = fc(concat([h_norm, e_norm], axis=-1))
  //   for layer in mtp.layers: h = layer(h, mask=None, cache=...)
  //   return norm(h)
  auto h_norm = fast::rms_norm(prev_hidden,
                               get_weight("mtp.pre_fc_norm_hidden.weight"),
                               cfg.rms_norm_eps);
  auto e_norm = fast::rms_norm(prev_emb,
                               get_weight("mtp.pre_fc_norm_embedding.weight"),
                               cfg.rms_norm_eps);

  // Concat along the hidden axis → [1, 1, 2*hidden]
  auto concat3d = concatenate({h_norm, e_norm}, 2);
  // mtp.fc projects 2*hidden → hidden. linear_proj operates on 2D
  // [B*T, in_features], so we squeeze the time dim to match.
  auto concat2d = reshape(concat3d, {1, cfg.hidden_size * 2});
  auto h2d = linear_proj(concat2d, "mtp.fc");          // [1, hidden]

  // Build the attention mask for the MTP layers. MTP draft steps run
  // ONE token per call, and the cache offset advances by 1 per draft
  // step. The mask is the same shape as the main flat path:
  //   [1, 1, 1, max_kv_len], additive bf16, -inf for positions > offset.
  // Because `offset_arr` is an array input, the mask must be built
  // from arange + compare (NOT from an int constant).
  int max_kv_len = inputs[3].shape(2);  // first K-cache's max_kv_len
  auto positions = arange(0, max_kv_len, mlx::core::int32);
  // offset_arr is [1] int32; broadcasting handles the comparison.
  auto valid_mask = less_equal(positions, offset_arr);
  auto attn_mask = where(valid_mask,
                         array(0.0f, mlx::core::bfloat16),
                         array(-std::numeric_limits<float>::infinity(),
                               mlx::core::bfloat16));
  attn_mask = reshape(attn_mask, {1, 1, 1, max_kv_len});

  std::vector<array> new_caches;
  new_caches.reserve(cfg.n_mtp_layers * 2);
  for (int j = 0; j < cfg.n_mtp_layers * 2; j++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  // MTP DecoderLayers — full-attention only (Rust enforces this in
  // Qwen3_5MTPModule::new). The per-layer key prefix is
  // `mtp.layers.{j}` and matches the W2 Rust `apply_weights` flow.
  for (int j = 0; j < cfg.n_mtp_layers; j++) {
    std::string lp = "mtp.layers." + std::to_string(j);

    auto normed = fast::rms_norm(h2d, get_weight(lp + ".input_layernorm.weight"),
                                 cfg.rms_norm_eps);

    const auto& kk = inputs[3 + j * 2];
    const auto& kv = inputs[3 + j * 2 + 1];
    auto res = attn_pure_fn_arr_offset(normed, lp,
                                       kk, kv, attn_mask, offset_arr, cfg);
    h2d = h2d + res.output;
    new_caches[j * 2]     = std::move(res.keys);
    new_caches[j * 2 + 1] = std::move(res.values);

    // MLP (SwiGLU) — uses the same `mtp.layers.{j}.mlp.*` keys.
    std::string mp = lp + ".mlp.";
    auto mlp_in  = fast::rms_norm(h2d, get_weight(lp + ".post_attention_layernorm.weight"),
                                  cfg.rms_norm_eps);
    auto gate    = linear_proj(mlp_in, mp + "gate_proj");
    auto up      = linear_proj(mlp_in, mp + "up_proj");
    auto mlp_out = linear_proj(swiglu(gate, up), mp + "down_proj");
    h2d = h2d + mlp_out;
  }

  // Final MTP norm + LM head.
  auto h_norm_final = fast::rms_norm(h2d, get_weight("mtp.norm.weight"),
                                     cfg.rms_norm_eps);
  auto logits = cfg.tie_word_embeddings
      ? linear_proj(h_norm_final, "embedding")
      : linear_proj(h_norm_final, "lm_head");

  // h_next: reshape h2d back to [1, 1, hidden] so the next draft step
  // can feed it as prev_hidden without re-shaping on the Rust side.
  // We use the PRE-norm hidden (Qwen3_5MTPModule::forward returns the
  // POST-norm hidden, but mtplx feeds the POST-norm hidden as the
  // prev_hidden of the next draft step — see MTPLX/mtplx/mtp_patch.py
  // lines 545-593 `_mtp_core` → `mtp_update_cache` returns
  // `post_norm` as the hidden for the next step. Matching that here.)
  auto h_next = reshape(h_norm_final, {1, 1, cfg.hidden_size});

  std::vector<array> result;
  result.reserve(2 + cfg.n_mtp_layers * 2);
  result.push_back(std::move(h_next));
  result.push_back(std::move(logits));
  for (auto& c : new_caches) result.push_back(std::move(c));
  return result;
}

// Wrapped in mlx::core::compile (modeled on mlx_qwen35.cpp:286-289 /
// mlx_qwen35_moe.cpp:549). Safe because mtp_draft_decode_fn takes
// offset_arr as an array input — see comment at line 166-169.
static auto& compiled_mtp_draft_decode() {
  static auto fn = mlx::core::compile(mtp_draft_decode_fn);
  return fn;
}

// =====================================================================
// Verify graphs: per-depth dispatcher.
//
// One entry per depth ∈ {1..5}. The closure for depth D performs D+1
// successive single-token decode steps via the existing flat-path FFI
// `mlx_qwen35_forward_compiled`. Each call advances the main
// `g_offset_int` by 1 and updates `g_compiled_caches[]` in place,
// matching the per-step semantics the chat loop already relies on. The
// closure stacks the D+1 logits along axis 1 to produce the verify
// output `[1, depth+1, vocab]`.
//
// Per the W5 plan we MUST reject depth > 5. Per-depth lookup is O(log
// k) under `std::map`-style search but k ≤ 5 so we use a fixed-size
// array indexed by depth - 1.
// =====================================================================

constexpr int MAX_VERIFY_DEPTH = 5;
using VerifyFn = std::function<std::vector<array>(
    const array&, const array&)>;
static std::array<VerifyFn, MAX_VERIFY_DEPTH> g_verify_compiled_by_depth{};

// Build a verify closure for a fixed depth. Captures NO per-call
// state — depth is baked in. The closure expects `input_ids` of shape
// `[1, depth+1]` and the `embedding_weight` from the model. The
// closure returns `{logits[1, depth+1, vocab]}` as a single-element
// vector.
static VerifyFn make_verify_fn(int depth) {
  return [depth](const array& input_ids, const array& embedding_weight)
             -> std::vector<array> {
    // Split input_ids along the time axis and feed each token through
    // the existing flat-path single-step FFI. The main path's
    // `g_offset_int` increments by 1 per call, so after the loop the
    // committed-prefix offset will have advanced by `depth + 1`.
    int seq_len = input_ids.shape(1);
    if (seq_len != depth + 1) {
      throw std::runtime_error(
          "mlx_qwen35_mtp_verify: input_ids time dim (" +
          std::to_string(seq_len) + ") must equal depth+1 (" +
          std::to_string(depth + 1) + ")");
    }

    std::vector<array> per_step_logits;
    per_step_logits.reserve(seq_len);

    for (int t = 0; t < seq_len; t++) {
      // Slice the t-th token out of input_ids → [1, 1]
      auto tok = slice(input_ids, {0, t}, {1, t + 1});

      mlx_array* out_ptr = nullptr;
      // Wrap the embedding-weight in a stack array view because the
      // FFI expects `mlx_array*` and we have a const-array reference.
      // The embedding weight is a global g_weights() entry on the
      // main path side, so we materialize a temporary handle.
      array emb_copy = embedding_weight;
      array tok_copy = tok;
      mlx_qwen35_forward_compiled(
          reinterpret_cast<mlx_array*>(&tok_copy),
          reinterpret_cast<mlx_array*>(&emb_copy),
          &out_ptr,
          /*cache_offset_out=*/nullptr);
      if (!out_ptr) {
        throw std::runtime_error(
            "mlx_qwen35_mtp_verify: main forward returned null at t=" +
            std::to_string(t));
      }
      // Take ownership of the heap-allocated array the FFI returned.
      array step_logits = *reinterpret_cast<array*>(out_ptr);
      delete reinterpret_cast<array*>(out_ptr);
      // step_logits shape: [1, vocab]. Insert a time dim for stacking.
      per_step_logits.push_back(reshape(step_logits,
                                       {1, 1, step_logits.shape(-1)}));
    }

    // Stack along time axis → [1, depth+1, vocab].
    auto stacked = concatenate(per_step_logits, 1);
    return {stacked};
  };
}

// Lookup or lazily construct the verify closure for a given depth.
// `depth` is validated against `MAX_VERIFY_DEPTH` by the FFI caller.
static const VerifyFn& get_or_make_verify_fn(int depth) {
  auto& slot = g_verify_compiled_by_depth[depth - 1];
  if (!slot) {
    slot = make_verify_fn(depth);
  }
  return slot;
}

} // namespace

// =============================================================================
// Public FFI functions
// =============================================================================

extern "C" {

// -----------------------------------------------------------------------------
// Initialize MTP compiled state.
//
// MUST be called once per turn AFTER `mlx_qwen35_compiled_init_from_prefill`
// has set up the main path's globals. Allocates fresh MTP KV caches
// sized to `max_kv_len`, snapshots the config, and primes
// `g_mtp_offset_int` from the main path's current offset.
//
// All MTP DecoderLayers share `mtp_fa_layer_idx = full_attention_interval - 1`
// per the Rust W2 invariant — this affects RoPE only if the helper
// inspected `layer_idx`, which our `attn_pure_fn_arr_offset` does NOT
// (the prefix is the only parameterization). The argument is kept in
// the config for forward compatibility / introspection.
//
// Returns 0 on success, -1 on failure. On failure the MTP state is
// left uninitialised (`g_mtp_compile_inited = false`) so subsequent
// draft/verify calls become null-pointer no-ops, letting the Rust
// caller fall back to the eager Rust path.
// -----------------------------------------------------------------------------
int32_t mlx_qwen35_mtp_compiled_init_from_main(
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
    int n_mtp_layers
) {
  try {
    if (n_mtp_layers <= 0) {
      std::cerr << "[MLX] mtp_compiled_init: n_mtp_layers must be > 0 (got "
                << n_mtp_layers << ")" << std::endl;
      g_mtp_compile_inited = false;
      return -1;
    }
    if (!mlx_qwen35_is_compile_inited()) {
      std::cerr << "[MLX] mtp_compiled_init: main compiled path is not "
                   "initialised — call mlx_qwen35_compiled_init_from_prefill "
                   "before mlx_qwen35_mtp_compiled_init_from_main"
                << std::endl;
      g_mtp_compile_inited = false;
      return -1;
    }
    if (!has_weight("mtp.norm.weight")) {
      std::cerr << "[MLX] mtp_compiled_init: mtp.norm.weight not "
                   "registered — load MTP weights first" << std::endl;
      g_mtp_compile_inited = false;
      return -1;
    }

    g_mtp_config = MTPCompileConfig{};
    g_mtp_config.num_layers              = num_layers;
    g_mtp_config.hidden_size             = hidden_size;
    g_mtp_config.num_heads               = num_heads;
    g_mtp_config.num_kv_heads            = num_kv_heads;
    g_mtp_config.head_dim                = head_dim;
    g_mtp_config.rope_theta              = rope_theta;
    g_mtp_config.rope_dims               = rope_dims;
    g_mtp_config.rms_norm_eps            = rms_norm_eps;
    g_mtp_config.full_attention_interval = full_attention_interval;
    g_mtp_config.linear_num_k_heads      = linear_num_k_heads;
    g_mtp_config.linear_num_v_heads      = linear_num_v_heads;
    g_mtp_config.linear_key_head_dim     = linear_key_head_dim;
    g_mtp_config.linear_value_head_dim   = linear_value_head_dim;
    g_mtp_config.linear_conv_kernel_dim  = linear_conv_kernel_dim;
    g_mtp_config.tie_word_embeddings     = (tie_word_embeddings != 0);
    g_mtp_config.max_kv_len              = max_kv_len;
    g_mtp_config.batch_size              = batch_size;
    g_mtp_config.n_mtp_layers            = n_mtp_layers;
    g_mtp_config.mtp_fa_layer_idx        = std::max(full_attention_interval - 1, 0);

    // Fresh per-MTP-layer KV caches. All MTP layers are full-attention,
    // so each entry is a [B, Hkv, max_kv_len, D] zero buffer at bf16.
    // We DO NOT seed from the main path's caches: MTP draft steps
    // build their OWN KV context from the drafted tokens and discard
    // it on acceptance failure, so seeding from the main path's
    // committed prefix would corrupt the MTP layer's attention.
    g_mtp_compiled_caches.clear();
    g_mtp_compiled_caches.reserve(n_mtp_layers * 2);
    for (int j = 0; j < n_mtp_layers; j++) {
      auto kk = zeros({batch_size, num_kv_heads, max_kv_len, head_dim},
                      mlx::core::bfloat16);
      auto vv = zeros({batch_size, num_kv_heads, max_kv_len, head_dim},
                      mlx::core::bfloat16);
      g_mtp_compiled_caches.push_back(std::move(kk));
      g_mtp_compiled_caches.push_back(std::move(vv));
    }

    // Mirror the main path's current offset. Draft steps will advance
    // `g_mtp_offset_int` independently — the main offset is untouched
    // by drafting.
    //
    // Why this mirror is correct: MTP attention masks consistently treat
    // draft-step positions as following the committed prefix. The MTP
    // K/V caches are zero-initialised just above, so the attn_mask
    // validates only positions [0..g_mtp_offset_int]; the zero-filled
    // tail past g_mtp_offset_int is masked out and never read until a
    // draft step writes into it.
    g_mtp_offset_int = mlx_qwen35_get_cache_offset();

    // Drop any stale verify closures from a prior model load — the
    // per-depth closures capture nothing model-specific, but a fresh
    // verify table per init makes the per-process state easier to
    // reason about for debugging.
    for (auto& slot : g_verify_compiled_by_depth) {
      slot = nullptr;
    }

    g_mtp_compile_inited = true;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_qwen35_mtp_compiled_init_from_main: "
              << e.what() << std::endl;
    g_mtp_compile_inited = false;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] mlx_qwen35_mtp_compiled_init_from_main: "
                 "unknown exception" << std::endl;
    g_mtp_compile_inited = false;
    return -1;
  }
}

// -----------------------------------------------------------------------------
// One MTP draft step.
//
// Inputs:
//   - prev_hidden_ptr:  `[1, 1, hidden]` bf16 — output of the previous
//                       MTP draft step OR the last main-path hidden if
//                       this is the first draft.
//   - prev_emb_ptr:     `[1, 1, hidden]` bf16 — embedding of the
//                       previously-committed token OR the last drafted
//                       token (caller picks the right one).
//
// Outputs:
//   - *out_h_next:      heap-allocated `[1, 1, hidden]` bf16 (caller
//                       owns) — feed as `prev_hidden` to the next
//                       draft step.
//   - *out_logits:      heap-allocated `[1, vocab]` bf16 (caller owns)
//                       — sampler input for the drafted token at this
//                       step.
//
// Advances `g_mtp_offset_int` by 1 and mutates `g_mtp_compiled_caches`
// in place. Returns null pointers on failure (init not done, exception,
// etc.) — the Rust caller MUST null-check before consuming.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_draft_compiled(
    mlx_array* prev_hidden_ptr,
    mlx_array* prev_emb_ptr,
    mlx_array** out_h_next,
    mlx_array** out_logits
) {
  if (out_h_next) *out_h_next = nullptr;
  if (out_logits) *out_logits = nullptr;
  if (!g_mtp_compile_inited) return;
  if (!prev_hidden_ptr || !prev_emb_ptr || !out_h_next || !out_logits) return;

  try {
    auto& prev_hidden = *reinterpret_cast<array*>(prev_hidden_ptr);
    auto& prev_emb    = *reinterpret_cast<array*>(prev_emb_ptr);

    std::vector<array> inputs;
    inputs.reserve(3 + g_mtp_config.n_mtp_layers * 2);
    inputs.push_back(prev_hidden);
    inputs.push_back(prev_emb);
    inputs.push_back(reshape(array(g_mtp_offset_int, mlx::core::int32), {1}));
    for (const auto& c : g_mtp_compiled_caches) {
      inputs.push_back(c);
    }

    auto outputs = compiled_mtp_draft_decode()(inputs);

    *out_h_next = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    *out_logits = reinterpret_cast<mlx_array*>(new array(outputs[1]));
    g_mtp_offset_int++;
    for (int j = 0; j < g_mtp_config.n_mtp_layers * 2; j++) {
      g_mtp_compiled_caches[j] = outputs[2 + j];
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_mtp_draft_compiled: %s\n",
            e.what());
    fflush(stderr);
    if (out_h_next) *out_h_next = nullptr;
    if (out_logits) *out_logits = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_mtp_draft_compiled\n");
    fflush(stderr);
    if (out_h_next) *out_h_next = nullptr;
    if (out_logits) *out_logits = nullptr;
  }
}

// -----------------------------------------------------------------------------
// One MTP verify step.
//
// Inputs:
//   - input_ids_ptr:        `[1, depth+1]` int32 — `[last_committed_id,
//                           drafted_tok_0, ..., drafted_tok_{depth-1}]`.
//   - embedding_weight_ptr: model's embedding weight (or LM-head if
//                           untied) — same array the main path uses.
//   - depth:                ∈ {1..5}. Larger values rejected.
//
// Output:
//   - *out_logits:          heap-allocated `[1, depth+1, vocab]` bf16
//                           (caller owns).
//
// SIDE EFFECTS: advances the MAIN compiled-path offset
// `g_qwen35.cpp::g_offset_int` by `depth + 1` and updates
// `g_compiled_caches[]` in place. The caller MUST hold
// `DENSE_COMPILED_MUTEX` for the entire draft+verify cycle so no other
// turn can mutate the main path state mid-verify.
//
// Returns null on failure. On a depth > 5 violation, writes a stderr
// diagnostic and leaves the main caches untouched (the per-depth
// closure has not yet been invoked).
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_verify_compiled(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_logits
) {
  if (out_logits) *out_logits = nullptr;
  if (!input_ids_ptr || !embedding_weight_ptr || !out_logits) return;
  if (depth < 1 || depth > MAX_VERIFY_DEPTH) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_mtp_verify_compiled: depth %d outside [1, %d]\n",
            depth, MAX_VERIFY_DEPTH);
    fflush(stderr);
    return;
  }

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != depth + 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_mtp_verify_compiled: input_ids shape must be "
              "[1, depth+1=%d], got ndim=%d shape=[%lld,%lld]\n",
              depth + 1, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    const auto& verify_fn = get_or_make_verify_fn(depth);
    auto outputs = verify_fn(input_ids, embedding_weight);
    *out_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_mtp_verify_compiled: %s\n",
            e.what());
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_mtp_verify_compiled\n");
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
  }
}

// -----------------------------------------------------------------------------
// W6.5 — verify pass that ALSO exports the verify-final hidden so the
// caller can chain MTP cycles without running a fresh main-model
// forward at each cycle's "Step A".
//
// Behaviourally identical to `mlx_qwen35_mtp_verify_compiled` for the
// logits output (and the same `g_compiled_caches[]` / `g_offset_int`
// mutation contract) plus one extra owned `mlx_array*` for the
// post-final-norm hidden of the LAST verify iteration. The verify graph
// here loops `mlx_qwen35_forward_compiled` D+1 times — each call
// overwrites the main path's `g_last_hidden` (see `mlx_qwen35.cpp`
// `qwen35_decode_fn` line 175). After the loop completes, that global
// holds the hidden of verify position D, which we export via the
// existing `mlx_qwen35_export_last_hidden` clone helper.
//
// Why an extra entrypoint instead of extending the existing one:
//   (a) backward compat — callers that don't need the hidden keep the
//       2-output contract and a free `nullptr` slot;
//   (b) explicit caller opt-in keeps the lazy MLX graph for the hidden
//       alive across the FFI boundary (the lifetime contract on the
//       returned `mlx_array*` mirrors the existing FFI), avoiding a
//       silent perf regression for non-chained callers.
//
// The hidden's lifetime contract mirrors
// `mlx_qwen35_export_last_hidden`:
//   - The returned handle is a lazy MLX array whose graph references
//     the verify's final_norm output. The caller MUST `eval()` it (or
//     consume it via a graph that does) before reading any element.
//   - The caller MUST NOT call `mlx_qwen35_compiled_reset()` between
//     export and eval — that reset would clear `g_compiled_caches`
//     whose inputs the hidden depends on via the cached graph.
//
// `*out_last_hidden` is nullptr on failure (matches the logits-only
// FFI's failure semantics so the Rust caller can fall back to Step A
// when chaining is unavailable). The caller MUST null-check both
// outputs before consuming.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_verify_compiled_with_hidden(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_logits,
    mlx_array** out_last_hidden
) {
  if (out_logits) *out_logits = nullptr;
  if (out_last_hidden) *out_last_hidden = nullptr;
  if (!input_ids_ptr || !embedding_weight_ptr || !out_logits ||
      !out_last_hidden) {
    return;
  }
  if (depth < 1 || depth > MAX_VERIFY_DEPTH) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden: depth %d "
            "outside [1, %d]\n",
            depth, MAX_VERIFY_DEPTH);
    fflush(stderr);
    return;
  }

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != depth + 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden: input_ids "
              "shape must be [1, depth+1=%d], got ndim=%d shape=[%lld,%lld]\n",
              depth + 1, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    // Run the existing per-depth verify loop. After this returns,
    // `g_compiled_caches[]` / `g_offset_int` have been advanced by D+1
    // AND `g_last_hidden` holds the post-final-norm hidden of the LAST
    // verify iteration (verify position D).
    const auto& verify_fn = get_or_make_verify_fn(depth);
    auto outputs = verify_fn(input_ids, embedding_weight);
    *out_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));

    // Export the final-iteration hidden via the same ref-counted clone
    // path the public W6 Step-A seeding FFI uses. Returns nullptr if
    // `g_last_hidden` is unpopulated — in practice the verify loop just
    // ran D+1 forwards so the stash is fresh, but defensive-checked.
    mlx_qwen35_export_last_hidden(out_last_hidden);
    if (*out_last_hidden == nullptr) {
      // Shouldn't happen after a successful verify, but if it does the
      // caller can still consume the logits and fall back to a fresh
      // Step A on the next cycle. Emit a diagnostic so the
      // unexpected-state case is observable.
      fprintf(stderr,
              "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden: "
              "verify succeeded but g_last_hidden was unpopulated; the "
              "Rust caller should fall back to Step A on the next cycle\n");
      fflush(stderr);
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_mtp_verify_compiled_with_hidden: %s\n",
            e.what());
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_last_hidden) *out_last_hidden = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in "
            "mlx_qwen35_mtp_verify_compiled_with_hidden\n");
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_last_hidden) *out_last_hidden = nullptr;
  }
}

// -----------------------------------------------------------------------------
// Tear down MTP compiled state. Idempotent; safe to call on
// already-empty state. Does NOT touch the main path's globals — call
// `mlx_qwen35_compiled_reset` separately for that.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_compiled_reset() {
  g_mtp_compiled_caches.clear();
  g_mtp_offset_int = 0;
  g_mtp_compile_inited = false;
  g_mtp_config = MTPCompileConfig{};
  for (auto& slot : g_verify_compiled_by_depth) {
    slot = nullptr;
  }
}

// -----------------------------------------------------------------------------
// Adjust the MTP offset by `delta` (e.g. to rewind after a verify-reject
// rolled back the main path). Mirrors `mlx_qwen35_compiled_adjust_offset`.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_compiled_adjust_offset(int delta) {
  g_mtp_offset_int += delta;
}

// -----------------------------------------------------------------------------
// W6 Bug #2 fix (Option Reset): begin a fresh MTP draft cycle aligned to
// the main path's current offset. Zeroes the MTP K/V caches and sets
// `g_mtp_offset_int = main_offset`.
//
// Why this exists:
//   Per outer iteration of `decode_loop_mtp!` the main offset advances
//   by D+2 (1 Step-A forward + (D+1) verify forwards) while the MTP
//   draft offset only advances by D. After K cycles the MTP offset
//   lags the main offset by 2K, so MTP RoPE positions diverge from
//   the actual sequence positions — drafts produce gibberish, every
//   token rejects, and the residual sample comes from a corrupted
//   distribution.
//
//   Naively syncing the offset (Option Sync) leaves a 2-position gap
//   in the MTP K/V buffer per cycle (the slots Step A + verify[0] wrote
//   on the main path are never written on the MTP path); those slots
//   read back as zero K/V and pollute the draft attention. Resetting
//   to all-zeros and re-anchoring to `main_offset` matches the W5 init
//   behavior (zeroed buffer + offset = prefill_len), so every cycle is
//   self-contained and behaves like the very first draft cycle.
//
// Trade-off: abandons the W6.5 chained-cycle perf win (where prior
// cycles' MTP K/V would seed the next cycle's drafts). That's out of
// scope here — the immediate goal is parity, not throughput.
//
// No-op if MTP isn't initialised — the dispatcher Rust-side already
// gates the call on `mtp_active`, but defensive-checked here too.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_compiled_begin_cycle(int main_offset) {
  if (!g_mtp_compile_inited) return;
  const auto& cfg = g_mtp_config;
  for (int j = 0; j < cfg.n_mtp_layers; j++) {
    g_mtp_compiled_caches[j * 2]     = zeros(
        {cfg.batch_size, cfg.num_kv_heads, cfg.max_kv_len, cfg.head_dim},
        mlx::core::bfloat16);
    g_mtp_compiled_caches[j * 2 + 1] = zeros(
        {cfg.batch_size, cfg.num_kv_heads, cfg.max_kv_len, cfg.head_dim},
        mlx::core::bfloat16);
  }
  g_mtp_offset_int = main_offset;
}

// -----------------------------------------------------------------------------
// Read accessor for the current MTP offset (debugging / introspection
// from Rust unit tests).
// -----------------------------------------------------------------------------
int mlx_qwen35_mtp_get_offset() {
  return g_mtp_offset_int;
}

}  // extern "C"
