#include "mlx_qwen35_common.h"
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>

using namespace qwen35_common;

// =============================================================================
// Qwen3.5 Dense Compiled Forward Pass
//
// Implements the entire Qwen3.5 dense model forward pass (single-token decode)
// in one FFI call. Weight storage, helpers, GDN and attention functions are
// shared with the MoE path via mlx_qwen35_common.h.
//
// A parallel paged-decode graph (`dense_compiled_decode_fn_paged`) routes
// full-attention layers through the paged-attention kernels while keeping
// linear-attention (GDN) layers on the same `gdn_pure_fn` helper as the flat
// path. The paged-path globals (`g_dense_paged_*`) are independent from the
// flat-path globals (`g_compiled_*`) so both graphs can coexist; the
// dispatcher chooses one or the other per turn but never mixes them within a
// single decode step.
// =============================================================================

namespace {

// Config for the compiled path (extends BaseConfig with compile-specific state)
struct CompileConfig : BaseConfig {};

static CompileConfig g_compile_config{};
static std::vector<array> g_compiled_caches;         // num_layers * 2 arrays
static std::optional<array> g_compiled_offset;       // scalar int32 (current decode position)
static int g_offset_int = 0;                         // C++ int mirror of g_compiled_offset
static bool g_compile_inited = false;

// MTP — last post-final-norm hidden of the LAST committed token, stashed at
// the LM-head boundary inside `qwen35_decode_fn` BEFORE the `lm_head` linear
// runs. Shape is `[B, hidden_size]` after the implicit reshape that `take` +
// the per-row final_norm produce (decode batch is `[1, 1]` → reshape({-1}) →
// `[1]` → `take` → `[1, hidden]`). The MTP seeding path consumes this via
// `mlx_qwen35_export_last_hidden` to get the first draft step's `prev_hidden`
// without re-running the main forward.
//
// Cleared on `mlx_qwen35_compiled_reset` so cross-turn stale handles never
// leak. `std::optional<array>` distinguishes "never populated" from
// "populated with zeros".
static std::optional<array> g_last_hidden;

// Paged-path sibling of `g_last_hidden`. Stashed at the LM-head boundary
// inside `dense_compiled_decode_fn_paged` BEFORE the `lm_head` linear runs.
// Shape is `[B, hidden_size]` after the implicit reshape (paged decode batch
// is `[1, 1]` → `take` → `[1, hidden]`). Consumed by the paged-MTP gate via
// `mlx_qwen35_export_last_hidden_paged` to seed the next MTP draft cycle
// without re-running the main forward. Cleared on `mlx_qwen35_compiled_reset`.
static std::optional<array> g_last_hidden_paged;

// MTP partial-accept snapshot of the GDN linear-attention caches (conv_state +
// recurrent_state) plus the decode offset, taken BEFORE the verify FFI runs
// its D+1 sequential forward passes. The verify loop mutates `g_compiled_caches`
// in place for every layer; for linear-attention layers the recurrent / conv
// state advances through D+1 tokens that may or may not be accepted. On
// rejection (any `accepted_drafts < depth`) we restore this snapshot so the
// linear state matches the pre-verify "after Step A" state; the caller then
// replays exactly K accepted drafts via `forward_compiled` to bring the linear
// state forward through the committed drafts only.
//
// Only LINEAR-attention layers are snapshotted — full-attention K/V slots are
// correctly handled by the existing offset-rewind path (later writes overwrite
// the stale slots, and attention masking hides them via `offset`).
//
// `g_linear_snapshot_offset` records the offset at snapshot time so restore can
// rewind the offset in lockstep. `g_linear_snapshot_taken` guards against an
// out-of-order restore that would silently install garbage state.
//
// Cleared by `mlx_qwen35_compiled_reset`.
static std::vector<array> g_compiled_linear_snapshot;
static int g_linear_snapshot_offset = 0;
static bool g_linear_snapshot_taken = false;

// Tape-replay rollback infrastructure.
//
// When `g_tape_recording_armed == true`, `qwen35_decode_fn` routes
// linear-attention layers through `gdn_pure_fn_with_tape` (which calls the
// tape-emitting Metal kernel) and appends the per-step `(tape, k, g, qkv)`
// tensors into the per-layer accumulators below. After the MTP verify loop's
// D+1 forwards, each accumulator holds a `[B, D+1, ...]` tensor recording every
// per-step innovation. On rollback the replay kernel applies the first
// `accepted_steps` of those innovations to the pre-verify snapshot state.
//
// Layout (each vector has `num_layers` slots; full-attention slots hold
// placeholders so per-layer indexing stays uniform):
//   - g_gdn_tape_acc[i]:        [B, accumulated_steps, Hv, Dv]  fp32
//   - g_gdn_k_tape_acc[i]:      [B, accumulated_steps, Hk, Dk]  model dtype
//   - g_gdn_g_tape_acc[i]:      [B, accumulated_steps, Hv]      fp32
//   - g_gdn_qkv_tape_acc[i]:    [B, accumulated_steps, conv_dim] model dtype
//
// Lifecycle:
//   - `mlx_qwen35_compiled_tape_arm()` clears the accumulators and sets the
//     recording flag. Called right BEFORE the verify FFI runs its D+1
//     sequential forwards (when tape-replay is ENABLED — env var
//     `MLX_MTP_USE_TAPE_REPLAY != "0"`).
//   - Each `qwen35_decode_fn` step appends one slice to each layer's
//     accumulator via `concatenate` along axis 1.
//   - `mlx_qwen35_compiled_tape_replay(accepted_steps)` consumes the first
//     `accepted_steps` slices to restore the snapshot state, then clears the
//     accumulators.
//   - `mlx_qwen35_compiled_tape_disarm()` is idempotent. Rejection and
//     full-accept paths normally consume the tape via `_tape_replay`, which
//     disarms internally; explicit disarm is only for cleanup.
//
// Cleared by `mlx_qwen35_compiled_reset`.
static bool g_tape_recording_armed = false;
static std::vector<std::optional<array>> g_gdn_tape_acc;    // [num_layers]
static std::vector<std::optional<array>> g_gdn_k_tape_acc;  // [num_layers]
static std::vector<std::optional<array>> g_gdn_g_tape_acc;  // [num_layers]
static std::vector<std::optional<array>> g_gdn_qkv_tape_acc;// [num_layers]

// =====================================================================
// Paged-decode globals.
//
// Independent from the flat-path globals above so both compile graphs can
// coexist while the Rust dispatcher decides per-turn whether to take the
// compiled paged path or fall back to the flat path.
//
// Layout (size = num_layers; one entry per layer indexed by `layer_idx`):
//   - g_dense_k_pools / g_dense_v_pools / g_dense_k_scales /
//     g_dense_v_scales: meaningful only for full-attention layers.
//     Linear-layer slots hold a small placeholder array — never read by the
//     paged graph.
//   - g_dense_paged_linear_caches: size = num_layers * 2. Slot `2i` and
//     `2i+1` hold conv_state and recurrent_state for layer `i` when that layer
//     is linear-attention. Full-attn slots hold placeholders.
//
// `g_dense_paged_inited` gates the `mlx_qwen35_forward_paged` FFI.
// `mlx_qwen35_init_paged` is the only way to flip it true; clearing happens in
// `mlx_qwen35_compiled_reset` so a single reset wipes BOTH graphs' state.
// =====================================================================
static CompileConfig g_dense_paged_config{};
static std::vector<array> g_dense_k_pools;          // [num_layers]
static std::vector<array> g_dense_v_pools;          // [num_layers]
static std::vector<array> g_dense_k_scales;         // [num_layers]
static std::vector<array> g_dense_v_scales;         // [num_layers]
static std::vector<array> g_dense_paged_linear_caches;  // [num_layers * 2]
static int g_dense_paged_offset_int = 0;
static bool g_dense_paged_inited = false;

// Paged-pool MTP partial-accept snapshot/restore for the GDN linear-attention
// recurrent + conv state. Sibling of the BHTD `g_compiled_linear_snapshot`.
// Paged verify mutates `g_dense_paged_linear_caches` after processing all D+1
// verify tokens; on partial-accept (accepted_drafts < depth) the committed
// state corresponds to fewer steps than the recorded ones, so the next forward
// must start from a state matching the accepted prefix. Full-attention slots
// are intentionally NOT snapshotted — the paged pool's `record_tokens` /
// `rollback_last_tokens` cursor on the Rust side handles K/V slot bookkeeping;
// later writes overwrite the stale trailing slots.
//
// Cleared by `mlx_qwen35_compiled_reset`.
static std::vector<array> g_dense_paged_linear_snapshot;
static bool g_dense_paged_linear_snapshot_taken = false;

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

  // T=1 decode: skip the per-step `[1,1,1,max_kv_len]` attention mask.
  // `attn_pure_fn(dynamic_kv=true)` slices the KV cache down to the valid range
  // `[0..offset+1]` and passes NO mask to SDPA. Mirrors upstream mlx-lm's
  // `create_attention_mask(N=1) → None` + `cache.state` (keys[..., :offset, :])
  // pattern. Faster SDPA kernel; byte-exact at T=0 because the masked-out
  // columns can never affect the per-row softmax (the mask is `-inf` exactly
  // where the sliced version has no entry at all).
  //
  // `qwen35_decode_fn` is NOT wrapped in `mlx::core::compile`, so using a
  // C++-int slice bound (`valid_len = offset+1`) here is safe; the graph
  // topology evolves with offset but is re-traced every call anyway. Compiled
  // paths (MoE flat / paged / batched verify / MTP draft) keep the static-mask
  // path because they need fixed shapes for the `mlx::core::compile` cache.

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
      if (g_tape_recording_armed) {
        // Emit tape during MTP verify forwards so the rollback path can replay
        // only the accepted prefix instead of re-running the main model. The
        // math is identical to `gdn_pure_fn`; the extra outputs are appended
        // into per-layer accumulators along the time axis so after the D+1
        // verify-loop iterations each accumulator holds the full per-step
        // record.
        auto res = gdn_pure_fn_with_tape(normed, i, cs, rs, cfg);
        layer_out = std::move(res.output);
        new_caches[i * 2]     = std::move(res.conv_state);
        new_caches[i * 2 + 1] = std::move(res.recurrent_state);

        auto append = [](std::optional<array>& slot, const array& step) {
          if (!slot.has_value()) {
            slot = step;
          } else {
            slot = concatenate({*slot, step}, 1);
          }
        };
        append(g_gdn_tape_acc[i],     res.tape);
        append(g_gdn_k_tape_acc[i],   res.k_tape);
        append(g_gdn_g_tape_acc[i],   res.g_tape);
        append(g_gdn_qkv_tape_acc[i], res.qkv_tape);
      } else {
        auto res = gdn_pure_fn(normed, i, cs, rs, cfg);
        layer_out = std::move(res.output);
        new_caches[i * 2]     = std::move(res.conv_state);
        new_caches[i * 2 + 1] = std::move(res.recurrent_state);
      }
    } else {
      const auto& kk = inputs[2 + i * 2];
      const auto& kv = inputs[2 + i * 2 + 1];
      // `dynamic_kv=true`: helper slices KV down to `[0..offset+1]` and skips
      // the mask entirely. The `attn_mask` slot is unused under this branch;
      // pass a zero-element placeholder.
      auto dummy_mask = zeros({}, mlx::core::bfloat16);
      auto res = attn_pure_fn(normed, i, kk, kv, dummy_mask, offset, cfg,
                              /*dynamic_kv=*/true);
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
  // MTP — stash the post-final-norm hidden of the last decoded token BEFORE
  // lm_head. Shape is `[1, hidden_size]` for the flat-path decode (batch=1,
  // single token). Overwritten every call so it always reflects the last
  // committed token's hidden state.
  g_last_hidden = h;
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
// Batched verify decode graph.
//
// One forward over `T = depth + 1` tokens (vs D+1 sequential
// `qwen35_decode_fn` calls). The batched graph emits `[1, D+1, hidden]`
// natively in one dispatch, avoiding the per-position `g_last_hidden` capture
// + `concatenate` accumulator append.
//
// Input vector layout:
//   [0]                      h_3d        [B, T, hidden]  bf16
//   [1]                      offset_arr  [1]             int32
//   For each layer i in [0, num_layers):
//     [2 + i*2 + 0]          cache_a     (k for full-attn, conv_state for linear)
//     [2 + i*2 + 1]          cache_b     (v for full-attn, recurrent_state for linear)
//
// Output vector layout:
//   [0]                      logits      [B, T, vocab]
//   [1]                      hiddens     [B, T, hidden]  pre-lm_head hidden
//   For each layer i:
//     [2 + i*2 + 0]          new_cache_a
//     [2 + i*2 + 1]          new_cache_b
//   For each layer i (TAPE variant only):
//     [extra_base + i*4 + 0] tape        [B, T, Hv, Dv]  fp32  — placeholder for full-attn
//     [extra_base + i*4 + 1] k_tape      [B, T, Hk, Dk]
//     [extra_base + i*4 + 2] g_tape      [B, T, Hv]      fp32
//     [extra_base + i*4 + 3] qkv_tape    [B, T, conv_dim]
//
// The graph is parameterised by `T` only (depth at trace time); the offset
// arrives as an array so the same compiled graph reuses across all decode
// positions for a given D. One graph per (depth, with_tape) is cached.
// =====================================================================

namespace {

// Bucketed verify graph.
//
// `bucket_kv_len` is the static SDPA key-column count baked into the compile
// trace for THIS entry of the per-bucket cache. The graph processes the same
// `[B, T, hidden]` input and emits the same outputs as the single-trace
// version, but SDPA reads only the first `bucket_kv_len` columns of each
// full-attn layer's KV cache.
//
// Invariant the caller MUST uphold: `bucket_kv_len >= g_offset_int + T`, so
// every valid key column at every row is inside the bucket prefix. The mask's
// `valid_2d` predicate is computed against `arange(0, bucket)`, which is
// exactly the K range SDPA sees. Tokens past `g_offset_int + T - 1` inside the
// bucket prefix are zero-initialised in the KV pool and `-inf`-masked,
// identical to the unbucketed trace's tail behavior.
//
// `bucket_kv_len == 0` selects the full-length path (SDPA sees the full
// `max_kv_len` cache). Used for prompts that exceed the largest bucket and as
// a fallback when bucketing is disabled.
struct SparseTargetSpec {
  bool enabled = false;
  int top_k = 0;
  float temperature = 1.0f;
  float top_p = 1.0f;
  int sampler_mode = 0;
};

static std::pair<array, array> mtplx_sparse_target_rows_from_logits(
    const array& logits_flat,
    const SparseTargetSpec& spec) {
  auto sampler_logits = astype(logits_flat, mlx::core::float32);
  if (spec.temperature != 1.0f) {
    sampler_logits = mlx::core::multiply(
        sampler_logits,
        array(1.0f / spec.temperature));
  }

  int rows = sampler_logits.shape(0);
  int vocab = sampler_logits.shape(1);
  int width = std::min(spec.top_k, vocab);
  auto partitioned = argpartition(negative(sampler_logits), width - 1, -1);
  auto top_idx = slice(partitioned, {0, 0}, {rows, width});
  auto top_vals = take_along_axis(sampler_logits, top_idx, -1);
  auto order = argsort(negative(top_vals), -1);
  top_idx = take_along_axis(top_idx, order, -1);
  top_vals = take_along_axis(top_vals, order, -1);

  auto log_total = logsumexp(sampler_logits, {-1}, true);
  auto top_probs = exp(mlx::core::subtract(top_vals, log_total));
  if (spec.top_p > 0.0f && spec.top_p < 1.0f) {
    auto cumulative_before = mlx::core::subtract(cumsum(top_probs, -1), top_probs);
    auto keep = less(cumulative_before, array(spec.top_p));
    top_probs = where(keep, top_probs, mlx::core::multiply(top_probs, array(0.0f)));
  }
  auto denom = sum(top_probs, {-1}, true);
  top_probs = top_probs / denom;

  return {astype(top_idx, mlx::core::int32), astype(top_probs, mlx::core::float32)};
}

template <bool WithTape, bool ArgmaxOnly = false>
static std::vector<array> qwen35_verify_batched_decode_fn_bucketed(
    const std::vector<array>& inputs,
    int bucket_kv_len,
    SparseTargetSpec sparse_target = {}) {
  const auto& cfg = g_compile_config;

  auto h_3d       = inputs[0];          // [B, T, hidden]
  auto offset_arr = inputs[1];          // [1] int32

  int B = h_3d.shape(0);
  int T = h_3d.shape(1);

  // Tail-causal mask: shape `[1, 1, T, sdpa_kv_len]`. At query row `t`,
  // valid keys are `[0..offset + t]`. Built ONCE per layer-set; reused
  // across every full-attention layer to keep the compile graph cheap.
  //
  // `sdpa_kv_len` is `bucket_kv_len` for bucketed traces, else the full
  // cache size from the input shape (legacy path).
  int first_fa = cfg.full_attention_interval - 1;
  int max_kv_len = inputs[2 + first_fa * 2].shape(2);
  int sdpa_kv_len = bucket_kv_len > 0 ? bucket_kv_len : max_kv_len;
  auto col_positions = arange(0, sdpa_kv_len, mlx::core::int32);          // [K]
  auto row_idx       = arange(0, T, mlx::core::int32);                    // [T]
  // valid = col <= offset + row  →  reshape to [T, K] then broadcast.
  auto col_row = reshape(col_positions, {1, sdpa_kv_len});                // [1, K]
  auto row_col = reshape(row_idx, {T, 1}) + offset_arr;                   // [T, 1]
  auto valid_2d = less_equal(col_row, row_col);                           // [T, K]
  auto attn_mask = where(valid_2d,
      array(0.0f, mlx::core::bfloat16),
      array(-std::numeric_limits<float>::infinity(), mlx::core::bfloat16));
  attn_mask = reshape(attn_mask, {1, 1, T, sdpa_kv_len});                 // [1, 1, T, K]

  std::vector<array> new_caches;
  new_caches.reserve(cfg.num_layers * 2);
  for (int i = 0; i < cfg.num_layers * 2; i++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  // Tape buffers — only filled when WithTape. Per-layer slots; full-attn
  // layers stay as scalar zero placeholders so the output stride stays
  // uniform.
  std::vector<array> tape_buf, k_tape_buf, g_tape_buf, qkv_tape_buf;
  if constexpr (WithTape) {
    tape_buf.reserve(cfg.num_layers);
    k_tape_buf.reserve(cfg.num_layers);
    g_tape_buf.reserve(cfg.num_layers);
    qkv_tape_buf.reserve(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; i++) {
      tape_buf.push_back(zeros({}, mlx::core::float32));
      k_tape_buf.push_back(zeros({}, mlx::core::bfloat16));
      g_tape_buf.push_back(zeros({}, mlx::core::float32));
      qkv_tape_buf.push_back(zeros({}, mlx::core::bfloat16));
    }
  }

  array h = h_3d;
  for (int i = 0; i < cfg.num_layers; i++) {
    bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
    std::string lp = "layers." + std::to_string(i);

    auto normed = fast::rms_norm(h, get_weight(lp + ".input_layernorm.weight"),
                                 cfg.rms_norm_eps);

    array layer_out = zeros({}, mlx::core::bfloat16);
    if (is_linear) {
      const auto& cs = inputs[2 + i * 2];
      const auto& rs = inputs[2 + i * 2 + 1];
      if constexpr (WithTape) {
        auto res = gdn_batched_verify_fn_with_tape(normed, i, cs, rs, cfg);
        layer_out = std::move(res.output);
        new_caches[i * 2]     = std::move(res.conv_state);
        new_caches[i * 2 + 1] = std::move(res.recurrent_state);
        tape_buf[i]     = std::move(res.tape);
        k_tape_buf[i]   = std::move(res.k_tape);
        g_tape_buf[i]   = std::move(res.g_tape);
        qkv_tape_buf[i] = std::move(res.qkv_tape);
      } else {
        auto res = gdn_batched_verify_fn(normed, i, cs, rs, cfg);
        layer_out = std::move(res.output);
        new_caches[i * 2]     = std::move(res.conv_state);
        new_caches[i * 2 + 1] = std::move(res.recurrent_state);
      }
    } else {
      const auto& kk = inputs[2 + i * 2];
      const auto& kv = inputs[2 + i * 2 + 1];
      auto res = attn_batched_verify_fn(normed, i, kk, kv, attn_mask, offset_arr, cfg,
                                        bucket_kv_len);
      layer_out = std::move(res.output);
      new_caches[i * 2]     = std::move(res.keys);
      new_caches[i * 2 + 1] = std::move(res.values);
    }
    h = h + layer_out;

    // MLP (SwiGLU) — same math as `qwen35_decode_fn`, but flatten batch +
    // time for the linear projections.
    std::string mp = lp + ".mlp.";
    int hidden = h.shape(2);
    auto h_flat = reshape(h, {B * T, hidden});
    auto mlp_in_flat = fast::rms_norm(h_flat, get_weight(lp + ".post_attention_layernorm.weight"),
                                      cfg.rms_norm_eps);
    auto gate    = linear_proj(mlp_in_flat, mp + "gate_proj");
    auto up      = linear_proj(mlp_in_flat, mp + "up_proj");
    auto mlp_out = linear_proj(swiglu(gate, up), mp + "down_proj");
    h = h + reshape(mlp_out, {B, T, hidden});
  }

  // Final norm + LM head — operate on flat 2D, return logits 3D.
  int hidden = h.shape(2);
  auto h_flat = reshape(h, {B * T, hidden});
  h_flat = fast::rms_norm(h_flat, get_weight("final_norm.weight"), cfg.rms_norm_eps);
  // Capture pre-lm_head hidden (post-final-norm). Same point as
  // `g_last_hidden` in `qwen35_decode_fn`. Shape is `[B*T, hidden]`; reshape to
  // `[B, T, hidden]` so the FFI hands it to the Rust caller in the contract
  // documented at `mlx_qwen35_mtp_verify_compiled_with_hidden`.
  auto hidden_out = reshape(h_flat, {B, T, hidden});

  auto logits_flat = cfg.tie_word_embeddings
      ? linear_proj(h_flat, "embedding")
      : linear_proj(h_flat, "lm_head");
  array logits = zeros({}, mlx::core::float32);
  array target_argmax = zeros({}, mlx::core::int32);
  array target_sparse_ids = zeros({}, mlx::core::int32);
  array target_sparse_probs = zeros({}, mlx::core::float32);
  if (sparse_target.enabled) {
    auto sparse_rows = mtplx_sparse_target_rows_from_logits(logits_flat, sparse_target);
    target_sparse_ids = std::move(sparse_rows.first);
    target_sparse_probs = std::move(sparse_rows.second);
  } else {
    target_argmax = reshape(mlx::core::argmax(logits_flat, /*axis=*/-1), {B, T});
    if constexpr (!ArgmaxOnly) {
      int vocab = logits_flat.shape(-1);
      logits = reshape(logits_flat, {B, T, vocab});
    }
  }

  std::vector<array> result;
  size_t reserve = 3 + cfg.num_layers * 2;
  if constexpr (WithTape) reserve += cfg.num_layers * 4;
  result.reserve(reserve);
  if (sparse_target.enabled) {
    result.push_back(std::move(target_sparse_ids));
    result.push_back(std::move(target_sparse_probs));
    result.push_back(std::move(hidden_out));
  } else {
    if constexpr (!ArgmaxOnly) {
      result.push_back(std::move(logits));
    }
    result.push_back(std::move(hidden_out));
    result.push_back(std::move(target_argmax));
  }
  for (auto& c : new_caches) result.push_back(std::move(c));
  if constexpr (WithTape) {
    for (int i = 0; i < cfg.num_layers; i++) {
      result.push_back(std::move(tape_buf[i]));
      result.push_back(std::move(k_tape_buf[i]));
      result.push_back(std::move(g_tape_buf[i]));
      result.push_back(std::move(qkv_tape_buf[i]));
    }
  }
  return result;
}

// Per-bucket compiled-graph cache for the batched verify forward.
//
// One compile per (bucket, with_tape) pair, where `bucket` is the static SDPA
// key-column count baked into the trace. The dispatcher (`bucket_index_for`)
// picks the smallest bucket >= `g_offset_int + T`, so SDPA reads only that
// prefix and the mask matches its column count. Speedup is proportional to
// `bucket / max_kv_len` per SDPA call — dominant at early decode positions on
// long-context models. (Without bucketing, one trace per with_tape variant
// processes the full padded `max_kv_len` key tensor even when
// `offset + T << max_kv_len`.)
//
// Populated lazily on first use; thread-safety mirrors the other `compiled_*()`
// helpers in this file (single-mutex per turn from `DENSE_COMPILED_MUTEX` on
// the Rust side).
using BatchedVerifyFn = std::function<std::vector<array>(const std::vector<array>&)>;

// Bucket sizes. The SDPA kernel-selection boundary, NOT a power of two, drives
// this set. MLX's Metal SDPA `eval_gpu` routes the
// `q.shape(2)<=8` verify/decode attention to the single-pass
// `sdpa_vector` kernel while `k.shape(2)` is small, and to the
// hierarchical `sdpa_vector_2pass` kernel once `k.shape(2)` crosses a
// threshold. The two kernels reduce the per-row softmax in a DIFFERENT
// floating-point order, so for the verify SDPA to stay bitwise identical
// to the AR T=1 decode the verify's `k.shape(2)` (= the bucket) must land
// in the SAME kernel the AR decode hits at that context length.
//
// The AR decode (`attn_pure_fn` `dynamic_kv=true`) slices KV to the
// exact valid length `offset+1`, so its `k.shape(2)` is the real context
// length. On this target — Apple M3 Max, `get_architecture()` is
// `applegpu_g15s` so `get_architecture().back() == 's'` — the threshold
// is `k.shape(2) >= 1024`: see `scaled_dot_product_attention.cpp`
// `eval_gpu` (`(devc=='d'||devc=='s') && k.shape(2) >= 1024`). Hence the
// `1023` bucket (NOT `1024`): every verify with `offset+T <= 1023` keeps
// `k.shape(2) == 1023 < 1024` and stays on `sdpa_vector`, matching the AR
// decode (which is also `< 1024` there). The `4095` bucket is retained
// for the >1024 contexts where AR is ALSO on `sdpa_vector_2pass`: there
// both paths use the 2-pass kernel and — for the `devc=='s'` GQA shapes
// here — the same `blocks` count (128 for `1024 < N <= 8192`), so the
// 2-pass reduction is N-independent and stays bit-exact row-for-row.
static constexpr std::array<int, 6> kVerifyBuckets = {256, 512, 1023, 2048, 4095, 8192};
static constexpr int kNumVerifyBuckets = kVerifyBuckets.size();
// Slot kNumVerifyBuckets is reserved for the LEGACY full-length graph —
// used when `offset + T > 8192` or as the safety fallback when the
// bucket dispatcher is disabled via `MLX_MTP_BUCKETED_VERIFY=0`.
static constexpr int kLegacyBucketIdx = kNumVerifyBuckets;
static constexpr int kTotalBucketSlots = kNumVerifyBuckets + 1;

// 2D dispatcher table: [bucket_slot][with_tape]. Lazily populated.
// Each entry wraps `mlx::core::compile(...)` over a lambda that captures
// its `bucket_kv_len` so each lambda has unique closure identity, forcing
// MLX's compile cache to allocate a per-bucket trace.
static std::array<std::array<BatchedVerifyFn, 2>, kTotalBucketSlots>
    g_verify_compiled_by_bucket{};

static std::array<std::array<BatchedVerifyFn, 2>, kTotalBucketSlots>
    g_verify_argmax_compiled_by_bucket{};

struct SparseVerifyFnSlot {
  SparseTargetSpec spec{};
  BatchedVerifyFn fn{};
};

static std::array<std::array<SparseVerifyFnSlot, 2>, kTotalBucketSlots>
    g_verify_sparse_compiled_by_bucket{};

// Dense paged AR-decode graph. Defined here so the reload path
// `invalidate_verify_compiled_tables()` (below) can null it. Lazily compiled in
// `compiled_dense_decode_paged()` and deliberately survives the per-turn reset
// — see that getter for the full rationale.
static BatchedVerifyFn g_dense_decode_paged_compiled{};

static bool same_sparse_target_spec(const SparseTargetSpec& a,
                                    const SparseTargetSpec& b) {
  return a.enabled == b.enabled &&
         a.top_k == b.top_k &&
         a.temperature == b.temperature &&
         a.top_p == b.top_p &&
         a.sampler_mode == b.sampler_mode;
}

// Bucket dispatcher opt-out. Default ON; set `MLX_MTP_BUCKETED_VERIFY` to
// `0` / `false` / `off` (case-insensitive, surrounding whitespace ignored) to
// force the legacy single-trace path (kLegacyBucketIdx). Safety hatch only;
// the bucket path is strictly parity-safe (the tail mask is identical math).
//
// Truthy/falsy parsing mirrors the Rust `MLX_MTP_*` readers in
// `crates/mlx-core/src/models/qwen3_5/chat_common.rs` so the convention
// is uniform across the MTP knob surface.
static bool bucketed_verify_disabled() {
  static const bool disabled = []() {
    const char* raw = std::getenv("MLX_MTP_BUCKETED_VERIFY");
    if (!raw) return false;
    std::string v(raw);
    size_t s = 0;
    while (s < v.size() && std::isspace(static_cast<unsigned char>(v[s]))) s++;
    size_t e = v.size();
    while (e > s && std::isspace(static_cast<unsigned char>(v[e - 1]))) e--;
    std::string trimmed = v.substr(s, e - s);
    for (char& c : trimmed) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return trimmed == "0" || trimmed == "false" || trimmed == "off";
  }();
  return disabled;
}

// Bucket selection: smallest bucket >= `needed`. Returns kLegacyBucketIdx
// when `needed` exceeds the largest bucket (or when bucketing is disabled).
//
// `needed` is `g_offset_int + T` — the maximum key column SDPA must read.
// Asserting `bucket_size_for(needed) >= needed` is a hard correctness
// invariant.
//
// `max_kv_len` is the allocated KV-cache width. A bucket WIDER than the
// cache must never be selected: the bucketed graph would build an SDPA
// mask of `[1, 1, T, bucket]` while the cache key tensor is only
// `max_kv_len` columns, and the two cannot broadcast (crash). When the
// smallest fitting bucket exceeds `max_kv_len`, fall back to the legacy
// full-length graph, whose mask is sized to the real `max_kv_len`
// (see `qwen35_verify_batched_decode_fn_bucketed`, `bucket_kv_len == 0`).
static int bucket_index_for(int needed, int max_kv_len) {
  if (bucketed_verify_disabled()) return kLegacyBucketIdx;
  for (int i = 0; i < kNumVerifyBuckets; i++) {
    if (kVerifyBuckets[i] >= needed) {
      if (kVerifyBuckets[i] > max_kv_len) return kLegacyBucketIdx;
      return i;
    }
  }
  return kLegacyBucketIdx;
}

static int bucket_size_for_idx(int idx) {
  return idx == kLegacyBucketIdx ? 0 : kVerifyBuckets[idx];
}

// Opt-OUT gate for the paged-pool MTP verify graph. Default ON. Set
// `MLX_MTP_VERIFY_PAGED_ATTN` to `0` / `false` / `off` (case-insensitive,
// surrounding whitespace ignored) to fall back to the dense BHTD verify path.
// Mirrored in Rust by `chat_common::mtp_verify_paged_attn_enabled()`.
static bool mtp_verify_paged_attn_enabled() {
  static const bool enabled = []() {
    const char* raw = std::getenv("MLX_MTP_VERIFY_PAGED_ATTN");
    if (!raw) return true;
    std::string v(raw);
    size_t s = 0;
    while (s < v.size() && std::isspace(static_cast<unsigned char>(v[s]))) s++;
    size_t e = v.size();
    while (e > s && std::isspace(static_cast<unsigned char>(v[e - 1]))) e--;
    std::string trimmed = v.substr(s, e - s);
    for (char& c : trimmed) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return !(trimmed == "0" || trimmed == "false" || trimmed == "off");
  }();
  return enabled;
}

// Build (and cache) the compiled function for a given (bucket, with_tape).
// The lambda capture of `bucket_size` (an `int`) gives each closure
// unique identity — MLX's compile cache then allocates a per-bucket trace.
static BatchedVerifyFn& get_or_compile_verify_bucket(int bucket_idx, bool with_tape) {
  auto& slot = g_verify_compiled_by_bucket[bucket_idx][with_tape ? 1 : 0];
  if (!slot) {
    int bucket_size = bucket_size_for_idx(bucket_idx);
    if (with_tape) {
      slot = mlx::core::compile([bucket_size](const std::vector<array>& inputs) {
        return qwen35_verify_batched_decode_fn_bucketed<true>(inputs, bucket_size);
      });
    } else {
      slot = mlx::core::compile([bucket_size](const std::vector<array>& inputs) {
        return qwen35_verify_batched_decode_fn_bucketed<false>(inputs, bucket_size);
      });
    }
  }
  return slot;
}

static BatchedVerifyFn& get_or_compile_verify_argmax_bucket(int bucket_idx, bool with_tape) {
  auto& slot = g_verify_argmax_compiled_by_bucket[bucket_idx][with_tape ? 1 : 0];
  if (!slot) {
    int bucket_size = bucket_size_for_idx(bucket_idx);
    if (with_tape) {
      slot = mlx::core::compile([bucket_size](const std::vector<array>& inputs) {
        return qwen35_verify_batched_decode_fn_bucketed<true, true>(inputs, bucket_size);
      });
    } else {
      slot = mlx::core::compile([bucket_size](const std::vector<array>& inputs) {
        return qwen35_verify_batched_decode_fn_bucketed<false, true>(inputs, bucket_size);
      });
    }
  }
  return slot;
}

static BatchedVerifyFn& get_or_compile_verify_sparse_bucket(
    int bucket_idx,
    bool with_tape,
    const SparseTargetSpec& spec) {
  auto& slot = g_verify_sparse_compiled_by_bucket[bucket_idx][with_tape ? 1 : 0];
  if (!slot.fn || !same_sparse_target_spec(slot.spec, spec)) {
    slot.spec = spec;
    int bucket_size = bucket_size_for_idx(bucket_idx);
    if (with_tape) {
      slot.fn = mlx::core::compile([bucket_size, spec](const std::vector<array>& inputs) {
        return qwen35_verify_batched_decode_fn_bucketed<true>(inputs, bucket_size, spec);
      });
    } else {
      slot.fn = mlx::core::compile([bucket_size, spec](const std::vector<array>& inputs) {
        return qwen35_verify_batched_decode_fn_bucketed<false>(inputs, bucket_size, spec);
      });
    }
  }
  return slot.fn;
}

// Paged-pool MTP verify graph. Sibling of
// `qwen35_verify_batched_decode_fn_bucketed` that reads K/V from the vLLM-style
// pool instead of the BHTD `[B, Hkv, max_kv_len, D]` cache. Linear-attention
// layers are unchanged; full-attention layers route through
// `attn_batched_verify_fn_paged` (paged_kv_write + paged_attention_varlen).
//
// Input vector layout:
//   [0]                      h_3d            [1, T, hidden]              bf16
//   [1]                      offset_arr      [1]                          int32
//   [2]                      block_table     [1, max_blocks_per_seq]      int32
//   [3]                      slot_mapping    [chunk_size_max]             int64
//   [4]                      seq_lens        [1]                          int32
//   [5]                      cu_seqlens_q    [2]                          int32
//   [6 .. 6 + 4N):           per-layer (stride 4):
//     linear:    (conv_state, recurrent_state, _placeholder_, _placeholder_)
//     full-attn: (k_pool,     v_pool,          k_scale,        v_scale)
//
// Output vector layout:
//   [0]                      logits          [1, T, vocab]
//   [1]                      hiddens         [1, T, hidden]
//   [2]                      target argmax   [1, T] int32
//   [3 .. 3 + 2N):           per-layer (stride 2):
//     linear:    (new_conv_state, new_recurrent_state)
//     full-attn: (new_k_pool,     new_v_pool)
//   [3 + 2N ..]              tape outputs (WithTape variant only) per linear layer
template <bool WithTape>
static std::vector<array> qwen35_verify_batched_decode_fn_paged(
    const std::vector<array>& inputs) {
  // Read layout from the paged config (`mlx_qwen35_init_paged`). The verify
  // graph is only callable through the paged-MTP gate, which requires
  // `g_dense_paged_inited == true`; BHTD `g_compile_config` may be unset on
  // pure-paged turns.
  const auto& cfg = g_dense_paged_config;

  auto h_3d         = inputs[0];
  auto offset_arr   = inputs[1];
  auto block_table  = inputs[2];
  auto slot_mapping = inputs[3];
  auto seq_lens     = inputs[4];
  auto cu_seqlens_q = inputs[5];

  constexpr int kHeader = 6;
  constexpr int kPerLayer = 4;
  constexpr int BLOCK_SIZE = 16;

  int B = h_3d.shape(0);
  int T = h_3d.shape(1);

  std::vector<array> new_caches;
  new_caches.reserve(cfg.num_layers * 2);
  for (int i = 0; i < cfg.num_layers * 2; i++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  std::vector<array> tape_buf, k_tape_buf, g_tape_buf, qkv_tape_buf;
  if constexpr (WithTape) {
    tape_buf.reserve(cfg.num_layers);
    k_tape_buf.reserve(cfg.num_layers);
    g_tape_buf.reserve(cfg.num_layers);
    qkv_tape_buf.reserve(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; i++) {
      tape_buf.push_back(zeros({}, mlx::core::float32));
      k_tape_buf.push_back(zeros({}, mlx::core::bfloat16));
      g_tape_buf.push_back(zeros({}, mlx::core::float32));
      qkv_tape_buf.push_back(zeros({}, mlx::core::bfloat16));
    }
  }

  array h = h_3d;
  for (int i = 0; i < cfg.num_layers; i++) {
    bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
    std::string lp = "layers." + std::to_string(i);

    auto normed = fast::rms_norm(h, get_weight(lp + ".input_layernorm.weight"),
                                 cfg.rms_norm_eps);

    int base = kHeader + i * kPerLayer;
    array layer_out = zeros({}, mlx::core::bfloat16);
    if (is_linear) {
      const auto& cs = inputs[base + 0];
      const auto& rs = inputs[base + 1];
      if constexpr (WithTape) {
        auto res = gdn_batched_verify_fn_with_tape(normed, i, cs, rs, cfg);
        layer_out = std::move(res.output);
        new_caches[i * 2]     = std::move(res.conv_state);
        new_caches[i * 2 + 1] = std::move(res.recurrent_state);
        tape_buf[i]     = std::move(res.tape);
        k_tape_buf[i]   = std::move(res.k_tape);
        g_tape_buf[i]   = std::move(res.g_tape);
        qkv_tape_buf[i] = std::move(res.qkv_tape);
      } else {
        auto res = gdn_batched_verify_fn(normed, i, cs, rs, cfg);
        layer_out = std::move(res.output);
        new_caches[i * 2]     = std::move(res.conv_state);
        new_caches[i * 2 + 1] = std::move(res.recurrent_state);
      }
    } else {
      const auto& k_pool  = inputs[base + 0];
      const auto& v_pool  = inputs[base + 1];
      const auto& k_scale = inputs[base + 2];
      const auto& v_scale = inputs[base + 3];
      auto res = attn_batched_verify_fn_paged(
          normed, i,
          k_pool, v_pool,
          k_scale, v_scale,
          offset_arr,
          block_table, slot_mapping,
          seq_lens, cu_seqlens_q,
          BLOCK_SIZE,
          cfg);
      layer_out = std::move(res.output);
      new_caches[i * 2]     = std::move(res.keys);
      new_caches[i * 2 + 1] = std::move(res.values);
    }
    h = h + layer_out;

    std::string mp = lp + ".mlp.";
    int hidden = h.shape(2);
    auto h_flat = reshape(h, {B * T, hidden});
    auto mlp_in_flat = fast::rms_norm(h_flat, get_weight(lp + ".post_attention_layernorm.weight"),
                                      cfg.rms_norm_eps);
    auto gate    = linear_proj(mlp_in_flat, mp + "gate_proj");
    auto up      = linear_proj(mlp_in_flat, mp + "up_proj");
    auto mlp_out = linear_proj(swiglu(gate, up), mp + "down_proj");
    h = h + reshape(mlp_out, {B, T, hidden});
  }

  int hidden = h.shape(2);
  auto h_flat = reshape(h, {B * T, hidden});
  h_flat = fast::rms_norm(h_flat, get_weight("final_norm.weight"), cfg.rms_norm_eps);
  auto hidden_out = reshape(h_flat, {B, T, hidden});

  auto logits_flat = cfg.tie_word_embeddings
      ? linear_proj(h_flat, "embedding")
      : linear_proj(h_flat, "lm_head");
  auto target_argmax = reshape(mlx::core::argmax(logits_flat, /*axis=*/-1), {B, T});
  int vocab = logits_flat.shape(-1);
  auto logits = reshape(logits_flat, {B, T, vocab});

  std::vector<array> result;
  size_t reserve = 3 + cfg.num_layers * 2;
  if constexpr (WithTape) reserve += cfg.num_layers * 4;
  result.reserve(reserve);
  result.push_back(std::move(logits));
  result.push_back(std::move(hidden_out));
  result.push_back(std::move(target_argmax));
  for (auto& c : new_caches) result.push_back(std::move(c));
  if constexpr (WithTape) {
    for (int i = 0; i < cfg.num_layers; i++) {
      result.push_back(std::move(tape_buf[i]));
      result.push_back(std::move(k_tape_buf[i]));
      result.push_back(std::move(g_tape_buf[i]));
      result.push_back(std::move(qkv_tape_buf[i]));
    }
  }
  return result;
}

static std::array<BatchedVerifyFn, 2> g_verify_compiled_paged{};

static BatchedVerifyFn& get_or_compile_verify_paged(bool with_tape) {
  auto& slot = g_verify_compiled_paged[with_tape ? 1 : 0];
  if (!slot) {
    // Route through the capturing-lambda helper so the baked-weight tape gets
    // a UNIQUE erasable fun_id; a captureless lambda would decay to a stable
    // free-fn address with no eviction hook and the reload-null would no-op.
    if (with_tape) {
      slot = compile_resettable_weight_graph(&qwen35_verify_batched_decode_fn_paged<true>);
    } else {
      slot = compile_resettable_weight_graph(&qwen35_verify_batched_decode_fn_paged<false>);
    }
  }
  return slot;
}

// Null EVERY MTP-verify dispatch slot so the next `get_or_compile_verify_*`
// re-traces against the CURRENT weight registry.
//
// The compiled verify bodies read weights via `get_weight(...)` INSIDE the
// traced closure (NOT as compile inputs), so `mlx::core::compile(...)` bakes
// the captured weight arrays into the cached tape and reuses them verbatim
// on every later call. These tables are PROCESS-WIDE and deliberately
// survive the per-turn `mlx_qwen35_compiled_reset()` (cross-turn reuse). On
// a model RELOAD (weights swapped under the same `model_id` gate) the baked
// weights are stale, so a second same-shape model would verify with the
// first model's weights — silent corruption.
//
// Assigning an empty `std::function{}` destroys the wrapper, frees its
// `fun_id`, and forces a NEW wrapper (new `fun_id`) to be built on the next
// call — which re-traces against the live registry on whatever thread next
// calls `get_or_compile_*`. This is correct regardless of thread because the
// re-trace is lazy on the inference thread; we do NOT rely on cross-thread
// `compile_clear_cache()` (MLX's CompilerCache is thread_local).
//
// MUST be called under the Rust `COMPILED_WEIGHTS_RWLOCK.write()` so it is
// serialized against in-flight compiled reads.
static void invalidate_verify_compiled_tables() {
  for (auto& by_tape : g_verify_compiled_by_bucket) {
    by_tape[0] = BatchedVerifyFn{};
    by_tape[1] = BatchedVerifyFn{};
  }
  for (auto& by_tape : g_verify_argmax_compiled_by_bucket) {
    by_tape[0] = BatchedVerifyFn{};
    by_tape[1] = BatchedVerifyFn{};
  }
  for (auto& by_tape : g_verify_sparse_compiled_by_bucket) {
    by_tape[0].fn = BatchedVerifyFn{};
    by_tape[0].spec = SparseTargetSpec{};
    by_tape[1].fn = BatchedVerifyFn{};
    by_tape[1].spec = SparseTargetSpec{};
  }
  g_verify_compiled_paged[0] = BatchedVerifyFn{};
  g_verify_compiled_paged[1] = BatchedVerifyFn{};
  // Dense paged AR-decode graph (also a weight-baking graph; compiled via the
  // capturing-lambda helper so this null actually erases the baked tape).
  g_dense_decode_paged_compiled = BatchedVerifyFn{};
}

}  // namespace

// =====================================================================
// Full-graph paged-decode compile function.
//
// Mirrors `moe_compiled_decode_fn_paged` from `mlx_qwen35_moe.cpp` but without
// expert routing — every layer either runs `gdn_pure_fn` (linear) or
// `attn_for_compile_paged` (full attention) followed by a dense SwiGLU MLP.
//
// Unlike the flat `qwen35_decode_fn` path (which captures a scalar
// `g_offset_int` and therefore invalidates the compile cache on every step),
// this graph takes the offset as an array input (`offset_arr`). All shapes are
// fixed at trace time, so `mlx::core::compile(...)` produces a cache key that
// stays valid across all decode steps within one turn.
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
//   [2]                  hidden_for_export:  [B, hidden] post-final-norm,
//                                            pre-lm_head — consumed by
//                                            the paged-MTP gate via
//                                            `mlx_qwen35_export_last_hidden_paged`
//   For each layer i:
//     If linear:
//       [3 + i*2 + 0]    new_conv_state
//       [3 + i*2 + 1]    new_recurrent_state
//     If full-attention:
//       [3 + i*2 + 0]    new_k_pool          (post-write pool tensor)
//       [3 + i*2 + 1]    new_v_pool
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

  // Hard-coded contract (matches the MoE paged graph).
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

  // Final norm + LM head. Capture the post-final-norm tensor BEFORE the
  // LM head projection so the paged-MTP gate can seed the next MTP draft
  // cycle from the live hidden state without a second main forward.
  h = fast::rms_norm(h, get_weight("final_norm.weight"), cfg.rms_norm_eps);
  auto hidden_for_export = h;
  if (cfg.tie_word_embeddings) {
    h = linear_proj(h, "embedding");
  } else {
    h = linear_proj(h, "lm_head");
  }

  auto new_offset = offset_arr + array(1, mlx::core::int32);

  std::vector<array> result;
  result.reserve(3 + cfg.num_layers * 2);
  result.push_back(std::move(h));
  result.push_back(std::move(new_offset));
  result.push_back(std::move(hidden_for_export));
  for (auto& c : new_caches) result.push_back(std::move(c));
  return result;
}

// Dense paged AR-decode graph. `dense_compiled_decode_fn_paged` reads weights
// via `get_weight(...)` inside the trace, so the weights are baked into the
// cached tape. Backed by the resettable FILE-SCOPE global
// `g_dense_decode_paged_compiled` (declared near the verify-bucket tables) and
// compiled through `compile_resettable_weight_graph` so the reload path
// (`invalidate_verify_compiled_tables()`) can null it and force a re-trace
// against the live registry. Deliberately survives the per-turn
// `mlx_qwen35_compiled_reset()` (cross-turn reuse); nulled ONLY on reload.
static BatchedVerifyFn& compiled_dense_decode_paged() {
  if (!g_dense_decode_paged_compiled) {
    g_dense_decode_paged_compiled =
        compile_resettable_weight_graph(dense_compiled_decode_fn_paged);
  }
  return g_dense_decode_paged_compiled;
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
  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_compiled_init_from_prefill: ENTER "
            "weight_count=%zu layer_count=%d max_kv_len=%d batch_size=%d "
            "prefill_offset=%d hidden_size=%d num_heads=%d num_kv_heads=%d\n",
            qwen35_common::g_weights().size(), num_layers, max_kv_len,
            batch_size, prefill_offset, hidden_size, num_heads, num_kv_heads);
  }
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

    // Size the per-layer tape accumulator vectors. They remain empty (each
    // slot is `std::nullopt`) until `mlx_qwen35_compiled_tape_arm` is called
    // and `qwen35_decode_fn` records the first step.
    g_gdn_tape_acc.assign(num_layers, std::nullopt);
    g_gdn_k_tape_acc.assign(num_layers, std::nullopt);
    g_gdn_g_tape_acc.assign(num_layers, std::nullopt);
    g_gdn_qkv_tape_acc.assign(num_layers, std::nullopt);
    g_tape_recording_armed = false;

    // Break the lazy RNG split chain from model initialization.
    auto rng_key = mlx::core::random::KeySequence::default_().next();
    mlx::core::eval({rng_key});
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_compiled_init_from_prefill: EXIT OK "
              "compiled_caches=%zu offset=%d\n",
              g_compiled_caches.size(), g_offset_int);
    }
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

// -----------------------------------------------------------------------------
// Batched verify forward.
//
// Runs ONE compiled forward over `T = depth + 1` tokens (vs D+1 sequential
// `mlx_qwen35_forward_compiled` calls). The compiled graph emits
// `[1, T, vocab]` logits + `[1, T, hidden]` post-final-norm hiddens in one
// dispatch.
//
// Tape integration: if `g_tape_recording_armed` is true at entry, the with-tape
// variant is invoked and the per-layer tape buffers
// `(tape, k_tape, g_tape, qkv_tape)` of shape `[1, T, ...]` are stashed
// DIRECTLY into the `g_gdn_*_tape_acc` slots. The rollback path consumes a
// `slice([:, 0..K, ...])` of these — the cumulative shape is identical.
//
// Inputs:
//   - input_ids:        `[1, T]` int32 tokens (T = depth + 1).
//   - embedding_weight: model's embedding table (or LM-head if untied).
//   - depth:            T = depth + 1; depth ∈ [1, 5] enforced by caller.
// Outputs (heap-allocated, caller owns):
//   - out_logits:       `[1, T, vocab]` bf16 logits.
//   - out_hiddens:      `[1, T, hidden_size]` bf16 post-final-norm.
// Side effects:
//   - `g_offset_int` += T
//   - `g_compiled_caches[]` updated in place with the post-verify state
//     (T new K/V slots written for full-attn layers; recurrent + conv
//     state advanced T steps for linear-attention layers).
//   - When `g_tape_recording_armed`: per-layer accumulators populated.
//
// Returns nullptrs on failure. The caller MUST hold `DENSE_COMPILED_MUTEX`
// and the `COMPILED_WEIGHTS_RWLOCK` read guard.
// -----------------------------------------------------------------------------
void mlx_qwen35_forward_batched_verify(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_logits,
    mlx_array** out_hiddens,
    mlx_array** out_argmax
) {
  if (out_logits) *out_logits = nullptr;
  if (out_hiddens) *out_hiddens = nullptr;
  if (out_argmax) *out_argmax = nullptr;
  if (!input_ids_ptr || !embedding_weight_ptr || !out_logits || !out_hiddens) {
    return;
  }
  if (!g_compile_inited) return;
  if (depth < 1 || depth > 5) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_forward_batched_verify: depth %d outside [1, 5]\n",
            depth);
    fflush(stderr);
    return;
  }
  const auto& cfg = g_compile_config;
  int T = depth + 1;

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_forward_batched_verify: ENTER depth=%d "
            "T=%d (input_ids=[1,%d]) offset=%d (RoPE base; causal mask) "
            "tape_armed=%d\n",
            depth, T, T, g_offset_int, g_tape_recording_armed ? 1 : 0);
  }

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    // Validate input_ids shape `[1, T]`.
    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != T) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_forward_batched_verify: input_ids shape must be "
              "[1, %d], got ndim=%d shape=[%lld,%lld]\n",
              T, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    // Embed: `[1, T]` int32 → `[1, T, hidden]` bf16.
    auto flat_ids = reshape(input_ids, {-1});               // [T]
    auto emb_flat = take(embedding_weight, flat_ids, 0);    // [T, hidden]
    auto h_3d = reshape(emb_flat, {1, T, cfg.hidden_size}); // [1, T, hidden]

    std::vector<array> inputs;
    inputs.reserve(2 + cfg.num_layers * 2);
    inputs.push_back(std::move(h_3d));
    inputs.push_back(reshape(array(g_offset_int, mlx::core::int32), {1}));
    for (const auto& c : g_compiled_caches) {
      inputs.push_back(c);
    }

    bool with_tape = g_tape_recording_armed;
    if (mtp_verify_paged_attn_enabled() && g_dense_paged_inited) {
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true)) {
        fprintf(stderr,
                "[MLX] MLX_MTP_VERIFY_PAGED_ATTN=1 with paged adapter live, "
                "but no Rust caller wires the paged-verify inputs yet — "
                "falling back to BHTD verify path. (Phase 4b/B1 scaffolding.)\n");
        fflush(stderr);
      }
    }
    // Pick the smallest bucket >= offset + T. The legacy full-length graph is
    // returned when offset + T exceeds the largest bucket, when the chosen
    // bucket would exceed the allocated KV cache width, or when the dispatcher
    // is disabled via env var.
    int bucket_idx = bucket_index_for(g_offset_int + T, cfg.max_kv_len);
    auto& fn = get_or_compile_verify_bucket(bucket_idx, with_tape);
    auto outputs = fn(inputs);

    // outputs[0]: logits  [1, T, vocab]
    // outputs[1]: hiddens [1, T, hidden]
    // outputs[2]: target argmax [1, T] int32
    // outputs[3 .. 3+2N): updated caches (N = num_layers)
    // If with_tape: outputs[3+2N ..] hold per-layer (tape, k_tape, g_tape, qkv_tape)
    //
    // Stage allocations into locals first: if `new array(...)` throws on
    // the SECOND call (`std::bad_alloc` under OOM) we'd otherwise leak the
    // first heap `array` already written to `*out_logits`. Only commit to
    // the out-pointers after both allocations succeed.
    array* logits_alloc  = new array(outputs[0]);
    array* hiddens_alloc = nullptr;
    array* argmax_alloc = nullptr;
    try {
      hiddens_alloc = new array(outputs[1]);
      if (out_argmax) {
        argmax_alloc = new array(outputs[2]);
      }
    } catch (...) {
      delete logits_alloc;
      delete hiddens_alloc;
      throw;
    }
    *out_logits  = reinterpret_cast<mlx_array*>(logits_alloc);
    *out_hiddens = reinterpret_cast<mlx_array*>(hiddens_alloc);
    if (out_argmax) {
      *out_argmax = reinterpret_cast<mlx_array*>(argmax_alloc);
    }

    // Update KV / linear caches in place.
    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_compiled_caches[i] = outputs[3 + i];
    }

    // Advance offset by T (one per token in the batch).
    g_offset_int += T;

    // Stash tape outputs into the per-layer accumulators. The batched graph
    // emits `[1, T, ...]` natively, so each slot gets a SINGLE assignment — no
    // `concatenate` loop. The slot lifetime / shape contract matches a per-step
    // append producing `[1, recorded_steps, ...]` via repeated
    // `concatenate(slot, step)` on axis 1.
    if (with_tape) {
      int extra_base = 3 + cfg.num_layers * 2;
      for (int i = 0; i < cfg.num_layers; i++) {
        bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
        if (!is_linear) continue;
        int base = extra_base + i * 4;
        g_gdn_tape_acc[i]     = outputs[base + 0];
        g_gdn_k_tape_acc[i]   = outputs[base + 1];
        g_gdn_g_tape_acc[i]   = outputs[base + 2];
        g_gdn_qkv_tape_acc[i] = outputs[base + 3];
      }
    }
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_forward_batched_verify: EXIT OK T=%d "
              "bucket_idx=%d with_tape=%d new_offset=%d\n",
              T, bucket_idx, with_tape ? 1 : 0, g_offset_int);
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_forward_batched_verify: %s\n",
            e.what());
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_forward_batched_verify\n");
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
  }
}

void mlx_qwen35_forward_batched_verify_argmax_only(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_hiddens,
    mlx_array** out_argmax
) {
  if (out_hiddens) *out_hiddens = nullptr;
  if (out_argmax) *out_argmax = nullptr;
  if (!input_ids_ptr || !embedding_weight_ptr || !out_hiddens || !out_argmax) {
    return;
  }
  if (!g_compile_inited) return;
  if (depth < 1 || depth > 5) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_forward_batched_verify_argmax_only: depth %d outside [1, 5]\n",
            depth);
    fflush(stderr);
    return;
  }
  const auto& cfg = g_compile_config;
  int T = depth + 1;

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_forward_batched_verify_argmax_only: ENTER depth=%d "
            "T=%d (input_ids=[1,%d]) offset=%d (RoPE base; causal mask) "
            "tape_armed=%d\n",
            depth, T, T, g_offset_int, g_tape_recording_armed ? 1 : 0);
  }

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != T) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_forward_batched_verify_argmax_only: input_ids shape must be "
              "[1, %d], got ndim=%d shape=[%lld,%lld]\n",
              T, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    auto flat_ids = reshape(input_ids, {-1});
    auto emb_flat = take(embedding_weight, flat_ids, 0);
    auto h_3d = reshape(emb_flat, {1, T, cfg.hidden_size});

    std::vector<array> inputs;
    inputs.reserve(2 + cfg.num_layers * 2);
    inputs.push_back(std::move(h_3d));
    inputs.push_back(reshape(array(g_offset_int, mlx::core::int32), {1}));
    for (const auto& c : g_compiled_caches) {
      inputs.push_back(c);
    }

    bool with_tape = g_tape_recording_armed;
    if (mtp_verify_paged_attn_enabled() && g_dense_paged_inited) {
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true)) {
        fprintf(stderr,
                "[MLX] MLX_MTP_VERIFY_PAGED_ATTN=1 with paged adapter live, "
                "but no Rust caller wires the paged-verify inputs yet — "
                "falling back to BHTD verify path. (Phase 4b/B1 scaffolding.)\n");
        fflush(stderr);
      }
    }

    int bucket_idx = bucket_index_for(g_offset_int + T, cfg.max_kv_len);
    auto& fn = get_or_compile_verify_argmax_bucket(bucket_idx, with_tape);
    auto outputs = fn(inputs);

    // outputs[0]: hiddens [1, T, hidden]
    // outputs[1]: target argmax [1, T] int32
    // outputs[2 .. 2+2N): updated caches (N = num_layers)
    // If with_tape: outputs[2+2N ..] hold per-layer (tape, k_tape, g_tape, qkv_tape)
    array* hiddens_alloc = new array(outputs[0]);
    array* argmax_alloc = nullptr;
    try {
      argmax_alloc = new array(outputs[1]);
    } catch (...) {
      delete hiddens_alloc;
      delete argmax_alloc;
      throw;
    }
    *out_hiddens = reinterpret_cast<mlx_array*>(hiddens_alloc);
    *out_argmax = reinterpret_cast<mlx_array*>(argmax_alloc);

    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_compiled_caches[i] = outputs[2 + i];
    }

    g_offset_int += T;

    if (with_tape) {
      int extra_base = 2 + cfg.num_layers * 2;
      for (int i = 0; i < cfg.num_layers; i++) {
        bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
        if (!is_linear) continue;
        int base = extra_base + i * 4;
        g_gdn_tape_acc[i]     = outputs[base + 0];
        g_gdn_k_tape_acc[i]   = outputs[base + 1];
        g_gdn_g_tape_acc[i]   = outputs[base + 2];
        g_gdn_qkv_tape_acc[i] = outputs[base + 3];
      }
    }
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_forward_batched_verify_argmax_only: EXIT OK T=%d "
              "bucket_idx=%d with_tape=%d new_offset=%d\n",
              T, bucket_idx, with_tape ? 1 : 0, g_offset_int);
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_forward_batched_verify_argmax_only: %s\n",
            e.what());
    fflush(stderr);
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_argmax) *out_argmax = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_forward_batched_verify_argmax_only\n");
    fflush(stderr);
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_argmax) *out_argmax = nullptr;
  }
}

// Stochastic MTP verifier sibling that returns compact target sparse rows
// instead of full verifier logits.
void mlx_qwen35_forward_batched_verify_sparse_target(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    float temperature,
    int top_k,
    float top_p,
    int sampler_mode,
    mlx_array** out_hiddens,
    mlx_array** out_target_ids,
    mlx_array** out_target_probs
) {
  if (out_hiddens) *out_hiddens = nullptr;
  if (out_target_ids) *out_target_ids = nullptr;
  if (out_target_probs) *out_target_probs = nullptr;
  if (!input_ids_ptr || !embedding_weight_ptr || !out_hiddens ||
      !out_target_ids || !out_target_probs) {
    return;
  }
  if (!g_compile_inited) return;
  if (depth < 1 || depth > 5) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_forward_batched_verify_sparse_target: depth %d outside [1, 5]\n",
            depth);
    fflush(stderr);
    return;
  }
  if (!std::isfinite(temperature) || temperature <= 0.0f || top_k <= 0 ||
      sampler_mode != 1) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_forward_batched_verify_sparse_target: unsupported sparse spec "
            "temperature=%g top_k=%d sampler_mode=%d\n",
            (double)temperature, top_k, sampler_mode);
    fflush(stderr);
    return;
  }
  const auto& cfg = g_compile_config;
  int T = depth + 1;

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != T) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_forward_batched_verify_sparse_target: input_ids shape must be "
              "[1, %d], got ndim=%d shape=[%lld,%lld]\n",
              T, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    int vocab = embedding_weight.shape(0);
    SparseTargetSpec spec{};
    spec.enabled = true;
    spec.top_k = std::min(top_k, vocab);
    spec.temperature = temperature;
    spec.top_p = top_p;
    spec.sampler_mode = sampler_mode;
    if (spec.top_k <= 0) return;

    auto flat_ids = reshape(input_ids, {-1});
    auto emb_flat = take(embedding_weight, flat_ids, 0);
    auto h_3d = reshape(emb_flat, {1, T, cfg.hidden_size});

    std::vector<array> inputs;
    inputs.reserve(2 + cfg.num_layers * 2);
    inputs.push_back(std::move(h_3d));
    inputs.push_back(reshape(array(g_offset_int, mlx::core::int32), {1}));
    for (const auto& c : g_compiled_caches) {
      inputs.push_back(c);
    }

    bool with_tape = g_tape_recording_armed;
    int bucket_idx = bucket_index_for(g_offset_int + T, cfg.max_kv_len);
    auto& fn = get_or_compile_verify_sparse_bucket(bucket_idx, with_tape, spec);
    auto outputs = fn(inputs);

    array* ids_alloc = new array(outputs[0]);
    array* probs_alloc = nullptr;
    array* hiddens_alloc = nullptr;
    try {
      probs_alloc = new array(outputs[1]);
      hiddens_alloc = new array(outputs[2]);
    } catch (...) {
      delete ids_alloc;
      delete probs_alloc;
      delete hiddens_alloc;
      throw;
    }
    *out_target_ids = reinterpret_cast<mlx_array*>(ids_alloc);
    *out_target_probs = reinterpret_cast<mlx_array*>(probs_alloc);
    *out_hiddens = reinterpret_cast<mlx_array*>(hiddens_alloc);

    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_compiled_caches[i] = outputs[3 + i];
    }
    g_offset_int += T;

    if (with_tape) {
      int extra_base = 3 + cfg.num_layers * 2;
      for (int i = 0; i < cfg.num_layers; i++) {
        bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
        if (!is_linear) continue;
        int base = extra_base + i * 4;
        g_gdn_tape_acc[i]     = outputs[base + 0];
        g_gdn_k_tape_acc[i]   = outputs[base + 1];
        g_gdn_g_tape_acc[i]   = outputs[base + 2];
        g_gdn_qkv_tape_acc[i] = outputs[base + 3];
      }
    }
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_forward_batched_verify_sparse_target: EXIT OK "
              "T=%d bucket_idx=%d top_k=%d with_tape=%d new_offset=%d\n",
              T, bucket_idx, spec.top_k, with_tape ? 1 : 0, g_offset_int);
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_forward_batched_verify_sparse_target: %s\n",
            e.what());
    fflush(stderr);
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_target_ids) *out_target_ids = nullptr;
    if (out_target_probs) *out_target_probs = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_forward_batched_verify_sparse_target\n");
    fflush(stderr);
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_target_ids) *out_target_ids = nullptr;
    if (out_target_probs) *out_target_probs = nullptr;
  }
}

// Paged-pool sibling of `mlx_qwen35_forward_batched_verify`. Reads K/V from
// `g_dense_k_pools[]` / `g_dense_v_pools[]` (populated by the paged adapter
// through `mlx_qwen35_init_paged`) instead of the BHTD `g_compiled_caches[]`.
// Linear-attention layers still source state from
// `g_dense_paged_linear_caches[]`.
//
// Caller MUST construct: `offset_arr` ([1] int32), `block_table`
// ([1, max_blocks_per_seq] int32), `slot_mapping` ([chunk_size_max]
// int64), `seq_lens` ([1] int32, post-write context), `cu_seqlens_q`
// ([2] int32 = [0, T] where T = depth + 1). The paged adapter's
// `build_paged_attention_inputs(D+1, ...)` already emits the first
// four; `cu_seqlens_q` is trivially constructible Rust-side.
//
// Output: `*out_logits` ← `[1, T, vocab]`, `*out_hiddens`
// ← `[1, T, hidden]`, `*out_argmax` ← `[1, T]`. All outputs are
// heap-allocated via `new array(...)`; the caller owns them and is
// responsible for `MxArray::from_handle`. On error all pointers are set
// to nullptr and a stderr diagnostic is emitted; global state (pools,
// linear caches, BHTD `g_offset_int`) is left untouched so the Rust
// caller can fall back.
//
// Tape recording follows the same arming as the BHTD path
// (`g_tape_recording_armed`). Tape outputs flow back into the same
// `g_gdn_*_tape_acc[]` per-layer accumulators so the existing replay
// path is reused.
//
// NB: this FFI does NOT advance any BHTD cursor (`g_offset_int` /
// `g_compiled_caches`). The paged adapter's `record_tokens` is the
// authoritative cursor for the pool, mutated Rust-side around the FFI.
void mlx_qwen35_forward_batched_verify_paged(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array* offset_arr_ptr,
    mlx_array* block_table_ptr,
    mlx_array* slot_mapping_ptr,
    mlx_array* seq_lens_ptr,
    mlx_array* cu_seqlens_q_ptr,
    mlx_array** out_logits,
    mlx_array** out_hiddens,
    mlx_array** out_argmax
) {
  if (out_logits) *out_logits = nullptr;
  if (out_hiddens) *out_hiddens = nullptr;
  if (out_argmax) *out_argmax = nullptr;
  if (!input_ids_ptr || !embedding_weight_ptr || !offset_arr_ptr ||
      !block_table_ptr || !slot_mapping_ptr || !seq_lens_ptr ||
      !cu_seqlens_q_ptr || !out_logits || !out_hiddens || !out_argmax) {
    return;
  }
  if (!g_dense_paged_inited) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_forward_batched_verify_paged: paged adapter "
            "not initialised (g_dense_paged_inited=false); call "
            "mlx_qwen35_init_paged first.\n");
    fflush(stderr);
    return;
  }
  if (depth < 1 || depth > 5) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_forward_batched_verify_paged: depth %d outside "
            "[1, 5]\n",
            depth);
    fflush(stderr);
    return;
  }
  // The paged verify graph reads pool / linear-cache layout from
  // `g_dense_paged_config` (set by `mlx_qwen35_init_paged`). The BHTD
  // `g_compile_config` is unset when callers reach this FFI from the paged-MTP
  // gate, so source the dimensions from the paged config instead. The two
  // configs share an identical layout for the fields this code reads
  // (`num_layers`, `hidden_size`, `full_attention_interval`).
  const auto& cfg = g_dense_paged_config;
  int T = depth + 1;

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_forward_batched_verify_paged: ENTER "
            "depth=%d T=%d tape_armed=%d\n",
            depth, T, g_tape_recording_armed ? 1 : 0);
  }

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);
    auto& offset_arr       = *reinterpret_cast<array*>(offset_arr_ptr);
    auto& block_table      = *reinterpret_cast<array*>(block_table_ptr);
    auto& slot_mapping     = *reinterpret_cast<array*>(slot_mapping_ptr);
    auto& seq_lens         = *reinterpret_cast<array*>(seq_lens_ptr);
    auto& cu_seqlens_q     = *reinterpret_cast<array*>(cu_seqlens_q_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != T) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_forward_batched_verify_paged: input_ids shape "
              "must be [1, %d], got ndim=%d shape=[%lld,%lld]\n",
              T, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    int expected_pool_layers = cfg.num_layers;
    if ((int)g_dense_k_pools.size() != expected_pool_layers ||
        (int)g_dense_v_pools.size() != expected_pool_layers ||
        (int)g_dense_k_scales.size() != expected_pool_layers ||
        (int)g_dense_v_scales.size() != expected_pool_layers ||
        (int)g_dense_paged_linear_caches.size() != expected_pool_layers * 2) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_forward_batched_verify_paged: pool/linear "
              "cache layer-count mismatch (cfg.num_layers=%d k_pools=%zu "
              "v_pools=%zu k_scales=%zu v_scales=%zu linear=%zu)\n",
              expected_pool_layers, g_dense_k_pools.size(),
              g_dense_v_pools.size(), g_dense_k_scales.size(),
              g_dense_v_scales.size(), g_dense_paged_linear_caches.size());
      fflush(stderr);
      return;
    }

    auto flat_ids = reshape(input_ids, {-1});
    auto emb_flat = take(embedding_weight, flat_ids, 0);
    auto h_3d = reshape(emb_flat, {1, T, cfg.hidden_size});

    std::vector<array> inputs;
    inputs.reserve(6 + cfg.num_layers * 4);
    inputs.push_back(std::move(h_3d));
    inputs.push_back(offset_arr);
    inputs.push_back(block_table);
    inputs.push_back(slot_mapping);
    inputs.push_back(seq_lens);
    inputs.push_back(cu_seqlens_q);
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      if (is_linear) {
        inputs.push_back(g_dense_paged_linear_caches[i * 2]);
        inputs.push_back(g_dense_paged_linear_caches[i * 2 + 1]);
        inputs.push_back(zeros({}, mlx::core::bfloat16));
        inputs.push_back(zeros({}, mlx::core::bfloat16));
      } else {
        inputs.push_back(g_dense_k_pools[i]);
        inputs.push_back(g_dense_v_pools[i]);
        inputs.push_back(g_dense_k_scales[i]);
        inputs.push_back(g_dense_v_scales[i]);
      }
    }

    bool with_tape = g_tape_recording_armed;
    auto& fn = get_or_compile_verify_paged(with_tape);
    auto outputs = fn(inputs);

    array* logits_alloc  = new array(outputs[0]);
    array* hiddens_alloc = nullptr;
    array* argmax_alloc  = nullptr;
    try {
      hiddens_alloc = new array(outputs[1]);
      argmax_alloc = new array(outputs[2]);
    } catch (...) {
      delete logits_alloc;
      delete hiddens_alloc;
      throw;
    }
    *out_logits  = reinterpret_cast<mlx_array*>(logits_alloc);
    *out_hiddens = reinterpret_cast<mlx_array*>(hiddens_alloc);
    *out_argmax  = reinterpret_cast<mlx_array*>(argmax_alloc);

    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      if (is_linear) {
        g_dense_paged_linear_caches[i * 2]     = outputs[3 + i * 2];
        g_dense_paged_linear_caches[i * 2 + 1] = outputs[3 + i * 2 + 1];
      } else {
        g_dense_k_pools[i] = outputs[3 + i * 2];
        g_dense_v_pools[i] = outputs[3 + i * 2 + 1];
      }
    }

    if (with_tape) {
      int extra_base = 3 + cfg.num_layers * 2;
      for (int i = 0; i < cfg.num_layers; i++) {
        bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
        if (!is_linear) continue;
        int base = extra_base + i * 4;
        g_gdn_tape_acc[i]     = outputs[base + 0];
        g_gdn_k_tape_acc[i]   = outputs[base + 1];
        g_gdn_g_tape_acc[i]   = outputs[base + 2];
        g_gdn_qkv_tape_acc[i] = outputs[base + 3];
      }
    }
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_forward_batched_verify_paged: EXIT OK "
              "T=%d with_tape=%d\n",
              T, with_tape ? 1 : 0);
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_forward_batched_verify_paged: %s\n",
            e.what());
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_argmax) *out_argmax = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_forward_batched_verify_paged\n");
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_argmax) *out_argmax = nullptr;
  }
}

// -----------------------------------------------------------------------------
// Eagerly compile the batched verify graph for ALL depths in {1..5} for both
// `WithTape=false` and `WithTape=true` variants.
//
// Runs ONE dummy verify forward per (depth, with_tape) pair to force
// `mlx::core::eval` of the compiled-graph outputs. MLX's internal compile cache
// keys on input shape — one (depth, with_tape) trace covers every subsequent
// verify at that same shape. Cost: ~10 dummy graph evals at model load; saves
// first-token latency on the first verify cycle of each prompt (otherwise the
// first cycle pays the trace+compile cost on its critical path).
//
// State preservation:
//   - Snapshots `g_compiled_caches[]`, `g_offset_int`, `g_compiled_offset`,
//     `g_tape_recording_armed`, and the four `g_gdn_*_tape_acc[]`
//     accumulators before any dummy call.
//   - Restores them AFTER the prewarm even if a dummy raises. This keeps
//     the main path's KV / offset state unchanged so the very next real
//     decode step sees identical inputs.
//
// Preconditions:
//   - `g_compile_inited` is true (main path was set up via
//     `mlx_qwen35_compiled_init_from_prefill`).
//   - Embedding weight (`"embedding"` or `"lm_head"`) is registered.
//
// Failure handling: any exception is logged to stderr and swallowed —
// prewarm is a best-effort optimization, not a correctness gate. On
// failure the verify graph falls back to its prior lazy-at-first-use
// path.
// -----------------------------------------------------------------------------
void mlx_qwen35_prewarm_verify_compiled() {
  if (!g_compile_inited) {
    return;
  }
  const auto& cfg = g_compile_config;
  if (g_compiled_caches.empty()) {
    return;
  }

  // Dummy embedding table for graph tracing. `mlx_qwen35_forward_batched_verify`
  // does `take(embedding_weight, flat_ids, 0)` then `reshape(.., {1, T,
  // hidden})`; the resulting `h_3d` — NOT `embedding_weight` itself — is
  // the compiled graph's input, so the trace is keyed only on
  // `[1, T, hidden]`, never on the embedding's shape. The prewarm feeds
  // all-zero `dummy_ids`, so a single dense bf16 row suffices: `take` of
  // index 0 yields `[T, hidden]` regardless of vocabulary size.
  //
  // We deliberately do NOT fetch the real `embedding.weight` from `g_weights`:
  // on quantized-embedding checkpoints that entry is a PACKED
  // `[vocab, hidden * bits / 32]` table, and `take` + `reshape` to dense
  // `[1, T, hidden]` would fail. A dense dummy is correct for every checkpoint
  // — the real verify path passes its own dense embedding via the FFI pointer
  // argument.
  array embedding_weight = zeros({1, cfg.hidden_size}, mlx::core::bfloat16);

  // Snapshot mutable state so post-prewarm the main path looks untouched.
  std::vector<array> saved_caches = g_compiled_caches;  // refcount bump per entry
  int saved_offset_int = g_offset_int;
  std::optional<array> saved_offset = g_compiled_offset;
  bool saved_tape_armed = g_tape_recording_armed;
  auto saved_tape_acc = g_gdn_tape_acc;
  auto saved_k_tape_acc = g_gdn_k_tape_acc;
  auto saved_g_tape_acc = g_gdn_g_tape_acc;
  auto saved_qkv_tape_acc = g_gdn_qkv_tape_acc;

  auto run_one = [&](int depth, bool with_tape) {
    int T = depth + 1;
    // Reset the main-path snapshot back to its starting state before each
    // dummy so all 10 prewarm calls use the SAME cache shape / offset.
    g_compiled_caches = saved_caches;
    g_offset_int = saved_offset_int;
    g_compiled_offset = saved_offset;
    if (with_tape) {
      // Arm tape recording so the FFI dispatches to `compiled_verify_batched_tape()`.
      g_gdn_tape_acc.assign(cfg.num_layers, std::nullopt);
      g_gdn_k_tape_acc.assign(cfg.num_layers, std::nullopt);
      g_gdn_g_tape_acc.assign(cfg.num_layers, std::nullopt);
      g_gdn_qkv_tape_acc.assign(cfg.num_layers, std::nullopt);
      g_tape_recording_armed = true;
    } else {
      g_tape_recording_armed = false;
    }

    array dummy_ids = zeros({1, T}, mlx::core::int32);
    array emb_copy = embedding_weight;
    mlx_array* logits_ptr = nullptr;
    mlx_array* hidden_ptr = nullptr;
    mlx_array* argmax_ptr = nullptr;
    mlx_qwen35_forward_batched_verify(
        reinterpret_cast<mlx_array*>(&dummy_ids),
        reinterpret_cast<mlx_array*>(&emb_copy),
        depth,
        &logits_ptr,
        &hidden_ptr,
        &argmax_ptr);
    if (!logits_ptr || !hidden_ptr || !argmax_ptr) {
      if (logits_ptr) delete reinterpret_cast<array*>(logits_ptr);
      if (hidden_ptr) delete reinterpret_cast<array*>(hidden_ptr);
      if (argmax_ptr) delete reinterpret_cast<array*>(argmax_ptr);
      throw std::runtime_error(
          "mlx_qwen35_prewarm_verify_compiled: batched verify returned null");
    }
    // Force trace+compile by evaluating the outputs. The compile cache
    // only populates after the first eval — building the lazy graph
    // alone is not enough.
    array logits = *reinterpret_cast<array*>(logits_ptr);
    delete reinterpret_cast<array*>(logits_ptr);
    array hiddens = *reinterpret_cast<array*>(hidden_ptr);
    delete reinterpret_cast<array*>(hidden_ptr);
    array argmax_ids = *reinterpret_cast<array*>(argmax_ptr);
    delete reinterpret_cast<array*>(argmax_ptr);
    std::vector<array> to_eval = {logits, hiddens, argmax_ids};
    if (with_tape) {
      for (auto& slot : g_gdn_tape_acc) if (slot) to_eval.push_back(*slot);
      for (auto& slot : g_gdn_k_tape_acc) if (slot) to_eval.push_back(*slot);
      for (auto& slot : g_gdn_g_tape_acc) if (slot) to_eval.push_back(*slot);
      for (auto& slot : g_gdn_qkv_tape_acc) if (slot) to_eval.push_back(*slot);
    }
    mlx::core::eval(std::move(to_eval));
  };

  try {
    // Prewarm strategy: trace ONLY the bucket the first verify cycle will hit.
    // With 6 buckets + legacy slot × 5 depths × 2 tape variants = 70 possible
    // traces, eagerly prewarming all of them at model load would cost ~75s
    // (~1s per dense forward at depth 5). Lazy-tracing other buckets shifts
    // that cost to first-use of each (bucket, depth, tape) combo — typically
    // <0.5s per shape because the underlying weight load + kernel dispatch is
    // already warm by the time the larger buckets are hit.
    //
    // The "first verify cycle bucket" is the one that fits
    // `g_offset_int + max_depth + 1` — i.e. the post-prefill offset plus the
    // maximum verify window. Calling `mlx_qwen35_forward_batched_verify` with
    // `g_offset_int = saved_offset_int` (real prefill offset) routes through
    // `bucket_index_for(...)` exactly like a real cycle would, so the trace MLX
    // caches IS the trace the first real cycle hits. Subsequent prompts at
    // different offsets that don't fit this bucket re-trace on first use
    // (one-time cost amortised over the turn).
    for (int d = 1; d <= 5; d++) run_one(d, /*with_tape=*/false);
    for (int d = 1; d <= 5; d++) run_one(d, /*with_tape=*/true);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_prewarm_verify_compiled: %s\n",
            e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_prewarm_verify_compiled\n");
    fflush(stderr);
  }

  // Restore the snapshot — even if a prewarm call mutated g_compiled_caches
  // or g_offset_int via the FFI, this puts the main path back exactly where
  // it was at function entry.
  g_compiled_caches = std::move(saved_caches);
  g_offset_int = saved_offset_int;
  g_compiled_offset = std::move(saved_offset);
  g_tape_recording_armed = saved_tape_armed;
  g_gdn_tape_acc = std::move(saved_tape_acc);
  g_gdn_k_tape_acc = std::move(saved_k_tape_acc);
  g_gdn_g_tape_acc = std::move(saved_g_tape_acc);
  g_gdn_qkv_tape_acc = std::move(saved_qkv_tape_acc);
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

// Same async_eval batch as `mlx_qwen35_eval_token_and_compiled_caches` but also
// folds an arbitrary `extra` array into the dispatch. Used by the
// chained-cycles MTP path (`MLX_MTP_CHAINED_CYCLES=1`) at end-of-iteration to
// fuse the `verify_hidden[K]` slice eval with the next cycle's first draft
// inputs. Without this, the slice stays lazy until the next-cycle draft graph
// is built — at which point materializing it forces a mid-cycle Metal
// command-buffer roundtrip on the chained path that the Step-A bypass doesn't
// pay (Step A produces `hidden` and `token` as siblings of a single fused eval).
//
// `extra_ptr` MAY be null, in which case behaviour is identical to
// `mlx_qwen35_eval_token_and_compiled_caches`.
void mlx_qwen35_eval_token_caches_and_extra(
    mlx_array* next_token_ptr, mlx_array* extra_ptr) {
  try {
    std::vector<array> to_eval;
    to_eval.reserve(2 + g_compiled_caches.size());
    to_eval.push_back(*reinterpret_cast<array*>(next_token_ptr));
    if (extra_ptr) {
      to_eval.push_back(*reinterpret_cast<array*>(extra_ptr));
    }
    for (const auto& c : g_compiled_caches) {
      to_eval.push_back(c);
    }
    mlx::core::async_eval(std::move(to_eval));
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in eval_token_caches_and_extra: %s\n", e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in eval_token_caches_and_extra\n");
    fflush(stderr);
  }
}

void mlx_qwen35_compiled_adjust_offset(int delta) {
  g_offset_int += delta;
  g_compiled_offset = array(g_offset_int, mlx::core::int32);
}

// MTP — snapshot the GDN linear-attention caches plus the decode offset.
// Called by the MTP cycle macro AFTER Step A and BEFORE verify. The snapshot
// keeps the pre-verify recurrent / conv state alive (MLX arrays are
// ref-counted; copy construction just bumps the refcount) while the global
// continues to mutate inside the verify loop. Restore via
// `mlx_qwen35_compiled_restore_linear_caches`.
//
// Only linear-attention layer slots are populated; full-attention slots are
// stored as bf16 zero placeholders so per-layer indexing stays uniform.
// Idempotent — calling twice overwrites the previous snapshot.
void mlx_qwen35_compiled_snapshot_linear_caches() {
  if (!g_compile_inited) {
    g_linear_snapshot_taken = false;
    return;
  }
  const auto& cfg = g_compile_config;
  try {
    g_compiled_linear_snapshot.clear();
    g_compiled_linear_snapshot.reserve(cfg.num_layers * 2);
    auto placeholder = []() { return zeros({}, mlx::core::bfloat16); };
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      if (is_linear) {
        // `array(other)` = refcount bump on the MLX array; the
        // underlying recurrent/conv buffer stays alive while the
        // global slot mutates inside the verify loop.
        g_compiled_linear_snapshot.push_back(array(g_compiled_caches[i * 2]));
        g_compiled_linear_snapshot.push_back(array(g_compiled_caches[i * 2 + 1]));
      } else {
        g_compiled_linear_snapshot.push_back(placeholder());
        g_compiled_linear_snapshot.push_back(placeholder());
      }
    }
    g_linear_snapshot_offset = g_offset_int;
    g_linear_snapshot_taken = true;
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_compiled_snapshot_linear_caches: %s\n", e.what());
    fflush(stderr);
    g_compiled_linear_snapshot.clear();
    g_linear_snapshot_taken = false;
  }
}

// MTP — restore the GDN linear caches AND the decode offset from the most
// recent snapshot. Called on verify rejection (any `accepted_drafts < depth`)
// BEFORE replaying the K accepted drafts via `mlx_qwen35_forward_compiled`.
// Full-attention K/V slots are intentionally left as-is — their offsets are
// rewound via the offset restore here, and later writes will overwrite the
// stale post-verify slots.
//
// No-op if no snapshot has been taken since the last reset.
void mlx_qwen35_compiled_restore_linear_caches() {
  if (!g_compile_inited || !g_linear_snapshot_taken) return;
  const auto& cfg = g_compile_config;
  try {
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      if (!is_linear) continue;
      // Restore via copy constructor — keeps the snapshot vector's
      // entries alive for potential repeated restores within the same
      // cycle (defensive; current callers restore at most once).
      g_compiled_caches[i * 2]     = array(g_compiled_linear_snapshot[i * 2]);
      g_compiled_caches[i * 2 + 1] = array(g_compiled_linear_snapshot[i * 2 + 1]);
    }
    g_offset_int = g_linear_snapshot_offset;
    g_compiled_offset = array(g_offset_int, mlx::core::int32);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_compiled_restore_linear_caches: %s\n", e.what());
    fflush(stderr);
  }
}

// Paged-pool sibling of `mlx_qwen35_compiled_snapshot_linear_caches`. Captures
// the pre-verify GDN linear-attention state out of
// `g_dense_paged_linear_caches` so a partial-accept rollback can restore it.
// Full-attention slots store bf16 zero placeholders so the per-layer indexing
// in the snapshot vector matches the cache vector. Idempotent — calling twice
// overwrites the previous snapshot.
void mlx_qwen35_compiled_snapshot_paged_linear_caches() {
  if (!g_dense_paged_inited) {
    g_dense_paged_linear_snapshot_taken = false;
    return;
  }
  const auto& cfg = g_dense_paged_config;
  if ((int)g_dense_paged_linear_caches.size() != cfg.num_layers * 2) {
    g_dense_paged_linear_snapshot_taken = false;
    return;
  }
  try {
    g_dense_paged_linear_snapshot.clear();
    g_dense_paged_linear_snapshot.reserve(cfg.num_layers * 2);
    auto placeholder = []() { return zeros({}, mlx::core::bfloat16); };
    for (int i = 0; i < cfg.num_layers; i++) {
      if (dense_paged_is_linear_layer(i)) {
        g_dense_paged_linear_snapshot.push_back(array(g_dense_paged_linear_caches[i * 2]));
        g_dense_paged_linear_snapshot.push_back(array(g_dense_paged_linear_caches[i * 2 + 1]));
      } else {
        g_dense_paged_linear_snapshot.push_back(placeholder());
        g_dense_paged_linear_snapshot.push_back(placeholder());
      }
    }
    g_dense_paged_linear_snapshot_taken = true;
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_compiled_snapshot_paged_linear_caches: %s\n",
            e.what());
    fflush(stderr);
    g_dense_paged_linear_snapshot.clear();
    g_dense_paged_linear_snapshot_taken = false;
  }
}

// Restore the paged GDN linear caches from the snapshot. Called on FULL
// rollback (no draft accepted) BEFORE the next forward. No-op when no snapshot
// is recorded. The paged-pool full-attention K/V slots are intentionally left
// as-is — the Rust adapter's block-table cursor rewinds via
// `rollback_last_tokens`, and the kernel's per-query effective-context derives
// only from the post-rollback `seq_lens` so stale slots are not read.
void mlx_qwen35_compiled_restore_paged_linear_caches() {
  if (!g_dense_paged_inited || !g_dense_paged_linear_snapshot_taken) return;
  const auto& cfg = g_dense_paged_config;
  if ((int)g_dense_paged_linear_caches.size() != cfg.num_layers * 2 ||
      (int)g_dense_paged_linear_snapshot.size() != cfg.num_layers * 2) {
    return;
  }
  try {
    for (int i = 0; i < cfg.num_layers; i++) {
      if (!dense_paged_is_linear_layer(i)) continue;
      g_dense_paged_linear_caches[i * 2] =
          array(g_dense_paged_linear_snapshot[i * 2]);
      g_dense_paged_linear_caches[i * 2 + 1] =
          array(g_dense_paged_linear_snapshot[i * 2 + 1]);
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_compiled_restore_paged_linear_caches: %s\n",
            e.what());
    fflush(stderr);
  }
}

// Partial-accept rollback for paged GDN linear caches.
//
// Restores the pre-verify snapshot, then re-runs the first `accepted_steps`
// (= K + 1) verify tokens through the GDN kernel so the linear state matches
// the accepted prefix. Paged sibling of `mlx_qwen35_compiled_tape_replay`.
// Falls back to a snapshot-only restore when called with `accepted_steps <= 0`
// (full reject) and logs to stderr otherwise.
//
// Preconditions:
//   - `mlx_qwen35_compiled_snapshot_paged_linear_caches` has been called
//     for this cycle.
//   - `0 <= accepted_steps <= depth + 1` (caller bounds-checks).
//
// On any precondition violation, logs to stderr and falls back to a
// pure snapshot-restore so the next forward starts from the pre-verify
// state — never a stale post-verify state.
void mlx_qwen35_compiled_replay_paged_linear_caches_for_accept(
    int accepted_steps, int depth) {
  if (!g_dense_paged_inited) return;
  if (!g_dense_paged_linear_snapshot_taken) {
    fprintf(stderr,
            "[MLX] replay_paged_linear_caches_for_accept: snapshot not taken — "
            "skipping\n");
    fflush(stderr);
    return;
  }
  if (depth < 1 || depth > 5) {
    fprintf(stderr,
            "[MLX] replay_paged_linear_caches_for_accept: depth %d outside "
            "[1, 5] — falling back to snapshot restore\n",
            depth);
    fflush(stderr);
    mlx_qwen35_compiled_restore_paged_linear_caches();
    return;
  }
  if (accepted_steps < 0 || accepted_steps > depth + 1) {
    fprintf(stderr,
            "[MLX] replay_paged_linear_caches_for_accept: accepted_steps %d "
            "outside [0, depth+1=%d] — falling back to snapshot restore\n",
            accepted_steps, depth + 1);
    fflush(stderr);
    mlx_qwen35_compiled_restore_paged_linear_caches();
    return;
  }
  if (accepted_steps == depth + 1) {
    return;
  }
  mlx_qwen35_compiled_restore_paged_linear_caches();
}

// Arm tape recording. After this call, `qwen35_decode_fn` routes
// linear-attention layers through `gdn_pure_fn_with_tape` and appends the
// per-step `(tape, k, g, qkv)` tensors into the per-layer accumulator vectors.
// Must be called BEFORE the MTP verify FFI's D+1 sequential forwards.
// Idempotent. No-op if `g_compile_inited` is false.
//
// The companion `_snapshot_linear_caches` is still required (the rollback path
// uses the snapshot as the starting state for the replay kernel). This call
// only turns on tape recording; the snapshot lives in
// `g_compiled_linear_snapshot`.
void mlx_qwen35_compiled_tape_arm() {
  if (!g_compile_inited) {
    g_tape_recording_armed = false;
    return;
  }
  const auto& cfg = g_compile_config;
  g_gdn_tape_acc.assign(cfg.num_layers, std::nullopt);
  g_gdn_k_tape_acc.assign(cfg.num_layers, std::nullopt);
  g_gdn_g_tape_acc.assign(cfg.num_layers, std::nullopt);
  g_gdn_qkv_tape_acc.assign(cfg.num_layers, std::nullopt);
  g_tape_recording_armed = true;
}

// Disarm tape recording and drop accumulators. Normally reached through
// `_tape_replay`; explicit calls are cleanup-only. Idempotent.
void mlx_qwen35_compiled_tape_disarm() {
  g_tape_recording_armed = false;
  // Drop refcounts so the lazy MLX graphs can be released. The vectors
  // stay sized so a subsequent arm doesn't have to re-allocate.
  for (auto& slot : g_gdn_tape_acc)     slot.reset();
  for (auto& slot : g_gdn_k_tape_acc)   slot.reset();
  for (auto& slot : g_gdn_g_tape_acc)   slot.reset();
  for (auto& slot : g_gdn_qkv_tape_acc) slot.reset();
}

// Restore the GDN linear-attention recurrent + conv states by applying the
// first `accepted_steps` recorded innovations to the pre-verify snapshot, then
// advance the decode offset to `snapshot_offset + accepted_steps`. After this
// call:
//   - g_compiled_caches[i*2]   == conv_state at "snapshot + accepted_steps innovations"
//   - g_compiled_caches[i*2+1] == recurrent_state at "snapshot + accepted_steps innovations"
//   - g_offset_int             == g_linear_snapshot_offset + accepted_steps
//
// `accepted_steps` here is `K + 1` in the MTP cycle terminology (the verify
// processes `last_committed_id + K accepted drafts` = `K + 1` tokens that
// survive). The caller computes this from `accepted_drafts + 1` and passes it
// as `accepted_steps`.
//
// Preconditions:
//   - `_snapshot_linear_caches` has been called for this cycle.
//   - `_tape_arm` has been called and the D+1 verify forwards have all
//     recorded into the accumulators.
//   - `1 <= accepted_steps <= recorded_steps`.
//
// On any precondition violation logs to stderr and leaves state
// untouched (the caller can fall back to the K+1 path on the next
// rollback). Always disarms recording afterward (success or failure)
// so a stale armed flag can't leak across cycles.
void mlx_qwen35_compiled_tape_replay(int accepted_steps) {
  if (!g_compile_inited) {
    g_tape_recording_armed = false;
    return;
  }
  if (!g_linear_snapshot_taken) {
    fprintf(stderr,
            "[MLX] tape_replay: snapshot not taken — falling back\n");
    fflush(stderr);
    mlx_qwen35_compiled_tape_disarm();
    return;
  }
  if (accepted_steps <= 0) {
    fprintf(stderr,
            "[MLX] tape_replay: accepted_steps=%d must be > 0\n",
            accepted_steps);
    fflush(stderr);
    mlx_qwen35_compiled_tape_disarm();
    return;
  }
  const auto& cfg = g_compile_config;
  try {
    int keep = cfg.linear_conv_kernel_dim - 1;
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      if (!is_linear) continue;
      if (!g_gdn_tape_acc[i].has_value() || !g_gdn_k_tape_acc[i].has_value() ||
          !g_gdn_g_tape_acc[i].has_value() || !g_gdn_qkv_tape_acc[i].has_value()) {
        fprintf(stderr,
                "[MLX] tape_replay: layer %d has empty accumulator — "
                "did the verify forwards record? Falling back.\n", i);
        fflush(stderr);
        mlx_qwen35_compiled_tape_disarm();
        return;
      }
      auto& tape_full = *g_gdn_tape_acc[i];
      auto& k_full    = *g_gdn_k_tape_acc[i];
      auto& g_full    = *g_gdn_g_tape_acc[i];
      auto& qkv_full  = *g_gdn_qkv_tape_acc[i];

      int recorded = tape_full.shape(1);
      if (accepted_steps > recorded) {
        fprintf(stderr,
                "[MLX] tape_replay: accepted_steps=%d > recorded=%d "
                "(layer %d). Falling back.\n", accepted_steps, recorded, i);
        fflush(stderr);
        mlx_qwen35_compiled_tape_disarm();
        return;
      }

      // Slice the first `accepted_steps` per-step entries.
      auto tape_pre = slice(tape_full, {0, 0, 0, 0},
                            {tape_full.shape(0), accepted_steps,
                             tape_full.shape(2), tape_full.shape(3)});
      auto k_pre    = slice(k_full,    {0, 0, 0, 0},
                            {k_full.shape(0), accepted_steps,
                             k_full.shape(2), k_full.shape(3)});
      auto g_pre    = slice(g_full,    {0, 0, 0},
                            {g_full.shape(0), accepted_steps, g_full.shape(2)});
      auto qkv_pre  = slice(qkv_full,  {0, 0, 0},
                            {qkv_full.shape(0), accepted_steps, qkv_full.shape(2)});

      // Recurrent state: replay innovations from the snapshot state.
      const array& snapshot_rs = g_compiled_linear_snapshot[i * 2 + 1];
      auto new_rs = tape_replay_kernel_call(tape_pre, k_pre, g_pre, snapshot_rs);

      // Conv state: window of (kernel_dim - 1) entries starting at
      // position `accepted_steps` in the augmented stream
      // `concat(snapshot_conv_state, qkv_recorded)`. Mirrors DFlash's
      // `_rebuild_conv_state` (recurrent_rollback_cache.py:123-140).
      const array& snapshot_cs = g_compiled_linear_snapshot[i * 2];
      auto new_cs = [&]() -> array {
        if (keep > 0) {
          auto conv_input = concatenate({snapshot_cs, qkv_pre}, 1);
          int total = conv_input.shape(1);
          int start = accepted_steps;
          int end   = std::min(start + keep, total);
          // `start + keep == accepted_steps + (kernel_dim - 1)` and
          // `total == keep + accepted_steps`, so `end == total` exactly.
          return slice(conv_input,
                       {0, start, 0},
                       {conv_input.shape(0), end, conv_input.shape(2)});
        }
        return snapshot_cs;
      }();

      g_compiled_caches[i * 2]     = std::move(new_cs);
      g_compiled_caches[i * 2 + 1] = std::move(new_rs);
    }

    // Advance the offset to "snapshot + accepted_steps". Mirrors what
    // the K+1 path produced via `restore_linear_caches` (which set
    // offset = snapshot_offset) + K `forward_compiled` calls (each
    // advancing offset by 1).
    g_offset_int = g_linear_snapshot_offset + accepted_steps;
    g_compiled_offset = array(g_offset_int, mlx::core::int32);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_compiled_tape_replay: %s\n",
            e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_compiled_tape_replay\n");
    fflush(stderr);
  }
  mlx_qwen35_compiled_tape_disarm();
}

// Paged variant of `mlx_qwen35_compiled_tape_arm`.
//
// The dense BHTD arm function gates on `g_compile_inited`, which is false on
// pure-paged turns. The paged verify graph
// (`qwen35_verify_batched_decode_fn_paged<true>`) already writes per-layer tape
// into the SHARED `g_gdn_*_tape_acc[]` accumulators when
// `g_tape_recording_armed == true`, so this just flips the flag based on
// `g_dense_paged_inited` instead.
void mlx_qwen35_compiled_tape_arm_paged() {
  if (!g_dense_paged_inited) {
    g_tape_recording_armed = false;
    return;
  }
  const auto& cfg = g_dense_paged_config;
  g_gdn_tape_acc.assign(cfg.num_layers, std::nullopt);
  g_gdn_k_tape_acc.assign(cfg.num_layers, std::nullopt);
  g_gdn_g_tape_acc.assign(cfg.num_layers, std::nullopt);
  g_gdn_qkv_tape_acc.assign(cfg.num_layers, std::nullopt);
  g_tape_recording_armed = true;
}

// Paged variant of `mlx_qwen35_compiled_tape_replay`.
//
// Reads the tape from the same shared `g_gdn_*_tape_acc[]` accumulators the
// BHTD replay reads, but applies the first `accepted_steps` innovations to
// `g_dense_paged_linear_snapshot[]` and writes the result back into
// `g_dense_paged_linear_caches[]`. Mirrors `mlx_qwen35_compiled_tape_replay`
// line-for-line on the GDN side; only the snapshot and target arrays differ.
void mlx_qwen35_compiled_tape_replay_paged(int accepted_steps) {
  if (!g_dense_paged_inited) {
    g_tape_recording_armed = false;
    return;
  }
  if (!g_dense_paged_linear_snapshot_taken) {
    fprintf(stderr,
            "[MLX] tape_replay_paged: snapshot not taken — falling back\n");
    fflush(stderr);
    mlx_qwen35_compiled_tape_disarm();
    return;
  }
  if (accepted_steps <= 0) {
    fprintf(stderr,
            "[MLX] tape_replay_paged: accepted_steps=%d must be > 0\n",
            accepted_steps);
    fflush(stderr);
    mlx_qwen35_compiled_tape_disarm();
    return;
  }
  const auto& cfg = g_dense_paged_config;
  try {
    int keep = cfg.linear_conv_kernel_dim - 1;
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
      if (!is_linear) continue;
      if (!g_gdn_tape_acc[i].has_value() || !g_gdn_k_tape_acc[i].has_value() ||
          !g_gdn_g_tape_acc[i].has_value() || !g_gdn_qkv_tape_acc[i].has_value()) {
        fprintf(stderr,
                "[MLX] tape_replay_paged: layer %d has empty accumulator — "
                "did the verify forwards record? Falling back.\n", i);
        fflush(stderr);
        mlx_qwen35_compiled_restore_paged_linear_caches();
        mlx_qwen35_compiled_tape_disarm();
        return;
      }
      auto& tape_full = *g_gdn_tape_acc[i];
      auto& k_full    = *g_gdn_k_tape_acc[i];
      auto& g_full    = *g_gdn_g_tape_acc[i];
      auto& qkv_full  = *g_gdn_qkv_tape_acc[i];

      int recorded = tape_full.shape(1);
      if (accepted_steps > recorded) {
        fprintf(stderr,
                "[MLX] tape_replay_paged: accepted_steps=%d > recorded=%d "
                "(layer %d). Falling back.\n", accepted_steps, recorded, i);
        fflush(stderr);
        mlx_qwen35_compiled_restore_paged_linear_caches();
        mlx_qwen35_compiled_tape_disarm();
        return;
      }

      auto tape_pre = slice(tape_full, {0, 0, 0, 0},
                            {tape_full.shape(0), accepted_steps,
                             tape_full.shape(2), tape_full.shape(3)});
      auto k_pre    = slice(k_full,    {0, 0, 0, 0},
                            {k_full.shape(0), accepted_steps,
                             k_full.shape(2), k_full.shape(3)});
      auto g_pre    = slice(g_full,    {0, 0, 0},
                            {g_full.shape(0), accepted_steps, g_full.shape(2)});
      auto qkv_pre  = slice(qkv_full,  {0, 0, 0},
                            {qkv_full.shape(0), accepted_steps, qkv_full.shape(2)});

      const array& snapshot_rs = g_dense_paged_linear_snapshot[i * 2 + 1];
      auto new_rs = tape_replay_kernel_call(tape_pre, k_pre, g_pre, snapshot_rs);

      const array& snapshot_cs = g_dense_paged_linear_snapshot[i * 2];
      auto new_cs = [&]() -> array {
        if (keep > 0) {
          auto conv_input = concatenate({snapshot_cs, qkv_pre}, 1);
          int total = conv_input.shape(1);
          int start = accepted_steps;
          int end   = std::min(start + keep, total);
          return slice(conv_input,
                       {0, start, 0},
                       {conv_input.shape(0), end, conv_input.shape(2)});
        }
        return snapshot_cs;
      }();

      g_dense_paged_linear_caches[i * 2]     = std::move(new_cs);
      g_dense_paged_linear_caches[i * 2 + 1] = std::move(new_rs);
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_compiled_tape_replay_paged: %s\n",
            e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_compiled_tape_replay_paged\n");
    fflush(stderr);
  }
  mlx_qwen35_compiled_tape_disarm();
}

// Invalidate ALL compiled MTP-verify dispatch tables so the next
// `get_or_compile_verify_*` re-traces against the CURRENT weight registry.
//
// Called by the Rust loaders (`register_weights_with_cpp` /
// `register_moe_weights_with_cpp`) on EVERY model reload, immediately after
// `mlx_clear_weights()` and INSIDE the `COMPILED_WEIGHTS_RWLOCK` write critical
// section, so it is serialized against in-flight compiled reads. Unlike
// `mlx_qwen35_compiled_reset()` (a PER-TURN reset that deliberately keeps the
// verify tables for cross-turn reuse), this is a PER-RELOAD invalidation:
// without it a second same-shape Qwen3.5/3.6 model loaded in the same process
// would verify speculative tokens with the FIRST model's baked weights (silent
// corruption).
//
// See `invalidate_verify_compiled_tables()` for the slot-nulling →
// lazy-re-trace mechanism and its thread-safety rationale. That helper also
// nulls the dense paged AR-decode graph. We additionally invalidate the dense
// MTP draft/commit graphs (defined in `mlx_qwen35_mtp_compiled.cpp`) so EVERY
// weight-baking graph re-traces against the reloaded registry.
void mlx_qwen35_invalidate_compiled_graphs() {
  invalidate_verify_compiled_tables();
  mlx_qwen35_mtp_invalidate_compiled_graphs();
}

// Reset BOTH the flat compiled state AND the paged-path globals. Keeping these
// symmetric is required because `mlx_qwen35_init_paged` flips
// `g_dense_paged_inited` to true independently of `g_compile_inited`; without
// clearing the paged side here, a later `mlx_qwen35_forward_paged()` would pass
// the init guard and reuse stale KV pools / scales / linear caches / offset
// from a previous request or model. Mirrors `mlx_qwen35_moe_reset`.
void mlx_qwen35_compiled_reset() {
  // Flat-path compiled globals.
  g_compiled_caches.clear();
  g_compiled_offset = std::nullopt;
  g_offset_int = 0;
  g_compile_inited = false;

  // Clear the stashed last-hidden so a subsequent reset → re-init turn doesn't
  // see a stale handle whose underlying buffer is no longer valid.
  g_last_hidden = std::nullopt;

  // Drop any pending linear-cache snapshot so it can't leak across turns /
  // model loads.
  g_compiled_linear_snapshot.clear();
  g_linear_snapshot_offset = 0;
  g_linear_snapshot_taken = false;

  // Drop any pending tape recording so a stale armed flag / half-recorded
  // accumulator can't leak across model reloads.
  g_tape_recording_armed = false;
  g_gdn_tape_acc.clear();
  g_gdn_k_tape_acc.clear();
  g_gdn_g_tape_acc.clear();
  g_gdn_qkv_tape_acc.clear();

  // Paged-path globals.
  g_dense_paged_config = CompileConfig{};
  g_dense_k_pools.clear();
  g_dense_v_pools.clear();
  g_dense_k_scales.clear();
  g_dense_v_scales.clear();
  g_dense_paged_linear_caches.clear();
  g_dense_paged_offset_int = 0;
  g_dense_paged_inited = false;

  g_dense_paged_linear_snapshot.clear();
  g_dense_paged_linear_snapshot_taken = false;

  // Drop the paged last-hidden stash so cross-turn handles can't outlive the
  // underlying compile cache.
  g_last_hidden_paged = std::nullopt;
}

// Export a heap-allocated deep copy of the post-final-norm hidden state of the
// last decoded token, captured by the most recent `mlx_qwen35_forward_compiled`
// invocation. Returns nullptr if no forward has run since the last reset / the
// stash is unpopulated. Caller owns the returned `mlx_array*` (use
// `mlx_array_delete`).
//
// The returned handle is a lazy MLX array whose graph references the
// final_norm output; the caller MUST `eval()` it before reading any
// element, and MUST NOT call `mlx_qwen35_compiled_reset()` between
// export and eval (the reset would clear `g_compiled_caches` whose
// inputs the hidden may still depend on via the cached graph).
void mlx_qwen35_export_last_hidden(mlx_array** out) {
  if (!out) return;
  *out = nullptr;
  if (!g_compile_inited || !g_last_hidden.has_value()) {
    return;
  }
  try {
    *out = reinterpret_cast<mlx_array*>(new array(*g_last_hidden));
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_export_last_hidden: %s\n", e.what());
    fflush(stderr);
    *out = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_export_last_hidden\n");
    fflush(stderr);
    *out = nullptr;
  }
}

// Paged-path sibling of `mlx_qwen35_export_last_hidden`. Exports a
// heap-allocated deep copy of the post-final-norm hidden captured by the most
// recent `mlx_qwen35_forward_paged` invocation. Returns nullptr if no paged
// forward has run since the last reset.
//
// Lifetime contract: the returned handle is a lazy MLX array whose graph
// references the paged decode's final_norm output. The caller MUST eval
// it before reading scalars, and MUST NOT call
// `mlx_qwen35_compiled_reset` between export and eval (the reset clears
// `g_dense_paged_*` globals whose handles the hidden may depend on via
// the cached graph).
void mlx_qwen35_export_last_hidden_paged(mlx_array** out) {
  if (!out) return;
  *out = nullptr;
  if (!g_dense_paged_inited || !g_last_hidden_paged.has_value()) {
    return;
  }
  try {
    *out = reinterpret_cast<mlx_array*>(new array(*g_last_hidden_paged));
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_export_last_hidden_paged: %s\n", e.what());
    fflush(stderr);
    *out = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_export_last_hidden_paged\n");
    fflush(stderr);
    *out = nullptr;
  }
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

// Exposed for `mlx_qwen35_mtp_compiled_init_from_main` so the MTP init can fail
// loudly if the main path hasn't been initialised yet. Without this guard,
// `mlx_qwen35_get_cache_offset()` silently returns 0 from a fresh
// `g_offset_int`, and the MTP path would build attention masks against a
// phantom prefix.
//
// The paged dense path inits via `mlx_qwen35_init_paged` (which sets
// `g_dense_paged_inited`) instead of `mlx_qwen35_compiled_init_from_prefill`.
// Both flags signal "weights have been registered and the main forward is
// ready to run for this model"; either is sufficient for the MTP init
// precondition.
int mlx_qwen35_is_compile_inited() {
  return (g_compile_inited || g_dense_paged_inited) ? 1 : 0;
}

// Test-only helper: forcibly mark the main compiled path as initialised (or
// not) without going through the full `init_from_prefill` flow that requires
// real per-layer KV cache arrays. Used by MTP FFI smoke tests so they can
// satisfy the `is_compile_inited` precondition in
// `mlx_qwen35_mtp_compiled_init_from_main` without standing up a full dense
// decoder cache. Production code MUST NOT call this — use
// `mlx_qwen35_compiled_init_from_prefill` instead.
void mlx_qwen35_compiled_test_force_inited(int inited) {
  g_compile_inited = (inited != 0);
}

// Test-only helper. Stands up `g_dense_paged_linear_caches[]`
// (size = num_layers * 2) populated with bf16 scalars chosen so the
// snapshot/restore round-trip is observable (slot value = layer * 100 +
// position * 2 + slot_offset, packed into a scalar bf16). Sets
// `g_dense_paged_inited = true` and `g_dense_paged_config` to a minimal shape
// that satisfies `dense_paged_is_linear_layer`. PRODUCTION CODE MUST NOT CALL
// THIS — use `mlx_qwen35_init_paged`.
void mlx_qwen35_compiled_test_force_paged_linear_caches(
    int num_layers, int full_attention_interval) {
  g_dense_paged_config = CompileConfig{};
  g_dense_paged_config.num_layers = num_layers;
  g_dense_paged_config.full_attention_interval = full_attention_interval;
  g_dense_paged_linear_caches.clear();
  g_dense_paged_linear_caches.reserve(num_layers * 2);
  for (int layer = 0; layer < num_layers; layer++) {
    float base = 0.125f * static_cast<float>(layer + 1);
    g_dense_paged_linear_caches.push_back(
        array(base + 0.0625f, mlx::core::bfloat16));
    g_dense_paged_linear_caches.push_back(
        array(base + 0.03125f, mlx::core::bfloat16));
  }
  g_dense_paged_inited = true;
}

// Test-only inspector: read the scalar bf16 value at slot `slot_idx` of
// `g_dense_paged_linear_caches`. Returns NaN if the slot is out of range or the
// array shape isn't a scalar. Caller MUST have previously called
// `mlx_qwen35_compiled_test_force_paged_linear_caches` (or hold paged state
// from a real init).
float mlx_qwen35_compiled_test_read_paged_linear_slot(int slot_idx) {
  if (slot_idx < 0 || slot_idx >= (int)g_dense_paged_linear_caches.size()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  try {
    auto& a = g_dense_paged_linear_caches[slot_idx];
    if (a.ndim() != 0) return std::numeric_limits<float>::quiet_NaN();
    array a_f32 = mlx::core::astype(a, mlx::core::float32);
    mlx::core::eval(a_f32);
    return a_f32.item<float>();
  } catch (...) {
    return std::numeric_limits<float>::quiet_NaN();
  }
}

// Test-only mutator: replace slot `slot_idx` with a fresh scalar bf16 of
// `value`. Used by tests to simulate the paged-verify FFI's in-place
// linear-cache mutation without standing up a real verify forward.
void mlx_qwen35_compiled_test_write_paged_linear_slot(int slot_idx, float value) {
  if (slot_idx < 0 || slot_idx >= (int)g_dense_paged_linear_caches.size()) return;
  g_dense_paged_linear_caches[slot_idx] = array(value, mlx::core::bfloat16);
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
// Paged Dense forward FFI.
//
// Coexists alongside `mlx_qwen35_forward_compiled` /
// `_compiled_init_from_prefill` while the Rust dispatcher decides per-turn
// which graph to run. A single `mlx_qwen35_compiled_reset()` wipes BOTH graphs'
// state.
// =============================================================================

// Initialize the paged Dense forward graph from per-layer pool / scale
// handles AND per-layer linear-attention recurrent caches.
//
// Layout contract (mirrors `mlx_qwen35_moe_init_paged`):
//   - `k_pool_handles[i]`: pointer to a `[num_blocks, num_kv_heads,
//     head_size/x_pack=8, block_size=16, x_pack=8]` bf16 array view.
//     Hard-coded bf16 (`KvDtype::Bf16`) and `block_size = 16`.
//   - `v_pool_handles[i]`: pointer to a `[num_blocks, num_kv_heads,
//     head_size, block_size=16]` bf16 array view.
//   - `k_scale_handles[i]` / `v_scale_handles[i]`: pointer to `[1]` f32
//     scale placeholders (currently 1.0; FP8 calibration is future work).
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
// CONTRACT: This FFI is decode-only — `input_ids` MUST have exactly one element
// and `slot_mapping` MUST be `[1]`. Chunked prefill (multi-token) is not
// supported. The contract is enforced explicitly: violating it returns null
// logits without modifying global state, so the Rust caller can fall back
// cleanly.
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

    // Contract: single-token decode only.
    //
    // `attn_for_compile_paged` builds new_k / new_v with shape
    // `[1, num_kv_heads, head_size]` and feeds `slot_mapping` directly into
    // `paged_kv_write`, which requires `slot_mapping.shape(0) == new_k.shape(0)`.
    // Multi-token (B > 1) is not supported.
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

    // Extract: [logits, new_offset, hidden_for_export, new_caches...]
    *output_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    g_last_hidden_paged = outputs[2];
    g_dense_paged_offset_int++;
    // Stash post-step caches back into the per-layer slots.
    for (int i = 0; i < cfg.num_layers; i++) {
      bool is_linear = dense_paged_is_linear_layer(i);
      auto& a = outputs[3 + i * 2];
      auto& b = outputs[3 + i * 2 + 1];
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
