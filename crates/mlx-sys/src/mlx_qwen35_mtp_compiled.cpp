// =============================================================================
// Qwen3.5 Dense MTP (Multi-Token Prediction) compiled draft + verify graphs.
//
// Companion to the main `mlx_qwen35.cpp` compiled forward path. Provides
// three FFI entrypoints:
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
//     `depth > 5` is rejected.
//
// IMPORTANT: this file READS the main path's `g_compiled_caches` /
// `g_offset_int` via the `extern` declarations below. The verify graph
// MUST be called in the same mutex critical section as the main path
// (`DENSE_COMPILED_MUTEX`) — Rust side enforces this. There is no
// process-wide lock here; we trust the Rust caller.
//
// One verify graph per depth (1..5) is held in `g_verify_compiled_by_depth`,
// populated on first use of each depth (first-use trace cost is one-time
// per process).
// =============================================================================

#include "mlx_qwen35_common.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
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
// and `mlx_qwen35_export_caches` for inspection.
//
// The verify graph is ONE compiled forward over T = D+1 tokens (see
// `mlx_qwen35_forward_batched_verify` in `mlx_qwen35.cpp`). This file's
// per-depth `g_verify_compiled_by_depth` table dispatches into that
// batched FFI; the heavy compile lives inside the batched-verify FFI,
// so this file's closure does shape validation + FFI marshalling only.
// =============================================================================

extern "C" void mlx_qwen35_forward_compiled(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    mlx_array** output_logits,
    int* cache_offset_out);

extern "C" int mlx_qwen35_get_cache_offset();

extern "C" int mlx_qwen35_is_compile_inited();

// Read the LAST stashed `g_last_hidden` (refcounted clone) from the main
// flat-path compiled state. To support chained MTP cycles the Rust caller
// slices position `K` (= number of accepted drafts) of the verify hiddens
// to seed the next cycle's first MTP draft — `verify_hidden[K]` is the
// prediction context for the committed token at position K+1 (bonus on
// full-accept, residual on rejection), matching the MTP head's training
// contract `(prev_hidden, embed(next_token)) -> next-next logits`.
//
// Returns null when the main path is uninitialised OR no forward has run
// since the last reset.
extern "C" void mlx_qwen35_export_last_hidden(mlx_array** out);

// One-shot batched verify forward. Runs the entire D+1-token verify on a
// single compiled graph and emits `[1, D+1, vocab]` logits +
// `[1, D+1, hidden]` hiddens + `[1, D+1]` argmax ids. Lives in
// `mlx_qwen35.cpp` because the graph references the main-path's
// `g_compiled_caches` / `g_offset_int` directly. Tape-replay arming is
// consumed via the existing `g_tape_recording_armed` global.
extern "C" void mlx_qwen35_forward_batched_verify(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_logits,
    mlx_array** out_hiddens,
    mlx_array** out_argmax);

extern "C" void mlx_qwen35_forward_batched_verify_argmax_only(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_hiddens,
    mlx_array** out_argmax);

extern "C" void mlx_qwen35_forward_batched_verify_sparse_target(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    float temperature,
    int top_k,
    float top_p,
    int sampler_mode,
    mlx_array** out_hiddens,
    mlx_array** out_target_ids,
    mlx_array** out_target_probs);

// Paged-pool sibling of `mlx_qwen35_forward_batched_verify`.
// Reads K/V from `g_dense_k_pools[]` / `g_dense_v_pools[]` instead of
// the BHTD `g_compiled_caches[]`. Caller MUST construct `offset_arr`,
// `block_table`, `slot_mapping`, `seq_lens`, and `cu_seqlens_q` (see
// the C++ docstring on the definition for shapes). Returns logits,
// hiddens, and `[1, D+1]` target argmax ids.
extern "C" void mlx_qwen35_forward_batched_verify_paged(
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
    mlx_array** out_argmax);

// Paged-pool partial-accept rollback machinery (paged siblings of
// `mlx_qwen35_compiled_snapshot_linear_caches` /
// `mlx_qwen35_compiled_restore_linear_caches`). Defined in
// `mlx_qwen35.cpp` where they have direct access to
// `g_dense_paged_linear_caches`. Used by the MTP-on-paged gate to
// rollback the GDN recurrent + conv state when the verify accepts
// fewer than `depth + 1` tokens.
extern "C" void mlx_qwen35_compiled_snapshot_paged_linear_caches();
extern "C" void mlx_qwen35_compiled_restore_paged_linear_caches();
extern "C" void mlx_qwen35_compiled_replay_paged_linear_caches_for_accept(
    int accepted_steps, int depth);

// Eagerly compile the batched verify graphs for both `WithTape=false`
// and `WithTape=true` over depths {1..5}. Defined in `mlx_qwen35.cpp`
// where it has direct access to the main path's globals
// (`g_compiled_caches`, `g_offset_int`, tape accumulators) for
// snapshot/restore. Best-effort: failures are logged + swallowed.
extern "C" void mlx_qwen35_prewarm_verify_compiled();

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
static int g_mtp_offset_int = 0;                  // local MTP cache write slot
// Start-of-cycle offset (= main offset captured by `begin_cycle`).
// The MTP attn_mask must mask out positions `[0..g_mtp_chain_start_int)`
// because the K/V at those slots is zero (the MTP cache is zero-initialised
// per cycle). Without that floor the softmax weight on the real chain K/V
// is diluted by `1 / chain_start` because every zero-K position contributes
// `exp(0) = 1` while the real-K position contributes `exp(q·k)` — a
// long-prefix decode (chain_start ~ 1024) reduces the chain's attention
// contribution to ~13%. Matches MTPLX's empty-`KVCache()` semantics where
// the chain is the only thing attended.
static int g_mtp_chain_start_int = 0;
static bool g_mtp_compile_inited = false;

// MTP K/V cache headroom.
//
// The MTP K/V cache is allocated `max_kv_len + MTP_CACHE_HEADROOM` slots.
// `begin_cycle` re-anchors the draft offset to `g_mtp_committed_len`
// (≤ sequence length ≤ max_kv_len) and the draft steps `slice_update`
// K/V at `[g_mtp_committed_len .. g_mtp_committed_len + depth)` BEFORE
// the commit's capacity check runs; a commit then writes
// `[g_mtp_committed_len .. g_mtp_committed_len + (depth + 2))`. Near the
// tail of a long generation `g_mtp_committed_len` approaches max_kv_len,
// so without headroom either write can run off the end of an exactly
// `max_kv_len`-sized buffer. `MTP_CACHE_HEADROOM` must cover the largest
// single draft/commit write: max draft depth 5 + the 2 boundary tokens
// of a commit = 7. We round up to 16 for alignment and slack. The
// draft / commit graphs all read the buffer's actual
// `max_kv_len` dimension via `inputs[...].shape(2)`, so a larger buffer
// just makes those graphs compile against the larger shape — the
// attention mask `arange(0, buffer_len)` and `slice_update` operate over
// the bigger buffer correctly. The commit FFI's capacity check is kept
// as defense-in-depth; with this headroom it should never trip.
constexpr int MTP_CACHE_HEADROOM = 16;

inline array draft_lm_head_proj(const array& x, const MTPCompileConfig& cfg) {
  if (has_weight("mtp_draft_lm_head.weight")) {
    return linear_proj(x, "mtp_draft_lm_head");
  }
  return cfg.tie_word_embeddings
      ? linear_proj(x, "embedding")
      : linear_proj(x, "lm_head");
}

// Committed-history MTP cache policy.
//
// `g_mtp_committed_len` is the number of local MTP cache slots that hold EXACT
// committed K/V, computed from the target model's post-final-norm hidden +
// input embedding of each committed token. Absolute RoPE positions are
// `g_mtp_position_base_int + local_slot`.
//
// Under the `committed` history policy the MTP cache PERSISTS across
// cycles (`begin_cycle` no longer zeroes it). Each cycle's
// `mlx_qwen35_mtp_compiled_commit` appends exact slots for the newly committed
// token sequence and advances this counter.
//
// `begin_cycle` then re-anchors the draft offset to this counter so the
// next cycle's draft steps write at local `[g_mtp_committed_len ..]` and the
// draft attention mask spans the compact committed prefix `[0 .. offset]`.
static int g_mtp_committed_len = 0;
// Absolute real-sequence position of local MTP cache slot 0. This is zero for
// full committed-history mode. In MTPLX-style last_window mode the cache stores
// only a prompt tail locally at slots `[0 .. g_mtp_committed_len)`, while RoPE
// still uses positions `[g_mtp_position_base_int .. base + len)`.
static int g_mtp_position_base_int = 0;

static inline int mtp_absolute_pos(int local_pos) {
  return g_mtp_position_base_int + local_pos;
}

// =====================================================================
// Draft graph: traced once, reused across all D draft steps.
//
// Inputs (vector order matters — compile keys on shapes only, but the
// closure captures positional indexes):
//   [0]                prev_hidden     [1, 1, hidden]  bf16
//   [1]                prev_emb        [1, 1, hidden]  bf16
//   [2]                offset_arr      [1]   int32 (local slice_update
//                                                 slot; advances by 1
//                                                 per draft step).
//   [3]                rope_offset_arr [1]   int32 (absolute RoPE
//                                                 position for the local
//                                                 slot).
//   [4]                chain_start_arr [1]   int32 (start of this MTP
//                                                 draft chain; constant
//                                                 across all D steps of
//                                                 one cycle, snapshotted
//                                                 by `begin_cycle`). Used
//                                                 to mask out the zero
//                                                 K/V slots in
//                                                 [0..chain_start) which
//                                                 would otherwise dilute
//                                                 the softmax weight on
//                                                 the real chain K/V by
//                                                 ~1/chain_start.
//   For each MTP layer j in [0, n_mtp_layers):
//     [5 + j*2 + 0]    K cache         [1, Hkv, max_kv_len, head_dim]
//     [5 + j*2 + 1]    V cache         [1, Hkv, max_kv_len, head_dim]
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
  auto prev_hidden     = inputs[0];      // [1, 1, hidden]
  auto prev_emb        = inputs[1];      // [1, 1, hidden]
  auto offset_arr      = inputs[2];      // [1] int32, local cache slot
  auto rope_offset_arr = inputs[3];      // [1] int32, absolute RoPE pos
  auto chain_start_arr = inputs[4];      // [1] int32

  // Mirror Qwen3_5MTPModule::forward (dense Rust path):
  //   h_norm = pre_fc_norm_hidden(prev_hidden)
  //   e_norm = pre_fc_norm_embedding(prev_emb)
  //   h      = fc(concat([e_norm, h_norm], axis=-1))
  //   for layer in mtp.layers: h = layer(h, mask=None, cache=...)
  //   return norm(h)
  auto h_norm = fast::rms_norm(prev_hidden,
                               get_weight("mtp.pre_fc_norm_hidden.weight"),
                               cfg.rms_norm_eps);
  auto e_norm = fast::rms_norm(prev_emb,
                               get_weight("mtp.pre_fc_norm_embedding.weight"),
                               cfg.rms_norm_eps);

  // Concat along the hidden axis → [1, 1, 2*hidden]. Order is
  // `[embedding, hidden]` — MTPLX `concat_order` default
  // `"embedding_hidden"`; the bias-free `mtp.fc` weight columns are
  // trained for that block layout.
  auto concat3d = concatenate({e_norm, h_norm}, 2);
  // mtp.fc projects 2*hidden → hidden. linear_proj operates on 2D
  // [B*T, in_features], so we squeeze the time dim to match.
  auto concat2d = reshape(concat3d, {1, cfg.hidden_size * 2});
  auto h2d = linear_proj(concat2d, "mtp.fc");          // [1, hidden]

  // Build the attention mask for the MTP layers. MTP draft steps run
  // ONE token per call, and the cache offset advances by 1 per draft
  // step. The mask is `[1, 1, 1, max_kv_len]` additive bf16:
  //   0 (allow)  iff `chain_start <= pos <= offset_arr`
  //   -inf       elsewhere
  // Why both bounds: the MTP K/V cache is zero-initialised by
  // `begin_cycle`; only positions `[chain_start..offset_arr]` hold real
  // K/V (written by THIS cycle's draft steps so far). The earlier
  // implementation used only `pos <= offset_arr`, which admitted ALL
  // zero K/V at `[0..chain_start)` and diluted the real chain's softmax
  // weight (one real K vs `chain_start` zero K → ~1/chain_start weight
  // on the real K for long-prefix decode). MTPLX gets this for free by
  // using mlx-lm's `KVCache()` which starts empty (`self.keys = None`)
  // — see `MTPLX/mtplx/mtp_patch.py:638` `make_mtp_cache`.
  int max_kv_len = inputs[5].shape(2);  // first K-cache's max_kv_len
  auto positions = arange(0, max_kv_len, mlx::core::int32);
  auto valid_mask = logical_and(greater_equal(positions, chain_start_arr),
                                less_equal(positions, offset_arr));
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
  // `mtp.layers.{j}` and matches the Rust `apply_weights` flow.
  for (int j = 0; j < cfg.n_mtp_layers; j++) {
    std::string lp = "mtp.layers." + std::to_string(j);

    auto normed = fast::rms_norm(h2d, get_weight(lp + ".input_layernorm.weight"),
                                 cfg.rms_norm_eps);

    const auto& kk = inputs[5 + j * 2];
    const auto& kv = inputs[5 + j * 2 + 1];
    auto res = attn_pure_fn_arr_rope_write_offset(
        normed, lp, kk, kv, attn_mask, rope_offset_arr, offset_arr, cfg);
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
  auto logits = draft_lm_head_proj(h_norm_final, cfg);

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

// MTP draft graph. Safe to compile because `mtp_draft_decode_fn` takes
// offset_arr as an array input.
//
// `mtp_draft_decode_fn` reads the MTP head weights via `get_weight("mtp.*")`
// inside the trace, so the weights are baked into the cached tape. Held in a
// resettable FILE-SCOPE global (NOT a function-local static, which cannot be
// nulled) and compiled through `compile_resettable_weight_graph` so a model
// reload can null it via `mlx_qwen35_mtp_invalidate_compiled_graphs()` and
// force a re-trace against the live registry. Deliberately survives the
// per-turn `mlx_qwen35_mtp_compiled_reset()` (cross-turn reuse) — nulled ONLY
// on reload.
static std::function<std::vector<array>(const std::vector<array>&)>
    g_mtp_draft_decode_compiled{};

static std::function<std::vector<array>(const std::vector<array>&)>&
compiled_mtp_draft_decode() {
  if (!g_mtp_draft_decode_compiled) {
    g_mtp_draft_decode_compiled =
        compile_resettable_weight_graph(mtp_draft_decode_fn);
  }
  return g_mtp_draft_decode_compiled;
}

// =====================================================================
// Committed-history commit graph.
//
// Computes the MTP layer-0 attention K/V for M committed tokens and
// writes them into the persistent MTP KV cache. One graph per cycle,
// compiled once per M (statically unrolled by template `M`).
//
// `M = K+2` on the default Step-A path — the FULL committed sequence
// `[last_committed_id, d_0..d_{K-1}, boundary]` emitted by one outer
// iteration. The boundary token (bonus on full accept, residual on
// reject) IS committed here so the MTP prefix advances by exactly M
// per cycle, matching the real decode sequence length — no prefix
// compression, no RoPE drift.
//
// The committed K/V for a token `x` MUST be bit-compatible with what a
// draft step at that absolute position would have produced — drafts in
// the next cycle attend over these slots, and a mis-keyed slot
// corrupts the draft attention. We therefore reuse the EXACT op
// sequence of `mtp_draft_decode_fn`'s per-token body (norm / fc /
// k_proj / k_norm / RoPE) up to but NOT including attention output:
// only K and V are needed to fill the cache.
//
// Token/hidden pairing: row `i` of `hidden_seq` already holds the hidden
// of the token BEFORE committed token `i` (the MTP `MTP(h(t), emb(t+1))`
// contract), and row `i` of `gathered_embs` holds the embedding of
// committed token `i`. The pairing is done Rust-side in
// `commit_mtp_compiled`, so this graph just consumes row `i` of each
// array at slot `i` — `fc([e_norm(emb_i), h_norm(hidden_i)])`.
//
// Inputs (vector order):
//   [0]              hidden_seq      [1, M, hidden] bf16
//                                    (hidden of the token BEFORE each
//                                     committed token, pre-paired
//                                     Rust-side).
//   [1]              gathered_embs   [1, M, hidden] bf16
//                                    (input embedding of each committed
//                                     token, pre-gathered Rust-side).
//   [2]              base_offset_arr [1] int32 — local cache slot of row 0
//                                    (= g_mtp_committed_len).
//                                    Threaded as an array input to keep
//                                    the compile cache shape-stable,
//                                    exactly like the draft graph's
//                                    offset_arr.
//   [3]              rope_base_arr   [1] int32 — absolute RoPE position of
//                                    row 0 (= position_base + local slot).
//   For each MTP layer j in [0, n_mtp_layers):
//     [4 + j*2 + 0]  K cache         [1, Hkv, max_kv_len, head_dim]
//     [4 + j*2 + 1]  V cache         [1, Hkv, max_kv_len, head_dim]
//
// Outputs (vector order):
//   For each MTP layer j:
//     [j*2 + 0]      new K cache (M slots written)
//     [j*2 + 1]      new V cache (M slots written)
//
// ALL M slots are written and the C++ caller advances
// `g_mtp_committed_len` by exactly M.
//
// `n_mtp_layers == 1` for the dense MTP target, but the loop is general.
// =====================================================================
template <int M>
static std::vector<array> mtp_commit_fn(const std::vector<array>& inputs) {
  static_assert(M >= 1 && M <= 7,
                "mtp_commit_fn: M must be in [1, 7]");
  const auto& cfg = g_mtp_config;
  auto hidden_seq    = inputs[0];   // [1, M, hidden]
  auto gathered_embs = inputs[1];   // [1, M, hidden]
  auto base_offset   = inputs[2];   // [1] int32, local cache slot
  auto rope_base     = inputs[3];   // [1] int32, absolute RoPE position

  // Cache vector that gets rewritten across the M positions.
  std::vector<array> caches;
  caches.reserve(cfg.n_mtp_layers * 2);
  for (int j = 0; j < cfg.n_mtp_layers * 2; j++) {
    caches.push_back(inputs[4 + j]);
  }

  // Statically unrolled over the M committed positions.
  for (int i = 0; i < M; i++) {
    // Slice position `i` → [1, 1, hidden] (matches the draft path's
    // [1, 1, hidden] prev_hidden / prev_emb contract). Row `i` of
    // `hidden_seq` is already h(prev(committed_token_i)) and row `i` of
    // `gathered_embs` is emb(committed_token_i) — paired Rust-side.
    auto h_i = slice(hidden_seq, {0, i, 0},
                     {1, i + 1, cfg.hidden_size});      // [1, 1, hidden]
    auto e_i = slice(gathered_embs, {0, i, 0},
                     {1, i + 1, cfg.hidden_size});      // [1, 1, hidden]

    // Local cache slot and absolute RoPE position may differ under the
    // MTPLX-style last_window policy: the cache stores a compact tail while
    // RoPE remains anchored to the real sequence.
    auto local_pos_i = base_offset + array(i, mlx::core::int32);
    auto rope_pos_i  = rope_base + array(i, mlx::core::int32);

    // ---- MTP head pre-attention body (mirrors mtp_draft_decode_fn) ----
    // h_norm = pre_fc_norm_hidden(h_i); e_norm = pre_fc_norm_embedding(e_i)
    // h2d    = fc(concat([e_norm, h_norm], axis=-1))   ([embedding,hidden])
    auto h_norm = fast::rms_norm(h_i,
                                 get_weight("mtp.pre_fc_norm_hidden.weight"),
                                 cfg.rms_norm_eps);
    auto e_norm = fast::rms_norm(e_i,
                                 get_weight("mtp.pre_fc_norm_embedding.weight"),
                                 cfg.rms_norm_eps);
    auto concat3d = concatenate({e_norm, h_norm}, 2);   // [1, 1, 2*hidden]
    auto concat2d = reshape(concat3d, {1, cfg.hidden_size * 2});
    auto h2d = linear_proj(concat2d, "mtp.fc");          // [1, hidden]

    // Layer-0 (only layer) input_layernorm → K/V projection.
    std::string lp = "mtp.layers.0";
    auto normed = fast::rms_norm(h2d, get_weight(lp + ".input_layernorm.weight"),
                                 cfg.rms_norm_eps);

    std::string pfx = lp + ".self_attn.";
    int B = normed.shape(0);
    auto keys   = linear_proj(normed, pfx + "k_proj");
    auto values = linear_proj(normed, pfx + "v_proj");
    if (has_weight(pfx + "k_proj.bias")) keys   = keys   + get_weight(pfx + "k_proj.bias");
    if (has_weight(pfx + "v_proj.bias")) values = values + get_weight(pfx + "v_proj.bias");

    keys   = reshape(keys,   {B, 1, cfg.num_kv_heads, cfg.head_dim});
    values = reshape(values, {B, 1, cfg.num_kv_heads, cfg.head_dim});

    // k_norm then RoPE at absolute position `rope_pos_i` — identical to the
    // draft graph so the committed K matches a draft-time query at the same
    // real sequence position.
    keys = fast::rms_norm(keys, get_weight(pfx + "k_norm.weight"),
                          cfg.rms_norm_eps);
    keys = fast::rope(keys, cfg.rope_dims, false, cfg.rope_theta, 1.0f, rope_pos_i);

    keys   = transpose(keys,   {0, 2, 1, 3});   // [B, Hkv, 1, D]
    values = transpose(values, {0, 2, 1, 3});

    // slice_update writes the single K/V slot at compact local cache slot
    // `local_pos_i`.
    caches[0] = mlx::core::slice_update(caches[0], keys,   local_pos_i, {2});
    caches[1] = mlx::core::slice_update(caches[1], values, local_pos_i, {2});
  }

  std::vector<array> result;
  result.reserve(cfg.n_mtp_layers * 2);
  for (auto& c : caches) result.push_back(std::move(c));
  return result;
}

using CommitFn = std::function<std::vector<array>(const std::vector<array>&)>;

// Valid committed-token counts:
//   * Step-A path: M = K+2, K ∈ [0, depth], depth ∈ [1, 5] → [2, 7].
//   * Chained committed-history path: M = K+1 → [1, 6].
constexpr int MIN_COMMIT_M = 1;
constexpr int MAX_COMMIT_M = 7;

// Per-M compiled commit graph table. One entry per M ∈ {1..7}; the one-token
// case is needed when chained committed-history rejects every draft and only
// the residual boundary is newly committed. The unrolled body differs per M.
//
// `mtp_commit_fn<M>` reads MTP weights via `get_weight("mtp.*")` inside the
// trace, so the weights are baked into the cached tape. This table is the
// resettable backing for those weight-baking graphs: each slot is lazily
// compiled through `compile_resettable_weight_graph` (UNIQUE erasable fun_id)
// and nulled ONLY on reload by `mlx_qwen35_mtp_invalidate_compiled_graphs()`.
// It deliberately SURVIVES the per-turn `mlx_qwen35_mtp_compiled_reset()`
// (cross-turn reuse) — DO NOT null it there (that reset nulls only the cheap
// `g_commit_compiled_by_m` dispatcher table below).
static std::array<CommitFn, MAX_COMMIT_M - MIN_COMMIT_M + 1>
    g_commit_graph_by_m{};

template <int M>
static CommitFn& compiled_mtp_commit() {
  static_assert(M >= MIN_COMMIT_M && M <= MAX_COMMIT_M,
                "compiled_mtp_commit: M out of range");
  auto& slot = g_commit_graph_by_m[M - MIN_COMMIT_M];
  if (!slot) {
    slot = compile_resettable_weight_graph(&mtp_commit_fn<M>);
  }
  return slot;
}

static CommitFn make_commit_dispatcher(int m) {
  switch (m) {
    case 1: return [](const std::vector<array>& in) {
      return compiled_mtp_commit<1>()(in);
    };
    case 2: return [](const std::vector<array>& in) {
      return compiled_mtp_commit<2>()(in);
    };
    case 3: return [](const std::vector<array>& in) {
      return compiled_mtp_commit<3>()(in);
    };
    case 4: return [](const std::vector<array>& in) {
      return compiled_mtp_commit<4>()(in);
    };
    case 5: return [](const std::vector<array>& in) {
      return compiled_mtp_commit<5>()(in);
    };
    case 6: return [](const std::vector<array>& in) {
      return compiled_mtp_commit<6>()(in);
    };
    case 7: return [](const std::vector<array>& in) {
      return compiled_mtp_commit<7>()(in);
    };
    default: return CommitFn{};
  }
}

static std::array<CommitFn, MAX_COMMIT_M - MIN_COMMIT_M + 1>
    g_commit_compiled_by_m{};

static const CommitFn& get_or_make_commit_fn(int m) {
  auto& slot = g_commit_compiled_by_m[m - MIN_COMMIT_M];
  if (!slot) {
    slot = make_commit_dispatcher(m);
  }
  return slot;
}

// =====================================================================
// Verify graphs: per-depth dispatcher.
//
// One entry per depth ∈ {1..5}. The closure for depth D dispatches to
// `mlx_qwen35_forward_batched_verify` which runs a SINGLE compiled graph
// over T = D+1 tokens, emitting `[1, D+1, vocab]` logits and
// `[1, D+1, hidden]` post-final-norm hiddens in one launch. The FFI also
// advances the main `g_offset_int` by D+1 and writes the D+1 KV slots in
// place via `slice_update`.
//
// depth > 5 is rejected. We use a fixed-size array indexed by depth - 1.
//
// The per-depth `g_verify_compiled_by_depth` table is retained for two
// reasons: (a) depth validation lives one place (the slot's existence
// implies `depth ∈ [1, 5]`), (b) the per-depth closure still owns the
// `seq_len == depth + 1` runtime check that catches shape drift from
// the Rust caller. The actual compiled graph cache lives inside
// `mlx_qwen35.cpp` (see `compiled_verify_batched_notape` /
// `compiled_verify_batched_tape`) where it's keyed on input shape by
// `mlx::core::compile` itself.
// =====================================================================

constexpr int MAX_VERIFY_DEPTH = 5;
using VerifyFn = std::function<std::vector<array>(
    const array&, const array&)>;
static std::array<VerifyFn, MAX_VERIFY_DEPTH> g_verify_compiled_by_depth{};

// Process-once gate for the heavy prewarm path.
//
// `mlx_qwen35_mtp_compiled_prewarm_verify` must run AT MOST ONCE per
// process (the heavy `mlx::core::compile` work is ~1.5s and reusing the
// cached compile is just a few cycles per call). The per-depth dispatch
// closures capture only their fixed depth and are safe to retain across
// per-turn MTP reset/re-init, so reset does not re-arm this flag.
static std::atomic<bool> g_prewarm_done{false};

// Build a verify closure for a fixed depth. Captures NO per-call state.
// The closure expects `input_ids` of shape `[1, depth+1]` and the
// `embedding_weight` from the model, and returns
// `{logits[1, depth+1, vocab], hiddens[1, depth+1, hidden_size]}` via one
// call to `mlx_qwen35_forward_batched_verify` (a single compiled graph
// over T = depth+1 tokens). Tape-replay side-channels are populated by the
// same dispatch when armed via `mlx_qwen35_compiled_tape_arm`.
//
// The two entrypoints `mlx_qwen35_mtp_verify_compiled` (logits only) and
// `mlx_qwen35_mtp_verify_compiled_with_hidden` both consume the same
// `{logits, hiddens}` tuple — the logits-only entrypoint drops the second.
static VerifyFn make_verify_fn(int depth) {
  return [depth](const array& input_ids, const array& embedding_weight)
             -> std::vector<array> {
    int seq_len = input_ids.shape(1);
    if (seq_len != depth + 1) {
      throw std::runtime_error(
          "mlx_qwen35_mtp_verify: input_ids time dim (" +
          std::to_string(seq_len) + ") must equal depth+1 (" +
          std::to_string(depth + 1) + ")");
    }

    // Heap-allocate temporaries because the cross-translation-unit FFI
    // accepts `mlx_array*` pointers, not const-array references.
    array tok_copy = input_ids;
    array emb_copy = embedding_weight;
    mlx_array* logits_ptr = nullptr;
    mlx_array* hidden_ptr = nullptr;
    mlx_array* argmax_ptr = nullptr;
    mlx_qwen35_forward_batched_verify(
        reinterpret_cast<mlx_array*>(&tok_copy),
        reinterpret_cast<mlx_array*>(&emb_copy),
        depth,
        &logits_ptr,
        &hidden_ptr,
        &argmax_ptr);
    if (!logits_ptr || !hidden_ptr || !argmax_ptr) {
      // Drop any half-allocated handle before erroring.
      if (logits_ptr) {
        delete reinterpret_cast<array*>(logits_ptr);
      }
      if (hidden_ptr) {
        delete reinterpret_cast<array*>(hidden_ptr);
      }
      if (argmax_ptr) {
        delete reinterpret_cast<array*>(argmax_ptr);
      }
      throw std::runtime_error(
          "mlx_qwen35_mtp_verify: batched verify forward returned null");
    }
    array logits = *reinterpret_cast<array*>(logits_ptr);
    delete reinterpret_cast<array*>(logits_ptr);
    array hiddens = *reinterpret_cast<array*>(hidden_ptr);
    delete reinterpret_cast<array*>(hidden_ptr);
    array argmax_ids = *reinterpret_cast<array*>(argmax_ptr);
    delete reinterpret_cast<array*>(argmax_ptr);
    return {logits, hiddens, argmax_ids};
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
// per the Rust invariant — this affects RoPE only if the helper
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
    // Headroom: the MTP K/V buffer is allocated
    // `max_kv_len + MTP_CACHE_HEADROOM` slots so a near-tail draft /
    // commit `slice_update` (up to depth + 2 = 7 slots past
    // g_mtp_committed_len) can never run off the end. `g_mtp_config.max_kv_len`
    // stores the ACTUAL buffer length so the commit FFI's capacity check
    // matches the buffer the draft/commit graphs `slice_update` into.
    const int mtp_buffer_len = max_kv_len + MTP_CACHE_HEADROOM;
    g_mtp_config.max_kv_len              = mtp_buffer_len;
    g_mtp_config.batch_size              = batch_size;
    g_mtp_config.n_mtp_layers            = n_mtp_layers;
    g_mtp_config.mtp_fa_layer_idx        = std::max(full_attention_interval - 1, 0);

    // Fresh per-MTP-layer KV caches. All MTP layers are full-attention,
    // so each entry is a [B, Hkv, mtp_buffer_len, D] zero buffer at bf16.
    // We DO NOT seed from the main path's caches: MTP draft steps
    // build their OWN KV context from the drafted tokens and discard
    // it on acceptance failure, so seeding from the main path's
    // committed prefix would corrupt the MTP layer's attention.
    g_mtp_compiled_caches.clear();
    g_mtp_compiled_caches.reserve(n_mtp_layers * 2);
    for (int j = 0; j < n_mtp_layers; j++) {
      auto kk = zeros({batch_size, num_kv_heads, mtp_buffer_len, head_dim},
                      mlx::core::bfloat16);
      auto vv = zeros({batch_size, num_kv_heads, mtp_buffer_len, head_dim},
                      mlx::core::bfloat16);
      g_mtp_compiled_caches.push_back(std::move(kk));
      g_mtp_compiled_caches.push_back(std::move(vv));
    }

    // Mirror the main path's current offset. Draft steps will advance
    // `g_mtp_offset_int` independently — the main offset is untouched
    // by drafting. `g_mtp_chain_start_int` snapshots this same offset
    // at every `begin_cycle` and is the LOWER bound the draft attn_mask
    // uses to exclude the zero K/V slots in `[0..chain_start)`. The
    // seed here is only relevant if a caller ever drafted before
    // calling begin_cycle — the Rust side does not, but the seed keeps
    // the two counters consistent for any debug introspection.
    g_mtp_offset_int = mlx_qwen35_get_cache_offset();
    g_mtp_chain_start_int = g_mtp_offset_int;
    // Committed-history policy. The MTP KV cache is its OWN sequence,
    // independent of the main model's absolute positions: the
    // prompt-prefix tokens are NEVER given MTP K/V (only tokens
    // produced during decode are committed). The committed-prefix
    // counter therefore starts at 0 — the first
    // `mlx_qwen35_mtp_compiled_commit` after the first verify writes
    // slots `[0 .. K+1)`, and subsequent cycles append.
    //
    // Starting the counter at `main_offset` would leave the slots
    // `[0 .. main_offset)` permanently zero yet inside the draft
    // attention window (`begin_cycle` sets `chain_start = 0`), diluting
    // the softmax weight on the real committed K/V by ~1/main_offset.
    g_mtp_committed_len = 0;
    g_mtp_position_base_int = 0;

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

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_mtp_draft_compiled: ENTER (per-step) "
            "mtp_offset=%d rope_offset=%d chain_start=%d "
            "fc_concat_order=[embedding,hidden]\n",
            g_mtp_offset_int, mtp_absolute_pos(g_mtp_offset_int),
            g_mtp_chain_start_int);
  }

  try {
    auto& prev_hidden = *reinterpret_cast<array*>(prev_hidden_ptr);
    auto& prev_emb    = *reinterpret_cast<array*>(prev_emb_ptr);

    std::vector<array> inputs;
    inputs.reserve(5 + g_mtp_config.n_mtp_layers * 2);
    inputs.push_back(prev_hidden);
    inputs.push_back(prev_emb);
    inputs.push_back(reshape(array(g_mtp_offset_int, mlx::core::int32), {1}));
    inputs.push_back(reshape(array(mtp_absolute_pos(g_mtp_offset_int), mlx::core::int32), {1}));
    inputs.push_back(reshape(array(g_mtp_chain_start_int, mlx::core::int32), {1}));
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
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_mtp_draft_compiled: EXIT OK "
              "new_mtp_offset=%d\n",
              g_mtp_offset_int);
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

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_mtp_verify_compiled: ENTER depth=%d\n",
            depth);
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
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_mtp_verify_compiled: EXIT OK depth=%d\n",
              depth);
    }
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
// Verify pass that ALSO exports the post-final-norm hidden state at EVERY
// verify position so the caller can chain MTP cycles without running a
// fresh main-model forward at each cycle's "Step A".
//
// Behaviourally identical to `mlx_qwen35_mtp_verify_compiled` for the
// logits output (and the same `g_compiled_caches[]` / `g_offset_int`
// mutation contract) plus one extra owned `mlx_array*` for ALL D+1
// per-position hiddens stacked along the time axis →
// `[1, depth+1, hidden_size]`. The hidden output is the verify graph's
// `[1, T, hidden]` post-final-norm slot returned directly.
//
// Why D+1 instead of just the last hidden: the Rust caller selects
// position `K` (= number of accepted drafts) — `verify_hidden[K]` is the
// prediction context for the committed token at position K+1 (bonus on
// full-accept, residual on rejection), matching the MTP head's training
// contract. Shipping only position D matches only when ALL drafts are
// accepted; partial-accept cycles would chain from the wrong context.
//
// Why an extra entrypoint instead of extending the existing one:
//   (a) callers that don't need the hidden keep the 2-output contract
//       and a free `nullptr` slot;
//   (b) explicit caller opt-in keeps the lazy MLX graph for the hidden
//       alive across the FFI boundary, avoiding a silent perf
//       regression for non-chained callers.
//
// The hidden's lifetime contract mirrors
// `mlx_qwen35_export_last_hidden`:
//   - The returned handle is a lazy MLX array whose graph references
//     the verify's per-step final_norm outputs. The caller MUST `eval()`
//     it (or consume it via a graph that does) before reading any
//     element. In practice the Rust caller slices position K first and
//     evals only the resulting `[1, 1, hidden]` slice, so only one
//     per-position final_norm path is realised on-device.
//   - The caller MUST NOT call `mlx_qwen35_compiled_reset()` between
//     export and eval — that reset would clear `g_compiled_caches`
//     whose inputs the hidden depends on via the cached graph.
//
// `*out_hiddens` is nullptr on failure (matches the logits-only
// FFI's failure semantics so the Rust caller can fall back to Step A
// when chaining is unavailable). The caller MUST null-check both
// outputs before consuming.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax(
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
  if (!input_ids_ptr || !embedding_weight_ptr || !out_logits ||
      !out_hiddens || !out_argmax) {
    return;
  }
  if (depth < 1 || depth > MAX_VERIFY_DEPTH) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax: depth %d "
            "outside [1, %d]\n",
            depth, MAX_VERIFY_DEPTH);
    fflush(stderr);
    return;
  }

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax: ENTER "
            "depth=%d\n",
            depth);
  }

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != depth + 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax: input_ids "
              "shape must be [1, depth+1=%d], got ndim=%d shape=[%lld,%lld]\n",
              depth + 1, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    // Run the per-depth verify. After this returns, `g_compiled_caches[]` /
    // `g_offset_int` have been advanced by D+1. The closure returns
    // `{logits[1, D+1, vocab], hiddens[1, D+1, hidden], argmax[1, D+1]}`.
    const auto& verify_fn = get_or_make_verify_fn(depth);
    auto outputs = verify_fn(input_ids, embedding_weight);
    if (outputs.size() < 3) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax: verify "
              "closure returned %zu outputs; expected 3 (logits, hiddens, argmax)\n",
              outputs.size());
      fflush(stderr);
      return;
    }
    *out_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    *out_hiddens = reinterpret_cast<mlx_array*>(new array(outputs[1]));
    *out_argmax = reinterpret_cast<mlx_array*>(new array(outputs[2]));
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax: "
              "EXIT OK depth=%d\n",
              depth);
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax: %s\n",
            e.what());
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_argmax) *out_argmax = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in "
            "mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax\n");
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_argmax) *out_argmax = nullptr;
  }
}

void mlx_qwen35_mtp_verify_compiled_with_hidden(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_logits,
    mlx_array** out_hiddens
) {
  mlx_array* out_argmax = nullptr;
  mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax(
      input_ids_ptr,
      embedding_weight_ptr,
      depth,
      out_logits,
      out_hiddens,
      &out_argmax);
  if (out_argmax) {
    delete reinterpret_cast<array*>(out_argmax);
  }
}

void mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax_only(
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
  if (depth < 1 || depth > MAX_VERIFY_DEPTH) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax_only: depth %d "
            "outside [1, %d]\n",
            depth, MAX_VERIFY_DEPTH);
    fflush(stderr);
    return;
  }

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax_only: ENTER "
            "depth=%d\n",
            depth);
  }

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != depth + 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax_only: input_ids "
              "shape must be [1, depth+1=%d], got ndim=%d shape=[%lld,%lld]\n",
              depth + 1, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    array tok_copy = input_ids;
    array emb_copy = embedding_weight;
    mlx_qwen35_forward_batched_verify_argmax_only(
        reinterpret_cast<mlx_array*>(&tok_copy),
        reinterpret_cast<mlx_array*>(&emb_copy),
        depth,
        out_hiddens,
        out_argmax);
    if (qwen35_common::mtp_trace_enabled() && *out_hiddens && *out_argmax) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax_only: "
              "EXIT OK depth=%d\n",
              depth);
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax_only: %s\n",
            e.what());
    fflush(stderr);
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_argmax) *out_argmax = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in "
            "mlx_qwen35_mtp_verify_compiled_with_hidden_and_argmax_only\n");
    fflush(stderr);
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_argmax) *out_argmax = nullptr;
  }
}

void mlx_qwen35_mtp_verify_compiled_with_hidden_and_sparse_target(
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
  if (depth < 1 || depth > MAX_VERIFY_DEPTH) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden_and_sparse_target: depth %d "
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
              "[MLX] mlx_qwen35_mtp_verify_compiled_with_hidden_and_sparse_target: input_ids "
              "shape must be [1, depth+1=%d], got ndim=%d shape=[%lld,%lld]\n",
              depth + 1, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    array tok_copy = input_ids;
    array emb_copy = embedding_weight;
    mlx_qwen35_forward_batched_verify_sparse_target(
        reinterpret_cast<mlx_array*>(&tok_copy),
        reinterpret_cast<mlx_array*>(&emb_copy),
        depth,
        temperature,
        top_k,
        top_p,
        sampler_mode,
        out_hiddens,
        out_target_ids,
        out_target_probs);
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in mlx_qwen35_mtp_verify_compiled_with_hidden_and_sparse_target: %s\n",
            e.what());
    fflush(stderr);
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_target_ids) *out_target_ids = nullptr;
    if (out_target_probs) *out_target_probs = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in "
            "mlx_qwen35_mtp_verify_compiled_with_hidden_and_sparse_target\n");
    fflush(stderr);
    if (out_hiddens) *out_hiddens = nullptr;
    if (out_target_ids) *out_target_ids = nullptr;
    if (out_target_probs) *out_target_probs = nullptr;
  }
}

// -----------------------------------------------------------------------------
// Pre-warm the per-depth verify dispatch closures AND the underlying
// MLX-compiled batched verify graphs.
//
// Wire this to fire IMMEDIATELY after `mlx_qwen35_mtp_compiled_init_from_main`
// returns 0 from the Rust caller. The work performed (first call only):
//   1. Populates `g_verify_compiled_by_depth[D-1]` for each D ∈ {1..5}
//      with its per-depth closure (saves a one-off `std::function` heap
//      allocation on the first verify of each depth).
//   2. Delegates to `mlx_qwen35_prewarm_verify_compiled()` (in
//      `mlx_qwen35.cpp`), which runs one dummy verify forward per
//      (depth, with_tape) pair to force `mlx::core::eval` of the
//      compiled graph outputs. After this returns, MLX's internal
//      compile cache holds 10 traces (5 depths × 2 tape variants) and
//      first-use of each shape during real verify cycles is a cache hit.
//
// Init runs once PER TURN, not once per process; the heavy work
// (~1.5s) is gated behind a once-flag so it fires exactly once per
// process. Per-turn reset does not clear the closure tables, so this
// guard stays process-once.
//
// No-op if MTP is uninitialised. Best-effort: any failure inside the
// underlying prewarm is logged + swallowed and the verify path simply
// falls back to lazy-at-first-use.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_compiled_prewarm_verify() {
  if (!g_mtp_compile_inited) {
    return;
  }
  // Atomic-bool gate. CAS-flipping `false -> true` is the entry permit.
  // Caller is serialised on `DENSE_COMPILED_MUTEX` (Rust-side), but using
  // an atomic keeps the check correct under any future relaxation.
  bool expected = false;
  if (!g_prewarm_done.compare_exchange_strong(expected, true)) {
    return;
  }
  // Step 1: populate the per-depth closures so first-use of each depth
  // doesn't allocate inside the verify call.
  try {
    for (int d = 1; d <= MAX_VERIFY_DEPTH; d++) {
      auto& slot = g_verify_compiled_by_depth[d - 1];
      if (!slot) {
        slot = make_verify_fn(d);
      }
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_mtp_compiled_prewarm_verify: closure "
            "population failed: %s\n", e.what());
    fflush(stderr);
    // Continue — the MLX-level prewarm below is still worth attempting.
  } catch (...) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_mtp_compiled_prewarm_verify: unknown "
            "exception during closure population\n");
    fflush(stderr);
  }
  // Step 2: force the heavy `mlx::core::compile` to run NOW (10 traces:
  // depths {1..5} × {WithTape=false, WithTape=true}). Snapshot/restore
  // of main-path state is handled inside that function.
  mlx_qwen35_prewarm_verify_compiled();
}

// -----------------------------------------------------------------------------
// Tear down MTP compiled state. Idempotent; safe to call on
// already-empty state. Does NOT touch the main path's globals — call
// `mlx_qwen35_compiled_reset` separately for that.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_compiled_reset() {
  g_mtp_compiled_caches.clear();
  g_mtp_offset_int = 0;
  g_mtp_chain_start_int = 0;
  // Committed-history counter resets with the rest of the MTP state so a
  // fresh turn starts with an empty committed prefix.
  g_mtp_committed_len = 0;
  g_mtp_position_base_int = 0;
  g_mtp_compile_inited = false;
  g_mtp_config = MTPCompileConfig{};
  // Keep the per-depth verify dispatchers across per-turn reset. They
  // capture only the static depth and call through the current global
  // compiled state, so retaining them avoids re-running heavy prewarm on
  // the next turn.
  // Drop the per-M commit dispatchers. The underlying weight-baking commit
  // graphs live in the file-scope `g_commit_graph_by_m` table and are
  // deliberately retained across the per-turn reset (they are nulled only
  // on model reload via mlx_qwen35_mtp_invalidate_compiled_graphs).
  for (auto& slot : g_commit_compiled_by_m) {
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
// Committed-history MTP cache policy: begin a fresh MTP draft cycle
// WITHOUT zeroing the persistent MTP K/V cache.
//
// A PERSISTENT committed-history cache (heads attend over the full
// committed prefix) nearly doubles draft acceptance vs. zeroing the
// cache each cycle and attending only over the current draft chain.
//
// Behavior:
//   - Do NOT zero / reallocate the caches. They persist across cycles;
//     `mlx_qwen35_mtp_compiled_commit` (called after each verify) keeps
//     `[0 .. g_mtp_committed_len)` filled with exact committed K/V.
//   - Re-anchor the draft offset to `g_mtp_committed_len` — the next
//     cycle's draft steps write at `[g_mtp_committed_len ..]`.
//   - Set `g_mtp_chain_start_int = 0`. The existing draft attn_mask
//     (`chain_start <= pos <= offset`) then collapses to plain causal
//     (`0 <= pos <= offset`), correct because the whole prefix
//     `[0 .. g_mtp_committed_len)` now holds real committed K/V.
//
// The `main_offset` argument is retained for FFI ABI stability; under
// the committed policy it is no longer used to re-anchor the offset
// (the commit fn drives `g_mtp_offset_int` / `g_mtp_committed_len`).
//
// No-op if MTP isn't initialised — the dispatcher Rust-side already
// gates the call on `mtp_active`, but defensive-checked here too.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_compiled_begin_cycle(int main_offset) {
  if (!g_mtp_compile_inited) return;
  (void)main_offset;  // committed-history policy: offset driven by commit
  g_mtp_offset_int = g_mtp_committed_len;
  g_mtp_chain_start_int = 0;
}

// Chained committed-history variant. The anchor token that seeds this
// cycle was already committed as the prior cycle's boundary, but the
// first MTP draft still needs to run `MTP(h(prev_anchor), emb(anchor))`
// so the draft graph can predict the token after the anchor. Start one
// slot before `g_mtp_committed_len` so that first draft overwrites the
// existing anchor slot in place instead of duplicating it at the end of
// the persistent prefix.
void mlx_qwen35_mtp_compiled_begin_chained_cycle(int main_offset) {
  if (!g_mtp_compile_inited) return;
  (void)main_offset;
  g_mtp_offset_int = std::max(g_mtp_committed_len - 1, 0);
  g_mtp_chain_start_int = 0;
}

// -----------------------------------------------------------------------------
// Append exact committed K/V to the persistent MTP cache.
//
// Called once per cycle, AFTER the accept loop has determined the
// accepted-draft count K and BEFORE the rollback. Writes M = K+2 exact
// MTP layer-0 K/V slots for the FULL committed sequence
// `[last_committed_id, d_0..d_{K-1}, boundary]` using the pre-assembled
// hidden / embedding rows, then advances `g_mtp_committed_len` by M.
//
// The boundary token (bonus on full accept, residual on reject) IS
// committed here. Its hidden (`verify_hiddens[:, K, :]`) is available at
// commit time, so the MTP prefix grows by exactly M = K+2 per cycle,
// matching the real decode sequence length (no prefix compression, no
// RoPE drift).
//
// Inputs:
//   - hidden_seq_ptr:    `[1, M, hidden]` bf16 — `hidden_seq[i]` is the
//                        hidden of the token BEFORE committed token `i`
//                        (the MTP `MTP(h(t), emb(t+1))` contract).
//                        Pre-assembled Rust-side: row 0 is the cycle's
//                        Step-A seed hidden, rows 1..K+1 are
//                        `verify_hiddens[:, 0:K+1, :]`.
//   - gathered_embs_ptr: `[1, M, hidden]` bf16 — input embedding of each
//                        committed token, pre-gathered Rust-side
//                        (avoids a quantized-embedding edge case in
//                        the graph).
//   - m:                 M = K+2 (or 1 for a one-token prompt-window seed).
//                        Both input arrays MUST have
//                        time dim M.
//
// SIDE EFFECTS: writes M K/V slots into `g_mtp_compiled_caches[]` at
// local slots `[g_mtp_committed_len .. g_mtp_committed_len+M)`, using
// absolute RoPE positions `[position_base + g_mtp_committed_len ..)`,
// then sets `g_mtp_committed_len += M` and
// `g_mtp_offset_int = g_mtp_committed_len`.
//
// Returns 0 on success (committed exactly M slots, `g_mtp_committed_len`
// advanced by M). Returns a distinct non-zero code on any failure
// (not-inited, bad args, capacity overflow, exception). On EVERY failure
// path `g_mtp_committed_len` is left UNCHANGED so the Rust caller can
// detect the desync and abort decode cleanly rather than anchoring the
// next cycle's drafts to a stale committed length.
//
// Failure codes:
//   1  MTP not initialised / null arg pointer
//   2  m outside [MIN_COMMIT_M, MAX_COMMIT_M]
//   3  capacity overflow (committed_len + m > max_kv_len)
//   4  std::exception thrown inside the commit graph
//   5  unknown (non-std) exception
// -----------------------------------------------------------------------------
int mlx_qwen35_mtp_compiled_commit(
    mlx_array* hidden_seq_ptr,
    mlx_array* gathered_embs_ptr,
    int m
) {
  if (!g_mtp_compile_inited) return 1;
  if (!hidden_seq_ptr || !gathered_embs_ptr) return 1;
  if (m < MIN_COMMIT_M || m > MAX_COMMIT_M) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_mtp_compiled_commit: m %d outside [%d, %d]\n",
            m, MIN_COMMIT_M, MAX_COMMIT_M);
    fflush(stderr);
    return 2;
  }
  // Guard the cache: writing M slots starting at g_mtp_committed_len
  // must stay inside the buffer. Compute the sum in 64-bit signed so the
  // capacity check itself cannot overflow.
  const int64_t projected_len =
      static_cast<int64_t>(g_mtp_committed_len) + static_cast<int64_t>(m);
  if (projected_len > static_cast<int64_t>(g_mtp_config.max_kv_len)) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_mtp_compiled_commit: committed_len %d + m %d "
            "exceeds max_kv_len %d — skipping commit\n",
            g_mtp_committed_len, m, g_mtp_config.max_kv_len);
    fflush(stderr);
    return 3;
  }

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_mtp_compiled_commit: ENTER m=%d "
            "committed_len=%d position_base=%d rope_base=%d\n",
            m, g_mtp_committed_len, g_mtp_position_base_int,
            mtp_absolute_pos(g_mtp_committed_len));
  }

  try {
    auto& hidden_seq    = *reinterpret_cast<array*>(hidden_seq_ptr);
    auto& gathered_embs = *reinterpret_cast<array*>(gathered_embs_ptr);

    std::vector<array> inputs;
    inputs.reserve(4 + g_mtp_config.n_mtp_layers * 2);
    inputs.push_back(hidden_seq);
    inputs.push_back(gathered_embs);
    inputs.push_back(reshape(array(g_mtp_committed_len, mlx::core::int32), {1}));
    inputs.push_back(reshape(
        array(mtp_absolute_pos(g_mtp_committed_len), mlx::core::int32), {1}));
    for (const auto& c : g_mtp_compiled_caches) {
      inputs.push_back(c);
    }

    const auto& commit_fn = get_or_make_commit_fn(m);
    if (!commit_fn) {
      throw std::runtime_error(
          "mlx_qwen35_mtp_compiled_commit: no commit dispatcher for m");
    }
    auto outputs = commit_fn(inputs);
    if (outputs.size() < static_cast<size_t>(g_mtp_config.n_mtp_layers * 2)) {
      throw std::runtime_error(
          "mlx_qwen35_mtp_compiled_commit: commit graph returned fewer "
          "outputs than expected");
    }
    for (int j = 0; j < g_mtp_config.n_mtp_layers * 2; j++) {
      g_mtp_compiled_caches[j] = outputs[j];
    }

    g_mtp_committed_len += m;
    g_mtp_offset_int = g_mtp_committed_len;

    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_mtp_compiled_commit: EXIT OK "
              "new_committed_len=%d position_base=%d new_mtp_offset=%d "
              "new_rope_offset=%d\n",
              g_mtp_committed_len, g_mtp_position_base_int, g_mtp_offset_int,
              mtp_absolute_pos(g_mtp_offset_int));
    }
    return 0;
  } catch (const std::exception& e) {
    // g_mtp_committed_len is advanced only after the graph fully
    // succeeds above, so a throw here leaves committed state unchanged.
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_mtp_compiled_commit: %s\n",
            e.what());
    fflush(stderr);
    return 4;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_mtp_compiled_commit\n");
    fflush(stderr);
    return 5;
  }
}

// -----------------------------------------------------------------------------
// Read accessor for the current absolute MTP RoPE offset (debugging /
// introspection from Rust unit tests).
// -----------------------------------------------------------------------------
int mlx_qwen35_mtp_get_offset() {
  return mtp_absolute_pos(g_mtp_offset_int);
}

// -----------------------------------------------------------------------------
// Read accessor for the persistent committed-history prefix length.
// -----------------------------------------------------------------------------
int mlx_qwen35_mtp_get_committed_len() {
  return g_mtp_committed_len;
}

// -----------------------------------------------------------------------------
// Configure the absolute position of local committed-history slot 0.
// Must be called before prompt-history seeding. It is intentionally local-slot
// preserving: callers still see/get/validate `g_mtp_committed_len` as the
// compact cache length.
// -----------------------------------------------------------------------------
void mlx_qwen35_mtp_set_position_base(int position_base) {
  g_mtp_position_base_int = std::max(position_base, 0);
}

int mlx_qwen35_mtp_get_position_base() {
  return g_mtp_position_base_int;
}

// Invalidate the compiled dense MTP draft graph and the per-M commit
// graphs so the next call re-traces against the CURRENT weight registry.
//
// Both `mtp_draft_decode_fn` and `mtp_commit_fn<M>` read the MTP head weights
// via `get_weight("mtp.*")` INSIDE the traced closure (NOT as compile inputs),
// so the captured weight arrays are baked into the cached tape. These graphs
// are PROCESS-WIDE and deliberately survive the per-turn
// `mlx_qwen35_mtp_compiled_reset()` (cross-turn reuse). On a model RELOAD the
// baked weights are stale, so a second same-shape MTP model loaded in the same
// process would draft/commit with the FIRST model's weights — silent
// corruption.
//
// Because every graph is compiled through `compile_resettable_weight_graph`,
// each carries a UNIQUE, erasable `fun_id`; assigning an empty
// `std::function{}` destroys the wrapper, ERASES its compile-cache entry, and
// forces a re-trace against the live registry on the next call. We null the
// RELOAD-scoped `g_commit_graph_by_m` (the actual weight-baking graphs), NOT
// the per-turn `g_commit_compiled_by_m` dispatcher table (cleared by the
// per-turn reset). Called transitively from
// `mlx_qwen35_invalidate_compiled_graphs()` (the dense reload entry point), so
// it runs INSIDE the Rust `COMPILED_WEIGHTS_RWLOCK` write critical section.
void mlx_qwen35_mtp_invalidate_compiled_graphs() {
  g_mtp_draft_decode_compiled =
      std::function<std::vector<array>(const std::vector<array>&)>{};
  for (auto& slot : g_commit_graph_by_m) {
    slot = CommitFn{};
  }
}

}  // extern "C"
