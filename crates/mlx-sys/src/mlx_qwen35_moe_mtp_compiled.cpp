// =============================================================================
// Qwen3.5 MoE MTP (Multi-Token Prediction) compiled draft + verify graphs.
//
// MoE twin of `mlx_qwen35_mtp_compiled.cpp` (W5 dense). Provides three FFI
// entrypoints that mirror the dense ones byte-for-byte:
//
//   - `mlx_qwen35_moe_mtp_compiled_init_from_main`: validates that the main
//     MoE compiled path is inited (via `mlx_qwen35_moe_is_compile_inited`),
//     validates `has_weight("mtp.norm.weight")`, allocates per-MTP-layer
//     KV caches sized to `max_kv_len`, snapshots config, and detects per-
//     MTP-layer quantization info from the loaded MTP weights (independent
//     of the main path's `g_layer_quant` / `g_dense_quant`).
//
//   - `mlx_qwen35_moe_mtp_draft_compiled`: one MTP draft step. Inputs are
//     `(prev_hidden, prev_emb)` — both `[1, 1, hidden]`. Returns the next
//     hidden state and the draft logits. Wrapped in `mlx::core::compile`
//     from the start, mirroring the MoE main compiled path
//     (`mlx_qwen35_moe.cpp:549`).
//
//   - `mlx_qwen35_moe_mtp_verify_compiled`: one verify pass on `depth+1`
//     tokens. Loops `mlx_qwen35_moe_forward` D+1 times (per-depth closure
//     table populated lazily) and stacks the per-step logits.
//
// MoE-specific divergences from the dense MTP file:
//
//   1. MLP branch: every MTP layer dispatches on `is_moe_layer(fa_idx, cfg)`
//      between a parameterised-prefix `sparse_moe_fn_pfx` (MoE routing,
//      switch_mlp + shared_expert) and a parameterised-prefix
//      `dense_mlp_fn_pfx` (plain SwiGLU). The per-MTP-layer quant info
//      comes from `g_mtp_layer_quant` / `g_mtp_dense_quant`, populated
//      from the loaded `mtp.layers.{j}.mlp.*` weights at init time —
//      independent of the main path's per-layer arrays.
//
//   2. The MoE branch uses `gather_mm` / `gather_qmm` which want 3D
//      pre-transposed expert weights `[E, in, out]`. The main MoE init
//      pre-computes these into a file-static map (`g_weight_transposes_3d`)
//      that is NOT visible across translation units. We maintain a private
//      transpose cache (`g_mtp_3d_transposes()`) populated at MTP init for
//      every `mtp.layers.{j}.mlp.switch_mlp.*.weight` key that exists in
//      `g_weights()` and is 3D.
//
//   3. The Rust wrapper docstring (in `qwen3_5_moe/model.rs`) MUST require
//      production callers to hold `MOE_COMPILED_MUTEX` (NOT the dense
//      `DENSE_COMPILED_MUTEX`) for the entire draft+verify cycle — the
//      verify path mutates the main MoE compiled caches in place via
//      `mlx_qwen35_moe_forward`.
//
// Locking otherwise mirrors the dense MTP file: no process-wide lock here,
// we trust the Rust caller.
// =============================================================================

#include "mlx_qwen35_common.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

using namespace qwen35_common;

// =============================================================================
// Cross-file shared state from `mlx_qwen35_moe.cpp`.
//
// Same rationale as the dense MTP file: go through the FFI surface rather
// than make the MoE main globals header-visible. See the long comment at
// the top of `mlx_qwen35_mtp_compiled.cpp` for the full reasoning.
// =============================================================================

extern "C" void mlx_qwen35_moe_forward(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    mlx_array** output_logits,
    int* cache_offset_out);

extern "C" int mlx_qwen35_moe_get_cache_offset();

extern "C" int mlx_qwen35_moe_is_compile_inited();

// W6.5 — read the LAST stashed `g_moe_last_hidden` (refcounted clone)
// from the MoE main compiled state. The verify graph in this file loops
// `mlx_qwen35_moe_forward` D+1 times; that function emits the hidden as
// graph output slot 2 (see `mlx_qwen35_moe.cpp` line 943) and stashes
// it into `g_moe_last_hidden` on every call.
//
// To support chained MTP cycles correctly we MUST capture the hidden
// after EACH of the D+1 iterations (not just the last) and surface them
// as a stacked `[1, D+1, hidden]` tensor. The Rust caller then slices
// position `K` (= number of accepted drafts) to seed the next cycle's
// first MTP draft — `verify_hidden[K]` is the prediction context for
// the committed-token-at-position-K+1 (bonus on full-accept, residual
// on rejection), matching the MTP head's training contract.
//
// Returns null when the main MoE path is uninitialised OR no forward
// has run since the last reset. Mirrors the public W6 Step-A seeding
// FFI; declared here too so the verify closure can thread it once per
// iteration without going through the FFI boundary twice.
extern "C" void mlx_qwen35_moe_export_last_hidden(mlx_array** out);

// W6.7 — One-shot batched MoE verify forward. Runs the entire D+1-token
// verify on a single compiled graph (vs the prior D+1 sequential
// `mlx_qwen35_moe_forward` calls) and emits `[1, D+1, vocab]` logits +
// `[1, D+1, hidden]` hiddens.
extern "C" void mlx_qwen35_moe_forward_batched_verify(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_logits,
    mlx_array** out_hiddens);

// W6.7 follow-up — eagerly compile the MoE batched verify graph for
// depths {1..5}. Defined in `mlx_qwen35_moe.cpp` where it has direct
// access to `g_moe_caches` / `g_moe_offset_int` for snapshot/restore.
// Best-effort: failures are logged + swallowed.
extern "C" void mlx_qwen35_moe_prewarm_verify_compiled();

namespace {

// =====================================================================
// MoE-specific config (mirrors `MoeConfig` in `mlx_qwen35_moe.cpp`).
//
// Duplicated here on purpose — the MoE main file's `MoeConfig` is in an
// anonymous namespace and not visible across translation units. Per the
// W5 MoE plan we accept this surgical duplication over refactoring the
// 1600-line main MoE file.
// =====================================================================
struct MoeConfigMtp : BaseConfig {
  int num_experts;
  int num_experts_per_tok;
  bool norm_topk_prob;
  int decoder_sparse_step;
  int n_mtp_layers;
  // full_attention_interval - 1 (clamped at 0). Now write-only — the two
  // former readers (`is_moe_layer_mtp(fa_idx, cfg)`) were replaced by
  // `mtp_layer_is_moe` below. Kept for forward compatibility / introspection,
  // mirroring the dense `mlx_qwen35_mtp_compiled.cpp` sibling field.
  int mtp_fa_layer_idx;
  bool mtp_layer_is_moe;  // precomputed by the Rust loader's is_moe_layer(fa_idx) — honors mlp_only_layers + the sparse-step modulo, which the FFI-side cannot see.
};

// Per-(MTP-layer) quant info — mirrors the structs in `mlx_qwen35_moe.cpp`
// but populated from the `mtp.layers.{j}.*` weights, not `layers.{i}.*`.
struct LayerQuantInfoMtp {
  bool sw_quant;  int sw_gs;  int sw_bits;  std::string sw_mode;
  bool sh_quant;  int sh_gs;  int sh_bits;  std::string sh_mode;
  bool g_quant;   int g_gs;   int g_bits;   std::string g_mode;
  bool sg_quant;  int sg_gs;  int sg_bits;  std::string sg_mode;
};

struct DenseMLPQuantInfoMtp {
  bool quant;  int gs;  int bits;  std::string mode;
};

// =====================================================================
// MTP-specific compiled state.
// =====================================================================
static MoeConfigMtp g_mtp_config{};
static std::vector<array> g_mtp_compiled_caches;  // 2 * n_mtp_layers (K,V)
static int g_mtp_offset_int = 0;
// W6.32 — start-of-cycle offset (= main offset captured by `begin_cycle`).
// MoE twin of `g_mtp_chain_start_int` in the dense MTP file; see that file
// for the rationale (mask out zero-K/V slots in `[0..chain_start)`).
static int g_mtp_chain_start_int = 0;
static bool g_mtp_compile_inited = false;

// Per-MTP-layer quant info, populated at init.
static std::vector<LayerQuantInfoMtp> g_mtp_layer_quant;
static std::vector<DenseMLPQuantInfoMtp> g_mtp_dense_quant;

// MTP-private 3D-transpose cache for expert weights.
//
// `gather_mm` / `gather_qmm` want `[E, in, out]` but checkpoints store
// `[E, out, in]`. The main MoE init pre-computes a static map for the
// `layers.{i}.*` keys; we maintain an independent map keyed by the
// `mtp.layers.{j}.*` weight names. Populated at init from every 3D
// weight that exists under the `mtp.` prefix.
static std::unordered_map<std::string, array> g_mtp_3d_transposes;

inline array get_weight_t3d_mtp(const std::string& name) {
  auto it = g_mtp_3d_transposes.find(name);
  if (it != g_mtp_3d_transposes.end()) {
    return it->second;  // copy bumps refcount
  }
  throw std::runtime_error("MTP 3D transpose not found for weight: " + name);
}

// =====================================================================
// Linear / switch-linear forwards — parameterised by `pfx` so the
// MoE-branch logic can be reused across `mtp.layers.{j}.mlp.*` prefixes.
//
// Copies of `linear_forward` / `quantized_linear_forward` /
// `switch_linear_forward` / `switch_linear_fwd` from `mlx_qwen35_moe.cpp`
// (kept inside this anon namespace; ODR-safe). The signatures match the
// originals — only the file-static `get_weight_t3d` is replaced with our
// `get_weight_t3d_mtp`.
// =====================================================================

array quantized_linear_forward_mtp(
    const array& x,
    const std::string& prefix,
    int gs, int bits,
    const std::string& mode) {
  auto w = get_weight(prefix + ".weight");
  auto scales = get_weight(prefix + ".scales");
  std::optional<array> biases = std::nullopt;
  if (has_weight(prefix + ".biases")) {
    biases = get_weight(prefix + ".biases");
  }
  return mlx::core::quantized_matmul(
      x, w, scales, biases,
      true,
      std::optional<int>(gs),
      std::optional<int>(bits),
      mode);
}

array linear_forward_mtp(
    const array& x,
    const std::string& prefix,
    bool is_quant, int gs, int bits, const std::string& mode) {
  if (is_quant && has_weight(prefix + ".scales")) {
    return quantized_linear_forward_mtp(x, prefix, gs, bits, mode);
  }
  return matmul(x, get_weight_t(prefix + ".weight"));
}

array switch_linear_forward_mtp(
    const array& x,
    const std::string& key,
    const array& indices,
    bool sorted) {
  return mlx::core::gather_mm(
      x,
      get_weight_t3d_mtp(key + ".weight"),
      std::nullopt,
      indices,
      sorted);
}

array switch_linear_fwd_mtp(
    const array& x,
    const std::string& prefix,
    const array& indices,
    bool sorted,
    bool is_quant, int /*gs_hint*/, int /*bits_hint*/, const std::string& /*mode_hint*/) {
  if (is_quant && has_weight(prefix + ".scales")) {
    auto w = get_weight(prefix + ".weight");
    auto scales = get_weight(prefix + ".scales");
    std::optional<array> biases = std::nullopt;
    if (has_weight(prefix + ".biases")) {
      biases = get_weight(prefix + ".biases");
    }

    int gs;
    int bits;
    std::string mode;
    if (auto info = lookup_quant_info(prefix)) {
      gs = info->group_size;
      bits = info->bits;
      mode = info->mode;
    } else {
      int w_cols = w.shape(-1);
      int s_cols = scales.shape(-1);
      gs = 64;
      int original_cols = s_cols * gs;
      bits = (w_cols * 32) / original_cols;
      mode = biases.has_value() ? "affine" : "mxfp8";
      if (!biases.has_value()) { gs = 32; bits = 8; }
    }

    return mlx::core::gather_qmm(
        x, w, scales, biases,
        std::nullopt,
        indices,
        true,
        std::optional<int>(gs),
        std::optional<int>(bits),
        mode,
        sorted);
  }
  return switch_linear_forward_mtp(x, prefix, indices, sorted);
}

// =====================================================================
// Gather sort / scatter unsort — verbatim copies of the helpers in
// `mlx_qwen35_moe.cpp`. Kept in the anon namespace to avoid ODR clashes.
// =====================================================================
struct GatherSortResultMtp {
  array x_sorted;
  array idx_sorted;
  array inv_order;
};

GatherSortResultMtp gather_sort_mtp(const array& x, const array& indices) {
  auto idx_shape = indices.shape();
  int m = idx_shape.back();

  auto flat_indices = reshape(indices, {-1});
  auto order = argsort(flat_indices, -1);
  auto inv_order = argsort(order, -1);
  auto idx_sorted = take(flat_indices, order, 0);

  auto x_shape = x.shape();
  int d = x_shape.back();
  auto x_flat = reshape(x, {-1, 1, d});
  auto m_arr = array(m, mlx::core::int32);
  auto token_indices = mlx::core::floor_divide(order, m_arr);
  auto x_sorted = take(x_flat, token_indices, 0);

  return {x_sorted, idx_sorted, inv_order};
}

array scatter_unsort_mtp(const array& x, const array& inv_order, const Shape& orig_shape) {
  auto unsorted = take(x, inv_order, 0);
  auto x_shape = unsorted.shape();
  Shape new_shape(orig_shape.begin(), orig_shape.end());
  for (size_t i = 1; i < x_shape.size(); i++) {
    new_shape.push_back(x_shape[i]);
  }
  return reshape(unsorted, new_shape);
}

// =====================================================================
// Sparse MoE block — parameterised by `layer_prefix` (e.g. "mtp.layers.0")
// so the same helper covers any `<prefix>.mlp.*` weight namespace.
//
// Mirrors `sparse_moe_fn` in `mlx_qwen35_moe.cpp` but reads weights from
// `<layer_prefix>.mlp.*` instead of `layers.{i}.mlp.*`.
// =====================================================================
array sparse_moe_fn_pfx(
    const array& x,                       // [B, hidden]
    const std::string& layer_prefix,
    const MoeConfigMtp& cfg,
    const LayerQuantInfoMtp& qi) {
  int B = x.shape(0);
  int hidden = cfg.hidden_size;
  int ne = B;
  int k = cfg.num_experts_per_tok;
  int num_exp = cfg.num_experts;

  std::string pfx = layer_prefix + ".mlp.";

  auto x_flat = x;

  auto router_logits = linear_forward_mtp(x_flat, pfx + "gate",
      qi.g_quant, qi.g_gs, qi.g_bits, qi.g_mode);
  auto routing_weights = mlx::core::softmax(router_logits, {-1}, /*precise=*/true);

  auto top_indices_full = argpartition(routing_weights, -k, -1);
  auto top_indices = slice(top_indices_full, {0, num_exp - k}, {ne, num_exp});
  auto top_weights = mlx::core::take_along_axis(routing_weights, top_indices, -1);

  if (cfg.norm_topk_prob) {
    auto wsum = sum(top_weights, {-1}, true);
    top_weights = top_weights / wsum;
  }

  auto x_expanded = reshape(x_flat, {ne, 1, 1, hidden});

  std::string sw_pfx = pfx + "switch_mlp.";
  bool do_sort = (top_indices.size() >= 64);

  array expert_out = zeros({}, mlx::core::bfloat16);
  if (do_sort) {
    auto sorted = gather_sort_mtp(x_expanded, top_indices);
    const auto& idx = sorted.idx_sorted;

    auto gate_out = switch_linear_fwd_mtp(sorted.x_sorted, sw_pfx + "gate_proj", idx, true,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    auto up_out = switch_linear_fwd_mtp(sorted.x_sorted, sw_pfx + "up_proj", idx, true,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    auto activated = swiglu(gate_out, up_out);
    auto result = switch_linear_fwd_mtp(activated, sw_pfx + "down_proj", idx, true,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    expert_out = scatter_unsort_mtp(result, sorted.inv_order, top_indices.shape());
  } else {
    auto gate_out = switch_linear_fwd_mtp(x_expanded, sw_pfx + "gate_proj", top_indices, false,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    auto up_out = switch_linear_fwd_mtp(x_expanded, sw_pfx + "up_proj", top_indices, false,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
    auto activated = swiglu(gate_out, up_out);
    expert_out = switch_linear_fwd_mtp(activated, sw_pfx + "down_proj", top_indices, false,
        qi.sw_quant, qi.sw_gs, qi.sw_bits, qi.sw_mode);
  }

  expert_out = squeeze(expert_out, {-2});

  auto weights_expanded = reshape(top_weights, {ne, k, 1});
  auto weighted = expert_out * weights_expanded;
  auto expert_output = sum(weighted, {1});

  // Shared expert — uses `linear_proj` (auto-detects bits per tensor)
  // to mirror the main MoE path's handling of mixed-bit recipes.
  std::string se_pfx = pfx + "shared_expert.";
  auto se_gate_in = linear_proj(x_flat, se_pfx + "gate_proj");
  auto se_up_in = linear_proj(x_flat, se_pfx + "up_proj");
  auto se_activated = swiglu(se_gate_in, se_up_in);
  auto shared_out = linear_proj(se_activated, se_pfx + "down_proj");

  auto shared_gate = linear_forward_mtp(x_flat, pfx + "shared_expert_gate",
      qi.sg_quant, qi.sg_gs, qi.sg_bits, qi.sg_mode);
  shared_gate = sigmoid(shared_gate);

  auto shared_contribution = shared_out * shared_gate;
  return expert_output + shared_contribution;
}

// =====================================================================
// Dense MLP block — parameterised by `layer_prefix`. Mirrors
// `dense_mlp_fn` in `mlx_qwen35_moe.cpp`.
// =====================================================================
array dense_mlp_fn_pfx(
    const array& x,
    const std::string& layer_prefix,
    const MoeConfigMtp& /*cfg*/,
    const DenseMLPQuantInfoMtp& /*qi*/) {
  std::string mp = layer_prefix + ".mlp.";
  auto gate = linear_proj(x, mp + "gate_proj");
  auto up   = linear_proj(x, mp + "up_proj");
  auto mlp_out = linear_proj(swiglu(gate, up), mp + "down_proj");
  return mlp_out;
}

// =====================================================================
// MTP MLP flavor (MoE vs dense) is NOT recomputed here. The Rust eager
// loader is the source of truth: it derives the MTP layer's flavor from
// `is_moe_layer(fa_idx)`, which honors `mlp_only_layers` + the sparse-step
// modulo — neither of which crosses the FFI boundary. That precomputed
// bool is passed through `mlx_qwen35_moe_mtp_compiled_init_from_main` and
// stored as `MoeConfigMtp.mtp_layer_is_moe`, so the compiled dispatch and
// the eager path can never disagree on the `mtp.layers.{j}.mlp.*` key
// schema (dense `mlp.{gate,up,down}_proj` vs MoE `switch_mlp.* + gate`).
// =====================================================================
// Per-MTP-layer quant detection.
//
// Mirrors the per-main-layer loop in `mlx_qwen35_moe_init_from_prefill`
// but probes `mtp.layers.{j}.mlp.*` keys instead of `layers.{i}.mlp.*`.
// Uses the same registry-first / heuristic fallback path as the main
// init via `detect_layer_quant` / `detect_router_gate_quant` (both
// declared in `mlx_qwen35_common.h`).
//
// For MoE-typed MTP layers we record `g_mtp_layer_quant[j]` with all
// four projection groups (switch, shared, router gate, shared-gate). For
// dense-typed MTP layers we record `g_mtp_dense_quant[j]` for the dense
// MLP only. The non-active branch is left default-constructed.
// =====================================================================
void detect_mtp_layer_quants(const MoeConfigMtp& cfg) {
  g_mtp_layer_quant.clear();
  g_mtp_layer_quant.reserve(cfg.n_mtp_layers);
  g_mtp_dense_quant.clear();
  g_mtp_dense_quant.reserve(cfg.n_mtp_layers);

  bool moe = cfg.mtp_layer_is_moe;

  for (int j = 0; j < cfg.n_mtp_layers; j++) {
    std::string pfx = "mtp.layers." + std::to_string(j) + ".mlp.";
    if (moe) {
      auto [sw_q, sw_gs, sw_bits, sw_mode] = detect_layer_quant(pfx + "switch_mlp.gate_proj");
      auto [sh_q, sh_gs, sh_bits, sh_mode] = detect_layer_quant(pfx + "shared_expert.gate_proj");
      auto [g_q, g_gs, g_bits, g_mode] = detect_router_gate_quant(pfx + "gate");
      auto [sg_q, sg_gs, sg_bits, sg_mode] = detect_router_gate_quant(pfx + "shared_expert_gate");

      g_mtp_layer_quant.push_back(LayerQuantInfoMtp{
          sw_q, sw_gs, sw_bits, sw_mode,
          sh_q, sh_gs, sh_bits, sh_mode,
          g_q, g_gs, g_bits, g_mode,
          sg_q, sg_gs, sg_bits, sg_mode,
      });
      g_mtp_dense_quant.push_back(DenseMLPQuantInfoMtp{false, 0, 0, ""});
    } else {
      g_mtp_layer_quant.push_back(LayerQuantInfoMtp{});
      auto [dq, dgs, dbits, dmode] = detect_layer_quant(pfx + "gate_proj");
      g_mtp_dense_quant.push_back(DenseMLPQuantInfoMtp{dq, dgs, dbits, dmode});
    }
  }
}

// =====================================================================
// Populate `g_mtp_3d_transposes` for every 3D `mtp.*` weight currently
// registered in `g_weights()`. Called at init time. The set is small
// (only switch_mlp.* projections live in 3D) so a single pass is cheap.
// =====================================================================
void prepopulate_mtp_3d_transposes() {
  g_mtp_3d_transposes.clear();
  std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
  for (const auto& [name, w] : g_weights()) {
    if (w.ndim() != 3) continue;
    // Only pre-transpose MTP-namespaced weights. Main-path 3D weights
    // are already in the main file's separate map.
    if (name.rfind("mtp.", 0) != 0) continue;
    g_mtp_3d_transposes.insert_or_assign(name, transpose(w, {0, 2, 1}));
  }
}

// =====================================================================
// Draft graph (traced once, reused across all D draft steps).
//
// Inputs:
//   [0]                prev_hidden     [1, 1, hidden]  bf16
//   [1]                prev_emb        [1, 1, hidden]  bf16
//   [2]                offset_arr      [1]             int32 (RoPE +
//                                                     slice_update start)
//   [3]                chain_start_arr [1]             int32 (lower
//                                                     attn-mask bound;
//                                                     see dense MTP file
//                                                     for rationale)
//   For each MTP layer j in [0, n_mtp_layers):
//     [4 + j*2 + 0]    K cache         [1, Hkv, max_kv_len, head_dim]
//     [4 + j*2 + 1]    V cache         [1, Hkv, max_kv_len, head_dim]
//
// Outputs:
//   [0]                h_next       [1, 1, hidden]
//   [1]                draft_logits [1, vocab]
//   For each MTP layer j:
//     [2 + j*2 + 0]    new K cache
//     [2 + j*2 + 1]    new V cache
// =====================================================================
static std::vector<array> moe_mtp_draft_decode_fn(const std::vector<array>& inputs) {
  const auto& cfg = g_mtp_config;
  auto prev_hidden     = inputs[0];
  auto prev_emb        = inputs[1];
  auto offset_arr      = inputs[2];
  auto chain_start_arr = inputs[3];

  auto h_norm = fast::rms_norm(prev_hidden,
                               get_weight("mtp.pre_fc_norm_hidden.weight"),
                               cfg.rms_norm_eps);
  auto e_norm = fast::rms_norm(prev_emb,
                               get_weight("mtp.pre_fc_norm_embedding.weight"),
                               cfg.rms_norm_eps);

  // Concat order `[embedding, hidden]` — MTPLX `concat_order` default
  // `"embedding_hidden"`; the bias-free `mtp.fc` columns expect that
  // block layout.
  auto concat3d = concatenate({e_norm, h_norm}, 2);
  auto concat2d = reshape(concat3d, {1, cfg.hidden_size * 2});
  auto h2d = linear_proj(concat2d, "mtp.fc");

  int max_kv_len = inputs[4].shape(2);
  auto positions = arange(0, max_kv_len, mlx::core::int32);
  // Mask: 0 (allow) iff `chain_start <= pos <= offset_arr`. See dense
  // MTP file for the full rationale on the lower bound.
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

  const bool moe = cfg.mtp_layer_is_moe;

  for (int j = 0; j < cfg.n_mtp_layers; j++) {
    std::string lp = "mtp.layers." + std::to_string(j);

    auto normed = fast::rms_norm(h2d, get_weight(lp + ".input_layernorm.weight"),
                                 cfg.rms_norm_eps);

    const auto& kk = inputs[4 + j * 2];
    const auto& kv = inputs[4 + j * 2 + 1];
    auto res = attn_pure_fn_arr_offset(normed, lp,
                                       kk, kv, attn_mask, offset_arr, cfg);
    h2d = h2d + res.output;
    new_caches[j * 2]     = std::move(res.keys);
    new_caches[j * 2 + 1] = std::move(res.values);

    auto mlp_in = fast::rms_norm(h2d, get_weight(lp + ".post_attention_layernorm.weight"),
                                 cfg.rms_norm_eps);

    if (moe) {
      h2d = h2d + sparse_moe_fn_pfx(mlp_in, lp, cfg, g_mtp_layer_quant[j]);
    } else {
      h2d = h2d + dense_mlp_fn_pfx(mlp_in, lp, cfg, g_mtp_dense_quant[j]);
    }
  }

  auto h_norm_final = fast::rms_norm(h2d, get_weight("mtp.norm.weight"),
                                     cfg.rms_norm_eps);
  auto logits = cfg.tie_word_embeddings
      ? linear_proj(h_norm_final, "embedding")
      : linear_proj(h_norm_final, "lm_head");

  auto h_next = reshape(h_norm_final, {1, 1, cfg.hidden_size});

  std::vector<array> result;
  result.reserve(2 + cfg.n_mtp_layers * 2);
  result.push_back(std::move(h_next));
  result.push_back(std::move(logits));
  for (auto& c : new_caches) result.push_back(std::move(c));
  return result;
}

// MoE MTP draft graph. The MoE main path takes `offset_arr` as an array input,
// same as us, so the compile cache is stable across decode steps. The dense MTP
// file makes the same choice (see `compiled_mtp_draft_decode` there).
//
// `moe_mtp_draft_decode_fn` reads the MoE MTP head weights via `get_weight(...)`
// inside the trace, so the weights are baked into the cached tape. Held in a
// resettable FILE-SCOPE global (NOT a function-local static, which cannot be
// nulled) and compiled through `compile_resettable_weight_graph` so a model
// reload can null it via `mlx_qwen35_moe_mtp_invalidate_compiled_graphs()` and
// force a re-trace against the live registry. Deliberately survives the
// per-turn `mlx_qwen35_moe_mtp_compiled_reset()` (cross-turn reuse) — nulled
// ONLY on reload.
static std::function<std::vector<array>(const std::vector<array>&)>
    g_moe_mtp_draft_decode_compiled{};

static std::function<std::vector<array>(const std::vector<array>&)>&
compiled_moe_mtp_draft_decode() {
  if (!g_moe_mtp_draft_decode_compiled) {
    g_moe_mtp_draft_decode_compiled =
        compile_resettable_weight_graph(moe_mtp_draft_decode_fn);
  }
  return g_moe_mtp_draft_decode_compiled;
}

// =====================================================================
// Verify graphs — per-depth dispatcher (mirrors dense MTP file).
// =====================================================================
constexpr int MAX_VERIFY_DEPTH = 5;
using VerifyFn = std::function<std::vector<array>(
    const array&, const array&)>;
static std::array<VerifyFn, MAX_VERIFY_DEPTH> g_verify_compiled_by_depth{};

// W6.7 follow-up #3 — Reset-aware once-flag for the MoE prewarm path.
// Mirrors the dense MTP file (`mlx_qwen35_mtp_compiled.cpp`); see the
// comment there for the full rationale. TL;DR: `std::once_flag` cannot be
// reset, so a `mlx_qwen35_moe_mtp_compiled_reset` + re-init would leave
// `g_verify_compiled_by_depth` null and force lazy-per-depth construction
// on every first verify. An atomic-bool gate lets the reset hook re-arm
// the prewarm.
static std::atomic<bool> g_prewarm_done{false};

// W6.7 — Dispatch to the new batched MoE verify FFI (one compiled
// forward over T = depth+1 tokens) and surface
// `{logits[1, T, vocab], hiddens[1, T, hidden]}`. See the dense MTP
// `make_verify_fn` for the design rationale.
static VerifyFn make_verify_fn(int depth) {
  return [depth](const array& input_ids, const array& embedding_weight)
             -> std::vector<array> {
    int seq_len = input_ids.shape(1);
    if (seq_len != depth + 1) {
      throw std::runtime_error(
          "mlx_qwen35_moe_mtp_verify: input_ids time dim (" +
          std::to_string(seq_len) + ") must equal depth+1 (" +
          std::to_string(depth + 1) + ")");
    }

    array tok_copy = input_ids;
    array emb_copy = embedding_weight;
    mlx_array* logits_ptr = nullptr;
    mlx_array* hidden_ptr = nullptr;
    mlx_qwen35_moe_forward_batched_verify(
        reinterpret_cast<mlx_array*>(&tok_copy),
        reinterpret_cast<mlx_array*>(&emb_copy),
        depth,
        &logits_ptr,
        &hidden_ptr);
    if (!logits_ptr || !hidden_ptr) {
      if (logits_ptr) {
        delete reinterpret_cast<array*>(logits_ptr);
      }
      if (hidden_ptr) {
        delete reinterpret_cast<array*>(hidden_ptr);
      }
      throw std::runtime_error(
          "mlx_qwen35_moe_mtp_verify: batched MoE verify forward returned null");
    }
    array logits = *reinterpret_cast<array*>(logits_ptr);
    delete reinterpret_cast<array*>(logits_ptr);
    array hiddens = *reinterpret_cast<array*>(hidden_ptr);
    delete reinterpret_cast<array*>(hidden_ptr);
    return {logits, hiddens};
  };
}

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
// Initialize MoE MTP compiled state.
//
// MUST be called once per turn AFTER `mlx_qwen35_moe_init_from_prefill`. The
// MTP path allocates its own per-MTP-layer KV caches (zeros) sized to
// `max_kv_len`, snapshots config, detects MTP-weight quant info, and
// pre-computes 3D transposes for `mtp.layers.{j}.mlp.switch_mlp.*` weights.
//
// Returns 0 on success, -1 on failure. On failure the MTP state is left
// uninitialised (`g_mtp_compile_inited = false`) so subsequent draft /
// verify calls become null-pointer no-ops — the Rust caller can then
// fall back to the eager Rust MTP path.
// -----------------------------------------------------------------------------
int32_t mlx_qwen35_moe_mtp_compiled_init_from_main(
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
    int n_mtp_layers,
    int num_experts,
    int num_experts_per_tok,
    int norm_topk_prob,
    int decoder_sparse_step,
    int mtp_layer_is_moe
) {
  try {
    if (n_mtp_layers <= 0) {
      std::cerr << "[MLX] moe_mtp_compiled_init: n_mtp_layers must be > 0 (got "
                << n_mtp_layers << ")" << std::endl;
      g_mtp_compile_inited = false;
      return -1;
    }
    if (!mlx_qwen35_moe_is_compile_inited()) {
      std::cerr << "[MLX] moe_mtp_compiled_init: main MoE compiled path is not "
                   "initialised — call mlx_qwen35_moe_init_from_prefill "
                   "before mlx_qwen35_moe_mtp_compiled_init_from_main"
                << std::endl;
      g_mtp_compile_inited = false;
      return -1;
    }
    if (!has_weight("mtp.norm.weight")) {
      std::cerr << "[MLX] moe_mtp_compiled_init: mtp.norm.weight not "
                   "registered — load MTP weights first" << std::endl;
      g_mtp_compile_inited = false;
      return -1;
    }

    g_mtp_config = MoeConfigMtp{};
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
    g_mtp_config.num_experts             = num_experts;
    g_mtp_config.num_experts_per_tok     = num_experts_per_tok;
    g_mtp_config.norm_topk_prob          = (norm_topk_prob != 0);
    g_mtp_config.decoder_sparse_step     = decoder_sparse_step;
    g_mtp_config.n_mtp_layers            = n_mtp_layers;
    g_mtp_config.mtp_fa_layer_idx        = std::max(full_attention_interval - 1, 0);
    g_mtp_config.mtp_layer_is_moe        = (mtp_layer_is_moe != 0);

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

    // Mirror the main MoE path's current offset. `g_mtp_chain_start_int`
    // gets re-snapshotted by every `begin_cycle`; this seed is only for
    // any debug introspection between init and the first cycle.
    g_mtp_offset_int = mlx_qwen35_moe_get_cache_offset();
    g_mtp_chain_start_int = g_mtp_offset_int;

    detect_mtp_layer_quants(g_mtp_config);
    prepopulate_mtp_3d_transposes();

    for (auto& slot : g_verify_compiled_by_depth) {
      slot = nullptr;
    }

    g_mtp_compile_inited = true;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_qwen35_moe_mtp_compiled_init_from_main: "
              << e.what() << std::endl;
    g_mtp_compile_inited = false;
    return -1;
  } catch (...) {
    std::cerr << "[MLX] mlx_qwen35_moe_mtp_compiled_init_from_main: "
                 "unknown exception" << std::endl;
    g_mtp_compile_inited = false;
    return -1;
  }
}

// -----------------------------------------------------------------------------
// One MoE MTP draft step. See dense FFI doc for argument/output semantics
// — the contract is identical. Mutates `g_mtp_compiled_caches` and
// advances `g_mtp_offset_int` by 1.
// -----------------------------------------------------------------------------
void mlx_qwen35_moe_mtp_draft_compiled(
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
            "[MTP-TRACE] mlx_qwen35_moe_mtp_draft_compiled: ENTER (per-step) "
            "mtp_offset=%d (RoPE base) chain_start=%d "
            "fc_concat_order=[embedding,hidden]\n",
            g_mtp_offset_int, g_mtp_chain_start_int);
  }

  try {
    auto& prev_hidden = *reinterpret_cast<array*>(prev_hidden_ptr);
    auto& prev_emb    = *reinterpret_cast<array*>(prev_emb_ptr);

    std::vector<array> inputs;
    inputs.reserve(4 + g_mtp_config.n_mtp_layers * 2);
    inputs.push_back(prev_hidden);
    inputs.push_back(prev_emb);
    inputs.push_back(reshape(array(g_mtp_offset_int, mlx::core::int32), {1}));
    inputs.push_back(reshape(array(g_mtp_chain_start_int, mlx::core::int32), {1}));
    for (const auto& c : g_mtp_compiled_caches) {
      inputs.push_back(c);
    }

    auto outputs = compiled_moe_mtp_draft_decode()(inputs);

    *out_h_next = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    *out_logits = reinterpret_cast<mlx_array*>(new array(outputs[1]));
    g_mtp_offset_int++;
    for (int j = 0; j < g_mtp_config.n_mtp_layers * 2; j++) {
      g_mtp_compiled_caches[j] = outputs[2 + j];
    }
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_moe_mtp_draft_compiled: EXIT OK "
              "new_mtp_offset=%d\n",
              g_mtp_offset_int);
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_mtp_draft_compiled: %s\n",
            e.what());
    fflush(stderr);
    if (out_h_next) *out_h_next = nullptr;
    if (out_logits) *out_logits = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_moe_mtp_draft_compiled\n");
    fflush(stderr);
    if (out_h_next) *out_h_next = nullptr;
    if (out_logits) *out_logits = nullptr;
  }
}

// -----------------------------------------------------------------------------
// One MoE MTP verify step.
//
// SIDE EFFECTS: advances the MAIN MoE compiled offset (`g_moe_offset_int`
// in `mlx_qwen35_moe.cpp`) by `depth + 1` and updates `g_moe_caches[]`
// in place. The caller MUST hold `MOE_COMPILED_MUTEX` (NOT
// `DENSE_COMPILED_MUTEX`) for the entire draft+verify cycle.
// -----------------------------------------------------------------------------
void mlx_qwen35_moe_mtp_verify_compiled(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_logits
) {
  if (out_logits) *out_logits = nullptr;
  if (!input_ids_ptr || !embedding_weight_ptr || !out_logits) return;
  if (depth < 1 || depth > MAX_VERIFY_DEPTH) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_moe_mtp_verify_compiled: depth %d outside [1, %d]\n",
            depth, MAX_VERIFY_DEPTH);
    fflush(stderr);
    return;
  }

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_moe_mtp_verify_compiled: ENTER depth=%d\n",
            depth);
  }

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != depth + 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_moe_mtp_verify_compiled: input_ids shape must be "
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
              "[MTP-TRACE] mlx_qwen35_moe_mtp_verify_compiled: EXIT OK "
              "depth=%d\n",
              depth);
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_moe_mtp_verify_compiled: %s\n",
            e.what());
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in mlx_qwen35_moe_mtp_verify_compiled\n");
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
  }
}

// -----------------------------------------------------------------------------
// W6.5 — MoE verify pass that ALSO exports the post-final-norm hidden
// at EVERY verify position so the caller can chain MTP cycles without
// running a fresh main-model forward at each cycle's "Step A". MoE
// twin of `mlx_qwen35_mtp_verify_compiled_with_hidden` — see that
// function's docstring for the rationale, lifetime contract, and
// failure mode.
//
// The MoE main forward (`mlx_qwen35_moe_forward`) writes
// `g_moe_last_hidden` on every call (graph output slot 2). The verify
// closure captures it after every iteration before the next forward
// overwrites it, then concatenates into `[1, depth+1, hidden_size]`.
// -----------------------------------------------------------------------------
void mlx_qwen35_moe_mtp_verify_compiled_with_hidden(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    int depth,
    mlx_array** out_logits,
    mlx_array** out_hiddens
) {
  if (out_logits) *out_logits = nullptr;
  if (out_hiddens) *out_hiddens = nullptr;
  if (!input_ids_ptr || !embedding_weight_ptr || !out_logits ||
      !out_hiddens) {
    return;
  }
  if (depth < 1 || depth > MAX_VERIFY_DEPTH) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_moe_mtp_verify_compiled_with_hidden: depth %d "
            "outside [1, %d]\n",
            depth, MAX_VERIFY_DEPTH);
    fflush(stderr);
    return;
  }

  if (qwen35_common::mtp_trace_enabled()) {
    fprintf(stderr,
            "[MTP-TRACE] mlx_qwen35_moe_mtp_verify_compiled_with_hidden: "
            "ENTER depth=%d\n",
            depth);
  }

  try {
    auto& input_ids        = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    if (input_ids.ndim() != 2 || input_ids.shape(0) != 1 ||
        input_ids.shape(1) != depth + 1) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_moe_mtp_verify_compiled_with_hidden: "
              "input_ids shape must be [1, depth+1=%d], got ndim=%d "
              "shape=[%lld,%lld]\n",
              depth + 1, input_ids.ndim(),
              input_ids.ndim() >= 1 ? (long long)input_ids.shape(0) : -1LL,
              input_ids.ndim() >= 2 ? (long long)input_ids.shape(1) : -1LL);
      fflush(stderr);
      return;
    }

    // Run the existing per-depth verify loop. After this returns,
    // `g_moe_caches[]` / `g_moe_offset_int` have been advanced by D+1.
    // The closure returns `{logits[1, D+1, vocab], hiddens[1, D+1,
    // hidden]}` — the hiddens were captured per-iteration before the
    // next MoE forward overwrote `g_moe_last_hidden`.
    const auto& verify_fn = get_or_make_verify_fn(depth);
    auto outputs = verify_fn(input_ids, embedding_weight);
    if (outputs.size() < 2) {
      fprintf(stderr,
              "[MLX] mlx_qwen35_moe_mtp_verify_compiled_with_hidden: "
              "verify closure returned %zu outputs; expected 2 (logits, "
              "hiddens)\n", outputs.size());
      fflush(stderr);
      return;
    }
    *out_logits  = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    *out_hiddens = reinterpret_cast<mlx_array*>(new array(outputs[1]));
    if (qwen35_common::mtp_trace_enabled()) {
      fprintf(stderr,
              "[MTP-TRACE] mlx_qwen35_moe_mtp_verify_compiled_with_hidden: "
              "EXIT OK depth=%d\n",
              depth);
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] Exception in "
            "mlx_qwen35_moe_mtp_verify_compiled_with_hidden: %s\n",
            e.what());
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
  } catch (...) {
    fprintf(stderr,
            "[MLX] Unknown exception in "
            "mlx_qwen35_moe_mtp_verify_compiled_with_hidden\n");
    fflush(stderr);
    if (out_logits) *out_logits = nullptr;
    if (out_hiddens) *out_hiddens = nullptr;
  }
}

// -----------------------------------------------------------------------------
// W6.7 follow-up — Pre-warm the per-depth verify dispatch closures AND the
// underlying MLX-compiled batched verify graph for the MoE main path.
//
// Wire this to fire IMMEDIATELY after
// `mlx_qwen35_moe_mtp_compiled_init_from_main` returns 0 from the Rust
// caller. MoE has only ONE compiled variant (W6.6 tape-replay is deferred
// for MoE) so this prewarms 5 shapes total instead of 10.
//
// W6.7 follow-up #2 — `init_moe_mtp_compiled_from_main` runs once PER
// TURN, not once per process. Gate the body behind a once-flag so the
// heavy work fires exactly once per process — on the FIRST turn (after
// MoE MTP init has set up the global state). Subsequent turns
// short-circuit immediately.
//
// W6.7 follow-up #3 — The flag is `std::atomic<bool>` rather than
// `std::once_flag` so `mlx_qwen35_moe_mtp_compiled_reset` can re-arm it.
// See the dense MTP file for the full rationale.
//
// Best-effort: any failure is logged + swallowed, leaving the verify
// path to fall back to lazy-at-first-use.
// -----------------------------------------------------------------------------
void mlx_qwen35_moe_mtp_compiled_prewarm_verify() {
  if (!g_mtp_compile_inited) {
    return;
  }
  // W6.7 follow-up #3 — Atomic-bool gate (reset-aware) replaces
  // `std::once_flag`. See the dense MTP file for the rationale.
  bool expected = false;
  if (!g_prewarm_done.compare_exchange_strong(expected, true)) {
    return;
  }
  try {
    for (int d = 1; d <= MAX_VERIFY_DEPTH; d++) {
      auto& slot = g_verify_compiled_by_depth[d - 1];
      if (!slot) {
        slot = make_verify_fn(d);
      }
    }
  } catch (const std::exception& e) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_moe_mtp_compiled_prewarm_verify: closure "
            "population failed: %s\n", e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr,
            "[MLX] mlx_qwen35_moe_mtp_compiled_prewarm_verify: unknown "
            "exception during closure population\n");
    fflush(stderr);
  }
  mlx_qwen35_moe_prewarm_verify_compiled();
}

// -----------------------------------------------------------------------------
// Tear down MoE MTP compiled state. Idempotent. Does NOT touch the main
// MoE path's globals — call `mlx_qwen35_moe_reset` separately for that.
// -----------------------------------------------------------------------------
void mlx_qwen35_moe_mtp_compiled_reset() {
  g_mtp_compiled_caches.clear();
  g_mtp_offset_int = 0;
  g_mtp_chain_start_int = 0;
  g_mtp_compile_inited = false;
  g_mtp_config = MoeConfigMtp{};
  g_mtp_layer_quant.clear();
  g_mtp_dense_quant.clear();
  g_mtp_3d_transposes.clear();
  for (auto& slot : g_verify_compiled_by_depth) {
    slot = nullptr;
  }
  // W6.7 follow-up #3 — Re-arm the prewarm gate so a subsequent re-init
  // can repopulate `g_verify_compiled_by_depth` via the prewarm path.
  g_prewarm_done.store(false);
}

// -----------------------------------------------------------------------------
// Adjust the MoE MTP offset by `delta` (e.g. to rewind after a
// verify-reject rolled back the main MoE path). Mirrors
// `mlx_qwen35_mtp_compiled_adjust_offset` on the dense side.
// -----------------------------------------------------------------------------
void mlx_qwen35_moe_mtp_compiled_adjust_offset(int delta) {
  g_mtp_offset_int += delta;
}

// -----------------------------------------------------------------------------
// W6 Bug #2 fix (Option Reset): begin a fresh MoE MTP draft cycle aligned
// to the main MoE path's current offset. Zeroes the MTP K/V caches and
// sets `g_mtp_offset_int = main_offset`. See the dense MTP file for the
// full rationale — the divergence and fix are identical here.
// -----------------------------------------------------------------------------
void mlx_qwen35_moe_mtp_compiled_begin_cycle(int main_offset) {
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
  // W6.32 — see dense MTP file for full rationale on the chain-start
  // lower-bound in the attn_mask.
  g_mtp_chain_start_int = main_offset;
}

// -----------------------------------------------------------------------------
// Read accessor for the current MoE MTP offset (debugging / introspection
// from Rust unit tests).
// -----------------------------------------------------------------------------
int mlx_qwen35_moe_mtp_get_offset() {
  return g_mtp_offset_int;
}

// PR #65 (mtp-reload P1 follow-up) — invalidate the compiled MoE MTP draft
// graph so the next call re-traces against the CURRENT weight registry.
//
// `moe_mtp_draft_decode_fn` reads the MoE MTP head weights via `get_weight(...)`
// INSIDE the traced closure (NOT as compile inputs), so the captured weight
// arrays are baked into the cached tape. The graph is PROCESS-WIDE and
// deliberately survives the per-turn `mlx_qwen35_moe_mtp_compiled_reset()`
// (cross-turn reuse). On a model RELOAD the baked weights are stale, so a
// second same-shape MoE-MTP model loaded in the same process would draft with
// the FIRST model's weights — silent corruption.
//
// Because the graph is compiled through `compile_resettable_weight_graph`, it
// carries a UNIQUE, erasable `fun_id`; assigning an empty `std::function{}`
// destroys the wrapper, ERASES its compile-cache entry, and forces a re-trace
// against the live registry on the next call. Called transitively from
// `mlx_qwen35_moe_invalidate_compiled_graphs()` (the MoE reload entry point),
// so it runs INSIDE the Rust `COMPILED_WEIGHTS_RWLOCK` write critical section.
void mlx_qwen35_moe_mtp_invalidate_compiled_graphs() {
  g_moe_mtp_draft_decode_compiled =
      std::function<std::vector<array>(const std::vector<array>&)>{};
}

}  // extern "C"
