#include "mlx_common.h"
#include "mlx_qwen35_common.h"

#include <atomic>
#include <cstdlib>

// =============================================================================
// Phase 6b: mx::compile-wrapped SwiGLU MLP fast path
//
// When enabled, the matmul → matmul → sigmoid → multiply → multiply → matmul
// chain is routed through `mlx::core::compile`. The compile pass leaves the
// three matmuls as separate non-fusable primitives but collapses the
// sigmoid + 2 multiplies into a single Compiled (fused element-wise) primitive
// — saving 2 kernel launches per MLP forward (28 layers in Qwen3.5-0.8B
// → ~56 dispatches/tok). The compiled tape is keyed by function-identity +
// input shapes, so for Tq=1 decode the trace happens once and is reused on
// every subsequent forward.
//
// Toggle: env var MLX_WGPU_COMPILE_MLP=1 OR runtime setter
// `mlx_set_swiglu_compile_enabled(bool)` (called from Rust/JS during init).
// Default OFF — this is a warm-up phase that de-risks compile for 6c/6d.
//
// Important subtleties:
//   1. Weight transposes stay OUTSIDE the compiled body — they are
//      metadata-only views, and putting them inside would pollute the
//      compile cache key with view-flag noise.
//   2. The body is a top-level non-capturing function (NOT a lambda), so
//      `mlx::core::compile` keys it by stable function identity. A
//      capturing lambda would create a new cache entry per call.
//   3. The runtime flag is a `std::atomic<bool>` so the JS-side setter and
//      env-var initial read can both feed it without a lock.
// =============================================================================

namespace {

// Process-wide kill switch for the compile MLP fast path. Initialized lazily
// from the MLX_WGPU_COMPILE_MLP env var on first read; can also be flipped
// at runtime via mlx_set_swiglu_compile_enabled() from Rust/JS init code
// (mirroring the wgpu_set_packed_bf16_enabled / wgpu_set_sdpa_fallback_forced
// pattern used elsewhere in this TU).
std::atomic<bool> g_compile_mlp_enabled{false};
std::atomic<bool> g_compile_mlp_env_read{false};

bool compile_mlp_enabled() {
  // Read env var exactly once and OR it into the runtime flag. After this,
  // any explicit setter calls override the env value because they store
  // unconditionally.
  if (!g_compile_mlp_env_read.exchange(true, std::memory_order_acq_rel)) {
    if (const char* v = std::getenv("MLX_WGPU_COMPILE_MLP")) {
      if (v[0] == '1') {
        g_compile_mlp_enabled.store(true, std::memory_order_relaxed);
      }
    }
  }
  return g_compile_mlp_enabled.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Phase 6c: mx::compile-wrapped GDN pre-recurrence fast path
//
// Same template as Phase 6b: an atomic enable flag, an env-var reader, a
// captureless free-function body, and a static-local compile accessor. The
// element-wise chains surrounding the non-fusable primitives (matmul, slice,
// reshape, concat, fast::rms_norm) get collapsed into fused Compiled kernels
// by the compile pass — the fused primitives themselves stay as one dispatch
// each. For Qwen3.5-0.8B decode every linear-attention layer (18 of them)
// fires this path once per token, so 1 trace + many reuses.
//
// Env var: MLX_WGPU_COMPILE_GDN_PRE=1
// Setter : mlx_set_gdn_pre_compile_enabled(bool)
//
// The body is only traced for the common steady-state decode shape (mask
// absent, conv_state present, Tq=1). Prefill and any other variant fall
// through to the eager path — this keeps the compile cache to a single
// entry and sidesteps the shape-keyed cache churn the plan warns about.
// ---------------------------------------------------------------------------
std::atomic<bool> g_compile_gdn_pre_enabled{false};
std::atomic<bool> g_compile_gdn_pre_env_read{false};

bool compile_gdn_pre_enabled() {
  if (!g_compile_gdn_pre_env_read.exchange(true, std::memory_order_acq_rel)) {
    if (const char* v = std::getenv("MLX_WGPU_COMPILE_GDN_PRE")) {
      if (v[0] == '1') {
        g_compile_gdn_pre_enabled.store(true, std::memory_order_relaxed);
      }
    }
  }
  return g_compile_gdn_pre_enabled.load(std::memory_order_relaxed);
}

// Captureless free function describing the GDN pre-recurrence tape as MLX
// sees it. Inputs layout MUST match exactly what the caller passes:
//
//   inputs[0] = x                [B, T, hidden]         activations
//   inputs[1] = w_qkvz_t         [hidden, qkvz_last]    pre-transposed
//   inputs[2] = w_ba_t           [hidden, 2*num_v_heads] pre-transposed
//   inputs[3] = w_conv_2d        [conv_dim, K]          pre-squeezed (axis=2)
//   inputs[4] = conv_state       [B, K-1, conv_dim]     non-null (decode)
//   inputs[5] = kh_hint          [num_k_heads, key_head_dim] uint8 zeros
//                                 (shape-only — data is a dummy constant.
//                                  encodes the head config that can't be
//                                  derived from the weight shapes alone.)
//
// Outputs returned in order: { q, k, v, z, a, b, new_conv_state }.
//
// All non-shape-derived integer config (rms_eps, kernel_size) comes from
// shape inspection of the inputs — the body captures NOTHING. This is what
// makes the function pointer identity stable across calls, which is what
// mlx::core::compile keys its cache on.
//
// Mirrors the eager path below op-for-op so the resulting tape is
// bit-identical. No operation reordering, no extra ops, no skipped
// metadata views.
std::vector<mlx::core::array> gdn_prefusion_body(
    const std::vector<mlx::core::array>& inputs) {
  // Inputs layout:
  //   [0] x          (B, T, hidden)
  //   [1] w_qkvz_t   (hidden, key_dim*2 + value_dim*2)  — pre-transposed
  //   [2] w_ba_t     (hidden, num_v_heads * 2)          — pre-transposed
  //   [3] w_conv_2d  (conv_dim, kernel_dim)             — pre-squeezed
  //   [4] conv_state (B, kernel_dim-1, conv_dim)
  //
  // All metadata-only view ops (transpose, squeeze) are done OUTSIDE this
  // body so the compile cache key stays on concrete shape/dtype signatures
  // — Phase 6b lesson.
  //
  // The body returns seven arrays in the same order as the eager path of
  // `mlx_gdn_prefusion_forward`:
  //   { q_flat, k_flat, v_flat, z, a, b, new_conv_state }
  // Per-head reshape + rms_norm + scaling on q/k happens in the caller AFTER
  // `compiled_gdn_prefusion()(ins)` returns, so the body does not need a
  // separate "head config hint" input (shape-derived config is tricky with
  // placeholder tracer arrays, and every extra input is a potential source
  // of bind-group layout mismatch in the Compiled primitive).
  const auto& x = inputs[0];
  const auto& w_qkvz_t = inputs[1];
  const auto& w_ba_t = inputs[2];
  const auto& w_conv_2d = inputs[3];
  const auto& conv_state = inputs[4];

  // Shape-derived config. .shape() is metadata and is safe inside the
  // traced body even on placeholder input arrays (which have nullptr data).
  int batch = static_cast<int>(x.shape()[0]);
  int seq_len = static_cast<int>(x.shape()[1]);
  int num_v_heads_x2 = static_cast<int>(w_ba_t.shape()[1]);
  int num_v_heads = num_v_heads_x2 / 2;
  int conv_dim = static_cast<int>(w_conv_2d.shape()[0]);
  int conv_kernel_dim = static_cast<int>(w_conv_2d.shape()[1]);
  int qkvz_last = static_cast<int>(w_qkvz_t.shape()[1]);
  // Sanity: qkvz_last = 2*key_dim + 2*value_dim. The conv_dim spans the
  // "qkv" portion = 2*key_dim + value_dim, so value_dim = qkvz_last - conv_dim
  // and key_dim = (conv_dim - value_dim) / 2. (Both are recoverable.)

  // 1. Projections — matmul is NOT fusable, so each stays as its own tape
  //    node, but the element-wise chains around them fuse into Compiled
  //    primitives which is exactly the win we're after.
  auto qkvz = matmul(x, w_qkvz_t);
  auto ba = matmul(x, w_ba_t);

  // 2. Split ba → b, a.
  auto b = slice(ba, {0, 0, 0}, {batch, seq_len, num_v_heads});
  auto a = slice(ba, {0, 0, num_v_heads}, {batch, seq_len, num_v_heads * 2});

  // 3. Split qkvz → qkv, z.
  auto qkv = slice(qkvz, {0, 0, 0}, {batch, seq_len, conv_dim});
  auto z = slice(qkvz, {0, 0, conv_dim}, {batch, seq_len, qkvz_last});

  // 4. Prepend conv_state to qkv (decode-only: conv_state is always present).
  auto conv_input = concatenate({conv_state, qkv}, 1);

  // 5. new_conv_state = last (K-1) rows of conv_input along axis 1.
  int total_len = static_cast<int>(conv_input.shape()[1]);
  int keep = conv_kernel_dim - 1;
  auto new_conv_state = slice(
      conv_input, {0, total_len - keep, 0}, {batch, total_len, conv_dim});

  // 6. Depthwise conv1d — full loop restored (op-for-op mirror).
  int conv_in_len = static_cast<int>(conv_input.shape()[1]);
  int t_out = conv_in_len - conv_kernel_dim + 1;
  array conv_out = [&]() {
    auto input_slice0 = slice(
        conv_input, {0, 0, 0}, {batch, t_out, conv_dim});
    auto w_col0 = slice(w_conv_2d, {0, 0}, {conv_dim, 1});
    auto w_k0 = reshape(w_col0, {1, 1, conv_dim});
    return input_slice0 * w_k0;
  }();
  for (int k = 1; k < conv_kernel_dim; ++k) {
    auto input_slice = slice(
        conv_input, {0, k, 0}, {batch, k + t_out, conv_dim});
    auto w_col = slice(w_conv_2d, {0, k}, {conv_dim, k + 1});
    auto w_k = reshape(w_col, {1, 1, conv_dim});
    auto term = input_slice * w_k;
    conv_out = conv_out + term;
  }

  // 7. Trim to seq_len if the depthwise produced extra rows.
  int conv_out_len = static_cast<int>(conv_out.shape()[1]);
  if (conv_out_len > seq_len) {
    conv_out = slice(conv_out,
                     {0, conv_out_len - seq_len, 0},
                     {batch, conv_out_len, conv_dim});
  }

  // 8. SiLU activation — the element-wise win that the compile pass fuses.
  conv_out = conv_out * sigmoid(conv_out);

  // 9. Split channels → q_flat, k_flat, v_flat. Per-head reshape + rms_norm
  //    + scaling happen in the caller AFTER the compile pass returns.
  // value_dim = qkvz_last - conv_dim   (invariant in the eager path)
  int value_dim = qkvz_last - conv_dim;
  int key_dim = (conv_dim - value_dim) / 2;
  auto q_flat = slice(conv_out, {0, 0, 0}, {batch, seq_len, key_dim});
  auto k_flat = slice(conv_out, {0, 0, key_dim}, {batch, seq_len, key_dim * 2});
  auto v_flat = slice(conv_out, {0, 0, key_dim * 2}, {batch, seq_len, conv_dim});

  // Return order matches the caller's out_* unpacking below:
  //   q_flat, k_flat, v_flat, z, a, b, new_conv_state
  return {q_flat, k_flat, v_flat, z, a, b, new_conv_state};
}

auto& compiled_gdn_prefusion() {
  static auto fn = mlx::core::compile(gdn_prefusion_body, /*shapeless=*/false);
  return fn;
}

// ---------------------------------------------------------------------------
// Phase 6d: mx::compile-wrapped GDN post-recurrence fast path
//
// Same template as Phase 6b / 6c: atomic enable flag, env-var reader,
// captureless free-function body, static-local compile accessor. The GDN
// post-recurrence body does four element-wise ops (sigmoid, mul, mul,
// optional add-bias) around one reduction (fast::rms_norm) and one matmul
// (out-proj). The compile pass collapses the three element-wise ops between
// the rms_norm and the matmul into a single fused Compiled kernel, saving
// ~3 dispatches per call. Across Qwen3.5-0.8B's 18 linear-attention layers
// that's ~54 dispatches/tok — smaller than Phase 6c's prefusion win, but
// same mechanism.
//
// Env var: MLX_WGPU_COMPILE_GDN_POST=1
// Setter : mlx_set_gdn_post_compile_enabled(bool)
//
// Trace-signature gating: only the no-bias path (the default Qwen3.5 GDN
// out_proj layout) is routed through the compile pass. Models with an
// out_proj bias fall through to the eager path. This keeps the compile
// cache to a single entry and avoids shape-key churn from the optional
// bias input.
// ---------------------------------------------------------------------------
std::atomic<bool> g_compile_gdn_post_enabled{false};
std::atomic<bool> g_compile_gdn_post_env_read{false};

bool compile_gdn_post_enabled() {
  if (!g_compile_gdn_post_env_read.exchange(true, std::memory_order_acq_rel)) {
    if (const char* v = std::getenv("MLX_WGPU_COMPILE_GDN_POST")) {
      if (v[0] == '1') {
        g_compile_gdn_post_enabled.store(true, std::memory_order_relaxed);
      }
    }
  }
  return g_compile_gdn_post_enabled.load(std::memory_order_relaxed);
}

// Captureless free function describing the GDN post-recurrence tape as MLX
// sees it. Inputs layout MUST match exactly what the caller passes:
//
//   inputs[0] = y_att          [B, T, Hv, Dv]         post-recurrence output
//   inputs[1] = z_4d           [B, T, Hv, Dv]         pre-reshaped gate
//   inputs[2] = norm_weight    [Dv]                   RMSNorm gain
//   inputs[3] = w_out_t        [Hv*Dv, hidden]        pre-transposed out_proj
//
// The z reshape (from [B, T, Hv*Dv] to [B, T, Hv, Dv]) and the out_proj
// weight transpose are done OUTSIDE the compile body — they are
// metadata-only view ops, and inside the body they'd pollute the compile
// cache key with view-flag noise (same lesson as Phase 6b/6c).
//
// Returns: { y_out } where y_out is (B, T, hidden). Bias add is handled
// outside the compile body in the caller (and is rare for Qwen3.5 GDN,
// see the gating in mlx_gdn_postfusion_forward below).
//
// Op order matches the eager path below exactly so the resulting tape is
// bit-identical — same reason as Phase 6c.
std::vector<mlx::core::array> gdn_postfusion_body(
    const std::vector<mlx::core::array>& inputs) {
  using mlx::core::array;
  const auto& y_att = inputs[0];
  const auto& z_4d = inputs[1];
  const auto& norm_weight = inputs[2];
  const auto& w_out_t = inputs[3];

  // Shape-derived config. We need the final flatten target from y_att's
  // 4D shape. .shape() is metadata and safe on placeholder trace arrays.
  int batch_size = static_cast<int>(y_att.shape()[0]);
  int seq_len = static_cast<int>(y_att.shape()[1]);
  int n_v_heads = static_cast<int>(y_att.shape()[2]);
  int v_head_dim = static_cast<int>(y_att.shape()[3]);
  int intermediate_size = n_v_heads * v_head_dim;
  // rms_eps is a scalar — we must use the same value as the eager path so
  // the two paths are byte-identical. Capturing it would break the
  // captureless-function guarantee, so we hardcode the single value that
  // ships in Qwen3.5 GDN (1e-6, see gated_delta_net.rs). If a future model
  // uses a different rms_eps, it falls through to the eager path via a
  // runtime check in the caller below.
  const float RMS_EPS = 1e-6f;

  // 1. RMSNorm over the last axis of y_att. fast::rms_norm is a reduction
  //    and stays as its own tape node — the compile pass cannot fuse it.
  auto normed = mlx::core::fast::rms_norm(
      y_att, std::optional<mlx::core::array>(norm_weight), RMS_EPS, {});

  // 2. swiglu(gate=z_4d, up=normed):
  //      silu_gate = z_4d * sigmoid(z_4d)
  //      gated     = silu_gate * normed
  //    These three element-wise ops collapse into one fused Compiled kernel
  //    under the compile pass — the Phase 6d win.
  auto sig = sigmoid(z_4d);
  auto silu_gate = z_4d * sig;
  auto gated = silu_gate * normed;

  // 3. Flatten heads: [B, T, Hv, Dv] → [B, T, Hv*Dv] (metadata only).
  auto gated_flat = reshape(gated, {batch_size, seq_len, intermediate_size});

  // 4. Out-proj matmul. Matmul stays as its own non-fusable primitive.
  auto y_out = matmul(gated_flat, w_out_t);

  return {y_out};
}

auto& compiled_gdn_postfusion() {
  static auto fn = mlx::core::compile(gdn_postfusion_body, /*shapeless=*/false);
  return fn;
}

// ---------------------------------------------------------------------------
// Phase 6e: mx::compile-wrapped GDN decay-gate (compute_g) fast path
//
// Same template as Phase 6b/6c/6d. The WASM decode path in Rust
// (crates/mlx-core/src/models/qwen3_5/gated_delta.rs, use_kernel=true
// branch) computes the decay gate in log-space as:
//
//   big_a  = exp(a_log)                              // dispatch 1
//   x      = a + dt_bias                              // dispatch 2
//   sp_raw = log(exp(x) + 1)                          // dispatches 3,4,5
//   sp     = where(x > 20, x, sp_raw)                 // dispatches 6,7
//   g_log  = -(big_a * sp)                            // dispatches 8,9
//
// That's ~9 element-wise dispatches per linear-attention layer per token.
// Qwen3.5-0.8B has 18 such layers, so this is the single biggest remaining
// source of non-FFI dispatch churn after Phase 6b/6c/6d. Every op here is
// element-wise so the compile pass fuses the whole chain into (in the best
// case) one Compiled kernel — an 8x-ish dispatch reduction per call.
//
// Env var: MLX_WGPU_COMPILE_GDN_G=1
// Setter : mlx_set_gdn_g_compile_enabled(bool)
//
// The body is concrete-shape traced (`shapeless=false`) — see the
// `compiled_gdn_compute_g()` accessor below for the full rationale. One
// trace per distinct [B, T, Hv] signature: decode Tq=1 is the hot path
// that gets reused across all tokens and layers; prefill Tq=prompt_len
// gets its own cache entry. This matches Phase 6c/6d.
//
// The body returns g_log (pre-outer-exp), matching the semantics of the
// existing WASM Rust inline path so the caller's `g = g_log.exp()` stays
// unchanged and the downstream gated_delta_ops_inner sees bit-for-bit the
// same tape as before (minus the fused element-wise chain).
// ---------------------------------------------------------------------------
std::atomic<bool> g_compile_gdn_g_enabled{false};
std::atomic<bool> g_compile_gdn_g_env_read{false};

bool compile_gdn_g_enabled() {
  if (!g_compile_gdn_g_env_read.exchange(true, std::memory_order_acq_rel)) {
    if (const char* v = std::getenv("MLX_WGPU_COMPILE_GDN_G")) {
      if (v[0] == '1') {
        g_compile_gdn_g_enabled.store(true, std::memory_order_relaxed);
      }
    }
  }
  return g_compile_gdn_g_enabled.load(std::memory_order_relaxed);
}

// Captureless free function describing the GDN decay-gate (compute_g) tape
// in log-space. Inputs layout MUST match exactly what
// mlx_gdn_compute_g_forward passes — note that ALL weight-shaped operands
// AND the scalar clamp threshold are PRE-BROADCAST to [B, T, Hv] OUTSIDE
// this body by the caller:
//
//   inputs[0] = big_a    [B, T, Hv]      pre-broadcast exp(a_log)
//   inputs[1] = a        [B, T, Hv]      from prefusion split
//   inputs[2] = dt_bias  [B, T, Hv]      pre-broadcast dt_bias
//   inputs[3] = thresh   [B, T, Hv]      pre-broadcast 20.0 clamp threshold
//
// Why pre-broadcast OUTSIDE the compile body (vs. letting the implicit
// broadcasts live inside): Phase 6e is the first compile-wrapped body in
// this file whose weight-shaped operands ([Hv]) have to broadcast up to
// per-token shape ([B, T, Hv]) inside the trace. Under `shapeless=true`
// the WebGPU compile backend's `build_tape_expression`
// (backend/webgpu/compiled.cpp:147-204) throws "static cast primitive
// with != 1 inputs" on the resulting Broadcast nodes because the dynamic
// `broadcast_arrays` branch (ops.cpp:1705-1740) emits multi-input
// Broadcast primitives with the broadcasted array PLUS stop-gradient
// references to every peer broadcast operand, and the CSE-collapsed
// tape walks `is_static_cast` with `in_tmps.size() != 1`.
//
// The fix is two-fold:
//   1. `shapeless=false` in `compiled_gdn_compute_g()` so broadcast_arrays
//      takes the non-dynamic branch (ops.cpp:1655-1668), which emits
//      single-input Broadcasts and early-outs when shapes match.
//   2. Pre-broadcasting every non-`a` operand (big_a, dt_bias_b, thresh)
//      to `a->shape()` before entering the trace so the non-dynamic
//      branch's early-out fires for every one of them — the trace sees
//      only same-shape element-wise ops (Exp, Add, Log, Multiply,
//      Negative, Greater, Select) with zero Broadcast nodes inside.
//
// The pre-broadcasts themselves are metadata/stride-only ops in MLX
// (zero-dispatch under the Phase 4 buffer-copy fast path) so there is
// no added GPU cost.
//
// Returns: { g_log } where
//     sp_raw = log(exp(x) + 1)
//     sp     = where(x > 20, x, sp_raw)                // stability clamp
//     g_log  = -(big_a * sp)
// and x = a + dt_bias. This exactly mirrors the pre-Phase-6e Rust inline
// path's softplus stability clamp, which is required for byte-identical
// greedy decode on any input above ~20 (the A/B harness deliberately
// injects +25 spikes into the >20 region to gate on the clamp).
//
// The tape is composed of unary + binary ops plus a Select ternary that
// the WebGPU compile pass fuses:
//   * Exp / Add / Log / Multiply / Negative — the original 6b/6c/6d
//     pattern.
//   * Greater + Select — Select handled at compiled.cpp:170-178, and
//     Greater is a binary op with a WGSL expression in op_exprs.h:232.
// `log1p` is intentionally NOT used: it would produce a Log1p primitive
// that differs bit-for-bit from `log(exp(x)+1)` for small x, breaking
// byte-identical greedy decode parity with the pre-Phase-6e Rust inline
// reference at `gated_delta.rs:45-56` (which writes
// `x.exp()?.add_scalar(1.0)?.log()?`).
//
// Op order mirrors the eager path in mlx_gdn_compute_g_forward below
// exactly, so the compile pass sees a tape whose eager-path counterpart
// is byte-for-byte equivalent — guaranteed by the A/B parity test in
// mlx_nn_ops.cpp (maxAbsErr === 0 gate).
std::vector<mlx::core::array> gdn_compute_g_body(
    const std::vector<mlx::core::array>& inputs) {
  using mlx::core::array;
  // All four operands arrive at [B, T, Hv]. See the block comment above
  // for the rationale (WebGPU compile backend's broadcast handling).
  const auto& big_a = inputs[0];
  const auto& a = inputs[1];
  const auto& dt_bias_b = inputs[2];
  const auto& thresh_b = inputs[3];

  auto x = a + dt_bias_b;                             // [B, T, Hv]
  // softplus via log(exp(x) + 1) with a where(x > 20, x, sp_raw)
  // stability clamp — op-for-op identical to the pre-Phase-6e Rust
  // inline path.
  auto exp_x = exp(x);
  auto one = array(1.0f, x.dtype());
  auto sp_raw = log(exp_x + one);
  auto sp = where(greater(x, thresh_b), x, sp_raw);
  auto g_log = negative(big_a * sp);                  // -big_a * sp
  return {g_log};
}

auto& compiled_gdn_compute_g() {
  // shapeless=false: must be concrete-shape tracing, NOT dynamic tracing.
  // Under dynamic tracing (shapeless=true) MLX's `broadcast_arrays` in
  // ops.cpp:1705-1740 takes the dynamic branch which emits `Broadcast`
  // primitives with multi-input layout (broadcasted array + stop-gradient
  // references to all other broadcast operands). The WebGPU compile
  // backend's build_tape_expression (backend/webgpu/compiled.cpp:147-204)
  // then hits `is_static_cast(prim)` and throws
  // "static cast primitive with != 1 inputs" because it assumes Broadcast
  // is the unary primitive with exactly 1 input. Concrete-shape tracing
  // uses the non-dynamic branch (ops.cpp:1655-1668) which creates unary
  // single-input Broadcasts and, critically, early-outs when shapes
  // already match — which they do in this body because the caller
  // pre-broadcasts the small operands before entering the trace.
  //
  // Shape-key cost: each distinct [B, T, Hv] signature triggers a new
  // trace + compile entry. In production decode T=1 is steady-state so
  // one trace is reused across all tokens and layers — same cache
  // behavior as Phase 6c/6d (which also use shapeless=false).
  static auto fn = mlx::core::compile(gdn_compute_g_body, /*shapeless=*/false);
  return fn;
}

// The compilable body. MUST be a top-level non-capturing function so
// `mlx::core::compile` keys it by stable function identity (the template
// overload in mlx/compile.h converts a capture-less lambda to a function
// pointer for this exact reason — using a free function makes that
// guarantee explicit).
//
// Inputs layout: [x, w_gate_t, w_up_t, w_down_t]
//   x          : (B, T, hidden)        — pre-transposed activations
//   w_gate_t   : (hidden, intermediate)
//   w_up_t     : (hidden, intermediate)
//   w_down_t   : (intermediate, hidden)
// Returns: { output } where output is (B, T, hidden).
std::vector<mlx::core::array> swiglu_mlp_body(
    const std::vector<mlx::core::array>& inputs) {
  using mlx::core::matmul;
  using mlx::core::sigmoid;
  const auto& x = inputs[0];
  const auto& w_gate_t = inputs[1];
  const auto& w_up_t = inputs[2];
  const auto& w_down_t = inputs[3];
  auto gate = matmul(x, w_gate_t);
  auto up = matmul(x, w_up_t);
  auto gate_act = gate * sigmoid(gate);  // silu(gate)
  auto gated = gate_act * up;
  auto output = matmul(gated, w_down_t);
  return {output};
}

std::vector<mlx::core::array> swiglu_body(
    const std::vector<mlx::core::array>& inputs) {
  using mlx::core::sigmoid;
  const auto& gate = inputs[0];
  const auto& up = inputs[1];
  auto gate_act = gate * sigmoid(gate);
  return {gate_act * up};
}

// Lazy-initialized compile cache. The first call traces swiglu_mlp_body for
// the input-shape-set; subsequent calls with the same shapes hit the
// internal CompilerCache (mlx/compile.cpp). For Qwen3.5-0.8B decode every
// MLP layer in every step has Tq=1 + identical hidden/intermediate sizes,
// so we get 1 trace + N reuses across the whole generation.
auto& compiled_swiglu_mlp() {
  static auto fn = mlx::core::compile(swiglu_mlp_body, /*shapeless=*/false);
  return fn;
}

auto& compiled_swiglu() {
  static auto fn = mlx::core::compile(swiglu_body, /*shapeless=*/false);
  return fn;
}

}  // namespace

extern "C" {

// Runtime setter for the compile MLP fast path. Mirrors the
// `mlx_wgpu_set_packed_bf16_enabled` / `mlx_wgpu_set_sdpa_fallback_forced`
// pattern in mlx_stream.cpp so the TS init message can flip it via
// `wgpuSetSwigluCompileEnabled(true)` without rebuilding the wasm binary.
//
// Marking the env-as-read flag on every set ensures that an explicit
// setter call always wins over a later env-var read.
void mlx_set_swiglu_compile_enabled(bool enabled) {
  g_compile_mlp_enabled.store(enabled, std::memory_order_relaxed);
  g_compile_mlp_env_read.store(true, std::memory_order_release);
}

bool mlx_get_swiglu_compile_enabled() {
  return compile_mlp_enabled();
}

mlx_array* mlx_swiglu_forward(mlx_array* gate_handle, mlx_array* up_handle) {
  try {
    if (!gate_handle || !up_handle) {
      throw std::runtime_error("required input handle is null");
    }
    auto gate = reinterpret_cast<array*>(gate_handle);
    auto up = reinterpret_cast<array*>(up_handle);

    if (compile_mlp_enabled()) {
      std::vector<array> ins{*gate, *up};
      auto outs = compiled_swiglu()(ins);
      return reinterpret_cast<mlx_array*>(new array(std::move(outs[0])));
    }

    auto gate_act = *gate * sigmoid(*gate);
    auto output = gate_act * *up;
    return reinterpret_cast<mlx_array*>(new array(std::move(output)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_swiglu_forward error: " << e.what() << std::endl;
    return nullptr;
  }
}

// Phase 6c: runtime setter/getter for the GDN pre-recurrence compile fast
// path. Mirrors the Phase 6b pair above exactly — flipping the process-wide
// std::atomic<bool> that mlx_gdn_prefusion_forward checks before each call.
// Marking env-as-read on set ensures explicit setter calls always win over
// a later env-var read.
void mlx_set_gdn_pre_compile_enabled(bool enabled) {
  g_compile_gdn_pre_enabled.store(enabled, std::memory_order_relaxed);
  g_compile_gdn_pre_env_read.store(true, std::memory_order_release);
}

bool mlx_get_gdn_pre_compile_enabled() {
  return compile_gdn_pre_enabled();
}

// Phase 6d: runtime setter/getter for the GDN post-recurrence compile fast
// path. Mirrors the Phase 6b/6c pairs above — flipping the process-wide
// std::atomic<bool> that mlx_gdn_postfusion_forward checks before each call.
// Marking env-as-read on set ensures explicit setter calls always win over
// a later env-var read.
void mlx_set_gdn_post_compile_enabled(bool enabled) {
  g_compile_gdn_post_enabled.store(enabled, std::memory_order_relaxed);
  g_compile_gdn_post_env_read.store(true, std::memory_order_release);
}

bool mlx_get_gdn_post_compile_enabled() {
  return compile_gdn_post_enabled();
}

// Phase 6e: runtime setter/getter for the GDN decay-gate (compute_g)
// compile fast path. Mirrors the Phase 6b/6c/6d pairs above — flipping
// the process-wide std::atomic<bool> that mlx_gdn_compute_g_forward
// checks before each call. Marking env-as-read on set ensures explicit
// setter calls always win over a later env-var read.
void mlx_set_gdn_g_compile_enabled(bool enabled) {
  g_compile_gdn_g_enabled.store(enabled, std::memory_order_relaxed);
  g_compile_gdn_g_env_read.store(true, std::memory_order_release);
}

bool mlx_get_gdn_g_compile_enabled() {
  return compile_gdn_g_enabled();
}

// Fused SwiGLU MLP forward pass
// Combines 5 operations into 1 FFI call:
// 1. gate = x @ w_gate.T
// 2. up = x @ w_up.T
// 3. gate_act = silu(gate) = gate * sigmoid(gate)
// 4. gated = gate_act * up
// 5. output = gated @ w_down.T
//
// Phase 6b: when MLX_WGPU_COMPILE_MLP=1 (or the runtime setter is on), the
// element-wise tail is routed through `mlx::core::compile` so the
// sigmoid + multiply + multiply chain collapses into a single Compiled
// kernel. Matmuls remain separate non-fusable primitives. Numerics are
// unchanged.
mlx_array* mlx_swiglu_mlp_forward(mlx_array* x_handle,
                                   mlx_array* w_gate_handle,
                                   mlx_array* w_up_handle,
                                   mlx_array* w_down_handle) {
  auto x = reinterpret_cast<array*>(x_handle);
  auto w_gate = reinterpret_cast<array*>(w_gate_handle);
  auto w_up = reinterpret_cast<array*>(w_up_handle);
  auto w_down = reinterpret_cast<array*>(w_down_handle);

  // Transpose weights: [out, in] -> [in, out] for matmul.
  // These are metadata-only view ops (no GPU dispatch) and stay OUTSIDE the
  // compiled body so the compile cache key is keyed only on the dtype/shape
  // of x and the transposed weights, not on view-flag noise.
  auto w_gate_t = transpose(*w_gate, {1, 0});
  auto w_up_t = transpose(*w_up, {1, 0});
  auto w_down_t = transpose(*w_down, {1, 0});

  if (compile_mlp_enabled()) {
    std::vector<array> ins{*x, w_gate_t, w_up_t, w_down_t};
    auto outs = compiled_swiglu_mlp()(ins);
    return reinterpret_cast<mlx_array*>(new array(std::move(outs[0])));
  }

  // Eager (default) path — unchanged from pre-Phase-6b.
  // gate = x @ w_gate.T
  auto gate = matmul(*x, w_gate_t);

  // up = x @ w_up.T
  auto up = matmul(*x, w_up_t);

  // silu(gate) = gate * sigmoid(gate)
  auto gate_act = gate * sigmoid(gate);

  // gated = gate_act * up
  auto gated = gate_act * up;

  // output = gated @ w_down.T
  auto output = matmul(gated, w_down_t);

  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// E39: Stacked SwiGLU MLP. Takes pre-stacked + pre-transposed weights computed
// once at model load (see MLP::finalize_gate_up in transformer/mlp.rs):
//   w_gate_up_t: [hidden, 2*intermediate]  (concatenate then transpose)
//   w_down_t:    [intermediate, hidden]    (transpose)
//
// Saves one of the two MLP matmuls (gate and up fused into one [hidden,
// 2*intermediate] matmul). For Qwen3.5-4B: hidden=2560, intermediate=9216, so
// the fused matmul is x @ W [B,T,2560] @ [2560, 18432]. Lookup of W tile/cache
// is amortized vs the two separate matmuls.
mlx_array* mlx_swiglu_mlp_forward_stacked(mlx_array* x_handle,
                                           mlx_array* w_gate_up_t_handle,
                                           mlx_array* w_down_t_handle) {
  auto x = reinterpret_cast<array*>(x_handle);
  auto w_gate_up_t = reinterpret_cast<array*>(w_gate_up_t_handle);
  auto w_down_t = reinterpret_cast<array*>(w_down_t_handle);

  // x: [B, T, hidden]; w_gate_up_t: [hidden, 2*intermediate]
  auto gate_up = matmul(*x, *w_gate_up_t);  // [B, T, 2*intermediate]

  int intermediate = static_cast<int>(gate_up.shape().back()) / 2;
  int B = static_cast<int>(gate_up.shape()[0]);
  int T = static_cast<int>(gate_up.shape()[1]);

  auto gate = slice(gate_up, {0, 0, 0}, {B, T, intermediate});
  auto up   = slice(gate_up, {0, 0, intermediate}, {B, T, 2 * intermediate});

  // E40: fused swiglu via mlx::core::compile (one Metal kernel for
  // sigmoid(gate)*gate*up). Helper is from mlx_qwen35_common.h.
  auto gated = qwen35_common::swiglu(gate, up);

  // down: gated @ w_down_t
  auto output = matmul(gated, *w_down_t);
  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Combines: norm -> attention -> residual -> norm -> mlp -> residual
// Reduces ~40 FFI calls to 1 per block
mlx_array* mlx_fused_transformer_block_forward(
    mlx_array* x_handle,
    // Layer norm weights
    mlx_array* input_norm_w_handle,
    mlx_array* post_attn_norm_w_handle,
    // Attention weights
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* w_o_handle,
    mlx_array* q_norm_w_handle,  // Can be nullptr
    mlx_array* k_norm_w_handle,  // Can be nullptr
    // MLP weights
    mlx_array* w_gate_handle,
    mlx_array* w_up_handle,
    mlx_array* w_down_handle,
    // Config
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float attn_scale,
    float rope_base,
    int rope_dims,
    float norm_eps,
    float qk_norm_eps,
    bool use_causal,
    int rope_offset) {

  auto x = reinterpret_cast<array*>(x_handle);
  auto input_norm_w = reinterpret_cast<array*>(input_norm_w_handle);
  auto post_attn_norm_w = reinterpret_cast<array*>(post_attn_norm_w_handle);
  auto w_q = reinterpret_cast<array*>(w_q_handle);
  auto w_k = reinterpret_cast<array*>(w_k_handle);
  auto w_v = reinterpret_cast<array*>(w_v_handle);
  auto w_o = reinterpret_cast<array*>(w_o_handle);
  auto w_gate = reinterpret_cast<array*>(w_gate_handle);
  auto w_up = reinterpret_cast<array*>(w_up_handle);
  auto w_down = reinterpret_cast<array*>(w_down_handle);

  // Get input shape
  int batch = static_cast<int>(x->shape()[0]);
  int seq_len = static_cast<int>(x->shape()[1]);

  // === Part 1: Self-Attention ===

  // 1. Input layer norm
  auto normed = fast::rms_norm(*x, std::optional<array>(*input_norm_w), norm_eps, {});

  // 2. Q/K/V projections
  auto w_q_t = transpose(*w_q, {1, 0});
  auto w_k_t = transpose(*w_k, {1, 0});
  auto w_v_t = transpose(*w_v, {1, 0});
  auto w_o_t = transpose(*w_o, {1, 0});

  auto queries = matmul(normed, w_q_t);
  auto keys = matmul(normed, w_k_t);
  auto values = matmul(normed, w_v_t);

  // 3. Reshape to multi-head format
  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  // 4. QK normalization
  if (q_norm_w_handle) {
    auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w_handle) {
    auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  // 5. Transpose to attention layout
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  // 6. Apply RoPE
  bool traditional = false;
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});

  // 7. Scaled dot-product attention
  std::string mask_mode = use_causal && seq_len > 1 ? "causal" : "";
  auto attn_output = fast::scaled_dot_product_attention(queries, keys, values, attn_scale, mask_mode, {}, std::nullopt, {});
  attn_output.eval();  // Force GPU sync after SDPA to prevent timeout

  // 8. Transpose back and reshape
  attn_output = transpose(attn_output, {0, 2, 1, 3});
  attn_output = reshape(attn_output, {batch, seq_len, n_heads * head_dim});

  // 9. Output projection
  attn_output = matmul(attn_output, w_o_t);

  // 10. Attention residual
  auto h = *x + attn_output;

  // === Part 2: MLP ===

  // 11. Post-attention layer norm
  auto mlp_input = fast::rms_norm(h, std::optional<array>(*post_attn_norm_w), norm_eps, {});

  // 12. MLP (SwiGLU)
  auto w_gate_t = transpose(*w_gate, {1, 0});
  auto w_up_t = transpose(*w_up, {1, 0});
  auto w_down_t = transpose(*w_down, {1, 0});

  auto gate = matmul(mlp_input, w_gate_t);
  auto up = matmul(mlp_input, w_up_t);
  auto gate_act = gate * sigmoid(gate);  // SiLU
  auto gated = gate_act * up;
  auto mlp_output = matmul(gated, w_down_t);

  // 13. MLP residual
  auto output = h + mlp_output;

  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}
// Fused Q/K/V projection with RoPE for cached attention
// Returns Q, K, V in attention layout (B, n_heads, L, head_dim) with RoPE applied
// This fuses: projection -> reshape -> qk_norm -> transpose -> RoPE
void mlx_fused_attention_qkv(
    mlx_array* x_handle,
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* q_norm_w_handle,  // Can be null
    mlx_array* k_norm_w_handle,  // Can be null
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_base,
    int rope_dims,
    float qk_norm_eps,
    int rope_offset,
    mlx_array** q_out,
    mlx_array** k_out,
    mlx_array** v_out
) {
    try {
        auto x = reinterpret_cast<array*>(x_handle);
        auto w_q = reinterpret_cast<array*>(w_q_handle);
        auto w_k = reinterpret_cast<array*>(w_k_handle);
        auto w_v = reinterpret_cast<array*>(w_v_handle);

        int batch = static_cast<int>(x->shape()[0]);
        int seq_len = static_cast<int>(x->shape()[1]);

        // Transpose weights for matmul: (hidden, proj) -> (proj, hidden)
        auto w_q_t = transpose(*w_q);
        auto w_k_t = transpose(*w_k);
        auto w_v_t = transpose(*w_v);

        // 1. Q/K/V projections
        auto queries = matmul(*x, w_q_t);  // (B, L, n_heads * head_dim)
        auto keys = matmul(*x, w_k_t);     // (B, L, n_kv_heads * head_dim)
        auto values = matmul(*x, w_v_t);   // (B, L, n_kv_heads * head_dim)

        // 2. Reshape to multi-head format: (B, L, n_heads, head_dim)
        queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
        keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
        values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

        // 3. Apply QK normalization BEFORE transpose (matching transformers)
        if (q_norm_w_handle) {
            auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
            queries = mlx::core::fast::rms_norm(queries, *q_norm_w, qk_norm_eps);
        }
        if (k_norm_w_handle) {
            auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
            keys = mlx::core::fast::rms_norm(keys, *k_norm_w, qk_norm_eps);
        }

        // 4. Transpose to attention layout: (B, n_heads, L, head_dim)
        queries = transpose(queries, {0, 2, 1, 3});
        keys = transpose(keys, {0, 2, 1, 3});
        values = transpose(values, {0, 2, 1, 3});

        // 5. Apply RoPE
        queries = mlx::core::fast::rope(queries, rope_dims, false, rope_base, 1.0f, rope_offset);
        keys = mlx::core::fast::rope(keys, rope_dims, false, rope_base, 1.0f, rope_offset);

        *q_out = reinterpret_cast<mlx_array*>(new array(std::move(queries)));
        *k_out = reinterpret_cast<mlx_array*>(new array(std::move(keys)));
        *v_out = reinterpret_cast<mlx_array*>(new array(std::move(values)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_fused_attention_qkv error: " << e.what() << std::endl;
        *q_out = nullptr;
        *k_out = nullptr;
        *v_out = nullptr;
    }
}

// Fused SDPA + output projection for cached attention
// Takes Q (B, n_heads, L, head_dim) and full cached K/V (B, n_kv_heads, total_len, head_dim)
// Returns output (B, L, hidden_size)
mlx_array* mlx_fused_attention_output(
    mlx_array* q_handle,
    mlx_array* k_handle,
    mlx_array* v_handle,
    mlx_array* w_o_handle,
    int n_heads,
    int head_dim,
    float attn_scale,
    bool use_causal
) {
    try {
        auto queries = reinterpret_cast<array*>(q_handle);
        auto keys = reinterpret_cast<array*>(k_handle);
        auto values = reinterpret_cast<array*>(v_handle);
        auto w_o = reinterpret_cast<array*>(w_o_handle);

        int batch = static_cast<int>(queries->shape()[0]);
        int q_len = static_cast<int>(queries->shape()[2]);
        int hidden_size = n_heads * head_dim;

        // SDPA - determine mask mode (valid modes: "causal", "array", or "" for none)
        std::string mask_mode = (use_causal && q_len > 1) ? "causal" : "";
        auto attn_output = mlx::core::fast::scaled_dot_product_attention(
            *queries, *keys, *values, attn_scale, mask_mode
        );
        attn_output.eval();  // Force GPU sync after expensive SDPA to prevent timeout

        // Transpose back: (B, n_heads, L, head_dim) -> (B, L, n_heads, head_dim)
        attn_output = transpose(attn_output, {0, 2, 1, 3});

        // Reshape: (B, L, n_heads, head_dim) -> (B, L, hidden_size)
        attn_output = reshape(attn_output, {batch, q_len, hidden_size});

        // Output projection
        auto w_o_t = transpose(*w_o);
        auto output = matmul(attn_output, w_o_t);

        return reinterpret_cast<mlx_array*>(new array(std::move(output)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_fused_attention_output error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Fused GDN pre-recurrence forward pass.
//
// Collapses the pre-recurrence portion of GatedDeltaNet::forward into a single
// FFI call. Replicates the exact op sequence from
// crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs:128-262 so greedy
// decode tokens remain byte-identical to the per-op path.
//
// Fuses:
//   qkvz = x @ w_qkvz.T
//   ba   = x @ w_ba.T
//   b, a = split(ba, num_v_heads, axis=2)
//   qkv, z = split(qkvz, conv_dim, axis=2)
//   if mask: qkv = where(mask[:, :, None], qkv, zeros_of_dtype(qkv))
//   conv_input = concat(conv_state_or_zeros, qkv, axis=1)
//   new_conv_state = conv_input[:, total_len - (kernel-1):, :]
//   conv_out = conv1d(conv_input, w_conv, 1, 0, 1, conv_dim)
//   conv_out = conv_out[:, -T:, :] if longer
//   conv_out = silu(conv_out)
//   q_flat = conv_out[:, :, :key_dim]
//   k_flat = conv_out[:, :, key_dim:2*key_dim]
//   v_flat = conv_out[:, :, 2*key_dim:conv_dim]
//   q = rms_norm_no_weight(q_flat.reshape(B,T,Hk,Dk), 1e-6) * inv_scale^2
//   k = rms_norm_no_weight(k_flat.reshape(B,T,Hk,Dk), 1e-6) * inv_scale
//   v = v_flat.reshape(B,T,Hv,Dv)
//
// z remains flat [B, T, value_dim] matching the slice output (NOT per-head).
//
// Writes 7 outputs via out-pointers. Returns 0 on success, -1 on error.
// Caller owns all returned mlx_array*.
int mlx_gdn_prefusion_forward(
    mlx_array* x_handle,            // [B, T, hidden]
    mlx_array* w_qkvz_handle,       // [key_dim*2 + value_dim*2, hidden]
    mlx_array* w_ba_handle,         // [num_v_heads*2, hidden]
    mlx_array* w_conv_handle,       // [conv_dim, kernel_dim, 1]
    mlx_array* conv_state_handle,   // [B, kernel_dim-1, conv_dim] or nullptr
    mlx_array* mask_handle,         // [B, T] or nullptr
    int num_k_heads,
    int num_v_heads,
    int key_head_dim,
    int value_head_dim,
    int conv_kernel_dim,
    float rms_eps,
    mlx_array** out_q,
    mlx_array** out_k,
    mlx_array** out_v,
    mlx_array** out_z,
    mlx_array** out_a,
    mlx_array** out_b,
    mlx_array** out_new_conv_state) {
    if (!out_q || !out_k || !out_v || !out_z || !out_a || !out_b || !out_new_conv_state) {
        std::cerr << "mlx_gdn_prefusion_forward error: required out-pointer is null"
                  << std::endl;
        return -1;
    }
    *out_q = nullptr;
    *out_k = nullptr;
    *out_v = nullptr;
    *out_z = nullptr;
    *out_a = nullptr;
    *out_b = nullptr;
    *out_new_conv_state = nullptr;
    if (!x_handle || !w_qkvz_handle || !w_ba_handle || !w_conv_handle) {
        std::cerr << "mlx_gdn_prefusion_forward error: required input handle is null"
                  << std::endl;
        return -1;
    }
    try {
        auto x = reinterpret_cast<array*>(x_handle);
        auto w_qkvz = reinterpret_cast<array*>(w_qkvz_handle);
        auto w_ba = reinterpret_cast<array*>(w_ba_handle);
        auto w_conv = reinterpret_cast<array*>(w_conv_handle);

        int batch = static_cast<int>(x->shape()[0]);
        int seq_len = static_cast<int>(x->shape()[1]);
        int key_dim = num_k_heads * key_head_dim;
        int value_dim = num_v_heads * value_head_dim;
        int conv_dim = key_dim * 2 + value_dim;

        // Transposes + squeeze are metadata-only — stay OUTSIDE the compile
        // body so the trace key isn't poisoned with view-flag noise (same
        // lesson learned in Phase 6b).
        auto w_qkvz_t = transpose(*w_qkvz, {1, 0});
        auto w_ba_t = transpose(*w_ba, {1, 0});

        // -----------------------------------------------------------------
        // Phase 6c: mx::compile-wrapped fast path.
        //
        // Only routed when the shape signature matches the stable
        // steady-state decode case: mask absent AND conv_state present.
        // Prefill (mask present OR conv_state null) falls through to the
        // eager path below — that keeps the compile cache keyed on one
        // shape signature, one trace per model, many reuses per token.
        //
        // The body returns { q_flat, k_flat, v_flat, z, a, b, new_conv_state }
        // — per-head reshape, rms_norm, and scaling on q/k happen HERE
        // in the caller (outside the compile body) for two reasons:
        //   1. `fast::rms_norm` is not fusable, so lifting it out of the
        //      body doesn't cost fused ops.
        //   2. It removes the need for a shape-hint input into the body,
        //      which was a source of bind-group layout mismatches in the
        //      Compiled primitive on WebGPU.
        // -----------------------------------------------------------------
        if (compile_gdn_pre_enabled() && !mask_handle && conv_state_handle) {
            auto conv_state = reinterpret_cast<array*>(conv_state_handle);
            // Pre-squeeze w_conv (axis=2) — metadata op, stays outside the
            // traced body for the same reason as the transposes above.
            auto w_conv_2d = squeeze(*w_conv, std::vector<int>{2});
            std::vector<array> ins{
                *x, w_qkvz_t, w_ba_t, w_conv_2d, *conv_state};
            auto outs = compiled_gdn_prefusion()(ins);
            // outs order matches gdn_prefusion_body's return:
            //   { q_flat, k_flat, v_flat, z, a, b, new_conv_state }
            auto q_flat = std::move(outs[0]);
            auto k_flat = std::move(outs[1]);
            auto v_flat = std::move(outs[2]);
            auto z = std::move(outs[3]);
            auto a = std::move(outs[4]);
            auto b = std::move(outs[5]);
            auto new_conv_state = std::move(outs[6]);

            // Per-head reshape + rms_norm (no weight) + inv_scale — the
            // same post-conv tail as the eager path, byte-identical.
            auto q_h = reshape(q_flat, {batch, seq_len, num_k_heads, key_head_dim});
            auto k_h = reshape(k_flat, {batch, seq_len, num_k_heads, key_head_dim});
            auto v_h = reshape(
                v_flat, {batch, seq_len, num_v_heads, value_head_dim});
            double inv_scale_d = std::pow(static_cast<double>(key_head_dim), -0.5);
            float q_scale_f = static_cast<float>(inv_scale_d * inv_scale_d);
            float k_scale_f = static_cast<float>(inv_scale_d);
            auto q_dtype = q_h.dtype();
            auto k_dtype = k_h.dtype();
            auto q = fast::rms_norm(q_h, std::nullopt, rms_eps, {});
            auto k = fast::rms_norm(k_h, std::nullopt, rms_eps, {});
            q = q * array(q_scale_f, q_dtype);
            k = k * array(k_scale_f, k_dtype);

            *out_q = reinterpret_cast<mlx_array*>(new array(std::move(q)));
            *out_k = reinterpret_cast<mlx_array*>(new array(std::move(k)));
            *out_v = reinterpret_cast<mlx_array*>(new array(std::move(v_h)));
            *out_z = reinterpret_cast<mlx_array*>(new array(std::move(z)));
            *out_a = reinterpret_cast<mlx_array*>(new array(std::move(a)));
            *out_b = reinterpret_cast<mlx_array*>(new array(std::move(b)));
            *out_new_conv_state = reinterpret_cast<mlx_array*>(
                new array(std::move(new_conv_state)));
            return 0;
        }

        // 1. Projections: qkvz = x @ w_qkvz.T, ba = x @ w_ba.T
        auto qkvz = matmul(*x, w_qkvz_t);  // [B, T, key_dim*2 + value_dim*2]
        auto ba = matmul(*x, w_ba_t);      // [B, T, num_v_heads*2]

        // 2. Split ba → b, a (each [B, T, num_v_heads]), via slice along axis 2.
        auto b = slice(ba, {0, 0, 0}, {batch, seq_len, num_v_heads});
        auto a = slice(ba, {0, 0, num_v_heads}, {batch, seq_len, num_v_heads * 2});

        // 3. Split qkvz → qkv (conv_dim wide), z (remaining).
        int qkvz_last = key_dim * 2 + value_dim * 2;
        auto qkv = slice(qkvz, {0, 0, 0}, {batch, seq_len, conv_dim});
        auto z = slice(qkvz, {0, 0, conv_dim}, {batch, seq_len, qkvz_last});

        // 4. Optional mask application: qkv = where(mask[:,:,None], qkv, zeros).
        //    Use qkv's dtype to avoid f32 promotion for bf16/f16 models.
        if (mask_handle) {
            auto mask = reinterpret_cast<array*>(mask_handle);
            auto m_3d = reshape(*mask, {batch, seq_len, 1});
            auto zero_scalar = zeros({1}, qkv.dtype());
            qkv = where(m_3d, qkv, zero_scalar);
        }

        // 5. Prepend conv_state (or zero pad) to qkv along the time axis.
        array conv_input = [&]() -> array {
            int pad_len = conv_kernel_dim - 1;
            if (conv_state_handle) {
                auto conv_state = reinterpret_cast<array*>(conv_state_handle);
                return concatenate({*conv_state, qkv}, 1);
            }
            auto zero_pad = zeros({batch, pad_len, conv_dim}, qkv.dtype());
            return concatenate({zero_pad, qkv}, 1);
        }();

        // 6. Save new conv_state = conv_input[:, total_len - (kernel-1):, :].
        int total_len = static_cast<int>(conv_input.shape()[1]);
        int keep = conv_kernel_dim - 1;
        array new_conv_state = (total_len >= keep)
            ? slice(conv_input,
                    {0, total_len - keep, 0},
                    {batch, total_len, conv_dim})
            : conv_input;

        // 7. Depthwise conv1d — replicate Rust Conv1d::forward_depthwise
        //    (crates/mlx-core/src/nn/conv1d.rs) byte-for-byte so both the
        //    per-op WASM path and this fused path build the same lazy graph
        //    and therefore produce identical bf16 values.
        //
        //    Weight: [C, K, 1] → squeeze(axis=2) → [C, K]
        //    For each k in [0..K):
        //      input_slice = conv_input[:, k:k+t_out, :]   (axis=1 slice)
        //      w_k         = weight_2d[:, k:k+1]           ([C, 1])
        //                      .squeeze(axis=1)             ([C])
        //                      .reshape({1, 1, -1})         ([1, 1, C])
        //      term        = input_slice * w_k
        //      result      = (k == 0) ? term : result + term
        int conv_in_len = static_cast<int>(conv_input.shape()[1]);
        int t_out = conv_in_len - conv_kernel_dim + 1;
        if (t_out <= 0) {
            throw std::runtime_error(
                "mlx_gdn_prefusion_forward: input seq_len < kernel_size");
        }
        auto weight_2d = squeeze(*w_conv, std::vector<int>{2});  // [C, K]
        array conv_out_acc = array(0.0f, conv_input.dtype());     // placeholder
        for (int k = 0; k < conv_kernel_dim; ++k) {
            auto input_slice = slice(
                conv_input,
                {0, k, 0},
                {batch, k + t_out, conv_dim});  // [B, t_out, C]
            auto w_col = slice(weight_2d, {0, k}, {conv_dim, k + 1});  // [C, 1]
            auto w_col_1d = squeeze(w_col, std::vector<int>{1});       // [C]
            auto w_k = reshape(w_col_1d, {1, 1, conv_dim});            // [1, 1, C]
            auto term = input_slice * w_k;
            conv_out_acc = (k == 0) ? term : (conv_out_acc + term);
        }
        auto conv_out = conv_out_acc;

        // 8. If conv produced more than seq_len timesteps, take the last seq_len.
        int conv_out_len = static_cast<int>(conv_out.shape()[1]);
        if (conv_out_len > seq_len) {
            conv_out = slice(conv_out,
                             {0, conv_out_len - seq_len, 0},
                             {batch, conv_out_len, conv_dim});
        }

        // 9. SiLU activation.
        conv_out = conv_out * sigmoid(conv_out);

        // 10. Split into q, k, v (flat along channel axis).
        auto q_flat = slice(conv_out, {0, 0, 0}, {batch, seq_len, key_dim});
        auto k_flat = slice(conv_out, {0, 0, key_dim}, {batch, seq_len, key_dim * 2});
        auto v_flat = slice(conv_out, {0, 0, key_dim * 2}, {batch, seq_len, conv_dim});

        // 11. Reshape to [B, T, H, D].
        auto q = reshape(q_flat, {batch, seq_len, num_k_heads, key_head_dim});
        auto k = reshape(k_flat, {batch, seq_len, num_k_heads, key_head_dim});
        auto v = reshape(v_flat, {batch, seq_len, num_v_heads, value_head_dim});

        // 12. RMS norm (no weight) and inv_scale application matching the
        //     Rust per-op path exactly:
        //       inv_scale = key_head_dim^(-0.5)
        //       q = rms_norm_no_weight(q, eps) * (inv_scale * inv_scale) as scalar
        //       k = rms_norm_no_weight(k, eps) * inv_scale as scalar
        //
        //     Multiply by a dtype-matched scalar array so bf16 stays bf16.
        //     Match Rust `(key_head_dim as f64).powf(-0.5)` precision by
        //     computing the scalar in double and casting to float at the
        //     point where mlx_array_mul_scalar builds its scalar array.
        double inv_scale_d = std::pow(static_cast<double>(key_head_dim), -0.5);
        float q_scale_f = static_cast<float>(inv_scale_d * inv_scale_d);
        float k_scale_f = static_cast<float>(inv_scale_d);
        auto q_dtype = q.dtype();
        auto k_dtype = k.dtype();
        q = fast::rms_norm(q, std::nullopt, rms_eps, {});
        k = fast::rms_norm(k, std::nullopt, rms_eps, {});
        q = q * array(q_scale_f, q_dtype);
        k = k * array(k_scale_f, k_dtype);

        *out_q = reinterpret_cast<mlx_array*>(new array(std::move(q)));
        *out_k = reinterpret_cast<mlx_array*>(new array(std::move(k)));
        *out_v = reinterpret_cast<mlx_array*>(new array(std::move(v)));
        *out_z = reinterpret_cast<mlx_array*>(new array(std::move(z)));
        *out_a = reinterpret_cast<mlx_array*>(new array(std::move(a)));
        *out_b = reinterpret_cast<mlx_array*>(new array(std::move(b)));
        *out_new_conv_state =
            reinterpret_cast<mlx_array*>(new array(std::move(new_conv_state)));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "mlx_gdn_prefusion_forward error: " << e.what() << std::endl;
        *out_q = nullptr;
        *out_k = nullptr;
        *out_v = nullptr;
        *out_z = nullptr;
        *out_a = nullptr;
        *out_b = nullptr;
        *out_new_conv_state = nullptr;
        return -1;
    }
}

// Fused GDN post-recurrence forward pass.
//
// Collapses the RMSNormGated + swiglu + out_proj tail of
// GatedDeltaNet::forward (see the post-recurrence branch in
// crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs) into a single FFI
// call so the WebGPU backend sees one lazy graph instead of 5 per-layer
// dispatches. Byte-identical greedy-decode parity is preserved by mirroring
// the Rust fallback op-for-op:
//
//   // Rust (per-op fallback):
//   let z = z.reshape(&[B, T, Hv, Dv]);                       // metadata
//   let normed = fast::rms_norm(y_att, norm_weight, eps);     // 1 dispatch
//   let sig = sigmoid(z_4d);                                  // 1 dispatch
//   let silu_gate = z_4d * sig;                               // 1 dispatch
//   let gated = silu_gate * normed;                           // 1 dispatch
//   let gated_flat = gated.reshape(&[B, T, Hv * Dv]);         // metadata
//   let y_out = matmul(gated_flat, out_proj_weight.T);        // 1 dispatch
//   let y_out = y_out + bias;  // only if bias is not null    // 1 dispatch
//
// Order of ops (sigmoid → mul(z, sig) → mul(silu_gate, normed) → matmul → add)
// is load-bearing — reordering would diverge bf16 accumulation and fail the
// byte-identical token-ID gate.
//
// Writes the final output via out_y. Returns 0 on success, -1 on error
// (out_y is set to null on error).
int mlx_gdn_postfusion_forward(
    mlx_array* y_att_handle,          // [B, T, Hv, Dv]
    mlx_array* z_handle,              // [B, T, Hv * Dv]
    mlx_array* norm_weight_handle,    // [Dv]
    mlx_array* out_proj_weight_handle,// [hidden, Hv * Dv] (Linear layout: [out, in])
    mlx_array* out_proj_bias_handle,  // [hidden] or nullptr
    float rms_eps,
    int batch_size,
    int seq_len,
    int n_v_heads,
    int v_head_dim,
    int intermediate_size,            // Hv * Dv
    mlx_array** out_y) {
    if (!out_y) {
        std::cerr << "mlx_gdn_postfusion_forward error: out_y is null" << std::endl;
        return -1;
    }
    *out_y = nullptr;
    if (!y_att_handle || !z_handle || !norm_weight_handle || !out_proj_weight_handle) {
        std::cerr << "mlx_gdn_postfusion_forward error: required input handle is null" << std::endl;
        return -1;
    }
    try {
        auto y_att = reinterpret_cast<array*>(y_att_handle);
        auto z = reinterpret_cast<array*>(z_handle);
        auto norm_weight = reinterpret_cast<array*>(norm_weight_handle);
        auto out_proj_weight = reinterpret_cast<array*>(out_proj_weight_handle);

        // 1. Reshape z from [B, T, Hv*Dv] to [B, T, Hv, Dv] (metadata only).
        //    Rust caller reshapes z to per-head format before calling norm —
        //    see gated_delta_net.rs:394-399.
        auto z_4d = reshape(*z, {batch_size, seq_len, n_v_heads, v_head_dim});

        // -----------------------------------------------------------------
        // Phase 6d: mx::compile-wrapped fast path.
        //
        // Routed only when (a) the caller has the compile flag on, (b)
        // there is no out_proj bias (the default Qwen3.5 GDN layout), and
        // (c) rms_eps matches the hardcoded value in the body. Any other
        // case falls through to the eager path below so the compile cache
        // key stays on one stable trace per shape signature.
        //
        // The body returns { y_out } = matmul(gated_flat, w_out_t), so the
        // optional bias add stays in the caller (and is unreachable here
        // because we gate on bias == null anyway).
        //
        // w_out_t = transpose(out_proj_weight) is metadata-only and stays
        // outside the compile body — same lesson as Phase 6b/6c.
        // -----------------------------------------------------------------
        constexpr float PHASE6D_RMS_EPS = 1e-6f;
        if (compile_gdn_post_enabled() &&
            out_proj_bias_handle == nullptr &&
            rms_eps == PHASE6D_RMS_EPS) {
            auto w_out_t = transpose(*out_proj_weight, {1, 0});
            std::vector<array> ins{*y_att, z_4d, *norm_weight, w_out_t};
            auto outs = compiled_gdn_postfusion()(ins);
            auto y_out = std::move(outs[0]);
            *out_y = reinterpret_cast<mlx_array*>(new array(std::move(y_out)));
            return 0;
        }

        // 2. RMSNormGated step 1 — rms_norm with weight [Dv] operates on the
        //    last axis of y_att. Rust: rms_norm_gated.rs:26 calls
        //    mlx::core::fast::rms_norm on the 4D y tensor.
        auto normed = fast::rms_norm(
            *y_att, std::optional<array>(*norm_weight), rms_eps, {});

        // 3. RMSNormGated step 2 — swiglu(gate=z, up=normed):
        //      silu_gate = z * sigmoid(z)
        //      result    = silu_gate * normed
        //    Rust: nn/activations.rs:160-164 (Activations::swiglu) invokes
        //    Self::silu(gate) which is
        //    nn/activations.rs:16-26 (x * sigmoid(x)), then `silu_gate.mul(up)`.
        auto sig = sigmoid(z_4d);
        auto silu_gate = z_4d * sig;
        auto gated = silu_gate * normed;

        // 4. Flatten heads: [B, T, Hv, Dv] → [B, T, Hv*Dv] (metadata only).
        //    Rust: gated_delta_net.rs:407 — `y_normed.reshape(&[B, T, value_dim])`.
        auto gated_flat = reshape(gated, {batch_size, seq_len, intermediate_size});

        // 5. Out-projection matmul. Linear stores weight as [out, in] and
        //    computes y = x @ weight.T — see nn/linear.rs:114-117. We
        //    transpose here to match exactly.
        auto w_out_t = transpose(*out_proj_weight, {1, 0});
        auto y_out = matmul(gated_flat, w_out_t);

        // 6. Optional bias. The default Qwen3.5 GDN out_proj has no bias, but
        //    we still support it for parity with Linear::forward when bias
        //    is present.
        if (out_proj_bias_handle) {
            auto bias = reinterpret_cast<array*>(out_proj_bias_handle);
            y_out = y_out + *bias;
        }

        *out_y = reinterpret_cast<mlx_array*>(new array(std::move(y_out)));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "mlx_gdn_postfusion_forward error: " << e.what() << std::endl;
        *out_y = nullptr;
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Phase 6e: Fused GDN decay-gate (compute_g) forward pass.
//
// Collapses the 9-op softplus / exp / multiply chain that the WASM Rust
// decode path builds inline (gated_delta.rs, use_kernel=true branch) into
// one FFI call. When the Phase 6e compile flag is on, the tape is routed
// through mlx::core::compile so the entire element-wise chain fuses into
// (best case) one Compiled kernel — ~8 dispatches saved per linear-
// attention layer per token. Across Qwen3.5-0.8B's 18 such layers that is
// ~144 dispatches/tok at the Phase 6e shape signature.
//
// Signature returns g_log (pre-outer-exp), matching the existing inline
// WASM semantics. The caller's downstream `g = g_log.exp()` step stays as
// its own dispatch so the surrounding decode tape is unchanged.
//
// Inputs:
//   a_log_handle   : [Hv]                   learnable log decay
//   a_handle       : [B, T, Hv]             from prefusion split
//   dt_bias_handle : [Hv]                   learnable bias
//
// Returns: int rc (0 success, -1 error); writes *out_g_log on success.
// Caller owns the returned mlx_array*.
int mlx_gdn_compute_g_forward(
    mlx_array* a_log_handle,
    mlx_array* a_handle,
    mlx_array* dt_bias_handle,
    mlx_array** out_g_log) {
    if (!out_g_log) {
        std::cerr << "mlx_gdn_compute_g_forward error: out_g_log is null" << std::endl;
        return -1;
    }
    *out_g_log = nullptr;
    if (!a_log_handle || !a_handle || !dt_bias_handle) {
        std::cerr << "mlx_gdn_compute_g_forward error: required input handle is null"
                  << std::endl;
        return -1;
    }
    try {
        auto a_log = reinterpret_cast<array*>(a_log_handle);
        auto a = reinterpret_cast<array*>(a_handle);
        auto dt_bias = reinterpret_cast<array*>(dt_bias_handle);

        // -----------------------------------------------------------------
        // Phase 6e: mx::compile-wrapped fast path.
        //
        // The trace is shape-keyed (`shapeless=false`): decode Tq=1 is the
        // steady-state hot path that gets reused across every token and
        // every layer after the first call; prefill Tq=prompt_len produces
        // its own cache entry. Same cache behavior as Phase 6c/6d.
        //
        // Pre-broadcast exp(a_log), dt_bias, and the scalar softplus
        // clamp threshold from their native shapes ([Hv] or []) up to
        // a->shape() BEFORE entering the compile body so the tape sees
        // only same-shape element-wise ops. Without this, WebGPU
        // compiled.cpp throws "static cast primitive with != 1 inputs"
        // on the Broadcast primitives that the compile pass emits for
        // the implicit rank-promotions — see the block comment on
        // `gdn_compute_g_body` above. The broadcasts themselves are
        // metadata/stride-only (zero-dispatch under Phase 4), so this
        // costs nothing at runtime.
        // -----------------------------------------------------------------
        if (compile_gdn_g_enabled()) {
            auto big_a_flat = exp(*a_log);                     // [Hv]
            auto big_a = broadcast_to(big_a_flat, a->shape()); // [B, T, Hv]
            auto dt_bias_b = broadcast_to(*dt_bias, a->shape());
            auto thresh_scalar = array(20.0f, a->dtype());
            auto thresh_b = broadcast_to(thresh_scalar, a->shape());
            std::vector<array> ins{big_a, *a, dt_bias_b, thresh_b};
            auto outs = compiled_gdn_compute_g()(ins);
            *out_g_log = reinterpret_cast<mlx_array*>(new array(std::move(outs[0])));
            return 0;
        }

        // Eager path — op-for-op mirror of `gdn_compute_g_body` above so
        // the synthetic A/B parity gate (maxAbsErr === 0) holds trivially.
        // The explicit pre-broadcasts match the compile-path shape
        // discipline; broadcast_to is stride-only so result bits are
        // identical to an implicit broadcast inside the following binary
        // ops.
        //
        // Includes the `where(x > 20, x, log(exp(x)+1))` softplus
        // stability clamp — this is the spec-required behavior that the
        // pre-Phase-6e Rust inline path computed, and the A/B harness
        // deliberately injects +25 spikes into the >20 region to gate
        // on it. Without the clamp, `exp(x)` overflows to +inf for x
        // beyond ~88 (f32) / ~11 (bf16), so the greedy decode path
        // would diverge on the same inputs.
        auto big_a_flat = exp(*a_log);
        auto big_a = broadcast_to(big_a_flat, a->shape());
        auto dt_bias_b = broadcast_to(*dt_bias, a->shape());
        auto thresh_scalar = array(20.0f, a->dtype());
        auto thresh_b = broadcast_to(thresh_scalar, a->shape());
        auto x = *a + dt_bias_b;
        auto exp_x = exp(x);
        auto one = array(1.0f, x.dtype());
        auto sp_raw = log(exp_x + one);
        auto sp = where(greater(x, thresh_b), x, sp_raw);
        auto g_log = negative(big_a * sp);

        *out_g_log = reinterpret_cast<mlx_array*>(new array(std::move(g_log)));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "mlx_gdn_compute_g_forward error: " << e.what() << std::endl;
        *out_g_log = nullptr;
        return -1;
    }
}

}  // extern "C"
