// Gated delta recurrence with the layer's tail fused in: one threadgroup per
// (batch, value head) runs the per-step kernel for every Dv column (4 columns
// per simdgroup, the `gated_delta_step_4vcol` arithmetic), keeps the window's
// output rows in threadgroup memory, then applies the gated RMSNorm and the
// z gate `sigmoid(z) * z * norm(y)` row by row. Structure after Splash's
// `verify_gdn_fused` (runtime/metal/kernels/decode/gdn.metal, Apache-2.0;
// see THIRD_PARTY_NOTICES); the arithmetic here is MLX's: `rms_single_row`
// (N_READS = 4 lanes over the head, f32 sum, precise rsqrt, bf16 round before
// the weight) and the compiled `sigmoid(gate) * gate * up` with every
// intermediate rounded to the activation dtype, so the output matches the
// unfused kernel -> fast::rms_norm -> compiled swiglu chain bit for bit.
//
// Two kernels come from this source (the host prepends the defines):
//   GDN_FUSED_COMPLETE = 0  mlx_node_gdn_fused_step: prepared inputs
//     q, k [B, T, Hk, Dk] bf16 (value head hv reads key head hv % Hk)
//     v    [B, T, Hv, Dv] bf16,  g [B, T, Hv] f32,  beta [B, T, Hv] bf16
//   GDN_FUSED_COMPLETE = 1  mlx_node_gdn_fused_complete: B = 1, and the
//     `qwen4_gdn_prepare` prologue runs in the same threadgroup — 4-tap
//     depthwise conv over [history ; qkv] + SiLU, q/k L2 norm (eps inside
//     the mean, q scaled by Dk^-1/2), decay/beta gates — into threadgroup
//     memory, with the same arithmetic as that kernel (f32 products summed
//     in tap order, no contraction, bf16 SiLU boundary), and the prepared
//     tape (q, k, v, decay, beta, next_history) also written to device for
//     the commit replay. Value head hv uses key head hv % Hk (tiled GGUF).
// Common:
//   state_in / state_out [B, Hv, Dv, Dk] f32 (state_out may alias a caller
//   buffer: the DFlash2 spare blob row)
//   z    [B, T, Hv * Dv] bf16, read through its row strides (a split view)
//   w    [Dv] bf16 norm weight;  out [B, T, Hv * Dv] bf16
// Threadgroups: B * Hv of 8 * Dv threads (Dv / 4 simdgroups); T <= Dv / 4
// (T <= Dv / 8 for the complete kernel's prepared rows).
R"(
struct GdnFusedParams {
  int T;           // window rows
  int hk;          // key heads
  int hv;          // value heads
  int z_stride_b;  // elements between z batches
  int z_stride_t;  // elements between z rows
  float eps;       // gated norm epsilon
  // GDN_FUSED_COMPLETE only: strides of the split views the prologue reads.
  int qkv_stride_t;
  int qkv_stride_c;
  int hist_stride_r;
  int hist_stride_c;
  int a_stride_t;
  int a_stride_c;
  int b_stride_t;
  int b_stride_c;
  int pending_T; // prior tape rows; dynamic keep is a separate explicit input
};

// `Sigmoid` from mlx/backend/metal/kernels/unary_ops.h, verbatim: the
// compiled z gate evaluates this at T = bfloat16_t, `qwen4_gdn_prepare`'s
// SiLU at bfloat16_t and its beta at float (`precise_sigmoid`).
template <typename T>
inline T gdn_fused_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

#if GDN_FUSED_COMPLETE
[[kernel]] void mlx_node_gdn_fused_complete(
    const device bfloat16_t* qkv [[buffer(0)]],
    const device bfloat16_t* a [[buffer(1)]],
    const device bfloat16_t* b [[buffer(2)]],
    const device float* conv [[buffer(3)]],
    const device bfloat16_t* history [[buffer(4)]],
    const device float* scale [[buffer(5)]],
    const device float* dt [[buffer(6)]],
    const device float* state_in [[buffer(7)]],
    const device bfloat16_t* z [[buffer(8)]],
    const device bfloat16_t* w [[buffer(9)]],
    device bfloat16_t* out [[buffer(10)]],
    device float* state_out [[buffer(11)]],
    device bfloat16_t* q_out [[buffer(12)]],
    device bfloat16_t* k_out [[buffer(13)]],
    device bfloat16_t* v_out [[buffer(14)]],
    device float* decay_out [[buffer(15)]],
    device bfloat16_t* beta_out [[buffer(16)]],
    device bfloat16_t* history_out [[buffer(17)]],
#if GDN_FUSED_PREPARED
    device bfloat16_t* prepared_bt [[buffer(18)]],
    device float* prepared_sums [[buffer(19)]],
    constant GdnFusedParams& p [[buffer(20)]],
#else
    constant GdnFusedParams& p [[buffer(18)]],
#endif
#if GDN_FUSED_DEFERRED
#if GDN_FUSED_PREPARED
#define GDN_PENDING_SLOT 21
#else
#define GDN_PENDING_SLOT 19
#endif
    const device bfloat16_t* pending_k [[buffer(GDN_PENDING_SLOT)]],
    const device bfloat16_t* pending_v [[buffer(GDN_PENDING_SLOT + 1)]],
    const device float* pending_g [[buffer(GDN_PENDING_SLOT + 2)]],
    const device bfloat16_t* pending_beta [[buffer(GDN_PENDING_SLOT + 3)]],
    const device bfloat16_t* pending_qkv [[buffer(GDN_PENDING_SLOT + 4)]],
    const device int* pending_keep [[buffer(GDN_PENDING_SLOT + 5)]],
#endif
#else
[[kernel]] void mlx_node_gdn_fused_step(
    const device bfloat16_t* q [[buffer(0)]],
    const device bfloat16_t* k [[buffer(1)]],
    const device bfloat16_t* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device bfloat16_t* beta [[buffer(4)]],
    const device float* state_in [[buffer(5)]],
    const device bfloat16_t* z [[buffer(6)]],
    const device bfloat16_t* w [[buffer(7)]],
    device bfloat16_t* out [[buffer(8)]],
    device float* state_out [[buffer(9)]],
    constant GdnFusedParams& p [[buffer(10)]],
#endif
    uint n [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int n_per_t = GDN_FUSED_N_PER_T;
  constexpr int Dk = 32 * n_per_t;
  constexpr int Dv = GDN_FUSED_DV;
  constexpr uint MaxT = Dv / 4;
  // Eight-row verification needs only eight output/preparation rows.
  // Keep the original SIMD count: storage specialization must not change
  // either arithmetic schedule, including the 32-SIMD-group default.
  constexpr uint StorageRows = GDN_FUSED_T8_STORAGE ? 8 : MaxT;
  threadgroup bfloat16_t y_sh[StorageRows * Dv];
  constexpr uint SimdGroups = GDN_FUSED_SCAN8 ? 8 : MaxT;

  const uint T = uint(p.T);
  const uint Hk = uint(p.hk);
  const uint Hv = uint(p.hv);
  const uint b_idx = n / Hv;
  const uint hv_idx = n % Hv;
  const uint hk_idx = hv_idx % Hk;
#if GDN_FUSED_DEFERRED
  // Production constructs this scalar only from a checked host keep count.
  // Clamp the internal raw-FFI operand defensively, without a device readback.
  const uint P = uint(clamp(pending_keep[0], 0, p.pending_T));
#endif

#if GDN_FUSED_COMPLETE
  // --- prologue: qwen4_gdn_prepare for this head's q, k, v vectors -------
  // Prepared rows for up to Dv / 8 tokens: with y_sh this stays inside the
  // 32 KiB threadgroup budget (the host limits T accordingly).
  constexpr uint PrepT = GDN_FUSED_T8_STORAGE ? 8 : Dv / 8;
  threadgroup bfloat16_t q_sh[PrepT * Dk];
  threadgroup bfloat16_t k_sh[PrepT * Dk];
  threadgroup bfloat16_t v_sh[PrepT * Dv];
  threadgroup float g_sh[PrepT];
  threadgroup bfloat16_t beta_sh[PrepT];
  {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    constexpr uint KC = 4;
    const uint W = (2 * Hk) * Dk + Hv * Dv;
    const bool writer = hv_idx < Hk;  // one of the Hv / Hk sharers writes q/k
    for (uint task = sg; task < 3 * T; task += SimdGroups) {
      const uint token = task % T;
      const uint kind = task / T;  // 0 = q, 1 = k, 2 = v
      const uint vec = kind == 0 ? hk_idx : (kind == 1 ? Hk + hk_idx : 2 * Hk + hv_idx);
      float values[4], squared = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        const uint ch = vec * 128 + lane * 4 + i;
        float total = 0.0f;
        for (uint tap = 0; tap < KC; ++tap) {
          const uint row = token + tap;
#if GDN_FUSED_DEFERRED
          const uint prior = P + row;
          // Do not read the virtual history for current rows: its prior index
          // may be beyond the retained tape, even though the select is masked.
          float value;
          if (row < KC - 1) value = float(prior < KC - 1
              ? history[prior * uint(p.hist_stride_r) + ch * uint(p.hist_stride_c)]
              : pending_qkv[(prior - (KC - 1)) * W + ch]);
          else value = float(qkv[(row - (KC - 1)) * uint(p.qkv_stride_t) + ch * uint(p.qkv_stride_c)]);
#else
          float value = float(row < KC - 1
              ? history[row * uint(p.hist_stride_r) + ch * uint(p.hist_stride_c)]
              : qkv[(row - (KC - 1)) * uint(p.qkv_stride_t) + ch * uint(p.qkv_stride_c)]);
#endif
          float product = value * conv[ch * KC + tap];
          total = product + total;
        }
        bfloat16_t raw = bfloat16_t(total);
        bfloat16_t activated = raw * gdn_fused_sigmoid(raw);
        values[i] = float(activated);
        squared += values[i] * values[i];
        if (token == 0 && (kind == 2 || writer)) {
          for (uint row = 0; row + 1 < KC; ++row) {
#if GDN_FUSED_DEFERRED
            const uint source = P + row;
            history_out[row * W + ch] = source < KC - 1
                ? history[source * uint(p.hist_stride_r) + ch * uint(p.hist_stride_c)]
                : pending_qkv[(source - (KC - 1)) * W + ch];
#else
            const uint source = T + row;
            history_out[row * W + ch] = source < KC - 1
                ? history[source * uint(p.hist_stride_r) + ch * uint(p.hist_stride_c)]
                : qkv[(source - (KC - 1)) * uint(p.qkv_stride_t) + ch * uint(p.qkv_stride_c)];
#endif
          }
        }
      }
      if (kind < 2) {
        float divisor = metal::precise::sqrt(simd_sum(squared) + 128.0f * 1e-6f);
        for (uint i = 0; i < 4; ++i) {
          const uint d = lane * 4 + i;
          bfloat16_t normalized = bfloat16_t(values[i] / divisor);
          if (kind == 0) {
            bfloat16_t query = normalized * bfloat16_t(0.08838834764831845f);
            q_sh[token * Dk + d] = query;
            if (writer) q_out[(token * Hk + hk_idx) * Dk + d] = query;
          } else {
            k_sh[token * Dk + d] = normalized;
            if (writer) k_out[(token * Hk + hk_idx) * Dk + d] = normalized;
          }
        }
      } else {
        for (uint i = 0; i < 4; ++i) {
          const uint d = lane * 4 + i;
          bfloat16_t value = bfloat16_t(values[i]);
          v_sh[token * Dv + d] = value;
          v_out[(token * Hv + hv_idx) * Dv + d] = value;
        }
      }
    }
    if (sg == 0 && lane < T) {
      const uint token = lane;
      float ax = float(a[token * uint(p.a_stride_t) + hv_idx * uint(p.a_stride_c)]) + dt[hv_idx];
      float softplus = metal::max(ax, 0.0f) + log1p(metal::precise::exp(-metal::abs(ax)));
      float decay = metal::precise::exp(softplus * scale[hv_idx]);
      bfloat16_t beta_v = bfloat16_t(gdn_fused_sigmoid(
          float(b[token * uint(p.b_stride_t) + hv_idx * uint(p.b_stride_c)])));
      g_sh[token] = decay;
      beta_sh[token] = beta_v;
      decay_out[token * Hv + hv_idx] = decay;
      beta_out[token * Hv + hv_idx] = beta_v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup bfloat16_t* q_ = q_sh;
  const threadgroup bfloat16_t* k_ = k_sh;
  const threadgroup bfloat16_t* v_ = v_sh;
  const threadgroup float* g_ = g_sh;
  const threadgroup bfloat16_t* beta_ = beta_sh;
  constexpr uint qk_step = Dk, v_step = Dv, gate_step = 1;
  constexpr uint gate_at = 0;
#else
  const device bfloat16_t* q_ = q + (b_idx * T * Hk + hk_idx) * Dk;
  const device bfloat16_t* k_ = k + (b_idx * T * Hk + hk_idx) * Dk;
  const device bfloat16_t* v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
  const device float* g_ = g + b_idx * T * Hv;
  const device bfloat16_t* beta_ = beta + b_idx * T * Hv;
  const uint qk_step = Hk * Dk, v_step = Hv * Dv, gate_step = Hv;
  const uint gate_at = hv_idx;
#endif

#if GDN_FUSED_SCAN8
  // Splash's two-state-row schedule on eight SIMD groups. Each state row
  // retains the native recurrence's exact operation/reduction order; only
  // assignment to SIMD groups and the order of independent rows change.
  float state_A[n_per_t], state_B[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    const int s_idx = n_per_t * lane + i;
    state_A[i] = state_in[(n * Dv + sg) * Dk + s_idx];
    state_B[i] = state_in[(n * Dv + sg + 8) * Dk + s_idx];
  }
  for (uint batch = 0; batch < Dv / 16; ++batch) {
    const uint dv_A = 16 * batch + sg;
    const uint dv_B = dv_A + 8;
    const uint base_A = (n * Dv + dv_A) * Dk;
    const uint base_B = (n * Dv + dv_B) * Dk;
    float upcoming_A[n_per_t] = {}, upcoming_B[n_per_t] = {};
    if (batch + 1 < Dv / 16) {
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * lane + i;
        upcoming_A[i] = state_in[base_A + 16 * Dk + s_idx];
        upcoming_B[i] = state_in[base_B + 16 * Dk + s_idx];
      }
    }
#if GDN_FUSED_DEFERRED
    for (uint t = 0; t < P; ++t) {
      const float g_val = pending_g[t * Hv + hv_idx];
      const float beta_val = float(pending_beta[t * Hv + hv_idx]);
      float kv_mem_A = 0.0f, kv_mem_B = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * lane + i;
        state_A[i] = state_A[i] * g_val;
        state_B[i] = state_B[i] * g_val;
        const float k_val = float(pending_k[(t * Hk + hk_idx) * Dk + s_idx]);
        kv_mem_A += state_A[i] * k_val;
        kv_mem_B += state_B[i] * k_val;
      }
      kv_mem_A = simd_sum(kv_mem_A);
      kv_mem_B = simd_sum(kv_mem_B);
      const float delta_A = (float(pending_v[(t * Hv + hv_idx) * Dv + dv_A]) - kv_mem_A) * beta_val;
      const float delta_B = (float(pending_v[(t * Hv + hv_idx) * Dv + dv_B]) - kv_mem_B) * beta_val;
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * lane + i;
        const float k_val = float(pending_k[(t * Hk + hk_idx) * Dk + s_idx]);
        state_A[i] = state_A[i] + k_val * delta_A;
        state_B[i] = state_B[i] + k_val * delta_B;
      }
    }
    for (int i = 0; i < n_per_t; ++i) {
      const int s_idx = n_per_t * lane + i;
      state_out[base_A + s_idx] = state_A[i];
      state_out[base_B + s_idx] = state_B[i];
    }
#endif
    for (uint t = 0; t < T; ++t) {
      const float g_val = g_[gate_at + t * gate_step];
      const float beta_val = static_cast<float>(beta_[gate_at + t * gate_step]);
      float kv_mem_A = 0.0f, kv_mem_B = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * lane + i;
        state_A[i] = state_A[i] * g_val;
        state_B[i] = state_B[i] * g_val;
        const float k_val = static_cast<float>(k_[t * qk_step + s_idx]);
        kv_mem_A += state_A[i] * k_val;
        kv_mem_B += state_B[i] * k_val;
      }
      kv_mem_A = simd_sum(kv_mem_A);
      kv_mem_B = simd_sum(kv_mem_B);
      const float delta_A = (static_cast<float>(v_[t * v_step + dv_A]) - kv_mem_A) * beta_val;
      const float delta_B = (static_cast<float>(v_[t * v_step + dv_B]) - kv_mem_B) * beta_val;
      float out_A = 0.0f, out_B = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * lane + i;
        const float k_val = static_cast<float>(k_[t * qk_step + s_idx]);
        const float q_val = static_cast<float>(q_[t * qk_step + s_idx]);
        state_A[i] = state_A[i] + k_val * delta_A;
        state_B[i] = state_B[i] + k_val * delta_B;
        out_A += state_A[i] * q_val;
        out_B += state_B[i] * q_val;
      }
      out_A = simd_sum(out_A);
      out_B = simd_sum(out_B);
      if (lane == 0) {
        y_sh[t * Dv + dv_A] = static_cast<bfloat16_t>(out_A);
        y_sh[t * Dv + dv_B] = static_cast<bfloat16_t>(out_B);
      }
    }
    for (int i = 0; i < n_per_t; ++i) {
      const int s_idx = n_per_t * lane + i;
#if !GDN_FUSED_DEFERRED
      state_out[base_A + s_idx] = state_A[i];
      state_out[base_B + s_idx] = state_B[i];
#endif
      state_A[i] = upcoming_A[i];
      state_B[i] = upcoming_B[i];
    }
  }
#else
  const uint dv_A = 4 * sg;
  const uint dv_B = dv_A + 1;
  const uint dv_C = dv_A + 2;
  const uint dv_D = dv_A + 3;
  const device float* i_state_A = state_in + (n * Dv + dv_A) * Dk;
  const device float* i_state_B = state_in + (n * Dv + dv_B) * Dk;
  const device float* i_state_C = state_in + (n * Dv + dv_C) * Dk;
  const device float* i_state_D = state_in + (n * Dv + dv_D) * Dk;
  device float* o_state_A = state_out + (n * Dv + dv_A) * Dk;
  device float* o_state_B = state_out + (n * Dv + dv_B) * Dk;
  device float* o_state_C = state_out + (n * Dv + dv_C) * Dk;
  device float* o_state_D = state_out + (n * Dv + dv_D) * Dk;

  float state_A[n_per_t];
  float state_B[n_per_t];
  float state_C[n_per_t];
  float state_D[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    const int s_idx = n_per_t * lane + i;
    state_A[i] = i_state_A[s_idx];
    state_B[i] = i_state_B[s_idx];
    state_C[i] = i_state_C[s_idx];
    state_D[i] = i_state_D[s_idx];
  }

#if GDN_FUSED_DEFERRED
    for (uint t = 0; t < P; ++t) {
      const float g_val = pending_g[t * Hv + hv_idx];
      const float beta_val = float(pending_beta[t * Hv + hv_idx]);
      float kv_mem_A = 0.0f, kv_mem_B = 0.0f, kv_mem_C = 0.0f, kv_mem_D = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * lane + i;
        state_A[i] = state_A[i] * g_val;
        state_B[i] = state_B[i] * g_val;
        state_C[i] = state_C[i] * g_val;
        state_D[i] = state_D[i] * g_val;
        const float k_val = float(pending_k[(t * Hk + hk_idx) * Dk + s_idx]);
        kv_mem_A += state_A[i] * k_val;
        kv_mem_B += state_B[i] * k_val;
        kv_mem_C += state_C[i] * k_val;
        kv_mem_D += state_D[i] * k_val;
      }
      kv_mem_A = simd_sum(kv_mem_A);
      kv_mem_B = simd_sum(kv_mem_B);
      kv_mem_C = simd_sum(kv_mem_C);
      kv_mem_D = simd_sum(kv_mem_D);
      const float delta_A = (float(pending_v[(t * Hv + hv_idx) * Dv + dv_A]) - kv_mem_A) * beta_val;
      const float delta_B = (float(pending_v[(t * Hv + hv_idx) * Dv + dv_B]) - kv_mem_B) * beta_val;
      const float delta_C = (float(pending_v[(t * Hv + hv_idx) * Dv + dv_C]) - kv_mem_C) * beta_val;
      const float delta_D = (float(pending_v[(t * Hv + hv_idx) * Dv + dv_D]) - kv_mem_D) * beta_val;
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * lane + i;
        const float k_val = float(pending_k[(t * Hk + hk_idx) * Dk + s_idx]);
        state_A[i] = state_A[i] + k_val * delta_A;
        state_B[i] = state_B[i] + k_val * delta_B;
        state_C[i] = state_C[i] + k_val * delta_C;
        state_D[i] = state_D[i] + k_val * delta_D;
      }
    }
    for (int i = 0; i < n_per_t; ++i) {
      const int s_idx = n_per_t * lane + i;
      o_state_A[s_idx] = state_A[i];
      o_state_B[s_idx] = state_B[i];
      o_state_C[s_idx] = state_C[i];
      o_state_D[s_idx] = state_D[i];
    }
#endif
  for (uint t = 0; t < T; ++t) {
    const float g_val = g_[gate_at];
    const float beta_val = static_cast<float>(beta_[gate_at]);

    float kv_mem_A = 0.0f, kv_mem_B = 0.0f, kv_mem_C = 0.0f, kv_mem_D = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      const int s_idx = n_per_t * lane + i;
      state_A[i] = state_A[i] * g_val;
      state_B[i] = state_B[i] * g_val;
      state_C[i] = state_C[i] * g_val;
      state_D[i] = state_D[i] * g_val;
      const float k_val = static_cast<float>(k_[s_idx]);
      kv_mem_A += state_A[i] * k_val;
      kv_mem_B += state_B[i] * k_val;
      kv_mem_C += state_C[i] * k_val;
      kv_mem_D += state_D[i] * k_val;
    }
    kv_mem_A = simd_sum(kv_mem_A);
    kv_mem_B = simd_sum(kv_mem_B);
    kv_mem_C = simd_sum(kv_mem_C);
    kv_mem_D = simd_sum(kv_mem_D);

    const float delta_A = (static_cast<float>(v_[dv_A]) - kv_mem_A) * beta_val;
    const float delta_B = (static_cast<float>(v_[dv_B]) - kv_mem_B) * beta_val;
    const float delta_C = (static_cast<float>(v_[dv_C]) - kv_mem_C) * beta_val;
    const float delta_D = (static_cast<float>(v_[dv_D]) - kv_mem_D) * beta_val;

    float out_A = 0.0f, out_B = 0.0f, out_C = 0.0f, out_D = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      const int s_idx = n_per_t * lane + i;
      const float k_val = static_cast<float>(k_[s_idx]);
      const float q_val = static_cast<float>(q_[s_idx]);
      state_A[i] = state_A[i] + k_val * delta_A;
      state_B[i] = state_B[i] + k_val * delta_B;
      state_C[i] = state_C[i] + k_val * delta_C;
      state_D[i] = state_D[i] + k_val * delta_D;
      out_A += state_A[i] * q_val;
      out_B += state_B[i] * q_val;
      out_C += state_C[i] * q_val;
      out_D += state_D[i] * q_val;
    }
    out_A = simd_sum(out_A);
    out_B = simd_sum(out_B);
    out_C = simd_sum(out_C);
    out_D = simd_sum(out_D);
    if (lane == 0) {
      threadgroup bfloat16_t* y_row = y_sh + t * Dv;
      y_row[dv_A] = static_cast<bfloat16_t>(out_A);
      y_row[dv_B] = static_cast<bfloat16_t>(out_B);
      y_row[dv_C] = static_cast<bfloat16_t>(out_C);
      y_row[dv_D] = static_cast<bfloat16_t>(out_D);
    }
    q_ += qk_step;
    k_ += qk_step;
    v_ += v_step;
    g_ += gate_step;
    beta_ += gate_step;
  }
  for (int i = 0; i < n_per_t; ++i) {
    const int s_idx = n_per_t * lane + i;
#if !GDN_FUSED_DEFERRED
    o_state_A[s_idx] = state_A[i];
    o_state_B[s_idx] = state_B[i];
    o_state_C[s_idx] = state_C[i];
    o_state_D[s_idx] = state_D[i];
#endif
  }
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One simdgroup per window row: rms_single_row's lane map (lane owns
  // dims 4 * lane .. 4 * lane + 3) and its two-level simd_sum, then the
  // compiled z gate with bf16 intermediates.
  if (sg < T) {
    const uint t = sg;
    const threadgroup bfloat16_t* y_row = y_sh + t * Dv + lane * 4;
    float thread_x[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
      thread_x[i] = y_row[i];
      acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    acc = simd_sum(lane == 0 ? acc : 0.0f);
    const float inv = metal::precise::rsqrt(acc / float(Dv) + p.eps);
    const device bfloat16_t* z_ = z + b_idx * uint(p.z_stride_b) +
        t * uint(p.z_stride_t) + hv_idx * Dv + lane * 4;
    const device bfloat16_t* w_ = w + lane * 4;
    device bfloat16_t* out_ = out + ((b_idx * T + t) * Hv + hv_idx) * Dv + lane * 4;
    bfloat16_t rounded[4];
    for (int i = 0; i < 4; ++i) {
      const bfloat16_t normed = w_[i] * static_cast<bfloat16_t>(thread_x[i] * inv);
      const bfloat16_t zt = z_[i];
      const bfloat16_t sig = gdn_fused_sigmoid(zt);
      const bfloat16_t gated = sig * zt;
      const bfloat16_t value = gated * normed;
      out_[i] = value;
      rounded[i] = value;
    }
#if GDN_FUSED_PREPARED
    // Exact kquant_qmv_sg8_prep<bf16,32> bytes, emitted from the same
    // rounded output values. Every scalar half has one writer; adjacent
    // rows write different halves of a uint32 pair, never read/modify/write.
    const uint span = (hv_idx * Dv + lane * 4) / 32;
    const uint fm = lane & 7u, fn = t & ~1u;
    const uint packed_lane = ((fm & 4u) << 2) | ((fm & 3u) << 1) |
                             ((fn & 4u) << 1) | ((fn & 2u) >> 1);
    for (uint i = 0; i < 4; ++i)
      prepared_bt[((span * 32 + packed_lane) * 4 + i) * 2 + (t & 1u)] = rounded[i];
    // The original prep's lane bits 1,2,4 represent fm bits 0,1,2.
    // This producer's adjacent lanes own adjacent fm, so xor 1,2,4
    // reproduces the exact same FP32 reduction tree over each 32 values.
    float sum = (float(rounded[0]) + float(rounded[1])) +
                (float(rounded[2]) + float(rounded[3]));
    sum += simd_shuffle_xor(sum, ushort(1));
    sum += simd_shuffle_xor(sum, ushort(2));
    sum += simd_shuffle_xor(sum, ushort(4));
    if ((lane & 7u) == 0) prepared_sums[span * 8 + t] = sum;
#endif
  }
}
)"
