// Chunked fused GDN prefill: ONE dispatch for a prompt window of any T.
// One (batch=1, value head) threadgroup walks the window in 32-token chunks.
// Per chunk: (a) a PARALLEL prologue — one simdgroup per row applies the
// rms_single_row q/k L2 normalization (lane*4 map, two-level simd_sum,
// `w[i] * bf16(x[i] * inv)` — the `prep_via_conv` tail bit-for-bit) and one
// lane per row computes the fused_gdn_gating decay/beta VERBATIM
// (softplus(x) = x > 20 ? x : log(1 + exp(x)), beta = 1/(1+exp(-b)), plain
// metal exp/log, a_log and dt_bias read as `float*` like that kernel, beta
// rounded to bf16 before the f32 widen) into threadgroup memory; (b) the
// unchanged 4-vcol per-step recurrence over the prepared rows, the f32
// register state carrying across chunk boundaries; (c) the rms_single_row +
// compiled-z-gate tail of `mlx_node_gdn_fused_step` on the chunk's y rows —
// so the gated output is bit-identical to the unfused prefill chain.
// Inputs are the PRE-NORM conv outputs read through their row strides (the
// split views of the conv/SiLU result — the copies `ensure_row_contiguous`
// would insert never happen). The non-tiled (standard) checkpoint maps
// value head hv to key head hv / (Hv / Hk), reproducing the host-side
// repeat expansion with no expanded buffers; `tiled` keeps the tiled-GGUF
// hv % Hk map. `state_in` may be a dummy when `has_state` is 0: the register
// state then starts at 0, matching a f32 zeros input exactly.
// Threadgroups: Hv of 8 * Dv threads (Dv / 4 simdgroups).
R"(
struct GdnPrefillParams {
  int T;            // window rows (arbitrary; chunked below)
  int hk;           // key heads
  int hv;           // value heads
  int tiled;        // 1: hk = hv % Hk (tiled GGUF); 0: hk = hv / (Hv / Hk)
  int has_state;    // 0: zero-init the register state
  int q_stride_t;   // element stride between q rows
  int k_stride_t;
  int v_stride_t;
  int a_stride_t;
  int b_stride_t;
  int z_stride_t;
  float eps;        // gated-norm epsilon (tail)
  float qk_eps;     // q/k rms_norm epsilon (prologue)
};

// `Sigmoid` from mlx/backend/metal/kernels/unary_ops.h, verbatim — the
// compiled z gate and the bf16 beta boundary both evaluate this.
template <typename T>
inline T gdn_prefill_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

[[kernel]] void mlx_node_gdn_fused_prefill(
    const device bfloat16_t* q [[buffer(0)]],
    const device bfloat16_t* k [[buffer(1)]],
    const device bfloat16_t* v [[buffer(2)]],
    const device bfloat16_t* a [[buffer(3)]],
    const device bfloat16_t* b [[buffer(4)]],
    const device float* a_log [[buffer(5)]],
    const device float* dt_bias [[buffer(6)]],
    const device float* state_in [[buffer(7)]],
    const device bfloat16_t* wq [[buffer(8)]],
    const device bfloat16_t* wk [[buffer(9)]],
    const device bfloat16_t* z [[buffer(10)]],
    const device bfloat16_t* w [[buffer(11)]],
    device bfloat16_t* out [[buffer(12)]],
    device float* state_out [[buffer(13)]],
    constant GdnPrefillParams& p [[buffer(14)]],
    uint n [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int n_per_t = GDN_PREFILL_N_PER_T;
  constexpr int Dk = 32 * n_per_t;
  constexpr int Dv = GDN_PREFILL_DV;
  constexpr uint Chunk = Dv / 4;
  threadgroup bfloat16_t y_sh[Chunk * Dv];
  threadgroup bfloat16_t q_sh[Chunk * Dk];
  threadgroup bfloat16_t k_sh[Chunk * Dk];
  threadgroup float g_sh[Chunk];
  threadgroup bfloat16_t beta_sh[Chunk];

  const uint T = uint(p.T);
  const uint Hk = uint(p.hk);
  const uint Hv = uint(p.hv);
  const uint hv_idx = n;
  const uint hk_idx =
      p.tiled ? hv_idx % Hk : hv_idx / (Hv / Hk);

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
    state_A[i] = p.has_state ? i_state_A[s_idx] : 0.0f;
    state_B[i] = p.has_state ? i_state_B[s_idx] : 0.0f;
    state_C[i] = p.has_state ? i_state_C[s_idx] : 0.0f;
    state_D[i] = p.has_state ? i_state_D[s_idx] : 0.0f;
  }

  const device bfloat16_t* wq_ = wq + lane * 4;
  const device bfloat16_t* wk_ = wk + lane * 4;

  for (uint t0 = 0; t0 < T; t0 += Chunk) {
    const uint rows = min(T - t0, Chunk);
    // --- parallel prologue: rms_single_row q/k norms + fused gating -----
    // Simdgroup `task` normalizes token task's q and k rows into q_sh/k_sh
    // (bf16 — the fast::rms_norm output dtype the recurrence then reads);
    // sg 0's lanes fill this head's decay/beta rows.
    {
      const device bfloat16_t* q_row = q + t0 * uint(p.q_stride_t) + hk_idx * Dk;
      const device bfloat16_t* k_row = k + t0 * uint(p.k_stride_t) + hk_idx * Dk;
      for (uint task = sg; task < rows; task += Dv / 4) {
        // thread_x lane map + two-level simd_sum of rms_norm.metal; the
        // `lane == 0 ? acc : 0` second sum is an exact identity like the
        // source's local_sums round-trip (one simdgroup per row there too).
        float qx[n_per_t], kx[n_per_t];
        float acc_q = 0.0f, acc_k = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          const uint s_idx = n_per_t * lane + i;
          qx[i] = static_cast<float>(q_row[task * uint(p.q_stride_t) + s_idx]);
          kx[i] = static_cast<float>(k_row[task * uint(p.k_stride_t) + s_idx]);
          acc_q += qx[i] * qx[i];
          acc_k += kx[i] * kx[i];
        }
        acc_q = simd_sum(acc_q);
        acc_k = simd_sum(acc_k);
        acc_q = simd_sum(lane == 0 ? acc_q : 0.0f);
        acc_k = simd_sum(lane == 0 ? acc_k : 0.0f);
        const float inv_q =
            metal::precise::rsqrt(acc_q / float(Dk) + p.qk_eps);
        const float inv_k =
            metal::precise::rsqrt(acc_k / float(Dk) + p.qk_eps);
        for (int i = 0; i < n_per_t; ++i) {
          const uint s_idx = n_per_t * lane + i;
          q_sh[task * Dk + s_idx] =
              wq_[i] * static_cast<bfloat16_t>(qx[i] * inv_q);
          k_sh[task * Dk + s_idx] =
              wk_[i] * static_cast<bfloat16_t>(kx[i] * inv_k);
        }
      }
      if (sg == 0 && lane < rows) {
        // fused_gdn_gating for (token t0 + lane, head hv), verbatim.
        float b_val = static_cast<float>(
            b[(t0 + lane) * uint(p.b_stride_t) + hv_idx]);
        float beta_val = 1.0f / (1.0f + exp(-b_val));
        float a_val = static_cast<float>(
            a[(t0 + lane) * uint(p.a_stride_t) + hv_idx]);
        float a_log_val = a_log[hv_idx];
        float dt_bias_val = dt_bias[hv_idx];
        float sp_input = a_val + dt_bias_val;
        float softplus_val =
            (sp_input > 20.0f) ? sp_input : log(1.0f + exp(sp_input));
        g_sh[lane] = exp(-exp(a_log_val) * softplus_val);
        beta_sh[lane] = static_cast<bfloat16_t>(beta_val);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- the 4-vcol per-step recurrence over the prepared rows ----------
    const threadgroup bfloat16_t* q_ = q_sh;
    const threadgroup bfloat16_t* k_ = k_sh;
    const device bfloat16_t* v_row = v + t0 * uint(p.v_stride_t) + hv_idx * Dv;
    const threadgroup float* g_ = g_sh;
    const threadgroup bfloat16_t* beta_ = beta_sh;
    for (uint t = 0; t < rows; ++t) {
      const float g_val = g_[0];
      const float beta_val = static_cast<float>(beta_[0]);

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

      const float delta_A =
          (static_cast<float>(v_row[dv_A]) - kv_mem_A) * beta_val;
      const float delta_B =
          (static_cast<float>(v_row[dv_B]) - kv_mem_B) * beta_val;
      const float delta_C =
          (static_cast<float>(v_row[dv_C]) - kv_mem_C) * beta_val;
      const float delta_D =
          (static_cast<float>(v_row[dv_D]) - kv_mem_D) * beta_val;

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
      q_ += Dk;
      k_ += Dk;
      v_row += uint(p.v_stride_t);
      g_ += 1;
      beta_ += 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One simdgroup per chunk row: rms_single_row's lane map and two-level
    // simd_sum, then the compiled z gate with bf16 intermediates — the
    // `gdn_fused_step` tail verbatim, `out` row-indexed by the token.
    if (sg < rows) {
      const uint t = t0 + sg;
      const threadgroup bfloat16_t* y_row = y_sh + sg * Dv + lane * 4;
      float thread_x[4];
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        thread_x[i] = y_row[i];
        acc += thread_x[i] * thread_x[i];
      }
      acc = simd_sum(acc);
      acc = simd_sum(lane == 0 ? acc : 0.0f);
      const float inv = metal::precise::rsqrt(acc / float(Dv) + p.eps);
      const device bfloat16_t* z_ =
          z + t * uint(p.z_stride_t) + hv_idx * Dv + lane * 4;
      const device bfloat16_t* w_ = w + lane * 4;
      device bfloat16_t* out_ = out + (t * Hv + hv_idx) * Dv + lane * 4;
      for (int i = 0; i < 4; ++i) {
        const bfloat16_t normed =
            w_[i] * static_cast<bfloat16_t>(thread_x[i] * inv);
        const bfloat16_t zt = z_[i];
        const bfloat16_t sig = gdn_prefill_sigmoid(zt);
        const bfloat16_t gated = sig * zt;
        out_[i] = gated * normed;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int i = 0; i < n_per_t; ++i) {
    const int s_idx = n_per_t * lane + i;
    o_state_A[s_idx] = state_A[i];
    o_state_B[s_idx] = state_B[i];
    o_state_C[s_idx] = state_C[i];
    o_state_D[s_idx] = state_D[i];
  }
}
)"
