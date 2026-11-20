// Fused GDN prefill in three dispatches with `mlx_node_gdn_fused_prefill`'s
// arithmetic (gdn_fused_prefill.metal.inc): every expression, lane map,
// simd_sum and bf16 rounding point below is that kernel's, so the output and
// state bits are its bits. Changing any of them changes the model's bits.
//   prologue  one simdgroup per (token, key head): q/k rms norm -> qn, kn;
//             one thread per (token, value head): gating -> g, beta
//   scan      16 state rows of one value head per 128-thread threadgroup,
//             16-token blocks of qn/kn/v/g/beta staged in threadgroup memory;
//             y (bf16) into `out`
//   tail      one simdgroup per (token, value head): gated norm + z gate,
//             in place on `out`
// Dk = Dv = 128.
R"(
struct GdnPrefillParams {
  int T;
  int hk;
  int hv;
  int tiled;
  int has_state;
  int q_stride_t;
  int k_stride_t;
  int v_stride_t;
  int a_stride_t;
  int b_stride_t;
  int z_stride_t;
  float eps;
  float qk_eps;
};

template <typename T>
inline T gdn_prefill_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

static constant constexpr const uint kGdnDk = 128;
static constant constexpr const uint kGdnDv = 128;
static constant constexpr const int kGdnPerLane = 4;
static constant constexpr const uint kGdnScanRows = 16;
static constant constexpr const uint kGdnScanThreads = 8 * kGdnScanRows;
static constant constexpr const uint kGdnScanBlock = 16;

// ScalarT = the storage of a_log / dt_bias (f32, or bf16 as a GGUF load
// keeps them); each value widens to float exactly as fused_gdn_gating reads it.
template <typename ScalarT>
inline void gdn_prefill_prologue_body(
    const device bfloat16_t* q,
    const device bfloat16_t* k,
    const device bfloat16_t* a,
    const device bfloat16_t* b,
    const device ScalarT* a_log,
    const device ScalarT* dt_bias,
    const device bfloat16_t* wq,
    const device bfloat16_t* wk,
    device bfloat16_t* qn,
    device bfloat16_t* kn,
    device float* g,
    device bfloat16_t* beta,
    constant GdnPrefillParams& p,
    uint gid,
    uint lane) {
  const uint T = uint(p.T);
  const uint Hk = uint(p.hk);
  const uint Hv = uint(p.hv);
  const uint task = gid / 32;
  if (task < T * Hk) {
    const uint t = task / Hk;
    const uint kh = task % Hk;
    const device bfloat16_t* q_row = q + t * uint(p.q_stride_t) + kh * kGdnDk;
    const device bfloat16_t* k_row = k + t * uint(p.k_stride_t) + kh * kGdnDk;
    const device bfloat16_t* wq_ = wq + lane * 4;
    const device bfloat16_t* wk_ = wk + lane * 4;
    float qx[kGdnPerLane], kx[kGdnPerLane];
    float acc_q = 0.0f, acc_k = 0.0f;
    for (int i = 0; i < kGdnPerLane; ++i) {
      const uint s_idx = kGdnPerLane * lane + i;
      qx[i] = static_cast<float>(q_row[s_idx]);
      kx[i] = static_cast<float>(k_row[s_idx]);
      acc_q += qx[i] * qx[i];
      acc_k += kx[i] * kx[i];
    }
    acc_q = simd_sum(acc_q);
    acc_k = simd_sum(acc_k);
    acc_q = simd_sum(lane == 0 ? acc_q : 0.0f);
    acc_k = simd_sum(lane == 0 ? acc_k : 0.0f);
    const float inv_q = metal::precise::rsqrt(acc_q / float(kGdnDk) + p.qk_eps);
    const float inv_k = metal::precise::rsqrt(acc_k / float(kGdnDk) + p.qk_eps);
    device bfloat16_t* qn_row = qn + (t * Hk + kh) * kGdnDk;
    device bfloat16_t* kn_row = kn + (t * Hk + kh) * kGdnDk;
    for (int i = 0; i < kGdnPerLane; ++i) {
      const uint s_idx = kGdnPerLane * lane + i;
      qn_row[s_idx] = wq_[i] * static_cast<bfloat16_t>(qx[i] * inv_q);
      kn_row[s_idx] = wk_[i] * static_cast<bfloat16_t>(kx[i] * inv_k);
    }
  }
  if (gid < T * Hv) {
    const uint t = gid / Hv;
    const uint hv_idx = gid % Hv;
    float b_val = static_cast<float>(b[t * uint(p.b_stride_t) + hv_idx]);
    float beta_val = 1.0f / (1.0f + exp(-b_val));
    float a_val = static_cast<float>(a[t * uint(p.a_stride_t) + hv_idx]);
    float a_log_val = static_cast<float>(a_log[hv_idx]);
    float dt_bias_val = static_cast<float>(dt_bias[hv_idx]);
    float sp_input = a_val + dt_bias_val;
    float softplus_val =
        (sp_input > 20.0f) ? sp_input : log(1.0f + exp(sp_input));
    g[gid] = exp(-exp(a_log_val) * softplus_val);
    beta[gid] = static_cast<bfloat16_t>(beta_val);
  }
}

#define GDN_PREFILL_PROLOGUE(name, ScalarT)                              \
  [[kernel]] void name(                                                  \
      const device bfloat16_t* q [[buffer(0)]],                          \
      const device bfloat16_t* k [[buffer(1)]],                          \
      const device bfloat16_t* a [[buffer(2)]],                          \
      const device bfloat16_t* b [[buffer(3)]],                          \
      const device ScalarT* a_log [[buffer(4)]],                         \
      const device ScalarT* dt_bias [[buffer(5)]],                       \
      const device bfloat16_t* wq [[buffer(6)]],                         \
      const device bfloat16_t* wk [[buffer(7)]],                         \
      device bfloat16_t* qn [[buffer(8)]],                               \
      device bfloat16_t* kn [[buffer(9)]],                               \
      device float* g [[buffer(10)]],                                    \
      device bfloat16_t* beta [[buffer(11)]],                            \
      constant GdnPrefillParams& p [[buffer(12)]],                       \
      uint gid [[thread_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                         \
    gdn_prefill_prologue_body<ScalarT>(                                  \
        q, k, a, b, a_log, dt_bias, wq, wk, qn, kn, g, beta, p, gid,     \
        lane);                                                           \
  }

GDN_PREFILL_PROLOGUE(mlx_node_gdn_prefill_prologue, float)
GDN_PREFILL_PROLOGUE(mlx_node_gdn_prefill_prologue_bf16_scalars, bfloat16_t)

[[kernel]] void mlx_node_gdn_prefill_scan(
    const device bfloat16_t* qn [[buffer(0)]],
    const device bfloat16_t* kn [[buffer(1)]],
    const device bfloat16_t* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device bfloat16_t* beta [[buffer(4)]],
    const device float* state_in [[buffer(5)]],
    device bfloat16_t* out [[buffer(6)]],
    device float* state_out [[buffer(7)]],
    constant GdnPrefillParams& p [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint Rows = kGdnScanRows;
  constexpr uint Threads = kGdnScanThreads;
  constexpr uint Block = kGdnScanBlock;
  constexpr uint QK4 = Block * kGdnDk / (4 * Threads);
  constexpr uint GroupsPerHead = kGdnDv / Rows;
  static_assert(QK4 * 4 * Threads == Block * kGdnDk, "q/k block per thread");
  static_assert(Block * Rows <= 4 * Threads, "value block per thread");
  threadgroup bfloat16_t k_sh[Block * kGdnDk];
  threadgroup bfloat16_t q_sh[Block * kGdnDk];
  threadgroup float v_sh[Block * Rows];
  threadgroup float g_sh[Block];
  threadgroup float beta_sh[Block];

  const uint T = uint(p.T);
  const uint Hk = uint(p.hk);
  const uint Hv = uint(p.hv);
  const uint hv_idx = group / GroupsPerHead;
  const uint row_base = (group % GroupsPerHead) * Rows;
  const uint hk_idx = p.tiled ? hv_idx % Hk : hv_idx / (Hv / Hk);

  const uint dv_A = row_base + 4 * sg;
  const uint dv_B = dv_A + 1;
  const uint dv_C = dv_A + 2;
  const uint dv_D = dv_A + 3;
  const device float* i_state_A = state_in + (hv_idx * kGdnDv + dv_A) * kGdnDk;
  const device float* i_state_B = state_in + (hv_idx * kGdnDv + dv_B) * kGdnDk;
  const device float* i_state_C = state_in + (hv_idx * kGdnDv + dv_C) * kGdnDk;
  const device float* i_state_D = state_in + (hv_idx * kGdnDv + dv_D) * kGdnDk;
  device float* o_state_A = state_out + (hv_idx * kGdnDv + dv_A) * kGdnDk;
  device float* o_state_B = state_out + (hv_idx * kGdnDv + dv_B) * kGdnDk;
  device float* o_state_C = state_out + (hv_idx * kGdnDv + dv_C) * kGdnDk;
  device float* o_state_D = state_out + (hv_idx * kGdnDv + dv_D) * kGdnDk;

  float state_A[kGdnPerLane];
  float state_B[kGdnPerLane];
  float state_C[kGdnPerLane];
  float state_D[kGdnPerLane];
  for (int i = 0; i < kGdnPerLane; ++i) {
    const int s_idx = kGdnPerLane * lane + i;
    state_A[i] = p.has_state ? i_state_A[s_idx] : 0.0f;
    state_B[i] = p.has_state ? i_state_B[s_idx] : 0.0f;
    state_C[i] = p.has_state ? i_state_C[s_idx] : 0.0f;
    state_D[i] = p.has_state ? i_state_D[s_idx] : 0.0f;
  }

  bfloat16_t k_fetch[QK4][4];
  bfloat16_t q_fetch[QK4][4];
  bfloat16_t v_fetch[4];
  float g_fetch = 0.0f;
  bfloat16_t beta_fetch = static_cast<bfloat16_t>(0.0f);
  for (uint n = 0; n < QK4; ++n) {
    for (uint j = 0; j < 4; ++j) {
      k_fetch[n][j] = static_cast<bfloat16_t>(0.0f);
      q_fetch[n][j] = static_cast<bfloat16_t>(0.0f);
    }
  }
  for (uint j = 0; j < 4; ++j) {
    v_fetch[j] = static_cast<bfloat16_t>(0.0f);
  }
  auto fetch = [&](uint start) {
    for (uint n = 0; n < QK4; ++n) {
      const uint e = (n * Threads + tid) * 4;
      const uint token = start + e / kGdnDk;
      if (token < T) {
        const uint base = (token * Hk + hk_idx) * kGdnDk + e % kGdnDk;
        for (uint j = 0; j < 4; ++j) {
          k_fetch[n][j] = kn[base + j];
          q_fetch[n][j] = qn[base + j];
        }
      }
    }
    if (tid * 4 < Block * Rows) {
      const uint e = tid * 4;
      const uint token = start + e / Rows;
      if (token < T) {
        const uint base =
            token * uint(p.v_stride_t) + hv_idx * kGdnDv + row_base + e % Rows;
        for (uint j = 0; j < 4; ++j) {
          v_fetch[j] = v[base + j];
        }
      }
    }
    if (tid < Block && start + tid < T) {
      g_fetch = g[(start + tid) * Hv + hv_idx];
      beta_fetch = beta[(start + tid) * Hv + hv_idx];
    }
  };
  auto stage = [&]() {
    for (uint n = 0; n < QK4; ++n) {
      const uint e = (n * Threads + tid) * 4;
      for (uint j = 0; j < 4; ++j) {
        k_sh[e + j] = k_fetch[n][j];
        q_sh[e + j] = q_fetch[n][j];
      }
    }
    if (tid * 4 < Block * Rows) {
      for (uint j = 0; j < 4; ++j) {
        v_sh[tid * 4 + j] = static_cast<float>(v_fetch[j]);
      }
    }
    if (tid < Block) {
      g_sh[tid] = g_fetch;
      beta_sh[tid] = static_cast<float>(beta_fetch);
    }
  };

  fetch(0);
  stage();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint start = 0; start < T; start += Block) {
    const bool more = start + Block < T;
    if (more) {
      fetch(start + Block);
    }
    const uint count = min(Block, T - start);
    for (uint t = 0; t < count; ++t) {
      const float g_val = g_sh[t];
      const float beta_val = beta_sh[t];
      const threadgroup bfloat16_t* k_ = k_sh + t * kGdnDk;
      const threadgroup bfloat16_t* q_ = q_sh + t * kGdnDk;
      const threadgroup float* v_row = v_sh + t * Rows + 4 * sg;

      float kv_mem_A = 0.0f, kv_mem_B = 0.0f, kv_mem_C = 0.0f, kv_mem_D = 0.0f;
      for (int i = 0; i < kGdnPerLane; ++i) {
        const int s_idx = kGdnPerLane * lane + i;
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

      const float delta_A = (v_row[0] - kv_mem_A) * beta_val;
      const float delta_B = (v_row[1] - kv_mem_B) * beta_val;
      const float delta_C = (v_row[2] - kv_mem_C) * beta_val;
      const float delta_D = (v_row[3] - kv_mem_D) * beta_val;

      float out_A = 0.0f, out_B = 0.0f, out_C = 0.0f, out_D = 0.0f;
      for (int i = 0; i < kGdnPerLane; ++i) {
        const int s_idx = kGdnPerLane * lane + i;
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
        device bfloat16_t* y_row = out + ((start + t) * Hv + hv_idx) * kGdnDv;
        y_row[dv_A] = static_cast<bfloat16_t>(out_A);
        y_row[dv_B] = static_cast<bfloat16_t>(out_B);
        y_row[dv_C] = static_cast<bfloat16_t>(out_C);
        y_row[dv_D] = static_cast<bfloat16_t>(out_D);
      }
    }
    if (more) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      stage();
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  for (int i = 0; i < kGdnPerLane; ++i) {
    const int s_idx = kGdnPerLane * lane + i;
    o_state_A[s_idx] = state_A[i];
    o_state_B[s_idx] = state_B[i];
    o_state_C[s_idx] = state_C[i];
    o_state_D[s_idx] = state_D[i];
  }
}

[[kernel]] void mlx_node_gdn_prefill_tail(
    device bfloat16_t* out [[buffer(0)]],
    const device bfloat16_t* z [[buffer(1)]],
    const device bfloat16_t* w [[buffer(2)]],
    constant GdnPrefillParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint T = uint(p.T);
  const uint Hv = uint(p.hv);
  const uint task = gid / 32;
  if (task >= T * Hv) {
    return;
  }
  const uint t = task / Hv;
  const uint hv_idx = task % Hv;
  device bfloat16_t* row = out + (t * Hv + hv_idx) * kGdnDv + lane * 4;
  float thread_x[4];
  float acc = 0.0f;
  for (int i = 0; i < 4; ++i) {
    thread_x[i] = row[i];
    acc += thread_x[i] * thread_x[i];
  }
  acc = simd_sum(acc);
  acc = simd_sum(lane == 0 ? acc : 0.0f);
  const float inv = metal::precise::rsqrt(acc / float(kGdnDv) + p.eps);
  const device bfloat16_t* z_ =
      z + t * uint(p.z_stride_t) + hv_idx * kGdnDv + lane * 4;
  const device bfloat16_t* w_ = w + lane * 4;
  for (int i = 0; i < 4; ++i) {
    const bfloat16_t normed =
        w_[i] * static_cast<bfloat16_t>(thread_x[i] * inv);
    const bfloat16_t zt = z_[i];
    const bfloat16_t sig = gdn_prefill_sigmoid(zt);
    const bfloat16_t gated = sig * zt;
    row[i] = gated * normed;
  }
}
)"
