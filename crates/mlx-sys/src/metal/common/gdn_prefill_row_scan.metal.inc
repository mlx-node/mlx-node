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

struct GdnPrefillConvParams {
  int T;
  int hk;
  int hv;
  int has_history;
  int qkv_stride_t;
  int hist_stride_r;
  int a_stride_t;
  int b_stride_t;
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

// Fused 4-tap depthwise conv1d + SiLU + q/k RMSNorm + decay/beta gates +
// conv_state carry. Preserves `conv1d_silu_k4.metal.inc` and
// `mlx_node_gdn_prefill_prologue` arithmetic bit for bit.
template <typename ScalarT>
inline void gdn_prefill_prologue_conv_body(
    const device bfloat16_t* qkv,
    const device float* conv_w,
    const device bfloat16_t* history,
    const device bfloat16_t* a,
    const device bfloat16_t* b,
    const device ScalarT* a_log,
    const device ScalarT* dt_bias,
    const device bfloat16_t* wq,
    const device bfloat16_t* wk,
    device bfloat16_t* qn,
    device bfloat16_t* kn,
    device bfloat16_t* v_out,
    device float* g,
    device bfloat16_t* beta,
    device bfloat16_t* history_out,
    constant GdnPrefillConvParams& p,
    uint gid,
    uint lane) {
  const uint T = uint(p.T);
  const uint Hk = uint(p.hk);
  const uint Hv = uint(p.hv);
  const uint num_vecs = 2 * Hk + Hv;
  const uint W = num_vecs * kGdnDk;
  const uint task = gid / 32;
  if (task < num_vecs * T) {
    // `vec` outer, `t` inner so adjacent simdgroups in a threadgroup share the
    // same 4-tap filter weights in L1 across consecutive tokens.
    const uint vec = task / T;
    const uint t = task % T;
    bfloat16_t acts[kGdnPerLane];
    for (int i = 0; i < kGdnPerLane; ++i) {
      const uint s_idx = kGdnPerLane * lane + i;
      const uint ch = vec * kGdnDk + s_idx;
      float total = 0.0f;
      {
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        for (uint tap = 0; tap < 4u; ++tap) {
          const uint row = t + tap;
          const bfloat16_t tap_val = (row < 3u)
              ? (p.has_history
                     ? history[ulong(row) * ulong(uint(p.hist_stride_r)) +
                               ulong(ch)]
                     : static_cast<bfloat16_t>(0.0f))
              : qkv[ulong(row - 3u) * ulong(uint(p.qkv_stride_t)) + ulong(ch)];
          const float product = static_cast<float>(tap_val) *
              conv_w[ulong(ch) * ulong(4) + ulong(tap)];
          total = product + total;
        }
      }
      const bfloat16_t raw = static_cast<bfloat16_t>(total);
      acts[i] = raw * gdn_prefill_sigmoid(raw);
      if (t == 0u) {
        for (uint row = 0; row < 3u; ++row) {
          const uint source = T + row;
          history_out[ulong(row) * ulong(W) + ulong(ch)] = (source < 3u)
              ? (p.has_history
                     ? history[ulong(source) * ulong(uint(p.hist_stride_r)) +
                               ulong(ch)]
                     : static_cast<bfloat16_t>(0.0f))
              : qkv[ulong(source - 3u) * ulong(uint(p.qkv_stride_t)) +
                    ulong(ch)];
        }
      }
    }
    if (vec < 2 * Hk) {
      float fx[kGdnPerLane];
      float acc = 0.0f;
      for (int i = 0; i < kGdnPerLane; ++i) {
        fx[i] = static_cast<float>(acts[i]);
        acc += fx[i] * fx[i];
      }
      acc = simd_sum(acc);
      acc = simd_sum(lane == 0 ? acc : 0.0f);
      const float inv = metal::precise::rsqrt(acc / float(kGdnDk) + p.qk_eps);
      const bool is_q = vec < Hk;
      const device bfloat16_t* wt = (is_q ? wq : wk) + lane * 4;
      const uint kh = is_q ? vec : (vec - Hk);
      device bfloat16_t* dst = (is_q ? qn : kn) +
          (ulong(t) * ulong(Hk) + ulong(kh)) * ulong(kGdnDk);
      for (int i = 0; i < kGdnPerLane; ++i) {
        const uint s_idx = kGdnPerLane * lane + i;
        dst[s_idx] = wt[i] * static_cast<bfloat16_t>(fx[i] * inv);
      }
    } else {
      const uint vh = vec - 2 * Hk;
      device bfloat16_t* dst =
          v_out + (ulong(t) * ulong(Hv) + ulong(vh)) * ulong(kGdnDv);
      for (int i = 0; i < kGdnPerLane; ++i) {
        const uint s_idx = kGdnPerLane * lane + i;
        dst[s_idx] = acts[i];
      }
    }
  }
  if (gid < T * Hv) {
    const uint t = gid / Hv;
    const uint hv_idx = gid % Hv;
    float b_val = static_cast<float>(
        b[ulong(t) * ulong(uint(p.b_stride_t)) + ulong(hv_idx)]);
    float beta_val = 1.0f / (1.0f + exp(-b_val));
    float a_val = static_cast<float>(
        a[ulong(t) * ulong(uint(p.a_stride_t)) + ulong(hv_idx)]);
    float a_log_val = static_cast<float>(a_log[hv_idx]);
    float dt_bias_val = static_cast<float>(dt_bias[hv_idx]);
    float sp_input = a_val + dt_bias_val;
    float softplus_val =
        (sp_input > 20.0f) ? sp_input : log(1.0f + exp(sp_input));
    g[gid] = exp(-exp(a_log_val) * softplus_val);
    beta[gid] = static_cast<bfloat16_t>(beta_val);
  }
}

#define GDN_PREFILL_PROLOGUE_CONV(name, ScalarT)                              \
  [[kernel]] void name(                                                       \
      const device bfloat16_t* qkv [[buffer(0)]],                             \
      const device float* conv_w [[buffer(1)]],                               \
      const device bfloat16_t* history [[buffer(2)]],                         \
      const device bfloat16_t* a [[buffer(3)]],                               \
      const device bfloat16_t* b [[buffer(4)]],                               \
      const device ScalarT* a_log [[buffer(5)]],                              \
      const device ScalarT* dt_bias [[buffer(6)]],                            \
      const device bfloat16_t* wq [[buffer(7)]],                              \
      const device bfloat16_t* wk [[buffer(8)]],                              \
      device bfloat16_t* qn [[buffer(9)]],                                    \
      device bfloat16_t* kn [[buffer(10)]],                                   \
      device bfloat16_t* v_out [[buffer(11)]],                                \
      device float* g [[buffer(12)]],                                         \
      device bfloat16_t* beta [[buffer(13)]],                                 \
      device bfloat16_t* history_out [[buffer(14)]],                          \
      constant GdnPrefillConvParams& p [[buffer(15)]],                        \
      uint gid [[thread_position_in_grid]],                                   \
      uint lane [[thread_index_in_simdgroup]]) {                              \
    gdn_prefill_prologue_conv_body<ScalarT>(                                  \
        qkv, conv_w, history, a, b, a_log, dt_bias, wq, wk, qn, kn, v_out, g, \
        beta, history_out, p, gid, lane);                                     \
  }

GDN_PREFILL_PROLOGUE_CONV(mlx_node_gdn_prefill_prologue_conv, float)
GDN_PREFILL_PROLOGUE_CONV(
    mlx_node_gdn_prefill_prologue_conv_bf16_scalars, bfloat16_t)

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
  constexpr uint Columns = 16;
  constexpr uint LanesPerRow = kGdnDk / Columns;
  constexpr uint GroupsPerHead = kGdnDv / Rows;
  static_assert(Rows == (Threads / 32) * (32 / LanesPerRow),
                "simdgroups cover threadgroup state rows");
  static_assert(Block * kGdnDk == Threads * 16,
                "every thread stages 16 keys and 16 queries");
  static_assert(Block * Rows == 32 * 8,
                "first simdgroup stages the value block");

  // 17,536 B threadgroup staging: pre-widened float4 k/q (16 KiB), float4 v
  // (1 KiB), and scalar gates (128 B).
  threadgroup float4 k_sh[Block * (kGdnDk / 4)];
  threadgroup float4 q_sh[Block * (kGdnDk / 4)];
  threadgroup float4 v_sh[Block * (Rows / 4)];
  threadgroup float g_sh[Block];
  threadgroup float beta_sh[Block];

  const uint T = uint(p.T);
  const uint Hk = uint(p.hk);
  const uint Hv = uint(p.hv);
  const uint hv_idx = group / GroupsPerHead;
  const uint row_base = (group % GroupsPerHead) * Rows;
  const uint hk_idx = p.tiled ? (hv_idx % Hk) : (hv_idx / (Hv / Hk));

  // 8 lanes per row, 4 rows per simdgroup: lane q owns contiguous columns
  // [q*16 .. q*16+15], matching 4 old 32-lane chunks so (s0+s1)+(s2+s3) plus
  // 3 xor shuffles reproduces AGX 32-lane simd_sum bit for bit.
  const uint row = sg * 4 + (lane >> 3);
  const uint q = lane & 7u;
  const uint col_base = q * Columns;
  const ulong state_offset =
      (ulong(hv_idx) * ulong(kGdnDv) + ulong(row_base + row)) * ulong(kGdnDk) +
      ulong(col_base);

  float4 state[4];
  if (p.has_state != 0) {
    const device float4* st_in =
        reinterpret_cast<const device float4*>(state_in + state_offset);
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
      state[c] = st_in[c];
    }
  } else {
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
      state[c] = float4(0.0f);
    }
  }

  bfloat4 key_fetch[2][2] = {};
  bfloat4 query_fetch[2][2] = {};
  bfloat4 value_fetch[2] = {};
  float g_fetch = 0.0f;
  float beta_fetch = 0.0f;

  auto fetch = [&](uint start) {
    for (uint n = 0; n < 2; ++n) {
      const uint element = (n * Threads + tid) * 8;
      const uint token = start + element / kGdnDk;
      if (token < T) {
        const ulong base =
            (ulong(token) * ulong(Hk) + ulong(hk_idx)) * ulong(kGdnDk) +
            ulong(element % kGdnDk);
        const device bfloat4* kp =
            reinterpret_cast<const device bfloat4*>(kn + base);
        const device bfloat4* qp =
            reinterpret_cast<const device bfloat4*>(qn + base);
        key_fetch[n][0] = kp[0];
        key_fetch[n][1] = kp[1];
        query_fetch[n][0] = qp[0];
        query_fetch[n][1] = qp[1];
      }
    }
    if (tid < 32u) {
      const uint element = tid * 8;
      const uint token = start + element / Rows;
      if (token < T) {
        const ulong base = ulong(token) * ulong(uint(p.v_stride_t)) +
            ulong(hv_idx * kGdnDv + row_base + (element % Rows));
        const device bfloat4* vp =
            reinterpret_cast<const device bfloat4*>(v + base);
        value_fetch[0] = vp[0];
        value_fetch[1] = vp[1];
      }
    }
    if (tid < Block && start + tid < T) {
      const ulong gate_idx = ulong(start + tid) * ulong(Hv) + ulong(hv_idx);
      g_fetch = g[gate_idx];
      beta_fetch = static_cast<float>(beta[gate_idx]);
    }
  };
  auto stage = [&]() {
    for (uint n = 0; n < 2; ++n) {
      const uint idx4 = (n * Threads + tid) * 2;
      k_sh[idx4 + 0] = float4(key_fetch[n][0]);
      k_sh[idx4 + 1] = float4(key_fetch[n][1]);
      q_sh[idx4 + 0] = float4(query_fetch[n][0]);
      q_sh[idx4 + 1] = float4(query_fetch[n][1]);
    }
    if (tid < 32u) {
      v_sh[tid * 2 + 0] = float4(value_fetch[0]);
      v_sh[tid * 2 + 1] = float4(value_fetch[1]);
    }
    if (tid < Block) {
      g_sh[tid] = g_fetch;
      beta_sh[tid] = beta_fetch;
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
    for (uint j = 0; j < count; ++j) {
      const float decay = g_sh[j];
      const float bt = beta_sh[j];
      const float val =
          reinterpret_cast<const threadgroup float*>(v_sh)[j * Rows + row];
      const threadgroup float4* key_ptr = k_sh + j * (kGdnDk / 4) + q * 4;
      const float4 key[4] = {key_ptr[0], key_ptr[1], key_ptr[2], key_ptr[3]};

      #pragma unroll
      for (uint c = 0; c < 4; ++c) {
        state[c] *= decay;
      }

      float kv;
      {
        #pragma clang fp reassociate(off)
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        s0 += state[0].x * key[0].x;
        s1 += state[1].x * key[1].x;
        s2 += state[2].x * key[2].x;
        s3 += state[3].x * key[3].x;
        s0 += state[0].y * key[0].y;
        s1 += state[1].y * key[1].y;
        s2 += state[2].y * key[2].y;
        s3 += state[3].y * key[3].y;
        s0 += state[0].z * key[0].z;
        s1 += state[1].z * key[1].z;
        s2 += state[2].z * key[2].z;
        s3 += state[3].z * key[3].z;
        s0 += state[0].w * key[0].w;
        s1 += state[1].w * key[1].w;
        s2 += state[2].w * key[2].w;
        s3 += state[3].w * key[3].w;
        kv = (s0 + s1) + (s2 + s3);
      }
      kv += simd_shuffle_xor(kv, ushort(1));
      kv += simd_shuffle_xor(kv, ushort(2));
      kv += simd_shuffle_xor(kv, ushort(4));
      const float delta = (val - kv) * bt;

      const threadgroup float4* query_ptr = q_sh + j * (kGdnDk / 4) + q * 4;
      const float4 query[4] = {
          query_ptr[0], query_ptr[1], query_ptr[2], query_ptr[3]};

      float out_val;
      {
        #pragma clang fp reassociate(off)
        state[0] = state[0] + key[0] * delta;
        state[1] = state[1] + key[1] * delta;
        state[2] = state[2] + key[2] * delta;
        state[3] = state[3] + key[3] * delta;
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        o0 += state[0].x * query[0].x;
        o1 += state[1].x * query[1].x;
        o2 += state[2].x * query[2].x;
        o3 += state[3].x * query[3].x;
        o0 += state[0].y * query[0].y;
        o1 += state[1].y * query[1].y;
        o2 += state[2].y * query[2].y;
        o3 += state[3].y * query[3].y;
        o0 += state[0].z * query[0].z;
        o1 += state[1].z * query[1].z;
        o2 += state[2].z * query[2].z;
        o3 += state[3].z * query[3].z;
        o0 += state[0].w * query[0].w;
        o1 += state[1].w * query[1].w;
        o2 += state[2].w * query[2].w;
        o3 += state[3].w * query[3].w;
        out_val = (o0 + o1) + (o2 + o3);
      }
      out_val += simd_shuffle_xor(out_val, ushort(1));
      out_val += simd_shuffle_xor(out_val, ushort(2));
      out_val += simd_shuffle_xor(out_val, ushort(4));
      if (q == 0u) {
        out[(ulong(start + j) * ulong(Hv) + ulong(hv_idx)) * ulong(kGdnDv) +
            ulong(row_base + row)] = static_cast<bfloat16_t>(out_val);
      }
    }
    if (more) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      stage();
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  device float4* st_out =
      reinterpret_cast<device float4*>(state_out + state_offset);
  #pragma unroll
  for (uint c = 0; c < 4; ++c) {
    st_out[c] = state[c];
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
