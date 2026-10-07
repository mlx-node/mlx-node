// Fused DFlash2 GDN commit for a group of linear layers in ONE dispatch
// (ported from Splash `verify_gdn_commit`,
// runtime/metal/kernels/decode/gdn.metal).
//
// Per layer this reproduces, bit for bit, what the per-layer path did:
//   * recurrent state: `gated_delta_replay` — walk the first `keep` recorded
//     tokens from the pre-verify f32 state in f32 registers (the state is
//     never rounded, matching serial T=1 decode with an f32 carry);
//   * conv state: the last K-1 rows of `[conv_in ; qkv[0..keep]]`
//     (`replay_conv_state`'s concat + split, now a plain copy).
//
// Metal binds at most 31 buffers per dispatch, so the host encodes the
// layers in groups of GDN_COMMIT_GROUP (5 tape arrays each) and every group
// writes its own slice of the shared output blobs.
//
//   rec_in / rec_out:   [L, Hv, Dv, Dk] f32 (one blob for every layer)
//   conv_in / conv_out: [L, K-1, W]     bf16
//   per layer: k [S, Hk, Dk] bf16, v [S, Hv, Dv] bf16, g [S, Hv] f32,
//              beta [S, Hv] bf16, qkv [S, W] bf16 with row stride qkv_stride
// Grid (threads): (32, Dv, G * Hv); threadgroup (32, 4, 1) — the lane map of
// `gated_delta_replay`: one simdgroup per (layer, hv, dv) state row, each
// lane owning Dk/32 consecutive dk entries.
R"(
#define GDN_COMMIT_GROUP 5

struct GdnCommitParams {
  int keep;        // accepted tokens to replay (1..=S)
  int window;      // recorded window length S (time stride of the tape)
  int layer_base;  // first blob layer of this group
  int group;       // layers in this group (<= GDN_COMMIT_GROUP)
  int hk;          // key heads
  int hv;          // value heads
  int dk;          // key head dim (multiple of 32)
  int dv;          // value head dim
  int conv_rows;   // K-1 history rows per layer
  int conv_width;  // W (conv channels)
  int qkv_stride[GDN_COMMIT_GROUP];  // elements between qkv rows, per layer
};

#define GDN_COMMIT_TAPE(i)                                 \
  const device bfloat16_t* k##i [[buffer(5 + 5 * i)]],     \
  const device bfloat16_t* v##i [[buffer(6 + 5 * i)]],     \
  const device float* g##i [[buffer(7 + 5 * i)]],          \
  const device bfloat16_t* beta##i [[buffer(8 + 5 * i)]],  \
  const device bfloat16_t* qkv##i [[buffer(9 + 5 * i)]]

#define GDN_COMMIT_SELECT(i) \
  case i: k_ = k##i; v_ = v##i; g_ = g##i; beta_ = beta##i; qkv_ = qkv##i; break;

[[kernel]] void mlx_node_gdn_commit(
    const device float* rec_in [[buffer(0)]],
    const device bfloat16_t* conv_in [[buffer(1)]],
    device float* rec_out [[buffer(2)]],
    device bfloat16_t* conv_out [[buffer(3)]],
    constant GdnCommitParams& p [[buffer(4)]],
    GDN_COMMIT_TAPE(0), GDN_COMMIT_TAPE(1), GDN_COMMIT_TAPE(2),
    GDN_COMMIT_TAPE(3), GDN_COMMIT_TAPE(4),
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
  const uint n = thread_position_in_grid.z;      // group layer * Hv + hv
  const uint li = n / p.hv;
  const uint hv_idx = n % p.hv;
  const uint hk_idx = hv_idx % p.hk;
  const uint dk_idx = thread_position_in_threadgroup.x;
  const uint dv_idx = thread_position_in_grid.y;
  if (li >= uint(p.group)) {
    return;
  }
  const uint layer = uint(p.layer_base) + li;

  const device bfloat16_t* k_;
  const device bfloat16_t* v_;
  const device float* g_;
  const device bfloat16_t* beta_;
  const device bfloat16_t* qkv_;
  switch (li) {
    GDN_COMMIT_SELECT(0)
    GDN_COMMIT_SELECT(1)
    GDN_COMMIT_SELECT(2)
    GDN_COMMIT_SELECT(3)
    default: GDN_COMMIT_SELECT(4)
  }

  // --- conv carry: out row r = [conv_in ; qkv[0..keep]][keep + r] -------
  {
    const uint conv_elems = uint(p.conv_rows) * uint(p.conv_width);
    const uint per_layer_threads = uint(p.hv) * uint(p.dv) * 32u;
    const uint t = (hv_idx * uint(p.dv) + dv_idx) * 32u + dk_idx;
    const uint qkv_stride = uint(p.qkv_stride[li]);
    const device bfloat16_t* layer_conv_in = conv_in + layer * conv_elems;
    device bfloat16_t* layer_conv_out = conv_out + layer * conv_elems;
    for (uint e = t; e < conv_elems; e += per_layer_threads) {
      const uint row = e / uint(p.conv_width);
      const uint ch = e % uint(p.conv_width);
      const uint src = uint(p.keep) + row;
      layer_conv_out[e] = src < uint(p.conv_rows)
          ? layer_conv_in[src * uint(p.conv_width) + ch]
          : qkv_[(src - uint(p.conv_rows)) * qkv_stride + ch];
    }
  }

  // --- recurrent replay (== gated_delta_replay) -------------------------
  // GDN_COMMIT_N_PER_T (= Dk / 32) is prepended by the host so the state
  // stays in registers, as the templated `gated_delta_replay` keeps it.
  constexpr int n_per_t = GDN_COMMIT_N_PER_T;
  k_ += hk_idx * p.dk;
  v_ += hv_idx * p.dv;

  const ulong row = ulong(layer) * uint(p.hv) + hv_idx;
  const device float* i_state = rec_in + (row * p.dv + dv_idx) * p.dk;
  device float* o_state = rec_out + (row * p.dv + dv_idx) * p.dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = i_state[n_per_t * dk_idx + i];
  }
  for (int t = 0; t < p.keep; ++t) {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      const int s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] * g_[hv_idx];
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);
    auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
    for (int i = 0; i < n_per_t; ++i) {
      const int s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] + k_[s_idx] * delta;
    }
    k_ += p.hk * p.dk;
    v_ += p.hv * p.dv;
    g_ += p.hv;
    beta_ += p.hv;
  }
  for (int i = 0; i < n_per_t; ++i) {
    o_state[n_per_t * dk_idx + i] = state[i];
  }
}
)"
