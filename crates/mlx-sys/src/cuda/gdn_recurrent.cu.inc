R"cuda(
// Scalar-gate GDN with Dk=Dv=128. Four warps own 16 value columns per block;
// each lane keeps 16 key rows of those columns in registers across all tokens.
extern "C" __global__ __launch_bounds__(128, 2) void mlx_node_gdn_recurrent(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, const float* state, float* out, int T, int H) {
  constexpr int D = 128;
  constexpr int COLS_PER_WARP = 4;
  constexpr int LANES_PER_COL = 8;
  constexpr int ROWS_PER_LANE = D / LANES_PER_COL;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int lane = threadIdx.x;
  const int col_in_warp = lane / LANES_PER_COL;
  const int lane_in_col = lane % LANES_PER_COL;
  const int col = (blockIdx.z * blockDim.y + threadIdx.y) * COLS_PER_WARP + col_in_warp;
  const long long state_offset = ((long long)b * H + h) * D * D + col * D;
  const float* state_in = state + state_offset;
  float s[ROWS_PER_LANE];
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    s[r] = state_in[r * LANES_PER_COL + lane_in_col];
  }

  for (int t = 0; t < T; ++t) {
    const long long input_offset = ((long long)b * T + t) * H * D + h * D;
    const long long gate_offset = ((long long)b * T + t) * H + h;
    float k_reg[ROWS_PER_LANE];
    float q_reg[ROWS_PER_LANE];
    float kv = 0.0f;
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const int i = r * LANES_PER_COL + lane_in_col;
      k_reg[r] = k[input_offset + i];
      q_reg[r] = q[input_offset + i];
      kv += s[r] * k_reg[r];
    }
    for (int shift = 4; shift > 0; shift >>= 1) {
      kv += __shfl_xor_sync(0xffffffffu, kv, shift, LANES_PER_COL);
    }
    const float decay = g[gate_offset];
    const float delta = (v[input_offset + col] - decay * kv) * beta[gate_offset];
    float y = 0.0f;
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      s[r] = decay * s[r] + k_reg[r] * delta;
      y += s[r] * q_reg[r];
    }
    for (int shift = 4; shift > 0; shift >>= 1) {
      y += __shfl_xor_sync(0xffffffffu, y, shift, LANES_PER_COL);
    }
    if (lane_in_col == 0) {
      out[input_offset + col] = y;
    }
  }

  float* state_out = out + (long long)gridDim.y * T * H * D + state_offset;
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    state_out[r * LANES_PER_COL + lane_in_col] = s[r];
  }
}
)cuda"
