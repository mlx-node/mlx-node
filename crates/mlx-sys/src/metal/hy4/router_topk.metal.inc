R"METAL(
const int M = logits_shape[0];
const int E = logits_shape[1];
const uint m = threadgroup_position_in_grid.y;
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
if (int(m) >= M) return;

constexpr int MAX_ITERS = (MAX_E + 127) / 128;
threadgroup float tg_s[MAX_E];
threadgroup float tg_sg_val[4];
threadgroup uint tg_sg_id[4];

const size_t row_off = size_t(m) * size_t(E);
float my_sel[MAX_ITERS];
bool my_taken[MAX_ITERS];

for (int i = 0; i < MAX_ITERS; ++i) {
  const int e = int(tid) + i * 128;
  if (e < E) {
    const float lx = logits[row_off + size_t(e)];
    // Match MLX Sigmoid (unary_ops.h) so GPU routing matches host routing bit-for-bit.
    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(lx)));
    const float s = (lx < 0.0f) ? y : (1.0f - y);
    float sel = HAS_BIAS ? (s + bias[e]) : s;
    if (metal::isnan(sel)) {
      sel = -INFINITY;
    }
    tg_s[e] = s;
    my_sel[i] = sel;
    my_taken[i] = false;
  } else {
    my_sel[i] = -INFINITY;
    my_taken[i] = true;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint chosen_id[K];
float chosen_s[K];

for (int k = 0; k < K; ++k) {
  float best_sel = -INFINITY;
  uint best_id = 0xFFFFFFFFu;
  for (int i = 0; i < MAX_ITERS; ++i) {
    if (!my_taken[i]) {
      const float v = my_sel[i];
      const uint id = tid + uint(i * 128);
      if (best_id == 0xFFFFFFFFu || v > best_sel || (v == best_sel && id < best_id)) {
        best_sel = v;
        best_id = id;
      }
    }
  }
  const float sg_max_val = simd_max(best_sel);
  const uint cand_id =
      (best_id != 0xFFFFFFFFu && best_sel == sg_max_val) ? best_id : 0xFFFFFFFFu;
  const uint sg_win_id = simd_min(cand_id);
  if (lane == 0) {
    tg_sg_val[sg] = sg_max_val;
    tg_sg_id[sg] = sg_win_id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float win_val = tg_sg_val[0];
  uint win_id = tg_sg_id[0];
  for (uint s = 1; s < 4; ++s) {
    const float v = tg_sg_val[s];
    const uint id = tg_sg_id[s];
    if (id != 0xFFFFFFFFu &&
        (win_id == 0xFFFFFFFFu || v > win_val || (v == win_val && id < win_id))) {
      win_val = v;
      win_id = id;
    }
  }
  if (tid == 0) {
    chosen_id[k] = win_id;
    chosen_s[k] = tg_s[win_id];
  }
  if ((win_id & 127u) == tid) {
    my_taken[win_id >> 7] = true;
  }
  if (k + 1 < K) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

if (tid == 0) {
  const float scale = params[0];
  const float eps = params[1];
  float sum_s = 0.0f;
  for (int i = 0; i < K; ++i) {
    sum_s += chosen_s[i];
  }
  sum_s += eps;
  for (int i = 1; i < K; ++i) {
    const uint key_id = chosen_id[i];
    const float key_s = chosen_s[i];
    int j = i - 1;
    while (j >= 0 && chosen_id[j] > key_id) {
      chosen_id[j + 1] = chosen_id[j];
      chosen_s[j + 1] = chosen_s[j];
      --j;
    }
    chosen_id[j + 1] = key_id;
    chosen_s[j + 1] = key_s;
  }
  const size_t out_off = size_t(m) * size_t(K);
  for (int i = 0; i < K; ++i) {
    ids[out_off + size_t(i)] = chosen_id[i];
    const float w =
        (NORM && K > 1) ? metal::precise::divide(chosen_s[i], sum_s) : chosen_s[i];
    weights[out_off + size_t(i)] = w * scale;
  }
}
)METAL"
