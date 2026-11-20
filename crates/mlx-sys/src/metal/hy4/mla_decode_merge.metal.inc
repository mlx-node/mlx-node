R"METAL(
constexpr int D_V4 = D_V / 4;
constexpr float MAX_FLOOR = -1e30f;

const int S = o_part_shape[0];
const int H = o_part_shape[1];
const int h = int(threadgroup_position_in_grid.y);
const int s = int(threadgroup_position_in_grid.z);
const int tid = int(thread_index_in_threadgroup);
if (s >= S || h >= H || tid >= D_V4) {
  return;
}

const size_t sh = size_t(s) * size_t(H) + size_t(h);
const size_t mp_base = sh * size_t(N_SPLITS);

float global_max = MAX_FLOOR;
for (int sp = 0; sp < N_SPLITS; ++sp) {
  if (sum_part[mp_base + size_t(sp)] > 0.0f) {
    global_max = metal::max(global_max, max_part[mp_base + size_t(sp)]);
  }
}

float total_sum = 0.0f;
float4 acc = float4(0.0f);
const device float4* op4 = reinterpret_cast<const device float4*>(
    o_part + mp_base * size_t(D_V));
for (int sp = 0; sp < N_SPLITS; ++sp) {
  const float sp_sum = sum_part[mp_base + size_t(sp)];
  if (sp_sum > 0.0f) {
    const float f = metal::exp(max_part[mp_base + size_t(sp)] - global_max);
    total_sum = metal::fma(sp_sum, f, total_sum);
    acc = metal::fma(
        op4[size_t(sp) * size_t(D_V4) + size_t(tid)], float4(f), acc);
  }
}

const float sink_h = sinks[h];
const float m_all = metal::max(global_max, sink_h);
const float k_scale = metal::exp(global_max - m_all);
const float s_term = metal::exp(sink_h - m_all);
const float den = total_sum * k_scale + s_term;
const float4 o_final =
    (total_sum > 0.0f && den > 0.0f) ? ((acc * k_scale) / den) : float4(0.0f);

device float4* out4 =
    reinterpret_cast<device float4*>(out + sh * size_t(D_V));
out4[tid] = o_final;
)METAL"
