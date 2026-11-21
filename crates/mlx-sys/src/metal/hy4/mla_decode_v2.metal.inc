R"METAL(
constexpr int DK = 576;
constexpr int DV = 512;
constexpr int HALF_DK = 288;
constexpr int TR = 16;
constexpr int NG = 2;
constexpr int NQ = 4;
constexpr int NT = NG * NQ * 32;
constexpr int DVS = DV / NQ;
constexpr int KBS = DK / 8 / NQ;
constexpr int HT = NG * 8;
constexpr float MAX_FLOOR = -1e30f;

const int S = q_shape[0];
const int H = q_shape[1];
const int KMAX = idx_shape[1];
const int N_KEYS = k0_shape[0];
const int n_splits = split_cfg[0];
const int k_split = split_cfg[1];

const int y_id = int(threadgroup_position_in_grid.y);
const int head_group = y_id / n_splits;
const int sp = y_id - head_group * n_splits;
const int h0 = head_group * HT;
const int s = int(threadgroup_position_in_grid.z);
if (s >= S || h0 >= H) {
  return;
}

const ushort tid = ushort(thread_index_in_threadgroup);
const ushort sgitg = ushort(simdgroup_index_in_threadgroup);
const short g = short(sgitg / NQ);
const short qs = short(sgitg % NQ);

// float4 backing guarantees 16-byte alignment for float4 K/V loads and O output staging.
threadgroup float4 sk_buf[TR * DK / 8];
threadgroup half* sk = reinterpret_cast<threadgroup half*>(sk_buf);
threadgroup float ss[NG * NQ * 8 * TR];
threadgroup half sp_buf[NG * 8 * TR];
threadgroup float sd[NG * 64];
threadgroup float sm[HT];
threadgroup float sS[HT];
threadgroup int sr[TR];
threadgroup int sn;

const float sc = scale[0];

const size_t q_s_off = size_t(s) * size_t(H) * size_t(DK);
for (int i = int(tid); i < HT * (DK / 4); i += NT) {
  const int lh = i / (DK / 4);
  const int c = i % (DK / 4);
  const int h = h0 + lh;
  const float4 v = (h < H)
      ? reinterpret_cast<const device float4*>(
            q + q_s_off + size_t(h) * size_t(DK))[c]
      : float4(0.0f);
  reinterpret_cast<threadgroup half4*>(sk)[i] = half4(v);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_half8x8 mq[KBS];
for (short b = 0; b < KBS; ++b) {
  simdgroup_load(mq[b], sk + g * 8 * DK + (qs * KBS + b) * 8, DK);
}

if (tid < HT) {
  sm[tid] = MAX_FLOOR;
  sS[tid] = 0.0f;
}

simdgroup_float8x8 mo[DVS / 8];
for (short i = 0; i < DVS / 8; ++i) {
  mo[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
}

const int k_start = sp * k_split;
const int k_end =
    (sp + 1 < n_splits) ? metal::min(k_start + k_split, KMAX) : KMAX;
const size_t idx_base = size_t(s) * size_t(KMAX);

const int tg = int(tid) / 128;
const int th = (int(tid) / TR) % 8;
const int tr = int(tid) % TR;

constexpr int DK8 = DK / 8;
constexpr int HALF_DK8 = HALF_DK / 8;

for (int rt = k_start; rt < k_end; rt += TR) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    int slot = -1;
    if (tid < TR) {
      const int r = rt + int(tid);
      if (r < k_end) {
        const int cand = idx[idx_base + size_t(r)];
        if (cand >= 0 && cand < N_KEYS) {
          slot = cand;
        }
      }
      sr[tid] = slot;
    }
    const bool any_valid = simd_any(slot >= 0);
    if (tid == 0) {
      sn = any_valid ? 1 : 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Sparse index rows can contain empty tiles before later valid entries.
  if (sn == 0) {
    continue;
  }

  for (int i = int(tid); i < TR * DK8; i += NT) {
    const int rr = i / DK8;
    const int c = i % DK8;
    const int slot = sr[rr];
    float4 v = float4(0.0f);
    if (slot >= 0) {
      if constexpr (SPLIT) {
        if (c < HALF_DK8) {
          v = reinterpret_cast<const device float4*>(
              k0 + size_t(slot) * size_t(HALF_DK))[c];
        } else {
          v = reinterpret_cast<const device float4*>(
              k1 + size_t(slot) * size_t(HALF_DK))[c - HALF_DK8];
        }
      } else {
        v = reinterpret_cast<const device float4*>(
            k0 + size_t(slot) * size_t(DK))[c];
      }
    }
    sk_buf[i] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  {
    simdgroup_float8x8 ms0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 ms1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (short b = 0; b < KBS; ++b) {
      const short kb = qs * KBS + b;
      simdgroup_half8x8 mk0;
      simdgroup_half8x8 mk1;
      simdgroup_load(mk0, sk + 0 * DK + kb * 8, DK, ulong2(0), true);
      simdgroup_load(mk1, sk + 8 * DK + kb * 8, DK, ulong2(0), true);
      simdgroup_multiply_accumulate(ms0, mq[b], mk0, ms0);
      simdgroup_multiply_accumulate(ms1, mq[b], mk1, ms1);
    }
    simdgroup_store(ms0, ss + (g * NQ + qs) * 8 * TR + 0, TR);
    simdgroup_store(ms1, ss + (g * NQ + qs) * 8 * TR + 8, TR);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  {
    const bool valid = (sr[tr] >= 0);
    float s_val = 0.0f;
    for (short n = 0; n < NQ; ++n) {
      s_val += ss[(tg * NQ + n) * 8 * TR + th * TR + tr];
    }
    s_val *= sc;

    float mx = valid ? s_val : MAX_FLOOR;
    mx = metal::max(mx, simd_shuffle_xor(mx, 8));
    mx = metal::max(mx, simd_shuffle_xor(mx, 4));
    mx = metal::max(mx, simd_shuffle_xor(mx, 2));
    mx = metal::max(mx, simd_shuffle_xor(mx, 1));

    const int hh = tg * 8 + th;
    const float mold = sm[hh];
    const float mnew = metal::max(mold, mx);
    const float p = valid ? metal::exp(s_val - mnew) : 0.0f;
    const float msc = metal::exp(mold - mnew);

    float ps = p;
    ps += simd_shuffle_xor(ps, 8);
    ps += simd_shuffle_xor(ps, 4);
    ps += simd_shuffle_xor(ps, 2);
    ps += simd_shuffle_xor(ps, 1);

    sp_buf[hh * TR + tr] = half(p);
    if (tr < 8) {
      sd[tg * 64 + th * 8 + tr] = (tr == th) ? msc : 0.0f;
    }
    // Lanes 0..15 of each simdgroup half read sm[hh] before lane 0 updates it.
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (tr == 0) {
      sm[hh] = mnew;
      sS[hh] = metal::fma(sS[hh], msc, ps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  {
    simdgroup_float8x8 md;
    simdgroup_half8x8 mp0;
    simdgroup_half8x8 mp1;
    simdgroup_load(md, sd + g * 64, 8);
    simdgroup_load(mp0, sp_buf + g * 8 * TR + 0, TR);
    simdgroup_load(mp1, sp_buf + g * 8 * TR + 8, TR);
    for (short i = 0; i < DVS / 8; ++i) {
      const short d0 = qs * DVS + i * 8;
      simdgroup_half8x8 mv0;
      simdgroup_half8x8 mv1;
      simdgroup_load(mv0, sk + 0 * DK + d0, DK);
      simdgroup_load(mv1, sk + 8 * DK + d0, DK);
      simdgroup_multiply(mo[i], md, mo[i]);
      simdgroup_multiply_accumulate(mo[i], mp0, mv0, mo[i]);
      simdgroup_multiply_accumulate(mo[i], mp1, mv1, mo[i]);
    }
  }
}

threadgroup float* so = reinterpret_cast<threadgroup float*>(sk_buf);
for (short gg = 0; gg < NG; ++gg) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g == gg) {
    for (short i = 0; i < DVS / 8; ++i) {
      simdgroup_store(mo[i], so + qs * DVS + i * 8, DV);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(tid); i < 8 * (DV / 4); i += NT) {
    const int lh = gg * 8 + i / (DV / 4);
    const int c = i % (DV / 4);
    const int h = h0 + lh;
    if (h < H) {
      const size_t mp_idx =
          (size_t(s) * size_t(H) + size_t(h)) * size_t(n_splits) + size_t(sp);
      device float4* out4 =
          reinterpret_cast<device float4*>(o_part + mp_idx * size_t(DV));
      out4[c] = (sS[lh] > 0.0f) ? sk_buf[i] : float4(0.0f);
    }
  }
}

if (tid < HT) {
  const int h = h0 + int(tid);
  if (h < H) {
    const size_t mp_idx =
        (size_t(s) * size_t(H) + size_t(h)) * size_t(n_splits) + size_t(sp);
    max_part[mp_idx] = sm[tid];
    sum_part[mp_idx] = sS[tid];
  }
}
)METAL"
