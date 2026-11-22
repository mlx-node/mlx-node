R"METAL(
constexpr int BM = 32;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int PAD = 36;

const int R = token_rows_shape[0];
const int N_TOK = x_shape[0];
const int K = x_shape[1];
const int S = wg_shape[0];
const int M = wg_shape[1];

const uint tile = threadgroup_position_in_grid.y;
const uint begin = tiles[3u * tile];
const uint end = tiles[3u * tile + 1u];
const uint slot = tiles[3u * tile + 2u];
if (begin >= end || end > uint(R) || slot >= uint(S)) return;

const uint col = threadgroup_position_in_grid.x * uint(BN);
if (col >= uint(M)) return;

const uint sg_idx = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg_idx * 32u + lane;
const short tm = short((sg_idx / 2u) * 16u);
const short tn = short((sg_idx % 2u) * 32u);
const short valid = short(max(0, min(16, int(end - begin) - int(tm))));

// BM=32, BN=64, BK=32, PAD=36 keeps threadgroup memory at 23,040 B (<= 32 KB).
alignas(16) threadgroup float As[BM * PAD];
alignas(16) threadgroup float Gs[BN * PAD];
alignas(16) threadgroup float Us[BN * PAD];

constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto ca = op.template get_left_input_cooperative_tensor<float, float, float>();
auto cb = op.template get_right_input_cooperative_tensor<float, float, float>();
using AT = metal::remove_addrspace_t<decltype(ca)>;
using BT = metal::remove_addrspace_t<decltype(cb)>;
auto cg = op.template get_destination_cooperative_tensor<AT, BT, float>();
auto cu = op.template get_destination_cooperative_tensor<AT, BT, float>();
constexpr int FE = NAXTile<float, 1, 2>::kElemsPerFrag;
STEEL_PRAGMA_UNROLL
for (int e = 0; e < 2 * FE; ++e) {
  cg[e] = 0.0f;
  cu[e] = 0.0f;
}

const uint ar = tid / 4u;
const uint ac = (tid % 4u) * 8u;
bool live = (begin + ar) < end;
const uint token = live ? token_rows[begin + ar] : 0u;
live = live && (token < uint(N_TOK));
float4 av[2];
if (live) {
  const device float4* src = reinterpret_cast<const device float4*>(
      x + size_t(token) * size_t(K) + size_t(ac));
  av[0] = src[0];
  av[1] = src[1];
} else {
  // Dead rows never change across K tiles; zero once before the first barrier.
  threadgroup float4* dst =
      reinterpret_cast<threadgroup float4*>(As + ar * PAD + ac);
  dst[0] = float4(0.0f);
  dst[1] = float4(0.0f);
}

const bool is_up = (sg_idx >= 2u);
const uint wr_local = tid & 63u;
const size_t wr = size_t(slot) * size_t(M) + size_t(col + wr_local);
const device uint8_t* wp =
    reinterpret_cast<const device uint8_t*>(is_up ? wu : wg) +
    wr * size_t(K / 2);
KQScales<float, 4, 8, true, KQ_LINEAR, 0> wscale(
    is_up ? su : sg, is_up ? bu : bg, wr * size_t(K / 32));
threadgroup float* w_dst = (is_up ? Us : Gs) + wr_local * PAD;

PackedGroup32<4> pw;
float sc, bi;
wscale.at(0, sc, bi);
pw.prefetch(wp, sc, bi);

for (int base = 0; base < K; base += BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  pw.store(w_dst);
  if (live) {
    threadgroup float4* dst =
        reinterpret_cast<threadgroup float4*>(As + ar * PAD + ac);
    dst[0] = av[0];
    dst[1] = av[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int next = base + BK;
  if (next < K) {
    if (live) {
      const device float4* src = reinterpret_cast<const device float4*>(
          x + size_t(token) * size_t(K) + size_t(next) + size_t(ac));
      av[0] = src[0];
      av[1] = src[1];
    }
    wscale.at(size_t(next / 32), sc, bi);
    pw.prefetch(wp + size_t(next / 2), sc, bi);
  }

  if (valid > 0) {
    NAXTile<float, 1, 2> A;
    NAXTile<float, 2, 2> B;
    A.template load<float, PAD, 1>(As + tm * PAD);
    B.template load<float, PAD, 1>(Gs + tn * PAD);
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < 2; ++k) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < FE; ++e) {
        ca[e] = A.frag_at(0, k)[e];
        cb[e] = B.frag_at(0, k)[e];
        cb[FE + e] = B.frag_at(1, k)[e];
      }
      op.run(ca, cb, cg);
    }
    B.template load<float, PAD, 1>(Us + tn * PAD);
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < 2; ++k) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < FE; ++e) {
        ca[e] = A.frag_at(0, k)[e];
        cb[e] = B.frag_at(0, k)[e];
        cb[FE + e] = B.frag_at(1, k)[e];
      }
      op.run(ca, cb, cu);
    }
  }
}

if (valid > 0) {
  const float lim = limit[0];
  NAXTile<float, 1, 2> D;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < 2; ++f) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < FE; ++e) {
      float gv = cg[f * FE + e];
      float uv = cu[f * FE + e];
      // Asymmetric clamp: never lower-clamp gate.
      if (lim > 0.0f) {
        gv = metal::min(gv, lim);
        uv = metal::clamp(uv, -lim, lim);
      }
      D.val_frags[f][e] = (gv / (1.0f + metal::exp(-gv))) * uv;
    }
  }
  device float* out_ptr =
      out + size_t(begin + uint(tm)) * size_t(M) + size_t(col + uint(tn));
  if (valid == 16) {
    D.store(out_ptr, M);
  } else {
    D.store_slice(out_ptr, M, short2(0, 0), short2(32, valid));
  }
}
)METAL"
