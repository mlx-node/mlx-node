// C=16/V=32 GDN prefill. BF16 hi/lo operands and the NVIDIA MMA schedule
// are adapted from llama.cpp's gated-delta-net-mma.cu (MIT). State stays FP32.
R"cuda(
#include <cuda_bf16.h>

namespace gdn_mma {

constexpr int D = 128;
constexpr int C = 16;
constexpr int V = 32;
constexpr int WARPS = 8;

__device__ __forceinline__ int lane_id() {
  return threadIdx.x;
}

__device__ __forceinline__ int acc_row(int i) {
  return (i / 2) * 8 + lane_id() / 4;
}

__device__ __forceinline__ int acc_col(int i) {
  return 2 * (lane_id() % 4) + i % 2;
}

struct Acc {
  float x[4];
};

struct AFrag {
  unsigned int x[4];
};

struct BFrag {
  unsigned int x[2];
};

// STRIDE is measured in 32-bit packed BF16 pairs. This is the llama.cpp
// 128-byte swizzle, expressed without GGML types or runtime macros.
template <int STRIDE>
__device__ __forceinline__ int swizzle_pair(int offset, int row) {
  constexpr int BYTES = STRIDE * 4;
  static_assert(BYTES % 16 == 0, "ldmatrix row must be 16-byte aligned");
  if constexpr (BYTES % 32 != 0) {
    return offset;
  } else if constexpr (BYTES % 64 != 0) {
    return offset ^ (((row / 4) % 2) << 2);
  } else if constexpr (BYTES % 128 != 0) {
    return offset ^ (((row / 2) % 4) << 2);
  } else {
    return offset ^ ((row % 8) << 2);
  }
}

template <int R, int COLS>
struct alignas(16) Matrix {
  nv_bfloat16 hi[R * COLS];
  nv_bfloat16 lo[R * COLS];

  static __device__ __forceinline__ int index(int r, int c) {
    return 2 * swizzle_pair<COLS / 2>(r * (COLS / 2) + c / 2, r) + c % 2;
  }

  __device__ __forceinline__ void store2(int r, int c, float2 x) {
    const int i = index(r, c);
    const nv_bfloat162 h = __float22bfloat162_rn(x);
    const float2 rounded = __bfloat1622float2(h);
    *reinterpret_cast<nv_bfloat162*>(hi + i) = h;
    *reinterpret_cast<nv_bfloat162*>(lo + i) =
        __float22bfloat162_rn(make_float2(x.x - rounded.x, x.y - rounded.y));
  }

  __device__ __forceinline__ void store(int r, int c, float x) {
    const int i = index(r, c);
    const nv_bfloat16 h = __float2bfloat16_rn(x);
    hi[i] = h;
    lo[i] = __float2bfloat16_rn(x - __bfloat162float(h));
  }
};

template <int COLS, int VALS>
struct alignas(128) Shared {
  Matrix<C, D> q, k;
  Matrix<C, C> p, inverse;
  Matrix<C, VALS> delta;
  union {
    Matrix<D, VALS> state;
    Matrix<C, VALS> solved;
  } scratch;
  float lower[C * C];
  float prefix[C];
  float beta[C];
};

using SharedV32 = Shared<C, V>;
static_assert(sizeof(SharedV32) == 38016, "unexpected V32 shared layout");
static_assert(sizeof(SharedV32) <= 48 * 1024, "opt-in smem would be required");

template <int STRIDE, bool TRANS>
__device__ __forceinline__ AFrag load_a_packed(
    const nv_bfloat162* base, int offset) {
  const int lane = lane_id();
  const int i = lane % 16;
  const int j = (lane / 16) * 4;
  const int ij = swizzle_pair<STRIDE>(offset + i * STRIDE + j, i);
  const nv_bfloat162* p = base + ij;
  AFrag a;
  if constexpr (TRANS) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a.x[0]), "=r"(a.x[2]), "=r"(a.x[1]), "=r"(a.x[3])
        : "l"(p));
  } else {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3])
        : "l"(p));
  }
  return a;
}

template <int STRIDE, bool TRANS>
__device__ __forceinline__ BFrag load_b_packed(
    const nv_bfloat162* base, int offset) {
  const int lane = lane_id();
  BFrag b;
  if constexpr (TRANS) {
    const int i = lane % 16;
    const int ij = swizzle_pair<STRIDE>(offset + i * STRIDE, i);
    const nv_bfloat162* p = base + ij;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];"
        : "=r"(b.x[0]), "=r"(b.x[1])
        : "l"(p));
  } else {
    const int i = lane % 8;
    const int j = ((lane / 8) * 4) % 8;
    const int ij = swizzle_pair<STRIDE>(offset + i * STRIDE + j, i);
    const nv_bfloat162* p = base + ij;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
        : "=r"(b.x[0]), "=r"(b.x[1])
        : "l"(p));
  }
  return b;
}

template <bool TRANS, int R, int COLS>
__device__ __forceinline__ AFrag load_a(
    const Matrix<R, COLS>& m, int row, int col, bool low) {
  const nv_bfloat16* data = low ? m.lo : m.hi;
  const auto* packed = reinterpret_cast<const nv_bfloat162*>(data);
  const int offset = TRANS ? col * (COLS / 2) + row / 2
                           : row * (COLS / 2) + col / 2;
  return load_a_packed<COLS / 2, TRANS>(packed, offset);
}

template <bool TRANS, int R, int COLS>
__device__ __forceinline__ BFrag load_b(
    const Matrix<R, COLS>& m, int row, int col, bool low) {
  const nv_bfloat16* data = low ? m.lo : m.hi;
  const auto* packed = reinterpret_cast<const nv_bfloat162*>(data);
  const int offset = TRANS ? col * (COLS / 2) + row / 2
                           : row * (COLS / 2) + col / 2;
  return load_b_packed<COLS / 2, TRANS>(packed, offset);
}

__device__ __forceinline__ void mma_bf16(Acc& c, const AFrag& a,
                                         const BFrag& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c.x[0]), "+f"(c.x[1]), "+f"(c.x[2]), "+f"(c.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
        "r"(b.x[0]), "r"(b.x[1]));
}

// B is stored as B^T. Omit only the second-order lo*lo term.
template <int K, bool TA = false, bool TB = false, int AR, int AC, int BR,
          int BC>
__device__ __forceinline__ void product(Acc& c, const Matrix<AR, AC>& a,
                                        const Matrix<BR, BC>& b, int row,
                                        int col) {
#pragma unroll
  for (int k = 0; k < K; k += 16) {
    const AFrag ah = load_a<TA>(a, row, k, false);
    const AFrag al = load_a<TA>(a, row, k, true);
    const BFrag bh = load_b<TB>(b, col, k, false);
    const BFrag bl = load_b<TB>(b, col, k, true);
    mma_bf16(c, al, bh);
    mma_bf16(c, ah, bl);
    mma_bf16(c, ah, bh);
  }
}

template <int R, int COLS>
__device__ __forceinline__ void store_acc(Matrix<R, COLS>& m, const Acc& x,
                                          int r, int c) {
#pragma unroll
  for (int i = 0; i < 4; i += 2) {
    m.store2(r + acc_row(i), c + acc_col(i),
             make_float2(x.x[i], x.x[i + 1]));
  }
}

}  // namespace gdn_mma

// One block owns 32 value columns of one (batch,head) and advances 16 tokens.
extern "C" __global__ __launch_bounds__(256, 2) void mlx_node_gdn_recurrent_mma(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, const float* state, float* out, int T, int H) {
  using namespace gdn_mma;
  extern __shared__ __align__(128) unsigned char gdn_smem[];
  auto& s = *reinterpret_cast<SharedV32*>(gdn_smem);

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int v0 = blockIdx.x * V;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const long long soff = ((long long)b * H + h) * D * D;
  float* out_head = out + ((long long)b * T * H + h) * D + v0;

  constexpr int STATE_TILES = (D / 16) * (V / 8);
  constexpr int PER_WARP = STATE_TILES / WARPS;
  static_assert(PER_WARP == 4, "unexpected state schedule");
  Acc state_reg[PER_WARP];
#pragma unroll
  for (int j = 0; j < PER_WARP; ++j) {
    const int ti = warp + j * WARPS;
    const int r = (ti / (V / 8)) * 16;
    const int c = (ti % (V / 8)) * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      state_reg[j].x[i] =
          state[soff + (v0 + c + acc_col(i)) * D + r + acc_row(i)];
    }
  }

  auto read_qk = [&](long long t, float4& q4, float4& k4) {
    q4 = k4 = make_float4(0.f, 0.f, 0.f, 0.f);
    if (t < T) {
      const long long off = ((long long)b * T + t) * H * D + h * D + 4 * lane;
      if (((reinterpret_cast<unsigned long long>(q + off) |
            reinterpret_cast<unsigned long long>(k + off)) &
           15) == 0) {
        q4 = *reinterpret_cast<const float4*>(q + off);
        k4 = *reinterpret_cast<const float4*>(k + off);
      } else {
        q4 = make_float4(q[off], q[off + 1], q[off + 2], q[off + 3]);
        k4 = make_float4(k[off], k[off + 1], k[off + 2], k[off + 3]);
      }
    }
  };

  // Each warp prefetches its first row of the next chunk.
  float4 next_q, next_k;
  read_qk(warp, next_q, next_k);

  for (int t0 = 0; t0 < T; t0 += C) {
    const int valid = min(C, T - t0);
#pragma unroll
    for (int j = 0; j < C / WARPS; ++j) {
      const int t = warp + WARPS * j;
      const int d = 4 * lane;
      float4 q4, k4;
      if (j == 0) {
        q4 = next_q;
        k4 = next_k;
      } else {
        read_qk(t0 + t, q4, k4);
      }
      s.q.store2(t, d, make_float2(q4.x, q4.y));
      s.q.store2(t, d + 2, make_float2(q4.z, q4.w));
      s.k.store2(t, d, make_float2(k4.x, k4.y));
      s.k.store2(t, d + 2, make_float2(k4.z, k4.w));
    }

    if (warp == 0) {
      const int t = lane;
      float log_decay = 0.f;
      float bt = 0.f;
      if (t < valid) {
        const long long goff = ((long long)b * T + t0 + t) * H + h;
        const float gate = g[goff];
        log_decay = gate == 0.f ? -104.f : logf(fmaxf(gate, 1.17549435e-38f));
        bt = beta[goff];
      }
#pragma unroll
      for (int offset = 1; offset < 32; offset *= 2) {
        const float previous = __shfl_up_sync(0xffffffffu, log_decay, offset);
        if (lane >= offset) {
          log_decay += previous;
        }
      }
      if (t < C) {
        s.prefix[t] = log_decay;
        s.beta[t] = bt;
      }
    }

#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
      const int ti = warp + j * WARPS;
      const int r = (ti / (V / 8)) * 16;
      const int c = (ti % (V / 8)) * 8;
      store_acc(s.scratch.state, state_reg[j], r, c);
    }
    __syncthreads();

    if (t0 + C < T) {
      read_qk(t0 + C + warp, next_q, next_k);
    }

    // delta = beta * (v - decay * K S), base = decay * Q S.
    // Warps 4..7 own one output tile; warps 0..3 concurrently form KK and QK.
    Acc base_output = {};
    if (warp >= 4) {
      const int c = (warp - 4) * 8;
      Acc qs = {}, ks = {};
      product<D, false, true>(qs, s.q, s.scratch.state, 0, c);
      product<D, false, true>(ks, s.k, s.scratch.state, 0, c);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = acc_row(i);
        const int col = c + acc_col(i);
        const long long input_off =
            ((long long)b * T + t0 + t) * H * D + h * D + v0 + col;
        const float vv = t < valid ? v[input_off] : 0.f;
        const float decay = expf(s.prefix[t]);
        s.delta.store(t, col, s.beta[t] * (vv - decay * ks.x[i]));
        base_output.x[i] = decay * qs.x[i];
      }
    } else {
      const bool gram = warp < 2;
      const int c = (warp % 2) * 8;
      Acc x = {};
      if (gram) {
        product<D>(x, s.k, s.k, 0, c);
      } else {
        product<D>(x, s.q, s.k, 0, c);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int rr = acc_row(i);
        const int cc = c + acc_col(i);
        const float value = rr < valid && cc <= rr
                                ? x.x[i] * expf(s.prefix[rr] - s.prefix[cc])
                                : 0.f;
        if (gram) {
          if (cc < rr) {
            s.lower[rr * C + cc] = value * s.beta[rr];
          }
        } else {
          s.p.store(rr, cc, value);
        }
      }
    }
    __syncthreads();

    // inverse = (I + strictly_lower(A))^-1, two independent rows per warp.
    for (int row = warp; row < C; row += WARPS) {
      float x = lane == row ? 1.f
                            : (lane < row ? -s.lower[row * C + lane] : 0.f);
      for (int pivot = row - 1; pivot > 0; --pivot) {
        const float xp = __shfl_sync(0xffffffffu, x, pivot);
        if (lane < pivot) {
          x -= xp * s.lower[pivot * C + lane];
        }
      }
      if (lane < C) {
        s.inverse.store(row, lane, x);
      }
    }
    __syncthreads();

    if (warp < 4) {
      const int c = warp * 8;
      Acc x = {};
      product<C, false, true>(x, s.inverse, s.delta, 0, c);
      store_acc(s.scratch.solved, x, 0, c);
    }
    __syncthreads();

    if (warp >= 4) {
      const int c = (warp - 4) * 8;
      Acc x = {};
      product<C, false, true>(x, s.p, s.scratch.solved, 0, c);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = acc_row(i);
        const int col = c + acc_col(i);
        if (t < valid) {
          out_head[(long long)(t0 + t) * H * D + col] =
              base_output.x[i] + x.x[i];
        }
      }
    }

    // Reuse delta for solved * decay-to-chunk-end before the K^T update.
    for (int i = tid; i < C * V; i += WARPS * 32) {
      const int t = i / V;
      const int col = i % V;
      const int j = s.scratch.solved.index(t, col);
      const float solved = __bfloat162float(s.scratch.solved.hi[j]) +
                           __bfloat162float(s.scratch.solved.lo[j]);
      s.delta.store(t, col,
                    solved * expf(s.prefix[valid - 1] - s.prefix[t]));
    }
    __syncthreads();

    const float decay = expf(s.prefix[valid - 1]);
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
      const int ti = warp + j * WARPS;
      const int r = (ti / (V / 8)) * 16;
      const int c = (ti % (V / 8)) * 8;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        state_reg[j].x[i] *= decay;
      }
      product<C, true, true>(state_reg[j], s.k, s.delta, r, c);
    }
    __syncthreads();
  }

  float* state_out = out + (long long)gridDim.z * T * H * D + soff;
#pragma unroll
  for (int j = 0; j < PER_WARP; ++j) {
    const int ti = warp + j * WARPS;
    const int r = (ti / (V / 8)) * 16;
    const int c = (ti % (V / 8)) * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      state_out[(v0 + c + acc_col(i)) * D + r + acc_row(i)] =
          state_reg[j].x[i];
    }
  }
}
)cuda"

