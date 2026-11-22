R"cuda(
#include <cuda_bf16.h>

// ============================================================================
// Qwen3.8-Flash-Next (qwen4_exp) Layer-1 PLE Fused Gate + Dilated Conv1d
// Geometry: H=2560, HC=4, W=10240, conv k=4, dilation=3, history=[9, 10240] BF16.
// ============================================================================

__device__ __forceinline__ float q4ple_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

__device__ __forceinline__ float q4ple_precise_sigmoid_f32(float x) {
  const float y = 1.0f / (1.0f + expf(fabsf(x)));
  return (x < 0.0f) ? y : (1.0f - y);
}

__device__ __forceinline__ nv_bfloat16 q4ple_bf16_silu(nv_bfloat16 raw) {
  const float f = __bfloat162float(raw);
  const nv_bfloat16 sig = __float2bfloat16_rn(q4ple_precise_sigmoid_f32(f));
  return __float2bfloat16_rn(f * __bfloat162float(sig));
}

template <bool W_FP32>
__device__ __forceinline__ void q4ple_load_w8(
    const void* w_ptr, size_t offset, float weight_offset, float out[8]) {
  if constexpr (W_FP32) {
    const float4 w0 = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(w_ptr) + offset);
    const float4 w1 = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(w_ptr) + offset + 4);
    out[0] = w0.x + weight_offset;
    out[1] = w0.y + weight_offset;
    out[2] = w0.z + weight_offset;
    out[3] = w0.w + weight_offset;
    out[4] = w1.x + weight_offset;
    out[5] = w1.y + weight_offset;
    out[6] = w1.z + weight_offset;
    out[7] = w1.w + weight_offset;
  } else {
    const uint4 wv = *reinterpret_cast<const uint4*>(reinterpret_cast<const nv_bfloat16*>(w_ptr) + offset);
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(wp[j]);
      out[2 * j] = f.x + weight_offset;
      out[2 * j + 1] = f.y + weight_offset;
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 1: qwen4_ple_gate_norm_cuda
// Fuses all 3 per-stream RMSNorms + signed-sqrt sigmoid gate + value broadcast:
//   k_norm    = rms_norm(key, norm_key_w, group=2560)
//   q_norm    = rms_norm(x, norm_query_w, group=2560)
//   score     = sum_h(k_norm * q_norm) / sqrt(2560)
//   gate      = sigmoid(sign(score) * sqrt(max(|score|, 1e-6)))
//   gated_val = bf16(gate * value)            -> [T, 4, 2560] BF16
//   conv_in   = rms_norm(gated_val, norm_conv_w, group=2560) -> [T, 4, 2560] BF16
// Grid: dim3(4, T, 1), Block: dim3(320, 1, 1) (10 warps, 1 uint4=8 bf16/thread).
// Works for ANY T >= 1 (decode T=1..16 AND prefill T=1024).
// ----------------------------------------------------------------------------
template <bool W_FP32>
__global__ __launch_bounds__(320, 3) void qwen4_ple_gate_norm_cuda(
    const nv_bfloat16* __restrict__ x,          // [T, 10240]
    const nv_bfloat16* __restrict__ key,        // [T, 10240]
    const nv_bfloat16* __restrict__ value,      // [T, 2560]
    const void* __restrict__ norm_key_w,        // [10240] f32 or bf16
    const void* __restrict__ norm_query_w,      // [10240] f32 or bf16
    const void* __restrict__ norm_conv_w,       // [10240] f32 or bf16
    nv_bfloat16* __restrict__ gated_val_out,    // [T, 10240]
    nv_bfloat16* __restrict__ conv_in_out,      // [T, 10240]
    int T,
    float eps,
    float weight_offset) {
  constexpr int H = 2560;
  constexpr int HC = 4;
  constexpr float INV_H = 1.0f / static_cast<float>(H);
  constexpr float INV_SQRT_H = 0.01976423537605237f; // 1 / sqrt(2560)

  const int s = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(blockIdx.y);
  if (t >= T) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const size_t stream_off = (static_cast<size_t>(t) * HC + s) * H;
  const size_t w_base = static_cast<size_t>(s) * H + static_cast<size_t>(tid) * 8;

  const uint4 kv = reinterpret_cast<const uint4*>(key + stream_off)[tid];
  const uint4 qv = reinterpret_cast<const uint4*>(x + stream_off)[tid];
  const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kv);
  const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(&qv);

  float kf[8], qf[8];
  float sq_k = 0.0f, sq_q = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 fk = __bfloat1622float2(kp[j]);
    const float2 fq = __bfloat1622float2(qp[j]);
    kf[2 * j] = fk.x;
    kf[2 * j + 1] = fk.y;
    qf[2 * j] = fq.x;
    qf[2 * j + 1] = fq.y;
    sq_k += fk.x * fk.x + fk.y * fk.y;
    sq_q += fq.x * fq.x + fq.y * fq.y;
  }

  __shared__ float warp_red[2][10];
  sq_k = q4ple_warp_sum(sq_k);
  sq_q = q4ple_warp_sum(sq_q);
  if (lane == 0) {
    warp_red[0][warp] = sq_k;
    warp_red[1][warp] = sq_q;
  }
  __syncthreads();
  if (warp == 0) {
    float tk = (lane < 10) ? warp_red[0][lane] : 0.0f;
    float tq = (lane < 10) ? warp_red[1][lane] : 0.0f;
    tk = q4ple_warp_sum(tk);
    tq = q4ple_warp_sum(tq);
    if (lane == 0) {
      warp_red[0][0] = sqrtf(tk * INV_H + eps);
      warp_red[1][0] = sqrtf(tq * INV_H + eps);
    }
  }
  __syncthreads();
  const float denom_k = warp_red[0][0];
  const float denom_q = warp_red[1][0];

  float wk[8], wq[8];
  q4ple_load_w8<W_FP32>(norm_key_w, w_base, weight_offset, wk);
  q4ple_load_w8<W_FP32>(norm_query_w, w_base, weight_offset, wq);

  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const nv_bfloat16 kn = __float2bfloat16_rn((kf[j] / denom_k) * wk[j]);
    const nv_bfloat16 qn = __float2bfloat16_rn((qf[j] / denom_q) * wq[j]);
    dot = fmaf(__bfloat162float(kn), __bfloat162float(qn), dot);
  }
  dot = q4ple_warp_sum(dot);
  if (lane == 0) {
    warp_red[0][warp] = dot;
  }
  __syncthreads();
  if (warp == 0) {
    float td = (lane < 10) ? warp_red[0][lane] : 0.0f;
    td = q4ple_warp_sum(td);
    if (lane == 0) {
      const float sc = td * INV_SQRT_H;
      const float sgn = (sc > 0.0f) ? 1.0f : ((sc < 0.0f) ? -1.0f : 0.0f);
      const float ssqrt = sgn * sqrtf(fmaxf(fabsf(sc), 1e-6f));
      const nv_bfloat16 g_bf = __float2bfloat16_rn(q4ple_precise_sigmoid_f32(ssqrt));
      warp_red[0][0] = __bfloat162float(g_bf);
    }
  }
  __syncthreads();
  const float gate_f = warp_red[0][0];

  const uint4 vv = reinterpret_cast<const uint4*>(value + static_cast<size_t>(t) * H)[tid];
  const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&vv);
  uint4 gvv;
  nv_bfloat16* gve = reinterpret_cast<nv_bfloat16*>(&gvv);
  float gvf[8];
  float sq_gv = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 fv = __bfloat1622float2(vp[j]);
    const nv_bfloat16 g0 = __float2bfloat16_rn(gate_f * fv.x);
    const nv_bfloat16 g1 = __float2bfloat16_rn(gate_f * fv.y);
    gve[2 * j] = g0;
    gve[2 * j + 1] = g1;
    const float f0 = __bfloat162float(g0);
    const float f1 = __bfloat162float(g1);
    gvf[2 * j] = f0;
    gvf[2 * j + 1] = f1;
    sq_gv += f0 * f0 + f1 * f1;
  }
  reinterpret_cast<uint4*>(gated_val_out + stream_off)[tid] = gvv;

  sq_gv = q4ple_warp_sum(sq_gv);
  if (lane == 0) {
    warp_red[0][warp] = sq_gv;
  }
  __syncthreads();
  if (warp == 0) {
    float tg = (lane < 10) ? warp_red[0][lane] : 0.0f;
    tg = q4ple_warp_sum(tg);
    if (lane == 0) {
      warp_red[0][0] = sqrtf(tg * INV_H + eps);
    }
  }
  __syncthreads();
  const float denom_c = warp_red[0][0];

  float wc[8];
  q4ple_load_w8<W_FP32>(norm_conv_w, w_base, weight_offset, wc);
  uint4 cv;
  nv_bfloat16* ce = reinterpret_cast<nv_bfloat16*>(&cv);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    ce[j] = __float2bfloat16_rn((gvf[j] / denom_c) * wc[j]);
  }
  reinterpret_cast<uint4*>(conv_in_out + stream_off)[tid] = cv;
}

// ----------------------------------------------------------------------------
// Kernel 2: qwen4_ple_conv_silu_add_cuda
// Fuses dilated causal depthwise conv1d (k=4, dilation=3, keep=9) + BF16 SiLU
// + ple = gated_val + silu_conv + optional residual x_out = x + ple
// + optional next_history[0..8] = full[T..T+8].
// When HAS_TAPS=true, each node t loads its 4 dilated taps from conv_taps[t*4..+4]:
//   tap >= 0        -> conv_in[tap * W + c]
//   -9 <= tap < 0   -> history[(tap + 9) * W + c] (or 0.0f if history == nullptr)
//   tap < -9        -> 0.0f
// On a linear chain, conv_taps[t] = [t-9, t-6, t-3, t], 0-ULP identical to
// HAS_TAPS=false.
// Inputs:
//   conv_in:      [T, 10240] bf16
//   gated_val:    [T, 10240] bf16
//   x:            [T, 10240] bf16 (used iff FUSE_RESIDUAL=true)
//   history:      [9, 10240] bf16 (may be nullptr -> treated as zeros!)
//   conv_w:       [10240, 4] f32 (W_FP32=true) or bf16 (W_FP32=false)
//   conv_taps:    [T, 4] int32 (used iff HAS_TAPS=true, else nullptr)
// Outputs:
//   out:          [T, 10240] bf16 (x + ple if FUSE_RESIDUAL else ple)
//   next_history: [9, 10240] bf16 (may be nullptr in tree mode; non-aliasing)
// Grid: dim3(40, grid_y, 1) where grid_y = min(T, 32), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
template <bool W_FP32, bool FUSE_RESIDUAL, bool HAS_TAPS>
__global__ __launch_bounds__(256, 4) void qwen4_ple_conv_silu_add_cuda(
    const nv_bfloat16* __restrict__ conv_in,
    const nv_bfloat16* __restrict__ gated_val,
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ history,
    const void* __restrict__ conv_w,
    const int* __restrict__ conv_taps,
    nv_bfloat16* __restrict__ out,
    nv_bfloat16* __restrict__ next_history,
    int T) {
  constexpr int W = 10240;
  constexpr int KEEP = 9; // (k-1) * dilation = 3 * 3
  const int c = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
  if (c >= W) return;

  float w0, w1, w2, w3;
  if constexpr (W_FP32) {
    const float4 wf = reinterpret_cast<const float4*>(conv_w)[c];
    w0 = wf.x;
    w1 = wf.y;
    w2 = wf.z;
    w3 = wf.w;
  } else {
    const uint2 wu = reinterpret_cast<const uint2*>(conv_w)[c];
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&wu.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&wu.y);
    w0 = __low2float(a);
    w1 = __high2float(a);
    w2 = __low2float(b);
    w3 = __high2float(b);
  }

  auto load_full = [&](int row) -> float {
    if (row < KEEP) {
      return history ? __bfloat162float(history[static_cast<size_t>(row) * W + c]) : 0.0f;
    }
    return __bfloat162float(conv_in[static_cast<size_t>(row - KEEP) * W + c]);
  };

  auto load_tap = [&](int idx) -> float {
    if (idx >= 0) {
      return __bfloat162float(conv_in[static_cast<size_t>(idx) * W + c]);
    }
    const int hidx = idx + KEEP;
    if (hidx >= 0 && history != nullptr) {
      return __bfloat162float(history[static_cast<size_t>(hidx) * W + c]);
    }
    return 0.0f;
  };

  for (int t = static_cast<int>(blockIdx.y); t < T; t += static_cast<int>(gridDim.y)) {
    float acc;
    if constexpr (HAS_TAPS) {
      const int4 taps = reinterpret_cast<const int4*>(conv_taps)[t];
      acc = load_tap(taps.x) * w0 +
            load_tap(taps.y) * w1 +
            load_tap(taps.z) * w2 +
            load_tap(taps.w) * w3;
    } else {
      acc = load_full(t) * w0 +
            load_full(t + 3) * w1 +
            load_full(t + 6) * w2 +
            load_full(t + 9) * w3;
    }
    const nv_bfloat16 silu_bf = q4ple_bf16_silu(__float2bfloat16_rn(acc));
    const size_t off = static_cast<size_t>(t) * W + c;
    const nv_bfloat16 ple = __float2bfloat16_rn(
        __bfloat162float(gated_val[off]) + __bfloat162float(silu_bf));
    if constexpr (FUSE_RESIDUAL) {
      out[off] = __float2bfloat16_rn(__bfloat162float(x[off]) + __bfloat162float(ple));
    } else {
      out[off] = ple;
    }
  }

  if (next_history != nullptr && blockIdx.y == 0) {
#pragma unroll
    for (int row = 0; row < KEEP; ++row) {
      const int src = T + row;
      nv_bfloat16 v;
      if (src < KEEP) {
        v = history ? history[static_cast<size_t>(src) * W + c] : __float2bfloat16_rn(0.0f);
      } else {
        v = conv_in[static_cast<size_t>(src - KEEP) * W + c];
      }
      next_history[static_cast<size_t>(row) * W + c] = v;
    }
  }
}
)cuda"
