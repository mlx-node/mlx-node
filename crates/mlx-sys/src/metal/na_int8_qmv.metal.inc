// Symmetric int8 MATVEC (QMV) for the sym8 DECODE path, for fast::metal_kernel
// JIT. This is the small-M (decode, M=1..~16) BW-bound analogue of the prefill
// int8 GEMM (na_int8_gemm.metal.inc): the prefill 128x64 tile wastes 127/128
// rows at M=1, so decode needs a dedicated matvec that streams each weight byte
// exactly once.
//
// Math (per-token symmetric int8 act quant + per-channel symmetric int8 weight):
//   acc[m,n] = sum_k x_i8[m,k] * w_i8[k,n]            (int32, exact)
//   y[m,n]   = (bfloat16) ( ((float)acc[m,n] * s_x[m]) * s_w[n] )
//
// WEIGHT LAYOUT (critical): the int8 weight arrives in the SAME [K,N] kernel
// layout that int8_w8a8_matmul consumes (w_i8[k,n] = original w[n,k]), stored
// row-contiguous so w_i8[k*N + n]. For a FIXED k, the N output channels are
// CONTIGUOUS in memory.
//
// =========================== 2D-BLOCKED GEMV ===========================
// ROOT CAUSE of the old naive kernel: one thread per output column with a fully
// SERIAL int32 K-loop and ZERO K-parallelism => only N threads total, each
// grinding all K alone => the 40-core M5 is starved (worse as K grows; that is
// why down/K9216 was 11.5x). This redesign parallelizes K across the threadgroup
// while keeping the weight reads coalesced over n.
//
// Threadgroup geometry (template params BN, BK; BN*BK <= 1024 threads/tg):
//   * Each threadgroup owns BN output columns [n0, n0+BN) for ONE token row m.
//   * The BK "k-workers" split the K reduction: worker kw handles
//     k = kw, kw+BK, kw+2BK, ...  (a strided slice of K).
//   * Flat thread layout tid = kw*BN + col, so threads with consecutive tid have
//     consecutive `col`. For a FIXED k, those BN threads read
//     w_i8[k*N + n0 + col] = a CONTIGUOUS byte run => fully COALESCED.
//   * int32 accum is exact: |x|,|w| <= 127, K <= 17408 -> max|acc| ~ 2.8e8.
//
// x staging (template STAGE_X): when STAGE_X != 0, x_i8[m, :] is staged once into
// threadgroup memory (xs, MAXK int8) so the BK*BN threads read each x byte from
// fast tg memory; when STAGE_X == 0 the x read is broadcast from global (all BN
// columns of a kw share x[k], so it caches well). Staging trades tg-memory
// pressure (occupancy) for fewer global x loads — A/B-tunable.
//
// Reduction: each kw worker keeps a register accumulator over its strided K
// slice, then the BK partials per column are folded via a threadgroup tile
// part[BK*BN] (laid out [kw][col] so a fixed kw row is contiguous in col) with a
// tree reduction over kw (BK power-of-two). kw==0 applies the s_x[m]*s_w[n]
// rescale and writes y (narrowed to bf16 INSIDE the kernel so a downstream bf16
// residual add is not promoted to f32).
//
// Bound names (fast::metal_kernel auto-generates the [[kernel]] sig + attrs):
//   x_i8 : const device int8_t*    [M,K] row-major int8 activations (from quant)
//   w_i8 : const device int8_t*    [K,N] row-major int8 weight (w_i8[k,n]=w[n,k])
//   s_x  : const device float*     [M]   per-row (per-token) activation scale
//   s_w  : const device float*     [N]   per-output-channel weight scale
//   M,N,K: const constant int32_t& scalars
//   y    : device bfloat16_t*      [M,N] row-major output
// Template params: <int BN, int BK, int STAGE_X, int MAXK>.
//
// DISPATCH (host side, mlx_na_int8.cpp / int8_qmv_core), dispatch_threads:
//   grid.x = ceil(N/BN) * (BN*BK)   (one tg per BN-column block, BN*BK thr/tg)
//   grid.y = M;     threadgroup = {BN*BK, 1, 1}.
R"(
  const int Ni = N;
  const int Ki = K;

  const uint tid = thread_position_in_threadgroup.x;   // 0 .. BN*BK-1
  const uint tgsize = (uint)(BN * BK);
  const int col  = (int)(tid % (uint)BN);              // 0..BN-1 (coalesced n)
  const int kw   = (int)(tid / (uint)BN);              // 0..BK-1 (k-worker)

  const int n0 = (int)threadgroup_position_in_grid.x * BN;  // first col of block
  const int m  = (int)threadgroup_position_in_grid.y;       // token row (grid.y=M)
  const int n  = n0 + col;
  const bool col_active = (n < Ni);

  const uint xrow_off = (uint)m * (uint)Ki;

  // Optionally stage x_i8[m, :] into threadgroup memory once.
  threadgroup int8_t xs[STAGE_X ? MAXK : 1];
  if (STAGE_X) {
    for (uint j = tid; j < (uint)Ki; j += tgsize) {
      xs[j] = x_i8[xrow_off + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Strided K reduction owned by this (col,kw) thread. int32 (exact).
  int acc = 0;
  if (col_active) {
    const device int8_t* wcol = w_i8 + (uint)n;   // w_i8[k*N + n]
    if (STAGE_X) {
      for (int k = kw; k < Ki; k += BK) {
        acc += (int)xs[k] * (int)wcol[(uint)k * (uint)Ni];
      }
    } else {
      const device int8_t* xptr = x_i8 + xrow_off;
      for (int k = kw; k < Ki; k += BK) {
        acc += (int)xptr[k] * (int)wcol[(uint)k * (uint)Ni];
      }
    }
  }

  // Fold the BK partials per column via a threadgroup tile [BK][BN].
  threadgroup int part[BK * BN];
  part[kw * BN + col] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Tree reduce over kw (BK power-of-two). Each step halves the live kw rows.
  for (int stride = BK / 2; stride > 0; stride >>= 1) {
    if (kw < stride) {
      part[kw * BN + col] += part[(kw + stride) * BN + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // kw==0 lane of each column applies the rescale + writes bf16 output.
  if (kw == 0 && col_active) {
    int total = part[col];
    float v = ((float)total * s_x[m]) * s_w[n];
    y[(uint)m * (uint)Ni + (uint)n] = (bfloat16_t)v;
  }
)"
