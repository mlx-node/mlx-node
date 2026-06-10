// W8A16 symmetric int8 MATVEC (QMV) for the sym8 DECODE path. Single-pass
// mixed-precision matvec: bf16 activations are read DIRECTLY from device memory
// (NO activation quant — no absmax pass, no int8 staging, no extra barriers)
// against the per-channel symmetric int8 weight. This is the decode fix for the
// fused W8A8 qmv (na_int8_qmv_fused.metal.inc), whose THREE in-kernel passes
// (absmax reduce -> int8 quant stage -> int8 matvec) are pure overhead at M=1:
// the memory bottleneck is the int8 WEIGHT stream (identical 1 byte/weight in
// both schemes) and int8 x int8 compute buys nothing at matvec arithmetic
// intensity. BONUS: decode becomes activation-EXACT (the only quantization
// error left is the weight's), so output drift vs bf16 drops.
//
// Math (per-channel symmetric int8 weight, bf16 activations, f32 accumulate):
//   acc[m,n] = sum_k (float)x[m,k] * (float)w_i8[k,n]     (f32 fma chain)
//   y[m,n]   = (bfloat16)( acc[m,n] * s_w[n] )
//
// WEIGHT LAYOUT: w_i8 is [K,N] row-major (w_i8[k,n] = original w[n,k]) — the
// SAME pre-transposed kernel operand the W8A8 GEMM/qmv consume; for a FIXED k
// the N output channels are CONTIGUOUS -> coalesced over n.
//
// THREADGROUP GEOMETRY (template BN, BK, VEC4, STAGE_X, MAXK) — identical
// 2D-block layout to the fused W8A8 qmv:
//   * Each tg owns BN output columns [n0,n0+BN) for ONE token row m.
//   * CC = VEC4 ? 4 : 1; NCOLT = BN / CC column-threads; NT = NCOLT * BK.
//   * tid = kw*NCOLT + ct: consecutive tids -> consecutive column-thread ct ->
//     contiguous weight reads (char4 groups under VEC4) stay coalesced.
//   * BK k-workers split the K reduction; f32 partials folded via a [BK][BN]
//     tg tile + tree reduce over kw (BK power-of-two). kw==0 applies s_w and
//     writes bf16 output.
//   * x[m,k] is a BROADCAST read: with NCOLT >= 32 every thread of a simdgroup
//     shares the same kw, hence the same k each iteration -> one transaction.
//     STAGE_X != 0 optionally stages x[m,:] into tg memory ONCE as bf16
//     (MAXK-bounded; host gates it off when 2*MAXK bytes blow the tg budget).
//     Measured A/B knob, OFF by default.
//
// Bound names (fast::metal_kernel auto-generates [[kernel]] sig + attrs):
//   x    : const device bfloat16_t*  [M,K] row-major bf16 activations
//   w_i8 : const device int8_t*      [K,N] row-major int8 weight (w_i8[k,n]=w[n,k])
//   s_w  : const device float*       [N]   per-output-channel weight scale
//   M,N,K: const constant int32_t&   scalars
//   y    : device bfloat16_t*        [M,N] row-major output
// Template params: <int BN, int BK, int VEC4, int STAGE_X, int MAXK>.
//
// DISPATCH (host int8_qmv_w8a16_core), dispatch_threads:
//   NT = (BN/(VEC4?4:1))*BK;  grid.x = ceil(N/BN)*NT;  grid.y = M;  tg = {NT,1,1}.
//   NT must be <= 1024 (Metal threadgroup cap) — host clamps BK down.
//   Host default is SHAPE-AWARE: N%128==0 -> BN=128/BK=32/VEC4=1 (each
//   simdgroup streams 128 contiguous weight bytes per k step — the
//   DRAM-transaction-width fix; in-stream 54.2us/call vs 67.3 scalar on the
//   4B decode mix), else BN=32/BK=32/VEC4=0 (scalar, any-N safe). The VEC4
//   path REQUIRES N % BN == 0: its char4 weight READS are unguarded (only
//   the epilogue write checks nn < N) — host hard-gates this.
R"(
  constexpr int CC    = VEC4 ? 4 : 1;       // columns per thread
  constexpr int NCOLT = BN / CC;            // column-threads per tg
  constexpr int NT    = NCOLT * BK;         // threads per tg

  const int Ni = N;
  const int Ki = K;

  const uint tid = thread_position_in_threadgroup.x;   // 0 .. NT-1
  const int ct  = (int)(tid % (uint)NCOLT);            // 0..NCOLT-1 (column group)
  const int kw  = (int)(tid / (uint)NCOLT);            // 0..BK-1 (k-worker)

  const int n0   = (int)threadgroup_position_in_grid.x * BN;  // first col of block
  const int m    = (int)threadgroup_position_in_grid.y;       // token row
  const int ncol = n0 + ct * CC;            // first column this thread owns
  const uint xrow_off = (uint)m * (uint)Ki;

  // Optional ONE-time bf16 stage of x[m,:] into tg memory (A/B knob, default
  // off — the global x read is a simdgroup-uniform broadcast that caches well).
  threadgroup bfloat16_t xs[STAGE_X ? MAXK : 1];
  if (STAGE_X) {
    for (uint j = tid; j < (uint)Ki; j += (uint)NT) {
      xs[j] = x[xrow_off + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- Single-pass mixed-precision matvec; CC columns/thread, f32 accum ----
  float acc[CC];
  for (int c = 0; c < CC; ++c) acc[c] = 0.0f;

  if (VEC4) {
    // char4 weight reads (4 contiguous columns). ncol guaranteed in-bounds &
    // 4-aligned by host gate (N%4==0, BN%4==0, N%BN==0 for all real shapes).
    const device char4* wv4 =
        (const device char4*)(w_i8 + (uint)ncol);   // stride N*sizeof(int8)
    for (int k = kw; k < Ki; k += BK) {
      float xv = STAGE_X ? (float)xs[k] : (float)x[xrow_off + (uint)k];
      char4 w4 = wv4[(uint)k * (uint)Ni / 4u];       // (k*N + ncol)/4 in char4 units
      acc[0] = fma(xv, (float)w4.x, acc[0]);
      acc[1] = fma(xv, (float)w4.y, acc[1]);
      acc[2] = fma(xv, (float)w4.z, acc[2]);
      acc[3] = fma(xv, (float)w4.w, acc[3]);
    }
  } else {
    if (ncol < Ni) {
      const device int8_t* wcol = w_i8 + (uint)ncol;
      for (int k = kw; k < Ki; k += BK) {
        float xv = STAGE_X ? (float)xs[k] : (float)x[xrow_off + (uint)k];
        acc[0] = fma(xv, (float)wcol[(uint)k * (uint)Ni], acc[0]);
      }
    }
  }

  // Fold BK partials per column via a [BK][BN] f32 tg tile + tree reduce over kw.
  threadgroup float part[BK * BN];
  for (int c = 0; c < CC; ++c) {
    part[kw * BN + ct * CC + c] = acc[c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int stride = BK / 2; stride > 0; stride >>= 1) {
    if (kw < stride) {
      for (int c = 0; c < CC; ++c) {
        part[kw * BN + ct * CC + c] += part[(kw + stride) * BN + ct * CC + c];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (kw == 0) {
    for (int c = 0; c < CC; ++c) {
      int nn = ncol + c;
      if (nn < Ni) {
        float total = part[ct * CC + c];
        y[(uint)m * (uint)Ni + (uint)nn] = (bfloat16_t)(total * s_w[nn]);
      }
    }
  }
)"
