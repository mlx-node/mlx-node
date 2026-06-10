// FUSED symmetric int8 MATVEC (QMV) for the sym8 DECODE path. Single-kernel
// analogue of the affine qmv: it takes the bf16 activation DIRECTLY, computes the
// per-token symmetric int8 activation quant (absmax -> s_x -> int8) INSIDE the
// kernel, then runs the 2D-blocked int8 matvec + s_x*s_w rescale. This removes
// the separate na_int8_quant kernel launch + the [M,K] int8 intermediate that the
// two-kernel int8_qmv path pays, putting us on equal footing with the single fused
// affine qmv (the parity baseline).
//
// Math (per-token symmetric int8 act quant + per-channel symmetric int8 weight):
//   absmax[m] = max_k |x[m,k]|                     (f32)
//   s_x[m]    = max(absmax[m], 1e-12) / 127
//   x_i8[m,k] = (int8) clamp(rint(x[m,k]/s_x[m]), -127, 127)
//   acc[m,n]  = sum_k x_i8[m,k] * w_i8[k,n]         (int32, exact)
//   y[m,n]    = (bfloat16)( (float)acc[m,n] * s_x[m] * s_w[n] )
// The x quant is BIT-IDENTICAL to na_int8_quant (same f32 absmax, same rint, same
// DIVIDE-by-sx).
//
// WEIGHT LAYOUT: w_i8 is [K,N] row-major (w_i8[k,n] = original w[n,k]); for a
// FIXED k the N output channels are CONTIGUOUS -> coalesced over n.
//
// VEC4 (template VEC4): each thread owns CC = (VEC4 ? 4 : 1) consecutive output
// columns. With VEC4, the per-k weight read is a single `char4` (4 contiguous
// bytes = 4 columns) -> 4x fewer memory transactions / loop instructions and
// 1/4 the thread count, which raises achieved BW and cuts loop overhead. Requires
// N % 4 == 0 and BN % 4 == 0 (host gates VEC4 off otherwise). Thread count per tg
// = (BN/CC) * BK.
//
// THREADGROUP GEOMETRY (template BN, BK, VEC4; MAXK = tg x-stage bound):
//   * Each tg owns BN output columns [n0,n0+BN) for ONE token row m.
//   * CC = VEC4 ? 4 : 1; NCOLT = BN / CC column-threads; NT = NCOLT * BK threads.
//   * tid = kw*NCOLT + ct: consecutive tids -> consecutive column-thread ct ->
//     consecutive 4-column groups -> char4 reads stay contiguous/coalesced.
//   * BK k-workers split the K reduction; CC partials/thread folded via a
//     [BK][BN] tg tile + tree reduce over kw. kw==0 writes bf16 output.
//
// Bound names (fast::metal_kernel auto-generates [[kernel]] sig + attrs):
//   x    : const device bfloat16_t*  [M,K] row-major bf16 activations
//   w_i8 : const device int8_t*      [K,N] row-major int8 weight (w_i8[k,n]=w[n,k])
//   s_w  : const device float*       [N]   per-output-channel weight scale
//   M,N,K: const constant int32_t&   scalars
//   y    : device bfloat16_t*        [M,N] row-major output
// Template params: <int BN, int BK, int VEC4, int MAXK>.
//
// DISPATCH (host int8_qmv_fused_core), dispatch_threads:
//   NT = (BN/(VEC4?4:1))*BK;  grid.x = ceil(N/BN)*NT;  grid.y = M;  tg = {NT,1,1}.
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

  // ---- Pass 1: per-row absmax in f32 (strided over global bf16 x) ----
  float local_absmax = 0.0f;
  for (uint j = tid; j < (uint)Ki; j += (uint)NT) {
    float v = (float)x[xrow_off + j];
    local_absmax = max(local_absmax, fabs(v));
  }
  threadgroup float amax_part[(NT + 31) / 32];
  float sg_absmax = simd_max(local_absmax);
  const uint lane = thread_index_in_simdgroup;
  const uint sg   = tid / 32u;
  constexpr uint NSG = (NT + 31) / 32;
  if (lane == 0) {
    amax_part[sg] = sg_absmax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float sx_bc;
  if (sg == 0) {
    float p = (lane < NSG) ? amax_part[lane] : 0.0f;
    float row_absmax = simd_max(p);
    if (lane == 0) {
      sx_bc = max(row_absmax, 1e-12f) / 127.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float sx = sx_bc;

  // ---- Pass 2: quantize x into staged int8 (DIVIDE, then rint+clamp) ----
  threadgroup int8_t xs[MAXK];
  for (uint j = tid; j < (uint)Ki; j += (uint)NT) {
    float v = (float)x[xrow_off + j];
    float q = clamp(rint(v / sx), -127.0f, 127.0f);
    xs[j] = (int8_t)q;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 2D-block matvec over staged int8 x; CC columns/thread ----
  int acc[CC];
  for (int c = 0; c < CC; ++c) acc[c] = 0;

  if (VEC4) {
    // char4 weight reads (4 contiguous columns). ncol guaranteed in-bounds &
    // 4-aligned by host gate (N%4==0, BN%4==0, N%BN==0 for all real shapes).
    const device char4* wv4 =
        (const device char4*)(w_i8 + (uint)ncol);   // stride N*sizeof(int8)
    for (int k = kw; k < Ki; k += BK) {
      int xq = (int)xs[k];
      char4 w4 = wv4[(uint)k * (uint)Ni / 4u];       // (k*N + ncol)/4 in char4 units
      acc[0] += xq * (int)w4.x;
      acc[1] += xq * (int)w4.y;
      acc[2] += xq * (int)w4.z;
      acc[3] += xq * (int)w4.w;
    }
  } else {
    if (ncol < Ni) {
      const device int8_t* wcol = w_i8 + (uint)ncol;
      for (int k = kw; k < Ki; k += BK) {
        acc[0] += (int)xs[k] * (int)wcol[(uint)k * (uint)Ni];
      }
    }
  }

  // Fold BK partials per column via a [BK][BN] tg tile + tree reduce over kw.
  threadgroup int part[BK * BN];
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
        int total = part[ct * CC + c];
        float vv = ((float)total * sx) * s_w[nn];
        y[(uint)m * (uint)Ni + (uint)nn] = (bfloat16_t)vv;
      }
    }
  }
)"
