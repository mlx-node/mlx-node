// W8A16 sym8 DECODE matvec (QMV), MLX-affine-qmv style: simdgroup-per-rows on
// the [N,K] CHECKPOINT orientation. This is the DRAM-streaming fix for the
// 2D-block kernel (na_int8_qmv_w8a16.metal.inc, in-stream 54.7us/call vs
// affine qmv's 31.6): same math, different decomposition —
//
//   * Each SIMDGROUP owns RPS consecutive output rows n (a threadgroup is NSG
//     simdgroups -> NSG*RPS rows). 32 lanes stride the K reduction.
//   * Per block iteration each lane loads LW CONTIGUOUS weight bytes from each
//     of its RPS rows (char4 groups -> a simdgroup streams LW*32 contiguous
//     bytes per row, e.g. 512B at LW=16 — the wide-transaction pattern MLX's
//     affine qmv_fast uses) and LW contiguous bf16 x values (loaded ONCE,
//     reused across all RPS rows).
//   * f32 accumulate in registers (float4 per row), ONE simd_sum cross-lane
//     fold per row at the end, lane 0 applies s_w[n] and writes bf16.
//   * ZERO threadgroup memory, ZERO barriers (the 2D-block kernel paid a
//     [BK][BN] f32 tile + ~5 barrier rounds per call).
//
// Math (identical to the 2D-block kernel, f32 accumulate):
//   y[m,n] = (bfloat16)( s_w[n] * sum_k (float)x[m,k] * (float)w_i8[n,k] )
//
// WEIGHT LAYOUT: w_i8 is the [N,K] ROW-MAJOR CHECKPOINT tensor (w_i8[n,k] —
// NOT the [K,N] kernel operand the GEMM consumes). For a fixed output row n
// the K reduction is CONTIGUOUS, so per-lane loads are wide. The host plumbs
// the [N,K] tensor separately (see qmv_w8a16_lazy / `{prefix}.weight_nk`).
//
// EDGES:
//   * K tail: lanes guard `kb < K`. Host gates K % 16 == 0 and LW <= 16 with
//     LW | 16, so a passing lane's LW-byte read never crosses K (kb and K are
//     both multiples of LW). char4 loads are 4B-aligned (row offsets are
//     multiples of K, K % 16 == 0; buffers are MLX-allocated, offset 0).
//   * N tail: a simdgroup whose first row is past N exits (uniform — no
//     barriers to desync); a partial simdgroup clamps DOWN to the last full
//     RPS window (host gates N >= RPS), recomputing a few rows another
//     simdgroup also computes — both write IDENTICAL values, benign.
//
// Bound names (fast::metal_kernel auto-generates [[kernel]] sig + attrs):
//   x    : const device bfloat16_t*  [M,K] row-major bf16 activations
//   w_i8 : const device int8_t*      [N,K] row-major int8 CHECKPOINT weight
//   s_w  : const device float*       [N]   per-output-channel weight scale
//   M,N,K: const constant int32_t&   scalars (M unused in-body; grid.y = M)
//   y    : device bfloat16_t*        [M,N] row-major output
// Template params: <int LW, int RPS, int NSG>
//   LW  = weight bytes per lane per row per iteration (4/8/16, divides 16)
//   RPS = output rows per simdgroup
//   NSG = simdgroups per threadgroup (tg = 32*NSG threads)
//
// DISPATCH (host int8_qmv_w8a16_sg_core):
//   grid.x = ceil(N / (NSG*RPS)) * 32*NSG;  grid.y = M;  tg = {32*NSG, 1, 1}.
R"(
  constexpr int LV = LW / 4;       // char4 chunks per lane load
  constexpr int BLOCK = LW * 32;   // k-values per simdgroup iteration

  const int Ni = N;
  const int Ki = K;

  const int m = (int)threadgroup_position_in_grid.y;
  int out_row = (int)threadgroup_position_in_grid.x * (NSG * RPS) +
                (int)simdgroup_index_in_threadgroup * RPS;
  if (out_row >= Ni) return;        // simdgroup-uniform exit (no barriers)
  out_row = min(out_row, Ni - RPS); // clamp: overlap-recompute, identical bits

  const device int8_t* wrow = w_i8 + (size_t)out_row * (size_t)Ki;
  const device bfloat16_t* xm = x + (size_t)m * (size_t)Ki;

  float4 acc[RPS];
  for (int r = 0; r < RPS; ++r) acc[r] = float4(0.0f);

  for (int kb = (int)thread_index_in_simdgroup * LW; kb < Ki; kb += BLOCK) {
    // x[m, kb .. kb+LW) loaded ONCE per block, shared across all RPS rows.
    float4 xt[LV];
    for (int j = 0; j < LV; ++j) {
      xt[j] = float4((float)xm[kb + 4 * j + 0], (float)xm[kb + 4 * j + 1],
                     (float)xm[kb + 4 * j + 2], (float)xm[kb + 4 * j + 3]);
    }
    for (int r = 0; r < RPS; ++r) {
      const device char4* wp =
          (const device char4*)(wrow + (size_t)r * (size_t)Ki + (size_t)kb);
      for (int j = 0; j < LV; ++j) {
        acc[r] = fma(float4(wp[j]), xt[j], acc[r]);
      }
    }
  }

  for (int r = 0; r < RPS; ++r) {
    float total = simd_sum(acc[r].x + acc[r].y + acc[r].z + acc[r].w);
    if (thread_index_in_simdgroup == 0) {
      const int n = out_row + r;
      y[(size_t)m * (size_t)Ni + (size_t)n] = (bfloat16_t)(total * s_w[n]);
    }
  }
)"
