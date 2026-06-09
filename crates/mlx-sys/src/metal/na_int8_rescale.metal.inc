// Fused int32 -> bf16 rescale for the int8 W8A8 GEMM accumulator, for
// fast::metal_kernel JIT. Replaces the lazy MLX op chain
//   astype(acc,f32) -> *s_x[:,None] -> *s_w[None,:] -> astype(bf16)
// (multiple passes over the big [M,N] tensor) with ONE elementwise pass.
//
// Bound names (auto-generated [[kernel]] signature):
//   acc : const device int32_t*  [M,N] row-major GEMM accumulator
//   s_x : const device float*    [M] per-row act scale (== [M,1])
//   s_w : const device float*    [N] per-output-channel weight scale
//   M,N : const constant int32_t& scalars
//   y   : device bfloat16_t*     [M,N] row-major output
// Attribute injected: thread_position_in_grid.
//
// Math MUST match the lazy chain's association EXACTLY:
//   y[m,n] = (bfloat16) ( ((float)acc[m,n] * s_x[m]) * s_w[n] )
// i.e. multiply by s_x FIRST, then s_w (the lazy chain is
// multiply(multiply(acc_f32, s_x), s_w_row)). The narrow to bf16 happens INSIDE
// the kernel so a downstream bf16 residual add is not promoted to f32 by an f32
// scale (the existing pitfall guard).
//
// 2D grid: thread_position_in_grid.x = column n (0..N-1),
// thread_position_in_grid.y = row m (0..M-1). The host splits the dispatch into
// (grid.x=N, grid.y=M) so neither extent overflows the int-typed metal_kernel
// grid tuple (a flat M*N grid casts negative at M*N >= 2^31; review #7). The
// flat element offset is reconstructed in 64-bit (m*N + n) so addressing stays
// correct even when M*N exceeds 2^31. Element coverage and the s_x[m]/s_w[n]
// (m=row, n=col) indexing are identical to the prior flat-1D layout.
R"(
  const uint n = thread_position_in_grid.x;
  const uint m = thread_position_in_grid.y;
  // Defensive guard (dispatch_threads launches exactly grid threads, so out-of-
  // range threads are not normally spawned; mirrors the prior gid>=total guard).
  if (n >= (uint)N || m >= (uint)M) {
    return;
  }
  const size_t idx = (size_t)m * (size_t)N + (size_t)n;
  float a = (float)acc[idx];
  float v = (a * s_x[m]) * s_w[n];
  y[idx] = (bfloat16_t)v;
)"
