// Affine-group W8A8 GEMM body for fast::metal_kernel JIT — TILED matmul2d v2.
//
// Computes, on the model's EXACT affine packed weight (no re-quant):
//   y[m,n] = s_x[m] * sum_g ( scale[n,g]*P_s[m,n,g] + bias_adj[n,g]*S[m,g] )
// where, for the affine MLX 8-bit pack with dequant w[n,k]=scale[n,g]*q[n,k]+
// bias[n,g] (q UNSIGNED uint8 in [0,255], group g=k/group_size), we SHIFT the
// weight to signed so MPP's matmul2d<int8_t,int8_t->int32_t> accepts it:
//   q_s[n,k] = q[n,k] - 128                              in [-128,127]
//   bias_adj[n,g] = 128*scale[n,g] + bias[n,g]
// and the two per-group partials are
//   P_s[m,n,g] = sum_{k in g} x_q[m,k] * q_s[n,k]        (one matmul2d / group)
//   S[m,g]     = sum_{k in g} x_q[m,k]                   (act group-sum, n-indep)
// Proof: P = sum x_q*q = sum x_q*(q_s+128) = P_s + 128*S, so
//   scale*P + bias*S = scale*P_s + (128*scale+bias)*S = scale*P_s + bias_adj*S.
//
// This is the TILED analogue of the symmetric W8A8 kernel
// (na_int8_gemm.metal.inc): per 128x64 output tile (8 simdgroups), it runs ONE
// mpp::tensor_ops::matmul2d<int8,int8->int32> per K=GROUP_W (64) group slice,
// keeping the per-group int32 P tile in a COOPERATIVE TENSOR (registers — NO
// device round-trip, NO threadgroup tile). It folds scale*P_s + bias_adj*S into
// a running f32 cooperative accumulator, then writes s_x[m]*acc -> bf16.
//
//   WHY cooperative_tensor (in-register), not threadgroup or device:
//   * A 128x64 int32 P tile = 32 KB = the ENTIRE threadgroup memory budget; no
//     room left, and MLX's fast::metal_kernel does NOT set a dynamic
//     threadgroup length (CustomKernel::shared_memory_ is unused), so only
//     STATIC tg arrays work — a 32 KB static tile is infeasible.
//   * Per-group DEVICE P would force K/64 (= 40 for K=2560, up to 272 for
//     K=17408) device write+read round-trips per output tile — bandwidth death.
//   * cooperative_tensor is the canonical MPP post-processing path (see the
//     MPPTensorOpsMatMul2d.h `simpleMatMulCooperative` example): the GEMM result
//     lives in thread-private registers, divided across the 8-simdgroup scope;
//     we post-process (scale/bias fold) in-register before the single store.
//
// Per-group K bounding: the descriptor's k is FIXED at the group width
// (GROUP_W = 64) — NOT dynamic_extent. This bounds each matmul2d.run to exactly
// GROUP_W K-elements from the slice origin (the canonical k-loop pattern in
// MPPTensorOpsMatMul2d.h's `matMulKLoop`), so a per-group slice with the FULL-K
// tensor extents reads exactly its group, not to end-of-K. M/N edge overhang
// (M%128, N%64) is handled by `slice()` (dynamic M/N extents, edge-checked) plus
// the explicit coordinate guard before each store.
//
// REQUIRES group_size == GROUP_W (64). The host gates K % group_size == 0 and
// passes group_size; this kernel hardcodes the matmul k at GROUP_W and asserts
// G == GROUP_W is the caller's contract (the affine MLX pack is group_size 64).
//
// Bound names (fast::metal_kernel auto-generates the [[kernel]] signature):
//   x_i8 : const device int8_t*   [M,K]    row-major int8 activations (x_q)
//   q_s  : const device int8_t*   [K,N]    SIGNED weight, contiguous [K,N] with
//          q_s[k*N + n] = q[n,k]-128. The matmul2d column-major (N,K) view reads
//          q_s[n + k*N] = q[n,k]-128 (same layout as the symmetric kernel b_kn).
//          Built ONCE at load (mlx_affine_w8a8_prepare).
//   scl  : const device float*    [N, K/G] per-(out-chan, group) f32 scale (kept)
//   badj : const device float*    [N, K/G] per-(out-chan, group) f32 bias_adj
//          (= 128*scale + bias). Built ONCE at load.
//   s_x  : const device float*    [M]      per-token act scale (== [M,1])
//   S    : const device int32_t*  [M, K/G] per-(token, group) act group-sum
//          sum_{k in g} x_q[m,k]. Built per-forward by the quant+S kernel.
//   M,N,K,G : const constant int32_t&      dims + group_size (== GROUP_W)
//   y    : device bfloat16_t*     [M,N]    row-major output
R"(
  constexpr int TM2 = 128;
  constexpr int TN2 = 64;
  constexpr int SG2 = 8;
  constexpr int GROUP_W = 64;   // affine group width == matmul2d k per group

  // Per-K-group (GROUP_W) matmul2d: FIXED k = GROUP_W bounds each run to exactly
  // the group; multiply mode overwrites the cooperative P tile each group;
  // int8 x int8 -> int32.
  constexpr auto pdesc = matmul2d_descriptor(
      TM2, TN2, GROUP_W,
      false, false, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<pdesc, execution_simdgroups<SG2>> pop;

  const int Mi = M;
  const int Ni = N;
  const int Ki = K;
  const int n_groups = Ki / GROUP_W;   // K % GROUP_W == 0 (host-gated)

  // CRITICAL (same as the symmetric kernel): the tensor ELEMENT TYPE must be a
  // NON-const int8_t — MPP matches `__is_same_v<leftValueType, int8_t>` exactly,
  // and MLX binds A/B as `const device int8_t*`, so const_cast first. Tensors
  // carry FULL extents; per-group slicing shifts the K origin and the fixed
  // descriptor k = GROUP_W bounds the traversal.
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      const_cast<device int8_t*>(x_i8), dextents<int32_t, 2>(Ki, Mi));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      const_cast<device int8_t*>(q_s), dextents<int32_t, 2>(Ni, Ki));

  const uint m0 = threadgroup_position_in_grid.y * TM2;  // tile row origin
  const uint n0 = threadgroup_position_in_grid.x * TN2;  // tile col origin

  // f32 accumulator cooperative tensor for this 128x64 output tile (registers).
  // Built from the SAME sliced operand decltypes as P below so the (m,n) element
  // layout is identical → element index i maps to the same local coordinate in
  // both acc and every per-group P.
  auto mA0 = tA.slice(0, m0);
  auto mB0 = tB.slice(n0, 0);
  auto acc = pop.get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), float>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      acc[i] = 0.0f;
    }
  }

  // Walk groups; each iteration runs ONE matmul2d over the GROUP_W-wide K slice
  // into P (overwrite), then folds scale*P + bias_adj*S into the f32 accumulator.
  for (int g = 0; g < n_groups; ++g) {
    const int k0 = g * GROUP_W;
    auto mA = tA.slice(k0, m0);   // K-origin k0, M-origin m0; descriptor k=GROUP_W
    auto mB = tB.slice(n0, k0);   // N-origin n0, K-origin k0
    auto P = pop.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int>();
    pop.run(mA, mB, P);

    // Fold this group's partial into acc element-by-element. The cooperative P
    // and acc share the SAME op/scope/tile layout, so element index i maps to
    // the SAME local (row=m, col=n) coordinate in both.
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < P.get_capacity(); ++i) {
      if (!P.is_valid_element(i)) {
        continue;
      }
      // Local tile coordinate of element i. The matmul2d destination coordinate
      // system is (N, M) order (see MPPTensorOpsMatMul2d.h: descriptor (64,32)
      // is documented as a "32 x 64" = N x M tile), so idx[0]=n-local [0,TN2),
      // idx[1]=m-local [0,TM2).
      auto idx = P.get_multidimensional_index(i);
      const uint n = n0 + (uint)idx[0];   // global col / out-channel (dim0 = N)
      const uint m = m0 + (uint)idx[1];   // global row / token        (dim1 = M)
      // Guard the tile overhang (M%128, N%64) — out-of-range coords skip.
      if (m >= (uint)Mi || n >= (uint)Ni) {
        continue;
      }
      const float scale = scl[(size_t)n * (size_t)n_groups + (size_t)g];
      const float badj_v = badj[(size_t)n * (size_t)n_groups + (size_t)g];
      const float Sv = (float)S[(size_t)m * (size_t)n_groups + (size_t)g];
      acc[i] += scale * (float)P[i] + badj_v * Sv;
    }
  }

  // Final write: y[m,n] = s_x[m] * acc[m,n], narrowed to bf16 inside the kernel
  // (the bf16-residual pitfall guard).
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) {
      continue;
    }
    auto idx = acc.get_multidimensional_index(i);
    const uint n = n0 + (uint)idx[0];   // dim0 = N-local
    const uint m = m0 + (uint)idx[1];   // dim1 = M-local
    if (m >= (uint)Mi || n >= (uint)Ni) {
      continue;
    }
    const float y_val = s_x[m] * acc[i];
    y[(size_t)m * (size_t)Ni + (size_t)n] = (bfloat16_t)y_val;
  }
)"
