// NA (Neural Accelerator) int8 GEMM body for fast::metal_kernel JIT.
//
// Promoted verbatim from the proven Stage-0 probe (mlx_na_int8_probe.cpp) and
// experiments/flashqla/na_int8/gemm.metal:gemm_body2<int8_t,int32_t,false>.
//
// fast::metal_kernel auto-generates the [[kernel]] signature; the bound device
// pointers are named A, B, C and the scalar int32 inputs M, N, K bind as
// `const constant int&`. The header arg injects:
//   #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
//   using namespace mpp::tensor_ops;
// (metal_stdlib + `using namespace metal` already come from metal::utils()).
//
// Buffer layout (row-major math, matches the host reference):
//   C[M,N] = A[M,K] @ B[K,N]
// In-kernel tensors are column-major via tensor_inline(ptr, extents):
//   A extents (K,M): A[k,m] at ptrA[k + m*K]   (A is M-major rows of length K)
//   B extents (N,K): B[n,k] at ptrB[n + k*N]   (B is K-major rows of length N)
//   C extents (N,M): C[n,m] at ptrC[n + m*N]   (C is M-major rows of length N)
//
// For a [N,K] weight stored row-major as W[n,k] at ptr[n*K + k], we WANT the math
// y[m,n] = sum_k x[m,k] * W[n,k] (i.e. x @ W^T). The caller passes B := the [N,K]
// weight buffer directly; the kernel's B extents (N,K) read B[n,k] = ptr[n + k*N],
// which is NOT W[n,k]. So the caller must hand a buffer whose column-major (N,K)
// view equals W[n,k] — i.e. a K-major [K,N] buffer where buf[k*N + n] = W[n,k].
// In practice the C++ op transposes the [N,K] weight to a [K,N] contiguous buffer
// before dispatch (see mlx_na_int8.cpp). The math then yields C[m,n] = x @ W^T.
R"(
  constexpr int TM2 = 128;
  constexpr int TN2 = 64;
  constexpr int SG2 = 8;

  constexpr auto desc = matmul2d_descriptor(
      TM2, TN2, static_cast<int>(dynamic_extent),
      false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<SG2>> op;

  const int Mi = M;
  const int Ni = N;
  const int Ki = K;

  // CRITICAL: the tensor ELEMENT TYPE must be NON-const int8_t. MPP's matmul2d
  // matches `__is_same_v<leftValueType, int8_t>` exactly; a `const int8_t`
  // element type deduces `leftValueType = const signed char` and FAILS the
  // "Unsupported type" static_assert. MLX binds A/B as `const device int8_t*`,
  // so const_cast to `device int8_t*` before constructing the tensor view.
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      const_cast<device int8_t*>(A), dextents<int32_t, 2>(Ki, Mi));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      const_cast<device int8_t*>(B), dextents<int32_t, 2>(Ni, Ki));
  tensor<device int32_t, dextents<int32_t, 2>, tensor_inline> tC(
      C, dextents<int32_t, 2>(Ni, Mi));

  const uint m0 = threadgroup_position_in_grid.y * TM2;
  const uint n0 = threadgroup_position_in_grid.x * TN2;
  auto mA = tA.slice(0, m0);
  auto mB = tB.slice(n0, 0);
  auto mC = tC.slice(n0, m0);
  op.run(mA, mB, mC);
)"
