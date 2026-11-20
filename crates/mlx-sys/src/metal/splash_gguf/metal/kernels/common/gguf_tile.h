// Adapted from Splash 1.6.1 (https://github.com/incoai/splash) d8af1054aab0f97de21d043e01aa98af3754f10e,
// runtime/metal/kernels/common/gguf_tile.h. Apache-2.0; see THIRD_PARTY_NOTICES. Changed: the residual and gate
// epilogues are mlx-node's separate ops on the bf16 sum (crates/mlx-sys/src/metal/kquant/kquant_nax.h store
// epilogues) in place of Splash's fp32 ones; they exist only in the -fno-fast-math build of the file
// (GGUF_PLANES_STORE_EPILOGUES), with MLX's own functors. Splash's fp32 residual and gate|up epilogues stay as
// test-only kernels (EpResidualFp32, gguf_swiglu_fp32) that the bf16 epilogue tests must tell apart.
#pragma once
#include <metal_stdlib>
#if GGUF_PLANES_STORE_EPILOGUES
// MLX's headers define these with external linkage; kquant_nax.h's air in paged_attn.metallib defines them too, and
// a second definition fails the metallib link. Renamed here; the epilogues call none of them.
#define expm1f_scaled_unchecked gguf_planes_expm1f_scaled_unchecked
#define expm1f gguf_planes_expm1f
#define erf gguf_planes_erf
#define erfinv gguf_planes_erfinv
#define ArcCos GgufPlanesArcCos
#define ArcSin GgufPlanesArcSin
#define ArcTan GgufPlanesArcTan
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/unary_ops.h"
#undef expm1f_scaled_unchecked
#undef expm1f
#undef erf
#undef erfinv
#undef ArcCos
#undef ArcSin
#undef ArcTan
#endif
using namespace metal;

enum GgufEpilogue : ushort { EpNone, EpResidual, EpUpWithGate, EpResidualFp32 };

#if GGUF_PLANES_STORE_EPILOGUES
// The compiled `sigmoid(gate) * gate * up` of mlx_qwen35_common.h, every intermediate a bf16, by MLX's functors.
inline bfloat gguf_swiglu(bfloat gate, bfloat up) {
#pragma clang fp reassociate(off)
  return ::Multiply{}(::Multiply{}(::Sigmoid{}(gate), gate), up);
}

// Test-only: Splash's fp32 gate|up store, bf16(up) * silu(bf16(gate)) in fp32 (activation.h splash_silu).
inline bfloat gguf_swiglu_fp32(float gate, float up) {
  const float g = float(bfloat(gate));
  return bfloat(float(bfloat(up)) * (g / (1.0f + fast::exp2(-1.44269504089f * g))));
}
#endif

// The output of sum v at element `at`: bf16(v); or, of the bf16-rounded sum y, Add(residual aux[at], y) or the
// swiglu of gate aux[at] and up y.
template <GgufEpilogue Ep, class Out = bfloat> inline Out gguf_epilogue(float v, device const bfloat *aux, ulong at) {
  if constexpr (Ep == EpNone) {
    return Out(v);
  } else {
#if GGUF_PLANES_STORE_EPILOGUES
    const bfloat y = bfloat(v);
    if constexpr (Ep == EpResidual) return ::Add{}(bfloat(aux[at]), y);
    else if constexpr (Ep == EpResidualFp32) return Out(v + float(aux[at]));
    else return gguf_swiglu(bfloat(aux[at]), y);
#else
    static_assert(Ep == EpNone, "store epilogues are built with GGUF_PLANES_STORE_EPILOGUES only");
    return Out(v);
#endif
  }
}
