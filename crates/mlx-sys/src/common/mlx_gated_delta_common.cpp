#include "mlx_common.h"

namespace mlx_node_internal {
#ifdef MLX_NODE_CUDA_ENABLED
std::optional<mlx::core::array> dflash2_conv_cuda(
    const mlx::core::array& x_arr,
    const mlx::core::array& dyn_arr,
    const mlx::core::array& base_arr,
    int32_t side);
#elif defined(MLX_NODE_METAL_ENABLED)
std::optional<mlx::core::array> dflash2_conv_metal(
    const mlx::core::array& x_arr,
    const mlx::core::array& dyn_arr,
    const mlx::core::array& base_arr,
    int32_t side,
    int B,
    int L,
    int H,
    int K,
    int G);
#endif
}  // namespace mlx_node_internal

namespace {
using namespace mlx::core;

static std::vector<array> compute_g_compiled_impl(const std::vector<array>& inputs) {
    const auto& a_log = inputs[0];   // bf16 (pre-cast by Rust loader)
    const auto& a = inputs[1];       // bf16
    const auto& dt_bias = inputs[2]; // bf16
    // All ops in bf16 — no dtype promotion, single fused kernel
    auto A = exp(a_log);
    auto x = a + dt_bias;
    // Numerically stable softplus: where(x > 20, x, log1p(exp(x)))
    // Naive log(exp(x)+1) overflows for large x in bf16/f16 (max ~65504).
    auto sp = where(greater(x, array(20.0f, a.dtype())), x, mlx::core::log1p(exp(x)));
    return {exp(negative(A * sp))};
}

static auto& get_compiled_compute_g() {
    static auto fn = mlx::core::compile(compute_g_compiled_impl, /* shapeless= */ true);
    return fn;
}

}  // anonymous namespace

extern "C" {

/// Compiled compute_g: g = exp(-exp(A_log.f32) * softplus(a + dt_bias)).astype(a.dtype)
///
/// Uses mlx::core::compile(shapeless=true) to cache the fused kernel graph,
/// matching mlx-lm's @partial(mx.compile, shapeless=True) decorator.
/// Called 30x per decode step (once per linear attention layer).
///
/// Shapes: A_log [Hv] (f32), a [B, T, Hv] (bf16), dt_bias [Hv] (bf16) → g [B, T, Hv] (bf16)
mlx_array* mlx_fused_compute_g(mlx_array* a_log_ptr, mlx_array* a_ptr, mlx_array* dt_bias_ptr) {
    if (!a_log_ptr || !a_ptr || !dt_bias_ptr) {
        std::cerr << "[MLX] mlx_fused_compute_g: null handle" << std::endl;
        return nullptr;
    }
    try {
        using namespace mlx::core;
        auto& a_log = *reinterpret_cast<array*>(a_log_ptr);
        auto& a = *reinterpret_cast<array*>(a_ptr);
        auto& dt_bias = *reinterpret_cast<array*>(dt_bias_ptr);

        auto result = get_compiled_compute_g()({a_log, a, dt_bias});
        return reinterpret_cast<mlx_array*>(new array(std::move(result[0])));
    } catch (const std::exception& e) {
        std::cerr << "[MLX] mlx_fused_compute_g: " << e.what() << std::endl;
        return nullptr;
    }
}

/// Fused grouped dynamic causal conv for the DFlash2 drafter.
///
/// out[b,l,h] = Σ_k (base[side,k,h] + dyn[b,l,side,k,g(h)]) · x[b,l-k,h]
///
/// Replaces the ~13-dispatch elementwise chain (pad/slice/astype/add/mul/add
/// per tap) with a single elementwise kernel; every add/mul rounds through
/// the model dtype inside the kernel, matching the chain bit-for-bit.
///
/// Inputs:
///   x:    [B, L, H]       - hidden states (model dtype)
///   dyn:  [B, L, 2, K, G] - dynamic kernel correction (same dtype)
///   base: [2, K, H]       - base conv kernels (same dtype)
///   side: scalar int32    - 0 = "before" tap set, 1 = "after"
///   L:    scalar int32    - sequence length
///
/// Output: [B, L, H]. Returns true on success; callers fall back to the
/// elementwise chain on false.
bool mlx_dflash2_conv(
    mlx_array* x_handle,
    mlx_array* dyn_handle,
    mlx_array* base_handle,
    int32_t side,
    mlx_array** out
) {
    try {
        auto& x_arr = *reinterpret_cast<array*>(x_handle);
        auto& dyn_arr = *reinterpret_cast<array*>(dyn_handle);
        auto& base_arr = *reinterpret_cast<array*>(base_handle);
        if (side < 0 || side > 1 || x_arr.ndim() != 3 || dyn_arr.ndim() != 5
            || base_arr.ndim() != 3) {
            throw std::invalid_argument("mlx_dflash2_conv: bad inputs");
        }
        if (x_arr.dtype() != dyn_arr.dtype() || x_arr.dtype() != base_arr.dtype()) {
            throw std::invalid_argument("mlx_dflash2_conv: dtype mismatch");
        }
        int B = x_arr.shape(0);
        int L = x_arr.shape(1);
        int H = x_arr.shape(2);
        int K = base_arr.shape(1);
        int G = dyn_arr.shape(4);
        if (H % G != 0 || dyn_arr.shape(0) != B || dyn_arr.shape(1) != L
            || dyn_arr.shape(2) != 2 || dyn_arr.shape(3) != K
            || base_arr.shape(0) != 2 || base_arr.shape(2) != H) {
            throw std::invalid_argument("mlx_dflash2_conv: inconsistent tensor dims");
        }

#ifdef MLX_NODE_CUDA_ENABLED
        if (auto result = mlx_node_internal::dflash2_conv_cuda(
                x_arr, dyn_arr, base_arr, side)) {
            *out = reinterpret_cast<mlx_array*>(new array(std::move(*result)));
            return true;
        }
        *out = nullptr;
        return false;
#elif defined(MLX_NODE_METAL_ENABLED)
        if (auto result = mlx_node_internal::dflash2_conv_metal(
                x_arr, dyn_arr, base_arr, side, B, L, H, K, G)) {
            *out = reinterpret_cast<mlx_array*>(new array(std::move(*result)));
            return true;
        }
        *out = nullptr;
        return false;
#else
        *out = nullptr;
        return false;
#endif
    } catch (const std::exception& e) {
        std::cerr << "mlx_dflash2_conv error: " << e.what() << std::endl;
        *out = nullptr;
        return false;
    }
}

}  // extern "C"
