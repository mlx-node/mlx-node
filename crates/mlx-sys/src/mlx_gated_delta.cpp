#include "mlx_common.h"

// Opaque handle for a compiled Metal kernel function
struct mlx_metal_kernel;

// Metal shader sources for the gated delta recurrence, indexed by variant:
//   [0] = non-vectorized, non-masked
//   [1] = non-vectorized, masked
//   [2] = vectorized, non-masked
//   [3] = vectorized, masked
static const char* gated_delta_sources[] = {
    #include "metal/gated_delta_step.metal.inc"
    ,
    #include "metal/gated_delta_step_mask.metal.inc"
    ,
    #include "metal/gated_delta_step_vec.metal.inc"
    ,
    #include "metal/gated_delta_step_vec_mask.metal.inc"
};

// Cache compiled kernels to avoid recompilation
static std::mutex kernel_cache_mutex;
static std::unordered_map<int, mlx::core::fast::CustomKernelFunction> kernel_cache;

static mlx::core::fast::CustomKernelFunction& get_or_create_kernel(bool has_mask, bool vectorized) {
    int key = (has_mask ? 1 : 0) | (vectorized ? 2 : 0);
    std::lock_guard<std::mutex> lock(kernel_cache_mutex);
    auto it = kernel_cache.find(key);
    if (it != kernel_cache.end()) {
        return it->second;
    }

    std::string suffix;
    if (vectorized) suffix += "_vec";
    if (has_mask) suffix += "_mask";

    std::vector<std::string> inputs = {"q", "k", "v", "g", "beta", "state_in", "T"};
    if (has_mask) {
        inputs.push_back("mask");
    }

    auto kernel = fast::metal_kernel(
        "gated_delta_step" + suffix,
        inputs,
        {"y", "state_out"},
        gated_delta_sources[key]
    );

    auto [inserted, success] = kernel_cache.emplace(key, std::move(kernel));
    return inserted->second;
}

extern "C" {

/// Run the gated delta recurrence using a custom Metal kernel.
///
/// Inputs:
///   q: [B, T, Hk, Dk]  - queries (GQA-expanded by caller)
///   k: [B, T, Hk, Dk]  - keys (GQA-expanded by caller)
///   v: [B, T, Hv, Dv]  - values
///   g: [B, T, Hv]       - decay gate (non-vectorized for Qwen3.5)
///   beta: [B, T, Hv]    - beta (sigmoid already applied by caller)
///   state: [B, Hv, Dv, Dk] - recurrent state
///   mask: [B, T] or nullptr - optional boolean mask
///
/// Outputs (returned via out_y, out_state):
///   y: [B, T, Hv, Dv]         - output
///   state_out: [B, Hv, Dv, Dk] - updated state
///
/// Returns true on success.
bool mlx_gated_delta_kernel(
    mlx_array* q_handle,
    mlx_array* k_handle,
    mlx_array* v_handle,
    mlx_array* g_handle,
    mlx_array* beta_handle,
    mlx_array* state_handle,
    mlx_array* mask_handle,  // nullptr if no mask
    mlx_array** out_y,
    mlx_array** out_state
) {
    try {
        auto& q_arr = *reinterpret_cast<array*>(q_handle);
        auto& k_arr = *reinterpret_cast<array*>(k_handle);
        auto& v_arr = *reinterpret_cast<array*>(v_handle);
        auto& g_arr = *reinterpret_cast<array*>(g_handle);
        auto& beta_arr = *reinterpret_cast<array*>(beta_handle);
        auto& state_arr = *reinterpret_cast<array*>(state_handle);

        bool has_mask = (mask_handle != nullptr);
        bool vectorized = (g_arr.ndim() == 4);

        int B = q_arr.shape(0);
        int T = q_arr.shape(1);
        int Hk = q_arr.shape(2);
        int Dk = q_arr.shape(3);
        int Hv = v_arr.shape(2);
        int Dv = v_arr.shape(3);

        auto input_type = q_arr.dtype();

        // T as a scalar array (int32)
        auto T_arr = array(T, mlx::core::int32);

        // Build input list
        std::vector<array> inputs = {q_arr, k_arr, v_arr, g_arr, beta_arr, state_arr, T_arr};
        if (has_mask) {
            inputs.push_back(*reinterpret_cast<array*>(mask_handle));
        }

        // Template args: InT (dtype), Dk, Dv, Hk, Hv
        std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
            {"InT", input_type},
            {"Dk", Dk},
            {"Dv", Dv},
            {"Hk", Hk},
            {"Hv", Hv},
        };

        auto& kernel = get_or_create_kernel(has_mask, vectorized);

        auto results = kernel(
            inputs,
            {Shape{B, T, Hv, Dv}, state_arr.shape()},  // output_shapes
            {input_type, input_type},                     // output_dtypes
            std::make_tuple(32, Dv, B * Hv),             // grid
            std::make_tuple(32, 4, 1),                    // threadgroup
            template_args,
            std::nullopt,                                 // init_value
            false,                                        // verbose
            mlx::core::default_stream(mlx::core::Device::gpu)
        );

        *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
        *out_state = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
        return true;
    } catch (const std::exception& e) {
        std::cerr << "mlx_gated_delta_kernel error: " << e.what() << std::endl;
        *out_y = nullptr;
        *out_state = nullptr;
        return false;
    }
}

/// Compiled compute_g: g = exp(-exp(A_log.f32) * softplus(a + dt_bias)).astype(a.dtype)
///
/// Uses mlx::core::compile(shapeless=true) to cache the fused kernel graph,
/// matching mlx-lm's @partial(mx.compile, shapeless=True) decorator.
/// Called 30x per decode step (once per linear attention layer).
///
/// Shapes: A_log [Hv] (f32), a [B, T, Hv] (bf16), dt_bias [Hv] (bf16) → g [B, T, Hv] (bf16)

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

}  // extern "C"
