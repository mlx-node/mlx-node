#include "mlx_common.h"

// Opaque handle for a compiled Metal kernel function
struct mlx_metal_kernel;

// Build the Metal shader source for the gated delta recurrence.
// has_mask: whether to include mask-checking logic
// vectorized: whether g has per-element (4D) or per-head (3D) shape
static std::string build_gated_delta_source(bool has_mask, bool vectorized) {
    const char* mask_source = has_mask ? "mask[b_idx * T + t]" : "true";
    const char* g_comment;
    const char* g_setup;
    const char* g_access;
    const char* g_advance;

    if (vectorized) {
        g_comment = "// g: [B, T, Hv, Dk]";
        g_setup   = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;";
        g_access  = "g_[s_idx]";
        g_advance = "g_ += Hv * Dk;";
    } else {
        g_comment = "// g: [B, T, Hv]";
        g_setup   = "auto g_ = g + b_idx * T * Hv;";
        g_access  = "g_[hv_idx]";
        g_advance = "g_ += Hv;";
    }

    std::string src;
    src += R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }
)";
    src += "        "; src += g_comment; src += "\n";
    src += "        "; src += g_setup; src += "\n";
    src += R"(        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if ()";
    src += mask_source;
    src += R"() {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * )";
    src += g_access;
    src += R"(;
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          )";
    src += g_advance;
    src += R"(
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )";
    return src;
}

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
        build_gated_delta_source(has_mask, vectorized)
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

/// Fused compute_g: g = exp(-exp(A_log) * softplus(a + dt_bias))
///
/// Combines 6 separate ops into a single C++ call so MLX's graph optimizer
/// can see the full expression. Avoids 6 separate FFI round-trips.
///
/// Shapes: A_log [Hv], a [B, T, Hv], dt_bias [Hv] → g [B, T, Hv]
mlx_array* mlx_fused_compute_g(mlx_array* a_log_ptr, mlx_array* a_ptr, mlx_array* dt_bias_ptr) {
    using namespace mlx::core;
    auto& a_log = *reinterpret_cast<array*>(a_log_ptr);
    auto& a = *reinterpret_cast<array*>(a_ptr);
    auto& dt_bias = *reinterpret_cast<array*>(dt_bias_ptr);

    // softplus(a + dt_bias) = max(x,0) + log1p(exp(-|x|))  (numerically stable)
    auto x = add(a, dt_bias, {});
    // Use input dtype for zero to avoid f32 promotion with bf16 inputs
    auto zero = zeros({}, x.dtype());
    auto max_x_0 = maximum(x, zero, {});
    auto abs_x = abs(x, {});
    auto neg_abs = negative(abs_x, {});
    auto sp = add(max_x_0, log1p(exp(neg_abs, {}), {}), {});

    // g = exp(-exp(A_log) * sp)
    auto a_exp = exp(a_log, {});
    auto neg_a_exp = negative(a_exp, {});
    auto exponent = multiply(neg_a_exp, sp, {});
    auto g = exp(exponent, {});

    return reinterpret_cast<mlx_array*>(new array(std::move(g)));
}

}  // extern "C"
