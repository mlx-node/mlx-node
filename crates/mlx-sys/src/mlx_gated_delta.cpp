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

static const char* gated_delta_chunked_source =
    #include "metal/gated_delta_chunked.metal.inc"
;

static const char* gated_delta_fused_gating_source =
    #include "metal/gated_delta_fused_gating.metal.inc"
;

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

/// Chunked gated delta recurrence for prefill (BT=32 tokens per chunk).
/// Accepts native [B, S, Hv, D] layout — no transposes needed.
/// All inputs must have GQA already expanded (Hk == Hv).
bool mlx_gated_delta_chunked(
    mlx_array* q_handle,
    mlx_array* k_handle,
    mlx_array* v_handle,
    mlx_array* g_handle,
    mlx_array* beta_handle,
    mlx_array* state_handle,
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

        // Native layout: q,k [B,S,Hv,Dk], v [B,S,Hv,Dv], g,beta [B,S,Hv], state [B,Hv,Dv,Dk]
        int B  = q_arr.shape(0);
        int S  = q_arr.shape(1);
        int Hv = v_arr.shape(2);
        int Dk = q_arr.shape(3);
        int Dv = v_arr.shape(3);

        constexpr int BT = 32;
        int DV_PER_TG = std::min(4, Dv);

        auto input_type = q_arr.dtype();
        auto S_arr = array(S, mlx::core::int32);

        // Pass tensors directly — no transpose, no reshape
        std::vector<array> inputs = {q_arr, k_arr, v_arr, g_arr, beta_arr, state_arr, S_arr};

        static std::mutex chunked_mutex;
        static std::optional<fast::CustomKernelFunction> chunked_kernel;
        {
            std::lock_guard<std::mutex> lock(chunked_mutex);
            if (!chunked_kernel.has_value()) {
                chunked_kernel = fast::metal_kernel(
                    "gated_delta_chunked",
                    {"q", "k", "v", "g", "beta", "state_in", "S"},
                    {"y", "state_out"},
                    gated_delta_chunked_source
                );
            }
        }

        std::vector<std::pair<std::string, fast::TemplateArg>> template_args = {
            {"InT", input_type},
            {"BT", BT},
            {"BK", Dk},
            {"DV_PER_TG", DV_PER_TG},
            {"Dv", Dv},
            {"Hv", Hv},
        };

        auto results = chunked_kernel.value()(
            inputs,
            // Output shapes match kernel's native write layout
            {Shape{B, S, Hv, Dv}, Shape{B, Hv, Dv, Dk}},
            {input_type, mlx::core::float32},
            std::make_tuple(32, Dv, B * Hv),         // Grid: (32, Dv, B*Hv)
            std::make_tuple(32, DV_PER_TG, 1),       // Threadgroup: (32, DV_PER_TG, 1)
            template_args,
            std::nullopt,
            false,
            mlx::core::default_stream(Device::gpu)
        );

        // Outputs already in native layout — just cast state f32 → model dtype
        auto& y_out = results[0];
        auto state_out = astype(results[1], input_type);

        *out_y = reinterpret_cast<mlx_array*>(new array(std::move(y_out)));
        *out_state = reinterpret_cast<mlx_array*>(new array(std::move(state_out)));
        return true;
    } catch (const std::exception& e) {
        std::cerr << "mlx_gated_delta_chunked error: " << e.what() << std::endl;
        *out_y = nullptr;
        *out_state = nullptr;
        return false;
    }
}

/// Fused GDN gating: computes beta = sigmoid(b) and g = -exp(a_log) * softplus(a + dt_bias).
/// Returns (beta, g) via output pointers.
/// a_log and dt_bias are always f32 (per-head). b, a are InT. beta is InT, g is f32.
bool mlx_fused_gdn_gating(
    mlx_array* b_handle,
    mlx_array* a_handle,
    mlx_array* a_log_handle,
    mlx_array* dt_bias_handle,
    int num_heads,
    int total_elements,
    mlx_array** out_beta,
    mlx_array** out_g
) {
    try {
        auto& b_arr = *reinterpret_cast<array*>(b_handle);
        auto& a_arr = *reinterpret_cast<array*>(a_handle);
        auto& a_log_arr = *reinterpret_cast<array*>(a_log_handle);
        auto& dt_bias_arr = *reinterpret_cast<array*>(dt_bias_handle);

        auto input_type = b_arr.dtype();

        auto total_arr = array(total_elements, mlx::core::int32);
        auto nheads_arr = array(num_heads, mlx::core::int32);

        std::vector<array> inputs = {b_arr, a_arr, a_log_arr, dt_bias_arr, total_arr, nheads_arr};

        static std::mutex gating_mutex;
        static std::optional<fast::CustomKernelFunction> gating_kernel;
        {
            std::lock_guard<std::mutex> lock(gating_mutex);
            if (!gating_kernel.has_value()) {
                gating_kernel = fast::metal_kernel(
                    "fused_gdn_gating",
                    {"b", "a", "a_log", "dt_bias", "total_elements", "num_heads"},
                    {"beta_out", "g_out"},
                    gated_delta_fused_gating_source
                );
            }
        }

        std::vector<std::pair<std::string, fast::TemplateArg>> template_args = {
            {"InT", input_type},
        };

        int threads = 256;
        int groups = (total_elements + threads - 1) / threads;

        auto results = gating_kernel.value()(
            inputs,
            {b_arr.shape(), b_arr.shape()},
            {input_type, mlx::core::float32},  // beta is InT, g is f32
            std::make_tuple(groups * threads, 1, 1),
            std::make_tuple(threads, 1, 1),
            template_args,
            std::nullopt,
            false,
            mlx::core::default_stream(Device::gpu)
        );

        *out_beta = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
        *out_g = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
        return true;
    } catch (const std::exception& e) {
        std::cerr << "mlx_fused_gdn_gating error: " << e.what() << std::endl;
        *out_beta = nullptr;
        *out_g = nullptr;
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

/// Returns the GPU architecture generation number.
/// M1=13, M2=14, M3=15, M4=16, M5=17.
/// Used by Rust to gate chunked GDN kernel on M5+ (Neural Accelerators).
int32_t mlx_gpu_architecture_gen() {
    try {
        auto& info = mlx::core::gpu::device_info(0);
        auto it = info.find("architecture");
        if (it == info.end()) return 0;
        auto& arch = std::get<std::string>(it->second);
        // Architecture string: "applegpu_g15s" → gen=15
        // Gen is the 2nd/3rd-to-last chars before the size letter
        if (arch.size() < 3) return 0;
        int gen = 0;
        // Parse digits before the last character (size letter)
        size_t i = arch.size() - 2;
        int multiplier = 1;
        while (i > 0 && arch[i] >= '0' && arch[i] <= '9') {
            gen += (arch[i] - '0') * multiplier;
            multiplier *= 10;
            i--;
        }
        return gen;
    } catch (...) {
        return 0;
    }
}

}  // extern "C"
