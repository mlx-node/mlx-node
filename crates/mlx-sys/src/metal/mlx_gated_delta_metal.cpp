#include "mlx_common.h"

#ifdef MLX_NODE_METAL_ENABLED
namespace {

// Parameterized Metal shader source for the gated delta recurrence; HAS_MASK
// and VEC_G are injected via the metal_kernel header for each (mask, vec)
// variant.
static const char* gated_delta_step_source =
    #include "metal/common/gated_delta_step.metal.inc"
;

// E47 (catalog D1): per-step kernel with 2 v-columns per simdgroup.
// Same shape contract as HAS_MASK=0/VEC_G=0 but each simdgroup processes
// dv_A=2y and dv_B=2y+1, sharing q[Dk] + k[Dk] loads.
// Grid Y must be halved by the dispatcher.
static const char* gated_delta_step_2vcol_source =
    #include "metal/common/gated_delta_step_2vcol.metal.inc"
;

// Qwen4 per-step kernel with 4 v-columns per simdgroup.
// Extends E47 by another factor. Grid Y must be quartered by the dispatcher.
static const char* gated_delta_step_4vcol_source =
    #include "metal/common/gated_delta_step_4vcol.metal.inc"
;

static const char* gated_delta_fused_gating_source =
    #include "metal/common/gated_delta_fused_gating.metal.inc"
;

// Fused accepted-prefix replay for the eager-MTP tape: one dispatch walks all
// accepted tokens, rounding the recurrent state through its own dtype (StT)
// after EVERY token so the result is bit-identical to chaining the per-step
// kernel at T=1.
static const char* gated_delta_replay_source =
    #include "metal/common/gated_delta_replay.metal.inc"
;

// Cache compiled kernels to avoid recompilation
static std::mutex kernel_cache_mutex;
static std::unordered_map<int, mlx::core::fast::CustomKernelFunction> kernel_cache;

// per_step_variant: 0=legacy, 1=2-vcol, 2=4-vcol.
static mlx::core::fast::CustomKernelFunction& get_or_create_kernel(
    bool has_mask, bool vectorized, int per_step_variant) {
    int key = (has_mask ? 1 : 0) | (vectorized ? 2 : 0) | (per_step_variant << 2);
    std::lock_guard<std::mutex> lock(kernel_cache_mutex);
    auto it = kernel_cache.find(key);
    if (it != kernel_cache.end()) {
        return it->second;
    }

    std::string suffix;
    if (vectorized) suffix += "_vec";
    if (has_mask) suffix += "_mask";
    if (per_step_variant == 1) suffix += "_2v";
    else if (per_step_variant == 2) suffix += "_4v";

    std::vector<std::string> inputs = {"q", "k", "v", "g", "beta", "state_in", "T"};
    if (has_mask) {
        inputs.push_back("mask");
    }

    const char* src;
    std::string header;
    if (per_step_variant == 1) {
        src = gated_delta_step_2vcol_source;
    } else if (per_step_variant == 2) {
        src = gated_delta_step_4vcol_source;
    } else {
        src = gated_delta_step_source;
        header = std::string("#define HAS_MASK ") + (has_mask ? "1\n" : "0\n") +
                 "#define VEC_G " + (vectorized ? "1\n" : "0\n");
    }

    auto kernel = fast::metal_kernel(
        "gated_delta_step" + suffix,
        inputs,
        {"y", "state_out"},
        src,
        header
    );

    auto [inserted, success] = kernel_cache.emplace(key, std::move(kernel));
    return inserted->second;
}

/// Run the gated delta recurrence using a custom Metal kernel.
///
/// Inputs:
///   q: [B, T, Hk, Dk]  - queries (expanded, or compact tiled-GGUF heads)
///   k: [B, T, Hk, Dk]  - keys (expanded, or compact tiled-GGUF heads)
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
static bool gated_delta_kernel_impl(
    mlx_array* q_handle,
    mlx_array* k_handle,
    mlx_array* v_handle,
    mlx_array* g_handle,
    mlx_array* beta_handle,
    mlx_array* state_handle,
    mlx_array* mask_handle,  // nullptr if no mask
    mlx_array** out_y,
    mlx_array** out_state,
    bool prefer_four,
    bool float_output = false
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

        // `y` follows the activation dtype; the recurrent state keeps the
        // dtype it was carried in (`StT`, f32 in production — never rounded
        // on store). Qwen4's `float_output` widens both to FP32 regardless
        // of the BF16 prompt storage.
        auto input_type = float_output ? mlx::core::float32 : q_arr.dtype();
        auto state_type = float_output ? mlx::core::float32 : state_arr.dtype();

        // T as a scalar array (int32)
        auto T_arr = array(T, mlx::core::int32);

        // Build input list
        std::vector<array> inputs = {q_arr, k_arr, v_arr, g_arr, beta_arr, state_arr, T_arr};
        if (has_mask) {
            inputs.push_back(*reinterpret_cast<array*>(mask_handle));
        }

        // Template args: InT (activation dtype), StT (state dtype), Dk, Dv, Hk, Hv
        std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
            {"InT", input_type},
            {"StT", state_type},
            {"Dk", Dk},
            {"Dv", Dv},
            {"Hk", Hk},
            {"Hv", Hv},
        };

        // Use two v-columns when eligible; Qwen4 can prefer four columns.
        int per_step_variant = 0;
        bool elig = !has_mask && !vectorized;
        if (elig) {
            if (prefer_four && Dv % 4 == 0) {
                per_step_variant = 2;
            } else if (Dv % 2 == 0) {
                per_step_variant = 1;
            }
        }
        auto& kernel = get_or_create_kernel(has_mask, vectorized, per_step_variant);

        int grid_y = Dv;
        if (per_step_variant == 1) grid_y = Dv / 2;
        else if (per_step_variant == 2) grid_y = Dv / 4;

        auto results = kernel(
            inputs,
            {Shape{B, T, Hv, Dv}, state_arr.shape()},  // output_shapes
            {input_type, state_type},                     // output_dtypes
            std::make_tuple(32, grid_y, B * Hv),         // grid
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

}  // namespace
#endif  // MLX_NODE_METAL_ENABLED

extern "C" {

/// Fused GDN gating: beta = sigmoid(b), g = exp(-exp(a_log) * softplus(a + dt_bias)).
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
#ifdef MLX_NODE_METAL_ENABLED
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
#else
    (void)b_handle;
    (void)a_handle;
    (void)a_log_handle;
    (void)dt_bias_handle;
    (void)num_heads;
    (void)total_elements;
    if (out_beta) *out_beta = nullptr;
    if (out_g) *out_g = nullptr;
    return false;
#endif
}

/// Returns the Apple GPU architecture generation number (Metal only).
/// M1=13, M2=14, M3=15, M4=16, M5=17. Returns 0 off Metal.
/// Shared by runtime capability checks and profiling.
int32_t mlx_gpu_architecture_gen() {
#ifdef MLX_NODE_METAL_ENABLED
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
#else
    return 0;
#endif
}

bool mlx_gated_delta_kernel(mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g,
    mlx_array* beta, mlx_array* state, mlx_array* mask, mlx_array** out_y, mlx_array** out_state) {
#ifdef MLX_NODE_METAL_ENABLED
    return gated_delta_kernel_impl(q,k,v,g,beta,state,mask,out_y,out_state,false);
#else
    (void)q; (void)k; (void)v; (void)g; (void)beta; (void)state; (void)mask;
    if (out_y) *out_y = nullptr;
    if (out_state) *out_state = nullptr;
    return false;
#endif
}

/// Fused accepted-prefix replay for the eager-MTP GDN tape.
///
/// Replays `replay_steps` tokens of the recorded verify window in ONE dispatch,
/// rounding the fp32 register state through the state dtype after every token
/// so the result is bit-identical to chaining `mlx_gated_delta_kernel` at T=1
/// (identity for the production f32 state). The recurrent update never reads
/// `q`, so the query input and `y` output do not exist here.
///
/// Inputs:
///   k: [B, S, Hk, Dk]  - recorded window keys (S = full window stride)
///   v: [B, S, Hv, Dv]  - recorded window values
///   g: [B, S, Hv]       - recorded decay gate (post-exp, f32)
///   beta: [B, S, Hv]    - recorded beta (post-sigmoid)
///   state: [B, Hv, Dv, Dk] - pre-verify snapshot state (f32 in production)
///   replay_steps: accepted prefix length (T loop bound, <= S)
///   window_stride: recorded window length S (batch/time striding)
///
/// Output (via out_state): [B, Hv, Dv, Dk] in the state dtype.
/// Returns true on success.
bool mlx_gated_delta_replay(
    mlx_array* k_handle,
    mlx_array* v_handle,
    mlx_array* g_handle,
    mlx_array* beta_handle,
    mlx_array* state_handle,
    int32_t replay_steps,
    int32_t window_stride,
    mlx_array** out_state
) {
#ifdef MLX_NODE_METAL_ENABLED
    try {
        auto& k_arr = *reinterpret_cast<array*>(k_handle);
        auto& v_arr = *reinterpret_cast<array*>(v_handle);
        auto& g_arr = *reinterpret_cast<array*>(g_handle);
        auto& beta_arr = *reinterpret_cast<array*>(beta_handle);
        auto& state_arr = *reinterpret_cast<array*>(state_handle);

        if (k_arr.ndim() != 4 || v_arr.ndim() != 4 || state_arr.ndim() != 4
            || g_arr.ndim() != 3 || beta_arr.ndim() != 3
            || replay_steps < 0 || replay_steps > window_stride) {
            throw std::invalid_argument("mlx_gated_delta_replay: bad inputs");
        }
        int B = v_arr.shape(0);
        int Hk = k_arr.shape(2);
        int Dk = k_arr.shape(3);
        int Hv = v_arr.shape(2);
        int Dv = v_arr.shape(3);
        if (Dk % 32 != 0
            || k_arr.shape(0) != B || k_arr.shape(1) != window_stride
            || v_arr.shape(1) != window_stride
            || g_arr.shape(0) != B || g_arr.shape(1) != window_stride || g_arr.shape(2) != Hv
            || beta_arr.shape(0) != B || beta_arr.shape(1) != window_stride
            || beta_arr.shape(2) != Hv
            || state_arr.shape(0) != B || state_arr.shape(1) != Hv
            || state_arr.shape(2) != Dv || state_arr.shape(3) != Dk) {
            throw std::invalid_argument("mlx_gated_delta_replay: inconsistent tensor dims");
        }

        // The per-token state store dtype — the T=1 chain rounds through the
        // carried state's dtype (`StT`) each call; f32 in production, so the
        // round is the identity and the replay equals the windowed kernel.
        auto state_type = state_arr.dtype();

        auto T_arr = array(replay_steps, mlx::core::int32);
        auto S_arr = array(window_stride, mlx::core::int32);

        std::vector<array> inputs = {k_arr, v_arr, g_arr, beta_arr, state_arr, T_arr, S_arr};

        std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
            {"StT", state_type},
            {"Dk", Dk},
            {"Dv", Dv},
            {"Hk", Hk},
            {"Hv", Hv},
        };

        static std::mutex replay_mutex;
        static std::optional<fast::CustomKernelFunction> replay_kernel;
        {
            std::lock_guard<std::mutex> lock(replay_mutex);
            if (!replay_kernel.has_value()) {
                replay_kernel = fast::metal_kernel(
                    "gated_delta_replay",
                    {"k", "v", "g", "beta", "state_in", "T", "S"},
                    {"state_out"},
                    gated_delta_replay_source
                );
            }
        }

        auto results = replay_kernel.value()(
            inputs,
            {state_arr.shape()},
            {state_type},
            std::make_tuple(32, Dv, B * Hv),          // Grid: same lane map as per-step
            std::make_tuple(32, 4, 1),                // Threadgroup
            template_args,
            std::nullopt,
            false,
            mlx::core::default_stream(mlx::core::Device::gpu)
        );

        *out_state = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
        return true;
    } catch (const std::exception& e) {
        std::cerr << "mlx_gated_delta_replay error: " << e.what() << std::endl;
        *out_state = nullptr;
        return false;
    }
#else
    (void)k_handle;
    (void)v_handle;
    (void)g_handle;
    (void)beta_handle;
    (void)state_handle;
    (void)replay_steps;
    (void)window_stride;
    if (out_state) *out_state = nullptr;
    return false;
#endif
}

// Qwen4's measured M5 shape benefits from four value rows per SIMD group.
// Other model families retain the existing default.
bool mlx_qwen4_gated_delta_kernel(mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g,
    mlx_array* beta, mlx_array* state, mlx_array* mask, mlx_array** out_y, mlx_array** out_state) {
#ifdef MLX_NODE_METAL_ENABLED
    if (!q || !v) return false;
    auto& query = *reinterpret_cast<array*>(q);
    auto& value = *reinterpret_cast<array*>(v);
    static const int arch = mlx_gpu_architecture_gen();
    bool measured_shape = arch >= 17
        && query.ndim() == 4 && value.ndim() == 4 && query.shape(0) == 1
        && query.shape(3) == 128 && value.shape(2) == 48 && value.shape(3) == 128
        && (query.dtype() == mlx::core::float32 || query.dtype() == mlx::core::bfloat16);
    return gated_delta_kernel_impl(q,k,v,g,beta,state,mask,out_y,out_state,measured_shape,true);
#else
    (void)q; (void)k; (void)v; (void)g; (void)beta; (void)state; (void)mask;
    if (out_y) *out_y = nullptr;
    if (out_state) *out_state = nullptr;
    return false;
#endif
}

}  // extern "C"
