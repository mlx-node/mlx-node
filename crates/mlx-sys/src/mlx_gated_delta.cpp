#include "mlx_common.h"

// Opaque handle for a compiled Metal kernel function
struct mlx_metal_kernel;

// Metal shader sources for the gated delta recurrence, indexed by variant:
//   [0] = non-vectorized, non-masked
//   [1] = non-vectorized, masked
//   [2] = vectorized, non-masked
//   [3] = vectorized, masked
static const char* gated_delta_sources[] = {
    #include "metal/common/gated_delta_step.metal.inc"
    ,
    #include "metal/common/gated_delta_step_mask.metal.inc"
    ,
    #include "metal/common/gated_delta_step_vec.metal.inc"
    ,
    #include "metal/common/gated_delta_step_vec_mask.metal.inc"
};

static const char* gated_delta_chunked_source =
    #include "metal/common/gated_delta_chunked.metal.inc"
;

// E47 (catalog D1): per-step kernel with 2 v-columns per simdgroup.
// Same shape contract as gated_delta_sources[0] (non-vec, non-mask) but each
// simdgroup processes dv_A=2y and dv_B=2y+1, sharing q[Dk] + k[Dk] loads.
// Grid Y must be halved by the dispatcher.
static const char* gated_delta_step_2vcol_source =
    #include "metal/common/gated_delta_step_2vcol.metal.inc"
;

// E48: per-step kernel with 4 v-columns per simdgroup.
// Extends E47 by another factor. Grid Y must be quartered by the dispatcher.
static const char* gated_delta_step_4vcol_source =
    #include "metal/common/gated_delta_step_4vcol.metal.inc"
;

static const char* gated_delta_step_4vcol_vector_source =
    #include "metal/common/gated_delta_step_4vcol_vector.metal.inc"
;

static const char* gated_delta_fused_gating_source =
    #include "metal/common/gated_delta_fused_gating.metal.inc"
;

// Fused accepted-prefix replay for the eager-MTP tape: one dispatch walks all
// accepted tokens, rounding the recurrent state through InT after EVERY token
// so the result is bit-identical to chaining the per-step kernel at T=1.
static const char* gated_delta_replay_source =
    #include "metal/common/gated_delta_replay.metal.inc"
;

// Cache compiled kernels to avoid recompilation
static std::mutex kernel_cache_mutex;
static std::unordered_map<int, mlx::core::fast::CustomKernelFunction> kernel_cache;

// per_step_variant: 0=legacy, 1=2-vcol, 2=4-vcol, 3=4-vcol vector loads with FP32 state.
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
    else if (per_step_variant == 3) suffix += "_4v_vector";

    std::vector<std::string> inputs = {"q", "k", "v", "g", "beta", "state_in", "T"};
    if (per_step_variant == 3) inputs.pop_back();
    if (has_mask) {
        inputs.push_back("mask");
    }

    const char* src;
    if (per_step_variant == 3) {
        src = gated_delta_step_4vcol_vector_source;
    } else if (per_step_variant == 1) {
        src = gated_delta_step_2vcol_source;
    } else if (per_step_variant == 2) {
        src = gated_delta_step_4vcol_source;
    } else {
        src = gated_delta_sources[key & 3];
    }

    auto kernel = fast::metal_kernel(
        "gated_delta_step" + suffix,
        inputs,
        {"y", "state_out"},
        src
    );

    auto [inserted, success] = kernel_cache.emplace(key, std::move(kernel));
    return inserted->second;
}

extern "C" {

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

        // Qwen's BF16 prompt storage is widened only in registers. Its output
        // and persistent recurrence remain FP32, independent of input storage.
        auto input_type = float_output ? mlx::core::float32 : q_arr.dtype();

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

        // per_step_variant: default = E47 (2 v-cols). Opt-in:
        //   MLX_ENABLE_E48_GDN_4VCOL=1 → 4 v-cols (E48 experimental).
        //   MLX_DISABLE_E47_GDN_2VCOL=1 → legacy 1 v-col (overrides E48).
        int per_step_variant = 0;
        bool elig = !has_mask && !vectorized;
        if (elig && std::getenv("MLX_DISABLE_E47_GDN_2VCOL") == nullptr) {
            if ((prefer_four || std::getenv("MLX_ENABLE_E48_GDN_4VCOL") != nullptr) && Dv % 4 == 0) {
                per_step_variant = 2;
            } else if (Dv % 2 == 0) {
                per_step_variant = 1;
            }
        }
        // Reference data-motion port only. Keep other shapes and model families
        // on their current kernels; the custom kernel materializes contiguous
        // input views before the aligned vector loads.
        auto vector_rows = std::getenv("MLX_QWEN4_GDN_VECTOR_ROWS");
        if (float_output && prefer_four && per_step_variant == 2 && T > 8
            && Dk == 128 && vector_rows && std::string(vector_rows) == "1"
            && q_arr.dtype() == k_arr.dtype() && q_arr.dtype() == v_arr.dtype()
            && state_arr.dtype() == mlx::core::float32
            && g_arr.dtype() == mlx::core::float32
            && beta_arr.dtype() == mlx::core::float32) {
            per_step_variant = 3;
            inputs.pop_back();
            template_args.emplace_back("Q", q_arr.dtype());
            template_args.emplace_back("T", T);
        }
        auto& kernel = get_or_create_kernel(has_mask, vectorized, per_step_variant);

        int grid_y = Dv;
        if (per_step_variant == 1) grid_y = Dv / 2;
        else if (per_step_variant >= 2) grid_y = Dv / 4;

        auto results = kernel(
            inputs,
            {Shape{B, T, Hv, Dv}, state_arr.shape()},  // output_shapes
            {input_type, input_type},                     // output_dtypes
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

bool mlx_gated_delta_kernel(mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g,
    mlx_array* beta, mlx_array* state, mlx_array* mask, mlx_array** out_y, mlx_array** out_state) {
    return gated_delta_kernel_impl(q,k,v,g,beta,state,mask,out_y,out_state,false);
}

/// Fused accepted-prefix replay for the eager-MTP GDN tape.
///
/// Replays `replay_steps` tokens of the recorded verify window in ONE dispatch,
/// rounding the fp32 register state through `InT` after every token so the
/// result is bit-identical to chaining `mlx_gated_delta_kernel` at T=1 — the
/// exact autoregressive state rounding the rollback contract requires. The
/// recurrent update never reads `q`, so the query input and `y` output do not
/// exist here.
///
/// Inputs:
///   k: [B, S, Hk, Dk]  - recorded window keys (S = full window stride)
///   v: [B, S, Hv, Dv]  - recorded window values
///   g: [B, S, Hv]       - recorded decay gate (post-exp, f32)
///   beta: [B, S, Hv]    - recorded beta (post-sigmoid)
///   state: [B, Hv, Dv, Dk] - pre-verify snapshot state (model dtype)
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

        // The per-token state store dtype — the T=1 chain rounds through
        // `input_type = q.dtype()` (the model dtype) each call; the tape
        // records k in that same dtype, so k's dtype reproduces it even for
        // a state snapshot stored at a different precision.
        auto input_type = k_arr.dtype();

        auto T_arr = array(replay_steps, mlx::core::int32);
        auto S_arr = array(window_stride, mlx::core::int32);

        std::vector<array> inputs = {k_arr, v_arr, g_arr, beta_arr, state_arr, T_arr, S_arr};

        std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
            {"InT", input_type},
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
            {input_type},
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
}

static const char* dflash2_conv_source =
#include "metal/common/dflash2_conv.metal.inc"
    ;

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

        auto side_arr = array(side, mlx::core::int32);
        auto L_arr = array(L, mlx::core::int32);
        std::vector<array> inputs = {x_arr, dyn_arr, base_arr, side_arr, L_arr};
        std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
            {"InT", x_arr.dtype()},
            {"H", H},
            {"K", K},
            {"G", G},
        };

        static std::mutex conv_mutex;
        static std::optional<fast::CustomKernelFunction> conv_kernel;
        {
            std::lock_guard<std::mutex> lock(conv_mutex);
            if (!conv_kernel.has_value()) {
                conv_kernel = fast::metal_kernel(
                    "dflash2_conv",
                    {"x", "dyn", "base", "side", "Lp"},
                    {"out"},
                    dflash2_conv_source,
                    "",
                    /* ensure_row_contiguous */ true,
                    /* atomic_outputs */ false);
            }
        }

        auto results = conv_kernel.value()(
            inputs,
            {x_arr.shape()},
            {x_arr.dtype()},
            std::make_tuple(H, L, B),               // grid
            std::make_tuple(256, 1, 1),             // threadgroup
            template_args,
            std::nullopt,
            false,
            mlx::core::default_stream(mlx::core::Device::gpu)
        );
        *out = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
        return true;
    } catch (const std::exception& e) {
        std::cerr << "mlx_dflash2_conv error: " << e.what() << std::endl;
        *out = nullptr;
        return false;
    }
}

// Qwen4's measured M5 shape benefits from four value rows per SIMD group.
// Other model families retain the existing default and environment controls.
bool mlx_qwen4_gated_delta_kernel(mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g,
    mlx_array* beta, mlx_array* state, mlx_array* mask, mlx_array** out_y, mlx_array** out_state) {
    if (!q || !v) return false;
    auto& query = *reinterpret_cast<array*>(q);
    auto& value = *reinterpret_cast<array*>(v);
    static const int arch = mlx_gpu_architecture_gen();
    auto setting = std::getenv("MLX_QWEN4_GDN_4ROWS");
    bool measured_shape = arch >= 17 && (!setting || std::string(setting) != "0")
        && query.ndim() == 4 && value.ndim() == 4 && query.shape(0) == 1
        && query.shape(3) == 128 && value.shape(2) == 48 && value.shape(3) == 128
        && (query.dtype() == mlx::core::float32 || query.dtype() == mlx::core::bfloat16);
    return gated_delta_kernel_impl(q,k,v,g,beta,state,mask,out_y,out_state,measured_shape,true);
}

}  // extern "C"
