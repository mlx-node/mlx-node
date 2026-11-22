#include "mlx_common.h"
#include <cstdio>

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace {

constexpr char dflash2_conv_cuda_source[] =
#include "cuda/dflash2_conv.cu.inc"
    ;

class DFlash2ConvCuda final : public mlx::core::UnaryPrimitive {
 public:
  DFlash2ConvCuda(mlx::core::Stream stream, int32_t side)
      : mlx::core::UnaryPrimitive(stream), side_(side) {}

  void eval_cpu(const std::vector<mlx::core::array>&, mlx::core::array&) override {
    throw std::runtime_error("DFlash2ConvCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      mlx::core::array& out) override {
    using namespace mlx::core;
    auto& encoder = cu::get_command_encoder(stream());
    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    const auto& x = owned[0];
    const auto& dyn = owned[1];
    const auto& base = owned[2];
    const int B = x.shape(0);
    const int L = x.shape(1);
    const int H = x.shape(2);
    const int K = base.shape(1);
    const int G = dyn.shape(4);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(x);
    encoder.set_input_array(dyn);
    encoder.set_input_array(base);
    encoder.set_output_array(out);

    const char* entry = nullptr;
    if (x.dtype() == bfloat16) {
      entry = "mlx_dflash2_conv_cuda_kernel<nv_bfloat16>";
    } else if (x.dtype() == float16) {
      entry = "mlx_dflash2_conv_cuda_kernel<__half>";
    } else if (x.dtype() == float32) {
      entry = "mlx_dflash2_conv_cuda_kernel<float>";
    } else {
      throw std::invalid_argument("DFlash2ConvCuda: unsupported dtype");
    }
    auto kernel = cu::jit_kernel(encoder.device(), dflash2_conv_cuda_source, entry);
    cu::KernelArgs args;
    args.append(x);
    args.append(dyn);
    args.append(base);
    args.append(out);
    args.append<int32_t>(B);
    args.append<int32_t>(L);
    args.append<int32_t>(H);
    args.append<int32_t>(K);
    args.append<int32_t>(G);
    args.append<int32_t>(side_);
    encoder.add_kernel_node_raw(
        kernel, dim3((H + 255) / 256, L, B), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(DFlash2ConvCuda)
  bool is_equivalent(const mlx::core::Primitive& other) const override {
    const auto* rhs = dynamic_cast<const DFlash2ConvCuda*>(&other);
    return rhs && rhs->side_ == side_;
  }
  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int32_t side_;
};

constexpr char gdn_prepare_tree_cuda_source[] = R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ float gdn_prep_tree_warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ nv_bfloat16 gdn_prep_tree_bf16_silu(nv_bfloat16 raw) {
  const float f = __bfloat162float(raw);
  const float e = expf(fabsf(f));
  const float y = 1.0f / (1.0f + e);
  const float sig = (f < 0.0f) ? y : (1.0f - y);
  const nv_bfloat16 sig_bf = __float2bfloat16(sig);
  return __float2bfloat16(__bfloat162float(raw) * __bfloat162float(sig_bf));
}

__device__ __forceinline__ float gdn_prep_tree_precise_sigmoid_f32(float x) {
  const float y = 1.0f / (1.0f + expf(fabsf(x)));
  return (x < 0.0f) ? y : (1.0f - y);
}

// Tree-conv1d variant of fused Qwen3.5/3.8 GDN preparation:
// 4-tap depthwise conv1d indexed by conv_taps[token, 0..3] + SiLU + q|k|v split
// + q/k RMS/L2 norm+scale + decay & beta gating + next conv history.
// For tap d in 0..3, src = conv_taps[token * 4 + d]:
//   src < 0  -> history[3 + src, ch]  (-3 -> 0, -2 -> 1, -1 -> 2; < -3 -> 0.0)
//   src >= 0 -> qkv[src, ch]
// For a 1D chain (conv_taps[t] = [t-3, t-2, t-1, t]), bit-identical (0 ULP)
// to mlx_gdn_prepare_cuda_kernel.
// Grid: dim3(20, T, 1), Block: dim3(32, 4, 1)
template <bool A_BF16, bool B_BF16, bool BETA_BF16>
__global__ __launch_bounds__(128, 4) void mlx_gdn_prepare_tree_cuda_kernel(
    const nv_bfloat16* __restrict__ qkv,
    const void* __restrict__ a_ptr,
    const void* __restrict__ b_ptr,
    const float* __restrict__ conv,
    const nv_bfloat16* __restrict__ history,
    const float* __restrict__ scale,
    const float* __restrict__ dt,
    const int* __restrict__ conv_taps,
    nv_bfloat16* __restrict__ q_out,
    nv_bfloat16* __restrict__ k_out,
    nv_bfloat16* __restrict__ v_out,
    float* __restrict__ decay_out,
    void* __restrict__ beta_out,
    nv_bfloat16* __restrict__ next_history,
    int T,
    long long qkv_s1, long long qkv_s2,
    long long a_s1, long long a_s2,
    long long b_s1, long long b_s2,
    long long hist_s0, long long hist_s1,
    int mean_eps) {
  constexpr int W = 10240;
  const int lane = threadIdx.x;
  const int vec = blockIdx.x * 4 + threadIdx.y; // 0..79
  const int token = blockIdx.y;                 // 0..T-1
  if (vec >= 80 || token >= T) {
    return;
  }

  int taps[4];
#pragma unroll
  for (int tap = 0; tap < 4; ++tap) {
    taps[tap] = conv_taps[token * 4 + tap];
  }

  float values[4];
  float squared = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int ch = vec * 128 + lane * 4 + i;
    float total = 0.0f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      const int src_idx = taps[tap];
      nv_bfloat16 src;
      if (src_idx < 0) {
        const int hrow = 3 + src_idx;
        src = (hrow >= 0 && hrow < 3)
            ? history[hrow * hist_s0 + ch * hist_s1]
            : __float2bfloat16(0.0f);
      } else {
        src = (src_idx < T)
            ? qkv[src_idx * qkv_s1 + ch * qkv_s2]
            : __float2bfloat16(0.0f);
      }
      total += __bfloat162float(src) * conv[ch * 4 + tap];
    }
    const nv_bfloat16 raw = __float2bfloat16(total);
    const nv_bfloat16 activated = gdn_prep_tree_bf16_silu(raw);
    const float vf = __bfloat162float(activated);
    values[i] = vf;
    squared += vf * vf;

    if (token == 0) {
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        const int source = T + row;
        next_history[row * W + ch] = (source < 3)
            ? history[source * hist_s0 + ch * hist_s1]
            : qkv[(source - 3) * qkv_s1 + ch * qkv_s2];
      }
    }
  }

  if (vec < 32) {
    const float sum_sq = gdn_prep_tree_warp_sum(squared);
    const float eps_term = mean_eps ? (128.0f * 1e-6f) : 1e-6f;
    const float divisor = sqrtf(sum_sq + eps_term);
    const nv_bfloat16 q_scale = __float2bfloat16(0.08838834764831845f);
    const float q_scale_f = __bfloat162float(q_scale);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int ch = (vec & 15) * 128 + lane * 4 + i;
      const nv_bfloat16 normalized = __float2bfloat16(values[i] / divisor);
      if (vec < 16) {
        q_out[token * 2048 + ch] = __float2bfloat16(__bfloat162float(normalized) * q_scale_f);
      } else {
        k_out[token * 2048 + ch] = normalized;
      }
    }
  } else {
    const int base = token * 6144 + (vec - 32) * 128 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v_out[base + i] = __float2bfloat16(values[i]);
    }
  }

  if (vec < 48 && lane == 0) {
    const int h = token * 48 + vec;
    const float a_val = A_BF16
        ? __bfloat162float(reinterpret_cast<const nv_bfloat16*>(a_ptr)[token * a_s1 + vec * a_s2])
        : reinterpret_cast<const float*>(a_ptr)[token * a_s1 + vec * a_s2];
    const float b_val = B_BF16
        ? __bfloat162float(reinterpret_cast<const nv_bfloat16*>(b_ptr)[token * b_s1 + vec * b_s2])
        : reinterpret_cast<const float*>(b_ptr)[token * b_s1 + vec * b_s2];
    const float ax = a_val + dt[vec];
    const float sp = fmaxf(ax, 0.0f) + log1pf(expf(-fabsf(ax)));
    decay_out[h] = expf(sp * scale[vec]);
    const float sig_b = gdn_prep_tree_precise_sigmoid_f32(b_val);
    if (BETA_BF16) {
      reinterpret_cast<nv_bfloat16*>(beta_out)[h] = __float2bfloat16(sig_b);
    } else {
      reinterpret_cast<float*>(beta_out)[h] = sig_b;
    }
  }
}
)cuda";

class GdnPrepareTreeCuda final : public mlx::core::Primitive {
 public:
  GdnPrepareTreeCuda(mlx::core::Stream stream, bool mean_eps, bool beta_bf16)
      : Primitive(stream), mean_eps_(mean_eps), beta_bf16_(beta_bf16) {}

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("GdnPrepareTreeCuda requires CUDA");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    using namespace mlx::core;
    auto& encoder = cu::get_command_encoder(stream());
    std::vector<array> owned;
    owned.reserve(8);
    for (int i = 0; i < 7; ++i) {
      if ((i == 3 || i == 5 || i == 6) && !inputs[i].flags().row_contiguous) {
        auto copy = contiguous_copy_gpu(inputs[i], stream());
        encoder.add_temporary(copy);
        owned.push_back(std::move(copy));
      } else {
        owned.push_back(inputs[i]);
      }
    }
    owned.push_back(cu::ensure_row_contiguous_aligned<4>(inputs[7], stream(), encoder));

    const auto& qkv = owned[0];
    const auto& a = owned[1];
    const auto& b = owned[2];
    const auto& history = owned[4];
    const auto& conv_taps = owned[7];
    const int T = qkv.shape(1);
    const bool valid_hist =
        (history.ndim() == 2 && history.shape() == Shape{3, 10240}) ||
        (history.ndim() == 3 && history.shape() == Shape{1, 3, 10240});
    if (qkv.ndim() != 3 || qkv.shape(0) != 1 || T < 1 || T > 4096 ||
        qkv.shape(2) != 10240 || qkv.dtype() != bfloat16 ||
        a.shape() != Shape{1, T, 48} || b.shape() != Shape{1, T, 48} ||
        !valid_hist || history.dtype() != bfloat16 ||
        conv_taps.ndim() != 2 || conv_taps.shape(0) != T ||
        conv_taps.shape(1) != 4 || conv_taps.dtype() != int32) {
      throw std::invalid_argument("GdnPrepareTreeCuda: invalid shape or dtype");
    }

    for (const auto& input : owned) {
      encoder.set_input_array(input);
    }
    for (auto& out : outputs) {
      out.set_data(cu::malloc_async(out.nbytes(), encoder));
      encoder.set_output_array(out);
    }

    const bool a_bf16 = a.dtype() == bfloat16;
    const bool b_bf16 = b.dtype() == bfloat16;
    static constexpr const char* kPrepTreeEntries[2][2][2] = {
        {{"mlx_gdn_prepare_tree_cuda_kernel<false,false,false>",
          "mlx_gdn_prepare_tree_cuda_kernel<false,false,true>"},
         {"mlx_gdn_prepare_tree_cuda_kernel<false,true,false>",
          "mlx_gdn_prepare_tree_cuda_kernel<false,true,true>"}},
        {{"mlx_gdn_prepare_tree_cuda_kernel<true,false,false>",
          "mlx_gdn_prepare_tree_cuda_kernel<true,false,true>"},
         {"mlx_gdn_prepare_tree_cuda_kernel<true,true,false>",
          "mlx_gdn_prepare_tree_cuda_kernel<true,true,true>"}},
    };
    const std::string_view entry = kPrepTreeEntries[a_bf16][b_bf16][beta_bf16_];
    auto kernel =
        cu::jit_kernel(encoder.device(), gdn_prepare_tree_cuda_source, entry);

    const int64_t hist_s0 =
        history.ndim() == 2 ? history.strides(0) : history.strides(1);
    const int64_t hist_s1 =
        history.ndim() == 2 ? history.strides(1) : history.strides(2);

    cu::KernelArgs args;
    for (const auto& input : owned) {
      args.append(input);
    }
    for (const auto& out : outputs) {
      args.append(out);
    }
    args.append<int32_t>(T);
    args.append<int64_t>(qkv.strides(1));
    args.append<int64_t>(qkv.strides(2));
    args.append<int64_t>(a.strides(1));
    args.append<int64_t>(a.strides(2));
    args.append<int64_t>(b.strides(1));
    args.append<int64_t>(b.strides(2));
    args.append<int64_t>(hist_s0);
    args.append<int64_t>(hist_s1);
    args.append<int32_t>(mean_eps_ ? 1 : 0);

    encoder.add_kernel_node_raw(
        kernel, dim3(20, T, 1), dim3(32, 4, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GdnPrepareTreeCuda)
  bool is_equivalent(const mlx::core::Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GdnPrepareTreeCuda*>(&other);
    return rhs && rhs->mean_eps_ == mean_eps_ && rhs->beta_bf16_ == beta_bf16_;
  }
  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    const int T = inputs[0].shape(1);
    return {
        {1, T, 16, 128}, {1, T, 16, 128}, {1, T, 48, 128},
        {1, T, 48}, {1, T, 48}, {3, 10240}};
  }

 private:
  bool mean_eps_;
  bool beta_bf16_;
};

} // namespace
#endif

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

// E47 (catalog D1): per-step kernel with 2 v-columns per simdgroup.
// Same shape contract as gated_delta_sources[0] (non-vec, non-mask) but each
// simdgroup processes dv_A=2y and dv_B=2y+1, sharing q[Dk] + k[Dk] loads.
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
    if (per_step_variant == 1) {
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
    return gated_delta_kernel_impl(q,k,v,g,beta,state,mask,out_y,out_state,false);
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

#ifdef MLX_NODE_CUDA_ENABLED
        if (mlx::core::default_device() == mlx::core::Device::gpu
            && (x_arr.dtype() == mlx::core::bfloat16
                || x_arr.dtype() == mlx::core::float16
                || x_arr.dtype() == mlx::core::float32)) {
            auto result = mlx::core::array(
                x_arr.shape(),
                x_arr.dtype(),
                std::make_shared<DFlash2ConvCuda>(
                    mlx::core::default_stream(mlx::core::Device::gpu), side),
                {x_arr, dyn_arr, base_arr});
            *out = reinterpret_cast<mlx_array*>(new array(std::move(result)));
            return true;
        }
#endif

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
// Other model families retain the existing default.
bool mlx_qwen4_gated_delta_kernel(mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g,
    mlx_array* beta, mlx_array* state, mlx_array* mask, mlx_array** out_y, mlx_array** out_state) {
    if (!q || !v) return false;
    auto& query = *reinterpret_cast<array*>(q);
    auto& value = *reinterpret_cast<array*>(v);
    static const int arch = mlx_gpu_architecture_gen();
    bool measured_shape = arch >= 17
        && query.ndim() == 4 && value.ndim() == 4 && query.shape(0) == 1
        && query.shape(3) == 128 && value.shape(2) == 48 && value.shape(3) == 128
        && (query.dtype() == mlx::core::float32 || query.dtype() == mlx::core::bfloat16);
    return gated_delta_kernel_impl(q,k,v,g,beta,state,mask,out_y,out_state,measured_shape,true);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Chunked fused GDN prefill (prompt-length windows, T > 32)
//
// One primitive per GDN layer for a whole prompt: fused_gdn_gating + the q/k
// rms_norm (prologue), the 4-vcol recurrence with 16 state rows of a value
// head per threadgroup (scan), the gated RMSNorm and z gate of
// gdn_fused_step (tail).
// Bit-identical to the unfused prefill chain (fast::metal_kernel recurrence
// -> fast::rms_norm -> compiled swiglu). A plain fast::Custom primitive —
// like GdnFused — so the pre-norm conv-output split views and the z/a/b
// gate views are read through their strides instead of being copied
// contiguous first. B = 1, Dk = Dv = 128.

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"
#include "mlx_test_counters.h"

namespace {

using namespace mlx::core;

static const char* kGdnPrefillHeadBody =
#include "metal/common/gdn_fused_prefill.metal.inc"
    ;
static const char* kGdnPrefillRowBody =
#include "metal/common/gdn_prefill_row_scan.metal.inc"
    ;

// Mirrors `GdnPrefillParams` in both kernel sources.
struct GdnPrefillParams {
  int T;
  int hk;
  int hv;
  int tiled;
  int has_state;
  int q_stride_t;
  int k_stride_t;
  int v_stride_t;
  int a_stride_t;
  int b_stride_t;
  int z_stride_t;
  float eps;
  float qk_eps;
};

// Product: state rows of a value head split over threadgroups. Head: one
// value head per 1024-thread threadgroup, the bit reference for tests.
enum class GdnPrefillGeometry { Rows, Head };

constexpr size_t kGdnRowThreads = 128;
constexpr size_t kGdnRowsPerGroup = 16;
constexpr size_t kGdnRowSideThreads = 256;

struct GdnPrefillPipelines {
  MTL::ComputePipelineState* head = nullptr;
  MTL::ComputePipelineState* prologue = nullptr;
  MTL::ComputePipelineState* scan = nullptr;
  MTL::ComputePipelineState* tail = nullptr;
};

MTL::ComputePipelineState* checked_pipeline(MTL::ComputePipelineState* p,
                                            size_t threads) {
  if (p->threadExecutionWidth() != 32 ||
      p->maxTotalThreadsPerThreadgroup() < NS::UInteger(threads)) {
    throw std::runtime_error(
        "gdn fused prefill cannot launch its threadgroup on this GPU");
  }
  return p;
}

GdnPrefillPipelines gdn_prefill_pipelines(metal::Device& device,
                                          GdnPrefillGeometry geometry, int dk,
                                          int dv, bool bf16_scalars) {
  GdnPrefillPipelines out;
  if (geometry == GdnPrefillGeometry::Head) {
    const std::string lib_name = "mlx_node_gdn_fused_prefill_dk" +
        std::to_string(dk) + "_dv" + std::to_string(dv);
    auto* library = device.get_library(lib_name, [dk, dv] {
      return std::string(metal::utils()) + "\n#define GDN_PREFILL_N_PER_T " +
          std::to_string(dk / 32) + "\n#define GDN_PREFILL_DV " +
          std::to_string(dv) + "\n" + kGdnPrefillHeadBody;
    });
    out.head = checked_pipeline(
        device.get_kernel("mlx_node_gdn_fused_prefill", library),
        size_t(8) * size_t(dv));
    return out;
  }
  auto* library = device.get_library("mlx_node_gdn_prefill_rows", [] {
    return std::string(metal::utils()) + "\n" + kGdnPrefillRowBody;
  });
  out.prologue = checked_pipeline(
      device.get_kernel(bf16_scalars
                            ? "mlx_node_gdn_prefill_prologue_bf16_scalars"
                            : "mlx_node_gdn_prefill_prologue",
                        library),
      kGdnRowSideThreads);
  out.scan = checked_pipeline(
      device.get_kernel("mlx_node_gdn_prefill_scan", library), kGdnRowThreads);
  out.tail = checked_pipeline(
      device.get_kernel("mlx_node_gdn_prefill_tail", library),
      kGdnRowSideThreads);
  return out;
}

size_t ceil_groups(size_t threads, size_t per_group) {
  return (threads + per_group - 1) / per_group;
}

// Input layout: q, k, v, a, b, a_log, dt_bias, state_in, wq, wk, z, w.
// Outputs: out [1, T, Hv * Dv] bf16, state_out [1, Hv, Dv, Dk] f32.
class GdnFusedPrefill final : public fast::Custom {
 public:
  GdnFusedPrefill(Stream stream, GdnPrefillGeometry geometry, int dk, int dv,
                  float eps, float qk_eps, bool tiled, bool has_state,
                  bool bf16_scalars)
      : Custom(stream,
               [](std::vector<array>) -> std::vector<array> {
                 throw std::runtime_error("GdnFusedPrefill has no fallback");
               }),
        geometry_(geometry),
        dk_(dk),
        dv_(dv),
        eps_(eps),
        qk_eps_(qk_eps),
        tiled_(tiled),
        has_state_(has_state),
        bf16_scalars_(bf16_scalars) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GdnFusedPrefill requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    bridge_testing::record(geometry_ == GdnPrefillGeometry::Rows
                               ? "gdn_prefill_row_scan"
                               : "gdn_prefill_head_scan");
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    auto& encoder = metal::get_command_encoder(stream);
    const auto pipelines =
        gdn_prefill_pipelines(device, geometry_, dk_, dv_, bf16_scalars_);

    GdnPrefillParams params{};
    params.tiled = tiled_ ? 1 : 0;
    params.has_state = has_state_ ? 1 : 0;
    params.eps = eps_;
    params.qk_eps = qk_eps_;

    // The strided split views (q/k/v rows, a/b/z rows) bind as-is and are
    // walked through their params strides; the dense sidecars bind
    // row-contiguous. Without a carried state (has_state == 0) the inputs
    // hold no state array at all and a scratch buffer stands in for the
    // binding the kernel never reads.
    const int z_at = has_state_ ? 10 : 9;
    std::vector<array> copies;
    std::vector<array> bound;
    bound.reserve(12);
    for (size_t i = 0; i < inputs.size(); ++i) {
      const array& x = inputs[i];
      const bool strided =
          (i <= 4) || (int(i) == z_at);  // q/k/v/a/b rows and z rows
      if (strided || x.flags().row_contiguous) {
        bound.push_back(x);
      } else {
        copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
        copy_gpu(x, copies.back(), CopyType::General, stream);
        bound.push_back(copies.back());
      }
    }
    if (!has_state_) {
      // Placeholder the kernel never reads (has_state == 0): insert it at
      // slot 7 where the kernel declares `state_in`.
      copies.push_back(array({1}, float32, nullptr, {}));
      copies.back().set_data(allocator::malloc(4));
      bound.insert(bound.begin() + 7, copies.back());
    }

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));

    params.T = inputs[0].shape(1);
    params.hk = inputs[0].shape(2);
    params.hv = inputs[2].shape(2);
    params.q_stride_t = int(inputs[0].strides(1));
    params.k_stride_t = int(inputs[1].strides(1));
    params.v_stride_t = int(inputs[2].strides(1));
    params.a_stride_t = int(inputs[3].strides(1));
    params.b_stride_t = int(inputs[4].strides(1));
    params.z_stride_t = int(inputs[z_at].strides(1));

    if (geometry_ == GdnPrefillGeometry::Head) {
      encoder.set_compute_pipeline_state(pipelines.head);
      int slot = 0;
      for (auto& b : bound) {
        encoder.set_input_array(b, slot++);
      }
      for (auto& o : outputs) {
        encoder.set_output_array(o, slot++);
      }
      encoder.set_bytes(params, slot);
      encoder.dispatch_threadgroups(MTL::Size(size_t(params.hv), 1, 1),
                                    MTL::Size(size_t(8) * size_t(dv_), 1, 1));
      encoder.add_temporaries(std::move(copies));
      return;
    }

    // bound: q k v a b a_log dt_bias state_in wq wk z w
    const size_t T = size_t(params.T);
    const size_t Hk = size_t(params.hk);
    const size_t Hv = size_t(params.hv);
    auto scratch = [](Shape shape, Dtype dtype) {
      array a(std::move(shape), dtype, nullptr, {});
      a.set_data(allocator::malloc(a.nbytes()));
      return a;
    };
    array qn = scratch(Shape{params.T, params.hk, dk_}, bfloat16);
    array kn = scratch(Shape{params.T, params.hk, dk_}, bfloat16);
    array g = scratch(Shape{params.T, params.hv}, float32);
    array beta = scratch(Shape{params.T, params.hv}, bfloat16);
    copies.insert(copies.end(), {qn, kn, g, beta});

    encoder.set_compute_pipeline_state(pipelines.prologue);
    encoder.set_input_array(bound[0], 0);
    encoder.set_input_array(bound[1], 1);
    encoder.set_input_array(bound[3], 2);
    encoder.set_input_array(bound[4], 3);
    encoder.set_input_array(bound[5], 4);
    encoder.set_input_array(bound[6], 5);
    encoder.set_input_array(bound[8], 6);
    encoder.set_input_array(bound[9], 7);
    encoder.set_output_array(qn, 8);
    encoder.set_output_array(kn, 9);
    encoder.set_output_array(g, 10);
    encoder.set_output_array(beta, 11);
    encoder.set_bytes(params, 12);
    encoder.dispatch_threadgroups(
        MTL::Size(ceil_groups(std::max(T * Hk * 32, T * Hv),
                              kGdnRowSideThreads),
                  1, 1),
        MTL::Size(kGdnRowSideThreads, 1, 1));

    encoder.set_compute_pipeline_state(pipelines.scan);
    encoder.set_input_array(qn, 0);
    encoder.set_input_array(kn, 1);
    encoder.set_input_array(bound[2], 2);
    encoder.set_input_array(g, 3);
    encoder.set_input_array(beta, 4);
    encoder.set_input_array(bound[7], 5);
    encoder.set_output_array(out, 6);
    encoder.set_output_array(outputs[1], 7);
    encoder.set_bytes(params, 8);
    // Grid after Splash's prefill GDN scan (splash runtime/ops/GDN.cpp:177-181).
    encoder.dispatch_threadgroups(
        MTL::Size(Hv * (size_t(dv_) / kGdnRowsPerGroup), 1, 1),
        MTL::Size(kGdnRowThreads, 1, 1));

    encoder.set_compute_pipeline_state(pipelines.tail);
    encoder.set_output_array(out, 0);
    encoder.set_input_array(bound[10], 1);
    encoder.set_input_array(bound[11], 2);
    encoder.set_bytes(params, 3);
    encoder.dispatch_threadgroups(
        MTL::Size(ceil_groups(T * Hv * 32, kGdnRowSideThreads), 1, 1),
        MTL::Size(kGdnRowSideThreads, 1, 1));
    encoder.add_temporaries(std::move(copies));
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    const auto& v = inputs[2];
    return {Shape{1, q.shape(1), v.shape(2) * v.shape(3)},
            Shape{1, v.shape(2), v.shape(3), q.shape(3)}};
  }

  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&,
                         const std::vector<array>&) override {
    throw std::runtime_error("GdnFusedPrefill is inference-only");
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const GdnFusedPrefill&>(other);
    return geometry_ == o.geometry_ && dk_ == o.dk_ && dv_ == o.dv_ &&
        eps_ == o.eps_ && qk_eps_ == o.qk_eps_ && tiled_ == o.tiled_ &&
        has_state_ == o.has_state_ && bf16_scalars_ == o.bf16_scalars_;
  }

  DEFINE_NAME(GdnFusedPrefill);

 private:
  GdnPrefillGeometry geometry_;
  int dk_;
  int dv_;
  float eps_;
  float qk_eps_;
  bool tiled_;
  bool has_state_;
  bool bf16_scalars_;
};

void expect_prefill(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(std::string("mlx_gdn_fused_prefill: ") + what);
  }
}

}  // namespace
#endif  // MLX_NODE_METAL_ENABLED

// Fused GDN prefill for one prompt window, any T (the prompt counterpart of
// `mlx_gdn_fused_step`, which caps at T <= 32).
//
//   q, k   [1, T, Hk, Dk] bf16 PRE-NORM conv outputs, rows may be strided
//   v      [1, T, Hv, Dv] bf16, rows may be strided
//   a, b   [1, T, Hv] bf16 gate views, rows may be strided
//   a_log, dt_bias [Hv], both f32 or both bf16 (a GGUF load keeps them
//     bf16); the prologue reads each value in its storage dtype and widens it
//     to float, as fused_gdn_gating does
//   state  [1, Hv, Dv, Dk] f32, or null → in-kernel zero init
//   wq, wk [Dk] bf16 folded q/k rms_norm weights (qk_norm_w_q/_k)
//   z      [1, T, Hv * Dv] bf16, rows may be strided
//   w      [Dv] bf16 gated-norm weight; eps its epsilon; qk_eps the q/k norm
//     epsilon (1e-6)
//   tiled: 1 for compact tiled-GGUF heads (hk = hv % Hk), 0 for the standard
//     repeat-expanded map (hk = hv / (Hv / Hk)) on unexpanded q/k.
//
// Writes `*out` [1, T, Hv * Dv] bf16 and `*out_state` [1, Hv, Dv, Dk] f32 —
// bit-identical to the unfused chain (q/k rms_norm_scaled -> fused gating ->
// q/k repeat or tile -> per-step kernel -> gated rms_norm -> swiglu). Returns
// false when off-contract or without Metal; the caller runs the chain.
static bool gdn_fused_prefill_entry(
    bool head_reference, mlx_array* q, mlx_array* k, mlx_array* v,
    mlx_array* a, mlx_array* b, mlx_array* a_log, mlx_array* dt_bias,
    mlx_array* state, mlx_array* wq, mlx_array* wk, mlx_array* z,
    mlx_array* w, float eps, float qk_eps, int tiled, mlx_array** out,
    mlx_array** out_state) {
  if (out) *out = nullptr;
  if (out_state) *out_state = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect_prefill(q && k && v && a && b && a_log && dt_bias && wq && wk && z &&
                       w && out && out_state,
                   "null argument");
    expect_prefill(metal::is_available() && default_device() == Device::gpu,
                   "requires the Metal device");
    const auto& qa = *reinterpret_cast<array*>(q);
    const auto& ka = *reinterpret_cast<array*>(k);
    const auto& va = *reinterpret_cast<array*>(v);
    const auto& aa = *reinterpret_cast<array*>(a);
    const auto& ba = *reinterpret_cast<array*>(b);
    const auto& ala = *reinterpret_cast<array*>(a_log);
    const auto& dta = *reinterpret_cast<array*>(dt_bias);
    const auto& wqa = *reinterpret_cast<array*>(wq);
    const auto& wka = *reinterpret_cast<array*>(wk);
    const auto& za = *reinterpret_cast<array*>(z);
    const auto& wa = *reinterpret_cast<array*>(w);
    expect_prefill(qa.ndim() == 4 && ka.ndim() == 4 && va.ndim() == 4 &&
                       aa.ndim() == 3 && ba.ndim() == 3 && za.ndim() == 3,
                   "ranks");
    const int T = va.shape(1), Hv = va.shape(2), Dv = va.shape(3);
    const int Hk = qa.shape(2), Dk = qa.shape(3);
    expect_prefill(T >= 1, "T");
    expect_prefill(Dk == 128 && Dv == 128, "Dk and Dv must be 128");
    expect_prefill(Hk > 0 && Hv % Hk == 0, "Hv must be a multiple of Hk");
    expect_prefill(qa.shape(0) == 1 && ka.shape(0) == 1 && va.shape(0) == 1 &&
                       aa.shape(0) == 1 && ba.shape(0) == 1 && za.shape(0) == 1,
                   "batch must be 1");
    expect_prefill(ka.shape() == qa.shape() &&
                       aa.shape() == Shape{1, T, Hv} &&
                       ba.shape() == aa.shape() &&
                       za.shape() == Shape{1, T, Hv * Dv},
                   "shapes");
    expect_prefill(qa.strides(3) == 1 && ka.strides(3) == 1 &&
                       va.strides(3) == 1 && qa.strides(2) == Dk &&
                       ka.strides(2) == Dk && va.strides(2) == Dv &&
                       aa.strides(2) == 1 && ba.strides(2) == 1 &&
                       za.strides(2) == 1,
                   "channel strides");
    expect_prefill(qa.dtype() == bfloat16 && ka.dtype() == bfloat16 &&
                       va.dtype() == bfloat16 && aa.dtype() == bfloat16 &&
                       ba.dtype() == bfloat16 && za.dtype() == bfloat16 &&
                       wqa.dtype() == bfloat16 && wka.dtype() == bfloat16 &&
                       wa.dtype() == bfloat16,
                   "dtypes (activations and weights bf16)");
    expect_prefill(wqa.shape() == Shape{Dk} && wka.shape() == Shape{Dk} &&
                       wa.shape() == Shape{Dv},
                   "norm weight shapes");
    expect_prefill(ala.size() == size_t(Hv) && dta.size() == size_t(Hv),
                   "gate scalar shapes");
    expect_prefill(ala.strides(0) == 1 && dta.strides(0) == 1,
                   "gate scalar strides");
    const bool bf16_scalars = ala.dtype() == bfloat16;
    expect_prefill(ala.dtype() == dta.dtype() &&
                       (bf16_scalars || ala.dtype() == float32),
                   "a_log and dt_bias must both be f32 or both bf16");
    expect_prefill(!(head_reference && bf16_scalars),
                   "the head kernel reads f32 a_log / dt_bias");
    std::optional<array> state_opt;
    Shape state_shape{1, Hv, Dv, Dk};
    if (state) {
      const auto& sa = *reinterpret_cast<array*>(state);
      expect_prefill(sa.shape() == state_shape && sa.dtype() == float32,
                     "state must be [1, Hv, Dv, Dk] f32");
      state_opt = sa;
    }
    std::vector<array> inputs{qa, ka, va, aa, ba, ala, dta};
    if (state_opt) {
      inputs.push_back(*state_opt);
    }
    inputs.insert(inputs.end(), {wqa, wka, za, wa});
    auto stream = default_stream(Device::gpu);
    const auto geometry = head_reference ? GdnPrefillGeometry::Head
                                         : GdnPrefillGeometry::Rows;
    gdn_prefill_pipelines(metal::device(stream.device), geometry, Dk, Dv,
                          bf16_scalars);
    // `has_state` keys the primitive AND the input count; when absent the
    // eval binds a scratch row (never read) in place of inputs[7].
    auto primitive = std::make_shared<GdnFusedPrefill>(
        stream, geometry, Dk, Dv, eps, qk_eps, tiled != 0,
        state_opt.has_value(), bf16_scalars);
    auto outputs = array::make_arrays(
        {Shape{1, T, Hv * Dv}, state_shape}, {bfloat16, float32}, primitive,
        std::move(inputs));
    *out = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_state =
        reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_gdn_fused_prefill: %s\n", e.what());
    return false;
  }
#else
  (void)head_reference; (void)q; (void)k; (void)v; (void)a; (void)b;
  (void)a_log; (void)dt_bias; (void)state; (void)wq; (void)wk; (void)z;
  (void)w; (void)eps; (void)qk_eps; (void)tiled;
  return false;
#endif
}

extern "C" bool mlx_gdn_fused_prefill(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* a, mlx_array* b,
    mlx_array* a_log, mlx_array* dt_bias, mlx_array* state, mlx_array* wq,
    mlx_array* wk, mlx_array* z, mlx_array* w, float eps, float qk_eps,
    int tiled, mlx_array** out, mlx_array** out_state) {
  return gdn_fused_prefill_entry(false, q, k, v, a, b, a_log, dt_bias, state,
                                 wq, wk, z, w, eps, qk_eps, tiled, out,
                                 out_state);
}

// The same entry on the one-value-head-per-threadgroup kernel: the bit
// reference of the tests (mlx-core declares it only under cfg(test)).
extern "C" bool mlx_gdn_fused_prefill_head_reference(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* a, mlx_array* b,
    mlx_array* a_log, mlx_array* dt_bias, mlx_array* state, mlx_array* wq,
    mlx_array* wk, mlx_array* z, mlx_array* w, float eps, float qk_eps,
    int tiled, mlx_array** out, mlx_array** out_state) {
  return gdn_fused_prefill_entry(true, q, k, v, a, b, a_log, dt_bias, state,
                                 wq, wk, z, w, eps, qk_eps, tiled, out,
                                 out_state);
}

// Tree-conv1d variant of fused CUDA GDN preparation (same inputs/outputs as
// mlx_gdn_prepare_cuda, plus int32[T, 4] `conv_taps`).
// For token t and tap d in 0..3:
//   src = conv_taps[t, d]
//   src < 0  -> history[3 + src, c] (-3 -> 0, -2 -> 1, -1 -> 2)
//   src >= 0 -> qkv[0, src, c]
// Outputs[0..5]: q [1,T,16,128] bf16, k [1,T,16,128] bf16, v [1,T,48,128] bf16,
// decay [1,T,48] f32, beta [1,T,48] (bf16/f32), next_history [3,10240] bf16.
extern "C" bool mlx_gdn_prepare_tree_cuda(
    mlx_array* qkv, mlx_array* a, mlx_array* b, mlx_array* conv,
    mlx_array* history, mlx_array* scale, mlx_array* dt,
    mlx_array* conv_taps, bool mean_eps, bool beta_input_dtype,
    mlx_array** outputs) {
  if (outputs) {
    for (int i = 0; i < 6; ++i) {
      outputs[i] = nullptr;
    }
  }
#ifdef MLX_NODE_CUDA_ENABLED
  if (!outputs) {
    return false;
  }
  try {
    using namespace mlx::core;
    if (default_device() != Device::gpu) {
      return false;
    }
    std::vector<array> in;
    in.reserve(8);
    for (auto* p : {qkv, a, b, conv, history, scale, dt, conv_taps}) {
      if (!p) {
        return false;
      }
      in.push_back(*reinterpret_cast<array*>(p));
    }
    const auto& x = in[0];
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(1) < 1 ||
        x.shape(1) > 4096 || x.shape(2) != 10240 || x.dtype() != bfloat16) {
      return false;
    }
    const int T = x.shape(1);
    for (int i : {1, 2}) {
      if (in[i].shape() != Shape{1, T, 48} ||
          (in[i].dtype() != float32 && in[i].dtype() != bfloat16)) {
        return false;
      }
    }
    const bool valid_hist =
        (in[4].ndim() == 2 && in[4].shape() == Shape{3, 10240}) ||
        (in[4].ndim() == 3 && in[4].shape() == Shape{1, 3, 10240});
    if (in[3].size() != 40960 || in[3].dtype() != float32 ||
        !valid_hist || in[4].dtype() != bfloat16 ||
        in[5].size() != 48 || in[5].dtype() != float32 ||
        in[6].size() != 48 || in[6].dtype() != float32 ||
        in[7].ndim() != 2 || in[7].shape(0) != T || in[7].shape(1) != 4 ||
        in[7].dtype() != int32) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        {{1, T, 16, 128}, {1, T, 16, 128}, {1, T, 48, 128},
         {1, T, 48}, {1, T, 48}, {3, 10240}},
        {bfloat16, bfloat16, bfloat16, float32,
         beta_input_dtype ? bfloat16 : float32, bfloat16},
        std::make_shared<GdnPrepareTreeCuda>(stream, mean_eps, beta_input_dtype),
        std::move(in));
    for (int i = 0; i < 6; ++i) {
      outputs[i] = reinterpret_cast<mlx_array*>(new array(std::move(results[i])));
    }
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_prepare_tree_cuda: %s\n", error.what());
    return false;
  }
#else
  (void)qkv; (void)a; (void)b; (void)conv; (void)history; (void)scale; (void)dt;
  (void)conv_taps; (void)mean_eps; (void)beta_input_dtype;
  return false;
#endif
}
