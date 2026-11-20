#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace mlx::core {
namespace {

constexpr char gdn_fused_step_source[] =
#include "cuda/gdn_fused_step.cu.inc"
    ;

constexpr char gdn_prepare_source[] =
#include "cuda/gdn_prepare.cu.inc"
    ;

class GdnFusedStepCuda : public Primitive {
public:
  GdnFusedStepCuda(Stream stream, float eps, bool tiled, bool has_dst)
      : Primitive(stream), eps_(eps), tiled_(tiled), has_dst_(has_dst) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GDN fused step CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto contiguous = [&](const array& x) {
      if (x.flags().row_contiguous) {
        return x;
      }
      auto copy = contiguous_copy_gpu(x, stream());
      encoder.add_temporary(copy);
      return copy;
    };
    std::vector<array> owned;
    owned.reserve(8);
    for (int i = 0; i < 8; ++i) {
      owned.push_back(i == 6 ? inputs[i] : contiguous(inputs[i]));
    }
    const auto& z = owned[6];
    if (z.strides(-1) != 1) {
      owned[6] = contiguous_copy_gpu(z, stream());
      encoder.add_temporary(owned[6]);
    }

    auto& out = outputs[0];
    auto& state_out = outputs[1];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    if (inputs.size() == 9) {
      if (!inputs[8].flags().row_contiguous || inputs[8].shape() != inputs[5].shape()) {
        throw std::invalid_argument("GdnFusedStepCuda: state_dst must be row-contiguous");
      }
      // `state_out` aliases `inputs[8]`; set_output_array tracks the dependency
      // without double-counting the state buffer in bytes_in_graph_.
      state_out.copy_shared_buffer(inputs[8]);
    } else {
      state_out.set_data(cu::malloc_async(state_out.nbytes(), encoder));
    }
    for (const auto& input : owned) {
      encoder.set_input_array(input);
    }
    encoder.set_output_array(out);
    encoder.set_output_array(state_out);

    const auto& v = owned[2];
    const auto& q = owned[0];
    const auto& z_bound = owned[6];
    const int B = v.shape(0);
    const int T = v.shape(1);
    const int Hv = v.shape(2);
    const int Hk = q.shape(2);
    if (T < 1 || T > 32 || Hk <= 0 || Hv % Hk != 0) {
      throw std::invalid_argument("GdnFusedStepCuda: unsupported replay shape");
    }
    cu::KernelArgs args;
    for (const auto& input : owned) {
      args.append(input);
    }
    args.append(out);
    args.append(state_out);
    args.append<int32_t>(T);
    args.append<int32_t>(Hk);
    args.append<int32_t>(Hv);
    args.append<int64_t>(z_bound.strides(0));
    args.append<int64_t>(z_bound.strides(1));
    args.append<float>(eps_);
    args.append<int32_t>(tiled_ ? 1 : 0);
    auto kernel = cu::jit_kernel(
        encoder.device(), gdn_fused_step_source, "mlx_gdn_fused_step_cuda_kernel");
    encoder.add_kernel_node_raw(
        kernel, dim3(Hv, B, 1), dim3(1024, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GdnFusedStepCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GdnFusedStepCuda*>(&other);
    return rhs && rhs->eps_ == eps_ && rhs->tiled_ == tiled_ &&
        rhs->has_dst_ == has_dst_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& v = inputs[2];
    return {{v.shape(0), v.shape(1), v.shape(2) * v.shape(3)}, inputs[5].shape()};
  }

private:
  float eps_;
  bool tiled_;
  bool has_dst_;
};

class GdnReplayCuda : public UnaryPrimitive {
public:
  GdnReplayCuda(Stream stream, int accepted_steps, bool tiled)
      : UnaryPrimitive(stream),
        accepted_steps_(accepted_steps),
        tiled_(tiled) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("GDN replay CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    for (const auto& input : owned) {
      encoder.set_input_array(input);
    }
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_output_array(out);

    const auto& k = owned[0];
    const auto& v = owned[1];
    const int B = v.shape(0);
    const int window_stride = v.shape(1);
    const int Hv = v.shape(2);
    const int Hk = k.shape(2);

    cu::KernelArgs args;
    for (const auto& input : owned) {
      args.append(input);
    }
    args.append(out);
    args.append<int32_t>(accepted_steps_);
    args.append<int32_t>(window_stride);
    args.append<int32_t>(Hk);
    args.append<int32_t>(Hv);
    args.append<int32_t>(tiled_ ? 1 : 0);
    auto kernel = cu::jit_kernel(
        encoder.device(), gdn_fused_step_source, "mlx_gdn_replay_cuda_kernel");
    encoder.add_kernel_node_raw(
        kernel, dim3(Hv, B, 1), dim3(1024, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GdnReplayCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GdnReplayCuda*>(&other);
    return rhs && rhs->accepted_steps_ == accepted_steps_ &&
        rhs->tiled_ == tiled_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[4].shape()};
  }

private:
  int accepted_steps_;
  bool tiled_;
};

class GdnPrepareCuda : public Primitive {
public:
  GdnPrepareCuda(Stream stream, bool mean_eps, bool beta_bf16)
      : Primitive(stream), mean_eps_(mean_eps), beta_bf16_(beta_bf16) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GDN prepare CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    std::vector<array> owned;
    owned.reserve(7);
    for (int i = 0; i < 7; ++i) {
      if ((i == 3 || i == 5 || i == 6) && !inputs[i].flags().row_contiguous) {
        auto copy = contiguous_copy_gpu(inputs[i], stream());
        encoder.add_temporary(copy);
        owned.push_back(std::move(copy));
      } else {
        owned.push_back(inputs[i]);
      }
    }
    for (const auto& input : owned) {
      encoder.set_input_array(input);
    }
    for (auto& out : outputs) {
      out.set_data(cu::malloc_async(out.nbytes(), encoder));
      encoder.set_output_array(out);
    }

    const auto& qkv = owned[0];
    const auto& a = owned[1];
    const auto& b = owned[2];
    const auto& history = owned[4];
    const int T = qkv.shape(1);
    const bool a_bf16 = a.dtype() == bfloat16;
    const bool b_bf16 = b.dtype() == bfloat16;
    static constexpr const char* kPrepEntries[2][2][2] = {
        {{"mlx_gdn_prepare_cuda_kernel<false,false,false>",
          "mlx_gdn_prepare_cuda_kernel<false,false,true>"},
         {"mlx_gdn_prepare_cuda_kernel<false,true,false>",
          "mlx_gdn_prepare_cuda_kernel<false,true,true>"}},
        {{"mlx_gdn_prepare_cuda_kernel<true,false,false>",
          "mlx_gdn_prepare_cuda_kernel<true,false,true>"},
         {"mlx_gdn_prepare_cuda_kernel<true,true,false>",
          "mlx_gdn_prepare_cuda_kernel<true,true,true>"}},
    };
    const std::string_view entry = kPrepEntries[a_bf16][b_bf16][beta_bf16_];
    auto kernel = cu::jit_kernel(encoder.device(), gdn_prepare_source, entry);

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
    args.append<int64_t>(history.strides(0));
    args.append<int64_t>(history.strides(1));
    args.append<int32_t>(mean_eps_ ? 1 : 0);

    encoder.add_kernel_node_raw(
        kernel, dim3(20, T, 1), dim3(32, 4, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GdnPrepareCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GdnPrepareCuda*>(&other);
    return rhs && rhs->mean_eps_ == mean_eps_ && rhs->beta_bf16_ == beta_bf16_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
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
} // namespace mlx::core
#endif

extern "C" bool mlx_gdn_fused_step_cuda(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state, mlx_array* z, mlx_array* w, float eps, int32_t tiled_heads,
    mlx_array* state_dst, mlx_array** out, mlx_array** out_state) {
#ifdef MLX_NODE_CUDA_ENABLED
  if (out) {
    *out = nullptr;
  }
  if (out_state) {
    *out_state = nullptr;
  }
  try {
    if (!q || !k || !v || !g || !beta || !state || !z || !w || !out || !out_state) {
      return false;
    }
    using namespace mlx::core;
    if (default_device() != Device::gpu) {
      return false;
    }
    const auto& qa = *reinterpret_cast<array*>(q);
    const auto& ka = *reinterpret_cast<array*>(k);
    const auto& va = *reinterpret_cast<array*>(v);
    const auto& ga = *reinterpret_cast<array*>(g);
    const auto& ba = *reinterpret_cast<array*>(beta);
    const auto& sa = *reinterpret_cast<array*>(state);
    const auto& za = *reinterpret_cast<array*>(z);
    const auto& wa = *reinterpret_cast<array*>(w);
    if (qa.ndim() != 4 || ka.ndim() != 4 || va.ndim() != 4 || ga.ndim() != 3 ||
        ba.ndim() != 3 || za.ndim() < 2 || wa.ndim() != 1) {
      return false;
    }
    const int B = va.shape(0);
    const int T = va.shape(1);
    const int Hv = va.shape(2);
    const int Dv = va.shape(3);
    const int Hk = qa.shape(2);
    const int Dk = qa.shape(3);
    if (T < 1 || T > 32 || Dk != 128 || Dv != 128 || Hk <= 0 || Hv % Hk != 0 ||
        qa.shape() != Shape{B, T, Hk, Dk} || ka.shape() != qa.shape() ||
        ga.shape() != Shape{B, T, Hv} || ba.shape() != ga.shape() ||
        sa.shape() != Shape{B, Hv, Dv, Dk} || za.shape(0) != B ||
        za.shape(1) != T || za.shape(-1) != Hv * Dv || wa.shape() != Shape{Dv}) {
      return false;
    }
    if (qa.dtype() != bfloat16 || ka.dtype() != bfloat16 || va.dtype() != bfloat16 ||
        ba.dtype() != bfloat16 || za.dtype() != bfloat16 || wa.dtype() != bfloat16 ||
        ga.dtype() != float32 || sa.dtype() != float32) {
      return false;
    }
    std::vector<array> inputs{qa, ka, va, ga, ba, sa, za, wa};
    if (state_dst) {
      const auto& dst = *reinterpret_cast<array*>(state_dst);
      if (dst.shape() != sa.shape() || dst.dtype() != float32 ||
          !dst.flags().row_contiguous) {
        return false;
      }
      inputs.push_back(dst);
    }
    auto stream = default_stream(Device::gpu);
    auto outputs = array::make_arrays(
        {Shape{B, T, Hv * Dv}, sa.shape()}, {bfloat16, float32},
        std::make_shared<GdnFusedStepCuda>(
            stream, eps, tiled_heads != 0, state_dst != nullptr),
        std::move(inputs));
    *out = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_state = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_fused_step_cuda: %s\n", error.what());
    return false;
  }
#else
  (void)q; (void)k; (void)v; (void)g; (void)beta; (void)state; (void)z; (void)w;
  (void)eps; (void)tiled_heads; (void)state_dst; (void)out; (void)out_state;
  return false;
#endif
}

extern "C" bool mlx_gdn_replay_cuda(
    mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta, mlx_array* state,
    int32_t accepted_steps, int32_t tiled_heads, mlx_array** out_state) {
#ifdef MLX_NODE_CUDA_ENABLED
  if (out_state) {
    *out_state = nullptr;
  }
  try {
    if (!k || !v || !g || !beta || !state || !out_state || accepted_steps < 1) {
      return false;
    }
    using namespace mlx::core;
    if (default_device() != Device::gpu) {
      return false;
    }
    const auto& ka = *reinterpret_cast<array*>(k);
    const auto& va = *reinterpret_cast<array*>(v);
    const auto& ga = *reinterpret_cast<array*>(g);
    const auto& ba = *reinterpret_cast<array*>(beta);
    const auto& sa = *reinterpret_cast<array*>(state);
    if (ka.ndim() != 4 || va.ndim() != 4 || ga.ndim() != 3 || ba.ndim() != 3 ||
        sa.ndim() != 4) {
      return false;
    }
    const int B = va.shape(0);
    const int S = va.shape(1);
    const int Hv = va.shape(2);
    const int Dv = va.shape(3);
    const int Hk = ka.shape(2);
    const int Dk = ka.shape(3);
    if (accepted_steps > S || Dk != 128 || Dv != 128 || Hk <= 0 || Hv % Hk != 0 ||
        ka.shape() != Shape{B, S, Hk, Dk} || ga.shape() != Shape{B, S, Hv} ||
        ba.shape() != Shape{B, S, Hv} || sa.shape() != Shape{B, Hv, Dv, Dk}) {
      return false;
    }
    if (ka.dtype() != bfloat16 || va.dtype() != bfloat16 ||
        ba.dtype() != bfloat16 || ga.dtype() != float32 ||
        sa.dtype() != float32) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    auto result = array(
        sa.shape(), float32,
        std::make_shared<GdnReplayCuda>(stream, accepted_steps, tiled_heads != 0),
        {ka, va, ga, ba, sa});
    *out_state = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_replay_cuda: %s\n", error.what());
    return false;
  }
#else
  (void)k; (void)v; (void)g; (void)beta; (void)state;
  (void)accepted_steps; (void)tiled_heads; (void)out_state;
  return false;
#endif
}

extern "C" bool mlx_gdn_prepare_cuda(
    mlx_array* qkv, mlx_array* a, mlx_array* b, mlx_array* conv,
    mlx_array* history, mlx_array* scale, mlx_array* dt, bool mean_eps,
    bool beta_input_dtype, mlx_array** outputs) {
#ifdef MLX_NODE_CUDA_ENABLED
  if (!outputs) {
    return false;
  }
  for (int i = 0; i < 6; ++i) {
    outputs[i] = nullptr;
  }
  try {
    using namespace mlx::core;
    if (default_device() != Device::gpu) {
      return false;
    }
    std::vector<array> in;
    in.reserve(7);
    for (auto* p : {qkv, a, b, conv, history, scale, dt}) {
      if (!p) {
        return false;
      }
      in.push_back(*reinterpret_cast<array*>(p));
    }
    const auto& x = in[0];
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(1) < 1 ||
        x.shape(1) > 1024 || x.shape(2) != 10240 || x.dtype() != bfloat16) {
      return false;
    }
    const int T = x.shape(1);
    for (int i : {1, 2}) {
      if (in[i].shape() != Shape{1, T, 48} ||
          (in[i].dtype() != float32 && in[i].dtype() != bfloat16)) {
        return false;
      }
    }
    if (in[3].size() != 40960 || in[3].dtype() != float32 ||
        in[4].shape() != Shape{3, 10240} || in[4].dtype() != bfloat16 ||
        in[5].size() != 48 || in[5].dtype() != float32 ||
        in[6].size() != 48 || in[6].dtype() != float32) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        {{1, T, 16, 128}, {1, T, 16, 128}, {1, T, 48, 128},
         {1, T, 48}, {1, T, 48}, {3, 10240}},
        {bfloat16, bfloat16, bfloat16, float32,
         beta_input_dtype ? bfloat16 : float32, bfloat16},
        std::make_shared<GdnPrepareCuda>(stream, mean_eps, beta_input_dtype),
        std::move(in));
    for (int i = 0; i < 6; ++i) {
      outputs[i] = reinterpret_cast<mlx_array*>(new array(std::move(results[i])));
    }
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_prepare_cuda: %s\n", error.what());
    return false;
  }
#else
  (void)qkv; (void)a; (void)b; (void)conv; (void)history; (void)scale; (void)dt;
  (void)mean_eps; (void)beta_input_dtype; (void)outputs;
  return false;
#endif
}
