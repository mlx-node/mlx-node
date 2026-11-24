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

constexpr char gdn_prefill_chain_source[] =
#include "cuda/gdn_prefill_chain.cu.inc"
    ;

class GdnFusedStepCuda : public Primitive {
public:
  GdnFusedStepCuda(
      Stream stream, float eps, bool tiled, bool has_dst, bool sigmoid_gate)
      : Primitive(stream),
        eps_(eps),
        tiled_(tiled),
        has_dst_(has_dst),
        sigmoid_gate_(sigmoid_gate) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GDN fused step CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    std::vector<array> owned;
    owned.reserve(8);
    owned.push_back(cu::ensure_row_contiguous_aligned<8>(inputs[0], stream(), encoder));
    owned.push_back(cu::ensure_row_contiguous_aligned<8>(inputs[1], stream(), encoder));
    owned.push_back(cu::ensure_row_contiguous_aligned<8>(inputs[2], stream(), encoder));
    owned.push_back(cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder));
    owned.push_back(
        inputs[4].dtype() == bfloat16
            ? cu::ensure_row_contiguous_aligned<2>(inputs[4], stream(), encoder)
            : cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder));
    owned.push_back(cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder));
    owned.push_back(inputs[6]);
    owned.push_back(
        inputs[7].dtype() == bfloat16
            ? cu::ensure_row_contiguous_aligned<2>(inputs[7], stream(), encoder)
            : cu::ensure_row_contiguous_aligned<4>(inputs[7], stream(), encoder));
    const auto& z = owned[6];
    const bool z_strided_ok =
        (z.ndim() == 3 && z.strides(2) == 1) ||
        (z.ndim() == 4 && z.strides(3) == 1 && z.strides(2) == 128);
    if (!z_strided_ok) {
      owned[6] = contiguous_copy_gpu(z, stream());
      encoder.add_temporary(owned[6]);
    }

    auto& out = outputs[0];
    auto& state_out = outputs[1];
    if (inputs.size() == 9) {
      if (!inputs[8].flags().row_contiguous || inputs[8].shape() != inputs[5].shape() ||
          inputs[8].dtype() != float32 || inputs[8].buffer().ptr() == nullptr ||
          (reinterpret_cast<uintptr_t>(gpu_ptr<void>(inputs[8])) & 15) != 0) {
        throw std::invalid_argument(
            "GdnFusedStepCuda: state_dst must be row-contiguous and 16B-aligned");
      }
      const void* dst_buf = inputs[8].buffer().ptr();
      for (size_t i = 0; i < owned.size(); ++i) {
        if (i == 5) {
          continue;
        }
        if (dst_buf == owned[i].buffer().ptr() ||
            dst_buf == inputs[i].buffer().ptr()) {
          throw std::invalid_argument(
              "GdnFusedStepCuda: state_dst aliases a non-state input buffer");
        }
      }
      if ((dst_buf == owned[5].buffer().ptr() &&
           gpu_ptr<void>(inputs[8]) != gpu_ptr<void>(owned[5])) ||
          (dst_buf == inputs[5].buffer().ptr() &&
           gpu_ptr<void>(inputs[8]) != gpu_ptr<void>(inputs[5]))) {
        throw std::invalid_argument(
            "GdnFusedStepCuda: state_dst partially overlaps state buffer");
      }
      // `state_out` aliases `inputs[8]`; set_output_array tracks the dependency
      // without double-counting the state buffer in bytes_in_graph_.
      state_out.copy_shared_buffer(inputs[8]);
    } else {
      state_out.set_data(cu::malloc_async(state_out.nbytes(), encoder));
    }
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    for (const auto& input : owned) {
      encoder.set_input_array(input);
    }
    encoder.set_output_array(out);
    encoder.set_output_array(state_out);

    const auto& v = owned[2];
    const auto& q = owned[0];
    const auto& beta = owned[4];
    const auto& z_bound = owned[6];
    const auto& w = owned[7];
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
    args.append<int32_t>(w.dtype() == float32 ? 1 : 0);
    static constexpr const char* kEntries[2][2] = {
        {"mlx_gdn_fused_step_cuda_kernel<false,false>",
         "mlx_gdn_fused_step_cuda_kernel<false,true>"},
        {"mlx_gdn_fused_step_cuda_kernel<true,false>",
         "mlx_gdn_fused_step_cuda_kernel<true,true>"},
    };
    const char* entry = kEntries[beta.dtype() == bfloat16][sigmoid_gate_];
    auto kernel = cu::jit_kernel(encoder.device(), gdn_fused_step_source, entry);
    encoder.add_kernel_node_raw(
        kernel, dim3(Hv, B, 1), dim3(1024, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GdnFusedStepCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GdnFusedStepCuda*>(&other);
    return rhs && rhs->eps_ == eps_ && rhs->tiled_ == tiled_ &&
        rhs->has_dst_ == has_dst_ && rhs->sigmoid_gate_ == sigmoid_gate_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& v = inputs[2];
    return {{v.shape(0), v.shape(1), v.shape(2) * v.shape(3)}, inputs[5].shape()};
  }

private:
  float eps_;
  bool tiled_;
  bool has_dst_;
  bool sigmoid_gate_;
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
    std::vector<array> owned;
    owned.reserve(5);
    owned.push_back(cu::ensure_row_contiguous_aligned<8>(inputs[0], stream(), encoder));
    owned.push_back(cu::ensure_row_contiguous_aligned<2>(inputs[1], stream(), encoder));
    owned.push_back(cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder));
    owned.push_back(
        inputs[3].dtype() == bfloat16
            ? cu::ensure_row_contiguous_aligned<2>(inputs[3], stream(), encoder)
            : cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder));
    owned.push_back(cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder));
    for (const auto& input : owned) {
      encoder.set_input_array(input);
    }
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_output_array(out);

    const auto& k = owned[0];
    const auto& v = owned[1];
    const auto& beta = owned[3];
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
    const char* entry = (beta.dtype() == bfloat16)
        ? "mlx_gdn_replay_cuda_kernel<true>"
        : "mlx_gdn_replay_cuda_kernel<false>";
    auto kernel = cu::jit_kernel(encoder.device(), gdn_fused_step_source, entry);
    encoder.add_kernel_node_raw(
        kernel, dim3(Hv * 4, B, 1), dim3(256, 1, 1), {}, 0, args.args());
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
  GdnPrepareCuda(
      Stream stream, bool mean_eps, bool beta_bf16, bool has_valid_lengths)
      : Primitive(stream),
        mean_eps_(mean_eps),
        beta_bf16_(beta_bf16),
        has_valid_lengths_(has_valid_lengths) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GDN prepare CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    std::vector<array> owned;
    owned.reserve(inputs.size());
    for (size_t i = 0; i < 7; ++i) {
      if (i == 3) {
        owned.push_back(
            cu::ensure_row_contiguous_aligned<16>(inputs[i], stream(), encoder));
      } else if (i == 5 || i == 6) {
        owned.push_back(
            cu::ensure_row_contiguous_aligned<4>(inputs[i], stream(), encoder));
      } else {
        owned.push_back(inputs[i]);
      }
    }
    if (has_valid_lengths_) {
      owned.push_back(
          cu::ensure_row_contiguous_aligned<4>(inputs[7], stream(), encoder));
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
    const int B = qkv.shape(0);
    const int T = qkv.shape(1);
    const int W = qkv.shape(2);
    const int Hv = a.shape(2);
    const int Hk = (W - Hv * 128) / 256;
    const int num_vecs = 2 * Hk + Hv;
    const int64_t hist_sb = (history.ndim() == 3) ? history.strides(0) : 0;
    const int64_t hist_s0 =
        (history.ndim() == 3) ? history.strides(1) : history.strides(0);
    const int64_t hist_s1 =
        (history.ndim() == 3) ? history.strides(2) : history.strides(1);
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
    static constexpr const char* kPrepTiledEntries[2][2][2] = {
        {{"mlx_gdn_prepare_tiled_cuda_kernel<false,false,false>",
          "mlx_gdn_prepare_tiled_cuda_kernel<false,false,true>"},
         {"mlx_gdn_prepare_tiled_cuda_kernel<false,true,false>",
          "mlx_gdn_prepare_tiled_cuda_kernel<false,true,true>"}},
        {{"mlx_gdn_prepare_tiled_cuda_kernel<true,false,false>",
          "mlx_gdn_prepare_tiled_cuda_kernel<true,false,true>"},
         {"mlx_gdn_prepare_tiled_cuda_kernel<true,true,false>",
          "mlx_gdn_prepare_tiled_cuda_kernel<true,true,true>"}},
    };
    const bool use_tiled = (T >= 16);
    const std::string_view entry = use_tiled
        ? kPrepTiledEntries[a_bf16][b_bf16][beta_bf16_]
        : kPrepEntries[a_bf16][b_bf16][beta_bf16_];
    auto kernel = cu::jit_kernel(encoder.device(), gdn_prepare_source, entry);

    cu::KernelArgs args;
    for (size_t i = 0; i < 7; ++i) {
      args.append(owned[i]);
    }
    for (const auto& out : outputs) {
      args.append(out);
    }
    if (has_valid_lengths_) {
      args.append(owned[7]);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append<int32_t>(T);
    args.append<int32_t>(Hk);
    args.append<int32_t>(Hv);
    args.append<int64_t>(qkv.strides(0));
    args.append<int64_t>(qkv.strides(1));
    args.append<int64_t>(qkv.strides(2));
    args.append<int64_t>(a.strides(0));
    args.append<int64_t>(a.strides(1));
    args.append<int64_t>(a.strides(2));
    args.append<int64_t>(b.strides(0));
    args.append<int64_t>(b.strides(1));
    args.append<int64_t>(b.strides(2));
    args.append<int64_t>(hist_sb);
    args.append<int64_t>(hist_s0);
    args.append<int64_t>(hist_s1);
    args.append<int32_t>(mean_eps_ ? 1 : 0);

    const uint32_t grid_x = static_cast<uint32_t>((num_vecs + 3) / 4);
    const uint32_t grid_y = use_tiled
        ? static_cast<uint32_t>((T + 31) / 32)
        : static_cast<uint32_t>(T);
    const uint32_t grid_z = static_cast<uint32_t>(B);
    encoder.add_kernel_node_raw(
        kernel, dim3(grid_x, grid_y, grid_z), dim3(32, 4, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GdnPrepareCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GdnPrepareCuda*>(&other);
    return rhs && rhs->mean_eps_ == mean_eps_ && rhs->beta_bf16_ == beta_bf16_ &&
        rhs->has_valid_lengths_ == has_valid_lengths_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int B = inputs[0].shape(0);
    const int T = inputs[0].shape(1);
    const int W = inputs[0].shape(2);
    const int Hv = inputs[1].shape(2);
    const int Hk = (W - Hv * 128) / 256;
    return {
        {B, T, Hk, 128}, {B, T, Hk, 128}, {B, T, Hv, 128},
        {B, T, Hv}, {B, T, Hv}, inputs[4].shape()};
  }

private:
  bool mean_eps_;
  bool beta_bf16_;
  bool has_valid_lengths_;
};

class GdnPrefillChainCuda : public Primitive {
public:
  GdnPrefillChainCuda(
      Stream stream, int tiled_heads, bool fuse_gated_norm, float eps,
      bool sigmoid_gate = false)
      : Primitive(stream),
        tiled_heads_(tiled_heads),
        fuse_gated_norm_(fuse_gated_norm),
        eps_(eps),
        sigmoid_gate_(sigmoid_gate) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GDN prefill chain CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array q = cu::ensure_row_contiguous_aligned<8>(inputs[0], stream(), encoder);
    array k = cu::ensure_row_contiguous_aligned<8>(inputs[1], stream(), encoder);
    array v = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array g = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);
    array beta = inputs[4].dtype() == bfloat16
        ? cu::ensure_row_contiguous_aligned<2>(inputs[4], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);
    array state = cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder);

    const int B = q.shape(0);
    const int W = q.shape(1);
    const int hk = q.shape(2);
    const int hv = v.shape(2);

    auto& y_out = outputs[0];
    auto& state_out = outputs[1];
    y_out.set_data(cu::malloc_async(y_out.nbytes(), encoder));
    state_out.set_data(cu::malloc_async(state_out.nbytes(), encoder));

    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(g);
    encoder.set_input_array(beta);
    encoder.set_input_array(state);

    int64_t z_stride_b = 0;
    int64_t z_stride_t = 0;
    int32_t w_is_f32 = 0;
    array z_owned(0.0f);
    array w_owned(0.0f);
    if (fuse_gated_norm_) {
      array z = inputs[6];
      const bool z_strided_ok =
          (z.ndim() == 3 && z.strides(2) == 1) ||
          (z.ndim() == 4 && z.strides(3) == 1 && z.strides(2) == 128);
      if (!z_strided_ok) {
        z = contiguous_copy_gpu(z, stream());
        encoder.add_temporary(z);
      }
      z_stride_b = z.strides(0);
      z_stride_t = z.strides(1);
      array w = inputs[7].dtype() == bfloat16
          ? cu::ensure_row_contiguous_aligned<2>(inputs[7], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<4>(inputs[7], stream(), encoder);
      w_is_f32 = (w.dtype() == float32) ? 1 : 0;
      z_owned = std::move(z);
      w_owned = std::move(w);
    }

    cu::KernelArgs args;
    args.append(q);
    args.append(k);
    args.append(v);
    args.append(g);
    args.append(beta);
    args.append(state);
    if (fuse_gated_norm_) {
      encoder.set_input_array(z_owned);
      encoder.set_input_array(w_owned);
      args.append(z_owned);
      args.append(w_owned);
    } else {
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
    }
    encoder.set_output_array(y_out);
    encoder.set_output_array(state_out);

    static constexpr const char* kEntries[2][2][2] = {
        {{"mlx_gdn_prefill_chain_cuda_kernel<false,false,false>",
          "mlx_gdn_prefill_chain_cuda_kernel<false,false,true>"},
         {"mlx_gdn_prefill_chain_cuda_kernel<false,true,false>",
          "mlx_gdn_prefill_chain_cuda_kernel<false,true,true>"}},
        {{"mlx_gdn_prefill_chain_cuda_kernel<true,false,false>",
          "mlx_gdn_prefill_chain_cuda_kernel<true,false,true>"},
         {"mlx_gdn_prefill_chain_cuda_kernel<true,true,false>",
          "mlx_gdn_prefill_chain_cuda_kernel<true,true,true>"}},
    };
    const bool sg = fuse_gated_norm_ && sigmoid_gate_;
    const char* entry = kEntries[beta.dtype() == bfloat16][fuse_gated_norm_][sg];
    auto kernel = cu::jit_kernel(encoder.device(), gdn_prefill_chain_source, entry);
    args.append(y_out);
    args.append(state_out);
    args.append<int32_t>(W);
    args.append<int32_t>(hk);
    args.append<int32_t>(hv);
    args.append<int32_t>(tiled_heads_);
    args.append<int64_t>(z_stride_b);
    args.append<int64_t>(z_stride_t);
    args.append<float>(eps_);
    args.append<int32_t>(w_is_f32);
    encoder.add_kernel_node_raw(
        kernel, dim3(hv, B, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GdnPrefillChainCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GdnPrefillChainCuda*>(&other);
    return rhs && rhs->tiled_heads_ == tiled_heads_ &&
        rhs->fuse_gated_norm_ == fuse_gated_norm_ && rhs->eps_ == eps_ &&
        rhs->sigmoid_gate_ == sigmoid_gate_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    if (fuse_gated_norm_) {
      const auto& v = inputs[2];
      return {{v.shape(0), v.shape(1), v.shape(2) * v.shape(3)}, inputs[5].shape()};
    }
    return {inputs[2].shape(), inputs[5].shape()};
  }

private:
  int tiled_heads_;
  bool fuse_gated_norm_;
  float eps_;
  bool sigmoid_gate_;
};

class GdnDeferredStepCuda : public Primitive {
public:
  GdnDeferredStepCuda(
      Stream stream, int prev_keep, float eps, bool tiled, bool sigmoid_gate)
      : Primitive(stream),
        prev_keep_(prev_keep),
        eps_(eps),
        tiled_(tiled),
        sigmoid_gate_(sigmoid_gate) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GDN deferred step CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array q = cu::ensure_row_contiguous_aligned<8>(inputs[0], stream(), encoder);
    array k = cu::ensure_row_contiguous_aligned<8>(inputs[1], stream(), encoder);
    array v = cu::ensure_row_contiguous_aligned<8>(inputs[2], stream(), encoder);
    array g = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);
    array beta = inputs[4].dtype() == bfloat16
        ? cu::ensure_row_contiguous_aligned<2>(inputs[4], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);
    array state_base =
        cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder);
    array z = inputs[6];
    const bool z_strided_ok =
        (z.ndim() == 3 && z.strides(2) == 1) ||
        (z.ndim() == 4 && z.strides(3) == 1 && z.strides(2) == 128);
    if (!z_strided_ok) {
      z = contiguous_copy_gpu(z, stream());
      encoder.add_temporary(z);
    }
    array w = inputs[7].dtype() == bfloat16
        ? cu::ensure_row_contiguous_aligned<2>(inputs[7], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<4>(inputs[7], stream(), encoder);

    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(g);
    encoder.set_input_array(beta);
    encoder.set_input_array(state_base);
    encoder.set_input_array(z);
    encoder.set_input_array(w);

    const int B = v.shape(0);
    const int T = v.shape(1);
    const int Hv = v.shape(2);
    const int Hk = q.shape(2);

    cu::KernelArgs args;
    args.append(q);
    args.append(k);
    args.append(v);
    args.append(g);
    args.append(beta);
    args.append(state_base);

    int prev_stride = 0;
    bool prev_beta_bf16 = true;
    if (prev_keep_ > 0) {
      array pk = cu::ensure_row_contiguous_aligned<8>(inputs[8], stream(), encoder);
      array pv = cu::ensure_row_contiguous_aligned<8>(inputs[9], stream(), encoder);
      array pg = cu::ensure_row_contiguous_aligned<4>(inputs[10], stream(), encoder);
      array pbeta = inputs[11].dtype() == bfloat16
          ? cu::ensure_row_contiguous_aligned<2>(inputs[11], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<4>(inputs[11], stream(), encoder);
      encoder.set_input_array(pk);
      encoder.set_input_array(pv);
      encoder.set_input_array(pg);
      encoder.set_input_array(pbeta);
      args.append(pk);
      args.append(pv);
      args.append(pg);
      args.append(pbeta);
      prev_stride = pv.shape(1);
      prev_beta_bf16 = (pbeta.dtype() == bfloat16);
    } else {
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
    }

    args.append(z);
    args.append(w);

    auto& out_y = outputs[0];
    out_y.set_data(cu::malloc_async(out_y.nbytes(), encoder));
    encoder.set_output_array(out_y);
    args.append(out_y);

    if (prev_keep_ > 0) {
      auto& out_state = outputs[1];
      out_state.set_data(cu::malloc_async(out_state.nbytes(), encoder));
      encoder.set_output_array(out_state);
      args.append(out_state);
    } else {
      args.append<CUdeviceptr>(0);
    }

    args.append<int32_t>(T);
    args.append<int32_t>(prev_keep_);
    args.append<int32_t>(prev_stride);
    args.append<int32_t>(Hk);
    args.append<int32_t>(Hv);
    args.append<int64_t>(z.strides(0));
    args.append<int64_t>(z.strides(1));
    args.append<float>(eps_);
    args.append<int32_t>(tiled_ ? 1 : 0);
    args.append<int32_t>(w.dtype() == float32 ? 1 : 0);

    static constexpr const char* kEntries[2][2][2] = {
        {{"mlx_gdn_deferred_step_cuda_kernel<false,false,false>",
          "mlx_gdn_deferred_step_cuda_kernel<false,false,true>"},
         {"mlx_gdn_deferred_step_cuda_kernel<false,true,false>",
          "mlx_gdn_deferred_step_cuda_kernel<false,true,true>"}},
        {{"mlx_gdn_deferred_step_cuda_kernel<true,false,false>",
          "mlx_gdn_deferred_step_cuda_kernel<true,false,true>"},
         {"mlx_gdn_deferred_step_cuda_kernel<true,true,false>",
          "mlx_gdn_deferred_step_cuda_kernel<true,true,true>"}},
    };
    const char* entry =
        kEntries[beta.dtype() == bfloat16][prev_beta_bf16][sigmoid_gate_];
    auto kernel = cu::jit_kernel(encoder.device(), gdn_fused_step_source, entry);
    encoder.add_kernel_node_raw(
        kernel, dim3(Hv, B, 1), dim3(512, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GdnDeferredStepCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GdnDeferredStepCuda*>(&other);
    return rhs && rhs->prev_keep_ == prev_keep_ && rhs->eps_ == eps_ &&
        rhs->tiled_ == tiled_ && rhs->sigmoid_gate_ == sigmoid_gate_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& v = inputs[2];
    Shape y_shape{v.shape(0), v.shape(1), v.shape(2) * v.shape(3)};
    if (prev_keep_ > 0) {
      return {y_shape, inputs[5].shape()};
    }
    return {y_shape};
  }

private:
  int prev_keep_;
  float eps_;
  bool tiled_;
  bool sigmoid_gate_;
};

class GdnTreeStepCuda : public Primitive {
public:
  GdnTreeStepCuda(
      Stream stream, int prev_keep, bool has_prev_indices, float eps,
      bool tiled, bool sigmoid_gate)
      : Primitive(stream),
        prev_keep_(prev_keep),
        has_prev_indices_(has_prev_indices),
        eps_(eps),
        tiled_(tiled),
        sigmoid_gate_(sigmoid_gate) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GDN tree step CUDA requires a GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array q = cu::ensure_row_contiguous_aligned<8>(inputs[0], stream(), encoder);
    array k = cu::ensure_row_contiguous_aligned<8>(inputs[1], stream(), encoder);
    array v = cu::ensure_row_contiguous_aligned<8>(inputs[2], stream(), encoder);
    array g = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);
    array beta = inputs[4].dtype() == bfloat16
        ? cu::ensure_row_contiguous_aligned<2>(inputs[4], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);
    array state_base =
        cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder);
    array z = inputs[6];
    const bool z_strided_ok =
        (z.ndim() == 3 && z.strides(2) == 1) ||
        (z.ndim() == 4 && z.strides(3) == 1 && z.strides(2) == 128);
    if (!z_strided_ok) {
      z = contiguous_copy_gpu(z, stream());
      encoder.add_temporary(z);
    }
    array w = inputs[7].dtype() == bfloat16
        ? cu::ensure_row_contiguous_aligned<2>(inputs[7], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<4>(inputs[7], stream(), encoder);
    array parent = cu::ensure_row_contiguous_aligned<4>(inputs[8], stream(), encoder);

    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(g);
    encoder.set_input_array(beta);
    encoder.set_input_array(state_base);
    encoder.set_input_array(z);
    encoder.set_input_array(w);
    encoder.set_input_array(parent);

    const int B = v.shape(0);
    const int T = v.shape(1);
    const int Hv = v.shape(2);
    const int Hk = q.shape(2);

    cu::KernelArgs args;
    args.append(q);
    args.append(k);
    args.append(v);
    args.append(g);
    args.append(beta);
    args.append(state_base);

    int prev_stride = 0;
    bool prev_beta_bf16 = true;
    if (prev_keep_ > 0) {
      array pk = cu::ensure_row_contiguous_aligned<8>(inputs[9], stream(), encoder);
      array pv = cu::ensure_row_contiguous_aligned<8>(inputs[10], stream(), encoder);
      array pg = cu::ensure_row_contiguous_aligned<4>(inputs[11], stream(), encoder);
      array pbeta = inputs[12].dtype() == bfloat16
          ? cu::ensure_row_contiguous_aligned<2>(inputs[12], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<4>(inputs[12], stream(), encoder);
      encoder.set_input_array(pk);
      encoder.set_input_array(pv);
      encoder.set_input_array(pg);
      encoder.set_input_array(pbeta);
      args.append(pk);
      args.append(pv);
      args.append(pg);
      args.append(pbeta);
      prev_stride = pv.shape(1);
      prev_beta_bf16 = (pbeta.dtype() == bfloat16);
      if (has_prev_indices_) {
        array pidx =
            cu::ensure_row_contiguous_aligned<4>(inputs[13], stream(), encoder);
        encoder.set_input_array(pidx);
        args.append(pidx);
      } else {
        args.append<CUdeviceptr>(0);
      }
    } else {
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
    }

    args.append(parent);
    args.append(z);
    args.append(w);

    auto& out_y = outputs[0];
    out_y.set_data(cu::malloc_async(out_y.nbytes(), encoder));
    encoder.set_output_array(out_y);
    args.append(out_y);

    if (prev_keep_ > 0) {
      auto& out_state = outputs[1];
      out_state.set_data(cu::malloc_async(out_state.nbytes(), encoder));
      encoder.set_output_array(out_state);
      args.append(out_state);
    } else {
      args.append<CUdeviceptr>(0);
    }

    args.append<int32_t>(T);
    args.append<int32_t>(prev_keep_);
    args.append<int32_t>(prev_stride);
    args.append<int32_t>(Hk);
    args.append<int32_t>(Hv);
    args.append<int64_t>(z.strides(0));
    args.append<int64_t>(z.strides(1));
    args.append<float>(eps_);
    args.append<int32_t>(tiled_ ? 1 : 0);
    args.append<int32_t>(w.dtype() == float32 ? 1 : 0);

    static constexpr const char* kEntries[2][2][2] = {
        {{"mlx_gdn_tree_step_cuda_kernel<false,false,false>",
          "mlx_gdn_tree_step_cuda_kernel<false,false,true>"},
         {"mlx_gdn_tree_step_cuda_kernel<false,true,false>",
          "mlx_gdn_tree_step_cuda_kernel<false,true,true>"}},
        {{"mlx_gdn_tree_step_cuda_kernel<true,false,false>",
          "mlx_gdn_tree_step_cuda_kernel<true,false,true>"},
         {"mlx_gdn_tree_step_cuda_kernel<true,true,false>",
          "mlx_gdn_tree_step_cuda_kernel<true,true,true>"}},
    };
    const char* entry =
        kEntries[beta.dtype() == bfloat16][prev_beta_bf16][sigmoid_gate_];
    auto kernel = cu::jit_kernel(encoder.device(), gdn_fused_step_source, entry);
    encoder.add_kernel_node_raw(
        kernel, dim3(Hv, B, 1), dim3(512, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(GdnTreeStepCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const GdnTreeStepCuda*>(&other);
    return rhs && rhs->prev_keep_ == prev_keep_ &&
        rhs->has_prev_indices_ == has_prev_indices_ &&
        rhs->eps_ == eps_ && rhs->tiled_ == tiled_ &&
        rhs->sigmoid_gate_ == sigmoid_gate_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& v = inputs[2];
    Shape y_shape{v.shape(0), v.shape(1), v.shape(2) * v.shape(3)};
    if (prev_keep_ > 0) {
      return {y_shape, inputs[5].shape()};
    }
    return {y_shape};
  }

private:
  int prev_keep_;
  bool has_prev_indices_;
  float eps_;
  bool tiled_;
  bool sigmoid_gate_;
};

} // namespace
} // namespace mlx::core
#endif

extern "C" bool mlx_gdn_fused_step_cuda_ex(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state, mlx_array* z, mlx_array* w, float eps, int32_t tiled_heads,
    bool sigmoid_gate, mlx_array* state_dst, mlx_array** out, mlx_array** out_state) {
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
        ba.ndim() != 3 || (za.ndim() != 3 && za.ndim() != 4) || wa.ndim() != 1) {
      return false;
    }
    const int B = va.shape(0);
    const int T = va.shape(1);
    const int Hv = va.shape(2);
    const int Dv = va.shape(3);
    const int Hk = qa.shape(2);
    const int Dk = qa.shape(3);
    const bool z_shape_ok =
        (za.ndim() == 3 && za.shape(0) == B && za.shape(1) == T &&
         za.shape(2) == Hv * Dv) ||
        (za.ndim() == 4 && za.shape(0) == B && za.shape(1) == T &&
         za.shape(2) == Hv && za.shape(3) == Dv);
    if (T < 1 || T > 32 || Dk != 128 || Dv != 128 || Hk <= 0 || Hv % Hk != 0 ||
        qa.shape() != Shape{B, T, Hk, Dk} || ka.shape() != qa.shape() ||
        ga.shape() != Shape{B, T, Hv} || ba.shape() != ga.shape() ||
        sa.shape() != Shape{B, Hv, Dv, Dk} || !z_shape_ok || wa.shape() != Shape{Dv}) {
      return false;
    }
    if (qa.dtype() != bfloat16 || ka.dtype() != bfloat16 || va.dtype() != bfloat16 ||
        (ba.dtype() != bfloat16 && ba.dtype() != float32) || za.dtype() != bfloat16 ||
        (wa.dtype() != bfloat16 && wa.dtype() != float32) ||
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
      auto has_buffer = [](const array& a) {
        return a.status() != array::Status::unscheduled &&
            a.data_shared_ptr() != nullptr && a.buffer().ptr() != nullptr;
      };
      if (has_buffer(dst)) {
        const void* dst_ptr = gpu_ptr<void>(dst);
        if ((reinterpret_cast<uintptr_t>(dst_ptr) & 15) != 0) {
          return false;
        }
        const void* dst_buf = dst.buffer().ptr();
        for (size_t i = 0; i < inputs.size(); ++i) {
          if (i == 5 || !has_buffer(inputs[i])) {
            continue;
          }
          if (dst_buf == inputs[i].buffer().ptr()) {
            return false;
          }
        }
        if (has_buffer(sa) && dst_buf == sa.buffer().ptr() &&
            dst_ptr != gpu_ptr<void>(sa)) {
          return false;
        }
      }
      inputs.push_back(dst);
    }
    auto stream = default_stream(Device::gpu);
    auto outputs = array::make_arrays(
        {Shape{B, T, Hv * Dv}, sa.shape()}, {bfloat16, float32},
        std::make_shared<GdnFusedStepCuda>(
            stream, eps, tiled_heads != 0, state_dst != nullptr, sigmoid_gate),
        std::move(inputs));
    *out = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_state = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_fused_step_cuda_ex: %s\n", error.what());
    return false;
  }
#else
  (void)q; (void)k; (void)v; (void)g; (void)beta; (void)state; (void)z; (void)w;
  (void)eps; (void)tiled_heads; (void)sigmoid_gate; (void)state_dst;
  if (out) { *out = nullptr; }
  if (out_state) { *out_state = nullptr; }
  return false;
#endif
}

extern "C" bool mlx_gdn_fused_step_cuda(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state, mlx_array* z, mlx_array* w, float eps, int32_t tiled_heads,
    mlx_array* state_dst, mlx_array** out, mlx_array** out_state) {
  return mlx_gdn_fused_step_cuda_ex(
      q, k, v, g, beta, state, z, w, eps, tiled_heads, false, state_dst, out,
      out_state);
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
        (ba.dtype() != bfloat16 && ba.dtype() != float32) || ga.dtype() != float32 ||
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
  (void)accepted_steps; (void)tiled_heads;
  if (out_state) { *out_state = nullptr; }
  return false;
#endif
}

extern "C" bool mlx_gdn_prepare_batched_cuda(
    mlx_array* qkv_handle, mlx_array* a_handle, mlx_array* b_handle,
    mlx_array* conv_handle, mlx_array* history_handle, mlx_array* scale_handle,
    mlx_array* dt_handle, mlx_array* valid_lengths_handle, bool mean_eps,
    bool bf16_beta, mlx_array** out_handles) {
#ifdef MLX_NODE_CUDA_ENABLED
  if (!out_handles) {
    return false;
  }
  for (int i = 0; i < 6; ++i) {
    out_handles[i] = nullptr;
  }
  try {
    using namespace mlx::core;
    if (default_device() != Device::gpu) {
      return false;
    }
    std::vector<array> in;
    in.reserve(valid_lengths_handle ? 8 : 7);
    for (auto* p : {qkv_handle, a_handle, b_handle, conv_handle, history_handle,
                    scale_handle, dt_handle}) {
      if (!p) {
        return false;
      }
      in.push_back(*reinterpret_cast<array*>(p));
    }
    const auto& x = in[0];
    const auto& a_arr = in[1];
    const auto& b_arr = in[2];
    if (x.ndim() != 3 || x.dtype() != bfloat16 ||
        a_arr.ndim() != 3 || b_arr.ndim() != 3) {
      return false;
    }
    const int B = x.shape(0);
    const int T = x.shape(1);
    const int W = x.shape(2);
    const int Hv = a_arr.shape(2);
    if (B < 1 || B > 65535 || T < 1 || T > 65535 || Hv < 1 ||
        static_cast<int64_t>(W) <= static_cast<int64_t>(Hv) * 128 ||
        ((W - Hv * 128) % 256) != 0) {
      return false;
    }
    const int Hk = (W - Hv * 128) / 256;
    if (Hk < 1 || (Hv % Hk) != 0) {
      return false;
    }
    if (a_arr.shape() != Shape{B, T, Hv} || b_arr.shape() != Shape{B, T, Hv} ||
        (a_arr.dtype() != float32 && a_arr.dtype() != bfloat16) ||
        (b_arr.dtype() != float32 && b_arr.dtype() != bfloat16)) {
      return false;
    }
    const auto& hist = in[4];
    const bool hist_shape_ok =
        (hist.ndim() == 2 && B == 1 && hist.shape() == Shape{3, W}) ||
        (hist.ndim() == 3 && hist.shape() == Shape{B, 3, W});
    if (in[3].size() != static_cast<size_t>(W) * 4 || in[3].dtype() != float32 ||
        !hist_shape_ok || hist.dtype() != bfloat16 ||
        in[5].size() != static_cast<size_t>(Hv) || in[5].dtype() != float32 ||
        in[6].size() != static_cast<size_t>(Hv) || in[6].dtype() != float32) {
      return false;
    }
    const Shape hist_shape = hist.shape();
    bool has_valid_lengths = false;
    if (valid_lengths_handle) {
      const auto& vl = *reinterpret_cast<array*>(valid_lengths_handle);
      if (vl.ndim() != 1 || vl.shape(0) != B || vl.dtype() != int32) {
        return false;
      }
      in.push_back(vl);
      has_valid_lengths = true;
    }
    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        {{B, T, Hk, 128}, {B, T, Hk, 128}, {B, T, Hv, 128},
         {B, T, Hv}, {B, T, Hv}, hist_shape},
        {bfloat16, bfloat16, bfloat16, float32,
         bf16_beta ? bfloat16 : float32, bfloat16},
        std::make_shared<GdnPrepareCuda>(
            stream, mean_eps, bf16_beta, has_valid_lengths),
        std::move(in));
    for (int i = 0; i < 6; ++i) {
      out_handles[i] = reinterpret_cast<mlx_array*>(new array(std::move(results[i])));
    }
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_prepare_batched_cuda: %s\n", error.what());
    return false;
  }
#else
  (void)qkv_handle; (void)a_handle; (void)b_handle; (void)conv_handle;
  (void)history_handle; (void)scale_handle; (void)dt_handle;
  (void)valid_lengths_handle; (void)mean_eps; (void)bf16_beta;
  if (out_handles) {
    for (int i = 0; i < 6; ++i) {
      out_handles[i] = nullptr;
    }
  }
  return false;
#endif
}

extern "C" bool mlx_gdn_prepare_cuda(
    mlx_array* qkv, mlx_array* a, mlx_array* b, mlx_array* conv,
    mlx_array* history, mlx_array* scale, mlx_array* dt, bool mean_eps,
    bool beta_input_dtype, mlx_array** outputs) {
  return mlx_gdn_prepare_batched_cuda(
      qkv, a, b, conv, history, scale, dt, nullptr, mean_eps,
      beta_input_dtype, outputs);
}

extern "C" bool mlx_gdn_prefill_chain_cuda(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state, int32_t tiled_heads,
    mlx_array** out_y, mlx_array** out_state) {
#ifdef MLX_NODE_CUDA_ENABLED
  if (out_y) {
    *out_y = nullptr;
  }
  if (out_state) {
    *out_state = nullptr;
  }
  try {
    using namespace mlx::core;
    if (!q || !k || !v || !g || !beta || !state || !out_y || !out_state ||
        default_device() != Device::gpu) {
      return false;
    }
    const auto& qa = *reinterpret_cast<array*>(q);
    const auto& ka = *reinterpret_cast<array*>(k);
    const auto& va = *reinterpret_cast<array*>(v);
    const auto& ga = *reinterpret_cast<array*>(g);
    const auto& ba = *reinterpret_cast<array*>(beta);
    const auto& sa = *reinterpret_cast<array*>(state);
    if (qa.ndim() != 4 || ka.ndim() != 4 || va.ndim() != 4 ||
        ga.ndim() != 3 || ba.ndim() != 3 || sa.ndim() != 4 ||
        qa.dtype() != bfloat16 || ka.dtype() != bfloat16 || va.dtype() != bfloat16 ||
        ga.dtype() != float32 || (ba.dtype() != bfloat16 && ba.dtype() != float32) ||
        sa.dtype() != float32) {
      return false;
    }
    const int B = qa.shape(0);
    const int W = qa.shape(1);
    const int hk = qa.shape(2);
    const int hv = va.shape(2);
    if (B < 1 || B > 65535 || W < 1 || hk < 1 || hv < 1 || (hv % hk) != 0 ||
        ka.shape(0) != B || va.shape(0) != B || ga.shape(0) != B ||
        ba.shape(0) != B || sa.shape(0) != B ||
        qa.shape(3) != 128 || ka.shape(3) != 128 || va.shape(3) != 128 ||
        ka.shape(1) != W || ka.shape(2) != hk || va.shape(1) != W ||
        ga.shape(1) != W || ga.shape(2) != hv ||
        ba.shape(1) != W || ba.shape(2) != hv ||
        sa.shape(1) != hv || sa.shape(2) != 128 || sa.shape(3) != 128) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        {va.shape(), sa.shape()},
        {bfloat16, float32},
        std::make_shared<GdnPrefillChainCuda>(
            stream, tiled_heads ? 1 : 0, false, 0.0f, false),
        {qa, ka, va, ga, ba, sa});
    *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_state = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_prefill_chain_cuda: %s\n", error.what());
    return false;
  }
#else
  (void)q; (void)k; (void)v; (void)g; (void)beta; (void)state;
  (void)tiled_heads;
  if (out_y) { *out_y = nullptr; }
  if (out_state) { *out_state = nullptr; }
  return false;
#endif
}

extern "C" bool mlx_gdn_prefill_chain_gated_cuda_ex(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state, mlx_array* z, mlx_array* norm_weight,
    float eps, int32_t tiled_heads, bool sigmoid_gate,
    mlx_array** out_y, mlx_array** out_state) {
#ifdef MLX_NODE_CUDA_ENABLED
  if (out_y) {
    *out_y = nullptr;
  }
  if (out_state) {
    *out_state = nullptr;
  }
  try {
    using namespace mlx::core;
    if (!q || !k || !v || !g || !beta || !state || !z || !norm_weight ||
        !out_y || !out_state || default_device() != Device::gpu) {
      return false;
    }
    const auto& qa = *reinterpret_cast<array*>(q);
    const auto& ka = *reinterpret_cast<array*>(k);
    const auto& va = *reinterpret_cast<array*>(v);
    const auto& ga = *reinterpret_cast<array*>(g);
    const auto& ba = *reinterpret_cast<array*>(beta);
    const auto& sa = *reinterpret_cast<array*>(state);
    const auto& za = *reinterpret_cast<array*>(z);
    const auto& wa = *reinterpret_cast<array*>(norm_weight);
    if (qa.ndim() != 4 || ka.ndim() != 4 || va.ndim() != 4 ||
        ga.ndim() != 3 || ba.ndim() != 3 || sa.ndim() != 4 ||
        (za.ndim() != 3 && za.ndim() != 4) || wa.ndim() != 1 ||
        qa.dtype() != bfloat16 || ka.dtype() != bfloat16 || va.dtype() != bfloat16 ||
        ga.dtype() != float32 || (ba.dtype() != bfloat16 && ba.dtype() != float32) ||
        sa.dtype() != float32 || za.dtype() != bfloat16 ||
        (wa.dtype() != bfloat16 && wa.dtype() != float32)) {
      return false;
    }
    const int B = qa.shape(0);
    const int W = qa.shape(1);
    const int hk = qa.shape(2);
    const int hv = va.shape(2);
    const bool z_shape_ok =
        (za.ndim() == 3 && za.shape(0) == B && za.shape(1) == W && za.shape(2) == hv * 128) ||
        (za.ndim() == 4 && za.shape(0) == B && za.shape(1) == W && za.shape(2) == hv && za.shape(3) == 128);
    if (B < 1 || B > 65535 || W < 1 || hk < 1 || hv < 1 || (hv % hk) != 0 ||
        ka.shape(0) != B || va.shape(0) != B || ga.shape(0) != B ||
        ba.shape(0) != B || sa.shape(0) != B ||
        qa.shape(3) != 128 || ka.shape(3) != 128 || va.shape(3) != 128 ||
        ka.shape(1) != W || ka.shape(2) != hk || va.shape(1) != W ||
        ga.shape(1) != W || ga.shape(2) != hv ||
        ba.shape(1) != W || ba.shape(2) != hv ||
        sa.shape(1) != hv || sa.shape(2) != 128 || sa.shape(3) != 128 ||
        !z_shape_ok || wa.shape(0) != 128) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        {Shape{B, W, hv * 128}, sa.shape()},
        {bfloat16, float32},
        std::make_shared<GdnPrefillChainCuda>(
            stream, tiled_heads ? 1 : 0, true, eps, sigmoid_gate),
        {qa, ka, va, ga, ba, sa, za, wa});
    *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_state = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_prefill_chain_gated_cuda_ex: %s\n", error.what());
    return false;
  }
#else
  (void)q; (void)k; (void)v; (void)g; (void)beta; (void)state;
  (void)z; (void)norm_weight; (void)eps; (void)tiled_heads; (void)sigmoid_gate;
  if (out_y) { *out_y = nullptr; }
  if (out_state) { *out_state = nullptr; }
  return false;
#endif
}

extern "C" bool mlx_gdn_prefill_chain_gated_cuda(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state, mlx_array* z, mlx_array* norm_weight,
    float eps, int32_t tiled_heads,
    mlx_array** out_y, mlx_array** out_state) {
  return mlx_gdn_prefill_chain_gated_cuda_ex(
      q, k, v, g, beta, state, z, norm_weight, eps, tiled_heads, false,
      out_y, out_state);
}

extern "C" bool mlx_gdn_fused_step_deferred_cuda_ex(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state_base,
    mlx_array* prev_k, mlx_array* prev_v, mlx_array* prev_g, mlx_array* prev_beta,
    int32_t prev_keep,
    mlx_array* z, mlx_array* norm_weight, float eps, int32_t tiled_heads,
    bool sigmoid_gate,
    mlx_array** out_y, mlx_array** out_committed_state) {
#ifdef MLX_NODE_CUDA_ENABLED
  if (out_y) {
    *out_y = nullptr;
  }
  if (out_committed_state) {
    *out_committed_state = nullptr;
  }
  try {
    if (!q || !k || !v || !g || !beta || !state_base || !z || !norm_weight ||
        !out_y || !out_committed_state || prev_keep < 0) {
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
    const auto& sa = *reinterpret_cast<array*>(state_base);
    const auto& za = *reinterpret_cast<array*>(z);
    const auto& wa = *reinterpret_cast<array*>(norm_weight);
    if (qa.ndim() != 4 || ka.ndim() != 4 || va.ndim() != 4 ||
        ga.ndim() != 3 || ba.ndim() != 3 || sa.ndim() != 4 ||
        (za.ndim() != 3 && za.ndim() != 4) || wa.ndim() != 1) {
      return false;
    }
    const int B = va.shape(0);
    const int T = va.shape(1);
    const int Hv = va.shape(2);
    const int Dv = va.shape(3);
    const int Hk = qa.shape(2);
    const int Dk = qa.shape(3);
    const bool z_shape_ok =
        (za.ndim() == 3 && za.shape(0) == B && za.shape(1) == T &&
         za.shape(2) == Hv * Dv) ||
        (za.ndim() == 4 && za.shape(0) == B && za.shape(1) == T &&
         za.shape(2) == Hv && za.shape(3) == Dv);
    if (B < 1 || T < 1 || T > 16 || Dk != 128 || Dv != 128 || Hk <= 0 ||
        Hv % Hk != 0 || qa.shape() != Shape{B, T, Hk, Dk} ||
        ka.shape() != qa.shape() || ga.shape() != Shape{B, T, Hv} ||
        ba.shape() != ga.shape() || sa.shape() != Shape{B, Hv, Dv, Dk} ||
        !z_shape_ok || wa.shape() != Shape{Dv}) {
      return false;
    }
    if (qa.dtype() != bfloat16 || ka.dtype() != bfloat16 || va.dtype() != bfloat16 ||
        ga.dtype() != float32 || (ba.dtype() != bfloat16 && ba.dtype() != float32) ||
        sa.dtype() != float32 || za.dtype() != bfloat16 ||
        (wa.dtype() != bfloat16 && wa.dtype() != float32)) {
      return false;
    }
    std::vector<array> inputs{qa, ka, va, ga, ba, sa, za, wa};
    if (prev_keep > 0) {
      if (!prev_k || !prev_v || !prev_g || !prev_beta) {
        return false;
      }
      const auto& pka = *reinterpret_cast<array*>(prev_k);
      const auto& pva = *reinterpret_cast<array*>(prev_v);
      const auto& pga = *reinterpret_cast<array*>(prev_g);
      const auto& pba = *reinterpret_cast<array*>(prev_beta);
      if (pka.ndim() != 4 || pva.ndim() != 4 || pga.ndim() != 3 || pba.ndim() != 3) {
        return false;
      }
      const int S = pva.shape(1);
      if (prev_keep > S || pka.shape() != Shape{B, S, Hk, Dk} ||
          pva.shape() != Shape{B, S, Hv, Dv} || pga.shape() != Shape{B, S, Hv} ||
          pba.shape() != Shape{B, S, Hv}) {
        return false;
      }
      if (pka.dtype() != bfloat16 || pva.dtype() != bfloat16 ||
          pga.dtype() != float32 ||
          (pba.dtype() != bfloat16 && pba.dtype() != float32)) {
        return false;
      }
      inputs.push_back(pka);
      inputs.push_back(pva);
      inputs.push_back(pga);
      inputs.push_back(pba);
    }
    auto stream = default_stream(Device::gpu);
    Shape y_shape{B, T, Hv * Dv};
    if (prev_keep > 0) {
      auto results = array::make_arrays(
          {y_shape, sa.shape()}, {bfloat16, float32},
          std::make_shared<GdnDeferredStepCuda>(
              stream, prev_keep, eps, tiled_heads != 0, sigmoid_gate),
          std::move(inputs));
      *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      *out_committed_state =
          reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    } else {
      auto results = array::make_arrays(
          {y_shape}, {bfloat16},
          std::make_shared<GdnDeferredStepCuda>(
              stream, 0, eps, tiled_heads != 0, sigmoid_gate),
          std::move(inputs));
      *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      *out_committed_state = reinterpret_cast<mlx_array*>(new array(sa));
    }
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_fused_step_deferred_cuda_ex: %s\n", error.what());
    return false;
  }
#else
  (void)q; (void)k; (void)v; (void)g; (void)beta; (void)state_base;
  (void)prev_k; (void)prev_v; (void)prev_g; (void)prev_beta;
  (void)prev_keep; (void)z; (void)norm_weight; (void)eps; (void)tiled_heads;
  (void)sigmoid_gate;
  if (out_y) { *out_y = nullptr; }
  if (out_committed_state) { *out_committed_state = nullptr; }
  return false;
#endif
}

extern "C" bool mlx_gdn_fused_step_deferred_cuda(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state_base,
    mlx_array* prev_k, mlx_array* prev_v, mlx_array* prev_g, mlx_array* prev_beta,
    int32_t prev_keep,
    mlx_array* z, mlx_array* norm_weight, float eps, int32_t tiled_heads,
    mlx_array** out_y, mlx_array** out_committed_state) {
  return mlx_gdn_fused_step_deferred_cuda_ex(
      q, k, v, g, beta, state_base, prev_k, prev_v, prev_g, prev_beta,
      prev_keep, z, norm_weight, eps, tiled_heads, false, out_y,
      out_committed_state);
}

extern "C" bool mlx_gdn_tree_step_cuda_ex(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state_base,
    mlx_array* prev_k, mlx_array* prev_v, mlx_array* prev_g, mlx_array* prev_beta,
    int32_t prev_keep,
    mlx_array* prev_indices,
    mlx_array* parent,
    mlx_array* z, mlx_array* norm_weight, float eps, int32_t tiled_heads,
    bool sigmoid_gate,
    mlx_array** out_y, mlx_array** out_committed_state) {
#ifdef MLX_NODE_CUDA_ENABLED
  if (out_y) {
    *out_y = nullptr;
  }
  if (out_committed_state) {
    *out_committed_state = nullptr;
  }
  try {
    if (!q || !k || !v || !g || !beta || !state_base || !parent || !z ||
        !norm_weight || !out_y || !out_committed_state || prev_keep < 0) {
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
    const auto& sa = *reinterpret_cast<array*>(state_base);
    const auto& pa = *reinterpret_cast<array*>(parent);
    const auto& za = *reinterpret_cast<array*>(z);
    const auto& wa = *reinterpret_cast<array*>(norm_weight);
    if (qa.ndim() != 4 || ka.ndim() != 4 || va.ndim() != 4 ||
        ga.ndim() != 3 || ba.ndim() != 3 || sa.ndim() != 4 ||
        pa.ndim() != 1 || (za.ndim() != 3 && za.ndim() != 4) || wa.ndim() != 1) {
      return false;
    }
    const int B = va.shape(0);
    const int T = va.shape(1);
    const int Hv = va.shape(2);
    const int Dv = va.shape(3);
    const int Hk = qa.shape(2);
    const int Dk = qa.shape(3);
    const bool z_shape_ok =
        (za.ndim() == 3 && za.shape(0) == B && za.shape(1) == T &&
         za.shape(2) == Hv * Dv) ||
        (za.ndim() == 4 && za.shape(0) == B && za.shape(1) == T &&
         za.shape(2) == Hv && za.shape(3) == Dv);
    if (B < 1 || T < 1 || T > 16 || Dk != 128 || Dv != 128 || Hk <= 0 ||
        Hv % Hk != 0 || qa.shape() != Shape{B, T, Hk, Dk} ||
        ka.shape() != qa.shape() || ga.shape() != Shape{B, T, Hv} ||
        ba.shape() != ga.shape() || sa.shape() != Shape{B, Hv, Dv, Dk} ||
        pa.shape() != Shape{T} || !z_shape_ok || wa.shape() != Shape{Dv}) {
      return false;
    }
    if (qa.dtype() != bfloat16 || ka.dtype() != bfloat16 || va.dtype() != bfloat16 ||
        ga.dtype() != float32 || (ba.dtype() != bfloat16 && ba.dtype() != float32) ||
        sa.dtype() != float32 || pa.dtype() != int32 || za.dtype() != bfloat16 ||
        (wa.dtype() != bfloat16 && wa.dtype() != float32)) {
      return false;
    }
    std::vector<array> inputs{qa, ka, va, ga, ba, sa, za, wa, pa};
    bool has_prev_indices = false;
    if (prev_keep > 0) {
      if (!prev_k || !prev_v || !prev_g || !prev_beta) {
        return false;
      }
      const auto& pka = *reinterpret_cast<array*>(prev_k);
      const auto& pva = *reinterpret_cast<array*>(prev_v);
      const auto& pga = *reinterpret_cast<array*>(prev_g);
      const auto& pba = *reinterpret_cast<array*>(prev_beta);
      if (pka.ndim() != 4 || pva.ndim() != 4 || pga.ndim() != 3 || pba.ndim() != 3) {
        return false;
      }
      const int S = pva.shape(1);
      if (prev_keep > S || pka.shape() != Shape{B, S, Hk, Dk} ||
          pva.shape() != Shape{B, S, Hv, Dv} || pga.shape() != Shape{B, S, Hv} ||
          pba.shape() != Shape{B, S, Hv}) {
        return false;
      }
      if (pka.dtype() != bfloat16 || pva.dtype() != bfloat16 ||
          pga.dtype() != float32 ||
          (pba.dtype() != bfloat16 && pba.dtype() != float32)) {
        return false;
      }
      inputs.push_back(pka);
      inputs.push_back(pva);
      inputs.push_back(pga);
      inputs.push_back(pba);
      if (prev_indices) {
        const auto& pidxa = *reinterpret_cast<array*>(prev_indices);
        if (pidxa.ndim() != 1 || pidxa.shape(0) != prev_keep ||
            pidxa.dtype() != int32) {
          return false;
        }
        inputs.push_back(pidxa);
        has_prev_indices = true;
      }
    }
    auto stream = default_stream(Device::gpu);
    Shape y_shape{B, T, Hv * Dv};
    if (prev_keep > 0) {
      auto results = array::make_arrays(
          {y_shape, sa.shape()}, {bfloat16, float32},
          std::make_shared<GdnTreeStepCuda>(
              stream, prev_keep, has_prev_indices, eps, tiled_heads != 0,
              sigmoid_gate),
          std::move(inputs));
      *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      *out_committed_state =
          reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    } else {
      auto results = array::make_arrays(
          {y_shape}, {bfloat16},
          std::make_shared<GdnTreeStepCuda>(
              stream, 0, false, eps, tiled_heads != 0, sigmoid_gate),
          std::move(inputs));
      *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      *out_committed_state = reinterpret_cast<mlx_array*>(new array(sa));
    }
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mlx_gdn_tree_step_cuda_ex: %s\n", error.what());
    return false;
  }
#else
  (void)q; (void)k; (void)v; (void)g; (void)beta; (void)state_base;
  (void)prev_k; (void)prev_v; (void)prev_g; (void)prev_beta;
  (void)prev_keep; (void)prev_indices; (void)parent;
  (void)z; (void)norm_weight; (void)eps; (void)tiled_heads; (void)sigmoid_gate;
  if (out_y) { *out_y = nullptr; }
  if (out_committed_state) { *out_committed_state = nullptr; }
  return false;
#endif
}

extern "C" bool mlx_gdn_tree_step_cuda(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state_base,
    mlx_array* prev_k, mlx_array* prev_v, mlx_array* prev_g, mlx_array* prev_beta,
    int32_t prev_keep,
    mlx_array* prev_indices,
    mlx_array* parent,
    mlx_array* z, mlx_array* norm_weight, float eps, int32_t tiled_heads,
    mlx_array** out_y, mlx_array** out_committed_state) {
  return mlx_gdn_tree_step_cuda_ex(
      q, k, v, g, beta, state_base, prev_k, prev_v, prev_g, prev_beta,
      prev_keep, prev_indices, parent, z, norm_weight, eps, tiled_heads, false,
      out_y, out_committed_state);
}
