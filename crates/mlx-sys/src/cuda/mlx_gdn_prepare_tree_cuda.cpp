#include "mlx_common.h"
#include <cstdio>

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace {

constexpr char gdn_prepare_tree_cuda_source[] =
#include "cuda/gdn_prepare_tree.cu.inc"
    ;

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

}  // namespace
#endif

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
  (void)qkv;
  (void)a;
  (void)b;
  (void)conv;
  (void)history;
  (void)scale;
  (void)dt;
  (void)conv_taps;
  (void)mean_eps;
  (void)beta_input_dtype;
  return false;
#endif
}
