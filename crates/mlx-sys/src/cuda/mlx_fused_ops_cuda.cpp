#include "mlx_common.h"

#include <cstdio>
#include <limits>
#include <stdexcept>

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace {

constexpr char add_rmsnorm_cuda_source[] =
#include "cuda/add_rmsnorm.cu.inc"
    ;

class FusedAddRmsNormCuda final : public mlx::core::Primitive {
 public:
  FusedAddRmsNormCuda(
      mlx::core::Stream stream,
      bool precise_f32 = false,
      bool has_res = true,
      bool has_eps_arr = true,
      float eps_val = 0.0f)
      : Primitive(stream),
        precise_f32_(precise_f32),
        has_res_(has_res),
        has_eps_arr_(has_eps_arr),
        eps_val_(eps_val) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("FusedAddRmsNormCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    using namespace mlx::core;
    auto& encoder = cu::get_command_encoder(stream());
    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    const auto& x = owned[0];
    const auto& w = has_res_ ? owned[2] : owned[1];
    const int64_t axis64 = x.shape(-1);
    const size_t total = x.size();
    if (x.ndim() < 1 || axis64 < 1 ||
        axis64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
        total == 0 || (total % static_cast<size_t>(axis64)) != 0 ||
        w.size() != static_cast<size_t>(axis64)) {
      throw std::invalid_argument("FusedAddRmsNormCuda: invalid shape");
    }
    const size_t rows64 = total / static_cast<size_t>(axis64);
    if (rows64 < 1 ||
        rows64 > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      throw std::invalid_argument("FusedAddRmsNormCuda: invalid row count");
    }
    const auto dt = x.dtype();
    const auto expected_w_dt = precise_f32_ ? float32 : dt;
    if (w.dtype() != expected_w_dt ||
        (precise_f32_ && dt != bfloat16 && dt != float16) ||
        (!precise_f32_ && dt != bfloat16 && dt != float16 && dt != float32)) {
      throw std::invalid_argument("FusedAddRmsNormCuda: invalid dtype");
    }
    if (has_res_) {
      const auto& res = owned[1];
      if (res.shape() != x.shape() || res.dtype() != dt) {
        throw std::invalid_argument("FusedAddRmsNormCuda: residual mismatch");
      }
    }

    encoder.set_input_array(x);
    if (has_res_) {
      encoder.set_input_array(owned[1]);
    }
    encoder.set_input_array(w);
    if (has_eps_arr_) {
      const auto& eps_arr = has_res_ ? owned[3] : owned[2];
      encoder.set_input_array(eps_arr);
    }

    if (has_res_) {
      auto& h_out = outputs[0];
      auto& normed_out = outputs[1];
      h_out.set_data(cu::malloc_async(h_out.nbytes(), encoder));
      normed_out.set_data(cu::malloc_async(normed_out.nbytes(), encoder));
      encoder.set_output_array(h_out);
      encoder.set_output_array(normed_out);
    } else {
      auto& normed_out = outputs[0];
      normed_out.set_data(cu::malloc_async(normed_out.nbytes(), encoder));
      encoder.set_output_array(normed_out);
    }

    const int axis = static_cast<int>(axis64);
    const int rows = static_cast<int>(rows64);
    const bool use_256 = axis <= 2048;
    const int block_size = use_256 ? 256 : 512;

    const int dt_idx = (dt == bfloat16) ? 0 : (dt == float16 ? 1 : 2);
    const int bs_idx = use_256 ? 0 : 1;
    const int p_idx = precise_f32_ ? 1 : 0;
    const int r_idx = has_res_ ? 1 : 0;
    static constexpr const char* kEntries[3][2][2][2] = {
        {
            {
                {"mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,256,false,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,256,false,true>"},
                {"mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,256,true,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,256,true,true>"},
            },
            {
                {"mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,512,false,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,512,false,true>"},
                {"mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,512,true,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<nv_bfloat16,512,true,true>"},
            },
        },
        {
            {
                {"mlx_fused_add_rmsnorm_cuda_kernel<__half,256,false,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<__half,256,false,true>"},
                {"mlx_fused_add_rmsnorm_cuda_kernel<__half,256,true,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<__half,256,true,true>"},
            },
            {
                {"mlx_fused_add_rmsnorm_cuda_kernel<__half,512,false,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<__half,512,false,true>"},
                {"mlx_fused_add_rmsnorm_cuda_kernel<__half,512,true,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<__half,512,true,true>"},
            },
        },
        {
            {
                {"mlx_fused_add_rmsnorm_cuda_kernel<float,256,false,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<float,256,false,true>"},
                {"mlx_fused_add_rmsnorm_cuda_kernel<float,256,true,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<float,256,true,true>"},
            },
            {
                {"mlx_fused_add_rmsnorm_cuda_kernel<float,512,false,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<float,512,false,true>"},
                {"mlx_fused_add_rmsnorm_cuda_kernel<float,512,true,false>",
                 "mlx_fused_add_rmsnorm_cuda_kernel<float,512,true,true>"},
            },
        },
    };
    const char* entry = kEntries[dt_idx][bs_idx][p_idx][r_idx];
    auto kernel = cu::jit_kernel(encoder.device(), add_rmsnorm_cuda_source, entry);
    cu::KernelArgs args;
    args.append(x);
    if (has_res_) {
      args.append(owned[1]);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(w);
    if (has_eps_arr_) {
      args.append(has_res_ ? owned[3] : owned[2]);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append<float>(eps_val_);
    if (has_res_) {
      args.append(outputs[0]);
      args.append(outputs[1]);
    } else {
      args.append<CUdeviceptr>(0);
      args.append(outputs[0]);
    }
    args.append<int32_t>(rows);
    args.append<int32_t>(axis);
    encoder.add_kernel_node_raw(
        kernel, dim3(static_cast<uint32_t>(rows), 1, 1),
        dim3(block_size, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(FusedAddRmsNormCuda)
  bool is_equivalent(const mlx::core::Primitive& other) const override {
    const auto* rhs = dynamic_cast<const FusedAddRmsNormCuda*>(&other);
    return rhs && rhs->precise_f32_ == precise_f32_ &&
        rhs->has_res_ == has_res_ && rhs->has_eps_arr_ == has_eps_arr_ &&
        (has_eps_arr_ || rhs->eps_val_ == eps_val_);
  }
  std::vector<mlx::core::Shape> output_shapes(const std::vector<array>& inputs) override {
    if (has_res_) {
      return {inputs[0].shape(), inputs[0].shape()};
    }
    return {inputs[0].shape()};
  }

 private:
  bool precise_f32_;
  bool has_res_;
  bool has_eps_arr_;
  float eps_val_;
};

class FusedGateBinaryCuda final : public mlx::core::UnaryPrimitive {
 public:
  FusedGateBinaryCuda(mlx::core::Stream stream, bool is_swiglu)
      : UnaryPrimitive(stream), is_swiglu_(is_swiglu) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("FusedGateBinaryCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    using namespace mlx::core;
    auto& encoder = cu::get_command_encoder(stream());
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    const int64_t n = static_cast<int64_t>(inputs[0].size());
    if (n <= 0) {
      return;
    }

    auto is_2d_row_strided = [](const array& arr, int64_t& cols, int64_t& rs) -> bool {
      const int nd = arr.ndim();
      if (nd < 2 || arr.strides(nd - 1) != 1) {
        return false;
      }
      cols = arr.shape(nd - 1);
      rs = arr.strides(nd - 2);
      if (cols <= 0 || rs < 0) {
        return false;
      }
      int64_t expected_stride = rs * arr.shape(nd - 2);
      for (int d = nd - 3; d >= 0; --d) {
        if (arr.shape(d) > 1 && arr.strides(d) != expected_stride) {
          return false;
        }
        expected_stride *= arr.shape(d);
      }
      return true;
    };

    const bool both_contig =
        inputs[0].flags().row_contiguous && inputs[1].flags().row_contiguous;
    int64_t a_cols = 0, a_rs = 0, b_cols = 0, b_rs = 0;
    if (!both_contig &&
        is_2d_row_strided(inputs[0], a_cols, a_rs) &&
        is_2d_row_strided(inputs[1], b_cols, b_rs) &&
        a_cols == b_cols) {
      const array& a = inputs[0];
      const array& b = inputs[1];
      const int64_t cols = a_cols;
      const int64_t rows = n / cols;
      encoder.set_input_array(a);
      encoder.set_input_array(b);
      encoder.set_output_array(out);
      const char* entry = nullptr;
      if (is_swiglu_) {
        entry = a.dtype() == bfloat16 ? "mlx_swiglu_strided_cuda_kernel<nv_bfloat16>"
            : a.dtype() == float16    ? "mlx_swiglu_strided_cuda_kernel<__half>"
                                      : "mlx_swiglu_strided_cuda_kernel<float>";
      } else {
        entry = a.dtype() == bfloat16 ? "mlx_sigmoid_mul_strided_cuda_kernel<nv_bfloat16>"
            : a.dtype() == float16    ? "mlx_sigmoid_mul_strided_cuda_kernel<__half>"
                                      : "mlx_sigmoid_mul_strided_cuda_kernel<float>";
      }
      const bool vec8 =
          (a.itemsize() == 2) && ((cols & 7) == 0) && ((a_rs & 7) == 0) && ((b_rs & 7) == 0) &&
          (((reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) |
             reinterpret_cast<uintptr_t>(gpu_ptr<void>(b)) |
             reinterpret_cast<uintptr_t>(gpu_ptr<void>(out))) & 15u) == 0);
      const int64_t work = vec8 ? (rows * (cols >> 3)) : n;
      auto kernel = cu::jit_kernel(encoder.device(), add_rmsnorm_cuda_source, entry);
      cu::KernelArgs args;
      args.append(a);
      args.append(b);
      args.append(out);
      args.append<int64_t>(rows);
      args.append<int64_t>(cols);
      args.append<int64_t>(a_rs);
      args.append<int64_t>(b_rs);
      encoder.add_kernel_node_raw(
          kernel, dim3(static_cast<uint32_t>((work + 255) / 256), 1, 1),
          dim3(256, 1, 1), {}, 0, args.args());
      return;
    }

    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    const auto& a = owned[0];
    const auto& b = owned[1];
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    const char* entry = nullptr;
    if (is_swiglu_) {
      entry = a.dtype() == bfloat16 ? "mlx_swiglu_cuda_kernel<nv_bfloat16>"
          : a.dtype() == float16    ? "mlx_swiglu_cuda_kernel<__half>"
                                    : "mlx_swiglu_cuda_kernel<float>";
    } else {
      entry = a.dtype() == bfloat16 ? "mlx_sigmoid_mul_cuda_kernel<nv_bfloat16>"
          : a.dtype() == float16    ? "mlx_sigmoid_mul_cuda_kernel<__half>"
                                    : "mlx_sigmoid_mul_cuda_kernel<float>";
    }
    const bool vec8 =
        (a.itemsize() == 2) && ((n & 7) == 0) &&
        (((reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) |
           reinterpret_cast<uintptr_t>(gpu_ptr<void>(b)) |
           reinterpret_cast<uintptr_t>(gpu_ptr<void>(out))) & 15u) == 0);
    const int64_t work = vec8 ? (n >> 3) : n;
    auto kernel = cu::jit_kernel(encoder.device(), add_rmsnorm_cuda_source, entry);
    cu::KernelArgs args;
    args.append(a);
    args.append(b);
    args.append(out);
    args.append<int64_t>(n);
    encoder.add_kernel_node_raw(
        kernel, dim3(static_cast<uint32_t>((work + 255) / 256), 1, 1),
        dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(FusedGateBinaryCuda)
  bool is_equivalent(const mlx::core::Primitive& other) const override {
    const auto* rhs = dynamic_cast<const FusedGateBinaryCuda*>(&other);
    return rhs && rhs->is_swiglu_ == is_swiglu_;
  }
  std::vector<mlx::core::Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  bool is_swiglu_;
};

}  // namespace
#endif

namespace mlx_node_internal {

#ifdef MLX_NODE_CUDA_ENABLED
std::optional<mlx::core::array> fused_gate_binary_cuda(
    const mlx::core::array& gate,
    const mlx::core::array& value,
    bool is_swiglu) {
  if (mlx::core::default_device() == mlx::core::Device::gpu &&
      gate.shape() == value.shape() && gate.dtype() == value.dtype() &&
      (gate.dtype() == mlx::core::bfloat16 ||
       gate.dtype() == mlx::core::float16 ||
       gate.dtype() == mlx::core::float32)) {
    return mlx::core::array(
        gate.shape(),
        gate.dtype(),
        std::make_shared<FusedGateBinaryCuda>(
            mlx::core::default_stream(mlx::core::Device::gpu), is_swiglu),
        {gate, value});
  }
  return std::nullopt;
}

std::optional<std::pair<mlx::core::array, mlx::core::array>>
fused_add_rmsnorm_cuda(
    const mlx::core::array& x,
    const mlx::core::array& res,
    const mlx::core::array& w,
    const mlx::core::array& eps) {
  if (mlx::core::default_device() != mlx::core::Device::gpu) {
    return std::nullopt;
  }
  const auto dt = x.dtype();
  auto outs = mlx::core::array::make_arrays(
      {x.shape(), x.shape()},
      {dt, dt},
      std::make_shared<FusedAddRmsNormCuda>(
          mlx::core::default_stream(mlx::core::Device::gpu)),
      {x, res, w, eps});
  return std::make_pair(std::move(outs[0]), std::move(outs[1]));
}
#else
std::optional<mlx::core::array> fused_gate_binary_cuda(
    const mlx::core::array&,
    const mlx::core::array&,
    bool) {
  return std::nullopt;
}

std::optional<std::pair<mlx::core::array, mlx::core::array>>
fused_add_rmsnorm_cuda(
    const mlx::core::array&,
    const mlx::core::array&,
    const mlx::core::array&,
    const mlx::core::array&) {
  return std::nullopt;
}
#endif

}  // namespace mlx_node_internal

// Fused residual-add + PreciseF32 RMSNorm on CUDA:
//   h = T(float(x) + float(residual))
//   normed = T((float(h) * rsqrt(mean(h^2) + eps)) * weight_f32)
// Bit-identical to h = x + residual; rms_norm(h.astype(f32), weight_f32, eps).astype(T).
extern "C" bool mlx_fused_add_rmsnorm_precise_f32_cuda(
    mlx_array* x_handle,
    mlx_array* residual_handle,
    mlx_array* weight_f32_handle,
    float eps,
    mlx_array** out_h,
    mlx_array** out_normed) {
  if (out_h) *out_h = nullptr;
  if (out_normed) *out_normed = nullptr;
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!x_handle || !residual_handle || !weight_f32_handle || !out_h ||
        !out_normed || default_device() != Device::gpu) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_handle);
    const auto& res = *reinterpret_cast<array*>(residual_handle);
    const auto& w = *reinterpret_cast<array*>(weight_f32_handle);
    const auto dt = x.dtype();
    if (x.ndim() < 1 || res.shape() != x.shape() || res.dtype() != dt ||
        (dt != bfloat16 && dt != float16) || w.dtype() != float32) {
      return false;
    }
    const int64_t axis64 = x.shape(-1);
    if (axis64 < 1 ||
        axis64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }
    const size_t axis = static_cast<size_t>(axis64);
    const size_t total = x.size();
    if (w.size() != axis || total == 0 || (total % axis) != 0 ||
        res.size() != total) {
      return false;
    }
    const size_t rows64 = total / axis;
    if (rows64 < 1 ||
        rows64 > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }
    auto outs = array::make_arrays(
        {x.shape(), x.shape()},
        {dt, dt},
        std::make_shared<FusedAddRmsNormCuda>(
            default_stream(Device::gpu),
            /*precise_f32=*/true,
            /*has_res=*/true,
            /*has_eps_arr=*/false,
            eps),
        {x, res, w});
    *out_h = reinterpret_cast<mlx_array*>(new array(std::move(outs[0])));
    *out_normed = reinterpret_cast<mlx_array*>(new array(std::move(outs[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(
        stderr, "mlx_fused_add_rmsnorm_precise_f32_cuda: %s\n", e.what());
    return false;
  } catch (...) {
    std::fprintf(
        stderr, "mlx_fused_add_rmsnorm_precise_f32_cuda: unknown exception\n");
    return false;
  }
#else
  (void)x_handle;
  (void)residual_handle;
  (void)weight_f32_handle;
  (void)eps;
  return false;
#endif
}

// Standalone PreciseF32 RMSNorm on CUDA (no residual):
//   normed = T((float(x) * rsqrt(mean(x^2) + eps)) * weight_f32)
// Bit-identical to rms_norm(x.astype(f32), weight_f32, eps).astype(T).
extern "C" mlx_array* mlx_rmsnorm_precise_f32_cuda(
    mlx_array* x_handle,
    mlx_array* weight_f32_handle,
    float eps) {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!x_handle || !weight_f32_handle || default_device() != Device::gpu) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_handle);
    const auto& w = *reinterpret_cast<array*>(weight_f32_handle);
    const auto dt = x.dtype();
    if (x.ndim() < 1 || (dt != bfloat16 && dt != float16) ||
        w.dtype() != float32) {
      return nullptr;
    }
    const int64_t axis64 = x.shape(-1);
    if (axis64 < 1 ||
        axis64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      return nullptr;
    }
    const size_t axis = static_cast<size_t>(axis64);
    const size_t total = x.size();
    if (w.size() != axis || total == 0 || (total % axis) != 0) {
      return nullptr;
    }
    const size_t rows64 = total / axis;
    if (rows64 < 1 ||
        rows64 > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      return nullptr;
    }
    auto outs = array::make_arrays(
        {x.shape()},
        {dt},
        std::make_shared<FusedAddRmsNormCuda>(
            default_stream(Device::gpu),
            /*precise_f32=*/true,
            /*has_res=*/false,
            /*has_eps_arr=*/false,
            eps),
        {x, w});
    return reinterpret_cast<mlx_array*>(new array(std::move(outs[0])));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_rmsnorm_precise_f32_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_rmsnorm_precise_f32_cuda: unknown exception\n");
    return nullptr;
  }
#else
  (void)x_handle;
  (void)weight_f32_handle;
  (void)eps;
  return nullptr;
#endif
}
