#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_cuda_ops.h"

#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace mlx::core::sdpa_few {
namespace {

constexpr char sdpa_tc_few_source[] =
#include "cuda/sdpa_tc_few.cu.inc"
    ;

class SdpaTcFewCuda : public UnaryPrimitive {
public:
  SdpaTcFewCuda(Stream stream, float scale, int mask_mode)
      : UnaryPrimitive(stream), scale_(scale), mask_mode_(mask_mode) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("SdpaTcFewCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto ensure_kv_aligned = [&](const array& a) -> array {
      const bool ok = a.strides(3) == 1 && (a.strides(2) % 8 == 0) &&
          (a.strides(1) % 8 == 0) && (a.strides(0) % 8 == 0) &&
          (reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) % 16 == 0);
      if (ok) {
        return a;
      }
      auto copy = contiguous_copy_gpu(a, stream());
      encoder.add_temporary(copy);
      return copy;
    };

    array q = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array k = ensure_kv_aligned(inputs[1]);
    array v = ensure_kv_aligned(inputs[2]);

    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int Q_len = q.shape(2);
    const int D = q.shape(3);
    const int Hkv = k.shape(1);
    const int K_len = k.shape(2);
    const int gqa = Hq / Hkv;
    const int M = gqa * Q_len;
    const int num_warps = (M + 15) / 16;
    if ((D != 128 && D != 256) || Q_len < 1 || Q_len > 8 || K_len < 1 ||
        M < 1 || M > 64 || (D == 256 && M > 48) ||
        num_warps < 1 || num_warps > 4) {
      throw std::invalid_argument("SdpaTcFewCuda: unsupported replay shape");
    }
    const int total_tiles = (K_len + 31) / 32;
    const int max_splits = (Hkv >= 8) ? 12 : 24;
    const int num_splits = std::min(max_splits, std::max(1, total_tiles));

    array partial_o(Shape{B, Hkv, num_splits, M, D}, float32, nullptr, {});
    array partial_m(Shape{B, Hkv, num_splits, M}, float32, nullptr, {});
    array partial_l(Shape{B, Hkv, num_splits, M}, float32, nullptr, {});
    partial_o.set_data(cu::malloc_async(partial_o.nbytes(), encoder));
    partial_m.set_data(cu::malloc_async(partial_m.nbytes(), encoder));
    partial_l.set_data(cu::malloc_async(partial_l.nbytes(), encoder));
    encoder.add_temporary(partial_o);
    encoder.add_temporary(partial_m);
    encoder.add_temporary(partial_l);

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    CUdeviceptr mask_ptr = 0;
    if (mask_mode_ == 2) {
      array m = cu::ensure_row_contiguous_aligned<1>(inputs[3], stream(), encoder);
      encoder.set_input_array(m);
      mask_ptr = reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(m));
    }
    encoder.set_output_array(partial_o);
    encoder.set_output_array(partial_m);
    encoder.set_output_array(partial_l);

    // [D==256 ? 1 : 0][num_warps - 1][mask_mode]
    static constexpr const char* kSplitEntries[2][4][3] = {
        {{"mlx_node_sdpa_tc_split<128,1,0>", "mlx_node_sdpa_tc_split<128,1,1>", "mlx_node_sdpa_tc_split<128,1,2>"},
         {"mlx_node_sdpa_tc_split<128,2,0>", "mlx_node_sdpa_tc_split<128,2,1>", "mlx_node_sdpa_tc_split<128,2,2>"},
         {"mlx_node_sdpa_tc_split<128,3,0>", "mlx_node_sdpa_tc_split<128,3,1>", "mlx_node_sdpa_tc_split<128,3,2>"},
         {"mlx_node_sdpa_tc_split<128,4,0>", "mlx_node_sdpa_tc_split<128,4,1>", "mlx_node_sdpa_tc_split<128,4,2>"}},
        {{"mlx_node_sdpa_tc_split<256,1,0>", "mlx_node_sdpa_tc_split<256,1,1>", "mlx_node_sdpa_tc_split<256,1,2>"},
         {"mlx_node_sdpa_tc_split<256,2,0>", "mlx_node_sdpa_tc_split<256,2,1>", "mlx_node_sdpa_tc_split<256,2,2>"},
         {"mlx_node_sdpa_tc_split<256,3,0>", "mlx_node_sdpa_tc_split<256,3,1>", "mlx_node_sdpa_tc_split<256,3,2>"},
         {"mlx_node_sdpa_tc_split<256,4,0>", "mlx_node_sdpa_tc_split<256,4,1>", "mlx_node_sdpa_tc_split<256,4,2>"}},
    };
    const int d_idx = (D == 256) ? 1 : 0;
    auto split_kernel = cu::jit_kernel(
        encoder.device(), sdpa_tc_few_source,
        kSplitEntries[d_idx][num_warps - 1][mask_mode_]);
    cu::KernelArgs sargs;
    sargs.append(q);
    sargs.append(k);
    sargs.append(v);
    sargs.append<CUdeviceptr>(mask_ptr);
    sargs.append(partial_o);
    sargs.append(partial_m);
    sargs.append(partial_l);
    sargs.append<int32_t>(Hkv);
    sargs.append<int32_t>(gqa);
    sargs.append<int32_t>(Q_len);
    sargs.append<int32_t>(K_len);
    sargs.append<int64_t>(k.strides(0));
    sargs.append<int64_t>(k.strides(1));
    sargs.append<int64_t>(k.strides(2));
    sargs.append<int64_t>(v.strides(0));
    sargs.append<int64_t>(v.strides(1));
    sargs.append<int64_t>(v.strides(2));
    sargs.append<float>(scale_);
    sargs.append<int32_t>(num_splits);
    encoder.add_kernel_node_raw(
        split_kernel, dim3(num_splits, Hkv, B), dim3(32, num_warps, 1), {}, 0,
        sargs.args());

    encoder.set_input_array(partial_o);
    encoder.set_input_array(partial_m);
    encoder.set_input_array(partial_l);
    encoder.set_output_array(out);
    static constexpr const char* kReduceEntries[2] = {
        "mlx_node_sdpa_tc_reduce<128>",
        "mlx_node_sdpa_tc_reduce<256>",
    };
    auto reduce_kernel = cu::jit_kernel(
        encoder.device(), sdpa_tc_few_source, kReduceEntries[d_idx]);
    cu::KernelArgs rargs;
    rargs.append(partial_o);
    rargs.append(partial_m);
    rargs.append(partial_l);
    rargs.append(out);
    rargs.append<int32_t>(Hkv);
    rargs.append<int32_t>(M);
    rargs.append<int32_t>(num_splits);
    encoder.add_kernel_node_raw(
        reduce_kernel, dim3(M, Hkv, B), dim3(D, 1, 1), {}, 0, rargs.args());
  }

  DEFINE_NAME(SdpaTcFewCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const SdpaTcFewCuda*>(&other);
    return rhs && rhs->scale_ == scale_ && rhs->mask_mode_ == mask_mode_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

private:
  float scale_;
  int mask_mode_;
};

} // namespace
} // namespace mlx::core::sdpa_few

extern "C" mlx_array* mlx_sdpa_tc_few_masked_cuda(
    mlx_array* q_handle, mlx_array* k_handle, mlx_array* v_handle,
    mlx_array* mask_handle, float scale, bool is_causal) {
  try {
    using namespace mlx::core;
    if (!q_handle || !k_handle || !v_handle ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    std::vector<array> inputs;
    inputs.reserve(4);
    inputs.push_back(*reinterpret_cast<array*>(q_handle));
    inputs.push_back(*reinterpret_cast<array*>(k_handle));
    inputs.push_back(*reinterpret_cast<array*>(v_handle));
    const auto& q = *reinterpret_cast<array*>(q_handle);
    const auto& k = *reinterpret_cast<array*>(k_handle);
    const auto& v = *reinterpret_cast<array*>(v_handle);
    if (q.dtype() != bfloat16 || k.dtype() != bfloat16 ||
        v.dtype() != bfloat16 || q.ndim() != 4 || k.ndim() != 4 ||
        v.ndim() != 4) {
      return nullptr;
    }
    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int Q_len = q.shape(2);
    const int D = q.shape(3);
    const int Hkv = k.shape(1);
    const int K_len = k.shape(2);
    if (B <= 0 || Hkv <= 0 || Hq < Hkv || Hq % Hkv != 0 ||
        (D != 128 && D != 256) || Q_len < 1 || Q_len > 8 ||
        k.shape(0) != B || k.shape(3) != D || v.shape() != k.shape()) {
      return nullptr;
    }
    const int M = (Hq / Hkv) * Q_len;
    // D=256 uses static smem of (NUM_WARPS*16 + 32) * (D + 8) * 2 bytes;
    // NUM_WARPS=4 (M > 48) would exceed 48 KB static smem (50,688 B).
    if (M < 1 || M > 64 || (D == 256 && M > 48)) {
      return nullptr;
    }
    int mask_mode = 0;
    if (mask_handle) {
      const auto& mask = *reinterpret_cast<array*>(mask_handle);
      if (is_causal || K_len < 1 || mask.dtype() != bool_ || mask.ndim() != 4 ||
          mask.shape(0) != 1 || mask.shape(1) != 1 ||
          mask.shape(2) != Q_len || mask.shape(3) != K_len) {
        return nullptr;
      }
      inputs.push_back(mask);
      mask_mode = 2;
    } else if (is_causal) {
      if (K_len < Q_len) {
        return nullptr;
      }
      mask_mode = 0;
    } else {
      if (K_len < 1) {
        return nullptr;
      }
      mask_mode = 1;
    }
    auto result = array(
        q.shape(), bfloat16,
        std::make_shared<sdpa_few::SdpaTcFewCuda>(
            default_stream(Device::gpu), scale, mask_mode),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_sdpa_tc_few_masked_cuda error: " << e.what() << std::endl;
  }
  return nullptr;
}

extern "C" mlx_array* mlx_sdpa_tc_few_cuda(
    mlx_array* q_handle, mlx_array* k_handle, mlx_array* v_handle,
    float scale) {
  return mlx_sdpa_tc_few_masked_cuda(
      q_handle, k_handle, v_handle, nullptr, scale, true);
}

#else

extern "C" mlx_array* mlx_sdpa_tc_few_masked_cuda(
    mlx_array* q_handle, mlx_array* k_handle, mlx_array* v_handle,
    mlx_array* mask_handle, float scale, bool is_causal) {
  (void)q_handle;
  (void)k_handle;
  (void)v_handle;
  (void)mask_handle;
  (void)scale;
  (void)is_causal;
  return nullptr;
}

extern "C" mlx_array* mlx_sdpa_tc_few_cuda(
    mlx_array* q_handle, mlx_array* k_handle, mlx_array* v_handle,
    float scale) {
  (void)q_handle;
  (void)k_handle;
  (void)v_handle;
  (void)scale;
  return nullptr;
}

#endif
