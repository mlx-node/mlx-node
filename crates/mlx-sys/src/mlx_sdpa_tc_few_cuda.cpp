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

constexpr char sdpa_prefill_d256_source[] =
#include "cuda/sdpa_prefill_d256.cu.inc"
    ;

class SdpaTcFewCuda : public UnaryPrimitive {
public:
  SdpaTcFewCuda(Stream stream, float scale, int mask_mode, int cache_len = 0)
      : UnaryPrimitive(stream),
        scale_(scale),
        mask_mode_(mask_mode),
        cache_len_(cache_len) {}

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
    const int num_q_tiles = (M <= 96) ? 1 : 2;
    const int M_tile = (num_q_tiles == 1) ? M : (gqa * 8);
    const int num_warps = (M_tile + 15) / 16;
    if ((D != 128 && D != 256) || Q_len < 1 || Q_len > 16 || K_len < 1 ||
        M < 1 || (num_q_tiles == 2 && (Q_len <= 8 || M_tile > 96)) ||
        num_warps < 1 || num_warps > 6) {
      throw std::invalid_argument("SdpaTcFewCuda: unsupported replay shape");
    }
    const int total_tiles = (K_len + 31) / 32;
    const int max_splits = (Hkv >= 8) ? 6 : ((Hkv >= 4) ? 8 : ((num_q_tiles == 2) ? 12 : 16));
    const int num_splits = std::min(max_splits, std::max(1, total_tiles));

    array partial_o(Shape{B, Hkv, num_splits, M, D}, bfloat16, nullptr, {});
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
    } else if (mask_mode_ == 3) {
      array tm = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);
      encoder.set_input_array(tm);
      mask_ptr = reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(tm));
    }
    encoder.set_output_array(partial_o);
    encoder.set_output_array(partial_m);
    encoder.set_output_array(partial_l);

    // [D==256 ? 1 : 0][num_warps - 1][mask_mode]
    // Q_HI_MODE=1 (sQ_hi in SMEM) for D=256 && num_warps<=3; else Q_HI_MODE=0.
    static constexpr const char* kSplitEntries[2][6][4] = {
        {{"mlx_node_sdpa_tc_split<128,1,0,0>", "mlx_node_sdpa_tc_split<128,1,1,0>",
          "mlx_node_sdpa_tc_split<128,1,2,0>", "mlx_node_sdpa_tc_split<128,1,3,0>"},
         {"mlx_node_sdpa_tc_split<128,2,0,0>", "mlx_node_sdpa_tc_split<128,2,1,0>",
          "mlx_node_sdpa_tc_split<128,2,2,0>", "mlx_node_sdpa_tc_split<128,2,3,0>"},
         {"mlx_node_sdpa_tc_split<128,3,0,0>", "mlx_node_sdpa_tc_split<128,3,1,0>",
          "mlx_node_sdpa_tc_split<128,3,2,0>", "mlx_node_sdpa_tc_split<128,3,3,0>"},
         {"mlx_node_sdpa_tc_split<128,4,0,0>", "mlx_node_sdpa_tc_split<128,4,1,0>",
          "mlx_node_sdpa_tc_split<128,4,2,0>", "mlx_node_sdpa_tc_split<128,4,3,0>"},
         {"mlx_node_sdpa_tc_split<128,5,0,0>", "mlx_node_sdpa_tc_split<128,5,1,0>",
          "mlx_node_sdpa_tc_split<128,5,2,0>", "mlx_node_sdpa_tc_split<128,5,3,0>"},
         {"mlx_node_sdpa_tc_split<128,6,0,0>", "mlx_node_sdpa_tc_split<128,6,1,0>",
          "mlx_node_sdpa_tc_split<128,6,2,0>", "mlx_node_sdpa_tc_split<128,6,3,0>"}},
        {{"mlx_node_sdpa_tc_split<256,1,0,1>", "mlx_node_sdpa_tc_split<256,1,1,1>",
          "mlx_node_sdpa_tc_split<256,1,2,1>", "mlx_node_sdpa_tc_split<256,1,3,1>"},
         {"mlx_node_sdpa_tc_split<256,2,0,1>", "mlx_node_sdpa_tc_split<256,2,1,1>",
          "mlx_node_sdpa_tc_split<256,2,2,1>", "mlx_node_sdpa_tc_split<256,2,3,1>"},
         {"mlx_node_sdpa_tc_split<256,3,0,1>", "mlx_node_sdpa_tc_split<256,3,1,1>",
          "mlx_node_sdpa_tc_split<256,3,2,1>", "mlx_node_sdpa_tc_split<256,3,3,1>"},
         {"mlx_node_sdpa_tc_split<256,4,0,0>", "mlx_node_sdpa_tc_split<256,4,1,0>",
          "mlx_node_sdpa_tc_split<256,4,2,0>", "mlx_node_sdpa_tc_split<256,4,3,0>"},
         {"mlx_node_sdpa_tc_split<256,5,0,0>", "mlx_node_sdpa_tc_split<256,5,1,0>",
          "mlx_node_sdpa_tc_split<256,5,2,0>", "mlx_node_sdpa_tc_split<256,5,3,0>"},
         {"mlx_node_sdpa_tc_split<256,6,0,0>", "mlx_node_sdpa_tc_split<256,6,1,0>",
          "mlx_node_sdpa_tc_split<256,6,2,0>", "mlx_node_sdpa_tc_split<256,6,3,0>"}},
    };
    const int d_idx = (D == 256) ? 1 : 0;
    auto split_kernel = cu::jit_kernel(
        encoder.device(), sdpa_tc_few_source,
        kSplitEntries[d_idx][num_warps - 1][mask_mode_]);
    constexpr float kLog2E = 1.4426950408889634f;
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
    sargs.append<int32_t>(cache_len_);
    sargs.append<int64_t>(k.strides(0));
    sargs.append<int64_t>(k.strides(1));
    sargs.append<int64_t>(k.strides(2));
    sargs.append<int64_t>(v.strides(0));
    sargs.append<int64_t>(v.strides(1));
    sargs.append<int64_t>(v.strides(2));
    sargs.append<float>(scale_ * kLog2E);
    sargs.append<int32_t>(num_splits);
    sargs.append<int32_t>(num_q_tiles);
    encoder.add_kernel_node_raw(
        split_kernel, dim3(num_splits, Hkv, B * num_q_tiles),
        dim3(32, num_warps, 1), {}, 0, sargs.args());

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
        reduce_kernel, dim3(M, Hkv, B), dim3(32, 1, 1), {}, 0, rargs.args());
  }

  DEFINE_NAME(SdpaTcFewCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const SdpaTcFewCuda*>(&other);
    return rhs && rhs->scale_ == scale_ && rhs->mask_mode_ == mask_mode_ &&
        rhs->cache_len_ == cache_len_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

private:
  float scale_;
  int mask_mode_;
  int cache_len_;
};

class SdpaPrefillD256Cuda : public UnaryPrimitive {
public:
  SdpaPrefillD256Cuda(Stream stream, float scale)
      : UnaryPrimitive(stream), scale_(scale) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("SdpaPrefillD256Cuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto ensure_strided_aligned = [&](const array& a) -> array {
      const bool ok = a.strides(3) == 1 &&
          a.strides(2) >= 0 && (a.strides(2) % 8 == 0) &&
          a.strides(1) >= 0 && (a.strides(1) % 8 == 0) &&
          a.strides(0) >= 0 && (a.strides(0) % 8 == 0) &&
          (reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) % 16 == 0);
      if (ok) {
        return a;
      }
      auto copy = contiguous_copy_gpu(a, stream());
      encoder.add_temporary(copy);
      return copy;
    };

    array q = ensure_strided_aligned(inputs[0]);
    array k = ensure_strided_aligned(inputs[1]);
    array v = ensure_strided_aligned(inputs[2]);

    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int Q_len = q.shape(2);
    const int D = q.shape(3);
    const int Hkv = k.shape(1);
    const int K_len = k.shape(2);
    if (B <= 0 || Hkv <= 0 || Hq < Hkv || Hq % Hkv != 0 || D != 256 ||
        Q_len < 9 || K_len < Q_len || k.shape(0) != B || k.shape(3) != D ||
        v.shape() != k.shape()) {
      throw std::invalid_argument("SdpaPrefillD256Cuda: unsupported replay shape");
    }
    const int gqa = Hq / Hkv;

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_output_array(out);

    auto kernel = cu::jit_kernel(
        encoder.device(), sdpa_prefill_d256_source, "mlx_node_sdpa_prefill_d256");
    constexpr float kLog2E = 1.4426950408889634f;
    cu::KernelArgs args;
    args.append(q);
    args.append(k);
    args.append(v);
    args.append(out);
    args.append<int32_t>(Hkv);
    args.append<int32_t>(gqa);
    args.append<int32_t>(Q_len);
    args.append<int32_t>(K_len);
    args.append<int64_t>(q.strides(0));
    args.append<int64_t>(q.strides(1));
    args.append<int64_t>(q.strides(2));
    args.append<int64_t>(k.strides(0));
    args.append<int64_t>(k.strides(1));
    args.append<int64_t>(k.strides(2));
    args.append<int64_t>(v.strides(0));
    args.append<int64_t>(v.strides(1));
    args.append<int64_t>(v.strides(2));
    args.append<float>(scale_ * kLog2E);

    const uint32_t num_q_blocks = static_cast<uint32_t>((Q_len + 63) / 64);
    encoder.add_kernel_node_raw(
        kernel, dim3(num_q_blocks * static_cast<uint32_t>(gqa), Hkv, B),
        dim3(128, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(SdpaPrefillD256Cuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const SdpaPrefillD256Cuda*>(&other);
    return rhs && rhs->scale_ == scale_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

private:
  float scale_;
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
        (D != 128 && D != 256) || Q_len < 1 || Q_len > 16 ||
        k.shape(0) != B || k.shape(3) != D || v.shape() != k.shape()) {
      return nullptr;
    }
    const int gqa = Hq / Hkv;
    const int M = gqa * Q_len;
    const bool ok_single = (M >= 1 && M <= 96);
    const bool ok_two_tile = (Q_len > 8 && gqa * 8 <= 96);
    if (!ok_single && !ok_two_tile) {
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

extern "C" mlx_array* mlx_sdpa_tc_tree_few_cuda(
    mlx_array* q_handle, mlx_array* k_handle, mlx_array* v_handle,
    mlx_array* tree_mask_handle, int32_t cache_len, float scale) {
  try {
    using namespace mlx::core;
    if (!q_handle || !k_handle || !v_handle || !tree_mask_handle ||
        cache_len < 0 || default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& q = *reinterpret_cast<array*>(q_handle);
    const auto& k = *reinterpret_cast<array*>(k_handle);
    const auto& v = *reinterpret_cast<array*>(v_handle);
    const auto& tm = *reinterpret_cast<array*>(tree_mask_handle);
    if (q.dtype() != bfloat16 || k.dtype() != bfloat16 ||
        v.dtype() != bfloat16 || tm.dtype() != uint32 ||
        q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || tm.ndim() != 1) {
      return nullptr;
    }
    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int Q_len = q.shape(2);
    const int D = q.shape(3);
    const int Hkv = k.shape(1);
    const int K_len = k.shape(2);
    if (B <= 0 || Hkv <= 0 || Hq < Hkv || Hq % Hkv != 0 ||
        (D != 128 && D != 256) || Q_len < 1 || Q_len > 16 || K_len < 1 ||
        k.shape(0) != B || k.shape(3) != D || v.shape() != k.shape() ||
        tm.shape(0) != Q_len) {
      return nullptr;
    }
    const int gqa = Hq / Hkv;
    const int M = gqa * Q_len;
    const bool ok_single = (M >= 1 && M <= 96);
    const bool ok_two_tile = (Q_len > 8 && gqa * 8 <= 96);
    if (!ok_single && !ok_two_tile) {
      return nullptr;
    }
    std::vector<array> inputs{q, k, v, tm};
    auto result = array(
        q.shape(), bfloat16,
        std::make_shared<sdpa_few::SdpaTcFewCuda>(
            default_stream(Device::gpu), scale, 3, cache_len),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_sdpa_tc_tree_few_cuda error: " << e.what() << std::endl;
  }
  return nullptr;
}

extern "C" mlx_array* mlx_sdpa_prefill_d256_cuda(
    mlx_array* q_handle, mlx_array* k_handle, mlx_array* v_handle,
    float scale) {
  try {
    using namespace mlx::core;
    if (!q_handle || !k_handle || !v_handle ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
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
    if (B <= 0 || Hkv <= 0 || Hq < Hkv || Hq % Hkv != 0 || D != 256 ||
        Q_len < 9 || K_len < Q_len || k.shape(0) != B || k.shape(3) != D ||
        v.shape() != k.shape()) {
      return nullptr;
    }
    std::vector<array> inputs = {q, k, v};
    auto result = array(
        q.shape(), bfloat16,
        std::make_shared<sdpa_few::SdpaPrefillD256Cuda>(
            default_stream(Device::gpu), scale),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_sdpa_prefill_d256_cuda error: " << e.what() << std::endl;
  }
  return nullptr;
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

extern "C" mlx_array* mlx_sdpa_tc_tree_few_cuda(
    mlx_array* q_handle, mlx_array* k_handle, mlx_array* v_handle,
    mlx_array* tree_mask_handle, int32_t cache_len, float scale) {
  (void)q_handle;
  (void)k_handle;
  (void)v_handle;
  (void)tree_mask_handle;
  (void)cache_len;
  (void)scale;
  return nullptr;
}

extern "C" mlx_array* mlx_sdpa_prefill_d256_cuda(
    mlx_array* q_handle, mlx_array* k_handle, mlx_array* v_handle,
    float scale) {
  (void)q_handle;
  (void)k_handle;
  (void)v_handle;
  (void)scale;
  return nullptr;
}

#endif
