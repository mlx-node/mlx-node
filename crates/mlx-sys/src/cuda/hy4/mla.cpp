#include "cuda/hy4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx/backend/common/utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mlx::core::hy4_ops {
namespace {

using namespace mlx_node::hy4_cuda;

constexpr char hy4_mla_few_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_mla_few.cu.inc"
    ;

constexpr char hy4_mla_prefill_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_mla_prefill.cu.inc"
    ;

// ============================================================================
// 1. GPU Ring Write (in-place aliased buffer write, CUDA Graph safe)
// ============================================================================
class Hy4RingWriteCuda final : public UnaryPrimitive {
 public:
  Hy4RingWriteCuda(
      Stream stream,
      int64_t start_pos,
      int32_t ring_cap,
      bool force_in_place)
      : UnaryPrimitive(stream),
        start_pos_(start_pos),
        ring_cap_(ring_cap),
        force_in_place_(force_in_place) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4RingWriteCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    const array& buf = inputs[0];
    if (!buf.flags().row_contiguous) {
      throw std::invalid_argument("Hy4RingWriteCuda: buf must be row-contiguous");
    }
    array rows = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    if (rows.buffer().ptr() == buf.buffer().ptr()) {
      rows = contiguous_copy_gpu(rows, stream());
      encoder.add_temporary(rows);
    }

    const int M = static_cast<int>(rows.shape(0));
    const int cols = static_cast<int>(rows.shape(1));
    const int row_bytes = cols * static_cast<int>(buf.itemsize());

    const bool buf_dense = buf.flags().row_contiguous &&
                           buf.size() == buf.data_size() && buf.offset() == 0;
    // Preserve MLX copy-on-write semantics by default so generic `ops::ring_write`
    // never mutates a shared/aliased buffer. Explicit CUDA fast-path callers that
    // guarantee exclusive ownership or bounded rollback slack pass `force_in_place=true`.
    if (buf_dense && (force_in_place_ || mlx::core::is_donatable(buf, out))) {
      out.copy_shared_buffer(buf);
    } else {
      copy_gpu(buf, out, CopyType::General, stream());
    }
    encoder.set_input_array(rows);
    encoder.set_output_array(out);

    cu::KernelArgs args;
    args.append(out);
    args.append(rows);
    args.append<int32_t>(M);
    args.append<int32_t>(row_bytes);
    args.append<int64_t>(start_pos_);
    args.append<int32_t>(ring_cap_);

    auto fn = cu::jit_kernel(
        encoder.device(), hy4_mla_few_source, "hy4_ring_write_cuda");
    encoder.add_kernel_node_raw(
        fn, dim3(M, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Hy4RingWriteCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4RingWriteCuda*>(&other);
    return rhs && rhs->start_pos_ == start_pos_ && rhs->ring_cap_ == ring_cap_ &&
        rhs->force_in_place_ == force_in_place_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int64_t start_pos_;
  int32_t ring_cap_;
  bool force_in_place_;
};

// ============================================================================
// 2. MLA Prep + In-Place KV Store
// ============================================================================
class Hy4MlaPrepKvStoreCuda final : public Primitive {
 public:
  Hy4MlaPrepKvStoreCuda(
      Stream stream,
      bool has_q,
      int64_t slot_start,
      int64_t rope_start,
      int32_t ring_cap,
      float norm_eps)
      : Primitive(stream),
        has_q_(has_q),
        slot_start_(slot_start),
        rope_start_(rope_start),
        ring_cap_(ring_cap),
        norm_eps_(norm_eps) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Hy4MlaPrepKvStoreCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    size_t idx = 0;
    auto prepare_strided_2d_aligned16 = [&](const array& in, int64_t& row_stride) -> array {
      const bool ptr_aligned16 =
          (reinterpret_cast<uintptr_t>(gpu_ptr<void>(in)) & 15) == 0;
      if (in.ndim() == 2 && in.strides(1) == 1 && ptr_aligned16) {
        if (in.shape(0) == 1) {
          row_stride = in.shape(1);
          return in;
        }
        if (in.strides(0) >= 0 && (in.strides(0) % 8) == 0) {
          row_stride = in.strides(0);
          return in;
        }
      }
      array c = cu::ensure_row_contiguous_aligned<16>(in, stream(), encoder);
      row_stride = c.shape(1);
      return c;
    };
    int64_t q_a_stride = 0;
    int64_t kv_a_stride = 0;
    std::optional<array> q_a;
    std::optional<array> q_a_norm;
    if (has_q_) {
      q_a.emplace(prepare_strided_2d_aligned16(inputs[idx++], q_a_stride));
    }
    array kv_a = prepare_strided_2d_aligned16(inputs[idx++], kv_a_stride);
    if (has_q_) {
      q_a_norm.emplace(cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder));
    }
    array kv_a_norm = cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    array inv_freq = cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    const array& kv_buf = inputs[idx++];

    if (!kv_buf.flags().row_contiguous) {
      throw std::invalid_argument("Hy4MlaPrepKvStoreCuda: kv_buf must be row-contiguous");
    }
    const void* kv_ptr = kv_buf.buffer().ptr();
    auto check_no_alias = [&](const array& a) {
      if (a.buffer().ptr() == kv_ptr) {
        throw std::invalid_argument("Hy4MlaPrepKvStoreCuda: kv_buf aliases read input");
      }
    };
    if (q_a) check_no_alias(*q_a);
    check_no_alias(kv_a);
    if (q_a_norm) check_no_alias(*q_a_norm);
    check_no_alias(kv_a_norm);
    check_no_alias(inv_freq);

    const int M = static_cast<int>(kv_a.shape(0));
    const int q_lora = has_q_ ? static_cast<int>(q_a->shape(1)) : 0;
    const int kv_lora = static_cast<int>(kv_a_norm.shape(0));
    const int half_rope = static_cast<int>(inv_freq.shape(0));

    array& out_kv = has_q_ ? outputs[1] : outputs[0];
    out_kv.copy_shared_buffer(kv_buf);

    if (q_a) encoder.set_input_array(*q_a);
    encoder.set_input_array(kv_a);
    if (q_a_norm) encoder.set_input_array(*q_a_norm);
    encoder.set_input_array(kv_a_norm);
    encoder.set_input_array(inv_freq);

    cu::KernelArgs args;
    if (has_q_) {
      array& out_qr = outputs[0];
      out_qr.set_data(cu::malloc_async(out_qr.nbytes(), encoder));
      encoder.set_output_array(out_qr);
      encoder.set_output_array(out_kv);
      args.append(*q_a);
      args.append(kv_a);
      args.append(*q_a_norm);
      args.append(kv_a_norm);
      args.append(inv_freq);
      args.append(out_qr);
      args.append(out_kv);
    } else {
      encoder.set_output_array(out_kv);
      args.append<CUdeviceptr>(0);
      args.append(kv_a);
      args.append<CUdeviceptr>(0);
      args.append(kv_a_norm);
      args.append(inv_freq);
      args.append<CUdeviceptr>(0);
      args.append(out_kv);
    }

    args.append<int32_t>(M);
    args.append<int32_t>(q_lora);
    args.append<int32_t>(kv_lora);
    args.append<int32_t>(half_rope);
    args.append<int64_t>(q_a_stride);
    args.append<int64_t>(kv_a_stride);
    args.append<int64_t>(slot_start_);
    args.append<int64_t>(rope_start_);
    args.append<int32_t>(ring_cap_);
    args.append<float>(norm_eps_);

    auto fn = cu::jit_kernel(
        encoder.device(), hy4_mla_few_source,
        has_q_ ? "hy4_mla_prep_kv_store_cuda<true>"
               : "hy4_mla_prep_kv_store_cuda<false>");
    encoder.add_kernel_node_raw(
        fn, dim3(M, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Hy4MlaPrepKvStoreCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4MlaPrepKvStoreCuda*>(&other);
    return rhs && rhs->has_q_ == has_q_ && rhs->slot_start_ == slot_start_ &&
        rhs->rope_start_ == rope_start_ && rhs->ring_cap_ == ring_cap_ &&
        rhs->norm_eps_ == norm_eps_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    if (has_q_) {
      return {inputs[0].shape(), inputs[5].shape()};
    }
    return {inputs[3].shape()};
  }

 private:
  bool has_q_;
  int64_t slot_start_;
  int64_t rope_start_;
  int32_t ring_cap_;
  float norm_eps_;
};

// ============================================================================
// 3. RoPE Q + Concat
// ============================================================================
class Hy4RopeNormQCuda final : public UnaryPrimitive {
 public:
  Hy4RopeNormQCuda(Stream stream, int64_t rope_start, int32_t nope_dim)
      : UnaryPrimitive(stream), rope_start_(rope_start), nope_dim_(nope_dim) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4RopeNormQCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array q_abs = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array q_raw = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array inv_freq = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);

    const int M = static_cast<int>(q_abs.shape(0));
    const int H = static_cast<int>(q_abs.shape(1));
    const int kv_lora = static_cast<int>(q_abs.shape(2));
    const int half_rope = static_cast<int>(inv_freq.shape(0));

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(q_abs);
    encoder.set_input_array(q_raw);
    encoder.set_input_array(inv_freq);
    encoder.set_output_array(out);

    cu::KernelArgs args;
    args.append(q_abs);
    args.append(q_raw);
    args.append(inv_freq);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(H);
    args.append<int32_t>(kv_lora);
    args.append<int32_t>(nope_dim_);
    args.append<int32_t>(half_rope);
    args.append<int64_t>(rope_start_);

    auto fn = cu::jit_kernel(
        encoder.device(), hy4_mla_few_source, "hy4_rope_norm_q_cuda");
    encoder.add_kernel_node_raw(
        fn, dim3(H, M, 1), dim3(128, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Hy4RopeNormQCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4RopeNormQCuda*>(&other);
    return rhs && rhs->rope_start_ == rope_start_ && rhs->nope_dim_ == nope_dim_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int M = inputs[0].shape(0);
    const int H = inputs[0].shape(1);
    const int kv_lora = inputs[0].shape(2);
    const int rope = inputs[2].shape(0) * 2;
    return {Shape{M, H, kv_lora + rope}};
  }

 private:
  int64_t rope_start_;
  int32_t nope_dim_;
};

// ============================================================================
// 4. Absorbed MLA Sink Softmax SDPA (Decode Split-K + Prefill TC / Generic)
// ============================================================================
class Hy4MlaSdpaCuda final : public UnaryPrimitive {
 public:
  Hy4MlaSdpaCuda(
      Stream stream,
      bool has_idx,
      int64_t slot_start,
      int32_t total_len,
      int32_t ring_cap,
      int32_t window,
      float scale,
      int32_t kv_lora)
      : UnaryPrimitive(stream),
        has_idx_(has_idx),
        slot_start_(slot_start),
        total_len_(total_len),
        ring_cap_(ring_cap),
        window_(window),
        scale_(scale),
        kv_lora_(kv_lora) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4MlaSdpaCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array q = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array kv_buf = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array sinks = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);
    std::optional<array> idx_arr;
    if (has_idx_) {
      idx_arr.emplace(
          cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder));
    }

    const int M = static_cast<int>(q.shape(0));
    const int H = static_cast<int>(q.shape(1));
    const int kv_row = static_cast<int>(q.shape(2));
    const int rope = kv_row - kv_lora_;
    const int cap = static_cast<int>(kv_buf.shape(0));
    const int kmax = has_idx_ ? static_cast<int>(idx_arr->shape(1)) : 0;

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    const bool is_512_64 = (kv_lora_ == 512 && rope == 64);

    if (M <= 16) {
      // Decode / few-token Split-K path
      const int max_keys = has_idx_
          ? kmax
          : ((window_ > 0) ? std::min(window_, total_len_) : total_len_);
      const int sm_count = device_sm_count(encoder.device().cuda_device());
      const int head_blocks = is_512_64 ? ((H + 15) / 16) : H;
      const int base_ctas = std::max(1, M * head_blocks);
      int num_splits = std::max(1, (sm_count * 2 + base_ctas - 1) / base_ctas);
      const int max_useful_splits = std::max(1, (max_keys + 31) / 32);
      num_splits = std::min(num_splits, max_useful_splits);
      const int split_cap = (max_keys <= 512) ? 8 : 24;
      num_splits = std::min(num_splits, split_cap);
      if (num_splits < 1) num_splits = 1;

      array partial_o = alloc_temp({M, H, num_splits, kv_lora_}, float32, encoder);
      array partial_m = alloc_temp({M, H, num_splits}, float32, encoder);
      array partial_l = alloc_temp({M, H, num_splits}, float32, encoder);

      encoder.set_input_array(q);
      encoder.set_input_array(kv_buf);
      if (idx_arr) encoder.set_input_array(*idx_arr);
      encoder.set_output_array(partial_o);
      encoder.set_output_array(partial_m);
      encoder.set_output_array(partial_l);

      if (is_512_64) {
        auto fn_sp = cu::jit_kernel(
            encoder.device(), hy4_mla_few_source,
            has_idx_ ? "hy4_mla_few_split_512_64<true>"
                     : "hy4_mla_few_split_512_64<false>");
        cu::KernelArgs sargs;
        sargs.append(q);
        sargs.append(kv_buf);
        if (idx_arr) sargs.append(*idx_arr);
        else sargs.append<CUdeviceptr>(0);
        sargs.append(partial_o);
        sargs.append(partial_m);
        sargs.append(partial_l);
        sargs.append<int32_t>(M);
        sargs.append<int32_t>(H);
        sargs.append<int32_t>(cap);
        sargs.append<int32_t>(kmax);
        sargs.append<int64_t>(slot_start_);
        sargs.append<int32_t>(total_len_);
        sargs.append<int32_t>(ring_cap_);
        sargs.append<int32_t>(window_);
        sargs.append<float>(scale_);
        sargs.append<int32_t>(num_splits);
        encoder.add_kernel_node_raw(
            fn_sp, dim3(num_splits, (H + 15) / 16, M), dim3(32, 8, 1), {}, 0,
            sargs.args());
      } else {
        auto fn_sp = cu::jit_kernel(
            encoder.device(), hy4_mla_few_source,
            has_idx_ ? "hy4_mla_few_split_generic<true>"
                     : "hy4_mla_few_split_generic<false>");
        const int smem_bytes = kv_lora_ * static_cast<int>(sizeof(float));
        ensure_dynamic_smem(fn_sp, smem_bytes);
        cu::KernelArgs sargs;
        sargs.append(q);
        sargs.append(kv_buf);
        if (idx_arr) sargs.append(*idx_arr);
        else sargs.append<CUdeviceptr>(0);
        sargs.append(partial_o);
        sargs.append(partial_m);
        sargs.append(partial_l);
        sargs.append<int32_t>(M);
        sargs.append<int32_t>(H);
        sargs.append<int32_t>(kv_lora_);
        sargs.append<int32_t>(kv_row);
        sargs.append<int32_t>(cap);
        sargs.append<int32_t>(kmax);
        sargs.append<int64_t>(slot_start_);
        sargs.append<int32_t>(total_len_);
        sargs.append<int32_t>(ring_cap_);
        sargs.append<int32_t>(window_);
        sargs.append<float>(scale_);
        sargs.append<int32_t>(num_splits);
        encoder.add_kernel_node_raw(
            fn_sp, dim3(num_splits, H, M), dim3(32, 1, 1), {}, smem_bytes,
            sargs.args());
      }

      encoder.set_input_array(partial_o);
      encoder.set_input_array(partial_m);
      encoder.set_input_array(partial_l);
      encoder.set_input_array(sinks);
      encoder.set_output_array(out);
      auto fn_red = cu::jit_kernel(
          encoder.device(), hy4_mla_few_source, "hy4_mla_few_reduce");
      cu::KernelArgs rargs;
      rargs.append(partial_o);
      rargs.append(partial_m);
      rargs.append(partial_l);
      rargs.append(sinks);
      rargs.append(out);
      rargs.append<int32_t>(M);
      rargs.append<int32_t>(H);
      rargs.append<int32_t>(kv_lora_);
      rargs.append<int32_t>(num_splits);
      encoder.add_kernel_node_raw(
          fn_red, dim3(H, M, 1), dim3(256, 1, 1), {}, 0, rargs.args());
      return;
    }

    // Prefill path (M >= 17)
    encoder.set_input_array(q);
    encoder.set_input_array(kv_buf);
    encoder.set_input_array(sinks);
    if (idx_arr) encoder.set_input_array(*idx_arr);
    encoder.set_output_array(out);

    if (is_512_64) {
      if (has_idx_) {
        auto fn_pre = cu::jit_kernel(
            encoder.device(), hy4_mla_prefill_source,
            "hy4_mla_prefill_tc_512_64<true>");
        cu::KernelArgs args;
        args.append(q);
        args.append(kv_buf);
        args.append(sinks);
        args.append(*idx_arr);
        args.append(out);
        args.append<int32_t>(M);
        args.append<int32_t>(H);
        args.append<int32_t>(cap);
        args.append<int32_t>(kmax);
        args.append<int64_t>(slot_start_);
        args.append<int32_t>(total_len_);
        args.append<int32_t>(ring_cap_);
        args.append<int32_t>(window_);
        args.append<float>(scale_);
        encoder.add_kernel_node_raw(
            fn_pre, dim3((H + 15) / 16, M, 1), dim3(32, 8, 1), {}, 0, args.args());
      } else {
        auto fn_pre = cu::jit_kernel(
            encoder.device(), hy4_mla_prefill_source,
            "hy4_mla_prefill_swa_q16_512_64");
        cu::KernelArgs args;
        args.append(q);
        args.append(kv_buf);
        args.append(sinks);
        args.append(out);
        args.append<int32_t>(M);
        args.append<int32_t>(H);
        args.append<int32_t>(cap);
        args.append<int64_t>(slot_start_);
        args.append<int32_t>(total_len_);
        args.append<int32_t>(ring_cap_);
        args.append<int32_t>(window_);
        args.append<float>(scale_);
        encoder.add_kernel_node_raw(
            fn_pre, dim3(H, (M + 15) / 16, 1), dim3(32, 8, 1), {}, 0, args.args());
      }
    } else {
      auto fn_pre = cu::jit_kernel(
          encoder.device(), hy4_mla_prefill_source,
          has_idx_ ? "hy4_mla_prefill_generic<true>"
                   : "hy4_mla_prefill_generic<false>");
      const int smem_bytes = kv_lora_ * static_cast<int>(sizeof(float));
      ensure_dynamic_smem(fn_pre, smem_bytes);
      cu::KernelArgs args;
      args.append(q);
      args.append(kv_buf);
      args.append(sinks);
      if (idx_arr) args.append(*idx_arr);
      else args.append<CUdeviceptr>(0);
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(H);
      args.append<int32_t>(kv_lora_);
      args.append<int32_t>(kv_row);
      args.append<int32_t>(cap);
      args.append<int32_t>(kmax);
      args.append<int64_t>(slot_start_);
      args.append<int32_t>(total_len_);
      args.append<int32_t>(ring_cap_);
      args.append<int32_t>(window_);
      args.append<float>(scale_);
      encoder.add_kernel_node_raw(
          fn_pre, dim3(H, M, 1), dim3(32, 1, 1), {}, smem_bytes, args.args());
    }
  }

  DEFINE_NAME(Hy4MlaSdpaCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4MlaSdpaCuda*>(&other);
    return rhs && rhs->has_idx_ == has_idx_ && rhs->slot_start_ == slot_start_ &&
        rhs->total_len_ == total_len_ && rhs->ring_cap_ == ring_cap_ &&
        rhs->window_ == window_ && rhs->scale_ == scale_ &&
        rhs->kv_lora_ == kv_lora_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{inputs[0].shape(0), inputs[0].shape(1), kv_lora_}};
  }

 private:
  bool has_idx_;
  int64_t slot_start_;
  int32_t total_len_;
  int32_t ring_cap_;
  int32_t window_;
  float scale_;
  int32_t kv_lora_;
};

}  // namespace
}  // namespace mlx::core::hy4_ops

extern "C" mlx_array* mlx_hy4_ring_write_cuda(
    mlx_array* buf_h,
    mlx_array* rows_h,
    int64_t start_pos,
    int32_t ring_cap,
    bool force_in_place) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!buf_h || !rows_h || start_pos < 0 ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& buf = *reinterpret_cast<array*>(buf_h);
    const auto& rows = *reinterpret_cast<array*>(rows_h);
    if ((buf.dtype() != float16 && buf.dtype() != float32 && buf.dtype() != bfloat16) ||
        rows.dtype() != buf.dtype() || buf.ndim() != 2 || rows.ndim() != 2 ||
        !buf.flags().row_contiguous) {
      return nullptr;
    }
    const int64_t cap = buf.shape(0);
    const int64_t cols = buf.shape(1);
    const int64_t M = rows.shape(0);
    if (M < 1 || M > 65535 || cols < 1 || rows.shape(1) != cols) {
      return nullptr;
    }
    if (ring_cap > 0) {
      if (ring_cap > cap) return nullptr;
    } else {
      if (start_pos + M > cap) return nullptr;
    }
    auto res = array(
        buf.shape(), buf.dtype(),
        std::make_shared<Hy4RingWriteCuda>(
            default_stream(Device::gpu), start_pos, ring_cap, force_in_place),
        {buf, rows});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_ring_write_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" bool mlx_hy4_mla_prep_kv_store_cuda(
    mlx_array* q_a_opt_h,
    mlx_array* kv_a_h,
    mlx_array* q_a_norm_opt_h,
    mlx_array* kv_a_norm_h,
    mlx_array* inv_freq_h,
    mlx_array* kv_buf_h,
    int64_t slot_start,
    int64_t rope_start,
    int32_t ring_cap,
    float norm_eps,
    mlx_array** out_qr,
    mlx_array** out_kv_buf) {
  if (out_qr) *out_qr = nullptr;
  if (out_kv_buf) *out_kv_buf = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    const bool has_q = (q_a_opt_h != nullptr);
    if (!kv_a_h || !kv_a_norm_h || !inv_freq_h || !kv_buf_h || !out_kv_buf ||
        (has_q && (!q_a_norm_opt_h || !out_qr)) ||
        (!has_q && q_a_norm_opt_h != nullptr) ||
        slot_start < 0 || rope_start < 0 ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& kv_a = *reinterpret_cast<array*>(kv_a_h);
    const auto& kv_a_norm = *reinterpret_cast<array*>(kv_a_norm_h);
    const auto& inv_freq = *reinterpret_cast<array*>(inv_freq_h);
    const auto& kv_buf = *reinterpret_cast<array*>(kv_buf_h);
    if (kv_a.dtype() != bfloat16 || kv_a_norm.dtype() != float32 ||
        inv_freq.dtype() != float32 || kv_buf.dtype() != float16 ||
        kv_a.ndim() != 2 || kv_a_norm.ndim() != 1 || inv_freq.ndim() != 1 ||
        kv_buf.ndim() != 2 || !kv_buf.flags().row_contiguous) {
      return false;
    }
    const int64_t M = kv_a.shape(0);
    const int64_t kv_row = kv_a.shape(1);
    const int64_t kv_lora = kv_a_norm.shape(0);
    const int64_t half_rope = inv_freq.shape(0);
    const int64_t cap = kv_buf.shape(0);
    if (M < 1 || M > 65535 || kv_lora < 1 || half_rope < 1 ||
        kv_row != kv_lora + half_rope * 2 || kv_buf.shape(1) != kv_row) {
      return false;
    }
    if (ring_cap > 0) {
      if (ring_cap > cap) return false;
    } else {
      if (slot_start + M > cap) return false;
    }

    std::vector<array> inputs;
    inputs.reserve(6);
    if (has_q) {
      const auto& q_a = *reinterpret_cast<array*>(q_a_opt_h);
      const auto& q_a_norm = *reinterpret_cast<array*>(q_a_norm_opt_h);
      if (q_a.dtype() != bfloat16 || q_a_norm.dtype() != float32 ||
          q_a.ndim() != 2 || q_a_norm.ndim() != 1 ||
          q_a.shape(0) != M || q_a.shape(1) < 1 ||
          q_a_norm.shape(0) != q_a.shape(1)) {
        return false;
      }
      inputs.push_back(q_a);
      inputs.push_back(kv_a);
      inputs.push_back(q_a_norm);
      inputs.push_back(kv_a_norm);
      inputs.push_back(inv_freq);
      inputs.push_back(kv_buf);
      auto results = array::make_arrays(
          {q_a.shape(), kv_buf.shape()}, {bfloat16, float16},
          std::make_shared<Hy4MlaPrepKvStoreCuda>(
              default_stream(Device::gpu), true,
              slot_start, rope_start, ring_cap, norm_eps),
          std::move(inputs));
      *out_qr = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      *out_kv_buf = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    } else {
      inputs.push_back(kv_a);
      inputs.push_back(kv_a_norm);
      inputs.push_back(inv_freq);
      inputs.push_back(kv_buf);
      auto res = array(
          kv_buf.shape(), float16,
          std::make_shared<Hy4MlaPrepKvStoreCuda>(
              default_stream(Device::gpu), false,
              slot_start, rope_start, ring_cap, norm_eps),
          std::move(inputs));
      *out_kv_buf = reinterpret_cast<mlx_array*>(new array(std::move(res)));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_mla_prep_kv_store_cuda: %s\n", e.what());
    return false;
  }
}

extern "C" mlx_array* mlx_hy4_rope_norm_q_cuda(
    mlx_array* q_abs_h,
    mlx_array* q_raw_h,
    mlx_array* inv_freq_h,
    int64_t rope_start,
    int32_t nope_dim) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!q_abs_h || !q_raw_h || !inv_freq_h || rope_start < 0 || nope_dim < 0 ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& q_abs = *reinterpret_cast<array*>(q_abs_h);
    const auto& q_raw = *reinterpret_cast<array*>(q_raw_h);
    const auto& inv_freq = *reinterpret_cast<array*>(inv_freq_h);
    if (q_abs.dtype() != bfloat16 || q_raw.dtype() != bfloat16 ||
        inv_freq.dtype() != float32 || q_abs.ndim() != 3 || inv_freq.ndim() != 1) {
      return nullptr;
    }
    const int64_t M = q_abs.shape(0);
    const int64_t H = q_abs.shape(1);
    const int64_t kv_lora = q_abs.shape(2);
    const int64_t half_rope = inv_freq.shape(0);
    const int64_t rope = half_rope * 2;
    const int64_t q_head = static_cast<int64_t>(nope_dim) + rope;
    if (M < 1 || M > 65535 || H < 1 || H > 65535 || kv_lora < 1 || half_rope < 1) {
      return nullptr;
    }
    const bool qraw_ok =
        (q_raw.ndim() == 3 && q_raw.shape(0) == M && q_raw.shape(1) == H &&
         q_raw.shape(2) == q_head) ||
        (q_raw.ndim() == 2 && q_raw.shape(0) == M && q_raw.shape(1) == H * q_head);
    if (!qraw_ok) {
      return nullptr;
    }
    auto res = array(
        Shape{static_cast<int>(M), static_cast<int>(H), static_cast<int>(kv_lora + rope)},
        bfloat16,
        std::make_shared<Hy4RopeNormQCuda>(
            default_stream(Device::gpu), rope_start, nope_dim),
        {q_abs, q_raw, inv_freq});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_rope_norm_q_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_hy4_mla_sdpa_cuda(
    mlx_array* q_h,
    mlx_array* kv_buf_h,
    mlx_array* sinks_h,
    mlx_array* idx_opt_h,
    int64_t slot_start,
    int32_t total_len,
    int32_t ring_cap,
    int32_t window,
    float scale,
    int32_t kv_lora) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!q_h || !kv_buf_h || !sinks_h || slot_start < 0 || total_len < 0 ||
        kv_lora < 1 || default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& q = *reinterpret_cast<array*>(q_h);
    const auto& kv_buf = *reinterpret_cast<array*>(kv_buf_h);
    const auto& sinks = *reinterpret_cast<array*>(sinks_h);
    if (q.dtype() != bfloat16 || kv_buf.dtype() != float16 ||
        sinks.dtype() != float32 || q.ndim() != 3 || kv_buf.ndim() != 2 ||
        sinks.ndim() != 1) {
      return nullptr;
    }
    const int64_t M = q.shape(0);
    const int64_t H = q.shape(1);
    const int64_t kv_row = q.shape(2);
    const int64_t cap = kv_buf.shape(0);
    if (M < 1 || M > 65535 || H < 1 || H > 65535 || kv_lora > kv_row ||
        kv_buf.shape(1) != kv_row || sinks.shape(0) != H || cap < 1) {
      return nullptr;
    }
    if (ring_cap > 0 && ring_cap > cap) {
      return nullptr;
    }

    std::vector<array> inputs{q, kv_buf, sinks};
    const bool has_idx = (idx_opt_h != nullptr);
    if (has_idx) {
      const auto& idx = *reinterpret_cast<array*>(idx_opt_h);
      if (idx.dtype() != int32 || idx.ndim() != 2 ||
          idx.shape(0) != M || idx.shape(1) < 1) {
        return nullptr;
      }
      inputs.push_back(idx);
    }

    auto res = array(
        Shape{static_cast<int>(M), static_cast<int>(H), kv_lora}, bfloat16,
        std::make_shared<Hy4MlaSdpaCuda>(
            default_stream(Device::gpu), has_idx, slot_start, total_len,
            ring_cap, window, scale, kv_lora),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_mla_sdpa_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" mlx_array* mlx_hy4_ring_write_cuda(
    mlx_array* buf_h,
    mlx_array* rows_h,
    int64_t start_pos,
    int32_t ring_cap,
    bool force_in_place) {
  (void)buf_h; (void)rows_h; (void)start_pos; (void)ring_cap;
  (void)force_in_place;
  return nullptr;
}

extern "C" bool mlx_hy4_mla_prep_kv_store_cuda(
    mlx_array* q_a_opt_h,
    mlx_array* kv_a_h,
    mlx_array* q_a_norm_opt_h,
    mlx_array* kv_a_norm_h,
    mlx_array* inv_freq_h,
    mlx_array* kv_buf_h,
    int64_t slot_start,
    int64_t rope_start,
    int32_t ring_cap,
    float norm_eps,
    mlx_array** out_qr,
    mlx_array** out_kv_buf) {
  (void)q_a_opt_h; (void)kv_a_h; (void)q_a_norm_opt_h; (void)kv_a_norm_h;
  (void)inv_freq_h; (void)kv_buf_h; (void)slot_start; (void)rope_start;
  (void)ring_cap; (void)norm_eps;
  if (out_qr) *out_qr = nullptr;
  if (out_kv_buf) *out_kv_buf = nullptr;
  return false;
}

extern "C" mlx_array* mlx_hy4_rope_norm_q_cuda(
    mlx_array* q_abs_h,
    mlx_array* q_raw_h,
    mlx_array* inv_freq_h,
    int64_t rope_start,
    int32_t nope_dim) {
  (void)q_abs_h; (void)q_raw_h; (void)inv_freq_h;
  (void)rope_start; (void)nope_dim;
  return nullptr;
}

extern "C" mlx_array* mlx_hy4_mla_sdpa_cuda(
    mlx_array* q_h,
    mlx_array* kv_buf_h,
    mlx_array* sinks_h,
    mlx_array* idx_opt_h,
    int64_t slot_start,
    int32_t total_len,
    int32_t ring_cap,
    int32_t window,
    float scale,
    int32_t kv_lora) {
  (void)q_h; (void)kv_buf_h; (void)sinks_h; (void)idx_opt_h;
  (void)slot_start; (void)total_len; (void)ring_cap; (void)window;
  (void)scale; (void)kv_lora;
  return nullptr;
}

#endif
