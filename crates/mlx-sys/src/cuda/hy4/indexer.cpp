#include "cuda/hy4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace mlx::core::hy4_ops {
namespace {

using namespace mlx_node::hy4_cuda;

constexpr char hy4_indexer_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_indexer.cu.inc"
    ;

// ============================================================================
// 1. K-Pool Compress (in-place aliased pooled_buf write, CUDA Graph safe)
// ============================================================================
class Hy4KpoolCompressCuda final : public UnaryPrimitive {
 public:
  Hy4KpoolCompressCuda(
      Stream stream,
      int64_t start_group,
      int32_t kpool,
      int32_t n_groups,
      bool ring_mode,
      float norm_eps)
      : UnaryPrimitive(stream),
        start_group_(start_group),
        kpool_(kpool),
        n_groups_(n_groups),
        ring_mode_(ring_mode),
        norm_eps_(norm_eps) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4KpoolCompressCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array tok_k = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w_val = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array w_gate = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array ape = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    array comp_norm = cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);
    array inv_freq = cu::ensure_row_contiguous_aligned<4>(inputs[5], stream(), encoder);
    const array& pooled_buf = inputs[6];

    if (!pooled_buf.flags().row_contiguous) {
      throw std::invalid_argument(
          "Hy4KpoolCompressCuda: pooled_buf must be row-contiguous");
    }
    const void* pb_ptr = pooled_buf.buffer().ptr();
    auto check_no_alias = [&](const array& a) {
      if (a.buffer().ptr() == pb_ptr) {
        throw std::invalid_argument(
            "Hy4KpoolCompressCuda: pooled_buf aliases read input");
      }
    };
    check_no_alias(tok_k);
    check_no_alias(w_val);
    check_no_alias(w_gate);
    check_no_alias(ape);
    check_no_alias(comp_norm);
    check_no_alias(inv_freq);

    const int n_tok = static_cast<int>(tok_k.shape(0));
    const int d_idx = static_cast<int>(tok_k.shape(1));
    const int n_groups = ring_mode_ ? n_groups_ : (n_tok / kpool_);
    const int ring_cap = ring_mode_ ? n_tok : 0;
    const int half_rope = static_cast<int>(inv_freq.shape(0));

    out.copy_shared_buffer(pooled_buf);
    // In linear mode, 0 groups is a pure no-op; in ring mode we still emit one
    // lightweight no-op block (`grid_x = 1`, `n_groups = 0`) so decode CUDA Graph
    // topology stays constant across every step and `tok_k` stays a live input.
    if (!ring_mode_ && n_groups <= 0) {
      return;
    }
    const int grid_x = std::max(n_groups, 1);

    encoder.set_input_array(tok_k);
    encoder.set_input_array(w_val);
    encoder.set_input_array(w_gate);
    encoder.set_input_array(ape);
    encoder.set_input_array(comp_norm);
    encoder.set_input_array(inv_freq);
    encoder.set_output_array(out);

    const bool tok_bf16 = (tok_k.dtype() == bfloat16);
    if (d_idx == 128 && kpool_ == 4) {
      auto fn = cu::jit_kernel(
          encoder.device(), hy4_indexer_source,
          tok_bf16 ? "hy4_kpool_compress_128_4<true>"
                   : "hy4_kpool_compress_128_4<false>");
      cu::KernelArgs args;
      args.append(tok_k);
      args.append(w_val);
      args.append(w_gate);
      args.append(ape);
      args.append(comp_norm);
      args.append(inv_freq);
      args.append(out);
      args.append<int32_t>(n_groups);
      args.append<int64_t>(start_group_);
      args.append<int32_t>(half_rope);
      args.append<int32_t>(ring_cap);
      args.append<float>(norm_eps_);
      encoder.add_kernel_node_raw(
          fn, dim3(grid_x, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
    } else {
      auto fn = cu::jit_kernel(
          encoder.device(), hy4_indexer_source,
          tok_bf16 ? "hy4_kpool_compress_generic<true>"
                   : "hy4_kpool_compress_generic<false>");
      const int smem_bytes = d_idx * static_cast<int>(sizeof(float));
      ensure_dynamic_smem(fn, smem_bytes);
      cu::KernelArgs args;
      args.append(tok_k);
      args.append(w_val);
      args.append(w_gate);
      args.append(ape);
      args.append(comp_norm);
      args.append(inv_freq);
      args.append(out);
      args.append<int32_t>(n_groups);
      args.append<int32_t>(d_idx);
      args.append<int32_t>(kpool_);
      args.append<int64_t>(start_group_);
      args.append<int32_t>(half_rope);
      args.append<int32_t>(ring_cap);
      args.append<float>(norm_eps_);
      encoder.add_kernel_node_raw(
          fn, dim3(grid_x, 1, 1), dim3(256, 1, 1), {}, smem_bytes, args.args());
    }
  }

  DEFINE_NAME(Hy4KpoolCompressCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4KpoolCompressCuda*>(&other);
    return rhs && rhs->start_group_ == start_group_ && rhs->kpool_ == kpool_ &&
        rhs->n_groups_ == n_groups_ && rhs->ring_mode_ == ring_mode_ &&
        rhs->norm_eps_ == norm_eps_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[6].shape()};
  }

 private:
  int64_t start_group_;
  int32_t kpool_;
  int32_t n_groups_;
  bool ring_mode_;
  float norm_eps_;
};

// ============================================================================
// 2. K-Pool Lightning Indexer Select
// ============================================================================
class Hy4IndexerSelectCuda final : public UnaryPrimitive {
 public:
  Hy4IndexerSelectCuda(
      Stream stream,
      int64_t slot_start,
      int64_t rope_start,
      int32_t kpool,
      int32_t n_top,
      int32_t kmax,
      bool select_tail,
      float weight_scale)
      : UnaryPrimitive(stream),
        slot_start_(slot_start),
        rope_start_(rope_start),
        kpool_(kpool),
        n_top_(n_top),
        kmax_(kmax),
        select_tail_(select_tail),
        weight_scale_(weight_scale) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4IndexerSelectCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array q_raw = cu::ensure_row_contiguous_aligned<2>(inputs[0], stream(), encoder);
    const array& w_in = inputs[1];
    array w_proj = (w_in.dtype() == float32)
        ? cu::ensure_row_contiguous_aligned<4>(w_in, stream(), encoder)
        : cu::ensure_row_contiguous_aligned<2>(w_in, stream(), encoder);
    const int d_idx = static_cast<int>(q_raw.shape(2));
    array pooled_buf = (d_idx == 128)
        ? cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
    array inv_freq = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);

    const int M = static_cast<int>(q_raw.shape(0));
    const int H_idx = static_cast<int>(q_raw.shape(1));
    const int half_rope = static_cast<int>(inv_freq.shape(0));

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    const int64_t p_last = slot_start_ + static_cast<int64_t>(M) - 1;
    const int64_t max_vis_ll = select_tail_
        ? (p_last / static_cast<int64_t>(kpool_))
        : ((p_last + 1) / static_cast<int64_t>(kpool_));
    const int max_vis = static_cast<int>(max_vis_ll);

    // Fast path: every query row in the batch is dense (vis <= n_top)
    if (max_vis <= n_top_) {
      encoder.set_output_array(out);
      auto fn_dense = cu::jit_kernel(
          encoder.device(), hy4_indexer_source, "hy4_indexer_dense_fill");
      cu::KernelArgs dargs;
      dargs.append(out);
      dargs.append<int32_t>(M);
      dargs.append<int32_t>(kmax_);
      dargs.append<int64_t>(slot_start_);
      encoder.add_kernel_node_raw(
          fn_dense, dim3(M, 1, 1), dim3(256, 1, 1), {}, 0, dargs.args());
      return;
    }

    if (n_top_ == 0) {
      encoder.set_output_array(out);
      auto fn_radix = cu::jit_kernel(
          encoder.device(), hy4_indexer_source, "hy4_indexer_radix_select");
      cu::KernelArgs targs;
      targs.append<CUdeviceptr>(0);
      targs.append(out);
      targs.append<int32_t>(0);
      targs.append<int32_t>(0);
      targs.append<int64_t>(slot_start_);
      targs.append<int32_t>(kpool_);
      targs.append<int32_t>(0);
      targs.append<int32_t>(kmax_);
      targs.append<int32_t>(select_tail_ ? 1 : 0);
      encoder.add_kernel_node_raw(
          fn_radix, dim3(M, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
      return;
    }

    const bool w_f32 = (w_proj.dtype() == float32);
    const bool use_tc_prefill = (M > 16 && d_idx == 128);
    const bool use_decode_128 = (M <= 16 && d_idx == 128);

    // Stage 1: Tail NORM-RoPE on q_idx_raw
    array q_roped = alloc_temp(
        {M, H_idx, d_idx}, use_tc_prefill ? bfloat16 : float32, encoder);
    encoder.set_input_array(q_raw);
    encoder.set_input_array(inv_freq);
    encoder.set_output_array(q_roped);
    auto fn_rope = cu::jit_kernel(
        encoder.device(), hy4_indexer_source,
        use_tc_prefill ? "hy4_indexer_q_rope<false>"
                       : "hy4_indexer_q_rope<true>");
    cu::KernelArgs rargs;
    rargs.append(q_raw);
    rargs.append(inv_freq);
    rargs.append(q_roped);
    rargs.append<int32_t>(M);
    rargs.append<int32_t>(H_idx);
    rargs.append<int32_t>(d_idx);
    rargs.append<int32_t>(half_rope);
    rargs.append<int64_t>(rope_start_);
    encoder.add_kernel_node_raw(
        fn_rope, dim3(H_idx, M, 1), dim3(128, 1, 1), {}, 0, rargs.args());

    auto fn_radix = cu::jit_kernel(
        encoder.device(), hy4_indexer_source, "hy4_indexer_radix_select");

    if (use_decode_128) {
      const int s_stride = (max_vis + 7) & ~7;
      array scores = alloc_temp({M, s_stride}, float32, encoder);

      encoder.set_input_array(q_roped);
      encoder.set_input_array(w_proj);
      encoder.set_input_array(pooled_buf);
      encoder.set_output_array(scores);
      auto fn_sc = cu::jit_kernel(
          encoder.device(), hy4_indexer_source,
          w_f32 ? "hy4_indexer_score_decode_128<true>"
                : "hy4_indexer_score_decode_128<false>");
      cu::KernelArgs sargs;
      sargs.append(q_roped);
      sargs.append(w_proj);
      sargs.append(pooled_buf);
      sargs.append(scores);
      sargs.append<int32_t>(M);
      sargs.append<int32_t>(H_idx);
      sargs.append<int32_t>(max_vis);
      sargs.append<int32_t>(s_stride);
      sargs.append<float>(weight_scale_);
      encoder.add_kernel_node_raw(
          fn_sc, dim3((max_vis + 31) / 32, 1, 1), dim3(256, 1, 1), {}, 0,
          sargs.args());

      encoder.set_input_array(scores);
      encoder.set_output_array(out);
      cu::KernelArgs targs;
      targs.append(scores);
      targs.append(out);
      targs.append<int32_t>(0);
      targs.append<int32_t>(s_stride);
      targs.append<int64_t>(slot_start_);
      targs.append<int32_t>(kpool_);
      targs.append<int32_t>(n_top_);
      targs.append<int32_t>(kmax_);
      targs.append<int32_t>(select_tail_ ? 1 : 0);
      encoder.add_kernel_node_raw(
          fn_radix, dim3(M, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
      return;
    }

    // Chunked scoring + radix-select to keep score scratch L2-resident (<= 8 MB)
    const int full_stride = (max_vis + 7) & ~7;
    constexpr size_t kTargetScratchFloats = 2 * 1024 * 1024;
    int q_chunk = static_cast<int>(
        kTargetScratchFloats / static_cast<size_t>(std::max(1, full_stride)));
    if (use_tc_prefill) {
      q_chunk = (q_chunk / 64) * 64;
      if (q_chunk < 64) q_chunk = 64;
    } else {
      if (q_chunk < 16) q_chunk = 16;
    }
    if (q_chunk > M) q_chunk = M;

    array scores = alloc_temp({q_chunk, full_stride}, float32, encoder);
    CUfunction fn_sc = use_tc_prefill
        ? cu::jit_kernel(
              encoder.device(), hy4_indexer_source,
              w_f32 ? "hy4_indexer_score_tc_128<true>"
                    : "hy4_indexer_score_tc_128<false>")
        : cu::jit_kernel(
              encoder.device(), hy4_indexer_source,
              w_f32 ? "hy4_indexer_score_generic<true>"
                    : "hy4_indexer_score_generic<false>");

    for (int q0 = 0; q0 < M; q0 += q_chunk) {
      const int q_cnt = std::min(q_chunk, M - q0);
      const int64_t p_chunk_last = slot_start_ + static_cast<int64_t>(q0 + q_cnt - 1);
      const int chunk_max_vis = static_cast<int>(select_tail_
          ? (p_chunk_last / static_cast<int64_t>(kpool_))
          : ((p_chunk_last + 1) / static_cast<int64_t>(kpool_)));

      if (chunk_max_vis <= n_top_) {
        encoder.set_output_array(out);
        cu::KernelArgs targs;
        targs.append<CUdeviceptr>(0);
        targs.append(out);
        targs.append<int32_t>(q0);
        targs.append<int32_t>(0);
        targs.append<int64_t>(slot_start_);
        targs.append<int32_t>(kpool_);
        targs.append<int32_t>(n_top_);
        targs.append<int32_t>(kmax_);
        targs.append<int32_t>(select_tail_ ? 1 : 0);
        encoder.add_kernel_node_raw(
            fn_radix, dim3(q_cnt, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
        continue;
      }

      const int s_stride = (chunk_max_vis + 7) & ~7;
      encoder.set_input_array(q_roped);
      encoder.set_input_array(w_proj);
      encoder.set_input_array(pooled_buf);
      encoder.set_output_array(scores);

      if (use_tc_prefill) {
        cu::KernelArgs sargs;
        sargs.append(q_roped);
        sargs.append(w_proj);
        sargs.append(pooled_buf);
        sargs.append(scores);
        sargs.append<int32_t>(H_idx);
        sargs.append<int32_t>(q0);
        sargs.append<int32_t>(q_cnt);
        sargs.append<int32_t>(chunk_max_vis);
        sargs.append<int32_t>(s_stride);
        sargs.append<int64_t>(slot_start_);
        sargs.append<int32_t>(kpool_);
        sargs.append<int32_t>(select_tail_ ? 1 : 0);
        sargs.append<float>(weight_scale_);
        encoder.add_kernel_node_raw(
            fn_sc, dim3((chunk_max_vis + 63) / 64, (q_cnt + 63) / 64, 1),
            dim3(256, 1, 1), {}, 0, sargs.args());
      } else {
        cu::KernelArgs sargs;
        sargs.append(q_roped);
        sargs.append(w_proj);
        sargs.append(pooled_buf);
        sargs.append(scores);
        sargs.append<int32_t>(H_idx);
        sargs.append<int32_t>(d_idx);
        sargs.append<int32_t>(q0);
        sargs.append<int32_t>(q_cnt);
        sargs.append<int32_t>(chunk_max_vis);
        sargs.append<int32_t>(s_stride);
        sargs.append<float>(weight_scale_);
        encoder.add_kernel_node_raw(
            fn_sc, dim3((chunk_max_vis + 7) / 8, q_cnt, 1),
            dim3(32, 8, 1), {}, 0, sargs.args());
      }

      encoder.set_input_array(scores);
      encoder.set_output_array(out);
      cu::KernelArgs targs;
      targs.append(scores);
      targs.append(out);
      targs.append<int32_t>(q0);
      targs.append<int32_t>(s_stride);
      targs.append<int64_t>(slot_start_);
      targs.append<int32_t>(kpool_);
      targs.append<int32_t>(n_top_);
      targs.append<int32_t>(kmax_);
      targs.append<int32_t>(select_tail_ ? 1 : 0);
      encoder.add_kernel_node_raw(
          fn_radix, dim3(q_cnt, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
    }
  }

  DEFINE_NAME(Hy4IndexerSelectCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4IndexerSelectCuda*>(&other);
    return rhs && rhs->slot_start_ == slot_start_ &&
        rhs->rope_start_ == rope_start_ && rhs->kpool_ == kpool_ &&
        rhs->n_top_ == n_top_ && rhs->kmax_ == kmax_ &&
        rhs->select_tail_ == select_tail_ &&
        rhs->weight_scale_ == weight_scale_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{inputs[0].shape(0), kmax_}};
  }

 private:
  int64_t slot_start_;
  int64_t rope_start_;
  int32_t kpool_;
  int32_t n_top_;
  int32_t kmax_;
  bool select_tail_;
  float weight_scale_;
};

}  // namespace
}  // namespace mlx::core::hy4_ops

extern "C" mlx_array* mlx_hy4_kpool_compress_cuda(
    mlx_array* tok_k_h,
    mlx_array* w_val_h,
    mlx_array* w_gate_h,
    mlx_array* ape_h,
    mlx_array* comp_norm_h,
    mlx_array* inv_freq_h,
    mlx_array* pooled_buf_h,
    int64_t start_group,
    int32_t kpool,
    float norm_eps) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!tok_k_h || !w_val_h || !w_gate_h || !ape_h || !comp_norm_h ||
        !inv_freq_h || !pooled_buf_h || start_group < 0 || kpool == 0 ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    // Positive `kpool` (1..=16) reads linear `tok_k [n_groups * kpool, d_idx]`.
    // Negative `kpool` encodes circular ring mode `-((n_groups << 8) | kp)`,
    // reading `tok_k [ring_cap, d_idx]` at `((start_group + g) * kp + s) % ring_cap`
    // and accepting `n_groups == 0` for constant decode CUDA Graph topology.
    const bool ring_mode = (kpool < 0);
    int32_t kp = kpool;
    int32_t explicit_n_groups = -1;
    if (ring_mode) {
      if (kpool == INT32_MIN) {
        return nullptr;
      }
      const int32_t raw = -kpool;
      kp = raw & 0xFF;
      explicit_n_groups = raw >> 8;
    }
    if (kp < 1 || kp > 16) {
      return nullptr;
    }
    const auto& tok_k = *reinterpret_cast<array*>(tok_k_h);
    const auto& w_val = *reinterpret_cast<array*>(w_val_h);
    const auto& w_gate = *reinterpret_cast<array*>(w_gate_h);
    const auto& ape = *reinterpret_cast<array*>(ape_h);
    const auto& comp_norm = *reinterpret_cast<array*>(comp_norm_h);
    const auto& inv_freq = *reinterpret_cast<array*>(inv_freq_h);
    const auto& pooled_buf = *reinterpret_cast<array*>(pooled_buf_h);
    if ((tok_k.dtype() != float32 && tok_k.dtype() != bfloat16) ||
        w_val.dtype() != float32 || w_gate.dtype() != float32 ||
        ape.dtype() != float32 || comp_norm.dtype() != float32 ||
        inv_freq.dtype() != float32 || pooled_buf.dtype() != float16 ||
        tok_k.ndim() != 2 || w_val.ndim() != 2 || w_gate.ndim() != 2 ||
        ape.ndim() != 2 || comp_norm.ndim() != 1 || inv_freq.ndim() != 1 ||
        pooled_buf.ndim() != 2 || !pooled_buf.flags().row_contiguous) {
      return nullptr;
    }
    const int64_t n_tok = tok_k.shape(0);
    const int64_t d_idx = tok_k.shape(1);
    const int64_t half_rope = inv_freq.shape(0);
    const int64_t cap_g = pooled_buf.shape(0);
    if (d_idx < 2 || half_rope < 0 || half_rope * 2 > d_idx) {
      return nullptr;
    }
    int64_t n_groups = 0;
    if (ring_mode) {
      if (n_tok < 1 || explicit_n_groups < 0 || explicit_n_groups > 65535 ||
          static_cast<int64_t>(explicit_n_groups) * kp > n_tok) {
        return nullptr;
      }
      n_groups = explicit_n_groups;
    } else {
      if (n_tok < 0 || (n_tok % kp) != 0) {
        return nullptr;
      }
      n_groups = n_tok / kp;
    }
    if (n_groups > 65535 || start_group + n_groups > cap_g ||
        w_val.shape(0) != d_idx || w_val.shape(1) != d_idx ||
        w_gate.shape(0) != d_idx || w_gate.shape(1) != d_idx ||
        ape.shape(0) != kp || ape.shape(1) != d_idx ||
        comp_norm.shape(0) != d_idx || pooled_buf.shape(1) != d_idx) {
      return nullptr;
    }
    auto res = array(
        pooled_buf.shape(), float16,
        std::make_shared<Hy4KpoolCompressCuda>(
            default_stream(Device::gpu),
            start_group,
            kp,
            static_cast<int32_t>(n_groups),
            ring_mode,
            norm_eps),
        {tok_k, w_val, w_gate, ape, comp_norm, inv_freq, pooled_buf});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_kpool_compress_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_hy4_indexer_select_cuda(
    mlx_array* q_idx_raw_h,
    mlx_array* w_proj_h,
    mlx_array* pooled_buf_h,
    mlx_array* inv_freq_h,
    int64_t slot_start,
    int64_t rope_start,
    int32_t kpool,
    int32_t top_k,
    bool select_tail,
    float weight_scale) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!q_idx_raw_h || !w_proj_h || !pooled_buf_h || !inv_freq_h ||
        slot_start < 0 || rope_start < 0 || kpool < 1 || top_k < kpool ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& q_raw = *reinterpret_cast<array*>(q_idx_raw_h);
    const auto& w_proj = *reinterpret_cast<array*>(w_proj_h);
    const auto& pooled_buf = *reinterpret_cast<array*>(pooled_buf_h);
    const auto& inv_freq = *reinterpret_cast<array*>(inv_freq_h);
    if (q_raw.dtype() != bfloat16 ||
        (w_proj.dtype() != bfloat16 && w_proj.dtype() != float32) ||
        pooled_buf.dtype() != float16 || inv_freq.dtype() != float32 ||
        q_raw.ndim() != 3 || w_proj.ndim() != 2 ||
        pooled_buf.ndim() != 2 || inv_freq.ndim() != 1) {
      return nullptr;
    }
    const int64_t M = q_raw.shape(0);
    const int64_t H_idx = q_raw.shape(1);
    const int64_t d_idx = q_raw.shape(2);
    const int64_t half_rope = inv_freq.shape(0);
    const int64_t cap_g = pooled_buf.shape(0);
    if (M < 1 || M > 65535 || H_idx < 1 || H_idx > 65535 || d_idx < 2 ||
        half_rope < 0 || half_rope * 2 > d_idx ||
        w_proj.shape(0) != M || w_proj.shape(1) != H_idx ||
        pooled_buf.shape(1) != d_idx) {
      return nullptr;
    }
    const int32_t n_top = top_k / kpool - (select_tail ? 1 : 0);
    const int32_t tail_slots = kpool - (select_tail ? 0 : 1);
    const int32_t kmax = kpool * n_top + tail_slots;
    if (n_top < 0 || kmax < 1) {
      return nullptr;
    }
    const int64_t p_last = slot_start + M - 1;
    const int64_t max_vis = select_tail
        ? (p_last / static_cast<int64_t>(kpool))
        : ((p_last + 1) / static_cast<int64_t>(kpool));
    if (max_vis > n_top && max_vis > cap_g) {
      return nullptr;
    }

    auto res = array(
        Shape{static_cast<int>(M), kmax}, int32,
        std::make_shared<Hy4IndexerSelectCuda>(
            default_stream(Device::gpu), slot_start, rope_start,
            kpool, n_top, kmax, select_tail, weight_scale),
        {q_raw, w_proj, pooled_buf, inv_freq});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_indexer_select_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" mlx_array* mlx_hy4_kpool_compress_cuda(
    mlx_array* tok_k_h,
    mlx_array* w_val_h,
    mlx_array* w_gate_h,
    mlx_array* ape_h,
    mlx_array* comp_norm_h,
    mlx_array* inv_freq_h,
    mlx_array* pooled_buf_h,
    int64_t start_group,
    int32_t kpool,
    float norm_eps) {
  (void)tok_k_h; (void)w_val_h; (void)w_gate_h; (void)ape_h;
  (void)comp_norm_h; (void)inv_freq_h; (void)pooled_buf_h;
  (void)start_group; (void)kpool; (void)norm_eps;
  return nullptr;
}

extern "C" mlx_array* mlx_hy4_indexer_select_cuda(
    mlx_array* q_idx_raw_h,
    mlx_array* w_proj_h,
    mlx_array* pooled_buf_h,
    mlx_array* inv_freq_h,
    int64_t slot_start,
    int64_t rope_start,
    int32_t kpool,
    int32_t top_k,
    bool select_tail,
    float weight_scale) {
  (void)q_idx_raw_h; (void)w_proj_h; (void)pooled_buf_h; (void)inv_freq_h;
  (void)slot_start; (void)rope_start; (void)kpool; (void)top_k;
  (void)select_tail; (void)weight_scale;
  return nullptr;
}

#endif
