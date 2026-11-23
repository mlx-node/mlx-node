#include "cuda/qwen4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mlx::core::qwen4_ops {
namespace {

using namespace mlx_node::qwen4_cuda;

constexpr char qwen4_qsa_source[] =
#include "cuda/qwen4_qsa_cuda.cu.inc"
    ;

// ============================================================================
// 4. QSA Sparse Indexer + Zero-KV-Gather Sparse SDPA
// ============================================================================
class Qwen4QsaSparseSdpaCuda final : public UnaryPrimitive {
 public:
  Qwen4QsaSparseSdpaCuda(
      Stream stream, int kv_len, int num_blocks, float scale, int mask_mode)
      : UnaryPrimitive(stream),
        kv_len_(kv_len),
        num_blocks_(num_blocks),
        scale_(scale),
        mask_mode_(mask_mode) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Qwen4QsaSparseSdpaCuda requires CUDA");
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
    array iq = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);

    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int Q_len = q.shape(2);
    const int D = q.shape(3);
    const int Hkv = k.shape(1);
    const int gqa = Hq / Hkv;
    const int base_pos = kv_len_ - Q_len;
    if (B != 1 || D != 256 || gqa < 1 || gqa > 16 || Q_len < 1 ||
        Q_len > 65535 || base_pos < 0 || kv_len_ > k.shape(2) ||
        num_blocks_ < 0) {
      throw std::invalid_argument("Qwen4QsaSparseSdpaCuda: unsupported replay shape");
    }

    array block_ids = alloc_temp({Q_len, 512}, int32, encoder);
    array num_selected = alloc_temp({Q_len}, int32, encoder);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    CUdeviceptr tree_mask_ptr = 0;
    std::optional<array> tree_mask_owned;
    if (mask_mode_ == 1) {
      tree_mask_owned.emplace(
          cu::ensure_row_contiguous_aligned<4>(inputs[5], stream(), encoder));
      tree_mask_ptr =
          reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(*tree_mask_owned));
    }

    const bool use_split_k = (Q_len <= 16);

    // Stage 1: Indexer Top-512 selection
    if (use_split_k) {
      if (num_blocks_ > 0) {
        array pooled_k =
            cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
        array scores = alloc_temp({Q_len, num_blocks_}, float32, encoder);

        encoder.set_input_array(iq);
        encoder.set_input_array(pooled_k);
        encoder.set_output_array(scores);
        auto k_score = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source, "qwen4_qsa_index_score_cuda");
        cu::KernelArgs sargs;
        sargs.append(iq);
        sargs.append(pooled_k);
        sargs.append(scores);
        sargs.append<int32_t>(Q_len);
        sargs.append<int32_t>(num_blocks_);
        encoder.add_kernel_node_raw(
            k_score, dim3((num_blocks_ + 31) / 32, 1, 1), dim3(256, 1, 1), {}, 0,
            sargs.args());

        encoder.set_input_array(scores);
        encoder.set_output_array(block_ids);
        encoder.set_output_array(num_selected);
        auto k_topk = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source, "qwen4_qsa_index_topk512_cuda");
        cu::KernelArgs targs;
        targs.append(scores);
        targs.append(block_ids);
        targs.append(num_selected);
        targs.append<int32_t>(num_blocks_);
        targs.append<int32_t>(base_pos);
        targs.append<int32_t>(mask_mode_);
        encoder.add_kernel_node_raw(
            k_topk, dim3(Q_len, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
      } else {
        encoder.set_output_array(block_ids);
        encoder.set_output_array(num_selected);
        auto k_topk = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source, "qwen4_qsa_index_topk512_cuda");
        cu::KernelArgs targs;
        targs.append<CUdeviceptr>(0);
        targs.append(block_ids);
        targs.append(num_selected);
        targs.append<int32_t>(0);
        targs.append<int32_t>(base_pos);
        targs.append<int32_t>(mask_mode_);
        encoder.add_kernel_node_raw(
            k_topk, dim3(Q_len, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
      }
    } else {
      if (num_blocks_ <= 512) {
        encoder.set_output_array(block_ids);
        encoder.set_output_array(num_selected);
        auto k_topk = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source,
            "qwen4_qsa_index_topk512_radix_cuda");
        cu::KernelArgs targs;
        targs.append<CUdeviceptr>(0);
        targs.append(block_ids);
        targs.append(num_selected);
        targs.append<int32_t>(0);
        targs.append<int32_t>(num_blocks_);
        targs.append<int32_t>(0);
        targs.append<int32_t>(base_pos);
        targs.append<int32_t>(mask_mode_);
        encoder.add_kernel_node_raw(
            k_topk, dim3(Q_len, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
      } else {
        array pooled_k =
            cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
        const int full_stride = (num_blocks_ + 7) & ~7;
        constexpr size_t kTargetScratchFloats = 2 * 1024 * 1024; // 8 MB L2-resident
        int q_chunk = static_cast<int>(
            kTargetScratchFloats / static_cast<size_t>(full_stride));
        q_chunk = (q_chunk / 64) * 64;
        if (q_chunk < 64) q_chunk = 64;
        if (q_chunk > Q_len) q_chunk = Q_len;

        array scores = alloc_temp({q_chunk, full_stride}, float32, encoder);
        auto k_score = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source,
            "qwen4_qsa_index_score_tc_cuda");
        auto k_topk = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source,
            "qwen4_qsa_index_topk512_radix_cuda");

        for (int q0 = 0; q0 < Q_len; q0 += q_chunk) {
          const int q_cnt = std::min(q_chunk, Q_len - q0);
          int max_comp = (mask_mode_ != 0)
              ? (base_pos >> 2)
              : ((base_pos + q0 + q_cnt) >> 2);
          if (max_comp > num_blocks_) max_comp = num_blocks_;
          if (max_comp <= 512) {
            encoder.set_output_array(block_ids);
            encoder.set_output_array(num_selected);
            cu::KernelArgs targs;
            targs.append<CUdeviceptr>(0);
            targs.append(block_ids);
            targs.append(num_selected);
            targs.append<int32_t>(q0);
            targs.append<int32_t>(num_blocks_);
            targs.append<int32_t>(0);
            targs.append<int32_t>(base_pos);
            targs.append<int32_t>(mask_mode_);
            encoder.add_kernel_node_raw(
                k_topk, dim3(q_cnt, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
          } else {
            const int s_stride = (max_comp + 7) & ~7;
            encoder.set_input_array(iq);
            encoder.set_input_array(pooled_k);
            encoder.set_output_array(scores);
            cu::KernelArgs sargs;
            sargs.append(iq);
            sargs.append(pooled_k);
            sargs.append(scores);
            sargs.append<int32_t>(Q_len);
            sargs.append<int32_t>(q0);
            sargs.append<int32_t>(q_cnt);
            sargs.append<int32_t>(max_comp);
            sargs.append<int32_t>(s_stride);
            encoder.add_kernel_node_raw(
                k_score, dim3((max_comp + 63) / 64, (q_cnt + 63) / 64, 1),
                dim3(256, 1, 1), {}, 0, sargs.args());

            encoder.set_input_array(scores);
            encoder.set_output_array(scores);
            encoder.set_output_array(block_ids);
            encoder.set_output_array(num_selected);
            cu::KernelArgs targs;
            targs.append(scores);
            targs.append(block_ids);
            targs.append(num_selected);
            targs.append<int32_t>(q0);
            targs.append<int32_t>(num_blocks_);
            targs.append<int32_t>(s_stride);
            targs.append<int32_t>(base_pos);
            targs.append<int32_t>(mask_mode_);
            encoder.add_kernel_node_raw(
                k_topk, dim3(q_cnt, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
          }
        }
      }
    }

    // Stage 2: Zero-gather Sparse SDPA (+ Split-K reduce for Q_len <= 16)
    const float scale_log2 = scale_ * kLog2E;
    if (use_split_k) {
      const int target_splits = std::max(2, 24 / Q_len);
      const int max_tiles_bound = (std::min(num_blocks_, 512) + 7) / 8 + 1;
      const int num_splits = std::min(target_splits, std::max(1, max_tiles_bound));

      array partial_o =
          alloc_temp({B, Hkv, num_splits, Q_len, gqa, 256}, bfloat16, encoder);
      array partial_m =
          alloc_temp({B, Hkv, num_splits, Q_len, gqa}, float32, encoder);
      array partial_l =
          alloc_temp({B, Hkv, num_splits, Q_len, gqa}, float32, encoder);

      encoder.set_input_array(q);
      encoder.set_input_array(k);
      encoder.set_input_array(v);
      encoder.set_input_array(block_ids);
      encoder.set_input_array(num_selected);
      if (mask_mode_ == 1) {
        encoder.set_input_array(*tree_mask_owned);
      }
      encoder.set_output_array(partial_o);
      encoder.set_output_array(partial_m);
      encoder.set_output_array(partial_l);

      const char* entry = (mask_mode_ == 1)
          ? "qwen4_qsa_sparse_sdpa_cuda<true, 1>"
          : "qwen4_qsa_sparse_sdpa_cuda<true, 0>";
      auto k_sdpa = cu::jit_kernel(encoder.device(), qwen4_qsa_source, entry);
      cu::KernelArgs args;
      args.append(q);
      args.append(k);
      args.append(v);
      args.append(block_ids);
      args.append(num_selected);
      args.append<CUdeviceptr>(tree_mask_ptr);
      args.append(partial_o);
      args.append(partial_m);
      args.append(partial_l);
      args.append<int32_t>(Hkv);
      args.append<int32_t>(gqa);
      args.append<int32_t>(Q_len);
      args.append<int32_t>(kv_len_);
      args.append<int32_t>(512);
      args.append<int32_t>(base_pos);
      args.append<int64_t>(k.strides(0));
      args.append<int64_t>(k.strides(1));
      args.append<int64_t>(k.strides(2));
      args.append<int64_t>(v.strides(0));
      args.append<int64_t>(v.strides(1));
      args.append<int64_t>(v.strides(2));
      args.append<float>(scale_log2);
      args.append<int32_t>(num_splits);
      encoder.add_kernel_node_raw(
          k_sdpa, dim3(num_splits, Q_len, B * Hkv), dim3(32, 4, 1), {}, 0,
          args.args());

      encoder.set_input_array(partial_o);
      encoder.set_input_array(partial_m);
      encoder.set_input_array(partial_l);
      encoder.set_output_array(out);
      auto k_red = cu::jit_kernel(
          encoder.device(), qwen4_qsa_source, "qwen4_qsa_sparse_reduce_cuda");
      cu::KernelArgs rargs;
      rargs.append(partial_o);
      rargs.append(partial_m);
      rargs.append(partial_l);
      rargs.append(out);
      rargs.append<int32_t>(Hkv);
      rargs.append<int32_t>(gqa);
      rargs.append<int32_t>(Q_len);
      rargs.append<int32_t>(num_splits);
      encoder.add_kernel_node_raw(
          k_red, dim3(Q_len * gqa, Hkv, B), dim3(32, 1, 1), {}, 0,
          rargs.args());
    } else if (gqa == 12 && mask_mode_ == 0) {
      constexpr int kMaxUnionTiles = 152; // 1216 blocks adaptive cutoff
      const int qt5 = (Q_len + 4) / 5;
      const int qt4 = (Q_len + 3) / 4;

      array tile_blks = alloc_temp({qt5, kMaxUnionTiles, 8}, int32, encoder);
      array tile_cmask = alloc_temp({qt5, kMaxUnionTiles, 5}, uint32, encoder);
      array tile_counts = alloc_temp({qt5}, int32, encoder);

      // 2A: Parallel 5-way block union + adaptive disjoint detection
      encoder.set_input_array(block_ids);
      encoder.set_input_array(num_selected);
      encoder.set_output_array(tile_blks);
      encoder.set_output_array(tile_cmask);
      encoder.set_output_array(tile_counts);
      auto k_pm = cu::jit_kernel(
          encoder.device(), qwen4_qsa_source, "qwen4_qsa_premerge_bq5_cuda");
      cu::KernelArgs pargs;
      pargs.append(block_ids);
      pargs.append(num_selected);
      pargs.append(tile_blks);
      pargs.append(tile_cmask);
      pargs.append(tile_counts);
      pargs.append<int32_t>(Q_len);
      pargs.append<int32_t>(512);
      pargs.append<int32_t>(kMaxUnionTiles);
      encoder.add_kernel_node_raw(
          k_pm, dim3(qt5, 1, 1), dim3(256, 1, 1), {}, 0, pargs.args());

      // 2B: BQ=5 Q-Tiled Sparse SDPA (60 Q-rows, 4w=128 thr, 8w/SM, 0 spills)
      encoder.set_input_array(q);
      encoder.set_input_array(k);
      encoder.set_input_array(v);
      encoder.set_input_array(tile_blks);
      encoder.set_input_array(tile_cmask);
      encoder.set_input_array(tile_counts);
      encoder.set_output_array(out);
      auto k_qt5 = cu::jit_kernel(
          encoder.device(), qwen4_qsa_source,
          "qwen4_qsa_sdpa_prefill_qtiled5_cuda");
      cu::KernelArgs qargs;
      qargs.append(q);
      qargs.append(k);
      qargs.append(v);
      qargs.append(tile_blks);
      qargs.append(tile_cmask);
      qargs.append(tile_counts);
      qargs.append(out);
      qargs.append<int32_t>(Hkv);
      qargs.append<int32_t>(Q_len);
      qargs.append<int32_t>(kv_len_);
      qargs.append<int32_t>(base_pos);
      qargs.append<int32_t>(kMaxUnionTiles);
      qargs.append<int64_t>(k.strides(1));
      qargs.append<int64_t>(k.strides(2));
      qargs.append<int64_t>(v.strides(1));
      qargs.append<int64_t>(v.strides(2));
      qargs.append<float>(scale_log2);
      encoder.add_kernel_node_raw(
          k_qt5, dim3(qt5, Hkv, 1), dim3(128, 1, 1), {}, 0, qargs.args());

      // 2C: Disjoint Fallback (4 independent warps, BN=16, 0 barriers; instant return if tile_counts>=0)
      encoder.set_input_array(q);
      encoder.set_input_array(k);
      encoder.set_input_array(v);
      encoder.set_input_array(block_ids);
      encoder.set_input_array(num_selected);
      encoder.set_input_array(tile_counts);
      encoder.set_output_array(out);
      auto k_ind = cu::jit_kernel(
          encoder.device(), qwen4_qsa_source,
          "qwen4_qsa_sdpa_prefill_indep4w_cuda");
      cu::KernelArgs iargs;
      iargs.append(q);
      iargs.append(k);
      iargs.append(v);
      iargs.append(block_ids);
      iargs.append(num_selected);
      iargs.append(tile_counts);
      iargs.append(out);
      iargs.append<int32_t>(Hkv);
      iargs.append<int32_t>(gqa);
      iargs.append<int32_t>(Q_len);
      iargs.append<int32_t>(kv_len_);
      iargs.append<int32_t>(512);
      iargs.append<int32_t>(base_pos);
      iargs.append<int64_t>(k.strides(1));
      iargs.append<int64_t>(k.strides(2));
      iargs.append<int64_t>(v.strides(1));
      iargs.append<int64_t>(v.strides(2));
      iargs.append<float>(scale_log2);
      encoder.add_kernel_node_raw(
          k_ind, dim3(qt4, Hkv, 1), dim3(128, 1, 1), {}, 0, iargs.args());
    } else {
      encoder.set_input_array(q);
      encoder.set_input_array(k);
      encoder.set_input_array(v);
      encoder.set_input_array(block_ids);
      encoder.set_input_array(num_selected);
      if (mask_mode_ == 1) {
        encoder.set_input_array(*tree_mask_owned);
      }
      encoder.set_output_array(out);

      const char* entry = (mask_mode_ == 1)
          ? "qwen4_qsa_sparse_sdpa_cuda<false, 1>"
          : "qwen4_qsa_sparse_sdpa_cuda<false, 0>";
      auto k_sdpa = cu::jit_kernel(encoder.device(), qwen4_qsa_source, entry);
      cu::KernelArgs args;
      args.append(q);
      args.append(k);
      args.append(v);
      args.append(block_ids);
      args.append(num_selected);
      args.append<CUdeviceptr>(tree_mask_ptr);
      args.append(out);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
      args.append<int32_t>(Hkv);
      args.append<int32_t>(gqa);
      args.append<int32_t>(Q_len);
      args.append<int32_t>(kv_len_);
      args.append<int32_t>(512);
      args.append<int32_t>(base_pos);
      args.append<int64_t>(k.strides(0));
      args.append<int64_t>(k.strides(1));
      args.append<int64_t>(k.strides(2));
      args.append<int64_t>(v.strides(0));
      args.append<int64_t>(v.strides(1));
      args.append<int64_t>(v.strides(2));
      args.append<float>(scale_log2);
      args.append<int32_t>(1);
      encoder.add_kernel_node_raw(
          k_sdpa, dim3(1, Q_len, B * Hkv), dim3(32, 4, 1), {}, 0, args.args());
    }
  }

  DEFINE_NAME(Qwen4QsaSparseSdpaCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4QsaSparseSdpaCuda*>(&other);
    return rhs && rhs->kv_len_ == kv_len_ && rhs->num_blocks_ == num_blocks_ &&
        rhs->scale_ == scale_ && rhs->mask_mode_ == mask_mode_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int kv_len_;
  int num_blocks_;
  float scale_;
  int mask_mode_;
};

}  // namespace
}  // namespace mlx::core::qwen4_ops

extern "C" mlx_array* mlx_qwen4_qsa_sparse_sdpa_cuda(
    mlx_array* q_h, mlx_array* k_h, mlx_array* v_h,
    mlx_array* iq_h, mlx_array* pooled_k_h, mlx_array* tree_mask_h,
    int32_t kv_len, int32_t num_blocks, float scale, float idx_scale,
    int32_t mask_mode) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    if (!q_h || !k_h || !v_h || !iq_h || !pooled_k_h ||
        !(idx_scale > 0.0f) || (mask_mode != 0 && mask_mode != 1) ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& q = *reinterpret_cast<array*>(q_h);
    const auto& k = *reinterpret_cast<array*>(k_h);
    const auto& v = *reinterpret_cast<array*>(v_h);
    const auto& iq = *reinterpret_cast<array*>(iq_h);
    const auto& pooled_k = *reinterpret_cast<array*>(pooled_k_h);
    if (q.dtype() != bfloat16 || k.dtype() != bfloat16 || v.dtype() != bfloat16 ||
        iq.dtype() != bfloat16 || pooled_k.dtype() != bfloat16 ||
        q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
      return nullptr;
    }
    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int Q_len = q.shape(2);
    const int D = q.shape(3);
    const int Hkv = k.shape(1);
    if (B != 1 || Hkv <= 0 || Hq < Hkv || Hq % Hkv != 0 || D != 256 ||
        Q_len < 1 || Q_len > 65535 ||
        k.shape(0) != 1 || k.shape(3) != 256 || v.shape() != k.shape() ||
        kv_len < Q_len || kv_len > k.shape(2) || num_blocks < 0) {
      return nullptr;
    }
    const int gqa = Hq / Hkv;
    if (gqa < 1 || gqa > 16) {
      return nullptr;
    }
    const bool iq_ok =
        (iq.ndim() == 4 && iq.shape(0) == 1 && iq.shape(1) == 4 &&
         iq.shape(2) == Q_len && iq.shape(3) == 128) ||
        (iq.ndim() == 3 && iq.shape(0) == 4 && iq.shape(1) == Q_len &&
         iq.shape(2) == 128);
    const bool pk_ok =
        (pooled_k.ndim() == 2 && pooled_k.shape(0) >= num_blocks &&
         pooled_k.shape(1) == 128) ||
        (pooled_k.ndim() == 3 && pooled_k.shape(0) == 1 &&
         pooled_k.shape(1) >= num_blocks && pooled_k.shape(2) == 128);
    if (!iq_ok || !pk_ok) {
      return nullptr;
    }

    std::vector<array> inputs{q, k, v, iq, pooled_k};
    if (mask_mode == 1) {
      if (!tree_mask_h || Q_len > 16) {
        return nullptr;
      }
      const auto& tm = *reinterpret_cast<array*>(tree_mask_h);
      if (tm.dtype() != uint32 || tm.ndim() != 1 || tm.shape(0) != Q_len) {
        return nullptr;
      }
      inputs.push_back(tm);
    } else if (tree_mask_h != nullptr) {
      return nullptr;
    }

    auto result = array(
        q.shape(), bfloat16,
        std::make_shared<Qwen4QsaSparseSdpaCuda>(
            default_stream(Device::gpu), kv_len, num_blocks, scale, mask_mode),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_qsa_sparse_sdpa_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" mlx_array* mlx_qwen4_qsa_sparse_sdpa_cuda(
    mlx_array* q_h, mlx_array* k_h, mlx_array* v_h,
    mlx_array* iq_h, mlx_array* pooled_k_h, mlx_array* tree_mask_h,
    int32_t kv_len, int32_t num_blocks, float scale, float idx_scale,
    int32_t mask_mode) {
  (void)q_h; (void)k_h; (void)v_h; (void)iq_h; (void)pooled_k_h;
  (void)tree_mask_h; (void)kv_len; (void)num_blocks; (void)scale;
  (void)idx_scale; (void)mask_mode;
  return nullptr;
}

#endif
