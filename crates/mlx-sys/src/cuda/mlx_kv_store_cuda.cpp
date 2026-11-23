#include "mlx_common.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"
#endif

namespace {

#ifdef MLX_NODE_CUDA_ENABLED
constexpr int kMaxTensors = 256;

using namespace mlx::core;

constexpr char kQkNormRopeKvCudaSrc[] =
#include "cuda/qk_norm_rope_kv.cu.inc"
    ;

constexpr char kKvStoreRowsCudaSrc[] =
#include "cuda/kv_store_rows.cu.inc"
    ;

class QkNormRopeKvStoreCuda final : public Primitive {
 public:
  QkNormRopeKvStoreCuda(
      Stream stream,
      float eps,
      int32_t rope_dims,
      float rope_base,
      float rope_scale,
      int32_t offset)
      : Primitive(stream),
        eps_(eps),
        rope_dims_(rope_dims),
        rope_base_(rope_base),
        rope_scale_(rope_scale),
        offset_(offset) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("QkNormRopeKvStoreCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto ensure_last_contiguous = [&](const array& x) {
      if (x.strides(3) == 1) {
        return x;
      }
      auto copy = contiguous_copy_gpu(x, stream());
      encoder.add_temporary(copy);
      return copy;
    };

    const array q_in = ensure_last_contiguous(inputs[0]);
    const array k_in = ensure_last_contiguous(inputs[1]);
    const array v_in = ensure_last_contiguous(inputs[2]);
    const array q_w = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);
    const array k_w = cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);
    const array& k_cache = inputs[5];
    const array& v_cache = inputs[6];

    const int B = q_in.shape(0);
    const int T = q_in.shape(1);
    const int Hq = q_in.shape(2);
    const int D = q_in.shape(3);
    const int Hkv = k_in.shape(2);
    const int cap = k_cache.shape(2);
    if (B < 1 || T < 1 || Hq < 1 || Hkv < 1 || D < 2 || D > 1024 || (D & 1) != 0 ||
        k_in.shape() != Shape{B, T, Hkv, D} || v_in.shape() != Shape{B, T, Hkv, D} ||
        k_cache.shape() != Shape{B, Hkv, cap, D} ||
        v_cache.shape() != Shape{B, Hkv, cap, D} ||
        k_cache.strides(3) != 1 || v_cache.strides(3) != 1 ||
        offset_ < 0 || int64_t(offset_) + T > cap ||
        rope_dims_ < 0 || rope_dims_ > D || (rope_dims_ & 1) != 0) {
      throw std::invalid_argument("QkNormRopeKvStoreCuda: invalid shape or bounds");
    }
    const void* kc_ptr = k_cache.buffer().ptr();
    const void* vc_ptr = v_cache.buffer().ptr();
    if (kc_ptr == vc_ptr ||
        kc_ptr == q_in.buffer().ptr() || kc_ptr == k_in.buffer().ptr() ||
        kc_ptr == v_in.buffer().ptr() ||
        vc_ptr == q_in.buffer().ptr() || vc_ptr == k_in.buffer().ptr() ||
        vc_ptr == v_in.buffer().ptr()) {
      throw std::invalid_argument("QkNormRopeKvStoreCuda: cache buffer aliases input");
    }

    auto& q_out = outputs[0];
    auto& k_cache_out = outputs[1];
    auto& v_cache_out = outputs[2];
    q_out.set_data(cu::malloc_async(q_out.nbytes(), encoder));
    k_cache_out.copy_shared_buffer(k_cache);
    v_cache_out.copy_shared_buffer(v_cache);

    encoder.set_input_array(q_in);
    encoder.set_input_array(k_in);
    encoder.set_input_array(v_in);
    encoder.set_input_array(q_w);
    encoder.set_input_array(k_w);
    // Do NOT call encoder.set_input_array(k_cache/v_cache): k_cache_out and
    // v_cache_out alias them and set_output_array already tracks dependencies
    // without inflating bytes_in_graph_ by the preallocated KV buffers.
    encoder.set_output_array(q_out);
    encoder.set_output_array(k_cache_out);
    encoder.set_output_array(v_cache_out);

    const char* entry = D == 256
        ? "mlx_qk_norm_rope_kv_store_cuda_kernel<true>"
        : "mlx_qk_norm_rope_kv_store_cuda_kernel<false>";
    auto kernel = cu::jit_kernel(encoder.device(), kQkNormRopeKvCudaSrc, entry);

    cu::KernelArgs args;
    args.append(q_in);
    args.append(k_in);
    args.append(v_in);
    args.append(q_w);
    args.append(k_w);
    args.append(q_out);
    args.append(k_cache_out);
    args.append(v_cache_out);
    args.append<int32_t>(T);
    args.append<int32_t>(Hq);
    args.append<int32_t>(D);
    args.append<int32_t>(rope_dims_);
    args.append<float>(std::log2(rope_base_));
    args.append<float>(rope_scale_);
    args.append<float>(eps_);
    args.append<int32_t>(offset_);
    args.append<int64_t>(q_in.strides(0));
    args.append<int64_t>(q_in.strides(1));
    args.append<int64_t>(q_in.strides(2));
    args.append<int64_t>(k_in.strides(0));
    args.append<int64_t>(k_in.strides(1));
    args.append<int64_t>(k_in.strides(2));
    args.append<int64_t>(v_in.strides(0));
    args.append<int64_t>(v_in.strides(1));
    args.append<int64_t>(v_in.strides(2));
    args.append<int64_t>(k_cache.strides(0));
    args.append<int64_t>(k_cache.strides(1));
    args.append<int64_t>(k_cache.strides(2));
    args.append<int64_t>(v_cache.strides(0));
    args.append<int64_t>(v_cache.strides(1));
    args.append<int64_t>(v_cache.strides(2));

    encoder.add_kernel_node_raw(
        kernel, dim3(T, Hq + Hkv, B), dim3(128, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(QkNormRopeKvStoreCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const QkNormRopeKvStoreCuda*>(&other);
    return rhs && rhs->eps_ == eps_ && rhs->rope_dims_ == rope_dims_ &&
        rhs->rope_base_ == rope_base_ && rhs->rope_scale_ == rope_scale_ &&
        rhs->offset_ == offset_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    return {
        {q.shape(0), q.shape(2), q.shape(1), q.shape(3)},
        inputs[5].shape(),
        inputs[6].shape(),
    };
  }

 private:
  float eps_;
  int32_t rope_dims_;
  float rope_base_;
  float rope_scale_;
  int32_t offset_;
};

class QkNormRopeKvStoreTreeCuda final : public Primitive {
 public:
  QkNormRopeKvStoreTreeCuda(
      Stream stream,
      float eps,
      int32_t rope_dims,
      float rope_base,
      float rope_scale,
      int32_t offset)
      : Primitive(stream),
        eps_(eps),
        rope_dims_(rope_dims),
        rope_base_(rope_base),
        rope_scale_(rope_scale),
        offset_(offset) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("QkNormRopeKvStoreTreeCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto ensure_last_contiguous = [&](const array& x) {
      if (x.strides(3) == 1) {
        return x;
      }
      auto copy = contiguous_copy_gpu(x, stream());
      encoder.add_temporary(copy);
      return copy;
    };

    const array q_in = ensure_last_contiguous(inputs[0]);
    const array k_in = ensure_last_contiguous(inputs[1]);
    const array v_in = ensure_last_contiguous(inputs[2]);
    const array q_w = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);
    const array k_w = cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);
    const array& k_cache = inputs[5];
    const array& v_cache = inputs[6];
    const array depths = cu::ensure_row_contiguous_aligned<4>(inputs[7], stream(), encoder);

    const int B = q_in.shape(0);
    const int T = q_in.shape(1);
    const int Hq = q_in.shape(2);
    const int D = q_in.shape(3);
    const int Hkv = k_in.shape(2);
    const int cap = k_cache.shape(2);
    if (B < 1 || T < 1 || Hq < 1 || Hkv < 1 || D < 2 || D > 1024 || (D & 1) != 0 ||
        k_in.shape() != Shape{B, T, Hkv, D} || v_in.shape() != Shape{B, T, Hkv, D} ||
        k_cache.shape() != Shape{B, Hkv, cap, D} ||
        v_cache.shape() != Shape{B, Hkv, cap, D} ||
        k_cache.strides(3) != 1 || v_cache.strides(3) != 1 ||
        depths.ndim() != 1 || depths.shape(0) != T || depths.dtype() != int32 ||
        offset_ < 0 || int64_t(offset_) + T > cap ||
        rope_dims_ < 0 || rope_dims_ > D || (rope_dims_ & 1) != 0) {
      throw std::invalid_argument("QkNormRopeKvStoreTreeCuda: invalid shape or bounds");
    }
    const void* kc_ptr = k_cache.buffer().ptr();
    const void* vc_ptr = v_cache.buffer().ptr();
    if (kc_ptr == vc_ptr ||
        kc_ptr == q_in.buffer().ptr() || kc_ptr == k_in.buffer().ptr() ||
        kc_ptr == v_in.buffer().ptr() || kc_ptr == depths.buffer().ptr() ||
        vc_ptr == q_in.buffer().ptr() || vc_ptr == k_in.buffer().ptr() ||
        vc_ptr == v_in.buffer().ptr() || vc_ptr == depths.buffer().ptr()) {
      throw std::invalid_argument("QkNormRopeKvStoreTreeCuda: cache buffer aliases input");
    }

    auto& q_out = outputs[0];
    auto& k_cache_out = outputs[1];
    auto& v_cache_out = outputs[2];
    q_out.set_data(cu::malloc_async(q_out.nbytes(), encoder));
    k_cache_out.copy_shared_buffer(k_cache);
    v_cache_out.copy_shared_buffer(v_cache);

    encoder.set_input_array(q_in);
    encoder.set_input_array(k_in);
    encoder.set_input_array(v_in);
    encoder.set_input_array(q_w);
    encoder.set_input_array(k_w);
    encoder.set_input_array(depths);
    // Do NOT call encoder.set_input_array(k_cache/v_cache): k_cache_out and
    // v_cache_out alias them and set_output_array already tracks dependencies
    // without inflating bytes_in_graph_ by the preallocated KV buffers.
    encoder.set_output_array(q_out);
    encoder.set_output_array(k_cache_out);
    encoder.set_output_array(v_cache_out);

    const char* entry = D == 256
        ? "mlx_qk_norm_rope_kv_store_tree_cuda_kernel<true>"
        : "mlx_qk_norm_rope_kv_store_tree_cuda_kernel<false>";
    auto kernel = cu::jit_kernel(encoder.device(), kQkNormRopeKvCudaSrc, entry);

    cu::KernelArgs args;
    args.append(q_in);
    args.append(k_in);
    args.append(v_in);
    args.append(q_w);
    args.append(k_w);
    args.append(depths);
    args.append(q_out);
    args.append(k_cache_out);
    args.append(v_cache_out);
    args.append<int32_t>(T);
    args.append<int32_t>(Hq);
    args.append<int32_t>(D);
    args.append<int32_t>(rope_dims_);
    args.append<float>(std::log2(rope_base_));
    args.append<float>(rope_scale_);
    args.append<float>(eps_);
    args.append<int32_t>(offset_);
    args.append<int64_t>(q_in.strides(0));
    args.append<int64_t>(q_in.strides(1));
    args.append<int64_t>(q_in.strides(2));
    args.append<int64_t>(k_in.strides(0));
    args.append<int64_t>(k_in.strides(1));
    args.append<int64_t>(k_in.strides(2));
    args.append<int64_t>(v_in.strides(0));
    args.append<int64_t>(v_in.strides(1));
    args.append<int64_t>(v_in.strides(2));
    args.append<int64_t>(k_cache.strides(0));
    args.append<int64_t>(k_cache.strides(1));
    args.append<int64_t>(k_cache.strides(2));
    args.append<int64_t>(v_cache.strides(0));
    args.append<int64_t>(v_cache.strides(1));
    args.append<int64_t>(v_cache.strides(2));

    encoder.add_kernel_node_raw(
        kernel, dim3(T, Hq + Hkv, B), dim3(128, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(QkNormRopeKvStoreTreeCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const QkNormRopeKvStoreTreeCuda*>(&other);
    return rhs && rhs->eps_ == eps_ && rhs->rope_dims_ == rope_dims_ &&
        rhs->rope_base_ == rope_base_ && rhs->rope_scale_ == rope_scale_ &&
        rhs->offset_ == offset_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    return {
        {q.shape(0), q.shape(2), q.shape(1), q.shape(3)},
        inputs[5].shape(),
        inputs[6].shape(),
    };
  }

 private:
  float eps_;
  int32_t rope_dims_;
  float rope_base_;
  float rope_scale_;
  int32_t offset_;
};

class KvCompactRowsCuda final : public Primitive {
 public:
  KvCompactRowsCuda(Stream stream, int32_t base_offset)
      : Primitive(stream), base_offset_(base_offset) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("KvCompactRowsCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    const array& k_cache = inputs[0];
    const array& v_cache = inputs[1];
    const array nodes =
        cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);

    const int B = k_cache.shape(0);
    const int Hkv = k_cache.shape(1);
    const int cap = k_cache.shape(2);
    const int D = k_cache.shape(3);
    const int keep = nodes.shape(0);
    const int itemsize = static_cast<int>(k_cache.itemsize());
    if (k_cache.ndim() != 4 || v_cache.ndim() != 4 ||
        k_cache.shape() != v_cache.shape() || k_cache.dtype() != v_cache.dtype() ||
        (itemsize != 1 && itemsize != 2 && itemsize != 4) ||
        nodes.ndim() != 1 || nodes.dtype() != int32 ||
        B < 1 || Hkv < 1 || D < 1 || keep < 0 || keep > 16 ||
        base_offset_ < 0 || int64_t(base_offset_) + keep > cap) {
      throw std::invalid_argument("KvCompactRowsCuda: invalid shape or bounds");
    }
    const void* kc_ptr = k_cache.buffer().ptr();
    const void* vc_ptr = v_cache.buffer().ptr();
    if (kc_ptr == vc_ptr ||
        kc_ptr == nodes.buffer().ptr() || vc_ptr == nodes.buffer().ptr()) {
      throw std::invalid_argument("KvCompactRowsCuda: cache buffer aliases input");
    }

    auto& k_out = outputs[0];
    auto& v_out = outputs[1];
    k_out.copy_shared_buffer(k_cache);
    v_out.copy_shared_buffer(v_cache);
    if (keep == 0) {
      return;
    }

    // Do NOT call encoder.set_input_array(k_cache/v_cache): k_out and v_out
    // alias them and set_output_array already tracks dependencies without
    // inflating bytes_in_graph_ by the preallocated KV buffers.
    encoder.set_input_array(nodes);
    encoder.set_output_array(k_out);
    encoder.set_output_array(v_out);

    const int row_bytes = D * itemsize;
    const int64_t kc_sb_b = k_cache.strides(0) * itemsize;
    const int64_t kc_sh_b = k_cache.strides(1) * itemsize;
    const int64_t kc_st_b = k_cache.strides(2) * itemsize;
    const int64_t vc_sb_b = v_cache.strides(0) * itemsize;
    const int64_t vc_sh_b = v_cache.strides(1) * itemsize;
    const int64_t vc_st_b = v_cache.strides(2) * itemsize;
    const int nvec = row_bytes >> 4;
    const bool use_vec16 =
        (row_bytes & 15) == 0 && nvec >= 1 && nvec <= 256 &&
        k_cache.strides(3) == 1 && v_cache.strides(3) == 1 &&
        ((reinterpret_cast<uintptr_t>(gpu_ptr<void>(k_out)) |
          reinterpret_cast<uintptr_t>(gpu_ptr<void>(v_out)) |
          static_cast<uintptr_t>(kc_sb_b) | static_cast<uintptr_t>(kc_sh_b) |
          static_cast<uintptr_t>(kc_st_b) | static_cast<uintptr_t>(vc_sb_b) |
          static_cast<uintptr_t>(vc_sh_b) | static_cast<uintptr_t>(vc_st_b)) & 15u) == 0;

    if (use_vec16) {
      auto kernel = cu::jit_kernel(
          encoder.device(), kQkNormRopeKvCudaSrc,
          "mlx_kv_compact_rows_vec16_cuda_kernel");
      cu::KernelArgs args;
      args.append(k_out);
      args.append(v_out);
      args.append(nodes);
      args.append<int32_t>(Hkv);
      args.append<int32_t>(cap);
      args.append<int32_t>(base_offset_);
      args.append<int32_t>(keep);
      args.append<int64_t>(kc_sb_b);
      args.append<int64_t>(kc_sh_b);
      args.append<int64_t>(kc_st_b);
      args.append<int64_t>(vc_sb_b);
      args.append<int64_t>(vc_sh_b);
      args.append<int64_t>(vc_st_b);
      encoder.add_kernel_node_raw(
          kernel, dim3(2 * Hkv, B, 1), dim3(nvec, 1, 1), {}, 0, args.args());
    } else {
      auto kernel = cu::jit_kernel(
          encoder.device(), kQkNormRopeKvCudaSrc,
          "mlx_kv_compact_rows_generic_cuda_kernel");
      cu::KernelArgs args;
      args.append(k_out);
      args.append(v_out);
      args.append(nodes);
      args.append<int32_t>(Hkv);
      args.append<int32_t>(cap);
      args.append<int32_t>(D);
      args.append<int32_t>(itemsize);
      args.append<int32_t>(base_offset_);
      args.append<int32_t>(keep);
      args.append<int64_t>(k_cache.strides(0));
      args.append<int64_t>(k_cache.strides(1));
      args.append<int64_t>(k_cache.strides(2));
      args.append<int64_t>(k_cache.strides(3));
      args.append<int64_t>(v_cache.strides(0));
      args.append<int64_t>(v_cache.strides(1));
      args.append<int64_t>(v_cache.strides(2));
      args.append<int64_t>(v_cache.strides(3));
      encoder.add_kernel_node_raw(
          kernel, dim3(2 * Hkv, B, 1), dim3(256, 1, 1), {}, 0, args.args());
    }
  }

  DEFINE_NAME(KvCompactRowsCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const KvCompactRowsCuda*>(&other);
    return rhs && rhs->base_offset_ == base_offset_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape(), inputs[1].shape()};
  }

 private:
  int32_t base_offset_;
};

class QkNormRopeCuda final : public Primitive {
 public:
  QkNormRopeCuda(
      Stream stream,
      float eps,
      float rope_base,
      float rope_scale,
      int32_t rope_dims)
      : Primitive(stream),
        eps_(eps),
        rope_base_(rope_base),
        rope_scale_(rope_scale),
        rope_dims_(rope_dims) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("QkNormRopeCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto ensure_last_contiguous = [&](const array& x) {
      if (x.strides(3) == 1) {
        return x;
      }
      auto copy = contiguous_copy_gpu(x, stream());
      encoder.add_temporary(copy);
      return copy;
    };

    const array q_in = ensure_last_contiguous(inputs[0]);
    const array k_in = ensure_last_contiguous(inputs[1]);
    const array q_w = cu::ensure_row_contiguous_aligned<8>(inputs[2], stream(), encoder);
    const array k_w = cu::ensure_row_contiguous_aligned<8>(inputs[3], stream(), encoder);
    const array offsets = cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);

    const int B = q_in.shape(0);
    const int T = q_in.shape(1);
    const int Hq = q_in.shape(2);
    const int D = q_in.shape(3);
    const int Hk = k_in.shape(2);
    const auto dt = q_in.dtype();
    const bool valid_offsets =
        offsets.dtype() == int32 &&
        ((offsets.ndim() == 1 && offsets.shape(0) == B) ||
         (offsets.ndim() == 0 && B == 1));
    if (B < 1 || T < 1 || Hq < 1 || Hk < 1 || D < 2 || D > 4096 || (D & 1) != 0 ||
        k_in.shape() != Shape{B, T, Hk, D} ||
        q_w.ndim() != 1 || q_w.shape(0) != D ||
        k_w.ndim() != 1 || k_w.shape(0) != D ||
        k_in.dtype() != dt || q_w.dtype() != dt || k_w.dtype() != dt ||
        (dt != bfloat16 && dt != float16 && dt != float32) ||
        !valid_offsets ||
        rope_dims_ < 0 || rope_dims_ > D || (rope_dims_ & 1) != 0) {
      throw std::invalid_argument("QkNormRopeCuda: invalid shape or dtype");
    }

    auto& q_out = outputs[0];
    auto& k_out = outputs[1];
    q_out.set_data(cu::malloc_async(q_out.nbytes(), encoder));
    k_out.set_data(cu::malloc_async(k_out.nbytes(), encoder));

    encoder.set_input_array(q_in);
    encoder.set_input_array(k_in);
    encoder.set_input_array(q_w);
    encoder.set_input_array(k_w);
    encoder.set_input_array(offsets);
    encoder.set_output_array(q_out);
    encoder.set_output_array(k_out);

    const int dt_idx = (dt == bfloat16) ? 0 : (dt == float16 ? 1 : 2);
    const int d_idx = (D == 128) ? 0 : (D == 256 ? 1 : 2);
    static constexpr const char* kEntries[3][3] = {
        {"mlx_qk_norm_rope_cuda_kernel<nv_bfloat16,128>",
         "mlx_qk_norm_rope_cuda_kernel<nv_bfloat16,256>",
         "mlx_qk_norm_rope_cuda_kernel<nv_bfloat16,0>"},
        {"mlx_qk_norm_rope_cuda_kernel<__half,128>",
         "mlx_qk_norm_rope_cuda_kernel<__half,256>",
         "mlx_qk_norm_rope_cuda_kernel<__half,0>"},
        {"mlx_qk_norm_rope_cuda_kernel<float,128>",
         "mlx_qk_norm_rope_cuda_kernel<float,256>",
         "mlx_qk_norm_rope_cuda_kernel<float,0>"},
    };
    auto kernel = cu::jit_kernel(
        encoder.device(), kQkNormRopeKvCudaSrc, kEntries[dt_idx][d_idx]);

    cu::KernelArgs args;
    args.append(q_in);
    args.append(k_in);
    args.append(q_w);
    args.append(k_w);
    args.append(offsets);
    args.append(q_out);
    args.append(k_out);
    args.append<int32_t>(T);
    args.append<int32_t>(Hq);
    args.append<int32_t>(Hk);
    args.append<int32_t>(D);
    args.append<int32_t>(rope_dims_);
    args.append<float>(std::log2(rope_base_));
    args.append<float>(rope_scale_);
    args.append<float>(eps_);
    args.append<int64_t>(q_in.strides(0));
    args.append<int64_t>(q_in.strides(1));
    args.append<int64_t>(q_in.strides(2));
    args.append<int64_t>(k_in.strides(0));
    args.append<int64_t>(k_in.strides(1));
    args.append<int64_t>(k_in.strides(2));

    encoder.add_kernel_node_raw(
        kernel, dim3(T, Hq + Hk, B), dim3(128, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(QkNormRopeCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const QkNormRopeCuda*>(&other);
    return rhs && rhs->eps_ == eps_ && rhs->rope_base_ == rope_base_ &&
        rhs->rope_scale_ == rope_scale_ && rhs->rope_dims_ == rope_dims_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    return {
        {q.shape(0), q.shape(2), q.shape(1), q.shape(3)},
        {k.shape(0), k.shape(2), k.shape(1), k.shape(3)},
    };
  }

 private:
  float eps_;
  float rope_base_;
  float rope_scale_;
  int32_t rope_dims_;
};

class KvStoreRowsCuda final : public Primitive {
 public:
  KvStoreRowsCuda(Stream stream, std::vector<int> offsets)
      : Primitive(stream), offsets_(std::move(offsets)) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("KvStoreRowsCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    const int n = static_cast<int>(offsets_.size());
    auto& encoder = cu::get_command_encoder(stream());
    auto kernel = cu::jit_kernel(
        encoder.device(), kKvStoreRowsCudaSrc, "mlx_node_kv_store_rows_cuda");
    auto vec16_kernel = cu::jit_kernel(
        encoder.device(), kKvStoreRowsCudaSrc, "mlx_node_kv_store_rows_vec16_cuda");
    for (int i = 0; i < n; ++i) {
      const auto& d = inputs[i];
      const auto& s = inputs[n + i];
      auto& o = outputs[i];
      const int nd = s.ndim();
      const int B = s.shape(0);
      const int H = s.shape(1);
      const int rows = s.shape(2);
      const int D = nd == 4 ? s.shape(3) : 1;
      const int itemsize = static_cast<int>(s.itemsize());
      if (d.ndim() != nd || (nd != 3 && nd != 4) || d.dtype() != s.dtype() ||
          (itemsize != 1 && itemsize != 2 && itemsize != 4) ||
          d.shape(0) != B || d.shape(1) != H ||
          (nd == 4 && d.shape(3) != D) ||
          offsets_[i] < 0 || int64_t(offsets_[i]) + rows > d.shape(2)) {
        throw std::invalid_argument("KvStoreRowsCuda: replay shape or bounds mismatch");
      }
      o.copy_shared_buffer(d);
      const int64_t total = static_cast<int64_t>(B) * H * rows * D;
      if (total <= 0) {
        continue;
      }
      // Do NOT call encoder.set_input_array(d): `o` aliases `d` and
      // set_output_array(o) already tracks buffer dependency without
      // inflating bytes_in_graph_ by the entire preallocated KV buffer.
      encoder.set_input_array(s);
      encoder.set_output_array(o);
      const int64_t ds0_b = d.strides(0) * itemsize;
      const int64_t ds1_b = d.strides(1) * itemsize;
      const int64_t ds2_b = d.strides(2) * itemsize;
      const int64_t ss0_b = s.strides(0) * itemsize;
      const int64_t ss1_b = s.strides(1) * itemsize;
      const int64_t ss2_b = s.strides(2) * itemsize;
      const bool use_vec16 =
          nd == 4 && itemsize == 2 && (D & 7) == 0 && (D >> 3) <= 1024 &&
          d.strides(3) == 1 && s.strides(3) == 1 &&
          ((reinterpret_cast<uintptr_t>(gpu_ptr<void>(o)) |
            reinterpret_cast<uintptr_t>(gpu_ptr<void>(s)) |
            static_cast<uintptr_t>(ds0_b) | static_cast<uintptr_t>(ds1_b) |
            static_cast<uintptr_t>(ds2_b) | static_cast<uintptr_t>(ss0_b) |
            static_cast<uintptr_t>(ss1_b) | static_cast<uintptr_t>(ss2_b)) & 15u) == 0;
      if (use_vec16) {
        cu::KernelArgs args;
        args.append(o);
        args.append(s);
        args.append<int32_t>(offsets_[i]);
        args.append<int64_t>(ds0_b);
        args.append<int64_t>(ds1_b);
        args.append<int64_t>(ds2_b);
        args.append<int64_t>(ss0_b);
        args.append<int64_t>(ss1_b);
        args.append<int64_t>(ss2_b);
        encoder.add_kernel_node_raw(
            vec16_kernel, dim3(rows, H, B), dim3(D >> 3, 1, 1), {}, 0,
            args.args());
      } else {
        cu::KernelArgs args;
        args.append(o);
        args.append(s);
        args.append<int32_t>(B);
        args.append<int32_t>(H);
        args.append<int32_t>(rows);
        args.append<int32_t>(D);
        args.append<int32_t>(itemsize);
        args.append<int32_t>(offsets_[i]);
        args.append<int64_t>(d.strides(0));
        args.append<int64_t>(d.strides(1));
        args.append<int64_t>(d.strides(2));
        args.append<int64_t>(nd == 4 ? d.strides(3) : 0);
        args.append<int64_t>(s.strides(0));
        args.append<int64_t>(s.strides(1));
        args.append<int64_t>(s.strides(2));
        args.append<int64_t>(nd == 4 ? s.strides(3) : 0);
        const uint32_t grid = static_cast<uint32_t>((total + 255) / 256);
        encoder.add_kernel_node_raw(
            kernel, dim3(grid), dim3(256), {}, 0, args.args());
      }
    }
  }

  DEFINE_NAME(KvStoreRowsCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const KvStoreRowsCuda*>(&other);
    return rhs && rhs->offsets_ == offsets_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int n = static_cast<int>(offsets_.size());
    std::vector<Shape> shapes;
    shapes.reserve(n);
    for (int i = 0; i < n; ++i) {
      shapes.push_back(inputs[i].shape());
    }
    return shapes;
  }

 private:
  std::vector<int> offsets_;
};
#endif

}  // namespace

namespace mlx_node_internal {

bool kv_store_rows_cuda(int32_t tensors, mlx_array* const* dst,
                        mlx_array* const* src, const int32_t* offsets,
                        mlx_array** out) {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!dst || !src || !out || !offsets || tensors < 1 || tensors > kMaxTensors ||
        default_device() != Device::gpu) {
      return false;
    }
    std::vector<array> inputs;
    std::vector<Shape> shapes;
    std::vector<Dtype> dtypes;
    inputs.reserve(tensors * 2);
    shapes.reserve(tensors);
    dtypes.reserve(tensors);
    for (int i = 0; i < tensors; ++i) {
      if (!dst[i] || !src[i] || offsets[i] < 0) {
        return false;
      }
      const auto& d = *reinterpret_cast<array*>(dst[i]);
      const auto& s = *reinterpret_cast<array*>(src[i]);
      const int itemsize = static_cast<int>(s.itemsize());
      if (d.ndim() != s.ndim() || (d.ndim() != 3 && d.ndim() != 4) ||
          d.dtype() != s.dtype() ||
          (itemsize != 1 && itemsize != 2 && itemsize != 4) ||
          d.shape(0) != s.shape(0) || d.shape(1) != s.shape(1) ||
          (d.ndim() == 4 && d.shape(3) != s.shape(3)) ||
          int64_t(offsets[i]) + s.shape(2) > d.shape(2)) {
        return false;
      }
      inputs.push_back(d);
      shapes.push_back(d.shape());
      dtypes.push_back(d.dtype());
    }
    for (int i = 0; i < tensors; ++i) {
      inputs.push_back(*reinterpret_cast<array*>(src[i]));
    }
    auto stream = default_stream(Device::gpu);
    auto primitive = std::make_shared<KvStoreRowsCuda>(
        stream, std::vector<int>(offsets, offsets + tensors));
    auto outputs = array::make_arrays(std::move(shapes), std::move(dtypes),
                                      primitive, std::move(inputs));
    for (int i = 0; i < tensors; ++i) {
      out[i] = reinterpret_cast<mlx_array*>(new array(std::move(outputs[i])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_kv_store_rows (cuda): %s\n", e.what());
    return false;
  }
#else
  (void)tensors;
  (void)dst;
  (void)src;
  (void)offsets;
  (void)out;
  return false;
#endif
}

}  // namespace mlx_node_internal

// Fused CUDA QK-Norm + non-traditional partial RoPE + in-place KV cache store.
// Replaces 6 launches (q_norm, k_norm, rope(q), rope(k), kv_store(K), kv_store(V))
// with 1 kernel launch:
//   q_in:     [B, T, Hq, D] bf16 (contiguous or strided, stride(-1) == 1)
//   k_in:     [B, T, Hkv, D] bf16
//   v_in:     [B, T, Hkv, D] bf16
//   q_norm_w: [D] bf16
//   k_norm_w: [D] bf16
//   k_cache:  [B, Hkv, cap, D] bf16 (mutated in-place at offset..offset+T)
//   v_cache:  [B, Hkv, cap, D] bf16 (mutated in-place at offset..offset+T)
// Outputs:
//   out_q:       [B, Hq, T, D] bf16 (transposed for SDPA)
//   out_k_cache: [B, Hkv, cap, D] bf16 (aliases k_cache buffer)
//   out_v_cache: [B, Hkv, cap, D] bf16 (aliases v_cache buffer)
extern "C" bool mlx_qk_norm_rope_kv_store_cuda(
    mlx_array* q_in,
    mlx_array* k_in,
    mlx_array* v_in,
    mlx_array* q_norm_w,
    mlx_array* k_norm_w,
    mlx_array* k_cache,
    mlx_array* v_cache,
    float eps,
    int32_t rope_dims,
    float rope_base,
    float rope_scale,
    int32_t offset,
    mlx_array** out_q,
    mlx_array** out_k_cache,
    mlx_array** out_v_cache) {
  if (out_q) *out_q = nullptr;
  if (out_k_cache) *out_k_cache = nullptr;
  if (out_v_cache) *out_v_cache = nullptr;
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!q_in || !k_in || !v_in || !q_norm_w || !k_norm_w || !k_cache || !v_cache ||
        !out_q || !out_k_cache || !out_v_cache || default_device() != Device::gpu) {
      return false;
    }
    const auto& qa = *reinterpret_cast<array*>(q_in);
    const auto& ka = *reinterpret_cast<array*>(k_in);
    const auto& va = *reinterpret_cast<array*>(v_in);
    const auto& qw = *reinterpret_cast<array*>(q_norm_w);
    const auto& kw = *reinterpret_cast<array*>(k_norm_w);
    const auto& kc = *reinterpret_cast<array*>(k_cache);
    const auto& vc = *reinterpret_cast<array*>(v_cache);
    if (qa.dtype() != bfloat16 || ka.dtype() != bfloat16 || va.dtype() != bfloat16 ||
        qw.dtype() != bfloat16 || kw.dtype() != bfloat16 ||
        kc.dtype() != bfloat16 || vc.dtype() != bfloat16) {
      return false;
    }
    if (qa.ndim() != 4 || ka.ndim() != 4 || va.ndim() != 4 ||
        qw.ndim() != 1 || kw.ndim() != 1 ||
        kc.ndim() != 4 || vc.ndim() != 4) {
      return false;
    }
    const int B = qa.shape(0);
    const int T = qa.shape(1);
    const int Hq = qa.shape(2);
    const int D = qa.shape(3);
    const int Hkv = ka.shape(2);
    const int cap = kc.shape(2);
    if (B < 1 || T < 1 || Hq < 1 || Hkv < 1 || D < 2 || D > 1024 || (D & 1) != 0 ||
        ka.shape() != Shape{B, T, Hkv, D} || va.shape() != Shape{B, T, Hkv, D} ||
        qw.shape() != Shape{D} || kw.shape() != Shape{D} ||
        kc.shape() != Shape{B, Hkv, cap, D} || vc.shape() != Shape{B, Hkv, cap, D} ||
        kc.strides(3) != 1 || vc.strides(3) != 1 ||
        offset < 0 || int64_t(offset) + T > cap ||
        rope_dims < 0 || rope_dims > D || (rope_dims & 1) != 0 ||
        !(rope_base > 0.0f)) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    auto outputs = array::make_arrays(
        {Shape{B, Hq, T, D}, kc.shape(), vc.shape()},
        {bfloat16, bfloat16, bfloat16},
        std::make_shared<QkNormRopeKvStoreCuda>(
            stream, eps, rope_dims, rope_base, rope_scale, offset),
        {qa, ka, va, qw, kw, kc, vc});
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_k_cache = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    *out_v_cache = reinterpret_cast<mlx_array*>(new array(std::move(outputs[2])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qk_norm_rope_kv_store_cuda: %s\n", e.what());
    return false;
  }
#else
  (void)q_in; (void)k_in; (void)v_in; (void)q_norm_w; (void)k_norm_w;
  (void)k_cache; (void)v_cache; (void)eps; (void)rope_dims;
  (void)rope_base; (void)rope_scale; (void)offset;
  return false;
#endif
}

// Tree-RoPE variant of fused CUDA QK-Norm + non-traditional partial RoPE +
// in-place KV cache store. RoPE position for token t is `offset + depths[t]`,
// while KV store still writes at slot `offset + t`.
extern "C" bool mlx_qk_norm_rope_kv_store_tree_cuda(
    mlx_array* q_in,
    mlx_array* k_in,
    mlx_array* v_in,
    mlx_array* q_norm_w,
    mlx_array* k_norm_w,
    mlx_array* k_cache,
    mlx_array* v_cache,
    mlx_array* depths,
    float eps,
    int32_t rope_dims,
    float rope_base,
    float rope_scale,
    int32_t offset,
    mlx_array** out_q,
    mlx_array** out_k_cache,
    mlx_array** out_v_cache) {
  if (out_q) *out_q = nullptr;
  if (out_k_cache) *out_k_cache = nullptr;
  if (out_v_cache) *out_v_cache = nullptr;
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!q_in || !k_in || !v_in || !q_norm_w || !k_norm_w || !k_cache || !v_cache ||
        !depths || !out_q || !out_k_cache || !out_v_cache ||
        default_device() != Device::gpu) {
      return false;
    }
    const auto& qa = *reinterpret_cast<array*>(q_in);
    const auto& ka = *reinterpret_cast<array*>(k_in);
    const auto& va = *reinterpret_cast<array*>(v_in);
    const auto& qw = *reinterpret_cast<array*>(q_norm_w);
    const auto& kw = *reinterpret_cast<array*>(k_norm_w);
    const auto& kc = *reinterpret_cast<array*>(k_cache);
    const auto& vc = *reinterpret_cast<array*>(v_cache);
    const auto& da = *reinterpret_cast<array*>(depths);
    if (qa.dtype() != bfloat16 || ka.dtype() != bfloat16 || va.dtype() != bfloat16 ||
        qw.dtype() != bfloat16 || kw.dtype() != bfloat16 ||
        kc.dtype() != bfloat16 || vc.dtype() != bfloat16 ||
        da.dtype() != int32) {
      return false;
    }
    if (qa.ndim() != 4 || ka.ndim() != 4 || va.ndim() != 4 ||
        qw.ndim() != 1 || kw.ndim() != 1 ||
        kc.ndim() != 4 || vc.ndim() != 4 || da.ndim() != 1) {
      return false;
    }
    const int B = qa.shape(0);
    const int T = qa.shape(1);
    const int Hq = qa.shape(2);
    const int D = qa.shape(3);
    const int Hkv = ka.shape(2);
    const int cap = kc.shape(2);
    if (B < 1 || T < 1 || Hq < 1 || Hkv < 1 || D < 2 || D > 1024 || (D & 1) != 0 ||
        ka.shape() != Shape{B, T, Hkv, D} || va.shape() != Shape{B, T, Hkv, D} ||
        qw.shape() != Shape{D} || kw.shape() != Shape{D} ||
        kc.shape() != Shape{B, Hkv, cap, D} || vc.shape() != Shape{B, Hkv, cap, D} ||
        kc.strides(3) != 1 || vc.strides(3) != 1 ||
        da.shape(0) != T ||
        offset < 0 || int64_t(offset) + T > cap ||
        rope_dims < 0 || rope_dims > D || (rope_dims & 1) != 0 ||
        !(rope_base > 0.0f)) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    auto outputs = array::make_arrays(
        {Shape{B, Hq, T, D}, kc.shape(), vc.shape()},
        {bfloat16, bfloat16, bfloat16},
        std::make_shared<QkNormRopeKvStoreTreeCuda>(
            stream, eps, rope_dims, rope_base, rope_scale, offset),
        {qa, ka, va, qw, kw, kc, vc, da});
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_k_cache = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    *out_v_cache = reinterpret_cast<mlx_array*>(new array(std::move(outputs[2])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qk_norm_rope_kv_store_tree_cuda: %s\n", e.what());
    return false;
  }
#else
  (void)q_in; (void)k_in; (void)v_in; (void)q_norm_w; (void)k_norm_w;
  (void)k_cache; (void)v_cache; (void)depths; (void)eps; (void)rope_dims;
  (void)rope_base; (void)rope_scale; (void)offset;
  return false;
#endif
}

// Race-free in-place KV cache row compaction for tree acceptance (keep <= 16).
// Copies cache[b, h, base_offset + accepted_nodes[r], :] ->
// cache[b, h, base_offset + r, :] for both K and V, returning aliased handles.
extern "C" bool mlx_kv_compact_rows_cuda(
    mlx_array* k_cache,
    mlx_array* v_cache,
    int32_t base_offset,
    mlx_array* accepted_nodes,
    mlx_array** out_k,
    mlx_array** out_v) {
  if (out_k) *out_k = nullptr;
  if (out_v) *out_v = nullptr;
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!k_cache || !v_cache || !accepted_nodes || !out_k || !out_v ||
        default_device() != Device::gpu) {
      return false;
    }
    const auto& kc = *reinterpret_cast<array*>(k_cache);
    const auto& vc = *reinterpret_cast<array*>(v_cache);
    const auto& nodes = *reinterpret_cast<array*>(accepted_nodes);
    const int itemsize = static_cast<int>(kc.itemsize());
    if (kc.ndim() != 4 || vc.ndim() != 4 ||
        kc.shape() != vc.shape() || kc.dtype() != vc.dtype() ||
        (itemsize != 1 && itemsize != 2 && itemsize != 4) ||
        nodes.ndim() != 1 || nodes.dtype() != int32) {
      return false;
    }
    const int B = kc.shape(0);
    const int Hkv = kc.shape(1);
    const int cap = kc.shape(2);
    const int D = kc.shape(3);
    const int keep = nodes.shape(0);
    if (B < 1 || Hkv < 1 || D < 1 || keep < 0 || keep > 16 ||
        base_offset < 0 || int64_t(base_offset) + keep > cap) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    auto outputs = array::make_arrays(
        {kc.shape(), vc.shape()},
        {kc.dtype(), vc.dtype()},
        std::make_shared<KvCompactRowsCuda>(stream, base_offset),
        {kc, vc, nodes});
    *out_k = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_v = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_kv_compact_rows_cuda: %s\n", e.what());
    return false;
  }
#else
  (void)k_cache; (void)v_cache; (void)base_offset; (void)accepted_nodes;
  return false;
#endif
}

// Standalone fused CUDA QK-Norm + Non-Traditional (half-split) RoPE, matching
// Metal's `mlx_qk_norm_rope` semantics:
//   q: [B, T, HQ, D], k: [B, T, HK, D] -> out_q: [B, HQ, T, D], out_k: [B, HK, T, D]
//   q_weight / k_weight: [D] plain RMSNorm weights (not 1 + w)
//   offsets: int32 [B] device array (or scalar for B == 1)
//   Supports bfloat16, float16, float32; fast paths for D = 128 and D = 256.
extern "C" bool mlx_qk_norm_rope_cuda(
    mlx_array* q,
    mlx_array* k,
    mlx_array* q_weight,
    mlx_array* k_weight,
    mlx_array* offsets,
    float eps,
    float rope_base,
    float rope_scale,
    int32_t rope_dims,
    mlx_array** out_q,
    mlx_array** out_k) {
  if (out_q) *out_q = nullptr;
  if (out_k) *out_k = nullptr;
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!q || !k || !q_weight || !k_weight || !offsets || !out_q || !out_k ||
        default_device() != Device::gpu) {
      return false;
    }
    const auto& qa = *reinterpret_cast<array*>(q);
    const auto& ka = *reinterpret_cast<array*>(k);
    const auto& qw = *reinterpret_cast<array*>(q_weight);
    const auto& kw = *reinterpret_cast<array*>(k_weight);
    const auto& oa = *reinterpret_cast<array*>(offsets);
    if (qa.ndim() != 4 || ka.ndim() != 4 || qw.ndim() != 1 || kw.ndim() != 1) {
      return false;
    }
    const int B = qa.shape(0);
    const int T = qa.shape(1);
    const int Hq = qa.shape(2);
    const int D = qa.shape(3);
    const int Hk = ka.shape(2);
    const auto dt = qa.dtype();
    const bool valid_offsets =
        oa.dtype() == int32 &&
        ((oa.ndim() == 1 && oa.shape(0) == B) || (oa.ndim() == 0 && B == 1));
    if (B < 1 || T < 1 || Hq < 1 || Hk < 1 || D < 2 || D > 4096 || (D & 1) != 0 ||
        ka.shape() != Shape{B, T, Hk, D} ||
        qw.shape(0) != D || kw.shape(0) != D ||
        (dt != bfloat16 && dt != float16 && dt != float32) ||
        ka.dtype() != dt || qw.dtype() != dt || kw.dtype() != dt ||
        !valid_offsets ||
        rope_dims < 0 || rope_dims > D || (rope_dims & 1) != 0 ||
        !(rope_base > 0.0f)) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    auto outputs = array::make_arrays(
        {Shape{B, Hq, T, D}, Shape{B, Hk, T, D}},
        {dt, dt},
        std::make_shared<QkNormRopeCuda>(stream, eps, rope_base, rope_scale, rope_dims),
        {qa, ka, qw, kw, oa});
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_k = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qk_norm_rope_cuda: %s\n", e.what());
    return false;
  }
#else
  (void)q; (void)k; (void)q_weight; (void)k_weight; (void)offsets;
  (void)eps; (void)rope_base; (void)rope_scale; (void)rope_dims;
  return false;
#endif
}
