// In-place KV row store for the flat Qwen3.5 cache: one Metal dispatch
// writes verify-block K and V rows (plus int8 scales) into the cache buffers
// at their write offsets (Splash `verify_attention_*_store`, Apache-2.0; see
// THIRD_PARTY_NOTICES). The outputs alias the cache inputs' buffers — the
// caller owns those buffers and replaces its handles with the outputs — so
// the slice_update copies become one dispatch. Up to four tensors bind
// directly (one layer); more go through a GPU-address table so every
// full-attention layer of a verify block is stored by ONE dispatch.

#include "mlx_common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"
#include "mlx_test_counters.h"
#elif defined(MLX_NODE_CUDA_ENABLED)
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"
#endif

namespace {

constexpr int kMaxBoundTensors = 4;
constexpr int kMaxTensors = 256;

#ifdef MLX_NODE_METAL_ENABLED
using namespace mlx::core;

const char* kKvStoreBody =
#include "metal/common/kv_store_rows.metal.inc"
    ;

// Mirrors `KvStoreParams` in the kernel source.
struct KvStoreParams {
  int tensors;
  int rows;
  int offset;
  int batch;
  int heads;
  int dims[kMaxBoundTensors];
  int itemsize[kMaxBoundTensors];
  uint32_t elems[kMaxBoundTensors];
  int64_t src_strides[kMaxBoundTensors][4];
  int64_t dst_strides[kMaxBoundTensors][4];
};

// Mirrors `KvStoreTensor` in the kernel source.
struct KvStoreTensor {
  uint64_t dst;
  uint64_t src;
  int64_t src_strides[4];
  int64_t dst_strides[4];
  uint32_t elems;
  int32_t itemsize;
  int32_t dims;
  int32_t rows;
  int32_t heads;
  int32_t offset;
};
static_assert(sizeof(KvStoreTensor) == 104, "KvStoreTensor layout");

MTL::ComputePipelineState* kv_store_pipeline(metal::Device& device,
                                             bool table) {
  auto* library = device.get_library("mlx_node_kv_store_rows", [] {
    return std::string(metal::utils()) + kKvStoreBody;
  });
  return device.get_kernel(
      table ? "mlx_node_kv_store_rows_table" : "mlx_node_kv_store_rows",
      library);
}

uint64_t gpu_address(const array& a) {
  return static_cast<const MTL::Buffer*>(a.buffer().ptr())->gpuAddress() +
      a.offset();
}

// Inputs: n cache buffers, then the n row blocks. Outputs: the n caches,
// aliasing the inputs' buffers.
class KvStoreRows final : public fast::Custom {
 public:
  KvStoreRows(Stream stream, std::vector<int> offsets, bool bound)
      : Custom(stream,
               [](std::vector<array>) -> std::vector<array> {
                 throw std::runtime_error("KvStoreRows has no fallback");
               }),
        offsets_(std::move(offsets)), bound_(bound) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("KvStoreRows requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    bridge_testing::record("kv_store_rows");
    const int tensors = int(offsets_.size());
    std::vector<KvStoreTensor> table(tensors);
    uint32_t max_elems = 0;
    for (int i = 0; i < tensors; ++i) {
      const auto& dst = inputs[i];
      const auto& src = inputs[tensors + i];
      auto& out = outputs[i];
      out.copy_shared_buffer(dst);
      // Scales are [B, H, T]; treat them as D = 1.
      const int nd = src.ndim();
      const int d = nd == 4 ? src.shape(3) : 1;
      if (dst.shape(0) != src.shape(0) || dst.shape(1) != src.shape(1) ||
          dst.ndim() != nd || (nd == 4 && dst.shape(3) != d) ||
          offsets_[i] + src.shape(2) > dst.shape(2) ||
          dst.itemsize() != src.itemsize()) {
        throw std::runtime_error("KvStoreRows: block does not fit the cache");
      }
      auto& t = table[i];
      t.dst = gpu_address(out);
      t.src = gpu_address(src);
      t.elems = uint32_t(src.size());
      t.itemsize = int32_t(src.itemsize());
      t.dims = d;
      t.rows = src.shape(2);
      t.heads = src.shape(1);
      t.offset = offsets_[i];
      for (int a = 0; a < 3; ++a) {
        t.src_strides[a] = src.strides(a);
        t.dst_strides[a] = dst.strides(a);
      }
      t.src_strides[3] = nd == 4 ? src.strides(3) : 0;
      t.dst_strides[3] = nd == 4 ? dst.strides(3) : 0;
      max_elems = std::max(max_elems, t.elems);
    }
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    auto& encoder = metal::get_command_encoder(stream);
    if (bound_) {
      KvStoreParams p{};
      p.tensors = tensors;
      p.offset = offsets_[0];
      p.rows = table[0].rows;
      p.batch = inputs[tensors].shape(0);
      p.heads = table[0].heads;
      for (int i = 0; i < tensors; ++i) {
        p.dims[i] = table[i].dims;
        p.itemsize[i] = table[i].itemsize;
        p.elems[i] = table[i].elems;
        for (int a = 0; a < 4; ++a) {
          p.src_strides[i][a] = table[i].src_strides[a];
          p.dst_strides[i][a] = table[i].dst_strides[a];
        }
      }
      encoder.set_compute_pipeline_state(kv_store_pipeline(device, false));
      for (int i = 0; i < kMaxBoundTensors; ++i) {
        const int j = std::min(i, tensors - 1);
        encoder.set_output_array(outputs[j], i);
        encoder.set_input_array(inputs[tensors + j], kMaxBoundTensors + i);
      }
      encoder.set_bytes(p, 2 * kMaxBoundTensors);
    } else {
      // The kernel reads every buffer through the table; bind each array
      // on a slot the kernel ignores so the encoder still tracks residency,
      // barriers and cross-command fences for it.
      constexpr int kTrackSlot = 1;
      array bytes(Shape{int(sizeof(KvStoreTensor)) * tensors}, uint8,
                  nullptr, {});
      bytes.set_data(allocator::malloc(bytes.nbytes()));
      std::memcpy(bytes.data<uint8_t>(), table.data(), bytes.nbytes());
      encoder.add_temporary(bytes);
      encoder.set_compute_pipeline_state(kv_store_pipeline(device, true));
      encoder.set_input_array(bytes, 0);
      for (int i = 0; i < tensors; ++i) {
        encoder.set_output_array(outputs[i], kTrackSlot);
        encoder.set_input_array(inputs[tensors + i], kTrackSlot);
      }
    }
    encoder.dispatch_threads(MTL::Size(max_elems, tensors, 1),
                             MTL::Size(256, 1, 1));
  }

  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&,
                         const std::vector<array>&) override {
    throw std::runtime_error("KvStoreRows is inference-only");
  }

  DEFINE_NAME(KvStoreRows)

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const KvStoreRows&>(other);
    return offsets_ == o.offsets_ && bound_ == o.bound_;
  }

 private:
  std::vector<int> offsets_;
  bool bound_;
};

// A draft proposal writes into the reserved tail of each live window. The
// window views are explicit graph inputs, so both their physical offset and
// live length change on a shapeless replay without baking host offsets into
// the tape. Return full buffers as well as views to preserve the dependency
// from the proposal store to the next accepted-context append.
class KvStoreWindowTail final : public fast::Custom {
 public:
  explicit KvStoreWindowTail(Stream stream)
      : Custom(stream, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("KvStoreWindowTail has no fallback");
        }) {}
  DEFINE_NAME(KvStoreWindowTail)
  bool is_equivalent(const Primitive&) const override { return true; }
  std::vector<Shape> output_shapes(const std::vector<array>& in) override {
    return {in[0].shape(), in[1].shape(), in[2].shape(), in[3].shape()};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("KvStoreWindowTail requires Metal");
  }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    // Validate every destination before recording any write. A malformed
    // replay fails closed rather than partially overwriting a prefix.
    for (int i = 0; i < 2; ++i) {
      const auto& full = in[i];
      const auto& view = in[2 + i];
      const auto& block = in[4 + i];
      if (full.ndim() != 4 || view.ndim() != 4 || block.ndim() != 4) {
        throw std::invalid_argument("KvStoreWindowTail: replay rank changed");
      }
      const auto row_bytes = size_t(full.shape(3)) * full.itemsize();
      if (full.dtype() != bfloat16 || view.dtype() != bfloat16 ||
          block.dtype() != bfloat16 || full.shape(0) != 1 ||
          !full.flags().row_contiguous || full.shape(0) != view.shape(0) ||
          full.shape(1) != view.shape(1) || full.shape(3) != view.shape(3) ||
          full.shape(0) != block.shape(0) || full.shape(1) != block.shape(1) ||
          full.shape(3) != block.shape(3) || block.shape(2) < 1 ||
          block.shape(2) > view.shape(2) ||
          full.buffer().ptr() != view.buffer().ptr() ||
          view.strides() != full.strides() || view.offset() < full.offset() ||
          row_bytes == 0 || (view.offset() - full.offset()) % row_bytes != 0 ||
          (view.offset() - full.offset()) / row_bytes + view.shape(2) >
              size_t(full.shape(2)) ||
          full.buffer().ptr() == block.buffer().ptr()) {
        throw std::invalid_argument("KvStoreWindowTail: invalid window alias or block");
      }
    }
    // K and V copies share a dispatch; neither source may alias the other destination.
    if (in[0].buffer().ptr() == in[1].buffer().ptr() ||
        in[0].buffer().ptr() == in[5].buffer().ptr() ||
        in[1].buffer().ptr() == in[4].buffer().ptr() ||
        in[2].shape() != in[3].shape() || in[4].shape() != in[5].shape()) {
      throw std::invalid_argument("KvStoreWindowTail: K/V geometry or ownership differs");
    }
    for (int i = 0; i < 4; ++i) out[i].copy_shared_buffer(in[i]);
    KvStoreParams p{};
    p.tensors = 2;
    p.rows = in[4].shape(2);
    p.offset = in[2].shape(2) - p.rows;
    p.batch = 1;
    p.heads = in[4].shape(1);
    uint32_t max_elems = 0;
    for (int i = 0; i < 2; ++i) {
      p.dims[i] = in[4 + i].shape(3);
      p.itemsize[i] = in[4 + i].itemsize();
      p.elems[i] = in[4 + i].size();
      max_elems = std::max(max_elems, p.elems[i]);
      for (int a = 0; a < 4; ++a) {
        p.src_strides[i][a] = in[4 + i].strides(a);
        p.dst_strides[i][a] = in[2 + i].strides(a);
      }
    }
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kv_store_pipeline(metal::device(stream().device), false));
    for (int i = 0; i < kMaxBoundTensors; ++i) {
      const int j = i % 2;
      encoder.set_output_array(out[2 + j], i);
      encoder.set_input_array(in[4 + j], kMaxBoundTensors + i);
    }
    encoder.set_bytes(p, 2 * kMaxBoundTensors);
    // Full aliases are returned too; make their write dependency visible to
    // command-buffer hazard tracking even though the kernel addresses views.
    encoder.set_output_array(out[0], 9);
    encoder.set_output_array(out[1], 9);
    encoder.dispatch_threads(MTL::Size(max_elems, 2, 1), MTL::Size(256, 1, 1));
  }
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override {
    throw std::runtime_error("KvStoreWindowTail is inference-only");
  }
};

// The pooled index keys of complete blocks into a page pool's index plane
// (Splash `block_indexer_pool` through `indexer_index_keys`): row r goes to
// plane row `slots[r]` of the `[blocks * index_rows, dim]` bf16 plane, in
// place. Inputs: plane, rows `[n, dim]`, int32 slots `[n]`. The output
// aliases the plane.
class PagedIndexStore final : public fast::Custom {
 public:
  explicit PagedIndexStore(Stream stream)
      : Custom(stream, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("PagedIndexStore has no fallback");
        }) {}
  DEFINE_NAME(PagedIndexStore)
  bool is_equivalent(const Primitive&) const override { return true; }
  std::vector<Shape> output_shapes(const std::vector<array>& in) override {
    return {in[0].shape()};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("PagedIndexStore requires Metal");
  }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    const auto& plane = in[0];
    const auto& rows = in[1];
    const auto& slots = in[2];
    const int64_t capacity = plane.size() / rows.shape(1);
    const int32_t* ids = slots.data<int32_t>();
    for (size_t i = 0; i < slots.size(); ++i) {
      if (ids[i] < 0 || ids[i] >= capacity) {
        throw std::invalid_argument(
            "PagedIndexStore: an index slot lies outside the plane");
      }
    }
    out[0].copy_shared_buffer(plane);
    auto& device = metal::device(stream().device);
    auto* library = device.get_library("mlx_node_paged_index_store", [] {
      return std::string(R"(
#include <metal_stdlib>
using namespace metal;
[[kernel]] void mlx_node_paged_index_store(
    device ushort* plane [[buffer(0)]],
    const device ushort* rows [[buffer(1)]],
    const device int* slots [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= count * dim) return;
  plane[ulong(slots[i / dim]) * dim + i % dim] = rows[i];
}
)");
    });
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(
        device.get_kernel("mlx_node_paged_index_store", library));
    encoder.set_output_array(out[0], 0);
    encoder.set_input_array(rows, 1);
    encoder.set_input_array(slots, 2);
    const uint32_t dim = rows.shape(1);
    const uint32_t count = rows.shape(0);
    encoder.set_bytes(dim, 3);
    encoder.set_bytes(count, 4);
    encoder.dispatch_threads(MTL::Size(size_t(dim) * count, 1, 1),
                             MTL::Size(256, 1, 1));
  }
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override {
    throw std::runtime_error("PagedIndexStore is inference-only");
  }
};

void expect(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(std::string("mlx_kv_store_rows: ") + what);
  }
}
#elif defined(MLX_NODE_CUDA_ENABLED)
using namespace mlx::core;

constexpr char kQkNormRopeKvCudaSrc[] =
#include "cuda/qk_norm_rope_kv.cu.inc"
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
#endif

}  // namespace

// Whether the table path (more than one layer's tensors in one dispatch) is
// available: the Metal device with residency sets.
extern "C" bool mlx_kv_store_batched_available() {
#ifdef MLX_NODE_METAL_ENABLED
  return metal::is_available() && default_device() == Device::gpu &&
      metal::device(Device::gpu).residency_sets().enabled();
#else
  return false;
#endif
}

// Write `tensors` row blocks (`src[i]`, `[B, H, T, D_i]` or `[B, H, T]` for
// scales) into the matching cache buffers (`dst[i]`, `[B, H, cap, D_i]` /
// `[B, H, cap]`) at row `offsets[i]`, in place, in one dispatch. `out[i]`
// receives the cache array handles the caller must adopt (same buffers). Up
// to four tensors bind directly; more need MLX residency sets (macOS 15+).
// Returns false (message on stderr) without Metal or on a contract
// violation; the caller keeps its slice_update path.
extern "C" bool mlx_kv_store_rows(int32_t tensors, mlx_array* const* dst,
                                  mlx_array* const* src,
                                  const int32_t* offsets, mlx_array** out) {
  for (int i = 0; out && i < tensors && i < kMaxTensors; ++i) out[i] = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(dst && src && out && offsets && tensors >= 1 &&
               tensors <= kMaxTensors,
           "bad arguments");
    expect(metal::is_available() && default_device() == Device::gpu,
           "requires the Metal device");
    std::vector<array> inputs;
    std::vector<Shape> shapes;
    std::vector<Dtype> dtypes;
    for (int i = 0; i < tensors; ++i) {
      expect(dst[i] && src[i] && offsets[i] >= 0, "null array or offset");
      const auto& d = *reinterpret_cast<array*>(dst[i]);
      const auto& s = *reinterpret_cast<array*>(src[i]);
      expect(d.ndim() == s.ndim() && (d.ndim() == 3 || d.ndim() == 4),
             "cache and block must both be [B, H, N(, D)]");
      expect(d.dtype() == s.dtype(), "cache and block dtypes differ");
      expect(d.shape(0) == s.shape(0) && d.shape(1) == s.shape(1) &&
                 (d.ndim() == 3 || d.shape(3) == s.shape(3)) &&
                 int64_t(offsets[i]) + s.shape(2) <= d.shape(2),
             "block does not fit the cache at offset");
      inputs.push_back(d);
      shapes.push_back(d.shape());
      dtypes.push_back(d.dtype());
    }
    // One layer's tensors share (B, H, T, offset) and bind directly; any
    // other set goes through the address table, which needs residency sets
    // (quiet decline: the caller stores layer by layer instead).
    const auto& first = *reinterpret_cast<array*>(src[0]);
    bool bound = tensors <= kMaxBoundTensors;
    for (int i = 1; bound && i < tensors; ++i) {
      const auto& s = *reinterpret_cast<array*>(src[i]);
      bound = offsets[i] == offsets[0] && s.shape(0) == first.shape(0) &&
          s.shape(1) == first.shape(1) && s.shape(2) == first.shape(2);
    }
    if (!bound && !metal::device(Device::gpu).residency_sets().enabled()) {
      return false;
    }
    for (int i = 0; i < tensors; ++i) {
      inputs.push_back(*reinterpret_cast<array*>(src[i]));
    }
    auto stream = default_stream(Device::gpu);
    auto primitive = std::make_shared<KvStoreRows>(
        stream, std::vector<int>(offsets, offsets + tensors), bound);
    auto outputs = array::make_arrays(std::move(shapes), std::move(dtypes),
                                      primitive, std::move(inputs));
    for (int i = 0; i < tensors; ++i) {
      out[i] = reinterpret_cast<mlx_array*>(new array(std::move(outputs[i])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_kv_store_rows: %s\n", e.what());
    return false;
  }
#elif defined(MLX_NODE_CUDA_ENABLED)
  try {
    using namespace mlx::core;
    if (!dst || !src || !out || !offsets || tensors < 1 || tensors > kMaxTensors ||
        default_device() != Device::gpu) {
      return false;
    }
    static constexpr char kKvStoreRowsCudaSrc[] =
#include "cuda/kv_store_rows.cu.inc"
        ;
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
  (void)tensors; (void)dst; (void)src; (void)offsets;
  return false;
#endif
}

// Inputs: full K/V buffers, live-plus-reserve K/V views, new K/V rows.
// Outputs: full K/V aliases followed by their written attention views.
extern "C" bool mlx_kv_store_window_tail(mlx_array* const* args, mlx_array** out) {
  for (int i = 0; out && i < 4; ++i) out[i] = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(args && out && metal::is_available() && default_device() == Device::gpu,
           "window tail requires Metal and non-null arguments");
    std::vector<array> inputs;
    for (int i = 0; i < 6; ++i) {
      expect(args[i], "null window tail input");
      const auto& a = *reinterpret_cast<array*>(args[i]);
      expect(a.ndim() == 4 && a.dtype() == bfloat16 && a.shape(0) == 1,
             "window tail requires BF16 [1,H,T,D]");
      inputs.push_back(a);
    }
    for (int i = 0; i < 2; ++i) {
      expect(inputs[i].shape(1) == inputs[2+i].shape(1) &&
                 inputs[i].shape(3) == inputs[2+i].shape(3) &&
                 inputs[i].shape(1) == inputs[4+i].shape(1) &&
                 inputs[i].shape(3) == inputs[4+i].shape(3) &&
                 inputs[4+i].shape(2) > 0 &&
                 inputs[4+i].shape(2) <= inputs[2+i].shape(2),
             "window tail geometry mismatch");
    }
    std::vector<Shape> shapes;
    for (int i = 0; i < 4; ++i) shapes.push_back(inputs[i].shape());
    auto outputs = array::make_arrays(std::move(shapes),
        std::vector<Dtype>(4, bfloat16),
        std::make_shared<KvStoreWindowTail>(default_stream(Device::gpu)), std::move(inputs));
    for (int i = 0; i < 4; ++i)
      out[i] = reinterpret_cast<mlx_array*>(new array(std::move(outputs[i])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_kv_store_window_tail: %s\n", e.what());
    return false;
  }
#else
  (void)args;
  return false;
#endif
}

// Store `rows` (bf16 `[n, dim]`) into the index plane `plane` (bf16, its
// last axis `dim`) at plane rows `slots` (int32 `[n]`), in place. `out`
// receives the plane handle to adopt (same buffer). False (message on
// stderr) without Metal or on a contract violation.
extern "C" bool mlx_paged_index_store(mlx_array* plane, mlx_array* rows,
                                      mlx_array* slots, mlx_array** out) {
  if (out) *out = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!plane || !rows || !slots || !out) {
      throw std::invalid_argument("null argument");
    }
    const auto& p = *reinterpret_cast<array*>(plane);
    const auto r = contiguous(*reinterpret_cast<array*>(rows));
    const auto& s = *reinterpret_cast<array*>(slots);
    if (!metal::is_available() || !(default_device() == Device::gpu) ||
        p.dtype() != bfloat16 || r.dtype() != bfloat16 || r.ndim() != 2 ||
        p.ndim() < 2 || p.shape(p.ndim() - 1) != r.shape(1) || s.ndim() != 1 ||
        s.dtype() != int32 || s.shape(0) != r.shape(0) ||
        !p.flags().row_contiguous || p.offset() != 0) {
      throw std::invalid_argument(
          "a bf16 plane, bf16 [n, dim] rows and int32 [n] slots");
    }
    auto outputs = array::make_arrays(
        {p.shape()}, {bfloat16},
        std::make_shared<PagedIndexStore>(default_stream(Device::gpu)),
        {p, r, s});
    *out = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_paged_index_store: %s\n", e.what());
    return false;
  }
#else
  (void)plane; (void)rows; (void)slots;
  return false;
#endif
}

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
