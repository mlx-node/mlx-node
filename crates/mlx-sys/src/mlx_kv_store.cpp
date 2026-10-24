// In-place KV row store for the flat Qwen3.5 cache: one Metal dispatch
// writes verify-block K and V rows (plus int8 scales) into the cache buffers
// at their write offsets (Splash `verify_attention_*_store`, Apache-2.0; see
// THIRD_PARTY_NOTICES). The outputs alias the cache inputs' buffers — the
// caller owns those buffers and replaces its handles with the outputs — so
// the slice_update copies become one dispatch. Up to four tensors bind
// directly (one layer); more go through a GPU-address table so every
// full-attention layer of a verify block is stored by ONE dispatch.

#include "mlx_common.h"

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

void expect(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(std::string("mlx_kv_store_rows: ") + what);
  }
}
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
  for (int i = 0; out && i < tensors && i <= kMaxTensors; ++i) out[i] = nullptr;
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
