// In-place KV row store for the flat Qwen3.5 cache: one Metal dispatch per
// layer writes the verify block's K and V rows (plus int8 scales) into the
// cache buffers at the write offset (Splash `verify_attention_*_store`,
// Apache-2.0; see THIRD_PARTY_NOTICES). The outputs alias the cache inputs'
// buffers — the caller owns those buffers and replaces its handles with the
// outputs — so the four slice_update copies per layer become one dispatch.

#include "mlx_common.h"

#include <cstdio>
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

constexpr int kMaxTensors = 4;

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
  int dims[kMaxTensors];
  int itemsize[kMaxTensors];
  uint32_t elems[kMaxTensors];
  int64_t src_strides[kMaxTensors][4];
  int64_t dst_strides[kMaxTensors][4];
};

MTL::ComputePipelineState* kv_store_pipeline(metal::Device& device) {
  const std::string name = "mlx_node_kv_store_rows";
  auto* library = device.get_library(
      name, [] { return std::string(metal::utils()) + kKvStoreBody; });
  return device.get_kernel(name, library);
}

// Inputs: n cache buffers, then the n row blocks. Outputs: the n caches,
// aliasing the inputs' buffers.
class KvStoreRows final : public fast::Custom {
 public:
  KvStoreRows(Stream stream, int tensors, int offset)
      : Custom(stream,
               [](std::vector<array>) -> std::vector<array> {
                 throw std::runtime_error("KvStoreRows has no fallback");
               }),
        tensors_(tensors), offset_(offset) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("KvStoreRows requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    bridge_testing::record("kv_store_rows");
    KvStoreParams p{};
    p.tensors = tensors_;
    p.offset = offset_;
    uint32_t max_elems = 0;
    for (int i = 0; i < tensors_; ++i) {
      const auto& dst = inputs[i];
      const auto& src = inputs[tensors_ + i];
      auto& out = outputs[i];
      out.copy_shared_buffer(dst);
      // Scales are [B, H, T]; treat them as D = 1.
      const int nd = src.ndim();
      const int d = nd == 4 ? src.shape(3) : 1;
      if (i == 0) {
        p.rows = src.shape(2);
        p.batch = src.shape(0);
        p.heads = src.shape(1);
      }
      if (src.shape(0) != p.batch || src.shape(1) != p.heads ||
          src.shape(2) != p.rows || dst.shape(0) != p.batch ||
          dst.shape(1) != p.heads || dst.ndim() != nd ||
          (nd == 4 && dst.shape(3) != d) ||
          offset_ + p.rows > dst.shape(2) || dst.itemsize() != src.itemsize()) {
        throw std::runtime_error("KvStoreRows: block does not fit the cache");
      }
      p.dims[i] = d;
      p.itemsize[i] = int(src.itemsize());
      p.elems[i] = uint32_t(src.size());
      for (int a = 0; a < 3; ++a) {
        p.src_strides[i][a] = src.strides(a);
        p.dst_strides[i][a] = dst.strides(a);
      }
      p.src_strides[i][3] = nd == 4 ? src.strides(3) : 0;
      p.dst_strides[i][3] = nd == 4 ? dst.strides(3) : 0;
      max_elems = std::max(max_elems, p.elems[i]);
    }
    auto& stream = this->stream();
    auto& encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(
        kv_store_pipeline(metal::device(stream.device)));
    for (int i = 0; i < kMaxTensors; ++i) {
      const int j = std::min(i, tensors_ - 1);
      encoder.set_output_array(outputs[j], i);
      encoder.set_input_array(inputs[tensors_ + j], kMaxTensors + i);
    }
    encoder.set_bytes(p, 2 * kMaxTensors);
    encoder.dispatch_threads(MTL::Size(max_elems, tensors_, 1),
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
    return tensors_ == o.tensors_ && offset_ == o.offset_;
  }

 private:
  int tensors_;
  int offset_;
};

void expect(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(std::string("mlx_kv_store_rows: ") + what);
  }
}
#endif

}  // namespace

// Write `tensors` row blocks (`src[i]`, `[B, H, T, D_i]` or `[B, H, T]` for
// scales) into the matching cache buffers (`dst[i]`, `[B, H, cap, D_i]` /
// `[B, H, cap]`) at row `offset`, in place, in one dispatch. `out[i]` receives
// the cache array handles the caller must adopt (same buffers). Returns false
// (message on stderr) without Metal or on a contract violation; the caller
// keeps its slice_update path.
extern "C" bool mlx_kv_store_rows(int32_t tensors, mlx_array* const* dst,
                                  mlx_array* const* src, int32_t offset,
                                  mlx_array** out) {
  for (int i = 0; out && i < tensors && i <= kMaxTensors; ++i) out[i] = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(dst && src && out && tensors >= 1 && tensors <= kMaxTensors &&
               offset >= 0,
           "bad arguments");
    expect(metal::is_available() && default_device() == Device::gpu,
           "requires the Metal device");
    std::vector<array> inputs;
    std::vector<Shape> shapes;
    std::vector<Dtype> dtypes;
    for (int i = 0; i < tensors; ++i) {
      expect(dst[i] && src[i], "null array");
      const auto& d = *reinterpret_cast<array*>(dst[i]);
      const auto& s = *reinterpret_cast<array*>(src[i]);
      expect(d.ndim() == s.ndim() && (d.ndim() == 3 || d.ndim() == 4),
             "cache and block must both be [B, H, N(, D)]");
      expect(d.dtype() == s.dtype(), "cache and block dtypes differ");
      expect(d.shape(0) == s.shape(0) && d.shape(1) == s.shape(1) &&
                 (d.ndim() == 3 || d.shape(3) == s.shape(3)) &&
                 int64_t(offset) + s.shape(2) <= d.shape(2),
             "block does not fit the cache at offset");
      inputs.push_back(d);
      shapes.push_back(d.shape());
      dtypes.push_back(d.dtype());
    }
    for (int i = 0; i < tensors; ++i) {
      inputs.push_back(*reinterpret_cast<array*>(src[i]));
    }
    auto stream = default_stream(Device::gpu);
    auto primitive = std::make_shared<KvStoreRows>(stream, tensors, offset);
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
  (void)tensors; (void)dst; (void)src; (void)offset;
  return false;
#endif
}
