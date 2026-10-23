// Splash's draft M32/D128 tensor-op schedule, adapted to explicit MLX views.
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "metal/dflash2_attention/abi.h"
#include <limits>
#include <memory>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx_paged_metallib.h"
#endif

namespace mlx::core::dflash2_attention {
bool available() {
#ifdef MLX_NODE_METAL_ENABLED
  return !detail::in_grad_tracing() && default_device() == Device::gpu &&
      is_available(Device::gpu) && metal::is_nax_available();
#else
  return false;
#endif
}

bool supported(const std::vector<array>& a) {
  if (a.size() != 3 && a.size() != 4) return false;
  if (a[0].shape() != Shape{1, 32, 8, 128} || a[1].ndim() != 4 ||
      a[1].shape(0) != 1 || a[1].shape(1) != 8 || a[1].shape(2) < 8 ||
      a[1].shape(2) > 2055 || a[1].shape(3) != 128 || a[2].shape() != a[1].shape())
    return false;
  // MPP operands retain 16-byte row/base alignment. Arbitrary element-offset
  // views use generic SDPA instead of relying on undocumented tensor alignment.
  for (size_t i = 0; i < 3; ++i) {
    if (a[i].dtype() != bfloat16 || a[i].strides(3) != 1 ||
        a[i].strides(2) < 128 || a[i].strides(2) > std::numeric_limits<int>::max() ||
        a[i].offset() % 16 || a[i].strides(1) % 8 || a[i].strides(2) % 8)
      return false;
  }
  if (a[0].strides(1) != 8 * a[0].strides(2)) return false;
  if (a.size() == 4 && (a[3].dtype() != bool_ ||
      (a[3].shape() != Shape{1, 1, 8, a[1].shape(2)} &&
       a[3].shape() != Shape{8, a[1].shape(2)}))) return false;
  return true;
}

class DraftAttention : public fast::Custom {
 public:
  DraftAttention(Stream s, float scale, bool mask)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("DFlash2 tensor-op attention requires Metal");
        }), scale_(scale), has_mask_(mask) {}
  DEFINE_NAME(DraftAttention)
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const DraftAttention&>(other);
    return scale_ == rhs.scale_ && has_mask_ == rhs.has_mask_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& a) override {
    return {{1, 8, 32, 128}};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DFlash2 tensor-op attention requires Metal");
  }
  void eval_gpu(const std::vector<array>& a, std::vector<array>& outputs) override {
#ifdef MLX_NODE_METAL_ENABLED
    if (!supported(a) || (a.size() == 4) != has_mask_)
      throw std::invalid_argument("DFlash2 tensor-op attention replay changed its contract");
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    // Four splits x eight KV heads. Unlike vector SDPA's large split grid,
    // every partial remains FP32 and occupies only 0.51 MiB at this shape.
    array partials({8, 4, 32 * 128 + 2 * 32}, float32, nullptr, {});
    partials.set_data(allocator::malloc(partials.nbytes()));
    enc.add_temporary(partials);
    array mask = has_mask_ ? a[3] : a[0];
    if (has_mask_ && !mask.flags().row_contiguous) {
      array contiguous(mask.shape(), mask.dtype(), nullptr, {});
      contiguous.set_data(allocator::malloc(contiguous.nbytes()));
      copy_gpu(mask, contiguous, CopyType::General, s);
      enc.add_temporary(contiguous);
      mask = std::move(contiguous);
    }
    DraftAttentionParams p{};
    p.query_head_stride = a[0].strides(1);
    p.query_row_stride = a[0].strides(2);
    p.key_head_stride = a[1].strides(1);
    p.key_row_stride = a[1].strides(2);
    p.value_head_stride = a[2].strides(1);
    p.value_row_stride = a[2].strides(2);
    p.key_rows = a[1].shape(2);
    p.has_mask = has_mask_;
    p.scale = scale_;
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "dflash2_attention", "split", "dflash2_attention_split"));
    for (unsigned i = 0; i < 3; ++i) enc.set_input_array(a[i], i);
    enc.set_input_array(mask, 3);
    enc.set_output_array(partials, 4);
    enc.set_bytes(p, 5);
    enc.dispatch_threadgroups(MTL::Size(8, 4, 1), MTL::Size(256, 1, 1));
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "dflash2_attention", "reduce", "dflash2_attention_reduce"));
    enc.set_input_array(partials, 0);
    enc.set_output_array(out, 1);
    enc.dispatch_threadgroups(MTL::Size(8, 1, 1), MTL::Size(256, 1, 1));
#else
    throw std::runtime_error("DFlash2 tensor-op attention requires Metal");
#endif
  }
 private:
  float scale_;
  bool has_mask_;
};
}

// 1: constructed; 0: unsupported (use generic SDPA); -1: invalid argument or
// construction error. The primitive is read-only and performs no host readback.
extern "C" int mlx_dflash2_attention_with_policy(mlx_array* queries, mlx_array* keys,
    mlx_array* values, mlx_array* mask, float scale, bool enabled, mlx_array** output) {
  if (output) *output = nullptr;
  try {
    if (!queries || !keys || !values || !output || !std::isfinite(scale) || scale <= 0)
      return -1;
    using namespace mlx::core;
    if (!enabled || !dflash2_attention::available()) return 0;
    std::vector<array> a{*reinterpret_cast<array*>(queries),
                         *reinterpret_cast<array*>(keys), *reinterpret_cast<array*>(values)};
    if (mask) a.push_back(*reinterpret_cast<array*>(mask));
    if (!dflash2_attention::supported(a)) return 0;
    auto result = array(Shape{1, 8, 32, 128}, bfloat16,
        std::make_shared<dflash2_attention::DraftAttention>(to_stream(Device::gpu), scale, mask != nullptr),
        std::move(a));
    // Reduce writes token-major storage so the shared Rust BHQD -> BQHD
    // flatten is a view. The public attention contract remains BHQD.
    result = transpose(result, {0, 2, 1, 3});
    *output = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "DFlash2 tensor-op attention: " << e.what() << '\n';
    return -1;
  }
}

// Retain the environment-driven diagnostic ABI. Loaded models use the explicit
// policy entrypoint above, so a process-wide flag change cannot alter replay.
extern "C" int mlx_dflash2_attention(mlx_array* queries, mlx_array* keys,
    mlx_array* values, mlx_array* mask, float scale, mlx_array** output) {
  const char* value = std::getenv("MLX_DFLASH2_ATTENTION_NAX");
  const bool enabled = value && std::strcmp(value, "1") == 0;
  return mlx_dflash2_attention_with_policy(queries, keys, values, mask, scale, enabled, output);
}
