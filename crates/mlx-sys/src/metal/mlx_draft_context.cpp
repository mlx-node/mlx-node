// The DFlash2 draft context window (Splash `DraftAttention::addWindowStore` /
// `addWindowLoad`): a u8 buffer of `slots` 4-bit rows of `width` values. The
// store and the copy write the window in place: their output aliases the
// window input, and the caller adopts it as its new handle.

#include "mlx_common.h"

#include <cstdio>
#include <stdexcept>
#include <string>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/fast_primitives.h"
#endif

namespace {

#ifdef MLX_NODE_METAL_ENABLED
using namespace mlx::core;

constexpr uint32_t kGroup = 64;
constexpr uint32_t kSimdgroups = 8;

uint64_t window_bytes(uint32_t width, uint32_t slots) {
  return uint64_t(slots) * (width / 2) + uint64_t(slots) * (width / kGroup) * 4;
}

// An in-place write addresses the window's buffer directly.
void expect_dense(const array& window) {
  if (!window.flags().row_contiguous || window.data_size() != window.size()) {
    throw std::invalid_argument("draft context window is not a dense buffer");
  }
}

// Mirrors `DraftContextWindowParams` in the kernel source.
struct WindowParams {
  uint32_t rows;
  uint32_t start_position;
  uint32_t width;
  uint32_t slots;
};

MTL::ComputePipelineState* pipeline(metal::Device& device, const char* name) {
  auto* library = device.get_library("mlx_node_draft_context", [] {
    return std::string(
#include "metal/common/draft_context.metal.inc"
    );
  });
  return device.get_kernel(name, library);
}

MTL::Size groups(uint32_t rows, uint32_t width) {
  const uint64_t tasks = uint64_t(rows) * (width / kGroup);
  return MTL::Size((tasks + kSimdgroups - 1) / kSimdgroups, 1, 1);
}

array row_contiguous(const array& input, const Stream& s,
                     metal::CommandEncoder& enc) {
  if (input.flags().row_contiguous) return input;
  array contiguous(input.shape(), input.dtype(), nullptr, {});
  contiguous.set_data(allocator::malloc(contiguous.nbytes()));
  copy_gpu(input, contiguous, CopyType::General, s);
  enc.add_temporary(contiguous);
  return contiguous;
}

// Inputs: window, rows `[n, width]` bf16. Output: the window, in place.
class DraftContextStore final : public fast::Custom {
 public:
  DraftContextStore(Stream s, WindowParams params)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("DraftContextStore has no fallback");
        }), params_(params) {}
  DEFINE_NAME(DraftContextStore)
  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const DraftContextStore&>(other);
    return params_.rows == o.params_.rows &&
        params_.start_position == o.params_.start_position &&
        params_.width == o.params_.width && params_.slots == o.params_.slots;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& in) override {
    return {in[0].shape()};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DraftContextStore requires Metal");
  }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    expect_dense(in[0]);
    const array rows = row_contiguous(in[1], s, enc);
    out[0].copy_shared_buffer(in[0]);
    enc.set_compute_pipeline_state(pipeline(device, "mlx_node_draft_context_store"));
    enc.set_input_array(rows, 0);
    enc.set_output_array(out[0], 1);
    enc.set_bytes(params_, 2);
    enc.dispatch_threadgroups(groups(params_.rows, params_.width),
                              MTL::Size(32 * kSimdgroups, 1, 1));
  }
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override {
    throw std::runtime_error("DraftContextStore is inference-only");
  }

 private:
  WindowParams params_;
};

// Input: window. Output: rows `[n, width]` bf16.
class DraftContextLoad final : public fast::Custom {
 public:
  DraftContextLoad(Stream s, WindowParams params)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("DraftContextLoad has no fallback");
        }), params_(params) {}
  DEFINE_NAME(DraftContextLoad)
  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const DraftContextLoad&>(other);
    return params_.rows == o.params_.rows &&
        params_.start_position == o.params_.start_position &&
        params_.width == o.params_.width && params_.slots == o.params_.slots;
  }
  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {{int(params_.rows), int(params_.width)}};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DraftContextLoad requires Metal");
  }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    const array window = row_contiguous(in[0], s, enc);
    out[0].set_data(allocator::malloc(out[0].nbytes()));
    enc.set_compute_pipeline_state(pipeline(device, "mlx_node_draft_context_load"));
    enc.set_input_array(window, 0);
    enc.set_output_array(out[0], 1);
    enc.set_bytes(params_, 2);
    enc.dispatch_threadgroups(groups(params_.rows, params_.width),
                              MTL::Size(32 * kSimdgroups, 1, 1));
  }
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override {
    throw std::runtime_error("DraftContextLoad is inference-only");
  }

 private:
  WindowParams params_;
};

// Inputs: destination window, source window. Output: the destination, in
// place.
class DraftContextCopy final : public fast::Custom {
 public:
  explicit DraftContextCopy(Stream s)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("DraftContextCopy has no fallback");
        }) {}
  DEFINE_NAME(DraftContextCopy)
  bool is_equivalent(const Primitive&) const override { return true; }
  std::vector<Shape> output_shapes(const std::vector<array>& in) override {
    return {in[0].shape()};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DraftContextCopy requires Metal");
  }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    expect_dense(in[0]);
    const array source = row_contiguous(in[1], s, enc);
    out[0].copy_shared_buffer(in[0]);
    const uint32_t words = uint32_t(in[0].nbytes() / 4);
    enc.set_compute_pipeline_state(pipeline(device, "mlx_node_draft_context_copy"));
    enc.set_input_array(source, 0);
    enc.set_output_array(out[0], 1);
    enc.set_bytes(words, 2);
    enc.dispatch_threads(MTL::Size(words, 1, 1), MTL::Size(256, 1, 1));
  }
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override {
    throw std::runtime_error("DraftContextCopy is inference-only");
  }
};

void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(what);
}

void expect_window(const array& window, uint32_t width, uint32_t slots) {
  expect(width > 0 && width % kGroup == 0 && slots > 0,
         "a window holds whole 64-value groups in at least one slot");
  expect(window.dtype() == uint8 && window.ndim() == 1 &&
             uint64_t(window.size()) == window_bytes(width, slots),
         "the window is a u8 buffer of its slots");
}

bool metal_ready() {
  return metal::is_available() && default_device() == Device::gpu;
}
#endif

}  // namespace

// Store bf16 `rows` `[n, width]` (rows `first_row..first_row + n`) into
// `window` at slots `p % slots`, in place; `out` is the window handle to
// adopt. `n` is at most `slots`. False (message on stderr) without Metal or
// on a contract violation.
extern "C" bool mlx_draft_context_store(mlx_array* window, mlx_array* rows,
                                        uint32_t first_row, uint32_t width,
                                        uint32_t slots, mlx_array** out) {
  if (out) *out = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(window && rows && out, "null argument");
    if (!metal_ready()) return false;
    const auto& w = *reinterpret_cast<array*>(window);
    const auto& r = *reinterpret_cast<array*>(rows);
    expect_window(w, width, slots);
    expect(r.dtype() == bfloat16 && r.ndim() == 2 && r.shape(1) == int(width) &&
               r.shape(0) >= 1 && uint32_t(r.shape(0)) <= slots,
           "rows are bf16 [n, width] with 1 <= n <= slots");
    const WindowParams params{uint32_t(r.shape(0)), first_row, width, slots};
    auto outputs = array::make_arrays(
        {w.shape()}, {uint8},
        std::make_shared<DraftContextStore>(default_stream(Device::gpu), params),
        {w, r});
    *out = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_draft_context_store: %s\n", e.what());
    return false;
  }
#else
  (void)window; (void)rows; (void)first_row; (void)width; (void)slots;
  return false;
#endif
}

// Rows `first_row..first_row + rows` of `window` as bf16 `[rows, width]`.
extern "C" bool mlx_draft_context_load(mlx_array* window, uint32_t first_row,
                                       uint32_t rows, uint32_t width,
                                       uint32_t slots, mlx_array** out) {
  if (out) *out = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(window && out, "null argument");
    if (!metal_ready()) return false;
    const auto& w = *reinterpret_cast<array*>(window);
    expect_window(w, width, slots);
    expect(rows >= 1 && rows <= slots, "1 <= rows <= slots");
    const WindowParams params{rows, first_row, width, slots};
    auto outputs = array::make_arrays(
        {{int(rows), int(width)}}, {bfloat16},
        std::make_shared<DraftContextLoad>(default_stream(Device::gpu), params),
        {w});
    *out = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_draft_context_load: %s\n", e.what());
    return false;
  }
#else
  (void)window; (void)first_row; (void)rows; (void)width; (void)slots;
  return false;
#endif
}

// Copy window `src`'s bytes into window `dst` in place; `out` is the `dst`
// handle to adopt.
extern "C" bool mlx_draft_context_copy(mlx_array* dst, mlx_array* src,
                                       mlx_array** out) {
  if (out) *out = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(dst && src && out, "null argument");
    if (!metal_ready()) return false;
    const auto& d = *reinterpret_cast<array*>(dst);
    const auto& s = *reinterpret_cast<array*>(src);
    expect(d.dtype() == uint8 && s.dtype() == uint8 && d.ndim() == 1 &&
               d.shape() == s.shape() && d.nbytes() % 4 == 0 &&
               d.id() != s.id(),
           "two distinct u8 windows of one size");
    auto outputs = array::make_arrays(
        {d.shape()}, {uint8},
        std::make_shared<DraftContextCopy>(default_stream(Device::gpu)), {d, s});
    *out = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_draft_context_copy: %s\n", e.what());
    return false;
  }
#else
  (void)dst; (void)src;
  return false;
#endif
}
