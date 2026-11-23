// Fused GDN step: the per-step gated delta recurrence of one window with the
// layer's gated RMSNorm and z gate in the same dispatch — and, in the
// "complete" form, the `qwen4_gdn_prepare` prologue (conv + SiLU + q/k norm
// + gates) too (structure after Splash `verify_gdn_fused`, Apache-2.0; see
// THIRD_PARTY_NOTICES). Plain primitives rather than `fast::metal_kernel`
// so `state_out` can alias a caller-owned buffer (the DFlash2 lane's other
// GDN cell, which the deferred verify folds into) and the split views (`z`,
// `qkv`, `a`, `b`) are read through their strides.

#include "mlx_common.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"
#include "mlx_test_counters.h"
#endif

namespace {

#ifdef MLX_NODE_METAL_ENABLED
using namespace mlx::core;

const char* kGdnFusedBody =
#include "metal/common/gdn_fused_step.metal.inc"
    ;

// Mirrors `GdnFusedParams` in the kernel source.
struct GdnFusedParams {
  int T;
  int hk;
  int hv;
  int z_stride_b;
  int z_stride_t;
  float eps;
  int qkv_stride_t;
  int qkv_stride_c;
  int hist_stride_r;
  int hist_stride_c;
  int a_stride_t;
  int a_stride_c;
  int b_stride_t;
  int b_stride_c;
  int pending_T;
  int tiled_heads;
};

MTL::ComputePipelineState* gdn_fused_pipeline(metal::Device& device, int dk,
                                              int dv, bool complete, bool scan8 = false, bool prepared = false, bool t8_storage = false, bool deferred = false) {
  const std::string name =
      complete ? "mlx_node_gdn_fused_complete" : "mlx_node_gdn_fused_step";
  const std::string lib_name =
      name + "_dk" + std::to_string(dk) + "_dv" + std::to_string(dv) + (scan8 ? "_scan8" : "") + (prepared ? "_sg8" : "") + (t8_storage ? "_t8_storage" : "") + (deferred ? "_deferred" : "");
  auto* library = device.get_library(lib_name, [dk, dv, complete, scan8, prepared, t8_storage, deferred] {
    return std::string(metal::utils()) + "\n#define GDN_FUSED_N_PER_T " +
        std::to_string(dk / 32) + "\n#define GDN_FUSED_DV " +
        std::to_string(dv) + "\n#define GDN_FUSED_COMPLETE " +
        (complete ? "1" : "0") + "\n#define GDN_FUSED_SCAN8 " +
        (scan8 ? "1" : "0") + "\n#define GDN_FUSED_PREPARED " +
        (prepared ? "1" : "0") + "\n#define GDN_FUSED_T8_STORAGE " +
        (t8_storage ? "1" : "0") + "\n#define GDN_FUSED_DEFERRED " +
        (deferred ? "1" : "0") + "\n" + kGdnFusedBody;
  });
  auto* pipeline = device.get_kernel(name, library);
  if (pipeline->threadExecutionWidth() != 32 ||
      pipeline->maxTotalThreadsPerThreadgroup() < NS::UInteger(scan8 ? 256 : 8 * dv)) {
    throw std::runtime_error(
        "gdn fused pipeline cannot hold one value head per threadgroup");
  }
  return pipeline;
}

// Pending tape is explicit graph data. Output state/history always mean the
// settled prior prefix, never the speculative current window, on every replay.
void validate_deferred(const std::vector<array>& a, int hk, int dk, int dv,
                       bool has_dst, bool has_history_dst) {
  auto check = [](bool ok) { if (!ok) throw std::invalid_argument("GDN deferred replay contract"); };
  check(a.size() == size_t(16 + int(has_dst) + int(has_history_dst)));
  check(a[0].ndim() == 3 && a[0].shape(0) == 1 && a[0].shape(1) >= 1 && a[0].shape(1) <= 16);
  check(a[7].ndim() == 4 && a[7].shape(0) == 1 && a[7].shape(2) == dv && a[7].shape(3) == dk);
  const int hv = a[7].shape(1), width = 2 * hk * dk + hv * dv;
  check(dk == 128 && dv == 128 && hk > 0 && hv % hk == 0 && a[0].shape(2) == width);
  check(a[10].ndim() == 4 && a[10].shape(0) == 1);
  const int t = a[10].shape(1);
  check(t >= 1 && t <= 16 && a[10].shape() == Shape{1,t,hk,dk});
  check(a[11].shape() == Shape{1,t,hv,dv} && a[12].shape() == Shape{1,t,hv} && a[13].shape() == a[12].shape());
  check(a[14].shape() == Shape{1,t,width} && a[15].shape() == Shape{1});
  check(a[10].dtype() == bfloat16 && a[11].dtype() == bfloat16 && a[12].dtype() == float32 &&
        a[13].dtype() == bfloat16 && a[14].dtype() == bfloat16 && a[15].dtype() == int32);
}

// Shared host side of both kernels. `Layout` fixes the input order:
//   step:     q, k, v, g, beta, state_in, z, w [, state_dst]
//   complete: qkv, a, b, conv, history, scale, dt, state_in, z, w
//             [, state_dst] [, history_dst]
// Outputs: out, state_out [, q, k, v, decay, beta, next_history
// [, prepared words, prepared sums] [, deferred raw inputs]]; state_out
// (and, complete, next_history) alias their destinations when given.
class GdnFused : public fast::Custom {
 public:
  GdnFused(Stream stream, bool complete, int dk, int dv, float eps,
           bool has_dst, bool has_history_dst, bool allow_scan8 = false, bool prepared = false, bool allow_t8_storage = false, bool deferred = false, bool tiled_heads = true)
      : Custom(stream,
               [](std::vector<array>) -> std::vector<array> {
                 throw std::runtime_error("GdnFused has no fallback");
               }),
        complete_(complete),
        dk_(dk),
        dv_(dv),
        eps_(eps),
        has_dst_(has_dst),
        has_history_dst_(has_history_dst), allow_scan8_(allow_scan8), prepared_(prepared), allow_t8_storage_(allow_t8_storage), deferred_(deferred), tiled_heads_(tiled_heads) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GdnFused requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    bridge_testing::record(complete_ ? "gdn_fused_complete" : "gdn_fused_step");
    if (deferred_) {
      validate_deferred(inputs, outputs[2].shape(2), dk_, dv_, has_dst_, has_history_dst_);
      bridge_testing::record("gdn_fused_deferred");
      // Check the actual evaluated allocations before any destination write.
      const int destinations = 16;
      for (size_t dst = destinations; dst < inputs.size(); ++dst) {
        for (size_t other = 0; other < dst; ++other) {
          if (inputs[dst].buffer().ptr() == inputs[other].buffer().ptr())
            throw std::invalid_argument("GDN deferred destination aliases a live input");
        }
      }
    }
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    auto& encoder = metal::get_command_encoder(stream);
    const bool scan8 = allow_scan8_ && complete_ && dk_ == 128 && dv_ == 128 &&
        inputs[0].shape(1) == 8;
    if (prepared_ && (!complete_ || inputs[0].shape(1) != 8 || dk_ != 128 || dv_ != 128))
      throw std::invalid_argument("prepared GDN replay must retain eight rows");
    // Select from replay inputs, not the first trace's width. Non-eight
    // rows retain the original general storage bounds and schedule.
    const bool t8_storage = allow_t8_storage_ && complete_ && inputs[0].shape(1) == 8;
    auto* pipeline = gdn_fused_pipeline(device, dk_, dv_, complete_, scan8, prepared_, t8_storage, deferred_);
    if (t8_storage) bridge_testing::record("gdn_t8_storage");
    if (prepared_) bridge_testing::record("gdn_sg8_producer");

    // Every copy is encoded before this kernel's pipeline and bindings:
    // the copy kernels rebind the encoder's buffer table.
    std::vector<array> copies;
    auto contiguous = [&](const array& x) -> array {
      if (x.flags().row_contiguous) {
        return x;
      }
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, stream);
      return copies.back();
    };
    const int fixed = complete_ ? 10 : 8;
    const int z_at = complete_ ? 8 : 6;
    const int state_at = complete_ ? 7 : 5;
    const int v_at = complete_ ? 0 : 2;
    const array& z = inputs[z_at];
    if (z.strides(2) != 1) {
      throw std::runtime_error("GdnFused: z needs a contiguous last axis");
    }
    GdnFusedParams params{};
    std::vector<array> bound;
    bound.reserve(fixed + (deferred_ ? 6 : 0));
    for (int i = 0; i < fixed + (deferred_ ? 6 : 0); ++i) {
      // The split views stay strided; everything else binds row-contiguous.
      const bool strided = i == z_at || (complete_ && (i == 0 || i == 1 || i == 2 || i == 4));
      bound.push_back(strided ? inputs[i] : contiguous(inputs[i]));
    }
    if (complete_) {
      const array& qkv = inputs[0];
      const array& a = inputs[1];
      const array& b = inputs[2];
      const array& history = inputs[4];
      params.qkv_stride_t = int(qkv.strides(1));
      params.qkv_stride_c = int(qkv.strides(2));
      params.a_stride_t = int(a.strides(1));
      params.a_stride_c = int(a.strides(2));
      params.b_stride_t = int(b.strides(1));
      params.b_stride_c = int(b.strides(2));
      params.hist_stride_r = int(history.strides(0));
      params.hist_stride_c = int(history.strides(1));
    }

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    int extra = fixed + (deferred_ ? 6 : 0);
    auto place = [&](array& o, bool in_place, const char* what) {
      if (!in_place) {
        o.set_data(allocator::malloc(o.nbytes()));
        return;
      }
      const array& dst = inputs[extra++];
      if (!dst.flags().row_contiguous || dst.shape() != o.shape()) {
        throw std::runtime_error(std::string("GdnFused: ") + what +
                                 " must be a row-contiguous view of the output's shape");
      }
      o.copy_shared_buffer(dst);
    };
    place(outputs[1], has_dst_, "state_dst");
    for (size_t i = 2; i < outputs.size(); ++i) {
      place(outputs[i], complete_ && i == 7 && has_history_dst_, "history_dst");
    }
    const array& state = inputs[state_at];
    params.T = inputs[v_at].shape(1);
    params.hv = state.shape(1);
    params.hk = complete_ ? int(outputs[2].shape(2)) : int(inputs[0].shape(2));
    params.z_stride_b = int(z.strides(0));
    params.z_stride_t = int(z.strides(1));
    params.eps = eps_;
    params.pending_T = deferred_ ? inputs[10].shape(1) : 0;
    params.tiled_heads = tiled_heads_ ? 1 : 0;

    encoder.set_compute_pipeline_state(pipeline);
    int slot = 0;
    for (int i = 0; i < fixed; ++i) {
      encoder.set_input_array(bound[i], slot++);
    }
    // The deferred raw-input output binds after the pending inputs.
    const size_t in_order = outputs.size() - (deferred_ ? 1 : 0);
    for (size_t i = 0; i < in_order; ++i) {
      encoder.set_output_array(outputs[i], slot++);
    }
    encoder.set_bytes(params, slot++);
    if (deferred_) {
      for (int i = 10; i < 16; ++i) encoder.set_input_array(bound[i], slot++);
      encoder.set_output_array(outputs.back(), slot++);
    }
    encoder.dispatch_threadgroups(
        MTL::Size(size_t(state.shape(0)) * size_t(params.hv), 1, 1),
        MTL::Size(scan8 ? 256 : size_t(8) * size_t(dv_), 1, 1));
    encoder.add_temporaries(std::move(copies));
  }

  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&,
                         const std::vector<array>&) override {
    throw std::runtime_error("GdnFused is inference-only");
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const GdnFused&>(other);
    return complete_ == o.complete_ && dk_ == o.dk_ && dv_ == o.dv_ &&
        eps_ == o.eps_ && has_dst_ == o.has_dst_ &&
        has_history_dst_ == o.has_history_dst_ && allow_scan8_ == o.allow_scan8_ &&
        prepared_ == o.prepared_ && allow_t8_storage_ == o.allow_t8_storage_ && deferred_ == o.deferred_ &&
        tiled_heads_ == o.tiled_heads_;
  }

 protected:
  bool complete_;
  int dk_;
  int dv_;
  float eps_;
  bool has_dst_;
  bool has_history_dst_;
  bool allow_scan8_;
  bool prepared_;
  bool allow_t8_storage_;
  bool deferred_;
  bool tiled_heads_;
};

class GdnFusedStep final : public GdnFused {
 public:
  GdnFusedStep(Stream stream, int dk, int dv, float eps, bool has_dst)
      : GdnFused(stream, false, dk, dv, eps, has_dst, false) {}

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& v = inputs[2];
    return {Shape{v.shape(0), v.shape(1), v.shape(2) * v.shape(3)},
            inputs[5].shape()};
  }

  DEFINE_NAME(GdnFusedStep)
};

class GdnFusedComplete final : public GdnFused {
 public:
  GdnFusedComplete(Stream stream, int hk, int dk, int dv, float eps,
                   bool has_dst, bool has_history_dst, bool allow_scan8, bool prepared = false, bool allow_t8_storage = false, bool deferred = false, bool tiled_heads = true)
      : GdnFused(stream, true, dk, dv, eps, has_dst, has_history_dst, allow_scan8, prepared, allow_t8_storage, deferred, tiled_heads),
        hk_(hk) {}

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    if (deferred_) validate_deferred(inputs, hk_, dk_, dv_, has_dst_, has_history_dst_);
    const auto& qkv = inputs[0];
    const auto& state = inputs[7];
    const int T = qkv.shape(1), hv = state.shape(1);
    std::vector<Shape> shapes{Shape{1, T, hv * dv_}, state.shape(), Shape{1, T, hk_, dk_},
            Shape{1, T, hk_, dk_}, Shape{1, T, hv, dv_}, Shape{1, T, hv},
            Shape{1, T, hv}, inputs[4].shape()};
    if (prepared_) {
      if (T != 8) throw std::invalid_argument("prepared GDN shape must retain eight rows");
      shapes.push_back(Shape{hv * dv_ * 4});
      shapes.push_back(Shape{hv * dv_ / 32 * 8});
    }
    if (deferred_) shapes.push_back(qkv.shape());
    return shapes;
  }

  bool is_equivalent(const Primitive& other) const override {
    return GdnFused::is_equivalent(other) &&
        hk_ == static_cast<const GdnFusedComplete&>(other).hk_;
  }

  // Stable diagnostic identity for the selected deferred-settlement graph.
  // Equality, output storage, bindings and arithmetic remain unchanged.
  const char* name() const override {
    return deferred_ ? "GdnFusedCompleteDeferred" : "GdnFusedComplete";
  }

 private:
  int hk_;
};

void expect(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(std::string("mlx_gdn_fused: ") + what);
  }
}

// Shared checks of the tail inputs; returns the optional destination.
std::optional<array> check_tail(const array& sa, const array& za,
                                const array& wa, int B, int T, int Hv, int Dv,
                                int Dk, float eps, mlx_array* state_dst) {
  expect(sa.ndim() == 4 && za.ndim() == 3 && wa.ndim() == 1, "tail ranks");
  expect(Dv == 128 && Dk % 32 == 0 && Dk <= 256,
         "Dv must be 128, Dk a multiple of 32");
  expect(sa.shape() == Shape{B, Hv, Dv, Dk} &&
             za.shape() == Shape{B, T, Hv * Dv} && wa.shape(0) == Dv,
         "tail shapes");
  expect(za.dtype() == bfloat16 && wa.dtype() == bfloat16 &&
             sa.dtype() == float32,
         "tail dtypes (z/w bf16, state f32)");
  expect(za.strides(2) == 1, "z needs a contiguous last axis");
  expect(std::isfinite(eps), "eps");
  if (!state_dst) {
    return std::nullopt;
  }
  const auto& da = *reinterpret_cast<array*>(state_dst);
  expect(da.shape() == sa.shape() && da.dtype() == float32,
         "state_dst must match the state");
  return da;
}

// ---------------------------------------------------------------------------
// Chunked fused GDN prefill (prompt-length windows, T > 32)
//
// One primitive per GDN layer for a whole prompt: fused_gdn_gating + the q/k
// rms_norm (prologue), the 4-vcol recurrence with 16 state rows of a value
// head per threadgroup (scan), the gated RMSNorm and z gate of
// gdn_fused_step (tail).
// Bit-identical to the unfused prefill chain (fast::metal_kernel recurrence
// -> fast::rms_norm -> compiled swiglu). A plain fast::Custom primitive —
// like GdnFused — so the pre-norm conv-output split views and the z/a/b
// gate views are read through their strides instead of being copied
// contiguous first. B = 1, Dk = Dv = 128.

const char* kGdnPrefillHeadBody =
#include "metal/common/gdn_fused_prefill.metal.inc"
    ;
const char* kGdnPrefillRowBody =
#include "metal/common/gdn_prefill_row_scan.metal.inc"
    ;

// Mirrors `GdnPrefillParams` in both kernel sources.
struct GdnPrefillParams {
  int T;
  int hk;
  int hv;
  int tiled;
  int has_state;
  int q_stride_t;
  int k_stride_t;
  int v_stride_t;
  int a_stride_t;
  int b_stride_t;
  int z_stride_t;
  float eps;
  float qk_eps;
};

// Product: state rows of a value head split over threadgroups. Head: one
// value head per 1024-thread threadgroup, the bit reference for tests.
enum class GdnPrefillGeometry { Rows, Head };

constexpr size_t kGdnRowThreads = 128;
constexpr size_t kGdnRowsPerGroup = 16;
constexpr size_t kGdnRowSideThreads = 256;

struct GdnPrefillPipelines {
  MTL::ComputePipelineState* head = nullptr;
  MTL::ComputePipelineState* prologue = nullptr;
  MTL::ComputePipelineState* scan = nullptr;
  MTL::ComputePipelineState* tail = nullptr;
};

MTL::ComputePipelineState* checked_pipeline(MTL::ComputePipelineState* p,
                                            size_t threads) {
  if (p->threadExecutionWidth() != 32 ||
      p->maxTotalThreadsPerThreadgroup() < NS::UInteger(threads)) {
    throw std::runtime_error(
        "gdn fused prefill cannot launch its threadgroup on this GPU");
  }
  return p;
}

GdnPrefillPipelines gdn_prefill_pipelines(metal::Device& device,
                                          GdnPrefillGeometry geometry, int dk,
                                          int dv, bool bf16_scalars) {
  GdnPrefillPipelines out;
  if (geometry == GdnPrefillGeometry::Head) {
    const std::string lib_name = "mlx_node_gdn_fused_prefill_dk" +
        std::to_string(dk) + "_dv" + std::to_string(dv);
    auto* library = device.get_library(lib_name, [dk, dv] {
      return std::string(metal::utils()) + "\n#define GDN_PREFILL_N_PER_T " +
          std::to_string(dk / 32) + "\n#define GDN_PREFILL_DV " +
          std::to_string(dv) + "\n" + kGdnPrefillHeadBody;
    });
    out.head = checked_pipeline(
        device.get_kernel("mlx_node_gdn_fused_prefill", library),
        size_t(8) * size_t(dv));
    return out;
  }
  auto* library = device.get_library("mlx_node_gdn_prefill_rows", [] {
    return std::string(metal::utils()) + "\n" + kGdnPrefillRowBody;
  });
  out.prologue = checked_pipeline(
      device.get_kernel(bf16_scalars
                            ? "mlx_node_gdn_prefill_prologue_bf16_scalars"
                            : "mlx_node_gdn_prefill_prologue",
                        library),
      kGdnRowSideThreads);
  out.scan = checked_pipeline(
      device.get_kernel("mlx_node_gdn_prefill_scan", library), kGdnRowThreads);
  out.tail = checked_pipeline(
      device.get_kernel("mlx_node_gdn_prefill_tail", library),
      kGdnRowSideThreads);
  return out;
}

size_t ceil_groups(size_t threads, size_t per_group) {
  return (threads + per_group - 1) / per_group;
}

// Input layout: q, k, v, a, b, a_log, dt_bias, state_in, wq, wk, z, w.
// Outputs: out [1, T, Hv * Dv] bf16, state_out [1, Hv, Dv, Dk] f32.
class GdnFusedPrefill final : public fast::Custom {
 public:
  GdnFusedPrefill(Stream stream, GdnPrefillGeometry geometry, int dk, int dv,
                  float eps, float qk_eps, bool tiled, bool has_state,
                  bool bf16_scalars)
      : Custom(stream,
               [](std::vector<array>) -> std::vector<array> {
                 throw std::runtime_error("GdnFusedPrefill has no fallback");
               }),
        geometry_(geometry),
        dk_(dk),
        dv_(dv),
        eps_(eps),
        qk_eps_(qk_eps),
        tiled_(tiled),
        has_state_(has_state),
        bf16_scalars_(bf16_scalars) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GdnFusedPrefill requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    bridge_testing::record(geometry_ == GdnPrefillGeometry::Rows
                               ? "gdn_prefill_row_scan"
                               : "gdn_prefill_head_scan");
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    auto& encoder = metal::get_command_encoder(stream);
    const auto pipelines =
        gdn_prefill_pipelines(device, geometry_, dk_, dv_, bf16_scalars_);

    GdnPrefillParams params{};
    params.tiled = tiled_ ? 1 : 0;
    params.has_state = has_state_ ? 1 : 0;
    params.eps = eps_;
    params.qk_eps = qk_eps_;

    // The strided split views (q/k/v rows, a/b/z rows) bind as-is and are
    // walked through their params strides; the dense sidecars bind
    // row-contiguous. Without a carried state (has_state == 0) the inputs
    // hold no state array at all and a scratch buffer stands in for the
    // binding the kernel never reads.
    const int z_at = has_state_ ? 10 : 9;
    std::vector<array> copies;
    std::vector<array> bound;
    bound.reserve(12);
    for (size_t i = 0; i < inputs.size(); ++i) {
      const array& x = inputs[i];
      const bool strided =
          (i <= 4) || (int(i) == z_at);  // q/k/v/a/b rows and z rows
      if (strided || x.flags().row_contiguous) {
        bound.push_back(x);
      } else {
        copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
        copy_gpu(x, copies.back(), CopyType::General, stream);
        bound.push_back(copies.back());
      }
    }
    if (!has_state_) {
      // Placeholder the kernel never reads (has_state == 0): insert it at
      // slot 7 where the kernel declares `state_in`.
      copies.push_back(array({1}, float32, nullptr, {}));
      copies.back().set_data(allocator::malloc(4));
      bound.insert(bound.begin() + 7, copies.back());
    }

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));

    params.T = inputs[0].shape(1);
    params.hk = inputs[0].shape(2);
    params.hv = inputs[2].shape(2);
    params.q_stride_t = int(inputs[0].strides(1));
    params.k_stride_t = int(inputs[1].strides(1));
    params.v_stride_t = int(inputs[2].strides(1));
    params.a_stride_t = int(inputs[3].strides(1));
    params.b_stride_t = int(inputs[4].strides(1));
    params.z_stride_t = int(inputs[z_at].strides(1));

    if (geometry_ == GdnPrefillGeometry::Head) {
      encoder.set_compute_pipeline_state(pipelines.head);
      int slot = 0;
      for (auto& b : bound) {
        encoder.set_input_array(b, slot++);
      }
      for (auto& o : outputs) {
        encoder.set_output_array(o, slot++);
      }
      encoder.set_bytes(params, slot);
      encoder.dispatch_threadgroups(MTL::Size(size_t(params.hv), 1, 1),
                                    MTL::Size(size_t(8) * size_t(dv_), 1, 1));
      encoder.add_temporaries(std::move(copies));
      return;
    }

    // bound: q k v a b a_log dt_bias state_in wq wk z w
    const size_t T = size_t(params.T);
    const size_t Hk = size_t(params.hk);
    const size_t Hv = size_t(params.hv);
    auto scratch = [](Shape shape, Dtype dtype) {
      array a(std::move(shape), dtype, nullptr, {});
      a.set_data(allocator::malloc(a.nbytes()));
      return a;
    };
    array qn = scratch(Shape{params.T, params.hk, dk_}, bfloat16);
    array kn = scratch(Shape{params.T, params.hk, dk_}, bfloat16);
    array g = scratch(Shape{params.T, params.hv}, float32);
    array beta = scratch(Shape{params.T, params.hv}, bfloat16);
    copies.insert(copies.end(), {qn, kn, g, beta});

    encoder.set_compute_pipeline_state(pipelines.prologue);
    encoder.set_input_array(bound[0], 0);
    encoder.set_input_array(bound[1], 1);
    encoder.set_input_array(bound[3], 2);
    encoder.set_input_array(bound[4], 3);
    encoder.set_input_array(bound[5], 4);
    encoder.set_input_array(bound[6], 5);
    encoder.set_input_array(bound[8], 6);
    encoder.set_input_array(bound[9], 7);
    encoder.set_output_array(qn, 8);
    encoder.set_output_array(kn, 9);
    encoder.set_output_array(g, 10);
    encoder.set_output_array(beta, 11);
    encoder.set_bytes(params, 12);
    encoder.dispatch_threadgroups(
        MTL::Size(ceil_groups(std::max(T * Hk * 32, T * Hv),
                              kGdnRowSideThreads),
                  1, 1),
        MTL::Size(kGdnRowSideThreads, 1, 1));

    encoder.set_compute_pipeline_state(pipelines.scan);
    encoder.set_input_array(qn, 0);
    encoder.set_input_array(kn, 1);
    encoder.set_input_array(bound[2], 2);
    encoder.set_input_array(g, 3);
    encoder.set_input_array(beta, 4);
    encoder.set_input_array(bound[7], 5);
    encoder.set_output_array(out, 6);
    encoder.set_output_array(outputs[1], 7);
    encoder.set_bytes(params, 8);
    // Grid after Splash's prefill GDN scan (splash runtime/ops/GDN.cpp:177-181).
    encoder.dispatch_threadgroups(
        MTL::Size(Hv * (size_t(dv_) / kGdnRowsPerGroup), 1, 1),
        MTL::Size(kGdnRowThreads, 1, 1));

    encoder.set_compute_pipeline_state(pipelines.tail);
    encoder.set_output_array(out, 0);
    encoder.set_input_array(bound[10], 1);
    encoder.set_input_array(bound[11], 2);
    encoder.set_bytes(params, 3);
    encoder.dispatch_threadgroups(
        MTL::Size(ceil_groups(T * Hv * 32, kGdnRowSideThreads), 1, 1),
        MTL::Size(kGdnRowSideThreads, 1, 1));
    encoder.add_temporaries(std::move(copies));
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    const auto& v = inputs[2];
    return {Shape{1, q.shape(1), v.shape(2) * v.shape(3)},
            Shape{1, v.shape(2), v.shape(3), q.shape(3)}};
  }

  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&,
                         const std::vector<array>&) override {
    throw std::runtime_error("GdnFusedPrefill is inference-only");
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const GdnFusedPrefill&>(other);
    return geometry_ == o.geometry_ && dk_ == o.dk_ && dv_ == o.dv_ &&
        eps_ == o.eps_ && qk_eps_ == o.qk_eps_ && tiled_ == o.tiled_ &&
        has_state_ == o.has_state_ && bf16_scalars_ == o.bf16_scalars_;
  }

  DEFINE_NAME(GdnFusedPrefill);

 private:
  GdnPrefillGeometry geometry_;
  int dk_;
  int dv_;
  float eps_;
  float qk_eps_;
  bool tiled_;
  bool has_state_;
  bool bf16_scalars_;
};

void expect_prefill(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(std::string("mlx_gdn_fused_prefill: ") + what);
  }
}
#endif

}  // namespace

// True when both arrays are evaluated and are the same bytes of the same
// buffer (same offset and size; shapes may differ by a view).
extern "C" bool mlx_array_aliases(const mlx_array* a, const mlx_array* b) {
  if (!a || !b) return false;
  const auto& x = *reinterpret_cast<const array*>(a);
  const auto& y = *reinterpret_cast<const array*>(b);
  if (!x.is_available() || !y.is_available()) return false;
  return x.buffer().ptr() == y.buffer().ptr() && x.offset() == y.offset() &&
      x.nbytes() == y.nbytes();
}

// Fused GDN recurrence + gated RMSNorm + z gate for one window.
//
//   q, k   [B, T, Hk, Dk] bf16 (Hk divides Hv; value head hv uses hv % Hk)
//   v      [B, T, Hv, Dv] bf16,  g [B, T, Hv] f32,  beta [B, T, Hv] bf16
//   state  [B, Hv, Dv, Dk] f32,  z [B, T, Hv * Dv] bf16 (rows may be strided)
//   w      [Dv] bf16 norm weight, eps the norm epsilon
//   state_dst: optional [B, Hv, Dv, Dk] f32 row-contiguous view the new state
//   is written into (its buffer becomes `*out_state`); null allocates.
//
// Writes `*out` [B, T, Hv * Dv] bf16 and `*out_state`. Returns false (message
// on stderr) when off-contract or without Metal; the caller keeps the
// unfused chain. Dk a multiple of 32 (<= 256), Dv = 128, T <= Dv / 4.
extern "C" bool mlx_gdn_fused_step(mlx_array* q, mlx_array* k, mlx_array* v,
                                   mlx_array* g, mlx_array* beta,
                                   mlx_array* state, mlx_array* z,
                                   mlx_array* w, float eps,
                                   mlx_array* state_dst, mlx_array** out,
                                   mlx_array** out_state) {
  if (out) *out = nullptr;
  if (out_state) *out_state = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(q && k && v && g && beta && state && z && w && out && out_state,
           "null argument");
    expect(metal::is_available() && default_device() == Device::gpu,
           "requires the Metal device");
    const auto& qa = *reinterpret_cast<array*>(q);
    const auto& ka = *reinterpret_cast<array*>(k);
    const auto& va = *reinterpret_cast<array*>(v);
    const auto& ga = *reinterpret_cast<array*>(g);
    const auto& ba = *reinterpret_cast<array*>(beta);
    const auto& sa = *reinterpret_cast<array*>(state);
    const auto& za = *reinterpret_cast<array*>(z);
    const auto& wa = *reinterpret_cast<array*>(w);
    expect(qa.ndim() == 4 && ka.ndim() == 4 && va.ndim() == 4 &&
               ga.ndim() == 3 && ba.ndim() == 3,
           "ranks");
    const int B = va.shape(0), T = va.shape(1), Hv = va.shape(2),
              Dv = va.shape(3);
    const int Hk = qa.shape(2), Dk = qa.shape(3);
    expect(T >= 1 && T <= Dv / 4, "T must be 1..=Dv/4");
    expect(Hk > 0 && Hv % Hk == 0, "Hv must be a multiple of Hk");
    expect(qa.shape() == Shape{B, T, Hk, Dk} && ka.shape() == qa.shape() &&
               ga.shape() == Shape{B, T, Hv} && ba.shape() == ga.shape(),
           "shapes");
    expect(qa.dtype() == bfloat16 && ka.dtype() == bfloat16 &&
               va.dtype() == bfloat16 && ba.dtype() == bfloat16 &&
               ga.dtype() == float32,
           "dtypes (q/k/v/beta bf16, g f32)");
    auto dst = check_tail(sa, za, wa, B, T, Hv, Dv, Dk, eps, state_dst);
    std::vector<array> inputs{qa, ka, va, ga, ba, sa, za, wa};
    if (dst) inputs.push_back(*dst);
    auto stream = default_stream(Device::gpu);
    gdn_fused_pipeline(metal::device(stream.device), Dk, Dv, false);
    auto primitive =
        std::make_shared<GdnFusedStep>(stream, Dk, Dv, eps, dst.has_value());
    auto outputs = array::make_arrays({Shape{B, T, Hv * Dv}, sa.shape()},
                                      {bfloat16, float32}, primitive,
                                      std::move(inputs));
    *out = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_state = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_gdn_fused_step: %s\n", e.what());
    return false;
  }
#else
  (void)q; (void)k; (void)v; (void)g; (void)beta; (void)state; (void)z; (void)w;
  (void)eps; (void)state_dst;
  return false;
#endif
}

// The complete fused GDN layer core: `qwen4_gdn_prepare` (4-tap conv + SiLU,
// q/k norm with eps inside the mean, decay/beta gates) + recurrence + gated
// RMSNorm + z gate in one dispatch, B = 1, Dk = Dv = 128, T <= 16, value
// head hv on key head hv % Hk (`tiled_heads`, GGUF rows) or hv / (Hv / Hk).
//
//   qkv     [1, T, 2 Hk Dk + Hv Dv] bf16 (post-mask, pre-conv; rows strided)
//   a, b    [1, T, Hv] bf16 (split views),  conv [W, 4] f32 tap-major
//   history [3, W] bf16,  scale [Hv] f32 (-exp(a_log)),  dt [Hv] f32
//   state, z, w, eps, state_dst as in `mlx_gdn_fused_step`
//   history_dst: optional [3, W] bf16 row-contiguous view `next_history` is
//   written into (the DFlash2 lane's other conv cell row); null allocates.
//
// `outputs`: out [1, T, Hv Dv] bf16, state [1, Hv, Dv, Dk] f32, then the
// tape q [1, T, Hk, Dk], k, v [1, T, Hv, Dv], decay [1, T, Hv] f32,
// beta [1, T, Hv] bf16, next_history [3, W] bf16 — bit-identical to
// `mlx_qwen4_gdn_prepare(mean_eps, beta_input_dtype)` followed by
// `mlx_gdn_fused_step`. Returns false when off-contract.
static bool gdn_fused_complete(mlx_array* qkv, mlx_array* a,
                                       mlx_array* b, mlx_array* conv,
                                       mlx_array* history, mlx_array* scale,
                                       mlx_array* dt, mlx_array* state,
                                       mlx_array* z, mlx_array* w, float eps,
                                       mlx_array* state_dst,
                                       mlx_array* history_dst,
                                       mlx_array** outputs, bool prepared, mlx_array* const* pending = nullptr,
                                       bool tiled_heads = true) {
  if (outputs) {
    for (int i = 0; i < (prepared ? 10 : 8) + (pending ? 1 : 0); ++i) outputs[i] = nullptr;
  }
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(qkv && a && b && conv && history && scale && dt && state && z &&
               w && outputs,
           "null argument");
    expect(metal::is_available() && default_device() == Device::gpu,
           "requires the Metal device");
    const auto& xa = *reinterpret_cast<array*>(qkv);
    const auto& aa = *reinterpret_cast<array*>(a);
    const auto& ba = *reinterpret_cast<array*>(b);
    const auto& ca = *reinterpret_cast<array*>(conv);
    const auto& ha = *reinterpret_cast<array*>(history);
    const auto& sca = *reinterpret_cast<array*>(scale);
    const auto& dta = *reinterpret_cast<array*>(dt);
    const auto& sa = *reinterpret_cast<array*>(state);
    const auto& za = *reinterpret_cast<array*>(z);
    const auto& wa = *reinterpret_cast<array*>(w);
    expect(xa.ndim() == 3 && aa.ndim() == 3 && ba.ndim() == 3 &&
               ha.ndim() == 2 && sa.ndim() == 4,
           "ranks");
    const int T = xa.shape(1), W = xa.shape(2);
    const int Hv = sa.shape(1), Dv = sa.shape(2), Dk = sa.shape(3);
    expect(xa.shape(0) == 1 && sa.shape(0) == 1, "batch must be 1");
    expect(Dk == 128 && Dv == 128, "Dk and Dv must be 128");
    expect(T >= 1 && T <= Dv / 8, "T must be 1..=Dv/8");
    expect(!prepared || T == 8, "prepared operands require eight rows");
    expect((W - Hv * Dv) > 0 && (W - Hv * Dv) % (2 * Dk) == 0, "qkv width");
    const int Hk = (W - Hv * Dv) / (2 * Dk);
    expect(Hv % Hk == 0, "Hv must be a multiple of Hk");
    expect(aa.shape() == Shape{1, T, Hv} && ba.shape() == aa.shape() &&
               ca.size() == size_t(W) * 4 && ha.shape() == Shape{3, W} &&
               sca.size() == size_t(Hv) && dta.size() == size_t(Hv),
           "prologue shapes");
    expect(xa.dtype() == bfloat16 && aa.dtype() == bfloat16 &&
               ba.dtype() == bfloat16 && ha.dtype() == bfloat16 &&
               ca.dtype() == float32 && sca.dtype() == float32 &&
               dta.dtype() == float32,
           "prologue dtypes");
    auto dst = check_tail(sa, za, wa, 1, T, Hv, Dv, Dk, eps, state_dst);
    std::vector<array> inputs{xa, aa, ba, ca, ha, sca, dta, sa, za, wa};
    if (pending) {
      for (int i = 0; i < 6; ++i) {
        expect(pending[i] != nullptr, "null deferred input");
        inputs.push_back(*reinterpret_cast<array*>(pending[i]));
      }
    }
    if (dst) inputs.push_back(*dst);
    if (history_dst) {
      const auto& hd = *reinterpret_cast<array*>(history_dst);
      expect(hd.shape() == ha.shape() && hd.dtype() == bfloat16,
             "history_dst must match the history");
      inputs.push_back(hd);
    }
    if (pending) validate_deferred(inputs, Hk, Dk, Dv, dst.has_value(), history_dst != nullptr);
    auto stream = default_stream(Device::gpu);
    const char* scan_env = std::getenv("MLX_GDN_SCAN8");
    const bool allow_scan8 = scan_env && std::string(scan_env) == "1";
    const char* storage_env = std::getenv("MLX_GDN_T8_STORAGE");
    const bool allow_t8_storage = storage_env && std::string(storage_env) == "1";
    gdn_fused_pipeline(metal::device(stream.device), Dk, Dv, true, allow_scan8 && T == 8, prepared, allow_t8_storage && T == 8, pending != nullptr);
    auto primitive = std::make_shared<GdnFusedComplete>(
        stream, Hk, Dk, Dv, eps, dst.has_value(), history_dst != nullptr, allow_scan8, prepared, allow_t8_storage, pending != nullptr, tiled_heads);
    std::vector<Shape> shapes{Shape{1, T, Hv * Dv}, sa.shape(), Shape{1, T, Hk, Dk},
         Shape{1, T, Hk, Dk}, Shape{1, T, Hv, Dv}, Shape{1, T, Hv},
         Shape{1, T, Hv}, ha.shape()};
    std::vector<Dtype> types{bfloat16, float32, bfloat16, bfloat16, bfloat16,
         float32, bfloat16, bfloat16};
    if (prepared) {
      shapes.push_back(Shape{Hv * Dv * 4}); types.push_back(uint32);
      shapes.push_back(Shape{Hv * Dv / 32 * 8}); types.push_back(float32);
    }
    if (pending) {
      shapes.push_back(xa.shape()); types.push_back(bfloat16);
    }
    auto result = array::make_arrays(std::move(shapes), std::move(types), primitive, std::move(inputs));
    // Publish only after every allocation succeeds. On failure the caller sees
    // the original all-null result, and RAII releases any staged handles.
    std::vector<std::unique_ptr<array>> handles;
    handles.reserve(result.size());
    for (auto& value : result) {
      handles.push_back(std::make_unique<array>(std::move(value)));
    }
    for (size_t i = 0; i < handles.size(); ++i) {
      outputs[i] = reinterpret_cast<mlx_array*>(handles[i].release());
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_gdn_fused_complete: %s\n", e.what());
    return false;
  }
#else
  (void)qkv; (void)a; (void)b; (void)conv; (void)history; (void)scale; (void)dt;
  (void)state; (void)z; (void)w; (void)eps; (void)state_dst; (void)history_dst;
  return false;
#endif
}

extern "C" bool mlx_gdn_fused_complete(mlx_array* qkv, mlx_array* a,
    mlx_array* b, mlx_array* conv, mlx_array* history, mlx_array* scale,
    mlx_array* dt, mlx_array* state, mlx_array* z, mlx_array* w, float eps,
    mlx_array* state_dst, mlx_array* history_dst, mlx_array** outputs) {
  return gdn_fused_complete(qkv, a, b, conv, history, scale, dt, state, z, w,
      eps, state_dst, history_dst, outputs, false);
}

extern "C" bool mlx_gdn_fused_complete_prepared(mlx_array* qkv, mlx_array* a,
    mlx_array* b, mlx_array* conv, mlx_array* history, mlx_array* scale,
    mlx_array* dt, mlx_array* state, mlx_array* z, mlx_array* w, float eps,
    mlx_array* state_dst, mlx_array* history_dst, mlx_array** outputs) {
  return gdn_fused_complete(qkv, a, b, conv, history, scale, dt, state, z, w,
      eps, state_dst, history_dst, outputs, true);
}

// Splash `gdn_decode_batch_phase`: the prior step's tape `pending` (k, v,
// decay, beta, raw inputs, keep int32[1]) is graph input data, folded before
// the window. State and history outputs are the folded state, never the
// window-final one. `outputs` gets one more array, last: the window's raw
// conv inputs [1, T, W] (the next step's pending raw inputs). Both head
// orders: `tiled_heads` (GGUF rows, hv % Hk) or grouped (hv / (Hv / Hk)).
extern "C" bool mlx_gdn_fused_complete_deferred(mlx_array* qkv, mlx_array* a,
    mlx_array* b, mlx_array* conv, mlx_array* history, mlx_array* scale,
    mlx_array* dt, mlx_array* state, mlx_array* z, mlx_array* w, float eps,
    mlx_array* state_dst, mlx_array* history_dst, mlx_array* const* pending,
    bool prepared, bool tiled_heads, mlx_array** outputs) {
  if (!pending) {
    if (outputs) for (int i = 0; i < (prepared ? 10 : 8) + 1; ++i) outputs[i] = nullptr;
    return false;
  }
  return gdn_fused_complete(qkv, a, b, conv, history, scale, dt, state, z, w,
      eps, state_dst, history_dst, outputs, prepared, pending, tiled_heads);
}

// Fused GDN prefill for one prompt window, any T (the prompt counterpart of
// `mlx_gdn_fused_step`, which caps at T <= 32).
//
//   q, k   [1, T, Hk, Dk] bf16 PRE-NORM conv outputs, rows may be strided
//   v      [1, T, Hv, Dv] bf16, rows may be strided
//   a, b   [1, T, Hv] bf16 gate views, rows may be strided
//   a_log, dt_bias [Hv], both f32 or both bf16 (a GGUF load keeps them
//     bf16); the prologue reads each value in its storage dtype and widens it
//     to float, as fused_gdn_gating does
//   state  [1, Hv, Dv, Dk] f32, or null → in-kernel zero init
//   wq, wk [Dk] bf16 folded q/k rms_norm weights (qk_norm_w_q/_k)
//   z      [1, T, Hv * Dv] bf16, rows may be strided
//   w      [Dv] bf16 gated-norm weight; eps its epsilon; qk_eps the q/k norm
//     epsilon (1e-6)
//   tiled: 1 for compact tiled-GGUF heads (hk = hv % Hk), 0 for the standard
//     repeat-expanded map (hk = hv / (Hv / Hk)) on unexpanded q/k.
//
// Writes `*out` [1, T, Hv * Dv] bf16 and `*out_state` [1, Hv, Dv, Dk] f32 —
// bit-identical to the unfused chain (q/k rms_norm_scaled -> fused gating ->
// q/k repeat or tile -> per-step kernel -> gated rms_norm -> swiglu). Returns
// false when off-contract or without Metal; the caller runs the chain.
static bool gdn_fused_prefill_entry(
    bool head_reference, mlx_array* q, mlx_array* k, mlx_array* v,
    mlx_array* a, mlx_array* b, mlx_array* a_log, mlx_array* dt_bias,
    mlx_array* state, mlx_array* wq, mlx_array* wk, mlx_array* z,
    mlx_array* w, float eps, float qk_eps, int tiled, mlx_array** out,
    mlx_array** out_state) {
  if (out) *out = nullptr;
  if (out_state) *out_state = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect_prefill(q && k && v && a && b && a_log && dt_bias && wq && wk && z &&
                       w && out && out_state,
                   "null argument");
    expect_prefill(metal::is_available() && default_device() == Device::gpu,
                   "requires the Metal device");
    const auto& qa = *reinterpret_cast<array*>(q);
    const auto& ka = *reinterpret_cast<array*>(k);
    const auto& va = *reinterpret_cast<array*>(v);
    const auto& aa = *reinterpret_cast<array*>(a);
    const auto& ba = *reinterpret_cast<array*>(b);
    const auto& ala = *reinterpret_cast<array*>(a_log);
    const auto& dta = *reinterpret_cast<array*>(dt_bias);
    const auto& wqa = *reinterpret_cast<array*>(wq);
    const auto& wka = *reinterpret_cast<array*>(wk);
    const auto& za = *reinterpret_cast<array*>(z);
    const auto& wa = *reinterpret_cast<array*>(w);
    expect_prefill(qa.ndim() == 4 && ka.ndim() == 4 && va.ndim() == 4 &&
                       aa.ndim() == 3 && ba.ndim() == 3 && za.ndim() == 3,
                   "ranks");
    const int T = va.shape(1), Hv = va.shape(2), Dv = va.shape(3);
    const int Hk = qa.shape(2), Dk = qa.shape(3);
    expect_prefill(T >= 1, "T");
    expect_prefill(Dk == 128 && Dv == 128, "Dk and Dv must be 128");
    expect_prefill(Hk > 0 && Hv % Hk == 0, "Hv must be a multiple of Hk");
    expect_prefill(qa.shape(0) == 1 && ka.shape(0) == 1 && va.shape(0) == 1 &&
                       aa.shape(0) == 1 && ba.shape(0) == 1 && za.shape(0) == 1,
                   "batch must be 1");
    expect_prefill(ka.shape() == qa.shape() &&
                       aa.shape() == Shape{1, T, Hv} &&
                       ba.shape() == aa.shape() &&
                       za.shape() == Shape{1, T, Hv * Dv},
                   "shapes");
    expect_prefill(qa.strides(3) == 1 && ka.strides(3) == 1 &&
                       va.strides(3) == 1 && qa.strides(2) == Dk &&
                       ka.strides(2) == Dk && va.strides(2) == Dv &&
                       aa.strides(2) == 1 && ba.strides(2) == 1 &&
                       za.strides(2) == 1,
                   "channel strides");
    expect_prefill(qa.dtype() == bfloat16 && ka.dtype() == bfloat16 &&
                       va.dtype() == bfloat16 && aa.dtype() == bfloat16 &&
                       ba.dtype() == bfloat16 && za.dtype() == bfloat16 &&
                       wqa.dtype() == bfloat16 && wka.dtype() == bfloat16 &&
                       wa.dtype() == bfloat16,
                   "dtypes (activations and weights bf16)");
    expect_prefill(wqa.shape() == Shape{Dk} && wka.shape() == Shape{Dk} &&
                       wa.shape() == Shape{Dv},
                   "norm weight shapes");
    expect_prefill(ala.size() == size_t(Hv) && dta.size() == size_t(Hv),
                   "gate scalar shapes");
    expect_prefill(ala.strides(0) == 1 && dta.strides(0) == 1,
                   "gate scalar strides");
    const bool bf16_scalars = ala.dtype() == bfloat16;
    expect_prefill(ala.dtype() == dta.dtype() &&
                       (bf16_scalars || ala.dtype() == float32),
                   "a_log and dt_bias must both be f32 or both bf16");
    expect_prefill(!(head_reference && bf16_scalars),
                   "the head kernel reads f32 a_log / dt_bias");
    std::optional<array> state_opt;
    Shape state_shape{1, Hv, Dv, Dk};
    if (state) {
      const auto& sa = *reinterpret_cast<array*>(state);
      expect_prefill(sa.shape() == state_shape && sa.dtype() == float32,
                     "state must be [1, Hv, Dv, Dk] f32");
      state_opt = sa;
    }
    std::vector<array> inputs{qa, ka, va, aa, ba, ala, dta};
    if (state_opt) {
      inputs.push_back(*state_opt);
    }
    inputs.insert(inputs.end(), {wqa, wka, za, wa});
    auto stream = default_stream(Device::gpu);
    const auto geometry = head_reference ? GdnPrefillGeometry::Head
                                         : GdnPrefillGeometry::Rows;
    gdn_prefill_pipelines(metal::device(stream.device), geometry, Dk, Dv,
                          bf16_scalars);
    // `has_state` keys the primitive AND the input count; when absent the
    // eval binds a scratch row (never read) in place of inputs[7].
    auto primitive = std::make_shared<GdnFusedPrefill>(
        stream, geometry, Dk, Dv, eps, qk_eps, tiled != 0,
        state_opt.has_value(), bf16_scalars);
    auto outputs = array::make_arrays(
        {Shape{1, T, Hv * Dv}, state_shape}, {bfloat16, float32}, primitive,
        std::move(inputs));
    *out = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_state =
        reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_gdn_fused_prefill: %s\n", e.what());
    return false;
  }
#else
  (void)head_reference; (void)q; (void)k; (void)v; (void)a; (void)b;
  (void)a_log; (void)dt_bias; (void)state; (void)wq; (void)wk; (void)z;
  (void)w; (void)eps; (void)qk_eps; (void)tiled;
  return false;
#endif
}

extern "C" bool mlx_gdn_fused_prefill(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* a, mlx_array* b,
    mlx_array* a_log, mlx_array* dt_bias, mlx_array* state, mlx_array* wq,
    mlx_array* wk, mlx_array* z, mlx_array* w, float eps, float qk_eps,
    int tiled, mlx_array** out, mlx_array** out_state) {
  return gdn_fused_prefill_entry(false, q, k, v, a, b, a_log, dt_bias, state,
                                 wq, wk, z, w, eps, qk_eps, tiled, out,
                                 out_state);
}

// The same entry on the one-value-head-per-threadgroup kernel: the bit
// reference of the tests (mlx-core declares it only under cfg(test)).
extern "C" bool mlx_gdn_fused_prefill_head_reference(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* a, mlx_array* b,
    mlx_array* a_log, mlx_array* dt_bias, mlx_array* state, mlx_array* wq,
    mlx_array* wk, mlx_array* z, mlx_array* w, float eps, float qk_eps,
    int tiled, mlx_array** out, mlx_array** out_state) {
  return gdn_fused_prefill_entry(true, q, k, v, a, b, a_log, dt_bias, state,
                                 wq, wk, z, w, eps, qk_eps, tiled, out,
                                 out_state);
}
