// Fused GDN step: the per-step gated delta recurrence of one window with the
// layer's gated RMSNorm and z gate in the same dispatch — and, in the
// "complete" form, the `qwen4_gdn_prepare` prologue (conv + SiLU + q/k norm
// + gates) too (structure after Splash `verify_gdn_fused`, Apache-2.0; see
// THIRD_PARTY_NOTICES). Plain primitives rather than `fast::metal_kernel`
// so `state_out` can alias a caller-owned buffer (the DFlash2 spare blob
// row: a fully accepted verify window then IS the committed state) and the
// split views (`z`, `qkv`, `a`, `b`) are read through their strides.

#include "mlx_common.h"

#include <cstdio>
#include <cstring>
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
};

MTL::ComputePipelineState* gdn_fused_pipeline(metal::Device& device, int dk,
                                              int dv, bool complete) {
  const std::string name =
      complete ? "mlx_node_gdn_fused_complete" : "mlx_node_gdn_fused_step";
  const std::string lib_name =
      name + "_dk" + std::to_string(dk) + "_dv" + std::to_string(dv);
  auto* library = device.get_library(lib_name, [dk, dv, complete] {
    return std::string(metal::utils()) + "\n#define GDN_FUSED_N_PER_T " +
        std::to_string(dk / 32) + "\n#define GDN_FUSED_DV " +
        std::to_string(dv) + "\n#define GDN_FUSED_COMPLETE " +
        (complete ? "1" : "0") + "\n" + kGdnFusedBody;
  });
  auto* pipeline = device.get_kernel(name, library);
  if (pipeline->threadExecutionWidth() != 32 ||
      pipeline->maxTotalThreadsPerThreadgroup() < NS::UInteger(8 * dv)) {
    throw std::runtime_error(
        "gdn fused pipeline cannot hold one value head per threadgroup");
  }
  return pipeline;
}

// Shared host side of both kernels. `Layout` fixes the input order:
//   step:     q, k, v, g, beta, state_in, z, w [, state_dst]
//   complete: qkv, a, b, conv, history, scale, dt, state_in, z, w
//             [, state_dst] [, history_dst]
// Outputs: out, state_out [, q, k, v, decay, beta, next_history]; state_out
// (and, complete, next_history) alias their destinations when given.
class GdnFused : public fast::Custom {
 public:
  GdnFused(Stream stream, bool complete, int dk, int dv, float eps,
           bool has_dst, bool has_history_dst)
      : Custom(stream,
               [](std::vector<array>) -> std::vector<array> {
                 throw std::runtime_error("GdnFused has no fallback");
               }),
        complete_(complete),
        dk_(dk),
        dv_(dv),
        eps_(eps),
        has_dst_(has_dst),
        has_history_dst_(has_history_dst) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GdnFused requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    bridge_testing::record(complete_ ? "gdn_fused_complete" : "gdn_fused_step");
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    auto& encoder = metal::get_command_encoder(stream);
    auto* pipeline = gdn_fused_pipeline(device, dk_, dv_, complete_);

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
    bound.reserve(fixed);
    for (int i = 0; i < fixed; ++i) {
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
    int extra = fixed;
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

    encoder.set_compute_pipeline_state(pipeline);
    int slot = 0;
    for (int i = 0; i < fixed; ++i) {
      encoder.set_input_array(bound[i], slot++);
    }
    for (auto& o : outputs) {
      encoder.set_output_array(o, slot++);
    }
    encoder.set_bytes(params, slot);
    encoder.dispatch_threadgroups(
        MTL::Size(size_t(state.shape(0)) * size_t(params.hv), 1, 1),
        MTL::Size(size_t(8) * size_t(dv_), 1, 1));
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
        has_history_dst_ == o.has_history_dst_;
  }

 protected:
  bool complete_;
  int dk_;
  int dv_;
  float eps_;
  bool has_dst_;
  bool has_history_dst_;
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
                   bool has_dst, bool has_history_dst)
      : GdnFused(stream, true, dk, dv, eps, has_dst, has_history_dst),
        hk_(hk) {}

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& qkv = inputs[0];
    const auto& state = inputs[7];
    const int T = qkv.shape(1), hv = state.shape(1);
    return {Shape{1, T, hv * dv_}, state.shape(), Shape{1, T, hk_, dk_},
            Shape{1, T, hk_, dk_}, Shape{1, T, hv, dv_}, Shape{1, T, hv},
            Shape{1, T, hv}, inputs[4].shape()};
  }

  bool is_equivalent(const Primitive& other) const override {
    return GdnFused::is_equivalent(other) &&
        hk_ == static_cast<const GdnFusedComplete&>(other).hk_;
  }

  DEFINE_NAME(GdnFusedComplete)

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
#endif

}  // namespace

// True when both arrays are evaluated and are the same bytes of the same
// buffer (same offset and size; shapes may differ by a view): the check that
// a verify's exported state really landed in the spare blob row before a
// full accept adopts it.
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
// head hv on key head hv % Hk (tiled GGUF heads).
//
//   qkv     [1, T, 2 Hk Dk + Hv Dv] bf16 (post-mask, pre-conv; rows strided)
//   a, b    [1, T, Hv] bf16 (split views),  conv [W, 4] f32 tap-major
//   history [3, W] bf16,  scale [Hv] f32 (-exp(a_log)),  dt [Hv] f32
//   state, z, w, eps, state_dst as in `mlx_gdn_fused_step`
//   history_dst: optional [3, W] bf16 row-contiguous view `next_history` is
//   written into (the spare conv blob row); null allocates.
//
// `outputs`: out [1, T, Hv Dv] bf16, state [1, Hv, Dv, Dk] f32, then the
// tape q [1, T, Hk, Dk], k, v [1, T, Hv, Dv], decay [1, T, Hv] f32,
// beta [1, T, Hv] bf16, next_history [3, W] bf16 — bit-identical to
// `mlx_qwen4_gdn_prepare(mean_eps, beta_input_dtype)` followed by
// `mlx_gdn_fused_step`. Returns false when off-contract.
extern "C" bool mlx_gdn_fused_complete(mlx_array* qkv, mlx_array* a,
                                       mlx_array* b, mlx_array* conv,
                                       mlx_array* history, mlx_array* scale,
                                       mlx_array* dt, mlx_array* state,
                                       mlx_array* z, mlx_array* w, float eps,
                                       mlx_array* state_dst,
                                       mlx_array* history_dst,
                                       mlx_array** outputs) {
  if (outputs) {
    for (int i = 0; i < 8; ++i) outputs[i] = nullptr;
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
    if (dst) inputs.push_back(*dst);
    if (history_dst) {
      const auto& hd = *reinterpret_cast<array*>(history_dst);
      expect(hd.shape() == ha.shape() && hd.dtype() == bfloat16,
             "history_dst must match the history");
      inputs.push_back(hd);
    }
    auto stream = default_stream(Device::gpu);
    gdn_fused_pipeline(metal::device(stream.device), Dk, Dv, true);
    auto primitive = std::make_shared<GdnFusedComplete>(
        stream, Hk, Dk, Dv, eps, dst.has_value(), history_dst != nullptr);
    auto result = array::make_arrays(
        {Shape{1, T, Hv * Dv}, sa.shape(), Shape{1, T, Hk, Dk},
         Shape{1, T, Hk, Dk}, Shape{1, T, Hv, Dv}, Shape{1, T, Hv},
         Shape{1, T, Hv}, ha.shape()},
        {bfloat16, float32, bfloat16, bfloat16, bfloat16, float32, bfloat16,
         bfloat16},
        primitive, std::move(inputs));
    for (int i = 0; i < 8; ++i) {
      outputs[i] = reinterpret_cast<mlx_array*>(new array(std::move(result[i])));
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
