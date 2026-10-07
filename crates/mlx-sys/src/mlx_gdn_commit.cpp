// Fused DFlash2 GDN commit: a few Metal dispatches replay the accepted
// prefix of every linear layer's recorded verify tape into the packed
// recurrent and conv state blobs (Splash `verify_gdn_commit`, Apache-2.0; see
// THIRD_PARTY_NOTICES). The arithmetic is `gated_delta_replay`'s f32 carry
// plus `replay_conv_state`'s row copy, so the result is bit-identical to the
// per-layer replays it replaces.
//
// Metal binds at most 31 buffers per dispatch and a layer's tape is 5 arrays,
// so one primitive evaluation encodes the layers in groups of 5 (10
// dispatches for 48 layers) into one pair of output blobs, concurrently.

#include "mlx_common.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"
#include "mlx_test_counters.h"
#endif

namespace {

constexpr int kTapeArraysPerLayer = 5;
constexpr int kLayersPerDispatch = 5;  // GDN_COMMIT_GROUP in the kernel
// rec_in, conv_in, rec_next, conv_next precede the per-layer tapes.
constexpr int kFixedInputs = 4;

#ifdef MLX_NODE_METAL_ENABLED
using namespace mlx::core;

const char* kGdnCommitBody =
#include "metal/common/gdn_commit.metal.inc"
    ;

// Mirrors `GdnCommitParams` in the kernel source.
struct GdnCommitParams {
  int keep;
  int window;
  int layer_base;
  int group;
  int hk;
  int hv;
  int dk;
  int dv;
  int conv_rows;
  int conv_width;
  int qkv_stride[kLayersPerDispatch];
};

MTL::ComputePipelineState* gdn_commit_pipeline(metal::Device& device, int dk) {
  const std::string name = "mlx_node_gdn_commit";
  const std::string lib_name = name + "_dk" + std::to_string(dk);
  auto* library = device.get_library(lib_name, [dk] {
    return std::string(metal::utils()) + "\n#define GDN_COMMIT_N_PER_T " +
        std::to_string(dk / 32) + "\n" + kGdnCommitBody;
  });
  auto* pipeline = device.get_kernel(name, library);
  if (pipeline->threadExecutionWidth() != 32 ||
      pipeline->maxTotalThreadsPerThreadgroup() < 128) {
    throw std::runtime_error("gdn commit pipeline violates its simd contract");
  }
  return pipeline;
}

// Inputs: rec_in, conv_in, rec_next, conv_next, then per layer k, v, g,
// beta, qkv. Outputs: rec_out, conv_out — written IN PLACE into the `next`
// blobs (ping-pong): the outputs alias those inputs' buffers, so the caller
// must hold no reader of the `next` contents (it swaps parity after the
// call). The encoder sees the `next` buffers as outputs, which orders the
// write after every earlier read of them.
class GdnCommit final : public fast::Custom {
 public:
  GdnCommit(Stream stream, GdnCommitParams params)
      : Custom(stream,
               [](std::vector<array>) -> std::vector<array> {
                 throw std::runtime_error("GdnCommit has no fallback");
               }),
        params_(params) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GdnCommit requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    bridge_testing::record("gdn_commit_all");
    const auto& rec_in = inputs[0];
    const auto& conv_in = inputs[1];
    const auto& rec_next = inputs[2];
    const auto& conv_next = inputs[3];
    auto& rec_out = outputs[0];
    auto& conv_out = outputs[1];

    const int layers = rec_in.shape(0);
    // Layouts are only known now: lazy arrays report contiguous strides
    // until they are evaluated.
    for (const auto* blob : {&rec_in, &conv_in, &rec_next, &conv_next}) {
      if (!blob->flags().row_contiguous || blob->offset() != 0) {
        throw std::runtime_error(
            "GdnCommit: state blobs must be whole row-contiguous buffers");
      }
    }
    if (rec_in.buffer().ptr() == rec_next.buffer().ptr() ||
        conv_in.buffer().ptr() == conv_next.buffer().ptr()) {
      throw std::runtime_error("GdnCommit: current and next blobs must differ");
    }
    rec_out.copy_shared_buffer(rec_next);
    conv_out.copy_shared_buffer(conv_next);
    for (int l = 0; l < layers; ++l) {
      const size_t at = kFixedInputs + size_t(l) * kTapeArraysPerLayer;
      for (int a = 0; a < kTapeArraysPerLayer - 1; ++a) {
        if (!inputs[at + a].flags().row_contiguous) {
          throw std::runtime_error("GdnCommit: k/v/g/beta must be row-contiguous");
        }
      }
      const auto& qkv = inputs[at + 4];
      if (qkv.strides(2) != 1 || qkv.strides(1) < qkv.shape(2)) {
        throw std::runtime_error("GdnCommit: qkv rows need a contiguous last axis");
      }
    }
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    auto& encoder = metal::get_command_encoder(stream);
    auto* pipeline = gdn_commit_pipeline(device, params_.dk);

    // Every group writes a disjoint slice of the output blobs: no barrier
    // between the group dispatches.
    auto concurrent = encoder.start_concurrent();
    for (int base = 0; base < layers; base += kLayersPerDispatch) {
      GdnCommitParams params = params_;
      params.layer_base = base;
      params.group = std::min(kLayersPerDispatch, layers - base);
      encoder.set_compute_pipeline_state(pipeline);
      encoder.set_input_array(rec_in, 0);
      encoder.set_input_array(conv_in, 1);
      encoder.set_output_array(rec_out, 2);
      encoder.set_output_array(conv_out, 3);
      for (int j = 0; j < kLayersPerDispatch; ++j) {
        // Slots past the group's last layer rebind that layer (never read).
        const int l = base + std::min(j, params.group - 1);
        const size_t at = kFixedInputs + size_t(l) * kTapeArraysPerLayer;
        for (int a = 0; a < kTapeArraysPerLayer; ++a) {
          encoder.set_input_array(inputs[at + a], 5 + 5 * j + a);
        }
        params.qkv_stride[j] = int(inputs[at + 4].strides(1));
      }
      encoder.set_bytes(params, 4);
      encoder.dispatch_threads(
          MTL::Size(32, params_.dv, size_t(params.group) * params_.hv),
          MTL::Size(32, 4, 1));
    }
  }

  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&,
                         const std::vector<array>&) override {
    throw std::runtime_error("GdnCommit is inference-only");
  }

  DEFINE_NAME(GdnCommit)

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const GdnCommit&>(other);
    return std::memcmp(&params_, &o.params_, sizeof(params_)) == 0;
  }

 private:
  GdnCommitParams params_;
};

void expect(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(std::string("mlx_gdn_commit_all: ") + what);
  }
}
#endif

}  // namespace

// Fused GDN commit over `layers` linear layers.
//
//   rec_in   [L, Hv, Dv, Dk] f32  — pre-verify recurrent blob
//   conv_in  [L, K-1, W]     bf16 — pre-verify conv history blob
//   k[l]     [1, S, Hk, Dk]  bf16, v[l] [1, S, Hv, Dv] bf16,
//   g[l]     [1, S, Hv]      f32,  beta[l] [1, S, Hv] bf16,
//   qkv[l]   [1, S, W]       bf16 (last axis contiguous, rows may be strided)
//   keep: accepted tokens (1..=S)
//
//   rec_next / conv_next: the spare blobs the result is written into (their
//   buffers become `*out_rec` / `*out_conv`; the caller swaps parity)
//
// Writes `*out_rec` / `*out_conv` (same shapes as the inputs). Returns false
// (message on stderr) on any contract violation or without Metal; the caller
// keeps the per-layer replay path.
extern "C" bool mlx_gdn_commit_all(mlx_array* rec_in, mlx_array* conv_in,
                                   mlx_array* rec_next, mlx_array* conv_next,
                                   int32_t layers, mlx_array* const* k,
                                   mlx_array* const* v, mlx_array* const* g,
                                   mlx_array* const* beta,
                                   mlx_array* const* qkv, int32_t keep,
                                   mlx_array** out_rec, mlx_array** out_conv) {
  if (out_rec) *out_rec = nullptr;
  if (out_conv) *out_conv = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(rec_in && conv_in && rec_next && conv_next && k && v && g && beta &&
               qkv && out_rec && out_conv && layers > 0,
           "null argument");
    expect(metal::is_available() && default_device() == Device::gpu,
           "requires the Metal device");
    const auto& rec = *reinterpret_cast<array*>(rec_in);
    const auto& conv = *reinterpret_cast<array*>(conv_in);
    expect(rec.ndim() == 4 && conv.ndim() == 3 && rec.shape(0) == layers &&
               conv.shape(0) == layers,
           "state blobs must be [L, Hv, Dv, Dk] and [L, K-1, W]");
    expect(rec.dtype() == float32 && conv.dtype() == bfloat16,
           "state blobs must be f32 recurrent, bf16 conv");
    const auto& rec_spare = *reinterpret_cast<array*>(rec_next);
    const auto& conv_spare = *reinterpret_cast<array*>(conv_next);
    expect(rec_spare.shape() == rec.shape() && conv_spare.shape() == conv.shape() &&
               rec_spare.dtype() == float32 && conv_spare.dtype() == bfloat16,
           "next blobs must match the current blobs");
    const int hv = rec.shape(1), dv = rec.shape(2), dk = rec.shape(3);
    const int conv_rows = conv.shape(1), width = conv.shape(2);
    expect(dk % 32 == 0 && dk <= 256 && dv % 4 == 0 && dv > 0,
           "Dk must be a multiple of 32 (<= 256), Dv of 4");
    std::vector<array> inputs;
    inputs.reserve(kFixedInputs + size_t(layers) * kTapeArraysPerLayer);
    inputs.insert(inputs.end(), {rec, conv, rec_spare, conv_spare});
    int window = -1, hk = -1;
    for (int l = 0; l < layers; ++l) {
      expect(k[l] && v[l] && g[l] && beta[l] && qkv[l], "null tape array");
      const auto& kl = *reinterpret_cast<array*>(k[l]);
      const auto& vl = *reinterpret_cast<array*>(v[l]);
      const auto& gl = *reinterpret_cast<array*>(g[l]);
      const auto& bl = *reinterpret_cast<array*>(beta[l]);
      const auto& ql = *reinterpret_cast<array*>(qkv[l]);
      expect(kl.ndim() == 4 && vl.ndim() == 4 && gl.ndim() == 3 &&
                 bl.ndim() == 3 && ql.ndim() == 3,
             "tape ranks");
      if (l == 0) {
        window = kl.shape(1);
        hk = kl.shape(2);
      }
      expect(kl.shape(0) == 1 && vl.shape(0) == 1 && gl.shape(0) == 1 &&
                 bl.shape(0) == 1 && ql.shape(0) == 1,
             "tape batch must be 1");
      expect(kl.shape(1) == window && vl.shape(1) == window &&
                 gl.shape(1) == window && bl.shape(1) == window &&
                 ql.shape(1) == window,
             "tape window lengths differ");
      expect(kl.shape(2) == hk && kl.shape(3) == dk && vl.shape(2) == hv &&
                 vl.shape(3) == dv && gl.shape(2) == hv && bl.shape(2) == hv &&
                 ql.shape(2) == width,
             "tape geometry does not match the state blobs");
      expect(kl.dtype() == bfloat16 && vl.dtype() == bfloat16 &&
                 gl.dtype() == float32 && bl.dtype() == bfloat16 &&
                 ql.dtype() == bfloat16,
             "tape dtypes (k/v/beta/qkv bf16, g f32)");
      // The kernel indexes k/v/g/beta row-major. Production tapes (the
      // gdn_prepare outputs, evaluated by acceptance) already are; a lazy or
      // strided tape goes through `contiguous`, a no-op when it already is.
      for (const auto* a : {&kl, &vl, &gl, &bl}) {
        inputs.push_back(a->is_available() && a->flags().row_contiguous
                             ? *a
                             : contiguous(*a));
      }
      inputs.push_back(ql);
    }
    expect(hk > 0 && hv % hk == 0, "Hv must be a multiple of Hk");
    expect(keep >= 1 && keep <= window, "keep out of the recorded window");
    GdnCommitParams params{};
    params.keep = keep;
    params.window = window;
    params.hk = hk;
    params.hv = hv;
    params.dk = dk;
    params.dv = dv;
    params.conv_rows = conv_rows;
    params.conv_width = width;
    auto stream = default_stream(Device::gpu);
    auto primitive = std::make_shared<GdnCommit>(stream, params);
    auto outputs = array::make_arrays({rec.shape(), conv.shape()},
                                      {float32, bfloat16}, primitive,
                                      std::move(inputs));
    *out_rec = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_conv = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_gdn_commit_all: %s\n", e.what());
    return false;
  }
#else
  (void)rec_in; (void)conv_in; (void)rec_next; (void)conv_next; (void)layers; (void)k; (void)v; (void)g;
  (void)beta; (void)qkv; (void)keep;
  return false;
#endif
}
