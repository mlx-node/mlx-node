// MLX `Custom` primitives for fused Inkling-2-Mini Metal kernels
// (`src/metal/inkling2/inkling2_ops.metal`, compiled into `paged_attn.metallib`):
//
//   - `Inkling2GateSink`:            Y = O * sigmoid(gate) * sigmoid(lse - sink) (zero-copy strided gate)
//   - `Inkling2Situ`:                Y = softcap(gate*s, c1) * sigmoid(gate*s) * softcap(up*s, c2)
//   - `Inkling2LoopBlendRmsNorm`:    FP32 no-FMA blend + sublayer RMSNorm -> BF16/F32 (2D/3D strided)
//   - `Inkling2ShortConvK4`:         single-seq static/dynamic k=4 ShortConv + rolling [3, C] state
//   - `Inkling2QrClip`:              strided Q & R L2-clip + optional global tau -> Q [T,H,D], R [T,H,d_rel] F32
//   - `Inkling2ShortConvK4KvPrep`:   fused single/ragged static/dynamic K+V ShortConv + K L2-clip + V RMSNorm
//   - `Inkling2BlockAttnRes`:        fused Block AttnRes over K<=8 sources (+ optional addend & RMSNorm)
//   - `Inkling2PairRmsNorm`:         fused [rms(dest), rms(msg)] -> [S, T, 2*D] FP32 (S=1 or S=3)
//   - `Inkling2ShortConvK4Ragged`:   multi-seq ragged k=4 ShortConv + optional FP32 residual epilogue
//   - `Inkling2MoeRoute`:            fused sigmoid + bias + top-K + norm + down_scale -> int32/f32
//   - `Inkling2MoeCombine`:          fused optional inv_order unsort + FP32 weighted sum -> BF16/F32
//   - `Inkling2LmHeadSoftcap`:       fused vocab slice + optional bias + softcap -> F32
//   - `Inkling2LmHeadSoftcapArgmax`: fused vocab slice + optional bias + softcap + 2-stage GPU argmax -> int32

#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx_paged_metallib.h"
#endif

namespace mlx::core::inkling2_ops {

uint32_t resolved_attn_res_vec4_max_t() {
  static const uint32_t kMaxT = []() -> uint32_t {
    if (const char* env = std::getenv("MLX_INKLING2_ATTN_RES_VEC4_MAX_T")) {
      const int v = std::atoi(env);
      return v > 0 ? static_cast<uint32_t>(v) : 0u;
    }
    return 64u;
  }();
  return kMaxT;
}

namespace {

struct Inkling2GateSinkParams {
  uint32_t rows;
  uint32_t head_dim;
  uint32_t num_heads;
  uint32_t gate_row_stride;
  uint32_t gate_is_f32;
  uint32_t sink_is_f32;
};
static_assert(sizeof(Inkling2GateSinkParams) == 24);

struct Inkling2SituParams {
  uint32_t num_rows;
  uint32_t inter_dim;
  float gate_softcap;
  float up_softcap;
  uint32_t interleaved;
  uint32_t has_outer_scale;
  uint32_t outer_scale_is_f32;
  uint32_t _pad;
};
static_assert(sizeof(Inkling2SituParams) == 32);

struct Inkling2LoopBlendParams {
  uint32_t rows;
  uint32_t dim;
  uint32_t num_streams;
  float eps;
  uint32_t dest_row_stride;
  uint32_t dest_stream_stride;
  uint32_t msg_row_stride;
  uint32_t msg_stream_stride;
  uint32_t gain_row_stride;
  uint32_t gain_stream_stride;
  uint32_t forget_row_stride;
  uint32_t forget_stream_stride;
  uint32_t weight_stream_stride;
  uint32_t flags;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(Inkling2LoopBlendParams) == 64);

struct Inkling2SconvK4Params {
  uint32_t seq_len;
  uint32_t channels;
  uint32_t is_dynamic;
  uint32_t num_kv_heads;
  uint32_t weight_is_f32;
  uint32_t history_is_f32;
  uint32_t dyn_is_f32;
  uint32_t x_row_stride;
};
static_assert(sizeof(Inkling2SconvK4Params) == 32);

struct Inkling2QrClipParams {
  uint32_t seq_len;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t d_rel;
  uint32_t q_row_stride;
  uint32_t r_row_stride;
  float qk_thresh;
  float rel_thresh;
  float eps;
  uint32_t flags;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(Inkling2QrClipParams) == 48);

struct Inkling2SconvK4KvPrepParams {
  uint32_t seq_len;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t num_seqs;
  uint32_t k_row_stride;
  uint32_t v_row_stride;
  float qk_thresh;
  float v_eps;
  float l2_eps;
  uint32_t is_dynamic;
  uint32_t packed_hist;
  uint32_t flags;
  uint32_t k_hist_f32_mask;
  uint32_t v_hist_f32_mask;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t k_hist_strides[4];
  uint32_t v_hist_strides[4];
};
static_assert(sizeof(Inkling2SconvK4KvPrepParams) == 96);

struct Inkling2BlockAttnResParams {
  uint32_t rows;
  uint32_t dim;
  uint32_t num_sources;
  uint32_t num_queries;
  float attn_res_eps;
  float norm_eps;
  uint32_t src_f32_mask;
  uint32_t flags;
  uint32_t src_strides[8];
  uint32_t addend_stride;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(sizeof(Inkling2BlockAttnResParams) == 80);

struct Inkling2PairRmsNormParams {
  uint32_t rows;
  uint32_t dim;
  uint32_t num_streams;
  float eps;
  uint32_t dest_row_stride;
  uint32_t dest_stream_stride;
  uint32_t msg_row_stride;
  uint32_t msg_stream_stride;
  uint32_t dest_is_f32;
  uint32_t msg_is_f32;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(Inkling2PairRmsNormParams) == 48);

struct Inkling2SconvK4RaggedParams {
  uint32_t seq_len;
  uint32_t channels;
  uint32_t num_seqs;
  uint32_t is_dynamic;
  uint32_t num_kv_heads;
  uint32_t packed_hist;
  uint32_t x_row_stride;
  uint32_t res_a_stride;
  uint32_t res_b_stride;
  uint32_t flags;
  uint32_t hist_f32_mask;
  uint32_t _pad;
};
static_assert(sizeof(Inkling2SconvK4RaggedParams) == 48);

struct Inkling2MoeRouteParams {
  uint32_t rows;
  uint32_t num_experts;
  uint32_t top_k;
  float route_scale;
  uint32_t flags;
  uint32_t logits_stride;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(Inkling2MoeRouteParams) == 32);

struct Inkling2MoeCombineParams {
  uint32_t rows;
  uint32_t top_k;
  uint32_t dim;
  uint32_t flags;
};
static_assert(sizeof(Inkling2MoeCombineParams) == 16);

struct Inkling2LmHeadSoftcapParams {
  uint32_t rows;
  uint32_t out_cols;
  uint32_t logits_stride;
  uint32_t bias_stride;
  float softcap;
  uint32_t num_splits;
  uint32_t flags;
  uint32_t _pad;
};
static_assert(sizeof(Inkling2LmHeadSoftcapParams) == 32);

struct Inkling2MarkovGateParams {
  uint32_t rows;
  uint32_t rank;
  uint32_t raw_stride;
  uint32_t ctx_stride;
  uint32_t state_stride;
  uint32_t flags;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(Inkling2MarkovGateParams) == 32);

struct Inkling2MoeRouterProjParams {
  uint32_t rows;
  uint32_t dim;
  uint32_t num_experts;
  uint32_t x_stride;
  uint32_t x_is_f32;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(sizeof(Inkling2MoeRouterProjParams) == 32);

struct Inkling2LoopCtrlDownParams {
  uint32_t rows;
  uint32_t dim;
  uint32_t rank;
  uint32_t num_streams;
  float eps;
  uint32_t dest_row_stride;
  uint32_t dest_stream_stride;
  uint32_t msg_row_stride;
  uint32_t msg_stream_stride;
  uint32_t flags;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(Inkling2LoopCtrlDownParams) == 48);

struct Inkling2LoopCtrlUpBlendParams {
  uint32_t rows;
  uint32_t dim;
  uint32_t rank;
  uint32_t num_streams;
  uint32_t num_d_tiles;
  uint32_t dest_row_stride;
  uint32_t dest_stream_stride;
  uint32_t msg_row_stride;
  uint32_t msg_stream_stride;
  uint32_t flags;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(Inkling2LoopCtrlUpBlendParams) == 48);

struct Inkling2RmsNormFromSqParams {
  uint32_t rows;
  uint32_t dim;
  uint32_t num_streams;
  uint32_t num_d_tiles;
  float eps;
  uint32_t weight_stream_stride;
  uint32_t weight_is_f32;
  uint32_t _pad;
};
static_assert(sizeof(Inkling2RmsNormFromSqParams) == 32);

struct Inkling2MarkovFoldParams {
  uint32_t batch;
  uint32_t out_cols;
  uint32_t rank;
  uint32_t latent_stride;
  uint32_t base_stride;
  float softcap;
  uint32_t num_splits;
  uint32_t flags;
};

struct Inkling2Bf16LinearParams {
  uint32_t rows;
  uint32_t out_dim;
  uint32_t in_dim;
  uint32_t num_splits;
  float softcap;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(sizeof(Inkling2MarkovFoldParams) == 32);

inline bool is_bf16_or_f32(Dtype dt) {
  return dt == bfloat16 || dt == float32;
}

inline bool is_sidecar_float(Dtype dt) {
  return dt == bfloat16 || dt == float16 || dt == float32;
}

inline uint32_t sidecar_dtype_code(Dtype dt) {
  if (dt == float32) return 1u;
  if (dt == float16) return 2u;
  return 0u;
}

#ifdef MLX_NODE_METAL_ENABLED
array ensure_row_contiguous(
    const array& input,
    Stream s,
    metal::CommandEncoder& enc) {
  if (input.flags().row_contiguous) {
    return input;
  }
  array out(input.shape(), input.dtype(), nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(input, out, CopyType::General, s);
  enc.add_temporary(out);
  return out;
}

array ensure_row_contiguous_aligned(
    const array& input,
    size_t byte_align,
    Stream s,
    metal::CommandEncoder& enc) {
  if (input.flags().row_contiguous && (input.offset() % byte_align) == 0) {
    return input;
  }
  array out(input.shape(), input.dtype(), nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(input, out, CopyType::General, s);
  enc.add_temporary(out);
  return out;
}

std::pair<array, uint32_t> ensure_2d_strided(
    const array& input,
    uint32_t rows,
    uint32_t cols,
    Stream s,
    metal::CommandEncoder& enc) {
  if (input.flags().row_contiguous) {
    return {input, cols};
  }
  if (input.ndim() == 2 &&
      static_cast<uint32_t>(input.shape(0)) == rows &&
      static_cast<uint32_t>(input.shape(1)) == cols &&
      input.strides(1) == 1 &&
      input.strides(0) >= 0) {
    return {input, static_cast<uint32_t>(input.strides(0))};
  }
  if (input.ndim() == 1 && rows == 1u &&
      static_cast<uint32_t>(input.shape(0)) == cols &&
      input.strides(0) == 1) {
    return {input, cols};
  }
  return {ensure_row_contiguous(input, s, enc), cols};
}

std::tuple<array, uint32_t, uint32_t> ensure_3d_stream_strided(
    const array& input,
    uint32_t streams,
    uint32_t rows,
    uint32_t cols,
    Stream s,
    metal::CommandEncoder& enc) {
  if (input.ndim() == 2 &&
      static_cast<uint32_t>(input.shape(0)) == rows &&
      static_cast<uint32_t>(input.shape(1)) == cols) {
    auto [arr, rs] = ensure_2d_strided(input, rows, cols, s, enc);
    return {arr, rs, 0u};
  }
  if (input.ndim() == 3 &&
      (static_cast<uint32_t>(input.shape(0)) == streams || input.shape(0) == 1) &&
      static_cast<uint32_t>(input.shape(1)) == rows &&
      static_cast<uint32_t>(input.shape(2)) == cols) {
    const bool bcast = (input.shape(0) == 1);
    if (input.strides(2) == 1 &&
        input.strides(1) >= 0 &&
        (bcast || input.strides(0) >= 0)) {
      return {
          input,
          static_cast<uint32_t>(input.strides(1)),
          bcast ? 0u : static_cast<uint32_t>(input.strides(0))};
    }
    array c = ensure_row_contiguous(input, s, enc);
    return {c, cols, bcast ? 0u : rows * cols};
  }
  array c = ensure_row_contiguous(input, s, enc);
  return {c, cols, streams <= 1u ? 0u : rows * cols};
}
#endif

// ============================================================================
// 1. Inkling2GateSink
// ============================================================================
class Inkling2GateSink : public fast::Custom {
 public:
  explicit Inkling2GateSink(Stream s, Dtype out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2GateSink requires Metal");
            }),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2GateSink)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2GateSink&>(other);
    return out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2GateSink requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    array o = ensure_row_contiguous(inputs[0], s, enc);
    array lse = ensure_row_contiguous(inputs[2], s, enc);
    array sink = ensure_row_contiguous(inputs[3], s, enc);

    const uint32_t t = static_cast<uint32_t>(o.shape(0));
    const uint32_t num_heads = static_cast<uint32_t>(o.shape(1));
    const uint32_t head_dim = static_cast<uint32_t>(o.shape(2));
    const uint32_t rows = t * num_heads;

    const uint32_t gate_cols = num_heads * head_dim;
    uint32_t gate_row_stride = gate_cols;
    array gate = inputs[1];
    if (gate.flags().row_contiguous) {
      gate_row_stride = gate_cols;
    } else if (
        gate.ndim() == 2 &&
        static_cast<uint32_t>(gate.shape(0)) == t &&
        static_cast<uint32_t>(gate.shape(1)) == gate_cols &&
        gate.strides(1) == 1 &&
        gate.strides(0) >= 0) {
      gate_row_stride = static_cast<uint32_t>(gate.strides(0));
    } else {
      gate = ensure_row_contiguous(gate, s, enc);
      gate_row_stride = gate_cols;
    }

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    Inkling2GateSinkParams params{
        rows,
        head_dim,
        num_heads,
        gate_row_stride,
        gate.dtype() == float32 ? 1u : 0u,
        sink.dtype() == float32 ? 1u : 0u};

    const char* kname = out_dtype_ == float32 ? "inkling2_gate_sink_f32"
                                              : "inkling2_gate_sink_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "gate_sink", kname));
    enc.set_input_array(o, 0);
    enc.set_input_array(gate, 1);
    enc.set_input_array(lse, 2);
    enc.set_input_array(sink, 3);
    enc.set_output_array(out, 4);
    enc.set_bytes(params, 5);
    const size_t tpg = std::min<size_t>(256, std::max<size_t>(32, head_dim));
    enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(tpg, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2GateSink requires Metal");
#endif
  }

 private:
  Dtype out_dtype_;
};

// ============================================================================
// 2. Inkling2Situ
// ============================================================================
class Inkling2Situ : public fast::Custom {
 public:
  Inkling2Situ(
      Stream s,
      float gate_softcap,
      float up_softcap,
      bool interleaved,
      bool has_outer_scale,
      Dtype out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2Situ requires Metal");
            }),
        gate_softcap_(gate_softcap),
        up_softcap_(up_softcap),
        interleaved_(interleaved),
        has_outer_scale_(has_outer_scale),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2Situ)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2Situ&>(other);
    return gate_softcap_ == rhs.gate_softcap_ &&
        up_softcap_ == rhs.up_softcap_ &&
        interleaved_ == rhs.interleaved_ &&
        has_outer_scale_ == rhs.has_outer_scale_ &&
        out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() /= 2;
    return {s};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2Situ requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    array gate_up = ensure_row_contiguous(inputs[0], s, enc);
    std::optional<array> outer_arr;
    std::optional<array> idx_arr;
    uint32_t outer_is_f32 = 1u;
    if (has_outer_scale_) {
      outer_arr = ensure_row_contiguous(inputs[1], s, enc);
      idx_arr = ensure_row_contiguous(inputs[2], s, enc);
      outer_is_f32 = outer_arr->dtype() == float32 ? 1u : 0u;
    }

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const uint32_t inter_dim = static_cast<uint32_t>(gate_up.shape(-1) / 2);
    const uint32_t num_rows =
        static_cast<uint32_t>(gate_up.size() / (static_cast<size_t>(inter_dim) * 2));

    const float dummy = 0.0f;
    const int32_t dummy_idx = 0;

    const char* kname =
        out_dtype_ == float32 ? "inkling2_situ_f32" : "inkling2_situ_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "situ", kname));
    enc.set_input_array(gate_up, 0);
    if (has_outer_scale_) {
      enc.set_input_array(*outer_arr, 1);
      enc.set_input_array(*idx_arr, 2);
    } else {
      enc.set_bytes(dummy, 1);
      enc.set_bytes(dummy_idx, 2);
    }
    Inkling2SituParams params{
        num_rows,
        inter_dim,
        gate_softcap_,
        up_softcap_,
        interleaved_ ? 1u : 0u,
        has_outer_scale_ ? 1u : 0u,
        outer_is_f32,
        0u};
    enc.set_output_array(out, 3);
    enc.set_bytes(params, 4);
    const size_t tgx = 256;
    const size_t gx = (inter_dim + tgx - 1) / tgx;
    enc.dispatch_threadgroups(MTL::Size(gx, num_rows, 1), MTL::Size(tgx, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2Situ requires Metal");
#endif
  }

 private:
  float gate_softcap_;
  float up_softcap_;
  bool interleaved_;
  bool has_outer_scale_;
  Dtype out_dtype_;
};

// ============================================================================
// 3. Inkling2LoopBlendRmsNorm (2D/3D strided zero-copy)
// ============================================================================
class Inkling2LoopBlendRmsNorm : public fast::Custom {
 public:
  Inkling2LoopBlendRmsNorm(Stream s, float eps, Dtype out_dtype, Shape out_shape)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2LoopBlendRmsNorm requires Metal");
            }),
        eps_(eps),
        out_dtype_(out_dtype),
        out_shape_(std::move(out_shape)) {}

  DEFINE_NAME(Inkling2LoopBlendRmsNorm)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2LoopBlendRmsNorm&>(other);
    return eps_ == rhs.eps_ && out_dtype_ == rhs.out_dtype_ &&
        out_shape_ == rhs.out_shape_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {out_shape_};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2LoopBlendRmsNorm requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t dim = static_cast<uint32_t>(out_shape_.back());
    const uint32_t num_streams =
        out_shape_.size() == 3 ? static_cast<uint32_t>(out_shape_[0]) : 1u;
    const uint32_t rows = out_shape_.size() == 3
        ? static_cast<uint32_t>(out_shape_[1])
        : static_cast<uint32_t>(outputs[0].size() / dim);

    auto [dest, dest_rs, dest_ss] =
        ensure_3d_stream_strided(inputs[0], num_streams, rows, dim, s, enc);
    auto [msg, msg_rs, msg_ss] =
        ensure_3d_stream_strided(inputs[1], num_streams, rows, dim, s, enc);
    auto [gain, gain_rs, gain_ss] =
        ensure_3d_stream_strided(inputs[2], num_streams, rows, dim, s, enc);
    auto [forget, forget_rs, forget_ss] =
        ensure_3d_stream_strided(inputs[3], num_streams, rows, dim, s, enc);
    array weight = ensure_row_contiguous(inputs[4], s, enc);
    const uint32_t weight_ss =
        (weight.ndim() == 2 && weight.shape(0) == static_cast<int>(num_streams) && num_streams > 1u)
        ? dim
        : 0u;

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const uint32_t flags =
        (dest.dtype() == float32 ? 1u : 0u) |
        ((msg.dtype() == float32 ? 1u : 0u) << 1u) |
        ((gain.dtype() == float32 ? 1u : 0u) << 2u) |
        ((forget.dtype() == float32 ? 1u : 0u) << 3u) |
        ((weight.dtype() == float32 ? 1u : 0u) << 4u);

    Inkling2LoopBlendParams params{
        rows,
        dim,
        num_streams,
        eps_,
        dest_rs,
        dest_ss,
        msg_rs,
        msg_ss,
        gain_rs,
        gain_ss,
        forget_rs,
        forget_ss,
        weight_ss,
        flags,
        0u,
        0u};

    const char* kname = out_dtype_ == float32
        ? "inkling2_loop_blend_rms_norm_f32"
        : "inkling2_loop_blend_rms_norm_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "loop_blend_rms_norm", kname));
    enc.set_input_array(dest, 0);
    enc.set_input_array(msg, 1);
    enc.set_input_array(gain, 2);
    enc.set_input_array(forget, 3);
    enc.set_input_array(weight, 4);
    enc.set_output_array(out, 5);
    enc.set_bytes(params, 6);
    enc.dispatch_threadgroups(MTL::Size(rows, num_streams, 1), MTL::Size(256, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2LoopBlendRmsNorm requires Metal");
#endif
  }

 private:
  float eps_;
  Dtype out_dtype_;
  Shape out_shape_;
};

// ============================================================================
// 4. Inkling2ShortConvK4 (single-sequence, strided x)
// ============================================================================
class Inkling2ShortConvK4 : public fast::Custom {
 public:
  Inkling2ShortConvK4(Stream s, bool is_dynamic, uint32_t num_kv_heads, Dtype out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2ShortConvK4 requires Metal");
            }),
        is_dynamic_(is_dynamic),
        num_kv_heads_(num_kv_heads),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2ShortConvK4)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2ShortConvK4&>(other);
    return is_dynamic_ == rhs.is_dynamic_ &&
        num_kv_heads_ == rhs.num_kv_heads_ &&
        out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& x = inputs[0];
    return {x.shape(), Shape{3, x.shape(1)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2ShortConvK4 requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t seq_len = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t channels = static_cast<uint32_t>(inputs[0].shape(1));
    auto [x, x_stride] = ensure_2d_strided(inputs[0], seq_len, channels, s, enc);
    array weight = ensure_row_contiguous(inputs[1], s, enc);
    array hist_in = ensure_row_contiguous(inputs[2], s, enc);
    std::optional<array> dyn_taps_arr;
    uint32_t dyn_is_f32 = 1u;
    if (is_dynamic_) {
      dyn_taps_arr = ensure_row_contiguous(inputs[3], s, enc);
      dyn_is_f32 = dyn_taps_arr->dtype() == float32 ? 1u : 0u;
    }

    auto& y_out = outputs[0];
    auto& hist_out = outputs[1];
    y_out.set_data(allocator::malloc(y_out.nbytes()));
    hist_out.set_data(allocator::malloc(hist_out.nbytes()));

    const float dummy = 0.0f;

    const char* kname = out_dtype_ == float32 ? "inkling2_sconv_k4_f32"
                                              : "inkling2_sconv_k4_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "sconv_k4", kname));
    enc.set_input_array(x, 0);
    enc.set_input_array(weight, 1);
    enc.set_input_array(hist_in, 2);
    if (is_dynamic_) {
      enc.set_input_array(*dyn_taps_arr, 3);
    } else {
      enc.set_bytes(dummy, 3);
    }
    Inkling2SconvK4Params params{
        seq_len,
        channels,
        is_dynamic_ ? 1u : 0u,
        num_kv_heads_,
        weight.dtype() == float32 ? 1u : 0u,
        hist_in.dtype() == float32 ? 1u : 0u,
        dyn_is_f32,
        x_stride};
    enc.set_output_array(y_out, 4);
    enc.set_output_array(hist_out, 5);
    enc.set_bytes(params, 6);
    const size_t tgx = 256;
    const size_t gx = (channels + tgx - 1) / tgx;
    enc.dispatch_threadgroups(MTL::Size(gx, seq_len, 1), MTL::Size(tgx, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2ShortConvK4 requires Metal");
#endif
  }

 private:
  bool is_dynamic_;
  uint32_t num_kv_heads_;
  Dtype out_dtype_;
};

// ============================================================================
// 5. Inkling2QrClip
// ============================================================================
class Inkling2QrClip : public fast::Custom {
 public:
  Inkling2QrClip(
      Stream s,
      uint32_t num_heads,
      uint32_t head_dim,
      uint32_t d_rel,
      float qk_thresh,
      float rel_thresh,
      float eps,
      bool has_tau,
      bool r_round_bf16,
      Dtype q_out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2QrClip requires Metal");
            }),
        num_heads_(num_heads),
        head_dim_(head_dim),
        d_rel_(d_rel),
        qk_thresh_(qk_thresh),
        rel_thresh_(rel_thresh),
        eps_(eps),
        has_tau_(has_tau),
        r_round_bf16_(r_round_bf16),
        q_out_dtype_(q_out_dtype) {}

  DEFINE_NAME(Inkling2QrClip)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2QrClip&>(other);
    return num_heads_ == rhs.num_heads_ &&
        head_dim_ == rhs.head_dim_ &&
        d_rel_ == rhs.d_rel_ &&
        qk_thresh_ == rhs.qk_thresh_ &&
        rel_thresh_ == rhs.rel_thresh_ &&
        eps_ == rhs.eps_ &&
        has_tau_ == rhs.has_tau_ &&
        r_round_bf16_ == rhs.r_round_bf16_ &&
        q_out_dtype_ == rhs.q_out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int t = inputs[0].shape(0);
    return {
        Shape{t, static_cast<int>(num_heads_), static_cast<int>(head_dim_)},
        Shape{t, static_cast<int>(num_heads_), static_cast<int>(d_rel_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2QrClip requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t seq_len = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t q_dim = num_heads_ * head_dim_;
    const uint32_t r_dim = num_heads_ * d_rel_;

    auto [q_in, q_stride] = ensure_2d_strided(inputs[0], seq_len, q_dim, s, enc);
    auto [r_in, r_stride] = ensure_2d_strided(inputs[1], seq_len, r_dim, s, enc);
    array q_clip = ensure_row_contiguous(inputs[2], s, enc);
    array rel_clip = ensure_row_contiguous(inputs[3], s, enc);
    std::optional<array> tau_arr;
    uint32_t tau_f32 = 1u;
    if (has_tau_) {
      tau_arr = ensure_row_contiguous(inputs[4], s, enc);
      tau_f32 = tau_arr->dtype() == float32 ? 1u : 0u;
    }

    auto& q_out = outputs[0];
    auto& r_out = outputs[1];
    q_out.set_data(allocator::malloc(q_out.nbytes()));
    r_out.set_data(allocator::malloc(r_out.nbytes()));

    const float dummy = 1.0f;

    const char* kname = q_out_dtype_ == float32 ? "inkling2_qr_clip_f32"
                                                : "inkling2_qr_clip_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "qr_clip", kname));
    enc.set_input_array(q_in, 0);
    enc.set_input_array(r_in, 1);
    enc.set_input_array(q_clip, 2);
    enc.set_input_array(rel_clip, 3);
    if (has_tau_) {
      enc.set_input_array(*tau_arr, 4);
    } else {
      enc.set_bytes(dummy, 4);
    }

    const uint32_t flags =
        (q_in.dtype() == float32 ? 1u : 0u) |
        ((r_in.dtype() == float32 ? 1u : 0u) << 1u) |
        ((q_clip.dtype() == float32 ? 1u : 0u) << 2u) |
        ((rel_clip.dtype() == float32 ? 1u : 0u) << 3u) |
        ((has_tau_ ? 1u : 0u) << 4u) |
        (tau_f32 << 5u) |
        ((r_round_bf16_ ? 1u : 0u) << 6u);

    Inkling2QrClipParams params{
        seq_len,
        num_heads_,
        head_dim_,
        d_rel_,
        q_stride,
        r_stride,
        qk_thresh_,
        rel_thresh_,
        eps_,
        flags,
        0u,
        0u};
    enc.set_output_array(q_out, 5);
    enc.set_output_array(r_out, 6);
    enc.set_bytes(params, 7);
    enc.dispatch_threadgroups(MTL::Size(num_heads_, seq_len, 1), MTL::Size(32, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2QrClip requires Metal");
#endif
  }

 private:
  uint32_t num_heads_;
  uint32_t head_dim_;
  uint32_t d_rel_;
  float qk_thresh_;
  float rel_thresh_;
  float eps_;
  bool has_tau_;
  bool r_round_bf16_;
  Dtype q_out_dtype_;
};

// ============================================================================
// 6. Inkling2ShortConvK4KvPrep
// ============================================================================
class Inkling2ShortConvK4KvPrep : public fast::Custom {
 public:
  Inkling2ShortConvK4KvPrep(
      Stream s,
      uint32_t num_kv_heads,
      uint32_t head_dim,
      uint32_t num_seqs,
      bool is_dynamic,
      bool packed_hist,
      float qk_thresh,
      float v_eps,
      float l2_eps,
      Dtype out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2ShortConvK4KvPrep requires Metal");
            }),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        num_seqs_(num_seqs),
        is_dynamic_(is_dynamic),
        packed_hist_(packed_hist),
        qk_thresh_(qk_thresh),
        v_eps_(v_eps),
        l2_eps_(l2_eps),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2ShortConvK4KvPrep)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2ShortConvK4KvPrep&>(other);
    return num_kv_heads_ == rhs.num_kv_heads_ &&
        head_dim_ == rhs.head_dim_ &&
        num_seqs_ == rhs.num_seqs_ &&
        is_dynamic_ == rhs.is_dynamic_ &&
        packed_hist_ == rhs.packed_hist_ &&
        qk_thresh_ == rhs.qk_thresh_ &&
        v_eps_ == rhs.v_eps_ &&
        l2_eps_ == rhs.l2_eps_ &&
        out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int t = inputs[0].shape(0);
    const int kv_dim = static_cast<int>(num_kv_heads_ * head_dim_);
    Shape hs = (num_seqs_ == 1u && !packed_hist_)
        ? Shape{3, kv_dim}
        : Shape{static_cast<int>(num_seqs_), 3, kv_dim};
    Shape kv_s{t, static_cast<int>(num_kv_heads_), static_cast<int>(head_dim_)};
    return {kv_s, kv_s, hs, hs};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2ShortConvK4KvPrep requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t seq_len = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t kv_dim = num_kv_heads_ * head_dim_;

    // Layout of `inputs`:
    //   0: k_in, 1: v_in, 2: k_weight, 3: v_weight
    //   [optional: dyn_k, dyn_v if is_dynamic_]
    //   [optional: cu_seqlens if num_seqs_ > 1]
    //   k_hists (1 if packed_hist_ else num_seqs_)
    //   v_hists (1 if packed_hist_ else num_seqs_)
    // Prepare ALL inputs BEFORE setting compute pipeline state (any `copy_gpu`
    // invoked by `ensure_*` would otherwise clobber the active pipeline/buffers).
    auto [k_in, k_stride] = ensure_2d_strided(inputs[0], seq_len, kv_dim, s, enc);
    auto [v_in, v_stride] = ensure_2d_strided(inputs[1], seq_len, kv_dim, s, enc);
    array kw = ensure_row_contiguous(inputs[2], s, enc);
    array vw = ensure_row_contiguous(inputs[3], s, enc);

    size_t idx = 4;
    std::optional<array> dk_arr;
    std::optional<array> dv_arr;
    uint32_t dk_f32 = 1u;
    uint32_t dv_f32 = 1u;
    if (is_dynamic_) {
      dk_arr = ensure_row_contiguous(inputs[idx++], s, enc);
      dv_arr = ensure_row_contiguous(inputs[idx++], s, enc);
      dk_f32 = dk_arr->dtype() == float32 ? 1u : 0u;
      dv_f32 = dv_arr->dtype() == float32 ? 1u : 0u;
    }

    std::optional<array> cu_arr;
    if (num_seqs_ > 1u) {
      cu_arr = ensure_row_contiguous(inputs[idx++], s, enc);
    }

    const uint32_t h_count = packed_hist_ ? 1u : num_seqs_;
    std::vector<array> kh_arrs;
    kh_arrs.reserve(h_count);
    uint32_t kh_mask = 0u;
    uint32_t kh_strides[4] = {kv_dim, kv_dim, kv_dim, kv_dim};
    for (uint32_t i = 0; i < h_count; ++i) {
      if (packed_hist_) {
        array kh = ensure_row_contiguous(inputs[idx++], s, enc);
        if (kh.dtype() == float32) kh_mask |= (1u << i);
        kh_arrs.push_back(std::move(kh));
      } else {
        auto [kh, ks] = ensure_2d_strided(inputs[idx++], 3u, kv_dim, s, enc);
        if (kh.dtype() == float32) kh_mask |= (1u << i);
        kh_strides[i] = ks;
        kh_arrs.push_back(std::move(kh));
      }
    }
    std::vector<array> vh_arrs;
    vh_arrs.reserve(h_count);
    uint32_t vh_mask = 0u;
    uint32_t vh_strides[4] = {kv_dim, kv_dim, kv_dim, kv_dim};
    for (uint32_t i = 0; i < h_count; ++i) {
      if (packed_hist_) {
        array vh = ensure_row_contiguous(inputs[idx++], s, enc);
        if (vh.dtype() == float32) vh_mask |= (1u << i);
        vh_arrs.push_back(std::move(vh));
      } else {
        auto [vh, vs] = ensure_2d_strided(inputs[idx++], 3u, kv_dim, s, enc);
        if (vh.dtype() == float32) vh_mask |= (1u << i);
        vh_strides[i] = vs;
        vh_arrs.push_back(std::move(vh));
      }
    }

    for (int o = 0; o < 4; ++o) {
      outputs[o].set_data(allocator::malloc(outputs[o].nbytes()));
    }

    const float dummy = 0.0f;
    const int32_t dummy_cu = 0;
    const char* kname = out_dtype_ == float32
        ? "inkling2_sconv_k4_kv_prep_f32"
        : "inkling2_sconv_k4_kv_prep_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "sconv_k4_kv_prep", kname));
    enc.set_input_array(k_in, 0);
    enc.set_input_array(v_in, 1);
    enc.set_input_array(kw, 2);
    enc.set_input_array(vw, 3);

    if (is_dynamic_) {
      enc.set_input_array(*dk_arr, 4);
      enc.set_input_array(*dv_arr, 5);
    } else {
      enc.set_bytes(dummy, 4);
      enc.set_bytes(dummy, 5);
    }

    if (num_seqs_ > 1u) {
      enc.set_input_array(*cu_arr, 6);
    } else {
      enc.set_bytes(dummy_cu, 6);
    }

    for (uint32_t i = 0; i < 4u; ++i) {
      if (i < h_count) {
        enc.set_input_array(kh_arrs[i], 7 + static_cast<int>(i));
      } else {
        enc.set_bytes(dummy, 7 + static_cast<int>(i));
      }
    }
    for (uint32_t i = 0; i < 4u; ++i) {
      if (i < h_count) {
        enc.set_input_array(vh_arrs[i], 11 + static_cast<int>(i));
      } else {
        enc.set_bytes(dummy, 11 + static_cast<int>(i));
      }
    }

    for (int o = 0; o < 4; ++o) {
      enc.set_output_array(outputs[o], 15 + o);
    }

    const uint32_t flags =
        (kw.dtype() == float32 ? 1u : 0u) |
        ((vw.dtype() == float32 ? 1u : 0u) << 1u) |
        (dk_f32 << 2u) |
        (dv_f32 << 3u);

    Inkling2SconvK4KvPrepParams params{
        seq_len,
        num_kv_heads_,
        head_dim_,
        num_seqs_,
        k_stride,
        v_stride,
        qk_thresh_,
        v_eps_,
        l2_eps_,
        is_dynamic_ ? 1u : 0u,
        packed_hist_ ? 1u : 0u,
        flags,
        kh_mask,
        vh_mask,
        0u,
        0u,
        {kh_strides[0], kh_strides[1], kh_strides[2], kh_strides[3]},
        {vh_strides[0], vh_strides[1], vh_strides[2], vh_strides[3]}};
    enc.set_bytes(params, 19);
    enc.dispatch_threadgroups(MTL::Size(2u * num_kv_heads_, seq_len, 1), MTL::Size(32, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2ShortConvK4KvPrep requires Metal");
#endif
  }

 private:
  uint32_t num_kv_heads_;
  uint32_t head_dim_;
  uint32_t num_seqs_;
  bool is_dynamic_;
  bool packed_hist_;
  float qk_thresh_;
  float v_eps_;
  float l2_eps_;
  Dtype out_dtype_;
};

// ============================================================================
// 7. Inkling2BlockAttnRes
// ============================================================================
class Inkling2BlockAttnRes : public fast::Custom {
 public:
  Inkling2BlockAttnRes(
      Stream s,
      uint32_t num_sources,
      uint32_t num_queries,
      bool has_addend,
      bool has_norm_weight,
      float attn_res_eps,
      float norm_eps,
      bool squeeze_q1,
      Dtype out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2BlockAttnRes requires Metal");
            }),
        num_sources_(num_sources),
        num_queries_(num_queries),
        has_addend_(has_addend),
        has_norm_weight_(has_norm_weight),
        attn_res_eps_(attn_res_eps),
        norm_eps_(norm_eps),
        squeeze_q1_(squeeze_q1),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2BlockAttnRes)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2BlockAttnRes&>(other);
    return num_sources_ == rhs.num_sources_ &&
        num_queries_ == rhs.num_queries_ &&
        has_addend_ == rhs.has_addend_ &&
        has_norm_weight_ == rhs.has_norm_weight_ &&
        attn_res_eps_ == rhs.attn_res_eps_ &&
        norm_eps_ == rhs.norm_eps_ &&
        squeeze_q1_ == rhs.squeeze_q1_ &&
        out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int t = inputs[0].shape(0);
    const int d = inputs[0].shape(1);
    if (num_queries_ == 1u && squeeze_q1_) {
      return {Shape{t, d}};
    }
    return {Shape{static_cast<int>(num_queries_), t, d}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2BlockAttnRes requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t rows = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t dim = static_cast<uint32_t>(inputs[0].shape(1));
    const float dummy = 0.0f;

    Inkling2BlockAttnResParams params{};
    params.rows = rows;
    params.dim = dim;
    params.num_sources = num_sources_;
    params.num_queries = num_queries_;
    params.attn_res_eps = attn_res_eps_;
    params.norm_eps = norm_eps_;

    // Prepare ALL inputs BEFORE setting compute pipeline state.
    size_t idx = 0;
    std::vector<array> src_arrs;
    src_arrs.reserve(num_sources_);
    for (uint32_t k = 0; k < num_sources_; ++k) {
      auto [src, rs] = ensure_2d_strided(inputs[idx++], rows, dim, s, enc);
      if (src.dtype() == float32) {
        params.src_f32_mask |= (1u << k);
      }
      params.src_strides[k] = rs;
      src_arrs.push_back(std::move(src));
    }

    uint32_t flags = 0u;
    std::optional<array> add_arr;
    if (has_addend_) {
      auto [add, rs] = ensure_2d_strided(inputs[idx++], rows, dim, s, enc);
      flags |= 1u;
      if (add.dtype() == float32) flags |= (1u << 1u);
      params.addend_stride = rs;
      add_arr = std::move(add);
    }

    array queries = ensure_row_contiguous(inputs[idx++], s, enc);

    std::optional<array> nw_arr;
    if (has_norm_weight_) {
      array nw = ensure_row_contiguous(inputs[idx++], s, enc);
      flags |= (1u << 2u);
      if (nw.dtype() == float32) flags |= (1u << 3u);
      const bool bcast = (nw.ndim() == 1) || (nw.shape(0) == 1);
      if (bcast) flags |= (1u << 4u);
      nw_arr = std::move(nw);
    }
    const uint32_t vec4_max_t = resolved_attn_res_vec4_max_t();
    bool vec4_ok = (rows <= vec4_max_t) && ((dim % 4u) == 0u) && ((queries.offset() % 16u) == 0u);
    if (vec4_ok) {
      for (uint32_t k = 0; k < num_sources_; ++k) {
        const size_t align = src_arrs[k].dtype() == float32 ? 16u : 8u;
        if ((params.src_strides[k] % 4u) != 0u || (src_arrs[k].offset() % align) != 0u) {
          vec4_ok = false;
          break;
        }
      }
    }
    if (vec4_ok && has_addend_) {
      const size_t align = add_arr->dtype() == float32 ? 16u : 8u;
      if ((params.addend_stride % 4u) != 0u || (add_arr->offset() % align) != 0u) {
        vec4_ok = false;
      }
    }
    if (vec4_ok && has_norm_weight_) {
      const size_t align = nw_arr->dtype() == float32 ? 16u : 8u;
      if ((nw_arr->offset() % align) != 0u) {
        vec4_ok = false;
      }
    }
    if (vec4_ok) {
      flags |= (1u << 5u);
    }
    params.flags = flags;

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const char* kname = out_dtype_ == float32
        ? "inkling2_block_attn_res_f32"
        : "inkling2_block_attn_res_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "block_attn_res", kname));
    for (uint32_t k = 0; k < 8u; ++k) {
      if (k < num_sources_) {
        enc.set_input_array(src_arrs[k], static_cast<int>(k));
      } else {
        enc.set_bytes(dummy, static_cast<int>(k));
      }
    }
    if (has_addend_) {
      enc.set_input_array(*add_arr, 8);
    } else {
      enc.set_bytes(dummy, 8);
    }
    enc.set_input_array(queries, 9);
    if (has_norm_weight_) {
      enc.set_input_array(*nw_arr, 10);
    } else {
      enc.set_bytes(dummy, 10);
    }
    enc.set_output_array(out, 11);
    enc.set_bytes(params, 12);
    enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2BlockAttnRes requires Metal");
#endif
  }

 private:
  uint32_t num_sources_;
  uint32_t num_queries_;
  bool has_addend_;
  bool has_norm_weight_;
  float attn_res_eps_;
  float norm_eps_;
  bool squeeze_q1_;
  Dtype out_dtype_;
};

// ============================================================================
// 8. Inkling2PairRmsNorm
// ============================================================================
class Inkling2PairRmsNorm : public fast::Custom {
 public:
  Inkling2PairRmsNorm(Stream s, float eps, Shape out_shape, Dtype out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2PairRmsNorm requires Metal");
            }),
        eps_(eps),
        out_shape_(std::move(out_shape)),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2PairRmsNorm)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2PairRmsNorm&>(other);
    return eps_ == rhs.eps_ && out_shape_ == rhs.out_shape_ &&
        out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {out_shape_};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2PairRmsNorm requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t num_streams =
        out_shape_.size() == 3 ? static_cast<uint32_t>(out_shape_[0]) : 1u;
    const uint32_t rows = out_shape_.size() == 3
        ? static_cast<uint32_t>(out_shape_[1])
        : static_cast<uint32_t>(out_shape_[0]);
    const uint32_t dim = static_cast<uint32_t>(out_shape_.back() / 2);

    auto [dest, dest_rs, dest_ss] =
        ensure_3d_stream_strided(inputs[0], num_streams, rows, dim, s, enc);
    auto [msg, msg_rs, msg_ss] =
        ensure_3d_stream_strided(inputs[1], num_streams, rows, dim, s, enc);

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    Inkling2PairRmsNormParams params{
        rows,
        dim,
        num_streams,
        eps_,
        dest_rs,
        dest_ss,
        msg_rs,
        msg_ss,
        dest.dtype() == float32 ? 1u : 0u,
        msg.dtype() == float32 ? 1u : 0u,
        0u,
        0u};

    const char* kname = out_dtype_ == float32
        ? "inkling2_pair_rms_norm_f32"
        : "inkling2_pair_rms_norm_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "pair_rms_norm", kname));
    enc.set_input_array(dest, 0);
    enc.set_input_array(msg, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(params, 3);
    enc.dispatch_threadgroups(MTL::Size(2u * num_streams, rows, 1), MTL::Size(256, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2PairRmsNorm requires Metal");
#endif
  }

 private:
  float eps_;
  Shape out_shape_;
  Dtype out_dtype_;
};

// ============================================================================
// 9. Inkling2ShortConvK4Ragged (+ optional FP32 residual epilogue)
// ============================================================================
class Inkling2ShortConvK4Ragged : public fast::Custom {
 public:
  Inkling2ShortConvK4Ragged(
      Stream s,
      uint32_t num_seqs,
      bool is_dynamic,
      uint32_t num_kv_heads,
      bool packed_hist,
      bool has_res_a,
      bool has_res_b,
      Dtype out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2ShortConvK4Ragged requires Metal");
            }),
        num_seqs_(num_seqs),
        is_dynamic_(is_dynamic),
        num_kv_heads_(num_kv_heads),
        packed_hist_(packed_hist),
        has_res_a_(has_res_a),
        has_res_b_(has_res_b),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2ShortConvK4Ragged)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2ShortConvK4Ragged&>(other);
    return num_seqs_ == rhs.num_seqs_ &&
        is_dynamic_ == rhs.is_dynamic_ &&
        num_kv_heads_ == rhs.num_kv_heads_ &&
        packed_hist_ == rhs.packed_hist_ &&
        has_res_a_ == rhs.has_res_a_ &&
        has_res_b_ == rhs.has_res_b_ &&
        out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& x = inputs[0];
    Shape hs = (num_seqs_ == 1u && !packed_hist_)
        ? Shape{3, x.shape(1)}
        : Shape{static_cast<int>(num_seqs_), 3, x.shape(1)};
    return {x.shape(), hs};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2ShortConvK4Ragged requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t seq_len = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t channels = static_cast<uint32_t>(inputs[0].shape(1));
    // Prepare ALL inputs BEFORE setting compute pipeline state.
    auto [x, x_stride] = ensure_2d_strided(inputs[0], seq_len, channels, s, enc);
    array w = ensure_row_contiguous(inputs[1], s, enc);

    uint32_t flags = (w.dtype() == float32 ? 1u : 0u);
    uint32_t res_a_stride = channels;
    uint32_t res_b_stride = channels;

    size_t idx = 2;
    std::optional<array> dyn_arr;
    if (is_dynamic_) {
      dyn_arr = ensure_row_contiguous(inputs[idx++], s, enc);
      if (dyn_arr->dtype() == float32) flags |= (1u << 1u);
    }

    std::optional<array> cu_arr;
    if (num_seqs_ > 1u) {
      cu_arr = ensure_row_contiguous(inputs[idx++], s, enc);
    }

    std::optional<array> ra_arr;
    if (has_res_a_) {
      auto [ra, rs] = ensure_2d_strided(inputs[idx++], seq_len, channels, s, enc);
      res_a_stride = rs;
      flags |= (1u << 2u);
      if (ra.dtype() == float32) flags |= (1u << 3u);
      ra_arr = std::move(ra);
    }

    std::optional<array> rb_arr;
    if (has_res_b_) {
      auto [rb, rs] = ensure_2d_strided(inputs[idx++], seq_len, channels, s, enc);
      res_b_stride = rs;
      flags |= (1u << 4u);
      if (rb.dtype() == float32) flags |= (1u << 5u);
      rb_arr = std::move(rb);
    }

    const uint32_t h_count = packed_hist_ ? 1u : num_seqs_;
    std::vector<array> h_arrs;
    h_arrs.reserve(h_count);
    uint32_t hist_mask = 0u;
    for (uint32_t i = 0; i < h_count; ++i) {
      array h = ensure_row_contiguous(inputs[idx++], s, enc);
      if (h.dtype() == float32) hist_mask |= (1u << i);
      h_arrs.push_back(std::move(h));
    }

    auto& y_out = outputs[0];
    auto& hist_out = outputs[1];
    y_out.set_data(allocator::malloc(y_out.nbytes()));
    hist_out.set_data(allocator::malloc(hist_out.nbytes()));

    const float dummy = 0.0f;
    const int32_t dummy_cu = 0;
    const char* kname = out_dtype_ == float32
        ? "inkling2_sconv_k4_ragged_f32"
        : "inkling2_sconv_k4_ragged_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "sconv_k4_ragged", kname));
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    if (is_dynamic_) {
      enc.set_input_array(*dyn_arr, 2);
    } else {
      enc.set_bytes(dummy, 2);
    }
    if (num_seqs_ > 1u) {
      enc.set_input_array(*cu_arr, 3);
    } else {
      enc.set_bytes(dummy_cu, 3);
    }
    if (has_res_a_) {
      enc.set_input_array(*ra_arr, 4);
    } else {
      enc.set_bytes(dummy, 4);
    }
    if (has_res_b_) {
      enc.set_input_array(*rb_arr, 5);
    } else {
      enc.set_bytes(dummy, 5);
    }
    for (uint32_t i = 0; i < 4u; ++i) {
      if (i < h_count) {
        enc.set_input_array(h_arrs[i], 6 + static_cast<int>(i));
      } else {
        enc.set_bytes(dummy, 6 + static_cast<int>(i));
      }
    }
    enc.set_output_array(y_out, 10);
    enc.set_output_array(hist_out, 11);

    Inkling2SconvK4RaggedParams params{
        seq_len,
        channels,
        num_seqs_,
        is_dynamic_ ? 1u : 0u,
        num_kv_heads_,
        packed_hist_ ? 1u : 0u,
        x_stride,
        res_a_stride,
        res_b_stride,
        flags,
        hist_mask,
        0u};
    enc.set_bytes(params, 12);
    const size_t tgx = 256;
    const size_t gx = (channels + tgx - 1) / tgx;
    enc.dispatch_threadgroups(MTL::Size(gx, seq_len, 1), MTL::Size(tgx, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2ShortConvK4Ragged requires Metal");
#endif
  }

 private:
  uint32_t num_seqs_;
  bool is_dynamic_;
  uint32_t num_kv_heads_;
  bool packed_hist_;
  bool has_res_a_;
  bool has_res_b_;
  Dtype out_dtype_;
};

// ============================================================================
// 10. Inkling2MoeRoute
// ============================================================================
class Inkling2MoeRoute : public fast::Custom {
 public:
  Inkling2MoeRoute(
      Stream s,
      uint32_t top_k,
      float route_scale,
      bool has_bias,
      bool has_down_scale)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2MoeRoute requires Metal");
            }),
        top_k_(top_k),
        route_scale_(route_scale),
        has_bias_(has_bias),
        has_down_scale_(has_down_scale) {}

  DEFINE_NAME(Inkling2MoeRoute)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2MoeRoute&>(other);
    return top_k_ == rhs.top_k_ &&
        route_scale_ == rhs.route_scale_ &&
        has_bias_ == rhs.has_bias_ &&
        has_down_scale_ == rhs.has_down_scale_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int t = inputs[0].shape(0);
    Shape s{t, static_cast<int>(top_k_)};
    return {s, s};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2MoeRoute requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t rows = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t num_experts = static_cast<uint32_t>(inputs[0].shape(1));
    // Prepare ALL inputs BEFORE setting compute pipeline state.
    auto [logits, logits_stride] =
        ensure_2d_strided(inputs[0], rows, num_experts, s, enc);

    uint32_t flags = (logits.dtype() == float32 ? 1u : 0u);
    size_t idx = 1;
    std::optional<array> bias_arr;
    if (has_bias_) {
      bias_arr = ensure_row_contiguous(inputs[idx++], s, enc);
      flags |= (1u << 1u);
      if (bias_arr->dtype() == float32) flags |= (1u << 2u);
    }
    std::optional<array> ds_arr;
    if (has_down_scale_) {
      ds_arr = ensure_row_contiguous(inputs[idx++], s, enc);
      flags |= (1u << 3u);
      if (ds_arr->dtype() == float32) flags |= (1u << 4u);
    }

    auto& out_idx = outputs[0];
    auto& out_w = outputs[1];
    out_idx.set_data(allocator::malloc(out_idx.nbytes()));
    out_w.set_data(allocator::malloc(out_w.nbytes()));

    const float dummy = 0.0f;
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "moe_route", "inkling2_moe_route_f32"));
    enc.set_input_array(logits, 0);
    if (has_bias_) {
      enc.set_input_array(*bias_arr, 1);
    } else {
      enc.set_bytes(dummy, 1);
    }
    if (has_down_scale_) {
      enc.set_input_array(*ds_arr, 2);
    } else {
      enc.set_bytes(dummy, 2);
    }
    enc.set_output_array(out_idx, 3);
    enc.set_output_array(out_w, 4);

    Inkling2MoeRouteParams params{
        rows,
        num_experts,
        top_k_,
        route_scale_,
        flags,
        logits_stride,
        0u,
        0u};
    enc.set_bytes(params, 5);
    enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(32, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2MoeRoute requires Metal");
#endif
  }

 private:
  uint32_t top_k_;
  float route_scale_;
  bool has_bias_;
  bool has_down_scale_;
};

// ============================================================================
// 14. Inkling2MoeRouterProj (coalesced exact-FP32 router projection for T <= 32)
// ============================================================================
class Inkling2MoeRouterProj : public fast::Custom {
 public:
  explicit Inkling2MoeRouterProj(Stream s, uint32_t num_experts)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2MoeRouterProj requires Metal");
            }),
        num_experts_(num_experts) {}

  DEFINE_NAME(Inkling2MoeRouterProj)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2MoeRouterProj&>(other);
    return num_experts_ == rhs.num_experts_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{inputs[0].shape(0), static_cast<int>(num_experts_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2MoeRouterProj requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t rows = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t dim = static_cast<uint32_t>(inputs[0].shape(1));
    const size_t x_vec_align = inputs[0].dtype() == float32 ? 16u : 8u;
    auto [x, x_stride] = ensure_2d_strided(inputs[0], rows, dim, s, enc);
    if ((x_stride % 4u) != 0u || (x.offset() % x_vec_align) != 0u) {
      x = ensure_row_contiguous_aligned(x, x_vec_align, s, enc);
      x_stride = dim;
    }
    array w = ensure_row_contiguous_aligned(inputs[1], 16u, s, enc);

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    Inkling2MoeRouterProjParams params{
        rows,
        dim,
        num_experts_,
        x_stride,
        x.dtype() == float32 ? 1u : 0u,
        0u,
        0u,
        0u};

    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "moe_router_proj", "inkling2_moe_router_proj_f32"));
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(params, 3);
    enc.dispatch_threadgroups(
        MTL::Size((num_experts_ + 3u) / 4u, (rows + 7u) / 8u, 1),
        MTL::Size(128, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2MoeRouterProj requires Metal");
#endif
  }

  private:
  uint32_t num_experts_;
};

// ============================================================================
// 15. Inkling2LoopCtrlDown (fused pair RMSNorm + coalesced exact-FP32 down_proj)
// ============================================================================
class Inkling2LoopCtrlDown : public fast::Custom {
 public:
  Inkling2LoopCtrlDown(Stream s, float eps, uint32_t rank, Shape out_shape)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2LoopCtrlDown requires Metal");
            }),
        eps_(eps),
        rank_(rank),
        out_shape_(std::move(out_shape)) {}

  DEFINE_NAME(Inkling2LoopCtrlDown)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2LoopCtrlDown&>(other);
    return eps_ == rhs.eps_ && rank_ == rhs.rank_ && out_shape_ == rhs.out_shape_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {out_shape_};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2LoopCtrlDown requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t num_streams =
        out_shape_.size() == 3 ? static_cast<uint32_t>(out_shape_[0]) : 1u;
    const uint32_t rows = out_shape_.size() == 3
        ? static_cast<uint32_t>(out_shape_[1])
        : static_cast<uint32_t>(out_shape_[0]);
    const uint32_t dim = static_cast<uint32_t>(inputs[1].shape(-1));

    auto [dest, dest_rs, dest_ss] =
        ensure_3d_stream_strided(inputs[0], num_streams, rows, dim, s, enc);
    const size_t dest_align = dest.dtype() == float32 ? 16u : 8u;
    if ((dest_rs % 4u) != 0u || (dest_ss % 4u) != 0u ||
        (dest.offset() % dest_align) != 0u) {
      const bool bcast = (dest_ss == 0u);
      dest = ensure_row_contiguous_aligned(dest, dest_align, s, enc);
      dest_rs = dim;
      dest_ss = bcast ? 0u : rows * dim;
    }

    auto [msg, msg_rs, msg_ss] =
        ensure_3d_stream_strided(inputs[1], num_streams, rows, dim, s, enc);
    const size_t msg_align = msg.dtype() == float32 ? 16u : 8u;
    if ((msg_rs % 4u) != 0u || (msg_ss % 4u) != 0u ||
        (msg.offset() % msg_align) != 0u) {
      const bool bcast = (msg_ss == 0u);
      msg = ensure_row_contiguous_aligned(msg, msg_align, s, enc);
      msg_rs = dim;
      msg_ss = bcast ? 0u : rows * dim;
    }

    const size_t w_align = inputs[2].dtype() == float32 ? 16u : 8u;
    array down_w = ensure_row_contiguous_aligned(inputs[2], w_align, s, enc);

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const uint32_t flags =
        (dest.dtype() == float32 ? 1u : 0u) |
        ((msg.dtype() == float32 ? 1u : 0u) << 1u) |
        ((down_w.dtype() == bfloat16 ? 1u : 0u) << 2u);

    Inkling2LoopCtrlDownParams params{
        rows,
        dim,
        rank_,
        num_streams,
        eps_,
        dest_rs,
        dest_ss,
        msg_rs,
        msg_ss,
        flags,
        0u,
        0u};

    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "loop_ctrl_down", "inkling2_loop_ctrl_down_f32"));
    enc.set_input_array(dest, 0);
    enc.set_input_array(msg, 1);
    enc.set_input_array(down_w, 2);
    enc.set_output_array(out, 3);
    enc.set_bytes(params, 4);
    enc.dispatch_threadgroups(
        MTL::Size(rank_, (rows + 7u) / 8u, num_streams),
        MTL::Size(128, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2LoopCtrlDown requires Metal");
#endif
  }

 private:
  float eps_;
  uint32_t rank_;
  Shape out_shape_;
};

// ============================================================================
// 16. Inkling2LoopCtrlUpBlendNorm (coalesced FP32 up_proj + blend + RMSNorm)
// ============================================================================
class Inkling2LoopCtrlUpBlendNorm : public fast::Custom {
 public:
  Inkling2LoopCtrlUpBlendNorm(
      Stream s,
      float eps,
      uint32_t rank,
      Dtype out_dtype,
      Shape out_shape)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2LoopCtrlUpBlendNorm requires Metal");
            }),
        eps_(eps),
        rank_(rank),
        out_dtype_(out_dtype),
        out_shape_(std::move(out_shape)) {}

  DEFINE_NAME(Inkling2LoopCtrlUpBlendNorm)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2LoopCtrlUpBlendNorm&>(other);
    return eps_ == rhs.eps_ && rank_ == rhs.rank_ &&
        out_dtype_ == rhs.out_dtype_ && out_shape_ == rhs.out_shape_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {out_shape_};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2LoopCtrlUpBlendNorm requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t num_streams =
        out_shape_.size() == 3 ? static_cast<uint32_t>(out_shape_[0]) : 1u;
    const uint32_t rows = out_shape_.size() == 3
        ? static_cast<uint32_t>(out_shape_[1])
        : static_cast<uint32_t>(out_shape_[0]);
    const uint32_t dim = static_cast<uint32_t>(out_shape_.back());
    const uint32_t num_d_tiles = (dim + 31u) / 32u;

    array h = ensure_row_contiguous_aligned(inputs[0], 16u, s, enc);
    const size_t gf_align = inputs[1].dtype() == float32 ? 16u : 8u;
    array gf_w = ensure_row_contiguous_aligned(inputs[1], gf_align, s, enc);
    auto [dest, dest_rs, dest_ss] =
        ensure_3d_stream_strided(inputs[2], num_streams, rows, dim, s, enc);
    auto [msg, msg_rs, msg_ss] =
        ensure_3d_stream_strided(inputs[3], num_streams, rows, dim, s, enc);
    array weight = ensure_row_contiguous(inputs[4], s, enc);
    const uint32_t weight_ss =
        (weight.ndim() == 2 &&
         weight.shape(0) == static_cast<int>(num_streams) && num_streams > 1u)
        ? dim
        : 0u;

    array blended(
        Shape{static_cast<int>(num_streams * rows), static_cast<int>(dim)},
        float32,
        nullptr,
        {});
    blended.set_data(allocator::malloc(blended.nbytes()));
    enc.add_temporary(blended);

    array partial_sq(
        Shape{static_cast<int>(num_streams * rows), static_cast<int>(num_d_tiles)},
        float32,
        nullptr,
        {});
    partial_sq.set_data(allocator::malloc(partial_sq.nbytes()));
    enc.add_temporary(partial_sq);

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const uint32_t flags =
        (dest.dtype() == float32 ? 1u : 0u) |
        ((msg.dtype() == float32 ? 1u : 0u) << 1u) |
        ((gf_w.dtype() == bfloat16 ? 1u : 0u) << 2u);

    Inkling2LoopCtrlUpBlendParams up_params{
        rows,
        dim,
        rank_,
        num_streams,
        num_d_tiles,
        dest_rs,
        dest_ss,
        msg_rs,
        msg_ss,
        flags,
        0u,
        0u};

    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device,
        "inkling2_ops",
        "loop_ctrl_up_blend",
        "inkling2_loop_ctrl_up_blend_f32"));
    enc.set_input_array(h, 0);
    enc.set_input_array(gf_w, 1);
    enc.set_input_array(dest, 2);
    enc.set_input_array(msg, 3);
    enc.set_output_array(blended, 4);
    enc.set_output_array(partial_sq, 5);
    enc.set_bytes(up_params, 6);
    enc.dispatch_threadgroups(
        MTL::Size(num_d_tiles, (rows + 7u) / 8u, num_streams),
        MTL::Size(128, 1, 1));

    Inkling2RmsNormFromSqParams norm_params{
        rows,
        dim,
        num_streams,
        num_d_tiles,
        eps_,
        weight_ss,
        weight.dtype() == float32 ? 1u : 0u,
        0u};

    const char* norm_kname = out_dtype_ == float32
        ? "inkling2_rms_norm_from_sq_f32"
        : "inkling2_rms_norm_from_sq_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "rms_norm_from_sq", norm_kname));
    enc.set_input_array(blended, 0);
    enc.set_input_array(partial_sq, 1);
    enc.set_input_array(weight, 2);
    enc.set_output_array(out, 3);
    enc.set_bytes(norm_params, 4);
    enc.dispatch_threadgroups(
        MTL::Size((dim + 127u) / 128u, rows, num_streams),
        MTL::Size(128, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2LoopCtrlUpBlendNorm requires Metal");
#endif
  }

 private:
  float eps_;
  uint32_t rank_;
  Dtype out_dtype_;
  Shape out_shape_;
};

// ============================================================================
// 11. Inkling2MoeCombine
// ============================================================================
class Inkling2MoeCombine : public fast::Custom {
 public:
  Inkling2MoeCombine(
      Stream s,
      uint32_t rows,
      uint32_t top_k,
      uint32_t dim,
      bool has_inv_order,
      Dtype out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2MoeCombine requires Metal");
            }),
        rows_(rows),
        top_k_(top_k),
        dim_(dim),
        has_inv_order_(has_inv_order),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2MoeCombine)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2MoeCombine&>(other);
    return rows_ == rhs.rows_ &&
        top_k_ == rhs.top_k_ &&
        dim_ == rhs.dim_ &&
        has_inv_order_ == rhs.has_inv_order_ &&
        out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {Shape{static_cast<int>(rows_), static_cast<int>(dim_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2MoeCombine requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    array down = ensure_row_contiguous(inputs[0], s, enc);
    array weights = ensure_row_contiguous(inputs[1], s, enc);
    std::optional<array> inv_arr;
    if (has_inv_order_) {
      inv_arr = ensure_row_contiguous(inputs[2], s, enc);
    }
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const uint32_t dummy_inv = 0u;
    const char* kname = out_dtype_ == float32
        ? "inkling2_moe_combine_f32"
        : "inkling2_moe_combine_bf16";
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "moe_combine", kname));
    enc.set_input_array(down, 0);
    enc.set_input_array(weights, 1);
    if (has_inv_order_) {
      enc.set_input_array(*inv_arr, 2);
    } else {
      enc.set_bytes(dummy_inv, 2);
    }
    enc.set_output_array(out, 3);

    const uint32_t flags =
        (down.dtype() == float32 ? 1u : 0u) |
        ((has_inv_order_ ? 1u : 0u) << 1u) |
        ((weights.dtype() == float32 ? 1u : 0u) << 2u);
    Inkling2MoeCombineParams params{rows_, top_k_, dim_, flags};
    enc.set_bytes(params, 4);
    const size_t tgx = 256;
    const size_t gx = (dim_ + tgx - 1) / tgx;
    enc.dispatch_threadgroups(MTL::Size(gx, rows_, 1), MTL::Size(tgx, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2MoeCombine requires Metal");
#endif
  }

 private:
  uint32_t rows_;
  uint32_t top_k_;
  uint32_t dim_;
  bool has_inv_order_;
  Dtype out_dtype_;
};

// ============================================================================
// 12a. Inkling2LmHeadSoftcap
// ============================================================================
class Inkling2LmHeadSoftcap : public fast::Custom {
 public:
  Inkling2LmHeadSoftcap(
      Stream s,
      uint32_t out_cols,
      float softcap,
      bool has_bias,
      bool bias_round_bf16)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2LmHeadSoftcap requires Metal");
            }),
        out_cols_(out_cols),
        softcap_(softcap),
        has_bias_(has_bias),
        bias_round_bf16_(bias_round_bf16) {}

  DEFINE_NAME(Inkling2LmHeadSoftcap)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2LmHeadSoftcap&>(other);
    return out_cols_ == rhs.out_cols_ &&
        softcap_ == rhs.softcap_ &&
        has_bias_ == rhs.has_bias_ &&
        bias_round_bf16_ == rhs.bias_round_bf16_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{inputs[0].shape(0), static_cast<int>(out_cols_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2LmHeadSoftcap requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t rows = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t in_cols = static_cast<uint32_t>(inputs[0].shape(1));
    auto [logits, logits_stride] =
        ensure_2d_strided(inputs[0], rows, in_cols, s, enc);

    uint32_t bias_stride = 0u;
    uint32_t flags = (logits.dtype() == float32 ? 1u : 0u);
    if (bias_round_bf16_) flags |= (1u << 3u);
    std::optional<array> bias_arr;
    if (has_bias_) {
      const uint32_t b_rows = static_cast<uint32_t>(inputs[1].shape(0));
      const uint32_t b_cols = static_cast<uint32_t>(inputs[1].shape(1));
      auto [bias, bs] = ensure_2d_strided(inputs[1], b_rows, b_cols, s, enc);
      bias_stride = (b_rows == 1u) ? 0u : bs;
      flags |= (1u << 1u);
      if (bias.dtype() == float32) flags |= (1u << 2u);
      bias_arr = std::move(bias);
    }

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const float dummy = 0.0f;
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "lm_head_softcap", "inkling2_lm_head_softcap_f32"));
    enc.set_input_array(logits, 0);
    if (has_bias_) {
      enc.set_input_array(*bias_arr, 1);
    } else {
      enc.set_bytes(dummy, 1);
    }
    enc.set_output_array(out, 2);

    Inkling2LmHeadSoftcapParams params{
        rows, out_cols_, logits_stride, bias_stride, softcap_, 0u, flags, 0u};
    enc.set_bytes(params, 3);
    const size_t tgx = 256;
    const size_t gx = (out_cols_ + tgx - 1) / tgx;
    enc.dispatch_threadgroups(MTL::Size(gx, rows, 1), MTL::Size(tgx, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2LmHeadSoftcap requires Metal");
#endif
  }

 private:
  uint32_t out_cols_;
  float softcap_;
  bool has_bias_;
  bool bias_round_bf16_;
};

// ============================================================================
// 12b. Inkling2LmHeadSoftcapArgmax (2-stage fused GPU argmax)
// ============================================================================
class Inkling2LmHeadSoftcapArgmax : public fast::Custom {
 public:
  Inkling2LmHeadSoftcapArgmax(
      Stream s,
      uint32_t out_cols,
      float softcap,
      bool has_bias,
      bool bias_round_bf16)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2LmHeadSoftcapArgmax requires Metal");
            }),
        out_cols_(out_cols),
        softcap_(softcap),
        has_bias_(has_bias),
        bias_round_bf16_(bias_round_bf16) {}

  DEFINE_NAME(Inkling2LmHeadSoftcapArgmax)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2LmHeadSoftcapArgmax&>(other);
    return out_cols_ == rhs.out_cols_ &&
        softcap_ == rhs.softcap_ &&
        has_bias_ == rhs.has_bias_ &&
        bias_round_bf16_ == rhs.bias_round_bf16_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{inputs[0].shape(0)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2LmHeadSoftcapArgmax requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    constexpr uint32_t kSplits = 64u;
    const uint32_t rows = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t in_cols = static_cast<uint32_t>(inputs[0].shape(1));
    auto [logits, logits_stride] =
        ensure_2d_strided(inputs[0], rows, in_cols, s, enc);

    uint32_t bias_stride = 0u;
    uint32_t flags = (logits.dtype() == float32 ? 1u : 0u);
    if (bias_round_bf16_) flags |= (1u << 3u);
    std::optional<array> bias_arr;
    if (has_bias_) {
      const uint32_t b_rows = static_cast<uint32_t>(inputs[1].shape(0));
      const uint32_t b_cols = static_cast<uint32_t>(inputs[1].shape(1));
      auto [bias, bs] = ensure_2d_strided(inputs[1], b_rows, b_cols, s, enc);
      bias_stride = (b_rows == 1u) ? 0u : bs;
      flags |= (1u << 1u);
      if (bias.dtype() == float32) flags |= (1u << 2u);
      bias_arr = std::move(bias);
    }

    array partial_vals(
        Shape{static_cast<int>(rows * kSplits)}, float32, nullptr, {});
    partial_vals.set_data(allocator::malloc(partial_vals.nbytes()));
    enc.add_temporary(partial_vals);

    array partial_idxs(
        Shape{static_cast<int>(rows * kSplits)}, int32, nullptr, {});
    partial_idxs.set_data(allocator::malloc(partial_idxs.nbytes()));
    enc.add_temporary(partial_idxs);

    const float dummy = 0.0f;
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device,
        "inkling2_ops",
        "lm_head_softcap_argmax_stage1",
        "inkling2_lm_head_softcap_argmax_stage1"));
    enc.set_input_array(logits, 0);
    if (has_bias_) {
      enc.set_input_array(*bias_arr, 1);
    } else {
      enc.set_bytes(dummy, 1);
    }
    enc.set_output_array(partial_vals, 2);
    enc.set_output_array(partial_idxs, 3);

    Inkling2LmHeadSoftcapParams params{
        rows, out_cols_, logits_stride, bias_stride, softcap_, kSplits, flags, 0u};
    enc.set_bytes(params, 4);
    enc.dispatch_threadgroups(MTL::Size(kSplits, rows, 1), MTL::Size(256, 1, 1));

    auto& out_ids = outputs[0];
    out_ids.set_data(allocator::malloc(out_ids.nbytes()));
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device,
        "inkling2_ops",
        "lm_head_softcap_argmax_stage2",
        "inkling2_lm_head_softcap_argmax_stage2"));
    enc.set_input_array(partial_vals, 0);
    enc.set_input_array(partial_idxs, 1);
    enc.set_output_array(out_ids, 2);
    enc.set_bytes(params, 3);
    enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(64, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2LmHeadSoftcapArgmax requires Metal");
#endif
  }

 private:
  uint32_t out_cols_;
  float softcap_;
  bool has_bias_;
  bool bias_round_bf16_;
};

// ============================================================================
// 13. Inkling2MarkovGate
// ============================================================================
class Inkling2MarkovGate : public fast::Custom {
 public:
  Inkling2MarkovGate(Stream s, uint32_t rank, bool has_ctx)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2MarkovGate requires Metal");
            }),
        rank_(rank),
        has_ctx_(has_ctx) {}

  DEFINE_NAME(Inkling2MarkovGate)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2MarkovGate&>(other);
    return rank_ == rhs.rank_ && has_ctx_ == rhs.has_ctx_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int rows = inputs[0].shape(0);
    return {Shape{rows, static_cast<int>(rank_)}, Shape{rows, static_cast<int>(rank_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2MarkovGate requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t rows = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t raw_cols = 3u * rank_;
    auto [raw_arr, raw_stride] =
        ensure_2d_strided(inputs[0], rows, raw_cols, s, enc);

    std::optional<array> ctx_arr;
    uint32_t ctx_stride = 0u;
    size_t state_idx = 1u;
    uint32_t flags = (raw_arr.dtype() == float32 ? 1u : 0u);
    if (has_ctx_) {
      auto [ctx, cs] = ensure_2d_strided(inputs[1], rows, raw_cols, s, enc);
      ctx_stride = cs;
      flags |= (1u << 1u);
      if (ctx.dtype() == float32) flags |= (1u << 2u);
      ctx_arr = std::move(ctx);
      state_idx = 2u;
    }

    auto [state_arr, state_stride] =
        ensure_2d_strided(inputs[state_idx], rows, rank_, s, enc);
    if (state_arr.dtype() == float32) flags |= (1u << 3u);

    auto& out_state = outputs[0];
    auto& out_latent = outputs[1];
    out_state.set_data(allocator::malloc(out_state.nbytes()));
    out_latent.set_data(allocator::malloc(out_latent.nbytes()));

    const float dummy = 0.0f;
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "markov_gate", "inkling2_markov_gate_bf16"));
    enc.set_input_array(raw_arr, 0);
    if (has_ctx_) {
      enc.set_input_array(*ctx_arr, 1);
    } else {
      enc.set_bytes(dummy, 1);
    }
    enc.set_input_array(state_arr, 2);
    enc.set_output_array(out_state, 3);
    enc.set_output_array(out_latent, 4);

    Inkling2MarkovGateParams params{
        rows, rank_, raw_stride, ctx_stride, state_stride, flags, 0u, 0u};
    enc.set_bytes(params, 5);
    const uint32_t tg_x = std::min<uint32_t>(256u, std::max<uint32_t>(32u, rank_));
    enc.dispatch_threads(MTL::Size(rank_, rows, 1), MTL::Size(tg_x, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2MarkovGate requires Metal");
#endif
  }

 private:
  uint32_t rank_;
  bool has_ctx_;
};

// ============================================================================
// 14. Inkling2MarkovFoldSoftcap / Inkling2MarkovFoldSoftcapArgmax
//     Fused Q8_0 (group_size=32) folded Markov head + fold_bias + base_logits + softcap
// ============================================================================
class Inkling2MarkovFoldSoftcap : public fast::Custom {
 public:
  Inkling2MarkovFoldSoftcap(
      Stream s,
      uint32_t out_cols,
      uint32_t rank,
      float softcap,
      bool has_biases,
      bool argmax)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2MarkovFoldSoftcap requires Metal");
            }),
        out_cols_(out_cols),
        rank_(rank),
        softcap_(softcap),
        has_biases_(has_biases),
        argmax_(argmax) {}

  DEFINE_NAME(Inkling2MarkovFoldSoftcap)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2MarkovFoldSoftcap&>(other);
    return out_cols_ == rhs.out_cols_ &&
        rank_ == rhs.rank_ &&
        softcap_ == rhs.softcap_ &&
        has_biases_ == rhs.has_biases_ &&
        argmax_ == rhs.argmax_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int batch = inputs[0].shape(0);
    if (argmax_) {
      return {Shape{batch}};
    }
    return {Shape{batch, static_cast<int>(out_cols_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2MarkovFoldSoftcap requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const uint32_t batch = static_cast<uint32_t>(inputs[0].shape(0));
    auto [latent, latent_stride] =
        ensure_2d_strided(inputs[0], batch, rank_, s, enc);
    array fold_w = ensure_row_contiguous_aligned(inputs[1], 16, s, enc);
    array fold_s = ensure_row_contiguous(inputs[2], s, enc);
    std::optional<array> fold_b;
    size_t idx = 3u;
    if (has_biases_) {
      fold_b = ensure_row_contiguous(inputs[idx++], s, enc);
    }
    array fold_bias = ensure_row_contiguous(inputs[idx++], s, enc);
    const uint32_t base_cols = static_cast<uint32_t>(inputs[idx].shape(1));
    auto [base_logits, base_stride] =
        ensure_2d_strided(inputs[idx], batch, base_cols, s, enc);

    uint32_t flags = (latent.dtype() == float32 ? 1u : 0u) |
        ((base_logits.dtype() == float32 ? 1u : 0u) << 1u) |
        ((fold_bias.dtype() == float32 ? 1u : 0u) << 2u) |
        (sidecar_dtype_code(fold_s.dtype()) << 3u) |
        ((has_biases_ ? 1u : 0u) << 5u);

    const float dummy = 0.0f;
    const uint32_t grid_b = (batch + 3u) / 4u;

    if (!argmax_) {
      auto& out = outputs[0];
      out.set_data(allocator::malloc(out.nbytes()));
      Inkling2MarkovFoldParams params{
          batch, out_cols_, rank_, latent_stride, base_stride, softcap_, 0u, flags};
      enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
          device, "inkling2_ops", "markov_fold_softcap", "inkling2_markov_fold_softcap_f32"));
      enc.set_input_array(latent, 0);
      enc.set_input_array(fold_w, 1);
      enc.set_input_array(fold_s, 2);
      if (has_biases_) {
        enc.set_input_array(*fold_b, 3);
      } else {
        enc.set_bytes(dummy, 3);
      }
      enc.set_input_array(fold_bias, 4);
      enc.set_input_array(base_logits, 5);
      enc.set_output_array(out, 6);
      enc.set_bytes(params, 7);
      const uint32_t gx = (out_cols_ + 31u) / 32u;
      enc.dispatch_threadgroups(MTL::Size(gx, grid_b, 1), MTL::Size(256, 1, 1));
      return;
    }

    const uint32_t kSplits = std::min<uint32_t>(256u, std::max<uint32_t>(1u, (out_cols_ + 31u) / 32u));
    array partial_vals(
        Shape{static_cast<int>(batch * kSplits)}, float32, nullptr, {});
    partial_vals.set_data(allocator::malloc(partial_vals.nbytes()));
    enc.add_temporary(partial_vals);

    array partial_idxs(
        Shape{static_cast<int>(batch * kSplits)}, int32, nullptr, {});
    partial_idxs.set_data(allocator::malloc(partial_idxs.nbytes()));
    enc.add_temporary(partial_idxs);

    Inkling2MarkovFoldParams params{
        batch, out_cols_, rank_, latent_stride, base_stride, softcap_, kSplits, flags};
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device,
        "inkling2_ops",
        "markov_fold_softcap_argmax_stage1",
        "inkling2_markov_fold_softcap_argmax_stage1"));
    enc.set_input_array(latent, 0);
    enc.set_input_array(fold_w, 1);
    enc.set_input_array(fold_s, 2);
    if (has_biases_) {
      enc.set_input_array(*fold_b, 3);
    } else {
      enc.set_bytes(dummy, 3);
    }
    enc.set_input_array(fold_bias, 4);
    enc.set_input_array(base_logits, 5);
    enc.set_output_array(partial_vals, 6);
    enc.set_output_array(partial_idxs, 7);
    enc.set_bytes(params, 8);
    enc.dispatch_threadgroups(MTL::Size(kSplits, grid_b, 1), MTL::Size(256, 1, 1));

    auto& out_ids = outputs[0];
    out_ids.set_data(allocator::malloc(out_ids.nbytes()));
    Inkling2LmHeadSoftcapParams stage2_params{
        batch, out_cols_, 0u, 0u, softcap_, kSplits, 0u, 0u};
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device,
        "inkling2_ops",
        "lm_head_softcap_argmax_stage2",
        "inkling2_lm_head_softcap_argmax_stage2"));
    enc.set_input_array(partial_vals, 0);
    enc.set_input_array(partial_idxs, 1);
    enc.set_output_array(out_ids, 2);
    enc.set_bytes(stage2_params, 3);
    enc.dispatch_threadgroups(MTL::Size(batch, 1, 1), MTL::Size(64, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2MarkovFoldSoftcap requires Metal");
#endif
  }

 private:
  uint32_t out_cols_;
  uint32_t rank_;
  float softcap_;
  bool has_biases_;
  bool argmax_;
};

// ============================================================================
// 15a. Inkling2Bf16Linear (`1 <= M <= 32`, `K % 4 == 0`, 0-ULP batch-size invariant)
// ============================================================================
class Inkling2Bf16Linear : public fast::Custom {
 public:
  Inkling2Bf16Linear(Stream s, uint32_t rows, uint32_t out_dim, uint32_t in_dim)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2Bf16Linear requires Metal");
            }),
        rows_(rows),
        out_dim_(out_dim),
        in_dim_(in_dim) {}

  DEFINE_NAME(Inkling2Bf16Linear)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2Bf16Linear&>(other);
    return rows_ == rhs.rows_ && out_dim_ == rhs.out_dim_ && in_dim_ == rhs.in_dim_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {Shape{static_cast<int>(rows_), static_cast<int>(out_dim_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2Bf16Linear requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const size_t align = (in_dim_ % 8u == 0u) ? 16u : 8u;
    array x = ensure_row_contiguous_aligned(inputs[0], align, s, enc);
    array w = ensure_row_contiguous_aligned(inputs[1], align, s, enc);
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    Inkling2Bf16LinearParams params{rows_, out_dim_, in_dim_, 0u, 0.0f, 0u, 0u, 0u};
    const uint32_t tiles_m = (rows_ + 31u) / 32u;
    const uint32_t tiles_n = (out_dim_ + 15u) / 16u;
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "inkling2_ops", "bf16_linear_batched", "inkling2_bf16_linear_batched"));
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(params, 3);
    enc.dispatch_threadgroups(MTL::Size(tiles_m * tiles_n, 1, 1), MTL::Size(128, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2Bf16Linear requires Metal");
#endif
  }

 private:
  uint32_t rows_;
  uint32_t out_dim_;
  uint32_t in_dim_;
};

// ============================================================================
// 15b. Inkling2Bf16LmHeadSoftcapArgmax (`1 <= M <= 32`, `K % 4 == 0`, 0 DRAM logits write!)
// ============================================================================
class Inkling2Bf16LmHeadSoftcapArgmax : public fast::Custom {
 public:
  Inkling2Bf16LmHeadSoftcapArgmax(
      Stream s,
      uint32_t rows,
      uint32_t out_cols,
      uint32_t in_dim,
      float softcap)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2Bf16LmHeadSoftcapArgmax requires Metal");
            }),
        rows_(rows),
        out_cols_(out_cols),
        in_dim_(in_dim),
        softcap_(softcap) {}

  DEFINE_NAME(Inkling2Bf16LmHeadSoftcapArgmax)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2Bf16LmHeadSoftcapArgmax&>(other);
    return rows_ == rhs.rows_ && out_cols_ == rhs.out_cols_ &&
        in_dim_ == rhs.in_dim_ && softcap_ == rhs.softcap_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {Shape{static_cast<int>(rows_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2Bf16LmHeadSoftcapArgmax requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const size_t align = (in_dim_ % 8u == 0u) ? 16u : 8u;
    array x = ensure_row_contiguous_aligned(inputs[0], align, s, enc);
    array w = ensure_row_contiguous_aligned(inputs[1], align, s, enc);

    const uint32_t num_splits = (out_cols_ + 31u) / 32u;
    array partial_vals(
        Shape{static_cast<int>(rows_ * num_splits)}, float32, nullptr, {});
    partial_vals.set_data(allocator::malloc(partial_vals.nbytes()));
    enc.add_temporary(partial_vals);

    array partial_idxs(
        Shape{static_cast<int>(rows_ * num_splits)}, int32, nullptr, {});
    partial_idxs.set_data(allocator::malloc(partial_idxs.nbytes()));
    enc.add_temporary(partial_idxs);

    Inkling2Bf16LinearParams p1_params{
        rows_, out_cols_, in_dim_, num_splits, softcap_, 0u, 0u, 0u};
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device,
        "inkling2_ops",
        "bf16_lm_head_softcap_argmax_p1",
        "inkling2_bf16_lm_head_softcap_argmax_p1"));
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_output_array(partial_vals, 2);
    enc.set_output_array(partial_idxs, 3);
    enc.set_bytes(p1_params, 4);
    enc.dispatch_threadgroups(MTL::Size(num_splits, 1, 1), MTL::Size(128, 1, 1));

    auto& out_ids = outputs[0];
    out_ids.set_data(allocator::malloc(out_ids.nbytes()));
    Inkling2LmHeadSoftcapParams stage2_params{
        rows_, out_cols_, 0u, 0u, softcap_, num_splits, 0u, 0u};
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device,
        "inkling2_ops",
        "lm_head_softcap_argmax_stage2",
        "inkling2_lm_head_softcap_argmax_stage2"));
    enc.set_input_array(partial_vals, 0);
    enc.set_input_array(partial_idxs, 1);
    enc.set_output_array(out_ids, 2);
    enc.set_bytes(stage2_params, 3);
    enc.dispatch_threadgroups(MTL::Size(rows_, 1, 1), MTL::Size(64, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2Bf16LmHeadSoftcapArgmax requires Metal");
#endif
  }

 private:
  uint32_t rows_;
  uint32_t out_cols_;
  uint32_t in_dim_;
  float softcap_;
};

} // namespace
} // namespace mlx::core::inkling2_ops

extern "C" {

mlx_array* mlx_inkling2_gate_sink(
    mlx_array* o_ptr,
    mlx_array* gate_ptr,
    mlx_array* lse_ptr,
    mlx_array* sink_ptr) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!o_ptr || !gate_ptr || !lse_ptr || !sink_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& o = *reinterpret_cast<array*>(o_ptr);
    const auto& gate = *reinterpret_cast<array*>(gate_ptr);
    const auto& lse = *reinterpret_cast<array*>(lse_ptr);
    const auto& sink = *reinterpret_cast<array*>(sink_ptr);
    if (o.ndim() != 3 || !is_bf16_or_f32(o.dtype()) ||
        !is_bf16_or_f32(gate.dtype()) || lse.dtype() != float32 ||
        !is_bf16_or_f32(sink.dtype())) {
      return nullptr;
    }
    const int t = o.shape(0);
    const int h = o.shape(1);
    const int d = o.shape(2);
    if (t <= 0 || h <= 0 || d <= 0 ||
        static_cast<int64_t>(gate.size()) != static_cast<int64_t>(o.size()) ||
        lse.ndim() != 2 || lse.shape(0) != t || lse.shape(1) != h ||
        sink.ndim() != 2 || sink.shape(0) != t || sink.shape(1) != h) {
      return nullptr;
    }
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2GateSink>(s, o.dtype());
    auto res = array(o.shape(), o.dtype(), std::move(prim), {o, gate, lse, sink});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_gate_sink", e.what());
    return nullptr;
  }
#else
  (void)o_ptr;
  (void)gate_ptr;
  (void)lse_ptr;
  (void)sink_ptr;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_situ(
    mlx_array* gate_up_ptr,
    mlx_array* outer_scale_ptr,
    mlx_array* expert_indices_ptr,
    float gate_softcap,
    float up_softcap,
    bool interleaved) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!gate_up_ptr) {
    return nullptr;
  }
  const bool has_scale = outer_scale_ptr != nullptr;
  if (has_scale != (expert_indices_ptr != nullptr)) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& gate_up = *reinterpret_cast<array*>(gate_up_ptr);
    if (gate_up.ndim() < 1 || !is_bf16_or_f32(gate_up.dtype()) ||
        gate_up.shape(-1) <= 0 || gate_up.shape(-1) % 2 != 0 ||
        gate_up.size() == 0) {
      return nullptr;
    }
    const int64_t num_rows = gate_up.size() / gate_up.shape(-1);
    std::vector<array> inputs = {gate_up};
    if (has_scale) {
      const auto& outer = *reinterpret_cast<array*>(outer_scale_ptr);
      const auto& idx = *reinterpret_cast<array*>(expert_indices_ptr);
      if (outer.ndim() != 1 || !is_bf16_or_f32(outer.dtype()) ||
          idx.dtype() != int32 || static_cast<int64_t>(idx.size()) != num_rows) {
        return nullptr;
      }
      inputs.push_back(outer);
      inputs.push_back(idx);
    }
    Shape out_shape = gate_up.shape();
    out_shape.back() /= 2;
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2Situ>(
        s, gate_softcap, up_softcap, interleaved, has_scale, gate_up.dtype());
    auto res = array(std::move(out_shape), gate_up.dtype(), std::move(prim), std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_situ", e.what());
    return nullptr;
  }
#else
  (void)gate_up_ptr;
  (void)outer_scale_ptr;
  (void)expert_indices_ptr;
  (void)gate_softcap;
  (void)up_softcap;
  (void)interleaved;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_loop_blend_rms_norm(
    mlx_array* dest_ptr,
    mlx_array* msg_ptr,
    mlx_array* gain_raw_ptr,
    mlx_array* forget_raw_ptr,
    mlx_array* norm_weight_ptr,
    float eps,
    bool output_f32) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!dest_ptr || !msg_ptr || !gain_raw_ptr || !forget_raw_ptr ||
      !norm_weight_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& dest = *reinterpret_cast<array*>(dest_ptr);
    const auto& msg = *reinterpret_cast<array*>(msg_ptr);
    const auto& gain = *reinterpret_cast<array*>(gain_raw_ptr);
    const auto& forget = *reinterpret_cast<array*>(forget_raw_ptr);
    const auto& weight = *reinterpret_cast<array*>(norm_weight_ptr);
    if (msg.ndim() < 1 || msg.size() == 0 ||
        !is_bf16_or_f32(dest.dtype()) || !is_bf16_or_f32(msg.dtype()) ||
        !is_bf16_or_f32(gain.dtype()) || !is_bf16_or_f32(forget.dtype()) ||
        !is_bf16_or_f32(weight.dtype()) ||
        gain.shape() != msg.shape() || forget.shape() != msg.shape()) {
      return nullptr;
    }
    const int dim = msg.shape(-1);
    if (msg.ndim() == 3) {
      const int streams = msg.shape(0);
      const int rows = msg.shape(1);
      const bool dest_ok =
          (dest.ndim() == 3 && (dest.shape(0) == streams || dest.shape(0) == 1) &&
           dest.shape(1) == rows && dest.shape(2) == dim) ||
          (dest.ndim() == 2 && dest.shape(0) == rows && dest.shape(1) == dim);
      const bool weight_ok =
          (weight.ndim() == 1 && weight.shape(0) == dim) ||
          (weight.ndim() == 2 && (weight.shape(0) == streams || weight.shape(0) == 1) &&
           weight.shape(1) == dim);
      if (!dest_ok || !weight_ok) {
        return nullptr;
      }
    } else {
      if (dest.shape() != msg.shape() || weight.ndim() != 1 || weight.shape(0) != dim) {
        return nullptr;
      }
    }
    const Dtype out_dtype = output_f32 ? float32 : bfloat16;
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2LoopBlendRmsNorm>(s, eps, out_dtype, msg.shape());
    auto res = array(
        msg.shape(), out_dtype, std::move(prim),
        {dest, msg, gain, forget, weight});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_loop_blend_rms_norm", e.what());
    return nullptr;
  }
#else
  (void)dest_ptr;
  (void)msg_ptr;
  (void)gain_raw_ptr;
  (void)forget_raw_ptr;
  (void)norm_weight_ptr;
  (void)eps;
  (void)output_f32;
  return nullptr;
#endif
}

bool mlx_inkling2_sconv_k4(
    mlx_array* x_ptr,
    mlx_array* weight_ptr,
    mlx_array* history_in_ptr,
    mlx_array* dyn_taps_ptr,
    mlx_array** out_y,
    mlx_array** out_history) {
  if (out_y) *out_y = nullptr;
  if (out_history) *out_history = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !weight_ptr || !history_in_ptr || !out_y || !out_history) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(weight_ptr);
    const auto& hist = *reinterpret_cast<array*>(history_in_ptr);
    if (x.ndim() != 2 || !is_bf16_or_f32(x.dtype()) ||
        x.shape(0) <= 0 || x.shape(1) <= 0 ||
        w.ndim() != 2 || !is_bf16_or_f32(w.dtype()) ||
        w.shape(0) != x.shape(1) || w.shape(1) != 4 ||
        hist.ndim() != 2 || !is_bf16_or_f32(hist.dtype()) ||
        hist.shape(0) != 3 || hist.shape(1) != x.shape(1)) {
      return false;
    }
    const bool is_dynamic = dyn_taps_ptr != nullptr;
    uint32_t num_kv_heads = 0;
    std::vector<array> inputs = {x, w, hist};
    if (is_dynamic) {
      const auto& dyn = *reinterpret_cast<array*>(dyn_taps_ptr);
      if (dyn.ndim() != 3 || !is_bf16_or_f32(dyn.dtype()) ||
          dyn.shape(0) != x.shape(0) || dyn.shape(1) <= 0 ||
          dyn.shape(2) != 4 || x.shape(1) % dyn.shape(1) != 0) {
        return false;
      }
      num_kv_heads = static_cast<uint32_t>(dyn.shape(1));
      inputs.push_back(dyn);
    }
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2ShortConvK4>(
        s, is_dynamic, num_kv_heads, x.dtype());
    auto results = array::make_arrays(
        {x.shape(), Shape{3, x.shape(1)}},
        {x.dtype(), x.dtype()},
        std::move(prim),
        std::move(inputs));
    *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_history = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_sconv_k4", e.what());
    return false;
  }
#else
  (void)x_ptr;
  (void)weight_ptr;
  (void)history_in_ptr;
  (void)dyn_taps_ptr;
  return false;
#endif
}

bool mlx_inkling2_qr_clip(
    mlx_array* q_in_ptr,
    mlx_array* r_in_ptr,
    mlx_array* q_clip_ptr,
    mlx_array* rel_clip_ptr,
    mlx_array* tau_ptr,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t d_rel,
    float qk_thresh,
    float rel_thresh,
    float eps,
    bool output_f32,
    bool r_round_bf16,
    mlx_array** out_q,
    mlx_array** out_r) {
  if (out_q) *out_q = nullptr;
  if (out_r) *out_r = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!q_in_ptr || !r_in_ptr || !q_clip_ptr || !rel_clip_ptr || !out_q || !out_r) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return false;
    }
    const auto& q_in = *reinterpret_cast<array*>(q_in_ptr);
    const auto& r_in = *reinterpret_cast<array*>(r_in_ptr);
    const auto& q_clip = *reinterpret_cast<array*>(q_clip_ptr);
    const auto& rel_clip = *reinterpret_cast<array*>(rel_clip_ptr);
    if (num_heads == 0 || head_dim == 0 || d_rel == 0 ||
        q_in.ndim() != 2 || r_in.ndim() != 2 ||
        q_in.shape(0) <= 0 || r_in.shape(0) != q_in.shape(0) ||
        static_cast<uint32_t>(q_in.shape(1)) != num_heads * head_dim ||
        static_cast<uint32_t>(r_in.shape(1)) != num_heads * d_rel ||
        !is_bf16_or_f32(q_in.dtype()) || !is_bf16_or_f32(r_in.dtype()) ||
        !is_bf16_or_f32(q_clip.dtype()) || !is_bf16_or_f32(rel_clip.dtype()) ||
        static_cast<uint64_t>(q_clip.size()) != static_cast<uint64_t>(num_heads) * head_dim ||
        static_cast<uint64_t>(rel_clip.size()) != static_cast<uint64_t>(num_heads) * d_rel) {
      return false;
    }
    std::vector<array> inputs = {q_in, r_in, q_clip, rel_clip};
    const bool has_tau = tau_ptr != nullptr;
    if (has_tau) {
      const auto& tau = *reinterpret_cast<array*>(tau_ptr);
      if (!is_bf16_or_f32(tau.dtype()) || static_cast<int>(tau.size()) != q_in.shape(0)) {
        return false;
      }
      inputs.push_back(tau);
    }
    const Dtype q_dtype = output_f32 ? float32 : bfloat16;
    const int t = q_in.shape(0);
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2QrClip>(
        s, num_heads, head_dim, d_rel, qk_thresh, rel_thresh, eps,
        has_tau, r_round_bf16, q_dtype);
    auto results = array::make_arrays(
        {Shape{t, static_cast<int>(num_heads), static_cast<int>(head_dim)},
         Shape{t, static_cast<int>(num_heads), static_cast<int>(d_rel)}},
        {q_dtype, float32},
        std::move(prim),
        std::move(inputs));
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_r = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_qr_clip", e.what());
    return false;
  }
#else
  (void)q_in_ptr; (void)r_in_ptr; (void)q_clip_ptr; (void)rel_clip_ptr; (void)tau_ptr;
  (void)num_heads; (void)head_dim; (void)d_rel; (void)qk_thresh; (void)rel_thresh;
  (void)eps; (void)output_f32; (void)r_round_bf16;
  return false;
#endif
}

bool mlx_inkling2_sconv_k4_kv_prep(
    mlx_array* k_in_ptr,
    mlx_array* v_in_ptr,
    mlx_array* k_weight_ptr,
    mlx_array* v_weight_ptr,
    mlx_array* dyn_k_ptr,
    mlx_array* dyn_v_ptr,
    mlx_array* cu_seqlens_ptr,
    mlx_array* const* k_hists_ptr,
    mlx_array* const* v_hists_ptr,
    uint32_t num_seqs,
    bool packed_hist,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    float qk_thresh,
    float v_eps,
    float l2_eps,
    mlx_array** out_k,
    mlx_array** out_v,
    mlx_array** out_k_hist,
    mlx_array** out_v_hist) {
  if (out_k) *out_k = nullptr;
  if (out_v) *out_v = nullptr;
  if (out_k_hist) *out_k_hist = nullptr;
  if (out_v_hist) *out_v_hist = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!k_in_ptr || !v_in_ptr || !k_weight_ptr || !v_weight_ptr ||
      !k_hists_ptr || !v_hists_ptr || !out_k || !out_v || !out_k_hist || !out_v_hist ||
      num_seqs == 0 || num_kv_heads == 0 || head_dim == 0 || head_dim > 256u) {
    return false;
  }
  if (!packed_hist && num_seqs > 4u) {
    return false;
  }
  const bool is_dynamic = (dyn_k_ptr != nullptr) || (dyn_v_ptr != nullptr);
  if (is_dynamic && (!dyn_k_ptr || !dyn_v_ptr)) {
    return false;
  }
  if (num_seqs > 1u && !cu_seqlens_ptr) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return false;
    }
    const auto& k_in = *reinterpret_cast<array*>(k_in_ptr);
    const auto& v_in = *reinterpret_cast<array*>(v_in_ptr);
    const auto& kw = *reinterpret_cast<array*>(k_weight_ptr);
    const auto& vw = *reinterpret_cast<array*>(v_weight_ptr);
    const int kv_dim = static_cast<int>(num_kv_heads * head_dim);
    if (k_in.ndim() != 2 || v_in.ndim() != 2 ||
        k_in.shape(0) <= 0 || k_in.shape(0) != v_in.shape(0) ||
        k_in.shape(1) != kv_dim || v_in.shape(1) != kv_dim ||
        k_in.dtype() != v_in.dtype() || !is_bf16_or_f32(k_in.dtype()) ||
        kw.ndim() != 2 || kw.shape(0) != kv_dim || kw.shape(1) != 4 || !is_bf16_or_f32(kw.dtype()) ||
        vw.ndim() != 2 || vw.shape(0) != kv_dim || vw.shape(1) != 4 || !is_bf16_or_f32(vw.dtype())) {
      return false;
    }
    std::vector<array> inputs = {k_in, v_in, kw, vw};
    if (is_dynamic) {
      const auto& dk = *reinterpret_cast<array*>(dyn_k_ptr);
      const auto& dv = *reinterpret_cast<array*>(dyn_v_ptr);
      if (dk.ndim() != 3 || dv.ndim() != 3 ||
          dk.shape(0) != k_in.shape(0) || dv.shape(0) != k_in.shape(0) ||
          dk.shape(1) != static_cast<int>(num_kv_heads) || dv.shape(1) != static_cast<int>(num_kv_heads) ||
          dk.shape(2) != 4 || dv.shape(2) != 4 ||
          !is_bf16_or_f32(dk.dtype()) || !is_bf16_or_f32(dv.dtype())) {
        return false;
      }
      inputs.push_back(dk);
      inputs.push_back(dv);
    }
    if (num_seqs > 1u) {
      const auto& cu = *reinterpret_cast<array*>(cu_seqlens_ptr);
      if (cu.ndim() != 1 || cu.dtype() != int32 || cu.shape(0) != static_cast<int>(num_seqs + 1u)) {
        return false;
      }
      inputs.push_back(cu);
    }
    const uint32_t h_count = packed_hist ? 1u : num_seqs;
    for (uint32_t i = 0; i < h_count; ++i) {
      if (!k_hists_ptr[i]) return false;
      const auto& kh = *reinterpret_cast<array*>(k_hists_ptr[i]);
      if (!is_bf16_or_f32(kh.dtype())) return false;
      if (packed_hist) {
        if (kh.ndim() != 3 || kh.shape(0) != static_cast<int>(num_seqs) || kh.shape(1) != 3 || kh.shape(2) != kv_dim) return false;
      } else {
        if (kh.ndim() != 2 || kh.shape(0) != 3 || kh.shape(1) != kv_dim) return false;
      }
      inputs.push_back(kh);
    }
    for (uint32_t i = 0; i < h_count; ++i) {
      if (!v_hists_ptr[i]) return false;
      const auto& vh = *reinterpret_cast<array*>(v_hists_ptr[i]);
      if (!is_bf16_or_f32(vh.dtype())) return false;
      if (packed_hist) {
        if (vh.ndim() != 3 || vh.shape(0) != static_cast<int>(num_seqs) || vh.shape(1) != 3 || vh.shape(2) != kv_dim) return false;
      } else {
        if (vh.ndim() != 2 || vh.shape(0) != 3 || vh.shape(1) != kv_dim) return false;
      }
      inputs.push_back(vh);
    }
    const Dtype dt = k_in.dtype();
    const int t = k_in.shape(0);
    Shape kv_s{t, static_cast<int>(num_kv_heads), static_cast<int>(head_dim)};
    Shape hs = (num_seqs == 1u && !packed_hist)
        ? Shape{3, kv_dim}
        : Shape{static_cast<int>(num_seqs), 3, kv_dim};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2ShortConvK4KvPrep>(
        s, num_kv_heads, head_dim, num_seqs, is_dynamic, packed_hist,
        qk_thresh, v_eps, l2_eps, dt);
    auto results = array::make_arrays(
        {kv_s, kv_s, hs, hs},
        {dt, dt, dt, dt},
        std::move(prim),
        std::move(inputs));
    *out_k = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_v = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    *out_k_hist = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
    *out_v_hist = reinterpret_cast<mlx_array*>(new array(std::move(results[3])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_sconv_k4_kv_prep", e.what());
    return false;
  }
#else
  (void)k_in_ptr; (void)v_in_ptr; (void)k_weight_ptr; (void)v_weight_ptr;
  (void)dyn_k_ptr; (void)dyn_v_ptr; (void)cu_seqlens_ptr; (void)k_hists_ptr; (void)v_hists_ptr;
  (void)num_seqs; (void)packed_hist; (void)num_kv_heads; (void)head_dim;
  (void)qk_thresh; (void)v_eps; (void)l2_eps;
  return false;
#endif
}

mlx_array* mlx_inkling2_block_attn_res(
    mlx_array* const* sources_ptr,
    uint32_t num_sources,
    mlx_array* addend_ptr,
    mlx_array* queries_ptr,
    mlx_array* norm_weight_ptr,
    float attn_res_eps,
    float norm_eps,
    bool squeeze_q1,
    bool output_f32) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!sources_ptr || num_sources == 0 || num_sources > 8u || !queries_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& queries = *reinterpret_cast<array*>(queries_ptr);
    if (queries.ndim() != 2 || queries.dtype() != float32 ||
        queries.shape(0) <= 0 || queries.shape(0) > 4 || queries.shape(1) <= 0) {
      return nullptr;
    }
    const uint32_t num_queries = static_cast<uint32_t>(queries.shape(0));
    const int dim = queries.shape(1);
    std::vector<array> inputs;
    inputs.reserve(num_sources + 3);
    int rows = -1;
    for (uint32_t k = 0; k < num_sources; ++k) {
      if (!sources_ptr[k]) return nullptr;
      const auto& src = *reinterpret_cast<array*>(sources_ptr[k]);
      if (src.ndim() != 2 || src.shape(0) <= 0 || src.shape(1) != dim ||
          !is_bf16_or_f32(src.dtype())) {
        return nullptr;
      }
      if (rows < 0) rows = src.shape(0);
      else if (src.shape(0) != rows) return nullptr;
      inputs.push_back(src);
    }
    const bool has_addend = addend_ptr != nullptr;
    if (has_addend) {
      const auto& add = *reinterpret_cast<array*>(addend_ptr);
      if (add.ndim() != 2 || add.shape(0) != rows || add.shape(1) != dim ||
          !is_bf16_or_f32(add.dtype())) {
        return nullptr;
      }
      inputs.push_back(add);
    }
    inputs.push_back(queries);
    const bool has_norm_weight = norm_weight_ptr != nullptr;
    if (has_norm_weight) {
      const auto& nw = *reinterpret_cast<array*>(norm_weight_ptr);
      const bool ok =
          is_bf16_or_f32(nw.dtype()) &&
          ((nw.ndim() == 1 && nw.shape(0) == dim) ||
           (nw.ndim() == 2 &&
            (nw.shape(0) == 1 || nw.shape(0) == static_cast<int>(num_queries)) &&
            nw.shape(1) == dim));
      if (!ok) return nullptr;
      inputs.push_back(nw);
    }
    const Dtype out_dtype = output_f32 ? float32 : bfloat16;
    Shape out_shape = (num_queries == 1u && squeeze_q1)
        ? Shape{rows, dim}
        : Shape{static_cast<int>(num_queries), rows, dim};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2BlockAttnRes>(
        s, num_sources, num_queries, has_addend, has_norm_weight,
        attn_res_eps, norm_eps, squeeze_q1, out_dtype);
    auto res = array(std::move(out_shape), out_dtype, std::move(prim), std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_block_attn_res", e.what());
    return nullptr;
  }
#else
  (void)sources_ptr; (void)num_sources; (void)addend_ptr; (void)queries_ptr;
  (void)norm_weight_ptr; (void)attn_res_eps; (void)norm_eps; (void)squeeze_q1; (void)output_f32;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_pair_rms_norm(
    mlx_array* dest_ptr,
    mlx_array* msg_ptr,
    float eps,
    bool output_f32) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!dest_ptr || !msg_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& dest = *reinterpret_cast<array*>(dest_ptr);
    const auto& msg = *reinterpret_cast<array*>(msg_ptr);
    if (!is_bf16_or_f32(dest.dtype()) || !is_bf16_or_f32(msg.dtype()) ||
        msg.size() == 0 || (msg.ndim() != 2 && msg.ndim() != 3)) {
      return nullptr;
    }
    Shape out_shape;
    if (msg.ndim() == 2) {
      const int rows = msg.shape(0);
      const int dim = msg.shape(1);
      if (dest.ndim() != 2 || dest.shape(0) != rows || dest.shape(1) != dim) {
        return nullptr;
      }
      out_shape = Shape{rows, 2 * dim};
    } else {
      const int streams = msg.shape(0);
      const int rows = msg.shape(1);
      const int dim = msg.shape(2);
      const bool dest_ok =
          (dest.ndim() == 3 && (dest.shape(0) == streams || dest.shape(0) == 1) &&
           dest.shape(1) == rows && dest.shape(2) == dim) ||
          (dest.ndim() == 2 && dest.shape(0) == rows && dest.shape(1) == dim);
      if (!dest_ok) {
        return nullptr;
      }
      out_shape = Shape{streams, rows, 2 * dim};
    }
    const Dtype out_dtype = output_f32 ? float32 : bfloat16;
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2PairRmsNorm>(s, eps, out_shape, out_dtype);
    auto res = array(out_shape, out_dtype, std::move(prim), {dest, msg});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_pair_rms_norm", e.what());
    return nullptr;
  }
#else
  (void)dest_ptr; (void)msg_ptr; (void)eps; (void)output_f32;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_moe_router_proj(
    mlx_array* x_ptr,
    mlx_array* weight_ptr) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !weight_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(weight_ptr);
    if (x.ndim() != 2 || w.ndim() != 2 ||
        !is_bf16_or_f32(x.dtype()) || w.dtype() != float32 ||
        x.shape(0) <= 0 || x.shape(1) <= 0 || (x.shape(1) % 4) != 0 ||
        w.shape(0) <= 0 || w.shape(1) != x.shape(1)) {
      return nullptr;
    }
    const uint32_t num_experts = static_cast<uint32_t>(w.shape(0));
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2MoeRouterProj>(s, num_experts);
    auto res = array(
        Shape{x.shape(0), static_cast<int>(num_experts)},
        float32,
        std::move(prim),
        {x, w});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_moe_router_proj", e.what());
    return nullptr;
  }
#else
  (void)x_ptr; (void)weight_ptr;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_loop_ctrl_down(
    mlx_array* dest_ptr,
    mlx_array* msg_ptr,
    mlx_array* down_w_ptr,
    float eps) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!dest_ptr || !msg_ptr || !down_w_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& dest = *reinterpret_cast<array*>(dest_ptr);
    const auto& msg = *reinterpret_cast<array*>(msg_ptr);
    const auto& down_w = *reinterpret_cast<array*>(down_w_ptr);
    if (!is_bf16_or_f32(dest.dtype()) || !is_bf16_or_f32(msg.dtype()) ||
        !is_bf16_or_f32(down_w.dtype()) || msg.size() == 0 ||
        (msg.ndim() != 2 && msg.ndim() != 3)) {
      return nullptr;
    }
    Shape out_shape;
    uint32_t rank = 0;
    if (msg.ndim() == 2) {
      const int rows = msg.shape(0);
      const int dim = msg.shape(1);
      if (rows <= 0 || dim <= 0 || (dim % 4) != 0 ||
          dest.ndim() != 2 || dest.shape(0) != rows || dest.shape(1) != dim ||
          down_w.ndim() != 2 || down_w.shape(0) <= 0 || down_w.shape(1) != 2 * dim) {
        return nullptr;
      }
      rank = static_cast<uint32_t>(down_w.shape(0));
      out_shape = Shape{rows, static_cast<int>(rank)};
    } else {
      const int streams = msg.shape(0);
      const int rows = msg.shape(1);
      const int dim = msg.shape(2);
      const bool dest_ok =
          (dest.ndim() == 3 && (dest.shape(0) == streams || dest.shape(0) == 1) &&
           dest.shape(1) == rows && dest.shape(2) == dim) ||
          (dest.ndim() == 2 && dest.shape(0) == rows && dest.shape(1) == dim);
      if (streams <= 0 || rows <= 0 || dim <= 0 || (dim % 4) != 0 || !dest_ok ||
          down_w.ndim() != 3 || down_w.shape(0) != streams ||
          down_w.shape(1) <= 0 || down_w.shape(2) != 2 * dim) {
        return nullptr;
      }
      rank = static_cast<uint32_t>(down_w.shape(1));
      out_shape = Shape{streams, rows, static_cast<int>(rank)};
    }
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2LoopCtrlDown>(s, eps, rank, out_shape);
    auto res = array(out_shape, float32, std::move(prim), {dest, msg, down_w});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_loop_ctrl_down", e.what());
    return nullptr;
  }
#else
  (void)dest_ptr; (void)msg_ptr; (void)down_w_ptr; (void)eps;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_loop_ctrl_up_blend_norm(
    mlx_array* h_ptr,
    mlx_array* gain_forget_w_ptr,
    mlx_array* dest_ptr,
    mlx_array* msg_ptr,
    mlx_array* norm_weight_ptr,
    float eps,
    bool output_f32) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!h_ptr || !gain_forget_w_ptr || !dest_ptr || !msg_ptr || !norm_weight_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& h = *reinterpret_cast<array*>(h_ptr);
    const auto& gf_w = *reinterpret_cast<array*>(gain_forget_w_ptr);
    const auto& dest = *reinterpret_cast<array*>(dest_ptr);
    const auto& msg = *reinterpret_cast<array*>(msg_ptr);
    const auto& weight = *reinterpret_cast<array*>(norm_weight_ptr);
    if (h.dtype() != float32 || !is_bf16_or_f32(gf_w.dtype()) ||
        !is_bf16_or_f32(dest.dtype()) || !is_bf16_or_f32(msg.dtype()) ||
        !is_bf16_or_f32(weight.dtype()) || msg.size() == 0 ||
        (msg.ndim() != 2 && msg.ndim() != 3)) {
      return nullptr;
    }
    uint32_t rank = 0;
    if (msg.ndim() == 2) {
      const int rows = msg.shape(0);
      const int dim = msg.shape(1);
      if (rows <= 0 || dim <= 0 ||
          dest.ndim() != 2 || dest.shape(0) != rows || dest.shape(1) != dim ||
          h.ndim() != 2 || h.shape(0) != rows || h.shape(1) <= 0 || (h.shape(1) % 4) != 0 ||
          gf_w.ndim() != 2 || gf_w.shape(0) != 2 * dim || gf_w.shape(1) != h.shape(1) ||
          weight.ndim() != 1 || weight.shape(0) != dim) {
        return nullptr;
      }
      rank = static_cast<uint32_t>(h.shape(1));
    } else {
      const int streams = msg.shape(0);
      const int rows = msg.shape(1);
      const int dim = msg.shape(2);
      const bool dest_ok =
          (dest.ndim() == 3 && (dest.shape(0) == streams || dest.shape(0) == 1) &&
           dest.shape(1) == rows && dest.shape(2) == dim) ||
          (dest.ndim() == 2 && dest.shape(0) == rows && dest.shape(1) == dim);
      const bool weight_ok =
          (weight.ndim() == 1 && weight.shape(0) == dim) ||
          (weight.ndim() == 2 && (weight.shape(0) == streams || weight.shape(0) == 1) &&
           weight.shape(1) == dim);
      if (streams <= 0 || rows <= 0 || dim <= 0 || !dest_ok || !weight_ok ||
          h.ndim() != 3 || h.shape(0) != streams || h.shape(1) != rows ||
          h.shape(2) <= 0 || (h.shape(2) % 4) != 0 ||
          gf_w.ndim() != 3 || gf_w.shape(0) != streams ||
          gf_w.shape(1) != 2 * dim || gf_w.shape(2) != h.shape(2)) {
        return nullptr;
      }
      rank = static_cast<uint32_t>(h.shape(2));
    }
    const Dtype out_dtype = output_f32 ? float32 : bfloat16;
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2LoopCtrlUpBlendNorm>(
        s, eps, rank, out_dtype, msg.shape());
    auto res = array(
        msg.shape(), out_dtype, std::move(prim),
        {h, gf_w, dest, msg, weight});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_loop_ctrl_up_blend_norm", e.what());
    return nullptr;
  }
#else
  (void)h_ptr; (void)gain_forget_w_ptr; (void)dest_ptr; (void)msg_ptr;
  (void)norm_weight_ptr; (void)eps; (void)output_f32;
  return nullptr;
#endif
}

bool mlx_inkling2_sconv_k4_ragged(
    mlx_array* x_ptr,
    mlx_array* weight_ptr,
    mlx_array* dyn_taps_ptr,
    mlx_array* cu_seqlens_ptr,
    mlx_array* res_a_ptr,
    mlx_array* res_b_ptr,
    mlx_array* const* hists_ptr,
    uint32_t num_seqs,
    bool packed_hist,
    mlx_array** out_y,
    mlx_array** out_history) {
  if (out_y) *out_y = nullptr;
  if (out_history) *out_history = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !weight_ptr || !hists_ptr || !out_y || !out_history || num_seqs == 0) {
    return false;
  }
  if (!packed_hist && num_seqs > 4u) {
    return false;
  }
  if (num_seqs > 1u && !cu_seqlens_ptr) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(weight_ptr);
    if (x.ndim() != 2 || !is_bf16_or_f32(x.dtype()) ||
        x.shape(0) <= 0 || x.shape(1) <= 0 ||
        w.ndim() != 2 || !is_bf16_or_f32(w.dtype()) ||
        w.shape(0) != x.shape(1) || w.shape(1) != 4) {
      return false;
    }
    const int t = x.shape(0);
    const int c = x.shape(1);
    std::vector<array> inputs = {x, w};
    const bool is_dynamic = dyn_taps_ptr != nullptr;
    uint32_t num_kv_heads = 0;
    if (is_dynamic) {
      const auto& dyn = *reinterpret_cast<array*>(dyn_taps_ptr);
      if (dyn.ndim() != 3 || !is_bf16_or_f32(dyn.dtype()) ||
          dyn.shape(0) != t || dyn.shape(1) <= 0 || dyn.shape(2) != 4 ||
          c % dyn.shape(1) != 0) {
        return false;
      }
      num_kv_heads = static_cast<uint32_t>(dyn.shape(1));
      inputs.push_back(dyn);
    }
    if (num_seqs > 1u) {
      const auto& cu = *reinterpret_cast<array*>(cu_seqlens_ptr);
      if (cu.ndim() != 1 || cu.dtype() != int32 || cu.shape(0) != static_cast<int>(num_seqs + 1u)) {
        return false;
      }
      inputs.push_back(cu);
    }
    const bool has_res_a = res_a_ptr != nullptr;
    if (has_res_a) {
      const auto& ra = *reinterpret_cast<array*>(res_a_ptr);
      if (ra.ndim() != 2 || ra.shape(0) != t || ra.shape(1) != c || !is_bf16_or_f32(ra.dtype())) {
        return false;
      }
      inputs.push_back(ra);
    }
    const bool has_res_b = res_b_ptr != nullptr;
    if (has_res_b) {
      const auto& rb = *reinterpret_cast<array*>(res_b_ptr);
      if (rb.ndim() != 2 || rb.shape(0) != t || rb.shape(1) != c || !is_bf16_or_f32(rb.dtype())) {
        return false;
      }
      inputs.push_back(rb);
    }
    const uint32_t h_count = packed_hist ? 1u : num_seqs;
    for (uint32_t i = 0; i < h_count; ++i) {
      if (!hists_ptr[i]) return false;
      const auto& h = *reinterpret_cast<array*>(hists_ptr[i]);
      if (!is_bf16_or_f32(h.dtype())) return false;
      if (packed_hist) {
        if (h.ndim() != 3 || h.shape(0) != static_cast<int>(num_seqs) || h.shape(1) != 3 || h.shape(2) != c) return false;
      } else {
        if (h.ndim() != 2 || h.shape(0) != 3 || h.shape(1) != c) return false;
      }
      inputs.push_back(h);
    }
    Shape hs = (num_seqs == 1u && !packed_hist)
        ? Shape{3, c}
        : Shape{static_cast<int>(num_seqs), 3, c};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2ShortConvK4Ragged>(
        s, num_seqs, is_dynamic, num_kv_heads, packed_hist, has_res_a, has_res_b, x.dtype());
    auto results = array::make_arrays(
        {x.shape(), hs},
        {x.dtype(), x.dtype()},
        std::move(prim),
        std::move(inputs));
    *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_history = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_sconv_k4_ragged", e.what());
    return false;
  }
#else
  (void)x_ptr; (void)weight_ptr; (void)dyn_taps_ptr; (void)cu_seqlens_ptr;
  (void)res_a_ptr; (void)res_b_ptr; (void)hists_ptr; (void)num_seqs; (void)packed_hist;
  return false;
#endif
}

bool mlx_inkling2_moe_route(
    mlx_array* logits_ptr,
    mlx_array* bias_ptr,
    mlx_array* down_scale_ptr,
    uint32_t top_k,
    float route_scale,
    mlx_array** out_indices,
    mlx_array** out_weights) {
  if (out_indices) *out_indices = nullptr;
  if (out_weights) *out_weights = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!logits_ptr || !out_indices || !out_weights || top_k == 0 || top_k > 16u) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return false;
    }
    const auto& logits = *reinterpret_cast<array*>(logits_ptr);
    if (logits.ndim() != 2 || !is_bf16_or_f32(logits.dtype()) ||
        logits.shape(0) <= 0 || logits.shape(1) < static_cast<int>(top_k) ||
        logits.shape(1) > 512) {
      return false;
    }
    const int num_experts = logits.shape(1);
    std::vector<array> inputs = {logits};
    const bool has_bias = bias_ptr != nullptr;
    if (has_bias) {
      const auto& b = *reinterpret_cast<array*>(bias_ptr);
      if (b.ndim() != 1 || b.shape(0) != num_experts || !is_bf16_or_f32(b.dtype())) {
        return false;
      }
      inputs.push_back(b);
    }
    const bool has_ds = down_scale_ptr != nullptr;
    if (has_ds) {
      const auto& ds = *reinterpret_cast<array*>(down_scale_ptr);
      if (ds.ndim() != 1 || ds.shape(0) != num_experts || !is_bf16_or_f32(ds.dtype())) {
        return false;
      }
      inputs.push_back(ds);
    }
    Shape out_s{logits.shape(0), static_cast<int>(top_k)};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2MoeRoute>(s, top_k, route_scale, has_bias, has_ds);
    auto results = array::make_arrays(
        {out_s, out_s},
        {int32, float32},
        std::move(prim),
        std::move(inputs));
    *out_indices = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_weights = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_moe_route", e.what());
    return false;
  }
#else
  (void)logits_ptr; (void)bias_ptr; (void)down_scale_ptr; (void)top_k; (void)route_scale;
  return false;
#endif
}

mlx_array* mlx_inkling2_moe_combine(
    mlx_array* down_out_ptr,
    mlx_array* weights_ptr,
    mlx_array* inv_order_ptr,
    bool output_f32) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!down_out_ptr || !weights_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& down = *reinterpret_cast<array*>(down_out_ptr);
    const auto& weights = *reinterpret_cast<array*>(weights_ptr);
    if (!is_bf16_or_f32(down.dtype()) || !is_bf16_or_f32(weights.dtype()) ||
        weights.ndim() != 2 || weights.shape(0) <= 0 ||
        weights.shape(1) <= 0 || weights.shape(1) > 16 ||
        down.ndim() < 2 || down.shape(-1) <= 0) {
      return nullptr;
    }
    const uint32_t rows = static_cast<uint32_t>(weights.shape(0));
    const uint32_t top_k = static_cast<uint32_t>(weights.shape(1));
    const uint32_t dim = static_cast<uint32_t>(down.shape(-1));
    if (static_cast<uint64_t>(down.size()) !=
        static_cast<uint64_t>(rows) * top_k * dim) {
      return nullptr;
    }
    std::vector<array> inputs = {down, weights};
    const bool has_inv = inv_order_ptr != nullptr;
    if (has_inv) {
      const auto& inv = *reinterpret_cast<array*>(inv_order_ptr);
      if ((inv.dtype() != uint32 && inv.dtype() != int32) ||
          static_cast<uint64_t>(inv.size()) != static_cast<uint64_t>(rows) * top_k) {
        return nullptr;
      }
      inputs.push_back(inv);
    }
    const Dtype out_dtype = output_f32 ? float32 : bfloat16;
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2MoeCombine>(
        s, rows, top_k, dim, has_inv, out_dtype);
    auto res = array(
        Shape{static_cast<int>(rows), static_cast<int>(dim)},
        out_dtype,
        std::move(prim),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_moe_combine", e.what());
    return nullptr;
  }
#else
  (void)down_out_ptr; (void)weights_ptr; (void)inv_order_ptr; (void)output_f32;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_lm_head_softcap(
    mlx_array* logits_ptr,
    mlx_array* bias_ptr,
    uint32_t out_cols,
    float softcap,
    bool bias_round_bf16) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!logits_ptr || out_cols == 0) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& logits = *reinterpret_cast<array*>(logits_ptr);
    if (logits.ndim() != 2 || !is_bf16_or_f32(logits.dtype()) ||
        logits.shape(0) <= 0 || static_cast<uint32_t>(logits.shape(1)) < out_cols) {
      return nullptr;
    }
    std::vector<array> inputs = {logits};
    const bool has_bias = bias_ptr != nullptr;
    if (has_bias) {
      const auto& bias = *reinterpret_cast<array*>(bias_ptr);
      if (bias.ndim() != 2 || !is_bf16_or_f32(bias.dtype()) ||
          (bias.shape(0) != 1 && bias.shape(0) != logits.shape(0)) ||
          static_cast<uint32_t>(bias.shape(1)) < out_cols) {
        return nullptr;
      }
      inputs.push_back(bias);
    }
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2LmHeadSoftcap>(
        s, out_cols, softcap, has_bias, bias_round_bf16);
    auto res = array(
        Shape{logits.shape(0), static_cast<int>(out_cols)},
        float32,
        std::move(prim),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_lm_head_softcap", e.what());
    return nullptr;
  }
#else
  (void)logits_ptr; (void)bias_ptr; (void)out_cols; (void)softcap; (void)bias_round_bf16;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_lm_head_softcap_argmax(
    mlx_array* logits_ptr,
    mlx_array* bias_ptr,
    uint32_t out_cols,
    float softcap,
    bool bias_round_bf16) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!logits_ptr || out_cols == 0) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& logits = *reinterpret_cast<array*>(logits_ptr);
    if (logits.ndim() != 2 || !is_bf16_or_f32(logits.dtype()) ||
        logits.shape(0) <= 0 || static_cast<uint32_t>(logits.shape(1)) < out_cols) {
      return nullptr;
    }
    std::vector<array> inputs = {logits};
    const bool has_bias = bias_ptr != nullptr;
    if (has_bias) {
      const auto& bias = *reinterpret_cast<array*>(bias_ptr);
      if (bias.ndim() != 2 || !is_bf16_or_f32(bias.dtype()) ||
          (bias.shape(0) != 1 && bias.shape(0) != logits.shape(0)) ||
          static_cast<uint32_t>(bias.shape(1)) < out_cols) {
        return nullptr;
      }
      inputs.push_back(bias);
    }
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2LmHeadSoftcapArgmax>(
        s, out_cols, softcap, has_bias, bias_round_bf16);
    auto res = array(
        Shape{logits.shape(0)},
        int32,
        std::move(prim),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_lm_head_softcap_argmax", e.what());
    return nullptr;
  }
#else
  (void)logits_ptr; (void)bias_ptr; (void)out_cols; (void)softcap; (void)bias_round_bf16;
  return nullptr;
#endif
}

bool mlx_inkling2_markov_gate(
    mlx_array* raw_ptr,
    mlx_array* ctx_ptr,
    mlx_array* state_ptr,
    mlx_array** out_next_state,
    mlx_array** out_latent) {
  if (out_next_state) *out_next_state = nullptr;
  if (out_latent) *out_latent = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!raw_ptr || !state_ptr || !out_next_state || !out_latent) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return false;
    }
    const auto& raw = *reinterpret_cast<array*>(raw_ptr);
    const auto& state = *reinterpret_cast<array*>(state_ptr);
    if (raw.ndim() != 2 || state.ndim() != 2 ||
        !is_bf16_or_f32(raw.dtype()) || !is_bf16_or_f32(state.dtype()) ||
        raw.shape(0) <= 0 || state.shape(0) != raw.shape(0) ||
        state.shape(1) <= 0 || raw.shape(1) != state.shape(1) * 3) {
      return false;
    }
    const uint32_t rank = static_cast<uint32_t>(state.shape(1));
    std::vector<array> inputs = {raw};
    const bool has_ctx = ctx_ptr != nullptr;
    if (has_ctx) {
      const auto& ctx = *reinterpret_cast<array*>(ctx_ptr);
      if (ctx.ndim() != 2 || !is_bf16_or_f32(ctx.dtype()) ||
          ctx.shape(0) != raw.shape(0) || ctx.shape(1) != raw.shape(1)) {
        return false;
      }
      inputs.push_back(ctx);
    }
    inputs.push_back(state);

    Shape out_s{raw.shape(0), static_cast<int>(rank)};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2MarkovGate>(s, rank, has_ctx);
    auto results = array::make_arrays(
        {out_s, out_s},
        {bfloat16, bfloat16},
        std::move(prim),
        std::move(inputs));
    *out_next_state = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_latent = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_markov_gate", e.what());
    return false;
  }
#else
  (void)raw_ptr; (void)ctx_ptr; (void)state_ptr;
  return false;
#endif
}

static mlx_array* mlx_inkling2_markov_fold_softcap_common(
    mlx_array* latent_ptr,
    mlx_array* fold_w_ptr,
    mlx_array* fold_scales_ptr,
    mlx_array* fold_biases_ptr,
    mlx_array* fold_bias_ptr,
    mlx_array* base_logits_ptr,
    uint32_t out_cols,
    float softcap,
    bool argmax) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!latent_ptr || !fold_w_ptr || !fold_scales_ptr || !fold_bias_ptr ||
      !base_logits_ptr || out_cols == 0) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& latent = *reinterpret_cast<array*>(latent_ptr);
    const auto& fold_w = *reinterpret_cast<array*>(fold_w_ptr);
    const auto& fold_s = *reinterpret_cast<array*>(fold_scales_ptr);
    const auto& fold_bias = *reinterpret_cast<array*>(fold_bias_ptr);
    const auto& base_logits = *reinterpret_cast<array*>(base_logits_ptr);

    if (latent.ndim() != 2 || !is_bf16_or_f32(latent.dtype()) ||
        latent.shape(0) <= 0 || latent.shape(1) <= 0 || (latent.shape(1) % 32) != 0) {
      return nullptr;
    }
    const int batch = latent.shape(0);
    const int rank = latent.shape(1);
    const int n_groups = rank / 32;
    if (fold_w.ndim() != 2 || fold_w.dtype() != uint32 ||
        static_cast<uint32_t>(fold_w.shape(0)) < out_cols ||
        fold_w.shape(1) != rank / 4) {
      return nullptr;
    }
    if (fold_s.ndim() != 2 || !is_sidecar_float(fold_s.dtype()) ||
        fold_s.shape(0) != fold_w.shape(0) || fold_s.shape(1) != n_groups) {
      return nullptr;
    }
    const bool has_biases = fold_biases_ptr != nullptr;
    std::vector<array> inputs = {latent, fold_w, fold_s};
    if (has_biases) {
      const auto& fold_b = *reinterpret_cast<array*>(fold_biases_ptr);
      if (fold_b.ndim() != 2 || fold_b.dtype() != fold_s.dtype() ||
          fold_b.shape(0) != fold_w.shape(0) || fold_b.shape(1) != n_groups) {
        return nullptr;
      }
      inputs.push_back(fold_b);
    }
    const bool bias_ok = is_bf16_or_f32(fold_bias.dtype()) &&
        ((fold_bias.ndim() == 1 && static_cast<uint32_t>(fold_bias.shape(0)) >= out_cols) ||
         (fold_bias.ndim() == 2 && fold_bias.shape(0) == 1 &&
          static_cast<uint32_t>(fold_bias.shape(1)) >= out_cols));
    if (!bias_ok) {
      return nullptr;
    }
    if (base_logits.ndim() != 2 || !is_bf16_or_f32(base_logits.dtype()) ||
        base_logits.shape(0) != batch ||
        static_cast<uint32_t>(base_logits.shape(1)) < out_cols) {
      return nullptr;
    }
    inputs.push_back(fold_bias);
    inputs.push_back(base_logits);

    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2MarkovFoldSoftcap>(
        s, out_cols, static_cast<uint32_t>(rank), softcap, has_biases, argmax);
    Shape out_shape = argmax ? Shape{batch} : Shape{batch, static_cast<int>(out_cols)};
    Dtype out_dtype = argmax ? int32 : float32;
    auto res = array(std::move(out_shape), out_dtype, std::move(prim), std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error(
        argmax ? "inkling2_markov_fold_softcap_argmax" : "inkling2_markov_fold_softcap",
        e.what());
    return nullptr;
  }
#else
  (void)latent_ptr; (void)fold_w_ptr; (void)fold_scales_ptr; (void)fold_biases_ptr;
  (void)fold_bias_ptr; (void)base_logits_ptr; (void)out_cols; (void)softcap; (void)argmax;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_markov_fold_softcap(
    mlx_array* latent_ptr,
    mlx_array* fold_w_ptr,
    mlx_array* fold_scales_ptr,
    mlx_array* fold_biases_ptr,
    mlx_array* fold_bias_ptr,
    mlx_array* base_logits_ptr,
    uint32_t out_cols,
    float softcap) {
  return mlx_inkling2_markov_fold_softcap_common(
      latent_ptr, fold_w_ptr, fold_scales_ptr, fold_biases_ptr,
      fold_bias_ptr, base_logits_ptr, out_cols, softcap, false);
}

mlx_array* mlx_inkling2_markov_fold_softcap_argmax(
    mlx_array* latent_ptr,
    mlx_array* fold_w_ptr,
    mlx_array* fold_scales_ptr,
    mlx_array* fold_biases_ptr,
    mlx_array* fold_bias_ptr,
    mlx_array* base_logits_ptr,
    uint32_t out_cols,
    float softcap) {
  return mlx_inkling2_markov_fold_softcap_common(
      latent_ptr, fold_w_ptr, fold_scales_ptr, fold_biases_ptr,
      fold_bias_ptr, base_logits_ptr, out_cols, softcap, true);
}

mlx_array* mlx_inkling2_bf16_linear(mlx_array* x_ptr, mlx_array* w_ptr) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !w_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available() || !metal::is_nax_available()) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(w_ptr);
    if (x.ndim() != 2 || w.ndim() != 2 ||
        x.dtype() != bfloat16 || w.dtype() != bfloat16 ||
        x.shape(0) <= 0 || x.shape(0) > 1024 ||
        w.shape(0) <= 0 || x.shape(1) <= 0 ||
        x.shape(1) != w.shape(1) || (x.shape(1) % 4) != 0) {
      return nullptr;
    }
    const uint32_t rows = static_cast<uint32_t>(x.shape(0));
    const uint32_t out_dim = static_cast<uint32_t>(w.shape(0));
    const uint32_t in_dim = static_cast<uint32_t>(x.shape(1));
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2Bf16Linear>(s, rows, out_dim, in_dim);
    auto res = array(
        Shape{static_cast<int>(rows), static_cast<int>(out_dim)},
        bfloat16,
        std::move(prim),
        {x, w});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_bf16_linear", e.what());
    return nullptr;
  }
#else
  (void)x_ptr; (void)w_ptr;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_bf16_lm_head_softcap_argmax(
    mlx_array* x_ptr,
    mlx_array* w_ptr,
    uint32_t out_cols,
    float softcap) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !w_ptr || out_cols == 0u) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available() || !metal::is_nax_available()) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(w_ptr);
    if (x.ndim() != 2 || w.ndim() != 2 ||
        x.dtype() != bfloat16 || w.dtype() != bfloat16 ||
        x.shape(0) <= 0 || x.shape(0) > 32 ||
        static_cast<uint32_t>(w.shape(0)) < out_cols ||
        x.shape(1) <= 0 || x.shape(1) != w.shape(1) || (x.shape(1) % 4) != 0) {
      return nullptr;
    }
    const uint32_t rows = static_cast<uint32_t>(x.shape(0));
    const uint32_t in_dim = static_cast<uint32_t>(x.shape(1));
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2Bf16LmHeadSoftcapArgmax>(
        s, rows, out_cols, in_dim, softcap);
    auto res = array(
        Shape{static_cast<int>(rows)},
        int32,
        std::move(prim),
        {x, w});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_bf16_lm_head_softcap_argmax", e.what());
    return nullptr;
  }
#else
  (void)x_ptr; (void)w_ptr; (void)out_cols; (void)softcap;
  return nullptr;
#endif
}

uint32_t mlx_inkling2_attn_res_vec4_max_t() {
#ifdef MLX_NODE_METAL_ENABLED
  return mlx::core::inkling2_ops::resolved_attn_res_vec4_max_t();
#else
  return 0u;
#endif
}

} // extern "C"
