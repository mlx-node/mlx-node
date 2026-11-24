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
#include <unordered_map>
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

uint32_t resolved_q8_simd_max_m() {
  static const uint32_t kMaxM = []() -> uint32_t {
    if (const char* env = std::getenv("MLX_INKLING2_Q8_SIMD_MAX_M")) {
      const int v = std::atoi(env);
      return v >= 0 ? std::min<uint32_t>(static_cast<uint32_t>(v), 32u) : 0u;
    }
    return 0u;
  }();
  return kMaxM;
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
  uint32_t has_tree;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(sizeof(Inkling2SconvK4Params) == 48);

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
  uint32_t has_tree;
  uint32_t _pad1;
  uint32_t k_hist_strides[4];
  uint32_t v_hist_strides[4];
};
static_assert(sizeof(Inkling2SconvK4KvPrepParams) == 96);

struct Inkling2SconvK4KvPrepPagedParams {
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
  uint32_t block_size;
  uint32_t kv_mode;
  uint32_t k_hist_strides[4];
  uint32_t v_hist_strides[4];
};
static_assert(sizeof(Inkling2SconvK4KvPrepPagedParams) == 96);

struct Inkling2QrClipSconvK4KvPrepPagedParams {
  Inkling2QrClipParams qr;
  Inkling2SconvK4KvPrepPagedParams kv;
};
static_assert(sizeof(Inkling2QrClipSconvK4KvPrepPagedParams) == 144);

struct Inkling2PagedKvPathCommitParams {
  uint32_t num_copies;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t block_size;
  uint32_t kv_mode;
  uint32_t _pad0;
  uint64_t total_slots;
};
static_assert(sizeof(Inkling2PagedKvPathCommitParams) == 32);

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
  uint32_t has_tree;
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

struct Inkling2Q8SymParams {
  uint32_t rows;
  uint32_t out_dim;
  uint32_t in_dim;
  uint32_t num_splits;
  float softcap;
  uint32_t aux_dim;
  uint32_t flags;
  uint32_t _pad;
};
static_assert(sizeof(Inkling2Q8SymParams) == 32);

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
inline bool validate_tree_rows_array(const array& tr, int expected_rows) {
  return tr.ndim() == 2 && tr.dtype() == uint32 &&
      tr.shape(0) == expected_rows && tr.shape(1) == 2;
}

MTL::ComputePipelineState* get_cached_inkling2_kernel(
    metal::Device& device,
    const char* op_name,
    const char* kernel_name) {
  thread_local metal::Device* cached_device = nullptr;
  thread_local std::unordered_map<const char*, MTL::ComputePipelineState*> cache;
  if (cached_device != &device) {
    cache.clear();
    cached_device = &device;
  }
  auto it = cache.find(kernel_name);
  if (it != cache.end()) {
    return it->second;
  }
  auto* pipe = fast::paged::get_prebuilt_kernel(
      device, "inkling2_ops", op_name, kernel_name);
  cache.emplace(kernel_name, pipe);
  return pipe;
}

// Bind an in-place KV pool output without charging the full 32K pool
// `data_size()` (~67M-268M elements > MLX's 50M CB limit) to
// `CommandEncoder::buffer_sizes_`. `set_buffer` records the buffer in both
// `all_inputs_` and `all_outputs_` for cross-CB fence sync, and
// `register_output_array` checks `prev_inputs_` (WAR barrier if a prior read
// touched this pool) and records `next_outputs_` (so the following
// `PagedAttention` read emits a RAW barrier automatically) — ZERO manual
// `enc.barrier()` needed for pure pool writes!
inline void bind_inplace_pool_output(
    metal::CommandEncoder& enc,
    const array& pool,
    int slot) {
  enc.set_buffer(
      static_cast<const MTL::Buffer*>(pool.buffer().ptr()),
      slot,
      pool.offset());
  enc.register_output_array(pool);
}

bool inkling2_encoder_alive(int stream_index) {
  auto& local = metal::get_command_encoders();
  if (local.find(stream_index) != local.end()) {
    return true;
  }
  auto& global = metal::get_global_command_encoders();
  return global.find(stream_index) != global.end();
}

template <typename Map>
void reap_destroyed_scratch(Map& entries) {
  for (auto it = entries.begin(); it != entries.end();) {
    // A missing encoder has already synchronized in its destructor.
    if (inkling2_encoder_alive(it->first)) {
      ++it;
    } else {
      it = entries.erase(it);
    }
  }
}

struct Inkling2OProjScratchEntry {
  // Owns `{buf}` in `owner.inputs()` so `enc.retain_inputs(owner)` pins the
  // Metal buffer on the unretained command buffer without stripping fences.
  array owner;
};

std::unordered_map<int, Inkling2OProjScratchEntry>& inkling2_oproj_scratch_pool() {
  static thread_local std::unordered_map<int, Inkling2OProjScratchEntry> by_stream;
  return by_stream;
}

constexpr size_t kInkling2MaxOProjScratchElems = size_t(1) << 20; // 2 MiB BF16

// Persistent per-stream `[T, D]` BF16 scratch for fused `o_proj + attn_post_conv`
// when no rollback tape is needed (zero per-visit `allocator::malloc`!).
array inkling2_oproj_scratch(
    const Stream& s,
    metal::CommandEncoder& enc,
    size_t min_elems) {
  if (min_elems > kInkling2MaxOProjScratchElems) {
    array tmp(Shape{static_cast<int>(min_elems)}, bfloat16, nullptr, {});
    tmp.set_data(allocator::malloc(tmp.nbytes()));
    enc.add_temporary(tmp);
    return tmp;
  }
  auto& by_stream = inkling2_oproj_scratch_pool();
  auto it = by_stream.find(s.index);
  if (it != by_stream.end()) {
    const auto& slots = it->second.owner.inputs();
    if (slots[0].size() >= min_elems) {
      enc.retain_inputs(it->second.owner);
      return slots[0];
    }
  } else {
    reap_destroyed_scratch(by_stream);
  }
  const size_t size = std::min(
      kInkling2MaxOProjScratchElems,
      std::max<size_t>(min_elems, size_t(32u) * 2304u));
  array buf(Shape{static_cast<int>(size)}, bfloat16, nullptr, {});
  buf.set_data(allocator::malloc(buf.nbytes()));
  array owner(Shape{}, uint8, nullptr, std::vector<array>{buf});
  enc.retain_inputs(owner);
  if (it != by_stream.end()) {
    // Previous dispatches on `enc` already pinned the old buffer via
    // `enc.retain_inputs(old_owner)`, so replacing the entry is safe.
    it->second.owner = std::move(owner);
  } else {
    by_stream.emplace(s.index, Inkling2OProjScratchEntry{std::move(owner)});
  }
  return buf;
}

struct Inkling2LoopCtrlScratch {
  array h;
  array blended;
  array partial_sq;
};

struct Inkling2LoopCtrlScratchEntry {
  // Owns `{h, blended, partial_sq}` in `owner.inputs()` so
  // `enc.retain_inputs(owner)` pins all three Metal buffers safely.
  array owner;
};

std::unordered_map<int, Inkling2LoopCtrlScratchEntry>& inkling2_loop_ctrl_scratch_pool() {
  static thread_local std::unordered_map<int, Inkling2LoopCtrlScratchEntry> by_stream;
  return by_stream;
}

constexpr size_t kInkling2MaxLoopScratchFloats = size_t(1) << 20; // 4 MiB total FP32

// Persistent per-stream FP32 scratch arena (`h`, `blended`, `partial_sq`) for
// `Inkling2LoopCtrlFused` and `Inkling2LoopCtrlUpBlendNorm` (< 1 MiB for S=3, T=32, D=2304).
Inkling2LoopCtrlScratch inkling2_loop_ctrl_scratch(
    const Stream& s,
    metal::CommandEncoder& enc,
    size_t min_h,
    size_t min_blended,
    size_t min_sq) {
  if (min_h + min_blended + min_sq > kInkling2MaxLoopScratchFloats) {
    auto mk = [&](size_t n) {
      array tmp(Shape{static_cast<int>(std::max<size_t>(n, 1u))}, float32, nullptr, {});
      tmp.set_data(allocator::malloc(tmp.nbytes()));
      enc.add_temporary(tmp);
      return tmp;
    };
    return {mk(min_h), mk(min_blended), mk(min_sq)};
  }
  auto& by_stream = inkling2_loop_ctrl_scratch_pool();
  auto it = by_stream.find(s.index);
  if (it != by_stream.end()) {
    const auto& slots = it->second.owner.inputs();
    if (slots[0].size() >= min_h &&
        slots[1].size() >= min_blended &&
        slots[2].size() >= min_sq) {
      enc.retain_inputs(it->second.owner);
      return {slots[0], slots[1], slots[2]};
    }
  } else {
    reap_destroyed_scratch(by_stream);
  }
  const size_t alloc_h = std::max<size_t>(min_h, size_t(3u) * 32u * 64u);
  const size_t alloc_blended = std::max<size_t>(min_blended, size_t(3u) * 32u * 2304u);
  const size_t alloc_sq = std::max<size_t>(min_sq, size_t(3u) * 32u * 128u);
  auto mk_persist = [](size_t n) {
    array buf(Shape{static_cast<int>(n)}, float32, nullptr, {});
    buf.set_data(allocator::malloc(buf.nbytes()));
    return buf;
  };
  array h = mk_persist(alloc_h);
  array blended = mk_persist(alloc_blended);
  array partial_sq = mk_persist(alloc_sq);
  array owner(Shape{}, uint8, nullptr, std::vector<array>{h, blended, partial_sq});
  enc.retain_inputs(owner);
  if (it != by_stream.end()) {
    // Previous dispatches on `enc` already pinned the old buffers via
    // `enc.retain_inputs(old_owner)`, so replacing the entry is safe.
    it->second.owner = std::move(owner);
  } else {
    by_stream.emplace(s.index, Inkling2LoopCtrlScratchEntry{std::move(owner)});
  }
  return {std::move(h), std::move(blended), std::move(partial_sq)};
}

// Shared exact BF16 linear encoder used by `Inkling2Bf16Linear` and `Inkling2OProjSconvResidual`.
void encode_bf16_linear(
    metal::Device& device,
    metal::CommandEncoder& enc,
    const array& x,
    const array& w,
    array& out,
    uint32_t rows,
    uint32_t out_dim,
    uint32_t in_dim) {
  Inkling2Bf16LinearParams params{rows, out_dim, in_dim, 0u, 0.0f, 0u, 0u, 0u};
  if (rows == 1u && (in_dim % 8u) == 0u) {
    const uint32_t num_tgs = (out_dim + 3u) / 4u;
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "bf16_linear_m1", "inkling2_bf16_linear_m1"));
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(params, 3);
    enc.dispatch_threadgroups(MTL::Size(num_tgs, 1, 1), MTL::Size(128, 1, 1));
  } else if (metal::is_nax_available() && rows > resolved_q8_simd_max_m()) {
    const bool use_m8 = rows <= 8u;
    const uint32_t tiles_m = use_m8 ? 1u : ((rows + 31u) / 32u);
    const uint32_t tiles_n = (out_dim + 15u) / 16u;
    const char* kname = use_m8 ? "inkling2_bf16_linear_nax_m8" : "inkling2_bf16_linear_batched";
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "bf16_linear_batched", kname));
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(params, 3);
    enc.dispatch_threadgroups(MTL::Size(tiles_m * tiles_n, 1, 1), MTL::Size(128, 1, 1));
  } else {
    const uint32_t rows_per_tg = rows <= 16u ? 16u : 8u;
    const uint32_t num_tgs = (out_dim + rows_per_tg - 1u) / rows_per_tg;
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "bf16_linear_simd", "inkling2_bf16_linear_simd"));
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(params, 3);
    enc.dispatch_threadgroups(MTL::Size(num_tgs, 1, 1), MTL::Size(128, 1, 1));
  }
}

// Shared exact Symmetric Q8_0 matmul encoder used by `Inkling2Q8SymMatmul` and
// `Inkling2OProjSconvResidual` (guarantees 0-ULP parity across all env modes).
void encode_q8_sym_matmul(
    metal::Device& device,
    metal::CommandEncoder& enc,
    const array& x,
    const array& w,
    const array& scales,
    const array* w_aux,
    array& out_main,
    array* out_aux,
    uint32_t rows,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t aux_dim,
    Dtype out_dtype) {
  const bool use_nax = metal::is_nax_available() && rows > resolved_q8_simd_max_m();
  const bool use_nax_m8_n16 = use_nax && rows <= 8u && out_dim <= 3072u;
  const bool use_nax_m8_n32 = use_nax && rows <= 8u && out_dim > 3072u;
  const uint32_t tiles_m =
      use_nax ? ((use_nax_m8_n16 || use_nax_m8_n32) ? 1u : ((rows + 31u) / 32u)) : 1u;
  const uint32_t rows_per_tg =
      use_nax ? (use_nax_m8_n16 ? 16u : 32u) : (rows <= 16u ? 16u : 8u);
  const uint32_t q8_tgs = tiles_m * ((out_dim + rows_per_tg - 1u) / rows_per_tg);
  const uint32_t aux_tgs =
      aux_dim > 0u ? tiles_m * ((aux_dim + rows_per_tg - 1u) / rows_per_tg) : 0u;
  const uint32_t flags =
      ((out_dtype == float32) ? 1u : 0u) | ((aux_dim > 0u) ? 2u : 0u);
  Inkling2Q8SymParams params{
      rows, out_dim, in_dim, 0u, 0.0f, aux_dim, flags, 0u};

  const uint32_t dummy = 0u;
  const char* kname = use_nax_m8_n16
      ? "inkling2_q8_sym_matmul_nax_m8_bf16"
      : (use_nax_m8_n32
             ? "inkling2_q8_sym_matmul_nax_m8_n32_bf16"
             : (use_nax ? "inkling2_q8_sym_matmul_nax_bf16"
                        : "inkling2_q8_sym_matmul_bf16"));
  enc.set_compute_pipeline_state(
      get_cached_inkling2_kernel(device, "q8_sym_matmul", kname));
  enc.set_input_array(x, 0);
  enc.set_input_array(w, 1);
  enc.set_input_array(scales, 2);
  if (aux_dim > 0u && w_aux != nullptr) {
    enc.set_input_array(*w_aux, 3);
  } else {
    enc.set_bytes(dummy, 3);
  }
  enc.set_output_array(out_main, 4);
  if (aux_dim > 0u && out_aux != nullptr) {
    enc.set_output_array(*out_aux, 5);
  } else {
    enc.set_output_array(out_main, 5);
  }
  enc.set_bytes(params, 6);
  enc.dispatch_threadgroups(
      MTL::Size(q8_tgs + aux_tgs, 1, 1), MTL::Size(128, 1, 1));
}

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
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "gate_sink", kname));
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
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "situ", kname));
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
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "loop_blend_rms_norm", kname));
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
  Inkling2ShortConvK4(
      Stream s,
      bool is_dynamic,
      uint32_t num_kv_heads,
      bool has_tree,
      Dtype out_dtype)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2ShortConvK4 requires Metal");
            }),
        is_dynamic_(is_dynamic),
        num_kv_heads_(num_kv_heads),
        has_tree_(has_tree),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2ShortConvK4)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2ShortConvK4&>(other);
    return is_dynamic_ == rhs.is_dynamic_ &&
        num_kv_heads_ == rhs.num_kv_heads_ &&
        has_tree_ == rhs.has_tree_ &&
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
    size_t idx = 3;
    std::optional<array> dyn_taps_arr;
    uint32_t dyn_is_f32 = 1u;
    if (is_dynamic_) {
      dyn_taps_arr = ensure_row_contiguous(inputs[idx++], s, enc);
      dyn_is_f32 = dyn_taps_arr->dtype() == float32 ? 1u : 0u;
    }
    std::optional<array> tree_arr;
    if (has_tree_) {
      tree_arr = ensure_row_contiguous_aligned(inputs[idx++], 8u, s, enc);
    }

    auto& y_out = outputs[0];
    auto& hist_out = outputs[1];
    y_out.set_data(allocator::malloc(y_out.nbytes()));
    hist_out.set_data(allocator::malloc(hist_out.nbytes()));

    const float dummy = 0.0f;
    const uint32_t dummy_tr[2] = {0u, 0u};

    const char* kname = out_dtype_ == float32 ? "inkling2_sconv_k4_f32"
                                              : "inkling2_sconv_k4_bf16";
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "sconv_k4", kname));
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
        x_stride,
        has_tree_ ? 1u : 0u,
        0u,
        0u,
        0u};
    enc.set_output_array(y_out, 4);
    enc.set_output_array(hist_out, 5);
    enc.set_bytes(params, 6);
    if (has_tree_) {
      enc.set_input_array(*tree_arr, 7);
    } else {
      enc.set_bytes(dummy_tr, 7);
    }
    const size_t tgx = (seq_len <= 16u) ? 64 : 256;
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
  bool has_tree_;
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
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "qr_clip", kname));
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
      bool has_tree,
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
        has_tree_(has_tree),
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
        has_tree_ == rhs.has_tree_ &&
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

    std::optional<array> tree_arr;
    if (has_tree_) {
      tree_arr = ensure_row_contiguous_aligned(inputs[idx++], 8u, s, enc);
    }

    for (int o = 0; o < 4; ++o) {
      outputs[o].set_data(allocator::malloc(outputs[o].nbytes()));
    }

    const float dummy = 0.0f;
    const int32_t dummy_cu = 0;
    const uint32_t dummy_tr[2] = {0u, 0u};
    const char* kname = out_dtype_ == float32
        ? "inkling2_sconv_k4_kv_prep_f32"
        : "inkling2_sconv_k4_kv_prep_bf16";
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "sconv_k4_kv_prep", kname));
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
        has_tree_ ? 1u : 0u,
        0u,
        {kh_strides[0], kh_strides[1], kh_strides[2], kh_strides[3]},
        {vh_strides[0], vh_strides[1], vh_strides[2], vh_strides[3]}};
    enc.set_bytes(params, 19);
    if (has_tree_) {
      enc.set_input_array(*tree_arr, 20);
    } else {
      enc.set_bytes(dummy_tr, 20);
    }
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
  bool has_tree_;
  float qk_thresh_;
  float v_eps_;
  float l2_eps_;
  Dtype out_dtype_;
};

// ============================================================================
// 6b. Inkling2ShortConvK4KvPrepPaged (fused ShortConv + L2-clip/RMSNorm + PagedKVWrite)
// ============================================================================
class Inkling2ShortConvK4KvPrepPaged : public fast::Custom {
 public:
  Inkling2ShortConvK4KvPrepPaged(
      Stream s,
      uint32_t num_kv_heads,
      uint32_t head_dim,
      uint32_t num_seqs,
      uint32_t block_size,
      uint32_t kv_mode,
      bool is_dynamic,
      bool packed_hist,
      bool has_tree,
      float qk_thresh,
      float v_eps,
      float l2_eps)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2ShortConvK4KvPrepPaged requires Metal");
            }),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        num_seqs_(num_seqs),
        block_size_(block_size),
        kv_mode_(kv_mode),
        is_dynamic_(is_dynamic),
        packed_hist_(packed_hist),
        has_tree_(has_tree),
        qk_thresh_(qk_thresh),
        v_eps_(v_eps),
        l2_eps_(l2_eps) {}

  DEFINE_NAME(Inkling2ShortConvK4KvPrepPaged)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2ShortConvK4KvPrepPaged&>(other);
    return num_kv_heads_ == rhs.num_kv_heads_ &&
        head_dim_ == rhs.head_dim_ &&
        num_seqs_ == rhs.num_seqs_ &&
        block_size_ == rhs.block_size_ &&
        kv_mode_ == rhs.kv_mode_ &&
        is_dynamic_ == rhs.is_dynamic_ &&
        packed_hist_ == rhs.packed_hist_ &&
        has_tree_ == rhs.has_tree_ &&
        qk_thresh_ == rhs.qk_thresh_ &&
        v_eps_ == rhs.v_eps_ &&
        l2_eps_ == rhs.l2_eps_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int kv_dim = static_cast<int>(num_kv_heads_ * head_dim_);
    Shape hs = (num_seqs_ == 1u && !packed_hist_)
        ? Shape{3, kv_dim}
        : Shape{static_cast<int>(num_seqs_), 3, kv_dim};
    // Outputs: {k_pool', v_pool', k_scale', v_scale', k_hist_out, v_hist_out}
    return {inputs[4].shape(), inputs[5].shape(), inputs[6].shape(), inputs[7].shape(), hs, hs};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2ShortConvK4KvPrepPaged requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    // Layout of `inputs`:
    //   0: k_in, 1: v_in, 2: k_weight, 3: v_weight
    //   4: k_pool, 5: v_pool, 6: k_scale, 7: v_scale, 8: slot_mapping
    //   [optional: dyn_k, dyn_v if is_dynamic_]
    //   [optional: cu_seqlens if num_seqs_ > 1]
    //   k_hists (1 if packed_hist_ else num_seqs_)
    //   v_hists (1 if packed_hist_ else num_seqs_)
    const uint32_t seq_len = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t kv_dim = num_kv_heads_ * head_dim_;

    const array& k_pool = inputs[4];
    const array& v_pool = inputs[5];
    const array& k_scale = inputs[6];
    const array& v_scale = inputs[7];
    const array& slot_mapping = inputs[8];

    // Runtime slot-id bounds check matching `PagedKVWrite::eval_gpu`.
    if (slot_mapping.shape(0) != static_cast<int>(seq_len) ||
        slot_mapping.dtype() != int64) {
      throw std::invalid_argument(
          "Inkling2ShortConvK4KvPrepPaged: invalid slot_mapping shape or dtype");
    }
    if (seq_len > 0u) {
      const int64_t pool_capacity =
          static_cast<int64_t>(k_pool.shape(0)) * static_cast<int64_t>(block_size_);
      const int64_t* slot_data = slot_mapping.data<int64_t>();
      int64_t max_slot = -1;
      for (uint32_t i = 0; i < seq_len; ++i) {
        if (slot_data[i] >= 0 && slot_data[i] > max_slot) {
          max_slot = slot_data[i];
        }
      }
      if (max_slot >= pool_capacity) {
        throw std::invalid_argument(
            "Inkling2ShortConvK4KvPrepPaged: slot_mapping exceeds pool capacity");
      }
    }

    // Prepare ALL inputs BEFORE setting compute pipeline state.
    auto [k_in, k_stride] = ensure_2d_strided(inputs[0], seq_len, kv_dim, s, enc);
    auto [v_in, v_stride] = ensure_2d_strided(inputs[1], seq_len, kv_dim, s, enc);
    array kw = ensure_row_contiguous(inputs[2], s, enc);
    array vw = ensure_row_contiguous(inputs[3], s, enc);
    array slots = ensure_row_contiguous(slot_mapping, s, enc);

    size_t idx = 9;
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

    std::optional<array> tree_arr;
    if (has_tree_) {
      tree_arr = ensure_row_contiguous_aligned(inputs[idx++], 8u, s, enc);
    }

    // Alias pool outputs in-place so MLX tracks the dependency chain!
    outputs[0].copy_shared_buffer(k_pool);
    outputs[1].copy_shared_buffer(v_pool);
    outputs[2].copy_shared_buffer(k_scale);
    outputs[3].copy_shared_buffer(v_scale);
    outputs[4].set_data(allocator::malloc(outputs[4].nbytes()));
    outputs[5].set_data(allocator::malloc(outputs[5].nbytes()));

    const float dummy = 0.0f;
    const int32_t dummy_cu = 0;
    const uint32_t dummy_tr[2] = {0u, 0u};
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "sconv_k4_kv_prep_paged",
        "inkling2_sconv_k4_kv_prep_paged_bf16"));
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

    enc.set_output_array(outputs[4], 15);
    enc.set_output_array(outputs[5], 16);
    bind_inplace_pool_output(enc, outputs[0], 17);
    bind_inplace_pool_output(enc, outputs[1], 18);
    if (kv_mode_ == 0u) {
      bind_inplace_pool_output(enc, outputs[2], 19);
      bind_inplace_pool_output(enc, outputs[3], 20);
    } else {
      enc.set_bytes(dummy, 19);
      enc.set_bytes(dummy, 20);
    }
    enc.set_input_array(slots, 21);

    const uint32_t flags =
        (kw.dtype() == float32 ? 1u : 0u) |
        ((vw.dtype() == float32 ? 1u : 0u) << 1u) |
        (dk_f32 << 2u) |
        (dv_f32 << 3u) |
        ((has_tree_ ? 1u : 0u) << 4u);

    Inkling2SconvK4KvPrepPagedParams params{
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
        block_size_,
        kv_mode_,
        {kh_strides[0], kh_strides[1], kh_strides[2], kh_strides[3]},
        {vh_strides[0], vh_strides[1], vh_strides[2], vh_strides[3]}};
    enc.set_bytes(params, 22);
    if (has_tree_) {
      enc.set_input_array(*tree_arr, 23);
    } else {
      enc.set_bytes(dummy_tr, 23);
    }
    enc.dispatch_threadgroups(MTL::Size(2u * num_kv_heads_, seq_len, 1), MTL::Size(32, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2ShortConvK4KvPrepPaged requires Metal");
#endif
  }

 private:
  uint32_t num_kv_heads_;
  uint32_t head_dim_;
  uint32_t num_seqs_;
  uint32_t block_size_;
  uint32_t kv_mode_;
  bool is_dynamic_;
  bool packed_hist_;
  bool has_tree_;
  float qk_thresh_;
  float v_eps_;
  float l2_eps_;
};

// ============================================================================
// 6b2. Inkling2QrClipSconvK4KvPrepPaged
//      40-core parallel fusion: `qr_clip` + `sconv_k4_kv_prep_paged` in ONE
//      Metal dispatch & ONE MLX graph op!
// ============================================================================
class Inkling2QrClipSconvK4KvPrepPaged : public fast::Custom {
 public:
  Inkling2QrClipSconvK4KvPrepPaged(
      Stream s,
      uint32_t num_q_heads,
      uint32_t num_kv_heads,
      uint32_t head_dim,
      uint32_t d_rel,
      uint32_t num_seqs,
      uint32_t block_size,
      uint32_t kv_mode,
      bool has_tau,
      bool r_round_bf16,
      bool is_dynamic,
      bool packed_hist,
      bool has_tree,
      float qk_thresh,
      float rel_thresh,
      float v_eps,
      float l2_eps)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2QrClipSconvK4KvPrepPaged requires Metal");
            }),
        num_q_heads_(num_q_heads),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        d_rel_(d_rel),
        num_seqs_(num_seqs),
        block_size_(block_size),
        kv_mode_(kv_mode),
        has_tau_(has_tau),
        r_round_bf16_(r_round_bf16),
        is_dynamic_(is_dynamic),
        packed_hist_(packed_hist),
        has_tree_(has_tree),
        qk_thresh_(qk_thresh),
        rel_thresh_(rel_thresh),
        v_eps_(v_eps),
        l2_eps_(l2_eps) {}

  DEFINE_NAME(Inkling2QrClipSconvK4KvPrepPaged)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2QrClipSconvK4KvPrepPaged&>(other);
    return num_q_heads_ == rhs.num_q_heads_ &&
        num_kv_heads_ == rhs.num_kv_heads_ &&
        head_dim_ == rhs.head_dim_ &&
        d_rel_ == rhs.d_rel_ &&
        num_seqs_ == rhs.num_seqs_ &&
        block_size_ == rhs.block_size_ &&
        kv_mode_ == rhs.kv_mode_ &&
        has_tau_ == rhs.has_tau_ &&
        r_round_bf16_ == rhs.r_round_bf16_ &&
        is_dynamic_ == rhs.is_dynamic_ &&
        packed_hist_ == rhs.packed_hist_ &&
        has_tree_ == rhs.has_tree_ &&
        qk_thresh_ == rhs.qk_thresh_ &&
        rel_thresh_ == rhs.rel_thresh_ &&
        v_eps_ == rhs.v_eps_ &&
        l2_eps_ == rhs.l2_eps_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int t = inputs[0].shape(0);
    const int kv_dim = static_cast<int>(num_kv_heads_ * head_dim_);
    Shape hs = (num_seqs_ == 1u && !packed_hist_)
        ? Shape{3, kv_dim}
        : Shape{static_cast<int>(num_seqs_), 3, kv_dim};
    // Outputs: {q_out, r_out, k_pool', v_pool', k_scale', v_scale', k_hist_out, v_hist_out}
    return {
        Shape{t, static_cast<int>(num_q_heads_), static_cast<int>(head_dim_)},
        Shape{t, static_cast<int>(num_q_heads_), static_cast<int>(d_rel_)},
        inputs[8].shape(),
        inputs[9].shape(),
        inputs[10].shape(),
        inputs[11].shape(),
        hs,
        hs};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2QrClipSconvK4KvPrepPaged requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    // Layout of `inputs`:
    //   0: q_in, 1: r_in, 2: q_clip, 3: rel_clip
    //   4: k_in, 5: v_in, 6: k_weight, 7: v_weight
    //   8: k_pool, 9: v_pool, 10: k_scale, 11: v_scale, 12: slot_mapping
    //   [optional: tau if has_tau_]
    //   [optional: dyn_k, dyn_v if is_dynamic_]
    //   [optional: cu_seqlens if num_seqs_ > 1]
    //   k_hists (1 if packed_hist_ else num_seqs_)
    //   v_hists (1 if packed_hist_ else num_seqs_)
    //   [optional: tree_rows if has_tree_]
    const uint32_t seq_len = static_cast<uint32_t>(inputs[0].shape(0));
    const uint32_t q_dim = num_q_heads_ * head_dim_;
    const uint32_t r_dim = num_q_heads_ * d_rel_;
    const uint32_t kv_dim = num_kv_heads_ * head_dim_;

    const array& k_pool = inputs[8];
    const array& v_pool = inputs[9];
    const array& k_scale = inputs[10];
    const array& v_scale = inputs[11];
    const array& slot_mapping = inputs[12];

    if (slot_mapping.shape(0) != static_cast<int>(seq_len) ||
        slot_mapping.dtype() != int64) {
      throw std::invalid_argument(
          "Inkling2QrClipSconvK4KvPrepPaged: invalid slot_mapping shape or dtype");
    }
    if (seq_len > 0u) {
      const int64_t pool_capacity =
          static_cast<int64_t>(k_pool.shape(0)) * static_cast<int64_t>(block_size_);
      const int64_t* slot_data = slot_mapping.data<int64_t>();
      int64_t max_slot = -1;
      for (uint32_t i = 0; i < seq_len; ++i) {
        if (slot_data[i] >= 0 && slot_data[i] > max_slot) {
          max_slot = slot_data[i];
        }
      }
      if (max_slot >= pool_capacity) {
        throw std::invalid_argument(
            "Inkling2QrClipSconvK4KvPrepPaged: slot_mapping exceeds pool capacity");
      }
    }

    // Prepare ALL inputs BEFORE setting compute pipeline state.
    auto [q_in, q_stride] = ensure_2d_strided(inputs[0], seq_len, q_dim, s, enc);
    auto [r_in, r_stride] = ensure_2d_strided(inputs[1], seq_len, r_dim, s, enc);
    array q_clip = ensure_row_contiguous(inputs[2], s, enc);
    array rel_clip = ensure_row_contiguous(inputs[3], s, enc);

    auto [k_in, k_stride] = ensure_2d_strided(inputs[4], seq_len, kv_dim, s, enc);
    auto [v_in, v_stride] = ensure_2d_strided(inputs[5], seq_len, kv_dim, s, enc);
    array kw = ensure_row_contiguous(inputs[6], s, enc);
    array vw = ensure_row_contiguous(inputs[7], s, enc);
    array slots = ensure_row_contiguous(slot_mapping, s, enc);

    size_t idx = 13;
    std::optional<array> tau_arr;
    uint32_t tau_f32 = 1u;
    if (has_tau_) {
      tau_arr = ensure_row_contiguous(inputs[idx++], s, enc);
      tau_f32 = tau_arr->dtype() == float32 ? 1u : 0u;
    }

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

    std::optional<array> tree_arr;
    if (has_tree_) {
      tree_arr = ensure_row_contiguous_aligned(inputs[idx++], 8u, s, enc);
    }

    outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
    outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));
    outputs[2].copy_shared_buffer(k_pool);
    outputs[3].copy_shared_buffer(v_pool);
    outputs[4].copy_shared_buffer(k_scale);
    outputs[5].copy_shared_buffer(v_scale);
    outputs[6].set_data(allocator::malloc(outputs[6].nbytes()));
    outputs[7].set_data(allocator::malloc(outputs[7].nbytes()));

    const float dummy = 0.0f;
    const float dummy_one = 1.0f;
    const int32_t dummy_cu = 0;
    const uint32_t dummy_tr[2] = {0u, 0u};

    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "qr_clip_sconv_k4_kv_prep_paged",
        "inkling2_qr_clip_sconv_k4_kv_prep_paged_bf16"));
    enc.set_input_array(q_in, 0);
    enc.set_input_array(r_in, 1);
    enc.set_input_array(q_clip, 2);
    enc.set_input_array(rel_clip, 3);
    if (has_tau_) {
      enc.set_input_array(*tau_arr, 4);
    } else {
      enc.set_bytes(dummy_one, 4);
    }
    enc.set_output_array(outputs[0], 5);
    enc.set_output_array(outputs[1], 6);
    enc.set_input_array(k_in, 7);
    enc.set_input_array(v_in, 8);
    enc.set_input_array(kw, 9);
    enc.set_input_array(vw, 10);
    if (is_dynamic_) {
      enc.set_input_array(*dk_arr, 11);
      enc.set_input_array(*dv_arr, 12);
    } else {
      enc.set_bytes(dummy, 11);
      enc.set_bytes(dummy, 12);
    }
    if (num_seqs_ > 1u) {
      enc.set_input_array(*cu_arr, 13);
    } else {
      enc.set_bytes(dummy_cu, 13);
    }
    for (uint32_t i = 0; i < 4u; ++i) {
      if (i < h_count) {
        enc.set_input_array(kh_arrs[i], 14 + static_cast<int>(i));
      } else {
        enc.set_bytes(dummy, 14 + static_cast<int>(i));
      }
    }
    for (uint32_t i = 0; i < 4u; ++i) {
      if (i < h_count) {
        enc.set_input_array(vh_arrs[i], 18 + static_cast<int>(i));
      } else {
        enc.set_bytes(dummy, 18 + static_cast<int>(i));
      }
    }
    enc.set_output_array(outputs[6], 22);
    enc.set_output_array(outputs[7], 23);
    bind_inplace_pool_output(enc, outputs[2], 24);
    bind_inplace_pool_output(enc, outputs[3], 25);
    if (kv_mode_ == 0u) {
      bind_inplace_pool_output(enc, outputs[4], 26);
      bind_inplace_pool_output(enc, outputs[5], 27);
    } else {
      enc.set_bytes(dummy, 26);
      enc.set_bytes(dummy, 27);
    }
    enc.set_input_array(slots, 28);
    if (has_tree_) {
      enc.set_input_array(*tree_arr, 29);
    } else {
      enc.set_bytes(dummy_tr, 29);
    }

    const uint32_t qr_flags =
        (q_in.dtype() == float32 ? 1u : 0u) |
        ((r_in.dtype() == float32 ? 1u : 0u) << 1u) |
        ((q_clip.dtype() == float32 ? 1u : 0u) << 2u) |
        ((rel_clip.dtype() == float32 ? 1u : 0u) << 3u) |
        ((has_tau_ ? 1u : 0u) << 4u) |
        (tau_f32 << 5u) |
        ((r_round_bf16_ ? 1u : 0u) << 6u);
    const uint32_t kv_flags =
        (kw.dtype() == float32 ? 1u : 0u) |
        ((vw.dtype() == float32 ? 1u : 0u) << 1u) |
        (dk_f32 << 2u) |
        (dv_f32 << 3u) |
        ((has_tree_ ? 1u : 0u) << 4u);

    Inkling2QrClipSconvK4KvPrepPagedParams params{
        Inkling2QrClipParams{
            seq_len,
            num_q_heads_,
            head_dim_,
            d_rel_,
            q_stride,
            r_stride,
            qk_thresh_,
            rel_thresh_,
            l2_eps_,
            qr_flags,
            0u,
            0u},
        Inkling2SconvK4KvPrepPagedParams{
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
            kv_flags,
            kh_mask,
            vh_mask,
            block_size_,
            kv_mode_,
            {kh_strides[0], kh_strides[1], kh_strides[2], kh_strides[3]},
            {vh_strides[0], vh_strides[1], vh_strides[2], vh_strides[3]}}};
    enc.set_bytes(params, 30);
    enc.dispatch_threadgroups(
        MTL::Size(num_q_heads_ + 2u * num_kv_heads_, seq_len, 1),
        MTL::Size(32, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2QrClipSconvK4KvPrepPaged requires Metal");
#endif
  }

 private:
  uint32_t num_q_heads_;
  uint32_t num_kv_heads_;
  uint32_t head_dim_;
  uint32_t d_rel_;
  uint32_t num_seqs_;
  uint32_t block_size_;
  uint32_t kv_mode_;
  bool has_tau_;
  bool r_round_bf16_;
  bool is_dynamic_;
  bool packed_hist_;
  bool has_tree_;
  float qk_thresh_;
  float rel_thresh_;
  float v_eps_;
  float l2_eps_;
};

// ============================================================================
// 6c. Inkling2PagedKvPathCommit (in-place paged KV slot compaction for tree commit)
// ============================================================================
class Inkling2PagedKvPathCommit : public fast::Custom {
 public:
  Inkling2PagedKvPathCommit(
      Stream s,
      uint32_t num_kv_heads,
      uint32_t head_dim,
      uint32_t block_size,
      uint32_t kv_mode)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2PagedKvPathCommit requires Metal");
            }),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        block_size_(block_size),
        kv_mode_(kv_mode) {}

  DEFINE_NAME(Inkling2PagedKvPathCommit)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2PagedKvPathCommit&>(other);
    return num_kv_heads_ == rhs.num_kv_heads_ &&
        head_dim_ == rhs.head_dim_ &&
        block_size_ == rhs.block_size_ &&
        kv_mode_ == rhs.kv_mode_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape(), inputs[1].shape(), inputs[2].shape(), inputs[3].shape()};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2PagedKvPathCommit requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    const array& k_pool = inputs[0];
    const array& v_pool = inputs[1];
    const array& k_scale = inputs[2];
    const array& v_scale = inputs[3];
    const array& src_slots = inputs[4];
    const array& dst_slots = inputs[5];

    const uint32_t num_copies = static_cast<uint32_t>(src_slots.shape(0));
    outputs[0].copy_shared_buffer(k_pool);
    outputs[1].copy_shared_buffer(v_pool);
    outputs[2].copy_shared_buffer(k_scale);
    outputs[3].copy_shared_buffer(v_scale);

    if (num_copies == 0u) {
      return;
    }

    array src_c = ensure_row_contiguous(src_slots, s, enc);
    array dst_c = ensure_row_contiguous(dst_slots, s, enc);

    const uint64_t pool_capacity = k_pool.shape(0) > 0
        ? static_cast<uint64_t>(k_pool.shape(0)) *
            static_cast<uint64_t>(block_size_)
        : 0ull;

    const float dummy = 0.0f;
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "paged_kv_path_commit", "inkling2_paged_kv_path_commit"));
    bind_inplace_pool_output(enc, outputs[0], 0);
    bind_inplace_pool_output(enc, outputs[1], 1);
    if (kv_mode_ == 0u) {
      bind_inplace_pool_output(enc, outputs[2], 2);
      bind_inplace_pool_output(enc, outputs[3], 3);
    } else {
      enc.set_bytes(dummy, 2);
      enc.set_bytes(dummy, 3);
    }
    enc.set_input_array(src_c, 4);
    enc.set_input_array(dst_c, 5);
    Inkling2PagedKvPathCommitParams params{
        num_copies,
        num_kv_heads_,
        head_dim_,
        block_size_,
        kv_mode_,
        0u,
        pool_capacity};
    enc.set_bytes(params, 6);
    // Keep BOTH pre- and post-dispatch barriers: `Inkling2PagedKvPathCommit`
    // reads `src_slots` and writes `dst_slots` in-place via `bind_inplace_pool_output`
    // (which only records `next_outputs_`, not `next_inputs_`). The pre-barrier
    // orders prior verify writes before our read; the post-barrier orders our
    // read/write before the next cycle's `KvPrepPaged` pool write (which only
    // checks `prev_inputs_` in `register_output_array`), preventing WAR/WAW races
    // inside MLX's concurrent Metal encoder.
    enc.barrier();
    enc.dispatch_threadgroups(MTL::Size(num_kv_heads_, 1, 1), MTL::Size(128, 1, 1));
    enc.barrier();
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2PagedKvPathCommit requires Metal");
#endif
  }

 private:
  uint32_t num_kv_heads_;
  uint32_t head_dim_;
  uint32_t block_size_;
  uint32_t kv_mode_;
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

    const char* kname = out_dtype_ == float32 ? "inkling2_block_attn_res_f32"
                                              : "inkling2_block_attn_res_bf16";
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "block_attn_res", kname));
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
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "pair_rms_norm", kname));
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
      bool has_tree,
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
        has_tree_(has_tree),
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
        has_tree_ == rhs.has_tree_ &&
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

    std::optional<array> tree_arr;
    if (has_tree_) {
      tree_arr = ensure_row_contiguous_aligned(inputs[idx++], 8u, s, enc);
    }

    auto& y_out = outputs[0];
    auto& hist_out = outputs[1];
    y_out.set_data(allocator::malloc(y_out.nbytes()));
    hist_out.set_data(allocator::malloc(hist_out.nbytes()));

    const float dummy = 0.0f;
    const int32_t dummy_cu = 0;
    const uint32_t dummy_tr[2] = {0u, 0u};
    const char* kname = out_dtype_ == float32
        ? "inkling2_sconv_k4_ragged_f32"
        : "inkling2_sconv_k4_ragged_bf16";
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "sconv_k4_ragged", kname));
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
        has_tree_ ? 1u : 0u};
    enc.set_bytes(params, 12);
    if (has_tree_) {
      enc.set_input_array(*tree_arr, 13);
    } else {
      enc.set_bytes(dummy_tr, 13);
    }
    const size_t tgx = (seq_len <= 16u) ? 64 : 256;
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
  bool has_tree_;
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
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "moe_route", "inkling2_moe_route_f32"));
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

    const uint32_t row_group = (rows > 8u && rows <= 16u) ? 16u : 8u;
    Inkling2MoeRouterProjParams params{
        rows,
        dim,
        num_experts_,
        x_stride,
        x.dtype() == float32 ? 1u : 0u,
        row_group,
        0u,
        0u};

    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "moe_router_proj", "inkling2_moe_router_proj_f32"));
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(params, 3);
    enc.dispatch_threadgroups(
        MTL::Size((num_experts_ + 1u) / 2u, (rows + row_group - 1u) / row_group, 1),
        MTL::Size(64, 1, 1));
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

    const uint32_t row_group = 8u;
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
        row_group,
        0u};

    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "loop_ctrl_down", "inkling2_loop_ctrl_down_f32"));
    enc.set_input_array(dest, 0);
    enc.set_input_array(msg, 1);
    enc.set_input_array(down_w, 2);
    enc.set_output_array(out, 3);
    enc.set_bytes(params, 4);
    enc.dispatch_threadgroups(
        MTL::Size(rank_, (rows + row_group - 1u) / row_group, num_streams),
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

    auto loop_scratch = inkling2_loop_ctrl_scratch(
        s, enc, 0u,
        static_cast<size_t>(num_streams) * rows * dim,
        static_cast<size_t>(num_streams) * rows * num_d_tiles);
    array& blended = loop_scratch.blended;
    array& partial_sq = loop_scratch.partial_sq;

    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const uint32_t flags =
        (dest.dtype() == float32 ? 1u : 0u) |
        ((msg.dtype() == float32 ? 1u : 0u) << 1u) |
        ((gf_w.dtype() == bfloat16 ? 1u : 0u) << 2u);

    const uint32_t row_group = 8u;
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
        row_group,
        0u};

    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device,
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
        MTL::Size(num_d_tiles, (rows + row_group - 1u) / row_group, num_streams),
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
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "rms_norm_from_sq", norm_kname));
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
// 16b. Inkling2LoopCtrlFused (end-to-end down + up_blend + rms_norm_from_sq
//      with persistent per-stream scratch and optional 3-way direct split!)
// ============================================================================
class Inkling2LoopCtrlFused : public fast::Custom {
 public:
  Inkling2LoopCtrlFused(
      Stream s,
      float attn_res_eps,
      float norm_eps,
      uint32_t rank,
      Dtype out_dtype,
      Shape out_shape,
      bool split_outputs)
      : Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2LoopCtrlFused requires Metal");
            }),
        attn_res_eps_(attn_res_eps),
        norm_eps_(norm_eps),
        rank_(rank),
        out_dtype_(out_dtype),
        out_shape_(std::move(out_shape)),
        split_outputs_(split_outputs) {}

  DEFINE_NAME(Inkling2LoopCtrlFused)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2LoopCtrlFused&>(other);
    return attn_res_eps_ == rhs.attn_res_eps_ &&
        norm_eps_ == rhs.norm_eps_ &&
        rank_ == rhs.rank_ &&
        out_dtype_ == rhs.out_dtype_ &&
        out_shape_ == rhs.out_shape_ &&
        split_outputs_ == rhs.split_outputs_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    if (split_outputs_) {
      Shape s2d{out_shape_[1], out_shape_[2]};
      return {s2d, s2d, s2d};
    }
    return {out_shape_};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2LoopCtrlFused requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    // Inputs: 0: dest, 1: msg, 2: down_w, 3: gain_forget_w, 4: norm_weight
    const uint32_t num_streams =
        out_shape_.size() == 3 ? static_cast<uint32_t>(out_shape_[0]) : 1u;
    const uint32_t rows = out_shape_.size() == 3
        ? static_cast<uint32_t>(out_shape_[1])
        : static_cast<uint32_t>(out_shape_[0]);
    const uint32_t dim = static_cast<uint32_t>(out_shape_.back());
    const uint32_t num_d_tiles = (dim + 31u) / 32u;

    // Prepare ALL inputs strictly BEFORE setting any compute pipeline state.
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

    const size_t down_align = inputs[2].dtype() == float32 ? 16u : 8u;
    array down_w = ensure_row_contiguous_aligned(inputs[2], down_align, s, enc);
    const size_t gf_align = inputs[3].dtype() == float32 ? 16u : 8u;
    array gf_w = ensure_row_contiguous_aligned(inputs[3], gf_align, s, enc);
    array weight = ensure_row_contiguous(inputs[4], s, enc);
    const uint32_t weight_ss =
        (weight.ndim() == 2 &&
         weight.shape(0) == static_cast<int>(num_streams) && num_streams > 1u)
        ? dim
        : 0u;

    auto scratch = inkling2_loop_ctrl_scratch(
        s, enc,
        static_cast<size_t>(num_streams) * rows * rank_,
        static_cast<size_t>(num_streams) * rows * dim,
        static_cast<size_t>(num_streams) * rows * num_d_tiles);
    array& h = scratch.h;
    array& blended = scratch.blended;
    array& partial_sq = scratch.partial_sq;

    for (auto& o : outputs) {
      o.set_data(allocator::malloc(o.nbytes()));
    }

    const uint32_t row_group = 8u;

    // 1. Pair RMSNorm + down_proj -> h [S, T, rank] (persistent scratch)
    const uint32_t down_flags =
        (dest.dtype() == float32 ? 1u : 0u) |
        ((msg.dtype() == float32 ? 1u : 0u) << 1u) |
        ((down_w.dtype() == bfloat16 ? 1u : 0u) << 2u);
    Inkling2LoopCtrlDownParams down_params{
        rows,
        dim,
        rank_,
        num_streams,
        attn_res_eps_,
        dest_rs,
        dest_ss,
        msg_rs,
        msg_ss,
        down_flags,
        row_group,
        0u};
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "loop_ctrl_down", "inkling2_loop_ctrl_down_f32"));
    enc.set_input_array(dest, 0);
    enc.set_input_array(msg, 1);
    enc.set_input_array(down_w, 2);
    enc.set_output_array(h, 3);
    enc.set_bytes(down_params, 4);
    enc.dispatch_threadgroups(
        MTL::Size(rank_, (rows + row_group - 1u) / row_group, num_streams),
        MTL::Size(128, 1, 1));

    // 2. up_proj + no-FMA blend + partial sum-of-squares (MLX auto-barriers on `h`)
    const uint32_t up_flags =
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
        up_flags,
        row_group,
        0u};
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "loop_ctrl_up_blend", "inkling2_loop_ctrl_up_blend_f32"));
    enc.set_input_array(h, 0);
    enc.set_input_array(gf_w, 1);
    enc.set_input_array(dest, 2);
    enc.set_input_array(msg, 3);
    enc.set_output_array(blended, 4);
    enc.set_output_array(partial_sq, 5);
    enc.set_bytes(up_params, 6);
    enc.dispatch_threadgroups(
        MTL::Size(num_d_tiles, (rows + row_group - 1u) / row_group, num_streams),
        MTL::Size(128, 1, 1));

    // 3. Sublayer RMSNorm -> single output OR 3 direct [T, D] outputs (MLX auto-barriers)
    Inkling2RmsNormFromSqParams norm_params{
        rows,
        dim,
        num_streams,
        num_d_tiles,
        norm_eps_,
        weight_ss,
        weight.dtype() == float32 ? 1u : 0u,
        0u};
    if (split_outputs_) {
      const char* norm_kname = out_dtype_ == float32
          ? "inkling2_rms_norm_from_sq_split3_f32"
          : "inkling2_rms_norm_from_sq_split3_bf16";
      enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
          device, "rms_norm_from_sq_split3", norm_kname));
      enc.set_input_array(blended, 0);
      enc.set_input_array(partial_sq, 1);
      enc.set_input_array(weight, 2);
      enc.set_output_array(outputs[0], 3);
      enc.set_output_array(outputs[1], 4);
      enc.set_output_array(outputs[2], 5);
      enc.set_bytes(norm_params, 6);
    } else {
      const char* norm_kname = out_dtype_ == float32
          ? "inkling2_rms_norm_from_sq_f32"
          : "inkling2_rms_norm_from_sq_bf16";
      enc.set_compute_pipeline_state(
          get_cached_inkling2_kernel(device, "rms_norm_from_sq", norm_kname));
      enc.set_input_array(blended, 0);
      enc.set_input_array(partial_sq, 1);
      enc.set_input_array(weight, 2);
      enc.set_output_array(outputs[0], 3);
      enc.set_bytes(norm_params, 4);
    }
    enc.dispatch_threadgroups(
        MTL::Size((dim + 127u) / 128u, rows, num_streams),
        MTL::Size(128, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2LoopCtrlFused requires Metal");
#endif
  }

 private:
  float attn_res_eps_;
  float norm_eps_;
  uint32_t rank_;
  Dtype out_dtype_;
  Shape out_shape_;
  bool split_outputs_;
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
    enc.set_compute_pipeline_state(
        get_cached_inkling2_kernel(device, "moe_combine", kname));
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
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "lm_head_softcap", "inkling2_lm_head_softcap_f32"));
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
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device,
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
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device,
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
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "markov_gate", "inkling2_markov_gate_bf16"));
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
      enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
          device, "markov_fold_softcap", "inkling2_markov_fold_softcap_f32"));
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
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device,
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
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device,
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
    encode_bf16_linear(device, enc, x, w, out, rows_, out_dim_, in_dim_);
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

    const bool use_nax = metal::is_nax_available() && rows_ > resolved_q8_simd_max_m();
    const uint32_t rows_per_tg = rows_ <= 16u ? 16u : 8u;
    const uint32_t num_splits = use_nax
        ? (out_cols_ + 31u) / 32u
        : std::min<uint32_t>(
              512u,
              std::max<uint32_t>(
                  1u, (out_cols_ + rows_per_tg - 1u) / rows_per_tg));
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
    const char* p1_name = use_nax
        ? "inkling2_bf16_lm_head_softcap_argmax_p1"
        : "inkling2_bf16_lm_head_softcap_argmax_simd_stage1";
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device,
        "bf16_lm_head_softcap_argmax_p1",
        p1_name));
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
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device,
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

// ============================================================================
// 16a. Inkling2LmHeadQ8SoftcapArgmax (`1 <= M <= 32`, `K % 32 == 0`, F16 scales, 0 biases)
// ============================================================================
class Inkling2LmHeadQ8SoftcapArgmax : public fast::Custom {
 public:
  Inkling2LmHeadQ8SoftcapArgmax(
      Stream s,
      uint32_t rows,
      uint32_t out_cols,
      uint32_t in_dim,
      float softcap)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2LmHeadQ8SoftcapArgmax requires Metal");
            }),
        rows_(rows),
        out_cols_(out_cols),
        in_dim_(in_dim),
        softcap_(softcap) {}

  DEFINE_NAME(Inkling2LmHeadQ8SoftcapArgmax)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2LmHeadQ8SoftcapArgmax&>(other);
    return rows_ == rhs.rows_ && out_cols_ == rhs.out_cols_ &&
        in_dim_ == rhs.in_dim_ && softcap_ == rhs.softcap_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {Shape{static_cast<int>(rows_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2LmHeadQ8SoftcapArgmax requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    array x = ensure_row_contiguous_aligned(inputs[0], 16u, s, enc);
    array w = ensure_row_contiguous_aligned(inputs[1], 16u, s, enc);
    array scales = ensure_row_contiguous(inputs[2], s, enc);

    const bool use_nax = metal::is_nax_available() && rows_ > resolved_q8_simd_max_m();
    const uint32_t num_splits = use_nax
        ? std::max<uint32_t>(1u, (out_cols_ + 63u) / 64u)
        : std::min<uint32_t>(
              512u,
              std::max<uint32_t>(
                  1u,
                  (out_cols_ + (rows_ <= 16u ? 15u : 7u)) /
                      (rows_ <= 16u ? 16u : 8u)));

    array partial_vals(
        Shape{static_cast<int>(rows_ * num_splits)}, float32, nullptr, {});
    partial_vals.set_data(allocator::malloc(partial_vals.nbytes()));
    enc.add_temporary(partial_vals);

    array partial_idxs(
        Shape{static_cast<int>(rows_ * num_splits)}, int32, nullptr, {});
    partial_idxs.set_data(allocator::malloc(partial_idxs.nbytes()));
    enc.add_temporary(partial_idxs);

    Inkling2Q8SymParams p1_params{
        rows_, out_cols_, in_dim_, num_splits, softcap_, 0u, 0u, 0u};
    const char* p1_name = use_nax
        ? "inkling2_lm_head_q8_nax_softcap_argmax_p1"
        : "inkling2_lm_head_q8_softcap_argmax_stage1";
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "lm_head_q8_softcap_argmax", p1_name));
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_input_array(scales, 2);
    enc.set_output_array(partial_vals, 3);
    enc.set_output_array(partial_idxs, 4);
    enc.set_bytes(p1_params, 5);
    enc.dispatch_threadgroups(MTL::Size(num_splits, 1, 1), MTL::Size(128, 1, 1));

    auto& out_ids = outputs[0];
    out_ids.set_data(allocator::malloc(out_ids.nbytes()));
    Inkling2LmHeadSoftcapParams stage2_params{
        rows_, out_cols_, 0u, 0u, softcap_, num_splits, 0u, 0u};
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device,
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
    throw std::runtime_error("Inkling2LmHeadQ8SoftcapArgmax requires Metal");
#endif
  }

 private:
  uint32_t rows_;
  uint32_t out_cols_;
  uint32_t in_dim_;
  float softcap_;
};

// ============================================================================
// 16b. Inkling2LmHeadQ8Bf16 (`1 <= M <= 32`, `K % 32 == 0`, BF16 OR FP32+softcap)
// ============================================================================
class Inkling2LmHeadQ8Bf16 : public fast::Custom {
 public:
  Inkling2LmHeadQ8Bf16(
      Stream s,
      uint32_t rows,
      uint32_t out_cols,
      uint32_t in_dim,
      float softcap,
      Dtype out_dtype)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2LmHeadQ8Bf16 requires Metal");
            }),
        rows_(rows),
        out_cols_(out_cols),
        in_dim_(in_dim),
        softcap_(softcap),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2LmHeadQ8Bf16)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2LmHeadQ8Bf16&>(other);
    return rows_ == rhs.rows_ && out_cols_ == rhs.out_cols_ &&
        in_dim_ == rhs.in_dim_ && softcap_ == rhs.softcap_ &&
        out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {Shape{static_cast<int>(rows_), static_cast<int>(out_cols_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2LmHeadQ8Bf16 requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    array x = ensure_row_contiguous_aligned(inputs[0], 16u, s, enc);
    array w = ensure_row_contiguous_aligned(inputs[1], 16u, s, enc);
    array scales = ensure_row_contiguous(inputs[2], s, enc);
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const uint32_t flags = (out_dtype_ == float32) ? 1u : 0u;
    Inkling2Q8SymParams params{
        rows_, out_cols_, in_dim_, 0u, softcap_, 0u, flags, 0u};

    if (metal::is_nax_available() && rows_ > resolved_q8_simd_max_m()) {
      const bool use_m8 = rows_ <= 8u;
      const uint32_t num_tgs = (out_cols_ + 31u) / 32u;
      const uint32_t dummy = 0u;
      const char* kname = use_m8 ? "inkling2_q8_sym_matmul_nax_m8_n32_bf16"
                                 : "inkling2_q8_sym_matmul_nax_bf16";
      enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
          device,
          "q8_sym_matmul_nax_bf16",
          kname));
      enc.set_input_array(x, 0);
      enc.set_input_array(w, 1);
      enc.set_input_array(scales, 2);
      enc.set_bytes(dummy, 3);
      enc.set_output_array(out, 4);
      enc.set_output_array(out, 5);
      enc.set_bytes(params, 6);
      enc.dispatch_threadgroups(MTL::Size(num_tgs, 1, 1), MTL::Size(128, 1, 1));
    } else {
      const uint32_t rows_per_tg = rows_ <= 16u ? 16u : 8u;
      const uint32_t num_tgs = (out_cols_ + rows_per_tg - 1u) / rows_per_tg;
      enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
          device, "lm_head_q8_bf16", "inkling2_lm_head_q8_bf16"));
      enc.set_input_array(x, 0);
      enc.set_input_array(w, 1);
      enc.set_input_array(scales, 2);
      enc.set_output_array(out, 3);
      enc.set_bytes(params, 4);
      enc.dispatch_threadgroups(MTL::Size(num_tgs, 1, 1), MTL::Size(128, 1, 1));
    }
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2LmHeadQ8Bf16 requires Metal");
#endif
  }

 private:
  uint32_t rows_;
  uint32_t out_cols_;
  uint32_t in_dim_;
  float softcap_;
  Dtype out_dtype_;
};

// ============================================================================
// 16c. Inkling2Q8SymMatmul (`1 <= M <= 32`, `K % 32 == 0`, optional fused BF16 tail!)
// ============================================================================
class Inkling2Q8SymMatmul : public fast::Custom {
 public:
  Inkling2Q8SymMatmul(
      Stream s,
      uint32_t rows,
      uint32_t out_dim,
      uint32_t in_dim,
      uint32_t aux_dim,
      Dtype out_dtype)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2Q8SymMatmul requires Metal");
            }),
        rows_(rows),
        out_dim_(out_dim),
        in_dim_(in_dim),
        aux_dim_(aux_dim),
        out_dtype_(out_dtype) {}

  DEFINE_NAME(Inkling2Q8SymMatmul)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2Q8SymMatmul&>(other);
    return rows_ == rhs.rows_ && out_dim_ == rhs.out_dim_ &&
        in_dim_ == rhs.in_dim_ && aux_dim_ == rhs.aux_dim_ &&
        out_dtype_ == rhs.out_dtype_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    if (aux_dim_ > 0u) {
      return {
          Shape{static_cast<int>(rows_), static_cast<int>(out_dim_)},
          Shape{static_cast<int>(rows_), static_cast<int>(aux_dim_)}};
    }
    return {Shape{static_cast<int>(rows_), static_cast<int>(out_dim_)}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2Q8SymMatmul requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    array x = ensure_row_contiguous_aligned(inputs[0], 16u, s, enc);
    array w = ensure_row_contiguous_aligned(inputs[1], 16u, s, enc);
    array scales = ensure_row_contiguous(inputs[2], s, enc);
    std::optional<array> w_aux;
    if (aux_dim_ > 0u) {
      w_aux = ensure_row_contiguous_aligned(inputs[3], 16u, s, enc);
    }

    auto& out_main = outputs[0];
    out_main.set_data(allocator::malloc(out_main.nbytes()));
    array* out_aux_ptr = nullptr;
    if (aux_dim_ > 0u) {
      outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));
      out_aux_ptr = &outputs[1];
    }
    encode_q8_sym_matmul(
        device, enc, x, w, scales,
        w_aux ? &(*w_aux) : nullptr,
        out_main, out_aux_ptr,
        rows_, out_dim_, in_dim_, aux_dim_, out_dtype_);
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2Q8SymMatmul requires Metal");
#endif
  }

 private:
  uint32_t rows_;
  uint32_t out_dim_;
  uint32_t in_dim_;
  uint32_t aux_dim_;
  Dtype out_dtype_;
};

// ============================================================================
// 16d. Inkling2OProjSconvResidual (fused o_proj [Q8_0 or BF16] + static k=4
//      ragged ShortConv + optional FP32 residual + optional rollback tape!)
// ============================================================================
class Inkling2OProjSconvResidual : public fast::Custom {
 public:
  Inkling2OProjSconvResidual(
      Stream s,
      uint32_t rows,
      uint32_t out_dim,
      uint32_t in_dim,
      bool is_q8,
      uint32_t num_seqs,
      bool packed_hist,
      bool has_res_a,
      bool has_tree,
      bool emit_tape)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2OProjSconvResidual requires Metal");
            }),
        rows_(rows),
        out_dim_(out_dim),
        in_dim_(in_dim),
        is_q8_(is_q8),
        num_seqs_(num_seqs),
        packed_hist_(packed_hist),
        has_res_a_(has_res_a),
        has_tree_(has_tree),
        emit_tape_(emit_tape) {}

  DEFINE_NAME(Inkling2OProjSconvResidual)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2OProjSconvResidual&>(other);
    return rows_ == rhs.rows_ && out_dim_ == rhs.out_dim_ &&
        in_dim_ == rhs.in_dim_ && is_q8_ == rhs.is_q8_ &&
        num_seqs_ == rhs.num_seqs_ && packed_hist_ == rhs.packed_hist_ &&
        has_res_a_ == rhs.has_res_a_ && has_tree_ == rhs.has_tree_ &&
        emit_tape_ == rhs.emit_tape_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    Shape y_s{static_cast<int>(rows_), static_cast<int>(out_dim_)};
    Shape h_s = (num_seqs_ == 1u && !packed_hist_)
        ? Shape{3, static_cast<int>(out_dim_)}
        : Shape{static_cast<int>(num_seqs_), 3, static_cast<int>(out_dim_)};
    if (emit_tape_) {
      return {y_s, h_s, y_s};
    }
    return {y_s, h_s};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2OProjSconvResidual requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
#ifdef MLX_NODE_METAL_ENABLED
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);

    // Input layout:
    //   0: y ([T, K] or [T, H, D] BF16)
    //   1: w_q8 ([C, K/4] u32) or w_bf16 ([C, K] bf16)
    //   [if is_q8_: 2: scales ([C, K/32] f16)]
    //   next: conv_w ([C, 4] bf16/f32)
    //   [optional: cu_seqlens if num_seqs_ > 1]
    //   [optional: res_a if has_res_a_]
    //   hists (1 if packed_hist_ else num_seqs_)
    //   [optional: tree_rows if has_tree_]
    const size_t x_align = (is_q8_ || (in_dim_ % 8u) == 0u) ? 16u : 8u;
    array x = ensure_row_contiguous_aligned(inputs[0], x_align, s, enc);
    array w = ensure_row_contiguous_aligned(inputs[1], x_align, s, enc);
    size_t idx = 2;
    std::optional<array> scales_opt;
    if (is_q8_) {
      scales_opt = ensure_row_contiguous(inputs[idx++], s, enc);
    }
    array conv_w = ensure_row_contiguous(inputs[idx++], s, enc);

    uint32_t flags = (conv_w.dtype() == float32 ? 1u : 0u);
    uint32_t res_a_stride = out_dim_;

    std::optional<array> cu_arr;
    if (num_seqs_ > 1u) {
      cu_arr = ensure_row_contiguous(inputs[idx++], s, enc);
    }

    std::optional<array> ra_arr;
    if (has_res_a_) {
      auto [ra, rs] = ensure_2d_strided(inputs[idx++], rows_, out_dim_, s, enc);
      res_a_stride = rs;
      flags |= (1u << 2u);
      if (ra.dtype() == float32) flags |= (1u << 3u);
      ra_arr = std::move(ra);
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

    std::optional<array> tree_arr;
    if (has_tree_) {
      tree_arr = ensure_row_contiguous_aligned(inputs[idx++], 8u, s, enc);
    }

    auto& y_out = outputs[0];
    auto& hist_out = outputs[1];
    y_out.set_data(allocator::malloc(y_out.nbytes()));
    hist_out.set_data(allocator::malloc(hist_out.nbytes()));

    // When `emit_tape_` is true, write `o_proj` directly into `outputs[2]` (zero
    // scratch and zero copy!). Otherwise reuse persistent per-stream BF16 scratch.
    std::optional<array> scratch_opt;
    if (emit_tape_) {
      outputs[2].set_data(allocator::malloc(outputs[2].nbytes()));
    } else {
      scratch_opt = inkling2_oproj_scratch(
          s, enc, static_cast<size_t>(rows_) * out_dim_);
    }
    array& proj_buf = emit_tape_ ? outputs[2] : *scratch_opt;

    // 1. Exact `o_proj` dispatch (Sym-Q8_0 or BF16) -> `proj_buf`
    if (is_q8_) {
      encode_q8_sym_matmul(
          device, enc, x, w, *scales_opt, nullptr, proj_buf, nullptr,
          rows_, out_dim_, in_dim_, 0u, bfloat16);
    } else {
      encode_bf16_linear(device, enc, x, w, proj_buf, rows_, out_dim_, in_dim_);
    }

    // 2. Exact static k=4 ragged ShortConv (+ optional FP32 residual).
    // MLX automatically emits the single buffer barrier on `proj_buf`!
    const float dummy = 0.0f;
    const int32_t dummy_cu = 0;
    const uint32_t dummy_tr[2] = {0u, 0u};
    enc.set_compute_pipeline_state(get_cached_inkling2_kernel(
        device, "sconv_k4_ragged", "inkling2_sconv_k4_ragged_bf16"));
    enc.set_input_array(proj_buf, 0);
    enc.set_input_array(conv_w, 1);
    enc.set_bytes(dummy, 2);
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
    enc.set_bytes(dummy, 5);
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
        rows_,
        out_dim_,
        num_seqs_,
        0u,
        0u,
        packed_hist_ ? 1u : 0u,
        out_dim_,
        res_a_stride,
        out_dim_,
        flags,
        hist_mask,
        has_tree_ ? 1u : 0u};
    enc.set_bytes(params, 12);
    if (has_tree_) {
      enc.set_input_array(*tree_arr, 13);
    } else {
      enc.set_bytes(dummy_tr, 13);
    }
    const size_t tgx = (rows_ <= 16u) ? 64 : 256;
    const size_t gx = (out_dim_ + tgx - 1) / tgx;
    enc.dispatch_threadgroups(MTL::Size(gx, rows_, 1), MTL::Size(tgx, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Inkling2OProjSconvResidual requires Metal");
#endif
  }

 private:
  uint32_t rows_;
  uint32_t out_dim_;
  uint32_t in_dim_;
  bool is_q8_;
  uint32_t num_seqs_;
  bool packed_hist_;
  bool has_res_a_;
  bool has_tree_;
  bool emit_tape_;
};

} // namespace

const std::vector<std::string>& kernel_names() {
  static const std::vector<std::string> names = {
      "inkling2_gate_sink_bf16",
      "inkling2_gate_sink_f32",
      "inkling2_situ_bf16",
      "inkling2_situ_f32",
      "inkling2_loop_blend_rms_norm_bf16",
      "inkling2_loop_blend_rms_norm_f32",
      "inkling2_sconv_k4_bf16",
      "inkling2_sconv_k4_f32",
      "inkling2_qr_clip_bf16",
      "inkling2_qr_clip_f32",
      "inkling2_sconv_k4_kv_prep_bf16",
      "inkling2_sconv_k4_kv_prep_f32",
      "inkling2_sconv_k4_kv_prep_paged_bf16",
      "inkling2_qr_clip_sconv_k4_kv_prep_paged_bf16",
      "inkling2_paged_kv_path_commit",
      "inkling2_block_attn_res_bf16",
      "inkling2_block_attn_res_f32",
      "inkling2_pair_rms_norm_f32",
      "inkling2_pair_rms_norm_bf16",
      "inkling2_sconv_k4_ragged_bf16",
      "inkling2_sconv_k4_ragged_f32",
      "inkling2_moe_router_proj_f32",
      "inkling2_loop_ctrl_down_f32",
      "inkling2_loop_ctrl_up_blend_f32",
      "inkling2_rms_norm_from_sq_bf16",
      "inkling2_rms_norm_from_sq_f32",
      "inkling2_rms_norm_from_sq_split3_bf16",
      "inkling2_rms_norm_from_sq_split3_f32",
      "inkling2_moe_route_f32",
      "inkling2_moe_combine_bf16",
      "inkling2_moe_combine_f32",
      "inkling2_lm_head_softcap_f32",
      "inkling2_lm_head_softcap_argmax_stage1",
      "inkling2_lm_head_softcap_argmax_stage2",
      "inkling2_markov_gate_bf16",
      "inkling2_markov_fold_softcap_f32",
      "inkling2_markov_fold_softcap_argmax_stage1",
      "inkling2_bf16_linear_m1",
      "inkling2_bf16_linear_nax_m8",
      "inkling2_bf16_linear_batched",
      "inkling2_bf16_linear_simd",
      "inkling2_bf16_lm_head_softcap_argmax_p1",
      "inkling2_bf16_lm_head_softcap_argmax_simd_stage1",
      "inkling2_lm_head_q8_softcap_argmax_stage1",
      "inkling2_lm_head_q8_bf16",
      "inkling2_q8_sym_matmul_bf16",
      "inkling2_q8_sym_matmul_nax_m8_bf16",
      "inkling2_q8_sym_matmul_nax_m8_n32_bf16",
      "inkling2_q8_sym_matmul_nax_bf16",
      "inkling2_lm_head_q8_nax_softcap_argmax_p1",
  };
  return names;
}

void clear_inkling2_ops_scratch() {
#ifdef MLX_NODE_METAL_ENABLED
  inkling2_oproj_scratch_pool().clear();
  inkling2_loop_ctrl_scratch_pool().clear();
#endif
}

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
    mlx_array* tree_rows_ptr,
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
    const bool has_tree = tree_rows_ptr != nullptr;
    if (has_tree) {
      const auto& tr = *reinterpret_cast<array*>(tree_rows_ptr);
      if (!validate_tree_rows_array(tr, x.shape(0))) {
        return false;
      }
      inputs.push_back(tr);
    }
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2ShortConvK4>(
        s, is_dynamic, num_kv_heads, has_tree, x.dtype());
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
  (void)tree_rows_ptr;
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
    mlx_array* tree_rows_ptr,
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
    const bool has_tree = tree_rows_ptr != nullptr;
    if (has_tree) {
      const auto& tr = *reinterpret_cast<array*>(tree_rows_ptr);
      if (!validate_tree_rows_array(tr, k_in.shape(0))) {
        return false;
      }
      inputs.push_back(tr);
    }
    const Dtype dt = k_in.dtype();
    const int t = k_in.shape(0);
    Shape kv_s{t, static_cast<int>(num_kv_heads), static_cast<int>(head_dim)};
    Shape hs = (num_seqs == 1u && !packed_hist)
        ? Shape{3, kv_dim}
        : Shape{static_cast<int>(num_seqs), 3, kv_dim};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2ShortConvK4KvPrep>(
        s, num_kv_heads, head_dim, num_seqs, is_dynamic, packed_hist, has_tree,
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
  (void)qk_thresh; (void)v_eps; (void)l2_eps; (void)tree_rows_ptr;
  return false;
#endif
}

bool mlx_inkling2_sconv_k4_kv_prep_paged(
    mlx_array* k_in_ptr,
    mlx_array* v_in_ptr,
    mlx_array* k_weight_ptr,
    mlx_array* v_weight_ptr,
    mlx_array* dyn_k_ptr,
    mlx_array* dyn_v_ptr,
    mlx_array* cu_seqlens_ptr,
    mlx_array* const* k_hists_ptr,
    mlx_array* const* v_hists_ptr,
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    mlx_array* slot_mapping_ptr,
    uint32_t num_seqs,
    bool packed_hist,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t block_size,
    uint32_t kv_mode,
    float qk_thresh,
    float v_eps,
    float l2_eps,
    mlx_array* tree_rows_ptr,
    mlx_array** out_k_pool,
    mlx_array** out_v_pool,
    mlx_array** out_k_scale,
    mlx_array** out_v_scale,
    mlx_array** out_k_hist,
    mlx_array** out_v_hist) {
  if (out_k_pool) *out_k_pool = nullptr;
  if (out_v_pool) *out_v_pool = nullptr;
  if (out_k_scale) *out_k_scale = nullptr;
  if (out_v_scale) *out_v_scale = nullptr;
  if (out_k_hist) *out_k_hist = nullptr;
  if (out_v_hist) *out_v_hist = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!k_in_ptr || !v_in_ptr || !k_weight_ptr || !v_weight_ptr ||
      !k_hists_ptr || !v_hists_ptr || !k_pool_ptr || !v_pool_ptr ||
      !k_scale_ptr || !v_scale_ptr || !slot_mapping_ptr ||
      !out_k_pool || !out_v_pool || !out_k_scale || !out_v_scale ||
      !out_k_hist || !out_v_hist ||
      num_seqs == 0 || num_kv_heads == 0 || head_dim == 0 || head_dim > 256u ||
      block_size == 0 || kv_mode > 2u) {
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
    const auto& k_pool = *reinterpret_cast<array*>(k_pool_ptr);
    const auto& v_pool = *reinterpret_cast<array*>(v_pool_ptr);
    const auto& k_scale = *reinterpret_cast<array*>(k_scale_ptr);
    const auto& v_scale = *reinterpret_cast<array*>(v_scale_ptr);
    const auto& slot_mapping = *reinterpret_cast<array*>(slot_mapping_ptr);
    const int kv_dim = static_cast<int>(num_kv_heads * head_dim);
    if (k_in.ndim() != 2 || v_in.ndim() != 2 ||
        k_in.shape(0) <= 0 || k_in.shape(0) != v_in.shape(0) ||
        k_in.shape(1) != kv_dim || v_in.shape(1) != kv_dim ||
        k_in.dtype() != bfloat16 || v_in.dtype() != bfloat16 ||
        kw.ndim() != 2 || kw.shape(0) != kv_dim || kw.shape(1) != 4 || !is_bf16_or_f32(kw.dtype()) ||
        vw.ndim() != 2 || vw.shape(0) != kv_dim || vw.shape(1) != 4 || !is_bf16_or_f32(vw.dtype()) ||
        slot_mapping.ndim() != 1 || slot_mapping.shape(0) != k_in.shape(0) ||
        slot_mapping.dtype() != int64) {
      return false;
    }
    if (!k_pool.flags().row_contiguous || !v_pool.flags().row_contiguous ||
        k_pool.offset() != 0 || v_pool.offset() != 0 ||
        k_pool.shape(0) <= 0 || k_pool.shape(0) != v_pool.shape(0)) {
      return false;
    }
    const int nb = k_pool.shape(0);
    const int nkv = static_cast<int>(num_kv_heads);
    const int hd = static_cast<int>(head_dim);
    const int bs = static_cast<int>(block_size);
    if (kv_mode == 0u) {
      // INT8-KTM
      if (k_pool.dtype() != int8 || v_pool.dtype() != int8 ||
          k_pool.shape() != Shape{nb, nkv, bs, hd} ||
          v_pool.shape() != Shape{nb, nkv, hd, bs} ||
          k_scale.dtype() != float32 || v_scale.dtype() != float32 ||
          !k_scale.flags().row_contiguous || !v_scale.flags().row_contiguous ||
          k_scale.offset() != 0 || v_scale.offset() != 0 ||
          k_scale.shape() != Shape{nb, nkv, bs} ||
          v_scale.shape() != Shape{nb, nkv, bs}) {
        return false;
      }
    } else if (kv_mode == 1u) {
      // BF16 vLLM (x=8)
      if (hd % 8 != 0 ||
          k_pool.dtype() != bfloat16 || v_pool.dtype() != bfloat16 ||
          k_pool.shape() != Shape{nb, nkv, hd / 8, bs, 8} ||
          v_pool.shape() != Shape{nb, nkv, hd, bs}) {
        return false;
      }
    } else {
      // BF16 values_token_major
      if (k_pool.dtype() != bfloat16 || v_pool.dtype() != bfloat16 ||
          k_pool.shape() != Shape{nb, nkv, bs, hd} ||
          v_pool.shape() != Shape{nb, nkv, bs, hd}) {
        return false;
      }
    }
    std::vector<array> inputs = {
        k_in, v_in, kw, vw, k_pool, v_pool, k_scale, v_scale, slot_mapping};
    if (is_dynamic) {
      const auto& dk = *reinterpret_cast<array*>(dyn_k_ptr);
      const auto& dv = *reinterpret_cast<array*>(dyn_v_ptr);
      if (dk.ndim() != 3 || dv.ndim() != 3 ||
          dk.shape(0) != k_in.shape(0) || dv.shape(0) != k_in.shape(0) ||
          dk.shape(1) != nkv || dv.shape(1) != nkv ||
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
    const bool has_tree = tree_rows_ptr != nullptr;
    if (has_tree) {
      const auto& tr = *reinterpret_cast<array*>(tree_rows_ptr);
      if (!validate_tree_rows_array(tr, k_in.shape(0))) {
        return false;
      }
      inputs.push_back(tr);
    }
    Shape hs = (num_seqs == 1u && !packed_hist)
        ? Shape{3, kv_dim}
        : Shape{static_cast<int>(num_seqs), 3, kv_dim};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2ShortConvK4KvPrepPaged>(
        s, num_kv_heads, head_dim, num_seqs, block_size, kv_mode,
        is_dynamic, packed_hist, has_tree, qk_thresh, v_eps, l2_eps);
    auto results = array::make_arrays(
        {k_pool.shape(), v_pool.shape(), k_scale.shape(), v_scale.shape(), hs, hs},
        {k_pool.dtype(), v_pool.dtype(), k_scale.dtype(), v_scale.dtype(), bfloat16, bfloat16},
        std::move(prim),
        std::move(inputs));
    *out_k_pool = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_v_pool = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    *out_k_scale = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
    *out_v_scale = reinterpret_cast<mlx_array*>(new array(std::move(results[3])));
    *out_k_hist = reinterpret_cast<mlx_array*>(new array(std::move(results[4])));
    *out_v_hist = reinterpret_cast<mlx_array*>(new array(std::move(results[5])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_sconv_k4_kv_prep_paged", e.what());
    return false;
  }
#else
  (void)k_in_ptr; (void)v_in_ptr; (void)k_weight_ptr; (void)v_weight_ptr;
  (void)dyn_k_ptr; (void)dyn_v_ptr; (void)cu_seqlens_ptr; (void)k_hists_ptr; (void)v_hists_ptr;
  (void)k_pool_ptr; (void)v_pool_ptr; (void)k_scale_ptr; (void)v_scale_ptr; (void)slot_mapping_ptr;
  (void)num_seqs; (void)packed_hist; (void)num_kv_heads; (void)head_dim;
  (void)block_size; (void)kv_mode; (void)qk_thresh; (void)v_eps; (void)l2_eps; (void)tree_rows_ptr;
  return false;
#endif
}

bool mlx_inkling2_qr_clip_sconv_k4_kv_prep_paged(
    mlx_array* q_in_ptr,
    mlx_array* r_in_ptr,
    mlx_array* q_clip_ptr,
    mlx_array* rel_clip_ptr,
    mlx_array* tau_ptr,
    mlx_array* k_in_ptr,
    mlx_array* v_in_ptr,
    mlx_array* k_weight_ptr,
    mlx_array* v_weight_ptr,
    mlx_array* dyn_k_ptr,
    mlx_array* dyn_v_ptr,
    mlx_array* cu_seqlens_ptr,
    mlx_array* const* k_hists_ptr,
    mlx_array* const* v_hists_ptr,
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    mlx_array* slot_mapping_ptr,
    uint32_t num_seqs,
    bool packed_hist,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t d_rel,
    uint32_t block_size,
    uint32_t kv_mode,
    float qk_thresh,
    float rel_thresh,
    float v_eps,
    float l2_eps,
    bool r_round_bf16,
    mlx_array* tree_rows_ptr,
    mlx_array** out_q,
    mlx_array** out_r,
    mlx_array** out_k_pool,
    mlx_array** out_v_pool,
    mlx_array** out_k_scale,
    mlx_array** out_v_scale,
    mlx_array** out_k_hist,
    mlx_array** out_v_hist) {
  if (out_q) *out_q = nullptr;
  if (out_r) *out_r = nullptr;
  if (out_k_pool) *out_k_pool = nullptr;
  if (out_v_pool) *out_v_pool = nullptr;
  if (out_k_scale) *out_k_scale = nullptr;
  if (out_v_scale) *out_v_scale = nullptr;
  if (out_k_hist) *out_k_hist = nullptr;
  if (out_v_hist) *out_v_hist = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!q_in_ptr || !r_in_ptr || !q_clip_ptr || !rel_clip_ptr ||
      !k_in_ptr || !v_in_ptr || !k_weight_ptr || !v_weight_ptr ||
      !k_hists_ptr || !v_hists_ptr || !k_pool_ptr || !v_pool_ptr ||
      !k_scale_ptr || !v_scale_ptr || !slot_mapping_ptr ||
      !out_q || !out_r || !out_k_pool || !out_v_pool || !out_k_scale || !out_v_scale ||
      !out_k_hist || !out_v_hist ||
      num_seqs == 0 || num_q_heads == 0 || num_kv_heads == 0 ||
      head_dim == 0 || head_dim > 256u || d_rel == 0 ||
      block_size == 0 || kv_mode > 2u) {
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
    const auto& q_in = *reinterpret_cast<array*>(q_in_ptr);
    const auto& r_in = *reinterpret_cast<array*>(r_in_ptr);
    const auto& q_clip = *reinterpret_cast<array*>(q_clip_ptr);
    const auto& rel_clip = *reinterpret_cast<array*>(rel_clip_ptr);
    const auto& k_in = *reinterpret_cast<array*>(k_in_ptr);
    const auto& v_in = *reinterpret_cast<array*>(v_in_ptr);
    const auto& kw = *reinterpret_cast<array*>(k_weight_ptr);
    const auto& vw = *reinterpret_cast<array*>(v_weight_ptr);
    const auto& k_pool = *reinterpret_cast<array*>(k_pool_ptr);
    const auto& v_pool = *reinterpret_cast<array*>(v_pool_ptr);
    const auto& k_scale = *reinterpret_cast<array*>(k_scale_ptr);
    const auto& v_scale = *reinterpret_cast<array*>(v_scale_ptr);
    const auto& slot_mapping = *reinterpret_cast<array*>(slot_mapping_ptr);
    const int kv_dim = static_cast<int>(num_kv_heads * head_dim);
    if (q_in.ndim() != 2 || r_in.ndim() != 2 ||
        q_in.shape(0) <= 0 || r_in.shape(0) != q_in.shape(0) ||
        static_cast<uint32_t>(q_in.shape(1)) != num_q_heads * head_dim ||
        static_cast<uint32_t>(r_in.shape(1)) != num_q_heads * d_rel ||
        !is_bf16_or_f32(q_in.dtype()) || !is_bf16_or_f32(r_in.dtype()) ||
        !is_bf16_or_f32(q_clip.dtype()) || !is_bf16_or_f32(rel_clip.dtype()) ||
        static_cast<uint64_t>(q_clip.size()) != static_cast<uint64_t>(num_q_heads) * head_dim ||
        static_cast<uint64_t>(rel_clip.size()) != static_cast<uint64_t>(num_q_heads) * d_rel ||
        k_in.ndim() != 2 || v_in.ndim() != 2 ||
        k_in.shape(0) != q_in.shape(0) || v_in.shape(0) != q_in.shape(0) ||
        k_in.shape(1) != kv_dim || v_in.shape(1) != kv_dim ||
        k_in.dtype() != bfloat16 || v_in.dtype() != bfloat16 ||
        kw.ndim() != 2 || kw.shape(0) != kv_dim || kw.shape(1) != 4 || !is_bf16_or_f32(kw.dtype()) ||
        vw.ndim() != 2 || vw.shape(0) != kv_dim || vw.shape(1) != 4 || !is_bf16_or_f32(vw.dtype()) ||
        slot_mapping.ndim() != 1 || slot_mapping.shape(0) != q_in.shape(0) ||
        slot_mapping.dtype() != int64) {
      return false;
    }
    if (!k_pool.flags().row_contiguous || !v_pool.flags().row_contiguous ||
        k_pool.offset() != 0 || v_pool.offset() != 0 ||
        k_pool.shape(0) <= 0 || k_pool.shape(0) != v_pool.shape(0)) {
      return false;
    }
    const int nb = k_pool.shape(0);
    const int nkv = static_cast<int>(num_kv_heads);
    const int hd = static_cast<int>(head_dim);
    const int bs = static_cast<int>(block_size);
    if (kv_mode == 0u) {
      if (k_pool.dtype() != int8 || v_pool.dtype() != int8 ||
          k_pool.shape() != Shape{nb, nkv, bs, hd} ||
          v_pool.shape() != Shape{nb, nkv, hd, bs} ||
          k_scale.dtype() != float32 || v_scale.dtype() != float32 ||
          !k_scale.flags().row_contiguous || !v_scale.flags().row_contiguous ||
          k_scale.offset() != 0 || v_scale.offset() != 0 ||
          k_scale.shape() != Shape{nb, nkv, bs} ||
          v_scale.shape() != Shape{nb, nkv, bs}) {
        return false;
      }
    } else if (kv_mode == 1u) {
      if (hd % 8 != 0 ||
          k_pool.dtype() != bfloat16 || v_pool.dtype() != bfloat16 ||
          k_pool.shape() != Shape{nb, nkv, hd / 8, bs, 8} ||
          v_pool.shape() != Shape{nb, nkv, hd, bs}) {
        return false;
      }
    } else {
      if (k_pool.dtype() != bfloat16 || v_pool.dtype() != bfloat16 ||
          k_pool.shape() != Shape{nb, nkv, bs, hd} ||
          v_pool.shape() != Shape{nb, nkv, bs, hd}) {
        return false;
      }
    }
    std::vector<array> inputs = {
        q_in, r_in, q_clip, rel_clip,
        k_in, v_in, kw, vw,
        k_pool, v_pool, k_scale, v_scale, slot_mapping};
    const bool has_tau = tau_ptr != nullptr;
    if (has_tau) {
      const auto& tau = *reinterpret_cast<array*>(tau_ptr);
      if (!is_bf16_or_f32(tau.dtype()) || static_cast<int>(tau.size()) != q_in.shape(0)) {
        return false;
      }
      inputs.push_back(tau);
    }
    if (is_dynamic) {
      const auto& dk = *reinterpret_cast<array*>(dyn_k_ptr);
      const auto& dv = *reinterpret_cast<array*>(dyn_v_ptr);
      if (dk.ndim() != 3 || dv.ndim() != 3 ||
          dk.shape(0) != q_in.shape(0) || dv.shape(0) != q_in.shape(0) ||
          dk.shape(1) != nkv || dv.shape(1) != nkv ||
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
    const bool has_tree = tree_rows_ptr != nullptr;
    if (has_tree) {
      const auto& tr = *reinterpret_cast<array*>(tree_rows_ptr);
      if (!validate_tree_rows_array(tr, q_in.shape(0))) {
        return false;
      }
      inputs.push_back(tr);
    }
    const int t = q_in.shape(0);
    Shape hs = (num_seqs == 1u && !packed_hist)
        ? Shape{3, kv_dim}
        : Shape{static_cast<int>(num_seqs), 3, kv_dim};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2QrClipSconvK4KvPrepPaged>(
        s, num_q_heads, num_kv_heads, head_dim, d_rel, num_seqs, block_size,
        kv_mode, has_tau, r_round_bf16, is_dynamic, packed_hist, has_tree,
        qk_thresh, rel_thresh, v_eps, l2_eps);
    auto results = array::make_arrays(
        {Shape{t, static_cast<int>(num_q_heads), static_cast<int>(head_dim)},
         Shape{t, static_cast<int>(num_q_heads), static_cast<int>(d_rel)},
         k_pool.shape(), v_pool.shape(), k_scale.shape(), v_scale.shape(),
         hs, hs},
        {bfloat16, float32,
         k_pool.dtype(), v_pool.dtype(), k_scale.dtype(), v_scale.dtype(),
         bfloat16, bfloat16},
        std::move(prim),
        std::move(inputs));
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_r = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    *out_k_pool = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
    *out_v_pool = reinterpret_cast<mlx_array*>(new array(std::move(results[3])));
    *out_k_scale = reinterpret_cast<mlx_array*>(new array(std::move(results[4])));
    *out_v_scale = reinterpret_cast<mlx_array*>(new array(std::move(results[5])));
    *out_k_hist = reinterpret_cast<mlx_array*>(new array(std::move(results[6])));
    *out_v_hist = reinterpret_cast<mlx_array*>(new array(std::move(results[7])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_qr_clip_sconv_k4_kv_prep_paged", e.what());
    return false;
  }
#else
  (void)q_in_ptr; (void)r_in_ptr; (void)q_clip_ptr; (void)rel_clip_ptr; (void)tau_ptr;
  (void)k_in_ptr; (void)v_in_ptr; (void)k_weight_ptr; (void)v_weight_ptr;
  (void)dyn_k_ptr; (void)dyn_v_ptr; (void)cu_seqlens_ptr; (void)k_hists_ptr; (void)v_hists_ptr;
  (void)k_pool_ptr; (void)v_pool_ptr; (void)k_scale_ptr; (void)v_scale_ptr; (void)slot_mapping_ptr;
  (void)num_seqs; (void)packed_hist; (void)num_q_heads; (void)num_kv_heads; (void)head_dim; (void)d_rel;
  (void)block_size; (void)kv_mode; (void)qk_thresh; (void)rel_thresh; (void)v_eps; (void)l2_eps;
  (void)r_round_bf16; (void)tree_rows_ptr;
  return false;
#endif
}

bool mlx_inkling2_paged_kv_path_commit(
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    mlx_array* src_slots_ptr,
    mlx_array* dst_slots_ptr,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t block_size,
    uint32_t kv_mode,
    mlx_array** out_k_pool,
    mlx_array** out_v_pool,
    mlx_array** out_k_scale,
    mlx_array** out_v_scale) {
  if (out_k_pool) *out_k_pool = nullptr;
  if (out_v_pool) *out_v_pool = nullptr;
  if (out_k_scale) *out_k_scale = nullptr;
  if (out_v_scale) *out_v_scale = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!k_pool_ptr || !v_pool_ptr || !k_scale_ptr || !v_scale_ptr ||
      !src_slots_ptr || !dst_slots_ptr ||
      !out_k_pool || !out_v_pool || !out_k_scale || !out_v_scale ||
      num_kv_heads == 0 || head_dim == 0 || block_size == 0 || kv_mode > 2u) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return false;
    }
    const auto& k_pool = *reinterpret_cast<array*>(k_pool_ptr);
    const auto& v_pool = *reinterpret_cast<array*>(v_pool_ptr);
    const auto& k_scale = *reinterpret_cast<array*>(k_scale_ptr);
    const auto& v_scale = *reinterpret_cast<array*>(v_scale_ptr);
    const auto& src_slots = *reinterpret_cast<array*>(src_slots_ptr);
    const auto& dst_slots = *reinterpret_cast<array*>(dst_slots_ptr);
    if (!k_pool.flags().row_contiguous || !v_pool.flags().row_contiguous ||
        k_pool.offset() != 0 || v_pool.offset() != 0 ||
        k_pool.shape(0) <= 0 || k_pool.shape(0) != v_pool.shape(0) ||
        src_slots.ndim() != 1 || dst_slots.ndim() != 1 ||
        src_slots.dtype() != int64 || dst_slots.dtype() != int64 ||
        src_slots.shape(0) != dst_slots.shape(0)) {
      return false;
    }
    const int nb = k_pool.shape(0);
    const int nkv = static_cast<int>(num_kv_heads);
    const int hd = static_cast<int>(head_dim);
    const int bs = static_cast<int>(block_size);
    if (kv_mode == 0u) {
      if (k_pool.dtype() != int8 || v_pool.dtype() != int8 ||
          k_pool.shape() != Shape{nb, nkv, bs, hd} ||
          v_pool.shape() != Shape{nb, nkv, hd, bs} ||
          k_scale.dtype() != float32 || v_scale.dtype() != float32 ||
          !k_scale.flags().row_contiguous || !v_scale.flags().row_contiguous ||
          k_scale.offset() != 0 || v_scale.offset() != 0 ||
          k_scale.shape() != Shape{nb, nkv, bs} ||
          v_scale.shape() != Shape{nb, nkv, bs}) {
        return false;
      }
    } else if (kv_mode == 1u) {
      if (hd % 8 != 0 ||
          k_pool.dtype() != bfloat16 || v_pool.dtype() != bfloat16 ||
          k_pool.shape() != Shape{nb, nkv, hd / 8, bs, 8} ||
          v_pool.shape() != Shape{nb, nkv, hd, bs}) {
        return false;
      }
    } else {
      if (k_pool.dtype() != bfloat16 || v_pool.dtype() != bfloat16 ||
          k_pool.shape() != Shape{nb, nkv, bs, hd} ||
          v_pool.shape() != Shape{nb, nkv, bs, hd}) {
        return false;
      }
    }
    const int64_t pool_capacity =
        static_cast<int64_t>(k_pool.shape(0)) * static_cast<int64_t>(block_size);
    if (src_slots.shape(0) > 0 && src_slots.is_available() &&
        dst_slots.is_available() && src_slots.data_shared_ptr() &&
        dst_slots.data_shared_ptr()) {
      const int64_t s_stride = src_slots.strides(0);
      const int64_t d_stride = dst_slots.strides(0);
      const int64_t* s_data = src_slots.data<int64_t>();
      const int64_t* d_data = dst_slots.data<int64_t>();
      for (int i = 0; i < src_slots.shape(0); ++i) {
        const int64_t sv = s_data[static_cast<int64_t>(i) * s_stride];
        const int64_t dv = d_data[static_cast<int64_t>(i) * d_stride];
        if (sv < 0 || dv < 0 || sv >= pool_capacity || dv >= pool_capacity) {
          return false;
        }
      }
    }
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2PagedKvPathCommit>(
        s, num_kv_heads, head_dim, block_size, kv_mode);
    auto results = array::make_arrays(
        {k_pool.shape(), v_pool.shape(), k_scale.shape(), v_scale.shape()},
        {k_pool.dtype(), v_pool.dtype(), k_scale.dtype(), v_scale.dtype()},
        std::move(prim),
        {k_pool, v_pool, k_scale, v_scale, src_slots, dst_slots});
    *out_k_pool = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_v_pool = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    *out_k_scale = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
    *out_v_scale = reinterpret_cast<mlx_array*>(new array(std::move(results[3])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_paged_kv_path_commit", e.what());
    return false;
  }
#else
  (void)k_pool_ptr; (void)v_pool_ptr; (void)k_scale_ptr; (void)v_scale_ptr;
  (void)src_slots_ptr; (void)dst_slots_ptr;
  (void)num_kv_heads; (void)head_dim; (void)block_size; (void)kv_mode;
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
    mlx_array* tree_rows_ptr,
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
    const bool has_tree = tree_rows_ptr != nullptr;
    if (has_tree) {
      const auto& tr = *reinterpret_cast<array*>(tree_rows_ptr);
      if (!validate_tree_rows_array(tr, t)) {
        return false;
      }
      inputs.push_back(tr);
    }
    Shape hs = (num_seqs == 1u && !packed_hist)
        ? Shape{3, c}
        : Shape{static_cast<int>(num_seqs), 3, c};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2ShortConvK4Ragged>(
        s, num_seqs, is_dynamic, num_kv_heads, packed_hist, has_res_a, has_res_b, has_tree, x.dtype());
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
  (void)res_a_ptr; (void)res_b_ptr; (void)hists_ptr; (void)num_seqs; (void)packed_hist; (void)tree_rows_ptr;
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
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(w_ptr);
    const int max_rows = metal::is_nax_available() ? 1024 : 32;
    if (x.ndim() != 2 || w.ndim() != 2 ||
        x.dtype() != bfloat16 || w.dtype() != bfloat16 ||
        x.shape(0) <= 0 || x.shape(0) > max_rows ||
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
    if (!metal::is_available()) {
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

mlx_array* mlx_inkling2_lm_head_q8_softcap_argmax(
    mlx_array* x_ptr,
    mlx_array* w_q8_ptr,
    mlx_array* scales_f16_ptr,
    uint32_t out_cols,
    float softcap) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !w_q8_ptr || !scales_f16_ptr || out_cols == 0u) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(w_q8_ptr);
    const auto& sc = *reinterpret_cast<array*>(scales_f16_ptr);
    if (x.ndim() != 2 || w.ndim() != 2 || sc.ndim() != 2 ||
        x.dtype() != bfloat16 || w.dtype() != uint32 || sc.dtype() != float16 ||
        x.shape(0) <= 0 || x.shape(0) > 32 ||
        x.shape(1) <= 0 || (x.shape(1) % 32) != 0 ||
        static_cast<uint32_t>(w.shape(0)) < out_cols ||
        w.shape(1) != x.shape(1) / 4 ||
        sc.shape(0) != w.shape(0) || sc.shape(1) != x.shape(1) / 32) {
      return nullptr;
    }
    const uint32_t rows = static_cast<uint32_t>(x.shape(0));
    const uint32_t in_dim = static_cast<uint32_t>(x.shape(1));
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2LmHeadQ8SoftcapArgmax>(
        s, rows, out_cols, in_dim, softcap);
    auto res = array(
        Shape{static_cast<int>(rows)},
        int32,
        std::move(prim),
        {x, w, sc});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_lm_head_q8_softcap_argmax", e.what());
    return nullptr;
  }
#else
  (void)x_ptr; (void)w_q8_ptr; (void)scales_f16_ptr; (void)out_cols; (void)softcap;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_lm_head_q8_bf16(
    mlx_array* x_ptr,
    mlx_array* w_q8_ptr,
    mlx_array* scales_f16_ptr,
    uint32_t out_cols,
    float softcap,
    bool output_f32) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !w_q8_ptr || !scales_f16_ptr || out_cols == 0u) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(w_q8_ptr);
    const auto& sc = *reinterpret_cast<array*>(scales_f16_ptr);
    if (x.ndim() != 2 || w.ndim() != 2 || sc.ndim() != 2 ||
        x.dtype() != bfloat16 || w.dtype() != uint32 || sc.dtype() != float16 ||
        x.shape(0) <= 0 || x.shape(0) > 32 ||
        x.shape(1) <= 0 || (x.shape(1) % 32) != 0 ||
        static_cast<uint32_t>(w.shape(0)) < out_cols ||
        w.shape(1) != x.shape(1) / 4 ||
        sc.shape(0) != w.shape(0) || sc.shape(1) != x.shape(1) / 32) {
      return nullptr;
    }
    const uint32_t rows = static_cast<uint32_t>(x.shape(0));
    const uint32_t in_dim = static_cast<uint32_t>(x.shape(1));
    const Dtype out_dtype = output_f32 ? float32 : bfloat16;
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2LmHeadQ8Bf16>(
        s, rows, out_cols, in_dim, softcap, out_dtype);
    auto res = array(
        Shape{static_cast<int>(rows), static_cast<int>(out_cols)},
        out_dtype,
        std::move(prim),
        {x, w, sc});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_lm_head_q8_bf16", e.what());
    return nullptr;
  }
#else
  (void)x_ptr; (void)w_q8_ptr; (void)scales_f16_ptr;
  (void)out_cols; (void)softcap; (void)output_f32;
  return nullptr;
#endif
}

mlx_array* mlx_inkling2_q8_sym_matmul_bf16(
    mlx_array* x_ptr,
    mlx_array* w_q8_ptr,
    mlx_array* scales_f16_ptr,
    bool output_f32) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !w_q8_ptr || !scales_f16_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(w_q8_ptr);
    const auto& sc = *reinterpret_cast<array*>(scales_f16_ptr);
    const int max_rows = metal::is_nax_available() ? 2048 : 32;
    if (x.ndim() != 2 || w.ndim() != 2 || sc.ndim() != 2 ||
        x.dtype() != bfloat16 || w.dtype() != uint32 || sc.dtype() != float16 ||
        x.shape(0) <= 0 || x.shape(0) > max_rows ||
        x.shape(1) <= 0 || (x.shape(1) % 32) != 0 ||
        w.shape(0) <= 0 || w.shape(1) != x.shape(1) / 4 ||
        sc.shape(0) != w.shape(0) || sc.shape(1) != x.shape(1) / 32) {
      return nullptr;
    }
    const uint32_t rows = static_cast<uint32_t>(x.shape(0));
    const uint32_t out_dim = static_cast<uint32_t>(w.shape(0));
    const uint32_t in_dim = static_cast<uint32_t>(x.shape(1));
    const Dtype out_dtype = output_f32 ? float32 : bfloat16;
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2Q8SymMatmul>(
        s, rows, out_dim, in_dim, 0u, out_dtype);
    auto res = array(
        Shape{static_cast<int>(rows), static_cast<int>(out_dim)},
        out_dtype,
        std::move(prim),
        {x, w, sc});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_q8_sym_matmul_bf16", e.what());
    return nullptr;
  }
#else
  (void)x_ptr; (void)w_q8_ptr; (void)scales_f16_ptr; (void)output_f32;
  return nullptr;
#endif
}

bool mlx_inkling2_q8_sym_matmul_bf16_with_aux(
    mlx_array* x_ptr,
    mlx_array* w_q8_ptr,
    mlx_array* scales_f16_ptr,
    mlx_array* w_aux_bf16_ptr,
    bool output_f32,
    mlx_array** out_main,
    mlx_array** out_aux) {
  if (out_main) *out_main = nullptr;
  if (out_aux) *out_aux = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !w_q8_ptr || !scales_f16_ptr || !w_aux_bf16_ptr || !out_main || !out_aux) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& w = *reinterpret_cast<array*>(w_q8_ptr);
    const auto& sc = *reinterpret_cast<array*>(scales_f16_ptr);
    const auto& w_aux = *reinterpret_cast<array*>(w_aux_bf16_ptr);
    const int max_rows = metal::is_nax_available() ? 2048 : 32;
    if (x.ndim() != 2 || w.ndim() != 2 || sc.ndim() != 2 || w_aux.ndim() != 2 ||
        x.dtype() != bfloat16 || w.dtype() != uint32 || sc.dtype() != float16 ||
        w_aux.dtype() != bfloat16 ||
        x.shape(0) <= 0 || x.shape(0) > max_rows ||
        x.shape(1) <= 0 || (x.shape(1) % 32) != 0 ||
        w.shape(0) <= 0 || w.shape(1) != x.shape(1) / 4 ||
        sc.shape(0) != w.shape(0) || sc.shape(1) != x.shape(1) / 32 ||
        w_aux.shape(0) <= 0 || w_aux.shape(1) != x.shape(1)) {
      return false;
    }
    const uint32_t rows = static_cast<uint32_t>(x.shape(0));
    const uint32_t out_dim = static_cast<uint32_t>(w.shape(0));
    const uint32_t in_dim = static_cast<uint32_t>(x.shape(1));
    const uint32_t aux_dim = static_cast<uint32_t>(w_aux.shape(0));
    const Dtype out_dtype = output_f32 ? float32 : bfloat16;
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2Q8SymMatmul>(
        s, rows, out_dim, in_dim, aux_dim, out_dtype);
    auto results = array::make_arrays(
        {Shape{static_cast<int>(rows), static_cast<int>(out_dim)},
         Shape{static_cast<int>(rows), static_cast<int>(aux_dim)}},
        {out_dtype, bfloat16},
        std::move(prim),
        {x, w, sc, w_aux});
    *out_main = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_aux = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_q8_sym_matmul_bf16_with_aux", e.what());
    return false;
  }
#else
  (void)x_ptr; (void)w_q8_ptr; (void)scales_f16_ptr; (void)w_aux_bf16_ptr; (void)output_f32;
  return false;
#endif
}

bool mlx_inkling2_loop_ctrl_fused(
    mlx_array* dest_ptr,
    mlx_array* msg_ptr,
    mlx_array* down_w_ptr,
    mlx_array* gain_forget_w_ptr,
    mlx_array* norm_weight_ptr,
    float attn_res_eps,
    float norm_eps,
    bool output_f32,
    bool split_outputs,
    mlx_array** out_0,
    mlx_array** out_1,
    mlx_array** out_2) {
  if (out_0) *out_0 = nullptr;
  if (out_1) *out_1 = nullptr;
  if (out_2) *out_2 = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!dest_ptr || !msg_ptr || !down_w_ptr || !gain_forget_w_ptr ||
      !norm_weight_ptr || !out_0) {
    return false;
  }
  if (split_outputs && (!out_1 || !out_2)) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!metal::is_available()) {
      return false;
    }
    const auto& dest = *reinterpret_cast<array*>(dest_ptr);
    const auto& msg = *reinterpret_cast<array*>(msg_ptr);
    const auto& down_w = *reinterpret_cast<array*>(down_w_ptr);
    const auto& gf_w = *reinterpret_cast<array*>(gain_forget_w_ptr);
    const auto& weight = *reinterpret_cast<array*>(norm_weight_ptr);
    if (!is_bf16_or_f32(dest.dtype()) || !is_bf16_or_f32(msg.dtype()) ||
        !is_bf16_or_f32(down_w.dtype()) || !is_bf16_or_f32(gf_w.dtype()) ||
        !is_bf16_or_f32(weight.dtype()) || msg.size() == 0 ||
        (msg.ndim() != 2 && msg.ndim() != 3)) {
      return false;
    }
    uint32_t rank = 0;
    if (msg.ndim() == 2) {
      if (split_outputs) {
        return false;
      }
      const int rows = msg.shape(0);
      const int dim = msg.shape(1);
      if (rows <= 0 || dim <= 0 || (dim % 4) != 0 ||
          dest.ndim() != 2 || dest.shape(0) != rows || dest.shape(1) != dim ||
          down_w.ndim() != 2 || down_w.shape(0) <= 0 || (down_w.shape(0) % 4) != 0 ||
          down_w.shape(1) != 2 * dim ||
          gf_w.ndim() != 2 || gf_w.shape(0) != 2 * dim || gf_w.shape(1) != down_w.shape(0) ||
          weight.ndim() != 1 || weight.shape(0) != dim) {
        return false;
      }
      rank = static_cast<uint32_t>(down_w.shape(0));
    } else {
      const int streams = msg.shape(0);
      const int rows = msg.shape(1);
      const int dim = msg.shape(2);
      if (split_outputs && streams != 3) {
        return false;
      }
      const bool dest_ok =
          (dest.ndim() == 3 && (dest.shape(0) == streams || dest.shape(0) == 1) &&
           dest.shape(1) == rows && dest.shape(2) == dim) ||
          (dest.ndim() == 2 && dest.shape(0) == rows && dest.shape(1) == dim);
      const bool weight_ok =
          (weight.ndim() == 1 && weight.shape(0) == dim) ||
          (weight.ndim() == 2 && (weight.shape(0) == streams || weight.shape(0) == 1) &&
           weight.shape(1) == dim);
      if (streams <= 0 || rows <= 0 || dim <= 0 || (dim % 4) != 0 ||
          !dest_ok || !weight_ok ||
          down_w.ndim() != 3 || down_w.shape(0) != streams ||
          down_w.shape(1) <= 0 || (down_w.shape(1) % 4) != 0 ||
          down_w.shape(2) != 2 * dim ||
          gf_w.ndim() != 3 || gf_w.shape(0) != streams ||
          gf_w.shape(1) != 2 * dim || gf_w.shape(2) != down_w.shape(1)) {
        return false;
      }
      rank = static_cast<uint32_t>(down_w.shape(1));
    }
    const Dtype out_dtype = output_f32 ? float32 : bfloat16;
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2LoopCtrlFused>(
        s, attn_res_eps, norm_eps, rank, out_dtype, msg.shape(), split_outputs);
    if (split_outputs) {
      Shape s2d{msg.shape(1), msg.shape(2)};
      auto results = array::make_arrays(
          {s2d, s2d, s2d},
          {out_dtype, out_dtype, out_dtype},
          std::move(prim),
          {dest, msg, down_w, gf_w, weight});
      *out_0 = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      *out_1 = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
      *out_2 = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
    } else {
      auto res = array(
          msg.shape(), out_dtype, std::move(prim),
          {dest, msg, down_w, gf_w, weight});
      *out_0 = reinterpret_cast<mlx_array*>(new array(std::move(res)));
    }
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_loop_ctrl_fused", e.what());
    return false;
  }
#else
  (void)dest_ptr; (void)msg_ptr; (void)down_w_ptr; (void)gain_forget_w_ptr;
  (void)norm_weight_ptr; (void)attn_res_eps; (void)norm_eps; (void)output_f32;
  (void)split_outputs;
  return false;
#endif
}

bool mlx_inkling2_oproj_sconv_k4_ragged(
    mlx_array* y_ptr,
    mlx_array* weight_ptr,
    mlx_array* scales_ptr,
    mlx_array* conv_weight_ptr,
    mlx_array* cu_seqlens_ptr,
    mlx_array* res_a_ptr,
    mlx_array* const* hists_ptr,
    uint32_t num_seqs,
    bool packed_hist,
    mlx_array* tree_rows_ptr,
    mlx_array** out_y,
    mlx_array** out_history,
    mlx_array** out_tape) {
  if (out_y) *out_y = nullptr;
  if (out_history) *out_history = nullptr;
  if (out_tape) *out_tape = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!y_ptr || !weight_ptr || !conv_weight_ptr || !hists_ptr ||
      !out_y || !out_history || num_seqs == 0u) {
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
    const auto& y = *reinterpret_cast<array*>(y_ptr);
    const auto& w = *reinterpret_cast<array*>(weight_ptr);
    const auto& conv_w = *reinterpret_cast<array*>(conv_weight_ptr);
    const bool is_q8 = scales_ptr != nullptr;
    if ((y.ndim() != 2 && y.ndim() != 3) || y.dtype() != bfloat16 ||
        w.ndim() != 2 || conv_w.ndim() != 2 || !is_bf16_or_f32(conv_w.dtype())) {
      return false;
    }
    const int t = y.shape(0);
    const int in_dim = (y.ndim() == 2) ? y.shape(1) : (y.shape(1) * y.shape(2));
    const int out_dim = w.shape(0);
    const int max_rows = is_q8
        ? (metal::is_nax_available() ? 2048 : 32)
        : (metal::is_nax_available() ? 1024 : 32);
    if (t <= 0 || t > max_rows || in_dim <= 0 || out_dim <= 0 ||
        conv_w.shape(0) != out_dim || conv_w.shape(1) != 4) {
      return false;
    }
    std::vector<array> inputs = {y, w};
    if (is_q8) {
      const auto& sc = *reinterpret_cast<array*>(scales_ptr);
      if (w.dtype() != uint32 || sc.ndim() != 2 || sc.dtype() != float16 ||
          (in_dim % 32) != 0 || w.shape(1) != in_dim / 4 ||
          sc.shape(0) != out_dim || sc.shape(1) != in_dim / 32) {
        return false;
      }
      inputs.push_back(sc);
    } else {
      if (w.dtype() != bfloat16 || (in_dim % 4) != 0 || w.shape(1) != in_dim) {
        return false;
      }
    }
    inputs.push_back(conv_w);

    if (num_seqs > 1u) {
      const auto& cu = *reinterpret_cast<array*>(cu_seqlens_ptr);
      if (cu.ndim() != 1 || cu.dtype() != int32 ||
          cu.shape(0) != static_cast<int>(num_seqs + 1u)) {
        return false;
      }
      inputs.push_back(cu);
    }
    const bool has_res_a = res_a_ptr != nullptr;
    if (has_res_a) {
      const auto& ra = *reinterpret_cast<array*>(res_a_ptr);
      if (ra.ndim() != 2 || ra.shape(0) != t || ra.shape(1) != out_dim ||
          !is_bf16_or_f32(ra.dtype())) {
        return false;
      }
      inputs.push_back(ra);
    }
    const uint32_t h_count = packed_hist ? 1u : num_seqs;
    for (uint32_t i = 0; i < h_count; ++i) {
      if (!hists_ptr[i]) return false;
      const auto& h = *reinterpret_cast<array*>(hists_ptr[i]);
      if (!is_bf16_or_f32(h.dtype())) return false;
      if (packed_hist) {
        if (h.ndim() != 3 || h.shape(0) != static_cast<int>(num_seqs) ||
            h.shape(1) != 3 || h.shape(2) != out_dim) {
          return false;
        }
      } else {
        if (h.ndim() != 2 || h.shape(0) != 3 || h.shape(1) != out_dim) {
          return false;
        }
      }
      inputs.push_back(h);
    }
    const bool has_tree = tree_rows_ptr != nullptr;
    if (has_tree) {
      const auto& tr = *reinterpret_cast<array*>(tree_rows_ptr);
      if (!validate_tree_rows_array(tr, t)) {
        return false;
      }
      inputs.push_back(tr);
    }
    const bool emit_tape = out_tape != nullptr;
    Shape y_s{t, out_dim};
    Shape h_s = (num_seqs == 1u && !packed_hist)
        ? Shape{3, out_dim}
        : Shape{static_cast<int>(num_seqs), 3, out_dim};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2OProjSconvResidual>(
        s, static_cast<uint32_t>(t), static_cast<uint32_t>(out_dim),
        static_cast<uint32_t>(in_dim), is_q8, num_seqs, packed_hist,
        has_res_a, has_tree, emit_tape);
    std::vector<Shape> out_shapes = {y_s, h_s};
    std::vector<Dtype> out_dtypes = {bfloat16, bfloat16};
    if (emit_tape) {
      out_shapes.push_back(y_s);
      out_dtypes.push_back(bfloat16);
    }
    auto results = array::make_arrays(
        std::move(out_shapes), std::move(out_dtypes), std::move(prim),
        std::move(inputs));
    *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_history = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    if (emit_tape) {
      *out_tape = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
    }
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_oproj_sconv_k4_ragged", e.what());
    return false;
  }
#else
  (void)y_ptr; (void)weight_ptr; (void)scales_ptr; (void)conv_weight_ptr;
  (void)cu_seqlens_ptr; (void)res_a_ptr; (void)hists_ptr; (void)num_seqs;
  (void)packed_hist; (void)tree_rows_ptr;
  return false;
#endif
}

} // extern "C"
