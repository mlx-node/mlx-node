#pragma once

#include "mlx/mlx.h"
#include "mlx/fast.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/memory.h"
#include "mlx/compile.h"
#include "mlx/compile_impl.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/gpu/device_info.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <mutex>

// Forward declaration of opaque handle type for FFI
struct mlx_array;

// Stream struct for FFI (matches Rust definition)
struct mlx_stream {
  int32_t index;
  int32_t device_type;  // 0 = CPU, 1 = GPU
};

namespace mlx::core::fast::paged {

// Standardized metadata that EVERY paged-attention model's compiled forward
// receives. Shapes are FIXED at compile time (sentinel-padded contents); only
// contents change per request — that's what keeps the compile cache hitting
// across calls instead of re-tracing on every shape change.
//
// The struct is a thin POD-style aggregator — it does NOT own the arrays.
// Callers (the Rust adapter) materialize the arrays once per request,
// hand the struct into the compiled graph, and the arrays die with the
// caller's stack frame. No copies, no ref counting beyond MLX's own
// `array` machinery.
struct PagedAttentionInputs {
  // Global token position of the first new token in this request, broadcast
  // as a `[1]` int32 array so the compile cache treats it as a tracer rather
  // than a fixed scalar (matches the per-request `offset` array threaded
  // through `mlx_qwen35.cpp` today).
  array offset_arr;

  // Per-request block table, sentinel-padded with -1 to a fixed
  // `[1, max_blocks_per_seq]` int32 shape. The kernel reads
  // `block_table[seq_idx, block_idx]` and uses -1 entries as a dispatch-time
  // skip signal (validated by `paged_attention` factory).
  array block_table;

  // Per-token slot mapping for the current write chunk, sentinel-padded with
  // -1 to a fixed `[chunk_size_max]` int64 shape. The kernel reads
  // `slot_mapping[token_idx]` and computes `block_id = slot / block_size`
  // for the K/V write target. Sentinel slots are skipped on dispatch
  // (validated factory-side).
  array slot_mapping;

  // Valid prefix length of `slot_mapping` (so the write kernel knows how
  // many tokens of the padded chunk are real). `[1]` int32.
  array num_valid_tokens;

  // Valid prefix length of `block_table` (so the gather kernel knows how
  // many blocks of the padded table are real). `[1]` int32.
  array num_valid_blocks;

  // Total context length so far for this single-request adapter (= number of
  // tokens already recorded after the chunk being written), threaded as a
  // `[1]` int32 array so the kernel reads `context_lens[seq_idx=0]` from
  // the right place. Mirrors vLLM's `seq_lens` argument to the
  // `paged_attention` kernel (one entry per dispatched sequence).
  array seq_lens;
};

}  // namespace mlx::core::fast::paged

// =============================================================================
// NA int8 W8A8 lazy graph builders (defined in mlx_na_int8.cpp).
//
// External-linkage C++ entry points for the sym8 (per-output-channel symmetric
// int8) linear path. The eager Rust `QuantizedLinear::forward_sym8` emits
// these graphs via the `mlx_w8a8_linear` / `mlx_int8_qmv` FFI wrappers, which
// delegate to these builders. Keeping the math in ONE definition means every
// consumer emits byte-identical graph nodes for the same inputs.
//
// Contract (both):
//   x    : [M,K] activations (bf16; non-bf16 defensively cast)
//   w_kn : [K,N] CONTIGUOUS int8 kernel operand, pre-transposed at load/
//          registration time (NOT the checkpoint [N,K] tensor)
//   s_w  : [N] f32 per-output-channel weight scale
//   returns LAZY bf16 [M,N] = x @ w^T (composes into the surrounding graph)
//
// All THROW std::runtime_error on contract violations (caller-visible
// fail-loud; the extern "C" wrappers translate to cerr + false for Rust).
// Dispatch rule mirrors Rust: M <= 2 -> qmv_w8a16 (decode matvec, bf16
// activations read directly — NO act quant), M >= 3 -> the W8A8 prefill GEMM.
// qmv_lazy is the OLD W8A8 decode matvec, kept for A/B (qmv_w8a16_lazy falls
// back to it under INT8_QMV_W8A16=0) and microbenches.
// qmv_w8a16_lazy takes BOTH weight orientations: w_kn [K,N] (the GEMM operand,
// consumed by the 2D-block fallback under INT8_QMV16_SG=0 and the W8A8
// reroute) and w_nk [N,K] (the CHECKPOINT orientation, consumed by the default
// simd_sum-style kernel — buffer-shared with the params tensor).
// =============================================================================
namespace na_int8 {
mlx::core::array w8a8_linear_lazy(const mlx::core::array& x,
                                  const mlx::core::array& w_kn,
                                  const mlx::core::array& s_w);
mlx::core::array qmv_lazy(const mlx::core::array& x,
                          const mlx::core::array& w_kn,
                          const mlx::core::array& s_w);
mlx::core::array qmv_w8a16_lazy(const mlx::core::array& x,
                                const mlx::core::array& w_kn,
                                const mlx::core::array& w_nk,
                                const mlx::core::array& s_w);
}  // namespace na_int8

namespace {
using mlx::core::add;
using mlx::core::arange;
using mlx::core::array;
using mlx::core::astype;
using mlx::core::clip;
using mlx::core::concatenate;
using mlx::core::copy;
using mlx::core::exp;
using mlx::core::eye;
using mlx::core::full;
using mlx::core::linspace;
using mlx::core::log;
using mlx::core::logsumexp;
using mlx::core::matmul;
using mlx::core::maximum;
using mlx::core::mean;
using mlx::core::minimum;
using mlx::core::ones;
using mlx::core::reshape;
using mlx::core::Shape;
using mlx::core::ShapeElem;
using mlx::core::slice;
using mlx::core::squeeze;
using mlx::core::stack;
using mlx::core::sum;
using mlx::core::take;
using mlx::core::transpose;
using mlx::core::zeros;

inline bool mlx_env_flag_enabled(const char* value) {
  if (!value) return false;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

inline void mlx_trace_native_error(const char* context, const char* detail) {
  if (!mlx_env_flag_enabled(std::getenv("MLX_INFERENCE_TRACE"))) return;
  const char* path = std::getenv("MLX_INFERENCE_TRACE_FILE");
  if (!path || !*path) return;
  std::ofstream out(path, std::ios::app);
  if (!out) return;
  out << "native_error context=" << (context ? context : "unknown") << " detail=\"";
  if (detail) {
    for (const char* p = detail; *p; ++p) {
      char c = *p;
      out << ((c == '\n' || c == '\r' || c == '"') ? ' ' : c);
    }
  }
  out << "\"\n";
}

// Comparison operations
using mlx::core::equal;
using mlx::core::greater;
using mlx::core::greater_equal;
using mlx::core::less;
using mlx::core::less_equal;
using mlx::core::not_equal;

// Logical operations
using mlx::core::logical_and;
using mlx::core::logical_not;
using mlx::core::logical_or;
using mlx::core::where;

// Reduction operations
using mlx::core::argmax;
using mlx::core::argmin;
using mlx::core::cumprod;
using mlx::core::cumsum;
using mlx::core::max;
using mlx::core::min;
using mlx::core::prod;
using mlx::core::std;
using mlx::core::var;

// Array manipulation
using mlx::core::argpartition;
using mlx::core::argsort;
using mlx::core::broadcast_to;
using mlx::core::expand_dims;
using mlx::core::pad;
using mlx::core::partition;
using mlx::core::repeat;
using mlx::core::roll;
using mlx::core::sort;
using mlx::core::split;
using mlx::core::tile;

// Math operations
using mlx::core::abs;
using mlx::core::ceil;
using mlx::core::cos;
using mlx::core::cosh;
using mlx::core::floor;
using mlx::core::negative;
using mlx::core::power;
using mlx::core::round;
using mlx::core::sign;
using mlx::core::sin;
using mlx::core::sinh;
using mlx::core::sqrt;
using mlx::core::square;
using mlx::core::tan;
using mlx::core::tanh;

// Fast operations
namespace fast = mlx::core::fast;

// Stream and evaluation
using mlx::core::async_eval;
using mlx::core::clear_cache;
using mlx::core::default_device;
using mlx::core::default_stream;
using mlx::core::Device;
using mlx::core::eval;
using mlx::core::flatten;
using mlx::core::new_stream;
using mlx::core::scatter;
using mlx::core::sigmoid;
using mlx::core::Stream;
using mlx::core::StreamContext;

Shape make_shape(const int64_t* dims, size_t ndim) {
  Shape shape;
  shape.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    shape.push_back(static_cast<ShapeElem>(dims[i]));
  }
  return shape;
}

std::vector<int> make_axes(const int32_t* axes, size_t len) {
  std::vector<int> result;
  result.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    result.push_back(static_cast<int>(axes[i]));
  }
  return result;
}

enum BridgeDType : int32_t {
  FLOAT32 = 0,
  INT32 = 1,
  FLOAT16 = 2,
  BFLOAT16 = 3,
  UINT32 = 4,
  UINT8 = 5,
  INT8 = 6,  // sym8 per-channel symmetric int8 checkpoint weights
};

mlx::core::Dtype to_mlx_dtype(int32_t code) {
  switch (code) {
    case FLOAT32:
      return mlx::core::float32;
    case INT32:
      return mlx::core::int32;
    case FLOAT16:
      return mlx::core::float16;
    case BFLOAT16:
      return mlx::core::bfloat16;
    case UINT32:
      return mlx::core::uint32;
    case UINT8:
      return mlx::core::uint8;
    case INT8:
      return mlx::core::int8;
    default:
      return mlx::core::float32;
  }
}

int32_t from_mlx_dtype(mlx::core::Dtype dtype) {
  switch (dtype) {
    case mlx::core::float32:
      return FLOAT32;
    case mlx::core::int32:
      return INT32;
    case mlx::core::float16:
      return FLOAT16;
    case mlx::core::bfloat16:
      return BFLOAT16;
    case mlx::core::uint32:
      return UINT32;
    case mlx::core::uint8:
      return UINT8;
    case mlx::core::int8:
      return INT8;
    default:
      return -1;
  }
}

bool copy_to_buffer(const array& arr, float* out, size_t len) {
  // Force materialization by adding zeros - this ensures broadcast values are
  // expanded
  auto zeros_arr = zeros(arr.shape(), arr.dtype());
  auto materialized = add(arr, zeros_arr);
  materialized.eval();

  // Now flatten and copy
  auto flat = flatten(materialized);
  auto host = (flat.dtype() == mlx::core::float32)
                  ? flat
                  : astype(flat, mlx::core::float32);
  host.eval();

  if (host.size() != len) {
    return false;
  }
  const float* data = host.data<float>();
  std::copy(data, data + len, out);
  return true;
}

bool copy_to_buffer(const array& arr, int32_t* out, size_t len) {
  // Force materialization by adding zeros - this ensures broadcast values are
  // expanded
  auto zeros_arr = zeros(arr.shape(), arr.dtype());
  auto materialized = add(arr, zeros_arr);
  materialized.eval();

  // Now flatten and copy
  auto flat = flatten(materialized);
  auto host = (flat.dtype() == mlx::core::int32)
                  ? flat
                  : astype(flat, mlx::core::int32);
  host.eval();

  if (host.size() != len) {
    return false;
  }
  const int32_t* data = host.data<int32_t>();
  std::copy(data, data + len, out);
  return true;
}

bool copy_to_buffer(const array& arr, uint32_t* out, size_t len) {
  // Force materialization by adding zeros - this ensures broadcast values are
  // expanded
  auto zeros_arr = zeros(arr.shape(), arr.dtype());
  auto materialized = add(arr, zeros_arr);
  materialized.eval();

  // Now flatten and copy
  auto flat = flatten(materialized);
  auto host = (flat.dtype() == mlx::core::uint32)
                  ? flat
                  : astype(flat, mlx::core::uint32);
  host.eval();

  if (host.size() != len) {
    return false;
  }
  const uint32_t* data = host.data<uint32_t>();
  std::copy(data, data + len, out);
  return true;
}

}  // namespace
