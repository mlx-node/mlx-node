// Implements the `PagedKVWrite` and `PagedAttention` MLX `Custom`
// primitives. Their `eval_gpu` paths dispatch onto MLX's own
// `metal::CommandEncoder` via `crates/mlx-sys/src/mlx_paged_dispatch.cpp`.
//
// Because the dispatch is on MLX's command queue, MLX's dependency
// tracking is correct: callers do not need to `eval()` ancestor arrays
// before invoking `paged_kv_write` / `paged_attention`, nor `eval()` the
// outputs before reading them — the standard MLX evaluation order
// guarantees correctness.

#include "mlx_paged_ops.h"
#include "mlx_common.h"
#include "mlx_paged_dispatch.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/compile.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::fast {

namespace {
void validate_paged_attention_plan(uint8_t route_hint, uint32_t grouped_stripes) {
  if (grouped_stripes != 0 &&
      (grouped_stripes < 4 || grouped_stripes > (route_hint == 3 ? 1024u : 256u) ||
       (grouped_stripes & (grouped_stripes - 1)) != 0 || (route_hint != 1 && route_hint != 3))) {
    throw std::invalid_argument("paged attention stripes require grouped route and a supported power of two");
  }
  if (route_hint > 3) {
    throw std::invalid_argument(
        "paged_attention_with_route_hint: route_hint must be 0, 1, 2, or 3");
  }
}


// The runtime validators establish `value >= 0` and `divisor > 0` before
// either caller reaches this helper. Widen before arithmetic so INT32_MAX is
// a valid sequence length rather than overflowing in `value + divisor - 1`.
constexpr int64_t ceil_div_nonnegative_i32(int32_t value, int32_t divisor) {
  const int64_t wide_value = static_cast<int64_t>(value);
  const int64_t wide_divisor = static_cast<int64_t>(divisor);
  return wide_value / wide_divisor + (wide_value % wide_divisor != 0);
}

static_assert(
    ceil_div_nonnegative_i32(std::numeric_limits<int32_t>::max(), 16) ==
    134217728);

// Distinct pool blocks the first ceil(seq_len / block_size) entries of every
// table row reach (Splash `PageStorage::writeTable` sorts and dedups the
// entries it records): a prefix two rows share counts once. Runs on every
// read after the runtime bounds checks put each entry in [0, num_blocks).
int64_t distinct_live_blocks(
    const int32_t* block_table,
    const int32_t* seq_lens,
    uint32_t num_seqs,
    uint32_t max_blocks_per_seq,
    int block_size,
    int32_t num_blocks) {
  std::vector<uint64_t> seen((static_cast<size_t>(num_blocks) + 63) / 64, 0);
  int64_t live = 0;
  for (uint32_t i = 0; i < num_seqs; ++i) {
    const int64_t used = std::min<int64_t>(
        ceil_div_nonnegative_i32(seq_lens[i], block_size), max_blocks_per_seq);
    const int32_t* row =
        block_table + static_cast<size_t>(i) * max_blocks_per_seq;
    for (int64_t j = 0; j < used; ++j) {
      const uint32_t blk = static_cast<uint32_t>(row[j]);
      uint64_t& word = seen[blk / 64];
      const uint64_t bit = uint64_t{1} << (blk % 64);
      live += (word & bit) == 0;
      word |= bit;
    }
  }
  return live;
}

// Derive `x_pack` (the inner-axis vLLM K-pool packing factor) from the
// on-cache KV dtype. `x_pack` is `16 / sizeof(scalar)`:
//   - Fp16 / Bf16 (2 bytes) → x_pack = 8
//   - Int8 (1 byte)         → x_pack = 16
// Used by the public factories to validate `k_pool.shape(2)` and
// `k_pool.shape(4)` against the dtype-derived layout.
int x_pack_for(KvDtype kv_dtype) {
  switch (kv_dtype) {
    case KvDtype::Fp16:
    case KvDtype::Bf16:
      return 8;
    case KvDtype::Int8:
      return 16;
  }
  // Unreachable — quiet the warning.
  return 8;
}

// Map the `KvDtype` enum to the matching MLX `Dtype` for io tensors:
// io == cache for Fp16/Bf16, bfloat16 for Int8.
mlx::core::Dtype io_dtype_for_kv_dtype(KvDtype kv_dtype) {
  switch (kv_dtype) {
    case KvDtype::Fp16:
      return mlx::core::float16;
    case KvDtype::Bf16:
      return mlx::core::bfloat16;
    case KvDtype::Int8:
      return mlx::core::bfloat16;
  }
  // Unreachable; quiet the warning.
  return mlx::core::bfloat16;
}

// Map the `KvDtype` enum to the matching MLX `Dtype` for the on-cache
// (K/V pool) storage:
//   - Fp16 cache → float16
//   - Bf16 cache → bfloat16
//   - Int8 cache → int8
mlx::core::Dtype cache_dtype_for_kv_dtype(KvDtype kv_dtype) {
  switch (kv_dtype) {
    case KvDtype::Fp16:
      return mlx::core::float16;
    case KvDtype::Bf16:
      return mlx::core::bfloat16;
    case KvDtype::Int8:
      return mlx::core::int8;
  }
  // Unreachable; quiet the warning.
  return mlx::core::bfloat16;
}

// Translate the public `KvDtype` enum into the dispatch-internal
// `paged::KvDtype` so we can call `mlx::core::fast::paged::dispatch_*`.
// Both enums have the same wire values; the cast just satisfies the
// type system.
mlx::core::fast::paged::KvDtype to_paged_dtype(KvDtype d) {
  switch (d) {
    case KvDtype::Fp16:
      return mlx::core::fast::paged::KvDtype::Fp16;
    case KvDtype::Bf16:
      return mlx::core::fast::paged::KvDtype::Bf16;
    case KvDtype::Int8:
      return mlx::core::fast::paged::KvDtype::Int8;
  }
  return mlx::core::fast::paged::KvDtype::Bf16;
}

static_assert(
    static_cast<uint8_t>(KvDtype::Fp16) ==
        static_cast<uint8_t>(mlx::core::fast::paged::KvDtype::Fp16) &&
    static_cast<uint8_t>(KvDtype::Bf16) ==
        static_cast<uint8_t>(mlx::core::fast::paged::KvDtype::Bf16) &&
    static_cast<uint8_t>(KvDtype::Int8) ==
        static_cast<uint8_t>(mlx::core::fast::paged::KvDtype::Int8));

// An int8 pool carries one f32 scale per (block, head, token) (Splash
// splash_q8_scale_index), written in place by the store and read by the
// attention kernels; a short plane is indexed past. Every other pool takes
// unit 1-element scales. The int8 kernels exist for head sizes 128 and 256
// (Splash's) and, on reads, block sizes 16 and 32.
void validate_kv_scales(
    const array& k_pool,
    const array& k_scale,
    const array& v_scale,
    int num_kv_heads,
    int block_size,
    int head_size,
    KvDtype kv_dtype,
    bool read,
    const char* op_name) {
  if (k_scale.dtype() != mlx::core::float32 ||
      v_scale.dtype() != mlx::core::float32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_scale / v_scale must be float32";
    throw std::invalid_argument(msg.str());
  }
  if (kv_dtype != KvDtype::Int8) {
    if (k_scale.size() != 1 || v_scale.size() != 1) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name
          << "] k_scale / v_scale must be 1-element arrays";
      throw std::invalid_argument(msg.str());
    }
    return;
  }
  const Shape expected{k_pool.ndim() > 0 ? k_pool.shape(0) : 0,
                       num_kv_heads,
                       block_size};
  if (k_scale.shape() != expected || v_scale.shape() != expected) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] int8 k_scale / v_scale must be float32 [num_blocks, "
        << "num_kv_heads, block_size] = [" << expected[0] << ", "
        << expected[1] << ", " << expected[2] << "]";
    throw std::invalid_argument(msg.str());
  }
  if (head_size != 128 && head_size != 256) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] int8 paged pools have head_size "
        << "128 or 256 only; got " << head_size;
    throw std::invalid_argument(msg.str());
  }
  if (read && block_size != 16 && block_size != 32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] int8 paged reads have block_size "
        << "16 or 32 only; got " << block_size;
    throw std::invalid_argument(msg.str());
  }
}

// Kernels use dense row-major indexing, so strided inputs need compaction.
// CommandEncoder::set_input_array binds the array's byte offset. Read-only
// activations (q, new K/V) may therefore be contiguous offset views; q's
// offset must be whole heads, the only q alignment the kernels address.
// Keep the stricter zero-offset contract for mutable pools and
// CPU-inspected metadata.
//
// The stricter zero-offset check passes on:
//   - tracer arrays from `mlx::core::compile()` (default flags
//     `{true, true, true}`, default offset 0; see `array.h:495`)
//   - shape-only arrays from negative-test helpers
//     (`array(Shape{...}, dtype, nullptr, {})`)
//   - normal `eval()`-ed arrays before any slice/transpose op
//
// The stricter zero-offset check rejects:
//   - results of `mlx::core::slice(...)` with nonzero start (offset != 0
//     and/or row_contiguous == false)
//   - results of `mlx::core::transpose(...)` (row_contiguous == false)
//   - any view returned by `array::copy_shared_buffer(...)` that
//     adjusted strides or set a non-default offset
void require_row_contiguous(
    const array& arr,
    const char* op_name,
    const char* input_name) {
  if (!arr.flags().row_contiguous) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] " << input_name
        << " must be row-contiguous; got non-row-contiguous view "
        << "(probably from a transpose / non-trivial slice). "
        << "Materialize a contiguous copy first via "
        << "`mlx::core::contiguous(arr)` before passing to the primitive.";
    throw std::invalid_argument(msg.str());
  }
}

void require_row_contiguous_head_offset(
    const array& arr,
    const char* op_name,
    const char* input_name) {
  require_row_contiguous(arr, op_name, input_name);
  if (arr.offset() == 0) {
    return;
  }
  const int64_t head_bytes = arr.ndim() == 0
      ? 0
      : static_cast<int64_t>(arr.shape(-1)) *
          static_cast<int64_t>(arr.itemsize());
  if (head_bytes == 0 || arr.offset() % head_bytes != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] " << input_name
        << " offset must be a multiple of one head (" << head_bytes
        << " bytes); got offset " << arr.offset()
        << ". Materialize a contiguous copy first via "
        << "`mlx::core::contiguous(arr)` before passing to the primitive.";
    throw std::invalid_argument(msg.str());
  }
}

void require_row_contiguous_zero_offset(
    const array& arr,
    const char* op_name,
    const char* input_name) {
  require_row_contiguous(arr, op_name, input_name);
  if (arr.offset() != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] " << input_name
        << " must have offset 0; got offset " << arr.offset()
        << " (probably from a sliced view starting partway into the "
        << "backing buffer). Materialize a contiguous copy first via "
        << "`mlx::core::contiguous(arr)` before passing to the primitive.";
    throw std::invalid_argument(msg.str());
  }
}

// The rows' score scales of an INT8-query decode read: f32 [rows,
// num_q_heads] over INT8 pages.
void validate_score_scales(
    const array& score_scales,
    int rows,
    int num_q_heads,
    KvDtype kv_dtype,
    const char* op_name) {
  require_row_contiguous_zero_offset(score_scales, op_name, "score_scales");
  if (kv_dtype != KvDtype::Int8 || score_scales.dtype() != mlx::core::float32 ||
      score_scales.shape() != Shape{rows, num_q_heads}) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] score_scales must be float32 [query rows, query heads] over INT8 pages";
    throw std::invalid_argument(msg.str());
  }
}

// =============================================================================
// Unified factory + eval_gpu validation.
//
// `mlx::core::compile`'s cache-hit replay bypasses the factory: the
// cache key only compares rank/shape/dtype, and `compile_replace`
// rebuilds the cached primitive with real inputs WITHOUT re-running the
// factory. So every structural / scalar / shape / dtype / contiguity
// check lives in a single helper that the factory AND eval_gpu BOTH
// call, closing the "factory-only check bypassed by compile cache"
// hazard class.
//
// The helper does NOT perform runtime data-dependent bounds checks
// (slot_mapping max < pool capacity for PagedKVWrite; seq_lens /
// block_table content checks for PagedAttention) — those require
// materialized data and stay in the dedicated runtime check immediately
// following the helper call.
//
// `op_name` is prefixed onto every error message so the caller knows
// which path raised (e.g. "[paged_kv_write]" for the factory,
// "PagedKVWrite::eval_gpu:" for the eval path).
// =============================================================================

void validate_paged_kv_write_inputs(
    const array& k_pool,
    const array& v_pool,
    const array& new_k,
    const array& new_v,
    const array& slot_mapping,
    const array& k_scale,
    const array& v_scale,
    int block_size,
    int num_kv_heads,
    int head_size,
    int x_pack,
    KvDtype kv_dtype,
    bool values_token_major,
    const char* op_name) {
  // 1. Dense layout on every input; only read-only new K/V accept offsets.
  require_row_contiguous_zero_offset(k_pool, op_name, "k_pool");
  require_row_contiguous_zero_offset(v_pool, op_name, "v_pool");
  require_row_contiguous(new_k, op_name, "new_k");
  require_row_contiguous(new_v, op_name, "new_v");
  require_row_contiguous_zero_offset(slot_mapping, op_name, "slot_mapping");
  require_row_contiguous_zero_offset(k_scale, op_name, "k_scale");
  require_row_contiguous_zero_offset(v_scale, op_name, "v_scale");

  // 2. Scalar-state positivity + x_pack/dtype agreement. Run BEFORE
  //    shape arithmetic so a zero scalar can never reach divide-by-zero
  //    arithmetic below.
  if (num_kv_heads <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] num_kv_heads (" << num_kv_heads
        << ") must be > 0.";
    throw std::invalid_argument(msg.str());
  }
  if (head_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] head_size (" << head_size
        << ") must be > 0.";
    throw std::invalid_argument(msg.str());
  }
  if (block_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_size (" << block_size
        << ") must be > 0.";
    throw std::invalid_argument(msg.str());
  }
  if (x_pack <= 0 || head_size % x_pack != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] x_pack (" << x_pack
        << ") must be positive and divide head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  // A rank-4 K without `values_token_major` is the keys-token-major page.
  const bool keys_token_major = !values_token_major && k_pool.ndim() == 4;
  // Caller's `x_pack` must agree with the dtype-derived value
  // (`16 / sizeof(kv_dtype)`). A mismatch means caller and dispatcher
  // disagree on the K-pool layout — guaranteed garbage on read.
  {
    // A token-major page is K with x = head_size.
    int x_pack_expected = values_token_major || keys_token_major
        ? head_size
        : x_pack_for(kv_dtype);
    if (x_pack != x_pack_expected) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] x_pack (" << x_pack
          << ") disagrees with the dtype-derived x_pack ("
          << x_pack_expected << ") for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
  }

  // 3. Dtype-vs-kv_dtype validation. Fires BEFORE the pairwise
  //    k_pool==v_pool / new_k==new_v checks so each input slot has its
  //    own dedicated rejection path.
  {
    auto expected_cache_dtype = cache_dtype_for_kv_dtype(kv_dtype);
    if (k_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] k_pool dtype " << k_pool.dtype()
          << " disagrees with the expected cache dtype "
          << expected_cache_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
    if (v_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] v_pool dtype " << v_pool.dtype()
          << " disagrees with the expected cache dtype "
          << expected_cache_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
  }
  {
    auto expected_io_dtype = io_dtype_for_kv_dtype(kv_dtype);
    if (new_k.dtype() != expected_io_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] new_k dtype " << new_k.dtype()
          << " disagrees with the expected io dtype "
          << expected_io_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
    if (new_v.dtype() != expected_io_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] new_v dtype " << new_v.dtype()
          << " disagrees with the expected io dtype "
          << expected_io_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
  }
  // Defense-in-depth pairwise checks (tripwire for future kv_dtype additions).
  if (k_pool.dtype() != v_pool.dtype()) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool dtype " << k_pool.dtype()
        << " disagrees with v_pool dtype " << v_pool.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (new_k.dtype() != new_v.dtype()) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k dtype " << new_k.dtype()
        << " disagrees with new_v dtype " << new_v.dtype();
    throw std::invalid_argument(msg.str());
  }
  validate_kv_scales(
      k_pool,
      k_scale,
      v_scale,
      num_kv_heads,
      block_size,
      head_size,
      kv_dtype,
      /*read=*/false,
      op_name);

  // 4. Pool shape validation. K layout (vLLM):
  //      [num_blocks, num_kv_heads, head_size/x, block_size, x]
  //    V layout: [num_blocks, num_kv_heads, head_size, block_size]
  //    Token-major: both [num_blocks, num_kv_heads, block_size, head_size].
  if (values_token_major) {
    const Shape page{k_pool.ndim() > 0 ? k_pool.shape(0) : 0,
                     num_kv_heads,
                     block_size,
                     head_size};
    if (k_pool.shape() != page || v_pool.shape() != page) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] a token-major page needs K and V "
          << "[num_blocks, " << num_kv_heads << ", " << block_size << ", "
          << head_size << "]";
      throw std::invalid_argument(msg.str());
    }
  } else if (keys_token_major) {
    if (kv_dtype != KvDtype::Int8 || block_size != 32) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] a keys-token-major page is int8 "
          << "with block_size 32; got kv_dtype " << static_cast<int>(kv_dtype)
          << ", block_size " << block_size;
      throw std::invalid_argument(msg.str());
    }
    const Shape keys{k_pool.shape(0), num_kv_heads, block_size, head_size};
    const Shape values{k_pool.shape(0), num_kv_heads, head_size, block_size};
    if (k_pool.shape() != keys || v_pool.shape() != values) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] a keys-token-major page needs K "
          << "[num_blocks, " << num_kv_heads << ", " << block_size << ", "
          << head_size << "] and V [num_blocks, " << num_kv_heads << ", "
          << head_size << ", " << block_size << "]";
      throw std::invalid_argument(msg.str());
    }
  } else if (k_pool.ndim() != 5) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool must be rank 5 "
        << "[num_blocks, num_kv_heads, head_size/x, block_size, x]; got rank "
        << k_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (!values_token_major && !keys_token_major) {
    if (v_pool.ndim() != 4) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] v_pool must be rank 4 "
          << "[num_blocks, num_kv_heads, head_size, block_size]; got rank "
          << v_pool.ndim();
      throw std::invalid_argument(msg.str());
    }
    if (k_pool.shape(1) != num_kv_heads) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] k_pool.shape(1) (" << k_pool.shape(1)
          << ") disagrees with num_kv_heads (" << num_kv_heads << ")";
      throw std::invalid_argument(msg.str());
    }
    if (k_pool.shape(2) != head_size / x_pack) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] k_pool.shape(2) (" << k_pool.shape(2)
          << ") disagrees with head_size/x_pack (" << head_size / x_pack << ")";
      throw std::invalid_argument(msg.str());
    }
    if (k_pool.shape(3) != block_size) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] k_pool.shape(3) (" << k_pool.shape(3)
          << ") disagrees with block_size (" << block_size << ")";
      throw std::invalid_argument(msg.str());
    }
    if (k_pool.shape(4) != x_pack) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] k_pool.shape(4) (" << k_pool.shape(4)
          << ") disagrees with x_pack (" << x_pack << ")";
      throw std::invalid_argument(msg.str());
    }
    if (v_pool.shape(1) != num_kv_heads) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] v_pool.shape(1) (" << v_pool.shape(1)
          << ") disagrees with num_kv_heads (" << num_kv_heads << ")";
      throw std::invalid_argument(msg.str());
    }
    if (v_pool.shape(2) != head_size) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] v_pool.shape(2) (" << v_pool.shape(2)
          << ") disagrees with head_size (" << head_size << ")";
      throw std::invalid_argument(msg.str());
    }
    if (v_pool.shape(3) != block_size) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] v_pool.shape(3) (" << v_pool.shape(3)
          << ") disagrees with block_size (" << block_size << ")";
      throw std::invalid_argument(msg.str());
    }
  }
  if (k_pool.shape(0) != v_pool.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool num_blocks (" << k_pool.shape(0)
        << ") disagrees with v_pool num_blocks (" << v_pool.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }

  // 5. new_k / new_v rank + per-dim agreement.
  if (new_k.ndim() != 3) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k must be rank 3 "
        << "[num_tokens, num_kv_heads, head_size]; got rank " << new_k.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (new_v.ndim() != 3) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_v must be rank 3 "
        << "[num_tokens, num_kv_heads, head_size]; got rank " << new_v.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (new_k.shape(1) != num_kv_heads || new_v.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k/new_v shape(1) ("
        << new_k.shape(1) << "/" << new_v.shape(1)
        << ") must equal num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (new_k.shape(2) != head_size || new_v.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k/new_v shape(2) ("
        << new_k.shape(2) << "/" << new_v.shape(2)
        << ") must equal head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (new_k.shape(0) != new_v.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] new_k tokens (" << new_k.shape(0)
        << ") disagrees with new_v tokens (" << new_v.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }

  // 6. slot_mapping rank/dtype/length.
  if (slot_mapping.ndim() != 1) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] slot_mapping must be rank 1 [num_tokens]; "
        << "got rank " << slot_mapping.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (slot_mapping.dtype() != mlx::core::int64) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] slot_mapping must be int64 (kernel reads it as "
        << "`int64_t*`); got dtype " << slot_mapping.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (slot_mapping.shape(0) != new_k.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] slot_mapping length ("
        << slot_mapping.shape(0) << ") disagrees with new_k tokens ("
        << new_k.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
}

// The decode kernels of the keys-token-major page exist for the int8
// 32-token page of group 6 at D 256 with bf16 queries only
// (paged_attention.metal, KeysTokenMajor=true).
void validate_keys_token_major_read_pools(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    KvDtype kv_dtype,
    const char* op_name) {
  if (kv_dtype != KvDtype::Int8 || block_size != 32 ||
      q.dtype() != mlx::core::bfloat16 ||
      !paged::paged_prefill_keys_token_major_geometry(
          num_q_heads / num_kv_heads, head_size)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] a keys-token-major page is read "
        << "by bf16 queries as int8 with block_size 32, group "
        << paged::kPagedPrefillKeysTokenMajorGroup << " and head_size "
        << paged::kPagedPrefillKeysTokenMajorHeadDim << " only; got kv_dtype "
        << static_cast<int>(kv_dtype) << ", block_size " << block_size
        << ", group " << num_q_heads / num_kv_heads << ", head_size "
        << head_size;
    throw std::invalid_argument(msg.str());
  }
  const Shape keys{k_pool.shape(0), num_kv_heads, block_size, head_size};
  const Shape values{k_pool.shape(0), num_kv_heads, head_size, block_size};
  if (k_pool.shape() != keys || v_pool.shape() != values) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] a keys-token-major page needs K "
        << "[num_blocks, " << num_kv_heads << ", " << block_size << ", "
        << head_size << "] and V [num_blocks, " << num_kv_heads << ", "
        << head_size << ", " << block_size << "]";
    throw std::invalid_argument(msg.str());
  }
}

// Sister helper for `paged_attention(...)`. Same rationale as
// `validate_paged_kv_write_inputs`: every structural check runs on BOTH
// the factory and eval_gpu paths so compile-cache replay cannot bypass
// any of them.
//
// `paged_attention` has no `x_pack` parameter — it derives from
// `kv_dtype` via `x_pack_for(...)`. We compute it locally and validate
// it the same way as the write-side helper.
void validate_paged_attention_inputs(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& k_scale,
    const array& v_scale,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int sliding_window,
    KvDtype kv_dtype,
    const char* op_name) {
  // 1. Contiguity + offset rules.
  require_row_contiguous_head_offset(q, op_name, "q");
  require_row_contiguous_zero_offset(k_pool, op_name, "k_pool");
  require_row_contiguous_zero_offset(v_pool, op_name, "v_pool");
  require_row_contiguous_zero_offset(block_table, op_name, "block_table");
  require_row_contiguous_zero_offset(seq_lens, op_name, "seq_lens");
  require_row_contiguous_zero_offset(k_scale, op_name, "k_scale");
  require_row_contiguous_zero_offset(v_scale, op_name, "v_scale");

  // 2. The Metal kernel masks K positions older than
  // `context_len - sliding_window` when sliding_window > 0. Negative
  // values are nonsensical (the only "no mask" sentinel is 0) — reject.
  if (sliding_window < 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] sliding_window (" << sliding_window
        << ") must be >= 0 (use 0 to disable the sliding mask).";
    throw std::invalid_argument(msg.str());
  }

  // 3. Scalar-state positivity + GQA divisibility.
  if (num_kv_heads <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] num_kv_heads (" << num_kv_heads
        << ") must be > 0; the Metal kernel computes "
        << "num_queries_per_kv = num_heads / num_kv_heads, which would "
        << "divide by zero.";
    throw std::invalid_argument(msg.str());
  }
  if (num_q_heads <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] num_q_heads (" << num_q_heads
        << ") must be > 0.";
    throw std::invalid_argument(msg.str());
  }
  if (num_q_heads % num_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] GQA grouping invalid: num_q_heads="
        << num_q_heads << " must be divisible by num_kv_heads="
        << num_kv_heads << " (got remainder "
        << (num_q_heads % num_kv_heads) << "). When num_q_heads < "
        << "num_kv_heads the kernel divides by zero; when "
        << "num_q_heads % num_kv_heads != 0 later heads index past "
        << "the KV-head dimension and read out-of-pool memory.";
    throw std::invalid_argument(msg.str());
  }
  if (block_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_size (" << block_size
        << ") must be > 0; eval_gpu's bounds check divides by it "
        << "(`(s + block_size - 1) / block_size`) and the Metal kernel "
        << "uses it as a grid extent.";
    throw std::invalid_argument(msg.str());
  }
  if (head_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] head_size (" << head_size
        << ") must be > 0; the Metal kernel uses it as a grid extent "
        << "and indexing stride.";
    throw std::invalid_argument(msg.str());
  }

  // Derive x_pack from kv_dtype and verify head_size divisibility.
  // A divergence between the dtype and the structural pool check
  // (k_pool.shape(2)/(4)) would otherwise surface as a kernel
  // buffer-size mismatch.
  int x_pack_expected = x_pack_for(kv_dtype);
  if (x_pack_expected <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] dtype-derived x_pack (" << x_pack_expected
        << ") must be > 0 (kv_dtype " << static_cast<int>(kv_dtype) << ").";
    throw std::invalid_argument(msg.str());
  }
  if (head_size % x_pack_expected != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] head_size (" << head_size
        << ") must be divisible by dtype-derived x_pack ("
        << x_pack_expected << ") for kv_dtype "
        << static_cast<int>(kv_dtype);
    throw std::invalid_argument(msg.str());
  }

  // 4. Scales.
  validate_kv_scales(
      k_pool,
      k_scale,
      v_scale,
      num_kv_heads,
      block_size,
      head_size,
      kv_dtype,
      /*read=*/true,
      op_name);

  // 5. q rank + dtype + per-dim shape.
  if (q.ndim() != 3) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] q must be rank 3 "
        << "[num_seqs, num_q_heads, head_size]; got rank " << q.ndim();
    throw std::invalid_argument(msg.str());
  }
  {
    auto expected_io_dtype = io_dtype_for_kv_dtype(kv_dtype);
    if (q.dtype() != expected_io_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] q dtype " << q.dtype()
          << " disagrees with the expected io dtype "
          << expected_io_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
    auto expected_cache_dtype = cache_dtype_for_kv_dtype(kv_dtype);
    if (k_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] k_pool dtype " << k_pool.dtype()
          << " disagrees with the expected cache dtype "
          << expected_cache_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
    if (v_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] v_pool dtype " << v_pool.dtype()
          << " disagrees with the expected cache dtype "
          << expected_cache_dtype << " for kv_dtype "
          << static_cast<int>(kv_dtype);
      throw std::invalid_argument(msg.str());
    }
  }
  if (q.shape(1) != num_q_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] q.shape(1) (" << q.shape(1)
        << ") disagrees with num_q_heads (" << num_q_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (q.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] q.shape(2) (" << q.shape(2)
        << ") disagrees with head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }

  // 6. Pool rank/shape validation. A rank-4 K is the keys-token-major page.
  const bool keys_token_major = k_pool.ndim() == 4;
  if (keys_token_major) {
    validate_keys_token_major_read_pools(
        q, k_pool, v_pool, block_size, num_q_heads, num_kv_heads, head_size,
        kv_dtype, op_name);
  }
  if (!keys_token_major && k_pool.ndim() != 5) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool must be rank 5 "
        << "[num_blocks, num_kv_heads, head_size/x, block_size, x]; got rank "
        << k_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.ndim() != 4) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool must be rank 4 "
        << "[num_blocks, num_kv_heads, head_size, block_size]; got rank "
        << v_pool.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(0) != v_pool.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool num_blocks (" << k_pool.shape(0)
        << ") disagrees with v_pool num_blocks (" << v_pool.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(1) != num_kv_heads || v_pool.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool/v_pool num_kv_heads ("
        << k_pool.shape(1) << "/" << v_pool.shape(1)
        << ") disagrees with num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (!keys_token_major &&
      (k_pool.shape(3) != block_size || v_pool.shape(3) != block_size)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool/v_pool block_size ("
        << k_pool.shape(3) << "/" << v_pool.shape(3)
        << ") disagrees with block_size (" << block_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool.shape(2) (" << v_pool.shape(2)
        << ") disagrees with head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (!keys_token_major && k_pool.shape(2) != head_size / x_pack_expected) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(2) (" << k_pool.shape(2)
        << ") disagrees with head_size/x_pack ("
        << head_size / x_pack_expected
        << ") (head_size=" << head_size << ", x_pack=" << x_pack_expected
        << " for kv_dtype " << static_cast<int>(kv_dtype) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (!keys_token_major && k_pool.shape(4) != x_pack_expected) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(4) (" << k_pool.shape(4)
        << ") disagrees with dtype-derived x_pack (" << x_pack_expected
        << ") for kv_dtype " << static_cast<int>(kv_dtype);
    throw std::invalid_argument(msg.str());
  }

  // 7. block_table rank / shape / dtype.
  if (block_table.ndim() != 2) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_table must be rank 2 "
        << "[num_seqs, max_blocks_per_seq]; got rank " << block_table.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (block_table.shape(0) != q.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_table.shape(0) ("
        << block_table.shape(0) << ") disagrees with q num_seqs ("
        << q.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (block_table.dtype() != mlx::core::int32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_table dtype must be int32 (kernel "
        << "reads it as 32-bit indices); got dtype " << block_table.dtype();
    throw std::invalid_argument(msg.str());
  }

  // 8. seq_lens rank / shape / dtype.
  if (seq_lens.ndim() != 1) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] seq_lens must be rank 1 [num_seqs]; got rank "
        << seq_lens.ndim();
    throw std::invalid_argument(msg.str());
  }
  if (seq_lens.shape(0) != q.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] seq_lens.shape(0) (" << seq_lens.shape(0)
        << ") disagrees with q num_seqs (" << q.shape(0) << ")";
    throw std::invalid_argument(msg.str());
  }
  if (seq_lens.dtype() != mlx::core::int32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] seq_lens dtype must be int32 (kernel reads "
        << "it as 32-bit lengths); got dtype " << seq_lens.dtype();
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

// =============================================================================
// PagedKVWrite implementation
// =============================================================================

void PagedKVWrite::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 7) {
    throw std::runtime_error(
        "PagedKVWrite: expected 7 inputs (k_pool, v_pool, new_k, new_v, "
        "slot_mapping, k_scale, v_scale)");
  }
  if (outputs.size() != 4) {
    throw std::runtime_error(
        "PagedKVWrite: expected 4 outputs (k_pool', v_pool', k_scale', "
        "v_scale')");
  }

  const array& k_pool = inputs[0];
  const array& v_pool = inputs[1];
  const array& new_k = inputs[2];
  const array& new_v = inputs[3];
  const array& slot_mapping = inputs[4];
  const array& k_scale = inputs[5];
  const array& v_scale = inputs[6];

  // Every structural / scalar / shape / dtype / contiguity check lives
  // in `validate_paged_kv_write_inputs`, which the public
  // `paged_kv_write(...)` factory ALSO calls. Running it here closes the
  // "factory-only check bypassed by compile cache" hazard:
  // `mlx::core::compile`'s cached re-traces rebuild the primitive with
  // real inputs via `compile_replace` WITHOUT re-running the factory.
  // The helper uses the primitive's stored scalar state, so a direct
  // construction with mismatched scalars is also rejected before the
  // dispatch path touches the buffers.
  validate_paged_kv_write_inputs(
      k_pool,
      v_pool,
      new_k,
      new_v,
      slot_mapping,
      k_scale,
      v_scale,
      block_size_,
      num_kv_heads_,
      head_size_,
      x_pack_,
      kv_dtype_,
      values_token_major_,
      "PagedKVWrite::eval_gpu");

  // Output arrays semantically alias the input pools (in-place write).
  // `copy_shared_buffer` makes the output `array` point at the same
  // backing buffer + offset / strides as the input pool. We do this
  // BEFORE the dispatch so the encoder's `set_output_array` call sees
  // the right buffer (the input pool's, shared into the output).
  outputs[0].copy_shared_buffer(k_pool);
  outputs[1].copy_shared_buffer(v_pool);
  outputs[2].copy_shared_buffer(k_scale);
  outputs[3].copy_shared_buffer(v_scale);

  // Determine num_tokens from new_k's leading dimension. The kernel
  // expects shape [num_tokens, num_kv_heads, head_size]. The validator
  // already enforced rank 3.
  int num_tokens = static_cast<int>(new_k.shape(0));

  // Slot-id bounds check (runs on EVERY runtime call, including the
  // compile-cached path). Stays out of the validator because it requires
  // materialized data that tracer arrays do not have. The Metal kernel
  // does NOT bounds-check `slot_idx`; a value `>= num_blocks *
  // block_size` writes past the K/V pool. MLX evaluates all primitive
  // inputs before invoking `eval_gpu`, so `slot_mapping.data<int64_t>()`
  // is materialized and safe to read host-side.
  if (slot_mapping.shape(0) > 0) {
    const int32_t num_blocks_runtime = static_cast<int32_t>(k_pool.shape(0));
    const int64_t pool_capacity = static_cast<int64_t>(num_blocks_runtime) *
        static_cast<int64_t>(block_size_);
    const int64_t* slot_data = slot_mapping.data<int64_t>();
    int64_t max_slot = -1;
    for (size_t i = 0; i < slot_mapping.size(); ++i) {
      // Negative slot ids are sentinels for "skip this token"; the
      // kernel handles them with an early `return`. Don't include them
      // in the max — only valid (non-negative) ids must stay in range.
      if (slot_data[i] >= 0 && slot_data[i] > max_slot) {
        max_slot = slot_data[i];
      }
    }
    if (max_slot >= pool_capacity) {
      std::ostringstream msg;
      msg << "[runtime] PagedKVWrite::eval_gpu: slot_mapping max ("
          << max_slot << ") exceeds pool capacity (num_blocks="
          << num_blocks_runtime << " * block_size=" << block_size_ << " = "
          << pool_capacity
          << "). The kernel does not bounds-check slot_idx; an out-of-range "
          << "slot would write past the K/V pool. This check fires on every "
          << "runtime call (including the compile-cached path).";
      throw std::invalid_argument(msg.str());
    }
  }

  // Dispatch onto MLX's command encoder. MLX's dependency tracking
  // handles ordering against any preceding/following ops.
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  auto& compute_encoder = mlx::core::metal::get_command_encoder(s);

  mlx::core::fast::paged::dispatch_reshape_and_cache(
      compute_encoder,
      d,
      new_k,
      new_v,
      outputs[0],
      outputs[1],
      slot_mapping,
      outputs[2],
      outputs[3],
      num_tokens,
      num_kv_heads_,
      head_size_,
      block_size_,
      x_pack_,
      to_paged_dtype(kv_dtype_),
      values_token_major_);
}

std::vector<array> PagedKVWrite::vjp(
    const std::vector<array>& /*primals*/,
    const std::vector<array>& /*cotangents*/,
    const std::vector<int>& /*argnums*/,
    const std::vector<array>& /*outputs*/) {
  throw std::runtime_error("PagedKVWrite is inference-only");
}

std::vector<Shape> PagedKVWrite::output_shapes(
    const std::vector<array>& inputs) {
  if (inputs.size() != 7) {
    throw std::runtime_error(
        "PagedKVWrite::output_shapes: expected 7 inputs");
  }
  return {
      inputs[0].shape(), inputs[1].shape(), inputs[5].shape(),
      inputs[6].shape()};
}

bool PagedKVWrite::is_equivalent(const Primitive& other) const {
  const PagedKVWrite& o = static_cast<const PagedKVWrite&>(other);
  return block_size_ == o.block_size_ && num_kv_heads_ == o.num_kv_heads_ &&
      head_size_ == o.head_size_ && x_pack_ == o.x_pack_ &&
      kv_dtype_ == o.kv_dtype_ &&
      values_token_major_ == o.values_token_major_;
}

// =============================================================================
// PagedAttention implementation
// =============================================================================

void PagedAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  validate_paged_attention_plan(route_hint_, grouped_stripes_);
  if (inputs.size() != 7 && inputs.size() != 8) {
    throw std::runtime_error(
        "PagedAttention: expected 7 or 8 inputs (q, k_pool, v_pool, block_table, "
        "seq_lens, k_scale, v_scale[, score_scales])");
  }
  if (outputs.size() != 1) {
    throw std::runtime_error("PagedAttention: expected 1 output");
  }

  const array& q = inputs[0];
  const array& k_pool = inputs[1];
  const array& v_pool = inputs[2];
  const array& block_table = inputs[3];
  const array& seq_lens = inputs[4];
  const array& k_scale = inputs[5];
  const array& v_scale = inputs[6];

  array& out = outputs[0];

  // Every structural / scalar / shape / dtype / contiguity check lives
  // in `validate_paged_attention_inputs`, which the public
  // `paged_attention(...)` factory ALSO calls. Running it here closes
  // the "factory-only check bypassed by compile cache" hazard:
  // `mlx::core::compile`'s cached re-traces rebuild the primitive with
  // real inputs via `compile_replace` WITHOUT re-running the factory.
  // The helper uses the primitive's stored scalar state, so a direct
  // construction with mismatched scalars is also rejected. Run BEFORE
  // any host reads (seq_lens.data<int32_t>(), block_table.data<int32_t>()
  // etc.) and BEFORE the encoder dispatch so the throw fires before we
  // touch the buffers.
  validate_paged_attention_inputs(
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      k_scale,
      v_scale,
      block_size_,
      num_q_heads_,
      num_kv_heads_,
      head_size_,
      sliding_window_,
      kv_dtype_,
      "PagedAttention::eval_gpu");

  const array* score_scales = inputs.size() == 8 ? &inputs[7] : nullptr;
  if (score_scales != nullptr) {
    validate_score_scales(
        *score_scales, q.shape(0), num_q_heads_, kv_dtype_, "PagedAttention::eval_gpu");
  }

  uint32_t num_seqs = static_cast<uint32_t>(q.shape(0));
  uint32_t max_blocks_per_seq = static_cast<uint32_t>(block_table.shape(1));

  // The validator already enforced seq_lens/block_table dtype == int32.
  // MLX evaluates all primitive inputs before invoking `eval_gpu`, so
  // `seq_lens.data<int32_t>()` is materialized and safe to read host-side.
  // Calling `mlx::core::eval(...)` here would recurse into the scheduler
  // and deadlock.
  const int32_t* seq_lens_data = seq_lens.data<int32_t>();

  // Per-runtime bounds check on `seq_lens` and `block_table` contents
  // (runs on EVERY runtime call, including the compile-cached path).
  //
  // The Metal kernel does NOT bounds-check either input. For each
  // sequence the kernel:
  //   1. reads `context_lens[seq_idx]` as `uint32_t` (so a negative
  //      int32 sneaks in as a huge unsigned context length),
  //   2. derives `num_context_blocks = ceil(context_len / block_size)`,
  //      and reads `block_tables[seq_idx * max_blocks_per_seq + j]`
  //      for each j in [0, num_context_blocks),
  //   3. uses that value directly as `physical_block_number *
  //      kv_block_stride` for K/V pool reads.
  //
  // Without per-call validation, a too-large `seq_lens[i]` reads past
  // that row's block-table region, and a negative or `>= num_blocks`
  // entry in `block_table` addresses outside the K/V pool. Like the
  // slot-mapping check above, this lives in `eval_gpu` because the
  // factory's structural validation only covers shape/dtype and the
  // compile-cached path bypasses the factory.
  //
  // Cost: O(num_seqs * max_blocks_per_seq) host-side reads per call.
  //
  // Run BEFORE the max-context computation below so a malformed input
  // surfaces the descriptive `std::invalid_argument` error instead of
  // the coarser "max_context_len > 0" runtime_error.
  if (num_seqs > 0 && max_blocks_per_seq > 0) {
    const int32_t num_blocks = static_cast<int32_t>(k_pool.shape(0));
    const int64_t max_seq_len_bound =
        static_cast<int64_t>(max_blocks_per_seq) *
        static_cast<int64_t>(block_size_);
    const int32_t* block_table_data = block_table.data<int32_t>();
    for (uint32_t i = 0; i < num_seqs; ++i) {
      const int32_t s = seq_lens_data[i];
      if (s < 0 || static_cast<int64_t>(s) > max_seq_len_bound) {
        std::ostringstream msg;
        msg << "[runtime] PagedAttention::eval_gpu: seq_lens[" << i << "] ("
            << s << ") out of range [0, max_blocks_per_seq * block_size = "
            << max_blocks_per_seq << " * " << block_size_ << " = "
            << max_seq_len_bound
            << "]. The kernel reads context_lens[seq_idx] as uint32_t and "
            << "derives num_context_blocks from it; an out-of-range value "
            << "either reads past the row's block-table region (positive "
            << "overflow) or is interpreted as a huge unsigned value "
            << "(negative). This check fires on every runtime call "
            << "(including the compile-cached path).";
        throw std::invalid_argument(msg.str());
      }
      const int64_t num_used_blocks =
          ceil_div_nonnegative_i32(s, block_size_);
      const size_t row_offset =
          static_cast<size_t>(i) * static_cast<size_t>(max_blocks_per_seq);
      for (int64_t j = 0; j < num_used_blocks; ++j) {
        const int32_t blk = block_table_data[row_offset + static_cast<size_t>(j)];
        if (blk < 0 || blk >= num_blocks) {
          std::ostringstream msg;
          msg << "[runtime] PagedAttention::eval_gpu: block_table[" << i
              << ", " << j << "] (" << blk
              << ") out of range [0, num_blocks = " << num_blocks
              << "). The kernel uses block_table entries directly as "
              << "physical_block_number * kv_block_stride for K/V reads; "
              << "a value outside [0, num_blocks) addresses arbitrary GPU "
              << "memory. This check fires on every runtime call "
              << "(including the compile-cached path).";
          throw std::invalid_argument(msg.str());
        }
      }
    }
  }

  // Determine max_context_len from seq_lens (max element). This is
  // the value that drives the V1/V2 branch. By the time we get here
  // every entry has already been validated as `>= 0` above.
  int32_t max_context_len = 0;
  for (size_t i = 0; i < seq_lens.size(); ++i) {
    if (seq_lens_data[i] > max_context_len) {
      max_context_len = seq_lens_data[i];
    }
  }
  if (max_context_len <= 0) {
    throw std::runtime_error(
        "PagedAttention: max_context_len from seq_lens must be > 0");
  }
  const int64_t live_blocks = distinct_live_blocks(
      block_table.data<int32_t>(),
      seq_lens_data,
      num_seqs,
      max_blocks_per_seq,
      block_size_,
      static_cast<int32_t>(k_pool.shape(0)));

  // Allocate the output buffer via MLX's allocator BEFORE dispatch —
  // the encoder's `set_output_array` reads `out.buffer().ptr()` so
  // the buffer must exist by the time the dispatch runs.
  out.set_data(allocator::malloc(out.nbytes()));

  // Dispatch onto MLX's command encoder. MLX's dependency tracking
  // handles ordering against any preceding/following ops, so callers do
  // not need to `eval()` ancestors before invoking `paged_attention` nor
  // `eval()` the output before reading.
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  auto& compute_encoder = mlx::core::metal::get_command_encoder(s);

  mlx::core::fast::paged::dispatch_paged_attention_auto(
      compute_encoder,
      d,
      s,
      out,
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      k_scale,
      v_scale,
      static_cast<int>(num_seqs),
      num_q_heads_,
      num_kv_heads_,
      head_size_,
      block_size_,
      max_context_len,
      static_cast<int>(max_blocks_per_seq),
      live_blocks,
      scale_,
      softcap_,
      sliding_window_,
      to_paged_dtype(kv_dtype_),
      static_cast<mlx::core::fast::paged::PagedAttentionRouteHint>(
          route_hint_),
      grouped_stripes_,
      score_scales);
}

std::vector<array> PagedAttention::vjp(
    const std::vector<array>& /*primals*/,
    const std::vector<array>& /*cotangents*/,
    const std::vector<int>& /*argnums*/,
    const std::vector<array>& /*outputs*/) {
  throw std::runtime_error("PagedAttention is inference-only");
}

std::vector<Shape> PagedAttention::output_shapes(
    const std::vector<array>& inputs) {
  if (inputs.empty()) {
    throw std::runtime_error("PagedAttention::output_shapes: empty inputs");
  }
  const auto& q = inputs[0];
  if (q.ndim() != 3) {
    throw std::runtime_error(
        "PagedAttention::output_shapes: q must be rank 3");
  }
  // Spec: output shape is {q_num_tokens, num_q_heads, head_size} from
  // scalar state. We DO NOT echo q.shape() — if q's trailing dims
  // disagree with our scalar state, we'd allocate a buffer of the
  // wrong size, which the kernel would then under- or over-write.
  // Validation against q.shape() lives in `eval_gpu` and the public
  // `paged_attention` factory; this method just reports the
  // shape MLX should allocate.
  return {Shape{q.shape(0), num_q_heads_, head_size_}};
}

bool PagedAttention::is_equivalent(const Primitive& other) const {
  const PagedAttention& o = static_cast<const PagedAttention&>(other);
  return scale_ == o.scale_ && softcap_ == o.softcap_ &&
      block_size_ == o.block_size_ && num_q_heads_ == o.num_q_heads_ &&
      num_kv_heads_ == o.num_kv_heads_ && head_size_ == o.head_size_ &&
      sliding_window_ == o.sliding_window_ && kv_dtype_ == o.kv_dtype_ &&
      route_hint_ == o.route_hint_ && grouped_stripes_ == o.grouped_stripes_;
}

// =============================================================================
// Public free functions
// =============================================================================

std::array<array, 4> paged_kv_write(
    const array& k_pool,
    const array& v_pool,
    const array& new_k,
    const array& new_v,
    const array& slot_mapping,
    const array& k_scale,
    const array& v_scale,
    int block_size,
    int num_kv_heads,
    int head_size,
    int x_pack,
    KvDtype kv_dtype,
    StreamOrDevice s_,
    bool values_token_major) {
  auto s = to_stream(s_);

  // No pure-MLX fallback exists (the kernel is the implementation). The
  // fallback is only invoked by VJP/JVP transformations, which we throw
  // on. Provide a stub that raises so unintended fallback paths surface
  // immediately.
  auto fallback = [](std::vector<array> /*inputs*/) -> std::vector<array> {
    throw std::runtime_error(
        "paged_kv_write has no fallback implementation (inference-only)");
  };

  // Structural / scalar / shape / dtype / contiguity checks live in the
  // shared helper that `PagedKVWrite::eval_gpu` ALSO calls — the single
  // source of truth for both paths (compile-cache replay bypasses the
  // factory).
  validate_paged_kv_write_inputs(
      k_pool,
      v_pool,
      new_k,
      new_v,
      slot_mapping,
      k_scale,
      v_scale,
      block_size,
      num_kv_heads,
      head_size,
      x_pack,
      kv_dtype,
      values_token_major,
      "paged_kv_write");

  // Slot-id range check (factory-only).
  //
  // The Metal kernel does NOT bounds-check `slot_idx`; a value
  // `>= num_blocks * block_size` writes past the K/V pool.
  //   - When called with concrete data (most production callers), the
  //     factory evals `max(slot_mapping)` here and throws on overflow.
  //   - When called inside `mlx::core::compile`'s trace
  //     (`mlx::core::detail::in_tracing()` is true), we SKIP the eval
  //     because tracer arrays have no backing data and `eval()` would
  //     fail. The mirrored runtime check inside `eval_gpu` covers the
  //     compile-cached path instead.
  //
  // Rust supplies materialized shared-memory slot metadata for both decode
  // and prefill. Scan it directly instead of launching a max reduction and
  // waiting once per layer. Lazy arrays retain the reduction fallback, and
  // tracing still skips this factory-only check; the
  // runtime scan in PagedKVWrite::eval_gpu remains authoritative for every
  // evaluated call, including compile-cache replay.
  if (!mlx::core::detail::in_tracing() && slot_mapping.shape(0) > 0) {
    int64_t max_slot_v;
    if (slot_mapping.is_available()) {
      const auto* slots = slot_mapping.data<int64_t>();
      const auto stride = slot_mapping.strides()[0];
      max_slot_v = slots[0];
      for (size_t i = 1; i < slot_mapping.size(); ++i) {
        max_slot_v = std::max(max_slot_v, slots[i * stride]);
      }
    } else {
      array max_slot = mlx::core::max(slot_mapping);
      mlx::core::eval(max_slot);
      max_slot_v = max_slot.item<int64_t>();
    }
    int64_t pool_capacity =
        static_cast<int64_t>(k_pool.shape(0)) * static_cast<int64_t>(block_size);
    if (max_slot_v >= pool_capacity) {
      // Runtime data-dependent guard: `[runtime]` prefix matches the
      // `[runtime] PagedKVWrite::eval_gpu` marker used by the same
      // bounds check inside `PagedKVWrite::eval_gpu`. Tagging both
      // throw sites with the same `[runtime]` prefix keeps slot_mapping
      // bounds errors uniformly distinguishable from `[validator]`
      // structural rejections, regardless of which path (factory vs.
      // compile-cached eval_gpu) caught the bad data.
      std::ostringstream msg;
      msg << "[runtime] [paged_kv_write] slot_mapping max (" << max_slot_v
          << ") exceeds pool capacity (num_blocks=" << k_pool.shape(0)
          << " * block_size=" << block_size << " = " << pool_capacity << ")";
      throw std::invalid_argument(msg.str());
    }
  }

  std::vector<array> inputs = {
      k_pool, v_pool, new_k, new_v, slot_mapping, k_scale, v_scale};

  auto primitive = std::make_shared<PagedKVWrite>(
      s,
      std::move(fallback),
      block_size,
      num_kv_heads,
      head_size,
      x_pack,
      kv_dtype,
      values_token_major);

  auto results = array::make_arrays(
      {k_pool.shape(), v_pool.shape(), k_scale.shape(), v_scale.shape()},
      {k_pool.dtype(), v_pool.dtype(), k_scale.dtype(), v_scale.dtype()},
      primitive,
      inputs);

  return {
      std::move(results[0]),
      std::move(results[1]),
      std::move(results[2]),
      std::move(results[3])};
}

array paged_attention(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& k_scale,
    const array& v_scale,
    float scale,
    float softcap,
    int sliding_window,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    KvDtype kv_dtype,
    StreamOrDevice s_) {
  return paged_attention_with_route_hint(
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      k_scale,
      v_scale,
      scale,
      softcap,
      sliding_window,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      kv_dtype,
      /*route_hint=*/0,
      s_);
}

array paged_attention_with_route_hint(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& k_scale,
    const array& v_scale,
    float scale,
    float softcap,
    int sliding_window,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    KvDtype kv_dtype,
    uint8_t route_hint,
    StreamOrDevice s_,
    uint32_t grouped_stripes,
    const std::optional<array>& score_scales) {
  validate_paged_attention_plan(route_hint, grouped_stripes);
  auto s = to_stream(s_);

  auto fallback = [](std::vector<array> /*inputs*/) -> std::vector<array> {
    throw std::runtime_error(
        "paged_attention has no fallback implementation (inference-only)");
  };

  // Structural / scalar / shape / dtype / contiguity checks live in the
  // shared helper that `PagedAttention::eval_gpu` ALSO calls — the
  // single source of truth for both paths (compile-cache replay bypasses
  // the factory).
  validate_paged_attention_inputs(
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      k_scale,
      v_scale,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window,
      kv_dtype,
      "paged_attention");

  // Output shape and dtype
  auto out_dtype = io_dtype_for_kv_dtype(kv_dtype);
  // Spec: output shape is {q.shape(0), num_q_heads, head_size} from
  // scalar state. Even though we just verified q.shape(1)/(2) agree
  // with state, we use state explicitly so this matches
  // PagedAttention::output_shapes — they MUST report the same shape
  // or MLX will allocate a buffer of the wrong size during compile
  // replay.
  Shape out_shape = {q.shape(0), num_q_heads, head_size};

  std::vector<array> inputs = {
      q, k_pool, v_pool, block_table, seq_lens, k_scale, v_scale};
  if (score_scales) {
    validate_score_scales(*score_scales, q.shape(0), num_q_heads, kv_dtype, "paged_attention");
    inputs.push_back(*score_scales);
  }

  auto primitive = std::make_shared<PagedAttention>(
      s,
      std::move(fallback),
      scale,
      softcap,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window,
      kv_dtype,
      route_hint,
      grouped_stripes);

  return array(std::move(out_shape), out_dtype, primitive, std::move(inputs));
}

namespace {

void validate_paged_attention_varlen_inputs(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& cu_seqlens_q,
    const array& k_scale,
    const array& v_scale,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int sliding_window,
    KvDtype kv_dtype,
    const char* op_name) {
  require_row_contiguous_head_offset(q, op_name, "q");
  require_row_contiguous_zero_offset(k_pool, op_name, "k_pool");
  require_row_contiguous_zero_offset(v_pool, op_name, "v_pool");
  require_row_contiguous_zero_offset(block_table, op_name, "block_table");
  require_row_contiguous_zero_offset(seq_lens, op_name, "seq_lens");
  require_row_contiguous_zero_offset(cu_seqlens_q, op_name, "cu_seqlens_q");
  require_row_contiguous_zero_offset(k_scale, op_name, "k_scale");
  require_row_contiguous_zero_offset(v_scale, op_name, "v_scale");

  if (sliding_window < 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] sliding_window ("
        << sliding_window << ") must be >= 0";
    throw std::invalid_argument(msg.str());
  }
  if (num_kv_heads <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] num_kv_heads (" << num_kv_heads
        << ") must be > 0";
    throw std::invalid_argument(msg.str());
  }
  if (num_q_heads <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] num_q_heads (" << num_q_heads
        << ") must be > 0";
    throw std::invalid_argument(msg.str());
  }
  if (num_q_heads % num_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] GQA grouping invalid: num_q_heads="
        << num_q_heads << " must be divisible by num_kv_heads="
        << num_kv_heads;
    throw std::invalid_argument(msg.str());
  }
  if (block_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_size (" << block_size
        << ") must be > 0";
    throw std::invalid_argument(msg.str());
  }
  if (head_size <= 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] head_size (" << head_size
        << ") must be > 0";
    throw std::invalid_argument(msg.str());
  }

  int x_pack_expected = x_pack_for(kv_dtype);
  if (head_size % x_pack_expected != 0) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] head_size (" << head_size
        << ") must be divisible by dtype-derived x_pack ("
        << x_pack_expected << ")";
    throw std::invalid_argument(msg.str());
  }

  validate_kv_scales(
      k_pool,
      k_scale,
      v_scale,
      num_kv_heads,
      block_size,
      head_size,
      kv_dtype,
      /*read=*/true,
      op_name);

  if (q.ndim() != 3) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] q must be rank 3 "
        << "[total_queries, num_q_heads, head_size]; got rank " << q.ndim();
    throw std::invalid_argument(msg.str());
  }
  {
    auto expected_io_dtype = io_dtype_for_kv_dtype(kv_dtype);
    if (q.dtype() != expected_io_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] q dtype " << q.dtype()
          << " disagrees with the expected io dtype "
          << expected_io_dtype;
      throw std::invalid_argument(msg.str());
    }
    auto expected_cache_dtype = cache_dtype_for_kv_dtype(kv_dtype);
    if (k_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] k_pool dtype "
          << k_pool.dtype() << " disagrees with the expected cache dtype "
          << expected_cache_dtype;
      throw std::invalid_argument(msg.str());
    }
    if (v_pool.dtype() != expected_cache_dtype) {
      std::ostringstream msg;
      msg << "[validator] [" << op_name << "] v_pool dtype "
          << v_pool.dtype() << " disagrees with the expected cache dtype "
          << expected_cache_dtype;
      throw std::invalid_argument(msg.str());
    }
  }
  if (q.shape(1) != num_q_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] q.shape(1) (" << q.shape(1)
        << ") disagrees with num_q_heads (" << num_q_heads << ")";
    throw std::invalid_argument(msg.str());
  }
  if (q.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] q.shape(2) (" << q.shape(2)
        << ") disagrees with head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }

  if (k_pool.ndim() != 5) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool must be rank 5";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.ndim() != 4) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool must be rank 4";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(0) != v_pool.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] k_pool / v_pool num_blocks mismatch";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(1) != num_kv_heads || v_pool.shape(1) != num_kv_heads) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] pool num_kv_heads mismatch";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(3) != block_size || v_pool.shape(3) != block_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] pool block_size mismatch";
    throw std::invalid_argument(msg.str());
  }
  if (v_pool.shape(2) != head_size) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] v_pool.shape(2) ("
        << v_pool.shape(2) << ") != head_size (" << head_size << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(2) != head_size / x_pack_expected) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(2) ("
        << k_pool.shape(2) << ") != head_size/x_pack ("
        << head_size / x_pack_expected << ")";
    throw std::invalid_argument(msg.str());
  }
  if (k_pool.shape(4) != x_pack_expected) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] k_pool.shape(4) ("
        << k_pool.shape(4) << ") != x_pack (" << x_pack_expected << ")";
    throw std::invalid_argument(msg.str());
  }

  if (block_table.ndim() != 2) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] block_table must be rank 2";
    throw std::invalid_argument(msg.str());
  }
  if (block_table.dtype() != mlx::core::int32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] block_table dtype must be int32";
    throw std::invalid_argument(msg.str());
  }
  if (seq_lens.ndim() != 1) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name << "] seq_lens must be rank 1";
    throw std::invalid_argument(msg.str());
  }
  if (seq_lens.dtype() != mlx::core::int32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] seq_lens dtype must be int32";
    throw std::invalid_argument(msg.str());
  }
  if (block_table.shape(0) != seq_lens.shape(0)) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] block_table.shape(0) != seq_lens.shape(0)";
    throw std::invalid_argument(msg.str());
  }

  if (cu_seqlens_q.ndim() != 1) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] cu_seqlens_q must be rank 1";
    throw std::invalid_argument(msg.str());
  }
  if (cu_seqlens_q.dtype() != mlx::core::int32) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] cu_seqlens_q dtype must be int32";
    throw std::invalid_argument(msg.str());
  }
  if (cu_seqlens_q.shape(0) != seq_lens.shape(0) + 1) {
    std::ostringstream msg;
    msg << "[validator] [" << op_name
        << "] cu_seqlens_q.shape(0) (" << cu_seqlens_q.shape(0)
        << ") must equal num_seqs + 1 (" << seq_lens.shape(0) + 1 << ")";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

void PagedAttentionVarlen::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 8 && inputs.size() != 9) {
    throw std::runtime_error(
        "PagedAttentionVarlen: expected 8 or 9 inputs (q, k_pool, v_pool, "
        "block_table, seq_lens, cu_seqlens_q, k_scale, v_scale[, score_scales])");
  }
  if (outputs.size() != 1) {
    throw std::runtime_error("PagedAttentionVarlen: expected 1 output");
  }

  const array& q = inputs[0];
  const array& k_pool = inputs[1];
  const array& v_pool = inputs[2];
  const array& block_table = inputs[3];
  const array& seq_lens = inputs[4];
  const array& cu_seqlens_q = inputs[5];
  const array& k_scale = inputs[6];
  const array& v_scale = inputs[7];

  array& out = outputs[0];

  validate_paged_attention_varlen_inputs(
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      cu_seqlens_q,
      k_scale,
      v_scale,
      block_size_,
      num_q_heads_,
      num_kv_heads_,
      head_size_,
      sliding_window_,
      kv_dtype_,
      "PagedAttentionVarlen::eval_gpu");

  const array* score_scales = inputs.size() == 9 ? &inputs[8] : nullptr;
  if (score_scales != nullptr) {
    validate_score_scales(
        *score_scales,
        q.shape(0),
        num_q_heads_,
        kv_dtype_,
        "PagedAttentionVarlen::eval_gpu");
  }

  uint32_t num_seqs = static_cast<uint32_t>(seq_lens.shape(0));
  uint32_t total_queries = static_cast<uint32_t>(q.shape(0));
  uint32_t max_blocks_per_seq = static_cast<uint32_t>(block_table.shape(1));

  // Runtime data-dependent guards on seq_lens / block_table / cu_seqlens_q.
  // The metal kernel reads `context_lens[seq_idx]` as `uint32_t`, walks
  // `block_table[seq_idx * max_blocks_per_seq + j]` for each block, and
  // does `cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]` to discover
  // each token's source sequence. None of those reads are bounds-checked
  // GPU-side.
  const int32_t* seq_lens_data = seq_lens.data<int32_t>();
  const int32_t* block_table_data = block_table.data<int32_t>();
  const int32_t* cu_seqlens_q_data = cu_seqlens_q.data<int32_t>();

  if (num_seqs > 0) {
    const int32_t num_blocks = static_cast<int32_t>(k_pool.shape(0));
    const int64_t max_seq_len_bound =
        static_cast<int64_t>(max_blocks_per_seq) *
        static_cast<int64_t>(block_size_);
    for (uint32_t i = 0; i < num_seqs; ++i) {
      const int32_t s = seq_lens_data[i];
      if (s < 0 || static_cast<int64_t>(s) > max_seq_len_bound) {
        std::ostringstream msg;
        msg << "[runtime] PagedAttentionVarlen::eval_gpu: seq_lens[" << i
            << "] (" << s << ") out of range [0, " << max_seq_len_bound
            << "]";
        throw std::invalid_argument(msg.str());
      }
      const int64_t num_used_blocks =
          ceil_div_nonnegative_i32(s, block_size_);
      const size_t row_offset =
          static_cast<size_t>(i) * static_cast<size_t>(max_blocks_per_seq);
      for (int64_t j = 0; j < num_used_blocks; ++j) {
        const int32_t blk = block_table_data[row_offset + static_cast<size_t>(j)];
        if (blk < 0 || blk >= num_blocks) {
          std::ostringstream msg;
          msg << "[runtime] PagedAttentionVarlen::eval_gpu: block_table[" << i
              << ", " << j << "] (" << blk << ") out of range [0, "
              << num_blocks << ")";
          throw std::invalid_argument(msg.str());
        }
      }
    }
  }

  if (cu_seqlens_q_data[0] != 0) {
    std::ostringstream msg;
    msg << "[runtime] PagedAttentionVarlen::eval_gpu: cu_seqlens_q[0] ("
        << cu_seqlens_q_data[0] << ") must be 0";
    throw std::invalid_argument(msg.str());
  }
  if (static_cast<uint32_t>(cu_seqlens_q_data[num_seqs]) != total_queries) {
    std::ostringstream msg;
    msg << "[runtime] PagedAttentionVarlen::eval_gpu: cu_seqlens_q["
        << num_seqs << "] (" << cu_seqlens_q_data[num_seqs]
        << ") must equal total_queries (" << total_queries << ")";
    throw std::invalid_argument(msg.str());
  }
  for (uint32_t i = 0; i < num_seqs; ++i) {
    if (cu_seqlens_q_data[i + 1] < cu_seqlens_q_data[i]) {
      std::ostringstream msg;
      msg << "[runtime] PagedAttentionVarlen::eval_gpu: cu_seqlens_q not "
          << "monotone non-decreasing at index " << i;
      throw std::invalid_argument(msg.str());
    }
    const int32_t query_span =
        cu_seqlens_q_data[i + 1] - cu_seqlens_q_data[i];
    if (query_span > seq_lens_data[i]) {
      std::ostringstream msg;
      msg << "[runtime] PagedAttentionVarlen::eval_gpu: query span for "
          << "sequence " << i << " (" << query_span
          << ") must not exceed seq_lens[" << i << "] ("
          << seq_lens_data[i] << ")";
      throw std::invalid_argument(msg.str());
    }
  }

  int32_t max_context_len = 0;
  for (uint32_t i = 0; i < num_seqs; ++i) {
    if (seq_lens_data[i] > max_context_len) {
      max_context_len = seq_lens_data[i];
    }
  }
  if (max_context_len <= 0) {
    throw std::runtime_error(
        "PagedAttentionVarlen: max_context_len from seq_lens must be > 0");
  }
  const int64_t live_blocks = distinct_live_blocks(
      block_table_data,
      seq_lens_data,
      num_seqs,
      max_blocks_per_seq,
      block_size_,
      static_cast<int32_t>(k_pool.shape(0)));

  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  auto& compute_encoder = mlx::core::metal::get_command_encoder(s);

  mlx::core::fast::paged::dispatch_paged_attention_varlen_auto(
      compute_encoder,
      d,
      s,
      out,
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      cu_seqlens_q,
      k_scale,
      v_scale,
      static_cast<int>(num_seqs),
      static_cast<int>(total_queries),
      num_q_heads_,
      num_kv_heads_,
      head_size_,
      block_size_,
      max_context_len,
      static_cast<int>(max_blocks_per_seq),
      live_blocks,
      scale_,
      softcap_,
      sliding_window_,
      to_paged_dtype(kv_dtype_),
      score_scales);
}

std::vector<array> PagedAttentionVarlen::vjp(
    const std::vector<array>& /*primals*/,
    const std::vector<array>& /*cotangents*/,
    const std::vector<int>& /*argnums*/,
    const std::vector<array>& /*outputs*/) {
  throw std::runtime_error("PagedAttentionVarlen is inference-only");
}

std::vector<Shape> PagedAttentionVarlen::output_shapes(
    const std::vector<array>& inputs) {
  if (inputs.empty()) {
    throw std::runtime_error(
        "PagedAttentionVarlen::output_shapes: empty inputs");
  }
  const auto& q = inputs[0];
  if (q.ndim() != 3) {
    throw std::runtime_error(
        "PagedAttentionVarlen::output_shapes: q must be rank 3");
  }
  return {Shape{q.shape(0), num_q_heads_, head_size_}};
}

bool PagedAttentionVarlen::is_equivalent(const Primitive& other) const {
  const PagedAttentionVarlen& o =
      static_cast<const PagedAttentionVarlen&>(other);
  return scale_ == o.scale_ && softcap_ == o.softcap_ &&
      block_size_ == o.block_size_ && num_q_heads_ == o.num_q_heads_ &&
      num_kv_heads_ == o.num_kv_heads_ && head_size_ == o.head_size_ &&
      sliding_window_ == o.sliding_window_ && kv_dtype_ == o.kv_dtype_;
}

array paged_attention_varlen(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& cu_seqlens_q,
    const array& k_scale,
    const array& v_scale,
    float scale,
    float softcap,
    int sliding_window,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    KvDtype kv_dtype,
    StreamOrDevice s_,
    const std::optional<array>& score_scales) {
  auto s = to_stream(s_);

  auto fallback = [](std::vector<array> /*inputs*/) -> std::vector<array> {
    throw std::runtime_error(
        "paged_attention_varlen has no fallback implementation (inference-only)");
  };

  validate_paged_attention_varlen_inputs(
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      cu_seqlens_q,
      k_scale,
      v_scale,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window,
      kv_dtype,
      "paged_attention_varlen");

  auto out_dtype = io_dtype_for_kv_dtype(kv_dtype);
  Shape out_shape = {q.shape(0), num_q_heads, head_size};

  std::vector<array> inputs = {
      q,
      k_pool,
      v_pool,
      block_table,
      seq_lens,
      cu_seqlens_q,
      k_scale,
      v_scale};
  if (score_scales) {
    validate_score_scales(
        *score_scales, q.shape(0), num_q_heads, kv_dtype, "paged_attention_varlen");
    inputs.push_back(*score_scales);
  }

  auto primitive = std::make_shared<PagedAttentionVarlen>(
      s,
      std::move(fallback),
      scale,
      softcap,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window,
      kv_dtype);

  return array(std::move(out_shape), out_dtype, primitive, std::move(inputs));
}

namespace {

[[noreturn]] void prefill_reject(const char* op_name, const std::string& what) {
  std::ostringstream msg;
  msg << "[validator] [" << op_name << "] " << what;
  throw std::invalid_argument(msg.str());
}

int paged_prefill_stride(int rows) {
  return (rows + 31) / 32 * 32;
}

// Every structural check of `PagedPrefillAttention`; the factory and
// eval_gpu both run it (a compile re-trace rebuilds the primitive without
// the factory).
void validate_paged_prefill_inputs(
    const array& q_staged,
    const array& k_pool,
    const array& v_pool,
    const array& k_scale,
    const array& v_scale,
    const array& block_table,
    const array& workspace,
    int committed_tokens,
    int rows,
    int splits,
    int max_splits,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    KvDtype kv_dtype,
    bool int8_queries,
    const char* op_name) {
  namespace paged = mlx::core::fast::paged;
  require_row_contiguous_head_offset(q_staged, op_name, "q_staged");
  require_row_contiguous_zero_offset(k_pool, op_name, "k_pool");
  require_row_contiguous_zero_offset(v_pool, op_name, "v_pool");
  require_row_contiguous_zero_offset(k_scale, op_name, "k_scale");
  require_row_contiguous_zero_offset(v_scale, op_name, "v_scale");
  require_row_contiguous_zero_offset(block_table, op_name, "block_table");
  require_row_contiguous_zero_offset(workspace, op_name, "workspace");
  if (kv_dtype != KvDtype::Bf16 && kv_dtype != KvDtype::Int8) {
    prefill_reject(op_name, "kv_dtype must be bf16 or int8");
  }
  if (int8_queries && kv_dtype != KvDtype::Int8) {
    prefill_reject(op_name, "INT8 queries take INT8 pages");
  }
  if (num_kv_heads <= 0 || num_q_heads <= 0 || num_q_heads % num_kv_heads != 0) {
    prefill_reject(
        op_name,
        "query heads (" + std::to_string(num_q_heads) +
            ") must be a positive multiple of KV heads (" +
            std::to_string(num_kv_heads) + ")");
  }
  const int group = num_q_heads / num_kv_heads;
  if (!paged::paged_prefill_attention_geometry(group, head_size)) {
    prefill_reject(
        op_name,
        "no paged prefill attention kernel for group " + std::to_string(group) +
            ", head_size " + std::to_string(head_size));
  }
  if (block_size != 16 && block_size != 32) {
    prefill_reject(op_name, "block_size must be 16 or 32; got " + std::to_string(block_size));
  }
  if (max_splits != paged::kPagedPrefillMaxSplits &&
      max_splits != paged::kPagedVerifyMaxSplits) {
    prefill_reject(op_name, "max_splits must be 32 (prefill) or 128 (verify)");
  }
  const int max_rows = max_splits == paged::kPagedVerifyMaxSplits
      ? paged::kPagedVerifyRows
      : paged::kPagedPrefillMaxRows;
  if (rows <= 0 || rows > max_rows) {
    prefill_reject(
        op_name,
        "rows (" + std::to_string(rows) + ") must be in 1.." + std::to_string(max_rows));
  }
  if (splits <= 0 || splits > max_splits) {
    prefill_reject(
        op_name,
        "splits (" + std::to_string(splits) + ") must be in 1.." +
            std::to_string(max_splits));
  }
  if (committed_tokens < 0) {
    prefill_reject(op_name, "committed_tokens must be >= 0");
  }
  const Shape staged{num_kv_heads, paged_prefill_stride(rows), group, head_size};
  if (q_staged.dtype() != mlx::core::bfloat16 || q_staged.shape() != staged) {
    prefill_reject(op_name, "q_staged must be bf16 [kv heads, stride, group, head_size]");
  }
  const int x = x_pack_for(kv_dtype);
  const auto cache_dtype = cache_dtype_for_kv_dtype(kv_dtype);
  if (k_pool.dtype() != cache_dtype || v_pool.dtype() != cache_dtype) {
    prefill_reject(op_name, "k_pool / v_pool dtype must match kv_dtype");
  }
  // A rank-4 K is the keys-token-major page (K [blocks, kv heads, 32,
  // head_size], V dimension-major); the dispatch routes by this rank.
  const bool keys_token_major = k_pool.ndim() == 4;
  if ((!keys_token_major && k_pool.ndim() != 5) || v_pool.ndim() != 4) {
    prefill_reject(
        op_name, "k_pool must be rank 5 (rank 4: keys token-major) and v_pool rank 4");
  }
  const int num_blocks = k_pool.shape(0);
  if (keys_token_major) {
    if (kv_dtype != KvDtype::Int8 || block_size != 32 ||
        !paged::paged_prefill_keys_token_major_geometry(group, head_size)) {
      prefill_reject(
          op_name,
          "a keys-token-major page is read by the int8 block-32 tile of group " +
              std::to_string(paged::kPagedPrefillKeysTokenMajorGroup) + ", head_size " +
              std::to_string(paged::kPagedPrefillKeysTokenMajorHeadDim) + " only");
    }
    if (k_pool.shape() != Shape{num_blocks, num_kv_heads, block_size, head_size} ||
        v_pool.shape() != Shape{num_blocks, num_kv_heads, head_size, block_size}) {
      prefill_reject(
          op_name,
          "keys-token-major planes do not match K [blocks, kv heads, block_size, head_size] "
          "and V [blocks, kv heads, head_size, block_size]");
    }
  } else if (
      k_pool.shape() != Shape{num_blocks, num_kv_heads, head_size / x, block_size, x} ||
      v_pool.shape() != Shape{num_blocks, num_kv_heads, head_size, block_size}) {
    prefill_reject(op_name, "pool planes do not match [blocks, kv heads, head_size, block_size]");
  }
  validate_kv_scales(
      k_pool, k_scale, v_scale, num_kv_heads, block_size, head_size, kv_dtype, true, op_name);
  if (block_table.dtype() != mlx::core::int32) {
    prefill_reject(op_name, "block_table must be int32");
  }
  const int64_t needed =
      (static_cast<int64_t>(committed_tokens) + rows + block_size - 1) / block_size;
  if (static_cast<int64_t>(block_table.size()) < needed) {
    prefill_reject(
        op_name,
        "block_table holds " + std::to_string(block_table.size()) + " entries; " +
            std::to_string(needed) + " cover the history and the rows");
  }
  if (workspace.dtype() != mlx::core::float32 ||
      static_cast<int64_t>(workspace.size()) <
          paged::paged_prefill_workspace_floats(rows, splits, num_q_heads, head_size)) {
    prefill_reject(op_name, "workspace must be float32 of at least the launch's partials");
  }
}

} // namespace

void PagedPrefillAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 7 || outputs.size() != 2) {
    throw std::runtime_error(
        "PagedPrefillAttention: expected 7 inputs and 2 outputs");
  }
  const array& q_staged = inputs[0];
  const array& k_pool = inputs[1];
  const array& v_pool = inputs[2];
  const array& k_scale = inputs[3];
  const array& v_scale = inputs[4];
  const array& block_table = inputs[5];
  const array& workspace = inputs[6];
  validate_paged_prefill_inputs(
      q_staged, k_pool, v_pool, k_scale, v_scale, block_table, workspace,
      committed_tokens_, rows_, splits_, max_splits_, block_size_, num_q_heads_,
      num_kv_heads_, head_size_, kv_dtype_, int8_queries_,
      "PagedPrefillAttention::eval_gpu");

  // The kernel reads table entries [0, ceil((committed + rows) / block_size))
  // without bounds checks.
  const int32_t* table = block_table.data<int32_t>();
  const int32_t num_blocks = static_cast<int32_t>(k_pool.shape(0));
  const int64_t used =
      (static_cast<int64_t>(committed_tokens_) + rows_ + block_size_ - 1) / block_size_;
  for (int64_t j = 0; j < used; ++j) {
    if (table[j] < 0 || table[j] >= num_blocks) {
      std::ostringstream msg;
      msg << "[runtime] PagedPrefillAttention::eval_gpu: block_table[" << j
          << "] (" << table[j] << ") out of range [0, " << num_blocks << ")";
      throw std::invalid_argument(msg.str());
    }
  }
  const int32_t visible = committed_tokens_ + rows_;
  const int64_t live_blocks = distinct_live_blocks(
      table, &visible, 1, static_cast<uint32_t>(used), block_size_, num_blocks);

  array& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  outputs[1].copy_shared_buffer(workspace);

  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  auto& encoder = mlx::core::metal::get_command_encoder(s);
  mlx::core::fast::paged::PagedPrefillParams params{
      static_cast<uint32_t>(committed_tokens_),
      static_cast<uint32_t>(rows_),
      static_cast<uint32_t>(q_staged.shape(1)),
      static_cast<uint32_t>(block_table.size()),
      static_cast<uint32_t>(splits_),
      static_cast<uint32_t>(max_splits_),
      static_cast<uint32_t>(block_size_),
      static_cast<uint32_t>(num_kv_heads_)};
  mlx::core::fast::paged::dispatch_paged_prefill_attention(
      encoder,
      d,
      out,
      outputs[1],
      q_staged,
      k_pool,
      v_pool,
      k_scale,
      v_scale,
      block_table,
      params,
      num_q_heads_,
      head_size_,
      to_paged_dtype(kv_dtype_),
      int8_queries_,
      live_blocks);
}

std::vector<array> PagedPrefillAttention::vjp(
    const std::vector<array>& /*primals*/,
    const std::vector<array>& /*cotangents*/,
    const std::vector<int>& /*argnums*/,
    const std::vector<array>& /*outputs*/) {
  throw std::runtime_error("PagedPrefillAttention is inference-only");
}

std::vector<Shape> PagedPrefillAttention::output_shapes(
    const std::vector<array>& inputs) {
  if (inputs.size() != 7) {
    throw std::runtime_error("PagedPrefillAttention::output_shapes: expected 7 inputs");
  }
  return {Shape{rows_, num_q_heads_, head_size_}, inputs[6].shape()};
}

bool PagedPrefillAttention::is_equivalent(const Primitive& other) const {
  const auto& o = static_cast<const PagedPrefillAttention&>(other);
  return committed_tokens_ == o.committed_tokens_ && rows_ == o.rows_ &&
      splits_ == o.splits_ && max_splits_ == o.max_splits_ &&
      block_size_ == o.block_size_ && num_q_heads_ == o.num_q_heads_ &&
      num_kv_heads_ == o.num_kv_heads_ && head_size_ == o.head_size_ &&
      kv_dtype_ == o.kv_dtype_ && int8_queries_ == o.int8_queries_;
}

void PagedQueryCodes::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 2 || outputs.size() != 2) {
    throw std::runtime_error("PagedQueryCodes: expected 2 inputs and 2 outputs");
  }
  const array& q = inputs[0];
  const array& positions = inputs[1];
  require_row_contiguous_zero_offset(q, "PagedQueryCodes::eval_gpu", "q");
  require_row_contiguous_zero_offset(positions, "PagedQueryCodes::eval_gpu", "positions");
  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  auto& encoder = mlx::core::metal::get_command_encoder(s);
  mlx::core::fast::paged::PagedQueryCodesParams params{
      static_cast<uint32_t>(q.shape(0)),
      static_cast<uint32_t>(num_q_heads_),
      from_,
      scale_};
  mlx::core::fast::paged::dispatch_paged_query_codes(
      encoder, d, q, positions, outputs[0], outputs[1], params, head_size_);
}

std::vector<array> PagedQueryCodes::vjp(
    const std::vector<array>& /*primals*/,
    const std::vector<array>& /*cotangents*/,
    const std::vector<int>& /*argnums*/,
    const std::vector<array>& /*outputs*/) {
  throw std::runtime_error("PagedQueryCodes is inference-only");
}

std::vector<Shape> PagedQueryCodes::output_shapes(const std::vector<array>& inputs) {
  if (inputs.size() != 2) {
    throw std::runtime_error("PagedQueryCodes::output_shapes: expected 2 inputs");
  }
  return {inputs[0].shape(), Shape{inputs[0].shape(0), num_q_heads_}};
}

bool PagedQueryCodes::is_equivalent(const Primitive& other) const {
  const auto& o = static_cast<const PagedQueryCodes&>(other);
  return num_q_heads_ == o.num_q_heads_ && head_size_ == o.head_size_ &&
      from_ == o.from_ && scale_ == o.scale_;
}

std::array<array, 2> paged_query_codes(
    const array& q,
    const array& positions,
    uint32_t from,
    float scale,
    StreamOrDevice s_) {
  const char* op_name = "paged_query_codes";
  auto s = to_stream(s_);
  if (q.ndim() != 3 || q.dtype() != mlx::core::bfloat16 ||
      (q.shape(2) != 128 && q.shape(2) != 256) || q.shape(0) <= 0 || q.shape(1) <= 0) {
    prefill_reject(op_name, "q must be bf16 [rows, query heads, 128 or 256]");
  }
  if (positions.ndim() != 1 || positions.dtype() != mlx::core::int32 ||
      positions.shape(0) != q.shape(0)) {
    prefill_reject(op_name, "positions must be int32 [rows]");
  }
  const int num_q_heads = q.shape(1);
  const int head_size = q.shape(2);
  auto fallback = [](std::vector<array>) -> std::vector<array> {
    throw std::runtime_error("paged_query_codes has no fallback implementation (inference-only)");
  };
  auto primitive = std::make_shared<PagedQueryCodes>(
      s, std::move(fallback), num_q_heads, head_size, from, scale);
  auto results = array::make_arrays(
      {q.shape(), Shape{q.shape(0), num_q_heads}},
      {mlx::core::bfloat16, mlx::core::float32},
      primitive,
      {contiguous(q, false, s), contiguous(positions, false, s)});
  return {std::move(results[0]), std::move(results[1])};
}

std::array<array, 2> paged_prefill_attention(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& k_scale,
    const array& v_scale,
    const array& block_table,
    const array& workspace,
    int committed_tokens,
    int splits,
    int max_splits,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    KvDtype kv_dtype,
    bool int8_queries,
    float scale,
    StreamOrDevice s_) {
  const char* op_name = "paged_prefill_attention";
  auto s = to_stream(s_);
  if (q.ndim() != 3 || q.shape(1) != num_q_heads || q.shape(2) != head_size) {
    prefill_reject(op_name, "q must be [rows, num_q_heads, head_size]");
  }
  if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) {
    prefill_reject(op_name, "query heads must be a multiple of KV heads");
  }
  // The kernels compile the score scale as a literal (Splash
  // splash_attention_scale).
  if (std::abs(scale * std::sqrt(static_cast<float>(head_size)) - 1.0f) > 1e-6f) {
    prefill_reject(op_name, "scale must be 1/sqrt(head_size)");
  }
  const int rows = q.shape(0);
  const int group = num_q_heads / num_kv_heads;
  const int stride = paged_prefill_stride(rows);
  // Splash Staging: queries [kv head][row][query head in group][dimension].
  array staged = transpose(
      reshape(astype(q, mlx::core::bfloat16, s), {rows, num_kv_heads, group, head_size}, s),
      {1, 0, 2, 3},
      s);
  staged = stride == rows
      ? contiguous(staged, false, s)
      : pad(staged, {1}, {0}, {stride - rows}, array(0.0f, mlx::core::bfloat16), "constant", s);

  validate_paged_prefill_inputs(
      staged, k_pool, v_pool, k_scale, v_scale, block_table, workspace,
      committed_tokens, rows, splits, max_splits, block_size, num_q_heads,
      num_kv_heads, head_size, kv_dtype, int8_queries, op_name);

  auto fallback = [](std::vector<array>) -> std::vector<array> {
    throw std::runtime_error(
        "paged_prefill_attention has no fallback implementation (inference-only)");
  };
  auto primitive = std::make_shared<PagedPrefillAttention>(
      s, std::move(fallback), committed_tokens, rows, splits, max_splits,
      block_size, num_q_heads, num_kv_heads, head_size, kv_dtype, int8_queries);
  auto results = array::make_arrays(
      {Shape{rows, num_q_heads, head_size}, workspace.shape()},
      {mlx::core::bfloat16, mlx::core::float32},
      primitive,
      {staged, k_pool, v_pool, k_scale, v_scale, block_table, workspace});
  return {std::move(results[0]), std::move(results[1])};
}

// =============================================================================
// Inkling-2-Mini Paged Attention & In-Place MPP Paged Prefill Primitives
// =============================================================================

namespace {

void validate_inkling2_pool_planes(
    const array& k_pool,
    const array& v_pool,
    const array& k_scale,
    const array& v_scale,
    int block_size,
    int num_kv_heads,
    int head_size,
    KvDtype kv_dtype,
    const char* op_name) {
  require_row_contiguous_zero_offset(k_pool, op_name, "k_pool");
  require_row_contiguous_zero_offset(v_pool, op_name, "v_pool");
  require_row_contiguous_zero_offset(k_scale, op_name, "k_scale");
  require_row_contiguous_zero_offset(v_scale, op_name, "v_scale");
  if (head_size != 128) {
    prefill_reject(op_name, "inkling2 paged attention requires head_size == 128");
  }
  if (kv_dtype != KvDtype::Int8 && kv_dtype != KvDtype::Bf16) {
    prefill_reject(op_name, "kv_dtype must be Int8 (32-token KTM) or Bf16");
  }
  const auto expected_cache_dtype = cache_dtype_for_kv_dtype(kv_dtype);
  if (k_pool.dtype() != expected_cache_dtype ||
      v_pool.dtype() != expected_cache_dtype) {
    prefill_reject(op_name, "k_pool / v_pool dtype disagrees with kv_dtype");
  }
  if (kv_dtype == KvDtype::Int8) {
    if (block_size != 32) {
      prefill_reject(op_name, "INT8 Inkling2 pool must use 32-token KTM (block_size == 32)");
    }
    if (k_pool.ndim() != 4 || v_pool.ndim() != 4) {
      prefill_reject(op_name, "INT8 KTM requires rank-4 k_pool and rank-4 v_pool");
    }
    const int nb = k_pool.shape(0);
    if (k_pool.shape() != Shape{nb, num_kv_heads, 32, 128} ||
        v_pool.shape() != Shape{nb, num_kv_heads, 128, 32}) {
      prefill_reject(
          op_name,
          "INT8 KTM expects K [nb, H_kv, 32, 128] and V [nb, H_kv, 128, 32]");
    }
  } else {
    if (block_size != 16 && block_size != 32) {
      prefill_reject(op_name, "BF16 Inkling2 pool requires block_size 16 or 32");
    }
    if (k_pool.ndim() != 5 || v_pool.ndim() != 4) {
      prefill_reject(op_name, "BF16 pool requires rank-5 k_pool and rank-4 v_pool");
    }
    const int nb = k_pool.shape(0);
    if (k_pool.shape() != Shape{nb, num_kv_heads, 16, block_size, 8} ||
        v_pool.shape() != Shape{nb, num_kv_heads, 128, block_size}) {
      prefill_reject(
          op_name,
          "BF16 pool expects K [nb, H_kv, 16, bs, 8] and V [nb, H_kv, 128, bs]");
    }
  }
  validate_kv_scales(
      k_pool, k_scale, v_scale, num_kv_heads, block_size, head_size, kv_dtype,
      /*read=*/true, op_name);
}

void validate_inkling2_rel_bias_ex(
    const array* rel_bias,
    const array* rel_bias_w,
    int rows,
    int num_q_heads,
    int rel_extent,
    const char* op_name) {
  if (rel_bias_w != nullptr) {
    if (rel_bias == nullptr || rel_extent <= 0) {
      prefill_reject(
          op_name,
          "rel_bias (r_for_rel) and rel_extent > 0 are required when rel_bias_w is provided");
    }
    require_row_contiguous_head_offset(*rel_bias, op_name, "r_for_rel");
    require_row_contiguous_zero_offset(*rel_bias_w, op_name, "rel_bias_w");
    if (rel_bias->dtype() != mlx::core::float32 ||
        rel_bias_w->dtype() != mlx::core::float32) {
      prefill_reject(op_name, "r_for_rel and rel_bias_w must be float32");
    }
    if (rel_bias->ndim() != 3 ||
        rel_bias->shape() != Shape{rows, num_q_heads, 16}) {
      prefill_reject(
          op_name, "r_for_rel must have shape [total_q, num_q_heads, 16]");
    }
    if (rel_bias_w->ndim() != 2 ||
        rel_bias_w->shape() != Shape{rel_extent, 16}) {
      prefill_reject(op_name, "rel_bias_w must have shape [rel_extent, 16]");
    }
    return;
  }
  if (rel_bias != nullptr) {
    if (rel_extent <= 0) {
      prefill_reject(op_name, "rel_extent must be > 0 when rel_bias is provided");
    }
    require_row_contiguous_head_offset(*rel_bias, op_name, "rel_bias");
    if (rel_bias->dtype() != mlx::core::bfloat16 &&
        rel_bias->dtype() != mlx::core::float32) {
      prefill_reject(op_name, "rel_bias must be bfloat16 or float32");
    }
    if (rel_bias->ndim() != 3 ||
        rel_bias->shape() != Shape{rows, num_q_heads, rel_extent}) {
      prefill_reject(
          op_name,
          "rel_bias must have shape [total_q, num_q_heads, rel_extent]");
    }
  } else if (rel_extent > 0) {
    prefill_reject(op_name, "rel_bias must be provided when rel_extent > 0");
  }
}

std::pair<uint32_t, uint32_t> validate_inkling2_gate_sink(
    const array* gate,
    const array* sink,
    int rows,
    int num_q_heads,
    int head_size,
    const char* op_name) {
  if ((gate == nullptr) != (sink == nullptr)) {
    prefill_reject(op_name, "gate and sink must both be provided or both null");
  }
  if (gate == nullptr) {
    return {0u, 0u};
  }
  if ((gate->dtype() != mlx::core::bfloat16 &&
       gate->dtype() != mlx::core::float32) ||
      (sink->dtype() != mlx::core::bfloat16 &&
       sink->dtype() != mlx::core::float32)) {
    prefill_reject(op_name, "gate and sink must be bfloat16 or float32");
  }
  const int gate_cols = num_q_heads * head_size;
  uint32_t gate_row_stride = static_cast<uint32_t>(gate_cols);
  if (gate->ndim() == 3 &&
      gate->shape() == Shape{rows, num_q_heads, head_size} &&
      gate->flags().row_contiguous) {
    gate_row_stride = static_cast<uint32_t>(gate_cols);
  } else if (
      gate->ndim() == 2 && gate->shape() == Shape{rows, gate_cols} &&
      (gate->flags().row_contiguous ||
       (gate->strides(1) == 1 && gate->strides(0) >= 0))) {
    gate_row_stride = gate->flags().row_contiguous
        ? static_cast<uint32_t>(gate_cols)
        : static_cast<uint32_t>(gate->strides(0));
  } else {
    prefill_reject(
        op_name,
        "gate must be row-contiguous [rows, H, D] or 2D strided [rows, H*D]");
  }
  uint32_t sink_row_stride = static_cast<uint32_t>(num_q_heads);
  if (sink->ndim() == 2 && sink->shape() == Shape{rows, num_q_heads} &&
      (sink->flags().row_contiguous ||
       (sink->strides(1) == 1 && sink->strides(0) >= 0))) {
    sink_row_stride = sink->flags().row_contiguous
        ? static_cast<uint32_t>(num_q_heads)
        : static_cast<uint32_t>(sink->strides(0));
  } else {
    prefill_reject(
        op_name, "sink must be row-contiguous or 2D strided [rows, H]");
  }
  return {gate_row_stride, sink_row_stride};
}

array prepare_inkling2_gate_input(
    const array& gate,
    int rows,
    int num_q_heads,
    int head_size,
    Stream s) {
  const int gate_cols = num_q_heads * head_size;
  if (gate.ndim() == 3 && gate.shape() == Shape{rows, num_q_heads, head_size} &&
      gate.flags().row_contiguous) {
    return gate;
  }
  if (gate.ndim() == 2 && gate.shape() == Shape{rows, gate_cols} &&
      (gate.flags().row_contiguous ||
       (gate.strides(1) == 1 && gate.strides(0) >= 0))) {
    return gate;
  }
  return contiguous(gate, false, s);
}

array prepare_inkling2_sink_input(
    const array& sink,
    int rows,
    int num_q_heads,
    Stream s) {
  if (sink.ndim() == 2 && sink.shape() == Shape{rows, num_q_heads} &&
      (sink.flags().row_contiguous ||
       (sink.strides(1) == 1 && sink.strides(0) >= 0))) {
    return sink;
  }
  return contiguous(sink, false, s);
}

void validate_inkling2_tree_rows(
    const array* tree_rows,
    int rows,
    const char* op_name) {
  if (tree_rows == nullptr) {
    return;
  }
  require_row_contiguous_zero_offset(*tree_rows, op_name, "tree_rows");
  if (tree_rows->ndim() != 2 || tree_rows->dtype() != mlx::core::uint32 ||
      tree_rows->shape(0) != rows || tree_rows->shape(1) != 2) {
    prefill_reject(op_name, "tree_rows must be uint32 [rows, 2]");
  }
}

void validate_inkling2_paged_attention_inputs(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& cu_seqlens_q,
    const array& k_scale,
    const array& v_scale,
    const array* rel_bias,
    const array* rel_bias_w,
    const array* gate,
    const array* sink,
    const array* score_scales,
    const array* tree_rows,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int sliding_window,
    int rel_extent,
    KvDtype kv_dtype,
    const char* op_name) {
  require_row_contiguous_head_offset(q, op_name, "q");
  require_row_contiguous_zero_offset(block_table, op_name, "block_table");
  require_row_contiguous_zero_offset(seq_lens, op_name, "seq_lens");
  require_row_contiguous_zero_offset(cu_seqlens_q, op_name, "cu_seqlens_q");
  if (sliding_window < 0 || rel_extent < 0) {
    prefill_reject(op_name, "sliding_window and rel_extent must be >= 0");
  }
  if (num_kv_heads <= 0 || num_q_heads <= 0 || num_q_heads % num_kv_heads != 0) {
    prefill_reject(op_name, "num_q_heads must be a positive multiple of num_kv_heads");
  }
  if (q.ndim() != 3 || q.dtype() != mlx::core::bfloat16 ||
      q.shape(0) <= 0 || q.shape(1) != num_q_heads || q.shape(2) != head_size) {
    prefill_reject(op_name, "q must be bfloat16 [total_q, num_q_heads, 128]");
  }
  validate_inkling2_pool_planes(
      k_pool, v_pool, k_scale, v_scale, block_size, num_kv_heads, head_size,
      kv_dtype, op_name);
  if (block_table.ndim() != 2 || block_table.dtype() != mlx::core::int32 ||
      seq_lens.ndim() != 1 || seq_lens.dtype() != mlx::core::int32 ||
      cu_seqlens_q.ndim() != 1 || cu_seqlens_q.dtype() != mlx::core::int32 ||
      block_table.shape(0) != seq_lens.shape(0) ||
      cu_seqlens_q.shape(0) != seq_lens.shape(0) + 1) {
    prefill_reject(op_name, "invalid block_table / seq_lens / cu_seqlens_q shape or dtype");
  }
  validate_inkling2_rel_bias_ex(
      rel_bias, rel_bias_w, q.shape(0), num_q_heads, rel_extent, op_name);
  validate_inkling2_gate_sink(
      gate, sink, q.shape(0), num_q_heads, head_size, op_name);
  if (score_scales != nullptr) {
    validate_score_scales(*score_scales, q.shape(0), num_q_heads, kv_dtype, op_name);
  }
  validate_inkling2_tree_rows(tree_rows, q.shape(0), op_name);
}

void validate_inkling2_paged_prefill_inputs(
    const array& q_staged,
    const array& k_pool,
    const array& v_pool,
    const array& k_scale,
    const array& v_scale,
    const array& block_table,
    const array& workspace,
    const array* rel_bias,
    const array* rel_bias_w,
    const array* gate,
    const array* sink,
    const array* tree_rows,
    int committed_tokens,
    int rows,
    int splits,
    int max_splits,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int sliding_window,
    int rel_extent,
    KvDtype kv_dtype,
    bool int8_queries,
    const char* op_name) {
  namespace paged = mlx::core::fast::paged;
  require_row_contiguous_head_offset(q_staged, op_name, "q_staged");
  require_row_contiguous_zero_offset(block_table, op_name, "block_table");
  require_row_contiguous_zero_offset(workspace, op_name, "workspace");
  if (sliding_window < 0 || rel_extent < 0 || committed_tokens < 0) {
    prefill_reject(op_name, "sliding_window, rel_extent, and committed_tokens must be >= 0");
  }
  if (int8_queries && kv_dtype != KvDtype::Int8) {
    prefill_reject(op_name, "INT8 queries require INT8 pages");
  }
  if (num_kv_heads <= 0 || num_q_heads != num_kv_heads * 3) {
    prefill_reject(op_name, "Inkling2 MPP prefill requires GQA group == 3 (num_q_heads == 3 * num_kv_heads)");
  }
  if (max_splits != paged::kPagedPrefillMaxSplits &&
      max_splits != paged::kPagedVerifyMaxSplits) {
    prefill_reject(op_name, "max_splits must be 32 (prefill) or 128 (verify)");
  }
  const int max_rows = max_splits == paged::kPagedVerifyMaxSplits
      ? paged::kPagedVerifyRows
      : paged::kPagedPrefillMaxRows;
  if (rows <= 0 || rows > max_rows) {
    prefill_reject(op_name, "rows out of range");
  }
  if (splits <= 0 || splits > max_splits) {
    prefill_reject(op_name, "splits out of range");
  }
  const Shape expected_q{rows, num_q_heads, head_size};
  if (q_staged.dtype() != mlx::core::bfloat16 || q_staged.shape() != expected_q) {
    prefill_reject(op_name, "q must be bf16 [rows, num_q_heads, 128]");
  }
  validate_inkling2_pool_planes(
      k_pool, v_pool, k_scale, v_scale, block_size, num_kv_heads, head_size,
      kv_dtype, op_name);
  if (block_table.dtype() != mlx::core::int32) {
    prefill_reject(op_name, "block_table must be int32");
  }
  const int64_t needed =
      (static_cast<int64_t>(committed_tokens) + rows + block_size - 1) / block_size;
  if (static_cast<int64_t>(block_table.size()) < needed) {
    prefill_reject(op_name, "block_table too short for committed_tokens + rows");
  }
  if (workspace.dtype() != mlx::core::float32 ||
      static_cast<int64_t>(workspace.size()) <
          paged::paged_prefill_workspace_floats(rows, splits, num_q_heads, head_size)) {
    prefill_reject(op_name, "workspace too small");
  }
  validate_inkling2_rel_bias_ex(
      rel_bias, rel_bias_w, rows, num_q_heads, rel_extent, op_name);
  validate_inkling2_gate_sink(
      gate, sink, rows, num_q_heads, head_size, op_name);
  validate_inkling2_tree_rows(tree_rows, rows, op_name);
}

} // namespace

void Inkling2PagedAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const size_t expected_in = 8 + (has_rel_bias_ ? 1 : 0) +
      (has_rel_bias_w_ ? 1 : 0) + (has_gate_sink_ ? 2 : 0) +
      (has_score_scales_ ? 1 : 0) + (has_tree_ ? 1 : 0);
  const size_t expected_out = write_lse_ ? 2 : 1;
  if (inputs.size() != expected_in || outputs.size() != expected_out) {
    throw std::runtime_error("Inkling2PagedAttention: input/output count mismatch");
  }
  const array& q = inputs[0];
  const array& k_pool = inputs[1];
  const array& v_pool = inputs[2];
  const array& block_table = inputs[3];
  const array& seq_lens = inputs[4];
  const array& cu_seqlens_q = inputs[5];
  const array& k_scale = inputs[6];
  const array& v_scale = inputs[7];
  size_t next_idx = 8;
  const array* rel_bias = has_rel_bias_ ? &inputs[next_idx++] : nullptr;
  const array* rel_bias_w = has_rel_bias_w_ ? &inputs[next_idx++] : nullptr;
  const array* gate = has_gate_sink_ ? &inputs[next_idx++] : nullptr;
  const array* sink = has_gate_sink_ ? &inputs[next_idx++] : nullptr;
  const array* score_scales = has_score_scales_ ? &inputs[next_idx++] : nullptr;
  const array* tree_rows = has_tree_ ? &inputs[next_idx++] : nullptr;

  validate_inkling2_paged_attention_inputs(
      q, k_pool, v_pool, block_table, seq_lens, cu_seqlens_q, k_scale, v_scale,
      rel_bias, rel_bias_w, gate, sink, score_scales, tree_rows, block_size_,
      num_q_heads_, num_kv_heads_, head_size_, sliding_window_, rel_extent_,
      kv_dtype_, "Inkling2PagedAttention::eval_gpu");
  const auto [gate_row_stride, sink_row_stride] = validate_inkling2_gate_sink(
      gate, sink, q.shape(0), num_q_heads_, head_size_,
      "Inkling2PagedAttention::eval_gpu");

  const uint32_t num_seqs = static_cast<uint32_t>(seq_lens.shape(0));
  const uint32_t total_queries = static_cast<uint32_t>(q.shape(0));
  const uint32_t max_blocks_per_seq = static_cast<uint32_t>(block_table.shape(1));
  const int32_t* seq_lens_data = seq_lens.data<int32_t>();
  const int32_t* block_table_data = block_table.data<int32_t>();
  const int32_t* cu_seqlens_q_data = cu_seqlens_q.data<int32_t>();
  const int32_t num_blocks = static_cast<int32_t>(k_pool.shape(0));
  const int64_t max_seq_len_bound =
      static_cast<int64_t>(max_blocks_per_seq) * block_size_;

  int32_t max_context_len = 0;
  for (uint32_t i = 0; i < num_seqs; ++i) {
    const int32_t s = seq_lens_data[i];
    if (s < 0 || static_cast<int64_t>(s) > max_seq_len_bound) {
      throw std::invalid_argument(
          "[runtime] Inkling2PagedAttention::eval_gpu: seq_lens out of range");
    }
    const int64_t used = ceil_div_nonnegative_i32(s, block_size_);
    const size_t row_off = static_cast<size_t>(i) * max_blocks_per_seq;
    for (int64_t j = 0; j < used; ++j) {
      const int32_t blk = block_table_data[row_off + static_cast<size_t>(j)];
      if (blk < 0 || blk >= num_blocks) {
        throw std::invalid_argument(
            "[runtime] Inkling2PagedAttention::eval_gpu: block_table entry out of range");
      }
    }
    if (s > max_context_len) {
      max_context_len = s;
    }
  }
  if (cu_seqlens_q_data[0] != 0 ||
      static_cast<uint32_t>(cu_seqlens_q_data[num_seqs]) != total_queries) {
    throw std::invalid_argument(
        "[runtime] Inkling2PagedAttention::eval_gpu: invalid cu_seqlens_q endpoints");
  }
  for (uint32_t i = 0; i < num_seqs; ++i) {
    const int32_t span = cu_seqlens_q_data[i + 1] - cu_seqlens_q_data[i];
    if (span < 0 || span > seq_lens_data[i]) {
      throw std::invalid_argument(
          "[runtime] Inkling2PagedAttention::eval_gpu: invalid query span");
    }
  }
  if (max_context_len <= 0) {
    throw std::runtime_error("Inkling2PagedAttention: max_context_len must be > 0");
  }
  const int64_t live_blocks = live_blocks_override_ >= 0
      ? std::min<int64_t>(live_blocks_override_, num_blocks)
      : distinct_live_blocks(
            block_table_data, seq_lens_data, num_seqs, max_blocks_per_seq,
            block_size_, num_blocks);

  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  array* lse_out_ptr = nullptr;
  if (write_lse_) {
    outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));
    lse_out_ptr = &outputs[1];
  }

  const uint32_t rel_mode = (rel_extent_ <= 0 || rel_bias == nullptr)
      ? 0u
      : (rel_bias_w != nullptr ? 2u : 1u);
  mlx::core::fast::paged::Inkling2AttnExtraParams extra{
      has_gate_sink_ ? 1u : 0u,
      gate_row_stride,
      (gate != nullptr && gate->dtype() == mlx::core::float32) ? 1u : 0u,
      sink_row_stride,
      (sink != nullptr && sink->dtype() == mlx::core::float32) ? 1u : 0u,
      write_lse_ ? 1u : 0u,
      rel_mode,
      has_tree_ ? 1u : 0u};

  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  auto& encoder = mlx::core::metal::get_command_encoder(s);
  mlx::core::fast::paged::dispatch_inkling2_paged_attention_auto(
      encoder, d, s, outputs[0], lse_out_ptr, q, k_pool, v_pool, block_table,
      seq_lens, cu_seqlens_q, k_scale, v_scale, rel_bias, rel_bias_w, gate,
      sink, extra, static_cast<int>(num_seqs), static_cast<int>(total_queries),
      num_q_heads_, num_kv_heads_, head_size_, block_size_, max_context_len,
      static_cast<int>(max_blocks_per_seq), live_blocks, scale_,
      sliding_window_, rel_extent_, to_paged_dtype(kv_dtype_), score_scales,
      tree_rows);
}

std::vector<array> Inkling2PagedAttention::vjp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("Inkling2PagedAttention is inference-only");
}

std::vector<Shape> Inkling2PagedAttention::output_shapes(
    const std::vector<array>& inputs) {
  const auto& q = inputs[0];
  if (write_lse_) {
    return {Shape{q.shape(0), num_q_heads_, head_size_},
            Shape{q.shape(0), num_q_heads_}};
  }
  return {Shape{q.shape(0), num_q_heads_, head_size_}};
}

bool Inkling2PagedAttention::is_equivalent(const Primitive& other) const {
  const auto& o = static_cast<const Inkling2PagedAttention&>(other);
  return scale_ == o.scale_ && block_size_ == o.block_size_ &&
      num_q_heads_ == o.num_q_heads_ && num_kv_heads_ == o.num_kv_heads_ &&
      head_size_ == o.head_size_ && sliding_window_ == o.sliding_window_ &&
      rel_extent_ == o.rel_extent_ && kv_dtype_ == o.kv_dtype_ &&
      has_rel_bias_ == o.has_rel_bias_ &&
      has_rel_bias_w_ == o.has_rel_bias_w_ &&
      has_gate_sink_ == o.has_gate_sink_ &&
      write_lse_ == o.write_lse_ &&
      has_score_scales_ == o.has_score_scales_ &&
      has_tree_ == o.has_tree_ &&
      live_blocks_override_ == o.live_blocks_override_;
}

std::vector<array> inkling2_paged_attention(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& block_table,
    const array& seq_lens,
    const array& cu_seqlens_q,
    const array& k_scale,
    const array& v_scale,
    const std::optional<array>& rel_bias,
    float scale,
    int sliding_window,
    int rel_extent,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    KvDtype kv_dtype,
    StreamOrDevice s_,
    const std::optional<array>& score_scales,
    const std::optional<array>& rel_bias_w,
    const std::optional<array>& gate,
    const std::optional<array>& sink,
    bool write_lse,
    const std::optional<array>& tree_rows,
    int live_blocks_override) {
  auto s = to_stream(s_);
  const int rows = q.shape(0);
  std::optional<array> gate_prep = gate
      ? std::optional<array>(
            prepare_inkling2_gate_input(*gate, rows, num_q_heads, head_size, s))
      : std::nullopt;
  std::optional<array> sink_prep = sink
      ? std::optional<array>(
            prepare_inkling2_sink_input(*sink, rows, num_q_heads, s))
      : std::nullopt;
  std::optional<array> tree_prep;
  if (tree_rows) {
    tree_prep = (tree_rows->flags().row_contiguous && tree_rows->offset() == 0)
        ? *tree_rows
        : contiguous(*tree_rows, false, s);
  }
  const array* rel_ptr = rel_bias ? &(*rel_bias) : nullptr;
  const array* rel_w_ptr = rel_bias_w ? &(*rel_bias_w) : nullptr;
  const array* gate_ptr = gate_prep ? &(*gate_prep) : nullptr;
  const array* sink_ptr = sink_prep ? &(*sink_prep) : nullptr;
  const array* ss_ptr = score_scales ? &(*score_scales) : nullptr;
  const array* tree_ptr = tree_prep ? &(*tree_prep) : nullptr;
  validate_inkling2_paged_attention_inputs(
      q, k_pool, v_pool, block_table, seq_lens, cu_seqlens_q, k_scale, v_scale,
      rel_ptr, rel_w_ptr, gate_ptr, sink_ptr, ss_ptr, tree_ptr, block_size,
      num_q_heads, num_kv_heads, head_size, sliding_window, rel_extent,
      kv_dtype, "inkling2_paged_attention");

  std::vector<array> inputs = {
      q, k_pool, v_pool, block_table, seq_lens, cu_seqlens_q, k_scale, v_scale};
  if (rel_bias) {
    inputs.push_back(*rel_bias);
  }
  if (rel_bias_w) {
    inputs.push_back(*rel_bias_w);
  }
  if (gate_prep && sink_prep) {
    inputs.push_back(*gate_prep);
    inputs.push_back(*sink_prep);
  }
  if (score_scales) {
    inputs.push_back(*score_scales);
  }
  if (tree_prep) {
    inputs.push_back(*tree_prep);
  }
  auto fallback = [](std::vector<array>) -> std::vector<array> {
    throw std::runtime_error(
        "inkling2_paged_attention has no fallback implementation");
  };
  const bool has_gate_sink = gate_prep.has_value() && sink_prep.has_value();
  auto prim = std::make_shared<Inkling2PagedAttention>(
      s, std::move(fallback), scale, block_size, num_q_heads, num_kv_heads,
      head_size, sliding_window, rel_extent, kv_dtype, rel_bias.has_value(),
      rel_bias_w.has_value(), has_gate_sink, write_lse,
      score_scales.has_value(), tree_prep.has_value(), live_blocks_override);
  std::vector<Shape> out_shapes = {Shape{rows, num_q_heads, head_size}};
  std::vector<Dtype> out_dtypes = {mlx::core::bfloat16};
  if (write_lse) {
    out_shapes.push_back(Shape{rows, num_q_heads});
    out_dtypes.push_back(mlx::core::float32);
  }
  return array::make_arrays(
      std::move(out_shapes), std::move(out_dtypes), std::move(prim),
      std::move(inputs));
}

void Inkling2PagedPrefillAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const size_t expected_in = 7 + (has_rel_bias_ ? 1 : 0) +
      (has_rel_bias_w_ ? 1 : 0) + (has_gate_sink_ ? 2 : 0) +
      (has_tree_ ? 1 : 0);
  const size_t expected_out = write_lse_ ? 3 : 2;
  if (inputs.size() != expected_in || outputs.size() != expected_out) {
    throw std::runtime_error(
        "Inkling2PagedPrefillAttention: input/output count mismatch");
  }
  const array& q_staged = inputs[0];
  const array& k_pool = inputs[1];
  const array& v_pool = inputs[2];
  const array& k_scale = inputs[3];
  const array& v_scale = inputs[4];
  const array& block_table = inputs[5];
  const array& workspace = inputs[6];
  size_t next_idx = 7;
  const array* rel_bias = has_rel_bias_ ? &inputs[next_idx++] : nullptr;
  const array* rel_bias_w = has_rel_bias_w_ ? &inputs[next_idx++] : nullptr;
  const array* gate = has_gate_sink_ ? &inputs[next_idx++] : nullptr;
  const array* sink = has_gate_sink_ ? &inputs[next_idx++] : nullptr;
  const array* tree_rows = has_tree_ ? &inputs[next_idx++] : nullptr;

  validate_inkling2_paged_prefill_inputs(
      q_staged, k_pool, v_pool, k_scale, v_scale, block_table, workspace,
      rel_bias, rel_bias_w, gate, sink, tree_rows, committed_tokens_, rows_,
      splits_, max_splits_, block_size_, num_q_heads_, num_kv_heads_,
      head_size_, sliding_window_, rel_extent_, kv_dtype_, int8_queries_,
      "Inkling2PagedPrefillAttention::eval_gpu");
  const auto [gate_row_stride, sink_row_stride] = validate_inkling2_gate_sink(
      gate, sink, rows_, num_q_heads_, head_size_,
      "Inkling2PagedPrefillAttention::eval_gpu");

  const int32_t* table = block_table.data<int32_t>();
  const int32_t num_blocks = static_cast<int32_t>(k_pool.shape(0));
  const int64_t used =
      (static_cast<int64_t>(committed_tokens_) + rows_ + block_size_ - 1) /
      block_size_;
  for (int64_t j = 0; j < used; ++j) {
    if (table[j] < 0 || table[j] >= num_blocks) {
      throw std::invalid_argument(
          "[runtime] Inkling2PagedPrefillAttention::eval_gpu: block_table out of range");
    }
  }
  const int32_t visible = committed_tokens_ + rows_;
  const int64_t live_blocks = live_blocks_override_ >= 0
      ? std::min<int64_t>(live_blocks_override_, num_blocks)
      : distinct_live_blocks(
            table, &visible, 1, static_cast<uint32_t>(used), block_size_,
            num_blocks);

  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  array* lse_out_ptr = nullptr;
  size_t ws_idx = 1;
  if (write_lse_) {
    outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));
    lse_out_ptr = &outputs[1];
    ws_idx = 2;
  }
  outputs[ws_idx].copy_shared_buffer(workspace);

  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  auto& encoder = mlx::core::metal::get_command_encoder(s);
  const bool use_int8_q =
      int8_queries_ && paged::paged_prefill_int8_queries_supported(d);
  const uint32_t rel_mode = (rel_extent_ <= 0 || rel_bias == nullptr)
      ? 0u
      : (rel_bias_w != nullptr ? 2u : 1u);
  mlx::core::fast::paged::Inkling2PagedPrefillParams params{
      static_cast<uint32_t>(committed_tokens_),
      static_cast<uint32_t>(rows_),
      static_cast<uint32_t>(paged_prefill_stride(rows_)),
      static_cast<uint32_t>(block_table.size()),
      static_cast<uint32_t>(splits_),
      static_cast<uint32_t>(max_splits_),
      static_cast<uint32_t>(block_size_),
      static_cast<uint32_t>(num_kv_heads_),
      static_cast<uint32_t>(sliding_window_),
      static_cast<uint32_t>(rel_extent_),
      (rel_bias != nullptr && rel_bias->dtype() == mlx::core::float32) ? 1u : 0u,
      rel_mode,
      has_gate_sink_ ? 1u : 0u,
      gate_row_stride,
      (gate != nullptr && gate->dtype() == mlx::core::float32) ? 1u : 0u,
      sink_row_stride,
      (sink != nullptr && sink->dtype() == mlx::core::float32) ? 1u : 0u,
      write_lse_ ? 1u : 0u,
      has_tree_ ? 1u : 0u,
      0u};
  mlx::core::fast::paged::dispatch_inkling2_paged_prefill_attention(
      encoder, d, outputs[0], lse_out_ptr, outputs[ws_idx], q_staged, k_pool,
      v_pool, k_scale, v_scale, block_table, rel_bias, rel_bias_w, gate, sink,
      params, num_q_heads_, head_size_, to_paged_dtype(kv_dtype_), use_int8_q,
      live_blocks, tree_rows);
}

std::vector<array> Inkling2PagedPrefillAttention::vjp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("Inkling2PagedPrefillAttention is inference-only");
}

std::vector<Shape> Inkling2PagedPrefillAttention::output_shapes(
    const std::vector<array>& inputs) {
  if (write_lse_) {
    return {Shape{rows_, num_q_heads_, head_size_},
            Shape{rows_, num_q_heads_},
            inputs[6].shape()};
  }
  return {Shape{rows_, num_q_heads_, head_size_}, inputs[6].shape()};
}

bool Inkling2PagedPrefillAttention::is_equivalent(const Primitive& other) const {
  const auto& o = static_cast<const Inkling2PagedPrefillAttention&>(other);
  return committed_tokens_ == o.committed_tokens_ && rows_ == o.rows_ &&
      splits_ == o.splits_ && max_splits_ == o.max_splits_ &&
      block_size_ == o.block_size_ && num_q_heads_ == o.num_q_heads_ &&
      num_kv_heads_ == o.num_kv_heads_ && head_size_ == o.head_size_ &&
      sliding_window_ == o.sliding_window_ && rel_extent_ == o.rel_extent_ &&
      kv_dtype_ == o.kv_dtype_ && int8_queries_ == o.int8_queries_ &&
      has_rel_bias_ == o.has_rel_bias_ &&
      has_rel_bias_w_ == o.has_rel_bias_w_ &&
      has_gate_sink_ == o.has_gate_sink_ &&
      write_lse_ == o.write_lse_ &&
      has_tree_ == o.has_tree_ &&
      live_blocks_override_ == o.live_blocks_override_;
}

std::vector<array> inkling2_paged_prefill_attention(
    const array& q,
    const array& k_pool,
    const array& v_pool,
    const array& k_scale,
    const array& v_scale,
    const array& block_table,
    const array& workspace,
    const std::optional<array>& rel_bias,
    int committed_tokens,
    int splits,
    int max_splits,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int sliding_window,
    int rel_extent,
    KvDtype kv_dtype,
    bool int8_queries,
    float scale,
    StreamOrDevice s_,
    const std::optional<array>& rel_bias_w,
    const std::optional<array>& gate,
    const std::optional<array>& sink,
    bool write_lse,
    const std::optional<array>& tree_rows,
    int live_blocks_override) {
  const char* op_name = "inkling2_paged_prefill_attention";
  auto s = to_stream(s_);
  if (q.ndim() != 3 || q.shape(1) != num_q_heads || q.shape(2) != head_size) {
    prefill_reject(op_name, "q must be [rows, num_q_heads, head_size]");
  }
  if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) {
    prefill_reject(op_name, "num_q_heads must be a positive multiple of num_kv_heads");
  }
  const int rows = q.shape(0);
  const int group = num_q_heads / num_kv_heads;
  const bool scale_is_default =
      std::abs(scale * std::sqrt(static_cast<float>(head_size)) - 1.0f) <= 1e-6f;

  // Portable fallback when MPP is unavailable on this GPU, or for non-G3/non-default-scale shapes.
  auto& dev = mlx::core::metal::device(s.device);
  const bool use_mpp = group == 3 && head_size == 128 && scale_is_default &&
      paged::inkling2_paged_prefill_mpp_supported(dev);
  if (!use_mpp) {
    const int32_t seq_len_val = committed_tokens + rows;
    const int32_t cu_vals[2] = {0, rows};
    array seq_lens_arr(&seq_len_val, Shape{1}, mlx::core::int32);
    array cu_arr(cu_vals, Shape{2}, mlx::core::int32);
    array bt_2d = block_table.ndim() == 2
        ? block_table
        : reshape(block_table, {1, static_cast<int>(block_table.size())}, s);
    auto q_bf16 = contiguous(astype(q, mlx::core::bfloat16, s), false, s);
    auto out_vec = inkling2_paged_attention(
        q_bf16, k_pool, v_pool, bt_2d, seq_lens_arr, cu_arr, k_scale, v_scale,
        rel_bias, scale, sliding_window, rel_extent, block_size, num_q_heads,
        num_kv_heads, head_size, kv_dtype, s, std::nullopt, rel_bias_w, gate,
        sink, write_lse, tree_rows, live_blocks_override);
    out_vec.push_back(workspace);
    return out_vec;
  }

  // Inkling2 MPP split kernels cooperatively load raw `[rows, num_q_heads, 128]`
  // BF16 queries straight into threadgroup SRAM `q_tile` (0-padded past `rows`),
  // eliminating 3-4 MLX graph ops (`astype`/`reshape`/`transpose`/`contiguous`/`pad`)
  // and 1 Metal transpose dispatch + DRAM roundtrip per visit!
  array staged = (q.dtype() == mlx::core::bfloat16 &&
                  q.flags().row_contiguous && q.offset() == 0)
      ? q
      : contiguous(astype(q, mlx::core::bfloat16, s), false, s);

  std::optional<array> gate_prep = gate
      ? std::optional<array>(
            prepare_inkling2_gate_input(*gate, rows, num_q_heads, head_size, s))
      : std::nullopt;
  std::optional<array> sink_prep = sink
      ? std::optional<array>(
            prepare_inkling2_sink_input(*sink, rows, num_q_heads, s))
      : std::nullopt;
  std::optional<array> tree_prep;
  if (tree_rows) {
    tree_prep = (tree_rows->flags().row_contiguous && tree_rows->offset() == 0)
        ? *tree_rows
        : contiguous(*tree_rows, false, s);
  }
  const array* rel_ptr = rel_bias ? &(*rel_bias) : nullptr;
  const array* rel_w_ptr = rel_bias_w ? &(*rel_bias_w) : nullptr;
  const array* gate_ptr = gate_prep ? &(*gate_prep) : nullptr;
  const array* sink_ptr = sink_prep ? &(*sink_prep) : nullptr;
  const array* tree_ptr = tree_prep ? &(*tree_prep) : nullptr;
  validate_inkling2_paged_prefill_inputs(
      staged, k_pool, v_pool, k_scale, v_scale, block_table, workspace,
      rel_ptr, rel_w_ptr, gate_ptr, sink_ptr, tree_ptr, committed_tokens, rows,
      splits, max_splits, block_size, num_q_heads, num_kv_heads, head_size,
      sliding_window, rel_extent, kv_dtype, int8_queries, op_name);

  std::vector<array> inputs = {
      staged, k_pool, v_pool, k_scale, v_scale, block_table, workspace};
  if (rel_bias) {
    inputs.push_back(*rel_bias);
  }
  if (rel_bias_w) {
    inputs.push_back(*rel_bias_w);
  }
  if (gate_prep && sink_prep) {
    inputs.push_back(*gate_prep);
    inputs.push_back(*sink_prep);
  }
  if (tree_prep) {
    inputs.push_back(*tree_prep);
  }
  auto fallback = [](std::vector<array>) -> std::vector<array> {
    throw std::runtime_error(
        "inkling2_paged_prefill_attention has no fallback implementation");
  };
  const bool has_gate_sink = gate_prep.has_value() && sink_prep.has_value();
  auto prim = std::make_shared<Inkling2PagedPrefillAttention>(
      s, std::move(fallback), committed_tokens, rows, splits, max_splits,
      block_size, num_q_heads, num_kv_heads, head_size, sliding_window,
      rel_extent, kv_dtype, int8_queries, rel_bias.has_value(),
      rel_bias_w.has_value(), has_gate_sink, write_lse,
      tree_prep.has_value(), live_blocks_override);
  std::vector<Shape> out_shapes = {Shape{rows, num_q_heads, head_size}};
  std::vector<Dtype> out_dtypes = {mlx::core::bfloat16};
  if (write_lse) {
    out_shapes.push_back(Shape{rows, num_q_heads});
    out_dtypes.push_back(mlx::core::float32);
  }
  out_shapes.push_back(workspace.shape());
  out_dtypes.push_back(mlx::core::float32);
  return array::make_arrays(
      std::move(out_shapes), std::move(out_dtypes), std::move(prim),
      std::move(inputs));
}

} // namespace mlx::core::fast

extern "C" {

/// Production FFI: emit a `PagedAttention` MLX Custom primitive and return
/// the resulting on-device array. This is intentionally a tiny bridge over
/// the C++ factory so Rust model code can use the same MLX-graph paged
/// attention primitive as the compiled Qwen path without copying attention
/// outputs through host memory.
///
/// Returns nullptr on bridge/factory validation errors; Rust callers keep the
/// existing read_kv_range + SDPA path as fallback. The returned array is still
/// lazy, so GPU dispatch errors surface later when MLX evaluates the graph.
mlx_array* mlx_paged_attention_forward_with_plan(
    mlx_array* q_ptr,
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* block_table_ptr,
    mlx_array* seq_lens_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    float scale,
    float softcap,
    int sliding_window,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    uint8_t kv_dtype_raw,
    uint8_t route_hint,
    uint32_t grouped_stripes,
    mlx_array* score_scales_ptr) {
  if (!q_ptr || !k_pool_ptr || !v_pool_ptr || !block_table_ptr ||
      !seq_lens_ptr || !k_scale_ptr || !v_scale_ptr) {
    return nullptr;
  }

  try {
    using namespace mlx::core;
    using namespace mlx::core::fast;

    auto& q_raw           = *reinterpret_cast<array*>(q_ptr);
    auto& k_pool          = *reinterpret_cast<array*>(k_pool_ptr);
    auto& v_pool          = *reinterpret_cast<array*>(v_pool_ptr);
    auto& block_table_raw = *reinterpret_cast<array*>(block_table_ptr);
    auto& seq_lens_raw    = *reinterpret_cast<array*>(seq_lens_ptr);
    auto& k_scale         = *reinterpret_cast<array*>(k_scale_ptr);
    auto& v_scale         = *reinterpret_cast<array*>(v_scale_ptr);

    require_known_kv_dtype(kv_dtype_raw);
    auto kv_dtype = static_cast<KvDtype>(kv_dtype_raw);

    // The public factory rejects non-row-contiguous/nonzero-offset inputs.
    // `q` is often produced by reshape/slice/rope chains in Rust model code,
    // so materialize q explicitly at this bridge.
    //
    // Metadata is different: `PagedAttention::eval_gpu` performs host-side
    // content checks on block_table/seq_lens before dispatch. Wrapping those
    // arrays in lazy `contiguous(...)` nodes can make the validator read lazy
    // metadata copies rather than the adapter-owned materialized arrays. Keep
    // the metadata contract strict here: Rust owns materialized, row-contiguous,
    // zero-offset metadata arrays, and the factory validates that contract.
    auto q = contiguous(q_raw);

    auto out = paged_attention_with_route_hint(
        q,
        k_pool,
        v_pool,
        block_table_raw,
        seq_lens_raw,
        k_scale,
        v_scale,
        scale,
        softcap,
        sliding_window,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_size,
        kv_dtype,
        route_hint,
        {},
        grouped_stripes,
        score_scales_ptr == nullptr
            ? std::nullopt
            : std::optional<array>(*reinterpret_cast<array*>(score_scales_ptr)));

    return reinterpret_cast<mlx_array*>(new array(std::move(out)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("paged_attention_forward_with_route", e.what());
    return nullptr;
  } catch (...) {
    mlx_trace_native_error(
        "paged_attention_forward_with_route", "unknown exception");
    return nullptr;
  }
}

mlx_array* mlx_paged_attention_forward(
    mlx_array* q_ptr,
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* block_table_ptr,
    mlx_array* seq_lens_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    float scale,
    float softcap,
    int sliding_window,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    uint8_t kv_dtype_raw) {
  return mlx_paged_attention_forward_with_plan(
      q_ptr,
      k_pool_ptr,
      v_pool_ptr,
      block_table_ptr,
      seq_lens_ptr,
      k_scale_ptr,
      v_scale_ptr,
      scale,
      softcap,
      sliding_window,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_size,
      kv_dtype_raw,
      /*route_hint=*/0,
      /*grouped_stripes=*/0,
      /*score_scales=*/nullptr);
}

/// Production FFI: emit a `PagedAttentionVarlen` MLX Custom primitive and
/// return the resulting lazy on-device array. This is the ragged-Q sibling of
/// `mlx_paged_attention_forward`: `block_table` / `seq_lens` describe one row
/// per sequence while `cu_seqlens_q` maps the flat query rows back to those
/// sequences.
///
/// Returns nullptr on bridge/factory validation errors. The returned array is
/// still lazy, so GPU dispatch errors surface later when MLX evaluates the
/// graph.
mlx_array* mlx_paged_attention_varlen_forward(
    mlx_array* q_ptr,
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* block_table_ptr,
    mlx_array* seq_lens_ptr,
    mlx_array* cu_seqlens_q_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    float scale,
    float softcap,
    int sliding_window,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    uint8_t kv_dtype_raw,
    mlx_array* score_scales_ptr) {
  if (!q_ptr || !k_pool_ptr || !v_pool_ptr || !block_table_ptr ||
      !seq_lens_ptr || !cu_seqlens_q_ptr || !k_scale_ptr || !v_scale_ptr) {
    return nullptr;
  }

  try {
    using namespace mlx::core;
    using namespace mlx::core::fast;

    require_known_kv_dtype(kv_dtype_raw);

    auto& q_raw             = *reinterpret_cast<array*>(q_ptr);
    auto& k_pool            = *reinterpret_cast<array*>(k_pool_ptr);
    auto& v_pool            = *reinterpret_cast<array*>(v_pool_ptr);
    auto& block_table_raw   = *reinterpret_cast<array*>(block_table_ptr);
    auto& seq_lens_raw      = *reinterpret_cast<array*>(seq_lens_ptr);
    auto& cu_seqlens_q_raw  = *reinterpret_cast<array*>(cu_seqlens_q_ptr);
    auto& k_scale           = *reinterpret_cast<array*>(k_scale_ptr);
    auto& v_scale           = *reinterpret_cast<array*>(v_scale_ptr);

    auto kv_dtype = static_cast<KvDtype>(kv_dtype_raw);

    // Match the single-row bridge: query activations may come from lazy
    // reshape/slice/rope chains, so make only q contiguous here. The varlen
    // primitive host-reads block_table, seq_lens, and cu_seqlens_q during GPU
    // evaluation; preserving those adapter-owned arrays lets the factory reject
    // non-row-contiguous or nonzero-offset metadata instead of hiding it behind
    // lazy contiguous copies.
    auto q = contiguous(q_raw);

    auto out = paged_attention_varlen(
        q,
        k_pool,
        v_pool,
        block_table_raw,
        seq_lens_raw,
        cu_seqlens_q_raw,
        k_scale,
        v_scale,
        scale,
        softcap,
        sliding_window,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_size,
        kv_dtype,
        {},
        score_scales_ptr == nullptr
            ? std::nullopt
            : std::optional<array>(*reinterpret_cast<array*>(score_scales_ptr)));

    return reinterpret_cast<mlx_array*>(new array(std::move(out)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("paged_attention_varlen_forward", e.what());
    return nullptr;
  } catch (...) {
    mlx_trace_native_error(
        "paged_attention_varlen_forward", "unknown exception");
    return nullptr;
  }
}


/// Production FFI: emit a `PagedQueryCodes`. Returns false on a validation
/// error; the outputs stay lazy.
bool mlx_paged_query_codes_forward(
    mlx_array* q_ptr,
    mlx_array* positions_ptr,
    uint32_t from,
    float scale,
    mlx_array** out_codes,
    mlx_array** out_score_scales) {
  if (out_codes) *out_codes = nullptr;
  if (out_score_scales) *out_score_scales = nullptr;
  if (!q_ptr || !positions_ptr || !out_codes || !out_score_scales) {
    return false;
  }
  try {
    using namespace mlx::core;
    auto results = fast::paged_query_codes(
        *reinterpret_cast<array*>(q_ptr),
        *reinterpret_cast<array*>(positions_ptr),
        from,
        scale);
    *out_codes = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_score_scales = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("paged_query_codes_forward", e.what());
    return false;
  } catch (...) {
    mlx_trace_native_error("paged_query_codes_forward", "unknown exception");
    return false;
  }
}

/// Production FFI: emit a `PagedPrefillAttention` (queries staged inside).
/// Returns false on a validation error; the outputs stay lazy.
bool mlx_paged_prefill_attention_forward(
    mlx_array* q_ptr,
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    mlx_array* block_table_ptr,
    mlx_array* workspace_ptr,
    int committed_tokens,
    int splits,
    int max_splits,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    uint8_t kv_dtype_raw,
    bool int8_queries,
    float scale,
    mlx_array** out_attention,
    mlx_array** out_workspace) {
  if (out_attention) *out_attention = nullptr;
  if (out_workspace) *out_workspace = nullptr;
  if (!q_ptr || !k_pool_ptr || !v_pool_ptr || !k_scale_ptr || !v_scale_ptr ||
      !block_table_ptr || !workspace_ptr || !out_attention || !out_workspace) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::fast;
    require_known_kv_dtype(kv_dtype_raw);
    auto results = paged_prefill_attention(
        *reinterpret_cast<array*>(q_ptr),
        *reinterpret_cast<array*>(k_pool_ptr),
        *reinterpret_cast<array*>(v_pool_ptr),
        *reinterpret_cast<array*>(k_scale_ptr),
        *reinterpret_cast<array*>(v_scale_ptr),
        *reinterpret_cast<array*>(block_table_ptr),
        *reinterpret_cast<array*>(workspace_ptr),
        committed_tokens,
        splits,
        max_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_size,
        static_cast<KvDtype>(kv_dtype_raw),
        int8_queries,
        scale);
    *out_attention = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_workspace = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("paged_prefill_attention_forward", e.what());
    return false;
  } catch (...) {
    mlx_trace_native_error("paged_prefill_attention_forward", "unknown exception");
    return false;
  }
}

bool mlx_inkling2_paged_attention_forward(
    mlx_array* q_ptr,
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* block_table_ptr,
    mlx_array* seq_lens_ptr,
    mlx_array* cu_seqlens_q_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    mlx_array* rel_bias_ptr,
    mlx_array* rel_bias_w_ptr,
    mlx_array* gate_ptr,
    mlx_array* sink_ptr,
    float scale,
    int sliding_window,
    int rel_extent,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    uint8_t kv_dtype_raw,
    mlx_array* score_scales_ptr,
    mlx_array* tree_rows_ptr,
    int live_blocks,
    mlx_array** out_attention,
    mlx_array** out_lse) {
  if (out_attention) *out_attention = nullptr;
  if (out_lse) *out_lse = nullptr;
  if (!q_ptr || !k_pool_ptr || !v_pool_ptr || !block_table_ptr ||
      !seq_lens_ptr || !cu_seqlens_q_ptr || !k_scale_ptr || !v_scale_ptr ||
      !out_attention) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::fast;
    require_known_kv_dtype(kv_dtype_raw);
    const auto& q_raw = *reinterpret_cast<array*>(q_ptr);
    array q = (q_raw.dtype() == bfloat16 && q_raw.flags().row_contiguous)
        ? q_raw
        : contiguous(astype(q_raw, bfloat16));
    std::optional<array> rel_bias;
    if (rel_bias_ptr != nullptr) {
      const auto& rb_raw = *reinterpret_cast<array*>(rel_bias_ptr);
      rel_bias = rb_raw.flags().row_contiguous ? rb_raw : contiguous(rb_raw);
    }
    std::optional<array> rel_bias_w;
    if (rel_bias_w_ptr != nullptr) {
      const auto& rw_raw = *reinterpret_cast<array*>(rel_bias_w_ptr);
      rel_bias_w = (rw_raw.flags().row_contiguous && rw_raw.offset() == 0)
          ? rw_raw
          : contiguous(rw_raw);
    }
    std::optional<array> gate = gate_ptr != nullptr
        ? std::optional<array>(*reinterpret_cast<array*>(gate_ptr))
        : std::nullopt;
    std::optional<array> sink = sink_ptr != nullptr
        ? std::optional<array>(*reinterpret_cast<array*>(sink_ptr))
        : std::nullopt;
    std::optional<array> score_scales = score_scales_ptr != nullptr
        ? std::optional<array>(contiguous(*reinterpret_cast<array*>(score_scales_ptr)))
        : std::nullopt;
    std::optional<array> tree_rows;
    if (tree_rows_ptr != nullptr) {
      const auto& tr_raw = *reinterpret_cast<array*>(tree_rows_ptr);
      tree_rows = (tr_raw.flags().row_contiguous && tr_raw.offset() == 0)
          ? tr_raw
          : contiguous(tr_raw);
    }
    const bool write_lse = (out_lse != nullptr);
    auto results = inkling2_paged_attention(
        q,
        *reinterpret_cast<array*>(k_pool_ptr),
        *reinterpret_cast<array*>(v_pool_ptr),
        *reinterpret_cast<array*>(block_table_ptr),
        *reinterpret_cast<array*>(seq_lens_ptr),
        *reinterpret_cast<array*>(cu_seqlens_q_ptr),
        *reinterpret_cast<array*>(k_scale_ptr),
        *reinterpret_cast<array*>(v_scale_ptr),
        rel_bias,
        scale,
        sliding_window,
        rel_extent,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_size,
        static_cast<KvDtype>(kv_dtype_raw),
        {},
        score_scales,
        rel_bias_w,
        gate,
        sink,
        write_lse,
        tree_rows,
        live_blocks);
    *out_attention = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    if (write_lse) {
      *out_lse = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    }
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_paged_attention_forward", e.what());
    return false;
  } catch (...) {
    mlx_trace_native_error("inkling2_paged_attention_forward", "unknown exception");
    return false;
  }
}

bool mlx_inkling2_paged_prefill_attention_forward(
    mlx_array* q_ptr,
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    mlx_array* block_table_ptr,
    mlx_array* workspace_ptr,
    mlx_array* rel_bias_ptr,
    mlx_array* rel_bias_w_ptr,
    mlx_array* gate_ptr,
    mlx_array* sink_ptr,
    int committed_tokens,
    int splits,
    int max_splits,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int sliding_window,
    int rel_extent,
    uint8_t kv_dtype_raw,
    bool int8_queries,
    float scale,
    mlx_array* tree_rows_ptr,
    int live_blocks,
    mlx_array** out_attention,
    mlx_array** out_lse,
    mlx_array** out_workspace) {
  if (out_attention) *out_attention = nullptr;
  if (out_lse) *out_lse = nullptr;
  if (out_workspace) *out_workspace = nullptr;
  if (!q_ptr || !k_pool_ptr || !v_pool_ptr || !k_scale_ptr || !v_scale_ptr ||
      !block_table_ptr || !workspace_ptr || !out_attention || !out_workspace) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::fast;
    require_known_kv_dtype(kv_dtype_raw);
    std::optional<array> rel_bias;
    if (rel_bias_ptr != nullptr) {
      const auto& rb_raw = *reinterpret_cast<array*>(rel_bias_ptr);
      rel_bias = rb_raw.flags().row_contiguous ? rb_raw : contiguous(rb_raw);
    }
    std::optional<array> rel_bias_w;
    if (rel_bias_w_ptr != nullptr) {
      const auto& rw_raw = *reinterpret_cast<array*>(rel_bias_w_ptr);
      rel_bias_w = (rw_raw.flags().row_contiguous && rw_raw.offset() == 0)
          ? rw_raw
          : contiguous(rw_raw);
    }
    std::optional<array> gate = gate_ptr != nullptr
        ? std::optional<array>(*reinterpret_cast<array*>(gate_ptr))
        : std::nullopt;
    std::optional<array> sink = sink_ptr != nullptr
        ? std::optional<array>(*reinterpret_cast<array*>(sink_ptr))
        : std::nullopt;
    std::optional<array> tree_rows;
    if (tree_rows_ptr != nullptr) {
      const auto& tr_raw = *reinterpret_cast<array*>(tree_rows_ptr);
      tree_rows = (tr_raw.flags().row_contiguous && tr_raw.offset() == 0)
          ? tr_raw
          : contiguous(tr_raw);
    }
    const bool write_lse = (out_lse != nullptr);
    auto results = inkling2_paged_prefill_attention(
        *reinterpret_cast<array*>(q_ptr),
        *reinterpret_cast<array*>(k_pool_ptr),
        *reinterpret_cast<array*>(v_pool_ptr),
        *reinterpret_cast<array*>(k_scale_ptr),
        *reinterpret_cast<array*>(v_scale_ptr),
        *reinterpret_cast<array*>(block_table_ptr),
        *reinterpret_cast<array*>(workspace_ptr),
        rel_bias,
        committed_tokens,
        splits,
        max_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_size,
        sliding_window,
        rel_extent,
        static_cast<KvDtype>(kv_dtype_raw),
        int8_queries,
        scale,
        {},
        rel_bias_w,
        gate,
        sink,
        write_lse,
        tree_rows,
        live_blocks);
    *out_attention = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    if (write_lse) {
      *out_lse = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
      *out_workspace = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
    } else {
      *out_workspace = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    }
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_paged_prefill_attention_forward", e.what());
    return false;
  } catch (...) {
    mlx_trace_native_error("inkling2_paged_prefill_attention_forward", "unknown exception");
    return false;
  }
}

/// Production FFI: emit a `PagedKVWrite` MLX Custom primitive and return the
/// dependency-carrying K/V pool output arrays. The returned arrays share the
/// same Metal buffers as the input pools, but they also carry the MLX graph edge
/// from `new_k/new_v` -> cache write. Rust callers must feed these arrays into
/// subsequent MLX paged-attention reads, or explicitly eval them before raw
/// Metal readers touch the pool.
///
/// Returns false on bridge/factory validation errors. GPU dispatch errors still
/// surface later when MLX evaluates the returned arrays.
bool mlx_paged_kv_write_forward(
    mlx_array* k_pool_ptr,
    mlx_array* v_pool_ptr,
    mlx_array* new_k_ptr,
    mlx_array* new_v_ptr,
    mlx_array* slot_mapping_ptr,
    mlx_array* k_scale_ptr,
    mlx_array* v_scale_ptr,
    int block_size,
    int num_kv_heads,
    int head_size,
    uint8_t kv_dtype_raw,
    bool values_token_major,
    mlx_array** out_k_pool_ptr,
    mlx_array** out_v_pool_ptr,
    mlx_array** out_k_scale_ptr,
    mlx_array** out_v_scale_ptr) {
  for (mlx_array** out :
       {out_k_pool_ptr, out_v_pool_ptr, out_k_scale_ptr, out_v_scale_ptr}) {
    if (out) {
      *out = nullptr;
    }
  }
  if (!k_pool_ptr || !v_pool_ptr || !new_k_ptr || !new_v_ptr ||
      !slot_mapping_ptr || !k_scale_ptr || !v_scale_ptr || !out_k_pool_ptr ||
      !out_v_pool_ptr || !out_k_scale_ptr || !out_v_scale_ptr) {
    return false;
  }

  try {
    using namespace mlx::core;
    using namespace mlx::core::fast;

    auto& k_pool       = *reinterpret_cast<array*>(k_pool_ptr);
    auto& v_pool       = *reinterpret_cast<array*>(v_pool_ptr);
    auto& new_k_raw    = *reinterpret_cast<array*>(new_k_ptr);
    auto& new_v_raw    = *reinterpret_cast<array*>(new_v_ptr);
    auto& slot_mapping = *reinterpret_cast<array*>(slot_mapping_ptr);
    auto& k_scale      = *reinterpret_cast<array*>(k_scale_ptr);
    auto& v_scale      = *reinterpret_cast<array*>(v_scale_ptr);

    require_known_kv_dtype(kv_dtype_raw);
    auto kv_dtype = static_cast<KvDtype>(kv_dtype_raw);

    // Rust model code commonly produces K/V via transpose/reshape chains.
    // Keep these as lazy MLX `contiguous(...)` nodes instead of extracting
    // Metal buffers on the Rust side; this is the point of the bridge.
    auto new_k = contiguous(new_k_raw);
    auto new_v = contiguous(new_v_raw);

    auto out = paged_kv_write(
        k_pool,
        v_pool,
        new_k,
        new_v,
        slot_mapping,
        k_scale,
        v_scale,
        block_size,
        num_kv_heads,
        head_size,
        values_token_major || k_pool.ndim() == 4 ? head_size
                                                 : x_pack_for(kv_dtype),
        kv_dtype,
        StreamOrDevice{},
        values_token_major);

    *out_k_pool_ptr = reinterpret_cast<mlx_array*>(new array(std::move(out[0])));
    *out_v_pool_ptr = reinterpret_cast<mlx_array*>(new array(std::move(out[1])));
    *out_k_scale_ptr =
        reinterpret_cast<mlx_array*>(new array(std::move(out[2])));
    *out_v_scale_ptr =
        reinterpret_cast<mlx_array*>(new array(std::move(out[3])));
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("paged_kv_write_forward", e.what());
    return false;
  } catch (...) {
    mlx_trace_native_error("paged_kv_write_forward", "unknown exception");
    return false;
  }
}

} // extern "C"
