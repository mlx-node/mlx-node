// C++ implementation of the paged-attention kernel dispatch. Encodes
// kernels onto MLX's `metal::CommandEncoder` so dependency tracking is
// correct without manual synchronization.
//
// Mirrors the Rust dispatcher in
// `crates/mlx-paged-attn/src/metal/{state,reshape_and_cache,paged_attention}.rs`
// kernel-name format, threadgroup-memory math, V1/V2 selection, V2
// auxiliary-buffer allocation. Any divergence between the Rust and C++
// paths is a bug — both sides invoke the same Metal kernels, just
// dispatched against different command queues.

#include "mlx_paged_dispatch.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/dtype.h"
#include "mlx/transforms.h"

namespace mlx::core::fast::paged {

namespace {

// Partition size for V2 kernel (mirrors `PARTITION_SIZE` constant in
// `crates/mlx-paged-attn/src/metal/paged_attention.rs`).
constexpr uint32_t kPartitionSize = 512;

// The first actual context in the auto route's 92K-ending bucket. A
// fixed-session A/B measured Hq16/Hkv2 at 91,795 context for 512 generated
// tokens: 128 stripes delivered 32.8416 tok/s versus 31.0484 for 64 (+5.78%,
// 3/3 paired wins). The raw 112K sweep showed no regression. Keep this geometry
// threshold mirrored in the Rust raw dispatcher and the model-free planner
// diagnostics.
constexpr int kD512Hq16Hkv2WideStripeContext = 88 * 1024 + 1;

// `NUM_THREADS` and `NUM_WARPS` baked into the forked kernels. These
// must agree with the `_nt256_nsl32` suffix in the kernel-name format
// (see `crates/mlx-paged-attn/src/metal/state.rs`).
constexpr int kNumThreads = 256;
constexpr int kNumSimdLanes = 32;
constexpr int kNumWarps = kNumThreads / kNumSimdLanes;

// =============================================================================
// .metallib loading
// =============================================================================
//
// The paged-attention kernels live in their own `.metallib`, separate
// from MLX's `mlx.metallib`. mlx-sys/build.rs compiles
// `crates/mlx-paged-attn/metal/*.metal` into
// `<OUT_DIR>/paged_attn.metallib` and copies it next to the
// crate-internal `mlx.metallib` so it ships in the same place.
//
// At runtime we call `Device::get_library(name, path)` once per
// process; subsequent calls hit MLX's library cache. We use the
// `path` overload (not the `builder` overload) so we feed Metal a
// pre-compiled `.metallib` rather than re-compiling source at
// runtime.

const std::string kPagedAttnLibraryName = "mlx_paged_attn";

// Resolve the path of the binary that contains this function, and look
// for `paged_attn.metallib` next to it. Mirrors the colocated-library
// search in `mlx/backend/metal/device.cpp::load_colocated_library`,
// with additional fallbacks for cargo test binaries living at
// target/<profile>/deps/ (the build script copies the metallib up one
// level to target/<profile>/, which is the parent of `deps/`).
std::filesystem::path paged_attn_metallib_path() {
  // Highest-priority override: an explicit env var (used by
  // tooling / tests that bundle the metallib at a custom path).
  if (const char* env_path = std::getenv("MLX_PAGED_ATTN_METALLIB")) {
    std::filesystem::path p(env_path);
    if (std::filesystem::exists(p)) {
      return p;
    }
  }

  Dl_info info;
  if (!dladdr(reinterpret_cast<void*>(&paged_attn_metallib_path), &info)) {
    throw std::runtime_error(
        "[mlx_paged_dispatch] dladdr failed; cannot locate "
        "paged_attn.metallib");
  }
  std::filesystem::path bin_dir =
      std::filesystem::path(info.dli_fname).parent_path();
  // Search candidates in order of preference. The first existing
  // path wins.
  std::vector<std::filesystem::path> candidates = {
      bin_dir / "paged_attn.metallib",
      // SwiftPM-style Resources subfolder.
      bin_dir / "Resources" / "paged_attn.metallib",
      // Cargo test binary at target/<profile>/deps/ — metallib copied
      // by build.rs to the parent target/<profile>/.
      bin_dir.parent_path() / "paged_attn.metallib",
  };
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  std::ostringstream msg;
  msg << "[mlx_paged_dispatch] paged_attn.metallib not found near binary "
      << bin_dir.string()
      << "; expected one of: 'paged_attn.metallib', "
         "'Resources/paged_attn.metallib', or in the parent directory. "
         "You can override with the MLX_PAGED_ATTN_METALLIB env var.";
  throw std::runtime_error(msg.str());
}

MTL::Library* get_paged_attn_library(mlx::core::metal::Device& device) {
  // `Device::get_library(name, path)` caches by name — first call
  // loads from `path`, subsequent calls hit the cache and ignore
  // `path`. This matches MLX's own colocated-library pattern.
  static std::once_flag resolve_path_once;
  static std::filesystem::path cached_path;
  std::call_once(resolve_path_once, []() {
    cached_path = paged_attn_metallib_path();
  });
  return device.get_library(kPagedAttnLibraryName, cached_path.string());
}

// =============================================================================
// Kernel-name formatting (must match `MetalState::*_kernel_name` in
// `crates/mlx-paged-attn/src/metal/state.rs` byte-for-byte)
// =============================================================================

const char* dtype_string(KvDtype d) {
  switch (d) {
    case KvDtype::Fp16:
      return "half";
    case KvDtype::Bf16:
      return "bfloat16_t";
    case KvDtype::Fp8:
      return "uchar";
  }
  // Unreachable — quiet warning.
  return "half";
}

// Map io dtype paired with cache dtype: io == cache for non-FP8,
// io = bfloat16 for FP8.
KvDtype io_dtype_for(KvDtype cache) {
  return cache == KvDtype::Fp8 ? KvDtype::Bf16 : cache;
}

mlx::core::Dtype mlx_dtype_for(KvDtype d) {
  switch (d) {
    case KvDtype::Fp16:
      return mlx::core::float16;
    case KvDtype::Bf16:
      return mlx::core::bfloat16;
    case KvDtype::Fp8:
      // FP8 is stored opaquely as bytes; the cache buffer is uint8 in
      // MLX. The kernel reinterprets via `to_cache<KV_T, uchar>`. For
      // io we never see Fp8 (io is bf16).
      return mlx::core::uint8;
  }
  return mlx::core::bfloat16;
}

size_t dtype_byte_size(KvDtype d) {
  switch (d) {
    case KvDtype::Fp16:
    case KvDtype::Bf16:
      return 2;
    case KvDtype::Fp8:
      return 1;
  }
  return 2;
}

// -----------------------------------------------------------------------
// Kernel-name memoization.
//
// Every name-builder function below runs on every PagedKVWrite /
// PagedAttention dispatch (per attention layer, per decode token /
// prefill chunk) purely to look the resulting string up in MLX's own
// `Device::get_kernel` cache (see `load_pipeline` below). The name is
// fully determined by a handful of compile-time-fixed-per-model
// parameters (dtype, head_size, block_size, alibi flag) that never
// change between calls for a loaded model, so re-running the
// `std::ostringstream` formatting on every call is pure repeated work.
// Memoize on those parameters instead — same idea as the
// `get_or_create_kernel` cache in `mlx_gated_delta.cpp`.
//
// The cache key is a `std::tuple` of the exact parameter values, so
// each field is compared independently at its full width via
// `std::tuple`'s built-in `operator<`. Distinct parameter tuples can
// therefore never alias onto the same key regardless of magnitude —
// there is no bit-packing, so an out-of-range field can never overflow
// into a neighbouring field and collide with another valid key. Using
// `std::map` gives us this field-wise ordering for free (no custom hash
// or comparator needed); its O(log n) lookup is irrelevant here since
// each cache holds only a handful of entries and is touched on the
// cold, first-per-tuple path only.
template <typename Key, typename Builder>
const std::string& memoized_kernel_name(
    std::map<Key, std::string>& cache,
    std::mutex& mutex,
    const Key& key,
    Builder&& build) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  // `std::map`, like `std::unordered_map`, is node-based and never
  // invalidates references to existing elements on insertion (only
  // iterators), so returning a reference into the map after `mutex` is
  // released below is safe.
  return cache.emplace(key, build()).first->second;
}

const std::string& reshape_and_cache_kernel_name(
    KvDtype input_dtype,
    KvDtype cache_dtype,
    bool use_fp8) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, KvDtype, bool>, std::string> cache;
  return memoized_kernel_name(
      cache, mutex, std::make_tuple(input_dtype, cache_dtype, use_fp8), [&] {
    std::ostringstream os;
    os << "reshape_and_cache_kv_" << dtype_string(input_dtype) << "_cache_"
       << dtype_string(cache_dtype);
    if (use_fp8) {
      os << "_fp8";
    }
    return os.str();
  });
}

const std::string& paged_attention_v1_kernel_name(
    KvDtype io_dtype,
    KvDtype cache_dtype,
    int head_size,
    int block_size,
    bool use_alibi) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, KvDtype, int, int, bool>, std::string>
      cache;
  return memoized_kernel_name(
      cache,
      mutex,
      std::make_tuple(io_dtype, cache_dtype, head_size, block_size, use_alibi),
      [&] {
    std::ostringstream os;
    os << "paged_attention_" << dtype_string(io_dtype) << "_cache_"
       << dtype_string(cache_dtype) << "_hs" << head_size << "_bs"
       << block_size << "_nt" << kNumThreads << "_nsl" << kNumSimdLanes
       << "_ps0";
    if (use_alibi) {
      os << "_alibi";
    }
    return os.str();
  });
}

const std::string& paged_attention_v2_kernel_name(
    KvDtype io_dtype,
    KvDtype cache_dtype,
    int head_size,
    int block_size,
    bool use_alibi) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, KvDtype, int, int, bool>, std::string>
      cache;
  return memoized_kernel_name(
      cache,
      mutex,
      std::make_tuple(io_dtype, cache_dtype, head_size, block_size, use_alibi),
      [&] {
    std::ostringstream os;
    os << "paged_attention_" << dtype_string(io_dtype) << "_cache_"
       << dtype_string(cache_dtype) << "_hs" << head_size << "_bs"
       << block_size << "_nt" << kNumThreads << "_nsl" << kNumSimdLanes
       << "_ps" << kPartitionSize;
    if (use_alibi) {
      os << "_alibi";
    }
    return os.str();
  });
}

const std::string& paged_attention_v2_reduce_kernel_name(
    KvDtype io_dtype,
    int head_size) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, int>, std::string> cache;
  return memoized_kernel_name(
      cache, mutex, std::make_tuple(io_dtype, head_size), [&] {
    std::ostringstream os;
    os << "paged_attention_v2_reduce_" << dtype_string(io_dtype) << "_hs"
       << head_size << "_nt" << kNumThreads << "_nsl" << kNumSimdLanes
       << "_ps" << kPartitionSize;
    return os.str();
  });
}

enum class GroupedPagedAttentionKind {
  None,
  Qwen35D256,
  D512Direct,
};

enum class GroupedD512Mode {
  Disabled,
  Auto,
  Force,
};

const std::string& paged_attention_grouped_kernel_name(
    GroupedPagedAttentionKind kind) {
  static const std::string qwen35 =
      "paged_attention_grouped_bfloat16_hs256_bs16_striped";
  static const std::string gemma4 =
      "paged_attention_grouped_bfloat16_hs512_bs16_striped";
  switch (kind) {
    case GroupedPagedAttentionKind::Qwen35D256:
      return qwen35;
    case GroupedPagedAttentionKind::D512Direct:
      return gemma4;
    case GroupedPagedAttentionKind::None:
      throw std::logic_error("grouped paged-attention kernel requested for None");
  }
  throw std::logic_error("invalid grouped paged-attention kind");
}

const std::string& paged_attention_grouped_reduce_kernel_name(
    GroupedPagedAttentionKind kind) {
  static const std::string qwen35 =
      "paged_attention_grouped_bfloat16_hs256_striped_reduce";
  static const std::string gemma4 =
      "paged_attention_grouped_bfloat16_hs512_striped_reduce";
  switch (kind) {
    case GroupedPagedAttentionKind::Qwen35D256:
      return qwen35;
    case GroupedPagedAttentionKind::D512Direct:
      return gemma4;
    case GroupedPagedAttentionKind::None:
      throw std::logic_error("grouped paged-attention reducer requested for None");
  }
  throw std::logic_error("invalid grouped paged-attention kind");
}

uint32_t grouped_d512_stripe_override() {
  // Diagnostic-only A/B knob. The D512 reducer supports these power-of-two
  // stripe counts, including counts below one 32-SIMD reducer group. Ignore
  // malformed values rather than risking an invalid auxiliary layout.
  static const uint32_t stripes = []() {
    const char* value = std::getenv("MLX_PAGED_GROUPED_D512_STRIPES");
    if (value == nullptr) {
      // Compatibility alias retained for existing Gemma 4 A/B scripts.
      value = std::getenv("MLX_PAGED_GROUPED_GEMMA4_STRIPES");
    }
    if (value == nullptr) {
      return uint32_t{0};
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0') {
      return uint32_t{0};
    }
    switch (parsed) {
      case 4:
      case 8:
      case 16:
      case 32:
      case 64:
      case 128:
      case 256:
        return static_cast<uint32_t>(parsed);
      default:
        return uint32_t{0};
    }
  }();
  return stripes;
}

uint32_t grouped_d512_default_stripe_count(
    int max_context_len,
    int num_q_heads,
    int num_kv_heads) {
  // Keep total stage-1 threadgroups approximately stable across the shipped
  // D512 GQA layouts. The grid already has one x-dimension group per KV head,
  // so using the single-KV 32/64/128 policy unchanged would over-dispatch
  // Hkv2/Hkv4 by 2x/4x.
  uint32_t base_stripes = 128;
  if (max_context_len <= 4096) {
    base_stripes = 32;
  } else if (max_context_len <= 8192) {
    base_stripes = 64;
  }
  const uint32_t kv_heads =
      static_cast<uint32_t>(std::max(num_kv_heads, 1));
  const uint32_t stripes = std::max(uint32_t{4}, base_stripes / kv_heads);
  if (num_q_heads == 16 && num_kv_heads == 2 &&
      max_context_len >= kD512Hq16Hkv2WideStripeContext) {
    return 128;
  }
  return stripes;
}

uint32_t grouped_d512_resolved_stripe_count(
    uint32_t override_stripes,
    int max_context_len,
    int num_q_heads,
    int num_kv_heads) {
  return override_stripes != 0
      ? override_stripes
      : grouped_d512_default_stripe_count(
            max_context_len, num_q_heads, num_kv_heads);
}

uint32_t grouped_stripe_count(
    GroupedPagedAttentionKind kind,
    int max_context_len,
    int num_q_heads,
    int num_kv_heads) {
  if (kind == GroupedPagedAttentionKind::D512Direct) {
    return grouped_d512_resolved_stripe_count(
        grouped_d512_stripe_override(),
        max_context_len,
        num_q_heads,
        num_kv_heads);
  }
  // Power-of-two, block-size-aligned stripe counts. Representative long
  // contexts mirror MLX's vector 2-pass occupancy curve: 16K -> 256,
  // 64K -> 512, and >64K -> 1024. Smaller contexts avoid over-dispatch.
  if (max_context_len <= 4096) {
    return 32;
  }
  if (max_context_len <= 8192) {
    return 64;
  }
  if (max_context_len < 16384) {
    return 128;
  }
  if (max_context_len <= 32768) {
    return 256;
  }
  if (max_context_len <= 65536) {
    return 512;
  }
  return 1024;
}

bool grouped_qwen35_paged_attention_enabled() {
  // Default-on escape hatch for A/B profiling and driver-specific rollback.
  // Cache once: dispatch is on the token hot path and must not call getenv for
  // every attention layer/token.
  static const bool enabled = []() {
    const char* value = std::getenv("MLX_PAGED_GROUPED_QWEN35");
    return value == nullptr || std::string(value) != "0";
  }();
  return enabled;
}

GroupedD512Mode grouped_d512_paged_attention_mode() {
  // Default-off until a model-level A/B establishes the crossover for each
  // supported runtime geometry. `force` remains the correctness/benchmark
  // override. Cache once because this is on the decode hot path.
  // Cache once because this runs on every global layer and decode token.
  static const GroupedD512Mode mode = []() {
    const char* value = std::getenv("MLX_PAGED_GROUPED_D512");
    if (value == nullptr) {
      // Compatibility alias retained for existing Gemma 4 deployments.
      value = std::getenv("MLX_PAGED_GROUPED_GEMMA4");
    }
    if (value == nullptr) {
      return GroupedD512Mode::Disabled;
    }
    const std::string setting(value);
    if (setting == "1" || setting == "on" || setting == "auto" ||
        setting == "true") {
      return GroupedD512Mode::Auto;
    }
    if (setting == "force") {
      return GroupedD512Mode::Force;
    }
    return GroupedD512Mode::Disabled;
  }();
  return mode;
}

bool grouped_qwen35_shape_matches(
    bool enabled,
    KvDtype io_dtype,
    KvDtype cache_dtype,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int query_rows,
    int max_context_len) {
  // Model-free A/B break-even on the target Apple GPU. GQA8 needs a more
  // conservative cutoff than dense GQA6; the two-row verifier benefits sooner
  // because the generic path repeats more partition work.
  const bool dense_heads = num_q_heads == 24 && num_kv_heads == 4;
  const bool moe_heads = num_q_heads == 16 && num_kv_heads == 2;
  const int min_context = query_rows == 1
      ? (moe_heads ? 32768 : 16384)
      : (moe_heads ? 16384 : 8192);
  return enabled &&
      io_dtype == KvDtype::Bf16 && cache_dtype == KvDtype::Bf16 &&
      num_seqs == 1 && (dense_heads || moe_heads) &&
      head_size == 256 && block_size == 16 &&
      (query_rows == 1 || query_rows == 2) &&
      max_context_len >= min_context;
}

bool use_grouped_qwen35_paged_attention(
    KvDtype io_dtype,
    KvDtype cache_dtype,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int query_rows,
    int max_context_len) {
  return grouped_qwen35_shape_matches(
      grouped_qwen35_paged_attention_enabled(),
      io_dtype,
      cache_dtype,
      num_seqs,
      num_q_heads,
      num_kv_heads,
      head_size,
      block_size,
      query_rows,
      max_context_len);
}

bool grouped_d512_shape_matches(
    GroupedD512Mode mode,
    KvDtype io_dtype,
    KvDtype cache_dtype,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int query_rows,
    int max_context_len) {
  const bool supported_heads =
      (num_q_heads == 8 && num_kv_heads == 1) ||
      (num_q_heads == 16 && num_kv_heads == 1) ||
      (num_q_heads == 16 && num_kv_heads == 2) ||
      (num_q_heads == 32 && num_kv_heads == 4);
  const bool exact_shape =
      io_dtype == KvDtype::Bf16 && cache_dtype == KvDtype::Bf16 &&
      num_seqs == 1 && supported_heads &&
      head_size == 512 && block_size == 16 && query_rows == 1 &&
      max_context_len > static_cast<int>(kPartitionSize);
  if (!exact_shape || mode == GroupedD512Mode::Disabled) {
    return false;
  }
  if (mode == GroupedD512Mode::Force) {
    return true;
  }
  // Preserve the former environment-Auto surface exactly. New geometries are
  // enabled only by the measured per-dispatch route policy.
  return num_q_heads == 16 && num_kv_heads == 1 &&
      max_context_len >= 3072 && max_context_len <= 16384;
}

GroupedPagedAttentionKind select_grouped_paged_attention(
    KvDtype io_dtype,
    KvDtype cache_dtype,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int query_rows,
    int max_context_len,
    PagedAttentionRouteHint route_hint =
        PagedAttentionRouteHint::Auto) {
  if (route_hint == PagedAttentionRouteHint::ForceGeneric) {
    return GroupedPagedAttentionKind::None;
  }
  if (route_hint == PagedAttentionRouteHint::ForceD512Staged) {
    return grouped_d512_shape_matches(
               GroupedD512Mode::Force,
               io_dtype,
               cache_dtype,
               num_seqs,
               num_q_heads,
               num_kv_heads,
               head_size,
               block_size,
               query_rows,
               max_context_len)
        ? GroupedPagedAttentionKind::D512Direct
        : GroupedPagedAttentionKind::None;
  }
  if (use_grouped_qwen35_paged_attention(
          io_dtype,
          cache_dtype,
          num_seqs,
          num_q_heads,
          num_kv_heads,
          head_size,
          block_size,
          query_rows,
          max_context_len)) {
    return GroupedPagedAttentionKind::Qwen35D256;
  }
  if (grouped_d512_shape_matches(
          grouped_d512_paged_attention_mode(),
          io_dtype,
          cache_dtype,
          num_seqs,
          num_q_heads,
          num_kv_heads,
          head_size,
          block_size,
          query_rows,
          max_context_len)) {
    return GroupedPagedAttentionKind::D512Direct;
  }
  return GroupedPagedAttentionKind::None;
}

const std::string& paged_attention_varlen_v1_kernel_name(
    KvDtype io_dtype,
    KvDtype cache_dtype,
    int head_size,
    int block_size,
    bool use_alibi) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, KvDtype, int, int, bool>, std::string>
      cache;
  return memoized_kernel_name(
      cache,
      mutex,
      std::make_tuple(io_dtype, cache_dtype, head_size, block_size, use_alibi),
      [&] {
    std::ostringstream os;
    os << "paged_attention_varlen_" << dtype_string(io_dtype) << "_cache_"
       << dtype_string(cache_dtype) << "_hs" << head_size << "_bs"
       << block_size << "_nt" << kNumThreads << "_nsl" << kNumSimdLanes
       << "_ps0";
    if (use_alibi) {
      os << "_alibi";
    }
    return os.str();
  });
}

const std::string& paged_attention_varlen_v2_kernel_name(
    KvDtype io_dtype,
    KvDtype cache_dtype,
    int head_size,
    int block_size,
    bool use_alibi) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, KvDtype, int, int, bool>, std::string>
      cache;
  return memoized_kernel_name(
      cache,
      mutex,
      std::make_tuple(io_dtype, cache_dtype, head_size, block_size, use_alibi),
      [&] {
    std::ostringstream os;
    os << "paged_attention_varlen_" << dtype_string(io_dtype) << "_cache_"
       << dtype_string(cache_dtype) << "_hs" << head_size << "_bs"
       << block_size << "_nt" << kNumThreads << "_nsl" << kNumSimdLanes
       << "_ps" << kPartitionSize;
    if (use_alibi) {
      os << "_alibi";
    }
    return os.str();
  });
}

const std::string& paged_attention_varlen_v2_reduce_kernel_name(
    KvDtype io_dtype,
    int head_size) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, int>, std::string> cache;
  return memoized_kernel_name(
      cache, mutex, std::make_tuple(io_dtype, head_size), [&] {
    std::ostringstream os;
    os << "paged_attention_varlen_v2_reduce_" << dtype_string(io_dtype)
       << "_hs" << head_size << "_nt" << kNumThreads << "_nsl"
       << kNumSimdLanes << "_ps" << kPartitionSize;
    return os.str();
  });
}

// =============================================================================
// Pipeline cache helper: load+cache a compute pipeline by kernel name.
// Wraps `Device::get_library` + `Device::get_kernel`. Both layers cache
// internally, so repeated calls with the same kernel_name are O(1).
// =============================================================================

MTL::ComputePipelineState* load_pipeline(
    mlx::core::metal::Device& device,
    const std::string& kernel_name) {
  MTL::Library* lib = get_paged_attn_library(device);
  // `get_kernel(base_name, mtl_lib)` in `device.h` is the lookup by
  // name only; it caches by `(library, base_name)`.
  return device.get_kernel(kernel_name, lib);
}

struct GroupedPipelineLimits {
  size_t stage_threads;
  size_t reduce_threads;
  bool available;
};

GroupedPipelineLimits load_grouped_pipeline_limits(
    mlx::core::metal::Device& device,
    GroupedPagedAttentionKind kind) noexcept {
  try {
    auto* stage = load_pipeline(device, paged_attention_grouped_kernel_name(kind));
    auto* reduce =
        load_pipeline(device, paged_attention_grouped_reduce_kernel_name(kind));
    if (stage == nullptr || reduce == nullptr) {
      std::fprintf(
          stderr,
          "[mlx][warn] grouped paged-attention pipeline lookup returned null; "
          "using generic V2\n");
      return GroupedPipelineLimits{0, 0, false};
    }
    return GroupedPipelineLimits{
        static_cast<size_t>(stage->maxTotalThreadsPerThreadgroup()),
        static_cast<size_t>(reduce->maxTotalThreadsPerThreadgroup()),
        true};
  } catch (const std::exception& error) {
    std::fprintf(
        stderr,
        "[mlx][warn] grouped paged-attention pipelines unavailable (%s); "
        "using generic V2\n",
        error.what());
    return GroupedPipelineLimits{0, 0, false};
  } catch (...) {
    std::fprintf(
        stderr,
        "[mlx][warn] grouped paged-attention pipelines unavailable; using "
        "generic V2\n");
    return GroupedPipelineLimits{0, 0, false};
  }
}

bool grouped_pipelines_supported(
    mlx::core::metal::Device& device,
    GroupedPagedAttentionKind kind,
    int num_q_heads,
    int num_kv_heads) {
  // There is one active Metal device. Cache capability independently for the
  // two concrete instantiations so an unavailable experimental Gemma pipeline
  // cannot disable the established Qwen path (or vice versa).
  const GroupedPipelineLimits* limits_ptr = nullptr;
  if (kind == GroupedPagedAttentionKind::Qwen35D256) {
    static const GroupedPipelineLimits qwen35_limits =
        load_grouped_pipeline_limits(device, kind);
    limits_ptr = &qwen35_limits;
  } else if (kind == GroupedPagedAttentionKind::D512Direct) {
    static const GroupedPipelineLimits d512_limits =
        load_grouped_pipeline_limits(device, kind);
    limits_ptr = &d512_limits;
  } else {
    return false;
  }
  const GroupedPipelineLimits& limits = *limits_ptr;
  if (!limits.available || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0) {
    return false;
  }
  const size_t required_stage_threads =
      static_cast<size_t>(32 * (num_q_heads / num_kv_heads));
  const bool supported =
      limits.stage_threads >= required_stage_threads &&
      limits.reduce_threads >= 1024;
  if (!supported) {
    static std::once_flag qwen_warning_once;
    static std::once_flag gemma_warning_once;
    std::once_flag& warning_once =
        kind == GroupedPagedAttentionKind::D512Direct
        ? gemma_warning_once
        : qwen_warning_once;
    std::call_once(warning_once, [&]() {
      std::fprintf(
          stderr,
          "[mlx][warn] grouped paged-attention threadgroup limits "
          "unsupported (stage=%lu, required_stage=%lu, reducer=%lu); "
          "using generic V2\n",
          static_cast<unsigned long>(limits.stage_threads),
          static_cast<unsigned long>(required_stage_threads),
          static_cast<unsigned long>(limits.reduce_threads));
    });
  }
  return supported;
}

// Test-only route observation. Production dispatch pays no atomic cost: the
// counter is touched only when the cached opt-in environment flag is exactly
// `1`. This lets integrated graph tests prove that numerical parity did not
// silently pass through generic V2.
std::atomic<uint64_t>& grouped_qwen35_test_probe_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

bool grouped_qwen35_test_probe_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("MLX_PAGED_GROUPED_QWEN35_TEST_PROBE");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

void record_grouped_qwen35_route_for_test() {
  if (grouped_qwen35_test_probe_enabled()) {
    grouped_qwen35_test_probe_counter().fetch_add(1, std::memory_order_relaxed);
  }
}

std::atomic<uint64_t>& grouped_d512_test_probe_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

bool grouped_d512_test_probe_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("MLX_PAGED_GROUPED_D512_TEST_PROBE");
    if (value == nullptr) {
      value = std::getenv("MLX_PAGED_GROUPED_GEMMA4_TEST_PROBE");
    }
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

void record_grouped_route_for_test(GroupedPagedAttentionKind kind) {
  if (kind == GroupedPagedAttentionKind::Qwen35D256) {
    record_grouped_qwen35_route_for_test();
  } else if (kind == GroupedPagedAttentionKind::D512Direct &&
             grouped_d512_test_probe_enabled()) {
    grouped_d512_test_probe_counter().fetch_add(1, std::memory_order_relaxed);
  }
}

} // namespace

extern "C" void mlx_paged_grouped_qwen35_test_probe_reset() {
  grouped_qwen35_test_probe_counter().store(0, std::memory_order_relaxed);
}

extern "C" uint64_t mlx_paged_grouped_qwen35_test_probe_count() {
  return grouped_qwen35_test_probe_counter().load(std::memory_order_relaxed);
}

extern "C" void mlx_paged_grouped_d512_test_probe_reset() {
  grouped_d512_test_probe_counter().store(0, std::memory_order_relaxed);
}

extern "C" uint64_t mlx_paged_grouped_d512_test_probe_count() {
  return grouped_d512_test_probe_counter().load(std::memory_order_relaxed);
}

extern "C" void mlx_paged_grouped_gemma4_test_probe_reset() {
  mlx_paged_grouped_d512_test_probe_reset();
}

extern "C" uint64_t mlx_paged_grouped_gemma4_test_probe_count() {
  return mlx_paged_grouped_d512_test_probe_count();
}

extern "C" int mlx_paged_grouped_d512_capability(
    int num_q_heads,
    int num_kv_heads) {
  const bool supported_heads =
      (num_q_heads == 8 && num_kv_heads == 1) ||
      (num_q_heads == 16 && num_kv_heads == 1) ||
      (num_q_heads == 16 && num_kv_heads == 2) ||
      (num_q_heads == 32 && num_kv_heads == 4);
  if (!supported_heads) {
    return 0;
  }
  try {
    auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto& device = mlx::core::metal::device(stream.device);
    return grouped_pipelines_supported(
               device,
               GroupedPagedAttentionKind::D512Direct,
               num_q_heads,
               num_kv_heads)
        ? 1
        : 0;
  } catch (...) {
    return -1;
  }
}

extern "C" int mlx_paged_grouped_qwen35_shape_guard_for_test(
    int num_q_heads,
    int num_kv_heads,
    int query_rows,
    int max_context_len) {
  return grouped_qwen35_shape_matches(
      /*enabled=*/true,
      KvDtype::Bf16,
      KvDtype::Bf16,
      /*num_seqs=*/1,
      num_q_heads,
      num_kv_heads,
      /*head_size=*/256,
      /*block_size=*/16,
      query_rows,
      max_context_len)
      ? 1
      : 0;
}

extern "C" int mlx_paged_grouped_d512_shape_guard_for_test(
    int selector_mode,
    int num_q_heads,
    int num_kv_heads,
    int query_rows,
    int max_context_len) {
  const GroupedD512Mode mode = selector_mode == 2
      ? GroupedD512Mode::Force
      : (selector_mode == 1
             ? GroupedD512Mode::Auto
             : GroupedD512Mode::Disabled);
  return grouped_d512_shape_matches(
      mode,
      KvDtype::Bf16,
      KvDtype::Bf16,
      /*num_seqs=*/1,
      num_q_heads,
      num_kv_heads,
      /*head_size=*/512,
      /*block_size=*/16,
      query_rows,
      max_context_len)
      ? 1
      : 0;
}

extern "C" uint32_t mlx_paged_grouped_d512_stripe_count_for_test(
    int num_q_heads,
    int num_kv_heads,
    int max_context_len,
    uint32_t override_stripes) {
  return grouped_d512_resolved_stripe_count(
      override_stripes, max_context_len, num_q_heads, num_kv_heads);
}

extern "C" int mlx_paged_grouped_gemma4_shape_guard_for_test(
    int selector_mode,
    int query_rows,
    int max_context_len) {
  return mlx_paged_grouped_d512_shape_guard_for_test(
      selector_mode,
      /*num_q_heads=*/16,
      /*num_kv_heads=*/1,
      query_rows,
      max_context_len);
}

// =============================================================================
// dispatch_reshape_and_cache
// =============================================================================
void dispatch_reshape_and_cache(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    const mlx::core::array& new_k,
    const mlx::core::array& new_v,
    mlx::core::array& k_pool,
    mlx::core::array& v_pool,
    const mlx::core::array& slot_mapping,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int block_size,
    int x_pack,
    KvDtype kv_dtype) {
  if (num_tokens == 0) {
    // No-op write — avoid encoding a zero-grid dispatch which Metal
    // would refuse anyway.
    return;
  }

  // io dtype == cache dtype for non-FP8, BF16 for FP8. The Rust
  // dispatcher derives this; we mirror so the kernel name matches what
  // the metallib instantiated.
  const KvDtype cache_dtype = kv_dtype;
  const KvDtype input_dtype =
      cache_dtype == KvDtype::Fp8 ? KvDtype::Bf16 : cache_dtype;
  const bool use_fp8 = cache_dtype == KvDtype::Fp8;

  const std::string& kernel_name =
      reshape_and_cache_kernel_name(input_dtype, cache_dtype, use_fp8);
  MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
  encoder.set_compute_pipeline_state(pipeline);

  // Compute strides as the Rust path does.
  const int32_t key_stride = num_kv_heads * head_size;
  const int32_t value_stride = key_stride;

  // Set inputs.
  //   buffer(0): new_k         (read-only)
  //   buffer(1): new_v         (read-only)
  //   buffer(2): k_pool        (read-write, in-place)
  //   buffer(3): v_pool        (read-write, in-place)
  //   buffer(4): slot_mapping  (read-only)
  //   buffer(5): k_scale       (read-only, used for FP8)
  //   buffer(6): v_scale       (read-only, used for FP8)
  encoder.set_input_array(new_k, 0);
  encoder.set_input_array(new_v, 1);
  encoder.set_output_array(k_pool, 2);
  encoder.set_output_array(v_pool, 3);
  encoder.set_input_array(slot_mapping, 4);
  encoder.set_input_array(k_scale, 5);
  encoder.set_input_array(v_scale, 6);

  // Constants — kernel takes them as `device const int &` which
  // Metal's `setBytes` can satisfy directly without an extra device
  // buffer. (Rust path allocates a separate small device buffer per
  // constant; we save those allocations by inlining.)
  encoder.set_bytes(key_stride, 7);
  encoder.set_bytes(value_stride, 8);
  encoder.set_bytes<int32_t>(num_kv_heads, 9);
  encoder.set_bytes<int32_t>(head_size, 10);
  encoder.set_bytes<int32_t>(block_size, 11);
  encoder.set_bytes<int32_t>(x_pack, 12);

  // Dispatch: 1 threadgroup per token, kNumThreads threads per
  // threadgroup. Mirrors Rust path.
  MTL::Size group = MTL::Size::Make(kNumThreads, 1, 1);
  MTL::Size grid = MTL::Size::Make(static_cast<size_t>(num_tokens), 1, 1);
  encoder.dispatch_threadgroups(grid, group);
}

// =============================================================================
// dispatch_paged_attention_v1 (helper used by dispatch_paged_attention_auto)
// =============================================================================
namespace {

void dispatch_paged_attention_v1_inner(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    float scale,
    float softcapping,
    int sliding_window,
    KvDtype io_dtype,
    KvDtype cache_dtype) {
  const std::string& kernel_name = paged_attention_v1_kernel_name(
      io_dtype, cache_dtype, head_size, block_size, /*use_alibi=*/false);
  MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
  encoder.set_compute_pipeline_state(pipeline);

  // V1 dispatch buffer layout (matches Rust path):
  //   buffer(0): exp_sums    (unused — pass dummy)
  //   buffer(1): max_logits  (unused — pass dummy)
  //   buffer(2): output
  //   buffer(3): queries
  //   buffer(4): key_cache (k_pool)
  //   buffer(5): value_cache (v_pool)
  //   buffer(6): k_scale
  //   buffer(7): v_scale
  //   buffer(8): num_kv_heads (constant int)
  //   buffer(9): scale (constant float)
  //   buffer(10): softcapping (constant float)
  //   buffer(11): block_tables
  //   buffer(12): context_lens
  //   buffer(13): max_num_blocks_per_seq (constant int)
  //   buffer(14): alibi_slopes (unused)
  //   buffer(15): q_stride (constant int)
  //   buffer(16): kv_block_stride (constant int)
  //   buffer(17): kv_head_stride (constant int)

  // Dummy 4-byte placeholder for unused bindings (exp_sums, max_logits,
  // alibi_slopes). MLX's `set_bytes` can satisfy `device const ..*`
  // bindings via a small inline payload — Metal will not read from
  // these buffers in the V1 (`PARTITION_SIZE = 0`) template
  // instantiation.
  const float dummy_zero = 0.0f;
  encoder.set_bytes(dummy_zero, 0);
  encoder.set_bytes(dummy_zero, 1);
  encoder.set_output_array(out, 2);
  encoder.set_input_array(q, 3);
  encoder.set_input_array(k_pool, 4);
  encoder.set_input_array(v_pool, 5);
  encoder.set_input_array(k_scale, 6);
  encoder.set_input_array(v_scale, 7);

  encoder.set_bytes<int32_t>(num_kv_heads, 8);
  encoder.set_bytes<float>(scale, 9);
  encoder.set_bytes<float>(softcapping, 10);
  encoder.set_input_array(block_table, 11);
  encoder.set_input_array(seq_lens, 12);
  encoder.set_bytes<int32_t>(max_blocks_per_seq, 13);
  // alibi_slopes — not used (use_alibi = false in the kernel
  // instantiation); pass a dummy.
  encoder.set_bytes(dummy_zero, 14);

  const int32_t q_stride = num_q_heads * head_size;
  const int32_t kv_block_stride = num_kv_heads * head_size * block_size;
  const int32_t kv_head_stride = head_size * block_size;
  encoder.set_bytes<int32_t>(q_stride, 15);
  encoder.set_bytes<int32_t>(kv_block_stride, 16);
  encoder.set_bytes<int32_t>(kv_head_stride, 17);

  // Sliding-window mask. 0 = full context (default).
  encoder.set_bytes<int32_t>(sliding_window, 18);

  // Threadgroup memory math: same dual-purpose layout as Rust V1. The
  // buffer serves two kernel stages and is sized to the max:
  //   logits stage: logits[max_seq_len] f32 + red_smem[2*NUM_WARPS] f32
  //   v-reduce stage: out_smem[(NUM_WARPS/2)*head_size] f32 + same red_smem
  const size_t logits_bytes =
      static_cast<size_t>(max_context_len) * sizeof(float);
  const size_t v_reduce_bytes =
      static_cast<size_t>(kNumWarps / 2) *
      static_cast<size_t>(head_size) * sizeof(float);
  const size_t red_smem_bytes =
      2 * static_cast<size_t>(kNumWarps) * sizeof(float);
  const size_t threadgroup_mem =
      std::max(logits_bytes, v_reduce_bytes) + red_smem_bytes;
  encoder.set_threadgroup_memory_length(threadgroup_mem, 0);

  // Dispatch: (num_q_heads, num_seqs, 1) threadgroups, kNumThreads each.
  MTL::Size group = MTL::Size::Make(kNumThreads, 1, 1);
  MTL::Size grid = MTL::Size::Make(
      static_cast<size_t>(num_q_heads),
      static_cast<size_t>(num_seqs),
      1);
  encoder.dispatch_threadgroups(grid, group);

  // Reference unused parameter so the compiler doesn't warn.
  (void)stream;
}

void dispatch_paged_attention_v2_inner(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    float scale,
    float softcapping,
    int sliding_window,
    KvDtype io_dtype,
    KvDtype cache_dtype,
    PagedAttentionRouteHint route_hint) {
  const GroupedPagedAttentionKind grouped_kind = select_grouped_paged_attention(
      io_dtype,
      cache_dtype,
      num_seqs,
      num_q_heads,
      num_kv_heads,
      head_size,
      block_size,
      /*query_rows=*/1,
      max_context_len,
      route_hint);
  const bool use_grouped =
      grouped_kind != GroupedPagedAttentionKind::None &&
      grouped_pipelines_supported(
          device, grouped_kind, num_q_heads, num_kv_heads);
  // Generic V2 uses contiguous 512-token partitions. The grouped path uses
  // MLX-style strided stripes and its dedicated second pass.
  const uint32_t max_num_partitions = use_grouped
      ? grouped_stripe_count(
            grouped_kind, max_context_len, num_q_heads, num_kv_heads)
      : (static_cast<uint32_t>(max_context_len) + kPartitionSize - 1) /
          kPartitionSize;

  // Auxiliary buffers. Rust path allocates `MTLResourceOptions::
  // StorageModePrivate` device buffers; we allocate via MLX so the
  // encoder tracks them. Sizes mirror the Rust path:
  //   exp_sums:   [num_seqs * num_heads * max_num_partitions] f32
  //   max_logits: same
  //   tmp_out:    [num_seqs * num_heads * max_num_partitions * head_size]
  //               in io dtype (NOT cache dtype — kernel writes io)
  //
  // We construct them as 1D MLX arrays, allocate via `set_data`, and
  // hand to MLX via `add_temporaries` so their lifetime extends past
  // this encoder's commit but ends after the kernels complete.
  // MLX's `Shape` element type is `int32_t` (a vector of ints). Use
  // 64-bit math for the size calculation to detect overflow, then
  // narrow to `int` for the shape constructor with an explicit
  // overflow guard.
  const int64_t exp_sums_size_i64 =
      static_cast<int64_t>(num_seqs) *
      static_cast<int64_t>(num_q_heads) *
      static_cast<int64_t>(max_num_partitions);
  const int64_t tmp_out_size_i64 =
      exp_sums_size_i64 * static_cast<int64_t>(head_size);
  if (exp_sums_size_i64 > std::numeric_limits<int>::max() ||
      tmp_out_size_i64 > std::numeric_limits<int>::max()) {
    throw std::runtime_error(
        "[mlx_paged_dispatch] V2 auxiliary buffer size exceeds INT_MAX; "
        "request too large for the kernel's int-sized tensor shapes");
  }
  const int exp_sums_size = static_cast<int>(exp_sums_size_i64);
  const int tmp_out_size = static_cast<int>(tmp_out_size_i64);

  mlx::core::array exp_sums(
      mlx::core::Shape{exp_sums_size},
      mlx::core::float32,
      nullptr,
      {});
  exp_sums.set_data(mlx::core::allocator::malloc(exp_sums.nbytes()));

  mlx::core::array max_logits(
      mlx::core::Shape{exp_sums_size},
      mlx::core::float32,
      nullptr,
      {});
  max_logits.set_data(mlx::core::allocator::malloc(max_logits.nbytes()));

  mlx::core::array tmp_out(
      mlx::core::Shape{tmp_out_size},
      mlx_dtype_for(io_dtype),
      nullptr,
      {});
  tmp_out.set_data(mlx::core::allocator::malloc(tmp_out.nbytes()));

  // Stage 1: partitioned attention.
  {
    const std::string& kernel_name = use_grouped
        ? paged_attention_grouped_kernel_name(grouped_kind)
        : paged_attention_v2_kernel_name(
              io_dtype,
              cache_dtype,
              head_size,
              block_size,
              /*use_alibi=*/false);
    MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_output_array(exp_sums, 0);
    encoder.set_output_array(max_logits, 1);
    encoder.set_output_array(tmp_out, 2);
    encoder.set_input_array(q, 3);
    encoder.set_input_array(k_pool, 4);
    encoder.set_input_array(v_pool, 5);
    encoder.set_input_array(k_scale, 6);
    encoder.set_input_array(v_scale, 7);

    encoder.set_bytes<int32_t>(num_kv_heads, 8);
    encoder.set_bytes<float>(scale, 9);
    encoder.set_bytes<float>(softcapping, 10);
    encoder.set_input_array(block_table, 11);
    encoder.set_input_array(seq_lens, 12);
    encoder.set_bytes<int32_t>(max_blocks_per_seq, 13);
    const float dummy_zero = 0.0f;
    encoder.set_bytes(dummy_zero, 14); // alibi_slopes

    const int32_t q_stride = num_q_heads * head_size;
    const int32_t kv_block_stride = num_kv_heads * head_size * block_size;
    const int32_t kv_head_stride = head_size * block_size;
    encoder.set_bytes<int32_t>(q_stride, 15);
    encoder.set_bytes<int32_t>(kv_block_stride, 16);
    encoder.set_bytes<int32_t>(kv_head_stride, 17);

    // Sliding-window mask. 0 = full context (default).
    encoder.set_bytes<int32_t>(sliding_window, 18);

    // Match MLX's long-context GQA vector geometry on the specialized path:
    // one SIMD group per query head, grouped under the KV head they share.
    const int gqa_factor = num_q_heads / num_kv_heads;
    MTL::Size group = use_grouped
        ? MTL::Size::Make(32, static_cast<size_t>(gqa_factor), 1)
        : MTL::Size::Make(kNumThreads, 1, 1);
    MTL::Size grid = use_grouped
        ? MTL::Size::Make(
              static_cast<size_t>(num_kv_heads),
              static_cast<size_t>(num_seqs),
              static_cast<size_t>(max_num_partitions))
        : MTL::Size::Make(
              static_cast<size_t>(num_q_heads),
              static_cast<size_t>(num_seqs),
              static_cast<size_t>(max_num_partitions));
    if (!use_grouped) {
      // Threadgroup memory: V2 partitions context into PARTITION_SIZE
      // chunks, so logits is sized by PARTITION_SIZE (not max_seq_len).
      // V-reduce phase still needs (NUM_WARPS/2) * head_size f32s.
      const size_t logits_bytes =
          static_cast<size_t>(kPartitionSize) * sizeof(float);
      const size_t v_reduce_bytes =
          static_cast<size_t>(kNumWarps / 2) *
          static_cast<size_t>(head_size) * sizeof(float);
      const size_t red_smem_bytes =
          2 * static_cast<size_t>(kNumWarps) * sizeof(float);
      const size_t threadgroup_mem =
          std::max(logits_bytes, v_reduce_bytes) + red_smem_bytes;
      encoder.set_threadgroup_memory_length(threadgroup_mem, 0);
    }
    if (use_grouped) {
      record_grouped_route_for_test(grouped_kind);
    }
    encoder.dispatch_threadgroups(grid, group);
  }

  // Stage 2: reduce partitions into final output.
  // This stage reads the auxiliary buffers stage 1 wrote. No manual
  // barrier needed: `exp_sums`/`max_logits`/`tmp_out` were registered
  // via `set_output_array` in stage 1 and re-set as `set_input_array`
  // here, so MLX's dependency tracking fences via `prev_outputs_`.
  {
    const std::string& kernel_name = use_grouped
        ? paged_attention_grouped_reduce_kernel_name(grouped_kind)
        : paged_attention_v2_reduce_kernel_name(io_dtype, head_size);
    MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_output_array(out, 0);
    encoder.set_input_array(exp_sums, 1);
    encoder.set_input_array(max_logits, 2);
    encoder.set_input_array(tmp_out, 3);
    encoder.set_input_array(seq_lens, 4); // context_lens
    encoder.set_bytes<int32_t>(static_cast<int32_t>(max_num_partitions), 5);

    if (!use_grouped) {
      // Threadgroup memory: 2 * max_num_partitions * sizeof(f32).
      const size_t threadgroup_mem =
          2 * static_cast<size_t>(max_num_partitions) * sizeof(float);
      encoder.set_threadgroup_memory_length(threadgroup_mem, 0);
    }

    MTL::Size group =
        MTL::Size::Make(use_grouped ? 1024 : kNumThreads, 1, 1);
    MTL::Size grid = MTL::Size::Make(
        static_cast<size_t>(num_q_heads),
        static_cast<size_t>(num_seqs),
        1);
    encoder.dispatch_threadgroups(grid, group);
  }

  // Hand temporaries to MLX so their refcount survives until the
  // encoder commits + the kernels complete. After this, the local
  // references can go out of scope without freeing the GPU buffers
  // prematurely.
  encoder.add_temporary(std::move(exp_sums));
  encoder.add_temporary(std::move(max_logits));
  encoder.add_temporary(std::move(tmp_out));

  (void)stream;
}

} // namespace

// =============================================================================
// dispatch_paged_attention_auto — public entry point. V1 if
// max_context_len <= PARTITION_SIZE, else V2.
// =============================================================================
void dispatch_paged_attention_auto(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    float scale,
    float softcap,
    int sliding_window,
    KvDtype kv_dtype,
    PagedAttentionRouteHint route_hint) {
  // The Metal kernel masks K positions older than
  // `context_len - sliding_window` when sliding_window > 0. Negative
  // values are illegal (only 0 is a valid "no mask" sentinel).
  if (sliding_window < 0) {
    std::ostringstream msg;
    msg << "[mlx_paged_dispatch] sliding_window=" << sliding_window
        << " must be >= 0 (use 0 to disable the sliding mask).";
    throw std::runtime_error(msg.str());
  }
  if (num_seqs == 0 || num_q_heads == 0 || head_size == 0 ||
      max_context_len <= 0 || max_blocks_per_seq <= 0) {
    std::ostringstream msg;
    msg << "[mlx_paged_dispatch] invalid dispatch dimensions"
        << " num_seqs=" << num_seqs << " num_q_heads=" << num_q_heads
        << " head_size=" << head_size
        << " max_context_len=" << max_context_len
        << " max_blocks_per_seq=" << max_blocks_per_seq;
    throw std::runtime_error(msg.str());
  }

  // io = bf16 for FP8, else io = cache dtype.
  const KvDtype io_dtype = io_dtype_for(kv_dtype);
  const KvDtype cache_dtype = kv_dtype;

  // softcap == 0.0 is the C++ caller's "disabled" sentinel; the kernel
  // expects 1.0 to mean disabled. Translate.
  const float softcapping = (softcap == 0.0f) ? 1.0f : softcap;

  if (static_cast<uint32_t>(max_context_len) <= kPartitionSize) {
    dispatch_paged_attention_v1_inner(
        encoder,
        device,
        stream,
        out,
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        num_seqs,
        num_q_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_context_len,
        max_blocks_per_seq,
        scale,
        softcapping,
        sliding_window,
        io_dtype,
        cache_dtype);
  } else {
    dispatch_paged_attention_v2_inner(
        encoder,
        device,
        stream,
        out,
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        num_seqs,
        num_q_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_context_len,
        max_blocks_per_seq,
        scale,
        softcapping,
        sliding_window,
        io_dtype,
        cache_dtype,
        route_hint);
  }

  // Reference unused parameter so the compiler doesn't warn.
  (void)dtype_byte_size(kv_dtype);
}

namespace {

void dispatch_paged_attention_varlen_v1_inner(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& cu_seqlens_q,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_seqs,
    int total_queries,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    float scale,
    float softcapping,
    int sliding_window,
    KvDtype io_dtype,
    KvDtype cache_dtype) {
  const std::string& kernel_name = paged_attention_varlen_v1_kernel_name(
      io_dtype, cache_dtype, head_size, block_size, /*use_alibi=*/false);
  MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
  encoder.set_compute_pipeline_state(pipeline);

  const float dummy_zero = 0.0f;
  encoder.set_bytes(dummy_zero, 0); // exp_sums (unused in V1)
  encoder.set_bytes(dummy_zero, 1); // max_logits (unused in V1)
  encoder.set_output_array(out, 2);
  encoder.set_input_array(q, 3);
  encoder.set_input_array(k_pool, 4);
  encoder.set_input_array(v_pool, 5);
  encoder.set_input_array(k_scale, 6);
  encoder.set_input_array(v_scale, 7);

  encoder.set_bytes<int32_t>(num_kv_heads, 8);
  encoder.set_bytes<float>(scale, 9);
  encoder.set_bytes<float>(softcapping, 10);
  encoder.set_input_array(block_table, 11);
  encoder.set_input_array(seq_lens, 12);
  encoder.set_bytes<int32_t>(max_blocks_per_seq, 13);
  encoder.set_bytes(dummy_zero, 14); // alibi_slopes

  const int32_t q_stride = num_q_heads * head_size;
  const int32_t kv_block_stride = num_kv_heads * head_size * block_size;
  const int32_t kv_head_stride = head_size * block_size;
  encoder.set_bytes<int32_t>(q_stride, 15);
  encoder.set_bytes<int32_t>(kv_block_stride, 16);
  encoder.set_bytes<int32_t>(kv_head_stride, 17);
  encoder.set_bytes<int32_t>(sliding_window, 18);

  encoder.set_input_array(cu_seqlens_q, 19);
  encoder.set_bytes<int32_t>(num_seqs, 20);

  const size_t logits_bytes =
      static_cast<size_t>(max_context_len) * sizeof(float);
  const size_t v_reduce_bytes =
      static_cast<size_t>(kNumWarps / 2) *
      static_cast<size_t>(head_size) * sizeof(float);
  const size_t red_smem_bytes =
      2 * static_cast<size_t>(kNumWarps) * sizeof(float);
  const size_t threadgroup_mem =
      std::max(logits_bytes, v_reduce_bytes) + red_smem_bytes;
  encoder.set_threadgroup_memory_length(threadgroup_mem, 0);

  MTL::Size group = MTL::Size::Make(kNumThreads, 1, 1);
  MTL::Size grid = MTL::Size::Make(
      static_cast<size_t>(num_q_heads),
      static_cast<size_t>(total_queries),
      1);
  encoder.dispatch_threadgroups(grid, group);

  (void)stream;
}

void dispatch_paged_attention_varlen_v2_inner(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& cu_seqlens_q,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_seqs,
    int total_queries,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    float scale,
    float softcapping,
    int sliding_window,
    KvDtype io_dtype,
    KvDtype cache_dtype) {
  const GroupedPagedAttentionKind grouped_kind = total_queries == 2
      ? select_grouped_paged_attention(
          io_dtype,
          cache_dtype,
          num_seqs,
          num_q_heads,
          num_kv_heads,
          head_size,
          block_size,
          total_queries,
          max_context_len)
      : GroupedPagedAttentionKind::None;
  const bool use_grouped =
      grouped_kind != GroupedPagedAttentionKind::None &&
      grouped_pipelines_supported(
          device, grouped_kind, num_q_heads, num_kv_heads);
  const uint32_t max_num_partitions = use_grouped
      ? grouped_stripe_count(
            grouped_kind, max_context_len, num_q_heads, num_kv_heads)
      : (static_cast<uint32_t>(max_context_len) + kPartitionSize - 1) /
          kPartitionSize;

  // Aux buffers indexed by q_token_idx (not seq_idx) — size by total_queries.
  const int64_t exp_sums_size_i64 =
      static_cast<int64_t>(total_queries) *
      static_cast<int64_t>(num_q_heads) *
      static_cast<int64_t>(max_num_partitions);
  const int64_t tmp_out_size_i64 =
      exp_sums_size_i64 * static_cast<int64_t>(head_size);
  if (exp_sums_size_i64 > std::numeric_limits<int>::max() ||
      tmp_out_size_i64 > std::numeric_limits<int>::max()) {
    throw std::runtime_error(
        "[mlx_paged_dispatch] V2 varlen auxiliary buffer size exceeds INT_MAX");
  }
  const int exp_sums_size = static_cast<int>(exp_sums_size_i64);
  const int tmp_out_size = static_cast<int>(tmp_out_size_i64);

  mlx::core::array exp_sums(
      mlx::core::Shape{exp_sums_size},
      mlx::core::float32,
      nullptr,
      {});
  exp_sums.set_data(mlx::core::allocator::malloc(exp_sums.nbytes()));

  mlx::core::array max_logits(
      mlx::core::Shape{exp_sums_size},
      mlx::core::float32,
      nullptr,
      {});
  max_logits.set_data(mlx::core::allocator::malloc(max_logits.nbytes()));

  mlx::core::array tmp_out(
      mlx::core::Shape{tmp_out_size},
      mlx_dtype_for(io_dtype),
      nullptr,
      {});
  tmp_out.set_data(mlx::core::allocator::malloc(tmp_out.nbytes()));

  // Stage 1: partitioned varlen attention.
  {
    const std::string& kernel_name = use_grouped
        ? paged_attention_grouped_kernel_name(grouped_kind)
        : paged_attention_varlen_v2_kernel_name(
              io_dtype,
              cache_dtype,
              head_size,
              block_size,
              /*use_alibi=*/false);
    MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_output_array(exp_sums, 0);
    encoder.set_output_array(max_logits, 1);
    encoder.set_output_array(tmp_out, 2);
    encoder.set_input_array(q, 3);
    encoder.set_input_array(k_pool, 4);
    encoder.set_input_array(v_pool, 5);
    encoder.set_input_array(k_scale, 6);
    encoder.set_input_array(v_scale, 7);

    encoder.set_bytes<int32_t>(num_kv_heads, 8);
    encoder.set_bytes<float>(scale, 9);
    encoder.set_bytes<float>(softcapping, 10);
    encoder.set_input_array(block_table, 11);
    encoder.set_input_array(seq_lens, 12);
    encoder.set_bytes<int32_t>(max_blocks_per_seq, 13);
    const float dummy_zero = 0.0f;
    encoder.set_bytes(dummy_zero, 14); // alibi_slopes

    const int32_t q_stride = num_q_heads * head_size;
    const int32_t kv_block_stride = num_kv_heads * head_size * block_size;
    const int32_t kv_head_stride = head_size * block_size;
    encoder.set_bytes<int32_t>(q_stride, 15);
    encoder.set_bytes<int32_t>(kv_block_stride, 16);
    encoder.set_bytes<int32_t>(kv_head_stride, 17);
    encoder.set_bytes<int32_t>(sliding_window, 18);

    encoder.set_input_array(cu_seqlens_q, 19);
    encoder.set_bytes<int32_t>(num_seqs, 20);

    // Keep one GQA-sized threadgroup per query row (192 threads for GQA6 or
    // 256 for GQA8). Combining both q_len=2 rows would cut occupancy sharply
    // on Apple GPUs; adjacent y groups still traverse identical pages and
    // retain cache locality.
    const int gqa_factor = num_q_heads / num_kv_heads;
    MTL::Size group = use_grouped
        ? MTL::Size::Make(
              32,
              static_cast<size_t>(gqa_factor),
              1)
        : MTL::Size::Make(kNumThreads, 1, 1);
    MTL::Size grid = use_grouped
        ? MTL::Size::Make(
              static_cast<size_t>(num_kv_heads),
              static_cast<size_t>(total_queries),
              static_cast<size_t>(max_num_partitions))
        : MTL::Size::Make(
              static_cast<size_t>(num_q_heads),
              static_cast<size_t>(total_queries),
              static_cast<size_t>(max_num_partitions));
    if (!use_grouped) {
      const size_t logits_bytes =
          static_cast<size_t>(kPartitionSize) * sizeof(float);
      const size_t v_reduce_bytes =
          static_cast<size_t>(kNumWarps / 2) *
          static_cast<size_t>(head_size) * sizeof(float);
      const size_t red_smem_bytes =
          2 * static_cast<size_t>(kNumWarps) * sizeof(float);
      const size_t threadgroup_mem =
          std::max(logits_bytes, v_reduce_bytes) + red_smem_bytes;
      encoder.set_threadgroup_memory_length(threadgroup_mem, 0);
    }
    if (use_grouped) {
      record_grouped_route_for_test(grouped_kind);
    }
    encoder.dispatch_threadgroups(grid, group);
  }

  // Stage 2: reduce partitions.
  {
    const std::string& kernel_name = use_grouped
        ? paged_attention_grouped_reduce_kernel_name(grouped_kind)
        : paged_attention_varlen_v2_reduce_kernel_name(io_dtype, head_size);
    MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_output_array(out, 0);
    encoder.set_input_array(exp_sums, 1);
    encoder.set_input_array(max_logits, 2);
    encoder.set_input_array(tmp_out, 3);
    encoder.set_input_array(seq_lens, 4);
    encoder.set_bytes<int32_t>(static_cast<int32_t>(max_num_partitions), 5);

    encoder.set_input_array(cu_seqlens_q, 6);
    encoder.set_bytes<int32_t>(num_seqs, 7);

    if (!use_grouped) {
      const size_t threadgroup_mem =
          2 * static_cast<size_t>(max_num_partitions) * sizeof(float);
      encoder.set_threadgroup_memory_length(threadgroup_mem, 0);
    }

    MTL::Size group =
        MTL::Size::Make(use_grouped ? 1024 : kNumThreads, 1, 1);
    MTL::Size grid = MTL::Size::Make(
        static_cast<size_t>(num_q_heads),
        static_cast<size_t>(total_queries),
        1);
    encoder.dispatch_threadgroups(grid, group);
  }

  encoder.add_temporary(std::move(exp_sums));
  encoder.add_temporary(std::move(max_logits));
  encoder.add_temporary(std::move(tmp_out));

  (void)stream;
}

} // namespace

void dispatch_paged_attention_varlen_auto(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& cu_seqlens_q,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int num_seqs,
    int total_queries,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    float scale,
    float softcap,
    int sliding_window,
    KvDtype kv_dtype) {
  if (sliding_window < 0) {
    std::ostringstream msg;
    msg << "[mlx_paged_dispatch] sliding_window=" << sliding_window
        << " must be >= 0 (use 0 to disable the sliding mask).";
    throw std::runtime_error(msg.str());
  }
  if (num_seqs == 0 || total_queries == 0 || num_q_heads == 0 ||
      head_size == 0 || max_context_len <= 0 || max_blocks_per_seq <= 0) {
    std::ostringstream msg;
    msg << "[mlx_paged_dispatch] invalid varlen dispatch dimensions"
        << " num_seqs=" << num_seqs << " total_queries=" << total_queries
        << " num_q_heads=" << num_q_heads << " head_size=" << head_size
        << " max_context_len=" << max_context_len
        << " max_blocks_per_seq=" << max_blocks_per_seq;
    throw std::runtime_error(msg.str());
  }

  const KvDtype io_dtype = io_dtype_for(kv_dtype);
  const KvDtype cache_dtype = kv_dtype;
  const float softcapping = (softcap == 0.0f) ? 1.0f : softcap;

  if (static_cast<uint32_t>(max_context_len) <= kPartitionSize) {
    dispatch_paged_attention_varlen_v1_inner(
        encoder,
        device,
        stream,
        out,
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        cu_seqlens_q,
        k_scale,
        v_scale,
        num_seqs,
        total_queries,
        num_q_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_context_len,
        max_blocks_per_seq,
        scale,
        softcapping,
        sliding_window,
        io_dtype,
        cache_dtype);
  } else {
    dispatch_paged_attention_varlen_v2_inner(
        encoder,
        device,
        stream,
        out,
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        cu_seqlens_q,
        k_scale,
        v_scale,
        num_seqs,
        total_queries,
        num_q_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_context_len,
        max_blocks_per_seq,
        scale,
        softcapping,
        sliding_window,
        io_dtype,
        cache_dtype);
  }
}

} // namespace mlx::core::fast::paged
