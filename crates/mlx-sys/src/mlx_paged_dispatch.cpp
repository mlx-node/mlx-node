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
#include "mlx_paged_metallib.h"
#include "mlx_metallib_fallback.h"

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
#include "mlx/backend/metal/metal.h"
#include "mlx/memory.h"
#include "mlx/dtype.h"
#include "mlx/transforms.h"

namespace mlx::core::fast::paged {

namespace {

// Partition size for V2 kernel (mirrors `PARTITION_SIZE` constant in
// `crates/mlx-paged-attn/src/metal/paged_attention.rs`).
constexpr uint32_t kPartitionSize = 512;
// Row-partitions an int8 varlen prefill's partials hold (Splash
// PagedAttention::prefillWorkspace of a 2,048-row chunk).
constexpr int64_t kVarlenWorkspaceRows = 2048;

// `NUM_THREADS` and `NUM_WARPS` baked into the forked kernels. These
// must agree with the `_nt256_nsl32` suffix in the kernel-name format
// (see `crates/mlx-paged-attn/src/metal/state.rs`).
constexpr int kNumThreads = 256;
constexpr int kNumSimdLanes = 32;
constexpr int kNumWarps = kNumThreads / kNumSimdLanes;

} // namespace

// =============================================================================
// .metallib loading
// =============================================================================
//
// `paged_attn.metallib` holds the paged-attention kernels and the prebuilt
// bridge kernels (mlx_kquant_metal.cpp, mlx_segmented_sdpa.cpp,
// mlx_affine_mixed_qmm.cpp). mlx-sys/build.rs compiles it into
// `<OUT_DIR>/paged_attn.metallib` and copies it next to the crate-internal
// `mlx.metallib` so it ships in the same place. `Device::get_library(name,
// path)` loads it once per process; later calls hit MLX's library cache.

namespace {
const std::string kPagedAttnLibraryName = "mlx_paged_attn";
} // namespace

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
#ifdef MLX_NODE_PAGED_ATTN_METALLIB
      // Last resort: the build-tree origin build.rs compiled, as MLX's own
      // METAL_PATH. The same string lets packages/core/metallib-select.ts
      // bind the addon to the mlx-sys build it linked against.
      std::filesystem::path(MLX_NODE_PAGED_ATTN_METALLIB),
#endif
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

MTL::ComputePipelineState* get_prebuilt_kernel(
    mlx::core::metal::Device& device,
    const char* tag,
    const char* family,
    const std::string& kname,
    const std::string& hash_name,
    const mlx::core::metal::MTLFCList& func_consts) {
  try {
    auto* lib = get_paged_attn_library(device);
    return device.get_kernel(kname, lib, hash_name, func_consts);
  } catch (const std::exception& e) {
    std::string path;
    try {
      path = paged_attn_metallib_path().string();
    } catch (const std::exception&) {
      path = "paged_attn.metallib";
    }
    std::ostringstream msg;
    msg << "[" << tag << "] Cannot load " << family << " kernel " << kname
        << " from " << path
        << ". The metallib is stale or incomplete; rebuild it with `yarn "
           "build:native`. "
        << e.what();
    throw PrebuiltKernelMissing(msg.str());
  }
}

namespace {

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
    case KvDtype::Int8:
      return "char";
  }
  // Unreachable — quiet warning.
  return "half";
}

// io == cache for Fp16/Bf16, bfloat16 for Int8.
KvDtype io_dtype_for(KvDtype cache) {
  return cache == KvDtype::Int8 ? KvDtype::Bf16 : cache;
}

mlx::core::Dtype mlx_dtype_for(KvDtype d) {
  switch (d) {
    case KvDtype::Fp16:
      return mlx::core::float16;
    case KvDtype::Bf16:
      return mlx::core::bfloat16;
    case KvDtype::Int8:
      return mlx::core::int8;
  }
  return mlx::core::bfloat16;
}

size_t dtype_byte_size(KvDtype d) {
  switch (d) {
    case KvDtype::Fp16:
    case KvDtype::Bf16:
      return 2;
    case KvDtype::Int8:
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
    bool values_token_major) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, KvDtype, bool>, std::string> cache;
  return memoized_kernel_name(
      cache,
      mutex,
      std::make_tuple(input_dtype, cache_dtype, values_token_major),
      [&] {
    std::ostringstream os;
    os << "reshape_and_cache_kv_" << dtype_string(input_dtype) << "_cache_"
       << dtype_string(cache_dtype) << (values_token_major ? "_vt" : "");
    return os.str();
  });
}

const std::string& reshape_and_cache_int8_kernel_name(
    int head_size,
    bool values_token_major) {
  static std::mutex mutex;
  static std::map<std::tuple<int, bool>, std::string> cache;
  return memoized_kernel_name(
      cache, mutex, std::make_tuple(head_size, values_token_major), [&] {
    std::ostringstream os;
    os << "reshape_and_cache_int8_kv_" << dtype_string(KvDtype::Bf16) << "_hd"
       << head_size << (values_token_major ? "_vt" : "");
    return os.str();
  });
}

// The keys-token-major page (K `[num_blocks, num_kv_heads, block_size,
// head_size]`, V dimension-major): `kv_int8_store_ktm.h`.
const std::string& reshape_and_cache_int8_ktm_kernel_name(int head_size) {
  static std::mutex mutex;
  static std::map<int, std::string> cache;
  return memoized_kernel_name(cache, mutex, head_size, [&] {
    std::ostringstream os;
    os << "reshape_and_cache_int8_kv_" << dtype_string(KvDtype::Bf16) << "_hd"
       << head_size << "_ktm";
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

// An int8 v2 pass writes f32 partials; its reducer is the Splash
// splash_attention_combine_splits order.
const std::string& paged_attention_v2_reduce_kernel_name(
    KvDtype io_dtype,
    KvDtype cache_dtype,
    int head_size) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, KvDtype, int>, std::string> cache;
  return memoized_kernel_name(
      cache, mutex, std::make_tuple(io_dtype, cache_dtype, head_size), [&] {
    std::ostringstream os;
    os << "paged_attention_v2_reduce_"
       << (cache_dtype == KvDtype::Int8 ? "int8" : dtype_string(io_dtype))
       << "_hs" << head_size << "_nt" << kNumThreads << "_nsl"
       << kNumSimdLanes << "_ps" << kPartitionSize;
    return os.str();
  });
}

// v2 partials: f32 for an int8 pool, the io dtype otherwise.
mlx::core::Dtype partial_dtype_for(KvDtype io_dtype, KvDtype cache_dtype) {
  return cache_dtype == KvDtype::Int8 ? mlx::core::float32
                                      : mlx_dtype_for(io_dtype);
}

enum class GroupedPagedAttentionKind {
  None,
  Qwen35D256,
  D128Direct,
  D512Direct,
};

enum class GroupedD512Mode {
  Disabled,
  Auto,
  Force,
};

const std::string& paged_attention_grouped_kernel_name(
    GroupedPagedAttentionKind kind,
    KvDtype cache_dtype) {
  if (cache_dtype == KvDtype::Int8) {
    static const std::string qwen35_int8 =
        "paged_attention_grouped_int8_hs256_bs16_striped";
    if (kind != GroupedPagedAttentionKind::Qwen35D256) {
      throw std::logic_error("grouped int8 paged attention is D256 only");
    }
    return qwen35_int8;
  }
  static const std::string qwen35 =
      "paged_attention_grouped_bfloat16_hs256_bs16_striped";
  static const std::string gemma4 =
      "paged_attention_grouped_bfloat16_hs512_bs16_striped";
  switch (kind) {
    case GroupedPagedAttentionKind::D128Direct: {
      static const std::string name = "paged_attention_grouped_bfloat16_hs128_bs16_striped";
      return name;
    }
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
    GroupedPagedAttentionKind kind,
    KvDtype cache_dtype) {
  if (cache_dtype == KvDtype::Int8) {
    static const std::string qwen35_int8 =
        "paged_attention_grouped_int8_hs256_striped_reduce";
    if (kind != GroupedPagedAttentionKind::Qwen35D256) {
      throw std::logic_error("grouped int8 paged attention is D256 only");
    }
    return qwen35_int8;
  }
  static const std::string qwen35 =
      "paged_attention_grouped_bfloat16_hs256_striped_reduce";
  static const std::string gemma4 =
      "paged_attention_grouped_bfloat16_hs512_striped_reduce";
  switch (kind) {
    case GroupedPagedAttentionKind::D128Direct: {
      static const std::string name = "paged_attention_grouped_bfloat16_hs128_striped_reduce";
      return name;
    }
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

// Partitions of an unplanned grouped dispatch. D512 and D128 have no table:
// the bf16 result changes with the count, so the model's device rule (the
// plan) or the D512 diagnostic override is the only source; 0 keeps generic
// V2, and D128 without a plan never leaves generic V2. The table below serves
// the Qwen3.5 D256 route only.
uint32_t grouped_stripe_count(
    GroupedPagedAttentionKind kind,
    int max_context_len) {
  if (kind == GroupedPagedAttentionKind::D512Direct) {
    return grouped_d512_stripe_override();
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
      io_dtype == KvDtype::Bf16 &&
      (cache_dtype == KvDtype::Bf16 || cache_dtype == KvDtype::Int8) &&
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
  if (route_hint == PagedAttentionRouteHint::ForceD128) {
    // Validated geometries: 32q/2kv (Muse-Glimmer, GQA 16) and 32q/8kv
    // (K2-Horizon, GQA 4). The kernel derives the GQA fan-out from the
    // threadgroup shape; `grouped_pipelines_supported` still gates the
    // required stage width downstream.
    const bool supported_heads =
        (num_q_heads == 32 && num_kv_heads == 2) ||
        (num_q_heads == 32 && num_kv_heads == 8);
    return io_dtype == KvDtype::Bf16 && cache_dtype == KvDtype::Bf16 &&
        num_seqs == 1 && supported_heads &&
        head_size == 128 && block_size == 16 && query_rows == 1 &&
        max_context_len > static_cast<int>(kPartitionSize)
        ? GroupedPagedAttentionKind::D128Direct
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
    KvDtype cache_dtype,
    int head_size) {
  static std::mutex mutex;
  static std::map<std::tuple<KvDtype, KvDtype, int>, std::string> cache;
  return memoized_kernel_name(
      cache, mutex, std::make_tuple(io_dtype, cache_dtype, head_size), [&] {
    std::ostringstream os;
    os << "paged_attention_varlen_v2_reduce_"
       << (cache_dtype == KvDtype::Int8 ? "int8" : dtype_string(io_dtype))
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
    GroupedPagedAttentionKind kind,
    KvDtype cache_dtype) noexcept {
  try {
    auto* stage = load_pipeline(
        device, paged_attention_grouped_kernel_name(kind, cache_dtype));
    auto* reduce = load_pipeline(
        device, paged_attention_grouped_reduce_kernel_name(kind, cache_dtype));
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
    KvDtype cache_dtype,
    int num_q_heads,
    int num_kv_heads) {
  // There is one active Metal device. Cache capability independently for the
  // concrete instantiations so an unavailable pipeline cannot disable the
  // supported paths for other head sizes.
  const GroupedPipelineLimits* limits_ptr = nullptr;
  if (kind == GroupedPagedAttentionKind::Qwen35D256 &&
      cache_dtype == KvDtype::Int8) {
    static const GroupedPipelineLimits qwen35_int8_limits =
        load_grouped_pipeline_limits(device, kind, KvDtype::Int8);
    limits_ptr = &qwen35_int8_limits;
  } else if (kind == GroupedPagedAttentionKind::Qwen35D256) {
    static const GroupedPipelineLimits qwen35_limits =
        load_grouped_pipeline_limits(device, kind, KvDtype::Bf16);
    limits_ptr = &qwen35_limits;
  } else if (kind == GroupedPagedAttentionKind::D128Direct) {
    static const GroupedPipelineLimits d128_limits =
        load_grouped_pipeline_limits(device, kind, KvDtype::Bf16);
    limits_ptr = &d128_limits;
  } else if (kind == GroupedPagedAttentionKind::D512Direct) {
    static const GroupedPipelineLimits d512_limits =
        load_grouped_pipeline_limits(device, kind, KvDtype::Bf16);
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

// Decode reads by page: 0 a call over a vLLM page; launches over the
// keys-token-major page: 1 striped, 2 generic, 3 v1.
std::atomic<uint64_t>& paged_decode_probe_counter(int kind) {
  static std::atomic<uint64_t> counters[4] = {{0}, {0}, {0}, {0}};
  return counters[kind];
}

void count_paged_decode(int kind) {
  paged_decode_probe_counter(kind).fetch_add(1, std::memory_order_relaxed);
}

// paged_attention_ktm.h. The kernels walk 16-token halves of a 32-token page.
constexpr int kKeysTokenMajorLogicalBlock = 16;

enum class KeysTokenMajorKernel { Generic, Striped };

struct KeysTokenMajorKernelFrom {
  int history;
  KeysTokenMajorKernel kernel;
};

// The kernel of one row past one partition: each entry up to the next one.
constexpr KeysTokenMajorKernelFrom kKeysTokenMajorOneRow[] = {
    {513, KeysTokenMajorKernel::Striped},
};

KeysTokenMajorKernel keys_token_major_one_row(int context) {
  KeysTokenMajorKernel kernel = kKeysTokenMajorOneRow[0].kernel;
  for (const auto& entry : kKeysTokenMajorOneRow) {
    if (context >= entry.history) {
      kernel = entry.kernel;
    }
  }
  return kernel;
}

const std::string& keys_token_major_striped_kernel_name() {
  static const std::string name = "paged_attention_grouped_int8_hs256_striped_ktm";
  return name;
}

const std::string& keys_token_major_generic_kernel_name(bool partitioned) {
  static const std::string v1 =
      "paged_attention_bfloat16_t_cache_char_hs256_nt256_nsl32_ps0_ktm";
  static const std::string v2 =
      "paged_attention_bfloat16_t_cache_char_hs256_nt256_nsl32_ps512_ktm";
  return partitioned ? v2 : v1;
}

bool keys_token_major_striped_supported(
    mlx::core::metal::Device& device,
    int num_q_heads,
    int num_kv_heads) {
  static const GroupedPipelineLimits limits = [&]() noexcept {
    try {
      auto* stage = load_pipeline(device, keys_token_major_striped_kernel_name());
      auto* reduce = load_pipeline(
          device,
          paged_attention_grouped_reduce_kernel_name(
              GroupedPagedAttentionKind::Qwen35D256, KvDtype::Int8));
      if (stage != nullptr && reduce != nullptr) {
        return GroupedPipelineLimits{
            static_cast<size_t>(stage->maxTotalThreadsPerThreadgroup()),
            static_cast<size_t>(reduce->maxTotalThreadsPerThreadgroup()),
            true};
      }
    } catch (...) {
    }
    std::fprintf(
        stderr,
        "[mlx][warn] keys-token-major striped decode pipeline unavailable; "
        "using its generic kernel\n");
    return GroupedPipelineLimits{0, 0, false};
  }();
  return limits.available && num_kv_heads > 0 &&
      num_q_heads % num_kv_heads == 0 &&
      limits.stage_threads >=
      static_cast<size_t>(32 * (num_q_heads / num_kv_heads)) &&
      limits.reduce_threads >= 1024;
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

extern "C" void mlx_paged_decode_test_probe_reset() {
  for (int kind = 0; kind < 4; ++kind) {
    paged_decode_probe_counter(kind).store(0, std::memory_order_relaxed);
  }
}

// The kinds of paged_decode_probe_counter.
extern "C" uint64_t mlx_paged_decode_test_probe_count(int kind) {
  if (kind < 0 || kind > 3) {
    return 0;
  }
  return paged_decode_probe_counter(kind).load(std::memory_order_relaxed);
}

extern "C" uint32_t mlx_paged_grouped_d128_max_stripes(
    uint32_t context, uint32_t attention_layers) {
  if (context <= kPartitionSize || attention_layers == 0) return 0;
  try {
    auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto& device = mlx::core::metal::device(stream.device);
    if (!grouped_pipelines_supported(
            device, GroupedPagedAttentionKind::D128Direct, KvDtype::Bf16, 32, 2))
      return 0;
    const size_t memory_limit = std::min(
        mlx::core::get_memory_limit(),
        static_cast<size_t>(device.mtl_device()->recommendedMaxWorkingSetSize()));
    const size_t active = mlx::core::get_active_memory();
    const size_t available = memory_limit > active ? memory_limit - active : 0;
    // FP32 sum/max plus BF16 partial output per Q head. Account for all
    // global layers retaining temporaries in an in-flight command buffer.
    constexpr size_t bytes_per_stripe = 32 * (2 * sizeof(float) + 128 * sizeof(uint16_t));
    const size_t memory_stripes = available / attention_layers / bytes_per_stripe;
    const size_t buffer_stripes = device.mtl_device()->maxBufferLength() / (32 * 128 * sizeof(uint16_t));
    const size_t tiles = (static_cast<size_t>(context) + 15) / 16;
    // 1024 is the tested kernel/ABI limit, shared with the existing D256
    // reducer. No device name, core-count guess, or fixed chosen plan.
    const size_t limit = std::min({size_t{1024}, memory_stripes, buffer_stripes, tiles});
    uint32_t stripes = 0;
    for (uint32_t candidate = 4; candidate <= limit; candidate *= 2) stripes = candidate;
    return stripes;
  } catch (...) {
    return 0;
  }
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
               KvDtype::Bf16,
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
    int max_context_len,
    uint8_t cache_dtype) {
  if (cache_dtype == 2 || cache_dtype > static_cast<uint8_t>(KvDtype::Int8)) {
    return 0;
  }
  return grouped_qwen35_shape_matches(
      /*enabled=*/true,
      KvDtype::Bf16,
      static_cast<KvDtype>(cache_dtype),
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
    mlx::core::array& k_scale,
    mlx::core::array& v_scale,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int block_size,
    int x_pack,
    KvDtype kv_dtype,
    bool values_token_major) {
  if (num_tokens == 0) {
    // No-op write — avoid encoding a zero-grid dispatch which Metal
    // would refuse anyway.
    return;
  }

  if (kv_dtype == KvDtype::Int8) {
    const bool keys_token_major = !values_token_major && k_pool.ndim() == 4;
    MTL::ComputePipelineState* pipeline = load_pipeline(
        device,
        keys_token_major
            ? reshape_and_cache_int8_ktm_kernel_name(head_size)
            : reshape_and_cache_int8_kernel_name(head_size, values_token_major));
    encoder.set_compute_pipeline_state(pipeline);
    const int32_t row_stride = num_kv_heads * head_size;
    encoder.set_input_array(new_k, 0);
    encoder.set_input_array(new_v, 1);
    encoder.set_output_array(k_pool, 2);
    encoder.set_output_array(v_pool, 3);
    encoder.set_input_array(slot_mapping, 4);
    encoder.set_output_array(k_scale, 5);
    encoder.set_output_array(v_scale, 6);
    encoder.set_bytes(row_stride, 7);
    encoder.set_bytes(row_stride, 8);
    encoder.set_bytes<int32_t>(num_kv_heads, 9);
    encoder.set_bytes<int32_t>(block_size, 10);
    encoder.set_bytes<int32_t>(num_tokens, 11);
    MTL::Size group = MTL::Size::Make(static_cast<size_t>(head_size), 1, 1);
    MTL::Size grid = MTL::Size::Make(
        2 * static_cast<size_t>(num_tokens) * static_cast<size_t>(num_kv_heads),
        1,
        1);
    encoder.dispatch_threadgroups(grid, group);
    return;
  }

  const KvDtype cache_dtype = kv_dtype;
  const KvDtype input_dtype = io_dtype_for(cache_dtype);

  const std::string& kernel_name =
      reshape_and_cache_kernel_name(
          input_dtype, cache_dtype, values_token_major);
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
  encoder.set_input_array(new_k, 0);
  encoder.set_input_array(new_v, 1);
  encoder.set_output_array(k_pool, 2);
  encoder.set_output_array(v_pool, 3);
  encoder.set_input_array(slot_mapping, 4);

  // Constants — kernel takes them as `device const int &` which
  // Metal's `setBytes` can satisfy directly without an extra device
  // buffer. (Rust path allocates a separate small device buffer per
  // constant; we save those allocations by inlining.)
  encoder.set_bytes(key_stride, 5);
  encoder.set_bytes(value_stride, 6);
  encoder.set_bytes<int32_t>(num_kv_heads, 7);
  encoder.set_bytes<int32_t>(head_size, 8);
  encoder.set_bytes<int32_t>(block_size, 9);
  encoder.set_bytes<int32_t>(x_pack, 10);

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

// Pool planes are charged by the blocks this read's tables reach, not the
// whole pool (Splash `EncoderResidency`: memory no command uses costs no
// command buffer anything). The estimate only schedules commits; binding,
// hazards and residency are unchanged.
void bind_pool_plane(
    mlx::core::metal::CommandEncoder& encoder,
    const mlx::core::array& plane,
    int idx,
    int64_t live_blocks) {
  const bool per_block = plane.ndim() >= 2 && plane.shape(0) > 0 &&
      plane.flags().row_contiguous && plane.offset() == 0 &&
      plane.data_size() == plane.size();
  if (!per_block) {
    encoder.set_input_array(plane, idx);
    return;
  }
  const size_t blocks = static_cast<size_t>(plane.shape(0));
  const size_t touched =
      std::min(static_cast<size_t>(std::max<int64_t>(live_blocks, 0)), blocks);
  encoder.set_input_array_with_budget(
      plane, idx, touched * (plane.data_size() / blocks));
}

void bind_pool_planes(
    mlx::core::metal::CommandEncoder& encoder,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    int64_t live_blocks) {
  bind_pool_plane(encoder, k_pool, 4, live_blocks);
  bind_pool_plane(encoder, v_pool, 5, live_blocks);
  bind_pool_plane(encoder, k_scale, 6, live_blocks);
  bind_pool_plane(encoder, v_scale, 7, live_blocks);
}

// buffer(21) / buffer(22) of the int8 decode-family kernels: the rows' score
// scales from `byte_offset`, or a flag that keeps `scale`.
void bind_score_scales(
    mlx::core::metal::CommandEncoder& encoder,
    const mlx::core::array* score_scales,
    int64_t byte_offset) {
  if (score_scales != nullptr) {
    encoder.set_input_array(*score_scales, 21, byte_offset);
    encoder.set_bytes<int32_t>(1, 22);
  } else {
    const float unused = 0.0f;
    encoder.set_bytes(unused, 21);
    encoder.set_bytes<int32_t>(0, 22);
  }
}

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
    int64_t live_blocks,
    float scale,
    float softcapping,
    int sliding_window,
    KvDtype io_dtype,
    KvDtype cache_dtype,
    const mlx::core::array* score_scales) {
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
  bind_pool_planes(encoder, k_pool, v_pool, k_scale, v_scale, live_blocks);

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
  bind_score_scales(encoder, score_scales, 0);

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
    int64_t live_blocks,
    float scale,
    float softcapping,
    int sliding_window,
    KvDtype io_dtype,
    KvDtype cache_dtype,
    PagedAttentionRouteHint route_hint,
    uint32_t planned_stripes,
    const mlx::core::array* score_scales) {
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
  const uint32_t grouped_partitions = planned_stripes != 0
      ? planned_stripes
      : grouped_stripe_count(grouped_kind, max_context_len);
  const bool use_grouped =
      grouped_kind != GroupedPagedAttentionKind::None &&
      (grouped_kind != GroupedPagedAttentionKind::D128Direct || planned_stripes != 0) &&
      grouped_partitions != 0 &&
      grouped_pipelines_supported(
          device, grouped_kind, cache_dtype, num_q_heads, num_kv_heads);
  // Generic V2 uses contiguous 512-token partitions. The bf16 grouped path
  // uses MLX-style strided stripes, the int8 one contiguous page ranges
  // (Splash splash_attention_pages_per_split); each has its own second pass.
  const uint32_t max_num_partitions = use_grouped
      ? grouped_partitions
      : (static_cast<uint32_t>(max_context_len) + kPartitionSize - 1) /
          kPartitionSize;

  // Auxiliary buffers. Rust path allocates `MTLResourceOptions::
  // StorageModePrivate` device buffers; we allocate via MLX so the
  // encoder tracks them. Sizes mirror the Rust path:
  //   exp_sums:   [num_seqs * num_heads * max_num_partitions] f32
  //   max_logits: same
  //   tmp_out:    [num_seqs * num_heads * max_num_partitions * head_size]
  //               in io dtype, f32 for an int8 pool
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
      partial_dtype_for(io_dtype, cache_dtype),
      nullptr,
      {});
  tmp_out.set_data(mlx::core::allocator::malloc(tmp_out.nbytes()));

  // Stage 1: partitioned attention.
  {
    const std::string& kernel_name = use_grouped
        ? paged_attention_grouped_kernel_name(grouped_kind, cache_dtype)
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
    bind_pool_planes(encoder, k_pool, v_pool, k_scale, v_scale, live_blocks);

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
    bind_score_scales(encoder, score_scales, 0);

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
        ? paged_attention_grouped_reduce_kernel_name(grouped_kind, cache_dtype)
        : paged_attention_v2_reduce_kernel_name(
              io_dtype, cache_dtype, head_size);
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

mlx::core::array float_scratch(int64_t elements, const char* what) {
  if (elements > std::numeric_limits<int>::max()) {
    throw std::runtime_error(
        std::string("[mlx_paged_dispatch] keys-token-major decode ") + what +
        " exceeds INT_MAX");
  }
  mlx::core::array a(
      mlx::core::Shape{static_cast<int>(std::max<int64_t>(elements, 1))},
      mlx::core::float32,
      nullptr,
      {});
  a.set_data(mlx::core::allocator::malloc(a.nbytes()));
  return a;
}

// The keys-token-major page (K rank 4). A row reads with the launch it takes
// alone, chosen by its own history only, so a lane in a wave reads the bits
// it reads alone: up to one partition the rows share one v1 launch (v1 reads
// each row on its own threadgroups); past it every row is its own launch.
// No vLLM-page kernel reads this page.
void dispatch_paged_attention_keys_token_major(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
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
    int64_t live_blocks,
    float scale,
    float softcapping,
    int sliding_window,
    PagedAttentionRouteHint route_hint,
    uint32_t planned_stripes,
    const mlx::core::array* score_scales) {
  const bool striped_allowed =
      keys_token_major_striped_supported(device, num_q_heads, num_kv_heads) &&
      route_hint == PagedAttentionRouteHint::Auto &&
      grouped_qwen35_shape_matches(
          grouped_qwen35_paged_attention_enabled(),
          KvDtype::Bf16,
          KvDtype::Int8,
          /*num_seqs=*/1,
          num_q_heads,
          num_kv_heads,
          head_size,
          kKeysTokenMajorLogicalBlock,
          /*query_rows=*/1,
          std::numeric_limits<int>::max());
  const auto row_kernel = [&](int context) {
    const KeysTokenMajorKernel kernel = keys_token_major_one_row(context);
    return kernel == KeysTokenMajorKernel::Striped && !striped_allowed
        ? KeysTokenMajorKernel::Generic
        : kernel;
  };

  const int32_t q_stride = num_q_heads * head_size;
  const int32_t kv_block_stride = num_kv_heads * head_size * block_size;
  const int32_t kv_head_stride = head_size * block_size;
  const int gqa_factor = num_q_heads / num_kv_heads;
  const float dummy_zero = 0.0f;
  const size_t v_reduce_bytes =
      static_cast<size_t>(kNumWarps / 2) * static_cast<size_t>(head_size) *
      sizeof(float);
  const size_t red_smem_bytes = 2 * static_cast<size_t>(kNumWarps) * sizeof(float);
  const auto partitions = [](int context) {
    return (static_cast<uint32_t>(context) + kPartitionSize - 1) / kPartitionSize;
  };

  // Row `first` of the call is row 0 of the kernel: every per-row binding
  // starts at that row.
  const auto bind_inputs = [&](int first, int64_t q_bytes) {
    encoder.set_input_array(q, 3, first * q_bytes);
    bind_pool_planes(encoder, k_pool, v_pool, k_scale, v_scale, live_blocks);
    encoder.set_bytes<int32_t>(num_kv_heads, 8);
    encoder.set_bytes<float>(scale, 9);
    encoder.set_bytes<float>(softcapping, 10);
    encoder.set_input_array(
        block_table, 11, int64_t{4} * first * max_blocks_per_seq);
    encoder.set_input_array(seq_lens, 12, int64_t{4} * first);
    encoder.set_bytes<int32_t>(max_blocks_per_seq, 13);
    encoder.set_bytes(dummy_zero, 14);
    encoder.set_bytes<int32_t>(q_stride, 15);
    encoder.set_bytes<int32_t>(kv_block_stride, 16);
    encoder.set_bytes<int32_t>(kv_head_stride, 17);
    encoder.set_bytes<int32_t>(sliding_window, 18);
    bind_score_scales(
        encoder,
        score_scales,
        int64_t{4} * first * num_q_heads);
  };
  const int64_t q_row_bytes = static_cast<int64_t>(q_stride) * q.itemsize();
  const int64_t out_row_bytes = static_cast<int64_t>(q_stride) * out.itemsize();

  const auto encode_v1 = [&](int first, int rows, int context) {
    encoder.set_compute_pipeline_state(load_pipeline(
        device, keys_token_major_generic_kernel_name(false)));
    encoder.set_bytes(dummy_zero, 0);
    encoder.set_bytes(dummy_zero, 1);
    encoder.set_output_array(out, 2, first * out_row_bytes);
    bind_inputs(first, q_row_bytes);
    encoder.set_threadgroup_memory_length(
        std::max(static_cast<size_t>(context) * sizeof(float), v_reduce_bytes) +
            red_smem_bytes,
        0);
    count_paged_decode(3);
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<size_t>(num_q_heads), static_cast<size_t>(rows), 1),
        MTL::Size::Make(kNumThreads, 1, 1));
  };

  // One row's partials: exp sums, max logits, partial outputs.
  struct RowPartials {
    int row;
    uint32_t parts;
    bool striped;
    mlx::core::array exp_sums;
    mlx::core::array max_logits;
    mlx::core::array tmp_out;
  };

  // The generic v2 or striped pass of row `p.row`, in `p.parts` partitions.
  const auto encode_partials = [&](RowPartials& p) {
    encoder.set_compute_pipeline_state(load_pipeline(
        device,
        p.striped ? keys_token_major_striped_kernel_name()
                  : keys_token_major_generic_kernel_name(true)));
    encoder.set_output_array(p.exp_sums, 0);
    encoder.set_output_array(p.max_logits, 1);
    encoder.set_output_array(p.tmp_out, 2);
    bind_inputs(p.row, q_row_bytes);
    if (p.striped) {
      count_paged_decode(1);
      encoder.dispatch_threadgroups(
          MTL::Size::Make(
              static_cast<size_t>(num_kv_heads), 1, static_cast<size_t>(p.parts)),
          MTL::Size::Make(32, static_cast<size_t>(gqa_factor), 1));
    } else {
      encoder.set_threadgroup_memory_length(
          std::max(static_cast<size_t>(kPartitionSize) * sizeof(float), v_reduce_bytes) +
              red_smem_bytes,
          0);
      count_paged_decode(2);
      encoder.dispatch_threadgroups(
          MTL::Size::Make(
              static_cast<size_t>(num_q_heads), 1, static_cast<size_t>(p.parts)),
          MTL::Size::Make(kNumThreads, 1, 1));
    }
  };

  const auto encode_reduce = [&](RowPartials& p) {
    const bool striped = p.striped;
    const int first = p.row;
    const uint32_t parts = p.parts;
    encoder.set_compute_pipeline_state(load_pipeline(
        device,
        striped ? paged_attention_grouped_reduce_kernel_name(
                      GroupedPagedAttentionKind::Qwen35D256, KvDtype::Int8)
                : paged_attention_v2_reduce_kernel_name(
                      KvDtype::Bf16, KvDtype::Int8, head_size)));
    encoder.set_output_array(out, 0, first * out_row_bytes);
    encoder.set_input_array(p.exp_sums, 1);
    encoder.set_input_array(p.max_logits, 2);
    encoder.set_input_array(p.tmp_out, 3);
    encoder.set_input_array(seq_lens, 4, int64_t{4} * first);
    encoder.set_bytes<int32_t>(static_cast<int32_t>(parts), 5);
    if (!striped) {
      encoder.set_threadgroup_memory_length(
          2 * static_cast<size_t>(parts) * sizeof(float), 0);
    }
    encoder.dispatch_threadgroups(
        MTL::Size::Make(static_cast<size_t>(num_q_heads), 1, 1),
        MTL::Size::Make(striped ? 1024 : kNumThreads, 1, 1));
  };

  if (static_cast<uint32_t>(max_context_len) <= kPartitionSize) {
    encode_v1(0, num_seqs, max_context_len);
    return;
  }

  const int32_t* lens = seq_lens.data<int32_t>();
  std::vector<int> v1_rows;
  std::vector<RowPartials> passes;
  passes.reserve(static_cast<size_t>(num_seqs));
  for (int row = 0; row < num_seqs; ++row) {
    const int context = lens[row];
    if (static_cast<uint32_t>(context) <= kPartitionSize) {
      v1_rows.push_back(row);
      continue;
    }
    const bool striped = row_kernel(context) == KeysTokenMajorKernel::Striped;
    const uint32_t parts = !striped ? partitions(context)
        : planned_stripes != 0      ? planned_stripes
                                    : grouped_stripe_count(
                                          GroupedPagedAttentionKind::Qwen35D256, context);
    const int64_t stats = int64_t{num_q_heads} * parts;
    // Each row owns its partials: a reduce never reads another row's pass.
    passes.push_back(RowPartials{
        row,
        parts,
        striped,
        float_scratch(stats, "exp sums"),
        float_scratch(stats, "max logits"),
        float_scratch(stats * head_size, "partials")});
  }
  for (auto& p : passes) {
    encode_partials(p);
  }
  for (const int row : v1_rows) {
    encode_v1(row, 1, lens[row]);
  }
  for (auto& p : passes) {
    encode_reduce(p);
  }
  for (auto& p : passes) {
    encoder.add_temporary(std::move(p.exp_sums));
    encoder.add_temporary(std::move(p.max_logits));
    encoder.add_temporary(std::move(p.tmp_out));
  }
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
    int64_t live_blocks,
    float scale,
    float softcap,
    int sliding_window,
    KvDtype kv_dtype,
    PagedAttentionRouteHint route_hint,
    uint32_t grouped_stripes,
    const mlx::core::array* score_scales) {
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

  const KvDtype io_dtype = io_dtype_for(kv_dtype);
  const KvDtype cache_dtype = kv_dtype;

  // softcap == 0.0 is the C++ caller's "disabled" sentinel; the kernel
  // expects 1.0 to mean disabled. Translate.
  const float softcapping = (softcap == 0.0f) ? 1.0f : softcap;

  // A rank-4 K is the keys-token-major page; the validator admits it only
  // whole (INT8, 32-token blocks, group 6, D 256, dimension-major V).
  if (k_pool.ndim() == 4) {
    dispatch_paged_attention_keys_token_major(
        encoder,
        device,
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
        live_blocks,
        scale,
        softcapping,
        sliding_window,
        route_hint,
        grouped_stripes,
        score_scales);
    return;
  }
  count_paged_decode(0);

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
        live_blocks,
        scale,
        softcapping,
        sliding_window,
        io_dtype,
        cache_dtype,
        score_scales);
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
        live_blocks,
        scale,
        softcapping,
        sliding_window,
        io_dtype,
        cache_dtype,
        route_hint,
        grouped_stripes,
        score_scales);
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
    int64_t live_blocks,
    float scale,
    float softcapping,
    int sliding_window,
    KvDtype io_dtype,
    KvDtype cache_dtype,
    const mlx::core::array* score_scales) {
  const std::string& kernel_name = paged_attention_varlen_v1_kernel_name(
      io_dtype, cache_dtype, head_size, block_size, /*use_alibi=*/false);
  MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
  encoder.set_compute_pipeline_state(pipeline);

  const float dummy_zero = 0.0f;
  encoder.set_bytes(dummy_zero, 0); // exp_sums (unused in V1)
  encoder.set_bytes(dummy_zero, 1); // max_logits (unused in V1)
  encoder.set_output_array(out, 2);
  encoder.set_input_array(q, 3);
  bind_pool_planes(encoder, k_pool, v_pool, k_scale, v_scale, live_blocks);

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
  bind_score_scales(encoder, score_scales, 0);

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
    int64_t live_blocks,
    float scale,
    float softcapping,
    int sliding_window,
    KvDtype io_dtype,
    KvDtype cache_dtype,
    const mlx::core::array* score_scales) {
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
          device, grouped_kind, cache_dtype, num_q_heads, num_kv_heads);
  const uint32_t max_num_partitions = use_grouped
      ? grouped_stripe_count(grouped_kind, max_context_len)
      : (static_cast<uint32_t>(max_context_len) + kPartitionSize - 1) /
          kPartitionSize;

  // An int8 prefill of one sequence runs its query rows in tiles of at most
  // kVarlenWorkspaceRows row-partitions (Splash
  // PagedAttention::prefillWorkspace: the partials are bounded at allocation
  // for every history). A row's partitions and reduce read only that row, so
  // the tiles give the same bits as one launch.
  const bool tiled = !use_grouped && cache_dtype == KvDtype::Int8 &&
      num_seqs == 1;
  const int tile_rows = tiled
      ? static_cast<int>(std::clamp<int64_t>(
            kVarlenWorkspaceRows / static_cast<int64_t>(max_num_partitions),
            1,
            total_queries))
      : total_queries;
  const int num_tiles = (total_queries + tile_rows - 1) / tile_rows;

  // Aux buffers indexed by the tile's q_token_idx: size by tile_rows.
  const int64_t exp_sums_size_i64 =
      static_cast<int64_t>(tile_rows) *
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
      partial_dtype_for(io_dtype, cache_dtype),
      nullptr,
      {});
  tmp_out.set_data(mlx::core::allocator::malloc(tmp_out.nbytes()));

  // Per tile {seq_len, 0, rows}: the tile is the last `rows` queries of a
  // sequence that ends at its last row, so each row keeps its causal bound.
  mlx::core::array tile_meta(
      mlx::core::Shape{3 * num_tiles}, mlx::core::int32, nullptr, {});
  if (tiled) {
    tile_meta.set_data(mlx::core::allocator::malloc(tile_meta.nbytes()));
    int32_t* meta = tile_meta.data<int32_t>();
    for (int t = 0; t < num_tiles; ++t) {
      const int end = std::min(total_queries, (t + 1) * tile_rows);
      meta[3 * t] = max_context_len - (total_queries - end);
      meta[3 * t + 1] = 0;
      meta[3 * t + 2] = end - t * tile_rows;
    }
  }

  const int32_t q_stride = num_q_heads * head_size;
  for (int t = 0; t < num_tiles; ++t) {
    const int first = t * tile_rows;
    const int rows = std::min(tile_rows, total_queries - first);
    const int64_t q_offset =
        static_cast<int64_t>(first) * q_stride * q.itemsize();
    const int64_t out_offset =
        static_cast<int64_t>(first) * q_stride * out.itemsize();
    const auto bind_seq_lens = [&](int idx) {
      if (tiled) {
        encoder.set_input_array(tile_meta, idx, int64_t{12} * t);
      } else {
        encoder.set_input_array(seq_lens, idx);
      }
    };
    const auto bind_cu_seqlens = [&](int idx) {
      if (tiled) {
        encoder.set_input_array(tile_meta, idx, int64_t{12} * t + 4);
      } else {
        encoder.set_input_array(cu_seqlens_q, idx);
      }
    };
    if (t > 0) {
      // The next stage 1 rewrites the partials the last reduce read.
      encoder.barrier();
    }

    // Stage 1: partitioned varlen attention.
    {
      const std::string& kernel_name = use_grouped
          ? paged_attention_grouped_kernel_name(grouped_kind, cache_dtype)
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
      encoder.set_input_array(q, 3, q_offset);
      bind_pool_planes(encoder, k_pool, v_pool, k_scale, v_scale, live_blocks);

      encoder.set_bytes<int32_t>(num_kv_heads, 8);
      encoder.set_bytes<float>(scale, 9);
      encoder.set_bytes<float>(softcapping, 10);
      encoder.set_input_array(block_table, 11);
      bind_seq_lens(12);
      encoder.set_bytes<int32_t>(max_blocks_per_seq, 13);
      const float dummy_zero = 0.0f;
      encoder.set_bytes(dummy_zero, 14); // alibi_slopes

      const int32_t kv_block_stride = num_kv_heads * head_size * block_size;
      const int32_t kv_head_stride = head_size * block_size;
      encoder.set_bytes<int32_t>(q_stride, 15);
      encoder.set_bytes<int32_t>(kv_block_stride, 16);
      encoder.set_bytes<int32_t>(kv_head_stride, 17);
      encoder.set_bytes<int32_t>(sliding_window, 18);

      bind_cu_seqlens(19);
      encoder.set_bytes<int32_t>(num_seqs, 20);
      bind_score_scales(
          encoder,
          score_scales,
          static_cast<int64_t>(first) * num_q_heads * static_cast<int64_t>(sizeof(float)));

      // Keep one GQA-sized threadgroup per query row (192 threads for GQA6
      // or 256 for GQA8). Combining both q_len=2 rows would cut occupancy
      // sharply on Apple GPUs; adjacent y groups still traverse identical
      // pages and retain cache locality.
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
                static_cast<size_t>(rows),
                static_cast<size_t>(max_num_partitions))
          : MTL::Size::Make(
                static_cast<size_t>(num_q_heads),
                static_cast<size_t>(rows),
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
          ? paged_attention_grouped_reduce_kernel_name(grouped_kind, cache_dtype)
          : paged_attention_varlen_v2_reduce_kernel_name(
                io_dtype, cache_dtype, head_size);
      MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
      encoder.set_compute_pipeline_state(pipeline);

      encoder.set_output_array(out, 0, out_offset);
      encoder.set_input_array(exp_sums, 1);
      encoder.set_input_array(max_logits, 2);
      encoder.set_input_array(tmp_out, 3);
      bind_seq_lens(4);
      encoder.set_bytes<int32_t>(static_cast<int32_t>(max_num_partitions), 5);

      bind_cu_seqlens(6);
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
          static_cast<size_t>(rows),
          1);
      encoder.dispatch_threadgroups(grid, group);
    }
  }

  if (tiled) {
    encoder.add_temporary(std::move(tile_meta));
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
    int64_t live_blocks,
    float scale,
    float softcap,
    int sliding_window,
    KvDtype kv_dtype,
    const mlx::core::array* score_scales) {
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
        live_blocks,
        scale,
        softcapping,
        sliding_window,
        io_dtype,
        cache_dtype,
        score_scales);
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
        live_blocks,
        scale,
        softcapping,
        sliding_window,
        io_dtype,
        cache_dtype,
        score_scales);
  }
}


// =============================================================================
// In-place prefill / verify attention (paged_prefill_attention.metal)
// =============================================================================

namespace {

constexpr int kPagedPrefillGroups[] = {4, 6, 8, 16};
constexpr int kPagedPrefillHeadDims[] = {128, 256};

// Splash SPLASH_ATTENTION_TILE_PARTS.
int paged_prefill_tile_parts(int group) {
  return group <= 8 ? 1 : group <= 16 ? 2 : group <= 32 ? 4 : 8;
}

std::string paged_prefill_suffix(int group, int head_size) {
  return "_g" + std::to_string(group) + "_d" + std::to_string(head_size);
}

// Splash splitKernel: INT8 pages scored by INT8 queries take the `q8q`
// entry.
std::string paged_prefill_split_name(
    KvDtype cache_dtype,
    bool int8_queries,
    int group,
    int head_size) {
  const char* format = cache_dtype == KvDtype::Int8 ? (int8_queries ? "q8q" : "q8") : "bf16";
  return std::string("paged_prefill_attention_") + format + "_split" +
      paged_prefill_suffix(group, head_size);
}

// The keys-token-major page (paged_prefill_attention_ktm.h): INT8, one
// 32-token block per page, heads 24/4 at D 256 only.
std::string paged_prefill_keys_token_major_split_name(bool int8_queries) {
  return paged_prefill_split_name(
             KvDtype::Int8,
             int8_queries,
             kPagedPrefillKeysTokenMajorGroup,
             kPagedPrefillKeysTokenMajorHeadDim) +
      "_ktm";
}

std::string paged_prefill_reduce_name(int group, int head_size) {
  return "paged_prefill_attention_reduce" + paged_prefill_suffix(group, head_size);
}

std::string paged_query_codes_name(int head_size) {
  return "paged_prefill_attention_query_codes_d" + std::to_string(head_size);
}

// Splash DeviceCapabilities::validationMessage: people know their chip, not
// its GPU family.
std::string apple_gpu_family_name(mlx::core::metal::Device& device) {
  static constexpr std::pair<MTL::GPUFamily, const char*> kFamilies[] = {
      {MTL::GPUFamilyApple10, "Apple GPU family 10 (M5)"},
      {MTL::GPUFamilyApple9, "Apple GPU family 9 (M3, M4)"},
      {MTL::GPUFamilyApple8, "Apple GPU family 8 (M2)"},
      {MTL::GPUFamilyApple7, "Apple GPU family 7 (M1)"},
  };
  for (const auto& [family, name] : kFamilies) {
    if (device.mtl_device()->supportsFamily(family)) {
      return name;
    }
  }
  return "no known Apple GPU family";
}

std::atomic<uint64_t>& paged_prefill_probe_counter(int kind) {
  static std::atomic<uint64_t> counters[5] = {{0}, {0}, {0}, {0}, {0}};
  return counters[kind];
}

std::atomic<uint32_t>& paged_prefill_probe_last_splits() {
  static std::atomic<uint32_t> value{0};
  return value;
}

std::atomic<uint32_t>& paged_prefill_probe_last_max_splits() {
  static std::atomic<uint32_t> value{0};
  return value;
}

} // namespace

bool paged_prefill_attention_geometry(int group, int head_size) {
  const bool group_ok = std::find(
                            std::begin(kPagedPrefillGroups),
                            std::end(kPagedPrefillGroups),
                            group) != std::end(kPagedPrefillGroups);
  const bool dim_ok = std::find(
                          std::begin(kPagedPrefillHeadDims),
                          std::end(kPagedPrefillHeadDims),
                          head_size) != std::end(kPagedPrefillHeadDims);
  return group_ok && dim_ok;
}

bool paged_prefill_keys_token_major_geometry(int group, int head_size) {
  return group == kPagedPrefillKeysTokenMajorGroup &&
      head_size == kPagedPrefillKeysTokenMajorHeadDim;
}

std::vector<std::string> paged_prefill_attention_kernel_names() {
  std::vector<std::string> names;
  for (int group : kPagedPrefillGroups) {
    for (int head_size : kPagedPrefillHeadDims) {
      names.push_back(paged_prefill_split_name(KvDtype::Int8, false, group, head_size));
      names.push_back(paged_prefill_split_name(KvDtype::Int8, true, group, head_size));
      names.push_back(paged_prefill_split_name(KvDtype::Bf16, false, group, head_size));
      names.push_back(paged_prefill_reduce_name(group, head_size));
    }
  }
  for (int head_size : kPagedPrefillHeadDims) {
    names.push_back(paged_query_codes_name(head_size));
  }
  names.push_back(paged_prefill_keys_token_major_split_name(false));
  names.push_back(paged_prefill_keys_token_major_split_name(true));
  return names;
}

MTL::ComputePipelineState* paged_prefill_pipeline(
    mlx::core::metal::Device& device,
    const std::string& name) {
  try {
    return get_prebuilt_kernel(device, "paged_prefill", "paged prefill attention", name);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "paged prefill attention kernel " + name + " has no pipeline on " +
        apple_gpu_family_name(device) + ": " + e.what());
  }
}

int64_t build_paged_prefill_attention_pipelines(mlx::core::metal::Device& device) {
  // Splash KernelCalibration: only Apple10 and later score INT8 queries
  // (Apple9's matrix operations share the FP32 pipe), so no other family
  // builds the q8q entries or the decode's query codes.
  const bool int8_queries = paged_prefill_int8_queries_supported(device);
  int64_t built = 0;
  for (const auto& name : paged_prefill_attention_kernel_names()) {
    if (!int8_queries &&
        (name.find("_q8q_") != std::string::npos ||
         name.find("_query_codes_") != std::string::npos)) {
      continue;
    }
    paged_prefill_pipeline(device, name);
    ++built;
  }
  return built;
}

bool paged_prefill_int8_queries_supported(mlx::core::metal::Device& device) {
  return device.mtl_device()->supportsFamily(MTL::GPUFamilyApple10);
}

int64_t paged_prefill_workspace_floats(
    int rows, int splits, int num_q_heads, int head_size) {
  const int64_t tiles =
      (static_cast<int64_t>(rows) + kPagedPrefillTileRows - 1) / kPagedPrefillTileRows;
  return tiles * splits * kPagedPrefillTileRows * num_q_heads *
      (static_cast<int64_t>(head_size) + 2);
}

void dispatch_paged_prefill_attention(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::array& out,
    mlx::core::array& workspace,
    const mlx::core::array& q_staged,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    const mlx::core::array& block_table,
    const PagedPrefillParams& params,
    int num_q_heads,
    int head_size,
    KvDtype cache_dtype,
    bool int8_queries,
    int64_t live_blocks) {
  const int kv_heads = static_cast<int>(params.kv_heads);
  const int group = num_q_heads / kv_heads;
  const int parts = paged_prefill_tile_parts(group);
  const int tiles = (static_cast<int>(params.rows) + kPagedPrefillTileRows - 1) /
      kPagedPrefillTileRows;
  const int64_t partial_floats = static_cast<int64_t>(tiles) * params.split_count *
      kPagedPrefillTileRows * num_q_heads * head_size;
  // A rank-4 K is the keys-token-major page; the validator admits it only
  // where its entry exists.
  const bool keys_token_major = k_pool.ndim() == 4;

  {
    const int kind = params.max_splits == kPagedVerifyMaxSplits ? 1 : 0;
    paged_prefill_probe_counter(kind).fetch_add(1, std::memory_order_relaxed);
    paged_prefill_probe_last_splits().store(params.split_count, std::memory_order_relaxed);
    paged_prefill_probe_last_max_splits().store(
        params.max_splits, std::memory_order_relaxed);
    if (int8_queries) {
      paged_prefill_probe_counter(3).fetch_add(1, std::memory_order_relaxed);
    }
    if (keys_token_major) {
      paged_prefill_probe_counter(4).fetch_add(1, std::memory_order_relaxed);
    }
  }

  {
    auto* pipeline = paged_prefill_pipeline(
        device,
        keys_token_major
            ? paged_prefill_keys_token_major_split_name(int8_queries)
            : paged_prefill_split_name(cache_dtype, int8_queries, group, head_size));
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(q_staged, 0);
    encoder.set_output_array(workspace, 1);
    encoder.set_output_array(workspace, 2, partial_floats * 4);
    encoder.set_input_array(block_table, 3);
    bind_pool_planes(encoder, k_pool, v_pool, k_scale, v_scale, live_blocks);
    encoder.set_bytes(params, 8);
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<size_t>(kv_heads * parts),
            static_cast<size_t>(tiles),
            static_cast<size_t>(params.split_count)),
        MTL::Size::Make(kNumThreads, 1, 1));
  }
  // The reduce reads every split's partials.
  encoder.barrier();
  {
    auto* pipeline = paged_prefill_pipeline(device, paged_prefill_reduce_name(group, head_size));
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(workspace, 0);
    encoder.set_input_array(workspace, 1, partial_floats * 4);
    encoder.set_output_array(out, 2);
    encoder.set_bytes(params, 3);
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<size_t>(kv_heads),
            static_cast<size_t>(kPagedPrefillTileRows * group),
            static_cast<size_t>(tiles)),
        MTL::Size::Make(static_cast<size_t>(head_size), 1, 1));
  }
}

void dispatch_paged_query_codes(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    const mlx::core::array& q,
    const mlx::core::array& positions,
    mlx::core::array& codes,
    mlx::core::array& score_scales,
    const PagedQueryCodesParams& params,
    int head_size) {
  auto* pipeline = paged_prefill_pipeline(device, paged_query_codes_name(head_size));
  encoder.set_compute_pipeline_state(pipeline);
  encoder.set_input_array(q, 0);
  encoder.set_input_array(positions, 1);
  encoder.set_output_array(codes, 2);
  encoder.set_output_array(score_scales, 3);
  encoder.set_bytes(params, 4);
  // Four threads per (row, query head): a threadgroup holds whole rows.
  const size_t threads =
      static_cast<size_t>(params.rows) * params.query_heads * 4;
  const size_t groups = (threads + kNumThreads - 1) / kNumThreads;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(groups, 1, 1), MTL::Size::Make(kNumThreads, 1, 1));
}

// =============================================================================
// Inkling-2-Mini Paged Attention dispatch (decode / varlen & MPP prefill)
// =============================================================================

namespace {

const std::string& inkling2_decode_kernel_name(
    KvDtype cache_dtype,
    int block_size,
    bool partitioned) {
  static const std::string ktm_v1 = "paged_attention_inkling2_ktm_hs128_ps0";
  static const std::string ktm_v2 = "paged_attention_inkling2_ktm_hs128_ps512";
  static const std::string bf16_b16_v1 = "paged_attention_inkling2_bf16_hs128_bs16_ps0";
  static const std::string bf16_b16_v2 = "paged_attention_inkling2_bf16_hs128_bs16_ps512";
  static const std::string bf16_b32_v1 = "paged_attention_inkling2_bf16_hs128_bs32_ps0";
  static const std::string bf16_b32_v2 = "paged_attention_inkling2_bf16_hs128_bs32_ps512";
  if (cache_dtype == KvDtype::Int8) {
    return partitioned ? ktm_v2 : ktm_v1;
  }
  if (block_size == 32) {
    return partitioned ? bf16_b32_v2 : bf16_b32_v1;
  }
  return partitioned ? bf16_b16_v2 : bf16_b16_v1;
}

const std::string& inkling2_decode_reduce_kernel_name() {
  static const std::string name = "paged_attention_inkling2_v2_reduce_hs128_ps512";
  return name;
}

const std::string& inkling2_prefill_split_name(
    KvDtype cache_dtype,
    bool int8_queries) {
  static const std::string q8_ktm = "inkling2_paged_prefill_q8_split_g3_d128_ktm";
  static const std::string q8q_ktm = "inkling2_paged_prefill_q8q_split_g3_d128_ktm";
  static const std::string bf16 = "inkling2_paged_prefill_bf16_split_g3_d128";
  if (cache_dtype == KvDtype::Int8) {
    return int8_queries ? q8q_ktm : q8_ktm;
  }
  return bf16;
}

const std::string& inkling2_prefill_reduce_name() {
  static const std::string name = "inkling2_paged_prefill_reduce_g3_d128";
  return name;
}

} // namespace

std::vector<std::string> inkling2_kernel_names() {
  return {
      "paged_attention_inkling2_ktm_hs128_ps0",
      "paged_attention_inkling2_ktm_hs128_ps512",
      "paged_attention_inkling2_bf16_hs128_bs16_ps0",
      "paged_attention_inkling2_bf16_hs128_bs16_ps512",
      "paged_attention_inkling2_bf16_hs128_bs32_ps0",
      "paged_attention_inkling2_bf16_hs128_bs32_ps512",
      "paged_attention_inkling2_v2_reduce_hs128_ps512",
      "inkling2_paged_prefill_q8_split_g3_d128_ktm",
      "inkling2_paged_prefill_q8q_split_g3_d128_ktm",
      "inkling2_paged_prefill_bf16_split_g3_d128",
      "inkling2_paged_prefill_reduce_g3_d128",
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
      "inkling2_bf16_linear_batched",
      "inkling2_bf16_lm_head_softcap_argmax_p1",
      "inkling2_moe_m1_gate_up_situ_nvfp4",
      "inkling2_moe_m1_gate_up_situ_q8",
      "inkling2_moe_m1_gate_up_situ_mxfp8",
      "inkling2_moe_m1_down_combine_nvfp4",
      "inkling2_moe_m1_down_combine_sconv_nvfp4",
      "inkling2_moe_m1_down_combine_q8",
      "inkling2_moe_m1_down_combine_sconv_q8",
      "inkling2_moe_m1_down_combine_mxfp8",
      "inkling2_moe_m1_down_combine_sconv_mxfp8",
      "inkling2_moe_group",
      "inkling2_moe_route_group_small",
      "inkling2_moe_gather",
      "inkling2_moe_simd_gate_up_situ_nvfp4",
      "inkling2_moe_simd_gate_up_situ_q8",
      "inkling2_moe_simd_gate_up_situ_mxfp8",
      "inkling2_moe_simd_down_nvfp4",
      "inkling2_moe_simd_down_q8",
      "inkling2_moe_simd_down_mxfp8",
      "inkling2_moe_gate_up_situ_nax_nvfp4_8",
      "inkling2_moe_gate_up_situ_nax_nvfp4_32",
      "inkling2_moe_gate_up_situ_nax_q8_8",
      "inkling2_moe_gate_up_situ_nax_q8_32",
      "inkling2_moe_down_nax_nvfp4_8",
      "inkling2_moe_down_nax_nvfp4_32",
      "inkling2_moe_down_nax_q8_8",
      "inkling2_moe_down_nax_q8_32",
      "inkling2_moe_grouped_combine_bf16",
      "inkling2_moe_grouped_combine_sconv_bf16",
  };
}

bool inkling2_paged_prefill_mpp_supported(mlx::core::metal::Device& device) {
  static const bool supported = [&]() noexcept {
    try {
      if (!device.mtl_device()->supportsFamily(MTL::GPUFamilyApple7)) {
        return false;
      }
      auto* q8 = paged_prefill_pipeline(device, inkling2_prefill_split_name(KvDtype::Int8, false));
      auto* bf = paged_prefill_pipeline(device, inkling2_prefill_split_name(KvDtype::Bf16, false));
      auto* red = paged_prefill_pipeline(device, inkling2_prefill_reduce_name());
      if (q8 == nullptr || bf == nullptr || red == nullptr) {
        return false;
      }
      if (paged_prefill_int8_queries_supported(device)) {
        auto* q8q = paged_prefill_pipeline(device, inkling2_prefill_split_name(KvDtype::Int8, true));
        if (q8q == nullptr) {
          return false;
        }
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }();
  return supported;
}

void dispatch_inkling2_paged_attention_auto(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::Stream stream,
    mlx::core::array& out,
    mlx::core::array& lse_out,
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    const mlx::core::array& cu_seqlens_q,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    const mlx::core::array* rel_bias,
    int num_seqs,
    int total_queries,
    int num_q_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int64_t live_blocks,
    float scale,
    int sliding_window,
    int rel_extent,
    KvDtype kv_dtype,
    const mlx::core::array* score_scales) {
  if (head_size != 128) {
    throw std::runtime_error(
        "[mlx_paged_dispatch] inkling2 paged attention requires head_size == 128");
  }
  const int32_t q_stride = num_q_heads * head_size;
  const int32_t kv_block_stride = num_kv_heads * head_size * block_size;
  const int32_t kv_head_stride = head_size * block_size;
  const int32_t rel_bias_is_f32 =
      (rel_bias != nullptr && rel_bias->dtype() == mlx::core::float32) ? 1 : 0;
  const float dummy_zero = 0.0f;
  const bool use_v1 = static_cast<uint32_t>(max_context_len) <= kPartitionSize;

  auto bind_common_stage1 = [&](bool is_ktm,
                                int64_t q_byte_off,
                                int64_t bias_byte_off,
                                int64_t score_scales_byte_off,
                                const mlx::core::array& seq_lens_arr,
                                int64_t seq_lens_off,
                                const mlx::core::array& cu_arr,
                                int64_t cu_off,
                                mlx::core::array* lse_opt,
                                int64_t lse_byte_off) {
    encoder.set_input_array(q, 3, q_byte_off);
    if (is_ktm) {
      bind_pool_planes(encoder, k_pool, v_pool, k_scale, v_scale, live_blocks);
      encoder.set_bytes<int32_t>(num_kv_heads, 8);
      encoder.set_bytes<float>(scale, 9);
      encoder.set_input_array(block_table, 10);
      encoder.set_input_array(seq_lens_arr, 11, seq_lens_off);
      encoder.set_bytes<int32_t>(max_blocks_per_seq, 12);
      encoder.set_bytes<int32_t>(q_stride, 13);
      encoder.set_bytes<int32_t>(kv_block_stride, 14);
      encoder.set_bytes<int32_t>(kv_head_stride, 15);
      encoder.set_bytes<int32_t>(sliding_window, 16);
      encoder.set_input_array(cu_arr, 17, cu_off);
      encoder.set_bytes<int32_t>(num_seqs, 18);
      if (rel_bias != nullptr && rel_extent > 0) {
        encoder.set_input_array(*rel_bias, 19, bias_byte_off);
      } else {
        encoder.set_bytes(dummy_zero, 19);
      }
      encoder.set_bytes<int32_t>(rel_extent, 20);
      encoder.set_bytes<int32_t>(rel_bias_is_f32, 21);
      if (score_scales != nullptr) {
        encoder.set_input_array(*score_scales, 22, score_scales_byte_off);
        encoder.set_bytes<int32_t>(1, 23);
      } else {
        encoder.set_bytes(dummy_zero, 22);
        encoder.set_bytes<int32_t>(0, 23);
      }
      if (lse_opt != nullptr) {
        encoder.set_output_array(*lse_opt, 24, lse_byte_off);
      } else {
        encoder.set_bytes(dummy_zero, 24);
      }
    } else {
      bind_pool_plane(encoder, k_pool, 4, live_blocks);
      bind_pool_plane(encoder, v_pool, 5, live_blocks);
      encoder.set_bytes<int32_t>(num_kv_heads, 6);
      encoder.set_bytes<float>(scale, 7);
      encoder.set_input_array(block_table, 8);
      encoder.set_input_array(seq_lens_arr, 9, seq_lens_off);
      encoder.set_bytes<int32_t>(max_blocks_per_seq, 10);
      encoder.set_bytes<int32_t>(q_stride, 11);
      encoder.set_bytes<int32_t>(kv_block_stride, 12);
      encoder.set_bytes<int32_t>(kv_head_stride, 13);
      encoder.set_bytes<int32_t>(sliding_window, 14);
      encoder.set_input_array(cu_arr, 15, cu_off);
      encoder.set_bytes<int32_t>(num_seqs, 16);
      if (rel_bias != nullptr && rel_extent > 0) {
        encoder.set_input_array(*rel_bias, 17, bias_byte_off);
      } else {
        encoder.set_bytes(dummy_zero, 17);
      }
      encoder.set_bytes<int32_t>(rel_extent, 18);
      encoder.set_bytes<int32_t>(rel_bias_is_f32, 19);
      if (lse_opt != nullptr) {
        encoder.set_output_array(*lse_opt, 20, lse_byte_off);
      } else {
        encoder.set_bytes(dummy_zero, 20);
      }
    }
  };

  const bool is_ktm = kv_dtype == KvDtype::Int8;
  if (use_v1) {
    const std::string& kernel_name =
        inkling2_decode_kernel_name(kv_dtype, block_size, /*partitioned=*/false);
    MTL::ComputePipelineState* pipeline = load_pipeline(device, kernel_name);
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_bytes(dummy_zero, 0);
    encoder.set_bytes(dummy_zero, 1);
    encoder.set_output_array(out, 2);
    bind_common_stage1(
        is_ktm, 0, 0, 0, seq_lens, 0, cu_seqlens_q, 0, &lse_out, 0);

    const size_t aligned_logits =
        (static_cast<size_t>(max_context_len) + 31) / 32 * 32;
    const size_t logits_bytes = aligned_logits * sizeof(float);
    const size_t v_reduce_bytes =
        static_cast<size_t>(kNumWarps / 2) * static_cast<size_t>(head_size) *
        sizeof(float);
    const size_t red_smem_bytes =
        2 * static_cast<size_t>(kNumWarps) * sizeof(float);
    const size_t threadgroup_mem =
        std::max(logits_bytes, v_reduce_bytes) + red_smem_bytes;
    encoder.set_threadgroup_memory_length(threadgroup_mem, 0);

    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<size_t>(num_q_heads),
            static_cast<size_t>(total_queries),
            1),
        MTL::Size::Make(kNumThreads, 1, 1));
    (void)stream;
    return;
  }

  // V2 partitioned path (context > 512) with single-sequence tiling to bound
  // temporary workspace memory even for large prefill fallbacks.
  const uint32_t max_num_partitions =
      (static_cast<uint32_t>(max_context_len) + kPartitionSize - 1) /
      kPartitionSize;
  const bool tiled = num_seqs == 1;
  const int tile_rows = tiled
      ? static_cast<int>(std::clamp<int64_t>(
            kVarlenWorkspaceRows / static_cast<int64_t>(max_num_partitions),
            1,
            total_queries))
      : total_queries;
  const int num_tiles = (total_queries + tile_rows - 1) / tile_rows;

  const int64_t exp_sums_size_i64 = static_cast<int64_t>(tile_rows) *
      static_cast<int64_t>(num_q_heads) *
      static_cast<int64_t>(max_num_partitions);
  const int64_t tmp_out_size_i64 =
      exp_sums_size_i64 * static_cast<int64_t>(head_size);
  if (exp_sums_size_i64 > std::numeric_limits<int>::max() ||
      tmp_out_size_i64 > std::numeric_limits<int>::max()) {
    throw std::runtime_error(
        "[mlx_paged_dispatch] Inkling2 V2 auxiliary buffer size exceeds INT_MAX");
  }
  const int exp_sums_size = static_cast<int>(exp_sums_size_i64);
  const int tmp_out_size = static_cast<int>(tmp_out_size_i64);

  mlx::core::array exp_sums(
      mlx::core::Shape{exp_sums_size}, mlx::core::float32, nullptr, {});
  exp_sums.set_data(mlx::core::allocator::malloc(exp_sums.nbytes()));
  mlx::core::array max_logits(
      mlx::core::Shape{exp_sums_size}, mlx::core::float32, nullptr, {});
  max_logits.set_data(mlx::core::allocator::malloc(max_logits.nbytes()));
  mlx::core::array tmp_out(
      mlx::core::Shape{tmp_out_size}, mlx::core::float32, nullptr, {});
  tmp_out.set_data(mlx::core::allocator::malloc(tmp_out.nbytes()));

  mlx::core::array tile_meta(
      mlx::core::Shape{3 * num_tiles}, mlx::core::int32, nullptr, {});
  if (tiled) {
    tile_meta.set_data(mlx::core::allocator::malloc(tile_meta.nbytes()));
    int32_t* meta = tile_meta.data<int32_t>();
    for (int t = 0; t < num_tiles; ++t) {
      const int end = std::min(total_queries, (t + 1) * tile_rows);
      meta[3 * t] = max_context_len - (total_queries - end);
      meta[3 * t + 1] = 0;
      meta[3 * t + 2] = end - t * tile_rows;
    }
  }

  const size_t stage1_logits_bytes =
      static_cast<size_t>(kPartitionSize) * sizeof(float);
  const size_t stage1_v_reduce_bytes =
      static_cast<size_t>(kNumWarps / 2) * static_cast<size_t>(head_size) *
      sizeof(float);
  const size_t stage1_red_smem_bytes =
      2 * static_cast<size_t>(kNumWarps) * sizeof(float);
  const size_t stage1_tg_mem =
      std::max(stage1_logits_bytes, stage1_v_reduce_bytes) +
      stage1_red_smem_bytes;
  const size_t reduce_tg_mem = std::max<size_t>(
      16, static_cast<size_t>(max_num_partitions) * sizeof(float));

  const std::string& stage1_name =
      inkling2_decode_kernel_name(kv_dtype, block_size, /*partitioned=*/true);
  MTL::ComputePipelineState* stage1_pipe = load_pipeline(device, stage1_name);
  MTL::ComputePipelineState* reduce_pipe =
      load_pipeline(device, inkling2_decode_reduce_kernel_name());

  for (int t = 0; t < num_tiles; ++t) {
    const int first = t * tile_rows;
    const int rows = std::min(tile_rows, total_queries - first);
    const int64_t q_offset =
        static_cast<int64_t>(first) * q_stride * q.itemsize();
    const int64_t out_offset =
        static_cast<int64_t>(first) * q_stride * out.itemsize();
    const int64_t lse_offset =
        static_cast<int64_t>(first) * num_q_heads * static_cast<int64_t>(sizeof(float));
    const int64_t bias_offset = (rel_bias != nullptr && rel_extent > 0)
        ? static_cast<int64_t>(first) * num_q_heads * rel_extent *
              static_cast<int64_t>(rel_bias->itemsize())
        : 0;
    const int64_t ss_offset =
        static_cast<int64_t>(first) * num_q_heads * static_cast<int64_t>(sizeof(float));
    const mlx::core::array& seq_arr = tiled ? tile_meta : seq_lens;
    const int64_t seq_off = tiled ? int64_t{12} * t : 0;
    const mlx::core::array& cu_arr = tiled ? tile_meta : cu_seqlens_q;
    const int64_t cu_off = tiled ? int64_t{12} * t + 4 : 0;

    if (t > 0) {
      encoder.barrier();
    }

    // Stage 1
    encoder.set_compute_pipeline_state(stage1_pipe);
    encoder.set_output_array(exp_sums, 0);
    encoder.set_output_array(max_logits, 1);
    encoder.set_output_array(tmp_out, 2);
    bind_common_stage1(
        is_ktm, q_offset, bias_offset, ss_offset, seq_arr, seq_off, cu_arr,
        cu_off, nullptr, 0);
    encoder.set_threadgroup_memory_length(stage1_tg_mem, 0);
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<size_t>(num_q_heads),
            static_cast<size_t>(rows),
            static_cast<size_t>(max_num_partitions)),
        MTL::Size::Make(kNumThreads, 1, 1));

    encoder.barrier();

    // Stage 2: shared FP32 reducer -> BF16 out + F32 lse
    encoder.set_compute_pipeline_state(reduce_pipe);
    encoder.set_output_array(out, 0, out_offset);
    encoder.set_output_array(lse_out, 1, lse_offset);
    encoder.set_input_array(exp_sums, 2);
    encoder.set_input_array(max_logits, 3);
    encoder.set_input_array(tmp_out, 4);
    encoder.set_input_array(seq_arr, 5, seq_off);
    encoder.set_bytes<int32_t>(static_cast<int32_t>(max_num_partitions), 6);
    encoder.set_input_array(cu_arr, 7, cu_off);
    encoder.set_bytes<int32_t>(num_seqs, 8);
    encoder.set_threadgroup_memory_length(reduce_tg_mem, 0);
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<size_t>(num_q_heads), static_cast<size_t>(rows), 1),
        MTL::Size::Make(kNumThreads, 1, 1));
  }

  if (tiled) {
    encoder.add_temporary(std::move(tile_meta));
  }
  encoder.add_temporary(std::move(exp_sums));
  encoder.add_temporary(std::move(max_logits));
  encoder.add_temporary(std::move(tmp_out));
  (void)stream;
}

void dispatch_inkling2_paged_prefill_attention(
    mlx::core::metal::CommandEncoder& encoder,
    mlx::core::metal::Device& device,
    mlx::core::array& out,
    mlx::core::array& lse_out,
    mlx::core::array& workspace,
    const mlx::core::array& q_staged,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& k_scale,
    const mlx::core::array& v_scale,
    const mlx::core::array& block_table,
    const mlx::core::array* rel_bias,
    const Inkling2PagedPrefillParams& params,
    int num_q_heads,
    int head_size,
    KvDtype cache_dtype,
    bool int8_queries,
    int64_t live_blocks) {
  const int kv_heads = static_cast<int>(params.kv_heads);
  const int group = num_q_heads / kv_heads;
  const int tiles = (static_cast<int>(params.rows) + kPagedPrefillTileRows - 1) /
      kPagedPrefillTileRows;
  const int64_t partial_floats = static_cast<int64_t>(tiles) *
      params.split_count * kPagedPrefillTileRows * num_q_heads * head_size;
  const float dummy_zero = 0.0f;

  // Stage 1: MPP split
  {
    auto* pipeline = paged_prefill_pipeline(
        device, inkling2_prefill_split_name(cache_dtype, int8_queries));
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(q_staged, 0);
    encoder.set_output_array(workspace, 1);
    encoder.set_output_array(workspace, 2, partial_floats * 4);
    encoder.set_input_array(block_table, 3);
    if (cache_dtype == KvDtype::Int8) {
      bind_pool_planes(encoder, k_pool, v_pool, k_scale, v_scale, live_blocks);
      if (rel_bias != nullptr && params.rel_extent > 0) {
        encoder.set_input_array(*rel_bias, 8);
      } else {
        encoder.set_bytes(dummy_zero, 8);
      }
      encoder.set_bytes(params, 9);
    } else {
      bind_pool_plane(encoder, k_pool, 4, live_blocks);
      bind_pool_plane(encoder, v_pool, 5, live_blocks);
      if (rel_bias != nullptr && params.rel_extent > 0) {
        encoder.set_input_array(*rel_bias, 6);
      } else {
        encoder.set_bytes(dummy_zero, 6);
      }
      encoder.set_bytes(params, 7);
    }
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<size_t>(kv_heads),
            static_cast<size_t>(tiles),
            static_cast<size_t>(params.split_count)),
        MTL::Size::Make(kNumThreads, 1, 1));
  }

  encoder.barrier();

  // Stage 2: reduce -> BF16 out + F32 lse
  {
    auto* pipeline =
        paged_prefill_pipeline(device, inkling2_prefill_reduce_name());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(workspace, 0);
    encoder.set_input_array(workspace, 1, partial_floats * 4);
    encoder.set_output_array(out, 2);
    encoder.set_output_array(lse_out, 3);
    encoder.set_bytes(params, 4);
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<size_t>(kv_heads),
            static_cast<size_t>(kPagedPrefillTileRows * group),
            static_cast<size_t>(tiles)),
        MTL::Size::Make(static_cast<size_t>(head_size), 1, 1));
  }
}

} // namespace mlx::core::fast::paged

extern "C" bool mlx_inkling2_paged_prefill_mpp_supported() {
  using namespace mlx::core;
  try {
    if (!metal::is_available()) {
      return false;
    }
    return fast::paged::inkling2_paged_prefill_mpp_supported(
        metal::device(Device::gpu));
  } catch (const std::exception&) {
    return false;
  }
}

extern "C" bool mlx_paged_prefill_mpp_builds() {
  using namespace mlx::core;
  fast::paged::paged_prefill_probe_counter(2).fetch_add(1, std::memory_order_relaxed);
  try {
    if (!metal::is_available()) {
      return false;
    }
    fast::paged::build_paged_prefill_attention_pipelines(metal::device(Device::gpu));
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

extern "C" void mlx_paged_prefill_test_probe_reset() {
  using namespace mlx::core::fast::paged;
  paged_prefill_probe_counter(0).store(0, std::memory_order_relaxed);
  paged_prefill_probe_counter(1).store(0, std::memory_order_relaxed);
  paged_prefill_probe_counter(3).store(0, std::memory_order_relaxed);
  paged_prefill_probe_counter(4).store(0, std::memory_order_relaxed);
  paged_prefill_probe_last_splits().store(0, std::memory_order_relaxed);
  paged_prefill_probe_last_max_splits().store(0, std::memory_order_relaxed);
}

// kind 0: prefill launches, 1: verify launches, 2: build probes (never
// reset), 3: launches of a q8q (INT8-query) entry, 4: launches of a
// keys-token-major entry.
extern "C" uint64_t mlx_paged_prefill_test_probe_count(int kind) {
  using namespace mlx::core::fast::paged;
  if (kind < 0 || kind > 4) {
    return 0;
  }
  return paged_prefill_probe_counter(kind).load(std::memory_order_relaxed);
}

extern "C" uint32_t mlx_paged_prefill_test_last_splits() {
  using namespace mlx::core::fast::paged;
  return paged_prefill_probe_last_splits().load(std::memory_order_relaxed);
}

extern "C" uint32_t mlx_paged_prefill_test_last_max_splits() {
  using namespace mlx::core::fast::paged;
  return paged_prefill_probe_last_max_splits().load(std::memory_order_relaxed);
}
