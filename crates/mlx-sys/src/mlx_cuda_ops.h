#pragma once

#ifdef MLX_NODE_CUDA_ENABLED
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>

#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/gpu/copy.h"

namespace mlx::core::cu {

// Deterministic 64-bit FNV-1a hash for stable NVRTC module names across platforms.
inline uint64_t fnv1a_64(std::string_view data) noexcept {
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : data) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

// Keep non-contiguous inputs alive through the encoded CUDA operation.
inline std::vector<array> contiguous_inputs(
    const std::vector<array>& inputs, const Stream& stream, CommandEncoder& encoder) {
  std::vector<array> owned;
  owned.reserve(inputs.size());
  for (const auto& input : inputs) {
    if (input.flags().row_contiguous) {
      owned.push_back(input);
    } else {
      auto copy = contiguous_copy_gpu(input, stream);
      encoder.add_temporary(copy);
      owned.push_back(std::move(copy));
    }
  }
  return owned;
}

// Ensure an array is both row-contiguous and base-pointer aligned to Align bytes
// (needed when vectorized uint2/uint4 kernels receive sliced views with non-zero offset).
template <size_t Align>
inline array ensure_row_contiguous_aligned(
    const array& input, const Stream& stream, CommandEncoder& encoder) {
  static_assert(Align > 0 && (Align & (Align - 1)) == 0, "Align must be a power of 2");
  if (input.flags().row_contiguous &&
      (reinterpret_cast<uintptr_t>(gpu_ptr<void>(input)) & (Align - 1)) == 0) {
    return input;
  }
  auto copy = contiguous_copy_gpu(input, stream);
  encoder.add_temporary(copy);
  return copy;
}

// NVRTC modules are keyed by source, entry point, and GPU architecture.
// Cache resolved CUfunction handles by static source pointer + entry string_view
// (backed by thread-local deque storage) so hot-path eval_gpu dispatches perform
// zero heap allocations and never re-hash kilobytes of CUDA source text.
inline CUfunction jit_kernel(
    Device& device, std::string_view source, std::string_view entry) {
  struct Key {
    int dev;
    const char* src_ptr;
    size_t src_len;
    std::string_view entry;
    bool operator==(const Key& o) const noexcept {
      return dev == o.dev && src_ptr == o.src_ptr && src_len == o.src_len &&
          entry == o.entry;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
      size_t h = std::hash<int>{}(k.dev);
      h ^= std::hash<const void*>{}(k.src_ptr) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<size_t>{}(k.src_len) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<std::string_view>{}(k.entry) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };
  struct Cache {
    std::deque<std::string> entry_storage;
    std::unordered_map<Key, CUfunction, KeyHash> map;
  };
  thread_local Cache cache;
  const Key lookup{device.cuda_device(), source.data(), source.size(), entry};
  if (auto it = cache.map.find(lookup); it != cache.map.end()) {
    return it->second;
  }
  const auto source_hash = fnv1a_64(source);
  std::string safe_entry(entry);
  for (char& c : safe_entry) {
    switch (c) {
      case '<':
      case '>':
      case ':':
      case '"':
      case '/':
      case '\\':
      case '|':
      case '?':
      case '*':
      case ',':
      case ' ':
        c = '_';
        break;
      default:
        break;
    }
  }
  const auto module_name = safe_entry + "_" + std::to_string(source_hash) +
      "_" + std::to_string(device.compute_capability_major()) + "_" +
      std::to_string(device.compute_capability_minor());
  auto& module = get_jit_module(device, module_name, [&] {
    return std::make_tuple(
        false, std::string(source), std::vector<std::string>{std::string(entry)});
  });
  CUfunction fn = module.get_kernel(std::string(entry));
  cache.entry_storage.emplace_back(entry);
  const Key stored{
      device.cuda_device(), source.data(), source.size(),
      std::string_view(cache.entry_storage.back())};
  cache.map.emplace(stored, fn);
  return fn;
}

} // namespace mlx::core::cu
#endif
