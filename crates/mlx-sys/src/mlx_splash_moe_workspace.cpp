#include "mlx_splash_moe_workspace.h"

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <utility>

namespace mlx::core::splash_moe::persistent_workspace {
namespace {

struct Entry {
  Entry(uint64_t identity, array owner)
      : identity(identity), owner(std::move(owner)) {}

  uint64_t identity;
  array owner;
};

using Pool = std::map<Stream, Entry>;

Pool &pool() {
  static thread_local Pool entries;
  return entries;
}

metal::CommandEncoder *find_encoder(Stream stream) {
  auto &local = metal::get_command_encoders();
  auto it = local.find(stream.index);
  if (it != local.end())
    return &it->second;
  auto &global = metal::get_global_command_encoders();
  it = global.find(stream.index);
  return it == global.end() ? nullptr : &it->second;
}

void reap_destroyed(Pool &entries) {
  for (auto it = entries.begin(); it != entries.end();) {
    // A missing encoder has already synchronized in its destructor.
    if (find_encoder(it->first)) {
      ++it;
    } else {
      it = entries.erase(it);
    }
  }
}

void trace_workspace(Stream stream, metal::CommandEncoder &encoder,
                     uint64_t identity, size_t slots, size_t bytes,
                     bool reused) {
  if (encoder.trace_level()) {
    std::fprintf(stderr,
                 "[metal-workspace] stream=%d identity=0x%016llx selected=1 "
                 "reused=%d slots=%zu bytes=%zu\n",
                 stream.index, static_cast<unsigned long long>(identity),
                 int(reused), slots, bytes);
  }
}

template <typename Retain>
const std::vector<array> *
acquire(Stream stream, metal::CommandEncoder &encoder, uint64_t identity,
        const std::vector<Spec> &specs, Retain &&retain) {
  constexpr size_t max_bytes = size_t(8) << 20;
  auto &entries = pool();
  if (entries.find(stream) == entries.end()) {
    // Stream indices are never reused: each first acquisition bounds stale
    // profiles left by ordinary clear_streams() before adding a new one.
    reap_destroyed(entries);
  }
  auto existing = entries.find(stream);
  if (identity == 0 || specs.empty() || specs.size() > 16 ||
      (existing != entries.end() && existing->second.identity != identity)) {
    return nullptr;
  }

  size_t total = 0;
  for (const auto &spec : specs) {
    const size_t itemsize = size_of(spec.dtype);
    if (!itemsize)
      return nullptr;
    size_t elements = 1;
    for (int dim : spec.shape) {
      if (dim <= 0 || size_t(dim) > (max_bytes / itemsize) / elements)
        return nullptr;
      elements *= size_t(dim);
    }
    const size_t bytes = elements * itemsize;
    if (bytes > max_bytes - total)
      return nullptr;
    total += bytes;
  }

  if (existing != entries.end()) {
    const auto &slots = existing->second.owner.inputs();
    if (slots.size() != specs.size())
      return nullptr;
    for (size_t i = 0; i < specs.size(); ++i) {
      if (slots[i].shape() != specs[i].shape ||
          slots[i].dtype() != specs[i].dtype)
        return nullptr;
    }
    retain(encoder, existing->second.owner);
    trace_workspace(stream, encoder, identity, specs.size(), total, true);
    return &slots;
  }

  std::vector<array> prepared;
  prepared.reserve(specs.size());
  for (const auto &spec : specs) {
    prepared.emplace_back(spec.shape, spec.dtype, nullptr,
                          std::vector<array>{});
    prepared.back().set_data(allocator::malloc(prepared.back().nbytes()));
  }
  array owner(Shape{}, uint8, nullptr, std::move(prepared));
  // Partial retention pins stay owned until coordinated graph cleanup. A
  // local synchronize can deadlock an uncommitted producer stream.
  retain(encoder, owner);
  auto inserted = entries.emplace(stream, Entry(identity, std::move(owner)));
  if (!inserted.second)
    throw std::runtime_error("persistent workspace publication race");
  trace_workspace(stream, encoder, identity, specs.size(), total, false);
  return &inserted.first->second.owner.inputs();
}

} // namespace

const std::vector<array> *try_workspace(Stream stream,
                                        metal::CommandEncoder &encoder,
                                        uint64_t identity,
                                        const std::vector<Spec> &specs) {
  return acquire(stream, encoder, identity, specs,
                 [](metal::CommandEncoder &enc, const array &owner) {
                   enc.retain_inputs(owner);
                 });
}

void drain(Stream stream) {
  auto &entries = pool();
  reap_destroyed(entries);
  if (auto *encoder = find_encoder(stream))
    encoder->synchronize();
  entries.erase(stream);
}

void clear_streams_and_workspaces() {
  auto &entries = pool();
  reap_destroyed(entries);
  // Worker clear_streams does not destroy global encoders; drain every profile.
  for (const auto &entry : entries) {
    if (auto *encoder = find_encoder(entry.first))
      encoder->synchronize();
  }
  mlx::core::clear_streams();
  entries.clear();
}

namespace testing {
const std::vector<array> *
try_workspace_with_retention(Stream stream, metal::CommandEncoder &encoder,
                             uint64_t identity, const std::vector<Spec> &specs,
                             Retention retention) {
  return acquire(stream, encoder, identity, specs, std::move(retention));
}

bool has_profile(Stream stream) {
  auto &entries = pool();
  reap_destroyed(entries);
  return entries.find(stream) != entries.end();
}
} // namespace testing

} // namespace mlx::core::splash_moe::persistent_workspace
#endif
