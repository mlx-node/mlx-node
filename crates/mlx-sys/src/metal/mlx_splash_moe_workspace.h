#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace mlx::core::metal {
class CommandEncoder;
}

namespace mlx::core::splash_moe::persistent_workspace {

struct Spec {
  Shape shape;
  Dtype dtype;
};

#ifdef MLX_NODE_METAL_ENABLED
// Grouped-MoE calls this only outside CommandEncoder::ConcurrentContext.
const std::vector<array> *try_workspace(Stream stream,
                                        metal::CommandEncoder &encoder,
                                        uint64_t identity,
                                        const std::vector<Spec> &specs);

// Synchronize before releasing a live profile. Failed synchronization retains
// it; a destroyed encoder has already synchronized and its profile is reaped.
void drain(Stream stream);
void clear_streams_and_workspaces();

namespace testing {
using Retention = std::function<void(metal::CommandEncoder &, const array &)>;
const std::vector<array> *
try_workspace_with_retention(Stream stream, metal::CommandEncoder &encoder,
                             uint64_t identity, const std::vector<Spec> &specs,
                             Retention retention);
bool has_profile(Stream stream);
} // namespace testing
#endif

} // namespace mlx::core::splash_moe::persistent_workspace
