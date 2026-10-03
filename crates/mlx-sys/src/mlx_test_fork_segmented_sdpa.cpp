// TEST-ONLY oracle: the MLX fork's segmented SDPA kernels and planners (fork
// commits 03914b9b3, a6189690e). Only the fork pinned at 053e43fec ships them;
// delete this file with the pin move.

#include "mlx_common.h"

#ifdef MLX_NODE_METAL_ENABLED

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/segmented_sdpa_plan.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/sdpa_vector_plan.h"
#include "mlx/transforms.h"
#include "mlx_segmented_sdpa.h"
#include "mlx_segmented_sdpa_plan.h"

namespace {

namespace fork = mlx::core::fast;
namespace ours = mlx::core::segmented_sdpa;
using mlx::core::array;

MTL::ComputePipelineState *fork_kernel(mlx::core::metal::Device &device,
                                       ours::SegmentedKernel kernel,
                                       const std::string &hash,
                                       const mlx::core::metal::MTLFCList &fc) {
  const char *name = nullptr;
  switch (kernel) {
  case ours::SegmentedKernel::one_pass:
    name = "sdpa_vector_segmented_bfloat16_t_256_256";
    break;
  case ours::SegmentedKernel::two_pass_1:
    name = "sdpa_vector_segmented_2pass_1_bfloat16_t_256_256";
    break;
  case ours::SegmentedKernel::verify_two_pass_1:
    name = "sdpa_vector_segmented_verify_2pass_1_bfloat16_t_256_256";
    break;
  }
  return device.get_kernel(name, "fork_" + hash, fc);
}

struct ForkKernels {
  ForkKernels() { ours::testing::kernel_override = &fork_kernel; }
  ~ForkKernels() { ours::testing::kernel_override = nullptr; }
};

mlx::core::metal::Device &gpu() {
  return mlx::core::metal::device(mlx::core::Device::gpu);
}

bool same(const fork::SegmentedSdpaLaunchPlan &a,
          const ours::SegmentedSdpaLaunchPlan &b) {
  return a.supported == b.supported && a.two_pass == b.two_pass &&
         a.partitions == b.partitions && a.stage1_threads == b.stage1_threads &&
         a.stage2_threads == b.stage2_threads;
}

} // namespace

extern "C" {

// The production op with every segmented kernel taken from the fork's
// mlx.metallib; evaluated before returning so the swap covers dispatch.
mlx_array *mlx_test_fork_segmented_sdpa_forward(mlx_array *q, mlx_array *pk,
                                                mlx_array *pv, mlx_array *nk,
                                                mlx_array *nv, float scale,
                                                bool causal) {
  try {
    ForkKernels swap;
    auto out = ours::segmented_sdpa(
        *reinterpret_cast<array *>(q), *reinterpret_cast<array *>(pk),
        *reinterpret_cast<array *>(pv), *reinterpret_cast<array *>(nk),
        *reinterpret_cast<array *>(nv), scale, causal, true);
    mlx::core::eval(out);
    return reinterpret_cast<mlx_array *>(new array(std::move(out)));
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mlx_test_fork_segmented_sdpa_forward: %s\n",
                 e.what());
    return nullptr;
  }
}

int mlx_test_fork_segmented_sdpa_max_query_length(int gqa_factor) {
  try {
    ForkKernels swap;
    return ours::segmented_max_query_length(gpu(), gqa_factor);
  } catch (const std::exception &) {
    return 0;
  }
}

// Mismatches between the fork's planners and ours over a sweep of inputs,
// including the fork's vector-SDPA policy on this device; -1 on error.
int64_t mlx_test_fork_segmented_sdpa_plan_mismatches(int64_t *out_checked) {
  try {
    int64_t checked = 0;
    int64_t mismatches = 0;
    auto &device = gpu();
    const char device_class = device.get_architecture().back();
    // Every length to 2048, then a stride, plus each power-of-two policy
    // boundary up to 2^17 and its neighbours.
    std::vector<int> lengths;
    for (int length = 1; length <= 140000;
         length += length < 2048 ? 1 : (length < 70000 ? 7 : 997)) {
      lengths.push_back(length);
    }
    for (int boundary = 2048; boundary <= (1 << 17); boundary *= 2) {
      for (int d = -2; d <= 2; ++d) {
        lengths.push_back(boundary + d);
      }
    }
    for (int length : lengths) {
      for (int q_heads : {1, 2, 4, 6, 8, 12, 16, 24, 32, 64}) {
        for (int kv_heads : {1, 2, 4, 8}) {
          if (q_heads % kv_heads != 0) {
            continue;
          }
          ++checked;
          mismatches += fork::sdpa_vector_uses_two_pass(device, length, q_heads,
                                                        kv_heads) !=
                        ours::sdpa_vector_uses_two_pass(device_class, length,
                                                        q_heads, kv_heads);
        }
      }
      for (int simdgroups = 1; simdgroups <= 256; ++simdgroups) {
        ++checked;
        mismatches +=
            fork::sdpa_vector_partition_count(device, length, simdgroups) !=
            ours::sdpa_vector_partition_count(
                device_class, length, simdgroups,
                mlx::core::env::get_var("MLX_SDPA_BLOCKS", 0));
      }
    }

    const size_t widths[] = {16, 32, 64};
    const size_t threads[] = {0, 32, 512, 960, 1023, 1024, 2048};
    const size_t memory[] = {0, 4096, 32768, 65536};
    for (int rows = -1; rows <= 10; ++rows) {
      for (int max_q = -1; max_q <= 10; ++max_q) {
        ++checked;
        mismatches += fork::segmented_verify_head_rows(rows, max_q) !=
                      ours::segmented_verify_head_rows(rows, max_q);
      }
      for (int gqa = 0; gqa <= 33; ++gqa) {
        for (int partitions : {0, 16, 31, 32, 48, 64, 128, 1024}) {
          for (size_t w : widths) {
            for (size_t t : threads) {
              for (size_t m : memory) {
                fork::SegmentedSdpaCapabilities f1{w, t, m, 32768};
                ours::SegmentedSdpaCapabilities o1{w, t, m, 32768};
                fork::SegmentedSdpaCapabilities f2{32, t, m, 32768};
                ours::SegmentedSdpaCapabilities o2{32, t, m, 32768};
                for (bool two_pass : {false, true}) {
                  ++checked;
                  mismatches +=
                      !same(fork::plan_segmented_sdpa_launch(
                                rows, gqa, two_pass, partitions, f1, &f2),
                            ours::plan_segmented_sdpa_launch(
                                rows, gqa, two_pass, partitions, o1, &o2));
                  ++checked;
                  mismatches +=
                      !same(fork::plan_segmented_sdpa_launch(
                                rows, gqa, two_pass, partitions, f1, nullptr),
                            ours::plan_segmented_sdpa_launch(
                                rows, gqa, two_pass, partitions, o1, nullptr));
                }
                ++checked;
                mismatches += !same(fork::plan_segmented_verify_launch(
                                        rows, gqa, partitions, f1, f2),
                                    ours::plan_segmented_verify_launch(
                                        rows, gqa, partitions, o1, o2));
              }
            }
          }
        }
      }
    }
    for (int head_rows = 0; head_rows <= 8; ++head_rows) {
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          const int parts[] = {32, 64, 128, 256};
          for (bool unified : {false, true}) {
            fork::SegmentedSdpaReductionPlan fh{(a & 1) != 0, parts[a]};
            fork::SegmentedSdpaReductionPlan ft{(b & 1) != 0, parts[b]};
            ++checked;
            mismatches += static_cast<int>(fork::select_segmented_verify_route(
                              head_rows, fh, ft, unified)) !=
                          static_cast<int>(ours::select_segmented_verify_route(
                              head_rows, {fh.two_pass, fh.partitions},
                              {ft.two_pass, ft.partitions}, unified));
          }
        }
      }
    }
    if (out_checked) {
      *out_checked = checked;
    }
    return mismatches;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mlx_test_fork_segmented_sdpa_plan_mismatches: %s\n",
                 e.what());
    return -1;
  }
}

} // extern "C"

#else

extern "C" mlx_array *
mlx_test_fork_segmented_sdpa_forward(mlx_array *, mlx_array *, mlx_array *,
                                     mlx_array *, mlx_array *, float, bool) {
  return nullptr;
}

extern "C" int mlx_test_fork_segmented_sdpa_max_query_length(int) { return 0; }

extern "C" int64_t mlx_test_fork_segmented_sdpa_plan_mismatches(int64_t *) {
  return -1;
}

#endif
