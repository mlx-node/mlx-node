// TEST-ONLY hook: checks paged_attn.metallib against the segmented SDPA and
// mixed-affine kernels their dispatchers can request. No production caller.

#include <cstdint>
#include <cstring>
#include <iostream>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx_affine_mixed_qmm.h"
#include "mlx_paged_metallib.h"
#include "mlx_segmented_sdpa.h"

#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#endif

// `family` is "segmented_sdpa" or "affine_mixed". `counts` receives {names the
// dispatcher can request, the family's functions in the library, pipelines
// built}. `report` receives "missing <name>" / "unexpected <name>" lines.
// `build_pipelines` also builds every pipeline the dispatcher can build (each
// function-constant specialization for segmented SDPA) through the
// dispatcher's own loader. False without Metal, for an unknown family, on
// error, or when `report` cannot hold the text.
extern "C" bool mlx_test_bridge_metallib_check(const char *family,
                                               bool build_pipelines,
                                               int64_t *counts, char *report,
                                               size_t len) {
#ifdef MLX_NODE_METAL_ENABLED
  using namespace mlx::core;
  try {
    if (!family || !counts || !report || len == 0 || !metal::is_available())
      return false;
    auto &d = metal::device(Device::gpu);

    std::string prefix;
    std::vector<std::string> names;
    std::function<int64_t()> build;
    if (std::strcmp(family, "segmented_sdpa") == 0) {
      prefix = "mlx_node_sdpa_segmented";
      names = segmented_sdpa::metal_kernel_names();
      build = [&d] {
        int64_t built = 0;
        for (const auto &sp : segmented_sdpa::metal_kernel_specializations()) {
          segmented_sdpa::segmented_kernel(d, sp);
          ++built;
        }
        return built;
      };
    } else if (std::strcmp(family, "affine_mixed") == 0) {
      prefix = "mlx_node_affine_qmv_wide_mixed";
      for (int width : affine_mixed::qmv_wide_widths())
        names.push_back(affine_mixed::qmv_wide_kernel_name(width));
      build = [&d] {
        int64_t built = 0;
        for (int width : affine_mixed::qmv_wide_widths()) {
          affine_mixed::qmv_wide_kernel(d, width);
          ++built;
        }
        return built;
      };
    } else {
      return false;
    }

    auto *lib = fast::paged::get_paged_attn_library(d);
    std::set<std::string> in_library;
    NS::Array *functions = lib->functionNames();
    for (NS::UInteger i = 0; i < functions->count(); ++i) {
      std::string name = functions->object<NS::String>(i)->utf8String();
      if (name.rfind(prefix, 0) == 0)
        in_library.insert(std::move(name));
    }

    std::ostringstream out;
    std::set<std::string> known(names.begin(), names.end());
    for (const auto &name : names) {
      if (!in_library.count(name))
        out << "missing " << name << "\n";
    }
    for (const auto &name : in_library) {
      if (!known.count(name))
        out << "unexpected " << name << "\n";
    }
    std::string text = out.str();
    int64_t built = 0;
    if (build_pipelines && text.empty())
      built = build();

    counts[0] = static_cast<int64_t>(names.size());
    counts[1] = static_cast<int64_t>(in_library.size());
    counts[2] = built;
    if (text.size() + 1 > len)
      return false;
    std::memcpy(report, text.c_str(), text.size() + 1);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_bridge_metallib_check: " << e.what() << std::endl;
    return false;
  }
#else
  (void)family;
  (void)build_pipelines;
  (void)counts;
  (void)report;
  (void)len;
  return false;
#endif
}
