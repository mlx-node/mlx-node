// Grouped & Fused NVFP4 / Q8_0 MoE primitive for Inkling-2-Mini on Apple Silicon
// (`src/metal/inkling2/inkling2_moe.metal`, compiled into `paged_attn.metallib`).
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "metal/inkling2/inkling2_moe_abi.h"

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx_paged_metallib.h"
#include "mlx_splash_moe_workspace.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlx::core::inkling2_moe {
namespace {

class Inkling2GroupedMoe : public fast::Custom {
 public:
  Inkling2GroupedMoe(Stream s, Inkling2MoeParams params, bool has_gu_aux, bool has_dn_aux)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2GroupedMoe requires Metal");
            }),
        p_(params),
        has_gu_aux_(has_gu_aux),
        has_dn_aux_(has_dn_aux) {}

  DEFINE_NAME(Inkling2GroupedMoe)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2GroupedMoe&>(other);
    return has_gu_aux_ == rhs.has_gu_aux_ &&
        has_dn_aux_ == rhs.has_dn_aux_ &&
        std::memcmp(&p_, &rhs.p_, sizeof(p_)) == 0;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2GroupedMoe requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;

 private:
  Inkling2MoeParams p_;
  bool has_gu_aux_;
  bool has_dn_aux_;
};

#ifdef MLX_NODE_METAL_ENABLED
inline array ensure_aligned_contiguous(
    const array& input,
    Stream s,
    metal::CommandEncoder& enc) {
  if (input.flags().row_contiguous && (input.offset() % 16 == 0)) {
    return input;
  }
  array copy(input.shape(), input.dtype(), nullptr, {});
  copy.set_data(allocator::malloc(copy.nbytes()));
  copy_gpu(
      input,
      copy,
      input.ndim() == 0 ? CopyType::Scalar : CopyType::General,
      s);
  enc.add_temporary(copy);
  return copy;
}

void Inkling2GroupedMoe::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& enc = metal::get_command_encoder(s);

  // Normalize all inputs strictly BEFORE binding any pipeline or arguments
  // (copy_gpu can rebind the active compute pipeline on the encoder).
  std::vector<array> a;
  a.reserve(inputs.size());
  for (const auto& inp : inputs) {
    a.push_back(ensure_aligned_contiguous(inp, s, enc));
  }

  Inkling2MoeParams p = p_;
  p.rows = static_cast<uint32_t>(a[0].shape(0));
  const bool is_q8 = (p.flags & 1u) != 0u;

  // Input layout:
  //   0: x [T, D] BF16
  //   1: top_k_indices [T, K] Int32
  //   2: gamma [T, K] F32
  //   3: gu_w [E, 2*I, K_gu_packed] uint32
  //   4: gu_s [E, 2*I, K_gu_groups] uint8 (NVFP4) or BF16/F32 (Q8)
  //   5: dn_w [E, D, K_dn_packed] uint32
  //   6: dn_s [E, D, K_dn_groups] uint8 (NVFP4) or BF16/F32 (Q8)
  //   7 (optional, if has_gu_aux_): gu_outer [E] F32 (NVFP4) or gu_b (Q8)
  //   7/8 (optional, if has_dn_aux_): dn_b (Q8)
  const array& x = a[0];
  const array& indices = a[1];
  const array& gamma = a[2];
  const array& gu_w = a[3];
  const array& gu_s = a[4];
  const array& dn_w = a[5];
  const array& dn_s = a[6];
  size_t next_idx = 7;
  const array* gu_aux = has_gu_aux_ ? &a[next_idx++] : nullptr;
  const array* dn_aux = has_dn_aux_ ? &a[next_idx++] : nullptr;

  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto set_kernel = [&](const std::string& name) {
    enc.set_compute_pipeline_state(
        fast::paged::get_prebuilt_kernel(d, "inkling2_moe", "grouped", name));
  };
  auto make_temp = [&](Shape shape, Dtype dtype) {
    array tmp(std::move(shape), dtype, nullptr, {});
    tmp.set_data(allocator::malloc(tmp.nbytes()));
    enc.add_temporary(tmp);
    return tmp;
  };

  const float dummy_f32 = 0.0f;

  // -------------------------------------------------------------------------
  // Fast zero-gather M=1 fused path (2 dispatches total, no NAX needed!).
  // -------------------------------------------------------------------------
  if (p.rows == 1u) {
    auto act = make_temp({static_cast<int>(p.top_k), static_cast<int>(p.intermediate)}, bfloat16);

    set_kernel(is_q8 ? "inkling2_moe_m1_gate_up_situ_q8" : "inkling2_moe_m1_gate_up_situ_nvfp4");
    enc.set_input_array(x, 0);
    enc.set_input_array(indices, 1);
    enc.set_input_array(gu_w, 2);
    enc.set_input_array(gu_s, 3);
    if (gu_aux) {
      enc.set_input_array(*gu_aux, 4);
    } else {
      enc.set_bytes(dummy_f32, 4);
    }
    enc.set_output_array(act, 5);
    enc.set_bytes(p, 6);
    const uint32_t gu_cols_per_tg = is_q8 ? 8u : 4u;
    enc.dispatch_threadgroups(
        MTL::Size((p.intermediate + gu_cols_per_tg - 1u) / gu_cols_per_tg, p.top_k, 1),
        MTL::Size(64, 1, 1));

    set_kernel(is_q8 ? "inkling2_moe_m1_down_combine_q8" : "inkling2_moe_m1_down_combine_nvfp4");
    enc.set_input_array(act, 0);
    enc.set_input_array(indices, 1);
    enc.set_input_array(gamma, 2);
    enc.set_input_array(dn_w, 3);
    enc.set_input_array(dn_s, 4);
    if (is_q8) {
      if (dn_aux) {
        enc.set_input_array(*dn_aux, 5);
      } else {
        enc.set_bytes(dummy_f32, 5);
      }
      enc.set_output_array(out, 6);
      enc.set_bytes(p, 7);
    } else {
      enc.set_output_array(out, 5);
      enc.set_bytes(p, 6);
    }
    const uint32_t dn_cols_per_tg = is_q8 ? 8u : 4u;
    enc.dispatch_threadgroups(
        MTL::Size((p.hidden + dn_cols_per_tg - 1u) / dn_cols_per_tg, 1, 1),
        MTL::Size(64, 1, 1));
    return;
  }

  // -------------------------------------------------------------------------
  // Grouped Indirect-Dispatch Path (T >= 2: M=4, 8, 32, Prefill).
  // -------------------------------------------------------------------------
  p.tile_rows = (p.rows <= 32u) ? 8u : 32u;
  if (const char* cutoff_env = std::getenv("MLX_INKLING2_MOE_COL_CUTOFF")) {
    const int max_c = (p.tile_rows == 8u) ? 32 : 8;
    const int v = std::atoi(cutoff_env);
    p.col_cutoff = static_cast<uint32_t>(std::clamp(v, 0, max_c));
  } else if (p.rows <= 32u) {
    // M=2..32 (4-lane AR, solo DSpark M=8, 4-lane DSpark M=16/24/32, and <=32-tok prefill chunks):
    // 100% 0-smem 1-pass coalesced SIMD multi-row QMV over `tile_rows=8` tiles! Every tile has
    // 1..=8 rows with the exact same FP32 reduction order as M=1 -> bit-exact greedy parity across
    // M=1, M=4, M=8, and M=32 (both NVFP4 & Q8_0), zero NAX dispatch overhead (-96 dispatches/step),
    // and full ~490 GB/s DRAM bandwidth on M5 Max!
    p.col_cutoff = 32u;
  } else {
    // M=33..512 (chunked prefill, e.g. 512-token rolling checkpoints): 1..4 rows/expert use 0-smem
    // SIMD, >=5 rows/expert use 2-SG BN=32/64 NAX tensor ops!
    p.col_cutoff = 4u;
  }

  const uint32_t total_routes = p.rows * p.top_k;
  const uint32_t max_tiles = std::min(
      total_routes,
      (total_routes + p.tile_rows - 1u) / p.tile_rows + p.experts - 1u);
  p.max_tiles = max_tiles;
  const int grouped_rows = static_cast<int>(max_tiles * p.tile_rows);

  // Use persistent stream workspace when total scratch <= 8 MiB (e.g. M=4 & M=8).
  const std::vector<array>* workspace = nullptr;
  const size_t est_bytes =
      size_t(max_tiles) * 8 +
      sizeof(Inkling2MoeTileCount) +
      size_t(grouped_rows) * 4 +
      size_t(total_routes) * 4 +
      size_t(max_tiles) * 8 +
      size_t(grouped_rows) * (2 * size_t(p.hidden) + size_t(p.intermediate)) * 2;
  if (est_bytes <= (size_t(8) << 20)) {
    const uint64_t identity =
        0x494E4B4C4D4F4501ULL ^
        (uint64_t(p.rows) << 32) ^
        (uint64_t(p.hidden) << 16) ^
        uint64_t(p.intermediate);
    const std::vector<splash_moe::persistent_workspace::Spec> specs = {
        {{static_cast<int>(max_tiles), 2}, uint32},
        {{static_cast<int>(sizeof(Inkling2MoeTileCount) / sizeof(uint32_t))}, uint32},
        {{grouped_rows}, uint32},
        {{static_cast<int>(total_routes)}, uint32},
        {{static_cast<int>(2u * max_tiles)}, uint32},
        {{grouped_rows, static_cast<int>(p.hidden)}, bfloat16},
        {{grouped_rows, static_cast<int>(p.intermediate)}, bfloat16},
        {{grouped_rows, static_cast<int>(p.hidden)}, bfloat16},
    };
    workspace = splash_moe::persistent_workspace::try_workspace(s, enc, identity, specs);
  }
  auto scratch = [&](size_t slot, Shape shape, Dtype dtype) {
    return workspace ? (*workspace)[slot] : make_temp(std::move(shape), dtype);
  };

  auto tiles = scratch(0, {static_cast<int>(max_tiles), 2}, uint32);
  auto count = scratch(1, {static_cast<int>(sizeof(Inkling2MoeTileCount) / sizeof(uint32_t))}, uint32);
  auto routes = scratch(2, {grouped_rows}, uint32);
  auto inverse = scratch(3, {static_cast<int>(total_routes)}, uint32);
  auto tile_lists = scratch(4, {static_cast<int>(2u * max_tiles)}, uint32);
  auto gathered = scratch(5, {grouped_rows, static_cast<int>(p.hidden)}, bfloat16);
  auto hidden = scratch(6, {grouped_rows, static_cast<int>(p.intermediate)}, bfloat16);
  auto projected = scratch(7, {grouped_rows, static_cast<int>(p.hidden)}, bfloat16);

  // 1. Group: GPU histogram + prefix sum + tile table + dual SIMD/NAX tile lists + indirect grids
  set_kernel("inkling2_moe_group");
  enc.set_input_array(indices, 0);
  enc.set_output_array(tiles, 1);
  enc.set_output_array(count, 2);
  enc.set_output_array(routes, 3);
  enc.set_output_array(inverse, 4);
  enc.set_output_array(tile_lists, 5);
  enc.set_bytes(p, 6);
  enc.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));

  // 2. Gather (indirect dispatch!)
  set_kernel("inkling2_moe_gather");
  enc.set_input_array(x, 0);
  enc.set_input_array(routes, 1);
  enc.set_input_array(tiles, 2);
  enc.set_input_array(count, 3);
  enc.set_output_array(gathered, 4);
  enc.set_bytes(p, 5);
  enc.dispatch_threadgroups_indirect(
      static_cast<const MTL::Buffer*>(count.buffer().ptr()),
      count.offset() + offsetof(Inkling2MoeTileCount, gather_grid),
      MTL::Size(256, 1, 1));

  const bool need_simd = p.col_cutoff > 0u;
  // `inkling2_moe_group` routes an expert to NAX whenever its total `count > p.col_cutoff`.
  // Since each token selects distinct experts, `count <= p.rows`, so NAX can be non-empty iff
  // `p.rows > p.col_cutoff` (even when `p.col_cutoff == p.tile_rows`!).
  const bool need_nax = p.rows > p.col_cutoff;
  const std::string suffix =
      std::string(is_q8 ? "q8_" : "nvfp4_") + std::to_string(p.tile_rows);

  // 3a. Zero-smem 1-pass coalesced SIMD gate_up + SiTU (indirect dispatch for rows <= col_cutoff)
  if (need_simd) {
    set_kernel(is_q8 ? "inkling2_moe_simd_gate_up_situ_q8" : "inkling2_moe_simd_gate_up_situ_nvfp4");
    enc.set_input_array(gathered, 0);
    enc.set_input_array(tiles, 1);
    enc.set_input_array(tile_lists, 2);
    enc.set_input_array(count, 3);
    enc.set_input_array(gu_w, 4);
    enc.set_input_array(gu_s, 5);
    if (gu_aux) {
      enc.set_input_array(*gu_aux, 6);
    } else {
      enc.set_bytes(dummy_f32, 6);
    }
    enc.set_output_array(hidden, 7);
    enc.set_bytes(p, 8);
    enc.dispatch_threadgroups_indirect(
        static_cast<const MTL::Buffer*>(count.buffer().ptr()),
        count.offset() + offsetof(Inkling2MoeTileCount, simd_gu_grid),
        MTL::Size(64, 1, 1));
  }

  // 3b. 2-SG BN=32 NAX gate_up + SiTU (8 KB smem, indirect dispatch for rows > col_cutoff)
  if (need_nax) {
    set_kernel("inkling2_moe_gate_up_situ_nax_" + suffix);
    enc.set_input_array(gathered, 0);
    enc.set_input_array(tiles, 1);
    enc.set_input_array(tile_lists, 2);
    enc.set_input_array(count, 3);
    enc.set_input_array(gu_w, 4);
    enc.set_input_array(gu_s, 5);
    if (gu_aux) {
      enc.set_input_array(*gu_aux, 6);
    } else {
      enc.set_bytes(dummy_f32, 6);
    }
    enc.set_output_array(hidden, 7);
    enc.set_bytes(p, 8);
    enc.dispatch_threadgroups_indirect(
        static_cast<const MTL::Buffer*>(count.buffer().ptr()),
        count.offset() + offsetof(Inkling2MoeTileCount, nax_gu_grid),
        MTL::Size(64, 1, 1));
  }

  // 4a. Zero-smem 1-pass coalesced SIMD down_proj (indirect dispatch for rows <= col_cutoff)
  if (need_simd) {
    set_kernel(is_q8 ? "inkling2_moe_simd_down_q8" : "inkling2_moe_simd_down_nvfp4");
    enc.set_input_array(hidden, 0);
    enc.set_input_array(tiles, 1);
    enc.set_input_array(tile_lists, 2);
    enc.set_input_array(count, 3);
    enc.set_input_array(dn_w, 4);
    enc.set_input_array(dn_s, 5);
    if (is_q8) {
      if (dn_aux) {
        enc.set_input_array(*dn_aux, 6);
      } else {
        enc.set_bytes(dummy_f32, 6);
      }
      enc.set_output_array(projected, 7);
      enc.set_bytes(p, 8);
    } else {
      enc.set_output_array(projected, 6);
      enc.set_bytes(p, 7);
    }
    enc.dispatch_threadgroups_indirect(
        static_cast<const MTL::Buffer*>(count.buffer().ptr()),
        count.offset() + offsetof(Inkling2MoeTileCount, simd_dn_grid),
        MTL::Size(64, 1, 1));
  }

  // 4b. 2-SG BN=64 NAX down_proj (8 KB smem, indirect dispatch for rows > col_cutoff)
  if (need_nax) {
    set_kernel("inkling2_moe_down_nax_" + suffix);
    enc.set_input_array(hidden, 0);
    enc.set_input_array(tiles, 1);
    enc.set_input_array(tile_lists, 2);
    enc.set_input_array(count, 3);
    enc.set_input_array(dn_w, 4);
    enc.set_input_array(dn_s, 5);
    if (dn_aux) {
      enc.set_input_array(*dn_aux, 6);
    } else {
      enc.set_bytes(dummy_f32, 6);
    }
    enc.set_output_array(projected, 7);
    enc.set_bytes(p, 8);
    enc.dispatch_threadgroups_indirect(
        static_cast<const MTL::Buffer*>(count.buffer().ptr()),
        count.offset() + offsetof(Inkling2MoeTileCount, nax_dn_grid),
        MTL::Size(64, 1, 1));
  }

  // 5. Weighted combine -> out [T, D] BF16
  set_kernel("inkling2_moe_grouped_combine_bf16");
  enc.set_input_array(projected, 0);
  enc.set_input_array(inverse, 1);
  enc.set_input_array(gamma, 2);
  enc.set_output_array(out, 3);
  enc.set_bytes(p, 4);
  enc.dispatch_threadgroups(
      MTL::Size(p.rows, (p.hidden + 255u) / 256u, 1),
      MTL::Size(256, 1, 1));
}
#else
void Inkling2GroupedMoe::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("Inkling2GroupedMoe requires Metal");
}
#endif

} // namespace

bool available_for_rows(uint32_t rows) {
#ifdef MLX_NODE_METAL_ENABLED
  if (detail::in_grad_tracing() || !(default_device() == Device::gpu) ||
      !is_available(Device::gpu) || !metal::is_available()) {
    return false;
  }
  const char* env = std::getenv("MLX_INKLING2_MOE_MODE");
  if (env && (std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0)) {
    return false;
  }
  if (rows == 1u) {
    if (env && std::strcmp(env, "grouped_only") == 0) {
      return false;
    }
    return true;
  }
  // Up to M=512 (covers M=1..32 decode/verify AND 512-token rolling-checkpoint prefill chunks
  // so cold-tier RAM/SSD prefix hits match fresh prefill bit-exact!). Above M=512 (long 2K/8K
  // full-chunk prefill), MLX's native 64x64x64 `gather_qmm_t_nax` avoids grouped gather/projected
  // copies (+15-25% faster on 2K/8K prefill!).
  uint32_t max_rows = 512u;
  if (const char* max_env = std::getenv("MLX_INKLING2_MOE_MAX_ROWS")) {
    const int v = std::atoi(max_env);
    if (v > 0) {
      max_rows = static_cast<uint32_t>(v);
    }
  }
  if (rows > max_rows) {
    return false;
  }
  return metal::is_nax_available();
#else
  (void)rows;
  return false;
#endif
}

} // namespace mlx::core::inkling2_moe

extern "C" {

bool mlx_inkling2_grouped_moe_available(uint32_t rows) {
  return mlx::core::inkling2_moe::available_for_rows(rows);
}

mlx_array* mlx_inkling2_grouped_moe(
    mlx_array* x_ptr,
    mlx_array* top_k_indices_ptr,
    mlx_array* gamma_ptr,
    mlx_array* gu_w_ptr,
    mlx_array* gu_s_ptr,
    mlx_array* gu_aux_ptr,
    mlx_array* dn_w_ptr,
    mlx_array* dn_s_ptr,
    mlx_array* dn_aux_ptr,
    bool is_q8,
    float gate_softcap,
    float up_softcap,
    bool round_bf16_before_situ) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !top_k_indices_ptr || !gamma_ptr ||
      !gu_w_ptr || !gu_s_ptr || !dn_w_ptr || !dn_s_ptr) {
    return nullptr;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_moe;

    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& idx = *reinterpret_cast<array*>(top_k_indices_ptr);
    const auto& gamma = *reinterpret_cast<array*>(gamma_ptr);
    const auto& gu_w = *reinterpret_cast<array*>(gu_w_ptr);
    const auto& gu_s = *reinterpret_cast<array*>(gu_s_ptr);
    const auto& dn_w = *reinterpret_cast<array*>(dn_w_ptr);
    const auto& dn_s = *reinterpret_cast<array*>(dn_s_ptr);

    if (x.ndim() != 2 || x.dtype() != bfloat16 ||
        idx.ndim() != 2 || idx.dtype() != int32 ||
        gamma.ndim() != 2 || gamma.dtype() != float32) {
      return nullptr;
    }
    const int T = x.shape(0);
    const int D = x.shape(1);
    const int K = idx.shape(1);
    if (T < 1 || T > 8192 || D < 64 || (D % 64) != 0 ||
        idx.shape(0) != T || gamma.shape(0) != T || gamma.shape(1) != K ||
        K < 1 || K > 16) {
      return nullptr;
    }
    if (!available_for_rows(static_cast<uint32_t>(T))) {
      return nullptr;
    }

    if (gu_w.ndim() != 3 || gu_w.dtype() != uint32 ||
        dn_w.ndim() != 3 || dn_w.dtype() != uint32 ||
        gu_s.ndim() != 3 || dn_s.ndim() != 3) {
      return nullptr;
    }
    const int E = gu_w.shape(0);
    const int N2 = gu_w.shape(1);
    if (E < 1 || E > 256 || (N2 % 2) != 0) {
      return nullptr;
    }
    const int I = N2 / 2;
    if (I < 64 || (I % 64) != 0 || dn_w.shape(0) != E || dn_w.shape(1) != D) {
      return nullptr;
    }

    uint32_t flags = is_q8 ? 1u : 0u;
    if (round_bf16_before_situ) {
      flags |= 8u;
    }

    std::vector<array> inputs = {x, idx, gamma, gu_w, gu_s, dn_w, dn_s};
    bool has_gu_aux = false;
    bool has_dn_aux = false;

    if (!is_q8) {
      // NVFP4: group_size=16, bits=4, uint8 E4M3 scales, optional F32 [E] gu_outer
      if (gu_w.shape(2) != D / 8 || dn_w.shape(2) != I / 8 ||
          gu_s.dtype() != uint8 || dn_s.dtype() != uint8 ||
          gu_s.shape() != Shape{E, N2, D / 16} ||
          dn_s.shape() != Shape{E, D, I / 16}) {
        return nullptr;
      }
      if (gu_aux_ptr) {
        const auto& gu_outer = *reinterpret_cast<array*>(gu_aux_ptr);
        if (gu_outer.ndim() != 1 || gu_outer.shape(0) != E || gu_outer.dtype() != float32) {
          return nullptr;
        }
        flags |= 2u;
        has_gu_aux = true;
        inputs.push_back(gu_outer);
      }
    } else {
      // Q8_0 affine: group_size=32, bits=8, BF16 or F32 scales & biases
      if (!gu_aux_ptr || !dn_aux_ptr) {
        return nullptr;
      }
      const auto& gu_b = *reinterpret_cast<array*>(gu_aux_ptr);
      const auto& dn_b = *reinterpret_cast<array*>(dn_aux_ptr);
      const Dtype sc_dt = gu_s.dtype();
      if ((sc_dt != bfloat16 && sc_dt != float32) ||
          dn_s.dtype() != sc_dt || gu_b.dtype() != sc_dt || dn_b.dtype() != sc_dt) {
        return nullptr;
      }
      if (gu_w.shape(2) != D / 4 || dn_w.shape(2) != I / 4 ||
          gu_s.shape() != Shape{E, N2, D / 32} ||
          gu_b.shape() != Shape{E, N2, D / 32} ||
          dn_s.shape() != Shape{E, D, I / 32} ||
          dn_b.shape() != Shape{E, D, I / 32}) {
        return nullptr;
      }
      if (sc_dt == float32) {
        flags |= 4u;
      }
      has_gu_aux = true;
      has_dn_aux = true;
      inputs.push_back(gu_b);
      inputs.push_back(dn_b);
    }

    Inkling2MoeParams p{};
    p.rows = static_cast<uint32_t>(T);
    p.hidden = static_cast<uint32_t>(D);
    p.intermediate = static_cast<uint32_t>(I);
    p.experts = static_cast<uint32_t>(E);
    p.top_k = static_cast<uint32_t>(K);
    p.tile_rows = (T <= 32) ? 8u : 32u;
    p.gate_softcap = gate_softcap;
    p.up_softcap = up_softcap;
    p.flags = flags;
    p.col_cutoff = (p.tile_rows == 8u) ? 4u : 2u;

    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2GroupedMoe>(s, p, has_gu_aux, has_dn_aux);
    auto res = array(x.shape(), bfloat16, std::move(prim), std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_grouped_moe", e.what());
    return nullptr;
  }
#else
  (void)x_ptr; (void)top_k_indices_ptr; (void)gamma_ptr;
  (void)gu_w_ptr; (void)gu_s_ptr; (void)gu_aux_ptr;
  (void)dn_w_ptr; (void)dn_s_ptr; (void)dn_aux_ptr;
  (void)is_q8; (void)gate_softcap; (void)up_softcap; (void)round_bf16_before_situ;
  return nullptr;
#endif
}

} // extern "C"
