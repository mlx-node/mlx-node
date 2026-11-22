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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlx::core::inkling2_moe {

uint32_t resolved_moe_mode() {
  static const uint32_t kMode = []() -> uint32_t {
    const char* env = std::getenv("MLX_INKLING2_MOE_MODE");
    if (!env) return 0u;
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0) return 1u;
    if (std::strcmp(env, "grouped_only") == 0) return 2u;
    return 0u;
  }();
  return kMode;
}

uint32_t resolved_moe_max_rows() {
  static const uint32_t kMaxRows = []() -> uint32_t {
    if (const char* max_env = std::getenv("MLX_INKLING2_MOE_MAX_ROWS")) {
      const int v = std::atoi(max_env);
      if (v > 0) {
        return static_cast<uint32_t>(v);
      }
    }
    return 32u;
  }();
  return kMaxRows;
}

// -1 means unset (use per-path defaults); >= 0 means explicit env override (clamped 0..=32).
int resolved_moe_col_cutoff_override() {
  static const int kCutoff = []() -> int {
    if (const char* cutoff_env = std::getenv("MLX_INKLING2_MOE_COL_CUTOFF")) {
      return std::clamp(std::atoi(cutoff_env), 0, 32);
    }
    return -1;
  }();
  return kCutoff;
}

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
  const bool is_mxfp8 = (p.flags & 64u) != 0u;
  const bool is_8bit = is_q8 || is_mxfp8;

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

    set_kernel(
        is_mxfp8 ? "inkling2_moe_m1_gate_up_situ_mxfp8"
                 : is_q8 ? "inkling2_moe_m1_gate_up_situ_q8"
                         : "inkling2_moe_m1_gate_up_situ_nvfp4");
    enc.set_input_array(x, 0);
    enc.set_input_array(indices, 1);
    enc.set_input_array(gu_w, 2);
    enc.set_input_array(gu_s, 3);
    if (is_mxfp8) {
      enc.set_output_array(act, 4);
      enc.set_bytes(p, 5);
    } else {
      if (gu_aux) {
        enc.set_input_array(*gu_aux, 4);
      } else {
        enc.set_bytes(dummy_f32, 4);
      }
      enc.set_output_array(act, 5);
      enc.set_bytes(p, 6);
    }
    const uint32_t gu_cols_per_tg = is_8bit ? 8u : 4u;
    enc.dispatch_threadgroups(
        MTL::Size((p.intermediate + gu_cols_per_tg - 1u) / gu_cols_per_tg, p.top_k, 1),
        MTL::Size(64, 1, 1));

    set_kernel(
        is_mxfp8 ? "inkling2_moe_m1_down_combine_mxfp8"
                 : is_q8 ? "inkling2_moe_m1_down_combine_q8"
                         : "inkling2_moe_m1_down_combine_nvfp4");
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
    const uint32_t dn_cols_per_tg = is_8bit ? 8u : 4u;
    enc.dispatch_threadgroups(
        MTL::Size((p.hidden + dn_cols_per_tg - 1u) / dn_cols_per_tg, 1, 1),
        MTL::Size(64, 1, 1));
    return;
  }

  // -------------------------------------------------------------------------
  // Grouped Indirect-Dispatch Path (T >= 2: M=4, 8, 32, Prefill).
  // -------------------------------------------------------------------------
  p.tile_rows = (p.rows <= 32u) ? 8u : 32u;
  if (is_mxfp8) {
    // MXFP8 uses 100% 0-smem coalesced SIMD multi-row QMV for T <= 32.
    p.col_cutoff = 32u;
  } else if (const int cutoff_override = resolved_moe_col_cutoff_override(); cutoff_override >= 0) {
    const int max_c = (p.tile_rows == 8u) ? 32 : 8;
    p.col_cutoff = static_cast<uint32_t>(std::clamp(cutoff_override, 0, max_c));
  } else if (p.rows <= 32u) {
    // M=2..32 (4-lane AR, solo DSpark M=8, 4-lane DSpark M=16/24/32, and <=32-tok prefill chunks):
    // 100% 0-smem 1-pass coalesced SIMD multi-row QMV over `tile_rows=8` tiles! Every tile has
    // 1..=8 rows with the exact same FP32 reduction order as M=1 -> bit-exact greedy parity across
    // M=1, M=4, M=8, and M=32 (NVFP4, Q8_0, & MXFP8), zero NAX dispatch overhead (-96 dispatches/step),
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

  const bool need_simd = p.col_cutoff > 0u;
  // `inkling2_moe_group` routes an expert to NAX whenever its total `count > p.col_cutoff`.
  // Since each token selects distinct experts, `count <= p.rows`, so NAX can be non-empty iff
  // `p.rows > p.col_cutoff` (even when `p.col_cutoff == p.tile_rows`!).
  const bool need_nax = p.rows > p.col_cutoff;

  // Persistent stream workspace up to 16 MiB. For `T <= 32`, size for the `T = 32` high-water
  // mark so M=1, M=4, M=8, M=16, M=24, and M=32 share ONE persistent workspace with zero allocator churn!
  constexpr size_t kWorkspaceCapBytes = size_t(16) << 20;
  const uint32_t ws_rows = (p.rows <= 32u) ? 32u : p.rows;
  const uint32_t ws_tile_rows = (ws_rows <= 32u) ? 8u : p.tile_rows;
  const uint32_t ws_total_routes = ws_rows * p.top_k;
  const uint32_t ws_max_tiles = std::min(
      ws_total_routes,
      (ws_total_routes + ws_tile_rows - 1u) / ws_tile_rows + p.experts - 1u);
  const int ws_grouped_rows = static_cast<int>(ws_max_tiles * ws_tile_rows);

  const std::vector<array>* workspace = nullptr;
  const size_t est_bytes =
      size_t(ws_max_tiles) * 8 +
      sizeof(Inkling2MoeTileCount) +
      size_t(ws_grouped_rows) * 4 +
      size_t(ws_total_routes) * 4 +
      size_t(ws_max_tiles) * 8 +
      size_t(ws_grouped_rows) * (2 * size_t(p.hidden) + size_t(p.intermediate)) * 2 +
      size_t(ws_total_routes) * 4;
  if (est_bytes <= kWorkspaceCapBytes) {
    const uint64_t identity =
        0x494E4B4C4D4F4502ULL ^
        (uint64_t(ws_max_tiles) << 40) ^
        (uint64_t(p.hidden) << 20) ^
        (uint64_t(p.intermediate) << 4) ^
        (uint64_t(ws_total_routes) << 1) ^
        1ULL;
    std::vector<splash_moe::persistent_workspace::Spec> specs = {
        {{static_cast<int>(ws_max_tiles), 2}, uint32},
        {{static_cast<int>(sizeof(Inkling2MoeTileCount) / sizeof(uint32_t))}, uint32},
        {{ws_grouped_rows}, uint32},
        {{static_cast<int>(ws_total_routes)}, uint32},
        {{static_cast<int>(2u * ws_max_tiles)}, uint32},
        {{ws_grouped_rows, static_cast<int>(p.intermediate)}, bfloat16},
        {{ws_grouped_rows, static_cast<int>(p.hidden)}, bfloat16},
        {{static_cast<int>(ws_total_routes)}, float32},
        {{ws_grouped_rows, static_cast<int>(p.hidden)}, bfloat16},
    };
    workspace = splash_moe::persistent_workspace::try_workspace_capped(
        s, enc, identity, specs, kWorkspaceCapBytes);
  }
  auto scratch = [&](size_t slot, Shape shape, Dtype dtype) {
    return workspace ? (*workspace)[slot] : make_temp(std::move(shape), dtype);
  };

  auto tiles = scratch(0, {static_cast<int>(max_tiles), 2}, uint32);
  auto count = scratch(1, {static_cast<int>(sizeof(Inkling2MoeTileCount) / sizeof(uint32_t))}, uint32);
  auto routes = scratch(2, {grouped_rows}, uint32);
  auto inverse = scratch(3, {static_cast<int>(total_routes)}, uint32);
  auto tile_lists = scratch(4, {static_cast<int>(2u * max_tiles)}, uint32);
  auto hidden = scratch(5, {grouped_rows, static_cast<int>(p.intermediate)}, bfloat16);
  auto projected = scratch(6, {grouped_rows, static_cast<int>(p.hidden)}, bfloat16);
  auto gathered = scratch(8, {grouped_rows, static_cast<int>(p.hidden)}, bfloat16);

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

  // 2. Gather (indirect dispatch: 0.05 ms/48 visits, keeps SIMD gate_up 0-smem & 0-barrier!)
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

  const std::string suffix =
      std::string(is_q8 ? "q8_" : "nvfp4_") + std::to_string(p.tile_rows);

  // 3a. Zero-smem 1-pass coalesced SIMD gate_up + SiTU (indirect dispatch for rows <= col_cutoff)
  if (need_simd) {
    set_kernel(
        is_mxfp8 ? "inkling2_moe_simd_gate_up_situ_mxfp8"
                 : is_q8 ? "inkling2_moe_simd_gate_up_situ_q8"
                         : "inkling2_moe_simd_gate_up_situ_nvfp4");
    enc.set_input_array(gathered, 0);
    enc.set_input_array(tiles, 1);
    enc.set_input_array(tile_lists, 2);
    enc.set_input_array(count, 3);
    enc.set_input_array(gu_w, 4);
    enc.set_input_array(gu_s, 5);
    if (is_mxfp8) {
      enc.set_output_array(hidden, 6);
      enc.set_bytes(p, 7);
    } else {
      if (gu_aux) {
        enc.set_input_array(*gu_aux, 6);
      } else {
        enc.set_bytes(dummy_f32, 6);
      }
      enc.set_output_array(hidden, 7);
      enc.set_bytes(p, 8);
    }
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
    set_kernel(
        is_mxfp8 ? "inkling2_moe_simd_down_mxfp8"
                 : is_q8 ? "inkling2_moe_simd_down_q8"
                         : "inkling2_moe_simd_down_nvfp4");
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

// ============================================================================
// End-to-End Fused T <= 32 MoE + Route/Group + Direct Gather + PostConv + Residual
// ============================================================================

struct Inkling2MoeRouteParamsLocal {
  uint32_t rows;
  uint32_t num_experts;
  uint32_t top_k;
  float route_scale;
  uint32_t flags;
  uint32_t logits_stride;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(Inkling2MoeRouteParamsLocal) == 32, "Inkling2MoeRouteParamsLocal ABI");

class Inkling2GroupedMoeFused : public fast::Custom {
 public:
  Inkling2GroupedMoeFused(
      Stream s,
      Inkling2MoeParams params,
      float route_scale,
      uint32_t num_seqs,
      std::vector<int32_t> cu_seqlens,
      bool packed_hist,
      bool has_bias,
      bool has_down_scale,
      bool has_gu_aux,
      bool has_dn_aux,
      bool has_res,
      bool emit_tape)
      : fast::Custom(
            s,
            [](std::vector<array>) -> std::vector<array> {
              throw std::runtime_error("Inkling2GroupedMoeFused requires Metal");
            }),
        p_(params),
        route_scale_(route_scale),
        num_seqs_(num_seqs),
        cu_seqlens_(std::move(cu_seqlens)),
        packed_hist_(packed_hist),
        has_bias_(has_bias),
        has_down_scale_(has_down_scale),
        has_gu_aux_(has_gu_aux),
        has_dn_aux_(has_dn_aux),
        has_res_(has_res),
        emit_tape_(emit_tape) {}

  DEFINE_NAME(Inkling2GroupedMoeFused)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Inkling2GroupedMoeFused&>(other);
    return route_scale_ == rhs.route_scale_ &&
        num_seqs_ == rhs.num_seqs_ &&
        cu_seqlens_ == rhs.cu_seqlens_ &&
        packed_hist_ == rhs.packed_hist_ &&
        has_bias_ == rhs.has_bias_ &&
        has_down_scale_ == rhs.has_down_scale_ &&
        has_gu_aux_ == rhs.has_gu_aux_ &&
        has_dn_aux_ == rhs.has_dn_aux_ &&
        has_res_ == rhs.has_res_ &&
        emit_tape_ == rhs.emit_tape_ &&
        std::memcmp(&p_, &rhs.p_, sizeof(p_)) == 0;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int d = inputs[0].shape(1);
    Shape hs = (num_seqs_ == 1u && !packed_hist_)
        ? Shape{3, d}
        : Shape{static_cast<int>(num_seqs_), 3, d};
    if (emit_tape_) {
      return {inputs[0].shape(), hs, inputs[0].shape()};
    }
    return {inputs[0].shape(), hs};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2GroupedMoeFused requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;

 private:
  Inkling2MoeParams p_;
  float route_scale_;
  uint32_t num_seqs_;
  std::vector<int32_t> cu_seqlens_;
  bool packed_hist_;
  bool has_bias_;
  bool has_down_scale_;
  bool has_gu_aux_;
  bool has_dn_aux_;
  bool has_res_;
  bool emit_tape_;
};

#ifdef MLX_NODE_METAL_ENABLED
void Inkling2GroupedMoeFused::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& enc = metal::get_command_encoder(s);

  std::vector<array> a;
  a.reserve(inputs.size());
  for (const auto& inp : inputs) {
    a.push_back(ensure_aligned_contiguous(inp, s, enc));
  }

  Inkling2MoeParams p = p_;
  p.rows = static_cast<uint32_t>(a[0].shape(0));
  const bool is_q8 = (p.flags & 1u) != 0u;
  const bool is_mxfp8 = (p.flags & 64u) != 0u;
  const bool is_8bit = is_q8 || is_mxfp8;

  const array& x = a[0];
  const array& router_logits = a[1];
  const array& gu_w = a[2];
  const array& gu_s = a[3];
  const array& dn_w = a[4];
  const array& dn_s = a[5];
  const array& post_w = a[6];
  size_t next_idx = 7;
  const array* bias = has_bias_ ? &a[next_idx++] : nullptr;
  const array* down_scale = has_down_scale_ ? &a[next_idx++] : nullptr;
  const array* gu_aux = has_gu_aux_ ? &a[next_idx++] : nullptr;
  const array* dn_aux = has_dn_aux_ ? &a[next_idx++] : nullptr;
  const array* res = has_res_ ? &a[next_idx++] : nullptr;
  const uint32_t h_count = packed_hist_ ? 1u : num_seqs_;
  const array* hists[4] = {nullptr, nullptr, nullptr, nullptr};
  uint32_t hist_f32_mask = 0u;
  for (uint32_t i = 0; i < h_count; ++i) {
    hists[i] = &a[next_idx++];
    if (hists[i]->dtype() == float32) {
      hist_f32_mask |= (1u << i);
    }
  }

  auto& out_y = outputs[0];
  auto& out_hist = outputs[1];
  out_y.set_data(allocator::malloc(out_y.nbytes()));
  out_hist.set_data(allocator::malloc(out_hist.nbytes()));
  array* out_tape = nullptr;
  if (emit_tape_) {
    out_tape = &outputs[2];
    out_tape->set_data(allocator::malloc(out_tape->nbytes()));
  }

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

  Inkling2MoePostConvParams pc{};
  pc.rows = p.rows;
  pc.hidden = p.hidden;
  pc.top_k = p.top_k;
  pc.num_seqs = num_seqs_;
  pc.packed_hist = packed_hist_ ? 1u : 0u;
  pc.res_stride = p.hidden;
  pc.flags = (post_w.dtype() == float32 ? 1u : 0u) |
      (has_res_ ? 2u : 0u) |
      ((has_res_ && res->dtype() == float32) ? 4u : 0u) |
      (emit_tape_ ? 8u : 0u);
  pc.hist_f32_mask = hist_f32_mask;

  const float dummy_f32 = 0.0f;

  // Shared 16 MiB persistent stream workspace sized for T=32 high-water mark so M=1..32
  // reuse the exact same GPU scratch buffers with zero per-visit `allocator::malloc` churn.
  constexpr size_t kWorkspaceCapBytes = size_t(16) << 20;
  const uint32_t ws_rows = 32u;
  const uint32_t ws_tile_rows = 8u;
  const uint32_t ws_total_routes = ws_rows * p.top_k;
  const uint32_t ws_max_tiles = std::min(
      ws_total_routes,
      (ws_total_routes + ws_tile_rows - 1u) / ws_tile_rows + p.experts - 1u);
  const int ws_grouped_rows = static_cast<int>(ws_max_tiles * ws_tile_rows);

  const std::vector<array>* workspace = nullptr;
  const size_t est_bytes =
      size_t(ws_max_tiles) * 8 +
      sizeof(Inkling2MoeTileCount) +
      size_t(ws_grouped_rows) * 4 +
      size_t(ws_total_routes) * 4 +
      size_t(ws_max_tiles) * 8 +
      size_t(ws_grouped_rows) * (2 * size_t(p.hidden) + size_t(p.intermediate)) * 2 +
      size_t(ws_total_routes) * 4;
  if (est_bytes <= kWorkspaceCapBytes) {
    const uint64_t identity =
        0x494E4B4C4D4F4502ULL ^
        (uint64_t(ws_max_tiles) << 40) ^
        (uint64_t(p.hidden) << 20) ^
        (uint64_t(p.intermediate) << 4) ^
        (uint64_t(ws_total_routes) << 1) ^
        1ULL;
    std::vector<splash_moe::persistent_workspace::Spec> specs = {
        {{static_cast<int>(ws_max_tiles), 2}, uint32},
        {{static_cast<int>(sizeof(Inkling2MoeTileCount) / sizeof(uint32_t))}, uint32},
        {{ws_grouped_rows}, uint32},
        {{static_cast<int>(ws_total_routes)}, uint32},
        {{static_cast<int>(2u * ws_max_tiles)}, uint32},
        {{ws_grouped_rows, static_cast<int>(p.intermediate)}, bfloat16},
        {{ws_grouped_rows, static_cast<int>(p.hidden)}, bfloat16},
        {{static_cast<int>(ws_total_routes)}, float32},
        {{ws_grouped_rows, static_cast<int>(p.hidden)}, bfloat16},
    };
    workspace = splash_moe::persistent_workspace::try_workspace_capped(
        s, enc, identity, specs, kWorkspaceCapBytes);
  }
  auto scratch = [&](size_t slot, Shape shape, Dtype dtype) {
    return workspace ? (*workspace)[slot] : make_temp(std::move(shape), dtype);
  };

  // -------------------------------------------------------------------------
  // M=1 Fused Path: route -> m1_gate_up_situ -> m1_down_combine_sconv (3 dispatches!)
  // -------------------------------------------------------------------------
  if (p.rows == 1u) {
    auto indices = scratch(2, {1, static_cast<int>(p.top_k)}, int32);
    auto gamma = scratch(7, {1, static_cast<int>(p.top_k)}, float32);
    auto act = scratch(5, {static_cast<int>(p.top_k), static_cast<int>(p.intermediate)}, bfloat16);

    uint32_t rflags = (router_logits.dtype() == float32 ? 1u : 0u);
    if (bias) {
      rflags |= (1u << 1u);
      if (bias->dtype() == float32) rflags |= (1u << 2u);
    }
    if (down_scale) {
      rflags |= (1u << 3u);
      if (down_scale->dtype() == float32) rflags |= (1u << 4u);
    }
    Inkling2MoeRouteParamsLocal rp{
        1u, p.experts, p.top_k, route_scale_, rflags, p.experts, 0u, 0u};
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        d, "inkling2_ops", "moe_route", "inkling2_moe_route_f32"));
    enc.set_input_array(router_logits, 0);
    if (bias) enc.set_input_array(*bias, 1); else enc.set_bytes(dummy_f32, 1);
    if (down_scale) enc.set_input_array(*down_scale, 2); else enc.set_bytes(dummy_f32, 2);
    enc.set_output_array(indices, 3);
    enc.set_output_array(gamma, 4);
    enc.set_bytes(rp, 5);
    enc.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));

    set_kernel(
        is_mxfp8 ? "inkling2_moe_m1_gate_up_situ_mxfp8"
                 : is_q8 ? "inkling2_moe_m1_gate_up_situ_q8"
                         : "inkling2_moe_m1_gate_up_situ_nvfp4");
    enc.set_input_array(x, 0);
    enc.set_input_array(indices, 1);
    enc.set_input_array(gu_w, 2);
    enc.set_input_array(gu_s, 3);
    if (is_mxfp8) {
      enc.set_output_array(act, 4);
      enc.set_bytes(p, 5);
    } else {
      if (gu_aux) enc.set_input_array(*gu_aux, 4); else enc.set_bytes(dummy_f32, 4);
      enc.set_output_array(act, 5);
      enc.set_bytes(p, 6);
    }
    const uint32_t gu_cols_per_tg = is_8bit ? 8u : 4u;
    enc.dispatch_threadgroups(
        MTL::Size((p.intermediate + gu_cols_per_tg - 1u) / gu_cols_per_tg, p.top_k, 1),
        MTL::Size(64, 1, 1));

    set_kernel(
        is_mxfp8 ? "inkling2_moe_m1_down_combine_sconv_mxfp8"
                 : is_q8 ? "inkling2_moe_m1_down_combine_sconv_q8"
                         : "inkling2_moe_m1_down_combine_sconv_nvfp4");
    enc.set_input_array(act, 0);
    enc.set_input_array(indices, 1);
    enc.set_input_array(gamma, 2);
    enc.set_input_array(dn_w, 3);
    enc.set_input_array(dn_s, 4);
    uint32_t bidx = 5u;
    if (is_q8) {
      if (dn_aux) enc.set_input_array(*dn_aux, bidx++); else enc.set_bytes(dummy_f32, bidx++);
    }
    enc.set_input_array(post_w, bidx++);
    enc.set_input_array(*hists[0], bidx++);
    if (res) enc.set_input_array(*res, bidx++); else enc.set_bytes(dummy_f32, bidx++);
    enc.set_output_array(out_y, bidx++);
    enc.set_output_array(out_hist, bidx++);
    if (out_tape) enc.set_output_array(*out_tape, bidx++); else enc.set_bytes(dummy_f32, bidx++);
    enc.set_bytes(p, bidx++);
    enc.set_bytes(pc, bidx++);
    const uint32_t dn_cols_per_tg = is_8bit ? 8u : 4u;
    enc.dispatch_threadgroups(
        MTL::Size((p.hidden + dn_cols_per_tg - 1u) / dn_cols_per_tg, 1, 1),
        MTL::Size(64, 1, 1));
    return;
  }

  // -------------------------------------------------------------------------
  // M=2..32 Fused Path: route_group_small -> gather -> gate_up+SiTU -> down -> combine_sconv
  // -------------------------------------------------------------------------
  p.tile_rows = 8u;
  if (is_mxfp8) {
    p.col_cutoff = 32u;
  } else if (const int cutoff_override = resolved_moe_col_cutoff_override(); cutoff_override >= 0) {
    p.col_cutoff = static_cast<uint32_t>(std::clamp(cutoff_override, 0, 32));
  } else {
    p.col_cutoff = 32u;
  }

  const uint32_t total_routes = p.rows * p.top_k;
  const uint32_t max_tiles = std::min(
      total_routes,
      (total_routes + p.tile_rows - 1u) / p.tile_rows + p.experts - 1u);
  p.max_tiles = max_tiles;
  const int grouped_rows = static_cast<int>(max_tiles * p.tile_rows);

  const bool need_simd = p.col_cutoff > 0u;
  const bool need_nax = p.rows > p.col_cutoff;

  auto tiles = scratch(0, {static_cast<int>(max_tiles), 2}, uint32);
  auto count = scratch(1, {static_cast<int>(sizeof(Inkling2MoeTileCount) / sizeof(uint32_t))}, uint32);
  auto routes = scratch(2, {grouped_rows}, uint32);
  auto inverse = scratch(3, {static_cast<int>(total_routes)}, uint32);
  auto tile_lists = scratch(4, {static_cast<int>(2u * max_tiles)}, uint32);
  auto hidden = scratch(5, {grouped_rows, static_cast<int>(p.intermediate)}, bfloat16);
  auto projected = scratch(6, {grouped_rows, static_cast<int>(p.hidden)}, bfloat16);
  auto gamma = scratch(7, {static_cast<int>(total_routes)}, float32);
  auto gathered = scratch(8, {grouped_rows, static_cast<int>(p.hidden)}, bfloat16);

  // 1. Fused Route + Group (1 single 256-thread TG!)
  Inkling2MoeRouteGroupParams rp{};
  rp.rows = p.rows;
  rp.hidden = p.hidden;
  rp.intermediate = p.intermediate;
  rp.experts = p.experts;
  rp.top_k = p.top_k;
  rp.tile_rows = p.tile_rows;
  rp.route_scale = route_scale_;
  rp.flags = (is_8bit ? 1u : 0u) |
      (router_logits.dtype() == float32 ? 2u : 0u) |
      (bias ? 4u : 0u) |
      ((bias && bias->dtype() == float32) ? 8u : 0u) |
      (down_scale ? 16u : 0u) |
      ((down_scale && down_scale->dtype() == float32) ? 32u : 0u);
  rp.col_cutoff = p.col_cutoff;
  rp.max_tiles = p.max_tiles;
  rp.logits_stride = p.experts;
  rp._pad0 = 0u;

  set_kernel("inkling2_moe_route_group_small");
  enc.set_input_array(router_logits, 0);
  if (bias) enc.set_input_array(*bias, 1); else enc.set_bytes(dummy_f32, 1);
  if (down_scale) enc.set_input_array(*down_scale, 2); else enc.set_bytes(dummy_f32, 2);
  enc.set_output_array(gamma, 3);
  enc.set_output_array(tiles, 4);
  enc.set_output_array(count, 5);
  enc.set_output_array(routes, 6);
  enc.set_output_array(inverse, 7);
  enc.set_output_array(tile_lists, 8);
  enc.set_bytes(rp, 9);
  enc.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));

  // 2. Gather (0.05 ms/48 visits; keeps SIMD gate_up 0-smem & 0-barrier!)
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

  const std::string suffix = std::string(is_q8 ? "q8_8" : "nvfp4_8");

  // 3a. Zero-smem 1-pass coalesced SIMD gate_up + SiTU
  if (need_simd) {
    set_kernel(
        is_mxfp8 ? "inkling2_moe_simd_gate_up_situ_mxfp8"
                 : is_q8 ? "inkling2_moe_simd_gate_up_situ_q8"
                         : "inkling2_moe_simd_gate_up_situ_nvfp4");
    enc.set_input_array(gathered, 0);
    enc.set_input_array(tiles, 1);
    enc.set_input_array(tile_lists, 2);
    enc.set_input_array(count, 3);
    enc.set_input_array(gu_w, 4);
    enc.set_input_array(gu_s, 5);
    if (is_mxfp8) {
      enc.set_output_array(hidden, 6);
      enc.set_bytes(p, 7);
    } else {
      if (gu_aux) enc.set_input_array(*gu_aux, 6); else enc.set_bytes(dummy_f32, 6);
      enc.set_output_array(hidden, 7);
      enc.set_bytes(p, 8);
    }
    enc.dispatch_threadgroups_indirect(
        static_cast<const MTL::Buffer*>(count.buffer().ptr()),
        count.offset() + offsetof(Inkling2MoeTileCount, simd_gu_grid),
        MTL::Size(64, 1, 1));
  }

  // 3b. NAX gate_up + SiTU (if col_cutoff < T)
  if (need_nax) {
    set_kernel("inkling2_moe_gate_up_situ_nax_" + suffix);
    enc.set_input_array(gathered, 0);
    enc.set_input_array(tiles, 1);
    enc.set_input_array(tile_lists, 2);
    enc.set_input_array(count, 3);
    enc.set_input_array(gu_w, 4);
    enc.set_input_array(gu_s, 5);
    if (gu_aux) enc.set_input_array(*gu_aux, 6); else enc.set_bytes(dummy_f32, 6);
    enc.set_output_array(hidden, 7);
    enc.set_bytes(p, 8);
    enc.dispatch_threadgroups_indirect(
        static_cast<const MTL::Buffer*>(count.buffer().ptr()),
        count.offset() + offsetof(Inkling2MoeTileCount, nax_gu_grid),
        MTL::Size(64, 1, 1));
  }

  // 4a. SIMD down_proj (full-cacheline 128B uint4 for NVFP4/Q8/MXFP8!)
  if (need_simd) {
    set_kernel(
        is_mxfp8 ? "inkling2_moe_simd_down_mxfp8"
                 : is_q8 ? "inkling2_moe_simd_down_q8"
                         : "inkling2_moe_simd_down_nvfp4");
    enc.set_input_array(hidden, 0);
    enc.set_input_array(tiles, 1);
    enc.set_input_array(tile_lists, 2);
    enc.set_input_array(count, 3);
    enc.set_input_array(dn_w, 4);
    enc.set_input_array(dn_s, 5);
    if (is_q8) {
      if (dn_aux) enc.set_input_array(*dn_aux, 6); else enc.set_bytes(dummy_f32, 6);
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

  // 4b. NAX down_proj (if col_cutoff < T)
  if (need_nax) {
    set_kernel("inkling2_moe_down_nax_" + suffix);
    enc.set_input_array(hidden, 0);
    enc.set_input_array(tiles, 1);
    enc.set_input_array(tile_lists, 2);
    enc.set_input_array(count, 3);
    enc.set_input_array(dn_w, 4);
    enc.set_input_array(dn_s, 5);
    if (dn_aux) enc.set_input_array(*dn_aux, 6); else enc.set_bytes(dummy_f32, 6);
    enc.set_output_array(projected, 7);
    enc.set_bytes(p, 8);
    enc.dispatch_threadgroups_indirect(
        static_cast<const MTL::Buffer*>(count.buffer().ptr()),
        count.offset() + offsetof(Inkling2MoeTileCount, nax_dn_grid),
        MTL::Size(64, 1, 1));
  }

  // 5. Fused Combine + Static ShortConv (k=4, ragged) + Optional FP32 Residual + HistOut (+ Tape)
  set_kernel("inkling2_moe_grouped_combine_sconv_bf16");
  enc.set_input_array(projected, 0);
  enc.set_input_array(inverse, 1);
  enc.set_input_array(gamma, 2);
  enc.set_input_array(post_w, 3);
  if (!cu_seqlens_.empty()) {
    enc.set_bytes(cu_seqlens_.data(), static_cast<int>(cu_seqlens_.size()), 4);
  } else {
    enc.set_bytes(dummy_f32, 4);
  }
  if (res) enc.set_input_array(*res, 5); else enc.set_bytes(dummy_f32, 5);
  for (uint32_t i = 0; i < 4u; ++i) {
    if (hists[i]) {
      enc.set_input_array(*hists[i], 6u + i);
    } else {
      enc.set_bytes(dummy_f32, 6u + i);
    }
  }
  enc.set_output_array(out_y, 10);
  enc.set_output_array(out_hist, 11);
  if (out_tape) enc.set_output_array(*out_tape, 12); else enc.set_bytes(dummy_f32, 12);
  enc.set_bytes(pc, 13);
  enc.dispatch_threadgroups(
      MTL::Size((p.hidden + 31u) / 32u, num_seqs_, 1),
      MTL::Size(256, 1, 1));
}
#else
void Inkling2GroupedMoeFused::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("Inkling2GroupedMoeFused requires Metal");
}
#endif

} // namespace

bool available_for_rows(uint32_t rows, bool check_max_rows = true) {
#ifdef MLX_NODE_METAL_ENABLED
  if (detail::in_grad_tracing() || !(default_device() == Device::gpu) ||
      !is_available(Device::gpu) || !metal::is_available()) {
    return false;
  }
  const uint32_t mode = resolved_moe_mode();
  if (mode == 1u) {
    return false;
  }
  if (rows == 1u) {
    if (mode == 2u) {
      return false;
    }
    return true;
  }
  if (check_max_rows) {
    // Up to M=32 (covers ALL M=1..32 AR decode, 4-lane AR, solo DSpark M=8, and 4-lane DSpark M=32
    // with 100% coalesced SIMD exact-FP32 parity). Above M=32 (prefill), MLX's native 64x64x64
    // `gather_qmm_t_nax` avoids grouped gather/projected copies (+15-25% faster on 2K/8K prefill)
    // and ensures consistent prefill behavior across all chunk sizes (T=56..2048).
    if (rows > resolved_moe_max_rows()) {
      return false;
    }
  }
  return metal::is_nax_available();
#else
  (void)rows;
  (void)check_max_rows;
  return false;
#endif
}

} // namespace mlx::core::inkling2_moe

extern "C" {

bool mlx_inkling2_grouped_moe_available(uint32_t rows) {
  return mlx::core::inkling2_moe::available_for_rows(rows);
}

uint32_t mlx_inkling2_moe_mode() {
#ifdef MLX_NODE_METAL_ENABLED
  return mlx::core::inkling2_moe::resolved_moe_mode();
#else
  return 0u;
#endif
}

uint32_t mlx_inkling2_moe_max_rows() {
#ifdef MLX_NODE_METAL_ENABLED
  return mlx::core::inkling2_moe::resolved_moe_max_rows();
#else
  return 0u;
#endif
}

int32_t mlx_inkling2_moe_col_cutoff_override() {
#ifdef MLX_NODE_METAL_ENABLED
  return mlx::core::inkling2_moe::resolved_moe_col_cutoff_override();
#else
  return -1;
#endif
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
    uint32_t quant_mode,
    bool interleaved_gate_up,
    float gate_softcap,
    float up_softcap,
    bool round_bf16_before_situ) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !top_k_indices_ptr || !gamma_ptr ||
      !gu_w_ptr || !gu_s_ptr || !dn_w_ptr || !dn_s_ptr || quant_mode > 2u) {
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
    if (T < 1 || T > 8192 || (quant_mode == 2u && T > 32) || D < 64 || (D % 64) != 0 ||
        idx.shape(0) != T || gamma.shape(0) != T || gamma.shape(1) != K ||
        K < 1 || K > 16) {
      return nullptr;
    }
    if (!available_for_rows(static_cast<uint32_t>(T), /*check_max_rows=*/false)) {
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

    uint32_t flags = (quant_mode == 1u ? 1u : 0u) | (quant_mode == 2u ? 64u : 0u);
    if (round_bf16_before_situ) {
      flags |= 8u;
    }
    if (interleaved_gate_up) {
      flags |= 32u;
    }

    std::vector<array> inputs = {x, idx, gamma, gu_w, gu_s, dn_w, dn_s};
    bool has_gu_aux = false;
    bool has_dn_aux = false;

    if (quant_mode == 0u) {
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
    } else if (quant_mode == 1u) {
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
    } else {
      // MXFP8: group_size=32, bits=8, uint8 UE8M0 scales, no biases/outer_scale
      if (gu_w.shape(2) != D / 4 || dn_w.shape(2) != I / 4 ||
          gu_s.dtype() != uint8 || dn_s.dtype() != uint8 ||
          gu_s.shape() != Shape{E, N2, D / 32} ||
          dn_s.shape() != Shape{E, D, I / 32}) {
        return nullptr;
      }
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
    p.col_cutoff = (quant_mode == 2u) ? 32u : (p.tile_rows == 8u) ? 4u : 2u;

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
  (void)quant_mode; (void)interleaved_gate_up;
  (void)gate_softcap; (void)up_softcap; (void)round_bf16_before_situ;
  return nullptr;
#endif
}

bool mlx_inkling2_grouped_moe_fused(
    mlx_array* x_ptr,
    mlx_array* router_logits_ptr,
    mlx_array* router_bias_ptr,
    mlx_array* down_outer_scale_ptr,
    mlx_array* gu_w_ptr,
    mlx_array* gu_s_ptr,
    mlx_array* gu_aux_ptr,
    mlx_array* dn_w_ptr,
    mlx_array* dn_s_ptr,
    mlx_array* dn_aux_ptr,
    mlx_array* post_conv_w_ptr,
    mlx_array* residual_ptr,
    const int32_t* cu_seqlens_host,
    uint32_t cu_seqlens_len,
    mlx_array* const* hists_ptr,
    uint32_t num_hists,
    uint32_t num_seqs,
    bool packed_hist,
    uint32_t top_k,
    float route_scale,
    uint32_t quant_mode,
    bool interleaved_gate_up,
    float gate_softcap,
    float up_softcap,
    bool round_bf16_before_situ,
    bool emit_tape,
    mlx_array** out_y,
    mlx_array** out_history,
    mlx_array** out_tape) {
  if (out_y) *out_y = nullptr;
  if (out_history) *out_history = nullptr;
  if (out_tape) *out_tape = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  if (!x_ptr || !router_logits_ptr || !gu_w_ptr || !gu_s_ptr ||
      !dn_w_ptr || !dn_s_ptr || !post_conv_w_ptr || !hists_ptr ||
      !out_y || !out_history || (emit_tape && !out_tape) ||
      num_seqs == 0u || quant_mode > 2u) {
    return false;
  }
  const uint32_t expected_hists = packed_hist ? 1u : num_seqs;
  if (num_hists != expected_hists) {
    return false;
  }
  if (!packed_hist && num_seqs > 4u) {
    return false;
  }
  if (!cu_seqlens_host || cu_seqlens_len != num_seqs + 1u) {
    return false;
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_moe;

    auto is_bf16_or_f32 = [](Dtype dt) { return dt == bfloat16 || dt == float32; };

    const auto& x = *reinterpret_cast<array*>(x_ptr);
    const auto& router_logits = *reinterpret_cast<array*>(router_logits_ptr);
    const auto& gu_w = *reinterpret_cast<array*>(gu_w_ptr);
    const auto& gu_s = *reinterpret_cast<array*>(gu_s_ptr);
    const auto& dn_w = *reinterpret_cast<array*>(dn_w_ptr);
    const auto& dn_s = *reinterpret_cast<array*>(dn_s_ptr);
    const auto& post_w = *reinterpret_cast<array*>(post_conv_w_ptr);

    if (x.ndim() != 2 || x.dtype() != bfloat16 ||
        router_logits.ndim() != 2 || !is_bf16_or_f32(router_logits.dtype()) ||
        post_w.ndim() != 2 || !is_bf16_or_f32(post_w.dtype())) {
      return false;
    }
    const int T = x.shape(0);
    const int D = x.shape(1);
    const int E = router_logits.shape(1);
    const int K = static_cast<int>(top_k);
    if (T < 1 || T > 32 || D < 64 || (D % 64) != 0 ||
        router_logits.shape(0) != T || E < 1 || E > 256 ||
        K < 1 || K > 16 || K > E || T * K > 512 ||
        post_w.shape(0) != D || post_w.shape(1) != 4) {
      return false;
    }
    if (T == 1 && (num_seqs != 1u || packed_hist)) {
      return false;
    }
    if (!available_for_rows(static_cast<uint32_t>(T))) {
      return false;
    }

    if (gu_w.ndim() != 3 || gu_w.dtype() != uint32 ||
        dn_w.ndim() != 3 || dn_w.dtype() != uint32 ||
        gu_s.ndim() != 3 || dn_s.ndim() != 3 ||
        gu_w.shape(0) != E || dn_w.shape(0) != E || dn_w.shape(1) != D) {
      return false;
    }
    const int N2 = gu_w.shape(1);
    if ((N2 % 2) != 0) {
      return false;
    }
    const int I = N2 / 2;
    if (I < 64 || (I % 64) != 0) {
      return false;
    }

    uint32_t flags = (quant_mode == 1u ? 1u : 0u) | (quant_mode == 2u ? 64u : 0u);
    if (round_bf16_before_situ) {
      flags |= 8u;
    }
    if (interleaved_gate_up) {
      flags |= 32u;
    }

    std::vector<array> inputs = {x, router_logits, gu_w, gu_s, dn_w, dn_s, post_w};
    const bool has_bias = router_bias_ptr != nullptr;
    if (has_bias) {
      const auto& b = *reinterpret_cast<array*>(router_bias_ptr);
      if (b.ndim() != 1 || b.shape(0) != E || !is_bf16_or_f32(b.dtype())) {
        return false;
      }
      inputs.push_back(b);
    }
    const bool has_down_scale = down_outer_scale_ptr != nullptr;
    if (has_down_scale) {
      const auto& ds = *reinterpret_cast<array*>(down_outer_scale_ptr);
      if (ds.ndim() != 1 || ds.shape(0) != E || !is_bf16_or_f32(ds.dtype())) {
        return false;
      }
      inputs.push_back(ds);
    }

    bool has_gu_aux = false;
    bool has_dn_aux = false;
    if (quant_mode == 0u) {
      if (gu_w.shape(2) != D / 8 || dn_w.shape(2) != I / 8 ||
          gu_s.dtype() != uint8 || dn_s.dtype() != uint8 ||
          gu_s.shape() != Shape{E, N2, D / 16} ||
          dn_s.shape() != Shape{E, D, I / 16}) {
        return false;
      }
      if (gu_aux_ptr) {
        const auto& gu_outer = *reinterpret_cast<array*>(gu_aux_ptr);
        if (gu_outer.ndim() != 1 || gu_outer.shape(0) != E || gu_outer.dtype() != float32) {
          return false;
        }
        flags |= 2u;
        has_gu_aux = true;
        inputs.push_back(gu_outer);
      }
    } else if (quant_mode == 1u) {
      if (!gu_aux_ptr || !dn_aux_ptr) {
        return false;
      }
      const auto& gu_b = *reinterpret_cast<array*>(gu_aux_ptr);
      const auto& dn_b = *reinterpret_cast<array*>(dn_aux_ptr);
      const Dtype sc_dt = gu_s.dtype();
      if ((sc_dt != bfloat16 && sc_dt != float32) ||
          dn_s.dtype() != sc_dt || gu_b.dtype() != sc_dt || dn_b.dtype() != sc_dt) {
        return false;
      }
      if (gu_w.shape(2) != D / 4 || dn_w.shape(2) != I / 4 ||
          gu_s.shape() != Shape{E, N2, D / 32} ||
          gu_b.shape() != Shape{E, N2, D / 32} ||
          dn_s.shape() != Shape{E, D, I / 32} ||
          dn_b.shape() != Shape{E, D, I / 32}) {
        return false;
      }
      if (sc_dt == float32) {
        flags |= 4u;
      }
      has_gu_aux = true;
      has_dn_aux = true;
      inputs.push_back(gu_b);
      inputs.push_back(dn_b);
    } else {
      if (gu_w.shape(2) != D / 4 || dn_w.shape(2) != I / 4 ||
          gu_s.dtype() != uint8 || dn_s.dtype() != uint8 ||
          gu_s.shape() != Shape{E, N2, D / 32} ||
          dn_s.shape() != Shape{E, D, I / 32}) {
        return false;
      }
    }

    const bool has_res = residual_ptr != nullptr;
    if (has_res) {
      const auto& r = *reinterpret_cast<array*>(residual_ptr);
      if (r.ndim() != 2 || r.shape(0) != T || r.shape(1) != D || !is_bf16_or_f32(r.dtype())) {
        return false;
      }
      inputs.push_back(r);
    }
    std::vector<int32_t> cu_seqlens;
    if (num_seqs > static_cast<uint32_t>(T) || cu_seqlens_host[0] != 0 ||
        cu_seqlens_host[num_seqs] != T) {
      return false;
    }
    for (uint32_t i = 0; i < num_seqs; ++i) {
      if (cu_seqlens_host[i + 1u] <= cu_seqlens_host[i]) {
        return false;
      }
    }
    if (num_seqs > 1u) {
      cu_seqlens.assign(cu_seqlens_host, cu_seqlens_host + cu_seqlens_len);
    }
    const uint32_t h_count = packed_hist ? 1u : num_seqs;
    for (uint32_t i = 0; i < h_count; ++i) {
      if (!hists_ptr[i]) return false;
      const auto& h = *reinterpret_cast<array*>(hists_ptr[i]);
      if (!is_bf16_or_f32(h.dtype())) return false;
      if (packed_hist) {
        if (h.ndim() != 3 || h.shape(0) != static_cast<int>(num_seqs) || h.shape(1) != 3 || h.shape(2) != D) {
          return false;
        }
      } else {
        if (h.ndim() != 2 || h.shape(0) != 3 || h.shape(1) != D) {
          return false;
        }
      }
      inputs.push_back(h);
    }

    Inkling2MoeParams p{};
    p.rows = static_cast<uint32_t>(T);
    p.hidden = static_cast<uint32_t>(D);
    p.intermediate = static_cast<uint32_t>(I);
    p.experts = static_cast<uint32_t>(E);
    p.top_k = top_k;
    p.tile_rows = 8u;
    p.gate_softcap = gate_softcap;
    p.up_softcap = up_softcap;
    p.flags = flags;
    p.col_cutoff = 32u;

    Shape hs = (num_seqs == 1u && !packed_hist)
        ? Shape{3, D}
        : Shape{static_cast<int>(num_seqs), 3, D};
    auto s = to_stream(Device::gpu);
    auto prim = std::make_shared<Inkling2GroupedMoeFused>(
        s,
        p,
        route_scale,
        num_seqs,
        std::move(cu_seqlens),
        packed_hist,
        has_bias,
        has_down_scale,
        has_gu_aux,
        has_dn_aux,
        has_res,
        emit_tape);
    std::vector<Shape> out_shapes = emit_tape ? std::vector<Shape>{x.shape(), hs, x.shape()}
                                              : std::vector<Shape>{x.shape(), hs};
    std::vector<Dtype> out_dtypes = emit_tape ? std::vector<Dtype>{bfloat16, bfloat16, bfloat16}
                                              : std::vector<Dtype>{bfloat16, bfloat16};
    auto results = array::make_arrays(
        std::move(out_shapes),
        std::move(out_dtypes),
        std::move(prim),
        std::move(inputs));
    *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_history = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    if (emit_tape) {
      *out_tape = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
    }
    return true;
  } catch (const std::exception& e) {
    mlx_trace_native_error("inkling2_grouped_moe_fused", e.what());
    return false;
  }
#else
  (void)x_ptr; (void)router_logits_ptr; (void)router_bias_ptr; (void)down_outer_scale_ptr;
  (void)gu_w_ptr; (void)gu_s_ptr; (void)gu_aux_ptr; (void)dn_w_ptr; (void)dn_s_ptr; (void)dn_aux_ptr;
  (void)post_conv_w_ptr; (void)residual_ptr; (void)cu_seqlens_host; (void)cu_seqlens_len; (void)hists_ptr;
  (void)num_hists; (void)num_seqs; (void)packed_hist; (void)top_k; (void)route_scale;
  (void)quant_mode; (void)interleaved_gate_up;
  (void)gate_softcap; (void)up_softcap; (void)round_bf16_before_situ; (void)emit_tape;
  return false;
#endif
}

} // extern "C"
