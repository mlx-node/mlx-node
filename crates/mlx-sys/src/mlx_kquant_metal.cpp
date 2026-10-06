// Metal dispatch for the K-quant primitives. The kernel choice, tile sizes and
// launch geometry mirror MLX's affine dispatcher in
// mlx/backend/metal/quantized.cpp; the kernels are prebuilt into
// paged_attn.metallib from metal/kquant/{kquant,kquant_nax,kquant_m8_nax}.metal.

#include "mlx_kquant.h"
#include "mlx_test_counters.h"

#ifdef MLX_NODE_METAL_ENABLED

#include "mlx/allocator.h"
#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/reduce.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"
#include "mlx_paged_metallib.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>

namespace mlx::core::kquant {

namespace {

const char *type_string(Dtype dtype) {
  switch (dtype) {
  case float32:
    return "float";
  case float16:
    return "float16_t";
  case bfloat16:
    return "bfloat16_t";
  default:
    break;
  }
  std::ostringstream msg;
  msg << "[kquant] Only float32, float16 and bfloat16 are supported but got "
      << dtype << ".";
  throw std::invalid_argument(msg.str());
}

} // namespace

// Kernel names, built the way kquant.metal / kquant_nax.metal instantiate
// them. The dispatchers below and metal_kernel_names() share these builders
// and parameter tables, so a name the dispatcher can request is a name the
// coverage test checks against the metallib.
namespace kernels {

constexpr Mode kModes[] = {Mode::Q6K,    Mode::Q4K,   Mode::Q5K,  Mode::Q3K,
                           Mode::IQ4NL,  Mode::IQ4XS, Mode::IQ3S, Mode::Q2K,
                           Mode::IQ2XXS, Mode::IQ2XS, Mode::IQ2S, Mode::IQ3XXS,
                           Mode::IQ1S,   Mode::IQ1M};
constexpr Dtype kTypes[] = {float32, float16, bfloat16};

// qmv_wide tiles 2..8 input vectors (multi-row matvecs only) at 8 k-lanes.
constexpr int kWideMinVectors = 2;
constexpr int kWideMaxVectors = 8;
constexpr int kWideKLanes = 8;
constexpr int kSplitKSmall = 8;
constexpr int kSplitKLarge = 32;

struct Tile {
  int bm, bn, bk, wm, wn;
};
constexpr Tile kNaxTile{64, 64, 64, 2, 2};
constexpr Tile kRhsTile{16, 32, 32, 1, 2};

// qmv_sg8 is bfloat16 only, for the modes kq_sg8::format decodes.
constexpr Dtype kSg8Type = bfloat16;
constexpr bool sg8_mode(Mode mode) {
  return mode == Mode::Q4K || mode == Mode::Q5K || mode == Mode::Q6K ||
         mode == Mode::IQ4XS;
}

std::string base(Mode mode, const char *family, Dtype type) {
  std::string name;
  concatenate(name, mode_name(mode), "_", family, "_", type_string(type),
              "_gs_", default_group_size(mode), "_b_", default_bits(mode));
  return name;
}
const char *batch(bool batched) { return batched ? "_batch_1" : "_batch_0"; }
const char *aligned(bool aligned_n) {
  return aligned_n ? "_alN_true" : "_alN_false";
}

std::string qmv(Mode m, Dtype t, bool fast, bool batched) {
  return base(m, fast ? "qmv_fast" : "qmv", t) + batch(batched);
}
std::string qmv_wide(Mode m, Dtype t, int vecs_per_tg, bool batched) {
  std::string name = base(m, "qmv_wide", t);
  concatenate(name, "_nv_", vecs_per_tg, "_kl_", kWideKLanes, batch(batched));
  return name;
}
// The Tiled64 layout ("_t64" family suffix, mlx_kquant.h): 2-D weights read
// as x @ w.T with N % 64 == 0, so only the transposed, aligned, unbatched
// kernels exist. M = 1 takes qmv_t64, so qmv_wide_t64 starts at nv_2.
std::string qmv_wide_t64(Mode m, Dtype t, int vecs_per_tg) {
  std::string name = base(m, "qmv_wide_t64", t);
  concatenate(name, "_nv_", vecs_per_tg, "_kl_", kWideKLanes, batch(false));
  return name;
}
// qmv_t64: the lane = row M = 1 matvec, 32 rows x k_splits simdgroups per
// threadgroup (kquant_qmv_t64 in kquant.h); 16 splits from
// kTiledQmvLongK inputs on, where 8 leave the GPU short of threadgroups.
constexpr int kTiledQmvKSplits[] = {8, 16};
constexpr int kTiledQmvLongK = 8192;
int tiled_qmv_k_splits(int K) {
  return K >= kTiledQmvLongK ? kTiledQmvKSplits[1] : kTiledQmvKSplits[0];
}
std::string qmv_t64(Mode m, Dtype t, int k_splits) {
  std::string name = base(m, "qmv_t64", t);
  concatenate(name, "_ks_", k_splits);
  return name;
}
std::string qmv_sg8(Mode m) { return base(m, "qmv_sg8", kSg8Type); }
// qmm_m8_nax: M = 8 bfloat16 on the tensor op; 64-column tiles and at most
// 8 K splits (kquant_m8_nax.h). Every mode in the Tiled64 layout; row-major
// only the modes qmv_sg8 does not decode (q3k, q2k, iq4nl and the grid
// formats), where it measured 1.1-1.4x of qmv_wide on the Qwen3.8 shapes on
// an M5 Max (the sg8 modes only tie qmv_sg8 row-major, 0.85-1.0x).
constexpr int kM8TileCols = 64;
constexpr int kM8MaxSplits = 8;
constexpr bool m8_nax_row_major_mode(Mode mode) {
  return mode == Mode::Q3K || mode == Mode::IQ4NL || mode == Mode::Q2K ||
         is_grid(mode);
}
std::string qmm_m8_nax(Mode m, bool tiled = false) {
  return base(m, tiled ? "qmm_m8_nax_t64" : "qmm_m8_nax", kSg8Type);
}
std::string qmv_sg8_prep(int group_size) {
  std::string name;
  concatenate(name, "kquant_qmv_sg8_prep_", type_string(kSg8Type), "_gs_",
              group_size);
  return name;
}
std::string qvm(Mode m, Dtype t, bool batched) {
  return base(m, "qvm", t) + batch(batched);
}
std::string qvm_split_k(Mode m, Dtype t, int split_k) {
  std::string name = base(m, "qvm_split_k", t);
  concatenate(name, "_spk_", split_k);
  return name;
}
std::string qmm_t_nax(Mode m, Dtype t, bool aligned_n, bool batched,
                      bool tiled = false) {
  std::string name = base(m, tiled ? "qmm_t_nax_t64" : "qmm_t_nax", t);
  concatenate(name, "_bm", kNaxTile.bm, "_bn", kNaxTile.bn, "_bk", kNaxTile.bk,
              "_wm", kNaxTile.wm, "_wn", kNaxTile.wn, aligned(aligned_n),
              batch(batched));
  return name;
}
std::string qmm_t(Mode m, Dtype t, bool aligned_n, bool batched,
                  bool tiled = false) {
  return base(m, tiled ? "qmm_t_t64" : "qmm_t", t) + aligned(aligned_n) +
         batch(batched);
}
std::string qmm_n(Mode m, Dtype t, bool batched) {
  return base(m, "qmm_n", t) + batch(batched);
}
std::string qmm_t_splitk(Mode m, Dtype t, bool aligned_n, bool tiled = false) {
  return base(m, tiled ? "qmm_t_splitk_t64" : "qmm_t_splitk", t) +
         aligned(aligned_n);
}
std::string gather_qmm_t(Mode m, Dtype t, bool aligned_n) {
  return base(m, "gather_qmm_t", t) + aligned(aligned_n);
}
std::string gather_qmm_n(Mode m, Dtype t) { return base(m, "gather_qmm_n", t); }
std::string gather_qmv(Mode m, Dtype t, bool fast) {
  return base(m, fast ? "gather_qmv_fast" : "gather_qmv", t);
}
std::string gather_qvm(Mode m, Dtype t) { return base(m, "gather_qvm", t); }
std::string gather_qmm_rhs(Mode m, Dtype t, bool transpose) {
  std::string name =
      base(m, transpose ? "gather_qmm_rhs_nt" : "gather_qmm_rhs_nn", t);
  concatenate(name, "_bm_", kRhsTile.bm, "_bn_", kRhsTile.bn, "_bk_",
              kRhsTile.bk, "_wm_", kRhsTile.wm, "_wn_", kRhsTile.wn);
  return name;
}
std::string dequantize(Mode m, Dtype t) { return base(m, "dequantize", t); }

} // namespace kernels

std::vector<KernelName> metal_kernel_names() {
  using namespace kernels;
  std::vector<KernelName> names;
  auto add = [&names](std::string name, bool nax = false) {
    names.push_back({std::move(name), nax});
  };
  for (Dtype t : kTypes) {
    for (Mode m : kModes) {
      for (bool b : {false, true}) {
        add(qmv(m, t, false, b));
        add(qmv(m, t, true, b));
        for (int v = kWideMinVectors; v <= kWideMaxVectors; ++v) {
          add(qmv_wide(m, t, v, b));
        }
        add(qvm(m, t, b));
        add(qmm_n(m, t, b));
        for (bool a : {false, true}) {
          add(qmm_t(m, t, a, b));
          add(qmm_t_nax(m, t, a, b), true);
        }
      }
      for (int split_k : {kSplitKSmall, kSplitKLarge}) {
        add(qvm_split_k(m, t, split_k));
      }
      for (bool a : {false, true}) {
        add(qmm_t_splitk(m, t, a));
        add(gather_qmm_t(m, t, a));
      }
      add(gather_qmm_n(m, t));
      add(gather_qmv(m, t, false));
      add(gather_qmv(m, t, true));
      add(gather_qvm(m, t));
      add(gather_qmm_rhs(m, t, true));
      add(gather_qmm_rhs(m, t, false));
      add(dequantize(m, t));
      // Tiled64: transposed, aligned, unbatched only.
      for (int v = kWideMinVectors; v <= kWideMaxVectors; ++v) {
        add(qmv_wide_t64(m, t, v));
      }
      for (int ks : kTiledQmvKSplits) {
        add(qmv_t64(m, t, ks));
      }
      add(qmm_t(m, t, true, false, true));
      add(qmm_t_splitk(m, t, true, true));
      add(qmm_t_nax(m, t, true, false, true), true);
    }
  }
  for (Mode m : kModes) {
    if (m8_nax_row_major_mode(m)) {
      add(qmm_m8_nax(m), true);
    }
    add(qmm_m8_nax(m, true), true);
  }
  std::vector<int> prep_group_sizes;
  for (Mode m : kModes) {
    if (!sg8_mode(m)) {
      continue;
    }
    add(qmv_sg8(m));
    int gs = default_group_size(m);
    if (std::find(prep_group_sizes.begin(), prep_group_sizes.end(), gs) ==
        prep_group_sizes.end()) {
      prep_group_sizes.push_back(gs);
      add(qmv_sg8_prep(gs));
    }
  }
  return names;
}

namespace {

// The prebuilt pipeline for `kname`. Function-constant kernels pass the
// specialization in `hash_name` / `func_consts`.
MTL::ComputePipelineState *
load_kernel(metal::Device &d, const std::string &kname,
            const std::string &hash_name = "",
            const metal::MTLFCList &func_consts = {}) {
  return fast::paged::get_prebuilt_kernel(d, "kquant", "K-quant", kname,
                                          hash_name, func_consts);
}

// Under MLX_METAL_COMMAND_TRACE (any value >= 1, the fork's command-trace
// diagnostic), prints each distinct (kernel, M, N, K) the dispatcher
// requests, once.
void log_route(const std::string &kname, int M, int N, int K) {
  static const bool enabled = env::get_var("MLX_METAL_COMMAND_TRACE", 0) >= 1;
  if (!enabled) {
    return;
  }
  static std::mutex mutex;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> lock(mutex);
  std::string key;
  concatenate(key, kname, " M=", M, " N=", N, " K=", K);
  if (seen.insert(key).second) {
    std::cerr << "[kquant route] " << key << std::endl;
  }
}

MTL::ComputePipelineState *get_kernel(metal::Device &d,
                                      std::string_view family,
                                      const std::string &kname, int M = 0,
                                      int N = 0, int K = 0) {
  bridge_testing::record(family);
  log_route(kname, M, N, K);
  return load_kernel(d, kname);
}

inline array ensure_row_contiguous(const array &x, const Stream &s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    metal::get_command_encoder(s).add_temporary(x_copy);
    return x_copy;
  }
  return x;
}

inline array ensure_row_contiguous_matrix(const array &x, const Stream &s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(x_copy);
  return x_copy;
}

inline int get_qmv_batch_limit(int D, int O, metal::Device &d) {
  auto arch_size = d.get_architecture().back();
  auto arch_gen = d.get_architecture_gen();
  if (arch_gen == 13 || arch_gen == 14) {
    switch (arch_size) {
    case 'd':
      if (D <= 2048 && O <= 2048) {
        return 32;
      } else if (D <= 4096 && O <= 4096) {
        return 18;
      } else {
        return 12;
      }
    default:
      if (D <= 2048 && O <= 2048) {
        return 14;
      } else if (D <= 4096 && O <= 4096) {
        return 10;
      } else {
        return 6;
      }
    }
  } else {
    switch (arch_size) {
    case 'd':
      if (D <= 2048 && O <= 2048) {
        return 32;
      } else if (D <= 4096 && O <= 4096) {
        return 18;
      } else {
        return 12;
      }
    default:
      if (D <= 2048 && O <= 2048) {
        return 18;
      } else if (D <= 4096 && O <= 4096) {
        return 12;
      } else {
        return 10;
      }
    }
  }
}

inline int add_strides_and_shapes(metal::CommandEncoder &enc, bool skip,
                                  const array &x, const array &w,
                                  const array &scales, const array &biases,
                                  int offset) {
  if (skip) {
    return 0;
  }
  int x_batch_ndims = x.ndim() - 2;
  int w_batch_ndims = w.ndim() - 2;
  enc.set_bytes(x_batch_ndims, offset++);
  enc.set_vector_bytes(x.shape(), offset++);
  enc.set_vector_bytes(x.strides(), offset++);
  enc.set_bytes(w_batch_ndims, offset++);
  enc.set_vector_bytes(w.shape(), offset++);
  enc.set_vector_bytes(w.strides(), offset++);
  enc.set_vector_bytes(scales.strides(), offset++);
  enc.set_vector_bytes(biases.strides(), offset++);
  return offset;
}

inline int add_gather_strides_and_shapes(metal::CommandEncoder &enc,
                                         const array &lhs_indices,
                                         const array &rhs_indices, int offset) {
  auto [shape, strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  int ndims = shape.size();
  enc.set_bytes(ndims, offset++);
  enc.set_vector_bytes(shape, offset++);
  enc.set_vector_bytes(strides[0], offset++);
  enc.set_vector_bytes(strides[1], offset++);
  return offset;
}

void set_weights(metal::CommandEncoder &enc, const array &w,
                 const array &scales, const array &biases) {
  enc.set_input_array(w, 0);
  enc.set_input_array(scales, 1);
  enc.set_input_array(biases, 2);
}

struct Operands {
  const array &x;
  const array &w;
  const array &scales;
  const array &biases;
  array &out;
  int group_size;
  int bits;
  Mode mode;
  metal::Device &d;
  const Stream &s;
  // Tiled64 weight layout (mlx_kquant.h): routes only to the "_t64" kernels.
  bool tiled = false;
};

// Kernel-family counter names carry the layout so a test can see that a
// tiled tensor never reached a row-major kernel.
std::string family_of(const char *family, bool tiled) {
  return tiled ? std::string(family) + "_t64" : std::string(family);
}

void qmv(const Operands &o, int M, int N, int K) {
  int B = o.out.size() / M / N;
  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  bool fast = N % bn == 0 && K % 512 == 0;
  auto kernel = get_kernel(o.d, fast ? "qmv_fast" : "qmv",
                           kernels::qmv(o.mode, o.x.dtype(), fast, B > 1), M,
                           N, K);
  auto &enc = metal::get_command_encoder(o.s);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(o.x, 3);
  enc.set_output_array(o.out, 4);
  enc.set_bytes(K, 5);
  enc.set_bytes(N, 6);
  add_strides_and_shapes(enc, B <= 1, o.x, o.w, o.scales, o.biases, 7);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

// K-quants share affine's ALU-per-load balance, so qmv_wide only beats qmv on
// gen-15+.
inline bool use_qmv_wide(metal::Device &d) {
  return d.get_architecture_gen() >= 15;
}

// The tiled M = 1 matvec, lane = row: grid (M, N / 32) of 32 x k_splits
// threads (kquant_qmv_t64). Measured against qmv_wide_t64 nv_1 at 8 and 4
// k-lanes on the Qwen3.8 shapes; lane = row won.
void qmv_t64(const Operands &o, int M, int N, int K) {
  constexpr int rows_per_tg = 32;
  const int k_splits = kernels::tiled_qmv_k_splits(K);
  if (bridge_testing::counting) {
    bridge_testing::record("qmv_t64_ks" + std::to_string(k_splits));
  }
  auto kernel = get_kernel(o.d, "qmv_t64",
                           kernels::qmv_t64(o.mode, o.x.dtype(), k_splits), M,
                           N, K);
  auto &enc = metal::get_command_encoder(o.s);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(o.x, 3);
  enc.set_output_array(o.out, 4);
  enc.set_bytes(K, 5);
  enc.set_bytes(N, 6);
  enc.dispatch_threadgroups(MTL::Size(M, N / rows_per_tg, 1),
                            MTL::Size(32, k_splits, 1));
}

void qmv_wide(const Operands &o, int M, int N, int K) {
  // Each tile re-reads the weights: fewest tiles, then the smallest tile that
  // fills them. Up to 8 vectors per tile only when N is large enough that the
  // row dimension already saturates the GPU.
  const int tile_cap = N >= 2048 ? kernels::kWideMaxVectors : 5;
  int n_tiles = (M + tile_cap - 1) / tile_cap;
  int vecs_per_tg = (M + n_tiles - 1) / n_tiles;
  constexpr int k_lanes = kernels::kWideKLanes;
  constexpr int num_simdgroups = 2;
  int B = o.out.size() / M / N;
  bool batched = B > 1;
  int rows_per_tg = (32 / k_lanes) * num_simdgroups;

  MTL::Size group_dims(32, num_simdgroups, 1);
  MTL::Size grid_dims((M + vecs_per_tg - 1) / vecs_per_tg,
                      (N + rows_per_tg - 1) / rows_per_tg, B);

  if (bridge_testing::counting) {
    bridge_testing::record(family_of("qmv_wide", o.tiled) + "_nv" +
                           std::to_string(vecs_per_tg));
  }
  auto kernel = get_kernel(
      o.d, family_of("qmv_wide", o.tiled),
      o.tiled ? kernels::qmv_wide_t64(o.mode, o.x.dtype(), vecs_per_tg)
              : kernels::qmv_wide(o.mode, o.x.dtype(), vecs_per_tg, batched),
      M, N, K);
  auto &enc = metal::get_command_encoder(o.s);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(o.x, 3);
  enc.set_output_array(o.out, 4);
  enc.set_bytes(K, 5);
  enc.set_bytes(N, 6);
  enc.set_bytes(M, 7);
  add_strides_and_shapes(enc, !batched, o.x, o.w, o.scales, o.biases, 8);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

// 8-row bfloat16 matvec on gen-17+, where it beats qmv_wide. The kernel has no
// batch strides, no column tail and only aligned vector loads (x: uint2,
// scales: uint4, biases: half2).
bool use_qmv_sg8(const Operands &o, int M, int N) {
  if (M != 8 || o.d.get_architecture_gen() < 17) {
    return false;
  }
  if (!kernels::sg8_mode(o.mode)) {
    return false;
  }
  return o.x.dtype() == kernels::kSg8Type && o.out.size() == size_t(8) * N &&
         N % 32 == 0 && o.x.offset() % 8 == 0 && o.scales.offset() % 16 == 0 &&
         o.biases.offset() % 4 == 0;
}

// Two dispatches: the prep packs x into MMA operand order with fp32 per-group
// row sums, then qmv_sg8 runs over N / 32 threadgroups.
void qmv_sg8(const Operands &o, int N, int K) {
  array bt({K * 4}, uint32, nullptr, {});
  array sums({(K / o.group_size) * 8}, float32, nullptr, {});
  bt.set_data(allocator::malloc(bt.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  auto &enc = metal::get_command_encoder(o.s);
  enc.add_temporary(bt);
  enc.add_temporary(sums);

  auto prep = load_kernel(o.d, kernels::qmv_sg8_prep(o.group_size));
  enc.set_compute_pipeline_state(prep);
  enc.set_input_array(o.x, 0);
  enc.set_output_array(bt, 1);
  enc.set_output_array(sums, 2);
  enc.set_bytes(K, 3);
  enc.dispatch_threadgroups(MTL::Size((K / 32 + 3) / 4, 1, 1),
                            MTL::Size(128, 1, 1));

  auto kernel = get_kernel(o.d, "qmv_sg8", kernels::qmv_sg8(o.mode), 8, N, K);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(bt, 3);
  enc.set_input_array(sums, 4);
  enc.set_output_array(o.out, 5);
  enc.set_bytes(K, 6);
  enc.set_bytes(N, 7);
  enc.dispatch_threadgroups(MTL::Size(N / 32, 1, 1), MTL::Size(128, 1, 1));
}

// GPU cores from the IORegistry `gpu-core-count` of the accelerator entry
// behind the Metal device (its registry ID, or an ancestor up to 4 levels;
// the first IOAccelerator service as the fallback). Metal itself does not
// expose the count. When nothing publishes it, 8: the smallest Apple
// silicon GPUs have 7 to 10, so the split rule errs towards fewer splits.
int gpu_core_count(metal::Device &d) {
  static const int cores = [&d]() {
    int count = 0;
    auto read = [&count](io_registry_entry_t entry) {
      if (!entry) {
        return false;
      }
      CFTypeRef value = IORegistryEntryCreateCFProperty(
          entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
      if (value) {
        int64_t number = 0;
        if (CFGetTypeID(value) == CFNumberGetTypeID() &&
            CFNumberGetValue(static_cast<CFNumberRef>(value),
                             kCFNumberSInt64Type, &number) &&
            number > 0 && number <= 4096) {
          count = static_cast<int>(number);
        }
        CFRelease(value);
      }
      return count != 0;
    };
    io_registry_entry_t entry = IOServiceGetMatchingService(
        kIOMainPortDefault,
        IORegistryEntryIDMatching(d.mtl_device()->registryID()));
    for (int depth = 0; entry && depth < 4 && !read(entry); ++depth) {
      io_registry_entry_t parent = MACH_PORT_NULL;
      if (IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent) !=
          KERN_SUCCESS) {
        parent = MACH_PORT_NULL;
      }
      IOObjectRelease(entry);
      entry = parent;
    }
    if (entry) {
      IOObjectRelease(entry);
    }
    if (!count) {
      io_registry_entry_t accelerator = IOServiceGetMatchingService(
          kIOMainPortDefault, IOServiceMatching("IOAccelerator"));
      if (accelerator) {
        read(accelerator);
        IOObjectRelease(accelerator);
      }
    }
    return count ? count : 8;
  }();
  return cores;
}

// K splits of an M = 8 tensor-op dispatch (Splash LinearGguf.cpp
// decodeSplits, staged tier {6 threadgroups per core, 512 inputs per
// partition}): doubles while the grid holds fewer than 6 threadgroups per
// core and each partition would keep 512 inputs of whole 32-input units.
int qmm_m8_nax_splits(int N, int K, int cores) {
  const int64_t tiles = N / kernels::kM8TileCols;
  int splits = 1;
  while (splits < kernels::kM8MaxSplits &&
         tiles * splits < int64_t(6) * cores && K / (2 * splits) >= 512 &&
         (K / 32) % (2 * splits) == 0) {
    splits *= 2;
  }
  return splits;
}

// Tensor-op M = 8 matmul on gen-17+ (is_nax_available): whole 64-column
// tiles, whole 32-input units, no batch, aligned operands (x rows as 16-byte
// tensor rows, scales as ushort pairs, biases as half2 pairs). Every Tiled64
// mode (1.18-1.40x of qmv_sg8 on the Qwen3.8 shapes); row-major only the
// modes of kernels::m8_nax_row_major_mode.
bool use_qmm_m8_nax(const Operands &o, int M, int N, int K) {
  if (M != 8 || !metal::is_nax_available()) {
    return false;
  }
  if (!o.tiled && !kernels::m8_nax_row_major_mode(o.mode)) {
    return false;
  }
  return o.x.dtype() == kernels::kSg8Type && o.out.dtype() == kernels::kSg8Type &&
         o.out.size() == size_t(8) * N && N % kernels::kM8TileCols == 0 &&
         K % 32 == 0 && o.x.offset() % 8 == 0 && o.scales.offset() % 2 == 0 &&
         o.biases.offset() % 2 == 0;
}

// The split-K arrival counters of a stream: one uint32 per 64-column tile,
// zero between dispatches (the last arriver of a tile resets its counter).
// Bound as an output so the encoder orders dispatches that share them; each
// stream has its own because MLX fences outputs per stream only.
array m8_nax_counters(const Stream &s, int tiles) {
  static std::mutex mutex;
  static std::unordered_map<int, array> by_stream;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = by_stream.find(s.index);
  if (it != by_stream.end() && it->second.size() >= size_t(tiles)) {
    return it->second;
  }
  auto &enc = metal::get_command_encoder(s);
  int size = std::max(tiles, 4096);
  array zero(uint32_t(0), uint32);
  array counters({size}, uint32, nullptr, {});
  fill_gpu(zero, counters, s);
  enc.add_temporary(zero);
  if (it != by_stream.end()) {
    enc.add_temporary(it->second);
    it->second = counters;
  } else {
    by_stream.emplace(s.index, counters);
  }
  return counters;
}

// One dispatch: grid (N / 64, splits) of 64 threads. Splits > 1 publish fp32
// partials [splits][8][N] in a temporary; the last-arriving partition of a
// tile sums them in split order (kquant_m8_nax.h).
void qmm_m8_nax(const Operands &o, int N, int K) {
  const int tiles = N / kernels::kM8TileCols;
  const int splits = qmm_m8_nax_splits(N, K, gpu_core_count(o.d));
  if (bridge_testing::counting) {
    bridge_testing::record("qmm_m8_nax_splits" + std::to_string(splits));
  }
  auto kernel = get_kernel(o.d, family_of("qmm_m8_nax", o.tiled),
                           kernels::qmm_m8_nax(o.mode, o.tiled), 8, N, K);
  auto &enc = metal::get_command_encoder(o.s);
  array partials = o.out;
  array counters = o.out;
  if (splits > 1) {
    partials = array({splits * 8 * N}, float32, nullptr, {});
    partials.set_data(allocator::malloc(partials.nbytes()));
    enc.add_temporary(partials);
    counters = m8_nax_counters(o.s, tiles);
  }
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(o.x, 3);
  enc.set_output_array(o.out, 4);
  enc.set_output_array(partials, 5);
  enc.set_output_array(counters, 6);
  enc.set_bytes(K, 7);
  enc.set_bytes(N, 8);
  enc.set_bytes(splits, 9);
  enc.dispatch_threadgroups(MTL::Size(tiles, splits, 1), MTL::Size(64, 1, 1));
}

// The K-quant qvm kernels tile 32 / pack_factor packs per thread, so a
// simdgroup covers 32 columns whatever the group size (16 for q6k/q3k).
constexpr int qvm_columns_per_simdgroup = 32;

void qvm_split_k(const Operands &o, int M, int N, int K) {
  auto &enc = metal::get_command_encoder(o.s);
  const auto &x = o.x;
  const auto &w = o.w;
  const auto &scales = o.scales;
  const auto &biases = o.biases;

  int split_k = K > 8192 ? kernels::kSplitKLarge : kernels::kSplitKSmall;
  int split_D = (K + split_k - 1) / split_k;
  int B = o.out.size() / M / N;
  B *= split_k;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  // Round up: IQ4_NL allows N in 32s, and the kernel skips a simdgroup whose
  // columns start at or past N.
  int bn = qvm_columns_per_simdgroup * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  auto x_shape = x.shape();
  auto x_strides = x.strides();
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    x_strides.insert(x_strides.begin(), 0);
  }
  int x_ndim = x_shape.size();
  int x_batch_ndims = x_ndim - 2;
  int w_batch_ndims = w.ndim() - 2;
  auto w_shape = w.shape();
  auto w_strides = w.strides();
  auto s_strides = scales.strides();

  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x_ndim - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * w.shape(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * scales.shape(-1));

  int final_block_size = K - (split_k - 1) * split_D;

  auto temp_shape = o.out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  enc.add_temporary(intermediate);

  auto kernel = get_kernel(o.d, "qvm_split_k",
                           kernels::qvm_split_k(o.mode, x.dtype(), split_k));
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, w, scales, biases);
  int c = 3;
  enc.set_input_array(x, c++);
  enc.set_output_array(intermediate, c++);
  enc.set_bytes(split_D, c++);
  enc.set_bytes(N, c++);
  enc.set_bytes(x_batch_ndims, c++);
  enc.set_vector_bytes(x_shape, c++);
  enc.set_vector_bytes(x_strides, c++);
  enc.set_bytes(w_batch_ndims, c++);
  enc.set_vector_bytes(w_shape, c++);
  enc.set_vector_bytes(w_strides, c++);
  enc.set_vector_bytes(s_strides, c++);
  auto b_strides = biases.strides();
  b_strides.insert(b_strides.end() - 2, split_D * biases.shape(-1));
  enc.set_vector_bytes(b_strides, c++);
  enc.set_bytes(final_block_size, c++);
  enc.dispatch_threadgroups(grid_dims, group_dims);

  int axis = intermediate.ndim() - 3;
  ReductionPlan plan(ReductionOpType::ContiguousStridedReduce,
                     {intermediate.shape(axis)}, {intermediate.strides(axis)});
  strided_reduce_general_dispatch(intermediate, o.out, "sum", plan, {axis}, enc,
                                  o.d, o.s);
}

void qvm(const Operands &o, int M, int N, int K) {
  int B = o.out.size() / M / N;
  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = qvm_columns_per_simdgroup * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  auto kernel =
      get_kernel(o.d, "qvm", kernels::qvm(o.mode, o.x.dtype(), B > 1));
  auto &enc = metal::get_command_encoder(o.s);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(o.x, 3);
  enc.set_output_array(o.out, 4);
  enc.set_bytes(K, 5);
  enc.set_bytes(N, 6);
  add_strides_and_shapes(enc, B <= 1, o.x, o.w, o.scales, o.biases, 7);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

// The NAX family covers qmm with a transposed weight only.
void qmm_nax(const Operands &o, int M, int N, int K) {
  int B = o.out.size() / M / N;
  constexpr auto tile = kernels::kNaxTile;
  MTL::Size group_dims(32, tile.wn, tile.wm);
  MTL::Size grid_dims((N + tile.bn - 1) / tile.bn, (M + tile.bm - 1) / tile.bm,
                      B);

  bool aligned = N % tile.bn == 0;
  auto kernel = get_kernel(
      o.d, family_of("qmm_t_nax", o.tiled),
      kernels::qmm_t_nax(o.mode, o.x.dtype(), aligned, B > 1, o.tiled), M, N,
      K);
  auto &enc = metal::get_command_encoder(o.s);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(o.x, 3);
  enc.set_output_array(o.out, 4);
  enc.set_bytes(K, 5);
  enc.set_bytes(N, 6);
  enc.set_bytes(M, 7);
  add_strides_and_shapes(enc, B <= 1, o.x, o.w, o.scales, o.biases, 8);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm(const Operands &o, bool transpose, int M, int N, int K) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || o.x.dtype() != float32)) {
    return qmm_nax(o, M, N, K);
  }

  int B = o.out.size() / M / N;
  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  bool aligned = N % 32 == 0;
  bool batched = B > 1;
  auto kernel = transpose
      ? get_kernel(
            o.d, family_of("qmm_t", o.tiled),
            kernels::qmm_t(o.mode, o.x.dtype(), aligned, batched, o.tiled), M,
            N, K)
      : get_kernel(o.d, "qmm_n", kernels::qmm_n(o.mode, o.x.dtype(), batched),
                   M, N, K);
  auto &enc = metal::get_command_encoder(o.s);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(o.x, 3);
  enc.set_output_array(o.out, 4);
  enc.set_bytes(K, 5);
  enc.set_bytes(N, 6);
  enc.set_bytes(M, 7);
  add_strides_and_shapes(enc, B <= 1, o.x, o.w, o.scales, o.biases, 8);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_splitk(const Operands &o, int M, int N, int K) {
  // Target ~512 threadgroups.
  int bm = 32, bn = 32;
  int n_tiles = (N + bn - 1) / bn;
  int m_tiles = (M + bm - 1) / bm;
  int current_tgs = n_tiles * m_tiles;
  int split_k = std::max(1, 512 / current_tgs);

  // qmm_t_splitk tiles K by BK=32 without bounding it, so every partition
  // must be whole BK tiles and whole groups.
  int k_align = o.group_size > 32 ? o.group_size : 32;
  split_k = std::min(split_k, K / k_align);
  while (split_k > 1 && (K % (split_k * k_align) != 0)) {
    split_k--;
  }
  if (split_k <= 1) {
    return qmm(o, true, M, N, K);
  }

  int k_partition_size = K / split_k;
  int split_k_partition_stride = M * N;

  auto &enc = metal::get_command_encoder(o.s);
  auto temp_shape = o.out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.begin(), split_k);
  array intermediate(temp_shape, o.x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  enc.add_temporary(intermediate);

  MTL::Size group_dims(32, 2, 2);
  MTL::Size grid_dims(n_tiles, m_tiles, split_k);

  bool aligned = N % 32 == 0;
  auto kernel = get_kernel(
      o.d, family_of("qmm_t_splitk", o.tiled),
      kernels::qmm_t_splitk(o.mode, o.x.dtype(), aligned, o.tiled), M, N, K);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(o.x, 3);
  enc.set_output_array(intermediate, 4);
  enc.set_bytes(K, 5);
  enc.set_bytes(N, 6);
  enc.set_bytes(M, 7);
  enc.set_bytes(k_partition_size, 8);
  enc.set_bytes(split_k_partition_stride, 9);
  enc.dispatch_threadgroups(grid_dims, group_dims);

  ReductionPlan plan(ReductionOpType::ContiguousStridedReduce,
                     {intermediate.shape(0)}, {intermediate.strides(0)});
  strided_reduce_general_dispatch(intermediate, o.out, "sum", plan, {0}, enc,
                                  o.d, o.s);
}

void gather_qmm(const Operands &o, const array &lhs_indices,
                const array &rhs_indices, bool transpose, int M, int N, int K) {
  int B = o.out.size() / M / N;
  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  bool aligned = N % 32 == 0;
  auto kernel =
      transpose
          ? get_kernel(o.d, "gather_qmm_t",
                       kernels::gather_qmm_t(o.mode, o.x.dtype(), aligned))
          : get_kernel(o.d, "gather_qmm_n",
                       kernels::gather_qmm_n(o.mode, o.x.dtype()));
  auto &enc = metal::get_command_encoder(o.s);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  int c = 3;
  enc.set_input_array(o.x, c++);
  enc.set_input_array(lhs_indices, c++);
  enc.set_input_array(rhs_indices, c++);
  enc.set_output_array(o.out, c++);
  enc.set_bytes(K, c++);
  enc.set_bytes(N, c++);
  enc.set_bytes(M, c++);
  c = add_strides_and_shapes(enc, false, o.x, o.w, o.scales, o.biases, c);
  add_gather_strides_and_shapes(enc, lhs_indices, rhs_indices, c);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmv(const Operands &o, const array &lhs_indices,
                const array &rhs_indices, int M, int N, int K) {
  int B = o.out.size() / M / N;
  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  bool fast = N % bn == 0 && K % 512 == 0;
  auto kernel = get_kernel(o.d, fast ? "gather_qmv_fast" : "gather_qmv",
                           kernels::gather_qmv(o.mode, o.x.dtype(), fast));
  auto &enc = metal::get_command_encoder(o.s);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  int c = 3;
  enc.set_input_array(o.x, c++);
  enc.set_input_array(lhs_indices, c++);
  enc.set_input_array(rhs_indices, c++);
  enc.set_output_array(o.out, c++);
  enc.set_bytes(K, c++);
  enc.set_bytes(N, c++);
  c = add_strides_and_shapes(enc, false, o.x, o.w, o.scales, o.biases, c);
  add_gather_strides_and_shapes(enc, lhs_indices, rhs_indices, c);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qvm(const Operands &o, const array &lhs_indices,
                const array &rhs_indices, int M, int N, int K) {
  int B = o.out.size() / M / N;
  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = qvm_columns_per_simdgroup * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  auto kernel =
      get_kernel(o.d, "gather_qvm", kernels::gather_qvm(o.mode, o.x.dtype()));
  auto &enc = metal::get_command_encoder(o.s);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  int c = 3;
  enc.set_input_array(o.x, c++);
  enc.set_input_array(lhs_indices, c++);
  enc.set_input_array(rhs_indices, c++);
  enc.set_output_array(o.out, c++);
  enc.set_bytes(K, c++);
  enc.set_bytes(N, c++);
  c = add_strides_and_shapes(enc, false, o.x, o.w, o.scales, o.biases, c);
  add_gather_strides_and_shapes(enc, lhs_indices, rhs_indices, c);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

// lhs_indices are implied: x is broadcast against rhs_indices and copied as
// if the lhs gather had been applied, then the sorted rhs walk reuses each
// expert's weights across consecutive rows.
void gather_qmm_rhs(const array &x_, const array &w_, const array &scales_,
                    const array &biases_, const array &indices_, array &out,
                    bool transpose, Mode mode, int M, int N, int K,
                    metal::Device &d, const Stream &s) {
  array indices = ensure_row_contiguous(indices_, s);
  auto broadcast_with_indices = [&s, &indices](const array &x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, s);
    }
    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, s);
  };
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, s);
  array scales = ensure_row_contiguous(scales_, s);

  constexpr auto tile = kernels::kRhsTile;
  const bool align_M = (M % tile.bm) == 0;
  const bool align_N = (N % tile.bn) == 0;
  const bool align_K = (K % tile.bk) == 0;

  std::string kname = kernels::gather_qmm_rhs(mode, x.dtype(), transpose);
  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };
  std::string hash_name;
  concatenate(hash_name, kname, "_align_M_", align_M ? 't' : 'n', "_align_N_",
              align_N ? 't' : 'n', "_align_K_", align_K ? 't' : 'n');

  auto &enc = metal::get_command_encoder(s);
  bridge_testing::record(transpose ? "gather_qmm_rhs_nt" : "gather_qmm_rhs_nn");
  auto kernel = load_kernel(d, kname, hash_name, func_consts);
  enc.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, tile.wn, tile.wm);
  MTL::Size grid_dims((N + tile.bn - 1) / tile.bn, (M + tile.bm - 1) / tile.bm,
                      1);

  int c = 0;
  enc.set_input_array(x, c++);
  enc.set_input_array(w, c++);
  enc.set_input_array(scales, c++);
  array biases = ensure_row_contiguous(biases_, s);
  enc.set_input_array(biases, c++);
  enc.set_input_array(indices, c++);
  enc.set_output_array(out, c++);
  enc.set_bytes(M, c++);
  enc.set_bytes(N, c++);
  enc.set_bytes(K, c++);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

int qmv_vector_limit(int K, int N, bool transpose, metal::Device &d) {
  int vector_limit = transpose ? get_qmv_batch_limit(K, N, d) : 4;
  if (const char *e = std::getenv("MLX_QMM_SPLITK_MIN_M")) {
    int v = std::atoi(e);
    if (v > 0) {
      vector_limit = v;
    }
  }
  return vector_limit;
}

} // namespace

void KQuantMatmul::eval_gpu(const std::vector<array> &inputs, array &out) {
  auto &s = stream();
  auto &d = metal::device(s.device);
  out.set_data(allocator::malloc(out.nbytes()));

  const bool tiled = layout_ == Layout::Tiled64;
  // The tiled kernels take x as one row-contiguous [M, K] (no batch strides).
  array x = tiled ? ensure_row_contiguous(inputs[0], s)
                  : ensure_row_contiguous_matrix(inputs[0], s);
  array w = ensure_row_contiguous_matrix(inputs[1], s);
  array scales = ensure_row_contiguous_matrix(inputs[2], s);
  array biases = ensure_row_contiguous_matrix(inputs[3], s);

  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);
  Operands o{x,     w,     scales, biases, out, group_size_,
             bits_, mode_, d,      s,      tiled};

  if (tiled) {
    // quantized_matmul() admits Tiled64 only as a 2-D transposed weight; the
    // routes below are the ported kernels, every other kernel reads the
    // row-major layout and must not be reached.
    if (!transpose_ || !non_batched) {
      throw std::runtime_error(
          "[quantized_matmul] The @t64 layout needs x @ w.T on a 2-D weight.");
    }
    if (M >= qmv_vector_limit(K, N, transpose_, d)) {
      qmm_splitk(o, M, N, K);
      return;
    }
    if (use_qmm_m8_nax(o, M, N, K)) {
      qmm_m8_nax(o, N, K);
      return;
    }
    if (M == 1) {
      qmv_t64(o, M, N, K);
      return;
    }
    qmv_wide(o, M, N, K);
    return;
  }

  if (M >= qmv_vector_limit(K, N, transpose_, d)) {
    int B = out.size() / M / N;
    if (transpose_ && B == 1) {
      qmm_splitk(o, M, N, K);
      return;
    }
    qmm(o, transpose_, M, N, K);
    return;
  }

  if (transpose_) {
    // MLX routes K of 64 or 128 to qmv_quad, which has no K-quant kernel;
    // only IQ4_NL (32-value blocks) can reach those K, and qmv covers it.
    if (use_qmm_m8_nax(o, M, N, K)) {
      qmm_m8_nax(o, N, K);
      return;
    }
    if (use_qmv_sg8(o, M, N)) {
      qmv_sg8(o, N, K);
      return;
    }
    if (M >= 2 && use_qmv_wide(d)) {
      qmv_wide(o, M, N, K);
      return;
    }
    qmv(o, M, N, K);
    return;
  }

  if (K < 1024) {
    qvm(o, M, N, K);
    return;
  }
  qvm_split_k(o, M, N, K);
}

void KQuantGatherQMM::eval_gpu(const std::vector<array> &inputs, array &out) {
  auto &s = stream();
  auto &d = metal::device(s.device);
  out.set_data(allocator::malloc(out.nbytes()));

  array x = ensure_row_contiguous_matrix(inputs[0], s);
  array w = ensure_row_contiguous_matrix(inputs[1], s);
  array scales = ensure_row_contiguous_matrix(inputs[2], s);
  array biases = ensure_row_contiguous_matrix(inputs[3], s);
  const array &lhs_indices = inputs[4];
  const array &rhs_indices = inputs[5];

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;

  // x and w are both walked in order, so the matmuls batch up and reuse the
  // loads of each expert's weights.
  if (M == 1 && B >= 16 && right_sorted_ && B / E >= 4) {
    gather_qmm_rhs(x, w, scales, biases, rhs_indices, out, transpose_, mode_,
                   x.size() / K, N, K, d, s);
    return;
  }

  Operands o{x, w, scales, biases, out, group_size_, bits_, mode_, d, s};
  if (M >= vector_limit) {
    gather_qmm(o, lhs_indices, rhs_indices, transpose_, M, N, K);
    return;
  }
  if (transpose_) {
    gather_qmv(o, lhs_indices, rhs_indices, M, N, K);
    return;
  }
  gather_qvm(o, lhs_indices, rhs_indices, M, N, K);
}

void KQuantDequantize::eval_gpu(const std::vector<array> &inputs, array &out) {
  auto &s = stream();
  auto &d = metal::device(s.device);
  out.set_data(allocator::malloc(out.nbytes()));
  auto &enc = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(inputs[0], s);
  auto scales = ensure_row_contiguous(inputs[1], s);
  auto biases = ensure_row_contiguous(inputs[2], s);

  auto kernel =
      get_kernel(d, "dequantize", kernels::dequantize(mode_, out.dtype()));
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, w, scales, biases);
  enc.set_output_array(out, 3);

  constexpr int uint8_per_uint32 = 4;
  // Values one thread decodes: a pack of the affine stream, or an 8-value
  // chunk of a grid unit (kquant_dequantize's KQ_GRID arm).
  int packs_per_int = is_grid(mode_)             ? 8
                      : (bits_ == 3 || bits_ == 5) ? 8
                      : bits_ == 6               ? 4
                                                 : 8 / bits_;
  size_t nthreads = out.size() / packs_per_int;
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() *= uint8_per_uint32;
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  enc.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core::kquant

#else

#include <stdexcept>

namespace mlx::core::kquant {

void KQuantMatmul::eval_gpu(const std::vector<array> &, array &) {
  throw std::runtime_error(
      "[quantized_matmul] K-quant modes are not implemented on this GPU "
      "backend.");
}

void KQuantGatherQMM::eval_gpu(const std::vector<array> &, array &) {
  throw std::runtime_error(
      "[gather_qmm] K-quant modes are not implemented on this GPU backend.");
}

void KQuantDequantize::eval_gpu(const std::vector<array> &, array &) {
  throw std::runtime_error(
      "[dequantize] K-quant modes are not implemented on this GPU backend.");
}

} // namespace mlx::core::kquant

#endif
