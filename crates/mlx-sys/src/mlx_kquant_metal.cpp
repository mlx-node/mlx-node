// Metal dispatch for the K-quant primitives. The kernel choice, tile sizes and
// launch geometry mirror MLX's affine dispatcher in
// mlx/backend/metal/quantized.cpp; the kernels are JIT-built from
// metal/kquant/{kquant,kquant_nax}.h.

#include "mlx_kquant.h"
#include "mlx_test_counters.h"

#ifdef MLX_NODE_METAL_ENABLED

#include "metal/common/quantized.h"
#include "mlx/allocator.h"
#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/reduce.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

#include <climits>
#include <cstdlib>
#include <sstream>

namespace mlx::core::quantized_preamble {
const char *utils();
}

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

// Same text as mlx::core::get_template_definition (bools stream as 0/1).
template <typename... Args>
std::string template_definition(const std::string &name,
                                const std::string &func, Args... args) {
  std::ostringstream s;
  s << func << "<";
  bool first = true;
  auto add_arg = [&s, &first](const auto &arg) {
    if (!first) {
      s << ", ";
    }
    first = false;
    s << arg;
  };
  (add_arg(args), ...);
  s << ">";
  return "\ntemplate [[host_name(\"" + name + "\")]] [[kernel]] decltype(" +
         s.str() + ") " + s.str() + ";\n";
}

MTL::ComputePipelineState *
build_kernel(metal::Device &d, const std::string &kname,
             const std::string &template_def, bool nax,
             const std::string &hash_name = "",
             const metal::MTLFCList &func_consts = {}) {
  auto *lib = d.get_library("mlx_node_" + kname, [&] {
    std::string source = quantized_preamble::utils();
    if (nax) {
      source += quantized_preamble::gemm_nax();
      source += quantized_preamble::quantized_utils();
      source += quantized_preamble::kquant_nax();
    } else {
      source += quantized_preamble::gemm();
      source += quantized_preamble::quantized_utils();
      source += quantized_preamble::kquant();
    }
    return source + template_def;
  });
  return d.get_kernel(kname, lib, hash_name, func_consts);
}

// `kquant_<func><T, group_size, bits, super_ratio, has_min, args...>`.
template <typename... Args>
MTL::ComputePipelineState *
get_kernel(metal::Device &d, const std::string &kname, const std::string &func,
           Mode mode, const std::string &type, int group_size, int bits,
           Args... args) {
  bridge_testing::record(func);
  return build_kernel(d, kname,
                      template_definition(kname, "kquant_" + func, type,
                                          group_size, bits, super_ratio(mode),
                                          has_sub_min(mode), args...),
                      false);
}

template <typename... Args>
MTL::ComputePipelineState *
get_nax_kernel(metal::Device &d, const std::string &kname,
               const std::string &func, Mode mode, const std::string &type,
               int group_size, int bits, Args... args) {
  bridge_testing::record(func);
  return build_kernel(d, kname,
                      template_definition(kname, "kquant_" + func, type,
                                          group_size, bits, super_ratio(mode),
                                          has_sub_min(mode), args...),
                      true);
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
};

void qmv(const Operands &o, int M, int N, int K) {
  int B = o.out.size() / M / N;
  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string type = type_string(o.x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;
  std::string kname;
  concatenate(kname, mode_name(o.mode), fast ? "_qmv_fast_" : "_qmv_", type,
              "_gs_", o.group_size, "_b_", o.bits,
              B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_kernel(o.d, kname, fast ? "qmv_fast" : "qmv", o.mode, type,
                           o.group_size, o.bits, B > 1);
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

void qmv_wide(const Operands &o, int M, int N, int K) {
  // Each tile re-reads the weights: fewest tiles, then the smallest tile that
  // fills them. Up to 8 vectors per tile only when N is large enough that the
  // row dimension already saturates the GPU.
  const int tile_cap = N >= 2048 ? 8 : 5;
  int n_tiles = (M + tile_cap - 1) / tile_cap;
  int vecs_per_tg = (M + n_tiles - 1) / n_tiles;
  // The K-quant qmv_wide is instantiated at k_lanes 8 only.
  constexpr int k_lanes = 8;
  constexpr int num_simdgroups = 2;
  int B = o.out.size() / M / N;
  bool batched = B > 1;
  int rows_per_tg = (32 / k_lanes) * num_simdgroups;

  MTL::Size group_dims(32, num_simdgroups, 1);
  MTL::Size grid_dims((M + vecs_per_tg - 1) / vecs_per_tg,
                      (N + rows_per_tg - 1) / rows_per_tg, B);

  std::string type = type_string(o.x.dtype());
  std::string kname;
  concatenate(kname, mode_name(o.mode), "_qmv_wide_", type, "_gs_",
              o.group_size, "_b_", o.bits, "_nv_", vecs_per_tg, "_kl_", k_lanes,
              batched ? "_batch_1" : "_batch_0");
  if (bridge_testing::counting) {
    bridge_testing::record("qmv_wide_nv" + std::to_string(vecs_per_tg));
  }
  auto kernel = get_kernel(o.d, kname, "qmv_wide", o.mode, type, o.group_size,
                           o.bits, vecs_per_tg, k_lanes, batched);
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
  if (o.mode != Mode::Q4K && o.mode != Mode::Q5K && o.mode != Mode::Q6K &&
      o.mode != Mode::IQ4XS) {
    return false;
  }
  return o.x.dtype() == bfloat16 && o.out.size() == size_t(8) * N &&
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

  std::string type = type_string(o.x.dtype());
  std::string prep_name;
  concatenate(prep_name, "kquant_qmv_sg8_prep_", type, "_gs_", o.group_size);
  auto prep = build_kernel(
      o.d, prep_name,
      template_definition(prep_name, "kquant_qmv_sg8_prep", type, o.group_size),
      false);
  enc.set_compute_pipeline_state(prep);
  enc.set_input_array(o.x, 0);
  enc.set_output_array(bt, 1);
  enc.set_output_array(sums, 2);
  enc.set_bytes(K, 3);
  enc.dispatch_threadgroups(MTL::Size((K / 32 + 3) / 4, 1, 1),
                            MTL::Size(128, 1, 1));

  std::string kname;
  concatenate(kname, mode_name(o.mode), "_qmv_sg8_", type, "_gs_", o.group_size,
              "_b_", o.bits);
  auto kernel =
      get_kernel(o.d, kname, "qmv_sg8", o.mode, type, o.group_size, o.bits);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, o.w, o.scales, o.biases);
  enc.set_input_array(bt, 3);
  enc.set_input_array(sums, 4);
  enc.set_output_array(o.out, 5);
  enc.set_bytes(K, 6);
  enc.set_bytes(N, 7);
  enc.dispatch_threadgroups(MTL::Size(N / 32, 1, 1), MTL::Size(128, 1, 1));
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

  int split_k = K > 8192 ? 32 : 8;
  int split_D = (K + split_k - 1) / split_k;
  int B = o.out.size() / M / N;
  B *= split_k;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  // This grid drops any N % bn tail. Untransposed, N is the packed axis and
  // whole super-blocks, a multiple of 256 for every mode but IQ4_NL (32).
  int bn = qvm_columns_per_simdgroup * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, N / bn, B);

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

  std::string type = type_string(x.dtype());
  std::string kname;
  concatenate(kname, mode_name(o.mode), "_qvm_split_k_", type, "_gs_",
              o.group_size, "_b_", o.bits, "_spk_", split_k);
  auto kernel = get_kernel(o.d, kname, "qvm_split_k", o.mode, type,
                           o.group_size, o.bits, split_k);
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

  std::string type = type_string(o.x.dtype());
  std::string kname;
  concatenate(kname, mode_name(o.mode), "_qvm_", type, "_gs_", o.group_size,
              "_b_", o.bits, B > 1 ? "_batch_1" : "_batch_0");
  auto kernel =
      get_kernel(o.d, kname, "qvm", o.mode, type, o.group_size, o.bits, B > 1);
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
  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  int bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  bool aligned = N % 64 == 0;
  bool batched = B > 1;
  std::string type = type_string(o.x.dtype());
  std::string kname;
  concatenate(kname, mode_name(o.mode), "_qmm_t_nax_", type, "_gs_",
              o.group_size, "_b_", o.bits, "_bm", bm, "_bn", bn, "_bk", bk,
              "_wm", wm, "_wn", wn, aligned ? "_alN_true" : "_alN_false",
              batched ? "_batch_1" : "_batch_0");
  auto kernel =
      get_nax_kernel(o.d, kname, "qmm_t_nax", o.mode, type, o.group_size,
                     o.bits, aligned, batched, bm, bk, bn, wm, wn);
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
  std::string type = type_string(o.x.dtype());
  std::string kname;
  concatenate(kname, mode_name(o.mode), transpose ? "_qmm_t_" : "_qmm_n_", type,
              "_gs_", o.group_size, "_b_", o.bits,
              transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
              batched ? "_batch_1" : "_batch_0");
  auto kernel = transpose ? get_kernel(o.d, kname, "qmm_t", o.mode, type,
                                       o.group_size, o.bits, aligned, batched)
                          : get_kernel(o.d, kname, "qmm_n", o.mode, type,
                                       o.group_size, o.bits, batched);
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
  std::string type = type_string(o.x.dtype());
  std::string kname;
  concatenate(kname, mode_name(o.mode), "_qmm_t_splitk_", type, "_gs_",
              o.group_size, "_b_", o.bits,
              aligned ? "_alN_true" : "_alN_false");
  auto kernel = get_kernel(o.d, kname, "qmm_t_splitk", o.mode, type,
                           o.group_size, o.bits, aligned);
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
  std::string type = type_string(o.x.dtype());
  std::string kname;
  concatenate(kname, mode_name(o.mode),
              transpose ? "_gather_qmm_t_" : "_gather_qmm_n_", type, "_gs_",
              o.group_size, "_b_", o.bits,
              transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  auto kernel = transpose ? get_kernel(o.d, kname, "gather_qmm_t", o.mode, type,
                                       o.group_size, o.bits, aligned)
                          : get_kernel(o.d, kname, "gather_qmm_n", o.mode, type,
                                       o.group_size, o.bits);
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

  std::string type = type_string(o.x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;
  std::string kname;
  concatenate(kname, mode_name(o.mode),
              fast ? "_gather_qmv_fast_" : "_gather_qmv_", type, "_gs_",
              o.group_size, "_b_", o.bits);
  auto kernel = get_kernel(o.d, kname, fast ? "gather_qmv_fast" : "gather_qmv",
                           o.mode, type, o.group_size, o.bits);
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

  std::string type = type_string(o.x.dtype());
  std::string kname;
  concatenate(kname, mode_name(o.mode), "_gather_qvm_", type, "_gs_",
              o.group_size, "_b_", o.bits);
  auto kernel =
      get_kernel(o.d, kname, "gather_qvm", o.mode, type, o.group_size, o.bits);
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
                    bool transpose, int group_size, int bits, Mode mode, int M,
                    int N, int K, metal::Device &d, const Stream &s) {
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

  int bm = 16, bn = 32, bk = 32;
  int wm = 1, wn = 2;
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  std::string type = type_string(x.dtype());
  std::string kname;
  concatenate(kname, mode_name(mode),
              transpose ? "_gather_qmm_rhs_nt_" : "_gather_qmm_rhs_nn_", type,
              "_gs_", group_size, "_b_", bits, "_bm_", bm, "_bn_", bn, "_bk_",
              bk, "_wm_", wm, "_wn_", wn);
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
  auto kernel = build_kernel(
      d, kname,
      template_definition(kname, "kquant_gather_qmm_rhs", type, group_size,
                          bits, super_ratio(mode), has_sub_min(mode), bm, bn,
                          bk, wm, wn, transpose),
      false, hash_name, func_consts);
  enc.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

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

  array x = ensure_row_contiguous_matrix(inputs[0], s);
  array w = ensure_row_contiguous_matrix(inputs[1], s);
  array scales = ensure_row_contiguous_matrix(inputs[2], s);
  array biases = ensure_row_contiguous_matrix(inputs[3], s);

  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);
  Operands o{x, w, scales, biases, out, group_size_, bits_, mode_, d, s};

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
    gather_qmm_rhs(x, w, scales, biases, rhs_indices, out, transpose_,
                   group_size_, bits_, mode_, x.size() / K, N, K, d, s);
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

  std::string type = type_string(out.dtype());
  std::string kname;
  concatenate(kname, mode_name(mode_), "_dequantize_", type, "_gs_",
              group_size_, "_b_", bits_);
  auto kernel =
      get_kernel(d, kname, "dequantize", mode_, type, group_size_, bits_);
  enc.set_compute_pipeline_state(kernel);
  set_weights(enc, w, scales, biases);
  enc.set_output_array(out, 3);

  constexpr int uint8_per_uint32 = 4;
  int packs_per_int = (bits_ == 3 || bits_ == 5) ? 8
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
