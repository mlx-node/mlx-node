// GPU repack of GGUF raw blocks into the runtime packed arrays. One
// `GgufRepackBand` / `GgufRepackBandCuda` primitive writes one band of rows of
// one tensor from a uint8 staging buffer into caller-allocated destination
// arrays: the outputs alias the destination buffers (`copy_shared_buffer`), so
// bands of one tensor chain through the primitive and the final outputs are
// the tensor arrays.

#include "mlx_common.h"
#include "mlx_gguf_repack.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx_node::gguf_repack {

mlx::core::Dtype scale_dtype(const KernelInfo& info) {
  switch (info.scales) {
    case ScaleType::I8:
      return mlx::core::int8;
    case ScaleType::F16:
      return mlx::core::float16;
    case ScaleType::U8:
      break;
  }
  return mlx::core::uint8;
}

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument(what);
}

// An evaluated array must be row-contiguous with no broadcast; a lazy one is
// checked again in eval_gpu.
void expect_dense(const mlx::core::array& a, const char* what) {
  if (a.status() == mlx::core::array::Status::available) {
    expect(a.flags().row_contiguous && a.data_size() == a.size(),
           std::string(what) + " is not a dense row-contiguous array");
  }
}

// Checks one destination array: dtype, last dim `cols`, and `rows * cols`
// elements in all. Returns rows.
uint64_t dst_rows(const mlx::core::array& a,
                  mlx::core::Dtype dtype,
                  uint64_t cols,
                  const char* what) {
  expect(a.dtype() == dtype, std::string(what) + " has the wrong dtype");
  expect(a.ndim() >= 1 && uint64_t(a.shape(-1)) == cols,
         std::string(what) + " last dimension must be " +
             std::to_string(cols));
  expect_dense(a, what);
  return uint64_t(a.size()) / cols;
}

BandPlan plan_band(int kernel,
                   uint32_t k,
                   uint32_t row0,
                   uint32_t rows,
                   uint64_t src_offset) {
  expect(kernel >= 0 && kernel < kKernelCount,
         "unknown GGUF repack kernel id " + std::to_string(kernel));
  const auto& info = kKernels[kernel];
  expect(k > 0 && k % info.block_values == 0,
         std::string(info.name) + ": K must be a positive multiple of " +
             std::to_string(info.block_values) + ", got " + std::to_string(k));
  expect(rows > 0, "band has no rows");
  expect(src_offset % info.offset_rule == 0,
         std::string(info.name) + ": staging offset " +
             std::to_string(src_offset) + " is not a multiple of " +
             std::to_string(info.offset_rule));
  const uint32_t kb = k / info.block_values;
  const uint64_t nb = uint64_t(rows) * kb;
  expect(nb * info.threads_per_blk <= std::numeric_limits<uint32_t>::max(),
         "band too large for one dispatch");
  if (info.t64) {
    expect(row0 % kTileRows == 0 && rows % kTileRows == 0,
           "Tiled64 band rows must start and end on 64-row tiles");
  }
  // Block byte offsets inside the kernels are 32-bit.
  expect(src_offset % 4 + nb * info.block_bytes <=
             std::numeric_limits<uint32_t>::max(),
         "band bytes too large for one dispatch");
  BandPlan plan{};
  plan.params.src_off = src_offset;
  plan.params.nb = uint32_t(nb);
  plan.params.kb = kb;
  plan.params.ku = kb * (info.block_values / kValuesPerUnit);
  plan.params.row0 = row0;
  plan.params.rows = rows;
  plan.params.tile0 = row0 / kTileRows;
  plan.src_bytes = nb * info.block_bytes;
  return plan;
}

}  // namespace mlx_node::gguf_repack

namespace {

void write_error(char* err, size_t cap, const std::string& msg) {
  if (!err || cap == 0) return;
  const size_t n = std::min(msg.size(), cap - 1);
  std::memcpy(err, msg.data(), n);
  err[n] = '\0';
}

}  // namespace

extern "C" {

// Whether the GPU repack can run: Metal is available and every kernel of the
// table is in paged_attn.metallib, or CUDA is available for supported kernels.
bool mlx_gguf_repack_available() {
  return mlx_node::gguf_repack::metal_repack_available() ||
      mlx_node::gguf_repack::cuda_repack_available();
}

// Repack rows [row0, row0 + rows) of a tensor of row length `k` from
// `staging` (uint8, the band's GGUF blocks start at byte `src_offset`) into
// `dst_w` / `dst_s` / `dst_b` (the whole tensor's arrays, row-major or
// Tiled64 per `kernel`). `*out_w` / `*out_s` / `*out_b` receive lazy arrays
// sharing the destination buffers; the caller adopts them in place of the
// destination handles. Kernels without `.biases` (Q4_0, Q8_0, MXFP4) take null
// `dst_b` and leave `*out_b` null (`out_b` may then be null). Returns 0, or
// nonzero with a message in `err`.
int32_t mlx_gguf_repack_band(mlx_array* staging, uint64_t src_offset,
                             int32_t kernel, uint32_t k, uint32_t row0,
                             uint32_t rows, mlx_array* dst_w,
                             mlx_array* dst_s, mlx_array* dst_b,
                             mlx_array** out_w, mlx_array** out_s,
                             mlx_array** out_b, char* err, size_t err_cap) {
  if (out_w) *out_w = nullptr;
  if (out_s) *out_s = nullptr;
  if (out_b) *out_b = nullptr;
  write_error(err, err_cap, "");
#if defined(MLX_NODE_METAL_ENABLED) || defined(MLX_NODE_CUDA_ENABLED)
  try {
    using namespace mlx_node::gguf_repack;
    using mlx::core::array;
    using mlx::core::Device;
    using mlx::core::Dtype;
    using mlx::core::Shape;

    expect(staging && dst_w && dst_s && out_w && out_s, "null argument");
#ifdef MLX_NODE_METAL_ENABLED
    metal_repack_validate(kernel);
#else
    cuda_repack_validate(kernel);
#endif
    const BandPlan plan = plan_band(kernel, k, row0, rows, src_offset);
    const auto& info = kKernels[kernel];
    const bool has_biases = info.bias_bytes != 0;
    if (has_biases) {
      expect(dst_b && out_b, std::string(info.name) + " needs a biases array");
    } else {
      expect(!dst_b, std::string(info.name) + " has no biases array");
    }
    const auto& st = *reinterpret_cast<array*>(staging);
    expect(st.dtype() == mlx::core::uint8, "staging must be uint8");
    expect_dense(st, "staging");
    expect(st.offset() % 16 == 0, "staging array offset is not 16-byte aligned");
    expect(src_offset + plan.src_bytes <= uint64_t(st.nbytes()),
           "band bytes run past the end of the staging array");

    const auto& w = *reinterpret_cast<array*>(dst_w);
    const auto& s = *reinterpret_cast<array*>(dst_s);
    const uint64_t kb = plan.params.kb;
    const uint64_t total =
        dst_rows(w, mlx::core::uint32, kb * info.weight_words, "weight");
    const Dtype sdt = scale_dtype(info);
    expect(dst_rows(s, sdt, kb * info.scale_bytes / mlx::core::size_of(sdt),
                    "scales") == total,
           "scales rows differ from weight rows");
    if (has_biases) {
      const auto& b = *reinterpret_cast<array*>(dst_b);
      expect(dst_rows(b, mlx::core::float16, kb * info.bias_bytes / 2,
                      "biases") == total,
             "biases rows differ from weight rows");
    }
    expect(uint64_t(row0) + rows <= total, "band runs past the last row");
    if (info.t64) {
      expect(total % kTileRows == 0, "Tiled64 tensor rows must be a multiple of 64");
    }

    auto stream = mlx::core::default_stream(Device::gpu);
    std::vector<Shape> shapes{w.shape(), s.shape()};
    std::vector<Dtype> dtypes{w.dtype(), s.dtype()};
    std::vector<array> inputs{st, w, s};
    if (has_biases) {
      const auto& b = *reinterpret_cast<array*>(dst_b);
      shapes.push_back(b.shape());
      dtypes.push_back(b.dtype());
      inputs.push_back(b);
    }
#ifdef MLX_NODE_METAL_ENABLED
    auto outputs = metal_repack_band(
        stream, kernel, plan.params, std::move(shapes), dtypes,
        std::move(inputs));
#else
    auto outputs = cuda_repack_band(
        stream, kernel, plan.params, std::move(shapes), dtypes,
        std::move(inputs));
#endif
    *out_w = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_s = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    if (has_biases) {
      *out_b = reinterpret_cast<mlx_array*>(new array(std::move(outputs[2])));
    }
    return 0;
  } catch (const std::exception& e) {
    write_error(err, err_cap, std::string("mlx_gguf_repack_band: ") + e.what());
    return 1;
  }
#else
  (void)staging; (void)src_offset; (void)kernel; (void)k; (void)row0;
  (void)rows; (void)dst_w; (void)dst_s; (void)dst_b;
  write_error(err, err_cap, "mlx_gguf_repack_band: GPU repack is unavailable (no GPU backend)");
  return 1;
#endif
}

// The kernel table's names, '\n'-joined, into `out` (NUL-terminated when it
// fits). Returns the joined length without the NUL; no Metal needed.
size_t mlx_gguf_repack_kernel_names(char* out, size_t cap) {
  std::string joined;
  for (int i = 0; i < mlx_node::gguf_repack::kKernelCount; ++i) {
    if (i) joined += '\n';
    joined += mlx_node::gguf_repack::kKernels[i].name;
  }
  if (out && cap > joined.size()) {
    std::memcpy(out, joined.c_str(), joined.size() + 1);
  }
  return joined.size();
}

}  // extern "C"
