// GPU repack of GGUF raw blocks into the K-quant runtime arrays. One
// `GgufRepackBand` primitive writes one band of rows of one tensor from a
// uint8 staging buffer into caller-allocated destination arrays: the outputs
// alias the destination buffers (`copy_shared_buffer`), so bands of one tensor
// chain through the primitive and the final outputs are the tensor arrays.
// The kernels come only from the prebuilt paged_attn.metallib (no JIT).

#include "mlx_common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MLX_NODE_METAL_ENABLED
#include <mutex>
#include <set>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/fast_primitives.h"
#include "mlx_paged_metallib.h"
#endif

namespace {

// Indexed by the kernel id the Rust side passes (`GpuRepackKind` x layout).
// Each id must stay at its index; new kernels are appended.
struct KernelInfo {
  const char* name;
  uint32_t block_bytes;     // GGUF bytes per 256-value block
  uint32_t weight_words;    // .weight uint32 words per block (8 * bits)
  uint32_t scale_bytes;     // .scales bytes per block
  uint32_t bias_bytes;      // .biases bytes per block
  uint32_t offset_rule;     // staging byte offset of a band must be a multiple
  uint32_t threads_per_blk; // grid = blocks * threads_per_blk
  bool t64;
};

constexpr KernelInfo kKernels[] = {
    {"mlx_node_gguf_repack_q4k", 144, 32, 16, 4, 16, 8, false},
    {"mlx_node_gguf_repack_q4k_t64", 144, 32, 16, 4, 16, 8, true},
    {"mlx_node_gguf_repack_q5k", 176, 40, 16, 4, 16, 8, false},
    {"mlx_node_gguf_repack_q5k_t64", 176, 40, 16, 4, 16, 8, true},
};
constexpr int kKernelCount = int(sizeof(kKernels) / sizeof(kKernels[0]));

void write_error(char* err, size_t cap, const std::string& msg) {
  if (!err || cap == 0) return;
  const size_t n = std::min(msg.size(), cap - 1);
  std::memcpy(err, msg.data(), n);
  err[n] = '\0';
}

#ifdef MLX_NODE_METAL_ENABLED
using namespace mlx::core;

constexpr uint32_t kValuesPerBlock = 256;
constexpr uint32_t kTileRows = 64;
constexpr uint32_t kThreadgroup = 256;

// Mirrors `GgufRepackParams` in gguf_repack.metal.
struct GgufRepackParams {
  uint64_t src_off;
  uint32_t nb;
  uint32_t kb;
  uint32_t ku;
  uint32_t row0;
  uint32_t rows;
  uint32_t tile0;
};
static_assert(sizeof(GgufRepackParams) == 32, "GgufRepackParams layout");

MTL::ComputePipelineState* repack_kernel(metal::Device& device, int kernel) {
  return fast::paged::get_prebuilt_kernel(device, "gguf_repack", "GGUF repack",
                                          kKernels[kernel].name);
}

// Encodes one band. `outs` are the destination arrays (weight, scales[,
// biases]) with their buffers set.
void encode_band(metal::Device& device, metal::CommandEncoder& encoder,
                 int kernel, const array& staging, std::vector<array>& outs,
                 const GgufRepackParams& p) {
  encoder.set_compute_pipeline_state(repack_kernel(device, kernel));
  encoder.set_input_array(staging, 0);
  for (size_t i = 0; i < outs.size(); ++i) {
    encoder.set_output_array(outs[i], int(1 + i));
  }
  encoder.set_bytes(p, 4);
  const uint64_t threads =
      uint64_t(p.nb) * kKernels[kernel].threads_per_blk;
  encoder.dispatch_threads(MTL::Size(threads, 1, 1),
                           MTL::Size(kThreadgroup, 1, 1));
}

// Inputs: staging, then the destination arrays. Outputs alias the
// destination buffers.
class GgufRepackBand final : public fast::Custom {
 public:
  GgufRepackBand(Stream stream, int kernel, GgufRepackParams params)
      : Custom(stream,
               [](std::vector<array>) -> std::vector<array> {
                 throw std::runtime_error("GgufRepackBand has no fallback");
               }),
        kernel_(kernel), params_(params) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GgufRepackBand requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto& dst = inputs[1 + i];
      if (!dst.flags().row_contiguous || dst.data_size() != dst.size()) {
        throw std::runtime_error(
            "GgufRepackBand: destination array is not row-contiguous");
      }
      outputs[i].copy_shared_buffer(dst);
    }
    if (!inputs[0].flags().row_contiguous) {
      throw std::runtime_error(
          "GgufRepackBand: staging array is not row-contiguous");
    }
    auto& s = this->stream();
    auto& device = metal::device(s.device);
    auto& encoder = metal::get_command_encoder(s);
    encode_band(device, encoder, kernel_, inputs[0], outputs, params_);
  }

  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&,
                         const std::vector<array>&) override {
    throw std::runtime_error("GgufRepackBand is inference-only");
  }

  DEFINE_NAME(GgufRepackBand)

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const GgufRepackBand&>(other);
    return kernel_ == o.kernel_ &&
        std::memcmp(&params_, &o.params_, sizeof(params_)) == 0;
  }

 private:
  int kernel_;
  GgufRepackParams params_;
};

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument(what);
}

// An evaluated array must be row-contiguous with no broadcast; a lazy one is
// checked again in eval_gpu.
void expect_dense(const array& a, const char* what) {
  if (a.status() == array::Status::available) {
    expect(a.flags().row_contiguous && a.data_size() == a.size(),
           std::string(what) + " is not a dense row-contiguous array");
  }
}

// Checks one destination array: dtype, last dim `cols`, and `rows * cols`
// elements in all. Returns rows.
uint64_t dst_rows(const array& a, Dtype dtype, uint64_t cols,
                  const char* what) {
  expect(a.dtype() == dtype, std::string(what) + " has the wrong dtype");
  expect(a.ndim() >= 1 && uint64_t(a.shape(-1)) == cols,
         std::string(what) + " last dimension must be " +
             std::to_string(cols));
  expect_dense(a, what);
  return uint64_t(a.size()) / cols;
}

struct BandPlan {
  GgufRepackParams params;
  uint64_t src_bytes;
};

BandPlan plan_band(int kernel, uint32_t k, uint32_t row0, uint32_t rows,
                   uint64_t src_offset) {
  expect(kernel >= 0 && kernel < kKernelCount,
         "unknown GGUF repack kernel id " + std::to_string(kernel));
  const auto& info = kKernels[kernel];
  expect(k > 0 && k % kValuesPerBlock == 0,
         "K must be a positive multiple of 256, got " + std::to_string(k));
  expect(rows > 0, "band has no rows");
  expect(src_offset % info.offset_rule == 0,
         std::string(info.name) + ": staging offset " +
             std::to_string(src_offset) + " is not a multiple of " +
             std::to_string(info.offset_rule));
  const uint32_t kb = k / kValuesPerBlock;
  const uint64_t nb = uint64_t(rows) * kb;
  expect(nb * info.threads_per_blk <= std::numeric_limits<uint32_t>::max(),
         "band too large for one dispatch");
  if (info.t64) {
    expect(row0 % kTileRows == 0 && rows % kTileRows == 0,
           "Tiled64 band rows must start and end on 64-row tiles");
  }
  BandPlan plan{};
  plan.params.src_off = src_offset;
  plan.params.nb = uint32_t(nb);
  plan.params.kb = kb;
  plan.params.ku = kb * 8;
  plan.params.row0 = row0;
  plan.params.rows = rows;
  plan.params.tile0 = row0 / kTileRows;
  plan.src_bytes = nb * info.block_bytes;
  return plan;
}

// All kernel names in the table are functions of the loaded library.
bool table_in_library() {
  static std::once_flag once;
  static bool ok = false;
  std::call_once(once, [] {
    try {
      if (!metal::is_available()) return;
      auto* lib = fast::paged::get_paged_attn_library(metal::device(Device::gpu));
      std::set<std::string> names;
      NS::Array* functions = lib->functionNames();
      for (NS::UInteger i = 0; i < functions->count(); ++i) {
        names.insert(functions->object<NS::String>(i)->utf8String());
      }
      for (const auto& info : kKernels) {
        if (!names.count(info.name)) return;
      }
      ok = true;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "mlx_gguf_repack_available: %s\n", e.what());
    }
  });
  return ok;
}

array alloc_array(Shape shape, Dtype dtype) {
  size_t bytes = size_of(dtype);
  for (auto d : shape) bytes *= size_t(d);
  return array(allocator::malloc(bytes), std::move(shape), dtype);
}
#endif

}  // namespace

extern "C" {

// Whether the GPU repack can run: Metal is available and every kernel of the
// table is in paged_attn.metallib.
bool mlx_gguf_repack_available() {
#ifdef MLX_NODE_METAL_ENABLED
  return table_in_library();
#else
  return false;
#endif
}

// Repack rows [row0, row0 + rows) of a tensor of row length `k` from
// `staging` (uint8, the band's GGUF blocks start at byte `src_offset`) into
// `dst_w` / `dst_s` / `dst_b` (the whole tensor's arrays, row-major or
// Tiled64 per `kernel`). `*out_w` / `*out_s` / `*out_b` receive lazy arrays
// sharing the destination buffers; the caller adopts them in place of the
// destination handles. Returns 0, or nonzero with a message in `err`.
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
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(staging && dst_w && dst_s && dst_b && out_w && out_s && out_b,
           "null argument");
    expect(metal::is_available(), "requires the Metal device");
    const BandPlan plan = plan_band(kernel, k, row0, rows, src_offset);
    const auto& info = kKernels[kernel];
    const auto& st = *reinterpret_cast<array*>(staging);
    expect(st.dtype() == uint8, "staging must be uint8");
    expect_dense(st, "staging");
    expect(st.offset() % 16 == 0, "staging array offset is not 16-byte aligned");
    expect(src_offset + plan.src_bytes <= uint64_t(st.nbytes()),
           "band bytes run past the end of the staging array");

    const auto& w = *reinterpret_cast<array*>(dst_w);
    const auto& s = *reinterpret_cast<array*>(dst_s);
    const auto& b = *reinterpret_cast<array*>(dst_b);
    const uint64_t kb = plan.params.kb;
    const uint64_t total = dst_rows(w, uint32, kb * info.weight_words, "weight");
    expect(dst_rows(s, uint8, kb * info.scale_bytes, "scales") == total,
           "scales rows differ from weight rows");
    expect(dst_rows(b, float16, kb * info.bias_bytes / 2, "biases") == total,
           "biases rows differ from weight rows");
    expect(uint64_t(row0) + rows <= total, "band runs past the last row");
    if (info.t64) {
      expect(total % kTileRows == 0, "Tiled64 tensor rows must be a multiple of 64");
    }

    auto stream = default_stream(Device::gpu);
    auto primitive = std::make_shared<GgufRepackBand>(stream, kernel, plan.params);
    auto outputs = array::make_arrays({w.shape(), s.shape(), b.shape()},
                                      {w.dtype(), s.dtype(), b.dtype()},
                                      primitive, {st, w, s, b});
    *out_w = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    *out_s = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    *out_b = reinterpret_cast<mlx_array*>(new array(std::move(outputs[2])));
    return 0;
  } catch (const std::exception& e) {
    write_error(err, err_cap, std::string("mlx_gguf_repack_band: ") + e.what());
    return 1;
  }
#else
  (void)staging; (void)src_offset; (void)kernel; (void)k; (void)row0;
  (void)rows; (void)dst_w; (void)dst_s; (void)dst_b;
  write_error(err, err_cap, "mlx_gguf_repack_band: GPU repack is unavailable (no Metal)");
  return 1;
#endif
}

// TEST ONLY (gate 2): GPU seconds per dispatch of `kernel` over
// floor(src_bytes / row_bytes) rows of length `k` (Tiled64: rounded down to
// 64 rows). The staging bytes are written by the CPU just before; all
// `iters` dispatches go into one command buffer that holds nothing else.
// `fresh_outputs` gives each dispatch its own new destination buffers (first
// touch included); otherwise one set is reused. Negative on error.
double mlx_gguf_repack_bench_gpu_seconds(int32_t kernel, uint64_t src_bytes,
                                         uint32_t k, bool fresh_outputs,
                                         uint32_t iters) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    expect(metal::is_available(), "requires the Metal device");
    expect(kernel >= 0 && kernel < kKernelCount, "unknown kernel id");
    expect(iters > 0, "iters must be positive");
    expect(k > 0 && k % kValuesPerBlock == 0, "K must be a multiple of 256");
    const auto& info = kKernels[kernel];
    const uint64_t row_bytes = uint64_t(k / kValuesPerBlock) * info.block_bytes;
    uint64_t rows = src_bytes / row_bytes;
    if (info.t64) rows -= rows % kTileRows;
    expect(rows > 0 && rows <= std::numeric_limits<int32_t>::max(),
           "src_bytes holds no whole row (Tiled64: 64 rows)");
    const BandPlan plan = plan_band(kernel, k, 0, uint32_t(rows), 0);

    expect(plan.src_bytes <= uint64_t(std::numeric_limits<int32_t>::max()),
           "src_bytes too large for the bench");
    array staging = alloc_array({int32_t(plan.src_bytes)}, uint8);
    uint32_t x = 0x9E3779B9u;
    auto* bytes = staging.data<uint8_t>();
    for (uint64_t i = 0; i < plan.src_bytes; ++i) {
      x = x * 1664525u + 1013904223u;
      bytes[i] = uint8_t(x >> 24);
    }

    const uint64_t kb = plan.params.kb;
    auto make_outs = [&] {
      return std::vector<array>{
          alloc_array({int32_t(rows), int32_t(kb * info.weight_words)}, uint32),
          alloc_array({int32_t(rows), int32_t(kb * info.scale_bytes)}, uint8),
          alloc_array({int32_t(rows), int32_t(kb * info.bias_bytes / 2)}, float16)};
    };
    std::vector<std::vector<array>> outs;
    outs.push_back(make_outs());
    if (fresh_outputs) {
      for (uint32_t i = 1; i < iters; ++i) outs.push_back(make_outs());
    }

    auto stream = default_stream(Device::gpu);
    synchronize(stream);
    auto& device = metal::device(Device::gpu);
    auto& encoder = metal::get_command_encoder(stream);
    auto buffer = NS::RetainPtr(encoder.get_command_buffer());
    for (uint32_t i = 0; i < iters; ++i) {
      encode_band(device, encoder, kernel, staging,
                  outs[fresh_outputs ? i : 0], plan.params);
    }
    encoder.synchronize();
    return (buffer->GPUEndTime() - buffer->GPUStartTime()) / double(iters);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_gguf_repack_bench_gpu_seconds: %s\n", e.what());
    return -1.0;
  }
#else
  (void)kernel; (void)src_bytes; (void)k; (void)fresh_outputs; (void)iters;
  return -1.0;
#endif
}

// The kernel table's names, '\n'-joined, into `out` (NUL-terminated when it
// fits). Returns the joined length without the NUL; no Metal needed.
size_t mlx_gguf_repack_kernel_names(char* out, size_t cap) {
  std::string joined;
  for (int i = 0; i < kKernelCount; ++i) {
    if (i) joined += '\n';
    joined += kKernels[i].name;
  }
  if (out && cap > joined.size()) {
    std::memcpy(out, joined.c_str(), joined.size() + 1);
  }
  return joined.size();
}

// TEST ONLY: the kernel table against the `mlx_node_gguf_repack_` functions of
// the loaded paged_attn.metallib, both ways. Writes "missing <name>" /
// "unexpected <name>" lines (empty when they match) into `out`. Returns 0, or
// -1 without Metal, when the library does not load, or when `out` cannot hold
// the report.
int32_t mlx_test_gguf_repack_metallib_check(char* out, size_t cap) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!out || cap == 0 || !metal::is_available()) return -1;
    auto* lib = fast::paged::get_paged_attn_library(metal::device(Device::gpu));
    const std::string prefix = "mlx_node_gguf_repack_";
    std::set<std::string> in_library;
    NS::Array* functions = lib->functionNames();
    for (NS::UInteger i = 0; i < functions->count(); ++i) {
      std::string name = functions->object<NS::String>(i)->utf8String();
      if (name.rfind(prefix, 0) == 0) in_library.insert(std::move(name));
    }
    std::set<std::string> table;
    for (const auto& info : kKernels) table.insert(info.name);
    std::string report;
    for (const auto& name : table) {
      if (!in_library.count(name)) report += "missing " + name + "\n";
    }
    for (const auto& name : in_library) {
      if (!table.count(name)) report += "unexpected " + name + "\n";
    }
    if (report.size() + 1 > cap) return -1;
    std::memcpy(out, report.c_str(), report.size() + 1);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_test_gguf_repack_metallib_check: %s\n", e.what());
    return -1;
  }
#else
  (void)out; (void)cap;
  return -1;
#endif
}

}  // extern "C"
