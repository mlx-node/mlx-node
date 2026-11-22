#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef MLX_NODE_METAL_ENABLED
#include "metal/common/quantized.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"
#endif

namespace {

struct Hy4RingWriteParams {
  uint32_t s;
  uint32_t cols;
  uint32_t start_slot;
  uint32_t cap;
};

#ifdef MLX_NODE_METAL_ENABLED
MTL::ComputePipelineState *
hy4_ring_write_pipeline(mlx::core::metal::Device &device, const char *name) {
  auto *library = device.get_library("mlx_node_hy4_ring_write", [] {
    return std::string(R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Hy4RingWriteParams {
  uint s;
  uint cols;
  uint start_slot;
  uint cap;
};

[[kernel]] void hy4_ring_write_u64(
    const device uint2* src [[buffer(0)]],
    device uint2* dst [[buffer(1)]],
    constant Hy4RingWriteParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.cols || gid.y >= p.s) {
    return;
  }
  const uint slot = (p.start_slot + gid.y) % p.cap;
  dst[ulong(slot) * p.cols + gid.x] = src[ulong(gid.y) * p.cols + gid.x];
}

[[kernel]] void hy4_ring_write_u16(
    const device uint16_t* src [[buffer(0)]],
    device uint16_t* dst [[buffer(1)]],
    constant Hy4RingWriteParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.cols || gid.y >= p.s) {
    return;
  }
  const uint slot = (p.start_slot + gid.y) % p.cap;
  dst[ulong(slot) * p.cols + gid.x] = src[ulong(gid.y) * p.cols + gid.x];
}

[[kernel]] void hy4_ring_write_u8(
    const device uint8_t* src [[buffer(0)]],
    device uint8_t* dst [[buffer(1)]],
    constant Hy4RingWriteParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.cols || gid.y >= p.s) {
    return;
  }
  const uint slot = (p.start_slot + gid.y) % p.cap;
  dst[ulong(slot) * p.cols + gid.x] = src[ulong(gid.y) * p.cols + gid.x];
}
)METAL");
  });
  return device.get_kernel(name, library);
}
#endif

// Circular row write into a pre-allocated [cap, width] ring buffer.
// Donates `buf` in place only when `is_donatable(buf, out)` holds; otherwise
// allocates and copies `buf` first so MLX's immutable-array invariant holds.
class Hy4RingWrite final : public mlx::core::Primitive {
 public:
  Hy4RingWrite(Stream stream, uint32_t start_slot, uint32_t cap)
      : Primitive(stream), start_slot_(start_slot), cap_(cap) {}

  DEFINE_NAME(Hy4RingWrite)

  std::vector<Shape> output_shapes(const std::vector<array> &inputs) override {
    return {inputs[0].shape()};
  }

  bool is_equivalent(const Primitive &other) const override {
    const auto &o = static_cast<const Hy4RingWrite &>(other);
    return start_slot_ == o.start_slot_ && cap_ == o.cap_;
  }

  void eval_cpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    const auto &buf = inputs[0];
    const auto &src = inputs[1];
    auto &out = outputs[0];
    auto s = stream();
    const uint32_t s_rows = static_cast<uint32_t>(src.shape(0));
    const uint32_t width = static_cast<uint32_t>(buf.shape(1));
    const size_t row_bytes = static_cast<size_t>(width) * buf.itemsize();

    const bool buf_dense = buf.flags().row_contiguous &&
                           buf.size() == buf.data_size() && buf.offset() == 0;
    if (buf_dense && mlx::core::is_donatable(buf, out)) {
      out.copy_shared_buffer(buf);
    } else if (s_rows == cap_) {
      out.set_data(mlx::core::allocator::malloc(out.nbytes()));
    } else if (buf.data_size() == 1) {
      mlx::core::copy_cpu(buf, out, mlx::core::CopyType::Scalar, s);
    } else if (buf_dense) {
      mlx::core::copy_cpu(buf, out, mlx::core::CopyType::Vector, s);
    } else {
      mlx::core::copy_cpu(buf, out, mlx::core::CopyType::General, s);
    }

    auto &encoder = mlx::core::cpu::get_command_encoder(s);
    array src_c = src;
    if (!src.flags().row_contiguous) {
      src_c = mlx::core::contiguous_copy_cpu(src, s);
      encoder.add_temporary(src_c);
    }
    encoder.set_input_array(src_c);
    encoder.set_output_array(out);
    const uint8_t *src_ptr = src_c.data<uint8_t>();
    uint8_t *dst_ptr = out.data<uint8_t>();
    const uint32_t start_slot = start_slot_;
    const uint32_t cap = cap_;
    encoder.dispatch([src_ptr, dst_ptr, s_rows, row_bytes, start_slot, cap]() {
      for (uint32_t r = 0; r < s_rows; ++r) {
        const uint32_t slot = (start_slot + r) % cap;
        std::memcpy(dst_ptr + static_cast<size_t>(slot) * row_bytes,
                    src_ptr + static_cast<size_t>(r) * row_bytes, row_bytes);
      }
    });
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
#ifdef MLX_NODE_METAL_ENABLED
    const auto &buf = inputs[0];
    const auto &src = inputs[1];
    auto &out = outputs[0];
    auto s = stream();
    const uint32_t s_rows = static_cast<uint32_t>(src.shape(0));
    const uint32_t width = static_cast<uint32_t>(buf.shape(1));
    const size_t row_bytes = static_cast<size_t>(width) * buf.itemsize();

    const bool buf_dense = buf.flags().row_contiguous &&
                           buf.size() == buf.data_size() && buf.offset() == 0;
    if (buf_dense && mlx::core::is_donatable(buf, out)) {
      out.copy_shared_buffer(buf);
    } else if (s_rows == cap_) {
      out.set_data(mlx::core::allocator::malloc(out.nbytes()));
    } else if (buf.data_size() == 1) {
      mlx::core::copy_gpu(buf, out, mlx::core::CopyType::Scalar, s);
    } else if (buf_dense) {
      mlx::core::copy_gpu(buf, out, mlx::core::CopyType::Vector, s);
    } else {
      mlx::core::copy_gpu(buf, out, mlx::core::CopyType::General, s);
    }

    auto &device = mlx::core::metal::device(s.device);
    auto &enc = mlx::core::metal::get_command_encoder(s);
    array src_c = src;
    if (!src.flags().row_contiguous) {
      src_c = array(src.shape(), src.dtype(), nullptr, {});
      mlx::core::copy_gpu(src, src_c, mlx::core::CopyType::General, s);
      enc.add_temporary(src_c);
    }

    const char *kernel_name = "hy4_ring_write_u8";
    uint32_t cols = static_cast<uint32_t>(row_bytes);
    if (row_bytes % 8 == 0 && src_c.offset() % 8 == 0) {
      kernel_name = "hy4_ring_write_u64";
      cols = static_cast<uint32_t>(row_bytes / 8);
    } else if (row_bytes % 2 == 0 && src_c.offset() % 2 == 0) {
      kernel_name = "hy4_ring_write_u16";
      cols = static_cast<uint32_t>(row_bytes / 2);
    }

    Hy4RingWriteParams params{s_rows, cols, start_slot_, cap_};
    enc.set_compute_pipeline_state(hy4_ring_write_pipeline(device, kernel_name));
    enc.set_input_array(src_c, 0);
    enc.set_output_array(out, 1);
    enc.set_bytes(params, 2);
    const uint32_t tg_x = std::min<uint32_t>(cols, 256u);
    enc.dispatch_threads(MTL::Size(cols, s_rows, 1), MTL::Size(tg_x, 1, 1));
#else
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Hy4RingWrite requires Metal");
#endif
  }

 private:
  uint32_t start_slot_;
  uint32_t cap_;
};

} // namespace

#ifdef MLX_NODE_METAL_ENABLED
using mlx::core::Dtype;
namespace {
using mlx::core::quantized_preamble::nax_header;
using mlx::core::quantized_preamble::qmv_header;

enum Hy4KernelId : int {
  kKernelHcMixReduce = 0,
  kKernelHcPostIdentity = 1,
  kKernelBf16LmHead = 2,
  kKernelQ8Decode = 3,
  kKernelKqGateUp = 4,
  kKernelKqDownCombine = 5,
  kKernelKqDownWeighted = 6,
  kKernelMlaSink = 7,
  kKernelMlaSplitK = 8,
  kKernelMlaMerge = 9,
  kKernelQ8Prefill = 10,
  kKernelMlaPrefillNax = 11,
  kKernelKqPrefillIndirectGu = 12,
  kKernelKqSortedCombine = 13,
  kKernelRouterTopk = 14,
  kKernelGgufGateUp = 15,
  kKernelGgufDownProj = 16,
  kKernelGgufPrefillIndirectGu = 17,
  kKernelGgufPrefillIndirectDown = 18,
  kKernelMlaDecodeV2 = 19,
  kKernelHcMixReduceNorm = 20,
  kKernelPostNormHcPost = 21,
  kKernelHcBridge = 22,
  kKernelQ8GroupedK192 = 23,
  kKernelQ8GroupedK512 = 24,
  kKernelRouterMv = 25,
  kKernelQ8DualDownAdd = 26,
  kKernelIndexerScoreDecode = 27,
  kKernelIndexerScorePrefill = 28,
  kKernelIndexerTopkCells = 29,
  kKernelMlaPrefillSlidingNax = 30,
  kKernelIndexerScorePrefillNax = 31,
  kKernelMoePrefillSortTiles = 32,
  kKernelGgufDedupSlots = 33,
  kKernelAffine4GateUp = 34,
  kKernelAffine4DownCombine = 35,
  kKernelMlaPrepQKv = 36,
  kKernelQ8GroupedK192Rope = 37,
  kKernelKpoolCompress = 38,
  kKernelPartialRopeTail = 39,
  kKernelEngramNormRows = 40,
  kKernelEngramGateAdd = 41,
  kKernelMlaDecodeNax = 42,
  kKernelIndexerScoreDecodeNax = 43,
  kKernelQ8DualDownPrefill = 44,
  kKernelGgufQ8_0EmbedRows = 45,
  kKernelGgufDownReduce = 46,
  kKernelF32HeadWeights = 47,
  kKernelMtpEhNormConcat = 48,
  kKernelPostNormAddRmsNorm = 49,
  kKernelMoePrefillSortHist = 50,
  kKernelMoePrefillSortScanTiles = 51,
  kKernelMoePrefillSortScatter = 52,
  kKernelAffine4DenseDecode = 53,
};

struct CachedPrimKey {
  int kernel_id;
  int stream_index;
  int gx;
  int gy;
  int gz;
  int tx;
  int ty;
  int tz;
  uint64_t tmpl_key;
  uint32_t const_mask;

  bool operator==(const CachedPrimKey &o) const {
    return kernel_id == o.kernel_id && stream_index == o.stream_index &&
           gx == o.gx && gy == o.gy && gz == o.gz && tx == o.tx &&
           ty == o.ty && tz == o.tz && tmpl_key == o.tmpl_key &&
           const_mask == o.const_mask;
  }
};

struct CachedPrimKeyHash {
  size_t operator()(const CachedPrimKey &k) const noexcept {
    uint64_t h = uint64_t(uint32_t(k.kernel_id)) |
                 (uint64_t(uint32_t(k.stream_index)) << 32);
    auto mix = [&h](uint64_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(uint64_t(uint32_t(k.gx)) | (uint64_t(uint32_t(k.gy)) << 32));
    mix(uint64_t(uint32_t(k.gz)) | (uint64_t(uint32_t(k.tx)) << 32));
    mix(uint64_t(uint32_t(k.ty)) | (uint64_t(uint32_t(k.tz)) << 32));
    mix(k.tmpl_key);
    mix(uint64_t(k.const_mask));
    return size_t(h);
  }
};

// metal_kernel() switches buffer address space between `constant` (size < 8)
// and `device` (size >= 8), and scalar `&` (ndim == 0) vs pointer `*`.
uint32_t compute_const_mask(const std::vector<array> &inputs) {
  uint32_t mask = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].size() < 8) {
      mask |= (1u << i);
    }
    if (inputs[i].ndim() == 0) {
      mask |= (1u << (16u + i));
    }
  }
  return mask;
}

// Reusing CustomKernel primitives avoids metal_kernel()'s per-dispatch source
// concatenation and signature hashing.
template <typename MakeFn>
std::vector<array> dispatch_cached_kernel(
    int kernel_id, const Stream &stream, std::tuple<int, int, int> grid,
    std::tuple<int, int, int> tg, uint64_t tmpl_key,
    const std::vector<array> &inputs, const std::vector<Shape> &out_shapes,
    const std::vector<Dtype> &out_dtypes, MakeFn &&make_outputs) {
  const CachedPrimKey key{
      kernel_id,
      stream.index,
      std::get<0>(grid),
      std::get<1>(grid),
      std::get<2>(grid),
      std::get<0>(tg),
      std::get<1>(tg),
      std::get<2>(tg),
      tmpl_key,
      compute_const_mask(inputs),
  };
  thread_local std::unordered_map<CachedPrimKey,
                                  std::shared_ptr<mlx::core::Primitive>,
                                  CachedPrimKeyHash>
      cache;
  if (auto it = cache.find(key); it != cache.end()) {
    auto prim = it->second;
    if (out_shapes.size() == 1) {
      return {array(out_shapes[0], out_dtypes[0], std::move(prim), inputs)};
    }
    return array::make_arrays(out_shapes, out_dtypes, std::move(prim), inputs);
  }
  auto res = make_outputs();
  if (!res.empty() && res[0].has_primitive()) {
    if (cache.size() < 4096) {
      cache.emplace(key, res[0].primitive_ptr());
    }
  }
  return res;
}

uint32_t float_bits(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

array cached_f32_scalar(float v) {
  const uint32_t key = float_bits(v);
  thread_local std::unordered_map<uint32_t, array> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  array arr(&v, {1}, mlx::core::float32);
  if (cache.size() < 128) {
    cache.emplace(key, arr);
  }
  return arr;
}

struct Float3Key {
  uint32_t a;
  uint32_t b;
  uint32_t c;

  bool operator==(const Float3Key &o) const {
    return a == o.a && b == o.b && c == o.c;
  }
};

struct Float3KeyHash {
  size_t operator()(const Float3Key &k) const noexcept {
    uint64_t h = uint64_t(k.a) | (uint64_t(k.b) << 32);
    h ^= uint64_t(k.c) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return size_t(h);
  }
};

array cached_i32_pair(int a, int b) {
  const uint64_t key =
      (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));
  thread_local std::unordered_map<uint64_t, array> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const int32_t vals[2] = {int32_t(a), int32_t(b)};
  array arr(vals, {2}, mlx::core::int32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

array cached_u64_scalar(uint64_t v) {
  thread_local std::unordered_map<uint64_t, array> cache;
  if (auto it = cache.find(v); it != cache.end()) {
    return it->second;
  }
  array arr(&v, {1}, mlx::core::uint64);
  if (cache.size() < 256) {
    cache.emplace(v, arr);
  }
  return arr;
}

struct U64PairKey {
  uint64_t a;
  uint64_t b;

  bool operator==(const U64PairKey &o) const { return a == o.a && b == o.b; }
};

struct U64PairKeyHash {
  size_t operator()(const U64PairKey &k) const noexcept {
    uint64_t h = k.a;
    h ^= k.b + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return size_t(h);
  }
};

array cached_u64_pair(uint64_t a, uint64_t b) {
  const U64PairKey key{a, b};
  thread_local std::unordered_map<U64PairKey, array, U64PairKeyHash> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const uint64_t vals[2] = {a, b};
  array arr(vals, {2}, mlx::core::uint64);
  if (cache.size() < 256) {
    cache.emplace(key, arr);
  }
  return arr;
}

bool check_gguf_buf(const array &buf, uint64_t off, uint64_t need_bytes,
                    uint64_t align) {
  if (buf.dtype() != mlx::core::uint8 || need_bytes < 8 || (off % align) != 0) {
    return false;
  }
  const uint64_t total = static_cast<uint64_t>(buf.nbytes());
  return off <= total && (total - off) >= need_bytes;
}

bool prefill_nax_env_disabled() {
  static const bool disabled = [] {
    const char *v = std::getenv("MLX_HY4_PREFILL_NAX");
    return v && std::strcmp(v, "0") == 0;
  }();
  return disabled;
}

bool mla_v2_env_disabled() {
  static const bool disabled = [] {
    const char *v = std::getenv("MLX_HY4_MLA_V2");
    return v && std::strcmp(v, "0") == 0;
  }();
  return disabled;
}

bool mla_decode_nax_env_disabled() {
  static const bool disabled = [] {
    const char *v = std::getenv("MLX_HY4_MLA_DECODE_NAX");
    return v && std::strcmp(v, "0") == 0;
  }();
  return disabled;
}

bool indexer_decode_nax_env_disabled() {
  static const bool disabled = [] {
    const char *v = std::getenv("MLX_HY4_INDEXER_DECODE_NAX");
    return v && std::strcmp(v, "0") == 0;
  }();
  return disabled;
}

bool gguf_gu_hilo_env() {
  static const bool enabled = [] {
    const char *v = std::getenv("MLX_HY4_GU_HILO");
    return v && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}

bool gguf_down_f32_env() {
  static const bool enabled = [] {
    const char *v = std::getenv("MLX_HY4_DOWN_F32");
    return v && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}

bool q8_prefill_1mma_env() {
  static const bool enabled = [] {
    const char *v = std::getenv("MLX_HY4_Q8_PREFILL_1MMA");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

bool q8_grouped_1mma_env() {
  static const bool enabled = [] {
    const char *v = std::getenv("MLX_HY4_Q8_GROUPED_1MMA");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

bool mla_prefill_2mma_env() {
  static const bool enabled = [] {
    const char *v = std::getenv("MLX_HY4_MLA_PREFILL_2MMA");
    return v && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}

bool prefill_q8_nax_env_disabled() {
  static const bool disabled = [] {
    const char *v = std::getenv("MLX_HY4_PREFILL_Q8_NAX");
    return v && std::strcmp(v, "0") == 0;
  }();
  return disabled;
}

struct Float4Key {
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;

  bool operator==(const Float4Key &o) const {
    return a == o.a && b == o.b && c == o.c && d == o.d;
  }
};

struct Float4KeyHash {
  size_t operator()(const Float4Key &k) const noexcept {
    uint64_t h = uint64_t(k.a) | (uint64_t(k.b) << 32);
    h ^= (uint64_t(k.c) | (uint64_t(k.d) << 32)) + 0x9e3779b97f4a7c15ULL +
         (h << 6) + (h >> 2);
    return size_t(h);
  }
};

array cached_f32_quad(float a, float b, float c, float d) {
  const Float4Key key{float_bits(a), float_bits(b), float_bits(c),
                      float_bits(d)};
  thread_local std::unordered_map<Float4Key, array, Float4KeyHash> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const float vals[4] = {a, b, c, d};
  array arr(vals, {4}, mlx::core::float32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

struct Float5Key {
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
  uint32_t e;

  bool operator==(const Float5Key &o) const {
    return a == o.a && b == o.b && c == o.c && d == o.d && e == o.e;
  }
};

struct Float5KeyHash {
  size_t operator()(const Float5Key &k) const noexcept {
    uint64_t h = uint64_t(k.a) | (uint64_t(k.b) << 32);
    h ^= (uint64_t(k.c) | (uint64_t(k.d) << 32)) + 0x9e3779b97f4a7c15ULL +
         (h << 6) + (h >> 2);
    h ^= uint64_t(k.e) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return size_t(h);
  }
};

array cached_f32_five(float a, float b, float c, float d, float e) {
  const Float5Key key{float_bits(a), float_bits(b), float_bits(c),
                      float_bits(d), float_bits(e)};
  thread_local std::unordered_map<Float5Key, array, Float5KeyHash> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const float vals[5] = {a, b, c, d, e};
  array arr(vals, {5}, mlx::core::float32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

const array &cached_zero_f32() {
  static const array arr = [] {
    const float zeros[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    return array(zeros, {8}, mlx::core::float32);
  }();
  return arr;
}

array cached_f32_pair(float a, float b) {
  const uint64_t key =
      (uint64_t(float_bits(a)) << 32) | uint64_t(float_bits(b));
  thread_local std::unordered_map<uint64_t, array> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const float vals[2] = {a, b};
  array arr(vals, {2}, mlx::core::float32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

array cached_f32_triple(float a, float b, float c) {
  const Float3Key key{float_bits(a), float_bits(b), float_bits(c)};
  thread_local std::unordered_map<Float3Key, array, Float3KeyHash> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const float vals[3] = {a, b, c};
  array arr(vals, {3}, mlx::core::float32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

array cached_row_ids(int rows, int top) {
  const int n_assign = rows * top;
  if (n_assign > 512) {
    std::vector<uint32_t> rids(static_cast<size_t>(n_assign));
    for (int r = 0; r < rows; ++r) {
      for (int t = 0; t < top; ++t) {
        rids[static_cast<size_t>(r * top + t)] = static_cast<uint32_t>(r);
      }
    }
    return array(rids.data(), {n_assign}, mlx::core::uint32);
  }
  const uint64_t key =
      (uint64_t(uint32_t(rows)) << 32) | uint64_t(uint32_t(top));
  thread_local std::unordered_map<uint64_t, array> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  std::vector<uint32_t> rids(static_cast<size_t>(n_assign));
  for (int r = 0; r < rows; ++r) {
    for (int t = 0; t < top; ++t) {
      rids[static_cast<size_t>(r * top + t)] = static_cast<uint32_t>(r);
    }
  }
  array arr(rids.data(), {n_assign}, mlx::core::uint32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

const array &cached_zero_f16() {
  static const array arr = [] {
    const mlx::core::float16_t zeros[8] = {
        mlx::core::float16_t(0.0f), mlx::core::float16_t(0.0f),
        mlx::core::float16_t(0.0f), mlx::core::float16_t(0.0f),
        mlx::core::float16_t(0.0f), mlx::core::float16_t(0.0f),
        mlx::core::float16_t(0.0f), mlx::core::float16_t(0.0f),
    };
    return array(zeros, {8}, mlx::core::float16);
  }();
  return arr;
}

const array &cached_zero_i32() {
  static const array arr = [] {
    const int32_t zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    return array(zeros, {8}, mlx::core::int32);
  }();
  return arr;
}

const array &cached_zero_u32() {
  static const array arr = [] {
    const uint32_t zeros[16] = {0};
    return array(zeros, {16}, mlx::core::uint32);
  }();
  return arr;
}

const array &cached_zero_u8() {
  static const array arr = [] {
    const uint8_t zeros[16] = {0};
    return array(zeros, {16}, mlx::core::uint8);
  }();
  return arr;
}

array cached_u32_five(uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                      uint32_t e) {
  const Float5Key key{a, b, c, d, e};
  thread_local std::unordered_map<Float5Key, array, Float5KeyHash> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const uint32_t vals[5] = {a, b, c, d, e};
  array arr(vals, {5}, mlx::core::uint32);
  if (cache.size() < 256) {
    cache.emplace(key, arr);
  }
  return arr;
}

array cached_u32_triple(uint32_t a, uint32_t b, uint32_t c) {
  const Float3Key key{a, b, c};
  thread_local std::unordered_map<Float3Key, array, Float3KeyHash> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const uint32_t vals[3] = {a, b, c};
  array arr(vals, {3}, mlx::core::uint32);
  if (cache.size() < 256) {
    cache.emplace(key, arr);
  }
  return arr;
}

array cached_u32_quad(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  const Float4Key key{a, b, c, d};
  thread_local std::unordered_map<Float4Key, array, Float4KeyHash> cache;
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const uint32_t vals[4] = {a, b, c, d};
  array arr(vals, {4}, mlx::core::uint32);
  if (cache.size() < 256) {
    cache.emplace(key, arr);
  }
  return arr;
}

// Materializes a fresh 0-offset row-contiguous buffer; mlx::core::copy() and
// mlx::core::contiguous() can preserve a shared buffer's misaligned offset().
class Hy4Materialize final : public mlx::core::UnaryPrimitive {
 public:
  explicit Hy4Materialize(Stream stream) : UnaryPrimitive(stream) {}

  DEFINE_NAME(Hy4Materialize)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

  void eval_cpu(const std::vector<array> &inputs, array &out) override {
    mlx::core::copy_cpu(inputs[0], out, mlx::core::CopyType::General, stream());
  }

  void eval_gpu(const std::vector<array> &inputs, array &out) override {
    mlx::core::copy_gpu(inputs[0], out, mlx::core::CopyType::General, stream());
  }
};

bool is_layout_aligned(const array &a, size_t elem_align,
                       bool require_contiguous) {
  const size_t byte_align = elem_align * a.itemsize();
  if (require_contiguous) {
    return a.flags().row_contiguous &&
           (elem_align <= 1 || a.offset() % byte_align == 0);
  }
  if (a.ndim() < 1 || a.strides().back() != 1 ||
      (elem_align > 1 && a.offset() % byte_align != 0)) {
    return false;
  }
  for (int i = 0; i + 1 < a.ndim(); ++i) {
    const int64_t st = a.strides(i);
    if (st < 0 || (elem_align > 1 && st % int64_t(elem_align) != 0)) {
      return false;
    }
  }
  return true;
}

// Eval-time layout guard for lazy views (e.g. unevaluated Slice whose real
// strides/offset are only known after eval): shares the buffer zero-copy when
// aligned, otherwise materializes a fresh 0-offset row-contiguous buffer.
class Hy4EnsureAligned final : public mlx::core::UnaryPrimitive {
 public:
  Hy4EnsureAligned(Stream stream, size_t elem_align, bool require_contiguous)
      : UnaryPrimitive(stream),
        elem_align_(elem_align),
        require_contiguous_(require_contiguous) {}

  DEFINE_NAME(Hy4EnsureAligned)
  DEFINE_INPUT_OUTPUT_SHAPE()

  bool is_equivalent(const Primitive &other) const override {
    const auto &o = static_cast<const Hy4EnsureAligned &>(other);
    return elem_align_ == o.elem_align_ &&
           require_contiguous_ == o.require_contiguous_;
  }

  void eval_cpu(const std::vector<array> &inputs, array &out) override {
    const auto &in = inputs[0];
    if (is_layout_aligned(in, elem_align_, require_contiguous_)) {
      out.copy_shared_buffer(in);
    } else {
      mlx::core::copy_cpu(in, out, mlx::core::CopyType::General, stream());
    }
  }

  void eval_gpu(const std::vector<array> &inputs, array &out) override {
    const auto &in = inputs[0];
    if (is_layout_aligned(in, elem_align_, require_contiguous_)) {
      out.copy_shared_buffer(in);
    } else {
      mlx::core::copy_gpu(in, out, mlx::core::CopyType::General, stream());
    }
  }

 private:
  size_t elem_align_;
  bool require_contiguous_;
};

array ensure_contiguous_aligned(const array &a, size_t elem_align) {
  if (!a.has_primitive()) {
    if (is_layout_aligned(a, elem_align, true)) {
      return a;
    }
    const auto stream = default_stream(Device::gpu);
    return array(a.shape(), a.dtype(), std::make_shared<Hy4Materialize>(stream),
                 {a});
  }
  const auto stream = default_stream(Device::gpu);
  return array(a.shape(), a.dtype(),
               std::make_shared<Hy4EnsureAligned>(stream, elem_align, true),
               {a});
}

// Accepts row-contiguous or strided views whose innermost stride is 1 and
// outer strides + base offset satisfy `elem_align`, avoiding copy() on
// multi-row column slices (gate_lin, q_nope, shared_gu, z).
array ensure_strided_aligned(const array &a, size_t elem_align) {
  if (!a.has_primitive()) {
    if (is_layout_aligned(a, elem_align, false)) {
      return a;
    }
    const auto stream = default_stream(Device::gpu);
    return array(a.shape(), a.dtype(), std::make_shared<Hy4Materialize>(stream),
                 {a});
  }
  const auto stream = default_stream(Device::gpu);
  return array(a.shape(), a.dtype(),
               std::make_shared<Hy4EnsureAligned>(stream, elem_align, false),
               {a});
}

const mlx::core::fast::CustomKernelFunction &hy4_hc_mix_reduce_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_hc_mix_reduce", {"x", "weight", "scale", "base", "params"},
      {"h", "post"},
#include "metal/hy4/hc_mix_reduce.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_hc_post_identity_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_hc_post_identity", {"y", "res", "post"}, {"out"},
#include "metal/hy4/hc_post_identity.metal.inc"
  );
  return kernel;
}

const std::string &hy4_hc_fused_header() {
  static const std::string header =
#include "metal/hy4/hc_fused.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_hc_mix_reduce_norm_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_hc_mix_reduce_norm",
      {"x", "weight", "scale", "base", "norm_w", "params"}, {"h_norm", "post"},
      R"METAL(
      constexpr int N_GATES = HEAD_ONLY ? HC : (2 * HC);
      threadgroup float tg_red[8 * (1 + N_GATES)];
      threadgroup float tg_pre[HC];
      threadgroup float tg_h2[8];
      hy4_hc_mix_reduce_norm_impl<HC, HEAD_ONLY, VEC4>(
          x, weight, scale, base, norm_w, params, h_norm, post,
          tg_red, tg_pre, tg_h2,
          x_shape[0], x_shape[2], threadgroup_position_in_grid.y,
          thread_index_in_threadgroup, thread_index_in_simdgroup,
          simdgroup_index_in_threadgroup);
      )METAL",
      hy4_hc_fused_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_post_norm_hc_post_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_post_norm_hc_post", {"y", "res", "post", "post_norm_w", "post_eps"},
      {"out"},
      R"METAL(
      threadgroup float tg_y2[8];
      hy4_post_norm_hc_post_impl<HC, VEC4>(
          y, res, post, post_norm_w, post_eps, out, tg_y2,
          res_shape[0], res_shape[2], threadgroup_position_in_grid.y,
          thread_index_in_threadgroup, thread_index_in_simdgroup,
          simdgroup_index_in_threadgroup);
      )METAL",
      hy4_hc_fused_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_hc_bridge_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_hc_bridge",
      {"y", "res", "post_prev", "post_norm_w", "next_weight", "next_scale",
       "next_base", "next_norm_w", "params"},
      {"x_out", "h_out", "post_next"},
      R"METAL(
      constexpr int N_GATES = HEAD_ONLY ? HC : (2 * HC);
      constexpr int R_MULT = (USE_REG && ROW2) ? 2 : 1;
      threadgroup float tg_red[R_MULT * 8 * (1 + N_GATES)];
      threadgroup float tg_pre[R_MULT * HC];
      threadgroup float tg_y2[R_MULT * 8];
      threadgroup float tg_h2[R_MULT * 8];
      hy4_hc_bridge_impl<HC, VEC4, SKIP_NEXT_NORM, HEAD_ONLY, USE_REG, ROW2>(
          y, res, post_prev, post_norm_w, next_weight, next_scale, next_base,
          next_norm_w, params, x_out, h_out, post_next,
          tg_red, tg_pre, tg_y2, tg_h2,
          res_shape[0], res_shape[2], threadgroup_position_in_grid.y,
          thread_index_in_threadgroup, thread_index_in_simdgroup,
          simdgroup_index_in_threadgroup);
      )METAL",
      hy4_hc_fused_header());
  return kernel;
}

struct Hy4HcBridgeParams {
  int32_t m;
  int32_t d;
  float post_eps;
  float hc_norm_eps;
  float hc_eps;
  float magnitude;
  float next_eps;
};

struct Hy4HcBridgePipelines {
  MTL::ComputePipelineState *pass1;
  MTL::ComputePipelineState *pass2;
};

Hy4HcBridgePipelines hy4_hc_bridge_pipelines(mlx::core::metal::Device &device,
                                             int hc, int nsplits, int tg_x1,
                                             int m_tile, bool skip_next_norm,
                                             bool head_only) {
  const std::string lib_name = "mlx_node_hy4_hc_bridge_" + std::to_string(hc) +
                               "_" + std::to_string(nsplits) + "_" +
                               std::to_string(tg_x1);
  auto *library = device.get_library(lib_name, [hc, nsplits, tg_x1] {
    std::string src = "#include <metal_stdlib>\nusing namespace metal;\n";
    src += "constexpr constant int HY4_HC = " + std::to_string(hc) + ";\n";
    src += "constexpr constant int HY4_NSPLITS = " + std::to_string(nsplits) +
           ";\n";
    src += "constexpr constant int HY4_TG_X = " + std::to_string(tg_x1) + ";\n";
    src += R"METAL(
struct Hy4HcBridgeParams {
  int m;
  int d;
  float post_eps;
  float hc_norm_eps;
  float hc_eps;
  float magnitude;
  float next_eps;
};
)METAL";
    src += hy4_hc_fused_header();
    src += R"METAL(
#define HY4_HC_P1_KERNEL(NAME, MT, HO)                                        \
  [[kernel]] void NAME(                                                       \
      const device float* y [[buffer(0)]],                                    \
      const device float* res [[buffer(1)]],                                  \
      const device float* post_prev [[buffer(2)]],                            \
      const device float* post_norm_w [[buffer(3)]],                          \
      const device float* next_weight [[buffer(4)]],                          \
      device float* x_out [[buffer(5)]],                                      \
      device float* partial [[buffer(6)]],                                    \
      constant Hy4HcBridgeParams& p [[buffer(7)]],                            \
      uint3 tgp [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_index_in_threadgroup]],                               \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint sg [[simdgroup_index_in_threadgroup]]) {                           \
    constexpr int N_GATES = (HO) ? HY4_HC : (2 * HY4_HC);                     \
    constexpr int NSG = HY4_TG_X >> 5;                                        \
    constexpr int RED_CAP = (NSG > 1) ? ((MT) * NSG * (1 + N_GATES)) : 1;     \
    threadgroup float tg_y2[(MT) * NSG];                                      \
    threadgroup float tg_red[RED_CAP];                                        \
    const float params[5] = {                                                 \
        p.post_eps, p.hc_norm_eps, p.hc_eps, p.magnitude, p.next_eps};        \
    hy4_hc_bridge_pass1_impl<HY4_HC, HY4_NSPLITS, HY4_TG_X, MT, HO>(          \
        y, res, post_prev, post_norm_w, next_weight, params,                  \
        x_out, partial, tg_y2, tg_red, p.m, p.d, tgp.y, tgp.z, tid, lane,     \
        sg);                                                                  \
  }

#define HY4_HC_P2_KERNEL(NAME, SN, HO)                                        \
  [[kernel]] void NAME(                                                       \
      const device float* x_out [[buffer(0)]],                                \
      const device float* y [[buffer(1)]],                                    \
      const device float* res [[buffer(2)]],                                  \
      const device float* post_prev [[buffer(3)]],                            \
      const device float* post_norm_w [[buffer(4)]],                          \
      const device float* partial [[buffer(5)]],                              \
      const device float* next_scale [[buffer(6)]],                           \
      const device float* next_base [[buffer(7)]],                            \
      const device float* next_norm_w [[buffer(8)]],                          \
      device float* h_out [[buffer(9)]],                                      \
      device float* post_next [[buffer(10)]],                                 \
      constant Hy4HcBridgeParams& p [[buffer(11)]],                           \
      uint3 tgp [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_index_in_threadgroup]],                               \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint sg [[simdgroup_index_in_threadgroup]]) {                           \
    threadgroup float tg_pre[HY4_HC];                                         \
    threadgroup float tg_h2[8];                                               \
    const float params[5] = {                                                 \
        p.post_eps, p.hc_norm_eps, p.hc_eps, p.magnitude, p.next_eps};        \
    hy4_hc_bridge_pass2_impl<HY4_HC, HY4_NSPLITS, SN, HO>(                    \
        x_out, y, res, post_prev, post_norm_w, partial, next_scale,           \
        next_base, next_norm_w, params, h_out, post_next, tg_pre, tg_h2,      \
        p.m, p.d, tgp.y, tid, lane, sg);                                      \
  }

HY4_HC_P1_KERNEL(hy4_hc_p1_f_m1, 1, false)
HY4_HC_P1_KERNEL(hy4_hc_p1_f_m2, 2, false)
HY4_HC_P1_KERNEL(hy4_hc_p1_f_m3, 3, false)
HY4_HC_P1_KERNEL(hy4_hc_p1_f_m4, 4, false)
HY4_HC_P1_KERNEL(hy4_hc_p1_h_m1, 1, true)
HY4_HC_P1_KERNEL(hy4_hc_p1_h_m2, 2, true)
HY4_HC_P1_KERNEL(hy4_hc_p1_h_m3, 3, true)
HY4_HC_P1_KERNEL(hy4_hc_p1_h_m4, 4, true)
HY4_HC_P2_KERNEL(hy4_hc_p2_00, false, false)
HY4_HC_P2_KERNEL(hy4_hc_p2_01, false, true)
HY4_HC_P2_KERNEL(hy4_hc_p2_10, true, false)
HY4_HC_P2_KERNEL(hy4_hc_p2_11, true, true)
)METAL";
    return src;
  });
  static constexpr const char *kP1Names[2][4] = {
      {"hy4_hc_p1_f_m1", "hy4_hc_p1_f_m2", "hy4_hc_p1_f_m3", "hy4_hc_p1_f_m4"},
      {"hy4_hc_p1_h_m1", "hy4_hc_p1_h_m2", "hy4_hc_p1_h_m3", "hy4_hc_p1_h_m4"},
  };
  const int mt_idx = std::clamp(m_tile, 1, 4) - 1;
  const char *p1_name = kP1Names[head_only ? 1 : 0][mt_idx];
  const char *p2_name = skip_next_norm
                            ? (head_only ? "hy4_hc_p2_11" : "hy4_hc_p2_10")
                            : (head_only ? "hy4_hc_p2_01" : "hy4_hc_p2_00");
  return {device.get_kernel(p1_name, library),
          device.get_kernel(p2_name, library)};
}

// Single-graph-op multi-TG Hyper-Connections bridge: pass1 tiles up to 4 rows
// across NSPLITS TGs so the gate matrix is read once per split, pass2 merges
// partials and emits h_out + post_next inside the same Metal command encoder.
class Hy4HcBridge final : public mlx::core::Primitive {
 public:
  Hy4HcBridge(Stream stream, int hc, int nsplits, int tg_x1, int m_tile,
              bool skip_next_norm, bool head_only, Hy4HcBridgeParams params)
      : Primitive(stream),
        hc_(hc),
        nsplits_(nsplits),
        tg_x1_(tg_x1),
        m_tile_(m_tile),
        skip_next_norm_(skip_next_norm),
        head_only_(head_only),
        params_(params) {}

  DEFINE_NAME(Hy4HcBridge)

  std::vector<Shape> output_shapes(const std::vector<array> &) override {
    const Shape x_shape =
        head_only_ ? Shape{1} : Shape{params_.m, hc_, params_.d};
    const Shape post_shape = head_only_ ? Shape{1} : Shape{params_.m, hc_};
    return {x_shape, Shape{params_.m, params_.d}, post_shape};
  }

  bool is_equivalent(const Primitive &other) const override {
    const auto &o = static_cast<const Hy4HcBridge &>(other);
    return hc_ == o.hc_ && nsplits_ == o.nsplits_ && tg_x1_ == o.tg_x1_ &&
           m_tile_ == o.m_tile_ && skip_next_norm_ == o.skip_next_norm_ &&
           head_only_ == o.head_only_ && params_.m == o.params_.m &&
           params_.d == o.params_.d &&
           float_bits(params_.post_eps) == float_bits(o.params_.post_eps) &&
           float_bits(params_.hc_norm_eps) ==
               float_bits(o.params_.hc_norm_eps) &&
           float_bits(params_.hc_eps) == float_bits(o.params_.hc_eps) &&
           float_bits(params_.magnitude) == float_bits(o.params_.magnitude) &&
           float_bits(params_.next_eps) == float_bits(o.params_.next_eps);
  }

  void eval_cpu(const std::vector<array> &, std::vector<array> &) override {
    throw std::runtime_error("Hy4HcBridge requires Metal");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto s = stream();
    auto &device = mlx::core::metal::device(s.device);
    auto &enc = mlx::core::metal::get_command_encoder(s);

    auto &x_out = outputs[0];
    auto &h_out = outputs[1];
    auto &post_next = outputs[2];
    x_out.set_data(mlx::core::allocator::malloc(x_out.nbytes()));
    h_out.set_data(mlx::core::allocator::malloc(h_out.nbytes()));
    post_next.set_data(mlx::core::allocator::malloc(post_next.nbytes()));

    const int n_gates = head_only_ ? hc_ : (2 * hc_);
    const int part_cols = 1 + n_gates + (head_only_ ? 1 : 0);
    array partials({params_.m, nsplits_, part_cols}, mlx::core::float32,
                   nullptr, {});
    partials.set_data(mlx::core::allocator::malloc(partials.nbytes()));
    enc.add_temporary(partials);

    std::vector<array> copies;
    auto check_input = [&copies, &s](const array &x, size_t byte_align) -> array {
      if (x.flags().row_contiguous &&
          (byte_align <= 1 || (x.offset() % byte_align) == 0)) {
        return x;
      }
      copies.emplace_back(x.shape(), x.dtype(), nullptr, std::vector<array>{});
      mlx::core::copy_gpu(x, copies.back(), mlx::core::CopyType::General, s);
      return copies.back();
    };
    const array in0 = check_input(inputs[0], 16);
    const array in1 = check_input(inputs[1], 16);
    const array in2 = check_input(inputs[2], 1);
    const array in3 = check_input(inputs[3], 16);
    const array in4 = check_input(inputs[4], 16);
    const array in5 = check_input(inputs[5], 1);
    const array in6 = check_input(inputs[6], 1);
    const array in7 =
        skip_next_norm_ ? inputs[7] : check_input(inputs[7], 16);

    auto pso = hy4_hc_bridge_pipelines(device, hc_, nsplits_, tg_x1_, m_tile_,
                                       skip_next_norm_, head_only_);
    const int n_mblk = (params_.m + m_tile_ - 1) / m_tile_;
    enc.set_compute_pipeline_state(pso.pass1);
    enc.set_input_array(in0, 0);
    enc.set_input_array(in1, 1);
    enc.set_input_array(in2, 2);
    enc.set_input_array(in3, 3);
    enc.set_input_array(in4, 4);
    enc.set_output_array(x_out, 5);
    enc.set_output_array(partials, 6);
    enc.set_bytes(params_, 7);
    enc.dispatch_threadgroups(MTL::Size(1, size_t(nsplits_), size_t(n_mblk)),
                              MTL::Size(size_t(tg_x1_), 1, 1));

    enc.set_compute_pipeline_state(pso.pass2);
    enc.set_input_array(x_out, 0);
    enc.set_input_array(in0, 1);
    enc.set_input_array(in1, 2);
    enc.set_input_array(in2, 3);
    enc.set_input_array(in3, 4);
    enc.set_input_array(partials, 5);
    enc.set_input_array(in5, 6);
    enc.set_input_array(in6, 7);
    enc.set_input_array(in7, 8);
    enc.set_output_array(h_out, 9);
    enc.set_output_array(post_next, 10);
    enc.set_bytes(params_, 11);
    enc.dispatch_threadgroups(MTL::Size(1, size_t(params_.m), 1),
                              MTL::Size(256, 1, 1));
    if (!copies.empty()) {
      enc.add_temporaries(std::move(copies));
    }
  }

 private:
  int hc_;
  int nsplits_;
  int tg_x1_;
  int m_tile_;
  bool skip_next_norm_;
  bool head_only_;
  Hy4HcBridgeParams params_;
};

const mlx::core::fast::CustomKernelFunction &hy4_f32_head_weights_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_f32_head_weights", {"x", "w", "scale"}, {"out"},
#include "metal/hy4/f32_head_weights.metal.inc"
  );
  return kernel;
}

const std::string &hy4_mtp_fused_header() {
  static const std::string header =
#include "metal/hy4/mtp_fused.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_mtp_eh_norm_concat_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mtp_eh_norm_concat", {"e_in", "h_in", "enorm", "hnorm", "eps"},
      {"out"},
      R"METAL(
      threadgroup float tg_sq[8];
      hy4_mtp_eh_norm_concat_impl<VEC4, USE_REG>(
          e_in, h_in, enorm, hnorm, eps, out, tg_sq,
          e_in_shape[0], e_in_shape[1],
          threadgroup_position_in_grid.y, threadgroup_position_in_grid.z,
          thread_index_in_threadgroup, thread_index_in_simdgroup,
          simdgroup_index_in_threadgroup);
      )METAL",
      hy4_mtp_fused_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_post_norm_add_rms_norm_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_post_norm_add_rms_norm", {"res", "y", "post_w", "next_w", "eps"},
      {"x_out", "h_out"},
      R"METAL(
      threadgroup float tg_y2[8];
      threadgroup float tg_x2[8];
      hy4_post_norm_add_rms_norm_impl<VEC4, USE_REG, NEED_X_OUT>(
          res, y, post_w, next_w, eps, x_out, h_out, tg_y2, tg_x2,
          res_shape[0], res_shape[1],
          threadgroup_position_in_grid.y,
          thread_index_in_threadgroup, thread_index_in_simdgroup,
          simdgroup_index_in_threadgroup);
      )METAL",
      hy4_mtp_fused_header());
  return kernel;
}

const std::string &hy4_mla_prep_q_kv_header() {
  static const std::string header =
#include "metal/hy4/mla_prep_q_kv.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_prep_q_kv_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_prep_q_kv",
      {"all", "q_norm_w", "kv_norm_w", "pos", "inv_freq", "norm_eps"},
      {"qr", "kv0", "kv1"},
      R"METAL(
      threadgroup float tg_red[16];
      hy4_mla_prep_q_kv_impl<VEC4, OUT_F16, SPLIT_OUT>(
          all, q_norm_w, kv_norm_w, pos, inv_freq, norm_eps, qr, kv0, kv1,
          tg_red, all_shape[0], q_norm_w_shape[0], kv_norm_w_shape[0],
          inv_freq_shape[0], size_t(all_strides[0]),
          threadgroup_position_in_grid.y, thread_index_in_threadgroup,
          thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
      )METAL",
      hy4_mla_prep_q_kv_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const std::string &hy4_engram_fused_header() {
  static const std::string header =
#include "metal/hy4/engram_fused.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_engram_norm_rows_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_engram_norm_rows", {"rows", "lookup_norm", "eps"}, {"out"},
      R"METAL(
      threadgroup float tg_sq[4];
      hy4_engram_norm_rows_impl(
          rows, lookup_norm, eps, out, tg_sq,
          rows_shape[0], rows_shape[1], rows_shape[2],
          threadgroup_position_in_grid.y, threadgroup_position_in_grid.z,
          thread_index_in_threadgroup, thread_index_in_simdgroup,
          simdgroup_index_in_threadgroup);
      )METAL",
      hy4_engram_fused_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_engram_gate_add_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_engram_gate_add",
      {"x", "kv_raw", "kv_b", "gate_w", "pre_gamma", "hc_scale", "keep_mask",
       "eps"},
      {"out"},
      R"METAL(
      threadgroup float tg_red[16];
      hy4_engram_gate_add_impl<HC, VEC4, USE_REG, HAS_MASK>(
          x, kv_raw, kv_b, gate_w, pre_gamma, hc_scale, keep_mask, eps,
          out, tg_red, x_shape[0], x_shape[2],
          threadgroup_position_in_grid.y, thread_index_in_threadgroup,
          thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
      )METAL",
      hy4_engram_fused_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_bf16_lm_head_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_bf16_lm_head", {"x", "w", "w_meta"}, {"out"},
#include "metal/hy4/bf16_lm_head.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_gguf_q8_0_embed_rows_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_gguf_q8_0_embed_rows", {"ids", "buf", "byte_off", "dims"}, {"out"},
#include "metal/hy4/gguf_q8_0_embed_rows.metal.inc"
  );
  return kernel;
}

mlx_array *bf16_lm_head_dispatch(const array &xa, const array &wc,
                                 uint64_t byte_offset, int v, bool mmap_u8) {
  const int m = xa.shape(0);
  const int k = xa.shape(1);
  if (xa.size() < 8 || wc.size() < 8) {
    return nullptr;
  }
  const bool fast_k = (k % 256 == 0);
  constexpr int nsg = 4;
  const int rps = (m == 1) ? 4 : 2;
  const int r_tile = m;
  const bool stream_x = (m > 4);
  const int rows_per_tg = nsg * rps;
  array xc = ensure_contiguous_aligned(xa, 4);
  array w_meta = cached_u64_pair(byte_offset, uint64_t(v));

  const auto stream = default_stream(Device::gpu);
  const std::tuple<int, int, int> grid{
      32, ((v + rows_per_tg - 1) / rows_per_tg) * nsg,
      (m + r_tile - 1) / r_tile};
  const std::tuple<int, int, int> tg{32, nsg, 1};
  const uint64_t tmpl_key = uint64_t(fast_k ? 1 : 0) |
                            (uint64_t(stream_x ? 1 : 0) << 1) |
                            (uint64_t(mmap_u8 ? 1 : 0) << 2) |
                            (uint64_t(r_tile) << 4) | (uint64_t(rps) << 12) |
                            (uint64_t(nsg) << 20);
  const std::vector<array> inputs{xc, wc, w_meta};
  const std::vector<Shape> out_shapes{{m, v}};
  const std::vector<Dtype> out_dtypes{mlx::core::float32};
  auto out = dispatch_cached_kernel(
      kKernelBf16LmHead, stream, grid, tg, tmpl_key, inputs, out_shapes,
      out_dtypes, [&] {
        return hy4_bf16_lm_head_kernel()(
            inputs, out_shapes, out_dtypes, grid, tg,
            {{"FAST_K", fast_k},
             {"R_TILE", r_tile},
             {"RPS", rps},
             {"NSG", nsg},
             {"STREAM_X", stream_x},
             {"MMAP_U8", mmap_u8}},
            std::nullopt, false, stream);
      });
  return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
}

const mlx::core::fast::CustomKernelFunction &hy4_q8_decode_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_dense_decode", {"x", "w", "scales", "biases"}, {"out"},
#include "metal/hy4/q8_dense_decode.metal.inc"
  );
  return kernel;
}

const std::string &hy4_affine4_dense_decode_header() {
  static const std::string header =
#include "metal/hy4/affine4_dense_decode.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &
hy4_affine4_dense_decode_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_affine4_dense_decode", {"x", "w", "scales", "biases"}, {"out"},
      R"METAL(
      hy4_affine4_dense_decode_impl<M_TILE, ROWS_PER_SG, FAST_K>(
          x, w, scales, biases, out,
          x_shape[0], w_shape[0], x_shape[1],
          threadgroup_position_in_grid.y, threadgroup_position_in_grid.z,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_affine4_dense_decode_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const std::string &hy4_q8_grouped_short_header() {
  static const std::string header =
#include "metal/hy4/q8_grouped_short.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_q8_grouped_k192_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_grouped_k192", {"x", "w", "scales", "biases"}, {"out"},
      R"METAL(
      const int M = x_shape[0];
      const int G = x_shape[1];
      const int N = w_shape[0] / G;
      hy4_q8_grouped_k192_impl<M_TILE, SYM_F16, false>(
          x, w, scales, biases, x, x, out, M, G, N, 0,
          size_t(x_strides[0]), size_t(x_strides[1]),
          threadgroup_position_in_grid.y, threadgroup_position_in_grid.z,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_q8_grouped_short_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_q8_grouped_k192_rope_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_grouped_k192_rope",
      {"x", "w", "scales", "biases", "pos", "inv_freq"}, {"out"},
      R"METAL(
      const int M = x_shape[0];
      const int G = x_shape[1];
      const int N = w_shape[0] / G;
      const int rope_dim = inv_freq_shape[0] * 2;
      hy4_q8_grouped_k192_impl<M_TILE, SYM_F16, true>(
          x, w, scales, biases, pos, inv_freq, out, M, G, N, rope_dim,
          size_t(x_strides[0]), size_t(x_strides[1]),
          threadgroup_position_in_grid.y, threadgroup_position_in_grid.z,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_q8_grouped_short_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_q8_grouped_k512_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_grouped_k512", {"x", "w", "scales", "biases", "gate"}, {"out"},
      R"METAL(
      const int M = x_shape[0];
      const int G = x_shape[1];
      const int N = w_shape[0] / G;
      const size_t g_s0 = size_t(gate_strides[0]);
      const size_t g_s1 = (gate_ndim == 3) ? size_t(gate_strides[1]) : size_t(N);
      hy4_q8_grouped_k512_impl<M_TILE, SYM_F16, GATED>(
          x, w, scales, biases, gate, out, M, G, N, g_s0, g_s1,
          threadgroup_position_in_grid.y, threadgroup_position_in_grid.z,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_q8_grouped_short_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_router_mv_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_router_mv", {"x", "w"}, {"out"},
#include "metal/hy4/router_mv.metal.inc"
  );
  return kernel;
}

const std::string &hy4_indexer_fused_header() {
  static const std::string header =
#include "metal/hy4/indexer_fused.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &
hy4_indexer_score_decode_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_indexer_score_decode", {"q", "w", "k_pool", "page_ids", "params"},
      {"out"},
      R"METAL(
      threadgroup float4 q_tg[16 * 32];
      threadgroup float w_tg[16];
      hy4_indexer_score_decode_impl<PAGED, INDEX_ROWS>(
          q, w, k_pool, page_ids, params, out, q_tg, w_tg,
          q_shape[1], threadgroup_position_in_grid.z,
          threadgroup_position_in_grid.y, thread_index_in_threadgroup,
          thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
      )METAL",
      hy4_indexer_fused_header());
  return kernel;
}

const std::string &hy4_q8_prefill_swz_header() {
  static const std::string header = R"METAL(
METAL_FUNC int hy4_swz_c8(int c8, int r) {
  return c8 ^ ((r & 3) << 1) ^ ((r & 4) >> 2);
}

template <short R_FRAGS, short K_FRAGS>
METAL_FUNC void hy4_load_tile_swz64(
    thread mlx::steel::NAXTile<half, R_FRAGS, K_FRAGS>& tile,
    const threadgroup half* src,
    const short fm,
    const short fn) {
  const int sub4 = int(fn) & 4;
  const int fn8 = int(fn) >> 3;
  const int r0_off = int(fm) * 64 + sub4;
  const int r1_off = (int(fm) + 8) * 64 + sub4;
  STEEL_PRAGMA_UNROLL
  for (short kk = 0; kk < K_FRAGS; ++kk) {
    const int c8 = int(kk) * 2 + fn8;
    const int sw_off = hy4_swz_c8(c8, int(fm)) << 3;
    STEEL_PRAGMA_UNROLL
    for (short rr = 0; rr < R_FRAGS; ++rr) {
      const threadgroup half* row_base = src + int(rr) * (16 * 64) + sw_off;
      const half4 va =
          *reinterpret_cast<const threadgroup half4*>(row_base + r0_off);
      const half4 vb =
          *reinterpret_cast<const threadgroup half4*>(row_base + r1_off);
      thread auto& f = tile.frag_at(rr, kk);
      f[0] = va.x;
      f[1] = va.y;
      f[2] = va.z;
      f[3] = va.w;
      f[4] = vb.x;
      f[5] = vb.y;
      f[6] = vb.z;
      f[7] = vb.w;
    }
  }
}
)METAL";
  return header;
}

const mlx::core::fast::CustomKernelFunction &
hy4_indexer_score_prefill_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_indexer_score_prefill",
      {"q", "w", "k_pool", "page_ids", "vis", "params"}, {"out"},
      R"METAL(
      constexpr uint STRIDE4 = 33u;
      threadgroup float4 k_tile_buf[32 * STRIDE4];
      threadgroup float4 q_tile_buf[16 * STRIDE4];
      threadgroup float s_dot[8 * 64];
      threadgroup float w_tg[16 * 16];
      hy4_indexer_score_prefill_impl<PAGED, INDEX_ROWS, HAS_VIS>(
          q, w, k_pool, page_ids, vis, params, out,
          k_tile_buf, q_tile_buf, s_dot, w_tg,
          q_shape[1], threadgroup_position_in_grid.y,
          threadgroup_position_in_grid.z, thread_index_in_threadgroup,
          thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
      )METAL",
      hy4_indexer_fused_header());
  return kernel;
}

const std::string &hy4_indexer_nax_header() {
  static const std::string header =
      std::string(mlx::core::quantized_preamble::nax()) +
      "#define HY4_INDEXER_NAX 1\n" + hy4_q8_prefill_swz_header() +
      hy4_indexer_fused_header();
  return header;
}

const mlx::core::fast::CustomKernelFunction &
hy4_indexer_score_prefill_nax_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_indexer_score_prefill_nax",
      {"q", "w", "k_pool", "page_ids", "vis", "params"}, {"out"},
      R"METAL(
      alignas(16) threadgroup half k_tile[64 * 128];
      hy4_indexer_score_prefill_nax_impl<PAGED, INDEX_ROWS, HAS_VIS>(
          q, w, k_pool, page_ids, vis, params, out, k_tile,
          q_shape[1], threadgroup_position_in_grid.y,
          threadgroup_position_in_grid.z, thread_index_in_threadgroup,
          thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
      )METAL",
      hy4_indexer_nax_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_indexer_score_decode_nax_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_indexer_score_decode_nax",
      {"q", "w", "k_pool", "page_ids", "params"}, {"out"},
      R"METAL(
      alignas(16) threadgroup half k_tile[32 * 136];
      hy4_indexer_score_decode_nax_impl<PAGED, INDEX_ROWS, SHARE_S>(
          q, w, k_pool, page_ids, params, out, k_tile,
          q_shape[1], threadgroup_position_in_grid.z,
          threadgroup_position_in_grid.y, thread_index_in_simdgroup);
      )METAL",
      hy4_indexer_nax_header());
  return kernel;
}

const std::string &hy4_moe_prefill_sort_header() {
  static const std::string header =
#include "metal/hy4/moe_prefill_sort_tiles.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &
hy4_moe_prefill_sort_tiles_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_moe_prefill_sort_tiles", {"ids", "params"},
      {"token_rows", "tiles", "inverse", "sorted_slots"},
      R"METAL(
      threadgroup atomic_uint tg_counts[1024];
      threadgroup uint tg_offsets[1024];
      threadgroup uint tg_tile_offsets[1024];
      threadgroup uint sg_c[8];
      threadgroup uint sg_t[8];
      hy4_moe_prefill_sort_single_impl<TILE_M>(
          ids, params, token_rows, tiles, inverse, sorted_slots,
          tg_counts, tg_offsets, tg_tile_offsets, sg_c, sg_t,
          thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
          thread_index_in_simdgroup);
      )METAL",
      hy4_moe_prefill_sort_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_moe_prefill_sort_hist_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_moe_prefill_sort_hist", {"ids", "params"}, {"block_counts"},
      R"METAL(
      threadgroup atomic_uint tg_hist[1024];
      hy4_moe_prefill_sort_hist_impl(
          ids, params, block_counts, tg_hist,
          threadgroup_position_in_grid.x, thread_index_in_threadgroup);
      )METAL",
      hy4_moe_prefill_sort_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_moe_prefill_sort_scan_tiles_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_moe_prefill_sort_scan_tiles", {"block_counts", "params"},
      {"block_offsets", "tiles", "meta"},
      R"METAL(
      threadgroup uint sg_c[8];
      threadgroup uint sg_t[8];
      hy4_moe_prefill_sort_scan_tiles_impl<TILE_M>(
          block_counts, params, block_offsets, tiles, meta, sg_c, sg_t,
          thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
          thread_index_in_simdgroup);
      )METAL",
      hy4_moe_prefill_sort_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_moe_prefill_sort_scatter_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_moe_prefill_sort_scatter",
      {"ids", "block_offsets", "meta", "params"},
      {"token_rows", "inverse", "sorted_slots"},
      R"METAL(
      threadgroup atomic_uint tg_cursor[1024];
      hy4_moe_prefill_sort_scatter_impl(
          ids, block_offsets, meta, params,
          token_rows, inverse, sorted_slots, tg_cursor,
          threadgroup_position_in_grid.x, thread_index_in_threadgroup);
      )METAL",
      hy4_moe_prefill_sort_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_indexer_topk_cells_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_indexer_topk_cells",
      {"scores", "visible_groups", "tail_starts", "tail_lens", "page_ids",
       "params"},
      {"out"},
      R"METAL(
      threadgroup atomic_uint hist[256];
      threadgroup uint sg_eq[8];
      threadgroup uint sg_keep[8];
      threadgroup uint sel_groups[512];
      threadgroup uint tg_state[3];
      hy4_indexer_topk_cells_impl<PAGED>(
          scores, visible_groups, tail_starts, tail_lens, page_ids, params, out,
          hist, sg_eq, sg_keep, sel_groups, tg_state,
          uint(scores_shape[0]), uint(scores_shape[1]),
          threadgroup_position_in_grid.y, thread_index_in_threadgroup,
          thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
      )METAL",
      hy4_indexer_fused_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_kpool_compress_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kpool_compress",
      {"tail_buf", "tok", "w_val", "w_gate", "ape", "comp_norm", "inv_freq",
       "params", "eps"},
      {"out"},
      R"METAL(
      threadgroup float4 x_tg[4 * 32];
      threadgroup float tg_pooled[128];
      threadgroup float tg_sq[4];
      const uint tid = simdgroup_index_in_threadgroup * 32u +
                       thread_index_in_simdgroup;
      hy4_kpool_compress_impl(
          tail_buf, tok, w_val, w_gate, ape, comp_norm, inv_freq, params, eps,
          out, x_tg, tg_pooled, tg_sq, inv_freq_shape[0] * 2,
          threadgroup_position_in_grid.z, tid, thread_index_in_simdgroup,
          simdgroup_index_in_threadgroup);
      )METAL",
      hy4_indexer_fused_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_partial_rope_tail_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_partial_rope_tail", {"x", "pos", "inv_freq"}, {"out"},
      R"METAL(
      const int M = x_shape[0];
      const int H = (x_ndim == 3) ? x_shape[1] : 1;
      const int D = (x_ndim == 3) ? x_shape[2] : x_shape[1];
      const size_t x_s0 = size_t(x_strides[0]);
      const size_t x_s1 = (x_ndim == 3) ? size_t(x_strides[1]) : 0u;
      hy4_partial_rope_tail_impl<TG_X, OUT_T>(
          x, pos, inv_freq, out, M, H, D, inv_freq_shape[0], x_s0, x_s1,
          thread_position_in_grid.y, threadgroup_position_in_grid.z,
          thread_position_in_threadgroup.x);
      )METAL",
      hy4_indexer_fused_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const std::string &hy4_q8_dual_down_header() {
  static const std::string header = R"METAL(
template <bool CLAMP>
inline float4 hy4_dual_swiglu4(float4 g, float4 u, float lim) {
  if constexpr (CLAMP) {
    g = metal::min(g, float4(lim));
    u = metal::clamp(u, float4(-lim), float4(lim));
  }
  const float4 sig = 1.0f / (1.0f + metal::exp2(-g * M_LOG2E_F));
  return (sig * g) * u;
}
)METAL";
  return header;
}

const std::string &hy4_q8_prefill_header() {
  static const std::string header =
      std::string(nax_header()) + hy4_q8_prefill_swz_header();
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_q8_dual_down_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_dual_down_add",
      {"routed_latent", "shared_gu", "w_up", "s_up", "w_sd", "s_sd", "limit"},
      {"out"},
#include "metal/hy4/q8_dual_down.metal.inc"
      , hy4_q8_dual_down_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const std::string &hy4_q8_dual_down_prefill_header() {
  static const std::string header =
      hy4_q8_prefill_header() + hy4_q8_dual_down_header();
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_q8_dual_down_prefill_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_dual_down_prefill",
      {"routed_latent", "shared_gu", "w_up", "s_up", "w_sd", "s_sd", "limit"},
      {"out"},
#include "metal/hy4/q8_dual_down_prefill.metal.inc"
      , hy4_q8_dual_down_prefill_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

int hy4_q8_decode_m_tile(int m) {
  if (m <= 1) {
    return 1;
  }
  if (m == 2) {
    return 2;
  }
  if (m <= 4) {
    return 4;
  }
  return 8;
}

int hy4_q8_short_m_tile(int m) {
  if (m <= 1) {
    return 1;
  }
  if (m == 2) {
    return 2;
  }
  return 4;
}

const mlx::core::fast::CustomKernelFunction &hy4_q8_prefill_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_dense_prefill", {"x", "w", "scales", "biases", "aux"}, {"out"},
#include "metal/hy4/q8_dense_prefill.metal.inc"
      , hy4_q8_prefill_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

mlx_array *hy4_q8_prefill_impl(mlx_array *x, mlx_array *w, mlx_array *scales,
                               mlx_array *biases, bool grouped, mlx_array *aux,
                               int epilogue, bool out_f16) {
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu) ||
        !mlx::core::metal::is_nax_available()) {
      return nullptr;
    }
    if (!x || !w || !scales) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    if (xa.ndim() != (grouped ? 3 : 2) || wa.ndim() != 2 || sa.ndim() != 2) {
      return nullptr;
    }
    const bool x_f16 = (xa.dtype() == mlx::core::float16);
    if ((!x_f16 && xa.dtype() != mlx::core::float32) ||
        wa.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int g = grouped ? xa.shape(1) : 1;
    const int k = grouped ? xa.shape(2) : xa.shape(1);
    if (m < 1 || g < 1 || k < 64 || k % 64 != 0 || wa.shape(0) < g ||
        wa.shape(0) % g != 0 || wa.shape(1) != k / 4) {
      return nullptr;
    }
    const int n = wa.shape(0) / g;
    if (n < 1 || sa.shape() != Shape{g * n, k / 32}) {
      return nullptr;
    }
    if (size_t(g) * size_t(k) > (1ULL << 25) ||
        size_t(g) * size_t(n) > (1ULL << 25)) {
      return nullptr;
    }
    const bool sym_f16 = (biases == nullptr);
    array bc = cached_zero_f16();
    if (sym_f16) {
      if (sa.dtype() != mlx::core::float16) {
        return nullptr;
      }
    } else {
      const auto &ba = *reinterpret_cast<array *>(biases);
      if (ba.ndim() != 2 || ba.shape() != sa.shape()) {
        return nullptr;
      }
      const bool both_f32 = (sa.dtype() == mlx::core::float32 &&
                             ba.dtype() == mlx::core::float32);
      const bool both_f16 = (sa.dtype() == mlx::core::float16 &&
                             ba.dtype() == mlx::core::float16);
      if (!both_f32 && !both_f16) {
        return nullptr;
      }
      bc = ensure_contiguous_aligned(ba, 1);
    }

    array auxc = cached_zero_f32();
    bool aux_f16 = false;
    if (epilogue == 1) {
      if (grouped || !aux) {
        return nullptr;
      }
      const auto &auxa = *reinterpret_cast<array *>(aux);
      aux_f16 = (auxa.dtype() == mlx::core::float16);
      if (auxa.ndim() != 2 ||
          (!aux_f16 && auxa.dtype() != mlx::core::float32) ||
          auxa.shape() != Shape{m, n} || auxa.size() < 8) {
        return nullptr;
      }
      auxc = ensure_strided_aligned(auxa, 1);
    } else if (epilogue == 2) {
      if (!grouped || !aux) {
        return nullptr;
      }
      const auto &auxa = *reinterpret_cast<array *>(aux);
      aux_f16 = (auxa.dtype() == mlx::core::float16);
      if ((!aux_f16 && auxa.dtype() != mlx::core::float32) ||
          auxa.size() < 8) {
        return nullptr;
      }
      if (auxa.ndim() == 2 && auxa.shape() == Shape{m, g * n}) {
        auxc = ensure_strided_aligned(auxa, 1);
      } else if (auxa.ndim() == 3 && auxa.shape() == Shape{m, g, n}) {
        auxc = reshape(ensure_contiguous_aligned(auxa, 1), {m, g * n});
      } else {
        return nullptr;
      }
    } else if (epilogue != 0) {
      return nullptr;
    }

    array xc = ensure_strided_aligned(xa, x_f16 ? 8 : 4);
    array wc = ensure_contiguous_aligned(wa, 4);
    array sc = ensure_contiguous_aligned(sa, 1);

    const bool fast_1mma =
        sym_f16 && q8_prefill_1mma_env() && (!grouped || q8_grouped_1mma_env());
    const int bm = (!sym_f16 || m <= 48 || (g == 1 && n <= 256)) ? 32 : 64;
    const int bn = (fast_1mma && bm == 64 && n >= 128) ? 128 : 64;
    const int num_sg = (fast_1mma && bn == 128) ? 8 : 4;
    const bool aligned_n = (n % bn == 0);
    const bool scales_f16 = (sa.dtype() == mlx::core::float16);
    const int grid_x = 32 * ((n + bn - 1) / bn);
    const int grid_y = num_sg * ((m + bm - 1) / bm);
    const int grid_z = g;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{grid_x, grid_y, grid_z};
    const std::tuple<int, int, int> tg{32, num_sg, 1};
    const Dtype compute_dtype = mlx::core::float16;
    const Dtype out_dtype = out_f16 ? mlx::core::float16 : mlx::core::float32;
    const uint64_t tmpl_key = uint64_t(bm) | (uint64_t(bn) << 8) |
                              (uint64_t(grouped ? 1 : 0) << 16) |
                              (uint64_t(sym_f16 ? 1 : 0) << 17) |
                              (uint64_t(fast_1mma ? 1 : 0) << 18) |
                              (uint64_t(aligned_n ? 1 : 0) << 19) |
                              (uint64_t(scales_f16 ? 1 : 0) << 20) |
                              (uint64_t(epilogue) << 21) |
                              (uint64_t(x_f16 ? 1 : 0) << 23) |
                              (uint64_t(aux_f16 ? 1 : 0) << 24) |
                              (uint64_t(out_f16 ? 1 : 0) << 25);
    const std::vector<array> inputs{xc, wc, sc, bc, auxc};
    const Shape out_shape = (grouped && epilogue == 2)
                                ? Shape{m, g * n}
                                : (grouped ? Shape{m, g, n} : Shape{m, n});
    const std::vector<Shape> out_shapes{out_shape};
    const std::vector<Dtype> out_dtypes{out_dtype};

    auto out = dispatch_cached_kernel(
        kKernelQ8Prefill, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_q8_prefill_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"M_TILE", bm},
               {"N_TILE", bn},
               {"GROUPED", grouped},
               {"SYM_F16", sym_f16},
               {"FAST_1MMA", fast_1mma},
               {"ALIGNED_N", aligned_n},
               {"EPILOGUE", epilogue},
               {"X_F16", x_f16},
               {"X_T", xa.dtype()},
               {"C", compute_dtype},
               {"OUT_T", out_dtype}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << (grouped ? "Hy4 q8_grouped_prefill: "
                          : "Hy4 q8_dense_prefill: ")
              << e.what() << std::endl;
    return nullptr;
  }
}

const std::string &hy4_kq_expert_header() {
  static const std::string header = qmv_header() +
#include "metal/hy4/kq_expert_decode.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_kq_gate_up_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_gate_up_swiglu",
      {"x", "row_ids", "slots", "wg", "sg", "bg", "wu", "su", "bu", "limit"},
      {"out"},
      R"METAL(
      const uint assign = threadgroup_position_in_grid.z;
      if (assign >= uint(slots_shape[0])) return;
      hy4_kq_gate_up_swiglu_impl<FAST, DEDUP>(
          x, row_ids, slots, wg, sg, bg, wu, su, bu, limit[0], out,
          x_shape[0], slots_shape[0], x_shape[1], wg_shape[1], wg_shape[0],
          assign, threadgroup_position_in_grid.y,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_kq_expert_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_kq_down_combine_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_down_combine", {"hidden", "slots", "weights", "wd", "sd", "bd"},
      {"out"},
      R"METAL(
      threadgroup float products[32 * RPS];
      hy4_kq_down_combine_impl<BITS, GS, SR, HM, FAST, RPS, KSG>(
          hidden, slots, weights, wd,
          reinterpret_cast<const device uint8_t*>(sd), bd, out, products,
          slots_shape[0], slots_shape[1], hidden_shape[1], wd_shape[1], wd_shape[0],
          threadgroup_position_in_grid.z, threadgroup_position_in_grid.y,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_kq_expert_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_kq_down_weighted_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_down_weighted", {"hidden", "slots", "weights", "wd", "sd", "bd"},
      {"out"},
      R"METAL(
      const uint assign = threadgroup_position_in_grid.z;
      if (assign >= uint(slots_shape[0])) return;
      hy4_kq_down_weighted_impl<BITS, GS, SR, HM, FAST, RPS>(
          hidden, slots[assign], weights[assign], wd,
          reinterpret_cast<const device uint8_t*>(sd), bd, out,
          slots_shape[0], hidden_shape[1], wd_shape[1], wd_shape[0],
          assign, threadgroup_position_in_grid.y,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_kq_expert_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_kq_prefill_indirect_gu_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_prefill_indirect_gu",
      {"x", "token_rows", "tiles", "wg", "sg", "bg", "wu", "su", "bu", "limit"},
      {"out"},
#include "metal/hy4/kq_prefill_indirect_gu.metal.inc"
      , nax_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_kq_sorted_combine_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_sorted_combine", {"down", "weights", "inverse"}, {"out"},
#include "metal/hy4/kq_sorted_combine.metal.inc"
  );
  return kernel;
}

const std::string &hy4_router_topk_header() {
  static const std::string header = R"METAL(
template <bool HAS_BIAS, typename LogitsPtr, typename BiasPtr>
inline void hy4_topk_load_one(
    LogitsPtr logits,
    BiasPtr bias,
    threadgroup float* tg_s,
    size_t row_off,
    int E,
    int e,
    thread float& vo,
    thread uint& io) {
  if (e < E) {
    const float lx = logits[row_off + size_t(e)];
    // Match MLX Sigmoid (unary_ops.h) so GPU routing matches host routing bit-for-bit.
    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(lx)));
    const float s = (lx < 0.0f) ? y : (1.0f - y);
    float sel = HAS_BIAS ? (s + bias[e]) : s;
    if (metal::isnan(sel)) {
      sel = -INFINITY;
    }
    tg_s[e] = s;
    vo = sel;
    io = uint(e);
  }
}

inline bool hy4_topk_better(float va, uint ia, float vb, uint ib) {
  return (ia != 0xFFFFFFFFu) &&
         (ib == 0xFFFFFFFFu || va > vb || (va == vb && ia < ib));
}

inline void hy4_topk_cswap(
    thread float& va,
    thread uint& ia,
    thread float& vb,
    thread uint& ib) {
  if (hy4_topk_better(vb, ib, va, ia)) {
    const float tv = va; va = vb; vb = tv;
    const uint ti = ia; ia = ib; ib = ti;
  }
}

inline void hy4_topk_bitonic_merge32(
    thread float& val,
    thread uint& id,
    uint lane) {
#pragma unroll
  for (uint s = 16u; s >= 1u; s >>= 1u) {
    const float ov = simd_shuffle_xor(val, ushort(s));
    const uint oi = simd_shuffle_xor(id, ushort(s));
    const bool upper = (lane & s) != 0u;
    const bool o_better = hy4_topk_better(ov, oi, val, id);
    if (upper ? !o_better : o_better) {
      val = ov;
      id = oi;
    }
  }
}
)METAL";
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_router_topk_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_router_topk", {"logits", "bias", "params"}, {"ids", "weights"},
#include "metal/hy4/router_topk.metal.inc"
      , hy4_router_topk_header());
  return kernel;
}

const std::string &hy4_gguf_decode_header() {
  static const std::string header =
#include "metal/hy4/gguf_expert_decode.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_gguf_dedup_slots_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_gguf_dedup_slots", {"slots"}, {"dedup"},
      R"METAL(
      constexpr uint NSG_DD = uint(TG_SIZE / 32);
      threadgroup uint tg_slots[TG_SIZE];
      threadgroup uint tg_sg_counts[4][NSG_DD];
      const uint tid = thread_index_in_threadgroup;
      const uint lane = thread_index_in_simdgroup;
      const uint sg = simdgroup_index_in_threadgroup;
      const uint n_assign = uint(slots_shape[0]);
      if (tid < n_assign) {
        tg_slots[tid] = slots[tid];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      uint my_slot = 0u;
      uint rank = 0u;
      if (tid < n_assign) {
        my_slot = tg_slots[tid];
        for (uint b = 0u; b < tid; ++b) {
          rank += uint(tg_slots[b] == my_slot);
        }
      }

      // Start a new chunk every 4 occurrences so every compacted entry has 1..=4 matches.
      const bool is_chunk = (tid < n_assign && (rank & 3u) == 0u);
      uint cnt = 0u;
      uint a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
      uint r0 = 0u, r1 = 0u, r2 = 0u, r3 = 0u;
      if (is_chunk) {
        for (uint b = tid; b < n_assign && cnt < 4u; ++b) {
          if (tg_slots[b] == my_slot) {
            const uint rid = b / uint(TOP);
            if (cnt == 0u) { a0 = b; r0 = rid; }
            else if (cnt == 1u) { a1 = b; r1 = rid; }
            else if (cnt == 2u) { a2 = b; r2 = rid; }
            else { a3 = b; r3 = rid; }
            ++cnt;
          }
        }
      }

      // Bucket sort descending by cnt (4, 3, 2, 1) to balance strided SG loops in down_combine.
      const uint bucket = is_chunk ? (4u - cnt) : 4u;
      uint my_sg_off = 0u;
#pragma unroll
      for (uint bk = 0u; bk < 4u; ++bk) {
        const uint flag = (bucket == bk) ? 1u : 0u;
        const uint sg_off = simd_prefix_exclusive_sum(flag);
        const uint sg_tot = simd_sum(flag);
        if (bucket == bk) {
          my_sg_off = sg_off;
        }
        if (lane == 0u) {
          tg_sg_counts[bk][sg] = sg_tot;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      uint bucket_tot[4];
#pragma unroll
      for (uint bk = 0u; bk < 4u; ++bk) {
        uint s = 0u;
        for (uint j = 0u; j < NSG_DD; ++j) {
          s += tg_sg_counts[bk][j];
        }
        bucket_tot[bk] = s;
      }

      device uint4* out4 = reinterpret_cast<device uint4*>(dedup);
      if (tid == 0u) {
        const uint total_uniq =
            bucket_tot[0] + bucket_tot[1] + bucket_tot[2] + bucket_tot[3];
        out4[0] = uint4(total_uniq, 0u, 0u, 0u);
      }
      if (is_chunk) {
        uint base = 0u;
        for (uint bk = 0u; bk < bucket; ++bk) {
          base += bucket_tot[bk];
        }
        for (uint j = 0u; j < sg; ++j) {
          base += tg_sg_counts[bucket][j];
        }
        const uint py = (a0 & 0x3FFu) | ((a1 & 0x3FFu) << 10) |
                        ((a2 & 0x3FFu) << 20) | ((cnt - 1u) << 30);
        const uint pz = (a3 & 0x3FFu) | ((r0 & 0x3FFu) << 10) |
                        ((r1 & 0x3FFu) << 20);
        const uint pw = (r2 & 0x3FFu) | ((r3 & 0x3FFu) << 10);
        out4[1u + base + my_sg_off] = uint4(my_slot, py, pz, pw);
      }
      )METAL");
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_gguf_gate_up_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_gguf_gate_up_swiglu",
      {"x", "row_ids", "slots", "dedup", "gate_buf", "up_buf", "offsets",
       "dims", "limit"},
      {"out"},
      R"METAL(
      const uint assign = threadgroup_position_in_grid.z;
      if (assign >= uint(slots_shape[0])) return;
      hy4_gguf_gate_up_swiglu_impl<FAST, DEDUP, MAX_MATCH, NSG, RPS>(
          x, row_ids, slots, dedup, gate_buf, offsets[0], up_buf, offsets[1],
          limit[0], out, x_shape[0], slots_shape[0], x_shape[1],
          size_t(x_strides[0]), dims[1], dims[0], assign,
          threadgroup_position_in_grid.y, simdgroup_index_in_threadgroup,
          thread_index_in_simdgroup);
      )METAL",
      hy4_gguf_decode_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_gguf_down_proj_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_gguf_down_proj",
      {"hidden", "slots", "weights", "dedup", "down_buf", "down_off", "dims"},
      {"partial"},
      R"METAL(
      const uint assign = threadgroup_position_in_grid.z;
      if (assign >= uint(slots_shape[0])) return;
      hy4_gguf_down_proj_impl<BITS, FAST, DEDUP, MAX_MATCH, NSG, RPS>(
          hidden, slots, weights, dedup, down_buf, down_off[0], partial,
          slots_shape[0], hidden_shape[1], dims[1], dims[0], assign,
          threadgroup_position_in_grid.y, simdgroup_index_in_threadgroup,
          thread_index_in_simdgroup);
      )METAL",
      hy4_gguf_decode_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_gguf_down_reduce_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_gguf_down_reduce",
      {"partial"},
      {"out"},
      R"METAL(
      hy4_gguf_down_reduce_impl<TOP, VEC4>(
          partial, out, partial_shape[0] / TOP, partial_shape[1],
          thread_position_in_grid.x, thread_position_in_grid.y);
      )METAL",
      hy4_gguf_decode_header());
  return kernel;
}

const std::string &hy4_affine4_decode_header() {
  static const std::string header =
#include "metal/hy4/affine4_expert_decode.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_affine4_gate_up_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_affine4_gate_up_swiglu",
      {"x", "row_ids", "slots", "dedup", "wg", "sg", "bg", "wu", "su", "bu",
       "limit"},
      {"out"},
      R"METAL(
      const uint assign = threadgroup_position_in_grid.z;
      if (assign >= uint(slots_shape[0])) return;
      hy4_affine4_gate_up_swiglu_impl<FAST_K, FAST_M, DEDUP, MAX_MATCH, NSG, RPS>(
          x, row_ids, slots, dedup, wg, sg, bg, wu, su, bu, limit[0], out,
          x_shape[0], slots_shape[0], x_shape[1], size_t(x_strides[0]),
          wg_shape[1], wg_shape[0], assign, threadgroup_position_in_grid.y,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_affine4_decode_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_affine4_down_combine_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_affine4_down_combine",
      {"hidden", "slots", "weights", "dedup", "wd", "sd", "bd"},
      {"out"},
      R"METAL(
      constexpr int PROD_CAP = DEDUP_ROWS ? 128 : 32;
      threadgroup float products[PROD_CAP * 4];
      hy4_affine4_down_combine_impl<TOP, FAST_M, FAST_H, DEDUP_ROWS, MAX_MATCH>(
          hidden, slots, weights, dedup, wd, sd, bd, out, products,
          slots_shape[0], hidden_shape[1], wd_shape[1], wd_shape[0],
          threadgroup_position_in_grid.z, threadgroup_position_in_grid.y,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_affine4_decode_header());
  return kernel;
}

bool validate_affine4_experts(const array &wg, const array &sg, const array &bg,
                              const array &wu, const array &su, const array &bu,
                              const array &wd, const array &sd, const array &bd,
                              int k, int &out_e, int &out_m, int &out_h) {
  if (k < 64 || k % 64 != 0) {
    return false;
  }
  if (wg.ndim() != 3 || sg.ndim() != 3 || bg.ndim() != 3 || wu.ndim() != 3 ||
      su.ndim() != 3 || bu.ndim() != 3 || wd.ndim() != 3 || sd.ndim() != 3 ||
      bd.ndim() != 3) {
    return false;
  }
  if (wg.dtype() != mlx::core::uint32 || wu.dtype() != mlx::core::uint32 ||
      wd.dtype() != mlx::core::uint32) {
    return false;
  }
  if (sg.dtype() != mlx::core::bfloat16 || bg.dtype() != mlx::core::bfloat16 ||
      su.dtype() != mlx::core::bfloat16 || bu.dtype() != mlx::core::bfloat16 ||
      sd.dtype() != mlx::core::bfloat16 || bd.dtype() != mlx::core::bfloat16) {
    return false;
  }
  const int e = wg.shape(0);
  const int m = wg.shape(1);
  const int h = wd.shape(1);
  if (e < 1 || m < 64 || m % 64 != 0 || h < 1) {
    return false;
  }
  const Shape gu_w{e, m, k / 8};
  const Shape gu_s{e, m, k / 64};
  const Shape d_w{e, h, m / 8};
  const Shape d_s{e, h, m / 64};
  if (wg.shape() != gu_w || wu.shape() != gu_w || sg.shape() != gu_s ||
      bg.shape() != gu_s || su.shape() != gu_s || bu.shape() != gu_s ||
      wd.shape() != d_w || sd.shape() != d_s || bd.shape() != d_s) {
    return false;
  }
  out_e = e;
  out_m = m;
  out_h = h;
  return true;
}

const std::string &hy4_gguf_nax_header() {
  static const std::string header =
      std::string(mlx::core::quantized_preamble::nax()) +
      hy4_q8_prefill_swz_header() + R"METAL(
using namespace mlx::steel;

inline uint32_t hy4_gguf_sc_byte(const thread uint32_t* sc_w, int idx) {
  return (sc_w[idx >> 2] >> ((idx & 3) * 8)) & 0xFFu;
}

inline void hy4_gguf_q4k_group_scale(
    const thread uint32_t* sc_w,
    float blk_d,
    float blk_dmin,
    int g,
    thread float& d_g,
    thread float& m_g) {
  uint32_t sc, m;
  if (g < 4) {
    sc = hy4_gguf_sc_byte(sc_w, g) & 63u;
    m = hy4_gguf_sc_byte(sc_w, g + 4) & 63u;
  } else {
    const uint32_t q4 = hy4_gguf_sc_byte(sc_w, g + 4);
    sc = (q4 & 0x0Fu) | ((hy4_gguf_sc_byte(sc_w, g - 4) >> 6) << 4);
    m = (q4 >> 4) | ((hy4_gguf_sc_byte(sc_w, g) >> 6) << 4);
  }
  d_g = blk_d * float(sc);
  m_g = -blk_dmin * float(m);
}
)METAL";
  return header;
}

const mlx::core::fast::CustomKernelFunction &
hy4_gguf_prefill_indirect_gu_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_gguf_prefill_indirect_gu",
      {"x", "token_rows", "tiles", "gate_buf", "up_buf", "offsets", "dims",
       "limit"},
      {"out"},
#include "metal/hy4/gguf_prefill_indirect_gu.metal.inc"
      , hy4_gguf_nax_header(), /*ensure_row_contiguous=*/false);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_gguf_prefill_indirect_down_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_gguf_prefill_indirect_down",
      {"hidden", "tiles", "down_buf", "down_off", "dims"}, {"out"},
#include "metal/hy4/gguf_prefill_indirect_down.metal.inc"
      , hy4_gguf_nax_header());
  return kernel;
}

bool validate_kq_banks(const array &wg, const array &sg, const array &bg,
                       const array &wu, const array &su, const array &bu,
                       const array &wd, const array &sd, const array &bd,
                       int k, int down_bits, int &out_s, int &out_m,
                       int &out_h) {
  if (wg.ndim() != 3 || sg.ndim() != 3 || bg.ndim() != 3 || wu.ndim() != 3 ||
      su.ndim() != 3 || bu.ndim() != 3 || wd.ndim() != 3 || sd.ndim() != 3 ||
      bd.ndim() != 3) {
    return false;
  }
  const int s = wg.shape(0);
  const int m = wg.shape(1);
  const int h = wd.shape(1);
  if (s < 1 || m < 256 || m % 256 != 0 || h < 1 || k < 256 || k % 256 != 0) {
    return false;
  }
  const Shape q4_w{s, m, k / 8};
  const Shape q4_s{s, m, k / 16};
  const Shape q4_b{s, m, k / 128};
  if (wg.dtype() != mlx::core::uint32 || wg.shape() != q4_w ||
      sg.dtype() != mlx::core::uint8 || sg.shape() != q4_s ||
      bg.dtype() != mlx::core::float16 || bg.shape() != q4_b ||
      wu.dtype() != mlx::core::uint32 || wu.shape() != q4_w ||
      su.dtype() != mlx::core::uint8 || su.shape() != q4_s ||
      bu.dtype() != mlx::core::float16 || bu.shape() != q4_b) {
    return false;
  }
  if (down_bits == 4) {
    if (wd.dtype() != mlx::core::uint32 || wd.shape() != Shape{s, h, m / 8} ||
        sd.dtype() != mlx::core::uint8 || sd.shape() != Shape{s, h, m / 16} ||
        bd.dtype() != mlx::core::float16 ||
        bd.shape() != Shape{s, h, m / 128}) {
      return false;
    }
  } else if (down_bits == 6) {
    if (wd.dtype() != mlx::core::uint32 ||
        wd.shape() != Shape{s, h, m * 6 / 32} ||
        sd.dtype() != mlx::core::int8 || sd.shape() != Shape{s, h, m / 16} ||
        bd.dtype() != mlx::core::float16 ||
        bd.shape() != Shape{s, h, m / 256}) {
      return false;
    }
  } else {
    return false;
  }
  out_s = s;
  out_m = m;
  out_h = h;
  return true;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_sink_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_sink_attention", {"q", "k0", "k1", "idx", "sinks", "scale"},
      {"out"},
#include "metal/hy4/mla_sink_attention.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_splitk_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_decode_splitk", {"q", "k0", "k1", "idx", "scale", "split_cfg"},
      {"o_part", "max_part", "sum_part"},
#include "metal/hy4/mla_decode_splitk.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_decode_v2_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_decode_v2", {"q", "k0", "k1", "idx", "scale", "split_cfg"},
      {"o_part", "max_part", "sum_part"},
#include "metal/hy4/mla_decode_v2.metal.inc"
      , "#include <metal_simdgroup_matrix>\n");
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_merge_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_decode_merge", {"o_part", "max_part", "sum_part", "sinks"},
      {"out"},
#include "metal/hy4/mla_decode_merge.metal.inc"
  );
  return kernel;
}

const std::string &hy4_mla_prefill_nax_header() {
  static const std::string header =
      std::string(mlx::core::quantized_preamble::nax()) + R"METAL(
using namespace mlx::steel;

struct Hy4MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct Hy4SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct Hy4MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct Hy4Exp2SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::exp2(x - y);
  }
};

template <typename C, int KV_STRIDE>
METAL_FUNC void hy4_load_kv_frag_tg(
    thread metal::vec<C, 8>& dst,
    const threadgroup half* s_kv,
    const int col_off,
    const short fm,
    const short fn) {
  const half4 va = *reinterpret_cast<const threadgroup half4*>(
      s_kv + int(fm) * KV_STRIDE + col_off + int(fn));
  const half4 vb = *reinterpret_cast<const threadgroup half4*>(
      s_kv + (int(fm) + 8) * KV_STRIDE + col_off + int(fn));
  dst[0] = C(va.x);
  dst[1] = C(va.y);
  dst[2] = C(va.z);
  dst[3] = C(va.w);
  dst[4] = C(vb.x);
  dst[5] = C(vb.y);
  dst[6] = C(vb.z);
  dst[7] = C(vb.w);
}

METAL_FUNC int hy4_kv_swz_c8(int c8, int r) {
  return c8 ^ ((r & 3) << 1) ^ ((r & 4) >> 2);
}

template <typename C, int KV_COLS>
METAL_FUNC void hy4_load_kv_frag_tg_swz(
    thread metal::vec<C, 8>& dst,
    const threadgroup half* s_kv,
    const int col_off,
    const short fm,
    const short fn) {
  const int c = col_off + int(fn);
  const int c8 = c >> 3;
  const int sub4 = c & 4;
  const int r0 = int(fm);
  const int r1 = r0 + 8;
  const int sw = hy4_kv_swz_c8(c8, r0) << 3;
  const half4 va = *reinterpret_cast<const threadgroup half4*>(
      s_kv + r0 * KV_COLS + sw + sub4);
  const half4 vb = *reinterpret_cast<const threadgroup half4*>(
      s_kv + r1 * KV_COLS + sw + sub4);
  dst[0] = C(va.x);
  dst[1] = C(va.y);
  dst[2] = C(va.z);
  dst[3] = C(va.w);
  dst[4] = C(vb.x);
  dst[5] = C(vb.y);
  dst[6] = C(vb.z);
  dst[7] = C(vb.w);
}

template <bool SPLIT, int D_K, int HALF>
METAL_FUNC void hy4_load_kv_frag_dev(
    thread metal::vec<half, 8>& dst,
    const device half* k0,
    const device half* k1,
    const int slot_a,
    const int slot_b,
    const int col_off,
    const short fn) {
  const device half* plane = (SPLIT && col_off >= HALF) ? k1 : k0;
  const int stride = SPLIT ? HALF : D_K;
  const int pcol = (SPLIT && col_off >= HALF) ? (col_off - HALF) : col_off;
  const half4 va = (slot_a >= 0)
      ? *reinterpret_cast<const device half4*>(
            plane + size_t(slot_a) * size_t(stride) + size_t(pcol + int(fn)))
      : half4(0.0h);
  const half4 vb = (slot_b >= 0)
      ? *reinterpret_cast<const device half4*>(
            plane + size_t(slot_b) * size_t(stride) + size_t(pcol + int(fn)))
      : half4(0.0h);
  dst[0] = va.x;
  dst[1] = va.y;
  dst[2] = va.z;
  dst[3] = va.w;
  dst[4] = vb.x;
  dst[5] = vb.y;
  dst[6] = vb.z;
  dst[7] = vb.w;
}

template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC constexpr void hy4_nax_mma_16x16x32(
    thread BaseNAXFrag::dtype_frag_t<CType>& C,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A0,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A1,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B1,
    metal::bool_constant<transpose_b>) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16,
      16,
      32,
      transpose_a,
      transpose_b,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      CType>();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; ++i) {
    ct_a[i] = A0[i];
    ct_a[BaseNAXFrag::kElemsPerFrag + i] = A1[i];
    ct_b[i] = B0[i];
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = B1[i];
    ct_c[i] = C[i];
  }
  gemm_op.run(ct_a, ct_b, ct_c);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; ++i) {
    C[i] = ct_c[i];
  }
}

template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC constexpr void hy4_nax_mma_16x16x16(
    thread BaseNAXFrag::dtype_frag_t<CType>& C,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B,
    metal::bool_constant<transpose_b>) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      transpose_a,
      transpose_b,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      CType>();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; ++i) {
    ct_a[i] = A[i];
    ct_b[i] = B[i];
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = BType(0);
    ct_c[i] = C[i];
    ct_c[BaseNAXFrag::kElemsPerFrag + i] = CType(0);
  }
  gemm_op.run(ct_a, ct_b, ct_c);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; ++i) {
    C[i] = ct_c[i];
  }
}
)METAL";
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_prefill_nax_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_prefill_nax", {"q", "q_pe", "k0", "k1", "idx", "sinks", "scale"},
      {"out"},
#include "metal/hy4/mla_prefill_nax.metal.inc"
      , hy4_mla_prefill_nax_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_decode_nax_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_decode_nax",
      {"q", "k0", "k1", "idx", "sinks", "scale", "split_cfg"},
      {"o_part", "max_part", "sum_part"},
#include "metal/hy4/mla_decode_nax.metal.inc"
      , hy4_mla_prefill_nax_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_mla_prefill_sliding_nax_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_prefill_sliding_nax",
      {"q", "q_pe", "k", "sinks", "scale", "params"}, {"out"},
#include "metal/hy4/mla_prefill_sliding_nax.metal.inc"
      , hy4_mla_prefill_nax_header());
  return kernel;
}

array hy4_mla_single_pass(const array &q, const array &k0, const array &k1,
                          const array &idx, const array &sinks,
                          const array &scale, int d_v, int d_rope, bool split,
                          int heads, int bn, const Stream &stream) {
  const int s = q.shape(0);
  const int h = q.shape(1);
  const int tg_threads = bn * 32;
  const int head_groups = h / heads;
  const std::tuple<int, int, int> grid{tg_threads, head_groups, s};
  const std::tuple<int, int, int> tg{tg_threads, 1, 1};
  const uint64_t tmpl_key = uint64_t(d_v) | (uint64_t(d_rope) << 16) |
                            (uint64_t(split ? 1 : 0) << 32) |
                            (uint64_t(heads) << 36) | (uint64_t(bn) << 44);
  const std::vector<array> inputs{q, k0, k1, idx, sinks, scale};
  const std::vector<Shape> out_shapes{{s, h, d_v}};
  const std::vector<Dtype> out_dtypes{mlx::core::float32};
  return dispatch_cached_kernel(
      kKernelMlaSink, stream, grid, tg, tmpl_key, inputs, out_shapes,
      out_dtypes, [&] {
        return hy4_mla_sink_kernel()(
            inputs, out_shapes, out_dtypes, grid, tg,
            {{"D_V", d_v},
             {"D_ROPE", d_rope},
             {"SPLIT", split},
             {"HEADS", heads},
             {"BN", bn}},
            std::nullopt, false, stream);
      })[0];
}

std::vector<array> hy4_mla_splitk_pass1(const array &q, const array &k0,
                                        const array &k1, const array &idx,
                                        const array &scale,
                                        const array &split_cfg, int d_v,
                                        int d_rope, bool split, int heads,
                                        int bn, int n_splits,
                                        const Stream &stream) {
  const int s = q.shape(0);
  const int h = q.shape(1);
  const int tg_threads = bn * 32;
  const int head_groups = h / heads;
  const std::tuple<int, int, int> grid{tg_threads, head_groups * n_splits, s};
  const std::tuple<int, int, int> tg{tg_threads, 1, 1};
  const uint64_t tmpl_key = uint64_t(d_v) | (uint64_t(d_rope) << 16) |
                            (uint64_t(split ? 1 : 0) << 32) |
                            (uint64_t(heads) << 36) | (uint64_t(bn) << 44);
  const std::vector<array> inputs{q, k0, k1, idx, scale, split_cfg};
  const std::vector<Shape> out_shapes{
      {s, h, n_splits, d_v}, {s, h, n_splits}, {s, h, n_splits}};
  const std::vector<Dtype> out_dtypes{
      mlx::core::float32, mlx::core::float32, mlx::core::float32};
  return dispatch_cached_kernel(
      kKernelMlaSplitK, stream, grid, tg, tmpl_key, inputs, out_shapes,
      out_dtypes, [&] {
        return hy4_mla_splitk_kernel()(
            inputs, out_shapes, out_dtypes, grid, tg,
            {{"D_V", d_v},
             {"D_ROPE", d_rope},
             {"SPLIT", split},
             {"HEADS", heads},
             {"BN", bn}},
            std::nullopt, false, stream);
      });
}

std::vector<array> hy4_mla_decode_v2_pass1(const array &q, const array &k0,
                                           const array &k1, const array &idx,
                                           const array &scale,
                                           const array &split_cfg, bool split,
                                           int n_splits, const Stream &stream) {
  const int s = q.shape(0);
  const int h = q.shape(1);
  const int head_groups = h / 16;
  const std::tuple<int, int, int> grid{256, head_groups * n_splits, s};
  const std::tuple<int, int, int> tg{256, 1, 1};
  const uint64_t tmpl_key = uint64_t(split ? 1 : 0);
  const std::vector<array> inputs{q, k0, k1, idx, scale, split_cfg};
  const std::vector<Shape> out_shapes{
      {s, h, n_splits, 512}, {s, h, n_splits}, {s, h, n_splits}};
  const std::vector<Dtype> out_dtypes{
      mlx::core::float32, mlx::core::float32, mlx::core::float32};
  return dispatch_cached_kernel(
      kKernelMlaDecodeV2, stream, grid, tg, tmpl_key, inputs, out_shapes,
      out_dtypes, [&] {
        return hy4_mla_decode_v2_kernel()(
            inputs, out_shapes, out_dtypes, grid, tg, {{"SPLIT", split}},
            std::nullopt, false, stream);
      });
}

std::vector<array> hy4_mla_decode_nax_pass1(const array &q, const array &k0,
                                            const array &k1, const array &idx,
                                            const array &sinks,
                                            const array &scale,
                                            const array &split_cfg, bool split,
                                            int n_splits, bool single_split,
                                            const Stream &stream) {
  const int s = q.shape(0);
  const int h = q.shape(1);
  constexpr int wn = 8;
  const int head_groups = h / 16;
  const std::tuple<int, int, int> grid{32, head_groups * n_splits * wn, s};
  const std::tuple<int, int, int> tg{32, wn, 1};
  const uint64_t tmpl_key = uint64_t(split ? 1 : 0) |
                            (uint64_t(single_split ? 1 : 0) << 1);
  const std::vector<array> inputs{q, k0, k1, idx, sinks, scale, split_cfg};
  const std::vector<Shape> out_shapes =
      single_split
          ? std::vector<Shape>{{s, h, 512}, {1}, {1}}
          : std::vector<Shape>{
                {s, h, n_splits, 512}, {s, h, n_splits}, {s, h, n_splits}};
  const std::vector<Dtype> out_dtypes{
      mlx::core::float32, mlx::core::float32, mlx::core::float32};
  return dispatch_cached_kernel(
      kKernelMlaDecodeNax, stream, grid, tg, tmpl_key, inputs, out_shapes,
      out_dtypes, [&] {
        return hy4_mla_decode_nax_kernel()(
            inputs, out_shapes, out_dtypes, grid, tg,
            {{"SPLIT", split}, {"SINGLE_SPLIT", single_split}}, std::nullopt,
            false, stream);
      });
}

array hy4_mla_splitk_merge(const array &o_part, const array &max_part,
                           const array &sum_part, const array &sinks, int d_v,
                           const Stream &stream) {
  const int s = o_part.shape(0);
  const int h = o_part.shape(1);
  const int tg_x = d_v / 4;
  const std::tuple<int, int, int> grid{tg_x, h, s};
  const std::tuple<int, int, int> tg{tg_x, 1, 1};
  const uint64_t tmpl_key = uint64_t(d_v);
  const std::vector<array> inputs{o_part, max_part, sum_part, sinks};
  const std::vector<Shape> out_shapes{{s, h, d_v}};
  const std::vector<Dtype> out_dtypes{mlx::core::float32};
  return dispatch_cached_kernel(
      kKernelMlaMerge, stream, grid, tg, tmpl_key, inputs, out_shapes,
      out_dtypes, [&] {
        return hy4_mla_merge_kernel()(
            inputs, out_shapes, out_dtypes, grid, tg, {{"D_V", d_v}},
            std::nullopt, false, stream);
      })[0];
}

int hy4_mla_pipeline_threads(const array &output) {
  thread_local std::unordered_map<std::string, int> cache;
  if (!output.has_primitive()) {
    return 0;
  }
  auto *custom =
      dynamic_cast<mlx::core::fast::CustomKernel *>(&output.primitive());
  if (!custom) {
    return 0;
  }
  const std::string &key = custom->library_name();
  if (auto found = cache.find(key); found != cache.end()) {
    return found->second;
  }
  int threads = 0;
  try {
    const auto state = custom->state();
    const auto &name = std::get<0>(state);
    const auto &source = std::get<1>(state);
    auto &device = mlx::core::metal::device(custom->stream().device);
    auto *library = device.get_library(
        key, mlx::core::CompileOptions(std::get<10>(state)),
        [&source] { return mlx::core::metal::utils() + source; });
    auto *pipeline = device.get_kernel(name, library);
    if (pipeline->threadExecutionWidth() == 32 &&
        pipeline->staticThreadgroupMemoryLength() <=
            device.mtl_device()->maxThreadgroupMemoryLength()) {
      threads = int(pipeline->maxTotalThreadsPerThreadgroup());
    }
  } catch (const std::exception &e) {
    std::cerr << "Hy4 mla_sink_attention: pipeline " << key << ": "
              << e.what() << std::endl;
    return 0;
  }
  cache.emplace(key, threads);
  return threads;
}

struct MlaConfig {
  int heads;
  int bn;
};

std::unordered_map<uint64_t, MlaConfig> &mla_config_cache() {
  thread_local std::unordered_map<uint64_t, MlaConfig> cache;
  return cache;
}

bool lookup_mla_config(int h, bool split, bool splitk, MlaConfig &out_cfg) {
  const uint64_t key = uint64_t(uint32_t(h)) |
                       (uint64_t(split ? 1u : 0u) << 32) |
                       (uint64_t(splitk ? 1u : 0u) << 33);
  auto &cache = mla_config_cache();
  if (auto it = cache.find(key); it != cache.end()) {
    out_cfg = it->second;
    return true;
  }
  return false;
}

void store_mla_config(int h, bool split, bool splitk, MlaConfig cfg) {
  const uint64_t key = uint64_t(uint32_t(h)) |
                       (uint64_t(split ? 1u : 0u) << 32) |
                       (uint64_t(splitk ? 1u : 0u) << 33);
  mla_config_cache().emplace(key, cfg);
}
} // namespace
#endif

extern "C" {

bool mlx_hy4_hc_mix_reduce(mlx_array *x, mlx_array *weight, mlx_array *scale,
                           mlx_array *base, float norm_eps, float hc_eps,
                           float magnitude, bool has_post, mlx_array **out_h,
                           mlx_array **out_post) {
  if (out_h)
    *out_h = nullptr;
  if (out_post)
    *out_post = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return false;
    }
    if (!x || !weight || !scale || !base || !out_h ||
        (has_post && !out_post)) {
      return false;
    }
    if (!std::isfinite(norm_eps) || !std::isfinite(hc_eps) ||
        !std::isfinite(magnitude)) {
      return false;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(weight);
    const auto &sa = *reinterpret_cast<array *>(scale);
    const auto &ba = *reinterpret_cast<array *>(base);
    if (xa.ndim() != 3 || wa.ndim() != 2 || sa.ndim() != 1 || ba.ndim() != 1) {
      return false;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::float32 ||
        sa.dtype() != mlx::core::float32 || ba.dtype() != mlx::core::float32) {
      return false;
    }
    const int m = xa.shape(0);
    const int hc = xa.shape(1);
    const int d = xa.shape(2);
    if (m <= 0 || hc <= 0 || hc > 16 || d <= 0) {
      return false;
    }
    const int n_gates = has_post ? 2 * hc : hc;
    const int n_scale = has_post ? 2 : 1;
    if (wa.shape(0) != n_gates || wa.shape(1) != hc * d ||
        sa.shape(0) != n_scale || ba.shape(0) != n_gates) {
      return false;
    }
    const bool vec4 = (d % 4 == 0);
    const size_t align = vec4 ? 4 : 1;
    array xc = ensure_contiguous_aligned(xa, align);
    array wc = ensure_contiguous_aligned(wa, align);
    array sc = ensure_contiguous_aligned(sa, 1);
    array bc = ensure_contiguous_aligned(ba, 1);
    array params = cached_f32_triple(norm_eps, hc_eps, magnitude);

    const Shape post_shape = has_post ? Shape{m, hc} : Shape{1};
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{256, m, 1};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(hc) | (uint64_t(has_post ? 1 : 0) << 8) |
                              (uint64_t(vec4 ? 1 : 0) << 9);
    const std::vector<array> inputs{xc, wc, sc, bc, params};
    const std::vector<Shape> out_shapes{{m, d}, post_shape};
    const std::vector<Dtype> out_dtypes{mlx::core::float32, mlx::core::float32};
    auto res = dispatch_cached_kernel(
        kKernelHcMixReduce, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_hc_mix_reduce_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"HC", hc}, {"HAS_POST", has_post}, {"VEC4", vec4}},
              std::nullopt, false, stream);
        });
    auto h = std::make_unique<array>(std::move(res[0]));
    if (has_post) {
      auto p = std::make_unique<array>(std::move(res[1]));
      *out_post = reinterpret_cast<mlx_array *>(p.release());
    }
    *out_h = reinterpret_cast<mlx_array *>(h.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 hc_mix_reduce: " << e.what() << std::endl;
    return false;
  }
#else
  (void)x;
  (void)weight;
  (void)scale;
  (void)base;
  (void)norm_eps;
  (void)hc_eps;
  (void)magnitude;
  (void)has_post;
  return false;
#endif
}

mlx_array *mlx_hy4_hc_post_identity(mlx_array *y, mlx_array *res,
                                    mlx_array *post) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!y || !res || !post) {
      return nullptr;
    }
    const auto &ya = *reinterpret_cast<array *>(y);
    const auto &ra = *reinterpret_cast<array *>(res);
    const auto &pa = *reinterpret_cast<array *>(post);
    if (ya.ndim() != 2 || ra.ndim() != 3 || pa.ndim() != 2) {
      return nullptr;
    }
    if (ya.dtype() != mlx::core::float32 || ra.dtype() != mlx::core::float32 ||
        pa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int m = ra.shape(0);
    const int hc = ra.shape(1);
    const int d = ra.shape(2);
    if (m <= 0 || hc <= 0 || hc > 16 || d <= 0 || ya.shape(0) != m ||
        ya.shape(1) != d || pa.shape(0) != m || pa.shape(1) != hc) {
      return nullptr;
    }
    const bool vec4 = (d % 4 == 0);
    const size_t align = vec4 ? 4 : 1;
    array yc = ensure_contiguous_aligned(ya, align);
    array rc = ensure_contiguous_aligned(ra, align);
    array pc = ensure_contiguous_aligned(pa, 1);

    const int cols = vec4 ? (d / 4) : d;
    const int tg_x = std::min(256, cols);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{cols, m, 1};
    const std::tuple<int, int, int> tg{tg_x, 1, 1};
    const uint64_t tmpl_key = uint64_t(hc) | (uint64_t(vec4 ? 1 : 0) << 8);
    const std::vector<array> inputs{yc, rc, pc};
    const std::vector<Shape> out_shapes{{m, hc, d}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelHcPostIdentity, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_hc_post_identity_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"HC", hc}, {"VEC4", vec4}}, std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 hc_post_identity: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)y;
  (void)res;
  (void)post;
  return nullptr;
#endif
}

bool mlx_hy4_hc_mix_reduce_norm(mlx_array *x, mlx_array *weight,
                                mlx_array *scale, mlx_array *base,
                                mlx_array *norm_w, float hc_norm_eps,
                                float hc_eps, float magnitude,
                                float out_norm_eps, bool head_only,
                                mlx_array **out_h_norm, mlx_array **out_post) {
  if (out_h_norm)
    *out_h_norm = nullptr;
  if (out_post)
    *out_post = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return false;
    }
    if (!x || !weight || !scale || !base || !norm_w || !out_h_norm ||
        (!head_only && !out_post)) {
      return false;
    }
    if (!std::isfinite(hc_norm_eps) || !std::isfinite(hc_eps) ||
        !std::isfinite(magnitude) || !std::isfinite(out_norm_eps)) {
      return false;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(weight);
    const auto &sa = *reinterpret_cast<array *>(scale);
    const auto &ba = *reinterpret_cast<array *>(base);
    const auto &nwa = *reinterpret_cast<array *>(norm_w);
    if (xa.ndim() != 3 || wa.ndim() != 2 || sa.ndim() != 1 || ba.ndim() != 1 ||
        nwa.ndim() != 1) {
      return false;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::float32 ||
        sa.dtype() != mlx::core::float32 || ba.dtype() != mlx::core::float32 ||
        nwa.dtype() != mlx::core::float32) {
      return false;
    }
    const int m = xa.shape(0);
    const int hc = xa.shape(1);
    const int d = xa.shape(2);
    if (m <= 0 || hc <= 0 || hc > 16 || d <= 0 || nwa.shape(0) != d) {
      return false;
    }
    const int n_gates = head_only ? hc : 2 * hc;
    const int n_scale = head_only ? 1 : 2;
    if (wa.shape(0) != n_gates || wa.shape(1) != hc * d ||
        sa.shape(0) != n_scale || ba.shape(0) != n_gates) {
      return false;
    }
    const bool vec4 = (d % 4 == 0) && (xa.size() >= 8) && (wa.size() >= 8) &&
                      (nwa.size() >= 8) && (size_t(m) * size_t(d) >= 8);
    const size_t align = vec4 ? 4 : 1;
    array xc = ensure_contiguous_aligned(xa, align);
    array wc = ensure_contiguous_aligned(wa, align);
    array sc = ensure_contiguous_aligned(sa, 1);
    array bc = ensure_contiguous_aligned(ba, 1);
    array nwc = ensure_contiguous_aligned(nwa, align);
    array params =
        cached_f32_quad(hc_norm_eps, hc_eps, magnitude, out_norm_eps);

    const Shape post_shape = head_only ? Shape{1} : Shape{m, hc};
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{256, m, 1};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(hc) |
                              (uint64_t(head_only ? 1 : 0) << 8) |
                              (uint64_t(vec4 ? 1 : 0) << 9);
    const std::vector<array> inputs{xc, wc, sc, bc, nwc, params};
    const std::vector<Shape> out_shapes{{m, d}, post_shape};
    const std::vector<Dtype> out_dtypes{mlx::core::float32, mlx::core::float32};
    auto res = dispatch_cached_kernel(
        kKernelHcMixReduceNorm, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_hc_mix_reduce_norm_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"HC", hc}, {"HEAD_ONLY", head_only}, {"VEC4", vec4}},
              std::nullopt, false, stream);
        });
    auto h = std::make_unique<array>(std::move(res[0]));
    if (!head_only) {
      auto p = std::make_unique<array>(std::move(res[1]));
      *out_post = reinterpret_cast<mlx_array *>(p.release());
    }
    *out_h_norm = reinterpret_cast<mlx_array *>(h.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 hc_mix_reduce_norm: " << e.what() << std::endl;
    return false;
  }
#else
  (void)x;
  (void)weight;
  (void)scale;
  (void)base;
  (void)norm_w;
  (void)hc_norm_eps;
  (void)hc_eps;
  (void)magnitude;
  (void)out_norm_eps;
  (void)head_only;
  return false;
#endif
}

mlx_array *mlx_hy4_post_norm_hc_post(mlx_array *y, mlx_array *res,
                                     mlx_array *post, mlx_array *post_norm_w,
                                     float post_eps) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!y || !res || !post || !post_norm_w || !std::isfinite(post_eps)) {
      return nullptr;
    }
    const auto &ya = *reinterpret_cast<array *>(y);
    const auto &ra = *reinterpret_cast<array *>(res);
    const auto &pa = *reinterpret_cast<array *>(post);
    const auto &pnwa = *reinterpret_cast<array *>(post_norm_w);
    if (ya.ndim() != 2 || ra.ndim() != 3 || pa.ndim() != 2 ||
        pnwa.ndim() != 1) {
      return nullptr;
    }
    if (ya.dtype() != mlx::core::float32 || ra.dtype() != mlx::core::float32 ||
        pa.dtype() != mlx::core::float32 ||
        pnwa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int m = ra.shape(0);
    const int hc = ra.shape(1);
    const int d = ra.shape(2);
    if (m <= 0 || hc <= 0 || hc > 16 || d <= 0 || ya.shape(0) != m ||
        ya.shape(1) != d || pa.shape(0) != m || pa.shape(1) != hc ||
        pnwa.shape(0) != d) {
      return nullptr;
    }
    const bool vec4 = (d % 4 == 0) && (ya.size() >= 8) && (ra.size() >= 8) &&
                      (pnwa.size() >= 8);
    const size_t align = vec4 ? 4 : 1;
    array yc = ensure_contiguous_aligned(ya, align);
    array rc = ensure_contiguous_aligned(ra, align);
    array pc = ensure_contiguous_aligned(pa, 1);
    array pnwc = ensure_contiguous_aligned(pnwa, align);
    array eps_arr = cached_f32_scalar(post_eps);

    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{256, m, 1};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(hc) | (uint64_t(vec4 ? 1 : 0) << 8);
    const std::vector<array> inputs{yc, rc, pc, pnwc, eps_arr};
    const std::vector<Shape> out_shapes{{m, hc, d}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelPostNormHcPost, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_post_norm_hc_post_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"HC", hc}, {"VEC4", vec4}}, std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 post_norm_hc_post: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)y;
  (void)res;
  (void)post;
  (void)post_norm_w;
  (void)post_eps;
  return nullptr;
#endif
}

bool mlx_hy4_hc_bridge(mlx_array *y, mlx_array *res, mlx_array *post_prev,
                       mlx_array *post_norm_w, mlx_array *next_weight,
                       mlx_array *next_scale, mlx_array *next_base,
                       mlx_array *next_norm_w, float post_eps,
                       float hc_norm_eps, float hc_eps, float magnitude,
                       float next_eps, bool skip_next_norm, bool head_only,
                       mlx_array **out_x, mlx_array **out_h,
                       mlx_array **out_post_next) {
  if (out_x)
    *out_x = nullptr;
  if (out_h)
    *out_h = nullptr;
  if (out_post_next)
    *out_post_next = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return false;
    }
    if (!y || !res || !post_prev || !post_norm_w || !next_weight ||
        !next_scale || !next_base || !out_h ||
        (!skip_next_norm && !next_norm_w) ||
        (!head_only && (!out_x || !out_post_next))) {
      return false;
    }
    if (!std::isfinite(post_eps) || !std::isfinite(hc_norm_eps) ||
        !std::isfinite(hc_eps) || !std::isfinite(magnitude) ||
        !std::isfinite(next_eps)) {
      return false;
    }
    const auto &ya = *reinterpret_cast<array *>(y);
    const auto &ra = *reinterpret_cast<array *>(res);
    const auto &ppa = *reinterpret_cast<array *>(post_prev);
    const auto &pnwa = *reinterpret_cast<array *>(post_norm_w);
    const auto &nwa = *reinterpret_cast<array *>(next_weight);
    const auto &nsa = *reinterpret_cast<array *>(next_scale);
    const auto &nba = *reinterpret_cast<array *>(next_base);
    if (ya.ndim() != 2 || ra.ndim() != 3 || ppa.ndim() != 2 ||
        pnwa.ndim() != 1 || nwa.ndim() != 2 || nsa.ndim() != 1 ||
        nba.ndim() != 1) {
      return false;
    }
    if (ya.dtype() != mlx::core::float32 || ra.dtype() != mlx::core::float32 ||
        ppa.dtype() != mlx::core::float32 ||
        pnwa.dtype() != mlx::core::float32 ||
        nwa.dtype() != mlx::core::float32 ||
        nsa.dtype() != mlx::core::float32 ||
        nba.dtype() != mlx::core::float32) {
      return false;
    }
    const int m = ra.shape(0);
    const int hc = ra.shape(1);
    const int d = ra.shape(2);
    if (m <= 0 || hc <= 0 || hc > 16 || d <= 0 || ya.shape(0) != m ||
        ya.shape(1) != d || ppa.shape(0) != m || ppa.shape(1) != hc ||
        pnwa.shape(0) != d) {
      return false;
    }
    const int n_gates = head_only ? hc : 2 * hc;
    const int n_scale = head_only ? 1 : 2;
    if (nwa.shape(0) != n_gates || nwa.shape(1) != hc * d ||
        nsa.shape(0) != n_scale || nba.shape(0) != n_gates) {
      return false;
    }
    array nnwc = cached_zero_f32();
    bool nnw_vec4_ok = true;
    if (!skip_next_norm) {
      const auto &nnwa = *reinterpret_cast<array *>(next_norm_w);
      if (nnwa.ndim() != 1 || nnwa.dtype() != mlx::core::float32 ||
          nnwa.shape(0) != d) {
        return false;
      }
      nnw_vec4_ok = (nnwa.size() >= 8);
      nnwc = ensure_contiguous_aligned(nnwa, (d % 4 == 0) ? 4 : 1);
    }
    const bool vec4 = (d % 4 == 0) && (ya.size() >= 8) && (ra.size() >= 8) &&
                      (pnwa.size() >= 8) && (nwa.size() >= 8) && nnw_vec4_ok &&
                      (size_t(m) * size_t(d) >= 8);
    const size_t align = vec4 ? 4 : 1;
    array yc = ensure_contiguous_aligned(ya, align);
    array rc = ensure_contiguous_aligned(ra, align);
    array ppc = ensure_contiguous_aligned(ppa, 1);
    array pnwc = ensure_contiguous_aligned(pnwa, align);
    array nwc = ensure_contiguous_aligned(nwa, align);
    array nsc = ensure_contiguous_aligned(nsa, 1);
    array nbc = ensure_contiguous_aligned(nba, 1);

    const Shape x_shape = head_only ? Shape{1} : Shape{m, hc, d};
    const Shape post_shape = head_only ? Shape{1} : Shape{m, hc};
    const auto stream = default_stream(Device::gpu);

    static const int nsplits = [] {
      const char *v = std::getenv("MLX_HY4_HC_BRIDGE_SPLITS");
      if (v && *v) {
        const int parsed = std::atoi(v);
        if (parsed == 1 || parsed == 4 || parsed == 6 || parsed == 8 ||
            parsed == 12 || parsed == 16 || parsed == 24 || parsed == 32) {
          return parsed;
        }
      }
      return 16;
    }();
    const bool use_multi_tg =
        vec4 && (d >= 1024) && (m <= 32) && (hc <= 8) && (nsplits > 1);
    if (use_multi_tg) {
      int tg_x1 = 128;
      if (nsplits == 4 || nsplits == 6 || nsplits == 12) {
        tg_x1 = 256;
      } else if (nsplits == 16) {
        tg_x1 = 192;
      } else if (nsplits == 32) {
        tg_x1 = 96;
      }
      const int m_tile = (m <= 4) ? m : 4;
      const Hy4HcBridgeParams bp{
          m, d, post_eps, hc_norm_eps, hc_eps, magnitude, next_eps};
      auto prim = std::make_shared<Hy4HcBridge>(stream, hc, nsplits, tg_x1,
                                                m_tile, skip_next_norm,
                                                head_only, bp);
      const std::vector<array> in_arrs{yc,  rc,  ppc,  pnwc,
                                       nwc, nsc, nbc, nnwc};
      const std::vector<Shape> out_s{x_shape, {m, d}, post_shape};
      const std::vector<Dtype> out_d{mlx::core::float32, mlx::core::float32,
                                     mlx::core::float32};
      auto out_arrs = array::make_arrays(out_s, out_d, std::move(prim), in_arrs);
      auto h_arr = std::make_unique<array>(std::move(out_arrs[1]));
      if (!head_only) {
        auto x_arr = std::make_unique<array>(std::move(out_arrs[0]));
        auto p_arr = std::make_unique<array>(std::move(out_arrs[2]));
        *out_x = reinterpret_cast<mlx_array *>(x_arr.release());
        *out_post_next = reinterpret_cast<mlx_array *>(p_arr.release());
      }
      *out_h = reinterpret_cast<mlx_array *>(h_arr.release());
      return true;
    }

    const bool use_reg = vec4 && (hc <= 4) && (d <= 3072);
    const bool row2 = use_reg && (m >= 64);
    const int grid_y = row2 ? ((m + 1) / 2) : m;
    array params =
        cached_f32_five(post_eps, hc_norm_eps, hc_eps, magnitude, next_eps);

    const std::tuple<int, int, int> grid{256, grid_y, 1};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(hc) | (uint64_t(vec4 ? 1 : 0) << 8) |
                              (uint64_t(skip_next_norm ? 1 : 0) << 9) |
                              (uint64_t(head_only ? 1 : 0) << 10) |
                              (uint64_t(use_reg ? 1 : 0) << 11) |
                              (uint64_t(row2 ? 1 : 0) << 12);
    const std::vector<array> inputs{yc,  rc,  ppc,  pnwc,  nwc,
                                    nsc, nbc, nnwc, params};
    const std::vector<Shape> out_shapes{x_shape, {m, d}, post_shape};
    const std::vector<Dtype> out_dtypes{mlx::core::float32, mlx::core::float32,
                                        mlx::core::float32};
    auto res_arrs = dispatch_cached_kernel(
        kKernelHcBridge, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_hc_bridge_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"HC", hc},
               {"VEC4", vec4},
               {"SKIP_NEXT_NORM", skip_next_norm},
               {"HEAD_ONLY", head_only},
               {"USE_REG", use_reg},
               {"ROW2", row2}},
              std::nullopt, false, stream);
        });
    auto h_arr = std::make_unique<array>(std::move(res_arrs[1]));
    if (!head_only) {
      auto x_arr = std::make_unique<array>(std::move(res_arrs[0]));
      auto p_arr = std::make_unique<array>(std::move(res_arrs[2]));
      *out_x = reinterpret_cast<mlx_array *>(x_arr.release());
      *out_post_next = reinterpret_cast<mlx_array *>(p_arr.release());
    }
    *out_h = reinterpret_cast<mlx_array *>(h_arr.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 hc_bridge: " << e.what() << std::endl;
    return false;
  }
#else
  (void)y;
  (void)res;
  (void)post_prev;
  (void)post_norm_w;
  (void)next_weight;
  (void)next_scale;
  (void)next_base;
  (void)next_norm_w;
  (void)post_eps;
  (void)hc_norm_eps;
  (void)hc_eps;
  (void)magnitude;
  (void)next_eps;
  (void)skip_next_norm;
  (void)head_only;
  return false;
#endif
}

mlx_array *mlx_hy4_bf16_lm_head(mlx_array *x, mlx_array *w) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    if (xa.ndim() != 2 || wa.ndim() != 2 ||
        xa.dtype() != mlx::core::float32 ||
        wa.dtype() != mlx::core::bfloat16) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int k = xa.shape(1);
    const int v = wa.shape(0);
    if (m < 1 || m > 16 || k <= 0 || k % 4 != 0 || v <= 0 || wa.shape(1) != k) {
      return nullptr;
    }
    const bool fast_k = (k % 256 == 0);
    array wc = ensure_contiguous_aligned(wa, fast_k ? 8 : 4);
    return bf16_lm_head_dispatch(xa, wc, 0ULL, v, false);
  } catch (const std::exception &e) {
    std::cerr << "Hy4 bf16_lm_head: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_bf16_lm_head_mmap(mlx_array *x, mlx_array *w_buf,
                                     uint64_t byte_offset, int vocab) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w_buf) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w_buf);
    if (xa.ndim() != 2 || xa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int k = xa.shape(1);
    if (m < 1 || m > 16 || k <= 0 || k % 4 != 0 || vocab <= 0) {
      return nullptr;
    }
    const uint64_t row_bytes = uint64_t(k) * 2ULL;
    if (uint64_t(vocab) > UINT64_MAX / row_bytes) {
      return nullptr;
    }
    const uint64_t need_bytes = uint64_t(vocab) * row_bytes;
    if (!check_gguf_buf(wa, byte_offset, need_bytes, 8)) {
      return nullptr;
    }
    array wc = ensure_contiguous_aligned(wa, 8);
    return bf16_lm_head_dispatch(xa, wc, byte_offset, vocab, true);
  } catch (const std::exception &e) {
    std::cerr << "Hy4 bf16_lm_head_mmap: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w_buf;
  (void)byte_offset;
  (void)vocab;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_gguf_q8_0_embed_rows(mlx_array *ids, mlx_array *buf,
                                        uint64_t byte_offset, int vocab,
                                        int dim) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!ids || !buf) {
      return nullptr;
    }
    const auto &ia = *reinterpret_cast<array *>(ids);
    const auto &ba = *reinterpret_cast<array *>(buf);
    if (ia.ndim() != 1) {
      return nullptr;
    }
    const bool signed_ids = (ia.dtype() == mlx::core::int32);
    if (!signed_ids && ia.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int m = ia.shape(0);
    if (m < 1 || vocab <= 0 || dim <= 0 || dim % 32 != 0) {
      return nullptr;
    }
    const uint64_t row_bytes = uint64_t(dim / 32) * 34ULL;
    if (uint64_t(vocab) > UINT64_MAX / row_bytes) {
      return nullptr;
    }
    const uint64_t need_bytes = uint64_t(vocab) * row_bytes;
    if (!check_gguf_buf(ba, byte_offset, need_bytes, 2)) {
      return nullptr;
    }
    array ic = ensure_contiguous_aligned(ia, 1);
    array bc = ensure_contiguous_aligned(ba, 2);
    array off_arr = cached_u64_scalar(byte_offset);
    array dims_arr = cached_i32_pair(vocab, dim);

    const int tg_x = std::min(dim, 256);
    const int gx = ((dim + tg_x - 1) / tg_x) * tg_x;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{gx, m, 1};
    const std::tuple<int, int, int> tg{tg_x, 1, 1};
    const uint64_t tmpl_key = uint64_t(signed_ids ? 1 : 0);
    const std::vector<array> inputs{ic, bc, off_arr, dims_arr};
    const std::vector<Shape> out_shapes{{m, dim}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelGgufQ8_0EmbedRows, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_gguf_q8_0_embed_rows_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"SIGNED_IDS", signed_ids}}, std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 gguf_q8_0_embed_rows: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)ids;
  (void)buf;
  (void)byte_offset;
  (void)vocab;
  (void)dim;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_dense_decode(mlx_array *x, mlx_array *w,
                                   mlx_array *scales, mlx_array *biases) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w || !scales) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    if (xa.ndim() != 2 || wa.ndim() != 2 || sa.ndim() != 2) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int k = xa.shape(1);
    const int n = wa.shape(0);
    if (m < 1 || m > 16 || k < 32 || k % 32 != 0 || n < 1 ||
        wa.shape(1) != k / 4 || sa.shape() != Shape{n, k / 32}) {
      return nullptr;
    }
    const bool sym_f16 = (biases == nullptr);
    array bc = cached_zero_f16();
    if (sym_f16) {
      if (sa.dtype() != mlx::core::float16) {
        return nullptr;
      }
    } else {
      const auto &ba = *reinterpret_cast<array *>(biases);
      if (ba.ndim() != 2 || ba.shape() != Shape{n, k / 32}) {
        return nullptr;
      }
      const bool both_f32 = (sa.dtype() == mlx::core::float32 &&
                             ba.dtype() == mlx::core::float32);
      const bool both_f16 = (sa.dtype() == mlx::core::float16 &&
                             ba.dtype() == mlx::core::float16);
      if (!both_f32 && !both_f16) {
        return nullptr;
      }
      bc = ensure_contiguous_aligned(ba, 1);
    }
    array xc = ensure_contiguous_aligned(xa, 4);
    array wc = ensure_contiguous_aligned(wa, 4);
    array sc = ensure_contiguous_aligned(sa, 1);

    const int m_tile = hy4_q8_decode_m_tile(m);
    const int rps = (m_tile == 1) ? 4 : 2;
    const int rows_per_tg = 4 * rps;
    const bool fast_k = (k % 512 == 0);
    const int grid_y = 4 * ((n + rows_per_tg - 1) / rows_per_tg);
    const int grid_z = (m + m_tile - 1) / m_tile;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32, grid_y, grid_z};
    const std::tuple<int, int, int> tg{32, 4, 1};
    const bool scales_f16 = (sa.dtype() == mlx::core::float16);
    const uint64_t tmpl_key = uint64_t(m_tile) |
                              (uint64_t(sym_f16 ? 1 : 0) << 9) |
                              (uint64_t(scales_f16 ? 1 : 0) << 10) |
                              (uint64_t(fast_k ? 1 : 0) << 11) |
                              (uint64_t(rps) << 13);
    const std::vector<array> inputs{xc, wc, sc, bc};
    const std::vector<Shape> out_shapes{{m, n}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};

    auto out = dispatch_cached_kernel(
        kKernelQ8Decode, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_q8_decode_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"M_TILE", m_tile},
               {"ROWS_PER_SG", rps},
               {"GROUPED", false},
               {"SYM_F16", sym_f16},
               {"FAST_K", fast_k}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 q8_dense_decode: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_affine4_dense_decode(mlx_array *x, mlx_array *w,
                                        mlx_array *scales, mlx_array *biases) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w || !scales || !biases) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    const auto &ba = *reinterpret_cast<array *>(biases);
    if (xa.ndim() != 2 || wa.ndim() != 2 || sa.ndim() != 2 || ba.ndim() != 2) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::uint32 ||
        sa.dtype() != mlx::core::bfloat16 ||
        ba.dtype() != mlx::core::bfloat16) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int k = xa.shape(1);
    const int n = wa.shape(0);
    if (m < 1 || m > 8 || k < 64 || k % 64 != 0 || n < 1 ||
        wa.shape(1) != k / 8 || sa.shape() != Shape{n, k / 64} ||
        ba.shape() != Shape{n, k / 64}) {
      return nullptr;
    }
    array xc = ensure_contiguous_aligned(xa, 4);
    array wc = ensure_contiguous_aligned(wa, 4);
    array sc = ensure_contiguous_aligned(sa, 1);
    array bc = ensure_contiguous_aligned(ba, 1);

    const int m_tile = hy4_q8_short_m_tile(m);
    const int rps = (m_tile == 1) ? 4 : 2;
    const int rows_per_tg = 4 * rps;
    const bool fast_k = (k % 512 == 0);
    const int grid_y = 4 * ((n + rows_per_tg - 1) / rows_per_tg);
    const int grid_z = (m + m_tile - 1) / m_tile;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32, grid_y, grid_z};
    const std::tuple<int, int, int> tg{32, 4, 1};
    const uint64_t tmpl_key = uint64_t(m_tile) | (uint64_t(rps) << 8) |
                              (uint64_t(fast_k ? 1 : 0) << 16);
    const std::vector<array> inputs{xc, wc, sc, bc};
    const std::vector<Shape> out_shapes{{m, n}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};

    auto out = dispatch_cached_kernel(
        kKernelAffine4DenseDecode, stream, grid, tg, tmpl_key, inputs,
        out_shapes, out_dtypes, [&] {
          return hy4_affine4_dense_decode_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"M_TILE", m_tile}, {"ROWS_PER_SG", rps}, {"FAST_K", fast_k}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 affine4_dense_decode: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_grouped_decode(mlx_array *x, mlx_array *w,
                                     mlx_array *scales, mlx_array *biases,
                                     mlx_array *gate) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w || !scales) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    if (xa.ndim() != 3 || wa.ndim() != 2 || sa.ndim() != 2) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int g = xa.shape(1);
    const int k = xa.shape(2);
    if (m < 1 || m > 16 || g < 1 || k < 32 || k % 32 != 0 ||
        wa.shape(0) < g || wa.shape(0) % g != 0 || wa.shape(1) != k / 4) {
      return nullptr;
    }
    const int n = wa.shape(0) / g;
    if (n < 1 || sa.shape() != Shape{g * n, k / 32}) {
      return nullptr;
    }
    const bool gated = (gate != nullptr);
    if (gated && k != 512) {
      return nullptr;
    }
    array gc = cached_f32_scalar(0.0f);
    if (gated) {
      const auto &ga = *reinterpret_cast<array *>(gate);
      if (ga.dtype() != mlx::core::float32) {
        return nullptr;
      }
      const bool ok2 = (ga.ndim() == 2 && ga.shape() == Shape{m, g * n});
      const bool ok3 = (ga.ndim() == 3 && ga.shape() == Shape{m, g, n});
      if (!ok2 && !ok3) {
        return nullptr;
      }
      gc = ensure_strided_aligned(ga, 1);
    }
    const bool sym_f16 = (biases == nullptr);
    array bc = cached_zero_f16();
    if (sym_f16) {
      if (sa.dtype() != mlx::core::float16) {
        return nullptr;
      }
    } else {
      const auto &ba = *reinterpret_cast<array *>(biases);
      if (ba.ndim() != 2 || ba.shape() != Shape{g * n, k / 32}) {
        return nullptr;
      }
      const bool both_f32 = (sa.dtype() == mlx::core::float32 &&
                             ba.dtype() == mlx::core::float32);
      const bool both_f16 = (sa.dtype() == mlx::core::float16 &&
                             ba.dtype() == mlx::core::float16);
      if (!both_f32 && !both_f16) {
        return nullptr;
      }
      bc = ensure_contiguous_aligned(ba, 1);
    }

    const auto stream = default_stream(Device::gpu);
    const std::vector<Shape> out_shapes =
        gated ? std::vector<Shape>{{m, g * n}} : std::vector<Shape>{{m, g, n}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};

    const bool scales_f16 = (sa.dtype() == mlx::core::float16);
    if (!gated && k == 192) {
      array xc = ensure_strided_aligned(xa, 4);
      array wc = ensure_contiguous_aligned(wa, 1);
      array sc = ensure_contiguous_aligned(sa, 1);
      const int m_tile = hy4_q8_short_m_tile(m);
      const int grid_y = 4 * ((n + 15) / 16);
      const int grid_z = g * ((m + m_tile - 1) / m_tile);
      const std::tuple<int, int, int> grid{32, grid_y, grid_z};
      const std::tuple<int, int, int> tg{32, 4, 1};
      const uint64_t tmpl_key = uint64_t(m_tile) |
                                (uint64_t(sym_f16 ? 1 : 0) << 8) |
                                (uint64_t(scales_f16 ? 1 : 0) << 9);
      const std::vector<array> inputs{xc, wc, sc, bc};
      auto out = dispatch_cached_kernel(
          kKernelQ8GroupedK192, stream, grid, tg, tmpl_key, inputs,
          out_shapes, out_dtypes, [&] {
            return hy4_q8_grouped_k192_kernel()(
                inputs, out_shapes, out_dtypes, grid, tg,
                {{"M_TILE", m_tile}, {"SYM_F16", sym_f16}}, std::nullopt, false,
                stream);
          });
      return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
    }

    if (k == 512) {
      array xc = ensure_contiguous_aligned(xa, 4);
      array wc = ensure_contiguous_aligned(wa, 4);
      array sc = ensure_contiguous_aligned(sa, 1);
      const int m_tile = hy4_q8_short_m_tile(m);
      const int grid_y = 4 * ((n + 7) / 8);
      const int grid_z = g * ((m + m_tile - 1) / m_tile);
      const std::tuple<int, int, int> grid{32, grid_y, grid_z};
      const std::tuple<int, int, int> tg{32, 4, 1};
      const uint64_t tmpl_key = uint64_t(m_tile) |
                                (uint64_t(sym_f16 ? 1 : 0) << 8) |
                                (uint64_t(scales_f16 ? 1 : 0) << 9) |
                                (uint64_t(gated ? 1 : 0) << 10);
      const std::vector<array> inputs{xc, wc, sc, bc, gc};
      auto out = dispatch_cached_kernel(
          kKernelQ8GroupedK512, stream, grid, tg, tmpl_key, inputs,
          out_shapes, out_dtypes, [&] {
            return hy4_q8_grouped_k512_kernel()(
                inputs, out_shapes, out_dtypes, grid, tg,
                {{"M_TILE", m_tile}, {"SYM_F16", sym_f16}, {"GATED", gated}},
                std::nullopt, false, stream);
          });
      return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
    }

    array xc = ensure_contiguous_aligned(xa, 4);
    array wc = ensure_contiguous_aligned(wa, 4);
    array sc = ensure_contiguous_aligned(sa, 1);

    const int m_tile = hy4_q8_decode_m_tile(m);
    const int rps = 2;
    const bool fast_k = (k % 512 == 0);
    const int grid_y = 4 * ((n + 7) / 8);
    const int grid_z = g * ((m + m_tile - 1) / m_tile);
    const std::tuple<int, int, int> grid{32, grid_y, grid_z};
    const std::tuple<int, int, int> tg{32, 4, 1};
    const uint64_t tmpl_key = uint64_t(m_tile) | (1ULL << 8) |
                              (uint64_t(sym_f16 ? 1 : 0) << 9) |
                              (uint64_t(scales_f16 ? 1 : 0) << 10) |
                              (uint64_t(fast_k ? 1 : 0) << 11) |
                              (uint64_t(rps) << 13);
    const std::vector<array> inputs{xc, wc, sc, bc};

    auto out = dispatch_cached_kernel(
        kKernelQ8Decode, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_q8_decode_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"M_TILE", m_tile},
               {"ROWS_PER_SG", rps},
               {"GROUPED", true},
               {"SYM_F16", sym_f16},
               {"FAST_K", fast_k}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 q8_grouped_decode: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  (void)gate;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_dense_prefill(mlx_array *x, mlx_array *w,
                                    mlx_array *scales, mlx_array *biases,
                                    bool out_f16) {
#ifdef MLX_NODE_METAL_ENABLED
  return hy4_q8_prefill_impl(x, w, scales, biases, false, nullptr, 0, out_f16);
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  (void)out_f16;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_dense_prefill_epilogue(mlx_array *x, mlx_array *w,
                                             mlx_array *scales,
                                             mlx_array *biases, mlx_array *aux,
                                             int epilogue, bool out_f16) {
#ifdef MLX_NODE_METAL_ENABLED
  return hy4_q8_prefill_impl(x, w, scales, biases, false, aux, epilogue,
                             out_f16);
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  (void)aux;
  (void)epilogue;
  (void)out_f16;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_grouped_prefill(mlx_array *x, mlx_array *w,
                                      mlx_array *scales, mlx_array *biases,
                                      bool out_f16) {
#ifdef MLX_NODE_METAL_ENABLED
  return hy4_q8_prefill_impl(x, w, scales, biases, true, nullptr, 0, out_f16);
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  (void)out_f16;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_grouped_prefill_gated(mlx_array *x, mlx_array *w,
                                            mlx_array *scales,
                                            mlx_array *biases,
                                            mlx_array *gate, bool out_f16) {
#ifdef MLX_NODE_METAL_ENABLED
  return hy4_q8_prefill_impl(x, w, scales, biases, true, gate, 2, out_f16);
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  (void)gate;
  (void)out_f16;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_kq_routed_decode(mlx_array *x, mlx_array *slots,
                                    mlx_array *weights, mlx_array *wg,
                                    mlx_array *sg, mlx_array *bg, mlx_array *wu,
                                    mlx_array *su, mlx_array *bu, mlx_array *wd,
                                    mlx_array *sd, mlx_array *bd, int down_bits,
                                    float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !slots || !weights || !wg || !sg || !bg || !wu || !su || !bu ||
        !wd || !sd || !bd || !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &sla = *reinterpret_cast<array *>(slots);
    const auto &wa = *reinterpret_cast<array *>(weights);
    const auto &wga = *reinterpret_cast<array *>(wg);
    const auto &sga = *reinterpret_cast<array *>(sg);
    const auto &bga = *reinterpret_cast<array *>(bg);
    const auto &wua = *reinterpret_cast<array *>(wu);
    const auto &sua = *reinterpret_cast<array *>(su);
    const auto &bua = *reinterpret_cast<array *>(bu);
    const auto &wda = *reinterpret_cast<array *>(wd);
    const auto &sda = *reinterpret_cast<array *>(sd);
    const auto &bda = *reinterpret_cast<array *>(bd);

    if (xa.ndim() != 2 || sla.ndim() != 2 || wa.ndim() != 2 ||
        xa.dtype() != mlx::core::float32 || sla.dtype() != mlx::core::uint32 ||
        wa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int rows = xa.shape(0);
    const int k = xa.shape(1);
    const int top = sla.shape(1);
    if (rows < 1 || top < 1 || top > 32 || sla.shape(0) != rows ||
        wa.shape() != Shape{rows, top}) {
      return nullptr;
    }
    int s = 0, m = 0, h = 0;
    if (!validate_kq_banks(wga, sga, bga, wua, sua, bua, wda, sda, bda, k,
                           down_bits, s, m, h)) {
      return nullptr;
    }
    (void)s;

    const int n_assign = rows * top;
    array row_ids = cached_row_ids(rows, top);
    array limit_arr = cached_f32_scalar(limit);

    array xc = ensure_contiguous_aligned(xa, 1);
    array slc = ensure_contiguous_aligned(sla, 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array wgc = ensure_contiguous_aligned(wga, 1);
    array sgc = ensure_contiguous_aligned(sga, 1);
    array bgc = ensure_contiguous_aligned(bga, 1);
    array wuc = ensure_contiguous_aligned(wua, 1);
    array suc = ensure_contiguous_aligned(sua, 1);
    array buc = ensure_contiguous_aligned(bua, 1);
    array wdc = ensure_contiguous_aligned(wda, 1);
    array sdc = ensure_contiguous_aligned(sda, 1);
    array bdc = ensure_contiguous_aligned(bda, 1);

    array slots_flat = reshape(slc, {n_assign});
    const bool fast_gu = (m % 8 == 0) && (k % 512 == 0);
    const bool dedup = (rows >= 2) && (n_assign <= 256);
    const uint64_t gu_tmpl =
        uint64_t(fast_gu ? 1 : 0) | (uint64_t(dedup ? 1 : 0) << 1);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> gu_grid{32, ((m + 7) / 8) * 2, n_assign};
    const std::tuple<int, int, int> gu_tg{32, 2, 1};
    const std::vector<array> gu_inputs{xc,  row_ids, slots_flat, wgc, sgc,
                                       bgc, wuc,     suc,        buc, limit_arr};
    const std::vector<Shape> gu_out_shapes{{n_assign, m}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto hidden = dispatch_cached_kernel(
        kKernelKqGateUp, stream, gu_grid, gu_tg, gu_tmpl, gu_inputs,
        gu_out_shapes, out_dtypes, [&] {
          return hy4_kq_gate_up_kernel()(
              gu_inputs, gu_out_shapes, out_dtypes, gu_grid, gu_tg,
              {{"FAST", fast_gu}, {"DEDUP", dedup}}, std::nullopt, false,
              stream);
        })[0];

    const bool fast_d = (h % 4 == 0) && (m % 512 == 0);
    const int gs = (down_bits == 6) ? 16 : 32;
    const int sr = (down_bits == 6) ? 16 : 8;
    const bool hm = (down_bits == 4);
    constexpr int rps = 4;
    constexpr int ksg = 8;
    const std::tuple<int, int, int> dc_grid{
        32, ((h + rps - 1) / rps) * ksg, rows};
    const std::tuple<int, int, int> dc_tg{32, ksg, 1};
    const uint64_t dc_tmpl = uint64_t(down_bits) | (uint64_t(gs) << 8) |
                             (uint64_t(sr) << 16) |
                             (uint64_t(hm ? 1 : 0) << 24) |
                             (uint64_t(fast_d ? 1 : 0) << 25) |
                             (uint64_t(rps) << 28) | (uint64_t(ksg) << 36);
    const std::vector<array> dc_inputs{hidden, slc, wc, wdc, sdc, bdc};
    const std::vector<Shape> dc_out_shapes{{rows, h}};
    auto out = dispatch_cached_kernel(
        kKernelKqDownCombine, stream, dc_grid, dc_tg, dc_tmpl, dc_inputs,
        dc_out_shapes, out_dtypes, [&] {
          return hy4_kq_down_combine_kernel()(
              dc_inputs, dc_out_shapes, out_dtypes, dc_grid, dc_tg,
              {{"BITS", down_bits},
               {"GS", gs},
               {"SR", sr},
               {"HM", hm},
               {"FAST", fast_d},
               {"RPS", rps},
               {"KSG", ksg}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 kq_routed_decode: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)slots;
  (void)weights;
  (void)wg;
  (void)sg;
  (void)bg;
  (void)wu;
  (void)su;
  (void)bu;
  (void)wd;
  (void)sd;
  (void)bd;
  (void)down_bits;
  (void)limit;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_kq_routed_subset(mlx_array *x, mlx_array *row_ids,
                                    mlx_array *slots, mlx_array *weights,
                                    mlx_array *wg, mlx_array *sg, mlx_array *bg,
                                    mlx_array *wu, mlx_array *su, mlx_array *bu,
                                    mlx_array *wd, mlx_array *sd, mlx_array *bd,
                                    int down_bits, float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !row_ids || !slots || !weights || !wg || !sg || !bg || !wu ||
        !su || !bu || !wd || !sd || !bd || !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &rida = *reinterpret_cast<array *>(row_ids);
    const auto &sla = *reinterpret_cast<array *>(slots);
    const auto &wa = *reinterpret_cast<array *>(weights);
    const auto &wga = *reinterpret_cast<array *>(wg);
    const auto &sga = *reinterpret_cast<array *>(sg);
    const auto &bga = *reinterpret_cast<array *>(bg);
    const auto &wua = *reinterpret_cast<array *>(wu);
    const auto &sua = *reinterpret_cast<array *>(su);
    const auto &bua = *reinterpret_cast<array *>(bu);
    const auto &wda = *reinterpret_cast<array *>(wd);
    const auto &sda = *reinterpret_cast<array *>(sd);
    const auto &bda = *reinterpret_cast<array *>(bd);

    if (xa.ndim() != 2 || rida.ndim() != 1 || sla.ndim() != 1 ||
        wa.ndim() != 1 || xa.dtype() != mlx::core::float32 ||
        rida.dtype() != mlx::core::uint32 || sla.dtype() != mlx::core::uint32 ||
        wa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int rows = xa.shape(0);
    const int k = xa.shape(1);
    const int n_assign = sla.shape(0);
    if (rows < 1 || n_assign < 1 || rida.shape(0) != n_assign ||
        wa.shape(0) != n_assign) {
      return nullptr;
    }
    int s = 0, m = 0, h = 0;
    if (!validate_kq_banks(wga, sga, bga, wua, sua, bua, wda, sda, bda, k,
                           down_bits, s, m, h)) {
      return nullptr;
    }
    (void)s;

    array limit_arr = cached_f32_scalar(limit);
    array xc = ensure_contiguous_aligned(xa, 1);
    array ridc = ensure_contiguous_aligned(rida, 1);
    array slc = ensure_contiguous_aligned(sla, 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array wgc = ensure_contiguous_aligned(wga, 1);
    array sgc = ensure_contiguous_aligned(sga, 1);
    array bgc = ensure_contiguous_aligned(bga, 1);
    array wuc = ensure_contiguous_aligned(wua, 1);
    array suc = ensure_contiguous_aligned(sua, 1);
    array buc = ensure_contiguous_aligned(bua, 1);
    array wdc = ensure_contiguous_aligned(wda, 1);
    array sdc = ensure_contiguous_aligned(sda, 1);
    array bdc = ensure_contiguous_aligned(bda, 1);

    const bool fast_gu = (m % 8 == 0) && (k % 512 == 0);
    const bool dedup = (rows >= 2) && (n_assign <= 256);
    const uint64_t gu_tmpl =
        uint64_t(fast_gu ? 1 : 0) | (uint64_t(dedup ? 1 : 0) << 1);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> gu_grid{32, ((m + 7) / 8) * 2, n_assign};
    const std::tuple<int, int, int> gu_tg{32, 2, 1};
    const std::vector<array> gu_inputs{xc,  ridc, slc, wgc, sgc,
                                       bgc, wuc,  suc, buc, limit_arr};
    const std::vector<Shape> gu_out_shapes{{n_assign, m}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto hidden = dispatch_cached_kernel(
        kKernelKqGateUp, stream, gu_grid, gu_tg, gu_tmpl, gu_inputs,
        gu_out_shapes, out_dtypes, [&] {
          return hy4_kq_gate_up_kernel()(
              gu_inputs, gu_out_shapes, out_dtypes, gu_grid, gu_tg,
              {{"FAST", fast_gu}, {"DEDUP", dedup}}, std::nullopt, false,
              stream);
        })[0];

    const bool fast_d = (h % 4 == 0) && (m % 512 == 0);
    const int gs = (down_bits == 6) ? 16 : 32;
    const int sr = (down_bits == 6) ? 16 : 8;
    const bool hm = (down_bits == 4);
    constexpr int rps = 4;
    const std::tuple<int, int, int> dw_grid{
        32, ((h + 2 * rps - 1) / (2 * rps)) * 2, n_assign};
    const std::tuple<int, int, int> dw_tg{32, 2, 1};
    const uint64_t dw_tmpl = uint64_t(down_bits) | (uint64_t(gs) << 8) |
                             (uint64_t(sr) << 16) |
                             (uint64_t(hm ? 1 : 0) << 24) |
                             (uint64_t(fast_d ? 1 : 0) << 25) |
                             (uint64_t(rps) << 28);
    const std::vector<array> dw_inputs{hidden, slc, wc, wdc, sdc, bdc};
    const std::vector<Shape> dw_out_shapes{{n_assign, h}};
    auto out = dispatch_cached_kernel(
        kKernelKqDownWeighted, stream, dw_grid, dw_tg, dw_tmpl, dw_inputs,
        dw_out_shapes, out_dtypes, [&] {
          return hy4_kq_down_weighted_kernel()(
              dw_inputs, dw_out_shapes, out_dtypes, dw_grid, dw_tg,
              {{"BITS", down_bits},
               {"GS", gs},
               {"SR", sr},
               {"HM", hm},
               {"FAST", fast_d},
               {"RPS", rps}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 kq_routed_subset: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)row_ids;
  (void)slots;
  (void)weights;
  (void)wg;
  (void)sg;
  (void)bg;
  (void)wu;
  (void)su;
  (void)bu;
  (void)wd;
  (void)sd;
  (void)bd;
  (void)down_bits;
  (void)limit;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_mla_sink_attention(mlx_array *q, mlx_array *k0,
                                      mlx_array *k1, mlx_array *idx,
                                      mlx_array *sinks, int v_width,
                                      float scale) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!q || !k0 || !idx || !sinks || !std::isfinite(scale)) {
      return nullptr;
    }
    const auto &qa = *reinterpret_cast<array *>(q);
    const auto &k0a = *reinterpret_cast<array *>(k0);
    const auto &idxa = *reinterpret_cast<array *>(idx);
    const auto &sinksa = *reinterpret_cast<array *>(sinks);
    const bool split = (k1 != nullptr);

    if (qa.ndim() != 3 || k0a.ndim() != 2 || idxa.ndim() != 2 ||
        sinksa.ndim() != 1) {
      return nullptr;
    }
    if (qa.dtype() != mlx::core::float32 || k0a.dtype() != mlx::core::float16 ||
        idxa.dtype() != mlx::core::int32 ||
        sinksa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int s = qa.shape(0);
    const int h = qa.shape(1);
    const int d_k = qa.shape(2);
    const int d_v = v_width;
    const int d_rope = d_k - d_v;
    const int n_keys = k0a.shape(0);
    const int kmax = idxa.shape(1);
    if (s < 1 || h < 1 || n_keys < 1 || kmax < 1 || idxa.shape(0) != s ||
        sinksa.shape(0) != h) {
      return nullptr;
    }
    // The 32-lane split requires V=512 (16/lane), RoPE=64 (2/lane), HALF=288.
    if (d_v != 512 || d_rope != 64) {
      return nullptr;
    }
    if (split) {
      const auto &k1a = *reinterpret_cast<array *>(k1);
      if (k1a.ndim() != 2 || k1a.dtype() != mlx::core::float16 ||
          k0a.shape(1) != 288 || k1a.shape(0) != n_keys ||
          k1a.shape(1) != 288) {
        return nullptr;
      }
    } else if (k0a.shape(1) != d_k) {
      return nullptr;
    }

    const bool try_nax = !mla_decode_nax_env_disabled() &&
                         mlx::core::metal::is_nax_available() &&
                         (h % 16 == 0) && (d_v == 512) && (d_rope == 64);
    const bool try_v2 = !mla_v2_env_disabled() && (h % 16 == 0) &&
                        (d_v == 512) && (d_rope == 64);
    array qc = ensure_contiguous_aligned(qa, (try_nax || try_v2) ? 4 : 1);
    array k0c = ensure_contiguous_aligned(k0a, (try_nax || try_v2) ? 8 : 4);
    array k1c = split ? ensure_contiguous_aligned(
                            *reinterpret_cast<array *>(k1),
                            (try_nax || try_v2) ? 8 : 4)
                      : cached_zero_f16();
    array idxc = ensure_contiguous_aligned(idxa, 1);
    array sinksc = ensure_contiguous_aligned(sinksa, 1);
    array scale_arr = cached_f32_scalar(scale);
    const auto stream = default_stream(Device::gpu);
    constexpr int kSplitKMaxChunk = 64;

    if (try_nax) {
      // Fixed per-split key blocks depend only on H (multiples of TR=32) so
      // row batching and trailing -1 padding never shift split boundaries.
      static const int k_split_h64 = [] {
        const char *v = std::getenv("MLX_HY4_MLA_KSPLIT_H64");
        if (v && *v) {
          const int p = std::atoi(v);
          if (p >= 32 && p <= 512 && (p % 32) == 0) {
            return p;
          }
        }
        return 64;
      }();
      static const int k_split_h32 = [] {
        const char *v = std::getenv("MLX_HY4_MLA_KSPLIT_H32");
        if (v && *v) {
          const int p = std::atoi(v);
          if (p >= 32 && p <= 2048 && (p % 32) == 0) {
            return p;
          }
        }
        return 128;
      }();
      const int k_split_nax = (h >= 64) ? k_split_h64 : k_split_h32;
      const int max_splits_nax =
          std::min(32, std::max(8, 2048 / k_split_nax));
      const int n_splits_nax =
          std::min(max_splits_nax, (kmax + k_split_nax - 1) / k_split_nax);
      const bool single_split = (n_splits_nax == 1);
      array split_cfg_nax = cached_i32_pair(n_splits_nax, k_split_nax);

      static std::atomic<int> nax_ok[2][2]{{{-1}, {-1}}, {{-1}, {-1}}};
      const int sp_idx = split ? 1 : 0;
      const int ss_idx = single_split ? 1 : 0;
      const int cached_nax =
          nax_ok[sp_idx][ss_idx].load(std::memory_order_acquire);

      auto run_nax_chunked =
          [&](std::optional<std::vector<array>> first_parts) -> array {
        if (s <= kSplitKMaxChunk) {
          auto parts =
              first_parts.has_value()
                  ? std::move(*first_parts)
                  : hy4_mla_decode_nax_pass1(qc, k0c, k1c, idxc, sinksc,
                                             scale_arr, split_cfg_nax, split,
                                             n_splits_nax, single_split,
                                             stream);
          if (single_split) {
            return std::move(parts[0]);
          }
          return hy4_mla_splitk_merge(parts[0], parts[1], parts[2], sinksc, d_v,
                                      stream);
        }
        std::vector<array> strips;
        strips.reserve(size_t((s + kSplitKMaxChunk - 1) / kSplitKMaxChunk));
        for (int row0 = 0; row0 < s; row0 += kSplitKMaxChunk) {
          const int row1 = std::min(row0 + kSplitKMaxChunk, s);
          std::vector<array> parts;
          if (row0 == 0 && first_parts.has_value()) {
            parts = std::move(*first_parts);
          } else {
            array q_sub = ensure_contiguous_aligned(
                slice(qc, {row0, 0, 0}, {row1, h, d_k}, stream), 4);
            array idx_sub = ensure_contiguous_aligned(
                slice(idxc, {row0, 0}, {row1, kmax}, stream), 1);
            parts = hy4_mla_decode_nax_pass1(q_sub, k0c, k1c, idx_sub, sinksc,
                                             scale_arr, split_cfg_nax, split,
                                             n_splits_nax, single_split,
                                             stream);
          }
          if (single_split) {
            strips.push_back(std::move(parts[0]));
          } else {
            strips.push_back(hy4_mla_splitk_merge(parts[0], parts[1], parts[2],
                                                  sinksc, d_v, stream));
          }
        }
        return concatenate(strips, 0, stream);
      };

      if (cached_nax == 1) {
        auto result = run_nax_chunked(std::nullopt);
        return reinterpret_cast<mlx_array *>(new array(std::move(result)));
      }
      if (cached_nax == -1) {
        array q_probe =
            (s > kSplitKMaxChunk)
                ? ensure_contiguous_aligned(
                      slice(qc, {0, 0, 0}, {kSplitKMaxChunk, h, d_k}, stream),
                      4)
                : qc;
        array idx_probe =
            (s > kSplitKMaxChunk)
                ? ensure_contiguous_aligned(
                      slice(idxc, {0, 0}, {kSplitKMaxChunk, kmax}, stream), 1)
                : idxc;
        auto parts = hy4_mla_decode_nax_pass1(q_probe, k0c, k1c, idx_probe,
                                              sinksc, scale_arr, split_cfg_nax,
                                              split, n_splits_nax, single_split,
                                              stream);
        const bool ok = (hy4_mla_pipeline_threads(parts[0]) >= 256);
        nax_ok[sp_idx][ss_idx].store(ok ? 1 : 0, std::memory_order_release);
        if (ok) {
          auto result = run_nax_chunked(std::move(parts));
          return reinterpret_cast<mlx_array *>(new array(std::move(result)));
        }
      }
    }

    if (try_v2) {
      // Fixed per-split key blocks depend only on H (multiples of TR=16) so
      // trailing -1 padding in idx never shifts split or tile boundaries.
      const int k_split_v2 = (h >= 64) ? 64 : 128;
      const int max_splits_v2 = (h >= 64) ? 8 : 16;
      const int n_splits_v2 =
          std::min(max_splits_v2, (kmax + k_split_v2 - 1) / k_split_v2);
      array split_cfg_v2 = cached_i32_pair(n_splits_v2, k_split_v2);

      static std::atomic<int> v2_ok[2]{{-1}, {-1}};
      const int cached_v2 =
          v2_ok[split ? 1 : 0].load(std::memory_order_acquire);

      auto run_v2_chunked =
          [&](std::optional<std::vector<array>> first_parts) -> array {
        if (s <= kSplitKMaxChunk) {
          auto parts =
              first_parts.has_value()
                  ? std::move(*first_parts)
                  : hy4_mla_decode_v2_pass1(qc, k0c, k1c, idxc, scale_arr,
                                            split_cfg_v2, split, n_splits_v2,
                                            stream);
          return hy4_mla_splitk_merge(parts[0], parts[1], parts[2], sinksc, d_v,
                                      stream);
        }
        std::vector<array> strips;
        strips.reserve(size_t((s + kSplitKMaxChunk - 1) / kSplitKMaxChunk));
        for (int row0 = 0; row0 < s; row0 += kSplitKMaxChunk) {
          const int row1 = std::min(row0 + kSplitKMaxChunk, s);
          std::vector<array> parts;
          if (row0 == 0 && first_parts.has_value()) {
            parts = std::move(*first_parts);
          } else {
            array q_sub = ensure_contiguous_aligned(
                slice(qc, {row0, 0, 0}, {row1, h, d_k}, stream), 4);
            array idx_sub = ensure_contiguous_aligned(
                slice(idxc, {row0, 0}, {row1, kmax}, stream), 1);
            parts = hy4_mla_decode_v2_pass1(q_sub, k0c, k1c, idx_sub, scale_arr,
                                            split_cfg_v2, split, n_splits_v2,
                                            stream);
          }
          strips.push_back(hy4_mla_splitk_merge(parts[0], parts[1], parts[2],
                                                sinksc, d_v, stream));
        }
        return concatenate(strips, 0, stream);
      };

      if (cached_v2 == 1) {
        auto result = run_v2_chunked(std::nullopt);
        return reinterpret_cast<mlx_array *>(new array(std::move(result)));
      }
      if (cached_v2 == -1) {
        array q_probe =
            (s > kSplitKMaxChunk)
                ? ensure_contiguous_aligned(
                      slice(qc, {0, 0, 0}, {kSplitKMaxChunk, h, d_k}, stream),
                      4)
                : qc;
        array idx_probe =
            (s > kSplitKMaxChunk)
                ? ensure_contiguous_aligned(
                      slice(idxc, {0, 0}, {kSplitKMaxChunk, kmax}, stream), 1)
                : idxc;
        auto parts = hy4_mla_decode_v2_pass1(q_probe, k0c, k1c, idx_probe,
                                             scale_arr, split_cfg_v2, split,
                                             n_splits_v2, stream);
        const bool ok = (hy4_mla_pipeline_threads(parts[0]) >= 256);
        v2_ok[split ? 1 : 0].store(ok ? 1 : 0, std::memory_order_release);
        if (ok) {
          auto result = run_v2_chunked(std::move(parts));
          return reinterpret_cast<mlx_array *>(new array(std::move(result)));
        }
      }
    }

    // Fixed per-split key blocks depend only on H so trailing -1 padding in
    // idx never shifts split boundaries or perturbs online softmax reduction.
    const int k_split = (h >= 64) ? 128 : 256;
    const int n_splits = std::min(8, (kmax + k_split - 1) / k_split);
    const bool use_splitk = (n_splits > 1);

    auto run_splitk_chunked = [&](const MlaConfig &c,
                                  std::optional<std::vector<array>> first_parts)
        -> array {
      array split_cfg = cached_i32_pair(n_splits, k_split);
      if (s <= kSplitKMaxChunk) {
        auto parts = first_parts.has_value()
                         ? std::move(*first_parts)
                         : hy4_mla_splitk_pass1(qc, k0c, k1c, idxc, scale_arr,
                                                split_cfg, d_v, d_rope, split,
                                                c.heads, c.bn, n_splits,
                                                stream);
        return hy4_mla_splitk_merge(parts[0], parts[1], parts[2], sinksc, d_v,
                                    stream);
      }
      std::vector<array> strips;
      strips.reserve(size_t((s + kSplitKMaxChunk - 1) / kSplitKMaxChunk));
      for (int row0 = 0; row0 < s; row0 += kSplitKMaxChunk) {
        const int row1 = std::min(row0 + kSplitKMaxChunk, s);
        std::vector<array> parts;
        if (row0 == 0 && first_parts.has_value()) {
          parts = std::move(*first_parts);
        } else {
          array q_sub = ensure_contiguous_aligned(
              slice(qc, {row0, 0, 0}, {row1, h, d_k}, stream), 1);
          array idx_sub = ensure_contiguous_aligned(
              slice(idxc, {row0, 0}, {row1, kmax}, stream), 1);
          parts = hy4_mla_splitk_pass1(q_sub, k0c, k1c, idx_sub, scale_arr,
                                       split_cfg, d_v, d_rope, split, c.heads,
                                       c.bn, n_splits, stream);
        }
        strips.push_back(hy4_mla_splitk_merge(parts[0], parts[1], parts[2],
                                              sinksc, d_v, stream));
      }
      return concatenate(strips, 0, stream);
    };

    MlaConfig cfg{};
    if (lookup_mla_config(h, split, use_splitk, cfg)) {
      if (!use_splitk) {
        auto result = hy4_mla_single_pass(qc, k0c, k1c, idxc, sinksc, scale_arr,
                                          d_v, d_rope, split, cfg.heads, cfg.bn,
                                          stream);
        return reinterpret_cast<mlx_array *>(new array(std::move(result)));
      }
      auto result = run_splitk_chunked(cfg, std::nullopt);
      return reinterpret_cast<mlx_array *>(new array(std::move(result)));
    }

    constexpr MlaConfig kCandidates[] = {
        {4, 8},
        {2, 16},
        {2, 8},
        {1, 16},
        {1, 8},
    };
    for (const auto &cand : kCandidates) {
      if (h % cand.heads != 0) {
        continue;
      }
      if (!use_splitk) {
        auto result = hy4_mla_single_pass(qc, k0c, k1c, idxc, sinksc, scale_arr,
                                          d_v, d_rope, split, cand.heads,
                                          cand.bn, stream);
        if (hy4_mla_pipeline_threads(result) >= cand.bn * 32) {
          store_mla_config(h, split, false, cand);
          return reinterpret_cast<mlx_array *>(new array(std::move(result)));
        }
      } else {
        array split_cfg = cached_i32_pair(n_splits, k_split);
        array q_probe =
            (s > kSplitKMaxChunk)
                ? ensure_contiguous_aligned(
                      slice(qc, {0, 0, 0}, {kSplitKMaxChunk, h, d_k}, stream),
                      1)
                : qc;
        array idx_probe =
            (s > kSplitKMaxChunk)
                ? ensure_contiguous_aligned(
                      slice(idxc, {0, 0}, {kSplitKMaxChunk, kmax}, stream), 1)
                : idxc;
        auto parts = hy4_mla_splitk_pass1(q_probe, k0c, k1c, idx_probe,
                                          scale_arr, split_cfg, d_v, d_rope,
                                          split, cand.heads, cand.bn, n_splits,
                                          stream);
        if (hy4_mla_pipeline_threads(parts[0]) >= cand.bn * 32) {
          store_mla_config(h, split, true, cand);
          auto result = run_splitk_chunked(cand, std::move(parts));
          return reinterpret_cast<mlx_array *>(new array(std::move(result)));
        }
      }
    }
    return nullptr;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 mla_sink_attention: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)q;
  (void)k0;
  (void)k1;
  (void)idx;
  (void)sinks;
  (void)v_width;
  (void)scale;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_mla_prefill_nax(mlx_array *q, mlx_array *q_pe,
                                   mlx_array *k0, mlx_array *k1, mlx_array *idx,
                                   mlx_array *sinks, int v_width, float scale,
                                   bool out_f16) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu) ||
        !mlx::core::metal::is_nax_available()) {
      return nullptr;
    }
    if (!q || !k0 || !idx || !sinks || !std::isfinite(scale)) {
      return nullptr;
    }
    const auto &qa = *reinterpret_cast<array *>(q);
    const auto &k0a = *reinterpret_cast<array *>(k0);
    const auto &idxa = *reinterpret_cast<array *>(idx);
    const auto &sinksa = *reinterpret_cast<array *>(sinks);
    const bool split = (k1 != nullptr);
    const bool split_q = (q_pe != nullptr);

    if (qa.ndim() != 3 || k0a.ndim() != 2 || idxa.ndim() != 2 ||
        sinksa.ndim() != 1) {
      return nullptr;
    }
    const bool q_f16 = (qa.dtype() == mlx::core::float16);
    if ((!q_f16 && qa.dtype() != mlx::core::float32) ||
        k0a.dtype() != mlx::core::float16 ||
        idxa.dtype() != mlx::core::int32 ||
        sinksa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int s = qa.shape(0);
    const int h = qa.shape(1);
    const int d_v = v_width;
    int d_rope = 0;
    int d_k = 0;
    array qpec = q_f16 ? cached_zero_f16() : cached_zero_f32();
    if (split_q) {
      const auto &qpea = *reinterpret_cast<array *>(q_pe);
      if (qpea.ndim() != 3 || qpea.dtype() != qa.dtype() ||
          qpea.shape(0) != s || qpea.shape(1) != h || qa.shape(2) != d_v) {
        return nullptr;
      }
      d_rope = qpea.shape(2);
      d_k = d_v + d_rope;
      qpec = ensure_contiguous_aligned(qpea, 1);
    } else {
      d_k = qa.shape(2);
      d_rope = d_k - d_v;
    }
    const int n_keys = k0a.shape(0);
    const int kmax = idxa.shape(1);
    if (s < 1 || h < 32 || h % 32 != 0 || n_keys < 1 || kmax < 1 ||
        idxa.shape(0) != s || sinksa.shape(0) != h) {
      return nullptr;
    }
    if (d_v != 512 || d_rope != 64) {
      return nullptr;
    }
    if (split) {
      const auto &k1a = *reinterpret_cast<array *>(k1);
      if (k1a.ndim() != 2 || k1a.dtype() != mlx::core::float16 ||
          k0a.shape(1) != 288 || k1a.shape(0) != n_keys ||
          k1a.shape(1) != 288) {
        return nullptr;
      }
    } else if (k0a.shape(1) != d_k) {
      return nullptr;
    }

    array qc = ensure_contiguous_aligned(qa, 1);
    array k0c = ensure_contiguous_aligned(k0a, 8);
    array k1c = split ? ensure_contiguous_aligned(
                            *reinterpret_cast<array *>(k1), 8)
                      : cached_zero_f16();
    array idxc = ensure_contiguous_aligned(idxa, 1);
    array sinksc = ensure_contiguous_aligned(sinksa, 1);
    array scale_arr = cached_f32_scalar(scale);
    const auto stream = default_stream(Device::gpu);

    constexpr int wm = 2;
    constexpr int wn = 4;
    constexpr int bh = wm * 16;
    constexpr int bk = 32;
    const bool fast_1mma = !mla_prefill_2mma_env();
    const Dtype out_dtype = out_f16 ? mlx::core::float16 : mlx::core::float32;
    const int head_groups = h / bh;
    const std::tuple<int, int, int> grid{32, head_groups * wm * wn, s};
    const std::tuple<int, int, int> tg{32, wm * wn, 1};
    const uint64_t tmpl_key = uint64_t(d_v) | (uint64_t(d_rope) << 16) |
                              (uint64_t(split ? 1 : 0) << 32) |
                              (uint64_t(split_q ? 1 : 0) << 33) |
                              (uint64_t(fast_1mma ? 1 : 0) << 34) |
                              (uint64_t(q_f16 ? 1 : 0) << 35) |
                              (uint64_t(wm) << 36) | (uint64_t(wn) << 40) |
                              (uint64_t(bk) << 44) |
                              (uint64_t(out_f16 ? 1 : 0) << 52);
    const std::vector<array> inputs{qc, qpec, k0c, k1c, idxc, sinksc, scale_arr};
    const std::vector<Shape> out_shapes{{s, h, d_v}};
    const std::vector<Dtype> out_dtypes{out_dtype};
    auto out = dispatch_cached_kernel(
        kKernelMlaPrefillNax, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_mla_prefill_nax_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"D_V", d_v},
               {"D_ROPE", d_rope},
               {"SPLIT", split},
               {"SPLIT_Q", split_q},
               {"FAST_1MMA", fast_1mma},
               {"Q_F16", q_f16},
               {"Q_T", qa.dtype()},
               {"OUT_T", out_dtype},
               {"WM", wm},
               {"WN", wn},
               {"BK", bk}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 mla_prefill_nax: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)q;
  (void)q_pe;
  (void)k0;
  (void)k1;
  (void)idx;
  (void)sinks;
  (void)v_width;
  (void)scale;
  (void)out_f16;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_mla_prefill_sliding_nax(mlx_array *q, mlx_array *q_pe,
                                           mlx_array *k, mlx_array *sinks,
                                           int v_width, float scale,
                                           int start_pos, bool is_ring,
                                           int ring_cap, int window_size,
                                           int first_key, bool out_f16) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu) ||
        !mlx::core::metal::is_nax_available()) {
      return nullptr;
    }
    if (!q || !k || !sinks || !std::isfinite(scale)) {
      return nullptr;
    }
    const auto &qa = *reinterpret_cast<array *>(q);
    const auto &ka = *reinterpret_cast<array *>(k);
    const auto &sinksa = *reinterpret_cast<array *>(sinks);
    const bool split_q = (q_pe != nullptr);

    if (qa.ndim() != 3 || ka.ndim() != 2 || sinksa.ndim() != 1) {
      return nullptr;
    }
    const bool q_f16 = (qa.dtype() == mlx::core::float16);
    if ((!q_f16 && qa.dtype() != mlx::core::float32) ||
        ka.dtype() != mlx::core::float16 ||
        sinksa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int s = qa.shape(0);
    const int h = qa.shape(1);
    const int d_v = v_width;
    int d_rope = 0;
    int d_k = 0;
    array qpec = q_f16 ? cached_zero_f16() : cached_zero_f32();
    if (split_q) {
      const auto &qpea = *reinterpret_cast<array *>(q_pe);
      if (qpea.ndim() != 3 || qpea.dtype() != qa.dtype() ||
          qpea.shape(0) != s || qpea.shape(1) != h || qa.shape(2) != d_v) {
        return nullptr;
      }
      d_rope = qpea.shape(2);
      d_k = d_v + d_rope;
      qpec = ensure_contiguous_aligned(qpea, 1);
    } else {
      d_k = qa.shape(2);
      d_rope = d_k - d_v;
    }
    const int n_keys = ka.shape(0);
    if (s < 1 || h < 1 || n_keys < 1 || sinksa.shape(0) != h ||
        ka.shape(1) != d_k) {
      return nullptr;
    }
    if (d_v != 512 || d_rope != 64) {
      return nullptr;
    }
    if (start_pos < 0 || window_size < 1 || first_key < 0) {
      return nullptr;
    }
    if (is_ring) {
      if (ring_cap < 1 || ring_cap > n_keys) {
        return nullptr;
      }
    } else {
      if (int64_t(start_pos) + int64_t(s) - int64_t(first_key) >
          int64_t(n_keys)) {
        return nullptr;
      }
    }

    array qc = ensure_contiguous_aligned(qa, 1);
    array kc = ensure_contiguous_aligned(ka, 8);
    array sinksc = ensure_contiguous_aligned(sinksa, 1);
    array scale_arr = cached_f32_scalar(scale);
    array params = cached_u32_quad(uint32_t(start_pos), uint32_t(ring_cap),
                                   uint32_t(window_size), uint32_t(first_key));
    const auto stream = default_stream(Device::gpu);

    constexpr int wm = 2;
    constexpr int wn = 4;
    constexpr int bq = 16;
    constexpr int bh = wm;
    const bool fast_1mma = !mla_prefill_2mma_env();
    const Dtype out_dtype = out_f16 ? mlx::core::float16 : mlx::core::float32;
    const int q_tiles = (s + bq - 1) / bq;
    const int head_tiles = (h + bh - 1) / bh;
    const std::tuple<int, int, int> grid{32, q_tiles * wm * wn, head_tiles};
    const std::tuple<int, int, int> tg{32, wm * wn, 1};
    const uint64_t tmpl_key = uint64_t(d_v) | (uint64_t(d_rope) << 16) |
                              (uint64_t(is_ring ? 1 : 0) << 32) |
                              (uint64_t(split_q ? 1 : 0) << 33) |
                              (uint64_t(fast_1mma ? 1 : 0) << 34) |
                              (uint64_t(q_f16 ? 1 : 0) << 35) |
                              (uint64_t(wm) << 36) | (uint64_t(wn) << 40) |
                              (uint64_t(out_f16 ? 1 : 0) << 44);
    const std::vector<array> inputs{qc, qpec, kc, sinksc, scale_arr, params};
    const std::vector<Shape> out_shapes{{s, h, d_v}};
    const std::vector<Dtype> out_dtypes{out_dtype};
    auto out = dispatch_cached_kernel(
        kKernelMlaPrefillSlidingNax, stream, grid, tg, tmpl_key, inputs,
        out_shapes, out_dtypes, [&] {
          return hy4_mla_prefill_sliding_nax_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"D_V", d_v},
               {"D_ROPE", d_rope},
               {"IS_RING", is_ring},
               {"SPLIT_Q", split_q},
               {"FAST_1MMA", fast_1mma},
               {"Q_F16", q_f16},
               {"Q_T", qa.dtype()},
               {"OUT_T", out_dtype},
               {"WM", wm},
               {"WN", wn}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 mla_prefill_sliding_nax: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)q;
  (void)q_pe;
  (void)k;
  (void)sinks;
  (void)v_width;
  (void)scale;
  (void)start_pos;
  (void)is_ring;
  (void)ring_cap;
  (void)window_size;
  (void)first_key;
  (void)out_f16;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_kq_prefill_indirect_gu(mlx_array *x, mlx_array *token_rows,
                                          mlx_array *tiles, mlx_array *wg,
                                          mlx_array *sg, mlx_array *bg,
                                          mlx_array *wu, mlx_array *su,
                                          mlx_array *bu, float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    static const bool env_disabled = [] {
      const char *v = std::getenv("MLX_HY4_PREFILL_INDIRECT_GU");
      return v && std::strcmp(v, "0") == 0;
    }();
    if (env_disabled || !mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu) ||
        !mlx::core::metal::is_nax_available()) {
      return nullptr;
    }
    if (!x || !token_rows || !tiles || !wg || !sg || !bg || !wu || !su || !bu ||
        !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &tra = *reinterpret_cast<array *>(token_rows);
    const auto &ta = *reinterpret_cast<array *>(tiles);
    const auto &wga = *reinterpret_cast<array *>(wg);
    const auto &sga = *reinterpret_cast<array *>(sg);
    const auto &bga = *reinterpret_cast<array *>(bg);
    const auto &wua = *reinterpret_cast<array *>(wu);
    const auto &sua = *reinterpret_cast<array *>(su);
    const auto &bua = *reinterpret_cast<array *>(bu);

    if (xa.ndim() != 2 || tra.ndim() != 1 || ta.ndim() != 2 ||
        wga.ndim() != 3 || sga.ndim() != 3 || bga.ndim() != 3 ||
        wua.ndim() != 3 || sua.ndim() != 3 || bua.ndim() != 3) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || tra.dtype() != mlx::core::uint32 ||
        ta.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int n_tok = xa.shape(0);
    const int k = xa.shape(1);
    const int r = tra.shape(0);
    const int n_tiles = ta.shape(0);
    const int s = wga.shape(0);
    const int m = wga.shape(1);
    if (n_tok < 1 || r < 1 || n_tiles < 1 || ta.shape(1) != 3 || s < 1 ||
        k < 256 || k % 256 != 0 || m < 64 || m % 64 != 0) {
      return nullptr;
    }
    const Shape q4_w{s, m, k / 8};
    const Shape q4_s{s, m, k / 16};
    const Shape q4_b{s, m, k / 128};
    if (wga.dtype() != mlx::core::uint32 || wga.shape() != q4_w ||
        sga.dtype() != mlx::core::uint8 || sga.shape() != q4_s ||
        bga.dtype() != mlx::core::float16 || bga.shape() != q4_b ||
        wua.dtype() != mlx::core::uint32 || wua.shape() != q4_w ||
        sua.dtype() != mlx::core::uint8 || sua.shape() != q4_s ||
        bua.dtype() != mlx::core::float16 || bua.shape() != q4_b) {
      return nullptr;
    }

    array xc = ensure_contiguous_aligned(xa, 4);
    array trc = ensure_contiguous_aligned(tra, 1);
    array tc = ensure_contiguous_aligned(ta, 1);
    array wgc = ensure_contiguous_aligned(wga, 1);
    array sgc = ensure_contiguous_aligned(sga, 1);
    array bgc = ensure_contiguous_aligned(bga, 1);
    array wuc = ensure_contiguous_aligned(wua, 1);
    array suc = ensure_contiguous_aligned(sua, 1);
    array buc = ensure_contiguous_aligned(bua, 1);
    array limit_arr = cached_f32_scalar(limit);

    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32 * (m / 64), 4 * n_tiles, 1};
    const std::tuple<int, int, int> tg{32, 4, 1};
    const std::vector<array> inputs{xc,  trc, tc,  wgc, sgc,
                                    bgc, wuc, suc, buc, limit_arr};
    const std::vector<Shape> out_shapes{{r, m}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelKqPrefillIndirectGu, stream, grid, tg, 0, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_kq_prefill_indirect_gu_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg, {}, std::nullopt, false,
              stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 kq_prefill_indirect_gu: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)token_rows;
  (void)tiles;
  (void)wg;
  (void)sg;
  (void)bg;
  (void)wu;
  (void)su;
  (void)bu;
  (void)limit;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_kq_sorted_combine(mlx_array *down, mlx_array *weights,
                                     mlx_array *inverse) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!down || !weights || !inverse) {
      return nullptr;
    }
    const auto &da = *reinterpret_cast<array *>(down);
    const auto &wa = *reinterpret_cast<array *>(weights);
    const auto &ia = *reinterpret_cast<array *>(inverse);
    const bool in_f16 = (da.dtype() == mlx::core::float16);
    if (da.ndim() != 2 || wa.ndim() != 2 || ia.ndim() != 2 ||
        (!in_f16 && da.dtype() != mlx::core::float32) ||
        wa.dtype() != mlx::core::float32 || ia.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int r = da.shape(0);
    const int h = da.shape(1);
    const int n_tok = wa.shape(0);
    const int top = wa.shape(1);
    if (r < 1 || h < 1 || n_tok < 1 || top < 1 || top > 32 ||
        ia.shape() != Shape{n_tok, top}) {
      return nullptr;
    }

    // Only cast down/out to device vec4* when size >= 8 so MLX binds both in
    // device address space rather than constant.
    const bool vec4 = (h % 4 == 0) && (da.size() >= 8) &&
                      (size_t(n_tok) * size_t(h) >= 8);
    array dc = ensure_contiguous_aligned(da, vec4 ? 4 : 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array ic = ensure_contiguous_aligned(ia, 1);

    const int cols = vec4 ? (h / 4) : h;
    const int tg_x = std::min(256, cols);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{cols, n_tok, 1};
    const std::tuple<int, int, int> tg{tg_x, 1, 1};
    const uint64_t tmpl_key = uint64_t(top) | (uint64_t(vec4 ? 1 : 0) << 8) |
                              (uint64_t(in_f16 ? 1 : 0) << 9);
    const std::vector<array> inputs{dc, wc, ic};
    const std::vector<Shape> out_shapes{{n_tok, h}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelKqSortedCombine, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_kq_sorted_combine_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"TOP", top}, {"VEC4", vec4}, {"IN_F16", in_f16}}, std::nullopt,
              false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 kq_sorted_combine: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)down;
  (void)weights;
  (void)inverse;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_router_mv(mlx_array *x, mlx_array *w) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const bool w_f16 = (wa.dtype() == mlx::core::float16);
    if (xa.ndim() != 2 || wa.ndim() != 2 ||
        xa.dtype() != mlx::core::float32 ||
        (!w_f16 && wa.dtype() != mlx::core::float32)) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int k = xa.shape(1);
    const int e = wa.shape(0);
    if (m < 1 || m > 16 || k < 32 || k % 32 != 0 || e < 1 ||
        wa.shape(1) != k) {
      return nullptr;
    }
    array xc = ensure_contiguous_aligned(xa, 4);
    array wc = ensure_contiguous_aligned(wa, 4);
    const int m_tile = hy4_q8_decode_m_tile(m);
    const bool fast_k = (k % 512 == 0);
    constexpr int num_sg = 2;
    constexpr int rps = 1;
    constexpr int experts_per_tg = num_sg * rps;
    const int n_etiles = (e + experts_per_tg - 1) / experts_per_tg;
    const int n_mtiles = (m + m_tile - 1) / m_tile;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32, num_sg, n_etiles * n_mtiles};
    const std::tuple<int, int, int> tg{32, num_sg, 1};
    const uint64_t tmpl_key = uint64_t(m_tile) |
                              (uint64_t(fast_k ? 1 : 0) << 8) |
                              (uint64_t(w_f16 ? 1 : 0) << 10) |
                              (uint64_t(num_sg) << 12) |
                              (uint64_t(rps) << 16);
    const std::vector<array> inputs{xc, wc};
    const std::vector<Shape> out_shapes{{m, e}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelRouterMv, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_router_mv_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"M_TILE", m_tile},
               {"NUM_SG", num_sg},
               {"RPS", rps},
               {"FAST_K", fast_k},
               {"W_F16", w_f16}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 router_mv: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_dual_down_add(mlx_array *routed_latent,
                                    mlx_array *shared_gu, mlx_array *w_up,
                                    mlx_array *s_up, mlx_array *w_sd,
                                    mlx_array *s_sd, float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!routed_latent || !shared_gu || !w_up || !s_up || !w_sd || !s_sd ||
        !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &rla = *reinterpret_cast<array *>(routed_latent);
    const auto &sgua = *reinterpret_cast<array *>(shared_gu);
    const auto &wupa = *reinterpret_cast<array *>(w_up);
    const auto &supa = *reinterpret_cast<array *>(s_up);
    const auto &wsda = *reinterpret_cast<array *>(w_sd);
    const auto &ssda = *reinterpret_cast<array *>(s_sd);
    if (rla.ndim() != 2 || sgua.ndim() != 2 || wupa.ndim() != 2 ||
        supa.ndim() != 2 || wsda.ndim() != 2 || ssda.ndim() != 2) {
      return nullptr;
    }
    if (rla.dtype() != mlx::core::float32 ||
        sgua.dtype() != mlx::core::float32 ||
        wupa.dtype() != mlx::core::uint32 ||
        supa.dtype() != mlx::core::float16 ||
        wsda.dtype() != mlx::core::uint32 ||
        ssda.dtype() != mlx::core::float16) {
      return nullptr;
    }
    const int m = rla.shape(0);
    const int k = rla.shape(1);
    const int n = wupa.shape(0);
    if (m < 1 || k < 32 || k % 32 != 0 || n < 1 ||
        sgua.shape() != Shape{m, 2 * k} || wupa.shape(1) != k / 4 ||
        supa.shape() != Shape{n, k / 32} || wsda.shape() != Shape{n, k / 4} ||
        ssda.shape() != Shape{n, k / 32}) {
      return nullptr;
    }
    if (m > 8 && (!mlx::core::metal::is_nax_available() ||
                  prefill_q8_nax_env_disabled() || k < 64 || k % 64 != 0)) {
      return nullptr;
    }
    array rlc = ensure_strided_aligned(rla, 4);
    array sguc = ensure_strided_aligned(sgua, 4);
    array wupc = ensure_contiguous_aligned(wupa, 4);
    array supc = ensure_contiguous_aligned(supa, 1);
    array wsdc = ensure_contiguous_aligned(wsda, 4);
    array ssdc = ensure_contiguous_aligned(ssda, 1);
    array limit_arr = cached_f32_scalar(limit);

    const bool clamp_act = (limit > 0.0f);
    const auto stream = default_stream(Device::gpu);
    const std::vector<array> inputs{rlc,  sguc, wupc,     supc,
                                    wsdc, ssdc, limit_arr};
    const std::vector<Shape> out_shapes{{m, n}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    if (m > 8) {
      const bool fast_1mma = q8_prefill_1mma_env();
      const int bm = (m <= 48 || n <= 256) ? 32 : 64;
      const int bn = (fast_1mma && bm == 64 && n >= 128) ? 128 : 64;
      const int num_sg = (fast_1mma && bn == 128) ? 8 : 4;
      const bool aligned_n = (n % bn == 0);
      const int grid_x = 32 * ((n + bn - 1) / bn);
      const int grid_y = num_sg * ((m + bm - 1) / bm);
      const std::tuple<int, int, int> grid{grid_x, grid_y, 1};
      const std::tuple<int, int, int> tg{32, num_sg, 1};
      const uint64_t tmpl_key = uint64_t(bm) | (uint64_t(bn) << 8) |
                                (uint64_t(fast_1mma ? 1 : 0) << 16) |
                                (uint64_t(aligned_n ? 1 : 0) << 17) |
                                (uint64_t(clamp_act ? 1 : 0) << 18);
      auto out = dispatch_cached_kernel(
          kKernelQ8DualDownPrefill, stream, grid, tg, tmpl_key, inputs,
          out_shapes, out_dtypes, [&] {
            return hy4_q8_dual_down_prefill_kernel()(
                inputs, out_shapes, out_dtypes, grid, tg,
                {{"M_TILE", bm},
                 {"N_TILE", bn},
                 {"FAST_1MMA", fast_1mma},
                 {"ALIGNED_N", aligned_n},
                 {"CLAMP", clamp_act}},
                std::nullopt, false, stream);
          });
      return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
    }

    const int m_tile = hy4_q8_decode_m_tile(m);
    const bool use_tg = (m_tile >= 2);
    constexpr int rps = 4;
    constexpr int rows_per_tg = 4 * rps;
    const bool fast_k = (k % 512 == 0);
    const int grid_y = 4 * ((n + rows_per_tg - 1) / rows_per_tg);
    const int grid_z = (m + m_tile - 1) / m_tile;
    const std::tuple<int, int, int> grid{32, grid_y, grid_z};
    const std::tuple<int, int, int> tg{32, 4, 1};
    const uint64_t tmpl_key = uint64_t(m_tile) |
                              (uint64_t(fast_k ? 1 : 0) << 8) |
                              (uint64_t(clamp_act ? 1 : 0) << 9) |
                              (uint64_t(rps) << 10) |
                              (uint64_t(use_tg ? 1 : 0) << 14);
    auto out = dispatch_cached_kernel(
        kKernelQ8DualDownAdd, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_q8_dual_down_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"M_TILE", m_tile},
               {"ROWS_PER_SG", rps},
               {"USE_TG", use_tg},
               {"FAST_K", fast_k},
               {"CLAMP", clamp_act}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 q8_dual_down_add: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)routed_latent;
  (void)shared_gu;
  (void)w_up;
  (void)s_up;
  (void)w_sd;
  (void)s_sd;
  (void)limit;
  return nullptr;
#endif
}

bool mlx_hy4_router_topk(mlx_array *logits, mlx_array *bias, float scale,
                         float eps, int k, bool norm, mlx_array **out_ids,
                         mlx_array **out_weights) {
  if (out_ids)
    *out_ids = nullptr;
  if (out_weights)
    *out_weights = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return false;
    }
    if (!logits || !out_ids || !out_weights || !std::isfinite(scale) ||
        !std::isfinite(eps)) {
      return false;
    }
    const auto &la = *reinterpret_cast<array *>(logits);
    if (la.ndim() != 2 || la.dtype() != mlx::core::float32) {
      return false;
    }
    const int m = la.shape(0);
    const int e = la.shape(1);
    if (m < 1 || e < 1 || e > 1024 || k < 1 || k > 32 || k > e) {
      return false;
    }
    const bool has_bias = (bias != nullptr);
    array bc = cached_f32_scalar(0.0f);
    if (has_bias) {
      const auto &ba = *reinterpret_cast<array *>(bias);
      if (ba.ndim() != 1 || ba.dtype() != mlx::core::float32 ||
          ba.shape(0) != e) {
        return false;
      }
      bc = ensure_contiguous_aligned(ba, 1);
    }
    array lc = ensure_contiguous_aligned(la, 1);
    array params = cached_f32_pair(scale, eps);

    const int max_e = (e <= 512) ? 512 : 1024;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{128, m, 1};
    const std::tuple<int, int, int> tg{128, 1, 1};
    const uint64_t tmpl_key = uint64_t(k) | (uint64_t(max_e) << 8) |
                              (uint64_t(has_bias ? 1 : 0) << 20) |
                              (uint64_t(norm ? 1 : 0) << 21);
    const std::vector<array> inputs{lc, bc, params};
    const std::vector<Shape> out_shapes{{m, k}, {m, k}};
    const std::vector<Dtype> out_dtypes{mlx::core::uint32, mlx::core::float32};
    auto res = dispatch_cached_kernel(
        kKernelRouterTopk, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_router_topk_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"K", k},
               {"MAX_E", max_e},
               {"HAS_BIAS", has_bias},
               {"NORM", norm}},
              std::nullopt, false, stream);
        });
    auto ids = std::make_unique<array>(std::move(res[0]));
    auto w = std::make_unique<array>(std::move(res[1]));
    *out_ids = reinterpret_cast<mlx_array *>(ids.release());
    *out_weights = reinterpret_cast<mlx_array *>(w.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 router_topk: " << e.what() << std::endl;
    return false;
  }
#else
  (void)logits;
  (void)bias;
  (void)scale;
  (void)eps;
  (void)k;
  (void)norm;
  return false;
#endif
}

mlx_array *mlx_hy4_gguf_routed_decode(mlx_array *x, mlx_array *slots,
                                      mlx_array *weights, mlx_array *gate_buf,
                                      uint64_t gate_off, mlx_array *up_buf,
                                      uint64_t up_off, mlx_array *down_buf,
                                      uint64_t down_off, int n_experts, int m,
                                      int h, int down_bits, float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !slots || !weights || !gate_buf || !up_buf || !down_buf ||
        !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &sla = *reinterpret_cast<array *>(slots);
    const auto &wa = *reinterpret_cast<array *>(weights);
    const auto &gba = *reinterpret_cast<array *>(gate_buf);
    const auto &uba = *reinterpret_cast<array *>(up_buf);
    const auto &dba = *reinterpret_cast<array *>(down_buf);

    if (xa.ndim() != 2 || sla.ndim() != 2 || wa.ndim() != 2 ||
        xa.dtype() != mlx::core::float32 || sla.dtype() != mlx::core::uint32 ||
        wa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int rows = xa.shape(0);
    const int k = xa.shape(1);
    const int top = sla.shape(1);
    if (rows < 1 || top < 1 || top > 32 || sla.shape(0) != rows ||
        wa.shape() != Shape{rows, top}) {
      return nullptr;
    }
    if (n_experts < 1 || k < 256 || k % 256 != 0 || m < 256 || m % 256 != 0 ||
        h < 1 || (down_bits != 4 && down_bits != 6)) {
      return nullptr;
    }
    const uint64_t gu_row_bytes = uint64_t(k / 256) * 144ULL;
    const uint64_t gu_rows = uint64_t(n_experts) * uint64_t(m);
    if (gu_rows > UINT64_MAX / gu_row_bytes) {
      return nullptr;
    }
    const uint64_t gu_bytes = gu_rows * gu_row_bytes;
    const uint64_t d_blk = (down_bits == 6) ? 210ULL : 144ULL;
    const uint64_t d_align = (down_bits == 6) ? 2ULL : 4ULL;
    const uint64_t d_row_bytes = uint64_t(m / 256) * d_blk;
    const uint64_t d_rows = uint64_t(n_experts) * uint64_t(h);
    if (d_rows > UINT64_MAX / d_row_bytes) {
      return nullptr;
    }
    const uint64_t d_bytes = d_rows * d_row_bytes;
    if (!check_gguf_buf(gba, gate_off, gu_bytes, 4) ||
        !check_gguf_buf(uba, up_off, gu_bytes, 4) ||
        !check_gguf_buf(dba, down_off, d_bytes, d_align)) {
      return nullptr;
    }

    const int n_assign = rows * top;
    array row_ids = cached_row_ids(rows, top);
    array gu_offs = cached_u64_pair(gate_off, up_off);
    array gu_dims = cached_i32_pair(n_experts, m);
    array limit_arr = cached_f32_scalar(limit);

    array xc = ensure_strided_aligned(xa, 4);
    array slc = ensure_contiguous_aligned(sla, 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array gbc = ensure_contiguous_aligned(gba, 4);
    array ubc = ensure_contiguous_aligned(uba, 4);
    array dbc = ensure_contiguous_aligned(dba, static_cast<size_t>(d_align));

    array slots_flat = reshape(slc, {n_assign});
    const auto stream = default_stream(Device::gpu);
    const bool dedup = (rows >= 2) && (n_assign <= 512);
    array dedup_table = cached_zero_u32();
    if (dedup) {
      const int dd_tg_size = (n_assign <= 128) ? 128 : ((n_assign <= 256) ? 256 : 512);
      const std::tuple<int, int, int> dd_grid{dd_tg_size, 1, 1};
      const std::tuple<int, int, int> dd_tg{dd_tg_size, 1, 1};
      const uint64_t dd_tmpl = uint64_t(top) | (uint64_t(dd_tg_size) << 16);
      const std::vector<array> dd_inputs{slots_flat};
      const std::vector<Shape> dd_shapes{{n_assign + 1, 4}};
      const std::vector<Dtype> dd_dtypes{mlx::core::uint32};
      dedup_table = dispatch_cached_kernel(
          kKernelGgufDedupSlots, stream, dd_grid, dd_tg, dd_tmpl,
          dd_inputs, dd_shapes, dd_dtypes, [&] {
            return hy4_gguf_dedup_slots_kernel()(
                dd_inputs, dd_shapes, dd_dtypes, dd_grid, dd_tg,
                {{"TOP", top}, {"TG_SIZE", dd_tg_size}}, std::nullopt, false, stream);
          })[0];
    }

    const int max_match = dedup ? 4 : 1;
    const int gu_rps = 4;
    const int gu_nsg = 4;
    const int gu_rows_per_tg = gu_nsg * gu_rps;
    const bool fast_gu = (m % gu_rows_per_tg == 0);
    const uint64_t gu_tmpl = uint64_t(fast_gu ? 1 : 0) |
                             (uint64_t(dedup ? 1 : 0) << 1) |
                             (uint64_t(max_match) << 4) |
                             (uint64_t(gu_rps) << 8) |
                             (uint64_t(gu_nsg) << 12);
    const std::tuple<int, int, int> gu_grid{
        32, ((m + gu_rows_per_tg - 1) / gu_rows_per_tg) * gu_nsg, n_assign};
    const std::tuple<int, int, int> gu_tg{32, gu_nsg, 1};
    const std::vector<array> gu_inputs{xc,          row_ids, slots_flat,
                                       dedup_table, gbc,     ubc,
                                       gu_offs,     gu_dims, limit_arr};
    const std::vector<Shape> gu_out_shapes{{n_assign, m}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto hidden = dispatch_cached_kernel(
        kKernelGgufGateUp, stream, gu_grid, gu_tg, gu_tmpl, gu_inputs,
        gu_out_shapes, out_dtypes, [&] {
          return hy4_gguf_gate_up_kernel()(
              gu_inputs, gu_out_shapes, out_dtypes, gu_grid, gu_tg,
              {{"FAST", fast_gu},
               {"DEDUP", dedup},
               {"MAX_MATCH", max_match},
               {"NSG", gu_nsg},
               {"RPS", gu_rps}},
              std::nullopt, false, stream);
        })[0];

    array d_off_arr = cached_u64_scalar(down_off);
    array d_dims = cached_i32_pair(n_experts, h);
    constexpr int dp_rps = 4;
    constexpr int dp_nsg = 4;
    constexpr int dp_rows_per_tg = dp_nsg * dp_rps;
    const bool fast_dp = (h % dp_rows_per_tg == 0);
    const int dp_max_match = dedup ? 4 : 1;
    array weights_flat = reshape(wc, {n_assign});
    const std::tuple<int, int, int> dp_grid{
        32, ((h + dp_rows_per_tg - 1) / dp_rows_per_tg) * dp_nsg, n_assign};
    const std::tuple<int, int, int> dp_tg{32, dp_nsg, 1};
    const uint64_t dp_tmpl = uint64_t(down_bits) |
                             (uint64_t(fast_dp ? 1 : 0) << 8) |
                             (uint64_t(dedup ? 1 : 0) << 9) |
                             (uint64_t(dp_max_match) << 12) |
                             (uint64_t(dp_rps) << 16) |
                             (uint64_t(dp_nsg) << 20);
    const std::vector<array> dp_inputs{hidden, slots_flat, weights_flat,
                                       dedup_table, dbc, d_off_arr, d_dims};
    const std::vector<Shape> dp_out_shapes{{n_assign, h}};
    auto partial = dispatch_cached_kernel(
        kKernelGgufDownProj, stream, dp_grid, dp_tg, dp_tmpl, dp_inputs,
        dp_out_shapes, out_dtypes, [&] {
          return hy4_gguf_down_proj_kernel()(
              dp_inputs, dp_out_shapes, out_dtypes, dp_grid, dp_tg,
              {{"BITS", down_bits},
               {"FAST", fast_dp},
               {"DEDUP", dedup},
               {"MAX_MATCH", dp_max_match},
               {"NSG", dp_nsg},
               {"RPS", dp_rps}},
              std::nullopt, false, stream);
        })[0];

    const bool vec4 = (h % 4 == 0) && (size_t(n_assign) * size_t(h) >= 8) &&
                      (size_t(rows) * size_t(h) >= 8);
    const int cols = vec4 ? (h / 4) : h;
    const int tg_x = std::min(cols, 128);
    const int gx = ((cols + tg_x - 1) / tg_x) * tg_x;
    const std::tuple<int, int, int> red_grid{gx, rows, 1};
    const std::tuple<int, int, int> red_tg{tg_x, 1, 1};
    const uint64_t red_tmpl = uint64_t(top) | (uint64_t(vec4 ? 1 : 0) << 8);
    const std::vector<array> red_inputs{partial};
    const std::vector<Shape> red_out_shapes{{rows, h}};
    auto out = dispatch_cached_kernel(
        kKernelGgufDownReduce, stream, red_grid, red_tg, red_tmpl, red_inputs,
        red_out_shapes, out_dtypes, [&] {
          return hy4_gguf_down_reduce_kernel()(
              red_inputs, red_out_shapes, out_dtypes, red_grid, red_tg,
              {{"TOP", top}, {"VEC4", vec4}}, std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 gguf_routed_decode: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)slots;
  (void)weights;
  (void)gate_buf;
  (void)gate_off;
  (void)up_buf;
  (void)up_off;
  (void)down_buf;
  (void)down_off;
  (void)n_experts;
  (void)m;
  (void)h;
  (void)down_bits;
  (void)limit;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_gguf_prefill_indirect_gu(
    mlx_array *x, mlx_array *token_rows, mlx_array *tiles, mlx_array *gate_buf,
    uint64_t gate_off, mlx_array *up_buf, uint64_t up_off, int n_experts, int m,
    float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (prefill_nax_env_disabled() || !mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu) ||
        !mlx::core::metal::is_nax_available()) {
      return nullptr;
    }
    if (!x || !token_rows || !tiles || !gate_buf || !up_buf ||
        !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &tra = *reinterpret_cast<array *>(token_rows);
    const auto &ta = *reinterpret_cast<array *>(tiles);
    const auto &gba = *reinterpret_cast<array *>(gate_buf);
    const auto &uba = *reinterpret_cast<array *>(up_buf);

    if (xa.ndim() != 2 || tra.ndim() != 1 || ta.ndim() != 2 ||
        xa.dtype() != mlx::core::float32 || tra.dtype() != mlx::core::uint32 ||
        ta.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int n_tok = xa.shape(0);
    const int k = xa.shape(1);
    const int r = tra.shape(0);
    const int n_tiles = ta.shape(0);
    if (n_tok < 1 || r < 1 || n_tiles < 1 || ta.shape(1) != 3 ||
        n_experts < 1 || m < 1 || k < 256 || k % 256 != 0) {
      return nullptr;
    }
    const uint64_t row_bytes = uint64_t(k / 256) * 144ULL;
    const uint64_t total_rows = uint64_t(n_experts) * uint64_t(m);
    if (total_rows > UINT64_MAX / row_bytes) {
      return nullptr;
    }
    const uint64_t need_bytes = total_rows * row_bytes;
    if (!check_gguf_buf(gba, gate_off, need_bytes, 4) ||
        !check_gguf_buf(uba, up_off, need_bytes, 4)) {
      return nullptr;
    }

    array xc = ensure_strided_aligned(xa, 4);
    array trc = ensure_contiguous_aligned(tra, 1);
    array tc = ensure_contiguous_aligned(ta, 1);
    array gbc = ensure_contiguous_aligned(gba, 4);
    array ubc = ensure_contiguous_aligned(uba, 4);
    array gu_offs = cached_u64_pair(gate_off, up_off);
    array gu_dims = cached_i32_pair(n_experts, m);
    array limit_arr = cached_f32_scalar(limit);

    const bool hilo = gguf_gu_hilo_env();
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32 * ((m + 63) / 64), 8 * n_tiles, 1};
    const std::tuple<int, int, int> tg{32, 8, 1};
    const uint64_t tmpl_key = uint64_t(hilo ? 1 : 0);
    const std::vector<array> inputs{xc,  trc,     tc,      gbc,
                                    ubc, gu_offs, gu_dims, limit_arr};
    const std::vector<Shape> out_shapes{{r, m}};
    const std::vector<Dtype> out_dtypes{mlx::core::float16};
    auto out = dispatch_cached_kernel(
        kKernelGgufPrefillIndirectGu, stream, grid, tg, tmpl_key, inputs,
        out_shapes, out_dtypes, [&] {
          return hy4_gguf_prefill_indirect_gu_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg, {{"HILO", hilo}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 gguf_prefill_indirect_gu: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)token_rows;
  (void)tiles;
  (void)gate_buf;
  (void)gate_off;
  (void)up_buf;
  (void)up_off;
  (void)n_experts;
  (void)m;
  (void)limit;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_gguf_prefill_indirect_down(mlx_array *hidden,
                                              mlx_array *tiles,
                                              mlx_array *down_buf,
                                              uint64_t down_off, int n_experts,
                                              int h, int down_bits) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (prefill_nax_env_disabled() || !mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu) ||
        !mlx::core::metal::is_nax_available()) {
      return nullptr;
    }
    if (!hidden || !tiles || !down_buf) {
      return nullptr;
    }
    const auto &ha = *reinterpret_cast<array *>(hidden);
    const auto &ta = *reinterpret_cast<array *>(tiles);
    const auto &dba = *reinterpret_cast<array *>(down_buf);

    if (ha.ndim() != 2 || ta.ndim() != 2 ||
        (ha.dtype() != mlx::core::float16 && ha.dtype() != mlx::core::float32) ||
        ta.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const bool in_f16 = (ha.dtype() == mlx::core::float16);
    const int r = ha.shape(0);
    const int m = ha.shape(1);
    const int n_tiles = ta.shape(0);
    if (r < 1 || n_tiles < 1 || ta.shape(1) != 3 || n_experts < 1 || h < 1 ||
        m < 256 || m % 256 != 0 || (down_bits != 4 && down_bits != 6)) {
      return nullptr;
    }
    const uint64_t d_blk = (down_bits == 6) ? 210ULL : 144ULL;
    const uint64_t d_align = (down_bits == 6) ? 2ULL : 4ULL;
    const uint64_t row_bytes = uint64_t(m / 256) * d_blk;
    const uint64_t total_rows = uint64_t(n_experts) * uint64_t(h);
    if (total_rows > UINT64_MAX / row_bytes) {
      return nullptr;
    }
    const uint64_t need_bytes = total_rows * row_bytes;
    if (!check_gguf_buf(dba, down_off, need_bytes, d_align)) {
      return nullptr;
    }

    array hc = ensure_contiguous_aligned(ha, in_f16 ? 8 : 4);
    array tc = ensure_contiguous_aligned(ta, 1);
    array dbc = ensure_contiguous_aligned(dba, static_cast<size_t>(d_align));
    array d_off_arr = cached_u64_scalar(down_off);
    array d_dims = cached_i32_pair(n_experts, h);

    const bool out_f16 = !gguf_down_f32_env();
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32 * ((h + 63) / 64), 4 * n_tiles, 1};
    const std::tuple<int, int, int> tg{32, 4, 1};
    const uint64_t tmpl_key = uint64_t(down_bits) |
                              (uint64_t(in_f16 ? 1 : 0) << 8) |
                              (uint64_t(out_f16 ? 1 : 0) << 9);
    const std::vector<array> inputs{hc, tc, dbc, d_off_arr, d_dims};
    const std::vector<Shape> out_shapes{{r, h}};
    const std::vector<Dtype> out_dtypes{out_f16 ? mlx::core::float16
                                                : mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelGgufPrefillIndirectDown, stream, grid, tg, tmpl_key, inputs,
        out_shapes, out_dtypes, [&] {
          return hy4_gguf_prefill_indirect_down_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"BITS", down_bits}, {"IN_F16", in_f16}}, std::nullopt, false,
              stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 gguf_prefill_indirect_down: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)hidden;
  (void)tiles;
  (void)down_buf;
  (void)down_off;
  (void)n_experts;
  (void)h;
  (void)down_bits;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_ring_write(mlx_array *buf, mlx_array *src, int start_slot,
                              int cap) {
  try {
    if (!buf || !src || cap < 1 || start_slot < 0 || start_slot >= cap) {
      return nullptr;
    }
    const auto &ba = *reinterpret_cast<array *>(buf);
    const auto &sa = *reinterpret_cast<array *>(src);
    if (ba.ndim() != 2 || sa.ndim() != 2 || ba.dtype() != sa.dtype()) {
      return nullptr;
    }
    const int width = ba.shape(1);
    const int s_rows = sa.shape(0);
    if (ba.shape(0) != cap || width < 1 || sa.shape(1) != width || s_rows < 1 ||
        s_rows > cap) {
      return nullptr;
    }
    const Device dev =
#ifdef MLX_NODE_METAL_ENABLED
        (mlx::core::metal::is_available() && default_device() == Device::gpu)
            ? Device::gpu
            : Device::cpu;
#else
        Device::cpu;
#endif
    const auto stream = default_stream(dev);
    auto prim = std::make_shared<Hy4RingWrite>(
        stream, static_cast<uint32_t>(start_slot), static_cast<uint32_t>(cap));
    array out(ba.shape(), ba.dtype(), std::move(prim), {ba, sa});
    return reinterpret_cast<mlx_array *>(new array(std::move(out)));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 ring_write: " << e.what() << std::endl;
    return nullptr;
  }
}

mlx_array *mlx_hy4_indexer_score(mlx_array *q, mlx_array *w, mlx_array *k_pool,
                                 mlx_array *page_ids,
                                 mlx_array *visible_groups, int n_groups,
                                 int start_g) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!q || !w || !k_pool || n_groups < 1 || start_g < 0) {
      return nullptr;
    }
    const auto &qa = *reinterpret_cast<array *>(q);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &ka = *reinterpret_cast<array *>(k_pool);
    if (qa.ndim() != 3 || wa.ndim() != 2 || qa.dtype() != mlx::core::float32 ||
        wa.dtype() != mlx::core::float32 || ka.dtype() != mlx::core::float16) {
      return nullptr;
    }
    const int s = qa.shape(0);
    const int h = qa.shape(1);
    const int d = qa.shape(2);
    if (s < 1 || h < 1 || h > 16 || d != 128 || wa.shape() != Shape{s, h}) {
      return nullptr;
    }

    const bool paged = (page_ids != nullptr);
    int index_rows = 1;
    array pc = cached_zero_i32();
    if (paged) {
      const auto &pa = *reinterpret_cast<array *>(page_ids);
      if (ka.ndim() != 3 || ka.shape(2) != 128 || ka.shape(1) < 1 ||
          ka.shape(1) > 32 || pa.ndim() != 1 || pa.dtype() != mlx::core::int32 ||
          pa.shape(0) < 1) {
        return nullptr;
      }
      index_rows = ka.shape(1);
      if (int64_t(start_g) + int64_t(n_groups) >
          int64_t(pa.shape(0)) * int64_t(index_rows)) {
        return nullptr;
      }
      pc = ensure_contiguous_aligned(pa, 1);
    } else {
      if (ka.ndim() != 2 || ka.shape(1) != 128 ||
          int64_t(start_g) + int64_t(n_groups) > int64_t(ka.shape(0))) {
        return nullptr;
      }
    }

    const bool has_vis = (visible_groups != nullptr);
    array vc = cached_zero_i32();
    if (has_vis) {
      const auto &va = *reinterpret_cast<array *>(visible_groups);
      if (va.ndim() != 1 || va.dtype() != mlx::core::int32 ||
          va.shape(0) != s) {
        return nullptr;
      }
      vc = ensure_contiguous_aligned(va, 1);
    }

    array qc = ensure_contiguous_aligned(qa, 16);
    array wc = ensure_contiguous_aligned(wa, 1);
    array kc = ensure_contiguous_aligned(ka, 8);
    array params = cached_u32_triple(uint32_t(s), uint32_t(n_groups),
                                     uint32_t(start_g));

    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> tg{32, 8, 1};
    const uint64_t tmpl_key =
        uint64_t(paged ? 1 : 0) | (uint64_t(index_rows) << 8);
    const std::vector<array> inputs{qc, wc, kc, pc, params};
    const std::vector<Shape> out_shapes{{s, n_groups}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};

    std::vector<array> out;
    if (s <= 16) {
      if (mlx::core::metal::is_nax_available() &&
          !indexer_decode_nax_env_disabled()) {
        // Share each staged 32-key tile across all S query rows when N_g is
        // large enough to saturate GPU cores without S on grid.z.
        const bool share_s = (s > 1 && n_groups >= 1024);
        const std::tuple<int, int, int> nax_tg{32, 1, 1};
        const std::tuple<int, int, int> grid{
            32, (n_groups + 31) / 32, share_s ? 1 : s};
        const uint64_t nax_tmpl_key =
            tmpl_key | (uint64_t(share_s ? 1 : 0) << 16);
        out = dispatch_cached_kernel(
            kKernelIndexerScoreDecodeNax, stream, grid, nax_tg, nax_tmpl_key,
            inputs, out_shapes, out_dtypes, [&] {
              return hy4_indexer_score_decode_nax_kernel()(
                  inputs, out_shapes, out_dtypes, grid, nax_tg,
                  {{"PAGED", paged},
                   {"INDEX_ROWS", index_rows},
                   {"SHARE_S", share_s}},
                  std::nullopt, false, stream);
            });
      } else {
        const std::tuple<int, int, int> grid{32, ((n_groups + 7) / 8) * 8, s};
        out = dispatch_cached_kernel(
            kKernelIndexerScoreDecode, stream, grid, tg, tmpl_key, inputs,
            out_shapes, out_dtypes, [&] {
              return hy4_indexer_score_decode_kernel()(
                  inputs, out_shapes, out_dtypes, grid, tg,
                  {{"PAGED", paged}, {"INDEX_ROWS", index_rows}}, std::nullopt,
                  false, stream);
            });
      }
    } else if (mlx::core::metal::is_nax_available()) {
      const std::vector<array> prefill_inputs{qc, wc, kc, pc, vc, params};
      const uint64_t prefill_tmpl_key =
          tmpl_key | (uint64_t(has_vis ? 1 : 0) << 17);
      const std::tuple<int, int, int> nax_tg{32, 4, 1};
      const std::tuple<int, int, int> grid{32, ((n_groups + 63) / 64) * 4,
                                           (s + 63) / 64};
      out = dispatch_cached_kernel(
          kKernelIndexerScorePrefillNax, stream, grid, nax_tg, prefill_tmpl_key,
          prefill_inputs, out_shapes, out_dtypes, [&] {
            return hy4_indexer_score_prefill_nax_kernel()(
                prefill_inputs, out_shapes, out_dtypes, grid, nax_tg,
                {{"PAGED", paged},
                 {"INDEX_ROWS", index_rows},
                 {"HAS_VIS", has_vis}},
                std::nullopt, false, stream);
          });
    } else {
      const std::vector<array> prefill_inputs{qc, wc, kc, pc, vc, params};
      const uint64_t prefill_tmpl_key =
          tmpl_key | (uint64_t(has_vis ? 1 : 0) << 17);
      const std::tuple<int, int, int> grid{32, ((n_groups + 31) / 32) * 8,
                                           (s + 15) / 16};
      out = dispatch_cached_kernel(
          kKernelIndexerScorePrefill, stream, grid, tg, prefill_tmpl_key,
          prefill_inputs, out_shapes, out_dtypes, [&] {
            return hy4_indexer_score_prefill_kernel()(
                prefill_inputs, out_shapes, out_dtypes, grid, tg,
                {{"PAGED", paged},
                 {"INDEX_ROWS", index_rows},
                 {"HAS_VIS", has_vis}},
                std::nullopt, false, stream);
          });
    }
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 indexer_score: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)q;
  (void)w;
  (void)k_pool;
  (void)page_ids;
  (void)visible_groups;
  (void)n_groups;
  (void)start_g;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_indexer_topk_cells(mlx_array *scores,
                                      mlx_array *visible_groups,
                                      mlx_array *tail_starts,
                                      mlx_array *tail_lens, mlx_array *page_ids,
                                      int n_top, int kpool, int kmax,
                                      int block_size) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!scores || !visible_groups || !tail_starts || !tail_lens ||
        n_top < 1 || n_top > 512 || kpool < 1 || kmax < 1 ||
        int64_t(n_top) * int64_t(kpool) > int64_t(kmax)) {
      return nullptr;
    }
    const auto &sa = *reinterpret_cast<array *>(scores);
    const auto &va = *reinterpret_cast<array *>(visible_groups);
    const auto &tsa = *reinterpret_cast<array *>(tail_starts);
    const auto &tla = *reinterpret_cast<array *>(tail_lens);
    if (sa.ndim() != 2 || sa.dtype() != mlx::core::float32 ||
        va.ndim() != 1 || va.dtype() != mlx::core::int32 ||
        tsa.ndim() != 1 || tsa.dtype() != mlx::core::int32 ||
        tla.ndim() != 1 || tla.dtype() != mlx::core::int32) {
      return nullptr;
    }
    const int s = sa.shape(0);
    const int n_g = sa.shape(1);
    if (s < 1 || n_g < 1 || va.shape(0) != s || tsa.shape(0) != s ||
        tla.shape(0) != s) {
      return nullptr;
    }

    const bool paged = (page_ids != nullptr);
    int bs = block_size;
    array pc = cached_zero_i32();
    if (paged) {
      if (bs < 1) {
        return nullptr;
      }
      const auto &pa = *reinterpret_cast<array *>(page_ids);
      if (pa.ndim() != 1 || pa.dtype() != mlx::core::int32 || pa.shape(0) < 1) {
        return nullptr;
      }
      pc = ensure_contiguous_aligned(pa, 1);
    } else {
      bs = 32;
    }

    array sc = ensure_contiguous_aligned(sa, 1);
    array vc = ensure_contiguous_aligned(va, 1);
    array tsc = ensure_contiguous_aligned(tsa, 1);
    array tlc = ensure_contiguous_aligned(tla, 1);
    array params = cached_u32_quad(uint32_t(n_top), uint32_t(kpool),
                                   uint32_t(kmax), uint32_t(bs));

    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{256, s, 1};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(paged ? 1 : 0);
    const std::vector<array> inputs{sc, vc, tsc, tlc, pc, params};
    const std::vector<Shape> out_shapes{{s, kmax}};
    const std::vector<Dtype> out_dtypes{mlx::core::int32};
    auto out = dispatch_cached_kernel(
        kKernelIndexerTopkCells, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_indexer_topk_cells_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg, {{"PAGED", paged}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 indexer_topk_cells: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)scores;
  (void)visible_groups;
  (void)tail_starts;
  (void)tail_lens;
  (void)page_ids;
  (void)n_top;
  (void)kpool;
  (void)kmax;
  (void)block_size;
  return nullptr;
#endif
}

bool mlx_hy4_moe_prefill_sort_tiles(mlx_array *ids, int n_experts, int tile_m,
                                    mlx_array **out_token_rows,
                                    mlx_array **out_tiles,
                                    mlx_array **out_inverse,
                                    mlx_array **out_sorted_slots) {
  if (out_token_rows)
    *out_token_rows = nullptr;
  if (out_tiles)
    *out_tiles = nullptr;
  if (out_inverse)
    *out_inverse = nullptr;
  if (out_sorted_slots)
    *out_sorted_slots = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return false;
    }
    if (!ids || !out_token_rows || !out_tiles || !out_inverse ||
        !out_sorted_slots || n_experts < 1 || n_experts > 1024 ||
        (tile_m != 32 && tile_m != 64)) {
      return false;
    }
    const auto &ida = *reinterpret_cast<array *>(ids);
    if (ida.ndim() != 2 || ida.dtype() != mlx::core::uint32) {
      return false;
    }
    const int rows = ida.shape(0);
    const int k = ida.shape(1);
    if (rows < 1 || k < 1 || k > 32) {
      return false;
    }
    const int64_t r64 = int64_t(rows) * int64_t(k);
    if (r64 > INT32_MAX) {
      return false;
    }
    const int r = int(r64);
    const int max_tiles = std::min(r, (r + tile_m - 1) / tile_m + n_experts);

    array idc = ensure_contiguous_aligned(ida, 1);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tile_tmpl = uint64_t(tile_m);

    std::unique_ptr<array> tr;
    std::unique_ptr<array> tl;
    std::unique_ptr<array> inv;
    std::unique_ptr<array> sl;

    if (r <= 2048) {
      array params = cached_u32_quad(uint32_t(r), uint32_t(k),
                                     uint32_t(n_experts), uint32_t(max_tiles));
      const std::tuple<int, int, int> grid{256, 1, 1};
      const std::vector<array> inputs{idc, params};
      const std::vector<Shape> out_shapes{
          {r}, {max_tiles, 3}, {rows, k}, {r}};
      const std::vector<Dtype> out_dtypes{
          mlx::core::uint32, mlx::core::uint32, mlx::core::uint32,
          mlx::core::uint32};
      auto res = dispatch_cached_kernel(
          kKernelMoePrefillSortTiles, stream, grid, tg, tile_tmpl, inputs,
          out_shapes, out_dtypes, [&] {
            return hy4_moe_prefill_sort_tiles_kernel()(
                inputs, out_shapes, out_dtypes, grid, tg, {{"TILE_M", tile_m}},
                std::nullopt, false, stream);
          });
      tr = std::make_unique<array>(std::move(res[0]));
      tl = std::make_unique<array>(std::move(res[1]));
      inv = std::make_unique<array>(std::move(res[2]));
      sl = std::make_unique<array>(std::move(res[3]));
    } else {
      const int n_tgs = std::clamp((r + 2047) / 2048, 2, 32);
      const int chunk = (r + n_tgs - 1) / n_tgs;
      array p_blk = cached_u32_five(uint32_t(r), uint32_t(k),
                                    uint32_t(n_experts), uint32_t(n_tgs),
                                    uint32_t(chunk));
      array p_scan = cached_u32_quad(uint32_t(r), uint32_t(n_experts),
                                     uint32_t(max_tiles), uint32_t(n_tgs));
      const std::tuple<int, int, int> blk_grid{256 * n_tgs, 1, 1};
      const std::tuple<int, int, int> scan_grid{256, 1, 1};

      const std::vector<array> hist_in{idc, p_blk};
      const std::vector<Shape> hist_shapes{{n_tgs, n_experts}};
      const std::vector<Dtype> hist_dtypes{mlx::core::uint32};
      auto block_counts = dispatch_cached_kernel(
          kKernelMoePrefillSortHist, stream, blk_grid, tg, 0, hist_in,
          hist_shapes, hist_dtypes, [&] {
            return hy4_moe_prefill_sort_hist_kernel()(
                hist_in, hist_shapes, hist_dtypes, blk_grid, tg, {},
                std::nullopt, false, stream);
          })[0];

      const std::vector<array> scan_in{block_counts, p_scan};
      const std::vector<Shape> scan_shapes{
          {n_tgs, n_experts}, {max_tiles, 3}, {1}};
      const std::vector<Dtype> scan_dtypes{
          mlx::core::uint32, mlx::core::uint32, mlx::core::uint32};
      auto scan_res = dispatch_cached_kernel(
          kKernelMoePrefillSortScanTiles, stream, scan_grid, tg, tile_tmpl,
          scan_in, scan_shapes, scan_dtypes, [&] {
            return hy4_moe_prefill_sort_scan_tiles_kernel()(
                scan_in, scan_shapes, scan_dtypes, scan_grid, tg,
                {{"TILE_M", tile_m}}, std::nullopt, false, stream);
          });

      const std::vector<array> scat_in{idc, scan_res[0], scan_res[2], p_blk};
      const std::vector<Shape> scat_shapes{{r}, {rows, k}, {r}};
      const std::vector<Dtype> scat_dtypes{
          mlx::core::uint32, mlx::core::uint32, mlx::core::uint32};
      auto scat_res = dispatch_cached_kernel(
          kKernelMoePrefillSortScatter, stream, blk_grid, tg, 0, scat_in,
          scat_shapes, scat_dtypes, [&] {
            return hy4_moe_prefill_sort_scatter_kernel()(
                scat_in, scat_shapes, scat_dtypes, blk_grid, tg, {},
                std::nullopt, false, stream);
          });

      tr = std::make_unique<array>(std::move(scat_res[0]));
      tl = std::make_unique<array>(std::move(scan_res[1]));
      inv = std::make_unique<array>(std::move(scat_res[1]));
      sl = std::make_unique<array>(std::move(scat_res[2]));
    }
    *out_token_rows = reinterpret_cast<mlx_array *>(tr.release());
    *out_tiles = reinterpret_cast<mlx_array *>(tl.release());
    *out_inverse = reinterpret_cast<mlx_array *>(inv.release());
    *out_sorted_slots = reinterpret_cast<mlx_array *>(sl.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 moe_prefill_sort_tiles: " << e.what() << std::endl;
    return false;
  }
#else
  (void)ids;
  (void)n_experts;
  (void)tile_m;
  return false;
#endif
}

mlx_array *mlx_hy4_affine4_routed_decode(mlx_array *x, mlx_array *slots,
                                         mlx_array *weights, mlx_array *wg,
                                         mlx_array *sg, mlx_array *bg,
                                         mlx_array *wu, mlx_array *su,
                                         mlx_array *bu, mlx_array *wd,
                                         mlx_array *sd, mlx_array *bd,
                                         float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !slots || !weights || !wg || !sg || !bg || !wu || !su || !bu ||
        !wd || !sd || !bd || !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &sla = *reinterpret_cast<array *>(slots);
    const auto &wa = *reinterpret_cast<array *>(weights);
    const auto &wga = *reinterpret_cast<array *>(wg);
    const auto &sga = *reinterpret_cast<array *>(sg);
    const auto &bga = *reinterpret_cast<array *>(bg);
    const auto &wua = *reinterpret_cast<array *>(wu);
    const auto &sua = *reinterpret_cast<array *>(su);
    const auto &bua = *reinterpret_cast<array *>(bu);
    const auto &wda = *reinterpret_cast<array *>(wd);
    const auto &sda = *reinterpret_cast<array *>(sd);
    const auto &bda = *reinterpret_cast<array *>(bd);

    if (xa.ndim() != 2 || sla.ndim() != 2 || wa.ndim() != 2 ||
        xa.dtype() != mlx::core::float32 || sla.dtype() != mlx::core::uint32 ||
        wa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int rows = xa.shape(0);
    const int k = xa.shape(1);
    const int top = sla.shape(1);
    if (rows < 1 || rows > 8 || top < 1 || top > 32 || sla.shape(0) != rows ||
        wa.shape() != Shape{rows, top}) {
      return nullptr;
    }
    int e = 0, m = 0, h = 0;
    if (!validate_affine4_experts(wga, sga, bga, wua, sua, bua, wda, sda, bda,
                                  k, e, m, h)) {
      return nullptr;
    }
    (void)e;

    const int n_assign = rows * top;
    const bool fast_k = (k % 512 == 0);
    const bool fast_m_down = (m % 512 == 0);
    array row_ids = cached_row_ids(rows, top);
    array limit_arr = cached_f32_scalar(limit);

    array xc = ensure_strided_aligned(xa, 4);
    array slc = ensure_contiguous_aligned(sla, 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array wgc = ensure_contiguous_aligned(wga, fast_k ? 2 : 1);
    array sgc = ensure_contiguous_aligned(sga, 1);
    array bgc = ensure_contiguous_aligned(bga, 1);
    array wuc = ensure_contiguous_aligned(wua, fast_k ? 2 : 1);
    array suc = ensure_contiguous_aligned(sua, 1);
    array buc = ensure_contiguous_aligned(bua, 1);
    array wdc = ensure_contiguous_aligned(wda, fast_m_down ? 2 : 1);
    array sdc = ensure_contiguous_aligned(sda, 1);
    array bdc = ensure_contiguous_aligned(bda, 1);

    array slots_flat = reshape(slc, {n_assign});
    const auto stream = default_stream(Device::gpu);
    const bool dedup = (rows >= 2) && (n_assign <= 128);
    array dedup_table = cached_zero_u32();
    if (dedup) {
      const std::tuple<int, int, int> dd_grid{128, 1, 1};
      const std::tuple<int, int, int> dd_tg{128, 1, 1};
      const uint64_t dd_tmpl = uint64_t(top) | (uint64_t(128) << 16);
      const std::vector<array> dd_inputs{slots_flat};
      const std::vector<Shape> dd_shapes{{n_assign + 1, 4}};
      const std::vector<Dtype> dd_dtypes{mlx::core::uint32};
      dedup_table = dispatch_cached_kernel(
          kKernelGgufDedupSlots, stream, dd_grid, dd_tg, dd_tmpl,
          dd_inputs, dd_shapes, dd_dtypes, [&] {
            return hy4_gguf_dedup_slots_kernel()(
                dd_inputs, dd_shapes, dd_dtypes, dd_grid, dd_tg,
                {{"TOP", top}, {"TG_SIZE", 128}}, std::nullopt, false, stream);
          })[0];
    }

    const int max_match = dedup ? 4 : 1;
    const int gu_rps = dedup ? 2 : 4;
    const int gu_nsg = 4;
    const int gu_rows_per_tg = gu_nsg * gu_rps;
    const bool fast_m = (m % gu_rows_per_tg == 0);
    const uint64_t gu_tmpl = uint64_t(fast_k ? 1 : 0) |
                             (uint64_t(fast_m ? 1 : 0) << 1) |
                             (uint64_t(dedup ? 1 : 0) << 2) |
                             (uint64_t(max_match) << 4) |
                             (uint64_t(gu_rps) << 8) |
                             (uint64_t(gu_nsg) << 12);
    const std::tuple<int, int, int> gu_grid{
        32, ((m + gu_rows_per_tg - 1) / gu_rows_per_tg) * gu_nsg, n_assign};
    const std::tuple<int, int, int> gu_tg{32, gu_nsg, 1};
    const std::vector<array> gu_inputs{xc,  row_ids, slots_flat, dedup_table,
                                       wgc, sgc,     bgc,        wuc,
                                       suc, buc,     limit_arr};
    const std::vector<Shape> gu_out_shapes{{n_assign, m}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto hidden = dispatch_cached_kernel(
        kKernelAffine4GateUp, stream, gu_grid, gu_tg, gu_tmpl, gu_inputs,
        gu_out_shapes, out_dtypes, [&] {
          return hy4_affine4_gate_up_kernel()(
              gu_inputs, gu_out_shapes, out_dtypes, gu_grid, gu_tg,
              {{"FAST_K", fast_k},
               {"FAST_M", fast_m},
               {"DEDUP", dedup},
               {"MAX_MATCH", max_match},
               {"NSG", gu_nsg},
               {"RPS", gu_rps}},
              std::nullopt, false, stream);
        })[0];

    const bool fast_h = (h % 4 == 0);
    const bool dc_dedup = dedup;
    const int dc_max_match = dc_dedup ? 4 : 1;
    constexpr int rps = 4;
    constexpr int ksg = 8;
    const std::tuple<int, int, int> dc_grid{
        32, ((h + rps - 1) / rps) * ksg, dc_dedup ? 1 : rows};
    const std::tuple<int, int, int> dc_tg{32, ksg, 1};
    const uint64_t dc_tmpl = uint64_t(top) |
                             (uint64_t(fast_m_down ? 1 : 0) << 8) |
                             (uint64_t(fast_h ? 1 : 0) << 9) |
                             (uint64_t(dc_dedup ? 1 : 0) << 10) |
                             (uint64_t(dc_max_match) << 12);
    const std::vector<array> dc_inputs{hidden, slc, wc, dedup_table,
                                       wdc,    sdc, bdc};
    const std::vector<Shape> dc_out_shapes{{rows, h}};
    auto out = dispatch_cached_kernel(
        kKernelAffine4DownCombine, stream, dc_grid, dc_tg, dc_tmpl, dc_inputs,
        dc_out_shapes, out_dtypes, [&] {
          return hy4_affine4_down_combine_kernel()(
              dc_inputs, dc_out_shapes, out_dtypes, dc_grid, dc_tg,
              {{"TOP", top},
               {"FAST_M", fast_m_down},
               {"FAST_H", fast_h},
               {"DEDUP_ROWS", dc_dedup},
               {"MAX_MATCH", dc_max_match}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 affine4_routed_decode: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)slots;
  (void)weights;
  (void)wg;
  (void)sg;
  (void)bg;
  (void)wu;
  (void)su;
  (void)bu;
  (void)wd;
  (void)sd;
  (void)bd;
  (void)limit;
  return nullptr;
#endif
}

bool mlx_hy4_mla_prep_q_kv(mlx_array *all, mlx_array *q_norm_w,
                           mlx_array *kv_norm_w, mlx_array *pos,
                           mlx_array *inv_freq, int q_lora, int kv_lora,
                           int rope_dim, float norm_eps, bool out_f16,
                           bool split_out, mlx_array **out_qr,
                           mlx_array **out_kv0, mlx_array **out_kv1) {
  if (out_qr)
    *out_qr = nullptr;
  if (out_kv0)
    *out_kv0 = nullptr;
  if (out_kv1)
    *out_kv1 = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return false;
    }
    if (!all || !q_norm_w || !kv_norm_w || !pos || !inv_freq || !out_qr ||
        !out_kv0 || (split_out && !out_kv1) || !std::isfinite(norm_eps)) {
      return false;
    }
    if (q_lora < 1 || kv_lora < 1 || rope_dim < 2 || (rope_dim % 2) != 0) {
      return false;
    }
    const int kv_row = kv_lora + rope_dim;
    if (split_out && (kv_row % 2) != 0) {
      return false;
    }
    const auto &all_a = *reinterpret_cast<array *>(all);
    const auto &qnw_a = *reinterpret_cast<array *>(q_norm_w);
    const auto &kvnw_a = *reinterpret_cast<array *>(kv_norm_w);
    const auto &pos_a = *reinterpret_cast<array *>(pos);
    const auto &inv_a = *reinterpret_cast<array *>(inv_freq);

    if (all_a.ndim() != 2 || qnw_a.ndim() != 1 || kvnw_a.ndim() != 1 ||
        pos_a.ndim() != 1 || inv_a.ndim() != 1) {
      return false;
    }
    const bool pos_is_i32 = (pos_a.dtype() == mlx::core::int32);
    if (all_a.dtype() != mlx::core::float32 ||
        qnw_a.dtype() != mlx::core::float32 ||
        kvnw_a.dtype() != mlx::core::float32 ||
        inv_a.dtype() != mlx::core::float32 ||
        (!pos_is_i32 && pos_a.dtype() != mlx::core::float32)) {
      return false;
    }
    const int m = all_a.shape(0);
    const int total = all_a.shape(1);
    if (m < 1 || total < q_lora + kv_row || qnw_a.shape(0) != q_lora ||
        kvnw_a.shape(0) != kv_lora || pos_a.shape(0) != m ||
        inv_a.shape(0) != rope_dim / 2) {
      return false;
    }

    const int half_kv = kv_row / 2;
    const bool dims_vec4 = (q_lora >= 8) && (kv_lora >= 8) &&
                           (q_lora % 4 == 0) && (kv_lora % 4 == 0) &&
                           (rope_dim % 4 == 0) &&
                           (!split_out || (half_kv % 4 == 0));
    array all_c = ensure_strided_aligned(all_a, dims_vec4 ? 4 : 1);
    const bool vec4 = dims_vec4 && (all_c.strides(0) % 4 == 0) &&
                      (all_c.offset() % (4 * all_c.itemsize()) == 0) &&
                      (all_c.size() >= 8);
    array qnw_c = ensure_contiguous_aligned(qnw_a, vec4 ? 4 : 1);
    array kvnw_c = ensure_contiguous_aligned(kvnw_a, vec4 ? 4 : 1);
    array pos_c = ensure_contiguous_aligned(pos_a, 1);
    array inv_c = ensure_contiguous_aligned(inv_a, 1);
    array eps_arr = cached_f32_scalar(norm_eps);

    const Dtype kv_dt = out_f16 ? mlx::core::float16 : mlx::core::float32;
    const Shape kv0_shape = split_out ? Shape{m, half_kv} : Shape{m, kv_row};
    const Shape kv1_shape = split_out ? Shape{m, half_kv} : Shape{1};
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{256, m, 1};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(vec4 ? 1 : 0) |
                              (uint64_t(out_f16 ? 1 : 0) << 1) |
                              (uint64_t(split_out ? 1 : 0) << 2) |
                              (uint64_t(pos_is_i32 ? 1 : 0) << 3);
    const std::vector<array> inputs{all_c, qnw_c, kvnw_c, pos_c, inv_c, eps_arr};
    const std::vector<Shape> out_shapes{{m, q_lora}, kv0_shape, kv1_shape};
    const std::vector<Dtype> out_dtypes{mlx::core::float32, kv_dt, kv_dt};
    auto res = dispatch_cached_kernel(
        kKernelMlaPrepQKv, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_mla_prep_q_kv_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"VEC4", vec4}, {"OUT_F16", out_f16}, {"SPLIT_OUT", split_out}},
              std::nullopt, false, stream);
        });
    auto qr_arr = std::make_unique<array>(std::move(res[0]));
    auto kv0_arr = std::make_unique<array>(std::move(res[1]));
    if (split_out) {
      auto kv1_arr = std::make_unique<array>(std::move(res[2]));
      *out_kv1 = reinterpret_cast<mlx_array *>(kv1_arr.release());
    }
    *out_qr = reinterpret_cast<mlx_array *>(qr_arr.release());
    *out_kv0 = reinterpret_cast<mlx_array *>(kv0_arr.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 mla_prep_q_kv: " << e.what() << std::endl;
    return false;
  }
#else
  (void)all;
  (void)q_norm_w;
  (void)kv_norm_w;
  (void)pos;
  (void)inv_freq;
  (void)q_lora;
  (void)kv_lora;
  (void)rope_dim;
  (void)norm_eps;
  (void)out_f16;
  (void)split_out;
  return false;
#endif
}

mlx_array *mlx_hy4_q8_grouped_k192_rope(mlx_array *x, mlx_array *w,
                                        mlx_array *scales, mlx_array *biases,
                                        mlx_array *pos, mlx_array *inv_freq) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w || !scales || !pos || !inv_freq) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    const auto &pa = *reinterpret_cast<array *>(pos);
    const auto &inva = *reinterpret_cast<array *>(inv_freq);
    if (xa.ndim() != 3 || wa.ndim() != 2 || sa.ndim() != 2 || pa.ndim() != 1 ||
        inva.ndim() != 1) {
      return nullptr;
    }
    const bool pos_is_i32 = (pa.dtype() == mlx::core::int32);
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::uint32 ||
        inva.dtype() != mlx::core::float32 ||
        (!pos_is_i32 && pa.dtype() != mlx::core::float32)) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int g = xa.shape(1);
    const int qk = xa.shape(2);
    const int half_rd = inva.shape(0);
    const int rd = half_rd * 2;
    if (m < 1 || m > 16 || g < 1 || half_rd < 1 || qk != 192 + rd ||
        (qk % 4) != 0 || pa.shape(0) != m || wa.shape(0) < g ||
        (wa.shape(0) % g) != 0 || wa.shape(1) != 48) {
      return nullptr;
    }
    const int n = wa.shape(0) / g;
    if (n < 1 || sa.shape() != Shape{g * n, 6}) {
      return nullptr;
    }
    const bool sym_f16 = (biases == nullptr);
    array bc = cached_zero_f16();
    if (sym_f16) {
      if (sa.dtype() != mlx::core::float16) {
        return nullptr;
      }
    } else {
      const auto &ba = *reinterpret_cast<array *>(biases);
      if (ba.ndim() != 2 || ba.shape() != Shape{g * n, 6}) {
        return nullptr;
      }
      const bool both_f32 = (sa.dtype() == mlx::core::float32 &&
                             ba.dtype() == mlx::core::float32);
      const bool both_f16 = (sa.dtype() == mlx::core::float16 &&
                             ba.dtype() == mlx::core::float16);
      if (!both_f32 && !both_f16) {
        return nullptr;
      }
      bc = ensure_contiguous_aligned(ba, 1);
    }

    array xc = ensure_strided_aligned(xa, 4);
    array wc = ensure_contiguous_aligned(wa, 1);
    array sc = ensure_contiguous_aligned(sa, 1);
    array pc = ensure_contiguous_aligned(pa, 1);
    array invc = ensure_contiguous_aligned(inva, 1);

    const int m_tile = hy4_q8_short_m_tile(m);
    const int grid_y = 4 * ((n + 15) / 16);
    const int grid_z = g * ((m + m_tile - 1) / m_tile);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32, grid_y, grid_z};
    const std::tuple<int, int, int> tg{32, 4, 1};
    const bool scales_f16 = (sa.dtype() == mlx::core::float16);
    const uint64_t tmpl_key = uint64_t(m_tile) |
                              (uint64_t(sym_f16 ? 1 : 0) << 8) |
                              (uint64_t(scales_f16 ? 1 : 0) << 9) |
                              (uint64_t(pos_is_i32 ? 1 : 0) << 10);
    const std::vector<array> inputs{xc, wc, sc, bc, pc, invc};
    const std::vector<Shape> out_shapes{{m, g, n + rd}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelQ8GroupedK192Rope, stream, grid, tg, tmpl_key, inputs,
        out_shapes, out_dtypes, [&] {
          return hy4_q8_grouped_k192_rope_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"M_TILE", m_tile}, {"SYM_F16", sym_f16}}, std::nullopt, false,
              stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 q8_grouped_k192_rope: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  (void)pos;
  (void)inv_freq;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_kpool_compress(mlx_array *tail_buf, mlx_array *tok,
                                  mlx_array *w_val, mlx_array *w_gate,
                                  mlx_array *ape, mlx_array *comp_norm,
                                  mlx_array *inv_freq, int n_groups,
                                  int first_group, int open_len,
                                  int tail_start_slot, int tail_cap,
                                  float norm_eps, int out_dtype_code) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!tok || !w_val || !w_gate || !ape || !comp_norm || !inv_freq ||
        !std::isfinite(norm_eps)) {
      return nullptr;
    }
    if (n_groups < 1 || first_group < 0 || open_len < 0 ||
        uint64_t(first_group) + uint64_t(n_groups) > (1ULL << 22)) {
      return nullptr;
    }
    Dtype out_dt = mlx::core::float32;
    if (out_dtype_code == 1) {
      out_dt = mlx::core::float16;
    } else if (out_dtype_code == 2) {
      out_dt = mlx::core::bfloat16;
    } else if (out_dtype_code != 0) {
      return nullptr;
    }

    const auto &tok_a = *reinterpret_cast<array *>(tok);
    const auto &wv_a = *reinterpret_cast<array *>(w_val);
    const auto &wg_a = *reinterpret_cast<array *>(w_gate);
    const auto &ape_a = *reinterpret_cast<array *>(ape);
    const auto &cn_a = *reinterpret_cast<array *>(comp_norm);
    const auto &inv_a = *reinterpret_cast<array *>(inv_freq);

    if (tok_a.ndim() != 2 || wv_a.ndim() != 2 || wg_a.ndim() != 2 ||
        ape_a.ndim() != 2 || cn_a.ndim() != 1 || inv_a.ndim() != 1) {
      return nullptr;
    }
    if (tok_a.dtype() != mlx::core::float32 ||
        wv_a.dtype() != mlx::core::float32 ||
        wg_a.dtype() != mlx::core::float32 ||
        ape_a.dtype() != mlx::core::float32 ||
        cn_a.dtype() != mlx::core::float32 ||
        inv_a.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int half_rd = inv_a.shape(0);
    if (tok_a.shape(1) != 128 || wv_a.shape() != Shape{128, 128} ||
        wg_a.shape() != Shape{128, 128} || ape_a.shape() != Shape{4, 128} ||
        cn_a.shape(0) != 128 || half_rd < 1 || half_rd >= 64) {
      return nullptr;
    }
    if (int64_t(open_len) + int64_t(tok_a.shape(0)) < int64_t(n_groups) * 4) {
      return nullptr;
    }

    array tail_c = cached_zero_f32();
    int eff_start = 0;
    int eff_cap = 1;
    if (open_len > 0) {
      if (!tail_buf || tail_cap < 1 || tail_start_slot < 0 ||
          tail_start_slot >= tail_cap || open_len > tail_cap) {
        return nullptr;
      }
      const auto &tb_a = *reinterpret_cast<array *>(tail_buf);
      if (tb_a.ndim() != 2 || tb_a.dtype() != mlx::core::float32 ||
          tb_a.shape(0) != tail_cap || tb_a.shape(1) != 128) {
        return nullptr;
      }
      tail_c = ensure_contiguous_aligned(tb_a, 4);
      eff_start = tail_start_slot;
      eff_cap = tail_cap;
    }

    array tok_c = ensure_contiguous_aligned(tok_a, 4);
    array wv_c = ensure_contiguous_aligned(wv_a, 4);
    array wg_c = ensure_contiguous_aligned(wg_a, 4);
    array ape_c = ensure_contiguous_aligned(ape_a, 1);
    array cn_c = ensure_contiguous_aligned(cn_a, 1);
    array inv_c = ensure_contiguous_aligned(inv_a, 1);
    array params = cached_u32_five(uint32_t(n_groups), uint32_t(first_group),
                                   uint32_t(open_len), uint32_t(eff_start),
                                   uint32_t(eff_cap));
    array eps_arr = cached_f32_scalar(norm_eps);

    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32, 8, n_groups};
    const std::tuple<int, int, int> tg{32, 8, 1};
    const uint64_t tmpl_key = uint64_t(out_dtype_code);
    const std::vector<array> inputs{tail_c, tok_c, wv_c,   wg_c,   ape_c,
                                    cn_c,   inv_c, params, eps_arr};
    const std::vector<Shape> out_shapes{{n_groups, 128}};
    const std::vector<Dtype> out_dtypes{out_dt};
    auto out = dispatch_cached_kernel(
        kKernelKpoolCompress, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_kpool_compress_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg, {}, std::nullopt, false,
              stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 kpool_compress: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)tail_buf;
  (void)tok;
  (void)w_val;
  (void)w_gate;
  (void)ape;
  (void)comp_norm;
  (void)inv_freq;
  (void)n_groups;
  (void)first_group;
  (void)open_len;
  (void)tail_start_slot;
  (void)tail_cap;
  (void)norm_eps;
  (void)out_dtype_code;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_partial_rope_tail(mlx_array *x, mlx_array *pos,
                                     mlx_array *inv_freq, bool out_f16) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !pos || !inv_freq) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &pa = *reinterpret_cast<array *>(pos);
    const auto &inva = *reinterpret_cast<array *>(inv_freq);
    if ((xa.ndim() != 2 && xa.ndim() != 3) || pa.ndim() != 1 ||
        inva.ndim() != 1) {
      return nullptr;
    }
    const bool x_f16 = (xa.dtype() == mlx::core::float16);
    const bool pos_is_i32 = (pa.dtype() == mlx::core::int32);
    if ((!x_f16 && xa.dtype() != mlx::core::float32) ||
        inva.dtype() != mlx::core::float32 ||
        (!pos_is_i32 && pa.dtype() != mlx::core::float32)) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int h = (xa.ndim() == 3) ? xa.shape(1) : 1;
    const int d = (xa.ndim() == 3) ? xa.shape(2) : xa.shape(1);
    const int half_rd = inva.shape(0);
    const int rd = half_rd * 2;
    if (m < 1 || h < 1 || d < 2 || half_rd < 1 || rd > d || pa.shape(0) != m) {
      return nullptr;
    }

    array xc = ensure_strided_aligned(xa, 1);
    array pc = ensure_contiguous_aligned(pa, 1);
    array invc = ensure_contiguous_aligned(inva, 1);

    const int head_w = d - rd;
    const int tg_x = (head_w == 0 && half_rd <= 32) ? 32 : ((d <= 128) ? 64 : 128);
    const int tg_y = (tg_x == 32 && h >= 4) ? 4 : 1;
    const int grid_y = ((h + tg_y - 1) / tg_y) * tg_y;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{tg_x, grid_y, m};
    const std::tuple<int, int, int> tg{tg_x, tg_y, 1};
    const Dtype out_dtype = out_f16 ? mlx::core::float16 : mlx::core::float32;
    const uint64_t tmpl_key = uint64_t(tg_x) |
                              (uint64_t(pos_is_i32 ? 1 : 0) << 8) |
                              (uint64_t(xa.ndim()) << 10) |
                              (uint64_t(x_f16 ? 1 : 0) << 12) |
                              (uint64_t(out_f16 ? 1 : 0) << 13);
    const std::vector<array> inputs{xc, pc, invc};
    const std::vector<Shape> out_shapes{xa.shape()};
    const std::vector<Dtype> out_dtypes{out_dtype};
    auto out = dispatch_cached_kernel(
        kKernelPartialRopeTail, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_partial_rope_tail_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"TG_X", tg_x}, {"OUT_T", out_dtype}}, std::nullopt, false,
              stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 partial_rope_tail: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)pos;
  (void)inv_freq;
  (void)out_f16;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_engram_norm_rows(mlx_array *rows, mlx_array *lookup_norm,
                                    float eps) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!rows || !lookup_norm || !std::isfinite(eps)) {
      return nullptr;
    }
    const auto &ra = *reinterpret_cast<array *>(rows);
    const auto &na = *reinterpret_cast<array *>(lookup_norm);
    if (ra.ndim() != 3 || na.ndim() != 2 || ra.dtype() != mlx::core::float32 ||
        na.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int m = ra.shape(0);
    const int cols = ra.shape(1);
    const int hd = ra.shape(2);
    if (m < 1 || cols < 1 || hd < 1 || na.shape(0) != cols ||
        na.shape(1) != hd) {
      return nullptr;
    }
    array rc = ensure_contiguous_aligned(ra, 1);
    array nc = ensure_contiguous_aligned(na, 1);
    array eps_arr = cached_f32_scalar(eps);

    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{128, cols, m};
    const std::tuple<int, int, int> tg{128, 1, 1};
    const std::vector<array> inputs{rc, nc, eps_arr};
    const std::vector<Shape> out_shapes{{m, cols * hd}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelEngramNormRows, stream, grid, tg, 0, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_engram_norm_rows_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg, {}, std::nullopt, false,
              stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 engram_norm_rows: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)rows;
  (void)lookup_norm;
  (void)eps;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_engram_gate_add(mlx_array *x, mlx_array *kv_raw,
                                   mlx_array *kv_b, mlx_array *gate_w,
                                   mlx_array *pre_gamma, mlx_array *hc_scale,
                                   mlx_array *keep_mask, float eps) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !kv_raw || !kv_b || !gate_w || !pre_gamma || !hc_scale ||
        !std::isfinite(eps)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &kva = *reinterpret_cast<array *>(kv_raw);
    const auto &kvba = *reinterpret_cast<array *>(kv_b);
    const auto &gwa = *reinterpret_cast<array *>(gate_w);
    const auto &pga = *reinterpret_cast<array *>(pre_gamma);
    const auto &hsa = *reinterpret_cast<array *>(hc_scale);
    if (xa.ndim() != 3 || kva.ndim() != 2 || kvba.ndim() != 2 ||
        gwa.ndim() != 2 || pga.ndim() != 1 || hsa.ndim() != 2) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || kva.dtype() != mlx::core::float32 ||
        kvba.dtype() != mlx::core::float32 ||
        gwa.dtype() != mlx::core::float32 ||
        pga.dtype() != mlx::core::float32 ||
        hsa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int hc = xa.shape(1);
    const int d = xa.shape(2);
    if (m < 1 || hc < 1 || hc > 16 || d < 1 || kva.shape(0) != m ||
        kva.shape(1) != (hc + 1) * d || kvba.shape() != Shape{hc + 1, d} ||
        gwa.shape() != Shape{hc, d} || pga.shape(0) != d ||
        hsa.shape() != Shape{hc, d}) {
      return nullptr;
    }
    const bool has_mask = (keep_mask != nullptr);
    array mask_c = cached_zero_u8();
    if (has_mask) {
      const auto &ma = *reinterpret_cast<array *>(keep_mask);
      if (ma.ndim() != 1 || ma.dtype() != mlx::core::uint8 || ma.shape(0) != m) {
        return nullptr;
      }
      mask_c = ensure_contiguous_aligned(ma, 1);
    }

    const bool vec4 = (d >= 8) && (d % 4 == 0);
    const bool use_reg = vec4 && (d <= 3072);
    const size_t align = vec4 ? 4 : 1;
    array xc = ensure_contiguous_aligned(xa, align);
    array kvc = ensure_contiguous_aligned(kva, align);
    array kvbc = ensure_contiguous_aligned(kvba, align);
    array gwc = ensure_contiguous_aligned(gwa, align);
    array pgc = ensure_contiguous_aligned(pga, align);
    array hsc = ensure_contiguous_aligned(hsa, align);
    array eps_arr = cached_f32_scalar(eps);

    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{256, m, 1};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(hc) | (uint64_t(vec4 ? 1 : 0) << 8) |
                              (uint64_t(use_reg ? 1 : 0) << 9) |
                              (uint64_t(has_mask ? 1 : 0) << 10);
    const std::vector<array> inputs{xc,  kvc, kvbc,   gwc,
                                    pgc, hsc, mask_c, eps_arr};
    const std::vector<Shape> out_shapes{{m, hc, d}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelEngramGateAdd, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_engram_gate_add_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"HC", hc},
               {"VEC4", vec4},
               {"USE_REG", use_reg},
               {"HAS_MASK", has_mask}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 engram_gate_add: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)kv_raw;
  (void)kv_b;
  (void)gate_w;
  (void)pre_gamma;
  (void)hc_scale;
  (void)keep_mask;
  (void)eps;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_f32_head_weights(mlx_array *x, mlx_array *w, float scale) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w || !std::isfinite(scale)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    if (xa.ndim() != 2 || wa.ndim() != 2 ||
        xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int k = xa.shape(1);
    const int h = wa.shape(0);
    if (m < 1 || m > 16 || k < 8 || (k % 4) != 0 || h < 1 ||
        wa.shape(1) != k || xa.size() < 8 || wa.size() < 8) {
      return nullptr;
    }
    array xc = ensure_contiguous_aligned(xa, 4);
    array wc = ensure_contiguous_aligned(wa, 4);
    array scale_arr = cached_f32_scalar(scale);
    const int m_tile = (m <= 4) ? m : ((m <= 8) ? 8 : 8);
    const int n_mtiles = (m + m_tile - 1) / m_tile;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{128, h, n_mtiles};
    const std::tuple<int, int, int> tg{128, 1, 1};
    const uint64_t tmpl_key = uint64_t(m_tile);
    const std::vector<array> inputs{xc, wc, scale_arr};
    const std::vector<Shape> out_shapes{{m, h}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelF32HeadWeights, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_f32_head_weights_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg, {{"M_TILE", m_tile}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 f32_head_weights: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  (void)scale;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_mtp_eh_norm_concat(mlx_array *e, mlx_array *h,
                                      mlx_array *enorm, mlx_array *hnorm,
                                      float eps) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!e || !h || !enorm || !hnorm || !std::isfinite(eps)) {
      return nullptr;
    }
    const auto &ea = *reinterpret_cast<array *>(e);
    const auto &ha = *reinterpret_cast<array *>(h);
    const auto &ena = *reinterpret_cast<array *>(enorm);
    const auto &hna = *reinterpret_cast<array *>(hnorm);
    auto ok_in_dt = [](Dtype dt) {
      return dt == mlx::core::float32 || dt == mlx::core::float16 ||
             dt == mlx::core::bfloat16;
    };
    if (ea.ndim() != 2 || ha.ndim() != 2 || ena.ndim() != 1 ||
        hna.ndim() != 1 || !ok_in_dt(ea.dtype()) || !ok_in_dt(ha.dtype()) ||
        ena.dtype() != mlx::core::float32 ||
        hna.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int m = ea.shape(0);
    const int d = ea.shape(1);
    if (m <= 0 || d <= 0 || ha.shape(0) != m || ha.shape(1) != d ||
        ena.shape(0) != d || hna.shape(0) != d) {
      return nullptr;
    }
    const bool vec4 = (d >= 8) && ((d % 4) == 0) && (ena.size() >= 8) &&
                      (hna.size() >= 8) && (size_t(m) * size_t(2 * d) >= 8);
    const bool use_reg = vec4 && (d <= 3072);
    const size_t align = vec4 ? 4 : 1;
    array ec = ensure_contiguous_aligned(ea, 1);
    array hc = ensure_contiguous_aligned(ha, 1);
    array enc = ensure_contiguous_aligned(ena, align);
    array hnc = ensure_contiguous_aligned(hna, align);
    array eps_arr = cached_f32_scalar(eps);

    auto dt_code = [](Dtype dt) -> uint64_t {
      return (dt == mlx::core::float32)
                 ? 0ULL
                 : ((dt == mlx::core::float16) ? 1ULL : 2ULL);
    };
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{256, m, 2};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(vec4 ? 1 : 0) |
                              (uint64_t(use_reg ? 1 : 0) << 1) |
                              (dt_code(ec.dtype()) << 4) |
                              (dt_code(hc.dtype()) << 12);
    const std::vector<array> inputs{ec, hc, enc, hnc, eps_arr};
    const std::vector<Shape> out_shapes{{m, 2 * d}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelMtpEhNormConcat, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_mtp_eh_norm_concat_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"VEC4", vec4}, {"USE_REG", use_reg}}, std::nullopt, false,
              stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 mtp_eh_norm_concat: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)e;
  (void)h;
  (void)enorm;
  (void)hnorm;
  (void)eps;
  return nullptr;
#endif
}

bool mlx_hy4_post_norm_add_rms_norm(mlx_array *res, mlx_array *y,
                                    mlx_array *post_w, mlx_array *next_w,
                                    float eps, bool need_x_out,
                                    mlx_array **out_x, mlx_array **out_h) {
  if (out_x)
    *out_x = nullptr;
  if (out_h)
    *out_h = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return false;
    }
    if (!res || !y || !post_w || !next_w || !out_h ||
        (need_x_out && !out_x) || !std::isfinite(eps)) {
      return false;
    }
    const auto &ra = *reinterpret_cast<array *>(res);
    const auto &ya = *reinterpret_cast<array *>(y);
    const auto &pwa = *reinterpret_cast<array *>(post_w);
    const auto &nwa = *reinterpret_cast<array *>(next_w);
    if (ra.ndim() != 2 || ya.ndim() != 2 || pwa.ndim() != 1 ||
        nwa.ndim() != 1 || ra.dtype() != mlx::core::float32 ||
        ya.dtype() != mlx::core::float32 ||
        pwa.dtype() != mlx::core::float32 ||
        nwa.dtype() != mlx::core::float32) {
      return false;
    }
    const int m = ra.shape(0);
    const int d = ra.shape(1);
    if (m <= 0 || d <= 0 || ya.shape(0) != m || ya.shape(1) != d ||
        pwa.shape(0) != d || nwa.shape(0) != d) {
      return false;
    }
    const bool vec4 = (d >= 8) && ((d % 4) == 0) && (ra.size() >= 8) &&
                      (ya.size() >= 8) && (pwa.size() >= 8) &&
                      (nwa.size() >= 8);
    const bool use_reg = vec4 && (d <= 3072);
    const size_t align = vec4 ? 4 : 1;
    array rc = ensure_contiguous_aligned(ra, align);
    array yc = ensure_contiguous_aligned(ya, align);
    array pwc = ensure_contiguous_aligned(pwa, align);
    array nwc = ensure_contiguous_aligned(nwa, align);
    array eps_arr = cached_f32_scalar(eps);

    const Shape x_shape = need_x_out ? Shape{m, d} : Shape{1};
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{256, m, 1};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(vec4 ? 1 : 0) |
                              (uint64_t(use_reg ? 1 : 0) << 1) |
                              (uint64_t(need_x_out ? 1 : 0) << 2);
    const std::vector<array> inputs{rc, yc, pwc, nwc, eps_arr};
    const std::vector<Shape> out_shapes{x_shape, {m, d}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32, mlx::core::float32};
    auto res_arrs = dispatch_cached_kernel(
        kKernelPostNormAddRmsNorm, stream, grid, tg, tmpl_key, inputs,
        out_shapes, out_dtypes, [&] {
          return hy4_post_norm_add_rms_norm_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"VEC4", vec4}, {"USE_REG", use_reg}, {"NEED_X_OUT", need_x_out}},
              std::nullopt, false, stream);
        });
    auto h_arr = std::make_unique<array>(std::move(res_arrs[1]));
    if (need_x_out) {
      auto x_arr = std::make_unique<array>(std::move(res_arrs[0]));
      *out_x = reinterpret_cast<mlx_array *>(x_arr.release());
    }
    *out_h = reinterpret_cast<mlx_array *>(h_arr.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 post_norm_add_rms_norm: " << e.what() << std::endl;
    return false;
  }
#else
  (void)res;
  (void)y;
  (void)post_w;
  (void)next_w;
  (void)eps;
  (void)need_x_out;
  return false;
#endif
}

} // extern "C"
