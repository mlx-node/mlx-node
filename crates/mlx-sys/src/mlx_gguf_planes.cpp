// GGUF projections on Splash's MDGG0001 weight planes. The kernels are Splash 1.6.1's
// (crates/mlx-sys/src/metal/splash_gguf, prebuilt into paged_attn.metallib; no JIT). The plan law is rewritten from
// Splash 1.6.1 d8af1054aab0f97de21d043e01aa98af3754f10e runtime/ops/KernelSelection.cpp (waves, stagedTime,
// waveSplits, bf16Projection) with the constants of runtime/ops/KernelCalibration.hpp (kApple10StagedLaw,
// kApple10SpreadFromCores, kApple10OnePassFromCores), and the encoding from runtime/ops/LinearGguf.cpp and
// runtime/model/GgufPreparation.cpp. Apache-2.0; see THIRD_PARTY_NOTICES.
//
// The plan is made for one lane (8 rows) and kept for every row count: rows 8k..8k+7 of a 16 / 24 / 32-row call
// are the bits of their own 8-row call.

#include "mlx_common.h"
#include "mlx_gguf_planes.h"
#include "mlx_test_counters.h"

#include "metal/abi/Gguf.h"
#include "metal/abi/GgufRepack.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/fast_primitives.h"
#include "mlx_kquant.h"
#include "mlx_paged_metallib.h"
#endif

extern "C" int mlx_gpu_reported_core_count();

namespace mlx::core::gguf_planes {

namespace {

constexpr uint32_t kRowTiers[] = {8, 16, 32};
constexpr const char* kEpilogues[] = {"a", "r", "g"};

struct WaveTier {
  uint32_t resident, saturating, following;
};
constexpr WaveTier kTile{10, 6, 2};
constexpr WaveTier kGateUpPair{16, 10, 3};
constexpr uint32_t kPartitionInputs = 512;
constexpr uint32_t kMaximumSplits = 8;
constexpr uint32_t kSpreadFromCores = 16;
constexpr uint32_t kOnePassFromCores = 16;
// Splash's DeviceCapabilities fallback when the IORegistry publishes no count.
constexpr uint32_t kFallbackCores = 32;
// The register tile (not ported) takes a grid of fewer than belowQuarters / 4 threadgroups per core; 6 is the
// largest boundary of kApple10DecodeRegister, so a tensor any format class would register is left out.
constexpr uint32_t kRegisterBelowQuarters = 6;

bool known_format(uint32_t fmt) { return fmt < GGUF_FMT_COUNT; }

uint32_t bits16(uint32_t fmt) {
  return kQuantFormats[fmt].block_bytes * 128 / kQuantFormats[fmt].block_elements;
}

struct PlaneBytes {
  uint64_t plane0, plane1, meta;
  uint64_t total() const { return plane0 + plane1 + meta; }
};

PlaneBytes plane_bytes(uint32_t fmt, uint64_t n, uint64_t k) {
  const QuantFormat& f = kQuantFormats[fmt];
  const uint64_t groups = k / 32;
  return {n * groups * f.plane0_bytes, n * groups * f.plane1_bytes, n * (groups / f.meta_groups) * f.meta_bytes};
}

uint64_t waves(uint64_t groups, uint32_t cores, uint32_t resident) {
  return ((groups + cores - 1) / cores + resident - 1) / resident;
}

uint64_t staged_time(uint64_t groups, uint32_t steps, uint32_t splits, uint32_t rows, uint32_t b16, uint32_t cores,
                     WaveTier tile) {
  const uint64_t count = waves(groups, cores, tile.resident), full = (count - 1) * tile.resident * cores,
                 floor = uint64_t{count > 1 ? tile.following : tile.saturating} * cores;
  const uint64_t step_bits = uint64_t{GGUF_TILE_COLUMNS} * 32 * b16 / 16,
                 partial_bits = splits > 1 ? 2 * uint64_t{rows} * GGUF_TILE_COLUMNS * 32 : 0;
  return steps * step_bits * (full + std::max(groups - full, floor)) + groups * partial_bits;
}

} // namespace

struct Plan {
  uint32_t staged;
  uint32_t splits;
  uint32_t one_pass;
  uint32_t spread;
  uint32_t sixteen;
};

namespace {
constexpr uint32_t kSixteenRowResident = 9;
constexpr uint32_t kSixteenBelowBits16 = 72;
constexpr uint32_t kSixteenWaves = 2;

// Splash's sixteenLane: a one-lane step it runs on the 16-row tile (reported; the port runs one lane at 8 rows).
bool sixteen_lane(uint64_t tiles, uint32_t b16, uint32_t cores) {
  if (!b16) return false;
  if (b16 < kSixteenBelowBits16) return true;
  const uint64_t w16 = waves(tiles, cores, kSixteenRowResident);
  return w16 <= kSixteenWaves && w16 == waves(tiles, cores, kTile.resident);
}
} // namespace

// Splash's staged plan of a decode step of one lane (8 rows): K splits and passes of least time by the wave law,
// ties keeping fewer partitions, then one pass; a gate/up pair (gate_b16, up_b16 nonzero; b16 their mean) of one
// format may run one pass from 16 cores; the walk spreads from 16 cores. A grid the register tile would take
// is left out of the plan (staged 0).
Plan plan(uint32_t n, uint32_t k, uint32_t b16, uint32_t cores, bool one_format, uint32_t gate_b16,
          uint32_t up_b16) {
  Plan out{0, 1, 0, 0, 0};
  const bool pair = gate_b16 && up_b16;
  if (!cores || !n || n % GGUF_TILE_COLUMNS || !k || k % 32 || !b16) return out;
  const bool one_pass = pair && one_format && cores >= kOnePassFromCores;
  if (4 * uint64_t{one_pass ? 2u : 1u} * (n / GGUF_TILE_COLUMNS) < uint64_t{kRegisterBelowQuarters} * cores)
    return out;
  constexpr uint32_t rows = 8;
  uint64_t least = UINT64_MAX;
  for (uint32_t splits = 1; splits <= kMaximumSplits && (k / 32) % splits == 0 &&
                            (splits == 1 || k / splits >= kPartitionInputs);
       splits *= 2) {
    const uint32_t steps = k / 32 / splits;
    const uint64_t tiles = uint64_t{n / GGUF_TILE_COLUMNS} * splits;
    if (one_pass) {
      const uint64_t time = staged_time(2 * tiles, steps, splits, rows, b16, cores, kGateUpPair);
      if (time < least) least = time, out.splits = splits, out.one_pass = 1;
    }
    const uint64_t time = (one_pass ? 2 : 1) * staged_time(tiles, steps, splits, rows, b16, cores, kTile);
    if (time < least) least = time, out.splits = splits, out.one_pass = 0;
  }
  out.staged = 1;
  out.spread = cores >= kSpreadFromCores;
  if (pair) {
    const Plan g = plan(n, k, gate_b16, cores, false, 0, 0), u = plan(n, k, up_b16, cores, false, 0, 0);
    out.sixteen = out.splits == 1 && g.sixteen && u.sixteen;
  } else {
    out.sixteen = out.splits == 1 && sixteen_lane(n / GGUF_TILE_COLUMNS, b16, cores);
  }
  return out;
}

std::vector<std::string> metal_kernel_names() {
  std::vector<std::string> names;
  for (const auto& f : kQuantFormats) {
    for (uint32_t rows : kRowTiers) {
      const std::string stem = std::string("gguf_decode_") + f.name + "_m" + std::to_string(rows);
      for (const char* ep : kEpilogues) names.push_back(stem + "_" + ep);
      names.push_back(stem + "_gate_up");
    }
  }
  for (uint32_t rows : kRowTiers) names.push_back("gguf_decode_fused_m" + std::to_string(rows));
  names.push_back("gguf_repack");
  return names;
}

std::vector<std::string> test_kernel_names() {
  return {"gguf_dequant", "gguf_dequant_epilogues", "gguf_decode_q4k_m8_r_fp32", "gguf_decode_q4k_m8_gate_up_fp32"};
}

namespace {
std::mutex test_library_mutex;
std::string test_library_path;
} // namespace

std::string testing::test_library() {
  std::lock_guard<std::mutex> lock(test_library_mutex);
  return test_library_path;
}

#ifdef MLX_NODE_METAL_ENABLED
namespace {

enum class Op : uint32_t { Decode, GateUp, Fused, Repack, Dequant };

struct Params {
  Op op;
  uint32_t epilogue;  // Decode: 0 plain, 1 residual, 2 gate; Dequant: 0 the fast-math build, 1 the epilogue build
  uint32_t rows;      // M of the call (8, 16, 24, 32)
  uint32_t k;
  uint32_t splits;
  uint32_t spread;
  uint32_t segments;
  uint32_t fmt[3];
  uint32_t n[3];
  uint32_t row0;       // Repack: first row of the band
  uint32_t band_rows;  // Repack: rows of the band
  uint32_t n_total;    // Repack: rows of the tensor
  uint64_t src_offset; // Repack: staging byte offset of the band
  int32_t mutation;
};

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument("gguf planes: " + what);
}

bool device_takes_planes() {
  if (!metal::is_available() || !metal::is_nax_available()) return false;
  return metal::device(Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyApple10);
}

uint32_t device_cores() {
  const int reported = mlx_gpu_reported_core_count();
  return reported > 0 ? uint32_t(reported) : kFallbackCores;
}

// The 8-row tiles of one call: one tile of 8, 16 or 32 rows; 24 rows as a 16-row and an 8-row tile.
std::vector<std::pair<uint32_t, uint32_t>> row_tiles(uint32_t rows) {
  if (rows == 24) return {{0, 16}, {16, 8}};
  return {{0, rows}};
}

MTL::ComputePipelineState* pipeline(metal::Device& d, const std::string& name, const std::string& hash = "",
                                    const metal::MTLFCList& consts = {}, const std::string& library = "") {
  if (!library.empty()) {
    auto* lib = d.get_library("gguf_planes_test:" + library, library);
    return d.get_kernel(name, lib, hash, consts);
  }
  return fast::paged::get_prebuilt_kernel(d, "gguf_planes", "GGUF planes", name, hash, consts);
}

MTL::ComputePipelineState* test_pipeline(metal::Device& d, const std::string& name) {
  const std::string path = testing::test_library();
  expect(!path.empty(), "test kernel " + name + " needs gguf_planes_test.metallib (mlx_test_gguf_planes_test_library)");
  return d.get_kernel(name, d.get_library("gguf_planes_test_kernels:" + path, path));
}

// The function constants of a fused pipeline: the segments' distinct formats, ascending, the last repeated to three.
std::array<uint32_t, 3> fused_constants(const uint32_t* fmt, uint32_t segments) {
  std::array<uint32_t, 3> f{};
  for (uint32_t i = 0; i < segments; ++i) f[i] = fmt[i];
  std::sort(f.begin(), f.begin() + segments);
  auto distinct = std::unique(f.begin(), f.begin() + segments);
  std::fill(distinct, f.end(), distinct[-1]);
  return f;
}

MTL::ComputePipelineState* fused_pipeline(metal::Device& d, uint32_t rows, const std::array<uint32_t, 3>& f,
                                          const std::string& library = "") {
  const std::string name = "gguf_decode_fused_m" + std::to_string(rows);
  const std::string hash =
      name + "_f" + std::to_string(f[0]) + "_" + std::to_string(f[1]) + "_" + std::to_string(f[2]);
  metal::MTLFCList consts = {
      {&f[0], MTL::DataType::DataTypeUInt, 0},
      {&f[1], MTL::DataType::DataTypeUInt, 1},
      {&f[2], MTL::DataType::DataTypeUInt, 2},
  };
  return pipeline(d, name, hash, consts, library);
}

void record(const std::string& family, uint32_t splits) {
  if (bridge_testing::counting) {
    bridge_testing::record(family);
    bridge_testing::record("gguf_planes_splits" + std::to_string(splits));
  }
}

class GgufPlanes final : public fast::Custom {
 public:
  GgufPlanes(Stream s, Params p, std::string library)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("gguf planes have no fallback");
        }),
        p_(p),
        library_(std::move(library)) {}
  DEFINE_NAME(GgufPlanes)
  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const GgufPlanes&>(other);
    return std::memcmp(&p_, &o.p_, sizeof(p_)) == 0 && library_ == o.library_;
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("gguf planes require Metal");
  }
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
                         const std::vector<array>&) override {
    throw std::runtime_error("gguf planes are inference-only");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    std::vector<array> in;
    in.reserve(inputs.size());
    for (const auto& a : inputs) {
      if (a.flags().row_contiguous && a.data_size() == a.size() && a.offset() % 16 == 0) {
        in.push_back(a);
      } else {
        array dense(a.shape(), a.dtype(), nullptr, {});
        dense.set_data(allocator::malloc(dense.nbytes()));
        copy_gpu(a, dense, CopyType::General, s);
        enc.add_temporary(dense);
        in.push_back(std::move(dense));
      }
    }
    switch (p_.op) {
      case Op::Decode: return decode(d, enc, in, outputs[0]);
      case Op::GateUp: return gate_up(d, enc, in, outputs[0]);
      case Op::Fused: return fused(d, enc, in, outputs[0]);
      case Op::Repack: return repack(d, enc, in, outputs[0]);
      case Op::Dequant: return dequant(d, enc, in, outputs[0]);
    }
  }

 private:
  Params p_;
  std::string library_;

  // plane0, plane1 and meta of rows [row0, row0 + ...) of an n-row tensor in `planes`, with the test faults.
  std::array<int64_t, 3> plane_offsets(uint32_t fmt, uint64_t n, uint64_t row0) const {
    const QuantFormat& f = kQuantFormats[fmt];
    const PlaneBytes b = plane_bytes(fmt, n, p_.k);
    const uint64_t groups = p_.k / 32;
    std::array<int64_t, 3> o{int64_t(row0 * groups * f.plane0_bytes),
                             int64_t(b.plane0 + row0 * groups * f.plane1_bytes),
                             int64_t(b.plane0 + b.plane1 + row0 * (groups / f.meta_groups) * f.meta_bytes)};
    if (p_.mutation == testing::kPlaneShift) o[0] += int64_t(QUANT_TILE_ROWS) * f.plane0_bytes;
    if (p_.mutation == testing::kMetaUnitOff) o[2] -= f.meta_bytes;
    return o;
  }

  // Split scratch of one dispatch: fp32 partials and the stream's arrival counters (both the output when unsplit).
  std::pair<array, array> split_scratch(metal::CommandEncoder& enc, array& out, uint64_t partial_floats,
                                        int counters) {
    if (p_.splits == 1) return {out, out};
    array partials({int(partial_floats)}, float32, nullptr, {});
    partials.set_data(allocator::malloc(partials.nbytes()));
    enc.add_temporary(partials);
    if (p_.mutation == testing::kCounterNotReset) {
      array fresh({counters}, uint32, nullptr, {});
      fresh.set_data(allocator::malloc(fresh.nbytes()));
      array one(uint32_t(p_.splits - 1), uint32);
      fill_gpu(one, fresh, stream());
      enc.add_temporary(one);
      enc.add_temporary(fresh);
      return {partials, fresh};
    }
    return {partials, kquant::tiled_projection_counters(stream(), counters)};
  }

  // The test kernels of the fp32 epilogue fault exist for Q4_K at 8 rows only.
  std::string fault_suffix(uint32_t fmt, uint32_t tile) const {
    if (p_.mutation != testing::kFp32Epilogue) return "";
    expect(fmt == GGUF_FMT_Q4K && tile == 8, "the fp32 epilogue fault has Q4_K 8-row kernels only");
    return "_fp32";
  }

  void decode(metal::Device& d, metal::CommandEncoder& enc, const std::vector<array>& in, array& out) {
    out.set_data(allocator::malloc(out.nbytes()));
    const uint32_t n = p_.n[0], k = p_.k, fmt = p_.fmt[0];
    const auto planes = plane_offsets(fmt, n, 0);
    const array& aux = p_.epilogue ? in[2] : in[0];
    for (auto [r0, tile] : row_tiles(p_.rows)) {
      const std::string fault = p_.epilogue == 1 ? fault_suffix(fmt, tile) : "";
      const std::string name = std::string("gguf_decode_") + kQuantFormats[fmt].name + "_m" + std::to_string(tile) +
                               "_" + kEpilogues[p_.epilogue] + fault;
      record(std::string("gguf_planes_") + kEpilogues[p_.epilogue] + "_m" + std::to_string(tile), p_.splits);
      auto [partials, counters] = split_scratch(enc, out, uint64_t(p_.splits) * tile * n, int(n / GGUF_TILE_COLUMNS));
      enc.set_compute_pipeline_state(fault.empty() ? pipeline(d, name, "", {}, library_) : test_pipeline(d, name));
      enc.set_input_array(in[0], 0, int64_t(r0) * k * 2);
      for (int i = 0; i < 3; ++i) enc.set_input_array(in[1], 1 + i, planes[i]);
      enc.set_output_array(out, 4, int64_t(r0) * n * 2);
      enc.set_output_array(partials, 5);
      enc.set_output_array(counters, 6);
      enc.set_input_array(aux, 7, p_.epilogue ? int64_t(r0) * n * 2 : 0);
      const GgufDecodeParams params{k, p_.splits, n, 0, p_.spread};
      enc.set_bytes(params, 8);
      enc.dispatch_threadgroups(MTL::Size(n / GGUF_TILE_COLUMNS, p_.splits, 1), MTL::Size(GGUF_STAGED_THREADS, 1, 1));
    }
  }

  // A merged gate|up tensor of 2n rows: gate rows [0, n), up rows [n, 2n).
  void gate_up(metal::Device& d, metal::CommandEncoder& enc, const std::vector<array>& in, array& out) {
    out.set_data(allocator::malloc(out.nbytes()));
    const uint32_t n = p_.n[0], k = p_.k, fmt = p_.fmt[0];
    const auto gate = plane_offsets(fmt, 2 * uint64_t(n), 0);
    const auto up = plane_offsets(fmt, 2 * uint64_t(n), n);
    for (auto [r0, tile] : row_tiles(p_.rows)) {
      const std::string fault = fault_suffix(fmt, tile);
      const std::string name = std::string("gguf_decode_") + kQuantFormats[fmt].name + "_m" + std::to_string(tile) +
                               "_gate_up" + fault;
      record("gguf_planes_gate_up_m" + std::to_string(tile), p_.splits);
      auto [partials, counters] =
          split_scratch(enc, out, uint64_t(p_.splits) * tile * 2 * n, int(n / GGUF_STAGED_COLUMNS));
      enc.set_compute_pipeline_state(fault.empty() ? pipeline(d, name, "", {}, library_) : test_pipeline(d, name));
      enc.set_input_array(in[0], 0, int64_t(r0) * k * 2);
      for (int i = 0; i < 3; ++i) enc.set_input_array(in[1], 1 + i, gate[i]);
      for (int i = 0; i < 3; ++i) enc.set_input_array(in[1], 4 + i, up[i]);
      enc.set_output_array(out, 7, int64_t(r0) * n * 2);
      enc.set_output_array(partials, 8);
      enc.set_output_array(counters, 9);
      const GgufDecodeParams params{k, p_.splits, n, 0, p_.spread};
      enc.set_bytes(params, 10);
      enc.dispatch_threadgroups(MTL::Size(n / GGUF_STAGED_COLUMNS, p_.splits, 1),
                                MTL::Size(GGUF_GATE_UP_THREADS, 1, 1));
    }
  }

  // Segments in tile order (most bits per weight first, as Splash's LinearGguf.cpp), each written at its own
  // column offset; the walk of a tile follows its index inside its segment, so the order moves no bit.
  void fused(metal::Device& d, metal::CommandEncoder& enc, const std::vector<array>& in, array& out) {
    out.set_data(allocator::malloc(out.nbytes()));
    const uint32_t k = p_.k, segments = p_.segments;
    uint32_t total = 0, column[3] = {0, 0, 0};
    for (uint32_t i = 0; i < segments; ++i) column[i] = total, total += p_.n[i];
    std::vector<uint32_t> order(segments);
    for (uint32_t i = 0; i < segments; ++i) order[i] = i;
    const auto bits_per_weight = [&](uint32_t i) {
      const QuantFormat& f = kQuantFormats[p_.fmt[i]];
      return (f.plane0_bytes + f.plane1_bytes) * 8.0 / 32.0 + f.meta_bytes * 8.0 / (32.0 * f.meta_groups);
    };
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return bits_per_weight(a) > bits_per_weight(b); });
    GgufDecodeFusedParams params{};
    params.input_size = k;
    params.splits = p_.splits;
    params.out_stride = total;
    params.spread = p_.spread;
    for (uint32_t j = 0; j < segments; ++j) {
      params.cols[j] = p_.n[order[j]];
      params.fmt[j] = p_.fmt[order[j]];
      params.offset[j] = column[order[j]];
    }
    const auto constants = fused_constants(params.fmt, segments);
    for (auto [r0, tile] : row_tiles(p_.rows)) {
      record("gguf_planes_fused_m" + std::to_string(tile), p_.splits);
      auto [partials, counters] =
          split_scratch(enc, out, uint64_t(p_.splits) * tile * total, int(total / GGUF_TILE_COLUMNS));
      enc.set_compute_pipeline_state(fused_pipeline(d, tile, constants, library_));
      enc.set_input_array(in[0], 0, int64_t(r0) * k * 2);
      for (uint32_t j = 0; j < 3; ++j) {
        const uint32_t seg = order[std::min(j, segments - 1)];
        const auto planes = plane_offsets(p_.fmt[seg], p_.n[seg], 0);
        for (int i = 0; i < 3; ++i) enc.set_input_array(in[1 + seg], 1 + 3 * j + i, planes[i]);
      }
      enc.set_output_array(out, 10, int64_t(r0) * total * 2);
      enc.set_output_array(partials, 11);
      enc.set_output_array(counters, 12);
      enc.set_bytes(params, 13);
      enc.dispatch_threadgroups(MTL::Size(total / GGUF_TILE_COLUMNS, p_.splits, 1),
                                MTL::Size(GGUF_STAGED_THREADS, 1, 1));
    }
  }

  // Rows [row0, row0 + band_rows) of an n_total-row tensor from the staged native rows into the tensor's planes:
  // the kernel writes a chunk's planes from its first tile on, so the band binds plane0 at its own rows and names
  // plane1 and meta at their distances from there.
  void repack(metal::Device& d, metal::CommandEncoder& enc, const std::vector<array>& in, array& out) {
    const auto& dst = in[1];
    out.copy_shared_buffer(dst);
    const uint32_t fmt = p_.fmt[0];
    const QuantFormat& f = kQuantFormats[fmt];
    const PlaneBytes b = plane_bytes(fmt, p_.n_total, p_.k);
    const uint64_t groups = p_.k / 32;
    const uint64_t at0 = uint64_t(p_.row0) * groups * f.plane0_bytes;
    const uint64_t at1 = b.plane0 + uint64_t(p_.row0) * groups * f.plane1_bytes;
    const uint64_t atm = b.plane0 + b.plane1 + uint64_t(p_.row0) * (groups / f.meta_groups) * f.meta_bytes;
    const GgufRepackParams params{p_.band_rows, p_.k, fmt, p_.k / f.block_elements * f.block_bytes,
                                  uint32_t(at1 - at0), uint32_t(atm - at0)};
    if (bridge_testing::counting) bridge_testing::record("gguf_planes_repack");
    enc.set_compute_pipeline_state(pipeline(d, "gguf_repack"));
    enc.set_input_array(in[0], 0, int64_t(p_.src_offset));
    enc.set_output_array(out, 1, int64_t(at0));
    enc.set_bytes(params, 2);
    enc.dispatch_threads(MTL::Size(uint64_t(p_.band_rows) * groups, 1, 1), MTL::Size(256, 1, 1));
  }

  void dequant(metal::Device& d, metal::CommandEncoder& enc, const std::vector<array>& in, array& out) {
    out.set_data(allocator::malloc(out.nbytes()));
    const uint32_t fmt = p_.fmt[0], n = p_.n[0];
    const auto planes = plane_offsets(fmt, n, 0);
    const GgufDequantParams params{n, p_.k, fmt};
    enc.set_compute_pipeline_state(test_pipeline(d, p_.epilogue ? "gguf_dequant_epilogues" : "gguf_dequant"));
    for (int i = 0; i < 3; ++i) enc.set_input_array(in[0], i, planes[i]);
    enc.set_output_array(out, 3);
    enc.set_bytes(params, 4);
    enc.dispatch_threads(MTL::Size(uint64_t(n) * (p_.k / 32), 1, 1), MTL::Size(256, 1, 1));
  }
};

void expect_planes(const array& planes, uint32_t fmt, uint64_t n, uint64_t k, const char* what) {
  expect(planes.dtype() == uint8 && planes.ndim() == 1 &&
             uint64_t(planes.size()) == plane_bytes(fmt, n, k).total(),
         std::string(what) + " must be uint8 [" + std::to_string(plane_bytes(fmt, n, k).total()) + "]");
}

// x [M, K] bf16 of M = 8 x lanes (1..4 lanes) and the plan's split of K.
void expect_rows(const array& x, uint32_t k, uint32_t splits, uint32_t spread) {
  expect(x.dtype() == bfloat16 && x.ndim() == 2 && x.shape(1) == int(k), "x must be bf16 [M, K]");
  const int m = x.shape(0);
  expect(m == 8 || m == 16 || m == 24 || m == 32, "x must have 8, 16, 24 or 32 rows");
  expect(splits >= 1 && splits <= kMaximumSplits && (splits & (splits - 1)) == 0 && (k / 32) % splits == 0,
         "splits must be a power of two up to 8 that divides K / 32");
  expect(spread <= 1, "spread is 0 or 1");
}

void expect_shape(uint32_t fmt, uint32_t n, uint32_t k) {
  expect(known_format(fmt), "unknown format " + std::to_string(fmt));
  expect(n > 0 && n % QUANT_TILE_ROWS == 0, "N must be a positive multiple of 64");
  expect(k >= 256 && k % 256 == 0, "K must be a positive multiple of 256");
}

array peer_leak(const array& x) {
  const int m = x.shape(0), k = x.shape(1);
  if (testing::mutation != testing::kPeerLeak || m == 8) return x;
  auto above = slice(x, {0, 0}, {m - 8, k});
  return add(x, concatenate({zeros({8, k}, bfloat16), above}, 0));
}

Params base(Op op, uint32_t rows, uint32_t k, uint32_t splits, uint32_t spread) {
  Params p;
  std::memset(&p, 0, sizeof(p));
  p.op = op;
  p.rows = rows;
  p.k = k;
  p.splits = splits;
  p.spread = testing::mutation == testing::kWalkOff ? 0 : spread;
  p.segments = 1;
  p.mutation = testing::mutation;
  return p;
}

array make(Params p, Shape shape, Dtype dtype, std::vector<array> inputs) {
  expect(device_takes_planes(), "need an Apple10 GPU with tensor ops");
  return array(std::move(shape), dtype,
               std::make_shared<GgufPlanes>(to_stream(Device::gpu), p, testing::library), std::move(inputs));
}

} // namespace

array matmul(const array& x, const array& planes, const array* aux, uint32_t fmt, uint32_t n, uint32_t k,
             uint32_t epilogue, uint32_t splits, uint32_t spread) {
  expect_shape(fmt, n, k);
  expect_rows(x, k, splits, spread);
  expect_planes(planes, fmt, n, k, "planes");
  expect(epilogue <= 2, "epilogue is 0 (plain), 1 (residual) or 2 (gate)");
  const int m = x.shape(0);
  std::vector<array> inputs{peer_leak(x), planes};
  if (epilogue) {
    expect(aux && aux->dtype() == bfloat16 && aux->shape() == Shape{m, int(n)}, "aux must be bf16 [M, N]");
    inputs.push_back(*aux);
  }
  Params p = base(Op::Decode, m, k, splits, spread);
  p.epilogue = epilogue;
  p.fmt[0] = fmt;
  p.n[0] = n;
  return make(p, {m, int(n)}, bfloat16, std::move(inputs));
}

array gate_up(const array& x, const array& planes, uint32_t fmt, uint32_t n, uint32_t k, uint32_t splits,
              uint32_t spread) {
  expect_shape(fmt, n, k);
  expect_rows(x, k, splits, spread);
  expect_planes(planes, fmt, 2 * uint64_t(n), k, "merged gate|up planes");
  Params p = base(Op::GateUp, x.shape(0), k, splits, spread);
  p.fmt[0] = fmt;
  p.n[0] = n;
  return make(p, {x.shape(0), int(n)}, bfloat16, {peer_leak(x), planes});
}

array fused(const array& x, const std::vector<array>& planes, const uint32_t* fmt, const uint32_t* n, uint32_t k,
            uint32_t splits, uint32_t spread) {
  const uint32_t segments = uint32_t(planes.size());
  expect(segments >= 1 && segments <= 3, "a fused projection has 1 to 3 segments");
  expect_rows(x, k, splits, spread);
  Params p = base(Op::Fused, x.shape(0), k, splits, spread);
  p.segments = segments;
  std::vector<array> inputs{peer_leak(x)};
  uint32_t total = 0;
  for (uint32_t i = 0; i < segments; ++i) {
    expect_shape(fmt[i], n[i], k);
    expect_planes(planes[i], fmt[i], n[i], k, "segment planes");
    p.fmt[i] = fmt[i];
    p.n[i] = n[i];
    total += n[i];
    inputs.push_back(planes[i]);
  }
  return make(p, {x.shape(0), int(total)}, bfloat16, std::move(inputs));
}

array repack_band(const array& staging, uint64_t src_offset, uint32_t fmt, uint32_t k, uint32_t n_total,
                  uint32_t row0, uint32_t rows, const array& dst) {
  expect_shape(fmt, n_total, k);
  expect(rows > 0 && row0 % QUANT_TILE_ROWS == 0 && rows % QUANT_TILE_ROWS == 0 && uint64_t(row0) + rows <= n_total,
         "a band is whole 64-row tiles inside the tensor");
  expect_planes(dst, fmt, n_total, k, "destination planes");
  expect(staging.dtype() == uint8 && staging.flags().row_contiguous && staging.offset() % 16 == 0,
         "staging must be a dense, 16-byte aligned uint8 array");
  const QuantFormat& f = kQuantFormats[fmt];
  const uint64_t row_bytes = uint64_t(k) / f.block_elements * f.block_bytes;
  expect(src_offset % 4 == 0, "the band's staging offset must be a multiple of 4");
  expect(src_offset + uint64_t(rows) * row_bytes <= uint64_t(staging.nbytes()), "the band runs past the staging");
  expect(plane_bytes(fmt, n_total, k).total() < (uint64_t(1) << 32), "planes of 4 GiB or more");
  expect(uint64_t(rows) * (k / 32) <= std::numeric_limits<uint32_t>::max(), "band too large for one dispatch");
  Params p = base(Op::Repack, 0, k, 1, 0);
  p.fmt[0] = fmt;
  p.n_total = n_total;
  p.row0 = row0;
  p.band_rows = rows;
  p.src_offset = src_offset;
  return make(p, dst.shape(), uint8, {staging, dst});
}

array dequant(const array& planes, uint32_t fmt, uint32_t n, uint32_t k, uint32_t build) {
  expect_shape(fmt, n, k);
  expect_planes(planes, fmt, n, k, "planes");
  expect(build <= 1, "build is 0 (fast math) or 1 (the epilogue build)");
  Params p = base(Op::Dequant, 0, k, 1, 0);
  p.epilogue = build;
  p.fmt[0] = fmt;
  p.n[0] = n;
  array out = make(p, {int(n), int(k)}, float16, {planes});
  if (testing::mutation != testing::kDecodeNan) return out;
  const array nan = full({1, 8}, std::numeric_limits<float>::quiet_NaN(), float16);
  const array row0 = concatenate({nan, slice(out, {0, 8}, {1, int(k)})}, 1);
  return concatenate({row0, slice(out, {1, 0}, {int(n), int(k)})}, 0);
}

// Builds the pipelines a planned projection needs at every row tier; false when the OS refuses one.
bool prepare(uint32_t kind, const uint32_t* fmt, uint32_t count) {
  try {
    if (!device_takes_planes() || count < 1 || count > 3) return false;
    auto& d = metal::device(Device::gpu);
    for (uint32_t i = 0; i < count; ++i)
      if (!known_format(fmt[i])) return false;
    for (uint32_t rows : kRowTiers) {
      const std::string stem = std::string("gguf_decode_") + kQuantFormats[fmt[0]].name + "_m" + std::to_string(rows);
      if (kind <= 2) pipeline(d, stem + "_" + kEpilogues[kind]);
      else if (kind == 3) pipeline(d, stem + "_gate_up");
      else if (kind == 4) fused_pipeline(d, rows, fused_constants(fmt, count));
      else return false;
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gguf planes: pipeline refused: %s\n", e.what());
    return false;
  }
}
#endif

} // namespace mlx::core::gguf_planes

using namespace mlx::core;

extern "C" {

struct GgufPlanesPlan {
  uint32_t staged;
  uint32_t splits;
  uint32_t one_pass;
  uint32_t spread;
  uint32_t sixteen;
};

// The kernel names, '\n'-joined; returns the length without the NUL (written only when `cap` exceeds it).
size_t mlx_gguf_planes_kernel_names(char* out, size_t cap) {
  std::string text;
  for (const auto& name : gguf_planes::metal_kernel_names()) text += name + "\n";
  if (out && cap > text.size()) std::memcpy(out, text.c_str(), text.size() + 1);
  return text.size();
}

// GGUF_FMT_* of a GGUF tensor type; GGUF_FMT_COUNT when the planes do not store it.
uint32_t mlx_gguf_planes_format_of(uint32_t ggml_type) { return gguf_format_of(ggml_type); }

// Sixteenths of a bit per weight of a format (Splash's weightBits16); 0 for an unknown one.
uint32_t mlx_gguf_planes_bits16(uint32_t fmt) { return fmt < GGUF_FMT_COUNT ? gguf_planes::bits16(fmt) : 0; }

// Bytes of the planes of an [n, k] tensor: plane0, plane1, meta into `parts` (may be null); the total, 0 when the
// format or the shape is not one the planes take.
uint64_t mlx_gguf_planes_bytes(uint32_t fmt, uint64_t n, uint64_t k, uint64_t* parts) {
  if (fmt >= GGUF_FMT_COUNT || !n || n % QUANT_TILE_ROWS || !k || k % 256) return 0;
  const auto b = gguf_planes::plane_bytes(fmt, n, k);
  if (parts) parts[0] = b.plane0, parts[1] = b.plane1, parts[2] = b.meta;
  return b.total();
}

// Sixteenths of a bit per weight of a fused projection's segments of `fmts` and `ns` rows (Splash's
// 128 * bytes / weights at one K); 0 for an unknown format.
uint32_t mlx_gguf_planes_fused_bits16(const uint32_t* fmts, const uint32_t* ns, uint32_t count) {
  if (!fmts || !ns) return 0;
  uint64_t weighted = 0, rows = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (fmts[i] >= GGUF_FMT_COUNT) return 0;
    weighted += uint64_t(ns[i]) * gguf_planes::bits16(fmts[i]);
    rows += ns[i];
  }
  return rows ? uint32_t(weighted / rows) : 0;
}

// The one-lane plan of an [n, k] projection of `bits16` sixteenths of a bit per weight on `cores` GPU cores. A
// gate/up pair names its tensors' bits (gate_bits16, up_bits16 nonzero; bits16 their mean) and whether they share
// one format. cores = 0 takes this device's: no plan (staged 0) off an Apple10 GPU with tensor ops.
bool mlx_gguf_planes_plan(uint32_t n, uint32_t k, uint32_t bits16, uint32_t cores, bool one_format,
                          uint32_t gate_bits16, uint32_t up_bits16, GgufPlanesPlan* out) {
  if (!out) return false;
  *out = {0, 1, 0, 0, 0};
#ifdef MLX_NODE_METAL_ENABLED
  if (!cores) {
    try {
      if (!gguf_planes::device_takes_planes()) return true;
    } catch (...) {
      return true;
    }
    cores = gguf_planes::device_cores();
  }
#else
  if (!cores) return true;
#endif
  const auto p = gguf_planes::plan(n, k, bits16, cores, one_format, gate_bits16, up_bits16);
  *out = {p.staged, p.splits, p.one_pass, p.spread, p.sixteen};
  return true;
}

// Whether this device runs the planes kernels (Metal, Apple10, tensor ops).
bool mlx_gguf_planes_available() {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    return gguf_planes::device_takes_planes();
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

// Builds every row tier's pipeline of one planned projection: kind 0 plain, 1 residual, 2 gate (fmts[0]),
// 3 gate|up pair (fmts[0]), 4 fused (fmts[0..count)). False when one is refused: the plan leaves the tensor out.
bool mlx_gguf_planes_prepare(uint32_t kind, const uint32_t* fmts, uint32_t count) {
#ifdef MLX_NODE_METAL_ENABLED
  return fmts && gguf_planes::prepare(kind, fmts, count);
#else
  (void)kind; (void)fmts; (void)count;
  return false;
#endif
}

static void report(const char* what, const std::exception& e) { std::fprintf(stderr, "%s: %s\n", what, e.what()); }

// Rows [row0, row0 + rows) of an n_total x k tensor of `fmt` from `staging` (uint8, the band's native rows from
// byte src_offset) into `dst` (the tensor's planes, uint8). `*out` receives a lazy array sharing dst's buffer, to
// pass as dst of the next band. 0, or nonzero with a message in `err`.
int32_t mlx_gguf_planes_repack_band(mlx_array* staging, uint64_t src_offset, uint32_t fmt, uint32_t k,
                                    uint32_t n_total, uint32_t row0, uint32_t rows, mlx_array* dst, mlx_array** out,
                                    char* err, size_t err_cap) {
  const auto fail = [&](const std::string& msg) {
    if (err && err_cap) {
      const size_t len = std::min(msg.size(), err_cap - 1);
      std::memcpy(err, msg.data(), len);
      err[len] = '\0';
    }
    return 1;
  };
  if (out) *out = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!staging || !dst || !out) return fail("mlx_gguf_planes_repack_band: null argument");
    auto result = gguf_planes::repack_band(*reinterpret_cast<array*>(staging), src_offset, fmt, k, n_total, row0,
                                           rows, *reinterpret_cast<array*>(dst));
    *out = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    return 0;
  } catch (const std::exception& e) {
    return fail(std::string("mlx_gguf_planes_repack_band: ") + e.what());
  }
#else
  (void)staging; (void)src_offset; (void)fmt; (void)k; (void)n_total; (void)row0; (void)rows; (void)dst;
  return fail("mlx_gguf_planes_repack_band: no Metal");
#endif
}

// x [M, k] bf16 (M = 8, 16, 24, 32) times the planes of an [n, k] tensor of `fmt`: epilogue 0 plain, 1 plus the
// residual aux [M, n], 2 the swiglu of gate aux [M, n] and this up projection; `splits` and `spread` from the
// one-lane plan. Null on invalid operands.
mlx_array* mlx_gguf_planes_matmul(mlx_array* x, mlx_array* planes, mlx_array* aux, uint32_t fmt, uint32_t n,
                                  uint32_t k, uint32_t epilogue, uint32_t splits, uint32_t spread) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!x || !planes) return nullptr;
    const array* a = aux ? reinterpret_cast<array*>(aux) : nullptr;
    return reinterpret_cast<mlx_array*>(new array(gguf_planes::matmul(*reinterpret_cast<array*>(x),
                                                                        *reinterpret_cast<array*>(planes), a, fmt,
                                                                        n, k, epilogue, splits, spread)));
  } catch (const std::exception& e) {
    report("mlx_gguf_planes_matmul", e);
  }
#else
  (void)x; (void)planes; (void)aux; (void)fmt; (void)n; (void)k; (void)epilogue; (void)splits; (void)spread;
#endif
  return nullptr;
}

// The swiglu of a merged gate|up tensor's halves (planes of 2n rows, gate first) in one pass: [M, n].
mlx_array* mlx_gguf_planes_gate_up(mlx_array* x, mlx_array* planes, uint32_t fmt, uint32_t n, uint32_t k,
                                   uint32_t splits, uint32_t spread) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!x || !planes) return nullptr;
    return reinterpret_cast<mlx_array*>(new array(gguf_planes::gate_up(
        *reinterpret_cast<array*>(x), *reinterpret_cast<array*>(planes), fmt, n, k, splits, spread)));
  } catch (const std::exception& e) {
    report("mlx_gguf_planes_gate_up", e);
  }
#else
  (void)x; (void)planes; (void)fmt; (void)n; (void)k; (void)splits; (void)spread;
#endif
  return nullptr;
}

// 1 to 3 segments of any formats in one dispatch, written side by side: [M, sum n].
mlx_array* mlx_gguf_planes_fused(mlx_array* x, mlx_array* const* planes, const uint32_t* fmts, const uint32_t* ns,
                                 uint32_t segments, uint32_t k, uint32_t splits, uint32_t spread) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!x || !planes || !fmts || !ns || segments < 1 || segments > 3) return nullptr;
    std::vector<array> p;
    for (uint32_t i = 0; i < segments; ++i) {
      if (!planes[i]) return nullptr;
      p.push_back(*reinterpret_cast<array*>(planes[i]));
    }
    return reinterpret_cast<mlx_array*>(
        new array(gguf_planes::fused(*reinterpret_cast<array*>(x), p, fmts, ns, k, splits, spread)));
  } catch (const std::exception& e) {
    report("mlx_gguf_planes_fused", e);
  }
#else
  (void)x; (void)planes; (void)fmts; (void)ns; (void)segments; (void)k; (void)splits; (void)spread;
#endif
  return nullptr;
}

// TEST-ONLY: the planes decoded to half [n, k] by the GEMM tiles' own decode, compiled in the fast-math build
// (build 0) or in the -fno-fast-math epilogue build (build 1).
mlx_array* mlx_test_gguf_planes_dequant(mlx_array* planes, uint32_t fmt, uint32_t n, uint32_t k, uint32_t build) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!planes) return nullptr;
    return reinterpret_cast<mlx_array*>(
        new array(gguf_planes::dequant(*reinterpret_cast<array*>(planes), fmt, n, k, build)));
  } catch (const std::exception& e) {
    report("mlx_test_gguf_planes_dequant", e);
  }
#else
  (void)planes; (void)fmt; (void)n; (void)k; (void)build;
#endif
  return nullptr;
}

// TEST-ONLY: the GPU seconds of each of `n` arrays evaluated alone, in order, each in one command buffer (the
// caller evaluates their weights and inputs first). False without Metal, on error, or when MLX may split a unit across command buffers
// (MLX_MAX_OPS_PER_BUFFER and MLX_MAX_MB_PER_BUFFER below 1000000).
bool mlx_test_gguf_planes_gpu_seconds(mlx_array* const* outs, size_t n, double* seconds) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!outs || !seconds) return false;
    auto s = default_stream(Device::gpu);
    auto [ops, mb] = metal::device(Device::gpu).get_max_ops_mb_per_buffer();
    if (ops < 1000000 || mb < 1000000) {
      std::fprintf(stderr, "mlx_test_gguf_planes_gpu_seconds: buffer limits %d ops / %d MB split units\n", ops, mb);
      return false;
    }
    for (size_t i = 0; i < n; ++i) {
      auto& out = *reinterpret_cast<array*>(outs[i]);
      synchronize(s);
      auto buffer = NS::RetainPtr(metal::get_command_encoder(s).get_command_buffer());
      eval(out);
      synchronize(s);
      seconds[i] = buffer->GPUEndTime() - buffer->GPUStartTime();
    }
    return true;
  } catch (const std::exception& e) {
    report("mlx_test_gguf_planes_gpu_seconds", e);
  }
#else
  (void)outs; (void)n; (void)seconds;
#endif
  return false;
}

// TEST-ONLY: this thread's fault (gguf_planes::testing::Mutation), 0 = none.
void mlx_test_gguf_planes_mutation(int32_t mutation) { gguf_planes::testing::mutation = mutation; }

// TEST-ONLY: the path of gguf_planes_test.metallib, which holds the test-only kernels, for every thread.
void mlx_test_gguf_planes_test_library(const char* path) {
  std::lock_guard<std::mutex> lock(gguf_planes::test_library_mutex);
  gguf_planes::test_library_path = path ? path : "";
}

// TEST-ONLY: checks gguf_planes_test.metallib against gguf_planes::test_kernel_names(): `report` receives one line
// per missing name and per other function; `*count` the library's functions. False without Metal, without the
// path, on error, or when `report` cannot hold the text.
bool mlx_test_gguf_planes_test_metallib_check(int64_t* count, char* report, size_t len) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    const std::string path = gguf_planes::testing::test_library();
    if (!count || !report || !len || path.empty() || !metal::is_available()) return false;
    auto& d = metal::device(Device::gpu);
    auto* lib = d.get_library("gguf_planes_test_kernels:" + path, path);
    std::vector<std::string> in_library;
    NS::Array* functions = lib->functionNames();
    for (NS::UInteger i = 0; i < functions->count(); ++i)
      in_library.push_back(functions->object<NS::String>(i)->utf8String());
    const auto known = gguf_planes::test_kernel_names();
    std::string text;
    for (const auto& name : known)
      if (std::find(in_library.begin(), in_library.end(), name) == in_library.end()) text += "missing " + name + "\n";
    for (const auto& name : in_library)
      if (std::find(known.begin(), known.end(), name) == known.end()) text += "unexpected " + name + "\n";
    *count = int64_t(in_library.size());
    if (text.size() + 1 > len) return false;
    std::memcpy(report, text.c_str(), text.size() + 1);
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_test_gguf_planes_test_metallib_check: %s\n", e.what());
  }
#else
  (void)count; (void)report; (void)len;
#endif
  return false;
}

} // extern "C"
