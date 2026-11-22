#pragma once

// GGUF projections on Splash's MDGG0001 weight planes (crates/mlx-sys/src/metal/splash_gguf, prebuilt into
// paged_attn.metallib): the load-time repack, the plan law and the staged decode GEMMs at 8 / 16 / 24 / 32 rows.

#include <cstdint>
#include <string>
#include <vector>

namespace mlx::core::gguf_planes {

// Every kernel name the product primitives can request from paged_attn.metallib, all "gguf_"-prefixed.
std::vector<std::string> metal_kernel_names();
// The test-only kernels (decode checks, fp32 epilogue fault): gguf_planes_test.metallib only.
std::vector<std::string> test_kernel_names();

namespace testing {
// Test-only faults, off unless a test sets one (mlx_test_gguf_planes_mutation).
enum Mutation : int {
  kNone = 0,
  kPlaneShift = 1,       // plane0 bound one K group (64 rows x P0 bytes) later
  kMetaUnitOff = 2,      // meta bound one meta unit (MB bytes) earlier
  kCounterNotReset = 3,  // split counters handed to the kernel at splits - 1 (a reset that never ran), not 0
  kPeerLeak = 4,         // rows 8.. of a 16 / 24 / 32-row call also read the rows 8 above them
  kNameRemoved = 5,      // the metallib check misses gguf_decode_iq4xs_m8_r in the library
  kWalkOff = 6,          // every decode call runs with the walk off, whatever spread it was given
  kDecodeNan = 7,        // the dequant's first 8 weights of row 0 come out NaN
  kFp32Epilogue = 8,     // Q4_K 8-row residual and gate|up calls take Splash's fp32 epilogue kernels
  kTestNameInProduct = 9, // the metallib check sees gguf_dequant in paged_attn.metallib
};
inline thread_local int mutation = kNone;
// Test-only: a metallib the next primitives built on this thread take their kernels from (empty: the prebuilt one).
inline thread_local std::string library;
// Test-only: the path of gguf_planes_test.metallib (mlx_test_gguf_planes_test_library); empty until a test sets it.
std::string test_library();
} // namespace testing

} // namespace mlx::core::gguf_planes
