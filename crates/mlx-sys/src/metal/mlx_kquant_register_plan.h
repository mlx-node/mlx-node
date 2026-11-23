#pragma once
#include <cstdint>
namespace mlx::core::kquant {
// Splash register tiers, adapted from64 to native32 output columns. Native
// SG8 prefetches whole256-input superblocks, so every split retains that unit.
inline int sg8_register_splits(int n, int k, int cores) {
  if (n <= 0 || n % 32 || k <= 0 || k % 256 || cores <= 0) return 1;
  const int64_t grid = n / 32;
  int splits = 1;
  auto asks = [&](int groups, int inputs, int maximum) {
    return splits < maximum && grid * splits < int64_t(groups) * cores && k / (2 * splits) >= inputs;
  };
  while ((k / 256) % (2 * splits) == 0 &&
         (asks(4,256,8) || asks(32,1024,8) || asks(3,512,16))) splits *= 2;
  return splits;
}
} // namespace mlx::core::kquant
