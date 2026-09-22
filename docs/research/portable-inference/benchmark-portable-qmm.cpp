// Manual portable QMM qualification; link against this checkout's mlx_ffi
// and MLX libraries. Arguments: mlx.metallib [N=17408] [K=5120].
// MLX_METAL_GPU_ARCH=applegpu_g15s selects SIMD kernels on an M5; it does
// not emulate an M3. MLX_METAL_NO_NAX is a compile flag, not an env override.
// Both arms reconstruct and evaluate fresh graphs; input/weight arrays are
// shared. Alternate arms within each repetition to limit ordering bias.
#include "mlx_common.h"
#include "mlx_portable_qmm.h"
#include <array>
#include <chrono>
#include <iomanip>
using namespace mlx::core;
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: benchmark-portable-qmm mlx.metallib [N] [K]\n";
    return 2;
  }
  metal::set_metallib_path(argv[1]);
  set_default_device(Device::gpu);
  const int n = argc > 2 ? std::atoi(argv[2]) : 17408;
  const int k = argc > 3 ? std::atoi(argv[3]) : 5120;
  if (n <= 0 || k <= 0 || n > 65536 || k > 65536 || k % 256 ||
      uint64_t(n) * uint64_t(k) > 256 * 1024 * 1024) {
    std::cerr
        << "Use positive dimensions, K divisible by 256, and N*K <= 256M\n";
    return 2;
  }
  uint32_t rng = 12345;
  auto next = [&] {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  for (const std::string mode : {"q5k", "iq4xs", "q6k", "q4k"}) {
    const int bits = mode == "q5k" ? 5 : mode == "q6k" ? 6 : 4;
    const int gs = mode == "q6k" ? 16 : 32;
    const bool minimum = mode == "q4k" || mode == "q5k";
    std::vector<uint32_t> ws(n * k * bits / 32);
    for (auto &v : ws)
      v = next();
    std::vector<uint8_t> ss(n * k / gs * (minimum ? 2 : 1));
    for (auto &v : ss)
      v = minimum ? next() % 48 + 1 : uint8_t(int(next() % 31) - 15);
    std::vector<float16_t> bs(n * k / 256 * (minimum ? 2 : 1));
    for (auto &v : bs)
      v = std::array<float, 6>{
          .03125f, .0625f, .125f, .19995f, .08331f, .25f}[next() % 6];
    array w(ws.data(), {n, k * bits / 32}, uint32);
    array s = minimum ? array(ss.data(), {n, k / gs * 2}, uint8)
                      : array(reinterpret_cast<int8_t *>(ss.data()),
                              {n, k / gs}, int8);
    array b(bs.data(), {n, k / 256 * (minimum ? 2 : 1)}, float16);
    eval(w, s, b);
    for (int m : {32, 128, 512, 2048}) {
      std::vector<bfloat16_t> xs(m * k);
      for (auto &v : xs)
        v = (float(int(next() >> 16) - 32768)) / 32768.f;
      array x(xs.data(), {m, k}, bfloat16);
      eval(x);
      setenv("MLX_PORTABLE_KQUANT", "1", 1);
      auto ref = quantized_matmul(x, w, s, b, true, gs, bits, mode);
      eval(ref);
      std::vector<double> samples[2];
      float error = 0;
      bool eligible = false;
      for (int rep = 0; rep < 11; ++rep) {
        for (int arm = 0; arm < 2; ++arm) {
          int which = (rep % 2) ? 1 - arm : arm;
          auto t = std::chrono::steady_clock::now();
          auto opt =
              which ? portable_kquant_matmul(x, w, s, b, true, gs, bits, mode)
                    : std::nullopt;
          eligible |= opt.has_value();
          auto y =
              opt ? *opt : quantized_matmul(x, w, s, b, true, gs, bits, mode);
          eval(y);
          if (rep >= 2)
            samples[which].push_back(std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - t)
                                         .count());
          if (rep == 0 && which) {
            auto diff = astype(max(abs(subtract(y, ref))), float32);
            eval(diff);
            error = diff.item<float>();
            if (!std::isfinite(error) || error != 0)
              throw std::runtime_error("SIMD parity failed");
          }
        }
      }
      for (auto &values : samples)
        std::sort(values.begin(), values.end());
      std::cout << mode << " M=" << m << " N=" << n << " K=" << k
                << " eligible=" << eligible << " baseline_ms=" << samples[0][4]
                << " portable_ms=" << samples[1][4]
                << " max_abs_vs_simd=" << error << std::endl;
    }
  }
}
