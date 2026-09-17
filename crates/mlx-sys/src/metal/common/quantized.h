#pragma once

#include <stdexcept>
#include <string>

#ifdef MLX_NODE_METAL_ENABLED
// Generated from the linked MLX sources by build.rs. These private preambles
// are available even when MLX itself uses a precompiled metallib.
namespace mlx::core::quantized_preamble {
const char *gemm();
const char *quantized_utils();
const char *kquant();
const char *nax();
const char *kquant_nax();

// Custom-kernel source comparisons run at every dispatch. Keep only the
// register helpers, without changing the vendored quantization arithmetic.
inline const std::string &qmv_header() {
  static const std::string header = [] {
    std::string source = kquant();
    const auto loader = source.find("struct QuantizedBlockLoader {");
    const auto end = loader == std::string::npos
                         ? std::string::npos
                         : source.rfind("\ntemplate <", loader);
    if (end == std::string::npos)
      throw std::runtime_error("Quantized register preamble boundary changed");
    source.resize(end);
    return source;
  }();
  return header;
}

inline const std::string &nax_header() {
  static const std::string header = [] {
    // Generic kernel bodies require MLX's private broadcasting helpers;
    // custom kernels only need the quantized decoders and block loaders.
    std::string source = kquant_nax();
    const auto function = source.find("METAL_FUNC void kquant_qmm_t_nax_tgp_impl(");
    const auto end = function == std::string::npos
                         ? std::string::npos
                         : source.rfind("\ntemplate <", function);
    if (end == std::string::npos)
      throw std::runtime_error("Quantized NAX preamble boundary changed");
    source.resize(end);
    return std::string(nax()) + source +
#include "packed_group32.metal.inc"
        ;
  }();
  return header;
}
} // namespace mlx::core::quantized_preamble
#endif
