// Copyright © 2023-2024 Apple Inc.

// ggml K-quant (Q6_K / Q4_K / Q5_K) kernels.
//
// Every K-quant sub-block is algebraically affine -- value = scale * q + bias
// -- so this is quantized.h with a single substitution: the per-group scalar
// load `s = scales[g]; b = biases[g]` becomes the two-level decode in KQScales.
// The importer repacks ggml's codes into the same LSB-first n-bit stream the
// affine kernels read, so get_pack_factor, get_bytes_per_pack, load_vector,
// qdot, qdot_safe, qouter and dequantize below are copies of the affine ones:
// they take (scale, bias) by value and never see a K-quant.
#include <metal_simdgroup>
#include <metal_stdlib>

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

MLX_MTL_CONST int8_t kIQ4NLValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,    25,  38,  53,  69,  89,  113};

template <typename U, bool nonlinear>
inline U kq_value(uint code, U scale, U bias) {
  if constexpr (nonlinear) {
    return scale * static_cast<U>(kIQ4NLValues[code]);
  } else {
    return scale * static_cast<U>(code) + bias;
  }
}

template <typename U, int values_per_thread, int bits, bool nonlinear>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if constexpr (nonlinear) {
    for (int i = 0; i < values_per_thread; i++) {
      const uint code = (w[i / 2] >> (4 * (i & 1))) & 0x0f;
      // load_vector's affine 4-bit path pre-divides successive activation
      // lanes by 16^lane so it can multiply packed, shifted nibbles directly.
      // IQ4_NL extracts each nibble before applying its nonlinear codebook, so
      // restore the original activation here.
      const U x = x_thread[i] * static_cast<U>(1 << (4 * (i & 3)));
      accum += x * static_cast<U>(kIQ4NLValues[code]);
    }
    return scale * accum;
  }

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits, bool nonlinear>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if constexpr (nonlinear) {
    for (int i = 0; i < N; i++) {
      const uint code = (w[i / 2] >> (4 * (i & 1))) & 0x0f;
      const U x = x_thread[i] * static_cast<U>(1 << (4 * (i & 3)));
      accum += x * static_cast<U>(kIQ4NLValues[code]);
    }
    return scale * accum;
  }

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits, bool nonlinear>
inline void
qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if constexpr (nonlinear) {
    for (int i = 0; i < values_per_thread; i++) {
      const uint code = (w[i / 2] >> (4 * (i & 1))) & 0x0f;
      result[i] += x * scale * static_cast<U>(kIQ4NLValues[code]);
    }
    return;
  }

  if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[5 * i];
      uint8_t w1 = w[5 * i + 1];
      uint8_t w2 = w[5 * i + 2];
      uint8_t w3 = w[5 * i + 3];
      uint8_t w4 = w[5 * i + 4];
      result[8 * i] += x * ((w0 & 0x1f) * scale + bias);
      result[8 * i + 1] +=
          x * ((((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * scale + bias);
      result[8 * i + 2] += x * (((w1 & 0x7c) >> 2) * scale + bias);
      result[8 * i + 3] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * scale + bias);
      result[8 * i + 4] +=
          x * ((((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * scale + bias);
      result[8 * i + 5] += x * (((w3 & 0x3e) >> 1) * scale + bias);
      result[8 * i + 6] +=
          x * ((((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * scale + bias);
      result[8 * i + 7] += x * (((w4 & 0xf8) >> 3) * scale + bias);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

// Decode one quantized block (scale * q + bias) into w_local. W (the output
// pointer type) serves the threadgroup block loader or a thread-local decode.
template <typename U, int N, int bits, bool nonlinear, typename W>
inline void dequantize(const device uint8_t* w, U scale, U bias, W w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if constexpr (nonlinear) {
    for (int i = 0; i < N; i++) {
      const uint code = (w[i / 2] >> (4 * (i & 1))) & 0x0f;
      w_local[i] = scale * static_cast<U>(kIQ4NLValues[code]);
    }
    return;
  }

  if (bits == 2) {
    U s[4] = {
        scale,
        scale / static_cast<U>(4.0f),
        scale / static_cast<U>(16.0f),
        scale / static_cast<U>(64.0f)};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = s[0] * (w[i] & 0x03) + bias;
      w_local[4 * i + 1] = s[1] * (w[i] & 0x0c) + bias;
      w_local[4 * i + 2] = s[2] * (w[i] & 0x30) + bias;
      w_local[4 * i + 3] = s[3] * (w[i] & 0xc0) + bias;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = (w[0] & 0x7) * scale + bias;
      w_local[1] = ((w[0] & 0x38) >> 3) * scale + bias;
      w_local[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
      w_local[3] = ((w[1] & 0xe) >> 1) * scale + bias;
      w_local[4] = ((w[1] & 0x70) >> 4) * scale + bias;
      w_local[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
      w_local[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
      w_local[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / static_cast<U>(16.0f)};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s[0] * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = (w[0] & 0x1f) * scale + bias;
      w_local[1] = (((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * scale + bias;
      w_local[2] = ((w[1] & 0x7c) >> 2) * scale + bias;
      w_local[3] = (((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * scale + bias;
      w_local[4] = (((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * scale + bias;
      w_local[5] = ((w[3] & 0x3e) >> 1) * scale + bias;
      w_local[6] = (((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * scale + bias;
      w_local[7] = ((w[4] & 0xf8) >> 3) * scale + bias;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = (w[0] & 0x3f) * scale + bias;
      w_local[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
      w_local[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
      w_local[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

// Decodes a block into the threadgroup tile. The affine loader hands
// dequantize() a T-typed scale and bias, but a K-quant scale is a product of an
// fp16 super-scale and an integer sub-scale, so it is worth more than T's
// mantissa; decode in float and round once on the store. That also sidesteps
// bfloat, which has no implicit conversion from float.
template <typename T, int N, int bits, bool nonlinear>
inline void dequantize_to(
    const device uint8_t* w,
    float scale,
    float bias,
    threadgroup T* w_local) {
  float w_thread[N];
  dequantize<float, N, bits, nonlinear>(w, scale, bias, w_thread);
  for (int i = 0; i < N; i++) {
    w_local[i] = static_cast<T>(w_thread[i]);
  }
}

// Decodes one group's (scale, bias) from the two K-quant scale levels.
//
//   q6k  scale = d[g / 16] * int8(sc[g])
//        bias  = -32 * scale                          (symmetric, q - 32)
//   q4k
//   q5k  as q4k with a fifth bit plane
//        scale = d[2 * (g / 8)]     * sc[2 * g]
//        bias  = -(d[2 * (g / 8) + 1] * sc[2 * g + 1])
//
// `.biases` carries ggml's super-block scale d (and dmin for q4k and q5k), not
// a bias; the bias is derived above.
//
// The decode is random access rather than a walk because the kernels reach
// groups out of order: qmv_wide strides its group loop by the lane count and
// qmv_fast starts at an offset that is not super-block aligned. Every row holds
// a whole number of 256-value super-blocks, so a flat group index stays aligned
// as it runs across rows, which is what lets a scale cursor walk off the end of
// one row into the next -- the same property the affine kernels rely on.
//
// Mirrors KQScales in mlx_kquant.cpp (CPU reference): same operand order and the
// same float32 conversions, so the two decodes agree bitwise.
template <typename U, int bits, int super_ratio, bool has_min>
struct KQScales {
  // Sub-scale entries per group: (sc, m) for q4k and q5k, sc alone for q6k.
  MLX_MTL_CONST int per_group = has_min ? 2 : 1;

  const device uint8_t* scales;
  const device float16_t* biases;
  size_t group;

  KQScales(
      const device uint8_t* scales_,
      const device float16_t* biases_,
      size_t group_ = 0)
      : scales(scales_), biases(biases_), group(group_) {}

  void at(size_t g, thread U& scale, thread U& bias) const {
    const size_t gi = group + g;
    const device float16_t* d = biases + (gi / super_ratio) * per_group;
    const device uint8_t* sc = scales + gi * per_group;
    if constexpr (has_min) {
      scale = static_cast<U>(d[0]) * static_cast<U>(sc[0]);
      bias = -(static_cast<U>(d[1]) * static_cast<U>(sc[1]));
    } else {
      // as_type is a bit reinterpretation, so this reads the ggml sub-scale as
      // signed exactly the way the CPU reference's static_cast<int8_t> does.
      scale = static_cast<U>(d[0]) * static_cast<U>(as_type<int8_t>(sc[0]));
      if constexpr (bits == 4) {
        // IQ4_NL / IQ4_XS apply the non-linear codebook in the dot/dequant
        // helper, so there is no affine zero point.
        bias = static_cast<U>(0.0f);
      } else {
        bias = static_cast<U>(-(1 << (bits - 1))) * scale;
      }
    }
  }

  // A view whose group 0 is this view's group n.
  KQScales offset(size_t n) const {
    return KQScales(scales, biases, group + n);
  }

  void advance(size_t n) {
    group += n;
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits,
    short super_ratio,
    bool has_min>
struct QuantizedBlockLoader {
  static_assert(
      bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
      "Template undefined for bits not in {3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  // Q6_K's group of 16 is narrower than the BK = 32 tile, so one tile spans
  // several groups. The affine loader asserts that away with
  // `BCOLS <= group_size`; this is fp_quantized.h's generalization, which
  // nvfp4's group of 16 needs for the same reason.
  MLX_MTL_CONST short group_steps = group_size < BCOLS ? 1 : group_size / BCOLS;
  MLX_MTL_CONST short scale_step = group_size < BCOLS ? BCOLS / group_size : 1;

  static_assert(
      (n_reads * pack_factor) <= group_size,
      "The number of reads per thread must be less than the group size.");

  using scales_t = KQScales<float, bits, super_ratio, has_min>;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  scales_t scales;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const scales_t scales_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_.offset(
            bi * src_ld / group_size + (bj * pack_factor) / group_size)) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    float scale;
    float bias;
    scales.at(0, scale, bias);
    for (int i = 0; i < n_reads; i++) {
      dequantize_to<T, pack_factor, bits, bits == 4 && !has_min>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    float scale;
    float bias;
    scales.at(0, scale, bias);
    for (int i = 0; i < n_reads; i++) {
      dequantize_to<T, pack_factor, bits, bits == 4 && !has_min>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales.advance(1);
        }
      } else {
        scales.advance(scale_step);
      }
    } else {
      scales.advance(group_stride);
    }
  }
};


template <typename T, int group_size, int bits, int super_ratio, bool has_min>
METAL_FUNC void kquant_qmv_fast_impl(
    const device uint32_t* w,
    KQScales<float, bits, super_ratio, has_min> scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales.advance(out_row * in_vec_size_g + simd_lid / scale_step_per_thread);
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);

      U s;
      U b;
      scales.at(row * in_vec_size_g, s, b);
      result[row] += qdot<U, values_per_thread, bits, bits == 4 && !has_min>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales.advance(block_size / group_size);
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits, int super_ratio, bool has_min>
METAL_FUNC void kquant_qmv_impl(
    const device uint32_t* w,
    KQScales<float, bits, super_ratio, has_min> scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales.advance(out_row * in_vec_size_g + simd_lid / scale_step_per_thread);
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);

        U s;
        U b;
        scales.at(row * in_vec_size_g, s, b);
        result[row] +=
            qdot<U, values_per_thread, bits, bits == 4 && !has_min>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales.advance(block_size / group_size);
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);

        U s;
        U b;
        scales.at(row * in_vec_size_g, s, b);
        result[row] += qdot_safe<U, values_per_thread, bits, bits == 4 && !has_min>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales.advance(
        used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread);
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);

        U s;
        U b;
        scales.at(row * in_vec_size_g, s, b);
        result[row] +=
            qdot<U, values_per_thread, bits, bits == 4 && !has_min>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales.advance(block_size / group_size);
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);

        U s;
        U b;
        scales.at(row * in_vec_size_g, s, b);
        result[row] += qdot_safe<U, values_per_thread, bits, bits == 4 && !has_min>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

// Affine analog of fp_qmv_wide. Weights carry a scale and bias per group, so
// each group is decoded in 8-value sub-chunks (scale * q + bias, registers
// bounded for any group_size) and reused across the vecs_per_tg vectors.
template <
    typename T,
    int group_size,
    int bits,
    int super_ratio,
    bool has_min,
    int vecs_per_tg,
    int k_lanes>
METAL_FUNC void kquant_qmv_wide_impl(
    const device uint32_t* w,
    KQScales<float, bits, super_ratio, has_min> scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;
  constexpr int sub = 8; // values per sub-chunk (== bits bytes, byte-aligned)

  typedef float U;

  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;

  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +
      results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;

  const int row = min(out_row, out_vec_size - 1);

  const int in_vec_size_w = in_vec_size * bits / 8; // bytes per weight row
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
  auto srow = scales.offset(row * in_vec_size_g);

  const device T* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) {
    xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
  }

  U result[vecs_per_tg] = {0};

  // Each lane reduces a strided subset of the row's groups: decode the group in
  // 8-value sub-chunks and reuse each chunk across the streamed vectors.
  for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {
    U scale;
    U bias;
    srow.at(g, scale, bias);
#pragma unroll
    for (int sc = 0; sc < group_size / sub; sc++) {
      const int k0 = g * group_size + sc * sub;
      const device uint8_t* wc = wrow + k0 * bits / 8;
      U w_dq[sub];
      dequantize<U, sub, bits, bits == 4 && !has_min>(
          wc, scale, bias, w_dq);
#pragma unroll
      for (int v = 0; v < vecs_per_tg; v++) {
        const device T* xc = xv[v] + k0;
        U acc = 0;
#pragma unroll
        for (int i = 0; i < sub; i++) {
          acc += static_cast<U>(xc[i]) * w_dq[i];
        }
        result[v] += acc;
      }
    }
  }

  // Reduce each vector's partial over its k_lanes with a shuffle ladder:
  // simd_sum would mix the results_per_simdgroup rows a simdgroup spans.
  for (int v = 0; v < vecs_per_tg; v++) {
    if constexpr (k_lanes >= 32) {
      result[v] += simd_shuffle_down(result[v], 16);
    }
    if constexpr (k_lanes >= 16) {
      result[v] += simd_shuffle_down(result[v], 8);
    }
    if constexpr (k_lanes >= 8) {
      result[v] += simd_shuffle_down(result[v], 4);
    }
    if constexpr (k_lanes >= 4) {
      result[v] += simd_shuffle_down(result[v], 2);
    }
    if constexpr (k_lanes >= 2) {
      result[v] += simd_shuffle_down(result[v], 1);
    }
  }

  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) {
        y[(vec0 + v) * out_vec_size + out_row] = static_cast<T>(result[v]);
      }
    }
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min>
METAL_FUNC void kquant_qvm_impl(
    const device uint32_t* w,
    KQScales<float, bits, super_ratio, has_min> scales,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    const int in_vec_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int tn = 32 / pack_factor;
  constexpr int block_size = SIMD_SIZE;
  // A thread owns 32 output columns per step. Q6_K's group of 16 makes that two
  // groups, so the outer product runs once per group instead of once per step.
  constexpr int groups_per_step = tn * pack_factor / group_size;
  constexpr int bytes_per_group = group_size * bits / 8;

  using W_T =
      typename ConditionalType<power_of_2_bits, uint32_t, uint8_t>::type;
  const device W_T* ws = (const device W_T*)w;

  typedef float U;
  typedef struct {
    W_T wi[tn * bytes_per_pack];
  } vec_w;

  thread vec_w w_local;
  thread U result[tn * pack_factor] = {0};
  thread U scale[groups_per_step] = {0};
  thread U bias[groups_per_step] = {0};
  thread U x_local = 0;

  // Adjust positions
  const int out_vec_size_w = out_vec_size * bytes_per_pack / pack_factor;
  const int out_vec_size_g = out_vec_size / group_size;
  int out_col = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);
  ws += out_col * bytes_per_pack / pack_factor + simd_lid * out_vec_size_w;
  scales.advance(out_col / group_size + simd_lid * out_vec_size_g);
  x += tid.x * in_vec_stride + simd_lid;
  y += tid.x * out_vec_size + out_col;

  if (out_col >= out_vec_size) {
    return;
  }

  // Loop over in_vec in blocks of block_size
  int remaining = in_vec_size % block_size;
  if (remaining == 0) {
    for (int i = 0; i < in_vec_size; i += block_size) {
      x_local = *x;
#pragma clang loop unroll(full)
      for (int g = 0; g < groups_per_step; g++) {
        scales.at(g, scale[g], bias[g]);
      }
      w_local = *((device vec_w*)ws);
#pragma clang loop unroll(full)
      for (int g = 0; g < groups_per_step; g++) {
        qouter<U, group_size, bits, bits == 4 && !has_min>(
            (thread uint8_t*)&w_local + g * bytes_per_group,
            x_local,
            scale[g],
            bias[g],
            result + g * group_size);
      }

      x += block_size;
      scales.advance(block_size * out_vec_size_g);
      ws += block_size * out_vec_size_w;
    }
  } else {
    for (int i = block_size; i < in_vec_size; i += block_size) {
      x_local = *x;
#pragma clang loop unroll(full)
      for (int g = 0; g < groups_per_step; g++) {
        scales.at(g, scale[g], bias[g]);
      }
      w_local = *((device vec_w*)ws);

#pragma clang loop unroll(full)
      for (int g = 0; g < groups_per_step; g++) {
        qouter<U, group_size, bits, bits == 4 && !has_min>(
            (thread uint8_t*)&w_local + g * bytes_per_group,
            x_local,
            scale[g],
            bias[g],
            result + g * group_size);
      }

      x += block_size;
      scales.advance(block_size * out_vec_size_g);
      ws += block_size * out_vec_size_w;
    }
    if (static_cast<int>(simd_lid) < remaining) {
      x_local = *x;
#pragma clang loop unroll(full)
      for (int g = 0; g < groups_per_step; g++) {
        scales.at(g, scale[g], bias[g]);
      }
      w_local = *((device vec_w*)ws);
    } else {
      x_local = 0;
#pragma clang loop unroll(full)
      for (int g = 0; g < groups_per_step; g++) {
        scale[g] = 0;
        bias[g] = 0;
      }
    }
#pragma clang loop unroll(full)
    for (int g = 0; g < groups_per_step; g++) {
      qouter<U, group_size, bits, bits == 4 && !has_min>(
          (thread uint8_t*)&w_local + g * bytes_per_group,
          x_local,
          scale[g],
          bias[g],
          result + g * group_size);
    }
  }

// Accumulate in the simdgroup
#pragma clang loop unroll(full)
  for (int k = 0; k < tn * pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  // Store the result
  if (simd_lid == 0) {
#pragma clang loop unroll(full)
    for (int k = 0; k < tn * pack_factor; k++) {
      y[k] = static_cast<T>(result[k]);
    }
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void kquant_qmm_t_impl(
    const device uint32_t* w,
    KQScales<float, bits, super_ratio, has_min> scales,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& K_eff,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      super_ratio,
      has_min>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales.advance(static_cast<size_t>(y_col) * K_g);
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void kquant_qmm_n_impl(
    const device uint32_t* w,
    KQScales<float, bits, super_ratio, has_min> scales,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using loader_x_t = mlx::steel::
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      super_ratio,
      has_min>;

  auto wl = (const device uint8_t*)w;

  // Set the block
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  scales.advance(y_col / group_size);
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, N, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, num_els));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, BM));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device uint8_t*& scales,
    const device float16_t*& biases,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device uint8_t*& scales,
    const device float16_t*& biases,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <
    typename T,
    int group_size,
    int bits,
    int super_ratio,
    bool has_min,
    bool batched>
[[kernel]] void kquant_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  kquant_qmv_fast_impl<T, group_size, bits, super_ratio, has_min>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    bool batched>
[[kernel]] void kquant_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  kquant_qmv_impl<T, group_size, bits, super_ratio, has_min>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int super_ratio,
    bool has_min,
    int vecs_per_tg,
    int k_lanes,
    bool batched>
[[kernel]] void kquant_qmv_wide(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float16_t* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  kquant_qmv_wide_impl<
      T,
      group_size,
      bits,
      super_ratio,
      has_min,
      vecs_per_tg,
      k_lanes>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      in_vec_size,
      out_vec_size,
      M,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    bool batched>
[[kernel]] void kquant_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  kquant_qvm_impl<T, group_size, bits, super_ratio, has_min>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      in_vec_size,
      out_vec_size,
      in_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    int split_k = 32>
[[kernel]] void kquant_qvm_split_k(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    const constant int& final_block_size [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      y,
      out_vec_size * M,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);

  // When (in_vec_size % split_k != 0) the final block needs to be smaller
  int in_vec_size_adj =
      tid.z % split_k == split_k - 1 ? final_block_size : in_vec_size;

  // The in_vec_stride is the full K dimension, not the partition size
  int in_vec_stride = (split_k - 1) * in_vec_size + final_block_size;

  kquant_qvm_impl<T, group_size, bits, super_ratio, has_min>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      in_vec_size_adj,
      out_vec_size,
      in_vec_stride,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    const bool aligned_N,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void kquant_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  kquant_qmm_t_impl<
      T,
      group_size,
      bits,
      super_ratio,
      has_min,
      aligned_N,
      BM,
      BK,
      BN>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      K,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void kquant_qmm_t_splitk(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& k_partition_size [[buffer(8)]],
    const constant int& split_k_partition_stride [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int k_start = tid.z * k_partition_size;
  x += k_start;

  auto wl = (const device uint8_t*)w;
  wl += k_start * bytes_per_pack / pack_factor;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  kquant_qmm_t_impl<
      T,
      group_size,
      bits,
      super_ratio,
      has_min,
      aligned_N,
      BM,
      BK,
      BN>(
      (const device uint32_t*)wl,
      KQScales<float, bits, super_ratio, has_min>(
          scales, biases, k_start / group_size),
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void kquant_qmm_n(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }

  kquant_qmm_n_impl<T, group_size, bits, super_ratio, has_min, BM, BK, BN>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int super_ratio,
    bool has_min>
[[kernel]] void kquant_gather_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  kquant_qmv_fast_impl<T, group_size, bits, super_ratio, has_min>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int super_ratio,
    bool has_min>
[[kernel]] void kquant_gather_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  kquant_qmv_impl<T, group_size, bits, super_ratio, has_min>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int super_ratio,
    bool has_min>
[[kernel]] void kquant_gather_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  kquant_qvm_impl<T, group_size, bits, super_ratio, has_min>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      in_vec_size,
      out_vec_size,
      in_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void kquant_gather_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  kquant_qmm_t_impl<
      T,
      group_size,
      bits,
      super_ratio,
      has_min,
      aligned_N,
      BM,
      BK,
      BN>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      K,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void kquant_gather_qmm_n(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  kquant_qmm_n_impl<T, group_size, bits, super_ratio, has_min, BM, BK, BN>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int super_ratio,
    bool has_min,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
[[kernel]] void kquant_gather_qmm_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device uint8_t* scales [[buffer(2)]],
    const device float16_t* biases [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      transpose,
      BK_padded,
      transpose ? BK_padded : BN_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      super_ratio,
      has_min>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  KQScales<float, bits, super_ratio, has_min> sb(
      scales,
      biases,
      transpose ? y_col_long * K_g : size_t(y_col / group_size));

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    // Prepare threadgroup loading operations
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(
        wl + index * stride_w,
        sb.offset(index * stride_s),
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    // Matrices are all aligned check nothing
    if (align_M && align_N) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }

      // Store results to device memory
      if (offset_next - offset == BM) {
        mma_op.store_result(y, N);
      } else {
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }
    } else {
      // Tile aligned so check outside of the hot loop
      if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
        gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }

        // Store results to device memory
        if (offset_next - offset == BM) {
          mma_op.store_result(y, N);
        } else {
          mma_op.store_result_slice(
              y, N, short2(0, offset), short2(BN, offset_next));
        }
      }

      // Tile partially aligned check rows
      else if (align_N || tgp_bn == BN) {
        gemm_loop_unaligned<false, true, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }

      // Tile partially aligned check cols
      else if (align_M || tgp_bm == BM) {
        gemm_loop_unaligned<true, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }

      // Nothing aligned so check both rows and cols
      else {
        gemm_loop_unaligned<false, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }
    }
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    int super_ratio,
    bool has_min>
[[kernel]] void kquant_dequantize(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t oindex = offset * pack_factor;
  size_t gindex = oindex / group_size;

  // Decode in float and round once on the store, as kq_dequantize does.
  float scale;
  float bias;
  KQScales<float, bits, super_ratio, has_min>(scales, biases).at(gindex, scale, bias);

  out += oindex;
  w += offset * bytes_per_pack;

  float values[pack_factor];
  dequantize<float, pack_factor, bits, bits == 4 && !has_min>(
      w, scale, bias, values);
#pragma clang loop unroll(full)
  for (int i = 0; i < pack_factor; i++) {
    out[i] = static_cast<T>(values[i]);
  }
}

// M = 8 bfloat16 matvec on simdgroup matrices for q4k, q5k, iq4xs and q6k.
//
// One simdgroup owns 8 output columns and walks the whole K. Each 8x8x8 bf16
// MMA takes A = 8 columns x 8 codes, B = 8 k x the 8 activation rows, C = fp32
// (column, row). A code q becomes the exact bf16 0x4300 | q = 128 + q, so the
// chain is seeded with -128 * sum(x) (-160 * sum(x) for q6k's q - 32); iq4xs
// codes are exact bf16 codebook values. A per-group fp32 epilogue applies the
// scale (and q4k/q5k's minimum). The K sum is reassociated, so the result is
// not bit-identical to qmv_wide.
//
// Lane map (thread_elements of simdgroup_matrix): a lane holds M[fm][fn] and
// M[fm][fn + 1]. In a 32-code group lane c = fn / 2 owns codes [8c, 8c + 8) of
// column fm and fragment j pairs codes (8c + j, 8c + 4 + j): logical k index
// kk of fragment j is physical k = 4 kk + j, which is what kquant_qmv_sg8_prep
// lays out in bt. q6k's 32-code span holds two 16-groups: fragments 0,1 cover
// the first (k = 2 kk + j), 2,3 the second.
namespace kq_sg8 {

enum Format : int { Q4K, Q5K, IQ4XS, Q6K, Unsupported };

template <int group_size, int bits, int super_ratio, bool has_min>
constexpr Format format() {
  if (group_size == 32 && super_ratio == 8 && bits == 4) {
    return has_min ? Q4K : IQ4XS;
  }
  if (group_size == 32 && super_ratio == 8 && bits == 5 && has_min) {
    return Q5K;
  }
  if (group_size == 16 && super_ratio == 16 && bits == 6 && !has_min) {
    return Q6K;
  }
  return Unsupported;
}

template <typename T>
METAL_FUNC thread vec<T, 2>& te(thread simdgroup_matrix<T, 8, 8>& m) {
  return reinterpret_cast<thread vec<T, 2>&>(m.thread_elements());
}

// c += a x b on register operands.
METAL_FUNC void mma(thread float2& c, uint a, uint b) {
  simdgroup_matrix<bfloat, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = as_type<bfloat2>(a);
  te(B) = as_type<bfloat2>(b);
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

METAL_FUNC float bf_lo(uint v) {
  return as_type<float>(v << 16);
}

METAL_FUNC float bf_hi(uint v) {
  return as_type<float>(v & 0xFFFF0000u);
}

// Sum over the 8 lanes that share fn (fm lives in lane bits 1, 2 and 4).
METAL_FUNC float fm_reduce(float v) {
  v += simd_shuffle_xor(v, ushort(2));
  v += simd_shuffle_xor(v, ushort(4));
  v += simd_shuffle_xor(v, ushort(16));
  return v;
}

// B operand pairs (x[fn][k], x[fn + 1][k]) for k = 4 fm + j, j = 0..3.
METAL_FUNC void b_pairs4(uint2 a, uint2 b, thread uint (&bq)[4]) {
  bq[0] = (a.x & 0xFFFFu) | (b.x << 16);
  bq[1] = (a.x >> 16) | (b.x & 0xFFFF0000u);
  bq[2] = (a.y & 0xFFFFu) | (b.y << 16);
  bq[3] = (a.y >> 16) | (b.y & 0xFFFF0000u);
}

constant constexpr int kIQ4[16] =
    {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// A unit is 32 consecutive k of one weight row: one group, or q6k's two
// 16-groups. A super-block is 8 units.
template <Format F>
struct Codes;

template <>
struct Codes<Q4K> {
  typedef uint W;
  static W load(const device uint32_t* row, uint u, uint c) {
    return row[u * 4 + c];
  }
};

template <>
struct Codes<IQ4XS> {
  typedef uint W;
  static W load(const device uint32_t* row, uint u, uint c) {
    return row[u * 4 + c];
  }
};

template <>
struct Codes<Q5K> {
  typedef uint2 W;
  // Lane c's 40 bits start at bit 8c of word c of the 5-word group.
  static W load(const device uint32_t* row, uint u, uint c) {
    return uint2(row[u * 5 + c], row[u * 5 + c + 1]);
  }
};

template <>
struct Codes<Q6K> {
  typedef uint4 W;
  // The first group's 24-bit chunk starts at bit 24c of the 6-word span, the
  // second's at bit 96 + 24c; .xy and .zw hold the words covering each. For
  // c = 3 the second chunk ends at bit 32 of word 5, so its high word is
  // unused and word 5 is re-read instead of reading past the span.
  static W load(const device uint32_t* row, uint u, uint c) {
    const uint wa = (24 * c) >> 5;
    const uint wb = (96 + 24 * c) >> 5;
    const device uint32_t* s = row + u * 6;
    return uint4(s[wa], s[wa + 1], s[wb], s[wb + 1 < 6 ? wb + 1 : wb]);
  }
};

// One super-block's scale data: .a = sub-scale bytes, .h = fp16 super scales.
struct SB {
  uint4 a;
  half2 h;
};

template <Format F>
METAL_FUNC SB load_sb(
    const device uint8_t* scales,
    const device half* biases,
    uint n,
    uint K,
    uint G) {
  SB s;
  if constexpr (F == IQ4XS) {
    const uint2 v = *reinterpret_cast<const device uint2*>(
        scales + size_t(n) * (K / 32) + G * 8);
    s.a = uint4(v, 0u, 0u);
    s.h = half2(biases[size_t(n) * (K / 256) + G], 0.0h);
  } else if constexpr (F == Q6K) {
    s.a = *reinterpret_cast<const device uint4*>(
        scales + size_t(n) * (K / 16) + G * 16);
    s.h = half2(biases[size_t(n) * (K / 256) + G], 0.0h);
  } else {
    s.a = *reinterpret_cast<const device uint4*>(
        scales + size_t(n) * (K / 16) + G * 16);
    s.h = *reinterpret_cast<const device half2*>(
        biases + size_t(n) * (K / 128) + G * 2);
  }
  return s;
}

} // namespace kq_sg8

// bt[u * 32 + lane] holds the 4 B pairs lane uses for unit u; sums[g * 8 + m]
// is the fp32 sum of row m over group g. One simdgroup per unit.
template <typename T, int group_size>
[[kernel]] void kquant_qmv_sg8_prep(
    const device T* x [[buffer(0)]],
    device uint4* bt [[buffer(1)]],
    device float* sums [[buffer(2)]],
    const constant int& in_vec_size [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using namespace kq_sg8;
  static_assert(is_same_v<T, bfloat>, "the MMA operands are bfloat16");
  static_assert(group_size == 16 || group_size == 32, "q6k or a 32-group");
  const uint K = in_vec_size;
  const uint s = gid * sgs + sg;
  if (s >= K / 32) {
    return;
  }
  const uint qid = lane >> 2;
  const uint fm = (qid & 4u) | ((lane >> 1) & 3u);
  const uint fn = ((qid & 2u) << 1) | ((lane & 1u) << 1);
  if constexpr (group_size == 32) {
    const uint2 a =
        reinterpret_cast<const device uint2*>(x + fn * K + 4 * fm)[s * 8];
    const uint2 b =
        reinterpret_cast<const device uint2*>(x + (fn + 1) * K + 4 * fm)[s * 8];
    uint bq[4];
    b_pairs4(a, b, bq);
    bt[s * 32 + lane] = uint4(bq[0], bq[1], bq[2], bq[3]);
    const float sa =
        fm_reduce((bf_lo(a.x) + bf_hi(a.x)) + (bf_lo(a.y) + bf_hi(a.y)));
    const float sb =
        fm_reduce((bf_lo(b.x) + bf_hi(b.x)) + (bf_lo(b.y) + bf_hi(b.y)));
    if (fm == 0) {
      sums[s * 8 + fn] = sa;
      sums[s * 8 + fn + 1] = sb;
    }
  } else {
    const device uint* xa =
        reinterpret_cast<const device uint*>(x + fn * K + 2 * fm);
    const device uint* xb =
        reinterpret_cast<const device uint*>(x + (fn + 1) * K + 2 * fm);
    const uint a0 = xa[s * 16];
    const uint b0 = xb[s * 16];
    const uint a1 = xa[s * 16 + 8];
    const uint b1 = xb[s * 16 + 8];
    bt[s * 32 + lane] = uint4(
        (a0 & 0xFFFFu) | (b0 << 16),
        (a0 >> 16) | (b0 & 0xFFFF0000u),
        (a1 & 0xFFFFu) | (b1 << 16),
        (a1 >> 16) | (b1 & 0xFFFF0000u));
    const float saA = fm_reduce(bf_lo(a0) + bf_hi(a0));
    const float sbA = fm_reduce(bf_lo(b0) + bf_hi(b0));
    const float saB = fm_reduce(bf_lo(a1) + bf_hi(a1));
    const float sbB = fm_reduce(bf_lo(b1) + bf_hi(b1));
    if (fm == 0) {
      sums[(2 * s) * 8 + fn] = saA;
      sums[(2 * s) * 8 + fn + 1] = sbA;
      sums[(2 * s + 1) * 8 + fn] = saB;
      sums[(2 * s + 1) * 8 + fn + 1] = sbB;
    }
  }
}

// Grid: N / 32 threadgroups of 4 simdgroups. Each super-block's codes and
// scales are loaded one iteration ahead (the last iteration re-reads its own).
template <typename T, int group_size, int bits, int super_ratio, bool has_min>
[[kernel, max_total_threads_per_threadgroup(128)]] void kquant_qmv_sg8(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device float16_t* biases [[buffer(2)]],
    const device uint4* bt [[buffer(3)]],
    const device float* sums [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  using namespace kq_sg8;
  constexpr Format F = format<group_size, bits, super_ratio, has_min>();
  static_assert(F != Unsupported, "no simdgroup-matrix decode for this mode");
  static_assert(is_same_v<T, bfloat>, "the MMA operands are bfloat16");
  typedef typename Codes<F>::W W;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4u) | ((lane >> 1) & 3u);
  const uint fn = ((qid & 2u) << 1) | ((lane & 1u) << 1);
  const uint c = fn >> 1;
  const uint K = in_vec_size;
  const uint N = out_vec_size;
  const uint rw = K * bits / 32;
  const uint nsb = K / 256;

  threadgroup uint lut2[256];
  if constexpr (F == IQ4XS) {
    // lut2[lo | hi << 4] = bf16(v[lo]) | bf16(v[hi]) << 16, both exact.
    for (uint i = tid; i < 256; i += 128) {
      const uint lo = as_type<ushort>(bfloat(float(kIQ4[i & 15])));
      const uint hi = as_type<ushort>(bfloat(float(kIQ4[i >> 4])));
      lut2[i] = lo | (hi << 16);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint n0 = tg * 32 + sg * 8 + fm;
  const device uint32_t* row = w + size_t(n0) * rw;
  float2 acc = float2(0.0f);

  SB scur = load_sb<F>(scales, biases, n0, K, 0);
  W wcur[8];
#pragma unroll
  for (uint gi = 0; gi < 8; ++gi) {
    wcur[gi] = Codes<F>::load(row, gi, c);
  }

  for (uint G = 0; G < nsb; ++G) {
    const uint Gl = G + 1 < nsb ? G + 1 : G;
    const SB snxt = load_sb<F>(scales, biases, n0, K, Gl);
    W wnxt[8];
#pragma unroll
    for (uint gi = 0; gi < 8; ++gi) {
      wnxt[gi] = Codes<F>::load(row, Gl * 8 + gi, c);
    }

#pragma unroll
    for (uint gi = 0; gi < 8; ++gi) {
      const uint u = G * 8 + gi;
      const uint4 t = bt[u * 32 + lane];
      uint bq[4] = {t.x, t.y, t.z, t.w};
      if constexpr (F != Q6K) {
        float sa = 0.0f;
        float sb = 0.0f;
        if constexpr (F != IQ4XS) {
          const float2 s2 =
              *reinterpret_cast<const device float2*>(sums + u * 8 + fn);
          sa = s2.x;
          sb = s2.y;
        }
        uint pq[4];
        if constexpr (F == Q4K) {
          const uint wd = wcur[gi];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            pq[j] = ((wd >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
          }
        } else if constexpr (F == Q5K) {
          const uint2 wv = wcur[gi];
          const ulong v = ((ulong(wv.y) << 32) | ulong(wv.x)) >> (8 * c);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint t5 = uint(v >> (5 * j));
            pq[j] = (t5 & 0x1Fu) | ((t5 >> 4) & 0x001F0000u) | 0x43004300u;
          }
        } else {
          const uint wd = wcur[gi];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint t4 = wd >> (4 * j);
            pq[j] = lut2[(t4 & 0xFu) | ((t4 >> 12) & 0xF0u)];
          }
        }
        float2 c0 =
            F == IQ4XS ? float2(0.0f) : float2(-128.0f * sa, -128.0f * sb);
        float2 c1 = float2(0.0f);
        mma(c0, pq[0], bq[0]);
        mma(c1, pq[1], bq[1]);
        mma(c0, pq[2], bq[2]);
        mma(c1, pq[3], bq[3]);
        const float2 d = c0 + c1;
        if constexpr (F == IQ4XS) {
          const uint byte = (gi < 4 ? scur.a.x : scur.a.y) >> (8 * (gi & 3));
          const float scale =
              float(scur.h.x) * float(as_type<char>(uchar(byte & 0xFFu)));
          acc = fma(d, float2(scale), acc);
        } else {
          const uint uu = scur.a[gi >> 1] >> (16 * (gi & 1));
          const float scale = float(scur.h.x) * float(uu & 0xFFu);
          const float bias = -(float(scur.h.y) * float((uu >> 8) & 0xFFu));
          acc = fma(d, float2(scale), acc);
          acc = fma(float2(sa, sb), float2(bias), acc);
        }
      } else {
        const float2 sA =
            *reinterpret_cast<const device float2*>(sums + (2 * u) * 8 + fn);
        const float2 sB = *reinterpret_cast<const device float2*>(
            sums + (2 * u + 1) * 8 + fn);
        const uint sha = (24 * c) & 31u;
        const uint shb = (96 + 24 * c) & 31u;
        const uint4 wv = wcur[gi];
        const ulong va = ((ulong(wv.y) << 32) | ulong(wv.x)) >> sha;
        const ulong vb = ((ulong(wv.w) << 32) | ulong(wv.z)) >> shb;
        uint pq[4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const uint ta = uint(va >> (6 * j));
          const uint tb = uint(vb >> (6 * j));
          pq[j] = (ta & 0x3Fu) | ((ta << 4) & 0x003F0000u) | 0x43004300u;
          pq[2 + j] = (tb & 0x3Fu) | ((tb << 4) & 0x003F0000u) | 0x43004300u;
        }
        // (q - 32) x = (128 + q) x - 160 x
        float2 cA = float2(-160.0f * sA.x, -160.0f * sA.y);
        float2 cB = float2(-160.0f * sB.x, -160.0f * sB.y);
        mma(cA, pq[0], bq[0]);
        mma(cB, pq[2], bq[2]);
        mma(cA, pq[1], bq[1]);
        mma(cB, pq[3], bq[3]);
        const uint gA = 2 * gi;
        const uint word = scur.a[gA >> 2];
        const float dS = float(scur.h.x);
        const float scA =
            dS * float(as_type<char>(uchar((word >> (8 * (gA & 3))) & 0xFFu)));
        const float scB = dS *
            float(as_type<char>(uchar((word >> (8 * ((gA + 1) & 3))) & 0xFFu)));
        acc = fma(cA, float2(scA), acc);
        acc = fma(cB, float2(scB), acc);
      }
    }

    scur = snxt;
#pragma unroll
    for (uint gi = 0; gi < 8; ++gi) {
      wcur[gi] = wnxt[gi];
    }
  }

  y[fn * N + n0] = static_cast<T>(acc.x);
  y[(fn + 1) * N + n0] = static_cast<T>(acc.y);
}
