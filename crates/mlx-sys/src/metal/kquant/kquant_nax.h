// Copyright © 2023-2024 Apple Inc.

// ggml K-quant (Q6_K / Q4_K / Q5_K) kernels on the NAX tensor-op path.
//
// This is quantized_nax.h with the same substitution kquant.h makes to
// quantized.h: the per-group scalar load `s = scales[g]; b = biases[g]`
// becomes the two-level decode in KQScales. NAX imposes no scale layout of its
// own -- the block loader dequantizes into a threadgroup tile and the tensor-op
// kernel reads that tile -- so only the loader changes.
//
// The one thing that does not carry over from kquant.h is its
// `n_reads * pack_factor <= group_size` assumption. NAX runs BK = BN = 64 with
// a 128-thread threadgroup, so a q6k thread reads 8 packs of 4 values = 32
// values, spanning two groups of 16. fp_quantized_nax.h already solves exactly
// this for nvfp4's group of 16; n_reads_per_scale / n_steps_per_read below are
// its solution, applied to the K-quant decode.
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace mlx::steel;

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

// Decode one quantized block (scale * q + bias) into w_local. W (the output
// pointer type) serves the threadgroup block loader or a thread-local decode.
MLX_MTL_CONST int8_t kIQ4NLValuesNAX[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,    25,  38,  53,  69,  89,  113};

template <typename U, int N, int bits, bool nonlinear, typename W>
inline void dequantize(const device uint8_t* w, U scale, U bias, W w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if constexpr (nonlinear) {
    for (int i = 0; i < N; i++) {
      const uint code = (w[i / 2] >> (4 * (i & 1))) & 0x0f;
      w_local[i] = scale * static_cast<U>(kIQ4NLValuesNAX[code]);
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

// --- Tiled64 layout (mlx_kquant.h; the same helpers as kquant.h) -----------
MLX_MTL_CONST int KQ_TILE_ROWS = 64;

template <int bits>
METAL_FUNC const device uint8_t* kq_tiled_row(
    const device uint32_t* w,
    uint n,
    uint units_per_row) {
  return reinterpret_cast<const device uint8_t*>(w) +
      (size_t(n / KQ_TILE_ROWS) * units_per_row * KQ_TILE_ROWS +
       n % KQ_TILE_ROWS) *
      (bits * 4);
}

template <int bits>
METAL_FUNC constexpr size_t kq_tiled_unit_stride() {
  return size_t(KQ_TILE_ROWS) * bits * 4;
}

template <typename U, int bits, int super_ratio, bool has_min>
struct KQScalesTiled {
  MLX_MTL_CONST int per_group = has_min ? 2 : 1;

  const device uint8_t* scales;
  const device float16_t* biases;
  size_t group;

  // Sub-scales sit per super-block: row r's super_ratio groups of
  // super-block G are the contiguous bytes at ((tile * nsb + G) * 64 + r) *
  // super_ratio * per_group, so one 16-byte load covers a super-block.
  KQScalesTiled(
      const device uint8_t* scales_,
      const device float16_t* biases_,
      int groups_per_row,
      uint row,
      size_t group_ = 0)
      : scales(
            scales_ +
            (size_t(row / KQ_TILE_ROWS) * (groups_per_row / super_ratio) *
                 KQ_TILE_ROWS +
             row % KQ_TILE_ROWS) *
                (super_ratio * per_group)),
        biases(
            biases_ +
            (size_t(row / KQ_TILE_ROWS) * (groups_per_row / super_ratio) *
                 KQ_TILE_ROWS +
             row % KQ_TILE_ROWS) *
                per_group),
        group(group_) {}

  void at(size_t g, thread U& scale, thread U& bias) const {
    const size_t gi = group + g;
    const device float16_t* d =
        biases + (gi / super_ratio) * KQ_TILE_ROWS * per_group;
    const device uint8_t* sc = scales +
        (gi / super_ratio) * KQ_TILE_ROWS * super_ratio * per_group +
        (gi % super_ratio) * per_group;
    if constexpr (has_min) {
      scale = static_cast<U>(d[0]) * static_cast<U>(sc[0]);
      bias = -(static_cast<U>(d[1]) * static_cast<U>(sc[1]));
    } else {
      scale = static_cast<U>(d[0]) * static_cast<U>(as_type<int8_t>(sc[0]));
      if constexpr (bits == 4) {
        bias = static_cast<U>(0.0f);
      } else {
        bias = static_cast<U>(-(1 << (bits - 1))) * scale;
      }
    }
  }

  KQScalesTiled offset(size_t n) const {
    KQScalesTiled c = *this;
    c.group += n;
    return c;
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
    bool has_min,
    bool tiled = false>
struct QuantizedBlockLoader {
  static_assert(
      bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
      "Template undefined for bits not in {3, 4, 5, 6, 8}");
  static_assert(
      !tiled || (reduction_dim == 1 && BCOLS % 32 == 0),
      "Tiled64 loads whole 32-code units of a transposed weight.");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  // A tile wider than a group needs several scales per tile; a thread whose
  // reads span more than one group needs several scales per *load*. The first
  // is `scale_step`, the second is the (n_reads_per_scale, n_steps_per_read)
  // split -- both are fp_quantized_nax.h's, which nvfp4 needs for the same
  // reason q6k does: a group of 16 inside a 64-wide tile.
  MLX_MTL_CONST short group_steps = group_size < BCOLS ? 1 : group_size / BCOLS;
  MLX_MTL_CONST short scale_step = group_size < BCOLS ? BCOLS / group_size : 1;

  MLX_MTL_CONST short n_reads_per_scale = (n_reads * pack_factor) <= group_size
      ? n_reads
      : (group_size / pack_factor);
  MLX_MTL_CONST short n_steps_per_read = n_reads / n_reads_per_scale;

  // Each run must be a whole number of packs covering exactly one group, and
  // the runs must tile the thread's reads exactly. Anything else would silently
  // decode part of a group with the neighbouring group's scale.
  static_assert(
      n_reads_per_scale * n_steps_per_read == n_reads,
      "The reads per thread must split evenly into per-group runs.");
  static_assert(
      (n_reads_per_scale * pack_factor) <= group_size,
      "A per-group run must not reach past its group.");
  // A Tiled64 thread's reads stay inside one unit, so its source is contiguous.
  static_assert(
      !tiled || (n_reads * pack_factor) <= 32,
      "A Tiled64 thread's reads must stay inside one 32-code unit.");

  using scales_t = typename ConditionalType<
      tiled,
      KQScalesTiled<float, bits, super_ratio, has_min>,
      KQScales<float, bits, super_ratio, has_min>>::type;

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
            bi * src_ld / group_size + (bj * pack_factor) / group_size)) {
    static_assert(!tiled, "the Tiled64 loader takes the whole tensor");
  }

  // Tiled64: the whole weight, its companions, K, the tile's first row
  // (n0, a multiple of BROWS) and its first input (k0, a multiple of BCOLS).
  QuantizedBlockLoader(
      const device uint32_t* w,
      const device uint8_t* scales_,
      const device float16_t* biases_,
      const int K,
      const int n0,
      const int k0,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(K),
        tile_stride((BCOLS / 32) * kq_tiled_unit_stride<bits>()),
        group_step_cnt(0),
        group_stride(0),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(kq_tiled_row<bits>(w, n0 + bi, K / 32) +
            size_t((k0 + bj * pack_factor) / 32) *
                kq_tiled_unit_stride<bits>() +
            ((bj * pack_factor) % 32) * bits / 8),
        scales(
            scales_,
            biases_,
            K / group_size,
            n0 + bi,
            (k0 + bj * pack_factor) / group_size) {
    static_assert(tiled, "the row-major loader takes a pre-offset source");
  }

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    int k = 0;
    for (int i = 0; i < n_steps_per_read; i++) {
      float scale;
      float bias;
      scales.at(i, scale, bias);
      for (int j = 0; j < n_reads_per_scale; j++) {
        dequantize_to<T, pack_factor, bits, bits == 4 && !has_min>(
            src + k * bytes_per_pack, scale, bias, dst + k * pack_factor);
        k++;
      }
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

    int k = 0;
    for (int i = 0; i < n_steps_per_read; i++) {
      float scale;
      float bias;
      scales.at(i, scale, bias);
      for (int j = 0; j < n_reads_per_scale; j++) {
        dequantize_to<T, pack_factor, bits, bits == 4 && !has_min>(
            (device uint8_t*)(src + k * bytes_per_pack),
            scale,
            bias,
            dst + k * pack_factor);
        k++;
      }
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

template <
    typename T,
    const int group_size,
    const int bits,
    const int super_ratio,
    const bool has_min,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    bool tiled = false>
METAL_FUNC void kquant_qmm_t_nax_tgp_impl(
    const device uint32_t* w,
    KQScales<float, bits, super_ratio, has_min> scales,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    // Tiled64 only: the whole companions (`w` is then the whole weight too).
    const device uint8_t* scales_tiled = nullptr,
    const device float16_t* biases_tiled = nullptr) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");
  static_assert(BM % WM == 0, "BM should be divisible by WM");
  static_assert(BN % WN == 0, "BN should be divisible by WN");
  static_assert(
      (BM / WM) % 16 == 0, "Each SIMDgroup M tile must be divisible by 16");
  static_assert(
      (BN / WN) % 16 == 0, "Each SIMDgroup N tile must be divisible by 16");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

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
      has_min,
      tiled>;

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

  // Make the weight loader
  loader_w_t loader_w = [&]() {
    if constexpr (tiled) {
      return loader_w_t(
          w, scales_tiled, biases_tiled, K, y_col, 0, Ws, simd_gid, simd_lid);
    } else {
      return loader_w_t(wl, scales, K, Ws, simd_gid, simd_lid);
    }
  }();

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(int(SN), N - (y_col + tn));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_w.next();
      }

      // Store results to device memory
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
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
    const int group_size,
    const int bits,
    const int super_ratio,
    const bool has_min,
    const bool aligned_N,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    bool tiled = false>
[[kernel]] void kquant_qmm_t_nax(
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
  static_assert(!(tiled && batched), "Tiled64 weights are 2-D");
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

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
  kquant_qmm_t_nax_tgp_impl<
      T,
      group_size,
      bits,
      super_ratio,
      has_min,
      aligned_N,
      BM,
      BK,
      BN,
      WM,
      WN,
      tiled>(
      w,
      KQScales<float, bits, super_ratio, has_min>(scales, biases),
      x,
      y,
      Ws,
      K,
      N,
      M,
      tid,
      lid,
      simd_gid,
      simd_lid,
      scales,
      biases);
}
