// Paged Attention Metal Kernel Source
// Inline MSL source for use with MLX's get_library() pattern
// Auto-generated from crates/mlx-paged-attn/metal/*.metal

#pragma once

#include <string>

namespace paged_attn {

// ============================================================================
// Common utilities - bfloat16 and FP8 support
// ============================================================================

static const char* UTILS_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)
typedef bfloat bfloat16_t;
#else

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  uint32_t float_bits = as_type<uint32_t>(x);
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

struct _MLX_BFloat16 {
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype)    \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {              \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)         \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }                                                                            \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

#define bfloat_binop(_op_, _operator_)                                         \
  bfloat_binop_base(_op_, _operator_, _MLX_BFloat16, _MLX_BFloat16,            \
                    _MLX_BFloat16, float);                                     \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                  \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                   \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);       \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

#define bfloat_compop(__op__, __operator__)                                    \
  bfloat_binop_base(__op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16,  \
                    float);                                                    \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);               \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);                \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);             \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);            \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);             \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

typedef struct _MLX_BFloat16 bfloat16_t;

#endif
)";

// ============================================================================
// FP8 conversion helpers
// ============================================================================

static const char* FLOAT8_SOURCE = R"(
// FP8 E4M3 (bias = 7)
inline float fp8_e4m3_to_float(uchar v) {
  const uint s = v >> 7;
  const uint exp = (v >> 3) & 0xF;
  const uint man = v & 0x7;

  if (exp == 0) {
    if (man == 0)
      return s ? -0.f : 0.f;
    const float m = float(man) / 8.f;
    float val = ldexp(m, 1 - 7);
    return s ? -val : val;
  }

  if (exp == 0xF) {
    if (man != 0)
      return NAN;
    return s ? -INFINITY : INFINITY;
  }

  const float m = 1.f + float(man) / 8.f;
  float val = ldexp(m, int(exp) - 7);
  return s ? -val : val;
}

namespace detail {
template <int EXP_BITS, int MAN_BITS, int BIAS>
inline uchar fp32_to_fp8(float f) {
  const uint bits = as_type<uint>(f);
  const uint s = bits >> 31;
  const uint absbits = bits & 0x7FFFFFFF;

  if (absbits >= 0x7F800000u) {
    return uchar((s << 7) | (((1u << EXP_BITS) - 1u) << MAN_BITS) |
                 (absbits != 0x7F800000u));
  }

  int e = int((absbits >> 23) & 0xFF) - 127;
  uint m = absbits & 0x7FFFFFu;
  const int EXP_MAX = (1 << EXP_BITS) - 2;

  int e_fp8 = e + BIAS;
  if (e_fp8 >= 1 && e_fp8 <= EXP_MAX) {
    const int shift = 23 - MAN_BITS;
    uint mant = m >> shift;
    const uint lsb = mant & 1u;
    const uint round = (m >> (shift - 1)) & 1u;
    const uint sticky = (m & ((1u << (shift - 1)) - 1u)) != 0u;
    mant += (round & (sticky | lsb));
    if (mant >> MAN_BITS) {
      mant = 0;
      ++e_fp8;
      if (e_fp8 > EXP_MAX)
        return uchar((s << 7) | (((1u << EXP_BITS) - 1u) << MAN_BITS));
    }
    return uchar((s << 7) | (uint(e_fp8) << MAN_BITS) |
                 (mant & ((1u << MAN_BITS) - 1u)));
  }

  if (e_fp8 < 1 - MAN_BITS)
    return uchar(s << 7);

  int rshift = (1 - e_fp8) + (23 - MAN_BITS);
  uint mant = (0x800000u | m);
  uint rounded = (mant + (1u << (rshift - 1))) >> rshift;
  if (rounded == 0)
    return uchar(s << 7);

  return uchar((s << 7) | (rounded & ((1u << MAN_BITS) - 1u)));
}
}

inline uchar float_to_fp8_e4m3(float f) {
  return detail::fp32_to_fp8<4, 3, 7>(f);
}
)";

// ============================================================================
// reshape_and_cache kernel
// ============================================================================

static const char* RESHAPE_AND_CACHE_SOURCE = R"(
template <typename KV_T, typename CACHE_T>
inline CACHE_T to_cache(KV_T v) = delete;

template <> inline uchar to_cache<float, uchar>(float v) {
  return float_to_fp8_e4m3(v);
}

template <> inline uchar to_cache<bfloat16_t, uchar>(bfloat16_t v) {
  return float_to_fp8_e4m3((float)v);
}

template <> inline uchar to_cache<half, uchar>(half v) {
  return float_to_fp8_e4m3((float)v);
}

template <> inline float to_cache<float, float>(float v) { return v; }

template <> inline bfloat16_t to_cache<bfloat16_t, bfloat16_t>(bfloat16_t v) {
  return v;
}

template <> inline half to_cache<half, half>(half v) { return v; }

// FORKED: Replaced function_constant with template parameter for MLX compatibility
// Original: constant bool use_fp8_scales [[function_constant(10)]];

template <typename KV_T, typename CACHE_T, bool USE_FP8_SCALES>
[[kernel]] void reshape_and_cache(
    const device KV_T *__restrict__ key [[buffer(0)]],
    const device KV_T *__restrict__ value [[buffer(1)]],
    device CACHE_T *__restrict__ key_cache [[buffer(2)]],
    device CACHE_T *__restrict__ value_cache [[buffer(3)]],
    const device int64_t *__restrict__ slot_mapping [[buffer(4)]],
    const device float *__restrict__ k_scale [[buffer(5)]],
    const device float *__restrict__ v_scale [[buffer(6)]],
    device const int &key_stride [[buffer(7)]],
    device const int &value_stride [[buffer(8)]],
    device const int &num_heads [[buffer(9)]],
    device const int &head_size [[buffer(10)]],
    device const int &block_size [[buffer(11)]],
    device const int &x [[buffer(12)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int64_t token_idx = gid;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = tid; i < n; i += threads_per_threadgroup) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;

    // FORKED: Use if constexpr for compile-time branching (MLX compatible)
    if constexpr (USE_FP8_SCALES) {
      key_cache[tgt_key_idx] =
          to_cache<KV_T, CACHE_T>(KV_T((float)key[src_key_idx] / *k_scale));
      value_cache[tgt_value_idx] =
          to_cache<KV_T, CACHE_T>(KV_T((float)value[src_value_idx] / *v_scale));
    } else {
      key_cache[tgt_key_idx] = to_cache<KV_T, CACHE_T>(key[src_key_idx]);
      value_cache[tgt_value_idx] = to_cache<KV_T, CACHE_T>(value[src_value_idx]);
    }
  }
}

// FORKED: Updated macro to include USE_FP8_SCALES template parameter
#define instantiate_reshape_and_cache_inner(kv_type, cache_type, use_fp8, suffix) \
  template [[host_name("reshape_and_cache_kv_" #kv_type                        \
                       "_cache_" #cache_type suffix)]] [[kernel]] void         \
  reshape_and_cache<kv_type, cache_type, use_fp8>(                             \
      const device kv_type *__restrict__ key [[buffer(0)]],                    \
      const device kv_type *__restrict__ value [[buffer(1)]],                  \
      device cache_type *__restrict__ key_cache [[buffer(2)]],                 \
      device cache_type *__restrict__ value_cache [[buffer(3)]],               \
      const device int64_t *__restrict__ slot_mapping [[buffer(4)]],           \
      const device float *__restrict__ k_scale [[buffer(5)]],                  \
      const device float *__restrict__ v_scale [[buffer(6)]],                  \
      device const int &key_stride [[buffer(7)]],                              \
      device const int &value_stride [[buffer(8)]],                            \
      device const int &num_heads [[buffer(9)]],                               \
      device const int &head_size [[buffer(10)]],                              \
      device const int &block_size [[buffer(11)]],                             \
      device const int &x [[buffer(12)]],                                      \
      uint gid [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

// Generate both FP8 and non-FP8 variants for each type combination
#define instantiate_reshape_and_cache(kv_type, cache_type)                     \
  instantiate_reshape_and_cache_inner(kv_type, cache_type, false, "");         \
  instantiate_reshape_and_cache_inner(kv_type, cache_type, true, "_fp8");

instantiate_reshape_and_cache(float, float);
instantiate_reshape_and_cache(bfloat16_t, bfloat16_t);
instantiate_reshape_and_cache(half, half);
instantiate_reshape_and_cache(float, uchar);
instantiate_reshape_and_cache(bfloat16_t, uchar);
instantiate_reshape_and_cache(half, uchar);
)";

// ============================================================================
// copy_blocks kernel
// ============================================================================

static const char* COPY_BLOCKS_SOURCE = R"(
template <typename T>
[[kernel]] void copy_blocks(device T *key_cache [[buffer(0)]],
                            device T *value_cache [[buffer(1)]],
                            const device int64_t *block_mapping [[buffer(2)]],
                            device const int &numel_per_block,
                            uint tgid [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint threads_per_threadgroup
                            [[threads_per_threadgroup]]) {
  const int pair_idx = tgid;

  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;

  for (int i = tid; i < numel_per_block; i += threads_per_threadgroup) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }

  for (int i = tid; i < numel_per_block; i += threads_per_threadgroup) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

#define instantiate_copy_blocks(type)                                          \
  template [[host_name("copy_blocks_" #type)]] [[kernel]] void                 \
  copy_blocks<type>(device type * key_cache [[buffer(0)]],                     \
                    device type * value_cache [[buffer(1)]],                   \
                    const device int64_t *block_mapping [[buffer(2)]],         \
                    device const int &numel_per_block,                         \
                    uint tgid [[threadgroup_position_in_grid]],                \
                    uint tid [[thread_position_in_threadgroup]],               \
                    uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_copy_blocks(float);
instantiate_copy_blocks(bfloat16_t);
instantiate_copy_blocks(half);
instantiate_copy_blocks(uchar);
)";

// ============================================================================
// Build the complete kernel source string
// ============================================================================

inline std::string get_reshape_and_cache_source() {
    return std::string(UTILS_SOURCE) + FLOAT8_SOURCE + RESHAPE_AND_CACHE_SOURCE;
}

inline std::string get_copy_blocks_source() {
    return std::string(UTILS_SOURCE) + COPY_BLOCKS_SOURCE;
}

// ============================================================================
// paged_attention kernel - includes vector types, dot products, and attention
// ============================================================================

static const char* PAGED_ATTENTION_SOURCE = R"(
#include <metal_simdgroup>

// ========================================== Generic vector types

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE> struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T> struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B> inline Acc mul(A a, B b);

template <typename T> inline float sum(T v);

template <typename T> inline float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T> inline float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

// FP32 vector data types.
struct Float8_ {
  float4 x;
  float4 y;
};

template <> struct Vec<float, 1> { using Type = float; };
template <> struct Vec<float, 2> { using Type = float2; };
template <> struct Vec<float, 4> { using Type = float4; };
template <> struct Vec<float, 8> { using Type = Float8_; };

template <> struct FloatVec<float> { using Type = float; };
template <> struct FloatVec<float2> { using Type = float2; };
template <> struct FloatVec<float4> { using Type = float4; };
template <> struct FloatVec<Float8_> { using Type = Float8_; };

template <> inline float mul(float a, float b) { return a * b; }
template <> inline float2 mul(float2 a, float2 b) { return a * b; }
template <> inline float4 mul(float4 a, float4 b) { return a * b; }

template <> inline Float8_ mul(Float8_ a, Float8_ b) {
  Float8_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float sum(float a) { return a; }
template <> inline float sum(float2 a) { return a.x + a.y; }
template <> inline float sum(float4 a) { return a.x + a.y + a.z + a.w; }
template <> inline float sum(Float8_ a) { return sum(a.x) + sum(a.y); }

inline Float8_ fma(Float8_ a, Float8_ b, Float8_ c) {
  Float8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread float &dst, float src) { dst = src; }
inline void from_float(thread float2 &dst, float2 src) { dst = src; }
inline void from_float(thread float4 &dst, float4 src) { dst = src; }
inline void from_float(thread Float8_ &dst, Float8_ src) { dst = src; }

// BF16 vector data types.
struct Bfloat2_ {
  bfloat16_t x;
  bfloat16_t y;
};

struct Bfloat4_ {
  Bfloat2_ x;
  Bfloat2_ y;
};

struct Bfloat8_ {
  Bfloat4_ x;
  Bfloat4_ y;
};

template <> struct Vec<bfloat16_t, 1> { using Type = bfloat16_t; };
template <> struct Vec<bfloat16_t, 2> { using Type = Bfloat2_; };
template <> struct Vec<bfloat16_t, 4> { using Type = Bfloat4_; };
template <> struct Vec<bfloat16_t, 8> { using Type = Bfloat8_; };

template <> struct FloatVec<bfloat16_t> { using Type = float; };
template <> struct FloatVec<Bfloat2_> { using Type = float2; };
template <> struct FloatVec<Bfloat4_> { using Type = float4; };
template <> struct FloatVec<Bfloat8_> { using Type = Float8_; };

template <> inline float mul(bfloat16_t a, bfloat16_t b) { return (float)a * (float)b; }
template <> inline bfloat16_t mul(bfloat16_t a, bfloat16_t b) { return a * b; }

template <> inline float2 mul(Bfloat2_ a, Bfloat2_ b) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f;
}
template <> inline Bfloat2_ mul(Bfloat2_ a, Bfloat2_ b) {
  Bfloat2_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float4 mul(Bfloat4_ a, Bfloat4_ b) {
  float2 x = mul<float2, Bfloat2_, Bfloat2_>(a.x, b.x);
  float2 y = mul<float2, Bfloat2_, Bfloat2_>(a.y, b.y);
  float4 c;
  c.x = x.x; c.y = x.y; c.z = y.x; c.w = y.y;
  return c;
}
template <> inline Bfloat4_ mul(Bfloat4_ a, Bfloat4_ b) {
  Bfloat4_ c;
  c.x = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.x, b.x);
  c.y = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.y, b.y);
  return c;
}

template <> inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Float8_ c;
  c.x = mul<float4, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<float4, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}
template <> inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Bfloat8_ c;
  c.x = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}

template <> inline float sum(bfloat16_t a) { return (float)a; }
template <> inline float sum(Bfloat2_ a) { return (float)a.x + (float)a.y; }
template <> inline float sum(Bfloat4_ a) { return sum(a.x) + sum(a.y); }
template <> inline float sum(Bfloat8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(bfloat16_t a, bfloat16_t b, float c) { return (float)a * (float)b + c; }
inline bfloat16_t fma(bfloat16_t a, bfloat16_t b, bfloat16_t c) { return a * b + c; }

inline float2 fma(Bfloat2_ a, Bfloat2_ b, float2 c) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f + c;
}
inline Bfloat2_ fma(Bfloat2_ a, Bfloat2_ b, Bfloat2_ c) {
  Bfloat2_ res;
  res.x = a.x * b.x + c.x;
  res.y = a.y * b.y + c.y;
  return res;
}

inline float4 fma(Bfloat4_ a, Bfloat4_ b, float4 c) {
  float4 res;
  res.x = fma(a.x.x, b.x.x, c.x);
  res.y = fma(a.x.y, b.x.y, c.y);
  res.z = fma(a.y.x, b.y.x, c.z);
  res.w = fma(a.y.y, b.y.y, c.w);
  return res;
}
inline Bfloat4_ fma(Bfloat4_ a, Bfloat4_ b, Bfloat4_ c) {
  Bfloat4_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
  Float8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}
inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
  Bfloat8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread bfloat16_t &dst, float src) { dst = static_cast<bfloat16_t>(src); }
inline void from_float(thread Bfloat2_ &dst, float2 src) {
  dst.x = static_cast<bfloat16_t>(src.x);
  dst.y = static_cast<bfloat16_t>(src.y);
}
inline void from_float(thread Bfloat4_ &dst, float4 src) {
  dst.x.x = static_cast<bfloat16_t>(src.x);
  dst.x.y = static_cast<bfloat16_t>(src.y);
  dst.y.x = static_cast<bfloat16_t>(src.z);
  dst.y.y = static_cast<bfloat16_t>(src.w);
}
inline void from_float(thread Bfloat8_ &dst, Float8_ src) {
  Bfloat4_ x, y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// FP16 vector data types.
struct Half8_ {
  half4 x;
  half4 y;
};

template <> struct Vec<half, 1> { using Type = half; };
template <> struct Vec<half, 2> { using Type = half2; };
template <> struct Vec<half, 4> { using Type = half4; };
template <> struct Vec<half, 8> { using Type = Half8_; };

template <> struct FloatVec<half> { using Type = float; };
template <> struct FloatVec<half2> { using Type = float2; };
template <> struct FloatVec<half4> { using Type = float4; };
template <> struct FloatVec<Half8_> { using Type = Float8_; };

template <> inline float mul(half a, half b) { return (float)a * (float)b; }
template <> inline half mul(half a, half b) { return a * b; }

template <> inline float2 mul(half2 a, half2 b) { return (float2)a * (float2)b; }
template <> inline half2 mul(half2 a, half2 b) { return a * b; }

template <> inline float4 mul(half4 a, half4 b) { return (float4)a * (float4)b; }
template <> inline half4 mul(half4 a, half4 b) { return a * b; }

template <> inline Float8_ mul(Half8_ a, Half8_ b) {
  Float8_ c;
  c.x = mul<float4, half4, half4>(a.x, b.x);
  c.y = mul<float4, half4, half4>(a.y, b.y);
  return c;
}
template <> inline Half8_ mul(Half8_ a, Half8_ b) {
  Half8_ c;
  c.x = mul<half4, half4, half4>(a.x, b.x);
  c.y = mul<half4, half4, half4>(a.y, b.y);
  return c;
}

template <> inline float sum(half a) { return (float)a; }
template <> inline float sum(half2 a) { return (float)a.x + (float)a.y; }
template <> inline float sum(half4 a) { return a.x + a.y + a.z + a.w; }
template <> inline float sum(Half8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(half a, half b, float c) { return (float)a * (float)b + c; }
inline float2 fma(half2 a, half2 b, float2 c) { return (float2)a * (float2)b + c; }
inline float4 fma(half4 a, half4 b, float4 c) { return (float4)a * (float4)b + c; }

inline Float8_ fma(Half8_ a, Half8_ b, Float8_ c) {
  Float8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}
inline Half8_ fma(Half8_ a, Half8_ b, Half8_ c) {
  Half8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread half &dst, float src) { dst = static_cast<half>(src); }
inline void from_float(thread half2 &dst, float2 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
}
inline void from_float(thread half4 &dst, float4 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
  dst.z = static_cast<half>(src.z);
  dst.w = static_cast<half>(src.w);
}
inline void from_float(thread Half8_ &dst, Float8_ src) {
  half4 x, y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// ========================================== FP8 (uchar) vector data types.

struct Uchar8_ {
  uchar4 x;
  uchar4 y;
};

template <> struct Vec<uchar, 1> { using Type = uchar; };
template <> struct Vec<uchar, 2> { using Type = uchar2; };
template <> struct Vec<uchar, 4> { using Type = uchar4; };
template <> struct Vec<uchar, 8> { using Type = Uchar8_; };

template <typename T> inline constexpr bool is_uchar() { return false; }
template <> inline constexpr bool is_uchar<uchar>() { return true; }

template <typename Vec, typename Quant_vec>
inline Vec fp8_convert(const thread Quant_vec &, float scale) {
  static_assert(sizeof(Vec) == 0, "Missing fp8_convert specialisation");
}

inline float __dequant_single(uchar v, float scale) {
  return fp8_e4m3_to_float(v) * scale;
}

template <> inline float fp8_convert<float, uchar>(const thread uchar &in, float scale) {
  return __dequant_single(in, scale);
}
template <> inline half fp8_convert<half, uchar>(const thread uchar &in, float scale) {
  return half(__dequant_single(in, scale));
}
template <> inline bfloat16_t fp8_convert<bfloat16_t, uchar>(const thread uchar &in, float scale) {
  return bfloat16_t(__dequant_single(in, scale));
}

template <> inline float2 fp8_convert<float2, uchar2>(const thread uchar2 &in, float scale) {
  return float2(__dequant_single(in.x, scale), __dequant_single(in.y, scale));
}
template <> inline half2 fp8_convert<half2, uchar2>(const thread uchar2 &in, float scale) {
  half2 out;
  out.x = half(__dequant_single(in.x, scale));
  out.y = half(__dequant_single(in.y, scale));
  return out;
}
template <> inline Bfloat2_ fp8_convert<Bfloat2_, uchar2>(const thread uchar2 &in, float scale) {
  Bfloat2_ out;
  out.x = bfloat16_t(__dequant_single(in.x, scale));
  out.y = bfloat16_t(__dequant_single(in.y, scale));
  return out;
}

template <> inline float4 fp8_convert<float4, uchar4>(const thread uchar4 &in, float scale) {
  return float4(__dequant_single(in.x, scale), __dequant_single(in.y, scale),
                __dequant_single(in.z, scale), __dequant_single(in.w, scale));
}
template <> inline half4 fp8_convert<half4, uchar4>(const thread uchar4 &in, float scale) {
  half4 out;
  out.x = half(__dequant_single(in.x, scale));
  out.y = half(__dequant_single(in.y, scale));
  out.z = half(__dequant_single(in.z, scale));
  out.w = half(__dequant_single(in.w, scale));
  return out;
}
template <> inline Bfloat4_ fp8_convert<Bfloat4_, uchar4>(const thread uchar4 &in, float scale) {
  Bfloat4_ out;
  out.x.x = bfloat16_t(__dequant_single(in.x, scale));
  out.x.y = bfloat16_t(__dequant_single(in.y, scale));
  out.y.x = bfloat16_t(__dequant_single(in.z, scale));
  out.y.y = bfloat16_t(__dequant_single(in.w, scale));
  return out;
}

template <> inline Float8_ fp8_convert<Float8_, Uchar8_>(const thread Uchar8_ &in, float scale) {
  Float8_ out;
  out.x = float4(__dequant_single(in.x.x, scale), __dequant_single(in.x.y, scale),
                 __dequant_single(in.x.z, scale), __dequant_single(in.x.w, scale));
  out.y = float4(__dequant_single(in.y.x, scale), __dequant_single(in.y.y, scale),
                 __dequant_single(in.y.z, scale), __dequant_single(in.y.w, scale));
  return out;
}
template <> inline Half8_ fp8_convert<Half8_, Uchar8_>(const thread Uchar8_ &in, float scale) {
  Half8_ out;
  out.x = half4(half(__dequant_single(in.x.x, scale)), half(__dequant_single(in.x.y, scale)),
                half(__dequant_single(in.x.z, scale)), half(__dequant_single(in.x.w, scale)));
  out.y = half4(half(__dequant_single(in.y.x, scale)), half(__dequant_single(in.y.y, scale)),
                half(__dequant_single(in.y.z, scale)), half(__dequant_single(in.y.w, scale)));
  return out;
}
template <> inline Bfloat8_ fp8_convert<Bfloat8_, Uchar8_>(const thread Uchar8_ &in, float scale) {
  Bfloat8_ out;
  out.x.x.x = bfloat16_t(__dequant_single(in.x.x, scale));
  out.x.x.y = bfloat16_t(__dequant_single(in.x.y, scale));
  out.x.y.x = bfloat16_t(__dequant_single(in.x.z, scale));
  out.x.y.y = bfloat16_t(__dequant_single(in.x.w, scale));
  out.y.x.x = bfloat16_t(__dequant_single(in.y.x, scale));
  out.y.x.y = bfloat16_t(__dequant_single(in.y.y, scale));
  out.y.y.x = bfloat16_t(__dequant_single(in.y.z, scale));
  out.y.y.y = bfloat16_t(__dequant_single(in.y.w, scale));
  return out;
}

// ========================================== Dot product utilities

template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline float qk_dot_(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += simd_shuffle_xor(qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE> struct Qk_dot {
  template <typename Vec, int N>
  static inline float dot(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

// ========================================== Block sum utility

template <int NUM_WARPS, int NUM_SIMD_LANES>
inline float block_sum(threadgroup float *red_smem, float sum, uint simd_tid, uint simd_lid) {
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }
  if (simd_lid == 0) {
    red_smem[simd_tid] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lid < NUM_WARPS) {
    sum = red_smem[simd_lid];
  }
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }
  return simd_shuffle(sum, 0);
}

// ========================================== Paged Attention kernel

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

// FORKED: Replaced function_constant with template parameters for MLX compatibility
// Original:
//   constant bool use_partitioning [[function_constant(10)]];
//   constant bool use_alibi [[function_constant(20)]];
//   constant bool use_fp8_scales [[function_constant(30)]];
// Note: use_partitioning is now derived from PARTITION_SIZE > 0
// Note: use_fp8_scales is handled by is_uchar<CACHE_T>() for FP8 cache types

template <typename T, typename CACHE_T, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
          int NUM_SIMD_LANES, int PARTITION_SIZE = 0, bool USE_ALIBI = false>
[[kernel]] void paged_attention(
    device float *exp_sums [[buffer(0)]],
    device float *max_logits [[buffer(1)]],
    device T *out [[buffer(2)]],
    device const T *q [[buffer(3)]],
    device const CACHE_T *k_cache [[buffer(4)]],
    device const CACHE_T *v_cache [[buffer(5)]],
    const device float *__restrict__ k_scale [[buffer(6)]],
    const device float *__restrict__ v_scale [[buffer(7)]],
    const constant int &num_kv_heads [[buffer(8)]],
    const constant float &scale [[buffer(9)]],
    const constant float &softcapping [[buffer(10)]],
    device const uint32_t *block_tables [[buffer(11)]],
    device const uint32_t *context_lens [[buffer(12)]],
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    device const float *alibi_slopes [[buffer(14)]],
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_block_stride [[buffer(16)]],
    const constant int &kv_head_stride [[buffer(17)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int seq_idx = threadgroup_position_in_grid.y;
  const int partition_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int thread_idx = thread_position_in_threadgroup.x;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const uint32_t context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(NUM_SIMD_LANES / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, NUM_SIMD_LANES);
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  const int head_idx = threadgroup_position_in_grid.x;
  const int num_heads = threadgroups_per_grid.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  // FORKED: Use template parameter instead of function constant
  const float alibi_slope = !USE_ALIBI ? 0.f : alibi_slopes[head_idx];

  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using K_vec = typename Vec<T, VEC_SIZE>::Type;
  using Q_vec = typename Vec<T, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<CACHE_T, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  const device T *q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  threadgroup Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const device Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float *logits = reinterpret_cast<threadgroup float *>(shared_mem);
  threadgroup float red_smem[2 * NUM_WARPS];

  constexpr int x = 16 / sizeof(CACHE_T);
  float qk_max = -FLT_MAX;

  const device uint32_t *block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);

    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * NUM_SIMD_LANES) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const device CACHE_T *k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

        if constexpr (is_uchar<CACHE_T>()) {
          Quant_vec k_vec_quant = *reinterpret_cast<const device Quant_vec *>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8_convert<K_vec, Quant_vec>(k_vec_quant, *k_scale);
        } else {
          k_vecs[j] = *reinterpret_cast<const device K_vec *>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        }
      }

      float qk = scale * Qk_dot<T, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);

      if (softcapping != 1.0) {
        qk = precise::tanh(qk / softcapping) * softcapping;
      }

      // FORKED: Use template parameter instead of function constant
      if constexpr (USE_ALIBI) {
        if (alibi_slope != 0) {
          int position_offset = token_idx - int(context_len) + 1;
          float alibi_bias = alibi_slope * float(position_offset);
          qk += alibi_bias;
        }
      }

      if (thread_group_offset == 0) {
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        qk_max = mask ? qk_max : max(qk_max, qk);
      }
    }
  }

#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  qk_max = simd_shuffle(qk_max, 0);

  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = exp(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(&red_smem[NUM_WARPS], exp_sum, simd_tid, simd_lid);

  const float inv_sum = divide(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // FORKED: Removed redundant use_partitioning function constant check
  // USE_PARTITIONING is already derived from PARTITION_SIZE > 0
  if (USE_PARTITIONING && thread_idx == 0) {
    device float *max_logits_ptr =
        max_logits + seq_idx * num_heads * max_num_partitions +
        head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    device float *exp_sums_ptr = exp_sums +
                                 seq_idx * num_heads * max_num_partitions +
                                 head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  constexpr int V_VEC_SIZE = MIN(16 / sizeof(T), BLOCK_SIZE);
  using V_vec = typename Vec<T, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<T, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;
  using V_quant_vec = typename Vec<CACHE_T, V_VEC_SIZE>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = NUM_SIMD_LANES / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  T zero_value = 0;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    Float_L_vec logits_float_vec = *reinterpret_cast<threadgroup Float_L_vec *>(
        logits + token_idx - start_token_idx);
    from_float(logits_vec, logits_float_vec);

    const device CACHE_T *v_ptr = v_cache + physical_block_number * kv_block_stride +
                                  kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if constexpr (is_uchar<CACHE_T>()) {
          V_quant_vec v_quant_vec = *reinterpret_cast<const device V_quant_vec *>(v_ptr + offset);
          v_vec = fp8_convert<V_vec, V_quant_vec>(v_quant_vec, *v_scale);
        } else {
          v_vec = *reinterpret_cast<const device V_vec *>(v_ptr + offset);
        }

        if (block_idx == num_context_blocks - 1) {
          thread T *v_vec_ptr = reinterpret_cast<thread T *>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += simd_shuffle_xor(acc, mask);
    }
    accs[i] = acc;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float *out_smem = reinterpret_cast<threadgroup float *>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    if (warp_idx >= mid && warp_idx < i) {
      threadgroup float *dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_idx < mid) {
      const threadgroup float *src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (warp_idx == 0) {
    device T *out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        *(out_ptr + row_idx) = T(accs[i]);
      }
    }
  }
}

template <typename T, int HEAD_SIZE, int NUM_THREADS, int NUM_SIMD_LANES, int PARTITION_SIZE = 0>
[[kernel]] void paged_attention_v2_reduce(
    device T *out [[buffer(0)]],
    const device float *exp_sums [[buffer(1)]],
    const device float *max_logits [[buffer(2)]],
    const device T *tmp_out [[buffer(3)]],
    device uint32_t *context_lens [[buffer(4)]],
    const constant int &max_num_partitions [[buffer(5)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int num_heads = threadgroups_per_grid.x;
  const int head_idx = threadgroup_position_in_grid.x;
  const int seq_idx = threadgroup_position_in_grid.y;
  const uint32_t context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    device T *out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const device T *tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE; i += threads_per_threadgroup.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  threadgroup float red_smem[2 * NUM_WARPS];

  threadgroup float *shared_max_logits = reinterpret_cast<threadgroup float *>(shared_mem);
  const device float *max_logits_ptr =
      max_logits + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions; i += threads_per_threadgroup.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = max(max_logit, l);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  max_logit = simd_shuffle(max_logit, 0);

  threadgroup float *shared_exp_sums = reinterpret_cast<threadgroup float *>(
      shared_mem + sizeof(float) * num_partitions);
  const device float *exp_sums_ptr = exp_sums +
                                     seq_idx * num_heads * max_num_partitions +
                                     head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions; i += threads_per_threadgroup.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * exp(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  global_exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(&red_smem[NUM_WARPS], global_exp_sum, simd_tid, simd_lid);
  const float inv_global_exp_sum = divide(1.0f, global_exp_sum + 1e-6f);

  const device T *tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  device T *out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
    }
    out_ptr[i] = T(acc);
  }
}

// FORKED: Updated macro to include USE_ALIBI template parameter
#define instantiate_paged_attention_inner(type, cache_type, head_size, block_size, num_threads, num_simd_lanes, partition_size, use_alibi, alibi_suffix) \
  template [[host_name("paged_attention_" #type "_cache_" #cache_type "_hs" #head_size "_bs" #block_size "_nt" #num_threads "_nsl" #num_simd_lanes "_ps" #partition_size alibi_suffix)]] \
  [[kernel]] void paged_attention<type, cache_type, head_size, block_size, num_threads, num_simd_lanes, partition_size, use_alibi>( \
      device float *exp_sums [[buffer(0)]], device float *max_logits [[buffer(1)]], \
      device type *out [[buffer(2)]], device const type *q [[buffer(3)]], \
      device const cache_type *k_cache [[buffer(4)]], device const cache_type *v_cache [[buffer(5)]], \
      const device float *__restrict__ k_scale [[buffer(6)]], const device float *__restrict__ v_scale [[buffer(7)]], \
      const constant int &num_kv_heads [[buffer(8)]], const constant float &scale [[buffer(9)]], \
      const constant float &softcapping [[buffer(10)]], device const uint32_t *block_tables [[buffer(11)]], \
      device const uint32_t *context_lens [[buffer(12)]], const constant int &max_num_blocks_per_seq [[buffer(13)]], \
      device const float *alibi_slopes [[buffer(14)]], const constant int &q_stride [[buffer(15)]], \
      const constant int &kv_block_stride [[buffer(16)]], const constant int &kv_head_stride [[buffer(17)]], \
      threadgroup char *shared_mem [[threadgroup(0)]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint simd_tid [[simdgroup_index_in_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_v2_reduce_inner(type, head_size, num_threads, num_simd_lanes, partition_size) \
  template [[host_name("paged_attention_v2_reduce_" #type "_hs" #head_size "_nt" #num_threads "_nsl" #num_simd_lanes "_ps" #partition_size)]] \
  [[kernel]] void paged_attention_v2_reduce<type, head_size, num_threads, num_simd_lanes, partition_size>( \
      device type *out [[buffer(0)]], const device float *exp_sums [[buffer(1)]], \
      const device float *max_logits [[buffer(2)]], const device type *tmp_out [[buffer(3)]], \
      device uint32_t *context_lens [[buffer(4)]], const constant int &max_num_partitions [[buffer(5)]], \
      threadgroup char *shared_mem [[threadgroup(0)]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint3 threads_per_threadgroup [[threads_per_threadgroup]], uint simd_tid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]);

// FORKED: Generate both alibi and non-alibi variants for each configuration
#define instantiate_paged_attention_alibi_variants(type, cache_type, head_size, block_size, num_threads, num_simd_lanes, partition_size) \
  instantiate_paged_attention_inner(type, cache_type, head_size, block_size, num_threads, num_simd_lanes, partition_size, false, ""); \
  instantiate_paged_attention_inner(type, cache_type, head_size, block_size, num_threads, num_simd_lanes, partition_size, true, "_alibi");

#define instantiate_paged_attention_heads(type, cache_type, block_size, num_threads, num_simd_lanes, partition_size) \
  instantiate_paged_attention_alibi_variants(type, cache_type, 64, block_size, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_alibi_variants(type, cache_type, 128, block_size, num_threads, num_simd_lanes, partition_size);

#define instantiate_paged_attention_v2_reduce_heads(type, num_threads, num_simd_lanes, partition_size) \
  instantiate_paged_attention_v2_reduce_inner(type, 64, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 128, num_threads, num_simd_lanes, partition_size);

#define instantiate_paged_attention_block_size(type, cache_type, num_threads, num_simd_lanes, partition_size) \
  instantiate_paged_attention_heads(type, cache_type, 16, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_heads(type, cache_type, 32, num_threads, num_simd_lanes, partition_size);

#define instantiate_paged_attention_v1(type, cache_type, num_simd_lanes) \
  instantiate_paged_attention_block_size(type, cache_type, 256, num_simd_lanes, 0);

#define instantiate_paged_attention_v2(type, cache_type, num_simd_lanes) \
  instantiate_paged_attention_block_size(type, cache_type, 256, num_simd_lanes, 512);

#define instantiate_paged_attention_v2_reduce(type, num_simd_lanes) \
  instantiate_paged_attention_v2_reduce_heads(type, 256, num_simd_lanes, 512);

// Instantiate for common Qwen3 configs (head_size=64/128, block_size=16/32)
instantiate_paged_attention_v1(float, float, 32);
instantiate_paged_attention_v1(bfloat16_t, bfloat16_t, 32);
instantiate_paged_attention_v1(half, half, 32);

instantiate_paged_attention_v2(float, float, 32);
instantiate_paged_attention_v2(bfloat16_t, bfloat16_t, 32);
instantiate_paged_attention_v2(half, half, 32);

instantiate_paged_attention_v2_reduce(float, 32);
instantiate_paged_attention_v2_reduce(bfloat16_t, 32);
instantiate_paged_attention_v2_reduce(half, 32);
)";

// ============================================================================
// Build the complete kernel source string
// ============================================================================

inline std::string get_paged_attention_source() {
    return std::string(UTILS_SOURCE) + FLOAT8_SOURCE + PAGED_ATTENTION_SOURCE;
}

} // namespace paged_attn
