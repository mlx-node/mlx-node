// Adapted from Splash 1.6.1 (https://github.com/incoai/splash) d8af1054aab0f97de21d043e01aa98af3754f10e,
// runtime/metal/kernels/common/quant_formats.h. Apache-2.0; see THIRD_PARTY_NOTICES. Changed: the formats cut to
// the eight of metal/abi/QuantFormat.h (Q4_K IQ4_XS IQ4_NL Q5_K Q6_K Q3_K Q8_0 IQ3_S); the register-tile format
// list and the other formats' decoders removed.
// GGUF decoding follows ggml's Metal dequantize.h / mul_mm.metal (MIT, The ggml authors).
#pragma once
#include "metal/abi/QuantFormat.h"
#include "metal/abi/QuantTables.h"
#include <metal_stdlib>
using namespace metal;

// Per-format decoding of one (row, group of 32) of the MDGG0001 image, shared
// by the GEMM kernels, to the values of llama.cpp's dequantize.h (MIT notice
// in THIRD_PARTY_NOTICES). A group is read by chunk (metal/abi/QuantFormat.h):
// chunk c holds pairs p = 0..3; pairs 0, 1 are elements 4c..4c+3 of the first
// 16-group and pairs 2, 3 are elements 16+4c..16+4c+3 of the second. A format
// has its id, its sizes P0, P1, MetaBytes and MetaGroups from kQuantFormats,
// its code offset Zero (0 but for the linear formats with a zero point), the
// Group of elements one coefficient covers (32, 16 when the halves of a group
// have their own, or every group of a meta unit when they share one: MLX
// affine's 64 and 128, PQ2_0's 128, Q2_0's 64, FP8's and BF16's 256) and
//   load(plane0, plane1) -> Payload, loadMeta(meta) -> Meta
//   chunk(Payload, c) -> Chunk
//   loadChunk(plane0, plane1, c) -> Chunk, the same chunk read on its own
//   coef(Meta, j) -> QuantCoef of group j of the meta unit
//     (ScaleInChunk: coef(Meta, Chunk), the group's scale is in every chunk
//     of it rather than in the meta unit)
// and one element accessor, by Kind:
//   QuantLinear    codes(Chunk) -> uint4, pair p in component p with e0 at
//                  bit 0 and e1 at bit 16; value = s * (code - Zero) + m
//   QuantCodebook  indices(Chunk) -> uint, byte p indexes pair p in the
//                  format's pair table (quant_pair_table, of its value(i));
//                  value = s * table value
//   QuantInt8      values(Chunk) -> uint2, the int8 values of pairs 0, 1 (x)
//                  and 2, 3 (y); value = s * int8
//   (both with Scale, the narrowest type that holds s exactly)
//   QuantGrid      grid(Chunk) -> uint2, the grid magnitudes of pairs 0, 1
//                  (x) and 2, 3 (y); signs(Chunk) bit 2p + i negates element
//                  i of pair p; value = s * signed magnitude
//   QuantFloat8    values(Chunk) -> uint2, the E4M3 bytes of pairs 0, 1 (x)
//                  and 2, 3 (y); value = s * quant_e4m3 of the byte, which
//                  is 2^-8 times the E4M3 value (so s carries the 2^8)
//   QuantFloat16   values(Chunk) -> uint4, pair p in component p with e0 at
//                  bits 0..15 and e1 at 16..31, each of the format's 16-bit
//                  Value type (bfloat or half); value = s * the Value
enum QuantKind : ushort { QuantLinear, QuantCodebook, QuantInt8, QuantGrid, QuantFloat8, QuantFloat16 };

// The E4M3 bytes at bits 0..7 and 16..23 of `pair` (sign, four exponent bits of bias 7, three mantissa bits) as
// halves 2^-8 times their values, exactly: the byte's seven magnitude bits placed so that its exponent field is
// the low four of the half's, its subnormals the half's, as MLX's fp8.h also converts them. The NaN encodings
// (0x7F, 0xFF) come out finite; no weight or scale holds one.
inline half2 quant_e4m3_pair(uint pair) { return as_type<half2>(((pair & 0x007F007Fu) << 7) | ((pair & 0x00800080u) << 8)); }
// The four E4M3 bytes of a word, in byte order, as quant_e4m3_pair's halves.
inline half4 quant_e4m3_word(uint word) {
  const half2 even = quant_e4m3_pair(word & 0x00FF00FFu), odd = quant_e4m3_pair((word >> 8) & 0x00FF00FFu);
  return half4(even.x, odd.x, even.y, odd.y);
}

// s.x scales and m.x offsets pairs 0, 1, s.y and m.y pairs 2, 3 (equal where
// one coefficient covers 32 elements or more: the register tile then reads the
// x halves alone).
struct QuantCoef {
  float2 s;
  float2 m = float2(0.0f);
};

// The id, sizes and traits of format F of kind K with code offset Z, whose
// coefficients cover G elements (32 * MetaGroups where the groups of a meta
// unit share one) and, with InChunk, whose scales are in every chunk of a
// group.
#define QUANT_FORMAT(F, K, Z, G, InChunk)                                                                        \
  enum : uint { Id = F, P0 = kQuantFormats[F].plane0_bytes, P1 = kQuantFormats[F].plane1_bytes, MetaBytes = kQuantFormats[F].meta_bytes }; \
  enum : ushort { MetaGroups = kQuantFormats[F].meta_groups, Zero = Z, Group = G };                              \
  static constexpr constant QuantKind Kind = K;                                                                  \
  static constexpr constant bool ScaleInChunk = InChunk

// Entries of a codebook format's pair table: one per index byte.
constant constexpr uint kQuantPairTableEntries = 256;
// Fills the threadgroup table of format F's value pairs, as half2 (staged tiles) or bfloat2 (register tiles), entry
// b = (F::value(b & 15), F::value(b >> 4)) for the index byte b of a pair, when F is a codebook format; called by all
// threads of the threadgroup. quant_pair_entries writes the entries alone and returns whether F has a table, for a
// threadgroup that fills several tables before one barrier.
template <class F, class T>
inline bool quant_pair_entries(threadgroup T *table, uint thread_index, uint threads) {
  if constexpr (F::Kind == QuantCodebook)
    for (uint i = thread_index; i < kQuantPairTableEntries; i += threads)
      table[i] = T(float2(F::value(i & 15), F::value(i >> 4)));
  return F::Kind == QuantCodebook;
}
template <class F, class T>
inline void quant_pair_table(threadgroup T *table, uint thread_index, uint threads) {
  if constexpr (F::Kind == QuantCodebook) {
    quant_pair_entries<F>(table, thread_index, threads);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The pair words of a word of 4-bit codes: pair p's e0 at bits 4p, e1 at 16 + 4p.
inline uint4 quant_nibble_pairs(uint word) { return (uint4(word) >> uint4(0, 4, 8, 12)) & 0x000F000Fu; }
// Pair-word order of bit fields: the 1-bit fields of the two chunk bytes of a halfword (bit 2p + i of byte b)
// go to bits 8b + 2p (e0) and 16 + 8b + 2p (e1); the 2-bit fields of a chunk halfword (bits 4p + 2i) to bits
// 4p (e0) and 16 + 4p (e1).
inline uint quant_spread1(uint bits) { return (bits & 0x5555u) | ((bits & 0xAAAAu) << 15); }
inline uint quant_spread2(uint bits) { return (bits & 0x3333u) | ((bits & 0xCCCCu) << 14); }

// block_q4_K / block_q5_K header: s = d * sc and m = -dmin * mn with the 6-bit sc, mn of group j. The 12 scale
// bytes are hdr.y (0-3), hdr.z (4-7) and hdr.w (8-11), taken with shifts: the group index is not a compile-time
// constant, and indexing a thread-local byte array or vector by it costs ~8% of the eight-row staged kernel on
// Apple10.
inline QuantCoef quant_k4_coef(uint4 hdr, ushort j) {
  uint sc, m;
  if (j < 4) {
    const uint sh = 8u * j;
    sc = (hdr.y >> sh) & 63u;
    m = (hdr.z >> sh) & 63u;
  } else {
    const uint sh = 8u * (j - 4), w = hdr.w >> sh;
    sc = (w & 0xFu) | (((hdr.y >> sh) >> 6) & 3u) << 4;
    m = ((w >> 4) & 0xFu) | (((hdr.z >> sh) >> 6) & 3u) << 4;
  }
  const half d = as_type<half>(ushort(hdr.x & 0xFFFF)), dmin = as_type<half>(ushort(hdr.x >> 16));
  return {float2(float(d) * float(sc)), float2(-float(dmin) * float(m))};
}

// Scale bytes 2j, 2j + 1 of 16 bytes: bytes 2(j & 1), 2(j & 1) + 1 of word j >> 1, in bits 0..15 (shifts, as for
// Q4_K).
inline uint quant_scale_pair(packed_uint4 bytes, ushort j) {
  const uint word = j < 2 ? bytes.x : j < 4 ? bytes.y : j < 6 ? bytes.z : bytes.w;
  return word >> (16u * (j & 1));
}

// Q4_K: plane0 4-bit codes; meta the 16-byte block header.
struct FmtQ4K {
  QUANT_FORMAT(GGUF_FMT_Q4K, QuantLinear, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q); }
  static QuantCoef coef(Meta hdr, ushort j) { return quant_k4_coef(hdr, j); }
};
// Q5_K: plane0 low 4 bits, plane1 the fifth bits (byte c = chunk c); meta as Q4_K.
struct FmtQ5K {
  QUANT_FORMAT(GGUF_FMT_Q5K, QuantLinear, 0, 32, false);
  struct Payload { uint4 a; uint b; }; typedef uint2 Chunk; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint4 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static Chunk chunk(Payload w, ushort c) { return uint2(w.a[c], quant_spread1(w.b >> (16 * (c >> 1))) >> (8 * (c & 1))); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) {
    return uint2(*((device uint *)(p0 + 4 * c)), quant_spread1(p1[c]));
  }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q.x) | (((uint4(q.y) >> uint4(0, 2, 4, 6)) & 0x00010001u) << 4); }
  static QuantCoef coef(Meta hdr, ushort j) { return quant_k4_coef(hdr, j); }
};
// Q6_K: plane0 low 4 bits, plane1 the high 2 bits (halfword c = chunk c); meta 16 int8 scales, then half d.
// value = d * sc * (q - 32) with one scale per 16-group.
struct FmtQ6K {
  QUANT_FORMAT(GGUF_FMT_Q6K, QuantLinear, 32, 16, false);
  struct Payload { uint4 a; uint2 b; }; typedef uint2 Chunk; struct Meta { packed_uint4 sc; uint d; };
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint4 *)p0), *((device uint2 *)p1)}; }
  static Meta loadMeta(device uchar *m) { Meta r; r.sc = *((device packed_uint4 *)m); r.d = *((device uint *)(m + 16)); return r; }
  static Chunk chunk(Payload w, ushort c) { return uint2(w.a[c], quant_spread2(w.b[c >> 1] >> (16 * (c & 1)))); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) {
    return uint2(*((device uint *)(p0 + 4 * c)), quant_spread2(*((device ushort *)(p1 + 2 * c))));
  }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q.x) | (((uint4(q.y) >> uint4(0, 4, 8, 12)) & 0x00030003u) << 4); }
  static QuantCoef coef(Meta mt, ushort j) {
    const float d = float(as_type<half>(ushort(mt.d & 0xFFFF)));
    const uint pair = quant_scale_pair(mt.sc, j);   // the int8 scales of both 16-groups
    return {float2(d * float(as_type<char>(uchar(pair & 0xFFu))), d * float(as_type<char>(uchar(pair >> 8))))};
  }
};
// Q3_K: plane0 low 2 bits (halfword c = chunk c), plane1 the hmask bits (byte c = chunk c); meta half d, 2 zero
// bytes, the 12 packed scale bytes. value = d * (sc - 32) * (q - 4) with one scale per 16-group.
struct FmtQ3K {
  QUANT_FORMAT(GGUF_FMT_Q3K, QuantLinear, 4, 16, false);
  struct Payload { uint2 a; uint b; }; typedef uint2 Chunk; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static Chunk chunk(Payload w, ushort c) {
    return uint2(quant_spread2(w.a[c >> 1] >> (16 * (c & 1))), quant_spread1(w.b >> (16 * (c >> 1))) >> (8 * (c & 1)));
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) {
    return uint2(quant_spread2(*((device ushort *)(p0 + 2 * c))), quant_spread1(p1[c]));
  }
  static uint4 codes(Chunk q) { return ((uint4(q.x) >> uint4(0, 4, 8, 12)) & 0x00030003u) | (((uint4(q.y) >> uint4(0, 2, 4, 6)) & 0x00010001u) << 2); }
  static QuantCoef coef(Meta mt, ushort j) {
    const float d = float(as_type<half>(ushort(mt.x & 0xFFFF)));
    const uint t0 = mt.y, t1 = mt.z, t2 = mt.w;   // scales[0..3], [4..7], [8..11]
    uint aux;
    switch (j >> 1) {
      case 0: aux = (t0 & 0x0f0f0f0fu) | (((t2 >> 0) & 0x03030303u) << 4); break;
      case 1: aux = (t1 & 0x0f0f0f0fu) | (((t2 >> 2) & 0x03030303u) << 4); break;
      case 2: aux = ((t0 >> 4) & 0x0f0f0f0fu) | (((t2 >> 4) & 0x03030303u) << 4); break;
      default: aux = ((t1 >> 4) & 0x0f0f0f0fu) | (((t2 >> 6) & 0x03030303u) << 4); break;
    }
    const uint pair = aux >> (16u * (j & 1));   // 6-bit scales 2j, 2j + 1 (shifts, as for Q4_K)
    return {float2(d * float(int(pair & 0xFFu) - 32), d * float(int((pair >> 8) & 0xFFu) - 32))};
  }
};
// IQ4_XS: plane0 codebook indices; meta half d, scales_h, scales_l[4]. value = d * (ls - 32) * codebook.
struct FmtIQ4XS {
  QUANT_FORMAT(GGUF_FMT_IQ4XS, QuantCodebook, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uint2 Meta; typedef float Scale;
  static half value(uint i) { return half(kIQ4NLValues[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint2 *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort j) {
    const half d = as_type<half>(ushort(mt.x & 0xFFFF)); const uint sh = mt.x >> 16;
    const int ls = int((mt.y >> (4 * j)) & 0xF) | int(((sh >> (2 * j)) & 3) << 4);
    return {float2(float(d) * float(ls - 32))};
  }
};
// IQ4_NL: plane0 codebook indices; meta half d per group. value = d * codebook.
struct FmtIQ4NL {
  QUANT_FORMAT(GGUF_FMT_IQ4NL, QuantCodebook, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef ushort Meta; typedef half Scale;
  static half value(uint i) { return half(kIQ4NLValues[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort) { return {float2(float(as_type<half>(mt)))}; }
};
// Q8_0: plane0 int8 values (bytes 8c..8c+7 = chunk c); meta half d per group.
struct FmtQ80 {
  QUANT_FORMAT(GGUF_FMT_Q80, QuantInt8, 0, 32, false);
  struct Payload { uint4 a; uint4 b; }; typedef uint2 Chunk; typedef ushort Meta; typedef half Scale;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0), *((device uint4 *)(p0 + 16))}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { const uint4 h = c < 2 ? w.a : w.b; return (c & 1) ? h.zw : h.xy; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint2 *)(p0 + 8 * c)); }
  static uint2 values(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort) { return {float2(float(as_type<half>(mt)))}; }
};
// IQ3_S: plane0 word c = the 8-bit grid indices of pairs 0, 1 and 2, 3, chunk c's sign bits, the two ninth index
// bits and the group's 4-bit scale; meta half d per super-block. value = d * (1 + 2 * scale) * signed grid value.
struct FmtIQ3S {
  QUANT_FORMAT(GGUF_FMT_IQ3S, QuantGrid, 0, 32, true);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint2 grid(Chunk q) { return uint2(kIQ3SGrid[(q & 0xFF) | ((q >> 16) & 0x100)], kIQ3SGrid[((q >> 8) & 0xFF) | ((q >> 17) & 0x100)]); }
  static uint signs(Chunk q) { return (q >> 16) & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return {float2(float(as_type<half>(mt)) * float(1 + 2 * ((q >> 26) & 0xF)))}; }
};

#undef QUANT_FORMAT

// Every format as X(format type, kernel name token), the token being its kQuantFormats name, in id order.
#define QUANT_FORMATS(X)                                                                                         \
  X(FmtQ4K, q4k) X(FmtIQ4XS, iq4xs) X(FmtIQ4NL, iq4nl) X(FmtQ5K, q5k) X(FmtQ6K, q6k) X(FmtQ3K, q3k) X(FmtQ80, q80) \
  X(FmtIQ3S, iq3s)

// Runs body(F()) with the format type of run-time format id `format` (GGUF_FMT_*), for kernels whose tiles pick
// their tensor, and so its format, at run time. The branch is uniform in a threadgroup. The host passes known ids
// only; any other decodes as the first format, so every id runs one body (with a path that runs none, the staged
// fused kernel failed its test under shader validation).
#define QUANT_FORMAT_CASE(F, f) case F::Id: body(F()); break;
template <class Body>
inline void quant_format_switch(uint format, Body body) {
  switch (format) {
  default:
    QUANT_FORMATS(QUANT_FORMAT_CASE)
  }
}
#undef QUANT_FORMAT_CASE

#define QUANT_FORMAT_ONE(F, f) +1
static_assert(0 QUANT_FORMATS(QUANT_FORMAT_ONE) == GGUF_FMT_COUNT, "QUANT_FORMATS lists every format");
#undef QUANT_FORMAT_ONE
template <uint N> constexpr bool quant_format_named(uint id, const constant char (&token)[N]) {
  for (uint i = 0; i < N; ++i)
    if (kQuantFormats[id].name[i] != token[i]) return false;
  return true;
}
#define QUANT_FORMAT_NAME(F, f) static_assert(quant_format_named(F::Id, #f), #f " is not its kQuantFormats name");
QUANT_FORMATS(QUANT_FORMAT_NAME)
#undef QUANT_FORMAT_NAME
