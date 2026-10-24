// GGUF raw blocks -> the runtime packed arrays (`.weight` uint32, `.scales`
// uint8 / int8 / float16, `.biases` float16 when the format has them),
// row-major or Tiled64, for one band of rows
// of a tensor. build.rs prebuilds this file into paged_attn.metallib; the host
// names below must match the kernel table in mlx_gguf_repack.cpp, which
// gguf_repack_metallib_names checks both ways.
//
// The staging bytes of the band start at `src_off`; output positions are
// global (band row `row0`, Tiled64 tile `tile0 = row0 / 64`), so bands of one
// tensor written in any order give the same arrays as one whole-tensor band.
// Every output byte must equal the CPU loader: gguf_kquant.rs KQuantRepacker
// (then kquant_tile_rows for Tiled64) for the K-quants, gguf.rs
// load_quantized_tensor for Q4_0, Q8_0 and MXFP4.

#include <metal_stdlib>

using namespace metal;

// Mirrors `GgufRepackParams` in mlx_gguf_repack.cpp.
struct GgufRepackParams {
  ulong src_off;
  uint nb;    // blocks in the band (rows * kb)
  uint kb;    // 256-value blocks per row
  uint ku;    // 32-value units per row (8 * kb)
  uint row0;  // first row of the band in the tensor
  uint rows;  // rows in the band
  uint tile0; // row0 / 64 (Tiled64)
};

// Low nibble of each of 8 bytes -> 8 packed 4-bit codes.
static inline uint nib8(uint a, uint b) {
  uint x = a & 0x0F0F0F0Fu;
  x = (x | (x >> 4)) & 0x00FF00FFu;
  x = (x | (x >> 8)) & 0x0000FFFFu;
  uint y = b & 0x0F0F0F0Fu;
  y = (y | (y >> 4)) & 0x00FF00FFu;
  y = (y | (y >> 8)) & 0x0000FFFFu;
  return x | (y << 16);
}

// Bytes a0, b0, a1, b1.
static inline uint il2(uint a, uint b) {
  return (a & 0xFFu) | ((b & 0xFFu) << 8) | ((a & 0xFF00u) << 8) |
      ((b & 0xFF00u) << 16);
}

// Q4_K / Q5_K scales: q0, q1, q2 = block words 1..3 (scale bytes 0..11) -> 16
// bytes (sc0, m0, ..., sc7, m7).
static inline uint4 k4_scales(uint q0, uint q1, uint q2) {
  uint sl = q0 & 0x3F3F3F3Fu, ml = q1 & 0x3F3F3F3Fu;
  uint sh = (q2 & 0x0F0F0F0Fu) | ((q0 >> 2) & 0x30303030u);
  uint mh = ((q2 >> 4) & 0x0F0F0F0Fu) | ((q1 >> 2) & 0x30303030u);
  return uint4(il2(sl, ml), il2(sl >> 16, ml >> 16), il2(sh, mh),
               il2(sh >> 16, mh >> 16));
}

// Q4_K unit u (values 32u..32u+31) from the 8 qs words of half u/2 -> 4 words.
static inline uint4 q4k_unit_from(uint4 a, uint4 b, uint u) {
  uint s = (u & 1u) * 4u;
  return uint4(nib8(a.x >> s, a.y >> s), nib8(a.z >> s, a.w >> s),
               nib8(b.x >> s, b.y >> s), nib8(b.z >> s, b.w >> s));
}

// Q5_K: 8 codes (lanes 8c..8c+7 of unit u) -> 40-bit value, code i at bits 5i.
static inline ulong q5k_8codes(uint qa, uint qb, uint ha, uint hb, uint u) {
  uint s = (u & 1u) * 4u;
  uint n = nib8(qa >> s, qb >> s);
  uint hm = ((ha >> u) & 0x01010101u);
  uint hn = ((hb >> u) & 0x01010101u);
  uint h = (hm & 1u) | ((hm >> 7) & 2u) | ((hm >> 14) & 4u) |
      ((hm >> 21) & 8u) | ((hn & 1u) << 4) | ((hn >> 3) & 0x20u) |
      ((hn >> 10) & 0x40u) | ((hn >> 17) & 0x80u);
  ulong t = 0;
#pragma unroll
  for (uint i = 0; i < 8; i++) {
    t |= (ulong)(((n >> (4u * i)) & 0xFu) | (((h >> i) & 1u) << 4))
        << (5u * i);
  }
  return t;
}

// Q5_K unit u from the block (uint4-aligned) -> 5 words.
static inline void q5k_unit(const device uint4* b4, uint u, thread uint* o) {
  uint hf = u >> 1;
  uint4 qs0 = b4[3 + 2 * hf], qs1 = b4[4 + 2 * hf];
  uint4 qh0 = b4[1], qh1 = b4[2];
  ulong x0 = q5k_8codes(qs0.x, qs0.y, qh0.x, qh0.y, u);
  ulong x1 = q5k_8codes(qs0.z, qs0.w, qh0.z, qh0.w, u);
  ulong x2 = q5k_8codes(qs1.x, qs1.y, qh1.x, qh1.y, u);
  ulong x3 = q5k_8codes(qs1.z, qs1.w, qh1.z, qh1.w, u);
  o[0] = (uint)x0;
  o[1] = (uint)(x0 >> 32) | (uint)(x1 << 8);
  o[2] = (uint)(x1 >> 24) | (uint)(x2 << 16);
  o[3] = (uint)(x2 >> 16) | (uint)(x3 << 24);
  o[4] = (uint)(x3 >> 8);
}

// ---------------------------------------------------------------- Q4_K
// 144-byte blocks (9 uint4). One thread per 32-value unit: grid = nb * 8.

[[kernel]] void mlx_node_gguf_repack_q4k(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  uint b = g >> 3, u = g & 7u;
  const device uint4* b4 =
      (const device uint4*)(staging + p.src_off) + (ulong)b * 9u;
  uint4 a = b4[1 + 2 * (u >> 1)], c = b4[2 + 2 * (u >> 1)];
  ulong gb = (ulong)p.row0 * p.kb + b;
  ((device uint4*)weight)[gb * 8u + u] = q4k_unit_from(a, c, u);
  if (u == 0) {
    uint4 h = b4[0];
    ((device uint4*)scales)[gb] = k4_scales(h.y, h.z, h.w);
    ((device uint*)biases)[gb] = h.x;
  }
}

// Tiled64, threads in tile order: g -> rr = g % 64, U = (g / 64) % ku,
// T = g / (64 * ku) (band-local tile); writes are contiguous.
[[kernel]] void mlx_node_gguf_repack_q4k_t64(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  uint rr = g & 63u, rest = g >> 6, U = rest % p.ku, T = rest / p.ku;
  uint n = T * 64u + rr, s = U >> 3, u = U & 7u;
  uint b = n * p.kb + s;
  const device uint4* b4 =
      (const device uint4*)(staging + p.src_off) + (ulong)b * 9u;
  uint4 a = b4[1 + 2 * (u >> 1)], c = b4[2 + 2 * (u >> 1)];
  ulong tg = (ulong)p.tile0 + T;
  ((device uint4*)weight)[(ulong)p.tile0 * p.ku * 64u + g] =
      q4k_unit_from(a, c, u);
  if (u == 0) {
    uint4 h = b4[0];
    ulong o = (tg * p.kb + s) * 64u + rr;
    ((device uint4*)scales)[o] = k4_scales(h.y, h.z, h.w);
    ((device uint*)biases)[o] = h.x;
  }
}

// ---------------------------------------------------------------- Q5_K
// 176-byte blocks (11 uint4). One thread per 32-value unit: grid = nb * 8.

[[kernel]] void mlx_node_gguf_repack_q5k(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  uint b = g >> 3, u = g & 7u;
  const device uint4* b4 =
      (const device uint4*)(staging + p.src_off) + (ulong)b * 11u;
  uint o[5];
  q5k_unit(b4, u, o);
  ulong gb = (ulong)p.row0 * p.kb + b;
  device uint* w = weight + (gb * 8u + u) * 5u;
  w[0] = o[0];
  w[1] = o[1];
  w[2] = o[2];
  w[3] = o[3];
  w[4] = o[4];
  if (u == 0) {
    uint4 h = b4[0];
    ((device uint4*)scales)[gb] = k4_scales(h.y, h.z, h.w);
    ((device uint*)biases)[gb] = h.x;
  }
}

[[kernel]] void mlx_node_gguf_repack_q5k_t64(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  uint rr = g & 63u, rest = g >> 6, U = rest % p.ku, T = rest / p.ku;
  uint n = T * 64u + rr, s = U >> 3, u = U & 7u;
  uint b = n * p.kb + s;
  const device uint4* b4 =
      (const device uint4*)(staging + p.src_off) + (ulong)b * 11u;
  uint o[5];
  q5k_unit(b4, u, o);
  ulong tg = (ulong)p.tile0 + T;
  device uint* w = weight + ((ulong)p.tile0 * p.ku * 64u + g) * 5u;
  w[0] = o[0];
  w[1] = o[1];
  w[2] = o[2];
  w[3] = o[3];
  w[4] = o[4];
  if (u == 0) {
    uint4 h = b4[0];
    ulong oo = (tg * p.kb + s) * 64u + rr;
    ((device uint4*)scales)[oo] = k4_scales(h.y, h.z, h.w);
    ((device uint*)biases)[oo] = h.x;
  }
}

// ---------------------------------------------------------------- 2-byte aligned blocks
// Q6_K (210 B) and Q3_K (110 B) blocks start at byte offsets 0 or 2 mod 4.
// `r` is the word-aligned base at or below the band start and `B` a block's
// byte offset from it. `ldu` reads one extra word only when misaligned, and
// that word ends inside the block; `ldh` is a 2-byte load. No read goes past
// the band's last byte.

static inline uint fun16(uint lo, uint hi, uint sh) {
  return (uint)(((((ulong)hi) << 32) | (ulong)lo) >> sh);
}

// N words starting at the 2-aligned byte offset `off`.
template <int N>
static inline void ldu(const device uint* r, uint off, thread uint* o) {
  uint i = off >> 2, sh = (off & 2u) << 3;
  uint w[N + 1];
#pragma unroll
  for (int k = 0; k < N; k++) w[k] = r[i + k];
  w[N] = sh ? r[i + N] : 0u;
#pragma unroll
  for (int k = 0; k < N; k++) o[k] = fun16(w[k], w[k + 1], sh);
}

// f16 bits at the 2-aligned byte offset `off`.
static inline ushort ldh(const device uint* r, uint off) {
  return ((const device ushort*)r)[off >> 1];
}

static inline const device uint* word_base(const device uchar* staging,
                                           ulong src_off) {
  return (const device uint*)(staging + (src_off & ~(ulong)3));
}

// 4 bytes (each 0..63) -> 24-bit LSB-first stream.
static inline uint pack6x4(uint c) {
  uint x = (c & 0x003F003Fu) | ((c & 0x3F003F00u) >> 2);
  return (x & 0xFFFu) | ((x >> 4) & 0xFFF000u);
}

// 4 bytes (each 0..7) -> 12 bits.
static inline uint pack3x4(uint c) {
  uint x = (c & 0x00070007u) | ((c & 0x07000700u) >> 5);
  return (x & 0x3Fu) | ((x >> 10) & 0xFC0u);
}

// Per byte: v (0..63) - 32 as int8.
static inline uint sub32b(uint v) {
  uint m = ~v & 0x20202020u;
  return (v ^ 0x20202020u) | (m << 1) | (m << 2);
}

// Q6_K: ql @0 (128), qh @128 (64), scales @192 (16), d @208. Unit u -> 6 words.
static inline void q6k_unit(const device uint* r, uint B, uint u,
                            thread uint* o) {
  uint hf = u >> 2, gi = u & 3u;
  uint q[8], h[8];
  ldu<8>(r, B + hf * 64u + (gi & 1u) * 32u, q);
  ldu<8>(r, B + 128u + hf * 32u, h);
  uint ns = (gi >> 1) * 4u, hs = gi * 2u;
  uint g[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    g[k] = pack6x4(((q[k] >> ns) & 0x0F0F0F0Fu) |
                   (((h[k] >> hs) & 0x03030303u) << 4));
  }
  o[0] = g[0] | (g[1] << 24);
  o[1] = (g[1] >> 8) | (g[2] << 16);
  o[2] = (g[2] >> 16) | (g[3] << 8);
  o[3] = g[4] | (g[5] << 24);
  o[4] = (g[5] >> 8) | (g[6] << 16);
  o[5] = (g[6] >> 16) | (g[7] << 8);
}

static inline uint4 q6k_scales(const device uint* r, uint B) {
  uint s[4];
  ldu<4>(r, B + 192u, s);
  return uint4(s[0], s[1], s[2], s[3]);
}

// Q3_K: hmask @0 (32), qs @32 (64), scales @96 (12), d @108.
static inline uint4 q3k_scales(const device uint* r, uint B) {
  uint s[3];
  ldu<3>(r, B + 96u, s);
  uint m = 0x0F0F0F0Fu, t = 0x03030303u;
  return uint4(sub32b((s[0] & m) | ((s[2] & t) << 4)),
               sub32b((s[1] & m) | (((s[2] >> 2) & t) << 4)),
               sub32b(((s[0] >> 4) & m) | (((s[2] >> 4) & t) << 4)),
               sub32b(((s[1] >> 4) & m) | (((s[2] >> 6) & t) << 4)));
}

// Q3_K half block hf (the 4 units that share their qs/hmask bytes) -> 12
// words, unit-major.
static inline void q3k_half(const device uint* r, uint B, uint hf,
                            thread uint* o) {
  uint q[8], h[8];
  ldu<8>(r, B + 32u + hf * 32u, q);
  ldu<8>(r, B, h);
#pragma unroll
  for (uint gr = 0; gr < 4; gr++) {
    uint qs = 2u * gr, hs = 4u * hf + gr;
    uint g[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      g[k] = pack3x4(((q[k] >> qs) & 0x03030303u) |
                     (((h[k] >> hs) & 0x01010101u) << 2));
    }
    thread uint* w = o + 3u * gr;
    w[0] = g[0] | (g[1] << 12) | (g[2] << 24);
    w[1] = (g[2] >> 8) | (g[3] << 4) | (g[4] << 16) | (g[5] << 28);
    w[2] = (g[5] >> 4) | (g[6] << 8) | (g[7] << 20);
  }
}

// IQ4_XS: d @0, scales_h u16 @2, scales_l @4 (4), qs @8 (128); 34 words,
// 8-byte aligned. Unit u -> 4 words.
static inline uint4 iq4xs_unit(const device uint* r, uint b, uint u) {
  const device uint2* p = (const device uint2*)(r + b * 34u + 2u + 4u * u);
  uint2 x = p[0], y = p[1];
  return uint4(nib8(x.x, x.y), nib8(y.x, y.y), nib8(x.x >> 4, x.y >> 4),
               nib8(y.x >> 4, y.y >> 4));
}

// w0 = d | scales_h << 16, w1 = scales_l -> 8 int8 sub-scales.
static inline uint2 iq4xs_scales(uint w0, uint w1) {
  uint le = w1 & 0x0F0F0F0Fu, lo = (w1 >> 4) & 0x0F0F0F0Fu;
  uint H = w0 >> 16;
  uint x = H & 0xFFu;
  x = (x | (x << 12)) & 0x000F000Fu;
  x = (x | (x << 6)) & 0x03030303u;
  uint y = (H >> 8) & 0xFFu;
  y = (y | (y << 12)) & 0x000F000Fu;
  y = (y | (y << 6)) & 0x03030303u;
  return uint2(sub32b(il2(le, lo) | (x << 4)),
               sub32b(il2(le >> 16, lo >> 16) | (y << 4)));
}

// ---------------------------------------------------------------- Q6_K
// 210-byte blocks. One thread per 32-value unit: grid = nb * 8.

[[kernel]] void mlx_node_gguf_repack_q6k(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  const device uint* r = word_base(staging, p.src_off);
  uint b = g >> 3, u = g & 7u;
  uint B = (uint)(p.src_off & 3ul) + b * 210u;
  uint o[6];
  q6k_unit(r, B, u, o);
  ulong gb = (ulong)p.row0 * p.kb + b;
  device uint2* w2 = (device uint2*)(weight + (gb * 8u + u) * 6u);
  w2[0] = uint2(o[0], o[1]);
  w2[1] = uint2(o[2], o[3]);
  w2[2] = uint2(o[4], o[5]);
  if (u == 0) {
    ((device uint4*)scales)[gb] = q6k_scales(r, B);
    ((device ushort*)biases)[gb] = ldh(r, B + 208u);
  }
}

[[kernel]] void mlx_node_gguf_repack_q6k_t64(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  const device uint* r = word_base(staging, p.src_off);
  uint rr = g & 63u, rest = g >> 6, U = rest % p.ku, T = rest / p.ku;
  uint n = T * 64u + rr, s = U >> 3, u = U & 7u;
  uint b = n * p.kb + s;
  uint B = (uint)(p.src_off & 3ul) + b * 210u;
  uint o[6];
  q6k_unit(r, B, u, o);
  ulong tg = (ulong)p.tile0 + T;
  device uint2* w2 =
      (device uint2*)(weight + ((ulong)p.tile0 * p.ku * 64u + g) * 6u);
  w2[0] = uint2(o[0], o[1]);
  w2[1] = uint2(o[2], o[3]);
  w2[2] = uint2(o[4], o[5]);
  if (u == 0) {
    ulong oo = (tg * p.kb + s) * 64u + rr;
    ((device uint4*)scales)[oo] = q6k_scales(r, B);
    ((device ushort*)biases)[oo] = ldh(r, B + 208u);
  }
}

// ---------------------------------------------------------------- IQ4_XS
// 136-byte blocks (8-byte aligned). One thread per 32-value unit:
// grid = nb * 8.

[[kernel]] void mlx_node_gguf_repack_iq4xs(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  const device uint* r = (const device uint*)(staging + p.src_off);
  uint b = g >> 3, u = g & 7u;
  ulong gb = (ulong)p.row0 * p.kb + b;
  ((device uint4*)weight)[gb * 8u + u] = iq4xs_unit(r, b, u);
  if (u == 0) {
    uint2 h = ((const device uint2*)r)[b * 17u];
    ((device uint2*)scales)[gb] = iq4xs_scales(h.x, h.y);
    ((device ushort*)biases)[gb] = (ushort)(h.x & 0xFFFFu);
  }
}

[[kernel]] void mlx_node_gguf_repack_iq4xs_t64(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  const device uint* r = (const device uint*)(staging + p.src_off);
  uint rr = g & 63u, rest = g >> 6, U = rest % p.ku, T = rest / p.ku;
  uint n = T * 64u + rr, s = U >> 3, u = U & 7u;
  uint b = n * p.kb + s;
  ulong tg = (ulong)p.tile0 + T;
  ((device uint4*)weight)[(ulong)p.tile0 * p.ku * 64u + g] =
      iq4xs_unit(r, b, u);
  if (u == 0) {
    ulong oo = (tg * p.kb + s) * 64u + rr;
    uint2 h = ((const device uint2*)r)[b * 17u];
    ((device uint2*)scales)[oo] = iq4xs_scales(h.x, h.y);
    ((device ushort*)biases)[oo] = (ushort)(h.x & 0xFFFFu);
  }
}

// ---------------------------------------------------------------- Q3_K
// 110-byte blocks. One thread per half block (4 units): grid = nb * 2.

[[kernel]] void mlx_node_gguf_repack_q3k(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 2u) return;
  const device uint* r = word_base(staging, p.src_off);
  uint b = g >> 1, hf = g & 1u;
  uint B = (uint)(p.src_off & 3ul) + b * 110u;
  uint o[12];
  q3k_half(r, B, hf, o);
  ulong gb = (ulong)p.row0 * p.kb + b;
  device uint4* w4 = (device uint4*)weight + (gb * 2u + hf) * 3u;
  w4[0] = uint4(o[0], o[1], o[2], o[3]);
  w4[1] = uint4(o[4], o[5], o[6], o[7]);
  w4[2] = uint4(o[8], o[9], o[10], o[11]);
  if (hf == 0) {
    ((device uint4*)scales)[gb] = q3k_scales(r, B);
    ((device ushort*)biases)[gb] = ldh(r, B + 108u);
  }
}

// Tiled64, threads in tile order over half blocks: g -> rr = g % 64,
// H = (g / 64) % (2 * kb), T = g / (128 * kb).
[[kernel]] void mlx_node_gguf_repack_q3k_t64(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 2u) return;
  const device uint* r = word_base(staging, p.src_off);
  uint rr = g & 63u, rest = g >> 6, H = rest % (p.kb * 2u),
       T = rest / (p.kb * 2u);
  uint s = H >> 1, hf = H & 1u, n = T * 64u + rr, b = n * p.kb + s;
  uint B = (uint)(p.src_off & 3ul) + b * 110u;
  uint o[12];
  q3k_half(r, B, hf, o);
  ulong tg = (ulong)p.tile0 + T;
#pragma unroll
  for (uint gr = 0; gr < 4; gr++) {
    device uint* w =
        weight + ((tg * p.ku + s * 8u + hf * 4u + gr) * 64u + rr) * 3u;
    w[0] = o[3u * gr];
    w[1] = o[3u * gr + 1u];
    w[2] = o[3u * gr + 2u];
  }
  if (hf == 0) {
    ulong oo = (tg * p.kb + s) * 64u + rr;
    ((device uint4*)scales)[oo] = q3k_scales(r, B);
    ((device ushort*)biases)[oo] = ldh(r, B + 108u);
  }
}

// ---------------------------------------------------------------- 32-value blocks
// Q4_0 / IQ4_NL (18 B), Q8_0 (34 B) and MXFP4 (17 B) blocks cover 32 values:
// one thread per block, kb = ku = blocks per row.

// `ldu` for a run that ends at the block's last byte: when misaligned the
// extra word is read as a 2-byte load, so nothing past the run is read.
template <int N>
static inline void ldu_end(const device uint* r, uint off, thread uint* o) {
  uint i = off >> 2, sh = (off & 2u) << 3;
  uint w[N + 1];
#pragma unroll
  for (int k = 0; k < N; k++) w[k] = r[i + k];
  w[N] = sh ? (uint)((const device ushort*)(r + i + N))[0] : 0u;
#pragma unroll
  for (int k = 0; k < N; k++) o[k] = fun16(w[k], w[k + 1], sh);
}

// 16 nibble bytes q0..q3 -> 4 words: codes 0..15 the low nibbles, 16..31 the
// high nibbles.
static inline uint4 nib32(uint q0, uint q1, uint q2, uint q3) {
  return uint4(nib8(q0, q1), nib8(q2, q3), nib8(q0 >> 4, q1 >> 4),
               nib8(q2 >> 4, q3 >> 4));
}

// IQ4_NL: d @0, qs @2 (16). `.scales` int8 1, `.biases` d.
static inline uint4 iq4nl_block(const device uint* r, uint B) {
  uint q[4];
  ldu_end<4>(r, B + 2u, q);
  return nib32(q[0], q[1], q[2], q[3]);
}

// Q4_0: d @0, qs @2 (16). `.weight` codes match IQ4_NL, `.scales` = d
// (float16), no `.biases`; the loader derives the symmetric offset later.
[[kernel]] void mlx_node_gguf_repack_q4_0(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device half* scales [[buffer(2)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb) return;
  const device uint* r = word_base(staging, p.src_off);
  uint B = (uint)(p.src_off & 3ul) + g * 18u;
  ulong gb = (ulong)p.row0 * p.kb + g;
  uint4 w = iq4nl_block(r, B);
  ((device uint4*)weight)[gb] = w;
  ((device ushort*)scales)[gb] = ldh(r, B);
}

[[kernel]] void mlx_node_gguf_repack_iq4nl(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device char* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb) return;
  const device uint* r = word_base(staging, p.src_off);
  uint B = (uint)(p.src_off & 3ul) + g * 18u;
  ulong gb = (ulong)p.row0 * p.kb + g;
  ((device uint4*)weight)[gb] = iq4nl_block(r, B);
  scales[gb] = 1;
  ((device ushort*)biases)[gb] = ldh(r, B);
}

[[kernel]] void mlx_node_gguf_repack_iq4nl_t64(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device char* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb) return;
  const device uint* r = word_base(staging, p.src_off);
  uint rr = g & 63u, rest = g >> 6, U = rest % p.ku, T = rest / p.ku;
  uint b = (T * 64u + rr) * p.kb + U;
  uint B = (uint)(p.src_off & 3ul) + b * 18u;
  ulong oo = (ulong)p.tile0 * p.ku * 64u + g;
  ((device uint4*)weight)[oo] = iq4nl_block(r, B);
  scales[oo] = 1;
  ((device ushort*)biases)[oo] = ldh(r, B);
}

// ---------------------------------------------------------------- IQ3_S
// 110-byte blocks: d @0, qs @2 (64), qh @66 (8), signs @74 (32), scales @106
// (4). Unit u -> words le32(qs + 8u), le32(qs + 8u + 4), le32(signs + 4u);
// `.scales` per unit (qh[u], scale nibble u). One thread per unit:
// grid = nb * 8.

static inline void iq3s_unit(const device uint* r, uint B, uint u,
                             thread uint* o) {
  uint q[2], s[1];
  ldu<2>(r, B + 2u + 8u * u, q);
  ldu<1>(r, B + 74u + 4u * u, s);
  o[0] = q[0];
  o[1] = q[1];
  o[2] = s[0];
}

// Two nibbles per byte of x (low 16 bits) -> 4 bytes, low nibble first.
static inline uint nibbytes(uint x) {
  return (x & 0xFu) | ((x & 0xF0u) << 4) | ((x & 0xF00u) << 8) |
      ((x & 0xF000u) << 12);
}

static inline uint4 iq3s_scales(const device uint* r, uint B) {
  uint h[2];
  ldu<2>(r, B + 66u, h);
  uint sc = (uint)ldh(r, B + 106u) | ((uint)ldh(r, B + 108u) << 16);
  uint n0 = nibbytes(sc), n1 = nibbytes(sc >> 16);
  return uint4(il2(h[0], n0), il2(h[0] >> 16, n0 >> 16), il2(h[1], n1),
               il2(h[1] >> 16, n1 >> 16));
}

[[kernel]] void mlx_node_gguf_repack_iq3s(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  const device uint* r = word_base(staging, p.src_off);
  uint b = g >> 3, u = g & 7u;
  uint B = (uint)(p.src_off & 3ul) + b * 110u;
  uint o[3];
  iq3s_unit(r, B, u, o);
  ulong gb = (ulong)p.row0 * p.kb + b;
  device uint* w = weight + (gb * 8u + u) * 3u;
  w[0] = o[0];
  w[1] = o[1];
  w[2] = o[2];
  if (u == 0) {
    ((device uint4*)scales)[gb] = iq3s_scales(r, B);
    ((device ushort*)biases)[gb] = ldh(r, B);
  }
}

[[kernel]] void mlx_node_gguf_repack_iq3s_t64(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device half* biases [[buffer(3)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb * 8u) return;
  const device uint* r = word_base(staging, p.src_off);
  uint rr = g & 63u, rest = g >> 6, U = rest % p.ku, T = rest / p.ku;
  uint s = U >> 3, u = U & 7u, b = (T * 64u + rr) * p.kb + s;
  uint B = (uint)(p.src_off & 3ul) + b * 110u;
  uint o[3];
  iq3s_unit(r, B, u, o);
  ulong tg = (ulong)p.tile0 + T;
  device uint* w = weight + ((ulong)p.tile0 * p.ku * 64u + g) * 3u;
  w[0] = o[0];
  w[1] = o[1];
  w[2] = o[2];
  if (u == 0) {
    ulong oo = (tg * p.kb + s) * 64u + rr;
    ((device uint4*)scales)[oo] = iq3s_scales(r, B);
    ((device ushort*)biases)[oo] = ldh(r, B);
  }
}

// ---------------------------------------------------------------- Q8_0
// 34-byte blocks: d @0, qs @2 (32 int8). MLX affine 8-bit: `.weight` = qs
// bytes ^ 0x80, `.scales` = d (float16), no `.biases`.

[[kernel]] void mlx_node_gguf_repack_q8_0(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device half* scales [[buffer(2)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb) return;
  const device uint* r = word_base(staging, p.src_off);
  uint B = (uint)(p.src_off & 3ul) + g * 34u;
  uint q[8];
  ldu_end<8>(r, B + 2u, q);
  ulong gb = (ulong)p.row0 * p.kb + g;
  device uint4* w4 = (device uint4*)weight + gb * 2u;
  w4[0] = uint4(q[0], q[1], q[2], q[3]) ^ 0x80808080u;
  w4[1] = uint4(q[4], q[5], q[6], q[7]) ^ 0x80808080u;
  ((device ushort*)scales)[gb] = ldh(r, B);
}

// ---------------------------------------------------------------- MXFP4
// 17-byte blocks (any byte offset; byte loads): E8M0 e @0, qs @1 (16).
// `.weight` codes 0..15 = low nibbles, 16..31 = high nibbles; `.scales` uint8
// e; no `.biases`.

static inline uint ldb4(const device uchar* s) {
  return (uint)s[0] | ((uint)s[1] << 8) | ((uint)s[2] << 16) |
      ((uint)s[3] << 24);
}

[[kernel]] void mlx_node_gguf_repack_mxfp4(
    const device uchar* staging [[buffer(0)]],
    device uint* weight [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    constant GgufRepackParams& p [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  if (g >= p.nb) return;
  const device uchar* s = staging + p.src_off + (ulong)g * 17u;
  ulong gb = (ulong)p.row0 * p.kb + g;
  ((device uint4*)weight)[gb] =
      nib32(ldb4(s + 1), ldb4(s + 5), ldb4(s + 9), ldb4(s + 13));
  scales[gb] = s[0];
}
