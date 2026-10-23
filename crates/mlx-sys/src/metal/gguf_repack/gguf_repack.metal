// GGUF raw blocks -> the K-quant runtime arrays (`.weight` uint32, `.scales`
// uint8, `.biases` float16), row-major or Tiled64, for one band of rows of a
// tensor. build.rs prebuilds this file into paged_attn.metallib; the host
// names below must match the kernel table in mlx_gguf_repack.cpp, which
// gguf_repack_metallib_names checks both ways.
//
// The staging bytes of the band start at `src_off`; output positions are
// global (band row `row0`, Tiled64 tile `tile0 = row0 / 64`), so bands of one
// tensor written in any order give the same arrays as one whole-tensor band.
// Every output byte must equal the CPU repacker (gguf_kquant.rs
// KQuantRepacker, then kquant_tile_rows for Tiled64).

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
