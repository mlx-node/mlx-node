// Adapted from Splash 1.6.1 (https://github.com/incoai/splash) d8af1054aab0f97de21d043e01aa98af3754f10e,
// runtime/metal/kernels/shared/gguf_repack.metal. Apache-2.0; see THIRD_PARTY_NOTICES. Changed: the eight formats
// of metal/abi/QuantFormat.h only (the MLX affine, native-copy and other formats' cases removed); moe_expert_load and
// its MoE includes removed.
// GGUF block layouts follow ggml (MIT, The ggml authors).
#pragma clang fp reassociate(off)
#include "metal/abi/GgufRepack.h"
#include <metal_stdlib>

using namespace metal;

// ---- weight preparation: native GGUF rows -> MDGG0001 planes (metal/abi/QuantFormat.h) ----
// One thread per (row n, 32-wide K group g) of a chunk the host staged in image order.
// Word w of the little-endian string of 32 slot values of `bits` bits each (1, 2, 4 or 8).
static inline uint gguf_bit_word(thread const uchar *slots, uint bits, uint w) {
  const uint per = 32 / bits;
  uint word = 0;
  for (uint i = 0; i < per; ++i) word |= uint(slots[w * per + i]) << (bits * i);
  return word;
}
static inline void gguf_store_bits(thread const uchar *slots, uint bits, device uchar *dst) {
  for (uint w = 0; w < bits; ++w) ((device uint *)dst)[w] = gguf_bit_word(slots, bits, w);
}
// 4-bit linear codes: word c holds slots 8c..8c+7, pair p at bits 4p (e0) and 16 + 4p (e1).
static inline void gguf_store_pairs(thread const uchar *slots, device uchar *dst) {
  for (uint c = 0; c < 4; ++c) {
    uint word = 0;
    for (uint i = 0; i < 8; ++i) word |= uint(slots[8 * c + i]) << ((i & 1) * 16 + 4 * (i >> 1));
    ((device uint *)dst)[c] = word;
  }
}
// Nibbles 0..3 of x at bits 0, 8, 16, 24 to bits 0, 16, 4, 20: four consecutive 4-bit codes as a pair word holds
// them (gguf_store_pairs).
static inline uint gguf_nibble_pairs(uint x) {
  return (x & 0xFu) | ((x >> 12) & 0xF0u) | ((x << 8) & 0xF0000u) | ((x >> 4) & 0xF00000u);
}
// Group j of a Q4_K (qs at byte 16) or Q5_K (qs at 48, with fifth bits) block from its words: a few shifts a word
// instead of a byte load and a pack per element. Pair word c is qs words c and 4 + c; the fifth bits, bit j of
// qh[4k..4k+3] gathered by the multiply, go to slots 8 (k % 4) + 4 (k / 4) on.
static inline void gguf_repack_k_words(device const uchar *blk, uint qs_at, bool fifth, uint j, device uchar *out0,
                                       device uchar *out1, device uchar *meta) {
  device const uint *qs = (device const uint *)(blk + qs_at + (j / 2) * 32);
  uint4 pairs;
  for (uint c = 0; c < 4; ++c)
    pairs[c] = gguf_nibble_pairs((qs[c] >> (4 * (j % 2))) & 0x0F0F0F0Fu) |
               gguf_nibble_pairs((qs[4 + c] >> (4 * (j % 2))) & 0x0F0F0F0Fu) << 8;
  *(device uint4 *)out0 = pairs;
  if (fifth) {
    device const uint *qh = (device const uint *)(blk + 16);
    uint bits = 0;
    for (uint k = 0; k < 8; ++k)
      bits |= ((((qh[k] >> j) & 0x01010101u) * 0x204081u) >> 21 & 0xFu) << (8 * (k % 4) + 4 * (k / 4));
    *(device uint *)out1 = bits;
  }
  device const uint *head = (device const uint *)blk;
  if (j == 0) *(device uint4 *)meta = uint4(head[0], head[1], head[2], head[3]);
}
// Group g (32 wide) of row n of a tensor of G groups a row in format fmt: its native row from `row` on, written
// into its plane0, plane1 and meta planes, each [rows / QUANT_TILE_ROWS][units][QUANT_TILE_ROWS]. Q4_K and Q5_K
// blocks are read as words: their sizes are multiples of 4, and rows are whole blocks after a tensor start on 4
// bytes in the staging.
static inline void gguf_repack_group(device const uchar *row, uint fmt, uint G, uint n, uint g, device uchar *plane0,
                                     device uchar *plane1, device uchar *meta_plane) {
  constant QuantFormat &f = kQuantFormats[fmt];
  const uint b = g / f.meta_groups, j = g % f.meta_groups;   // native block b holds meta unit b
  device const uchar *blk = row + ulong(b) * f.block_bytes;
  device uchar *out0 = plane0 + quant_tile_index(n, g, G) * f.plane0_bytes;
  device uchar *out1 = plane1 + quant_tile_index(n, g, G) * f.plane1_bytes;
  device uchar *meta = meta_plane + quant_tile_index(n, b, G / f.meta_groups) * f.meta_bytes;
  uchar lo[32], hi[32];   // per slot: the (low) code and its high bits
  switch (fmt) {
    case GGUF_FMT_Q4K:
      gguf_repack_k_words(blk, 16, false, j, out0, out1, meta);
      break;
    case GGUF_FMT_Q5K:
      gguf_repack_k_words(blk, 48, true, j, out0, out1, meta);
      break;
    case GGUF_FMT_IQ4XS: {
      for (uint l = 0; l < 16; ++l) { const uchar q = blk[8 + 16 * j + l]; lo[quant_slot(l)] = q & 15; lo[quant_slot(16 + l)] = q >> 4; }
      gguf_store_bits(lo, 4, out0);
      if (j == 0) for (uint i = 0; i < 8; ++i) meta[i] = blk[i];
      break;
    }
    case GGUF_FMT_Q6K: {
      const uint hb = j / 4, quarter = j % 4;
      for (uint e = 0; e < 32; ++e) {
        lo[quant_slot(e)] = (blk[64 * hb + 32 * (quarter & 1) + e] >> (4 * (quarter >> 1))) & 15;
        hi[quant_slot(e)] = (blk[128 + 32 * hb + e] >> (2 * quarter)) & 3;
      }
      gguf_store_pairs(lo, out0);
      gguf_store_bits(hi, 2, out1);
      if (j == 0) { for (uint i = 0; i < 16; ++i) meta[i] = blk[192 + i]; meta[16] = blk[208]; meta[17] = blk[209]; meta[18] = 0; meta[19] = 0; }
      break;
    }
    case GGUF_FMT_Q3K: {
      const uint hb = j / 4, jj = j % 4;
      for (uint e = 0; e < 32; ++e) {
        lo[quant_slot(e)] = (blk[32 + 32 * hb + e] >> (2 * jj)) & 3;
        hi[quant_slot(e)] = (blk[e] >> j) & 1;
      }
      gguf_store_bits(lo, 2, out0);
      gguf_store_bits(hi, 1, out1);
      if (j == 0) { meta[0] = blk[108]; meta[1] = blk[109]; meta[2] = 0; meta[3] = 0; for (uint i = 0; i < 12; ++i) meta[4 + i] = blk[96 + i]; }
      break;
    }
    case GGUF_FMT_Q80: {
      for (uint e = 0; e < 32; ++e) lo[quant_slot(e)] = blk[2 + e];
      gguf_store_bits(lo, 8, out0);
      meta[0] = blk[0]; meta[1] = blk[1];
      break;
    }
    case GGUF_FMT_IQ4NL: {   // the meta unit, then qs[16]
      for (uint l = 0; l < 16; ++l) { const uchar q = blk[f.meta_bytes + l]; lo[quant_slot(l)] = q & 15; lo[quant_slot(16 + l)] = q >> 4; }
      gguf_store_bits(lo, 4, out0);
      for (uint i = 0; i < f.meta_bytes; ++i) meta[i] = blk[i];
      break;
    }
    case GGUF_FMT_IQ3S:
    default: {  // IQ3_S: grid entry t covers elements 4t..4t+3 (qs[t], ninth bit t of qh); sign bit e negates element e
      device const uchar *qs = blk + 2 + 8 * j, *signs = blk + 74 + 4 * j;
      const uint qh = blk[66 + j], scale = (blk[106 + j / 2] >> (4 * (j % 2))) & 15;
      for (uint e = 0; e < 32; ++e) hi[quant_slot(e)] = (signs[e / 8] >> (e % 8)) & 1;
      const uint sign = gguf_bit_word(hi, 1, 0);
      for (uint c = 0; c < 4; ++c)
        ((device uint *)out0)[c] = uint(qs[c]) | uint(qs[4 + c]) << 8 | ((sign >> (8 * c)) & 0xFF) << 16 |
                                   ((qh >> c) & 1) << 24 | ((qh >> (4 + c)) & 1) << 25 | scale << 26;
      if (j == 0) { meta[0] = blk[0]; meta[1] = blk[1]; }
      break;
    }
  }
}

kernel void gguf_repack(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                        constant GgufRepackParams &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  const uint G = p.input_size / 32;
  if (t >= p.rows * G) return;
  const uint n = t / G, g = t % G;
  gguf_repack_group(src + ulong(n) * p.src_row_bytes, p.fmt, G, n, g, dst, dst + p.dst_plane1, dst + p.dst_meta);
}
