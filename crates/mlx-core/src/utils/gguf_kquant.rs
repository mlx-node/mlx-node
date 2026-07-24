//! ggml K-quant block bytes -> the MLX K-quant array contract.
//!
//! GGUF stores Q4_K / Q5_K / Q6_K as fixed-size super-blocks of 256 values with
//! the codes interleaved across nibble and bit planes. MLX's K-quant decode
//! wants three plain arrays per tensor instead:
//!
//! ```text
//!   q6k  bits=6 group_size=16
//!     .weight uint32[N, K*6/32]  LSB-first 6-bit stream, code in [0, 63]
//!     .scales int8  [N, K/16]    ggml sc, verbatim, SIGNED
//!     .biases f16   [N, K/256]   ggml d, verbatim
//!     decode  scale = .biases[g >> 4] * .scales[g];  bias = -32 * scale
//!
//!   q4k  bits=4 group_size=32   (q5k is the same with bits=5)
//!     .weight uint32[N, K*4/32]  LSB-first 4-bit stream, code in [0, 15]
//!     .scales uint8 [N, 2*K/32]  (sc, m)   interleaved at 2g, 2g+1
//!     .biases f16   [N, 2*K/256] (d, dmin) interleaved at 2G, 2G+1
//!     decode  scale = .biases[2*(g >> 3)]       * .scales[2g]       (d    * sc)
//!             bias  = -(.biases[2*(g >> 3) + 1] * .scales[2g + 1])  (dmin * m)
//! ```
//!
//! with `g` the logical group index `v / group_size` and `G` the super-block
//! index `v / 256`. `.biases` holds `d`, which is a SCALE, not a bias; the name
//! is reused so the `.scales` / `.biases` "is quantized" sentinel sites
//! elsewhere in the loader keep working unchanged.
//!
//! Sub-scales own CONTIGUOUS output groups: `sc[j]` covers `[16j, 16j+16)` for
//! q6k and `[32j, 32j+32)` for q4k/q5k. ggml's interleaving affects code storage
//! only, never scale ownership.
//!
//! Two things make the repack more than a memcpy:
//!
//! 1. Q6_K's ql/qh de-swizzle. A super-block is two 128-value halves; inside a
//!    half the four 32-value groups live in the low/high nibble of two 32-byte
//!    `ql` planes plus a 2-bit field of one 32-byte `qh` plane. The index
//!    arithmetic below is the same as `load_q6k_tensor_bf16` in
//!    `crates/mlx-core/src/utils/gguf.rs:641`.
//!
//! 2. 5- and 6-bit codes straddle uint32 words, so packing needs a bit cursor
//!    rather than `word |= code << (i * bits)`. 4-bit does not straddle, but one
//!    cursor covers all three widths.
//!
//! Rows repack independently, so [`KQuantRepacker`] takes a tensor a chunk of
//! rows at a time and a loader never has to hold a whole multi-hundred-megabyte
//! GGUF payload resident. [`repack_kquant`] is the whole-tensor wrapper.
//!
//! Correctness is gated by `crates/mlx-core/tests/kquant_ggml_parity.rs`, which
//! drives this repacker and MLX's CPU decode against ggml's own decoders.

use napi::{Error, Result};

/// Values per ggml super-block. `ggml-common.h:89`.
pub const QK_K: usize = 256;

/// ggml K-quant formats mlx-node can import.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KQuantFormat {
    Q4K,
    Q5K,
    Q6K,
}

impl KQuantFormat {
    /// Bit width of one code.
    pub fn bits(self) -> usize {
        match self {
            Self::Q4K => 4,
            Self::Q5K => 5,
            Self::Q6K => 6,
        }
    }

    /// Values covered by one sub-scale.
    pub fn group_size(self) -> usize {
        match self {
            Self::Q4K | Self::Q5K => 32,
            Self::Q6K => 16,
        }
    }

    /// Size of one ggml super-block in bytes. `ggml-common.h:326/344/361`.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// The `mode` string MLX's quantized ops dispatch on.
    pub fn mlx_mode(self) -> &'static str {
        match self {
            Self::Q4K => "q4k",
            Self::Q5K => "q5k",
            Self::Q6K => "q6k",
        }
    }

    /// GGUF tensor type id. `ggml.h` `GGML_TYPE_Q4_K` / `Q5_K` / `Q6_K`.
    pub fn gguf_type(self) -> u32 {
        match self {
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
        }
    }

    /// Whether `.scales` is signed int8 (q6k) rather than uint8 (q4k/q5k).
    pub fn scales_are_signed(self) -> bool {
        matches!(self, Self::Q6K)
    }

    /// `.weight` columns for a row of `k` values.
    pub fn weight_cols(self, k: usize) -> usize {
        k * self.bits() / 32
    }

    /// `.scales` columns for a row of `k` values. q6k stores one sub-scale per
    /// 16 values; q4k/q5k store a (sc, m) pair per 32.
    pub fn scales_cols(self, k: usize) -> usize {
        match self {
            Self::Q6K => k / 16,
            Self::Q4K | Self::Q5K => 2 * (k / 32),
        }
    }

    /// `.biases` columns for a row of `k` values. q6k stores `d`; q4k/q5k store
    /// the `(d, dmin)` pair.
    pub fn biases_cols(self, k: usize) -> usize {
        match self {
            Self::Q6K => k / QK_K,
            Self::Q4K | Self::Q5K => 2 * (k / QK_K),
        }
    }

    /// Bytes one row of `k` values occupies in the GGUF payload. `k` is assumed
    /// to be a multiple of `QK_K`, which `KQuantRepacker::new` enforces.
    pub fn row_bytes(self, k: usize) -> usize {
        (k / QK_K) * self.block_bytes()
    }
}

/// `.scales` for one tensor, in the dtype MLX expects for the format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KQuantScales {
    /// q6k: ggml's signed int8 sub-scales, verbatim.
    Signed(Vec<i8>),
    /// q4k/q5k: `(sc, m)` pairs unpacked out of ggml's 6-bit fields.
    Unsigned(Vec<u8>),
}

/// The three MLX arrays for one repacked tensor, flattened row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KQuantArrays {
    /// `uint32[rows, format.weight_cols(k)]`
    pub weight: Vec<u32>,
    /// `int8[rows, k/16]` or `uint8[rows, 2*k/32]`
    pub scales: KQuantScales,
    /// f16 bit patterns, `[rows, format.biases_cols(k)]`
    pub biases: Vec<u16>,
}

/// ggml's 6-bit (sub-scale, min) unpacker. `ggml-quants.c:880`.
///
/// `j < 4` reads two whole bytes; `j >= 4` splices a low nibble with two bits
/// stolen from the top of an earlier byte. Both branches run for every
/// super-block, since `j` sweeps 0..8.
///
/// A Rust transliteration of a ggml function is exactly the kind of thing that
/// drifts, so `rust_get_scale_min_k4_matches_ggml_exhaustively` in
/// `crates/mlx-core/tests/kquant_ggml_parity.rs` sweeps every input byte
/// combination each branch can see against the vendored C.
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// LSB-first bit cursor over a row's `uint32` words.
///
/// MLX packs n-bit codes as a plain LSB-first bitstream: bit `j` of code `i`
/// lands at absolute bit index `i * bits + j`, and word `w` holds bit indices
/// `[32w, 32w + 32)` with the word's LSB the lower index. That matches the
/// writer at `mlx/backend/cpu/quantized.cpp:1303` and the `extract_bits` reader
/// at `mlx/backend/cpu/quantized.cpp:75`. Accumulating straight into `u32`
/// words (rather than bytes that get reinterpreted) keeps this independent of
/// host endianness.
struct BitPacker<'a> {
    out: &'a mut Vec<u32>,
    acc: u64,
    nbits: u32,
}

impl<'a> BitPacker<'a> {
    fn new(out: &'a mut Vec<u32>) -> Self {
        Self {
            out,
            acc: 0,
            nbits: 0,
        }
    }

    fn push(&mut self, code: u32, bits: usize) {
        // acc never holds more than 31 pending bits before a push, so 31 + 6
        // cannot overflow a u64.
        self.acc |= u64::from(code) << self.nbits;
        self.nbits += bits as u32;
        while self.nbits >= 32 {
            self.out.push(self.acc as u32);
            self.acc >>= 32;
            self.nbits -= 32;
        }
    }

    /// Every K-quant row is a whole number of super-blocks and `256 * bits` is
    /// always a multiple of 32, so the cursor must land on a word boundary.
    fn finish(self) -> Result<()> {
        if self.nbits != 0 {
            return Err(Error::from_reason(format!(
                "K-quant repack: bit cursor ended {} bits into a uint32 word",
                self.nbits
            )));
        }
        Ok(())
    }
}

// ggml-common.h:361 — block_q6_K field offsets.
const Q6K_QL_OFFSET: usize = 0;
const Q6K_QH_OFFSET: usize = 128;
const Q6K_SCALES_OFFSET: usize = 192;
const Q6K_D_OFFSET: usize = 208;

// ggml-common.h:326 — block_q4_K field offsets.
const Q4K_D_OFFSET: usize = 0;
const Q4K_DMIN_OFFSET: usize = 2;
const Q4K_SCALES_OFFSET: usize = 4;
const Q4K_QS_OFFSET: usize = 16;

// ggml-common.h:344 — block_q5_K field offsets.
const Q5K_D_OFFSET: usize = 0;
const Q5K_DMIN_OFFSET: usize = 2;
const Q5K_SCALES_OFFSET: usize = 4;
const Q5K_QH_OFFSET: usize = 16;
const Q5K_QS_OFFSET: usize = 48;

/// Q6_K code at logical index `v` of a super-block. Cross-checked against
/// `dequantize_row_q6_K` (`ggml-quants.c:1939`): with `v = n + 32k + l` this
/// reproduces ggml's `q1`/`q2`/`q3`/`q4` fetches for `k = 0..4`. The `- 32`
/// offset is ggml's and is not applied here; the contract folds it into the
/// bias.
pub fn q6k_code(blk: &[u8], v: usize) -> u32 {
    let group32 = v / 32;
    let lane = v % 32;
    let group_in_half = group32 % 4;

    let ql_index = Q6K_QL_OFFSET + (group32 / 4) * 64 + (group_in_half % 2) * 32 + lane;
    let ql_shift = (group_in_half / 2) * 4;
    let low = u32::from((blk[ql_index] >> ql_shift) & 0x0f);

    let qh_index = Q6K_QH_OFFSET + (group32 / 4) * 32 + lane;
    let qh_shift = group_in_half * 2;
    let high = u32::from((blk[qh_index] >> qh_shift) & 0x03);

    low | (high << 4)
}

/// Q4_K code at logical index `v`. The super-block is four 64-value halves; in
/// half `h` the first 32 values are the low nibbles of `qs[32h .. 32h+32)` and
/// the next 32 are the high nibbles of the same bytes.
pub fn q4k_code(blk: &[u8], v: usize) -> u32 {
    let half = v / 64;
    let r = v % 64;
    let byte = blk[Q4K_QS_OFFSET + 32 * half + (r % 32)];
    u32::from(if r < 32 { byte & 0x0f } else { byte >> 4 })
}

/// Q5_K code at logical index `v`: Q4_K's nibble layout plus one bit from `qh`.
/// ggml does not advance the `qh` pointer across halves — it walks the bit
/// position instead (`u1 <<= 2, u2 <<= 2`) — so `qh` is indexed by lane alone
/// and the bit index is `2*half` for the low-nibble group, `2*half + 1` for the
/// high-nibble group.
pub fn q5k_code(blk: &[u8], v: usize) -> u32 {
    let half = v / 64;
    let r = v % 64;
    let lane = r % 32;
    let lo = blk[Q5K_QS_OFFSET + 32 * half + lane];
    let nibble = if r < 32 { lo & 0x0f } else { lo >> 4 };
    let high_bit = 2 * half + usize::from(r >= 32);
    let high = (blk[Q5K_QH_OFFSET + lane] >> high_bit) & 0x01;
    u32::from(nibble | (high << 4))
}

/// Reserve room for `rows * cols` more elements, reporting a failed allocation
/// as an error rather than aborting the process — a K-quant destination for a
/// large tensor runs to hundreds of megabytes.
fn reserve_rows<T>(dst: &mut Vec<T>, rows: usize, cols: usize, what: &str) -> Result<()> {
    let extra = rows
        .checked_mul(cols)
        .ok_or_else(|| Error::from_reason(format!("K-quant repack: .{what} length overflow")))?;
    dst.try_reserve(extra).map_err(|e| {
        Error::from_reason(format!(
            "K-quant repack: cannot reserve .{what} for {rows} rows: {e}"
        ))
    })
}

/// Repacks a tensor row by row, so a caller can walk a multi-gigabyte GGUF a
/// chunk at a time instead of holding the whole payload resident.
///
/// Appending in chunks produces exactly the arrays a single whole-tensor call
/// would: every row is a whole number of super-blocks and `256 * bits` is a
/// multiple of 32 for all three widths, so each row's code stream starts and
/// ends on a `uint32` boundary. `chunked_repack_equals_whole_tensor_repack`
/// pins that.
///
/// The destination still grows to the full repacked tensor — only the *source*
/// is streamed. Nothing here dequantizes; the output is the same bits at ggml's
/// byte size.
pub struct KQuantRepacker {
    format: KQuantFormat,
    k: usize,
    rows: usize,
    weight: Vec<u32>,
    /// q6k `.scales`; empty for q4k/q5k.
    scales_i8: Vec<i8>,
    /// q4k/q5k `.scales`; empty for q6k.
    scales_u8: Vec<u8>,
    biases: Vec<u16>,
}

impl KQuantRepacker {
    /// `rows_hint` sizes the destination up front. `push_rows` accepts any
    /// number of rows regardless, but passing the tensor's true row count keeps
    /// the destination to a single allocation.
    pub fn new(format: KQuantFormat, k: usize, rows_hint: usize) -> Result<Self> {
        if k == 0 || !k.is_multiple_of(QK_K) {
            return Err(Error::from_reason(format!(
                "{} repack: K must be a positive multiple of {QK_K}, got {k}",
                format.mlx_mode()
            )));
        }
        let mut repacker = Self {
            format,
            k,
            rows: 0,
            weight: Vec::new(),
            scales_i8: Vec::new(),
            scales_u8: Vec::new(),
            biases: Vec::new(),
        };
        repacker.reserve(rows_hint)?;
        Ok(repacker)
    }

    fn reserve(&mut self, rows: usize) -> Result<()> {
        let (format, k) = (self.format, self.k);
        reserve_rows(&mut self.weight, rows, format.weight_cols(k), "weight")?;
        reserve_rows(&mut self.biases, rows, format.biases_cols(k), "biases")?;
        if format.scales_are_signed() {
            reserve_rows(&mut self.scales_i8, rows, format.scales_cols(k), "scales")
        } else {
            reserve_rows(&mut self.scales_u8, rows, format.scales_cols(k), "scales")
        }
    }

    /// Rows appended so far.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Append `rows` consecutive rows. `blocks` must hold exactly
    /// `rows * format.row_bytes(k)` bytes of ggml super-blocks, row-major, and
    /// must continue where the previous call left off.
    pub fn push_rows(&mut self, blocks: &[u8], rows: usize) -> Result<()> {
        let (format, k) = (self.format, self.k);
        let block_bytes = format.block_bytes();
        let sb_per_row = k / QK_K;
        let want_bytes = rows
            .checked_mul(sb_per_row)
            .and_then(|n| n.checked_mul(block_bytes))
            .ok_or_else(|| Error::from_reason("K-quant repack: tensor size overflow"))?;
        if blocks.len() != want_bytes {
            return Err(Error::from_reason(format!(
                "{} repack: expected {want_bytes} bytes for {rows}x{k}, got {}",
                format.mlx_mode(),
                blocks.len()
            )));
        }
        self.reserve(rows)?;

        let bits = format.bits();
        for row in 0..rows {
            let mut packer = BitPacker::new(&mut self.weight);
            for sb in 0..sb_per_row {
                let start = (row * sb_per_row + sb) * block_bytes;
                let blk = &blocks[start..start + block_bytes];

                match format {
                    KQuantFormat::Q6K => {
                        // .biases[G] = ggml d, raw f16 bits — no float math, so
                        // exactness is free.
                        self.biases.push(u16::from_le_bytes([
                            blk[Q6K_D_OFFSET],
                            blk[Q6K_D_OFFSET + 1],
                        ]));
                        // .scales[16G + j] = ggml sc[j], verbatim int8.
                        for j in 0..QK_K / 16 {
                            self.scales_i8.push(blk[Q6K_SCALES_OFFSET + j] as i8);
                        }
                        for v in 0..QK_K {
                            packer.push(q6k_code(blk, v), bits);
                        }
                    }
                    KQuantFormat::Q4K | KQuantFormat::Q5K => {
                        let (d_off, dmin_off, sc_off) = match format {
                            KQuantFormat::Q4K => (Q4K_D_OFFSET, Q4K_DMIN_OFFSET, Q4K_SCALES_OFFSET),
                            _ => (Q5K_D_OFFSET, Q5K_DMIN_OFFSET, Q5K_SCALES_OFFSET),
                        };
                        // .biases[2G] = d, .biases[2G+1] = dmin — raw f16 bits.
                        self.biases
                            .push(u16::from_le_bytes([blk[d_off], blk[d_off + 1]]));
                        self.biases
                            .push(u16::from_le_bytes([blk[dmin_off], blk[dmin_off + 1]]));
                        // .scales[2g] = sc[j], .scales[2g+1] = m[j]. ggml packs
                        // the eight pairs as 6-bit fields across 12 bytes;
                        // storing them unpacked costs 0.125 bpw and keeps the
                        // affine kernel's per-group pointer walk intact.
                        let packed = &blk[sc_off..sc_off + 12];
                        for j in 0..QK_K / 32 {
                            let (sc, m) = get_scale_min_k4(j, packed);
                            self.scales_u8.push(sc);
                            self.scales_u8.push(m);
                        }
                        for v in 0..QK_K {
                            let code = if format == KQuantFormat::Q4K {
                                q4k_code(blk, v)
                            } else {
                                q5k_code(blk, v)
                            };
                            packer.push(code, bits);
                        }
                    }
                }
            }
            packer.finish()?;
        }
        self.rows += rows;
        Ok(())
    }

    /// The three MLX arrays for the rows appended so far.
    pub fn finish(self) -> KQuantArrays {
        KQuantArrays {
            weight: self.weight,
            scales: if self.format.scales_are_signed() {
                KQuantScales::Signed(self.scales_i8)
            } else {
                KQuantScales::Unsigned(self.scales_u8)
            },
            biases: self.biases,
        }
    }
}

/// Repack `rows * (k / 256)` contiguous ggml super-blocks into the MLX K-quant
/// array contract. `blocks` is the tensor's GGUF payload, row-major.
///
/// Whole-tensor convenience over [`KQuantRepacker`]; a loader that does not want
/// the entire payload resident should drive the repacker directly.
pub fn repack_kquant(
    format: KQuantFormat,
    blocks: &[u8],
    rows: usize,
    k: usize,
) -> Result<KQuantArrays> {
    let mut repacker = KQuantRepacker::new(format, k, rows)?;
    repacker.push_rows(blocks, rows)?;
    Ok(repacker.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repack_shapes_follow_the_array_contract() {
        for (format, k) in [
            (KQuantFormat::Q4K, 256),
            (KQuantFormat::Q5K, 512),
            (KQuantFormat::Q6K, 1024),
        ] {
            let rows = 3;
            let blocks = vec![0u8; rows * (k / QK_K) * format.block_bytes()];
            let out = repack_kquant(format, &blocks, rows, k).expect("repack");
            assert_eq!(out.weight.len(), rows * format.weight_cols(k));
            assert_eq!(out.biases.len(), rows * format.biases_cols(k));
            let scales_len = match &out.scales {
                KQuantScales::Signed(v) => v.len(),
                KQuantScales::Unsigned(v) => v.len(),
            };
            assert_eq!(scales_len, rows * format.scales_cols(k));
        }
    }

    #[test]
    fn repack_rejects_a_truncated_payload() {
        let blocks = vec![0u8; 209];
        let err = repack_kquant(KQuantFormat::Q6K, &blocks, 1, 256).expect_err("short payload");
        assert!(err.reason.contains("expected 210 bytes"), "{}", err.reason);
    }

    #[test]
    fn repack_rejects_a_k_that_is_not_a_super_block_multiple() {
        let blocks = vec![0u8; 210];
        let err = repack_kquant(KQuantFormat::Q6K, &blocks, 1, 128).expect_err("bad K");
        assert!(err.reason.contains("multiple of 256"), "{}", err.reason);
    }

    /// Deterministic block bytes: fixed-seed LCG, no wall clock and no `rand`.
    fn lcg_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn chunked_repack_equals_whole_tensor_repack() {
        for (format, k) in [
            (KQuantFormat::Q4K, 512),
            (KQuantFormat::Q5K, 256),
            (KQuantFormat::Q6K, 768),
        ] {
            let rows = 5;
            let row_bytes = format.row_bytes(k);
            let blocks = lcg_bytes(u64::from(format.gguf_type()), rows * row_bytes);
            let whole = repack_kquant(format, &blocks, rows, k).expect("whole-tensor repack");

            for chunk_rows in [1, 2, 5] {
                let mut repacker = KQuantRepacker::new(format, k, rows).expect("repacker");
                let mut done = 0;
                while done < rows {
                    let n = chunk_rows.min(rows - done);
                    let at = done * row_bytes;
                    repacker
                        .push_rows(&blocks[at..at + n * row_bytes], n)
                        .expect("push chunk");
                    done += n;
                }
                assert_eq!(repacker.rows(), rows);
                assert_eq!(
                    repacker.finish(),
                    whole,
                    "{} chunked {chunk_rows} rows at a time",
                    format.mlx_mode()
                );
            }
        }
    }

    #[test]
    fn repacker_grows_past_its_row_hint() {
        let (format, k) = (KQuantFormat::Q5K, 256);
        let blocks = lcg_bytes(7, 4 * format.row_bytes(k));
        let mut repacker = KQuantRepacker::new(format, k, 0).expect("repacker");
        repacker.push_rows(&blocks, 4).expect("push rows");
        assert_eq!(repacker.rows(), 4);
        assert_eq!(
            repacker.finish(),
            repack_kquant(format, &blocks, 4, k).expect("whole-tensor repack")
        );
    }

    #[test]
    fn push_rows_rejects_a_chunk_that_is_not_whole_rows() {
        let mut repacker = KQuantRepacker::new(KQuantFormat::Q4K, 256, 2).expect("repacker");
        let err = repacker.push_rows(&[0u8; 143], 1).expect_err("partial row");
        assert!(err.reason.contains("expected 144 bytes"), "{}", err.reason);
    }
}
