//! K-quant parity gate: MLX's CPU Q4_K / Q5_K / Q6_K decode against ggml's own
//! decoders.
//!
//! Nothing here is a checked-in golden blob. Every vector is synthesised at test
//! time from a fixed-seed LCG (no `time()`, no `rand()`, no wall clock), decoded
//! by ggml's decoders vendored VERBATIM in `crates/mlx-core/vendor/ggml/`,
//! repacked into the MLX array contract by the production repacker in
//! `crates/mlx-core/src/utils/gguf_kquant.rs`, and decoded again by MLX itself
//! through `mlx_dequantize` on the CPU stream. Regenerating on every run is
//! strictly stronger than a blob: a blob only proves the decode still matches
//! whatever produced the blob.
//!
//! Three independent things are asserted per case:
//!
//! 1. the repack is LOSSLESS — every code, sub-scale and super-scale read back
//!    out of the three MLX arrays equals the ggml block field it came from;
//! 2. MLX's decode matches ggml's, bitwise (see the Q6_K caveat below);
//! 3. the case set is not vacuous — every code value, every sub-scale, and the
//!    degenerate super-scales were actually exercised.
//!
//! Run:
//!   cargo test -p mlx-core --release --test kquant_ggml_parity
//!
//! Q6_K CAVEAT. Q6_K is compared zero-sign-insensitively, and only Q6_K.
//! ggml computes `(d*sc) * (code - 32)` with the offset applied in integer
//! arithmetic; the MLX array contract folds the offset into the group bias as
//! `bias = -32 * scale` and computes `scale*code + bias`. Where ggml yields
//! `-0.0` (negative sub-scale, zero code) the folded form yields `+0.0`,
//! because IEEE-754 round-to-nearest makes `x + (-x)` positive for every finite
//! `x`. That is the ONLY way the two forms differ: an exhaustive sweep over all
//! 63,488 finite halves x 256 int8 sub-scales x 64 codes (1,040,187,392
//! combinations) finds divergences at zero magnitude and nowhere else.
//! Unfolding it would give up the affine-kernel reuse the whole design rests
//! on, for a difference no downstream arithmetic can observe. The gate still
//! asserts that every tolerated pair is zero on BOTH sides, so a real numeric
//! divergence cannot hide inside the exemption. Q4_K and Q5_K have no such
//! reassociation and are compared strictly bitwise.
//!
//! The sweep above is over FINITE super-scales. The one non-finite super-scale
//! the fold also parts on, `d = inf`, is pinned separately by
//! `q6k_non_finite_super_scale_diverges_from_ggml` and is unreachable from a
//! real file.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::PathBuf;

use mlx_core::array::MxArray;
use mlx_core::utils::gguf_kquant::{
    KQuantArrays, KQuantFormat, KQuantScales, QK_K, get_scale_min_k4, q4k_code, q5k_code, q6k_code,
    repack_kquant,
};

// ---------------------------------------------------------------------------
// ggml ground truth — vendored verbatim, compiled by build.rs
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// `ggml-quants.c:1529`
    fn dequantize_row_q4_K(x: *const u8, y: *mut f32, k: i64);
    /// `ggml-quants.c:1731`
    fn dequantize_row_q5_K(x: *const u8, y: *mut f32, k: i64);
    /// `ggml-quants.c:1939`
    fn dequantize_row_q6_K(x: *const u8, y: *mut f32, k: i64);
    /// `ggml-quants.c:880`
    fn ggml_ref_get_scale_min_k4(j: i32, q: *const u8, d: *mut u8, m: *mut u8);
}

fn ggml_get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    let mut d = 0u8;
    let mut m = 0u8;
    // SAFETY: `q` is the 12-byte packed scale field; the reference reads only
    // indices in [0, 12) for j in [0, 8).
    unsafe { ggml_ref_get_scale_min_k4(j as i32, q.as_ptr(), &mut d, &mut m) };
    (d, m)
}

fn ggml_decode_row(format: KQuantFormat, row_blocks: &[u8], k: usize) -> Vec<f32> {
    let mut out = vec![0f32; k];
    // SAFETY: `row_blocks` holds exactly k/256 well-formed blocks (checked by
    // the caller) and is 8-byte aligned via `BlockBuf`; `out` has room for k
    // floats.
    unsafe {
        let x = row_blocks.as_ptr();
        let y = out.as_mut_ptr();
        match format {
            KQuantFormat::Q4K => dequantize_row_q4_K(x, y, k as i64),
            KQuantFormat::Q5K => dequantize_row_q5_K(x, y, k as i64),
            KQuantFormat::Q6K => dequantize_row_q6_K(x, y, k as i64),
        }
    }
    out
}

/// ggml's block structs embed `ggml_half`, so the reference decoders issue
/// 16-bit loads against this buffer. A bare `Vec<u8>` only promises alignment
/// 1; backing the bytes with `u64` makes every block start 8-byte aligned.
struct BlockBuf {
    words: Vec<u64>,
    len: usize,
}

impl BlockBuf {
    fn zeroed(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(8)],
            len,
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: u64 has no padding or invalid bit patterns, and `len` never
        // exceeds the allocation.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above; the borrow is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast::<u8>(), self.len) }
    }
}

// ---------------------------------------------------------------------------
// Deterministic PRNG — 64-bit LCG, fixed seed, never time/rand
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

// ---------------------------------------------------------------------------
// ggml block writers — inverse of the de-swizzle in gguf_kquant.rs
//
// Test-only. Every writer is round-trip checked against the production reader
// inside the generator, so a bug here cannot silently weaken the gate.
// ---------------------------------------------------------------------------

fn q6k_set_code(blk: &mut [u8], v: usize, code: u32) {
    let group32 = v / 32;
    let lane = v % 32;
    let gih = group32 % 4;
    let ql_index = (group32 / 4) * 64 + (gih % 2) * 32 + lane;
    let ql_shift = (gih / 2) * 4;
    blk[ql_index] = (blk[ql_index] & !(0x0fu8 << ql_shift)) | (((code & 0x0f) as u8) << ql_shift);
    let qh_index = 128 + (group32 / 4) * 32 + lane;
    let qh_shift = gih * 2;
    blk[qh_index] =
        (blk[qh_index] & !(0x03u8 << qh_shift)) | ((((code >> 4) & 0x03) as u8) << qh_shift);
}

fn q45k_set_nibble(blk: &mut [u8], qs_offset: usize, v: usize, code: u32) {
    let half = v / 64;
    let r = v % 64;
    let idx = qs_offset + 32 * half + (r % 32);
    if r < 32 {
        blk[idx] = (blk[idx] & 0xf0) | ((code & 0x0f) as u8);
    } else {
        blk[idx] = (blk[idx] & 0x0f) | (((code & 0x0f) as u8) << 4);
    }
}

fn q4k_set_code(blk: &mut [u8], v: usize, code: u32) {
    q45k_set_nibble(blk, 16, v, code);
}

fn q5k_set_code(blk: &mut [u8], v: usize, code: u32) {
    q45k_set_nibble(blk, 48, v, code);
    let half = v / 64;
    let r = v % 64;
    let hidx = 16 + (r % 32);
    let hbit = 2 * half + usize::from(r >= 32);
    blk[hidx] = (blk[hidx] & !(1u8 << hbit)) | ((((code >> 4) & 1) as u8) << hbit);
}

/// ggml's own 6-bit (sc, m) packer — `ggml-quants.c:1588` (`quantize_row_q4_K_impl`)
/// and `:1795` (q5_K). Requires the 12 bytes pre-zeroed; upstream uses `|=`.
fn ggml_pack_scale_min_k4(scales12: &mut [u8], ls: &[u8; 8], lm: &[u8; 8]) {
    scales12[..12].fill(0);
    for j in 0..8 {
        let l_s = ls[j];
        let l_m = lm[j];
        if j < 4 {
            scales12[j] = l_s;
            scales12[j + 4] = l_m;
        } else {
            scales12[j + 4] = (l_s & 0xF) | ((l_m & 0xF) << 4);
            scales12[j - 4] |= (l_s >> 4) << 6;
            scales12[j] |= (l_m >> 4) << 6;
        }
    }
}

// ---------------------------------------------------------------------------
// Generation profiles
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Profile {
    /// mixed magnitudes, everything in a realistic weight range
    Typical,
    /// uniformly random codes / sub-scales / super-scales
    RandomAll,
    /// all-zero block: d = 0, every code 0, every sub-scale 0
    Zero,
    /// super-scale is a binary16 subnormal
    DenormalD,
    /// d = smallest normal half, 2^-14
    TinyD,
    /// d = dmin = largest finite half, 65504
    MaxD,
    /// d and dmin both negative
    NegD,
    /// every code 0
    CodeMin,
    /// every code at max (63 / 15 / 31)
    CodeMax,
    /// codes walk 0..=max so every representable code appears
    CodeSweep,
    /// sub-scales walk their whole domain across consecutive super-blocks
    SubscaleSweep,
    /// sub-scales cycle through their extreme values
    SubscaleExtreme,
    /// q6k only: every sub-scale negative, including -128
    SubscaleNeg,
    /// codes alternate 0 / max, sub-scales alternate extremes
    Alternating,
}

const COMMON_PROFILES: &[Profile] = &[
    Profile::Typical,
    Profile::RandomAll,
    Profile::Zero,
    Profile::DenormalD,
    Profile::TinyD,
    Profile::MaxD,
    Profile::NegD,
    Profile::CodeMin,
    Profile::CodeMax,
    Profile::CodeSweep,
    Profile::SubscaleExtreme,
    Profile::Alternating,
];

const Q6K_PROFILES: &[Profile] = &[
    Profile::Typical,
    Profile::RandomAll,
    Profile::Zero,
    Profile::DenormalD,
    Profile::TinyD,
    Profile::MaxD,
    Profile::NegD,
    Profile::CodeMin,
    Profile::CodeMax,
    Profile::CodeSweep,
    Profile::SubscaleExtreme,
    Profile::Alternating,
    Profile::SubscaleNeg,
];

/// A finite, non-inf/non-NaN half with the exponent field forced into
/// `[lo_exp, hi_exp]`. Inf/NaN super-scales are out of contract for a weight
/// tensor and are excluded on purpose.
fn random_finite_half(rng: &mut Lcg, lo_exp: u32, hi_exp: u32) -> u16 {
    let sign = rng.next() & 1;
    let exp = lo_exp + rng.below(hi_exp - lo_exp + 1);
    let man = rng.next() & 0x3ff;
    ((sign << 15) | (exp << 10) | man) as u16
}

fn profile_super_scales(p: Profile, rng: &mut Lcg) -> (u16, u16) {
    match p {
        Profile::Zero => (0x0000, 0x0000),
        // subnormal: exponent field 0, mantissa in [1, 1023]
        Profile::DenormalD => (
            (((rng.next() & 1) << 15) | (1 + rng.below(1023))) as u16,
            (1 + rng.below(1023)) as u16,
        ),
        Profile::TinyD => (0x0400, 0x0400),
        Profile::MaxD => (0x7bff, 0x7bff),
        // sign bit set, exponent in the realistic band
        Profile::NegD => (
            random_finite_half(rng, 7, 13) | 0x8000,
            random_finite_half(rng, 7, 13) | 0x8000,
        ),
        // realistic weight scales live near 2^-8 .. 2^-2 -> exponent field 7..13
        Profile::Typical => (
            random_finite_half(rng, 7, 13),
            random_finite_half(rng, 7, 13),
        ),
        _ => (
            random_finite_half(rng, 0, 30),
            random_finite_half(rng, 0, 30),
        ),
    }
}

fn profile_codes(p: Profile, rng: &mut Lcg, bits: usize, codes: &mut [u32; QK_K]) {
    let max_code = (1u32 << bits) - 1;
    for (v, slot) in codes.iter_mut().enumerate() {
        *slot = match p {
            Profile::Zero | Profile::CodeMin => 0,
            Profile::CodeMax => max_code,
            Profile::CodeSweep => (v as u32) % (max_code + 1),
            Profile::Alternating => {
                if v & 1 == 1 {
                    max_code
                } else {
                    0
                }
            }
            Profile::Typical => {
                // cluster in the middle but still touch both rails
                match rng.below(100) {
                    0..=2 => 0,
                    3..=5 => max_code,
                    _ => rng.below(max_code + 1),
                }
            }
            _ => rng.below(max_code + 1),
        };
    }
}

/// q6k sub-scales are SIGNED int8. Negatives are the easy thing to get wrong,
/// so they get dedicated profiles and appear in every random profile.
fn profile_q6k_subscales(p: Profile, rng: &mut Lcg, block_seq: usize, sc: &mut [i8; 16]) {
    const EXTREMES: [i8; 8] = [-128, -127, -85, -1, 0, 1, 85, 127];
    for (j, slot) in sc.iter_mut().enumerate() {
        *slot = match p {
            Profile::Zero => 0,
            Profile::SubscaleExtreme | Profile::Alternating => EXTREMES[j % 8],
            // 16 sub-scales per block x 16 consecutive blocks = all 256 int8 values
            Profile::SubscaleSweep => ((block_seq * 16 + j) % 256) as u8 as i8,
            Profile::SubscaleNeg => (-1 - rng.below(128) as i32) as i8,
            Profile::Typical => {
                (rng.below(127) as i32 - if rng.next() & 1 == 1 { 63 } else { 0 }) as i8
            }
            _ => rng.next() as u8 as i8,
        };
    }
}

/// q4k/q5k sub-scales are UNSIGNED 6-bit, `[0, 63]`, for both `sc` and `m`.
fn profile_q45k_subscales(
    p: Profile,
    rng: &mut Lcg,
    block_seq: usize,
    ls: &mut [u8; 8],
    lm: &mut [u8; 8],
) {
    const EXTREMES: [u8; 8] = [0, 1, 2, 31, 32, 61, 62, 63];
    for j in 0..8 {
        let (s, m) = match p {
            Profile::Zero => (0, 0),
            // SubscaleNeg has no signed counterpart here; reuse the extremes
            Profile::SubscaleExtreme | Profile::SubscaleNeg => {
                (EXTREMES[j % 8], EXTREMES[(j + 4) % 8])
            }
            // 8 pairs per block x 8 consecutive blocks = all 64 values, for both
            Profile::SubscaleSweep => (
                ((block_seq * 8 + j) % 64) as u8,
                ((block_seq * 8 + j + 32) % 64) as u8,
            ),
            Profile::Alternating => {
                if j & 1 == 1 {
                    (63, 0)
                } else {
                    (0, 63)
                }
            }
            _ => (rng.below(64) as u8, rng.below(64) as u8),
        };
        ls[j] = s;
        lm[j] = m;
    }
}

// ---------------------------------------------------------------------------
// Block synthesis, with generator self-checks
// ---------------------------------------------------------------------------

fn gen_block(format: KQuantFormat, blk: &mut [u8], p: Profile, rng: &mut Lcg, block_seq: usize) {
    blk.fill(0);
    let (d, dmin) = profile_super_scales(p, rng);
    let mut codes = [0u32; QK_K];
    profile_codes(p, rng, format.bits(), &mut codes);

    match format {
        KQuantFormat::Q6K => {
            blk[208..210].copy_from_slice(&d.to_le_bytes());
            let mut sc = [0i8; 16];
            profile_q6k_subscales(p, rng, block_seq, &mut sc);
            for j in 0..16 {
                blk[192 + j] = sc[j] as u8;
            }
            for (v, &code) in codes.iter().enumerate() {
                q6k_set_code(blk, v, code);
            }
            for (v, &code) in codes.iter().enumerate() {
                assert_eq!(
                    q6k_code(blk, v),
                    code,
                    "q6k writer/reader disagree at v={v}"
                );
            }
        }
        KQuantFormat::Q4K | KQuantFormat::Q5K => {
            // block_q4_K and block_q5_K share their first 16 bytes:
            // d, dmin, then the 12-byte packed scale field.
            blk[0..2].copy_from_slice(&d.to_le_bytes());
            blk[2..4].copy_from_slice(&dmin.to_le_bytes());
            let sc_off = 4usize;
            let mut ls = [0u8; 8];
            let mut lm = [0u8; 8];
            profile_q45k_subscales(p, rng, block_seq, &mut ls, &mut lm);
            ggml_pack_scale_min_k4(&mut blk[sc_off..sc_off + 12], &ls, &lm);
            for (v, &code) in codes.iter().enumerate() {
                if format == KQuantFormat::Q4K {
                    q4k_set_code(blk, v, code);
                } else {
                    q5k_set_code(blk, v, code);
                }
            }
            for (v, &code) in codes.iter().enumerate() {
                let got = if format == KQuantFormat::Q4K {
                    q4k_code(blk, v)
                } else {
                    q5k_code(blk, v)
                };
                assert_eq!(got, code, "{:?} writer/reader disagree at v={v}", format);
            }
            // The packed 6-bit scale field must round-trip through ggml's own
            // unpacker, otherwise the "sub-scale" this case claims to cover is
            // not the one ggml sees.
            for j in 0..8 {
                let (sc, m) = ggml_get_scale_min_k4(j, &blk[sc_off..sc_off + 12]);
                assert_eq!((sc, m), (ls[j], lm[j]), "scale packer disagrees at j={j}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Case table
// ---------------------------------------------------------------------------

struct CaseSpec {
    name: &'static str,
    rows: usize,
    k: usize,
    rotation: &'static [Profile],
}

fn case_specs(format: KQuantFormat) -> Vec<CaseSpec> {
    let all = if format == KQuantFormat::Q6K {
        Q6K_PROFILES
    } else {
        COMMON_PROFILES
    };
    let mut out = vec![
        CaseSpec {
            name: "typical_n1_k256",
            rows: 1,
            k: 256,
            rotation: &[Profile::Typical],
        },
        CaseSpec {
            name: "typical_n8_k512",
            rows: 8,
            k: 512,
            rotation: &[Profile::Typical],
        },
        CaseSpec {
            name: "typical_n128_k256",
            rows: 128,
            k: 256,
            rotation: &[Profile::Typical],
        },
        CaseSpec {
            name: "random_n8_k1024",
            rows: 8,
            k: 1024,
            rotation: &[Profile::RandomAll],
        },
        CaseSpec {
            name: "random_n128_k512",
            rows: 128,
            k: 512,
            rotation: &[Profile::RandomAll],
        },
        CaseSpec {
            name: "zero_n8_k256",
            rows: 8,
            k: 256,
            rotation: &[Profile::Zero],
        },
        CaseSpec {
            name: "denormal_d_n8_k512",
            rows: 8,
            k: 512,
            rotation: &[Profile::DenormalD],
        },
        CaseSpec {
            name: "tiny_and_max_d_n8_k512",
            rows: 8,
            k: 512,
            rotation: &[Profile::TinyD, Profile::MaxD],
        },
        CaseSpec {
            name: "negative_d_n8_k256",
            rows: 8,
            k: 256,
            rotation: &[Profile::NegD],
        },
        CaseSpec {
            name: "code_saturation_n8_k512",
            rows: 8,
            k: 512,
            rotation: &[Profile::CodeMin, Profile::CodeMax],
        },
        CaseSpec {
            name: "code_sweep_n8_k256",
            rows: 8,
            k: 256,
            rotation: &[Profile::CodeSweep],
        },
        // 16 super-blocks per row walks every sub-scale value exactly once
        CaseSpec {
            name: "subscale_sweep_n2_k4096",
            rows: 2,
            k: 4096,
            rotation: &[Profile::SubscaleSweep],
        },
        CaseSpec {
            name: "subscale_extremes_n8_k256",
            rows: 8,
            k: 256,
            rotation: &[Profile::SubscaleExtreme],
        },
        CaseSpec {
            name: "alternating_n8_k256",
            rows: 8,
            k: 256,
            rotation: &[Profile::Alternating],
        },
        // Every profile inside one row: exercises the G = g>>4 / g>>3
        // super-block indexing and the per-row bit cursor across 4 super-blocks.
        CaseSpec {
            name: "edge_rotation_n1_k1024",
            rows: 1,
            k: 1024,
            rotation: all,
        },
        CaseSpec {
            name: "edge_rotation_n8_k1024",
            rows: 8,
            k: 1024,
            rotation: all,
        },
        CaseSpec {
            name: "edge_rotation_n128_k1024",
            rows: 128,
            k: 1024,
            rotation: all,
        },
    ];
    if format == KQuantFormat::Q6K {
        out.push(CaseSpec {
            name: "negative_subscales_n8_k512",
            rows: 8,
            k: 512,
            rotation: &[Profile::SubscaleNeg],
        });
        out.push(CaseSpec {
            name: "neg_subscale_code_max_n8_k256",
            rows: 8,
            k: 256,
            rotation: &[Profile::SubscaleNeg, Profile::CodeMax],
        });
    }
    out
}

// ---------------------------------------------------------------------------
// MLX driver
// ---------------------------------------------------------------------------

/// Pin MLX's process-wide default device and stream to the CPU. This file is
/// the reference the Metal gate is measured against, so it has to compare the
/// CPU decode against ggml and nothing else; `kquant_mode_guards.rs` is what
/// holds Metal to this reference. `.cargo/config.toml` pins
/// `RUST_TEST_THREADS=1`, so mutating the global is safe here.
fn pin_cpu_stream() {
    // SAFETY: plain global setters in the MLX FFI; both catch internally.
    unsafe {
        mlx_sys::mlx_set_default_device(0);
        let cpu = mlx_sys::mlx_default_stream(0);
        mlx_sys::mlx_set_default_stream(cpu);
    }
}

/// Read an evaluated float32 array back element by element.
///
/// NOT `mlx_array_to_float32`: that helper materialises through
/// `add(arr, zeros)` (`crates/mlx-sys/src/mlx_common.h:328`), and
/// `-0.0 + 0.0 == +0.0`, so it would silently launder exactly the signed-zero
/// distinction this gate is measuring. `mlx_array_item_at_float32` reads
/// `arr.data<float>()[i]` with no arithmetic.
fn read_f32_exact(handle: *mut mlx_sys::mlx_array, len: usize) -> Vec<f32> {
    let mut h = handle;
    // SAFETY: `handle` is a live array produced by mlx_dequantize.
    let ok = unsafe { mlx_sys::mlx_eval(&mut h, 1) };
    assert!(ok, "eval of the MLX dequantize output failed");
    let mut out = vec![0f32; len];
    for (i, slot) in out.iter_mut().enumerate() {
        // SAFETY: i < len == array size, checked by the caller.
        let ok = unsafe { mlx_sys::mlx_array_item_at_float32(handle, i, slot) };
        assert!(ok, "mlx_array_item_at_float32 failed at index {i}");
    }
    out
}

/// Drive MLX's own K-quant decode over the repacked arrays.
fn mlx_decode(format: KQuantFormat, arrays: &KQuantArrays, rows: usize, k: usize) -> Vec<f32> {
    let rows_i = rows as i64;
    let weight = MxArray::from_uint32(&arrays.weight, &[rows_i, format.weight_cols(k) as i64])
        .expect("weight array");
    let scales = match &arrays.scales {
        KQuantScales::Signed(v) => MxArray::from_int8(v, &[rows_i, format.scales_cols(k) as i64]),
        KQuantScales::Unsigned(v) => {
            MxArray::from_uint8(v, &[rows_i, format.scales_cols(k) as i64])
        }
    }
    .expect("scales array");
    let biases = MxArray::from_float16(&arrays.biases, &[rows_i, format.biases_cols(k) as i64])
        .expect("biases array");

    let mode = CString::new(format.mlx_mode()).expect("mode");
    // SAFETY: the three handles outlive the call; out_dtype 0 is float32.
    let handle = unsafe {
        mlx_sys::mlx_dequantize(
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            format.group_size() as i32,
            format.bits() as i32,
            0,
            mode.as_ptr(),
        )
    };
    assert!(
        !handle.is_null(),
        "mlx_dequantize rejected a well-formed {} tensor ({rows}x{k})",
        format.mlx_mode()
    );
    // SAFETY: non-null handle from mlx_dequantize.
    let len = unsafe { mlx_sys::mlx_array_size(handle) };
    assert_eq!(
        len,
        rows * k,
        "{} decode produced {len} values",
        format.mlx_mode()
    );
    let out = read_f32_exact(handle, len);
    // SAFETY: the handle is owned here and not referenced afterwards.
    unsafe { mlx_sys::mlx_array_delete(handle) };
    out
}

// ---------------------------------------------------------------------------
// Coverage
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Coverage {
    values: u64,
    /// code value -> times seen
    codes: BTreeMap<u32, u64>,
    /// q6k: sub-scale as u8; q4k/q5k: sc in [0, 64)
    subscales: BTreeMap<u8, u64>,
    /// q4k/q5k only: m in [0, 64)
    submins: BTreeMap<u8, u64>,
    rows_seen: BTreeMap<usize, u64>,
    super_blocks_per_row: BTreeMap<usize, u64>,
    d_subnormal: u64,
    d_zero: u64,
    d_negative: u64,
    /// `get_scale_min_k4` with j < 4
    branch_lo: u64,
    /// `get_scale_min_k4` with j >= 4
    branch_hi: u64,
    /// a j >= 4 unpack whose result needed the two bits spliced in from an
    /// earlier byte (i.e. the value does not fit in the low nibble)
    branch_hi_spliced: u64,
    /// non-zero code whose bit range crosses a uint32 word boundary
    straddling_codes: u64,
    ref_negative_zeros: u64,
    ref_abs_max: f32,
}

impl Coverage {
    fn note_code(&mut self, code: u32, v: usize, bits: usize) {
        *self.codes.entry(code).or_default() += 1;
        let bit0 = v * bits;
        if code != 0 && bit0 / 32 != (bit0 + bits - 1) / 32 {
            self.straddling_codes += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

struct Mismatch {
    case: &'static str,
    row: usize,
    v: usize,
    code: u32,
    want: f32,
    got: f32,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "case {} row={} v={} code={}: ggml {:.9e} (0x{:08x}) != mlx {:.9e} (0x{:08x})",
            self.case,
            self.row,
            self.v,
            self.code,
            self.want,
            self.want.to_bits(),
            self.got,
            self.got.to_bits()
        )
    }
}

/// Everything one format's sweep produced.
struct FormatRun {
    coverage: Coverage,
    /// first 16 real disagreements, for the failure message
    samples: Vec<Mismatch>,
    /// total real disagreements
    hard: u64,
    /// disagreements absorbed by the documented Q6_K signed-zero exemption
    signed_zero_only: u64,
}

/// Runs every case for one format.
fn run_format(format: KQuantFormat) -> FormatRun {
    pin_cpu_stream();
    let bits = format.bits();
    let block_bytes = format.block_bytes();
    // One fixed seed per format. No time(), no rand().
    let mut rng =
        Lcg::new(0x5EED_0000u64.wrapping_add((bits as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    let mut cov = Coverage::default();
    let mut samples = Vec::new();
    let mut hard = 0u64;
    let mut signed_zero_only = 0u64;

    for spec in case_specs(format) {
        let sb_per_row = spec.k / QK_K;
        let n_blocks = spec.rows * sb_per_row;
        *cov.rows_seen.entry(spec.rows).or_default() += 1;
        *cov.super_blocks_per_row.entry(sb_per_row).or_default() += 1;

        // 1. synthesise ggml bytes
        let mut buf = BlockBuf::zeroed(n_blocks * block_bytes);
        for b in 0..n_blocks {
            let p = spec.rotation[b % spec.rotation.len()];
            let start = b * block_bytes;
            let bytes = buf.bytes_mut();
            gen_block(
                format,
                &mut bytes[start..start + block_bytes],
                p,
                &mut rng,
                b,
            );
        }

        // 2. ggml reference decode
        let mut reference = Vec::with_capacity(spec.rows * spec.k);
        for row in 0..spec.rows {
            let start = row * sb_per_row * block_bytes;
            reference.extend_from_slice(&ggml_decode_row(
                format,
                &buf.bytes()[start..start + sb_per_row * block_bytes],
                spec.k,
            ));
        }

        // 3. repack through the production repacker
        let arrays = repack_kquant(format, buf.bytes(), spec.rows, spec.k).expect("repack");
        assert_eq!(
            arrays.weight.len(),
            spec.rows * format.weight_cols(spec.k),
            "{} {}: .weight shape violates the array contract",
            format.mlx_mode(),
            spec.name
        );
        assert_eq!(
            arrays.biases.len(),
            spec.rows * format.biases_cols(spec.k),
            "{} {}: .biases shape violates the array contract",
            format.mlx_mode(),
            spec.name
        );

        // 4. losslessness: everything read back out of the three arrays must
        //    equal the ggml block field it came from
        let weight_row_words = format.weight_cols(spec.k);
        let scales_row = format.scales_cols(spec.k);
        let biases_row = format.biases_cols(spec.k);
        for row in 0..spec.rows {
            let w = &arrays.weight[row * weight_row_words..(row + 1) * weight_row_words];
            let b = &arrays.biases[row * biases_row..(row + 1) * biases_row];
            for sb in 0..sb_per_row {
                let start = (row * sb_per_row + sb) * block_bytes;
                let blk = &buf.bytes()[start..start + block_bytes];
                match format {
                    KQuantFormat::Q6K => {
                        let want_d = u16::from_le_bytes([blk[208], blk[209]]);
                        assert_eq!(
                            b[sb],
                            want_d,
                            "{} {}: .biases lost d",
                            format.mlx_mode(),
                            spec.name
                        );
                        note_super_scale(&mut cov, want_d);
                        let KQuantScales::Signed(all) = &arrays.scales else {
                            panic!("q6k must produce signed scales");
                        };
                        let s = &all[row * scales_row..(row + 1) * scales_row];
                        for j in 0..16 {
                            let want = blk[192 + j] as i8;
                            assert_eq!(
                                s[sb * 16 + j],
                                want,
                                "{} {}: .scales lost sub-scale {j}",
                                format.mlx_mode(),
                                spec.name
                            );
                            *cov.subscales.entry(want as u8).or_default() += 1;
                        }
                    }
                    KQuantFormat::Q4K | KQuantFormat::Q5K => {
                        let want_d = u16::from_le_bytes([blk[0], blk[1]]);
                        let want_dmin = u16::from_le_bytes([blk[2], blk[3]]);
                        assert_eq!(
                            (b[2 * sb], b[2 * sb + 1]),
                            (want_d, want_dmin),
                            "{} {}: .biases lost (d, dmin)",
                            format.mlx_mode(),
                            spec.name
                        );
                        note_super_scale(&mut cov, want_d);
                        let KQuantScales::Unsigned(all) = &arrays.scales else {
                            panic!("q4k/q5k must produce unsigned scales");
                        };
                        let s = &all[row * scales_row..(row + 1) * scales_row];
                        for j in 0..8 {
                            // ggml's own unpacker is the oracle for the Rust one
                            let (want_sc, want_m) = ggml_get_scale_min_k4(j, &blk[4..16]);
                            let g = sb * 8 + j;
                            assert_eq!(
                                (s[2 * g], s[2 * g + 1]),
                                (want_sc, want_m),
                                "{} {}: .scales lost (sc, m) at j={j}",
                                format.mlx_mode(),
                                spec.name
                            );
                            if j < 4 {
                                cov.branch_lo += 1;
                            } else {
                                cov.branch_hi += 1;
                                if want_sc > 0x0f || want_m > 0x0f {
                                    cov.branch_hi_spliced += 1;
                                }
                            }
                            *cov.subscales.entry(want_sc).or_default() += 1;
                            *cov.submins.entry(want_m).or_default() += 1;
                        }
                    }
                }
            }
            for v in 0..spec.k {
                let start = (row * sb_per_row + v / QK_K) * block_bytes;
                let blk = &buf.bytes()[start..start + block_bytes];
                let want_code = match format {
                    KQuantFormat::Q6K => q6k_code(blk, v % QK_K),
                    KQuantFormat::Q4K => q4k_code(blk, v % QK_K),
                    KQuantFormat::Q5K => q5k_code(blk, v % QK_K),
                };
                let got_code = extract_code(w, v, bits);
                assert_eq!(
                    got_code,
                    want_code,
                    "{} {}: .weight bitstream lost the code at row={row} v={v}",
                    format.mlx_mode(),
                    spec.name
                );
                cov.note_code(want_code, v, bits);
            }
        }

        // 5. MLX decode vs ggml
        let got = mlx_decode(format, &arrays, spec.rows, spec.k);
        for row in 0..spec.rows {
            for v in 0..spec.k {
                let i = row * spec.k + v;
                let want = reference[i];
                let mine = got[i];
                cov.values += 1;
                if want.to_bits() == 0x8000_0000 {
                    cov.ref_negative_zeros += 1;
                }
                if want.abs() > cov.ref_abs_max {
                    cov.ref_abs_max = want.abs();
                }
                if want.to_bits() == mine.to_bits() {
                    continue;
                }
                let start = (row * sb_per_row + v / QK_K) * block_bytes;
                let blk = &buf.bytes()[start..start + block_bytes];
                let code = match format {
                    KQuantFormat::Q6K => q6k_code(blk, v % QK_K),
                    KQuantFormat::Q4K => q4k_code(blk, v % QK_K),
                    KQuantFormat::Q5K => q5k_code(blk, v % QK_K),
                };
                // Q6_K's folded bias turns ggml's -0.0 into +0.0 and nothing
                // else. Both sides must be exactly zero to qualify, and only
                // Q6_K gets the exemption — for Q4_K/Q5_K this stays a failure.
                if want == 0.0 && mine == 0.0 {
                    signed_zero_only += 1;
                    if format == KQuantFormat::Q6K {
                        continue;
                    }
                }
                hard += 1;
                if samples.len() < 16 {
                    samples.push(Mismatch {
                        case: spec.name,
                        row,
                        v,
                        code,
                        want,
                        got: mine,
                    });
                }
            }
        }
    }
    FormatRun {
        coverage: cov,
        samples,
        hard,
        signed_zero_only,
    }
}

fn note_super_scale(cov: &mut Coverage, d: u16) {
    if (d >> 10) & 0x1f == 0 && d & 0x3ff != 0 {
        cov.d_subnormal += 1;
    }
    if d & 0x7fff == 0 {
        cov.d_zero += 1;
    }
    if d >> 15 != 0 {
        cov.d_negative += 1;
    }
}

/// Read code `v` back out of a row's LSB-first uint32 bitstream.
fn extract_code(row: &[u32], v: usize, bits: usize) -> u32 {
    let mut code = 0u32;
    for j in 0..bits {
        let b = v * bits + j;
        code |= ((row[b / 32] >> (b % 32)) & 1) << j;
    }
    code
}

fn assert_coverage(format: KQuantFormat, cov: &Coverage) {
    let bits = format.bits();
    for code in 0..(1u32 << bits) {
        assert!(
            cov.codes.contains_key(&code),
            "{}: code {code} never generated — the case set is vacuous for it",
            format.mlx_mode()
        );
    }
    match format {
        // all 256 int8 values, so negatives and -128 (0x80) are in here
        KQuantFormat::Q6K => {
            for raw in 0..=255u8 {
                assert!(
                    cov.subscales.contains_key(&raw),
                    "{}: int8 sub-scale {} never generated",
                    format.mlx_mode(),
                    raw as i8
                );
            }
        }
        KQuantFormat::Q4K | KQuantFormat::Q5K => {
            for raw in 0..64u8 {
                assert!(
                    cov.subscales.contains_key(&raw),
                    "{}: sub-scale {raw} never generated",
                    format.mlx_mode()
                );
                assert!(
                    cov.submins.contains_key(&raw),
                    "{}: sub-min {raw} never generated",
                    format.mlx_mode()
                );
            }
            assert!(cov.branch_lo > 0, "get_scale_min_k4 j<4 branch never taken");
            assert!(
                cov.branch_hi > 0,
                "get_scale_min_k4 j>=4 branch never taken"
            );
            assert!(
                cov.branch_hi_spliced > 0,
                "get_scale_min_k4 j>=4 branch never produced a value needing the \
                 two bits spliced in from an earlier byte — the branch ran but \
                 its hard half was never exercised"
            );
        }
    }
    assert!(cov.d_subnormal > 0, "no binary16-subnormal super-scale");
    assert!(cov.d_zero > 0, "no zero super-scale");
    assert!(cov.d_negative > 0, "no negative super-scale");
    for rows in [1usize, 8, 128] {
        assert!(
            cov.rows_seen.contains_key(&rows),
            "no case with {rows} rows"
        );
    }
    assert!(
        cov.super_blocks_per_row.keys().any(|&n| n > 1),
        "no case with more than one super-block per row"
    );
    // 4-bit codes cannot cross a uint32 word: 4 divides 32. 5- and 6-bit codes
    // do, which is why the packer needs a bit cursor at all.
    if bits == 4 {
        assert_eq!(
            cov.straddling_codes, 0,
            "a 4-bit code straddled a uint32 word, which is arithmetically impossible"
        );
    } else {
        assert!(
            cov.straddling_codes > 0,
            "{}: no non-zero code ever crossed a uint32 word boundary",
            format.mlx_mode()
        );
    }
}

fn report(format: KQuantFormat, run: &FormatRun) {
    let cov = &run.coverage;
    println!(
        "{}: {} values, {} distinct codes, {} distinct sub-scales, \
         subnormal d {}, zero d {}, negative d {}, |max| {:.6e}, \
         ggml -0.0 values {}, signed-zero-only diffs {}",
        format.mlx_mode(),
        cov.values,
        cov.codes.len(),
        cov.subscales.len(),
        cov.d_subnormal,
        cov.d_zero,
        cov.d_negative,
        cov.ref_abs_max,
        cov.ref_negative_zeros,
        run.signed_zero_only
    );
}

fn assert_no_mismatches(format: KQuantFormat, run: &FormatRun) {
    if run.hard == 0 {
        return;
    }
    let mut msg = format!(
        "{}: MLX's CPU decode disagrees with ggml on {} of {} values (first {} shown)\n",
        format.mlx_mode(),
        run.hard,
        run.coverage.values,
        run.samples.len()
    );
    for m in &run.samples {
        msg.push_str("  ");
        msg.push_str(&m.to_string());
        msg.push('\n');
    }
    panic!("{msg}");
}

#[test]
fn q4k_matches_ggml_bitwise() {
    let run = run_format(KQuantFormat::Q4K);
    report(KQuantFormat::Q4K, &run);
    assert_coverage(KQuantFormat::Q4K, &run.coverage);
    assert_no_mismatches(KQuantFormat::Q4K, &run);
    assert_eq!(
        run.signed_zero_only, 0,
        "q4k has no bias reassociation, so it has no signed-zero exemption"
    );
}

#[test]
fn q5k_matches_ggml_bitwise() {
    let run = run_format(KQuantFormat::Q5K);
    report(KQuantFormat::Q5K, &run);
    assert_coverage(KQuantFormat::Q5K, &run.coverage);
    assert_no_mismatches(KQuantFormat::Q5K, &run);
    assert_eq!(
        run.signed_zero_only, 0,
        "q5k has no bias reassociation, so it has no signed-zero exemption"
    );
}

/// Q6_K is the one format compared zero-sign-insensitively, and only at zero
/// magnitude. See the module header: the contract folds ggml's integer `-32`
/// offset into `bias = -32 * scale`, and `x + (-x)` is `+0.0` in IEEE-754
/// round-to-nearest, so a negative sub-scale times a zero code reads `-0.0`
/// from ggml and `+0.0` from MLX. Every other bit must match exactly.
#[test]
fn q6k_matches_ggml_except_for_the_sign_of_zero() {
    let run = run_format(KQuantFormat::Q6K);
    report(KQuantFormat::Q6K, &run);
    assert_coverage(KQuantFormat::Q6K, &run.coverage);
    assert_no_mismatches(KQuantFormat::Q6K, &run);
    assert!(
        run.signed_zero_only > 0,
        "q6k: the signed-zero exemption was never exercised, so the case set \
         never produced a negative sub-scale next to a zero code"
    );
    // The folded form can only ever emit +0.0 (`scale*code + (-32*scale)` is
    // `x + (-x)` at the zero crossing), so every -0.0 ggml produces must land
    // in the exemption. Any left over would mean MLX emitted -0.0 too, which
    // this decode cannot do.
    assert_eq!(
        run.coverage.ref_negative_zeros, run.signed_zero_only,
        "q6k: ggml produced {} negative zeros but {} were absorbed by the exemption",
        run.coverage.ref_negative_zeros, run.signed_zero_only
    );
}

/// Q6_K's four interleave positions plus the half-block boundary, isolated.
///
/// A super-block is two 128-value halves. Inside a half, group `g in [0,4)`
/// lives in `ql` nibble `g/2` of plane `g%2` and in `qh` bit-pair `g`. The
/// general cases above hit all four every time; this one puts a distinct code
/// at each so a swapped pair localises here instead of showing up as 200k
/// scattered diffs.
#[test]
fn q6k_interleave_positions_and_half_block_boundary() {
    pin_cpu_stream();
    let format = KQuantFormat::Q6K;
    let mut buf = BlockBuf::zeroed(format.block_bytes());
    {
        let blk = buf.bytes_mut();
        // d = 1.0 so the decoded value is exactly sub_scale * (code - 32)
        blk[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        for j in 0..16 {
            blk[192 + j] = (j as i8 + 1) as u8;
        }
        // one distinct code per interleave position, then the half boundary
        for (v, code) in [
            (0usize, 63u32),
            (32, 1),
            (64, 62),
            (96, 2),
            (127, 33),
            (128, 31),
        ] {
            q6k_set_code(blk, v, code);
        }
    }
    let reference = ggml_decode_row(format, buf.bytes(), QK_K);
    // The generator must actually have put six distinguishable values in place;
    // otherwise "MLX agrees" would be vacuous.
    for (v, code) in [
        (0usize, 63u32),
        (32, 1),
        (64, 62),
        (96, 2),
        (127, 33),
        (128, 31),
    ] {
        // sub-scale j owns [16j, 16j+16) and was written as j + 1
        let sub_scale = (v / 16) as f32 + 1.0;
        let want = sub_scale * (code as f32 - 32.0);
        assert_eq!(
            reference[v], want,
            "generator failed to place code {code} at v={v}"
        );
        assert_eq!(q6k_code(buf.bytes(), v), code, "de-swizzle lost v={v}");
    }
    let arrays = repack_kquant(format, buf.bytes(), 1, QK_K).expect("repack");
    let got = mlx_decode(format, &arrays, 1, QK_K);
    for v in 0..QK_K {
        if reference[v] == 0.0 && got[v] == 0.0 {
            continue; // documented signed-zero exemption
        }
        assert_eq!(
            reference[v].to_bits(),
            got[v].to_bits(),
            "v={v}: ggml {} != mlx {}",
            reference[v],
            got[v]
        );
    }
}

/// A DIVERGENCE, not an equivalence: Q6_K with a non-finite super-scale.
///
/// The folded bias is `-32 * scale`, so at `d = inf` every value in the
/// super-block is `inf * code + -inf` and reads NaN. ggml computes
/// `(d * sub_scale) * (code - 32)` and keeps a signed infinity for every code
/// but 32, where its own `inf * 0` is NaN too.
///
/// No GGUF can carry this: the quantizer derives `d` from finite weights. It is
/// pinned so it stays a known divergence rather than being rediscovered as a
/// mystery. A NaN `d` is asserted equal on both sides, which together with the
/// exhaustive finite sweep covers every `d` a binary16 can hold.
///
/// If this test fails the fold changed, and the note in
/// `crates/mlx-sys/mlx/mlx/backend/cpu/quantized.cpp` must change with it.
#[test]
fn q6k_non_finite_super_scale_diverges_from_ggml() {
    pin_cpu_stream();
    let format = KQuantFormat::Q6K;

    // d = +inf, every sub-scale non-zero and both signs represented, codes
    // sweeping 0..64 so 32 (ggml's own NaN) and both rails are in every run.
    let mut buf = BlockBuf::zeroed(format.block_bytes());
    {
        let blk = buf.bytes_mut();
        blk[208..210].copy_from_slice(&0x7c00u16.to_le_bytes());
        for j in 0..16 {
            let mag = j as i8 + 1;
            let sc = if j % 2 == 0 { mag } else { -mag };
            blk[192 + j] = sc as u8;
        }
        for v in 0..QK_K {
            q6k_set_code(blk, v, (v % 64) as u32);
        }
    }
    let reference = ggml_decode_row(format, buf.bytes(), QK_K);
    let arrays = repack_kquant(format, buf.bytes(), 1, QK_K).expect("repack");
    let got = mlx_decode(format, &arrays, 1, QK_K);

    let mut ref_pos_inf = 0u32;
    let mut ref_neg_inf = 0u32;
    for v in 0..QK_K {
        let code = (v % 64) as i32;
        let sub_scale = v / 16 + 1;
        let sub_sign = if (v / 16) % 2 == 0 { 1i32 } else { -1 };
        assert_eq!(
            q6k_code(buf.bytes(), v),
            code as u32,
            "generator failed to place code {code} at v={v}"
        );
        assert!(
            got[v].is_nan(),
            "v={v} code={code} sub_scale={}: the folded bias no longer collapses \
             an infinite d to NaN, mlx gave {}",
            sub_sign * sub_scale as i32,
            got[v]
        );
        if code == 32 {
            assert!(reference[v].is_nan(), "v={v}: ggml inf * 0 is not NaN");
            continue;
        }
        let want_positive = sub_sign * (code - 32) > 0;
        assert!(
            reference[v].is_infinite() && reference[v].is_sign_positive() == want_positive,
            "v={v} code={code}: ggml gave {} where an infinity of sign {} was \
             expected — the reference decoder changed",
            reference[v],
            if want_positive { '+' } else { '-' }
        );
        if want_positive {
            ref_pos_inf += 1;
        } else {
            ref_neg_inf += 1;
        }
    }
    assert!(
        ref_pos_inf > 0 && ref_neg_inf > 0,
        "the case never produced both infinity signs: +{ref_pos_inf} -{ref_neg_inf}"
    );
    println!("q6k d=+inf: mlx NaN everywhere, ggml +inf x{ref_pos_inf} -inf x{ref_neg_inf}");

    // A NaN d is the other non-finite super-scale, and there the two agree.
    let mut nan_buf = BlockBuf::zeroed(format.block_bytes());
    {
        let blk = nan_buf.bytes_mut();
        blk[208..210].copy_from_slice(&0x7e00u16.to_le_bytes());
        for j in 0..16 {
            blk[192 + j] = (j as i8 + 1) as u8;
        }
        for v in 0..QK_K {
            q6k_set_code(blk, v, (v % 64) as u32);
        }
    }
    let nan_reference = ggml_decode_row(format, nan_buf.bytes(), QK_K);
    let nan_arrays = repack_kquant(format, nan_buf.bytes(), 1, QK_K).expect("repack");
    let nan_got = mlx_decode(format, &nan_arrays, 1, QK_K);
    for v in 0..QK_K {
        assert!(
            nan_reference[v].is_nan() && nan_got[v].is_nan(),
            "v={v}: a NaN d must give NaN on both sides, got ggml {} mlx {}",
            nan_reference[v],
            nan_got[v]
        );
    }
}

/// Q4_K / Q5_K: both `get_scale_min_k4` branches with the hard half of `j >= 4`
/// actually loaded, plus a code that crosses a uint32 word.
///
/// `j >= 4` splices two bits out of the top of `q[j-4]` / `q[j]` into bits 4-5
/// of the result, so a sub-scale in `[16, 63]` at `j >= 4` is the only thing
/// that exercises it. A Q4_K code can never straddle a word (4 divides 32); a
/// Q5_K code at `v = 6` occupies bits 30..35 and does.
#[test]
fn q45k_scale_min_k4_branches_and_word_straddling_code() {
    pin_cpu_stream();
    for format in [KQuantFormat::Q4K, KQuantFormat::Q5K] {
        let max_code = (1u32 << format.bits()) - 1;
        let mut buf = BlockBuf::zeroed(format.block_bytes());
        // j < 4 keeps its values inside the low nibble; j >= 4 needs the splice
        let ls: [u8; 8] = [0, 5, 15, 3, 63, 48, 16, 33];
        let lm: [u8; 8] = [1, 2, 14, 15, 62, 17, 63, 32];
        {
            let blk = buf.bytes_mut();
            blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
            blk[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
            ggml_pack_scale_min_k4(&mut blk[4..16], &ls, &lm);
            for v in 0..QK_K {
                let code = (v as u32 * 7 + 1) % (max_code + 1);
                if format == KQuantFormat::Q4K {
                    q4k_set_code(blk, v, code);
                } else {
                    q5k_set_code(blk, v, code);
                }
            }
        }
        // the packed field must round-trip through ggml's unpacker, including
        // the spliced high bits
        for j in 0..8 {
            let (sc, m) = ggml_get_scale_min_k4(j, &buf.bytes()[4..16]);
            assert_eq!((sc, m), (ls[j], lm[j]), "{:?} j={j}", format);
            assert_eq!(
                get_scale_min_k4(j, &buf.bytes()[4..16]),
                (sc, m),
                "{:?}: the Rust get_scale_min_k4 drifted at j={j}",
                format
            );
            if j >= 4 {
                assert!(
                    ls[j] > 0x0f || lm[j] > 0x0f,
                    "j={j} does not exercise the splice"
                );
            }
        }

        let arrays = repack_kquant(format, buf.bytes(), 1, QK_K).expect("repack");
        // a 5-bit code at v=6 spans bits 30..35 and crosses word 0 -> 1
        if format == KQuantFormat::Q5K {
            let v = 6usize;
            let bit0 = v * 5;
            assert_ne!(bit0 / 32, (bit0 + 4) / 32, "v=6 is not a straddling code");
            assert_eq!(
                extract_code(&arrays.weight, v, 5),
                q5k_code(buf.bytes(), v),
                "q5k: the word-straddling code at v=6 was mis-packed"
            );
        } else {
            for v in 0..QK_K {
                let bit0 = v * 4;
                assert_eq!(
                    bit0 / 32,
                    (bit0 + 3) / 32,
                    "a 4-bit code straddled at v={v}"
                );
            }
        }

        let reference = ggml_decode_row(format, buf.bytes(), QK_K);
        let got = mlx_decode(format, &arrays, 1, QK_K);
        for v in 0..QK_K {
            assert_eq!(
                reference[v].to_bits(),
                got[v].to_bits(),
                "{:?} v={v}: ggml {} != mlx {}",
                format,
                reference[v],
                got[v]
            );
        }
    }
}

/// The Rust `get_scale_min_k4` in `gguf_kquant.rs` is a transliteration of a
/// ggml function that lives in a production import path, so it is swept against
/// the vendored C over every input byte combination each branch can observe.
/// `j < 4` reads `q[j]` and `q[j+4]`; `j >= 4` reads `q[j+4]`, `q[j-4]` and
/// `q[j]`. Every other byte is irrelevant, so this is exhaustive, not sampled.
#[test]
fn rust_get_scale_min_k4_matches_ggml_exhaustively() {
    let mut q = [0u8; 12];
    let mut checked = 0u64;
    for j in 0..4usize {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                q.fill(0);
                q[j] = a;
                q[j + 4] = b;
                assert_eq!(
                    get_scale_min_k4(j, &q),
                    ggml_get_scale_min_k4(j, &q),
                    "j={j} q[{j}]={a} q[{}]={b}",
                    j + 4
                );
                checked += 1;
            }
        }
    }
    for j in 4..8usize {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                for c in 0..=255u8 {
                    q.fill(0);
                    q[j + 4] = a;
                    q[j - 4] = b;
                    q[j] = c;
                    assert_eq!(
                        get_scale_min_k4(j, &q),
                        ggml_get_scale_min_k4(j, &q),
                        "j={j} q[{}]={a} q[{}]={b} q[{j}]={c}",
                        j + 4,
                        j - 4
                    );
                    checked += 1;
                }
            }
        }
    }
    println!("get_scale_min_k4: {checked} exhaustive combinations, all equal");
}

// ---------------------------------------------------------------------------
// Provenance guard — the in-tree port of the scratchpad's check_verbatim.py
// ---------------------------------------------------------------------------

/// The four ggml spans the vendored reference must reproduce verbatim.
const SPANS: [(&str, usize, usize); 4] = [
    ("get_scale_min_k4", 880, 887),
    ("dequantize_row_q4_K", 1529, 1551),
    ("dequantize_row_q5_K", 1731, 1756),
    ("dequantize_row_q6_K", 1939, 1968),
];

/// The three sanctioned edits, applied to both sides so only real drift shows.
///
/// * `get_scale_min_k4` loses `static inline` and gains an `ggml_ref_` prefix so
///   the repacker can reuse ggml's own sub-scale unpacking;
/// * `GGML_FP16_TO_FP32` becomes `ggml_ref_fp16_to_fp32`, the same exact
///   widening (`ggml-impl.h`'s `__ARM_NEON` `__fp16` round-trip);
/// * `GGML_RESTRICT` stays as written — the macro is redefined locally.
fn canon(line: &str) -> String {
    line.trim_end()
        .replace(
            "static inline void get_scale_min_k4",
            "void get_scale_min_k4",
        )
        .replace(
            "static inline void ggml_ref_get_scale_min_k4",
            "void get_scale_min_k4",
        )
        .replace("void ggml_ref_get_scale_min_k4", "void get_scale_min_k4")
        .replace("GGML_FP16_TO_FP32", "FP16")
        .replace("ggml_ref_fp16_to_fp32", "FP16")
}

fn vendor_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/ggml")
}

/// Upstream lines for `span`, either from the checked-in anchor or, when
/// `GGML_QUANTS_C` points at a real llama.cpp checkout, straight out of it.
fn upstream_span(name: &str, first: usize, last: usize) -> Vec<String> {
    if let Some(path) = std::env::var_os("GGML_QUANTS_C") {
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("GGML_QUANTS_C={path:?}: {e}"));
        let lines: Vec<&str> = src.split('\n').collect();
        assert!(
            lines.len() >= last,
            "GGML_QUANTS_C={path:?} has {} lines, need {last}",
            lines.len()
        );
        return lines[first - 1..last].iter().map(|l| canon(l)).collect();
    }
    let anchor = vendor_dir().join("ggml_quants_upstream.inc");
    let text =
        std::fs::read_to_string(&anchor).unwrap_or_else(|e| panic!("{}: {e}", anchor.display()));
    let marker = format!("// @@ {name} ggml-quants.c:{first}-{last}");
    let mut out = Vec::new();
    let mut inside = false;
    for line in text.split('\n') {
        if line.starts_with("// @@ ") {
            if inside {
                break;
            }
            inside = line == marker;
            continue;
        }
        if inside {
            out.push(canon(line));
        }
    }
    assert!(!out.is_empty(), "anchor is missing the span for {name}");
    // trailing blank line between spans
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    assert_eq!(
        out.len(),
        last - first + 1,
        "anchor span {name} has {} lines, upstream span is {}",
        out.len(),
        last - first + 1
    );
    out
}

/// A reference that silently drifts from ggml is worse than no reference. This
/// diffs the compiled-in decoders against the upstream text line for line,
/// modulo the three edits `canon` normalises away.
#[test]
fn vendored_ggml_reference_is_verbatim() {
    let mine_path = vendor_dir().join("ggml_kquant_ref.c");
    let mine_src = std::fs::read_to_string(&mine_path)
        .unwrap_or_else(|e| panic!("{}: {e}", mine_path.display()));
    let mine: Vec<String> = mine_src.split('\n').map(canon).collect();

    for (name, first, last) in SPANS {
        let upstream = upstream_span(name, first, last);
        let start = mine
            .iter()
            .position(|l| *l == upstream[0])
            .unwrap_or_else(|| {
                panic!(
                    "{} does not contain {name} (ggml-quants.c:{first})",
                    mine_path.display()
                )
            });
        let got = &mine[start..(start + upstream.len()).min(mine.len())];
        if got == upstream.as_slice() {
            println!("  verbatim ok: {name:24} ggml-quants.c:{first}-{last}");
            continue;
        }
        let mut msg = format!("VENDORED ggml CODE HAS DRIFTED in {name}\n");
        for (i, (want, have)) in upstream.iter().zip(got.iter()).enumerate() {
            if want != have {
                msg.push_str(&format!("  ggml-quants.c:{}\n", first + i));
                msg.push_str(&format!("    ggml: {want:?}\n"));
                msg.push_str(&format!("    ours: {have:?}\n"));
            }
        }
        if got.len() != upstream.len() {
            msg.push_str(&format!(
                "  span truncated: {} lines vendored, {} upstream\n",
                got.len(),
                upstream.len()
            ));
        }
        panic!("{msg}");
    }
}
