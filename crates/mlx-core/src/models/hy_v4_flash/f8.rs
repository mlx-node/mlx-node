//! F8_E4M3 codes (OCP "fn": bias 7, no inf, 0x7F / 0xFF NaN), one byte per
//! value and no block scale: the Hy4 Engram table, which the PR #32517 file
//! stores as GGUF type 43 and mlx-node's parser reads as MXFP8.

use std::sync::LazyLock;

use napi::{Error, Result};

use crate::utils::gguf::e4m3_to_f32;

static LUT: LazyLock<[f32; 256]> = LazyLock::new(|| std::array::from_fn(|c| e4m3_to_f32(c as u8)));

pub fn decode(code: u8) -> f32 {
    LUT[usize::from(code)]
}

pub fn is_nan_code(code: u8) -> bool {
    code & 0x7F == 0x7F
}

/// Decodes `codes` into `out` (same length); a NaN code fails, naming its
/// index.
pub fn decode_f8_rows(codes: &[u8], out: &mut [f32]) -> Result<()> {
    if codes.len() != out.len() {
        return Err(Error::from_reason(format!(
            "F8_E4M3 decode: {} codes for {} values",
            codes.len(),
            out.len()
        )));
    }
    let lut = &*LUT;
    for (i, (&c, o)) in codes.iter().zip(out.iter_mut()).enumerate() {
        if is_nan_code(c) {
            return Err(Error::from_reason(format!(
                "F8_E4M3 decode: NaN code {c:#04x} at value {i}"
            )));
        }
        *o = lut[usize::from(c)];
    }
    Ok(())
}
