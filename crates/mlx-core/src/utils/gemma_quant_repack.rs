//! Lossless repack of Google "gemma-QAT" per-output-channel **symmetric**
//! packed weights into MLX group-wise **affine** quantized form.
//!
//! ## Scope
//!
//! This helper handles tensors with **one scale per output row**: linear
//! projections, `lm_head`, and the vocabulary `embed_tokens` embedding.
//! `weight_scale` has shape `[out, 1]` (one f32 per output row, no zero-point).
//! Dequant is `w = q_signed * weight_scale[row]`.
//!
//! Tensors with **per-group** scales — such as the per-layer embedding
//! `embed_tokens_per_layer` whose scale shape is `[vocab, num_groups]` (one
//! scale per 256-element block) — are NOT in scope for this helper and are
//! handled by a separate conversion path.
//!
//! ## Mapping onto MLX affine
//!
//! The unsigned stored value (the raw nibble/crumb, BEFORE subtracting the
//! symmetric offset 2^(bits-1)) is exactly MLX's `q_unsigned ∈ [0, 2^bits-1]`.
//! So the per-output-channel symmetric form maps losslessly onto MLX affine:
//!
//! ```text
//! MLX affine:  w[o, c] = q_unsigned[o,c] * scales[o,g] + biases[o,g]   (g = c / group_size)
//! Google:      w[o, c] = (q_unsigned[o,c] - 2^(bits-1)) * weight_scale[o]
//! ⇒
//!   pack the RAW nibble/crumb (no subtraction) into MLX uint32 layout
//!   scales[o, g] = weight_scale[o]                    (broadcast per-row scale to every group)
//!   biases[o, g] = -(2^(bits-1)) * weight_scale[o]    (constant per row)
//! ```
//!
//! This mirrors the GGUF Q4_0 → MLX affine repack in [`super::gguf`], which
//! packs 8×4-bit per u32 little-endian with `bias = -8 * scale`. Here the
//! bit-width and packed-input order are generalized to Google's 2/4-bit layout
//! (low bits first along the input dim).

/// Lossless repack of Google gemma-QAT per-output-channel symmetric weights
/// (2 or 4 bit) into MLX group-wise affine form.
///
/// # Arguments
/// * `packed` — row-major `[out, in/(8/bits)]` Google `.weight` bytes
/// * `weight_scale` — per-row scale, length == `out_features`
/// * `out_features` — number of output channels (rows)
/// * `in_features` — number of input channels (logical columns per row)
/// * `bits` — 2 or 4
/// * `group_size` — MLX affine group size (e.g. 64 for linear projections)
///
/// # Returns
/// `(weight, scales, biases)` as raw Vecs:
/// * `weight` — uint32, row-major `[out, in*bits/32]`
/// * `scales` — f32, row-major `[out, in/group_size]`
/// * `biases` — f32, row-major `[out, in/group_size]`
///
/// # Panics
/// Panics if the inputs are inconsistent (wrong `bits`, mismatched lengths, or
/// dimensions that are not divisible by the packing/group constraints). These
/// are programmer errors in the caller (the converter), not runtime data.
pub fn repack_symmetric_to_mlx_affine(
    packed: &[u8],
    weight_scale: &[f32],
    out_features: usize,
    in_features: usize,
    bits: u32,
    group_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    assert!(bits == 2 || bits == 4, "bits must be 2 or 4, got {bits}");
    assert_eq!(
        weight_scale.len(),
        out_features,
        "weight_scale length must equal out_features"
    );

    let values_per_byte = 8 / bits as usize; // 4-bit → 2, 2-bit → 4
    let values_per_u32 = 32 / bits as usize; // 4-bit → 8, 2-bit → 16

    assert_eq!(
        in_features % values_per_byte,
        0,
        "in_features ({in_features}) must be divisible by values_per_byte ({values_per_byte})"
    );
    assert_eq!(
        in_features % values_per_u32,
        0,
        "in_features ({in_features}) must be divisible by values_per_u32 ({values_per_u32})"
    );
    assert_eq!(
        in_features % group_size,
        0,
        "in_features ({in_features}) must be divisible by group_size ({group_size})"
    );

    let bytes_per_row = in_features / values_per_byte;
    assert_eq!(
        packed.len(),
        out_features * bytes_per_row,
        "packed length ({}) must equal out_features * bytes_per_row ({})",
        packed.len(),
        out_features * bytes_per_row
    );

    let u32_per_row = in_features * bits as usize / 32;
    let groups_per_row = in_features / group_size;
    // Symmetric offset folded entirely into the affine bias.
    let zero_point = (1u32 << (bits - 1)) as f32; // 8 @ 4-bit, 2 @ 2-bit
    let nibble_mask: u32 = (1u32 << bits) - 1; // 0xF @ 4-bit, 0x3 @ 2-bit

    let mut weight = vec![0u32; out_features * u32_per_row];
    let mut scales = vec![0f32; out_features * groups_per_row];
    let mut biases = vec![0f32; out_features * groups_per_row];

    for o in 0..out_features {
        let s = weight_scale[o];
        let bias = -zero_point * s;
        for g in 0..groups_per_row {
            scales[o * groups_per_row + g] = s;
            biases[o * groups_per_row + g] = bias;
        }

        let row_bytes = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
        let row_out = &mut weight[o * u32_per_row..(o + 1) * u32_per_row];

        // Walk the logical input dim in order, emitting the RAW unsigned
        // nibble/crumb (low bits first within each byte, matching Google's
        // unpack order), and pack `values_per_u32` of them little-endian into
        // each u32 word (value j in bits [bits*j .. bits*j+bits-1]).
        for col in 0..in_features {
            let byte = row_bytes[col / values_per_byte];
            let slot_in_byte = col % values_per_byte;
            let q_unsigned = (u32::from(byte) >> (slot_in_byte * bits as usize)) & nibble_mask;

            let word = col / values_per_u32;
            let slot_in_word = col % values_per_u32;
            row_out[word] |= q_unsigned << (slot_in_word * bits as usize);
        }
    }

    (weight, scales, biases)
}

/// Lossless repack of Google gemma-QAT **per-group** symmetric weights into MLX
/// group-wise affine form, replicating each source group's scale across the
/// `src_group_size / dst_group_size` MLX sub-groups it spans.
///
/// This is the per-group sibling of [`repack_symmetric_to_mlx_affine`]. It exists
/// for the per-layer embedding `embed_tokens_per_layer`, whose `embedding_scale`
/// has shape `[rows, num_src_groups]` (one f32 per `src_group_size`-wide block of
/// the logical input dim) rather than one scale per row. MLX affine only accepts
/// `group_size ∈ {32, 64, 128}`, so a 256-wide source group is emitted as two
/// adjacent 128-wide MLX groups carrying the same (constant-within-the-block)
/// scale — lossless.
///
/// # Arguments
/// * `packed` — row-major `[rows, in/(8/bits)]` Google packed bytes
/// * `group_scale` — row-major `[rows, num_src_groups]` per-source-group scale
/// * `rows` — number of output rows (vocab entries for the PLE embedding)
/// * `in_features` — logical input channels per row
/// * `bits` — 2 or 4
/// * `src_group_size` — width of a source scale block (e.g. 256)
/// * `dst_group_size` — MLX affine group size (e.g. 128); must divide `src_group_size`
///
/// # Returns
/// `(weight, scales, biases)` as raw Vecs:
/// * `weight` — uint32, row-major `[rows, in*bits/32]`
/// * `scales` — f32, row-major `[rows, in/dst_group_size]`
/// * `biases` — f32, row-major `[rows, in/dst_group_size]`
///
/// # Panics
/// Panics on inconsistent inputs (wrong `bits`, mismatched lengths, or dimensions
/// that violate the packing / group constraints). These are programmer errors in
/// the caller (the converter), not runtime data.
pub fn repack_symmetric_per_group_to_mlx_affine(
    packed: &[u8],
    group_scale: &[f32],
    rows: usize,
    in_features: usize,
    bits: u32,
    src_group_size: usize,
    dst_group_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    assert!(bits == 2 || bits == 4, "bits must be 2 or 4, got {bits}");
    assert!(dst_group_size > 0, "dst_group_size must be > 0");
    assert_eq!(
        src_group_size % dst_group_size,
        0,
        "src_group_size ({src_group_size}) must be divisible by dst_group_size ({dst_group_size})"
    );

    let values_per_byte = 8 / bits as usize; // 4-bit → 2, 2-bit → 4
    let values_per_u32 = 32 / bits as usize; // 4-bit → 8, 2-bit → 16

    assert_eq!(
        in_features % values_per_byte,
        0,
        "in_features ({in_features}) must be divisible by values_per_byte ({values_per_byte})"
    );
    assert_eq!(
        in_features % values_per_u32,
        0,
        "in_features ({in_features}) must be divisible by values_per_u32 ({values_per_u32})"
    );
    assert_eq!(
        in_features % src_group_size,
        0,
        "in_features ({in_features}) must be divisible by src_group_size ({src_group_size})"
    );
    assert_eq!(
        in_features % dst_group_size,
        0,
        "in_features ({in_features}) must be divisible by dst_group_size ({dst_group_size})"
    );

    let num_src_groups = in_features / src_group_size;
    assert_eq!(
        group_scale.len(),
        rows * num_src_groups,
        "group_scale length ({}) must equal rows * num_src_groups ({})",
        group_scale.len(),
        rows * num_src_groups
    );

    let bytes_per_row = in_features / values_per_byte;
    assert_eq!(
        packed.len(),
        rows * bytes_per_row,
        "packed length ({}) must equal rows * bytes_per_row ({})",
        packed.len(),
        rows * bytes_per_row
    );

    let u32_per_row = in_features * bits as usize / 32;
    let dst_groups_per_row = in_features / dst_group_size;
    let sub_per_src = src_group_size / dst_group_size; // 256/128 = 2
    let zero_point = (1u32 << (bits - 1)) as f32; // 8 @ 4-bit, 2 @ 2-bit
    let nibble_mask: u32 = (1u32 << bits) - 1;

    let mut weight = vec![0u32; rows * u32_per_row];
    let mut scales = vec![0f32; rows * dst_groups_per_row];
    let mut biases = vec![0f32; rows * dst_groups_per_row];

    for o in 0..rows {
        // Replicate each source-group scale across the dst sub-groups it spans.
        for dg in 0..dst_groups_per_row {
            let src_g = dg / sub_per_src;
            let s = group_scale[o * num_src_groups + src_g];
            scales[o * dst_groups_per_row + dg] = s;
            biases[o * dst_groups_per_row + dg] = -zero_point * s;
        }

        let row_bytes = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
        let row_out = &mut weight[o * u32_per_row..(o + 1) * u32_per_row];

        for col in 0..in_features {
            let byte = row_bytes[col / values_per_byte];
            let slot_in_byte = col % values_per_byte;
            let q_unsigned = (u32::from(byte) >> (slot_in_byte * bits as usize)) & nibble_mask;

            let word = col / values_per_u32;
            let slot_in_word = col % values_per_u32;
            row_out[word] |= q_unsigned << (slot_in_word * bits as usize);
        }
    }

    (weight, scales, biases)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::MxArray;
    use std::ffi::CString;

    /// Dequantize an MLX affine triplet via the FFI and return a flat f32 Vec.
    fn mlx_affine_dequant(
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        out_features: usize,
        in_features: usize,
        bits: u32,
        group_size: usize,
    ) -> Vec<f32> {
        let u32_per_row = in_features * bits as usize / 32;
        let groups_per_row = in_features / group_size;

        let w = MxArray::from_uint32(weight, &[out_features as i64, u32_per_row as i64])
            .expect("from_uint32");
        let s = MxArray::from_float32(scales, &[out_features as i64, groups_per_row as i64])
            .expect("from_float32 scales");
        let b = MxArray::from_float32(biases, &[out_features as i64, groups_per_row as i64])
            .expect("from_float32 biases");

        let mode = CString::new("affine").unwrap();
        let handle = unsafe {
            mlx_sys::mlx_dequantize(
                w.as_raw_ptr(),
                s.as_raw_ptr(),
                b.as_raw_ptr(),
                group_size as i32,
                bits as i32,
                -1,
                mode.as_ptr(),
            )
        };
        assert!(!handle.is_null(), "mlx_dequantize returned null");
        let dequant = MxArray::from_handle(handle, "dequant").expect("from_handle");
        dequant.eval();
        dequant.to_float32().expect("to_float32").to_vec()
    }

    /// Byte-pack a row of `q_signed` values the way Google does.
    /// 4-bit: 2 values/byte, low nibble first, stored as (q + 8) & 0xF.
    /// 2-bit: 4 crumbs/byte, low crumb first, stored as (q + 2) & 0x3.
    fn google_byte_pack(q_signed: &[i32], bits: u32) -> Vec<u8> {
        let values_per_byte = 8 / bits as usize;
        let offset = 1i32 << (bits - 1);
        let mask = (1u32 << bits) - 1;
        assert_eq!(q_signed.len() % values_per_byte, 0);
        let mut out = Vec::with_capacity(q_signed.len() / values_per_byte);
        for chunk in q_signed.chunks(values_per_byte) {
            let mut byte = 0u8;
            for (slot, &q) in chunk.iter().enumerate() {
                let raw = ((q + offset) as u32) & mask;
                byte |= (raw as u8) << (slot * bits as usize);
            }
            out.push(byte);
        }
        out
    }

    /// Test A: synthetic round-trip for 4-bit and 2-bit, including the
    /// low/high nibble-order check (0xF7 → lo=-1, hi=+7 at 4-bit).
    #[test]
    fn test_synthetic_roundtrip() {
        let out_features = 4;
        let in_features = 128;
        let group_size = 64;

        // ── 4-bit ──────────────────────────────────────────────────────────
        {
            let bits = 4;
            // Deterministic q_signed pattern in [-8, 7], distinct per row.
            let mut q_signed = vec![0i32; out_features * in_features];
            for o in 0..out_features {
                for c in 0..in_features {
                    let v = ((o * 13 + c * 7) % 16) as i32 - 8; // [-8, 7]
                    q_signed[o * in_features + c] = v;
                }
            }
            // Force the nibble-order witness: 0xF7 → lo=7→-1, hi=15→+7.
            // packed byte for (col0=lo, col1=hi) must be 0xF7.
            q_signed[0] = -1; // lo nibble of byte 0
            q_signed[1] = 7; // hi nibble of byte 0
            let mut scales_per_row = vec![0f32; out_features];
            for (o, s) in scales_per_row.iter_mut().enumerate() {
                *s = 0.001 * (o as f32 + 1.0);
            }

            // Byte-pack row-major.
            let mut packed = Vec::new();
            for o in 0..out_features {
                packed.extend(google_byte_pack(
                    &q_signed[o * in_features..(o + 1) * in_features],
                    bits,
                ));
            }
            // Witness the nibble order explicitly.
            assert_eq!(packed[0], 0xF7, "byte 0 must be 0xF7 (lo=-1, hi=+7)");

            let (weight, scales, biases) = repack_symmetric_to_mlx_affine(
                &packed,
                &scales_per_row,
                out_features,
                in_features,
                bits,
                group_size,
            );

            // ── CPU oracle: unpack weight back to q_unsigned, recompute dequant
            //    without touching MLX, and verify against q_signed * scale.
            {
                let values_per_u32 = 32 / bits as usize;
                let nibble_mask = (1u32 << bits) - 1;
                let zero_point = (1u32 << (bits - 1)) as f64;
                let groups_per_row = in_features / group_size;
                for o in 0..out_features {
                    for c in 0..in_features {
                        let word = c / values_per_u32;
                        let slot = c % values_per_u32;
                        let q_u = ((weight[o * (in_features * bits as usize / 32) + word]
                            >> (slot * bits as usize))
                            & nibble_mask) as f64;
                        let g = c / group_size;
                        let s = scales[o * groups_per_row + g] as f64;
                        let b = biases[o * groups_per_row + g] as f64;
                        let got_cpu = q_u * s + b;
                        let expected =
                            q_signed[o * in_features + c] as f64 * scales_per_row[o] as f64;
                        assert!(
                            (got_cpu - expected).abs() < 1e-6,
                            "4-bit CPU oracle mismatch at [{o},{c}]: got {got_cpu}, expected {expected} (q_u={q_u}, zero_point={zero_point})"
                        );
                    }
                }
            }

            let dequant = mlx_affine_dequant(
                &weight,
                &scales,
                &biases,
                out_features,
                in_features,
                bits,
                group_size,
            );

            for o in 0..out_features {
                for c in 0..in_features {
                    let expected = q_signed[o * in_features + c] as f32 * scales_per_row[o];
                    let got = dequant[o * in_features + c];
                    assert!(
                        (got - expected).abs() < 1e-6,
                        "4-bit mismatch at [{o},{c}]: got {got}, expected {expected}"
                    );
                }
            }
            // Direct nibble-order witness in the dequantized output.
            assert!((dequant[0] + scales_per_row[0]).abs() < 1e-6);
            assert!((dequant[1] - (7.0 * scales_per_row[0])).abs() < 1e-6);
        }

        // ── 2-bit ──────────────────────────────────────────────────────────
        {
            let bits = 2;
            let mut q_signed = vec![0i32; out_features * in_features];
            for o in 0..out_features {
                for c in 0..in_features {
                    let v = ((o * 3 + c) % 4) as i32 - 2; // [-2, 1]
                    q_signed[o * in_features + c] = v;
                }
            }
            let mut scales_per_row = vec![0f32; out_features];
            for (o, s) in scales_per_row.iter_mut().enumerate() {
                *s = 0.01 * (o as f32 + 1.0);
            }

            let mut packed = Vec::new();
            for o in 0..out_features {
                packed.extend(google_byte_pack(
                    &q_signed[o * in_features..(o + 1) * in_features],
                    bits,
                ));
            }

            let (weight, scales, biases) = repack_symmetric_to_mlx_affine(
                &packed,
                &scales_per_row,
                out_features,
                in_features,
                bits,
                group_size,
            );

            // ── CPU oracle for 2-bit ───────────────────────────────────────
            {
                let values_per_u32 = 32 / bits as usize;
                let nibble_mask = (1u32 << bits) - 1;
                let groups_per_row = in_features / group_size;
                for o in 0..out_features {
                    for c in 0..in_features {
                        let word = c / values_per_u32;
                        let slot = c % values_per_u32;
                        let q_u = ((weight[o * (in_features * bits as usize / 32) + word]
                            >> (slot * bits as usize))
                            & nibble_mask) as f64;
                        let g = c / group_size;
                        let s = scales[o * groups_per_row + g] as f64;
                        let b = biases[o * groups_per_row + g] as f64;
                        let got_cpu = q_u * s + b;
                        let expected =
                            q_signed[o * in_features + c] as f64 * scales_per_row[o] as f64;
                        assert!(
                            (got_cpu - expected).abs() < 1e-6,
                            "2-bit CPU oracle mismatch at [{o},{c}]: got {got_cpu}, expected {expected}"
                        );
                    }
                }
            }

            let dequant = mlx_affine_dequant(
                &weight,
                &scales,
                &biases,
                out_features,
                in_features,
                bits,
                group_size,
            );

            for o in 0..out_features {
                for c in 0..in_features {
                    let expected = q_signed[o * in_features + c] as f32 * scales_per_row[o];
                    let got = dequant[o * in_features + c];
                    assert!(
                        (got - expected).abs() < 1e-6,
                        "2-bit mismatch at [{o},{c}]: got {got}, expected {expected}"
                    );
                }
            }
        }
    }

    /// Per-group repack: two adjacent 256-wide source blocks with DIFFERENT
    /// scales must land the correct scale in each of their two 128-wide MLX
    /// sub-groups, and dequant must equal `q_signed * scale`.
    #[test]
    fn test_per_group_roundtrip_different_block_scales() {
        let bits = 4u32;
        let rows = 3;
        let src_group_size = 256;
        let dst_group_size = 128;
        let in_features = 512; // 2 source groups → 4 dst groups
        let num_src_groups = in_features / src_group_size; // 2

        // Distinct q_signed per (row, col) in [-8, 7].
        let mut q_signed = vec![0i32; rows * in_features];
        for o in 0..rows {
            for c in 0..in_features {
                q_signed[o * in_features + c] = ((o * 5 + c * 3) % 16) as i32 - 8;
            }
        }

        // Two source blocks per row with DIFFERENT scales.
        let mut group_scale = vec![0f32; rows * num_src_groups];
        for o in 0..rows {
            group_scale[o * num_src_groups] = 0.002 * (o as f32 + 1.0); // block 0
            group_scale[o * num_src_groups + 1] = 0.05 * (o as f32 + 1.0); // block 1 (different)
        }

        let mut packed = Vec::new();
        for o in 0..rows {
            packed.extend(google_byte_pack(
                &q_signed[o * in_features..(o + 1) * in_features],
                bits,
            ));
        }

        let (weight, scales, biases) = repack_symmetric_per_group_to_mlx_affine(
            &packed,
            &group_scale,
            rows,
            in_features,
            bits,
            src_group_size,
            dst_group_size,
        );

        // Witness scale placement: dst groups 0,1 carry block-0 scale; 2,3 block-1.
        let dst_groups_per_row = in_features / dst_group_size; // 4
        for o in 0..rows {
            let s0 = group_scale[o * num_src_groups];
            let s1 = group_scale[o * num_src_groups + 1];
            assert_eq!(scales[o * dst_groups_per_row], s0);
            assert_eq!(scales[o * dst_groups_per_row + 1], s0);
            assert_eq!(scales[o * dst_groups_per_row + 2], s1);
            assert_eq!(scales[o * dst_groups_per_row + 3], s1);
            assert!((biases[o * dst_groups_per_row] - (-8.0 * s0)).abs() < 1e-9);
            assert!((biases[o * dst_groups_per_row + 3] - (-8.0 * s1)).abs() < 1e-9);
        }

        let dequant = mlx_affine_dequant(
            &weight,
            &scales,
            &biases,
            rows,
            in_features,
            bits,
            dst_group_size,
        );
        for o in 0..rows {
            for c in 0..in_features {
                let src_g = c / src_group_size;
                let expected =
                    q_signed[o * in_features + c] as f32 * group_scale[o * num_src_groups + src_g];
                let got = dequant[o * in_features + c];
                assert!(
                    (got - expected).abs() < 1e-6,
                    "per-group mismatch at [{o},{c}]: got {got}, expected {expected}"
                );
            }
        }
    }

    // ── Real-checkpoint golden cross-check (gated on file presence) ─────────

    const CKPT: &str =
        "/Users/brooklyn/.mlx-node/models/gemma-4-e2b-it-qat-mobile-transformers/model.safetensors";

    struct StTensor {
        dtype: String,
        shape: Vec<usize>,
        data_start: u64, // absolute byte offset of the tensor data in the file
        nbytes: usize,
    }

    /// Minimal safetensors header parse: returns (name → tensor-info).
    fn parse_safetensors_header(path: &str) -> std::collections::HashMap<String, StTensor> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(path).expect("open checkpoint");
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).expect("read header len");
        let header_len = u64::from_le_bytes(len_buf);
        let mut hdr = vec![0u8; header_len as usize];
        f.seek(SeekFrom::Start(8)).unwrap();
        f.read_exact(&mut hdr).expect("read header json");
        let json: serde_json::Value = serde_json::from_slice(&hdr).expect("parse header json");
        let data_base = 8 + header_len;

        let mut map = std::collections::HashMap::new();
        for (k, v) in json.as_object().expect("header object") {
            if k == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap().to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let offs = v["data_offsets"].as_array().unwrap();
            let begin = offs[0].as_u64().unwrap();
            let end = offs[1].as_u64().unwrap();
            map.insert(
                k.clone(),
                StTensor {
                    dtype,
                    shape,
                    data_start: data_base + begin,
                    nbytes: (end - begin) as usize,
                },
            );
        }
        map
    }

    fn read_tensor_bytes(path: &str, t: &StTensor) -> Vec<u8> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(path).expect("open checkpoint");
        f.seek(SeekFrom::Start(t.data_start)).unwrap();
        let mut buf = vec![0u8; t.nbytes];
        f.read_exact(&mut buf).expect("read tensor bytes");
        buf
    }

    fn read_f32_row(packed: &[u8]) -> Vec<f32> {
        packed
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Repack a single row, dequantize it, and compare its first 16 values
    /// against an independently-computed numpy golden.
    fn check_golden_row(
        path: &str,
        header: &std::collections::HashMap<String, StTensor>,
        weight_key: &str,
        scale_key: &str,
        bits: u32,
        group_size: usize,
        row: usize,
        expected_scale: f32,
        expected_bytes_prefix: &[u8],
        golden: &[f32],
    ) {
        let w = header.get(weight_key).expect("weight in header");
        let s = header.get(scale_key).expect("scale in header");
        assert_eq!(w.dtype, "U8");
        assert_eq!(s.dtype, "F32");

        let out_features = w.shape[0];
        let bytes_per_row = w.shape[1];
        let in_features = bytes_per_row * (8 / bits as usize);

        let weight_bytes = read_tensor_bytes(path, w);
        let scale_bytes = read_tensor_bytes(path, s);
        let scales = read_f32_row(&scale_bytes);

        // Validate against the brief's stated row scale and packed prefix.
        assert!(
            (scales[row] - expected_scale).abs() < 1e-9,
            "{weight_key} row {row} scale {} != expected {expected_scale}",
            scales[row]
        );
        let row_bytes = &weight_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
        assert_eq!(
            &row_bytes[..expected_bytes_prefix.len()],
            expected_bytes_prefix,
            "{weight_key} row {row} packed prefix mismatch"
        );

        // Repack the single row in isolation (out_features = 1).
        let (weight, sc, bi) = repack_symmetric_to_mlx_affine(
            row_bytes,
            &[scales[row]],
            1,
            in_features,
            bits,
            group_size,
        );
        let _ = out_features;

        let dequant = mlx_affine_dequant(&weight, &sc, &bi, 1, in_features, bits, group_size);

        for (i, &g) in golden.iter().enumerate() {
            assert!(
                (dequant[i] - g).abs() < 1e-6,
                "{weight_key} row {row} dequant[{i}] = {} != golden {g}",
                dequant[i]
            );
        }
    }

    // The scale and golden literals below are copied verbatim from the
    // independently numpy-computed values in the task brief; keep their full
    // precision (f32 truncates them losslessly for the < 1e-6 comparison).
    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_real_checkpoint_golden() {
        if !std::path::Path::new(CKPT).exists() {
            eprintln!("SKIP test_real_checkpoint_golden: checkpoint not found at {CKPT}");
            return;
        }
        let header = parse_safetensors_header(CKPT);

        // ── 4-bit q_proj ────────────────────────────────────────────────────
        let q_w = "model.language_model.layers.0.self_attn.q_proj.weight";
        let q_s = "model.language_model.layers.0.self_attn.q_proj.weight_scale";

        check_golden_row(
            CKPT,
            &header,
            q_w,
            q_s,
            4,
            64,
            0,
            0.001_944_516_087,
            &[247, 169, 108, 123, 135, 117, 171, 130],
            &[
                -1.94451609e-03,
                1.36116126e-02,
                1.94451609e-03,
                3.88903217e-03,
                7.77806435e-03,
                -3.88903217e-03,
                5.83354826e-03,
                -1.94451609e-03,
                -1.94451609e-03,
                0.0,
                -5.83354826e-03,
                -1.94451609e-03,
                5.83354826e-03,
                3.88903217e-03,
                -1.16670965e-02,
                0.0,
            ],
        );

        check_golden_row(
            CKPT,
            &header,
            q_w,
            q_s,
            4,
            64,
            1,
            0.005_506_917_369,
            &[156, 75, 110, 169, 105, 124, 235, 145],
            &[
                2.20276695e-02,
                5.50691737e-03,
                1.65207521e-02,
                -2.20276695e-02,
                3.30415042e-02,
                -1.10138347e-02,
                5.50691737e-03,
                1.10138347e-02,
                5.50691737e-03,
                -1.10138347e-02,
                2.20276695e-02,
                -5.50691737e-03,
                1.65207521e-02,
                3.30415042e-02,
                -3.85484216e-02,
                5.50691737e-03,
            ],
        );

        // ── 2-bit down_proj ─────────────────────────────────────────────────
        let d_w = "model.language_model.layers.20.mlp.down_proj.weight";
        let d_s = "model.language_model.layers.20.mlp.down_proj.weight_scale";

        check_golden_row(
            CKPT,
            &header,
            d_w,
            d_s,
            2,
            64,
            0,
            0.010_422_741_99,
            &[170, 255, 154, 213, 150, 237, 230, 190],
            &[
                0.0,
                0.0,
                0.0,
                0.0,
                1.04227420e-02,
                1.04227420e-02,
                1.04227420e-02,
                1.04227420e-02,
                0.0,
                0.0,
                -1.04227420e-02,
                0.0,
                -1.04227420e-02,
                -1.04227420e-02,
                -1.04227420e-02,
                1.04227420e-02,
            ],
        );

        check_golden_row(
            CKPT,
            &header,
            d_w,
            d_s,
            2,
            64,
            1,
            0.010_286_430_83,
            &[166, 183, 250, 110, 174, 99, 151, 109],
            &[
                0.0,
                -1.02864308e-02,
                0.0,
                0.0,
                1.02864308e-02,
                -1.02864308e-02,
                1.02864308e-02,
                0.0,
                0.0,
                0.0,
                1.02864308e-02,
                1.02864308e-02,
                0.0,
                1.02864308e-02,
                0.0,
                -1.02864308e-02,
            ],
        );
    }
}
