use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

/// Rotate the last dimension by splitting it in half and concatenating [-x2, x1].
fn rotate_half(x: &MxArray) -> Result<MxArray> {
    let ndim = x.ndim()?;
    let last = x.shape_at(ndim - 1)?;
    let half = last / 2;

    let x1 = slice_last_dim(x, 0, half)?;
    let x2 = slice_last_dim(x, half, last)?;
    let neg_x2 = x2.negative()?;
    MxArray::concatenate(&neg_x2, &x1, -1)
}

/// Slice the last dimension of `x` in the range `[start, stop)`.
fn slice_last_dim(x: &MxArray, start: i64, stop: i64) -> Result<MxArray> {
    let ndim = x.ndim()? as usize;
    let handle = unsafe { mlx_sys::mlx_array_slice_axis(x.as_raw_ptr(), ndim - 1, start, stop) };
    MxArray::from_handle(handle, "slice_last_dim")
}

/// Apply 2D multidimensional RoPE to `inputs`.
///
/// # Arguments
/// - `inputs` — `[B, L, N_heads, head_dim]`
/// - `positions` — `[B, L, 2]` spatial positions
/// - `base_frequency` — RoPE base (default 100.0 for the vision encoder)
pub fn apply_multidimensional_rope(
    inputs: &MxArray,
    positions: &MxArray,
    base_frequency: f64,
) -> Result<MxArray> {
    let head_dim = inputs.shape_at(inputs.ndim()? - 1)?;
    let pos_ndim = positions.ndim()?;
    if pos_ndim != 3 {
        return Err(Error::from_reason(format!(
            "apply_multidimensional_rope: expected positions of rank 3 [B, L, 2], got rank {pos_ndim}"
        )));
    }

    // 2D RoPE: positions is [B, L, 2]
    let ndim = positions.shape_at(pos_ndim - 1)?; // should be 2
    let channels_per_dim = 2 * (head_dim / (2 * ndim));
    let half_per_dim = channels_per_dim / 2;

    let in_dtype = inputs.dtype()?;
    let mut result_parts: Vec<MxArray> = Vec::with_capacity(ndim as usize);

    for d in 0..ndim {
        let x_part = slice_last_dim(inputs, d * channels_per_dim, (d + 1) * channels_per_dim)?;

        let freq_exponents = {
            let exps = MxArray::arange(0.0, half_per_dim as f64, None, Some(DType::Float32))?;
            exps.mul_scalar(2.0 / channels_per_dim as f64)?
        };
        let base = MxArray::scalar_float(base_frequency)?;
        let timescale = base.power(&freq_exponents)?;

        // positions[..., d:d+1]: [B, L, 1]
        let pos_d = slice_last_dim(positions, d, d + 1)?;
        let pos_f32 = pos_d.astype(DType::Float32)?;
        let sinusoid_inp = pos_f32.div(&timescale)?;

        let cos_d = sinusoid_inp.cos()?;
        let sin_d = sinusoid_inp.sin()?;

        // [B, L, half_per_dim] -> [B, L, channels_per_dim]
        let cos_d = MxArray::concatenate(&cos_d, &cos_d, -1)?;
        let sin_d = MxArray::concatenate(&sin_d, &sin_d, -1)?;

        // Cast to inputs dtype and expand for N_heads: [B, L, 1, channels_per_dim]
        let cos_d = cos_d.astype(in_dtype)?.expand_dims(2)?;
        let sin_d = sin_d.astype(in_dtype)?.expand_dims(2)?;

        let rotated = rotate_half(&x_part)?;
        let y_part = x_part.mul(&cos_d)?.add(&rotated.mul(&sin_d)?)?;
        result_parts.push(y_part);
    }

    let refs: Vec<&MxArray> = result_parts.iter().collect();
    MxArray::concatenate_many(refs, Some(-1))
}
