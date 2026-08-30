use crate::array::MxArray;
use napi::bindgen_prelude::*;

/// Creates a causal attention mask to prevent attention to future positions.
///
/// This creates a boolean mask where TRUE indicates positions to mask out.
/// Currently using UPPER triangular (linds < rinds) which empirically works better
/// than lower triangular, despite being counterintuitive.
///
/// Used in autoregressive models to enforce causality.
///
/// # Arguments
/// * `seq_len` - Sequence length (N)
/// * `offset` - Optional offset for incremental generation (default: 0)
/// * `window_size` - Optional sliding window size for local attention
///
/// # Returns
/// Boolean mask array of shape [seq_len, seq_len]
///
/// # Examples
/// ```no_run
/// # use mlx_core::array::mask::create_causal_mask;
/// // Basic causal mask (offset=0)
/// let mask = create_causal_mask(5, None, None).unwrap();
///
/// // With offset (for incremental generation)
/// let mask = create_causal_mask(3, Some(10), None).unwrap();
/// // Indices: [10, 11, 12] instead of [0, 1, 2]
///
/// // With sliding window (local attention)
/// let mask = create_causal_mask(5, None, Some(2)).unwrap();
/// ```
pub fn create_causal_mask(
    seq_len: i32,
    offset: Option<i32>,
    window_size: Option<i32>,
) -> Result<MxArray> {
    let offset = offset.unwrap_or(0);

    // Create column indices: [0, 1, ..., offset+seq_len-1]
    // This is 'rinds' in mlx-lm - includes ALL positions (cached + new)
    let total_positions = offset + seq_len;
    let rinds = MxArray::arange(0.0, total_positions as f64, Some(1.0), None)?;
    let rinds = rinds.reshape(&[1, total_positions as i64])?; // Shape: [1, offset+seq_len]

    // Create row indices: [offset, offset+1, ..., offset+seq_len-1]
    // This is 'linds' in mlx-lm - only NEW query positions
    let linds = MxArray::arange(offset as f64, (offset + seq_len) as f64, Some(1.0), None)?;
    let linds = linds.reshape(&[seq_len as i64, 1])?; // Shape: [seq_len, 1]

    // Create base causal mask: linds >= rinds (lower triangular + diagonal)
    // IMPORTANT: MLX boolean mask semantics are OPPOSITE to PyTorch!
    // In MLX: TRUE = keep score, FALSE = mask out (-inf)
    // In PyTorch: TRUE = mask out, FALSE = keep
    //
    // For causal masking, we want to:
    // - Keep past positions and self (linds >= rinds) → TRUE in MLX
    // - Mask future positions (linds < rinds) → FALSE in MLX
    let mut mask = linds.greater_equal(&rinds)?;

    // Apply sliding window if specified
    // mask = mask & (linds < rinds + window_size)
    if let Some(window_size) = window_size {
        use napi::Either;
        let window_limit = rinds.add(&MxArray::full(
            &[1, total_positions as i64],
            Either::A(window_size as f64),
            None,
        )?)?;
        let window_mask = linds.less(&window_limit)?;
        mask = mask.logical_and(&window_mask)?;
    }

    Ok(mask)
}
