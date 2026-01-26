//! Sequence padding utilities
//!
//! This module provides functions for padding variable-length sequences to uniform length,
//! which is essential for batched processing in transformers.
//!
//! ## Performance
//!
//! These functions perform all padding operations on GPU using MLX native operations.
//! No GPU→CPU data transfers occur - only metadata (sequence lengths) is read.

use napi::bindgen_prelude::*;

use super::{DType, MxArray};

/// Result from padding sequences with masks
pub struct PaddedSequences {
    padded: MxArray,
    masks: MxArray,
}

impl PaddedSequences {
    pub(crate) fn new(padded: MxArray, masks: MxArray) -> Self {
        Self { padded, masks }
    }

    pub fn get_padded(&self) -> Result<MxArray> {
        Ok(self.padded.clone())
    }

    pub fn get_masks(&self) -> Result<MxArray> {
        Ok(self.masks.clone())
    }
}

/// Pad variable-length sequences to uniform length (for integers/tokens)
///
/// Takes a list of 1D sequences and pads them to the maximum length.
/// Returns both the padded sequences and binary masks indicating real vs padded positions.
///
/// # Performance
///
/// This function performs all operations on GPU:
/// - Uses `shape_at()` for O(1) metadata access (no GPU sync)
/// - Uses `MxArray::pad()` for GPU-native padding
/// - Uses `MxArray::stack()` for GPU stacking
/// - No GPU→CPU data transfers occur
///
/// # Arguments
/// * `sequences` - Vector of 1D arrays with variable lengths
/// * `pad_value` - Value to use for padding (default: 0)
///
/// # Returns
/// Object with `padded` (shape: [num_seqs, max_len]) and `masks` (same shape, 1.0 for real tokens, 0.0 for padding)
pub fn pad_sequences(sequences: Vec<&MxArray>, pad_value: i32) -> Result<PaddedSequences> {
    if sequences.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "sequences cannot be empty".to_string(),
        ));
    }

    // 1. Find max length using shape_at (no GPU sync, just metadata)
    let mut max_len = 0i64;
    let mut lengths = Vec::with_capacity(sequences.len());

    for seq in &sequences {
        let ndim = seq.ndim()?;
        if ndim != 1 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("All sequences must be 1D, got {}D", ndim),
            ));
        }
        let seq_len = seq.shape_at(0)?;
        lengths.push(seq_len);
        max_len = max_len.max(seq_len);
    }

    // 2. Pad each sequence on GPU using MLX pad operation
    let padded_seqs: Vec<MxArray> = sequences
        .iter()
        .zip(lengths.iter())
        .map(|(seq, &len)| {
            let pad_amount = (max_len - len) as i32;
            if pad_amount > 0 {
                // Pad on the right: [before, after] for 1D array
                seq.pad(&[0, pad_amount], pad_value as f64)
            } else {
                Ok((*seq).clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;

    // 3. Stack all padded sequences on GPU
    let padded_refs: Vec<&MxArray> = padded_seqs.iter().collect();
    let padded = MxArray::stack(padded_refs, Some(0))?;

    // 4. Create masks on GPU using broadcasting
    let masks = create_padding_mask_gpu(&lengths, max_len)?;

    Ok(PaddedSequences::new(padded, masks))
}

/// Create padding masks entirely on GPU using broadcasting
///
/// Creates a mask where masks[i, j] = 1.0 if j < lengths[i] else 0.0
fn create_padding_mask_gpu(lengths: &[i64], max_len: i64) -> Result<MxArray> {
    let num_seqs = lengths.len();

    // Create indices [0, 1, 2, ..., max_len-1] on GPU
    let indices = MxArray::arange(0.0, max_len as f64, None, Some(DType::Float32))?;

    // Create lengths as column vector [num_seqs, 1] on GPU
    // This is a small transfer (just the lengths, e.g., 16 floats = 64 bytes)
    let lengths_f32: Vec<f32> = lengths.iter().map(|&l| l as f32).collect();
    let lengths_arr = MxArray::from_float32(&lengths_f32, &[num_seqs as i64, 1])?;

    // Broadcast comparison: indices < lengths gives mask
    // indices shape: [max_len] -> broadcast to [num_seqs, max_len]
    // lengths shape: [num_seqs, 1] -> broadcast to [num_seqs, max_len]
    let mask = indices.less(&lengths_arr)?;

    // Convert bool mask to float32 (1.0 / 0.0)
    mask.astype(DType::Float32)
}

/// Result from left-padding sequences
pub struct LeftPaddedSequences {
    padded: MxArray,
    left_padding: Vec<i32>,
}

impl std::fmt::Debug for LeftPaddedSequences {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeftPaddedSequences")
            .field("left_padding", &self.left_padding)
            .field("padded", &"<MxArray>")
            .finish()
    }
}

impl LeftPaddedSequences {
    pub(crate) fn new(padded: MxArray, left_padding: Vec<i32>) -> Self {
        Self {
            padded,
            left_padding,
        }
    }

    pub fn get_padded(&self) -> Result<MxArray> {
        Ok(self.padded.clone())
    }

    pub fn get_left_padding(&self) -> Vec<i32> {
        self.left_padding.clone()
    }
}

/// Pad variable-length sequences to uniform length with LEFT padding.
///
/// This is required by BatchKVCache which expects left-padded sequences.
/// For example, given prompts:
/// - [1, 3, 5]
/// - [7]
/// - [2, 6, 8, 9]
///
/// Returns:
/// - padded: [[0, 1, 3, 5], [0, 0, 0, 7], [2, 6, 8, 9]]
/// - left_padding: [1, 3, 0]
///
/// # Performance
///
/// This function performs all operations on GPU:
/// - Uses `shape_at()` for O(1) metadata access (no GPU sync)
/// - Uses `MxArray::pad()` for GPU-native padding
/// - Uses `MxArray::stack()` for GPU stacking
/// - No GPU→CPU data transfers occur
///
/// # Arguments
/// * `sequences` - Vector of 1D arrays with variable lengths
/// * `pad_value` - Value to use for padding (typically 0 for pad token)
///
/// # Returns
/// LeftPaddedSequences with `padded` (shape: [num_seqs, max_len]) and `left_padding` amounts
pub fn left_pad_sequences(sequences: Vec<&MxArray>, pad_value: i32) -> Result<LeftPaddedSequences> {
    if sequences.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "sequences cannot be empty".to_string(),
        ));
    }

    // 1. Find max length using shape_at (no GPU sync, just metadata)
    let mut max_len = 0i64;
    let mut lengths = Vec::with_capacity(sequences.len());

    for seq in &sequences {
        let ndim = seq.ndim()?;
        if ndim != 1 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("All sequences must be 1D, got {}D", ndim),
            ));
        }
        let seq_len = seq.shape_at(0)?;
        lengths.push(seq_len);
        max_len = max_len.max(seq_len);
    }

    // 2. Calculate left padding amounts
    let left_padding: Vec<i32> = lengths.iter().map(|&len| (max_len - len) as i32).collect();

    // 3. Pad each sequence on the LEFT using MLX pad operation
    let padded_seqs: Vec<MxArray> = sequences
        .iter()
        .zip(lengths.iter())
        .map(|(seq, &len)| {
            let pad_amount = (max_len - len) as i32;
            if pad_amount > 0 {
                // Pad on the LEFT: [before, after] for 1D array
                seq.pad(&[pad_amount, 0], pad_value as f64)
            } else {
                Ok((*seq).clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;

    // 4. Stack all padded sequences on GPU
    let padded_refs: Vec<&MxArray> = padded_seqs.iter().collect();
    let padded = MxArray::stack(padded_refs, Some(0))?;

    Ok(LeftPaddedSequences::new(padded, left_padding))
}

/// Pad variable-length float sequences to uniform length
///
/// Takes a list of 1D float sequences (e.g., log probabilities) and pads them to the maximum length.
///
/// # Performance
///
/// This function performs all operations on GPU:
/// - Uses `shape_at()` for O(1) metadata access (no GPU sync)
/// - Uses `MxArray::pad()` for GPU-native padding
/// - Uses `MxArray::stack()` for GPU stacking
/// - No GPU→CPU data transfers occur
///
/// # Arguments
/// * `sequences` - Vector of 1D float arrays with variable lengths
/// * `pad_value` - Value to use for padding (default: 0.0)
///
/// # Returns
/// Padded array with shape [num_seqs, max_len]
pub fn pad_float_sequences(sequences: Vec<&MxArray>, pad_value: f64) -> Result<MxArray> {
    if sequences.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "sequences cannot be empty".to_string(),
        ));
    }

    // 1. Find max length using shape_at (no GPU sync, just metadata)
    let mut max_len = 0i64;

    for seq in &sequences {
        let ndim = seq.ndim()?;
        if ndim != 1 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("All sequences must be 1D, got {}D", ndim),
            ));
        }
        let seq_len = seq.shape_at(0)?;
        max_len = max_len.max(seq_len);
    }

    // 2. Pad each sequence on GPU using MLX pad operation
    let padded_seqs: Vec<MxArray> = sequences
        .iter()
        .map(|seq| {
            let len = seq.shape_at(0)?;
            let pad_amount = (max_len - len) as i32;
            if pad_amount > 0 {
                // Pad on the right: [before, after] for 1D array
                seq.pad(&[0, pad_amount], pad_value)
            } else {
                Ok((*seq).clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;

    // 3. Stack all padded sequences on GPU
    let padded_refs: Vec<&MxArray> = padded_seqs.iter().collect();
    MxArray::stack(padded_refs, Some(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_left_pad_sequences_basic() {
        // Test: [1, 3, 5], [7], [2, 6, 8, 9] -> padded with left padding
        let seq1 = MxArray::from_int32(&[1, 3, 5], &[3]).unwrap();
        let seq2 = MxArray::from_int32(&[7], &[1]).unwrap();
        let seq3 = MxArray::from_int32(&[2, 6, 8, 9], &[4]).unwrap();

        let result = left_pad_sequences(vec![&seq1, &seq2, &seq3], 0).unwrap();

        // Check left_padding amounts
        let left_padding = result.get_left_padding();
        assert_eq!(left_padding, vec![1, 3, 0]);

        // Check padded array shape
        let padded = result.get_padded().unwrap();
        let shape = padded.shape().unwrap();
        assert_eq!(shape.to_vec(), vec![3, 4]);

        // Check padded values
        padded.eval();
        let values = padded.to_int32().unwrap();
        // Row 0: [0, 1, 3, 5] (1 padding on left)
        assert_eq!(values[0], 0);
        assert_eq!(values[1], 1);
        assert_eq!(values[2], 3);
        assert_eq!(values[3], 5);
        // Row 1: [0, 0, 0, 7] (3 padding on left)
        assert_eq!(values[4], 0);
        assert_eq!(values[5], 0);
        assert_eq!(values[6], 0);
        assert_eq!(values[7], 7);
        // Row 2: [2, 6, 8, 9] (0 padding)
        assert_eq!(values[8], 2);
        assert_eq!(values[9], 6);
        assert_eq!(values[10], 8);
        assert_eq!(values[11], 9);
    }

    #[test]
    fn test_left_pad_sequences_all_same_length() {
        // When all sequences have same length, no padding needed
        let seq1 = MxArray::from_int32(&[1, 2, 3], &[3]).unwrap();
        let seq2 = MxArray::from_int32(&[4, 5, 6], &[3]).unwrap();

        let result = left_pad_sequences(vec![&seq1, &seq2], 0).unwrap();

        let left_padding = result.get_left_padding();
        assert_eq!(left_padding, vec![0, 0]);

        let padded = result.get_padded().unwrap();
        let shape = padded.shape().unwrap();
        assert_eq!(shape.to_vec(), vec![2, 3]);
    }

    #[test]
    fn test_left_pad_sequences_single_sequence() {
        let seq = MxArray::from_int32(&[1, 2, 3, 4, 5], &[5]).unwrap();

        let result = left_pad_sequences(vec![&seq], 0).unwrap();

        let left_padding = result.get_left_padding();
        assert_eq!(left_padding, vec![0]);

        let padded = result.get_padded().unwrap();
        let shape = padded.shape().unwrap();
        assert_eq!(shape.to_vec(), vec![1, 5]);
    }

    #[test]
    fn test_left_pad_sequences_custom_pad_value() {
        let seq1 = MxArray::from_int32(&[1, 2], &[2]).unwrap();
        let seq2 = MxArray::from_int32(&[3], &[1]).unwrap();

        // Use -1 as pad value
        let result = left_pad_sequences(vec![&seq1, &seq2], -1).unwrap();

        let padded = result.get_padded().unwrap();
        padded.eval();
        let values = padded.to_int32().unwrap();

        // Row 0: [1, 2] (no padding)
        assert_eq!(values[0], 1);
        assert_eq!(values[1], 2);
        // Row 1: [-1, 3] (1 padding with -1)
        assert_eq!(values[2], -1);
        assert_eq!(values[3], 3);
    }

    #[test]
    fn test_left_pad_sequences_empty_error() {
        let result = left_pad_sequences(vec![], 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }
}
