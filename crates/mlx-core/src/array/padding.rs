//! Sequence padding utilities
//!
//! This module provides functions for padding variable-length sequences to uniform length,
//! which is essential for batched processing in transformers.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::MxArray;

/// Result from padding sequences with masks
#[napi]
pub struct PaddedSequences {
    padded: MxArray,
    masks: MxArray,
}

#[napi]
impl PaddedSequences {
    #[napi(getter)]
    pub fn get_padded(&self) -> Result<MxArray> {
        Ok(self.padded.clone())
    }

    #[napi(getter)]
    pub fn get_masks(&self) -> Result<MxArray> {
        Ok(self.masks.clone())
    }
}

// Constructor for internal use
impl PaddedSequences {
    pub(crate) fn new(padded: MxArray, masks: MxArray) -> Self {
        Self { padded, masks }
    }
}

/// Pad variable-length sequences to uniform length (for integers/tokens)
///
/// Takes a list of 1D sequences and pads them to the maximum length.
/// Returns both the padded sequences and binary masks indicating real vs padded positions.
///
/// # Arguments
/// * `sequences` - Vector of 1D arrays with variable lengths
/// * `pad_value` - Value to use for padding (default: 0)
///
/// # Returns
/// Object with `padded` (shape: [num_seqs, max_len]) and `masks` (same shape, 1.0 for real tokens, 0.0 for padding)
#[napi(js_name = "padSequences")]
pub fn pad_sequences(sequences: Vec<&MxArray>, pad_value: i32) -> Result<PaddedSequences> {
    if sequences.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "sequences cannot be empty".to_string(),
        ));
    }

    // Find max length
    let mut max_len = 0i64;
    let mut lengths = Vec::new();

    for seq in &sequences {
        let shape = seq.shape()?;
        if shape.len() != 1 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("All sequences must be 1D, got {:?}D", shape.len()),
            ));
        }
        let seq_len = shape[0];
        lengths.push(seq_len);
        max_len = max_len.max(seq_len);
    }

    let num_seqs = sequences.len();

    // Create padded data and masks
    let mut padded_data = Vec::with_capacity(num_seqs * max_len as usize);
    let mut mask_data = Vec::with_capacity(num_seqs * max_len as usize);

    for (i, seq) in sequences.iter().enumerate() {
        let tokens = seq.to_int32()?;
        let seq_len = lengths[i] as usize;

        // Add actual tokens
        for j in 0..seq_len {
            padded_data.push(tokens[j]);
            mask_data.push(1.0);
        }

        // Add padding
        for _ in seq_len..(max_len as usize) {
            padded_data.push(pad_value);
            mask_data.push(0.0);
        }
    }

    let padded = MxArray::from_int32(&padded_data, &[num_seqs as i64, max_len])?;
    let masks = MxArray::from_float32(&mask_data, &[num_seqs as i64, max_len])?;

    Ok(PaddedSequences::new(padded, masks))
}

/// Pad variable-length float sequences to uniform length
///
/// Takes a list of 1D float sequences (e.g., log probabilities) and pads them to the maximum length.
///
/// # Arguments
/// * `sequences` - Vector of 1D float arrays with variable lengths
/// * `pad_value` - Value to use for padding (default: 0.0)
///
/// # Returns
/// Padded array with shape [num_seqs, max_len]
#[napi(js_name = "padFloatSequences")]
pub fn pad_float_sequences(sequences: Vec<&MxArray>, pad_value: f64) -> Result<MxArray> {
    if sequences.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "sequences cannot be empty".to_string(),
        ));
    }

    // Find max length
    let mut max_len = 0i64;
    let mut lengths = Vec::new();

    for seq in &sequences {
        let shape = seq.shape()?;
        if shape.len() != 1 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("All sequences must be 1D, got {:?}D", shape.len()),
            ));
        }
        let seq_len = shape[0];
        lengths.push(seq_len);
        max_len = max_len.max(seq_len);
    }

    let num_seqs = sequences.len();

    // Create padded data
    let mut padded_data = Vec::with_capacity(num_seqs * max_len as usize);

    for (i, seq) in sequences.iter().enumerate() {
        let values = seq.to_float32()?;
        let seq_len = lengths[i] as usize;

        // Add actual values
        for j in 0..seq_len {
            padded_data.push(values[j]);
        }

        // Add padding
        for _ in seq_len..(max_len as usize) {
            padded_data.push(pad_value as f32);
        }
    }

    MxArray::from_float32(&padded_data, &[num_seqs as i64, max_len])
}
