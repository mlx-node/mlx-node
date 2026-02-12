//! CTC Head
//!
//! Simple linear projection from hidden dimension to number of classes (characters + blank).
//! The blank token is typically at index 0 for CTC decoding.

use crate::array::MxArray;
use crate::nn::Linear;
use napi::bindgen_prelude::*;

/// CTC (Connectionist Temporal Classification) head.
///
/// Projects from the SVTR neck's hidden dimension to the character vocabulary.
/// Output logits are used for CTC decoding (argmax -> collapse -> remove blanks).
///
/// **Design note on softmax placement**: PaddleOCR's `CTCHead.forward()` applies
/// `F.softmax(predicts, axis=2)` when `not self.training` (inference mode). We
/// intentionally omit softmax here and instead apply it in `ctc_greedy_decode()`.
/// This produces identical results because:
/// - argmax(logits) == argmax(softmax(logits)) (softmax is monotonic)
/// - Probability scores are extracted from the softmax output in both cases
///
/// Reference: `ppocr/modeling/heads/rec_ctc_head.py`
pub struct CTCHead {
    /// Linear projection: hidden_dim -> num_classes
    fc: Linear,
}

impl CTCHead {
    pub fn new(fc: Linear) -> Self {
        Self { fc }
    }

    /// Forward pass — returns raw logits (softmax applied in decode step).
    ///
    /// Input: [B, seq_len, hidden_dim]
    /// Output: [B, seq_len, num_classes] (raw logits, not probabilities)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        self.fc.forward(input)
    }
}

/// CTC greedy decoder.
///
/// Decodes the CTC output logits into character indices by:
/// 1. Apply softmax to convert logits to probabilities
/// 2. Argmax at each timestep
/// 3. Collapse consecutive duplicate indices
/// 4. Remove blank token (index 0)
///
/// This is equivalent to PaddleOCR's approach where softmax is applied in the
/// CTC head's forward pass and `CTCLabelDecode` calls `preds.argmax(axis=2)`
/// and `preds.max(axis=2)` on the already-softmaxed output. Both produce
/// identical character indices and confidence scores.
///
/// # Arguments
/// * `logits` - [B, seq_len, num_classes] raw logits from CTCHead
///
/// # Returns
/// * Vec of (char_indices, scores) per batch element
pub fn ctc_greedy_decode(logits: &MxArray) -> Result<Vec<(Vec<usize>, Vec<f32>)>> {
    let shape = logits.shape()?;
    if shape.len() != 3 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("CTC logits must be 3D [B, T, C], got {}D", shape.len()),
        ));
    }
    let batch = shape[0] as usize;
    let seq_len = shape[1] as usize;
    let num_classes = shape[2] as usize;

    // Softmax to get probabilities
    let probs = crate::nn::activations::Activations::softmax(logits, Some(-1))?;

    // Argmax along last dimension
    let indices = logits.argmax(-1, Some(false))?;
    indices.eval();
    probs.eval();

    let indices_flat = indices.to_int32()?.to_vec();
    let probs_flat = probs.to_float32()?.to_vec();

    let mut results = Vec::with_capacity(batch);

    for b in 0..batch {
        let mut chars = Vec::new();
        let mut scores = Vec::new();
        let mut prev_idx: i32 = -1;

        for t in 0..seq_len {
            let flat_idx = b * seq_len + t;
            let idx = indices_flat[flat_idx];

            // Skip blank (index 0) and consecutive duplicates
            if idx != 0 && idx != prev_idx {
                chars.push(idx as usize);
                // Get the probability of this character
                let prob_offset = (b * seq_len + t) * num_classes + idx as usize;
                scores.push(probs_flat[prob_offset]);
            }
            prev_idx = idx;
        }

        results.push((chars, scores));
    }

    Ok(results)
}
