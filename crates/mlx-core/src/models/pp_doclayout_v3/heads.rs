//! PP-DocLayoutV3 Prediction Heads
//!
//! Detection, classification, mask, and reading order heads for the PP-DocLayoutV3 model.
//! Ported from the HuggingFace Transformers implementation.
//!
//! Components:
//! - **MLPPredictionHead**: Stack of Linear + ReLU layers (used for bbox_embed and mask_query_head)
//! - **GlobalPointer**: Reading order prediction via bilinear attention between queries
//!
//! Reference: modeling_pp_doclayout_v3.py (PPDocLayoutV3MLPPredictionHead, PPDocLayoutV3GlobalPointer)

use crate::array::MxArray;
use crate::nn::activations::Activations;
use crate::nn::linear::Linear;
use napi::Either;
use napi::bindgen_prelude::*;

// ============================================================================
// MLP Prediction Head
// ============================================================================

/// Multi-layer perceptron prediction head.
///
/// Used for:
/// - `bbox_embed`: Linear(256→256→4), 3 layers with ReLU between
/// - `mask_query_head`: Linear(256→256→32), 3 layers with ReLU between
///
/// Architecture: Linear → ReLU → Linear → ReLU → ... → Linear (no activation on last)
///
/// Corresponds to PPDocLayoutV3MLPPredictionHead in the reference implementation.
pub struct MLPPredictionHead {
    /// Stack of linear layers
    layers: Vec<Linear>,
    /// Total number of layers (ReLU applied to all but last)
    num_layers: usize,
}

impl MLPPredictionHead {
    /// Create a new MLPPredictionHead.
    ///
    /// # Arguments
    /// * `input_dim` - Input dimension
    /// * `d_model` - Hidden dimension (used for intermediate layers)
    /// * `output_dim` - Output dimension
    /// * `num_layers` - Total number of linear layers
    pub fn new(input_dim: u32, d_model: u32, output_dim: u32, num_layers: usize) -> Result<Self> {
        // Build layer dimensions: [input_dim] + [d_model] * (num_layers - 1) → [d_model] * (num_layers - 1) + [output_dim]
        // E.g. for num_layers=3, input=256, hidden=256, output=4:
        //   layers = [Linear(256,256), Linear(256,256), Linear(256,4)]
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let in_dim = if i == 0 { input_dim } else { d_model };
            let out_dim = if i == num_layers - 1 {
                output_dim
            } else {
                d_model
            };
            layers.push(Linear::new(in_dim, out_dim, Some(true))?);
        }

        Ok(Self { layers, num_layers })
    }

    /// Create from pre-loaded weights.
    ///
    /// # Arguments
    /// * `layers` - Pre-constructed Linear layers
    pub fn from_layers(layers: Vec<Linear>) -> Self {
        let num_layers = layers.len();
        Self { layers, num_layers }
    }

    /// Forward pass: Linear → ReLU → Linear → ReLU → ... → Linear
    ///
    /// Input: [batch, num_queries, input_dim]
    /// Output: [batch, num_queries, output_dim]
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let mut hidden = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden)?;
            // Apply ReLU to all but the last layer
            if i < self.num_layers - 1 {
                hidden = Activations::relu(&hidden)?;
            }
        }
        Ok(hidden)
    }
}

// ============================================================================
// Global Pointer (Reading Order Prediction)
// ============================================================================

/// Reading order prediction head using bilinear attention.
///
/// Projects input to query/key pairs, computes scaled dot-product attention
/// between queries and keys, then masks the lower triangle (self-connections
/// and backward connections) with -inf.
///
/// Architecture:
/// 1. Linear(d_model, head_size * 2) → split into queries and keys
/// 2. queries @ keys^T / sqrt(head_size)
/// 3. Mask lower triangle with -1e4
///
/// Output: order logits [batch, num_queries, num_queries]
/// Upper triangle values represent reading order relationships.
///
/// Corresponds to PPDocLayoutV3GlobalPointer in the reference implementation.
pub struct GlobalPointer {
    /// Linear projection to query + key space
    dense: Linear,
    /// Head size for scaling (default: 64)
    head_size: i32,
}

impl GlobalPointer {
    /// Create a new GlobalPointer.
    ///
    /// # Arguments
    /// * `d_model` - Input dimension (default: 256)
    /// * `head_size` - Size of each query/key head (default: 64)
    pub fn new(d_model: u32, head_size: i32) -> Result<Self> {
        if head_size <= 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("head_size must be positive, got {head_size}"),
            ));
        }
        let dense = Linear::new(d_model, (head_size * 2) as u32, Some(true))?;
        Ok(Self { dense, head_size })
    }

    /// Create from pre-loaded weights.
    pub fn from_weights(dense: Linear, head_size: i32) -> Self {
        Self { dense, head_size }
    }

    /// Forward pass: compute reading order logits.
    ///
    /// Input: [batch, num_queries, d_model]
    /// Output: [batch, num_queries, num_queries]
    ///
    /// The output represents reading order relationships:
    /// - logits[b, i, j] > 0 means query i comes before query j
    /// - Lower triangle is masked with -1e4 (only upper triangle is meaningful)
    pub fn forward(&self, inputs: &MxArray) -> Result<MxArray> {
        let shape = inputs.shape()?;
        let shape: Vec<i64> = shape.as_ref().to_vec();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Project to query + key space: [B, S, 2 * head_size]
        let qk_proj = self.dense.forward(inputs)?;

        // Reshape to [B, S, 2, head_size]
        let qk_proj = qk_proj.reshape(&[batch_size, seq_len, 2, self.head_size as i64])?;

        // Split into queries [B, S, head_size] and keys [B, S, head_size]
        // queries = qk_proj[:, :, 0, :]
        let queries = qk_proj.slice(
            &[0, 0, 0, 0],
            &[batch_size, seq_len, 1, self.head_size as i64],
        )?;
        let queries = queries.squeeze(Some(&[2]))?;

        // keys = qk_proj[:, :, 1, :]
        let keys = qk_proj.slice(
            &[0, 0, 1, 0],
            &[batch_size, seq_len, 2, self.head_size as i64],
        )?;
        let keys = keys.squeeze(Some(&[2]))?;

        // Compute attention: queries @ keys^T / sqrt(head_size)
        // queries: [B, S, head_size], keys: [B, S, head_size]
        // keys^T: [B, head_size, S]
        let keys_t = keys.transpose(Some(&[0, 2, 1]))?;
        let scale = (self.head_size as f64).sqrt();
        let logits = queries.matmul(&keys_t)?;
        let logits = logits.div_scalar(scale)?;

        // Create lower-triangular mask and fill with -1e4
        // tril mask: True where row >= col (lower triangle + diagonal)
        // In PyTorch: mask = torch.tril(torch.ones(S, S)).bool()
        //             logits = logits.masked_fill(mask.unsqueeze(0), -1e4)
        //
        // We create this using arange comparisons:
        // row_idx >= col_idx → lower triangle (True)
        let row_idx = MxArray::arange(0.0, seq_len as f64, Some(1.0), None)?;
        let row_idx = row_idx.reshape(&[seq_len, 1])?;
        let col_idx = MxArray::arange(0.0, seq_len as f64, Some(1.0), None)?;
        let col_idx = col_idx.reshape(&[1, seq_len])?;

        // mask[i,j] = True where i >= j (lower triangle including diagonal)
        let mask = row_idx.greater_equal(&col_idx)?;

        // Expand mask to [1, S, S] for broadcasting with [B, S, S]
        let mask = mask.expand_dims(0)?;

        // Fill masked positions with -1e4
        let neg_inf = MxArray::full(&[1, seq_len, seq_len], Either::A(-1e4), None)?;
        let logits = mask.where_(&neg_inf, &logits)?;

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_prediction_head_shapes() {
        // bbox_embed: 256 → 256 → 4 (3 layers)
        let head = MLPPredictionHead::new(256, 256, 4, 3).unwrap();
        let input = MxArray::random_uniform(&[1, 300, 256], -1.0, 1.0, None).unwrap();
        let output = head.forward(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 300, 4]);
    }

    #[test]
    fn test_mlp_prediction_head_mask() {
        // mask_query_head: 256 → 256 → 32 (3 layers)
        let head = MLPPredictionHead::new(256, 256, 32, 3).unwrap();
        let input = MxArray::random_uniform(&[1, 300, 256], -1.0, 1.0, None).unwrap();
        let output = head.forward(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 300, 32]);
    }

    #[test]
    fn test_mlp_prediction_head_2_layers() {
        // query_pos_head: 4 → 512 → 256 (2 layers)
        let head = MLPPredictionHead::new(4, 512, 256, 2).unwrap();
        let input = MxArray::random_uniform(&[1, 300, 4], -1.0, 1.0, None).unwrap();
        let output = head.forward(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 300, 256]);
    }

    #[test]
    fn test_global_pointer_shapes() {
        let gp = GlobalPointer::new(256, 64).unwrap();
        let input = MxArray::random_uniform(&[1, 300, 256], -1.0, 1.0, None).unwrap();
        let output = gp.forward(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 300, 300]);
    }

    #[test]
    fn test_global_pointer_masking() {
        let gp = GlobalPointer::new(8, 4).unwrap();
        // Small example: batch=1, 4 queries, d_model=8
        let input = MxArray::random_uniform(&[1, 4, 8], -1.0, 1.0, None).unwrap();
        let output = gp.forward(&input).unwrap();
        output.eval();

        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 4, 4]);

        // Check that lower triangle is masked to -1e4
        let data = output.to_float32().unwrap();
        let data: Vec<f32> = data.to_vec();

        // Lower triangle + diagonal should be -1e4
        // Position (0,0) = -1e4 (diagonal)
        assert!((data[0] - (-1e4)).abs() < 1.0);
        // Position (1,0) = -1e4 (below diagonal)
        assert!((data[4] - (-1e4)).abs() < 1.0);
        // Position (1,1) = -1e4 (diagonal)
        assert!((data[5] - (-1e4)).abs() < 1.0);

        // Position (0,1) should NOT be -1e4 (upper triangle)
        assert!((data[1] - (-1e4)).abs() > 1.0);
    }
}
