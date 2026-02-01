//! Spatial Projector
//!
//! Internal implementation - not exposed to TypeScript.
//! Projects vision features to the language model dimension,
//! reducing token count by merging spatial patches.

use crate::array::MxArray;
use crate::nn::activations::Activations;
use crate::nn::{LayerNorm, Linear};
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// Spatial Merge Projector (internal)
///
/// Reduces the number of vision tokens by merging spatial patches,
/// then projects to the language model dimension.
pub struct SpatialProjector {
    /// Pre-projection LayerNorm
    pre_norm: Arc<LayerNorm>,
    /// First linear layer (after spatial merge)
    linear_1: Arc<Linear>,
    /// Second linear layer (output projection)
    linear_2: Arc<Linear>,
    /// Spatial merge size (e.g., 2 merges 2x2 patches into 1)
    spatial_merge_size: u32,
}

impl SpatialProjector {
    /// Create a new Spatial Projector
    ///
    /// # Arguments
    /// * `dim` - Vision model dimension
    /// * `context_dim` - Language model dimension (output dimension)
    /// * `spatial_merge_size` - How many patches to merge (e.g., 2 for 2x2)
    /// * `pre_norm_weight` - LayerNorm weight [dim]
    /// * `pre_norm_bias` - LayerNorm bias [dim]
    /// * `linear_1_weight` - First linear weight [hidden, dim * merge^2]
    /// * `linear_1_bias` - First linear bias [hidden]
    /// * `linear_2_weight` - Second linear weight [context_dim, hidden]
    /// * `linear_2_bias` - Second linear bias [context_dim]
    pub fn new(
        spatial_merge_size: u32,
        pre_norm_weight: &MxArray,
        pre_norm_bias: &MxArray,
        linear_1_weight: &MxArray,
        linear_1_bias: &MxArray,
        linear_2_weight: &MxArray,
        linear_2_bias: &MxArray,
    ) -> Result<Self> {
        let pre_norm = LayerNorm::from_weights(pre_norm_weight, Some(pre_norm_bias), Some(1e-6))?;
        let linear_1 = Linear::from_weights(linear_1_weight, Some(linear_1_bias))?;
        let linear_2 = Linear::from_weights(linear_2_weight, Some(linear_2_bias))?;

        Ok(Self {
            pre_norm: Arc::new(pre_norm),
            linear_1: Arc::new(linear_1),
            linear_2: Arc::new(linear_2),
            spatial_merge_size,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Vision features [total_patches, dim]
    /// * `grid_thw` - Grid dimensions [num_images, 3] where each row is [t, h, w]
    ///
    /// # Returns
    /// * Projected features [total_merged_patches, context_dim]
    pub fn forward(&self, x: &MxArray, grid_thw: &MxArray) -> Result<MxArray> {
        // Validate grid_thw shape
        let grid_shape = grid_thw.shape()?;
        if grid_shape.len() != 2 || grid_shape[1] != 3 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "grid_thw must have shape [num_images, 3], got {:?}",
                    grid_shape.as_ref()
                ),
            ));
        }
        let num_images = grid_shape[0];

        // Get grid values
        let grid_data = grid_thw.to_int32()?;

        // Process each image and collect results
        let mut processed_features: Vec<MxArray> = Vec::new();
        let mut start_idx = 0i64;

        for img_idx in 0..num_images as usize {
            let t = grid_data[img_idx * 3] as i64;
            let h = grid_data[img_idx * 3 + 1] as i64;
            let w = grid_data[img_idx * 3 + 2] as i64;

            // Validate h and w are divisible by spatial_merge_size
            let merge = self.spatial_merge_size as i64;
            if h % merge != 0 {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "Image {} height ({}) must be divisible by spatial_merge_size ({})",
                        img_idx, h, merge
                    ),
                ));
            }
            if w % merge != 0 {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "Image {} width ({}) must be divisible by spatial_merge_size ({})",
                        img_idx, w, merge
                    ),
                ));
            }

            let num_patches = t * h * w;
            let end_idx = start_idx + num_patches;

            // Extract features for this image: [num_patches, dim]
            let x_img = x.slice_axis(0, start_idx, end_idx)?;

            // Apply pre-norm
            let x_normed = self.pre_norm.forward(&x_img)?;

            // Get feature dimension
            let dim = x_normed.shape()?[1];

            // Compute merged grid size
            let merge = self.spatial_merge_size as i64;
            let h_block = h / merge;
            let w_block = w / merge;

            // Reshape for spatial merge:
            // [t*h*w, d] -> [t, h_block, merge, w_block, merge, d]
            let x_reshaped = x_normed.reshape(&[t, h_block, merge, w_block, merge, dim])?;

            // Transpose to group merge dimensions together:
            // [t, h_block, w_block, merge, merge, d]
            let x_transposed = x_reshaped.transpose(Some(&[0, 1, 3, 2, 4, 5]))?;

            // Flatten merged patches:
            // [t * h_block * w_block, merge * merge * d]
            let merged_patches = t * h_block * w_block;
            let merged_dim = merge * merge * dim;
            let x_flat = x_transposed.reshape(&[merged_patches, merged_dim])?;

            // Apply MLP: linear_1 -> GELU -> linear_2
            let hidden = self.linear_1.forward(&x_flat)?;
            let activated = Activations::gelu(&hidden)?;
            let output = self.linear_2.forward(&activated)?;

            processed_features.push(output);
            start_idx = end_idx;
        }

        // Concatenate all processed features
        if processed_features.len() == 1 {
            Ok(processed_features.remove(0))
        } else {
            let refs: Vec<&MxArray> = processed_features.iter().collect();
            MxArray::concatenate_many(refs, Some(0))
        }
    }

    /// Get the spatial merge size
    pub fn spatial_merge_size(&self) -> u32 {
        self.spatial_merge_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_array(shape: &[i64]) -> MxArray {
        let size: usize = shape.iter().map(|&s| s as usize).product();
        let data: Vec<f32> = (0..size)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        MxArray::from_float32(&data, shape).unwrap()
    }

    #[test]
    fn test_spatial_projector() {
        let dim = 16i64;
        let context_dim = 32i64;
        let merge = 2u32;
        let hidden_size = dim * (merge as i64).pow(2); // 16 * 4 = 64

        // Create weights
        let pre_norm_weight_data = vec![1.0f32; dim as usize];
        let pre_norm_weight = MxArray::from_float32(&pre_norm_weight_data, &[dim]).unwrap();
        let pre_norm_bias_data = vec![0.0f32; dim as usize];
        let pre_norm_bias = MxArray::from_float32(&pre_norm_bias_data, &[dim]).unwrap();

        let linear_1_weight = random_array(&[hidden_size, hidden_size]);
        let linear_1_bias_data = vec![0.0f32; hidden_size as usize];
        let linear_1_bias = MxArray::from_float32(&linear_1_bias_data, &[hidden_size]).unwrap();

        let linear_2_weight = random_array(&[context_dim, hidden_size]);
        let linear_2_bias_data = vec![0.0f32; context_dim as usize];
        let linear_2_bias = MxArray::from_float32(&linear_2_bias_data, &[context_dim]).unwrap();

        let projector = SpatialProjector::new(
            merge,
            &pre_norm_weight,
            &pre_norm_bias,
            &linear_1_weight,
            &linear_1_bias,
            &linear_2_weight,
            &linear_2_bias,
        )
        .unwrap();

        // Input: 1 image with grid [1, 4, 4] = 16 patches
        let x = random_array(&[16, dim]);
        let grid_thw_data = vec![1i32, 4, 4];
        let grid_thw = MxArray::from_int32(&grid_thw_data, &[1, 3]).unwrap();

        let output = projector.forward(&x, &grid_thw).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();

        // After 2x2 merge: 16 patches -> 4 patches
        // Output: [4, context_dim=32]
        assert_eq!(shape, vec![4, context_dim]);
    }

    #[test]
    fn test_spatial_projector_multiple_images() {
        let dim = 8i64;
        let context_dim = 16i64;
        let merge = 2u32;
        let hidden_size = dim * (merge as i64).pow(2); // 8 * 4 = 32

        // Create weights (simplified for test)
        let pre_norm_weight_data = vec![1.0f32; dim as usize];
        let pre_norm_weight = MxArray::from_float32(&pre_norm_weight_data, &[dim]).unwrap();
        let pre_norm_bias_data = vec![0.0f32; dim as usize];
        let pre_norm_bias = MxArray::from_float32(&pre_norm_bias_data, &[dim]).unwrap();

        let linear_1_weight = random_array(&[hidden_size, hidden_size]);
        let linear_1_bias_data = vec![0.0f32; hidden_size as usize];
        let linear_1_bias = MxArray::from_float32(&linear_1_bias_data, &[hidden_size]).unwrap();

        let linear_2_weight = random_array(&[context_dim, hidden_size]);
        let linear_2_bias_data = vec![0.0f32; context_dim as usize];
        let linear_2_bias = MxArray::from_float32(&linear_2_bias_data, &[context_dim]).unwrap();

        let projector = SpatialProjector::new(
            merge,
            &pre_norm_weight,
            &pre_norm_bias,
            &linear_1_weight,
            &linear_1_bias,
            &linear_2_weight,
            &linear_2_bias,
        )
        .unwrap();

        // Input: 2 images
        // Image 1: [1, 4, 4] = 16 patches
        // Image 2: [1, 2, 2] = 4 patches
        // Total: 20 patches
        let x = random_array(&[20, dim]);
        let grid_thw_data = vec![1i32, 4, 4, 1, 2, 2];
        let grid_thw = MxArray::from_int32(&grid_thw_data, &[2, 3]).unwrap();

        let output = projector.forward(&x, &grid_thw).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();

        // After 2x2 merge:
        // Image 1: 16 -> 4 patches
        // Image 2: 4 -> 1 patch
        // Total: 5 merged patches
        assert_eq!(shape, vec![5, context_dim]);
    }
}
