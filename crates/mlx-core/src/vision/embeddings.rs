//! Vision Embeddings
//!
//! Internal implementation - not exposed to TypeScript.
//! Patch embedding and position embedding for vision transformers.

use crate::array::MxArray;
use crate::nn::Embedding;
use crate::vision::conv2d::Conv2d;
use crate::vision::interpolate::bilinear_interpolate;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// Patch Embedding (internal)
///
/// Converts image patches to embeddings using a convolution layer.
/// This is the first stage of vision transformers that extracts patch tokens.
pub struct PatchEmbedding {
    /// Convolution layer for patch extraction
    patch_conv: Arc<Conv2d>,
    /// Size of each patch
    patch_size: u32,
}

impl PatchEmbedding {
    /// Create a new Patch Embedding layer
    ///
    /// # Arguments
    /// * `patch_size` - Size of each image patch (typically 14 or 16)
    /// * `in_channels` - Number of input channels (3 for RGB)
    /// * `embed_dim` - Embedding dimension
    /// * `weight` - Convolution weights [embed_dim, patch_size, patch_size, in_channels]
    pub fn new(patch_size: u32, weight: &MxArray) -> Result<Self> {
        // Create conv layer with stride = kernel_size = patch_size
        let conv = Conv2d::new(
            weight,
            None,
            Some(vec![patch_size, patch_size]), // stride
            Some(vec![0, 0]),                   // padding
            None,                               // dilation
            None,                               // groups
        )?;

        Ok(Self {
            patch_conv: Arc::new(conv),
            patch_size,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `pixel_values` - Input tensor [batch, height, width, channels] (NHWC)
    ///
    /// # Returns
    /// * Patch embeddings [batch, num_patches, embed_dim]
    pub fn forward(&self, pixel_values: &MxArray) -> Result<MxArray> {
        // Apply patch convolution: [batch, h, w, c] -> [batch, h/p, w/p, embed_dim]
        let patches = self.patch_conv.forward(pixel_values)?;

        let shape = patches.shape()?;
        let batch = shape[0];
        let h = shape[1];
        let w = shape[2];
        let embed_dim = shape[3];

        // Flatten spatial dimensions: [batch, h*w, embed_dim]
        patches.reshape(&[batch, h * w, embed_dim])
    }

    /// Get the patch size
    pub fn patch_size(&self) -> u32 {
        self.patch_size
    }

    /// Get the convolution weight
    pub fn weight(&self) -> MxArray {
        self.patch_conv.weight()
    }
}

/// Vision Position Embedding (internal)
///
/// Learnable position embeddings for vision transformer patches.
/// Supports interpolation for different image sizes.
pub struct VisionPositionEmbedding {
    /// Embedding layer for positions
    embedding: Arc<Embedding>,
    /// Base grid size (sqrt of num_positions)
    base_size: u32,
}

impl VisionPositionEmbedding {
    /// Create a new Vision Position Embedding
    ///
    /// # Arguments
    /// * `num_positions` - Number of position embeddings (typically (image_size/patch_size)^2)
    /// * `embed_dim` - Embedding dimension
    /// * `weight` - Embedding weights [num_positions, embed_dim]
    pub fn new(num_positions: u32, _embed_dim: u32, weight: &MxArray) -> Result<Self> {
        let embedding = Embedding::from_weight(weight)?;
        let base_size = (num_positions as f32).sqrt() as u32;

        Ok(Self {
            embedding: Arc::new(embedding),
            base_size,
        })
    }

    /// Get position embeddings, interpolating if necessary
    ///
    /// # Arguments
    /// * `height` - Target grid height (in patches)
    /// * `width` - Target grid width (in patches)
    ///
    /// # Returns
    /// * Position embeddings [height * width, embed_dim]
    pub fn forward(&self, height: u32, width: u32) -> Result<MxArray> {
        // If target size matches base size, return directly
        if height == self.base_size && width == self.base_size {
            let num_pos = (self.base_size * self.base_size) as i64;
            let position_ids = MxArray::arange(0.0, num_pos as f64, Some(1.0), None)?
                .astype(crate::array::DType::Int32)?;
            return self.embedding.forward(&position_ids);
        }

        // Interpolate position embeddings for different sizes
        self.interpolate_pos_encoding(height, width)
    }

    /// Interpolate position embeddings to target size
    fn interpolate_pos_encoding(&self, height: u32, width: u32) -> Result<MxArray> {
        let num_positions = (self.base_size * self.base_size) as i64;
        let embed_dim = self.embedding.embedding_dim() as i64;

        // Get all position embeddings
        let position_ids = MxArray::arange(0.0, num_positions as f64, Some(1.0), None)?
            .astype(crate::array::DType::Int32)?;
        let pos_embed = self.embedding.forward(&position_ids)?;

        // Reshape to 2D grid: [sqrt_num_pos, sqrt_num_pos, embed_dim]
        let base = self.base_size as i64;
        let pos_embed_2d = pos_embed.reshape(&[base, base, embed_dim])?;

        // Interpolate to target size
        let interpolated = bilinear_interpolate(&pos_embed_2d, height as i64, width as i64)?;

        // Flatten back: [height * width, embed_dim]
        interpolated.reshape(&[(height * width) as i64, embed_dim])
    }

    /// Get the base size
    pub fn base_size(&self) -> u32 {
        self.base_size
    }

    /// Get the embedding weight
    pub fn weight(&self) -> MxArray {
        self.embedding.weight()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_embedding() {
        // Create a small patch embedding
        let patch_size = 4u32;
        let in_c = 3i64;
        let embed_dim = 16i64;

        // Weight: [embed_dim, patch_size, patch_size, in_c]
        let weight_data: Vec<f32> = (0..(embed_dim * patch_size as i64 * patch_size as i64 * in_c)
            as usize)
            .map(|i| (i as f32 - 50.0) / 100.0)
            .collect();
        let weight = MxArray::from_float32(
            &weight_data,
            &[embed_dim, patch_size as i64, patch_size as i64, in_c],
        )
        .unwrap();

        let patch_embed = PatchEmbedding::new(patch_size, &weight).unwrap();

        // Input: [1, 8, 8, 3] -> [1, 4, 16] (2x2 patches, 16 dim)
        let input_data: Vec<f32> = (0..(8 * 8 * 3) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let input = MxArray::from_float32(&input_data, &[1, 8, 8, 3]).unwrap();

        let output = patch_embed.forward(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();

        // 8/4 = 2 patches per dimension, 2*2 = 4 total patches
        assert_eq!(shape, vec![1, 4, 16]);
    }

    #[test]
    fn test_position_embedding_same_size() {
        let num_pos = 16u32; // 4x4 grid
        let embed_dim = 8u32;

        let weight_data: Vec<f32> = (0..(num_pos * embed_dim) as usize)
            .map(|i| i as f32 / 100.0)
            .collect();
        let weight =
            MxArray::from_float32(&weight_data, &[num_pos as i64, embed_dim as i64]).unwrap();

        let pos_embed = VisionPositionEmbedding::new(num_pos, embed_dim, &weight).unwrap();

        // Same size as base (4x4)
        let output = pos_embed.forward(4, 4).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();

        assert_eq!(shape, vec![16, 8]);
    }

    #[test]
    fn test_position_embedding_interpolation() {
        let num_pos = 16u32; // 4x4 grid
        let embed_dim = 8u32;

        let weight_data: Vec<f32> = (0..(num_pos * embed_dim) as usize)
            .map(|i| i as f32 / 100.0)
            .collect();
        let weight =
            MxArray::from_float32(&weight_data, &[num_pos as i64, embed_dim as i64]).unwrap();

        let pos_embed = VisionPositionEmbedding::new(num_pos, embed_dim, &weight).unwrap();

        // Different size (2x2)
        let output = pos_embed.forward(2, 2).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();

        assert_eq!(shape, vec![4, 8]);
    }
}
