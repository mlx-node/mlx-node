//! PP-DocLayoutV3 Image Processing
//!
//! Image preprocessing pipeline: load → resize to 800x800 → rescale to [0,1] → NHWC tensor.
//! PP-DocLayoutV3 uses image_mean=[0,0,0] and image_std=[1,1,1], so no normalization
//! beyond rescaling is needed.

use crate::array::MxArray;
use image::imageops::FilterType;
use napi::bindgen_prelude::*;

/// Image processor for PP-DocLayoutV3 document layout analysis.
///
/// Resizes images to 800x800 and rescales pixel values to [0, 1].
/// Output is in NHWC format [1, 800, 800, 3] as required by the MLX backbone.
pub struct PPDocLayoutV3ImageProcessor {
    target_height: u32,
    target_width: u32,
}

impl Default for PPDocLayoutV3ImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl PPDocLayoutV3ImageProcessor {
    /// Create a new image processor with default 800x800 target size.
    pub fn new() -> Self {
        Self {
            target_height: 800,
            target_width: 800,
        }
    }

    /// Process encoded image bytes for model input.
    ///
    /// # Arguments
    /// * `data` - Encoded image bytes (PNG/JPEG)
    ///
    /// # Returns
    /// * `(pixel_values, original_height, original_width)` where:
    ///   - pixel_values: MxArray [1, 800, 800, 3] in NHWC format, float32, rescaled to [0,1]
    ///   - original_height: Original image height before resize
    ///   - original_width: Original image width before resize
    pub fn process(&self, data: &[u8]) -> Result<(MxArray, u32, u32)> {
        let img = image::load_from_memory(data).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to decode image: {}", e),
            )
        })?;

        // Get original dimensions
        let original_width = img.width();
        let original_height = img.height();

        // Resize to target dimensions using bicubic (CatmullRom) interpolation
        let resized = img.resize_exact(
            self.target_width,
            self.target_height,
            FilterType::CatmullRom,
        );

        // Convert to RGB8
        let rgb_img = resized.to_rgb8();

        // Convert pixels to float32 and rescale to [0, 1]
        let h = self.target_height as usize;
        let w = self.target_width as usize;
        let channels = 3usize;
        let pixel_data: Vec<f32> = rgb_img.as_raw().iter().map(|&b| b as f32 / 255.0).collect();

        // Create MxArray [1, H, W, 3] in NHWC format
        let pixel_values =
            MxArray::from_float32(&pixel_data, &[1, h as i64, w as i64, channels as i64])?;

        Ok((pixel_values, original_height, original_width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_processor_creation() {
        let processor = PPDocLayoutV3ImageProcessor::new();
        assert_eq!(processor.target_height, 800);
        assert_eq!(processor.target_width, 800);
    }
}
