//! Shared Image Processing for Vision-Language Models
//!
//! Handles smart resizing, normalization, patch extraction, and batch
//! aggregation for Qwen VLMs and PaddleOCR-VL.

use crate::array::MxArray;
use image::RgbImage;
use napi::bindgen_prelude::*;

/// Smart resize that maintains aspect ratio within pixel bounds (internal)
pub fn smart_resize(
    height: i32,
    width: i32,
    factor: i32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<(i32, i32)> {
    // Validate inputs to prevent division by zero
    if height <= 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("height must be positive, got {}", height),
        ));
    }
    if width <= 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("width must be positive, got {}", width),
        ));
    }
    if factor <= 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("factor must be positive, got {}", factor),
        ));
    }

    let mut h = height;
    let mut w = width;

    // Ensure minimum dimensions
    if h < factor {
        w = (w * factor) / h;
        h = factor;
    }
    if w < factor {
        h = (h * factor) / w;
        w = factor;
    }

    // Check aspect ratio
    let aspect = (h.max(w) as f64) / (h.min(w) as f64);
    if aspect > 200.0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Absolute aspect ratio must be smaller than 200, got {:.1}",
                aspect
            ),
        ));
    }

    // Round to factor
    let mut h_bar = ((h as f64 / factor as f64).round() * factor as f64) as i32;
    let mut w_bar = ((w as f64 / factor as f64).round() * factor as f64) as i32;

    // Adjust to fit within pixel bounds
    let total_pixels = h_bar * w_bar;

    if total_pixels > max_pixels {
        let beta = ((h as f64 * w as f64) / max_pixels as f64).sqrt();
        h_bar = ((h as f64 / beta / factor as f64).floor() * factor as f64) as i32;
        w_bar = ((w as f64 / beta / factor as f64).floor() * factor as f64) as i32;

        // Guard against zero-sized output when max_pixels is too small
        if h_bar <= 0 || w_bar <= 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Invalid resize dimensions [{}, {}]: maxPixels ({}) may be too small for factor ({}). Minimum maxPixels should be {}.",
                    h_bar,
                    w_bar,
                    max_pixels,
                    factor,
                    factor * factor
                ),
            ));
        }
    } else if total_pixels < min_pixels {
        let beta = (min_pixels as f64 / (h as f64 * w as f64)).sqrt();
        h_bar = ((h as f64 * beta / factor as f64).ceil() * factor as f64) as i32;
        w_bar = ((w as f64 * beta / factor as f64).ceil() * factor as f64) as i32;
    }

    Ok((h_bar, w_bar))
}

/// Image processing configuration (internal)
#[derive(Debug, Clone)]
pub struct ImageProcessorConfig {
    pub min_pixels: i32,
    pub max_pixels: i32,
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub merge_size: i32,
    pub image_mean: Vec<f64>,
    pub image_std: Vec<f64>,
    pub do_rescale: bool,
    pub do_normalize: bool,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            min_pixels: 147384,
            max_pixels: 2822400,
            patch_size: 14,
            temporal_patch_size: 1,
            merge_size: 2,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
            do_rescale: true,
            do_normalize: true,
        }
    }
}

/// Processed single image output (internal)
pub struct ProcessedImage {
    /// Pixel values as MxArray [num_patches, channels, patch_h, patch_w]
    pixel_values: MxArray,
    /// Grid dimensions [t, h, w]
    image_grid_thw: Vec<i32>,
}

impl ProcessedImage {
    /// Create a new ProcessedImage
    pub fn new(pixel_values: MxArray, image_grid_thw: Vec<i32>) -> Self {
        Self {
            pixel_values,
            image_grid_thw,
        }
    }

    /// Get pixel values [num_patches, channels, patch_h, patch_w]
    pub fn pixel_values(&self) -> MxArray {
        self.pixel_values.clone()
    }

    /// Get grid dimensions [t, h, w]
    pub fn image_grid_thw(&self) -> Vec<i32> {
        self.image_grid_thw.clone()
    }
}

/// Aggregate individually processed images into a batched result.
///
/// Concatenates pixel values along axis 0 and builds the grid_thw array.
/// Shared by both PaddleOCR-VL and Qwen VLMs.
pub fn aggregate_processed_images(images: Vec<ProcessedImage>) -> Result<ProcessedImages> {
    if images.is_empty() {
        return Err(Error::new(Status::InvalidArg, "images cannot be empty"));
    }

    let num_images = images.len() as i64;
    let mut all_pixel_values: Vec<MxArray> = Vec::with_capacity(images.len());
    let mut all_grid_thw: Vec<i32> = Vec::with_capacity(images.len() * 3);

    for processed in images {
        all_pixel_values.push(processed.pixel_values());
        all_grid_thw.extend_from_slice(&processed.image_grid_thw());
    }

    let pixel_values = if all_pixel_values.len() == 1 {
        all_pixel_values.remove(0)
    } else {
        let refs: Vec<&MxArray> = all_pixel_values.iter().collect();
        MxArray::concatenate_many(refs, Some(0))?
    };

    let grid_thw = MxArray::from_int32(&all_grid_thw, &[num_images, 3])?;
    Ok(ProcessedImages::new(pixel_values, grid_thw))
}

/// Processed multiple images output (internal)
///
/// Used internally by VLM models to pass batch-processed image data.
pub struct ProcessedImages {
    /// Pixel values as MxArray [total_patches, channels, patch_h, patch_w]
    pixel_values: MxArray,
    /// Grid dimensions [num_images, 3] with [t, h, w] per image
    grid_thw: MxArray,
}

impl ProcessedImages {
    /// Create a new ProcessedImages
    pub fn new(pixel_values: MxArray, grid_thw: MxArray) -> Self {
        Self {
            pixel_values,
            grid_thw,
        }
    }

    /// Get pixel values [total_patches, C, patch_h, patch_w]
    pub fn pixel_values(&self) -> MxArray {
        self.pixel_values.clone()
    }

    /// Get grid dimensions [num_images, 3]
    pub fn grid_thw(&self) -> MxArray {
        self.grid_thw.clone()
    }
}

/// Normalize a resized RGB image and pack pixels into `[num_patches, 3, patch_h, patch_w]`
/// with `grid_thw = [1, grid_h, grid_w]`.
///
/// Shared by both PaddleOCR-VL and Qwen VLM image processors.
pub fn extract_image_patches(
    rgb_img: &RgbImage,
    config: &ImageProcessorConfig,
) -> Result<ProcessedImage> {
    let patch_size = usize::try_from(config.patch_size).map_err(|_| {
        Error::new(
            Status::InvalidArg,
            format!("patch_size must be positive, got {}", config.patch_size),
        )
    })?;
    if patch_size == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "patch_size must be positive, got 0",
        ));
    }

    let (width_u32, height_u32) = rgb_img.dimensions();
    let (height, width) = (height_u32 as usize, width_u32 as usize);
    let channels = 3usize;

    let mean: [f32; 3] = if config.do_normalize {
        if config.image_mean.len() < channels {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "image_mean must have at least {channels} elements, got {}",
                    config.image_mean.len()
                ),
            ));
        }
        [
            config.image_mean[0] as f32,
            config.image_mean[1] as f32,
            config.image_mean[2] as f32,
        ]
    } else {
        [0.0; 3]
    };
    let std: [f32; 3] = if config.do_normalize {
        if config.image_std.len() < channels {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "image_std must have at least {channels} elements, got {}",
                    config.image_std.len()
                ),
            ));
        }
        [
            config.image_std[0] as f32,
            config.image_std[1] as f32,
            config.image_std[2] as f32,
        ]
    } else {
        [1.0; 3]
    };

    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let grid_t = 1usize;
    let num_patches = grid_t * grid_h * grid_w;

    // Extract normalized patches directly in [num_patches, C, patch_h, patch_w] order
    let mut patch_data: Vec<f32> =
        Vec::with_capacity(num_patches * channels * patch_size * patch_size);

    for ph in 0..grid_h {
        for pw in 0..grid_w {
            for c in 0..channels {
                for py in 0..patch_size {
                    let y = (ph * patch_size + py) as u32;
                    for px in 0..patch_size {
                        let x = (pw * patch_size + px) as u32;
                        let mut value = rgb_img.get_pixel(x, y)[c] as f32;
                        if config.do_rescale {
                            value /= 255.0;
                        }
                        if config.do_normalize {
                            value = (value - mean[c]) / std[c];
                        }
                        patch_data.push(value);
                    }
                }
            }
        }
    }

    let pixel_values = MxArray::from_float32(
        &patch_data,
        &[
            num_patches as i64,
            channels as i64,
            patch_size as i64,
            patch_size as i64,
        ],
    )?;

    Ok(ProcessedImage::new(
        pixel_values,
        vec![grid_t as i32, grid_h as i32, grid_w as i32],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_resize_normal() {
        // Normal case - within bounds
        let (h, w) = smart_resize(384, 384, 28, 147384, 2822400).unwrap();
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w >= 147384);
        assert!(h * w <= 2822400);
    }

    #[test]
    fn test_smart_resize_too_small() {
        // Image too small - should be scaled up
        let (h, w) = smart_resize(100, 100, 28, 147384, 2822400).unwrap();
        assert!(h * w >= 147384);
    }

    #[test]
    fn test_smart_resize_too_large() {
        // Image too large - should be scaled down
        let (h, w) = smart_resize(4000, 4000, 28, 147384, 2822400).unwrap();
        assert!(h * w <= 2822400);
    }

    #[test]
    fn test_smart_resize_aspect_ratio() {
        // Very wide image
        let result = smart_resize(100, 30000, 28, 147384, 2822400);
        // Should fail due to aspect ratio > 200
        assert!(result.is_err());
    }

    #[test]
    fn test_smart_resize_divisibility() {
        // Result should always be divisible by factor
        let (h, w) = smart_resize(500, 700, 28, 147384, 2822400).unwrap();
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_smart_resize_zero_height() {
        // Zero height should return error, not panic
        let result = smart_resize(0, 100, 28, 147384, 2822400);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("height must be positive"));
    }

    #[test]
    fn test_smart_resize_zero_width() {
        // Zero width should return error, not panic
        let result = smart_resize(100, 0, 28, 147384, 2822400);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("width must be positive"));
    }

    #[test]
    fn test_smart_resize_zero_factor() {
        // Zero factor should return error, not panic
        let result = smart_resize(100, 100, 0, 147384, 2822400);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("factor must be positive"));
    }

    #[test]
    fn test_smart_resize_negative_inputs() {
        // Negative values should also return errors
        assert!(smart_resize(-100, 100, 28, 147384, 2822400).is_err());
        assert!(smart_resize(100, -100, 28, 147384, 2822400).is_err());
        assert!(smart_resize(100, 100, -28, 147384, 2822400).is_err());
    }
}
