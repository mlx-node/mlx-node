/**
 * Image Processing for PaddleOCR-VL
 *
 * Handles image preprocessing including smart resizing,
 * normalization, and patch extraction.
 */
use crate::array::MxArray;
use image::ImageReader;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::io::Cursor;
use std::path::Path;

/// Smart resize that maintains aspect ratio within pixel bounds
///
/// # Arguments
/// * `height` - Original image height
/// * `width` - Original image width
/// * `factor` - Resize factor (patch_size * merge_size, e.g., 28)
/// * `min_pixels` - Minimum total pixels (default 147384)
/// * `max_pixels` - Maximum total pixels (default 2822400)
///
/// # Returns
/// * Tuple of (new_height, new_width) that satisfies constraints
#[napi]
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
    } else if total_pixels < min_pixels {
        let beta = (min_pixels as f64 / (h as f64 * w as f64)).sqrt();
        h_bar = ((h as f64 * beta / factor as f64).ceil() * factor as f64) as i32;
        w_bar = ((w as f64 * beta / factor as f64).ceil() * factor as f64) as i32;
    }

    Ok((h_bar, w_bar))
}

/// Image processing configuration
#[napi(object)]
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

/// Processed image output
#[napi(js_name = "ProcessedImage")]
pub struct ProcessedImage {
    /// Pixel values as MxArray [num_patches, channels, patch_h, patch_w]
    pixel_values: MxArray,
    /// Grid dimensions [t, h, w]
    image_grid_thw: Vec<i32>,
    /// Original image dimensions [height, width]
    original_size: Vec<i32>,
    /// Resized dimensions [height, width]
    resized_size: Vec<i32>,
}

#[napi]
impl ProcessedImage {
    /// Get pixel values [num_patches, channels, patch_h, patch_w]
    #[napi(getter)]
    pub fn pixel_values(&self) -> MxArray {
        self.pixel_values.clone()
    }

    /// Get grid dimensions [t, h, w]
    #[napi(getter)]
    pub fn image_grid_thw(&self) -> Vec<i32> {
        self.image_grid_thw.clone()
    }

    /// Get image_grid_thw as MxArray for model input
    #[napi]
    pub fn get_grid_thw_array(&self) -> Result<MxArray> {
        MxArray::from_int32(&self.image_grid_thw, &[1, 3])
    }

    /// Get original image dimensions [height, width]
    #[napi(getter)]
    pub fn original_size(&self) -> Vec<i32> {
        self.original_size.clone()
    }

    /// Get resized dimensions [height, width]
    #[napi(getter)]
    pub fn resized_size(&self) -> Vec<i32> {
        self.resized_size.clone()
    }

    /// Get number of vision tokens after spatial merge
    #[napi]
    pub fn num_vision_tokens(&self, merge_size: i32) -> Result<i32> {
        if merge_size <= 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("merge_size must be positive, got {}", merge_size),
            ));
        }
        let t = self.image_grid_thw[0];
        let h = self.image_grid_thw[1];
        let w = self.image_grid_thw[2];
        Ok(t * (h / merge_size) * (w / merge_size))
    }
}

/// Image Processor for PaddleOCR-VL
#[napi(js_name = "ImageProcessor")]
pub struct ImageProcessor {
    config: ImageProcessorConfig,
}

#[napi]
impl ImageProcessor {
    #[napi(constructor)]
    pub fn new(config: Option<ImageProcessorConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
        }
    }

    /// Get the resize factor (patch_size * merge_size)
    #[napi(getter)]
    pub fn resize_factor(&self) -> i32 {
        self.config.patch_size * self.config.merge_size
    }

    /// Compute target size for an image
    #[napi]
    pub fn get_target_size(&self, height: i32, width: i32) -> Result<(i32, i32)> {
        smart_resize(
            height,
            width,
            self.resize_factor(),
            self.config.min_pixels,
            self.config.max_pixels,
        )
    }

    /// Get configuration
    #[napi(getter)]
    pub fn config(&self) -> ImageProcessorConfig {
        self.config.clone()
    }

    /// Process an image from file path
    #[napi]
    pub fn process_file(&self, path: String) -> Result<ProcessedImage> {
        let img = load_image_from_path(&path)?;
        self.process_image(img)
    }

    /// Process an image from bytes (Buffer)
    #[napi]
    pub fn process_bytes(&self, data: &[u8]) -> Result<ProcessedImage> {
        let img = load_image_from_bytes(data)?;
        self.process_image(img)
    }

    /// Internal: Process a loaded image
    fn process_image(&self, img: DynamicImage) -> Result<ProcessedImage> {
        let (orig_width, orig_height) = img.dimensions();

        // Smart resize
        let (new_height, new_width) = smart_resize(
            orig_height as i32,
            orig_width as i32,
            self.resize_factor(),
            self.config.min_pixels,
            self.config.max_pixels,
        )?;

        // Resize image
        let resized = img.resize_exact(
            new_width as u32,
            new_height as u32,
            FilterType::CatmullRom, // Bicubic equivalent
        );

        // Convert to RGB
        let rgb_img: RgbImage = resized.to_rgb8();

        // Convert to float and normalize
        let (height, width) = (new_height as usize, new_width as usize);
        let channels = 3usize;
        let mut pixel_data: Vec<f32> = Vec::with_capacity(height * width * channels);

        let mean: Vec<f32> = self.config.image_mean.iter().map(|&x| x as f32).collect();
        let std: Vec<f32> = self.config.image_std.iter().map(|&x| x as f32).collect();

        for y in 0..height {
            for x in 0..width {
                let pixel = rgb_img.get_pixel(x as u32, y as u32);
                for c in 0..channels {
                    let mut value = pixel[c] as f32;

                    // Rescale to [0, 1]
                    if self.config.do_rescale {
                        value /= 255.0;
                    }

                    // Normalize
                    if self.config.do_normalize {
                        value = (value - mean[c]) / std[c];
                    }

                    pixel_data.push(value);
                }
            }
        }

        // Reshape to patches
        // Input: [H, W, C] stored as [H * W * C]
        // Output: [num_patches, C, patch_h, patch_w]
        let patch_size = self.config.patch_size as usize;
        let grid_h = height / patch_size;
        let grid_w = width / patch_size;
        let grid_t = 1; // temporal dimension
        let num_patches = grid_t * grid_h * grid_w;

        // Reorder data into patches
        let mut patch_data: Vec<f32> =
            Vec::with_capacity(num_patches * channels * patch_size * patch_size);

        for ph in 0..grid_h {
            for pw in 0..grid_w {
                // For each patch, extract [C, patch_h, patch_w]
                for c in 0..channels {
                    for py in 0..patch_size {
                        for px in 0..patch_size {
                            let y = ph * patch_size + py;
                            let x = pw * patch_size + px;
                            let idx = (y * width + x) * channels + c;
                            patch_data.push(pixel_data[idx]);
                        }
                    }
                }
            }
        }

        // Create MxArray [num_patches, C, patch_h, patch_w]
        let pixel_values = MxArray::from_float32(
            &patch_data,
            &[
                num_patches as i64,
                channels as i64,
                patch_size as i64,
                patch_size as i64,
            ],
        )?;

        Ok(ProcessedImage {
            pixel_values,
            image_grid_thw: vec![grid_t as i32, grid_h as i32, grid_w as i32],
            original_size: vec![orig_height as i32, orig_width as i32],
            resized_size: vec![new_height, new_width],
        })
    }
}

/// Load image from file path
fn load_image_from_path(path: &str) -> Result<DynamicImage> {
    let path = Path::new(path);
    if !path.exists() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("Image file not found: {}", path.display()),
        ));
    }

    ImageReader::open(path)
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to open image: {}", e),
            )
        })?
        .decode()
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to decode image: {}", e),
            )
        })
}

/// Load image from bytes
fn load_image_from_bytes(data: &[u8]) -> Result<DynamicImage> {
    ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to guess image format: {}", e),
            )
        })?
        .decode()
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to decode image: {}", e),
            )
        })
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
    fn test_image_processor() {
        let processor = ImageProcessor::new(None);
        assert_eq!(processor.resize_factor(), 28); // 14 * 2
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
