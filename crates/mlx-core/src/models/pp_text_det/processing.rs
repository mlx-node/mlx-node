//! Text Detection Image Preprocessing
//!
//! Resize, normalize, and convert images for the text detection model.
//! Implements PaddleOCR's DetResizeForTest with configurable limit_type and limit_side_len.
//!
//! PaddleOCR uses OpenCV which loads images as BGR. Our image loading uses RGB.
//! We swap R and B channels before normalization to match PaddleOCR's convention.

use crate::array::MxArray;
use image::imageops::FilterType;

/// PaddleOCR uses cv2.INTER_LINEAR (bilinear). The closest equivalent
/// in the `image` crate is `FilterType::Triangle` (bilinear interpolation).
const RESIZE_FILTER: FilterType = FilterType::Triangle;
use napi::bindgen_prelude::*;

/// Limit type for resize behavior, matching PaddleOCR's DetResizeForTest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LimitType {
    /// Scale so that max(h, w) <= limit_side_len (don't upscale)
    Max,
    /// Scale so that min(h, w) >= limit_side_len (don't downscale beyond max constraint)
    Min,
}

/// Image processor for text detection.
pub struct TextDetImageProcessor {
    /// Side length limit (PaddleOCR inference CLI default: 960)
    pub limit_side_len: u32,
    /// Limit type: "max" or "min" (PaddleOCR inference CLI default: "max")
    pub limit_type: LimitType,
    /// Image mean for normalization [B, G, R] (BGR order, matching PaddleOCR)
    mean: [f32; 3],
    /// Image std for normalization [B, G, R] (BGR order, matching PaddleOCR)
    std: [f32; 3],
}

impl Default for TextDetImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextDetImageProcessor {
    /// Create a new processor with PaddleOCR inference CLI defaults.
    ///
    /// PaddleOCR's inference CLI (tools/infer/utility.py) uses:
    ///   det_limit_side_len=960, det_limit_type="max"
    /// which are the defaults most users expect.
    pub fn new() -> Self {
        Self {
            limit_side_len: 960,
            limit_type: LimitType::Max,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// Create a processor with custom limit_side_len and limit_type.
    pub fn with_config(limit_side_len: u32, limit_type: LimitType) -> Self {
        Self {
            limit_side_len,
            limit_type,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// Compute the resize ratio based on limit_type and limit_side_len.
    ///
    /// Matches PaddleOCR's DetResizeForTest.resize_image_type0():
    ///   - limit_type == "max": if max(h,w) > limit, ratio = limit / max(h,w), else 1.0
    ///   - limit_type == "min": if min(h,w) < limit, ratio = limit / min(h,w), else 1.0
    fn compute_ratio(&self, h: u32, w: u32) -> f64 {
        let limit = self.limit_side_len as f64;
        match self.limit_type {
            LimitType::Max => {
                if h.max(w) as f64 > limit {
                    if h > w {
                        limit / h as f64
                    } else {
                        limit / w as f64
                    }
                } else {
                    1.0
                }
            }
            LimitType::Min => {
                if (h.min(w) as f64) < limit {
                    if h < w {
                        limit / h as f64
                    } else {
                        limit / w as f64
                    }
                } else {
                    1.0
                }
            }
        }
    }

    /// Process encoded image bytes.
    ///
    /// Returns (pixel_values, original_h, original_w, resized_h, resized_w)
    /// pixel_values: [1, padded_h, padded_w, 3] normalized float32
    pub fn process(&self, data: &[u8]) -> Result<(MxArray, u32, u32, u32, u32)> {
        let img = image::load_from_memory(data).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to decode image: {e}"),
            )
        })?;

        let orig_w = img.width();
        let orig_h = img.height();

        // Compute resize ratio using PaddleOCR's DetResizeForTest logic
        let ratio = self.compute_ratio(orig_h, orig_w);
        let new_h = (orig_h as f64 * ratio) as u32;
        let new_w = (orig_w as f64 * ratio) as u32;

        // Round to nearest multiple of 32 (not ceil) -- matches PaddleOCR:
        //   resize_h = max(int(round(resize_h / 32) * 32), 32)
        let round_w = ((new_w as f64 / 32.0).round() as u32 * 32).max(32);
        let round_h = ((new_h as f64 / 32.0).round() as u32 * 32).max(32);

        // Resize directly to rounded dimensions (no padding)
        let resized = img.resize_exact(round_w, round_h, RESIZE_FILTER);
        let rgb_img = resized.to_rgb8();

        let h = round_h as usize;
        let w = round_w as usize;
        let mut pixel_data: Vec<f32> = vec![0.0; h * w * 3];

        // Channel mapping: RGB->BGR. PaddleOCR loads as BGR (cv2.imread) and
        // the mean/std values are in BGR order. We swap R and B before normalizing.
        let channel_map = [2usize, 1, 0]; // BGR ordering

        for y in 0..h {
            for x in 0..w {
                let pixel = rgb_img.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let src_c = channel_map[c];
                    let val = pixel[src_c] as f32 / 255.0;
                    let normalized = (val - self.mean[c]) / self.std[c];
                    pixel_data[(y * w + x) * 3 + c] = normalized;
                }
            }
        }

        let pixel_values = MxArray::from_float32(&pixel_data, &[1, h as i64, w as i64, 3])?;

        Ok((pixel_values, orig_h, orig_w, round_h, round_w))
    }

    /// Process raw image bytes (RGB, already decoded).
    ///
    /// # Arguments
    /// * `rgb_data` - Raw RGB pixel data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// Returns (pixel_values, resized_h, resized_w)
    pub fn process_crop(
        &self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(MxArray, u32, u32)> {
        if width == 0 || height == 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Image dimensions must be non-zero, got {}x{}",
                    width, height
                ),
            ));
        }

        // Compute resize ratio using PaddleOCR's DetResizeForTest logic
        let ratio = self.compute_ratio(height, width);
        let new_h = (height as f64 * ratio) as u32;
        let new_w = (width as f64 * ratio) as u32;

        // Round to nearest multiple of 32 (not ceil) -- matches PaddleOCR
        let round_w = ((new_w as f64 / 32.0).round() as u32 * 32).max(32);
        let round_h = ((new_h as f64 / 32.0).round() as u32 * 32).max(32);

        // Create image from raw data and resize directly to rounded dimensions
        let img = image::RgbImage::from_raw(width, height, rgb_data.to_vec())
            .ok_or_else(|| Error::new(Status::InvalidArg, "Invalid RGB data dimensions"))?;
        let resized = image::imageops::resize(&img, round_w, round_h, RESIZE_FILTER);

        let h = round_h as usize;
        let w = round_w as usize;
        let mut pixel_data: Vec<f32> = vec![0.0; h * w * 3];

        // Channel mapping: RGB->BGR (same as process)
        let channel_map = [2usize, 1, 0];

        for y in 0..h {
            for x in 0..w {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let src_c = channel_map[c];
                    let val = pixel[src_c] as f32 / 255.0;
                    let normalized = (val - self.mean[c]) / self.std[c];
                    pixel_data[(y * w + x) * 3 + c] = normalized;
                }
            }
        }

        let pixel_values = MxArray::from_float32(&pixel_data, &[1, h as i64, w as i64, 3])?;

        Ok((pixel_values, round_h, round_w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let proc = TextDetImageProcessor::new();
        assert_eq!(proc.limit_side_len, 960);
        assert_eq!(proc.limit_type, LimitType::Max);
    }

    #[test]
    fn test_processor_with_config() {
        let proc = TextDetImageProcessor::with_config(960, LimitType::Max);
        assert_eq!(proc.limit_side_len, 960);
        assert_eq!(proc.limit_type, LimitType::Max);
    }

    #[test]
    fn test_compute_ratio_max_type() {
        let proc = TextDetImageProcessor::with_config(960, LimitType::Max);
        // Image 1920x1080: max=1920 > 960, ratio = 960/1920 = 0.5
        let ratio = proc.compute_ratio(1080, 1920);
        assert!((ratio - 0.5).abs() < 1e-6);
        // Image 640x480: max=640 < 960, ratio = 1.0
        let ratio = proc.compute_ratio(480, 640);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_ratio_min_type() {
        let proc = TextDetImageProcessor::with_config(736, LimitType::Min);
        // Image 1920x1080: min=1080 > 736, ratio = 1.0
        let ratio = proc.compute_ratio(1080, 1920);
        assert!((ratio - 1.0).abs() < 1e-6);
        // Image 640x480: min=480 < 736, ratio = 736/480
        let ratio = proc.compute_ratio(480, 640);
        assert!((ratio - 736.0 / 480.0).abs() < 1e-6);
        // Image 320x240: min=240 < 736, ratio = 736/240
        let ratio = proc.compute_ratio(240, 320);
        assert!((ratio - 736.0 / 240.0).abs() < 1e-6);
    }
}
