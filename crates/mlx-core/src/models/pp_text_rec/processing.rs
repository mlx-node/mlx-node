//! Text Recognition Image Preprocessing
//!
//! Resize cropped text line images to fixed height (48) with variable width,
//! then normalize for the recognition model.
//!
//! **Dynamic width expansion** (matching PaddleOCR's behavior):
//! - For single images: the output width equals the resized width based on aspect
//!   ratio, clamped to `max_width`. No zero-padding is applied.
//! - For batches: the maximum aspect ratio across all images is computed, and all
//!   images are padded to a common width = min(ceil(target_height * max_ratio), max_width).
//!
//! PaddleOCR uses OpenCV which loads images as BGR. Our image loading uses RGB.
//! We swap R<->B channels before normalization to match PaddleOCR's convention.

use crate::array::MxArray;
use image::imageops::FilterType;

/// PaddleOCR uses cv2.INTER_LINEAR (bilinear). The closest equivalent
/// in the `image` crate is `FilterType::Triangle` (bilinear interpolation).
const RESIZE_FILTER: FilterType = FilterType::Triangle;
use napi::bindgen_prelude::*;

/// Image processor for text recognition.
pub struct TextRecImageProcessor {
    /// Target image height
    target_height: u32,
    /// Maximum image width (upper bound to prevent OOM)
    max_width: u32,
    /// Image mean for normalization in BGR order (applied after RGB→BGR channel swap)
    mean: [f32; 3],
    /// Image std for normalization in BGR order (applied after RGB→BGR channel swap)
    std: [f32; 3],
}

impl TextRecImageProcessor {
    pub fn new(target_height: u32, max_width: u32) -> Self {
        Self {
            target_height,
            max_width,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    /// Compute the resized width for a given image, preserving aspect ratio.
    ///
    /// Following PaddleOCR's `resize_norm_img`:
    ///   resized_w = ceil(imgH * (w / h)), clamped to [1, img_w_limit]
    fn compute_resized_width(&self, width: u32, height: u32, img_w_limit: u32) -> u32 {
        if height == 0 {
            return 1;
        }
        let ratio = width as f64 / height as f64;
        let new_w = (self.target_height as f64 * ratio).ceil() as u32;
        new_w.max(1).min(img_w_limit)
    }

    /// Process encoded image bytes for recognition.
    ///
    /// The output width equals the actual resized width (based on aspect ratio),
    /// clamped to `max_width`. No zero-padding is applied for single images.
    ///
    /// Returns pixel_values: [1, target_height, resized_width, 3] normalized float32
    pub fn process(&self, data: &[u8]) -> Result<MxArray> {
        let img = image::load_from_memory(data).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to decode image: {e}"),
            )
        })?;

        let rgb_img = img.to_rgb8();
        self.process_rgb_single(&rgb_img, rgb_img.width(), rgb_img.height())
    }

    /// Process raw RGB image data (single image, no padding).
    ///
    /// The output width equals the actual resized width (based on aspect ratio),
    /// clamped to `max_width`. No zero-padding is applied.
    ///
    /// # Arguments
    /// * `rgb_data` - Raw RGB pixel data
    /// * `width` - Image width
    /// * `height` - Image height
    pub fn process_crop(&self, rgb_data: &[u8], width: u32, height: u32) -> Result<MxArray> {
        let img = image::RgbImage::from_raw(width, height, rgb_data.to_vec())
            .ok_or_else(|| Error::new(Status::InvalidArg, "Invalid RGB data dimensions"))?;
        self.process_rgb_single(&img, width, height)
    }

    /// Process a single RGB image without zero-padding.
    ///
    /// Resizes to target_height, computes width from aspect ratio (clamped to max_width),
    /// and outputs a tensor of shape [1, target_height, resized_width, 3].
    /// Swaps R<->B channels to match PaddleOCR's BGR convention before normalization.
    fn process_rgb_single(
        &self,
        img: &image::RgbImage,
        width: u32,
        height: u32,
    ) -> Result<MxArray> {
        let new_w = self.compute_resized_width(width, height, self.max_width);
        let new_h = self.target_height;

        // Resize
        let resized = image::imageops::resize(img, new_w, new_h, RESIZE_FILTER);

        // Output is exactly the resized dimensions (no padding)
        let out_w = new_w as usize;
        let out_h = new_h as usize;
        let mut pixel_data: Vec<f32> = Vec::with_capacity(out_h * out_w * 3);

        // Channel mapping: RGB->BGR. Pixel[0]=R->out[2], Pixel[1]=G->out[1], Pixel[2]=B->out[0]
        let channel_map = [2usize, 1, 0]; // BGR ordering

        for y in 0..out_h {
            for x in 0..out_w {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for (c, &src_c) in channel_map.iter().enumerate() {
                    let val = pixel[src_c] as f32 / 255.0;
                    let normalized = (val - self.mean[c]) / self.std[c];
                    pixel_data.push(normalized);
                }
            }
        }

        MxArray::from_float32(&pixel_data, &[1, out_h as i64, out_w as i64, 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let proc = TextRecImageProcessor::new(48, 960);
        assert_eq!(proc.target_height, 48);
        assert_eq!(proc.max_width, 960);
    }

    #[test]
    fn test_compute_resized_width_narrow() {
        let proc = TextRecImageProcessor::new(48, 960);
        // Image 100x48 (ratio=2.08), new_w = ceil(48*100/48) = 100
        assert_eq!(proc.compute_resized_width(100, 48, 960), 100);
    }

    #[test]
    fn test_compute_resized_width_wide() {
        let proc = TextRecImageProcessor::new(48, 960);
        // Image 2000x48 (ratio=41.67), new_w = ceil(48*2000/48) = 2000 -> clamped to 960
        assert_eq!(proc.compute_resized_width(2000, 48, 960), 960);
    }

    #[test]
    fn test_compute_resized_width_tall() {
        let proc = TextRecImageProcessor::new(48, 960);
        // Image 320x96 (ratio=3.33), new_w = ceil(48*3.33) = 160
        assert_eq!(proc.compute_resized_width(320, 96, 960), 160);
    }
}
