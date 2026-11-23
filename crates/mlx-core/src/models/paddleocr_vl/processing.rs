use crate::vision::processing::extract_image_patches;
/**
 * Image Processing for PaddleOCR-VL (Internal)
 *
 * Handles image preprocessing including smart resizing,
 * normalization, and patch extraction.
 *
 * This module is internal - users interact via VLModel::chat() with imagePaths config.
 */
pub use crate::vision::processing::{
    ImageProcessorConfig, ProcessedImage, ProcessedImages, aggregate_processed_images, smart_resize,
};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use napi::bindgen_prelude::*;

/// Image Processor for PaddleOCR-VL (internal)
///
/// Users should use VLModel::chat() with imagePaths config instead.
pub struct ImageProcessor {
    config: ImageProcessorConfig,
}

impl ImageProcessor {
    pub fn new(config: Option<ImageProcessorConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
        }
    }

    /// Get the resize factor (patch_size * merge_size)
    pub fn resize_factor(&self) -> i32 {
        self.config.patch_size * self.config.merge_size
    }

    /// Process an image from encoded bytes
    pub fn process_bytes(&self, data: &[u8]) -> Result<ProcessedImage> {
        let img = image::load_from_memory(data).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to decode image: {e}"),
            )
        })?;
        self.process_image(img)
    }

    /// Process multiple images from encoded bytes
    ///
    /// Used internally by VLModel::chat() - users pass images to chat() directly.
    pub fn process_many(&self, images: &[&[u8]]) -> Result<ProcessedImages> {
        let processed: Vec<ProcessedImage> = images
            .iter()
            .map(|data| self.process_bytes(data))
            .collect::<Result<_>>()?;
        aggregate_processed_images(processed)
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

        // Convert to RGB and extract normalized patches
        let rgb_img: RgbImage = resized.to_rgb8();
        extract_image_patches(&rgb_img, &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_processor() {
        let processor = ImageProcessor::new(None);
        assert_eq!(processor.resize_factor(), 28); // 14 * 2
    }
}
