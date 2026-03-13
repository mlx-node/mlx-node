/// Image Processing for Qwen3.5-VL
///
/// Handles image preprocessing including smart resizing, normalization,
/// and patch extraction. Adapted from PaddleOCR-VL processing module
/// with Qwen3.5-VL specific parameters.
use crate::array::MxArray;
use crate::models::paddleocr_vl::processing::{
    ImageProcessorConfig, ProcessedImage, ProcessedImages, smart_resize,
};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use napi::bindgen_prelude::*;

/// Qwen3.5-VL image processor configuration
pub fn qwen35_vl_processor_config() -> ImageProcessorConfig {
    ImageProcessorConfig {
        min_pixels: 147384,
        max_pixels: 2822400,
        patch_size: 16,
        temporal_patch_size: 2, // Qwen3.5-VL uses temporal_patch_size=2
        merge_size: 2,
        image_mean: vec![0.5, 0.5, 0.5],
        image_std: vec![0.5, 0.5, 0.5],
        do_rescale: true,
        do_normalize: true,
    }
}

/// Image processor for Qwen3.5-VL
///
/// Processes images into patches suitable for the vision encoder.
/// For images (not video), the temporal dimension is handled by
/// duplicating the frame (temporal_patch_size=2).
pub struct Qwen35VLImageProcessor {
    config: ImageProcessorConfig,
}

impl Qwen35VLImageProcessor {
    pub fn new(config: Option<ImageProcessorConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(qwen35_vl_processor_config),
        }
    }

    /// Get the resize factor (patch_size * merge_size)
    pub fn resize_factor(&self) -> i32 {
        self.config.patch_size * self.config.merge_size
    }

    /// Process a single image from encoded bytes
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
    pub fn process_many(&self, images: &[&[u8]]) -> Result<ProcessedImages> {
        if images.is_empty() {
            return Err(Error::new(Status::InvalidArg, "images cannot be empty"));
        }

        let mut all_pixel_values: Vec<MxArray> = Vec::new();
        let mut all_grid_thw: Vec<i32> = Vec::new();

        for data in images {
            let processed = self.process_bytes(data)?;
            all_pixel_values.push(processed.pixel_values());
            all_grid_thw.extend_from_slice(&processed.image_grid_thw());
        }

        let pixel_values = if all_pixel_values.len() == 1 {
            all_pixel_values.remove(0)
        } else {
            let refs: Vec<&MxArray> = all_pixel_values.iter().collect();
            MxArray::concatenate_many(refs, Some(0))?
        };

        let num_images = images.len() as i64;
        let grid_thw = MxArray::from_int32(&all_grid_thw, &[num_images, 3])?;

        Ok(ProcessedImages::new(pixel_values, grid_thw))
    }

    /// Internal: Process a loaded image
    fn process_image(&self, img: DynamicImage) -> Result<ProcessedImage> {
        let (orig_width, orig_height) = img.dimensions();

        // Smart resize to maintain aspect ratio within pixel bounds
        let (new_height, new_width) = smart_resize(
            orig_height as i32,
            orig_width as i32,
            self.resize_factor(),
            self.config.min_pixels,
            self.config.max_pixels,
        )?;

        let resized = img.resize_exact(new_width as u32, new_height as u32, FilterType::CatmullRom);
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
                    if self.config.do_rescale {
                        value /= 255.0;
                    }
                    if self.config.do_normalize {
                        value = (value - mean[c]) / std[c];
                    }
                    pixel_data.push(value);
                }
            }
        }

        // Reshape to patches
        // Qwen3.5-VL uses patch_size=16.
        // For images (not video), temporal_patch_size=2 means we duplicate the frame.
        let patch_size = self.config.patch_size as usize;
        let grid_h = height / patch_size;
        let grid_w = width / patch_size;
        let grid_t = 1; // temporal dimension for images
        let num_patches = grid_t * grid_h * grid_w;

        // Reorder data into patches: [num_patches, C, patch_h, patch_w]
        let mut patch_data: Vec<f32> =
            Vec::with_capacity(num_patches * channels * patch_size * patch_size);

        for ph in 0..grid_h {
            for pw in 0..grid_w {
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
}
