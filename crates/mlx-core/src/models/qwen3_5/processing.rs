/// Image Processing for Qwen3.5-VL
///
/// Handles image preprocessing including smart resizing, normalization,
/// and patch extraction. Adapted from PaddleOCR-VL processing module
/// with Qwen3.5-VL specific parameters.
use crate::array::MxArray;
use crate::models::paddleocr_vl::processing::{
    ImageProcessorConfig, ProcessedImage, ProcessedImages, aggregate_processed_images, smart_resize,
};
use image::ImageReader;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use napi::bindgen_prelude::*;
use std::io::Cursor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen35VLImageGeometry {
    resized_height: usize,
    resized_width: usize,
    grid_t: i32,
    grid_h: i32,
    grid_w: i32,
}

/// Compute the merged language-token count for one Qwen3.5 vision grid.
///
/// Both the non-mutating capacity planner and the real image-processing path
/// use this helper so pre-SSE prompt sizing cannot drift from the number of
/// image embeddings produced during prefill.
pub(crate) fn merged_image_token_count(
    grid_t: i32,
    grid_h: i32,
    grid_w: i32,
    spatial_merge_size: i32,
) -> Result<usize> {
    if spatial_merge_size <= 0 {
        return Err(Error::from_reason(format!(
            "spatial_merge_size must be positive, got {spatial_merge_size}"
        )));
    }
    if grid_t < 0 || grid_h < 0 || grid_w < 0 {
        return Err(Error::from_reason(format!(
            "vision grid dimensions must be non-negative, got [{grid_t}, {grid_h}, {grid_w}]"
        )));
    }

    let patches = i64::from(grid_t)
        .checked_mul(i64::from(grid_h))
        .and_then(|n| n.checked_mul(i64::from(grid_w)))
        .ok_or_else(|| Error::from_reason("vision grid patch count overflow"))?;
    let merge = i64::from(spatial_merge_size);
    let merge_factor = merge
        .checked_mul(merge)
        .ok_or_else(|| Error::from_reason("spatial merge factor overflow"))?;
    usize::try_from(patches / merge_factor)
        .map_err(|_| Error::from_reason("merged image token count overflow"))
}

/// Qwen3.5-VL image processor configuration
fn qwen35_vl_processor_config() -> ImageProcessorConfig {
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
    fn resize_factor(&self) -> i32 {
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
        let processed: Vec<ProcessedImage> = images
            .iter()
            .map(|data| self.process_bytes(data))
            .collect::<Result<_>>()?;
        aggregate_processed_images(processed)
    }

    /// Plan the exact merged language-token count for each encoded image
    /// without decoding pixels into f32 patches or allocating any MLX arrays.
    ///
    /// Capacity preflight only depends on encoded dimensions, smart-resize
    /// policy, patch size, and the model's spatial merge size. Inspecting image
    /// dimensions through `ImageReader::into_dimensions` avoids the transient
    /// RGB/f32 tensors created by [`Self::process_bytes`] while sharing the same
    /// geometry helper as the real processing path.
    pub(crate) fn plan_merged_token_counts(
        &self,
        images: &[&[u8]],
        spatial_merge_size: i32,
    ) -> Result<Vec<usize>> {
        images
            .iter()
            .map(|data| {
                let reader = ImageReader::new(Cursor::new(*data))
                    .with_guessed_format()
                    .map_err(|e| {
                        Error::new(
                            Status::GenericFailure,
                            format!("Failed to inspect image format: {e}"),
                        )
                    })?;
                let (width, height) = reader.into_dimensions().map_err(|e| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Failed to inspect image dimensions: {e}"),
                    )
                })?;
                let geometry = self.plan_geometry(width, height)?;
                merged_image_token_count(
                    geometry.grid_t,
                    geometry.grid_h,
                    geometry.grid_w,
                    spatial_merge_size,
                )
            })
            .collect()
    }

    fn plan_geometry(&self, orig_width: u32, orig_height: u32) -> Result<Qwen35VLImageGeometry> {
        // Smart resize to maintain aspect ratio within pixel bounds. This is
        // the single geometry source used by both planning and processing.
        let (new_height, new_width) = smart_resize(
            orig_height as i32,
            orig_width as i32,
            self.resize_factor(),
            self.config.min_pixels,
            self.config.max_pixels,
        )?;
        let patch_size = self.config.patch_size;
        Ok(Qwen35VLImageGeometry {
            resized_height: new_height as usize,
            resized_width: new_width as usize,
            grid_t: 1,
            grid_h: new_height / patch_size,
            grid_w: new_width / patch_size,
        })
    }

    /// Internal: Process a loaded image
    fn process_image(&self, img: DynamicImage) -> Result<ProcessedImage> {
        let (orig_width, orig_height) = img.dimensions();

        let geometry = self.plan_geometry(orig_width, orig_height)?;
        let new_height = geometry.resized_height;
        let new_width = geometry.resized_width;

        let resized = img.resize_exact(new_width as u32, new_height as u32, FilterType::CatmullRom);
        let rgb_img: RgbImage = resized.to_rgb8();

        // Convert to float and normalize
        let (height, width) = (new_height, new_width);
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
        let grid_h = geometry.grid_h as usize;
        let grid_w = geometry.grid_w as usize;
        let grid_t = geometry.grid_t as usize; // temporal dimension for images
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

#[cfg(test)]
mod tests {
    use super::{ImageProcessorConfig, Qwen35VLImageProcessor};
    use crate::models::gemma4::image_processor::Gemma4ImageProcessor;
    use image::{DynamicImage, ImageFormat, RgbImage};
    use std::io::Cursor;

    // Valid 1x1 GIF89a image with a two-entry global color table.
    const ONE_PIXEL_GIF: &[u8] = &[
        0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00,
        0x00, 0xff, 0xff, 0xff, 0x21, 0xf9, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44, 0x01, 0x00, 0x3b,
    ];

    #[test]
    fn qwen35_and_gemma4_processors_accept_gif_bytes() {
        let qwen = Qwen35VLImageProcessor::new(Some(ImageProcessorConfig {
            min_pixels: 16 * 16,
            max_pixels: 16 * 16,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 1,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            do_rescale: true,
            do_normalize: true,
        }));
        let qwen_image = qwen
            .process_bytes(ONE_PIXEL_GIF)
            .expect("Qwen3.5 GIF decode");
        assert_eq!(qwen_image.image_grid_thw(), vec![1, 1, 1]);

        let gemma = Gemma4ImageProcessor::new(1, 1, 1);
        let gemma_image = gemma
            .process_bytes(ONE_PIXEL_GIF)
            .expect("Gemma4 GIF decode");
        assert_eq!(gemma_image.num_soft_tokens, 1);
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = DynamicImage::ImageRgb8(RgbImage::new(width, height));
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, ImageFormat::Png)
            .expect("encode test PNG");
        encoded.into_inner()
    }

    #[test]
    fn cpu_geometry_plan_matches_real_processing_grids() {
        let processor = Qwen35VLImageProcessor::new(Some(ImageProcessorConfig {
            min_pixels: 32 * 32,
            max_pixels: 64 * 64,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            do_rescale: true,
            do_normalize: true,
        }));
        let wide = png(64, 32);
        let square = png(32, 32);
        let planned = processor
            .plan_merged_token_counts(&[&wide, &square], 2)
            .expect("plan image tokens");

        let wide_processed = processor.process_bytes(&wide).expect("process wide image");
        let square_processed = processor
            .process_bytes(&square)
            .expect("process square image");

        assert_eq!(wide_processed.image_grid_thw(), vec![1, 2, 4]);
        assert_eq!(square_processed.image_grid_thw(), vec![1, 2, 2]);
        assert_eq!(planned, vec![2, 1]);
    }
}
