use crate::array::MxArray;
use image::imageops::FilterType;
use image::{GenericImageView, RgbImage};
use napi::bindgen_prelude::*;

pub struct Gemma4ImageProcessor {
    pub patch_size: i32,
    pub max_soft_tokens: i32,
    pub pooling_kernel_size: i32,
}

pub struct ProcessedGemma4Image {
    pub pixel_values: MxArray,
    pub num_soft_tokens: i32,
}

impl Gemma4ImageProcessor {
    pub fn new(patch_size: i32, max_soft_tokens: i32, pooling_kernel_size: i32) -> Self {
        Self {
            patch_size,
            max_soft_tokens,
            pooling_kernel_size,
        }
    }

    /// Process raw image bytes (PNG/JPEG) into pixel values + soft token count.
    pub fn process_bytes(&self, data: &[u8]) -> Result<ProcessedGemma4Image> {
        let img = image::load_from_memory(data).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to decode image: {e}"),
            )
        })?;

        let (orig_width, orig_height) = img.dimensions();
        let (target_width, target_height) = self.compute_target_size(orig_height, orig_width);

        let resized = if target_height == orig_height && target_width == orig_width {
            img.to_rgb8()
        } else {
            img.resize_exact(target_width, target_height, FilterType::CatmullRom)
                .to_rgb8()
        };

        let pixel_values = rgb_to_mx_array(&resized)?;
        let num_soft_tokens =
            self.compute_num_soft_tokens(target_height as i32, target_width as i32);

        Ok(ProcessedGemma4Image {
            pixel_values,
            num_soft_tokens,
        })
    }

    fn compute_target_size(&self, height: u32, width: u32) -> (u32, u32) {
        let patch_size = self.patch_size as f64;
        let pooling_kernel_size = self.pooling_kernel_size as f64;
        let max_patches = self.max_soft_tokens as f64 * pooling_kernel_size * pooling_kernel_size;
        let target_px = max_patches * patch_size * patch_size;
        let factor = (target_px / (height as f64 * width as f64)).sqrt();
        let side_mult = (self.pooling_kernel_size * self.patch_size) as f64;

        let target_height = (factor * height as f64 / side_mult).floor() as i64 * side_mult as i64;
        let target_width = (factor * width as f64 / side_mult).floor() as i64 * side_mult as i64;

        let side_mult_i = side_mult as i64;
        let max_side_length = self.max_soft_tokens as i64 * side_mult_i;

        let (final_height, final_width) = if target_height == 0 && target_width == 0 {
            // Both zero: should not happen with valid inputs, but handle gracefully
            (side_mult_i, side_mult_i)
        } else if target_height == 0 {
            let th = side_mult_i;
            let tw =
                ((width as f64 / height as f64).floor() as i64 * side_mult_i).min(max_side_length);
            (th, tw.max(side_mult_i))
        } else if target_width == 0 {
            let tw = side_mult_i;
            let th =
                ((height as f64 / width as f64).floor() as i64 * side_mult_i).min(max_side_length);
            (th.max(side_mult_i), tw)
        } else {
            (target_height, target_width)
        };

        (final_width as u32, final_height as u32)
    }

    fn compute_num_soft_tokens(&self, height: i32, width: i32) -> i32 {
        let patch_h = height / self.patch_size;
        let patch_w = width / self.patch_size;
        patch_h * patch_w / (self.pooling_kernel_size * self.pooling_kernel_size)
    }
}

fn rgb_to_mx_array(rgb_img: &RgbImage) -> Result<MxArray> {
    let width = rgb_img.width() as usize;
    let height = rgb_img.height() as usize;

    // Build CHW float32 data (rescale ÷255, no normalization)
    let mut chw_data: Vec<f32> = vec![0.0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x as u32, y as u32);
            for c in 0..3usize {
                chw_data[c * height * width + y * width + x] = pixel[c] as f32 / 255.0;
            }
        }
    }

    MxArray::from_float32(&chw_data, &[1, 3, height as i64, width as i64])
}
