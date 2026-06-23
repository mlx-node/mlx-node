use crate::array::MxArray;
use image::imageops::FilterType;
use image::{GenericImageView, RgbImage};
use napi::bindgen_prelude::*;

pub struct Gemma4ImageProcessor {
    pub patch_size: i32,
    pub max_soft_tokens: i32,
    pub pooling_kernel_size: i32,
    /// Encoder-free unified path. When set, `process_bytes` patchifies at
    /// `model_patch_size` (= patch_size × pooling_kernel_size) and returns
    /// flattened `[n, model_patch_size^2 * 3]` patches plus `[n, 2]` position
    /// ids, with `num_soft_tokens = n` (no pooling division). The SigLIP path
    /// (when `None`) is unchanged.
    pub unified_model_patch_size: Option<i32>,
}

pub struct ProcessedGemma4Image {
    pub pixel_values: MxArray,
    pub num_soft_tokens: i32,
    /// `[n, 2]` int32 `(x, y)` patch grid coordinates. `Some` only for the
    /// unified encoder-free path; `None` for the SigLIP vision tower (which
    /// computes its own patch grid inside the encoder).
    pub position_ids: Option<MxArray>,
}

impl Gemma4ImageProcessor {
    pub fn new(patch_size: i32, max_soft_tokens: i32, pooling_kernel_size: i32) -> Self {
        Self {
            patch_size,
            max_soft_tokens,
            pooling_kernel_size,
            unified_model_patch_size: None,
        }
    }

    /// Construct the encoder-free unified image processor.
    pub fn new_unified(
        patch_size: i32,
        max_soft_tokens: i32,
        pooling_kernel_size: i32,
        model_patch_size: i32,
    ) -> Self {
        Self {
            patch_size,
            max_soft_tokens,
            pooling_kernel_size,
            unified_model_patch_size: Some(model_patch_size),
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

        if let Some(model_patch_size) = self.unified_model_patch_size {
            return self.process_unified(&resized, model_patch_size);
        }

        let pixel_values = rgb_to_mx_array(&resized)?;
        let num_soft_tokens =
            self.compute_num_soft_tokens(target_height as i32, target_width as i32);

        Ok(ProcessedGemma4Image {
            pixel_values,
            num_soft_tokens,
            position_ids: None,
        })
    }

    /// Encoder-free patchify: turn a resized RGB image into `[n, p^2 * 3]`
    /// flattened patches and `[n, 2]` `(x, y)` position ids, matching
    /// `_convert_image_to_model_patches` (reshape → transpose(1,3,2,4,0) →
    /// flatten, with a meshgrid(arange(pw), arange(ph), indexing="xy")).
    fn process_unified(
        &self,
        rgb_img: &RgbImage,
        model_patch_size: i32,
    ) -> Result<ProcessedGemma4Image> {
        let p = model_patch_size as usize;
        let width = rgb_img.width() as usize;
        let height = rgb_img.height() as usize;
        let patch_h = height / p;
        let patch_w = width / p;
        let num_patches = patch_h * patch_w;
        let patch_dim = p * p * 3;

        // Build patches in (patch_row, patch_col, py, px, channel) order, which
        // is the layout produced by reshaping channel-first pixels to
        // (C, ph, p, pw, p) and transposing to (ph, pw, p, p, C).
        let mut patch_data = vec![0.0f32; num_patches * patch_dim];
        for pr in 0..patch_h {
            for pc in 0..patch_w {
                let patch_idx = pr * patch_w + pc;
                let base = patch_idx * patch_dim;
                for py in 0..p {
                    for px in 0..p {
                        let y = pr * p + py;
                        let x = pc * p + px;
                        let pixel = rgb_img.get_pixel(x as u32, y as u32);
                        for c in 0..3usize {
                            // flatten order within a patch: (py, px, c)
                            let offset = (py * p + px) * 3 + c;
                            patch_data[base + offset] = pixel[c] as f32 / 255.0;
                        }
                    }
                }
            }
        }
        let pixel_values =
            MxArray::from_float32(&patch_data, &[num_patches as i64, patch_dim as i64])?;

        // position_ids: meshgrid(arange(pw), arange(ph), indexing="xy")
        // → row-major over (ph, pw) with each entry (x=col, y=row).
        let mut pos_data: Vec<i32> = Vec::with_capacity(num_patches * 2);
        for pr in 0..patch_h {
            for pc in 0..patch_w {
                pos_data.push(pc as i32); // x
                pos_data.push(pr as i32); // y
            }
        }
        let position_ids = MxArray::from_int32(&pos_data, &[num_patches as i64, 2])?;

        Ok(ProcessedGemma4Image {
            pixel_values,
            num_soft_tokens: num_patches as i32,
            position_ids: Some(position_ids),
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
