//! UVDoc Image Preprocessing and Postprocessing
//!
//! Preprocessing: Resize to 488x712, normalize to [0, 1] float32.
//! Postprocessing: Apply displacement field via bilinear interpolation.

use crate::array::MxArray;
use image::imageops::FilterType;
use image::{ImageFormat, RgbImage};
use napi::bindgen_prelude::*;
use std::io::Cursor;

use super::config::UVDocConfig;

/// Preprocess encoded image bytes for UVDoc inference.
///
/// Returns (pixel_values, original_width, original_height).
/// pixel_values: [1, H, W, 3] NHWC float32 in [0, 1]
pub fn preprocess(data: &[u8], config: &UVDocConfig) -> Result<(MxArray, u32, u32)> {
    let img = image::load_from_memory(data).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to decode image: {e}"),
        )
    })?;

    let orig_w = img.width();
    let orig_h = img.height();

    // img_size is (width, height) per cv2.resize convention
    let (target_w, target_h) = config.img_size;
    let resized = img.resize_exact(target_w, target_h, FilterType::Triangle);
    let rgb_img = resized.to_rgb8();

    let h = target_h as usize;
    let w = target_w as usize;
    let mut pixel_data: Vec<f32> = vec![0.0; h * w * 3];

    // Normalize to [0, 1] (UVDoc uses simple /255 normalization)
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb_img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                pixel_data[(y * w + x) * 3 + c] = pixel[c] as f32 / 255.0;
            }
        }
    }

    let pixel_values = MxArray::from_float32(&pixel_data, &[1, h as i64, w as i64, 3])?;
    Ok((pixel_values, orig_w, orig_h))
}

/// Apply a 2D displacement field to unwarp encoded image bytes using bilinear interpolation.
///
/// The displacement field is a grid of 2D positions indicating where each output
/// pixel should sample from in the input image (in normalized [-1, 1] coordinates).
///
/// # Arguments
/// * `data` - Encoded image bytes (PNG/JPEG)
/// * `grid_data` - Displacement field as flat f32 array in NHWC layout: [1, Gh, Gw, 2]
///   At each (y, x) position: [x_coord, y_coord]
/// * `grid_h` - Grid height
/// * `grid_w` - Grid width
///
/// # Returns
/// * PNG-encoded bytes of the unwarped image
pub fn apply_displacement_field(
    data: &[u8],
    grid_data: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> Result<Vec<u8>> {
    let img = image::load_from_memory(data).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to decode image: {e}"),
        )
    })?;

    let rgb_img = img.to_rgb8();
    let (img_w, img_h) = (rgb_img.width(), rgb_img.height());

    let expected_len = grid_h * grid_w * 2;
    if grid_data.len() < expected_len {
        return Err(Error::new(
            Status::GenericFailure,
            format!(
                "grid_data length {} < expected {} (grid {}x{}x2)",
                grid_data.len(),
                expected_len,
                grid_h,
                grid_w
            ),
        ));
    }

    // Upsample the grid from (Gh, Gw) to (img_h, img_w) using bilinear interpolation
    let out_h = img_h as usize;
    let out_w = img_w as usize;

    let mut output = RgbImage::new(img_w, img_h);

    for oy in 0..out_h {
        for ox in 0..out_w {
            // Map output pixel to grid coordinates
            let gx_f = (ox as f64 / (out_w - 1).max(1) as f64) * (grid_w - 1) as f64;
            let gy_f = (oy as f64 / (out_h - 1).max(1) as f64) * (grid_h - 1) as f64;

            // Bilinear interpolation in the grid
            let gx0 = gx_f.floor() as usize;
            let gy0 = gy_f.floor() as usize;
            let gx1 = (gx0 + 1).min(grid_w - 1);
            let gy1 = (gy0 + 1).min(grid_h - 1);
            let fx = gx_f - gx0 as f64;
            let fy = gy_f - gy0 as f64;

            // grid_data layout: NHWC [1, Gh, Gw, 2] → offset = gy * Gw * 2 + gx * 2 + ch
            let sample_coords = |ch: usize, gy: usize, gx: usize| -> f64 {
                grid_data[gy * grid_w * 2 + gx * 2 + ch] as f64
            };

            // Interpolate x and y displacement from grid
            let mut sample = [0.0f64; 2];
            for (ch, s) in sample.iter_mut().enumerate() {
                let v00 = sample_coords(ch, gy0, gx0);
                let v01 = sample_coords(ch, gy0, gx1);
                let v10 = sample_coords(ch, gy1, gx0);
                let v11 = sample_coords(ch, gy1, gx1);
                *s = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }

            // Convert from normalized [-1, 1] to pixel coordinates (align_corners=True)
            let src_x =
                ((sample[0] + 1.0) / 2.0 * (img_w - 1) as f64).clamp(0.0, (img_w - 1) as f64);
            let src_y =
                ((sample[1] + 1.0) / 2.0 * (img_h - 1) as f64).clamp(0.0, (img_h - 1) as f64);

            // Bilinear interpolation in source image
            let sx0 = src_x.floor() as u32;
            let sy0 = src_y.floor() as u32;
            let sx1 = (sx0 + 1).min(img_w - 1);
            let sy1 = (sy0 + 1).min(img_h - 1);
            let sfx = src_x - sx0 as f64;
            let sfy = src_y - sy0 as f64;

            let p00 = rgb_img.get_pixel(sx0, sy0);
            let p01 = rgb_img.get_pixel(sx1, sy0);
            let p10 = rgb_img.get_pixel(sx0, sy1);
            let p11 = rgb_img.get_pixel(sx1, sy1);

            let mut pixel = [0u8; 3];
            for c in 0..3 {
                let v = p00[c] as f64 * (1.0 - sfx) * (1.0 - sfy)
                    + p01[c] as f64 * sfx * (1.0 - sfy)
                    + p10[c] as f64 * (1.0 - sfx) * sfy
                    + p11[c] as f64 * sfx * sfy;
                pixel[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            output.put_pixel(ox as u32, oy as u32, image::Rgb(pixel));
        }
    }

    let mut buf = Vec::new();
    output
        .write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to encode unwarped image: {e}"),
            )
        })?;
    Ok(buf)
}
