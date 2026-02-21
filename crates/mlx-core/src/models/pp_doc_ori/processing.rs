//! Document Orientation Image Preprocessing
//!
//! Resize to 224x224, normalize with ImageNet mean/std, RGB channel order.
//! This matches PaddleOCR's classification preprocessing.

use crate::array::MxArray;
use image::ImageFormat;
use image::imageops::FilterType;
use napi::bindgen_prelude::*;
use std::io::Cursor;

/// ImageNet normalization constants (RGB order)
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Target input size for PP-LCNet classification
const INPUT_SIZE: u32 = 224;

/// Preprocess encoded image bytes for orientation classification.
///
/// Returns pixel_values as [1, 224, 224, 3] NHWC normalized float32.
pub fn preprocess(data: &[u8]) -> Result<MxArray> {
    let img = image::load_from_memory(data).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to decode image: {e}"),
        )
    })?;

    // Resize to 224x224
    let resized = img.resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle);
    let rgb_img = resized.to_rgb8();

    let h = INPUT_SIZE as usize;
    let w = INPUT_SIZE as usize;
    let mut pixel_data: Vec<f32> = vec![0.0; h * w * 3];

    // RGB order (unlike text detection which uses BGR)
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb_img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - MEAN[c]) / STD[c];
                pixel_data[(y * w + x) * 3 + c] = normalized;
            }
        }
    }

    MxArray::from_float32(&pixel_data, &[1, h as i64, w as i64, 3])
}

/// Rotate encoded image bytes by the given angle (0, 90, 180, 270) and return PNG bytes.
pub fn rotate(data: &[u8], angle: u32) -> Result<Vec<u8>> {
    let img = image::load_from_memory(data).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to decode image: {e}"),
        )
    })?;

    let rotated = match angle {
        0 => img,
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Invalid rotation angle: {angle}. Must be 0, 90, 180, or 270"),
            ));
        }
    };

    let mut buf = Vec::new();
    rotated
        .write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to encode rotated image: {e}"),
            )
        })?;
    Ok(buf)
}

/// Compute the correction angle needed to make the image upright.
/// If the image is detected as rotated by `detected_angle`, we need to rotate
/// by `(360 - detected_angle) % 360` to correct it.
pub fn correction_angle(detected_angle: u32) -> u32 {
    (360 - (detected_angle % 360)) % 360
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correction_angle() {
        assert_eq!(correction_angle(0), 0);
        assert_eq!(correction_angle(90), 270);
        assert_eq!(correction_angle(180), 180);
        assert_eq!(correction_angle(270), 90);
    }
}
