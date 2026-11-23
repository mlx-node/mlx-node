//! Shared Image-to-NHWC Float32 Normalization Helper for CNN Models

use crate::array::MxArray;
use image::RgbImage;
use napi::bindgen_prelude::*;

/// Convert an `RgbImage` to a `[1, H, W, 3]` NHWC float32 `MxArray`,
/// rescaling to `[0, 1]` with optional channel swap (`RGB -> BGR`) and
/// optional per-channel `(val - mean[c]) / std[c]` normalization.
pub fn rgb_to_nhwc_f32(
    rgb_img: &RgbImage,
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
    swap_rb: bool,
) -> Result<MxArray> {
    let h = rgb_img.height() as usize;
    let w = rgb_img.width() as usize;

    let pixel_data: Vec<f32> = if mean.is_none() && std.is_none() && !swap_rb {
        rgb_img.as_raw().iter().map(|&b| b as f32 / 255.0).collect()
    } else {
        let channel_map = if swap_rb { [2usize, 1, 0] } else { [0, 1, 2] };
        let mean = mean.unwrap_or([0.0; 3]);
        let std = std.unwrap_or([1.0; 3]);
        let mut buf = Vec::with_capacity(h * w * 3);
        for rgb in rgb_img.as_raw().as_chunks::<3>().0 {
            for (c, &src_c) in channel_map.iter().enumerate() {
                let val = rgb[src_c] as f32 / 255.0;
                buf.push((val - mean[c]) / std[c]);
            }
        }
        buf
    };

    MxArray::from_float32(&pixel_data, &[1, h as i64, w as i64, 3])
}
