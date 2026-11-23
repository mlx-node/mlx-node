//! Enlarged top-header crop for party/entity name fields on white-margin documents.
//!
//! Exact port of upstream `clairic_runtime/image_party_names.py`.

use std::sync::LazyLock;

use image::RgbImage;
use regex::Regex;
use serde_json::{Map, Value};

use crate::orientation::rgb_to_luma_bt601;

use super::{boolean_runs, crop_rgb, is_truthy, non_bg_bbox, resize_bicubic};

/// Stable view cache identifier matching `image_party_names.VIEW_ID`.
pub const VIEW_ID: &str = "party-name-header-lines-v1";

/// Prompt instruction appended when supplying the enlarged header-lines second view.
pub const VIEW_INSTRUCTION: &str = " The first image is the complete document. The second image enlarges its first printed lines. Use both views of the same source to read the complete requested name; preserve every digit in the name.";

const MAX_HEADER_BANDS: usize = 6;
const MIN_BAND_HEIGHT: usize = 8;
const MAX_BAND_HEIGHT: usize = 32;
const MAX_INK_DENSITY: f64 = 0.22;
const PADDING_PX: u32 = 24;

static NAME_RE: LazyLock<Option<Regex>> = LazyLock::new(|| Regex::new(r"(?i)\bname\b").ok());
static ROLE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:seller|merchant|vendor|customer|bill[- ]?to|recipient|payee|payer|guest|organization|business|company|legal)\b",
    )
    .ok()
});
static OTHER_VALUE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:address|number|identifier|code|date|row|label|amount)\b").ok()
});

/// Return the field names whose instructions request a party/organization name.
///
/// Exact port of `image_party_names.py::fields`.
pub fn fields(schema: &Map<String, Value>) -> Vec<String> {
    let mut selected = Vec::new();
    for (name, spec) in schema {
        if spec.get("type").and_then(Value::as_str) != Some("text")
            || is_truthy(spec.get("knowledge"))
        {
            continue;
        }
        let instructions = match spec.get("instructions") {
            None => "",
            Some(Value::String(s)) => s.as_str(),
            Some(_) => continue,
        };
        let requested = instructions.split(['.', '\n']).next().unwrap_or("");
        let has_name = NAME_RE.as_ref().is_some_and(|re| re.is_match(requested));
        let has_role = ROLE_RE.as_ref().is_some_and(|re| re.is_match(requested));
        let has_other = OTHER_VALUE_RE
            .as_ref()
            .is_some_and(|re| re.is_match(requested));
        if has_name && has_role && !has_other {
            selected.push(name.clone());
        }
    }
    selected
}

/// Crop and upscale (up to `3.0x`, capped at `max_pixels`) the first up-to-6 printed text bands
/// of a white-background document.
///
/// Exact port of `image_party_names.py::header_detail`.
pub fn header_detail(image: &RgbImage, max_pixels: usize) -> Option<RgbImage> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    if w == 0 || h == 0 || image.get_pixel(0, 0).0 != [255, 255, 255] {
        return None;
    }

    let luma = rgb_to_luma_bt601(image);
    let mut ink_count: usize = 0;
    let mut row_has_ink = vec![false; h];
    for y in 0..h {
        let row = &luma[y * w..(y + 1) * w];
        let mut any = false;
        for &px in row {
            if px < 200 {
                ink_count += 1;
                any = true;
            }
        }
        row_has_ink[y] = any;
    }

    let total_pixels = w.saturating_mul(h);
    if total_pixels == 0 || (ink_count as f64) / (total_pixels as f64) > MAX_INK_DENSITY {
        return None;
    }

    let all_bands = boolean_runs(&row_has_ink);
    let bands = &all_bands[..all_bands.len().min(MAX_HEADER_BANDS)];
    if bands.is_empty()
        || bands
            .iter()
            .any(|&(top, bottom)| !(MIN_BAND_HEIGHT..=MAX_BAND_HEIGHT).contains(&(bottom - top)))
    {
        return None;
    }

    let first_top = bands[0].0 as u32;
    let last_bottom = bands[bands.len() - 1].1 as u32;
    let top = first_top.saturating_sub(PADDING_PX);
    let bottom = (last_bottom.saturating_add(PADDING_PX)).min(image.height());

    let block = crop_rgb(image, 0, top, image.width(), bottom);
    let (bbox_left, _, bbox_right, _) = non_bg_bbox(&block, [255, 255, 255])?;

    let left = bbox_left.saturating_sub(PADDING_PX);
    let right = (bbox_right.saturating_add(PADDING_PX)).min(block.width());
    let trimmed = crop_rgb(&block, left, 0, right, block.height());
    let bw = trimmed.width();
    let bh = trimmed.height();
    if bw == 0 || bh == 0 {
        return None;
    }

    let scale = 3.0_f64.min(((max_pixels as f64) / (f64::from(bw) * f64::from(bh))).sqrt());
    let new_w = ((f64::from(bw) * scale) as u32).max(1);
    let new_h = ((f64::from(bh) * scale) as u32).max(1);
    Some(resize_bicubic(&trimmed, new_w, new_h))
}

/// Return `[image, detail]` when [`header_detail`] succeeds, else `[image]`.
pub fn views(image: &RgbImage, max_pixels: usize) -> Vec<RgbImage> {
    match header_detail(image, max_pixels) {
        Some(detail) => vec![image.clone(), detail],
        None => vec![image.clone()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crops::DEFAULT_MAX_PIXELS;
    use image::Rgb;
    use serde_json::json;

    #[test]
    fn selects_party_name_fields_and_rejects_other_values() {
        let schema: Map<String, Value> = serde_json::from_value(json!({
            "vendor": { "type": "text", "instructions": "Extract the vendor legal name. Preserve digits." },
            "vendor_addr": { "type": "text", "instructions": "Extract the vendor name and address." },
            "invoice_no": { "type": "text", "instructions": "Extract the invoice number." }
        }))
        .unwrap();
        assert_eq!(fields(&schema), vec!["vendor".to_string()]);
    }

    #[test]
    fn crops_and_upscales_first_printed_bands() {
        let mut img = RgbImage::from_pixel(800, 1000, Rgb([255, 255, 255]));
        // Draw 3 horizontal text bands of height 16px each
        for band_y in [60u32, 100, 140] {
            for y in band_y..(band_y + 16) {
                for x in 100..300 {
                    img.put_pixel(x, y, Rgb([20, 20, 20]));
                }
            }
        }
        let detail = header_detail(&img, DEFAULT_MAX_PIXELS).unwrap();
        // Horizontal trimmed width: (300 - 100) + 48 = 248 -> *3 = 744
        // Vertical crop: (156 + 24) - (60 - 24) = 144 -> *3 = 432
        assert_eq!(detail.dimensions(), (248 * 3, 144 * 3));
        assert_eq!(views(&img, DEFAULT_MAX_PIXELS).len(), 2);
    }
}
