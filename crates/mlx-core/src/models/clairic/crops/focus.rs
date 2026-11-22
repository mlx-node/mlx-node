//! Sparse-document focusing and verbatim scalar-field preservation.
//!
//! Exact port of upstream `clairic_runtime/image_focus.py`.

use image::RgbImage;
use serde_json::{Map, Value};

use super::{DEFAULT_MAX_PIXELS, crop_rgb, is_truthy, non_bg_bbox, resize_bicubic};

/// Instruction suffix appended to eligible text fields by [`preserve_scalar_fields`].
pub const COPY_INSTRUCTION: &str = " Copy the complete printed value verbatim, including every digit, punctuation and suffix. Do not shorten, sanitize or canonicalize it.";

/// Margin in pixels added around the non-white content bounding box.
const FOCUS_PADDING_PX: u32 = 24;

/// Return `true` if a single field spec qualifies for [`COPY_INSTRUCTION`] preservation.
pub fn should_preserve_scalar_field(spec: &Value) -> bool {
    let is_text = spec.get("type").and_then(Value::as_str) == Some("text");
    let has_knowledge = is_truthy(spec.get("knowledge"));
    let instructions_ok = match spec.get("instructions") {
        None => true,
        Some(v) => v.is_string(),
    };
    is_text && !has_knowledge && instructions_ok
}

/// Append [`COPY_INSTRUCTION`] to every non-knowledge `"text"` field with string instructions.
///
/// Exact port of `image_focus.py::preserve_scalar_fields`.
pub fn preserve_scalar_fields(fields: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::with_capacity(fields.len());
    for (name, spec) in fields {
        if should_preserve_scalar_field(spec)
            && let Some(obj) = spec.as_object()
        {
            let mut cloned = obj.clone();
            let current = obj
                .get("instructions")
                .and_then(Value::as_str)
                .unwrap_or("");
            cloned.insert(
                "instructions".to_string(),
                Value::String(format!("{current}{COPY_INSTRUCTION}")),
            );
            out.insert(name.clone(), Value::Object(cloned));
            continue;
        }
        out.insert(name.clone(), spec.clone());
    }
    out
}

/// Crop away white margins (`(255, 255, 255)`), pad by 24px, check content area <= 50%,
/// and upscale with bicubic (`CatmullRom`) up to `max_scale` and `max_pixels`.
///
/// Returns `Some(focused)` when the image was actually cropped & upscaled, or `None`
/// when the original image should be retained unchanged (mirroring `focused is not image`
/// in upstream Python).
pub fn focus_document(image: &RgbImage, max_pixels: usize, max_scale: f64) -> Option<RgbImage> {
    let (left, top, right, bottom) = non_bg_bbox(image, [255, 255, 255])?;
    let left = left.saturating_sub(FOCUS_PADDING_PX);
    let top = top.saturating_sub(FOCUS_PADDING_PX);
    let right = (right.saturating_add(FOCUS_PADDING_PX)).min(image.width());
    let bottom = (bottom.saturating_add(FOCUS_PADDING_PX)).min(image.height());

    let width = right.saturating_sub(left);
    let height = bottom.saturating_sub(top);
    if width == 0 || height == 0 {
        return None;
    }

    let crop_area = f64::from(width) * f64::from(height);
    let full_area = f64::from(image.width()) * f64::from(image.height());
    if crop_area > full_area / 2.0 {
        return None;
    }

    let scale = max_scale.min(((max_pixels as f64) / crop_area).sqrt());
    if scale <= 1.0 {
        return None;
    }

    let focused = crop_rgb(image, left, top, right, bottom);
    let new_w = ((f64::from(width) * scale) as u32).max(1);
    let new_h = ((f64::from(height) * scale) as u32).max(1);
    Some(resize_bicubic(&focused, new_w, new_h))
}

/// Focus `image` with upstream default parameters (`max_pixels = 1_600_000`, `max_scale = 2.0`),
/// returning the focused crop if applicable or cloning the original image otherwise.
pub fn focus_document_or_clone(image: &RgbImage, max_pixels: usize, max_scale: f64) -> RgbImage {
    focus_document(image, max_pixels, max_scale).unwrap_or_else(|| image.clone())
}

/// Focus with default settings (`1_600_000` pixels, `2.0x` max scale).
pub fn focus_document_default(image: &RgbImage) -> Option<RgbImage> {
    focus_document(image, DEFAULT_MAX_PIXELS, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use serde_json::json;

    #[test]
    fn focuses_sparse_white_margin_invoice() {
        let mut img = RgbImage::from_pixel(1000, 1000, Rgb([255, 255, 255]));
        // Small 200x150 dark block in center -> with 24px pad = 248x198 = 49_104 <= 500_000
        for y in 400..550 {
            for x in 400..600 {
                img.put_pixel(x, y, Rgb([15, 15, 15]));
            }
        }
        let focused = focus_document(&img, DEFAULT_MAX_PIXELS, 2.0).unwrap();
        assert_eq!(focused.width(), 248 * 2);
        assert_eq!(focused.height(), 198 * 2);
    }

    #[test]
    fn skips_dense_or_blank_images() {
        let blank = RgbImage::from_pixel(400, 400, Rgb([255, 255, 255]));
        assert!(focus_document_default(&blank).is_none());

        let dense = RgbImage::from_pixel(400, 400, Rgb([30, 30, 30]));
        assert!(focus_document_default(&dense).is_none());
    }

    #[test]
    fn preserve_scalar_fields_updates_only_non_knowledge_text() {
        let schema: Map<String, Value> = serde_json::from_value(json!({
            "invoice_id": { "type": "text", "instructions": "Read invoice id." },
            "total": { "type": "number", "instructions": "Total amount." },
            "known": { "type": "text", "knowledge": true, "instructions": "Static." }
        }))
        .unwrap();
        let updated = preserve_scalar_fields(&schema);
        assert_eq!(
            updated["invoice_id"]["instructions"].as_str().unwrap(),
            format!("Read invoice id.{COPY_INSTRUCTION}")
        );
        assert_eq!(
            updated["total"]["instructions"].as_str().unwrap(),
            "Total amount."
        );
        assert_eq!(
            updated["known"]["instructions"].as_str().unwrap(),
            "Static."
        );
    }
}
