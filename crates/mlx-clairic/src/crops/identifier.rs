//! Identifier detail, header, and adaptive-background content-band crops.
//!
//! Exact port of upstream `clairic_runtime/image_identifier.py`.

use std::sync::LazyLock;

use image::RgbImage;
use regex::Regex;
use serde_json::{Map, Value};

use super::{crop_rgb, focus, is_truthy, non_bg_bbox, resize_bicubic};

/// Stable view cache identifier matching `image_identifier.VIEW_ID`.
pub const VIEW_ID: &str = "identifier-content-band-v2";

const CONTENT_BG_SAMPLE_SIZE: u32 = 64;
const CONTENT_MIN_BG_CHANNEL: u8 = 160;
const CONTENT_DIFF_THRESHOLD: u8 = 32;
const PADDING_PX: u32 = 24;

static ROLE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:invoice|receipt|purchase order|order|tracking|reference|transaction|account|application)\s+(?:number|identifier|id|code)\b",
    )
    .ok()
});
static EXCLUDED_ROLE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:not|exclude|excluding|ignore|rather than)\b[^.;:\n]*$").ok()
});
static LABEL_SEP_RE: LazyLock<Option<Regex>> = LazyLock::new(|| Regex::new(r"[_-]+").ok());

/// Return `true` if `value` positively mentions an identifier role (not preceded by exclusion).
pub fn requested_identifier(value: &str) -> bool {
    let Some(role_re) = ROLE_RE.as_ref() else {
        return false;
    };
    role_re.find_iter(value).any(|m| {
        !EXCLUDED_ROLE_RE
            .as_ref()
            .is_some_and(|excl| excl.is_match(&value[..m.start()]))
    })
}

/// Select non-knowledge `"text"` fields requesting an identifier by name or instructions.
///
/// Exact port of `image_identifier.py::fields`.
pub fn fields(schema: &Map<String, Value>) -> Vec<String> {
    let mut selected = Vec::new();
    for (name, spec) in schema {
        if spec.get("type").and_then(Value::as_str) != Some("text")
            || is_truthy(spec.get("knowledge"))
        {
            continue;
        }
        let label = match LABEL_SEP_RE.as_ref() {
            Some(re) => re.replace_all(name, " "),
            None => std::borrow::Cow::Borrowed(name.as_str()),
        };
        let instructions = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        if requested_identifier(&label) || requested_identifier(instructions) {
            selected.push(name.clone());
        }
    }
    selected
}

/// Focus a white-margin document up to `3.0x` scale, matching `image_identifier.detail_view`.
///
/// Returns `Some(focused)` if cropped & upscaled, or `None` if unchanged.
pub fn detail_view(image: &RgbImage, max_pixels: usize) -> Option<RgbImage> {
    focus::focus_document(image, max_pixels, 3.0)
}

/// Crop the top 40% of a non-uniform RGB document and upscale up to `3.0x` / `max_pixels`.
///
/// Exact port of `image_identifier.py::header_view`.
pub fn header_view(image: &RgbImage, max_pixels: usize) -> Option<RgbImage> {
    if image.width() == 0 || image.height() == 0 {
        return None;
    }
    let corner = image.get_pixel(0, 0).0;
    non_bg_bbox(image, corner)?;

    let crop_h = ((f64::from(image.height()) * 0.4).ceil() as u32)
        .max(1)
        .min(image.height());
    let header = crop_rgb(image, 0, 0, image.width(), crop_h);
    let w = header.width();
    let h = header.height();
    let scale = 3.0_f64.min(((max_pixels as f64) / (f64::from(w) * f64::from(h))).sqrt());
    let new_w = ((f64::from(w) * scale) as u32).max(1);
    let new_h = ((f64::from(h) * scale) as u32).max(1);
    Some(resize_bicubic(&header, new_w, new_h))
}

/// Build base identifier views: returns `(vec![focused], true)` if [`detail_view`] succeeded,
/// otherwise `(vec![image, header], false)` (or `(vec![image], false)` if `header_view` is `None`).
///
/// When `skip_focus` is `true` and `max_pixels == 1_600_000`, skips the redundant focus check
/// (matching upstream `unfocused_image` fast path).
pub fn views(image: &RgbImage, max_pixels: usize, skip_focus: bool) -> (Vec<RgbImage>, bool) {
    if !(skip_focus && max_pixels == super::DEFAULT_MAX_PIXELS)
        && let Some(focused) = detail_view(image, max_pixels)
    {
        return (vec![focused], true);
    }
    match header_view(image, max_pixels) {
        Some(header) => (vec![image.clone(), header], false),
        None => (vec![image.clone()], false),
    }
}

/// Compute Pillow `ImageStat.Stat.median` per RGB channel on an image.
fn rgb_channel_medians(image: &RgbImage) -> [u8; 3] {
    let mut hists = [[0usize; 256]; 3];
    let raw = image.as_raw();
    let pixel_count = raw.len() / 3;
    let (chunks, _) = raw.as_chunks::<3>();
    for chunk in chunks {
        hists[0][usize::from(chunk[0])] += 1;
        hists[1][usize::from(chunk[1])] += 1;
        hists[2][usize::from(chunk[2])] += 1;
    }
    let half = pixel_count / 2;
    let mut medians = [0u8; 3];
    for c in 0..3 {
        let mut acc = 0usize;
        for (val, &cnt) in hists[c].iter().enumerate() {
            acc += cnt;
            if acc > half {
                medians[c] = val as u8;
                break;
            }
        }
    }
    medians
}

/// Detect the content region against a light (per-channel median >= 160) background sampled
/// at `64x64`, threshold `max(|R-bg|, |G-bg|, |B-bg|) >= 32`, pad 24px, require area <= 50%,
/// and upscale bicubic up to `3.0x` / `max_pixels`.
///
/// Exact port of `image_identifier.py::content_view`.
pub fn content_view(image: &RgbImage, max_pixels: usize) -> Option<RgbImage> {
    let img_w = image.width();
    let img_h = image.height();
    if img_w == 0 || img_h == 0 {
        return None;
    }

    let sampled = resize_bicubic(image, CONTENT_BG_SAMPLE_SIZE, CONTENT_BG_SAMPLE_SIZE);
    let bg = rgb_channel_medians(&sampled);
    if bg[0].min(bg[1]).min(bg[2]) < CONTENT_MIN_BG_CHANNEL {
        return None;
    }

    let raw = image.as_raw();
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for y in 0..img_h {
        let row_off = (y as usize) * (img_w as usize) * 3;
        for x in 0..img_w {
            let off = row_off + (x as usize) * 3;
            let dr = raw[off].abs_diff(bg[0]);
            let dg = raw[off + 1].abs_diff(bg[1]);
            let db = raw[off + 2].abs_diff(bg[2]);
            if dr.max(dg).max(db) >= CONTENT_DIFF_THRESHOLD {
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
                found = true;
            }
        }
    }

    if !found {
        return None;
    }

    let left = min_x.saturating_sub(PADDING_PX);
    let top = min_y.saturating_sub(PADDING_PX);
    let right = (max_x + 1).saturating_add(PADDING_PX).min(img_w);
    let bottom = (max_y + 1).saturating_add(PADDING_PX).min(img_h);

    let w = right.saturating_sub(left);
    let h = bottom.saturating_sub(top);
    if w == 0 || h == 0 {
        return None;
    }

    let crop_area = f64::from(w) * f64::from(h);
    let full_area = f64::from(img_w) * f64::from(img_h);
    if crop_area > full_area / 2.0 {
        return None;
    }

    let scale = 3.0_f64.min(((max_pixels as f64) / crop_area).sqrt());
    if scale <= 1.0 {
        return None;
    }

    let cropped = crop_rgb(image, left, top, right, bottom);
    let new_w = ((f64::from(w) * scale) as u32).max(1);
    let new_h = ((f64::from(h) * scale) as u32).max(1);
    Some(resize_bicubic(&cropped, new_w, new_h))
}

/// Build the full identifier view set: `[focused]` if focused, else `[image, header?, content?]`.
///
/// Exact port of `image_identifier.py::content_views`.
pub fn content_views(image: &RgbImage, max_pixels: usize, skip_focus: bool) -> Vec<RgbImage> {
    let (mut selected, was_focused) = views(image, max_pixels, skip_focus);
    if was_focused {
        return selected;
    }
    if let Some(content) = content_view(image, max_pixels) {
        selected.push(content);
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crops::DEFAULT_MAX_PIXELS;
    use image::Rgb;
    use serde_json::json;

    #[test]
    fn identifier_field_selection_respects_exclusions() {
        let schema: Map<String, Value> = serde_json::from_value(json!({
            "invoice_number": { "type": "text", "instructions": "Read value." },
            "po_ref": { "type": "text", "instructions": "Extract the purchase order number." },
            "other": { "type": "text", "instructions": "Extract memo, not invoice number." },
            "count": { "type": "number", "instructions": "Invoice number." }
        }))
        .unwrap();
        let selected = fields(&schema);
        assert_eq!(
            selected,
            vec!["invoice_number".to_string(), "po_ref".to_string()]
        );
    }

    #[test]
    fn content_view_detects_off_white_background_document() {
        // Slightly off-white paper (240, 238, 230) so focus_document (diff vs 255) won't crop,
        // while content_view (median bg >= 160, diff >= 32) WILL isolate and upscale the block.
        let mut img = RgbImage::from_pixel(1000, 1000, Rgb([240, 238, 230]));
        for y in 200..320 {
            for x in 300..500 {
                img.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
        assert!(detail_view(&img, DEFAULT_MAX_PIXELS).is_none());
        let content = content_view(&img, DEFAULT_MAX_PIXELS).unwrap();
        // (200 + 48) * 3 = 744, (120 + 48) * 3 = 504
        assert_eq!(content.dimensions(), (248 * 3, 168 * 3));

        let all = content_views(&img, DEFAULT_MAX_PIXELS, false);
        assert_eq!(all.len(), 3); // original, header_view, content_view
    }
}
