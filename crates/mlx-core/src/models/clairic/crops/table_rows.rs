//! Physical table-row detection, header+row collage generation, and row assembly.
//!
//! Exact port of upstream `clairic_runtime/image_table_rows.py`.

use image::RgbImage;
use serde_json::{Map, Value, json};

use crate::models::clairic::orientation::rgb_to_luma_bt601;

use super::{
    DEFAULT_MAX_PIXELS, boolean_runs, crop_rgb, median_f64, paste_rgb, resize_bicubic, white_canvas,
};

const MIN_IMAGE_WIDTH: u32 = 100;
const MIN_IMAGE_HEIGHT: u32 = 60;
const MAX_NONWHITE_DENSITY: f64 = 0.22;
const MIN_BAND_HEIGHT: usize = 10;
const ROW_PAD_PX: u32 = 3;
const HEADER_ROW_GAP_PX: u32 = 12;

/// Detected single physical table with a header band and 3+ body row bands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowTable {
    pub header: (u32, u32),
    pub rows: Vec<(u32, u32)>,
}

impl RowTable {
    /// Build one header+row collage per body row (each band padded ±3px, stacked vertically
    /// with a 12px white gap), matching `image_table_rows.py::RowTable.views`.
    pub fn views(&self, image: &RgbImage) -> Vec<RgbImage> {
        let w = image.width();
        let h = image.height();
        let mut result = Vec::with_capacity(self.rows.len());
        let header = crop_rgb(
            image,
            0,
            self.header.0.saturating_sub(ROW_PAD_PX),
            w,
            self.header.1.saturating_add(ROW_PAD_PX).min(h),
        );
        for &(top, bottom) in &self.rows {
            let row = crop_rgb(
                image,
                0,
                top.saturating_sub(ROW_PAD_PX),
                w,
                bottom.saturating_add(ROW_PAD_PX).min(h),
            );
            let mut view = white_canvas(w, header.height() + row.height() + HEADER_ROW_GAP_PX);
            paste_rgb(&mut view, &header, 0, 0);
            paste_rgb(&mut view, &row, 0, header.height() + HEADER_ROW_GAP_PX);
            result.push(view);
        }
        result
    }

    /// Build upscaled header+row views matching `image_table_rows.py::prepare`
    /// (`scale = 2` for Flash, `scale = 4` for 27B, capped at `max_pixels`).
    pub fn scaled_views(&self, image: &RgbImage, scale: f64, max_pixels: usize) -> Vec<RgbImage> {
        self.views(image)
            .into_iter()
            .map(|view| scale_row_view(&view, scale, max_pixels))
            .collect()
    }
}

/// Upscale a single row view by `min(scale, sqrt(max_pixels / (w * h)))` using bicubic
/// if the factor exceeds `1.0`.
pub fn scale_row_view(view: &RgbImage, scale: f64, max_pixels: usize) -> RgbImage {
    let area = f64::from(view.width()) * f64::from(view.height());
    if area <= 0.0 {
        return view.clone();
    }
    let factor = scale.min(((max_pixels as f64) / area).sqrt());
    if factor > 1.0 {
        let new_w = ((f64::from(view.width()) * factor) as u32).max(1);
        let new_h = ((f64::from(view.height()) * factor) as u32).max(1);
        resize_bicubic(view, new_w, new_h)
    } else {
        view.clone()
    }
}

/// Shared vertical separator detection within a single horizontal band `[top..bottom)`.
///
/// Returns `(left, right)` column intervals for every separator satisfying:
/// - `sum(ink[:, col]) >= 0.9 * height`
/// - width `<= max(3, height * 0.18)`
/// - `left >= 3` and `right <= image_width - 3`
/// - 3px ink-free margins on both left (`[left-3..left)`) and right (`[right..right+3)`)
pub(crate) fn detect_band_separators(
    ink: &[bool],
    width: usize,
    top: usize,
    bottom: usize,
) -> (Vec<(usize, usize)>, Vec<bool>) {
    let height = bottom - top;
    let mut columns = vec![false; width];
    let mut col_sums = vec![0usize; width];

    for y in top..bottom {
        let row_off = y * width;
        for x in 0..width {
            if ink[row_off + x] {
                columns[x] = true;
                col_sums[x] += 1;
            }
        }
    }

    let threshold = 0.9 * (height as f64);
    let sep_mask: Vec<bool> = col_sums.iter().map(|&c| (c as f64) >= threshold).collect();
    let max_sep_w = 3.0_f64.max((height as f64) * 0.18);

    let mut separators = Vec::new();
    for (left, right) in boolean_runs(&sep_mask) {
        if ((right - left) as f64) > max_sep_w
            || left < 3
            || right > width.saturating_sub(3)
            || columns[left - 3..left].iter().any(|&v| v)
            || columns[right..right + 3].iter().any(|&v| v)
        {
            continue;
        }
        separators.push((left, right));
    }

    (separators, columns)
}

/// Detect a bordered/separator-delimited table in `image`.
///
/// Exact port of `clairic_runtime/image_table_rows.py::detect`.
pub fn detect(image: &RgbImage) -> Option<RowTable> {
    if image.width() < MIN_IMAGE_WIDTH || image.height() < MIN_IMAGE_HEIGHT {
        return None;
    }
    let w = image.width() as usize;
    let h = image.height() as usize;
    let grey = rgb_to_luma_bt601(image);

    let nonwhite_count = grey.iter().filter(|&&p| p < 245).count();
    if (nonwhite_count as f64) / (grey.len() as f64) > MAX_NONWHITE_DENSITY {
        return None;
    }

    let ink: Vec<bool> = grey.iter().map(|&p| p < 200).collect();
    let mut row_has_ink = vec![false; h];
    for y in 0..h {
        row_has_ink[y] = ink[y * w..(y + 1) * w].iter().any(|&v| v);
    }

    let bands = boolean_runs(&row_has_ink);
    struct Candidate {
        band_index: usize,
        top: usize,
        bottom: usize,
        separators: Vec<f64>,
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    for (index, &(top, bottom)) in bands.iter().enumerate() {
        let height = bottom - top;
        if height < MIN_BAND_HEIGHT {
            continue;
        }
        let (seps, _) = detect_band_separators(&ink, w, top, bottom);
        if seps.len() >= 2 {
            let centers = seps
                .into_iter()
                .map(|(l, r)| (l + r) as f64 / 2.0)
                .collect();
            candidates.push(Candidate {
                band_index: index,
                top,
                bottom,
                separators: centers,
            });
        }
    }

    if candidates.len() < 4 {
        return None;
    }

    let mut groups: Vec<Vec<&Candidate>> = Vec::new();
    for candidate in &candidates {
        let starts_new = groups
            .last()
            .and_then(|g| g.last())
            .is_none_or(|prev| candidate.band_index != prev.band_index + 1);
        if starts_new {
            groups.push(Vec::new());
        }
        if let Some(last) = groups.last_mut() {
            last.push(candidate);
        }
    }

    if groups.len() != 1 {
        return None;
    }

    let group = &groups[0];
    if group.len() < 4 {
        return None;
    }
    let expected_seps = group[0].separators.len();
    if group
        .iter()
        .any(|row| row.separators.len() != expected_seps)
    {
        return None;
    }

    let header = group[0];
    let body = &group[1..];

    let mut heights_f: Vec<f64> = group.iter().map(|r| (r.bottom - r.top) as f64).collect();
    let min_h = heights_f.iter().copied().fold(f64::INFINITY, f64::min);
    let max_h = heights_f.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let med_h = median_f64(&mut heights_f);

    let mut spacings: Vec<f64> = group
        .windows(2)
        .map(|w| (w[1].top - w[0].top) as f64)
        .collect();
    let min_sp = spacings.iter().copied().fold(f64::INFINITY, f64::min);
    let max_sp = spacings.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut middle = Vec::with_capacity(expected_seps);
    for col_idx in 0..expected_seps {
        let mut col_vals: Vec<f64> = body.iter().map(|r| r.separators[col_idx]).collect();
        middle.push(median_f64(&mut col_vals));
    }

    let mut max_dev = 0.0f64;
    for row in body {
        for (pos, &med) in row.separators.iter().zip(middle.iter()) {
            max_dev = max_dev.max((pos - med).abs());
        }
    }

    if max_h > 1.2 * min_h
        || max_sp > 1.3 * min_sp
        || max_dev > med_h * 1.2
        || (header.separators[0] - middle[0]).abs() < med_h * 2.0
    {
        return None;
    }

    // Silence unused-mut warning on `spacings`
    let _ = &mut spacings;

    Some(RowTable {
        header: (header.top as u32, header.bottom as u32),
        rows: body
            .iter()
            .map(|r| (r.top as u32, r.bottom as u32))
            .collect(),
    })
}

/// Build the per-row nullable object schema passed to the AR decoder for a physical row view.
///
/// Exact port of `image_table_rows.py:152-159`.
pub fn physical_row_spec(list_spec: &Value) -> Option<Value> {
    let properties = list_spec
        .get("items")
        .and_then(|item| item.get("properties"))
        .cloned()?;
    let instructions = list_spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    Some(json!({
        "type": "object",
        "nullable": true,
        "properties": properties,
        "instructions": format!(
            "Check whether the single data row below the column header matches the requested scope. Return null for an excluded row. Otherwise return the complete printed row. {instructions} Missing cells are null. Do not count or select a different row."
        ),
    }))
}

/// Deterministic view cache key for one physical table row view.
pub fn view_cache_key(table: &RowTable, row_index: usize, scale: f64) -> Option<String> {
    let row = table.rows.get(row_index)?;
    Some(format!(
        "table-row-v1:({}, {}):({}, {}):{scale}",
        table.header.0, table.header.1, row.0, row.1
    ))
}

/// Assemble per-row prediction JSON objects into a list answer matching `image_table_rows.py::assemble`.
pub fn assemble_rows(rows: &[&Value]) -> Value {
    let selected: Vec<&Value> = rows
        .iter()
        .copied()
        .filter(|row| !row.get("value").is_none_or(Value::is_null))
        .collect();
    let complete = rows.iter().all(|row| {
        row.get("complete")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    });
    let valid = complete
        && rows
            .iter()
            .all(|row| row.get("valid").and_then(Value::as_bool).unwrap_or(false));

    let p_null: f64 = rows
        .iter()
        .map(|row| row.get("p_null").and_then(Value::as_f64).unwrap_or(0.0))
        .product();

    let mut answer = Map::new();
    answer.insert(
        "value".to_string(),
        Value::Array(
            selected
                .iter()
                .map(|r| r.get("value").cloned().unwrap_or(Value::Null))
                .collect(),
        ),
    );
    answer.insert("complete".to_string(), Value::Bool(complete));
    answer.insert("valid".to_string(), Value::Bool(valid));
    answer.insert("confidence".to_string(), json!(0.0));
    answer.insert(
        "confidences".to_string(),
        Value::Array(
            selected
                .iter()
                .map(|r| json!(r.get("confidence").and_then(Value::as_f64).unwrap_or(0.0)))
                .collect(),
        ),
    );
    answer.insert(
        "confidence_method".to_string(),
        Value::String("ar_physical_image_row_sequence_probability".to_string()),
    );
    answer.insert("p_null".to_string(), json!(p_null));
    answer.insert(
        "p_nulls".to_string(),
        Value::Array(
            selected
                .iter()
                .map(|r| json!(r.get("p_null").and_then(Value::as_f64).unwrap_or(0.0)))
                .collect(),
        ),
    );
    answer.insert(
        "p_null_method".to_string(),
        Value::String("product_physical_image_row_exclusion_probabilities".to_string()),
    );
    answer.insert(
        "spans".to_string(),
        Value::Array(vec![Value::Null; selected.len()]),
    );

    if valid
        && rows
            .iter()
            .all(|r| r.get("log_probability").and_then(Value::as_f64).is_some())
    {
        let probability: f64 = rows
            .iter()
            .filter_map(|r| r.get("log_probability").and_then(Value::as_f64))
            .sum();
        if probability.is_finite() {
            answer.insert("log_probability".to_string(), json!(probability));
            answer.insert("confidence".to_string(), json!(probability.min(0.0).exp()));
        }
    }

    Value::Object(answer)
}

/// Default scaled views helper using `DEFAULT_MAX_PIXELS`.
pub fn default_scaled_views(table: &RowTable, image: &RgbImage, scale: f64) -> Vec<RgbImage> {
    table.scaled_views(image, scale, DEFAULT_MAX_PIXELS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn draw_table_band(
        img: &mut RgbImage,
        top: u32,
        height: u32,
        sep_xs: &[u32],
        text_xs: &[(u32, u32)],
    ) {
        let dark = Rgb([15, 15, 15]);
        // Vertical separators spanning full band height
        for &sx in sep_xs {
            for y in top..(top + height) {
                img.put_pixel(sx, y, dark);
                img.put_pixel(sx + 1, y, dark);
            }
        }
        // Short text strokes safely away from separator 3px margins
        for &(x0, x1) in text_xs {
            for y in (top + 4)..(top + height - 4) {
                for x in x0..x1 {
                    img.put_pixel(x, y, dark);
                }
            }
        }
    }

    #[test]
    fn detects_separated_table_and_generates_header_row_views() {
        let mut img = RgbImage::from_pixel(400, 300, Rgb([255, 255, 255]));
        // Header band at y=20..36 (h=16): first separator at x=120 (>= 2*16 = 32px away from body sep 60)
        draw_table_band(
            &mut img,
            20,
            16,
            &[120, 260],
            &[(20, 90), (140, 220), (280, 340)],
        );
        // 3 body bands at y=50, 80, 110 (spacing=30, h=16), separators at x=60 and x=260
        for &y in &[50u32, 80, 110] {
            draw_table_band(
                &mut img,
                y,
                16,
                &[60, 260],
                &[(20, 50), (80, 200), (280, 340)],
            );
        }

        let table = detect(&img).expect("should detect table");
        assert_eq!(table.header, (20, 36));
        assert_eq!(table.rows, vec![(50, 66), (80, 96), (110, 126)]);

        let raw_views = table.views(&img);
        assert_eq!(raw_views.len(), 3);
        // Header padded ±3 = 22px, row padded ±3 = 22px, gap = 12px => height 56px
        assert_eq!(raw_views[0].dimensions(), (400, 56));

        let scaled_flash = table.scaled_views(&img, 2.0, DEFAULT_MAX_PIXELS);
        assert_eq!(scaled_flash[0].dimensions(), (800, 112));
        let scaled_27b = table.scaled_views(&img, 4.0, DEFAULT_MAX_PIXELS);
        assert_eq!(scaled_27b[0].dimensions(), (1600, 224));
    }

    #[test]
    fn assemble_rows_computes_confidence_and_p_null() {
        let r1 = json!({
            "value": {"item": "A"},
            "complete": true,
            "valid": true,
            "confidence": 0.9,
            "p_null": 0.1,
            "log_probability": -0.1
        });
        let r2 = json!({
            "value": null,
            "complete": true,
            "valid": true,
            "confidence": 0.8,
            "p_null": 0.8,
            "log_probability": -0.2
        });
        let assembled = assemble_rows(&[&r1, &r2]);
        assert_eq!(assembled["valid"], Value::Bool(true));
        assert_eq!(assembled["value"].as_array().unwrap().len(), 1);
        assert!((assembled["p_null"].as_f64().unwrap() - 0.08).abs() < 1e-9);
        assert!((assembled["log_probability"].as_f64().unwrap() - (-0.3)).abs() < 1e-9);
    }
}
