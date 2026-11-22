//! Image crop, focus, and row-isolation modules for Clairic vision extraction.
//!
//! All bicubic resizes use `image::imageops::FilterType::CatmullRom` (Keys cubic `a = -0.5`),
//! matching `PIL.Image.Resampling.BICUBIC`.

pub mod financial_lines;
pub mod focus;
pub mod identifier;
pub mod party_names;
pub mod pdf_number_focus;
pub mod po_supplement;
pub mod ragged_rows;
pub mod table_rows;

use std::sync::LazyLock;

use image::{
    Rgb, RgbImage,
    imageops::{self, FilterType},
};
use regex::Regex;
use serde_json::Value;

/// Keys cubic (`a = -0.5`) resampling filter matching `PIL.Image.Resampling.BICUBIC`.
pub const BICUBIC: FilterType = FilterType::CatmullRom;
/// Default vision pixel budget (`1_600_000`) shared across crop modules.
pub const DEFAULT_MAX_PIXELS: usize = 1_600_000;
/// Default list-item cap matching upstream `schema.MAX_LIST_ITEMS`.
pub const MAX_LIST_ITEMS: usize = 32;

/// Return `true` if a JSON field value is Python-truthy (non-null, non-false, non-empty, non-zero).
pub(crate) fn is_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|v| v != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// Find the tight bounding box `(left, top, right, bottom)` (exclusive `right`/`bottom`)
/// of all pixels differing from `bg`, matching `ImageChops.difference(img, bg).getbbox()`.
pub(crate) fn non_bg_bbox(image: &RgbImage, bg: [u8; 3]) -> Option<(u32, u32, u32, u32)> {
    let w = image.width();
    let h = image.height();
    if w == 0 || h == 0 {
        return None;
    }
    let raw = image.as_raw();
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for y in 0..h {
        let row_start = (y as usize) * (w as usize) * 3;
        for x in 0..w {
            let off = row_start + (x as usize) * 3;
            if raw[off] != bg[0] || raw[off + 1] != bg[1] || raw[off + 2] != bg[2] {
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

    if found {
        Some((min_x, min_y, max_x + 1, max_y + 1))
    } else {
        None
    }
}

/// Crop a rectangular sub-image `[left..right, top..bottom)` from `image`.
pub(crate) fn crop_rgb(image: &RgbImage, left: u32, top: u32, right: u32, bottom: u32) -> RgbImage {
    let left = left.min(image.width());
    let top = top.min(image.height());
    let right = right.min(image.width()).max(left);
    let bottom = bottom.min(image.height()).max(top);
    let w = right - left;
    let h = bottom - top;
    imageops::crop_imm(image, left, top, w, h).to_image()
}

/// Allocate a solid white `(255, 255, 255)` RGB canvas.
pub(crate) fn white_canvas(width: u32, height: u32) -> RgbImage {
    RgbImage::from_pixel(width, height, Rgb([255, 255, 255]))
}

/// Copy `src` into `dst` at top-left `(dst_x, dst_y)`.
pub(crate) fn paste_rgb(dst: &mut RgbImage, src: &RgbImage, dst_x: u32, dst_y: u32) {
    let max_w = src.width().min(dst.width().saturating_sub(dst_x));
    let max_h = src.height().min(dst.height().saturating_sub(dst_y));
    for y in 0..max_h {
        for x in 0..max_w {
            let px = *src.get_pixel(x, y);
            dst.put_pixel(dst_x + x, dst_y + y, px);
        }
    }
}

/// Resize an `RgbImage` using `FilterType::CatmullRom` (matching `PIL.Image.Resampling.BICUBIC`).
pub(crate) fn resize_bicubic(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    imageops::resize(image, width.max(1), height.max(1), BICUBIC)
}

/// Extract half-open `[start, end)` intervals of contiguous `true` runs, matching
/// `image_table_rows._runs`.
pub(crate) fn boolean_runs(values: &[bool]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, &flag) in values.iter().enumerate() {
        match (start, flag) {
            (None, true) => start = Some(idx),
            (Some(s), false) => {
                runs.push((s, idx));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        runs.push((s, values.len()));
    }
    runs
}

/// Compute the median of a non-empty `f64` slice matching `numpy.median`.
pub(crate) fn median_f64(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

// ---- Self-contained ports of upstream `grounding.state_only`,
//      `table_rows.join_wrapped_descriptions`, and `printed_numbers.normalize`.

static STATE_ONLY_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:from (?:the )?(?:accompanying (?:state )?|state )text only|supplied in the accompanying state text)\b",
    )
    .ok()
});
static NULL_ABSENCE_GUARD_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:^|[.;\n])\s*return\s+null\s+(?:only\s+)?if\s+(?:not\s+(?:explicitly\s+)?printed|absent|missing)\s*(?:[.;]|$)",
    )
    .ok()
});
static STATE_ONLY_DISALLOWED_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:if|unless|compare|combine)\b").ok());
static EXCLUDED_SOURCE_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:not|ignore|exclude|excluding)\b[^.;:\n]*$").ok());

/// Exact port of `grounding.py::state_only`.
pub(crate) fn state_only_spec(spec: &Value) -> bool {
    let field_type = spec.get("type").and_then(Value::as_str);
    if !matches!(field_type, Some("text" | "number")) || is_truthy(spec.get("knowledge")) {
        return false;
    }
    let instructions = spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let scope = match NULL_ABSENCE_GUARD_RE.as_ref() {
        Some(re) => re.replace_all(instructions, " "),
        None => std::borrow::Cow::Borrowed(instructions),
    };
    if STATE_ONLY_DISALLOWED_RE
        .as_ref()
        .is_some_and(|re| re.is_match(&scope))
    {
        return false;
    }
    let Some(state_re) = STATE_ONLY_RE.as_ref() else {
        return false;
    };
    state_re.find_iter(instructions).any(|m| {
        !EXCLUDED_SOURCE_RE
            .as_ref()
            .is_some_and(|re| re.is_match(&instructions[..m.start()]))
    })
}

static WRAPPED_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:wrapped|multiline|multi-line|continuation)\b").ok());
static JOINING_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:join|combine|merge)\b").ok());
static NEGATIVE_WRAP_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:not|no|never|without|separate\w*|independent\w*)\b|\b\w+n['’]t\b").ok()
});
static POSITIVE_WRAP_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:join|combine)\b[^.;\n]*\b(?:wrapped|multiline|multi-line|continuation)\b[^.;\n]*\b(?:descriptions?|lines?|text)\b",
    )
    .ok()
});

/// Exact port of `table_rows.py::join_wrapped_descriptions`.
pub(crate) fn join_wrapped_descriptions(instructions: &str) -> bool {
    let mut positive = false;
    for clause in instructions.split(['.', ';', '\n']) {
        let wrapped = WRAPPED_RE.as_ref().is_some_and(|re| re.is_match(clause));
        let joining = JOINING_RE.as_ref().is_some_and(|re| re.is_match(clause));
        let negative = NEGATIVE_WRAP_RE
            .as_ref()
            .is_some_and(|re| re.is_match(clause));
        if negative && (wrapped || joining) {
            return false;
        }
        if POSITIVE_WRAP_RE
            .as_ref()
            .is_some_and(|re| re.is_match(clause))
        {
            positive = true;
        }
    }
    positive
}

static PRINTED_NUMBER_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(?P<open>\()?\s*(?P<before>[+-]?)\s*(?:[$€£¥]|USD|EUR|GBP|CAD|AUD|JPY)?\s*(?P<after>[+-]?)\s*(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?P<close>\))?\s*$",
    )
    .ok()
});

/// Parse a printed 2-decimal monetary/numeric cell matching
/// `printed_numbers.normalize(raw, {"type": "number", "decimals": 2})`.
pub(crate) fn parse_printed_number_2dp(raw: &str) -> std::result::Result<f64, &'static str> {
    let trimmed = raw.trim();
    let Some(caps) = PRINTED_NUMBER_RE
        .as_ref()
        .and_then(|re| re.captures(trimmed))
    else {
        return Err("Unsupported printed number");
    };
    let open = caps.name("open").is_some();
    let close = caps.name("close").is_some();
    let before = caps.name("before").map_or("", |m| m.as_str());
    let after = caps.name("after").map_or("", |m| m.as_str());
    if open != close
        || (!before.is_empty() && !after.is_empty())
        || (open && (!before.is_empty() || !after.is_empty()))
    {
        return Err("Unsupported printed number");
    }
    let num_str = caps
        .name("number")
        .map(|m| m.as_str().replace(',', ""))
        .ok_or("Unsupported printed number")?;
    let mut val: f64 = num_str.parse().map_err(|_| "Unsupported printed number")?;
    if !val.is_finite() {
        return Err("Nonfinite numeric answer");
    }
    if open || before == "-" || after == "-" {
        val = -val;
    }
    let rounded = (val * 100.0).round() / 100.0;
    if !rounded.is_finite() {
        return Err("Nonfinite numeric answer");
    }
    Ok(if rounded == 0.0 { 0.0 } else { rounded })
}
