//! Ragged separator-delimited row detection, isolation, eligibility, and assembly.
//!
//! Exact port of upstream `clairic_runtime/image_ragged_rows.py`.

use std::{collections::HashSet, sync::LazyLock};

use image::RgbImage;
use regex::Regex;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::{media::hex_digest, orientation::rgb_to_luma_bt601};

use super::{
    boolean_runs, crop_rgb, is_truthy, join_wrapped_descriptions,
    table_rows::{self, detect_band_separators, scale_row_view},
};

/// Stable view cache identifier matching `image_ragged_rows.VIEW_ID`.
pub const VIEW_ID: &str = "ragged-table-v1";
/// Gate boolean field name matching `image_ragged_rows.TARGET_GATE`.
pub const TARGET_GATE: &str = "target_is_billed_item";
/// Target row object field name matching `image_ragged_rows.TARGET_ROW`.
pub const TARGET_ROW: &str = "target_row";
/// Shared target instruction matching `image_ragged_rows.TARGET_INSTRUCTION`.
pub const TARGET_INSTRUCTION: &str = "Image 1 is the complete source document. Image 2 isolates one physical row from image 1. Classify and read only the target row in image 2. Use image 1 to identify the table column labels. A column header is not a billed item. Do not select a neighboring row or count row positions. ";

const MIN_IMAGE_WIDTH: u32 = 100;
const MIN_IMAGE_HEIGHT: u32 = 60;
const MAX_NONWHITE_DENSITY: f64 = 0.22;
const MIN_BAND_HEIGHT: usize = 10;
const ROW_PAD_PX: u32 = 3;

static BILLED_SCOPE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:line[ _-]?items?|billed\s+(?:products?|services?|items?|goods))\b").ok()
});
static FINANCIAL_SCOPE_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:tax(?:es)?|fees?|surcharges?|payments?)\b").ok());
static EXCLUDE_SPLIT_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\bExclude\b").ok());
static NEGATED_EXCLUDE_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:not|never|\w+n['’]t)(?:\s+\w+){0,2}\s*$").ok());
static EXCLUDE_OVERRIDE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:include|including|keep|retain|except|unless|not|no|none)\b").ok()
});

/// Detected ragged separator-delimited row bands (`(top, bottom)` pairs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaggedRows {
    pub rows: Vec<(u32, u32)>,
}

impl RaggedRows {
    /// Crop each row band padded by ±3px vertically across the full image width.
    pub fn views(&self, image: &RgbImage) -> Vec<RgbImage> {
        let w = image.width();
        let h = image.height();
        self.rows
            .iter()
            .map(|&(top, bottom)| {
                crop_rgb(
                    image,
                    0,
                    top.saturating_sub(ROW_PAD_PX),
                    w,
                    bottom.saturating_add(ROW_PAD_PX).min(h),
                )
            })
            .collect()
    }

    /// Crop and bicubic-upscale each row band by `min(scale, sqrt(max_pixels / (w * h)))`.
    pub fn scaled_views(&self, image: &RgbImage, scale: f64, max_pixels: usize) -> Vec<RgbImage> {
        self.views(image)
            .into_iter()
            .map(|view| scale_row_view(&view, scale, max_pixels))
            .collect()
    }
}

/// Detect a ragged separator-delimited table (when `table_rows::detect` does not match,
/// all candidate bands are contiguous with identical separator counts, empty right of
/// the last separator + 3px, and consistent band heights and spacings).
///
/// Exact port of `clairic_runtime/image_ragged_rows.py::detect`.
pub fn detect(image: &RgbImage) -> Option<RaggedRows> {
    if table_rows::detect(image).is_some() {
        return None;
    }
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

    struct Candidate {
        band_index: usize,
        top: usize,
        bottom: usize,
        separators: Vec<(usize, usize)>,
        columns: Vec<bool>,
    }

    let mut candidates = Vec::new();
    for (index, &(top, bottom)) in boolean_runs(&row_has_ink).iter().enumerate() {
        let height = bottom - top;
        if height < MIN_BAND_HEIGHT {
            continue;
        }
        let (separators, columns) = detect_band_separators(&ink, w, top, bottom);
        if separators.len() >= 2 {
            candidates.push(Candidate {
                band_index: index,
                top,
                bottom,
                separators,
                columns,
            });
        }
    }

    if candidates.len() < 2 {
        return None;
    }
    if candidates
        .windows(2)
        .any(|pair| pair[1].band_index != pair[0].band_index + 1)
    {
        return None;
    }
    let sep_count = candidates[0].separators.len();
    if candidates.iter().any(|c| c.separators.len() != sep_count) {
        return None;
    }
    // Check right-of-last-separator + 3px is completely ink-free for every candidate row.
    if candidates.iter().any(|c| {
        let last_right = c.separators.last().map_or(w, |&(_, r)| r);
        let start = (last_right + 3).min(w);
        c.columns[start..].iter().any(|&v| v)
    }) {
        return None;
    }

    let min_h = candidates
        .iter()
        .map(|c| c.bottom - c.top)
        .min()
        .unwrap_or(0) as f64;
    let max_h = candidates
        .iter()
        .map(|c| c.bottom - c.top)
        .max()
        .unwrap_or(0) as f64;
    let min_sp = candidates
        .windows(2)
        .map(|p| p[1].top - p[0].top)
        .min()
        .unwrap_or(0) as f64;
    let max_sp = candidates
        .windows(2)
        .map(|p| p[1].top - p[0].top)
        .max()
        .unwrap_or(0) as f64;

    if max_h > 1.2 * min_h || max_sp > 1.3 * min_sp {
        return None;
    }

    Some(RaggedRows {
        rows: candidates
            .into_iter()
            .map(|c| (c.top as u32, c.bottom as u32))
            .collect(),
    })
}

/// Check whether a list field spec is eligible for ragged-row isolation.
///
/// Exact port of `image_ragged_rows.py::_eligible`.
pub fn is_eligible(spec: &Value) -> bool {
    let Some(item) = spec.get("items").and_then(Value::as_object) else {
        return false;
    };
    let Some(properties) = item.get("properties").and_then(Value::as_object) else {
        return false;
    };
    let instructions = spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");

    let (scope, exclusion) = match EXCLUDE_SPLIT_RE
        .as_ref()
        .and_then(|re| re.find(instructions))
    {
        Some(m) => (&instructions[..m.start()], Some(&instructions[m.end()..])),
        None => (instructions, None),
    };

    if let Some(after_exclude) = exclusion {
        if NEGATED_EXCLUDE_RE
            .as_ref()
            .is_some_and(|re| re.is_match(scope))
        {
            return false;
        }
        let mut split_iter = after_exclude.splitn(2, ['.', ';', '\n']);
        let excluded = split_iter.next().unwrap_or("");
        let later = split_iter.next();
        let later_has_financial =
            later.is_some_and(|l| FINANCIAL_SCOPE_RE.as_ref().is_some_and(|re| re.is_match(l)));
        let excluded_overridden = FINANCIAL_SCOPE_RE
            .as_ref()
            .is_some_and(|re| re.is_match(excluded))
            && EXCLUDE_OVERRIDE_RE
                .as_ref()
                .is_some_and(|re| re.is_match(excluded));
        if later_has_financial || excluded_overridden {
            return false;
        }
    }

    spec.get("type").and_then(Value::as_str) == Some("list")
        && item.get("type").and_then(Value::as_str) == Some("object")
        && !join_wrapped_descriptions(instructions)
        && !is_truthy(spec.get("knowledge"))
        && properties.len() >= 3
        && BILLED_SCOPE_RE
            .as_ref()
            .is_some_and(|re| re.is_match(scope))
        && !FINANCIAL_SCOPE_RE
            .as_ref()
            .is_some_and(|re| re.is_match(scope))
        && properties.values().all(|child| {
            matches!(
                child.get("type").and_then(Value::as_str),
                Some("text" | "number")
            ) && !is_truthy(child.get("knowledge"))
        })
}

/// Build the `{ target_is_billed_item, target_row }` child field specs for one isolated row.
///
/// Exact port of `image_ragged_rows.py::row_fields`.
pub fn row_fields(spec: &Value) -> Option<Map<String, Value>> {
    let properties = spec
        .get("items")
        .and_then(|i| i.get("properties"))
        .cloned()?;
    let instructions = spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let wrapped = join_wrapped_descriptions(instructions);
    let membership = if wrapped {
        "A logical item can span several printed bands. Include it exactly once, at its first description band. Return false for a later continuation band belonging to that preceding item, even when its numeric cells are printed on the continuation band. Distinct separate source rows remain separate even if they repeat. "
    } else {
        ""
    };
    let readout = if wrapped {
        "For the target item's first band, use the full document to read its complete wrapped description and cells from its continuation bands. Never borrow from a neighboring logical item. "
    } else {
        ""
    };

    let mut out = Map::new();
    out.insert(
        TARGET_GATE.to_string(),
        json!({
            "type": "boolean",
            "nullable": false,
            "instructions": format!(
                "{TARGET_INSTRUCTION}{membership}Is the target row a billed product or service row? Return false for the column header, totals, tax, fees, or payment rows. {instructions}"
            ),
        }),
    );
    out.insert(
        TARGET_ROW.to_string(),
        json!({
            "type": "object",
            "nullable": false,
            "properties": properties,
            "instructions": format!(
                "{TARGET_INSTRUCTION}{readout}Read the complete printed target row as an object. Whether it belongs in the requested list is decided separately. Always return an object with every requested cell. Missing cells and absent columns are null. Do not relabel an Amount column as Unit price. Never compute missing values."
            ),
        }),
    );
    Some(out)
}

/// Generate a collision-free gate field name (`"ragged_gate_" + sha256(child)[..16]`).
pub fn gate_field_name(child: &str, existing_fields: &HashSet<String>) -> String {
    let digest = hex_digest(&Sha256::digest(child.as_bytes()));
    let mut gate = format!("ragged_gate_{}", &digest[..16]);
    while existing_fields.contains(&gate) {
        gate.push('_');
    }
    gate
}

/// Deterministic view cache key for one ragged row view.
pub fn view_cache_key(band: (u32, u32), scale: f64) -> String {
    format!("{VIEW_ID}:({}, {}):scale{scale}", band.0, band.1)
}

fn valid_score(value: Option<&Value>, probability: bool) -> bool {
    let Some(v) = value else {
        return false;
    };
    if v.is_boolean() {
        return false;
    }
    let Some(num) = v.as_f64() else {
        return false;
    };
    num.is_finite()
        && if probability {
            (0.0..=1.0).contains(&num)
        } else {
            num <= 0.0
        }
}

/// Assemble `(gate_prediction, copy_prediction)` pairs into the list field answer,
/// matching `image_ragged_rows.py::assemble`.
pub fn assemble_rows(pairs: &[(&Value, &Value)], max_items: usize) -> Value {
    let selected: Vec<(&Value, &Value)> = pairs
        .iter()
        .copied()
        .filter(|(gate, _)| gate.get("value").and_then(Value::as_bool) == Some(true))
        .collect();

    let complete = pairs.iter().all(|(gate, _)| {
        gate.get("complete")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }) && selected.iter().all(|(_, copy)| {
        copy.get("complete")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    });

    let exclusion_scored = pairs
        .iter()
        .all(|(gate, _)| valid_score(gate.get("p_excluded"), true));

    let relevant_log_probs_ok = pairs
        .iter()
        .all(|(gate, _)| valid_score(gate.get("log_probability"), false))
        && selected
            .iter()
            .all(|(_, copy)| valid_score(copy.get("log_probability"), false));

    let mut scored = exclusion_scored && relevant_log_probs_ok;
    let mut probability = if scored {
        let sum: f64 = pairs
            .iter()
            .filter_map(|(g, _)| g.get("log_probability").and_then(Value::as_f64))
            .chain(
                selected
                    .iter()
                    .filter_map(|(_, c)| c.get("log_probability").and_then(Value::as_f64)),
            )
            .sum();
        if sum.is_finite() {
            Some(sum)
        } else {
            scored = false;
            None
        }
    } else {
        None
    };

    let within_limit = selected.len() <= max_items;
    let valid = complete
        && scored
        && within_limit
        && pairs.iter().all(|(gate, _)| {
            gate.get("valid").and_then(Value::as_bool).unwrap_or(false)
                && gate.get("value").is_some_and(Value::is_boolean)
        })
        && selected.iter().all(|(_, copy)| {
            copy.get("valid").and_then(Value::as_bool).unwrap_or(false)
                && copy.get("value").is_some_and(Value::is_object)
        });

    let p_null = if exclusion_scored {
        pairs
            .iter()
            .map(|(g, _)| g.get("p_excluded").and_then(Value::as_f64).unwrap_or(0.0))
            .product()
    } else {
        0.0
    };

    let mut answer = Map::new();
    answer.insert(
        "value".to_string(),
        Value::Array(
            selected
                .iter()
                .map(|(_, c)| c.get("value").cloned().unwrap_or(Value::Null))
                .collect(),
        ),
    );
    answer.insert("valid".to_string(), Value::Bool(valid));
    answer.insert("complete".to_string(), Value::Bool(complete));
    answer.insert(
        "confidence".to_string(),
        json!(if valid {
            probability.map_or(0.0, f64::exp)
        } else {
            0.0
        }),
    );
    answer.insert(
        "confidence_method".to_string(),
        Value::String("ar_physical_image_row_eligibility_and_copy_probability".to_string()),
    );
    answer.insert(
        "confidences".to_string(),
        Value::Array(
            selected
                .iter()
                .map(|(g, c)| {
                    if scored {
                        let lp = g
                            .get("log_probability")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0)
                            + c.get("log_probability")
                                .and_then(Value::as_f64)
                                .unwrap_or(0.0);
                        json!(lp.exp())
                    } else {
                        json!(0.0)
                    }
                })
                .collect(),
        ),
    );
    answer.insert("p_null".to_string(), json!(p_null));
    answer.insert(
        "p_null_method".to_string(),
        Value::String(
            if exclusion_scored {
                "product_physical_image_row_exclusion_probabilities"
            } else {
                "unavailable_physical_image_row_exclusion_probability"
            }
            .to_string(),
        ),
    );
    answer.insert(
        "p_nulls".to_string(),
        Value::Array(
            selected
                .iter()
                .map(|(g, _)| {
                    if exclusion_scored {
                        json!(g.get("p_excluded").and_then(Value::as_f64).unwrap_or(0.0))
                    } else {
                        json!(0.0)
                    }
                })
                .collect(),
        ),
    );
    answer.insert(
        "spans".to_string(),
        Value::Array(vec![Value::Null; selected.len()]),
    );

    if let Some(prob) = probability.take() {
        answer.insert("log_probability".to_string(), json!(prob));
    }
    if !valid {
        let reason = if !within_limit {
            "ragged_row_output_exceeds_max_items"
        } else if !scored {
            "ragged_row_unscored"
        } else {
            "ragged_row_read_incomplete"
        };
        answer.insert(
            "failure_reason".to_string(),
            Value::String(reason.to_string()),
        );
    }

    Value::Object(answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crops::DEFAULT_MAX_PIXELS;
    use image::Rgb;

    fn draw_ragged_band(img: &mut RgbImage, top: u32, height: u32, sep_xs: &[u32]) {
        let dark = Rgb([15, 15, 15]);
        // Text only to the LEFT of the last separator (nothing right of last_sep + 3)
        for y in (top + 4)..(top + height - 4) {
            for x in 15..35 {
                img.put_pixel(x, y, dark);
            }
            for x in (sep_xs[0] + 10)..(sep_xs[1] - 10) {
                img.put_pixel(x, y, dark);
            }
        }
        for &sx in sep_xs {
            for y in top..(top + height) {
                img.put_pixel(sx, y, dark);
                img.put_pixel(sx + 1, y, dark);
            }
        }
    }

    #[test]
    fn detects_ragged_rows_with_trailing_separator_and_crops_views() {
        let mut img = RgbImage::from_pixel(400, 260, Rgb([255, 255, 255]));
        // 3 rows (too few for table_rows which requires >=4, and trailing separator has no ink to its right)
        draw_ragged_band(&mut img, 20, 16, &[60, 220]);
        draw_ragged_band(&mut img, 50, 16, &[80, 250]);
        draw_ragged_band(&mut img, 80, 16, &[70, 240]);

        let ragged = detect(&img).expect("should detect ragged rows");
        assert_eq!(ragged.rows, vec![(20, 36), (50, 66), (80, 96)]);

        let views = ragged.views(&img);
        assert_eq!(views.len(), 3);
        assert_eq!(views[0].dimensions(), (400, 22));

        let scaled = ragged.scaled_views(&img, 2.0, DEFAULT_MAX_PIXELS);
        assert_eq!(scaled[0].dimensions(), (800, 44));
    }

    #[test]
    fn eligibility_checks_billed_scope_and_exclusions() {
        let eligible = json!({
            "type": "list",
            "instructions": "Extract billed line items. Exclude tax and fees.",
            "items": {
                "type": "object",
                "properties": {
                    "description": { "type": "text" },
                    "qty": { "type": "number" },
                    "amount": { "type": "number" }
                }
            }
        });
        assert!(is_eligible(&eligible));

        let wrapped = json!({
            "type": "list",
            "instructions": "Extract billed line items. Join wrapped descriptions.",
            "items": eligible["items"]
        });
        assert!(!is_eligible(&wrapped));

        let fields = row_fields(&eligible).unwrap();
        assert!(fields.contains_key(TARGET_GATE));
        assert!(fields.contains_key(TARGET_ROW));
    }
}
