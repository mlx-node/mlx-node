//! Financial-line band detection, 2x bicubic line crops, classification, and row assembly.
//!
//! Exact port of upstream `clairic_runtime/image_financial_lines.py`.

use std::{collections::HashSet, sync::LazyLock};

use image::RgbImage;
use napi::Result;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::models::clairic::{errors::inference_error, orientation::rgb_to_luma_bt601};

use super::{
    MAX_LIST_ITEMS, boolean_runs, crop_rgb, is_truthy, parse_printed_number_2dp, resize_bicubic,
};

/// Verbatim line-copy instruction matching `image_financial_lines.LINE["instructions"]`.
pub const LINE_INSTRUCTION: &str = "Copy the complete single printed line verbatim, including all words, prefixes, punctuation and numbers. Do not extract a name or normalize any part. Return only the complete line.";

const MIN_IMAGE_WIDTH: u32 = 100;
const MAX_INK_DENSITY: f64 = 0.22;
const MIN_BANDS: usize = 2;
const MIN_BAND_HEIGHT: usize = 8;
const MAX_BAND_HEIGHT: usize = 32;
const BAND_PAD_PX: u32 = 3;

/// Role assigned to a printed financial line by [`role`] / [`scope`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinancialLineRole {
    Summary,
    Payment,
    Tax,
    Fee,
    Charge,
}

impl FinancialLineRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Payment => "payment",
            Self::Tax => "tax",
            Self::Fee => "fee",
            Self::Charge => "charge",
        }
    }
}

static TAX_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\btax(?:es)?(?:\s+(?:refund|credit|adjustment))?$").ok());
static FEE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:fees?|surcharges?)(?:\s+(?:refund|credit|adjustment))?$").ok()
});
static SUMMARY_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(?:sub[ -]?total|grand total|total|amount due|balance(?: due| forward)?|tax total|fee total)$",
    )
    .ok()
});
static PAYMENT_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(?:payments?\s+(?:received|instructions?|due|method|status)|(?:received|posted|applied)\s+payments?)\b|\bpayments?$",
    )
    .ok()
});
static TRANSFORM_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:deduplicat\w*|unique|aggregat\w*|summari[sz]\w*|combin\w*|group\w*|averag\w*|calculat\w*|normaliz\w*|convert\w*|except|unless|instead|knowledge)\b",
    )
    .ok()
});
static WORD_RE: LazyLock<Option<Regex>> = LazyLock::new(|| Regex::new(r"[a-z]+").ok());
static POSITIVE_SPLIT_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)[.\n]|\bExclude\b").ok());
static TAX_SCOPE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)^Only\s+(?:(?:explicit|printed|separately\s+printed)\s+)?tax(?:es)?\s+rows$")
        .ok()
});
static FEE_SCOPE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)^Only\s+(?:(?:explicit|printed|separately\s+printed)\s+)?fees?\s+rows$").ok()
});
static CHARGE_SCOPE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^Only\s+billed\s+(?:(?:goods?|products?|services?)\s+(?:(?:and|or)\s+)?)+(?:charge\s+)?rows$",
    )
    .ok()
});
static EXCLUDE_TAX_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\bExclude\b[^.\n]*\btax(?:es)?\b").ok());
static EXCLUDE_FEE_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\bExclude\b[^.\n]*\bfees?\b").ok());
static EXCLUDE_PAYMENT_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\bExclude\b[^.\n]*\bpayments?\b").ok());
static NEGATED_EXCLUDE_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:not|never)\s+exclude\b").ok());

static PIPE_OR_COLON_LINE_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"^(.*?)\s*(?:\||:)\s*(\(?[+-]?\$?[\d,]+\.\d{2}\)?)\s*\|?\s*$").ok()
});
static DOLLAR_SPACE_LINE_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"^(.*?)\s+(\(?[+-]?\$[\d,]+\.\d{2}\)?)\s*$").ok());

static SCOPE_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "only",
        "explicit",
        "printed",
        "separately",
        "tax",
        "taxes",
        "row",
        "rows",
        "fee",
        "fees",
        "billed",
        "goods",
        "product",
        "products",
        "services",
        "service",
        "charge",
        "charges",
        "exclude",
        "excluding",
        "all",
        "subtotal",
        "subtotals",
        "total",
        "totals",
        "grand",
        "payment",
        "payments",
        "amount",
        "due",
        "and",
        "or",
        "null",
        "if",
        "no",
        "when",
        "the",
        "document",
        "has",
        "a",
        "must",
        "be",
        "surcharge",
        "surcharges",
        "return",
        "instructions",
    ]
    .into_iter()
    .collect()
});

/// Canonical single-line read spec (`image_financial_lines.LINE`).
pub fn line_spec() -> Value {
    json!({
        "type": "text",
        "nullable": false,
        "instructions": LINE_INSTRUCTION,
    })
}

/// Build the parent scope-presence boolean-or-null enum spec matching
/// `image_financial_lines.py:281-288`.
pub fn presence_spec(list_spec: &Value) -> Value {
    let instructions = list_spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    json!({
        "type": "enum",
        "enum": [true, null],
        "nullable": true,
        "instructions": format!(
            "Decide whether at least one printed row satisfies this requested list scope. {instructions} Return true if any matching printed row exists; return null only if there are no matching rows. A matching row with amount zero or a negative amount is present. Check the row's own label; a neighboring tax or fee does not remove an eligible service row. This asks only whether a matching row exists, not its value or row count."
        ),
    })
}

/// Classify a printed line label into `summary`, `payment`, `tax`, `fee`, or `charge`.
///
/// Exact port of `image_financial_lines.py::role`.
pub fn role(label: &str) -> FinancialLineRole {
    let collapsed: String = label.split_whitespace().collect::<Vec<_>>().join(" ");
    let cleaned = collapsed.trim_end_matches([':', ' ']);
    if SUMMARY_RE.as_ref().is_some_and(|re| re.is_match(cleaned)) {
        return FinancialLineRole::Summary;
    }
    if PAYMENT_RE.as_ref().is_some_and(|re| re.is_match(cleaned)) {
        return FinancialLineRole::Payment;
    }
    if TAX_RE.as_ref().is_some_and(|re| re.is_match(cleaned)) {
        return FinancialLineRole::Tax;
    }
    if FEE_RE.as_ref().is_some_and(|re| re.is_match(cleaned)) {
        FinancialLineRole::Fee
    } else {
        FinancialLineRole::Charge
    }
}

/// Determine whether a list field spec qualifies for financial-line binding, returning
/// `Some(Tax | Fee | Charge)` or `None`.
///
/// Exact port of `image_financial_lines.py::scope`.
pub fn scope(spec: &Value) -> Option<FinancialLineRole> {
    if spec.get("type").and_then(Value::as_str) != Some("list") || is_truthy(spec.get("knowledge"))
    {
        return None;
    }
    let item = spec.get("items").and_then(Value::as_object)?;
    if item.get("type").and_then(Value::as_str) != Some("object") {
        return None;
    }
    let properties = item.get("properties").and_then(Value::as_object)?;
    if properties.len() != 2 {
        return None;
    }

    let mut types = Vec::with_capacity(2);
    let mut amount_spec: Option<&Map<String, Value>> = None;

    for child in properties.values() {
        let obj = child.as_object()?;
        if is_truthy(obj.get("knowledge")) {
            return None;
        }
        for key in obj.keys() {
            if !matches!(
                key.as_str(),
                "type" | "instructions" | "nullable" | "decimals"
            ) {
                return None;
            }
        }
        let child_inst = obj
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        if TRANSFORM_RE
            .as_ref()
            .is_some_and(|re| re.is_match(child_inst))
        {
            return None;
        }
        let t = obj.get("type").and_then(Value::as_str).unwrap_or("");
        if t == "number" {
            amount_spec = Some(obj);
        }
        types.push(t);
    }

    types.sort_unstable();
    if types != ["number", "text"] {
        return None;
    }

    let amount = amount_spec?;
    let decimals_is_2 = amount.get("decimals").and_then(Value::as_i64) == Some(2);
    if !decimals_is_2 || is_truthy(amount.get("unit")) || is_truthy(amount.get("integer")) {
        return None;
    }

    let instructions = match spec.get("instructions") {
        None => "",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => return None,
    };
    if TRANSFORM_RE
        .as_ref()
        .is_some_and(|re| re.is_match(instructions))
    {
        return None;
    }

    let lower = instructions.to_ascii_lowercase();
    if let Some(word_re) = WORD_RE.as_ref() {
        for m in word_re.find_iter(&lower) {
            if !SCOPE_WORDS.contains(m.as_str()) {
                return None;
            }
        }
    }

    let positive = match POSITIVE_SPLIT_RE
        .as_ref()
        .and_then(|re| re.find(instructions))
    {
        Some(m) => instructions[..m.start()].trim(),
        None => instructions.trim(),
    };

    if TAX_SCOPE_RE
        .as_ref()
        .is_some_and(|re| re.is_match(positive))
    {
        return Some(FinancialLineRole::Tax);
    }
    if FEE_SCOPE_RE
        .as_ref()
        .is_some_and(|re| re.is_match(positive))
    {
        return Some(FinancialLineRole::Fee);
    }
    if CHARGE_SCOPE_RE
        .as_ref()
        .is_some_and(|re| re.is_match(positive))
        && EXCLUDE_TAX_RE
            .as_ref()
            .is_some_and(|re| re.is_match(instructions))
        && EXCLUDE_FEE_RE
            .as_ref()
            .is_some_and(|re| re.is_match(instructions))
        && EXCLUDE_PAYMENT_RE
            .as_ref()
            .is_some_and(|re| re.is_match(instructions))
        && !NEGATED_EXCLUDE_RE
            .as_ref()
            .is_some_and(|re| re.is_match(instructions))
    {
        return Some(FinancialLineRole::Charge);
    }

    None
}

/// Parse a literal printed line into `(label, amount)` when it ends with a 2-decimal cell
/// separated by `|`, `:`, or whitespace before `$`.
///
/// Exact port of `image_financial_lines.py::parse`.
pub fn parse_line(line: &str) -> std::result::Result<Option<(String, f64)>, &'static str> {
    let caps = PIPE_OR_COLON_LINE_RE
        .as_ref()
        .and_then(|re| re.captures(line))
        .or_else(|| {
            DOLLAR_SPACE_LINE_RE
                .as_ref()
                .and_then(|re| re.captures(line))
        });

    let Some(caps) = caps else {
        return Ok(None);
    };
    let (Some(raw_label), Some(raw_num)) = (caps.get(1), caps.get(2)) else {
        return Ok(None);
    };
    let label = raw_label.as_str().trim_end_matches([' ', '|', ':']);
    if label.is_empty() || label.contains('|') {
        return Ok(None);
    }
    let normalized = parse_printed_number_2dp(raw_num.as_str())?;
    Ok(Some((label.to_string(), normalized)))
}

/// Detect horizontal text bands on a white-background image (`width >= 100`, `(0,0) == white`,
/// BT.601 ink `< 200` density `<= 0.22`, `2..=32` bands each of height `8..=32`).
///
/// Exact port of `image_financial_lines.py::bands`.
pub fn bands(image: &RgbImage) -> Option<Vec<(u32, u32)>> {
    if image.width() < MIN_IMAGE_WIDTH
        || image.height() == 0
        || image.get_pixel(0, 0).0 != [255, 255, 255]
    {
        return None;
    }
    let w = image.width() as usize;
    let h = image.height() as usize;
    let luma = rgb_to_luma_bt601(image);

    let mut ink_count = 0usize;
    let mut row_has_ink = vec![false; h];
    for y in 0..h {
        let mut any = false;
        for &px in &luma[y * w..(y + 1) * w] {
            if px < 200 {
                ink_count += 1;
                any = true;
            }
        }
        row_has_ink[y] = any;
    }

    let total = w.saturating_mul(h);
    if total == 0 || (ink_count as f64) / (total as f64) > MAX_INK_DENSITY {
        return None;
    }

    let rows = boolean_runs(&row_has_ink);
    if !(MIN_BANDS..=MAX_LIST_ITEMS).contains(&rows.len())
        || rows
            .iter()
            .any(|&(top, bottom)| !(MIN_BAND_HEIGHT..=MAX_BAND_HEIGHT).contains(&(bottom - top)))
    {
        return None;
    }

    Some(
        rows.into_iter()
            .map(|(top, bottom)| (top as u32, bottom as u32))
            .collect(),
    )
}

/// Crop a single band ±3px vertically and upscale `2x` bicubic (`CatmullRom`).
///
/// Exact port of `image_financial_lines.py:266-269`.
pub fn band_view(image: &RgbImage, top: u32, bottom: u32) -> RgbImage {
    let cropped = crop_rgb(
        image,
        0,
        top.saturating_sub(BAND_PAD_PX),
        image.width(),
        bottom.saturating_add(BAND_PAD_PX).min(image.height()),
    );
    resize_bicubic(&cropped, cropped.width() * 2, cropped.height() * 2)
}

/// Build `2x` bicubic crops for every detected band.
pub fn band_views(image: &RgbImage, detected_bands: &[(u32, u32)]) -> Vec<RgbImage> {
    detected_bands
        .iter()
        .map(|&(top, bottom)| band_view(image, top, bottom))
        .collect()
}

/// Deterministic view cache key for one financial-line band.
pub fn view_cache_key(top: u32, bottom: u32) -> String {
    format!("financial-line-v1:{top}:{bottom}")
}

/// Assemble copied line predictions and presence gate prediction into the list field answer,
/// matching `image_financial_lines.py::assemble`.
pub fn assemble_plan(
    list_spec: &Value,
    target_role: FinancialLineRole,
    copies: &[&Value],
    presence: &Value,
) -> Result<Value> {
    let properties = list_spec
        .get("items")
        .and_then(|i| i.get("properties"))
        .and_then(Value::as_object)
        .ok_or_else(|| inference_error("Financial-line plan missing item properties"))?;

    let text_key = properties
        .iter()
        .find(|(_, v)| v.get("type").and_then(Value::as_str) == Some("text"))
        .map(|(k, _)| k.as_str())
        .ok_or_else(|| inference_error("Financial-line plan missing text property"))?;
    let number_key = properties
        .iter()
        .find(|(_, v)| v.get("type").and_then(Value::as_str) == Some("number"))
        .map(|(k, _)| k.as_str())
        .ok_or_else(|| inference_error("Financial-line plan missing number property"))?;

    let mut parsed: Vec<Option<(String, f64)>> = Vec::with_capacity(copies.len());
    let mut parse_valid = true;
    for answer in copies {
        if let Some(s) = answer.get("value").and_then(Value::as_str) {
            match parse_line(s) {
                Ok(row) => parsed.push(row),
                Err(_) => {
                    parsed.push(None);
                    parse_valid = false;
                }
            }
        } else {
            parsed.push(None);
        }
    }

    let selected: Vec<(&Value, &(String, f64))> = copies
        .iter()
        .zip(parsed.iter())
        .filter_map(|(&ans, row_opt)| {
            let row = row_opt.as_ref()?;
            if role(&row.0) == target_role {
                Some((ans, row))
            } else {
                None
            }
        })
        .collect();

    let values: Vec<Value> = selected
        .iter()
        .map(|(_, (label, num))| {
            let mut obj = Map::new();
            obj.insert(text_key.to_string(), Value::String(label.clone()));
            obj.insert(number_key.to_string(), json!(*num));
            Value::Object(obj)
        })
        .collect();

    let complete = copies
        .iter()
        .all(|a| a.get("complete").and_then(Value::as_bool).unwrap_or(false))
        && presence
            .get("complete")
            .and_then(Value::as_bool)
            .unwrap_or(false);

    let max_items = list_spec
        .get("max_items")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(MAX_LIST_ITEMS);

    let valid = parse_valid
        && complete
        && copies
            .iter()
            .all(|a| a.get("valid").and_then(Value::as_bool).unwrap_or(false))
        && presence
            .get("valid")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        && values.len() <= max_items;

    let p_null = presence
        .get("p_null")
        .filter(|v| !v.is_boolean())
        .and_then(Value::as_f64)
        .filter(|&v| v.is_finite() && (0.0..=1.0).contains(&v))
        .ok_or_else(|| inference_error("Invalid financial-line model scores"))?;

    let mut copy_log_prob_sum = 0.0f64;
    for answer in copies {
        let lp = answer
            .get("log_probability")
            .filter(|v| !v.is_boolean())
            .and_then(Value::as_f64)
            .filter(|&v| v.is_finite() && v <= 0.0)
            .ok_or_else(|| inference_error("Invalid financial-line model scores"))?;
        copy_log_prob_sum += lp;
    }

    let event = if values.is_empty() {
        p_null
    } else {
        1.0 - p_null
    };
    let probability = if event > 0.0 {
        Some(copy_log_prob_sum + event.ln())
    } else {
        None
    };

    let mut answer = Map::new();
    answer.insert("value".to_string(), Value::Array(values));
    answer.insert("valid".to_string(), Value::Bool(valid));
    answer.insert("complete".to_string(), Value::Bool(complete));
    answer.insert("confidence".to_string(), json!(0.0));
    answer.insert(
        "confidence_method".to_string(),
        Value::String("ar_literal_image_lines_and_scope_presence_probability".to_string()),
    );
    answer.insert(
        "confidences".to_string(),
        Value::Array(
            selected
                .iter()
                .map(|(c, _)| json!(c.get("confidence").and_then(Value::as_f64).unwrap_or(0.0)))
                .collect(),
        ),
    );
    answer.insert("p_null".to_string(), json!(p_null));
    answer.insert(
        "p_null_method".to_string(),
        Value::String("ar_parent_scope_presence_null_probability".to_string()),
    );
    answer.insert(
        "p_nulls".to_string(),
        Value::Array(
            selected
                .iter()
                .map(|(c, _)| json!(c.get("p_null").and_then(Value::as_f64).unwrap_or(0.0)))
                .collect(),
        ),
    );
    answer.insert(
        "spans".to_string(),
        Value::Array(vec![Value::Null; selected.len()]),
    );

    if let Some(prob) = probability {
        answer.insert("log_probability".to_string(), json!(prob));
        answer.insert(
            "confidence".to_string(),
            json!(if valid { prob.exp() } else { 0.0 }),
        );
    }

    Ok(Value::Object(answer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn role_and_parse_line_match_upstream() {
        assert_eq!(role("Sales Tax:"), FinancialLineRole::Tax);
        assert_eq!(role("Delivery Fee"), FinancialLineRole::Fee);
        assert_eq!(role("Grand Total"), FinancialLineRole::Summary);
        assert_eq!(role("Payment Received"), FinancialLineRole::Payment);
        assert_eq!(role("Consulting Services"), FinancialLineRole::Charge);

        let (label, val) = parse_line("Sales Tax | $12.50").unwrap().unwrap();
        assert_eq!(label, "Sales Tax");
        assert!((val - 12.50).abs() < 1e-9);

        let (label2, val2) = parse_line("Late Fee ($5.25)").unwrap().unwrap();
        assert_eq!(label2, "Late Fee");
        assert!((val2 - (-5.25)).abs() < 1e-9);
    }

    #[test]
    fn scope_identifies_tax_fee_and_charge_lists() {
        let tax_spec = json!({
            "type": "list",
            "instructions": "Only explicit tax rows.",
            "items": {
                "type": "object",
                "properties": {
                    "label": { "type": "text" },
                    "amount": { "type": "number", "decimals": 2 }
                }
            }
        });
        assert_eq!(scope(&tax_spec), Some(FinancialLineRole::Tax));

        let charge_spec = json!({
            "type": "list",
            "instructions": "Only billed goods and services rows. Exclude tax, fees, payments and totals.",
            "items": tax_spec["items"]
        });
        assert_eq!(scope(&charge_spec), Some(FinancialLineRole::Charge));

        let invalid_spec = json!({
            "type": "list",
            "instructions": "Only tax rows. Summarize by state.",
            "items": tax_spec["items"]
        });
        assert_eq!(scope(&invalid_spec), None);
    }

    #[test]
    fn bands_and_2x_views_detect_financial_lines() {
        let mut img = RgbImage::from_pixel(400, 300, Rgb([255, 255, 255]));
        for &y0 in &[30u32, 60, 90, 120] {
            for y in y0..(y0 + 14) {
                for x in 40..320 {
                    img.put_pixel(x, y, Rgb([25, 25, 25]));
                }
            }
        }
        let detected = bands(&img).expect("should detect 4 financial line bands");
        assert_eq!(detected, vec![(30, 44), (60, 74), (90, 104), (120, 134)]);

        let views = band_views(&img, &detected);
        assert_eq!(views.len(), 4);
        // Band height 14 + 6 padding = 20 -> 2x = 40; width 400 -> 2x = 800
        assert_eq!(views[0].dimensions(), (800, 40));
    }

    #[test]
    fn assemble_plan_filters_by_role_and_scores() {
        let spec = json!({
            "type": "list",
            "instructions": "Only explicit tax rows.",
            "items": {
                "type": "object",
                "properties": {
                    "label": { "type": "text" },
                    "amount": { "type": "number", "decimals": 2 }
                }
            }
        });
        let c1 = json!({
            "value": "Widget | $10.00",
            "complete": true,
            "valid": true,
            "confidence": 0.9,
            "p_null": 0.05,
            "log_probability": -0.1
        });
        let c2 = json!({
            "value": "Sales Tax | $0.85",
            "complete": true,
            "valid": true,
            "confidence": 0.95,
            "p_null": 0.02,
            "log_probability": -0.05
        });
        let presence = json!({
            "value": true,
            "complete": true,
            "valid": true,
            "p_null": 0.01
        });
        let out = assemble_plan(&spec, FinancialLineRole::Tax, &[&c1, &c2], &presence).unwrap();
        assert_eq!(out["valid"], Value::Bool(true));
        let arr = out["value"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["label"], Value::String("Sales Tax".to_string()));
        assert!((arr[0]["amount"].as_f64().unwrap() - 0.85).abs() < 1e-9);
    }
}
