//! Core Clairic schema definitions, Proleptic Gregorian date/time math, forecast
//! expansion, estimate range scaling/snapping, span lookup, and field validation.
//!
//! Direct port of upstream `clairic_runtime/schema.py`.

use super::errors::invalid_request;
use napi::Result;
use regex::Regex;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub const READ_TYPES: &[&str] = &["text", "number", "date", "time", "datetime", "duration"];
pub const FIELD_TYPES: &[&str] = &[
    "text", "number", "date", "time", "datetime", "duration", "list", "estimate",
];
pub const EVIDENCE_CONSTRUCTION: &str = "evidence-v1";
pub const ANSWER_ROUTES: &[&str] = &["copy", "normalize", "derive", "knowledge", "abstain"];
pub const REQUEST_TYPES: &[&str] = &[
    "text", "number", "date", "time", "datetime", "duration", "list", "estimate", "compute",
    "forecast",
];
pub const LIST_ITEM_TYPES: &[&str] = &["text", "number", "date"];
pub const DURATION_UNITS: &[&str] = &[
    "seconds", "minutes", "hours", "days", "weeks", "months", "years",
];
pub const MAX_LIST_ITEMS: usize = 32;
pub const MAX_DECIMALS: usize = 6;
pub const MAX_FORECAST_HORIZON: usize = 64;
pub const FORECAST_INTERVALS: &[&str] = &[
    "step", "minute", "hour", "day", "week", "month", "quarter", "year",
];
pub const DEFAULT_NULL_THRESHOLD: f64 = 0.5;

pub const SYSTEM_PREAMBLE: &str = "Fill each field from the input with a value of its type. Text copies a span of the input. Numbers, dates, times, datetimes, and durations are read from the input or worked out from it. A list holds every matching value in input order. A field is not stated when the input does not determine it. Estimates predict a value inside the declared range.";
pub const EVIDENCE_SYSTEM_PREAMBLE: &str = "Fill each field with a value of its type, following its instructions and the question's scope. For document-specific extraction, use the supplied input and return not stated when the requested information is missing. Preserve exact text, names and numbers unless normalization or a derived answer is requested. General factual questions may use reliable prior knowledge unless the instructions restrict answers to the supplied evidence; return not stated when the answer cannot be determined reliably. A list holds every matching item in the requested order, including repeated items. Estimates and forecasts predict distributions over the declared range and supported below-range or above-range outcomes. Follow the requested units, time intervals and horizons.";

/// Parsed Proleptic Gregorian calendar date (`YYYY-MM-DD`, year `1..=9999`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IsoDate {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

impl IsoDate {
    pub fn to_iso_string(self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    /// Proleptic Gregorian ordinal matching Python `datetime.date.toordinal()` (`0001-01-01 == 1`).
    pub fn to_ordinal(self) -> i64 {
        date_to_ordinal(self.year, self.month, self.day)
    }
}

/// Return `true` iff `year` is a leap year in the Proleptic Gregorian calendar.
pub const fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Number of days in `month` (`1..=12`) of `year` (returns `0` for invalid month).
pub const fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

const DAYS_BEFORE_MONTH: [i64; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
const DI400Y: i64 = 146_097;
const DI100Y: i64 = 36_524;
const DI4Y: i64 = 1_461;

fn days_before_year(year: i32) -> i64 {
    let y = i64::from(year) - 1;
    y * 365 + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400)
}

/// Convert `(year, month, day)` to Python's `datetime.date.toordinal()` (`0001-01-01 == 1`).
pub fn date_to_ordinal(year: i32, month: u32, day: u32) -> i64 {
    let leap = if month > 2 && is_leap_year(year) {
        1
    } else {
        0
    };
    days_before_year(year) + DAYS_BEFORE_MONTH[month as usize] + leap + i64::from(day)
}

/// Convert a Proleptic Gregorian ordinal (`1..=3_652_059`) to an [`IsoDate`], matching
/// CPython's `datetime.date.fromordinal`.
pub fn ordinal_to_date(ordinal: i64) -> Result<IsoDate> {
    if !(1..=3_652_059).contains(&ordinal) {
        return Err(invalid_request(format!(
            "ordinal {ordinal} is out of valid ISO date range"
        )));
    }
    let mut n = ordinal - 1;
    let n400 = n / DI400Y;
    n %= DI400Y;
    let mut year = n400 * 400 + 1;

    let n100 = n / DI100Y;
    n %= DI100Y;

    let n4 = n / DI4Y;
    n %= DI4Y;

    let n1 = n / 365;
    n %= 365;

    year += n100 * 100 + n4 * 4 + n1;
    if n1 == 4 || n100 == 4 {
        return Ok(IsoDate {
            year: (year - 1) as i32,
            month: 12,
            day: 31,
        });
    }

    let year_i32 = year as i32;
    let leap = is_leap_year(year_i32);
    let mut month = ((n + 50) >> 5) as usize;
    let mut preceding = DAYS_BEFORE_MONTH[month] + if month > 2 && leap { 1 } else { 0 };
    if preceding > n {
        month -= 1;
        preceding = DAYS_BEFORE_MONTH[month] + if month > 2 && leap { 1 } else { 0 };
    }
    let day = (n - preceding + 1) as u32;
    Ok(IsoDate {
        year: year_i32,
        month: month as u32,
        day,
    })
}

/// Parse an ISO-8601 calendar date `YYYY-MM-DD` (matching `datetime.date.fromisoformat`).
pub fn parse_iso_date(value: &str) -> Result<IsoDate> {
    let bytes = value.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return Err(invalid_request(format!("invalid ISO date {value:?}")));
    }
    let year: i32 = value[0..4]
        .parse()
        .map_err(|_| invalid_request(format!("invalid ISO date {value:?}")))?;
    let month: u32 = value[5..7]
        .parse()
        .map_err(|_| invalid_request(format!("invalid ISO date {value:?}")))?;
    let day: u32 = value[8..10]
        .parse()
        .map_err(|_| invalid_request(format!("invalid ISO date {value:?}")))?;
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return Err(invalid_request(format!("invalid ISO date {value:?}")));
    }
    let max_day = days_in_month(year, month);
    if day < 1 || day > max_day {
        return Err(invalid_request(format!("invalid ISO date {value:?}")));
    }
    Ok(IsoDate { year, month, day })
}

/// Format an ISO date from validated components.
pub fn format_iso_date(year: i32, month: u32, day: u32) -> String {
    format!("{year:04}-{month:02}-{day:02}")
}

/// Parse `HH:MM` or `HH:MM:SS` (no microseconds or timezone), matching `schema.py::parse_time`.
pub fn parse_time(value: &str) -> Result<(u32, u32, u32)> {
    let bytes = value.as_bytes();
    let (h_str, m_str, s_str) = match bytes.len() {
        5 if bytes[2] == b':' => (&value[0..2], &value[3..5], None),
        8 if bytes[2] == b':' && bytes[5] == b':' => {
            (&value[0..2], &value[3..5], Some(&value[6..8]))
        }
        _ => {
            return Err(invalid_request(format!(
                "time {value:?} must be HH:MM or HH:MM:SS"
            )));
        }
    };
    if !h_str.bytes().all(|b| b.is_ascii_digit())
        || !m_str.bytes().all(|b| b.is_ascii_digit())
        || s_str.is_some_and(|s| !s.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err(invalid_request(format!(
            "time {value:?} must be HH:MM or HH:MM:SS"
        )));
    }
    let hour: u32 = h_str
        .parse()
        .map_err(|_| invalid_request(format!("time {value:?} must be HH:MM or HH:MM:SS")))?;
    let minute: u32 = m_str
        .parse()
        .map_err(|_| invalid_request(format!("time {value:?} must be HH:MM or HH:MM:SS")))?;
    let second: u32 = match s_str {
        Some(s) => s
            .parse()
            .map_err(|_| invalid_request(format!("time {value:?} must be HH:MM or HH:MM:SS")))?,
        None => 0,
    };
    if hour > 23 || minute > 59 || second > 59 {
        return Err(invalid_request(format!(
            "time {value:?} must be HH:MM or HH:MM:SS"
        )));
    }
    Ok((hour, minute, second))
}

/// Format `(hour, minute, second)` as `HH:MM:SS`.
pub fn format_time(hour: u32, minute: u32, second: u32) -> String {
    format!("{hour:02}:{minute:02}:{second:02}")
}

/// Parse `YYYY-MM-DDTHH:MM[:SS]`, matching `schema.py::parse_datetime`.
pub fn parse_datetime(value: &str) -> Result<(String, (u32, u32, u32))> {
    let Some((day, clock)) = value.split_once('T') else {
        return Err(invalid_request(format!(
            "datetime must be YYYY-MM-DDTHH:MM[:SS], got {value:?}"
        )));
    };
    let parsed_day = parse_iso_date(day)?;
    let parsed_time = parse_time(clock)?;
    Ok((parsed_day.to_iso_string(), parsed_time))
}

/// Check if `name` matches `^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$`.
pub fn is_valid_field_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }
    let first = bytes[0];
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return false;
    }
    bytes[1..]
        .iter()
        .all(|&b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b == b'-')
}

/// Build the synthetic scalar list query field name for `(parent, index)` (1-based),
/// matching `schema.py::list_item_name`.
pub fn list_item_name(parent: &str, index: usize) -> String {
    let suffix = format!(".item{index}");
    if parent.len() + suffix.len() <= 64 {
        return format!("{parent}{suffix}");
    }
    let mut hasher = Sha256::new();
    hasher.update(parent.as_bytes());
    let digest = hasher.finalize();
    let hex8 = format!(
        "{:02x}{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2], digest[3]
    );
    let prefix_len = 64usize.saturating_sub(suffix.len() + 9);
    let prefix: String = parent.chars().take(prefix_len).collect();
    format!("{prefix}_{hex8}{suffix}")
}

/// Return contiguous 1-based scalar list query field names for `parent`, matching
/// `schema.py::list_query_fields`.
pub fn list_query_fields(fields: &Map<String, Value>, parent: &str) -> Result<Vec<String>> {
    let spec = fields
        .get(parent)
        .ok_or_else(|| invalid_request(format!("missing list field {parent:?}")))?;
    let item_type = match spec.get("items") {
        Some(Value::Object(obj)) => obj.get("type").and_then(Value::as_str),
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    };
    let mut queries: BTreeMap<usize, String> = BTreeMap::new();
    for (name, child) in fields {
        let Some((_, ordinal)) = name.rsplit_once(".item") else {
            continue;
        };
        if ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let Ok(index) = ordinal.parse::<usize>() else {
            continue;
        };
        let child_type = child.get("type").and_then(Value::as_str);
        if index > 0 && *name == list_item_name(parent, index) && child_type == item_type {
            queries.insert(index, name.clone());
        }
    }
    let expected_len = queries.len();
    for i in 1..=expected_len {
        if !queries.contains_key(&i) {
            return Err(invalid_request(format!(
                "noncontiguous scalar list queries for {parent:?}"
            )));
        }
    }
    Ok((1..=expected_len)
        .filter_map(|i| queries.remove(&i))
        .collect())
}

/// Assemble scalar list predictions from per-item query predictions under `evidence-v1`,
/// matching `schema.py::assemble_list_predictions`.
pub fn assemble_list_predictions(
    record: &Value,
    predictions: &Map<String, Value>,
) -> Result<Map<String, Value>> {
    let construction = record
        .get("_meta")
        .and_then(|m| m.get("construction"))
        .and_then(Value::as_str);
    if construction != Some(EVIDENCE_CONSTRUCTION) {
        return Ok(predictions.clone());
    }
    let fields = record
        .get("fields")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_request("record missing fields object"))?;
    let mut merged = predictions.clone();
    for (parent, spec) in fields {
        if spec.get("type").and_then(Value::as_str) != Some("list") {
            continue;
        }
        let queries = list_query_fields(fields, parent)?;
        if queries.is_empty() {
            continue;
        }
        let mut values: Vec<Value> = Vec::new();
        let mut confidences: Vec<Option<f64>> = Vec::new();
        let mut probabilities: Vec<Option<f64>> = Vec::new();
        for name in &queries {
            let item = predictions.get(name).and_then(Value::as_object);
            let Some(item_obj) = item else {
                return Err(invalid_request(format!(
                    "missing scalar list prediction for {name:?}"
                )));
            };
            let Some(val) = item_obj.get("value") else {
                return Err(invalid_request(format!(
                    "missing scalar list prediction for {name:?}"
                )));
            };
            if val.is_null() {
                probabilities.push(item_obj.get("p_null").and_then(Value::as_f64));
                break;
            }
            values.push(val.clone());
            let conf = item_obj.get("confidence").and_then(Value::as_f64);
            let p_null = item_obj.get("p_null").and_then(Value::as_f64);
            confidences.push(conf);
            probabilities.push(match (p_null, conf) {
                (Some(pn), Some(c)) => Some((1.0 - pn) * c),
                _ => None,
            });
        }
        let mut assembled = Map::new();
        assembled.insert("value".to_string(), Value::Array(values));
        if confidences.iter().all(Option::is_some) {
            assembled.insert(
                "confidences".to_string(),
                Value::Array(
                    confidences
                        .into_iter()
                        .flatten()
                        .map(|c| json!(c))
                        .collect(),
                ),
            );
        }
        if !probabilities.is_empty() && probabilities.iter().all(Option::is_some) {
            let prod: f64 = probabilities.into_iter().flatten().product();
            assembled.insert("confidence".to_string(), json!(prod));
        }
        merged.insert(parent.clone(), Value::Object(assembled));
    }
    Ok(merged)
}

/// Format a floating-point number with `sig_figs` significant digits matching Python's `f"{v:.{N}g}"`.
pub fn format_g(value: f64, sig_figs: usize) -> String {
    let sig_figs = sig_figs.max(1);
    if !value.is_finite() {
        return value.to_string();
    }
    let sign = if value.is_sign_negative() && value != 0.0 {
        "-"
    } else {
        ""
    };
    let abs = value.abs();
    if abs == 0.0 {
        return if value.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }
    // Use scientific notation with sig_figs - 1 fractional digits to get exact rounded mantissa & exponent.
    let sci = format!("{abs:.prec$e}", prec = sig_figs - 1);
    let Some((mantissa, exp_str)) = sci.split_once('e') else {
        return format!("{sign}{sci}");
    };
    let exp: i32 = exp_str.parse().unwrap_or(0);
    if exp < -4 || exp >= sig_figs as i32 {
        let trimmed_m = if mantissa.contains('.') {
            mantissa.trim_end_matches('0').trim_end_matches('.')
        } else {
            mantissa
        };
        let exp_sign = if exp >= 0 { "+" } else { "-" };
        return format!("{sign}{trimmed_m}e{exp_sign}{:02}", exp.abs());
    }
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let fixed = if exp >= 0 {
        let int_digits = (exp as usize) + 1;
        if int_digits >= digits.len() {
            let mut out = digits;
            for _ in 0..(int_digits - out.len()) {
                out.push('0');
            }
            out
        } else {
            let frac = digits[int_digits..].trim_end_matches('0');
            if frac.is_empty() {
                digits[..int_digits].to_string()
            } else {
                format!("{}.{}", &digits[..int_digits], frac)
            }
        }
    } else {
        let zeros = (-exp - 1) as usize;
        let frac = digits.trim_end_matches('0');
        format!("0.{}{frac}", "0".repeat(zeros))
    };
    format!("{sign}{fixed}")
}

/// Format an estimate bound matching `schema.py::format_bound`.
pub fn format_bound(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1_000_000_000_000_000.0 {
        format!("{}", value as i64)
    } else {
        format_g(value, 6)
    }
}

/// Format the numeric options suffix (`, integer` or `, N decimals`), matching `schema.py::number_options`.
pub fn number_options(spec: &Value) -> String {
    if spec
        .get("integer")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return ", integer".to_string();
    }
    if let Some(d) = spec.get("decimals").and_then(Value::as_i64) {
        return format!(", {d} decimals");
    }
    String::new()
}

/// Return the duration unit (defaults to `"seconds"`), matching `schema.py::duration_unit`.
pub fn duration_unit(spec: &Value) -> String {
    spec.get("unit")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("seconds")
        .to_string()
}

/// Return the allowed decimal places (`Some(0)` if `integer`, else `decimals`), matching
/// `schema.py::allowed_decimals`.
pub fn allowed_decimals(spec: &Value) -> Option<usize> {
    if spec
        .get("integer")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some(0);
    }
    spec.get("decimals")
        .and_then(Value::as_u64)
        .map(|d| d as usize)
}

/// Format the estimate range header fragment matching `schema.py::field_line`.
pub fn format_estimate_range(spec: &Value) -> String {
    let unit = spec
        .get("unit")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if is_date_estimate(spec) {
        let min_s = spec.get("min").and_then(Value::as_str).unwrap_or("");
        let max_s = spec.get("max").and_then(Value::as_str).unwrap_or("");
        let mut head = format!("estimate date, {min_s} to {max_s}");
        let granularity = date_granularity(spec);
        if granularity != "day" {
            head.push_str(&format!(", by {granularity}"));
        } else if let Some(step) = spec.get("step").and_then(Value::as_f64)
            && step != 0.0
            && step != 1.0
        {
            head.push_str(&format!(", step {} days", format_bound(step)));
        }
        head
    } else {
        let min_v = spec.get("min").and_then(Value::as_f64).unwrap_or(0.0);
        let max_v = spec.get("max").and_then(Value::as_f64).unwrap_or(0.0);
        let scale = spec
            .get("scale")
            .and_then(Value::as_str)
            .unwrap_or("linear");
        let unit_part = if unit.is_empty() {
            String::new()
        } else {
            format!(" {unit}")
        };
        let mut head = format!(
            "estimate{unit_part}, {} to {}, {scale} scale",
            format_bound(min_v),
            format_bound(max_v)
        );
        if let Some(step) = spec.get("step").and_then(Value::as_f64)
            && step != 0.0
        {
            head.push_str(&format!(", step {}", format_bound(step)));
        }
        head
    }
}

/// Render a field specification line for the prompt, matching `schema.py::field_line`.
pub fn field_line(name: &str, spec: &Value, preserve_instructions: bool) -> String {
    let kind = spec.get("type").and_then(Value::as_str).unwrap_or("");
    let unit = spec
        .get("unit")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    let unit_paren = if unit.is_empty() {
        String::new()
    } else {
        format!(" ({unit})")
    };
    let head = match kind {
        "estimate" => format_estimate_range(spec),
        "list" => {
            let items = spec
                .get("items")
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default();
            format!("list of {items}{}{unit_paren}", number_options(spec))
        }
        "duration" => format!(
            "duration in {}{}",
            duration_unit(spec),
            number_options(spec)
        ),
        _ => {
            let opts = if kind == "number" {
                number_options(spec)
            } else {
                String::new()
            };
            format!("{kind}{opts}{unit_paren}")
        }
    };
    let raw_inst = spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let instructions = if preserve_instructions {
        raw_inst.to_string()
    } else {
        raw_inst.split_whitespace().collect::<Vec<_>>().join(" ")
    };
    if instructions.is_empty() {
        format!("{name}: {head}.")
    } else {
        format!("{name}: {head}. {instructions}")
    }
}

/// Alias for [`field_line`].
pub fn format_field_line(name: &str, spec: &Value, preserve_instructions: bool) -> String {
    field_line(name, spec, preserve_instructions)
}

/// Build the per-step instruction string for an expanded forecast field, matching
/// `schema.py::forecast_instructions`.
pub fn forecast_instructions(instructions: &str, step: usize, interval: &str) -> String {
    let cleaned = instructions
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let trimmed = cleaned.trim_end_matches('.');
    let what = if trimmed.is_empty() {
        "the series"
    } else {
        trimmed
    };
    let unit = if interval == "step" { "step" } else { interval };
    let plural = if step > 1 { "s" } else { "" };
    format!("Forecast {what} {step} {unit}{plural} after the last observation.")
}

/// Expand every `type: "forecast"` field into `horizon` `estimate` fields (`{name}.1..={name}.{horizon}`),
/// matching `schema.py::expand_forecasts`.
pub fn expand_forecasts(
    fields: &Map<String, Value>,
) -> Result<(Map<String, Value>, Map<String, Value>)> {
    let mut expanded: Map<String, Value> = Map::new();
    let mut groups: Map<String, Value> = Map::new();
    for (name, spec) in fields {
        let Some(spec_obj) = spec.as_object() else {
            expanded.insert(name.clone(), spec.clone());
            continue;
        };
        if spec_obj.get("type").and_then(Value::as_str) != Some("forecast") {
            expanded.insert(name.clone(), spec.clone());
            continue;
        }
        let horizon_val = spec_obj.get("horizon");
        let horizon = match horizon_val {
            Some(v) if !v.is_boolean() => v.as_i64(),
            _ => None,
        };
        let Some(horizon_i64) = horizon else {
            return Err(invalid_request(format!(
                "forecast {name} needs an integer horizon from 1 to {MAX_FORECAST_HORIZON}"
            )));
        };
        if !(1..=(MAX_FORECAST_HORIZON as i64)).contains(&horizon_i64) {
            return Err(invalid_request(format!(
                "forecast {name} needs an integer horizon from 1 to {MAX_FORECAST_HORIZON}"
            )));
        }
        let horizon_usize = horizon_i64 as usize;
        let interval = match spec_obj.get("interval") {
            None => "step",
            Some(Value::String(s)) if FORECAST_INTERVALS.contains(&s.as_str()) => s.as_str(),
            Some(other) => {
                return Err(invalid_request(format!(
                    "forecast {name} has an unknown interval {other}"
                )));
            }
        };
        let inst_str = spec_obj
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        let mut step_names: Vec<Value> = Vec::with_capacity(horizon_usize);
        for step in 1..=horizon_usize {
            let step_name = format!("{name}.{step}");
            if fields.contains_key(&step_name) || expanded.contains_key(&step_name) {
                return Err(invalid_request(format!(
                    "forecast {name} step {step_name} collides with another field"
                )));
            }
            let mut child = Map::new();
            child.insert("type".to_string(), Value::String("estimate".to_string()));
            for (k, v) in spec_obj {
                if !matches!(k.as_str(), "type" | "horizon" | "interval" | "instructions") {
                    child.insert(k.clone(), v.clone());
                }
            }
            child.insert(
                "instructions".to_string(),
                Value::String(forecast_instructions(inst_str, step, interval)),
            );
            expanded.insert(step_name.clone(), Value::Object(child));
            step_names.push(Value::String(step_name));
        }
        groups.insert(name.clone(), Value::Array(step_names));
    }
    Ok((expanded, groups))
}

/// Typed variant of [`expand_forecasts`] returning `Vec<String>` groups.
pub fn expand_forecasts_grouped(
    fields: &Map<String, Value>,
) -> Result<(Map<String, Value>, Vec<(String, Vec<String>)>)> {
    let (expanded, groups_map) = expand_forecasts(fields)?;
    let mut groups = Vec::with_capacity(groups_map.len());
    for (name, arr) in groups_map {
        let names = arr
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        groups.push((name, names));
    }
    Ok((expanded, groups))
}

/// Normalize request fields by expanding forecast horizons and validating all field definitions.
pub fn normalize_request_fields(
    fields: &Map<String, Value>,
) -> Result<(Map<String, Value>, Vec<(String, Vec<String>)>)> {
    let (expanded, groups) = expand_forecasts_grouped(fields)?;
    validate_fields(&Value::Object(expanded.clone()))?;
    Ok((expanded, groups))
}

pub fn raw_unit(value: f64, low: f64, high: f64, scale: &str) -> f64 {
    if scale == "log" {
        if value <= 0.0 {
            return -1.0;
        }
        (value.ln() - low.ln()) / (high.ln() - low.ln())
    } else {
        (value - low) / (high - low)
    }
}

pub fn to_unit(value: f64, low: f64, high: f64, scale: &str) -> f64 {
    raw_unit(value, low, high, scale).clamp(0.0, 1.0)
}

pub fn from_unit(position: f64, low: f64, high: f64, scale: &str) -> f64 {
    let pos = position.clamp(0.0, 1.0);
    let value = if scale == "log" {
        (low.ln() + pos * (high.ln() - low.ln())).exp()
    } else {
        low + pos * (high - low)
    };
    value.clamp(low, high)
}

pub fn resolution_in_unit(spec: &Value, resolution: Option<f64>) -> Result<f64> {
    let Some(res) = resolution else {
        return Ok(0.0);
    };
    if res == 0.0 {
        return Ok(0.0);
    }
    let (low, high, scale) = estimate_range(spec)?;
    if scale == "log" {
        Ok(res.ln_1p() / (high.ln() - low.ln()))
    } else {
        Ok(res / (high - low))
    }
}

pub fn is_date_estimate(spec: &Value) -> bool {
    spec.get("type").and_then(Value::as_str) == Some("estimate")
        && spec.get("min").is_some_and(Value::is_string)
}

pub fn date_granularity(spec: &Value) -> &str {
    match spec.get("step").and_then(Value::as_str) {
        Some("month") => "month",
        Some("year") => "year",
        _ => "day",
    }
}

pub fn date_index(value: &str, granularity: &str) -> Result<f64> {
    let date = parse_iso_date(value)?;
    match granularity {
        "month" => Ok(f64::from(date.year * 12 + (date.month as i32) - 1)),
        "year" => Ok(f64::from(date.year)),
        _ => Ok(date.to_ordinal() as f64),
    }
}

pub fn index_date(index: f64, granularity: &str) -> Result<String> {
    let whole = index.round_ties_even() as i64;
    match granularity {
        "month" => {
            let year = whole.div_euclid(12) as i32;
            let month = (whole.rem_euclid(12) + 1) as u32;
            if !(1..=9999).contains(&year) {
                return Err(invalid_request(format!("year {year} out of ISO range")));
            }
            Ok(format_iso_date(year, month, 1))
        }
        "year" => {
            let year = whole as i32;
            if !(1..=9999).contains(&year) {
                return Err(invalid_request(format!("year {year} out of ISO range")));
            }
            Ok(format_iso_date(year, 1, 1))
        }
        _ => Ok(ordinal_to_date(whole)?.to_iso_string()),
    }
}

pub fn estimate_range(spec: &Value) -> Result<(f64, f64, String)> {
    if is_date_estimate(spec) {
        let granularity = date_granularity(spec);
        let min_s = spec
            .get("min")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_request("date estimate missing min"))?;
        let max_s = spec
            .get("max")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_request("date estimate missing max"))?;
        return Ok((
            date_index(min_s, granularity)?,
            date_index(max_s, granularity)?,
            "linear".to_string(),
        ));
    }
    let low = spec
        .get("min")
        .and_then(as_non_bool_f64)
        .ok_or_else(|| invalid_request("estimate missing numeric min"))?;
    let high = spec
        .get("max")
        .and_then(as_non_bool_f64)
        .ok_or_else(|| invalid_request("estimate missing numeric max"))?;
    let scale = spec
        .get("scale")
        .and_then(Value::as_str)
        .unwrap_or("linear")
        .to_string();
    Ok((low, high, scale))
}

pub fn estimate_step(spec: &Value) -> f64 {
    if is_date_estimate(spec) {
        if date_granularity(spec) != "day" {
            1.0
        } else {
            spec.get("step")
                .and_then(as_non_bool_f64)
                .filter(|&v| v != 0.0)
                .unwrap_or(1.0)
        }
    } else {
        spec.get("step").and_then(as_non_bool_f64).unwrap_or(0.0)
    }
}

pub fn estimate_number(spec: &Value, value: &Value) -> Result<f64> {
    if is_date_estimate(spec) {
        let s = value
            .as_str()
            .ok_or_else(|| invalid_request("date estimate value must be a string"))?;
        date_index(s, date_granularity(spec))
    } else {
        as_non_bool_f64(value)
            .ok_or_else(|| invalid_request("numeric estimate value must be a number"))
    }
}

pub fn estimate_value(spec: &Value, number: f64) -> Result<Value> {
    if is_date_estimate(spec) {
        let (low, high, _) = estimate_range(spec)?;
        let clamped = number.clamp(low, high);
        Ok(Value::String(index_date(clamped, date_granularity(spec))?))
    } else {
        Ok(json!(number))
    }
}

fn round_12g(v: f64) -> f64 {
    format_g(v, 12).parse::<f64>().unwrap_or(v)
}

pub fn grid_bounds(step: f64, low: f64, high: f64) -> (f64, f64) {
    let first = (low / step - 1e-9).ceil() * step;
    let last = (high / step + 1e-9).floor() * step;
    (round_12g(first), round_12g(last))
}

pub fn snap(value: f64, step: f64, low: f64, high: f64) -> f64 {
    let (first, last) = grid_bounds(step, low, high);
    let snapped = ((value / step).round_ties_even() * step).clamp(first, last);
    round_12g(snapped)
}

pub fn snap_quantile(value: f64, step: f64, low: f64, high: f64) -> f64 {
    let (first, last) = grid_bounds(step, low, high);
    let snapped = (((value - step / 2.0) / step - 1e-9).ceil() * step).clamp(first, last);
    round_12g(snapped)
}

pub fn step_interval(spec: &Value, value: f64) -> Result<(f64, f64)> {
    let step = estimate_step(spec);
    let (low, high, _) = estimate_range(spec)?;
    if step <= 0.0 {
        return Ok((value, value));
    }
    Ok((
        (value - step / 2.0).max(low),
        (value + step / 2.0).min(high),
    ))
}

fn cuts_word(chars: &[char], start: usize, end: usize) -> bool {
    let joined = |left: char, right: char| {
        left.is_ascii() && left.is_alphanumeric() && right.is_ascii() && right.is_alphanumeric()
    };
    (start > 0 && joined(chars[start - 1], chars[start]))
        || (end < chars.len() && joined(chars[end - 1], chars[end]))
}

fn prefer_whole(chars: &[char], spans: Vec<(usize, usize)>, limit: usize) -> Vec<(usize, usize)> {
    let whole: Vec<(usize, usize)> = spans
        .iter()
        .copied()
        .filter(|&(s, e)| !cuts_word(chars, s, e))
        .collect();
    let chosen = if whole.is_empty() { spans } else { whole };
    chosen.into_iter().take(limit).collect()
}

/// Convert a UTF-8 byte index into a Unicode char index using a precomputed byte-to-char map.
pub(crate) fn byte_to_char_map(text: &str) -> Vec<usize> {
    let mut map = vec![0usize; text.len() + 1];
    for (ci, (bi, ch)) in text.char_indices().enumerate() {
        let end = bi + ch.len_utf8();
        for slot in &mut map[bi..end] {
            *slot = ci;
        }
        map[end] = ci + 1;
    }
    map
}

/// Find occurrences of `value` in `text`, returning **Unicode character offset spans**
/// `(char_start, char_end)` matching `schema.py::find_spans`.
pub fn find_spans(text: &str, value: &str, limit: usize) -> Vec<(usize, usize)> {
    let value = value.trim();
    if value.is_empty() || limit == 0 {
        return Vec::new();
    }
    let scan = (limit * 8).max(64);
    let chars: Vec<char> = text.chars().collect();
    let val_chars: Vec<char> = value.chars().collect();
    let val_char_len = val_chars.len();
    let b2c = byte_to_char_map(text);

    // 1. Exact substring search
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut search_byte = 0;
    while spans.len() < scan && search_byte <= text.len() {
        let Some(rel) = text[search_byte..].find(value) else {
            break;
        };
        let byte_start = search_byte + rel;
        let char_start = b2c[byte_start];
        spans.push((char_start, char_start + val_char_len));
        // Advance by one Unicode char (matching Python `start + 1`).
        let Some(next_ch) = text[byte_start..].chars().next() else {
            break;
        };
        search_byte = byte_start + next_ch.len_utf8();
    }
    if !spans.is_empty() {
        return prefer_whole(&chars, spans, limit);
    }

    // 2. Case-insensitive fixed-length search (when `text.to_lowercase()` preserves char length)
    let lowered = text.to_lowercase();
    let target = value.to_lowercase();
    let lowered_chars = lowered.chars().count();
    if lowered_chars == chars.len() {
        let lowered_b2c = byte_to_char_map(&lowered);
        let target_char_len = target.chars().count();
        let mut search_b = 0;
        while spans.len() < scan && search_b <= lowered.len() {
            let Some(rel) = lowered[search_b..].find(&target) else {
                break;
            };
            let byte_start = search_b + rel;
            let char_start = lowered_b2c[byte_start];
            spans.push((char_start, char_start + target_char_len));
            let Some(next_ch) = lowered[byte_start..].chars().next() else {
                break;
            };
            search_b = byte_start + next_ch.len_utf8();
        }
        if !spans.is_empty() {
            return prefer_whole(&chars, spans, limit);
        }
    }

    // 3. Whitespace-collapsed case-insensitive regex search
    let pieces: Vec<&str> = value.split_whitespace().collect();
    if pieces.is_empty() {
        return Vec::new();
    }
    let pattern_str = format!(
        "(?i){}",
        pieces
            .iter()
            .map(|p| regex::escape(p))
            .collect::<Vec<_>>()
            .join(r"\s+")
    );
    let Ok(re) = Regex::new(&pattern_str) else {
        return Vec::new();
    };
    let spans: Vec<(usize, usize)> = re
        .find_iter(text)
        .take(scan)
        .map(|m| (b2c[m.start()], b2c[m.end()]))
        .collect();
    prefer_whole(&chars, spans, limit)
}

pub fn null_threshold(spec: &Value) -> f64 {
    spec.get("null_threshold")
        .and_then(as_non_bool_f64)
        .unwrap_or(DEFAULT_NULL_THRESHOLD)
}

/// Validate a fields dictionary matching `schema.py::validate_fields`.
pub fn validate_fields(fields: &Value) -> Result<()> {
    let Some(map) = fields.as_object() else {
        return Err(invalid_request("record must declare at least one field"));
    };
    if map.is_empty() {
        return Err(invalid_request("record must declare at least one field"));
    }
    for (name, spec) in map {
        if !is_valid_field_name(name) {
            return Err(invalid_request(format!("invalid field name {name:?}")));
        }
        let spec_obj = spec.as_object();
        let mut kind = spec_obj.and_then(|o| o.get("type")).and_then(Value::as_str);

        if let Some(obj) = spec_obj
            && let Some(knowledge) = obj.get("knowledge")
        {
            if !knowledge.is_boolean() {
                return Err(invalid_request(format!(
                    "field {name}: knowledge must be true or false"
                )));
            }
            let list_item = obj.get("items").and_then(Value::as_str);
            let allowed = matches!(kind, Some("text" | "number"))
                || (kind == Some("list") && matches!(list_item, Some("text" | "number")));
            if !allowed {
                return Err(invalid_request(format!(
                    "field {name}: knowledge applies to text and number fields only"
                )));
            }
        }

        if kind == Some("compute") {
            let obj = spec_obj
                .ok_or_else(|| invalid_request(format!("field {name} must be an object")))?;
            let nullable_ok = obj
                .get("nullable")
                .map(|v| v == &Value::Bool(true))
                .unwrap_or(true);
            let has_grammar = [
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "enum",
            ]
            .iter()
            .any(|k| obj.contains_key(*k));
            if !nullable_ok || has_grammar {
                return Err(invalid_request(format!(
                    "compute {name} does not support AR grammar constraints"
                )));
            }
            kind = Some("number");
        }

        let Some(kind_str) = kind else {
            return Err(invalid_request(format!(
                "field {name} has unknown type None"
            )));
        };
        if !FIELD_TYPES.contains(&kind_str) && !matches!(kind_str, "boolean" | "enum" | "object") {
            return Err(invalid_request(format!(
                "field {name} has unknown type {kind_str:?}"
            )));
        }
        let obj =
            spec_obj.ok_or_else(|| invalid_request(format!("field {name} must be an object")))?;

        if let Some(threshold_val) = obj.get("null_threshold") {
            if matches!(kind_str, "estimate" | "list") {
                return Err(invalid_request(format!(
                    "{kind_str} {name} does not take null_threshold"
                )));
            }
            let Some(threshold) = as_non_bool_f64(threshold_val) else {
                return Err(invalid_request(format!(
                    "field {name} needs null_threshold in (0, 1]"
                )));
            };
            if !(threshold > 0.0 && threshold <= 1.0) {
                return Err(invalid_request(format!(
                    "field {name} needs null_threshold in (0, 1]"
                )));
            }
        }

        let row_items = obj
            .get("items")
            .and_then(Value::as_object)
            .and_then(|o| o.get("type"))
            .and_then(Value::as_str)
            == Some("object");
        if kind_str == "list" && !row_items {
            let item_str = obj.get("items").and_then(Value::as_str);
            if !item_str.is_some_and(|s| LIST_ITEM_TYPES.contains(&s)) {
                return Err(invalid_request(format!(
                    "list {name} needs items in {LIST_ITEM_TYPES:?}"
                )));
            }
        }

        if kind_str == "duration" && !DURATION_UNITS.contains(&duration_unit(spec).as_str()) {
            return Err(invalid_request(format!(
                "duration {name} needs a unit in {DURATION_UNITS:?}"
            )));
        }

        let numeric = matches!(kind_str, "number" | "duration")
            || (kind_str == "list" && obj.get("items").and_then(Value::as_str) == Some("number"));
        if !numeric && (obj.contains_key("integer") || obj.contains_key("decimals")) {
            return Err(invalid_request(format!(
                "field {name}: integer and decimals apply to numbers and durations only"
            )));
        }
        if let Some(int_val) = obj.get("integer")
            && !int_val.is_boolean()
        {
            return Err(invalid_request(format!(
                "field {name}: integer must be true or false"
            )));
        }
        if let Some(dec_val) = obj.get("decimals") {
            let dec = if dec_val.is_boolean() {
                None
            } else {
                dec_val.as_i64()
            };
            let Some(dec_i64) = dec else {
                return Err(invalid_request(format!(
                    "field {name}: decimals must be an integer from 0 to {MAX_DECIMALS}"
                )));
            };
            if !(0..=(MAX_DECIMALS as i64)).contains(&dec_i64) {
                return Err(invalid_request(format!(
                    "field {name}: decimals must be an integer from 0 to {MAX_DECIMALS}"
                )));
            }
            if obj.get("integer").and_then(Value::as_bool).unwrap_or(false) && dec_i64 != 0 {
                return Err(invalid_request(format!(
                    "field {name}: an integer field has 0 decimals"
                )));
            }
        }

        if kind_str == "estimate" {
            if is_date_estimate(spec) {
                validate_date_range(name, spec)?;
            } else {
                validate_number_range(name, spec)?;
            }
        }
    }
    Ok(())
}

pub fn validate_number_range(name: &str, spec: &Value) -> Result<()> {
    let low = spec.get("min").and_then(as_non_bool_f64).ok_or_else(|| {
        invalid_request(format!("estimate {name} needs numeric min, max, and step"))
    })?;
    let high = spec.get("max").and_then(as_non_bool_f64).ok_or_else(|| {
        invalid_request(format!("estimate {name} needs numeric min, max, and step"))
    })?;
    let step = match spec.get("step") {
        None | Some(Value::Null) => None,
        Some(v) => Some(as_non_bool_f64(v).ok_or_else(|| {
            invalid_request(format!("estimate {name} needs numeric min, max, and step"))
        })?),
    };
    let scale = match spec.get("scale") {
        None => "linear",
        Some(Value::String(s)) => s.as_str(),
        _ => "",
    };
    if !(low.is_finite() && high.is_finite() && low < high) || !matches!(scale, "linear" | "log") {
        return Err(invalid_request(format!(
            "field {name} has an invalid range"
        )));
    }
    if scale == "log" && low <= 0.0 {
        return Err(invalid_request(format!(
            "log field {name} needs a positive minimum"
        )));
    }
    if let Some(s) = step
        && (!s.is_finite() || s <= 0.0 || (high - low) / s < 1.0 || (high - low) / s > 1_000_000.0)
    {
        return Err(invalid_request(format!(
            "estimate {name} has an invalid step {}",
            spec.get("step").unwrap_or(&Value::Null)
        )));
    }
    Ok(())
}

pub fn validate_date_range(name: &str, spec: &Value) -> Result<()> {
    let (low, high, _) = estimate_range(spec).map_err(|_| {
        invalid_request(format!(
            "date estimate {name} needs ISO bounds and a step in days, 'month', or 'year'"
        ))
    })?;
    let step = if date_granularity(spec) == "day" {
        match spec.get("step") {
            None | Some(Value::Null) => 1.0,
            Some(v) => as_non_bool_f64(v).ok_or_else(|| {
                invalid_request(format!(
                    "date estimate {name} needs ISO bounds and a step in days, 'month', or 'year'"
                ))
            })?,
        }
    } else {
        1.0
    };
    let scale_ok = match spec.get("scale") {
        None => true,
        Some(Value::String(s)) => s == "linear",
        _ => false,
    };
    if !spec.get("max").is_some_and(Value::is_string)
        || !scale_ok
        || !low.is_finite()
        || !high.is_finite()
        || low >= high
    {
        return Err(invalid_request(format!(
            "date estimate {name} has an invalid range"
        )));
    }
    if !step.is_finite() || step.fract() != 0.0 || step < 1.0 || (high - low) / step < 1.0 {
        return Err(invalid_request(format!(
            "date estimate {name} has an invalid step {}",
            spec.get("step").unwrap_or(&Value::Null)
        )));
    }
    Ok(())
}

pub fn validate_reference_date(value: Option<&Value>) -> Result<()> {
    let Some(val) = value else {
        return Ok(());
    };
    if val.is_null() {
        return Ok(());
    }
    let Some(s) = val.as_str() else {
        return Err(invalid_request(format!(
            "reference_date must be an ISO date, got {val}"
        )));
    };
    parse_iso_date(s)
        .map(|_| ())
        .map_err(|_| invalid_request(format!("reference_date must be an ISO date, got {s:?}")))
}

fn as_non_bool_f64(v: &Value) -> Option<f64> {
    if v.is_boolean() {
        return None;
    }
    v.as_f64()
}

/// Cached regex matching valid field names (`^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$`).
pub fn field_name_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Static literal pattern is guaranteed valid.
        Regex::new(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$").unwrap_or_else(|_| unreachable!())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn proleptic_gregorian_ordinal_roundtrip_and_leap_years() {
        assert_eq!(parse_iso_date("0001-01-01").unwrap().to_ordinal(), 1);
        assert_eq!(parse_iso_date("1970-01-01").unwrap().to_ordinal(), 719_163);
        assert_eq!(parse_iso_date("2024-02-29").unwrap().to_ordinal(), 738_945);
        assert!(parse_iso_date("2023-02-29").is_err());
        assert!(parse_iso_date("1900-02-29").is_err());
        assert!(parse_iso_date("2000-02-29").is_ok());

        for iso in [
            "0001-01-01",
            "1800-03-01",
            "1900-02-28",
            "1970-01-01",
            "2000-02-29",
            "2024-12-31",
            "2199-06-15",
            "9999-12-31",
        ] {
            let d = parse_iso_date(iso).unwrap();
            let roundtrip = ordinal_to_date(d.to_ordinal()).unwrap();
            assert_eq!(roundtrip, d);
            assert_eq!(roundtrip.to_iso_string(), iso);
        }
    }

    #[test]
    fn parse_time_and_datetime() {
        assert_eq!(parse_time("09:05").unwrap(), (9, 5, 0));
        assert_eq!(parse_time("23:59:59").unwrap(), (23, 59, 59));
        assert!(parse_time("24:00:00").is_err());
        assert!(parse_time("09:05:00.123").is_err());
        assert!(parse_time("09:05:00Z").is_err());

        let (d, t) = parse_datetime("2025-07-04T14:30:05").unwrap();
        assert_eq!(d, "2025-07-04");
        assert_eq!(t, (14, 30, 5));
        assert!(parse_datetime("2025-07-04 14:30:05").is_err());
    }

    #[test]
    fn forecast_expansion_and_collision_rejection() {
        let fields = json!({
            "sales": {
                "type": "forecast",
                "min": 0,
                "max": 1000,
                "horizon": 3,
                "interval": "month",
                "unit": "USD",
                "instructions": "Monthly revenue."
            }
        });
        let (expanded, groups) = expand_forecasts_grouped(fields.as_object().unwrap()).unwrap();
        assert_eq!(expanded.len(), 3);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "sales");
        assert_eq!(groups[0].1, vec!["sales.1", "sales.2", "sales.3"]);
        assert_eq!(
            expanded["sales.1"]["instructions"],
            "Forecast Monthly revenue 1 month after the last observation."
        );
        assert_eq!(
            expanded["sales.2"]["instructions"],
            "Forecast Monthly revenue 2 months after the last observation."
        );

        let bad_horizon = json!({
            "sales": {"type": "forecast", "min": 0, "max": 10, "horizon": 65}
        });
        assert!(expand_forecasts(bad_horizon.as_object().unwrap()).is_err());

        let collision = json!({
            "sales": {"type": "forecast", "min": 0, "max": 10, "horizon": 2},
            "sales.1": {"type": "number"}
        });
        assert!(expand_forecasts(collision.as_object().unwrap()).is_err());
    }

    #[test]
    fn field_line_formatting_and_list_item_hashing() {
        assert_eq!(
            field_line(
                "price",
                &json!({"type": "number", "decimals": 2, "unit": "USD", "instructions": "Total   cost"}),
                false
            ),
            "price: number, 2 decimals (USD). Total cost"
        );
        assert_eq!(
            field_line(
                "eta",
                &json!({"type": "estimate", "min": "2025-01-01", "max": "2026-01-01", "step": "month"}),
                false
            ),
            "eta: estimate date, 2025-01-01 to 2026-01-01, by month."
        );
        assert_eq!(list_item_name("items", 1), "items.item1");
        let long_parent = "a".repeat(62);
        let hashed = list_item_name(&long_parent, 12);
        assert!(hashed.len() <= 64);
        assert!(hashed.ends_with(".item12"));
    }

    #[test]
    fn find_spans_prefers_whole_words_and_unicode_char_offsets() {
        let text = "Café total: €100, cafe latte";
        assert_eq!(find_spans(text, "café", 16), vec![(0, 4)]);
        assert_eq!(find_spans(text, "€100", 16), vec![(12, 16)]);
        assert_eq!(
            find_spans("cat concatenate cat", "cat", 16),
            vec![(0, 3), (16, 19)]
        );
        assert_eq!(
            find_spans("hello   \n  world", "hello world", 16),
            vec![(0, 16)]
        );
    }

    #[test]
    fn estimate_scaling_snapping_and_date_index() {
        assert!((from_unit(0.5, 1.0, 100.0, "log") - 10.0).abs() < 1e-12);
        assert!((to_unit(10.0, 1.0, 100.0, "log") - 0.5).abs() < 1e-12);
        assert_eq!(snap(12.4, 5.0, 0.0, 100.0), 10.0);
        assert_eq!(snap(12.6, 5.0, 0.0, 100.0), 15.0);

        let idx = date_index("2025-03-01", "month").unwrap();
        assert_eq!(index_date(idx, "month").unwrap(), "2025-03-01");
        assert_eq!(index_date(2030.0, "year").unwrap(), "2030-01-01");
    }
}
