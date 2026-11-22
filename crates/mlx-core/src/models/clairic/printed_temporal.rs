//! Literal printed date/time/datetime detection, rewriting, and normalization.
//!
//! Direct port of upstream `clairic_runtime/printed_temporal.py`.

use super::{errors::invalid_request, schema};
use fancy_regex::Regex as FancyRegex;
use napi::Result;
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

pub const TYPES: &[&str] = &["date", "time", "datetime"];
pub const COPY_INSTRUCTION: &str = " Copy only a date or time explicitly printed for this exact requested role. Return null if this role has no printed date or time. Do not use invoice or due dates for service or billing period boundaries. Do not infer a period from its duration. Copy the requested date or time exactly as printed, including AM or PM when present. Preserve every digit. Do not normalize or convert the value. Return null only if the requested value is absent.";

const DATE_ROLES: &[&str] = &[
    "invoice_date",
    "due_date",
    "period_start",
    "period_end",
    "start_date",
    "end_date",
];

fn literal_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:print\w*|explicit\w*|verbatim|shown)\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn derived_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:calculat\w*|comput\w*|deriv\w*|infer\w*|after|before|ago|next|previous|following|prior|add|subtract|plus|minus|earlier|later|offset|duration|terms)\b|\bnet\s*\d+\b|\b(?:first|last)\s+day\s+of\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn document_date_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:invoice|billing|service|subscription|document|receipt)\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn clock_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\A\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AP]\.?M\.?)?\s*\z")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn ymd_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(r"\A(\d{4})([-/.])(\d{1,2})\2(\d{1,2})\z")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn numeric_date_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(r"\A(\d{1,2})([-/.])(\d{1,2})\2(\d{4})\z")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn mdy_named_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\A([A-Za-z]+)\s+(\d{1,2})(?:,\s+|\s+)(\d{4})\z")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn dmy_named_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\A(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\z").unwrap_or_else(|_| unreachable!())
    })
}

fn datetime_split_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\A(.+?)(?:T|\s+(?:at\s+)?)(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]\.?M\.?)?)\z")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn is_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

fn parse_english_month(month: &str) -> Option<u32> {
    match month.to_ascii_lowercase().as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

fn validated_iso(year: i32, month: u32, day: u32) -> Option<String> {
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return None;
    }
    let max_day = schema::days_in_month(year, month);
    if day < 1 || day > max_day {
        return None;
    }
    Some(schema::format_iso_date(year, month, day))
}

/// Mark explicitly printed temporal fields as `source_only: true`
/// (`printed_temporal.py::explicit_fields`).
pub fn explicit_fields(fields: &Map<String, Value>) -> Map<String, Value> {
    let mut result = fields.clone();
    for (name, spec_val) in fields {
        let Some(spec) = spec_val.as_object() else {
            continue;
        };
        let kind = spec.get("type").and_then(Value::as_str).unwrap_or("");
        let date_list = kind == "list" && spec.get("items").and_then(Value::as_str) == Some("date");
        if (!TYPES.contains(&kind) && !date_list)
            || spec.contains_key("source_only")
            || spec.contains_key("null_threshold")
            || is_truthy(spec.get("knowledge"))
        {
            continue;
        }
        let instructions = match spec.get("instructions") {
            None => "",
            Some(Value::String(s)) => s.as_str(),
            Some(_) => continue,
        };
        let role = DATE_ROLES.contains(&name.to_lowercase().as_str())
            && document_date_regex().is_match(instructions);
        if (literal_regex().is_match(instructions) || role)
            && !derived_regex().is_match(instructions)
        {
            let mut updated = spec.clone();
            updated.insert("source_only".to_string(), Value::Bool(true));
            result.insert(name.clone(), Value::Object(updated));
        }
    }
    result
}

/// Parse a 12-hour or 24-hour printed clock time into `HH:MM:SS` (`printed_temporal.py::clock`).
pub fn clock(value: &str) -> Result<String> {
    let Some(caps) = clock_regex().captures(value) else {
        return Err(invalid_request("Unsupported printed time"));
    };
    let mut hour: u32 = caps
        .get(1)
        .and_then(|m| m.as_str().parse().ok())
        .ok_or_else(|| invalid_request("Unsupported printed time"))?;
    let minute: u32 = caps
        .get(2)
        .and_then(|m| m.as_str().parse().ok())
        .ok_or_else(|| invalid_request("Unsupported printed time"))?;
    let second: u32 = match caps.get(3) {
        Some(m) => m
            .as_str()
            .parse()
            .map_err(|_| invalid_request("Unsupported printed time"))?,
        None => 0,
    };
    if let Some(meridiem) = caps.get(4) {
        if !(1..=12).contains(&hour) {
            return Err(invalid_request("Invalid 12-hour clock"));
        }
        let cleaned = meridiem.as_str().replace('.', "");
        let pm_offset = if cleaned.eq_ignore_ascii_case("PM") {
            12
        } else {
            0
        };
        hour = (hour % 12) + pm_offset;
    }
    if hour >= 24 || minute >= 60 || second >= 60 {
        return Err(invalid_request("Invalid printed time"));
    }
    Ok(format!("{hour:02}:{minute:02}:{second:02}"))
}

/// Parse a printed date in any of the 9 English/ISO formats or an unambiguous numeric
/// `DD/MM/YYYY` / `MM/DD/YYYY` format (`printed_temporal.py::day`).
pub fn day(value: &str) -> Result<String> {
    let trimmed = value.trim();

    // Formats 1-3: %Y-%m-%d, %Y/%m/%d, %Y.%m.%d
    if let Ok(Some(caps)) = ymd_regex().captures(trimmed)
        && let (Some(y), Some(m), Some(d)) = (
            caps.get(1).and_then(|v| v.as_str().parse::<i32>().ok()),
            caps.get(3).and_then(|v| v.as_str().parse::<u32>().ok()),
            caps.get(4).and_then(|v| v.as_str().parse::<u32>().ok()),
        )
        && let Some(iso) = validated_iso(y, m, d)
    {
        return Ok(iso);
    }

    // Formats 4-7: %B %d, %Y | %b %d, %Y | %B %d %Y | %b %d %Y
    if let Some(caps) = mdy_named_regex().captures(trimmed)
        && let (Some(month), Some(day_num), Some(year)) = (
            caps.get(1).and_then(|v| parse_english_month(v.as_str())),
            caps.get(2).and_then(|v| v.as_str().parse::<u32>().ok()),
            caps.get(3).and_then(|v| v.as_str().parse::<i32>().ok()),
        )
        && let Some(iso) = validated_iso(year, month, day_num)
    {
        return Ok(iso);
    }

    // Formats 8-9: %d %B %Y | %d %b %Y
    if let Some(caps) = dmy_named_regex().captures(trimmed)
        && let (Some(day_num), Some(month), Some(year)) = (
            caps.get(1).and_then(|v| v.as_str().parse::<u32>().ok()),
            caps.get(2).and_then(|v| parse_english_month(v.as_str())),
            caps.get(3).and_then(|v| v.as_str().parse::<i32>().ok()),
        )
        && let Some(iso) = validated_iso(year, month, day_num)
    {
        return Ok(iso);
    }

    // Unambiguous numeric DD[-/.]MM[-/.]YYYY vs MM[-/.]DD[-/.]YYYY
    if let Ok(Some(caps)) = numeric_date_regex().captures(trimmed)
        && let (Some(first), Some(second), Some(year)) = (
            caps.get(1).and_then(|v| v.as_str().parse::<u32>().ok()),
            caps.get(3).and_then(|v| v.as_str().parse::<u32>().ok()),
            caps.get(4).and_then(|v| v.as_str().parse::<i32>().ok()),
        )
    {
        let mut readings: BTreeSet<String> = BTreeSet::new();
        for (month, date) in [(first, second), (second, first)] {
            if let Some(iso) = validated_iso(year, month, date) {
                readings.insert(iso);
            }
        }
        if readings.len() == 1
            && let Some(only) = readings.into_iter().next()
        {
            return Ok(only);
        }
    }

    Err(invalid_request("Unsupported or ambiguous printed date"))
}

/// Normalize a printed temporal answer (`"date"`, `"time"`, or `"datetime"`)
/// (`printed_temporal.py::normalize`).
pub fn normalize(value: &Value, kind: &str) -> Result<Value> {
    if value.is_null() {
        return Ok(Value::Null);
    }
    let Some(s) = value.as_str() else {
        return Err(invalid_request(
            "A printed temporal answer must be a string",
        ));
    };
    match kind {
        "time" => clock(s).map(Value::String),
        "date" => day(s).map(Value::String),
        "datetime" => {
            let Some(caps) = datetime_split_regex().captures(s.trim()) else {
                return Err(invalid_request("Unsupported printed datetime"));
            };
            let date_part = caps
                .get(1)
                .map(|m| m.as_str())
                .ok_or_else(|| invalid_request("Unsupported printed datetime"))?;
            let time_part = caps
                .get(2)
                .map(|m| m.as_str())
                .ok_or_else(|| invalid_request("Unsupported printed datetime"))?;
            Ok(Value::String(format!(
                "{}T{}",
                day(date_part)?,
                clock(time_part)?
            )))
        }
        _ => Err(invalid_request("Unsupported temporal type")),
    }
}

/// Rewrite `source_only` temporal fields into literal text queries, returning
/// `(rewritten_fields, temporal_kinds)` (`printed_temporal.py::prepare`).
pub fn prepare(fields: &Map<String, Value>) -> (Map<String, Value>, BTreeMap<String, String>) {
    let mut out = fields.clone();
    let mut kinds: BTreeMap<String, String> = BTreeMap::new();

    for (name, spec_val) in fields {
        let Some(spec) = spec_val.as_object() else {
            continue;
        };
        if !is_truthy(spec.get("source_only")) {
            continue;
        }
        let kind = spec.get("type").and_then(Value::as_str).unwrap_or("");
        if kind == "list" && spec.get("items").and_then(Value::as_str) == Some("date") {
            let mut updated = spec.clone();
            let inst = updated
                .get("instructions")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            updated.insert("items".to_string(), Value::String("text".to_string()));
            updated.insert(
                "instructions".to_string(),
                Value::String(format!("{inst}{COPY_INSTRUCTION}")),
            );
            out.insert(name.clone(), Value::Object(updated));
            continue;
        }
        if !TYPES.contains(&kind) {
            continue;
        }
        kinds.insert(name.clone(), kind.to_string());
        let nullable = spec
            .get("nullable")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let inst = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        out.insert(
            name.clone(),
            json!({
                "type": "text",
                "nullable": nullable,
                "instructions": format!("{inst}{COPY_INSTRUCTION}")
            }),
        );
    }

    (out, kinds)
}

/// Normalize printed-temporal predictions (`printed_temporal.py::normalize_predictions`).
pub fn normalize_predictions(
    record: &Value,
    predictions: &Map<String, Value>,
) -> Map<String, Value> {
    let mut result = predictions.clone();
    let Some(temporal_map) = record
        .get("_meta")
        .and_then(|m| m.get("printed_temporal"))
        .and_then(Value::as_object)
    else {
        return result;
    };
    let has_media = is_truthy(record.get("_media_payload"));
    let input_text = record.get("input").and_then(Value::as_str).unwrap_or("");

    for (name, kind_val) in temporal_map {
        let Some(kind) = kind_val.as_str() else {
            continue;
        };
        let Some(orig_obj) = result.get(name).and_then(Value::as_object) else {
            continue;
        };
        let mut answer = orig_obj.clone();
        if !is_truthy(answer.get("valid")) {
            result.insert(name.clone(), Value::Object(answer));
            continue;
        }
        let printed = answer.get("value").cloned().unwrap_or(Value::Null);
        answer.insert("printed_value".to_string(), printed.clone());
        if !printed.is_null()
            && !has_media
            && !printed.as_str().is_some_and(|s| input_text.contains(s))
        {
            answer.insert("value".to_string(), Value::Null);
            answer.insert("valid".to_string(), Value::Bool(false));
            answer.insert("confidence".to_string(), json!(0.0));
            answer.insert("span".to_string(), Value::Null);
            answer.insert(
                "failure_reason".to_string(),
                Value::String("not_in_source".to_string()),
            );
            result.insert(name.clone(), Value::Object(answer));
            continue;
        }
        match normalize(&printed, kind) {
            Ok(normalized) => {
                answer.insert("value".to_string(), normalized);
                answer.insert(
                    "normalization_method".to_string(),
                    Value::String("printed_temporal_v1".to_string()),
                );
            }
            Err(_) => {
                answer.insert("value".to_string(), Value::Null);
                answer.insert("valid".to_string(), Value::Bool(false));
                answer.insert("confidence".to_string(), json!(0.0));
                answer.insert(
                    "failure_reason".to_string(),
                    Value::String("invalid_printed_temporal".to_string()),
                );
            }
        }
        result.insert(name.clone(), Value::Object(answer));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn explicit_fields_marks_printed_and_role_dates_but_skips_derived() {
        let fields = json!({
            "invoice_date": {"type": "date", "instructions": "Invoice issue date."},
            "due_date": {"type": "date", "instructions": "Invoice due date, net 30 days after issue."},
            "printed_time": {"type": "time", "instructions": "Verbatim timestamp shown."},
            "all_dates": {"type": "list", "items": "date", "instructions": "Explicitly printed dates."},
            "already": {"type": "date", "source_only": false, "instructions": "Printed date."}
        });
        let out = explicit_fields(fields.as_object().unwrap());
        assert_eq!(out["invoice_date"]["source_only"], true);
        assert!(out["due_date"].get("source_only").is_none());
        assert_eq!(out["printed_time"]["source_only"], true);
        assert_eq!(out["all_dates"]["source_only"], true);
        assert_eq!(out["already"]["source_only"], false);
    }

    #[test]
    fn clock_day_and_datetime_normalization() {
        assert_eq!(clock("9:05 AM").unwrap(), "09:05:00");
        assert_eq!(clock("12:00 a.m.").unwrap(), "00:00:00");
        assert_eq!(clock("12:30 P.M.").unwrap(), "12:30:00");
        assert_eq!(clock("23:59:58").unwrap(), "23:59:58");
        assert!(clock("13:00 PM").is_err());
        assert!(clock("24:00").is_err());

        // All 9 named / ISO formats
        assert_eq!(day("2024-03-05").unwrap(), "2024-03-05");
        assert_eq!(day("2024/3/5").unwrap(), "2024-03-05");
        assert_eq!(day("2024.03.05").unwrap(), "2024-03-05");
        assert_eq!(day("March 5, 2024").unwrap(), "2024-03-05");
        assert_eq!(day("Mar 5, 2024").unwrap(), "2024-03-05");
        assert_eq!(day("March 5 2024").unwrap(), "2024-03-05");
        assert_eq!(day("Mar 5 2024").unwrap(), "2024-03-05");
        assert_eq!(day("5 March 2024").unwrap(), "2024-03-05");
        assert_eq!(day("05 Mar 2024").unwrap(), "2024-03-05");

        // Unambiguous numeric DD/MM/YYYY and MM/DD/YYYY + equal day/month
        assert_eq!(day("25/03/2024").unwrap(), "2024-03-25");
        assert_eq!(day("03/25/2024").unwrap(), "2024-03-25");
        assert_eq!(day("07-07-2024").unwrap(), "2024-07-07");
        // Ambiguous numeric date rejected
        assert!(day("03/04/2024").is_err());
        // Mismatched separator rejected
        assert!(day("25/03-2024").is_err());

        // Datetime
        assert_eq!(
            normalize(&json!("March 5, 2024 at 2:15 PM"), "datetime").unwrap(),
            json!("2024-03-05T14:15:00")
        );
        assert_eq!(
            normalize(&json!("2024-03-05T08:00:00"), "datetime").unwrap(),
            json!("2024-03-05T08:00:00")
        );
        assert_eq!(normalize(&json!(null), "date").unwrap(), json!(null));
    }

    #[test]
    fn prepare_and_normalize_predictions_end_to_end() {
        let fields = json!({
            "issued": {"type": "date", "source_only": true, "instructions": "Printed issue date."},
            "dates": {"type": "list", "items": "date", "source_only": true, "instructions": "Dates."}
        });
        let (prepared, kinds) = prepare(fields.as_object().unwrap());
        assert_eq!(prepared["issued"]["type"], "text");
        assert_eq!(prepared["dates"]["items"], "text");
        assert_eq!(kinds.get("issued").map(String::as_str), Some("date"));
        assert!(!kinds.contains_key("dates"));

        let record = json!({
            "input": "Issued: 25/03/2024",
            "fields": prepared,
            "_meta": {"printed_temporal": kinds}
        });
        let preds = json!({
            "issued": {"valid": true, "value": "25/03/2024", "confidence": 0.95}
        });
        let out = normalize_predictions(&record, preds.as_object().unwrap());
        assert_eq!(out["issued"]["value"], "2024-03-25");
        assert_eq!(out["issued"]["printed_value"], "25/03/2024");
        assert_eq!(out["issued"]["normalization_method"], "printed_temporal_v1");
    }
}
