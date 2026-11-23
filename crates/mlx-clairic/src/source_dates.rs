//! Unambiguous source-date candidate extraction, EBNF constraints, and supplement helpers.
//!
//! Direct port of upstream `clairic_runtime/source_dates.py` (plus the date supplement
//! plan helper from `backend.py:1307-1350`).

use super::{errors::invalid_request, schema};
use crate::errors::Result;
use crate::grammar::GrammarSpec;
use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

pub const VERSION: &str = "source-date-candidates-v1";
pub const INSTRUCTION: &str = "Return the complete calendar date for this requested field as YYYY-MM-DD. Choose only a date explicitly printed for the requested role in the supplied source. Use the full day, month, and year, including legal ordinal date wording. Do not derive an unprinted date from a duration, term length, another date, or general knowledge. Return null if the requested role has no printed date. A date for another role does not fill this field.";

/// Regex fragment matching English month names & abbreviations (with optional trailing dot),
/// exported for downstream modules such as `receipt_totals`.
pub const MONTH_PATTERN: &str = "(?:september|february|november|december|january|october|august|april|march|sept|july|june|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\\.?";

fn parse_month(month: &str) -> Option<u32> {
    match month.trim_end_matches('.').to_ascii_lowercase().as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "sept" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

fn dmy_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        let pat = format!(
            r"(?i)(?<!\w)(?:the\s+)?(?<day>\d{{1,2}})(?<ordinal>st|nd|rd|th)?\s+(?:(?:day\s+)?of\s+)?(?<month>{MONTH_PATTERN}),?\s+(?<year>\d{{4}})(?!\w)"
        );
        FancyRegex::new(&pat).unwrap_or_else(|_| unreachable!())
    })
}

fn mdy_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        let pat = format!(
            r"(?i)(?<!\w)(?<month>{MONTH_PATTERN})\s+(?:the\s+)?(?<day>\d{{1,2}})(?<ordinal>st|nd|rd|th)?,?\s+(?<year>\d{{4}})(?!\w)"
        );
        FancyRegex::new(&pat).unwrap_or_else(|_| unreachable!())
    })
}

fn ymd_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(
            r"(?<![\w./-])(?<year>\d{4})(?<sep>[-/.])(?<month>\d{1,2})\k<sep>(?<day>\d{1,2})(?!(?:[\w/-]|\.\w))",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn numeric_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(
            r"(?<![\w./-])(?<first>\d{1,2})(?<sep>[-/.])(?<second>\d{1,2})\k<sep>(?<year>\d{4})(?!(?:[\w/-]|\.\w))",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn derived_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:calculat\w*|comput\w*|deriv\w*|infer\w*|after|before|ago|next|previous|following|prior|add|subtract|plus|minus|earlier|later|offset|duration|anniversar\w*|turns?)\b|\b(?:first|last)\s+day\s+of\b|\bnet\s*\d+\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn relative_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:today|tomorrow|yesterday|now)\b|\bcurrent\s+(?:calendar\s+)?date\b|\bbusiness\s+days?\b|\b(?:days?|weeks?|months?|years?)\s+from\b|\b(?:this|last)\s+(?:week|month|year)\b|\b(?:in|within)\s+(?:\d+|[a-z]+(?:[\s-]+[a-z]+){0,2})\s+(?:business\s+)?(?:days?|weeks?|months?|years?)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn reading_intent_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:dates?|dated|calendar\s+day|printed|explicit(?:ly)?|stated|shown|recorded|written|literal|verbatim)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn expression_request_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:value\s+(?:of|for)|evaluat\w*|expression|formula|variable|constant|function|code)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn calendar_expression_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?i)\b(?:add|sub(?:tract)?|plus|minus|offset|shift|set)[_]?(?:business[_]?)?(?:days?|weeks?|months?|years?)\s*\(|\b(?:date_?(?:add|sub|diff)|timedelta|relativedelta|DateOffset|EDATE|EOMONTH|WORKDAY)\s*\(|\.\s*(?:set(?:UTC)?(?:Date|Month|FullYear)|add|subtract)\s*\(|\binterval\s+(?:['"]|\d)|\b(?:date|datetime)(?:\s*\.\s*\w+)?\s*\([^;\n]*\)\s*[+-]"#,
        )
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

/// Return `true` iff `spec` is eligible for automatic source-date candidate grounding
/// (`source_dates.py::eligible`).
pub fn eligible(spec: &Value) -> bool {
    let Some(obj) = spec.as_object() else {
        return false;
    };
    if obj.get("type").and_then(Value::as_str) != Some("date")
        || is_truthy(obj.get("knowledge"))
        || obj.contains_key("source_only")
        || obj.contains_key("null_threshold")
    {
        return false;
    }
    let instructions = match obj.get("instructions") {
        None => "",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => return false,
    };
    reading_intent_regex().is_match(instructions)
        && !derived_regex().is_match(instructions)
        && !relative_regex().is_match(instructions)
        && !expression_request_regex().is_match(instructions)
}

/// Extracted printed date candidate (`source_dates.py::Candidate`).
///
/// `start` and `end` are Unicode character offsets into the source string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Candidate {
    pub start: usize,
    pub end: usize,
    pub printed: String,
    pub iso_values: Vec<String>,
    pub notation: &'static str,
}

impl Candidate {
    pub fn ambiguous(&self) -> bool {
        self.iso_values.len() != 1
    }

    pub fn to_json(&self) -> Value {
        json!({
            "start": self.start,
            "end": self.end,
            "printed": self.printed,
            "iso_values": self.iso_values,
            "notation": self.notation,
        })
    }
}

fn iso(year: i32, month: u32, day: u32) -> Option<String> {
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return None;
    }
    let max_day = schema::days_in_month(year, month);
    if day < 1 || day > max_day {
        return None;
    }
    Some(schema::format_iso_date(year, month, day))
}

fn ordinal_valid(day: u32, suffix: Option<&str>) -> bool {
    let Some(suf) = suffix else {
        return true;
    };
    if suf.is_empty() {
        return true;
    }
    let expected = if (11..=13).contains(&(day % 100)) {
        "th"
    } else {
        match day % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    };
    suf.eq_ignore_ascii_case(expected)
}

/// Extract all printed calendar date candidates from `source` (`source_dates.py::candidates`).
///
/// `numeric_order` may be `None`, `Some("mdy")`, or `Some("dmy")`.
pub fn candidates(source: &str, numeric_order: Option<&str>) -> Result<Vec<Candidate>> {
    if !matches!(numeric_order, None | Some("mdy") | Some("dmy")) {
        return Err(invalid_request(
            "Numeric date order must be explicitly mdy or dmy",
        ));
    }

    let b2c = schema::byte_to_char_map(source);
    let mut found: HashSet<Candidate> = HashSet::new();

    for pattern in [dmy_regex(), mdy_regex()] {
        for caps in pattern.captures_iter(source).flatten() {
            let Some(full) = caps.get(0) else {
                continue;
            };
            let Some(day_num) = caps
                .name("day")
                .and_then(|m| m.as_str().parse::<u32>().ok())
            else {
                continue;
            };
            let ordinal = caps.name("ordinal").map(|m| m.as_str());
            if !ordinal_valid(day_num, ordinal) {
                continue;
            }
            let Some(month) = caps.name("month").and_then(|m| parse_month(m.as_str())) else {
                continue;
            };
            let Some(year) = caps
                .name("year")
                .and_then(|m| m.as_str().parse::<i32>().ok())
            else {
                continue;
            };
            if let Some(value) = iso(year, month, day_num) {
                found.insert(Candidate {
                    start: b2c[full.start()],
                    end: b2c[full.end()],
                    printed: full.as_str().to_string(),
                    iso_values: vec![value],
                    notation: "named_month",
                });
            }
        }
    }

    for caps in ymd_regex().captures_iter(source).flatten() {
        let Some(full) = caps.get(0) else {
            continue;
        };
        if let (Some(y), Some(m), Some(d)) = (
            caps.name("year")
                .and_then(|v| v.as_str().parse::<i32>().ok()),
            caps.name("month")
                .and_then(|v| v.as_str().parse::<u32>().ok()),
            caps.name("day")
                .and_then(|v| v.as_str().parse::<u32>().ok()),
        ) && let Some(value) = iso(y, m, d)
        {
            found.insert(Candidate {
                start: b2c[full.start()],
                end: b2c[full.end()],
                printed: full.as_str().to_string(),
                iso_values: vec![value],
                notation: "year_first",
            });
        }
    }

    let orders: &[&str] = match numeric_order {
        Some("mdy") => &["mdy"],
        Some("dmy") => &["dmy"],
        _ => &["mdy", "dmy"],
    };
    for caps in numeric_regex().captures_iter(source).flatten() {
        let Some(full) = caps.get(0) else {
            continue;
        };
        let (Some(first), Some(second), Some(year)) = (
            caps.name("first")
                .and_then(|v| v.as_str().parse::<u32>().ok()),
            caps.name("second")
                .and_then(|v| v.as_str().parse::<u32>().ok()),
            caps.name("year")
                .and_then(|v| v.as_str().parse::<i32>().ok()),
        ) else {
            continue;
        };
        let mut readings: BTreeSet<String> = BTreeSet::new();
        for &order in orders {
            let (month, day_num) = if order == "mdy" {
                (first, second)
            } else {
                (second, first)
            };
            if let Some(value) = iso(year, month, day_num) {
                readings.insert(value);
            }
        }
        if !readings.is_empty() {
            found.insert(Candidate {
                start: b2c[full.start()],
                end: b2c[full.end()],
                printed: full.as_str().to_string(),
                iso_values: readings.into_iter().collect(),
                notation: "numeric",
            });
        }
    }

    let mut result: Vec<Candidate> = found.into_iter().collect();
    result.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));
    Ok(result)
}

fn slice_chars(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

/// Source-date candidate index & grammar builder (`source_dates.py::DateSource`).
#[derive(Debug, Clone)]
pub struct DateSource {
    pub text: String,
    pub candidates: Vec<Candidate>,
    pub ambiguous: Vec<Candidate>,
    pub support: HashMap<String, Vec<Candidate>>,
    /// Distinct unambiguous ISO date values in first-seen source order.
    pub values: Vec<String>,
}

impl DateSource {
    pub fn new(source: &str, numeric_order: Option<&str>) -> Result<Self> {
        let extracted = candidates(source, numeric_order)?;
        let mut ambiguous = Vec::new();
        let mut support: HashMap<String, Vec<Candidate>> = HashMap::new();
        let mut values: Vec<String> = Vec::new();

        for item in &extracted {
            if item.ambiguous() {
                ambiguous.push(item.clone());
            } else if let Some(iso_val) = item.iso_values.first() {
                if !support.contains_key(iso_val) {
                    values.push(iso_val.clone());
                }
                support
                    .entry(iso_val.clone())
                    .or_default()
                    .push(item.clone());
            }
        }

        Ok(Self {
            text: source.to_string(),
            candidates: extracted,
            ambiguous,
            support,
            values,
        })
    }

    /// Build the Clairic EBNF rule string constraining output to unambiguous source dates
    /// (and optionally `"null"`).
    pub fn constraint_ebnf(&self, nullable: bool) -> Result<String> {
        let mut literals: Vec<String> =
            Vec::with_capacity(self.values.len() + usize::from(nullable));
        for value in &self.values {
            let quoted = serde_json::to_string(value)
                .map_err(|e| invalid_request(format!("failed to encode date literal: {e}")))?;
            literals.push(quoted);
        }
        if nullable {
            literals.push("null".to_string());
        }
        if literals.is_empty() {
            return Err(invalid_request(
                "No unambiguous source date satisfies a nonnullable date contract",
            ));
        }
        let mut alternatives = Vec::with_capacity(literals.len());
        for lit in &literals {
            let ebnf_lit = serde_json::to_string(lit)
                .map_err(|e| invalid_request(format!("failed to encode EBNF literal: {e}")))?;
            alternatives.push(ebnf_lit);
        }
        Ok(format!("root ::= {}", alternatives.join(" | ")))
    }

    /// Build a [`GrammarSpec::ClairicEbnf`] constraint (`source_dates.py::DateSource.constraint`).
    pub fn constraint(&self, nullable: bool) -> Result<GrammarSpec> {
        Ok(GrammarSpec::ClairicEbnf(self.constraint_ebnf(nullable)?))
    }

    /// Look up whether `value` is an unambiguous source date and its unique char span if unique.
    pub fn span_str(&self, value: &str) -> (bool, Option<[usize; 2]>) {
        let Some(matches) = self.support.get(value) else {
            return (false, None);
        };
        let intervals: HashSet<(usize, usize)> = matches.iter().map(|c| (c.start, c.end)).collect();
        if intervals.is_empty() {
            (false, None)
        } else if intervals.len() == 1
            && let Some(&(s, e)) = intervals.iter().next()
        {
            (true, Some([s, e]))
        } else {
            (true, None)
        }
    }

    /// JSON-value overload of [`Self::span_str`].
    pub fn span(&self, value: &Value) -> (bool, Option<[usize; 2]>) {
        match value.as_str() {
            Some(s) => self.span_str(s),
            None => (false, None),
        }
    }

    pub fn evidence(&self, value: &str) -> Vec<Value> {
        self.support
            .get(value)
            .map(|items| items.iter().map(Candidate::to_json).collect())
            .unwrap_or_default()
    }

    /// Annotate date supplement predictions with grounding spans and evidence
    /// (`source_dates.py::DateSource.annotate_predictions`).
    pub fn annotate_predictions(
        &self,
        predictions: &Map<String, Value>,
        names: &[String],
    ) -> Map<String, Value> {
        let mut result = predictions.clone();
        for name in names {
            let Some(orig) = predictions.get(name).and_then(Value::as_object) else {
                continue;
            };
            let mut answer = orig.clone();
            answer.insert(
                "date_decoder".to_string(),
                Value::String(VERSION.to_string()),
            );
            answer.insert(
                "score_representation".to_string(),
                Value::String("native_json_iso_date_wire".to_string()),
            );

            let valid = is_truthy(answer.get("valid"));
            let complete = is_truthy(answer.get("complete"));
            let value_opt = answer.get("value").cloned();
            if !valid || !complete || value_opt.as_ref().is_none_or(Value::is_null) {
                result.insert(name.clone(), Value::Object(answer));
                continue;
            }
            let val = value_opt.unwrap_or(Value::Null);
            let (found, span) = self.span(&val);
            if !found {
                answer.insert("valid".to_string(), Value::Bool(false));
                answer.insert("value".to_string(), Value::Null);
                answer.insert("span".to_string(), Value::Null);
                answer.insert(
                    "failure_reason".to_string(),
                    Value::String("not_a_printed_date_candidate".to_string()),
                );
                result.insert(name.clone(), Value::Object(answer));
                continue;
            }
            let val_str = val.as_str().unwrap_or("");
            answer.insert(
                "grounding_method".to_string(),
                Value::String("printed_date_candidates_source".to_string()),
            );
            answer.insert(
                "span".to_string(),
                match span {
                    Some([s, e]) => json!([s, e]),
                    None => Value::Null,
                },
            );
            answer.insert(
                "source_date_candidates".to_string(),
                Value::Array(self.evidence(val_str)),
            );
            answer.remove("span_method");
            if let Some([s, e]) = span {
                answer.insert(
                    "span_method".to_string(),
                    Value::String("unique_normalized_printed_date_match".to_string()),
                );
                answer.insert(
                    "printed_value".to_string(),
                    Value::String(slice_chars(&self.text, s, e)),
                );
            }
            result.insert(name.clone(), Value::Object(answer));
        }
        result
    }

    /// Merge verified source-date supplement predictions into `original`
    /// (`source_dates.py::DateSource.merge_predictions`).
    pub fn merge_predictions(
        &self,
        original: &Map<String, Value>,
        alternatives: &Map<String, Value>,
        names: &[String],
    ) -> Map<String, Value> {
        if !self.automatic_route_safe() {
            return original.clone();
        }
        let mut result = original.clone();
        for name in names {
            let Some(previous) = original.get(name).and_then(Value::as_object) else {
                continue;
            };
            let Some(answer) = alternatives.get(name).and_then(Value::as_object) else {
                continue;
            };
            let prev_type_ok = match previous.get("type") {
                None | Some(Value::Null) => true,
                Some(Value::String(s)) => s == "date",
                Some(_) => false,
            };
            if !prev_type_ok {
                continue;
            }
            let Some(prev_val) = previous.get("value").and_then(Value::as_str) else {
                continue;
            };
            if self.support.contains_key(prev_val) {
                continue;
            }
            let Some(value) = answer.get("value").and_then(Value::as_str) else {
                continue;
            };
            let expected_span = match self.span_str(value).1 {
                Some([s, e]) => json!([s, e]),
                None => Value::Null,
            };
            let answer_span = answer.get("span").cloned().unwrap_or(Value::Null);
            let Some(raw) = answer.get("raw").and_then(Value::as_str) else {
                continue;
            };
            if answer.get("valid") != Some(&Value::Bool(true))
                || answer.get("complete") != Some(&Value::Bool(true))
                || !self.support.contains_key(value)
                || answer.get("date_decoder").and_then(Value::as_str) != Some(VERSION)
                || answer.get("grounding_method").and_then(Value::as_str)
                    != Some("printed_date_candidates_source")
                || answer_span != expected_span
            {
                continue;
            }
            let Ok(parsed_date) = schema::parse_iso_date(value) else {
                continue;
            };
            if parsed_date.to_iso_string() != value {
                continue;
            }
            let Ok(parsed_raw) = serde_json::from_str::<Value>(raw) else {
                continue;
            };
            if parsed_raw.as_str() != Some(value) {
                continue;
            }
            if value == prev_val {
                continue;
            }
            let mut merged = answer.clone();
            merged.insert("type".to_string(), Value::String("date".to_string()));
            result.insert(name.clone(), Value::Object(merged));
        }
        result
    }

    pub fn automatic_route_safe(&self) -> bool {
        !self.values.is_empty()
            && self.ambiguous.is_empty()
            && !calendar_expression_regex().is_match(&self.text)
    }

    pub fn allows(&self, spec: &Value) -> bool {
        eligible(spec) && self.automatic_route_safe()
    }

    pub fn report(&self) -> Value {
        json!({
            "version": VERSION,
            "values": self.values,
            "candidates": self.candidates.iter().map(Candidate::to_json).collect::<Vec<_>>(),
            "ambiguous_candidates": self.ambiguous.iter().map(Candidate::to_json).collect::<Vec<_>>(),
            "calendar_expression_in_source": calendar_expression_regex().is_match(&self.text),
            "automatic_route_safe": self.automatic_route_safe(),
        })
    }
}

/// Supplement plan built for text-only requests with eligible date fields (`backend.py:1307-1350`).
#[derive(Debug, Clone)]
pub struct DateSupplementPlan {
    pub source: DateSource,
    pub names: Vec<String>,
    pub internal_fields: Map<String, Value>,
    pub constraints: HashMap<String, GrammarSpec>,
}

/// Build a [`DateSupplementPlan`] for eligible date fields in a text-only request.
pub fn build_supplement_plan(
    state: &str,
    request_fields: &Map<String, Value>,
    printed_temporal_fields: &HashSet<String>,
    has_media: bool,
) -> Result<Option<DateSupplementPlan>> {
    if has_media {
        return Ok(None);
    }
    let filtered: Vec<(&String, &Value)> = request_fields
        .iter()
        .filter(|(name, spec)| eligible(spec) && !printed_temporal_fields.contains(name.as_str()))
        .collect();
    if filtered.is_empty() {
        return Ok(None);
    }
    let source = DateSource::new(state, None)?;
    let mut names = Vec::new();
    let mut internal_fields = Map::new();
    let mut constraints = HashMap::new();

    for (name, spec) in filtered {
        if !source.allows(spec) {
            continue;
        }
        let nullable = spec
            .get("nullable")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let orig_inst = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        internal_fields.insert(
            name.clone(),
            json!({
                "type": "text",
                "nullable": nullable,
                "instructions": format!("{INSTRUCTION}\n{orig_inst}")
            }),
        );
        constraints.insert(name.clone(), source.constraint(nullable)?);
        names.push(name.clone());
    }

    if names.is_empty() {
        return Ok(None);
    }

    Ok(Some(DateSupplementPlan {
        source,
        names,
        internal_fields,
        constraints,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_named_ordinal_iso_and_numeric_dates_with_char_offsets() {
        let text = "Signed on the 21st day of Sept. 2024; due 2024-10-15; alt 25/11/2024; bad 22st Jan 2024.";
        let found = candidates(text, None).unwrap();
        let values: Vec<&[String]> = found.iter().map(|c| c.iso_values.as_slice()).collect();
        assert_eq!(
            values,
            vec![
                &["2024-09-21".to_string()][..],
                &["2024-10-15".to_string()][..],
                &["2024-11-25".to_string()][..],
            ]
        );
        // Verify char offset slicing matches printed text
        for c in &found {
            assert_eq!(slice_chars(text, c.start, c.end), c.printed);
        }

        // Unicode prefix preserves char offsets
        let uni = "Café — March 3rd, 2025";
        let uni_found = candidates(uni, None).unwrap();
        assert_eq!(uni_found.len(), 1);
        assert_eq!(uni_found[0].start, 7);
        assert_eq!(
            slice_chars(uni, uni_found[0].start, uni_found[0].end),
            "March 3rd, 2025"
        );
    }

    #[test]
    fn ambiguous_numeric_and_calendar_expressions_disable_auto_route() {
        let ambig = DateSource::new("Date: 03/04/2024", None).unwrap();
        assert!(!ambig.automatic_route_safe());
        assert_eq!(ambig.ambiguous.len(), 1);

        let forced = DateSource::new("Date: 03/04/2024", Some("dmy")).unwrap();
        assert!(forced.automatic_route_safe());
        assert_eq!(forced.values, vec!["2024-04-03"]);

        let expr = DateSource::new("due = date_add(2024-01-01, 30)", None).unwrap();
        assert!(!expr.automatic_route_safe());
    }

    #[test]
    fn constraint_annotate_merge_and_supplement_plan() {
        let state = "Contract dated 1st of April, 2025. Expires 2026-04-01.";
        let fields = json!({
            "effective_date": {"type": "date", "instructions": "Stated effective date of the contract."},
            "derived_date": {"type": "date", "instructions": "Date 30 days after effective date."}
        });
        let plan =
            build_supplement_plan(state, fields.as_object().unwrap(), &HashSet::new(), false)
                .unwrap()
                .unwrap();
        assert_eq!(plan.names, vec!["effective_date"]);
        let GrammarSpec::ClairicEbnf(ebnf) = &plan.constraints["effective_date"] else {
            panic!("expected ClairicEbnf");
        };
        assert_eq!(
            ebnf,
            r#"root ::= "\"2025-04-01\"" | "\"2026-04-01\"" | "null""#
        );

        let raw_alt = json!({
            "effective_date": {
                "valid": true,
                "complete": true,
                "value": "2025-04-01",
                "raw": "\"2025-04-01\""
            }
        });
        let annotated = plan
            .source
            .annotate_predictions(raw_alt.as_object().unwrap(), &plan.names);
        assert_eq!(
            annotated["effective_date"]["printed_value"],
            "1st of April, 2025"
        );
        assert_eq!(
            annotated["effective_date"]["span_method"],
            "unique_normalized_printed_date_match"
        );

        // Original had a hallucinated date not in source => merge replaces it
        let original = json!({
            "effective_date": {
                "type": "date",
                "valid": true,
                "complete": true,
                "value": "2025-05-01"
            }
        });
        let merged =
            plan.source
                .merge_predictions(original.as_object().unwrap(), &annotated, &plan.names);
        assert_eq!(merged["effective_date"]["value"], "2025-04-01");
        assert_eq!(merged["effective_date"]["type"], "date");
    }
}
