//! Source-occurrence numeric sum candidate extraction, grammar construction, and validation.
//!
//! Direct port of upstream `clairic_runtime/source_sums.py`.

use super::{
    canonical as canonical_bytes, char_to_byte_map, is_currency_symbol, is_truthy, sha256_hex,
    slice_chars,
};
use crate::grammar::GrammarSpec;
use crate::models::clairic::{
    ar, decimal_util, grounding, json_validate, schema, typed_prompt::numeral_run_regex,
};
use bigdecimal::{BigDecimal, Zero};
use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

pub const MAX_SOURCE_CANDIDATES: usize = 512;
pub const MAX_OPERANDS: usize = 32;

pub const INSTRUCTION: &str = "Select the printed numeric source occurrences that are operands of the requested sum. Follow the field instructions to identify the requested entities, rows, scope and inclusions. Return exactly one compact JSON array of candidate IDs in their source order. Include every matching occurrence once, even when different occurrences have equal amounts. Select the final amount for each requested record; do not also select its component amounts when those components are already included in that final amount. Exclude unrelated amounts, dates, identifiers and aggregate totals outside the requested scope. Do not calculate the sum, invent an amount, deduplicate equal source occurrences or reuse an ID. The original document is reproduced verbatim below; candidate contexts are source excerpts. Return null if no requested operands are present or the operand scope cannot be determined. A printed zero or negative amount is present. End immediately after the array or null.";

pub(crate) const CURRENCY_CODES: &[&str] = &[
    "AUD", "BRL", "CAD", "CHF", "CNY", "DKK", "EUR", "GBP", "HKD", "INR", "JPY", "KRW", "MXN",
    "NOK", "NZD", "SEK", "SGD", "USD", "ZAR",
];

pub(crate) fn is_currency_code(upper: &str) -> bool {
    CURRENCY_CODES.binary_search(&upper).is_ok()
}

fn sum_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^\s*(?:the\s+)?sum\s+of\b").unwrap_or_else(|_| unreachable!())
    })
}

fn protected_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:not|never|no|without|don't|avoid|unless|if|copy|copies|copied|printed|verbatim|literal|quoted|label|named|called)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn amount_prefix_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\A(?P<open>\()?[ \t]*(?P<before>[+-]?)[ \t]*(?P<currency>\$|[A-Z]{3})?[ \t]*(?P<after>[+-]?)[ \t]*\z",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn iso_date_prefix_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(r"\A\d{4}-\d{2}-\d{2}(?!\d)").unwrap_or_else(|_| unreachable!())
    })
}

fn closing_paren_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\A[ \t]*\)").unwrap_or_else(|_| unreachable!()))
}

/// Canonical JSON bytes matching `source_sums.py::canonical`.
pub fn canonical(value: &Value) -> Vec<u8> {
    canonical_bytes(value)
}

/// Return `true` iff field `name` in `record` is eligible for source-occurrence summing
/// (`source_sums.py::eligible`).
pub fn eligible(record: &Value, name: &str, has_media: bool) -> bool {
    let Some(spec) = record.get("fields").and_then(|f| f.get(name)) else {
        return false;
    };
    if !spec.is_object() {
        return false;
    }
    let annotation = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(|f| f.get(name));
    let in_source_row_gates = record
        .get("_source_row_gates")
        .map(|v| match v {
            Value::Array(arr) => arr.iter().any(|x| x.as_str() == Some(name)),
            Value::Object(obj) => obj.contains_key(name),
            _ => false,
        })
        .unwrap_or(false);
    let instructions = match spec.get("instructions") {
        None => "",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => return false,
    };

    !has_media
        && !is_truthy(record.get("images"))
        && !is_truthy(record.get("documents"))
        && !is_truthy(record.get("_media_payload"))
        && spec.get("type").and_then(Value::as_str) == Some("number")
        && !is_truthy(spec.get("source_only"))
        && !is_truthy(spec.get("knowledge"))
        && !is_truthy(spec.get("_physical_row_scope"))
        && !is_truthy(annotation.and_then(|a| a.get("parent")))
        && !in_source_row_gates
        && !grounding::state_only(spec)
        && !grounding::document_only(spec)
        && sum_regex().is_match(instructions)
        && !protected_regex().is_match(instructions)
}

fn canonical_decimal(d: &BigDecimal) -> BigDecimal {
    if d.is_zero() {
        BigDecimal::from(0)
    } else {
        d.normalized()
    }
}

/// Compute signed numeric readings for one source span (`source_sums.py::_signed_readings`).
pub(crate) fn signed_readings(
    source: &str,
    chars: &[char],
    c2b: &[usize],
    b2c: &[usize],
    start: usize,
    end: usize,
    values: &[BigDecimal],
) -> Vec<BigDecimal> {
    let Some(&b_start) = c2b.get(start) else {
        return Vec::new();
    };
    let Some(&b_end) = c2b.get(end) else {
        return Vec::new();
    };
    if b_start >= b_end || b_end > source.len() {
        return Vec::new();
    }
    let Ok(Some(numeral)) = numeral_run_regex().find_from_pos(&source[..b_end], b_start) else {
        return Vec::new();
    };
    let left = b2c[numeral.start()];
    let numeral_str = numeral.as_str();

    let mut prefix_start = left;
    let min_prefix = left.saturating_sub(64);
    while prefix_start > min_prefix {
        let ch = chars[prefix_start - 1];
        if matches!(ch, ' ' | '\t' | '(' | '+' | '-' | '\u{2212}') || is_currency_symbol(ch) {
            prefix_start -= 1;
        } else if prefix_start >= 3 {
            let code: String = chars[prefix_start - 3..prefix_start].iter().collect();
            let upper = code.to_ascii_uppercase();
            if is_currency_code(&upper)
                && (prefix_start == 3 || !chars[prefix_start - 4].is_alphanumeric())
            {
                prefix_start -= 3;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    let prefix = slice_chars(source, c2b, prefix_start, left);
    let leading_ws_chars = prefix.chars().take_while(|c| c.is_whitespace()).count();
    let first = prefix_start + leading_ws_chars;
    let prefix_trimmed = prefix.trim();

    if matches!(prefix_trimmed, "-" | "\u{2212}")
        && iso_date_prefix_regex()
            .is_match(&source[c2b[left]..])
            .unwrap_or(false)
    {
        return dedup_decimals(values.iter().map(BigDecimal::abs));
    }

    if first < left
        && matches!(chars[first], '+' | '-' | '\u{2212}')
        && first > 0
        && chars[first - 1].is_alphanumeric()
    {
        if matches!(prefix_trimmed, "+" | "-" | "\u{2212}") {
            return dedup_decimals(values.iter().map(BigDecimal::abs));
        }
        return Vec::new();
    }

    let normalized_full: String = prefix
        .chars()
        .map(|ch| {
            if is_currency_symbol(ch) {
                "$".to_string()
            } else if ch == '\u{2212}' {
                "-".to_string()
            } else {
                ch.to_uppercase().to_string()
            }
        })
        .collect();
    let normalized = normalized_full.trim_start();

    let Some(caps) = amount_prefix_regex().captures(normalized) else {
        return Vec::new();
    };
    let has_open = caps.name("open").is_some();
    let before = caps.name("before").map(|m| m.as_str()).unwrap_or("");
    let has_currency = caps.name("currency").is_some();
    let after_sign = caps.name("after").map(|m| m.as_str()).unwrap_or("");
    if !before.is_empty() && !after_sign.is_empty() {
        return Vec::new();
    }
    let explicit_sign = if !before.is_empty() {
        before
    } else {
        after_sign
    };
    let mut negative = explicit_sign == "-";

    if has_open {
        let after_char_idx = if end > 0 && chars[end - 1] == ')' {
            end - 1
        } else {
            end
        };
        let after = &source[c2b[after_char_idx]..];
        let Some(closing) = closing_paren_regex().find(after) else {
            return Vec::new();
        };
        if !explicit_sign.is_empty() {
            return Vec::new();
        }
        let rest_line = after[closing.end()..].split('\n').next().unwrap_or("");
        let rest = rest_line.trim_matches(|c| matches!(c, ' ' | '\t' | '.' | ';' | ','));
        negative = has_currency
            || numeral_str.contains('.')
            || numeral_str.contains(',')
            || rest.is_empty();
    }

    dedup_decimals(values.iter().map(|v| {
        let a = v.abs();
        if negative { -a } else { a }
    }))
}

fn dedup_decimals<I: IntoIterator<Item = BigDecimal>>(iter: I) -> Vec<BigDecimal> {
    let mut seen: HashSet<BigDecimal> = HashSet::new();
    let mut out = Vec::new();
    for val in iter {
        let norm = canonical_decimal(&val);
        if seen.insert(norm.clone()) {
            out.push(norm);
        }
    }
    out
}

/// Extract ordered non-overlapping numeric source candidates (`source_sums.py::_source_candidates`).
pub fn source_candidates(source: &str) -> Option<Vec<Value>> {
    let chars: Vec<char> = source.chars().collect();
    let c2b = char_to_byte_map(source);
    let b2c = schema::byte_to_char_map(source);

    let mut readings: BTreeMap<(usize, usize), Vec<BigDecimal>> = BTreeMap::new();
    for (value, spans) in grounding::numbers(source) {
        for span in spans {
            readings.entry(span).or_default().push(value.clone());
        }
    }

    if readings.is_empty() || readings.len() > MAX_SOURCE_CANDIDATES {
        return None;
    }

    let char_len = chars.len();
    let mut previous_end = 0usize;
    let mut candidates = Vec::with_capacity(readings.len());

    for (index, ((start, end), raw_values)) in readings.into_iter().enumerate() {
        if start < previous_end || !(start < end && end <= char_len) {
            return None;
        }
        previous_end = end;

        let line_start = chars[..start]
            .iter()
            .rposition(|&c| c == '\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        let line_end = chars[end..]
            .iter()
            .position(|&c| c == '\n')
            .map(|off| end + off)
            .unwrap_or(char_len);
        let context_start = line_start.max(start.saturating_sub(64));
        let context_end = line_end.min(end.saturating_add(64));

        let values = signed_readings(source, &chars, &c2b, &b2c, start, end, &raw_values);
        let mut decimal_options: Vec<String> = values
            .iter()
            .map(decimal_util::normalize_decimal_str)
            .collect();
        decimal_options.sort();
        decimal_options.dedup();

        let decimal_field = if values.len() == 1 {
            Value::String(decimal_options[0].clone())
        } else {
            Value::Null
        };

        let mut obj = Map::new();
        obj.insert("id".to_string(), Value::String(format!("N{index}")));
        obj.insert("span".to_string(), json!([start, end]));
        obj.insert(
            "printed".to_string(),
            Value::String(slice_chars(source, &c2b, start, end).to_string()),
        );
        obj.insert("decimal".to_string(), decimal_field);
        obj.insert(
            "decimal_options".to_string(),
            Value::Array(decimal_options.into_iter().map(Value::String).collect()),
        );
        obj.insert(
            "context".to_string(),
            Value::String(slice_chars(source, &c2b, context_start, context_end).to_string()),
        );
        candidates.push(Value::Object(obj));
    }

    Some(candidates)
}

/// Build the monotonic candidate-subset EBNF grammar (`source_sums.py::_grammar`).
pub fn candidate_subset_ebnf(candidates: &[Value]) -> String {
    let mut lines = Vec::with_capacity(candidates.len() + 1);
    lines.push(r#"root ::= "null" | "[" p0 "]""#.to_string());
    for (index, item) in candidates.iter().enumerate() {
        let id = item.get("id").and_then(Value::as_str).unwrap_or("");
        let inner = serde_json::to_string(id).unwrap_or_default();
        let literal = serde_json::to_string(&inner).unwrap_or_default();
        let mut rule = format!("p{index} ::= {literal}");
        if index + 1 < candidates.len() {
            let next = index + 1;
            rule.push_str(&format!(r#" | {literal} "," p{next} | p{next}"#));
        }
        lines.push(rule);
    }
    lines.join("\n")
}

/// Build the [`GrammarSpec::ClairicEbnf`] constraint for `candidates`.
pub fn candidate_subset_grammar(candidates: &[Value]) -> GrammarSpec {
    GrammarSpec::ClairicEbnf(candidate_subset_ebnf(candidates))
}

/// Prepare a source-occurrence sum task for `field_name` in `record` (`source_sums.py::prepare_task`).
pub fn prepare_task(record: &Value, field_name: &str, has_media: bool) -> Option<Value> {
    if !eligible(record, field_name, has_media) {
        return None;
    }
    let source = record.get("input").and_then(Value::as_str)?;
    let candidates = source_candidates(source)?;
    if candidates.is_empty() {
        return None;
    }
    let spec = record.get("fields")?.get(field_name)?.clone();
    let displayed: Vec<Value> = candidates
        .iter()
        .map(|item| {
            let mut m = Map::new();
            m.insert(
                "id".to_string(),
                item.get("id").cloned().unwrap_or(Value::Null),
            );
            m.insert(
                "printed".to_string(),
                item.get("printed").cloned().unwrap_or(Value::Null),
            );
            m.insert(
                "context".to_string(),
                item.get("context").cloned().unwrap_or(Value::Null),
            );
            Value::Object(m)
        })
        .collect();
    let id_enum: Vec<Value> = candidates
        .iter()
        .map(|item| item.get("id").cloned().unwrap_or(Value::Null))
        .collect();

    let json_schema = json!({
        "anyOf": [
            {"type": "null"},
            {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_OPERANDS,
                "uniqueItems": true,
                "items": {
                    "type": "string",
                    "enum": id_enum,
                },
            },
        ]
    });

    let instructions = spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let displayed_json = ar::compact_json(&Value::Array(displayed));
    let prompt = format!(
        "{INSTRUCTION}\nField: {field_name}\nField instructions: {instructions}\nCandidates in source order: {displayed_json}\n\nOriginal document:\n{source}"
    );
    let ebnf = candidate_subset_ebnf(&candidates);
    let source_sha256 = sha256_hex(source.as_bytes());
    let field_spec_sha256 = sha256_hex(&canonical(&spec));

    let mut out = Map::new();
    out.insert("version".to_string(), json!(2));
    out.insert("kind".to_string(), json!("source-occurrence-sum-v2"));
    out.insert("field".to_string(), Value::String(field_name.to_string()));
    out.insert("input_text".to_string(), Value::String(source.to_string()));
    out.insert("prompt".to_string(), Value::String(prompt));
    out.insert("grammar".to_string(), json!({"ebnf": ebnf}));
    out.insert("json_schema".to_string(), json_schema);
    out.insert("candidates".to_string(), Value::Array(candidates));
    out.insert("field_spec".to_string(), spec);
    out.insert("max_operands".to_string(), json!(MAX_OPERANDS));
    out.insert("source_sha256".to_string(), Value::String(source_sha256));
    out.insert(
        "field_spec_sha256".to_string(),
        Value::String(field_spec_sha256),
    );
    out.insert(
        "semantic_completeness_proven_by_grounding".to_string(),
        Value::Bool(false),
    );
    Some(Value::Object(out))
}

fn format_exact_decimal_sum(total: &BigDecimal) -> String {
    if total.is_zero() {
        return decimal_util::to_plain_string(total);
    }
    let exp = decimal_util::adjusted_exponent(total);
    if exp.abs() <= 256 && total.fractional_digit_count().abs() <= 256 {
        decimal_util::to_plain_string(total)
    } else {
        total.to_string()
    }
}

/// Validate a raw JSON candidate-ID array (or `"null"`) against `task`, computing the exact
/// `BigDecimal` sum and validating against `ar::schema(field_spec)` (`source_sums.py::validate_result`).
pub fn validate_result(task: &Value, raw_json: &str, complete: bool) -> Value {
    let mut result = Map::new();
    result.insert("valid".to_string(), Value::Bool(false));
    result.insert("value".to_string(), Value::Null);
    result.insert("operands".to_string(), Value::Array(Vec::new()));
    result.insert("decimal_sum".to_string(), Value::Null);
    result.insert("operand_count".to_string(), json!(0));
    result.insert("error".to_string(), Value::Null);
    result.insert(
        "semantic_completeness_proven_by_grounding".to_string(),
        Value::Bool(false),
    );

    if let Err(err) = validate_result_inner(task, raw_json, complete, &mut result) {
        result.insert("valid".to_string(), Value::Bool(false));
        result.insert("value".to_string(), Value::Null);
        result.insert("operands".to_string(), Value::Array(Vec::new()));
        result.insert("decimal_sum".to_string(), Value::Null);
        result.insert("operand_count".to_string(), json!(0));
        result.insert("error".to_string(), Value::String(err));
    }

    Value::Object(result)
}

fn validate_result_inner(
    task: &Value,
    raw_json: &str,
    complete: bool,
    result: &mut Map<String, Value>,
) -> std::result::Result<(), String> {
    if !complete {
        return Err("incomplete_generation".to_string());
    }
    let source = task
        .get("input_text")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_input_text".to_string())?;
    let expected_source_sha = task
        .get("source_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_source_sha256".to_string())?;
    if sha256_hex(source.as_bytes()) != expected_source_sha {
        return Err("source_changed".to_string());
    }
    let field_spec = task
        .get("field_spec")
        .ok_or_else(|| "missing_field_spec".to_string())?;
    let expected_spec_sha = task
        .get("field_spec_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_field_spec_sha256".to_string())?;
    if sha256_hex(&canonical(field_spec)) != expected_spec_sha {
        return Err("field_spec_changed".to_string());
    }

    let ids: Value = serde_json::from_str(raw_json).map_err(|e| e.to_string())?;
    let field_schema = ar::schema(field_spec).map_err(|e| e.to_string())?;

    if ids.is_null() {
        if !json_validate::is_valid(&field_schema, &Value::Null) {
            return Err("schema_validation_failed".to_string());
        }
        result.insert("valid".to_string(), Value::Bool(true));
        result.insert("error".to_string(), Value::Null);
        return Ok(());
    }

    let max_operands = task
        .get("max_operands")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| "missing_max_operands".to_string())?;
    let Some(id_array) = ids.as_array() else {
        return Err("invalid_operand_count".to_string());
    };
    if id_array.is_empty() || id_array.len() > max_operands {
        return Err("invalid_operand_count".to_string());
    }

    let mut id_strings = Vec::with_capacity(id_array.len());
    let mut seen_ids = HashSet::with_capacity(id_array.len());
    for entry in id_array {
        let Some(s) = entry.as_str() else {
            return Err("operand_id_not_string".to_string());
        };
        if !seen_ids.insert(s) {
            return Err("reused_source_occurrence".to_string());
        }
        id_strings.push(s);
    }

    let recomputed = source_candidates(source);
    let task_candidates = task.get("candidates").and_then(Value::as_array);
    let Some(recomputed_vec) = recomputed else {
        return Err("source_candidate_mapping_changed".to_string());
    };
    if task_candidates != Some(&recomputed_vec) {
        return Err("source_candidate_mapping_changed".to_string());
    }

    let mut indexed: HashMap<&str, (usize, &Value)> = HashMap::with_capacity(recomputed_vec.len());
    for (idx, item) in recomputed_vec.iter().enumerate() {
        if let Some(id) = item.get("id").and_then(Value::as_str) {
            indexed.insert(id, (idx, item));
        }
    }

    let mut indices = Vec::with_capacity(id_strings.len());
    let mut operands = Vec::with_capacity(id_strings.len());
    for identifier in &id_strings {
        let Some(&(idx, item)) = indexed.get(*identifier) else {
            return Err("unknown_source_occurrence".to_string());
        };
        indices.push(idx);
        operands.push(item.clone());
    }

    let mut sorted_indices = indices.clone();
    sorted_indices.sort_unstable();
    if indices != sorted_indices {
        return Err("source_order_changed".to_string());
    }

    let mut amounts = Vec::with_capacity(operands.len());
    for item in &operands {
        let Some(dec_str) = item.get("decimal").and_then(Value::as_str) else {
            return Err("ambiguous_operand_spelling".to_string());
        };
        let amount = decimal_util::parse_decimal(dec_str)
            .ok_or_else(|| "invalid_operand_decimal".to_string())?;
        amounts.push(amount);
    }

    let mut total = BigDecimal::from(0);
    for amount in &amounts {
        total += amount;
    }
    let decimal_sum = format_exact_decimal_sum(&total);
    let normalized_value = ar::normalize_number(&Value::String(decimal_sum.clone()), field_spec)
        .map_err(|e| e.to_string())?;
    if !json_validate::is_valid(&field_schema, &normalized_value) {
        return Err("schema_validation_failed".to_string());
    }

    result.insert("valid".to_string(), Value::Bool(true));
    result.insert("value".to_string(), normalized_value);
    result.insert("operand_count".to_string(), json!(operands.len()));
    result.insert("operands".to_string(), Value::Array(operands));
    result.insert("decimal_sum".to_string(), Value::String(decimal_sum));
    result.insert("error".to_string(), Value::Null);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eligibility_checks_sum_instructions_and_guards() {
        let base_record = json!({
            "input": "Coffee $4.50 and Lunch $12.25",
            "fields": {
                "total": {
                    "type": "number",
                    "instructions": "Sum of all meal expenses."
                },
                "protected": {
                    "type": "number",
                    "instructions": "Sum of expenses unless refunded."
                },
                "source_only_field": {
                    "type": "number",
                    "source_only": true,
                    "instructions": "The sum of meal expenses."
                }
            }
        });
        assert!(eligible(&base_record, "total", false));
        assert!(!eligible(&base_record, "total", true));
        assert!(!eligible(&base_record, "protected", false));
        assert!(!eligible(&base_record, "source_only_field", false));
    }

    #[test]
    fn signed_currency_readings_handle_signs_accounting_parens_and_dates() {
        let source = "Date 2025-03-14\nCharge: $10.25\nRefund: -$2.50\nCredit: ($1.75)\nCode A-100\nEuro: €3.00\nUSD -4.00";
        let candidates = source_candidates(source).unwrap();
        let by_printed: HashMap<&str, Option<&str>> = candidates
            .iter()
            .map(|c| (c["printed"].as_str().unwrap(), c["decimal"].as_str()))
            .collect();
        assert_eq!(by_printed.get("$10.25"), Some(&Some("10.25")));
        assert_eq!(by_printed.get("-$2.50"), Some(&Some("-2.5")));
        assert_eq!(by_printed.get("($1.75)"), Some(&Some("-1.75")));
        assert_eq!(by_printed.get("€3.00"), Some(&Some("3")));
        assert_eq!(by_printed.get("-4.00"), Some(&Some("-4")));
        // Hyphenated identifier A-100 keeps positive value 100 even when printed span captures "-100"
        assert_eq!(by_printed.get("-100"), Some(&Some("100")));
    }

    #[test]
    fn prepare_and_validate_exact_bigdecimal_sum_and_schema() {
        let record = json!({
            "input": "Receipt A: $10.10\nReceipt B: $20.20\nRefund C: ($0.30)",
            "fields": {
                "meal_sum": {
                    "type": "number",
                    "decimals": 2,
                    "minimum": 0,
                    "instructions": "The sum of receipt amounts."
                }
            }
        });
        let task = prepare_task(&record, "meal_sum", false).unwrap();
        assert_eq!(task["kind"], "source-occurrence-sum-v2");
        let ebnf = task["grammar"]["ebnf"].as_str().unwrap();
        assert!(ebnf.starts_with(r#"root ::= "null" | "[" p0 "]""#));
        assert!(ebnf.contains(r#"p0 ::= "\"N0\"" | "\"N0\"" "," p1 | p1"#));

        let checked = validate_result(&task, r#"["N0","N1","N2"]"#, true);
        assert_eq!(checked["valid"], json!(true));
        assert_eq!(checked["value"], json!(30.0));
        assert_eq!(checked["decimal_sum"], json!("30.0"));
        assert_eq!(checked["operand_count"], json!(3));

        // Negative result fails `minimum: 0` schema validation
        let negative_checked = validate_result(&task, r#"["N2"]"#, true);
        assert_eq!(negative_checked["valid"], json!(false));
        assert_eq!(negative_checked["error"], json!("schema_validation_failed"));

        // Reversed order fails
        let bad_order = validate_result(&task, r#"["N1","N0"]"#, true);
        assert_eq!(bad_order["valid"], json!(false));
        assert_eq!(bad_order["error"], json!("source_order_changed"));

        // Duplicate ID fails
        let dup = validate_result(&task, r#"["N0","N0"]"#, true);
        assert_eq!(dup["valid"], json!(false));
        assert_eq!(dup["error"], json!("reused_source_occurrence"));

        // Null is allowed by default nullable=true
        let null_ok = validate_result(&task, "null", true);
        assert_eq!(null_ok["valid"], json!(true));
        assert_eq!(null_ok["value"], json!(null));
    }

    #[test]
    fn validate_rejects_non_nullable_null_and_incomplete() {
        let record = json!({
            "input": "Item: $5.00",
            "fields": {
                "total": {
                    "type": "number",
                    "nullable": false,
                    "instructions": "Sum of all items."
                }
            }
        });
        let task = prepare_task(&record, "total", false).unwrap();
        let res = validate_result(&task, "null", true);
        assert_eq!(res["valid"], json!(false));

        let incomplete = validate_result(&task, r#"["N0"]"#, false);
        assert_eq!(incomplete["valid"], json!(false));
        assert_eq!(incomplete["error"], json!("incomplete_generation"));
    }

    #[test]
    fn unicode_char_spans_and_high_precision_decimal_sum() {
        let source = "Café ☕ €12.50\nBistro — €7.50";
        let candidates = source_candidates(source).unwrap();
        assert_eq!(candidates.len(), 2);
        let chars: Vec<char> = source.chars().collect();
        for cand in &candidates {
            let span = cand["span"].as_array().unwrap();
            let s = span[0].as_u64().unwrap() as usize;
            let e = span[1].as_u64().unwrap() as usize;
            let sliced: String = chars[s..e].iter().collect();
            assert_eq!(sliced, cand["printed"].as_str().unwrap());
        }
        let record = json!({
            "input": source,
            "fields": {
                "euro_sum": {
                    "type": "number",
                    "unit": "EUR",
                    "instructions": "Sum of all café expenses."
                }
            }
        });
        let task = prepare_task(&record, "euro_sum", false).unwrap();
        let checked = validate_result(&task, r#"["N0","N1"]"#, true);
        assert_eq!(checked["valid"], json!(true));
        assert_eq!(checked["value"], json!(20));
        assert_eq!(checked["decimal_sum"], json!("20.0"));
    }
}
