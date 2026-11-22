//! Literal printed-number and received-payment row rewriting & normalization.
//!
//! Direct port of upstream `clairic_runtime/printed_numbers.py`.

use super::{ar, decimal_util, errors::invalid_request, json_validate};
use bigdecimal::num_traits::ToPrimitive;
use napi::Result;
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::sync::OnceLock;

pub const COPY_INSTRUCTION: &str = " Copy the complete requested number exactly as printed, including every digit, currency symbol, grouping separator, sign or parentheses. Do not calculate, normalize or round. Return null only if this requested value is absent.";
pub const PAYMENT_PRESENCE_INSTRUCTION: &str = " Every separately printed payment actually received is a present row, including an amount of zero, a negative amount, or a few cents. Copy all matching printed received-payment rows in their printed order, preserving duplicate rows. Do not return null or end the list while a matching received-payment row remains.";

fn number_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\A\s*(?P<open>\()?\s*(?P<before>[+-]?)\s*(?:[$€£¥]|USD|EUR|GBP|CAD|AUD|JPY)?\s*(?P<after>[+-]?)\s*(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?P<close>\))?\s*\z",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn transform_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:calculat\w*|comput\w*|deriv\w*|convert\w*|summar\w*|aggregat\w*|group\w*|combin\w*|deduplicat\w*|normaliz\w*)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn negated_transform_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:do\s+not|never|don't)\s+(?:calculate\w*|compute\w*|derive\w*|convert\w*|summarize\w*|aggregate\w*|group\w*|combine\w*|deduplicate\w*|normalize\w*)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn positive_split_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)[.\n]|\bExclude\b").unwrap_or_else(|_| unreachable!()))
}

fn payments_received_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\bpayments?\b[^.\n]*\breceived\b").unwrap_or_else(|_| unreachable!())
    })
}

fn received_payments_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\breceived\b[^.\n]*\bpayments?\b").unwrap_or_else(|_| unreachable!())
    })
}

fn payment_negation_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:not|never|exclude|excluding|ignore|without)\b")
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

fn literal(spec: &Value) -> bool {
    let Some(obj) = spec.as_object() else {
        return false;
    };
    if is_truthy(obj.get("knowledge")) {
        return false;
    }
    let instructions = match obj.get("instructions") {
        None => "",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => return false,
    };
    let cleaned = negated_transform_regex().replace_all(instructions, "");
    !transform_regex().is_match(&cleaned)
}

fn payment_property(spec: &Value) -> Option<String> {
    let obj = spec.as_object()?;
    if obj.get("type").and_then(Value::as_str) != Some("object") || !literal(spec) {
        return None;
    }
    let properties = obj.get("properties").and_then(Value::as_object)?;
    if properties.len() != 2 {
        return None;
    }
    let mut text_count = 0usize;
    let mut number_name: Option<String> = None;
    let mut number_count = 0usize;
    for (prop_name, child) in properties {
        if !child.is_object() || !literal(child) {
            return None;
        }
        match child.get("type").and_then(Value::as_str) {
            Some("text") => text_count += 1,
            Some("number") => {
                number_count += 1;
                number_name = Some(prop_name.clone());
            }
            _ => {}
        }
    }
    if text_count == 1 && number_count == 1 {
        number_name
    } else {
        None
    }
}

/// Return names of literal received-payment row-list fields (`printed_numbers.py::payment_lists`).
pub fn payment_lists(fields: &Map<String, Value>) -> Vec<String> {
    let mut selected = Vec::new();
    for (name, spec) in fields {
        let Some(obj) = spec.as_object() else {
            continue;
        };
        if obj.get("type").and_then(Value::as_str) != Some("list") || !literal(spec) {
            continue;
        }
        let Some(item) = obj.get("items") else {
            continue;
        };
        if !item.is_object() || payment_property(item).is_none() {
            continue;
        }
        let instructions = obj
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        let positive = match positive_split_regex().find(instructions) {
            Some(m) => &instructions[..m.start()],
            None => instructions,
        };
        if (payments_received_regex().is_match(positive)
            || received_payments_regex().is_match(positive))
            && !payment_negation_regex().is_match(positive)
        {
            selected.push(name.clone());
        }
    }
    selected
}

/// Parse and normalize a printed number string according to `spec` (`printed_numbers.py::normalize`).
pub fn normalize(value: &Value, spec: &Value) -> Result<Value> {
    if value.is_null() {
        return Ok(Value::Null);
    }
    let Some(text) = value.as_str() else {
        return Err(invalid_request("A printed number must be a string"));
    };
    let mut raw = text.trim();
    if let Some(unit) = spec.get("unit").and_then(Value::as_str)
        && !unit.is_empty()
        && let Some(stripped) = raw.strip_suffix(unit)
    {
        raw = stripped.trim_end();
    }
    let Some(caps) = number_regex().captures(raw) else {
        return Err(invalid_request("Unsupported printed number"));
    };
    let has_open = caps.name("open").is_some();
    let has_close = caps.name("close").is_some();
    let before = caps.name("before").map(|m| m.as_str()).unwrap_or("");
    let after = caps.name("after").map(|m| m.as_str()).unwrap_or("");
    if has_open != has_close
        || (!before.is_empty() && !after.is_empty())
        || (has_open && (!before.is_empty() || !after.is_empty()))
    {
        return Err(invalid_request("Unsupported printed number"));
    }

    let number_str = caps
        .name("number")
        .map(|m| m.as_str().replace(',', ""))
        .unwrap_or_default();
    let mut number = decimal_util::parse_decimal(&number_str)
        .ok_or_else(|| invalid_request("Unsupported printed number"))?;
    if !number.to_f64().is_some_and(f64::is_finite) {
        return Err(invalid_request("Nonfinite numeric answer"));
    }
    if has_open || before == "-" || after == "-" {
        number = -number;
    }

    let result = ar::normalize_number(&Value::String(number.to_string()), spec)?;
    if let Some(f) = result.as_f64()
        && !f.is_finite()
    {
        return Err(invalid_request("Nonfinite numeric answer"));
    }
    let schema = ar::schema(spec)?;
    if !json_validate::is_valid(&schema, &result) {
        return Err(invalid_request("Printed number failed schema validation"));
    }
    Ok(result)
}

fn text_spec(spec: &Value) -> Value {
    let nullable = spec
        .get("nullable")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let instructions = spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    json!({
        "type": "text",
        "nullable": nullable,
        "instructions": format!("{instructions}{COPY_INSTRUCTION}")
    })
}

/// Rewrite `source_only` numbers, number lists, and payment rows into literal text queries,
/// returning `(rewritten_fields, contracts)` (`printed_numbers.py::prepare`).
pub fn prepare<I, S>(
    fields: &Map<String, Value>,
    payment_rows: I,
) -> (Map<String, Value>, Map<String, Value>)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut out = fields.clone();
    let mut contracts: Map<String, Value> = Map::new();
    let payment_set: HashSet<String> = payment_rows
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect();

    for parent in payment_lists(&out) {
        let prefix = format!("{parent}.item");
        let children: Vec<String> = payment_set
            .iter()
            .filter(|name| {
                name.strip_prefix(&prefix).is_some_and(|rest| {
                    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
                })
            })
            .cloned()
            .collect();
        if children.is_empty() {
            continue;
        }
        let parent_inst = out
            .get(&parent)
            .and_then(|v| v.get("instructions"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if parent_inst.contains(PAYMENT_PRESENCE_INSTRUCTION) {
            continue;
        }
        let expanded_parent_inst = format!("{parent_inst}{PAYMENT_PRESENCE_INSTRUCTION}");
        if let Some(parent_obj) = out.get_mut(&parent).and_then(Value::as_object_mut) {
            parent_obj.insert(
                "instructions".to_string(),
                Value::String(expanded_parent_inst.clone()),
            );
        }
        for child in children {
            let Some(child_val) = out.get(&child) else {
                continue;
            };
            if payment_property(child_val).is_none() {
                continue;
            }
            let requested = child_val
                .get("instructions")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let updated_inst = if !parent_inst.is_empty() && requested.contains(&parent_inst) {
                requested.replacen(&parent_inst, &expanded_parent_inst, 1)
            } else {
                format!("{requested}{PAYMENT_PRESENCE_INSTRUCTION}")
            };
            if let Some(child_obj) = out.get_mut(&child).and_then(Value::as_object_mut) {
                child_obj.insert("instructions".to_string(), Value::String(updated_inst));
            }
        }
    }

    let names: Vec<String> = out.keys().cloned().collect();
    for name in names {
        let Some(spec) = out.get(&name).cloned() else {
            continue;
        };
        let kind = spec.get("type").and_then(Value::as_str).unwrap_or("");
        let source_only = is_truthy(spec.get("source_only"));
        if kind == "number" && source_only {
            contracts.insert(name.clone(), json!({"scalar": spec}));
            out.insert(name, text_spec(&spec));
        } else if kind == "list"
            && source_only
            && spec.get("items").and_then(Value::as_str) == Some("number")
        {
            let mut updated = spec.as_object().cloned().unwrap_or_default();
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
            out.insert(name, Value::Object(updated));
        } else if payment_set.contains(&name)
            && let Some(number_name) = payment_property(&spec)
            && let Some(props) = spec.get("properties").and_then(Value::as_object)
            && let Some(number_spec) = props.get(&number_name).cloned()
        {
            let mut contract_props = Map::new();
            contract_props.insert(number_name.clone(), number_spec.clone());
            contracts.insert(
                name.clone(),
                json!({
                    "row": spec,
                    "properties": Value::Object(contract_props),
                }),
            );
            let mut new_props = props.clone();
            new_props.insert(number_name, text_spec(&number_spec));
            let mut updated = spec.as_object().cloned().unwrap_or_default();
            updated.insert("properties".to_string(), Value::Object(new_props));
            out.insert(name, Value::Object(updated));
        }
    }

    (out, contracts)
}

enum RowError {
    NotInSource,
    Invalid,
}

fn normalize_row(
    raw: &Value,
    contract: &Value,
    record: &Value,
    name: &str,
) -> std::result::Result<Value, RowError> {
    let result = if raw.is_null() {
        Value::Null
    } else if let Some(raw_obj) = raw.as_object() {
        let mut out = raw_obj.clone();
        let current_props = record
            .get("fields")
            .and_then(|f| f.get(name))
            .and_then(|s| s.get("properties"))
            .and_then(Value::as_object);
        let contract_props = contract
            .get("properties")
            .and_then(Value::as_object)
            .ok_or(RowError::Invalid)?;
        let has_media = is_truthy(record.get("_media_payload"));
        let input_text = record.get("input").and_then(Value::as_str).unwrap_or("");

        for (prop_name, spec) in contract_props {
            let value = raw_obj.get(prop_name).ok_or(RowError::Invalid)?;
            let is_numeric_prop = current_props
                .and_then(|p| p.get(prop_name))
                .and_then(|s| s.get("type"))
                .and_then(Value::as_str)
                == Some("number");
            if is_numeric_prop {
                if !value.is_null() && (value.is_boolean() || !value.is_number()) {
                    return Err(RowError::Invalid);
                }
                let normalized =
                    ar::normalize_number(value, spec).map_err(|_| RowError::Invalid)?;
                out.insert(prop_name.clone(), normalized);
            } else {
                if !value.is_null() && !has_media {
                    let Some(s) = value.as_str() else {
                        return Err(RowError::NotInSource);
                    };
                    if !input_text.contains(s) {
                        return Err(RowError::NotInSource);
                    }
                }
                let normalized = normalize(value, spec).map_err(|_| RowError::Invalid)?;
                out.insert(prop_name.clone(), normalized);
            }
        }
        Value::Object(out)
    } else {
        return Err(RowError::Invalid);
    };

    let row_spec = contract.get("row").ok_or(RowError::Invalid)?;
    let schema = ar::schema(row_spec).map_err(|_| RowError::Invalid)?;
    if !json_validate::is_valid(&schema, &result) {
        return Err(RowError::Invalid);
    }
    Ok(result)
}

/// Normalize printed-number predictions in place (`printed_numbers.py::normalize_predictions`).
pub fn normalize_predictions(
    record: &Value,
    predictions: &Map<String, Value>,
) -> Map<String, Value> {
    let mut result = predictions.clone();
    let Some(contracts) = record
        .get("_meta")
        .and_then(|m| m.get("printed_numbers"))
        .and_then(Value::as_object)
    else {
        return result;
    };
    let has_media = is_truthy(record.get("_media_payload"));
    let input_text = record.get("input").and_then(Value::as_str).unwrap_or("");

    for (name, contract) in contracts {
        let Some(orig_obj) = result.get(name).and_then(Value::as_object) else {
            continue;
        };
        let mut answer = orig_obj.clone();
        if !is_truthy(answer.get("valid")) {
            result.insert(name.clone(), Value::Object(answer));
            continue;
        }
        let raw = answer.get("value").cloned().unwrap_or(Value::Null);
        let outcome: std::result::Result<Value, RowError> = if contract.get("properties").is_some()
        {
            normalize_row(&raw, contract, record, name)
        } else {
            let mut source_ok = true;
            if !raw.is_null() && !has_media {
                source_ok = raw.as_str().is_some_and(|s| input_text.contains(s));
            }
            if !source_ok {
                Err(RowError::NotInSource)
            } else {
                match contract.get("scalar") {
                    Some(scalar_spec) => {
                        normalize(&raw, scalar_spec).map_err(|_| RowError::Invalid)
                    }
                    None => Err(RowError::Invalid),
                }
            }
        };

        match outcome {
            Ok(normalized) => {
                answer.insert("value".to_string(), normalized);
                answer.insert("printed_value".to_string(), raw);
                answer.insert(
                    "normalization_method".to_string(),
                    Value::String("printed_number_v1".to_string()),
                );
            }
            Err(err) => {
                answer.insert("value".to_string(), Value::Null);
                answer.insert("valid".to_string(), Value::Bool(false));
                answer.insert("confidence".to_string(), json!(0.0));
                answer.insert("span".to_string(), Value::Null);
                answer.insert(
                    "failure_reason".to_string(),
                    Value::String(
                        match err {
                            RowError::NotInSource => "not_in_source",
                            RowError::Invalid => "invalid_printed_number",
                        }
                        .to_string(),
                    ),
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
    fn normalize_parses_currency_signs_parentheses_units_and_bounds() {
        let spec = json!({"type": "number", "decimals": 2});
        assert_eq!(
            normalize(&json!("$1,234.50"), &spec).unwrap(),
            json!(1234.5)
        );
        assert_eq!(normalize(&json!("($45.25)"), &spec).unwrap(), json!(-45.25));
        assert_eq!(normalize(&json!("-€10.005"), &spec).unwrap(), json!(-10.01));
        assert_eq!(normalize(&json!("USD -99.1"), &spec).unwrap(), json!(-99.1));
        assert_eq!(normalize(&json!(null), &spec).unwrap(), json!(null));

        let with_unit = json!({"type": "number", "unit": "kg", "integer": true});
        assert_eq!(normalize(&json!("42 kg"), &with_unit).unwrap(), json!(42));

        // Reject invalid syntax / mismatched parens / double signs / out-of-bounds
        assert!(normalize(&json!("($45.25"), &spec).is_err());
        assert!(normalize(&json!("(-$45.25)"), &spec).is_err());
        assert!(normalize(&json!("+$ -10"), &spec).is_err());
        assert!(normalize(&json!(123), &spec).is_err());

        let bounded = json!({"type": "number", "minimum": 0});
        assert!(normalize(&json!("-5"), &bounded).is_err());
    }

    #[test]
    fn prepare_and_normalize_predictions_for_scalar_and_payment_rows() {
        let fields = json!({
            "total": {
                "type": "number",
                "decimals": 2,
                "source_only": true,
                "instructions": "Grand total."
            },
            "payments": {
                "type": "list",
                "instructions": "Extract all payments received. Exclude pending charges.",
                "items": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "text", "instructions": "Payment method."},
                        "amount": {"type": "number", "decimals": 2, "instructions": "Amount."}
                    }
                }
            },
            "payments.item1": {
                "type": "object",
                "instructions": "Extract all payments received. Exclude pending charges.",
                "properties": {
                    "method": {"type": "text", "instructions": "Payment method."},
                    "amount": {"type": "number", "decimals": 2, "instructions": "Amount."}
                }
            }
        });
        let (prepared, contracts) = prepare(fields.as_object().unwrap(), ["payments.item1"]);
        assert_eq!(prepared["total"]["type"], "text");
        assert!(
            prepared["total"]["instructions"]
                .as_str()
                .unwrap()
                .ends_with(COPY_INSTRUCTION)
        );
        assert!(
            prepared["payments"]["instructions"]
                .as_str()
                .unwrap()
                .contains(PAYMENT_PRESENCE_INSTRUCTION)
        );
        assert_eq!(
            prepared["payments.item1"]["properties"]["amount"]["type"],
            "text"
        );
        assert!(contracts.contains_key("total"));
        assert!(contracts.contains_key("payments.item1"));

        let record = json!({
            "input": "Total: $1,200.50\nVisa payment: ($50.00)",
            "fields": prepared,
            "_meta": {"printed_numbers": contracts}
        });
        let preds = json!({
            "total": {"valid": true, "value": "$1,200.50", "confidence": 0.9},
            "payments.item1": {"valid": true, "value": {"method": "Visa", "amount": "($50.00)"}, "confidence": 0.8}
        });
        let out = normalize_predictions(&record, preds.as_object().unwrap());
        assert_eq!(out["total"]["value"], json!(1200.5));
        assert_eq!(out["total"]["printed_value"], json!("$1,200.50"));
        assert_eq!(out["total"]["normalization_method"], "printed_number_v1");
        assert_eq!(
            out["payments.item1"]["value"],
            json!({"method": "Visa", "amount": -50.0})
        );

        // Value not present in input fails with not_in_source
        let bad_preds = json!({
            "total": {"valid": true, "value": "$999.00", "confidence": 0.9}
        });
        let bad_out = normalize_predictions(&record, bad_preds.as_object().unwrap());
        assert_eq!(bad_out["total"]["valid"], false);
        assert_eq!(bad_out["total"]["failure_reason"], "not_in_source");
    }
}
