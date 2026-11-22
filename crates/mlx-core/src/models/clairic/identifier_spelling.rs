//! Codepoint-spaced purchase-order reference wire encoding, decoding, validation,
//! and fallback helpers.
//!
//! Direct port of upstream `clairic_runtime/identifier_spelling.py`.

use super::{ar, errors::invalid_request, json_validate, scopes};
use napi::Result;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

pub const VERSION: &str = "identifier-codepoint-spelling-v1";
pub const PDF_VIEW_ID: &str = "pdf-po-codepoint-spelling-focus-v1";
pub const MANY_IMAGE_VIEW_ID: &str = "po-three-four-images-detail3-800000px-v2";
pub const MANY_IMAGE_MAX_PIXELS: usize = 800_000;
pub const INSTRUCTION: &str = "Spell the printed purchase order reference with one space between each individual character. Preserve the actual characters; do not normalize letters and digits.";

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

/// Decode a space-separated codepoint wire string (`"P O - 1 2"` -> `"PO-12"`)
/// (`identifier_spelling.py::decode_wire`).
pub fn decode_wire(value: &str) -> Result<String> {
    let chars: Vec<char> = value.chars().collect();
    if !chars.is_empty()
        && (chars.len().is_multiple_of(2) || chars.iter().skip(1).step_by(2).any(|&c| c != ' '))
    {
        return Err(invalid_request("Invalid identifier spelling separators"));
    }
    Ok(chars.into_iter().step_by(2).collect())
}

/// Decode a JSON wire value (`identifier_spelling.py::decode_wire`).
pub fn decode_wire_value(value: &Value) -> Result<String> {
    let Some(s) = value.as_str() else {
        return Err(invalid_request("Identifier spelling wire must be a string"));
    };
    decode_wire(s)
}

/// Rewrite eligible purchase-order reference fields in `record` to use the codepoint-spelling
/// wire format (`identifier_spelling.py::prepare`).
pub fn prepare(
    record: &mut Value,
    public_fields: &Map<String, Value>,
    query_labels: &BTreeMap<String, String>,
    has_media: bool,
) {
    if !has_media || query_labels.is_empty() {
        return;
    }
    let (_, eligible) = scopes::identifier::prepare_fields(public_fields, true);
    let mut contracts: Map<String, Value> = Map::new();

    for (name, label) in query_labels {
        if label != "purchase_order_reference"
            || eligible.get(name).map(String::as_str) != Some(label.as_str())
        {
            continue;
        }
        let Some(public_spec) = public_fields.get(name) else {
            continue;
        };
        let Some(spec_obj) = record
            .get("fields")
            .and_then(|f| f.get(name))
            .and_then(Value::as_object)
            .cloned()
        else {
            continue;
        };

        let mut wire = Map::new();
        for (k, v) in &spec_obj {
            if !matches!(k.as_str(), "minLength" | "maxLength" | "pattern") {
                wire.insert(k.clone(), v.clone());
            }
        }
        let orig_inst = spec_obj
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        wire.insert(
            "instructions".to_string(),
            Value::String(format!("{INSTRUCTION}\n{orig_inst}")),
        );

        if let Some(fields_mut) = record.get_mut("fields").and_then(Value::as_object_mut) {
            fields_mut.insert(name.clone(), Value::Object(wire));
        }
        contracts.insert(
            name.clone(),
            json!({
                "scalar": public_spec.clone(),
                "wire": VERSION,
                "query_name": name,
            }),
        );
    }

    if !contracts.is_empty()
        && let Some(rec_obj) = record.as_object_mut()
    {
        let meta = rec_obj
            .entry("_meta".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(meta_obj) = meta.as_object_mut() {
            meta_obj.insert("identifier_spelling".to_string(), Value::Object(contracts));
        }
    }
}

/// Decode and validate codepoint-spelled identifier predictions
/// (`identifier_spelling.py::normalize_predictions`).
pub fn normalize_predictions(
    record: &Value,
    predictions: &Map<String, Value>,
) -> Map<String, Value> {
    let mut result = predictions.clone();
    let Some(contracts) = record
        .get("_meta")
        .and_then(|m| m.get("identifier_spelling"))
        .and_then(Value::as_object)
    else {
        return result;
    };

    for (name, contract) in contracts {
        let Some(orig) = result.get(name).and_then(Value::as_object) else {
            continue;
        };
        if !is_truthy(orig.get("valid"))
            || !is_truthy(orig.get("complete"))
            || orig.get("wire_decoder").and_then(Value::as_str) == Some(VERSION)
        {
            continue;
        }
        let mut answer = orig.clone();
        let wire = orig.get("value").cloned().unwrap_or(Value::Null);
        answer.insert("wire_value".to_string(), wire.clone());
        answer.insert(
            "wire_decoder".to_string(),
            Value::String(VERSION.to_string()),
        );
        answer.insert(
            "native_decoder".to_string(),
            orig.get("decoder").cloned().unwrap_or(Value::Null),
        );
        answer.insert(
            "decoder".to_string(),
            Value::String(format!("{}+{VERSION}", ar::VERSION)),
        );
        answer.insert(
            "score_representation".to_string(),
            Value::String("native_json_codepoint_spelling_wire".to_string()),
        );
        if let Some(span) = answer.get("span")
            && !span.is_null()
        {
            answer.insert("native_wire_span".to_string(), span.clone());
        }
        if let Some(span_method) = answer.remove("span_method") {
            answer.insert("native_wire_span_method".to_string(), span_method);
        }
        answer.insert("span".to_string(), Value::Null);

        let decoded_res = (|| -> Result<Value> {
            let decoded = if wire.is_null() {
                Value::Null
            } else {
                Value::String(decode_wire_value(&wire)?)
            };
            let scalar_spec = contract
                .get("scalar")
                .ok_or_else(|| invalid_request("missing identifier spelling scalar contract"))?;
            let schema = ar::schema(scalar_spec)?;
            if !json_validate::is_valid(&schema, &decoded) {
                return Err(invalid_request(
                    "decoded identifier failed schema validation",
                ));
            }
            Ok(decoded)
        })();

        match decoded_res {
            Ok(decoded) => {
                answer.insert("value".to_string(), decoded);
            }
            Err(_) => {
                answer.insert("value".to_string(), Value::Null);
                answer.insert("valid".to_string(), Value::Bool(false));
                answer.insert(
                    "failure_reason".to_string(),
                    Value::String("invalid_identifier_spelling".to_string()),
                );
            }
        }
        result.insert(name.clone(), Value::Object(answer));
    }

    result
}

/// Select active field names participating in identifier spelling (`VERSION` wire check optional).
pub fn selected_spelling_fields(
    record: &Value,
    active_task_names: &HashSet<String>,
    require_wire_version: bool,
) -> Vec<String> {
    let Some(contracts) = record
        .get("_meta")
        .and_then(|m| m.get("identifier_spelling"))
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };
    contracts
        .iter()
        .filter(|(name, contract)| {
            active_task_names.contains(name.as_str())
                && (!require_wire_version
                    || contract.get("wire").and_then(Value::as_str) == Some(VERSION))
        })
        .map(|(name, _)| name.clone())
        .collect()
}

/// Metadata payload recorded on `record["_meta"]["identifier_spelling_pdf_view"]`
/// (`identifier_spelling.py::prepare_pdf_tasks`).
pub fn pdf_view_metadata(fields: &[String], image_count: usize, pdf_page_count: usize) -> Value {
    json!({
        "version": PDF_VIEW_ID,
        "fields": fields,
        "image_count": image_count,
        "pdf_page_count": pdf_page_count,
    })
}

/// Metadata payload recorded on `record["_meta"]["identifier_spelling_image_view"]`
/// (`identifier_spelling.py::prepare_many_image_tasks`).
pub fn many_image_view_metadata(fields: &[String], image_count: usize) -> Value {
    json!({
        "version": MANY_IMAGE_VIEW_ID,
        "fields": fields,
        "image_count": image_count,
        "max_scale": 3.0,
        "max_pixels": MANY_IMAGE_MAX_PIXELS,
    })
}

/// Return field names whose predictions failed with `invalid_identifier_spelling`.
pub fn failed_spelling_fields(predictions: &Map<String, Value>) -> HashSet<String> {
    predictions
        .iter()
        .filter(|(_, pred)| {
            pred.get("failure_reason").and_then(Value::as_str)
                == Some("invalid_identifier_spelling")
        })
        .map(|(name, _)| name.clone())
        .collect()
}

/// Filter a JSON-encoded fallback record's `_ar_tasks` down to fields that failed spelling
/// (`identifier_spelling.py::fallback_record`).
pub fn fallback_record(record: &Value, predictions: &Map<String, Value>) -> Option<Value> {
    let selected = failed_spelling_fields(predictions);
    if selected.is_empty() {
        return None;
    }
    let fallback = record.get("_identifier_spelling_fallback")?.as_object()?;
    let all_tasks = fallback.get("_ar_tasks")?.as_array()?;
    let tasks: Vec<Value> = all_tasks
        .iter()
        .filter(|task| {
            task.get(0)
                .or_else(|| task.get("name"))
                .and_then(Value::as_str)
                .is_some_and(|name| selected.contains(name))
        })
        .cloned()
        .collect();
    if tasks.is_empty() {
        return None;
    }
    let mut out = fallback.clone();
    out.insert("_ar_tasks".to_string(), Value::Array(tasks));
    out.insert("_ar_prefix".to_string(), Value::Array(Vec::new()));
    out.insert("_ar_group_prefixes".to_string(), Value::Array(Vec::new()));
    out.insert("_ar_media_groups".to_string(), Value::Array(Vec::new()));
    out.insert("_null_recovery".to_string(), Value::Bool(true));
    Some(Value::Object(out))
}

/// Merge native-scalar fallback predictions into `predictions`, updating token counts and
/// recording the prior spelling attempt (`identifier_spelling.py::merge_fallback`).
pub fn merge_fallback(
    predictions: &Map<String, Value>,
    alternatives: &Map<String, Value>,
    fallback_input_lengths: &HashMap<String, usize>,
) -> Map<String, Value> {
    let mut result = predictions.clone();
    for (name, alt_val) in alternatives {
        let Some(alt_obj) = alt_val.as_object() else {
            continue;
        };
        let Some(orig_val) = predictions.get(name) else {
            continue;
        };
        let orig_tokens = orig_val.get("tokens").and_then(Value::as_u64).unwrap_or(0);
        let alt_tokens = alt_obj.get("tokens").and_then(Value::as_u64).unwrap_or(0);
        let orig_retry_in = orig_val
            .get("retry_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let fallback_len = fallback_input_lengths.get(name).copied().unwrap_or(0) as u64;

        let mut merged = alt_obj.clone();
        merged.insert(
            "identifier_spelling_fallback".to_string(),
            Value::Bool(true),
        );
        merged.insert("identifier_spelling_attempt".to_string(), orig_val.clone());
        merged.insert(
            "score_representation".to_string(),
            Value::String("native_json_scalar_fallback".to_string()),
        );
        merged.insert("tokens".to_string(), json!(orig_tokens + alt_tokens));
        merged.insert(
            "retry_input_tokens".to_string(),
            json!(orig_retry_in + fallback_len),
        );
        result.insert(name.clone(), Value::Object(merged));
    }
    result
}

/// Convenience variant of [`merge_fallback`] extracting task input lengths from `fallback["_ar_tasks"]`.
pub fn merge_fallback_from_record(
    predictions: &Map<String, Value>,
    alternatives: &Map<String, Value>,
    fallback: &Value,
) -> Map<String, Value> {
    let mut lengths: HashMap<String, usize> = HashMap::new();
    if let Some(tasks) = fallback.get("_ar_tasks").and_then(Value::as_array) {
        for task in tasks {
            let name = task
                .get(0)
                .or_else(|| task.get("name"))
                .and_then(Value::as_str);
            let ids_len = task
                .get(1)
                .or_else(|| task.get("input_ids"))
                .and_then(Value::as_array)
                .map(Vec::len);
            if let (Some(n), Some(len)) = (name, ids_len) {
                lengths.insert(n.to_string(), len);
            }
        }
    }
    merge_fallback(predictions, alternatives, &lengths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decode_wire_valid_and_invalid() {
        assert_eq!(decode_wire("").unwrap(), "");
        assert_eq!(decode_wire("A").unwrap(), "A");
        assert_eq!(decode_wire("P O - 4 2 0 9").unwrap(), "PO-4209");
        assert_eq!(decode_wire("é   x").unwrap(), "é x");
        assert!(decode_wire("PO").is_err());
        assert!(decode_wire("P-O").is_err());
        assert!(decode_wire("P O ").is_err());
        assert!(decode_wire_value(&json!(123)).is_err());
    }

    #[test]
    fn prepare_normalize_and_fallback_flow() {
        let public_fields = json!({
            "reference": {
                "type": "text",
                "pattern": "^PO-[0-9]+$",
                "minLength": 4,
                "instructions": "Extract the printed purchase order reference."
            }
        });
        let public_map = public_fields.as_object().unwrap();
        let (scoped, labels) = scopes::identifier::prepare_fields(public_map, true);
        let mut record = json!({
            "fields": scoped,
            "_meta": {}
        });
        prepare(&mut record, public_map, &labels, true);

        // Wire spec stripped pattern/minLength and prepended INSTRUCTION
        assert!(record["fields"]["reference"].get("pattern").is_none());
        assert!(record["fields"]["reference"].get("minLength").is_none());
        assert!(
            record["fields"]["reference"]["instructions"]
                .as_str()
                .unwrap()
                .starts_with(INSTRUCTION)
        );
        assert_eq!(
            record["_meta"]["identifier_spelling"]["reference"]["wire"],
            VERSION
        );

        // Valid wire decodes and passes original pattern
        let good_preds = json!({
            "reference": {
                "valid": true,
                "complete": true,
                "value": "P O - 9 8 7",
                "decoder": ar::VERSION,
                "span": [0, 11],
                "span_method": "exact",
                "tokens": 6
            }
        });
        let norm_good = normalize_predictions(&record, good_preds.as_object().unwrap());
        assert_eq!(norm_good["reference"]["valid"], true);
        assert_eq!(norm_good["reference"]["value"], "PO-987");
        assert_eq!(norm_good["reference"]["wire_value"], "P O - 9 8 7");
        assert_eq!(norm_good["reference"]["native_wire_span"], json!([0, 11]));
        assert!(norm_good["reference"]["span"].is_null());

        // Invalid pattern fails with invalid_identifier_spelling and triggers fallback
        let bad_preds = json!({
            "reference": {
                "valid": true,
                "complete": true,
                "value": "I N V - 1",
                "decoder": ar::VERSION,
                "tokens": 5
            }
        });
        let norm_bad = normalize_predictions(&record, bad_preds.as_object().unwrap());
        assert_eq!(norm_bad["reference"]["valid"], false);
        assert_eq!(
            norm_bad["reference"]["failure_reason"],
            "invalid_identifier_spelling"
        );

        record["_identifier_spelling_fallback"] = json!({
            "_ar_tasks": [["reference", [1, 2, 3, 4], null]]
        });
        let fb = fallback_record(&record, &norm_bad).unwrap();
        assert_eq!(fb["_null_recovery"], true);
        assert_eq!(fb["_ar_tasks"].as_array().unwrap().len(), 1);

        let alt = json!({
            "reference": {
                "valid": true,
                "complete": true,
                "value": "PO-987",
                "tokens": 3
            }
        });
        let merged = merge_fallback_from_record(&norm_bad, alt.as_object().unwrap(), &fb);
        assert_eq!(merged["reference"]["value"], "PO-987");
        assert_eq!(merged["reference"]["identifier_spelling_fallback"], true);
        assert_eq!(merged["reference"]["tokens"], 8);
        assert_eq!(merged["reference"]["retry_input_tokens"], 4);
    }
}
