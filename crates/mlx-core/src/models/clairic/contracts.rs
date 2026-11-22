//! Public request contract validation for all 13 Clairic field types.
//!
//! Direct port of upstream `clairic_runtime/contracts.py`.

use super::{ar, errors::invalid_request, json_validate, schema};
use napi::Result;
use serde_json::{Map, Value, json};
use std::collections::HashSet;

pub const COMMON_OPTIONS: &[&str] = &["type", "instructions", "nullable"];
/// Maximum per-list query expansion budget allowed by Clairic.
pub const MAX_LIST_EXPANSION_BUDGET: usize = 512;
pub const NUMERIC_OPTIONS: &[&str] = &["integer", "decimals", "unit"];
pub const GRAMMAR_NUMBER_OPTIONS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "enum",
];

/// Return the allowed type-specific option keys (excluding `COMMON_OPTIONS`) for a public field type.
pub fn allowed_type_options(kind: &str) -> Option<&'static [&'static str]> {
    match kind {
        "text" => Some(&["knowledge", "minLength", "maxLength", "pattern", "enum"]),
        "number" => Some(&[
            "integer",
            "decimals",
            "unit",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "enum",
            "knowledge",
            "source_only",
        ]),
        "compute" => Some(&["integer", "decimals", "unit", "null_threshold"]),
        "date" | "time" | "datetime" => Some(&["null_threshold", "source_only"]),
        "duration" => Some(&["integer", "decimals", "unit", "null_threshold"]),
        "boolean" => Some(&[]),
        "enum" => Some(&["enum"]),
        "object" => Some(&["properties"]),
        "list" => Some(&[
            "items",
            "max_items",
            "knowledge",
            "source_only",
            "integer",
            "decimals",
            "unit",
            "enum",
        ]),
        "estimate" => Some(&["min", "max", "scale", "step", "unit"]),
        "forecast" => Some(&["min", "max", "scale", "step", "unit", "horizon", "interval"]),
        _ => None,
    }
}

/// Normalize `spec["items"]` into a field spec object (`contracts.py::item_spec`).
pub fn item_spec(spec: &Value) -> Value {
    match spec.get("items") {
        Some(Value::Object(obj)) => Value::Object(obj.clone()),
        Some(other) => json!({"type": other}),
        None => json!({"type": null}),
    }
}

/// Validate a public request's field declarations against all 13 public field types
/// (`contracts.py::validate_public_fields`).
pub fn validate_public_fields(fields: &Map<String, Value>) -> Result<()> {
    let mut checked_scalar_schemas: HashSet<Vec<String>> = HashSet::new();
    validate_public_fields_inner(fields, &mut checked_scalar_schemas)
}

/// Full public request validation: validates public contract options, expands forecasts,
/// validates normalized schema fields, and validates optional `reference_date`.
pub fn validate_public_request(
    fields: &Value,
    reference_date: Option<&Value>,
) -> Result<(Map<String, Value>, Vec<(String, Vec<String>)>)> {
    let Some(map) = fields.as_object() else {
        return Err(invalid_request("record must declare at least one field"));
    };
    if map.is_empty() {
        return Err(invalid_request("record must declare at least one field"));
    }
    validate_public_fields(map)?;
    let (expanded, groups) = schema::expand_forecasts_grouped(map)?;
    schema::validate_fields(&Value::Object(expanded.clone()))?;
    schema::validate_reference_date(reference_date)?;
    Ok((expanded, groups))
}

fn validate_public_fields_inner(
    fields: &Map<String, Value>,
    checked_scalar_schemas: &mut HashSet<Vec<String>>,
) -> Result<()> {
    for (name, spec) in fields {
        let spec_obj = spec.as_object();
        let kind = spec_obj.and_then(|o| o.get("type")).and_then(Value::as_str);
        let Some(kind) = kind else {
            return Err(invalid_request(format!("field {name}: unknown field type")));
        };
        let Some(allowed_specific) = allowed_type_options(kind) else {
            return Err(invalid_request(format!("field {name}: unknown field type")));
        };
        let obj =
            spec_obj.ok_or_else(|| invalid_request(format!("field {name}: unknown field type")))?;

        let mut unknown: Vec<&str> = obj
            .keys()
            .map(String::as_str)
            .filter(|k| !COMMON_OPTIONS.contains(k) && !allowed_specific.contains(k))
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(invalid_request(format!(
                "field {name}: unknown options {unknown:?}"
            )));
        }

        if let Some(inst) = obj.get("instructions")
            && !inst.is_string()
        {
            return Err(invalid_request(format!(
                "field {name}: instructions must be a string"
            )));
        }
        if let Some(nullable) = obj.get("nullable")
            && !nullable.is_boolean()
        {
            return Err(invalid_request(format!(
                "field {name}: nullable must be true or false"
            )));
        }

        if let Some(source_only_val) = obj.get("source_only") {
            let Some(source_only) = source_only_val.as_bool() else {
                return Err(invalid_request(format!(
                    "field {name}: source_only must be true or false"
                )));
            };
            if kind == "list"
                && !matches!(
                    obj.get("items").and_then(Value::as_str),
                    Some("date" | "number")
                )
            {
                return Err(invalid_request(format!(
                    "field {name}: source_only lists require date or number items"
                )));
            }
            if source_only && obj.contains_key("null_threshold") {
                return Err(invalid_request(format!(
                    "field {name}: source_only does not support typed null_threshold"
                )));
            }
            if source_only
                && obj
                    .get("knowledge")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            {
                return Err(invalid_request(format!(
                    "field {name}: source_only cannot permit knowledge answers"
                )));
            }
        }

        if let Some(en) = obj.get("enum")
            && !en.as_array().is_some_and(|a| !a.is_empty())
        {
            return Err(invalid_request(format!(
                "field {name}: enum must be a nonempty array"
            )));
        }
        if kind == "enum" && !obj.contains_key("enum") {
            return Err(invalid_request(format!(
                "field {name}: enum values are required"
            )));
        }

        if kind == "object" {
            let Some(properties) = obj.get("properties").and_then(Value::as_object) else {
                return Err(invalid_request(format!(
                    "field {name}: object properties are required"
                )));
            };
            if properties.is_empty() {
                return Err(invalid_request(format!(
                    "field {name}: object properties are required"
                )));
            }
            for child in properties.values() {
                let valid_child = child
                    .as_object()
                    .and_then(|c| c.get("type"))
                    .and_then(Value::as_str)
                    .is_some_and(|t| matches!(t, "text" | "number" | "boolean" | "enum"));
                if !valid_child {
                    return Err(invalid_request(format!(
                        "field {name}: row properties must be text, number, boolean or enum"
                    )));
                }
            }
            if properties.values().any(|child| {
                child
                    .get("source_only")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            }) {
                return Err(invalid_request(format!(
                    "field {name}: source_only is supported on scalar numbers and number lists, not row properties"
                )));
            }
            validate_public_fields_inner(properties, checked_scalar_schemas)?;
            schema::validate_fields(&Value::Object(properties.clone()))?;
        }

        if kind == "list" {
            if !obj.contains_key("items") {
                return Err(invalid_request(format!("list {name}: items are required")));
            }
            let child = item_spec(spec);
            let child_type = child.get("type").and_then(Value::as_str);
            if obj.get("items").is_some_and(Value::is_object) && child_type != Some("object") {
                return Err(invalid_request(format!(
                    "list {name}: structured items must be objects"
                )));
            }
            if child_type == Some("object") {
                let mut wrapper = Map::new();
                wrapper.insert(format!("{name}[]"), child);
                validate_public_fields_inner(&wrapper, checked_scalar_schemas)?;
                let forbidden_on_row_list = obj.keys().any(|k| {
                    NUMERIC_OPTIONS.contains(&k.as_str()) || k == "knowledge" || k == "enum"
                });
                if forbidden_on_row_list {
                    return Err(invalid_request(format!(
                        "list {name}: put numeric or knowledge options on row properties"
                    )));
                }
            }
            if let Some(maximum) = obj.get("max_items") {
                let parsed = if maximum.is_boolean() {
                    None
                } else {
                    maximum.as_i64().filter(|&v| v >= 0)
                };
                let Some(val) = parsed else {
                    return Err(invalid_request(format!(
                        "list {name}: max_items must be a nonnegative integer"
                    )));
                };
                if val as u64 > MAX_LIST_EXPANSION_BUDGET as u64 {
                    return Err(invalid_request(format!(
                        "list {name}: max_items ({val}) exceeds maximum allowed ({MAX_LIST_EXPANSION_BUDGET})"
                    )));
                }
            }
        }

        if matches!(kind, "text" | "number" | "boolean" | "enum" | "object") {
            let built = ar::schema(spec)?;
            let scalar_key = built.as_object().and_then(|s_obj| {
                if s_obj.len() == 1 && s_obj.contains_key("type") {
                    match s_obj.get("type") {
                        Some(Value::String(s)) => Some(vec![s.clone()]),
                        Some(Value::Array(arr)) => Some(
                            arr.iter()
                                .filter_map(|v| v.as_str().map(str::to_string))
                                .collect(),
                        ),
                        _ => None,
                    }
                } else {
                    None
                }
            });
            let needs_check = scalar_key
                .as_ref()
                .is_none_or(|k| !checked_scalar_schemas.contains(k));
            if needs_check {
                json_validate::check_schema(&built).map_err(|err| {
                    invalid_request(format!(
                        "field {name}: invalid schema constraint: {}",
                        err.reason
                    ))
                })?;
                if let Some(k) = scalar_key {
                    checked_scalar_schemas.insert(k);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_all_thirteen_public_field_types() {
        let fields = json!({
            "t": {"type": "text", "minLength": 1, "pattern": "^[A-Za-z]+$", "knowledge": false},
            "n": {"type": "number", "decimals": 2, "minimum": 0, "source_only": true},
            "c": {"type": "compute", "integer": true, "null_threshold": 0.4},
            "d": {"type": "date", "source_only": false, "null_threshold": 0.5},
            "tm": {"type": "time", "source_only": true},
            "dt": {"type": "datetime"},
            "dur": {"type": "duration", "unit": "hours", "decimals": 1},
            "b": {"type": "boolean", "nullable": false},
            "e": {"type": "enum", "enum": ["open", "closed"]},
            "o": {
                "type": "object",
                "properties": {
                    "label": {"type": "text"},
                    "amount": {"type": "number", "decimals": 2},
                    "paid": {"type": "boolean"},
                    "status": {"type": "enum", "enum": ["ok", "err"]}
                }
            },
            "l_scalar": {"type": "list", "items": "number", "source_only": true, "max_items": 10},
            "l_rows": {
                "type": "list",
                "items": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "text"},
                        "qty": {"type": "number", "integer": true}
                    }
                },
                "max_items": 16
            },
            "est": {"type": "estimate", "min": 0, "max": 100, "step": 5, "scale": "linear"},
            "fc": {"type": "forecast", "min": 1, "max": 1000, "scale": "log", "horizon": 4, "interval": "quarter"}
        });
        let (expanded, groups) =
            validate_public_request(&fields, Some(&json!("2026-01-15"))).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "fc");
        assert_eq!(groups[0].1.len(), 4);
        assert!(expanded.contains_key("fc.1"));
        assert!(expanded.contains_key("fc.4"));
        assert!(!expanded.contains_key("fc"));
    }

    #[test]
    fn rejects_invalid_options_and_schema_constraints() {
        let bad_unknown = json!({"a": {"type": "boolean", "minimum": 1}});
        assert!(validate_public_request(&bad_unknown, None).is_err());

        let bad_regex = json!({"a": {"type": "text", "pattern": "(unclosed"}});
        assert!(validate_public_request(&bad_regex, None).is_err());

        let bad_source_only_list =
            json!({"a": {"type": "list", "items": "text", "source_only": true}});
        assert!(validate_public_request(&bad_source_only_list, None).is_err());

        let bad_source_only_knowledge =
            json!({"a": {"type": "number", "source_only": true, "knowledge": true}});
        assert!(validate_public_request(&bad_source_only_knowledge, None).is_err());

        let bad_source_only_row_prop = json!({
            "o": {
                "type": "object",
                "properties": {"x": {"type": "number", "source_only": true}}
            }
        });
        assert!(validate_public_request(&bad_source_only_row_prop, None).is_err());

        let bad_compute = json!({"c": {"type": "compute", "nullable": false}});
        assert!(validate_public_request(&bad_compute, None).is_err());

        let bad_ref_date = json!({"a": {"type": "text"}});
        assert!(validate_public_request(&bad_ref_date, Some(&json!("not-a-date"))).is_err());
    }
}
