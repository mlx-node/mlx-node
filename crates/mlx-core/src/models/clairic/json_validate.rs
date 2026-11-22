//! Flat Draft 2020-12 JSON Schema validator for Clairic AR schemas and contracts.
//!
//! Covers every JSON Schema construct produced by [`super::ar::schema`] and validated by
//! [`super::contracts::validate_public_fields`]: `type`, `enum`, `const`, `minLength`,
//! `maxLength`, `pattern`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`,
//! `properties`, `required`, `additionalProperties`, `items`, `minItems`, `maxItems`, `anyOf`.

use super::decimal_util;
use fancy_regex::Regex;
use napi::{Error, Result};
use serde_json::Value;
use std::collections::HashSet;

const VALID_TYPES: &[&str] = &[
    "string", "number", "integer", "boolean", "null", "object", "array",
];

/// Validate that `schema` is a well-formed JSON Schema (Draft 2020-12 subset),
/// matching `Draft202012Validator.check_schema(schema)`.
pub fn check_schema(schema: &Value) -> Result<()> {
    match schema {
        Value::Bool(_) => Ok(()),
        Value::Object(map) => {
            if let Some(ty) = map.get("type") {
                check_type_keyword(ty)?;
            }
            if let Some(en) = map.get("enum")
                && !en.is_array()
            {
                return Err(Error::from_reason("enum must be an array"));
            }
            for key in ["minLength", "maxLength", "minItems", "maxItems"] {
                if let Some(val) = map.get(key)
                    && !is_non_negative_integer(val)
                {
                    return Err(Error::from_reason(format!(
                        "{key} must be a non-negative integer"
                    )));
                }
            }
            if let Some(pat) = map.get("pattern") {
                let s = pat
                    .as_str()
                    .ok_or_else(|| Error::from_reason("pattern must be a string"))?;
                Regex::new(s).map_err(|e| {
                    Error::from_reason(format!("pattern {s:?} is not a valid regex: {e}"))
                })?;
            }
            for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
                if let Some(val) = map.get(key)
                    && !is_finite_number(val)
                {
                    return Err(Error::from_reason(format!("{key} must be a finite number")));
                }
            }
            if let Some(props) = map.get("properties") {
                let obj = props
                    .as_object()
                    .ok_or_else(|| Error::from_reason("properties must be an object"))?;
                for sub in obj.values() {
                    check_schema(sub)?;
                }
            }
            if let Some(req) = map.get("required") {
                let arr = req
                    .as_array()
                    .ok_or_else(|| Error::from_reason("required must be an array"))?;
                let mut seen = HashSet::with_capacity(arr.len());
                for item in arr {
                    let s = item
                        .as_str()
                        .ok_or_else(|| Error::from_reason("required entries must be strings"))?;
                    if !seen.insert(s) {
                        return Err(Error::from_reason(format!(
                            "duplicate entry {s:?} in required"
                        )));
                    }
                }
            }
            if let Some(ap) = map.get("additionalProperties") {
                check_schema(ap)?;
            }
            if let Some(items) = map.get("items") {
                check_schema(items)?;
            }
            if let Some(any_of) = map.get("anyOf") {
                let arr = any_of
                    .as_array()
                    .ok_or_else(|| Error::from_reason("anyOf must be an array"))?;
                if arr.is_empty() {
                    return Err(Error::from_reason("anyOf must be a non-empty array"));
                }
                for sub in arr {
                    check_schema(sub)?;
                }
            }
            Ok(())
        }
        _ => Err(Error::from_reason("schema must be an object or boolean")),
    }
}

/// Return `true` iff `value` satisfies the Draft 2020-12 JSON Schema `schema`.
pub fn is_valid(schema: &Value, value: &Value) -> bool {
    match schema {
        Value::Bool(b) => *b,
        Value::Object(map) => {
            if let Some(ty) = map.get("type")
                && !matches_type(ty, value)
            {
                return false;
            }
            if let Some(Value::Array(allowed)) = map.get("enum")
                && !allowed.iter().any(|candidate| json_equal(candidate, value))
            {
                return false;
            }
            if let Some(expected) = map.get("const")
                && !json_equal(expected, value)
            {
                return false;
            }
            if let Some(s) = value.as_str() {
                let char_len = s.chars().count();
                if let Some(min_len) = map.get("minLength").and_then(as_non_negative_usize)
                    && char_len < min_len
                {
                    return false;
                }
                if let Some(max_len) = map.get("maxLength").and_then(as_non_negative_usize)
                    && char_len > max_len
                {
                    return false;
                }
                if let Some(Value::String(pat)) = map.get("pattern") {
                    match Regex::new(pat) {
                        Ok(re) => {
                            if !re.is_match(s).unwrap_or(false) {
                                return false;
                            }
                        }
                        Err(_) => return false,
                    }
                }
            }
            if value.is_number()
                && let Some(val_dec) = decimal_util::from_json_value(value)
            {
                if let Some(bound) = map.get("minimum").and_then(decimal_util::from_json_value)
                    && val_dec < bound
                {
                    return false;
                }
                if let Some(bound) = map.get("maximum").and_then(decimal_util::from_json_value)
                    && val_dec > bound
                {
                    return false;
                }
                if let Some(bound) = map
                    .get("exclusiveMinimum")
                    .and_then(decimal_util::from_json_value)
                    && val_dec <= bound
                {
                    return false;
                }
                if let Some(bound) = map
                    .get("exclusiveMaximum")
                    .and_then(decimal_util::from_json_value)
                    && val_dec >= bound
                {
                    return false;
                }
            }
            if let Some(obj) = value.as_object() {
                if let Some(Value::Array(req)) = map.get("required") {
                    for key in req {
                        if let Some(k) = key.as_str()
                            && !obj.contains_key(k)
                        {
                            return false;
                        }
                    }
                }
                let props = map.get("properties").and_then(Value::as_object);
                if let Some(props_map) = props {
                    for (k, sub) in props_map {
                        if let Some(v) = obj.get(k)
                            && !is_valid(sub, v)
                        {
                            return false;
                        }
                    }
                }
                if let Some(ap) = map.get("additionalProperties") {
                    for (k, v) in obj {
                        let in_props = props.is_some_and(|p| p.contains_key(k));
                        if !in_props && !is_valid(ap, v) {
                            return false;
                        }
                    }
                }
            }
            if let Some(arr) = value.as_array() {
                if let Some(min_items) = map.get("minItems").and_then(as_non_negative_usize)
                    && arr.len() < min_items
                {
                    return false;
                }
                if let Some(max_items) = map.get("maxItems").and_then(as_non_negative_usize)
                    && arr.len() > max_items
                {
                    return false;
                }
                if let Some(items_schema) = map.get("items") {
                    for item in arr {
                        if !is_valid(items_schema, item) {
                            return false;
                        }
                    }
                }
            }
            if let Some(Value::Array(branches)) = map.get("anyOf")
                && !branches.iter().any(|sub| is_valid(sub, value))
            {
                return false;
            }
            true
        }
        _ => false,
    }
}

/// JSON Schema equality (`1 == 1.0`, booleans distinct from numbers, recursive for arrays/objects).
pub fn json_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Number(_), Value::Number(_)) => {
            match (
                decimal_util::from_json_value(left),
                decimal_util::from_json_value(right),
            ) {
                (Some(a), Some(b)) => a == b,
                _ => left == right,
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| json_equal(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|bv| json_equal(v, bv)))
        }
        _ => false,
    }
}

fn check_type_keyword(ty: &Value) -> Result<()> {
    match ty {
        Value::String(s) => {
            if VALID_TYPES.contains(&s.as_str()) {
                Ok(())
            } else {
                Err(Error::from_reason(format!("unknown schema type {s:?}")))
            }
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                return Err(Error::from_reason("type array must be non-empty"));
            }
            let mut seen = HashSet::with_capacity(arr.len());
            for item in arr {
                let s = item
                    .as_str()
                    .ok_or_else(|| Error::from_reason("type array entries must be strings"))?;
                if !VALID_TYPES.contains(&s) {
                    return Err(Error::from_reason(format!("unknown schema type {s:?}")));
                }
                if !seen.insert(s) {
                    return Err(Error::from_reason(format!(
                        "duplicate entry {s:?} in type array"
                    )));
                }
            }
            Ok(())
        }
        _ => Err(Error::from_reason(
            "type must be a string or an array of strings",
        )),
    }
}

fn matches_type(ty: &Value, value: &Value) -> bool {
    match ty {
        Value::String(s) => matches_single_type(s, value),
        Value::Array(arr) => arr
            .iter()
            .filter_map(Value::as_str)
            .any(|s| matches_single_type(s, value)),
        _ => false,
    }
}

fn matches_single_type(ty: &str, value: &Value) -> bool {
    match ty {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "number" => value.is_number(),
        "integer" => {
            if value.is_i64() || value.is_u64() {
                true
            } else if let Some(d) = value
                .is_number()
                .then(|| decimal_util::from_json_value(value))
                .flatten()
            {
                decimal_util::is_integral(&d)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_non_negative_integer(val: &Value) -> bool {
    if val.is_boolean() || !val.is_number() {
        return false;
    }
    if let Some(u) = val.as_u64() {
        let _ = u;
        return true;
    }
    if let Some(i) = val.as_i64() {
        return i >= 0;
    }
    if let Some(d) = decimal_util::from_json_value(val) {
        return decimal_util::is_integral(&d) && d >= 0;
    }
    false
}

fn as_non_negative_usize(val: &Value) -> Option<usize> {
    if val.is_boolean() || !val.is_number() {
        return None;
    }
    if let Some(u) = val.as_u64() {
        return usize::try_from(u).ok();
    }
    if let Some(i) = val.as_i64() {
        return usize::try_from(i).ok();
    }
    if let Some(f) = val.as_f64()
        && f.is_finite()
        && f >= 0.0
        && f.fract() == 0.0
    {
        return Some(f as usize);
    }
    None
}

fn is_finite_number(val: &Value) -> bool {
    if !val.is_number() {
        return false;
    }
    val.as_f64().is_some_and(f64::is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn check_schema_accepts_valid_and_rejects_bad_regex_and_bounds() {
        let good = json!({
            "type": ["string", "null"],
            "minLength": 1,
            "maxLength": 16,
            "pattern": "^[A-Z]{2}-\\d+$"
        });
        assert!(check_schema(&good).is_ok());

        let bad_regex = json!({
            "type": "string",
            "pattern": "[unclosed"
        });
        assert!(check_schema(&bad_regex).is_err());

        let bad_min_len = json!({
            "type": "string",
            "minLength": -1
        });
        assert!(check_schema(&bad_min_len).is_err());

        let bad_type = json!({
            "type": "float"
        });
        assert!(check_schema(&bad_type).is_err());

        let bad_min = json!({
            "type": "number",
            "minimum": "10"
        });
        assert!(check_schema(&bad_min).is_err());
    }

    #[test]
    fn validates_all_types_bounds_pattern_any_of_and_objects() {
        assert!(is_valid(&json!({"type": "integer"}), &json!(42)));
        assert!(is_valid(&json!({"type": "integer"}), &json!(42.0)));
        assert!(!is_valid(&json!({"type": "integer"}), &json!(42.5)));
        assert!(!is_valid(&json!({"type": "integer"}), &json!(true)));
        assert!(!is_valid(&json!({"type": "number"}), &json!(false)));

        let str_schema = json!({
            "type": ["string", "null"],
            "minLength": 2,
            "maxLength": 5,
            "pattern": "^[a-z]+$"
        });
        assert!(is_valid(&str_schema, &json!(null)));
        assert!(is_valid(&str_schema, &json!("abc")));
        assert!(!is_valid(&str_schema, &json!("a")));
        assert!(!is_valid(&str_schema, &json!("abcdef")));
        assert!(!is_valid(&str_schema, &json!("ABC")));

        // Unicode code point length check (not byte length)
        let unicode_len = json!({"type": "string", "minLength": 2, "maxLength": 2});
        assert!(is_valid(&unicode_len, &json!("éü")));
        assert!(!is_valid(&unicode_len, &json!("é")));

        let num_schema = json!({
            "type": "number",
            "minimum": 0,
            "exclusiveMaximum": 10
        });
        assert!(is_valid(&num_schema, &json!(0)));
        assert!(is_valid(&num_schema, &json!(9.999)));
        assert!(!is_valid(&num_schema, &json!(10)));
        assert!(!is_valid(&num_schema, &json!(-0.1)));

        let obj_schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "qty": {"type": ["integer", "null"]}
            },
            "required": ["name", "qty"],
            "additionalProperties": false
        });
        assert!(is_valid(&obj_schema, &json!({"name": "widget", "qty": 3})));
        assert!(is_valid(
            &obj_schema,
            &json!({"name": "widget", "qty": null})
        ));
        assert!(!is_valid(&obj_schema, &json!({"name": "widget"})));
        assert!(!is_valid(
            &obj_schema,
            &json!({"name": "widget", "qty": 3, "extra": 1})
        ));

        let arr_schema = json!({
            "type": ["array", "null"],
            "minItems": 1,
            "maxItems": 2,
            "items": {"type": "string"}
        });
        assert!(is_valid(&arr_schema, &json!(null)));
        assert!(is_valid(&arr_schema, &json!(["a"])));
        assert!(!is_valid(&arr_schema, &json!([])));
        assert!(!is_valid(&arr_schema, &json!(["a", "b", "c"])));
        assert!(!is_valid(&arr_schema, &json!([1])));

        let any_of = json!({
            "anyOf": [
                {"type": "string", "pattern": "^foo"},
                {"type": "integer", "minimum": 10}
            ]
        });
        assert!(is_valid(&any_of, &json!("foobar")));
        assert!(is_valid(&any_of, &json!(12)));
        assert!(!is_valid(&any_of, &json!(5)));
        assert!(!is_valid(&any_of, &json!("bar")));

        let enum_schema = json!({"enum": [1, "a", null]});
        assert!(is_valid(&enum_schema, &json!(1.0)));
        assert!(is_valid(&enum_schema, &json!("a")));
        assert!(is_valid(&enum_schema, &json!(null)));
        assert!(!is_valid(&enum_schema, &json!(true)));
    }
}
