//! Autoregressive (Path A) field schema, numeric normalization, prompt instructions,
//! and typed-record splitting.
//!
//! Direct port of upstream `clairic_runtime/ar.py` (including the `image_rows`
//! inventory/proposal schema & instruction helpers invoked by `ar.py`).

use super::decimal_util;
use super::errors::invalid_request;
use super::schema::MAX_LIST_ITEMS;
use napi::Result;
use serde_json::{Map, Value, json};
use std::collections::HashSet;

pub const VERSION: &str = "parallel-ar-json-scalar-v1";
pub const EOS: &str = "<|im_end|>";
pub const EXTRACTION_INSTRUCTION: &str = "Extract the requested value from the supplied input, following the field instructions and scope.";
pub const MEDIA_EVIDENCE_INSTRUCTION: &str = "The attached images are the document pages and contain extraction evidence. Use all attached pages together with the accompanying text, following the field instructions and requested scope.";
pub const COMPUTATION_INSTRUCTION: &str = "Answer the requested field using the supplied input and field instructions. Perform any required arithmetic, reasoning, normalization, or unit conversion. A computable answer is available even when it is not written literally in the source.";
pub const KNOWLEDGE_INSTRUCTION: &str = "Answer the requested field using reliable prior knowledge and any relevant supplied input. This field explicitly permits knowledge: true. The answer does not need to appear in the document. Missing source evidence does not make a known factual answer unavailable. Return null only when neither reliable prior knowledge nor the input determines the answer. Follow the field instructions and scope.";
pub const RECOVERY_INSTRUCTION: &str = "Answer the requested field using the supplied input and field instructions. Copy the matching source value exactly. Check labels, surrounding context and requested entity. Do not calculate, convert units, complete a name, or substitute a related field. Return the value if it is explicitly present; return null if it is absent.";
pub const BOOLEAN_INSTRUCTION: &str = " This field is a Boolean statement about the input. Evaluate its condition using the supplied evidence: return true when the condition is affirmed, false when the condition is explicitly negated, and null only when evidence does not determine whether it holds. The literal words true and false do not need to be printed.";

const IMAGE_ROW_LABEL_INSTRUCTIONS: &str = "Brief exact printed identifier of this requested row, without secondary description or numeric cells. Follow the requested field scope: the label may name a product, service, tax, fee, payment or other row.";

/// Return `true` iff `spec` is marked as an image-row inventory list (`image_rows.is_inventory`).
pub fn is_inventory(spec: &Value) -> bool {
    spec.get("_image_row_inventory")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// JSON Schema for an image-row inventory list (`image_rows.inventory_schema`).
pub fn inventory_schema(spec: &Value) -> Value {
    let max_items = spec
        .get("max_items")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(MAX_LIST_ITEMS)
        + 1;
    json!({
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": max_items,
        "items": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": false
        }
    })
}

/// Return `true` iff `spec` is handled by the AR (Path A) decoder (`ar.py::names`).
pub fn is_ar_field(spec: &Value) -> bool {
    matches!(
        spec.get("type").and_then(Value::as_str),
        Some("text" | "number" | "boolean" | "enum" | "object")
    ) || is_inventory(spec)
}

/// Normalize a numeric answer according to `spec`'s `integer` and `decimals` constraints,
/// using exact `ROUND_HALF_UP` decimal quantization (`ar.py::normalize_number`).
pub fn normalize_number(value: &Value, spec: &Value) -> Result<Value> {
    if value.is_null() || spec.get("type").and_then(Value::as_str) != Some("number") {
        return Ok(value.clone());
    }
    if value.is_boolean() {
        return Err(invalid_request("A boolean is not a numeric answer"));
    }
    let decimals: Option<i64> = if spec
        .get("integer")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        Some(0)
    } else {
        spec.get("decimals").and_then(Value::as_i64)
    };

    let Some(dec) = decimal_util::from_json_value(value) else {
        return Err(invalid_request("Nonfinite number"));
    };

    match decimals {
        None => {
            if let Some(f) = value.as_f64()
                && !f.is_finite()
            {
                return Err(invalid_request("Nonfinite number"));
            }
            decimal_util::to_json_number(&dec).ok_or_else(|| invalid_request("Nonfinite number"))
        }
        Some(d) => {
            let rounded = decimal_util::round_half_up_checked(&dec, d).ok_or_else(|| {
                invalid_request("Cannot represent numeric answer at the requested precision")
            })?;
            if d == 0 {
                decimal_util::to_decoded_json_number(&rounded, true).ok_or_else(|| {
                    invalid_request("Cannot represent numeric answer at the requested precision")
                })
            } else {
                let f = rounded
                    .to_string()
                    .parse::<f64>()
                    .ok()
                    .filter(|f| f.is_finite())
                    .ok_or_else(|| invalid_request("Nonfinite number"))?;
                let clean = if f == 0.0 { 0.0 } else { f };
                serde_json::Number::from_f64(clean)
                    .map(Value::Number)
                    .ok_or_else(|| invalid_request("Nonfinite number"))
            }
        }
    }
}

/// Return the field names in `record["fields"]` routed to Path A (AR), preserving order.
pub fn names(record: &Value) -> Vec<String> {
    let Some(fields) = record.get("fields").and_then(Value::as_object) else {
        return Vec::new();
    };
    fields
        .iter()
        .filter(|(_, spec)| is_ar_field(spec))
        .map(|(name, _)| name.clone())
        .collect()
}

/// Build the Draft 2020-12 JSON Schema for an AR field spec (`ar.py::schema`).
pub fn schema(spec: &Value) -> Result<Value> {
    if is_inventory(spec) {
        return Ok(inventory_schema(spec));
    }
    let kind = spec
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_request("field spec missing type"))?;
    let nullable = spec
        .get("nullable")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    if kind == "object" {
        let props_in = spec
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid_request("object field missing properties"))?;
        let mut props_out = Map::new();
        let mut required = Vec::with_capacity(props_in.len());
        for (name, child) in props_in {
            props_out.insert(name.clone(), schema(child)?);
            required.push(Value::String(name.clone()));
        }
        let mut obj = Map::new();
        obj.insert(
            "type".to_string(),
            if nullable {
                json!(["object", "null"])
            } else {
                json!("object")
            },
        );
        obj.insert("properties".to_string(), Value::Object(props_out));
        obj.insert("required".to_string(), Value::Array(required));
        obj.insert("additionalProperties".to_string(), Value::Bool(false));
        return Ok(Value::Object(obj));
    }

    if kind == "boolean" {
        let mut obj = Map::new();
        obj.insert(
            "type".to_string(),
            if nullable {
                json!(["boolean", "null"])
            } else {
                json!("boolean")
            },
        );
        return Ok(Value::Object(obj));
    }

    if kind == "enum" {
        let mut values = spec
            .get("enum")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| invalid_request("enum field missing enum values"))?;
        if nullable && !values.iter().any(Value::is_null) {
            values.push(Value::Null);
        }
        let mut obj = Map::new();
        obj.insert("enum".to_string(), Value::Array(values));
        return Ok(Value::Object(obj));
    }

    let json_kind = if kind == "text" {
        "string"
    } else if spec
        .get("integer")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        "integer"
    } else {
        "number"
    };
    let mut result = Map::new();
    result.insert(
        "type".to_string(),
        if nullable {
            json!([json_kind, "null"])
        } else {
            json!(json_kind)
        },
    );
    if let Some(spec_obj) = spec.as_object() {
        if json_kind == "string" {
            for key in ["minLength", "maxLength", "pattern", "enum"] {
                if let Some(v) = spec_obj.get(key) {
                    result.insert(key.to_string(), v.clone());
                }
            }
        } else if matches!(json_kind, "number" | "integer") {
            for key in [
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "enum",
            ] {
                if let Some(v) = spec_obj.get(key) {
                    result.insert(key.to_string(), v.clone());
                }
            }
        }
    }
    Ok(Value::Object(result))
}

/// Alias for [`schema`].
pub fn ar_schema(spec: &Value) -> Result<Value> {
    schema(spec)
}

/// Return a field's effective spec, prepending parent list instructions when annotated
/// (`ar.py::complete_spec`).
pub fn complete_spec(record: &Value, name: &str) -> Result<Value> {
    let fields = record
        .get("fields")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_request("record missing fields"))?;
    let mut spec = fields
        .get(name)
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| invalid_request(format!("missing field {name:?}")))?;
    let annotation = field_annotation(record, name);
    if let Some(parent) = annotation
        .and_then(|a| a.get("parent"))
        .and_then(Value::as_str)
        && !parent.is_empty()
    {
        let parent_spec = fields.get(parent).and_then(Value::as_object);
        let Some(p_obj) = parent_spec else {
            return Err(invalid_request(format!(
                "Missing parent list definition: {name}"
            )));
        };
        if p_obj.get("type").and_then(Value::as_str) != Some("list") {
            return Err(invalid_request(format!(
                "Missing parent list definition: {name}"
            )));
        }
        let p_inst = p_obj
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        let c_inst = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        spec.insert(
            "instructions".to_string(),
            Value::String(format!(
                "Parent list {parent}: {p_inst}\nRequested element: {c_inst}"
            )),
        );
    }
    Ok(Value::Object(spec))
}

/// Build the AR prompt instruction block for `name` in `record` (`ar.py::instruction`).
pub fn instruction(record: &Value, name: &str, compact_lists: bool) -> Result<String> {
    if let Some(img_inst) = image_row_instruction(record, name)? {
        return Ok(img_inst);
    }
    let mut spec = complete_spec(record, name)?;
    let annotation = field_annotation(record, name);
    let has_parent = annotation
        .and_then(|a| a.get("parent"))
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    let physical_row_scope = spec
        .get("_physical_row_scope")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let in_source_row_gates = record
        .get("_source_row_gates")
        .map(|v| match v {
            Value::Array(arr) => arr.iter().any(|x| x.as_str() == Some(name)),
            Value::Object(obj) => obj.contains_key(name),
            _ => false,
        })
        .unwrap_or(false);

    if spec.get("type").and_then(Value::as_str) == Some("boolean")
        && !has_parent
        && !physical_row_scope
        && !in_source_row_gates
        && let Some(obj) = spec.as_object_mut()
    {
        let existing = obj
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        obj.insert(
            "instructions".to_string(),
            Value::String(format!("{existing}{BOOLEAN_INSTRUCTION}")),
        );
    }

    let mut numeric_instructions = String::new();
    if spec.get("type").and_then(Value::as_str) == Some("number")
        && let Some(unit) = spec.get("unit")
        && !unit.is_null()
        && unit.as_str() != Some("")
    {
        let unit_str = match unit {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        numeric_instructions.push_str(&format!(" Return the value in {unit_str}."));
    }

    let intro_head = if spec
        .get("knowledge")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        KNOWLEDGE_INSTRUCTION
    } else {
        EXTRACTION_INSTRUCTION
    };
    let intro = format!(
        "{intro_head} Return exactly one JSON scalar matching the field schema: quote and escape strings, write numbers without digit-group separators, and use null only when the value is unavailable and null is allowed. Zero is a present number. Preserve repeated source words and list entries. End immediately after the value."
    );

    let parent_opt = annotation
        .and_then(|a| a.get("parent"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());
    let in_parent_source_fields = parent_opt.is_some_and(|parent| {
        field_annotation(record, parent)
            .and_then(|a| a.get("source_fields"))
            .and_then(Value::as_array)
            .is_some_and(|arr| arr.iter().any(|v| v.as_str() == Some(name)))
    });
    let source_bound = physical_row_scope || in_parent_source_fields;
    let kind = spec.get("type").and_then(Value::as_str).unwrap_or("");
    let schema_compact = compact_json(&schema(&spec)?);

    if let Some(parent) = parent_opt
        && compact_lists
        && kind != "object"
        && !source_bound
    {
        let parent_inst = record
            .get("fields")
            .and_then(|f| f.get(parent))
            .and_then(|p| p.get("instructions"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let index = annotation
            .and_then(|a| a.get("index"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        return Ok(format!(
            "{intro}\nSchema: {schema_compact}\nInstructions: {parent_inst}{numeric_instructions}\nList: {parent}\nRequested element: Item {index} of {parent}, following requested order and duplicate policy. Return null if this item does not exist.\nField: {name}"
        ));
    }

    if kind == "object" {
        let row_instruction = if source_bound {
            "Check whether this exact physical source row satisfies the requested row scope and exclusions. Return null if this physical row does not match the requested scope, even though it exists. If it qualifies, return one JSON object matching the schema, using only this row's cells. "
        } else {
            "Extract one complete requested row from the supplied document. Return one JSON object matching the schema, or null only if that row does not exist. Keep all properties on the same physical row. "
        };
        let inst = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        let props = spec.get("properties").cloned().unwrap_or(Value::Null);
        return Ok(format!(
            "{row_instruction}Use null for a missing cell in an existing row. Preserve distinct rows with identical values. Follow the requested row scope and exclusions. End immediately after the object or null.\nField: {name}\nSchema: {schema_compact}\nInstructions: {inst}\nProperty instructions: {}",
            py_default_json(&props)
        ));
    }

    let display_name = record
        .get("_source_row_names")
        .and_then(|m| m.get(name))
        .and_then(Value::as_str)
        .unwrap_or(name);
    let inst = spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    Ok(format!(
        "{intro}\nField: {display_name}\nSchema: {schema_compact}\nInstructions: {inst}{numeric_instructions}"
    ))
}

/// Alias for [`instruction`].
pub fn ar_instruction(record: &Value, name: &str, compact_lists: bool) -> Result<String> {
    instruction(record, name, compact_lists)
}

/// Split out the remaining typed-head (Path B) fields of `record` (`ar.py::typed_record`).
///
/// Converts `compute` fields into `number` fields, omits all AR fields and any list
/// whose query/source fields are all handled by AR. Returns `None` if no typed fields remain.
pub fn typed_record(record: &Value) -> Option<Value> {
    let record_obj = record.as_object()?;
    let fields_in = record_obj.get("fields")?.as_object()?;
    let active: HashSet<String> = names(record).into_iter().collect();
    let annotations = record_obj
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object);

    let mut omitted = active.clone();
    for (name, spec) in fields_in {
        if spec.get("type").and_then(Value::as_str) != Some("list") {
            continue;
        }
        let ann = annotations.and_then(|a| a.get(name));
        let queries = ann
            .and_then(|a| a.get("query_fields").or_else(|| a.get("source_fields")))
            .and_then(Value::as_array);
        if let Some(q) = queries
            && !q.is_empty()
            && q.iter()
                .all(|v| v.as_str().is_some_and(|s| active.contains(s)))
        {
            omitted.insert(name.clone());
        }
    }

    let mut fields_out = Map::new();
    for (name, spec) in fields_in {
        if omitted.contains(name) {
            continue;
        }
        if spec.get("type").and_then(Value::as_str) == Some("compute") {
            let mut copy = spec.as_object().cloned().unwrap_or_default();
            copy.insert("type".to_string(), Value::String("number".to_string()));
            fields_out.insert(name.clone(), Value::Object(copy));
        } else {
            fields_out.insert(name.clone(), spec.clone());
        }
    }
    if fields_out.is_empty() {
        return None;
    }

    let mut result = record_obj.clone();
    if let Some(answers) = record_obj.get("answers").and_then(Value::as_object) {
        let mut filtered = Map::new();
        for k in fields_out.keys() {
            if let Some(v) = answers.get(k) {
                filtered.insert(k.clone(), v.clone());
            }
        }
        result.insert("answers".to_string(), Value::Object(filtered));
    }
    result.insert("fields".to_string(), Value::Object(fields_out));
    Some(Value::Object(result))
}

fn field_annotation<'a>(record: &'a Value, name: &str) -> Option<&'a Value> {
    record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(|f| f.get(name))
}

fn image_row_instruction(record: &Value, name: &str) -> Result<Option<String>> {
    let Some(spec) = record.get("fields").and_then(|f| f.get(name)) else {
        return Ok(None);
    };
    if is_inventory(spec) {
        let label_spec = json!({
            "label": {
                "type": "text",
                "nullable": false,
                "instructions": IMAGE_ROW_LABEL_INSTRUCTIONS
            }
        });
        let inst = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        return Ok(Some(format!(
            "Extract the requested value from the supplied input, following the field instructions and scope. Return all matching rows together as one JSON array. Return null only if there are no matching rows. Each object describes one physical row. Missing cells are null inside that object; they do not make the row absent. Keep wrapped text in its row and preserve distinct duplicate rows. Stop after the last matching row and close the array. End immediately after the array or null.\nField: {name}\nSchema: {}\nInstructions: {inst}\nProperty instructions: {}",
            compact_json(&inventory_schema(spec)),
            py_default_json(&label_spec)
        )));
    }
    if spec
        .get("_image_row_proposal")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        if spec
            .get("_share_image_row_description")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && let Some(shared) = shared_description_instruction(record, name, spec)?
        {
            return Ok(Some(shared));
        }
        let inst = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        let props = spec.get("properties").cloned().unwrap_or(Value::Null);
        return Ok(Some(format!(
            "Extract one complete requested row from the supplied document. Return one JSON object matching the schema. Keep all properties on the same physical row. Use null for a missing cell in an existing row. Preserve distinct rows with identical values. Follow the requested row scope and exclusions. End immediately after the object.\nField: {name}\nSchema: {}\nInstructions: {inst}\nProperty instructions: {}",
            compact_json(&schema(spec)?),
            py_default_json(&props)
        )));
    }
    Ok(None)
}

fn shared_description_instruction(
    record: &Value,
    name: &str,
    spec: &Value,
) -> Result<Option<String>> {
    let ann = field_annotation(record, name);
    let parent = ann.and_then(|a| a.get("parent")).and_then(Value::as_str);
    let index = ann
        .and_then(|a| a.get("index"))
        .filter(|v| !v.is_boolean())
        .and_then(Value::as_i64);
    let (Some(parent), Some(index)) = (parent, index) else {
        return Ok(None);
    };
    if parent.is_empty() || index < 1 {
        return Ok(None);
    }
    let Some(props) = spec.get("properties") else {
        return Ok(None);
    };
    let Some(desc) = props.get("description") else {
        return Ok(None);
    };
    if desc.get("type").and_then(Value::as_str) != Some("text")
        || desc
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("")
            .is_empty()
    {
        return Ok(None);
    }
    let ordinal = format!("Item {index} of {parent} in the supplied document. ");
    let scope = spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let Some(rest) = scope.strip_prefix(&ordinal) else {
        return Ok(None);
    };
    let desc_obj = json!({"description": desc});
    Ok(Some(format!(
        "Extract one complete requested row from the supplied document. Return one JSON object matching the schema. Keep all properties on the same physical row. Use null for a missing cell in an existing row. Preserve distinct rows with identical values. Follow the requested row scope and exclusions. End immediately after the object.\nSchema: {}\nInstructions: {rest}\nProperty instructions: {}\nRequested row: {}\nField: {name}\nProperty instructions: {}",
        compact_json(&schema(spec)?),
        py_default_json(props),
        ordinal.trim_end(),
        py_default_json(&desc_obj)
    )))
}

/// Compact JSON serialization matching Python `json.dumps(v, ensure_ascii=False, separators=(",", ":"))`.
pub fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// JSON serialization matching Python's default `json.dumps(v, ensure_ascii=False)`
/// (`", "` between items and `": "` between keys and values).
pub fn py_default_json(value: &Value) -> String {
    let mut out = String::new();
    write_py_json(value, &mut out);
    out
}

fn write_py_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(_) => {
            if let Ok(s) = serde_json::to_string(value) {
                out.push_str(&s);
            }
        }
        Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                if let Ok(ks) = serde_json::to_string(k) {
                    out.push_str(&ks);
                }
                out.push_str(": ");
                write_py_json(v, out);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_number_rounds_half_up_and_rejects_bool() {
        let num_spec = json!({"type": "number", "decimals": 2});
        assert_eq!(
            normalize_number(&json!(1.005), &num_spec).unwrap(),
            json!(1.01)
        );
        assert_eq!(
            normalize_number(&json!("2.345"), &num_spec).unwrap(),
            json!(2.35)
        );
        assert_eq!(
            normalize_number(&json!(null), &num_spec).unwrap(),
            json!(null)
        );
        assert!(normalize_number(&json!(true), &num_spec).is_err());

        let int_spec = json!({"type": "number", "integer": true});
        assert_eq!(normalize_number(&json!(2.5), &int_spec).unwrap(), json!(3));
        assert_eq!(
            normalize_number(&json!(-2.5), &int_spec).unwrap(),
            json!(-3)
        );

        let unconstrained = json!({"type": "number"});
        assert_eq!(
            normalize_number(&json!(42.0), &unconstrained).unwrap(),
            json!(42)
        );
    }

    #[test]
    fn ar_schema_for_all_ar_types() {
        assert_eq!(
            schema(&json!({"type": "text", "minLength": 1, "pattern": "^[A-Z]+$"})).unwrap(),
            json!({"type": ["string", "null"], "minLength": 1, "pattern": "^[A-Z]+$"})
        );
        assert_eq!(
            schema(&json!({"type": "number", "integer": true, "nullable": false, "minimum": 0}))
                .unwrap(),
            json!({"type": "integer", "minimum": 0})
        );
        assert_eq!(
            schema(&json!({"type": "boolean"})).unwrap(),
            json!({"type": ["boolean", "null"]})
        );
        assert_eq!(
            schema(&json!({"type": "enum", "enum": ["a", "b"]})).unwrap(),
            json!({"enum": ["a", "b", null]})
        );
        assert_eq!(
            schema(&json!({
                "type": "object",
                "properties": {
                    "sku": {"type": "text", "nullable": false},
                    "qty": {"type": "number", "integer": true}
                }
            }))
            .unwrap(),
            json!({
                "type": ["object", "null"],
                "properties": {
                    "sku": {"type": "string"},
                    "qty": {"type": ["integer", "null"]}
                },
                "required": ["sku", "qty"],
                "additionalProperties": false
            })
        );
    }

    #[test]
    fn typed_record_splits_ar_and_converts_compute() {
        let rec = json!({
            "input": "Total is 5 + 3",
            "fields": {
                "vendor": {"type": "text"},
                "sum": {"type": "compute"},
                "due": {"type": "date"}
            }
        });
        let typed = typed_record(&rec).unwrap();
        let fields = typed["fields"].as_object().unwrap();
        assert!(!fields.contains_key("vendor"));
        assert_eq!(fields["sum"]["type"], "number");
        assert_eq!(fields["due"]["type"], "date");
    }

    #[test]
    fn instruction_appends_boolean_and_unit_guidance() {
        let rec = json!({
            "input": "Paid: yes. Total: 12 EUR",
            "fields": {
                "paid": {"type": "boolean", "instructions": "Is it paid?"},
                "amount": {"type": "number", "unit": "EUR", "instructions": "Total amount"}
            }
        });
        let b_inst = instruction(&rec, "paid", true).unwrap();
        assert!(b_inst.contains(BOOLEAN_INSTRUCTION));
        let n_inst = instruction(&rec, "amount", true).unwrap();
        assert!(n_inst.contains("Return the value in EUR."));
    }
}
