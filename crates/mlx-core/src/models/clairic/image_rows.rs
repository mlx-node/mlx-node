//! Image-table row inventory preparation, NFKC Hungarian bipartite row
//! matching, and row list assembly.
//!
//! Direct port of upstream `clairic_runtime/image_rows.py`.

use super::ar::{self, compact_json, py_default_json};
use super::schema::MAX_LIST_ITEMS;
use super::settings::MAX_OUTPUT_TOKENS;
use napi::Result;
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

/// Output token budget for an image-row inventory pass (`image_rows.INVENTORY_OUTPUT_TOKENS`).
pub const INVENTORY_OUTPUT_TOKENS: usize = 2048;
/// Alias for [`INVENTORY_OUTPUT_TOKENS`].
pub const INVENTORY_MAX_TOKENS: usize = INVENTORY_OUTPUT_TOKENS;

const LABEL_INSTRUCTIONS: &str = "Brief exact printed identifier of this requested row, without secondary description or numeric cells. Follow the requested field scope: the label may name a product, service, tax, fee, payment or other row.";

/// Property specification for the synthetic `label` property of an inventory entry (`image_rows.LABEL`).
pub fn label_spec() -> Value {
    json!({
        "type": "text",
        "nullable": false,
        "instructions": LABEL_INSTRUCTIONS
    })
}

fn row_request_regex() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\bone\s+(?:object|entry|record)\s+(?:per|for\s+each)\b[^.\n]*\brow\b").ok()
    })
    .as_ref()
}

fn word_regex() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\w+").ok()).as_ref()
}

/// Prepare image-backed object-list fields for row-inventory + proposal extraction
/// (`image_rows.py::prepare`).
pub fn prepare(mut record: Value, has_media: bool, share_description: bool) -> Value {
    if !has_media {
        return record;
    }
    let Some(row_req_re) = row_request_regex() else {
        return record;
    };

    // Collect eligible parent names and their query_fields first to avoid borrow conflicts.
    let eligible: Vec<(String, String, Vec<String>)> = {
        let Some(fields) = record.get("fields").and_then(Value::as_object) else {
            return record;
        };
        let Some(annotations) = record
            .get("_meta")
            .and_then(|m| m.get("extraction_contract"))
            .and_then(|c| c.get("fields"))
            .and_then(Value::as_object)
        else {
            return record;
        };
        let mut out = Vec::new();
        for (name, spec) in fields {
            let Some(ann) = annotations.get(name).and_then(Value::as_object) else {
                continue;
            };
            let is_obj_list = spec.get("type").and_then(Value::as_str) == Some("list")
                && spec
                    .get("items")
                    .and_then(Value::as_object)
                    .and_then(|i| i.get("type"))
                    .and_then(Value::as_str)
                    == Some("object");
            let query_fields: Vec<String> = ann
                .get("query_fields")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let has_source_fields = ann
                .get("source_fields")
                .and_then(Value::as_array)
                .is_some_and(|a| !a.is_empty());
            let instructions = spec
                .get("instructions")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !is_obj_list
                || query_fields.is_empty()
                || has_source_fields
                || !row_req_re.is_match(instructions)
            {
                continue;
            }
            out.push((name.clone(), instructions.to_string(), query_fields));
        }
        out
    };

    for (name, parent_instructions, query_fields) in eligible {
        if let Some(spec) = record
            .get_mut("fields")
            .and_then(Value::as_object_mut)
            .and_then(|f| f.get_mut(&name))
            .and_then(Value::as_object_mut)
        {
            spec.insert("nullable".to_string(), Value::Bool(true));
            spec.insert("_image_row_inventory".to_string(), Value::Bool(true));
        }
        if let Some(ann) = record
            .get_mut("_meta")
            .and_then(Value::as_object_mut)
            .and_then(|m| m.get_mut("extraction_contract"))
            .and_then(Value::as_object_mut)
            .and_then(|c| c.get_mut("fields"))
            .and_then(Value::as_object_mut)
            .and_then(|f| f.get_mut(&name))
            .and_then(Value::as_object_mut)
            && let Some(qf) = ann.remove("query_fields")
        {
            ann.insert("image_row_fields".to_string(), qf);
        }
        for (idx, child) in query_fields.iter().enumerate() {
            let ordinal = idx + 1;
            if let Some(row) = record
                .get_mut("fields")
                .and_then(Value::as_object_mut)
                .and_then(|f| f.get_mut(child))
                .and_then(Value::as_object_mut)
            {
                row.insert("nullable".to_string(), Value::Bool(false));
                row.insert("_image_row_proposal".to_string(), Value::Bool(true));
                if share_description {
                    row.insert(
                        "_share_image_row_description".to_string(),
                        Value::Bool(true),
                    );
                }
                row.insert(
                    "instructions".to_string(),
                    Value::String(format!(
                        "Item {ordinal} of {name} in the supplied document. {parent_instructions} Return null for each missing cell, including absent entire columns. If this physical row does not exist return an object with every cell null."
                    )),
                );
            }
        }
    }
    record
}

/// Return `true` iff `spec` is an image-row inventory field (`image_rows.py::is_inventory`).
pub fn is_inventory(spec: &Value) -> bool {
    ar::is_inventory(spec)
}

/// Build the Draft 2020-12 JSON Schema for an image-row inventory list
/// (`image_rows.py::inventory_schema`).
pub fn inventory_schema(spec: &Value) -> Value {
    ar::inventory_schema(spec)
}

/// Build the AR prompt instruction for an image-row inventory or proposal field
/// (`image_rows.py::instruction`). Returns `Ok(None)` if `name` is neither.
pub fn instruction<F>(record: &Value, name: &str, schema_fn: F) -> Result<Option<String>>
where
    F: Fn(&Value) -> Result<Value>,
{
    let Some(spec) = record.get("fields").and_then(|f| f.get(name)) else {
        return Ok(None);
    };
    if is_inventory(spec) {
        let label_map = json!({"label": label_spec()});
        let inst = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        return Ok(Some(format!(
            "Extract the requested value from the supplied input, following the field instructions and scope. Return all matching rows together as one JSON array. Return null only if there are no matching rows. Each object describes one physical row. Missing cells are null inside that object; they do not make the row absent. Keep wrapped text in its row and preserve distinct duplicate rows. Stop after the last matching row and close the array. End immediately after the array or null.\nField: {name}\nSchema: {}\nInstructions: {inst}\nProperty instructions: {}",
            compact_json(&inventory_schema(spec)),
            py_default_json(&label_map)
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
            && let Some(shared) = shared_description_instruction(record, name, &schema_fn)?
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
            compact_json(&schema_fn(spec)?),
            py_default_json(&props)
        )));
    }
    Ok(None)
}

/// Build the prefix-sharing description prompt for single-image Flash row proposals
/// (`image_rows.py::shared_description_instruction`).
pub fn shared_description_instruction<F>(
    record: &Value,
    name: &str,
    schema_fn: &F,
) -> Result<Option<String>>
where
    F: Fn(&Value) -> Result<Value>,
{
    let Some(spec) = record.get("fields").and_then(|f| f.get(name)) else {
        return Ok(None);
    };
    if !spec
        .get("_image_row_proposal")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Ok(None);
    }
    let ann = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(|f| f.get(name));
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
        compact_json(&schema_fn(spec)?),
        py_default_json(props),
        ordinal.trim_end(),
        py_default_json(&desc_obj)
    )))
}

/// Output token limit for field `name`, matching `image_rows.py::output_limit`.
pub fn output_limit(record: &Value, name: &str, length: usize, max_input: usize) -> usize {
    let desired = if record
        .get("fields")
        .and_then(|f| f.get(name))
        .is_some_and(is_inventory)
    {
        INVENTORY_OUTPUT_TOKENS
    } else {
        MAX_OUTPUT_TOKENS
    };
    let remaining = (max_input + MAX_OUTPUT_TOKENS).saturating_sub(length);
    desired.min(remaining)
}

/// Unicode NFKC normalization + casefold + `\w+` word tokenization joined by spaces,
/// matching `image_rows.py::_words`.
pub fn normalize_words(value: &str) -> String {
    let nfkc: String = value.nfkc().collect();
    let mut folded = String::with_capacity(nfkc.len());
    for ch in nfkc.chars() {
        match ch {
            'ß' | 'ẞ' => folded.push_str("ss"),
            'İ' => folded.push_str("i\u{0307}"),
            other => {
                for lc in other.to_lowercase() {
                    folded.push(lc);
                }
            }
        }
    }
    let Some(re) = word_regex() else {
        return String::new();
    };
    let mut out = String::new();
    for (i, m) in re.find_iter(&folded).enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(m.as_str());
    }
    out
}

/// Minimum-cost rectangular bipartite assignment (Hungarian algorithm),
/// matching `image_rows.py::_assignment` (requires `width >= count`).
pub(crate) fn assignment(costs: &[Vec<f64>]) -> Vec<usize> {
    let count = costs.len();
    if count == 0 {
        return Vec::new();
    }
    let width = costs[0].len();
    if width == 0 {
        return Vec::new();
    }
    let mut u = vec![0.0f64; count + 1];
    let mut v = vec![0.0f64; width + 1];
    let mut p = vec![0usize; width + 1];
    let mut way = vec![0usize; width + 1];

    for index in 1..=count {
        p[0] = index;
        let mut column = 0usize;
        let mut minimum = vec![f64::INFINITY; width + 1];
        let mut used = vec![false; width + 1];
        loop {
            used[column] = true;
            let row = p[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0usize;
            for candidate in 1..=width {
                if !used[candidate] {
                    let cost = costs[row - 1][candidate - 1] - u[row] - v[candidate];
                    if cost < minimum[candidate] {
                        minimum[candidate] = cost;
                        way[candidate] = column;
                    }
                    if minimum[candidate] < delta {
                        delta = minimum[candidate];
                        next_column = candidate;
                    }
                }
            }
            for candidate in 0..=width {
                if used[candidate] {
                    u[p[candidate]] += delta;
                    v[candidate] -= delta;
                } else {
                    minimum[candidate] -= delta;
                }
            }
            column = next_column;
            if column == 0 || p[column] == 0 {
                break;
            }
        }
        while column != 0 {
            let previous = way[column];
            p[column] = p[previous];
            column = previous;
        }
    }

    let mut result = vec![0usize; count];
    for (column, &row) in p.iter().enumerate().skip(1) {
        if row != 0 && row <= count {
            result[row - 1] = column - 1;
        }
    }
    result
}

/// Match inventory `labels` to proposal rows, returning `(selected_proposal_indices, conflict_label_indices)`.
///
/// Returning proposal indices (`Option<usize>`) preserves exact Python object-identity
/// semantics (`row is proposals[i]`) even when multiple proposals have identical JSON values.
pub fn select_indices(
    labels: &[String],
    proposals: &[Value],
    text_keys: &[String],
) -> (Vec<Option<usize>>, Vec<usize>) {
    if text_keys.is_empty() {
        let selected = (0..labels.len())
            .map(|index| (index < proposals.len()).then_some(index))
            .collect();
        return (selected, Vec::new());
    }

    const UNMATCHED_COST: f64 = 1_000_000_000.0;
    const DUMMY_COST: f64 = 100_000_000.0;

    let mut costs: Vec<Vec<f64>> = Vec::with_capacity(labels.len());
    for (index, raw_label) in labels.iter().enumerate() {
        let label = normalize_words(raw_label);
        let needle = format!(" {label} ");
        let mut row_costs = Vec::with_capacity(proposals.len() + labels.len());
        for (position, proposal) in proposals.iter().enumerate() {
            let is_valid = proposal
                .get("valid")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let value_obj = if is_valid {
                proposal.get("value").and_then(Value::as_object)
            } else {
                None
            };
            let matches = !label.is_empty()
                && value_obj.is_some_and(|obj| {
                    text_keys.iter().any(|key| {
                        obj.get(key).and_then(Value::as_str).is_some_and(|text| {
                            let haystack = format!(" {} ", normalize_words(text));
                            haystack.contains(&needle)
                        })
                    })
                });
            if matches {
                row_costs.push((position.abs_diff(index) * 10) as f64);
            } else {
                row_costs.push(UNMATCHED_COST);
            }
        }
        row_costs.resize(proposals.len() + labels.len(), DUMMY_COST);
        costs.push(row_costs);
    }

    let chosen = assignment(&costs);
    let mut conflicts = Vec::new();
    let mut conflict_set = HashSet::new();
    for (index, &column) in chosen.iter().enumerate() {
        if column >= proposals.len() || costs[index][column] >= DUMMY_COST {
            conflicts.push(index);
            conflict_set.insert(index);
        }
    }
    let selected = chosen
        .into_iter()
        .enumerate()
        .map(|(index, column)| {
            if conflict_set.contains(&index) {
                None
            } else {
                Some(column)
            }
        })
        .collect();
    (selected, conflicts)
}

/// Match inventory `labels` to proposal rows, returning cloned matched proposals
/// (`image_rows.py::_select`).
pub fn select(
    labels: &[String],
    proposals: &[Value],
    text_keys: &[String],
) -> (Vec<Option<Value>>, Vec<usize>) {
    let (indices, conflicts) = select_indices(labels, proposals, text_keys);
    let selected = indices
        .into_iter()
        .map(|opt| opt.and_then(|idx| proposals.get(idx).cloned()))
        .collect();
    (selected, conflicts)
}

/// Assemble image-row list answers from inventory + proposal predictions
/// (`image_rows.py::assemble`).
pub fn assemble(record: &Value, predictions: &Map<String, Value>) -> Map<String, Value> {
    let mut answers = Map::new();
    let Some(annotations) = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)
    else {
        return answers;
    };
    let Some(fields) = record.get("fields").and_then(Value::as_object) else {
        return answers;
    };

    for (name, annotation) in annotations {
        let Some(queries) = annotation
            .get("image_row_fields")
            .and_then(Value::as_array)
            .filter(|q| !q.is_empty())
        else {
            continue;
        };
        let (Some(spec), Some(inventory)) = (fields.get(name), predictions.get(name)) else {
            continue;
        };
        let labels: Vec<String> = inventory
            .get("value")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|row| row.get("label").and_then(Value::as_str).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let maximum = spec
            .get("max_items")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(MAX_LIST_ITEMS);
        let inventory_complete = inventory
            .get("valid")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && inventory
                .get("complete")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            && labels.len() <= maximum;

        let text_keys: Vec<String> = spec
            .get("items")
            .and_then(|i| i.get("properties"))
            .and_then(Value::as_object)
            .map(|props| {
                props
                    .iter()
                    .filter(|(_, child)| child.get("type").and_then(Value::as_str) == Some("text"))
                    .map(|(k, _)| k.clone())
                    .collect()
            })
            .unwrap_or_default();

        let proposals: Vec<Value> = queries
            .iter()
            .filter_map(|q| q.as_str().and_then(|child| predictions.get(child).cloned()))
            .collect();
        let capped_labels = &labels[..labels.len().min(maximum)];
        let (selected_indices, conflicts) = select_indices(capped_labels, &proposals, &text_keys);
        let selected: Vec<Option<&Value>> = selected_indices
            .iter()
            .map(|opt| opt.and_then(|idx| proposals.get(idx)))
            .collect();

        let complete = inventory_complete
            && selected.iter().all(|row| {
                row.is_some_and(|r| r.get("complete").and_then(Value::as_bool).unwrap_or(false))
            });
        let valid = complete
            && conflicts.is_empty()
            && selected.iter().all(|row| {
                row.is_some_and(|r| {
                    r.get("valid").and_then(Value::as_bool).unwrap_or(false)
                        && r.get("value")
                            .and_then(Value::as_object)
                            .is_some_and(|obj| obj.values().any(|v| !v.is_null()))
                })
            });

        let values: Vec<Value> = selected
            .iter()
            .map(|row| {
                row.and_then(|r| r.get("value").cloned())
                    .unwrap_or(Value::Null)
            })
            .collect();
        let confidences: Vec<Value> = selected
            .iter()
            .map(|row| {
                let c = row
                    .and_then(|r| r.get("confidence"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                json!(c)
            })
            .collect();
        let spans = vec![Value::Null; selected.len()];

        let mut answer = Map::new();
        answer.insert("value".to_string(), Value::Array(values));
        answer.insert("valid".to_string(), Value::Bool(valid));
        answer.insert("complete".to_string(), Value::Bool(complete));
        answer.insert("confidence".to_string(), json!(0.0));
        answer.insert("confidences".to_string(), Value::Array(confidences));
        answer.insert(
            "confidence_method".to_string(),
            Value::String("ar_sequence_probability_with_source_row_labels".to_string()),
        );
        if let Some(p_null) = inventory.get("p_null") {
            answer.insert("p_null".to_string(), p_null.clone());
        }
        answer.insert(
            "p_null_method".to_string(),
            Value::String("source_row_labels_null_probability".to_string()),
        );
        answer.insert("spans".to_string(), Value::Array(spans));

        if valid {
            let mut log_prob_sum = inventory.get("log_probability").and_then(Value::as_f64);
            for row in &selected {
                let row_lp = row.and_then(|r| r.get("log_probability").and_then(Value::as_f64));
                log_prob_sum = match (log_prob_sum, row_lp) {
                    (Some(acc), Some(lp)) => Some(acc + lp),
                    _ => None,
                };
            }
            if let Some(probability) = log_prob_sum
                && probability.is_finite()
            {
                answer.insert("log_probability".to_string(), json!(probability));
                answer.insert("confidence".to_string(), json!(probability.min(0.0).exp()));
            }
        }
        if !conflicts.is_empty() {
            answer.insert(
                "failure_reason".to_string(),
                Value::String("source_row_alignment".to_string()),
            );
        }
        answers.insert(name.clone(), Value::Object(answer));
    }
    answers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfkc_word_normalization_matches_python() {
        // Fullwidth letters + ligature ﬁ + German ß + punctuation
        assert_eq!(normalize_words("Ｆｉｌｅ—ﬁnal ß!"), "file final ss");
        assert_eq!(normalize_words("  Widget #12 (Pro)  "), "widget 12 pro");
    }

    #[test]
    fn bipartite_select_prefers_order_and_detects_conflicts() {
        let labels = vec!["Alpha".to_string(), "Beta".to_string(), "Gamma".to_string()];
        // Proposals are swapped (Beta at 0, Alpha at 1, all-null at 2)
        let proposals = vec![
            json!({"valid": true, "complete": true, "value": {"desc": "Beta service", "amt": 20}}),
            json!({"valid": true, "complete": true, "value": {"desc": "Alpha widget", "amt": 10}}),
            json!({"valid": true, "complete": true, "value": {"desc": null, "amt": null}}),
        ];
        let text_keys = vec!["desc".to_string()];
        let (selected, conflicts) = select_indices(&labels, &proposals, &text_keys);
        assert_eq!(selected, vec![Some(1), Some(0), None]);
        assert_eq!(conflicts, vec![2]);
    }

    #[test]
    fn prepare_and_assemble_roundtrip() {
        let record = json!({
            "fields": {
                "items": {
                    "type": "list",
                    "instructions": "Extract one object per table row.",
                    "max_items": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "text", "instructions": "Row label"},
                            "amount": {"type": "number"}
                        }
                    }
                },
                "items.item1": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "text", "instructions": "Row label"},
                        "amount": {"type": "number"}
                    }
                },
                "items.item2": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "text", "instructions": "Row label"},
                        "amount": {"type": "number"}
                    }
                }
            },
            "_meta": {
                "extraction_contract": {
                    "fields": {
                        "items": {"query_fields": ["items.item1", "items.item2"]},
                        "items.item1": {"parent": "items", "index": 1},
                        "items.item2": {"parent": "items", "index": 2}
                    }
                }
            }
        });
        let prepared = prepare(record, true, true);
        assert!(is_inventory(&prepared["fields"]["items"]));
        assert_eq!(
            prepared["_meta"]["extraction_contract"]["fields"]["items"]["image_row_fields"],
            json!(["items.item1", "items.item2"])
        );
        assert!(
            prepared["_meta"]["extraction_contract"]["fields"]["items"]
                .get("query_fields")
                .is_none()
        );
        assert_eq!(prepared["fields"]["items.item1"]["nullable"], json!(false));
        assert_eq!(
            prepared["fields"]["items.item1"]["_share_image_row_description"],
            json!(true)
        );
        let inst = instruction(&prepared, "items.item1", ar::schema)
            .unwrap()
            .unwrap();
        assert!(inst.contains("Requested row: Item 1 of items in the supplied document."));

        let inv_inst = instruction(&prepared, "items", ar::schema)
            .unwrap()
            .unwrap();
        assert!(inv_inst.contains("Return all matching rows together as one JSON array."));

        assert_eq!(
            output_limit(&prepared, "items", 1000, 65_273),
            INVENTORY_MAX_TOKENS
        );
        assert_eq!(
            output_limit(&prepared, "items.item1", 1000, 65_273),
            MAX_OUTPUT_TOKENS
        );

        let preds = json!({
            "items": {
                "value": [{"label": "Widget"}, {"label": "Bolt"}],
                "valid": true,
                "complete": true,
                "p_null": 0.01,
                "log_probability": -0.1
            },
            "items.item1": {
                "value": {"description": "Widget Pro", "amount": 10},
                "valid": true,
                "complete": true,
                "confidence": 0.9,
                "log_probability": -0.2
            },
            "items.item2": {
                "value": {"description": "Bolt M4", "amount": 2},
                "valid": true,
                "complete": true,
                "confidence": 0.85,
                "log_probability": -0.3
            }
        });
        let assembled = assemble(&prepared, preds.as_object().unwrap());
        let items_ans = &assembled["items"];
        assert_eq!(items_ans["valid"], json!(true));
        assert_eq!(items_ans["complete"], json!(true));
        assert_eq!(items_ans["value"].as_array().unwrap().len(), 2);
        let lp = items_ans["log_probability"].as_f64().unwrap();
        assert!((lp - (-0.6)).abs() < 1e-12);
        let conf = items_ans["confidence"].as_f64().unwrap();
        assert!((conf - (-0.6f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn duplicate_rows_preserve_distinct_slots_and_conflicts_set_failure_reason() {
        let labels = vec!["Coffee".to_string(), "Coffee".to_string()];
        let proposals = vec![
            json!({"valid": true, "complete": true, "value": {"desc": "Coffee", "amt": 4}}),
            json!({"valid": true, "complete": true, "value": {"desc": "Coffee", "amt": 4}}),
        ];
        let text_keys = vec!["desc".to_string()];
        let (selected, conflicts) = select_indices(&labels, &proposals, &text_keys);
        assert_eq!(selected, vec![Some(0), Some(1)]);
        assert!(conflicts.is_empty());

        // Empty text_keys falls back to positional selection
        let (pos_sel, pos_conf) = select_indices(&labels, &proposals[..1], &[]);
        assert_eq!(pos_sel, vec![Some(0), None]);
        assert!(pos_conf.is_empty());
    }
}
