//! Recover unresolved image rows backed by a completed row inventory.
//!
//! Direct port of upstream `clairic_runtime/row_omission.py`.

use super::{ar, image_rows, json_validate, schema::MAX_LIST_ITEMS};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Minimal AR task metadata needed by [`prepare`] and [`merge`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskInfo {
    pub name: String,
    pub input_len: usize,
    pub has_media: bool,
}

/// Per-parent recovery state tracked between [`prepare`] and [`merge`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowOmissionContext {
    pub parent: String,
    pub queries: Vec<String>,
    pub labels: Vec<String>,
    pub text_keys: Vec<String>,
    /// Matched proposal slot index in `queries` for each inventory label (`None` if conflicted).
    pub selected_indices: Vec<Option<usize>>,
    pub conflicts: Vec<usize>,
    pub children: Vec<String>,
}

/// Output of [`prepare`]: a repair record plus per-parent recovery contexts.
#[derive(Debug, Clone)]
pub struct RowOmissionPlan {
    /// Cloned record configured for a single omission-recovery retry pass
    /// (`_null_recovery: true`, `_row_omission_recovery: true`, stricter `_ar_constraints`).
    pub repair_record: Value,
    /// Ordered names of proposal tasks to re-run with stricter `anyOf` constraints.
    pub retry_task_names: Vec<String>,
    /// Stricter `{"json_schema": "..."}` constraint per retry task.
    pub retry_constraints: BTreeMap<String, Value>,
    /// Input token lengths for each retry task (used for retry token accounting in [`merge`]).
    pub retry_input_lengths: HashMap<String, usize>,
    /// Per-parent recovery contexts.
    pub contexts: Vec<RowOmissionContext>,
}

/// Build the stricter `{"anyOf": [...]}` JSON Schema requiring at least one text
/// column in `text_keys` to be a non-null `"string"` (`row_omission.py` lines 97-102).
pub fn build_non_null_text_anyof_schema(
    base_schema: &Value,
    text_keys: &[String],
) -> Option<Value> {
    let props = base_schema.get("properties")?.as_object()?;
    if text_keys.is_empty() || !text_keys.iter().all(|k| props.contains_key(k)) {
        return None;
    }
    let mut branches = Vec::with_capacity(text_keys.len());
    for key in text_keys {
        let mut branch = base_schema.clone();
        let prop = branch
            .get_mut("properties")
            .and_then(Value::as_object_mut)
            .and_then(|p| p.get_mut(key))
            .and_then(Value::as_object_mut)?;
        prop.insert("type".to_string(), Value::String("string".to_string()));
        branches.push(branch);
    }
    Some(json!({"anyOf": branches}))
}

fn is_truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// Detect inventory labels matched to all-null proposals and prepare a stricter
/// `anyOf` retry pass (`row_omission.py::prepare`).
pub fn prepare(
    record: &Value,
    predictions: &Map<String, Value>,
    tasks: &[TaskInfo],
) -> Option<RowOmissionPlan> {
    if is_truthy(record.get("_null_recovery")) || is_truthy(record.get("_row_omission_recovery")) {
        return None;
    }
    let annotations = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)?;
    let fields = record.get("fields").and_then(Value::as_object)?;

    let mut task_map: HashMap<&str, &TaskInfo> = HashMap::with_capacity(tasks.len());
    for task in tasks {
        if task_map.insert(task.name.as_str(), task).is_some() {
            return None;
        }
    }

    let mut constraints: Map<String, Value> = record
        .get("_ar_constraints")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut contexts: Vec<RowOmissionContext> = Vec::new();
    let mut retry_task_names: Vec<String> = Vec::new();
    let mut retry_constraints: BTreeMap<String, Value> = BTreeMap::new();
    let mut retry_input_lengths: HashMap<String, usize> = HashMap::new();

    for (parent, annotation_val) in annotations {
        let Some(annotation) = annotation_val.as_object() else {
            continue;
        };
        if annotation.len() != 1 || !annotation.contains_key("image_row_fields") {
            continue;
        }
        let Some(queries_arr) = annotation.get("image_row_fields").and_then(Value::as_array) else {
            continue;
        };
        let queries: Vec<String> = queries_arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        if queries.is_empty() || queries.len() != queries_arr.len() {
            continue;
        }
        let unique_queries: HashSet<&str> = queries.iter().map(String::as_str).collect();
        if unique_queries.len() != queries.len() {
            continue;
        }

        let (Some(spec), Some(inventory)) = (fields.get(parent), predictions.get(parent)) else {
            continue;
        };
        if !image_rows::is_inventory(spec)
            || is_truthy(spec.get("knowledge"))
            || !is_truthy(inventory.get("valid"))
            || !is_truthy(inventory.get("complete"))
        {
            continue;
        }

        let Some(inv_rows) = inventory.get("value").and_then(Value::as_array) else {
            continue;
        };
        let max_items = spec
            .get("max_items")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(MAX_LIST_ITEMS);
        if inv_rows.is_empty() || inv_rows.len() > max_items.min(queries.len()) {
            continue;
        }
        let mut labels = Vec::with_capacity(inv_rows.len());
        let mut labels_ok = true;
        for row in inv_rows {
            let Some(obj) = row.as_object() else {
                labels_ok = false;
                break;
            };
            if obj.len() != 1 {
                labels_ok = false;
                break;
            }
            let Some(label) = obj.get("label").and_then(Value::as_str) else {
                labels_ok = false;
                break;
            };
            if label.trim().is_empty() {
                labels_ok = false;
                break;
            }
            labels.push(label.to_string());
        }
        if !labels_ok || queries.iter().any(|child| !predictions.contains_key(child)) {
            continue;
        }

        let text_keys: Vec<String> = spec
            .get("items")
            .and_then(|i| i.get("properties"))
            .and_then(Value::as_object)
            .map(|props| {
                props
                    .iter()
                    .filter(|(_, child)| {
                        child.get("type").and_then(Value::as_str) == Some("text")
                            && !is_truthy(child.get("knowledge"))
                    })
                    .map(|(k, _)| k.clone())
                    .collect()
            })
            .unwrap_or_default();
        if text_keys.is_empty() {
            continue;
        }

        let proposals: Vec<Value> = queries
            .iter()
            .filter_map(|child| predictions.get(child).cloned())
            .collect();
        let (selected_indices, conflicts) =
            image_rows::select_indices(&labels, &proposals, &text_keys);
        if conflicts.is_empty() {
            continue;
        }
        let prev_selected_ok = selected_indices.iter().flatten().all(|&pos| {
            proposals
                .get(pos)
                .is_some_and(|row| is_truthy(row.get("valid")) && is_truthy(row.get("complete")))
        });
        if !prev_selected_ok {
            continue;
        }

        let mut children: Vec<String> = Vec::with_capacity(conflicts.len());
        let mut local_constraints: BTreeMap<String, Value> = BTreeMap::new();
        let mut all_conflicts_recoverable = true;

        for &index in &conflicts {
            let Some(child) = queries.get(index) else {
                all_conflicts_recoverable = false;
                break;
            };
            let (Some(row), Some(child_spec)) = (predictions.get(child), fields.get(child)) else {
                all_conflicts_recoverable = false;
                break;
            };
            let expected_ann = json!({"parent": parent, "index": index + 1});
            let ann_matches = annotations
                .get(child)
                .is_some_and(|a| json_validate::json_equal(a, &expected_ann));
            let row_value_all_null = row
                .get("value")
                .and_then(Value::as_object)
                .is_some_and(|obj| !obj.is_empty() && obj.values().all(Value::is_null));
            let already_selected = selected_indices.contains(&Some(index));
            let task = task_map.get(child.as_str()).copied();
            let native_schema_str = constraints
                .get(child)
                .and_then(Value::as_object)
                .filter(|obj| obj.len() == 1)
                .and_then(|obj| obj.get("json_schema"))
                .and_then(Value::as_str);

            if !ann_matches
                || child_spec.get("type").and_then(Value::as_str) != Some("object")
                || !is_truthy(child_spec.get("_image_row_proposal"))
                || is_truthy(child_spec.get("knowledge"))
                || !is_truthy(row.get("valid"))
                || !is_truthy(row.get("complete"))
                || !row_value_all_null
                || already_selected
                || task.is_none_or(|t| !t.has_media)
            {
                all_conflicts_recoverable = false;
                break;
            }
            let Some(schema_str) = native_schema_str else {
                all_conflicts_recoverable = false;
                break;
            };
            let Ok(parsed_schema) = serde_json::from_str::<Value>(schema_str) else {
                all_conflicts_recoverable = false;
                break;
            };
            let Ok(expected_schema) = ar::schema(child_spec) else {
                all_conflicts_recoverable = false;
                break;
            };
            if !json_validate::json_equal(&parsed_schema, &expected_schema) {
                all_conflicts_recoverable = false;
                break;
            }
            let Some(anyof_schema) = build_non_null_text_anyof_schema(&parsed_schema, &text_keys)
            else {
                all_conflicts_recoverable = false;
                break;
            };
            let Ok(anyof_json) = serde_json::to_string(&anyof_schema) else {
                all_conflicts_recoverable = false;
                break;
            };
            local_constraints.insert(child.clone(), json!({"json_schema": anyof_json}));
            children.push(child.clone());
        }

        if all_conflicts_recoverable {
            for (k, v) in local_constraints {
                constraints.insert(k.clone(), v.clone());
                retry_constraints.insert(k, v);
            }
            for child in &children {
                if let Some(task) = task_map.get(child.as_str()) {
                    retry_input_lengths.insert(child.clone(), task.input_len);
                }
                retry_task_names.push(child.clone());
            }
            contexts.push(RowOmissionContext {
                parent: parent.clone(),
                queries,
                labels,
                text_keys,
                selected_indices,
                conflicts,
                children,
            });
        }
    }

    if retry_task_names.is_empty() {
        return None;
    }

    let mut repair_record = record.clone();
    if let Some(obj) = repair_record.as_object_mut() {
        obj.insert("_ar_constraints".to_string(), Value::Object(constraints));
        obj.insert("_ar_prefix".to_string(), Value::Null);
        obj.insert("_ar_group_prefixes".to_string(), Value::Array(Vec::new()));
        obj.insert("_ar_media_groups".to_string(), Value::Array(Vec::new()));
        obj.insert("_null_recovery".to_string(), Value::Bool(true));
        obj.insert("_row_omission_recovery".to_string(), Value::Bool(true));
    }

    Some(RowOmissionPlan {
        repair_record,
        retry_task_names,
        retry_constraints,
        retry_input_lengths,
        contexts,
    })
}

/// Merge row-omission retry alternatives back into `predictions` only when all
/// conflicts resolve while preserving every previously matched proposal slot
/// (`row_omission.py::merge`).
pub fn merge(
    predictions: &Map<String, Value>,
    alternatives: &Map<String, Value>,
    plan: &RowOmissionPlan,
) -> (Map<String, Value>, Vec<Value>) {
    let mut result = predictions.clone();
    let mut evidence: Vec<Value> = Vec::with_capacity(plan.contexts.len());

    for context in &plan.contexts {
        let mut candidates: Map<String, Value> = Map::new();
        for child in &context.children {
            let Some(original) = predictions.get(child).and_then(Value::as_object) else {
                continue;
            };
            let alternative = alternatives.get(child).and_then(Value::as_object);
            let orig_tokens = original.get("tokens").and_then(Value::as_u64).unwrap_or(0);
            let alt_tokens = alternative
                .and_then(|a| a.get("tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let orig_retry_in = original
                .get("retry_input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let added_in = plan.retry_input_lengths.get(child).copied().unwrap_or(0) as u64;
            let total_tokens = orig_tokens + alt_tokens;
            let total_retry_in = orig_retry_in + added_in;

            let mut accounted_orig = original.clone();
            accounted_orig.insert("tokens".to_string(), json!(total_tokens));
            accounted_orig.insert("retry_input_tokens".to_string(), json!(total_retry_in));
            result.insert(child.clone(), Value::Object(accounted_orig));

            if let Some(alt) = alternative {
                let alt_valid = is_truthy(alt.get("valid"));
                let alt_complete = is_truthy(alt.get("complete"));
                let alt_non_null = alt
                    .get("value")
                    .and_then(Value::as_object)
                    .is_some_and(|obj| obj.values().any(|v| !v.is_null()));
                if alt_valid && alt_complete && alt_non_null {
                    let mut cand = alt.clone();
                    cand.insert("tokens".to_string(), json!(total_tokens));
                    cand.insert("retry_input_tokens".to_string(), json!(total_retry_in));
                    cand.insert(
                        "image_row_omission_recovered".to_string(),
                        Value::Bool(true),
                    );
                    candidates.insert(child.clone(), Value::Object(cand));
                }
            }
        }

        let mut trial = result.clone();
        for (k, v) in &candidates {
            trial.insert(k.clone(), v.clone());
        }
        let trial_proposals: Vec<Value> = context
            .queries
            .iter()
            .filter_map(|name| trial.get(name).cloned())
            .collect();
        let (new_selected, new_conflicts) =
            image_rows::select_indices(&context.labels, &trial_proposals, &context.text_keys);
        let unchanged = context
            .selected_indices
            .iter()
            .enumerate()
            .all(|(idx, &prev)| prev.is_none() || new_selected.get(idx).copied().flatten() == prev);
        let accepted =
            new_conflicts.is_empty() && unchanged && candidates.len() == context.children.len();
        if accepted {
            for (k, v) in candidates {
                result.insert(k, v);
            }
        }
        evidence.push(json!({
            "parent": context.parent,
            "original_conflicts": context.conflicts,
            "retry_children": context.children,
            "conflicts_after_retry": new_conflicts,
            "previously_matched_proposals_preserved_exactly": unchanged,
            "accepted": accepted,
        }));
    }

    (result, evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record_and_tasks() -> (Value, Vec<TaskInfo>) {
        let child_spec = json!({
            "type": "object",
            "nullable": false,
            "_image_row_proposal": true,
            "properties": {
                "description": {"type": "text"},
                "code": {"type": "text"},
                "amount": {"type": "number"}
            }
        });
        let child_schema_str = serde_json::to_string(&ar::schema(&child_spec).unwrap()).unwrap();
        let record = json!({
            "fields": {
                "items": {
                    "type": "list",
                    "nullable": true,
                    "_image_row_inventory": true,
                    "max_items": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "text"},
                            "code": {"type": "text"},
                            "amount": {"type": "number"}
                        }
                    }
                },
                "items.item1": child_spec.clone(),
                "items.item2": child_spec
            },
            "_meta": {
                "extraction_contract": {
                    "fields": {
                        "items": {"image_row_fields": ["items.item1", "items.item2"]},
                        "items.item1": {"parent": "items", "index": 1},
                        "items.item2": {"parent": "items", "index": 2}
                    }
                }
            },
            "_ar_constraints": {
                "items.item1": {"json_schema": child_schema_str.clone()},
                "items.item2": {"json_schema": child_schema_str}
            }
        });
        let tasks = vec![
            TaskInfo {
                name: "items".to_string(),
                input_len: 100,
                has_media: true,
            },
            TaskInfo {
                name: "items.item1".to_string(),
                input_len: 110,
                has_media: true,
            },
            TaskInfo {
                name: "items.item2".to_string(),
                input_len: 120,
                has_media: true,
            },
        ];
        (record, tasks)
    }

    #[test]
    fn prepares_anyof_schema_and_merges_recovered_row() {
        let (record, tasks) = sample_record_and_tasks();
        let preds_val = json!({
            "items": {
                "value": [{"label": "Widget"}, {"label": "Bolt"}],
                "valid": true,
                "complete": true,
                "tokens": 12
            },
            "items.item1": {
                "value": {"description": "Widget Pro", "code": "W1", "amount": 10},
                "valid": true,
                "complete": true,
                "tokens": 15
            },
            "items.item2": {
                "value": {"description": null, "code": null, "amount": null},
                "valid": true,
                "complete": true,
                "tokens": 8
            }
        });
        let preds = preds_val.as_object().unwrap();

        let plan = prepare(&record, preds, &tasks).unwrap();
        assert_eq!(plan.retry_task_names, vec!["items.item2"]);
        assert_eq!(plan.repair_record["_row_omission_recovery"], json!(true));
        assert_eq!(plan.repair_record["_null_recovery"], json!(true));

        // Verify the generated anyOf schema is valid JSON Schema and rejects all-null
        let schema_str = plan.retry_constraints["items.item2"]["json_schema"]
            .as_str()
            .unwrap();
        let anyof_schema: Value = serde_json::from_str(schema_str).unwrap();
        assert!(json_validate::check_schema(&anyof_schema).is_ok());
        assert_eq!(anyof_schema["anyOf"].as_array().unwrap().len(), 2);
        assert!(!json_validate::is_valid(
            &anyof_schema,
            &json!({"description": null, "code": null, "amount": null})
        ));
        assert!(json_validate::is_valid(
            &anyof_schema,
            &json!({"description": "Bolt M4", "code": null, "amount": 2})
        ));
        assert!(json_validate::is_valid(
            &anyof_schema,
            &json!({"description": null, "code": "Bolt", "amount": 2})
        ));

        // Merge a valid recovery that matches "Bolt"
        let alts_val = json!({
            "items.item2": {
                "value": {"description": "Bolt M4", "code": "B4", "amount": 2},
                "valid": true,
                "complete": true,
                "tokens": 14
            }
        });
        let (merged, evidence) = merge(preds, alts_val.as_object().unwrap(), &plan);
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0]["accepted"], json!(true));
        assert_eq!(
            merged["items.item2"]["value"]["description"],
            json!("Bolt M4")
        );
        assert_eq!(
            merged["items.item2"]["image_row_omission_recovered"],
            json!(true)
        );
        assert_eq!(merged["items.item2"]["tokens"], json!(22));
        assert_eq!(merged["items.item2"]["retry_input_tokens"], json!(120));
    }

    #[test]
    fn rejects_merge_when_retry_still_conflicts_but_accounts_tokens() {
        let (record, tasks) = sample_record_and_tasks();
        let preds_val = json!({
            "items": {
                "value": [{"label": "Widget"}, {"label": "Bolt"}],
                "valid": true,
                "complete": true,
                "tokens": 12
            },
            "items.item1": {
                "value": {"description": "Widget Pro", "code": "W1", "amount": 10},
                "valid": true,
                "complete": true,
                "tokens": 15
            },
            "items.item2": {
                "value": {"description": null, "code": null, "amount": null},
                "valid": true,
                "complete": true,
                "tokens": 8
            }
        });
        let preds = preds_val.as_object().unwrap();
        let plan = prepare(&record, preds, &tasks).unwrap();

        // Retry returns unrelated text that does not match "Bolt"
        let alts_val = json!({
            "items.item2": {
                "value": {"description": "Unrelated", "code": null, "amount": 5},
                "valid": true,
                "complete": true,
                "tokens": 11
            }
        });
        let (merged, evidence) = merge(preds, alts_val.as_object().unwrap(), &plan);
        assert_eq!(evidence[0]["accepted"], json!(false));
        assert!(merged["items.item2"]["value"]["description"].is_null());
        assert_eq!(merged["items.item2"]["tokens"], json!(19));
        assert_eq!(merged["items.item2"]["retry_input_tokens"], json!(120));
    }

    #[test]
    fn prepare_skips_when_already_recovering_or_conflict_row_not_all_null() {
        let (mut record, tasks) = sample_record_and_tasks();
        let preds_val = json!({
            "items": {
                "value": [{"label": "Widget"}, {"label": "Bolt"}],
                "valid": true,
                "complete": true,
                "tokens": 12
            },
            "items.item1": {
                "value": {"description": "Widget Pro", "code": "W1", "amount": 10},
                "valid": true,
                "complete": true,
                "tokens": 15
            },
            "items.item2": {
                "value": {"description": null, "code": null, "amount": 99},
                "valid": true,
                "complete": true,
                "tokens": 8
            }
        });
        // Non-null amount cell means item2 is NOT an all-null omission -> skip
        assert!(prepare(&record, preds_val.as_object().unwrap(), &tasks).is_none());

        // Already in recovery -> skip
        let all_null_preds = json!({
            "items": preds_val["items"],
            "items.item1": preds_val["items.item1"],
            "items.item2": {
                "value": {"description": null, "code": null, "amount": null},
                "valid": true,
                "complete": true,
                "tokens": 8
            }
        });
        record["_row_omission_recovery"] = json!(true);
        assert!(prepare(&record, all_null_preds.as_object().unwrap(), &tasks).is_none());
    }
}
