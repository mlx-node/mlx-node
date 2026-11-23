//! Autoregressive batch-completion schedulers for scalar lists, object lists,
//! and image-row cohorts.
//!
//! Direct port of upstream:
//! - `clairic_runtime/batch_completion.py` (first-item list gating + 8-row image cohort gating)
//! - `clairic_runtime/ordinary_list_completion.py` (exponential frontier scheduling for Flash text scalar lists)
//! - `clairic_runtime/text_list_completion.py` (27B object-list frontier gating + scalar delegation)

use super::ar;
use super::errors::{inference_error, invalid_request};
use super::image_rows;
use super::json_validate;
use super::schema::{self, EVIDENCE_CONSTRUCTION, MAX_LIST_ITEMS};
use crate::errors::Result;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Image-row cohort window size (`batch_completion.py` cohort = 8).
pub const ROW_COHORT_SIZE: usize = 8;
/// Initial object/text list frontier (`text_list_completion.py::FRONTIER = 8`).
pub const TEXT_LIST_FRONTIER: usize = 8;

/// Descriptor for one AR task in a batch passed to the completion schedulers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArTaskDescriptor {
    pub name: String,
    pub rid: String,
    pub input_ids: Vec<u32>,
    pub has_media: bool,
}

/// Output of one AR task returned by the inference engine callback (or synthesized
/// when a scheduler omits an unexecuted tail slot).
#[derive(Debug, Clone, PartialEq)]
pub struct ArBatchItemResult {
    /// Request/task identity matching [`ArTaskDescriptor::rid`].
    pub id: String,
    /// Generated output token IDs (including trailing EOS if stopped on EOS).
    pub output_ids: Option<Vec<u32>>,
    /// Decoded completion text (used as fallback when `output_ids` is `None`).
    pub text: String,
    /// Number of completion tokens generated.
    pub completion_tokens: usize,
    /// Finish reason type (e.g. `Some("stop")` or `Some("length")`).
    pub finish_reason_type: Option<String>,
    /// Matched stop token ID when `finish_reason_type == Some("stop")`.
    pub finish_reason_matched: Option<u32>,
    /// Per-token `(logprob, token_id)` pairs for sequence confidence calculation.
    pub output_token_logprobs: Vec<(f64, u32)>,
    /// First-step tracked token `(logprob, token_id)` pairs (e.g., `"null"`, `"false"`).
    pub output_token_ids_logprobs: Vec<Vec<(f64, u32)>>,
    /// True when this slot was omitted by a completion scheduler (`_clair_unused_row_proposal`).
    pub unused_row_proposal: bool,
    /// Scheduler-specific omission marker (e.g. `"_first_item_unexecuted"`).
    pub omission_marker: Option<String>,
}

impl ArBatchItemResult {
    /// Construct an omitted placeholder slot matching upstream `_clair_unused_row_proposal`.
    pub fn omitted(id: impl Into<String>, marker: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            output_ids: None,
            text: String::new(),
            completion_tokens: 0,
            finish_reason_type: None,
            finish_reason_matched: None,
            output_token_logprobs: Vec::new(),
            output_token_ids_logprobs: Vec::new(),
            unused_row_proposal: true,
            omission_marker: Some(marker.into()),
        }
    }
}

/// Cohort metadata for an image-row inventory list (`batch_completion.py::_row_cohorts`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowCohort {
    pub inventory: usize,
    pub proposals: Vec<usize>,
    pub text_keys: Vec<String>,
}

/// Callback parsing a raw AR batch item into a structured prediction JSON value.
pub type ParseResultFn<'a> = &'a dyn Fn(&ArBatchItemResult, &Value, &str) -> Value;
/// Callback normalizing retained predictions (`printed_temporal` + `printed_numbers`).
pub type NormalizePredictionsFn<'a> = &'a dyn Fn(&Value, &Map<String, Value>) -> Map<String, Value>;
/// Callback decoding generated token IDs into raw text.
pub type DecodeTokensFn<'a> = &'a dyn Fn(&[u32]) -> Result<String>;

/// Frontier endpoints for [`ordinary_list_generate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryListFrontiers {
    pub initial_frontier: usize,
    pub next_frontier: usize,
}

impl OrdinaryListFrontiers {
    /// Disabled frontier scheduling (`initial_frontier = 0`, delegates to [`batch_generate`]).
    pub const DISABLED: Self = Self {
        initial_frontier: 0,
        next_frontier: 0,
    };
    /// Default Flash text-only scalar list frontier (`32 -> 64 -> 128...`).
    pub const FLASH_DEFAULT: Self = Self {
        initial_frontier: 32,
        next_frontier: 64,
    };
    /// Delegated frontier used by [`text_list_generate`] (`8 -> 32 -> 64...`).
    pub const TEXT_LIST_DELEGATED: Self = Self {
        initial_frontier: 8,
        next_frontier: 32,
    };

    pub const fn new(initial_frontier: usize, next_frontier: usize) -> Self {
        Self {
            initial_frontier,
            next_frontier,
        }
    }
}

/// Optional parser/normalizer hooks used by image-row cohort gating (`_cohort_proof`).
pub struct CohortHooks<'a> {
    pub parse_result: ParseResultFn<'a>,
    pub normalize: NormalizePredictionsFn<'a>,
}

/// Build the synthetic unused prediction record for an omitted slot
/// (`batch_completion.py::unused_prediction`).
pub fn unused_prediction(result: &ArBatchItemResult) -> Value {
    json!({
        "value": null,
        "valid": false,
        "complete": false,
        "tokens": result.completion_tokens,
        "_unused_row_proposal": true,
    })
}

/// Return `true` iff `result` stopped cleanly on `eos` with text `"null"`
/// (`batch_completion.py::_empty_inventory`).
pub fn empty_inventory(
    result: &ArBatchItemResult,
    eos: u32,
    decode_tokens: &dyn Fn(&[u32]) -> Result<String>,
) -> bool {
    if result.finish_reason_type.as_deref() != Some("stop")
        || result.finish_reason_matched != Some(eos)
    {
        return false;
    }
    if let Some(ref ids) = result.output_ids {
        let trimmed = if ids.last() == Some(&eos) {
            &ids[..ids.len() - 1]
        } else {
            ids.as_slice()
        };
        return decode_tokens(trimmed).is_ok_and(|text| text.trim() == "null");
    }
    let text = result.text.strip_suffix(ar::EOS).unwrap_or(&result.text);
    text.trim() == "null"
}

/// Strict null-completion check required for gating tail items
/// (`batch_completion.py::_completed_null` / `text_list_completion.py::completed_null`).
pub fn completed_null(
    result: &ArBatchItemResult,
    eos: u32,
    decode_tokens: &dyn Fn(&[u32]) -> Result<String>,
) -> bool {
    if result.finish_reason_type.as_deref() != Some("stop")
        || result.finish_reason_matched != Some(eos)
    {
        return false;
    }
    let Some(ref ids) = result.output_ids else {
        return false;
    };
    if ids.is_empty() || ids.last() != Some(&eos) || result.completion_tokens != ids.len() {
        return false;
    }
    decode_tokens(&ids[..ids.len() - 1]).is_ok_and(|text| text.trim() == "null")
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

fn source_row_gates_set(record: &Value) -> HashSet<&str> {
    let mut out = HashSet::new();
    match record.get("_source_row_gates") {
        Some(Value::Array(arr)) => {
            for item in arr {
                if let Some(s) = item.as_str() {
                    out.insert(s);
                }
            }
        }
        Some(Value::Object(map)) => {
            for k in map.keys() {
                out.insert(k.as_str());
            }
        }
        _ => {}
    }
    out
}

fn has_forbidden_marker(spec: &Value) -> bool {
    [
        "_physical_row_scope",
        "_image_row_inventory",
        "_image_row_proposal",
    ]
    .iter()
    .any(|k| is_truthy(spec.get(*k)))
}

fn in_excluded_plans(record: &Value, parent: &str) -> bool {
    ["_financial_line_plans", "_list_source_bounds"]
        .iter()
        .any(|plan_key| {
            record
                .get(*plan_key)
                .and_then(Value::as_object)
                .is_some_and(|m| m.contains_key(parent))
        })
}

fn validate_distinct_task_rids(tasks: &[ArTaskDescriptor], msg: &str) -> Result<()> {
    let mut seen = HashSet::with_capacity(tasks.len());
    for task in tasks {
        if task.rid.is_empty() || !seen.insert(task.rid.as_str()) {
            return Err(invalid_request(msg));
        }
    }
    Ok(())
}

/// Identify ordinary scalar list groups (`text`/`number`/`date`) eligible for
/// first-item or frontier gating (`batch_completion.py::_ordinary_lists`).
pub fn ordinary_lists(
    record: &Value,
    tasks: &[ArTaskDescriptor],
) -> Result<BTreeMap<String, Vec<usize>>> {
    if is_truthy(record.get("_null_recovery")) {
        return Ok(BTreeMap::new());
    }
    let Some(fields) = record.get("fields").and_then(Value::as_object) else {
        return Ok(BTreeMap::new());
    };
    let Some(annotations) = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)
    else {
        return Ok(BTreeMap::new());
    };

    let mut indices: HashMap<&str, usize> = HashMap::with_capacity(tasks.len());
    for (idx, task) in tasks.iter().enumerate() {
        if indices.insert(task.name.as_str(), idx).is_some() {
            return Err(invalid_request("AR task names must be distinct"));
        }
    }

    let source_gates = source_row_gates_set(record);
    let mut claimed: HashSet<String> = HashSet::new();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    for (parent, ann_val) in annotations {
        let Some(ann) = ann_val.as_object() else {
            continue;
        };
        let Some(spec) = fields.get(parent) else {
            continue;
        };
        let item_kind = match spec.get("items") {
            Some(Value::Object(obj)) => obj.get("type").and_then(Value::as_str),
            Some(Value::String(s)) => Some(s.as_str()),
            _ => None,
        };
        let Some(queries_arr) = ann.get("query_fields").and_then(Value::as_array) else {
            continue;
        };
        if spec.get("type").and_then(Value::as_str) != Some("list")
            || !matches!(item_kind, Some("text" | "number" | "date"))
            || is_truthy(spec.get("knowledge"))
            || ann.len() != 1
            || queries_arr.len() < 2
            || has_forbidden_marker(spec)
            || indices.contains_key(parent.as_str())
            || in_excluded_plans(record, parent)
        {
            continue;
        }

        let mut valid = true;
        let mut group_indices = Vec::with_capacity(queries_arr.len());
        let mut query_names = Vec::with_capacity(queries_arr.len());

        for (i, q_val) in queries_arr.iter().enumerate() {
            let ordinal = i + 1;
            let Some(name) = q_val.as_str() else {
                valid = false;
                break;
            };
            let Some(&task_idx) = indices.get(name) else {
                valid = false;
                break;
            };
            let Some(child) = fields.get(name) else {
                valid = false;
                break;
            };
            let expected_owner = json!({"parent": parent, "index": ordinal});
            let owner_ok = annotations
                .get(name)
                .is_some_and(|o| json_validate::json_equal(o, &expected_owner));
            let nullable_is_true = matches!(child.get("nullable"), None | Some(Value::Bool(true)));
            let enum_ok = match child.get("enum") {
                None => true,
                Some(Value::Array(arr)) => arr.iter().any(Value::is_null),
                _ => false,
            };
            if name != schema::list_item_name(parent, ordinal)
                || claimed.contains(name)
                || !owner_ok
                || !matches!(
                    child.get("type").and_then(Value::as_str),
                    Some("text" | "number" | "date")
                )
                || !nullable_is_true
                || has_forbidden_marker(child)
                || source_gates.contains(name)
                || !enum_ok
            {
                valid = false;
                break;
            }
            group_indices.push(task_idx);
            query_names.push(name.to_string());
        }

        if valid {
            for q in query_names {
                claimed.insert(q);
            }
            groups.insert(parent.clone(), group_indices);
        }
    }

    Ok(groups)
}

/// Identify image-row cohorts eligible for 8-row cohort gating
/// (`batch_completion.py::_row_cohorts`).
pub fn row_cohorts(record: &Value, tasks: &[ArTaskDescriptor]) -> BTreeMap<String, RowCohort> {
    let mut result = BTreeMap::new();
    let Some(fields) = record.get("fields").and_then(Value::as_object) else {
        return result;
    };
    let Some(annotations) = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)
    else {
        return result;
    };
    let indices: HashMap<&str, usize> = tasks
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();

    for (parent, ann_val) in annotations {
        let Some(ann) = ann_val.as_object() else {
            continue;
        };
        if ann.len() != 1 || !ann.contains_key("image_row_fields") {
            continue;
        }
        let Some(spec) = fields.get(parent) else {
            continue;
        };
        let Some(&inv_idx) = indices.get(parent.as_str()) else {
            continue;
        };
        let Some(queries_arr) = ann.get("image_row_fields").and_then(Value::as_array) else {
            continue;
        };
        if !is_truthy(spec.get("_image_row_inventory"))
            || queries_arr.len() <= ROW_COHORT_SIZE
            || in_excluded_plans(record, parent)
            || spec
                .get("items")
                .and_then(Value::as_object)
                .and_then(|i| i.get("type"))
                .and_then(Value::as_str)
                != Some("object")
        {
            continue;
        }
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
        if text_keys.is_empty() {
            continue;
        }
        let mut proposals = Vec::with_capacity(queries_arr.len());
        let mut valid = true;
        for (i, q_val) in queries_arr.iter().enumerate() {
            let ordinal = i + 1;
            let Some(name) = q_val.as_str() else {
                valid = false;
                break;
            };
            let Some(&idx) = indices.get(name) else {
                valid = false;
                break;
            };
            let expected_owner = json!({"parent": parent, "index": ordinal});
            let owner_ok = annotations
                .get(name)
                .is_some_and(|o| json_validate::json_equal(o, &expected_owner));
            let child_ok = fields.get(name).is_some_and(|c| {
                c.get("type").and_then(Value::as_str) == Some("object")
                    && is_truthy(c.get("_image_row_proposal"))
            });
            if name != schema::list_item_name(parent, ordinal) || !owner_ok || !child_ok {
                valid = false;
                break;
            }
            proposals.push(idx);
        }
        if valid {
            result.insert(
                parent.clone(),
                RowCohort {
                    inventory: inv_idx,
                    proposals,
                    text_keys,
                },
            );
        }
    }
    result
}

fn normalize_retained(
    record: &Value,
    predictions: &Map<String, Value>,
    normalize: NormalizePredictionsFn<'_>,
) -> Map<String, Value> {
    let mut view = record.clone();
    if let Some(meta) = view.get_mut("_meta").and_then(Value::as_object_mut) {
        for kind in ["printed_temporal", "printed_numbers"] {
            let filtered = meta
                .get(kind)
                .and_then(Value::as_object)
                .map(|m| {
                    let mut out = Map::new();
                    for (name, spec) in m {
                        if predictions.contains_key(name) {
                            out.insert(name.clone(), spec.clone());
                        }
                    }
                    out
                })
                .unwrap_or_default();
            meta.insert(kind.to_string(), Value::Object(filtered));
        }
    }
    normalize(&view, predictions)
}

/// Verify whether proposals `1..=len(labels)` prove zero-conflict same-ordinal
/// alignment with a completed inventory (`batch_completion.py::_cohort_proof`).
pub fn cohort_proof(
    record: &Value,
    tasks: &[ArTaskDescriptor],
    results: &[Option<ArBatchItemResult>],
    parent: &str,
    cohort: &RowCohort,
    hooks: &CohortHooks<'_>,
) -> Option<Value> {
    let fields = record.get("fields")?.as_object()?;
    let spec = fields.get(parent)?;
    let source_input = record.get("input").and_then(Value::as_str).unwrap_or("");
    let native = results.get(cohort.inventory)?.as_ref()?;
    let inventory = (hooks.parse_result)(native, spec, source_input);
    if !is_truthy(inventory.get("valid")) || !is_truthy(inventory.get("complete")) {
        return None;
    }
    let inv_value = inventory.get("value").cloned().unwrap_or(Value::Null);
    let mut labels = Vec::new();
    if !inv_value.is_null() {
        let rows = inv_value.as_array()?;
        for row in rows {
            let obj = row.as_object()?;
            if obj.len() != 1 {
                return None;
            }
            let label = obj.get("label")?.as_str()?;
            labels.push(label.to_string());
        }
    }
    let max_items = spec
        .get("max_items")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(MAX_LIST_ITEMS);
    if labels.len() > ROW_COHORT_SIZE || labels.len() > max_items {
        return None;
    }

    let mut retained = Map::new();
    retained.insert(parent.to_string(), inventory);
    for &idx in cohort.proposals.iter().take(labels.len()) {
        let task = tasks.get(idx)?;
        let res = results.get(idx)?.as_ref()?;
        if res.unused_row_proposal {
            return None;
        }
        let child_spec = fields.get(&task.name)?;
        let parsed = (hooks.parse_result)(res, child_spec, source_input);
        retained.insert(task.name.clone(), parsed);
    }

    let normalized = normalize_retained(record, &retained, hooks.normalize);
    let norm_inv = normalized.get(parent)?;
    if !is_truthy(norm_inv.get("valid"))
        || !is_truthy(norm_inv.get("complete"))
        || !json_validate::json_equal(norm_inv.get("value").unwrap_or(&Value::Null), &inv_value)
    {
        return None;
    }

    let mut proposals = Vec::with_capacity(labels.len());
    for &idx in cohort.proposals.iter().take(labels.len()) {
        let name = &tasks[idx].name;
        let parsed = normalized.get(name)?;
        let non_null_obj = parsed
            .get("value")
            .and_then(Value::as_object)
            .is_some_and(|obj| obj.values().any(|v| !v.is_null()));
        if !is_truthy(parsed.get("valid")) || !is_truthy(parsed.get("complete")) || !non_null_obj {
            return None;
        }
        proposals.push(parsed.clone());
    }

    let (selected_indices, conflicts) =
        image_rows::select_indices(&labels, &proposals, &cohort.text_keys);
    if !conflicts.is_empty()
        || selected_indices
            .iter()
            .enumerate()
            .any(|(idx, &sel)| sel != Some(idx))
    {
        return None;
    }

    let selected_ordinals: Vec<usize> = (1..=labels.len()).collect();
    Some(json!({
        "inventory_labels": labels,
        "selected_ordinals": selected_ordinals,
        "assignment_cost": 0,
        "cohort": ROW_COHORT_SIZE,
        "original_proposal_capacity": cohort.proposals.len(),
        "inventory_valid_complete": true,
        "same_ordinal_matches": true,
    }))
}

/// First-item scalar-list gating + 8-row image cohort gating scheduler
/// (`batch_completion.py::generate`).
pub fn batch_generate<F>(
    record: &mut Value,
    tasks: &[ArTaskDescriptor],
    eos: u32,
    decode_tokens: &dyn Fn(&[u32]) -> Result<String>,
    cohort_hooks: Option<&CohortHooks<'_>>,
    mut run_batch: F,
) -> Result<Vec<ArBatchItemResult>>
where
    F: FnMut(&[usize]) -> Result<Vec<ArBatchItemResult>>,
{
    let size = tasks.len();
    let all_indices: Vec<usize> = (0..size).collect();
    if !is_truthy(record.get("_first_item_list_gating")) || is_truthy(record.get("_null_recovery"))
    {
        return run_batch(&all_indices);
    }

    let groups = ordinary_lists(record, tasks)?;
    let cohorts = if cohort_hooks.is_some() {
        row_cohorts(record, tasks)
    } else {
        BTreeMap::new()
    };
    if groups.is_empty() && cohorts.is_empty() {
        return run_batch(&all_indices);
    }

    validate_distinct_task_rids(
        tasks,
        "Cohort scheduling requires the original distinct AR batch",
    )?;

    let mut tails: HashSet<usize> = HashSet::new();
    for group in groups.values() {
        for &idx in group.iter().skip(1) {
            tails.insert(idx);
        }
    }
    for cohort in cohorts.values() {
        for &idx in cohort.proposals.iter().skip(ROW_COHORT_SIZE) {
            tails.insert(idx);
        }
    }

    let mut results: Vec<Option<ArBatchItemResult>> = vec![None; size];
    let mut submit = |results: &mut [Option<ArBatchItemResult>], indices: &[usize]| -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        let received = run_batch(indices)?;
        if received.len() != indices.len() {
            return Err(inference_error(
                "Native completion differs from its cohort submission",
            ));
        }
        for (&idx, res) in indices.iter().zip(received) {
            if res.id != tasks[idx].rid {
                return Err(inference_error(
                    "Native completion differs from its cohort submission",
                ));
            }
            results[idx] = Some(res);
        }
        Ok(())
    };

    let wave1: Vec<usize> = (0..size).filter(|idx| !tails.contains(idx)).collect();
    submit(&mut results, &wave1)?;

    let mut unexecuted: BTreeMap<String, usize> = BTreeMap::new();
    let mut continued: Vec<usize> = Vec::new();

    for group in groups.values() {
        let first_is_null = results[group[0]]
            .as_ref()
            .is_some_and(|r| completed_null(r, eos, decode_tokens));
        if first_is_null {
            for &idx in group.iter().skip(1) {
                unexecuted.insert(tasks[idx].name.clone(), tasks[idx].input_ids.len());
                results[idx] = Some(ArBatchItemResult::omitted(
                    &tasks[idx].rid,
                    "_first_item_unexecuted",
                ));
            }
        } else {
            continued.extend(group.iter().skip(1).copied());
        }
    }

    let mut proofs: Vec<Value> = Vec::with_capacity(cohorts.len());
    if let Some(hooks) = cohort_hooks {
        for (parent, cohort) in &cohorts {
            let proof = cohort_proof(record, tasks, &results, parent, cohort, hooks);
            let fallback = proof.is_none();
            proofs.push(json!({
                "parent": parent,
                "proof": proof,
                "fallback": fallback,
            }));
            if fallback {
                continued.extend(cohort.proposals.iter().skip(ROW_COHORT_SIZE).copied());
            } else {
                for &idx in cohort.proposals.iter().skip(ROW_COHORT_SIZE) {
                    unexecuted.insert(tasks[idx].name.clone(), tasks[idx].input_ids.len());
                    results[idx] = Some(ArBatchItemResult::omitted(
                        &tasks[idx].rid,
                        "_row_cohort_unexecuted",
                    ));
                }
            }
        }
    }

    continued.sort_unstable();
    submit(&mut results, &continued)?;

    let finalized: Vec<ArBatchItemResult> = results
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| inference_error("Cohort scheduling left an unaccounted result slot"))
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(obj) = record.as_object_mut() {
        let unexec_map: Map<String, Value> =
            unexecuted.into_iter().map(|(k, v)| (k, json!(v))).collect();
        obj.insert(
            "_unexecuted_list_items".to_string(),
            Value::Object(unexec_map),
        );
        obj.insert("_row_cohort_gating".to_string(), Value::Array(proofs));
    }

    Ok(finalized)
}

/// Return `true` iff `record` and `tasks` are eligible for exponential frontier
/// scheduling on text-only scalar lists (`ordinary_list_completion.py::eligible`).
pub fn ordinary_list_eligible(record: &Value, tasks: &[ArTaskDescriptor]) -> bool {
    let has_object_lists = record
        .get("fields")
        .and_then(Value::as_object)
        .is_some_and(|fields| {
            fields.values().any(|spec| {
                spec.get("type").and_then(Value::as_str) == Some("list")
                    && spec
                        .get("items")
                        .and_then(Value::as_object)
                        .and_then(|i| i.get("type"))
                        .and_then(Value::as_str)
                        == Some("object")
            })
        });
    let construction = record
        .get("_meta")
        .and_then(|m| m.get("construction"))
        .and_then(Value::as_str);
    let guarded = !is_truthy(record.get("_null_recovery"))
        && !has_object_lists
        && construction == Some(EVIDENCE_CONSTRUCTION)
        && !is_truthy(record.get("_media_payload"))
        && !is_truthy(record.get("_original_images"))
        && tasks.iter().all(|t| !t.has_media);

    guarded && ordinary_lists(record, tasks).is_ok_and(|g| !g.is_empty())
}

/// Exponential frontier scheduler for Flash text-only scalar lists
/// (`ordinary_list_completion.py::generate`, e.g. `initial_frontier = 32`,
/// `next_frontier = 64`, doubling until a frontier contains `null`).
pub fn ordinary_list_generate<F>(
    record: &mut Value,
    tasks: &[ArTaskDescriptor],
    eos: u32,
    frontiers: OrdinaryListFrontiers,
    decode_tokens: DecodeTokensFn<'_>,
    cohort_hooks: Option<&CohortHooks<'_>>,
    mut run_batch: F,
) -> Result<Vec<ArBatchItemResult>>
where
    F: FnMut(&[usize]) -> Result<Vec<ArBatchItemResult>>,
{
    let initial_frontier = frontiers.initial_frontier;
    let next_frontier = frontiers.next_frontier;
    let enabled = initial_frontier != 0;
    if enabled && next_frontier <= initial_frontier {
        return Err(invalid_request(
            "Ordinary list scheduling requires positive increasing integer endpoints",
        ));
    }

    let groups = if enabled && ordinary_list_eligible(record, tasks) {
        ordinary_lists(record, tasks)?
    } else {
        BTreeMap::new()
    };
    if groups.is_empty() {
        return batch_generate(record, tasks, eos, decode_tokens, cohort_hooks, run_batch);
    }

    validate_distinct_task_rids(
        tasks,
        "Ordinary list scheduling requires original distinct task bindings",
    )?;

    let size = tasks.len();
    let mut results: Vec<Option<ArBatchItemResult>> = vec![None; size];
    let mut unexecuted: BTreeMap<String, usize> = BTreeMap::new();

    let mut submit = |results: &mut [Option<ArBatchItemResult>], indices: &[usize]| -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        let received = run_batch(indices)?;
        if received.len() != indices.len() {
            return Err(inference_error(
                "Ordinary list scheduling received wrong batch size",
            ));
        }
        for (&idx, res) in indices.iter().zip(received) {
            if res.id != tasks[idx].rid {
                return Err(inference_error(
                    "Ordinary list scheduling received wrong task identity",
                ));
            }
            results[idx] = Some(res);
        }
        Ok(())
    };

    let mut tails: HashSet<usize> = HashSet::new();
    for group in groups.values() {
        for &idx in group.iter().skip(initial_frontier) {
            tails.insert(idx);
        }
    }
    let wave1: Vec<usize> = (0..size).filter(|idx| !tails.contains(idx)).collect();
    submit(&mut results, &wave1)?;

    let mut active = groups;
    let mut start = 0usize;
    let mut end = initial_frontier;
    let mut following = next_frontier;

    while !active.is_empty() {
        let mut continued: Vec<usize> = Vec::new();
        let active_keys: Vec<String> = active.keys().cloned().collect();
        for parent in active_keys {
            let Some(group) = active.get(&parent).cloned() else {
                continue;
            };
            let slice_start = start.min(group.len());
            let slice_end = end.min(group.len());
            let stopped = group[slice_start..slice_end].iter().any(|&idx| {
                results[idx]
                    .as_ref()
                    .is_some_and(|r| completed_null(r, eos, decode_tokens))
            });
            if stopped {
                for &idx in &group[slice_end..] {
                    unexecuted.insert(tasks[idx].name.clone(), tasks[idx].input_ids.len());
                    results[idx] = Some(ArBatchItemResult::omitted(
                        &tasks[idx].rid,
                        "_ordinary_list_unexecuted",
                    ));
                }
                active.remove(&parent);
            } else if end >= group.len() {
                active.remove(&parent);
            } else {
                let next_end = following.min(group.len());
                continued.extend_from_slice(&group[slice_end..next_end]);
            }
        }
        if !continued.is_empty() {
            continued.sort_unstable();
            submit(&mut results, &continued)?;
        }
        start = end;
        end = following;
        following = following.saturating_mul(2);
    }

    let finalized: Vec<ArBatchItemResult> = results
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| {
                inference_error("Ordinary list scheduling left an unaccounted result")
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(obj) = record.as_object_mut() {
        let unexec_map: Map<String, Value> =
            unexecuted.into_iter().map(|(k, v)| (k, json!(v))).collect();
        obj.insert(
            "_unexecuted_list_items".to_string(),
            Value::Object(unexec_map),
        );
    }

    Ok(finalized)
}

/// Identify object list groups with `> TEXT_LIST_FRONTIER` items eligible for
/// frontier gating (`text_list_completion.py::_object_lists`).
pub fn object_lists(
    record: &Value,
    tasks: &[ArTaskDescriptor],
) -> Result<BTreeMap<String, Vec<usize>>> {
    if is_truthy(record.get("_null_recovery")) {
        return Ok(BTreeMap::new());
    }
    let Some(fields) = record.get("fields").and_then(Value::as_object) else {
        return Ok(BTreeMap::new());
    };
    let Some(annotations) = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)
    else {
        return Ok(BTreeMap::new());
    };

    let mut indices: HashMap<&str, usize> = HashMap::with_capacity(tasks.len());
    for (idx, task) in tasks.iter().enumerate() {
        if indices.insert(task.name.as_str(), idx).is_some() {
            return Err(invalid_request("AR task names must be distinct"));
        }
    }

    let source_gates = source_row_gates_set(record);
    let mut claimed: HashSet<String> = HashSet::new();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    for (parent, ann_val) in annotations {
        let Some(ann) = ann_val.as_object() else {
            continue;
        };
        let Some(spec) = fields.get(parent) else {
            continue;
        };
        let item_is_object = spec
            .get("items")
            .and_then(Value::as_object)
            .and_then(|i| i.get("type"))
            .and_then(Value::as_str)
            == Some("object");
        let Some(queries_arr) = ann.get("query_fields").and_then(Value::as_array) else {
            continue;
        };
        if spec.get("type").and_then(Value::as_str) != Some("list")
            || !item_is_object
            || is_truthy(spec.get("knowledge"))
            || ann.len() != 1
            || queries_arr.len() <= TEXT_LIST_FRONTIER
            || has_forbidden_marker(spec)
            || indices.contains_key(parent.as_str())
            || in_excluded_plans(record, parent)
        {
            continue;
        }

        let mut valid = true;
        let mut group_indices = Vec::with_capacity(queries_arr.len());
        let mut query_names = Vec::with_capacity(queries_arr.len());

        for (i, q_val) in queries_arr.iter().enumerate() {
            let ordinal = i + 1;
            let Some(name) = q_val.as_str() else {
                valid = false;
                break;
            };
            let Some(&task_idx) = indices.get(name) else {
                valid = false;
                break;
            };
            let Some(child) = fields.get(name) else {
                valid = false;
                break;
            };
            let expected_owner = json!({"parent": parent, "index": ordinal});
            let owner_ok = annotations
                .get(name)
                .is_some_and(|o| json_validate::json_equal(o, &expected_owner));
            let nullable_is_true = matches!(child.get("nullable"), None | Some(Value::Bool(true)));
            let enum_ok = match child.get("enum") {
                None => true,
                Some(Value::Array(arr)) => arr.iter().any(Value::is_null),
                _ => false,
            };
            if name != schema::list_item_name(parent, ordinal)
                || claimed.contains(name)
                || !owner_ok
                || child.get("type").and_then(Value::as_str) != Some("object")
                || !nullable_is_true
                || has_forbidden_marker(child)
                || source_gates.contains(name)
                || !enum_ok
            {
                valid = false;
                break;
            }
            group_indices.push(task_idx);
            query_names.push(name.to_string());
        }

        if valid {
            for q in query_names {
                claimed.insert(q);
            }
            groups.insert(parent.clone(), group_indices);
        }
    }

    Ok(groups)
}

fn text_list_groups(
    record: &Value,
    tasks: &[ArTaskDescriptor],
) -> Result<BTreeMap<String, Vec<usize>>> {
    let construction = record
        .get("_meta")
        .and_then(|m| m.get("construction"))
        .and_then(Value::as_str);
    if is_truthy(record.get("_null_recovery")) || construction != Some(EVIDENCE_CONSTRUCTION) {
        return Ok(BTreeMap::new());
    }
    let mut groups = object_lists(record, tasks)?;
    if !is_truthy(record.get("_media_payload"))
        && !is_truthy(record.get("_original_images"))
        && tasks.iter().all(|t| !t.has_media)
    {
        for (parent, group) in ordinary_lists(record, tasks)? {
            if group.len() > TEXT_LIST_FRONTIER {
                groups.insert(parent, group);
            }
        }
    }
    Ok(groups)
}

/// Return `true` iff `record` and `tasks` are eligible for 27B object/text list
/// frontier gating (`text_list_completion.py::eligible`).
pub fn text_list_eligible(record: &Value, tasks: &[ArTaskDescriptor]) -> bool {
    text_list_groups(record, tasks).is_ok_and(|g| !g.is_empty())
}

/// 27B object-list frontier gating (`FRONTIER = 8`) + text-only scalar list
/// delegation (`8 -> 32 -> 64...`) (`text_list_completion.py::generate`).
pub fn text_list_generate<F>(
    record: &mut Value,
    tasks: &[ArTaskDescriptor],
    eos: u32,
    decode_tokens: &dyn Fn(&[u32]) -> Result<String>,
    cohort_hooks: Option<&CohortHooks<'_>>,
    mut run_batch: F,
) -> Result<Vec<ArBatchItemResult>>
where
    F: FnMut(&[usize]) -> Result<Vec<ArBatchItemResult>>,
{
    if text_list_eligible(record, tasks) && ordinary_list_eligible(record, tasks) {
        return ordinary_list_generate(
            record,
            tasks,
            eos,
            OrdinaryListFrontiers::TEXT_LIST_DELEGATED,
            decode_tokens,
            cohort_hooks,
            run_batch,
        );
    }
    if !text_list_eligible(record, tasks) {
        return batch_generate(record, tasks, eos, decode_tokens, cohort_hooks, run_batch);
    }

    let groups = text_list_groups(record, tasks)?;
    validate_distinct_task_rids(
        tasks,
        "List frontier requires the original distinct AR batch",
    )?;

    let size = tasks.len();
    let mut deferred: HashSet<usize> = HashSet::new();
    for group in groups.values() {
        for &idx in group.iter().skip(TEXT_LIST_FRONTIER) {
            deferred.insert(idx);
        }
    }

    let mut results: Vec<Option<ArBatchItemResult>> = vec![None; size];
    let lengths: HashMap<&str, usize> = tasks
        .iter()
        .map(|t| (t.name.as_str(), t.input_ids.len()))
        .collect();
    let mut unexecuted: BTreeMap<String, usize> = BTreeMap::new();

    let mut submit = |record_ref: &mut Value,
                      results: &mut [Option<ArBatchItemResult>],
                      unexecuted: &mut BTreeMap<String, usize>,
                      indices: &[usize]|
     -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        let subset: Vec<ArTaskDescriptor> = indices.iter().map(|&i| tasks[i].clone()).collect();
        if let Some(obj) = record_ref.as_object_mut() {
            obj.remove("_unexecuted_list_items");
        }
        let received = batch_generate(
            record_ref,
            &subset,
            eos,
            decode_tokens,
            cohort_hooks,
            |sub_indices: &[usize]| {
                let mapped: Vec<usize> = sub_indices.iter().map(|&j| indices[j]).collect();
                run_batch(&mapped)
            },
        )?;
        let accounted: HashMap<String, usize> = record_ref
            .get("_unexecuted_list_items")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n as usize)))
                    .collect()
            })
            .unwrap_or_default();

        for (name, &len) in &accounted {
            if lengths.get(name.as_str()).copied() != Some(len) {
                return Err(inference_error(
                    "Shared scheduler changed original omission accounting",
                ));
            }
        }
        if received.len() != indices.len() {
            return Err(inference_error(
                "Native completion differs from its list-frontier submission",
            ));
        }
        for (&idx, res) in indices.iter().zip(received) {
            let name = &tasks[idx].name;
            if accounted.contains_key(name) {
                if !res.unused_row_proposal || res.completion_tokens != 0 {
                    return Err(inference_error(
                        "Shared scheduler omission lacks its unexecuted result slot",
                    ));
                }
            } else if res.id != tasks[idx].rid {
                return Err(inference_error(
                    "Native completion differs from its list-frontier submission",
                ));
            }
            results[idx] = Some(res);
        }
        for (name, len) in accounted {
            if let Some(&prev) = unexecuted.get(&name)
                && prev != len
            {
                return Err(inference_error(
                    "Schedulers disagree on original omitted prompt length",
                ));
            }
            unexecuted.insert(name, len);
        }
        Ok(())
    };

    let wave1: Vec<usize> = (0..size).filter(|idx| !deferred.contains(idx)).collect();
    submit(record, &mut results, &mut unexecuted, &wave1)?;

    let mut continued: Vec<usize> = Vec::new();
    for group in groups.values() {
        let stopped = group.iter().take(TEXT_LIST_FRONTIER).any(|&idx| {
            results[idx]
                .as_ref()
                .is_some_and(|r| completed_null(r, eos, decode_tokens))
        });
        if stopped {
            for &idx in group.iter().skip(TEXT_LIST_FRONTIER) {
                let name = &tasks[idx].name;
                if unexecuted.contains_key(name) {
                    return Err(inference_error(
                        "List frontier attempted to omit a slot twice",
                    ));
                }
                unexecuted.insert(name.clone(), tasks[idx].input_ids.len());
                results[idx] = Some(ArBatchItemResult::omitted(
                    &tasks[idx].rid,
                    "_text_frontier_unexecuted",
                ));
            }
        } else {
            continued.extend(group.iter().skip(TEXT_LIST_FRONTIER).copied());
        }
    }

    continued.sort_unstable();
    submit(record, &mut results, &mut unexecuted, &continued)?;

    let finalized: Vec<ArBatchItemResult> = results
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| inference_error("List frontier left an unaccounted result slot"))
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(obj) = record.as_object_mut() {
        let unexec_map: Map<String, Value> =
            unexecuted.into_iter().map(|(k, v)| (k, json!(v))).collect();
        obj.insert(
            "_unexecuted_list_items".to_string(),
            Value::Object(unexec_map),
        );
    }

    Ok(finalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: u32 = 999;
    const TOK_NULL: u32 = 1;
    const TOK_VAL: u32 = 2;

    fn mock_decode(ids: &[u32]) -> Result<String> {
        match ids {
            [TOK_NULL] => Ok("null".to_string()),
            [TOK_VAL] => Ok("\"val\"".to_string()),
            _ => Ok(format!("{ids:?}")),
        }
    }

    fn make_result(rid: &str, token: u32) -> ArBatchItemResult {
        ArBatchItemResult {
            id: rid.to_string(),
            output_ids: Some(vec![token, EOS]),
            text: if token == TOK_NULL {
                "null".to_string()
            } else {
                "\"val\"".to_string()
            },
            completion_tokens: 2,
            finish_reason_type: Some("stop".to_string()),
            finish_reason_matched: Some(EOS),
            output_token_logprobs: vec![(-0.05, token), (-0.01, EOS)],
            output_token_ids_logprobs: Vec::new(),
            unused_row_proposal: false,
            omission_marker: None,
        }
    }

    fn scalar_list_fixture(count: usize, gating: bool) -> (Value, Vec<ArTaskDescriptor>) {
        let mut fields = Map::new();
        let mut ann = Map::new();
        let mut query_names = Vec::with_capacity(count);
        let mut tasks = Vec::with_capacity(count);

        fields.insert(
            "tags".to_string(),
            json!({"type": "list", "items": "text", "instructions": "All tags"}),
        );
        for i in 1..=count {
            let name = schema::list_item_name("tags", i);
            fields.insert(name.clone(), json!({"type": "text", "nullable": true}));
            ann.insert(name.clone(), json!({"parent": "tags", "index": i}));
            query_names.push(Value::String(name.clone()));
            tasks.push(ArTaskDescriptor {
                name,
                rid: format!("rid-{i}"),
                input_ids: vec![10; 20 + i],
                has_media: false,
            });
        }
        ann.insert("tags".to_string(), json!({"query_fields": query_names}));
        let record = json!({
            "input": "sample",
            "fields": fields,
            "_first_item_list_gating": gating,
            "_meta": {
                "construction": EVIDENCE_CONSTRUCTION,
                "extraction_contract": {"fields": ann}
            }
        });
        (record, tasks)
    }

    #[test]
    fn batch_completion_first_item_gating_omits_tail_on_null() {
        let (mut record, tasks) = scalar_list_fixture(5, true);
        let mut waves: Vec<Vec<usize>> = Vec::new();
        let results = batch_generate(&mut record, &tasks, EOS, &mock_decode, None, |indices| {
            waves.push(indices.to_vec());
            Ok(indices
                .iter()
                .map(|&i| make_result(&tasks[i].rid, TOK_NULL))
                .collect())
        })
        .unwrap();

        // Only item1 (index 0) ran; items 2..5 were omitted!
        assert_eq!(waves, vec![vec![0]]);
        assert_eq!(results.len(), 5);
        assert!(!results[0].unused_row_proposal);
        for r in &results[1..] {
            assert!(r.unused_row_proposal);
            assert_eq!(r.omission_marker.as_deref(), Some("_first_item_unexecuted"));
            assert_eq!(r.completion_tokens, 0);
        }
        let unexec = record["_unexecuted_list_items"].as_object().unwrap();
        assert_eq!(unexec.len(), 4);
        assert_eq!(unexec["tags.item2"], json!(22));
    }

    #[test]
    fn batch_completion_first_item_gating_continues_when_non_null() {
        let (mut record, tasks) = scalar_list_fixture(4, true);
        let mut waves: Vec<Vec<usize>> = Vec::new();
        let results = batch_generate(&mut record, &tasks, EOS, &mock_decode, None, |indices| {
            waves.push(indices.to_vec());
            Ok(indices
                .iter()
                .map(|&i| make_result(&tasks[i].rid, TOK_VAL))
                .collect())
        })
        .unwrap();

        assert_eq!(waves, vec![vec![0], vec![1, 2, 3]]);
        assert!(results.iter().all(|r| !r.unused_row_proposal));
        assert!(
            record["_unexecuted_list_items"]
                .as_object()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn batch_completion_image_row_cohort_gating_omits_proposals_9_to_n() {
        let proposal_count = 12;
        let mut fields = Map::new();
        let mut ann = Map::new();
        let mut query_names = Vec::new();
        let mut tasks = Vec::new();

        fields.insert(
            "rows".to_string(),
            json!({
                "type": "list",
                "nullable": true,
                "_image_row_inventory": true,
                "max_items": 12,
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "text"},
                        "amount": {"type": "number"}
                    }
                }
            }),
        );
        tasks.push(ArTaskDescriptor {
            name: "rows".to_string(),
            rid: "rid-inv".to_string(),
            input_ids: vec![1; 50],
            has_media: true,
        });

        for i in 1..=proposal_count {
            let name = schema::list_item_name("rows", i);
            fields.insert(
                name.clone(),
                json!({
                    "type": "object",
                    "nullable": false,
                    "_image_row_proposal": true,
                    "properties": {
                        "description": {"type": "text"},
                        "amount": {"type": "number"}
                    }
                }),
            );
            ann.insert(name.clone(), json!({"parent": "rows", "index": i}));
            query_names.push(Value::String(name.clone()));
            tasks.push(ArTaskDescriptor {
                name,
                rid: format!("rid-row-{i}"),
                input_ids: vec![2; 40],
                has_media: true,
            });
        }
        ann.insert("rows".to_string(), json!({"image_row_fields": query_names}));

        let mut record = json!({
            "input": "",
            "fields": fields,
            "_first_item_list_gating": true,
            "_meta": {
                "construction": EVIDENCE_CONSTRUCTION,
                "extraction_contract": {"fields": ann}
            }
        });

        // Mock inventory returns 2 labels ("Alpha", "Beta"), matching proposals 1 and 2.
        let parse_fn = |res: &ArBatchItemResult, _spec: &Value, _src: &str| -> Value {
            let val: Value = serde_json::from_str(&res.text).unwrap_or(Value::Null);
            json!({"value": val, "valid": true, "complete": true})
        };
        let norm_fn =
            |_rec: &Value, preds: &Map<String, Value>| -> Map<String, Value> { preds.clone() };
        let hooks = CohortHooks {
            parse_result: &parse_fn,
            normalize: &norm_fn,
        };

        let mut waves: Vec<Vec<usize>> = Vec::new();
        let results = batch_generate(
            &mut record,
            &tasks,
            EOS,
            &mock_decode,
            Some(&hooks),
            |indices| {
                waves.push(indices.to_vec());
                Ok(indices
                    .iter()
                    .map(|&idx| {
                        let text = match idx {
                            0 => r#"[{"label":"Alpha"},{"label":"Beta"}]"#.to_string(),
                            1 => r#"{"description":"Alpha item","amount":10}"#.to_string(),
                            2 => r#"{"description":"Beta item","amount":20}"#.to_string(),
                            _ => r#"{"description":null,"amount":null}"#.to_string(),
                        };
                        ArBatchItemResult {
                            id: tasks[idx].rid.clone(),
                            output_ids: None,
                            text,
                            completion_tokens: 5,
                            finish_reason_type: Some("stop".to_string()),
                            finish_reason_matched: Some(EOS),
                            output_token_logprobs: Vec::new(),
                            output_token_ids_logprobs: Vec::new(),
                            unused_row_proposal: false,
                            omission_marker: None,
                        }
                    })
                    .collect())
            },
        )
        .unwrap();

        // Only inventory (0) + proposals 1..=8 (indices 1..=8) ran! Proposals 9..=12 omitted!
        assert_eq!(waves, vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8]]);
        for result in &results[9..=12] {
            assert!(result.unused_row_proposal);
            assert_eq!(
                result.omission_marker.as_deref(),
                Some("_row_cohort_unexecuted")
            );
        }
        let proofs = record["_row_cohort_gating"].as_array().unwrap();
        assert_eq!(proofs.len(), 1);
        assert_eq!(proofs[0]["fallback"], json!(false));
        assert_eq!(proofs[0]["proof"]["assignment_cost"], json!(0));
    }

    #[test]
    fn ordinary_list_completion_doubles_frontiers_and_stops_on_null() {
        // 100 items, null appears at index 40 (inside 32..64 frontier) -> 64..100 skipped!
        let (mut record, tasks) = scalar_list_fixture(100, false);
        assert!(ordinary_list_eligible(&record, &tasks));

        let mut waves: Vec<Vec<usize>> = Vec::new();
        let results = ordinary_list_generate(
            &mut record,
            &tasks,
            EOS,
            OrdinaryListFrontiers::FLASH_DEFAULT,
            &mock_decode,
            None,
            |indices| {
                waves.push(indices.to_vec());
                Ok(indices
                    .iter()
                    .map(|&idx| {
                        let tok = if idx >= 40 { TOK_NULL } else { TOK_VAL };
                        make_result(&tasks[idx].rid, tok)
                    })
                    .collect())
            },
        )
        .unwrap();

        assert_eq!(waves.len(), 2);
        assert_eq!(waves[0], (0..32).collect::<Vec<_>>());
        assert_eq!(waves[1], (32..64).collect::<Vec<_>>());
        for result in &results[..64] {
            assert!(!result.unused_row_proposal);
        }
        for result in &results[64..100] {
            assert!(result.unused_row_proposal);
            assert_eq!(
                result.omission_marker.as_deref(),
                Some("_ordinary_list_unexecuted")
            );
        }
        assert_eq!(
            record["_unexecuted_list_items"].as_object().unwrap().len(),
            36
        );
    }

    #[test]
    fn text_list_completion_delegates_scalar_lists_8_to_32_to_64() {
        let (mut record, tasks) = scalar_list_fixture(50, false);
        assert!(text_list_eligible(&record, &tasks));
        assert!(ordinary_list_eligible(&record, &tasks));

        let mut waves: Vec<Vec<usize>> = Vec::new();
        let results = text_list_generate(&mut record, &tasks, EOS, &mock_decode, None, |indices| {
            waves.push(indices.to_vec());
            Ok(indices
                .iter()
                .map(|&idx| {
                    // Null at index 15 -> frontier 8..32 catches it, 32..50 skipped!
                    let tok = if idx >= 15 { TOK_NULL } else { TOK_VAL };
                    make_result(&tasks[idx].rid, tok)
                })
                .collect())
        })
        .unwrap();

        assert_eq!(waves.len(), 2);
        assert_eq!(waves[0], (0..8).collect::<Vec<_>>());
        assert_eq!(waves[1], (8..32).collect::<Vec<_>>());
        for result in &results[32..50] {
            assert!(result.unused_row_proposal);
            assert_eq!(
                result.omission_marker.as_deref(),
                Some("_ordinary_list_unexecuted")
            );
        }
    }

    #[test]
    fn text_list_completion_gates_27b_object_list_at_frontier_8() {
        let count = 16;
        let mut fields = Map::new();
        let mut ann = Map::new();
        let mut query_names = Vec::new();
        let mut tasks = Vec::new();

        fields.insert(
            "entries".to_string(),
            json!({
                "type": "list",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "text"}}
                }
            }),
        );
        for i in 1..=count {
            let name = schema::list_item_name("entries", i);
            fields.insert(
                name.clone(),
                json!({
                    "type": "object",
                    "nullable": true,
                    "properties": {"name": {"type": "text"}}
                }),
            );
            ann.insert(name.clone(), json!({"parent": "entries", "index": i}));
            query_names.push(Value::String(name.clone()));
            tasks.push(ArTaskDescriptor {
                name,
                rid: format!("obj-rid-{i}"),
                input_ids: vec![7; 30],
                has_media: false,
            });
        }
        ann.insert("entries".to_string(), json!({"query_fields": query_names}));

        let mut record = json!({
            "input": "doc",
            "fields": fields,
            "_first_item_list_gating": true,
            "_meta": {
                "construction": EVIDENCE_CONSTRUCTION,
                "extraction_contract": {"fields": ann}
            }
        });

        assert!(text_list_eligible(&record, &tasks));
        // Object list disqualifies ordinary_list_eligible
        assert!(!ordinary_list_eligible(&record, &tasks));

        let mut waves: Vec<Vec<usize>> = Vec::new();
        let results = text_list_generate(&mut record, &tasks, EOS, &mock_decode, None, |indices| {
            waves.push(indices.to_vec());
            Ok(indices
                .iter()
                .map(|&idx| {
                    // Null at index 3 (inside frontier 0..8) -> indices 8..16 omitted!
                    let tok = if idx >= 3 { TOK_NULL } else { TOK_VAL };
                    make_result(&tasks[idx].rid, tok)
                })
                .collect())
        })
        .unwrap();

        assert_eq!(waves, vec![(0..8).collect::<Vec<_>>()]);
        for result in &results[..8] {
            assert!(!result.unused_row_proposal);
        }
        for result in &results[8..16] {
            assert!(result.unused_row_proposal);
            assert_eq!(
                result.omission_marker.as_deref(),
                Some("_text_frontier_unexecuted")
            );
        }
        assert_eq!(
            record["_unexecuted_list_items"].as_object().unwrap().len(),
            8
        );
    }

    #[test]
    fn image_row_cohort_falls_back_to_wave_2_when_proof_has_conflict() {
        let proposal_count = 10;
        let mut fields = Map::new();
        let mut ann = Map::new();
        let mut query_names = Vec::new();
        let mut tasks = Vec::new();

        fields.insert(
            "rows".to_string(),
            json!({
                "type": "list",
                "nullable": true,
                "_image_row_inventory": true,
                "max_items": 10,
                "items": {
                    "type": "object",
                    "properties": {"description": {"type": "text"}}
                }
            }),
        );
        tasks.push(ArTaskDescriptor {
            name: "rows".to_string(),
            rid: "rid-inv".to_string(),
            input_ids: vec![1; 50],
            has_media: true,
        });
        for i in 1..=proposal_count {
            let name = schema::list_item_name("rows", i);
            fields.insert(
                name.clone(),
                json!({
                    "type": "object",
                    "nullable": false,
                    "_image_row_proposal": true,
                    "properties": {"description": {"type": "text"}}
                }),
            );
            ann.insert(name.clone(), json!({"parent": "rows", "index": i}));
            query_names.push(Value::String(name.clone()));
            tasks.push(ArTaskDescriptor {
                name,
                rid: format!("rid-row-{i}"),
                input_ids: vec![2; 40],
                has_media: true,
            });
        }
        ann.insert("rows".to_string(), json!({"image_row_fields": query_names}));

        let mut record = json!({
            "input": "",
            "fields": fields,
            "_first_item_list_gating": true,
            "_meta": {
                "construction": EVIDENCE_CONSTRUCTION,
                "extraction_contract": {"fields": ann}
            }
        });

        let parse_fn = |res: &ArBatchItemResult, _spec: &Value, _src: &str| -> Value {
            let val: Value = serde_json::from_str(&res.text).unwrap_or(Value::Null);
            json!({"value": val, "valid": true, "complete": true})
        };
        let norm_fn =
            |_rec: &Value, preds: &Map<String, Value>| -> Map<String, Value> { preds.clone() };
        let hooks = CohortHooks {
            parse_result: &parse_fn,
            normalize: &norm_fn,
        };

        let mut waves: Vec<Vec<usize>> = Vec::new();
        let results = batch_generate(
            &mut record,
            &tasks,
            EOS,
            &mock_decode,
            Some(&hooks),
            |indices| {
                waves.push(indices.to_vec());
                Ok(indices
                    .iter()
                    .map(|&idx| {
                        // Inventory expects "Alpha" & "Beta", but proposal 2 returns "Wrong" -> conflict!
                        let text = match idx {
                            0 => r#"[{"label":"Alpha"},{"label":"Beta"}]"#.to_string(),
                            1 => r#"{"description":"Alpha"}"#.to_string(),
                            2 => r#"{"description":"Wrong"}"#.to_string(),
                            _ => r#"{"description":null}"#.to_string(),
                        };
                        ArBatchItemResult {
                            id: tasks[idx].rid.clone(),
                            output_ids: None,
                            text,
                            completion_tokens: 5,
                            finish_reason_type: Some("stop".to_string()),
                            finish_reason_matched: Some(EOS),
                            output_token_logprobs: Vec::new(),
                            output_token_ids_logprobs: Vec::new(),
                            unused_row_proposal: false,
                            omission_marker: None,
                        }
                    })
                    .collect())
            },
        )
        .unwrap();

        // Proof failed -> wave 2 ran proposals 9..=10 (indices 9, 10)
        assert_eq!(waves, vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8], vec![9, 10]]);
        assert!(results.iter().all(|r| !r.unused_row_proposal));
        assert_eq!(record["_row_cohort_gating"][0]["fallback"], json!(true));
    }

    #[test]
    fn text_list_completion_27b_object_list_runs_tail_when_no_null_in_frontier() {
        let count = 12;
        let mut fields = Map::new();
        let mut ann = Map::new();
        let mut query_names = Vec::new();
        let mut tasks = Vec::new();

        fields.insert(
            "entries".to_string(),
            json!({
                "type": "list",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "text"}}
                }
            }),
        );
        for i in 1..=count {
            let name = schema::list_item_name("entries", i);
            fields.insert(
                name.clone(),
                json!({
                    "type": "object",
                    "nullable": true,
                    "properties": {"name": {"type": "text"}}
                }),
            );
            ann.insert(name.clone(), json!({"parent": "entries", "index": i}));
            query_names.push(Value::String(name.clone()));
            tasks.push(ArTaskDescriptor {
                name,
                rid: format!("obj-rid-{i}"),
                input_ids: vec![7; 30],
                has_media: false,
            });
        }
        ann.insert("entries".to_string(), json!({"query_fields": query_names}));

        let mut record = json!({
            "input": "doc",
            "fields": fields,
            "_first_item_list_gating": true,
            "_meta": {
                "construction": EVIDENCE_CONSTRUCTION,
                "extraction_contract": {"fields": ann}
            }
        });

        let mut waves: Vec<Vec<usize>> = Vec::new();
        let results = text_list_generate(&mut record, &tasks, EOS, &mock_decode, None, |indices| {
            waves.push(indices.to_vec());
            Ok(indices
                .iter()
                .map(|&idx| make_result(&tasks[idx].rid, TOK_VAL))
                .collect())
        })
        .unwrap();

        assert_eq!(
            waves,
            vec![(0..8).collect::<Vec<_>>(), (8..12).collect::<Vec<_>>()]
        );
        assert!(results.iter().all(|r| !r.unused_row_proposal));
        assert!(
            record["_unexecuted_list_items"]
                .as_object()
                .unwrap()
                .is_empty()
        );
    }
}
