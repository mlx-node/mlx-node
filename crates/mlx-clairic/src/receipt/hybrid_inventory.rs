//! Hybrid receipt paragraph inventory, unresolved-coverage checks, and runtime validation.
//!
//! Direct port of upstream `clairic_runtime/receipt_hybrid_inventory.py`.

use super::{
    canonical as canonical_bytes, char_to_byte_map, extract_span, find_char_spans_in_range,
    is_currency_symbol, sha256_hex, slice_chars, source_sums as sums, totals as base,
};
use crate::{ar, schema};
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashSet};
use std::sync::OnceLock;

pub const BOOLEAN_EBNF: &str = r#"root ::= "true" | "false" | "null""#;
pub const MAX_DECISIONS: usize = base::MAX_BLOCKS;
pub const HYBRID_PACKET_KIND: &str = "receipt-hybrid-paragraph-runtime-v1";

const HYBRID_PREFIX_ADDENDUM: &str = "When the candidate marks one exact occurrence, evaluate only that marked occurrence as a whole receipt total; its paragraph may contain multiple records or totals. Every candidate still requires its own category, entity, date and requested-scope decision.\n";

fn metadata_marker_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\A[ \t]+(?:method|reference|confirmation|booking|card|date|time)\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn line_with_newline_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^[^\n]*\n").unwrap_or_else(|_| unreachable!()))
}

/// Canonical JSON bytes matching `receipt_hybrid_inventory.py::canonical`.
pub fn canonical(value: &Value) -> Vec<u8> {
    canonical_bytes(value)
}

/// Delegate to [`base::metadata_spans`] (`receipt_hybrid_inventory.py::metadata_spans`).
pub fn metadata_spans(source: &str) -> Vec<(usize, usize)> {
    base::metadata_spans(source)
}

/// Detect unresolved record fragments (records missing numeric totals or structured
/// component-only fragments without a whole total) (`receipt_hybrid_inventory.py::unresolved_record_coverage`).
pub fn unresolved_record_coverage(
    source: &str,
    start: usize,
    end: usize,
    monetary: &[Value],
) -> Vec<Value> {
    let chars: Vec<char> = source.chars().collect();
    let c2b = char_to_byte_map(source);
    let b2c = schema::byte_to_char_map(source);

    let component_label = |item: &Value, lower: usize| -> Option<(usize, usize)> {
        let (left, _) = extract_span(item.get("span"))?;
        let comp_start = lower.max(left.saturating_sub(48));
        for (l_start, l_end) in find_char_spans_in_range(
            base::component_regex(),
            source,
            &c2b,
            &b2c,
            comp_start,
            left,
        ) {
            let gap = base::replace_sc_with(slice_chars(source, &c2b, l_end, left), '$');
            if base::bridge_regex().is_match(&gap) {
                return Some((l_start, l_end));
            }
        }
        None
    };

    let is_metadata_marker = |m_end: usize| -> bool {
        let line_end = chars[m_end..end]
            .iter()
            .position(|&c| c == '\n')
            .map(|off| m_end + off)
            .unwrap_or(end);
        if monetary.iter().any(|item| {
            extract_span(item.get("span")).is_some_and(|(s, _)| m_end <= s && s < line_end)
        }) {
            return false;
        }
        metadata_marker_regex().is_match(slice_chars(source, &c2b, m_end, line_end))
    };

    let markers: Vec<(usize, usize)> =
        find_char_spans_in_range(base::record_regex(), source, &c2b, &b2c, start, end)
            .into_iter()
            .filter(|&(_, m_end)| !is_metadata_marker(m_end))
            .collect();

    let mut boundaries: BTreeSet<usize> = BTreeSet::new();
    boundaries.insert(start);
    boundaries.insert(end);
    let mut marker_starts: HashSet<usize> = HashSet::with_capacity(markers.len());
    for &(m_start, _) in &markers {
        boundaries.insert(m_start);
        marker_starts.insert(m_start);
    }

    let metadata = metadata_spans(source);
    for pair in monetary.windows(2) {
        if component_label(&pair[1], start).is_none() {
            continue;
        }
        let (Some((_, left)), Some((right, _))) = (
            extract_span(pair[0].get("span")),
            extract_span(pair[1].get("span")),
        ) else {
            continue;
        };
        if left > right {
            continue;
        }
        for (line_start, line_end) in
            find_char_spans_in_range(line_with_newline_regex(), source, &c2b, &b2c, left, right)
        {
            if metadata
                .iter()
                .any(|&(lower, upper)| lower < line_end && line_start < upper)
            {
                continue;
            }
            let line_text = slice_chars(source, &c2b, line_start, line_end);
            let mut gap = base::replace_sc_with(line_text, ' ');
            for pattern in [
                base::total_regex(),
                base::component_regex(),
                base::component_connector_regex(),
                base::currency_code_regex(),
            ] {
                gap = pattern.replace_all(&gap, " ").into_owned();
            }
            if !base::amount_gap_regex().is_match(&gap) && !base::record_regex().is_match(line_text)
            {
                boundaries.insert(line_start);
            }
        }
    }

    let ordered_boundaries: Vec<usize> = boundaries.into_iter().collect();
    let mut unresolved = Vec::new();
    for pair in ordered_boundaries.windows(2) {
        let (lower, upper) = (pair[0], pair[1]);
        let amounts: Vec<&Value> = monetary
            .iter()
            .filter(|item| {
                extract_span(item.get("span")).is_some_and(|(s, _)| lower <= s && s < upper)
            })
            .collect();
        if amounts.is_empty() {
            if marker_starts.contains(&lower) {
                let mut entry = Map::new();
                entry.insert("span".to_string(), json!([lower, upper]));
                entry.insert(
                    "text".to_string(),
                    Value::String(slice_chars(source, &c2b, lower, upper).to_string()),
                );
                entry.insert(
                    "reason".to_string(),
                    json!("record_without_numeric_total_candidate"),
                );
                unresolved.push(Value::Object(entry));
            }
            continue;
        }

        let labels: Vec<Option<(usize, usize)>> = amounts
            .iter()
            .map(|item| component_label(item, lower))
            .collect();
        let structured = marker_starts.contains(&lower)
            || labels.iter().all(|label_opt| {
                let Some((label_start, _)) = label_opt else {
                    return false;
                };
                let line_begin = chars[lower..*label_start]
                    .iter()
                    .rposition(|&c| c == '\n')
                    .map(|off| lower + off + 1)
                    .unwrap_or(lower);
                slice_chars(source, &c2b, line_begin, *label_start)
                    .trim()
                    .is_empty()
            });

        if structured && labels.iter().all(Option::is_some) {
            let mut entry = Map::new();
            entry.insert("span".to_string(), json!([lower, upper]));
            entry.insert(
                "text".to_string(),
                Value::String(slice_chars(source, &c2b, lower, upper).to_string()),
            );
            entry.insert(
                "reason".to_string(),
                json!("component_amounts_without_whole_total_candidate"),
            );
            entry.insert(
                "occurrence_ids".to_string(),
                Value::Array(
                    amounts
                        .iter()
                        .map(|item| item.get("id").cloned().unwrap_or(Value::Null))
                        .collect(),
                ),
            );
            unresolved.push(Value::Object(entry));
        }
    }

    unresolved
}

/// Partition `task["input_text"]` paragraphs into whole-total receipt blocks, unresolved
/// monetary occurrences, numericless unresolved fragments, and unresolved coverage findings
/// (`receipt_hybrid_inventory.py::inventory`).
pub fn inventory(task: &Value) -> (Vec<Value>, Vec<Value>, Vec<Value>) {
    let Some(source) = task.get("input_text").and_then(Value::as_str) else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
    let Some(all_candidates) = task.get("candidates").and_then(Value::as_array) else {
        return (Vec::new(), Vec::new(), Vec::new());
    };

    let chars: Vec<char> = source.chars().collect();
    let c2b = char_to_byte_map(source);
    let metadata = metadata_spans(source);

    let mut paragraphs = Vec::new();
    let mut unresolved = Vec::new();
    let mut unresolved_coverage = Vec::new();

    for (start, end) in base::paragraphs(source) {
        let numbers: Vec<&Value> = all_candidates
            .iter()
            .filter(|item| {
                extract_span(item.get("span")).is_some_and(|(s, _)| start <= s && s < end)
            })
            .collect();
        let monetary_refs: Vec<&Value> = numbers
            .iter()
            .copied()
            .filter(|item| !base::is_non_amount_item(&chars, &metadata, item))
            .collect();

        let candidates: Vec<Value> =
            if let Some(blocks) = base::blocks_in_paragraphs(task, Some(&[(start, end)])) {
                blocks
                    .into_iter()
                    .map(|block| {
                        let mut m = Map::new();
                        m.insert("kind".to_string(), json!("whole_total"));
                        m.insert(
                            "occurrence".to_string(),
                            block
                                .get("total_occurrence")
                                .cloned()
                                .unwrap_or(Value::Null),
                        );
                        m.insert("source_block".to_string(), block);
                        Value::Object(m)
                    })
                    .collect()
            } else if !monetary_refs.is_empty() {
                let monetary_owned: Vec<Value> = monetary_refs.iter().map(|&v| v.clone()).collect();
                unresolved_coverage.extend(unresolved_record_coverage(
                    source,
                    start,
                    end,
                    &monetary_owned,
                ));
                monetary_refs
                    .iter()
                    .map(|item| {
                        let mut occ = Map::new();
                        occ.insert(
                            "id".to_string(),
                            item.get("id").cloned().unwrap_or(Value::Null),
                        );
                        occ.insert(
                            "span".to_string(),
                            item.get("span").cloned().unwrap_or(Value::Null),
                        );
                        occ.insert(
                            "printed".to_string(),
                            item.get("printed").cloned().unwrap_or(Value::Null),
                        );
                        let mut m = Map::new();
                        m.insert("kind".to_string(), json!("unresolved_occurrence"));
                        m.insert("occurrence".to_string(), Value::Object(occ));
                        Value::Object(m)
                    })
                    .collect()
            } else {
                let text = slice_chars(source, &c2b, start, end);
                if base::record_regex().is_match(text)
                    || base::total_regex().is_match(text)
                    || base::currency_code_regex().is_match(text)
                    || text.chars().any(is_currency_symbol)
                {
                    let mut u = Map::new();
                    u.insert("span".to_string(), json!([start, end]));
                    u.insert("text".to_string(), Value::String(text.to_string()));
                    unresolved.push(Value::Object(u));
                }
                continue;
            };

        let mut para = Map::new();
        para.insert("span".to_string(), json!([start, end]));
        para.insert(
            "text".to_string(),
            Value::String(slice_chars(source, &c2b, start, end).to_string()),
        );
        para.insert("candidates".to_string(), Value::Array(candidates));
        para.insert(
            "numeric_occurrence_ids".to_string(),
            Value::Array(
                numbers
                    .iter()
                    .map(|item| item.get("id").cloned().unwrap_or(Value::Null))
                    .collect(),
            ),
        );
        para.insert(
            "metadata_occurrence_ids".to_string(),
            Value::Array(
                numbers
                    .iter()
                    .filter(|item| base::is_non_amount_item(&chars, &metadata, item))
                    .map(|item| item.get("id").cloned().unwrap_or(Value::Null))
                    .collect(),
            ),
        );
        para.insert(
            "monetary_occurrence_ids".to_string(),
            Value::Array(
                monetary_refs
                    .iter()
                    .map(|item| item.get("id").cloned().unwrap_or(Value::Null))
                    .collect(),
            ),
        );
        paragraphs.push(Value::Object(para));
    }

    (paragraphs, unresolved, unresolved_coverage)
}

/// Build a hybrid paragraph runtime packet from a `source_sums` task
/// (`receipt_hybrid_inventory.py::_from_sum_task`).
pub(crate) fn from_sum_task(task: &Value) -> Option<Value> {
    let (paragraphs, unresolved, unresolved_coverage) = inventory(task);
    if !unresolved_coverage.is_empty() {
        return None;
    }

    let mut descriptors = Vec::new();
    for paragraph in &paragraphs {
        let para_span = paragraph.get("span").cloned().unwrap_or(Value::Null);
        let para_text = paragraph.get("text").cloned().unwrap_or(Value::Null);
        let Some(cands) = paragraph.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        for candidate in cands {
            let mut desc = candidate.as_object()?.clone();
            desc.insert("paragraph_span".to_string(), para_span.clone());
            desc.insert("source_paragraph".to_string(), para_text.clone());
            descriptors.push(Value::Object(desc));
        }
    }

    if descriptors.is_empty() || descriptors.len() > MAX_DECISIONS {
        return None;
    }

    let mut prefix = base::build_shared_prefix(task)?;
    prefix.push_str(HYBRID_PREFIX_ADDENDUM);

    let mut tasks = Vec::with_capacity(descriptors.len());
    let mut seen_ids = HashSet::with_capacity(descriptors.len());
    for descriptor in &descriptors {
        let suffix = if descriptor.get("kind").and_then(Value::as_str) == Some("whole_total") {
            let source_block = descriptor.get("source_block")?;
            format!(
                "\nCandidate source block: {}",
                ar::compact_json(source_block)
            )
        } else {
            format!(
                "\nCandidate exact source occurrence: {}",
                ar::compact_json(descriptor)
            )
        };
        let occ_id = descriptor
            .get("occurrence")
            .and_then(|o| o.get("id"))
            .and_then(Value::as_str)?;
        if !seen_ids.insert(occ_id) {
            return None;
        }
        let mut t = Map::new();
        t.insert(
            "occurrence_id".to_string(),
            Value::String(occ_id.to_string()),
        );
        t.insert(
            "prompt".to_string(),
            Value::String(format!("{prefix}{suffix}")),
        );
        t.insert("suffix".to_string(), Value::String(suffix));
        t.insert(
            "json_schema".to_string(),
            json!({"type": ["boolean", "null"]}),
        );
        t.insert("grammar".to_string(), json!({"ebnf": BOOLEAN_EBNF}));
        tasks.push(Value::Object(t));
    }

    let para_texts: Vec<&str> = paragraphs
        .iter()
        .filter_map(|row| row.get("text").and_then(Value::as_str))
        .collect();
    let currency_compatible = base::compatible_currencies(task, para_texts);
    let prefix_sha = sha256_hex(prefix.as_bytes());

    let mut packet = Map::new();
    packet.insert("version".to_string(), json!(1));
    packet.insert("kind".to_string(), json!(HYBRID_PACKET_KIND));
    packet.insert("sum_task".to_string(), task.clone());
    packet.insert("paragraphs".to_string(), Value::Array(paragraphs));
    packet.insert("descriptors".to_string(), Value::Array(descriptors));
    packet.insert("tasks".to_string(), Value::Array(tasks));
    packet.insert(
        "currency_compatible".to_string(),
        Value::Bool(currency_compatible),
    );
    packet.insert(
        "unresolved_fragments_without_numeric_candidates".to_string(),
        Value::Array(unresolved),
    );
    packet.insert("shared_prefix".to_string(), Value::String(prefix));
    packet.insert(
        "shared_prefix_sha256".to_string(),
        Value::String(prefix_sha),
    );
    packet.insert("diagnostic_only".to_string(), Value::Bool(false));
    packet.insert("public_repair_path".to_string(), Value::Bool(true));
    packet.insert(
        "serving_admission_requires_backend_guard".to_string(),
        Value::Bool(true),
    );
    packet.insert("runtime_changes".to_string(), Value::Bool(true));
    packet.insert(
        "all_candidates_require_complete_decisions".to_string(),
        Value::Bool(true),
    );
    packet.insert("native_typed_boolean_head".to_string(), Value::Bool(false));
    packet.insert(
        "semantic_completeness_proven_by_grounding".to_string(),
        Value::Bool(false),
    );
    Some(Value::Object(packet))
}

/// Prepare a hybrid receipt inventory packet directly (`receipt_hybrid_inventory.py::prepare_task`).
pub fn prepare_task(record: &Value, field_name: &str, has_media: bool) -> Option<Value> {
    if !base::eligible(record, field_name, has_media) {
        return None;
    }
    let task = sums::prepare_task(record, field_name, has_media)?;
    from_sum_task(&task)
}

fn is_close_rel(a: f64, b: f64, rel_tol: f64) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() <= rel_tol * a.abs().max(b.abs())
}

/// Validate boolean decisions for a hybrid paragraph runtime packet and compute the exact
/// `BigDecimal` sum (`receipt_hybrid_inventory.py::validate_result`).
pub fn validate_result(packet: &Value, decisions: &Map<String, Value>) -> Value {
    let required_count = packet
        .get("tasks")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);

    let mut result = Map::new();
    result.insert("valid".to_string(), Value::Bool(false));
    result.insert("value".to_string(), Value::Null);
    result.insert("error".to_string(), Value::Null);
    result.insert("decision_count".to_string(), json!(0));
    result.insert("required_decision_count".to_string(), json!(required_count));
    result.insert("diagnostic_only".to_string(), Value::Bool(false));
    result.insert("public_repair_path".to_string(), Value::Bool(true));
    result.insert(
        "serving_admission_requires_backend_guard".to_string(),
        Value::Bool(true),
    );
    result.insert(
        "backend_constructs_public_answer".to_string(),
        Value::Bool(true),
    );
    result.insert(
        "semantic_completeness_proven_by_grounding".to_string(),
        Value::Bool(false),
    );

    if let Err(err) = validate_result_inner(packet, decisions, &mut result) {
        result.insert("valid".to_string(), Value::Bool(false));
        result.insert("value".to_string(), Value::Null);
        result.insert("error".to_string(), Value::String(err));
    }

    Value::Object(result)
}

fn validate_result_inner(
    packet: &Value,
    decisions: &Map<String, Value>,
    result: &mut Map<String, Value>,
) -> std::result::Result<(), String> {
    let task = packet
        .get("sum_task")
        .ok_or_else(|| "missing_sum_task".to_string())?;
    let input_text = task
        .get("input_text")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_input_text".to_string())?;
    let field = task
        .get("field")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_field".to_string())?;
    let field_spec = task
        .get("field_spec")
        .ok_or_else(|| "missing_field_spec".to_string())?;

    let rebuilt = base::prepare_task(
        &json!({
            "input": input_text,
            "fields": { field: field_spec },
        }),
        field,
        false,
    );
    if rebuilt.as_ref().map(canonical).as_deref() != Some(canonical(packet).as_slice()) {
        return Err("hybrid_packet_prompt_or_source_mapping_changed".to_string());
    }

    let unresolved = packet
        .get("unresolved_fragments_without_numeric_candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing_unresolved_fragments".to_string())?;
    if !unresolved.is_empty() {
        return Err("unresolved_fragment_without_numeric_candidate".to_string());
    }

    let packet_tasks = packet
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing_tasks".to_string())?;
    let mut expected = Vec::with_capacity(packet_tasks.len());
    for item in packet_tasks {
        let occ_id = item
            .get("occurrence_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing_occurrence_id".to_string())?;
        expected.push(occ_id.to_string());
    }

    let expected_set: HashSet<&str> = expected.iter().map(String::as_str).collect();
    let decision_set: HashSet<&str> = decisions.keys().map(String::as_str).collect();
    if decision_set != expected_set {
        return Err("hybrid_decision_coverage_incomplete_or_extra".to_string());
    }

    let mut selected: Vec<String> = Vec::new();
    let mut decision_count = 0usize;
    for identifier in &expected {
        let Some(reading) = decisions.get(identifier).and_then(Value::as_object) else {
            return Err(format!("unresolved_or_incomplete_boolean:{identifier}"));
        };
        let raw_json = reading
            .get("raw_json")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("unresolved_or_incomplete_boolean:{identifier}"))?;
        let parsed: Value = serde_json::from_str(raw_json).map_err(|e| e.to_string())?;
        let Some(flag) = parsed.as_bool() else {
            return Err(format!("unresolved_or_incomplete_boolean:{identifier}"));
        };
        if reading.get("complete").and_then(Value::as_bool) != Some(true) {
            return Err(format!("unresolved_or_incomplete_boolean:{identifier}"));
        }

        let score_val = reading.get("log_probability");
        let conf_val = reading.get("confidence");
        let tokens_val = reading.get("tokens");

        let valid_score = score_val
            .filter(|v| !v.is_boolean())
            .and_then(Value::as_f64)
            .filter(|&s| s.is_finite() && s <= 0.0);
        let valid_conf = conf_val
            .filter(|v| !v.is_boolean())
            .and_then(Value::as_f64)
            .filter(|&c| c.is_finite() && (0.0..=1.0).contains(&c));
        let valid_tokens = tokens_val
            .filter(|v| v.is_i64() || v.is_u64())
            .and_then(Value::as_i64)
            .filter(|&t| (1..=8).contains(&t));

        let (Some(score), Some(confidence), Some(_tokens)) =
            (valid_score, valid_conf, valid_tokens)
        else {
            return Err(format!(
                "native_boolean_score_or_token_contract_failed:{identifier}"
            ));
        };
        if !is_close_rel(confidence, score.exp(), 2e-15) {
            return Err(format!(
                "native_boolean_score_or_token_contract_failed:{identifier}"
            ));
        }

        decision_count += 1;
        if flag {
            selected.push(identifier.clone());
        }
    }
    result.insert("decision_count".to_string(), json!(decision_count));

    let scores = base::selection_scores(decisions, &expected)
        .ok_or_else(|| "native_joint_boolean_scores_required".to_string())?;
    if packet.get("currency_compatible").and_then(Value::as_bool) != Some(true) {
        return Err("explicit_currency_conflict_no_conversion".to_string());
    }

    let raw_selected = if selected.is_empty() {
        "null".to_string()
    } else {
        ar::compact_json(&Value::Array(
            selected.iter().cloned().map(Value::String).collect(),
        ))
    };
    let checked = sums::validate_result(task, &raw_selected, true);
    if let Some(checked_obj) = checked.as_object() {
        for (k, v) in checked_obj {
            result.insert(k.clone(), v.clone());
        }
    }
    result.insert(
        "selected_ids".to_string(),
        Value::Array(selected.into_iter().map(Value::String).collect()),
    );
    result.insert("selection_scores".to_string(), scores);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_hybrid_decision(val: bool, log_prob: f64, tokens: i64) -> Value {
        let raw = if val { "true" } else { "false" };
        json!({
            "value": val,
            "raw": raw,
            "raw_json": raw,
            "valid": true,
            "complete": true,
            "decoder": ar::VERSION,
            "confidence_method": "ar_sequence_probability",
            "log_probability": log_prob,
            "confidence": log_prob.exp(),
            "tokens": tokens
        })
    }

    #[test]
    fn hybrid_inventory_parses_mixed_whole_total_and_prose_occurrences() {
        // Paragraph 1 is a structured receipt block (whole_total).
        // Paragraph 2 has two inline totals in one paragraph (falls back to hybrid unresolved_occurrence).
        let record = json!({
            "input": "Receipt 1 - Dinner\nFood: $18.00\nTip: $2.00\nTotal: $20.00\n\nTaxi ride on May 2 cost $14.50 and airport shuttle cost $15.50.",
            "fields": {
                "receipt_total": {
                    "type": "number",
                    "decimals": 2,
                    "instructions": "Sum of all receipt totals."
                }
            }
        });
        let packet = base::prepare_task(&record, "receipt_total", false).unwrap();
        assert_eq!(packet["kind"], json!(HYBRID_PACKET_KIND));
        assert_eq!(packet["currency_compatible"], json!(true));
        let descriptors = packet["descriptors"].as_array().unwrap();
        assert_eq!(descriptors.len(), 3);
        assert_eq!(descriptors[0]["kind"], json!("whole_total"));
        assert_eq!(descriptors[0]["occurrence"]["printed"], json!("$20.00"));
        assert_eq!(descriptors[1]["kind"], json!("unresolved_occurrence"));
        assert_eq!(descriptors[1]["occurrence"]["printed"], json!("$14.50"));
        assert_eq!(descriptors[2]["kind"], json!("unresolved_occurrence"));
        assert_eq!(descriptors[2]["occurrence"]["printed"], json!("$15.50"));

        // Verify exact supports_sum works on hybrid packets too!
        assert_eq!(base::supports_sum(&packet, &json!(50.0)), Some(true));
        assert_eq!(base::supports_sum(&packet, &json!(34.5)), Some(true));
        assert_eq!(base::supports_sum(&packet, &json!(18.0)), Some(false));

        // Validate decisions through totals::validate_result -> hybrid_inventory::validate_result
        let tasks = packet["tasks"].as_array().unwrap();
        let mut decisions = Map::new();
        decisions.insert(
            tasks[0]["occurrence_id"].as_str().unwrap().to_string(),
            make_hybrid_decision(true, -0.01, 1),
        );
        decisions.insert(
            tasks[1]["occurrence_id"].as_str().unwrap().to_string(),
            make_hybrid_decision(true, -0.02, 1),
        );
        decisions.insert(
            tasks[2]["occurrence_id"].as_str().unwrap().to_string(),
            make_hybrid_decision(false, -0.03, 1),
        );

        let checked = base::validate_result(&packet, &decisions);
        assert_eq!(checked["valid"], json!(true));
        assert_eq!(checked["value"], json!(34.5));
        assert_eq!(checked["operand_count"], json!(2));
        assert_eq!(checked["decision_count"], json!(3));
    }

    #[test]
    fn hybrid_inventory_rejects_component_only_record_without_whole_total() {
        let record = json!({
            "input": "Receipt 1 - Bistro\nFood: $25.00\nTax: $2.50\nTip: $4.00",
            "fields": {
                "receipt_total": {
                    "type": "number",
                    "instructions": "Sum of all receipt totals."
                }
            }
        });
        let task = sums::prepare_task(&record, "receipt_total", false).unwrap();
        let (_, _, unresolved_cov) = inventory(&task);
        assert_eq!(unresolved_cov.len(), 1);
        assert_eq!(
            unresolved_cov[0]["reason"],
            json!("component_amounts_without_whole_total_candidate")
        );
        assert!(base::prepare_task(&record, "receipt_total", false).is_none());
    }

    #[test]
    fn hybrid_validation_enforces_token_and_confidence_contract() {
        let record = json!({
            "input": "Receipt 1 - Dinner\nTotal: $20.00\n\nTaxi ride cost $10.00 and parking cost $5.00.",
            "fields": {
                "receipt_total": {
                    "type": "number",
                    "decimals": 2,
                    "instructions": "Sum of all receipt totals."
                }
            }
        });
        let packet = base::prepare_task(&record, "receipt_total", false).unwrap();
        let tasks = packet["tasks"].as_array().unwrap();

        // Bad token count (> 8)
        let mut decisions = Map::new();
        decisions.insert(
            tasks[0]["occurrence_id"].as_str().unwrap().to_string(),
            make_hybrid_decision(true, -0.01, 9),
        );
        decisions.insert(
            tasks[1]["occurrence_id"].as_str().unwrap().to_string(),
            make_hybrid_decision(true, -0.02, 1),
        );
        decisions.insert(
            tasks[2]["occurrence_id"].as_str().unwrap().to_string(),
            make_hybrid_decision(false, -0.03, 1),
        );
        let checked = validate_result(&packet, &decisions);
        assert_eq!(checked["valid"], json!(false));
        assert!(
            checked["error"]
                .as_str()
                .unwrap()
                .starts_with("native_boolean_score_or_token_contract_failed:")
        );

        // Mismatched confidence vs exp(log_probability)
        let mut bad_conf = make_hybrid_decision(true, -0.1, 1);
        bad_conf["confidence"] = json!(0.5);
        decisions.insert(
            tasks[0]["occurrence_id"].as_str().unwrap().to_string(),
            bad_conf,
        );
        let checked2 = validate_result(&packet, &decisions);
        assert_eq!(checked2["valid"], json!(false));
        assert!(
            checked2["error"]
                .as_str()
                .unwrap()
                .starts_with("native_boolean_score_or_token_contract_failed:")
        );
    }
}
