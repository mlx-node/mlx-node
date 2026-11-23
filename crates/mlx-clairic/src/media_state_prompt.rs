//! 27B multi-image & boolean+media+state prompt rewriter (`MEDIA_EVIDENCE_INSTRUCTION`
//! + `Accompanying text:` boundary).
//!
//! Direct port of upstream `clairic_runtime/media_state_prompt.py`.

use super::{ar, errors::invalid_request, grounding};
use crate::errors::Result;
use serde_json::{Map, Value};
use std::collections::HashSet;
use tokenizers::Tokenizer;

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

/// Return `true` iff a request qualifies for media-evidence prompt rewriting
/// (`media_state_prompt.py::eligible`).
pub fn eligible(
    state: &str,
    fields: &Map<String, Value>,
    image_count: usize,
    document_count: usize,
) -> bool {
    if state.is_empty() || (image_count == 0 && document_count == 0) {
        return false;
    }
    if image_count > 1 {
        return true;
    }
    fields.values().any(|spec| {
        spec.get("type").and_then(Value::as_str) == Some("boolean")
            && !is_truthy(spec.get("knowledge"))
            && !grounding::state_only(spec)
    })
}

/// Rewrite a decoded prompt string to prepend [`ar::MEDIA_EVIDENCE_INSTRUCTION`] and
/// replace `Document:` with `Accompanying text:`.
pub fn rewrite_media_evidence_text(text: &str, record: &Value, name: &str) -> Result<String> {
    let instruction = ar::instruction(record, name, false)?.trim_end().to_string();
    let boundary = format!("{instruction}\n\nDocument:\n");
    let existing = format!(
        "{}\n\n{instruction}\n\nAccompanying text:\n",
        ar::MEDIA_EVIDENCE_INSTRUCTION
    );
    if text.matches(&existing).count() == 1 {
        return Ok(text.to_string());
    }
    if text.matches(&boundary).count() != 1 || text.contains(ar::MEDIA_EVIDENCE_INSTRUCTION) {
        return Err(invalid_request(
            "Expected exactly one original instruction and accompanying-state boundary",
        ));
    }
    Ok(text.replace(&boundary, &existing))
}

/// Rewrite a single AR task's token IDs with strict lossless tokenizer roundtrip checks
/// and `max_length` verification (`media_state_prompt.py::rewrite_media_evidence`).
pub fn rewrite_media_evidence_ids(
    name: &str,
    ids: &[u32],
    has_media: bool,
    record: &Value,
    tokenizer: &Tokenizer,
    max_length: usize,
) -> Result<Vec<u32>> {
    if !has_media {
        return Err(invalid_request(
            "Media evidence guidance requires actual task media",
        ));
    }
    let text = tokenizer
        .decode(ids, false)
        .map_err(|e| invalid_request(format!("Media evidence prompt decode failed: {e}")))?;
    let roundtrip = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| invalid_request(format!("Media evidence prompt re-encode failed: {e}")))?;
    if roundtrip.get_ids() != ids {
        return Err(invalid_request(
            "Media evidence prompt must make a lossless tokenizer roundtrip",
        ));
    }

    let revised = rewrite_media_evidence_text(&text, record, name)?;
    if revised == text {
        return Ok(ids.to_vec());
    }

    let revised_enc = tokenizer
        .encode(revised.as_str(), false)
        .map_err(|e| invalid_request(format!("Media evidence revised encode failed: {e}")))?;
    let revised_ids = revised_enc.get_ids().to_vec();
    let decoded_back = tokenizer
        .decode(&revised_ids, false)
        .map_err(|e| invalid_request(format!("Media evidence revised decode failed: {e}")))?;
    if decoded_back != revised {
        return Err(invalid_request(
            "Media evidence revised prompt must make a lossless tokenizer roundtrip",
        ));
    }
    if revised_ids.len() > max_length {
        return Err(invalid_request(
            "Media evidence prompt exceeds the complete model context",
        ));
    }
    Ok(revised_ids)
}

/// Determine whether a specific AR task should be rewritten under `media_state_prompt.py::prepare`.
pub fn should_rewrite_task(
    name: &str,
    has_media: bool,
    record: &Value,
    request_fields: &Map<String, Value>,
    image_count: usize,
) -> bool {
    if !has_media {
        return false;
    }
    let meta = record.get("_meta");
    let annotations = meta
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object);

    let mut protected: HashSet<&str> = HashSet::new();
    if let Some(spelling) = meta
        .and_then(|m| m.get("identifier_spelling"))
        .and_then(Value::as_object)
    {
        protected.extend(spelling.keys().map(String::as_str));
    }
    if let Some(po_supp) = meta
        .and_then(|m| m.get("model_po_supplement"))
        .and_then(|s| s.get("fields"))
        .and_then(Value::as_array)
    {
        protected.extend(po_supp.iter().filter_map(Value::as_str));
    }

    let parent = annotations
        .and_then(|a| a.get(name))
        .and_then(|f| f.get("parent"))
        .and_then(Value::as_str)
        .unwrap_or(name);

    if protected.contains(name) || protected.contains(parent) {
        return false;
    }
    let Some(spec) = request_fields.get(parent) else {
        return false;
    };
    if is_truthy(spec.get("knowledge")) || grounding::state_only(spec) {
        return false;
    }
    let multi_image = image_count > 1;
    multi_image || (name == parent && spec.get("type").and_then(Value::as_str) == Some("boolean"))
}

/// Rewrite all eligible tasks in `tasks` (`(name, input_ids, media)`), returning
/// `Some(updated_tasks)` if any task changed or `None` if nothing changed or any rewrite failed
/// (atomic fallback matching `media_state_prompt.py::prepare`).
pub fn prepare_tasks<M: Clone>(
    state: &str,
    request_fields: &Map<String, Value>,
    image_count: usize,
    document_count: usize,
    record: &Value,
    tasks: &[(String, Vec<u32>, Option<M>)],
    tokenizer: &Tokenizer,
    max_length: usize,
) -> Option<Vec<(String, Vec<u32>, Option<M>)>> {
    if !eligible(state, request_fields, image_count, document_count) {
        return None;
    }
    let mut updated = Vec::with_capacity(tasks.len());
    let mut changed = false;

    for (name, ids, media) in tasks {
        let selected =
            should_rewrite_task(name, media.is_some(), record, request_fields, image_count);
        if selected {
            match rewrite_media_evidence_ids(
                name,
                ids,
                media.is_some(),
                record,
                tokenizer,
                max_length,
            ) {
                Ok(revised_ids) => {
                    if revised_ids != *ids {
                        changed = true;
                    }
                    updated.push((name.clone(), revised_ids, media.clone()));
                }
                Err(_) => return None,
            }
        } else {
            updated.push((name.clone(), ids.clone(), media.clone()));
        }
    }

    if changed { Some(updated) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn load_lossless_tokenizer() -> Tokenizer {
        // Build a minimal byte-level BPE tokenizer with matching ByteLevel decoder so
        // arbitrary ASCII/UTF-8 text makes an exact encode -> decode -> encode roundtrip
        // without depending on external checkpoint files.
        let mut vocab = serde_json::Map::new();
        let mut next_id: u32 = 0;
        let mut bs: Vec<u8> = (b'!'..=b'~')
            .chain(0xa1u8..=0xacu8)
            .chain(0xaeu8..=0xffu8)
            .collect();
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n = 0u32;
        for b in 0u8..=255u8 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        for c in cs {
            let ch = char::from_u32(c).unwrap();
            vocab.insert(ch.to_string(), serde_json::Value::from(next_id));
            next_id += 1;
        }
        let tok_json = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            },
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": []
            }
        });
        Tokenizer::from_bytes(tok_json.to_string().as_bytes()).expect("build byte-level tokenizer")
    }

    #[test]
    fn eligibility_checks_multi_image_and_boolean_media_state() {
        let bool_fields = json!({
            "signed": {"type": "boolean", "instructions": "Is the document signed?"}
        });
        let text_fields = json!({
            "vendor": {"type": "text", "instructions": "Vendor name."}
        });
        let state_only_bool = json!({
            "flag": {"type": "boolean", "instructions": "From the accompanying state text only."}
        });

        // Empty state or no media => false
        assert!(!eligible("", bool_fields.as_object().unwrap(), 1, 0));
        assert!(!eligible("note", bool_fields.as_object().unwrap(), 0, 0));

        // Multi-image => true even for text fields
        assert!(eligible("note", text_fields.as_object().unwrap(), 2, 0));

        // Single image + boolean => true
        assert!(eligible("note", bool_fields.as_object().unwrap(), 1, 0));

        // Single image + text only => false
        assert!(!eligible("note", text_fields.as_object().unwrap(), 1, 0));

        // Note: state_only only returns true for text/number (same as grounding.py), so boolean is eligible
        assert!(eligible("note", state_only_bool.as_object().unwrap(), 1, 0));
    }

    #[test]
    fn rewrite_media_evidence_ids_roundtrips_and_respects_protected_fields() {
        let tokenizer = load_lossless_tokenizer();
        let record = json!({
            "input": "Accompanying note",
            "fields": {
                "signed": {"type": "boolean", "instructions": "Is the receipt signed?"},
                "reference": {"type": "text", "instructions": "PO reference."}
            },
            "_meta": {
                "identifier_spelling": {
                    "reference": {"wire": "identifier-codepoint-spelling-v1"}
                }
            }
        });

        let inst = ar::instruction(&record, "signed", false).unwrap();
        let prompt = format!(
            "prefix\n{}\n\nDocument:\nAccompanying note\n",
            inst.trim_end()
        );
        let ids = tokenizer
            .encode(prompt.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();

        let revised_ids =
            rewrite_media_evidence_ids("signed", &ids, true, &record, &tokenizer, 4096).unwrap();
        let revised_text = tokenizer.decode(&revised_ids, false).unwrap();
        assert!(revised_text.contains(ar::MEDIA_EVIDENCE_INSTRUCTION));
        assert!(revised_text.contains("\n\nAccompanying text:\nAccompanying note"));
        assert!(!revised_text.contains("\n\nDocument:\n"));

        // Idempotent on second call
        let revised_again =
            rewrite_media_evidence_ids("signed", &revised_ids, true, &record, &tokenizer, 4096)
                .unwrap();
        assert_eq!(revised_again, revised_ids);

        // Rejects missing media and max_length overflow
        assert!(
            rewrite_media_evidence_ids("signed", &ids, false, &record, &tokenizer, 4096).is_err()
        );
        assert!(rewrite_media_evidence_ids("signed", &ids, true, &record, &tokenizer, 5).is_err());

        // Protected field is skipped in prepare_tasks
        let req_fields = record["fields"].as_object().unwrap();
        assert!(should_rewrite_task("signed", true, &record, req_fields, 2));
        assert!(!should_rewrite_task(
            "reference",
            true,
            &record,
            req_fields,
            2
        ));

        let tasks = vec![
            ("signed".to_string(), ids.clone(), Some("img_payload")),
            ("reference".to_string(), vec![1, 2, 3], Some("img_payload")),
        ];
        let prepared = prepare_tasks(
            "Accompanying note",
            req_fields,
            2,
            0,
            &record,
            &tasks,
            &tokenizer,
            4096,
        )
        .unwrap();
        assert_eq!(prepared[0].1, revised_ids);
        assert_eq!(prepared[1].1, vec![1, 2, 3]);
    }
}
