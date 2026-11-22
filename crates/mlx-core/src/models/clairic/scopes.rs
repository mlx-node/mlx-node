//! Entity, financial, and identifier field-scope prompt rewriters.
//!
//! Direct port of upstream:
//! - `clairic_runtime/entity_scope.py`
//! - `clairic_runtime/financial_scope.py`
//! - `clairic_runtime/identifier_scope.py`

use super::profile::ClairicProfile;
use regex::Regex;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::sync::OnceLock;

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

fn instructions_str(spec: &Map<String, Value>) -> Option<&str> {
    match spec.get("instructions") {
        None => Some(""),
        Some(Value::String(s)) => Some(s.as_str()),
        Some(_) => None,
    }
}

/// Entity name scope guidance (`clairic_runtime/entity_scope.py`).
pub mod entity {
    use super::*;

    pub const COPY_NAME_INSTRUCTION: &str =
        " Include attached digits and trailing numeric segments that are part of the printed name.";

    fn name_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"(?i)\bname\b").unwrap_or_else(|_| unreachable!()))
    }

    fn role_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(
                r"(?i)\b(?:seller|merchant|vendor|customer|bill[- ]?to|recipient|payee|payer|guest|organization|business|company|legal)\b",
            )
            .unwrap_or_else(|_| unreachable!())
        })
    }

    fn other_value_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\b(?:address|number|identifier|code|date|row|label|amount)\b")
                .unwrap_or_else(|_| unreachable!())
        })
    }

    /// Append [`COPY_NAME_INSTRUCTION`] to media entity-name text fields (`entity_scope.py::prepare`).
    pub fn prepare(fields: &Map<String, Value>, has_media: bool) -> Map<String, Value> {
        if !has_media {
            return fields.clone();
        }
        let mut result = fields.clone();
        for (name, spec_val) in fields {
            let Some(spec) = spec_val.as_object() else {
                continue;
            };
            if spec.get("type").and_then(Value::as_str) != Some("text")
                || is_truthy(spec.get("knowledge"))
            {
                continue;
            }
            let Some(instructions) = instructions_str(spec) else {
                continue;
            };
            let requested = instructions.split(['.', '\n']).next().unwrap_or("");
            if name_regex().is_match(requested)
                && role_regex().is_match(requested)
                && !other_value_regex().is_match(requested)
            {
                let mut updated = spec.clone();
                updated.insert(
                    "instructions".to_string(),
                    Value::String(format!("{instructions}{COPY_NAME_INSTRUCTION}")),
                );
                result.insert(name.clone(), Value::Object(updated));
            }
        }
        result
    }
}

/// Financial row-list scope guidance (`clairic_runtime/financial_scope.py`).
pub mod financial {
    use super::*;

    pub const TAX_INSTRUCTION: &str = " A separately printed tax row is present even when its amount is zero or negative, including a tax refund. Copy every matching printed tax row. Do not return null or end the list while a matching printed tax row remains.";
    pub const FEE_INSTRUCTION: &str = " A separately printed fee or surcharge row is present even when its amount is zero or negative, including a fee refund. Copy every matching printed fee or surcharge row. Do not return null or end the list while a matching printed fee or surcharge row remains.";
    pub const SERVICE_INSTRUCTION: &str = " Use the row's own printed label. Include an ordinary billed product or service charge. Exclude a row whose own label identifies it as a separately printed tax, fee, surcharge, subtotal, total or payment. A neighboring fee or tax does not exclude an otherwise eligible service row.";

    fn positive_split_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"(?i)[.\n]|\bExclude\b").unwrap_or_else(|_| unreachable!()))
    }

    fn tax_positive_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\A\s*Only\b[^.\n]*\btax(?:es)?\b[^.\n]*\brows\b")
                .unwrap_or_else(|_| unreachable!())
        })
    }

    fn tax_exclude_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(
                r"(?i)\b(?:fees?|products?|services?|goods|summar\w*|calculat\w*|group\w*)\b",
            )
            .unwrap_or_else(|_| unreachable!())
        })
    }

    fn fee_positive_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\A\s*Only\b[^.\n]*\bfees?\b[^.\n]*\brows\b")
                .unwrap_or_else(|_| unreachable!())
        })
    }

    fn fee_exclude_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(
                r"(?i)\b(?:tax(?:es)?|products?|services?|goods|summar\w*|calculat\w*|group\w*|deduplicat\w*|aggregat\w*|combin\w*|normaliz\w*)\b",
            )
            .unwrap_or_else(|_| unreachable!())
        })
    }

    fn service_positive_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\A\s*Only\s+billed\b[^.\n]*\brows\b")
                .unwrap_or_else(|_| unreachable!())
        })
    }

    fn service_goods_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\b(?:goods|products?|services?)\b").unwrap_or_else(|_| unreachable!())
        })
    }

    fn service_exclude_positive_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(
                r"(?i)\b(?:tax(?:es)?|fees?|surcharges?|summar\w*|calculat\w*|group\w*|deduplicat\w*|aggregat\w*|combin\w*|normaliz\w*)\b",
            )
            .unwrap_or_else(|_| unreachable!())
        })
    }

    fn service_exclude_clause_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\bExclude\b[^.\n]*\b(?:fees?|surcharges?)\b")
                .unwrap_or_else(|_| unreachable!())
        })
    }

    /// Rewrite text-only two-column financial row-list instructions (`financial_scope.py::prepare`).
    pub fn prepare(
        fields: &Map<String, Value>,
        has_media: bool,
        fee_guidance: bool,
        service_guidance: bool,
    ) -> Map<String, Value> {
        if has_media {
            return fields.clone();
        }
        let mut result = fields.clone();
        for (name, spec_val) in fields {
            let Some(spec) = spec_val.as_object() else {
                continue;
            };
            if spec.get("type").and_then(Value::as_str) != Some("list") {
                continue;
            }
            let Some(item) = spec.get("items").and_then(Value::as_object) else {
                continue;
            };
            if item.get("type").and_then(Value::as_str) != Some("object") {
                continue;
            }
            let Some(instructions) = instructions_str(spec) else {
                continue;
            };
            let Some(props_map) = item.get("properties").and_then(Value::as_object) else {
                continue;
            };
            if props_map.len() != 2 {
                continue;
            }
            let mut num_count = 0usize;
            let mut text_count = 0usize;
            let mut all_objs = true;
            let mut any_knowledge = false;
            for child in props_map.values() {
                let Some(c_obj) = child.as_object() else {
                    all_objs = false;
                    break;
                };
                match c_obj.get("type").and_then(Value::as_str) {
                    Some("number") => num_count += 1,
                    Some("text") => text_count += 1,
                    _ => {}
                }
                if is_truthy(c_obj.get("knowledge")) {
                    any_knowledge = true;
                }
            }
            if !all_objs
                || num_count != 1
                || text_count != 1
                || is_truthy(spec.get("knowledge"))
                || any_knowledge
            {
                continue;
            }

            let positive = match positive_split_regex().find(instructions) {
                Some(m) => &instructions[..m.start()],
                None => instructions,
            };

            let addition = if tax_positive_regex().is_match(positive)
                && !tax_exclude_regex().is_match(positive)
            {
                Some(TAX_INSTRUCTION)
            } else if fee_guidance
                && fee_positive_regex().is_match(positive)
                && !fee_exclude_regex().is_match(positive)
            {
                Some(FEE_INSTRUCTION)
            } else if service_guidance
                && service_positive_regex().is_match(positive)
                && service_goods_regex().is_match(positive)
                && !service_exclude_positive_regex().is_match(positive)
                && service_exclude_clause_regex().is_match(instructions)
            {
                Some(SERVICE_INSTRUCTION)
            } else {
                None
            };

            if let Some(extra) = addition {
                let mut updated = spec.clone();
                updated.insert(
                    "instructions".to_string(),
                    Value::String(format!("{instructions}{extra}")),
                );
                result.insert(name.clone(), Value::Object(updated));
            }
        }
        result
    }

    /// Profile-aware wrapper preserving Flash (`service_guidance`) vs 27B (`fee_guidance`).
    pub fn prepare_for_profile(
        fields: &Map<String, Value>,
        has_media: bool,
        profile: ClairicProfile,
    ) -> Map<String, Value> {
        prepare(
            fields,
            has_media,
            profile.financial_fee_guidance(),
            profile.financial_service_guidance(),
        )
    }
}

/// Identifier role-scope guidance (`clairic_runtime/identifier_scope.py`).
pub mod identifier {
    use super::*;

    pub const ROLE_NOTE: &str = " Select the value by its printed label and requested role. Do not infer the role from the spelling or prefix of an identifier. A similar-looking value under another label is not the requested value. Return null if that role is absent.";

    const GENERIC_NAMES: &[&str] = &["reference", "value"];

    fn purchase_order_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\bpurchase[\s-]+order\s+reference\b")
                .unwrap_or_else(|_| unreachable!())
        })
    }

    fn literal_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\b(?:exact|printed|verbatim|copy|extract|read)\b")
                .unwrap_or_else(|_| unreachable!())
        })
    }

    fn excluded_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\b(?:not|exclude|excluding|ignore|rather than)\b[^.;:\n]*$")
                .unwrap_or_else(|_| unreachable!())
        })
    }

    fn other_role_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(
                r"(?i)\b(?:invoice|receipt|payment|tracking|sales[\s-]+order|account)\s+(?:reference|number|identifier|id|code)\b",
            )
            .unwrap_or_else(|_| unreachable!())
        })
    }

    fn transform_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(
                r"(?i)\b(?:compare|combine|join|concatenate|compute|calculate|sum|count|hash|mask|normalize|convert|format|summarize|(?:first|last)\s+\d+)\b",
            )
            .unwrap_or_else(|_| unreachable!())
        })
    }

    fn state_text_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\b(?:state|accompanying)\s+(?:state\s+)?text\b")
                .unwrap_or_else(|_| unreachable!())
        })
    }

    fn quoted_literal_target_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(
                r#"(?i)\b(?:copy|extract|read|return)\s+(?:the\s+)?(?:(?:exact|printed|literal|verbatim)\s+)*(?:phrase|words?|sentence|string)\s*:?\s*(?:"([^"\n]+)"|'([^'\n]+)'|“([^”\n]+)”|‘([^’\n]+)’|`([^`\n]+)`)"#,
            )
            .unwrap_or_else(|_| unreachable!())
        })
    }

    fn literal_command_negation_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\b(?:never|don['’]t)\b[^.;:\n]*$").unwrap_or_else(|_| unreachable!())
        })
    }

    fn positive(pattern: &Regex, instructions: &str) -> bool {
        pattern
            .find_iter(instructions)
            .any(|m| !excluded_regex().is_match(&instructions[..m.start()]))
    }

    fn literal_target(instructions: &str) -> bool {
        for caps in quoted_literal_target_regex().captures_iter(instructions) {
            let start = caps.get(0).map(|m| m.start()).unwrap_or(0);
            let prefix = &instructions[..start];
            if !(excluded_regex().is_match(prefix)
                || literal_command_negation_regex().is_match(prefix))
            {
                for idx in 1..=5 {
                    if let Some(m) = caps.get(idx)
                        && purchase_order_regex().is_match(m.as_str())
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Rewrite generic media purchase-order reference fields and return `(fields, query_labels)`
    /// (`identifier_scope.py::prepare_fields`).
    pub fn prepare_fields(
        fields: &Map<String, Value>,
        has_media: bool,
    ) -> (Map<String, Value>, BTreeMap<String, String>) {
        if !has_media {
            return (fields.clone(), BTreeMap::new());
        }
        let mut changed: Map<String, Value> = Map::new();
        let mut labels: BTreeMap<String, String> = BTreeMap::new();

        for (name, spec_val) in fields {
            let lowered = name.to_lowercase();
            if !GENERIC_NAMES.contains(&lowered.as_str()) {
                continue;
            }
            let Some(spec) = spec_val.as_object() else {
                continue;
            };
            if spec.get("type").and_then(Value::as_str) != Some("text")
                || is_truthy(spec.get("knowledge"))
                || spec.get("nullable") == Some(&Value::Bool(false))
                || spec.contains_key("enum")
            {
                continue;
            }
            let Some(instructions) = instructions_str(spec) else {
                continue;
            };
            if !literal_regex().is_match(instructions)
                || transform_regex().is_match(instructions)
                || state_text_regex().is_match(instructions)
                || literal_target(instructions)
                || !positive(purchase_order_regex(), instructions)
                || positive(other_role_regex(), instructions)
            {
                continue;
            }
            if !instructions.ends_with(ROLE_NOTE) {
                let mut updated = spec.clone();
                updated.insert(
                    "instructions".to_string(),
                    Value::String(format!("{instructions}{ROLE_NOTE}")),
                );
                changed.insert(name.clone(), Value::Object(updated));
            }
            labels.insert(name.clone(), "purchase_order_reference".to_string());
        }

        if changed.is_empty() {
            (fields.clone(), labels)
        } else {
            let mut merged = fields.clone();
            for (k, v) in changed {
                merged.insert(k, v);
            }
            (merged, labels)
        }
    }
}

pub use entity::prepare as prepare_entity_scope;
pub use financial::prepare as prepare_financial_scope;
pub use financial::prepare_for_profile as prepare_financial_scope_for_profile;
pub use identifier::prepare_fields as prepare_identifier_scope;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn entity_scope_rewrites_media_role_names_only() {
        let fields = json!({
            "vendor": {"type": "text", "instructions": "Printed merchant name. Do not include address."},
            "addr": {"type": "text", "instructions": "Printed vendor name and address."},
            "know": {"type": "text", "knowledge": true, "instructions": "Company name."},
            "num": {"type": "number", "instructions": "Vendor name"}
        });
        let map = fields.as_object().unwrap();

        // No media => unchanged
        let no_media = entity::prepare(map, false);
        assert_eq!(no_media, *map);

        // With media => only vendor gets COPY_NAME_INSTRUCTION
        let with_media = entity::prepare(map, true);
        assert!(
            with_media["vendor"]["instructions"]
                .as_str()
                .unwrap()
                .ends_with(entity::COPY_NAME_INSTRUCTION)
        );
        assert_eq!(with_media["addr"], map["addr"]);
        assert_eq!(with_media["know"], map["know"]);
        assert_eq!(with_media["num"], map["num"]);
    }

    #[test]
    fn financial_scope_respects_flash_vs_27b_and_media() {
        let row_item = json!({
            "type": "object",
            "properties": {
                "label": {"type": "text"},
                "amount": {"type": "number"}
            }
        });
        let fields = json!({
            "taxes": {
                "type": "list",
                "items": row_item,
                "instructions": "Only printed tax rows. Exclude subtotals."
            },
            "fees": {
                "type": "list",
                "items": row_item,
                "instructions": "Only printed fee rows. Exclude taxes."
            },
            "services": {
                "type": "list",
                "items": row_item,
                "instructions": "Only billed service rows. Exclude fees and surcharges."
            },
            "mixed": {
                "type": "list",
                "items": row_item,
                "instructions": "Only tax and fee rows."
            }
        });
        let map = fields.as_object().unwrap();

        // Media disables financial_scope
        assert_eq!(
            financial::prepare_for_profile(map, true, ClairicProfile::Flash),
            *map
        );

        // Flash: tax + service guidance, no fee guidance
        let flash = financial::prepare_for_profile(map, false, ClairicProfile::Flash);
        assert!(
            flash["taxes"]["instructions"]
                .as_str()
                .unwrap()
                .ends_with(financial::TAX_INSTRUCTION)
        );
        assert_eq!(flash["fees"], map["fees"]);
        assert!(
            flash["services"]["instructions"]
                .as_str()
                .unwrap()
                .ends_with(financial::SERVICE_INSTRUCTION)
        );
        assert_eq!(flash["mixed"], map["mixed"]);

        // 27B: tax + fee guidance, no service guidance
        let large = financial::prepare_for_profile(map, false, ClairicProfile::Large27B);
        assert!(
            large["taxes"]["instructions"]
                .as_str()
                .unwrap()
                .ends_with(financial::TAX_INSTRUCTION)
        );
        assert!(
            large["fees"]["instructions"]
                .as_str()
                .unwrap()
                .ends_with(financial::FEE_INSTRUCTION)
        );
        assert_eq!(large["services"], map["services"]);
    }

    #[test]
    fn identifier_scope_matches_po_reference_and_rejects_exclusions() {
        let fields = json!({
            "Reference": {
                "type": "text",
                "instructions": "Extract the printed purchase order reference. Exclude invoice number."
            },
            "value": {
                "type": "text",
                "instructions": "Copy the exact phrase \"purchase order reference\"."
            },
            "other": {
                "type": "text",
                "instructions": "Extract the printed purchase order reference."
            }
        });
        let map = fields.as_object().unwrap();

        // Without media => empty
        let (no_m, no_l) = identifier::prepare_fields(map, false);
        assert_eq!(no_m, *map);
        assert!(no_l.is_empty());

        // With media => "Reference" matches (invoice number is excluded!), "value" rejected (quoted literal target), "other" rejected (non-generic name)
        let (res, labels) = identifier::prepare_fields(map, true);
        assert_eq!(labels.len(), 1);
        assert_eq!(
            labels.get("Reference").map(String::as_str),
            Some("purchase_order_reference")
        );
        assert!(
            res["Reference"]["instructions"]
                .as_str()
                .unwrap()
                .ends_with(identifier::ROLE_NOTE)
        );
        assert_eq!(res["value"], map["value"]);
        assert_eq!(res["other"], map["other"]);

        // Idempotent when ROLE_NOTE already present
        let (res2, labels2) = identifier::prepare_fields(&res, true);
        assert_eq!(res2["Reference"], res["Reference"]);
        assert_eq!(labels2, labels);
    }

    #[test]
    fn identifier_scope_rejects_other_role_transform_state_and_nonnullable() {
        let fields = json!({
            "reference": {
                "type": "text",
                "instructions": "Extract the printed purchase order reference or invoice number."
            },
            "value": {
                "type": "text",
                "instructions": "Read the purchase-order reference from the accompanying state text."
            }
        });
        let (_, labels) = identifier::prepare_fields(fields.as_object().unwrap(), true);
        assert!(labels.is_empty());

        let non_nullable = json!({
            "reference": {
                "type": "text",
                "nullable": false,
                "instructions": "Extract the printed purchase order reference."
            }
        });
        let (_, labels_nn) = identifier::prepare_fields(non_nullable.as_object().unwrap(), true);
        assert!(labels_nn.is_empty());
    }
}
