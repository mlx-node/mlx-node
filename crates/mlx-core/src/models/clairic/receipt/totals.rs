//! Receipt block parsing, exact meet-in-the-middle subset sum verification,
//! boolean AR task builder, and receipt-sum validation/merge.
//!
//! Direct port of upstream `clairic_runtime/receipt_totals.py` (plus the boolean
//! AR repair task builder & merge helpers from `backend.py:1175-1291`).

use super::{
    accurate_fsum, canonical as canonical_bytes, char_to_byte_map, extract_span,
    find_char_spans_fancy, find_char_spans_in_range, hybrid_inventory, is_currency_symbol,
    sha256_hex, slice_chars, source_sums as sums,
};
use crate::grammar::GrammarSpec;
use crate::models::clairic::{
    ar, ar_engine::ArGenTask, decimal_util, json_validate, schema, source_dates,
};
use bigdecimal::{BigDecimal, Zero, num_traits::ToPrimitive};
use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

pub const MAX_BLOCKS: usize = 128;
pub const MAX_LABEL_AMOUNT_GAP: usize = 24;
pub const MAX_SUPPORT_BLOCKS: usize = 24;
pub const MAX_SUPPORT_PRECISION: usize = 256;

/// Max output tokens per boolean receipt-block inclusion task (`backend.py:1220`).
pub const BOOLEAN_MAX_NEW_TOKENS: usize = 8;

/// Batch size for executing receipt boolean AR tasks (`backend.py:1249`).
pub const BOOLEAN_BATCH_SIZE: usize = 32;

/// Decoder identifier recorded on merged receipt-sum repairs (`backend.py:1279`).
pub const RECEIPT_BLOCK_SUM_DECODER: &str = "source_receipt_block_sum_v1";

pub(crate) const SHARED_PROMPT_GUIDANCE: &str = "\nDecide whether the candidate block is one receipt record in the requested sum. Use its category, entity, date and the original requested scope. Return true if and only if this record's explicit total is one requested operand. A total label alone does not establish inclusion. Hotel, airline and claim-summary blocks remain candidates; include them only when the requested receipt scope includes them. A claim summary is not an additional underlying receipt. Do not separately select food, subtotal, tax, tip or other components already included in a total. A zero or refund remains a present amount. Distinct source records remain distinct even with equal totals. Do not calculate the sum. Return false for an excluded record; return null if the evidence does not determine its inclusion. No expected values are supplied. Return exactly one JSON Boolean or null, then end immediately.\n";

fn currency_alternation() -> String {
    sums::CURRENCY_CODES.join("|")
}

fn receipt_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\breceipts?\b").unwrap_or_else(|_| unreachable!()))
}

fn requested_total_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\btotals?\b").unwrap_or_else(|_| unreachable!()))
}

fn compound_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(
            r"(?i)\b(?:averag\w*|mean|median|subtract\w*|deduct\w*|minus|less|plus|add(?:s|ed|ing)?|multipl\w*|times|divid\w*|quotient|ratio|percent(?:age)?|discount\w*|surcharge\w*|convert\w*|conversion\w*|exchange|fx|clamp\w*|cap(?:ped)?|net[ \t]+of)\b|[*/\u{00d7}\u{00f7}%=]|(?<=\s)[+-][ \t]*\d|\b(?:in|to)[ \t]+cents?\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn round_each_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let round = r"(?:round\w*|ceil\w*|floor\w*|truncat\w*)";
        let pat = format!(
            r"(?i)\b{round}\b[^.;\n]*(?:\b(?:each|every|per|individually|before)\b|\breceipts?\b)|\b(?:each|every|per|individual(?:ly)?)\b[^.;\n]*\b{round}\b"
        );
        Regex::new(&pat).unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn total_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:amount[ \t]+charged|(?:grand[ \t]+)?total(?:[ \t]+(?:charges|claimed|paid|due|amount))?)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn separator_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\r?\n[ \t]*\r?\n").unwrap_or_else(|_| unreachable!()))
}

pub(crate) fn record_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:receipt|payment)\b").unwrap_or_else(|_| unreachable!()))
}

pub(crate) fn reference_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:(?:room|receipt)[ \t]+\d+|(?:odometer[ \t]+(?:start|end)|(?:trip[ \t]+)?miles)[ \t]+\d+|[a-z][a-z0-9_]*[-/#][a-z0-9_/-]*\d[a-z0-9_/-]*|[a-z]+\d+[a-z0-9]*)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn measurement_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b\d+(?:\.\d+)?[ \t]+(?:miles?|nights?)\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn distance_rate_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(r"(?i)(?<!\w)(?:\$|USD[ \t]+)?\d+(?:\.\d+)?/miles?\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn card_reference_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\*{2,}\d+\b").unwrap_or_else(|_| unreachable!()))
}

pub(crate) fn confirmation_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:conf(?:irmation)?|booking|reference)[ \t]*[#:]?[ \t]+[a-z0-9]+\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn clock_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn short_date_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        let month = source_dates::MONTH_PATTERN;
        let pat = format!(r"(?i)(?<!\w)(?:{month}[ \t]+\d{{1,2}}|\d{{1,2}}[ \t-]+{month})(?!\w)");
        FancyRegex::new(&pat).unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn date_cell_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(r"(?m)^[ \t]*(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])(?=[ \t])")
            .unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn component_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:sub[ \t-]?total|food|tax(?:es)?(?:/fees)?|tip|fare|room(?:[ \t]+charge)?|occupancy[ \t]+tax|facility[ \t]+fee|balance[ \t]+due|ticket[ \t]+was)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn component_connector_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:incl(?:uding)?|and)\b").unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn amount_gap_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\A[\s.,:;=()+\-\u{2212}/&*]*\z").unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn bridge_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let pat = format!(
            r"(?i)\A(?:[\s:=()$+\-\u{{2212}}]|{})*\z",
            currency_alternation()
        );
        Regex::new(&pat).unwrap_or_else(|_| unreachable!())
    })
}

pub(crate) fn currency_code_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let pat = format!(r"(?i)\b(?:{})\b", currency_alternation());
        Regex::new(&pat).unwrap_or_else(|_| unreachable!())
    })
}

fn currency_symbol_codes(ch: char) -> Option<&'static [&'static str]> {
    match ch {
        '$' => Some(&["USD", "AUD", "CAD", "HKD", "NZD", "SGD", "MXN", "BRL"]),
        '€' => Some(&["EUR"]),
        '£' => Some(&["GBP"]),
        '¥' => Some(&["JPY", "CNY"]),
        '₹' => Some(&["INR"]),
        '₩' => Some(&["KRW"]),
        _ => None,
    }
}

fn currency_symbol_str_codes(s: &str) -> Option<&'static [&'static str]> {
    let mut chars = s.chars();
    let first = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    currency_symbol_codes(first)
}

/// Canonical JSON bytes matching `receipt_totals.py::canonical`.
pub fn canonical(value: &Value) -> Vec<u8> {
    canonical_bytes(value)
}

/// Split `source` into trimmed non-empty paragraph Unicode character spans (`receipt_totals.py::_paragraphs`).
pub fn paragraphs(source: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = source.chars().collect();
    let b2c = schema::byte_to_char_map(source);
    let mut spans = Vec::new();
    let mut start_char = 0usize;

    let mut boundaries: Vec<Option<(usize, usize)>> = separator_regex()
        .find_iter(source)
        .map(|m| Some((b2c[m.start()], b2c[m.end()])))
        .collect();
    boundaries.push(None);

    for boundary in boundaries {
        let end_char = match boundary {
            Some((b_start, _)) => b_start,
            None => chars.len(),
        };
        let mut left = start_char;
        let mut right = end_char;
        while left < right && chars[left].is_whitespace() {
            left += 1;
        }
        while right > left && chars[right - 1].is_whitespace() {
            right -= 1;
        }
        if left < right {
            spans.push((left, right));
        }
        if let Some((_, b_end)) = boundary {
            start_char = b_end;
        }
    }

    spans
}

/// Collect non-amount metadata spans (dates, references, measurements, card refs, clocks, etc.)
/// as Unicode character offset tuples.
pub fn metadata_spans(source: &str) -> Vec<(usize, usize)> {
    let b2c = schema::byte_to_char_map(source);
    let mut spans: Vec<(usize, usize)> = source_dates::candidates(source, None)
        .unwrap_or_default()
        .into_iter()
        .map(|item| (item.start, item.end))
        .collect();

    for m in reference_regex().find_iter(source) {
        let leading_alpha: String = m
            .as_str()
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect();
        let upper = leading_alpha.to_ascii_uppercase();
        if !sums::is_currency_code(&upper) {
            spans.push((b2c[m.start()], b2c[m.end()]));
        }
    }

    for pattern in [
        measurement_regex(),
        card_reference_regex(),
        confirmation_regex(),
        clock_regex(),
    ] {
        for m in pattern.find_iter(source) {
            spans.push((b2c[m.start()], b2c[m.end()]));
        }
    }

    for pattern in [distance_rate_regex(), short_date_regex(), date_cell_regex()] {
        spans.extend(find_char_spans_fancy(pattern, source, &b2c));
    }

    spans
}

pub(crate) fn is_non_amount_item(
    chars: &[char],
    metadata: &[(usize, usize)],
    item: &Value,
) -> bool {
    let Some((left, right)) = extract_span(item.get("span")) else {
        return false;
    };
    metadata.iter().any(|&(m_start, m_end)| {
        right <= m_end
            && (m_start <= left
                || (left < m_start
                    && m_start <= chars.len()
                    && chars[left..m_start]
                        .iter()
                        .all(|c| matches!(c, ' ' | '\t' | '+' | '-'))))
    })
}

pub(crate) fn replace_sc_with(text: &str, replacement: char) -> String {
    text.chars()
        .map(|c| {
            if is_currency_symbol(c) {
                replacement
            } else {
                c
            }
        })
        .collect()
}

/// Parse structured single-total receipt blocks from `task` (`receipt_totals.py::_blocks`).
pub(crate) fn blocks_in_paragraphs(
    task: &Value,
    explicit_paragraphs: Option<&[(usize, usize)]>,
) -> Option<Vec<Value>> {
    let source = task.get("input_text")?.as_str()?;
    let candidates = task.get("candidates")?.as_array()?;
    let chars: Vec<char> = source.chars().collect();
    let c2b = char_to_byte_map(source);
    let b2c = schema::byte_to_char_map(source);
    let metadata = metadata_spans(source);

    let owned_paragraphs;
    let para_slice: &[(usize, usize)] = match explicit_paragraphs {
        Some(p) => p,
        None => {
            owned_paragraphs = paragraphs(source);
            &owned_paragraphs
        }
    };

    let mut blocks = Vec::new();
    for &(start, end) in para_slice {
        let labels = find_char_spans_in_range(total_regex(), source, &c2b, &b2c, start, end);
        let numbers: Vec<&Value> = candidates
            .iter()
            .filter(|item| {
                extract_span(item.get("span")).is_some_and(|(s, _)| start <= s && s < end)
            })
            .collect();

        if labels.is_empty() {
            let text = slice_chars(source, &c2b, start, end);
            if numbers
                .iter()
                .any(|item| !is_non_amount_item(&chars, &metadata, item))
                || record_regex().is_match(text)
                || currency_code_regex().is_match(text)
                || text.chars().any(is_currency_symbol)
            {
                return None;
            }
            continue;
        }

        let record_count =
            find_char_spans_in_range(record_regex(), source, &c2b, &b2c, start, end).len();
        if labels.len() != 1 || record_count > 1 {
            return None;
        }
        let (label_start, label_end) = labels[0];

        let mut linked: Vec<&Value> = Vec::new();
        for item in candidates {
            let Some((left, right)) = extract_span(item.get("span")) else {
                continue;
            };
            if !(label_end <= left && left <= label_end + MAX_LABEL_AMOUNT_GAP) || right > end {
                continue;
            }
            let gap = replace_sc_with(slice_chars(source, &c2b, label_end, left), '$');
            if bridge_regex().is_match(&gap) {
                linked.push(item);
            }
        }
        if linked.is_empty() {
            for item in candidates {
                let Some((left, right)) = extract_span(item.get("span")) else {
                    continue;
                };
                if left < start
                    || !(right <= label_start && label_start <= right + MAX_LABEL_AMOUNT_GAP)
                {
                    continue;
                }
                let gap = replace_sc_with(slice_chars(source, &c2b, right, label_start), '$');
                if bridge_regex().is_match(&gap) {
                    linked.push(item);
                }
            }
        }
        if linked.len() != 1 {
            return None;
        }
        let chosen = linked[0];
        let chosen_id = chosen.get("id").and_then(Value::as_str);

        for &other in &numbers {
            if other.get("id").and_then(Value::as_str) == chosen_id
                || is_non_amount_item(&chars, &metadata, other)
            {
                continue;
            }
            let (left, _) = extract_span(other.get("span"))?;
            let comp_start = start.max(left.saturating_sub(48));
            let components =
                find_char_spans_in_range(component_regex(), source, &c2b, &b2c, comp_start, left);
            let has_bridged_component = components.iter().any(|&(_, comp_end)| {
                let gap = replace_sc_with(slice_chars(source, &c2b, comp_end, left), '$');
                bridge_regex().is_match(&gap)
            });
            if !has_bridged_component {
                return None;
            }
        }

        let monetary: Vec<&Value> = numbers
            .iter()
            .copied()
            .filter(|other| !is_non_amount_item(&chars, &metadata, other))
            .collect();

        for pair in monetary.windows(2) {
            let (Some((_, left)), Some((right, _))) = (
                extract_span(pair[0].get("span")),
                extract_span(pair[1].get("span")),
            ) else {
                return None;
            };
            if left > right || right > chars.len() {
                return None;
            }
            let mut gap_chars: Vec<char> = chars[left..right].to_vec();
            for &(meta_left, meta_right) in &metadata {
                let lo = left.max(meta_left);
                let hi = right.min(meta_right);
                for offset in lo..hi {
                    gap_chars[offset - left] = ' ';
                }
            }
            let mut gap: String = gap_chars
                .into_iter()
                .map(|c| if is_currency_symbol(c) { ' ' } else { c })
                .collect();
            for pattern in [
                total_regex(),
                component_regex(),
                component_connector_regex(),
                currency_code_regex(),
            ] {
                gap = pattern.replace_all(&gap, " ").into_owned();
            }
            if !amount_gap_regex().is_match(&gap) {
                return None;
            }
        }

        let mut occ = Map::new();
        occ.insert(
            "id".to_string(),
            chosen.get("id").cloned().unwrap_or(Value::Null),
        );
        occ.insert(
            "span".to_string(),
            chosen.get("span").cloned().unwrap_or(Value::Null),
        );
        occ.insert(
            "printed".to_string(),
            chosen.get("printed").cloned().unwrap_or(Value::Null),
        );

        let mut block = Map::new();
        block.insert("block_span".to_string(), json!([start, end]));
        block.insert(
            "source_block".to_string(),
            Value::String(slice_chars(source, &c2b, start, end).to_string()),
        );
        block.insert(
            "total_label".to_string(),
            Value::String(slice_chars(source, &c2b, label_start, label_end).to_string()),
        );
        block.insert(
            "total_label_span".to_string(),
            json!([label_start, label_end]),
        );
        block.insert("total_occurrence".to_string(), Value::Object(occ));
        blocks.push(Value::Object(block));

        if blocks.len() > MAX_BLOCKS {
            return None;
        }
    }

    if blocks.is_empty() {
        None
    } else {
        Some(blocks)
    }
}

/// Parse structured receipt blocks from a `source_sums` task (`receipt_totals.py::_blocks`).
pub fn parse_blocks(task: &Value) -> Option<Vec<Value>> {
    blocks_in_paragraphs(task, None)
}

/// Check that the field unit, instructions, and all candidate blocks have a non-empty
/// intersection of compatible currency codes (`receipt_totals.py::_compatible_currencies`).
pub(crate) fn compatible_currencies<'a, I>(task: &'a Value, block_texts: I) -> bool
where
    I: IntoIterator<Item = &'a str>,
{
    let mut allowed: HashSet<&'static str> = sums::CURRENCY_CODES.iter().copied().collect();
    let Some(field_spec) = task.get("field_spec") else {
        return false;
    };

    if let Some(unit) = field_spec.get("unit")
        && !unit.is_null()
        && unit.as_str() != Some("")
    {
        let Some(unit_str) = unit.as_str() else {
            return false;
        };
        let trimmed = unit_str.trim();
        let normalized = trimmed.to_ascii_uppercase();
        if sums::is_currency_code(&normalized) {
            allowed.retain(|c| *c == normalized);
        } else if let Some(codes) = currency_symbol_str_codes(trimmed) {
            allowed.retain(|c| codes.contains(c));
        } else {
            return false;
        }
    }

    let instructions = field_spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    for text in std::iter::once(instructions).chain(block_texts) {
        for m in currency_code_regex().find_iter(text) {
            let upper = m.as_str().to_ascii_uppercase();
            allowed.retain(|c| *c == upper);
        }
        for ch in text.chars() {
            if is_currency_symbol(ch) {
                if let Some(codes) = currency_symbol_codes(ch) {
                    allowed.retain(|c| codes.contains(c));
                } else {
                    allowed.clear();
                }
            }
        }
        if allowed.is_empty() {
            return false;
        }
    }

    true
}

pub(crate) fn build_shared_prefix(task: &Value) -> Option<String> {
    let input_text = task.get("input_text")?.as_str()?;
    let field = task.get("field")?.as_str()?;
    let field_spec = task.get("field_spec")?;
    Some(format!(
        "Classify each source block using the entire supplied document.\nOriginal document:\n{input_text}\n\nRequested numeric field: {field}\nOriginal field schema and instructions: {}{SHARED_PROMPT_GUIDANCE}",
        ar::compact_json(field_spec)
    ))
}

fn from_sum_task(task: &Value) -> Option<Value> {
    let blocks = blocks_in_paragraphs(task, None)?;
    let block_texts: Vec<&str> = blocks
        .iter()
        .filter_map(|b| b.get("source_block").and_then(Value::as_str))
        .collect();
    if !compatible_currencies(task, block_texts) {
        return None;
    }

    let prefix = build_shared_prefix(task)?;
    let mut tasks = Vec::with_capacity(blocks.len());
    for block in &blocks {
        let suffix = format!("\nCandidate source block: {}", ar::compact_json(block));
        let occ_id = block
            .get("total_occurrence")
            .and_then(|o| o.get("id"))
            .and_then(Value::as_str)?;
        let mut t = Map::new();
        t.insert(
            "occurrence_id".to_string(),
            Value::String(occ_id.to_string()),
        );
        t.insert("suffix".to_string(), Value::String(suffix.clone()));
        t.insert(
            "prompt".to_string(),
            Value::String(format!("{prefix}{suffix}")),
        );
        t.insert(
            "json_schema".to_string(),
            json!({"type": ["boolean", "null"]}),
        );
        t.insert(
            "grammar".to_string(),
            json!({"ebnf": hybrid_inventory::BOOLEAN_EBNF}),
        );
        tasks.push(Value::Object(t));
    }

    let prefix_sha = sha256_hex(prefix.as_bytes());
    let mut packet = Map::new();
    packet.insert("version".to_string(), json!(1));
    packet.insert(
        "kind".to_string(),
        json!("source-receipt-block-boolean-sum-diagnostic-v1"),
    );
    packet.insert("sum_task".to_string(), task.clone());
    packet.insert("shared_prefix".to_string(), Value::String(prefix));
    packet.insert(
        "shared_prefix_sha256".to_string(),
        Value::String(prefix_sha),
    );
    packet.insert("tasks".to_string(), Value::Array(tasks));
    packet.insert("blocks".to_string(), Value::Array(blocks));
    packet.insert(
        "all_candidates_require_complete_decisions".to_string(),
        Value::Bool(true),
    );
    packet.insert("native_typed_boolean_head".to_string(), Value::Bool(false));
    packet.insert("runtime_changes".to_string(), Value::Bool(false));
    packet.insert(
        "semantic_completeness_proven_by_grounding".to_string(),
        Value::Bool(false),
    );
    Some(Value::Object(packet))
}

/// Return `true` iff `field_name` in `record` is eligible for receipt-total summing
/// (`receipt_totals.py::eligible`).
pub fn eligible(record: &Value, field_name: &str, has_media: bool) -> bool {
    if !sums::eligible(record, field_name, has_media) {
        return false;
    }
    let Some(instructions) = record
        .get("fields")
        .and_then(|f| f.get(field_name))
        .and_then(|s| s.get("instructions"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    receipt_regex().is_match(instructions)
        && requested_total_regex().is_match(instructions)
        && !compound_regex().is_match(instructions).unwrap_or(true)
        && !round_each_regex().is_match(instructions)
}

/// Return names of eligible receipt-sum fields in a text-only request (`backend.py:1351-1366`).
pub fn eligible_fields(state: &str, fields: &Map<String, Value>, has_media: bool) -> Vec<String> {
    if has_media {
        return Vec::new();
    }
    let source = json!({
        "input": state,
        "fields": Value::Object(fields.clone()),
    });
    fields
        .keys()
        .filter(|name| eligible(&source, name, false))
        .cloned()
        .collect()
}

/// Prepare a receipt-total boolean-selection packet (either structured block packet or
/// hybrid paragraph runtime packet) (`receipt_totals.py::prepare_task`).
pub fn prepare_task(record: &Value, field_name: &str, has_media: bool) -> Option<Value> {
    if !eligible(record, field_name, has_media) {
        return None;
    }
    let task = sums::prepare_task(record, field_name, has_media)?;
    if let Some(legacy) = from_sum_task(&task) {
        return Some(legacy);
    }
    let packet = hybrid_inventory::from_sum_task(&task)?;
    let currency_ok = packet
        .get("currency_compatible")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let unresolved_empty = packet
        .get("unresolved_fragments_without_numeric_candidates")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty);
    let task_count = packet
        .get("tasks")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(usize::MAX);
    if !currency_ok || !unresolved_empty || task_count > MAX_BLOCKS {
        return None;
    }
    Some(packet)
}

fn canonical_decimal(d: &BigDecimal) -> BigDecimal {
    if d.is_zero() {
        BigDecimal::from(0)
    } else {
        d.normalized()
    }
}

fn decimal_exponent(value: &BigDecimal) -> i64 {
    if value.is_zero() {
        0
    } else {
        -value.fractional_digit_count()
    }
}

/// Exact Python-equivalent `Decimal(str(ar.normalize_number(value, spec)))` preserving
/// arbitrary-precision integers (up to `MAX_SUPPORT_PRECISION`) while matching `float`
/// rounding semantics for non-integral decimals.
fn normalized_decimal_for_spec(value: &BigDecimal, spec: &Value) -> Option<BigDecimal> {
    if spec.get("type").and_then(Value::as_str) != Some("number") {
        return Some(canonical_decimal(value));
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
    match decimals {
        None => {
            if decimal_util::is_integral(value) {
                Some(canonical_decimal(&value.with_scale(0)))
            } else {
                let f = value.to_f64().filter(|f| f.is_finite())?;
                decimal_util::from_f64(f).map(|d| canonical_decimal(&d))
            }
        }
        Some(d) => {
            let rounded = decimal_util::round_half_up_checked(value, d)?;
            if d == 0 {
                Some(canonical_decimal(&rounded.with_scale(0)))
            } else {
                let f = rounded.to_f64().filter(|f| f.is_finite())?;
                decimal_util::from_f64(f).map(|dec| canonical_decimal(&dec))
            }
        }
    }
}

fn subset_totals(items: &[BigDecimal]) -> HashMap<BigDecimal, bool> {
    let cap = 1usize
        .checked_shl(items.len().min(12) as u32)
        .unwrap_or(4096);
    let mut totals: HashMap<BigDecimal, bool> = HashMap::with_capacity(cap);
    totals.insert(BigDecimal::from(0), false);
    for amount in items {
        let previous: Vec<BigDecimal> = totals.keys().cloned().collect();
        for value in previous {
            let sum = canonical_decimal(&(value + amount));
            totals.insert(sum, true);
        }
    }
    totals
}

/// Exact meet-in-the-middle subset sum verification over `<= MAX_SUPPORT_BLOCKS` (24)
/// arbitrary-precision `BigDecimal` amounts (`<= MAX_SUPPORT_PRECISION` = 256 digits)
/// (`receipt_totals.py::supports_sum`).
pub fn supports_sum_decimal(packet: &Value, target: &BigDecimal) -> Option<bool> {
    let task = packet.get("sum_task")?;
    let hybrid =
        packet.get("kind").and_then(Value::as_str) == Some(hybrid_inventory::HYBRID_PACKET_KIND);
    let blocks = packet
        .get(if hybrid { "descriptors" } else { "blocks" })?
        .as_array()?;
    if blocks.is_empty() || blocks.len() > MAX_SUPPORT_BLOCKS {
        return None;
    }

    let input_text = task.get("input_text")?.as_str()?;
    let field = task.get("field")?.as_str()?;
    let field_spec = task.get("field_spec")?;
    let recomputed = prepare_task(
        &json!({
            "input": input_text,
            "fields": { field: field_spec },
        }),
        field,
        false,
    )?;
    if canonical(&recomputed) != canonical(packet) {
        return None;
    }

    let candidates = task.get("candidates")?.as_array()?;
    let mut indexed: HashMap<&str, &Value> = HashMap::with_capacity(candidates.len());
    for item in candidates {
        if let Some(id) = item.get("id").and_then(Value::as_str) {
            indexed.insert(id, item);
        }
    }

    let mut amounts = Vec::with_capacity(blocks.len());
    for block in blocks {
        let occ_key = if hybrid {
            "occurrence"
        } else {
            "total_occurrence"
        };
        let occ_id = block.get(occ_key)?.get("id")?.as_str()?;
        let item = indexed.get(occ_id)?;
        let dec_str = item.get("decimal")?.as_str()?;
        let amount = decimal_util::parse_decimal(dec_str)?;
        amounts.push(amount);
    }

    let mut highest = decimal_util::adjusted_exponent(target);
    let mut lowest = decimal_exponent(target);
    for amount in &amounts {
        highest = highest.max(decimal_util::adjusted_exponent(amount));
        lowest = lowest.min(decimal_exponent(amount));
    }
    let count_digits = amounts.len().to_string().len() as i64;
    let required_digits = highest
        .checked_sub(lowest)?
        .checked_add(count_digits)?
        .checked_add(3)?;
    let precision = required_digits.max(28);
    if precision > MAX_SUPPORT_PRECISION as i64 {
        return None;
    }

    let target_canonical = canonical_decimal(target);
    if normalized_decimal_for_spec(&target_canonical, field_spec).as_ref()
        != Some(&target_canonical)
    {
        return None;
    }

    let middle = amounts.len() / 2;
    let left = subset_totals(&amounts[..middle]);
    let right = subset_totals(&amounts[middle..]);

    for (value, &nonempty) in &left {
        let rest = canonical_decimal(&(&target_canonical - value));
        if let Some(&right_nonempty) = right.get(&rest)
            && (nonempty || right_nonempty)
        {
            let combined = value + &rest;
            if normalized_decimal_for_spec(&combined, field_spec).as_ref()
                == Some(&target_canonical)
            {
                return Some(true);
            }
        }
    }

    if amounts.iter().any(|amount| {
        normalized_decimal_for_spec(amount, field_spec).as_ref() != Some(&canonical_decimal(amount))
    }) {
        return None;
    }
    if required_digits > 15
        && amounts
            .iter()
            .any(|amount| !decimal_util::is_integral(amount))
    {
        return None;
    }

    Some(false)
}

/// JSON-value wrapper around [`supports_sum_decimal`] matching `receipt_totals.py::supports_sum`.
pub fn supports_sum(packet: &Value, primary_value: &Value) -> Option<bool> {
    if primary_value.is_boolean() || !primary_value.is_number() {
        return None;
    }
    let target = decimal_util::from_json_value(primary_value)?;
    supports_sum_decimal(packet, &target)
}

/// Compute joint sequence log-probability and confidence across all boolean decisions
/// (`receipt_totals.py::_selection_scores`).
pub(crate) fn selection_scores(
    decisions: &Map<String, Value>,
    expected: &[String],
) -> Option<Value> {
    let mut scores = Vec::with_capacity(expected.len());
    for identifier in expected {
        let row = decisions.get(identifier)?.as_object()?;
        if row.get("valid").and_then(Value::as_bool) != Some(true)
            || row.get("complete").and_then(Value::as_bool) != Some(true)
        {
            return None;
        }
        let val = row.get("value")?.as_bool()?;
        let raw_json = row.get("raw_json")?.as_str()?;
        let parsed: Value = serde_json::from_str(raw_json).ok()?;
        if parsed.as_bool() != Some(val) {
            return None;
        }
        if row.get("decoder").and_then(Value::as_str) != Some(ar::VERSION)
            || row.get("confidence_method").and_then(Value::as_str)
                != Some("ar_sequence_probability")
        {
            return None;
        }
        let score_val = row.get("log_probability")?;
        if score_val.is_boolean() {
            return None;
        }
        let score = score_val.as_f64()?;
        if !score.is_finite() || score > 0.0 {
            return None;
        }
        scores.push(score);
    }

    let total = accurate_fsum(&scores);
    if !total.is_finite() {
        return None;
    }
    let confidence = total.exp();
    if !confidence.is_finite() {
        return None;
    }

    let mut out = Map::new();
    out.insert("log_probability".to_string(), json!(total));
    out.insert("confidence".to_string(), json!(confidence));
    out.insert(
        "confidence_method".to_string(),
        json!("receipt_block_boolean_joint_sequence_probability"),
    );
    out.insert(
        "score_scope".to_string(),
        json!("all_receipt_block_inclusion_decisions"),
    );
    out.insert("decision_count".to_string(), json!(scores.len()));
    Some(Value::Object(out))
}

/// Validate boolean receipt-block decisions against `packet` and compute the exact
/// `BigDecimal` sum (`receipt_totals.py::validate_result`).
pub fn validate_result(packet: &Value, decisions: &Map<String, Value>) -> Value {
    if packet.get("kind").and_then(Value::as_str) == Some(hybrid_inventory::HYBRID_PACKET_KIND) {
        return hybrid_inventory::validate_result(packet, decisions);
    }

    let mut result = Map::new();
    result.insert("valid".to_string(), Value::Bool(false));
    result.insert("value".to_string(), Value::Null);
    result.insert("operands".to_string(), Value::Array(Vec::new()));
    result.insert("decimal_sum".to_string(), Value::Null);
    result.insert("operand_count".to_string(), json!(0));
    result.insert("error".to_string(), Value::Null);
    result.insert(
        "selection_method".to_string(),
        json!("per_receipt_block_boolean_diagnostic"),
    );
    result.insert("decision_count".to_string(), json!(0));
    result.insert("required_decision_count".to_string(), json!(0));
    result.insert(
        "semantic_completeness_proven_by_grounding".to_string(),
        Value::Bool(false),
    );

    if let Err(err) = validate_result_inner(packet, decisions, &mut result) {
        result.insert("valid".to_string(), Value::Bool(false));
        result.insert("value".to_string(), Value::Null);
        result.insert("operands".to_string(), Value::Array(Vec::new()));
        result.insert("decimal_sum".to_string(), Value::Null);
        result.insert("operand_count".to_string(), json!(0));
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
    let packet_tasks = packet
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing_tasks".to_string())?;
    result.insert(
        "required_decision_count".to_string(),
        json!(packet_tasks.len()),
    );

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

    let recomputed = prepare_task(
        &json!({
            "input": input_text,
            "fields": { field: field_spec },
        }),
        field,
        false,
    );
    if recomputed.as_ref().map(canonical).as_deref() != Some(canonical(packet).as_slice()) {
        return Err("receipt_block_prompt_or_source_mapping_changed".to_string());
    }

    let mut expected = Vec::with_capacity(packet_tasks.len());
    for row in packet_tasks {
        let occ_id = row
            .get("occurrence_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing_occurrence_id".to_string())?;
        expected.push(occ_id.to_string());
    }

    let expected_set: HashSet<&str> = expected.iter().map(String::as_str).collect();
    let decision_set: HashSet<&str> = decisions.keys().map(String::as_str).collect();
    if decision_set != expected_set {
        return Err("receipt_block_coverage_incomplete_or_extra".to_string());
    }

    let mut selected: Vec<String> = Vec::new();
    let mut decision_count = 0usize;
    for identifier in &expected {
        let Some(row) = decisions.get(identifier).and_then(Value::as_object) else {
            return Err(format!("incomplete_boolean_generation:{identifier}"));
        };
        if row.get("complete").and_then(Value::as_bool) != Some(true) {
            return Err(format!("incomplete_boolean_generation:{identifier}"));
        }
        let raw_json = row
            .get("raw_json")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("missing_raw_json:{identifier}"))?;
        let value: Value = serde_json::from_str(raw_json).map_err(|e| e.to_string())?;
        if value.is_null() {
            return Err(format!("unresolved_boolean_inclusion:{identifier}"));
        }
        let Some(flag) = value.as_bool() else {
            return Err(format!("invalid_boolean_scalar:{identifier}"));
        };
        decision_count += 1;
        if flag {
            selected.push(identifier.clone());
        }
    }
    result.insert("decision_count".to_string(), json!(decision_count));

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
    if result.get("valid").and_then(Value::as_bool) == Some(true)
        && let Some(scores) = selection_scores(decisions, &expected)
    {
        result.insert("selection_scores".to_string(), scores);
    }

    Ok(())
}

// ============================================================================
// Boolean AR task builder & sum repair merge helpers (for Wave 4B orchestrator)
// ============================================================================

/// Specification for one boolean receipt-block inclusion AR task (`backend.py:1194-1222`).
#[derive(Debug, Clone)]
pub struct ReceiptBooleanArTaskSpec {
    pub occurrence_id: String,
    pub prompt: String,
    pub suffix: String,
    pub grammar: GrammarSpec,
    pub json_schema: Value,
    pub max_new_tokens: usize,
}

/// Prepared receipt-sum repair plan for one field (`backend.py:1175-1224`).
#[derive(Debug, Clone)]
pub struct PreparedReceiptRepair {
    pub field_name: String,
    pub packet: Value,
    pub tasks: Vec<ReceiptBooleanArTaskSpec>,
    pub batch_size: usize,
    pub max_new_tokens: usize,
}

/// Extract typed boolean AR task specs from a prepared receipt packet.
pub fn build_boolean_ar_tasks(packet: &Value) -> Vec<ReceiptBooleanArTaskSpec> {
    let Some(tasks) = packet.get("tasks").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(tasks.len());
    for task in tasks {
        let Some(occurrence_id) = task.get("occurrence_id").and_then(Value::as_str) else {
            continue;
        };
        let Some(prompt) = task.get("prompt").and_then(Value::as_str) else {
            continue;
        };
        let suffix = task
            .get("suffix")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let ebnf = task
            .get("grammar")
            .and_then(|g| g.get("ebnf"))
            .and_then(Value::as_str)
            .unwrap_or(hybrid_inventory::BOOLEAN_EBNF);
        let json_schema = task
            .get("json_schema")
            .cloned()
            .unwrap_or_else(|| json!({"type": ["boolean", "null"]}));
        out.push(ReceiptBooleanArTaskSpec {
            occurrence_id: occurrence_id.to_string(),
            prompt: prompt.to_string(),
            suffix,
            grammar: GrammarSpec::ClairicEbnf(ebnf.to_string()),
            json_schema,
            max_new_tokens: BOOLEAN_MAX_NEW_TOKENS,
        });
    }
    out
}

/// Convert a tokenized [`ReceiptBooleanArTaskSpec`] into an [`ArGenTask`] for [`crate::models::clairic::ar_engine::ClairicEngine`].
pub fn build_ar_gen_task(
    spec: &ReceiptBooleanArTaskSpec,
    input_ids: Vec<u32>,
    stop_token_ids: Vec<u32>,
) -> ArGenTask {
    ArGenTask {
        id: spec.occurrence_id.clone(),
        input_ids,
        media: None,
        media_key: None,
        grammar: spec.grammar.clone(),
        max_new_tokens: spec.max_new_tokens,
        stop_token_ids,
    }
}

/// Screen primary predictions and build prepared receipt repairs for all `_receipt_sum_fields`
/// (`backend.py::_encode_receipt_repairs`).
pub fn prepare_repairs(
    record: &Value,
    predictions: &Map<String, Value>,
    has_media: bool,
) -> Vec<PreparedReceiptRepair> {
    let Some(field_names) = record.get("_receipt_sum_fields").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut prepared = Vec::new();
    for entry in field_names {
        let Some(name) = entry.as_str() else {
            continue;
        };
        let Some(previous) = predictions.get(name).and_then(Value::as_object) else {
            continue;
        };
        let value_ok = previous
            .get("value")
            .is_some_and(|v| !v.is_boolean() && v.as_f64().is_some_and(f64::is_finite));
        let decoder_ok = previous.get("decoder").and_then(Value::as_str) == Some(ar::VERSION);
        let p_null_ok = previous.get("p_null").is_some_and(|v| {
            !v.is_boolean()
                && v.as_f64()
                    .is_some_and(|p| p.is_finite() && (0.0..=1.0).contains(&p))
        });
        if !value_ok || !decoder_ok || !p_null_ok {
            continue;
        }
        let Some(packet) = prepare_task(record, name, has_media) else {
            continue;
        };
        let tasks = build_boolean_ar_tasks(&packet);
        if tasks.is_empty() {
            continue;
        }
        prepared.push(PreparedReceiptRepair {
            field_name: name.to_string(),
            packet,
            tasks,
            batch_size: BOOLEAN_BATCH_SIZE,
            max_new_tokens: BOOLEAN_MAX_NEW_TOKENS,
        });
    }
    prepared
}

/// Normalize one AR decision answer into the map entry format expected by [`validate_result`]
/// (`{**answer, "raw_json": answer["raw"]}` in `backend.py:1263`).
pub fn build_decision_entry(answer: &Value) -> Value {
    let mut map = answer.as_object().cloned().unwrap_or_default();
    if !map.contains_key("raw_json")
        && let Some(raw) = map.get("raw").cloned()
    {
        map.insert("raw_json".to_string(), raw);
    }
    Value::Object(map)
}

/// Validate boolean receipt decisions and build the merged prediction override if the
/// repaired sum is valid, non-null, scored, and differs from `primary_prediction["value"]`
/// (`backend.py:1267-1290`).
pub fn merge_receipt_repair(
    packet: &Value,
    decisions: &Map<String, Value>,
    primary_prediction: &Value,
) -> Option<Value> {
    let checked = validate_result(packet, decisions);
    if checked.get("valid").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    let new_value = checked.get("value")?;
    if new_value.is_null() {
        return None;
    }
    let selection_scores = checked.get("selection_scores")?.as_object()?;
    if selection_scores.is_empty() {
        return None;
    }
    if let Some(prev_val) = primary_prediction.get("value")
        && json_validate::json_equal(new_value, prev_val)
    {
        return None;
    }

    let operands: Vec<Value> = checked
        .get("operands")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|item| {
                    let mut m = Map::new();
                    m.insert(
                        "span".to_string(),
                        item.get("span").cloned().unwrap_or(Value::Null),
                    );
                    m.insert(
                        "printed".to_string(),
                        item.get("printed").cloned().unwrap_or(Value::Null),
                    );
                    Value::Object(m)
                })
                .collect()
        })
        .unwrap_or_default();

    let total_tokens: u64 = decisions
        .values()
        .filter_map(|answer| answer.get("tokens").and_then(Value::as_u64))
        .sum();

    let mut update = Map::new();
    update.insert("value".to_string(), new_value.clone());
    update.insert("valid".to_string(), Value::Bool(true));
    update.insert("complete".to_string(), Value::Bool(true));
    update.insert(
        "decoder".to_string(),
        Value::String(RECEIPT_BLOCK_SUM_DECODER.to_string()),
    );
    update.insert("span".to_string(), Value::Null);
    update.insert(
        "raw".to_string(),
        checked.get("decimal_sum").cloned().unwrap_or(Value::Null),
    );
    for (k, v) in selection_scores {
        update.insert(k.clone(), v.clone());
    }
    update.insert(
        "p_null".to_string(),
        primary_prediction
            .get("p_null")
            .cloned()
            .unwrap_or(Value::Null),
    );
    update.insert(
        "p_null_method".to_string(),
        json!("primary_numeric_ar_first_token_null_probability"),
    );
    update.insert(
        "source_receipt_operands".to_string(),
        Value::Array(operands),
    );
    update.insert("tokens".to_string(), json!(total_tokens));

    Some(Value::Object(update))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_receipt_record() -> Value {
        json!({
            "input": "Expense report 2025-05-01\n\nReceipt 101 - Cafe\nFood: $12.00\nTax: $1.20\nTip: $2.00\nTotal: $15.20\n\nReceipt 102 - Taxi\n05/01 10:15\nCard ****4242\nFare: $24.80\nTotal: $24.80\n\nReceipt 103 - Refund\nMay 2\nAmount charged: -$5.00",
            "fields": {
                "receipt_total": {
                    "type": "number",
                    "decimals": 2,
                    "instructions": "The sum of all receipt totals."
                }
            }
        })
    }

    fn make_boolean_decision(val: bool, log_prob: f64, tokens: u64) -> Value {
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
    fn eligibility_rejects_compound_and_round_each() {
        let ok = sample_receipt_record();
        assert!(eligible(&ok, "receipt_total", false));

        let compound = json!({
            "input": "Total: $10.00",
            "fields": {
                "f": {
                    "type": "number",
                    "instructions": "Sum of receipt totals minus tax."
                },
                "f2": {
                    "type": "number",
                    "instructions": "Sum of receipt totals + 5."
                },
                "f3": {
                    "type": "number",
                    "instructions": "Sum of receipt totals, rounding each receipt to the nearest dollar."
                }
            }
        });
        assert!(!eligible(&compound, "f", false));
        assert!(!eligible(&compound, "f2", false));
        assert!(!eligible(&compound, "f3", false));
    }

    #[test]
    fn parses_structured_receipt_blocks_and_builds_boolean_tasks() {
        let record = sample_receipt_record();
        let packet = prepare_task(&record, "receipt_total", false).unwrap();
        assert_eq!(
            packet["kind"],
            json!("source-receipt-block-boolean-sum-diagnostic-v1")
        );
        let blocks = packet["blocks"].as_array().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["total_occurrence"]["printed"], json!("$15.20"));
        assert_eq!(blocks[1]["total_occurrence"]["printed"], json!("$24.80"));
        assert_eq!(blocks[2]["total_occurrence"]["printed"], json!("-$5.00"));

        let ar_tasks = build_boolean_ar_tasks(&packet);
        assert_eq!(ar_tasks.len(), 3);
        assert_eq!(ar_tasks[0].max_new_tokens, BOOLEAN_MAX_NEW_TOKENS);
        assert!(matches!(
            &ar_tasks[0].grammar,
            GrammarSpec::ClairicEbnf(ebnf) if ebnf == hybrid_inventory::BOOLEAN_EBNF
        ));

        let gen_task = build_ar_gen_task(&ar_tasks[0], vec![1, 2, 3], vec![99]);
        assert_eq!(gen_task.id, ar_tasks[0].occurrence_id);
        assert_eq!(gen_task.max_new_tokens, 8);
    }

    #[test]
    fn supports_sum_positive_negative_and_non_match() {
        let record = sample_receipt_record();
        let packet = prepare_task(&record, "receipt_total", false).unwrap();
        // 15.20 + 24.80 - 5.00 = 35.00
        assert_eq!(supports_sum(&packet, &json!(35.0)), Some(true));
        // 15.20 + 24.80 = 40.00
        assert_eq!(supports_sum(&packet, &json!(40)), Some(true));
        // Negative single subset: -5.00
        assert_eq!(supports_sum(&packet, &json!(-5.0)), Some(true));
        // 15.20 - 5.00 = 10.20
        assert_eq!(supports_sum(&packet, &json!(10.2)), Some(true));
        // Non-match: 12.00 (food component, not a receipt total)
        assert_eq!(supports_sum(&packet, &json!(12.0)), Some(false));
        // Empty subset (0) is not supported since no non-empty subset sums to 0
        assert_eq!(supports_sum(&packet, &json!(0)), Some(false));
        // Reject booleans and strings
        assert_eq!(supports_sum(&packet, &json!(true)), None);
        assert_eq!(supports_sum(&packet, &json!("35.0")), None);
    }

    #[test]
    fn supports_sum_meet_in_the_middle_20_items() {
        let mut lines = Vec::new();
        for i in 0..20u64 {
            let amt = 1u64 << i;
            lines.push(format!("Receipt {}\nTotal: ${amt}", i + 1));
        }
        let record = json!({
            "input": lines.join("\n\n"),
            "fields": {
                "receipt_total": {
                    "type": "number",
                    "instructions": "Sum of all receipt totals."
                }
            }
        });
        let packet = prepare_task(&record, "receipt_total", false).unwrap();
        assert_eq!(packet["blocks"].as_array().unwrap().len(), 20);
        // Any 20-bit non-zero mask is a unique subset sum!
        let target: u64 = (1 << 0) + (1 << 7) + (1 << 13) + (1 << 19);
        assert_eq!(supports_sum(&packet, &json!(target)), Some(true));
        // Sum exceeding all 20 items is rejected
        assert_eq!(supports_sum(&packet, &json!((1u64 << 20) + 5)), Some(false));
    }

    #[test]
    fn supports_sum_high_precision_over_28_digits() {
        let a = "100000000000000000000000000000000001"; // 36 digits
        let b = "200000000000000000000000000000000002"; // 36 digits
        let sum_ab = "300000000000000000000000000000000003";
        let non_match = "300000000000000000000000000000000004";
        let record = json!({
            "input": format!("Receipt 1\nTotal: ${a}\n\nReceipt 2\nTotal: ${b}"),
            "fields": {
                "receipt_total": {
                    "type": "number",
                    "instructions": "Sum of receipt totals."
                }
            }
        });
        let packet = prepare_task(&record, "receipt_total", false).unwrap();
        let target_ok = decimal_util::parse_decimal(sum_ab).unwrap();
        let target_bad = decimal_util::parse_decimal(non_match).unwrap();
        assert_eq!(supports_sum_decimal(&packet, &target_ok), Some(true));
        assert_eq!(supports_sum_decimal(&packet, &target_bad), Some(false));
    }

    #[test]
    fn validate_and_merge_receipt_repair() {
        let record = sample_receipt_record();
        let packet = prepare_task(&record, "receipt_total", false).unwrap();
        let tasks = packet["tasks"].as_array().unwrap();

        let mut decisions = Map::new();
        decisions.insert(
            tasks[0]["occurrence_id"].as_str().unwrap().to_string(),
            make_boolean_decision(true, -0.05, 1),
        );
        decisions.insert(
            tasks[1]["occurrence_id"].as_str().unwrap().to_string(),
            make_boolean_decision(true, -0.10, 1),
        );
        decisions.insert(
            tasks[2]["occurrence_id"].as_str().unwrap().to_string(),
            make_boolean_decision(true, -0.02, 1),
        );

        let checked = validate_result(&packet, &decisions);
        assert_eq!(checked["valid"], json!(true));
        assert_eq!(checked["value"], json!(35.0));
        assert_eq!(checked["operand_count"], json!(3));
        assert_eq!(checked["decision_count"], json!(3));
        assert_eq!(
            checked["selection_scores"]["confidence_method"],
            json!("receipt_block_boolean_joint_sequence_probability")
        );

        let primary_wrong = json!({
            "value": 40.0,
            "decoder": ar::VERSION,
            "p_null": 0.01
        });
        let merged = merge_receipt_repair(&packet, &decisions, &primary_wrong).unwrap();
        assert_eq!(merged["value"], json!(35.0));
        assert_eq!(merged["decoder"], json!(RECEIPT_BLOCK_SUM_DECODER));
        assert_eq!(merged["p_null"], json!(0.01));
        assert_eq!(merged["tokens"], json!(3));
        assert_eq!(
            merged["source_receipt_operands"].as_array().unwrap().len(),
            3
        );

        // If primary already equals the repaired sum, merge returns None (no-op)
        let primary_same = json!({
            "value": 35,
            "decoder": ar::VERSION,
            "p_null": 0.01
        });
        assert!(merge_receipt_repair(&packet, &decisions, &primary_same).is_none());
    }

    #[test]
    fn currency_conflict_blocks_packet_preparation() {
        let record = json!({
            "input": "Receipt 1\nTotal: $10.00\n\nReceipt 2\nTotal: €20.00",
            "fields": {
                "receipt_total": {
                    "type": "number",
                    "instructions": "Sum of all receipt totals."
                }
            }
        });
        assert!(prepare_task(&record, "receipt_total", false).is_none());
    }

    #[test]
    fn eligible_fields_and_prepare_repairs_screen_primary_predictions() {
        let mut record = sample_receipt_record();
        let fields = record["fields"].as_object().unwrap().clone();
        let names = eligible_fields(record["input"].as_str().unwrap(), &fields, false);
        assert_eq!(names, vec!["receipt_total".to_string()]);
        assert!(eligible_fields(record["input"].as_str().unwrap(), &fields, true).is_empty());

        record["_receipt_sum_fields"] = json!(names);
        let mut preds = Map::new();
        preds.insert(
            "receipt_total".to_string(),
            json!({
                "value": 40.0,
                "decoder": ar::VERSION,
                "p_null": 0.02
            }),
        );
        let repairs = prepare_repairs(&record, &preds, false);
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].field_name, "receipt_total");
        assert_eq!(repairs[0].batch_size, BOOLEAN_BATCH_SIZE);
        assert_eq!(repairs[0].max_new_tokens, BOOLEAN_MAX_NEW_TOKENS);
        assert_eq!(repairs[0].tasks.len(), 3);

        // Build decision entry mirrors "raw" -> "raw_json"
        let decision = build_decision_entry(&json!({"raw": "true", "value": true}));
        assert_eq!(decision["raw_json"], json!("true"));
    }
}
