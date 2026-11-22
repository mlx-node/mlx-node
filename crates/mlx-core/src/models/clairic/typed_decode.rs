//! Typed-head (Path B) output compaction/expansion and decoding for text, text lists,
//! numbers (with base-1000 group rescoring & copy pointer), dates, times, datetimes,
//! number/date lists, estimates (lexical CDF projection + histogram blend + 99 quantiles),
//! forecasts, answer-slot spelling, and SequenceMatcher similarity.
//!
//! Direct port of upstream `clairic_runtime/model.py` (lines 1538-2708 and 2878-2999).

use super::{
    decimal_util,
    errors::invalid_request,
    head::HeadOutputs,
    schema,
    typed_prompt::{
        self, ABOVE, BELOW, COPY_CONFIDENT, COPY_MIN_CONFIDENCE, DATE_WIDTH, EDGE_NEGATIVE,
        EDGE_ZERO, EXPONENT_CLASSES, EncodedRecord, EstimateEdgeParts, GROUP_CLASSES, GROUP_DIGITS,
        GROUP_LOW, LEXICAL_DIGITS, LIST_KINDS, LIST_SLOTS, MASKED, MAX_SPAN, MIN_EXPONENT,
        NUMBER_DIGITS, NUMBER_GROUPS, NUMBER_WIDTH, TAG_BEGIN, TAG_INSIDE, TIME_WIDTH,
        YEAR_CLASSES, YEAR_DIGITS, YEAR_LOW, YEAR_OUTSIDE,
    },
};
use crate::array::MxArray;
use bigdecimal::{BigDecimal, Zero};
use napi::{Error, Result};
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

pub const SPELL_END: &str = "\n";
pub const SPELL_SEPARATOR: &str = " | ";
pub const SPELL_MIN_CONFIDENCE: f64 = 0.3;

const PLACEHOLDER_WORDS: &[&str] = &[
    "a",
    "above",
    "an",
    "app",
    "attach",
    "attached",
    "attachment",
    "below",
    "doc",
    "document",
    "documents",
    "file",
    "heic",
    "here",
    "image",
    "images",
    "is",
    "jpeg",
    "jpg",
    "mobile",
    "page",
    "pdf",
    "photo",
    "photos",
    "pic",
    "picture",
    "please",
    "png",
    "scan",
    "scanned",
    "screenshot",
    "see",
    "the",
    "this",
    "upload",
    "uploaded",
    "via",
];

/// Materialized CPU outputs from [`HeadOutputs`] or [`expand_output`] consumed by [`decode_record`].
#[derive(Debug, Clone, Default)]
pub struct DecodedOutputs {
    pub null: Vec<f32>,
    pub text_best: Option<Vec<usize>>,
    pub text_probability: Option<Vec<f32>>,
    pub tags_labels: Option<Vec<Vec<usize>>>,
    pub tags_log_probability: Option<Vec<f32>>,
    pub trim_labels: Option<Vec<[usize; 2]>>,
    pub number_copy_best: Option<Vec<usize>>,
    pub number_copy_probability: Option<Vec<f32>>,
    pub number_copy_gate: Option<Vec<f32>>,
    pub number: Option<Vec<Vec<f32>>>,
    pub number_groups: Option<Vec<Vec<f32>>>,
    pub date: Option<Vec<Vec<f32>>>,
    pub time: Option<Vec<Vec<f32>>>,
    pub estimate: Option<Vec<Vec<f32>>>,
    pub estimate_offset: Option<Vec<Vec<f32>>>,
    pub estimate_overflow: Option<Vec<[f32; 2]>>,
    pub estimate_number: Option<Vec<Vec<f32>>>,
    pub estimate_gate: Option<Vec<f32>>,
    pub number_list: Option<Vec<Vec<Vec<f32>>>>,
    pub number_list_end: Option<Vec<Vec<f32>>>,
    pub number_list_groups: Option<Vec<Vec<Vec<f32>>>>,
    pub date_list: Option<Vec<Vec<Vec<f32>>>>,
    pub date_list_end: Option<Vec<Vec<f32>>>,
    pub slot_ids: Option<Vec<Vec<u32>>>,
    pub slot_log_probs: Option<Vec<Vec<f32>>>,
    pub spell_ids: Option<Vec<Vec<u32>>>,
    pub spell_log_probs: Option<Vec<Vec<f32>>>,
}

impl DecodedOutputs {
    /// Evaluate and materialize [`HeadOutputs`] (plus optional answer-slot token IDs & log-probs)
    /// into CPU vectors ready for [`decode_record`].
    pub fn from_head_outputs(
        out: &HeadOutputs,
        slot_ids: Option<Vec<Vec<u32>>>,
        slot_log_probs: Option<Vec<Vec<f32>>>,
    ) -> Result<Self> {
        let mut refs: Vec<&MxArray> = vec![&out.null];
        for arr in [
            &out.trim,
            &out.text,
            &out.tags,
            &out.number,
            &out.number_groups,
            &out.number_copy,
            &out.number_copy_gate,
            &out.date,
            &out.time,
            &out.estimate,
            &out.estimate_offset,
            &out.estimate_overflow,
            &out.estimate_number,
            &out.estimate_gate,
            &out.number_list,
            &out.number_list_end,
            &out.number_list_groups,
            &out.date_list,
            &out.date_list_end,
        ]
        .into_iter()
        .flatten()
        {
            refs.push(arr);
        }
        MxArray::eval_arrays(&refs)?;

        let mut decoded = Self {
            null: out.null.to_float32()?.to_vec(),
            slot_ids,
            slot_log_probs,
            ..Self::default()
        };

        if let Some(ref text_arr) = out.text {
            let shape = text_arr.shape()?;
            let rows = shape[0] as usize;
            let row_len = (shape[1] * shape[2]) as usize;
            let flat = text_arr.to_float32()?;
            let mut best = Vec::with_capacity(rows);
            let mut probs = Vec::with_capacity(rows);
            for r in 0..rows {
                let slice = &flat[r * row_len..(r + 1) * row_len];
                let (idx, prob) = argmax_and_softmax_max(slice);
                best.push(idx);
                probs.push(prob as f32);
            }
            decoded.text_best = Some(best);
            decoded.text_probability = Some(probs);
        }

        if let Some(ref tags_arr) = out.tags {
            let shape = tags_arr.shape()?;
            let rows = shape[0] as usize;
            let tokens = shape[1] as usize;
            let classes = shape[2] as usize;
            let flat = tags_arr.to_float32()?;
            let mut all_labels = Vec::with_capacity(rows);
            let mut all_log_probs = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut row_labels = Vec::with_capacity(tokens);
                let mut sum_lp = 0.0f64;
                for t in 0..tokens {
                    let base = (r * tokens + t) * classes;
                    let lp = log_softmax_f64(&flat[base..base + classes]);
                    let (best_c, best_lp) = argmax_f64(&lp);
                    row_labels.push(best_c);
                    sum_lp += best_lp;
                }
                all_labels.push(row_labels);
                all_log_probs.push(sum_lp as f32);
            }
            decoded.tags_labels = Some(all_labels);
            decoded.tags_log_probability = Some(all_log_probs);
        }

        if let Some(ref trim_arr) = out.trim {
            let shape = trim_arr.shape()?;
            let tokens = shape[0] as usize;
            let sides = shape[1] as usize;
            let classes = shape[2] as usize;
            let flat = trim_arr.to_float32()?;
            let mut trims = Vec::with_capacity(tokens);
            for t in 0..tokens {
                let base0 = t * sides * classes;
                let base1 = base0 + classes;
                let left = argmax_f32(&flat[base0..base0 + classes]);
                let right = argmax_f32(&flat[base1..base1 + classes]);
                trims.push([left, right]);
            }
            decoded.trim_labels = Some(trims);
        }

        if let Some(ref copy_arr) = out.number_copy {
            let shape = copy_arr.shape()?;
            let rows = shape[0] as usize;
            let row_len = (shape[1] * shape[2]) as usize;
            let flat = copy_arr.to_float32()?;
            let mut best = Vec::with_capacity(rows);
            let mut probs = Vec::with_capacity(rows);
            for r in 0..rows {
                let slice = &flat[r * row_len..(r + 1) * row_len];
                let (idx, prob) = argmax_and_softmax_max(slice);
                best.push(idx);
                probs.push(prob as f32);
            }
            decoded.number_copy_best = Some(best);
            decoded.number_copy_probability = Some(probs);
        }

        if let Some(ref arr) = out.number_copy_gate {
            decoded.number_copy_gate = Some(arr.to_float32()?.to_vec());
        }
        if let Some(ref arr) = out.number {
            decoded.number = Some(extract_2d(arr)?);
        }
        if let Some(ref arr) = out.number_groups {
            let shape = arr.shape()?;
            let rows = shape[0] as usize;
            let width = (shape[1] * shape[2]) as usize;
            decoded.number_groups = Some(split_chunks(&arr.to_float32()?, rows, width));
        }
        if let Some(ref arr) = out.date {
            decoded.date = Some(extract_2d(arr)?);
        }
        if let Some(ref arr) = out.time {
            decoded.time = Some(extract_2d(arr)?);
        }
        if let Some(ref arr) = out.estimate {
            decoded.estimate = Some(extract_2d(arr)?);
        }
        if let Some(ref arr) = out.estimate_offset {
            decoded.estimate_offset = Some(extract_2d(arr)?);
        }
        if let Some(ref arr) = out.estimate_overflow {
            let rows = extract_2d(arr)?;
            decoded.estimate_overflow = Some(
                rows.into_iter()
                    .map(|r| {
                        [
                            r.first().copied().unwrap_or(0.0),
                            r.get(1).copied().unwrap_or(0.0),
                        ]
                    })
                    .collect(),
            );
        }
        if let Some(ref arr) = out.estimate_number {
            decoded.estimate_number = Some(extract_2d(arr)?);
        }
        if let Some(ref arr) = out.estimate_gate {
            decoded.estimate_gate = Some(arr.to_float32()?.to_vec());
        }
        if let Some(ref arr) = out.number_list {
            decoded.number_list = Some(extract_3d(arr)?);
        }
        if let Some(ref arr) = out.number_list_end {
            decoded.number_list_end = Some(extract_2d(arr)?);
        }
        if let Some(ref arr) = out.number_list_groups {
            let shape = arr.shape()?;
            let rows = shape[0] as usize;
            let slots = shape[1] as usize;
            let width = (shape[2] * shape[3]) as usize;
            let flat = arr.to_float32()?;
            let mut out_3d = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut row_v = Vec::with_capacity(slots);
                for s in 0..slots {
                    let base = (r * slots + s) * width;
                    row_v.push(flat[base..base + width].to_vec());
                }
                out_3d.push(row_v);
            }
            decoded.number_list_groups = Some(out_3d);
        }
        if let Some(ref arr) = out.date_list {
            decoded.date_list = Some(extract_3d(arr)?);
        }
        if let Some(ref arr) = out.date_list_end {
            decoded.date_list_end = Some(extract_2d(arr)?);
        }

        Ok(decoded)
    }
}

/// Flatten [`DecodedOutputs`] into the compact float vector format (`model.py::compact_output`).
pub fn compact_output(output: &DecodedOutputs) -> Vec<f32> {
    let mut parts: Vec<f32> = Vec::new();
    parts.extend_from_slice(&output.null);
    if let (Some(best), Some(probs)) = (&output.text_best, &output.text_probability) {
        parts.extend(best.iter().map(|&v| v as f32));
        parts.extend_from_slice(probs);
    }
    if let (Some(labels), Some(lps)) = (&output.tags_labels, &output.tags_log_probability) {
        for row in labels {
            parts.extend(row.iter().map(|&v| v as f32));
        }
        parts.extend_from_slice(lps);
    }
    if let Some(trims) = &output.trim_labels {
        for pair in trims {
            parts.push(pair[0] as f32);
            parts.push(pair[1] as f32);
        }
    }
    if let (Some(best), Some(probs)) = (&output.number_copy_best, &output.number_copy_probability) {
        parts.extend(best.iter().map(|&v| v as f32));
        parts.extend_from_slice(probs);
    }
    let extend_2d = |parts: &mut Vec<f32>, opt: &Option<Vec<Vec<f32>>>| {
        if let Some(rows) = opt {
            for r in rows {
                parts.extend_from_slice(r);
            }
        }
    };
    let extend_3d = |parts: &mut Vec<f32>, opt: &Option<Vec<Vec<Vec<f32>>>>| {
        if let Some(rows) = opt {
            for r in rows {
                for s in r {
                    parts.extend_from_slice(s);
                }
            }
        }
    };
    // Exact COMPACT_KEYS order from model.py:2720-2736
    extend_2d(&mut parts, &output.number);
    extend_2d(&mut parts, &output.date);
    extend_2d(&mut parts, &output.time);
    extend_2d(&mut parts, &output.estimate);
    extend_2d(&mut parts, &output.estimate_offset);
    if let Some(ovf) = &output.estimate_overflow {
        for pair in ovf {
            parts.push(pair[0]);
            parts.push(pair[1]);
        }
    }
    extend_2d(&mut parts, &output.number_list_end);
    extend_3d(&mut parts, &output.number_list);
    extend_2d(&mut parts, &output.date_list_end);
    extend_3d(&mut parts, &output.date_list);
    extend_2d(&mut parts, &output.estimate_number);
    if let Some(gate) = &output.estimate_gate {
        parts.extend_from_slice(gate);
    }
    extend_2d(&mut parts, &output.number_groups);
    extend_3d(&mut parts, &output.number_list_groups);
    if let Some(gate) = &output.number_copy_gate {
        parts.extend_from_slice(gate);
    }
    if let (Some(sids), Some(slps)) = (&output.slot_ids, &output.slot_log_probs) {
        for row in sids {
            parts.extend(row.iter().map(|&id| id as f32));
        }
        for row in slps {
            parts.extend_from_slice(row);
        }
    }
    parts
}

/// Reconstruct [`DecodedOutputs`] from a compact float vector (`model.py::expand_output`).
#[allow(clippy::too_many_arguments)]
pub fn expand_output(
    flat: &[f32],
    record: &EncodedRecord,
    bins: usize,
    estimate_lexical: bool,
    span_trim: bool,
    number_groups: bool,
    number_pointer: bool,
) -> Result<DecodedOutputs> {
    let mut text_n = 0usize;
    let mut tags_n = 0usize;
    let mut num_n = 0usize;
    let mut date_n = 0usize;
    let mut time_n = 0usize;
    let mut est_n = 0usize;
    let mut nl_n = 0usize;
    let mut dl_n = 0usize;
    for f in &record.fields {
        match f.kind.as_str() {
            "text" => text_n += 1,
            "text_list" => tags_n += 1,
            "number" => num_n += 1,
            "date" => date_n += 1,
            "time" => time_n += 1,
            "datetime" => {
                date_n += 1;
                time_n += 1;
            }
            "estimate" => est_n += 1,
            "number_list" => nl_n += 1,
            "date_list" => dl_n += 1,
            _ => {}
        }
    }

    let mut cursor = 0usize;
    let take = |cursor: &mut usize, size: usize| -> Result<&[f32]> {
        if *cursor + size > flat.len() {
            return Err(invalid_request(
                "compact output is shorter than the layout needs",
            ));
        }
        let slice = &flat[*cursor..*cursor + size];
        *cursor += size;
        Ok(slice)
    };

    let mut out = DecodedOutputs {
        null: take(&mut cursor, record.fields.len())?.to_vec(),
        ..DecodedOutputs::default()
    };

    if text_n > 0 {
        out.text_best = Some(
            take(&mut cursor, text_n)?
                .iter()
                .map(|&v| v.round() as usize)
                .collect(),
        );
        out.text_probability = Some(take(&mut cursor, text_n)?.to_vec());
    }
    if tags_n > 0 {
        let tok_len = record.offsets.len();
        let raw = take(&mut cursor, tags_n * tok_len)?;
        out.tags_labels = Some(
            (0..tags_n)
                .map(|r| {
                    raw[r * tok_len..(r + 1) * tok_len]
                        .iter()
                        .map(|&v| v.round() as usize)
                        .collect()
                })
                .collect(),
        );
        out.tags_log_probability = Some(take(&mut cursor, tags_n)?.to_vec());
    }
    if span_trim && (text_n > 0 || tags_n > 0) {
        let tok_len = record.offsets.len();
        let raw = take(&mut cursor, tok_len * 2)?;
        out.trim_labels = Some(
            (0..tok_len)
                .map(|t| [raw[t * 2].round() as usize, raw[t * 2 + 1].round() as usize])
                .collect(),
        );
    }
    if number_pointer && num_n > 0 {
        out.number_copy_best = Some(
            take(&mut cursor, num_n)?
                .iter()
                .map(|&v| v.round() as usize)
                .collect(),
        );
        out.number_copy_probability = Some(take(&mut cursor, num_n)?.to_vec());
    }

    if num_n > 0 {
        out.number = Some(split_chunks(
            take(&mut cursor, num_n * NUMBER_WIDTH)?,
            num_n,
            NUMBER_WIDTH,
        ));
    }
    if date_n > 0 {
        out.date = Some(split_chunks(
            take(&mut cursor, date_n * DATE_WIDTH)?,
            date_n,
            DATE_WIDTH,
        ));
    }
    if time_n > 0 {
        out.time = Some(split_chunks(
            take(&mut cursor, time_n * TIME_WIDTH)?,
            time_n,
            TIME_WIDTH,
        ));
    }
    if est_n > 0 {
        out.estimate = Some(split_chunks(take(&mut cursor, est_n * bins)?, est_n, bins));
        out.estimate_offset = Some(split_chunks(take(&mut cursor, est_n * bins)?, est_n, bins));
        let ovf = take(&mut cursor, est_n * 2)?;
        out.estimate_overflow = Some((0..est_n).map(|r| [ovf[r * 2], ovf[r * 2 + 1]]).collect());
    }
    if nl_n > 0 {
        out.number_list_end = Some(split_chunks(
            take(&mut cursor, nl_n * LIST_SLOTS)?,
            nl_n,
            LIST_SLOTS,
        ));
        let raw_nl = take(&mut cursor, nl_n * LIST_SLOTS * NUMBER_WIDTH)?;
        out.number_list = Some(split_3d(raw_nl, nl_n, LIST_SLOTS, NUMBER_WIDTH));
    }
    if dl_n > 0 {
        out.date_list_end = Some(split_chunks(
            take(&mut cursor, dl_n * LIST_SLOTS)?,
            dl_n,
            LIST_SLOTS,
        ));
        let raw_dl = take(&mut cursor, dl_n * LIST_SLOTS * DATE_WIDTH)?;
        out.date_list = Some(split_3d(raw_dl, dl_n, LIST_SLOTS, DATE_WIDTH));
    }
    if estimate_lexical && est_n > 0 {
        out.estimate_number = Some(split_chunks(
            take(&mut cursor, est_n * NUMBER_WIDTH)?,
            est_n,
            NUMBER_WIDTH,
        ));
        out.estimate_gate = Some(take(&mut cursor, est_n)?.to_vec());
    }
    if number_groups && num_n > 0 {
        let gw = NUMBER_GROUPS * GROUP_CLASSES;
        out.number_groups = Some(split_chunks(take(&mut cursor, num_n * gw)?, num_n, gw));
    }
    if number_groups && nl_n > 0 {
        let gw = NUMBER_GROUPS * GROUP_CLASSES;
        let raw = take(&mut cursor, nl_n * LIST_SLOTS * gw)?;
        out.number_list_groups = Some(split_3d(raw, nl_n, LIST_SLOTS, gw));
    }
    if number_pointer && num_n > 0 {
        out.number_copy_gate = Some(take(&mut cursor, num_n)?.to_vec());
    }

    let slotted: Vec<usize> = record
        .fields
        .iter()
        .filter(|f| !f.answer_slots.is_empty())
        .map(|f| f.answer_slots.len())
        .collect();
    if !slotted.is_empty() {
        let max_len = slotted.iter().copied().max().unwrap_or(0);
        let raw_ids = take(&mut cursor, slotted.len() * max_len)?;
        let raw_lps = take(&mut cursor, slotted.len() * max_len)?;
        out.slot_ids = Some(
            (0..slotted.len())
                .map(|r| {
                    raw_ids[r * max_len..(r + 1) * max_len]
                        .iter()
                        .map(|&v| v.round() as u32)
                        .collect()
                })
                .collect(),
        );
        out.slot_log_probs = Some(split_chunks(raw_lps, slotted.len(), max_len));
    }

    if cursor != flat.len() {
        return Err(invalid_request(
            "compact output is longer than the layout needs",
        ));
    }
    Ok(out)
}

// ============================================================================
// SequenceMatcher ratio & spelled-answer routing
// ============================================================================

/// Exact port of Python `difflib.SequenceMatcher(None, left, right).ratio()`.
pub fn sequence_matcher_ratio(left: &str, right: &str) -> f64 {
    let a: Vec<char> = left.chars().collect();
    let b: Vec<char> = right.chars().collect();
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (idx, &ch) in b.iter().enumerate() {
        b2j.entry(ch).or_default().push(idx);
    }
    if b.len() >= 200 {
        let threshold = b.len() / 100 + 1;
        b2j.retain(|_, indices| indices.len() <= threshold);
    }

    let find_longest = |alo: usize, ahi: usize, blo: usize, bhi: usize| -> (usize, usize, usize) {
        let (mut best_i, mut best_j, mut best_size) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for (i, &ch) in a.iter().enumerate().take(ahi).skip(alo) {
            let mut new_j2len: HashMap<usize, usize> = HashMap::new();
            if let Some(positions) = b2j.get(&ch) {
                for &j in positions {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = j2len.get(&j.wrapping_sub(1)).copied().unwrap_or(0) + 1;
                    new_j2len.insert(j, k);
                    if k > best_size {
                        best_i = i + 1 - k;
                        best_j = j + 1 - k;
                        best_size = k;
                    }
                }
            }
            j2len = new_j2len;
        }
        (best_i, best_j, best_size)
    };

    let mut queue = vec![(0usize, a.len(), 0usize, b.len())];
    let mut matches = 0usize;
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = find_longest(alo, ahi, blo, bhi);
        if k > 0 {
            matches += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    2.0 * (matches as f64) / (total as f64)
}

fn simple_alnum(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// Alphanumeric-normalized [`sequence_matcher_ratio`] (`model.py::_similarity`).
pub fn similarity(left: Option<&str>, right: Option<&str>) -> f64 {
    let (Some(l), Some(r)) = (left, right) else {
        return 0.0;
    };
    let (sl, sr) = (simple_alnum(l), simple_alnum(r));
    if sl.is_empty() || sr.is_empty() {
        return 0.0;
    }
    sequence_matcher_ratio(&sl, &sr)
}

fn slice_chars(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn state_match(
    text: &str,
    value: &str,
    threshold: f64,
    max_words: usize,
) -> Option<(usize, usize)> {
    if text.is_empty() || value.is_empty() {
        return None;
    }
    let spans = schema::find_spans(text, value, 16);
    if let Some(&first) = spans.first() {
        return Some(first);
    }
    static WORD_RE: OnceLock<Regex> = OnceLock::new();
    let re = WORD_RE.get_or_init(|| Regex::new(r"\S+").unwrap_or_else(|_| unreachable!()));
    let b2c = schema::byte_to_char_map(text);
    let words: Vec<(usize, usize)> = re
        .find_iter(text)
        .take(max_words)
        .map(|m| (b2c[m.start()], b2c[m.end()]))
        .collect();
    let size = value.split_whitespace().count();
    let mut best = threshold;
    let mut best_span = None;
    let min_w = size.saturating_sub(1).max(1);
    for width in min_w..=(size + 1) {
        if width > words.len() {
            continue;
        }
        for first in 0..=(words.len() - width) {
            let start = words[first].0;
            let end = words[first + width - 1].1;
            let candidate = slice_chars(text, start, end);
            let stripped =
                candidate.trim_matches(&['.', ',', ';', ':', '(', ')', '[', ']', '"', '\''][..]);
            let score = similarity(Some(stripped), Some(value));
            if score >= best {
                best = score;
                best_span = Some((start, end));
            }
        }
    }
    best_span
}

fn stutters(value: &str) -> bool {
    let simple = simple_alnum(value);
    let chars: Vec<char> = simple.chars().collect();
    for w in chars.windows(3) {
        if w[0] == w[1] && w[1] == w[2] {
            return true;
        }
    }
    // Check `([a-z]{4,})\1`
    let n = chars.len();
    for start in 0..n {
        if !chars[start].is_ascii_lowercase() {
            continue;
        }
        let max_len = (n - start) / 2;
        for len in 4..=max_len {
            let seg = &chars[start..start + len];
            if seg.iter().all(|c| c.is_ascii_lowercase())
                && seg == &chars[start + len..start + 2 * len]
            {
                return true;
            }
        }
    }
    false
}

fn digit_code(value: &str) -> bool {
    let simple = simple_alnum(value);
    if simple.is_empty() {
        return false;
    }
    let digits = simple.chars().filter(|c| c.is_numeric()).count();
    2 * digits >= simple.chars().count()
}

fn placeholder_text(value: Option<&str>) -> bool {
    let Some(v) = value else {
        return false;
    };
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"[a-z0-9]+").unwrap_or_else(|_| unreachable!()));
    let lower = v.to_lowercase();
    let mut saw_any = false;
    for m in re.find_iter(&lower) {
        saw_any = true;
        if !PLACEHOLDER_WORDS.contains(&m.as_str()) {
            return false;
        }
    }
    saw_any
}

fn defers(pointer_value: Option<&str>, spelled: &str) -> bool {
    let Some(pv) = pointer_value else {
        return false;
    };
    stutters(spelled) || (digit_code(spelled) && pv.chars().any(|c| c.is_numeric()))
}

pub fn spelled_answer(
    ids: &[u32],
    log_probs: &[f32],
    tokenizer: &Tokenizer,
    end_id: u32,
) -> Result<(String, f64, Vec<(String, f64)>)> {
    let stop = ids.iter().position(|&id| id == end_id).unwrap_or(ids.len());
    let mut items: Vec<(Vec<u32>, Vec<f64>)> = Vec::new();
    let mut current: Vec<u32> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    for (&tok, &lp) in ids[..stop].iter().zip(log_probs[..stop].iter()) {
        let piece = tokenizer
            .decode(&[tok], false)
            .map_err(|e| Error::from_reason(format!("tokenizer decode failed: {e}")))?;
        if piece.trim() == SPELL_SEPARATOR.trim() {
            items.push((current, scores));
            current = Vec::new();
            scores = Vec::new();
        } else {
            current.push(tok);
            scores.push(f64::from(lp));
        }
    }
    items.push((current, scores));

    let mut parts = Vec::new();
    for (item_ids, item_scores) in items {
        let decoded = tokenizer
            .decode(&item_ids, false)
            .map_err(|e| Error::from_reason(format!("tokenizer decode failed: {e}")))?;
        let words: Vec<&str> = decoded.split_whitespace().collect();
        if !words.is_empty() && !item_scores.is_empty() {
            let mean_lp: f64 = item_scores.iter().sum::<f64>() / (item_scores.len() as f64);
            parts.push((words.join(" "), mean_lp.exp()));
        }
    }
    let text = parts
        .iter()
        .map(|(p, _)| p.as_str())
        .collect::<Vec<_>>()
        .join(SPELL_SEPARATOR);
    let take_lp = (stop + 1).min(ids.len()).min(log_probs.len());
    let total_prob = log_probs[..take_lp]
        .iter()
        .map(|&v| f64::from(v))
        .sum::<f64>()
        .exp();
    Ok((text, total_prob, parts))
}

#[allow(clippy::too_many_arguments)]
pub fn evidence_text_answer(
    prediction: &Map<String, Value>,
    ids: &[u32],
    log_probs: &[f32],
    tokenizer: &Tokenizer,
    end_id: u32,
    text: &str,
    spec: &Value,
    has_images: bool,
) -> Result<Map<String, Value>> {
    let Some(stop) = ids.iter().position(|&id| id == end_id) else {
        return Ok(prediction.clone());
    };
    let decoded = tokenizer
        .decode(&ids[..stop], false)
        .map_err(|e| Error::from_reason(format!("tokenizer decode failed: {e}")))?;
    let value = decoded.trim().to_string();
    let take_lp = (stop + 1).min(log_probs.len());
    let confidence = log_probs[..take_lp]
        .iter()
        .map(|&v| f64::from(v))
        .sum::<f64>()
        .exp();
    let exact: Vec<(usize, usize)> = if value.is_empty() {
        Vec::new()
    } else {
        schema::find_spans(text, &value, 16)
            .into_iter()
            .filter(|&(s, e)| slice_chars(text, s, e) == value)
            .collect()
    };
    let p_null = prediction
        .get("p_null")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let present = p_null < schema::null_threshold(spec);
    let mut out = prediction.clone();
    out.insert(
        "guess".to_string(),
        if value.is_empty() {
            Value::Null
        } else {
            Value::String(value.clone())
        },
    );
    out.insert(
        "value".to_string(),
        if !value.is_empty() && present {
            Value::String(value)
        } else {
            Value::Null
        },
    );
    out.insert(
        "span".to_string(),
        match exact.first() {
            Some(&(s, e)) => json!([s, e]),
            None => Value::Null,
        },
    );
    out.insert("confidence".to_string(), json!(confidence));
    out.insert(
        "source".to_string(),
        Value::String(
            if !exact.is_empty() {
                "input"
            } else if has_images {
                "image"
            } else {
                "readout"
            }
            .to_string(),
        ),
    );
    Ok(out)
}

pub fn route_spelled(
    prediction: &Map<String, Value>,
    kind: &str,
    spelled: &str,
    items: &[(String, f64)],
    text: &str,
    defer_lists: bool,
) -> Map<String, Value> {
    if kind == "text" {
        let mut pred = prediction.clone();
        if placeholder_text(pred.get("guess").and_then(Value::as_str)) {
            pred.insert("value".to_string(), Value::Null);
            pred.insert("guess".to_string(), Value::Null);
            pred.insert("span".to_string(), Value::Null);
            pred.insert("confidence".to_string(), json!(0.0));
        }
        let filtered: Vec<&(String, f64)> = items
            .iter()
            .filter(|(item, _)| !placeholder_text(Some(item.as_str())))
            .collect();
        if filtered.is_empty() {
            return pred;
        }
        let value = if filtered.len() == 1 {
            filtered[0].0.clone()
        } else {
            spelled.to_string()
        };
        let confidence = filtered
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::INFINITY, f64::min);
        let mut routed = pred.clone();
        routed.insert("spell_confidence".to_string(), json!(confidence));
        let guess_str = pred.get("guess").and_then(Value::as_str);
        if similarity(guess_str, Some(value.as_str())) >= 0.6 {
            if routed.get("value").is_none_or(Value::is_null)
                && guess_str.is_some_and(|g| !g.is_empty())
                && confidence >= SPELL_MIN_CONFIDENCE
            {
                routed.insert("value".to_string(), pred["guess"].clone());
            }
            return routed;
        }
        if let Some((s, e)) = state_match(text, &value, 0.8, 4000) {
            let matched = slice_chars(text, s, e);
            routed.insert("value".to_string(), Value::String(matched.clone()));
            routed.insert("guess".to_string(), Value::String(matched));
            routed.insert("span".to_string(), json!([s, e]));
            return routed;
        }
        let ptr_val = pred.get("value").and_then(Value::as_str);
        if confidence >= SPELL_MIN_CONFIDENCE && !defers(ptr_val, &value) {
            routed.insert("value".to_string(), Value::String(value.clone()));
            routed.insert("guess".to_string(), Value::String(value));
            routed.insert("span".to_string(), Value::Null);
            routed.insert("confidence".to_string(), json!(confidence));
            routed.insert("source".to_string(), Value::String("image".to_string()));
        }
        return routed;
    }

    // text_list
    let mut pointer_values: Vec<String> = prediction
        .get("value")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let mut pointer_spans: Vec<Option<Value>> = prediction
        .get("spans")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|v| if v.is_null() { None } else { Some(v.clone()) })
                .collect()
        })
        .unwrap_or_default();
    if pointer_values
        .iter()
        .any(|v| placeholder_text(Some(v.as_str())))
    {
        let mut kept_v = Vec::new();
        let mut kept_s = Vec::new();
        for (idx, val) in pointer_values.into_iter().enumerate() {
            if !placeholder_text(Some(val.as_str())) {
                kept_v.push(val);
                kept_s.push(pointer_spans.get(idx).cloned().flatten());
            }
        }
        pointer_values = kept_v;
        pointer_spans = kept_s;
    }
    let filtered: Vec<&(String, f64)> = items
        .iter()
        .filter(|(item, _)| !placeholder_text(Some(item.as_str())))
        .collect();
    let mut values: Vec<String> = Vec::new();
    let mut spans: Vec<Value> = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();

    for (item, confidence) in filtered {
        let best_match = pointer_values
            .iter()
            .enumerate()
            .map(|(i, pv)| (i, similarity(Some(pv.as_str()), Some(item.as_str()))))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((m_idx, m_score)) = best_match
            && !used.contains(&m_idx)
            && m_score >= 0.6
        {
            used.insert(m_idx);
            values.push(pointer_values[m_idx].clone());
            spans.push(
                pointer_spans
                    .get(m_idx)
                    .cloned()
                    .flatten()
                    .unwrap_or(Value::Null),
            );
            continue;
        }
        if let Some((s, e)) = state_match(text, item, 0.8, 4000) {
            let val = slice_chars(text, s, e);
            let sv = simple_alnum(&val);
            if !values.iter().any(|o| simple_alnum(o) == sv) {
                values.push(val);
                spans.push(json!([s, e]));
            }
        } else {
            let deferred = !pointer_values.is_empty()
                && (defer_lists || defers(Some(pointer_values[0].as_str()), item));
            let si = simple_alnum(item);
            if *confidence >= SPELL_MIN_CONFIDENCE
                && !deferred
                && !values.iter().any(|o| simple_alnum(o) == si)
            {
                values.push(item.clone());
                spans.push(Value::Null);
            }
        }
    }
    for (idx, val) in pointer_values.iter().enumerate() {
        let sv = simple_alnum(val);
        if !used.contains(&idx) && !values.iter().any(|o| simple_alnum(o) == sv) {
            values.push(val.clone());
            spans.push(
                pointer_spans
                    .get(idx)
                    .cloned()
                    .flatten()
                    .unwrap_or(Value::Null),
            );
        }
    }
    let mut routed = prediction.clone();
    let has_null_span = spans.iter().any(Value::is_null);
    routed.insert(
        "value".to_string(),
        Value::Array(values.into_iter().map(Value::String).collect()),
    );
    routed.insert("spans".to_string(), Value::Array(spans));
    if has_null_span {
        routed.insert("source".to_string(), Value::String("image".to_string()));
    }
    routed
}

// ============================================================================
// Number, Date, Time, Text Span, and Estimate Decoders
// ============================================================================

/// Decode a lexical number head vector (`[151]`) with optional base-1000 group rescoring
/// (`[10 * 1000]`), matching `model.py::decode_number`.
pub fn decode_number(logits: &[f32], spec: &Value, groups: Option<&[f32]>) -> Result<(Value, f64)> {
    if logits.len() != NUMBER_WIDTH {
        return Err(invalid_request(format!(
            "decode_number expects {NUMBER_WIDTH} logits, got {}",
            logits.len()
        )));
    }
    let sign_logits = &logits[0..2];
    let exp_logits = &logits[2..2 + EXPONENT_CLASSES];
    let digit_flat = &logits[2 + EXPONENT_CLASSES..NUMBER_WIDTH];

    let decimals = schema::allowed_decimals(spec);
    let exp_log = log_softmax_f64(exp_logits);
    let (exponent_class, exp_lp) = argmax_f64(&exp_log);
    let mut log_prob = exp_lp;
    if exponent_class == 0 {
        return Ok((json!(0), log_prob.exp()));
    }
    let power = (exponent_class as i64) - 1 + MIN_EXPONENT;
    let kept_i64 = match decimals {
        None => NUMBER_DIGITS as i64,
        Some(d) => power + (d as i64) + 1,
    };
    if kept_i64 < 0 {
        return Ok((json!(0), log_prob.exp()));
    }

    let mut digit_log = [[0.0f64; 10]; NUMBER_DIGITS];
    let mut digits = [0usize; NUMBER_DIGITS];
    for pos in 0..NUMBER_DIGITS {
        let lp = log_softmax_f64(&digit_flat[pos * 10..(pos + 1) * 10]);
        digit_log[pos].copy_from_slice(&lp[..10]);
        if pos == 0 {
            digit_log[0][0] = f64::from(MASKED);
        }
        let (best_d, _) = argmax_f64(&digit_log[pos]);
        digits[pos] = best_d;
    }

    let sign_log = log_softmax_f64(sign_logits);
    let (sign, sign_lp) = argmax_f64(&sign_log);
    let kept = (kept_i64 as usize).min(NUMBER_DIGITS);
    log_prob += sign_lp;

    match groups {
        None => {
            for pos in 0..kept {
                log_prob += digit_log[pos][digits[pos]];
            }
            if kept < NUMBER_DIGITS {
                let slice = if digits[kept] >= 5 {
                    &digit_log[kept][5..10]
                } else {
                    &digit_log[kept][0..5]
                };
                log_prob += logsumexp_f64(slice);
            }
        }
        Some(learned) => {
            if learned.len() != NUMBER_GROUPS * GROUP_CLASSES {
                return Err(invalid_request(format!(
                    "decode_number groups expects {} values, got {}",
                    NUMBER_GROUPS * GROUP_CLASSES,
                    learned.len()
                )));
            }
            let mut group_slots = [[NUMBER_DIGITS; GROUP_DIGITS]; NUMBER_GROUPS];
            let mut group_present = [false; NUMBER_GROUPS];
            let mut group_logs: Vec<Vec<f64>> = vec![Vec::new(); NUMBER_GROUPS];
            let mut counted = [false; NUMBER_GROUPS];

            for g in 0..NUMBER_GROUPS {
                let group_idx = GROUP_LOW + (g as i64);
                let mut any_in = false;
                for (place, slot_out) in group_slots[g].iter_mut().enumerate().take(GROUP_DIGITS) {
                    let place_exp = (GROUP_DIGITS - 1 - place) as i64;
                    let slot = power - (group_idx * (GROUP_DIGITS as i64) + place_exp);
                    if slot >= 0 && (slot as usize) < NUMBER_DIGITS {
                        *slot_out = slot as usize;
                        any_in = true;
                    }
                }
                group_present[g] = any_in;
                if !any_in {
                    continue;
                }
                let learned_base = g * GROUP_CLASSES;
                let mut scores = vec![0.0f64; GROUP_CLASSES];
                let s0 = group_slots[g][0];
                let s1 = group_slots[g][1];
                let s2 = group_slots[g][2];
                for c in 0..GROUP_CLASSES {
                    let d0 = c / 100;
                    let d1 = (c / 10) % 10;
                    let d2 = c % 10;
                    let t0 = if s0 < NUMBER_DIGITS {
                        digit_log[s0][d0]
                    } else if d0 == 0 {
                        0.0
                    } else {
                        f64::from(MASKED)
                    };
                    let t1 = if s1 < NUMBER_DIGITS {
                        digit_log[s1][d1]
                    } else if d1 == 0 {
                        0.0
                    } else {
                        f64::from(MASKED)
                    };
                    let t2 = if s2 < NUMBER_DIGITS {
                        digit_log[s2][d2]
                    } else if d2 == 0 {
                        0.0
                    } else {
                        f64::from(MASKED)
                    };
                    scores[c] = t0 + t1 + t2 + f64::from(learned[learned_base + c]);
                }
                let gl = log_softmax_slice_f64(&scores);
                let (best_c, best_lp) = argmax_f64(&gl);
                let c_digits = [best_c / 100, (best_c / 10) % 10, best_c % 10];
                for place in 0..GROUP_DIGITS {
                    let slot = group_slots[g][place];
                    if slot < NUMBER_DIGITS {
                        digits[slot] = c_digits[place];
                    }
                }
                if group_slots[g].iter().any(|&slot| slot < kept) {
                    counted[g] = true;
                    log_prob += best_lp;
                }
                group_logs[g] = gl;
            }

            if kept < NUMBER_DIGITS {
                let mut found_group = false;
                for g in 0..NUMBER_GROUPS {
                    if let Some(place) = group_slots[g].iter().position(|&s| s == kept) {
                        found_group = true;
                        if !counted[g] && !group_logs[g].is_empty() {
                            let ge5 = digits[kept] >= 5;
                            let mut subset = Vec::with_capacity(GROUP_CLASSES / 2);
                            for (c, &lp) in group_logs[g].iter().enumerate() {
                                let d = match place {
                                    0 => c / 100,
                                    1 => (c / 10) % 10,
                                    _ => c % 10,
                                };
                                if (d >= 5) == ge5 {
                                    subset.push(lp);
                                }
                            }
                            log_prob += logsumexp_f64(&subset);
                        }
                        break;
                    }
                }
                if !found_group {
                    let slice = if digits[kept] >= 5 {
                        &digit_log[kept][5..10]
                    } else {
                        &digit_log[kept][0..5]
                    };
                    log_prob += logsumexp_f64(slice);
                }
            }
        }
    }

    let digits_str: String = digits.iter().map(|&d| (b'0' + (d as u8)) as char).collect();
    let int_dec = decimal_util::parse_decimal(&digits_str)
        .ok_or_else(|| invalid_request("failed to parse decoded number digits"))?;
    let mut magnitude = decimal_util::scaleb(&int_dec, power - (NUMBER_DIGITS as i64) + 1);
    if let Some(d) = decimals {
        magnitude = decimal_util::round_half_up(&magnitude, d as i64);
    }
    if magnitude.is_zero() {
        return Ok((json!(0), log_prob.exp()));
    }
    let signed = if sign == 1 { -magnitude } else { magnitude };
    let val = decimal_util::to_decoded_json_number(&signed, decimals == Some(0))
        .ok_or_else(|| invalid_request("decoded number is non-finite"))?;
    Ok((val, log_prob.exp()))
}

fn number_from_numeral(
    magnitude: &BigDecimal,
    logits: &[f32],
    spec: &Value,
    check_exponent: bool,
) -> Option<Value> {
    if logits.len() != NUMBER_WIDTH {
        return None;
    }
    let sign_logits = &logits[0..2];
    let exp_logits = &logits[2..2 + EXPONENT_CLASSES];
    if check_exponent {
        let mag_f64 = magnitude.to_string().parse::<f64>().ok()?;
        let target = typed_prompt::number_components(mag_f64)?;
        if argmax_f32(exp_logits) != target.1 {
            return None;
        }
    }
    let decimals = schema::allowed_decimals(spec);
    let adjusted = match decimals {
        Some(d) => decimal_util::round_half_up(magnitude, d as i64),
        None => magnitude.clone(),
    };
    if adjusted.is_zero() {
        return Some(json!(0));
    }
    let signed = if argmax_f32(sign_logits) == 1 {
        -adjusted
    } else {
        adjusted
    };
    decimal_util::to_decoded_json_number(&signed, decimals == Some(0))
}

fn number_from_numerals(
    values: &[BigDecimal],
    logits: &[f32],
    spec: &Value,
    readout: Option<f64>,
    confident: bool,
) -> Option<Value> {
    let mut agreeing: Vec<Value> = values
        .iter()
        .filter_map(|mag| number_from_numeral(mag, logits, spec, true))
        .collect();
    if agreeing.is_empty() && confident && !values.is_empty() {
        let best = values.iter().min_by(|a, b| {
            let dist = |m: &BigDecimal| -> f64 {
                let Some(ro) = readout else {
                    return 0.0;
                };
                if ro == 0.0 || m.is_zero() {
                    return 0.0;
                }
                let mf = m.to_string().parse::<f64>().unwrap_or(0.0);
                if mf <= 0.0 {
                    return f64::INFINITY;
                }
                (mf.log10() - ro.abs().log10()).abs()
            };
            dist(a).total_cmp(&dist(b))
        });
        if let Some(chosen) = best
            && let Some(val) = number_from_numeral(chosen, logits, spec, false)
        {
            agreeing.push(val);
        }
    }
    if agreeing.is_empty() {
        return None;
    }
    let Some(ro) = readout else {
        return Some(agreeing.remove(0));
    };
    agreeing.into_iter().min_by(|a, b| {
        let da = (a.as_f64().unwrap_or(0.0) - ro).abs();
        let db = (b.as_f64().unwrap_or(0.0) - ro).abs();
        da.total_cmp(&db)
    })
}

/// Return `true` iff `readout` and `copied` have identical normalized significant digits
/// but differ by a power of 10 (`model.py::rescaled`).
pub fn rescaled(readout: f64, copied: f64) -> bool {
    let (Some(first), Some(second)) = (
        decimal_util::from_f64(readout.abs()),
        decimal_util::from_f64(copied.abs()),
    ) else {
        return false;
    };
    if first.is_zero() || second.is_zero() || first == second {
        return false;
    }
    decimal_util::significant_digits(&first) == decimal_util::significant_digits(&second)
}

/// Attempt to copy a printed numeral from `record.text` via the number copy pointer
/// (`model.py::copied_number`).
pub fn copied_number(
    output: &DecodedOutputs,
    row: usize,
    record: &EncodedRecord,
    spec: &Value,
    readout: Option<f64>,
) -> Option<(Value, f64, [usize; 2])> {
    let gate_vec = output.number_copy_gate.as_ref()?;
    let has_images = record
        .record
        .get("images")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty());
    if has_images || record.text.is_empty() {
        return None;
    }
    if std::env::var("SZ_DISABLE_NUMBER_COPY").as_deref() == Ok("1") {
        return None;
    }
    let gate = sigmoid_f64(f64::from(*gate_vec.get(row)?));
    if gate < 0.5 {
        return None;
    }
    let best = *output.number_copy_best.as_ref()?.get(row)?;
    let probability = f64::from(*output.number_copy_probability.as_ref()?.get(row)?);
    let start = best / MAX_SPAN;
    let length = best % MAX_SPAN;
    if start >= record.offsets.len() {
        return None;
    }
    let last = (start + length).min(record.offsets.len() - 1);
    let chars: Vec<char> = record.text.chars().collect();
    let mut char_start = record.offsets[start].0;
    let char_end = record.offsets[last].1;
    while char_start < char_end && char_start < chars.len() && !chars[char_start].is_alphanumeric()
    {
        char_start += 1;
    }

    let every_numerals = typed_prompt::read_numerals(&record.text, true);
    let around: Vec<(Vec<BigDecimal>, usize, usize)> = every_numerals
        .into_iter()
        .filter(|&(_, s, e)| s < char_end && e > char_start)
        .collect();
    if around.is_empty() {
        return None;
    }
    let nearest = around.into_iter().min_by(|a, b| {
        let da = a.1.abs_diff(char_start) + a.2.abs_diff(char_end);
        let db = b.1.abs_diff(char_start) + b.2.abs_diff(char_end);
        da.cmp(&db).then_with(|| (b.2 - b.1).cmp(&(a.2 - a.1)))
    })?;

    let whole: Vec<(Vec<BigDecimal>, usize, usize)> =
        typed_prompt::read_numerals(&record.text, false)
            .into_iter()
            .filter(|&(_, s, e)| {
                s <= char_start && e >= char_end && (e - s) > (nearest.2 - nearest.1)
            })
            .collect();
    let mut choices: Vec<(Vec<BigDecimal>, usize, usize)> = Vec::with_capacity(2);
    let whole_span = whole.first().map(|w| (w.1, w.2));
    if let Some(first_whole) = whole.into_iter().next() {
        choices.push(first_whole);
    }
    choices.push(nearest.clone());

    if gate * probability < COPY_MIN_CONFIDENCE {
        return None;
    }
    let sure = gate >= COPY_CONFIDENT;
    let number_logits = output.number.as_ref()?.get(row)?;
    let convention = typed_prompt::numeral_convention(&record.text);
    let b2c = schema::byte_to_char_map(&record.text);
    let c2b: Vec<usize> = record
        .text
        .char_indices()
        .map(|(bi, _)| bi)
        .chain(std::iter::once(record.text.len()))
        .collect();

    let mut chosen_value: Option<Value> = None;
    let (mut numeral_start, mut numeral_end) = (nearest.1, nearest.2);
    for (values, s, e) in &choices {
        let start_b = c2b[*s];
        let printed = match typed_prompt::numeral_run_regex()
            .find_from_pos(&record.text, start_b)
            .ok()
            .flatten()
        {
            Some(m) if m.start() == start_b => {
                let run_end_c = b2c[m.end()];
                slice_chars(&record.text, *s, (*e).min(run_end_c))
            }
            _ => String::new(),
        };
        let settled =
            convention.is_some() && typed_prompt::numeral_values(&printed, None).len() > 1;
        let enclosing = whole_span == Some((*s, *e));
        let val = number_from_numerals(
            values,
            number_logits,
            spec,
            readout,
            settled || (sure && enclosing && values.len() == 1),
        );
        if val.is_some() {
            chosen_value = val;
            numeral_start = *s;
            numeral_end = *e;
            break;
        }
    }
    if chosen_value.is_none() && sure {
        let (values, s, e) = &choices[0];
        numeral_start = *s;
        numeral_end = *e;
        chosen_value = number_from_numerals(values, number_logits, spec, readout, true);
    }
    let val = chosen_value?;
    if let (Some(ro), Some(cv)) = (readout, val.as_f64())
        && rescaled(ro, cv)
    {
        return None;
    }
    Some((val, gate * probability, [numeral_start, numeral_end]))
}

/// Decode a lexical date head vector (`[484]`) into `(YYYY-MM-DD, confidence)` (`model.py::decode_date`).
pub fn decode_date(logits: &[f32]) -> Result<(String, f64)> {
    if logits.len() != DATE_WIDTH {
        return Err(invalid_request(format!(
            "decode_date expects {DATE_WIDTH} logits, got {}",
            logits.len()
        )));
    }
    let digit_flat = &logits[0..YEAR_DIGITS * 10];
    let year_off = YEAR_DIGITS * 10;
    let year_logits = &logits[year_off..year_off + YEAR_CLASSES];
    let month_off = year_off + YEAR_CLASSES;
    let month_logits = &logits[month_off..month_off + 12];
    let day_logits = &logits[month_off + 12..DATE_WIDTH];

    let class_log = log_softmax_f64(year_logits);
    let (year_class, class_lp) = argmax_f64(&class_log);
    let mut log_prob = class_lp;
    let year: i32 = if year_class < YEAR_OUTSIDE {
        YEAR_LOW + (year_class as i32)
    } else {
        let mut dlogs = [[0.0f64; 10]; YEAR_DIGITS];
        let mut digits = [0usize; YEAR_DIGITS];
        for pos in 0..YEAR_DIGITS {
            let lp = log_softmax_f64(&digit_flat[pos * 10..(pos + 1) * 10]);
            dlogs[pos].copy_from_slice(&lp[..10]);
            digits[pos] = argmax_f64(&dlogs[pos]).0;
        }
        if digits == [0, 0, 0, 0] {
            let mut masked = dlogs[3];
            masked[0] = f64::from(MASKED);
            digits[3] = argmax_f64(&masked).0;
        }
        for pos in 0..YEAR_DIGITS {
            log_prob += dlogs[pos][digits[pos]];
        }
        (digits[0] * 1000 + digits[1] * 100 + digits[2] * 10 + digits[3]) as i32
    };

    let month_log = log_softmax_f64(month_logits);
    let day_log = log_softmax_f64(day_logits);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_m = 0usize;
    let mut best_d = 0usize;
    for (m, &ml) in month_log.iter().enumerate().take(12) {
        let max_days = schema::days_in_month(year, (m + 1) as u32) as usize;
        for (d, &dl) in day_log.iter().enumerate().take(31) {
            let score = if d >= max_days {
                f64::from(MASKED)
            } else {
                ml + dl
            };
            if score > best_score {
                best_score = score;
                best_m = m;
                best_d = d;
            }
        }
    }
    log_prob += best_score;
    Ok((
        schema::format_iso_date(year, (best_m + 1) as u32, (best_d + 1) as u32),
        log_prob.exp(),
    ))
}

/// Decode a lexical time head vector (`[144]`) into `(HH:MM:SS, confidence)` (`model.py::decode_time`).
pub fn decode_time(logits: &[f32]) -> Result<(String, f64)> {
    if logits.len() != TIME_WIDTH {
        return Err(invalid_request(format!(
            "decode_time expects {TIME_WIDTH} logits, got {}",
            logits.len()
        )));
    }
    let h_log = log_softmax_f64(&logits[0..24]);
    let m_log = log_softmax_f64(&logits[24..84]);
    let s_log = log_softmax_f64(&logits[84..144]);
    let (h, h_lp) = argmax_f64(&h_log);
    let (m, m_lp) = argmax_f64(&m_log);
    let (s, s_lp) = argmax_f64(&s_log);
    Ok((
        schema::format_time(h as u32, m as u32, s as u32),
        (h_lp + m_lp + s_lp).exp(),
    ))
}

/// Decode BIO token labels into deduplicated text strings and char offset spans (`model.py::decode_spans`).
pub fn decode_spans(
    labels: &[usize],
    offsets: &[(usize, usize)],
    text: &str,
    trims: Option<&[[usize; 2]]>,
) -> (Vec<String>, Vec<[usize; 2]>) {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    for (pos, &label) in labels.iter().enumerate() {
        if label == TAG_BEGIN {
            if let Some(c) = current {
                spans.push(c);
            }
            current = Some((pos, pos));
        } else if label == TAG_INSIDE {
            if let Some((_, ref mut last)) = current {
                *last = pos;
            }
        } else if let Some(c) = current.take() {
            spans.push(c);
        }
    }
    if let Some(c) = current {
        spans.push(c);
    }

    let mut values: Vec<String> = Vec::new();
    let mut char_spans: Vec<[usize; 2]> = Vec::new();
    for (first, last) in spans {
        if first >= offsets.len() || last >= offsets.len() {
            continue;
        }
        let (char_start, char_end) = typed_prompt::trimmed_span(text, offsets, first, last, trims);
        let piece = slice_chars(text, char_start, char_end);
        if piece.trim().is_empty() {
            continue;
        }
        char_spans.push([char_start, char_end]);
        if !values.contains(&piece) {
            values.push(piece);
        }
    }
    (values, char_spans)
}

/// Project lexical number logits onto histogram bin log-probabilities via exact FP64 CDF
/// (`model.py::lexical_bin_log_probs`).
pub fn lexical_bin_log_probs(number_logits: &[f32], parts: &EstimateEdgeParts) -> Vec<f32> {
    let sign_probs = softmax_f64(&number_logits[0..2]);
    let exp_probs = softmax_f64(&number_logits[2..2 + EXPONENT_CLASSES]);
    let zero = exp_probs[0];
    let denom = (1.0 - zero).max(1e-300);
    let nonzero_len = EXPONENT_CLASSES - 1;
    let mut nonzero = vec![0.0f64; nonzero_len];
    let mut below_class = vec![0.0f64; nonzero_len];
    let mut acc = 0.0f64;
    for i in 0..nonzero_len {
        let p = exp_probs[1 + i] / denom;
        nonzero[i] = p;
        below_class[i] = acc;
        acc += p;
    }

    let digit_flat = &number_logits[2 + EXPONENT_CLASSES..NUMBER_WIDTH];
    let mut digit_probs = [[0.0f64; 10]; LEXICAL_DIGITS];
    let mut below_digit = [[0.0f64; 10]; LEXICAL_DIGITS];
    for pos in 0..LEXICAL_DIGITS {
        let mut row = [0.0f64; 10];
        for d in 0..10 {
            row[d] = f64::from(digit_flat[pos * 10 + d]);
        }
        if pos == 0 {
            row[0] = f64::NEG_INFINITY;
        }
        let probs = softmax_slice_f64(&row);
        let mut d_acc = 0.0f64;
        for d in 0..10 {
            digit_probs[pos][d] = probs[d];
            below_digit[pos][d] = d_acc;
            d_acc += probs[d];
        }
    }

    let negative = sign_probs[1] * (1.0 - zero);
    let positive = sign_probs[0] * (1.0 - zero);
    let edges = parts.sign.len();
    let mut cdf = Vec::with_capacity(edges);
    for e in 0..edges {
        let class_idx = parts.class[e].saturating_sub(1).min(nonzero_len - 1);
        let mut within = parts.frac[e];
        for pos in (0..LEXICAL_DIGITS).rev() {
            let idx = parts.digits[e][pos] as usize;
            within = below_digit[pos][idx] + digit_probs[pos][idx] * within;
        }
        let magnitude = below_class[class_idx] + nonzero[class_idx] * within;
        let c = match parts.sign[e] {
            EDGE_NEGATIVE => negative * (1.0 - magnitude),
            EDGE_ZERO => negative,
            _ => negative + zero + positive * magnitude,
        };
        cdf.push(c);
    }

    let bins = edges.saturating_sub(1);
    let mut mass = Vec::with_capacity(bins);
    let mut mass_sum = 0.0f64;
    for b in 0..bins {
        let m = (cdf[b + 1] - cdf[b]).max(0.0) + 1e-30;
        mass.push(m);
        mass_sum += m;
    }
    let log_sum = mass_sum.ln();
    mass.into_iter()
        .map(|m| (m.ln() - log_sum) as f32)
        .collect()
}

/// Blend histogram bin logits with lexical CDF bin log-probs via `estimate_gate`
/// (`model.py::estimate_log_probs`).
pub fn estimate_log_probs(
    range_logits: &[f32],
    number_logits: Option<&[f32]>,
    gate: Option<f32>,
    parts: Option<&EstimateEdgeParts>,
) -> Vec<f32> {
    let range_part = log_softmax_f64(range_logits);
    let (Some(num_l), Some(g), Some(edge_parts)) = (number_logits, gate, parts) else {
        return range_part.into_iter().map(|v| v as f32).collect();
    };
    let lexical_part = lexical_bin_log_probs(num_l, edge_parts);
    let lg_pos = log_sigmoid_f64(f64::from(g));
    let lg_neg = log_sigmoid_f64(-f64::from(g));
    range_part
        .into_iter()
        .zip(lexical_part)
        .map(|(rp, lp)| logaddexp_f64(lg_pos + f64::from(lp), lg_neg + rp) as f32)
        .collect()
}

pub fn window_log_probs(log_probs: &[f32], window: (usize, usize)) -> Vec<f32> {
    let masked: Vec<f32> = log_probs
        .iter()
        .enumerate()
        .map(|(idx, &v)| {
            if idx >= window.0 && idx <= window.1 {
                v
            } else {
                MASKED
            }
        })
        .collect();
    log_softmax_f64(&masked)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

fn quantile_from_cdf(cdf: &[f64], probabilities: &[f64], fraction: f64, bins: usize) -> f64 {
    let index = cdf
        .partition_point(|&c| c < fraction)
        .min(bins.saturating_sub(1));
    let previous = if index > 0 { cdf[index - 1] } else { 0.0 };
    let inside = if probabilities[index] > 0.0 {
        (fraction - previous) / probabilities[index]
    } else {
        0.5
    };
    (index as f64 + inside.clamp(0.0, 1.0)) / (bins as f64)
}

/// Decode an estimate prediction (`value`, `mean`, `median`, `p10`, `p90`, 99 `quantiles`,
/// and `p_below`/`p_above` for numeric estimates), matching `model.py::decode_estimate`.
pub fn decode_estimate(
    logits: &[f32],
    offsets: &[f32],
    overflow: &[f32; 2],
    spec: &Value,
) -> Result<Value> {
    let probs = softmax_f64(logits);
    let bins = probs.len();
    let mut cdf = Vec::with_capacity(bins);
    let mut acc = 0.0f64;
    for &p in &probs {
        acc += p;
        cdf.push(acc);
    }
    let shift: Vec<f64> = offsets.iter().map(|&o| sigmoid_f64(f64::from(o))).collect();
    let (low, high, scale) = schema::estimate_range(spec)?;
    let centers: Vec<f64> = (0..bins)
        .map(|i| (i as f64 + shift[i]) / (bins as f64))
        .collect();
    let mean: f64 = probs
        .iter()
        .zip(centers.iter())
        .map(|(&p, &c)| p * schema::from_unit(c, low, high, &scale))
        .sum();
    let mut quantiles_f: Vec<f64> = (1..100)
        .map(|frac| {
            let pos = quantile_from_cdf(&cdf, &probs, (frac as f64) / 100.0, bins);
            schema::from_unit(pos, low, high, &scale)
        })
        .collect();
    let mut value_f = if scale == "log" {
        let unit_mean: f64 = probs.iter().zip(centers.iter()).map(|(&p, &c)| p * c).sum();
        schema::from_unit(unit_mean, low, high, &scale)
    } else {
        mean
    };
    let step = schema::estimate_step(spec);
    if step > 0.0 {
        for q in &mut quantiles_f {
            *q = schema::snap_quantile(*q, step, low, high);
        }
        value_f = schema::snap(mean, step, low, high);
    }

    let (value_json, mean_json, quantiles_json) = if schema::is_date_estimate(spec) {
        let vj = schema::estimate_value(spec, value_f)?;
        let mj = schema::estimate_value(spec, mean)?;
        let mut qj = Vec::with_capacity(99);
        for &q in &quantiles_f {
            qj.push(schema::estimate_value(spec, q)?);
        }
        (vj, mj, qj)
    } else {
        (
            json!(value_f),
            json!(mean),
            quantiles_f.into_iter().map(|q| json!(q)).collect(),
        )
    };

    let mut pred = Map::new();
    pred.insert("value".to_string(), value_json);
    pred.insert("mean".to_string(), mean_json);
    pred.insert("median".to_string(), quantiles_json[49].clone());
    pred.insert("p10".to_string(), quantiles_json[9].clone());
    pred.insert("p90".to_string(), quantiles_json[89].clone());
    pred.insert("quantiles".to_string(), Value::Array(quantiles_json));

    if !schema::is_date_estimate(spec) {
        let ovf_in = [0.0f32, overflow[0], overflow[1]];
        let sides = softmax_f64(&ovf_in);
        pred.insert("p_below".to_string(), json!(sides[BELOW]));
        pred.insert("p_above".to_string(), json!(sides[ABOVE]));
    }
    Ok(Value::Object(pred))
}

fn estimate_row_log_probs(
    output: &DecodedOutputs,
    row: usize,
    spec: &Value,
    window: Option<(usize, usize)>,
) -> Result<Vec<f32>> {
    let logits = output
        .estimate
        .as_ref()
        .and_then(|v| v.get(row))
        .ok_or_else(|| invalid_request("missing estimate logits"))?;
    let parts = if output.estimate_number.is_some() {
        typed_prompt::estimate_edge_parts(spec, logits.len())?
    } else {
        None
    };
    let lp = match parts {
        Some(ref edge_parts) => {
            let num_l = output
                .estimate_number
                .as_ref()
                .and_then(|v| v.get(row))
                .map(Vec::as_slice);
            let gate = output
                .estimate_gate
                .as_ref()
                .and_then(|v| v.get(row))
                .copied();
            estimate_log_probs(logits, num_l, gate, Some(edge_parts))
        }
        None => log_softmax_f64(logits)
            .into_iter()
            .map(|v| v as f32)
            .collect(),
    };
    Ok(match window {
        Some(w) => window_log_probs(&lp, w),
        None => lp,
    })
}

/// Decode all typed fields of an [`EncodedRecord`] from materialized [`DecodedOutputs`]
/// (`model.py::decode_record`).
pub fn decode_record(
    output: &DecodedOutputs,
    record: &EncodedRecord,
    tokenizer: Option<&Tokenizer>,
    end_id: Option<u32>,
) -> Result<Map<String, Value>> {
    let fields = &record.fields;
    let record_fields = record
        .record
        .get("fields")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_request("record missing fields"))?;
    let p_null: Vec<f64> = output
        .null
        .iter()
        .map(|&x| sigmoid_f64(f64::from(x)))
        .collect();

    let mut text_map: HashMap<usize, usize> = HashMap::new();
    let mut tags_map: HashMap<usize, usize> = HashMap::new();
    let mut num_map: HashMap<usize, usize> = HashMap::new();
    let mut date_map: HashMap<usize, usize> = HashMap::new();
    let mut time_map: HashMap<usize, usize> = HashMap::new();
    let mut est_map: HashMap<usize, usize> = HashMap::new();
    let mut nl_map: HashMap<usize, usize> = HashMap::new();
    let mut dl_map: HashMap<usize, usize> = HashMap::new();

    for (pos, item) in fields.iter().enumerate() {
        match item.kind.as_str() {
            "text" => {
                let n = text_map.len();
                text_map.insert(pos, n);
            }
            "text_list" => {
                let n = tags_map.len();
                tags_map.insert(pos, n);
            }
            "number" => {
                let n = num_map.len();
                num_map.insert(pos, n);
            }
            "date" => {
                let n = date_map.len();
                date_map.insert(pos, n);
            }
            "time" => {
                let n = time_map.len();
                time_map.insert(pos, n);
            }
            "datetime" => {
                let dn = date_map.len();
                date_map.insert(pos, dn);
                let tn = time_map.len();
                time_map.insert(pos, tn);
            }
            "estimate" => {
                let n = est_map.len();
                est_map.insert(pos, n);
            }
            "number_list" => {
                let n = nl_map.len();
                nl_map.insert(pos, n);
            }
            "date_list" => {
                let n = dl_map.len();
                dl_map.insert(pos, n);
            }
            _ => {}
        }
    }

    let trims = output.trim_labels.as_deref();
    let mut predictions: Map<String, Value> = Map::new();

    for (pos, item) in fields.iter().enumerate() {
        let spec = record_fields
            .get(&item.name)
            .ok_or_else(|| invalid_request(format!("missing spec for {:?}", item.name)))?;
        let kind = item.kind.as_str();

        if kind == "estimate" {
            let row = est_map[&pos];
            let lp = estimate_row_log_probs(output, row, spec, item.window)?;
            let offsets = output
                .estimate_offset
                .as_ref()
                .and_then(|v| v.get(row))
                .ok_or_else(|| invalid_request("missing estimate_offset"))?;
            let overflow = output
                .estimate_overflow
                .as_ref()
                .and_then(|v| v.get(row))
                .ok_or_else(|| invalid_request("missing estimate_overflow"))?;
            let pred = decode_estimate(&lp, offsets, overflow, spec)?;
            predictions.insert(item.name.clone(), pred);
            continue;
        }

        if kind == "text_list" {
            let row = tags_map[&pos];
            let labels = output
                .tags_labels
                .as_ref()
                .and_then(|v| v.get(row))
                .ok_or_else(|| invalid_request("missing tags_labels"))?;
            let (values, spans) = decode_spans(labels, &record.offsets, &record.text, trims);
            let mut pred = Map::new();
            pred.insert(
                "value".to_string(),
                Value::Array(values.into_iter().map(Value::String).collect()),
            );
            pred.insert(
                "spans".to_string(),
                Value::Array(spans.into_iter().map(|s| json!(s)).collect()),
            );
            if let Some(lp) = output
                .tags_log_probability
                .as_ref()
                .and_then(|v| v.get(row))
            {
                pred.insert("confidence".to_string(), json!(f64::from(*lp).exp()));
            }
            predictions.insert(item.name.clone(), Value::Object(pred));
            continue;
        }

        if LIST_KINDS.contains(&kind) {
            let row = if kind == "number_list" {
                nl_map[&pos]
            } else {
                dl_map[&pos]
            };
            let end_logits = if kind == "number_list" {
                output.number_list_end.as_ref().and_then(|v| v.get(row))
            } else {
                output.date_list_end.as_ref().and_then(|v| v.get(row))
            }
            .ok_or_else(|| invalid_request(format!("missing {kind}_end logits")))?;
            let ends: Vec<f64> = end_logits
                .iter()
                .map(|&e| sigmoid_f64(f64::from(e)))
                .collect();
            let slot_limit = ends.len().min(LIST_SLOTS);
            let mut values: Vec<Value> = Vec::new();
            let mut confidences: Vec<Value> = Vec::new();
            let mut log_prob = 0.0f64;

            for (slot, &end_p) in ends.iter().enumerate().take(slot_limit) {
                if end_p >= 0.5 {
                    log_prob += end_p.max(1e-12).ln();
                    break;
                }
                log_prob += (1.0 - end_p).max(1e-12).ln();
                let (val, conf) = if kind == "number_list" {
                    let slot_logits = output
                        .number_list
                        .as_ref()
                        .and_then(|v| v.get(row))
                        .and_then(|r| r.get(slot))
                        .ok_or_else(|| invalid_request("missing number_list slot logits"))?;
                    let slot_groups = output
                        .number_list_groups
                        .as_ref()
                        .and_then(|v| v.get(row))
                        .and_then(|r| r.get(slot))
                        .map(Vec::as_slice);
                    decode_number(slot_logits, spec, slot_groups)?
                } else {
                    let slot_logits = output
                        .date_list
                        .as_ref()
                        .and_then(|v| v.get(row))
                        .and_then(|r| r.get(slot))
                        .ok_or_else(|| invalid_request("missing date_list slot logits"))?;
                    let (d, c) = decode_date(slot_logits)?;
                    (Value::String(d), c)
                };
                values.push(val);
                confidences.push(json!(conf));
                log_prob += conf.max(1e-12).ln();
            }
            let mut pred = Map::new();
            pred.insert("value".to_string(), Value::Array(values));
            pred.insert("confidences".to_string(), Value::Array(confidences));
            pred.insert("confidence".to_string(), json!(log_prob.exp()));
            predictions.insert(item.name.clone(), Value::Object(pred));
            continue;
        }

        let mut pred = Map::new();
        let pn = *p_null
            .get(pos)
            .ok_or_else(|| invalid_request("missing null logit"))?;
        pred.insert("p_null".to_string(), json!(pn));

        let (value, confidence): (Value, f64) = match kind {
            "text" => {
                let row = text_map[&pos];
                let best = *output
                    .text_best
                    .as_ref()
                    .and_then(|v| v.get(row))
                    .ok_or_else(|| invalid_request("missing text_best"))?;
                let prob = f64::from(
                    *output
                        .text_probability
                        .as_ref()
                        .and_then(|v| v.get(row))
                        .ok_or_else(|| invalid_request("missing text_probability"))?,
                );
                if record.text.is_empty() || record.offsets.is_empty() {
                    pred.insert("span".to_string(), Value::Null);
                    (Value::Null, prob)
                } else {
                    let start = (best / MAX_SPAN).min(record.offsets.len() - 1);
                    let length = best % MAX_SPAN;
                    let last = (start + length).min(record.offsets.len() - 1);
                    let (cs, ce) = typed_prompt::trimmed_span(
                        &record.text,
                        &record.offsets,
                        start,
                        last,
                        trims,
                    );
                    pred.insert("span".to_string(), json!([cs, ce]));
                    (Value::String(slice_chars(&record.text, cs, ce)), prob)
                }
            }
            "number" => {
                let row = num_map[&pos];
                let num_logits = output
                    .number
                    .as_ref()
                    .and_then(|v| v.get(row))
                    .ok_or_else(|| invalid_request("missing number logits"))?;
                let num_groups = output
                    .number_groups
                    .as_ref()
                    .and_then(|v| v.get(row))
                    .map(Vec::as_slice);
                let (mut val, mut conf) = decode_number(num_logits, spec, num_groups)?;
                let readout_f = val.as_f64();
                let mut copied = copied_number(output, row, record, spec, readout_f);
                let veto_enabled =
                    std::env::var("SZ_NUMBER_COPY_VETO").unwrap_or_else(|_| "1".to_string()) == "1";
                if copied.is_some()
                    && !val.is_null()
                    && veto_enabled
                    && !typed_prompt::numeral_spans_value(&record.text, &val, false).is_empty()
                {
                    copied = None;
                }
                if let Some((c_val, c_conf, span)) = copied {
                    val = c_val;
                    conf = c_conf;
                    pred.insert("span".to_string(), json!(span));
                    pred.insert("source".to_string(), Value::String("input".to_string()));
                }
                if spec.get("type").and_then(Value::as_str) == Some("duration") {
                    pred.insert(
                        "unit".to_string(),
                        Value::String(schema::duration_unit(spec)),
                    );
                }
                (val, conf)
            }
            "date" => {
                let row = date_map[&pos];
                let d_logits = output
                    .date
                    .as_ref()
                    .and_then(|v| v.get(row))
                    .ok_or_else(|| invalid_request("missing date logits"))?;
                let (d, c) = decode_date(d_logits)?;
                (Value::String(d), c)
            }
            "time" => {
                let row = time_map[&pos];
                let t_logits = output
                    .time
                    .as_ref()
                    .and_then(|v| v.get(row))
                    .ok_or_else(|| invalid_request("missing time logits"))?;
                let (t, c) = decode_time(t_logits)?;
                (Value::String(t), c)
            }
            "datetime" => {
                let d_row = date_map[&pos];
                let t_row = time_map[&pos];
                let d_logits = output
                    .date
                    .as_ref()
                    .and_then(|v| v.get(d_row))
                    .ok_or_else(|| invalid_request("missing date logits for datetime"))?;
                let t_logits = output
                    .time
                    .as_ref()
                    .and_then(|v| v.get(t_row))
                    .ok_or_else(|| invalid_request("missing time logits for datetime"))?;
                let (day, dc) = decode_date(d_logits)?;
                let (clock, tc) = decode_time(t_logits)?;
                (Value::String(format!("{day}T{clock}")), dc * tc)
            }
            other => {
                return Err(invalid_request(format!(
                    "unsupported typed field kind {other:?}"
                )));
            }
        };

        pred.insert("guess".to_string(), value.clone());
        pred.insert("confidence".to_string(), json!(confidence));
        pred.insert(
            "value".to_string(),
            if pn >= schema::null_threshold(spec) {
                Value::Null
            } else {
                value
            },
        );
        predictions.insert(item.name.clone(), Value::Object(pred));
    }

    if let (Some(tok), Some(eid), Some(spell_ids), Some(spell_lps)) = (
        tokenizer,
        end_id,
        &output.spell_ids,
        &output.spell_log_probs,
    ) {
        let text_count = text_map.len();
        for (pos, item) in fields.iter().enumerate() {
            let row = match item.kind.as_str() {
                "text" => text_map[&pos],
                "text_list" => text_count + tags_map[&pos],
                _ => continue,
            };
            let (spelled, _, parts) = spelled_answer(&spell_ids[row], &spell_lps[row], tok, eid)?;
            if let Some(Value::Object(pred_obj)) = predictions.get(&item.name) {
                let updated =
                    route_spelled(pred_obj, &item.kind, &spelled, &parts, &record.text, false);
                predictions.insert(item.name.clone(), Value::Object(updated));
            }
        }
    }

    if let (Some(tok), Some(slot_ids), Some(slot_lps)) =
        (tokenizer, &output.slot_ids, &output.slot_log_probs)
    {
        let end_tokens = typed_prompt::tokenize_ids(tok, SPELL_END)?;
        let slot_end = *end_tokens
            .first()
            .ok_or_else(|| invalid_request("SPELL_END did not tokenize"))?;
        let slot_rows: Vec<usize> = fields
            .iter()
            .enumerate()
            .filter(|(_, item)| !item.answer_slots.is_empty())
            .map(|(pos, _)| pos)
            .collect();
        let is_evidence = record
            .record
            .get("_meta")
            .and_then(|m| m.get("construction"))
            .and_then(Value::as_str)
            == Some(schema::EVIDENCE_CONSTRUCTION);
        let has_images = record
            .record
            .get("images")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty());

        for (row, &pos) in slot_rows.iter().enumerate() {
            let item = &fields[pos];
            if !matches!(item.kind.as_str(), "text" | "text_list") {
                continue;
            }
            let count = item.answer_slots.len().min(slot_ids[row].len());
            let (spelled, _, parts) = spelled_answer(
                &slot_ids[row][..count],
                &slot_lps[row][..count],
                tok,
                slot_end,
            )?;
            let Some(Value::Object(pred_obj)) = predictions.get(&item.name).cloned() else {
                continue;
            };
            let updated = if is_evidence && item.kind == "text" {
                let spec = &record_fields[&item.name];
                evidence_text_answer(
                    &pred_obj,
                    &slot_ids[row][..count],
                    &slot_lps[row][..count],
                    tok,
                    slot_end,
                    &record.text,
                    spec,
                    has_images,
                )?
            } else {
                route_spelled(&pred_obj, &item.kind, &spelled, &parts, &record.text, true)
            };
            predictions.insert(item.name.clone(), Value::Object(updated));
        }
    }

    Ok(predictions)
}

/// Convenience entry point that converts [`HeadOutputs`] -> [`DecodedOutputs`] and runs [`decode_record`].
pub fn decode_head_outputs(
    output: &HeadOutputs,
    record: &EncodedRecord,
    tokenizer: Option<&Tokenizer>,
    slot_ids: Option<Vec<Vec<u32>>>,
    slot_log_probs: Option<Vec<Vec<f32>>>,
) -> Result<Map<String, Value>> {
    let decoded = DecodedOutputs::from_head_outputs(output, slot_ids, slot_log_probs)?;
    decode_record(&decoded, record, tokenizer, None)
}

/// Merge expanded forecast step estimates (`name.1..=name.H`) back into structured forecast
/// objects (`value`, `median`, `p10`, `p90`, `steps`), matching `model.py::merge_forecasts`.
pub fn merge_forecasts(
    predictions: &Map<String, Value>,
    groups: &[(String, Vec<String>)],
) -> Result<Map<String, Value>> {
    let grouped_steps: HashSet<&str> = groups
        .iter()
        .flat_map(|(_, names)| names.iter().map(String::as_str))
        .collect();
    let mut merged = Map::new();
    for (name, pred) in predictions {
        if !grouped_steps.contains(name.as_str()) {
            merged.insert(name.clone(), pred.clone());
        }
    }
    for (name, step_names) in groups {
        let mut steps = Vec::with_capacity(step_names.len());
        for step_name in step_names {
            let step_pred = predictions
                .get(step_name)
                .ok_or_else(|| invalid_request(format!("missing forecast step {step_name:?}")))?;
            steps.push(step_pred.clone());
        }
        let mut forecast_obj = Map::new();
        for key in ["value", "median", "p10", "p90"] {
            let series: Vec<Value> = steps
                .iter()
                .map(|s| s.get(key).cloned().unwrap_or(Value::Null))
                .collect();
            forecast_obj.insert(key.to_string(), Value::Array(series));
        }
        forecast_obj.insert("steps".to_string(), Value::Array(steps));
        merged.insert(name.clone(), Value::Object(forecast_obj));
    }
    Ok(merged)
}

// ============================================================================
// Math & tensor extraction helpers
// ============================================================================

fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn log_sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        -((-x).exp().ln_1p())
    } else {
        x - x.exp().ln_1p()
    }
}

fn logaddexp_f64(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn logsumexp_f64(values: &[f64]) -> f64 {
    let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_v.is_finite() {
        return max_v;
    }
    let sum: f64 = values.iter().map(|&v| (v - max_v).exp()).sum();
    max_v + sum.ln()
}

fn log_softmax_f64(values: &[f32]) -> Vec<f64> {
    let f64s: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    log_softmax_slice_f64(&f64s)
}

fn log_softmax_slice_f64(values: &[f64]) -> Vec<f64> {
    let lse = logsumexp_f64(values);
    values.iter().map(|&v| v - lse).collect()
}

fn softmax_f64(values: &[f32]) -> Vec<f64> {
    let f64s: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    softmax_slice_f64(&f64s)
}

fn softmax_slice_f64(values: &[f64]) -> Vec<f64> {
    let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_v.is_finite() {
        return vec![0.0; values.len()];
    }
    let exps: Vec<f64> = values.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![0.0; values.len()]
    } else {
        exps.into_iter().map(|e| e / sum).collect()
    }
}

fn argmax_f32(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx
}

fn argmax_f64(values: &[f64]) -> (usize, f64) {
    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (idx, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    (best_idx, best_val)
}

fn argmax_and_softmax_max(values: &[f32]) -> (usize, f64) {
    let lp = log_softmax_f64(values);
    let (idx, best_lp) = argmax_f64(&lp);
    (idx, best_lp.exp())
}

fn split_chunks(flat: &[f32], rows: usize, width: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|r| flat[r * width..(r + 1) * width].to_vec())
        .collect()
}

fn split_3d(flat: &[f32], rows: usize, slots: usize, width: usize) -> Vec<Vec<Vec<f32>>> {
    (0..rows)
        .map(|r| {
            (0..slots)
                .map(|s| {
                    let base = (r * slots + s) * width;
                    flat[base..base + width].to_vec()
                })
                .collect()
        })
        .collect()
}

fn extract_2d(arr: &MxArray) -> Result<Vec<Vec<f32>>> {
    let shape = arr.shape()?;
    let rows = shape[0] as usize;
    let width = shape[1] as usize;
    Ok(split_chunks(&arr.to_float32()?, rows, width))
}

fn extract_3d(arr: &MxArray) -> Result<Vec<Vec<Vec<f32>>>> {
    let shape = arr.shape()?;
    let rows = shape[0] as usize;
    let slots = shape[1] as usize;
    let width = shape[2] as usize;
    Ok(split_3d(&arr.to_float32()?, rows, slots, width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::clairic::{
        head::{self, ClairicHead, ClairicHeadConfig},
        typed_prompt::{EncodeRecordOptions, build_prediction_views, encode_record},
    };
    use serde_json::json;
    use std::path::PathBuf;

    fn clef_tokenizer() -> Tokenizer {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../__test__/fixtures/clef/tokenizer.json");
        Tokenizer::from_file(&path).expect("load clef fixture tokenizer")
    }

    fn det_data(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = ((x >> 40) as f32) / ((1u32 << 24) as f32);
            out.push((u * 2.0 - 1.0) * scale);
        }
        out
    }

    fn synthetic_head(hidden_size: i64, bins: i64) -> ClairicHead {
        let cfg = ClairicHeadConfig {
            hidden_size,
            bins,
            pointer_width: 16,
            readout: "lexical".to_string(),
            route_width: 32,
            route_layers: 1,
            route_heads: 4,
            slot_rank: 8,
            estimate_lexical: true,
            span_trim: true,
            loop_feedback: false,
            answer_slots: true,
            answer_slot_routing: true,
            number_groups: true,
            number_pointer: true,
        };
        let mut weights = HashMap::new();
        let h = cfg.hidden_size;
        let w = cfg.route_width;
        let p = cfg.pointer_width;
        let r = cfg.slot_rank;
        let mut seed = 1u64;
        fn add(weights: &mut HashMap<String, MxArray>, seed: &mut u64, name: &str, shape: &[i64]) {
            let numel: usize = shape.iter().map(|&d| d as usize).product::<usize>().max(1);
            let data = if name.ends_with(".weight") && shape.len() == 1 {
                vec![1.0f32; numel]
            } else if name == "lexical_scale" {
                vec![2.0, 2.0, 1.5]
            } else {
                det_data(*seed, numel, 0.2)
            };
            *seed += 1;
            weights.insert(
                name.to_string(),
                MxArray::from_float32(&data, shape).unwrap(),
            );
        }
        fn norm(weights: &mut HashMap<String, MxArray>, seed: &mut u64, prefix: &str, width: i64) {
            add(weights, seed, &format!("{prefix}.weight"), &[width]);
            add(weights, seed, &format!("{prefix}.bias"), &[width]);
        }
        fn linear(
            weights: &mut HashMap<String, MxArray>,
            seed: &mut u64,
            prefix: &str,
            inp: i64,
            out: i64,
            bias: bool,
        ) {
            add(weights, seed, &format!("{prefix}.weight"), &[out, inp]);
            if bias {
                add(weights, seed, &format!("{prefix}.bias"), &[out]);
            }
        }
        add(&mut weights, &mut seed, "slot_route_scale", &[8]);
        norm(&mut weights, &mut seed, "norm", h);
        norm(&mut weights, &mut seed, "token_norm", h);
        linear(&mut weights, &mut seed, "null", h, 1, true);
        linear(&mut weights, &mut seed, "span_query", h, 2 * p, true);
        linear(&mut weights, &mut seed, "span_key", h, 2 * p, true);
        add(&mut weights, &mut seed, "length_bias", &[head::MAX_SPAN]);
        linear(&mut weights, &mut seed, "tag_query", h, 2 * p, true);
        linear(&mut weights, &mut seed, "tag_key", h, p, true);
        add(&mut weights, &mut seed, "tag_bias", &[2]);
        linear(
            &mut weights,
            &mut seed,
            "trim",
            h,
            2 * head::TRIM_CLASSES,
            true,
        );
        linear(&mut weights, &mut seed, "route_in", h, w, false);
        add(
            &mut weights,
            &mut seed,
            "kind_embedding.weight",
            &[head::KIND_EMBEDDING_ROWS, w],
        );
        linear(&mut weights, &mut seed, "memory_in", h, w, false);
        for n in ["norm1", "norm2", "norm3"] {
            norm(&mut weights, &mut seed, &format!("route_layers.0.{n}"), w);
        }
        for attn in ["self_attn", "multihead_attn"] {
            let pref = format!("route_layers.0.{attn}");
            add(
                &mut weights,
                &mut seed,
                &format!("{pref}.in_proj_weight"),
                &[3 * w, w],
            );
            add(
                &mut weights,
                &mut seed,
                &format!("{pref}.in_proj_bias"),
                &[3 * w],
            );
            linear(
                &mut weights,
                &mut seed,
                &format!("{pref}.out_proj"),
                w,
                w,
                true,
            );
        }
        linear(
            &mut weights,
            &mut seed,
            "route_layers.0.linear1",
            w,
            4 * w,
            true,
        );
        linear(
            &mut weights,
            &mut seed,
            "route_layers.0.linear2",
            4 * w,
            w,
            true,
        );
        norm(&mut weights, &mut seed, "route_norm", w);
        linear(&mut weights, &mut seed, "route_out", w, h, true);
        add(&mut weights, &mut seed, "route_gate", &[]);
        add(&mut weights, &mut seed, "digit_rows", &[10, h]);
        add(&mut weights, &mut seed, "month_rows", &[12, h]);
        add(&mut weights, &mut seed, "sign_rows", &[2, h]);
        add(&mut weights, &mut seed, "lexical_scale", &[3]);
        linear(
            &mut weights,
            &mut seed,
            "slot_down",
            h,
            (head::DIGIT_SLOTS + 2) * r,
            false,
        );
        add(
            &mut weights,
            &mut seed,
            "slot_up",
            &[head::DIGIT_SLOTS, r, 10],
        );
        add(&mut weights, &mut seed, "month_up", &[r, 12]);
        add(&mut weights, &mut seed, "sign_up", &[r, 2]);
        linear(
            &mut weights,
            &mut seed,
            "slot_residual",
            w,
            head::DIGIT_SLOTS * 10 + 12 + 2,
            true,
        );
        add(&mut weights, &mut seed, "residual_gate", &[]);
        linear(
            &mut weights,
            &mut seed,
            "exponent",
            h,
            head::EXPONENT_CLASSES,
            true,
        );
        linear(&mut weights, &mut seed, "item_end", h, 1, true);
        linear(&mut weights, &mut seed, "bins", h, bins, true);
        linear(&mut weights, &mut seed, "offset", h, bins, true);
        linear(&mut weights, &mut seed, "overflow", h, 2, true);
        linear(&mut weights, &mut seed, "estimate_gate", h, 1, true);
        linear(
            &mut weights,
            &mut seed,
            "estimate_spread",
            h,
            1 + head::LEXICAL_DIGITS,
            true,
        );
        linear(
            &mut weights,
            &mut seed,
            "number_group_down",
            h,
            head::GROUP_RANK,
            true,
        );
        linear(
            &mut weights,
            &mut seed,
            "number_group_up",
            head::GROUP_RANK,
            head::NUMBER_GROUPS * head::GROUP_CLASSES,
            true,
        );
        linear(&mut weights, &mut seed, "copy_query", h, 2 * p, true);
        linear(&mut weights, &mut seed, "copy_gate", h, 1, true);
        ClairicHead::from_weights(cfg, weights, hidden_size, false).unwrap()
    }

    #[test]
    fn sequence_matcher_matches_python_difflib() {
        assert!((sequence_matcher_ratio("abcd", "abcd") - 1.0).abs() < 1e-12);
        assert!((sequence_matcher_ratio("abxcd", "abcd") - (8.0 / 9.0)).abs() < 1e-12);
        assert!((sequence_matcher_ratio("", "") - 1.0).abs() < 1e-12);
        assert!((similarity(Some("Acme, Corp."), Some("acme corp")) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn decode_number_date_time_and_group_rescoring() {
        // Construct logits for -123.45: sign=1, power=2 -> exp_class = 2 - (-12) + 1 = 15, digits=123450000000
        let mut logits = vec![-10.0f32; NUMBER_WIDTH];
        logits[1] = 10.0; // negative
        logits[2 + 15] = 10.0; // power = 2
        let target_digits = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0];
        for (pos, &d) in target_digits.iter().enumerate() {
            logits[2 + EXPONENT_CLASSES + pos * 10 + d] = 10.0;
        }
        let spec = json!({"type": "number", "decimals": 2});
        let (val, conf) = decode_number(&logits, &spec, None).unwrap();
        assert_eq!(val, json!(-123.45));
        assert!(conf > 0.99);

        // With group rescoring overriding hundreds group (group_idx=0 -> g=4) to 789 -> -789.45
        // (each wrong digit costs ~20 nats, so 3 digits cost ~60 nats; +80 overcomes it)
        let mut groups = vec![0.0f32; NUMBER_GROUPS * GROUP_CLASSES];
        groups[4 * GROUP_CLASSES + 789] = 80.0;
        let (gval, _) = decode_number(&logits, &spec, Some(&groups)).unwrap();
        assert_eq!(gval, json!(-789.45));

        // Date: 2024-02-29 (leap year) -> year_class = 2024 - 1800 = 224, month=1 (Feb), day=28 (29th)
        let mut d_logits = vec![-10.0f32; DATE_WIDTH];
        d_logits[YEAR_DIGITS * 10 + 224] = 10.0;
        d_logits[YEAR_DIGITS * 10 + YEAR_CLASSES + 1] = 10.0;
        d_logits[YEAR_DIGITS * 10 + YEAR_CLASSES + 12 + 28] = 10.0;
        let (date_str, d_conf) = decode_date(&d_logits).unwrap();
        assert_eq!(date_str, "2024-02-29");
        assert!(d_conf > 0.99);

        // Non-leap year 2023 should mask Feb 29 and pick next best valid day
        let mut d2023 = vec![-10.0f32; DATE_WIDTH];
        d2023[YEAR_DIGITS * 10 + 223] = 10.0;
        d2023[YEAR_DIGITS * 10 + YEAR_CLASSES + 1] = 10.0;
        d2023[YEAR_DIGITS * 10 + YEAR_CLASSES + 12 + 28] = 10.0; // invalid Feb 29
        d2023[YEAR_DIGITS * 10 + YEAR_CLASSES + 12 + 27] = 5.0; // valid Feb 28
        let (date_2023, _) = decode_date(&d2023).unwrap();
        assert_eq!(date_2023, "2023-02-28");

        // Time: 08:15:42
        let mut t_logits = vec![-10.0f32; TIME_WIDTH];
        t_logits[8] = 10.0;
        t_logits[24 + 15] = 10.0;
        t_logits[84 + 42] = 10.0;
        let (time_str, t_conf) = decode_time(&t_logits).unwrap();
        assert_eq!(time_str, "08:15:42");
        assert!(t_conf > 0.99);
    }

    #[test]
    fn end_to_end_synthetic_clairic_head_forward_to_decode_record_and_compact_roundtrip() {
        let tok = clef_tokenizer();
        let record = json!({
            "input": "Vendor: Acme Corp. Amount: 1,234.56 USD. Date: 2025-05-01. Time: 14:30:00.",
            "fields": {
                "vendor": {"type": "text"},
                "amount": {"type": "number", "decimals": 2},
                "date": {"type": "date"},
                "clock": {"type": "time"},
                "eta": {"type": "estimate", "min": 0, "max": 100},
                "fc.1": {"type": "estimate", "min": 0, "max": 100},
                "fc.2": {"type": "estimate", "min": 0, "max": 100},
                "amounts": {"type": "list", "items": "number"},
                "dates": {"type": "list", "items": "date"},
                "items": {"type": "list", "items": "text"}
            }
        });
        let opts = EncodeRecordOptions {
            max_length: 4096,
            ..EncodeRecordOptions::default()
        };
        let encoded = encode_record(&tok, &record, &opts).unwrap().unwrap();
        let (start, stop, views, _as_layout) = build_prediction_views(&encoded).unwrap();

        let hidden_size = 64i64;
        let bins = 16i64;
        let head = synthetic_head(hidden_size, bins);
        let seq_len = encoded.input_ids.len() as i64;
        let hidden = MxArray::from_float32(
            &det_data(777, (seq_len * hidden_size) as usize, 0.4),
            &[seq_len, hidden_size],
        )
        .unwrap();
        let memory = head.memory(&hidden, stop as i64).unwrap();
        let head_out = head
            .forward(&hidden, start as i64, stop as i64, &views, &memory)
            .unwrap();

        let decoded_out = DecodedOutputs::from_head_outputs(&head_out, None, None).unwrap();
        // Verify compact_output -> expand_output lossless roundtrip
        let flat = compact_output(&decoded_out);
        let expanded =
            expand_output(&flat, &encoded, bins as usize, true, true, true, true).unwrap();
        assert_eq!(expanded.null.len(), decoded_out.null.len());

        let preds = decode_record(&expanded, &encoded, Some(&tok), None).unwrap();
        assert!(preds.contains_key("vendor"));
        assert!(preds.contains_key("amount"));
        assert!(preds.contains_key("date"));
        assert!(preds.contains_key("clock"));
        assert!(preds.contains_key("amounts"));
        assert!(preds.contains_key("dates"));
        assert!(preds.contains_key("items"));

        let est = &preds["eta"];
        let quantiles = est["quantiles"].as_array().unwrap();
        assert_eq!(quantiles.len(), 99);
        assert_eq!(est["median"], quantiles[49]);
        assert_eq!(est["p10"], quantiles[9]);
        assert_eq!(est["p90"], quantiles[89]);
        assert!(est["p_below"].as_f64().unwrap().is_finite());
        assert!(est["p_above"].as_f64().unwrap().is_finite());

        // Verify forecast merge
        let merged = merge_forecasts(
            &preds,
            &[(
                "fc".to_string(),
                vec!["fc.1".to_string(), "fc.2".to_string()],
            )],
        )
        .unwrap();
        assert!(merged.contains_key("fc"));
        assert!(!merged.contains_key("fc.1"));
        assert_eq!(merged["fc"]["steps"].as_array().unwrap().len(), 2);
        assert_eq!(merged["fc"]["value"].as_array().unwrap().len(), 2);
    }
}
