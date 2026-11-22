//! Typed-head (Path B) prompt encoding, numeral parsing/convention detection,
//! estimate edge tables, prediction layout serialization, and `PredictionViews` construction.
//!
//! Direct port of upstream `clairic_runtime/model.py` (lines 1-835 and 2711-2876).

use super::{
    decimal_util,
    errors::invalid_request,
    head::{self, PredictionViews},
    schema,
};
use bigdecimal::BigDecimal;
use fancy_regex::Regex as FancyRegex;
use napi::{Error, Result};
use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

pub const SYSTEM_PREFIX: &str = concat!(
    "<|im_start|>system\n",
    "Fill each field from the input with a value of its type. Text copies a span of the input. Numbers, dates, times, datetimes, and durations are read from the input or worked out from it. A list holds every matching value in input order. A field is not stated when the input does not determine it. Estimates predict a value inside the declared range.",
    "\nFields:\n"
);
pub const EVIDENCE_SYSTEM_PREFIX: &str = concat!(
    "<|im_start|>system\n",
    "Fill each field with a value of its type, following its instructions and the question's scope. For document-specific extraction, use the supplied input and return not stated when the requested information is missing. Preserve exact text, names and numbers unless normalization or a derived answer is requested. General factual questions may use reliable prior knowledge unless the instructions restrict answers to the supplied evidence; return not stated when the answer cannot be determined reliably. A list holds every matching item in the requested order, including repeated items. Estimates and forecasts predict distributions over the declared range and supported below-range or above-range outcomes. Follow the requested units, time intervals and horizons.",
    "\nFields:\n"
);
pub const USER_PREFIX: &str = "<|im_end|>\n<|im_start|>user\n";
pub const ASSISTANT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n";
pub const ANSWER_CUE: &str = "\nAnswer:";

pub const QUERY_TOKENS: usize = 256;
pub const MAX_SPAN: usize = head::MAX_SPAN as usize;
pub const TRIM_CLASSES: usize = head::TRIM_CLASSES as usize;
pub const MIN_INPUT_TOKENS: usize = 16;
pub const BINS: usize = head::BINS as usize;
pub const POINTER_WIDTH: usize = head::POINTER_WIDTH as usize;
pub const NUMBER_DIGITS: usize = head::NUMBER_DIGITS as usize;
pub const MIN_EXPONENT: i64 = head::MIN_EXPONENT;
pub const MAX_EXPONENT: i64 = head::MAX_EXPONENT;
pub const EXPONENT_CLASSES: usize = head::EXPONENT_CLASSES as usize;
pub const YEAR_DIGITS: usize = head::YEAR_DIGITS as usize;
pub const YEAR_LOW: i32 = head::YEAR_LOW;
pub const YEAR_HIGH: i32 = head::YEAR_HIGH;
pub const YEAR_OUTSIDE: usize = head::YEAR_OUTSIDE as usize;
pub const YEAR_CLASSES: usize = head::YEAR_CLASSES as usize;
pub const LIST_SLOTS: usize = schema::MAX_LIST_ITEMS;
pub const INSIDE: usize = 0;
pub const BELOW: usize = 1;
pub const ABOVE: usize = 2;
pub const LEXICAL_DIGITS: usize = head::LEXICAL_DIGITS as usize;
pub const ESTIMATE_SPREAD_LIMIT: f64 = head::ESTIMATE_SPREAD_LIMIT;
pub const EDGE_NEGATIVE: u8 = 0;
pub const EDGE_ZERO: u8 = 1;
pub const EDGE_POSITIVE: u8 = 2;
pub const TAG_OUTSIDE: usize = 0;
pub const TAG_BEGIN: usize = 1;
pub const TAG_INSIDE: usize = 2;
pub const NUMBER_WIDTH: usize = head::NUMBER_WIDTH as usize;
pub const GROUP_DIGITS: usize = head::GROUP_DIGITS as usize;
pub const GROUP_CLASSES: usize = head::GROUP_CLASSES as usize;
pub const GROUP_LOW: i64 = head::GROUP_LOW;
pub const GROUP_HIGH: i64 = head::GROUP_HIGH;
pub const NUMBER_GROUPS: usize = head::NUMBER_GROUPS as usize;
pub const COPY_CONFIDENT: f64 = 0.9;
pub const COPY_MIN_CONFIDENCE: f64 = 0.5;
pub const SLOT_TEXT: &str = "_";
pub const SLOT_CUE_TOKENS: usize = 32;
pub const DATE_WIDTH: usize = head::DATE_WIDTH as usize;
pub const TIME_WIDTH: usize = head::TIME_WIDTH as usize;
pub const MASKED: f32 = -1_000_000_000.0;
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
pub const NUMERAL_SPAN: usize = 48;
pub const NUMERAL_SEPARATORS: &str = ",.'\u{00a0}\u{202f}\u{2009}\u{2007} \u{066b}\u{066c}";

pub const SCALAR_KINDS: &[&str] = &["text", "number", "date", "time", "datetime"];
pub const LIST_KINDS: &[&str] = &["number_list", "date_list"];
pub const FIELD_KINDS: &[&str] = &[
    "text",
    "number",
    "date",
    "time",
    "datetime",
    "estimate",
    "text_list",
    "number_list",
    "date_list",
];
pub const ROUTE_KINDS: &[&str] = &head::ROUTE_KINDS;

/// Prompt template variant strings (`model.py::PromptLayout`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptLayout {
    pub schema_header: &'static str,
    pub query_header: &'static str,
    pub answer_cue: &'static str,
    pub slot_header: &'static str,
}

pub const PROMPT_LAYOUTS: [PromptLayout; 3] = [
    PromptLayout {
        schema_header: "",
        query_header: "",
        answer_cue: ANSWER_CUE,
        slot_header: "",
    },
    PromptLayout {
        schema_header: "Field definition:\n",
        query_header: "\nField query:",
        answer_cue: "\nValue:",
        slot_header: "\nAnswer field:",
    },
    PromptLayout {
        schema_header: "Requested field:\n",
        query_header: "\nRequested field:",
        answer_cue: "\nResult:",
        slot_header: "\nResult for field:",
    },
];

pub fn prompt_layout(template_variant: usize, complete_field: bool) -> Result<PromptLayout> {
    if template_variant >= PROMPT_LAYOUTS.len() {
        return Err(invalid_request("template_variant must be 0, 1, or 2"));
    }
    if template_variant != 0 && !complete_field {
        return Err(invalid_request(
            "alternative layouts require the evidence construction",
        ));
    }
    Ok(PROMPT_LAYOUTS[template_variant])
}

pub fn slot_count_for_kind(kind: &str) -> Option<usize> {
    match kind {
        "text" => Some(80),
        "text_list" => Some(64),
        "number" => Some(32),
        _ => None,
    }
}

pub fn head_kind(spec: &Value) -> String {
    let kind = spec.get("type").and_then(Value::as_str).unwrap_or("");
    if kind == "duration" {
        "number".to_string()
    } else if kind == "list" {
        let items = spec.get("items").and_then(Value::as_str).unwrap_or("");
        format!("{items}_list")
    } else {
        kind.to_string()
    }
}

/// Decompose a finite float into `(sign_bit, exponent_class, 12 digits)` matching `model.py::number_components`.
pub fn number_components(value: f64) -> Option<(u8, usize, [u8; NUMBER_DIGITS])> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some((0, 0, [0u8; NUMBER_DIGITS]));
    }
    let sci = format!("{:.prec$e}", value.abs(), prec = NUMBER_DIGITS - 1);
    let (mantissa, exp_str) = sci.split_once('e')?;
    let power: i64 = exp_str.parse().ok()?;
    if !(MIN_EXPONENT..=MAX_EXPONENT).contains(&power) {
        return None;
    }
    let mut digits = [0u8; NUMBER_DIGITS];
    let mut idx = 0;
    for b in mantissa.bytes() {
        if b.is_ascii_digit() && idx < NUMBER_DIGITS {
            digits[idx] = b - b'0';
            idx += 1;
        }
    }
    if idx != NUMBER_DIGITS {
        return None;
    }
    let sign = u8::from(value < 0.0);
    let exp_class = (power - MIN_EXPONENT + 1) as usize;
    Some((sign, exp_class, digits))
}

/// Decompose a bin edge value for lexical estimate CDF projection (`model.py::edge_number`).
pub fn edge_number(value: f64) -> (u8, usize, [u8; LEXICAL_DIGITS], f64) {
    if value == 0.0 || !value.is_finite() {
        return (EDGE_ZERO, 1, [0u8; LEXICAL_DIGITS], 0.0);
    }
    let sci = format!("{:.prec$e}", value.abs(), prec = LEXICAL_DIGITS + 8);
    let Some((mantissa, exp_str)) = sci.split_once('e') else {
        return (EDGE_ZERO, 1, [0u8; LEXICAL_DIGITS], 0.0);
    };
    let power: i64 = exp_str.parse().unwrap_or(0);
    let sign = if value < 0.0 {
        EDGE_NEGATIVE
    } else {
        EDGE_POSITIVE
    };
    if power < MIN_EXPONENT {
        return (sign, 1, [0u8; LEXICAL_DIGITS], 0.0);
    }
    if power > MAX_EXPONENT {
        return (sign, EXPONENT_CLASSES - 1, [9u8; LEXICAL_DIGITS], 1.0);
    }
    let digits_str: String = mantissa.chars().filter(|&c| c != '.').collect();
    let mut lead = [0u8; LEXICAL_DIGITS];
    for (i, b) in digits_str.bytes().take(LEXICAL_DIGITS).enumerate() {
        lead[i] = b.saturating_sub(b'0');
    }
    let frac_str = if digits_str.len() > LEXICAL_DIGITS {
        format!("0.{}", &digits_str[LEXICAL_DIGITS..])
    } else {
        "0.0".to_string()
    };
    let frac = frac_str.parse::<f64>().unwrap_or(0.0);
    (sign, (power - MIN_EXPONENT + 1) as usize, lead, frac)
}

/// Precomputed bin edge decompositions (`bins + 1` edges) for lexical estimate projection.
#[derive(Debug, Clone)]
pub struct EstimateEdgeParts {
    pub sign: Vec<u8>,
    pub class: Vec<usize>,
    pub digits: Vec<[u8; LEXICAL_DIGITS]>,
    pub frac: Vec<f64>,
}

pub fn lexical_estimate(spec: &Value) -> bool {
    !schema::is_date_estimate(spec) || schema::date_granularity(spec) == "year"
}

pub fn estimate_edge_parts(spec: &Value, bins: usize) -> Result<Option<EstimateEdgeParts>> {
    if !lexical_estimate(spec) {
        return Ok(None);
    }
    let (low, high, scale) = schema::estimate_range(spec)?;
    let count = bins + 1;
    let mut sign = Vec::with_capacity(count);
    let mut class = Vec::with_capacity(count);
    let mut digits = Vec::with_capacity(count);
    let mut frac = Vec::with_capacity(count);
    for idx in 0..=bins {
        let pos = (idx as f64) / (bins as f64);
        let (s, c, d, f) = edge_number(schema::from_unit(pos, low, high, &scale));
        sign.push(s);
        class.push(c);
        digits.push(d);
        frac.push(f);
    }
    Ok(Some(EstimateEdgeParts {
        sign,
        class,
        digits,
        frac,
    }))
}

pub fn prefix_window(
    spec: &Value,
    prefix: &(u8, usize, Vec<u8>),
    bins: usize,
) -> Result<Option<(usize, usize)>> {
    let (sign, exponent_class, digits) = prefix;
    let power = (*exponent_class as i64) - 1 + MIN_EXPONENT;
    let shift = power - (digits.len() as i64) + 1;
    let unit = decimal_util::scaleb(&BigDecimal::from(1), shift);
    let digits_str: String = digits.iter().map(|d| (b'0' + d) as char).collect();
    let Some(int_dec) = decimal_util::parse_decimal(&digits_str) else {
        return Ok(None);
    };
    let start = &int_dec * &unit;
    let end = &start + &unit;
    let (mut first, mut last) = (
        start.to_string().parse::<f64>().unwrap_or(0.0),
        end.to_string().parse::<f64>().unwrap_or(0.0),
    );
    if *sign != 0 {
        (first, last) = (-last, -first);
    }
    let (low, high, scale) = schema::estimate_range(spec)?;
    let a = schema::raw_unit(first, low, high, &scale);
    let b = schema::raw_unit(last, low, high, &scale);
    let (lower, upper) = if a <= b { (a, b) } else { (b, a) };
    if upper <= 0.0 || lower >= 1.0 {
        return Ok(None);
    }
    let bins_f = bins as f64;
    let first_bin = ((lower * bins_f).floor() as isize).clamp(0, (bins - 1) as isize) as usize;
    let last_bin = (((upper * bins_f).ceil() as isize) - 1)
        .clamp(first_bin as isize, (bins - 1) as isize) as usize;
    Ok(Some((first_bin, last_bin)))
}

/// Format a decimal as a stripped plain string (`model.py::plain_number`).
pub fn plain_number(value: &BigDecimal) -> String {
    decimal_util::to_stripped_plain_string(value)
}

// ============================================================================
// Numeral parsing & convention detection (exported for Wave 3 grounding/sums)
// ============================================================================

/// Compiled `NUMERAL_RUN` regex (`model.py:93-95`).
pub fn numeral_run_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(
            r"(?:(?<![\d.])\.(?=\d))?\d+(?:[,.'\u{00a0}\u{202f}\u{2009}\u{2007} \u{066b}\u{066c}]\d+)*",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

struct NumeralReading {
    regex: Regex,
    separators: &'static str,
    decimal_mark: char,
}

fn numeral_readings() -> &'static [NumeralReading] {
    static READINGS: OnceLock<Vec<NumeralReading>> = OnceLock::new();
    READINGS.get_or_init(|| {
        let specs: [(&str, &str, char); 6] = [
            (
                r"^(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)$",
                ",",
                '.',
            ),
            (
                r"^(?:\d{1,3}(?:[.\u{00a0}\u{202f}\u{2009}\u{2007}' ]\d{3})+(?:,\d+)?|\d+(?:,\d+)?)$",
                ".\u{00a0}\u{202f}\u{2009}\u{2007}' ",
                ',',
            ),
            (
                r"^(?:\d{1,3}(?:[\u{00a0}\u{202f}\u{2009}\u{2007}' ]\d{3})+(?:\.\d+)?)$",
                "\u{00a0}\u{202f}\u{2009}\u{2007}' ",
                '.',
            ),
            (r"^(?:\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?)$", ",", '.'),
            (
                r"^(?:\d{1,3}(?:\u{066c}\d{3})+(?:\u{066b}\d+)?|\d+\u{066b}\d+)$",
                "\u{066c}",
                '\u{066b}',
            ),
            (
                r"^(?:\d{1,2}(?:\u{066c}\d{2})+\u{066c}\d{3}(?:\u{066b}\d+)?)$",
                "\u{066c}",
                '\u{066b}',
            ),
        ];
        specs
            .into_iter()
            .map(|(pat, seps, mark)| NumeralReading {
                regex: Regex::new(pat).unwrap_or_else(|_| unreachable!()),
                separators: seps,
                decimal_mark: mark,
            })
            .collect()
    })
}

fn scale_exponent(word: &str) -> Option<i64> {
    match word {
        "thousand" | "thousands" | "Thousand" | "k" | "K" | "ألف" | "हज़ार" => Some(3),
        "lakh" | "lakhs" | "lac" | "lacs" | "Lakh" | "Lakhs" | "लाख" => Some(5),
        "million" | "millions" | "Million" | "MILLION" | "mn" | "Mn" | "mln" | "M" | "MM"
        | "مليون" => Some(6),
        "crore" | "crores" | "Crore" | "Crores" | "cr" | "Cr" | "करोड़" => Some(7),
        "billion" | "billions" | "Billion" | "BILLION" | "bn" | "Bn" | "B" | "مليار" => {
            Some(9)
        }
        "trillion" | "Trillion" | "tn" => Some(12),
        _ => None,
    }
}

fn numeral_scale_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        let mut words = vec![
            "thousand",
            "thousands",
            "million",
            "millions",
            "mn",
            "mln",
            "MM",
            "billion",
            "billions",
            "bn",
            "trillion",
            "tn",
            "lakh",
            "lakhs",
            "lac",
            "lacs",
            "crore",
            "crores",
            "cr",
            "Cr",
            "Mn",
            "Bn",
            "Lakh",
            "Lakhs",
            "Crore",
            "Crores",
            "Million",
            "Billion",
            "Thousand",
            "Trillion",
            "MILLION",
            "BILLION",
            "ألف",
            "مليون",
            "مليار",
            "हज़ार",
            "लाख",
            "करोड़",
        ];
        words.sort_by_key(|a| std::cmp::Reverse(a.chars().count()));
        let joined = words
            .into_iter()
            .map(regex::escape)
            .collect::<Vec<_>>()
            .join("|");
        let pattern =
            format!(r"\A(?:[ \u{{00a0}}\u{{202f}}\u{{2009}}]?({joined})|(k|K|M|B))(?![^\W\d_])");
        FancyRegex::new(&pattern).unwrap_or_else(|_| unreachable!())
    })
}

fn convention_language_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\b(?:Rechnung|Gesamtbetrag|Betrag|Lieferschein|Factuur|Bedrag|Totaal|Pakbon|Fattura|Importo|Totale|Fatura|Nota fiscal|Factura|Importe|Albarán|Facture|Montant)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn convention_language_us_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b(?:México|Mexico|MXN|CFDI|RFC)\b").unwrap_or_else(|_| unreachable!())
    })
}

fn convention_stated_eu_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)european (?:number )?(?:notation|format)|(?:dot|period|point)s? (?:separates?|for|marks?) (?:the )?thousands|comma (?:separates?|for|marks?|as) (?:the )?decimal|comma decimals?|decimal comma|dot thousands|1\.234,56|1 234,56",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn convention_stated_us_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)comma (?:separates?|for|marks?) (?:the )?thousands|(?:dot|period|point) (?:separates?|for|marks?|as) (?:the )?decimal|comma thousands|dot decimals?|1,234\.56",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn to_ascii_digit(ch: char) -> char {
    match ch {
        '0'..='9' => ch,
        '\u{0660}'..='\u{0669}' => ((ch as u32 - 0x0660) as u8 + b'0') as char,
        '\u{06F0}'..='\u{06F9}' => ((ch as u32 - 0x06F0) as u8 + b'0') as char,
        '\u{0966}'..='\u{096F}' => ((ch as u32 - 0x0966) as u8 + b'0') as char,
        '\u{FF10}'..='\u{FF19}' => ((ch as u32 - 0xFF10) as u8 + b'0') as char,
        _ => ch,
    }
}

/// Parse all valid numeric interpretations of `printed` under an optional `"us"` / `"eu"`
/// convention (`model.py::numeral_values`).
pub fn numeral_values(printed: &str, convention: Option<&str>) -> Vec<BigDecimal> {
    let readings = numeral_readings();
    let both = convention.is_some()
        && readings[0].regex.is_match(printed)
        && readings[1].regex.is_match(printed);
    let mut values: Vec<BigDecimal> = Vec::new();
    for (index, reading) in readings.iter().enumerate() {
        if both && index == if convention == Some("us") { 1 } else { 0 } {
            continue;
        }
        if reading.regex.is_match(printed) {
            let mut plain = String::with_capacity(printed.len());
            for ch in printed.chars() {
                if reading.separators.contains(ch) {
                    continue;
                }
                if ch == reading.decimal_mark {
                    plain.push('.');
                } else {
                    plain.push(to_ascii_digit(ch));
                }
            }
            if let Some(val) = decimal_util::parse_decimal(&plain)
                && !values.contains(&val)
            {
                values.push(val);
            }
        }
    }
    values
}

/// Infer whether `text` uses `"us"` or `"eu"` numeric punctuation (`model.py::numeral_convention`).
pub fn numeral_convention(text: &str) -> Option<&'static str> {
    let stated_eu = convention_stated_eu_regex().is_match(text);
    let stated_us = convention_stated_us_regex().is_match(text);
    match (stated_eu, stated_us) {
        (true, false) => return Some("eu"),
        (false, true) => return Some("us"),
        _ => {}
    }

    let readings = numeral_readings();
    let mut shown: Vec<&'static str> = Vec::new();
    let mut pos = 0usize;
    while pos < text.len() {
        let Ok(Some(m)) = numeral_run_regex().find_from_pos(text, pos) else {
            break;
        };
        let printed = m.as_str();
        pos = m.end().max(pos + 1);
        if !printed.contains(',') && !printed.contains('.') {
            continue;
        }
        let us = readings[0].regex.is_match(printed);
        let eu = readings[1].regex.is_match(printed);
        if us != eu {
            shown.push(if us { "us" } else { "eu" });
        }
    }
    if shown.len() >= 2 && shown.iter().all(|&s| s == shown[0]) {
        return Some(shown[0]);
    }
    if shown.is_empty()
        && convention_language_regex().is_match(text)
        && !convention_language_us_regex().is_match(text)
    {
        return Some("eu");
    }
    None
}

fn char_to_byte_map(text: &str) -> Vec<usize> {
    let mut map = Vec::with_capacity(text.chars().count() + 1);
    for (bi, _) in text.char_indices() {
        map.push(bi);
    }
    map.push(text.len());
    map
}

fn numeral_candidates(text: &str) -> Vec<(Vec<BigDecimal>, usize, usize, bool)> {
    if text.is_empty() {
        return Vec::new();
    }
    let b2c = schema::byte_to_char_map(text);
    let c2b = char_to_byte_map(text);
    let chars: Vec<char> = text.chars().collect();
    let convention = numeral_convention(text);

    let mut found: Vec<(Vec<BigDecimal>, usize, usize, bool)> = Vec::new();
    let mut pos = 0usize;
    while pos < text.len() {
        let Ok(Some(run)) = numeral_run_regex().find_from_pos(text, pos) else {
            break;
        };
        let begin_byte = run.start();
        let finish_byte = run.end();
        pos = finish_byte.max(pos + 1);

        let begin = b2c[begin_byte];
        let finish = b2c[finish_byte];

        let cuts: Vec<usize> = (begin..finish)
            .filter(|&idx| idx > begin && NUMERAL_SEPARATORS.contains(chars[idx]))
            .collect();
        let mut starts = Vec::with_capacity(cuts.len() + 1);
        starts.push(begin);
        for &idx in &cuts {
            starts.push(idx + 1);
        }
        let mut ends = Vec::with_capacity(cuts.len() + 1);
        ends.extend_from_slice(&cuts);
        ends.push(finish);

        let scale_match = numeral_scale_regex()
            .captures(&text[finish_byte..])
            .ok()
            .flatten();
        let (exponent, scale_end_char) = match scale_match {
            Some(caps) => {
                let word = caps
                    .get(1)
                    .or_else(|| caps.get(2))
                    .map(|m| m.as_str())
                    .unwrap_or("");
                let m_end = caps
                    .get(0)
                    .map(|m| finish_byte + m.end())
                    .unwrap_or(finish_byte);
                (scale_exponent(word), b2c[m_end])
            }
            None => (None, finish),
        };

        let mut run_found: Vec<(Vec<BigDecimal>, usize, usize)> = Vec::new();
        for &start in &starts {
            let first_end_idx = ends.partition_point(|&e| e < start + 1);
            for &end in &ends[first_end_idx..] {
                if end - start > NUMERAL_SPAN {
                    break;
                }
                let slice = &text[c2b[start]..c2b[end]];
                let values = numeral_values(slice, convention);
                if values.is_empty() {
                    continue;
                }
                if end == finish
                    && let Some(exp) = exponent
                {
                    let mut combined: Vec<BigDecimal> = values
                        .iter()
                        .map(|v| decimal_util::scaleb(v, exp))
                        .collect();
                    for v in values {
                        if !combined.contains(&v) {
                            combined.push(v);
                        }
                    }
                    run_found.push((combined, start, scale_end_char));
                } else {
                    run_found.push((values, start, end));
                }
            }
        }

        let mut by_start: HashMap<usize, (usize, usize)> = HashMap::new();
        for &(_, start, end) in &run_found {
            by_start
                .entry(start)
                .and_modify(|e| {
                    if end > e.1 {
                        *e = (start, end);
                    }
                })
                .or_insert((start, end));
        }

        let mut whole: HashSet<(usize, usize)> = HashSet::new();
        let mut position = begin;
        loop {
            let longest = by_start.get(&position).copied();
            let target = match longest {
                None => position + 1,
                Some((_, end)) => end,
            };
            if let Some(span) = longest {
                whole.insert(span);
            }
            let following = starts.partition_point(|&s| s < target);
            if following >= starts.len() {
                break;
            }
            position = starts[following];
        }

        for (values, start, end) in run_found {
            let is_whole = whole.contains(&(start, end));
            found.push((values, start, end, is_whole));
        }
    }
    found
}

/// Extract numerals from `text` as `(candidate_values, char_start, char_end)` sorted by
/// `(char_start, -char_end)` (`model.py::read_numerals`).
pub fn read_numerals(text: &str, every: bool) -> Vec<(Vec<BigDecimal>, usize, usize)> {
    let mut found: Vec<(Vec<BigDecimal>, usize, usize)> = numeral_candidates(text)
        .into_iter()
        .filter(|(_, _, _, whole)| every || *whole)
        .map(|(vals, start, end, _)| (vals, start, end))
        .collect();
    found.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| b.2.cmp(&a.2)));
    found
}

/// Find character offset spans `(char_start, char_end)` in `text` whose numeric value
/// equals `|value|` (`model.py::numeral_spans`).
pub fn numeral_spans(text: &str, value: &BigDecimal, every: bool) -> Vec<(usize, usize)> {
    let target = value.abs();
    let candidates = numeral_candidates(text);
    let mut whole: Vec<(usize, usize)> = candidates
        .iter()
        .filter(|(vals, _, _, longest)| *longest && vals.iter().any(|v| v == &target))
        .map(|(_, start, end, _)| (*start, *end))
        .collect();
    if !whole.is_empty() || !every {
        whole.sort_unstable();
        return whole;
    }
    let mut any_set: HashSet<(usize, usize)> = HashSet::new();
    for (vals, start, end, _) in candidates {
        if vals.iter().any(|v| v == &target) {
            any_set.insert((start, end));
        }
    }
    let mut result: Vec<(usize, usize)> = any_set.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    result
}

/// Convenience wrapper for [`numeral_spans`] accepting a JSON value (`Value::Number`/`Value::String`).
pub fn numeral_spans_value(text: &str, value: &Value, every: bool) -> Vec<(usize, usize)> {
    let Some(dec) = decimal_util::from_json_value(value) else {
        return Vec::new();
    };
    numeral_spans(text, &dec, every)
}

// ============================================================================
// Prompt encoding & layout structures
// ============================================================================

#[derive(Debug, Clone)]
pub struct EncodedField {
    pub name: String,
    pub kind: String,
    pub query: usize,
    pub items: Vec<usize>,
    pub prefix: Option<(u8, usize, Vec<u8>)>,
    pub window: Option<(usize, usize)>,
    pub answer_slots: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Commit {
    pub text: String,
    pub done: bool,
    pub prefix: Option<(u8, usize, Vec<u8>)>,
    pub probability: f64,
}

#[derive(Debug, Clone, Default)]
pub struct EncodedMedia {
    pub ids: Vec<u32>,
    pub expansion: usize,
}

impl EncodedMedia {
    pub fn length(&self) -> usize {
        self.ids.len() + self.expansion
    }
}

#[derive(Debug, Clone)]
pub struct EncodedRecord {
    pub input_ids: Vec<u32>,
    pub input_start: usize,
    /// Unicode character offsets `(char_start, char_end)` for each token of `text`.
    pub offsets: Vec<(usize, usize)>,
    pub fields: Vec<EncodedField>,
    pub text: String,
    pub record: Value,
    pub media_token_offset: Option<usize>,
    pub image_expansion: usize,
}

#[derive(Debug, Clone)]
pub struct EncodeRecordOptions {
    pub max_length: usize,
    pub media: Option<EncodedMedia>,
    pub commits: HashMap<String, Commit>,
    pub answer_slots: bool,
    pub text_answer_slots: bool,
    pub template_variant: usize,
}

impl Default for EncodeRecordOptions {
    fn default() -> Self {
        Self {
            max_length: 8192,
            media: None,
            commits: HashMap::new(),
            answer_slots: false,
            text_answer_slots: false,
            template_variant: 0,
        }
    }
}

/// Tokenize trusted template text without adding special tokens (preserving control markers
/// such as `<|im_start|>` and `<|im_end|>`).
pub fn tokenize_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| Error::from_reason(format!("tokenizer encode failed: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

/// Tokenize untrusted user input splitting any literal special tokens (matching HF
/// `split_special_tokens=True`) and returning `(token_ids, char_offsets)`.
pub fn tokenize_input_with_char_offsets(
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| Error::from_reason(format!("tokenizer encode failed: {e}")))?;
    let added = tokenizer.get_added_vocabulary().get_added_tokens_decoder();
    let final_enc = if encoding.get_ids().iter().any(|id| added.contains_key(id)) {
        let mut literal = tokenizer.clone();
        literal
            .add_special_tokens(
                added
                    .values()
                    .filter(|t| !t.special)
                    .cloned()
                    .map(|t| t.special(true)),
            )
            .map_err(|e| Error::from_reason(format!("tokenizer add_special_tokens failed: {e}")))?;
        literal.set_encode_special_tokens(true);
        literal
            .encode(text, false)
            .map_err(|e| Error::from_reason(format!("tokenizer literal encode failed: {e}")))?
    } else {
        encoding
    };
    let b2c = schema::byte_to_char_map(text);
    let max_b = text.len();
    let offsets = final_enc
        .get_offsets()
        .iter()
        .map(|&(b0, b1)| (b2c[b0.min(max_b)], b2c[b1.min(max_b)]))
        .collect();
    Ok((final_enc.get_ids().to_vec(), offsets))
}

pub fn schema_line(name: &str, spec: &Value, complete_field: bool) -> String {
    format!("- {}\n", schema::field_line(name, spec, complete_field))
}

pub fn reference_line(record: &Value) -> String {
    match record.get("reference_date").and_then(Value::as_str) {
        Some(r) if !r.is_empty() => format!("Reference date: {r}\n"),
        _ => String::new(),
    }
}

pub fn evidence_construction(record: &Value, text_answer_slots: bool) -> bool {
    text_answer_slots
        || record
            .get("_meta")
            .and_then(|m| m.get("construction"))
            .and_then(Value::as_str)
            == Some(schema::EVIDENCE_CONSTRUCTION)
}

pub fn system_prefix(record: &Value, text_answer_slots: bool) -> &'static str {
    if evidence_construction(record, text_answer_slots) {
        EVIDENCE_SYSTEM_PREFIX
    } else {
        SYSTEM_PREFIX
    }
}

pub fn slot_token_id(tokenizer: &Tokenizer) -> Result<u32> {
    let ids = tokenize_ids(tokenizer, SLOT_TEXT)?;
    if ids.len() != 1 {
        return Err(invalid_request(format!(
            "answer slots need {SLOT_TEXT:?} to be one token"
        )));
    }
    Ok(ids[0])
}

pub fn slot_block(
    tokenizer: &Tokenizer,
    name: &str,
    spec: &Value,
    complete_field: bool,
    template_variant: usize,
) -> Result<Vec<u32>> {
    let kind = head_kind(spec);
    let Some(count) = slot_count_for_kind(&kind) else {
        return Ok(Vec::new());
    };
    let layout = prompt_layout(template_variant, complete_field)?;
    let cue = tokenize_ids(tokenizer, &format!("\n{name}:"))?;
    let mut ids = tokenize_ids(tokenizer, layout.slot_header)?;
    if complete_field {
        ids.extend_from_slice(&cue);
    } else {
        let take = cue.len().min(SLOT_CUE_TOKENS);
        ids.extend_from_slice(&cue[..take]);
    }
    let slot_id = slot_token_id(tokenizer)?;
    ids.resize(ids.len() + count, slot_id);
    Ok(ids)
}

pub fn query_block(
    tokenizer: &Tokenizer,
    name: &str,
    spec: &Value,
    commit: Option<&Commit>,
    complete_field: bool,
    template_variant: usize,
) -> Result<(Vec<u32>, usize, Vec<usize>)> {
    let layout = prompt_layout(template_variant, complete_field)?;
    let line = tokenize_ids(
        tokenizer,
        &format!("\n{}", schema::field_line(name, spec, complete_field)),
    )?;
    let mut ids = tokenize_ids(tokenizer, layout.query_header)?;
    if complete_field {
        ids.extend_from_slice(&line);
    } else {
        let take = line.len().min(QUERY_TOKENS);
        ids.extend_from_slice(&line[..take]);
    }
    ids.extend_from_slice(&tokenize_ids(tokenizer, layout.answer_cue)?);
    if let Some(c) = commit {
        ids.extend_from_slice(&tokenize_ids(tokenizer, &format!(" {}", c.text))?);
    }
    let query = ids
        .len()
        .checked_sub(1)
        .ok_or_else(|| invalid_request("empty query block"))?;
    let mut items = Vec::new();
    let kind = head_kind(spec);
    if LIST_KINDS.contains(&kind.as_str()) && commit.is_none() {
        for idx in 1..=LIST_SLOTS {
            let cue = format!("\n{idx}.");
            ids.extend_from_slice(&tokenize_ids(tokenizer, &cue)?);
            items.push(ids.len() - 1);
        }
    }
    Ok((ids, query, items))
}

pub fn clean_cut(text: &str, offsets: &[(usize, usize)], budget: usize, lookback: usize) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let lowest = budget.saturating_sub(lookback);
    for keep in (lowest + 1..=budget).rev() {
        let end = offsets[keep - 1].1;
        if end >= chars.len()
            || (end > 0 && chars[end].is_whitespace() && !chars[end - 1].is_whitespace())
        {
            return keep;
        }
    }
    for keep in (lowest + 1..=budget).rev() {
        let end = offsets[keep - 1].1;
        if end > 0
            && end < chars.len()
            && (chars[end].is_whitespace() || chars[end - 1].is_whitespace())
        {
            return keep;
        }
    }
    budget
}

pub fn char_span_to_tokens(
    starts: &[usize],
    ends: &[usize],
    char_start: usize,
    char_end: usize,
) -> Option<(usize, usize)> {
    let first = ends.partition_point(|&e| e <= char_start);
    let last_plus_one = starts.partition_point(|&s| s < char_end);
    if first >= starts.len() || last_plus_one == 0 || last_plus_one - 1 < first {
        return None;
    }
    Some((first, last_plus_one - 1))
}

pub fn span_base(text: &str, starts: &[usize], ends: &[usize], first: usize) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut base = starts[first];
    while base + 1 < ends[first] && base < chars.len() && chars[base].is_whitespace() {
        base += 1;
    }
    base
}

/// Apply token-boundary character trims (`model.py::trimmed_span`), returning Unicode char offsets.
pub fn trimmed_span(
    text: &str,
    offsets: &[(usize, usize)],
    first: usize,
    last: usize,
    trims: Option<&[[usize; 2]]>,
) -> (usize, usize) {
    let chars: Vec<char> = text.chars().collect();
    let mut char_start = offsets[first].0;
    let mut char_end = offsets[last].1;
    while char_start + 1 < offsets[first].1
        && char_start < chars.len()
        && chars[char_start].is_whitespace()
    {
        char_start += 1;
    }
    if let Some(tr) = trims {
        let start =
            char_start.max((char_start + tr[first][0]).min(offsets[first].1.saturating_sub(1)));
        let end = char_end.min(
            char_end
                .saturating_sub(tr[last][1])
                .max(offsets[last].0 + 1),
        );
        if start < end {
            char_start = start;
            char_end = end;
        }
    }
    while char_start + 1 < char_end && char_start < chars.len() && chars[char_start].is_whitespace()
    {
        char_start += 1;
    }
    (char_start, char_end)
}

/// Encode a typed-head record into token IDs, char offsets, query positions, list-item cues,
/// and optional answer slots (`model.py::encode_record`).
pub fn encode_record(
    tokenizer: &Tokenizer,
    record: &Value,
    opts: &EncodeRecordOptions,
) -> Result<Option<EncodedRecord>> {
    if opts.text_answer_slots && !opts.answer_slots {
        return Err(invalid_request("text_answer_slots needs answer_slots"));
    }
    let has_images = record
        .get("images")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty());
    if has_images && opts.media.is_none() {
        return Err(invalid_request(
            "records with images need their encoded media",
        ));
    }
    let default_media = EncodedMedia::default();
    let media = opts.media.as_ref().unwrap_or(&default_media);
    let fields_map = record
        .get("fields")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_request("record missing fields"))?;
    let items: Vec<(&String, &Value)> = fields_map.iter().collect();
    let complete_field = evidence_construction(record, opts.text_answer_slots);
    let layout = prompt_layout(opts.template_variant, complete_field)?;

    let mut ids = tokenize_ids(tokenizer, system_prefix(record, opts.text_answer_slots))?;
    for &(name, spec) in &items {
        ids.extend_from_slice(&tokenize_ids(tokenizer, layout.schema_header)?);
        ids.extend_from_slice(&tokenize_ids(
            tokenizer,
            &schema_line(name, spec, complete_field),
        )?);
    }
    ids.extend_from_slice(&tokenize_ids(tokenizer, &reference_line(record))?);
    ids.extend_from_slice(&tokenize_ids(tokenizer, USER_PREFIX)?);
    let media_start = ids.len();
    ids.extend_from_slice(&media.ids);
    let input_start = ids.len();
    let suffix = tokenize_ids(tokenizer, ASSISTANT_SUFFIX)?;

    let mut blocks = Vec::with_capacity(items.len());
    for &(name, spec) in &items {
        blocks.push(query_block(
            tokenizer,
            name,
            spec,
            opts.commits.get(name.as_str()),
            complete_field,
            opts.template_variant,
        )?);
    }

    let mut slot_blocks: HashMap<usize, Vec<u32>> = HashMap::new();
    if opts.answer_slots && (!media.ids.is_empty() || opts.text_answer_slots) {
        for (index, &(name, spec)) in items.iter().enumerate() {
            let is_done = opts.commits.get(name.as_str()).is_some_and(|c| c.done);
            if !is_done {
                let block =
                    slot_block(tokenizer, name, spec, complete_field, opts.template_variant)?;
                if !block.is_empty() {
                    slot_blocks.insert(index, block);
                }
            }
        }
    }

    let blocks_len: usize = blocks.iter().map(|b| b.0.len()).sum();
    let slots_len: usize = slot_blocks.values().map(Vec::len).sum();
    let overhead = ids.len() + media.expansion + suffix.len() + blocks_len + slots_len;
    if opts.max_length < overhead + MIN_INPUT_TOKENS {
        return Ok(None);
    }
    let budget = opts.max_length - overhead;

    let raw_input = record
        .get("input")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_request("record missing string input"))?;
    let mut text = raw_input.to_string();
    let (mut input_ids, mut offsets) = if !media.ids.is_empty() && text.trim().is_empty() {
        text.clear();
        (tokenize_ids(tokenizer, "\n")?, vec![(0, 0)])
    } else {
        tokenize_input_with_char_offsets(tokenizer, &text)?
    };

    if input_ids.len() > budget {
        let keep = clean_cut(&text, &offsets, budget, 32);
        let cut_chars = offsets[keep - 1].1;
        text = text.chars().take(cut_chars).collect();
        (input_ids, offsets) = tokenize_input_with_char_offsets(tokenizer, &text)?;
        if input_ids.len() > budget {
            input_ids.truncate(budget);
            offsets.truncate(budget);
            if let Some(&(_, last_char)) = offsets.last() {
                text = text.chars().take(last_char).collect();
            }
        }
    }
    if input_ids.is_empty() {
        return Ok(None);
    }

    ids.extend_from_slice(&input_ids);
    ids.extend_from_slice(&suffix);

    let done: HashSet<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, (name, _))| opts.commits.get(name.as_str()).is_some_and(|c| c.done))
        .map(|(idx, _)| idx)
        .collect();
    let mut order: Vec<usize> = (0..items.len()).collect();
    if !done.is_empty() {
        order = (0..items.len())
            .filter(|i| done.contains(i))
            .chain((0..items.len()).filter(|i| !done.contains(i)))
            .collect();
    }

    let mut queries: HashMap<usize, usize> = HashMap::with_capacity(items.len());
    let mut list_slots: HashMap<usize, Vec<usize>> = HashMap::with_capacity(items.len());
    for &index in &order {
        let (ref block, query_off, ref item_offs) = blocks[index];
        let start = ids.len();
        ids.extend_from_slice(block);
        queries.insert(index, start + query_off);
        list_slots.insert(index, item_offs.iter().map(|&o| start + o).collect());
    }

    let mut answer_positions: HashMap<usize, Vec<usize>> = HashMap::new();
    for &index in &order {
        if let Some(sblock) = slot_blocks.get(&index) {
            ids.extend_from_slice(sblock);
            let kind = head_kind(items[index].1);
            if let Some(count) = slot_count_for_kind(&kind) {
                let end = ids.len();
                answer_positions.insert(index, (end - count..end).collect());
            }
        }
    }

    let mut fields: Vec<EncodedField> = Vec::new();
    for (index, &(name, spec)) in items.iter().enumerate() {
        if done.contains(&index) {
            continue;
        }
        let kind = head_kind(spec);
        let mut encoded = EncodedField {
            name: name.clone(),
            kind: kind.clone(),
            query: queries[&index],
            items: list_slots.remove(&index).unwrap_or_default(),
            prefix: None,
            window: None,
            answer_slots: answer_positions.remove(&index).unwrap_or_default(),
        };
        if let Some(commit) = opts.commits.get(name.as_str())
            && let Some(ref prefix) = commit.prefix
            && matches!(encoded.kind.as_str(), "number" | "estimate")
        {
            encoded.prefix = Some(prefix.clone());
            if encoded.kind == "estimate" {
                encoded.window = prefix_window(spec, prefix, BINS)?;
            }
        }
        fields.push(encoded);
    }
    if fields.is_empty() {
        return Ok(None);
    }

    Ok(Some(EncodedRecord {
        input_ids: ids,
        input_start,
        offsets,
        fields,
        text,
        record: record.clone(),
        media_token_offset: if opts.media.is_some() {
            Some(media_start)
        } else {
            None
        },
        image_expansion: media.expansion,
    }))
}

// ============================================================================
// Prediction layout & PredictionViews / AnswerSlotLayout builders
// ============================================================================

/// Packed answer-slot routing inputs ready to pass directly into
/// [`head::ClairicHead::answer_slot_states`].
#[derive(Debug, Clone, Default)]
pub struct AnswerSlotLayout {
    /// Indices into `EncodedRecord::fields` that carry answer slots.
    pub slot_fields: Vec<usize>,
    /// Query token positions (`[N_slot]`).
    pub field_queries: Vec<i64>,
    /// Route kind indices into [`ROUTE_KINDS`] (`[N_slot]`).
    pub slot_kinds: Vec<i64>,
    /// Right-padded slot token positions (`[N_slot, max_slot_len]`).
    pub slot_positions: Vec<Vec<i64>>,
    /// Unpadded slot lengths (`[N_slot]`).
    pub slot_lengths: Vec<i64>,
}

impl AnswerSlotLayout {
    pub fn is_empty(&self) -> bool {
        self.slot_fields.is_empty()
    }
}

/// Parsed fields of a serialized prediction layout (`model.py::parse_prediction_layout`).
#[derive(Debug, Clone)]
pub struct ParsedPredictionLayout {
    pub input_start: usize,
    pub input_length: usize,
    pub kinds: Vec<String>,
    pub queries: Vec<usize>,
    pub items: Vec<Vec<usize>>,
    pub slot_fields: Vec<usize>,
    pub answer_slots: Vec<Vec<usize>>,
}

/// Serialize `record` into the flat integer prediction layout (`model.py::prediction_layout`).
pub fn prediction_layout(record: &EncodedRecord) -> Result<Vec<usize>> {
    let mut layout = vec![
        record.input_ids.len(),
        record.input_start,
        record.offsets.len(),
        record.fields.len(),
    ];
    for item in &record.fields {
        let kind_idx = FIELD_KINDS
            .iter()
            .position(|&k| k == item.kind)
            .ok_or_else(|| invalid_request(format!("unknown field kind {:?}", item.kind)))?;
        layout.push(kind_idx);
        layout.push(item.query);
        layout.push(item.items.len());
        layout.extend_from_slice(&item.items);
    }
    let slotted: Vec<(usize, &[usize])> = record
        .fields
        .iter()
        .enumerate()
        .filter(|(_, item)| !item.answer_slots.is_empty())
        .map(|(idx, item)| (idx, item.answer_slots.as_slice()))
        .collect();
    if !slotted.is_empty() {
        layout.push(slotted.len());
        for (idx, slots) in slotted {
            layout.push(idx);
            layout.push(slots.len());
            layout.extend_from_slice(slots);
        }
    }
    Ok(layout)
}

/// Validate and shift a serialized prediction layout for a hidden-state sequence of length
/// `num_tokens` (`model.py::parse_prediction_layout`).
pub fn parse_prediction_layout(
    layout: &[usize],
    num_tokens: usize,
    with_slots: bool,
) -> Result<ParsedPredictionLayout> {
    if layout.len() < 4 {
        return Err(invalid_request(
            "prediction_layout must have at least 4 ints",
        ));
    }
    let (prompt_length, input_start, input_length, count) =
        (layout[0], layout[1], layout[2], layout[3]);
    if num_tokens < prompt_length
        || count < 1
        || input_length < 1
        || input_start + input_length > prompt_length
    {
        return Err(invalid_request(
            "prediction_layout does not match the prompt",
        ));
    }
    let shift = num_tokens - prompt_length;
    let mut kinds = Vec::with_capacity(count);
    let mut queries = Vec::with_capacity(count);
    let mut items = Vec::with_capacity(count);
    let mut cursor = 4usize;

    for _ in 0..count {
        if cursor + 3 > layout.len() {
            return Err(invalid_request("prediction_layout is truncated"));
        }
        let (kind_idx, query, size) = (layout[cursor], layout[cursor + 1], layout[cursor + 2]);
        cursor += 3;
        if kind_idx >= FIELD_KINDS.len()
            || query < input_start + input_length
            || query >= prompt_length
        {
            return Err(invalid_request("prediction_layout has an invalid field"));
        }
        let kind_str = FIELD_KINDS[kind_idx];
        let expected = if LIST_KINDS.contains(&kind_str) {
            LIST_SLOTS
        } else {
            0
        };
        if cursor + size > layout.len() {
            return Err(invalid_request("prediction_layout is truncated"));
        }
        let slots = &layout[cursor..cursor + size];
        cursor += size;
        if size != expected || slots.iter().any(|&s| s <= query || s >= prompt_length) {
            return Err(invalid_request(
                "prediction_layout has invalid item positions",
            ));
        }
        kinds.push(kind_str.to_string());
        queries.push(query + shift);
        items.push(slots.iter().map(|&s| s + shift).collect());
    }

    let mut slot_fields = Vec::new();
    let mut answer_slots = Vec::new();
    if cursor < layout.len() {
        let slotted = layout[cursor];
        cursor += 1;
        if slotted == 0 || slotted > count {
            return Err(invalid_request(
                "prediction_layout has invalid answer slots",
            ));
        }
        for _ in 0..slotted {
            if cursor + 2 > layout.len() {
                return Err(invalid_request("prediction_layout is truncated"));
            }
            let (index, size) = (layout[cursor], layout[cursor + 1]);
            cursor += 2;
            if cursor + size > layout.len() {
                return Err(invalid_request("prediction_layout is truncated"));
            }
            let positions = &layout[cursor..cursor + size];
            cursor += size;
            let monotonic = slot_fields.last().is_none_or(|&prev| index > prev);
            let orig_query = queries[index] - shift;
            if index >= count
                || !monotonic
                || size < 1
                || positions
                    .iter()
                    .any(|&p| p <= orig_query || p >= prompt_length)
            {
                return Err(invalid_request(
                    "prediction_layout has invalid answer slots",
                ));
            }
            slot_fields.push(index);
            answer_slots.push(positions.iter().map(|&p| p + shift).collect());
        }
    }
    if cursor != layout.len() {
        return Err(invalid_request("prediction_layout has trailing values"));
    }

    Ok(ParsedPredictionLayout {
        input_start: input_start + shift,
        input_length,
        kinds,
        queries,
        items,
        slot_fields: if with_slots { slot_fields } else { Vec::new() },
        answer_slots: if with_slots { answer_slots } else { Vec::new() },
    })
}

/// Build `(input_start, stop, PredictionViews, AnswerSlotLayout)` from an [`EncodedRecord`]
/// for direct consumption by [`head::ClairicHead::forward`] and
/// [`head::ClairicHead::answer_slot_states`].
pub fn build_prediction_views(
    record: &EncodedRecord,
) -> Result<(usize, usize, PredictionViews, AnswerSlotLayout)> {
    build_prediction_views_for_len(record, record.input_ids.len() + record.image_expansion)
}

/// Variant of [`build_prediction_views`] accepting an explicit backbone hidden sequence length
/// `num_tokens` (used when vision embeddings expand the token count).
pub fn build_prediction_views_for_len(
    record: &EncodedRecord,
    num_tokens: usize,
) -> Result<(usize, usize, PredictionViews, AnswerSlotLayout)> {
    let layout = prediction_layout(record)?;
    let parsed = parse_prediction_layout(&layout, num_tokens, true)?;
    let stop = parsed.input_start + parsed.input_length;

    let mut views = PredictionViews {
        query_positions: parsed.queries.iter().map(|&q| q as i64).collect(),
        ..PredictionViews::default()
    };
    for (row, kind) in parsed.kinds.iter().enumerate() {
        let r = row as i64;
        match kind.as_str() {
            "text" => views.text_rows.push(r),
            "text_list" => views.tags_rows.push(r),
            "number" => views.number_rows.push(r),
            "date" => views.date_rows.push(r),
            "time" => views.time_rows.push(r),
            "datetime" => {
                views.date_rows.push(r);
                views.time_rows.push(r);
            }
            "estimate" => views.estimate_rows.push(r),
            "number_list" => {
                views.number_list_rows.push(r);
                views
                    .number_list_items
                    .push(parsed.items[row].iter().map(|&p| p as i64).collect());
            }
            "date_list" => {
                views.date_list_rows.push(r);
                views
                    .date_list_items
                    .push(parsed.items[row].iter().map(|&p| p as i64).collect());
            }
            other => {
                return Err(invalid_request(format!(
                    "unsupported field kind {other:?} in prediction views"
                )));
            }
        }
    }

    let mut as_layout = AnswerSlotLayout::default();
    if !parsed.answer_slots.is_empty() {
        let max_width = parsed.answer_slots.iter().map(Vec::len).max().unwrap_or(0);
        for (&field_idx, slots) in parsed.slot_fields.iter().zip(parsed.answer_slots.iter()) {
            let route_name = if parsed.kinds[field_idx] == "text_list" {
                "tags"
            } else {
                parsed.kinds[field_idx].as_str()
            };
            let route_idx = ROUTE_KINDS
                .iter()
                .position(|&k| k == route_name)
                .ok_or_else(|| {
                    invalid_request(format!(
                        "answer slot field has unknown route kind {route_name:?}"
                    ))
                })?;
            let last_pos = *slots
                .last()
                .ok_or_else(|| invalid_request("answer slot list cannot be empty"))?
                as i64;
            let mut padded: Vec<i64> = slots.iter().map(|&p| p as i64).collect();
            padded.resize(max_width, last_pos);
            as_layout.slot_fields.push(field_idx);
            as_layout
                .field_queries
                .push(views.query_positions[field_idx]);
            as_layout.slot_kinds.push(route_idx as i64);
            as_layout.slot_positions.push(padded);
            as_layout.slot_lengths.push(slots.len() as i64);
        }
    }

    Ok((parsed.input_start, stop, views, as_layout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn clef_tokenizer() -> Tokenizer {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../__test__/fixtures/clef/tokenizer.json");
        Tokenizer::from_file(&path).expect("load clef fixture tokenizer")
    }

    #[test]
    fn numeral_convention_and_values_us_vs_eu() {
        assert_eq!(
            numeral_convention("Subtotal 1,234.50 and Total 9,876.54"),
            Some("us")
        );
        assert_eq!(
            numeral_convention("Zwischensumme 1.234,50 und Gesamtbetrag 9.876,54"),
            Some("eu")
        );
        assert_eq!(
            numeral_convention("Rechnung Nr. 42, Betrag 1.200"),
            Some("eu")
        );
        assert_eq!(numeral_convention("Factura México CFDI MXN 1.200"), None);
        assert_eq!(
            numeral_convention("Uses European number notation: 1.234"),
            Some("eu")
        );

        let ambiguous = "1.234";
        assert_eq!(numeral_values(ambiguous, None).len(), 2);
        assert_eq!(
            numeral_values(ambiguous, Some("us")),
            vec![decimal_util::parse_decimal("1.234").unwrap()]
        );
        assert_eq!(
            numeral_values(ambiguous, Some("eu")),
            vec![decimal_util::parse_decimal("1234").unwrap()]
        );

        let scaled = read_numerals("Revenue was $2.5 million in Q4.", false);
        assert!(
            scaled[0]
                .0
                .contains(&decimal_util::parse_decimal("2500000").unwrap())
        );

        let spans = numeral_spans(
            "Total: €1.234,56 (net 1.000,00)",
            &decimal_util::parse_decimal("1234.56").unwrap(),
            false,
        );
        assert_eq!(spans, vec![(8, 16)]);
    }

    #[test]
    fn encode_record_with_clef_tokenizer_and_prediction_views() {
        let tok = clef_tokenizer();
        let record = json!({
            "input": "Invoice #42 dated 2025-05-01 at 14:30:00 for Acme Corp. Total: $123.45.",
            "fields": {
                "vendor": {"type": "text", "instructions": "Company name"},
                "total": {"type": "number", "decimals": 2},
                "date": {"type": "date"},
                "clock": {"type": "time"},
                "eta": {"type": "estimate", "min": 0, "max": 500},
                "amounts": {"type": "list", "items": "number"},
                "dates": {"type": "list", "items": "date"},
                "tags": {"type": "list", "items": "text"}
            },
            "_meta": {"construction": "evidence-v1"}
        });
        let opts = EncodeRecordOptions {
            max_length: 4096,
            answer_slots: true,
            text_answer_slots: true,
            ..EncodeRecordOptions::default()
        };
        let encoded = encode_record(&tok, &record, &opts)
            .unwrap()
            .expect("record should fit");
        assert_eq!(encoded.text, record["input"].as_str().unwrap());
        assert_eq!(encoded.fields.len(), 8);
        assert_eq!(encoded.fields[0].answer_slots.len(), 80); // text
        assert_eq!(encoded.fields[1].answer_slots.len(), 32); // number
        assert_eq!(encoded.fields[7].answer_slots.len(), 64); // text_list
        assert_eq!(encoded.fields[5].items.len(), LIST_SLOTS);
        assert_eq!(encoded.fields[6].items.len(), LIST_SLOTS);

        let (start, stop, views, as_layout) = build_prediction_views(&encoded).unwrap();
        assert_eq!(start, encoded.input_start);
        assert_eq!(stop, start + encoded.offsets.len());
        assert_eq!(views.query_positions.len(), 8);
        assert_eq!(views.text_rows, vec![0]);
        assert_eq!(views.number_rows, vec![1]);
        assert_eq!(views.date_rows, vec![2]);
        assert_eq!(views.time_rows, vec![3]);
        assert_eq!(views.estimate_rows, vec![4]);
        assert_eq!(views.number_list_rows, vec![5]);
        assert_eq!(views.date_list_rows, vec![6]);
        assert_eq!(views.tags_rows, vec![7]);
        assert_eq!(views.number_list_items.len(), 1);
        assert_eq!(views.number_list_items[0].len(), LIST_SLOTS);
        assert_eq!(as_layout.slot_fields, vec![0, 1, 7]);
        assert_eq!(as_layout.slot_lengths, vec![80, 32, 64]);
        assert_eq!(as_layout.slot_positions[0].len(), 80);
        assert_eq!(as_layout.slot_positions[1].len(), 80);
    }
}
