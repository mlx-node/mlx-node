//! Source text/number grounding, whitespace normalization, and grammar constraint builder.
//!
//! Direct port of upstream `clairic_runtime/grounding.py`, emitting:
//! - [`GrammarSpec::GroundedText`] for text suffix-automaton grounding (`SuffixAutomatonMasker`)
//! - [`GrammarSpec::ClairicEbnf`] for grounded number literal unions
//! - [`GrammarSpec::JsonSchema`] for grounded enums and `> MAX_GRAMMAR_CHARACTERS` / regex fallback

use super::{
    ar, decimal_util,
    errors::invalid_request,
    json_validate, schema,
    typed_prompt::{numeral_convention, numeral_run_regex, numeral_values},
};
use crate::errors::Result;
use crate::grammar::{GrammarSpec, suffix_masker::build_clairic_ebnf_for_text};
use bigdecimal::{BigDecimal, Zero, num_traits::ToPrimitive};
use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

pub const MAX_GRAMMAR_CHARACTERS: usize = 4096;

const CURRENCY_CODES: &[&str] = &[
    "USD", "EUR", "GBP", "JPY", "CNY", "INR", "AUD", "CAD", "CHF", "HKD", "NZD", "SGD", "SEK",
    "NOK", "DKK", "KRW", "BRL", "MXN", "ZAR",
];

fn state_only_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:from (?:the )?(?:accompanying (?:state )?|state )text only|supplied in the accompanying state text)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn null_absence_guard_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(
            r"(?i)(?:^|(?<=[.;\n]))\s*return\s+null\s+(?:only\s+)?if\s+(?:not\s+(?:explicitly\s+)?printed|absent|missing)\s*(?:[.;]|$)",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn excluded_source_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:not|ignore|exclude|excluding)\b[^.;:\n]*$")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn document_only_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let doc = r"(?:(?:this|the)(?:\s+(?:supplied|provided))?|supplied|provided)\s+(?:note|document|text)";
        let pat = format!(r"(?i)\b(?:use\s+(?:only\s+{doc}|{doc}\s+only)|from\s+{doc}\s+only)\b");
        Regex::new(&pat).unwrap_or_else(|_| unreachable!())
    })
}

fn document_excluded_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:not|never|don't|ignore|exclude|excluding)\b[^.;:\n]*$")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn derived_number_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:calculat\w*|comput\w*|deriv\w*|convert\w*|conversion\w*|count\w*|sum\w*|averag\w*|mean|median|difference|ratio|growth|product|multipli\w*|multiply|subtract\w*|minus|divid\w*|division|quotient|remainder|add|adding|addition|plus|times|percent\w*|proportion|fraction)\b|\b(?:how many|how much|total number of|number of (?:line )?(?:items|rows|entries|records))\b|[+*/\u{00d7}\u{00f7}=]|\s[-\u{2212}]\s",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn exact_text_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:verbatim|raw|literal\w*|print\w*|exact\w*|characters?|digits?|whitespace|spacing|newlines?)\b|\b(?:line\s+breaks?|white\s+space|do\s+not\s+normaliz\w*)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn conditional_scope_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:if|unless|compare|combine)\b").unwrap_or_else(|_| unreachable!())
    })
}

fn explicit_media_scope_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:page|image)\s+\d+\b").unwrap_or_else(|_| unreachable!())
    })
}

fn numeral_split_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[^, \u{00a0}\u{202f}\u{2009}\u{2007}]+").unwrap_or_else(|_| unreachable!())
    })
}

fn exponent_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\A[eE]([-+]?\d+)").unwrap_or_else(|_| unreachable!()))
}

fn unicode_currency_symbol_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\A\p{Sc}\z").unwrap_or_else(|_| unreachable!()))
}

fn is_currency_symbol(ch: char) -> bool {
    if ch.is_ascii() {
        return ch == '$';
    }
    let mut buf = [0u8; 4];
    unicode_currency_symbol_regex().is_match(ch.encode_utf8(&mut buf))
}

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

fn canonical_decimal(d: &BigDecimal) -> BigDecimal {
    if d.is_zero() {
        BigDecimal::from(0)
    } else {
        d.normalized()
    }
}

/// Return `true` iff `spec` allows whitespace-normalized text grounding
/// (`grounding.py::allows_whitespace_normalization`).
pub fn allows_whitespace_normalization(spec: &Value) -> bool {
    let Some(obj) = spec.as_object() else {
        return false;
    };
    if obj.get("type").and_then(Value::as_str) != Some("text")
        || is_truthy(obj.get("knowledge"))
        || obj.contains_key("source_only")
        || is_truthy(obj.get("_physical_row_scope"))
    {
        return false;
    }
    for key in ["enum", "pattern", "minLength", "maxLength"] {
        if obj.contains_key(key) {
            return false;
        }
    }
    let instructions = obj
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    !exact_text_regex().is_match(instructions)
}

/// Collapse runs of Unicode whitespace in `source` to single ASCII spaces, returning
/// `(normalized_string, char_offset_ranges)` (`grounding.py::collapse_whitespace`).
pub fn collapse_whitespace(source: &str) -> (String, Vec<(usize, usize)>) {
    let chars: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len());
    let mut offsets: Vec<(usize, usize)> = Vec::with_capacity(chars.len());
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            let start = i;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            out.push(' ');
            offsets.push((start, i));
        } else {
            out.push(chars[i]);
            offsets.push((i, i + 1));
            i += 1;
        }
    }
    (out, offsets)
}

/// Return `true` iff `spec` restricts extraction to accompanying state text only
/// (`grounding.py::state_only`).
pub fn state_only(spec: &Value) -> bool {
    let Some(obj) = spec.as_object() else {
        return false;
    };
    let kind = obj.get("type").and_then(Value::as_str).unwrap_or("");
    if !matches!(kind, "text" | "number") || is_truthy(obj.get("knowledge")) {
        return false;
    }
    let instructions = obj
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let scope = null_absence_guard_regex().replace_all(instructions, "");
    if conditional_scope_regex().is_match(&scope) {
        return false;
    }
    state_only_regex()
        .find_iter(instructions)
        .any(|m| !excluded_source_regex().is_match(&instructions[..m.start()]))
}

/// Return `true` iff `spec` explicitly scopes extraction to a numbered page or image
/// (`grounding.py::explicit_media_scope`).
pub fn explicit_media_scope(spec: &Value) -> bool {
    let Some(obj) = spec.as_object() else {
        return false;
    };
    let kind = obj.get("type").and_then(Value::as_str).unwrap_or("");
    let instructions = obj
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    matches!(kind, "text" | "number")
        && !is_truthy(obj.get("knowledge"))
        && explicit_media_scope_regex().is_match(instructions)
}

/// Return `true` iff `spec` restricts extraction to the supplied note/document/text only
/// (`grounding.py::document_only`).
pub fn document_only(spec: &Value) -> bool {
    let Some(obj) = spec.as_object() else {
        return false;
    };
    let kind = obj.get("type").and_then(Value::as_str).unwrap_or("");
    if !matches!(kind, "text" | "number") || is_truthy(obj.get("knowledge")) {
        return false;
    }
    let instructions = obj
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let scope = null_absence_guard_regex().replace_all(instructions, "");
    if conditional_scope_regex().is_match(&scope) {
        return false;
    }
    if kind == "number" && derived_number_regex().is_match(&scope) {
        return false;
    }
    document_only_regex()
        .find_iter(instructions)
        .any(|m| !document_excluded_regex().is_match(&instructions[..m.start()]))
}

/// Return `true` iff source grounding applies to `spec` given whether media is attached
/// (`grounding.py::applies`).
pub fn applies(spec: &Value, has_media: bool) -> bool {
    if has_media {
        return false;
    }
    let Some(obj) = spec.as_object() else {
        return false;
    };
    let kind = obj.get("type").and_then(Value::as_str).unwrap_or("");
    matches!(kind, "text" | "number")
        && !is_truthy(obj.get("knowledge"))
        && (kind == "text"
            || is_truthy(obj.get("source_only"))
            || state_only(spec)
            || document_only(spec))
}

fn document_number_span(chars: &[char], span: (usize, usize)) -> bool {
    let (start, _) = span;
    let mut left = start;
    while left > 0
        && (chars[left - 1].is_alphanumeric() || matches!(chars[left - 1], '_' | '.' | '/' | '-'))
    {
        left -= 1;
    }
    let mut prefix_start = left;
    while prefix_start < start && matches!(chars[prefix_start], '_' | '.' | '/' | '-') {
        prefix_start += 1;
    }
    if prefix_start >= start {
        return true;
    }
    if !chars[prefix_start].is_alphabetic() {
        return true;
    }
    let prefix: String = chars[prefix_start..start].iter().collect();
    let upper = prefix.to_ascii_uppercase();
    CURRENCY_CODES.contains(&upper.as_str())
}

/// Build the Clairic EBNF text grammar string directly (`grounding.py::text_grammar`).
pub fn text_grammar(
    source: &str,
    nullable: bool,
    normalized_source: Option<&str>,
) -> Result<String> {
    if source.is_empty() {
        if !nullable {
            return Err(invalid_request(
                "Empty source permits no non-null text value",
            ));
        }
        return Ok(r#"root ::= "null""#.to_string());
    }
    Ok(build_clairic_ebnf_for_text(
        source,
        normalized_source,
        nullable,
    ))
}

/// Extract all grounded numeric values and their Unicode char-offset spans from `source`
/// (`grounding.py::numbers`), preserving first-seen value order.
pub fn numbers(source: &str) -> Vec<(BigDecimal, Vec<(usize, usize)>)> {
    if source.is_empty() {
        return Vec::new();
    }
    let convention = numeral_convention(source);
    let chars: Vec<char> = source.chars().collect();
    let b2c = schema::byte_to_char_map(source);

    let mut index_by_value: HashMap<BigDecimal, usize> = HashMap::new();
    let mut ordered: Vec<(BigDecimal, Vec<(usize, usize)>)> = Vec::new();
    let mut consumed_end_byte = 0usize;
    let mut search_pos = 0usize;

    while search_pos < source.len() {
        let Ok(Some(run)) = numeral_run_regex().find_from_pos(source, search_pos) else {
            break;
        };
        let run_start_byte = run.start();
        let run_end_byte = run.end();
        search_pos = run_end_byte.max(search_pos + 1);

        if run_start_byte < consumed_end_byte {
            continue;
        }
        let run_str = run.as_str();
        let readings: Vec<(&str, usize, usize, usize)> =
            if !numeral_values(run_str, convention).is_empty() {
                vec![(
                    run_str,
                    b2c[run_start_byte],
                    b2c[run_end_byte],
                    run_end_byte,
                )]
            } else {
                numeral_split_regex()
                    .find_iter(run_str)
                    .map(|m| {
                        let abs_s = run_start_byte + m.start();
                        let abs_e = run_start_byte + m.end();
                        (m.as_str(), b2c[abs_s], b2c[abs_e], abs_e)
                    })
                    .collect()
            };

        for (printed, mut left, mut end, end_byte) in readings {
            let mut values = numeral_values(printed, convention);
            if values.is_empty() {
                continue;
            }
            if let Some(exp_caps) = exponent_regex().captures(&source[end_byte..]) {
                let full_len = exp_caps.get(0).map(|m| m.len()).unwrap_or(0);
                consumed_end_byte = end_byte + full_len;
                let exp_str = exp_caps.get(1).map(|m| m.as_str()).unwrap_or("");
                if exp_str.len() > 4 {
                    continue;
                }
                let Ok(exp_val) = exp_str.parse::<i64>() else {
                    continue;
                };
                if exp_val.abs() > 308 {
                    continue;
                }
                values = values
                    .into_iter()
                    .map(|v| decimal_util::scaleb(&v, exp_val))
                    .collect();
                end = b2c[end_byte + full_len];
            }

            let mut start = left;
            let min_start = left.saturating_sub(32);
            while start > min_start
                && (matches!(chars[start - 1], ' ' | '\t' | '-' | '+' | '\u{2212}')
                    || is_currency_symbol(chars[start - 1]))
            {
                start -= 1;
            }

            let signed_slice = &chars[start..left];
            let mut ltrim = 0usize;
            while ltrim < signed_slice.len() && signed_slice[ltrim].is_whitespace() {
                ltrim += 1;
            }
            let mut rtrim = signed_slice.len();
            while rtrim > ltrim && signed_slice[rtrim - 1].is_whitespace() {
                rtrim -= 1;
            }
            let signed = &signed_slice[ltrim..rtrim];
            let mut negative = signed.iter().any(|&c| c == '-' || c == '\u{2212}')
                && (start == 0 || !chars[start - 1].is_alphanumeric());
            left = start + ltrim;

            let before_start = left.saturating_sub(32);
            let before = &chars[before_start..left];
            let after_end = (end + 64).min(chars.len());
            let after = &chars[end..after_end];

            let mut prefix_len = before.len();
            while prefix_len > 0 && before[prefix_len - 1].is_whitespace() {
                prefix_len -= 1;
            }
            let prefix = &before[..prefix_len];

            let mut suffix_skip = 0usize;
            while suffix_skip < after.len() && after[suffix_skip].is_whitespace() {
                suffix_skip += 1;
            }
            let suffix = &after[suffix_skip..];

            let printed_or_signed_has_marker = printed
                .chars()
                .chain(signed.iter().copied())
                .any(|c| c == '.' || c == ',' || is_currency_symbol(c));
            let line_rest_blank = if suffix.len() <= 1 {
                true
            } else {
                suffix[1..]
                    .iter()
                    .take_while(|&&c| c != '\n')
                    .all(|c| c.is_whitespace())
            };
            let accounting = printed_or_signed_has_marker || line_rest_blank;

            if prefix.last() == Some(&'(') && suffix.first() == Some(&')') && accounting {
                negative = true;
                left -= (before.len() - prefix.len()) + 1;
                end += suffix_skip + 1;
            }

            for mut value in values {
                if negative {
                    value = -value;
                }
                let roundtrip_ok = if decimal_util::is_integral(&value) {
                    true
                } else {
                    value
                        .to_f64()
                        .filter(|f| f.is_finite())
                        .and_then(decimal_util::from_f64)
                        .is_some_and(|d| d == value)
                };
                if !roundtrip_ok {
                    continue;
                }
                let key = canonical_decimal(&value);
                if let Some(&idx) = index_by_value.get(&key) {
                    ordered[idx].1.push((left, end));
                } else {
                    let idx = ordered.len();
                    index_by_value.insert(key.clone(), idx);
                    ordered.push((key, vec![(left, end)]));
                }
            }
        }
    }

    ordered
}

/// Convert a grounded `BigDecimal` into `(json_value, ebnf_literal)` matching Python's
/// `int(v) if v == v.to_integral_value() else float(v)` and `str(Decimal(str(value)))`.
fn grounded_decimal_to_json_and_literal(value: &BigDecimal) -> Option<(Value, String)> {
    if value.is_zero() {
        return Some((Value::Number(serde_json::Number::from(0)), "0".to_string()));
    }
    if decimal_util::is_integral(value) {
        let plain = decimal_util::to_stripped_plain_string(value);
        let json_val = decimal_util::to_json_number(value)?;
        return Some((json_val, plain));
    }
    let f = value.to_f64().filter(|f| f.is_finite())?;
    let clean = if f == 0.0 { 0.0 } else { f };
    let num = serde_json::Number::from_f64(clean)?;
    let lit = num.to_string();
    Some((Value::Number(num), lit))
}

fn decimal_ebnf_literal_from_json(value: &Value) -> Option<String> {
    if value.is_null() {
        return Some("null".to_string());
    }
    let dec = decimal_util::from_json_value(value)?;
    if decimal_util::is_integral(&dec) {
        Some(decimal_util::to_stripped_plain_string(&dec))
    } else {
        value.as_f64().filter(|f| f.is_finite()).map(|f| {
            let clean = if f == 0.0 { 0.0 } else { f };
            serde_json::Number::from_f64(clean)
                .map(|n| n.to_string())
                .unwrap_or_else(|| clean.to_string())
        })
    }
}

/// Source text & numeral provenance index and grammar-constraint compiler
/// (`grounding.py::Source`).
pub struct Source {
    pub text: String,
    pub normalize_whitespace: bool,
    chars: Vec<char>,
    b2c: Vec<usize>,
    numeral_order: Vec<BigDecimal>,
    numerals: HashMap<BigDecimal, Vec<(usize, usize)>>,
    normalized_source: OnceLock<(String, Vec<(usize, usize)>)>,
    constraints: Mutex<HashMap<(String, bool, bool), GrammarSpec>>,
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source")
            .field("text_len", &self.text.len())
            .field("normalize_whitespace", &self.normalize_whitespace)
            .field("numeral_count", &self.numeral_order.len())
            .finish()
    }
}

impl Source {
    pub fn new(text: &str, normalize_whitespace: bool) -> Self {
        let extracted = numbers(text);
        let mut numeral_order = Vec::with_capacity(extracted.len());
        let mut numerals = HashMap::with_capacity(extracted.len());
        for (val, spans) in extracted {
            numeral_order.push(val.clone());
            numerals.insert(val, spans);
        }
        Self {
            text: text.to_string(),
            normalize_whitespace,
            chars: text.chars().collect(),
            b2c: schema::byte_to_char_map(text),
            numeral_order,
            numerals,
            normalized_source: OnceLock::new(),
            constraints: Mutex::new(HashMap::new()),
        }
    }

    pub fn numerals(&self) -> &HashMap<BigDecimal, Vec<(usize, usize)>> {
        &self.numerals
    }

    pub fn whitespace_normalization(&self, spec: &Value) -> bool {
        self.normalize_whitespace && allows_whitespace_normalization(spec)
    }

    fn whitespace_source(&self) -> &(String, Vec<(usize, usize)>) {
        self.normalized_source
            .get_or_init(|| collapse_whitespace(&self.text))
    }

    /// Find all raw + whitespace-collapsed char-offset occurrences of `value`
    /// (`grounding.py::Source.normalized_span`).
    pub fn normalized_span(&self, value: &str) -> (bool, Option<[usize; 2]>) {
        if value.is_empty() {
            return (false, None);
        }
        let val_char_len = value.chars().count();
        let mut matches: HashSet<(usize, usize)> = HashSet::new();

        let mut search_byte = 0usize;
        while search_byte <= self.text.len() {
            let Some(rel) = self.text[search_byte..].find(value) else {
                break;
            };
            let byte_start = search_byte + rel;
            let char_start = self.b2c[byte_start];
            matches.insert((char_start, char_start + val_char_len));
            let Some(ch) = self.text[byte_start..].chars().next() else {
                break;
            };
            search_byte = byte_start + ch.len_utf8();
        }

        let (normalized, offsets) = self.whitespace_source();
        let norm_b2c = schema::byte_to_char_map(normalized);
        let mut norm_byte = 0usize;
        while norm_byte <= normalized.len() {
            let Some(rel) = normalized[norm_byte..].find(value) else {
                break;
            };
            let byte_start = norm_byte + rel;
            let char_start = norm_b2c[byte_start];
            let end_idx = char_start + val_char_len - 1;
            if end_idx < offsets.len() {
                matches.insert((offsets[char_start].0, offsets[end_idx].1));
            }
            let Some(ch) = normalized[byte_start..].chars().next() else {
                break;
            };
            norm_byte = byte_start + ch.len_utf8();
        }

        if matches.is_empty() {
            (false, None)
        } else if matches.len() == 1
            && let Some(&(s, e)) = matches.iter().next()
        {
            (true, Some([s, e]))
        } else {
            (true, None)
        }
    }

    /// Look up whether `value` appears in source and return its unique Unicode char span
    /// (`grounding.py::Source.span`).
    pub fn span(&self, value: &Value, spec: Option<&Value>) -> (bool, Option<[usize; 2]>) {
        if let Some(s) = value.as_str() {
            if spec.is_some_and(|sp| self.whitespace_normalization(sp)) {
                return self.normalized_span(s);
            }
            if s.is_empty() {
                return (false, None);
            }
            let Some(byte_start) = self.text.find(s) else {
                return (false, None);
            };
            let next_byte = match self.text[byte_start..].chars().next() {
                Some(ch) => byte_start + ch.len_utf8(),
                None => self.text.len(),
            };
            let unique = self.text[next_byte..].find(s).is_none();
            let char_start = self.b2c[byte_start];
            let char_end = self.b2c[byte_start + s.len()];
            return (
                true,
                if unique {
                    Some([char_start, char_end])
                } else {
                    None
                },
            );
        }
        if !value.is_number() {
            return (false, None);
        }
        let Some(number) = decimal_util::from_json_value(value) else {
            return (false, None);
        };
        let key = canonical_decimal(&number);
        let Some(matches) = self.numerals.get(&key) else {
            return (false, None);
        };
        if matches.is_empty() {
            (false, None)
        } else if matches.len() == 1 {
            (true, Some([matches[0].0, matches[0].1]))
        } else {
            (true, None)
        }
    }

    /// Build and cache the [`GrammarSpec`] constraining `spec` to source-grounded outputs
    /// (`grounding.py::Source.constraint`).
    pub fn constraint(&self, spec: &Value) -> Result<GrammarSpec> {
        let kind = spec.get("type").and_then(Value::as_str).unwrap_or("");
        let document_numbers = kind == "number"
            && document_only(spec)
            && !is_truthy(spec.get("source_only"))
            && !state_only(spec);
        let schema = ar::schema(spec)?;
        let ws_norm = self.whitespace_normalization(spec);
        let key = (ar::compact_json(&schema), document_numbers, ws_norm);

        if let Ok(cache) = self.constraints.lock()
            && let Some(cached) = cache.get(&key)
        {
            return Ok(cached.clone());
        }

        let built = self.build_constraint(spec, &schema, document_numbers)?;
        if let Ok(mut cache) = self.constraints.lock() {
            cache.insert(key, built.clone());
        }
        Ok(built)
    }

    fn build_constraint(
        &self,
        spec: &Value,
        schema: &Value,
        document_numbers: bool,
    ) -> Result<GrammarSpec> {
        let kind = spec.get("type").and_then(Value::as_str).unwrap_or("");
        let has_enum = schema.get("enum").is_some();

        if kind == "number" || has_enum {
            let filtered_numerals: HashSet<BigDecimal> = if document_numbers {
                self.numerals
                    .iter()
                    .filter(|(_, spans)| {
                        spans
                            .iter()
                            .any(|&span| document_number_span(&self.chars, span))
                    })
                    .map(|(v, _)| v.clone())
                    .collect()
            } else {
                HashSet::new()
            };

            let mut values: Vec<(Value, Option<String>)> = Vec::new();
            if let Some(enum_arr) = schema.get("enum").and_then(Value::as_array) {
                for candidate in enum_arr {
                    if candidate.is_null() {
                        continue;
                    }
                    let doc_ok = if document_numbers {
                        decimal_util::from_json_value(candidate)
                            .map(|d| canonical_decimal(&d))
                            .is_some_and(|k| filtered_numerals.contains(&k))
                    } else {
                        true
                    };
                    if json_validate::is_valid(schema, candidate)
                        && doc_ok
                        && self.span(candidate, None).0
                    {
                        let lit = if kind == "number" {
                            decimal_ebnf_literal_from_json(candidate)
                        } else {
                            None
                        };
                        values.push((candidate.clone(), lit));
                    }
                }
            } else {
                for dec in &self.numeral_order {
                    if document_numbers && !filtered_numerals.contains(dec) {
                        continue;
                    }
                    let Some((json_val, lit)) = grounded_decimal_to_json_and_literal(dec) else {
                        continue;
                    };
                    if json_validate::is_valid(schema, &json_val) && self.span(&json_val, None).0 {
                        values.push((json_val, Some(lit)));
                    }
                }
            }

            if json_validate::is_valid(schema, &Value::Null) {
                values.push((Value::Null, Some("null".to_string())));
            }
            if values.is_empty() {
                return Err(invalid_request(
                    "Field constraints permit no literal source value or null",
                ));
            }

            if kind == "number" {
                let mut alts = Vec::with_capacity(values.len());
                for (val, lit_opt) in &values {
                    let lit = match lit_opt {
                        Some(s) => s.clone(),
                        None => decimal_ebnf_literal_from_json(val)
                            .ok_or_else(|| invalid_request("Invalid numeric literal"))?,
                    };
                    let quoted = serde_json::to_string(&lit).map_err(|e| {
                        invalid_request(format!("failed to encode number literal: {e}"))
                    })?;
                    alts.push(quoted);
                }
                return Ok(GrammarSpec::ClairicEbnf(format!(
                    "root ::= {}",
                    alts.join(" | ")
                )));
            }

            let mut enum_schema: Map<String, Value> =
                schema.as_object().cloned().unwrap_or_default();
            enum_schema.insert(
                "enum".to_string(),
                Value::Array(values.into_iter().map(|(v, _)| v).collect()),
            );
            return Ok(GrammarSpec::JsonSchema(Value::Object(enum_schema)));
        }

        let has_str_constraint = ["minLength", "maxLength", "pattern"]
            .iter()
            .any(|k| schema.get(*k).is_some());
        if self.chars.len() > MAX_GRAMMAR_CHARACTERS || has_str_constraint {
            return Ok(GrammarSpec::JsonSchema(schema.clone()));
        }

        let nullable = spec
            .get("nullable")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if self.text.is_empty() && !nullable {
            return Err(invalid_request(
                "Empty source permits no non-null text value",
            ));
        }
        let normalized_source = if self.whitespace_normalization(spec) {
            let norm = &self.whitespace_source().0;
            if norm != &self.text {
                Some(norm.clone())
            } else {
                None
            }
        } else {
            None
        };
        Ok(GrammarSpec::GroundedText {
            source: self.text.clone(),
            normalized_source,
            nullable,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scope_predicates_and_whitespace_collapse() {
        assert!(state_only(&json!({
            "type": "text",
            "instructions": "Extract code from the accompanying state text only. Return null if absent."
        })));
        assert!(!state_only(&json!({
            "type": "text",
            "instructions": "Extract code if available from the accompanying state text only."
        })));
        assert!(!state_only(&json!({
            "type": "text",
            "instructions": "Do not read from state text only."
        })));

        assert!(document_only(&json!({
            "type": "number",
            "instructions": "Total amount from this supplied document only."
        })));
        assert!(!document_only(&json!({
            "type": "number",
            "instructions": "Calculate the sum from this supplied document only."
        })));

        assert!(explicit_media_scope(&json!({
            "type": "text",
            "instructions": "Read the header on page 2."
        })));

        assert!(allows_whitespace_normalization(&json!({
            "type": "text",
            "instructions": "Vendor name."
        })));
        assert!(!allows_whitespace_normalization(&json!({
            "type": "text",
            "instructions": "Copy verbatim preserving whitespace."
        })));

        let (norm, offsets) = collapse_whitespace("foo  \n\tbar");
        assert_eq!(norm, "foo bar");
        assert_eq!(offsets[3], (3, 7));
        assert_eq!(offsets[4], (7, 8));
    }

    #[test]
    fn numbers_handles_us_eu_scientific_currency_and_accounting_negatives() {
        let us = numbers(
            "Subtotal $1,234.56; Discount ($45.00); Big 1.5e300; Tiny 2e-200; Refund: -€9.50",
        );
        let map: HashMap<BigDecimal, Vec<(usize, usize)>> = us.into_iter().collect();
        assert!(map.contains_key(&canonical_decimal(
            &decimal_util::parse_decimal("1234.56").unwrap()
        )));
        assert!(map.contains_key(&canonical_decimal(
            &decimal_util::parse_decimal("-45").unwrap()
        )));
        assert!(map.contains_key(&canonical_decimal(
            &decimal_util::parse_decimal("1.5e300").unwrap()
        )));
        assert!(map.contains_key(&canonical_decimal(
            &decimal_util::parse_decimal("2e-200").unwrap()
        )));
        assert!(map.contains_key(&canonical_decimal(
            &decimal_util::parse_decimal("-9.5").unwrap()
        )));

        let eu = numbers("Rechnung Gesamtbetrag: €1.234,56 und 2.500,00");
        let eu_vals: Vec<BigDecimal> = eu.into_iter().map(|(v, _)| v).collect();
        assert!(eu_vals.contains(&canonical_decimal(
            &decimal_util::parse_decimal("1234.56").unwrap()
        )));
        assert!(eu_vals.contains(&canonical_decimal(
            &decimal_util::parse_decimal("2500").unwrap()
        )));

        // Exponent beyond ±308 is skipped
        let huge = numbers("Value 1e309");
        assert!(huge.is_empty());
    }

    #[test]
    fn source_constraints_emit_grounded_text_ebnf_and_json_schema() {
        let src = Source::new("Invoice  #A100\nTotal: $42.50\nCode: OPEN", true);

        // 1. Text => GrammarSpec::GroundedText with normalized_source
        let text_spec = json!({"type": "text", "instructions": "Extract invoice status."});
        let g_text = src.constraint(&text_spec).unwrap();
        match g_text {
            GrammarSpec::GroundedText {
                source,
                normalized_source,
                nullable,
            } => {
                assert_eq!(source, src.text);
                assert_eq!(
                    normalized_source.as_deref(),
                    Some("Invoice #A100 Total: $42.50 Code: OPEN")
                );
                assert!(nullable);
            }
            other => panic!("expected GroundedText, got {other:?}"),
        }

        // Normalized span maps back to raw char offsets
        assert_eq!(
            src.span(&json!("Invoice #A100"), Some(&text_spec)),
            (true, Some([0, 14]))
        );

        // 2. Number => GrammarSpec::ClairicEbnf (document_only filters out "#A100" prefix? Note A100 has alpha prefix 'A'!)
        let doc_num_spec = json!({
            "type": "number",
            "instructions": "Extract amount from this supplied document only."
        });
        let g_num = src.constraint(&doc_num_spec).unwrap();
        match g_num {
            GrammarSpec::ClairicEbnf(ebnf) => {
                assert_eq!(ebnf, r#"root ::= "42.5" | "null""#);
            }
            other => panic!("expected ClairicEbnf, got {other:?}"),
        }

        // 3. Grounded enum => GrammarSpec::JsonSchema filtered to source values
        // (For type "text", `null` is only permitted if included in `enum`, matching ar.py + Draft202012.)
        let enum_spec = json!({"type": "text", "enum": ["OPEN", "CLOSED", null]});
        let g_enum = src.constraint(&enum_spec).unwrap();
        match g_enum {
            GrammarSpec::JsonSchema(val) => {
                assert_eq!(val["enum"], json!(["OPEN", null]));
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
        let non_null_enum_spec = json!({"type": "text", "enum": ["OPEN", "CLOSED"]});
        match src.constraint(&non_null_enum_spec).unwrap() {
            GrammarSpec::JsonSchema(val) => {
                assert_eq!(val["enum"], json!(["OPEN"]));
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }

        // 4. >4096-char source falls back to JsonSchema
        let long_text = "a".repeat(MAX_GRAMMAR_CHARACTERS + 1);
        let long_src = Source::new(&long_text, false);
        let g_long = long_src.constraint(&text_spec).unwrap();
        assert!(matches!(g_long, GrammarSpec::JsonSchema(_)));

        // 5. Empty source + non-nullable errors
        let empty_src = Source::new("", false);
        assert!(
            empty_src
                .constraint(&json!({"type": "text", "nullable": false}))
                .is_err()
        );
        assert!(matches!(
            empty_src
                .constraint(&json!({"type": "text", "nullable": true}))
                .unwrap(),
            GrammarSpec::GroundedText { nullable: true, .. }
        ));
    }
}
