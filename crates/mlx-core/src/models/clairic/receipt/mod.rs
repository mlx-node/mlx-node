//! Receipt sum, receipt block totals, and hybrid receipt inventory extraction/repair.
//!
//! Direct port of upstream:
//! - `clairic_runtime/source_sums.py` -> [`source_sums`]
//! - `clairic_runtime/receipt_totals.py` -> [`totals`]
//! - `clairic_runtime/receipt_hybrid_inventory.py` -> [`hybrid_inventory`]

pub mod hybrid_inventory;
pub mod source_sums;
pub mod totals;

use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::sync::OnceLock;

/// Canonical UTF-8 JSON bytes with recursively sorted object keys and compact separators,
/// matching Python `json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()`.
pub fn canonical(value: &Value) -> Vec<u8> {
    let sorted = sort_value_keys(value);
    serde_json::to_vec(&sorted).unwrap_or_default()
}

fn sort_value_keys(value: &Value) -> Value {
    match value {
        Value::Array(arr) => Value::Array(arr.iter().map(sort_value_keys).collect()),
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = Map::with_capacity(map.len());
            for k in keys {
                if let Some(v) = map.get(k) {
                    out.insert(k.clone(), sort_value_keys(v));
                }
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// Lowercase hex SHA-256 digest of `bytes`.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

/// Python-style truthiness check for optional JSON values.
pub(crate) fn is_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

fn unicode_currency_symbol_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\A\p{Sc}\z").unwrap_or_else(|_| unreachable!()))
}

/// Return `true` iff `ch` has Unicode General Category `Sc` (Currency_Symbol).
pub(crate) fn is_currency_symbol(ch: char) -> bool {
    if ch.is_ascii() {
        return ch == '$';
    }
    let mut buf = [0u8; 4];
    unicode_currency_symbol_regex().is_match(ch.encode_utf8(&mut buf))
}

/// Map Unicode character index `i` to UTF-8 byte offset `c2b[i]`, with `c2b[char_len] == text.len()`.
pub(crate) fn char_to_byte_map(text: &str) -> Vec<usize> {
    let mut map: Vec<usize> = text.char_indices().map(|(bi, _)| bi).collect();
    map.push(text.len());
    map
}

/// Slice `source` by Unicode character offsets `[c_start..c_end]`.
pub(crate) fn slice_chars<'a>(
    source: &'a str,
    c2b: &[usize],
    c_start: usize,
    c_end: usize,
) -> &'a str {
    let Some(&b_start) = c2b.get(c_start) else {
        return "";
    };
    let Some(&b_end) = c2b.get(c_end) else {
        return "";
    };
    if b_start <= b_end && b_end <= source.len() {
        &source[b_start..b_end]
    } else {
        ""
    }
}

/// Extract `[start, end]` Unicode character span from a JSON array `[u64, u64]`.
pub(crate) fn extract_span(value: Option<&Value>) -> Option<(usize, usize)> {
    let arr = value?.as_array()?;
    if arr.len() != 2 {
        return None;
    }
    let start = usize::try_from(arr[0].as_u64()?).ok()?;
    let end = usize::try_from(arr[1].as_u64()?).ok()?;
    Some((start, end))
}

/// Find all matches of `re` in `source` within Unicode char range `[c_start, c_end]`,
/// returning `(char_start, char_end)` spans matching Python's `re.finditer(source, c_start, c_end)`.
pub(crate) fn find_char_spans_in_range(
    re: &Regex,
    source: &str,
    c2b: &[usize],
    b2c: &[usize],
    c_start: usize,
    c_end: usize,
) -> Vec<(usize, usize)> {
    let Some(&b_start) = c2b.get(c_start) else {
        return Vec::new();
    };
    let Some(&b_end) = c2b.get(c_end) else {
        return Vec::new();
    };
    if b_start > b_end || b_end > source.len() {
        return Vec::new();
    }
    let slice = &source[..b_end];
    let mut out = Vec::new();
    let mut pos = b_start;
    while pos <= b_end {
        let Some(m) = re.find_at(slice, pos) else {
            break;
        };
        out.push((b2c[m.start()], b2c[m.end()]));
        if m.end() > pos {
            pos = m.end();
        } else {
            match slice[pos..].chars().next() {
                Some(ch) => pos += ch.len_utf8(),
                None => break,
            }
        }
    }
    out
}

/// Find all matches of a lookaround `FancyRegex` over full `source`, returning `(char_start, char_end)` spans.
pub(crate) fn find_char_spans_fancy(
    re: &FancyRegex,
    source: &str,
    b2c: &[usize],
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos <= source.len() {
        let Ok(Some(m)) = re.find_from_pos(source, pos) else {
            break;
        };
        out.push((b2c[m.start()], b2c[m.end()]));
        if m.end() > pos {
            pos = m.end();
        } else {
            match source[pos..].chars().next() {
                Some(ch) => pos += ch.len_utf8(),
                None => break,
            }
        }
    }
    out
}

/// Compensated (Neumaier) floating-point summation approximating Python's `math.fsum`.
pub(crate) fn accurate_fsum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &x in values {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}
