//! List query budget computation (`clairic_runtime/lists.py`).

use super::errors::{ClairicError, invalid_request};
use regex::Regex;
use serde_json::Value;
use std::fmt;
use std::sync::OnceLock;

/// Record-rejection error matching upstream `lists.py::RejectRecord`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectRecord {
    pub reason: String,
    pub field: Option<String>,
}

impl RejectRecord {
    pub fn new(reason: impl Into<String>, field: Option<String>) -> Self {
        Self {
            reason: reason.into(),
            field,
        }
    }

    pub fn into_clairic_error(self) -> ClairicError {
        invalid_request(self.to_string())
    }
}

impl fmt::Display for RejectRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.field {
            Some(field) if !field.is_empty() => write!(f, "{}: {field}", self.reason),
            _ => f.write_str(&self.reason),
        }
    }
}

impl std::error::Error for RejectRecord {}

impl From<RejectRecord> for ClairicError {
    fn from(err: RejectRecord) -> Self {
        err.into_clairic_error()
    }
}

fn char_list_request_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:(?:extract|list|return|split|read)\s+(?:(?:the|of)\s+)?|(?:each|every|all|individual|separate)\s+)(?:(?:single|individual)\s+)?(?:characters?|letters?|digits?|symbols?)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn preserve_prefix_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:preserv\w*|keep|retain\w*|including|with)\s*$")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn row_list_primary_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:rows?|top to bottom|line items?|item lines?|receipt order|printed row order)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn row_list_doc_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:receipt|invoice|table)\b").unwrap_or_else(|_| unreachable!())
    })
}

fn row_list_item_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:items?|products?|quantit\w*|prices?|amounts?|names?)\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn lexical_unit_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\w+|[^\w\s]").unwrap_or_else(|_| unreachable!()))
}

/// Count non-blank lines following Python `str.splitlines()` semantics.
fn count_nonblank_lines(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    let mut line_has_non_ws = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        let is_line_break = matches!(
            ch,
            '\n' | '\r'
                | '\x0b'
                | '\x0c'
                | '\x1c'
                | '\x1d'
                | '\x1e'
                | '\u{0085}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_line_break {
            if ch == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
            }
            if line_has_non_ws {
                count += 1;
                line_has_non_ws = false;
            }
        } else if !ch.is_whitespace() {
            line_has_non_ws = true;
        }
    }
    if line_has_non_ws {
        count += 1;
    }
    count
}

/// Slice the last `max_chars` Unicode scalar values before byte offset `byte_end`.
fn lookback_chars(s: &str, byte_end: usize, max_chars: usize) -> &str {
    let prefix = &s[..byte_end];
    let mut count = 0usize;
    let mut start = byte_end;
    for (idx, _) in prefix.char_indices().rev() {
        start = idx;
        count += 1;
        if count >= max_chars {
            break;
        }
    }
    if count == 0 { "" } else { &s[start..byte_end] }
}

/// Compute `(query_budget, basis)` for a list field (`lists.py::list_query_budget`).
pub fn list_query_budget(
    text: &str,
    spec: &Value,
) -> std::result::Result<(usize, &'static str), RejectRecord> {
    if let Some(maximum) = spec.get("max_items")
        && !maximum.is_null()
    {
        if maximum.is_boolean() {
            return Err(RejectRecord::new("invalid_explicit_list_budget", None));
        }
        let parsed = if let Some(u) = maximum.as_u64() {
            usize::try_from(u).ok()
        } else if let Some(i) = maximum.as_i64() {
            usize::try_from(i).ok()
        } else {
            None
        };
        let Some(max_val) = parsed else {
            return Err(RejectRecord::new("invalid_explicit_list_budget", None));
        };
        return Ok((max_val.saturating_add(1), "request_max_items"));
    }

    let instructions_owned: String;
    let instructions: &str = match spec.get("instructions") {
        None => "",
        Some(Value::String(s)) => s.as_str(),
        Some(other) => {
            instructions_owned = other.to_string();
            instructions_owned.as_str()
        }
    };

    for m in char_list_request_regex().find_iter(instructions) {
        let prefix = lookback_chars(instructions, m.start(), 25);
        if !preserve_prefix_regex().is_match(prefix) {
            let non_ws = text.chars().filter(|c| !c.is_whitespace()).count();
            return Ok((non_ws + 1, "source_characters"));
        }
    }

    let nonblank_lines = count_nonblank_lines(text);
    let row_list = row_list_primary_regex().is_match(instructions)
        || (row_list_doc_regex().is_match(instructions)
            && row_list_item_regex().is_match(instructions));
    if row_list && nonblank_lines > 1 {
        return Ok((nonblank_lines + 1, "nonblank_source_lines"));
    }

    let lexical_units = lexical_unit_regex().find_iter(text).count();
    Ok((lexical_units + 1, "source_lexical_units"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn explicit_max_items_budget_and_validation() {
        assert_eq!(
            list_query_budget("a b c", &json!({"max_items": 5})),
            Ok((6, "request_max_items"))
        );
        assert_eq!(
            list_query_budget("a b c", &json!({"max_items": 0})),
            Ok((1, "request_max_items"))
        );
        assert!(list_query_budget("a", &json!({"max_items": -1})).is_err());
        assert!(list_query_budget("a", &json!({"max_items": true})).is_err());
        assert!(list_query_budget("a", &json!({"max_items": 2.5})).is_err());
        assert!(list_query_budget("a", &json!({"max_items": "5"})).is_err());
    }

    #[test]
    fn source_characters_budget_and_preserve_guard() {
        let spec = json!({"instructions": "Extract each individual digit from the code."});
        assert_eq!(
            list_query_budget("A1 B2 3", &spec),
            Ok((6, "source_characters"))
        );

        let preserved =
            json!({"instructions": "Extract words, preserving all characters and symbols."});
        let (budget, basis) = list_query_budget("hello, world!", &preserved).unwrap();
        assert_eq!(basis, "source_lexical_units");
        assert_eq!(budget, 5); // hello , world ! => 4 + 1
    }

    #[test]
    fn nonblank_source_lines_for_row_lists() {
        let text = "Coffee 3.50\n\nTea 2.00\nMuffin 4.25\n";
        let spec = json!({"instructions": "Extract invoice line items in receipt order."});
        assert_eq!(
            list_query_budget(text, &spec),
            Ok((4, "nonblank_source_lines"))
        );

        let receipt_items = json!({"instructions": "List the prices on this receipt."});
        assert_eq!(
            list_query_budget(text, &receipt_items),
            Ok((4, "nonblank_source_lines"))
        );

        // Single line falls back to lexical units
        assert_eq!(
            list_query_budget("Coffee 3.50", &spec),
            Ok((5, "source_lexical_units")) // Coffee, 3, ., 50 => 4 + 1
        );
    }
}
