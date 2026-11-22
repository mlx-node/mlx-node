//! Translator from Clairic / llama.cpp GBNF (`root ::= ...`) syntax to
//! `llguidance` Lark (`start: ...`) syntax.
//!
//! Supports all four grammar templates Clairic emits:
//! 1. Number literal alternation: `root ::= "12.5" | "100" | "null"`
//! 2. ISO date alternation: `root ::= "\"2024-03-15\"" | "null"`
//! 3. Boolean-or-null: `root ::= "true" | "false" | "null"`
//! 4. Suffix-automaton DAG: `root ::= "null" | "\"" s0 | "\"" n0`, `s0 ::= ...`
//!
//! Also handles general GBNF character classes (`[a-z0-9]`, `[^"\\]`),
//! grouping `(...)`, quantifiers (`*`, `+`, `?`), comments (`# ...`), and
//! multi-line rule definitions.

use std::fmt::Write as _;

use napi::{Error, Result};

/// Translate a Clairic / GBNF grammar into `llguidance` Lark grammar syntax.
pub fn clairic_ebnf_to_lark(ebnf: &str) -> Result<String> {
    let rules = split_gbnf_rules(ebnf)?;
    if rules.is_empty() {
        return Err(Error::from_reason("Clairic EBNF grammar has no rules"));
    }

    let mut out = String::with_capacity(ebnf.len() + 64);
    let mut has_start = false;

    for (raw_lhs, raw_rhs) in &rules {
        let lhs = map_rule_name(raw_lhs);
        if lhs == "start" {
            has_start = true;
        }
        let rhs = translate_rhs(raw_rhs)?;
        if rhs.trim().is_empty() {
            return Err(Error::from_reason(format!(
                "Empty right-hand side for EBNF rule `{raw_lhs}`"
            )));
        }
        let _ = writeln!(out, "{lhs}: {rhs}");
    }

    if !has_start {
        return Err(Error::from_reason(
            "Clairic EBNF grammar must define a `root` (or `start`) rule",
        ));
    }

    Ok(out)
}

/// Strip `# ...` comments from a single line without touching `#` inside
/// double-quoted string literals or `[...]` character classes.
fn strip_line_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut i = 0;
    let mut in_str = false;
    let mut in_class = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_str = false;
            }
        } else if in_class {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == b']' {
                in_class = false;
            }
        } else {
            match b {
                b'#' => return &line[..i],
                b'"' => in_str = true,
                b'[' => in_class = true,
                _ => {}
            }
        }
        i += 1;
    }
    line
}

/// Parse `<name> ::= <rhs>` header if `line` starts a new rule definition.
fn parse_rule_header(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim_start();
    let bytes = trimmed.as_bytes();
    let mut end = 0;
    while end < bytes.len() && is_ident_char(bytes[end]) {
        end += 1;
    }
    if end == 0 {
        return None;
    }
    let name = &trimmed[..end];
    let rest = trimmed[end..].trim_start();
    let rhs = rest.strip_prefix("::=")?;
    Some((name, rhs))
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

fn split_gbnf_rules(ebnf: &str) -> Result<Vec<(String, String)>> {
    let mut rules: Vec<(String, String)> = Vec::new();

    for (line_idx, raw_line) in ebnf.lines().enumerate() {
        let line = strip_line_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if let Some((name, rhs)) = parse_rule_header(line) {
            rules.push((name.to_string(), rhs.trim().to_string()));
        } else if let Some((_, prev_rhs)) = rules.last_mut() {
            if !prev_rhs.is_empty() {
                prev_rhs.push(' ');
            }
            prev_rhs.push_str(line);
        } else {
            return Err(Error::from_reason(format!(
                "Unexpected token before first EBNF rule on line {}: {line}",
                line_idx + 1
            )));
        }
    }

    Ok(rules)
}

/// Map a GBNF rule identifier to a valid `llguidance` Lark rule name.
///
/// Lark rules must start with `[a-z]` (uppercase starts denote terminals/tokens).
fn map_rule_name(name: &str) -> String {
    if name == "root" {
        return "start".to_string();
    }
    let needs_prefix = name
        .as_bytes()
        .first()
        .is_none_or(|b| !b.is_ascii_lowercase());
    let mut mapped = String::with_capacity(name.len() + usize::from(needs_prefix) * 2);
    if needs_prefix {
        mapped.push_str("r_");
    }
    for ch in name.chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-' {
            mapped.push(ch);
        } else if ch.is_ascii_uppercase() {
            mapped.push(ch.to_ascii_lowercase());
        } else {
            mapped.push('_');
        }
    }
    mapped
}

fn translate_rhs(rhs: &str) -> Result<String> {
    let bytes = rhs.as_bytes();
    let mut i = 0;
    let mut out = String::with_capacity(rhs.len() + 16);

    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b' ' | b'\t' | b'\r' | b'\n' => {
                if !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                }
                i += 1;
            }
            b'"' => {
                let (lark_str, next_i) = translate_gbnf_string(rhs, i)?;
                out.push_str(&lark_str);
                i = next_i;
            }
            b'[' => {
                let (lark_rx, next_i) = translate_char_class(rhs, i)?;
                out.push_str(&lark_rx);
                i = next_i;
            }
            b'|' | b'(' | b')' | b'*' | b'+' | b'?' => {
                out.push(char::from(b));
                i += 1;
            }
            _ if is_ident_char(b) => {
                let start = i;
                while i < bytes.len() && is_ident_char(bytes[i]) {
                    i += 1;
                }
                let ident = &rhs[start..i];
                out.push_str(&map_rule_name(ident));
            }
            _ => {
                return Err(Error::from_reason(format!(
                    "Unsupported character `{}` in Clairic EBNF expression: {rhs}",
                    rhs[i..].chars().next().unwrap_or('?')
                )));
            }
        }
    }

    Ok(out.trim().to_string())
}

/// Translate a double-quoted GBNF string literal into a `llguidance` Lark string literal.
fn translate_gbnf_string(src: &str, start: usize) -> Result<(String, usize)> {
    let mut chars = src[start..].char_indices();
    let Some((_, '"')) = chars.next() else {
        return Err(Error::from_reason("Expected opening quote in EBNF string"));
    };

    let mut out = String::from("\"");
    while let Some((rel_idx, ch)) = chars.next() {
        match ch {
            '"' => {
                out.push('"');
                return Ok((out, start + rel_idx + 1));
            }
            '\\' => {
                let Some((_, esc)) = chars.next() else {
                    return Err(Error::from_reason("Unterminated escape in EBNF string"));
                };
                match esc {
                    '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                        out.push('\\');
                        out.push(esc);
                    }
                    'x' => {
                        let hex = take_hex(&mut chars, 2)?;
                        out.push_str("\\u00");
                        out.push_str(&hex);
                    }
                    'u' => {
                        let hex = take_hex(&mut chars, 4)?;
                        out.push_str("\\u");
                        out.push_str(&hex);
                    }
                    'U' => {
                        let hex = take_hex(&mut chars, 8)?;
                        let cp = u32::from_str_radix(&hex, 16).map_err(|_| {
                            Error::from_reason(format!("Invalid \\U escape in EBNF: \\U{hex}"))
                        })?;
                        let decoded = char::from_u32(cp).ok_or_else(|| {
                            Error::from_reason(format!("Invalid unicode scalar \\U{hex} in EBNF"))
                        })?;
                        push_lark_char(&mut out, decoded);
                    }
                    other => {
                        // Preserve literal character safely as JSON/Lark char.
                        push_lark_char(&mut out, other);
                    }
                }
            }
            other => {
                push_lark_char(&mut out, other);
            }
        }
    }

    Err(Error::from_reason("Unterminated string literal in EBNF"))
}

fn take_hex(chars: &mut std::str::CharIndices<'_>, count: usize) -> Result<String> {
    let mut hex = String::with_capacity(count);
    for _ in 0..count {
        match chars.next() {
            Some((_, c)) if c.is_ascii_hexdigit() => hex.push(c),
            _ => {
                return Err(Error::from_reason(format!(
                    "Expected {count} hex digits in EBNF escape sequence"
                )));
            }
        }
    }
    Ok(hex)
}

fn push_lark_char(out: &mut String, ch: char) {
    match ch {
        '"' => out.push_str("\\\""),
        '\\' => out.push_str("\\\\"),
        '\x08' => out.push_str("\\b"),
        '\x0C' => out.push_str("\\f"),
        '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"),
        '\t' => out.push_str("\\t"),
        c if (c as u32) < 0x20 || c == '\x7F' => {
            let _ = write!(out, "\\u{:04x}", c as u32);
        }
        c => out.push(c),
    }
}

/// Translate a GBNF character class `[...]` into a `llguidance` Lark regex `/[...]/`.
fn translate_char_class(src: &str, start: usize) -> Result<(String, usize)> {
    let mut chars = src[start..].char_indices();
    let Some((_, '[')) = chars.next() else {
        return Err(Error::from_reason("Expected `[` in EBNF character class"));
    };

    let mut out = String::from("/[");
    let mut first = true;
    while let Some((rel_idx, ch)) = chars.next() {
        match ch {
            ']' if !first => {
                out.push_str("]/");
                return Ok((out, start + rel_idx + 1));
            }
            '\\' => {
                let Some((_, esc)) = chars.next() else {
                    return Err(Error::from_reason(
                        "Unterminated escape in EBNF character class",
                    ));
                };
                out.push('\\');
                out.push(esc);
            }
            '/' => {
                out.push_str("\\/");
            }
            other => {
                out.push(other);
            }
        }
        first = false;
    }

    Err(Error::from_reason("Unterminated character class in EBNF"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_alternation() {
        let ebnf = r#"root ::= "12.5" | "100" | "null""#;
        let lark = clairic_ebnf_to_lark(ebnf).expect("translate number alternation");
        assert_eq!(lark.trim(), r#"start: "12.5" | "100" | "null""#);
    }

    #[test]
    fn test_iso_date_alternation() {
        let ebnf = r#"root ::= "\"2024-03-15\"" | "null""#;
        let lark = clairic_ebnf_to_lark(ebnf).expect("translate iso date");
        assert_eq!(lark.trim(), r#"start: "\"2024-03-15\"" | "null""#);
    }

    #[test]
    fn test_boolean_or_null() {
        let ebnf = r#"root ::= "true" | "false" | "null""#;
        let lark = clairic_ebnf_to_lark(ebnf).expect("translate boolean");
        assert_eq!(lark.trim(), r#"start: "true" | "false" | "null""#);
    }

    #[test]
    fn test_suffix_automaton_dag() {
        let ebnf = r#"
# Clairic grounded text suffix automaton
root ::= "null" | "\"" s0 | "\"" n0
s0 ::= "a" s1 | "b" s2
s1 ::= "\"" | "b" s2
s2 ::= "\""
n0 ::= "A" n1
n1 ::= "\""
"#;
        let lark = clairic_ebnf_to_lark(ebnf).expect("translate suffix dag");
        assert!(lark.contains(r#"start: "null" | "\"" s0 | "\"" n0"#));
        assert!(lark.contains(r#"s0: "a" s1 | "b" s2"#));
        assert!(lark.contains(r#"s1: "\"" | "b" s2"#));
        assert!(lark.contains(r#"s2: "\"""#));
        assert!(lark.contains(r#"n0: "A" n1"#));
    }

    #[test]
    fn test_char_classes_and_escapes() {
        let ebnf = "root ::= [0-9]+ \"\\x20\" \"\\U0001F680\"";
        let lark = clairic_ebnf_to_lark(ebnf).expect("translate char class");
        assert_eq!(lark.trim(), r#"start: /[0-9]/+ "\u0020" "🚀""#);
    }

    #[test]
    fn test_missing_root_errors() {
        assert!(clairic_ebnf_to_lark(r#"foo ::= "bar""#).is_err());
        assert!(clairic_ebnf_to_lark("   # comment only").is_err());
    }
}
