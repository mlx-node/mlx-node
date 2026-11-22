//! Pure-Rust streaming HTML table parser with exact Unicode character offsets
//! and HTML entity decoding.
//!
//! Direct port of upstream `clairic_runtime/html_rows.py`, plus a reusable
//! `HTMLParser`-compatible streaming tokenizer shared with `source_rows.rs`.

use super::schema::byte_to_char_map;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// One parsed `<tr>` row from an HTML `<table>`, with exact Unicode character
/// offsets matching Python's `str` indexing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HtmlRow {
    pub cells: Vec<String>,
    pub line: String,
    pub offset: usize,
    pub spans: Vec<Option<[usize; 2]>>,
}

/// Streaming callback interface mirroring Python's `html.parser.HTMLParser`.
pub(crate) trait HtmlHandler {
    fn handle_starttag(&mut self, tag: &str, attrs: &[(String, Option<String>)], char_pos: usize);
    fn handle_endtag(&mut self, tag: &str, char_pos: usize, byte_pos: usize);
    fn handle_data(&mut self, data: &str, char_pos: usize);
}

/// Decode a single HTML character or named entity reference (the body between `&` and `;`).
/// Returns `None` if unrecognized (in which case the caller keeps the literal `&...` text).
pub fn decode_html_entity(body: &str) -> Option<String> {
    if let Some(rest) = body.strip_prefix('#') {
        let code = if let Some(hex) = rest.strip_prefix('x').or_else(|| rest.strip_prefix('X')) {
            if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return None;
            }
            u32::from_str_radix(hex, 16).ok()?
        } else {
            if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            rest.parse::<u32>().ok()?
        };
        return Some(decode_numeric_codepoint(code).to_string());
    }
    decode_named_entity(body).map(str::to_string)
}

/// Map a numeric HTML character reference to a Unicode scalar value following
/// WHATWG / Python `html.unescape` rules (including Windows-1252 C1 replacements).
fn decode_numeric_codepoint(code: u32) -> char {
    match code {
        0x00 => '\u{FFFD}',
        0x80 => '\u{20AC}',
        0x82 => '\u{201A}',
        0x83 => '\u{0192}',
        0x84 => '\u{201E}',
        0x85 => '\u{2026}',
        0x86 => '\u{2020}',
        0x87 => '\u{2021}',
        0x88 => '\u{02C6}',
        0x89 => '\u{2030}',
        0x8A => '\u{0160}',
        0x8B => '\u{2039}',
        0x8C => '\u{0152}',
        0x8E => '\u{017D}',
        0x91 => '\u{2018}',
        0x92 => '\u{2019}',
        0x93 => '\u{201C}',
        0x94 => '\u{201D}',
        0x95 => '\u{2022}',
        0x96 => '\u{2013}',
        0x97 => '\u{2014}',
        0x98 => '\u{02DC}',
        0x99 => '\u{2122}',
        0x9A => '\u{0161}',
        0x9B => '\u{203A}',
        0x9C => '\u{0153}',
        0x9E => '\u{017E}',
        0x9F => '\u{0178}',
        c if (0xD800..=0xDFFF).contains(&c) || c > 0x10FFFF => '\u{FFFD}',
        c => char::from_u32(c).unwrap_or('\u{FFFD}'),
    }
}

fn decode_named_entity(name: &str) -> Option<&'static str> {
    match name {
        "amp" | "AMP" => Some("&"),
        "lt" | "LT" => Some("<"),
        "gt" | "GT" => Some(">"),
        "quot" | "QUOT" => Some("\""),
        "apos" => Some("'"),
        "nbsp" | "NonBreakingSpace" => Some("\u{00A0}"),
        "ensp" => Some("\u{2002}"),
        "emsp" => Some("\u{2003}"),
        "thinsp" => Some("\u{2009}"),
        "shy" => Some("\u{00AD}"),
        "zwnj" => Some("\u{200C}"),
        "zwj" => Some("\u{200D}"),
        "ndash" => Some("\u{2013}"),
        "mdash" => Some("\u{2014}"),
        "lsquo" => Some("\u{2018}"),
        "rsquo" => Some("\u{2019}"),
        "sbquo" => Some("\u{201A}"),
        "ldquo" => Some("\u{201C}"),
        "rdquo" => Some("\u{201D}"),
        "bdquo" => Some("\u{201E}"),
        "bull" => Some("\u{2022}"),
        "hellip" => Some("\u{2026}"),
        "permil" => Some("\u{2030}"),
        "prime" => Some("\u{2032}"),
        "Prime" => Some("\u{2033}"),
        "lsaquo" => Some("\u{2039}"),
        "rsaquo" => Some("\u{203A}"),
        "laquo" => Some("\u{00AB}"),
        "raquo" => Some("\u{00BB}"),
        "iexcl" => Some("\u{00A1}"),
        "cent" => Some("\u{00A2}"),
        "pound" => Some("\u{00A3}"),
        "curren" => Some("\u{00A4}"),
        "yen" => Some("\u{00A5}"),
        "brvbar" => Some("\u{00A6}"),
        "sect" => Some("\u{00A7}"),
        "uml" => Some("\u{00A8}"),
        "copy" | "COPY" => Some("\u{00A9}"),
        "ordf" => Some("\u{00AA}"),
        "not" => Some("\u{00AC}"),
        "reg" | "REG" => Some("\u{00AE}"),
        "macr" => Some("\u{00AF}"),
        "deg" => Some("\u{00B0}"),
        "plusmn" => Some("\u{00B1}"),
        "sup2" => Some("\u{00B2}"),
        "sup3" => Some("\u{00B3}"),
        "acute" => Some("\u{00B4}"),
        "micro" => Some("\u{00B5}"),
        "para" => Some("\u{00B6}"),
        "middot" => Some("\u{00B7}"),
        "cedil" => Some("\u{00B8}"),
        "sup1" => Some("\u{00B9}"),
        "ordm" => Some("\u{00BA}"),
        "frac14" => Some("\u{00BC}"),
        "frac12" => Some("\u{00BD}"),
        "frac34" => Some("\u{00BE}"),
        "iquest" => Some("\u{00BF}"),
        "times" => Some("\u{00D7}"),
        "divide" => Some("\u{00F7}"),
        "minus" => Some("\u{2212}"),
        "euro" => Some("\u{20AC}"),
        "trade" | "TRADE" => Some("\u{2122}"),
        "Agrave" => Some("\u{00C0}"),
        "Aacute" => Some("\u{00C1}"),
        "Acirc" => Some("\u{00C2}"),
        "Atilde" => Some("\u{00C3}"),
        "Auml" => Some("\u{00C4}"),
        "Aring" => Some("\u{00C5}"),
        "AElig" => Some("\u{00C6}"),
        "Ccedil" => Some("\u{00C7}"),
        "Egrave" => Some("\u{00C8}"),
        "Eacute" => Some("\u{00C9}"),
        "Ecirc" => Some("\u{00CA}"),
        "Euml" => Some("\u{00CB}"),
        "Igrave" => Some("\u{00CC}"),
        "Iacute" => Some("\u{00CD}"),
        "Icirc" => Some("\u{00CE}"),
        "Iuml" => Some("\u{00CF}"),
        "Ntilde" => Some("\u{00D1}"),
        "Ograve" => Some("\u{00D2}"),
        "Oacute" => Some("\u{00D3}"),
        "Ocirc" => Some("\u{00D4}"),
        "Otilde" => Some("\u{00D5}"),
        "Ouml" => Some("\u{00D6}"),
        "Oslash" => Some("\u{00D8}"),
        "Ugrave" => Some("\u{00D9}"),
        "Uacute" => Some("\u{00DA}"),
        "Ucirc" => Some("\u{00DB}"),
        "Uuml" => Some("\u{00DC}"),
        "szlig" => Some("\u{00DF}"),
        "agrave" => Some("\u{00E0}"),
        "aacute" => Some("\u{00E1}"),
        "acirc" => Some("\u{00E2}"),
        "atilde" => Some("\u{00E3}"),
        "auml" => Some("\u{00E4}"),
        "aring" => Some("\u{00E5}"),
        "aelig" => Some("\u{00E6}"),
        "ccedil" => Some("\u{00E7}"),
        "egrave" => Some("\u{00E8}"),
        "eacute" => Some("\u{00E9}"),
        "ecirc" => Some("\u{00EA}"),
        "euml" => Some("\u{00EB}"),
        "igrave" => Some("\u{00EC}"),
        "iacute" => Some("\u{00ED}"),
        "icirc" => Some("\u{00EE}"),
        "iuml" => Some("\u{00EF}"),
        "ntilde" => Some("\u{00F1}"),
        "ograve" => Some("\u{00F2}"),
        "oacute" => Some("\u{00F3}"),
        "ocirc" => Some("\u{00F4}"),
        "otilde" => Some("\u{00F5}"),
        "ouml" => Some("\u{00F6}"),
        "oslash" => Some("\u{00F8}"),
        "ugrave" => Some("\u{00F9}"),
        "uacute" => Some("\u{00FA}"),
        "ucirc" => Some("\u{00FB}"),
        "uuml" => Some("\u{00FC}"),
        "yuml" => Some("\u{00FF}"),
        "le" => Some("\u{2264}"),
        "ge" => Some("\u{2265}"),
        "ne" => Some("\u{2260}"),
        "approx" => Some("\u{2248}"),
        "infin" => Some("\u{221E}"),
        "larr" => Some("\u{2190}"),
        "uarr" => Some("\u{2191}"),
        "rarr" => Some("\u{2192}"),
        "darr" => Some("\u{2193}"),
        "harr" => Some("\u{2194}"),
        _ => None,
    }
}

/// Unescape HTML entities inside a text or attribute string, matching Python `html.unescape`.
pub fn unescape_html(input: &str) -> String {
    if !input.contains('&') {
        return input.to_string();
    }
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'&'
            && let Some((decoded, consumed)) = parse_entity_at(input, i)
        {
            out.push_str(&decoded);
            i += consumed;
            continue;
        }
        let Some(ch) = input[i..].chars().next() else {
            break;
        };
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Try to parse an HTML character/entity reference starting at byte `start` (`&`).
/// Returns `(decoded_string, byte_len_consumed)`.
fn parse_entity_at(input: &str, start: usize) -> Option<(String, usize)> {
    let rest = &input[start + 1..];
    let bytes = rest.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    // Look for `;` within a bounded window (longest HTML5 named entity is ~32 chars).
    let limit = bytes.len().min(36);
    if bytes[0] == b'#' {
        let mut end = 1;
        if end < limit && (bytes[end] == b'x' || bytes[end] == b'X') {
            end += 1;
            let digits_start = end;
            while end < limit && bytes[end].is_ascii_hexdigit() {
                end += 1;
            }
            if end == digits_start {
                return None;
            }
        } else {
            let digits_start = end;
            while end < limit && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end == digits_start {
                return None;
            }
        }
        let body = &rest[..end];
        let consumed = if end < bytes.len() && bytes[end] == b';' {
            1 + end + 1
        } else {
            1 + end
        };
        let decoded = decode_html_entity(body)?;
        return Some((decoded, consumed));
    }

    if bytes[0].is_ascii_alphabetic() {
        let mut end = 1;
        while end < limit && bytes[end].is_ascii_alphanumeric() {
            end += 1;
        }
        if end < bytes.len() && bytes[end] == b';' {
            let body = &rest[..end];
            if let Some(decoded) = decode_html_entity(body) {
                return Some((decoded, 1 + end + 1));
            }
        }
    }
    None
}

/// Find the next entity reference in `text` (used when `convert_charrefs = false`
/// to mirror Python's `HTMLParser` splitting `handle_data` around `&...;`).
fn find_next_entity_span(text: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'&'
            && let Some((_, consumed)) = parse_entity_at(text, i)
        {
            return Some((i, i + consumed));
        }
        i += 1;
    }
    None
}

fn is_html_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0C)
}

fn is_tag_name_start(b: u8) -> bool {
    b.is_ascii_alphabetic()
}

fn is_tag_name_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b':' | b'.')
}

struct ParsedStartTag {
    tag: String,
    attrs: Vec<(String, Option<String>)>,
    self_closing: bool,
    next_byte: usize,
}

/// Parse a start tag starting at `source[start..]` where `source.as_bytes()[start] == b'<'`.
fn parse_starttag(source: &str, start: usize) -> Option<ParsedStartTag> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    if start + 1 >= len || !is_tag_name_start(bytes[start + 1]) {
        return None;
    }
    let mut i = start + 1;
    while i < len && is_tag_name_char(bytes[i]) {
        i += 1;
    }
    let tag = source[start + 1..i].to_ascii_lowercase();
    let mut attrs: Vec<(String, Option<String>)> = Vec::new();
    let mut self_closing = false;

    loop {
        while i < len && is_html_whitespace(bytes[i]) {
            i += 1;
        }
        if i >= len {
            return None;
        }
        if bytes[i] == b'>' {
            return Some(ParsedStartTag {
                tag,
                attrs,
                self_closing,
                next_byte: i + 1,
            });
        }
        if bytes[i] == b'/' {
            if i + 1 < len && bytes[i + 1] == b'>' {
                self_closing = true;
                return Some(ParsedStartTag {
                    tag,
                    attrs,
                    self_closing,
                    next_byte: i + 2,
                });
            }
            i += 1;
            continue;
        }

        // Parse attribute name.
        let name_start = i;
        while i < len && !is_html_whitespace(bytes[i]) && !matches!(bytes[i], b'=' | b'>' | b'/') {
            i += 1;
        }
        if i == name_start {
            // Skip stray byte to guarantee forward progress.
            i += 1;
            continue;
        }
        let attr_name = source[name_start..i].to_ascii_lowercase();

        while i < len && is_html_whitespace(bytes[i]) {
            i += 1;
        }
        let attr_val = if i < len && bytes[i] == b'=' {
            i += 1;
            while i < len && is_html_whitespace(bytes[i]) {
                i += 1;
            }
            if i >= len {
                return None;
            }
            if bytes[i] == b'"' || bytes[i] == b'\'' {
                let quote = bytes[i];
                i += 1;
                let val_start = i;
                while i < len && bytes[i] != quote {
                    i += 1;
                }
                if i >= len {
                    return None;
                }
                let raw_val = &source[val_start..i];
                i += 1; // consume closing quote
                Some(unescape_html(raw_val))
            } else {
                let val_start = i;
                while i < len && !is_html_whitespace(bytes[i]) && bytes[i] != b'>' {
                    if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'>' {
                        break;
                    }
                    i += 1;
                }
                Some(unescape_html(&source[val_start..i]))
            }
        } else {
            None
        };
        attrs.push((attr_name, attr_val));
    }
}

/// Parse an end tag `</name ...>` starting at `start`.
/// Returns `(tag_lower, next_byte_index)`.
fn parse_endtag(source: &str, start: usize) -> Option<(String, usize)> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    if start + 2 >= len || bytes[start] != b'<' || bytes[start + 1] != b'/' {
        return None;
    }
    let mut i = start + 2;
    while i < len && is_html_whitespace(bytes[i]) {
        i += 1;
    }
    let name_start = i;
    if i >= len || !is_tag_name_start(bytes[i]) {
        return None;
    }
    while i < len && is_tag_name_char(bytes[i]) {
        i += 1;
    }
    let tag = source[name_start..i].to_ascii_lowercase();
    while i < len && bytes[i] != b'>' {
        i += 1;
    }
    if i >= len {
        return None;
    }
    Some((tag, i + 1))
}

/// Stream HTML events from `source` into `handler`, matching Python's
/// `html.parser.HTMLParser(convert_charrefs=convert_charrefs)`.
pub(crate) fn parse_html<H: HtmlHandler>(source: &str, convert_charrefs: bool, handler: &mut H) {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let b2c = byte_to_char_map(source);
    let mut i = 0usize;
    let mut raw_text_tag: Option<String> = None;

    let mut data_buf = String::new();
    let mut data_start_char: Option<usize> = None;

    let flush_data = |buf: &mut String, start_char: &mut Option<usize>, h: &mut H| {
        if let Some(pos) = start_char.take()
            && !buf.is_empty()
        {
            if convert_charrefs {
                let decoded = unescape_html(buf);
                h.handle_data(&decoded, pos);
            } else {
                // Mirror Python HTMLParser(convert_charrefs=False): split data around char/entity refs.
                let mut rest = buf.as_str();
                let mut cur_char = pos;
                while !rest.is_empty() {
                    if let Some((ent_start, ent_end)) = find_next_entity_span(rest) {
                        if ent_start > 0 {
                            let prefix = &rest[..ent_start];
                            h.handle_data(prefix, cur_char);
                            cur_char += prefix.chars().count();
                        }
                        cur_char += rest[ent_start..ent_end].chars().count();
                        rest = &rest[ent_end..];
                    } else {
                        h.handle_data(rest, cur_char);
                        break;
                    }
                }
            }
            buf.clear();
        }
    };

    while i < len {
        if let Some(ref cdata_tag) = raw_text_tag {
            // Inside <script> or <style>: scan until matching </script> or </style>.
            if bytes[i] == b'<'
                && i + 2 < len
                && bytes[i + 1] == b'/'
                && let Some((end_tag, next_i)) = parse_endtag(source, i)
                && end_tag == *cdata_tag
            {
                if let Some(pos) = data_start_char.take()
                    && !data_buf.is_empty()
                {
                    handler.handle_data(&data_buf, pos);
                    data_buf.clear();
                }
                raw_text_tag = None;
                handler.handle_endtag(&end_tag, b2c[i], i);
                i = next_i;
                continue;
            }
            if data_start_char.is_none() {
                data_start_char = Some(b2c[i]);
            }
            let Some(ch) = source[i..].chars().next() else {
                break;
            };
            data_buf.push(ch);
            i += ch.len_utf8();
            continue;
        }

        if bytes[i] != b'<' {
            let chunk_start = i;
            if data_start_char.is_none() {
                data_start_char = Some(b2c[chunk_start]);
            }
            while i < len && bytes[i] != b'<' {
                i += 1;
            }
            data_buf.push_str(&source[chunk_start..i]);
            continue;
        }

        // Comment: <!-- ... -->
        if source[i..].starts_with("<!--") {
            flush_data(&mut data_buf, &mut data_start_char, handler);
            if let Some(rel) = source[i + 4..].find("-->") {
                i = i + 4 + rel + 3;
            } else {
                break;
            }
            continue;
        }

        // CDATA: <![CDATA[ ... ]]>
        if source[i..].starts_with("<![CDATA[") {
            flush_data(&mut data_buf, &mut data_start_char, handler);
            if let Some(rel) = source[i + 9..].find("]]>") {
                i = i + 9 + rel + 3;
            } else {
                break;
            }
            continue;
        }

        // Declaration: <! ... >
        if i + 1 < len && bytes[i + 1] == b'!' {
            flush_data(&mut data_buf, &mut data_start_char, handler);
            if let Some(rel) = source[i + 2..].find('>') {
                i = i + 2 + rel + 1;
            } else {
                break;
            }
            continue;
        }

        // Processing instruction: <? ... >
        if i + 1 < len && bytes[i + 1] == b'?' {
            flush_data(&mut data_buf, &mut data_start_char, handler);
            if let Some(rel) = source[i + 2..].find('>') {
                i = i + 2 + rel + 1;
            } else {
                break;
            }
            continue;
        }

        // End tag: </tag>
        if i + 1 < len && bytes[i + 1] == b'/' {
            if let Some((tag, next_i)) = parse_endtag(source, i) {
                flush_data(&mut data_buf, &mut data_start_char, handler);
                handler.handle_endtag(&tag, b2c[i], i);
                i = next_i;
                continue;
            }
            // Bogus comment `</...>` without valid tag name
            if let Some(rel) = source[i + 2..].find('>') {
                flush_data(&mut data_buf, &mut data_start_char, handler);
                i = i + 2 + rel + 1;
                continue;
            }
        }

        // Start tag: <tag ...>
        if let Some(parsed) = parse_starttag(source, i) {
            flush_data(&mut data_buf, &mut data_start_char, handler);
            let char_pos = b2c[i];
            handler.handle_starttag(&parsed.tag, &parsed.attrs, char_pos);
            if parsed.self_closing {
                handler.handle_endtag(&parsed.tag, char_pos, i);
            } else if matches!(parsed.tag.as_str(), "script" | "style") {
                raw_text_tag = Some(parsed.tag);
            }
            i = parsed.next_byte;
            continue;
        }

        // Stray `<`: treat as literal text data.
        if data_start_char.is_none() {
            data_start_char = Some(b2c[i]);
        }
        data_buf.push('<');
        i += 1;
    }

    if raw_text_tag.is_some() {
        if let Some(pos) = data_start_char.take()
            && !data_buf.is_empty()
        {
            handler.handle_data(&data_buf, pos);
            data_buf.clear();
        }
    } else {
        flush_data(&mut data_buf, &mut data_start_char, handler);
    }
}

fn hidden_style_regex() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)(?:display\s*:\s*none|visibility\s*:\s*hidden)").ok())
        .as_ref()
}

/// Collapse runs of Unicode whitespace into single ASCII spaces and trim,
/// matching Python `re.sub(r"\s+", " ", s).strip()`.
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !out.is_empty() {
                in_ws = true;
            }
        } else {
            if in_ws {
                out.push(' ');
                in_ws = false;
            }
            out.push(ch);
        }
    }
    out
}

/// Build a char-index -> UTF-8 byte-index lookup table of length `char_count + 1`.
fn char_to_byte_map(text: &str) -> Vec<usize> {
    let mut map: Vec<usize> = text.char_indices().map(|(bi, _)| bi).collect();
    map.push(text.len());
    map
}

struct TablesParser<'a> {
    source: &'a str,
    c2b: Vec<usize>,
    tables: Vec<Vec<HtmlRow>>,
    table: Option<Vec<HtmlRow>>,
    row: Option<Vec<String>>,
    cell: Option<Vec<String>>,
    cell_tag: Option<String>,
    cell_parts: Option<Vec<(String, usize)>>,
    row_spans: Option<Vec<Option<[usize; 2]>>>,
    row_start: Option<usize>,
    depth: usize,
    invalid: bool,
}

impl<'a> TablesParser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            c2b: char_to_byte_map(source),
            tables: Vec::new(),
            table: None,
            row: None,
            cell: None,
            cell_tag: None,
            cell_parts: None,
            row_spans: None,
            row_start: None,
            depth: 0,
            invalid: false,
        }
    }

    fn source_starts_with_at_char(&self, value: &str, char_pos: usize) -> bool {
        let Some(&byte_pos) = self.c2b.get(char_pos) else {
            return false;
        };
        self.source
            .get(byte_pos..)
            .is_some_and(|tail| tail.starts_with(value))
    }
}

impl HtmlHandler for TablesParser<'_> {
    fn handle_starttag(&mut self, tag: &str, attrs: &[(String, Option<String>)], char_pos: usize) {
        if tag == "table" {
            self.depth += 1;
            if self.depth != 1 {
                self.invalid = true;
            } else {
                self.table = Some(Vec::new());
            }
        }
        if self.depth == 0 {
            return;
        }
        let mut attr_map: HashMap<&str, Option<&str>> = HashMap::with_capacity(attrs.len());
        for (k, v) in attrs {
            attr_map.insert(k.as_str(), v.as_deref());
        }
        let aria_hidden = attr_map
            .get("aria-hidden")
            .and_then(|v| *v)
            .is_some_and(|v| v.eq_ignore_ascii_case("true"));
        let style_hidden = attr_map
            .get("style")
            .and_then(|v| *v)
            .is_some_and(|v| hidden_style_regex().is_some_and(|re| re.is_match(v)));
        if attr_map.contains_key("hidden")
            || aria_hidden
            || style_hidden
            || matches!(tag, "script" | "style")
        {
            self.invalid = true;
        }
        if tag == "tr" {
            if self.row.is_some() || self.cell.is_some() {
                self.invalid = true;
            }
            self.row = Some(Vec::new());
            self.row_start = Some(char_pos);
            self.row_spans = Some(Vec::new());
        } else if matches!(tag, "td" | "th") {
            if self.row.is_none() || self.cell.is_some() {
                self.invalid = true;
            }
            let bad_rowspan = attr_map.get("rowspan").is_some_and(|v| *v != Some("1"));
            let bad_colspan = attr_map.get("colspan").is_some_and(|v| *v != Some("1"));
            if bad_rowspan || bad_colspan {
                self.invalid = true;
            }
            self.cell = Some(Vec::new());
            self.cell_tag = Some(tag.to_string());
            self.cell_parts = Some(Vec::new());
        } else if matches!(tag, "br" | "p" | "div" | "ul" | "ol" | "li")
            && let Some(ref mut cell) = self.cell
        {
            cell.push(" ".to_string());
        }
    }

    fn handle_data(&mut self, data: &str, char_pos: usize) {
        if let Some(ref mut cell) = self.cell {
            cell.push(data.to_string());
            if let Some(ref mut parts) = self.cell_parts {
                parts.push((data.to_string(), char_pos));
            }
        } else if self.row.is_some() && !data.trim().is_empty() {
            self.invalid = true;
        }
    }

    fn handle_endtag(&mut self, tag: &str, _char_pos: usize, byte_pos: usize) {
        if self.depth == 0 {
            return;
        }
        if matches!(tag, "td" | "th") {
            if self.cell.is_none() || self.row.is_none() || self.cell_tag.as_deref() != Some(tag) {
                self.invalid = true;
            } else if let (Some(cell), Some(parts)) = (self.cell.as_ref(), self.cell_parts.as_ref())
            {
                let joined: String = cell.concat();
                let value = collapse_whitespace(&joined);
                let val_char_len = value.chars().count();
                let mut matches: Vec<usize> = Vec::new();
                if !value.is_empty() {
                    for (data, position) in parts {
                        let mut iter = data.match_indices(&value);
                        if let Some((first_byte, _)) = iter.next()
                            && iter.next().is_none()
                        {
                            let rel_chars = data[..first_byte].chars().count();
                            let candidate = position + rel_chars;
                            if self.source_starts_with_at_char(&value, candidate) {
                                matches.push(candidate);
                            }
                        }
                    }
                }
                let span = if matches.len() == 1 {
                    let start = matches[0];
                    Some([start, start + val_char_len])
                } else {
                    None
                };
                if let (Some(row), Some(row_spans)) = (self.row.as_mut(), self.row_spans.as_mut()) {
                    row.push(value);
                    row_spans.push(span);
                }
            }
            self.cell = None;
            self.cell_tag = None;
            self.cell_parts = None;
        } else if tag == "tr" {
            if self.row.is_none() || self.cell.is_some() || self.table.is_none() {
                self.invalid = true;
            } else if let (Some(cells), Some(row_start), Some(spans), Some(table)) = (
                self.row.take(),
                self.row_start.take(),
                self.row_spans.take(),
                self.table.as_mut(),
            ) {
                let start_byte = self.c2b.get(row_start).copied().unwrap_or(0);
                let end_byte = self
                    .source
                    .get(byte_pos..)
                    .and_then(|s| s.find('>'))
                    .map(|rel| byte_pos + rel + 1)
                    .unwrap_or(self.source.len());
                let line = self
                    .source
                    .get(start_byte..end_byte)
                    .unwrap_or("")
                    .to_string();
                table.push(HtmlRow {
                    cells,
                    line,
                    offset: row_start,
                    spans,
                });
            }
            self.row = None;
            self.row_start = None;
            self.row_spans = None;
        } else if tag == "table" {
            if self.depth == 1 {
                if self.row.is_some() || self.cell.is_some() {
                    self.invalid = true;
                }
                if let Some(table) = self.table.take() {
                    self.tables.push(table);
                }
            }
            self.depth -= 1;
        } else if matches!(tag, "p" | "div" | "ul" | "ol" | "li")
            && let Some(ref mut cell) = self.cell
        {
            cell.push(" ".to_string());
        }
    }
}

/// Parse all top-level HTML `<table>` elements in `source`, matching
/// `html_rows.py::tables`. Returns `None` if any table is malformed, nested,
/// hidden, spanned (`rowspan`/`colspan != 1`), or contains `<script>`/`<style>`.
pub fn tables(source: &str) -> Option<Vec<Vec<HtmlRow>>> {
    let mut parser = TablesParser::new(source);
    parse_html(source, true, &mut parser);
    if parser.invalid || parser.depth != 0 || parser.row.is_some() || parser.cell.is_some() {
        return None;
    }
    Some(parser.tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_table_with_exact_char_offsets() {
        let html = "Intro café\n<table><tr><th>Item</th><th>Qty</th></tr><tr><td>Widget A</td><td>12</td></tr></table>";
        let parsed = tables(html).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].len(), 2);

        let header = &parsed[0][0];
        assert_eq!(header.cells, vec!["Item", "Qty"]);
        let chars: Vec<char> = html.chars().collect();
        let row0_chars: String = chars[header.offset..header.offset + header.line.chars().count()]
            .iter()
            .collect();
        assert_eq!(row0_chars, "<tr><th>Item</th><th>Qty</th></tr>");
        assert_eq!(header.line, "<tr><th>Item</th><th>Qty</th></tr>");

        let s0 = header.spans[0].unwrap();
        let extracted0: String = chars[s0[0]..s0[1]].iter().collect();
        assert_eq!(extracted0, "Item");

        let body = &parsed[0][1];
        assert_eq!(body.cells, vec!["Widget A", "12"]);
        let w_span = body.spans[0].unwrap();
        let w_extracted: String = chars[w_span[0]..w_span[1]].iter().collect();
        assert_eq!(w_extracted, "Widget A");
        let q_span = body.spans[1].unwrap();
        let q_extracted: String = chars[q_span[0]..q_span[1]].iter().collect();
        assert_eq!(q_extracted, "12");
    }

    #[test]
    fn decodes_html_entities_and_clears_span_when_entity_differs_from_source() {
        let html = "<table><tr><td>AT&amp;T &lt;Corp&gt; &quot;A&quot; &#39;B&#39; &#x20AC;10&nbsp;net</td><td>&#65;</td></tr></table>";
        let parsed = tables(html).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].len(), 1);
        // NBSP is collapsed by `\s+` into a single ASCII space just like Python `re.sub(r"\s+", " ", ...)`.
        assert_eq!(
            parsed[0][0].cells,
            vec!["AT&T <Corp> \"A\" 'B' \u{20AC}10 net", "A"]
        );
        // Neither cell matches source literally because of entity encoding -> spans are None.
        assert_eq!(parsed[0][0].spans, vec![None, None]);
    }

    #[test]
    fn handles_block_tags_inside_cells_as_whitespace() {
        let html = "<table><tr><td>Line 1<br/>Line 2<div>Line 3</div></td><td>42</td></tr></table>";
        let parsed = tables(html).unwrap();
        assert_eq!(parsed[0][0].cells, vec!["Line 1 Line 2 Line 3", "42"]);
        assert_eq!(parsed[0][0].spans[0], None);
        assert!(parsed[0][0].spans[1].is_some());
    }

    #[test]
    fn rejects_nested_hidden_spanned_and_malformed_tables() {
        // Nested table
        assert!(
            tables("<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>").is_none()
        );
        // Hidden attribute
        assert!(tables("<table><tr hidden><td>x</td></tr></table>").is_none());
        // aria-hidden="true"
        assert!(tables("<table><tr aria-hidden=\"TRUE\"><td>x</td></tr></table>").is_none());
        // display: none
        assert!(
            tables("<table><tr style=\"color:red; display : none\"><td>x</td></tr></table>")
                .is_none()
        );
        // visibility: hidden
        assert!(tables("<table><td style=\"visibility:hidden\">x</td></table>").is_none());
        // script inside table
        assert!(tables("<table><tr><td><script>alert(1)</script></td></tr></table>").is_none());
        // colspan != 1
        assert!(tables("<table><tr><td colspan=\"2\">x</td></tr></table>").is_none());
        // rowspan != 1
        assert!(tables("<table><tr><td rowspan=\"2\">x</td></tr></table>").is_none());
        // rowspan=1 & colspan=1 is allowed
        assert!(
            tables("<table><tr><td rowspan=\"1\" colspan=\"1\">ok</td></tr></table>").is_some()
        );
        // Mismatched td/th
        assert!(tables("<table><tr><td>x</th></tr></table>").is_none());
        // Text directly inside tr outside cell
        assert!(tables("<table><tr>stray<td>x</td></tr></table>").is_none());
        // Unclosed table
        assert!(tables("<table><tr><td>x</td></tr>").is_none());
    }

    #[test]
    fn ignores_script_outside_table_and_handles_multiple_tables_and_win1252_entities() {
        let html = "<script>var s = '<table><tr><td>fake';</script>\
                    <table><tr><td>&#128;50 &#8212; ok</td><td/></tr></table>\
                    <table><tr><th>K</th></tr><tr><td>V</td></tr></table>";
        let parsed = tables(html).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0][0].cells, vec!["\u{20AC}50 \u{2014} ok", ""]);
        assert_eq!(parsed[1].len(), 2);
        assert_eq!(parsed[1][0].cells, vec!["K"]);
        assert_eq!(parsed[1][1].cells, vec!["V"]);
    }
}
