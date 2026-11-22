//! Structural table row counting and source-row list bounding.
//!
//! Direct port of upstream `clairic_runtime/source_rows.py`.

use super::html_rows::{HtmlHandler, parse_html};
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;

fn one_per_row_regex() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\bone\s+(?:[\w-]+\s+){0,5}per\s+(?:[\w-]+\s+){0,5}row\b").ok()
    })
    .as_ref()
}

fn negative_row_regex() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:not|never|multiple|two|three)\b|don't").ok())
        .as_ref()
}

fn table_tag_regex() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)<table(?:\s|>)").ok())
        .as_ref()
}

fn digit_regex() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\d").ok()).as_ref()
}

/// Split lines matching Python's `str.splitlines()` (without keeping line endings).
fn py_splitlines(s: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let bytes = s.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let Some(ch) = s[i..].chars().next() else {
            break;
        };
        let is_break = matches!(
            ch,
            '\n' | '\r'
                | '\u{000B}'
                | '\u{000C}'
                | '\u{001C}'
                | '\u{001D}'
                | '\u{001E}'
                | '\u{0085}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_break {
            lines.push(&s[start..i]);
            i += ch.len_utf8();
            if ch == '\r' && i < bytes.len() && bytes[i] == b'\n' {
                i += 1;
            }
            start = i;
        } else {
            i += ch.len_utf8();
        }
    }
    if start < s.len() {
        lines.push(&s[start..]);
    }
    lines
}

/// Streaming HTML table structure checker mirroring `source_rows.py::_Table`
/// (`HTMLParser(convert_charrefs=False)`).
struct TableStructureParser {
    tables: usize,
    depth: i32,
    rows: Vec<HashSet<String>>,
    closed: bool,
    row_header: bool,
    in_cell: bool,
    ambiguous: bool,
}

impl TableStructureParser {
    fn new() -> Self {
        Self {
            tables: 0,
            depth: 0,
            rows: Vec::new(),
            closed: false,
            row_header: false,
            in_cell: false,
            ambiguous: false,
        }
    }
}

impl HtmlHandler for TableStructureParser {
    fn handle_starttag(&mut self, tag: &str, attrs: &[(String, Option<String>)], _char_pos: usize) {
        if self.in_cell && matches!(tag, "br" | "div" | "p" | "ul" | "ol" | "li" | "table") {
            self.ambiguous = true;
        }
        if tag == "table" {
            self.tables += 1;
            self.depth += 1;
        }
        if self.depth == 1 {
            if tag == "tr" {
                self.rows.push(HashSet::new());
            } else if matches!(tag, "td" | "th")
                && let Some(last_row) = self.rows.last_mut()
            {
                self.in_cell = true;
                if attrs.iter().any(|(k, v)| {
                    matches!(k.as_str(), "rowspan" | "colspan") && v.as_deref() != Some("1")
                }) {
                    self.ambiguous = true;
                }
                last_row.insert(tag.to_string());
                if self.rows.len() == 1 {
                    let scope = attrs
                        .iter()
                        .rev()
                        .find(|(k, _)| k == "scope")
                        .and_then(|(_, v)| v.as_deref())
                        .unwrap_or("");
                    if scope.eq_ignore_ascii_case("row") || scope.eq_ignore_ascii_case("rowgroup") {
                        self.row_header = true;
                    }
                }
            }
        }
    }

    fn handle_endtag(&mut self, tag: &str, _char_pos: usize, _byte_pos: usize) {
        if matches!(tag, "td" | "th") {
            self.in_cell = false;
        }
        if tag == "table" {
            self.depth -= 1;
            self.closed = self.depth == 0;
        }
    }

    fn handle_data(&mut self, data: &str, _char_pos: usize) {
        if self.in_cell
            && py_splitlines(data)
                .into_iter()
                .filter(|line| !line.trim().is_empty())
                .count()
                > 1
        {
            self.ambiguous = true;
        }
    }
}

/// Strict RFC-4180 / Python `csv.reader(io.StringIO(source), delimiter=d, strict=True)` parser.
///
/// Mirrors CPython `_csv.c` state machine with default Excel dialect (`quotechar='"'`,
/// `doublequote=True`, `strict=True`). Returns `None` on any `csv.Error`.
pub fn parse_csv_strict(source: &str, delimiter: char) -> Option<Vec<Vec<String>>> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        StartRecord,
        StartField,
        InField,
        InQuotedField,
        AfterQuotedField,
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut state = State::StartRecord;

    let mut chars = source.chars().peekable();
    while let Some(ch) = chars.next() {
        match state {
            State::StartRecord => {
                if ch == '\r' || ch == '\n' {
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    // Empty line -> empty row `[]`, matching Python csv.reader.
                    rows.push(Vec::new());
                    continue;
                }
                if ch == '"' {
                    state = State::InQuotedField;
                } else if ch == delimiter {
                    row.push(String::new());
                    state = State::StartField;
                } else {
                    field.push(ch);
                    state = State::InField;
                }
            }
            State::StartField => {
                if ch == '\r' || ch == '\n' {
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                    state = State::StartRecord;
                } else if ch == '"' {
                    state = State::InQuotedField;
                } else if ch == delimiter {
                    row.push(std::mem::take(&mut field));
                    state = State::StartField;
                } else {
                    field.push(ch);
                    state = State::InField;
                }
            }
            State::InField => {
                if ch == '\r' || ch == '\n' {
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                    state = State::StartRecord;
                } else if ch == delimiter {
                    row.push(std::mem::take(&mut field));
                    state = State::StartField;
                } else {
                    field.push(ch);
                }
            }
            State::InQuotedField => {
                if ch == '"' {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        field.push('"');
                        state = State::InQuotedField;
                    } else {
                        state = State::AfterQuotedField;
                    }
                } else {
                    field.push(ch);
                }
            }
            State::AfterQuotedField => {
                if ch == delimiter {
                    row.push(std::mem::take(&mut field));
                    state = State::StartField;
                } else if ch == '\r' || ch == '\n' {
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                    state = State::StartRecord;
                } else {
                    // Under strict=True, any non-delimiter/non-EOL char after closing quote is csv.Error.
                    return None;
                }
            }
        }
    }

    match state {
        State::StartRecord => {}
        State::StartField | State::InField | State::AfterQuotedField => {
            row.push(field);
            rows.push(row);
        }
        State::InQuotedField => {
            // Unclosed quoted field -> csv.Error ("unexpected end of data").
            return None;
        }
    }
    Some(rows)
}

/// Count data rows in an unambiguous HTML or delimited table (`source_rows.py::table_rows`).
///
/// Returns `Some(data_row_count)` (excluding the header row) or `None` when the
/// source is not a single unambiguous structured table.
pub fn table_rows(source: &str) -> Option<usize> {
    if table_tag_regex().is_some_and(|re| re.is_match(source)) {
        let mut parser = TableStructureParser::new();
        parse_html(source, false, &mut parser);
        let first_row_all_th = parser
            .rows
            .first()
            .is_some_and(|tags| tags.len() == 1 && tags.contains("th"));
        if parser.tables != 1
            || !parser.closed
            || parser.depth != 0
            || parser.rows.is_empty()
            || !first_row_all_th
            || parser.row_header
            || parser.ambiguous
        {
            return None;
        }
        return Some(parser.rows.len() - 1);
    }

    let lines = py_splitlines(source);
    if lines.len() < 2 {
        return None;
    }
    let digit_re = digit_regex()?;
    for delimiter in [',', '\t', ';', '|'] {
        if !lines[0].contains(delimiter) {
            continue;
        }
        let Some(rows) = parse_csv_strict(source, delimiter) else {
            continue;
        };
        let Some(header) = rows.first() else {
            continue;
        };
        let col_count = header.len();
        if col_count < 2 {
            continue;
        }
        if rows
            .iter()
            .any(|row| row.is_empty() || row.len() != col_count)
        {
            continue;
        }
        if header
            .iter()
            .any(|cell| cell.trim().is_empty() || digit_re.is_match(cell))
        {
            continue;
        }
        let body_has_digit = rows[1..]
            .iter()
            .flat_map(|row| row.iter())
            .any(|cell| digit_re.is_match(cell));
        if !body_has_digit {
            continue;
        }
        return Some(rows.len() - 1);
    }
    None
}

/// Compute structural row count bounds for eligible scalar list fields (`source_rows.py::list_bounds`).
pub fn list_bounds(
    source: &str,
    fields: &Map<String, Value>,
    has_images: bool,
) -> BTreeMap<String, usize> {
    if has_images {
        return BTreeMap::new();
    }
    let Some(one_re) = one_per_row_regex() else {
        return BTreeMap::new();
    };
    let Some(neg_re) = negative_row_regex() else {
        return BTreeMap::new();
    };
    let mut parents: Vec<String> = Vec::new();
    for (name, spec) in fields {
        let instructions = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        let is_scalar_list = spec.get("type").and_then(Value::as_str) == Some("list")
            && matches!(
                spec.get("items").and_then(Value::as_str),
                Some("text" | "number")
            );
        let knowledge = spec
            .get("knowledge")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if is_scalar_list
            && !knowledge
            && one_re.is_match(instructions)
            && !neg_re.is_match(instructions)
        {
            parents.push(name.clone());
        }
    }
    if parents.is_empty() {
        return BTreeMap::new();
    }
    let Some(count) = table_rows(source) else {
        return BTreeMap::new();
    };
    parents.into_iter().map(|name| (name, count)).collect()
}

/// Force the `(count + 1)`-th item prediction of each bounded list to a
/// structurally-grounded `null` terminator (`source_rows.py::bounded_predictions`).
pub fn bounded_predictions<F>(
    predictions: &Map<String, Value>,
    fields: &Map<String, Value>,
    bounds: &BTreeMap<String, usize>,
    item_name: F,
) -> Map<String, Value>
where
    F: Fn(&str, usize) -> String,
{
    if bounds.is_empty() {
        return predictions.clone();
    }
    let mut result = predictions.clone();
    for (parent, &count) in bounds {
        let name = item_name(parent, count + 1);
        if !fields.contains_key(&name) || !result.contains_key(&name) {
            continue;
        }
        let Some(previous) = result.get(&name).and_then(Value::as_object) else {
            continue;
        };
        let is_null = previous.get("value").is_none_or(Value::is_null);
        let is_valid = previous
            .get("valid")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if is_null && is_valid {
            continue;
        }
        let mut updated = previous.clone();
        updated.insert("value".to_string(), Value::Null);
        updated.insert("valid".to_string(), Value::Bool(true));
        updated.insert("complete".to_string(), Value::Bool(true));
        updated.insert("span".to_string(), Value::Null);
        updated.insert("confidence".to_string(), json!(1.0));
        updated.insert("log_probability".to_string(), json!(0.0));
        updated.insert(
            "confidence_method".to_string(),
            Value::String("source_structure".to_string()),
        );
        updated.insert(
            "grounding_method".to_string(),
            Value::String("source_row_bound".to_string()),
        );
        result.insert(name, Value::Object(updated));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::clairic::schema::list_item_name;

    #[test]
    fn counts_csv_tsv_semicolon_and_pipe_tables() {
        assert_eq!(
            table_rows("item,qty,price\nApple,2,1.50\nBanana,5,0.75\n"),
            Some(2)
        );
        assert_eq!(table_rows("item\tqty\nWidget\t10\nGadget\t20"), Some(2));
        assert_eq!(table_rows("name;amount\nFee;5.00"), Some(1));
        assert_eq!(
            table_rows("col_a | col_b\nAlpha | 100\nBeta | 200"),
            Some(2)
        );
        // Quoted field with embedded comma
        assert_eq!(
            table_rows("item,price\n\"Widget, Pro\",19.99\nBolt,2.50"),
            Some(2)
        );
    }

    #[test]
    fn rejects_invalid_or_ambiguous_delimited_tables() {
        // Strict CSV error: unescaped quote in middle after closing quote
        assert_eq!(table_rows("a,b\n\"bad\"extra,1"), None);
        // Unclosed quote
        assert_eq!(table_rows("a,b\n\"unclosed,1"), None);
        // Digit in header
        assert_eq!(table_rows("col1,col2\nfoo,2"), None);
        // No digits in body
        assert_eq!(table_rows("a,b\nfoo,bar"), None);
        // Ragged rows
        assert_eq!(table_rows("a,b\n1,2\n3"), None);
        // Leading/trailing empty pipe cells (Markdown outer pipes) -> empty header cell rejected
        assert_eq!(
            table_rows("| Item | Qty |\n| --- | --- |\n| Widget | 3 |"),
            None
        );
        // Single line
        assert_eq!(table_rows("a,b"), None);
    }

    #[test]
    fn counts_html_tables_and_rejects_ambiguous_structures() {
        let ok = "<table><tr><th>Item</th><th>Qty</th></tr><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table>";
        assert_eq!(table_rows(ok), Some(2));

        // Multiple tables rejected
        let two = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table><table><tr><th>B</th></tr><tr><td>2</td></tr></table>";
        assert_eq!(table_rows(two), None);

        // Header not exclusively <th>
        let bad_hdr = "<table><tr><th>A</th><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>";
        assert_eq!(table_rows(bad_hdr), None);

        // Row header scope="row" rejected
        let row_hdr = "<table><tr><th scope=\"row\">A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>";
        assert_eq!(table_rows(row_hdr), None);

        // Cell with <br> rejected as ambiguous
        let br_cell =
            "<table><tr><th>A</th><th>B</th></tr><tr><td>x<br>y</td><td>1</td></tr></table>";
        assert_eq!(table_rows(br_cell), None);

        // Multi-line data inside cell rejected as ambiguous
        let multiline =
            "<table><tr><th>A</th><th>B</th></tr><tr><td>line1\nline2</td><td>1</td></tr></table>";
        assert_eq!(table_rows(multiline), None);

        // colspan != 1 rejected
        let spanned =
            "<table><tr><th>A</th><th>B</th></tr><tr><td colspan=\"2\">1</td></tr></table>";
        assert_eq!(table_rows(spanned), None);
    }

    #[test]
    fn list_bounds_and_bounded_predictions() {
        let fields_val = json!({
            "amounts": {
                "type": "list",
                "items": "number",
                "instructions": "Extract one amount per table row."
            },
            "amounts.item3": {"type": "number"},
            "excluded": {
                "type": "list",
                "items": "text",
                "instructions": "Do not return one entry per row."
            },
            "known": {
                "type": "list",
                "items": "text",
                "knowledge": true,
                "instructions": "Return one item per row."
            }
        });
        let fields = fields_val.as_object().unwrap();
        let csv = "item,amount\nWidget,10\nBolt,20\n";

        // Disabled when has_images == true
        assert!(list_bounds(csv, fields, true).is_empty());

        let bounds = list_bounds(csv, fields, false);
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds.get("amounts"), Some(&2));

        let preds_val = json!({
            "amounts.item3": {
                "value": 99,
                "valid": true,
                "complete": true,
                "confidence": 0.6,
                "log_probability": -0.5
            }
        });
        let bounded = bounded_predictions(
            preds_val.as_object().unwrap(),
            fields,
            &bounds,
            list_item_name,
        );
        let item3 = &bounded["amounts.item3"];
        assert!(item3["value"].is_null());
        assert_eq!(item3["valid"], json!(true));
        assert_eq!(item3["complete"], json!(true));
        assert_eq!(item3["confidence"], json!(1.0));
        assert_eq!(item3["log_probability"], json!(0.0));
        assert_eq!(item3["confidence_method"], json!("source_structure"));
        assert_eq!(item3["grounding_method"], json!("source_row_bound"));

        // Already-null valid prediction is preserved untouched
        let already_null = json!({
            "amounts.item3": {
                "value": null,
                "valid": true,
                "complete": true,
                "confidence": 0.91,
                "confidence_method": "ar_sequence_probability"
            }
        });
        let kept = bounded_predictions(
            already_null.as_object().unwrap(),
            fields,
            &bounds,
            list_item_name,
        );
        assert_eq!(kept["amounts.item3"]["confidence"], json!(0.91));
        assert_eq!(
            kept["amounts.item3"]["confidence_method"],
            json!("ar_sequence_probability")
        );
    }
}
