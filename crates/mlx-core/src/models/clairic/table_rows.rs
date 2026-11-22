//! Structured text/HTML table row extraction, column matching, wrapped-row
//! merging, eligibility/local-scope prompt rewriting, and list assembly.
//!
//! Direct port of upstream `clairic_runtime/table_rows.py`.

use super::{
    ar, errors::invalid_request, html_rows, json_validate, printed_numbers, schema::MAX_LIST_ITEMS,
    source_rows,
};
use fancy_regex::Regex as FancyRegex;
use napi::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

/// Synthetic input text substituted for locally scoped row-eligibility queries
/// (`table_rows.LOCAL_SCOPE_INPUT`).
pub const LOCAL_SCOPE_INPUT: &str =
    "The row to classify is provided in the requested field instructions.";

/// Internal field spec marker indicating physical-row scoping (`_physical_row_scope`).
pub const PHYSICAL_ROW_SCOPE_KEY: &str = "_physical_row_scope";

const SCOPE_PRIORITY_INSTRUCTION: &str = "Evaluate exclusions first for this exact row; an exclusion overrides a general category match. ";
const SCOPE_FEE_INSTRUCTION: &str = "Include only a separately printed fee or surcharge. A billed product or service charge, tax, total or payment is not a fee merely because it has a monetary amount. The row label must explicitly establish that it is a fee or surcharge. ";
const SCOPE_TAX_INSTRUCTION: &str = "Include only an explicitly identified tax, such as sales tax, VAT or GST. A product, service charge, fee, total or payment is not a tax merely because it has an amount. ";
const SCOPE_CHARGE_INSTRUCTION: &str = "Include only billed goods or service charges. A separately identified tax, fee, surcharge, payment, subtotal or total is excluded, even though it is a monetary charge. ";

fn is_false(v: &bool) -> bool {
    !*v
}

/// One parsed physical table row with exact Unicode character spans and optional
/// eligibility / local-scope metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRow {
    pub line: String,
    pub values: Map<String, Value>,
    pub spans: BTreeMap<String, Option<[usize; 2]>>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub eligibility_only: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_instruction: Option<String>,
}

impl SourceRow {
    /// Convert `spans` into a JSON object mapping each property name to `[start, end]` or `null`.
    pub fn spans_value(&self) -> Value {
        let mut obj = Map::with_capacity(self.spans.len());
        for (name, span) in &self.spans {
            let v = match span {
                Some([start, end]) => json!([*start, *end]),
                None => Value::Null,
            };
            obj.insert(name.clone(), v);
        }
        Value::Object(obj)
    }
}

fn non_word_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[\W_]+").unwrap_or_else(|_| unreachable!()))
}

fn wrapped_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:wrapped|multiline|multi-line|continuation)\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn joining_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:join|combine|merge)\b").unwrap_or_else(|_| unreachable!())
    })
}

fn negative_wrapped_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:not|no|never|without|separate\w*|independent\w*)\b|\b\w+n['\u{2019}]t\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn positive_wrapped_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:join|combine)\b[^.;\n]*\b(?:wrapped|multiline|multi-line|continuation)\b[^.;\n]*\b(?:descriptions?|lines?|text)\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn html_table_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)<table\b").unwrap_or_else(|_| unreachable!()))
}

fn md_separator_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\A:?-{3,}:?\z").unwrap_or_else(|_| unreachable!()))
}

fn scope_split_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)[.\n]|\bExclude\b").unwrap_or_else(|_| unreachable!()))
}

fn role_fee_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\bfees?\b").unwrap_or_else(|_| unreachable!()))
}

fn role_tax_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\btax(?:es)?\b").unwrap_or_else(|_| unreachable!()))
}

fn role_charge_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:charges?|goods|products?|services?)\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn only_rows_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\A\s*Only\b[^.\n]*\brows\b").unwrap_or_else(|_| unreachable!())
    })
}

fn disallowed_scoped_op_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:summar\w*|calculat\w*|combin\w*|group\w*|aggregat\w*|deduplicat\w*)\b|\buse\b[^.\n]*\bas\b",
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn exact_label_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:exact|verbatim)\b[^.\n]*\b(?:printed|row)\b[^.\n]*\blabel\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn explicit_rows_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\bone\s+(?:object|entry|record)\s+(?:per|for\s+each)\b[^.\n]*\brow\b")
            .unwrap_or_else(|_| unreachable!())
    })
}

fn membership_guard_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(
            r#"(?i)[\r\n`"\u{201c}\u{201d}]|(?<!\w)'[^'\r\n]+'(?!\w)|\u{2018}[^\u{2019}]+\u{2019}|\b(?:literal|verbatim|quoted|quote|quotation|code|snippet|block|note|label|phrase)\b"#,
        )
        .unwrap_or_else(|_| unreachable!())
    })
}

fn negative_join_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let target = r"(?:(?:wrapped|multiline|multi-line|continuation)\s+)?(?:descriptions?(?:\s+lines?)?|lines?|(?:distinct|separate|duplicate|repeated|physical)\s+rows?)";
        let pat = format!(
            r"(?i)\A\s*(?:do\s+not|don't|never)\s+(?:join|combine|merge)\s+{target}(?:\s+or\s+(?:join|combine|merge)\s+{target})*\s*[.;]?\s*\z"
        );
        Regex::new(&pat).unwrap_or_else(|_| unreachable!())
    })
}

fn clause_split_regex() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(r"(?<=[.;])(?=\s|\z)|(?<=\n)").unwrap_or_else(|_| unreachable!())
    })
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

/// Casefold a string matching Python's `str.casefold()` for common Unicode cases.
fn casefold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'ß' | 'ẞ' => out.push_str("ss"),
            _ => {
                for lc in ch.to_lowercase() {
                    out.push(lc);
                }
            }
        }
    }
    out
}

/// Normalize a column header or property name (`table_rows.py::_column`).
fn normalize_column(value: &str) -> String {
    non_word_regex()
        .replace_all(&casefold(value), "")
        .into_owned()
}

/// Map header cell texts to canonical property keys, aliasing `qty` -> `quantity`
/// when `quantity` is defined and `qty` is not (`table_rows.py::_header_keys`).
fn header_keys(cells: &[String], names: &HashMap<String, String>) -> Vec<String> {
    cells
        .iter()
        .map(|cell| {
            let key = normalize_column(cell);
            if key == "qty" && !names.contains_key("qty") && names.contains_key("quantity") {
                "quantity".to_string()
            } else {
                key
            }
        })
        .collect()
}

/// Split lines keeping line endings, matching Python `str.splitlines(keepends=True)`.
fn py_splitlines_keepends(s: &str) -> Vec<&str> {
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
            i += ch.len_utf8();
            if ch == '\r' && i < bytes.len() && bytes[i] == b'\n' {
                i += 1;
            }
            lines.push(&s[start..i]);
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

/// Parse one line using strict RFC-4180 CSV rules (`next(csv.reader([line], delimiter=d, strict=True))`).
fn parse_csv_line(line: &str, delimiter: char) -> Option<Vec<String>> {
    let mut rows = source_rows::parse_csv_strict(line, delimiter)?;
    if rows.len() == 1 { rows.pop() } else { None }
}

/// Python-equivalent `haystack.find(needle)` returning Unicode char offset (or `-1` if absent).
fn py_find_char(haystack: &str, needle: &str) -> isize {
    match haystack.find(needle) {
        Some(byte_idx) => haystack[..byte_idx].chars().count() as isize,
        None => -1,
    }
}

/// Split `text` using a `fancy_regex::Regex`, correctly advancing across zero-width matches
/// (matching Python's `re.split`).
fn fancy_split<'a>(re: &FancyRegex, text: &'a str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut last_split = 0usize;
    let mut search_pos = 0usize;
    let len = text.len();

    while search_pos <= len {
        match re.find_from_pos(text, search_pos) {
            Ok(Some(m)) => {
                out.push(&text[last_split..m.start()]);
                last_split = m.end();
                if m.start() == m.end() {
                    if m.end() >= len {
                        break;
                    }
                    let step = text[m.end()..].chars().next().map_or(1, char::len_utf8);
                    search_pos = m.end() + step;
                } else {
                    search_pos = m.end();
                }
            }
            _ => break,
        }
    }
    out.push(&text[last_split..]);
    out
}

/// Return `true` iff `instructions` explicitly asks to join/combine wrapped or
/// multiline descriptions without any negating clause (`table_rows.py::join_wrapped_descriptions`).
pub fn join_wrapped_descriptions(instructions: &str) -> bool {
    let mut positive = false;
    for clause in instructions.split(['.', ';', '\n']) {
        let wrapped = wrapped_regex().is_match(clause);
        let joining = joining_regex().is_match(clause);
        let negative = negative_wrapped_regex().is_match(clause);
        if negative && (wrapped || joining) {
            return false;
        }
        positive |= positive_wrapped_regex().is_match(clause);
    }
    positive
}

fn wrapped_rows(rows: Vec<SourceRow>, properties: &Map<String, Value>) -> Option<Vec<SourceRow>> {
    let texts: Vec<&String> = properties
        .iter()
        .filter(|(_, spec)| spec.get("type").and_then(Value::as_str) == Some("text"))
        .map(|(name, _)| name)
        .collect();
    let numbers: Vec<&String> = properties
        .iter()
        .filter(|(_, spec)| spec.get("type").and_then(Value::as_str) == Some("number"))
        .map(|(name, _)| name)
        .collect();
    if texts.len() != 1 || numbers.is_empty() {
        return Some(rows);
    }
    let text = texts[0];
    let text_spec = properties.get(text)?;
    let text_schema = ar::schema(text_spec).ok()?;

    let mut joined: Vec<SourceRow> = Vec::with_capacity(rows.len());
    for row in rows {
        let can_merge = if let Some(previous) = joined.last() {
            let prev_text = previous
                .values
                .get(text)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty());
            let cur_text = row
                .values
                .get(text)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty());
            match (prev_text, cur_text) {
                (Some(pt), Some(ct)) => {
                    previous.spans.get(text).is_some_and(Option::is_some)
                        && row.spans.get(text).is_some_and(Option::is_some)
                        && numbers
                            .iter()
                            .all(|k| previous.values.get(*k).is_none_or(Value::is_null))
                        && numbers
                            .iter()
                            .any(|k| row.values.get(*k).is_some_and(|v| !v.is_null()))
                        && py_find_char(&row.line, ct) >= py_find_char(&previous.line, pt) + 2
                }
                _ => false,
            }
        } else {
            false
        };

        if can_merge && let Some(previous) = joined.last_mut() {
            let prev_text = previous
                .values
                .get(text)
                .and_then(Value::as_str)
                .unwrap_or("");
            let cur_text = row.values.get(text).and_then(Value::as_str).unwrap_or("");
            let combined = format!("{prev_text} {cur_text}");
            let combined_val = Value::String(combined);
            if !json_validate::is_valid(&text_schema, &combined_val) {
                return None;
            }
            let mut merged_values = row.values;
            merged_values.insert(text.clone(), combined_val);
            let mut merged_spans = row.spans;
            merged_spans.insert(text.clone(), None);
            *previous = SourceRow {
                line: format!("{}\n{}", previous.line, row.line),
                values: merged_values,
                spans: merged_spans,
                eligibility_only: row.eligibility_only,
                scope_instruction: row.scope_instruction,
            };
        } else {
            joined.push(row);
        }
    }
    Some(joined)
}

fn source_row(
    cells: &[String],
    keys: &[String],
    line: &str,
    offset: usize,
    properties: &Map<String, Value>,
    names: &HashMap<String, String>,
) -> Option<SourceRow> {
    if cells.len() != keys.len() {
        return None;
    }
    let mut values: Map<String, Value> = Map::new();
    let mut spans: BTreeMap<String, Option<[usize; 2]>> = BTreeMap::new();

    for (key, cell) in keys.iter().zip(cells.iter()) {
        let name = names.get(key)?;
        let spec = properties.get(name)?;
        let value = if cell.is_empty() {
            Value::Null
        } else if spec.get("type").and_then(Value::as_str) == Some("number") {
            printed_numbers::normalize(&Value::String(cell.clone()), spec).ok()?
        } else {
            Value::String(cell.clone())
        };
        let field_schema = ar::schema(spec).ok()?;
        if !json_validate::is_valid(&field_schema, &value) {
            return None;
        }
        values.insert(name.clone(), value);
        let span = if !cell.is_empty() && line.matches(cell.as_str()).count() == 1 {
            line.find(cell.as_str()).map(|byte_start| {
                let start = line[..byte_start].chars().count();
                let len = cell.chars().count();
                [offset + start, offset + start + len]
            })
        } else {
            None
        };
        spans.insert(name.clone(), span);
    }

    for (name, spec) in properties {
        if !values.contains_key(name) {
            let field_schema = ar::schema(spec).ok()?;
            if !json_validate::is_valid(&field_schema, &Value::Null) {
                return None;
            }
            values.insert(name.clone(), Value::Null);
            spans.insert(name.clone(), None);
        }
    }

    Some(SourceRow {
        line: line.trim_end_matches(['\r', '\n']).to_string(),
        values,
        spans,
        eligibility_only: false,
        scope_instruction: None,
    })
}

fn html_candidates(
    source: &str,
    properties: &Map<String, Value>,
    names: &HashMap<String, String>,
) -> Option<Vec<SourceRow>> {
    let tables = html_rows::tables(source)?;
    if tables.len() != 1 || tables[0].len() < 2 {
        return None;
    }
    let table = &tables[0];
    let header = &table[0];
    let body = &table[1..];
    let keys: Vec<String> = header.cells.iter().map(|c| normalize_column(c)).collect();
    let unique: HashSet<&str> = keys.iter().map(String::as_str).collect();
    if keys.len() < 2 || unique.len() != keys.len() || keys.iter().any(|k| !names.contains_key(k)) {
        return None;
    }

    let mut rows = Vec::with_capacity(body.len());
    for row in body {
        if row.cells.len() != keys.len() || row.cells.iter().all(String::is_empty) {
            return None;
        }
        let mut parsed = source_row(&row.cells, &keys, &row.line, row.offset, properties, names)?;
        if row.spans.len() != keys.len() {
            return None;
        }
        for (key, span) in keys.iter().zip(row.spans.iter()) {
            let name = names.get(key)?;
            parsed.spans.insert(name.clone(), *span);
        }
        rows.push(parsed);
    }
    Some(rows)
}

/// Extract candidate physical rows from a single unambiguous delimited, Markdown,
/// or HTML table in `source` matching `properties` (`table_rows.py::candidates`).
pub fn candidates(
    source: &str,
    properties: &Map<String, Value>,
    wrapped: bool,
) -> Option<Vec<SourceRow>> {
    for spec in properties.values() {
        let kind = spec.get("type").and_then(Value::as_str);
        if !matches!(kind, Some("text" | "number")) || is_truthy(spec.get("knowledge")) {
            return None;
        }
    }
    let mut names: HashMap<String, String> = HashMap::with_capacity(properties.len());
    for name in properties.keys() {
        names.insert(normalize_column(name), name.clone());
    }
    if names.len() != properties.len() {
        return None;
    }
    if html_table_regex().is_match(source) {
        return html_candidates(source, properties, &names);
    }

    let lines = py_splitlines_keepends(source);
    let mut offsets: Vec<usize> = Vec::with_capacity(lines.len());
    let mut offset = 0usize;
    for line in &lines {
        offsets.push(offset);
        offset += line.chars().count();
    }

    let mut tables: Vec<Option<Vec<SourceRow>>> = Vec::new();
    for delimiter in ['|', '\t', ',', ';'] {
        for (index, line) in lines.iter().enumerate() {
            if !line.contains(delimiter) {
                continue;
            }
            let Some(raw_header) = parse_csv_line(line, delimiter) else {
                continue;
            };
            let mut header: Vec<String> = raw_header
                .into_iter()
                .map(|c| c.trim().to_string())
                .collect();
            let leading = delimiter == '|'
                && header.len() > 2
                && header.first().is_some_and(String::is_empty);
            let trailing =
                delimiter == '|' && header.len() > 2 && header.last().is_some_and(String::is_empty);
            if trailing {
                header.pop();
            }
            if leading && !header.is_empty() {
                header.remove(0);
            }
            let keys = header_keys(&header, &names);
            let unique: HashSet<&str> = keys.iter().map(String::as_str).collect();
            if keys.len() < 2
                || unique.len() != keys.len()
                || keys.iter().any(|k| !names.contains_key(k))
            {
                continue;
            }

            let mut rows: Vec<SourceRow> = Vec::new();
            for row_index in (index + 1)..lines.len() {
                let text = lines[row_index];
                if !text.contains(delimiter) {
                    break;
                }
                let raw_cells = parse_csv_line(text, delimiter)?;
                let mut cells: Vec<String> = raw_cells
                    .into_iter()
                    .map(|c| c.trim().to_string())
                    .collect();
                if trailing && cells.last().is_some_and(String::is_empty) {
                    cells.pop();
                }
                if leading && cells.first().is_some_and(String::is_empty) {
                    cells.remove(0);
                }
                if cells.len() != keys.len() {
                    return None;
                }
                if rows.is_empty() && cells.iter().all(|c| md_separator_regex().is_match(c)) {
                    continue;
                }
                if cells.iter().all(String::is_empty) {
                    return None;
                }
                let row = source_row(&cells, &keys, text, offsets[row_index], properties, &names)?;
                rows.push(row);
            }
            tables.push(if wrapped {
                wrapped_rows(rows, properties)
            } else {
                Some(rows)
            });
        }
    }

    if tables.len() == 1 {
        tables.pop().flatten()
    } else {
        None
    }
}

fn scope_instruction(instructions: &str) -> Option<String> {
    let requested = match scope_split_regex().find(instructions) {
        Some(m) => &instructions[..m.start()],
        None => instructions,
    };
    let is_fee = role_fee_regex().is_match(requested);
    let is_tax = role_tax_regex().is_match(requested);
    let is_charge = role_charge_regex().is_match(requested);
    let role_count = usize::from(is_fee) + usize::from(is_tax) + usize::from(is_charge);
    if role_count != 1 {
        return None;
    }
    if is_fee {
        Some(format!(
            "{SCOPE_PRIORITY_INSTRUCTION}{SCOPE_FEE_INSTRUCTION}"
        ))
    } else if is_tax {
        Some(format!(
            "{SCOPE_PRIORITY_INSTRUCTION}{SCOPE_TAX_INSTRUCTION}"
        ))
    } else if is_charge {
        Some(format!(
            "{SCOPE_PRIORITY_INSTRUCTION}{SCOPE_CHARGE_INSTRUCTION}"
        ))
    } else {
        None
    }
}

/// Extract headerless 2-column delimited `(label, amount)` pairs for scoped
/// fee/tax/charge eligibility lists (`table_rows.py::scoped_pairs`).
pub fn scoped_pairs(source: &str, spec: &Value) -> Option<Vec<SourceRow>> {
    let instructions = spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let scope_inst = scope_instruction(instructions)?;
    let properties = spec
        .get("items")
        .and_then(|i| i.get("properties"))
        .and_then(Value::as_object)?;
    if properties.len() != 2
        || !only_rows_regex().is_match(instructions)
        || disallowed_scoped_op_regex().is_match(instructions)
    {
        return None;
    }

    let texts: Vec<&String> = properties
        .iter()
        .filter(|(_, child)| child.get("type").and_then(Value::as_str) == Some("text"))
        .map(|(name, _)| name)
        .collect();
    let numbers: Vec<&String> = properties
        .iter()
        .filter(|(_, child)| child.get("type").and_then(Value::as_str) == Some("number"))
        .map(|(name, _)| name)
        .collect();
    if texts.len() != 1
        || numbers.len() != 1
        || properties.values().any(|c| is_truthy(c.get("knowledge")))
    {
        return None;
    }
    let text_instructions = properties
        .get(texts[0])
        .and_then(|c| c.get("instructions"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if !exact_label_regex().is_match(text_instructions) {
        return None;
    }

    let mut names: HashMap<String, String> = HashMap::with_capacity(properties.len());
    for name in properties.keys() {
        names.insert(normalize_column(name), name.clone());
    }
    if names.len() != properties.len() {
        return None;
    }
    let keys = vec![normalize_column(texts[0]), normalize_column(numbers[0])];

    let lines = py_splitlines_keepends(source);
    let mut tables: Vec<Vec<SourceRow>> = Vec::new();
    for delimiter in ['|', '\t', ';'] {
        let mut rows: Vec<SourceRow> = Vec::new();
        let mut offset = 0usize;
        for line in &lines {
            if line.contains(delimiter) {
                let raw_cells = parse_csv_line(line, delimiter)?;
                let cells: Vec<String> = raw_cells
                    .into_iter()
                    .map(|c| c.trim().to_string())
                    .collect();
                if cells.len() != 2 || cells.iter().any(String::is_empty) {
                    return None;
                }
                let mut row = source_row(&cells, &keys, line, offset, properties, &names)?;
                row.eligibility_only = true;
                row.scope_instruction = Some(scope_inst.clone());
                rows.push(row);
            }
            offset += line.chars().count();
        }
        if rows.len() >= 2 {
            tables.push(rows);
        }
    }

    if tables.len() == 1 {
        tables.pop()
    } else {
        None
    }
}

/// Detect eligible table-backed object lists in `fields` and parse their physical
/// source rows (`table_rows.py::prepare`).
pub fn prepare(
    source: &str,
    fields: &Map<String, Value>,
    has_media: bool,
    allow_scoped_pairs: bool,
) -> BTreeMap<String, Vec<SourceRow>> {
    if has_media {
        return BTreeMap::new();
    }
    let mut result: BTreeMap<String, Vec<SourceRow>> = BTreeMap::new();
    for (name, spec) in fields {
        if spec.get("type").and_then(Value::as_str) != Some("list") {
            continue;
        }
        let Some(item) = spec.get("items").and_then(Value::as_object) else {
            continue;
        };
        if item.get("type").and_then(Value::as_str) != Some("object") {
            continue;
        }
        let Some(properties) = item.get("properties").and_then(Value::as_object) else {
            continue;
        };
        let instructions = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        let explicit_rows = explicit_rows_regex().is_match(instructions);
        let wrapped = join_wrapped_descriptions(instructions);
        let rows = if explicit_rows {
            candidates(source, properties, wrapped)
        } else if allow_scoped_pairs {
            scoped_pairs(source, spec)
        } else {
            None
        };
        let max_items = spec
            .get("max_items")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(MAX_LIST_ITEMS);
        if let Some(mut rows) = rows
            && !rows.is_empty()
            && rows.len() <= max_items
        {
            let has_missing_number_column = properties.iter().any(|(key, child)| {
                child.get("type").and_then(Value::as_str) == Some("number")
                    && rows
                        .iter()
                        .all(|row| row.values.get(key).is_none_or(Value::is_null))
            });
            if has_missing_number_column {
                for row in &mut rows {
                    row.eligibility_only = true;
                }
            }
            result.insert(name.clone(), rows);
        }
    }
    result
}

/// Strip standalone negative-join clauses (e.g., `"Do not join wrapped lines."`)
/// from list instructions when safe to do so (`table_rows.py::membership_instructions`).
pub fn membership_instructions(instructions: &str) -> String {
    if membership_guard_regex()
        .is_match(instructions)
        .unwrap_or(false)
    {
        return instructions.to_string();
    }
    let clauses = fancy_split(clause_split_regex(), instructions);
    let neg_re = negative_join_regex();
    if !clauses.iter().any(|clause| neg_re.is_match(clause)) {
        return instructions.to_string();
    }
    let filtered: String = clauses
        .into_iter()
        .filter(|clause| !neg_re.is_match(clause))
        .collect();
    filtered.trim().to_string()
}

/// Build the per-row AR field spec (either boolean eligibility gate or singleton
/// `{"enum": [value]}` object copy) for one candidate `row` (`table_rows.py::row_spec`).
pub fn row_spec(spec: &Value, row: &SourceRow, eligibility: bool) -> Result<Value> {
    let missing: Vec<&str> = row
        .values
        .iter()
        .filter(|(_, v)| v.is_null())
        .map(|(k, _)| k.as_str())
        .collect();
    let missing_instruction = if missing.is_empty() {
        String::new()
    } else {
        format!(
            "Missing cells or entire columns do not exclude an otherwise matching row. These source cells are missing and must remain null: {}. ",
            missing.join(", ")
        )
    };
    let scope = membership_instructions(
        spec.get("instructions")
            .and_then(Value::as_str)
            .unwrap_or(""),
    );
    if eligibility {
        let row_scope_inst = row.scope_instruction.as_deref().unwrap_or("");
        return Ok(json!({
            "type": "boolean",
            "nullable": false,
            "instructions": format!(
                "Classify exactly this physical source row for inclusion in the requested list. Return true if and only if this row satisfies the requested list instructions. Return false if the row is excluded, unrelated, or does not belong to the requested list, even though it exists. This is an eligibility decision, not a request to copy its cells. {missing_instruction}{row_scope_inst}Requested list instructions: {scope}\nExact physical source row: {}",
                row.line
            ),
        }));
    }

    let items_obj = spec
        .get("items")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_request("list spec missing items object"))?;
    let item_props = items_obj
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_request("list items spec missing properties"))?;

    let mut properties = Map::with_capacity(item_props.len());
    for (name, child) in item_props {
        let value = row.values.get(name).cloned().unwrap_or(Value::Null);
        let mut child_obj = child.as_object().cloned().unwrap_or_default();
        child_obj.insert("nullable".to_string(), Value::Bool(value.is_null()));
        child_obj.insert("enum".to_string(), Value::Array(vec![value]));
        properties.insert(name.clone(), Value::Object(child_obj));
    }

    let line_json = serde_json::to_string(&row.line).unwrap_or_else(|_| "\"\"".to_string());
    let mut out = items_obj.clone();
    out.insert("properties".to_string(), Value::Object(properties));
    out.insert("nullable".to_string(), Value::Bool(true));
    out.insert(
        "instructions".to_string(),
        Value::String(format!(
            "{scope}\nRead only this exact physical source row: {line_json}\nReturn null if this row does not satisfy the requested scope. Otherwise copy its cells together. {missing_instruction}Do not substitute a different row. Missing cells are null. This query has no list ordinal to count."
        )),
    );
    Ok(Value::Object(out))
}

/// Flag every field spec in `fields` with `"_physical_row_scope": true`
/// (matching `backend.py`'s `physical_row_view` tagging).
pub fn flag_physical_row_scope(fields: &mut Map<String, Value>) {
    for spec in fields.values_mut() {
        if let Some(obj) = spec.as_object_mut() {
            obj.insert(PHYSICAL_ROW_SCOPE_KEY.to_string(), Value::Bool(true));
        }
    }
}

/// Build local-scope synthetic input overrides and display names (`source_scope_N`)
/// when every row carries a `scope_instruction` (`table_rows.py::local_scope_context`).
pub fn local_scope_context(
    rows: &[SourceRow],
    queries: &[String],
    offset: usize,
) -> (BTreeMap<String, String>, BTreeMap<String, String>) {
    if rows.is_empty()
        || rows.len() != queries.len()
        || !rows.iter().all(|r| {
            r.scope_instruction
                .as_deref()
                .is_some_and(|s| !s.is_empty())
        })
    {
        return (BTreeMap::new(), BTreeMap::new());
    }
    let mut inputs = BTreeMap::new();
    let mut names = BTreeMap::new();
    for (idx, query) in queries.iter().enumerate() {
        inputs.insert(query.clone(), LOCAL_SCOPE_INPUT.to_string());
        names.insert(query.clone(), format!("source_scope_{}", offset + idx + 1));
    }
    (inputs, names)
}

/// Assemble per-row AR predictions back into a structured object-list prediction
/// (`table_rows.py::assemble`).
pub fn assemble(
    predictions: &Map<String, Value>,
    rows: &[SourceRow],
    queries: &[String],
    eligibility: bool,
) -> Result<Value> {
    if rows.len() != queries.len() {
        return Err(invalid_request(format!(
            "source_rows length ({}) does not match source_fields length ({})",
            rows.len(),
            queries.len()
        )));
    }
    let mut checks: Vec<&Value> = Vec::with_capacity(queries.len());
    for name in queries {
        let pred = predictions
            .get(name)
            .ok_or_else(|| invalid_request(format!("missing source row prediction {name:?}")))?;
        checks.push(pred);
    }

    let mut selected: Vec<(&SourceRow, &Value)> = Vec::new();
    for (row, &answer) in rows.iter().zip(checks.iter()) {
        let include = if eligibility {
            answer.get("value") == Some(&Value::Bool(true))
        } else {
            answer.get("value").is_some_and(|v| !v.is_null())
        };
        if include {
            selected.push((row, answer));
        }
    }

    let mut valid = checks.iter().all(|answer| {
        answer
            .get("valid")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && answer
                .get("complete")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    });

    let mut scored = true;
    let (p_null, p_null_method) = if eligibility {
        valid &= checks
            .iter()
            .all(|answer| answer.get("value").is_some_and(Value::is_boolean));
        scored = checks.iter().all(|answer| {
            answer
                .get("p_excluded")
                .and_then(Value::as_f64)
                .is_some_and(|score| score.is_finite() && (0.0..=1.0).contains(&score))
        });
        valid &= scored;
        let p_null = if scored
            && checks.iter().all(|answer| {
                answer
                    .get("p_excluded")
                    .and_then(Value::as_f64)
                    .is_some_and(|s| s > 0.0)
            }) {
            let log_sum: f64 = checks
                .iter()
                .map(|answer| {
                    answer
                        .get("p_excluded")
                        .and_then(Value::as_f64)
                        .unwrap_or(1.0)
                        .ln()
                })
                .sum();
            log_sum.exp()
        } else {
            0.0
        };
        let method = if scored {
            "product_source_row_exclusion_probabilities"
        } else {
            "unavailable_source_row_exclusion_probability"
        };
        (p_null, method)
    } else {
        let p_null = if checks.is_empty() {
            0.0
        } else {
            checks
                .iter()
                .map(|answer| answer.get("p_null").and_then(Value::as_f64).unwrap_or(0.0))
                .fold(f64::INFINITY, f64::min)
        };
        (p_null, "minimum_source_row_null_probability")
    };

    let lp_sum: f64 = checks
        .iter()
        .map(|answer| {
            answer
                .get("log_probability")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NEG_INFINITY)
        })
        .sum();
    let log_probability = lp_sum.is_finite().then_some(lp_sum);

    let confidence = match (valid, log_probability) {
        (true, Some(lp)) => lp.min(0.0).exp(),
        _ => 0.0,
    };

    let values: Vec<Value> = selected
        .iter()
        .map(|(row, answer)| {
            if eligibility {
                Value::Object(row.values.clone())
            } else {
                answer.get("value").cloned().unwrap_or(Value::Null)
            }
        })
        .collect();
    let confidences: Vec<Value> = selected
        .iter()
        .map(|(_, answer)| {
            json!(
                answer
                    .get("confidence")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
            )
        })
        .collect();
    let p_null_key = if eligibility { "p_excluded" } else { "p_null" };
    let p_nulls: Vec<Value> = selected
        .iter()
        .map(|(_, answer)| {
            json!(
                answer
                    .get(p_null_key)
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
            )
        })
        .collect();
    let spans: Vec<Value> = vec![Value::Null; selected.len()];
    let property_spans: Vec<Value> = selected.iter().map(|(row, _)| row.spans_value()).collect();

    let mut result = Map::new();
    result.insert("value".to_string(), Value::Array(values));
    result.insert("valid".to_string(), Value::Bool(valid));
    result.insert("complete".to_string(), Value::Bool(valid));
    result.insert("confidence".to_string(), json!(confidence));
    result.insert(
        "confidence_method".to_string(),
        Value::String(
            if eligibility {
                "ar_source_row_eligibility_probability"
            } else {
                "ar_source_row_sequence_probability"
            }
            .to_string(),
        ),
    );
    result.insert("p_null".to_string(), json!(p_null));
    result.insert(
        "p_null_method".to_string(),
        Value::String(p_null_method.to_string()),
    );
    result.insert("confidences".to_string(), Value::Array(confidences));
    if !(eligibility && !scored) {
        result.insert("p_nulls".to_string(), Value::Array(p_nulls));
    }
    result.insert("spans".to_string(), Value::Array(spans));
    result.insert("property_spans".to_string(), Value::Array(property_spans));
    if let Some(lp) = log_probability {
        result.insert("log_probability".to_string(), json!(lp));
    }
    if !valid {
        result.insert(
            "failure_reason".to_string(),
            Value::String(
                if eligibility && !scored {
                    "source_row_eligibility_unscored"
                } else {
                    "source_row_read_incomplete"
                }
                .to_string(),
            ),
        );
    }
    Ok(Value::Object(result))
}

/// Assemble every table-backed list annotated with `source_fields` in `record`
/// (matching `backend.py`'s post-prediction assembly loop).
pub fn assemble_record(
    record: &Value,
    predictions: &Map<String, Value>,
) -> Result<Map<String, Value>> {
    let mut assembled = Map::new();
    let Some(annotations) = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)
    else {
        return Ok(assembled);
    };
    for (name, annotation) in annotations {
        let Some(source_fields_arr) = annotation.get("source_fields").and_then(Value::as_array)
        else {
            continue;
        };
        if source_fields_arr.is_empty() {
            continue;
        }
        let queries: Vec<String> = source_fields_arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        let Some(rows_val) = annotation.get("source_rows") else {
            continue;
        };
        let rows: Vec<SourceRow> = serde_json::from_value(rows_val.clone())
            .map_err(|e| invalid_request(format!("invalid source_rows for {name:?}: {e}")))?;
        let eligibility = annotation
            .get("source_row_eligibility")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let merged = assemble(predictions, &rows, &queries, eligibility)?;
        assembled.insert(name.clone(), merged);
    }
    Ok(assembled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item_qty_price_props() -> Map<String, Value> {
        json!({
            "description": {"type": "text", "nullable": true},
            "quantity": {"type": "number", "integer": true, "nullable": true},
            "unit_price": {"type": "number", "decimals": 2, "nullable": true}
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn parses_csv_tsv_semicolon_and_markdown_tables_with_exact_spans() {
        // CSV with Qty -> quantity alias and quoted cell with comma + currency
        let csv = "Intro café\nDescription,Qty,Unit Price\n\"Bolt, M8\",4,$12.50\nWasher,,$0.25\n";
        let rows = candidates(csv, &item_qty_price_props(), false).unwrap();
        assert_eq!(rows.len(), 2);

        assert_eq!(rows[0].values["description"], json!("Bolt, M8"));
        assert_eq!(rows[0].values["quantity"], json!(4));
        assert_eq!(rows[0].values["unit_price"], json!(12.5));
        let chars: Vec<char> = csv.chars().collect();
        let d_span = rows[0].spans["description"].unwrap();
        let extracted: String = chars[d_span[0]..d_span[1]].iter().collect();
        assert_eq!(extracted, "Bolt, M8");
        let p_span = rows[0].spans["unit_price"].unwrap();
        let extracted_p: String = chars[p_span[0]..p_span[1]].iter().collect();
        assert_eq!(extracted_p, "$12.50");

        // Missing numeric cell -> null with None span
        assert_eq!(rows[1].values["description"], json!("Washer"));
        assert!(rows[1].values["quantity"].is_null());
        assert_eq!(rows[1].spans["quantity"], None);
        assert_eq!(rows[1].values["unit_price"], json!(0.25));

        // TSV
        let tsv = "description\tquantity\tunit_price\nWidget\t2\t9.99\n";
        let tsv_rows = candidates(tsv, &item_qty_price_props(), false).unwrap();
        assert_eq!(tsv_rows.len(), 1);
        assert_eq!(tsv_rows[0].values["quantity"], json!(2));

        // Semicolon
        let semi = "description;unit_price\nSensor;45.00\n";
        let semi_rows = candidates(semi, &item_qty_price_props(), false).unwrap();
        assert_eq!(semi_rows.len(), 1);
        assert!(semi_rows[0].values["quantity"].is_null());
        assert_eq!(semi_rows[0].spans["quantity"], None);

        // Markdown pipe table with outer pipes & alignment separator row
        let md = "| Description | Qty | Unit Price |\n| :--- | ---: | :---: |\n| Alpha | 3 | $10.00 |\n| Beta | 1 | ($2.50) |\n";
        let md_rows = candidates(md, &item_qty_price_props(), false).unwrap();
        assert_eq!(md_rows.len(), 2);
        assert_eq!(md_rows[0].values["description"], json!("Alpha"));
        assert_eq!(md_rows[0].values["quantity"], json!(3));
        assert_eq!(md_rows[0].values["unit_price"], json!(10.0));
        assert_eq!(md_rows[1].values["unit_price"], json!(-2.5));
    }

    #[test]
    fn parses_html_table_and_preserves_html_cell_spans() {
        let html = "Invoice café\n<table><tr><th>Description</th><th>Unit Price</th></tr><tr><td>Consulting</td><td>$150.00</td></tr><tr><td>AT&amp;T</td><td></td></tr></table>";
        let rows = candidates(html, &item_qty_price_props(), false).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values["description"], json!("Consulting"));
        assert_eq!(rows[0].values["unit_price"], json!(150.0));
        assert!(rows[0].values["quantity"].is_null());
        assert_eq!(rows[0].spans["quantity"], None);

        let chars: Vec<char> = html.chars().collect();
        let d_span = rows[0].spans["description"].unwrap();
        let extracted: String = chars[d_span[0]..d_span[1]].iter().collect();
        assert_eq!(extracted, "Consulting");

        // HTML entity cell decodes text and clears span
        assert_eq!(rows[1].values["description"], json!("AT&T"));
        assert_eq!(rows[1].spans["description"], None);
        assert!(rows[1].values["unit_price"].is_null());
    }

    #[test]
    fn joins_wrapped_descriptions_only_when_allowed_and_valid() {
        assert!(join_wrapped_descriptions(
            "Return one object per row. Combine wrapped descriptions."
        ));
        assert!(!join_wrapped_descriptions(
            "Return one object per row. Do not join wrapped descriptions."
        ));
        assert!(!join_wrapped_descriptions(
            "Keep multiline descriptions separate. Combine wrapped text."
        ));
        assert!(!join_wrapped_descriptions(
            "Don’t merge continuation lines."
        ));

        let props = json!({
            "description": {"type": "text", "nullable": true, "maxLength": 40},
            "amount": {"type": "number", "decimals": 2, "nullable": true}
        })
        .as_object()
        .unwrap()
        .clone();

        // Continuation line is indented >= 2 chars relative to previous line's description start
        let source = "| Description | Amount |\n| --- | --- |\n| Cloud hosting |  |\n|   monthly tier | $99.00 |\n";
        let raw_rows = candidates(source, &props, false).unwrap();
        assert_eq!(raw_rows.len(), 2);

        let joined = candidates(source, &props, true).unwrap();
        assert_eq!(joined.len(), 1);
        assert_eq!(
            joined[0].values["description"],
            json!("Cloud hosting monthly tier")
        );
        assert_eq!(joined[0].values["amount"], json!(99.0));
        assert_eq!(joined[0].spans["description"], None);
        assert!(joined[0].spans["amount"].is_some());
        assert!(joined[0].line.contains('\n'));

        // Fails schema validation when joined description exceeds maxLength -> None
        let strict_props = json!({
            "description": {"type": "text", "nullable": true, "maxLength": 15},
            "amount": {"type": "number", "decimals": 2, "nullable": true}
        })
        .as_object()
        .unwrap()
        .clone();
        assert!(candidates(source, &strict_props, true).is_none());
    }

    #[test]
    fn rejects_ambiguous_or_invalid_tables() {
        let props = item_qty_price_props();
        // Multiple tables -> rejected
        let two_tables = "Description,Qty\nA,1\n\nDescription,Qty\nB,2\n";
        assert!(candidates(two_tables, &props, false).is_none());

        // Non-nullable missing column -> rejected
        let required_props = json!({
            "description": {"type": "text", "nullable": false},
            "quantity": {"type": "number", "integer": true, "nullable": false}
        })
        .as_object()
        .unwrap()
        .clone();
        let missing_cell = "description,quantity\nWidget,\n";
        assert!(candidates(missing_cell, &required_props, false).is_none());

        // Unsupported property type (boolean) -> rejected
        let bool_props = json!({
            "description": {"type": "text"},
            "active": {"type": "boolean"}
        })
        .as_object()
        .unwrap()
        .clone();
        assert!(candidates("description,active\nA,true\n", &bool_props, false).is_none());

        // Knowledge property -> rejected
        let know_props = json!({
            "description": {"type": "text", "knowledge": true},
            "quantity": {"type": "number"}
        })
        .as_object()
        .unwrap()
        .clone();
        assert!(candidates("description,quantity\nA,1\n", &know_props, false).is_none());

        // Duplicate normalized column names -> rejected
        let dup_props = json!({
            "unit_price": {"type": "number"},
            "unit-price": {"type": "number"}
        })
        .as_object()
        .unwrap()
        .clone();
        assert!(candidates("unit_price,unit-price\n1,2\n", &dup_props, false).is_none());

        // Ragged body row -> rejected
        assert!(candidates("description,quantity\nA,1\nB,2,3\n", &props, false).is_none());

        // Invalid number -> rejected
        assert!(candidates("description,quantity\nA,not_a_num\n", &props, false).is_none());
    }

    #[test]
    fn scoped_pairs_and_local_scope_context() {
        let spec = json!({
            "type": "list",
            "instructions": "Only fee rows. Exclude taxes and product charges.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "text",
                        "instructions": "Exact printed row label."
                    },
                    "amount": {
                        "type": "number",
                        "decimals": 2,
                        "instructions": "Printed amount."
                    }
                }
            }
        });
        let source = "Consulting | $200.00\nLate fee | $15.00\nSales tax | $12.00\n";
        let rows = scoped_pairs(source, &spec).unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.eligibility_only));
        assert!(
            rows[0]
                .scope_instruction
                .as_deref()
                .unwrap()
                .contains("separately printed fee or surcharge")
        );

        let queries = vec![
            "fees.item1".to_string(),
            "fees.item2".to_string(),
            "fees.item3".to_string(),
        ];
        let (inputs, names) = local_scope_context(&rows, &queries, 2);
        assert_eq!(inputs["fees.item1"], LOCAL_SCOPE_INPUT);
        assert_eq!(names["fees.item1"], "source_scope_3");
        assert_eq!(names["fees.item3"], "source_scope_5");

        // Disallowed aggregation instruction -> rejected
        let bad_spec = json!({
            "type": "list",
            "instructions": "Only fee rows. Summarize by category.",
            "items": spec["items"]
        });
        assert!(scoped_pairs(source, &bad_spec).is_none());
    }

    #[test]
    fn membership_instructions_strips_negative_join_and_preserves_quoted() {
        assert_eq!(
            membership_instructions("Return one object per row. Do not join wrapped descriptions."),
            "Return one object per row."
        );
        assert_eq!(
            membership_instructions(
                "Keep all rows; never merge duplicate rows or combine continuation lines."
            ),
            "Keep all rows;"
        );
        // Guarded by quoted text or keyword -> unchanged
        let quoted = "Return one row per 'item'. Do not join wrapped lines.";
        assert_eq!(membership_instructions(quoted), quoted);
        let literal = "Keep verbatim rows. Do not join wrapped lines.";
        assert_eq!(membership_instructions(literal), literal);
    }

    #[test]
    fn prepare_row_spec_physical_scope_and_assemble_both_modes() {
        let fields_val = json!({
            "items": {
                "type": "list",
                "instructions": "Return one object per table row. Do not join wrapped lines.",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "text", "nullable": true},
                        "quantity": {"type": "number", "integer": true, "nullable": true},
                        "unit_price": {"type": "number", "decimals": 2, "nullable": true}
                    }
                }
            }
        });
        let fields = fields_val.as_object().unwrap();

        // Missing numeric column (unit_price omitted) forces eligibility_only = true
        let csv_missing_col = "Description,Qty\nBolt,4\nNut,10\n";
        let prepared = prepare(csv_missing_col, fields, false, false);
        assert_eq!(prepared.len(), 1);
        let rows = &prepared["items"];
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.eligibility_only));

        // Disabled when has_media == true
        assert!(prepare(csv_missing_col, fields, true, false).is_empty());

        // Singleton enum copy spec
        let copy_spec = row_spec(&fields["items"], &rows[0], false).unwrap();
        assert_eq!(copy_spec["nullable"], json!(true));
        assert_eq!(
            copy_spec["properties"]["description"]["enum"],
            json!(["Bolt"])
        );
        assert_eq!(
            copy_spec["properties"]["description"]["nullable"],
            json!(false)
        );
        assert_eq!(copy_spec["properties"]["quantity"]["enum"], json!([4]));
        assert_eq!(copy_spec["properties"]["unit_price"]["enum"], json!([null]));
        assert_eq!(
            copy_spec["properties"]["unit_price"]["nullable"],
            json!(true)
        );
        assert!(
            copy_spec["instructions"]
                .as_str()
                .unwrap()
                .contains("must remain null: unit_price")
        );
        // Verified valid against ar::schema
        let schema = ar::schema(&copy_spec).unwrap();
        assert!(json_validate::is_valid(
            &schema,
            &json!({"description": "Bolt", "quantity": 4, "unit_price": null})
        ));
        assert!(!json_validate::is_valid(
            &schema,
            &json!({"description": "Other", "quantity": 4, "unit_price": null})
        ));

        // Eligibility spec
        let elig_spec = row_spec(&fields["items"], &rows[0], true).unwrap();
        assert_eq!(elig_spec["type"], json!("boolean"));
        assert_eq!(elig_spec["nullable"], json!(false));
        assert!(
            elig_spec["instructions"]
                .as_str()
                .unwrap()
                .contains("Exact physical source row: Bolt,4")
        );

        // Flag _physical_row_scope
        let mut child_fields = Map::new();
        child_fields.insert("items.item1".to_string(), copy_spec);
        flag_physical_row_scope(&mut child_fields);
        assert_eq!(
            child_fields["items.item1"][PHYSICAL_ROW_SCOPE_KEY],
            json!(true)
        );

        // Assemble in non-eligibility (copy) mode
        let queries = vec!["items.item1".to_string(), "items.item2".to_string()];
        let copy_preds = json!({
            "items.item1": {
                "value": {"description": "Bolt", "quantity": 4, "unit_price": null},
                "valid": true,
                "complete": true,
                "confidence": 0.9,
                "log_probability": -0.1,
                "p_null": 0.05
            },
            "items.item2": {
                "value": null,
                "valid": true,
                "complete": true,
                "confidence": 0.8,
                "log_probability": -0.2,
                "p_null": 0.85
            }
        });
        let assembled_copy =
            assemble(copy_preds.as_object().unwrap(), rows, &queries, false).unwrap();
        assert_eq!(assembled_copy["valid"], json!(true));
        assert_eq!(assembled_copy["value"].as_array().unwrap().len(), 1);
        assert_eq!(assembled_copy["p_null"], json!(0.05));
        assert_eq!(
            assembled_copy["confidence_method"],
            json!("ar_source_row_sequence_probability")
        );
        assert_eq!(
            assembled_copy["p_null_method"],
            json!("minimum_source_row_null_probability")
        );
        assert_eq!(assembled_copy["spans"], json!([null]));
        assert_eq!(
            assembled_copy["property_spans"][0]["description"],
            json!([16, 20])
        );

        // Assemble in eligibility mode (scored)
        let elig_preds = json!({
            "items.item1": {
                "value": true,
                "valid": true,
                "complete": true,
                "confidence": 0.95,
                "log_probability": -0.05,
                "p_excluded": 0.1
            },
            "items.item2": {
                "value": false,
                "valid": true,
                "complete": true,
                "confidence": 0.8,
                "log_probability": -0.2,
                "p_excluded": 0.8
            }
        });
        let assembled_elig =
            assemble(elig_preds.as_object().unwrap(), rows, &queries, true).unwrap();
        assert_eq!(assembled_elig["valid"], json!(true));
        assert_eq!(assembled_elig["value"].as_array().unwrap().len(), 1);
        assert_eq!(assembled_elig["value"][0]["description"], json!("Bolt"));
        assert!((assembled_elig["p_null"].as_f64().unwrap() - 0.08).abs() < 1e-12);
        assert_eq!(assembled_elig["p_nulls"], json!([0.1]));
        assert_eq!(
            assembled_elig["confidence_method"],
            json!("ar_source_row_eligibility_probability")
        );

        // Assemble in eligibility mode when unscored -> valid=false, p_nulls omitted, failure_reason set
        let unscored_preds = json!({
            "items.item1": {
                "value": true,
                "valid": true,
                "complete": true,
                "confidence": 0.95,
                "log_probability": -0.05
            },
            "items.item2": {
                "value": false,
                "valid": true,
                "complete": true,
                "confidence": 0.8,
                "log_probability": -0.2
            }
        });
        let assembled_unscored =
            assemble(unscored_preds.as_object().unwrap(), rows, &queries, true).unwrap();
        assert_eq!(assembled_unscored["valid"], json!(false));
        assert!(assembled_unscored.get("p_nulls").is_none());
        assert_eq!(
            assembled_unscored["failure_reason"],
            json!("source_row_eligibility_unscored")
        );
        assert_eq!(
            assembled_unscored["p_null_method"],
            json!("unavailable_source_row_exclusion_probability")
        );

        // Test assemble_record helper with serialized SourceRows in _meta
        let record = json!({
            "_meta": {
                "extraction_contract": {
                    "fields": {
                        "items": {
                            "source_fields": queries,
                            "source_rows": rows,
                            "source_row_eligibility": true
                        }
                    }
                }
            }
        });
        let rec_out = assemble_record(&record, elig_preds.as_object().unwrap()).unwrap();
        assert_eq!(rec_out["items"], assembled_elig);
    }
}
